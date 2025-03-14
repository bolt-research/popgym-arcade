"""
This file to plot the partial and full curves for all algorithms in the same plot for all environments.
"""
import pandas as pd 
import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
import jax.numpy as jnp
from jax import lax 

def f(name):
    WINDOW_SIZE = 100
    SIGMA = 100
    INTERP_POINTS = 1000
    NORMALIZING_FACTOR = 200

    ENV_MAX_STEPS = {
        "CountRecallEasy": 2e7,
        "CountRecallMedium": 2e7,
        "CountRecallHard": 2e7,
        "BattleShipEasy": 2e7,
        "BattleShipMedium": 2e7,
        "BattleShipHard": 2e7,
        # other environments with default max steps 1e7
    }
    AXIS_FONT = {'fontsize': 9, 'labelpad': 8}
    TICK_FONT = {'labelsize': 8}

    api = wandb.Api()
    runs = api.runs("bolt-um/Arcade-RLC")
    filtered_runs = [run for run in runs if run.state == "finished"]
    print(f"Total runs: {len(runs)}, Completed runs: {len(filtered_runs)}")

    METRIC_MAPPING = {
        "PQN": {"return_col": "returned_episode_returns", "time_col": "env_step"},
        "PQN_RNN": {"return_col": "returned_episode_returns", "time_col": "env_step"},
        "default": {"return_col": "episodic return", "time_col": "TOTAL_TIMESTEPS"}
    }

    def process_run(run):
        """Process individual W&B run with dynamic max steps per environment"""
        try:
            config = {k: v for k, v in run.config.items() if not k.startswith('_')}
            env_name = config.get("ENV_NAME", "UnknownEnv")
            partial_status = str(config.get("PARTIAL", False))
            
            if env_name in ENV_MAX_STEPS:
                env_max_step = ENV_MAX_STEPS[env_name]
            else:
                env_max_step = 1e7
            
            alg_name = config.get("ALG_NAME", "").upper()
            memory_type = "MLP"
            if alg_name == "PQN_RNN":
                memory_type = config.get("MEMORY_TYPE", "Unknown").capitalize()

            metric_map = METRIC_MAPPING.get(alg_name, METRIC_MAPPING["default"])
            # history = run.history(keys=[metric_map["return_col"], metric_map["time_col"]])
            history = list(run.scan_history(keys=[metric_map["return_col"], metric_map["time_col"]]))
            history = pd.DataFrame(history, columns=[metric_map["return_col"], metric_map["time_col"]])
            
            history["true_steps"] = history[metric_map["time_col"]].clip(upper=env_max_step)
            history = history.sort_values(metric_map["time_col"]).drop_duplicates(subset=['true_steps'])

            if len(history) < 2:
                print(f"Skipping {run.name} due to insufficient data points")
                return None

            # Get first and last values for extrapolation
            first_return = history[metric_map["return_col"]].iloc[0]
            last_return = history[metric_map["return_col"]].iloc[-1]

            # Create unified interpolation grid for this environment
            unified_steps = np.linspace(0, env_max_step, INTERP_POINTS)
            unified_steps = np.round(unified_steps, decimals=5)
            scale_factor = NORMALIZING_FACTOR / env_max_step

            # Interpolate returns to uniform grid
            interp_func = interp1d(
                history['true_steps'], 
                history[metric_map["return_col"]],
                kind='linear',
                bounds_error=False,
                fill_value=(first_return, last_return)
            )
            interpolated_returns = interp_func(unified_steps)

            smoothed_returns = pd.Series(interpolated_returns).ewm(
                span=100,        
                adjust=False,    
                min_periods=1    
            ).mean().values

            # Compute cumulative maximum using JAX
            cummax_returns = lax.cummax(jnp.array(smoothed_returns))

            return pd.DataFrame({
                "Algorithm": f"{alg_name} ({memory_type})",
                "Return": interpolated_returns,
                "Smoothed Return": smoothed_returns,
                "Cummax Return": np.array(cummax_returns),  # Convert back to NumPy
                "True Steps": unified_steps,
                "EnvName": env_name,
                "Partial": partial_status,
                "Seed": str(config.get("SEED", 0)),
                "run_id": run.id,
                "StepsNormalized": unified_steps * scale_factor,
                "EnvMaxStep": env_max_step,
                "ScaleFactor": scale_factor
            })

        except Exception as e:
            print(f"Error processing {run.name}: {str(e)}")
        return None

    # Process all runs and combine data
    # all_data = [df for run in filtered_runs if (df := process_run(run)) is not None]
    
    # if not all_data:
    #     print("No valid data to process")
    #     exit()
    # runs_df = pd.concat(all_data, ignore_index=True)
    # runs_df.to_pickle("newdata.pkl")

    runs_df = pd.read_pickle("newdata.pkl")

    # Generate interpolation grid for each environment
    interpolated_data = []
    for (alg, env, partial), group in runs_df.groupby(['Algorithm', 'EnvName', 'Partial']):
        all_steps = group['True Steps'].unique()
        if len(all_steps) != INTERP_POINTS:
            print(f"Alignment error in {alg}-{env}: {len(all_steps)} vs {INTERP_POINTS}")
            continue

        pivot_df = group.pivot_table(
            index='True Steps',
            columns=['run_id', 'Partial'], 
            values='Smoothed Return',
            aggfunc='first'
        )

        # Calculate cumulative maximum for each run
        cummax_df = group.pivot_table(
            index='True Steps',
            columns=['run_id', 'Partial'], 
            values='Cummax Return',
            aggfunc='first'
        )

        # Compute mean and std for smoothed returns
        stats_df = pd.DataFrame({
            'Steps': pivot_df.index,
            'Mean': pivot_df.mean(axis=1),
            'Cummax Mean': cummax_df.mean(axis=1),  # Mean of cumulative max
            'Std': pivot_df.std(axis=1),
            'Count': pivot_df.count(axis=1)
        })

        # Compute lower and upper bounds for confidence interval
        stats_df['Lower'] = stats_df['Mean'] - stats_df['Std']
        stats_df['Upper'] = stats_df['Mean'] + stats_df['Std']

        # Apply cummax to lower and upper bounds
        stats_df['Lower'] = np.array(lax.cummax(jnp.array(stats_df['Lower'])))
        stats_df['Upper'] = np.array(lax.cummax(jnp.array(stats_df['Upper'])))

        # stats_df['StepsNormalized'] = stats_df['Steps'] * (NORMALIZING_FACTOR / group['EnvMaxStep'].iloc[0])
        stats_df['StepsNormalized'] = stats_df['Steps'] / group['EnvMaxStep'].iloc[0]

        interpolated_data.append(pd.DataFrame({
            'Algorithm': alg,
            'EnvName': env,
            'Partial': partial,  # Include 'Partial' column
            'Steps': stats_df['Steps'],
            'Smoothed': stats_df['Mean'],
            'Cummax': stats_df['Cummax Mean'],  # Add cumulative max
            'Lower': stats_df['Lower'],  # Cumulative max applied to lower bound
            'Upper': stats_df['Upper'],  # Cumulative max applied to upper bound
            'StepsNormalized': stats_df['StepsNormalized'],
            'EnvMaxStep': group['EnvMaxStep'].iloc[0]
        }))

    final_df = pd.concat(interpolated_data, ignore_index=True)
    df_battleship = final_df[final_df['EnvName'].str.contains("BattleShip")].copy()

    def plot_comparative_curves(df, name):
        """Plot comparative curves for all environments"""
        algorithms = df['Algorithm'].unique().tolist()
        palette = sns.color_palette("husl", n_colors=len(algorithms))  # Use a standard palette for algorithms
        
        # plt.style.use('seaborn-v0_8')
        
        envs = df[['EnvName', 'EnvMaxStep']].drop_duplicates()
        envs['BaseName'] = envs['EnvName'].apply(lambda x: x.replace("Easy", "").replace("Medium", "").replace("Hard", ""))
        envs['Difficulty'] = envs['EnvName'].apply(lambda x: 0 if "Easy" in x else 1 if "Medium" in x else 2 if "Hard" in x else 3)
        envs = envs.sort_values(by=['BaseName', 'Difficulty']).head(2)
        sns.set()         
        n_plots = len(envs)
        n_cols = min(3, n_plots)
        n_rows = int(np.ceil(n_plots / n_cols))
        fig, axes = plt.subplots(
            n_rows, n_cols, 
            figsize=(16, n_rows * 4.5), 
            sharey=True  
        )
        # fig = plt.figure(figsize=(12, n_rows*4.5))
        # gs = fig.add_gridspec(n_rows, n_cols, hspace=0.35, wspace=0.25,
                            # bottom=0.12, top=0.92)
        if n_rows * n_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        # axes[0].set_xticks([0.5, 1.0])
        # axes[1].set_xticks([0.0, 0.5, 1.0])
        for idx, (_, row) in enumerate(envs.iterrows()):
            # ax = fig.add_subplot(gs[idx//n_cols, idx%n_cols])
            ax = axes[idx]
            env, max_step = row[['EnvName', 'EnvMaxStep']]
            env_data = df[df.EnvName == env]
            # ax.set_xlim(0, 200)
            ax.set_xlim(0.0, 1.0)
            env_data_filtered = env_data[np.isfinite(env_data['Smoothed'])]
            
            y_min = env_data_filtered['Smoothed'].min()
            y_max = env_data_filtered['Smoothed'].max()
            
            # ax.set_ylim(y_min - 0.05*(y_max-y_min), 
            #         y_max + 0.05*(y_max-y_min))
            ax.set_ylim(0.0, 1.0)
            ax.set_xticks([0.0, 0.5, 1.0])
            ax.set_yticks([0.0, 0.5, 1.0])
            ax.tick_params(axis='x', pad=5)
            ax.tick_params(axis='y', pad=15)
            ax.text(1.07, -0.15,
                f"{max_step:.0e}".replace("+", "").replace("0", ""),
                transform=ax.transAxes,
                ha='right',  
                va='top',
                fontsize=35,
                color='#666666',
                bbox=dict(facecolor='white', alpha=0.8, 
                            edgecolor='none', pad=2))
            
            for alg_idx, alg in enumerate(algorithms):
                for partial_status in ['True', 'False']:  # Plot both Partial and Full statuses
                    alg_data = env_data[(env_data.Algorithm == alg) & (env_data.Partial == partial_status)]
                    if not alg_data.empty:
                        # Use distinct colors for partial and full
                        if partial_status == 'True':
                            color = palette[alg_idx]
                        else:
                            color = sns.desaturate(palette[alg_idx], 0.5)
                        
                        line_style = '--' if partial_status == 'True' else '-'  # Use dashed lines for Partial
                        ax.plot(
                            alg_data['StepsNormalized'], 
                            alg_data['Cummax'],
                            color=color, 
                            linewidth=2.5,
                            alpha=0.9,
                            label=f"{alg} ({'Partial' if partial_status == 'True' else 'Full'})",
                            linestyle=line_style,
                            solid_capstyle='round',
                            zorder=5
                        )
                        
                        ax.fill_between(
                            alg_data['StepsNormalized'],
                            alg_data['Lower'],
                            alg_data['Upper'],
                            color=color, 
                            alpha=0.2,
                            linewidth=0,
                            edgecolor=None,
                            zorder=2
                        )
            
            ax.set_xlabel("Env Steps", fontsize=35, labelpad=8)
            # ax.set_ylabel("Episodic Return", fontsize=20)
            ax.tick_params(labelsize=30)
            ax.set_title(f"{env}", 
                        fontsize=35, pad=12,
                        fontweight='semibold')
            
            ax.grid(True, alpha=0.8, linestyle='-', linewidth=0.8)
        fig.text(
            0.04, 0.5, "Episodic Return",
            va='center', rotation='vertical', fontsize=35
        )
        # handles, labels = ax.get_legend_handles_labels()
        # fig.legend(handles, labels,
        #         loc='upper center', 
        #         ncol=min(4, len(algorithms)),
        #         bbox_to_anchor=(0.5, 1.02),
        #         frameon=True,
        #         framealpha=0.95,
        #         edgecolor='#DDDDDD',
        #         title="Algorithm Types",
        #         title_fontsize=9,
        #         fontsize=8)
        
        plt.savefig("{}.pdf".format(name), 
                dpi=300, 
                bbox_inches='tight',
                facecolor='white')
        # plt.show()
        plt.close()
        
    # plot_comparative_curves(final_df, name)
    def plot_env_curves(df, name):
        """Plot separate curves per environment and save a separate legend PDF."""
        algorithms = df['Algorithm'].unique().tolist()
        palette = sns.color_palette("husl", n_colors=len(algorithms))
        sns.set()
        # Group environments (sorting by base name and difficulty)
        envs = df[['EnvName', 'EnvMaxStep']].drop_duplicates()
        envs['BaseName'] = envs['EnvName'].apply(lambda x: x.replace("Easy", "").replace("Medium", "").replace("Hard", ""))
        envs['Difficulty'] = envs['EnvName'].apply(lambda x: 0 if "Easy" in x else 1 if "Medium" in x else 2 if "Hard" in x else 3)
        envs = envs.sort_values(by=['BaseName', 'Difficulty'])
        
        # Prepare storage for legend handles
        legend_handles = []
        legend_labels = []
        
        # Loop over each environment and create a separate figure/pdf
        for idx, row in envs.iterrows():
            env = row['EnvName']
            max_step = row['EnvMaxStep']
            fig, ax = plt.subplots(figsize=(6,4))
            # ax.set_xlim(0, 200)
            ax.set_xlim(0, 1)
            
            env_data = df[df.EnvName == env]
            env_data_filtered = env_data[np.isfinite(env_data['Smoothed'])]
            y_min = env_data_filtered['Smoothed'].min()
            y_max = env_data_filtered['Smoothed'].max()
            ax.set_ylim(y_min - 0.1*(y_max-y_min), y_max + 0.1*(y_max-y_min))
            # ax.set_ylim(0.0, 1.0)
            ax.text(1.05, -0.15,
                    f"{max_step:.0e}".replace("+", "").replace("0", ""),
                    transform=ax.transAxes,
                    ha='right',  
                    va='top',
                    fontsize=20,
                    color='#666666',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))
            
            # Plot curves per algorithm and partial status
            for alg_idx, alg in enumerate(algorithms):
                for partial_status in ['True', 'False']:
                    alg_data = env_data[(env_data.Algorithm == alg) & (env_data.Partial == partial_status)]
                    if not alg_data.empty:
                        # Choose line color and style based on partial status
                        color = palette[alg_idx] if partial_status == 'True' else sns.desaturate(palette[alg_idx], 0.5)
                        line_style = '--' if partial_status == 'True' else '-'
                        
                        line, = ax.plot(
                            alg_data['StepsNormalized'], 
                            alg_data['Cummax'], 
                            color=color, 
                            linewidth=2.5, 
                            alpha=0.9,
                            linestyle=line_style,
                            solid_capstyle='round',
                            zorder=5
                        )
                        
                        label = f"{alg} ({'Partial' if partial_status=='True' else 'Full'})"
                        # Add handle & label once (global legend)
                        if label not in legend_labels:
                            legend_handles.append(line)
                            legend_labels.append(label)
                        
                        ax.fill_between(
                            alg_data['StepsNormalized'],
                            alg_data['Lower'],
                            alg_data['Upper'],
                            color=color, 
                            alpha=0.2,
                            linewidth=0,
                            edgecolor=None,
                            zorder=2
                        )
            
            ax.set_xlabel("Env Steps", fontsize=20, labelpad=8)
            ax.set_ylabel("Episodic Return", fontsize=20, labelpad=8)
            ax.set_title(env, fontsize=20, pad=12, fontweight='semibold')
            ax.tick_params(labelsize=20)
            ax.grid(True, alpha=0.8, linestyle='-', linewidth=0.8)
            
            # Save individual environment figure as its own pdf
            fig.savefig(f"{env}_{name}_curve.pdf", dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)
        
        # Create separate figure just for the legend and save it
        # fig_legend = plt.figure(figsize=(8, 1))
        # legend = fig_legend.legend(
        #     legend_handles, legend_labels,
        #     loc='center', 
        #     ncol=min(4, len(legend_handles)),
        #     frameon=True,
        #     framealpha=0.95,
        #     edgecolor='#ffffff',
        #     title="",
        #     title_fontsize=9,
        #     fontsize=20
        # )
        # # Remove axes for legend figure
        # fig_legend.canvas.draw()
        # fig_legend.savefig(f"legend.pdf", dpi=300, bbox_inches='tight', facecolor='white')
        # plt.close(fig_legend)

    # plot_env_curves(final_df, name)
    plot_comparative_curves(df_battleship, name)

for i in range(1):
    f(f"plot_PartialCompare{i}")
    print(f"plot_PartialCompare{i} done")

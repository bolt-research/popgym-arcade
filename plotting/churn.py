"""
Plotting churn ratio difference between partial and full observability
"""
import pandas as pd 
import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
import jax.numpy as jnp
from jax import lax 
import numpy as np

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
    runs = api.runs("bolt-um/Arcade-RLC-Churn")
    filtered_runs = [run for run in runs if run.state == "finished"]
    print(f"Total runs: {len(runs)}, Completed runs: {len(filtered_runs)}")

    METRIC_MAPPING = {
        "PQN": {"churn_ratio": "churn_ratio", "time_col": "env_step"},
        "PQN_RNN": {"churn_ratio": "churn_ratio", "time_col": "env_step"},
        "default": {"churn_ratio": "churn_ratio", "time_col": "TOTAL_TIMESTEPS"}
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
            history = list(run.scan_history(keys=[metric_map["churn_ratio"], metric_map["time_col"]]))
            history = pd.DataFrame(history, columns=[metric_map["churn_ratio"], metric_map["time_col"]])
            
            history["true_steps"] = history[metric_map["time_col"]].clip(upper=env_max_step)
            history = history.sort_values(metric_map["time_col"]).drop_duplicates(subset=['true_steps'])

            if len(history) < 2:
                print(f"Skipping {run.name} due to insufficient data points")
                return None

            # Get first and last values for extrapolation
            first_return = history[metric_map["churn_ratio"]].iloc[0]
            last_return = history[metric_map["churn_ratio"]].iloc[-1]

            # Create unified interpolation grid for this environment
            unified_steps = np.linspace(0, env_max_step, INTERP_POINTS)
            unified_steps = np.round(unified_steps, decimals=5)
            scale_factor = NORMALIZING_FACTOR / env_max_step

            # Interpolate returns to uniform grid
            interp_func = interp1d(
                history['true_steps'], 
                history[metric_map["churn_ratio"]],
                kind='linear',
                bounds_error=False,
                fill_value=(first_return, last_return)
            )
            interpolated_churn_ratio = interp_func(unified_steps)

            return pd.DataFrame({
                "Algorithm": f"{alg_name} ({memory_type})",
                "churn_ratio": interpolated_churn_ratio,
                # "Smoothed Return": smoothed_returns,
                # "Cummax Return": np.array(cummax_returns),  # Convert back to NumPy
                "True Steps": unified_steps,
                "EnvName": env_name,
                "Partial": partial_status,
                "Seed": str(config.get("SEED", 0)),
                "run_id": run.id,
                "StepsNormalized": unified_steps / env_max_step,
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
    # runs_df.to_pickle("churnratiodata.pkl")

    runs_df = pd.read_pickle("churnratiodata.pkl")
    # print(f"Total runs processed: {runs_df}")

    diff_df = pd.DataFrame()

    for env_name in runs_df['EnvName'].unique():
        env_data = runs_df[runs_df['EnvName'] == env_name]

        partial_true = env_data[env_data['Partial'] == 'True']
        partial_false = env_data[env_data['Partial'] == 'False']
    
        merged = pd.merge(
            partial_true[['StepsNormalized', 'churn_ratio']],
            partial_false[['StepsNormalized', 'churn_ratio']],
            on='StepsNormalized',
            suffixes=('_true', '_false'),
            how='inner'
        )
    
        merged['churn_diff'] = np.abs(merged['churn_ratio_true'] - merged['churn_ratio_false'])
        merged['EnvName'] = env_name.replace('Easy', '')

        # diff_df = pd.concat([diff_df, merged[['EnvName', 'StepsNormalized', 'churn_diff']]], ignore_index=True)
        merged['churn_diff_cummax'] = merged.groupby('EnvName')['churn_diff'].cummax()
        # merged['churn_diff_avg'] = merged.groupby('EnvName')['churn_diff'].transform('mean')
        merged['churn_diff_avg'] = merged.groupby('EnvName')['churn_diff'].transform(
            lambda x: x.rolling(window=20, min_periods=1).mean()
        )
  
        diff_df = pd.concat([diff_df, merged[['EnvName', 'StepsNormalized', 'churn_diff', 'churn_diff_cummax', 'churn_diff_avg']]], 
                       ignore_index=True)


    plt.figure(figsize=(12, 7))
    sns.set()
    sns.lineplot(
        data=diff_df,
        x='StepsNormalized',
        y='churn_diff_avg',
        hue='EnvName',
        palette='Spectral',
        linewidth=2.5
    )

    plt.title('Relative Policy Churn', fontsize=35)
    plt.xlabel('Training Progress', fontsize=35)
    plt.ylabel('POMDP/MDP Difference', fontsize=35)
    plt.tick_params(axis='both', which='major', labelsize=35)
    plt.legend(title='', loc='upper left', fontsize=20, ncol=2)
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig("{}.pdf".format(name), 
            dpi=300, 
            bbox_inches='tight',
            facecolor='white')


for i in range(1):
    f(f"churn{i}")
    print(f"churn{i} done")
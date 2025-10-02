"""
This file is to plot the MDP and POMDP results separately.
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
        "MineSweeperEasy": 2e7,
        "MineSweeperMedium": 2e7,
        "MineSweeperHard": 2e7,
        "NavigatorEasy": 2e7,
        "NavigatorMedium": 2e7,
        "NavigatorHard": 2e7,
        # other environments with default max steps 1e7
    }
    AXIS_FONT = {'fontsize': 9, 'labelpad': 8}
    TICK_FONT = {'labelsize': 8}

    api = wandb.Api()
    runs = api.runs("bolt-um/Arcade-NIPS")
    filtered_runs = [run for run in runs if run.state == "finished"]
    print(f"Total runs: {len(runs)}, Completed runs: {len(filtered_runs)}")

    METRIC_MAPPING = {
        "PQN": {"return_col": "returned_episode_returns", "time_col": "env_step"},
        "PQN_RNN": {"return_col": "returned_episode_returns", "time_col": "env_step"},
        "default": {"return_col": "episodic return", "time_col": "TOTAL_TIMESTEPS"}
        # "PPO": {"return_col": "episodic return", "time_col": "global step"},
        # "PPO_RNN": {"return_col": "episodic return", "time_col": "global step"},
        # "default": {"return_col": "episodic return", "time_col": "TOTAL_TIMESTEPS"}
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
            
            # alg_name = config.get("ALG_NAME", "").upper()
            # For PQN
            alg_name = config.get("ALG_NAME", "").upper()
            
            # For PPO
            # alg_name = config.get("TRAIN_TYPE", "").upper()
            
            memory_type = "MLP"
            if alg_name == "PQN_RNN":
                memory_type = config.get("MEMORY_TYPE", "Unknown").capitalize()

            metric_map = METRIC_MAPPING.get(alg_name, METRIC_MAPPING["default"])
            # history = run.scan_history(keys=[metric_map["return_col"], metric_map["time_col"]])
            history = list(run.scan_history(keys=[metric_map["return_col"], metric_map["time_col"]]))
            history = pd.DataFrame(history, columns=[metric_map["return_col"], metric_map["time_col"]])
            
            history["true_steps"] = history[metric_map["time_col"]].clip(upper=env_max_step)
            history = history.sort_values(metric_map["time_col"]).drop_duplicates(subset=['true_steps'])
            # print(f"{run.name} with {(history)} data points")
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
            # smoothed_returns = pd.Series(interpolated_returns).rolling(window=WINDOW_SIZE, min_periods=1).mean().values

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
    # # # save the data
    # runs_df.to_csv("pqn_gru.csv")

    # load the data
    runs_df = pd.read_csv("PQN_20250912.csv")
    # runs_df = pd.read_csv("pqn128datatimestep.csv")
    runs_df['FinalReturn'] = runs_df['Cummax Return'].astype(float)

    normal_dict = [
                   "BattleShipEasy",
                   "BattleShipMedium",
                   "BattleShipHard",
                   "MineSweeperEasy",
                   "MineSweeperMedium",
                    "MineSweeperHard",
                    # "NavigatorEasy",
                    # "NavigatorMedium",
                    # "NavigatorHard",
                    "BreakoutEasy",
                    "BreakoutMedium",
                    "BreakoutHard",
                    "TetrisEasy",
                    "TetrisMedium",
                    "TetrisHard",
                   ]
    # Normalize the FinalReturn for each environment
    for env in normal_dict:
        mask = runs_df['EnvName'] == env
        if mask.any():
            # Special case for Breakout environments
            if env in ["BreakoutEasy", "BreakoutMedium", "BreakoutHard"]:
                env_min = -1.0
                env_max = 0.6
            else:
                env_min = -1.0
                env_max = 1.0
            runs_df.loc[mask, 'FinalReturn'] = (
                runs_df.loc[mask, 'FinalReturn'] - env_min
            ) / (env_max - env_min)

    # store the data
    # runs_df.to_csv("pqn128datatimestep.csv")
    # runs_df.to_csv("pqn_gru.csv", index=False)
    # First aggregate across seeds within each environment for each model and Partial status:
    # For each (Algorithm, Partial, EnvName) group, take the maximum final return across seeds.
    seedgroup = runs_df.groupby(['Algorithm', 'Partial', 'EnvName', 'run_id', 'Seed'])['FinalReturn'].max().reset_index()
    
    # seedgroup.to_csv("pqn128seedgroup.csv")
    # seedgroup.to_csv("pqn_gru_seedgroup.csv", index=False)

    
    overseedgroup = seedgroup.groupby(['Algorithm', 'Partial', 'EnvName']).agg(
        mean=('FinalReturn', 'mean'),
        std=('FinalReturn', 'std'),
        median=('FinalReturn', 'median'),
        q25 = ( 'FinalReturn', lambda x: x.quantile(0.25) ),
        q75 = ( 'FinalReturn', lambda x: x.quantile(0.75) ),
        count=('FinalReturn', 'count')
    ).reset_index()

    overseedgroup['ci_lower'] = overseedgroup['mean'] - 1.96 * overseedgroup['std'] / np.sqrt(overseedgroup['count'])
    overseedgroup['ci_upper'] = overseedgroup['mean'] + 1.96 * overseedgroup['std'] / np.sqrt(overseedgroup['count'])

    overseedgroup.to_csv("finalpqn20250912env_group_with_ci.csv", index=False)
    # overseedgroup.to_csv("pqn_gru_env_group_with_ci.csv", index=False)

    env_group = overseedgroup.groupby(['Algorithm', 'Partial']).agg(
        mean=('mean', 'mean'),
        std=('std', 'mean'),
        median=('median', 'mean'),
        q25 = ( 'q25', 'mean' ),
        q75 = ( 'q75', 'mean' ),
        count=('count', 'sum')
    ).reset_index()

    # env_group.to_csv("finalpqn128model_group.csv", index=False)
    # env_group.to_csv("pqn_gru_model_group.csv", index=False)
    env_group.to_csv("finalpqn20250912model_group.csv", index=False)

    # env_group = seedgroup.groupby(['Algorithm', 'Partial', 'EnvName'])['FinalReturn'].agg(['mean', 'std']).reset_index()
    # # max for each seed then aggregate

    # env_group.to_csv("pqn128env_group.csv")

    # Now aggregate across the environments and difficults: compute mean and std for each (Algorithm, Partial)
    # model_group = env_group.groupby(['Algorithm', 'Partial']).agg(
    #     mean=('mean', 'mean'),
    #     std=('std', 'mean')
    # ).reset_index()
    # model_group.to_csv("pqn128model_group.csv")

    # # Pivot the table so that rows = Model and columns for Partial outcomes.
    # # This will produce MultiIndex columns; then we rename them.
    # pivot = {}
    # for algo, group in runs_df.groupby('Algorithm'):
    #     table = group.pivot(index='EnvName', columns='Partial', values=['mean', 'std'])
    #     pivot[algo] = table
    #     # Rename columns so that "False" becomes "MDP" and "True" becomes "POMDP".
    #     table.columns = table.columns.map(
    #         lambda x: "MDP" if (x[0] == "mean" and str(x[1]) == "False") else
    #                 ("POMDP" if (x[0] == "mean" and str(x[1]) == "True") else
    #                 ("MDP_std" if (x[0] == "std" and str(x[1]) == "False") else
    #                     ("POMDP_std" if (x[0] == "std" and str(x[1]) == "True") else f"{x[0]}_{x[1]}")))
    #     )

    #     # Compute the overall performance (MDP+POMDP) for the mean as the average of the two
    #     table['MDP+POMDP'] = table[['MDP', 'POMDP']].mean(axis=1)
        
    #     # Optionally, compute a combined variance (average the variances, here approximated via std)
    #     table['MDP+POMDP_std'] = table[['MDP_std', 'POMDP_std']].mean(axis=1)
    # for algo, table in pivot.items():
    #     print(f"\n{algo}")
    #     print(table)
    # # Print or save the table
    # # print(table)
    #     table.to_csv(f"{algo}.csv", index=True)

for i in range(1):
    f(f"plot_{i}")
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
    runs = api.runs("bolt-um/Arcade-GDN")
    
    filtered_runs = [
        run
        for run in runs
            if (
                run.state == "finished"
            )
    ]
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
            memory_type = "MLP"
            if alg_name == "PQN_RNN":
                memory_type = config.get("MEMORY_TYPE", "Unknown").capitalize()

            # For PPO
            # alg_name = config.get("TRAIN_TYPE", "").upper()
            # memory_type = "MLP"
            # if alg_name == "PPO_RNN":
                # memory_type = config.get("MEMORY_TYPE", "Unknown").capitalize()

            metric_map = METRIC_MAPPING.get(alg_name, METRIC_MAPPING["default"])
            history = list(run.scan_history(keys=[metric_map["return_col"], metric_map["time_col"]]))
            history = pd.DataFrame(history, columns=[metric_map["return_col"], metric_map["time_col"]])
            
            history["true_steps"] = history[metric_map["time_col"]].clip(upper=env_max_step)
            history = history.sort_values(metric_map["time_col"]).drop_duplicates(subset=['true_steps'])
            if len(history) < 2:
                print(f"Skipping {run.name} due to insufficient data points")
                return None

            first_return = history[metric_map["return_col"]].iloc[0]
            last_return = history[metric_map["return_col"]].iloc[-1]

            unified_steps = np.linspace(0, env_max_step, INTERP_POINTS)
            unified_steps = np.round(unified_steps, decimals=5)
            scale_factor = NORMALIZING_FACTOR / env_max_step

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

            cummax_returns = lax.cummax(jnp.array(smoothed_returns))

            return pd.DataFrame({
                "Algorithm": f"{alg_name} ({memory_type})",
                "Return": interpolated_returns,
                "Smoothed Return": smoothed_returns,
                "Cummax Return": np.array(cummax_returns),
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
    all_data = [df for run in filtered_runs if (df := process_run(run)) is not None]
    if not all_data:
        print("No valid data to process")
        exit()
    runs_df = pd.concat(all_data, ignore_index=True)

    runs_df.to_csv("YOUR.csv", index=False)


    runs_df = pd.read_csv("YOUR.csv")

    runs_df['FinalReturn'] = runs_df['Cummax Return'].astype(float)

    normal_dict = [
                   "BattleShipEasy",
                   "BattleShipMedium",
                   "BattleShipHard",
                   "MineSweeperEasy",
                   "MineSweeperMedium",
                    "MineSweeperHard",
                    "BreakoutEasy",
                    "BreakoutMedium",
                    "BreakoutHard",
                    "TetrisEasy",
                    "TetrisMedium",
                    "TetrisHard",
                   ]

    for env in normal_dict:
        mask = runs_df['EnvName'] == env
        if mask.any():
            if env in ["BreakoutEasy", "BreakoutMedium", "BreakoutHard"]:
                env_min = -1.0
                env_max = 0.6
            else:
                env_min = -1.0
                env_max = 1.0
            runs_df.loc[mask, 'FinalReturn'] = (
                runs_df.loc[mask, 'FinalReturn'] - env_min
            ) / (env_max - env_min)


    seedgroup = runs_df.groupby(['Algorithm', 'Partial', 'EnvName', 'run_id', 'Seed'])['FinalReturn'].max().reset_index()
    
    overseedgroup = seedgroup.groupby(['Algorithm', 'Partial', 'EnvName']).agg(
        mean=('FinalReturn', 'mean'),
        std=('FinalReturn', lambda x: x.std(ddof=1) if len(x) > 1 else 0.0),
        median=('FinalReturn', 'median'),
        q25 = ( 'FinalReturn', lambda x: x.quantile(0.25) ),
        q75 = ( 'FinalReturn', lambda x: x.quantile(0.75) ),
        count=('FinalReturn', 'count')
    ).reset_index()

    overseedgroup['ci_lower'] = overseedgroup['mean'] - 1.96 * overseedgroup['std'] / np.sqrt(overseedgroup['count'])
    overseedgroup['ci_upper'] = overseedgroup['mean'] + 1.96 * overseedgroup['std'] / np.sqrt(overseedgroup['count'])

    # Per-environment table (one row per EnvName × Algorithm × Partial)
    overseedgroup.to_csv("per_env_returns.csv", index=False)

    env_group = overseedgroup.groupby(['Algorithm', 'Partial']).agg(
        mean=('mean', 'mean'),
        std=('std', 'mean'),
        median=('median', 'mean'),
        q25 = ( 'q25', 'mean' ),
        q75 = ( 'q75', 'mean' ),
        count=('count', 'sum')
    ).reset_index()


    env_group.to_csv("pqn_gdn_model_group.csv", index=False) # save your data here, e.g. "pqn_gru_model_group.csv"

    model_group = env_group.groupby(['Algorithm', 'Partial']).agg(
        mean=('mean', 'mean'),
        std=('std', 'mean')
    ).reset_index()
    model_group.to_csv("your_output_csv.csv") # data for obs gap and memory bias.

    # Print a per-environment table like the paper (MDP Return vs POMDP Return)
    table_data = overseedgroup.copy()
    table_data['value_str'] = table_data.apply(
        lambda r: f"{r['mean']:.2f} ± {r['std']:.2f}", axis=1
    )
    # Pivot: rows = EnvName, columns = Partial (False=MDP, True=POMDP)
    pivot = table_data.pivot_table(
        index='EnvName',
        columns='Partial',
        values='value_str',
        aggfunc='first'
    )
    pivot.columns = ['MDP Return' if str(c) == 'False' else 'POMDP Return' for c in pivot.columns]
    pivot = pivot.reset_index().rename(columns={'EnvName': 'Environment'})
    print("\nPer-environment returns table:")
    print(pivot.to_string(index=False))
    pivot.to_csv("per_env_table.csv", index=False)
    print("\nSaved to per_env_table.csv")


for i in range(1):
    f(f"plot_{i}")
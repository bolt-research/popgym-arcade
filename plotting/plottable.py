"""
This file is to plot the MDP and POMDP results separately.

Usage example:
    python plottable.py --entity Your_entity --project Your_project --raw-csv runs.csv \
        --model-group-csv pqn_gdn_model_group.csv --output-csv obs_gap.csv
"""

import argparse
import pandas as pd
import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
import jax.numpy as jnp
from jax import lax


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process W&B runs and compute MDP/POMDP return statistics."
    )

    # --- W&B connection ---
    parser.add_argument("--entity", type=str, default="bolt-um",
                        help="W&B entity (username or team). Default: bolt-um")
    parser.add_argument("--project", type=str, default="Arcade-GDN",
                        help="W&B project name. Default: Arcade-GDN")

    # --- Processing hyperparameters ---
    parser.add_argument("--interp-points", type=int, default=1000,
                        help="Number of interpolation points per run. Default: 1000")
    parser.add_argument("--normalizing-factor", type=float, default=200,
                        help="Step-axis normalizing factor. Default: 200")
    parser.add_argument("--ewm-span", type=int, default=100,
                        help="EWM smoothing span. Default: 100")

    # --- Output file names ---
    parser.add_argument("--raw-csv", type=str, default="runs_raw.csv",
                        help="Path for the raw per-step CSV fetched from W&B. Default: runs_raw.csv")
    parser.add_argument("--per-env-returns-csv", type=str, default="per_env_returns.csv",
                        help="Path for per-environment aggregated returns CSV. Default: per_env_returns.csv")
    parser.add_argument("--model-group-csv", type=str, default="model_group.csv",
                        help="Path for the per-algorithm model-group CSV. Default: model_group.csv")
    parser.add_argument("--output-csv", type=str, default="output.csv",
                        help="Path for the final obs-gap / memory-bias CSV. Default: output.csv")
    parser.add_argument("--per-env-table-csv", type=str, default="per_env_table.csv",
                        help="Path for the printable per-environment table CSV. Default: per_env_table.csv")

    # --- Control flow ---
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip W&B fetch and load --raw-csv directly (for re-running analysis).")

    return parser.parse_args()


def fetch_and_process(args):
    """Fetch runs from W&B, process them, and save to args.raw_csv."""

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
        # other environments default to 1e7
    }

    METRIC_MAPPING = {
        "PQN":     {"return_col": "returned_episode_returns", "time_col": "env_step"},
        "PQN_RNN": {"return_col": "returned_episode_returns", "time_col": "env_step"},
        "default": {"return_col": "episodic return",          "time_col": "TOTAL_TIMESTEPS"},
        # "PPO":     {"return_col": "episodic return", "time_col": "global step"},
        # "PPO_RNN": {"return_col": "episodic return", "time_col": "global step"},
    }

    api = wandb.Api()
    runs = api.runs(f"{args.entity}/{args.project}")
    filtered_runs = [r for r in runs if r.state == "finished"]
    print(f"Total runs: {len(runs)}, Completed runs: {len(filtered_runs)}")

    def process_run(run):
        """Process one W&B run; returns a DataFrame or None."""
        try:
            config = {k: v for k, v in run.config.items() if not k.startswith('_')}
            env_name = config.get("ENV_NAME", "UnknownEnv")
            partial_status = str(config.get("PARTIAL", False))

            env_max_step = ENV_MAX_STEPS.get(env_name, 1e7)

            # For PQN
            alg_name = config.get("ALG_NAME", "").upper()
            memory_type = "MLP"
            if alg_name == "PQN_RNN":
                memory_type = config.get("MEMORY_TYPE", "Unknown").capitalize()

            # For PPO (uncomment and swap when needed)
            # alg_name = config.get("TRAIN_TYPE", "").upper()
            # memory_type = "MLP"
            # if alg_name == "PPO_RNN":
            #     memory_type = config.get("MEMORY_TYPE", "Unknown").capitalize()

            metric_map = METRIC_MAPPING.get(alg_name, METRIC_MAPPING["default"])
            history = pd.DataFrame(
                list(run.scan_history(keys=[metric_map["return_col"], metric_map["time_col"]])),
                columns=[metric_map["return_col"], metric_map["time_col"]],
            )

            history["true_steps"] = history[metric_map["time_col"]].clip(upper=env_max_step)
            history = history.sort_values(metric_map["time_col"]).drop_duplicates(subset=["true_steps"])
            if len(history) < 2:
                print(f"Skipping {run.name}: insufficient data points")
                return None

            first_return = history[metric_map["return_col"]].iloc[0]
            last_return  = history[metric_map["return_col"]].iloc[-1]

            unified_steps = np.round(np.linspace(0, env_max_step, args.interp_points), decimals=5)
            scale_factor  = args.normalizing_factor / env_max_step

            interp_func = interp1d(
                history["true_steps"],
                history[metric_map["return_col"]],
                kind="linear",
                bounds_error=False,
                fill_value=(first_return, last_return),
            )
            interpolated_returns = interp_func(unified_steps)

            smoothed_returns = (
                pd.Series(interpolated_returns)
                .ewm(span=args.ewm_span, adjust=False, min_periods=1)
                .mean()
                .values
            )

            cummax_returns = lax.cummax(jnp.array(smoothed_returns))

            return pd.DataFrame({
                "Algorithm":      f"{alg_name} ({memory_type})",
                "Return":         interpolated_returns,
                "Smoothed Return": smoothed_returns,
                "Cummax Return":  np.array(cummax_returns),
                "True Steps":     unified_steps,
                "EnvName":        env_name,
                "Partial":        partial_status,
                "Seed":           str(config.get("SEED", 0)),
                "run_id":         run.id,
                "StepsNormalized": unified_steps * scale_factor,
                "EnvMaxStep":     env_max_step,
                "ScaleFactor":    scale_factor,
            })

        except Exception as e:
            print(f"Error processing {run.name}: {e}")
        return None

    all_data = [df for run in filtered_runs if (df := process_run(run)) is not None]
    if not all_data:
        print("No valid data to process")
        raise SystemExit(1)

    runs_df = pd.concat(all_data, ignore_index=True)
    runs_df.to_csv(args.raw_csv, index=False)
    print(f"Raw data saved to {args.raw_csv}")
    return runs_df


def analyse(args, runs_df):
    """Compute aggregated statistics and write output CSVs."""

    runs_df["FinalReturn"] = runs_df["Cummax Return"].astype(float)

    normal_dict = [
        "BattleShipEasy", "BattleShipMedium", "BattleShipHard",
        "MineSweeperEasy", "MineSweeperMedium", "MineSweeperHard",
        "BreakoutEasy", "BreakoutMedium", "BreakoutHard",
        "TetrisEasy", "TetrisMedium", "TetrisHard",
    ]

    for env in normal_dict:
        mask = runs_df["EnvName"] == env
        if mask.any():
            env_min = -1.0
            env_max = 0.6 if env in ["BreakoutEasy", "BreakoutMedium", "BreakoutHard"] else 1.0
            runs_df.loc[mask, "FinalReturn"] = (
                runs_df.loc[mask, "FinalReturn"] - env_min
            ) / (env_max - env_min)

    seedgroup = (
        runs_df
        .groupby(["Algorithm", "Partial", "EnvName", "run_id", "Seed"])["FinalReturn"]
        .max()
        .reset_index()
    )

    overseedgroup = seedgroup.groupby(["Algorithm", "Partial", "EnvName"]).agg(
        mean=("FinalReturn", "mean"),
        std=("FinalReturn", lambda x: x.std(ddof=1) if len(x) > 1 else 0.0),
        median=("FinalReturn", "median"),
        q25=("FinalReturn", lambda x: x.quantile(0.25)),
        q75=("FinalReturn", lambda x: x.quantile(0.75)),
        count=("FinalReturn", "count"),
    ).reset_index()

    overseedgroup["ci_lower"] = (
        overseedgroup["mean"] - 1.96 * overseedgroup["std"] / np.sqrt(overseedgroup["count"])
    )
    overseedgroup["ci_upper"] = (
        overseedgroup["mean"] + 1.96 * overseedgroup["std"] / np.sqrt(overseedgroup["count"])
    )

    overseedgroup.to_csv(args.per_env_returns_csv, index=False)
    print(f"Per-env returns saved to {args.per_env_returns_csv}")

    env_group = overseedgroup.groupby(["Algorithm", "Partial"]).agg(
        mean=("mean", "mean"),
        std=("std", "mean"),
        median=("median", "mean"),
        q25=("q25", "mean"),
        q75=("q75", "mean"),
        count=("count", "sum"),
    ).reset_index()

    env_group.to_csv(args.model_group_csv, index=False)
    print(f"Model-group data saved to {args.model_group_csv}")

    model_group = env_group.groupby(["Algorithm", "Partial"]).agg(
        mean=("mean", "mean"),
        std=("std", "mean"),
    ).reset_index()
    model_group.to_csv(args.output_csv, index=False)
    print(f"Output (obs gap / memory bias) saved to {args.output_csv}")

    # Print per-environment table (MDP vs POMDP)
    table_data = overseedgroup.copy()
    table_data["value_str"] = table_data.apply(
        lambda r: f"{r['mean']:.2f} \u00b1 {r['std']:.2f}", axis=1
    )
    pivot = table_data.pivot_table(
        index="EnvName", columns="Partial", values="value_str", aggfunc="first"
    )
    pivot.columns = [
        "MDP Return" if str(c) == "False" else "POMDP Return" for c in pivot.columns
    ]
    pivot = pivot.reset_index().rename(columns={"EnvName": "Environment"})
    print("\nPer-environment returns table:")
    print(pivot.to_string(index=False))
    pivot.to_csv(args.per_env_table_csv, index=False)
    print(f"\nSaved to {args.per_env_table_csv}")


def main():
    args = parse_args()

    if args.skip_fetch:
        print(f"Skipping W&B fetch — loading from {args.raw_csv}")
        runs_df = pd.read_csv(args.raw_csv)
    else:
        runs_df = fetch_and_process(args)

    analyse(args, runs_df)


if __name__ == "__main__":
    main()

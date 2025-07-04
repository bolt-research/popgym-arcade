"""
This file to plot the partial and full curves for all algorithms in the same plot for all environments.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jax import lax
from scipy.interpolate import interp1d

import wandb


def f(name):
    WINDOW_SIZE = 100
    SIGMA = 100
    INTERP_POINTS = 1000
    NORMALIZING_FACTOR = 200

    ENV_MAX_STEPS = {
        "CountRecallEasy": 1e8,
        "CountRecallMedium": 1e8,
        "CountRecallHard": 1e8,
        "BattleShipEasy": 1e8,
        "BattleShipMedium": 1e8,
        "BattleShipHard": 1e8,
        # other environments with default max steps 1e7
    }
    AXIS_FONT = {"fontsize": 9, "labelpad": 8}
    TICK_FONT = {"labelsize": 8}

    api = wandb.Api()
    runs = api.runs("bolt-um/Arcade-RLC-Grad")
    filtered_runs = [run for run in runs if run.state == "finished"]
    print(f"Total runs: {len(runs)}, Completed runs: {len(filtered_runs)}")

    METRIC_MAPPING = {
        "PQN": {"return_col": "returned_episode_returns", "time_col": "env_step"},
        "PQN_RNN": {"return_col": "returned_episode_returns", "time_col": "env_step"},
        "default": {"return_col": "episodic return", "time_col": "TOTAL_TIMESTEPS"},
    }

    def process_run(run):
        """Process individual W&B run with dynamic max steps per environment"""
        try:
            config = {k: v for k, v in run.config.items() if not k.startswith("_")}
            env_name = config.get("ENV_NAME", "UnknownEnv")
            partial_status = str(config.get("PARTIAL", False))

            if env_name in ENV_MAX_STEPS:
                env_max_step = ENV_MAX_STEPS[env_name]
            else:
                env_max_step = 1e8

            alg_name = config.get("ALG_NAME", "").upper()
            memory_type = "MLP"
            if alg_name == "PQN_RNN":
                memory_type = config.get("MEMORY_TYPE", "Unknown").capitalize()

            metric_map = METRIC_MAPPING.get(alg_name, METRIC_MAPPING["default"])
            # history = run.history(keys=[metric_map["return_col"], metric_map["time_col"]])
            history = list(
                run.scan_history(
                    keys=[metric_map["return_col"], metric_map["time_col"]]
                )
            )
            history = pd.DataFrame(
                history, columns=[metric_map["return_col"], metric_map["time_col"]]
            )

            history["true_steps"] = history[metric_map["time_col"]].clip(
                upper=env_max_step
            )
            history = history.sort_values(metric_map["time_col"]).drop_duplicates(
                subset=["true_steps"]
            )

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
                history["true_steps"],
                history[metric_map["return_col"]],
                kind="linear",
                bounds_error=False,
                fill_value=(first_return, last_return),
            )
            interpolated_returns = interp_func(unified_steps)

            smoothed_returns = (
                pd.Series(interpolated_returns)
                .ewm(span=100, adjust=False, min_periods=1)
                .mean()
                .values
            )

            # Compute cumulative maximum using JAX
            cummax_returns = lax.cummax(jnp.array(smoothed_returns))

            return pd.DataFrame(
                {
                    "Algorithm": f"{alg_name} ({memory_type})",
                    "Return": interpolated_returns,
                    "Smoothed Return": smoothed_returns,
                    "Cummax Return": np.array(cummax_returns),  # Convert back to NumPy
                    "True Steps": unified_steps,
                    "EnvName": env_name,
                    "Partial": partial_status,
                    "Seed": str(config.get("SEED", 0)),
                    "run_id": run.id,
                    "StepsNormalized": unified_steps / env_max_step,
                    "EnvMaxStep": env_max_step,
                    "ScaleFactor": scale_factor,
                }
            )

        except Exception as e:
            print(f"Error processing {run.name}: {str(e)}")
        return None

    # Process all runs and combine data
    # all_data = [df for run in filtered_runs if (df := process_run(run)) is not None]

    # if not all_data:
    #     print("No valid data to process")
    #     exit()
    # runs_df = pd.concat(all_data, ignore_index=True)
    # runs_df.to_pickle("rlcgrad.pkl")

    runs_df = pd.read_pickle("rlcgrad.pkl")

    def plot_comparative_curves(runs_df, name):
        """Plot comparative curves for all environments in a single plot"""
        runs_df["EnvBaseName"] = runs_df["EnvName"].apply(
            lambda x: x.replace("Easy", "").replace("Medium", "").replace("Hard", "")
        )

        envs = runs_df["EnvBaseName"].unique()

        palette = sns.color_palette("husl", len(envs))
        env_color_map = dict(zip(envs, palette))

        partial_map = {"True": "POMDP", "False": "MDP"}
        max_step = 1e8
        plt.figure(figsize=(12, 7))
        sns.set()
        plt.text(
            1,
            -0.15,
            f"{max_step:.0e}".replace("+", "").replace("0", ""),
            transform=plt.gca().transAxes,
            ha="right",
            va="top",
            fontsize=35,
            color="#666666",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2),
        )

        pomdp_handles = []
        pomdp_labels = []
        mdp_handles = []
        mdp_labels = []
        for env_base in envs:
            for partial_status in ["False", "True"]:
                data = runs_df[
                    (runs_df["EnvBaseName"] == env_base)
                    & (runs_df["Partial"] == partial_status)
                ]
                if not data.empty:
                    color = env_color_map[env_base]
                    line_style = "--" if partial_status == "True" else "-"
                    label = f"{env_base} - {partial_map[partial_status]}"

                    line = plt.plot(
                        data["StepsNormalized"],
                        data["Cummax Return"],
                        color=color,
                        linewidth=2.5,
                        linestyle=line_style,
                        label=label,
                    )[0]
                    if partial_status == "True":
                        pomdp_handles.append(line)
                        pomdp_labels.append(label)
                    else:
                        mdp_handles.append(line)
                        mdp_labels.append(label)

        plt.xlabel("Env Steps", fontsize=35)
        plt.ylabel("Episodic Return", fontsize=35)
        plt.tick_params(axis="both", which="major", labelsize=35)
        plt.grid(True, alpha=0.5)

        handles = mdp_handles + pomdp_handles
        labels = mdp_labels + pomdp_labels
        plt.legend(handles, labels, loc="best", fontsize=22, ncol=2)

        plt.title("LRU", fontsize=35, pad=12, fontweight="semibold")
        plt.tight_layout()
        plt.savefig(
            "{}.pdf".format(name), dpi=300, bbox_inches="tight", facecolor="white"
        )
        plt.close()

    plot_comparative_curves(runs_df, name)


for i in range(1):
    f(f"rlcgrad{i}")

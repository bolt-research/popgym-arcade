#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import re

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from popgym_arcade.baselines.model.builder import ActorCriticRNN
from recall_density import get_policy_terminal_saliency_map


def _easy_max_steps_for_env(env_name: str) -> int:

    mapping = {
        # Classic
        "CartPoleEasy": 200,
        "NoisyCartPoleEasy": 200,
        # Memory games
        "CountRecallEasy": 126,  # 100 + 26
        "AutoEncodeEasy": 260,   # 26 * 1 * 2 * 5
        # Gridworlds
        "NavigatorEasy": 64,     # 8 * 8
        "BattleShipEasy": 128,   # 8 * 8 * 2
        "MineSweeperEasy": 32,   # 4 * 4 * 2
        # Arcade-style
        "BreakoutEasy": 2000,
        "SkittlesEasy": 100,
        "TetrisEasy": 3000,
    }
    if env_name in mapping:
        return mapping[env_name]
    # Fallback: conservative default
    return 200


def run_multiple_seeds_and_save_csv(config, seeds, pkls_dir, max_steps=None, output_csv=None):
    """
    Run saliency analysis on multiple seeds and save the results in a CSV file.

    Args:
        config: Configuration dictionary
        seeds: List of seeds to run
        pkls_dir: Directory containing the model pkl files
        max_steps: Maximum number of steps for each episode
        output_csv: Path to save the CSV file (default: auto-generated based on config)

    Returns:
        Path to the saved CSV file
    """
    prefix = config.get("PREFIX", "PPO_RNN")

    # Create a default output path if none provided
    if output_csv is None:
        output_csv = f'saliency_results_ppo_{config["MEMORY_TYPE"]}_{config["ENV_NAME"]}_Partial={config["PARTIAL"]}.csv'

    # List to store results
    all_results = []

    # Store saliency distributions for each seed
    for seed_value in seeds:
        print(f"Processing seed {seed_value}...")

        # Update config with current seed
        config["SEED"] = seed_value

        # Create the model path for this seed
        model_path = os.path.join(
            pkls_dir,
            f"{prefix}_{config['MEMORY_TYPE']}_{config['ENV_NAME']}_model_Partial={config['PARTIAL']}_SEED={config['MODEL_SEED']}.pkl",
        )

        if not os.path.exists(model_path):
            print(f"[warn] Model file not found: {model_path}")
            continue

        # Initialize random key for this seed
        rng = jax.random.PRNGKey(seed_value)

        # Initialize and load the model
        network = ActorCriticRNN(
            rng, rnn_type=config["MEMORY_TYPE"], obs_size=config["OBS_SIZE"]
        )

        try:
            model = eqx.tree_deserialise_leaves(model_path, network)
        except Exception as e:
            print(f"[error] Failed to deserialise {model_path}: {e}")
            continue

        # Define path for saving the distribution for this seed
        dist_save_path = f'dist_ppo_{config["MEMORY_TYPE"]}_{config["ENV_NAME"]}_Partial={config["PARTIAL"]}_SEED={seed_value}.npy'

        # Run terminal saliency analysis
        try:
            grads_obs = get_policy_terminal_saliency_map(
                rng,
                model,
                config,
            )
        except Exception as e:
            print(f"[error] Failed to compute saliency map for seed {seed_value}: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Collapse to per-timestep importance
        if grads_obs.size == 0:
            dist = jnp.array([])
        else:
            grads_obs = jnp.abs(grads_obs).sum(axis=(1, 2, 3))
            denom = grads_obs.sum()
            dist = jnp.where(denom > 0, grads_obs / denom, jnp.zeros_like(grads_obs))
        print(f"Distribution sum: {dist.sum()}")
        # Convert JAX array to numpy for DataFrame
        dist_np = np.array(dist)

        # Create result dictionary
        result = {
            "seed": seed_value,
            "distribution": dist_np,
            "length": len(dist_np),
            "dist_path": dist_save_path,
        }

        all_results.append(result)
        print(f"Seed {seed_value} completed. Distribution length: {len(dist_np)}")

    if not all_results:
        print("No results collected for this model config.")
        return None

    # Process results for CSV format
    csv_data = []
    # Prefer per-env Easy maximum unless user provided an explicit max_steps
    env_max = (
        int(max_steps)
        if max_steps is not None
        else _easy_max_steps_for_env(config["ENV_NAME"])
    )
    max_length = env_max if all_results else 0

    for result in all_results:
        # Pad/trim distribution to env max length
        padded_dist = np.zeros(max_length)
        upto = min(result["length"], max_length)
        padded_dist[:upto] = result["distribution"][:upto]

        # Create row data
        row = {
            "seed": result["seed"],
            "length": result["length"],
            "dist_path": result["dist_path"],
        }

        for i in range(max_length):
            norm_pos = i / max_length if max_length > 0 else 0
            row[f"pos_{norm_pos:.3f}"] = padded_dist[i]

        csv_data.append(row)

    df = pd.DataFrame(csv_data)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

    return output_csv


def _parse_model_filename(filename: str):
    """
    Parse filename like:
    PPO_RNN_{MEMORY_TYPE}_{ENV_NAME}_model_Partial={True|False}_SEED={MODEL_SEED}.pkl

    Returns a dict with keys: PREFIX, MEMORY_TYPE, ENV_NAME, PARTIAL (bool), MODEL_SEED (int)
    """
    pattern = r"^(?P<prefix>PPO_RNN)_(?P<memory>[^_]+)_(?P<env>.+?)_model_Partial=(?P<partial>True|False)_SEED=(?P<seed>\d+)\.pkl$"
    m = re.match(pattern, filename)
    if not m:
        return None
    return {
        "PREFIX": m.group("prefix"),
        "MEMORY_TYPE": m.group("memory"),
        "ENV_NAME": m.group("env"),
        "PARTIAL": True if m.group("partial") == "True" else False,
        "MODEL_SEED": int(m.group("seed")),
    }


def _ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _parse_seeds_arg(seeds_arg: str):
    # Accept formats like: "0,1,2,3,4" or "0..4"
    if ".." in seeds_arg:
        start, end = seeds_arg.split("..", 1)
        return list(range(int(start), int(end) + 1))
    if "," in seeds_arg:
        return [int(s.strip()) for s in seeds_arg.split(",") if s.strip()]
    # Single integer
    return [int(seeds_arg)]


def _collect_pkl_files(root: str):
    """Recursively yield (file_dir, filename) for every .pkl under root."""
    for dirpath, _, files in os.walk(root):
        for f in sorted(files):
            if f.endswith(".pkl"):
                yield dirpath, f


def main():
    parser = argparse.ArgumentParser(
        description="Generate saliency CSVs for all PPO weights found under a directory tree.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("pkls_dir", type=str, help="Root directory to search for model .pkl files (recursive)")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4",
                        help="Random seeds. e.g. '0,1,2,3,4' or '0..4' or '0'")
    parser.add_argument("--obs_size", type=int, default=128,
                        help="Observation size for model construction")
    parser.add_argument("--out_dir", type=str, default="saliency_csv",
                        help="Directory to save saliency CSVs")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override max episode steps (default: auto-detected per env name)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip if output CSV already exists")
    args = parser.parse_args()

    if not os.path.isdir(args.pkls_dir):
        raise SystemExit(f"Directory not found: {args.pkls_dir}")

    seeds = _parse_seeds_arg(args.seeds)
    _ensure_dir(args.out_dir)

    pkl_files = list(_collect_pkl_files(args.pkls_dir))
    if not pkl_files:
        raise SystemExit(f"No .pkl files found under: {args.pkls_dir}")
    print(f"Found {len(pkl_files)} .pkl file(s) under {args.pkls_dir}")

    for file_dir, fname in pkl_files:
        meta = _parse_model_filename(fname)
        if meta is None:
            print(f"[warn] Skipping unrecognized file name: {fname}")
            continue

        config = {
            "ENV_NAME": meta["ENV_NAME"],
            "PARTIAL": meta["PARTIAL"],
            "MEMORY_TYPE": meta["MEMORY_TYPE"],
            "OBS_SIZE": int(args.obs_size),
            "MODEL_SEED": meta["MODEL_SEED"],
            "PREFIX": meta["PREFIX"],
        }

        out_csv = os.path.join(
            args.out_dir,
            f"saliency_results_ppo_{config['MEMORY_TYPE']}_{config['ENV_NAME']}_Partial={config['PARTIAL']}_MODELSEED={config['MODEL_SEED']}.csv",
        )
        if args.skip_existing and os.path.exists(out_csv):
            print(f"[skip] {out_csv} exists")
            continue

        print(
            f"Generating: MEMORY={config['MEMORY_TYPE']}, ENV={config['ENV_NAME']}, "
            f"Partial={config['PARTIAL']}, ModelSeed={config['MODEL_SEED']}  ({file_dir}/{fname})"
        )
        try:
            run_multiple_seeds_and_save_csv(
                config=config, seeds=seeds, pkls_dir=file_dir,
                max_steps=args.max_steps, output_csv=out_csv,
            )
        except Exception as e:
            print(f"[error] Failed to process {fname}: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()

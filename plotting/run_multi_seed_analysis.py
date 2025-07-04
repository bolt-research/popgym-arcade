#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from popgym_arcade.baselines.model.builder import QNetworkRNN
from popgym_arcade.baselines.utils import get_terminal_saliency_maps


def run_multiple_seeds_and_save_csv(config, seeds, max_steps=200, output_csv=None):
    """
    Run saliency analysis on multiple seeds and save the results in a CSV file.

    Args:
        config: Configuration dictionary
        seeds: List of seeds to run
        max_steps: Maximum number of steps for each episode
        output_csv: Path to save the CSV file (default: auto-generated based on config)

    Returns:
        Path to the saved CSV file
    """
    # Create a default output path if none provided
    if output_csv is None:
        output_csv = f'saliency_results_{config["MEMORY_TYPE"]}_{config["ENV_NAME"]}_Partial={config["PARTIAL"]}.csv'

    # List to store results
    all_results = []

    # Store saliency distributions for each seed
    for seed_value in seeds:
        print(f"Processing seed {seed_value}...")

        # Update config with current seed
        config["SEED"] = seed_value

        # Create the model path for this seed
        model_path = f"pkls_gradients/PQN_RNN_{config['MEMORY_TYPE']}_{config['ENV_NAME']}_model_Partial={config['PARTIAL']}_SEED={config['MODEL_SEED']}.pkl"

        # Initialize random key for this seed
        rng = jax.random.PRNGKey(seed_value)

        # Initialize and load the model
        network = QNetworkRNN(
            rng, rnn_type=config["MEMORY_TYPE"], obs_size=config["OBS_SIZE"]
        )
        # try:
        model = eqx.tree_deserialise_leaves(model_path, network)

        # Define path for saving the distribution for this seed
        dist_save_path = f'dist_{config["MEMORY_TYPE"]}_{config["ENV_NAME"]}_Partial={config["PARTIAL"]}_SEED={seed_value}.npy'

        # Run terminal saliency analysis
        grads_obs = get_terminal_saliency_maps(
            rng,
            model,
            config,
        )

        # print(grads_obs.shape)
        # grads_obs = grads_obs.squeeze(1)

        grads_obs = jnp.abs(grads_obs).sum(axis=(1, 2, 3))
        dist = grads_obs / grads_obs.sum()
        print(dist.sum())
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

        # except Exception as e:
        #     raise e
        #     # print(f"Error processing seed {seed_value}: {e}")

    # Process results for CSV format
    csv_data = []
    max_length = max([r["length"] for r in all_results]) if all_results else 0

    for result in all_results:
        # Pad distribution to max length if needed
        padded_dist = np.zeros(max_length)
        padded_dist[: result["length"]] = result["distribution"]

        # Create row data
        row = {
            "seed": result["seed"],
            "length": result["length"],
            "dist_path": result["dist_path"],
        }

        # Add each position value as a separate column
        for i in range(max_length):
            norm_pos = i / max_length if max_length > 0 else 0
            row[f"pos_{norm_pos:.3f}"] = padded_dist[i]

        csv_data.append(row)

    # Create DataFrame and save to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

    return output_csv

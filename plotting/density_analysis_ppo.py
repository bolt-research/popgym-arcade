#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import re
import sys
import traceback
from typing import Any, Dict, Optional, Tuple

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import popgym_arcade
from popgym_arcade.baselines.model import ActorCriticRNN, add_batch_dim
from popgym_arcade.wrappers import LogWrapper

from plotting.utils import (
    RecallDensityResult,
    algorithm_label_from_prefix,
    collect_pkl_files,
    ensure_dir,
    parse_seeds_arg,
    save_recall_density_csv,
    save_saliency_bar_data,
)


def get_gradient_ppo(
    seed: jax.random.PRNGKey,
    model: eqx.Module,
    config: Dict[str, Any],
    initial_state_and_obs: Optional[Tuple[Any, Any]] = None,
    max_episode_steps: int = 10,
) -> chex.Array:
    """Compute terminal-state gradients for a PPO recurrent policy."""
    env, env_params = popgym_arcade.make(
        config["ENV_NAME"],
        partial_obs=config["PARTIAL"],
        obs_size=config["OBS_SIZE"],
    )
    env = LogWrapper(env)
    reset = lambda rng: env.reset(rng, env_params)
    step = lambda rng, env_state, action: env.step(rng, env_state, action, env_params)

    if initial_state_and_obs is None:
        seed, reset_key = jax.random.split(seed)
        obs, env_state = reset(reset_key)
    else:
        env_state, obs = initial_state_and_obs
    obs = obs.astype(jnp.float32)

    def step_env(actor_state, critic_state, env_state, obs, done, action, seed):
        seed, step_key = jax.random.split(seed)
        obs_in = add_batch_dim(add_batch_dim(obs, 1), 1)
        done_in = add_batch_dim(add_batch_dim(done, 1), 1)
        action_in = add_batch_dim(add_batch_dim(action, 1), 1)
        actor_state, critic_state, policy, _ = model(actor_state, critic_state, (obs_in, done_in))

        del action_in  # PPO action history is not replayed for the gradient target.
        action = lax.stop_gradient(policy).logits[-1].squeeze(axis=0).argmax(axis=-1)
        obs, env_state, _, done, _ = step(step_key, env_state, action)
        return actor_state, critic_state, env_state, obs, done, action, seed

    seed, carry_key = jax.random.split(seed)
    actor_state, critic_state = model.initialize_carry(key=carry_key)
    actor_state = add_batch_dim(actor_state, 1)
    critic_state = add_batch_dim(critic_state, 1)

    done = jnp.zeros((), dtype=bool)
    action = jnp.zeros((), dtype=int)
    observations = [obs]
    dones = [done]

    for _ in range(max_episode_steps):
        actor_state, critic_state, env_state, obs, done, action, seed = jax.jit(step_env)(
            actor_state, critic_state, env_state, obs, done, action, seed
        )
        observations.append(obs.astype(jnp.float32))
        dones.append(done)
        if jnp.any(done):
            break

    observations = jnp.stack(observations, axis=0)
    dones = jnp.stack(dones, axis=0)

    def compute_logits_sum(obs_batch, done_batch):
        actor_state, critic_state = model.initialize_carry(key=seed)
        actor_state = add_batch_dim(actor_state, 1)
        critic_state = add_batch_dim(critic_state, 1)
        obs_in = add_batch_dim(obs_batch, 1, axis=1)
        done_in = add_batch_dim(done_batch, 1, axis=1)
        _, _, policy, _ = model(actor_state, critic_state, (obs_in, done_in))
        return policy.logits[-1].squeeze(axis=0).sum()

    # Use [:-1] to include the step that caused termination.
    return jax.grad(compute_logits_sum)(observations[:-1], dones[:-1])


def compute_recall_density(
    rng: jax.random.PRNGKey,
    model: eqx.Module,
    config: Dict[str, Any],
) -> np.ndarray:
    """Convert terminal gradients into a normalized per-timestep density."""
    grads_obs = get_gradient_ppo(rng, model, config)
    if grads_obs.size == 0:
        return np.array([])

    timestep_grads = jnp.abs(grads_obs).sum(axis=(1, 2, 3))
    denom = timestep_grads.sum()
    dist = jnp.where(denom > 0, timestep_grads / denom, jnp.zeros_like(timestep_grads))
    print(f"Distribution sum: {dist.sum()}")
    return np.array(dist)


def _parse_model_filename(filename: str):
    """Parse PPO recurrent checkpoint filenames."""
    pattern = (
        r"^(?P<prefix>PPO_RNN)_(?P<memory>[^_]+)_(?P<env>.+?)_model_"
        r"Partial=(?P<partial>True|False)_SEED=(?P<seed>\d+)\.pkl$"
    )
    match = re.match(pattern, filename)
    if not match:
        return None
    return {
        "PREFIX": match.group("prefix"),
        "MEMORY_TYPE": match.group("memory"),
        "ENV_NAME": match.group("env"),
        "PARTIAL": match.group("partial") == "True",
        "MODEL_SEED": int(match.group("seed")),
    }


def _build_model_path(config: Dict[str, Any], pkls_dir: str) -> str:
    return os.path.join(
        pkls_dir,
        (
            f"{config['PREFIX']}_{config['MEMORY_TYPE']}_{config['ENV_NAME']}_model_"
            f"Partial={config['PARTIAL']}_SEED={config['MODEL_SEED']}.pkl"
        ),
    )


def _build_distribution_stub(config: Dict[str, Any], seed_value: int) -> str:
    algorithm = algorithm_label_from_prefix(config["PREFIX"])
    return (
        f"dist_{algorithm}_{config['MEMORY_TYPE']}_{config['ENV_NAME']}_"
        f"Partial={config['PARTIAL']}_SEED={seed_value}.npy"
    )


def _build_output_csv_path(config: Dict[str, Any], out_dir: str) -> str:
    algorithm = algorithm_label_from_prefix(config["PREFIX"])
    return os.path.join(
        out_dir,
        (
            f"saliency_results_{algorithm}_{config['MEMORY_TYPE']}_{config['ENV_NAME']}_"
            f"Partial={config['PARTIAL']}_MODELSEED={config['MODEL_SEED']}.csv"
        ),
    )


def _load_model(model_path: str, config: Dict[str, Any], rng: jax.random.PRNGKey) -> eqx.Module:
    network = ActorCriticRNN(
        rng,
        rnn_type=config["MEMORY_TYPE"],
        obs_size=config["OBS_SIZE"],
    )
    return eqx.tree_deserialise_leaves(model_path, network)


def _compute_seed_result(
    config: Dict[str, Any],
    pkls_dir: str,
    seed_value: int,
) -> Optional[RecallDensityResult]:
    config_for_seed = dict(config)
    config_for_seed["SEED"] = seed_value

    model_path = _build_model_path(config_for_seed, pkls_dir)
    if not os.path.exists(model_path):
        print(f"[warn] Model file not found: {model_path}")
        return None

    rng = jax.random.PRNGKey(seed_value)
    try:
        model = _load_model(model_path, config_for_seed, rng)
    except Exception as exc:
        print(f"[error] Failed to deserialise {model_path}: {exc}")
        return None

    try:
        distribution = compute_recall_density(rng, model, config_for_seed)
    except Exception as exc:
        print(f"[error] Failed to compute saliency map for seed {seed_value}: {exc}")
        traceback.print_exc()
        return None

    return RecallDensityResult(
        seed=seed_value,
        distribution=distribution,
        dist_path=_build_distribution_stub(config_for_seed, seed_value),
    )


def run_multiple_seeds_and_save_csv(
    config: Dict[str, Any],
    seeds: list[int],
    pkls_dir: str,
    max_steps: Optional[int] = None,
    output_csv: Optional[str] = None,
) -> Optional[str]:
    """Run recall-density analysis across seeds and save one padded CSV."""
    if output_csv is None:
        output_csv = _build_output_csv_path(config, out_dir=".")

    results = []
    for seed_value in seeds:
        print(f"Processing seed {seed_value}...")
        result = _compute_seed_result(config, pkls_dir, seed_value)
        if result is None:
            continue
        results.append(result)
        print(f"Seed {seed_value} completed. Distribution length: {result.length}")

    if not results:
        print("No results collected for this model config.")
        return None

    return save_recall_density_csv(
        results=results,
        env_name=config["ENV_NAME"],
        output_csv=output_csv,
        max_steps=max_steps,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate recall-density CSVs for PPO recurrent checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "pkls_dir",
        type=str,
        help="Root directory to search for model .pkl files recursively",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4",
        help="Random seeds, e.g. '0,1,2,3,4', '0..4', or '0'",
    )
    parser.add_argument(
        "--obs_size",
        type=int,
        default=128,
        help="Observation size for model construction",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="saliency_csv",
        help="Directory to save recall-density CSVs",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Override max episode steps; otherwise infer from env name",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip model configs whose output CSV already exists",
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default=None,
        help="Optional path to save aggregated bar-chart data across generated saliency CSVs",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.pkls_dir):
        raise SystemExit(f"Directory not found: {args.pkls_dir}")

    seeds = parse_seeds_arg(args.seeds)
    ensure_dir(args.out_dir)

    pkl_files = list(collect_pkl_files(args.pkls_dir))
    if not pkl_files:
        raise SystemExit(f"No .pkl files found under: {args.pkls_dir}")
    print(f"Found {len(pkl_files)} .pkl file(s) under {args.pkls_dir}")

    for file_dir, filename in pkl_files:
        meta = _parse_model_filename(filename)
        if meta is None:
            print(f"[warn] Skipping unrecognized file name: {filename}")
            continue

        config = {
            "ENV_NAME": meta["ENV_NAME"],
            "PARTIAL": meta["PARTIAL"],
            "MEMORY_TYPE": meta["MEMORY_TYPE"],
            "OBS_SIZE": int(args.obs_size),
            "MODEL_SEED": meta["MODEL_SEED"],
            "PREFIX": meta["PREFIX"],
        }
        out_csv = _build_output_csv_path(config, args.out_dir)

        if args.skip_existing and os.path.exists(out_csv):
            print(f"[skip] {out_csv} exists")
            continue

        print(
            f"Generating: MEMORY={config['MEMORY_TYPE']}, ENV={config['ENV_NAME']}, "
            f"Partial={config['PARTIAL']}, ModelSeed={config['MODEL_SEED']} "
            f"({file_dir}/{filename})"
        )
        try:
            run_multiple_seeds_and_save_csv(
                config=config,
                seeds=seeds,
                pkls_dir=file_dir,
                max_steps=args.max_steps,
                output_csv=out_csv,
            )
        except Exception as exc:
            print(f"[error] Failed to process {filename}: {exc}")
            traceback.print_exc()

    if args.summary_csv is not None:
        summary_dir = os.path.dirname(args.summary_csv)
        if summary_dir:
            ensure_dir(summary_dir)
        save_saliency_bar_data(args.out_dir, args.summary_csv)


if __name__ == "__main__":
    main()

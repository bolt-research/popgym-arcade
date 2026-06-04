#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
from typing import Any, Dict, Optional, Tuple

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
from jax import lax
from matplotlib import pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import popgym_arcade
from popgym_arcade.baselines.model import QNetworkRNN, add_batch_dim
from plotting.heatmap import HeatMap
from popgym_arcade.wrappers import LogWrapper


def get_qnetwork_saliency_maps(
    seed: jax.random.PRNGKey,
    model: eqx.Module,
    config: Dict[str, Any],
    max_steps: int = 5,
    initial_state_and_obs: Optional[Tuple[Any, Any]] = None,
) -> Tuple[list, chex.Array, list]:
    """Compute PQN saliency maps across a rollout."""
    seed, reset_key = jax.random.split(seed)
    env, env_params = popgym_arcade.make(
        config["ENV_NAME"], partial_obs=config["PARTIAL"], obs_size=config["OBS_SIZE"]
    )
    env = LogWrapper(env)
    n_envs = 1
    vmap_reset = lambda n: lambda rng: jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(rng, n), env_params
    )
    vmap_step = lambda n: lambda rng, env_state, action: jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )(jax.random.split(rng, n), env_state, action, env_params)

    if initial_state_and_obs is None:
        obs_seq, env_state = vmap_reset(n_envs)(reset_key)
    else:
        env_state, obs_seq = initial_state_and_obs

    done_seq = jnp.zeros(n_envs, dtype=bool)
    action_seq = jnp.zeros(n_envs, dtype=int)
    obs_seq = obs_seq[jnp.newaxis, :].astype(jnp.float32)
    done_seq = done_seq[jnp.newaxis, :]
    action_seq = action_seq[jnp.newaxis, :]
    grads = []
    grad_accumulator = []

    def step_env_and_compute_grads(env_state, obs_seq, action_seq, done_seq, key):
        def q_val_fn(obs_batch, action_batch, done_batch):
            hidden_state = model.initialize_carry(key=key)
            hidden_state = add_batch_dim(hidden_state, n_envs)
            _, q_values = model(hidden_state, obs_batch, done_batch, action_batch)
            action = jnp.argmax(lax.stop_gradient(q_values)[-1], axis=-1)
            new_obs, new_state, _, new_done, _ = vmap_step(n_envs)(seed, env_state, action)
            return q_values[-1].sum(), (new_state, new_obs, action, new_done)

        grads_obs, (new_state, new_obs, action, new_done) = jax.grad(
            q_val_fn, argnums=0, has_aux=True
        )(obs_seq, action_seq, done_seq)
        obs_seq = jnp.concatenate([obs_seq, new_obs[jnp.newaxis, :].astype(jnp.float32)])
        action_seq = jnp.concatenate([action_seq, action[jnp.newaxis, :]])
        done_seq = jnp.concatenate([done_seq, new_done[jnp.newaxis, :]])
        return grads_obs, new_state, obs_seq, action_seq, done_seq

    for _ in range(max_steps):
        seed, rng = jax.random.split(seed)
        grads_obs, env_state, obs_seq, action_seq, done_seq = jax.jit(
            step_env_and_compute_grads
        )(env_state, obs_seq, action_seq, done_seq, rng)
        grads.append(grads_obs)
        grad_accumulator.append(jnp.sum(grads_obs, axis=0))
        if done_seq[-1].any():
            break

    return grads, obs_seq, grad_accumulator


def plot_qnetwork_pixel_vis(
    maps: list,
    obs_seq: chex.Array,
    config: Dict[str, Any],
    alpha: float = 0.5,
    gaussian_std: int = 6,
    cmap: str = "afmhot",
    use_latex: bool = False,
    output_path: Optional[str] = None,
) -> None:
    """Render a single-row PQN saliency overlay figure."""
    sns.set(style="whitegrid", palette="pastel", font_scale=1.2)
    if use_latex:
        plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    if output_path is None:
        output_path = (
            f"pqn_saliency_overlay_{config['MEMORY_TYPE']}_{config['ENV_NAME']}_"
            f"partial={config['PARTIAL']}_seed={config['SEED']}.pdf"
        )

    saliency_maps = np.asarray(jnp.abs(maps[-1]).squeeze(axis=1).mean(axis=-1))
    num_frames = len(saliency_maps)
    fig, axes = plt.subplots(1, num_frames, figsize=(4 * num_frames, 4))
    if num_frames == 1:
        axes = [axes]

    image_artist = None
    vmin = float(np.min(saliency_maps))
    vmax = float(np.max(saliency_maps))

    for index, axis in enumerate(axes):
        observation = np.asarray(obs_seq[index]).squeeze()
        if observation.ndim == 3 and observation.shape[-1] == 1:
            observation = observation[..., 0]
        elif observation.ndim == 3 and observation.shape[-1] not in (3, 4):
            observation = observation.mean(axis=-1)

        heat_map = HeatMap(observation, saliency_maps[index], gaussian_std=gaussian_std)

        if np.asarray(heat_map.image).ndim == 2:
            axis.imshow(heat_map.image, cmap="gray")
        else:
            axis.imshow(heat_map.image)

        image_artist = axis.imshow(
            heat_map.heat_map,
            alpha=alpha,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        offset = num_frames - 1 - index
        if use_latex:
            title = r"$o_{t}$" if offset == 0 else rf"$o_{{t-{offset}}}$"
        else:
            title = "o_t" if offset == 0 else f"o_t-{offset}"
        axis.set_title(title, fontsize=24, pad=16)
        axis.axis("off")

    colorbar_axis = fig.add_axes([0.92, 0.18, 0.015, 0.64])
    colorbar = fig.colorbar(image_artist, cax=colorbar_axis, orientation="vertical")
    colorbar.ax.tick_params(labelsize=14)
    colorbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(r"$\mathdefault{%.1e}$"))
    colorbar.update_ticks()

    plt.subplots_adjust(left=0.03, right=0.9, bottom=0.08, top=0.88, wspace=0.05)
    plt.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize PQN pixel saliency maps for a trained recurrent Q-network."
    )
    parser.add_argument("--model-path", type=str, required=True, help="Path to model weights (.pkl)")
    parser.add_argument("--env-name", type=str, required=True, help="Environment name")
    parser.add_argument("--memory-type", type=str, required=True, help="Recurrent memory type")
    parser.add_argument("--seed", type=int, default=0, help="Evaluation seed")
    parser.add_argument("--obs-size", type=int, default=128, help="Observation size")
    parser.add_argument("--partial", action="store_true", help="Use partial observability")
    parser.add_argument("--max-steps", type=int, default=30, help="Maximum rollout steps")
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay transparency")
    parser.add_argument("--gaussian-std", type=int, default=6, help="Gaussian smoothing std")
    parser.add_argument("--cmap", type=str, default="afmhot", help="Heatmap color map")
    parser.add_argument("--use-latex", action="store_true", help="Enable LaTeX rendering")
    parser.add_argument("--output", type=str, default=None, help="Optional output PDF path")
    return parser.parse_args()


def main():
    args = parse_args()
    config = {
        "ENV_NAME": args.env_name,
        "PARTIAL": args.partial,
        "MEMORY_TYPE": args.memory_type,
        "SEED": args.seed,
        "OBS_SIZE": args.obs_size,
        "MODEL_PATH": args.model_path,
    }

    rng = jax.random.PRNGKey(config["SEED"])
    network = QNetworkRNN(rng, rnn_type=config["MEMORY_TYPE"], obs_size=config["OBS_SIZE"])
    model = eqx.tree_deserialise_leaves(config["MODEL_PATH"], network)
    grads, obs_seq, _ = get_qnetwork_saliency_maps(rng, model, config, max_steps=args.max_steps)
    plot_qnetwork_pixel_vis(
        grads,
        obs_seq,
        config,
        alpha=args.alpha,
        gaussian_std=args.gaussian_std,
        cmap=args.cmap,
        use_latex=args.use_latex,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()

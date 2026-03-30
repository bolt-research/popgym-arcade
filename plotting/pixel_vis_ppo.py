"""
This file is to visualize the PPO pixel saliency maps for a trained recurrent policy.

Usage example:
    python pixel_vis_ppo.py --model-path PATH_TO_YOUR_MODEL_WEIGHTS.pkl --env-name ENV_NAME --memory-type MEMORY_TYPE --seed SEED --obs-size OBS_SIZE --partial PARTIAL --max-steps MAX_STEPS --mode MODE --use-latex USE_LATEX --output OUTPUT_PATH
"""

import argparse
import os
import sys
from typing import Any, Dict, Optional, Tuple

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.ticker as ticker
import seaborn as sns
from jax import lax
from matplotlib import pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import popgym_arcade
from popgym_arcade.baselines.model import ActorCriticRNN, add_batch_dim
from popgym_arcade.wrappers import LogWrapper


def get_policy_saliency_map(
    seed: jax.random.PRNGKey,
    model: eqx.Module,
    config: Dict[str, Any],
    max_steps: int = 5,
    initial_state_and_obs: Optional[Tuple[Any, Any]] = None,
) -> Tuple[list, chex.Array, list]:
    """Compute PPO saliency maps across a rollout."""
    seed, reset_key = jax.random.split(seed)
    env, env_params = popgym_arcade.make(
        config["ENV_NAME"], partial_obs=config["PARTIAL"], obs_size=config["OBS_SIZE"]
    )
    env = LogWrapper(env)
    reset = lambda rng: env.reset(rng, env_params)
    step_fn = lambda rng, env_state, action: env.step(rng, env_state, action, env_params)

    if initial_state_and_obs is None:
        obs, env_state = reset(reset_key)
    else:
        env_state, obs = initial_state_and_obs
    obs = obs.astype(jnp.float32)

    done = jnp.zeros((), dtype=bool)
    obs_seq = obs[jnp.newaxis, :]
    done_seq = done[jnp.newaxis]
    grads = []
    grad_accumulator = []

    def step_env_and_compute_grads(env_state, obs_seq, done_seq, key):
        def policy_logits_fn(obs_batch, done_batch):
            actor_state, critic_state = model.initialize_carry(key=key)
            actor_state = add_batch_dim(actor_state, 1)
            critic_state = add_batch_dim(critic_state, 1)
            obs_in = add_batch_dim(obs_batch, 1, axis=1)
            done_in = add_batch_dim(done_batch, 1, axis=1)
            _, _, policy, _ = model(actor_state, critic_state, (obs_in, done_in))
            action = lax.stop_gradient(policy).logits[-1].squeeze(axis=0).argmax(axis=-1)
            step_key, _ = jax.random.split(key)
            new_obs, new_state, _, new_done, _ = step_fn(step_key, env_state, action)
            return policy.logits[-1].squeeze(axis=0).sum(), (new_state, new_obs, new_done)

        grads_obs, (new_state, new_obs, new_done) = jax.grad(
            policy_logits_fn, argnums=0, has_aux=True
        )(obs_seq, done_seq)
        obs_seq = jnp.concatenate([obs_seq, new_obs[jnp.newaxis, :].astype(jnp.float32)])
        done_seq = jnp.concatenate([done_seq, new_done[jnp.newaxis]])
        return grads_obs, new_state, obs_seq, done_seq

    for _ in range(max_steps):
        seed, rng = jax.random.split(seed)
        grads_obs, env_state, obs_seq, done_seq = jax.jit(step_env_and_compute_grads)(
            env_state, obs_seq, done_seq, rng
        )
        grads.append(grads_obs)
        grad_accumulator.append(jnp.sum(grads_obs, axis=0))
        if done_seq[-1].any():
            break

    return grads, obs_seq, grad_accumulator


def vis_fn(
    maps: list,
    obs_seq: chex.Array,
    config: Dict[str, Any],
    cmap: str = "hot",
    mode: str = "line",
    use_latex: bool = False,
    output_path: Optional[str] = None,
) -> None:
    """Visualize saliency maps alongside observations."""
    sns.set(style="whitegrid", palette="pastel", font_scale=1.2)
    if use_latex:
        plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    length = len(maps)
    if output_path is None:
        output_path = (
            f"pixel_vis_ppo_{config['MEMORY_TYPE']}_{config['ENV_NAME']}_"
            f"PARTIAL={config['PARTIAL']}_SEED={config['SEED']}.pdf"
        )

    if mode == "line":
        fig, axs = plt.subplots(2, length, figsize=(30, 8))
        maps_last = jnp.abs(maps[-1])
        for i in range(length):
            obs_ax = axs[0][i]
            obs_ax.imshow(obs_seq[i], cmap="gray")
            obs_ax.set_title(rf"$o_{{{i}}}$" if use_latex else f"o{i}", fontsize=25, pad=20)
            obs_ax.axis("off")

            map_ax = axs[1][i]
            saliency_map = maps_last[i].mean(axis=-1)
            im = map_ax.imshow(saliency_map, cmap=cmap)
            if use_latex:
                title = (
                    rf"$\sum\limits_{{a \in A}}\left|\frac{{\partial \pi(\hat{{s}}_{{{length - 1}}})}}"
                    rf"{{\partial o_{{{i}}}}}\right|$"
                )
            else:
                title = f"dpi(s{length - 1}) / do{i}"
            map_ax.set_title(title, fontsize=25, pad=30)
            map_ax.axis("off")

        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical", label="Gradient Magnitude")
        cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
        plt.subplots_adjust(hspace=0.1, right=0.9)
    elif mode == "grid":
        maps = [jnp.abs(m) for m in maps]
        fig, axs = plt.subplots(length, length + 1, figsize=(120, 150))
        for i in range(length):
            obs_ax = axs[i][0]
            obs_ax.imshow(obs_seq[i], cmap="gray")
            obs_ax.set_title(rf"$o_{{{i}}}$" if use_latex else f"o{i}", fontsize=100, pad=90)
            obs_ax.axis("off")
            for j in range(length):
                map_ax = axs[i][j + 1]
                if j < maps[i].shape[0]:
                    im = map_ax.imshow(maps[i][j].mean(axis=-1), cmap=cmap)
                    if use_latex:
                        title = (
                            rf"$\sum\limits_{{a \in A}}\left|\frac{{\partial \pi(\hat{{s}}_{{{length - 1}}})}}"
                            rf"{{\partial o_{{{j}}}}}\right|$"
                        )
                    else:
                        title = f"dpi(s{length - 1}) / do{j}"
                    map_ax.set_title(title, fontsize=100, pad=110)
                map_ax.axis("off")

        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical", label="Gradient Magnitude")
        cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
        cbar.ax.tick_params(axis="y", labelsize=80)
        cbar.set_label("Gradient Magnitude", fontsize=100)
        plt.subplots_adjust(hspace=0.1, right=0.9)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    plt.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize PPO pixel saliency maps for a trained recurrent policy."
    )
    parser.add_argument("--model-path", type=str, required=True, help="Path to model weights (.pkl)")
    parser.add_argument("--env-name", type=str, required=True, help="Environment name")
    parser.add_argument("--memory-type", type=str, required=True, help="Recurrent memory type")
    parser.add_argument("--seed", type=int, default=0, help="Evaluation seed")
    parser.add_argument("--obs-size", type=int, default=128, help="Observation size")
    parser.add_argument("--partial", action="store_true", help="Use partial observability")
    parser.add_argument("--max-steps", type=int, default=30, help="Maximum rollout steps")
    parser.add_argument("--mode", type=str, default="line", choices=["line", "grid"])
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
    network = ActorCriticRNN(rng, rnn_type=config["MEMORY_TYPE"], obs_size=config["OBS_SIZE"])
    model = eqx.tree_deserialise_leaves(config["MODEL_PATH"], network)
    grads, obs_seq, _ = get_policy_saliency_map(rng, model, config, max_steps=args.max_steps)
    vis_fn(grads, obs_seq, config, mode=args.mode, use_latex=args.use_latex, output_path=args.output)


if __name__ == "__main__":
    main()

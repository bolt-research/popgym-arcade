import csv  # Import csv for direct CSV writing if needed
import os  # Import os for path manipulation
from typing import Any, Dict, List, Optional, Tuple, Union

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd  # Import pandas for CSV handling
import seaborn as sns
from jax import lax
from matplotlib import pyplot as plt

import popgym_arcade
from memax.equinox.train_utils import add_batch_dim
from popgym_arcade.wrappers import LogWrapper


@eqx.filter_jit
def filter_scan(f, init, xs, *args, **kwargs):
    """Same as lax.scan, but allows to have eqx.Module in carry"""
    init_dynamic_carry, static_carry = eqx.partition(init, eqx.is_array)

    def to_scan(dynamic_carry, x):
        carry = eqx.combine(dynamic_carry, static_carry)
        new_carry, out = f(carry, x)
        dynamic_new_carry, _ = eqx.partition(new_carry, eqx.is_array)
        return dynamic_new_carry, out

    out_carry, out_ys = lax.scan(to_scan, init_dynamic_carry, xs, *args, **kwargs)
    return eqx.combine(out_carry, static_carry), out_ys


def get_saliency_maps(
    seed: jax.random.PRNGKey,
    model: eqx.Module,
    config: Dict,
    max_steps: int = 5,
    initial_state_and_obs: Optional[Tuple[Any, Any]] = None,
) -> Tuple[list, chex.Array, list]:
    """
    Computes saliency maps for visualizing model attention patterns in given environments.

    Args:
        seed: JAX PRNG key for reproducible randomization
        model: Pre-trained model containing parameter weights to analyze
        config: Configuration dictionary containing model and environment settings
        max_steps: Max number of sequential steps to generate visualization for.
                The total number may be less, if the episode resets before reaching max_steps.
        initial_state: Optional tuple (initial_state, initial_obs) for the env, otherwise will
                use the reset function to get the initial state.

    Returns:
        grads: List of gradient-based saliency maps. The i-th element contains i saliency maps
               showing feature importance at each timestep. Each map is a JAX array matching
               the observation space dimensions.
        obs_seq: Sequence of environment observations captured during analysis
        grad_accumulator: Cumulative sum of saliency maps across timesteps. The i-th element
                          represents aggregated feature importance up to that step.

    Example:
        When analyzing 10 timesteps:
        - `grads[4]` contains 4 saliency maps showing per-step feature importance
        - `grad_accumulator[7]` provides the accumulated importance map through step 7
        - All outputs maintain the original observation dimensions for direct visual comparison
    """

    seed, _rng = jax.random.split(seed)
    env, env_params = popgym_arcade.make(
        config["ENV_NAME"], partial_obs=config["PARTIAL"], obs_size=config["OBS_SIZE"]
    )
    env = LogWrapper(env)
    n_envs = 1
    vmap_reset = lambda n_envs: lambda rng: jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(rng, n_envs), env_params
    )
    vmap_step = lambda n_envs: lambda rng, env_state, action: jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )(jax.random.split(rng, n_envs), env_state, action, env_params)
    if initial_state_and_obs is None:
        obs_seq, env_state = vmap_reset(n_envs)(_rng)
        obs_seq = obs_seq / 255.0
    else:
        env_state, obs_seq = initial_state_and_obs
        obs_seq = obs_seq / 255.0
    done_seq = jnp.zeros(n_envs, dtype=bool)
    action_seq = jnp.zeros(n_envs, dtype=int)
    obs_seq = obs_seq[jnp.newaxis, :].astype(jnp.float32)
    done_seq = done_seq[jnp.newaxis, :]
    action_seq = action_seq[jnp.newaxis, :]
    # save all the grads separate from each state
    grads = []
    # save cumulated grads
    grad_accumulator = []

    def step_env_and_compute_grads(env_state, obs_seq, action_seq, done_seq, key):

        def q_val_fn(obs_batch, action_batch, done_batch):
            hs = model.initialize_carry(key=key)
            hs = add_batch_dim(hs, n_envs)
            _, q_val = model(hs, obs_batch, done_batch, action_batch)
            q_val_action = lax.stop_gradient(q_val)
            action = jnp.argmax(q_val_action[-1], axis=-1)
            new_obs, new_state, reward, new_done, info = vmap_step(n_envs)(
                seed, env_state, action
            )
            new_obs = new_obs / 255.0
            return q_val[-1].sum(), (new_state, new_obs, action, new_done)

        grads_obs, (new_state, new_obs, action, new_done) = jax.grad(
            q_val_fn, argnums=0, has_aux=True
        )(obs_seq, action_seq, done_seq)
        obs_seq = jnp.concatenate([obs_seq, new_obs[jnp.newaxis, :]].astype(jnp.float32))
        action_seq = jnp.concatenate([action_seq, action[jnp.newaxis, :]])
        done_seq = jnp.concatenate([done_seq, new_done[jnp.newaxis, :]])
        return grads_obs, new_state, obs_seq, action_seq, done_seq

    for _ in range(max_steps):
        rng, _rng = jax.random.split(seed, 2)
        grads_obs, env_state, obs_seq, action_seq, done_seq = jax.jit(
            step_env_and_compute_grads
        )(env_state, obs_seq, action_seq, done_seq, rng)
        grads.append(grads_obs)
        grad_accumulator.append(jnp.sum(grads_obs, axis=0))
        if done_seq[-1].any():
            break

    return grads, obs_seq, grad_accumulator


def get_terminal_saliency_maps(
    seed: jax.random.PRNGKey,
    model: eqx.Module,
    config: Dict,
    initial_state_and_obs: Optional[Tuple[Any, Any]] = None,
) -> Tuple[list, chex.Array, list]:
    env, env_params = popgym_arcade.make(
        config["ENV_NAME"], partial_obs=config["PARTIAL"], obs_size=config["OBS_SIZE"]
    )
    env = LogWrapper(env)
    reset = lambda rng: env.reset(rng, env_params)
    step = lambda rng, env_state, action: env.step(rng, env_state, action, env_params)
    if initial_state_and_obs is None:
        seed, _rng = jax.random.split(seed)
        obs, env_state = reset(_rng)
        obs = obs.astype(jnp.float32) / 255.0
    else:
        env_state, obs = initial_state_and_obs
        obs = obs.astype(jnp.float32) / 255.0

    # Step 1: Compute rollout until terminal state

    def step_env(hs, env_state, obs, done, action, seed):
        seed, step_key = jax.random.split(seed)
        # Add time and batch dim for model
        inputs = [add_batch_dim(add_batch_dim(x, 1), 1) for x in [obs, done, action]]

        hs, q_val = model(hs, *inputs)
        # Remove batch dim
        q_val = jnp.squeeze(q_val, (0, 1))
        action = jnp.argmax(q_val, axis=-1)
        obs, env_state, reward, done, info = step(step_key, env_state, action)
        obs = obs.astype(jnp.float32) / 255.0
        return (hs, env_state, obs, done, action, seed)

    seed, rng = jax.random.split(seed)
    hs = model.initialize_carry(key=rng)
    hs = add_batch_dim(hs, 1)
    done = jnp.zeros((), dtype=bool)
    action = jnp.zeros((), dtype=int)
    observations, dones, actions = [obs], [done], [action]
    while not jnp.any(done):
        hs, env_state, obs, done, action, seed = jax.jit(step_env)(
            hs, env_state, obs, done, action, seed
        )
        observations.append(obs.astype(jnp.float32))
        dones.append(done)
        actions.append(action)

    observations = jnp.stack(observations, axis=0)
    dones = jnp.stack(dones, axis=0)
    actions = jnp.stack(actions, axis=0)

    # Step 2: Compute gradients at terminal state
    def compute_q(obs_batch, action_batch, done_batch):
        hs = model.initialize_carry(key=seed)
        hs = add_batch_dim(hs, 1)
        inputs = [
            add_batch_dim(x, 1, axis=1) for x in [obs_batch, done_batch, action_batch]
        ]
        _, q_val = model(hs, *inputs)
        return jnp.abs(q_val[-1].sum())

    # Use -2 because if done == true, obs is next episode
    return jax.grad(compute_q)(observations[:-2], actions[:-2], dones[:-2])


def vis_fn(
    maps: list,
    obs_seq: chex.Array,
    config: dict,
    cmap: str = "hot",
    mode: str = "line",
    use_latex: bool = True,
) -> None:
    """
    Generates visualizations of model attention patterns using saliency mapping techniques.

    Args:
        maps: Sequential collection of gradient-based importance maps. Each element
              contains activation patterns for corresponding timesteps.
        obs_seq: Temporal sequence of input observations captured from environment states
        config: Configuration parameters containing environment specifications and
                model hyperparameters
        cmap: Color palette for heatmap visualization (default: 'hot')
        mode: Layout configuration selector:
              - 'line': Sequential horizontal display for time series analysis
              - 'grid': Matrix layout comparing observation-attention relationships
        use_latex: Boolean flag to enable LaTeX rendering for titles and labels
                If you don't have latex installed, you will get an error unless this is false

    Visualizes:
        Dual-channel displays showing original observations (top) with corresponding
        gradient activation patterns (bottom) when using 'line' mode. 'grid' mode
        generates comparative matrices demonstrating attention evolution across steps.
    """

    sns.set(style="whitegrid", palette="pastel", font_scale=1.2)
    if use_latex:
        plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    length = len(maps)
    if mode == "line":
        fig, axs = plt.subplots(
            2, length, figsize=(30, 8)
        )  # Adjusted figure size if necessary
        maps_last = jnp.abs(maps[-1])
        for i in range(length):
            # Top row: Original observations
            obs = axs[0][i]
            obs.imshow(obs_seq[i].squeeze(axis=0), cmap="gray")
            if use_latex:
                obs.set_title(rf"$o_{{{i}}}$", fontsize=25, pad=20)
            else:
                obs.set_title(f"o{i}", fontsize=25, pad=20)
            obs.axis("off")

            # Bottom row: Saliency map
            map_ax = axs[1][i]
            saliency_map = maps_last[i].squeeze(axis=0).mean(axis=-1)
            im = map_ax.imshow(saliency_map, cmap="hot")
            if use_latex:
                map_ax.set_title(
                    rf"$\sum\limits_{{a \in A}}\left|\frac{{\partial Q(\hat{{s}}_{{{length - 1}}}, a_{{{length - 1}}})}}{{\partial o_{{{i}}}}}\right|$",
                    fontsize=25,
                    pad=30,
                )
            else:
                map_ax.set_title(
                    f"dQ(s{length - 1}, a{length - 1})", fontsize=25, pad=30
                )
            map_ax.axis("off")

        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(
            im, cax=cbar_ax, orientation="vertical", label="Gradient Magnitude"
        )
        formatter = ticker.FormatStrFormatter("%.4f")
        cbar.ax.yaxis.set_major_formatter(formatter)
        plt.subplots_adjust(
            hspace=0.1, right=0.9
        )  # Adjust the main plot to the left to make room for the colorbar
        plt.savefig(
            f'{config["ENV_NAME"]}_PARTIAL={config["PARTIAL"]}_SEED={config["SEED"]}.pdf',
            format="pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()
    elif mode == "grid":
        maps = [jnp.abs(m) for m in maps]
        fig, axs = plt.subplots(length, length + 1, figsize=(120, 150))
        for i in range(length):
            obs_ax = axs[i][0]
            obs_ax.imshow(obs_seq[i].squeeze(axis=0), cmap="gray")
            if use_latex:
                obs_ax.set_title(rf"$o_{{{i}}}$", fontsize=100, pad=90)
            else:
                obs_ax.set_title(f"$o{i}", fontsize=100, pad=90)
            obs_ax.axis("off")
            for j in range(length):
                # print(maps[i].shape)
                if j < maps[i].shape[0]:
                    map_ax = axs[i][j + 1]
                    im = map_ax.imshow(
                        maps[i][j].squeeze(axis=0).mean(axis=-1), cmap=cmap
                    )
                    if use_latex:
                        map_ax.set_title(
                            rf"$\sum\limits_{{a \in A}}\left|\frac{{\partial Q(\hat{{s}}_{{{length - 1}}}, a_{{{length - 1}}})}}{{\partial o_{{{j}}}}}\right|$",
                            fontsize=100,
                            pad=110,
                        )
                    else:
                        map_ax.set_title(
                            f"$dQ(s{length - 1}, a{length - 1}) / do{j} $",
                            fontsize=100,
                            pad=110,
                        )
                    map_ax.axis("off")
                else:
                    map_ax = axs[i][j + 1]
                    map_ax.axis("off")

        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(
            im, cax=cbar_ax, orientation="vertical", label="Gradient Magnitude"
        )
        formatter = ticker.FormatStrFormatter("%.4f")
        cbar.ax.yaxis.set_major_formatter(formatter)
        cbar.ax.tick_params(axis="y", labelsize=80)
        cbar.set_label("Gradient Magnitude", fontsize=100)
        plt.subplots_adjust(
            hspace=0.1, right=0.9
        )  # Adjust the main plot to the left to make room for the colorbar
        plt.savefig(
            f'{config["MEMORY_TYPE"]}_{config["ENV_NAME"]}_PARTIAL={config["PARTIAL"]}_SEED={config["SEED"]}.pdf',
            format="pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

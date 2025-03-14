import chex
import equinox as eqx
from jax import lax
from typing import Tuple, Dict
import jax
import jax.numpy as jnp
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from popgym_arcade.baselines.model.memorax import add_batch_dim
from popgym_arcade.wrappers import LogWrapper
import popgym_arcade


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
        num_steps=5
) -> Tuple[list, chex.Array, list]:
    """
    Computes saliency maps for visualizing model attention patterns in given environments.

    Args:
        seed: JAX PRNG key for reproducible randomization
        model: Pre-trained model containing parameter weights to analyze
        config: Configuration dictionary containing model and environment settings
        num_steps: Number of sequential steps to generate visualization for

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
    env, env_params = popgym_arcade.make(config["ENV_NAME"], partial_obs=config["PARTIAL"])
    env = LogWrapper(env)
    n_envs = 1
    vmap_reset = lambda n_envs: lambda rng: jax.vmap(env.reset, in_axes=(0, None))(
                jax.random.split(rng, n_envs), env_params
            )
    vmap_step = lambda n_envs: lambda rng, env_state, action: jax.vmap(env.step, in_axes=(0, 0, 0, None))(
        jax.random.split(rng, n_envs), env_state, action, env_params
    )
    obs_seq, env_state = vmap_reset(n_envs)(_rng)
    done_seq = jnp.zeros(n_envs, dtype=bool)
    action_seq = jnp.zeros(n_envs, dtype=int)
    obs_seq = obs_seq[jnp.newaxis, :]
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
            new_obs, new_state, reward, new_done, info = vmap_step(n_envs)(seed, env_state, action)
            return q_val[-1].sum(), (new_state, new_obs, action, new_done)

        grads_obs, (new_state, new_obs, action, new_done) = jax.grad(q_val_fn, argnums=0, has_aux=True)(
            obs_seq,
            action_seq,
            done_seq
        )
        obs_seq = jnp.concatenate([obs_seq, new_obs[jnp.newaxis, :]])
        action_seq = jnp.concatenate([action_seq, action[jnp.newaxis, :]])
        done_seq = jnp.concatenate([done_seq, new_done[jnp.newaxis, :]])
        return grads_obs, new_state, obs_seq, action_seq, done_seq

    for _ in range(num_steps):
        rng, _rng = jax.random.split(seed, 2)
        grads_obs, env_state, obs_seq, action_seq, done_seq = step_env_and_compute_grads(
            env_state,
            obs_seq,
            action_seq,
            done_seq,
            rng
        )
        grads.append(grads_obs)
        grad_accumulator.append(jnp.sum(grads_obs, axis=0))

    return grads, obs_seq, grad_accumulator



def vis_fn(
    maps: list,
    obs_seq: chex.Array,
    config: dict,
    cmap: str = 'hot',
    mode: str = 'line'
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

    Visualizes:
        Dual-channel displays showing original observations (top) with corresponding
        gradient activation patterns (bottom) when using 'line' mode. 'grid' mode
        generates comparative matrices demonstrating attention evolution across steps.
    """

    sns.set(style="whitegrid", palette="pastel", font_scale=1.2)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    length = len(maps)
    if mode == 'line':
        fig, axs = plt.subplots(2, length, figsize=(30, 8))  # Adjusted figure size if necessary
        maps_last = jnp.abs(maps[-1])
        for i in range(length):
            # Top row: Original observations
            obs = axs[0][i]
            obs.imshow(obs_seq[i].squeeze(axis=0), cmap='gray')
            obs.set_title(rf"$o_{{{i}}}$", fontsize=25, pad=20)
            obs.axis('off')

            # Bottom row: Saliency map
            map_ax = axs[1][i]
            saliency_map = maps_last[i].squeeze(axis=0).mean(axis=-1)
            im = map_ax.imshow(saliency_map, cmap='hot')
            map_ax.set_title(rf"$\sum\limits_{{a \in A}}\left|\frac{{\partial Q(\hat{{s}}_{{{length-1}}}, a_{{{length-1}}})}}{{\partial o_{{{i}}}}}\right|$", fontsize=25, pad=30)
            map_ax.axis('off')

        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical', label="Gradient Magnitude")
        formatter = ticker.FormatStrFormatter('%.4f')
        cbar.ax.yaxis.set_major_formatter(formatter)
        plt.subplots_adjust(hspace=0.1, right=0.9)  # Adjust the main plot to the left to make room for the colorbar
        plt.savefig(f'{config["ENV_NAME"]}_PARTIAL={config["PARTIAL"]}_SEED={config["SEED"]}.pdf', format='pdf', dpi=300, bbox_inches='tight')
        plt.show()
    elif mode == 'grid':
        maps = [jnp.abs(m) for m in maps]
        fig, axs = plt.subplots(length, length + 1, figsize=(120, 150))
        for i in range(length):
            obs_ax = axs[i][0]
            obs_ax.imshow(obs_seq[i].squeeze(axis=0), cmap='gray')
            obs_ax.set_title(rf"$o_{{{i}}}$", fontsize=100, pad=90)
            obs_ax.axis('off')
            for j in range(length):
                # print(maps[i].shape)
                if j < maps[i].shape[0]:
                    map_ax = axs[i][j+1]
                    im = map_ax.imshow(maps[i][j].squeeze(axis=0).mean(axis=-1), cmap=cmap)
                    map_ax.set_title(
                        rf"$\sum\limits_{{a \in A}}\left|\frac{{\partial Q(\hat{{s}}_{{{length - 1}}}, a_{{{length - 1}}})}}{{\partial o_{{{j}}}}}\right|$",
                        fontsize=100, pad=110)
                    map_ax.axis('off')
                else:
                    map_ax = axs[i][j+1]
                    map_ax.axis('off')

        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical', label="Gradient Magnitude")
        formatter = ticker.FormatStrFormatter('%.4f')
        cbar.ax.yaxis.set_major_formatter(formatter)
        cbar.ax.tick_params(axis='y', labelsize=80)
        cbar.set_label("Gradient Magnitude", fontsize=100)
        plt.subplots_adjust(hspace=0.1, right=0.9)  # Adjust the main plot to the left to make room for the colorbar
        plt.savefig(f'{mode}_{config["ENV_NAME"]}_PARTIAL={config["PARTIAL"]}_SEED={config["SEED"]}.pdf', format='pdf', dpi=300, bbox_inches='tight')
        plt.show()






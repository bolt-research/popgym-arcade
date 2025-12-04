# Require pip install moviepy==1.0.3
from typing import NamedTuple
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import chex

import optax
import equinox as eqx
import popgym_arcade
from popgym_arcade.wrappers import LogWrapper
from memax.equinox.train_utils import add_batch_dim
from popgym_arcade.baselines.model import QNetworkRNN
import wandb
from popgym_arcade.baselines.pqn_rnn import debug_shape

import os
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# This is the number of steps or frames to evaluate
STEPS = 101

def evaluate(model, config, sweep_seed):
    seed = jax.random.PRNGKey(sweep_seed)
    seed, _rng = jax.random.split(seed)
    env, env_params = popgym_arcade.make(config["ENV_NAME"], partial_obs=config["PARTIAL"], obs_size=config["OBS_SIZE"])
    vmap_reset = lambda n_envs: lambda rng: jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(rng, n_envs), env_params
    )
    vmap_step = lambda n_envs: lambda rng, env_state, action: jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )(jax.random.split(rng, n_envs), env_state, action, env_params)

    def run_evaluation(rng):
        # Reset environment
        obs, state = vmap_reset(2)(rng)
        init_done = jnp.zeros(2, dtype=bool)
        init_action = jnp.zeros(2, dtype=int)
        init_hs = model.initialize_carry(key=rng)
        hs = add_batch_dim(init_hs, 2)


        frame_shape = obs[0].shape
        frames = jnp.zeros((STEPS, *frame_shape), dtype=jnp.float32)
        # Store initial observation
        frame = jnp.asarray(obs[0]) / 255.0
        print(f"obs dtype: {frame.dtype} min: {frame.min()} max: {frame.max()}")
        frames = frames.at[0].set(frame)
        normal_qvals = jnp.zeros((STEPS, 2, 5))
        carry = (hs, obs, init_done, init_action, state, frames, rng)

        def evaluate_step(carry, i):
            hs, obs, done, action, state, frames, _rng = carry
            _rng, rng_step = jax.random.split(_rng, 2)

            obs_batch = obs[jnp.newaxis, :]
            done_batch = done[jnp.newaxis, :]
            action_batch = action[jnp.newaxis, :]
            # jax.debug.print("hs shape: {}", debug_shape(hs)) # tuple (2, 512) (2,)
            # jax.debug.print("obs_batch shape: {}", obs_batch.shape) # Shape (1, 2, 128, 128, 3)
            # jax.debug.print("done_batch shape: {}", done_batch.shape) # Shape (1, 2)
            # jax.debug.print("action_batch shape: {}", action_batch.shape) # Shape (1, 2)
            hs, q_val = model(hs, obs_batch, done_batch, action_batch)
            q_val = lax.stop_gradient(q_val)
            q_val = q_val.squeeze(axis=0)
            # jax.debug.print("q_val shape: {}", q_val.shape) # Shape (2, n_actions)

            action = jnp.argmax(q_val, axis=-1)

            obs, new_state, reward, done, info = vmap_step(2)(rng_step, state, action)
            state = new_state

            frame = jnp.asarray(obs[0]) / 255.0
            frame = (frame).astype(jnp.float32)
            frames = frames.at[i + 1].set(frame)

            carry = (hs, obs, done, action, state, frames, _rng)
            return carry, q_val

        def body_fun(i, val):
            carry, normal_qvals = val
            carry, q_val = evaluate_step(carry, i)
            normal_qvals = normal_qvals.at[i].set(q_val)

            return (carry, normal_qvals)

        carry, normal_qvals = lax.fori_loop(0, STEPS, body_fun, (carry, normal_qvals))
        _, _, _, _, _, frames, _rng = carry
        return frames, _rng, normal_qvals
    # imageio.mimsave('{}_{}_{}_Partial={}_SEED={}.gif'.format(config["TRAIN_TYPE"], config["MEMORY_TYPE"], config["ENV_NAME"], config["PARTIAL"], config["SEED"]), frames)
    # wandb.log({"{}_{}_{}_model_Partial={}_SEED={}".format(config["TRAIN_TYPE"], config["MEMORY_TYPE"], config["ENV_NAME"], config["PARTIAL"], config["SEED"]): wandb.Video(frames, fps=4)})
    wandb.init(project=f'{config["PROJECT"]}', name=f'{config["TRAIN_TYPE"]}_{config["MEMORY_TYPE"]}_{config["ENV_NAME"]}_Partial={config["PARTIAL"]}_SEED={config["SEED"]}')

    # Rollout - Split RNG to keep baseline evaluation independent
    _rng, baseline_rng, noise_rng = jax.random.split(_rng, 3)
    noiseless_frames, baseline_rng, normal_qvals = run_evaluation(baseline_rng) # Shape (STEPS, 128, 128, 3), normal_qvals shape (STEPS, 2, n_actions)
    # check range of noiseless_frames
    # print(f"noiseless_frames: min: {noiseless_frames.min()} max: {noiseless_frames.max()}")
    def add_noise(o, _rng):
        noise = jax.random.normal(_rng, o.shape) * 1.0
        return o + noise

    def _swap_frames(frames: jnp.ndarray, i: int, j: int) -> jnp.ndarray:
        fi, fj = frames[i], frames[j]
        swapped = frames.at[i].set(fj)
        swapped = swapped.at[j].set(fi)
        return swapped

    def sweep_consecutive_swaps(frames: jnp.ndarray, rng):
        """
        Generate trajectories by swapping consecutive frames across the rollout:
        (1,2), then (2,3), ..., up to (STEPS-2, STEPS-1).

        Returns:
        - pairs: list of (i, j) swapped indices
        - last_qs: list of np.ndarray last-step Q-values (n_actions,) for each swapped trajectory
        - swap_images: list of tuples (frame_at_j_after_swap, frame_at_i_after_swap) as np.ndarray for plotting
        - rng: updated rng
        """
        pairs = []
        last_qs = []
        swap_images = []
        init_hs = model.initialize_carry(key=rng)
        for i in range(1, STEPS - 1):
            rng, sub_rng = jax.random.split(rng)
            swapped_frames = _swap_frames(frames, i, i + 1)
            q_vals = qvals_for_frames(swapped_frames, sub_rng, init_hs)  # (STEPS, 2, n_actions)
            q_vals_np = np.array(q_vals)
            # last_qs.append(q_vals_np[-1, 0, :])
            q_vals_plot = q_vals_np[:, 0, :]  # shape (STEPS, 5)
            last_qs.append(q_vals_plot[-1])
            
            pairs.append((i, i + 1))
            # store both swapped locations for visualization: (j, i)
            img_j = np.array(swapped_frames[i], dtype=np.float32)
            img_i = np.array(swapped_frames[i+1], dtype=np.float32)
            swap_images.append((img_j, img_i))
        return pairs, last_qs, swap_images, rng

    def sweep_custom_swap(frames, x, y, rng, reverse_range: bool = False, shuffle_range: bool = False):
        if not (0 <= x < STEPS and 0 <= y < STEPS):
            raise ValueError(f"Swap indices out of range: x={x}, y={y}, STEPS={STEPS}")
        if reverse_range:
            if not (0 <= x < y < STEPS):
                raise ValueError(f"Reverse range requires 0 <= x < y < STEPS, got x={x}, y={y}, STEPS={STEPS}")
            prefix = frames[: x + 1]
            mid = jnp.flip(frames[x + 1 : y + 1], axis=0)
            suffix = frames[y + 1 :]
            swapped_frames = jnp.concatenate([prefix, mid, suffix], axis=0)

        elif shuffle_range:
            if not (0 <= x < y < STEPS):
                raise ValueError(f"Shuffle range requires 0 <= x < y < STEPS, got x={x}, y={y}, STEPS={STEPS}")
            prefix = frames[: x + 1]
            mid = frames[x + 1 : y + 1]
            suffix = frames[y + 1 :]

            rng, perm_rng = jax.random.split(rng)
            perm = jax.random.permutation(perm_rng, mid.shape[0])  # permutation over [x+1..y]
            mid_shuffled = mid[perm]

            swapped_frames = jnp.concatenate([prefix, mid_shuffled, suffix], axis=0)
            # print("*"*60)
            # print("*"*60)
            # print("*"*60)
        else:
            swapped_frames = _swap_frames(frames, x, y)
            
        rng, rng_init = jax.random.split(rng)
        init_hs = model.initialize_carry(key=rng_init)
        rng, sub_rng = jax.random.split(rng)
        q_vals = qvals_for_frames(swapped_frames, sub_rng, init_hs)
        last_q = np.array(q_vals)[-1, 0, :]
        # print(f"Check swapped frames shape: {swapped_frames.shape}")
        return (x, y), last_q, swapped_frames, rng

    def qvals_for_frames(frames, rng, init_hs):
            hs = add_batch_dim(init_hs, 2)
            init_done = jnp.zeros(2, dtype=bool)
            init_action = jnp.zeros(2, dtype=int)
            q_vals = jnp.zeros((STEPS, 2, 5))

            def process_step(carry, frame):
                hs, done, action = carry
                obs = jnp.asarray(frame)
                # jax.debug.print("frame shape: {}", obs.shape)  # Shape (128, 128, 3)
                obs = jnp.stack([obs, obs], axis=0)  # Simulate 2 environments

                obs_batch = obs[jnp.newaxis, :]  # Shape (1, 2, 128, 128, 3)
                done_batch = done[jnp.newaxis, :]  # Shape (1, 2)
                action_batch = action[jnp.newaxis, :]  # Shape (1, 2)

                hs, q_val = model(hs, obs_batch, done_batch, action_batch)
                q_val = lax.stop_gradient(q_val)
                q_val = q_val.squeeze(axis=0)  # Shape (2, n_actions)
                # jax.debug.print("=q_val shape: {}", q_val.shape)  # Shape (2, n_actions)
                
                # Update action based on Q-values for next step
                new_action = jnp.argmax(q_val, axis=-1)
                carry = (hs, done, new_action)
                return carry, q_val

            def body_fun(i, val):
                carry, q_vals = val
                carry, q_val = process_step(carry, frames[i])
                q_vals = q_vals.at[i].set(q_val)
                return (carry, q_vals)

            carry = (hs, init_done, init_action)
            # plt.imshow(frames[-1])
            # plt.show()
            _, q_vals = lax.fori_loop(0, STEPS, body_fun, (carry, q_vals))
            return q_vals  # Shape (STEPS, 2, n_actions)


    last_qs = []
    noisy_frames = []
    num_noise = STEPS - 1
    final_frames = []

    for noise_idx in range(1, num_noise + 1): # This is how many trajectories we want to generate
        noise_rng, sub_rng_noise, sub_rng_qvals, init_rng = jax.random.split(noise_rng, 4)
        init_hs = model.initialize_carry(key=init_rng)
        

        injected_frames = jnp.array(noiseless_frames)

        noisy_frame = add_noise(injected_frames[noise_idx], sub_rng_noise)
        injected_frames = injected_frames.at[noise_idx].set(noisy_frame) # Shape (STEPS, 128, 128, 3)

        noisy_frames.append(noisy_frame)
        final_frames.append(injected_frames)

        q_vals = qvals_for_frames(injected_frames, sub_rng_qvals, init_hs) 
        q_vals = jnp.array(q_vals)
        # print(f"{noise_idx}{q_vals}")  # Shape (STEPS, 2, n_actions)
        q_vals_np = np.array(q_vals)  # shape (STEPS, 2, 5)
        q_vals_plot = q_vals_np[:, 0, :]  # shape (STEPS, 5)
        last_qs.append(q_vals_plot[-1])

        # frames = np.arange(q_vals_plot.shape[0])
        n_actions = q_vals_plot.shape[1]
    

    # swap_pairs, swap_last_qs, swap_images, _rng = sweep_consecutive_swaps(noiseless_frames, _rng)


    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{amsfonts}'
    # plt.rcParams['text.latex.preamble'] = r'\usepackage{sfmath} \boldmath'

    def plot(noisy_frames, noiseless_frames, final_frames, normal_qvals, last_qs):
        sns.set()
        # fig, axes = plt.subplots(num_noise, 4, figsize=(10, 3 * num_noise))
        # fig, axes = plt.subplots(5, 4, figsize=(4, 3 * num_noise))
        width_ratios = [1, 1, 1, 1] 
        
        # Limit to first 5 rows regardless of STEPS value
        max_plot_rows = 5
        plot_rows = min(max_plot_rows, num_noise)
    
        # Create the subplots using gridspec_kw to set the column width ratios.
        fig, axes = plt.subplots(
            plot_rows, 4,
            figsize=(10, 3 * plot_rows),
            gridspec_kw={'width_ratios': width_ratios}
        )

        normal_last_qvals = normal_qvals[:, 0, :][-1] 
        last_qs = np.array(last_qs) # shape (10, 5)
        all_qvals = [normal_last_qvals] + [last_qs[idx] for idx in range(num_noise)]
        ymin = min(q.min() for q in all_qvals)
        ymax = max(q.max() for q in all_qvals)

        y_range = ymax - ymin
        ymin = ymin - 0.1 * y_range
        ymax = ymax + 0.1 * y_range
        
        action_symbols = ['↑', '↓', '←', '→', 'x']

        differences = last_qs - normal_last_qvals[np.newaxis, :]
        max_abs_diff = np.max(np.abs(differences)) if len(differences) > 0 else 1e-6
        
        axes[0, 0].imshow(noiseless_frames[0])
        axes[0, 0].set_title(r'$s_0$', fontsize=20)
        axes[0, 1].imshow(noiseless_frames[1])
        axes[0, 1].set_title(r'$s_1$', fontsize=20)
        axes[0, 2].imshow(noiseless_frames[-1])
        axes[0, 2].set_title(rf'$s_{{{num_noise}}}$', fontsize=20)
        max_idx = np.argmax(normal_last_qvals)
        colors = ["#b3b3b3"] * len(normal_last_qvals)
        colors[max_idx] = "#354e6a"

        axes[0, 3].bar(
            np.arange(normal_last_qvals.shape[0]),
            normal_last_qvals - jnp.mean(normal_last_qvals, axis=0, keepdims=True),
            color=colors,
            edgecolor='black',
        )
        # Advantage = Q(s^hat, a) - 1/|A| sum_a Q(s^hat, a)
        # Q(s^{{\widehat{{}}}}_{{{num_noise}}}, a_{{{num_noise}}})
        axes[0, 3].set_title(rf'$A(s^{{\hat{{}}}}_{{{num_noise}}}, a_{{{num_noise}}})$', fontsize=20)
        axes[0, 3].set_xticks(np.arange(len(normal_last_qvals)))


        axes[0, 3].set_xticklabels(action_symbols, fontsize=18)
        axes[0, 3].set_yticks([])


        for idx in range(plot_rows - 1):

            qvals = last_qs[idx]
            max_idx = np.argmax(qvals)

            colors = ["#b3b3b3"] * len(qvals)
            colors[max_idx] = "#a33a2f"
            
            axes[idx + 1, 0].imshow(final_frames[idx][0])
            axes[idx + 1, 0].set_title(r'$s_0$', fontsize=20)
            axes[idx + 1, 1].imshow(final_frames[idx][1+idx])
            axes[idx + 1, 1].set_title(rf'$s_{{{idx+1}}} + \epsilon$', fontsize=20)
            axes[idx + 1, 2].imshow(final_frames[idx][-1])
            axes[idx + 1, 2].set_title(rf'$s_{{{num_noise}}}$', fontsize=20)

            axes[idx + 1, 3].bar(
                np.arange(last_qs[idx].shape[0]),
                last_qs[idx] - jnp.mean(last_qs[idx], axis=0, keepdims=True),
                color=colors,
                edgecolor='black'
            )
            # 
            axes[idx + 1, 3].set_title(rf'$A(s^{{\hat{{}}}}_{{{num_noise}}}, a_{{{num_noise}}})$', fontsize=20)
            axes[idx + 1, 3].tick_params(axis='y', labelsize=20)
            axes[idx + 1, 3].set_yticks([])

            axes[idx + 1, 3].set_xticks(np.arange(len(last_qs[idx])))

            axes[idx + 1, 3].set_xticklabels(action_symbols, fontsize=18)

        

        for row in axes:
            for ax in row[:3]:
                ax.axis('off')

        plt.tight_layout()
        
        plt.savefig(f"new_pkls_test/{config['ENV_NAME']}_seed={sweep_seed}.pdf", dpi=300, bbox_inches='tight')
        plt.close()

    def batch_plot(noisy_frames, noiseless_frames, normal_qvals, last_qs):
        sns.set()
        plt.rcParams['text.usetex'] = True
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{amsfonts}'

        BATCH_SIZE = 20  # Number of rows per batch
        total_rows = num_noise + 1
        num_batches = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE

        normal_last_qvals = normal_qvals[:, 0, :][-1]
        last_qs = np.array(last_qs)  # shape (num_noise, 5)
        all_qvals = [normal_last_qvals] + [last_qs[idx] for idx in range(num_noise)]
        ymin = min(q.min() for q in all_qvals)
        ymax = max(q.max() for q in all_qvals) + 0.1
        action_symbols = ['↑', '↓', '←', '→', '4']

        for batch_idx in range(num_batches):
            start = batch_idx * BATCH_SIZE
            end = min((batch_idx + 1) * BATCH_SIZE, total_rows)
            batch_rows = end - start

            fig, axes = plt.subplots(batch_rows, 4, figsize=(10, 3 * batch_rows))
            if batch_rows == 1:
                axes = axes[None, :]  # Ensure axes is 2D

            for row_idx in range(batch_rows):
                idx = start + row_idx
                if idx == 0:
                    # first row
                    axes[row_idx, 0].imshow(np.array(noiseless_frames[0], dtype=np.float32))
                    axes[row_idx, 0].set_title(r'$O_0$', fontsize=20)
                    axes[row_idx, 1].imshow(np.array(noiseless_frames[1], dtype=np.float32))
                    axes[row_idx, 1].set_title(r'$O_1$', fontsize=20)
                    axes[row_idx, 2].imshow(np.array(noiseless_frames[-1], dtype=np.float32))
                    axes[row_idx, 2].set_title(rf'$O_{{{num_noise}}}$', fontsize=20)
                    max_idx = np.argmax(normal_last_qvals)
                    colors = ["#BBBBBB"] * len(normal_last_qvals)
                    colors[max_idx] = "lightblue"
                    axes[row_idx, 3].bar(
                        np.arange(normal_last_qvals.shape[0]),
                        normal_last_qvals,
                        color=colors,
                        edgecolor='black',
                    )
                    axes[row_idx, 3].set_title(rf'$Q(s_{{{num_noise}}}, a_{{{num_noise}}})$', fontsize=20)
                    axes[row_idx, 3].set_xticks(np.arange(len(normal_last_qvals)))
                    axes[row_idx, 3].set_xticklabels(action_symbols, fontsize=10)
                    axes[row_idx, 3].set_yticks([])
                    axes[row_idx, 3].yaxis.set_visible(False)
                else:
                    qvals = last_qs[idx - 1]
                    max_idx = np.argmax(qvals)
                    colors = ["#BBBBBB"] * len(qvals)
                    colors[max_idx] = "#FFB6C1"
                    axes[row_idx, 0].imshow(np.array(noiseless_frames[0], dtype=np.float32))
                    axes[row_idx, 0].set_title(r'$O_0$', fontsize=20)
                    axes[row_idx, 1].imshow(np.array(noisy_frames[idx - 1], dtype=np.float32))
                    axes[row_idx, 1].set_title(rf'$O_{{{idx}}} + \epsilon$', fontsize=20)
                    axes[row_idx, 2].imshow(np.array(noiseless_frames[-1], dtype=np.float32))
                    axes[row_idx, 2].set_title(rf'$O_{{{num_noise}}}$', fontsize=20)
                    axes[row_idx, 3].bar(
                        np.arange(qvals.shape[0]),
                        qvals,
                        color=colors,
                        edgecolor='black'
                    )
                    axes[row_idx, 3].set_title(rf'$Q(s_{{{num_noise}}}, a_{{{num_noise}}})$', fontsize=20)
                    axes[row_idx, 3].set_yticks([])
                    axes[row_idx, 3].yaxis.set_visible(False)
                    axes[row_idx, 3].set_xticks(np.arange(len(qvals)))
                    axes[row_idx, 3].set_xticklabels(action_symbols, fontsize=10)

                for ax in axes[row_idx, :3]:
                    ax.axis('off')

            plt.tight_layout()
            plt.subplots_adjust(wspace=0.1)

            for row in axes:
                ax2 = row[2]
                ax3 = row[3]
                pos2 = ax2.get_position()
                pos3 = ax3.get_position()
                new_spacing = 0.04
                new_ax3_x0 = pos2.x1 + new_spacing
                ax3.set_position([new_ax3_x0, pos3.y0, pos3.width, pos3.height])

            plt.savefig(f"new_pkls_test/batch_{config['ENV_NAME']}_{batch_idx}.png", dpi=300, bbox_inches='tight')
            plt.close()

    def batch_plot_swaps(noiseless_frames, normal_qvals, swap_pairs, swap_last_qs, swap_images, title_prefix: str = "swaps"):
        """
        Plot swap trajectories in batches. Each row shows 5 panels:
        [O_0, swapped O_j, swapped O_i, O_T, Q-values]. The first row is a baseline
        with [O_0, O_1, O_2, O_T, baseline Q-values].
        """
        sns.set()
        plt.rcParams['text.usetex'] = True
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{amsfonts}'

        BATCH_SIZE = 20
        total_rows = len(swap_pairs) + 1
        num_batches = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE

        normal_last_qvals = normal_qvals[:, 0, :][-1]
        action_symbols = ['↑', '↓', '←', '→', '4']

        for batch_idx in range(num_batches):
            start = batch_idx * BATCH_SIZE
            end = min((batch_idx + 1) * BATCH_SIZE, total_rows)
            batch_rows = end - start

            fig, axes = plt.subplots(batch_rows, 5, figsize=(12, 3 * batch_rows))
            if batch_rows == 1:
                axes = axes[None, :]

            for row_idx in range(batch_rows):
                idx = start + row_idx
                if idx == 0:
                    # Baseline row
                    axes[row_idx, 0].imshow(np.array(noiseless_frames[0], dtype=np.float32))
                    axes[row_idx, 0].set_title(r'$O_0$', fontsize=16)
                    axes[row_idx, 1].imshow(np.array(noiseless_frames[1], dtype=np.float32))
                    axes[row_idx, 1].set_title(r'$O_1$', fontsize=16)
                    axes[row_idx, 2].imshow(np.array(noiseless_frames[2], dtype=np.float32))
                    axes[row_idx, 2].set_title(r'$O_2$', fontsize=16)
                    axes[row_idx, 3].imshow(np.array(noiseless_frames[-1], dtype=np.float32))
                    axes[row_idx, 3].set_title(rf'$O_{{{STEPS-1}}}$', fontsize=16)

                    max_idx = int(np.argmax(np.array(normal_last_qvals)))
                    colors = ["#BBBBBB"] * len(normal_last_qvals)
                    colors[max_idx] = "lightblue"
                    axes[row_idx, 4].bar(np.arange(normal_last_qvals.shape[0]), normal_last_qvals, color=colors, edgecolor='black')
                    axes[row_idx, 4].set_title(r'$Q(s_T, a)$', fontsize=16)
                    axes[row_idx, 4].set_xticks(np.arange(len(normal_last_qvals)))
                    axes[row_idx, 4].set_xticklabels(action_symbols, fontsize=10)
                    axes[row_idx, 4].set_yticks([])
                    axes[row_idx, 4].yaxis.set_visible(False)
                else:
                    pair = swap_pairs[idx - 1]
                    (img_j, img_i) = swap_images[idx - 1]
                    axes[row_idx, 0].imshow(np.array(noiseless_frames[0], dtype=np.float32))
                    axes[row_idx, 0].set_title(r'$O_0$', fontsize=16)
                    axes[row_idx, 1].imshow(img_j)
                    axes[row_idx, 1].set_title(rf'$O_{{{pair[1]}}}$', fontsize=16)
                    axes[row_idx, 2].imshow(img_i)
                    axes[row_idx, 2].set_title(rf'$O_{{{pair[0]}}}$', fontsize=16)
                    axes[row_idx, 3].imshow(np.array(noiseless_frames[-1], dtype=np.float32))
                    axes[row_idx, 3].set_title(rf'$O_{{{STEPS-1}}}$', fontsize=16)

                    qvals = np.array(swap_last_qs[idx - 1])
                    max_idx = int(np.argmax(qvals))
                    colors = ["#BBBBBB"] * len(qvals)
                    colors[max_idx] = "#FFB6C1"
                    axes[row_idx, 4].bar(np.arange(qvals.shape[0]), qvals, color=colors, edgecolor='black')
                    axes[row_idx, 4].set_title(r'$Q(s_T, a)$', fontsize=16)
                    axes[row_idx, 4].set_yticks([])
                    axes[row_idx, 4].yaxis.set_visible(False)
                    axes[row_idx, 4].set_xticks(np.arange(len(qvals)))
                    axes[row_idx, 4].set_xticklabels(action_symbols, fontsize=10)

                for ax in axes[row_idx, :4]:
                    ax.axis('off')

            plt.tight_layout()
            plt.subplots_adjust(wspace=0.1)

            # Adjust spacing between last image column and bar chart if needed
            for row in axes:
                ax3 = row[3]
                ax4 = row[4]
                pos3 = ax3.get_position()
                pos4 = ax4.get_position()
                new_spacing = 0.04
                new_ax4_x0 = pos3.x1 + new_spacing
                ax4.set_position([new_ax4_x0, pos4.y0, pos4.width, pos4.height])

            plt.savefig(f"summary_{title_prefix}_batch_{batch_idx}.png", dpi=300, bbox_inches='tight')
            plt.close()

    def batch_plot_custom_swap(noiseless_frames, normal_qvals, pair, last_q, swapped_frames, title_prefix: str = "custom_swap", reverse_range: bool = False, shuffle_range: bool = False):
        """
        Plot exactly two rows: baseline and a single custom swap (i, j).
        Columns: [O_0, swapped O_j, swapped O_i, O_T, Q-values].
        """
        sns.set()
        plt.rcParams['text.usetex'] = True
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{amsfonts}'

        action_symbols = ['↑', '↓', '←', '→', '4']
        normal_last_qvals = normal_qvals[:, 0, :][-1]

        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        # Swap row (row 1)
        i, j = pair
        print("="*60)
        print(f"Swapping O_{i} and O_{j}")
        print("="*60)

        # Determine where original O_j and O_i contents reside after transformation
        if reverse_range or shuffle_range:
            # frames[x] stays; block (x+1..y) reversed => original O_j moves to x+1; original O_i stays at x
            idx_oj_after = i + 1
            idx_oi_after = i
        else:
            # simple swap: O_j -> index i, O_i -> index j
            idx_oj_after = i
            idx_oi_after = j
        # Baseline row (row 0)
        axes[0, 0].imshow(np.array(noiseless_frames[0], dtype=np.float32))
        axes[0, 0].set_title(r'$O_0$', fontsize=16)
        axes[0, 1].imshow(np.array(noiseless_frames[i], dtype=np.float32))
        axes[0, 1].set_title(rf'$O_{{{i}}}$', fontsize=16)
        axes[0, 2].imshow(np.array(noiseless_frames[j], dtype=np.float32))
        axes[0, 2].set_title(rf'$O_{{{j}}}$', fontsize=16)
        axes[0, 3].imshow(np.array(noiseless_frames[-1], dtype=np.float32))
        axes[0, 3].set_title(rf'$O_{{{STEPS-1}}}$', fontsize=16)

        base_max = int(np.argmax(np.array(normal_last_qvals)))
        base_colors = ["#BBBBBB"] * len(normal_last_qvals)
        base_colors[base_max] = "lightblue"
        axes[0, 4].bar(np.arange(normal_last_qvals.shape[0]), normal_last_qvals, color=base_colors, edgecolor='black')
        axes[0, 4].set_title(r'$Q(s_T, a)$', fontsize=16)
        axes[0, 4].set_xticks(np.arange(len(normal_last_qvals)))
        axes[0, 4].set_xticklabels(action_symbols, fontsize=10)
        axes[0, 4].set_yticks([])
        axes[0, 4].yaxis.set_visible(False)



        axes[1, 0].imshow(np.array(noiseless_frames[0], dtype=np.float32))
        axes[1, 0].set_title(r'$O_0$', fontsize=16)
        axes[1, 1].imshow(np.array(swapped_frames[idx_oj_after], dtype=np.float32))
        axes[1, 1].set_title(rf'$O_{{{j}}}$', fontsize=16)
        axes[1, 2].imshow(np.array(swapped_frames[idx_oi_after], dtype=np.float32))
        axes[1, 2].set_title(rf'$O_{{{i}}}$', fontsize=16)
        axes[1, 3].imshow(np.array(noiseless_frames[-1], dtype=np.float32))
        axes[1, 3].set_title(rf'$O_{{{STEPS-1}}}$', fontsize=16)

        qvals = np.array(last_q)
        swap_max = int(np.argmax(qvals))
        swap_colors = ["#BBBBBB"] * len(qvals)
        swap_colors[swap_max] = "#FFB6C1"
        axes[1, 4].bar(np.arange(qvals.shape[0]), qvals, color=swap_colors, edgecolor='black')
        axes[1, 4].set_title(r'$Q(s_T, a)$', fontsize=16)
        axes[1, 4].set_yticks([])
        axes[1, 4].yaxis.set_visible(False)
        axes[1, 4].set_xticks(np.arange(len(qvals)))
        axes[1, 4].set_xticklabels(action_symbols, fontsize=10)

        # # Hide axes for image panels like in batch_plot
        # for r in range(2):
        #     for ax in axes[r, :4]:
        #         ax.axis('off')

        # plt.tight_layout()
        plt.subplots_adjust(wspace=0.1, hspace=0.6, bottom=0.28)

    # # Adjust spacing between last image and bar chart similar to batch_plot
    # (moved underbrace texts to after positions are computed below)
        #     pos4 = ax4.get_position()
        #     new_spacing = 0.04
        #     new_ax4_x0 = pos3.x1 + new_spacing
        #     ax4.set_position([new_ax4_x0, pos4.y0, pos4.width, pos4.height])

        # plt.savefig(f"{ENV_NAMES}_{title_prefix}_i{i}_j{j}.png", dpi=300, bbox_inches='tight')
        # plt.close()
        # axes[1, 4].set_xticklabels(action_symbols, fontsize=10)

        for r in range(2):
            for ax in axes[r, :4]:
                ax.axis('off')

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.1)

        # Adjust spacing between last image and bar chart
        for row in axes:
            ax3 = row[3]
            ax4 = row[4]
            pos3 = ax3.get_position()
            pos4 = ax4.get_position()
            new_spacing = 0.04
            new_ax4_x0 = pos3.x1 + new_spacing
            ax4.set_position([new_ax4_x0, pos4.y0, pos4.width, pos4.height])

        plt.savefig(f"{ENV_NAMES}_{title_prefix}_i{i}_j{j}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def batch_plot_custom_swap_braced(noiseless_frames, normal_qvals, pair, last_q, swapped_frames, title_prefix: str = "custom_swap_braced", reverse_range: bool = False, shuffle_range: bool = False):
        """
        Two-row plot with a baseline layout on row 0 and a shuffled layout on row 1.
        Row 0: [O_0, ..., O_10, O_11, ..., O_T, Q-values] with two upward underbraces.
        Row 1: [O_0, concat(O_1..O_9), O_10, O_11, (blank), O_T, Q-values] with two upward underbraces.
        """
        sns.set()
        plt.rcParams['text.usetex'] = True
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{amsfonts}'

        action_symbols = ['↑', '↓', '←', '→', '4']
        normal_last_qvals = normal_qvals[:, 0, :][-1]

        i, j = pair

        fig, axes = plt.subplots(2, 7, figsize=(12, 6))

        # Row 0: baseline
        axes[0, 0].imshow(np.array(noiseless_frames[0], dtype=np.float32))
        axes[0, 0].set_title(r'$O_0$', fontsize=16)
        axes[0, 1].axis('off')
        axes[0, 1].text(0.5, 0.5, '...', ha='center', va='center', fontsize=20, transform=axes[0, 1].transAxes)

        # idx10 = 10 if len(noiseless_frames) > 10 else max(0, len(noiseless_frames) - 1)
        # idx11 = 11 if len(noiseless_frames) > 11 else max(0, len(noiseless_frames) - 1)
        idx10, idx11 = j, j + 1

        axes[0, 2].imshow(np.array(noiseless_frames[idx10], dtype=np.float32))
        axes[0, 2].set_title(r'$O_{10}$', fontsize=16)
        axes[0, 3].imshow(np.array(noiseless_frames[idx11], dtype=np.float32))
        axes[0, 3].set_title(r'$O_{11}$', fontsize=16)
        axes[0, 4].axis('off')
        axes[0, 4].text(0.5, 0.5, '...', ha='center', va='center', fontsize=20, transform=axes[0, 4].transAxes)
        axes[0, 5].imshow(np.array(noiseless_frames[-1], dtype=np.float32))
        axes[0, 5].set_title(rf'$O_{{{STEPS-1}}}$', fontsize=16)

        base_max = int(np.argmax(np.array(normal_last_qvals)))
        base_colors = ["#BBBBBB"] * len(normal_last_qvals)
        base_colors[base_max] = "lightblue"
        axes[0, 6].bar(np.arange(normal_last_qvals.shape[0]), normal_last_qvals, color=base_colors, edgecolor='black')
        axes[0, 6].set_title(rf'$Q(s_{{{STEPS-1}}}, a)$', fontsize=16)
        axes[0, 6].set_xticks(np.arange(len(normal_last_qvals)))
        axes[0, 6].set_xticklabels(action_symbols, fontsize=10)
        axes[0, 6].set_yticks([])
        axes[0, 6].yaxis.set_visible(False)

        for c in [0, 2, 3, 5]:
            axes[0, c].axis('off')

        # Row 1: shuffled segment
        axes[1, 0].imshow(np.array(swapped_frames[0], dtype=np.float32))
        axes[1, 0].set_title(r'$O_0$', fontsize=16)
        axes[1, 0].axis('off')

        axes[1, 1].axis('off')
        # img = plt.imread(os.path.join("./output_images/", 'skittles1.png'))

        # img = np.array(img, dtype=np.float32)
        import cv2
        original_image = np.array(swapped_frames[1], dtype=np.float32)
        print(f"Original image shape: {original_image.shape}")
        new_width = original_image.shape[1] // 2
        new_height = original_image.shape[0] // 2
        new_dimensions = (new_width, new_height)
        resized_image = cv2.resize(original_image, new_dimensions, interpolation=cv2.INTER_AREA)

        axes[1, 1].imshow(resized_image)

        axes[1, 2].imshow(np.array(swapped_frames[idx10], dtype=np.float32))
        axes[1, 2].set_title(rf'$O_{{{idx10}}}$', fontsize=16)
        axes[1, 2].axis('off')

        axes[1, 3].imshow(np.array(swapped_frames[idx11], dtype=np.float32))
        axes[1, 3].set_title(rf'$O_{{{idx11}}}$', fontsize=16)
        axes[1, 3].axis('off')
        axes[1, 4].axis('off')
        axes[1, 4].text(0.5, 0.5, '...', ha='center', va='center', fontsize=20, transform=axes[1, 4].transAxes)
        axes[1, 5].imshow(np.array(swapped_frames[-1], dtype=np.float32))
        axes[1, 5].set_title(rf'$O_{{{STEPS-1}}}$', fontsize=16)
        axes[1, 5].axis('off')

        qvals = np.array(last_q)
        swap_max = int(np.argmax(qvals))
        swap_colors = ["#BBBBBB"] * len(qvals)
        swap_colors[swap_max] = "#FFB6C1"
        axes[1, 6].bar(np.arange(qvals.shape[0]), qvals, color=swap_colors, edgecolor='black')
        axes[1, 6].set_title(rf'$Q(s_{{{STEPS - 1}}}, a)$', fontsize=16)
        axes[1, 6].set_xticks(np.arange(len(qvals)))
        axes[1, 6].set_xticklabels(action_symbols, fontsize=10)
        axes[1, 6].set_yticks([])
        axes[1, 6].yaxis.set_visible(False)

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.1, hspace=0.7, bottom=0.32)

        # Increase spacing between last image (col 5) and bars (col 6)
        for r in range(2):
            ax_img_last = axes[r, 5]
            ax_bar = axes[r, 6]
            pos_img = ax_img_last.get_position()
            pos_bar = ax_bar.get_position()
            new_ax_bar_x0 = pos_img.x1 + 0.04
            ax_bar.set_position([new_ax_bar_x0, pos_bar.y0, pos_bar.width, pos_bar.height])

        # Upward underbraces for both rows
        pos00 = axes[0, 0].get_position(); pos02 = axes[0, 2].get_position()
        pos03 = axes[0, 3].get_position(); pos05 = axes[0, 5].get_position()
        y_brace0 = min(pos00.y0, pos02.y0, pos03.y0, pos05.y0) - 0.05
        x_center_left0 = (pos00.x0 + pos02.x1) / 2
        x_center_right0 = (pos03.x0 + pos05.x1) / 2
        fig.text(x_center_left0, y_brace0, r'$\underbrace{\phantom{MMMMMMMMMMMM}}$', ha='center', va='center', transform=fig.transFigure, fontsize=18)
        fig.text(x_center_right0, y_brace0, r'$\underbrace{\phantom{MMMMMMMMMMMM}}$', ha='center', va='center', transform=fig.transFigure, fontsize=18)

        pos10 = axes[1, 0].get_position(); pos12 = axes[1, 2].get_position()
        pos13 = axes[1, 3].get_position(); pos15 = axes[1, 5].get_position()
        y_brace1 = min(pos10.y0, pos12.y0, pos13.y0, pos15.y0) - 0.05
        x_center_group1 = (pos10.x0 + pos12.x1) / 2
        x_center_tail1 = (pos13.x0 + pos15.x1) / 2
        fig.text(x_center_group1, y_brace1, r'$\underbrace{\phantom{MMMMMMMMMMMM}}$', ha='center', va='center', transform=fig.transFigure, fontsize=18)
        fig.text(x_center_tail1, y_brace1, r'$\underbrace{\phantom{MMMMMMMMMMMM}}$', ha='center', va='center', transform=fig.transFigure, fontsize=18)

        # Add y-axis at the bottom of the whole image
        # y_axis_bottom = y_brace1 - 0.08
        # fig.text(0.05, y_axis_bottom, r'$\uparrow$', ha='center', va='center', transform=fig.transFigure, fontsize=16, rotation=0)
        # fig.text(0.02, y_axis_bottom + 0.02, r'$y$', ha='center', va='center', transform=fig.transFigure, fontsize=14)
        
        # # Add horizontal line for y-axis
        # fig.text(0.05, y_axis_bottom - 0.01, r'$\rule{0.8\textwidth}{0.5pt}$', ha='left', va='center', transform=fig.transFigure, fontsize=8)

        plt.savefig(f"new_{ENV_NAMES}_{title_prefix}_i{i}_j{j}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_diagonal_stacked_comparison(noiseless_frames, normal_qvals, pair, swapped_frames, last_q, title_prefix: str = "diagonal_comparison"):
        """
        Plot a 2x3 grid:
        Row 0: [diagonal stack of noiseless_frames[1,5,10], diagonal stack of noiseless_frames[11,25,40], noiseless Q-values]
        Row 1: [diagonal stack of swapped_frames[1,5,10], diagonal stack of swapped_frames[11,25,40], swapped Q-values]
        """
        sns.set()
        plt.rcParams['text.usetex'] = True
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{amsfonts}'

        action_symbols = ['↑', '↓', '←', '→', 'x']
        normal_last_qvals = normal_qvals[:, 0, :][-1]

        fig, axes = plt.subplots(2, 3, figsize=(10, 8))

        def create_diagonal_stack(frames_list, ax, titles):
            """Create a diagonal stack of 3 images with the last one at top"""
            ax.set_xlim(0, 50)
            ax.set_ylim(0, 50)
            ax.set_aspect('equal')
            
            # Positions for diagonal stacking from top-left to bottom-right (hides bottom parts)
            positions = [(0, 16), (10, 8), (20, 0)]  # Top-left to bottom-right direction
            sizes = [(30, 30), (30, 30), (30, 30)]  # all same size
            
            for i, (frame, pos, size, title) in enumerate(zip(frames_list, positions, sizes, titles)):
                # Create a small subplot for each image
                extent = [pos[0], pos[0] + size[0], pos[1], pos[1] + size[1]]
                ax.imshow(np.array(frame, dtype=np.float32), extent=extent)
                # Add text label - position it better relative to the stacked images
                text_x = pos[0] + size[0]/2
                text_y = pos[1] - 1  # Position text above the image
                ax.text(text_x, text_y, title, 
                       ha='center', va='top', fontsize=8, 
                       bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.9))
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')

        # Row 0: Noiseless frames
        # [0,0] - diagonal stack of frames 1, 5, 10 (last at top)
        x, y = pair
        frames_group1 = [noiseless_frames[x], noiseless_frames[int(y//2)], noiseless_frames[y]]
        # titles_group1 = [r'$O_1$', r'$O_5$', r'$O_{10}$']
        titles_group1 = [r'', r'', r'']
        create_diagonal_stack(frames_group1, axes[0, 0], titles_group1)
        axes[0, 0].set_title(rf'$s_{{{x-1}}}...s_{{{y-1}}}$', fontsize=20)

        # [0,1] - diagonal stack of frames 11, 25, 40 (last at top)
        frames_group2 = [noiseless_frames[y+1], noiseless_frames[int((STEPS - 1)//2)], noiseless_frames[STEPS - 1]]
        titles_group2 = [r'', r'', r'']
        create_diagonal_stack(frames_group2, axes[0, 1], titles_group2)
        axes[0, 1].set_title(rf'$s_{{{y}}}...s_{{{STEPS - 1}}}$', fontsize=20)

        # [0,2] - Q-values for noiseless
        base_max = int(np.argmax(np.array(normal_last_qvals)))
        base_colors = ["#b3b3b3"] * len(normal_last_qvals)
        base_colors[base_max] = "#354e6a"
        axes[0, 2].bar(np.arange(normal_last_qvals.shape[0]), normal_last_qvals - jnp.mean(normal_last_qvals, axis=0, keepdims=True), 
                      color=base_colors, edgecolor='black')
        axes[0, 2].set_title(rf'$A(s^{{\hat{{}}}}_{{{STEPS-1}}}, a_{{{STEPS-1}}})$', fontsize=20)
        axes[0, 2].set_xticks(np.arange(len(normal_last_qvals)))
        axes[0, 2].set_xticklabels(action_symbols, fontsize=18)
        axes[0, 2].set_yticks([])
        axes[0, 2].yaxis.set_visible(False)

        # Row 1: Swapped frames
        # [1,0] - diagonal stack of swapped frames 1, 5, 10 (last at top)
        swapped_group1 = [swapped_frames[x], swapped_frames[int(y//2)], swapped_frames[y]]
        create_diagonal_stack(swapped_group1, axes[1, 0], titles_group1)
        axes[1, 0].set_title(rf'Shuffled($s_{{{x-1}}}...s_{{{y-1}}}$)', fontsize=20)

        # [1,1] - diagonal stack of swapped frames 11, 25, 40 (last at top)
        swapped_group2 = [swapped_frames[y+1], swapped_frames[int((STEPS - 1)//2)], swapped_frames[STEPS - 1]]
        create_diagonal_stack(swapped_group2, axes[1, 1], titles_group2)
        axes[1, 1].set_title(rf'$s_{{{y}}}...s_{{{STEPS - 1}}}$', fontsize=20)

        # [1,2] - Q-values for swapped
        qvals = np.array(last_q)
        swap_max = int(np.argmax(qvals))
        swap_colors = ["#b3b3b3"] * len(qvals)
        swap_colors[swap_max] = "#a33a2f"
        axes[1, 2].bar(np.arange(qvals.shape[0]), qvals - jnp.mean(qvals, axis=0, keepdims=True),
                      color=swap_colors, edgecolor='black')
        axes[1, 2].set_title(rf'$A(s^{{\hat{{}}}}_{{{STEPS-1}}}, a_{{{STEPS-1}}})$', fontsize=20)
        axes[1, 2].set_xticks(np.arange(len(qvals)))
        axes[1, 2].set_xticklabels(action_symbols, fontsize=18)
        axes[1, 2].set_yticks([])
        axes[1, 2].yaxis.set_visible(False)

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.1, hspace=0.3)  # Reduced gaps between subplots
        outdir = "./output_images/"

        plt.savefig(f"{outdir}{config['ENV_NAME']}_{title_prefix}_{sweep_seed}.pdf", dpi=300, bbox_inches='tight')
        plt.close()

    # Debug: Print some Q-values to check for duplicates
    print("Debugging Q-values:")
    print(f"Normal last Q-values: {normal_qvals[:, 0, :][-1]}")
    for i in range(min(5, len(last_qs))):
        print(f"Trajectory {i+1} last Q-values: {last_qs[i]}")
    print("=" * 50)
    
    plot(noisy_frames, noiseless_frames, final_frames, normal_qvals, last_qs)
    # batch_plot(noisy_frames, noiseless_frames, normal_qvals, last_qs)
    # use
    
    # (x, y), last_q, swapped_frames, _rng = sweep_custom_swap(noiseless_frames, 1, 10, _rng, reverse_range=False, shuffle_range=True)
    # batch_plot_custom_swap_braced(noiseless_frames, normal_qvals, (x, y), last_q, swapped_frames, title_prefix="custom_swap_braced", reverse_range=False, shuffle_range=True)
    # plot_diagonal_stacked_comparison(noiseless_frames, normal_qvals, (x, y), swapped_frames, last_q, title_prefix="diagonal_comparison")


os.environ["WANDB_MODE"] = "disabled"
MEMORY_TYPES = {"lru"}
# , "mingru", "fart"
ENV_NAMES = {
    # "BattleShipEasy",
    "CartPoleEasy",
    # "MineSweeperEasy",
    # "NavigatorEasy",
}
PATH = "./pkls_gradients/"
for filename in os.listdir(PATH):
    if filename.startswith("PQN_RNN_"):
        parts = filename.split('_')
        train_type = "_".join(parts[:2])  # "PQN_RNN"
        memory_type = parts[2].lower()
        env_name = parts[3]
        partial_part = parts[5]
        seed_part = parts[6]
    else:
        continue
    
    # Extract Partial and SEED values
    partial = partial_part.split('=')[1]
    seed = seed_part.split('=')[1].replace('.pkl', '')
    # Check if this file matches our criteria
    if (train_type == "PQN_RNN" and 
        partial.lower() == "false" and 
        memory_type in MEMORY_TYPES and 
        env_name in ENV_NAMES):
        
        # Create config
        config = {
            "ENV_NAME": env_name,
            "OBS_SIZE": 128,
            "MEMORY_TYPE": memory_type,
            "PARTIAL": False,
            "TRAIN_TYPE": train_type,
            "SEED": int(seed),
            "PROJECT": ""
        }
        print(f"Evaluating {filename} with config: {config}")

        rng = jax.random.PRNGKey(config["SEED"])
        rng, _rng = jax.random.split(rng)
        network = QNetworkRNN(_rng, config["OBS_SIZE"], config["MEMORY_TYPE"])
        model = eqx.tree_deserialise_leaves(PATH + filename, network)
       
        evaluate(model, config, 0)
# Require pip install moviepy==1.0.3
import os
from typing import NamedTuple

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import seaborn as sns
from jax import lax

import popgym_arcade
import wandb
from popgym_arcade.baselines.model import QNetworkRNN, add_batch_dim
from popgym_arcade.baselines.pqn_rnn import debug_shape
from popgym_arcade.wrappers import LogWrapper

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]

# This is the number of steps or frames to evaluate
STEPS = 101


def evaluate(model, config):
    seed = jax.random.PRNGKey(11)
    seed, _rng = jax.random.split(seed)
    env, env_params = popgym_arcade.make(
        config["ENV_NAME"], partial_obs=config["PARTIAL"], obs_size=config["OBS_SIZE"]
    )
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
        frame = jnp.asarray(obs[0])
        frame = (frame * 255).astype(jnp.float32)
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

            frame = jnp.asarray(obs[0])
            frame = (frame * 255).astype(jnp.float32)
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
    wandb.init(
        project=f'{config["PROJECT"]}',
        name=f'{config["TRAIN_TYPE"]}_{config["MEMORY_TYPE"]}_{config["ENV_NAME"]}_Partial={config["PARTIAL"]}_SEED={config["SEED"]}',
    )

    # Rollout
    noiseless_frames, _rng, normal_qvals = run_evaluation(
        _rng
    )  # Shape (STEPS, 128, 128, 3), normal_qvals shape (STEPS, 2, n_actions)

    def add_noise(o, _rng):
        noise = jax.random.normal(_rng, o.shape) * 1.0
        return noise + o

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
            carry = (hs, done, action)
            return carry, q_val

        def body_fun(i, val):
            carry, q_vals = val
            carry, q_val = process_step(carry, frames[i])
            q_vals = q_vals.at[i].set(q_val)
            return (carry, q_vals)

        carry = (hs, init_done, init_action)
        _, q_vals = lax.fori_loop(0, STEPS, body_fun, (carry, q_vals))
        return q_vals  # Shape (STEPS, 2, n_actions)

    last_qs = []
    noisy_frames = []
    num_noise = STEPS - 1

    for noise_idx in range(
        1, num_noise + 1
    ):  # This is how many trajectories we want to generate
        init_hs = model.initialize_carry(key=_rng)
        rng, sub_rng = jax.random.split(_rng)

        frames = noiseless_frames.copy()

        noisy_frame = add_noise(frames[noise_idx], sub_rng)
        noisy_frame = (noisy_frame * 255).astype(jnp.float32)  # Shape (128, 128, 3)
        frames = frames.at[noise_idx].set(noisy_frame)  # Shape (STEPS, 128, 128, 3)
        noisy_frames.append(np.array(noisy_frame, dtype=np.uint8))

        frames_np = np.array(frames, dtype=np.uint8)  # Shape (STEPS, 128, 128, 3)
        frames_np = frames_np.transpose((0, 3, 1, 2))

        log_key = "{}_{}_{}_model_Partial={}_SEED={}_NoiseIdx={}".format(
            config["TRAIN_TYPE"],
            config["MEMORY_TYPE"],
            config["ENV_NAME"],
            config["PARTIAL"],
            config["SEED"],
            noise_idx,
        )
        wandb.log({log_key: wandb.Video(frames_np, fps=4)})

        q_vals = qvals_for_frames(frames, sub_rng, init_hs)
        q_vals = jnp.array(q_vals)
        # print(f"{noise_idx}{q_vals}")  # Shape (STEPS, 2, n_actions)
        q_vals_np = np.array(q_vals)  # shape (STEPS, 2, 5)
        q_vals_plot = q_vals_np[:, 0, :]  # shape (STEPS, 5)
        last_qs.append(q_vals_plot[-1])

        frames = np.arange(q_vals_plot.shape[0])
        n_actions = q_vals_plot.shape[1]

        _rng = rng

    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = (
        r"\usepackage{amsmath} \usepackage{amssymb} \usepackage{amsfonts}"
    )

    def plot(noisy_frames, noiseless_frames, normal_qvals, last_qs):
        sns.set()
        fig, axes = plt.subplots(num_noise + 1, 4, figsize=(10, 3 * (num_noise + 1)))

        normal_last_qvals = normal_qvals[:, 0, :][-1]
        last_qs = np.array(last_qs)  # shape (10, 5)
        all_qvals = [normal_last_qvals] + [last_qs[idx] for idx in range(num_noise)]
        ymin = min(q.min() for q in all_qvals)
        ymax = max(q.max() for q in all_qvals) + 0.1
        action_symbols = ["↑", "↓", "←", "→", "4"]

        axes[0, 0].imshow(np.array(noiseless_frames[0], dtype=np.uint8))
        axes[0, 0].set_title(r"$O_0$", fontsize=20)
        axes[0, 1].imshow(np.array(noiseless_frames[1], dtype=np.uint8))
        axes[0, 1].set_title(r"$O_1$", fontsize=20)
        axes[0, 2].imshow(np.array(noiseless_frames[-1], dtype=np.uint8))
        axes[0, 2].set_title(rf"$O_{{{num_noise}}}$", fontsize=20)
        max_idx = np.argmax(normal_last_qvals)
        colors = ["#BBBBBB"] * len(normal_last_qvals)
        colors[max_idx] = "lightblue"

        axes[0, 3].bar(
            np.arange(normal_last_qvals.shape[0]),
            normal_last_qvals,
            color=colors,
            edgecolor="black",
        )
        axes[0, 3].set_title(rf"$Q(s_{{{num_noise}}}, a_{{{num_noise}}})$", fontsize=20)
        axes[0, 3].set_xticks(np.arange(len(normal_last_qvals)))

        axes[0, 3].set_xticklabels(action_symbols, fontsize=10)
        # axes[0, 3].set_ylim(ymin, ymax)
        # axes[0, 3].set_yticks(np.arange(0.0, ymax + 0.05, 0.1))
        # axes[0, 3].tick_params(axis='y', labelsize=20)
        axes[0, 3].set_yticks([])
        axes[0, 3].yaxis.set_visible(False)

        for idx in range(num_noise):

            qvals = last_qs[idx]
            max_idx = np.argmax(qvals)
            colors = ["#BBBBBB"] * len(qvals)
            colors[max_idx] = "#FFB6C1"
            axes[idx + 1, 0].imshow(np.array(noiseless_frames[0], dtype=np.uint8))
            axes[idx + 1, 0].set_title(r"$O_0$", fontsize=20)
            axes[idx + 1, 1].imshow(np.array(noisy_frames[idx], dtype=np.uint8))
            axes[idx + 1, 1].set_title(rf"$O_{{{idx+1}}} + \epsilon$", fontsize=20)
            axes[idx + 1, 2].imshow(np.array(noiseless_frames[-1], dtype=np.uint8))
            axes[idx + 1, 2].set_title(rf"$O_{{{num_noise}}}$", fontsize=20)
            axes[idx + 1, 3].bar(
                np.arange(last_qs[idx].shape[0]),
                last_qs[idx],
                color=colors,
                edgecolor="black",
            )
            axes[idx + 1, 3].set_title(
                rf"$Q(s_{{{num_noise}}}, a_{{{num_noise}}})$", fontsize=20
            )
            # axes[idx , 3].set_ylim(ymin, ymax)
            # axes[idx + 1, 3].set_yticks(np.arange(0.0, ymax + 0.05, 0.1))

            # axes[idx + 1, 3].tick_params(axis='y', labelsize=20)
            axes[idx + 1, 3].set_yticks([])
            axes[idx + 1, 3].yaxis.set_visible(False)
            axes[idx + 1, 3].set_xticks(np.arange(len(last_qs[idx])))

            axes[idx + 1, 3].set_xticklabels(action_symbols, fontsize=10)

        for row in axes:
            for ax in row[:3]:
                ax.axis("off")

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.1)

        for row in axes:
            ax2 = row[2]
            ax3 = row[3]

            pos2 = ax2.get_position()
            pos3 = ax3.get_position()

            new_spacing = 0.04  # Adjust this value to increase/decrease the space between ax2 and ax3
            new_ax3_x0 = pos2.x1 + new_spacing

            ax3.set_position([new_ax3_x0, pos3.y0, pos3.width, pos3.height])

        plt.savefig("summary.png", dpi=300, bbox_inches="tight")
        plt.close()

    def batch_plot(noisy_frames, noiseless_frames, normal_qvals, last_qs):
        sns.set()
        plt.rcParams["text.usetex"] = True
        plt.rcParams["text.latex.preamble"] = (
            r"\usepackage{amsmath} \usepackage{amssymb} \usepackage{amsfonts}"
        )

        BATCH_SIZE = 20  # Number of rows per batch
        total_rows = num_noise + 1
        num_batches = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE

        normal_last_qvals = normal_qvals[:, 0, :][-1]
        last_qs = np.array(last_qs)  # shape (num_noise, 5)
        all_qvals = [normal_last_qvals] + [last_qs[idx] for idx in range(num_noise)]
        ymin = min(q.min() for q in all_qvals)
        ymax = max(q.max() for q in all_qvals) + 0.1
        action_symbols = ["↑", "↓", "←", "→", "4"]

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
                    axes[row_idx, 0].imshow(
                        np.array(noiseless_frames[0], dtype=np.uint8)
                    )
                    axes[row_idx, 0].set_title(r"$O_0$", fontsize=20)
                    axes[row_idx, 1].imshow(
                        np.array(noiseless_frames[1], dtype=np.uint8)
                    )
                    axes[row_idx, 1].set_title(r"$O_1$", fontsize=20)
                    axes[row_idx, 2].imshow(
                        np.array(noiseless_frames[-1], dtype=np.uint8)
                    )
                    axes[row_idx, 2].set_title(rf"$O_{{{num_noise}}}$", fontsize=20)
                    max_idx = np.argmax(normal_last_qvals)
                    colors = ["#BBBBBB"] * len(normal_last_qvals)
                    colors[max_idx] = "lightblue"
                    axes[row_idx, 3].bar(
                        np.arange(normal_last_qvals.shape[0]),
                        normal_last_qvals,
                        color=colors,
                        edgecolor="black",
                    )
                    axes[row_idx, 3].set_title(
                        rf"$Q(s_{{{num_noise}}}, a_{{{num_noise}}})$", fontsize=20
                    )
                    axes[row_idx, 3].set_xticks(np.arange(len(normal_last_qvals)))
                    axes[row_idx, 3].set_xticklabels(action_symbols, fontsize=10)
                    axes[row_idx, 3].set_yticks([])
                    axes[row_idx, 3].yaxis.set_visible(False)
                else:
                    qvals = last_qs[idx - 1]
                    max_idx = np.argmax(qvals)
                    colors = ["#BBBBBB"] * len(qvals)
                    colors[max_idx] = "#FFB6C1"
                    axes[row_idx, 0].imshow(
                        np.array(noiseless_frames[0], dtype=np.uint8)
                    )
                    axes[row_idx, 0].set_title(r"$O_0$", fontsize=20)
                    axes[row_idx, 1].imshow(
                        np.array(noisy_frames[idx - 1], dtype=np.uint8)
                    )
                    axes[row_idx, 1].set_title(
                        rf"$O_{{{idx}}} + \epsilon$", fontsize=20
                    )
                    axes[row_idx, 2].imshow(
                        np.array(noiseless_frames[-1], dtype=np.uint8)
                    )
                    axes[row_idx, 2].set_title(rf"$O_{{{num_noise}}}$", fontsize=20)
                    axes[row_idx, 3].bar(
                        np.arange(qvals.shape[0]),
                        qvals,
                        color=colors,
                        edgecolor="black",
                    )
                    axes[row_idx, 3].set_title(
                        rf"$Q(s_{{{num_noise}}}, a_{{{num_noise}}})$", fontsize=20
                    )
                    axes[row_idx, 3].set_yticks([])
                    axes[row_idx, 3].yaxis.set_visible(False)
                    axes[row_idx, 3].set_xticks(np.arange(len(qvals)))
                    axes[row_idx, 3].set_xticklabels(action_symbols, fontsize=10)

                for ax in axes[row_idx, :3]:
                    ax.axis("off")

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

            plt.savefig(f"summary_batch_{batch_idx}.png", dpi=300, bbox_inches="tight")
            plt.close()

    # plot(noisy_frames, noiseless_frames, normal_qvals, last_qs)
    batch_plot(noisy_frames, noiseless_frames, normal_qvals, last_qs)


os.environ["WANDB_MODE"] = "disabled"
MEMORY_TYPES = {"lru"}
# , "mingru", "fart"
ENV_NAMES = {
    # "AutoEncodeEasy",
    # "BattleShipEasy",
    "CartPoleEasy",
    # "NoisyCartPoleEasy",
    # "CountRecallEasy",
    # "MineSweeperEasy",
    # "NavigatorEasy",
}
PATH = "./pkls_gradients/"
for filename in os.listdir(PATH):
    if filename.startswith("PQN_RNN_"):
        parts = filename.split("_")
        train_type = "_".join(parts[:2])  # "PQN_RNN"
        memory_type = parts[2].lower()
        env_name = parts[3]
        partial_part = parts[5]
        seed_part = parts[6]
    else:
        continue

    # Extract Partial and SEED values
    partial = partial_part.split("=")[1]
    seed = seed_part.split("=")[1].replace(".pkl", "")
    # Check if this file matches our criteria
    if (
        train_type == "PQN_RNN"
        and partial.lower() == "false"
        and memory_type in MEMORY_TYPES
        and env_name in ENV_NAMES
    ):

        # Create config
        config = {
            "ENV_NAME": env_name,
            "OBS_SIZE": 128,
            "MEMORY_TYPE": memory_type,
            "PARTIAL": False,
            "TRAIN_TYPE": train_type,
            "SEED": int(seed),
            "PROJECT": "noiseva",
        }
        print(f"Evaluating {filename} with config: {config}")

        rng = jax.random.PRNGKey(config["SEED"])
        rng, _rng = jax.random.split(rng)
        network = QNetworkRNN(_rng, config["OBS_SIZE"], config["MEMORY_TYPE"])
        model = eqx.tree_deserialise_leaves(PATH + filename, network)
        evaluate(model, config)

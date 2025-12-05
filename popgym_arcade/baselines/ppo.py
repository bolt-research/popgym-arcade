import time
from typing import Any, Dict, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

import popgym_arcade
import wandb
from popgym_arcade.baselines.model import ActorCritic
from popgym_arcade.baselines.utils import filter_scan
from popgym_arcade.wrappers import LogWrapper


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env, env_params = popgym_arcade.make(
        config["ENV_NAME"], partial_obs=config["PARTIAL"], obs_size=config["OBS_SIZE"]
    )
    env = LogWrapper(env)

    lr_schedule = optax.linear_schedule(
        init_value=config["LR"],
        end_value=1e-10,
        transition_steps=(config["NUM_UPDATES"])
        * config["NUM_MINIBATCHES"]
        * config["UPDATE_EPOCHS"],
    )

    def train(rng):
        rng, _rng = jax.random.split(rng)

        network = ActorCritic(key=_rng, obs_size=config["OBS_SIZE"])
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=lr_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=config["LR"], eps=1e-5),
            )

        key, _key = jax.random.split(rng)
        opt_state = tx.init(eqx.filter(network, eqx.is_array))
        reset_rng = jax.random.split(_key, config["NUM_ENVS"])
        obsv, env_state = eqx.filter_vmap(env.reset, in_axes=(0, None))(
            reset_rng, env_params
        )

        def update_step(runner_state, _):
            def step_fn(runner_state, _):
                network, opt_state, tx, env_state, last_obs, rng = runner_state
                # Select one action for a step
                rng, _rng = jax.random.split(rng)
                pi, value = network(last_obs)
                action = pi.sample(key=_rng)
                log_prob = pi.log_prob(action)

                # Step the environment
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = eqx.filter_vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (network, opt_state, tx, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = filter_scan(
                step_fn, runner_state, None, config["NUM_STEPS"]
            )
            network, opt_state, tx, env_state, last_obs, rng = runner_state
            pi, last_val = network(last_obs)

            def calculate_gae(traj_batch, last_val):
                def get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    # delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    # gae = sum_{l=0}^{\infin} (gamma * lambda)^l * delta_{t + l}
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = calculate_gae(traj_batch, last_val)

            def update_epoch(update_state, _):

                def update_minibatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info
                    network, opt_state, tx = train_state

                    def loss_fn(network, traj_batch, gae, targets):
                        pi, value = network(traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # Calculate Critic Loss
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # Calculate Actor Loss
                        # Setting the extent of model updating
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()
                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        data = {
                            "actor_loss": loss_actor,
                            "value_loss": value_loss,
                            "entropy_loss": entropy,
                        }
                        return total_loss, data

                    (total_loss, _), grads = eqx.filter_value_and_grad(
                        loss_fn, has_aux=True
                    )(network, traj_batch, advantages, targets)
                    updates, new_opt_state = tx.update(
                        grads,
                        opt_state,
                        eqx.filter(network, eqx.is_array),
                    )
                    new_network = eqx.apply_updates(network, updates)
                    new_train_state = (new_network, new_opt_state, tx)
                    return new_train_state, total_loss

                network, opt_state, tx, traj_batch, advantages, targets, rng = (
                    update_state
                )
                rng, _rng = jax.random.split(rng)

                # Batching and Shuffling
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "Batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )

                # Doing minibatch update
                mini_batches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state = (network, opt_state, tx)
                train_state, total_loss = filter_scan(
                    update_minibatch, train_state, mini_batches
                )
                network, opt_state, tx = train_state
                update_state = (
                    network,
                    opt_state,
                    tx,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            update_state = (
                network,
                opt_state,
                tx,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = filter_scan(
                update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            network = update_state[0]
            opt_state = update_state[1]
            tx = update_state[2]
            metric = traj_batch.info
            rng = update_state[-1]

            if config.get("DEBUG"):

                def callback(info):
                    return_values = info["returned_episode_returns"][
                        info["returned_episode"]
                    ]
                    timesteps = (
                        info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    )
                    for t in range(len(timesteps)):
                        log_dict = {
                            "global step": timesteps[t],
                            "episodic return": return_values[t],
                        }
                        if timesteps[t] % 100 == 0:
                            wandb.log(log_dict)

                        print(
                            f"global step={timesteps[t]}, episodic return={return_values[t]}"
                        )

                jax.debug.callback(callback, metric)
            runner_state = (network, opt_state, tx, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (network, opt_state, tx, env_state, obsv, rng)
        runner_state, metric = filter_scan(
            update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metric": metric}

    return train


def evaluate(model, config):
    seed = jax.random.PRNGKey(10)
    seed, _rng = jax.random.split(seed)
    env, env_params = popgym_arcade.make(
        config["ENV_NAME"], partial_obs=config["PARTIAL"], obs_size=config["OBS_SIZE"]
    )
    env = LogWrapper(env)
    vmap_reset = lambda n_envs: lambda rng: jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(rng, n_envs), env_params
    )
    vmap_step = lambda n_envs: lambda rng, env_state, action: jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )(jax.random.split(rng, n_envs), env_state, action, env_params)

    if config["WANDB_MODE"] != "disabled":
        obs, state = vmap_reset(2)(_rng)

        frames = []
        for i in range(1000):
            rng, rng_act, rng_step, _rng = jax.random.split(_rng, 4)
            pi, critic = model(obs)
            action = pi.sample(_rng)
            obs, new_state, reward, term, _ = vmap_step(2)(rng_step, state, action)
            state = new_state
            frame = jnp.asarray(obs[0])
            frames.append(frame)

        frames = np.array(frames, dtype=np.uint8)
        frames = frames.transpose((0, 3, 1, 2))
        wandb.log({"Video": wandb.Video(frames, fps=4, format="gif")})

    # imageio.mimsave(f"{config["TRAIN_TYPE"]}_{config["ENV_NAME"]}_Partial={config["PARTIAL"]}_Seed={config["SEED"]}.gif", frames)


def ppo_run(config: Dict[str, Any]):
    if config["WANDB_MODE"] != "disabled":
        # initialize the wandb
        wandb.init(
            entity=config["ENTITY"],
            project=config["PROJECT"],
            tags=["PPO", config["ENV_NAME"].upper(), f"jax_{jax.__version__}"],
            name=f'{config["TRAIN_TYPE"]}_{config["ENV_NAME"]}_Partial={config["PARTIAL"]}_Seed={config["SEED"]}',
            config=config,
            mode=config["WANDB_MODE"],
        )

    rng = jax.random.PRNGKey(config["SEED"])
    t0 = time.time()
    rng_array = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = eqx.filter_jit(eqx.filter_vmap(make_train(config)))
    outs = jax.block_until_ready(train_vjit(rng_array))
    print(f"Took {time.time() - t0} seconds to complete.")
    runner_state = outs["runner_state"]
    network = runner_state[0]
    network_squeezed = jax.tree.map(
        lambda x: (
            x.squeeze(0)
            if (hasattr(x, "ndim") and x.ndim > 1 and x.shape[0] == 1)
            else x
        ),
        network,
    )
    eqx.tree_serialise_leaves(
        "{}_{}_model_Partial={}_SEED={}.pkl".format(
            config["TRAIN_TYPE"], config["ENV_NAME"], config["PARTIAL"], config["SEED"]
        ),
        network_squeezed,
    )
    rng, _rng = jax.random.split(rng)
    network = ActorCritic(key=_rng, obs_size=config["OBS_SIZE"])
    model = eqx.tree_deserialise_leaves(
        "{}_{}_model_Partial={}_SEED={}.pkl".format(
            config["TRAIN_TYPE"], config["ENV_NAME"], config["PARTIAL"], config["SEED"]
        ),
        network,
    )
    evaluate(model, config)

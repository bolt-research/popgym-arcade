import time
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import lax

import popgym_arcade
import wandb
from memax.equinox.train_utils import add_batch_dim
from popgym_arcade.baselines.model import ActorCriticRNN
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

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    lr_schedule = optax.linear_schedule(
        init_value=config["LR"],
        end_value=1e-10,
        transition_steps=(config["NUM_UPDATES"])
        * config["NUM_MINIBATCHES"]
        * config["UPDATE_EPOCHS"],
    )

    def train(rng):
        # INIT NETWORK
        rng, _rng, rng_init = jax.random.split(rng, 3)
        network = ActorCriticRNN(
            key=_rng, obs_size=config["OBS_SIZE"], rnn_type=config["MEMORY_TYPE"]
        )
        actor_init_hstate, critic_init_hstate = network.initialize_carry(key=rng_init)
        actor_init_hstate = add_batch_dim(actor_init_hstate, config["NUM_ENVS"])
        critic_init_hstate = add_batch_dim(critic_init_hstate, config["NUM_ENVS"])
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=lr_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        opt_state = tx.init(eqx.filter(network, eqx.is_array))

        # INIT ENV
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                (
                    network,
                    opt_state,
                    tx,
                    env_state,
                    last_obs,
                    last_done,
                    actor_hstate,
                    critic_hstate,
                    rng,
                ) = runner_state

                rng, _rng = jax.random.split(rng)

                # SELECT ACTION
                ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
                actor_hstate, critic_hstate, pi, value = network(
                    actor_hstate, critic_hstate, ac_in
                )
                action = pi.sample(key=_rng)
                log_prob = pi.log_prob(action)

                value, action, log_prob = (
                    value.squeeze(0),
                    action.squeeze(0),
                    log_prob.squeeze(0),
                )
                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)
                transition = Transition(
                    last_done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (
                    network,
                    opt_state,
                    tx,
                    env_state,
                    obsv,
                    done,
                    actor_hstate,
                    critic_hstate,
                    rng,
                )
                return runner_state, transition

            actor_initial_hstate = runner_state[-3]
            critic_initial_hstate = runner_state[-2]
            runner_state, traj_batch = filter_scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            (
                network,
                opt_state,
                tx,
                env_state,
                last_obs,
                last_done,
                actor_hstate,
                critic_hstate,
                rng,
            ) = runner_state
            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
            _, _, _, last_val = network(actor_hstate, critic_hstate, ac_in)
            last_val = last_val.squeeze(0)

            def _calculate_gae(traj_batch, last_val, last_done):
                def _get_advantages(carry, transition):
                    gae, next_value, next_done = carry
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = (
                        reward + config["GAMMA"] * next_value * (1 - next_done) - value
                    )
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - next_done) * gae
                    )
                    return (gae, value, done), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val, last_done),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val, last_done)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    (
                        actor_init_hstate,
                        critic_init_hstate,
                        traj_batch,
                        advantages,
                        targets,
                    ) = batch_info
                    network, opt_state, tx = train_state

                    def _loss_fn(
                        network,
                        actor_init_hstate,
                        critic_init_hstate,
                        traj_batch,
                        gae,
                        targets,
                    ):
                        # RERUN NETWORK

                        actor_hstate_med = jax.tree.map(
                            lambda s: s[0], actor_init_hstate
                        )
                        critic_hstate_med = jax.tree.map(
                            lambda s: s[0], critic_init_hstate
                        )

                        _, _, pi, value = network(
                            actor_hstate_med,
                            critic_hstate_med,
                            (traj_batch.obs, traj_batch.done),
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
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
                        return total_loss, (value_loss, loss_actor, entropy)

                    (total_loss, _), grads = eqx.filter_value_and_grad(
                        _loss_fn, has_aux=True
                    )(
                        network,
                        actor_init_hstate,
                        critic_init_hstate,
                        traj_batch,
                        advantages,
                        targets,
                    )
                    updates, new_opt_state = tx.update(
                        grads,
                        opt_state,
                        eqx.filter(network, eqx.is_array),
                    )
                    new_network = eqx.apply_updates(network, updates)
                    new_train_state = (new_network, new_opt_state, tx)
                    return new_train_state, total_loss

                (
                    network,
                    opt_state,
                    tx,
                    actor_init_hstate,
                    critic_init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state

                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])
                batch = (
                    actor_init_hstate,
                    critic_init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                )

                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )
                train_state = (network, opt_state, tx)
                train_state, total_loss = filter_scan(
                    _update_minbatch, train_state, minibatches
                )
                network, opt_state, tx = train_state
                update_state = (
                    network,
                    opt_state,
                    tx,
                    actor_init_hstate,
                    critic_init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            actor_init_hstate = jax.tree.map(
                lambda a: jnp.expand_dims(a, axis=0), actor_initial_hstate
            )
            critic_init_hstate = jax.tree.map(
                lambda a: jnp.expand_dims(a, axis=0), critic_initial_hstate
            )
            # init_hstate = initial_hstate[None, :]  # TBH
            update_state = (
                network,
                opt_state,
                tx,
                actor_init_hstate,
                critic_init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = filter_scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
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
                        print(
                            f"global step={timesteps[t]}, episodic return={return_values[t]}"
                        )
                        if timesteps[t] % 100 == 0:
                            wandb.log(log_dict)

                jax.debug.callback(callback, metric)

            runner_state = (
                network,
                opt_state,
                tx,
                env_state,
                last_obs,
                last_done,
                actor_hstate,
                critic_hstate,
                rng,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            network,
            opt_state,
            tx,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ENVS"]), dtype=bool),
            actor_init_hstate,
            critic_init_hstate,
            _rng,
        )
        runner_state, metric = filter_scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
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

    obs, state = vmap_reset(2)(_rng)
    init_done = jnp.zeros(2, dtype=bool)
    init_action = jnp.zeros(2, dtype=jnp.int8)
    actor_hstate, critic_hstate = model.initialize_carry(key=_rng)
    actor_hstate = add_batch_dim(actor_hstate, 2)
    critic_hstate = add_batch_dim(critic_hstate, 2)

    # Initialize frames array
    frame_shape = obs[0].shape
    frames = jnp.zeros((500, *frame_shape), dtype=jnp.uint8)

    carry = (
        actor_hstate,
        critic_hstate,
        obs,
        init_done,
        init_action,
        state,
        frames,
        _rng,
    )
    if config["WANDB_MODE"] != "disabled":
        wandb.init(project=f'{config["PROJECT"]}')

    def evaluate_step(carry, i):
        actor_hstate, critic_hstate, obs, done, action, state, frames, _rng = carry
        _rng, rng_step = jax.random.split(_rng, 2)
        obs_batch = obs[jnp.newaxis, :]
        done_batch = done[jnp.newaxis, :]
        # action_batch = action[jnp.newaxis, :]
        ac_in = (obs_batch, done_batch)
        actor_hstate, critic_hstate, pi, value = model(
            actor_hstate, critic_hstate, ac_in
        )
        action = pi.sample(_rng)
        action = jnp.squeeze(action, axis=0)
        obs, new_state, reward, done, info = vmap_step(2)(rng_step, state, action)
        state = new_state
        frame = jnp.asarray(obs[0])
        # Update frames array at index i
        frames = frames.at[i].set(frame)
        carry = (actor_hstate, critic_hstate, obs, done, action, state, frames, _rng)
        return carry, reward

    def body_fun(i, carry):
        carry, _ = evaluate_step(carry, i)
        return carry

    carry = lax.fori_loop(0, 500, body_fun, carry)
    _, _, _, _, _, _, frames, _rng = carry
    frames = np.array(frames, dtype=np.uint8)
    frames = frames.transpose((0, 3, 1, 2))
    if config["WANDB_MODE"] != "disabled":
        wandb.log({"{}".format(config["ENV_NAME"]): wandb.Video(frames, fps=4)})


def ppo_rnn_run(config):
    if config["WANDB_MODE"] != "disabled":
        wandb.init(
            entity=config["ENTITY"],
            project=config["PROJECT"],
            tags=["PPO", config["ENV_NAME"].upper(), f"jax_{jax.__version__}"],
            name=f'{config["TRAIN_TYPE"]}_{config["MEMORY_TYPE"]}_{config["ENV_NAME"]}_Partial={config["PARTIAL"]}_Seed={config["SEED"]}',
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
        "{}_{}_{}_model_Partial={}_SEED={}.pkl".format(
            config["TRAIN_TYPE"],
            config["MEMORY_TYPE"],
            config["ENV_NAME"],
            config["PARTIAL"],
            config["SEED"],
        ),
        network_squeezed,
    )
    rng, _rng = jax.random.split(rng)
    network = ActorCriticRNN(_rng, config["OBS_SIZE"], config["MEMORY_TYPE"])
    model = eqx.tree_deserialise_leaves(
        "{}_{}_{}_model_Partial={}_SEED={}.pkl".format(
            config["TRAIN_TYPE"],
            config["MEMORY_TYPE"],
            config["ENV_NAME"],
            config["PARTIAL"],
            config["SEED"],
        ),
        network,
    )
    # visualize_grad(model, config)
    evaluate(model, config)

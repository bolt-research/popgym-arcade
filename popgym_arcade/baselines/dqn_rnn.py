import copy
import time
from typing import Any, NamedTuple

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import lax
import flashbax as fbx

import popgym_arcade
import wandb
from popgym_arcade.baselines.model import QNetworkRNN, add_batch_dim
from popgym_arcade.wrappers import LogWrapper


def debug_shape(x):
    import equinox as eqx

    return eqx.tree_pprint(jax.tree.map(lambda x: {x.shape: x.dtype}, x))


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



class Transition(NamedTuple):
    last_hs: chex.Array
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    last_done: chex.Array
    last_action: chex.Array
    q_vals: chex.Array
    infos: chex.Array


class State(eqx.Module):
    """Base class"""

    def replace(self, **kwargs):
        """Replaces existing fields.

        E.g., s = State(bork=1, dork=2)
        s.replace(dork=3)
        print(s)
            >> State(bork=1, dork=3)
        """
        fields = self.__dataclass_fields__
        assert set(kwargs.keys()).issubset(fields)
        new_pytree = {}
        for k in fields:
            if k in kwargs:
                new_pytree[k] = kwargs[k]
            else:
                new_pytree[k] = getattr(self, k)
        return type(self)(**new_pytree)


class TrainState(State):
    model: eqx.Module
    opt: optax.GradientTransformation
    opt_state: optax.OptState
    timesteps: jnp.ndarray
    n_updates: jnp.ndarray
    grad_steps: jnp.ndarray
    buffer_state: Any


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    config["NUM_UPDATES_DECAY"] = (
        config["TOTAL_TIMESTEPS_DECAY"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    assert config["NUM_ENVS"] % config[
        "NUM_MINIBATCHES"
    ] == 0, "NUM_MINIBATCHES must divide NUM_ENVS"

    env, env_params = popgym_arcade.make(
        config["ENV_NAME"], partial_obs=config["PARTIAL"], obs_size=config["OBS_SIZE"]
    )
    env = LogWrapper(env)
    # config["TEST_NUM_STEPS"] = config.get(
    #     "TEST_NUM_STEPS", env_params.max_steps_in_episode
    # )
    vmap_reset = lambda n_envs: lambda rng: jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(rng, n_envs), env_params
    )
    vmap_step = lambda n_envs: lambda rng, env_state, action: jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )(jax.random.split(rng, n_envs), env_state, action, env_params)

    # epsilon-greedy exploration
    def eps_greedy_exploration(rng, q_vals, eps):
        rng_a, rng_e = jax.random.split(
            rng
        )  # a key for sampling random actions and one for picking
        greedy_actions = jnp.argmax(q_vals, axis=-1)
        chosed_actions = jnp.where(
            jax.random.uniform(rng_e, greedy_actions.shape)
            < eps,  # pick the actions that should be random
            jax.random.randint(
                rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]
            ),  # sample random actions,
            greedy_actions,
        )
        return chosed_actions

    def train(rng):
        original_rng = rng[0]
        eps_scheduler = optax.linear_schedule(
            config["EPS_START"],
            config["EPS_FINISH"],
            (config["EPS_DECAY"])
            * config["NUM_UPDATES_DECAY"]
            * config["NUM_MINIBATCHES"],
        )

        lr_scheduler = optax.linear_schedule(
            init_value=config["LR"],
            end_value=5e-7,
            transition_steps=(config["NUM_UPDATES_DECAY"])
            * config["NUM_MINIBATCHES"]
            * config["NUM_EPOCHS"],
        )
        lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]
        rng, _rng, rng_init = jax.random.split(rng, 3)

        network = QNetworkRNN(rng, config["OBS_SIZE"], config["MEMORY_TYPE"])

        hidden_state = network.initialize_carry(key=rng_init)
        hidden_state = add_batch_dim(hidden_state, config["NUM_ENVS"])

        opt = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.radam(learning_rate=lr),
        )

        rng, _rng = jax.random.split(rng)
        opt_state = opt.init(eqx.filter(network, eqx.is_array))

        # Replay buffer setup
        BATCH_SIZE = config["NUM_ENVS"] * 2
        # Default to smaller buffer for images to avoid OOM (1000 seqs * 128 steps * 128*128*3 bytes ~ 7GB)
        BUFFER_SIZE = config.get("BUFFER_SIZE", 1000)
        buffer = fbx.make_flat_buffer(
            max_length=BUFFER_SIZE,
            min_length=config["NUM_ENVS"] * 2,
            sample_batch_size=BATCH_SIZE,
        )

        dummy_obs = jnp.zeros((config["MEMORY_WINDOW"] + config["NUM_STEPS"], config["OBS_SIZE"], config["OBS_SIZE"], 3), dtype=jnp.float32)
        dummy_action = jnp.zeros((config["MEMORY_WINDOW"] + config["NUM_STEPS"],), dtype=jnp.int32)
        dummy_target = jnp.zeros((config["MEMORY_WINDOW"] + config["NUM_STEPS"],), dtype=jnp.float32)
        dummy_last_hs = jax.tree.map(lambda x: jnp.zeros((config["MEMORY_WINDOW"] + config["NUM_STEPS"], *x.shape[1:]), dtype=x.dtype), hidden_state)
        dummy_last_done = jnp.zeros((config["MEMORY_WINDOW"] + config["NUM_STEPS"],), dtype=bool)
        dummy_last_action = jnp.zeros((config["MEMORY_WINDOW"] + config["NUM_STEPS"],), dtype=jnp.int32)
        
        dummy_transition = {
            "obs": dummy_obs,
            "action": dummy_action,
            "target": dummy_target,
            "last_hs": dummy_last_hs,
            "last_done": dummy_last_done,
            "last_action": dummy_last_action
        }
        buffer_state = buffer.init(dummy_transition)

        train_state = TrainState(
            model=network,
            opt=opt,
            opt_state=opt_state,
            timesteps=jnp.array(0),
            n_updates=jnp.array(0),
            grad_steps=jnp.array(0),
            buffer_state=buffer_state,
        )

        # TRAINING LOOP
        def _update_step(runner_state, unused):

            train_state, memory_transitions, expl_state, test_metrics, rng = (
                runner_state
            )

            # SAMPLE PHASE
            def _step_env(runner_state, _):
                train_state, memory_transitions, expl_state, test_metrics, rng = (
                    runner_state
                )
                hs, last_obs, last_done, last_action, env_state = expl_state
                rng, rng_a, rng_s = jax.random.split(rng, 3)

                _obs = last_obs[np.newaxis]  # (1 (dummy time), num_envs, obs_size)
                _done = last_done[np.newaxis]  # (1 (dummy time), num_envs)
                _last_action = last_action[np.newaxis]  # (1 (dummy time), num_envs)

                new_hs, q_vals = train_state.model(
                    hs,
                    _obs,
                    _done,
                    _last_action,
                )  # (num_envs, hidden_size), (1, num_envs, num_actions)
                q_vals = q_vals.squeeze(
                    axis=0
                )  # (num_envs, num_actions) remove the time dim

                _rngs = jax.random.split(rng_a, config["NUM_ENVS"])
                eps = jnp.full(config["NUM_ENVS"], eps_scheduler(train_state.n_updates))
                new_action = eqx.filter_vmap(eps_greedy_exploration)(_rngs, q_vals, eps)

                new_obs, new_env_state, reward, new_done, info = vmap_step(
                    config["NUM_ENVS"]
                )(rng_s, env_state, new_action)

                transition = Transition(
                    last_hs=hs,
                    obs=last_obs,
                    action=new_action,
                    reward=config.get("REW_SCALE", 1) * reward,
                    done=new_done,
                    last_done=last_done,
                    last_action=last_action,
                    q_vals=q_vals,
                    infos=info,
                )
                new_expl_state = (new_hs, new_obs, new_done, new_action, new_env_state)
                runner_state = (
                    train_state,
                    memory_transitions,
                    new_expl_state,
                    test_metrics,
                    rng,
                )
                return runner_state, transition

            # step the env
            rng, _rng = jax.random.split(rng)
            runner_state, transitions = filter_scan(
                _step_env,
                runner_state,
                None,
                config["NUM_STEPS"],
            )
            train_state, memory_transitions, expl_state, test_metrics, rng = (
                runner_state
            )
            expl_state = tuple(expl_state)

            train_state = train_state.replace(
                timesteps=train_state.timesteps
                + config["NUM_STEPS"] * config["NUM_ENVS"]
            )  # update timesteps count

            # insert the transitions into the memory
            memory_transitions = jax.tree.map(
                lambda x, y: jnp.concatenate([x[config["NUM_STEPS"] :], y], axis=0),
                memory_transitions,
                transitions,
            )

            # Compute targets
            hs, last_obs, last_done, last_action, env_state = expl_state
            _obs = last_obs[np.newaxis]
            _done = last_done[np.newaxis]
            _last_action = last_action[np.newaxis]
            
            _, last_q_vals = train_state.model(hs, _obs, _done, _last_action)
            last_q = jnp.max(last_q_vals[0], axis=-1)

            def _compute_targets(last_q, q_vals, reward, done):
                def _get_target(lambda_returns_and_next_q, rew_q_done):
                    reward, q, done = rew_q_done
                    lambda_returns, next_q = lambda_returns_and_next_q
                    target_bootstrap = (
                        reward + config["GAMMA"] * (1 - done) * next_q
                    )
                    delta = lambda_returns - next_q
                    lambda_returns = (
                        target_bootstrap
                        + config["GAMMA"] * config["LAMBDA"] * delta
                    )
                    lambda_returns = (1 - done) * lambda_returns + done * reward
                    next_q = jnp.max(q, axis=-1)
                    return (lambda_returns, next_q), lambda_returns

                lambda_returns = (
                    reward[-1] + config["GAMMA"] * (1 - done[-1]) * last_q
                )
                last_q_scan = jnp.max(q_vals[-1], axis=-1)
                _, targets = jax.lax.scan(
                    _get_target,
                    (lambda_returns, last_q_scan),
                    jax.tree.map(lambda x: x[:-1], (reward, q_vals, done)),
                    reverse=True,
                )
                targets = jnp.concatenate([targets, lambda_returns[np.newaxis]])
                return targets

            targets = _compute_targets(
                last_q,
                memory_transitions.q_vals,
                memory_transitions.reward,
                memory_transitions.done
            )

            # Add to buffer
            flat_obs = jnp.swapaxes(memory_transitions.obs, 0, 1)
            flat_action = jnp.swapaxes(memory_transitions.action, 0, 1)
            flat_target = jnp.swapaxes(targets, 0, 1)
            flat_last_hs = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), memory_transitions.last_hs)
            flat_last_done = jnp.swapaxes(memory_transitions.last_done, 0, 1)
            flat_last_action = jnp.swapaxes(memory_transitions.last_action, 0, 1)

            def add_to_buffer(buffer_state, i):
                item = {
                    "obs": flat_obs[i],
                    "action": flat_action[i],
                    "target": flat_target[i],
                    "last_hs": jax.tree.map(lambda x: x[i], flat_last_hs),
                    "last_done": flat_last_done[i],
                    "last_action": flat_last_action[i]
                }
                return buffer.add(buffer_state, item), None

            new_buffer_state, _ = jax.lax.scan(
                add_to_buffer,
                train_state.buffer_state,
                jnp.arange(flat_obs.shape[0])
            )
            train_state = train_state.replace(buffer_state=new_buffer_state)

            # NETWORKS UPDATE
            dynamic_train_state, static_train_state = eqx.partition(train_state, eqx.is_array)

            def _do_update(operand):
                dynamic_train_state, rng = operand
                train_state = eqx.combine(dynamic_train_state, static_train_state)
                
                rng, key = jax.random.split(rng)
                batch = buffer.sample(train_state.buffer_state, key)
                
                # Reshape for minibatches
                # batch.experience.first leaves shape: (BATCH_SIZE, ...) -> (NUM_MINIBATCHES, MINIBATCH_SIZE, ...)
                minibatch_size = BATCH_SIZE // config["NUM_MINIBATCHES"]
                
                def reshape_for_minibatch(x):
                    # x: (BATCH_SIZE, ...)
                    return x.reshape((config["NUM_MINIBATCHES"], minibatch_size, *x.shape[1:]))

                minibatches = jax.tree.map(reshape_for_minibatch, batch.experience.first)

                def _learn_epoch(carry, _):
                    train_state, rng = carry
                    
                    def _learn_minibatch(carry, minibatch):
                        train_state, rng = carry
                        
                        minibatch_obs = jnp.swapaxes(minibatch["obs"], 0, 1)
                        minibatch_action = jnp.swapaxes(minibatch["action"], 0, 1)
                        minibatch_target = jnp.swapaxes(minibatch["target"], 0, 1)
                        minibatch_last_hs = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), minibatch["last_hs"])
                        minibatch_last_done = jnp.swapaxes(minibatch["last_done"], 0, 1)
                        minibatch_last_action = jnp.swapaxes(minibatch["last_action"], 0, 1)

                        hs = jax.tree.map(lambda hs: hs[0], minibatch_last_hs)
                        agent_in = (
                            minibatch_obs,
                            minibatch_last_done,
                            minibatch_last_action,
                        )

                        def _loss_fn(network):
                            hidden_state, q_vals = network(
                                hs,
                                *agent_in,
                            )
                            
                            chosen_action_qvals = jnp.take_along_axis(
                                q_vals,
                                jnp.expand_dims(minibatch_action, axis=-1),
                                axis=-1,
                            ).squeeze(
                                axis=-1
                            )
                            
                            loss = 0.5 * jnp.square(chosen_action_qvals - minibatch_target).mean()
                            return loss, chosen_action_qvals

                        (loss, qvals), grads = eqx.filter_value_and_grad(
                            _loss_fn, has_aux=True
                        )(train_state.model)
                        updates, new_opt_state = train_state.opt.update(
                            grads,
                            train_state.opt_state,
                            eqx.filter(train_state.model, eqx.is_array),
                        )
                        new_network = eqx.apply_updates(train_state.model, updates)
                        new_train_state = train_state.replace(
                            model=new_network,
                            opt_state=new_opt_state,
                            grad_steps=train_state.grad_steps + 1,
                        )
                        return (new_train_state, rng), (loss, qvals)

                    (train_state, rng), (loss, qvals) = filter_scan(
                        _learn_minibatch, (train_state, rng), minibatches
                    )
                    return (train_state, rng), (loss, qvals)

                (train_state, rng), (loss, qvals) = filter_scan(
                    _learn_epoch, (train_state, rng), None, config["NUM_EPOCHS"]
                )
                train_state = train_state.replace(n_updates=train_state.n_updates + 1)
                
                new_dynamic_train_state, _ = eqx.partition(train_state, eqx.is_array)
                return new_dynamic_train_state, rng, loss.mean(), qvals.mean()

            def _skip_update(operand):
                dynamic_train_state, rng = operand
                return dynamic_train_state, rng, 0.0, 0.0

            dynamic_train_state, rng, loss, qvals = jax.lax.cond(
                buffer.can_sample(train_state.buffer_state),
                _do_update,
                _skip_update,
                (dynamic_train_state, rng)
            )
            train_state = eqx.combine(dynamic_train_state, static_train_state)
            metrics = {
                "env_step": train_state.timesteps,
                "update_steps": train_state.n_updates,
                "grad_steps": train_state.grad_steps,
                "td_loss": loss.mean(),
                "qvals": qvals.mean(),
            }
            metrics.update({k: v.mean() for k, v in transitions.infos.items()})

            if config.get("TEST_DURING_TRAINING", False):
                rng, _rng = jax.random.split(rng)
                test_metrics = jax.lax.cond(
                    train_state.n_updates
                    % int(config["NUM_UPDATES"] * config["TEST_INTERVAL"])
                    == 0,
                    lambda _: get_test_metrics(train_state, _rng),
                    lambda _: test_metrics,
                    operand=None,
                )
                metrics.update({f"test_{k}": v for k, v in test_metrics.items()})

            # report on wandb if required
            if config["WANDB_MODE"] != "disabled":

                def callback(metrics, original_rng):
                    if config.get("WANDB_LOG_ALL_SEEDS", False):
                        metrics.update(
                            {
                                f"rng{int(original_rng)}/{k}": v
                                for k, v in metrics.items()
                            }
                        )
                    wandb.log(metrics, step=metrics["update_steps"])

                jax.debug.callback(callback, metrics, original_rng)

            runner_state = (
                train_state,
                memory_transitions,
                tuple(expl_state),
                test_metrics,
                rng,
            )

            return runner_state, metrics

        def get_test_metrics(train_state, rng):

            if not config.get("TEST_DURING_TRAINING", False):
                return None

            def _greedy_env_step(carry, _):
                train_state, step_state = carry
                hs, last_obs, last_done, last_action, env_state, rng = step_state
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                _obs = last_obs[np.newaxis]  # (1 (dummy time), num_envs, obs_size)
                _done = last_done[np.newaxis]  # (1 (dummy time), num_envs)
                _last_action = last_action[np.newaxis]  # (1 (dummy time), num_envs)
                new_hs, q_vals = train_state.model(
                    hs,
                    _obs,
                    _done,
                    _last_action,
                )  # (num_envs, hidden_size), (1, num_envs, num_actions)
                q_vals = q_vals.squeeze(
                    axis=0
                )  # (num_envs, num_actions) remove the time dim
                eps = jnp.full(config["TEST_NUM_ENVS"], config["EPS_TEST"])
                new_action = jax.vmap(eps_greedy_exploration)(
                    jax.random.split(rng_a, config["TEST_NUM_ENVS"]), q_vals, eps
                )
                new_obs, new_env_state, reward, new_done, info = vmap_step(
                    config["TEST_NUM_ENVS"]
                )(rng_s, env_state, new_action)
                step_state = (new_hs, new_obs, new_done, new_action, new_env_state, rng)
                carry = (train_state, step_state)
                return carry, info

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = vmap_reset(config["TEST_NUM_ENVS"])(_rng)
            init_done = jnp.zeros((config["TEST_NUM_ENVS"]), dtype=bool)
            init_action = jnp.zeros((config["TEST_NUM_ENVS"]), dtype=int)
            init_hs = train_state.model.initialize_carry(key=_rng)
            init_hs = add_batch_dim(init_hs, config["TEST_NUM_ENVS"])

            step_state = (
                init_hs,
                init_obs,
                init_done,
                init_action,
                env_state,
                _rng,
            )
            carry = (train_state, step_state)
            carry, infos = filter_scan(
                _greedy_env_step, carry, None, config["TEST_NUM_STEPS"]
            )
            # return mean of done infos
            done_infos = jax.tree.map(
                lambda x: jnp.nanmean(
                    jnp.where(
                        infos["returned_episode"],
                        x,
                        jnp.nan,
                    )
                ),
                infos,
            )
            return done_infos

        rng, _rng = jax.random.split(rng)
        test_metrics = get_test_metrics(train_state, _rng)

        rng, _rng = jax.random.split(rng)
        obs, env_state = vmap_reset(config["NUM_ENVS"])(_rng)
        init_dones = jnp.zeros((config["NUM_ENVS"]), dtype=bool)
        init_action = jnp.zeros((config["NUM_ENVS"]), dtype=int)

        expl_state = (hidden_state, obs, init_dones, init_action, env_state)

        # step randomly to have the initial memory window
        def _random_step(carry, _):
            _carry, rng = carry
            train_state, expl_state = _carry
            hs, last_obs, last_done, last_action, env_state = expl_state
            rng, rng_a, rng_s = jax.random.split(rng, 3)
            _obs = last_obs[np.newaxis]  # (1 (dummy time), num_envs, obs_size)
            _done = last_done[np.newaxis]  # (1 (dummy time), num_envs)
            _last_action = last_action[np.newaxis]  # (1 (dummy time), num_envs)
            new_hs, q_vals = train_state.model(
                hs,
                _obs,
                _done,
                _last_action,
            )  # (num_envs, hidden_size), (1, num_envs, num_actions)
            q_vals = q_vals.squeeze(
                axis=0
            )  # (num_envs, num_actions) remove the time dim
            _rngs = jax.random.split(rng_a, config["NUM_ENVS"])
            eps = jnp.full(config["NUM_ENVS"], 1.0)  # random actions
            new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)
            new_obs, new_env_state, reward, new_done, info = vmap_step(
                config["NUM_ENVS"]
            )(rng_s, env_state, new_action)
            transition = Transition(
                last_hs=hs,
                obs=last_obs,
                action=new_action,
                reward=config.get("REW_SCALE", 1) * reward,
                done=new_done,
                last_done=last_done,
                last_action=last_action,
                q_vals=q_vals,
                infos=info,
            )
            new_expl_state = (new_hs, new_obs, new_done, new_action, new_env_state)
            _carry = (train_state, new_expl_state)
            carry = (_carry, rng)
            return carry, transition

        rng, _rng = jax.random.split(rng)
        _carry = (train_state, expl_state)
        (_carry, rng), memory_transitions = filter_scan(
            _random_step,
            (_carry, _rng),
            None,
            config["MEMORY_WINDOW"] + config["NUM_STEPS"],
        )
        train_state, expl_state = _carry
        expl_state = tuple(expl_state)

        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, memory_transitions, expl_state, test_metrics, _rng)

        runner_state, metrics = filter_scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metrics}

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
    init_action = jnp.zeros(2, dtype=int)
    init_hs = model.initialize_carry(key=_rng)
    hs = add_batch_dim(init_hs, 2)

    # Initialize frames array
    frame_shape = obs[0].shape
    frames = jnp.zeros((500, *frame_shape), dtype=jnp.uint8)

    carry = (hs, obs, init_done, init_action, state, frames, _rng)
    if config["WANDB_MODE"] != "disabled":
        wandb.init(project=f'{config["PROJECT"]}')

    def evaluate_step(carry, i):
        hs, obs, done, action, state, frames, _rng = carry
        _rng, rng_step = jax.random.split(_rng, 2)
        obs_batch = obs[jnp.newaxis, :]
        done_batch = done[jnp.newaxis, :]
        action_batch = action[jnp.newaxis, :]
        hs, q_val = model(hs, obs_batch, done_batch, action_batch)
        q_val = lax.stop_gradient(q_val)
        q_val = q_val.squeeze(axis=0)
        action = jnp.argmax(q_val, axis=-1)
        obs, new_state, reward, done, info = vmap_step(2)(rng_step, state, action)
        state = new_state
        frame = jnp.asarray(obs[0]) 
        # Update frames array at index i
        frames = frames.at[i].set(frame)
        carry = (hs, obs, done, action, state, frames, _rng)
        return carry, reward

    def body_fun(i, carry):
        carry, _ = evaluate_step(carry, i)
        return carry

    if config["WANDB_MODE"] != "disabled":
        carry = lax.fori_loop(0, 500, body_fun, carry)
        _, _, _, _, _, frames, _rng = carry
        frames = np.array(frames, dtype=np.uint8)
        frames = frames.transpose((0, 3, 1, 2))
        wandb.log(
            {
                "{}_{}_{}_model_Partial={}_SEED={}".format(
                    config["TRAIN_TYPE"],
                    config["MEMORY_TYPE"],
                    config["ENV_NAME"],
                    config["PARTIAL"],
                    config["SEED"],
                ): wandb.Video(frames, fps=4)
            }
        )


def single_run(config):
    alg_name = config.get("ALG_NAME", "dqn_rnn")
    env_name = config["ENV_NAME"]

    if config["WANDB_MODE"] != "disabled":
        wandb.init(
            entity=config["ENTITY"],
            project=config["PROJECT"],
            tags=[
                alg_name.upper(),
                env_name.upper(),
                f"jax_{jax.__version__}",
            ],
            name=f'{config["TRAIN_TYPE"]}_{config["MEMORY_TYPE"]}_{config["ENV_NAME"]}_{"SEED="}{config["SEED"]}_{"Partial="}{config["PARTIAL"]}',
            config=config,
            mode=config["WANDB_MODE"],
        )
        wandb.run.log_code(".")

    rng = jax.random.PRNGKey(config["SEED"])

    t0 = time.time()
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = eqx.filter_jit(eqx.filter_vmap(make_train(config)))
    outs = jax.block_until_ready(train_vjit(rngs))
    
    # outs = jax.block_until_ready(eqx.filter_jit(eqx.filter_vmap(train))(rngs))
    print(f"Took {time.time() - t0} seconds to complete.")

    # evaluate
    train_state = outs["runner_state"][0]

    network_squeezed = jax.tree.map(
        lambda x: (
            x.squeeze(0)
            if (hasattr(x, "ndim") and x.ndim > 1 and x.shape[0] == 1)
            else x
        ),
        train_state.model,
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
    network = QNetworkRNN(_rng, config["OBS_SIZE"], config["MEMORY_TYPE"])
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

    evaluate(model, config)


def tune(default_config):
    """Hyperparameter sweep with wandb."""

    # default_config = {**default_config, **default_config["alg"]}
    alg_name = default_config.get("ALG_NAME", "dqn_rnn")
    env_name = default_config["ENV_NAME"]

    def wrapped_make_train():
        if default_config["WANDB_MODE"] != "disabled":
            wandb.init(project=default_config["PROJECT"])

        config = copy.deepcopy(default_config)
        for k, v in dict(wandb.config).items():
            config[k] = v

        print("running experiment with params:", config)
        t0 = time.time()
        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_vjit = eqx.filter_jit(eqx.filter_vmap(make_train(config)))
        outs = jax.block_until_ready(train_vjit(rngs))
        print(f"Took {time.time() - t0} seconds to complete.")

    sweep_config = {
        "name": f"{alg_name}_{env_name}",
        "method": "bayes",
        "metric": {
            "name": "test_returned_episode_returns",
            "goal": "maximize",
        },
        "parameters": {
            "LR": {
                "values": [
                    0.001,
                    0.0001,
                    0.0002,
                    0.0003,
                    0.0005,
                    0.00005,
                    0.000075,
                ]
            },
        },
    }

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    wandb.agent(sweep_id, wrapped_make_train, count=1000)


def dqn_rnn_run(config):
    if config["HYP_TUNE"]:
        tune(config)
    else:
        single_run(config)

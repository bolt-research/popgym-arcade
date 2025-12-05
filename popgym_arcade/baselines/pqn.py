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

import popgym_arcade
import wandb
from popgym_arcade.baselines.model import QNetwork
from popgym_arcade.wrappers import LogWrapper


def debug_shape(x):
    jax.tree.map(lambda x: print(x.shape) if hasattr(x, "shape") else x, x)


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


class BoltzmannActor(eqx.Module):
    """Actor that follows a boltzmann (softmax) policy based on the Q values"""

    model: QNetwork

    def __call__(self, x: jax.Array, temperature: jax.Array, key: jax.Array):
        logits = self.model(x)
        # normalize logits, automatic temperature scaling
        logits = logits / (1e-7 + jnp.std(logits))
        return jax.random.categorical(key, logits / temperature)


class Transition(NamedTuple):
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    next_obs: chex.Array
    q_val: chex.Array
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


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    config["NUM_UPDATES_DECAY"] = (
        config["TOTAL_TIMESTEPS_DECAY"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    assert (config["NUM_STEPS"] * config["NUM_ENVS"]) % config[
        "NUM_MINIBATCHES"
    ] == 0, "NUM_MINIBATCHES must divide NUM_STEPS*NUM_ENVS"

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
        rng, _rng = jax.random.split(rng)
        network = QNetwork(rng, obs_size=config["OBS_SIZE"])
        opt = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.radam(learning_rate=lr),
        )

        rng, _rng = jax.random.split(rng)
        opt_state = opt.init(eqx.filter(network, eqx.is_array))
        train_state = TrainState(
            model=network,
            opt=opt,
            opt_state=opt_state,
            timesteps=jnp.array(0),
            n_updates=jnp.array(0),
            grad_steps=jnp.array(0),
        )

        # TRAINING LOOP
        def _update_step(runner_state, unused):

            train_state, obsv, env_state, test_metrics, rng = runner_state

            # SAMPLE PHASE
            def _step_env(carry, _):
                train_state, last_obs, env_state, test_metrics, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                q_vals = train_state.model(last_obs)

                # different eps for each env
                _rngs = jax.random.split(rng_a, config["NUM_ENVS"])
                eps = jnp.full(config["NUM_ENVS"], eps_scheduler(train_state.n_updates))
                new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)

                new_obs, new_env_state, reward, new_done, info = vmap_step(
                    config["NUM_ENVS"]
                )(rng_s, env_state, new_action)

                transition = Transition(
                    obs=last_obs,
                    action=new_action,
                    reward=config.get("REW_SCALE", 1) * reward,
                    done=new_done,
                    next_obs=new_obs,
                    q_val=q_vals,
                    infos=info,
                )
                carry = (train_state, new_obs, new_env_state, test_metrics, rng)
                return carry, transition

            # step the env
            rng, _rng = jax.random.split(rng)
            runner_state, transitions = filter_scan(
                _step_env,
                runner_state,
                None,
                config["NUM_STEPS"],
            )
            train_state, obsv, env_state, test_metrics, rng = runner_state

            train_state = train_state.replace(
                timesteps=train_state.timesteps
                + config["NUM_STEPS"] * config["NUM_ENVS"]
            )  # update timesteps count

            last_q = train_state.model(transitions.next_obs[-1])
            last_q = jnp.max(last_q, axis=-1)

            def _get_target(lambda_returns_and_next_q, transition):
                lambda_returns, next_q = lambda_returns_and_next_q
                target_bootstrap = (
                    transition.reward + config["GAMMA"] * (1 - transition.done) * next_q
                )
                delta = lambda_returns - next_q
                lambda_returns = (
                    target_bootstrap + config["GAMMA"] * config["LAMBDA"] * delta
                )
                lambda_returns = (
                    1 - transition.done
                ) * lambda_returns + transition.done * transition.reward
                next_q = jnp.max(transition.q_val, axis=-1)
                return (lambda_returns, next_q), lambda_returns

            last_q = last_q * (1 - transitions.done[-1])
            lambda_returns = transitions.reward[-1] + config["GAMMA"] * last_q
            _, targets = jax.lax.scan(
                _get_target,
                (lambda_returns, last_q),
                jax.tree.map(lambda x: x[:-1], transitions),
                reverse=True,
            )
            lambda_targets = jnp.concatenate((targets, lambda_returns[np.newaxis]))

            # NETWORKS UPDATE
            def _learn_epoch(carry, _):
                update_state, rng = carry
                train_state, lambda_targets, transitions = update_state

                def _learn_phase(carry, minibatch_and_target):
                    train_state, rng = carry
                    minibatch, target = minibatch_and_target

                    def _loss_fn(network):
                        # (batch_size*2, num_actions)
                        q_vals = network(minibatch.obs)
                        chosen_action_qvals = jnp.take_along_axis(
                            q_vals,
                            jnp.expand_dims(minibatch.action, axis=-1),
                            axis=-1,
                        ).squeeze(axis=-1)

                        loss = 0.5 * jnp.square(chosen_action_qvals - target).mean()

                        return loss, chosen_action_qvals

                    (loss, qvals), grads = eqx.filter_value_and_grad(
                        _loss_fn, has_aux=True
                    )(train_state.model)
                    updates, new_opt_state = train_state.opt.update(
                        grads,
                        train_state.opt_state,
                        eqx.filter(
                            train_state.model, eqx.is_array
                        ),  # TODO is_inexact_array
                    )
                    new_network = eqx.apply_updates(train_state.model, updates)
                    new_train_state = train_state.replace(
                        model=new_network,
                        opt_state=new_opt_state,
                        grad_steps=train_state.grad_steps + 1,
                    )
                    # update_state = (new_train_state, lambda_targets, transitions)
                    return (new_train_state, rng), (loss, qvals)

                def preprocess_transition(x, rng):
                    x = x.reshape(
                        -1, *x.shape[2:]
                    )  # num_steps*num_envs (batch_size), ...
                    x = jax.random.permutation(rng, x)  # shuffle the transitions
                    x = x.reshape(
                        config["NUM_MINIBATCHES"], -1, *x.shape[1:]
                    )  # num_mini_updates, batch_size/num_mini_updates, ...
                    return x

                rng, _rng = jax.random.split(rng)
                minibatches = jax.tree_util.tree_map(
                    lambda x: preprocess_transition(x, _rng), transitions
                )  # num_actors*num_envs (batch_size), ...
                targets = jax.tree.map(
                    lambda x: preprocess_transition(x, _rng), lambda_targets
                )

                rng, _rng = jax.random.split(rng)
                (train_state, rng), (loss, qvals) = filter_scan(
                    _learn_phase, (train_state, rng), (minibatches, targets)
                )
                update_state = (train_state, lambda_targets, transitions)
                return (update_state, rng), (loss, qvals)

            rng, _rng = jax.random.split(rng)
            update_state = (train_state, lambda_targets, transitions)
            (update_state, rng), (loss, qvals) = filter_scan(
                _learn_epoch, (update_state, rng), None, config["NUM_EPOCHS"]
            )
            train_state, lambda_targets, transitions = update_state
            train_state = train_state.replace(n_updates=train_state.n_updates + 1)
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
                    lambda _: get_test_metrics(train_state.model, _rng),
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

            runner_state = (train_state, obsv, env_state, test_metrics, rng)

            return runner_state, metrics

        def get_test_metrics(network, rng):

            if not config.get("TEST_DURING_TRAINING", False):
                return None

            def _env_step(carry, _):
                env_state, last_obs, rng = carry
                rng, _rng = jax.random.split(rng)
                q_vals = network(last_obs)
                eps = jnp.full(config["TEST_NUM_ENVS"], config["EPS_TEST"])
                action = jax.vmap(eps_greedy_exploration)(
                    jax.random.split(_rng, config["TEST_NUM_ENVS"]), q_vals, eps
                )
                new_obs, new_env_state, reward, done, info = vmap_step(
                    config["TEST_NUM_ENVS"]
                )(_rng, env_state, action)
                return (new_env_state, new_obs, rng), info

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = vmap_reset(config["TEST_NUM_ENVS"])(_rng)
            _, infos = filter_scan(
                _env_step, (env_state, init_obs, _rng), None, config["TEST_NUM_STEPS"]
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
        test_metrics = get_test_metrics(train_state.model, _rng)

        rng, _rng = jax.random.split(rng)
        obsv, env_state = vmap_reset(config["NUM_ENVS"])(_rng)

        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, obsv, env_state, test_metrics, _rng)

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

    if config["WANDB_MODE"] != "disabled":
        wandb.init(project=f'{config["PROJECT"]}')
        frames = []
        for i in range(500):
            rng, rng_act, rng_step, _rng = jax.random.split(_rng, 4)
            q_val = model(obs)
            action = jnp.argmax(q_val, axis=-1)
            obs, new_state, reward, term, _ = vmap_step(2)(rng_step, state, action)
            state = new_state
            frame = np.asarray(obs[0])
            frames.append(frame)
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
    alg_name = config.get("ALG_NAME", "PQN")
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
            name=f'{config["TRAIN_TYPE"]}_{config["ENV_NAME"]}_{"SEED="}{config["SEED"]}_{"Partial="}{config["PARTIAL"]}',
            config=config,
            mode=config["WANDB_MODE"],
        )
        wandb.run.log_code(".")

    rng = jax.random.PRNGKey(config["SEED"])

    t0 = time.time()
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = eqx.filter_jit(eqx.filter_vmap(make_train(config)))
    outs = jax.block_until_ready(train_vjit(rngs))
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
        "{}_{}_model_Partial={}_SEED={}.pkl".format(
            config["TRAIN_TYPE"], config["ENV_NAME"], config["PARTIAL"], config["SEED"]
        ),
        network_squeezed,
    )
    rng, _rng = jax.random.split(rng)
    network = QNetwork(rng, obs_size=config["OBS_SIZE"])
    model = eqx.tree_deserialise_leaves(
        "{}_{}_model_Partial={}_SEED={}.pkl".format(
            config["TRAIN_TYPE"], config["ENV_NAME"], config["PARTIAL"], config["SEED"]
        ),
        network,
    )

    evaluate(model, config)


def tune(default_config):
    """Hyperparameter sweep with wandb."""

    default_config = {**default_config, **default_config["alg"]}
    alg_name = default_config.get("ALG_NAME", "pqn")
    env_name = default_config["ENV_NAME"]

    def wrapped_make_train():
        if default_config["WANDB_MODE"] != "disabled":
            wandb.init(project=default_config["PROJECT"])

        config = copy.deepcopy(default_config)
        for k, v in dict(wandb.config).items():
            config[k] = v

        print("running experiment with params:", config)
        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_vjit = jax.jit(jax.vmap(make_train(config)))
        outs = jax.block_until_ready(train_vjit(rngs))
        # return outs

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
                    0.0005,
                    0.0001,
                    0.00005,
                ]
            },
        },
    }

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    wandb.agent(sweep_id, wrapped_make_train, count=1000)


def pqn_run(config):
    if config["HYP_TUNE"]:
        tune(config)
    else:
        single_run(config)

import jax
import pytest

import popgym_arcade
from popgym_arcade.registration import REGISTERED_ENVIRONMENTS


@pytest.mark.parametrize("env_name", REGISTERED_ENVIRONMENTS)
@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("obs_size", [128, 256])
def test_reset_and_step_short(env_name, partial, obs_size):
    env, env_params = popgym_arcade.make(
        env_name, partial_obs=partial, obs_size=obs_size
    )
    reset = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))
    step = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0, None)))

    # Initialize four vectorized environments
    n_envs = 2
    # Initialize PRNG keys
    key = jax.random.key(0)
    reset_keys = jax.random.split(key, n_envs)

    # Reset environments
    observation, env_state = reset(reset_keys, env_params)

    # Step the POMDPs
    for t in range(10):
        # Propagate some randomness
        action_key, step_key = jax.random.split(jax.random.key(t))
        action_keys = jax.random.split(action_key, n_envs)
        step_keys = jax.random.split(step_key, n_envs)
        # Pick actions at random
        actions = jax.vmap(env.action_space(env_params).sample)(action_keys)
        # Step the env to the next state
        # No need to reset, gymnax automatically resets when done
        observation, env_state, reward, done, info = step(
            step_keys, env_state, actions, env_params
        )
        # Check obs space is correct
        assert env.observation_space(env_params).contains(
            observation
        ), "Invalid observation space"


if __name__ == "__main__":
    pytest.main()

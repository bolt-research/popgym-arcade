"""
Compute the average steps per second of the environment.

"""

import csv
import os
import sys
import time

import gymnax
import jax
import jax.numpy as jnp
from jaxlib.xla_extension import XlaRuntimeError

import popgym_arcade

env_name = os.getenv("ENV_NAME", "MineSweeperEasy")
partial_obs = os.getenv("PARTIAL_OBS", "False") == "True"

n_envs = int(os.getenv("NUM_ENVS", 512))
n_steps = int(os.getenv("NUM_STEPS", 512))
seed = jax.random.PRNGKey(0)

# Test FPS for popgym arcade environments
# env, env_params = popgym_arcade.make(env_name, partial_obs=partial_obs)

# Test FPS for gymnax environments
# Source: https://github.com/RobertTLange/gymnax
env, env_params = gymnax.make(env_name)

# jax.config.update("jax_enable_x64", True)


def test_multi_env_fps(
    env=env, env_params=env_params, seed=seed, num_envs=n_envs, num_steps=n_steps
):
    """Test FPS for multiple environments."""
    vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
    vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
    vmap_sample = jax.vmap(env.action_space(env_params).sample)
    seeds = jax.random.split(seed, num_envs)
    obs, states = vmap_reset(seeds, env_params)

    for _ in range(num_steps):
        action = vmap_sample(seeds)
        obs, states, rewards, dones, _ = vmap_step(seeds, states, action, env_params)

    return obs


fps = jax.jit(test_multi_env_fps)
carry = fps(seed=jax.random.PRNGKey(1))
carry.block_until_ready()

start = time.time()
carry = fps(seed=jax.random.PRNGKey(2))
carry.block_until_ready()
end = time.time()

runtime = end - start
fps = n_envs * n_steps / runtime
print(f"env: {env_name}, partial_obs: {partial_obs}")
print(f"time: {end - start}s")
print(f"{env_name} - Multi Env - Envs: {n_envs}, Steps: {n_steps}, FPS: {fps}")
csv_file = "gymnaxfpsdata.csv"
write_header = not os.path.exists(csv_file)
with open(csv_file, mode="a", newline="") as file:
    writer = csv.writer(file)
    if write_header:
        writer.writerow(["Environment", "Partial Obs", "Num Envs", "Num Steps", "FPS"])
    writer.writerow([env_name, partial_obs, n_envs, n_steps, f"{fps:.0f}"])

print("Testing complete. Results appended to fps_results.csv")

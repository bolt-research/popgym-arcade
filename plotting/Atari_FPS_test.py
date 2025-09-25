import csv
import os
import sys
import time

import ale_py
import gymnasium as gym
import jax
import jax.numpy as jnp

import sys

sys.path.extend([""])

# import matplotlib.pyplot as plt

gym.register_envs(ale_py)
n_envs = int(os.getenv("NUM_ENVS", 512))
n_steps = int(os.getenv("NUM_STEPS", 32))
seed = int(os.getenv("SEED"))

env_name = os.getenv("ENV_NAME", "Pong-v4")
env = gym.make_vec(env_name, num_envs=n_envs, vectorization_mode="sync")
env_params = None


def test_multi_env_fps(
    env=env, env_params=env_params, seed=seed, n_envs=n_envs, n_steps=n_steps
):
    """Test FPS for multiple environments."""

    obs, infos = env.reset(seed=seed)

    for _ in range(n_steps):
        _ = env.action_space.seed(seed)
        actions = env.action_space.sample()
        obs, rewards, terminates, truncates, infos = env.step(actions)
        # plt.imshow(obs[0])
        # plt.show()

    return obs


start = time.time()
test_multi_env_fps(env, env_params, seed, n_envs, n_steps)
end = time.time()

runtime = end - start
fps = n_envs * n_steps / runtime
print(f"time: {end - start}s")
print(f"{env_name} - Multi Env - Envs: {n_envs}, Steps: {n_steps}, FPS: {fps}")
csv_file = "atari_fps_results.csv"
write_header = not os.path.exists(csv_file)
with open(csv_file, mode="a", newline="") as file:
    writer = csv.writer(file)
    if write_header:
        writer.writerow(["Environment", "Num Envs", "Num Steps", "FPS", "Seed"])
    writer.writerow([env_name, n_envs, n_steps, f"{fps:.0f}", seed])

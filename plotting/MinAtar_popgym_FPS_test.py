import multiprocessing
import time

# Test FPS for MinAtar environment
# Source: https://github.com/kenjyoung/MinAtar/tree/master
import gymnasium as gym
import popgym
from popgym.envs.battleship import Battleship

TestEnv = gym.make("MinAtar/Asterix-v1")

NUM_STEPS = 512

# Test FPS for popgym environment
# Source: https://github.com/proroklab/popgym
# TestEnv = Battleship()


def run_sample(e, num_steps):
    env = e
    env.reset()
    start = time.time()
    for i in range(num_steps):
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        print(obs.shape)
        if terminated or truncated:
            env.reset()
    end = time.time()
    elapsed = end - start
    fps = num_steps / elapsed
    return fps


def main():
    print(f"Testing environment: {TestEnv}")
    for n in range(1, 10):
        num_workers = 2**n

        # Single environment test (for baseline reference)
        fps_single = run_sample(TestEnv, NUM_STEPS)
        print(f"{TestEnv} (1x) FPS: {fps_single:.0f}")

        with multiprocessing.Pool(processes=num_workers) as p:
            envs = num_workers * [TestEnv]
            steps = num_workers * [int(NUM_STEPS // num_workers)]
            fps_multi = sum(p.starmap(run_sample, zip(envs, steps)))
            print(f"{TestEnv} ({num_workers}x) FPS: {fps_multi:.0f}")


if __name__ == "__main__":
    main()

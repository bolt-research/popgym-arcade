"""
Compute the average steps per second of the environment.

"""
import popgym_arcade
import time
import jax
import numpy as np

seed = jax.random.PRNGKey(0)

env, env_params = popgym_arcade.make("CartPoleEasy", partial_obs=False)

fps_list = []

for _ in range(1000):
    obs, state = env.reset(seed, env_params)
    steps = 0
    start = time.time()
    done = False
    while not done:
        obs, state, reward, done, _ = env.step(seed, state, action=env.action_space(env_params).sample(seed))
        steps += 1
        # state = new_state
    fps = steps / (time.time() - start)
    fps_list.append(fps)
fps_list = np.array(fps_list)
print("Mean steps per second: " + str(np.mean(fps_list)) + " std: " + str(np.std(fps_list)))


# step fucntiono will automatically reset the environment when done is True

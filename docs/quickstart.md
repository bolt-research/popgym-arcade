
# Creating and Stepping Environments
Our tasks are `gymnax` environments and work with wrappers and code designed to work with `gymnax`. The following example demonstrates how to integrate POPGym Arcade into your code. 

```python
import popgym_arcade
import jax

# Create POMDP env variant
env, env_params = popgym_arcade.make("BattleShipEasy", partial_obs=True)

# Let's vectorize and compile the env
# Note when you are training a policy, it is better to compile your policy_update rather than the env_step
reset = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))
step = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0, None)))
    
# Initialize four vectorized environments
n_envs = 4
# Initialize PRNG keys
key = jax.random.key(0)
reset_keys = jax.random.split(key, n_envs)
    
# Reset environments
observation, env_state = reset(reset_keys, env_params)

# Step the POMDP
for t in range(10):
    # Propagate some randomness
    action_key, step_key = jax.random.split(jax.random.key(t))
    action_keys = jax.random.split(action_key, n_envs)
    step_keys = jax.random.split(step_key, n_envs)
    # Pick actions at random
    actions = jax.vmap(env.action_space(env_params).sample)(action_keys)
    # Step the env to the next state
    # No need to reset after initial reset, gymnax automatically resets when done
    observation, env_state, reward, done, info = step(step_keys, env_state, actions, env_params)

# POMDP and MDP variants share states
# We can plug the POMDP states into the MDP and continue playing
mdp, mdp_params = popgym_arcade.make("BattleShipEasy", partial_obs=False)
mdp_reset = jax.jit(jax.vmap(mdp.reset, in_axes=(0, None)))
mdp_step = jax.jit(jax.vmap(mdp.step, in_axes=(0, 0, 0, None)))

action_keys = jax.random.split(jax.random.key(t + 1), n_envs)
step_keys = jax.random.split(jax.random.key(t + 2), n_envs)
markov_state, env_state, reward, done, info = mdp_step(step_keys, env_state, actions, mdp_params)
```
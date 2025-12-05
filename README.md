# POPGym Arcade - GPU-Accelerated MDPs and POMDPs 

POPGym Arcade is a GPU-accelerated Atari-style benchmark and suite of analysis tools for reinforcement learning.

[![Tests](https://github.com/bolt-research/popgym-arcade/actions/workflows/run-tests.yaml/badge.svg)](https://github.com/bolt-research/popgym-arcade/actions/workflows/run-tests.yaml)

## Tasks
POPGym Arcade contains pixel-based tasks in the style of the [Arcade Learning Environment](https://github.com/Farama-Foundation/Arcade-Learning-Environment).

<div style="display: flex; flex-wrap: wrap; gap: 10px; justify-content: space-between; padding: 10px;">
    <img src="imgs/cartpole_f.gif" alt="GIF 1" style="width: 100px; height: 100px;">
    <img src="imgs/cartpole_p.gif" alt="GIF 1" style="width: 100px; height: 100px;">
    <img src="imgs/autoencode_f.gif" alt="GIF 2" style="width: 100px; height: 100px;">
    <img src="imgs/autoencode_p.gif" alt="GIF 2" style="width: 100px; height: 100px;">
    <img src="imgs/breakout_f.gif" alt="GIF 3" style="width: 100px; height: 100px;">
    <img src="imgs/breakout_p.gif" alt="GIF 3" style="width: 100px; height: 100px;">
    <img src="imgs/minesweeper_f.gif" alt="GIF 4" style="width: 100px; height: 100px;">
    <img src="imgs/minesweeper_p.gif" alt="GIF 4" style="width: 100px; height: 100px;">
    <img src="imgs/tetris_f.gif" alt="GIF 5" style="width: 100px; height: 100px;">
    <img src="imgs/tetris_p.gif" alt="GIF 5" style="width: 100px; height: 100px;">
    <img src="imgs/skittles_f.gif" alt="GIF 6" style="width: 100px; height: 100px;">
    <img src="imgs/skittles_p.gif" alt="GIF 6" style="width: 100px; height: 100px;">
    <img src="imgs/navigator_f.gif" alt="GIF 7" style="width: 100px; height: 100px;">
    <img src="imgs/navigator_p.gif" alt="GIF 7" style="width: 100px; height: 100px;">
    <img src="imgs/countrecall_f.gif" alt="GIF 8" style="width: 100px; height: 100px;">
    <img src="imgs/countrecall_p.gif" alt="GIF 8" style="width: 100px; height: 100px;">
    <img src="imgs/battleship_f.gif" alt="GIF 9" style="width: 100px; height: 100px;">
    <img src="imgs/battleship_p.gif" alt="GIF 9" style="width: 100px; height: 100px;">
    <img src="imgs/ncartpole_f.gif" alt="GIF 10" style="width: 100px; height: 100px;">
    <img src="imgs/ncartpole_p.gif" alt="GIF 10" style="width: 100px; height: 100px;">
</div>

Each environment provides:
- Three difficulty settings
- One observation and action space shared across all envs
- Fully observable and partially observable configurations
- Fast and easy GPU vectorization using `jax`
- Standardized returns in `[0,1]` or `[-1, 1]`


### Throughput
Expect ~10M frames per second on an RTX4090. Most of our policies converge in less than 60 minutes of training. 

<img src="imgs/fps.png" height="192" />  
<!-- img src="imgs/wandb.png" height="192" / --> 


## Baselines
We provide a [single training script](popgym_arcade/train.py) for all algorithms and memory models. The [`memax`](https://github.com/smorad/memax) library provides 18 different memory models for use in our script.

**RL Algorithms**
- [PQN](https://arxiv.org/abs/2407.04811) 
- [PPO](https://arxiv.org/abs/1707.06347)
- [DQN](https://arxiv.org/abs/1312.5602)

## Getting Started

To install the environments, run

```bash
pip install popgym-arcade
```
If you plan to use our training scripts, install the baselines as well. If you want to play the games yourself, also use the `human` flag.

```bash
pip install 'popgym-arcade[baselines,human]'
```

**Note:** If you do not already have `jax` installed, we install CPU `jax` by default. For GPU acceleration, run `pip install jax[cuda12]` after installing `popgym-arcade`.

### Human Play
The [play script](popgym_arcade/play.py) installed with `pip install popgym-arcade[human]` lets you play the games yourself using the arrow keys and spacebar.

```bash
popgym-arcade-play NoisyCartPoleEasy        # play MDP 256 pixel version
popgym-arcade-play BattleShipEasy -p -o 128 # play POMDP 128 pixel version
```

### Creating and Stepping Environments
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

## Memory Introspection Tools 
We implement visualization tools to probe which pixels persist in agent memory, and their
impact on Q value predictions. Try the code below or our [example script](plotting/plot_grads.ipynb) to under how your agent uses memory.

<img src="imgs/grads_example.png" height="192" />


```python
from popgym_arcade.baselines.model.builder import QNetworkRNN
from popgym_arcade.baselines.utils import get_saliency_maps, vis_fn
import equinox as eqx
import jax

config = {
    # Env string
    "ENV_NAME": "NavigatorEasy",
    # Whether to use full or partial observability
    "PARTIAL": True,
    # Memory model type (see models directory)
    "MEMORY_TYPE": "lru",
    # Evaluation episode seed
    "SEED": 0,
    # Observation size in pixels (128 or 256)
    "OBS_SIZE": 128,
}

# Initialize the random key
rng = jax.random.PRNGKey(config["SEED"])

# Initialize the model
network = QNetworkRNN(rng, rnn_type=config["MEMORY_TYPE"], obs_size=config["OBS_SIZE"])
# Load the model
model = eqx.tree_deserialise_leaves("PATH_TO_YOUR_MODEL_WEIGHTS.pkl", network)
# Compute the saliency maps
grads, obs_seq, grad_accumulator = get_saliency_maps(rng, model, config)
# Visualize the saliency maps
# If you have latex installed, set use_latex=True
vis_fn(grads, obs_seq, config, use_latex=False)
```

## Other Useful Libraries
- [`stable-gymnax`](https://github.com/smorad/stable-gymnax) - A (stable) `jax`-capable `gymnasium` API
- [`memax`](https://github.com/smorad/memax) - Recurrent models for `jax` 
- [`popgym`](https://github.com/proroklab/popgym) - The original collection of POMDPs, implemented in `numpy`
- [`popjaxrl`](https://github.com/luchris429/popjaxrl) - A `jax` version of `popgym`
- [`popjym`](https://github.com/EdanToledo/popjym) - A more readable version of `popjaxrl` environments that served as a basis for our work

## Citation
```
@article{wang2025popgym,
  title={Investigating Memory in RL with POPGym Arcade},
  author={Wang, Zekang and He, Zhe and Zhang, Borong and Toledo, Edan and Morad, Steven},
  journal={arXiv preprint arXiv:2503.01450},
  year={2025}
}
```

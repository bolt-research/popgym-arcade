# POPGym Arcade - GPU-Accelerated POMDPs 

[![Tests](https://github.com/bolt-research/popgym-arcade/actions/workflows/python_app.yaml/badge.svg)](https://github.com/bolt-research/popgym-arcade/actions/workflows/python_app.yaml)

<div style="display: flex; flex-direction: column; align-items: center; gap: 20px;">
    <div style="display: flex; flex-wrap: wrap; gap: 10px; justify-content: space-between;
                border: 2px solid #3498db; border-radius: 10px; 
                padding: 10px;  
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); 
                background: linear-gradient(135deg, #ffffff, #ffe4e1);">
        <img src="imgs/minesweeper_f.gif" alt="GIF 1" style="width: 100px; height: 100px; border-radius: 5px;">
        <img src="imgs/countrecall_f.gif" alt="GIF 2" style="width: 100px; height: 100px; border-radius: 5px;">
        <img src="imgs/battleship_f.gif" alt="GIF 3" style="width: 100px; height: 100px; border-radius: 5px;">
        <img src="imgs/cartpole_f.gif" alt="GIF 4" style="width: 100px; height: 100px; border-radius: 5px;">
        <img src="imgs/ncartpole_f.gif" alt="GIF 5" style="width: 100px; height: 100px; border-radius: 5px;">
        <img src="imgs/autoencode_f.gif" alt="GIF 6" style="width: 100px; height: 100px; border-radius: 5px;">
        <img src="imgs/navigator_f.gif" alt="GIF 7" style="width: 100px; height: 100px; border-radius: 5px;">
    </div>
    <div style="display: flex; flex-wrap: wrap; gap: 10px; justify-content: space-between;
                border: 2px solid #e74c3c; border-radius: 10px; 
                padding: 10px; 
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); 
                background: linear-gradient(135deg, #ffffff, #ffe4e1);">
        <img src="imgs/minesweeper_p.gif" alt="GIF 1" style="width: 100px; height: 100px; border-radius: 5px;">
        <img src="imgs/countrecall_p.gif" alt="GIF 2" style="width: 100px; height: 100px; border-radius: 5px;">
        <img src="imgs/battleship_p.gif" alt="GIF 3" style="width: 100px; height: 100px; border-radius: 5px;">
        <img src="imgs/cartpole_p.gif" alt="GIF 4" style="width: 100px; height: 100px; border-radius: 5px;">
        <img src="imgs/ncartpole_p.gif" alt="GIF 5" style="width: 100px; height: 100px; border-radius: 5px;">
        <img src="imgs/autoencode_p.gif" alt="GIF 6" style="width: 100px; height: 100px; border-radius: 5px;">
        <img src="imgs/navigator_p.gif" alt="GIF 7" style="width: 100px; height: 100px; border-radius: 5px;">
    </div>
</div>


[//]: # (<p float="left">)

[//]: # (    <img src="imgs/minesweeper_f.gif" width="96" height="96" /> )

[//]: # (    <img src="imgs/countrecall_f.gif" width="96" height="96" /> )

[//]: # (    <img src="imgs/battleship_f.gif" width="96" height="96" /> )

[//]: # (    <img src="imgs/cartpole_f.gif" width="96" height="96" /> )

[//]: # (    <img src="imgs/ncartpole_f.gif" width="96" height="96" /> )

[//]: # (    <img src="imgs/autoencode_f.gif" width="96" height="96" /> )

[//]: # (    <img src="imgs/navigator_f.gif" width="96" height="96" /> )

[//]: # (</p>)

[//]: # ()
[//]: # (<p float="left">)

[//]: # (    <img src="imgs/minesweeper_p.gif" width="96" height="96" /> )

[//]: # (    <img src="imgs/countrecall_p.gif" width="96" height="96" /> )

[//]: # (    <img src="imgs/battleship_p.gif" width="96" height="96" /> )

[//]: # (    <img src="imgs/cartpole_p.gif" width="96" height="96" /> )

[//]: # (    <img src="imgs/ncartpole_p.gif" width="96" height="96" /> )

[//]: # (    <img src="imgs/autoencode_p.gif" width="96" height="96" /> )

[//]: # (    <img src="imgs/navigator_p.gif" width="96" height="96" /> )

[//]: # (</p>)

POPGym Arcade contains 7 pixel-based POMDPs in the style of the [Arcade Learning Environment](https://github.com/Farama-Foundation/Arcade-Learning-Environment). Each environment provides:
- 3 Difficulty settings
- Common observation and action space shared across all envs
- Fully observable and partially observable configurations
- Fast and easy GPU vectorization using `jax.vmap` and `jax.jit`

## Gradient Visualization
We also provide tools to visualize how policies use memory. 
<img src="imgs/grads_example.jpg" height="192" />

See [below](#Memory-Introspection-Tools) for further instructions.

## Throughput
You can expect millions of frames per second on a consumer-grade GPU. With `obs_size=128`, most policies converge within 30-60 minutes of training. 

<img src="imgs/fps.png" height="192" />  
<img src="imgs/wandb.png" height="192" /> 

## Getting Started


### Installation 

To install the environments, run

```bash
pip install popgym-arcade
```
If you plan to use our training scripts, install the baselines as well

```bash
pip install 'popgym-arcade[baselines]'
```

### Human Play
To best understand the environments, you should try and play them yourself. The [play script](popgym_arcade/play.py) lets you play the games yourself using the arrow keys and spacebar.

```bash
popgym-arcade-play NoisyCartPoleEasy        # play MDP 256 pixel version
popgym-arcade-play BattleShipEasy -p -o 128 # play POMDP 128 pixel version
```

### Creating and Stepping Environments
Our envs are `gymnax` envs, so you can use your wrappers and code designed to work with `gymnax`. The following example demonstrates how to integrate POPGym Arcade into your code. 

```python
import popgym_arcade
import jax

# Create both POMDP and MDP env variants
pomdp, pomdp_params = popgym_arcade.make("BattleShipEasy", partial_obs=True)
mdp, mdp_params = popgym_arcade.make("BattleShipEasy", partial_obs=False)

# Let's vectorize and compile the envs
# Note when you are training a policy, it is better to compile your policy_update rather than the env_step
pomdp_reset = jax.jit(jax.vmap(pomdp.reset, in_axes=(0, None)))
pomdp_step = jax.jit(jax.vmap(pomdp.step, in_axes=(0, 0, 0, None)))
mdp_reset = jax.jit(jax.vmap(mdp.reset, in_axes=(0, None)))
mdp_step = jax.jit(jax.vmap(mdp.step, in_axes=(0, 0, 0, None)))
    
# Initialize four vectorized environments
n_envs = 4
# Initialize PRNG keys
key = jax.random.key(0)
reset_keys = jax.random.split(key, n_envs)
    
# Reset environments
observation, env_state = pomdp_reset(reset_keys, pomdp_params)

# Step the POMDPs
for t in range(10):
    # Propagate some randomness
    action_key, step_key = jax.random.split(jax.random.key(t))
    action_keys = jax.random.split(action_key, n_envs)
    step_keys = jax.random.split(step_key, n_envs)
    # Pick actions at random
    actions = jax.vmap(pomdp.action_space(pomdp_params).sample)(action_keys)
    # Step the env to the next state
    # No need to reset, gymnax automatically resets when done
    observation, env_state, reward, done, info = pomdp_step(step_keys, env_state, actions, pomdp_params)

# POMDP and MDP variants share states
# We can plug the POMDP states into the MDP and continue playing 
action_keys = jax.random.split(jax.random.key(t + 1), n_envs)
step_keys = jax.random.split(jax.random.key(t + 2), n_envs)
markov_state, env_state, reward, done, info = mdp_step(step_keys, env_state, actions, mdp_params)
```

## Memory Introspection Tools 
We implement visualization tools to probe which pixels persist in agent memory, and their
impact on Q value predictions. Try code below or [vis example](plotting/plot_grads.ipynb) to visualize the memory your agent uses

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
- [`gymnax`](https://github.com/RobertTLange/gymnax) - The (deprecated) `jax`-capable `gymnasium` API
- [`stable-gymnax`](https://github.com/smorad/stable-gymnax) - A maintained and patched version of `gymnax`
- [`popgym`](https://github.com/proroklab/popgym) - The original collection of POMDPs, implemented in `numpy`
- [`popjaxrl`](https://github.com/luchris429/popjaxrl) - A `jax` version of `popgym`
- [`popjym`](https://github.com/EdanToledo/popjym) - A more readable version of `popjaxrl` environments that served as a basis for our work

## Citation
```
@article{wang2025popgym,
  title={POPGym Arcade: Parallel Pixelated POMDPs},
  author={Wang, Zekang and He, Zhe and Zhang, Borong and Toledo, Edan and Morad, Steven},
  journal={arXiv preprint arXiv:2503.01450},
  year={2025}
}
```

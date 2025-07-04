import equinox as eqx
import jax
import jax.numpy as jnp

import popgym_arcade
from popgym_arcade.baselines.model.builder import QNetworkRNN
from popgym_arcade.baselines.utils import get_saliency_maps, vis_fn
from popgym_arcade.wrappers import LogWrapper

##
## Simpler approach:
## Compute gradients using random initial state
##
config = {
    "ENV_NAME": "MineSweeperEasy",
    "PARTIAL": False,
    "MEMORY_TYPE": "lru",
    "SEED": 0,
    "OBS_SIZE": 128,
}
# Path to your model weights
config["MODEL_PATH"] = (
    f"nips_analysis_128/PQN_RNN_{config['MEMORY_TYPE']}_{config['ENV_NAME']}_model_Partial={config['PARTIAL']}_SEED=0.pkl"
)

# Initialize the random key
rng = jax.random.PRNGKey(config["SEED"])

# Initialize the model
network = QNetworkRNN(rng, rnn_type=config["MEMORY_TYPE"], obs_size=config["OBS_SIZE"])
# Load the model
model = eqx.tree_deserialise_leaves(config["MODEL_PATH"], network)
# Compute the saliency maps
grads, obs_seq, grad_accumulator = get_saliency_maps(rng, model, config, max_steps=30)
# Visualize the saliency maps
vis_fn(grads, obs_seq, config, use_latex=True)


##
## More complex approach
## Generate custom initial state and then compute gradients
##
config = {
    "ENV_NAME": "NavigatorEasy",
    "PARTIAL": True,
    "MEMORY_TYPE": "lru",
    "SEED": 0,
    "OBS_SIZE": 128,
}
config["MODEL_PATH"] = (
    f"nips_analysis_128/PQN_RNN_{config['MEMORY_TYPE']}_{config['ENV_NAME']}_model_Partial={config['PARTIAL']}_SEED=0.pkl"
)


rng = jax.random.PRNGKey(config["SEED"])
# Initialize the model
network = QNetworkRNN(rng, rnn_type=config["MEMORY_TYPE"], obs_size=config["OBS_SIZE"])
# Load the model
model = eqx.tree_deserialise_leaves(config["MODEL_PATH"], network)

# Setup initial state
seed, _rng = jax.random.split(jax.random.key(config["SEED"]))
env, env_params = popgym_arcade.make(
    config["ENV_NAME"], partial_obs=config["PARTIAL"], obs_size=config["OBS_SIZE"]
)
env = LogWrapper(env)
n_envs = 1
vmap_reset = lambda n_envs: lambda rng: jax.vmap(env.reset, in_axes=(0, None))(
    jax.random.split(rng, n_envs), env_params
)
vmap_step = lambda n_envs: lambda rng, env_state, action: jax.vmap(
    env.step, in_axes=(0, 0, 0, None)
)(jax.random.split(rng, n_envs), env_state, action, env_params)
init_obs, init_state = vmap_reset(n_envs)(_rng)

# Replace initial state with custom initial state
new_init_state = eqx.tree_at(
    lambda x: x.env_state.action_x, init_state, replace=jnp.array([6])
)
new_init_state = eqx.tree_at(
    lambda x: x.env_state.action_y, new_init_state, replace=jnp.array([6])
)
board = (
    new_init_state.env_state.board.at[jnp.where(new_init_state.env_state.board == 2)]
    .set(0)
    .at[:, 1, 1]
    .set(2)
)
# Can also set the entire board manually if needed
# board = (
#     jnp.zeros_like(new_init_state.env_state.board)
#     # tnt
#     .at[0, 3, 2].set(1)
#     .at[0, 4, 2].set(1)
#     .at[0, 5, 3].set(1)
#     .at[0, 6, 3].set(1)
#     # goal
#     .at[0, 6, 6].set(2)
# )
new_init_state = eqx.tree_at(lambda x: x.env_state.board, new_init_state, replace=board)
new_init_obs = jax.vmap(env.get_obs)(new_init_state.env_state)


# Compute the saliency maps
grads, obs_seq, grad_accumulator = get_saliency_maps(
    rng,
    model,
    config,
    max_steps=10,
    initial_state_and_obs=(new_init_state, new_init_obs),
)
# Visualize the saliency maps
vis_fn(grads, obs_seq, config, use_latex=True)

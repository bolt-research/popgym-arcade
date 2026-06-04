
# Memory Introspection Tools 
We implement visualization tools to probe which pixels persist in agent memory, and their
impact on Q value predictions. Try the code below to under how your agent uses memory.

<img src="../imgs/grads_example.png" height="192" />


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

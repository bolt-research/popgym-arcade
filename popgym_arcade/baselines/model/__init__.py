from popgym_arcade.baselines.model.builder import (
    ActorCritic,
    ActorCriticRNN,
    QNetwork,
    QNetworkRNN,
)

from memorax.equinox.train_utils import (
    add_batch_dim,
    get_residual_memory_models,
)
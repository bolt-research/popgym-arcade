from typing import Callable, Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Shaped

import popgym_arcade.baselines.model.memorax.groups as groups
from popgym_arcade.baselines.model.memorax.magmas import (
    GRU,
    LSTM,
    MGU,
    Elman,
    Spherical,
)
from popgym_arcade.baselines.model.memorax.models import ResidualModel
from popgym_arcade.baselines.model.memorax.semigroups import (
    FART,
    GILR,
    LRU,
    MLP,
    LinearRecurrent,
    LogBayes,
    MinGRU,
    NAbs,
    NMax,
    PSpherical,
)


def add_batch_dim(h, batch_size: int, axis: int = 0) -> Shaped[Array, "Batch ..."]:
    """Given an recurrent state (pytree) `h`, add a new batch dimension of size `batch_size`.

    E.g., add_batch_dim(h, 32) will return a new state with shape (32, *h.shape). The state will
    be repeated along the new batch dimension.
    """
    expand = lambda x: jnp.repeat(jnp.expand_dims(x, axis), batch_size, axis=axis)
    h = jax.tree.map(expand, h)
    return h


def cross_entropy(
    y_hat: Shaped[Array, "Batch ... Classes"], y: Shaped[Array, "Batch ... Classes"]
) -> Shaped[Array, "1"]:
    return -jnp.mean(jnp.sum(y * jax.nn.log_softmax(y_hat, axis=-1), axis=-1))


def accuracy(
    y_hat: Shaped[Array, "Batch ... Classes"], y: Shaped[Array, "Batch ... Classes"]
) -> Shaped[Array, "1"]:
    return jnp.mean(jnp.argmax(y, axis=-1) == jnp.argmax(y_hat, axis=-1))


def update_model(
    model: groups.Module,
    loss_fn: Callable,
    opt: optax.GradientTransformation,
    opt_state: optax.OptState,
    x: Shaped[Array, "Batch ..."],
    y: Shaped[Array, "Batch ..."],
    key=None,
) -> Tuple[groups.Module, optax.OptState, Dict[str, Array]]:
    """Update the model using the given loss function and optimizer."""
    grads, loss_info = eqx.filter_grad(loss_fn, has_aux=True)(model, x, y, key)
    updates, opt_state = opt.update(
        grads, opt_state, params=eqx.filter(model, eqx.is_inexact_array)
    )
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_info


@eqx.filter_jit
def scan_one_epoch(
    model: groups.Module,
    opt: optax.GradientTransformation,
    opt_state: optax.OptState,
    loss_fn: Callable,
    xs: Shaped[Array, "Datapoint ..."],
    ys: Shaped[Array, "Datapoint ..."],
    batch_size: int,
    batch_index: Shaped[Array, "Batch ..."],
    *,
    key: jax.random.PRNGKey,
) -> Tuple[groups.Module, optax.OptState, Dict[str, Array]]:
    """Train a single epoch using the scan operator. Functions as a dataloader and train loop."""
    assert (
        xs.shape[0] == ys.shape[0]
    ), f"batch size mismatch: {xs.shape[0]} != {ys.shape[0]}"
    params, static = eqx.partition(model, eqx.is_array)

    def get_batch(x, y, step):
        """Returns a specific batch of size `batch_size` from `x` and `y`."""
        start = step * batch_size
        x_batch = jax.lax.dynamic_slice_in_dim(x, start, batch_size, 0)
        y_batch = jax.lax.dynamic_slice_in_dim(y, start, batch_size, 0)
        return x_batch, y_batch

    def inner(carry, index):
        params, opt_state, key = carry
        x, y = get_batch(xs, ys, index)
        key = jax.random.split(key)[0]
        model = eqx.combine(params, static)
        # JIT this otherwise it takes ages to compile the epoch
        params, opt_state, metrics = update_model(
            model, loss_fn, opt, opt_state, x, y, key=key
        )
        params, _ = eqx.partition(params, eqx.is_array)
        return (params, opt_state, key), metrics

    (params, opt_state, key), epoch_metrics = jax.lax.scan(
        inner,
        (params, opt_state, key),
        batch_index,
    )
    model = eqx.combine(params, static)
    return model, opt_state, epoch_metrics


def get_monoids(
    recurrent_size: int,
    key: jax.random.PRNGKey,
) -> Dict[str, groups.Module]:
    return {
        # "double": DoubleMonoid(recurrent_size),
        # "pspherical": PSphericalMonoid(recurrent_size),
        # "ffm": FFMSemigroup(recurrent_size, recurrent_size, recurrent_size, key=key),
        # "nlse": NLSEMonoid(recurrent_size),
        # "fart": FARTSemigroup(recurrent_size),
        # "lru": LRUSemigroup(recurrent_size),
        # "tslru": TSLRUMonoid(recurrent_size),
        # "nabs": NAbsSemigroup(recurrent_size),
        # "nmax": NMaxMonoid(recurrent_size),
        # "nbroken": NBrokenMonoid(recurrent_size, key=key),
        # "linear_rnn": LinearRNNSemigroup(recurrent_size),
        # "gilr": GILRSemigroup(recurrent_size),
        # "log_bayes": LogBayesSemigroup(recurrent_size),
    }


def get_residual_memory_model(
    input: int,
    hidden: int,
    output: int,
    num_layers: int = 2,
    rnn_type: str = "lru",
    *,
    key: jax.random.PRNGKey,
) -> groups.Module:
    layers = {
        "nabs": lambda recurrent_size, key: NAbs(
            recurrent_size=recurrent_size, key=key
        ),
        "nmax": lambda recurrent_size, key: NMax(
            recurrent_size=recurrent_size, key=key
        ),
        "fart": lambda recurrent_size, key: FART(
            hidden_size=recurrent_size,
            recurrent_size=round(recurrent_size**0.5),
            key=key,
        ),
        "pspherical": lambda recurrent_size, key: PSpherical(
            recurrent_size=round(recurrent_size**0.5),
            hidden_size=recurrent_size,
            key=key,
        ),
        "lru": lambda recurrent_size, key: LRU(
            hidden_size=recurrent_size, recurrent_size=recurrent_size, key=key
        ),
        "linear_rnn": lambda recurrent_size, key: LinearRecurrent(
            recurrent_size=recurrent_size, key=key
        ),
        "gilr": lambda recurrent_size, key: GILR(
            recurrent_size=recurrent_size, key=key
        ),
        "log_bayes": lambda recurrent_size, key: LogBayes(
            recurrent_size=recurrent_size, key=key
        ),
        "mingru": lambda recurrent_size, key: MinGRU(
            recurrent_size=recurrent_size, key=key
        ),
        "mlp": lambda recurrent_size, key: MLP(recurrent_size=recurrent_size, key=key),
        # magmas
        "gru": lambda recurrent_size, key: GRU(recurrent_size=recurrent_size, key=key),
        "elman": lambda recurrent_size, key: Elman(
            hidden_size=recurrent_size, recurrent_size=recurrent_size, key=key
        ),
        "ln_elman": lambda recurrent_size, key: Elman(
            hidden_size=recurrent_size,
            recurrent_size=recurrent_size,
            ln_variant=True,
            key=key,
        ),
        "spherical": lambda recurrent_size, key: Spherical(
            hidden_size=recurrent_size, recurrent_size=recurrent_size, key=key
        ),
        "mgu": lambda recurrent_size, key: MGU(recurrent_size=recurrent_size, key=key),
        "lstm": lambda recurrent_size, key: LSTM(
            recurrent_size=recurrent_size, key=key
        ),
    }
    return ResidualModel(
        make_layer_fn=layers[rnn_type],
        input_size=input,
        recurrent_size=hidden,
        output_size=output,
        num_layers=num_layers,
        key=key,
    )

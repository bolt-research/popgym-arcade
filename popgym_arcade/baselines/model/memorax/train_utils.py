from functools import partial
from typing import Callable, Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Shaped

# import memorax
import popgym_arcade.baselines.model.memorax.groups as groups
from popgym_arcade.baselines.model.memorax.magmas.elman import Elman
from popgym_arcade.baselines.model.memorax.magmas.gru import GRU
from popgym_arcade.baselines.model.memorax.magmas.mgu import MGU
from popgym_arcade.baselines.model.memorax.magmas.spherical import Spherical
from popgym_arcade.baselines.model.memorax.models.residual import ResidualModel
from popgym_arcade.baselines.model.memorax.semigroups.bayes import LogBayes, LogBayesSemigroup
from popgym_arcade.baselines.model.memorax.semigroups.fart import FART, FARTSemigroup
from popgym_arcade.baselines.model.memorax.semigroups.ffm import FFM, FFMSemigroup
from popgym_arcade.baselines.model.memorax.semigroups.gilr import GILR, GILRSemigroup
from popgym_arcade.baselines.model.memorax.semigroups.lrnn import LinearRecurrent, LinearRNNSemigroup
from popgym_arcade.baselines.model.memorax.semigroups.lru import LRU, LRUSemigroup
from popgym_arcade.baselines.model.memorax.semigroups.mingru import MinGRU
from popgym_arcade.baselines.model.memorax.semigroups.nabs import NAbs, NAbsSemigroup
from popgym_arcade.baselines.model.memorax.semigroups.nbroken import NBroken, NBrokenMonoid
from popgym_arcade.baselines.model.memorax.semigroups.mlp import MLP
# from semigroups.nlse import NLSE, NLSEMonoid
from popgym_arcade.baselines.model.memorax.semigroups.nmax import NMax, NMaxMonoid
# from semigroups.plru import PLRU
from popgym_arcade.baselines.model.memorax.semigroups.spherical import PSpherical, PSphericalMonoid
# from semigroups.tests import DoubleMonoid
from popgym_arcade.baselines.model.memorax.semigroups.s6 import S6, S6Monoid


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


# def monoid_associative_loss(
#     model: memorax.groups.Module,
#     x: Shaped[Array, "Batch Time Feature"],
# ) -> Tuple[Shaped[Array, "1"], Dict[str, Array]]:
#     x1 = x[:, :-2]
#     x2 = x[:, 1:-1]
#     x3 = x[:, 2:]
#
#     a = monoid(monoid(x1, x2), x3)
#     b = monoid(x1, monoid(x2, x3))
#     return jnp.mean(jnp.square(a - b))


# def loss_classify_terminal_output(
#     model: memorax.groups.Module,
#     x: Shaped[Array, "Batch Time Feature"],
#     y: Shaped[Array, "Batch Classes"],
#     key = None
# ) -> Tuple[Shaped[Array, "1"], Dict[str, Array]]:
#     """Given a sequence of inputs x1, ..., xn and predicted outputs y1p, ..., y1n,
#     return the cross entropy loss between the true yn and predicted y1n.
#
#     Args:
#         model: memorax.groups.Module
#         x: (batch, time, in_feature)
#         y: (batch, num_classes)
#
#     Returns:
#         loss: scalar
#         info: dict
#     """
#     batch_size = x.shape[0]
#     seq_len = x.shape[1]
#     assert (
#         x.shape[0] == y.shape[0]
#     ), f"batch size mismatch: {x.shape[0]} != {y.shape[0]}"
#     assert x.ndim == 3, f"expected 3d input, got {x.ndim}d"
#     assert y.ndim == 2, f"expected 2d input, got {y.ndim}d"
#
#     starts = jnp.zeros((batch_size, seq_len), dtype=bool)
#     key, init_key, model_key = jax.random.split(key, 3)
#     init_key = jax.random.split(init_key, batch_size)
#     # TODO: These all initialize in the same state, probably do not want this
#     h0 = eqx.filter_vmap(model.initialize_carry)(init_key)
#
#     model_key = jax.random.split(model_key, batch_size)
#
#     _, y_preds = eqx.filter_vmap(model)(h0, (x, starts), model_key)
#     # batch, time, feature
#     y_pred = y_preds[:, -1]
#
#     loss = cross_entropy(y_pred, y)
#     acc = accuracy(y_pred, y)
#     return loss, {"loss": loss, "accuracy": acc}
#
# def loss_variational_classify_terminal_output(
#     model: memorax.groups.Module,
#     x: Shaped[Array, "Batch Time Feature"],
#     y: Shaped[Array, "Batch Classes"],
#     kl_weight: 0.01,
#     key = None
# ) -> Tuple[Shaped[Array, "1"], Dict[str, Array]]:
#     """Given a sequence of inputs x1, ..., xn and predicted outputs y1p, ..., y1n,
#     return the cross entropy loss between the true yn and predicted y1n.
#
#     Args:
#         model: memorax.groups.Module
#         x: (batch, time, in_feature)
#         y: (batch, num_classes)
#
#     Returns:
#         loss: scalar
#         info: dict
#     """
#     batch_size = x.shape[0]
#     seq_len = x.shape[1]
#     assert (
#         x.shape[0] == y.shape[0]
#     ), f"batch size mismatch: {x.shape[0]} != {y.shape[0]}"
#     assert x.ndim == 3, f"expected 3d input, got {x.ndim}d"
#     assert y.ndim == 2, f"expected 2d input, got {y.ndim}d"
#
#     starts = jnp.zeros((batch_size, seq_len), dtype=bool)
#     key, init_key, model_key = jax.random.split(key, 3)
#     init_key = jax.random.split(init_key, batch_size)
#     # TODO: These all initialize in the same state, probably do not want this
#     #h0 = add_batch_dim(model.initialize_carry(init_key), batch_size)
#     h0 = eqx.filter_vmap(model.initialize_carry)(init_key)
#
#     model_key = jax.random.split(model_key, batch_size)
#
#     states, y_preds = eqx.filter_vmap(model)(h0, (x, starts), model_key)
#     # batch, time, feature
#     y_pred = y_preds[:, -1]
#
#     loss = cross_entropy(y_pred, y)
#     #kl_loss = kl_weight * 0.5 *
#     acc = accuracy(y_pred, y)
#     return loss, {"loss": loss, "accuracy": acc}
#
#
#

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


def get_residual_memory_models(
    input: int,
    hidden: int,
    output: int,
    num_layers: int = 2,
    *,
    key: jax.random.PRNGKey,
) -> Dict[str, groups.Module]:
    layers = {
        # monoids
        # "nbroken": lambda recurrent_size, key: NBroken(
        #     recurrent_size=recurrent_size, key=key
        # ),
        # "nabs": lambda recurrent_size, key: NAbs(
        #     recurrent_size=recurrent_size, key=key
        # ),
        # "nmax": lambda recurrent_size, key: NMax(
        #     recurrent_size=recurrent_size, key=key
        # ),
        "fart": lambda recurrent_size, key: FART(
           hidden_size=recurrent_size, recurrent_size=round(recurrent_size ** 0.5), key=key
        ),
        # "S6": lambda recurrent_size, key: S6(
        #     hidden_size=recurrent_size, recurrent_size=recurrent_size, key=key
        # ),
        # "mingru": lambda recurrent_size, key: MinGRU(
        #     recurrent_size=recurrent_size, key=key
        # ),
        # "pspherical": lambda recurrent_size, key: PSpherical(
        #     recurrent_size=round(recurrent_size ** 0.5),
        #     hidden_size=recurrent_size,
        #     key=key
        # ),
        # "plru": lambda recurrent_size, key: PLRU(
        #     hidden_size=recurrent_size, recurrent_size=recurrent_size, key=key
        # ),
        # # "lru": lambda recurrent_size, key: LRU(
        # #     hidden_size=recurrent_size, recurrent_size=recurrent_size, key=key
        # # ),
        # "linear_rnn": lambda recurrent_size, key: LinearRecurrent(
        #     recurrent_size=recurrent_size, key=key
        # ),
        # "gilr": lambda recurrent_size, key: GILR(
        #     recurrent_size=recurrent_size, key=key
        # ),
        # "log_bayes": lambda recurrent_size, key: LogBayes(
        #     recurrent_size=round(recurrent_size ** 0.5), key=key
        # ),
        # # magmas
        # "gru": lambda recurrent_size, key: GRU(recurrent_size=recurrent_size, key=key),
        # # "elman": lambda recurrent_size, key: Elman(
        # #    hidden_size=recurrent_size, recurrent_size=recurrent_size, key=key
        # # ),
        # # "ln_elman": lambda recurrent_size, key: Elman(
        # #    hidden_size=recurrent_size,
        # #    recurrent_size=recurrent_size,
        # #    ln_variant=True,
        # #    key=key,
        # # ),
        # "spherical": lambda recurrent_size, key: Spherical(
        #     hidden_size=recurrent_size, recurrent_size=recurrent_size, key=key
        # ),
        # "mgu": lambda recurrent_size, key: MGU(recurrent_size=recurrent_size, key=key),
    }
    return {
        name: ResidualModel(
            make_layer_fn=fn,
            input_size=input,
            recurrent_size=hidden,
            output_size=output,
            num_layers=num_layers,
            key=key,
        )
        for name, fn in layers.items()
    }


def get_residual_memory_model(
    input: int,
    hidden: int,
    output: int,
    num_layers: int = 2,
    rnn_type: str = "lru",
    *,
    key: jax.random.PRNGKey
) -> groups.Module:

    layers = {
        "nabs": lambda recurrent_size, key: NAbs(
            recurrent_size=recurrent_size, key=key
        ),
        "nmax": lambda recurrent_size, key: NMax(
            recurrent_size=recurrent_size, key=key
        ),
        "fart": lambda recurrent_size, key: FART(
           hidden_size=recurrent_size, recurrent_size=round(recurrent_size ** 0.5), key=key
        ),
        "pspherical": lambda recurrent_size, key: PSpherical(
            recurrent_size=round(recurrent_size ** 0.5),
            hidden_size=recurrent_size,
            key=key
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
        "mingru": lambda recurrent_size, key: MinGRU(recurrent_size=recurrent_size, key=key),
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
        # "lstm": lambda recurrent_size, key: LSTM(recurrent_size=recurrent_size, key=key),
    }
    return ResidualModel(
        make_layer_fn=layers[rnn_type],
        input_size=input,
        recurrent_size=hidden,
        output_size=output,
        num_layers=num_layers,
        key=key,
    )

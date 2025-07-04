from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from equinox import nn
from jaxtyping import Array, Float, PRNGKeyArray, Shaped, jaxtyped

from popgym_arcade.baselines.model.memorax.gras import GRAS
from popgym_arcade.baselines.model.memorax.groups import (
    BinaryAlgebra,
    Resettable,
    SetAction,
)
from popgym_arcade.baselines.model.memorax.mtypes import (
    Input,
    InputEmbedding,
    StartFlag,
)
from popgym_arcade.baselines.model.memorax.scans import set_action_scan

LSTMRecurrentState = Tuple[Float[Array, "Recurrent"], Float[Array, "Recurrent"]]
LSTMRecurrentStateWithReset = Tuple[LSTMRecurrentState, StartFlag]


class LSTMMagma(SetAction):
    """
    The Long Short-Term Memory Magma

    Paper: https://www.bioinf.jku.at/publications/older/2604.pdf
    """

    recurrent_size: int
    U_z: nn.Linear
    U_r: nn.Linear
    U_h: nn.Linear
    W_z: nn.Linear
    W_r: nn.Linear
    W_h: nn.Linear

    def __init__(self, recurrent_size: int, key):
        self.recurrent_size = recurrent_size
        keys = jax.random.split(key, 8)
        self.U_f = nn.Linear(
            recurrent_size, recurrent_size, use_bias=False, key=keys[0]
        )
        self.U_i = nn.Linear(
            recurrent_size, recurrent_size, use_bias=False, key=keys[1]
        )
        self.U_o = nn.Linear(
            recurrent_size, recurrent_size, use_bias=False, key=keys[2]
        )
        self.U_c = nn.Linear(
            recurrent_size, recurrent_size, use_bias=False, key=keys[3]
        )

        self.W_f = nn.Linear(recurrent_size, recurrent_size, key=keys[4])
        self.W_i = nn.Linear(recurrent_size, recurrent_size, key=keys[5])
        self.W_o = nn.Linear(recurrent_size, recurrent_size, key=keys[6])
        self.W_c = nn.Linear(recurrent_size, recurrent_size, key=keys[7])

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, carry: LSTMRecurrentState, input: Float[Array, "Recurrent"]
    ) -> LSTMRecurrentState:
        c, h = carry
        f_f = jax.nn.sigmoid(self.W_f(input) + self.U_f(h))
        f_i = jax.nn.sigmoid(self.W_i(input) + self.U_i(h))
        f_o = jax.nn.sigmoid(self.W_o(input) + self.U_o(h))
        f_c = jax.nn.sigmoid(self.W_c(input) + self.U_c(h))

        c = f_f * c + f_i * f_c
        h = f_o * c

        return (c, h)

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> LSTMRecurrentState:
        return (
            jnp.zeros((self.recurrent_size,)),
            jnp.zeros((self.recurrent_size,)),
        )


class LSTM(GRAS):
    """
    The Long Short-Term Memory

    Paper: https://www.bioinf.jku.at/publications/older/2604.pdf
    """

    algebra: BinaryAlgebra
    scan: Callable[
        [
            Callable[
                [LSTMRecurrentStateWithReset, LSTMRecurrentStateWithReset],
                LSTMRecurrentStateWithReset,
            ],
            LSTMRecurrentStateWithReset,
            LSTMRecurrentStateWithReset,
        ],
        LSTMRecurrentStateWithReset,
    ]

    def __init__(self, recurrent_size, key):
        keys = jax.random.split(key, 3)
        self.algebra = Resettable(LSTMMagma(recurrent_size, key=keys[0]))
        self.scan = set_action_scan

    @jaxtyped(typechecker=typechecker)
    def forward_map(
        self, x: Input, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> LSTMRecurrentStateWithReset:
        emb, start = x
        return emb, start

    @jaxtyped(typechecker=typechecker)
    def backward_map(
        self,
        h: LSTMRecurrentStateWithReset,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) -> Float[Array, "{self.hidden_size}"]:
        z, reset_flag = h
        emb, start = x
        return z

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> LSTMRecurrentState:
        return self.algebra.initialize_carry(key)

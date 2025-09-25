from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from equinox import nn
from jaxtyping import Array, Float, PRNGKeyArray, Shaped, jaxtyped

from ..gras import GRAS
from ..groups import BinaryAlgebra, Resettable, Semigroup
from ..mtypes import Input, StartFlag
from ..scans import semigroup_scan

GILRRecurrentState = Float[Array, "Hidden"]
GILRRecurrentStateWithReset = Tuple[GILRRecurrentState, StartFlag]


class GILRSemigroup(Semigroup):
    recurrent_size: int

    def __init__(self, recurrent_size: int):
        self.recurrent_size = recurrent_size

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> GILRRecurrentState:
        return jnp.zeros((self.recurrent_size,))

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, carry: GILRRecurrentState, input: GILRRecurrentState
    ) -> GILRRecurrentState:
        return carry + input


class GILR(GRAS):
    """A Gated Impulse Linear Recurrent layer.

    You might want to use this as a building block for a more complex model.
    """

    recurrent_size: int
    scan: Callable[
        [
            Callable[
                [GILRRecurrentStateWithReset, GILRRecurrentStateWithReset],
                GILRRecurrentStateWithReset,
            ],
            GILRRecurrentStateWithReset,
            GILRRecurrentStateWithReset,
        ],
        GILRRecurrentStateWithReset,
    ]
    algebra: BinaryAlgebra

    g: nn.Sequential
    i: nn.Sequential

    def __init__(self, recurrent_size, key):
        self.recurrent_size = recurrent_size
        self.algebra = Resettable(GILRSemigroup(recurrent_size))
        self.scan = semigroup_scan

        keys = jax.random.split(key)

        self.g = nn.Sequential(
            [
                nn.Linear(recurrent_size, recurrent_size, key=keys[1]),
                nn.Lambda(jax.nn.sigmoid),
            ]
        )
        self.i = nn.Sequential(
            [
                nn.Linear(recurrent_size, recurrent_size, key=keys[1]),
                nn.Lambda(jax.nn.leaky_relu),
            ]
        )

    @jaxtyped(typechecker=typechecker)
    def forward_map(
        self, x: Input, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> GILRRecurrentStateWithReset:
        emb, start = x
        z = emb
        return z, start

    @jaxtyped(typechecker=typechecker)
    def backward_map(
        self,
        h: GILRRecurrentStateWithReset,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) -> Float[Array, "{self.recurrent_size}"]:
        emb, start = x
        state, reset_carry = h
        g = self.g(emb)
        return g * state + (1 - g) * emb

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> GILRRecurrentStateWithReset:
        return self.algebra.initialize_carry(key)

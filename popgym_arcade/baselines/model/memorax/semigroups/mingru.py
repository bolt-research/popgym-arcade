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

MinGRURecurrentState = Tuple[Float[Array, "Hidden"], Float[Array, "Hidden"]]
MinGRURecurrentStateWithReset = Tuple[MinGRURecurrentState, StartFlag]


class MinGRUSemigroup(Semigroup):
    recurrent_size: int

    def __init__(self, recurrent_size: int):
        self.recurrent_size = recurrent_size

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> MinGRURecurrentState:
        return (
            jnp.zeros((self.recurrent_size,)),
            jnp.zeros((self.recurrent_size,)),
        )

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, carry: MinGRURecurrentState, input: MinGRURecurrentState
    ) -> MinGRURecurrentState:
        prev_state, _ = carry
        state, gate = input

        return (1 - gate) * prev_state + gate * state, gate


class MinGRU(GRAS):
    """A simple Linear Recurrent layer.

    You might want to use this as a building block for a more complex model.
    """

    recurrent_size: int
    scan: Callable[
        [
            Callable[
                [MinGRURecurrentStateWithReset, MinGRURecurrentStateWithReset],
                MinGRURecurrentStateWithReset,
            ],
            MinGRURecurrentStateWithReset,
            MinGRURecurrentStateWithReset,
        ],
        MinGRURecurrentStateWithReset,
    ]
    algebra: BinaryAlgebra

    gate: nn.Sequential
    project: nn.Linear

    def __init__(self, recurrent_size, key):
        self.recurrent_size = recurrent_size
        self.algebra = Resettable(MinGRUSemigroup(recurrent_size))
        self.scan = semigroup_scan

        keys = jax.random.split(key)

        self.gate = nn.Sequential(
            [
                nn.Linear(recurrent_size, recurrent_size, key=keys[0]),
                nn.Lambda(jax.nn.sigmoid),
            ]
        )
        self.project = nn.Linear(recurrent_size, recurrent_size, key=keys[1])

    @jaxtyped(typechecker=typechecker)
    def forward_map(
        self, x: Input, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> MinGRURecurrentStateWithReset:
        emb, start = x
        gate = self.gate(emb)
        z = self.project(emb)
        return (z, gate), start

    @jaxtyped(typechecker=typechecker)
    def backward_map(
        self,
        h: MinGRURecurrentStateWithReset,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) -> Float[Array, "{self.recurrent_size}"]:
        emb, start = x
        (state, gate), reset_carry = h
        return state

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> MinGRURecurrentStateWithReset:
        return self.algebra.initialize_carry(key)

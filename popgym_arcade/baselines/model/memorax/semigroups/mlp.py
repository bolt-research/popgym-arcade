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

NOOPRecurrentState = Float[Array, ""]
NOOPRecurrentStateWithReset = Tuple[NOOPRecurrentState, StartFlag]


class NOOPSemigroup(Semigroup):

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> NOOPRecurrentState:
        return jnp.zeros((0,))

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, carry: NOOPRecurrentState, input: NOOPRecurrentState
    ) -> NOOPRecurrentState:
        return input


class MLP(GRAS):
    """A Gated Impulse Linear Recurrent layer.

    You might want to use this as a building block for a more complex model.
    """

    recurrent_size: int
    scan: Callable[
        [
            Callable[
                [NOOPRecurrentStateWithReset, NOOPRecurrentStateWithReset],
                NOOPRecurrentStateWithReset,
            ],
            NOOPRecurrentStateWithReset,
            NOOPRecurrentStateWithReset,
        ],
        NOOPRecurrentStateWithReset,
    ]
    algebra: BinaryAlgebra

    mlp: nn.Sequential

    def __init__(self, recurrent_size, key):
        self.recurrent_size = recurrent_size
        self.algebra = Resettable(NOOPSemigroup())
        self.scan = semigroup_scan

        keys = jax.random.split(key)

        self.mlp = nn.Sequential(
            [
                nn.Linear(recurrent_size, recurrent_size, key=keys[0]),
                nn.Lambda(jax.nn.leaky_relu),
                nn.Linear(recurrent_size, recurrent_size, key=keys[1]),
                nn.Lambda(jax.nn.leaky_relu),
            ]
        )

    @jaxtyped(typechecker=typechecker)
    def forward_map(
        self, x: Input, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> NOOPRecurrentStateWithReset:
        return x

    @jaxtyped(typechecker=typechecker)
    def backward_map(
        self,
        h: NOOPRecurrentStateWithReset,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) -> Float[Array, "{self.recurrent_size}"]:
        emb, start = x
        state, reset_carry = h
        return self.mlp(emb)

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> NOOPRecurrentStateWithReset:
        return self.algebra.initialize_carry(key)

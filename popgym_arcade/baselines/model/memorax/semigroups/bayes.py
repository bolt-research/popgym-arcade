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

# Type-check a function

LogBayesRecurrentState = Float[Array, "Hidden"]
LogBayesRecurrentStateWithReset = Tuple[LogBayesRecurrentState, StartFlag]


class LogBayesSemigroup(Semigroup):
    recurrent_size: int

    def __init__(self, recurrent_size: int):
        self.recurrent_size = recurrent_size

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> LogBayesRecurrentState:
        return jnp.ones((self.recurrent_size,)) * -jnp.log(self.recurrent_size)

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, carry: LogBayesRecurrentState, input: LogBayesRecurrentState
    ) -> LogBayesRecurrentState:
        # log space P(s' | s) * P(s)
        # P(output) = P(input) * P(carry)
        # log P(output) = log P(input) + log P(carry)
        return carry + input  # In log space, equivalent to carry @ input


class LogBayes(GRAS):
    """A simple Bayesian memory layer.

    You might want to use this as a building block for a more complex model.
    """

    recurrent_size: int
    scan: Callable[
        [
            Callable[
                [LogBayesRecurrentStateWithReset, LogBayesRecurrentStateWithReset],
                LogBayesRecurrentStateWithReset,
            ],
            LogBayesRecurrentStateWithReset,
            LogBayesRecurrentStateWithReset,
        ],
        LogBayesRecurrentStateWithReset,
    ]
    algebra: BinaryAlgebra

    project: nn.Linear

    def __init__(self, recurrent_size, key):
        self.recurrent_size = recurrent_size
        self.algebra = Resettable(LogBayesSemigroup(recurrent_size))
        self.scan = semigroup_scan

        keys = jax.random.split(key)

        self.project = nn.Linear(recurrent_size, recurrent_size, key=keys[0])

    @jaxtyped(typechecker=typechecker)
    def forward_map(
        self, x: Input, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> LogBayesRecurrentStateWithReset:
        emb, start = x
        z = jax.nn.log_softmax(self.project(emb))
        return z, start

    @jaxtyped(typechecker=typechecker)
    def backward_map(
        self,
        h: LogBayesRecurrentStateWithReset,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) -> Float[Array, "{self.recurrent_size}"]:
        emb, start = x
        state, reset_carry = h
        # Convert log
        out = jnp.exp(state)
        return out

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> LogBayesRecurrentStateWithReset:
        return self.algebra.initialize_carry(key)

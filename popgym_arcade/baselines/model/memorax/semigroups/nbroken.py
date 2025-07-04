from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from equinox import filter_vmap, nn
from jaxtyping import Array, Float, PRNGKeyArray, Shaped, jaxtyped

from ..gras import GRAS
from ..groups import BinaryAlgebra, Resettable, Semigroup
from ..mtypes import Input, StartFlag
from ..scans import semigroup_scan

NBrokenRecurrentState = Float[Array, "Hidden"]
NBrokenRecurrentStateWithReset = Tuple[NBrokenRecurrentState, StartFlag]


class NBrokenMonoid(Semigroup):
    recurrent_size: int
    W: nn.Linear

    def __init__(self, recurrent_size: int, key):
        self.recurrent_size = recurrent_size
        self.W = nn.Linear(2 * recurrent_size, recurrent_size, key=key)

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> NBrokenRecurrentState:
        return jnp.zeros((self.recurrent_size))

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, carry: NBrokenRecurrentState, input: NBrokenRecurrentState
    ) -> NBrokenRecurrentState:
        return jax.nn.tanh(self.W(jnp.concatenate([carry, input])))


class NBroken(GRAS):
    """A Gated Impulse Linear Recurrent layer.

    You might want to use this as a building block for a more complex model.
    """

    recurrent_size: int
    scan: Callable[
        [
            Callable[
                [NBrokenRecurrentStateWithReset, NBrokenRecurrentStateWithReset],
                NBrokenRecurrentStateWithReset,
            ],
            NBrokenRecurrentStateWithReset,
            NBrokenRecurrentStateWithReset,
        ],
        NBrokenRecurrentStateWithReset,
    ]
    algebra: BinaryAlgebra

    g: nn.Sequential

    def __init__(self, recurrent_size, key):
        self.recurrent_size = recurrent_size
        keys = jax.random.split(key)
        self.algebra = Resettable(NBrokenMonoid(recurrent_size, key=keys[0]))
        self.scan = semigroup_scan

        self.g = nn.Sequential(
            [
                nn.Linear(recurrent_size, recurrent_size, key=keys[1]),
                nn.Lambda(jax.nn.sigmoid),
            ]
        )

    @jaxtyped(typechecker=typechecker)
    def forward_map(
        self, x: Input, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> NBrokenRecurrentStateWithReset:
        emb, start = x
        z = emb
        return z, start

    @jaxtyped(typechecker=typechecker)
    def backward_map(
        self,
        h: NBrokenRecurrentStateWithReset,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) -> Float[Array, "{self.recurrent_size}"]:
        emb, start = x
        state, reset_carry = h
        z = state / jnp.linalg.norm(state, ord=1)
        return z
        # g = self.g(emb)
        # return g * z + (1 - g) * emb

    def compute_associative_error(
        self,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) -> Float[Array, ""]:
        """Calls the mapping and scan functions.

        You probably do not need to override this."""

        def assoc(monoid, x1, x2, x3):
            a = monoid(monoid(x1, x2), x3)
            b = monoid(x1, monoid(x2, x3))
            return a, b

        x = filter_vmap(self.forward_map)((x, jnp.zeros_like(x, dtype=bool)))
        # change numels to implicitly do
        # f(a(b, c)) and f(f(a, b), c)
        x1 = x[:-2]
        x2 = x[1:-1]
        x3 = x[2:]
        h, h_alt = filter_vmap(assoc, in_axes=(None, 0, 0, 0))(x1, x2, x3)
        return jnp.mean(jnp.square(h - h_alt))
        # h11 = eqx.filter_vmap(self.algebra)(x1, x2)
        # h12 = eqx.filter_vmap(self.algebra)(h11, x3)

        # h21 = eqx.filter_vmap(self.algebra)(x2, x3)
        # h22 = eqx.filter_vmap(self.algebra)(x1, h21)
        # return jnp.mean(jnp.square(h22 - h12))

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> NBrokenRecurrentStateWithReset:
        return self.algebra.initialize_carry(key)

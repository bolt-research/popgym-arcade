from typing import Callable, List, Optional, Tuple

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from equinox import nn
from jaxtyping import Array, Float, PRNGKeyArray, Shaped, jaxtyped

from popgym_arcade.baselines.model.memorax.gras import GRAS
from popgym_arcade.baselines.model.memorax.groups import (
    BinaryAlgebra,
    Module,
    Resettable,
    SetAction,
)
from popgym_arcade.baselines.model.memorax.mtypes import Input, StartFlag
from popgym_arcade.baselines.model.memorax.scans import set_action_scan

SphericalRecurrentState = Float[Array, "Recurrent"]
SphericalRecurrentStateWithReset = Tuple[SphericalRecurrentState, StartFlag]


class SphericalMagma(SetAction):
    """
    The Spherical Magma (recurrent update) from
    https://onlinelibrary.wiley.com/doi/abs/10.1207/s15516709cog1402_1.
    """

    recurrent_size: int
    project: nn.Linear
    initial_state: jax.Array

    def __init__(self, recurrent_size: int, sequence_length: int = 1024, *, key):
        self.recurrent_size = recurrent_size
        proj_size = int(self.recurrent_size * (self.recurrent_size - 1) / 2)
        self.project = nn.Linear(recurrent_size, proj_size, key=key)
        self.initial_state = jnp.ones((self.recurrent_size,))
        # self.step_size = 1 / jnp.array([sequence_length]) * 2 * jnp.pi

    @jaxtyped(typechecker=typechecker)
    def rot(self, z: Array) -> Array:
        q = self.project(z)
        A = jnp.zeros((self.recurrent_size, self.recurrent_size))
        tri_idx = jnp.triu_indices_from(A, 1)
        A = A.at[tri_idx].set(q)
        A = A - A.T
        R = jax.scipy.linalg.expm(A)
        # self.step_size *  jnp.linalg.norm(R, ord='frobenius', axis=(-2, -1))
        return R

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, carry: SphericalRecurrentState, input: SphericalRecurrentState
    ) -> SphericalRecurrentState:
        R = self.rot(input)
        return R @ carry
        # return jax.nn.tanh(self.U_h(carry) + input)

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> SphericalRecurrentState:
        return self.initial_state / jnp.linalg.norm(self.initial_state)


class Spherical(GRAS):
    """The Spherical RNN from
    https://onlinelibrary.wiley.com/doi/abs/10.1207/s15516709cog1402_1."""

    algebra: BinaryAlgebra
    scan: Callable[
        [
            Callable[
                [SphericalRecurrentStateWithReset, SphericalRecurrentStateWithReset],
                SphericalRecurrentStateWithReset,
            ],
            SphericalRecurrentStateWithReset,
            SphericalRecurrentStateWithReset,
        ],
        SphericalRecurrentStateWithReset,
    ]
    W_y: nn.Linear
    recurrent_size: int
    hidden_size: int

    def __init__(self, recurrent_size, hidden_size, key):
        self.recurrent_size = recurrent_size
        self.hidden_size = hidden_size
        keys = jax.random.split(key)
        self.algebra = Resettable(SphericalMagma(recurrent_size, key=keys[0]))
        self.scan = set_action_scan
        self.W_y = nn.Linear(recurrent_size, hidden_size, key=keys[1])

    @jaxtyped(typechecker=typechecker)
    def forward_map(
        self, x: Input, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> SphericalRecurrentStateWithReset:
        emb, start = x
        return emb, start

    @jaxtyped(typechecker=typechecker)
    def backward_map(
        self,
        h: SphericalRecurrentStateWithReset,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) -> Float[Array, "{self.hidden_size}"]:
        z, reset_flag = h
        emb, start = x
        return self.W_y(z)

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> SphericalRecurrentStateWithReset:
        return self.algebra.initialize_carry(key)

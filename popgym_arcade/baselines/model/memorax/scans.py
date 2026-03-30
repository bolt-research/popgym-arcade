from typing import Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, Shaped

from popgym_arcade.baselines.model.memorax.mtypes import RecurrentState
from popgym_arcade.baselines.model.memorax.utils import debug_shape


def set_action_scan(
    magma_op: Callable[[RecurrentState, RecurrentState], RecurrentState],
    state: RecurrentState,
    input: RecurrentState,
):
    """Update the recurrent state using an ordered scan.

    Executes a magma scan, which works with virtually any recurrent model.
    See https://en.wikipedia.org/wiki/Magma_(algebra) for information about magmas.
    """
    assert jax.tree.structure(state) == jax.tree.structure(
        input
    ), f"Mismatched structures passed to scan, {jax.tree.structure(state)} and {jax.tree.structure(input)}"

    def wrapped_magma_op(carry, xs):
        out = magma_op(carry, xs)
        return out, out  # Ensure the output carry matches the input structure

    _, new_state = jax.lax.scan(
        f=wrapped_magma_op,
        init=state,
        xs=input,
    )
    assert jax.tree.structure(new_state) == jax.tree.structure(
        input
    ), f"Mismatched structures returned from scan, {jax.tree.structure(input)} and {jax.tree.structure(new_state)}"
    return new_state


def semigroup_scan(
    monoid_op: Callable[[RecurrentState, RecurrentState], RecurrentState],
    state: RecurrentState,
    input: RecurrentState,
    axis: int = 0,
) -> RecurrentState:
    """Update the recurrent state using an associative scan.

    Executes a monoidal scan. The monoid_op MUST be associative, i.e.,
        .. math::

            f(a, f(b,c)) = f(f(a,b), c)

    See https://en.wikipedia.org/wiki/Monoid for information about monoids.

    """
    assert jax.tree.structure(state) == jax.tree.structure(
        input
    ), f"Mismatched structures passed to scan, {jax.tree.structure(state)} and {jax.tree.structure(input)}"

    # Both scan initial state and inputs must have the same numbers of dims
    # Usually, we have h0 = (features,) and x = (time, features)
    # Expand h0 = (1, features)
    singleton_state = jax.tree.map(lambda s: jnp.expand_dims(s, axis=axis), state)
    # Concatenate the previous state to the inputs and scan over the result
    # This ensures the previous recurrent state contributes to the current batch
    scan_inputs = jax.tree.map(
        lambda s, x: jnp.concatenate([s, x], axis=axis), singleton_state, input
    )

    new_state = jax.lax.associative_scan(
        fn=eqx.filter_vmap(monoid_op), elems=scan_inputs, axis=axis
    )

    # The zeroth index corresponds to the previous recurrent state
    # We just use it to ensure continuity
    # We do not actually want to use these values, so slice them away
    new_state = jax.tree.map(
        lambda x: jax.lax.slice_in_dim(x, start_index=1, limit_index=None, axis=axis),
        new_state,
    )

    assert jax.tree.structure(new_state) == jax.tree.structure(
        input
    ), f"Mismatched structures returned from scan, {debug_shape(input)} and {debug_shape(new_state)}"
    assert all(
        jax.tree.leaves(jax.tree.map(lambda x, y: x.shape == y.shape, input, new_state))
    ), f"Shapes do not match {debug_shape(input)} and {debug_shape(new_state)}"

    return new_state

from functools import partial

import jax
import jax.numpy as jnp

from popgym_arcade.baselines.model.memorax.groups import Semigroup
from popgym_arcade.baselines.model.memorax.train_utils import get_monoids


def random_state(state, key):
    if state.dtype in [jnp.float32, jnp.complex64]:
        return jax.random.normal(key, state.shape, dtype=state.dtype)
    elif state.dtype in [jnp.int32]:
        return jax.random.randint(key, state.shape, 0, 5, dtype=state.dtype)
    elif state.dtype in [jnp.bool_]:
        return jax.random.bernoulli(key, 0.5, state.shape, dtype=state.dtype)
    else:
        raise NotImplementedError(
            f"Random state not implemented for dtype {state.dtype}"
        )


def map_assert(monoid, a, b):
    is_equal = jnp.allclose(a, b)
    error = jnp.abs(a - b)
    if not is_equal:
        print(
            f"Monoid {type(monoid).__name__} failed associativity test:\n{a} != \n{b}, \nerror: {error}"
        )


def prove_monoid_correctness(monoid: Semigroup):
    initial_state = monoid.initialize_carry()
    x1 = jax.tree.map(partial(random_state, key=jax.random.PRNGKey(1)), initial_state)
    x2 = jax.tree.map(partial(random_state, key=jax.random.PRNGKey(2)), initial_state)
    x3 = jax.tree.map(partial(random_state, key=jax.random.PRNGKey(3)), initial_state)

    breakpoint()

    a = monoid(monoid(x1, x2), x3)
    b = monoid(x1, monoid(x2, x3))

    is_equal = jax.tree.map(jnp.allclose, a, b)
    if isinstance(is_equal, tuple):
        is_equal = all(is_equal)
    else:
        is_equal = jnp.all(is_equal)

    jax.tree.map(partial(map_assert, monoid), a, b)
    # error = jax.tree.map(lambda a, b: jnp.abs(a - b), a, b)
    # assert jnp.all(is_equal), f"Monoid {monoid} failed associativity test:\n{a} != \n{b}, \nerror: {error}"


for name, monoid in get_monoids(recurrent_size=3, key=jax.random.PRNGKey(0)).items():
    prove_monoid_correctness(monoid)

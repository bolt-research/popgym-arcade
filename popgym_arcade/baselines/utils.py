import equinox as eqx
from jax import lax


@eqx.filter_jit
def filter_scan(f, init, xs, *args, **kwargs):
    """Same as lax.scan, but allows to have eqx.Module in carry"""
    init_dynamic_carry, static_carry = eqx.partition(init, eqx.is_array)

    def to_scan(dynamic_carry, x):
        carry = eqx.combine(dynamic_carry, static_carry)
        new_carry, out = f(carry, x)
        dynamic_new_carry, _ = eqx.partition(new_carry, eqx.is_array)
        return dynamic_new_carry, out

    out_carry, out_ys = lax.scan(to_scan, init_dynamic_carry, xs, *args, **kwargs)
    return eqx.combine(out_carry, static_carry), out_ys

import jax
import jax.numpy as jnp
import functools
from jax.experimental import jet


def ddf(psi_fn):
    @jax.jit
    def dd_psi_fn(z, v):
        """
        Computes the directional derivative of psi in the direction v
        """
        _, ts = jax.jvp(psi_fn, (z,), (v,))
        return ts

    return dd_psi_fn


def dd2f(psi_fn):
    return lambda x, v: ddf(lambda xx: ddf(psi_fn)(xx, v))


def nth_dir_deriv(f, only_n=False, ravel_out=True):
    """
    TODO: Thought this was just a curiosity, but it seems to be faster than jet.jet
    and therefore probably needs a docstring
    """

    @functools.partial(jax.jit, static_argnums=(2,))
    def dnfdvn_internal(x, v, n):
        if n == 0:
            return f(x)
        elif n == 1:
            ps, ts = jax.jvp(f, (x,), (v,))
            return ts, ps
        elif n >= 2:
            ps, ts, *lower_ps = jax.jvp(
                lambda xx: dnfdvn_internal(xx, v, n - 1), (x,), (v,), has_aux=True
            )
            return ts, (ps, lower_ps)

    @functools.partial(jax.jit, static_argnums=(2,))
    def dnfdvn(x, v, n):
        o, *s = dnfdvn_internal(x, v, n)
        if only_n:
            return o
        if ravel_out:
            return jax.tree_util.tree_leaves((o, s))[::-1]
        else:
            return o, s

    return dnfdvn


def nth_dir_deriv_jet(f, only_n=False, ravel_out=True):
    @functools.partial(jax.jit, static_argnums=(2,))
    def dnfdvn_jet(x, v, n):
        tns = [v] + [jnp.zeros_like(v)] * (n - 1)
        f0, tns = jet.jet(f, (x,), (tns,))
        if only_n:
            return tns[-1]
        if ravel_out:
            return jax.tree_util.tree_leaves((f0, tns))[::-1]
        return f0, tns

    return dnfdvn_jet


# dd_psi_fn = ddf(psi_fn)
# dd2_psi_fn = dd2f(psi_fn)
# v0_zeros = jnp.zeros_like(v0)
# f0, (f1, f2) = jet.jet(psi_fn, (z,), ((v0, v0_zeros),))
# print(f1 - dd_psi_fn(z, v0))
# print(f2 - dd2_psi_fn(z, v0))

# @functools.partial(jax.jit, static_argnums=(2,))
# def ddn_psi_fn(z, v, n):
#     """
#     Computes the nth directional derivative of psi in the direction v
#     """
#     if n == 0:
#         return psi_fn(z)
#     else:
#         _, ts = jax.jvp(lambda zz: ddn_psi_fn(zz, v, n - 1), (z,), (v,))
#         return ts

# f0, f1, f2, f3 = nth_dir_deriv(psi_fn)(z, v0, 3)
# ff3 = nth_dir_deriv(psi_fn, only_n=True)(z, v0, 3)
# assert np.abs(ff3 - f3) < 1e-5

# def test_derivatives(z, v, n):
#     assert n >= 1
#     pre_tangents = [v] + [jnp.zeros_like(v)] * (n - 1)
#     f0, tangents = jet.jet(psi_fn, (z,), (pre_tangents,))
#     tangents = [f0] + tangents
#     print("n \t| Jet \t\t| DD \t\t| Diff")
#     for i in range(n):
#         ddv = ddn_psi_fn(z, v, i)
#         print(
#             "n = ",
#             i,
#             "\t| ",
#             float(tangents[i]),
#             "\t| ",
#             float(ddv),
#             "\t| ",
#             float(tangents[i] - ddv),
#         )


# @functools.partial(jax.jit, static_argnums=(2,))
# def ddn_psi_fn_jet(z, v, n):
#


# for i in range(1, 6):
#     lowered = ddn_psi_fn.lower(z, v0, i)
#     compiled = lowered.compile()
#     flops = compiled.cost_analysis()[0]["flops"]
#     print(f"n = {i} | {flops} flops")

# for i in range(1, 6):
#     lowered = ddn_psi_fn_jet.lower(z, v0, i)
#     compiled = lowered.compile()
#     flops = compiled.cost_analysis()[0]["flops"]
#     print(f"n = {i} | {flops} flops")

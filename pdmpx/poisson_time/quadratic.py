import jax
import jax.numpy as jnp

# from pdmpx.poisson_time._cpu.quadratic import (
#     solve_quadratic_integral_equation as cpu_solve_quadratic_integral_equation,
# )

from .linear import ab_poisson_time


def _sign(x):
    return jnp.where(x >= 0, 1.0, -1.0)


def solve_cubic_eq(p0, p1, p2, p3):  # , only_real_roots=True):
    """
    Returns the real roots of the real cubic polynomial
        p0 + p1 x + p2 x^2 + p3 x^3
    """
    a, b, c, d = p0, p1, p2, p3
    # x3 + a2x2 + a1x + a0
    a0 = a / d
    a1 = b / d
    a2 = c / d

    q = a1 / 3 - a2**2 / 9
    r = (a1 * a2 - 3 * a0) / 6 - a2**3 / 27

    def numerical_recipes_one_real_root():
        A = (jnp.abs(r) + jnp.sqrt(r**2 + q**3)) ** (1 / 3)
        t1 = (A - q / A) * _sign(r)
        x1 = t1 - a2 / 3
        return jnp.array([x1, jnp.nan, jnp.nan])

    def viete_three_real_roots():
        theta = jax.lax.cond(
            q == 0.0,
            lambda: 0.0,
            lambda: jnp.real(jnp.arccos(r / (-q) ** (3 / 2))),
        )

        phi1 = theta / 3
        phi2 = theta / 3 - 2 * jnp.pi / 3
        phi3 = theta / 3 + 2 * jnp.pi / 3

        x1 = 2 * jnp.sqrt(-q) * jnp.cos(phi1) - a2 / 3
        x2 = 2 * jnp.sqrt(-q) * jnp.cos(phi2) - a2 / 3
        x3 = 2 * jnp.sqrt(-q) * jnp.cos(phi3) - a2 / 3

        return jnp.array([x3, x2, x1])

    return jax.lax.cond(
        q**3 + r**2 > 0,
        numerical_recipes_one_real_root,
        viete_three_real_roots,
    )


def min_root(p0, p1, p2, p3, lower_bound=-jnp.inf):
    roots = solve_cubic_eq(p0, p1, p2, p3)
    roots = jnp.where(roots >= lower_bound, roots, jnp.nan)
    return jnp.nanmin(roots)


# @jax.jit
def solve_quadratic_integral_equation(y, a, b, c):
    """
    Solves t = argmin { y = \\int_0^t (a + bx + cx^2)+ dx }
    """

    def almost_zero(v):
        return jnp.abs(v) < 1e-8

    def P(x):
        return a * x + (b / 2) * x**2 + (c / 3) * x**3

    def solve_P(y, lower_bound=0.0) -> float:
        """
        Solve argmin x P(x) = y
        """
        return min_root(-y, a, b / 2, c / 3, lower_bound=lower_bound)

    def c_almost_zero():
        return ab_poisson_time(jnp.exp(-y), a, b)

    def c_positive():
        # find zeros of p(x):
        # a/c + b/c x + x^2 = 0
        # (x + b/2c)^2 = -a/c + (b/2c)^2
        q = -a / c + (b / (2 * c)) ** 2

        # if q <= 0: always non-negative
        def q_positive():
            r0 = jnp.maximum(0.0, -b / (2 * c) - jnp.sqrt(q))
            r1 = jnp.maximum(0.0, -b / (2 * c) + jnp.sqrt(q))
            return jax.lax.cond(
                P(r0) > y,
                lambda: solve_P(y),
                lambda: solve_P(y + P(r1) - P(r0), lower_bound=r1),
            )

        return jax.lax.cond(q <= 0, lambda: solve_P(y), q_positive)

    def c_negative():
        # find zeros of p(x):
        q = -a / c + (b / (2 * c)) ** 2

        def q_negative():
            return jnp.inf  # no solution

        def q_positive():
            r0 = -b / (2 * c) - jnp.sqrt(q)
            r1 = -b / (2 * c) + jnp.sqrt(q)
            r0 = jnp.maximum(0.0, -b / (2 * c) - jnp.sqrt(q))
            r1 = jnp.maximum(0.0, -b / (2 * c) + jnp.sqrt(q))
            s0 = P(r1) - P(r0)  # total mass in [(r0)+, (r1)+]
            return jax.lax.cond(
                y <= s0,
                lambda: solve_P(y + P(r0), lower_bound=r0),
                lambda: jnp.inf,
            )

        return jax.lax.cond(q <= 0, q_negative, q_positive)

    return jax.lax.cond(
        almost_zero(c),
        c_almost_zero,
        lambda: jax.lax.cond(c > 0, c_positive, c_negative),
    )


def abc_poisson_time(u, a, b, c):
    """
    Returns the first arrival time of Poisson process with time-dependent rate
        rate = (a + b*t + c*t^2)+
    Args:
        u: Uniform random variable in [0, 1)
        a: Constant term
        b: Linear term
        c: Quadratic term
    """
    return solve_quadratic_integral_equation(-jnp.log(1.0 - u), a, b, c)

import jax
import jax.numpy as jnp
import numpy as np
from pdmpx.poisson_time.quadratic import (
    solve_quadratic_integral_equation,
)

from pdmpx.poisson_time._cpu.quadratic import (
    solve_quadratic_integral_equation as cpu_solve_quadratic_integral_equation,
)


def test_solve_quadratic_integral_equation2():
    np.random.seed(0)
    xs = np.linspace(0.0, 5.0, 10000)
    fails = 0
    jit_solve_quadratic_integral_equation = jax.jit(solve_quadratic_integral_equation)
    for i in range(100):
        a, b = np.random.normal(loc=2.0, scale=2.0, size=2)
        c = 1.0 if np.random.uniform() < 0.5 else -1.0
        y = np.random.uniform() * 2.0
        alg_sol = jit_solve_quadratic_integral_equation(y, a, b, c)
        cpu_sol = cpu_solve_quadratic_integral_equation(y, a, b, c)

        if np.abs(cpu_sol - alg_sol) > 1e-3:
            fails += 1

    assert fails == 0, f"solve_quadratic_integral_equation failed {fails} out of 100"

import numpy as np
import jax
import jax.random as jr
from pdmpx.poisson_time.linear import ab_poisson_time as jax_ab_poisson_time
from pdmpx.poisson_time._cpu.linear import ab_poisson_time as cpu_ab_poisson_time


def allclose(a, b):
    return np.allclose(np.array(a), np.array(b), atol=1e-6, equal_nan=True)


def test_ab_poisson_time():
    us = jr.uniform(jr.key(0), (5,))
    cus = np.array(us)

    ab_poisson_time = jax.jit(jax_ab_poisson_time)

    a, b = 1.0, -2.0
    assert allclose(cpu_ab_poisson_time(cus[0], a, b), ab_poisson_time(us[0], a, b))

    a, b = 0.0, -2.0
    assert allclose(cpu_ab_poisson_time(cus[1], a, b), ab_poisson_time(us[1], a, b))

    a, b = 2.0, 0.0
    assert allclose(-np.log(1.0 - cus[2]) / a, ab_poisson_time(us[2], a, b))

    a, b = 1.0, -0.0
    assert allclose(-np.log(1.0 - cus[3]) / a, ab_poisson_time(us[3], a, b))

    a, b = 10.0, 1.0
    assert allclose(cpu_ab_poisson_time(cus[4], a, b), ab_poisson_time(us[4], a, b))

    a, b = 0.09240358, -0.48632997
    u = 0.2
    assert ab_poisson_time(u, a, b) == np.inf

    # sanity check
    a, b = 1.0, -1.0
    u = 0.2
    assert ab_poisson_time(u, a, b) <= 1.0

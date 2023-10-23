import numpy as np
import jax.random as jr
import pdmpx.poisson_time.linear as linear
from ._cpu.linear import ab_poisson_time as cpu_ab_poisson_time


def allclose(a, b):
    return np.allclose(np.array(a), np.array(b), atol=1e-6, equal_nan=True)


def test_ab_poisson_time():
    us = jr.uniform(jr.key(0), (5,))
    cus = np.array(us)

    a, b = 1.0, -2.0
    assert allclose(
        cpu_ab_poisson_time(cus[0], a, b), linear.ab_poisson_time(us[0], a, b)
    )

    a, b = 0.0, -2.0
    assert allclose(
        cpu_ab_poisson_time(cus[1], a, b), linear.ab_poisson_time(us[1], a, b)
    )

    a, b = 2.0, 0.0
    assert allclose(-np.log(cus[2]) / a, linear.ab_poisson_time(us[2], a, b))

    a, b = 1.0, -0.0
    assert allclose(-np.log(cus[3]) / a, linear.ab_poisson_time(us[3], a, b))

    a, b = 10.0, 1.0
    assert allclose(
        cpu_ab_poisson_time(cus[4], a, b), linear.ab_poisson_time(us[4], a, b)
    )

import jax
import jax.numpy as jnp

# import numpy as np
# import numba
import functools as ft


def exp_rand(key, rate=1.0):
    """
    Exponential random variable
    """
    return -jnp.log(jax.random.uniform(key)) / rate


def poisson_time(key, rate=1.0):
    """
    First arrival time of Poisson process with constant rate `rate`
    """
    return exp_rand(key, rate)


def poisson_time_stepf(key, rates, ts):
    """
    First arrival time of Poisson process with time-varying piecewise constant rate `rates`.
    """
    raise NotImplementedError
    # reference time
    x = exp_rand(key)

    dts = jnp.diff(ts)

    return ts + exp_rand(key, rates)

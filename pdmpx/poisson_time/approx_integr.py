import jax
import jax.numpy as jnp
import numpy as np
import numba
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
    # reference time
    x = exp_rand(key)

    dts = jnp.diff(ts)

    return ts + exp_rand(key, rates)


@numba.jit(nopython=True)
def trapezoid(x, y):
    return 0.5 * np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]))


@numba.jit(nopython=True)
def trapezoid_cum(x, y):
    return 0.5 * np.cumsum((x[1:] - x[:-1]) * (y[1:] + y[:-1]))


class TrapezoidTime:
    def __call__(self, ts, rates):
        ref_time = np.random.exponential()
        raise NotImplementedError

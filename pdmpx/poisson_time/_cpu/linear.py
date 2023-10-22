import numpy as np
import numba


@numba.jit(nopython=True)
def ab_poisson_time(u: float, a: float, b: float) -> float:  # TODO: linear_poisson_time
    """
    First arrival time of Poisson process with time-dependent rate `rate = (a + b*t)+`
    """
    if b < 0:
        if a <= 0:
            return np.inf
        else:
            y = -np.log(u)  # reference time
            if -a / b < y:
                return (-a - np.sqrt(2 * b * y + a**2)) / b  # solve
            else:
                return np.inf
    elif b == 0:
        if a <= 0:
            return np.inf
        else:
            return -np.log(u) / a
    else:  # b > 0
        if a < 0:
            y = -np.log(u)
            return (-a + np.sqrt(2 * b * y)) / b
        else:
            y = -np.log(u)
            return (-a + np.sqrt(2 * b * y + a**2)) / b


linear_poisson_time = ab_poisson_time


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

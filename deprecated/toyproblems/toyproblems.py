# https://en.wikipedia.org/wiki/Test_functions_for_optimization

import numpy as np
import jax.numpy as jnp
import jax

from typing import NamedTuple, Callable


class Problem(NamedTuple):
    name: str
    dim: int
    primal: Callable[[jnp.ndarray], float]
    search_domain: jnp.ndarray
    global_min: jnp.ndarray
    global_min_value: float
    local_mins: jnp.ndarray
    local_mins_values: jnp.ndarray


class Rastrigin:
    def __init__(self, dim=2):
        self.name = "Rastrigin"
        self.dim = dim
        self.search_domain = jnp.array([[-5.12, 5.12]] * dim)
        self.global_min = jnp.zeros(dim)
        self.global_min_value = 0.0
        self.local_mins = jnp.array([jnp.array([0.0, 0.0])])  # wrong
        self.local_mins_values = jnp.array([0.0])

    def __call__(self, x):
        return 10 * self.dim + jnp.sum(x**2 - 10 * jnp.cos(2 * np.pi * x), axis=-1)


class Ackley:
    def __init__(self, dim=2):
        self.name = "Ackley"
        self.dim = dim
        self.search_domain = jnp.array([[-5.0, 5.0]] * dim)
        self.global_min = jnp.zeros(dim)
        self.global_min_value = 0.0
        self.local_mins = jnp.array([jnp.array([0.0, 0.0])])  # wrong
        self.local_mins_values = jnp.array([0.0])

    def __call__(self, x):
        return (
            -20 * jnp.exp(-0.2 * jnp.sqrt(jnp.sum(x**2, axis=-1) / self.dim))
            - jnp.exp(jnp.sum(jnp.cos(2 * np.pi * x), axis=-1) / self.dim)
            + 20
            + jnp.exp(1)
        )


class Rosenbrock:
    def __init__(self, dim=2):
        self.name = "Rosenbrock"
        self.dim = dim
        self.search_domain = jnp.array([[-5.0, 5.0]] * dim)
        self.global_min = jnp.ones(dim)
        self.global_min_value = 0.0
        self.local_mins = jnp.array([jnp.array([0.0, 0.0])])
        self.local_mins_values = jnp.array([0.0])

    def __call__(self, x):
        return jnp.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


class Beale:
    def __init__(self):
        self.name = "Beale"
        self.dim = 2
        self.search_domain = jnp.array([[-4.5, 4.5]] * self.dim)
        self.global_min = jnp.array([3.0, 0.5])
        self.global_min_value = 0.0
        self.local_mins = jnp.array([jnp.array([0.0, 0.0])])
        self.local_mins_values = jnp.array([0.0])


class Eggholder:
    def __init__(self):
        self.dim = 2
        self.search_domain = jnp.array([[-1000, 1000] * self.dim])
        self.global_min = jnp.array([512.0, 404.2319])
        self.global_min_value = -959.6407

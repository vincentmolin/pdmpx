import jax.random as jr
import pytest


class KeyStream:
    def __init__(self, seed=0):
        self.key = jr.key(seed)
        self.n = 1

    def __call__(self):
        self.n += 1
        return jr.fold_in(self.key, self.n)


@pytest.fixture
def nextkey():
    return KeyStream()

from ..pdmp import AbstractKernel


class IdentityKernel(AbstractKernel):
    def __call__(self, rng, state, timer_event):
        return state

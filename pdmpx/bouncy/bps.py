from ..pdmp import (
    PDMP,
    PDMPState,
    AbstractDynamics,
    AbstractKernel,
    AbstractTimer,
    TimerEvent,
    PyTree,
)

from .dynamics import LinearDynamics
from .bounces import BounceKernel, BounceTimer
from .refreshments import RefreshmentKernel

from ..timers import ConstantRateTimer
from ..queues import Factor, SimpleFactorQueue

from typing import NamedTuple, Sequence, Tuple, Callable, Dict, Optional, Union, Any


class BPSState(NamedTuple):
    params: PyTree
    velocities: PyTree


class BouncyParticleSampler(PDMP):
    def __init__(self, potential, refreshment_rate, valid_time, normalize_velocities = True):
        """
        Args:
            potential: Differentiable potential function of the target distribution (up to an additive constant).
            refreshment_rate: Rate of refreshments.
            valid_time: Time to trust the linear approximation of the rate.
            normalize_velocities: Run in unit speed.
        """
        bounce_factor = Factor(
            BounceTimer(potential, valid_time),
            BounceKernel(potential)
        )
        refreshment_factor = Factor(
            ConstantRateTimer(refreshment_rate),
            RefreshmentKernel(normalize_velocities)
        )
        queue = SimpleFactorQueue([bounce_factor, refreshment_factor])
        dynamics = LinearDynamics()
        super().__init__(dynamics=dynamics, timer=queue.timer, kernel=queue.kernel)
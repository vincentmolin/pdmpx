import jax
import jax.numpy as jnp

from pdmpx import PDMPState
from pdmpx.queues import SimpleFactorQueue, Factor
from pdmpx.timers import ConstantRateTimer
from pdmpx.utils.kernels import IdentityKernel

import pytest


def test_simple_factor_queue():
    crr_slow = Factor(
        ConstantRateTimer(0.001),
        IdentityKernel(),
    )
    crr_fast = Factor(
        ConstantRateTimer(100.0),
        IdentityKernel(),
    )

    sfq = SimpleFactorQueue([crr_slow, crr_fast])

    state = {"params": jnp.zeros((2,)), "velocities": jnp.zeros((2,))}

    event = sfq.timer(jax.random.key(0), state)
    assert event.time > 0.0
    assert event.dirty == 1.0
    assert event.params["simple_factor_queue"]["next_event_idx"] == 1
    new_state = sfq.kernel(jax.random.key(0), state, event)

    jit_event = jax.jit(sfq.timer)(jax.random.key(0), state)
    assert jit_event.time == pytest.approx(event.time, 0.01)

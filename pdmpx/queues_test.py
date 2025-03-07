import jax
import jax.numpy as jnp

from pdmpx import PDMPState, TimerEvent
from pdmpx.queues import SimpleFactorQueue, Factor
from pdmpx.timers import ConstantRateTimer
from pdmpx.utils.kernels import IdentityKernel

import pytest


def mk_dummy_timer(time, dirty=True, params={}):
    return lambda rng, state: TimerEvent(time, dirty, params)


def mk_dummy_kernel(dummy_state):
    return lambda rng, state, timer_event: dummy_state


def test_simple_factor_queue(nextkey):

    crr_slow = Factor(
        ConstantRateTimer(0.001),
        mk_dummy_kernel({"id": 0}),
    )
    crr_fast = Factor(
        ConstantRateTimer(100.0),
        mk_dummy_kernel({"id": 1}),
    )

    sfq = SimpleFactorQueue([crr_slow, crr_fast])

    state = {"params": jnp.zeros((2,)), "velocities": jnp.zeros((2,))}

    key0 = nextkey()
    event = sfq.timer(key0, state)
    assert event.time > 0.0
    assert event.dirty == 1.0
    assert event.params["simple_factor_queue"]["next_event_idx"] == 1
    new_state = sfq.kernel(nextkey(), state, event)

    assert new_state["id"] == 1
    jit_event = jax.jit(sfq.timer)(key0, state)
    assert jit_event.time == pytest.approx(event.time, 0.01)

    dfs = [
        Factor(mk_dummy_timer(t_trig), mk_dummy_kernel({"t": t_trig}))
        for t_trig in [0.1, 1.0, 2.0, jnp.inf]
    ]
    sfq = SimpleFactorQueue(dfs)
    event = sfq.timer(nextkey(), state)
    assert event.time == 0.1
    new_state = sfq.kernel(nextkey(), state, event)
    assert new_state["t"] == 0.1

    sfq3 = SimpleFactorQueue([dfs[1], dfs[2], dfs[0], dfs[3]])
    event = sfq3.timer(nextkey(), state)
    assert event.time == 0.1
    new_state = sfq3.kernel(nextkey(), state, event)
    assert new_state["t"] == 0.1

import jax
import jax.numpy as jnp

from pdmpx import PDMPState
from pdmpx.timers import (
    ConstantRateTimer,
    LinearApproxTimer,
    LinearThinningTimer,
    LinearThinningSlack,
    TimerEvent,
)

from pdmpx.dynamics import LinearDynamics


def test_constant_rate_timer():
    crt = ConstantRateTimer(1.0)
    assert crt.rate == 1.0
    state = PDMPState(jnp.zeros((2,)), jnp.zeros((2,)))

    dummy_context = {"a": True, "b": {"c": 1.0}}

    event, ctx = crt(jax.random.key(0), state, {})
    assert isinstance(event, TimerEvent)
    assert event.time > 0.0
    assert event.bound == 0.0
    assert ctx == {}
    assert event.time == jax.random.exponential(jax.random.key(0))

    event, ctx = crt(jax.random.key(1), state, dummy_context)
    assert isinstance(event, TimerEvent)
    assert dummy_context == ctx  # {"a": True, "b": {"c": 1.0}}

    jit_crt = jax.jit(crt)
    jit_event, jit_ctx = jit_crt(jax.random.key(1), state, dummy_context)
    assert isinstance(jit_event, TimerEvent)
    assert dummy_context == jit_ctx  # {"a": True, "b": {"c": 1.0}}
    assert event.time == jit_event.time
    assert event.bound == jit_event.bound


def test_linear_approx_timer():
    dynamics = LinearDynamics()

    def potential(params, context={}):
        return jnp.sum(params**2)

    def rate_fn(state: PDMPState, context={}):
        _, dpot = jax.jvp(
            lambda t: potential(dynamics.forward(t, state).params, context),
            (0.0,),
            (1.0,),
        )
        return dpot

    lat = LinearApproxTimer(rate_fn, 1.0, has_aux=False, dynamics=dynamics)
    state = PDMPState(jnp.ones((2,)), jnp.ones((2,)))
    lat_event, lat_ctx = lat(jax.random.key(0), state, {"a": 2.0})

    assert isinstance(lat_event, TimerEvent)

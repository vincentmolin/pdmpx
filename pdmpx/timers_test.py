import jax
import jax.numpy as jnp

from pdmpx import PDMPState
from pdmpx.timers import (
    ConstantRateTimer,
    LinearApproxTimer,
    # LinearThinningTimer,
    # LinearThinningSlack,
    TimerEvent,
    # QuadraticApproxTimer,
)


def test_constant_rate_timer():
    crt = ConstantRateTimer(1.0)
    assert crt.rate == 1.0
    state = {"a": jnp.zeros((2,)), "b": jnp.zeros((2,))}

    event = crt(jax.random.key(0), state)
    assert isinstance(event, TimerEvent)
    assert event.time > 0.0
    assert event.dirty == 1.0
    assert event.time == jax.random.exponential(jax.random.key(0))

    event = crt(jax.random.key(1), state)
    assert isinstance(event, TimerEvent)

    jit_crt = jax.jit(crt)
    jit_event = jit_crt(jax.random.key(1), state)
    assert isinstance(jit_event, TimerEvent)
    assert event.time == jit_event.time
    assert event.dirty == jit_event.dirty


# def test_linear_approx_timer():
#     dynamics = LinearDynamics()

#     def potential(params, context={}):
#         return jnp.sum(params**2)

#     def rate_fn(state: PDMPState, context={}):
#         _, dpot = jax.jvp(
#             lambda t: potential(dynamics.forward(t, state).params, context),
#             (0.0,),
#             (1.0,),
#         )
#         return dpot

#     lat = LinearApproxTimer(rate_fn, 1.0, has_aux=False, dynamics=dynamics)
#     state = PDMPState(jnp.ones((2,)), jnp.ones((2,)))
#     lat_event, lat_ctx = lat(jax.random.key(0), state, {"a": 2.0})

#     assert isinstance(lat_event, TimerEvent)


# def test_linear_thinning_timer():
#     dynamics = LinearDynamics()

#     def potential(params, context={}):
#         return jnp.sum(params**2)

#     def rate_fn(state: PDMPState, context={}):
#         _, dpot = jax.jvp(
#             lambda t: potential(dynamics.forward(t, state).params, context),
#             (0.0,),
#             (1.0,),
#         )
#         return dpot

#     ltt = LinearThinningTimer(rate_fn, 1.0, has_aux=False, dynamics=dynamics)
#     state = PDMPState(jnp.ones((2,)), jnp.ones((2,)))
#     ltt_event, ltt_ctx = ltt(jax.random.key(0), state, {"a": 2.0})

#     assert isinstance(ltt_event, TimerEvent)

#     ltt_event, _ = jax.jit(ltt)(jax.random.key(0), state, {"a": 2.0})

#     assert isinstance(ltt_event, TimerEvent)


# def test_quadratic_approx_timer():
#     dynamics = LinearDynamics()

#     def potential(params, context={}):
#         return jnp.sum(params**4)

#     def rate_fn(state: PDMPState, context={}):
#         _, dpot = jax.jvp(
#             lambda t: potential(dynamics.forward(t, state).params, context),
#             (0.0,),
#             (1.0,),
#         )
#         return dpot

#     qat = QuadraticApproxTimer(rate_fn, 1.0, has_aux=False, dynamics=dynamics)
#     state = PDMPState(jnp.ones((2,)), jnp.ones((2,)))
#     qat_event, qat_ctx = qat(jax.random.key(0), state, {"a": 2.0})

#     assert isinstance(qat_event, TimerEvent)

#     qat_event, _ = jax.jit(qat)(jax.random.key(0), state, {"a": 2.0})

#     assert isinstance(qat_event, TimerEvent)

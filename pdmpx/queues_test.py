import jax
import jax.numpy as jnp

from pdmpx import PDMPState
from pdmpx.queues import SimpleFactorQueue
from pdmpx.pdmp import AbstractFactor
from pdmpx.refreshments import ConstantRateRefreshments


def test_simple_factor_queue():
    crr_slow = ConstantRateRefreshments(0.001, normalize_velocities=False)
    crr_fast = ConstantRateRefreshments(100.0, normalize_velocities=True)

    sfq = SimpleFactorQueue([crr_slow, crr_fast])

    state = PDMPState(jnp.zeros((2,)), jnp.zeros((2,)))

    event, ctx = sfq.timer(jax.random.key(0), state, {})
    assert event.time > 0.0
    assert event.bound == 0.0
    # assert ctx == {"simple_factor_queue": {"next_event_idx": 0}}
    new_state = sfq.kernel(jax.random.key(0), state, ctx)
    assert jnp.all(new_state.params == state.params)
    assert not jnp.all(new_state.velocities == state.velocities)
    assert jnp.allclose(jnp.linalg.norm(new_state.velocities), 1.0)

import jax
import jax.numpy as jnp
from pdmpx import PDMPState, TimerEvent
from pdmpx.bouncy import RefreshmentKernel, BPSState


def test_refreshment_kernel():
    key = jax.random.key(0)
    x = jnp.zeros((5,))
    v = jax.random.normal(key, (5,))
    state = BPSState(x, v)

    dummy_event = TimerEvent(1.0, 1.0, {"a": True, "b": {"c": 1.0}})
    nrk = RefreshmentKernel(normalize_velocities=True)
    normed1 = nrk(jax.random.key(1), state, dummy_event)
    normed2 = nrk(jax.random.key(1), state, dummy_event)
    normed3 = nrk(jax.random.key(2), state, dummy_event)
    assert jnp.all(normed1.params == state.params)
    assert jnp.all(normed2.params == state.params)
    assert jnp.all(normed3.params == state.params)
    assert jnp.all(normed1.velocities == normed2.velocities)
    assert jnp.allclose(jnp.linalg.norm(normed1.velocities), 1.0)
    assert not jnp.allclose(normed3.velocities, normed2.velocities)

    rk = RefreshmentKernel(normalize_velocities=False)
    refr1 = rk(jax.random.key(1), state, dummy_event)
    refr2 = rk(jax.random.key(2), state, dummy_event)
    assert jnp.all(refr1.params == state.params)
    assert jnp.all(refr2.params == state.params)
    assert not jnp.all(refr1.velocities == refr2.velocities)

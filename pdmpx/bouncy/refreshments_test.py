import jax
import jax.numpy as jnp
from pdmpx import PDMPState
from pdmpx.refreshments import RefreshmentKernel


def test_refreshment_kernel():
    key = jax.random.key(0)
    x = jnp.zeros((5,))
    v = jax.random.normal(key, (5,))
    state = PDMPState(x, v)

    dummy_context = {"a": True, "b": {"c": 1.0}}
    nrk = RefreshmentKernel(normalize_velocities=True)
    normed1 = nrk(jax.random.key(1), state)
    normed2 = nrk(jax.random.key(1), state, dummy_context)
    normed3 = nrk(jax.random.key(2), state)
    assert jnp.all(normed1.params == state.params)
    assert jnp.all(normed2.params == state.params)
    assert jnp.all(normed3.params == state.params)
    assert jnp.all(normed1.velocities == normed2.velocities)
    assert jnp.allclose(jnp.linalg.norm(normed1.velocities), 1.0)
    assert not jnp.allclose(normed3.velocities, normed2.velocities)

    rk = RefreshmentKernel(normalize_velocities=False)
    refr1 = rk(jax.random.key(1), state)
    refr2 = rk(jax.random.key(2), state, dummy_context)
    assert jnp.all(refr1.params == state.params)
    assert jnp.all(refr2.params == state.params)
    assert not jnp.all(refr1.velocities == refr2.velocities)

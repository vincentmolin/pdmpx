import jax
import jax.numpy as jnp
import tree_math as tm
from pdmpx import PDMPState
from pdmpx.bouncy import BouncyParticleSampler, BPSReflectionFactor
from pdmpx.dynamics import LinearDynamics


def test_bouncy_reflections():
    def potential(x, ctx=None):
        return jnp.sum(x**2) / 2.0

    state = PDMPState(jnp.ones((1,)), jnp.ones((1,)))

    ref_factor = BPSReflectionFactor(potential, normalize_velocities=False)

    tev, ctx = ref_factor.timer(jax.random.key(0), state, {})
    assert ctx == {}
    assert tev.time > 0.0
    assert tev.bound == 0.0

    forw_state = LinearDynamics().forward(tev.time, state)
    new_state = ref_factor.kernel(jax.random.key(0), forw_state, ctx)
    assert jnp.all(new_state.params == forw_state.params)
    assert jnp.allclose(new_state.velocities, -forw_state.velocities)
    assert jnp.allclose(
        BPSReflectionFactor(potential, normalize_velocities=True)
        .kernel(jax.random.key(0), forw_state, ctx)
        .velocities,
        -1.0,
    )


def test_bouncy_particle_sampler():
    def potential(x, ctx=None):
        return jnp.sum(x**2) / 2.0 + jnp.sum(x**4) / 4.0

    bps = BouncyParticleSampler(potential, 0.001, normalize_velocities=False)
    state = PDMPState(jnp.ones((3,)), jnp.ones((3,)))

    ev, ctx, dirt = jax.jit(bps.get_next_event)(jax.random.key(0), state, {})
    assert dirt
    assert jnp.allclose(ev.new_state.velocities, -state.velocities)
    assert len(bps.simulate(jax.random.key(0), state, 2.0, {})) > 1

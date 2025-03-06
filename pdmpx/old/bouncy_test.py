import jax
import jax.numpy as jnp
import tree_math as tm
from pdmpx import PDMPState
from pdmpx.bouncy import BouncyParticleSampler, BPSBounceFactor
from pdmpx.dynamics import LinearDynamics
import pdmpx


def test_bouncy_bounces():
    def potential(x, ctx=None):
        return jnp.sum(x**2) / 2.0

    state = PDMPState(jnp.ones((1,)), jnp.ones((1,)))

    ref_factor = BPSBounceFactor(potential, normalize_velocities=False)

    tev, ctx = ref_factor.timer(jax.random.key(0), state, {})
    assert ctx == {}
    assert tev.time > 0.0
    assert tev.bound == 0.0

    forw_state = LinearDynamics().forward(tev.time, state)
    new_state = ref_factor.kernel(jax.random.key(0), forw_state, ctx)
    assert jnp.all(new_state.params == forw_state.params)
    assert jnp.allclose(new_state.velocities, -forw_state.velocities)
    assert jnp.allclose(
        BPSBounceFactor(potential, normalize_velocities=True)
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

    def potential_no_ctx(x):
        return potential(x, {})

    bps = BouncyParticleSampler(potential, 0.001, normalize_velocities=False)
    state = PDMPState(jnp.ones((3,)), jnp.ones((3,)))

    ev, ctx, dirt = jax.jit(bps.get_next_event)(jax.random.key(0), state, {})
    assert dirt
    assert jnp.allclose(ev.new_state.velocities, -state.velocities)
    assert len(bps.simulate(jax.random.key(0), state, 2.0, {})) > 1


def test_cold_bps():
    # TODO: Simplify this test
    def rosenbrock(x):
        return jnp.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

    x0 = jnp.array([-1.0, 1.0])
    v0 = jax.grad(rosenbrock)(x0)
    state = PDMPState(params=x0, velocities=v0)

    T = 0.1
    gradient_mix = 0.2
    coldness = lambda t: (t + 1) * 0.5
    refreshment_rate = 0.1

    @jax.jit
    def next_event(rng, state, context):
        rng, key = jax.random.split(rng)

        cold = coldness(context["time"])
        gmix = gradient_mix  # (context["time"])

        return rng, *pdmpx.bouncy.ColdBouncyParticleSampler(
            potential=rosenbrock,
            refreshment_rate=refreshment_rate,
            gradient_mix=gmix,
            coldness=cold,
            valid_time=0.1,
            normalize_velocities=True,
        ).get_next_event(key, state, context)

    events = []
    context = {"time": 0.0}
    rng = jax.random.key(0)

    while context["time"] < T:
        rng, ev, context, dirty = next_event(rng, state, context)
        context["time"] = ev.time
        state = ev.new_state
        if dirty:
            events.append(ev)

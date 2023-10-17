import jax
import jax.numpy as jnp
import numba
import numpy as np
import functools
from tqdm import tqdm
import matplotlib.pyplot as plt
from .poisson_time import ab_poisson_time

# from pdmp.poisson_time import ab_poisson_time


def linear_dynamics(x, v, dt):
    """
    Linear dynamics
    """
    return x + dt * v, v


@jax.jit
def project_ab(a, b):
    """
    Project the vector a onto b
    """
    c = jnp.dot(a, b) / jnp.dot(b, b)
    return c * b


@jax.jit
def unit_vector(x, eps=1e-8):
    return x / (jnp.linalg.norm(x) + eps)


@functools.partial(jax.jit, static_argnames=("normalize",))
def oscn(key, v, grad_psi, rho, normalize=False):
    """
    Resample v using the oscn method
    """
    v_p = project_ab(v, grad_psi)
    v_t = v - v_p
    z = jax.random.normal(key, v.shape)
    z = z - project_ab(z, grad_psi)
    v_new = -v_p + rho * v_t + jnp.sqrt(1 - rho**2) * z
    return unit_vector(v_new) if normalize else v_new


def oscnbps_jax(
    grad_psi_fn, x0, v0, T, delta_max=1.0, c=1.0, rho=0.98, seed=jax.random.PRNGKey(0)
):
    """
    OSCN Bouncy Particle Sampler targeting the distribution with density proportional to exp(-psi(x))

    Args:
        grad_psi_fn: function that returns the gradient of psi
        x0: initial position
        v0: initial velocity
        T: time horizon
        delta_max: maximum time step
        c: tuning parameter
        rho: tuning parameter
        seed: random seed
    Returns:
        xs: list of event positions
        vs: list of velocities
        ts: list of event times
        accepted: number of accepted proposals
        suggested: number of suggested proposals
    """

    t = 0.0
    x = x0
    v = v0
    accepted = 0
    suggested = 0
    grad_evals = 0

    xs = [np.array(jax.device_get(x))]
    vs = [np.array(jax.device_get(v))]
    ts = [0.0]

    @jax.jit
    def rate_fn(x, v):
        grad = grad_psi_fn(x)
        return jnp.maximum(jnp.dot(grad, v), 0.0)

    # normv = jnp.linalg.norm(v)

    rng = seed

    @jax.jit
    def _next_event_candidate(rng, x, v, c):
        """
        Simulates forward until the next event candidate time
        """

        def body_fn(rng, t, tau, x, v, rate_bound, grad_evals):
            t += tau
            x += tau * v
            rate_bound = rate_fn(x, v) + c * delta_max * jnp.linalg.norm(v)
            grad_evals += 1
            rng, key = jax.random.split(rng)
            tau = jnp.minimum(jax.random.exponential(key) / rate_bound, delta_max)
            return rng, t, tau, x, v, rate_bound, grad_evals

        cond_fn = lambda rng, t, tau, x, v, rate_bound, grad_evals: tau == delta_max
        init_val = body_fn(rng, 0.0, 0.0, x, v, 0.0, 0)
        rng, t, tau, x, v, rate_bound, grad_evals = jax.lax.while_loop(
            lambda args: cond_fn(*args), lambda args: body_fn(*args), init_val
        )
        return t + tau, x + tau * v, rate_bound, grad_evals

    @jax.jit
    def _next_event(rng, x, v, c):
        """
        Simulates forward until the next reflection event is accepted
        """
        grad_evals = 0

        def cond_fn(rng, t, x, v, u, a, sug, grad_evals):
            return (u >= a) * (a < 1.0)

        def body_fn(rng, t, x, v, u, a, sug, grad_evals):
            rng, ukey, ekey = jax.random.split(rng, 3)
            u = jax.random.uniform(ukey)
            tau, x, rate_bound, evals = _next_event_candidate(ekey, x, v, c)
            grad_evals += evals
            t += tau
            sug += 1
            a = rate_fn(x, v) / rate_bound
            return rng, t, x, v, u, a, sug, grad_evals

        init_val = body_fn(rng, 0.0, x, v, None, None, 0, 0)

        rng, t, x, v, u, a, sug, grad_evals = jax.lax.while_loop(
            lambda args: cond_fn(*args), lambda args: body_fn(*args), init_val
        )

        rng, key = jax.random.split(rng)
        v = oscn(key, v, grad_psi_fn(x), rho)

        return rng, t, x, v, sug, grad_evals + 1, a

    with tqdm(total=T) as pbar:
        while t < T:
            rng, tau, xo, vo, sug, evals, a = _next_event(rng, x, v, c)
            if a > 1.0:
                c = c * 2.0
                print(f"a > 1.0, doubling c. c = {c}")
            else:
                x = xo
                v = vo
                t += tau
                suggested += sug
                grad_evals += evals
                accepted += 1
                xs.append(np.array(jax.device_get(x)))
                vs.append(np.array(jax.device_get(v)))
                ts.append(t)
                pbar.update(float(tau))

    return xs, vs, ts, accepted, suggested, grad_evals


def oscnbps(
    psi_fn,
    x0,
    v0,
    T,
    dynamics=linear_dynamics,
    delta_max=1.0,
    c=[1.0, 1.0],
    rho=0.98,
    seed=jax.random.PRNGKey(0),
):
    """
    OSCN Bouncy Particle Sampler targeting the distribution with density proportional to exp(-psi(x))

    Gradient evaluations done on GPU, the rest is computed CPU side

    Args:
        grad_psi_fn: function that returns the gradient of psi
        x0: initial position
        v0: initial velocity
        T: time horizon
        delta_max: maximum time step
        c: tuning parameter
        rho: tuning parameter
        seed: random seed
    Returns:
        xs: list of event positions
        vs: list of velocities
        ts: list of event times
        accepted: number of accepted proposals
        suggested: number of suggested proposals
    """

    t = 0.0
    x = x0
    v = v0
    accepted = 0
    suggested = 0
    delta_max_hits = 0
    grad_evals = 0

    xs = [np.array(jax.device_get(x))]
    vs = [np.array(jax.device_get(v))]
    ts = [0.0]

    @jax.jit
    def grad_psi_fn(x):
        """
        Gradient of psi
        """
        return jax.grad(psi_fn)(x)

    @jax.jit
    def rate_fn(x, v):
        """
        (Negative) rate of the Poisson process
        """
        _, rate = jax.jvp(lambda t: psi_fn(dynamics(x, v, t)[0]), (0.0,), (1.0,))
        return rate

    @jax.jit
    def drate_fn(x, v):
        """
        Derivative of the rate of the Poisson process
        """
        rate, drate = jax.jvp(lambda t: rate_fn(*dynamics(x, v, t)), (0.0,), (1.0,))
        return rate, drate

    rng = seed

    with tqdm(total=T) as pbar:
        while t < T:
            rate, drate = np.array(drate_fn(x, v))
            grad_evals += 1
            a, b = rate + c[0], drate + c[1]
            u = np.random.uniform()
            tau = np.minimum(ab_poisson_time(u, a, b), delta_max)
            x, v = dynamics(x, v, tau)
            t += tau
            pbar.update(float(tau))
            if tau == delta_max:
                delta_max_hits += 1
            else:
                # x, v = dynamics(x, v, tau)
                rate = np.array(rate_fn(x, v))
                acceptance_probability = np.maximum(rate, 0.0) / (a + tau * b)
                u = np.random.uniform()
                suggested += 1
                grad_evals += 1
                if acceptance_probability > 1.0:
                    print(f"acceptance_probability > 1.0: {acceptance_probability}")
                if u < acceptance_probability:
                    accepted += 1
                    rng, key = jax.random.split(rng)
                    v = oscn(key, v, grad_psi_fn(x), rho)
                    xs.append(np.array(jax.device_get(x)))
                    vs.append(np.array(jax.device_get(v)))
                    ts.append(t)

    return xs, vs, ts, accepted, suggested, grad_evals, delta_max_hits


def oscnbps_interactive(
    psi_fn,
    x0,
    v0,
    T,
    dynamics=linear_dynamics,
    delta_max=1.0,
    c=[1.0, 1.0],
    rho=0.98,
    seed=jax.random.PRNGKey(0),
    last_txt="c",
    plot_callbacks=[],
):
    """
    OSCN Bouncy Particle Sampler targeting the distribution with density proportional to exp(-psi(x))

    Gradient evaluations done on GPU, the rest is computed CPU side

    Args:
        grad_psi_fn: function that returns the gradient of psi
        x0: initial position
        v0: initial velocity
        T: time horizon
        delta_max: maximum time step
        c: tuning parameter
        rho: tuning parameter
        seed: random seed
    Returns:
        xs: list of event positions
        vs: list of velocities
        ts: list of event times
        accepted: number of accepted proposals
        suggested: number of suggested proposals
    """

    t = 0.0
    x = x0
    v = v0
    accepted = 0
    suggested = 0
    delta_max_hits = 0
    grad_evals = 0

    xs = [np.array(jax.device_get(x))]
    vs = [np.array(jax.device_get(v))]
    ts = [0.0]

    @jax.jit
    def grad_psi_fn(x):
        """
        Gradient of psi
        """
        return jax.grad(psi_fn)(x)

    @jax.jit
    def rate_fn(x, v):
        """
        (Negative) rate of the Poisson process
        """
        _, rate = jax.jvp(lambda t: psi_fn(dynamics(x, v, t)[0]), (0.0,), (1.0,))
        return rate

    @jax.jit
    def drate_fn(x, v):
        """
        Derivative of the rate of the Poisson process
        """
        rate, drate = jax.jvp(lambda t: rate_fn(*dynamics(x, v, t)), (0.0,), (1.0,))
        return rate, drate

    vrate_fn = jax.vmap(rate_fn, in_axes=(0, 0))
    vdynamics = jax.vmap(dynamics, in_axes=(None, None, 0))

    def plot_forward(x, v, rate, drate, t, c, tau, t_horizon, n_samples=100):
        tmesh = jnp.linspace(0, t_horizon, n_samples)
        xmesh, vmesh = vdynamics(x, v, tmesh)
        rates = vrate_fn(xmesh, vmesh)

        plt.plot(tmesh + t, rates, "x-", markersize=4, label="rates")
        plt.plot(
            [t, t + t_horizon],
            [rate, rate + drate * t_horizon],
            "--",
            label="linear approx",
        )
        plt.plot(
            [t, t + t_horizon],
            [rate + c[0], rate + c[0] + (drate + c[1]) * t_horizon],
            "--",
            label="with slack",
        )

        if tau < t_horizon:
            plt.axvline(t + tau, label="proposal")
            rate_tau = rate_fn(*dynamics(x, v, tau))
            bound_tau = rate + c[0] + (drate + c[1]) * tau
            plt.plot([t + tau], [rate_tau], "o", label="proposal rate")
            plt.plot([t + tau], [bound_tau], "o", label="proposal bound")
        else:
            plt.title("delta_max hit")

        plt.axhline(0.0, color="black")
        plt.ylabel("rate")
        plt.xlabel("time")
        plt.legend()
        plt.show()

    rng = seed

    last_txt = last_txt
    next_plot = 0.0

    def handle_input(txt):
        nonlocal last_txt
        nonlocal next_plot
        if txt == "c":
            last_txt = txt
            next_plot = 0.0
        elif txt == "":
            handle_input(last_txt)
        elif txt == "q":
            raise KeyboardInterrupt
        elif txt.startswith("s"):
            next_plot = next_plot + float(txt[1:])
            last_txt = txt
        elif txt.startswith("t"):
            next_plot = float(txt[1:])
            last_txt = txt
        else:
            print(f"Unknown command: {txt}")
            handle_input(input("Enter command: "))

    # with tqdm(total=T) as pbar:
    while t < T:
        rate, drate = np.array(drate_fn(x, v))
        grad_evals += 1
        a, b = rate + c[0], drate + c[1]
        u = np.random.uniform()
        tau = np.minimum(ab_poisson_time(u, a, b), delta_max)

        if t + tau > next_plot:
            plot_forward(x, v, rate, drate, t, c, tau, delta_max)
            for cb in plot_callbacks:
                cb(x, v, t, tau)
            handle_input(input("Enter command:"))

        x, v = dynamics(x, v, tau)
        t += tau
        #       pbar.update(float(tau))
        if tau == delta_max:
            delta_max_hits += 1
        else:
            # x, v = dynamics(x, v, tau)
            rate = np.array(rate_fn(x, v))
            acceptance_probability = np.maximum(rate, 0.0) / (a + tau * b)
            u = np.random.uniform()
            suggested += 1
            grad_evals += 1
            if acceptance_probability > 1.0:
                print(f"acceptance_probability > 1.0: {acceptance_probability}")
            if u < acceptance_probability:
                print(f"Event accepted at t={t}")
                accepted += 1
                rng, key = jax.random.split(rng)
                v = oscn(key, v, grad_psi_fn(x), rho)
                xs.append(np.array(jax.device_get(x)))
                vs.append(np.array(jax.device_get(v)))
                ts.append(t)
            else:
                print(f"Event rejected at t={t}")

    return xs, vs, ts, accepted, suggested, grad_evals, delta_max_hits


# if __name__=="__main__":
#     x0 = z
#     delta_max = 0.05
#     c = [1.0,1.0]
#     dynamics = linear_dynamics

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate


@jax.jit
def f(x):
    return (
        0.5 * x.T @ x
        + 0.5 * x.T @ jnp.array([1.0, 0.0])
        + 0.5
        + jnp.sum((x - 0.25) ** 4)
        + 2.0 * jnp.sin(5.0 * x[0])
        + 2.0 * jnp.sin(5.0 * x[1])
    )


def contourplot(ax, f, x1, x2, n=100, fill=False):
    """
    Returns a figure with a contour plot of f and a colorbar.
    """
    x1 = jnp.linspace(*x1, n)
    x2 = jnp.linspace(*x2, n)
    X1, X2 = jnp.meshgrid(x1, x2)
    Z = jnp.array([f(jnp.array([x1, x2])) for x1, x2 in zip(X1.ravel(), X2.ravel())])
    Z = Z.reshape(X1.shape)
    if fill:
        CS = ax.contourf(X1, X2, Z, 20)
    else:
        CS = ax.contour(X1, X2, Z, 20)
    # ax.clabel(CS, inline=1, fontsize=10)
    # fig.colorbar(CS)
    return ax


fig, ax = plt.subplots()
contourplot(ax, f, [-1.5, 1.5], [-1.5, 1.5], fill=True)


@jax.jit
def step(x, v, dt, elasticity=1.0):
    x0 = x + (dt / 2) * v
    gf = jax.grad(f)(x0)
    v = v - jnp.dot(gf, v) / jnp.dot(gf, gf) * gf
    v = elasticity * v - (1 - elasticity) * gf / jnp.sqrt(jnp.dot(gf, gf))
    v = v / jnp.sqrt(jnp.dot(v, v))
    x = x + dt * v
    return x, v


x0 = jnp.array([-0.7, -0.7])
v0 = jnp.array([1.0, 0.0])

fig, ax = plt.subplots()
ax = contourplot(ax, f, [-1.5, 1.5], [-1.5, 2.25], fill=True)
dt = 0.01
n = 1000
for el in [1.0, 0.9, 0.8]:
    x = x0
    v = v0
    xs = []
    vs = []
    for i in range(n):
        x, v = step(x, v, dt, elasticity=el)
        xs.append(x)
        vs.append(v)

    ax.plot([x[0] for x in xs], [x[1] for x in xs], label=f"el = {el}")

for rate in [0.5, 1, 2]:
    x = x0
    v = v0
    xs = []
    vs = []
    for tt in np.linspace(0, 1, n):
        x, v = step(x, v, dt, elasticity=1 - tt**rate)
        xs.append(x)
        vs.append(v)

    ax.plot([x[0] for x in xs], [x[1] for x in xs], label=f"el = 1-t**{rate}")
ax.legend()
fig


@jax.jit
def step(x, v, dt, elasticity=1.0, temp=1.0):
    x0 = x + (dt / 2) * v
    gf = jax.grad(f)(x0)
    v = v - jnp.dot(gf, v) / jnp.dot(gf, gf) * gf
    v = v / jnp.sqrt(jnp.dot(v, v))
    v = elasticity * v - (1 - elasticity) * gf / jnp.sqrt(jnp.dot(gf, gf))
    x = x + dt * v
    return x, v


x0 = jnp.array([-0.7, -0.7])
v0 = jnp.array([1.0, 0.0])

fig, ax = plt.subplots(2, 1)
ax = contourplot(ax, f, [-1.5, 1.5], [-1.5, 2.25], fill=True)
dt = 0.01
n = 1000
for el in [1.0, 0.9, 0.8]:
    x = x0
    v = v0
    xs = []
    vs = []
    for i in range(n):
        x, v = step(x, v, dt, elasticity=el)
        xs.append(x)
        vs.append(v)

    ax.plot([x[0] for x in xs], [x[1] for x in xs], label=f"el = {el}")

ax.legend()
fig


beta1 = 1.0
Z1 = integrate.dblquad(
    lambda x, y: np.exp(-beta1 * f(np.array([x, y]))), -np.inf, np.inf, -np.inf, np.inf
)
coldf1 = lambda x: -beta1 * f(x) - np.log(Z1[0])
fig, axs = plt.subplots(1, 2)
contourplot(axs[0], coldf1, [-1.5, 1.5], [-1.5, 2.25])
fig
beta2 = 2.0
Z2 = integrate.dblquad(
    lambda x, y: np.exp(-beta2 * f(np.array([x, y]))), -np.inf, np.inf, -np.inf, np.inf
)
coldf2 = lambda x: -beta2 * f(x) - np.log(Z2[0])
ax = contourplot(axs[1], coldf2, [-1.5, 1.5], [-1.5, 2.25])

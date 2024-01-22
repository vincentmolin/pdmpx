import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


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


def contourplot(f, x1, x2, n=100, fill=False):
    """
    Returns a figure with a contour plot of f and a colorbar.
    """
    x1 = jnp.linspace(*x1, n)
    x2 = jnp.linspace(*x2, n)
    X1, X2 = jnp.meshgrid(x1, x2)
    Z = jnp.array([f(jnp.array([x1, x2])) for x1, x2 in zip(X1.ravel(), X2.ravel())])
    Z = Z.reshape(X1.shape)
    fig, ax = plt.subplots()
    if fill:
        CS = ax.contourf(X1, X2, Z, 20)
    else:
        CS = ax.contour(X1, X2, Z, 20)
    # ax.clabel(CS, inline=1, fontsize=10)
    # fig.colorbar(CS)
    return fig, ax


fig, ax = contourplot(f, [-1.5, 1.5], [-1.5, 1.5])
fig


@jax.jit
def step(x, v, dt, elasticity=1.0):
    x0 = x + (dt / 2) * v
    gf = jax.grad(f)(x0)
    v = v - jnp.dot(gf, v) / jnp.dot(gf, gf) * gf
    v = v / jnp.sqrt(jnp.dot(v, v))
    v = elasticity * v - (1 - elasticity) * gf / jnp.sqrt(jnp.dot(gf, gf))
    x = x + dt * v
    return x, v


x0 = jnp.array([-0.7, -0.7])
v0 = jnp.array([1.0, 0.0])

fig, ax = contourplot(f, [-1.5, 1.5], [-1.5, 2.25], fill=True)
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

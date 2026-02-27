import numpy as np
from scipy.interpolate import make_interp_spline


def discretize_trajectory(xs, ts, T, mesh=None, dt=0.1, k=1):
    """
    Discretize a trajectory (xs, ts) to a mesh of times `mesh` if given
    or with the mesh `np.arange(0,T,dt)`.
    """
    xs = np.array(xs)  # .transpose()
    ts = np.array(ts)
    linear_spline = make_interp_spline(ts, xs, k=k)
    mesh = mesh if mesh is not None else np.arange(0, T, dt)
    return linear_spline(mesh)

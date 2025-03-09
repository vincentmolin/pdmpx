from inspect import signature
import functools as ft
import jax.random as jr


def _copy_fn_metadata(fn, new_fn):
    if fn.__name__ is not None:
        new_fn.__name__ = fn.__name__

    if fn.__doc__ is not None:
        new_fn.__doc__ = fn.__doc__

    if fn.__annotations__ is not None:
        new_fn.__annotations__ = fn.__annotations__

    return new_fn


def rng_wrap(fn):
    def new_fn(rng, *args, **kwargs):
        rng, key = jr.split(rng)
        return rng, *fn(key, *args, **kwargs)

    return _copy_fn_metadata(fn, new_fn)


def num_args(fn):
    return len(signature(fn).parameters)


def maybe_add_dummy_args(fn):
    if num_args(fn) >= 2:
        return fn
    elif num_args(fn) == 1:

        def new_fn(x, *_):
            return fn(x)

    else:

        def new_fn(x, *_):
            return fn()

    return _copy_fn_metadata(fn, new_fn)

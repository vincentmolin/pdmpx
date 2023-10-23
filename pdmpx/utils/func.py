from inspect import signature
import functools as ft


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

    if fn.__name__ is not None:
        new_fn.__name__ = fn.__name__

    if fn.__doc__ is not None:
        new_fn.__doc__ = fn.__doc__
    return new_fn

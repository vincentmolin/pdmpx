from pdmpx.utils.func import num_args, maybe_add_dummy_args


def test_num_args():
    assert num_args(lambda x: x) == 1
    assert num_args(lambda x, y: x) == 2
    assert num_args(lambda x, *_: x) >= 2

    def f1(x):
        return x

    def f2(x, y):
        return x

    def f3(x, *_):
        return x

    assert num_args(f1) == 1
    assert num_args(f2) == 2
    assert num_args(f3) >= 2


def test_maybe_add_dummy_args():
    _constant = lambda: 1.0
    constant = maybe_add_dummy_args(_constant)
    assert num_args(constant) >= 2
    assert constant(None, None) == 1.0

    _identity = lambda x: x
    identity = maybe_add_dummy_args(_identity)
    assert identity(1.0) == 1.0
    assert identity(1.0, None) == 1.0
    assert num_args(identity) >= 2

    _with_ctx = lambda x, ctx: x
    with_ctx = maybe_add_dummy_args(_with_ctx)
    assert with_ctx(1.0, {}) == 1.0
    assert num_args(with_ctx) == num_args(_with_ctx)

    def _with_doc(x, ctx):
        """Docstring"""
        return x

    with_doc = maybe_add_dummy_args(_with_doc)
    assert with_doc(1.0, {}) == 1.0
    assert with_doc.__doc__ == _with_doc.__doc__
    assert with_doc.__name__ == _with_doc.__name__

    def _with_doc_no_ctx(x):
        """Docstring"""
        return x

    with_doc_no_ctx = maybe_add_dummy_args(_with_doc_no_ctx)
    assert with_doc_no_ctx(1.0, {}) == 1.0
    assert with_doc_no_ctx.__doc__ == _with_doc_no_ctx.__doc__
    assert with_doc_no_ctx.__name__ == _with_doc_no_ctx.__name__

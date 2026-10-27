import itertools
import math

import numpy as np

import unittest
import numba
from numba import (deferred_type, float64, jit, literal_unroll, njit,
                   objmode, optional)
from numba.core import types
from numba.experimental import jitclass
from numba.tests.support import TestCase


def return_double_or_none(x):
    if x:
        ret = None
    else:
        ret = 1.2
    return ret


def return_different_statement(x):
    if x:
        return None
    else:
        return 1.2


def return_bool_optional_or_none(x, y):
    if y:
        z = False
    else:
        z = None
    if x == 2:
        # A boolean
        return True
    elif x == 1:
        # A runtime optional
        return z
    else:
        # None
        return None


def is_this_a_none(x):
    if x:
        val_or_none = None
    else:
        val_or_none = x

    if val_or_none is None:
        return x - 1

    if val_or_none is not None:
        return x + 1


def a_is_b(a, b):
    """
    Note in nopython mode, this operation does not make much sense.
    Because we don't have objects anymore.
    `a is b` is always False if not operating on None and Optional type
    """
    return a is b


def a_is_not_b(a, b):
    """
    This is `not (a is b)`
    """
    return a is not b


class TestOptional(TestCase):

    _numba_parallel_test_ = False

    def test_return_double_or_none(self):
        pyfunc = return_double_or_none
        cfunc = njit((types.boolean,))(pyfunc)

        for v in [True, False]:
            self.assertPreciseEqual(pyfunc(v), cfunc(v))

    def test_return_different_statement(self):
        pyfunc = return_different_statement
        cfunc = njit((types.boolean,))(pyfunc)

        for v in [True, False]:
            self.assertPreciseEqual(pyfunc(v), cfunc(v))

    def test_return_bool_optional_or_none(self):
        pyfunc = return_bool_optional_or_none
        cfunc = njit((types.int32, types.int32,))(pyfunc)

        for x, y in itertools.product((0, 1, 2), (0, 1)):
            self.assertPreciseEqual(pyfunc(x, y), cfunc(x, y))

    def test_is_this_a_none(self):
        pyfunc = is_this_a_none
        cfunc = njit((types.intp,))(pyfunc)

        for v in [-1, 0, 1, 2]:
            self.assertPreciseEqual(pyfunc(v), cfunc(v))

    def test_is_this_a_none_objmode(self):
        pyfunc = is_this_a_none
        cfunc = jit((types.intp,), forceobj=True)(pyfunc)
        self.assertTrue(cfunc.overloads[cfunc.signatures[0]].objectmode)
        for v in [-1, 0, 1, 2]:
            self.assertPreciseEqual(pyfunc(v), cfunc(v))

    def test_a_is_b_intp(self):
        pyfunc = a_is_b
        cfunc = njit((types.intp, types.intp))(pyfunc)
        # integer identity relies on `==`
        self.assertTrue(cfunc(1, 1))
        self.assertFalse(cfunc(1, 2))

    def test_a_is_not_b_intp(self):
        pyfunc = a_is_not_b
        cfunc = njit((types.intp, types.intp))(pyfunc)
        # integer identity relies on `==`
        self.assertFalse(cfunc(1, 1))
        self.assertTrue(cfunc(1, 2))

    def test_optional_float(self):
        def pyfunc(x, y):
            if y is None:
                return x
            else:
                return x + y

        cfunc = njit("(float64, optional(float64))")(pyfunc)
        self.assertAlmostEqual(pyfunc(1., 12.3), cfunc(1., 12.3))
        self.assertAlmostEqual(pyfunc(1., None), cfunc(1., None))

    def test_optional_array(self):
        def pyfunc(x, y):
            if y is None:
                return x
            else:
                y[0] += x
                return y[0]

        cfunc = njit("(float32, optional(float32[:]))")(pyfunc)
        cy = np.array([12.3], dtype=np.float32)
        py = cy.copy()
        self.assertAlmostEqual(pyfunc(1., py), cfunc(1., cy))
        np.testing.assert_almost_equal(py, cy)
        self.assertAlmostEqual(pyfunc(1., None), cfunc(1., None))

    def test_optional_array_error(self):
        def pyfunc(y):
            return y[0]

        cfunc = njit("(optional(int32[:]),)")(pyfunc)
        with self.assertRaises(TypeError) as raised:
            cfunc(None)
        self.assertIn('expected array(int32, 1d, A), got None',
                      str(raised.exception))

        y = np.array([0xabcd], dtype=np.int32)
        self.assertEqual(cfunc(y), pyfunc(y))

    def test_optional_array_attribute(self):
        """
        Check that we can access attribute of an optional
        """
        def pyfunc(arr, do_it):
            opt = None
            if do_it:  # forces `opt` to be an optional of arr
                opt = arr
            return opt.shape[0]

        cfunc = njit(pyfunc)
        arr = np.arange(5)
        self.assertEqual(pyfunc(arr, True), cfunc(arr, True))

    def test_assign_to_optional(self):
        """
        Check that we can assign to a variable of optional type
        """
        @njit
        def make_optional(val, get_none):
            if get_none:
                ret = None
            else:
                ret = val
            return ret

        @njit
        def foo(val, run_second):
            a = make_optional(val, True)
            if run_second:
                a = make_optional(val, False)
            return a

        self.assertIsNone(foo(123, False))
        self.assertEqual(foo(231, True), 231)

    def test_optional_thru_omitted_arg(self):
        """
        Issue 1868
        """

        def pyfunc(x=None):
            if x is None:
                x = 1
            return x

        cfunc = njit(pyfunc)
        self.assertEqual(pyfunc(), cfunc())
        self.assertEqual(pyfunc(3), cfunc(3))

    def test_optional_unpack(self):
        """
        Issue 2171
        """
        def pyfunc(x):
            if x is None:
                return
            else:
                a, b = x
                return a, b

        tup = types.Tuple([types.intp] * 2)
        opt_tup = types.Optional(tup)
        sig = (opt_tup,)
        cfunc = njit(sig)(pyfunc)
        self.assertEqual(pyfunc(None), cfunc(None))
        self.assertEqual(pyfunc((1, 2)), cfunc((1, 2)))

    def test_many_optional_none_returns(self):
        """
        Issue #4058
        """
        @njit
        def foo(maybe):
            lx = None
            if maybe:
                lx = 10
            return 1, lx

        def work():
            tmp = []
            for _ in range(20000):
                maybe = False
                _ = foo(maybe)

        # this caused "Fatal Python error: deallocating None" as there was no
        # incref being made on the returned None.
        work()


class TestExplicitOptionalCasts(TestCase):
    """
    Each cast of an Optional to its value is a statement of its own, which
    the parfor passes see.
    """

    _numba_parallel_test_ = False

    def compile_both(self, pyfunc, sig=None):
        # plain, and with parfors lowered sequentially
        return [njit(sig, **options)(pyfunc)
                for options in ({}, {'parallel': {'sequential': True}})]

    def test_tuple_item(self):
        def pyfunc(c):
            u = (1, 2.5) if c else None
            if u is None:
                return 0.0
            return u[0] + u[1]

        for cfunc in self.compile_both(pyfunc):
            self.assertPreciseEqual(cfunc(True), pyfunc(True))
            self.assertPreciseEqual(cfunc(False), pyfunc(False))

    def test_merged_array(self):
        def pyfunc(a, c):
            u = a if c else None
            if u is None:
                return None
            return u + 1.0

        a = np.arange(6.0).reshape(2, 3)
        for cfunc in self.compile_both(pyfunc):
            self.assertPreciseEqual(cfunc(a, True), pyfunc(a, True))
            self.assertIsNone(cfunc(a, False))

    def test_array_argument(self):
        def pyfunc(u):
            return u + 1.0

        a = np.arange(6.0).reshape(2, 3)
        sig = "float64[:, ::1](optional(float64[:, ::1]))"
        for cfunc in self.compile_both(pyfunc, sig):
            self.assertPreciseEqual(cfunc(a), pyfunc(a))
            with self.assertRaises(TypeError) as raised:
                cfunc(None)
            self.assertIn('expected array(float64, 2d, C), got None',
                          str(raised.exception))

    def test_fusion(self):
        # one cast of u for the block, so the expressions on it fuse
        def pyfunc(a, c):
            u = a if c else None
            if u is None:
                return a
            x = u + 1.0
            y = u * 2.0
            return x + y

        a = np.arange(6.0).reshape(2, 3)
        cfunc = njit(parallel={'sequential': True})(pyfunc)
        self.assertPreciseEqual(cfunc(a, True), pyfunc(a, True))
        diagnostics = cfunc.get_metadata(cfunc.signatures[0])
        diagnostics = diagnostics['parfor_diagnostics']
        fused = sum(map(len, diagnostics.fusion_info.values()))
        self.assertEqual(len(diagnostics.initial_parfors) - fused, 1)

    def test_call_arguments(self):
        # cast where a callee takes the value; kept by keyword and in a star
        # parameter, where callees take the Optional
        generic = njit(lambda a, b=2.0, c=3.0: a + 10 * b + 100 * c)

        @njit
        def star(a, *rest):
            s = a
            for x in literal_unroll(rest):
                if x is not None:
                    s += x
            return s

        def pyfunc(c):
            u = 1.5 if c else None
            if u is None:
                return star(1.0, u, 2.0)
            return (math.copysign(u, -1.0) + generic(1.0, c=u)
                    + star(1.0, u, 2.0))

        for cfunc in self.compile_both(pyfunc):
            self.assertPreciseEqual(cfunc(True), pyfunc(True))
            # star takes the Optional, None too (from Python, it can't)
            self.assertPreciseEqual(cfunc(False), 3.0)

    def test_objmode_argument(self):
        # objmode boxes an Optional as it is typed: None reaches the block
        def pyfunc(c):
            u = 1.5 if c else None
            with objmode(out='float64'):
                out = -1.0 if u is None else u * 2
            return out

        for cfunc in self.compile_both(pyfunc):
            self.assertPreciseEqual(cfunc(True), 3.0)
            self.assertPreciseEqual(cfunc(False), -1.0)

    def test_generator_return(self):
        # a generator's return value isn't lowered
        @njit
        def gen(c):
            u = 1 if c else None
            yield 1
            return u

        def pyfunc(c):
            s = 0
            for x in gen(c):
                s += x
            return s

        for cfunc in self.compile_both(pyfunc):
            self.assertPreciseEqual(cfunc(True), pyfunc(True))
            self.assertPreciseEqual(cfunc(False), pyfunc(False))

    def test_deferred_type(self):
        # a deferred type defined as an Optional takes the Optional
        optional_float = deferred_type()

        @jitclass([('value', optional_float)])
        class Holder:
            def __init__(self):
                self.value = None

        optional_float.define(optional(float64))

        def pyfunc(c):
            h = Holder()
            u = 1.5 if c else None
            h.value = u
            return h

        for cfunc in self.compile_both(pyfunc):
            self.assertEqual(cfunc(True).value, 1.5)
            self.assertIsNone(cfunc(False).value)

    def test_lifted_loop(self):
        # loop lifting compiles the loop for each signature from shared IR
        @jit(forceobj=True, looplift=True)
        def cfunc(n, c, d):
            y = d['a']
            s = y
            for i in range(n):
                u = y if c else None
                if u is not None:
                    s = s + u
            return s

        for d in ({'a': 1.5}, {'a': 2}):
            self.assertPreciseEqual(cfunc(3, True, d),
                                    cfunc.py_func(3, True, d))

    def test_stencil_argument(self):
        # numba.stencil's options reach the call as keywords its signature
        # doesn't bind: StencilPass rewrites the call, which isn't lowered
        def pyfunc(n, w, c):
            def kernel(a, w):
                cum = a[-w + 1]
                for i in range(-w + 1, w + 1):
                    cum += a[i + 1]
                return 0.3 * cum
            a = np.arange(n * 1.0)
            v = w if c else None
            return (numba.stencil(kernel, neighborhood=((-w, w),),
                                  index_offsets=(-1,))(a, v),
                    numba.stencil(kernel, neighborhood=((-w, w),),
                                  index_offsets=(-1,))(a, w))

        cfunc = njit(parallel={'sequential': True})(pyfunc)
        got, expected = cfunc(10, 2, True)
        self.assertPreciseEqual(got, expected)
        with self.assertRaises(TypeError):
            cfunc(10, 2, False)


if __name__ == '__main__':
    unittest.main()

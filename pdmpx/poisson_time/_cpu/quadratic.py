import numpy as np
import numba
import functools as ft


@numba.jit(nopython=True)
def ab_poisson_time(u: float, a: float, b: float) -> float:  # TODO: linear_poisson_time
    """
    First arrival time of Poisson process with time-dependent rate `rate = (a + b*t)+`
    """
    if b < 0:
        if a <= 0:
            return np.inf
        else:
            y = -np.log(u)  # reference time
            if -a / b < y:
                return (-a - np.sqrt(2 * b * y + a**2)) / b  # solve
            else:
                return np.inf
    elif b == 0:
        if a <= 0:
            return np.inf
        else:
            return -np.log(u) / a
    else:  # b > 0
        if a < 0:
            y = -np.log(u)
            return (-a + np.sqrt(2 * b * y)) / b
        else:
            y = -np.log(u)
            return (-a + np.sqrt(2 * b * y + a**2)) / b


def almost_zero(x, tol=1e-6):
    return np.abs(x) < tol


@numba.jit(nopython=True)
def solve_cubic_eq(poly):  # , only_real_roots=True):
    """
    Returns the real roots of the real cubic polynomial
        poly[0] + poly[1]x + poly[2]x^2 + poly[3]x^3
    """
    a, b, c, d = poly
    assert d != 0.0
    # x3 + a2x2 + a1x + a0
    a0 = a / d
    a1 = b / d
    a2 = c / d

    q = a1 / 3 - a2**2 / 9
    r = (a1 * a2 - 3 * a0) / 6 - a2**3 / 27

    if r**2 + q**3 > 0:  # Numerical Recipes, only one real solution
        A = (np.abs(r) + np.sqrt(r**2 + q**3)) ** (1 / 3)
        t1 = A - q / A
        if r < 0:
            t1 = -t1
        x1 = t1 - a2 / 3
        return np.array([x1])
        # if only_real_roots:
        #     return np.array([x1])
        # else:
        # x2 = -t1 / 2 - a2 / 3 + (np.sqrt(3) * (A + q / A) / 2) * 1j
        # x3 = np.conj(x2)
        # return np.array([x1, x2, x3])

    else:  # Viete, three real
        if q == 0:
            theta = 0
        else:
            theta = np.arccos(r / (-q) ** (3 / 2))

        phi1 = theta / 3
        phi2 = theta / 3 - 2 * np.pi / 3
        phi3 = theta / 3 + 2 * np.pi / 3

        x1 = 2 * np.sqrt(-q) * np.cos(phi1) - a2 / 3
        x2 = 2 * np.sqrt(-q) * np.cos(phi2) - a2 / 3
        x3 = 2 * np.sqrt(-q) * np.cos(phi3) - a2 / 3

        return np.array([x3, x2, x1])


def solve_quadratic_integral_equation(y, a, b, c):
    """
    Solves t = argmin { y = \\int_0^t (a + bx + cx^2)+ dx }
    """
    assert y >= 0.0

    def almost_zero(v):
        return np.abs(v) < 1e-8

    def P(x):
        return a * x + (b / 2) * x**2 + (c / 3) * x**3

    def solve_P(y, min_root=0.0) -> float:
        """
        Solve argmin x P(x) = y
        """
        roots = solve_cubic_eq([-y, a, b / 2, c / 3])
        return roots[roots >= min_root].min()

    # c ≃ 0: ab_poisson_time
    if almost_zero(c):
        return ab_poisson_time(np.exp(-y), a, b)  # ._.
    elif c > 0:
        # find zeros of p(x):
        # a/c + b/c x + x^2 = 0
        # (x + b/2c)^2 = -a/c + (b/2c)^2
        q = -a / c + (b / (2 * c)) ** 2
        # if q <= 0: always non-negative
        if q <= 0:
            return solve_P(y)
        if q > 0:
            r0 = -b / (2 * c) - np.sqrt(q)
            r1 = r0 + 2 * np.sqrt(q)
            if r0 <= 0.0:
                if r1 < 0.0:
                    return solve_P(y)
                else:
                    return solve_P(y + P(r1), min_root=r1 + 1e-5)
            else:
                if P(r0) > y:
                    return solve_P(y)
                else:
                    return solve_P(y + P(r1) - P(r0), min_root=r1)
    else:  # c < 0
        # find zeros of p(x):
        q = -a / c + (b / (2 * c)) ** 2
        if q <= 0:
            return np.inf  # no solution
        else:
            r0 = -b / (2 * c) - np.sqrt(q)
            r1 = r0 + 2 * np.sqrt(q)
            if r1 < 0.0:
                return np.inf
            if r0 < 0.0:
                if P(r1) > y:
                    return solve_P(y)
                else:
                    return np.inf
            else:  # r0 >= 0
                s0 = P(r1) - P(r0)
                if s0 > y:
                    return solve_P(y + P(r0), min_root=r0)
                else:
                    return np.inf


def solve_quadratic_integral_equation2(y, a, b, c):
    """
    Solves t = argmin { y = \\int_0^t (a + bx + cx^2)+ dx }
    """
    assert y >= 0.0

    def almost_zero(v):
        return np.abs(v) < 1e-8

    def P(x):
        return a * x + (b / 2) * x**2 + (c / 3) * x**3

    def solve_P(y, min_root=0.0) -> float:
        """
        Solve argmin x P(x) = y
        """
        roots = solve_cubic_eq([-y, a, b / 2, c / 3])
        return roots[roots >= min_root].min()

    # c ≃ 0: ab_poisson_time
    if almost_zero(c):
        return ab_poisson_time(np.exp(-y), a, b)  # ._.
    elif c > 0:
        # find zeros of p(x):
        # a/c + b/c x + x^2 = 0
        # (x + b/2c)^2 = -a/c + (b/2c)^2
        q = -a / c + (b / (2 * c)) ** 2
        # if q <= 0: always non-negative
        if q <= 0:
            return solve_P(y)
        if q > 0:
            r0 = -b / (2 * c) - np.sqrt(q)
            r1 = r0 + 2 * np.sqrt(q)
            if r0 <= 0.0:
                if r1 < 0.0:
                    return solve_P(y)
                else:
                    return solve_P(y + P(r1), min_root=r1 + 1e-5)
            else:
                if P(r0) > y:
                    return solve_P(y)
                else:
                    return solve_P(y + P(r1) - P(r0), min_root=r1)
    else:  # c < 0
        # find zeros of p(x):
        q = -a / c + (b / (2 * c)) ** 2
        if q <= 0:
            return np.inf  # no solution
        else:
            r0 = -b / (2 * c) - np.sqrt(q)
            r1 = -b / (2 * c) + np.sqrt(q)
            if r1 < 0.0:
                return np.inf
            if r0 < 0.0:
                if P(r1) > y:
                    return solve_P(y)
                else:
                    return np.inf
            else:  # r0 >= 0
                s0 = P(r1) - P(r0)
                if s0 > y:
                    return solve_P(y + P(r0), min_root=r0)
                else:
                    return np.inf


def test_solve_quadratic_integral_equation2():
    np.random.seed(0)
    xs = np.linspace(0.0, 5.0, 10000)
    for i in range(100000):
        a, b = np.random.normal(loc=2.0, scale=2.0, size=2)
        c = 1.0 if np.random.uniform() < 0.5 else -1.0

        y = np.random.uniform() * 2.0

        ps = np.maximum(a + b * xs + c * xs**2, 0)

        int_ps = np.cumsum(ps) * (xs[1] - xs[0])

        diffs = np.abs(int_ps - y)
        if diffs.min() > 1e-3:
            num_sol = np.inf
        else:
            num_sol = xs[np.argmin(diffs)]

        alg_sol = solve_quadratic_integral_equation(y, a, b, c)
        alg2_sol = solve_quadratic_integral_equation2(y, a, b, c)

        if np.abs(alg_sol - alg2_sol) > 1e-3:
            fails += 1
            print("fail")
            # print(
            #     f"num_sol {num_sol}, alg_sol {alg_sol} \n (a,b,c) = ({a},{b},{c})\n y = {y}"
            # )

    print("fails: ", fails)
    # import matplotlib.pyplot as plt

    # plt.plot(xs, ps)
    # alg_sol
    # (lambda x: a * x + (b / 2) * x**2 + (c / 3) * x**3)(num_sol) - y


test_solve_quadratic_integral_equation2()


def test_solve_quadratic_integral_equation():
    np.random.seed(0)
    xs = np.linspace(0.0, 5.0, 10000)
    for i in range(100000):
        a, b = np.random.normal(loc=2.0, scale=2.0, size=2)
        c = 1.0 if np.random.uniform() < 0.5 else -1.0

        y = np.random.uniform() * 2.0

        ps = np.maximum(a + b * xs + c * xs**2, 0)

        int_ps = np.cumsum(ps) * (xs[1] - xs[0])

        diffs = np.abs(int_ps - y)
        if diffs.min() > 1e-3:
            num_sol = np.inf
        else:
            num_sol = xs[np.argmin(diffs)]

        alg_sol = solve_quadratic_integral_equation(y, a, b, c)
        fails = 0
        if np.abs(num_sol - alg_sol) > 1e-1:
            fails += 1
            # print("fail")
            # print(
            #     f"num_sol {num_sol}, alg_sol {alg_sol} \n (a,b,c) = ({a},{b},{c})\n y = {y}"
            # )

    print("fails: ", fails)
    # import matplotlib.pyplot as plt

    # plt.plot(xs, ps)
    # alg_sol
    # (lambda x: a * x + (b / 2) * x**2 + (c / 3) * x**3)(num_sol) - y


test_solve_quadratic_integral_equation()


# # @numba.jit(nopython=True)
# def solve_cubic_eq(a, b, c, d):
#     # Cubic equation solver for polynomial (degree=3)
#     # http://en.wikipedia.org/wiki/Cubic_function   Lagrange's method
#     # only returns real values
#     a1 = 1 / d
#     E1 = -c * a1
#     E2 = b * a1
#     E3 = -a * a1
#     s0 = E1
#     E12 = E1 * E1
#     A = 2 * E1 * E12 - 9 * E1 * E2 + 27 * E3  # = s1^3 + s2^3
#     B = E12 - 3 * E2  # = s1 s2
#     # quadratic equation: z^2 - Az + B^3=0  where roots are equal to s1^3 and s2^3
#     discr = np.sqrt(A * A - 4 * B * B * B + 0j)

#     if (
#         np.real(np.conj(A) * discr) >= 0
#     ):  # scalar product to decide the sign yielding bigger magnitude
#         s1 = np.exp(np.log(0.5 * (A + discr)) / 3)
#     else:
#         s1 = np.exp(np.log(0.5 * (A - discr)) / 3)
#     if s1 == 0:
#         s2 = s1
#     else:
#         s2 = B / s1

#     zeta1 = np.complex128(-0.5 + np.sqrt(3.0) * 0.5j)
#     zeta2 = np.conj(zeta1)

#     r0 = (s0 + s1 + s2) / 3
#     r1 = (s0 + s1 * zeta2 + s2 * zeta1) / 3
#     r2 = (s0 + s1 * zeta1 + s2 * zeta2) / 3

#     if (
#         -27 * d**2 * a**2
#         + 18 * d * c * b * a
#         - 4 * d * b**3
#         - 4 * c**3
#         + c**2 * b**2
#         > 0
#     ):
#         return np.real([r0, r1, r2])
#     else:
#         return np.real([r0])


# a, b, c = 1.0, 2.0, 3.0

# fact = np.polynomial.Polynomial((-2, 1))
# p = fact**2 - 1
# a, b, c = p.coef


# def P(x):
#     return a * x + (b / 2) * x**2 + (c / 3) * x**3


# def solve_P(y, a, b, c):
#     """
#     Solve int_0^x p(t) dt = y, t >= 0
#     """
#     roots = solve_cubic_eq(-y - P(0), a, b / 2, c / 3)
#     real_roots = roots[np.isreal(roots)]
#     positive_roots = roots[roots > 0]
#     return np.min(positive_roots)

#     # shift = np.polynomial.Polynomial((-x0, 1.0))
#     # roots = (P(shift) - y).roots()
#     # real_roots = roots[np.isreal(roots)]
#     # return real_roots


# solve_P(1.0, a, b, c)
# y = 1.0
# a, b, c
# roots = solve_cubic_eq(-y - P(0), a, b / 2, c / 3)
# real_roots = np.real(roots[np.imag(roots) < 1e-15])
# positive_roots = real_roots[real_roots > 0]
# np.min(positive_roots)
# P(np.min(real_roots))

# np.sqrt()

# solve_cubic_eq = numba.jit(solve_cubic_eq, nopython=True)
# roots = solve_cubic_eq(0.0, 1.0, 1.0, 1.0)
# roots
# p = [0.0, 1.0, 12.0, 2.0]
# solve_cubic_eq(*p)
# np.allclose(solve_cubic_eq(p), np.polynomial.Polynomial(p).roots())

# for i in range(100):
#     p = np.random.normal(loc=2.0, scale=2.0, size=4)
#     p[3] = 1.0 if np.random.uniform() < 0.5 else -1.0
#     np_pol = np.polynomial.Polynomial(p)
#     print("  poly: ", np_pol)
#     roots = np_pol.roots()
#     roots = roots[np.isreal(roots)]
#     allclose = np.allclose(roots, solve_cubic_eq(p))
#     print(f"{i} - all roots close: {allclose}")
#     if not allclose:
#         break

# import matplotlib.pyplot as plt
# P(r0) - P(0) + P(s) - P(r1) = 0
# a + bs + cs2 + ds3 + (P(r0) - P(0) - P(r1)) = 0
# end

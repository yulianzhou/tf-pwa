"""
This module provides functions to describe the lineshapes of the intermediate particles, namely generalized
Breit-Wigner function. Users can also define new lineshape using the function wrapper **regist_lineshape()**.
"""

import fractions
import functools
import math
import warnings

import sympy as sym

from .tensorflow_wrapper import tf

breit_wigner_dict = {}


def to_complex(i):
    if i.dtype in [tf.float32, tf.float64]:
        return tf.complex(i, tf.zeros_like(i))
    return i


def regist_lineshape(name=None):
    """
    It will be used as a wrapper to define various Breit-Wigner functions

    :param name: String name of the BW function
    :return: A function used in a wrapper
    """

    def fopt(f):
        name_t = name
        if name_t is None:
            name_t = f.__name__
        if name_t in breit_wigner_dict:
            warnings.warn(
                "override breit wigner function :", name
            )  # warning to users
        breit_wigner_dict[name_t] = f  # function
        return f

    return fopt


@regist_lineshape("one")
def one(*args):
    """
    A uniform function
    """
    return tf.complex(
        1.0, 0.0
    )  # breit_wigner_dict["one"]==tf.complex(1.0,0.0)


@regist_lineshape("BW")
def BW(m, m0, g0, *args):
    """
    Breit-Wigner function

    .. math::
        BW(m) = \\frac{1}{m_0^2 - m^2 -  i m_0 \\Gamma_0 }

    """
    m0 = tf.cast(m0, m.dtype)
    gamma = tf.cast(g0, m.dtype)
    x = m0 * m0 - m * m
    y = m0 * gamma
    s = x * x + y * y
    ret = tf.complex(x / s, y / s)
    return ret


@regist_lineshape("default")  # 两个名字
@regist_lineshape("BWR")  # BW with running width
def BWR(m, m0, g0, q, q0, L, d):
    """
    Relativistic Breit-Wigner function (with running width). It's also set as the default lineshape.

    .. math::
        BW(m) = \\frac{1}{m_0^2 - m^2 -  i m_0 \\Gamma(m)}

    """
    gamma = Gamma(m, g0, q, q0, L, m0, d)
    num = 1.0
    m0 = tf.cast(m0, m.dtype)
    x = m0 * m0 - m * m
    y = m0 * gamma
    s = x * x + y * y
    ret = tf.complex(x / s, y / s)
    return ret


@regist_lineshape("a1_1260")
def a1_1260(m, m0):
    m2 = m * m
    m4 = m2 * m2
    m6 = m4 * m2
    gamma = tf.where(m < 0.78, 0, tf.where(m > 1.4, -2.036 + 2.781 * m - 0.6528 * m2, -111.37 + 677.225 * m - 1684.15 * m2 + 2191.93 * m2 * m - 1575.88 * m4 + 594.6 * m4 *m - 92.16 * m6))
    m0 = tf.cast(m0, m.dtype)
    x = m0 * m0 - m * m
    y = m0 * gamma
    s = x * x + y * y
    ret = tf.complex(x / s, y / s)
    return ret


# added by xiexh for GS model rho
def twoBodyCMmom(m_0, m_1, m_2):
    """relative momentum for 0 -> 1 + 2"""
    M12S = m_1 + m_2
    M12D = m_1 - m_2
    if hasattr(M12S, "dtype"):
        m_0 = tf.convert_to_tensor(m_0, dtype=M12S.dtype)
    #    m_eff = tf.where(m_0 > M12S, m_0, M12S)
    #    p = (m_eff - M12S) * (m_eff + M12S) * (m_eff - M12D) * (m_eff + M12D)
    # if p is negative, which results from bad data, the return value is 0.0
    # print("p", tf.where(p==0), m_0, m_1, m_2)
    p = (m_0 - M12S) * (m_0 + M12S) * (m_0 - M12D) * (m_0 + M12D)
    zeros = tf.zeros_like(m_0)
    ret = tf.where(p > 0, tf.sqrt(p) / (2 * m_0), zeros)
    return ret


def hFun(s, daug2Mass, daug3Mass):
    _pi = 3.14159265359
    _pi = tf.cast(_pi, s.dtype)

    sm = daug2Mass + daug3Mass
    sqrt_s = tf.sqrt(s)
    k_s = twoBodyCMmom(tf.sqrt(s), daug2Mass, daug3Mass)

    ret = (
        (2.0 / _pi)
        * (k_s / sqrt_s)
        * tf.math.log((sqrt_s + 2.0 * k_s) / (sm), name="log")
    )
    ret = tf.cast(ret, s.dtype)
    return ret


def dh_dsFun(s, daug2Mass, daug3Mass):
    _pi = 3.14159265359
    _pi = tf.cast(_pi, s.dtype)
    k_s = twoBodyCMmom(tf.sqrt(s), daug2Mass, daug3Mass)

    ret = hFun(s, daug2Mass, daug3Mass) * (
        1.0 / (8.0 * tf.pow(k_s, 2)) - 1.0 / (2.0 * s)
    ) + 1.0 / (2.0 * _pi * s)
    ret = tf.cast(ret, s.dtype)
    return ret


def dFun(s, daug2Mass, daug3Mass):
    _pi = 3.14159265359
    _pi = tf.cast(_pi, s.dtype)
    sm = daug2Mass + daug3Mass
    sm24 = sm * sm / 4.0
    m = tf.sqrt(s)
    k_m2 = twoBodyCMmom(tf.sqrt(s), daug2Mass, daug3Mass)

    ret = (
        3.0
        / _pi
        * sm24
        / tf.pow(k_m2, 2)
        * tf.math.log((m + 2 * k_m2) / sm, name="log")
        + m / (2 * _pi * k_m2)
        - sm24 * m / (_pi * tf.pow(k_m2, 3))
    )
    ret = tf.cast(ret, s.dtype)
    return ret


def fsFun(s, m2, gam, daug2Mass, daug3Mass):
    k_s = twoBodyCMmom(tf.sqrt(s), daug2Mass, daug3Mass)
    k_Am2 = twoBodyCMmom(tf.sqrt(m2), daug2Mass, daug3Mass)

    f = gam * m2 / tf.pow(k_Am2, 3)
    f *= tf.pow(k_s, 2) * (
        hFun(s, daug2Mass, daug3Mass) - hFun(m2, daug2Mass, daug3Mass)
    ) + (m2 - s) * tf.pow(k_Am2, 2) * dh_dsFun(m2, daug2Mass, daug3Mass)

    f = tf.cast(f, s.dtype)
    return f


# Gounaris-Sakurai model for rho
def GS(m, m0, g0, q, q0, L, d, c_daug2Mass=0.13957039, c_daug3Mass=0.1349768):
    gamma = Gamma(m, g0, q, q0, L, m0, d)
    c_daug2Mass = tf.cast(c_daug2Mass, m.dtype)
    c_daug3Mass = tf.cast(c_daug3Mass, m.dtype)

    D = 1.0 + dFun(m0 * m0, c_daug2Mass, c_daug3Mass) * g0 / m0
    E = m0 * m0 - m * m + fsFun(m * m, m0 * m0, g0, c_daug2Mass, c_daug3Mass)
    F = m0 * gamma

    D /= E * E + F * F
    ret = tf.complex(D * E, D * F)

    return ret


# added by xiexh end


def BWR2(m, m0, g0, q2, q02, L, d):
    """
    Relativistic Breit-Wigner function (with running width). Allow complex :math:`\\Gamma`.

    .. math::
        BW(m) = \\frac{1}{m_0^2 - m^2 -  i m_0 \\Gamma(m)}

    """
    gamma = Gamma2(m, g0, q2, q02, L, m0, d)
    num = 1.0
    m0 = tf.cast(m0, m.dtype)
    x = tf.cast(m0 * m0 - m * m, gamma.dtype)
    y = tf.cast(m0, gamma.dtype) * gamma
    d = x - 1j * y
    bw_x = tf.math.real(d)
    bw_y = tf.math.imag(d)
    bw_r2 = bw_x * bw_x + bw_y * bw_y
    ret = tf.complex(bw_x / bw_r2, bw_y / bw_r2)
    return ret


def BWR_normal(m, m0, g0, q2, q02, L, d):
    """
    Relativistic Breit-Wigner function (with running width) with a normal factor.

    .. math::
        BW(m) = \\frac{\\sqrt{m_0 \\Gamma(m)}}{m_0^2 - m^2 -  i m_0 \\Gamma(m)}

    """
    gamma = Gamma2(m, g0, q2, q02, L, m0, d)
    num = 1.0
    m0 = tf.cast(m0, m.dtype)
    x = tf.cast(m0 * m0 - m * m, gamma.dtype)
    y = tf.cast(m0, gamma.dtype) * gamma
    ret = tf.sqrt(tf.cast(m0, gamma.dtype) * gamma) / (x - 1j * y)
    return ret


def Gamma(m, gamma0, q, q0, L, m0, d):
    """
    Running width in the RBW

    .. math::
        \\Gamma(m) = \\Gamma_0 \\left(\\frac{q}{q_0}\\right)^{2L+1}\\frac{m_0}{m} B_{L}'^2(q,q_0,d)

    """
    q0 = tf.cast(q0, q.dtype)
    _epsilon = 1e-15
    qq0 = tf.where(q0 > _epsilon, (q / q0) ** (2 * L + 1), 1.0)
    mm0 = tf.cast(m0, m.dtype) / m
    bp = Bprime(L, q, q0, d) ** 2
    gammaM = gamma0 * qq0 * mm0 * tf.cast(bp, qq0.dtype)
    return gammaM


def Gamma2(m, gamma0, q2, q02, L, m0, d):
    """
    Running width in the RBW

    .. math::
        \\Gamma(m) = \\Gamma_0 \\left(\\frac{q}{q_0}\\right)^{2L+1}\\frac{m_0}{m} B_{L}'^2(q,q_0,d)

    """
    q02 = tf.cast(q02, q2.dtype)
    _epsilon = 1e-15
    qq0 = q2 / q02
    qq0 = to_complex(qq0**L) * tf.sqrt(to_complex(qq0))
    mm0 = tf.cast(m0, m.dtype) / m
    z0 = q02 * d**2
    z = q2 * d**2
    bp = Bprime_polynomial(L, z0) / Bprime_polynomial(L, z)
    gammaM = qq0 * tf.cast(gamma0 * bp * mm0, qq0.dtype)
    return gammaM


def Bprime_q2(L, q2, q02, d):
    """
    Blatt-Weisskopf barrier factors.
    """
    q02 = tf.cast(q02, q2.dtype)
    _epsilon = 1e-15
    z0 = q02 * d**2
    z = q2 * d**2
    bp = Bprime_polynomial(L, z0) / Bprime_polynomial(L, z)
    return tf.sqrt(tf.where(bp > 0, bp, 1.0))


def Bprime_num(L, q, d):
    """
    The numerator (as well as the denominator) inside the square root in the barrier factor

    """
    z = (q * d) ** 2
    bp = Bprime_polynomial(L, z)
    return tf.sqrt(bp)


def Bprime(L, q, q0, d):
    """
    Blatt-Weisskopf barrier factors. E.g. the first three orders

    ===========  ===================================================
     :math:`L`                :math:`B_L'(q,q_0,d)`
    ===========  ===================================================
     0                                 1
     1                 :math:`\\sqrt{\\frac{(q_0d)^2+1}{(qd)^2+1}}`
     2            :math:`\\sqrt{\\frac{(q_0d)^4+3*(q_0d)^2+9}{(qd)^4+3*(qd)^2+9}}`
    ===========  ===================================================

    :math:`d` is 3.0 by default.
    """
    num = Bprime_num(L, q0, d)
    denom = Bprime_num(L, q, d)
    return tf.cast(num, denom.dtype) / denom


def barrier_factor(l, q, q0, d=3.0, axis=0):  # cache q^l * B_l 只用于H里
    """
    Barrier factor multiplied with :math:`q^L`, which is used as a combination in the amplitude expressions. The values
    are cached for :math:`L` ranging from 0 to **l**.
    """
    ret = []
    for i in l:
        tmp = q**i * tf.cast(Bprime(i, q, q0, d), q.dtype)
        ret.append(tmp)
    return tf.stack(ret)


def barrier_factor2(l, q, q0, d=3.0, axis=-1):  # cache q^l * B_l 只用于H里
    """
    ???
    """
    ret = []
    for i in l:
        tmp = q**i * tf.cast(Bprime(i, q, q0, d), q.dtype)
        ret.append(tf.reshape(tmp, (-1, 1)))
    return tf.concat(ret, axis=axis)


def Bprime_polynomial(l, z):
    """
    It stores the Blatt-Weisskopf polynomial up to the fifth order (:math:`L=5`)

    :param l: The order
    :param z: The variable in the polynomial
    :return: The calculated value
    """
    coeff = {
        0: [1.0],
        1: [1.0, 1.0],
        2: [1.0, 3.0, 9.0],
        3: [1.0, 6.0, 45.0, 225.0],
        4: [1.0, 10.0, 135.0, 1575.0, 11025.0],
        5: [1.0, 15.0, 315.0, 6300.0, 99225.0, 893025.0],
    }
    l = int(l + 0.01)
    if l not in coeff:
        coeff[l] = [float(i) for i in get_bprime_coeff(l)]
        # raise NotImplementedError
    z = tf.convert_to_tensor(z)
    cof = [tf.convert_to_tensor(i, z.dtype) for i in coeff[l]]
    ret = tf.math.polyval(cof, z)
    return ret


def reverse_bessel_polynomials(n, x):
    """Reverse Bessel polynomials.

    .. math::
        \\theta_{n}(x) = \\sum_{k=0}^{n} \\frac{(n+k)!}{(n-k)!k!} \\frac{x^{n-k}}{2^k}

    """
    ret = 0
    for k in range(n + 1):
        c = fractions.Fraction(
            math.factorial(n + k),
            math.factorial(n - k) * math.factorial(k) * 2**k,
        )
        ret += c * x ** (n - k)
    return ret


@functools.lru_cache()
def get_bprime_coeff(l):
    """The coefficients of polynomial in Bprime function.

    .. math::
        |\\theta_{l}(jw)|^2 = \\sum_{i=0}^{l} c_i w^{2 i}

    """
    x = sym.Symbol("x")
    theta = reverse_bessel_polynomials(l, x)
    w = sym.Symbol("w", real=True)
    Hjw = theta.subs({"x": sym.I * w})
    Hjw2 = sym.Poly(Hjw * Hjw.conjugate(), w)
    coeffs = Hjw2.as_dict()
    ret = [coeffs.get((2 * l - 2 * i,), 0) for i in range(l + 1)]
    return ret


def complex_q(s, m1, m2):
    q2 = (s - (m1 + m2) ** 2) * (s - (m1 - m2) ** 2) / (4 * s)
    q = tf.sqrt(tf.complex(q2, tf.zeros_like(q2)))
    return q


def chew_mandelstam(m, m1, m2):
    """
    Chew-Mandelstam function in `PDG 2024 Eq 50.44 <https://pdg.lbl.gov/2024/reviews/rpp2024-rev-resonances.pdf>`_ multiply :math:`16\\pi` factor.

    .. math::
        \\Sigma(m) = \\frac{1}{\\pi}\\left[
            \\frac{2q}{m} \\ln \\left(\\frac{ m_1^2 + m_2^2 - m^2 + 2mq }{ 2 m_1 m_2}\\right)
            - (m_1^2 - m_2^2) (\\frac{1}{m^2} - \\frac{1}{(m_1+m_2)^2}) \\ln \\frac{m_1}{m_2}
        \\right]

    for :math:`m>(m_1+m_2)`

    .. math::
         Im\\Sigma(m) = \\frac{1}{i}\\frac{1}{\\pi} \\frac{2q}{m} \\ln (-1) = \\frac{2q}{m}

    .. math::
         Re\\Sigma(m) = \\frac{1}{\\pi}\\left[
            \\frac{2q}{m} \\ln \\left( \\frac{ m^2 - m_1^2 - m_2^2 - 2mq }{ 2 m_1 m_2}\\right)
            - (m_1^2 - m_2^2) (\\frac{1}{m^2} - \\frac{1}{(m_1+m_2)^2}) \\ln \\frac{m_1}{m_2}
        \\right]


    """
    s = m * m
    C = lambda x: tf.complex(x, tf.zeros_like(x))
    m1 = tf.cast(m1, s.dtype)
    m2 = tf.cast(m2, s.dtype)
    q = complex_q(s, m1, m2)
    s1 = m1 * m1
    s2 = m2 * m2
    a = (
        C(2 / m)
        * q
        * tf.math.log((C(s1 + s2 - s) + C(2 * m) * q) / C(2 * m1 * m2))
    )
    b = (s1 - s2) * (1 / s - 1 / (m1 + m2) ** 2) * tf.math.log(m1 / m2)
    ret = a - C(b)
    return ret / math.pi


def chew_mandelstam_l(m, m1, m2, l):
    """

    TODO. Function from `J.Math.Phys. 25 (1984) 3540 <https://inspirehep.net/literature/182258>`_ , with some modifies to be same as `chew_mandelstam` function.

    compare with `chew_mandelstam` function.


    :math:`m_1=0.4, m_2=0.1`

    .. plot::

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from tf_pwa.breit_wigner import chew_mandelstam, chew_mandelstam_l
        >>> m = np.linspace(0.3, 2.0)
        >>> y1 = chew_mandelstam_l(m, 0.4, 0.1, l=0)
        >>> y2 = chew_mandelstam(m, 0.4, 0.1)
        >>> _ = plt.plot(m, np.real(y1), label="Re $C_0(m)$")
        >>> _ = plt.plot(m, np.imag(y1), label="Im $C_0(m)$")
        >>> _ = plt.plot(m, np.real(y1)-np.mean(np.real(y1-y2)), label="Re $C_0(m) - \\delta$")
        >>> _ = plt.plot(m, np.real(y2), ls="--", label="Re $\\Sigma(m)$")
        >>> _ = plt.plot(m, np.imag(y2), ls="--", label="Im $\\Sigma(m)$")
        >>> _ = plt.legend()

    :math:`m_1=m_2=0.4`

    .. plot::

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from tf_pwa.breit_wigner import chew_mandelstam, chew_mandelstam_l
        >>> m = np.linspace(0.3, 2.0)
        >>> y1 = chew_mandelstam_l(m, 0.4, 0.4, l=0)
        >>> y2 = chew_mandelstam(m, 0.4, 0.4)
        >>> _ = plt.plot(m, np.real(y1), label="Re $C_0(m)$")
        >>> _ = plt.plot(m, np.imag(y1), label="Im $C_0(m)$")
        >>> _ = plt.plot(m, np.real(y1)-np.mean(np.real(y1-y2)), label="Re $C_0(m)- \\delta$")
        >>> _ = plt.plot(m, np.real(y2), ls="--", label="Re $\\Sigma(m)$")
        >>> _ = plt.plot(m, np.imag(y2), ls="--", label="Im $\\Sigma(m)$")
        >>> _ = plt.legend()

    .. plot::

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from tf_pwa.breit_wigner import chew_mandelstam, chew_mandelstam_l
        >>> m = np.linspace(0.3, 2.0)
        >>> y1 = chew_mandelstam_l(m, 0.4, 0.1, l=0)
        >>> y2 = chew_mandelstam_l(m, 0.4, 0.1, l=1)
        >>> y3 = chew_mandelstam_l(m, 0.4, 0.1, l=2)
        >>> _ = plt.plot(m, np.real(y1), label="Re $C_0(m)$")
        >>> _ = plt.plot(m, np.imag(y1), label="Im $C_0(m)$")
        >>> _ = plt.plot(m, np.real(y2), ls="--", label=" Re $C_1(m)$")
        >>> _ = plt.plot(m, np.imag(y2), ls="--", label="Im $C_1(m)$")
        >>> _ = plt.plot(m, np.real(y3), ls=":", label="Re $C_2(m)$")
        >>> _ = plt.plot(m, np.imag(y3), ls=":", label="Im $C_2(m)$")
        >>> _ = plt.legend()

    """
    s = m * m
    C = lambda x: tf.complex(x, tf.zeros_like(x))
    same_mass = abs(m1 - m2) < 1e-6
    m1 = tf.cast(m1, s.dtype)
    m2 = tf.cast(m2, s.dtype)
    s1 = m1 * m1
    s2 = m2 * m2
    a = (m1 + m2) ** 2
    b = (m1 - m2) ** 2
    lam = (s - a) * (s - b) / s / s

    lam_n = [lam**i for i in range(l + 1)]

    ret_1_1 = 0
    for r in range(l + 1):
        ret_1_1 = ret_1_1 + lam_n[l - r] / (2 * r + 1)
    ret_1 = (
        -(
            C(lam) ** (l + 0.5)
            * (
                tf.math.log(
                    -(
                        (
                            (tf.sqrt(C(s - a)) + tf.sqrt(C(s - b)))
                            / C(2 * tf.sqrt(m1 * m2))
                        )
                        ** (2)
                    )
                )
            )
            + C(ret_1_1)
        )
        / math.pi
    )
    # print(ret_1)
    # return ret_1
    if same_mass:  # b = 0
        return ret_1

    mu = (a - b) ** 2 / (16 * a * b)
    nu = (a + b) / (2 * tf.sqrt(a * b))
    omega = nu - tf.sqrt(a * b) / s
    mu_n = [mu**i for i in range(l + 1)]

    f1, f2 = [1], [1]
    for i in range(1, l + 1):
        f1.append(f1[-1] * i)
        f2.append(f2[-1] * (2 * i - 1))  # TODO: (2n-1)!! for n=0?
    Omega_n = [(-2) ** n * f2[n] / f1[n] for n in range(l + 1)]

    ret_2_1 = omega * tf.math.log(m1 / m2) / math.pi
    ret_2_2 = Omega_n[0] * lam_n[l]
    for i in range(1, l + 1):
        ret_2_2 = ret_2_2 + Omega_n[i] * lam_n[l - i] * mu_n[i]
    ret_2 = ret_2_1 * ret_2_2
    ret_3_1 = omega * nu / 2 / math.pi
    ret_3_2 = 0
    for p in range(1, l + 1):
        ret_3_2_1 = 0
        for q in range(l - p + 1):
            ret_3_2_1 = ret_3_2_1 + Omega_n[q] * lam_n[l - p - q] * mu_n[q]
        ret_3_2_2 = 0
        for r in range(p - 1):
            ret_3_2_2 = ret_3_2_2 + Omega_n[r] * mu_n[r]
        ret_3_2 = ret_3_2 + ret_3_2_1 * ret_3_2_2 / p
    ret_3 = ret_3_1 * ret_3_2
    return ret_1 + C(ret_2 + ret_3)

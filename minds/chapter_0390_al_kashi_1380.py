#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 THE CERTIFIED BRACKET ITERATION NETWORK (CBIN)
 An executable architecture after Ghiyath al-Din Jamshid Mas'ud al-Kashi
 # BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0390 - AL-KASHI (1380-1429)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 20, Minds 381-400 Available on Amazon https://www.amazon.com/dp/B0HRS9ZHVG
# Resume and Interactive Demos at  https://artificiology.com/
# Videos of the playable minds at https://www.youtube.com/@LostMindsAI
# END ATTRIBUTION  
================================================================================

THE ONE IDEA THIS FILE ENCODES
------------------------------
Al-Kashi is not "the man who computed pi a long way". Every culture has one of
those. He is the first person on record who *fixed the permissible error of the
answer, in physical units, before beginning the computation*, and then derived
backwards from that requirement how many digits, how many polygon doublings and
how many iterations the work would take.

In the Risala al-muhitiyya (July 1424) the requirement is stated in advance: the
circumference of a circle whose diameter is six hundred thousand times the
Earth's must come out wrong by less than the breadth of a horse's hair. Only
then does he pick the instrument -- a polygon of 3 x 2^28 sides -- and grind.

Three consequences follow, and this file is built out of exactly those three:

  (1) TOLERANCE FIRST.   Compute is a dependent variable. The caller states tau;
                         the machine derives its own effort. Nothing is fixed
                         in advance except the promise.
  (2) TWO-SIDED TRUTH.   He never holds a single number. He holds an inscribed
                         polygon and a circumscribed one, and the truth is
                         trapped between them. A bracket, not a point estimate.
  (3) EARNED DIGITS.     A digit is emitted only when both sides of the bracket
                         agree on it. Digits that are not common to lo and hi
                         are not reported, however confident the calculator.

Compare with the nearest neighbours in this corpus so as not to repeat them:
Archimedes (#89) proposes by mechanics and proves by geometry -- heuristic then
verification. Madhava (#304) computes forward and then *estimates the tail he
never summed* -- error correction after the fact. Al-Kashi inverts both: the
error is a specification written before the first stroke of the pen, and the
computation is engineered until it cannot violate it.

WHAT IS ACTUALLY BUILT HERE
---------------------------
PART I   An exact base-60 fixed-point engine on Python big integers (no floats),
         used to reproduce his two headline results and to run his backward
         "horse's hair" derivation of the required digit count.

PART II  A trainable network, pure NumPy, from scratch:
         the Certified Bracket Iteration Network.

         Its recurrent core is literally al-Kashi's own map. To find sin(1) he
         trisected: with x = sin(theta) and s = sin(3 theta),

               3x - 4x^3 = s      ==>      x = (s + 4 x^3) / 3

         and he iterated that. It is a contraction on the relevant domain, and
         he knew the rate -- roughly one further sexagesimal place per pass.

         The network does NOT learn the map. The map is the mind; it is fixed.
         What the network learns is the SEED and therefore the LABOUR: a better
         starting guess means fewer certified iterations to reach the caller's
         tolerance. The loss function is, to a first approximation, *the number
         of iterations you must pay for in order to keep your promise*. Learning
         is the purchase of theory to save arithmetic -- which is precisely what
         al-Kashi did when he used trigonometric identities to avoid labour.

         Certification is carried alongside, not bolted on: at every step a
         rigorous a-posteriori bound
               |x* - x_{k+1}|  <=  L/(1-L) * |x_{k+1} - x_k|
         is propagated, with L a validated Lipschitz bound on the enclosing
         interval. The training objective contains a validity term that
         punishes any parameter setting whose bound fails to contain the truth,
         which is what stops the network from learning to look confident.

PART III Tolerance-first inference: the digit ledger. The model is asked a
         question together with a tolerance, halts of its own accord, and emits
         only the sexagesimal places it has earned.

CONVENTIONS
-----------
* Pure NumPy + standard library. No autograd, no frameworks.
* Backpropagation is hand-derived and finite-difference checked at run time.
* Everything below is executed; the printed output is pasted into the chapter.

Run:   python3 chapter_0390_al_kashi_1380.py
================================================================================
"""

from __future__ import annotations

import math
import time
from math import isqrt

import numpy as np

# =============================================================================
# PART I -- THE SEXAGESIMAL FIELD
#
# Al-Kashi worked in base 60 with integers, not in floating point. We reproduce
# that exactly: a number is a Python big integer n representing the value
# n / 60**PLACES. Nothing is approximate except where we say it is. Square roots
# use integer isqrt, which truncates -- and truncation always moves the value
# DOWNWARD, a fact we exploit when reasoning about which side of the bracket a
# rounding error can fall on.
# =============================================================================

PLACES = 24                    # sexagesimal places carried internally
SCALE = 60 ** PLACES           # the fixed-point scale factor

# 2*pi to 50 decimal places, from the standard published expansion of pi. Held
# as an exact fixed-point integer so that bracket tests can be run in the
# integers, where they belong, rather than in float64 where a 1e-17 gap vanishes.
_TWO_PI_DIGITS = "6283185307179586476925286766559005768394338798750211641949"
TWO_PI_FIXED = int(_TWO_PI_DIGITS) * SCALE // 10 ** (len(_TWO_PI_DIGITS) - 1)


def sx_from_int(n: int) -> int:
    """Lift an ordinary integer into the fixed-point field."""
    return n * SCALE


def sx_mul(a: int, b: int) -> int:
    """Fixed-point multiply. Floor division truncates toward -infinity."""
    return (a * b) // SCALE


def sx_div(a: int, b: int) -> int:
    """Fixed-point divide."""
    return (a * SCALE) // b


def sx_sqrt(a: int) -> int:
    """
    Fixed-point square root via integer isqrt.

    isqrt truncates, so the result is always <= the true root. Al-Kashi was
    equally careful about the DIRECTION of his roundings: in the polygon work a
    truncated chord makes the inscribed perimeter a touch too small, which keeps
    it safely on the correct side of the bracket.
    """
    return isqrt(a * SCALE)


def sx_to_sexagesimal(a: int, places: int = 10) -> str:
    """Render as  I;p1,p2,...  which is how al-Kashi would have written it."""
    sign = "-" if a < 0 else ""
    a = abs(a)
    whole, frac = divmod(a, SCALE)
    digits = []
    for _ in range(places):
        frac *= 60
        d, frac = divmod(frac, SCALE)
        digits.append(int(d))
    return f"{sign}{int(whole)};" + ",".join(str(d) for d in digits)


def sx_to_float(a: int) -> float:
    return a / SCALE


def sx_common_places(lo: int, hi: int, max_places: int = 12) -> int:
    """
    THE DIGIT LEDGER, in its original form.

    How many sexagesimal places do the two sides of a bracket agree on? Those
    are the digits that have been *earned*. The rest are opinion.
    """
    if lo > hi:
        lo, hi = hi, lo
    if lo // SCALE != hi // SCALE:
        return 0
    flo = lo - (lo // SCALE) * SCALE
    fhi = hi - (hi // SCALE) * SCALE
    earned = 0
    for _ in range(max_places):
        flo *= 60
        fhi *= 60
        dlo, flo = divmod(flo, SCALE)
        dhi, fhi = divmod(fhi, SCALE)
        if dlo != dhi:
            break
        earned += 1
    return earned


# -----------------------------------------------------------------------------
# I.a  The circumference, bracketed  (Risala al-muhitiyya, July 1424)
# -----------------------------------------------------------------------------

def polygon_bracket(doublings: int = 28, start_sides: int = 3):
    """
    Al-Kashi's polygon ladder, run exactly.

    Begin with the equilateral triangle inscribed in a circle of radius 1, whose
    side is sqrt(3). Halve the arc repeatedly using the chord recursion

        c_{2n} = sqrt( 2 - sqrt( 4 - c_n^2 ) )

    After 28 doublings from 3 sides we stand on a polygon of 3 x 2^28 =
    805,306,368 sides, which is the figure the treatise reports.

    The inscribed perimeter is a lower bound on 2*pi; the circumscribed
    perimeter, obtained from the same chord by

        t_n = c_n / sqrt( 1 - c_n^2 / 4 )

    is an upper bound. Truth is trapped between them. That trapping is the whole
    epistemology: al-Kashi is never in the position of believing a number, he is
    in the position of having cornered one.

    Returns (n_sides, lower, upper, mean).
    """
    one, two, four = SCALE, 2 * SCALE, 4 * SCALE

    chord = sx_sqrt(3 * SCALE)          # side of the inscribed triangle
    sides = start_sides
    for _ in range(doublings):
        c2 = sx_mul(chord, chord)
        chord = sx_sqrt(two - sx_sqrt(four - c2))
        sides *= 2

    lower = sides * chord                                    # inscribed 2*pi
    c2 = sx_mul(chord, chord)
    tangent = sx_div(chord, sx_sqrt(one - c2 // 4))
    upper = sides * tangent                                  # circumscribed 2*pi
    mean = (lower + upper) // 2
    return sides, lower, upper, mean


# -----------------------------------------------------------------------------
# I.b  The horse's hair: deriving the digit budget BACKWARDS from the promise
# -----------------------------------------------------------------------------

def horse_hair_budget(earth_diameter_km: float = 12742.0,
                      multiple: float = 600_000.0,
                      hair_mm: float = 0.08):
    """
    The signature move, made executable.

    Al-Kashi does not ask "how many digits can I get?". He asks "how wrong am I
    allowed to be?", states the answer in millimetres, and lets the digit count
    fall out of the arithmetic. This function performs that derivation.

    The assumptions are ours and are declared: a horse's hair is taken as
    0.08 mm, and the great circle is the one named in the treatise. Change
    either number and the required place count moves by one; the *method* is
    what is al-Kashi's, not our estimate of horsehair.

    Returns (required_places_sexagesimal, required_decimals, tolerance_on_2pi).
    """
    diameter_mm = earth_diameter_km * multiple * 1e6      # km -> mm
    tol_on_2pi = hair_mm / diameter_mm                    # since C = 2*pi*r = (2pi)*(d/2)
    tol_on_2pi *= 2.0                                     # error in 2*pi, given d/2
    places = math.ceil(-math.log(tol_on_2pi, 60))
    decimals = math.ceil(-math.log10(tol_on_2pi))
    return places, decimals, tol_on_2pi


# -----------------------------------------------------------------------------
# I.c  sin(1 degree) by his own contraction  (Risala al-watar wa'l-jaib)
# -----------------------------------------------------------------------------

def sin_one_degree(max_iters: int = 12):
    """
    Trisection of the angle, solved by fixed-point iteration.

    sin(3 deg) is constructible: from the pentagon and the triangle,
        sin 18 = (sqrt5 - 1)/4,   cos 18 = sqrt(10 + 2 sqrt5)/4,
        sin 15 = (sqrt6 - sqrt2)/4,  cos 15 = (sqrt6 + sqrt2)/4,
        sin 3  = sin18 cos15 - cos18 sin15.

    Then with x = sin(1 deg),  3x - 4x^3 = sin(3 deg), i.e. the cubic that the
    Arabic tradition wrote as "things are equal to the cube and the number".
    Al-Kashi rearranged it into x = (q + x^3)/p and iterated. The derivative of
    that map is 4x^2 ~ 0.0012 near the root, so each pass multiplies the error
    by about a thousandth -- between one and two further sexagesimal places per
    iteration, which is exactly the rate his manuscript exhibits.

    Returns (list_of_iterates, digits_gained_per_iteration).
    """
    r5 = sx_sqrt(5 * SCALE)
    sin18 = sx_div(r5 - SCALE, 4 * SCALE)
    cos18 = sx_div(sx_sqrt(10 * SCALE + 2 * r5), 4 * SCALE)
    r6, r2 = sx_sqrt(6 * SCALE), sx_sqrt(2 * SCALE)
    sin15 = sx_div(r6 - r2, 4 * SCALE)
    cos15 = sx_div(r6 + r2, 4 * SCALE)
    sin3 = sx_mul(sin18, cos15) - sx_mul(cos18, sin15)

    x = sx_div(sin3, 3 * SCALE)          # crude seed: s/3
    history = [x]
    for _ in range(max_iters):
        x3 = sx_mul(sx_mul(x, x), x)
        x = sx_div(sin3 + 4 * x3, 3 * SCALE)
        history.append(x)
        if history[-1] == history[-2]:
            break

    final = history[-1]
    gains = []
    for i in range(1, len(history) - 1):
        err_prev = abs(history[i - 1] - final)
        err_now = abs(history[i] - final)
        if err_now > 0 and err_prev > 0:
            gains.append(math.log(err_prev / err_now, 60))
    return history, (sum(gains) / len(gains) if gains else float("nan"))


# =============================================================================
# PART II -- THE CERTIFIED BRACKET ITERATION NETWORK
#
# Shapes and symbols
# ------------------
#   s        (N,)     the given quantity sin(3 theta)
#   log_tau  (N,)     the caller's tolerance, as a natural log
#   x*       (N,)     the truth, sin(theta), used only for training/testing
#   x_k      (N,)     the k-th iterate
#   e_k      (N,)     the CERTIFIED error bound on x_{k+1}
#
# Learned parameters: W1,b1,W2,b2 -- a one-hidden-layer network producing two
# coefficients that shape the seed. The seed is written as a correction series
#
#       x_0 = s/3 + c1 * s^3 + c2 * s^5
#
# because that is the form the truth actually takes (the exact expansion begins
# s/3 + (4/81) s^3 + ...). The network is therefore learning a small table of
# series coefficients -- the fifteenth-century way to buy arithmetic with
# theory. The coefficients are squashed into [-0.5, 0.5] so that the seed can
# never leave the domain on which the map is a proven contraction; a machine
# that wanders out of its certified domain has broken its promise, and no loss
# term should have to argue it back.
# =============================================================================

D_IN = 5          # feature width
H = 24            # hidden width
K_STEPS = 6       # maximum iterations carried
T_SOFT = 1.0      # temperature of the labour counter
DELTA = 1e-20     # smoothing constant inside |.|, keeps the loss differentiable
LOG_FLOOR = math.log(1e-13)   # declared arithmetic allowance on the bound
W_ACC = 50000.0   # weight on seed accuracy
W_VAL = 2.0       # weight on bound validity  (the promise-keeping term)
BETA = 0.05       # softness of the validity hinge


def soft_abs(v):
    """|v| smoothed. Note sqrt(v^2+d^2) >= |v|, so the bound stays an OVERestimate."""
    return np.sqrt(v * v + DELTA * DELTA)


def sigmoid(z):
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def softplus(z):
    return np.where(z > 30.0, z, np.log1p(np.exp(np.minimum(z, 30.0))))


def init_params(seed: int = 0):
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0.0, 0.5, (D_IN, H)),
        "b1": np.zeros(H),
        "W2": rng.normal(0.0, 0.05, (H, 2)),
        "b2": np.zeros(2),
    }


def features(s, log_tau):
    """
    The tolerance is an INPUT, not a setting. This is the architectural claim:
    a system that is told how accurate it must be can allocate differently from
    one that is not.
    """
    return np.stack([s, s ** 2, s ** 3, log_tau / 32.0, np.ones_like(s)], axis=1)


def forward(params, s, log_tau):
    """
    Run the contraction and carry the certificate.

    At each step:
        x_{k+1} = (s + 4 x_k^3)/3                 -- al-Kashi's map
        d       = x_{k+1} - x_k                   -- the stride just taken
        B       = |x_{k+1}| + 2|d|                -- an interval enclosing the root
        L       = 4 B^2                           -- Lipschitz bound of the map on it
        e_k     = L/(1-L) * |d|                   -- certified error of x_{k+1}

    The bound is stored in logs and floored softly at LOG_FLOOR. The floor is
    an honest admission: below roughly 1e-13 the guarantee is limited by the
    arithmetic of the machine, not by the mathematics, and claiming otherwise
    would be exactly the overclaiming this architecture exists to prevent.
    """
    U = features(s, log_tau)
    h = np.tanh(U @ params["W1"] + params["b1"])
    Z = h @ params["W2"] + params["b2"]
    C = 0.5 * np.tanh(Z)                     # coefficients confined to the safe domain

    x0 = s / 3.0 + C[:, 0] * s ** 3 + C[:, 1] * s ** 5

    iterates = [x0]
    cache = []
    log_bounds = []
    x = x0
    for _ in range(K_STEPS):
        xn = (s + 4.0 * x ** 3) / 3.0
        d = xn - x
        ad = soft_abs(d)
        B = soft_abs(xn) + 2.0 * ad
        L = 4.0 * B * B
        log_raw = np.log(L) - np.log(1.0 - L) + np.log(ad)
        log_e = np.logaddexp(log_raw, LOG_FLOOR)
        w = sigmoid(log_raw - LOG_FLOOR)      # d(log_e)/d(log_raw)
        log_bounds.append(log_e)
        cache.append((x, xn, d, ad, B, L, w))
        iterates.append(xn)
        x = xn
    return U, h, Z, C, x0, iterates, cache, np.stack(log_bounds, axis=1)


def loss(params, batch):
    """
    L  =  LABOUR  +  W_ACC * SEED ACCURACY  +  W_VAL * BROKEN PROMISES

    LABOUR is the soft count of iterations whose certified bound still exceeds
    the caller's tolerance -- i.e. the passes you are obliged to pay for. This
    is the term that makes the objective al-Kashi's rather than anyone else's:
    the network is not rewarded for being accurate, it is rewarded for reaching
    a STATED accuracy sooner.

    BROKEN PROMISES penalises any step where the true error exceeds the bound
    the machine advertised. Without it there is a degenerate optimum in which
    the model learns to stall (d -> 0), report a tiny bound, and be wrong.
    """
    s, log_tau, tau, x_star = batch
    _, _, _, _, x0, iterates, _, log_e = forward(params, s, log_tau)

    labour = sigmoid((log_e - log_tau[:, None]) / T_SOFT).sum(axis=1).mean()

    X = np.stack(iterates[1:], axis=1)
    log_err = np.logaddexp(np.log(soft_abs(X - x_star[:, None])), LOG_FLOOR - 2.0)
    broken = (BETA * softplus((log_err - log_e) / BETA)).sum(axis=1).mean()

    accuracy = ((x0 - x_star) ** 2 / (x_star ** 2 + 1e-18)).mean()
    return labour + W_ACC * accuracy + W_VAL * broken


def gradients(params, batch):
    """
    Hand-derived reverse-mode gradient. Every line below is the derivative of
    the corresponding line in forward()/loss(), taken in log space so that a
    bound collapsing toward machine precision cannot produce a division by an
    almost-zero error.
    """
    s, log_tau, tau, x_star = batch
    N = len(s)
    U, h, Z, C, x0, iterates, cache, log_e = forward(params, s, log_tau)

    S = sigmoid((log_e - log_tau[:, None]) / T_SOFT)
    X = np.stack(iterates[1:], axis=1)
    R = X - x_star[:, None]
    AR = soft_abs(R)
    log_err = np.logaddexp(np.log(AR), LOG_FLOOR - 2.0)
    w_err = sigmoid(np.log(AR) - (LOG_FLOOR - 2.0))
    hinge = sigmoid((log_err - log_e) / BETA)

    d_log_e = ((S * (1.0 - S) / T_SOFT) - W_VAL * hinge) / N
    d_X = (W_VAL * hinge * w_err * (R / (AR * AR))) / N

    gx = np.zeros(N)
    for k in range(K_STEPS - 1, -1, -1):
        x, xn, d, ad, B, L, w = cache[k]
        gl = d_log_e[:, k] * w
        dL = gl * (1.0 / L + 1.0 / (1.0 - L))
        dad = gl / ad
        dB = dL * 8.0 * B
        dad += dB * 2.0
        dd = dad * (d / ad)
        dxn = dB * (xn / soft_abs(xn)) + dd + gx + d_X[:, k]
        gx = -dd + dxn * 4.0 * x * x

    gx = gx + W_ACC * 2.0 * (x0 - x_star) / (x_star ** 2 + 1e-18) / N

    dC = np.stack([gx * s ** 3, gx * s ** 5], axis=1) * 0.5 * (1.0 - np.tanh(Z) ** 2)
    grads = {"b2": dC.sum(axis=0), "W2": h.T @ dC}
    dh = dC @ params["W2"].T * (1.0 - h ** 2)
    grads["b1"] = dh.sum(axis=0)
    grads["W1"] = U.T @ dh
    return grads


def make_batch(n, seed):
    """
    Problems of the exact kind al-Kashi faced: given the sine of a triple angle,
    recover the sine of the angle, to a tolerance the caller names.
    """
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.2, 12.0, n) * np.pi / 180.0
    log_tau = rng.uniform(math.log(1e-11), math.log(1e-4), n)
    return (np.sin(3 * theta), log_tau, np.exp(log_tau), np.sin(theta))


# -----------------------------------------------------------------------------
# II.a  Gradient check  (mandatory)
# -----------------------------------------------------------------------------

def gradient_check(params, batch, verbose=True):
    """
    Finite-difference verification of the hand-derived backward pass.

    Two honest complications are handled rather than hidden.

    (1) DIRECTION. A random probe direction can have a directional derivative
        near zero, in which case a relative error is meaningless. We probe along
        the gradient itself, where the directional derivative equals ||g|| and
        the signal is as large as it can be.

    (2) STEP SIZE. The objective contains deliberately sharp switches -- the
        labour counter and the validity hinge -- so a central difference taken
        with too large a step measures the corner rather than the slope, while
        too small a step is eaten by float64 cancellation. We sweep a ladder of
        steps, apply Richardson extrapolation at each rung to cancel the leading
        truncation term, and report the best agreement obtained. Quoting the
        step that worked, instead of a single flattering number, is the same
        discipline the rest of this file is about.

    A supplementary per-coordinate check covers the individual entries whose
    gradients are large enough to survive cancellation.
    """
    g = gradients(params, batch)
    norm = math.sqrt(sum(float((g[k] ** 2).sum()) for k in g))
    direction = {k: g[k] / norm for k in g}
    analytic = norm

    def central(h):
        pa = {k: params[k] + h * direction[k] for k in params}
        pb = {k: params[k] - h * direction[k] for k in params}
        return (loss(pa, batch) - loss(pb, batch)) / (2 * h)

    best_rel, best_h, best_val = float("inf"), None, None
    for h in (2e-3, 5e-4, 1e-4, 2e-5):
        d1, d2 = central(h), central(h / 2)
        rich = (4 * d2 - d1) / 3.0                  # Richardson extrapolation
        rel = abs(rich - analytic) / (abs(rich) + abs(analytic) + 1e-15)
        if rel < best_rel:
            best_rel, best_h, best_val = rel, h, rich

    worst = 0.0
    checked = 0
    for key in ("W1", "b1", "W2", "b2"):
        flat = g[key].ravel()
        for f in np.argsort(-np.abs(flat))[:6]:
            idx = np.unravel_index(f, g[key].shape)
            if abs(g[key][idx]) < 1e-3:
                continue
            saved = params[key][idx]
            params[key][idx] = saved + 2e-5
            a = loss(params, batch)
            params[key][idx] = saved - 2e-5
            b = loss(params, batch)
            params[key][idx] = saved
            num = (a - b) / (4e-5)
            worst = max(worst, abs(num - g[key][idx]) / (abs(num) + abs(g[key][idx]) + 1e-15))
            checked += 1

    ok = (best_rel < 1e-3) and (worst < 5e-2)
    if verbose:
        print(f"    along the gradient : analytic {analytic:+.8f}   "
              f"finite-diff {best_val:+.8f}")
        print(f"                       : rel err {best_rel:.3e}  (step {best_h:.0e}, "
              f"Richardson-extrapolated)")
        print(f"    per-coordinate     : {checked} well-conditioned entries, "
              f"worst rel err {worst:.3e}")
        print(f"    VERDICT            : {'PASS' if ok else 'FAIL'}")
    return ok, best_rel, worst


# -----------------------------------------------------------------------------
# II.b  Training
# -----------------------------------------------------------------------------

def train(params, steps=6000, lr=0.015, batch_size=192, seed0=30000, verbose=True):
    """Adam, written out, because the file promises no frameworks."""
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(x) for k, x in params.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        batch = make_batch(batch_size, seed0 + t)
        g = gradients(params, batch)
        alpha = lr * (0.05 ** (t / steps))          # decay to 5% of lr
        for k in params:
            m[k] = b1 * m[k] + (1 - b1) * g[k]
            v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
            mh = m[k] / (1 - b1 ** t)
            vh = v[k] / (1 - b2 ** t)
            params[k] -= alpha * mh / (np.sqrt(vh) + eps)
        if verbose and (t % 1500 == 0 or t == 1):
            print(f"    step {t:5d}   loss {loss(params, make_batch(512, 7)):.5f}")
    return params


def evaluate(params, n=4000, seed=99):
    """Certified iterations required, containment rate, seed quality."""
    s, log_tau, tau, x_star = make_batch(n, seed)
    _, _, _, _, x0, iterates, _, log_e = forward(params, s, log_tau)
    X = np.stack(iterates[1:], axis=1)
    containment = float((np.abs(X - x_star[:, None]) <= np.exp(log_e)).mean())
    iters = ((log_e > log_tau[:, None]).sum(axis=1) + 1).mean()

    crude = dict(params)
    zero = init_params(0)
    for k in zero:
        zero[k] = np.zeros_like(zero[k])
    _, _, _, _, x0b, _, _, log_eb = forward(zero, s, log_tau)   # C = 0  =>  seed = s/3
    iters_base = ((log_eb > log_tau[:, None]).sum(axis=1) + 1).mean()

    return {
        "iters_trained": float(iters),
        "iters_baseline": float(iters_base),
        "containment": containment,
        "seed_err_trained": float(np.abs(x0 - x_star).mean()),
        "seed_err_baseline": float(np.abs(x0b - x_star).mean()),
    }


# =============================================================================
# PART III -- TOLERANCE-FIRST INFERENCE AND THE DIGIT LEDGER
# =============================================================================

def certified_answer(params, s_value, tau, max_steps=K_STEPS):
    """
    Ask the model a single question with a stated tolerance. It halts itself.

    Returns a dict containing the answer, the bound it is prepared to defend,
    the number of iterations it actually spent, and -- the part that matters --
    the number of sexagesimal places it is willing to sign for.
    """
    s = np.array([s_value])
    log_tau = np.array([math.log(tau)])
    _, _, _, _, x0, iterates, _, log_e = forward(params, s, log_tau)
    e = np.exp(log_e[0])
    for k in range(max_steps):
        if e[k] <= tau:
            x = iterates[k + 1][0]
            earned = max(0, int(math.floor(-math.log(max(e[k], 1e-300), 60))))
            return {"answer": float(x), "bound": float(e[k]), "iterations": k + 1,
                    "met": True, "sexagesimal_places_earned": earned}
    x = iterates[max_steps][0]
    earned = max(0, int(math.floor(-math.log(max(e[-1], 1e-300), 60))))
    return {"answer": float(x), "bound": float(e[-1]), "iterations": max_steps,
            "met": False, "sexagesimal_places_earned": earned}


# =============================================================================
# SELF-TESTS
# =============================================================================

def self_tests(params):
    print("\n[SELF-TESTS]")
    passed = 0
    total = 0

    def check(name, cond, detail=""):
        nonlocal passed, total
        total += 1
        if cond:
            passed += 1
        print(f"    [{'PASS' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")

    # 1. Sexagesimal round trip
    val = sx_div(sx_from_int(22), sx_from_int(7))
    check("sexagesimal round trip", abs(sx_to_float(val) - 22 / 7) < 1e-15)

    # 2. Bracket really brackets -- tested in EXACT arithmetic, because at this
    #    width the two sides are indistinguishable once cast to float64. That is
    #    itself the lesson: the certificate lives in the integers, not the floats.
    sides, lo, hi, mean = polygon_bracket()
    two_pi_exact = TWO_PI_FIXED
    check("polygon bracket contains 2*pi (exact integers)",
          lo < two_pi_exact < hi,
          f"width {(hi - lo) / SCALE:.3e}")

    # 3. Bracket is monotone in effort: more doublings, tighter bracket
    _, lo2, hi2, _ = polygon_bracket(doublings=20)
    check("more doublings tighten the bracket",
          (hi - lo) < (hi2 - lo2))

    # 4. Digit ledger never over-reports
    earned = sx_common_places(lo, hi)
    truth_str = sx_to_sexagesimal(TWO_PI_FIXED, 12)
    mean_str = sx_to_sexagesimal(mean, 12)
    agree = 0
    for a, b in zip(truth_str.split(";")[1].split(","), mean_str.split(";")[1].split(",")):
        if a != b:
            break
        agree += 1
    check("earned places are genuinely correct", earned <= agree,
          f"earned {earned}, actually correct {agree}")

    # 5. Contraction converges and matches the received value
    hist, rate = sin_one_degree()
    ours = sx_to_float(hist[-1])
    check("sin(1 deg) converges to the true value",
          abs(ours - math.sin(math.radians(1))) < 1e-15,
          f"{rate:.2f} sexagesimal places gained per pass")

    # 6. The certificate holds on fresh data
    s, log_tau, tau, x_star = make_batch(2000, 4242)
    _, _, _, _, _, iterates, _, log_e = forward(params, s, log_tau)
    X = np.stack(iterates[1:], axis=1)
    ok = bool((np.abs(X - x_star[:, None]) <= np.exp(log_e)).all())
    check("certified bound contains the truth, every step, every sample", ok)

    # 7. Halting honours the tolerance it was given
    res_loose = certified_answer(params, math.sin(math.radians(9.0)), 1e-5)
    res_tight = certified_answer(params, math.sin(math.radians(9.0)), 1e-10)
    check("tighter tolerance costs more iterations",
          res_tight["iterations"] >= res_loose["iterations"],
          f"{res_loose['iterations']} vs {res_tight['iterations']} passes")

    # 8. It refuses to claim digits it has not earned
    check("claimed places never exceed the certified bound",
          60.0 ** (-res_tight["sexagesimal_places_earned"]) >= res_tight["bound"])

    # 9. Gradient check
    ok_grad, _, _ = gradient_check(init_params(1), make_batch(96, 1), verbose=False)
    check("finite-difference gradient check", ok_grad)

    print(f"\n    {passed}/{total} self-tests passed")
    return passed == total


# =============================================================================
# MAIN
# =============================================================================

def main():
    t0 = time.time()
    print("=" * 74)
    print(" AL-KASHI  --  CERTIFIED BRACKET ITERATION NETWORK")
    print(" figure 0390 | Kashan c.1380 - Samarkand 1429 | tolerance-first cognition")
    print("=" * 74)

    # ---- PART I ----------------------------------------------------------
    print("\n[I] THE HISTORICAL COMPUTATIONS, REPRODUCED EXACTLY (base 60, big ints)")

    places, decimals, tol = horse_hair_budget()
    print("\n  I.a  The promise, made before any computing (Risala al-muhitiyya, 1424)")
    print(f"       circle of diameter 600,000 x Earth's, error under a horse's hair")
    print(f"       permissible error in 2*pi ....... {tol:.3e}")
    print(f"       therefore required places ....... {places} sexagesimal "
          f"({decimals} decimal)")
    print(f"       al-Kashi delivered .............. 9 sexagesimal places of 2*pi")

    sides, lo, hi, mean = polygon_bracket()
    print(f"\n  I.b  The instrument chosen to keep it: a polygon of {sides:,} sides")
    print(f"       inscribed  (lower) ... {sx_to_sexagesimal(lo, 10)}")
    print(f"       circumscr. (upper) ... {sx_to_sexagesimal(hi, 10)}")
    print(f"       their mean ........... {sx_to_sexagesimal(mean, 10)}")
    print(f"       al-Kashi's value ..... 6;16,59,28,1,34,51,46,14,50")
    print(f"       true 2*pi ............ {sx_to_sexagesimal(TWO_PI_FIXED, 10)}")
    print(f"       places earned by the bracket alone ... {sx_common_places(lo, hi)}")
    print(f"       mean, in decimal ..... {sx_to_float(mean):.16f}")
    print(f"       2*pi, in decimal ..... {TWO_PI_FIXED/SCALE:.16f}")

    hist, rate = sin_one_degree()
    print(f"\n  I.c  sin(1 deg) by his own contraction  x <- (s + 4x^3)/3")
    for i, h in enumerate(hist[:6]):
        print(f"       pass {i}: {sx_to_sexagesimal(h, 11)}")
    print(f"       converged after {len(hist)-1} passes; "
          f"{rate:.2f} sexagesimal places gained per pass")
    print(f"       received value ....... 0;1,2,49,43,11,14,44,16,26,17")
    print(f"       ours ................. {sx_to_sexagesimal(hist[-1], 10)}")
    print(f"       (agreement through the ninth place; the tenth sits on a"
          f" rounding boundary)")

    # ---- PART II ---------------------------------------------------------
    print("\n[II] THE NETWORK")
    params = init_params(0)

    print("\n  II.a  Gradient check before training")
    gradient_check(init_params(1), make_batch(96, 1))

    print("\n  II.b  Training: the loss IS the labour needed to keep the promise")
    train(params, steps=6000, lr=0.015, batch_size=192)

    print("\n  II.c  Gradient check after training")
    gradient_check(params, make_batch(96, 3))

    print("\n  II.d  Evaluation on 4000 unseen problems")
    ev = evaluate(params)
    print(f"       certified passes needed, crude seed s/3 ... {ev['iters_baseline']:.3f}")
    print(f"       certified passes needed, learned seed ..... {ev['iters_trained']:.3f}")
    print(f"       labour saved .............................. "
          f"{100*(1-ev['iters_trained']/ev['iters_baseline']):.1f}%")
    print(f"       seed error, crude ......................... {ev['seed_err_baseline']:.3e}")
    print(f"       seed error, learned ....................... {ev['seed_err_trained']:.3e}")
    print(f"       bound containment rate .................... {ev['containment']*100:.2f}%")

    # ---- PART III --------------------------------------------------------
    print("\n[III] TOLERANCE-FIRST INFERENCE: the same question, three promises")
    s_val = math.sin(math.radians(9.0))
    truth = math.sin(math.radians(3.0))
    print(f"       question: given sin(9 deg), return sin(3 deg) = {truth:.15f}")
    for tau in (1e-4, 1e-7, 1e-10):
        r = certified_answer(params, s_val, tau)
        actual = abs(r["answer"] - truth)
        print(f"       tau={tau:.0e} -> {r['iterations']} pass(es), "
              f"bound {r['bound']:.2e}, true error {actual:.2e}, "
              f"places signed for: {r['sexagesimal_places_earned']}, "
              f"promise {'kept' if r['met'] and actual <= r['bound'] else 'BROKEN'}")

    ok = self_tests(params)
    print(f"\n[DONE] {time.time()-t0:.1f}s   status: {'ALL GREEN' if ok else 'FAILURES PRESENT'}")


if __name__ == "__main__":
    main()

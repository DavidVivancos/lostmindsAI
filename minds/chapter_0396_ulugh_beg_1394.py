#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 # BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0396 - ULUGH BEG (1394-1449)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 20, Minds 381-400 Available on Amazon https://www.amazon.com/dp/B0HRS9ZHVG
# Resume and Interactive Demos at  https://artificiology.com/
# Videos of the playable minds at https://www.youtube.com/@LostMindsAI
# END ATTRIBUTION  
 THE MERIDIAN MACHINE -- an error-budget architecture with three separate levers
================================================================================

WHAT THIS FILE IS
-----------------
A complete, from-scratch, pure-NumPy learning system built to embody one
cognitive idea, and only that idea. No PyTorch, no autograd, no attention, no
transformer. Every gradient below is derived by hand and verified against
central finite differences before any training is allowed to start.

THE IDEA (Ulugh Beg's, read off his surviving data)
---------------------------------------------------
Ulugh Beg ran the first large, well-funded, decades-long measurement programme
in history whose raw output we can still audit star by star. In 1437 his
observatory published ecliptic coordinates for 1018 stars. Modern comparison of
that catalogue against Hipparcos (Verbunt & van Gent 2012, A&A 544, A31) shows
three quantitatively distinct things happened at once relative to Ptolemy:

    (1) the fraction of entries wrong by more than 150 arcmin fell from about
        2%   (Ptolemy)  to about  0.1%  (Ulugh Beg)   -- a ~20x collapse of
        the TAIL;
    (2) the width of the error distribution fell only modestly, from
        sigma ~ 27'/23'  to  sigma ~ 22'/18'  in longitude/latitude -- the
        MIDDLE barely moved;
    (3) a NEW systematic offset appeared: Ulugh Beg's longitudes sit about
        -11' from truth where Ptolemy's sat at about +7'. The old bias was
        removed by re-observation; a fresh one of comparable size replaced it.

His main instrument, the Fakhri sextant, had a radius of 40.04 m with 0.698 m
of arc per degree -- readable to roughly 5 arcseconds. The catalogue's scatter
is 20 arcminutes. The instrument was ~250x finer than the product it made.

So the lesson his own data teaches, which he never wrote down and probably never
framed this way, is:

    PRECISION, TAIL-RISK AND BIAS ARE THREE INDEPENDENT QUANTITIES.
    EACH HAS ITS OWN LEVER. NO LEVER SUBSTITUTES FOR ANOTHER.
    ENLARGING THE APPARATUS BUYS ONLY THE FIRST, AND ONLY DOWN TO A FLOOR.

THE THREE LEVERS, IMPLEMENTED HERE AS FIRST-CLASS OBJECTS
---------------------------------------------------------
    LEVER 1  APERTURE (R, metres)
             Cuts graduation noise as 1/R. Floors out against an irreducible
             pointing/seeing term that no amount of marble can touch.
             -> in AI terms: scale.

    LEVER 2  VERIFICATION PASSES (V)
             Median-of-V re-observation. Cuts the blunder/tail rate. Does
             almost nothing to the core width and NOTHING to bias.
             -> in AI terms: evaluation, redundancy, review process.

    LEVER 3  A SECOND, STRUCTURALLY DIFFERENT INSTRUMENT
             The only lever that touches systematic bias, because bias is a
             property of the reference frame and is invisible from inside a
             single frame -- a flexible model simply absorbs it into its own
             parameters. Two instruments with non-parallel sensitivity vectors
             make the frame error identifiable.
             -> in AI terms: independent measurement, not more of the same.

THE MODEL
---------
Deliberately NOT a modern default. The predictor is a Ptolemaic rotor bank:
each body carries a mean longitude, a mean motion, and two epicycles. This is
the exact kinematic vocabulary Ulugh Beg used and never revised -- the Zij is,
as van Dalen notes, a standard Ptolemaic work with no adjustment to the
planetary models. He reformed measurement, not theory. The architecture honours
that: the world-model is frozen in its 15th-century form, and all the
intelligence is in the error budget bolted onto it.

Bolted on top: a shared three-component FRAME BIAS vector b = (dphi, deps,
dlam0) -- an error in the assumed latitude, in the obliquity of the ecliptic,
and in the longitude zero point. These are exactly the three constants the
Fakhri sextant existed to pin down.

Emission is metered by AL-KASHI'S FIXED-POINT CUBIC. Ulugh Beg's collaborator
Ghiyath al-Din Jamshid al-Kashi solved sin(1 deg) by iterating the trisection
cubic to a tolerance declared in advance. That routine is implemented and
verified here and used as the machine's precision-on-demand output stage --
with the chapter's caveat attached in code: a declared tolerance governs the
RANDOM part of the answer and says nothing whatever about the systematic part.

RUN
---
    python3 chapter_0396_ulugh_beg_1394.py            # full suite
    python3 chapter_0396_ulugh_beg_1394.py --quick    # short version

Requires only numpy.
"""

import argparse
import sys
import time

import numpy as np

DEG = np.pi / 180.0
ARCMIN = 1.0 / 60.0
ARCSEC = 1.0 / 3600.0

# ==============================================================================
# SECTION 0 -- AL-KASHI'S FIXED-POINT REFINER
#
# The trisection identity  sin(3t) = 3 sin(t) - 4 sin^3(t)  turns "find sin(1
# degree)" into the cubic  4x^3 - 3x + s = 0  with s = sin(3 deg). Al-Kashi
# rearranged it as the fixed-point map
#
#       x  <-  (s + 4 x^3) / 3
#
# whose contraction factor near the root is 4x^2 ~ 0.0037, i.e. roughly two and
# a half decimal digits gained per pass -- knowable in advance, which is why he
# could promise a digit count before starting.
#
# In this file it does real work: the reduction of a raw arc-length reading on
# the sextant into an angle needs small-angle sines to better than the
# instrument's own graduation, and this is how they are obtained.
# ==============================================================================


def alkashi_refine(s, tol, x0=None, max_iter=200):
    """Solve 4x^3 - 3x + s = 0 for the small root by al-Kashi's iteration.

    Returns (root, iterations_spent). The iteration count is returned because
    in al-Kashi's practice effort is a DEPENDENT variable of a declared
    tolerance, not a free choice made afterwards.
    """
    x = (s / 3.0) if x0 is None else x0
    for k in range(1, max_iter + 1):
        x_new = (s + 4.0 * x ** 3) / 3.0
        if abs(x_new - x) < tol:
            return x_new, k
        x = x_new
    return x, max_iter


def sin_one_degree(tol=1e-18):
    """sin(1 deg) via the trisection cubic seeded from sin(3 deg)."""
    return alkashi_refine(np.sin(3.0 * DEG), tol)


# ==============================================================================
# SECTION 1 -- THE SKY (ground truth)
#
# Seven bodies with their real sidereal mean motions in degrees per day. Each
# gets a first-order equation of centre, so the true longitude is genuinely
# quasi-periodic rather than linear -- which is precisely why a deferent alone
# will not fit it and an epicycle is needed.
# ==============================================================================

BODY_NAMES = ["Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn"]

# mean motion, deg/day (real values)
MEAN_MOTION = np.array(
    [0.9856474, 13.1763581, 4.0923771, 1.6021304, 0.5240208, 0.0830912, 0.0334597]
)
# equation-of-centre amplitude, degrees (real order of magnitude)
ECC_AMP = np.array([1.915, 6.289, 23.44, 0.78, 10.69, 5.53, 6.36])

N_BODY = len(BODY_NAMES)


def true_longitude(k_idx, t_days, lam0, varpi):
    """Ecliptic longitude in degrees, wrapped to [0, 360)."""
    n = MEAN_MOTION[k_idx]
    e = ECC_AMP[k_idx]
    mean_anom = n * t_days + varpi[k_idx]
    lam = lam0[k_idx] + n * t_days + e * np.sin(mean_anom * DEG)
    return np.mod(lam, 360.0)


def declination(lam_deg, beta_deg, eps_deg):
    """sin(dec) = sin(beta) cos(eps) + cos(beta) sin(eps) sin(lam)."""
    lam, beta, eps = lam_deg * DEG, beta_deg * DEG, eps_deg * DEG
    sd = np.sin(beta) * np.cos(eps) + np.cos(beta) * np.sin(eps) * np.sin(lam)
    return np.arcsin(np.clip(sd, -1.0, 1.0)) / DEG


# ==============================================================================
# SECTION 2 -- FRAME-BIAS SENSITIVITY (the Jacobian columns J)
#
# The whole architecture turns on the fact that a frame error does not show up
# as a constant. It shows up as a STAR-DEPENDENT PATTERN, and the pattern is
# different for differently-built instruments. That difference is the only
# thing in the universe that makes the frame error recoverable.
#
# CHANNEL 0 -- FAKHRI SEXTANT (meridian altitude, reduced to longitude)
#   An error dphi in the assumed latitude is a pure declination offset. Carrying
#   it through sin(dec) = sin(b)cos(e) + cos(b)sin(e)sin(lam), differentiated at
#   fixed beta, gives   dlam/ddec = cos(dec) / (cos(beta) sin(eps) cos(lam)).
#   An obliquity error at fixed measured declination gives
#     dlam/deps = (sin(beta)sin(eps) - cos(beta)cos(eps)sin(lam))
#                 / (cos(beta) sin(eps) cos(lam)).
#   The zero point of the ecliptic ring is not used, so its coefficient is 0.
#
#   Note the 1/cos(lam): near the solstitial colures a meridian altitude carries
#   almost no longitude information and the propagation diverges. Samarkand's
#   observers knew to avoid that region; the sampler below enforces it as a cut.
#
# CHANNEL 1 -- ARMILLARY SPHERE (direct ecliptic longitude vs a reference star)
#   Does not use the horizon, so dphi has coefficient 0 -- this is the crucial
#   structural difference. Tilting the ecliptic ring by deps about the equinox
#   line perturbs longitude by  tan(beta) cos(lam). The ring's own zero point
#   enters with coefficient exactly 1.
# ==============================================================================

COS_LAM_FLOOR = 0.40  # observational cut: no longitudes taken near the colures


def frame_jacobian(lam_deg, beta_deg, dec_deg, eps_deg, channel):
    """(n,3) sensitivity of a measured longitude to (dphi, deps, dlam0).

    dlam0 -- the adopted longitude of the vernal equinox -- is the ORIGIN of
    the scale every instrument reports on, so its coefficient is 1 everywhere.
    That is what makes it the hardest of the three: it is perfectly mimicked by
    a common mechanical zero shared across the instruments, and no amount of
    cross-checking between them can tell the two apart. Only an observation
    anchored to an external event can (SECTION 4).
    """
    lam, beta, eps, dec = lam_deg * DEG, beta_deg * DEG, eps_deg * DEG, dec_deg * DEG
    n = len(lam_deg)
    J = np.zeros((n, 3))
    J[:, 2] = 1.0

    cl = np.sign(np.cos(lam)) * np.maximum(np.abs(np.cos(lam)), COS_LAM_FLOOR)
    denom = np.cos(beta) * np.sin(eps) * cl

    sext = channel == 0
    J[sext, 0] = -np.cos(dec[sext]) / denom[sext]
    J[sext, 1] = (
        np.sin(beta[sext]) * np.sin(eps)
        - np.cos(beta[sext]) * np.cos(eps) * np.sin(lam[sext])
    ) / denom[sext]

    arm = ~sext
    J[arm, 1] = np.tan(beta[arm]) * np.cos(lam[arm])
    return J


# ==============================================================================
# SECTION 3 -- THE INSTRUMENT (LEVERS 1 AND 2)
#
# Graduation noise. On the real Fakhri sextant one degree of arc occupied
# 0.698 m at R = 40.04 m. A mason who can incise and an observer who can read a
# fixed LINEAR precision of about 1 mm therefore resolves
#
#     0.001 m / 0.698 m/deg  =  0.001433 deg  =  5.16 arcsec
#
# which is exactly the ~5 arcsecond figure the sources report for the
# instrument. That number is not put in by hand below; it falls out of R.
#
# Pointing floor. Naked-eye pointing plus atmosphere contributes a term of
# order one to two arcminutes that is completely independent of R. This is why
# a 5-arcsecond instrument produced a 20-arcminute catalogue: the stars were
# never graduation-limited. LEVER 1 cannot go below this floor. Ever.
#
# Blunders. A separate, heavy-tailed contamination: a misread degree, a
# misidentified star, a transposed digit in the register. Its rate is a
# property of PROCEDURE, not of the marble, and LEVER 2 -- taking V independent
# passes and keeping the median -- is what suppresses it.
# ==============================================================================

LINEAR_READ_PRECISION_M = 0.001  # 1 mm, what a hand and an eye can do on stone
BLUNDER_RATE = 0.06              # per single pass
BLUNDER_SCALE_DEG = 1.2          # gross-error magnitude

# Per-channel hardware. The two instruments are not scaled versions of each
# other; they are different machines with different limits, which is the whole
# point of owning both.
#
#   CHANNEL 0, the Fakhri sextant: enormous, immovable, meridian-only. Its
#   graduation is the finest ever cut before the telescope, and a projected
#   solar image landing on marble is an easy target, so its pointing floor is
#   low. Radius is the swept variable -- this is LEVER 1.
#
#   CHANNEL 1, the armillary sphere on the roof: small, portable, points
#   anywhere. Evans (1987) argued from the catalogue's own number patterns that
#   Samarkand's stellar longitudes came off an armillary graduated to 15
#   arcminutes with interpolation, not off the great sextant at all. Sighting a
#   faint star along a ring by eye is a much worse pointing problem, so its
#   floor is an order of magnitude higher. Its radius is FIXED: you cannot
#   enlarge your way out of this instrument, which is exactly why Ulugh Beg's
#   solar constants are superb and his star positions are not.
ARMILLARY_RADIUS_M = 0.70
POINTING_FLOOR_DEG = np.array([2.0 * ARCMIN, 11.0 * ARCMIN])


def graduation_sigma_deg(radius_m):
    """Angular resolution set purely by the physical size of the arc."""
    arc_per_deg = radius_m * DEG  # metres of arc per degree
    return LINEAR_READ_PRECISION_M / arc_per_deg


def channel_sigmas(radius_m):
    """(graduation, total) noise per channel, in degrees."""
    grad = np.array([graduation_sigma_deg(radius_m),
                     graduation_sigma_deg(ARMILLARY_RADIUS_M)])
    return grad, np.hypot(grad, POINTING_FLOOR_DEG)


def instrument_read(clean_deg, channel, radius_m, passes, rng):
    """Apply LEVER 1 (aperture) and LEVER 2 (verification passes).

    Neither lever touches the frame bias already baked into `clean_deg`: bias
    is invisible to both, which is the finding this whole file exists to make
    checkable.
    """
    n = len(clean_deg)
    grad, sig = channel_sigmas(radius_m)
    g_i, s_i = grad[channel], sig[channel]

    draws = np.empty((passes, n))
    for _ in range(passes):
        y = clean_deg + rng.normal(0.0, 1.0, n) * s_i
        hit = rng.random(n) < BLUNDER_RATE
        y = y + hit * rng.standard_t(2.0, n) * BLUNDER_SCALE_DEG
        draws[_] = np.round(y / g_i) * g_i  # readings snap to the cut lattice
    return np.median(draws, axis=0), sig


# ==============================================================================
# SECTION 4 -- THE REGISTER (dataset assembly)
# ==============================================================================


def equinox_times(lam0, varpi, t_max, target_lam):
    """Dates in the campaign on which the true solar longitude equals target_lam.

    LEVER 4 lives here. An equinox is not a comparison between two readings; it
    is a physical event that DEFINES the zero of the ecliptic scale. Observing
    the Sun on that day yields an absolute longitude with no instrumental zero
    hiding inside it, which is the only way the origin of the frame ever gets
    pinned. The sources are explicit that determining the point of the vernal
    equinox was among the Fakhri sextant's principal jobs.
    """
    def resid(tt):
        ma = MEAN_MOTION[0] * tt + varpi[0]
        lam = lam0[0] + MEAN_MOTION[0] * tt + ECC_AMP[0] * np.sin(ma * DEG)
        return np.mod(lam - target_lam + 180.0, 360.0) - 180.0

    grid = np.linspace(0.0, t_max, 40000)
    d = resid(grid)
    cross = np.where((np.sign(d[:-1]) != np.sign(d[1:])) & (np.abs(d[:-1]) < 10))[0]
    lo, hi = grid[cross], grid[cross + 1]
    for _ in range(60):  # bisection to machine precision on the crossing date
        mid = 0.5 * (lo + hi)
        same = np.sign(resid(mid)) == np.sign(resid(lo))
        lo = np.where(same, mid, lo)
        hi = np.where(same, hi, mid)
    return 0.5 * (lo + hi)


class Register:
    """One observing campaign, in the form the calculators would have used."""

    def __init__(self, n_obs, radius_m, passes, bias_true, channels, seed=0,
                 eps_true=23.5047, years=20.0, n_anchor=0):
        rng = np.random.default_rng(seed)
        self.radius_m, self.passes = radius_m, passes
        self.bias_true = np.asarray(bias_true, dtype=float)
        self.eps_true = eps_true

        # one fixed sky realisation, shared by every campaign in this file
        sky = np.random.default_rng(20250316)
        self.lam0 = sky.uniform(0, 360, N_BODY)
        self.varpi = sky.uniform(0, 360, N_BODY)
        self.beta_body = sky.uniform(-6.0, 6.0, N_BODY)
        self.beta_body[0] = 0.0  # the Sun defines the ecliptic

        t_max = years * 365.25
        over = int(n_obs * 3.0) + 400
        k = rng.integers(0, N_BODY, over)
        t = rng.uniform(0.0, t_max, over)
        ch = np.asarray(rng.choice(channels, over), dtype=int)
        lam = true_longitude(k, t, self.lam0, self.varpi)
        keep = np.abs(np.cos(lam * DEG)) > COS_LAM_FLOOR
        k, t, ch = k[keep][:n_obs], t[keep][:n_obs], ch[keep][:n_obs]
        anchor = np.zeros(len(k), dtype=bool)

        # LEVER 4: solar observations on the two equinoxes, always on the sextant
        if n_anchor > 0:
            ta = []
            for target in (0.0, 180.0):
                ts = equinox_times(self.lam0, self.varpi, t_max, target)
                if len(ts):
                    ta.append(rng.choice(ts, size=max(1, n_anchor // 2),
                                         replace=True))
            ta = np.concatenate(ta)
            k = np.concatenate([k, np.zeros(len(ta), dtype=int)])
            t = np.concatenate([t, ta])
            ch = np.concatenate([ch, np.zeros(len(ta), dtype=int)])
            anchor = np.concatenate([anchor, np.ones(len(ta), dtype=bool)])

        lam = true_longitude(k, t, self.lam0, self.varpi)
        beta = self.beta_body[k]
        dec = declination(lam, beta, eps_true)
        J = frame_jacobian(lam, beta, dec, eps_true, ch)

        clean = lam + J @ self.bias_true
        y, self.sigma_used = instrument_read(clean, ch, radius_m, passes, rng)

        self.k, self.t, self.ch, self.lam_true = k, t, ch, lam
        self.beta, self.dec, self.J, self.y = beta, dec, J, y
        self.clean, self.anchor, self.n = clean, anchor, len(y)

    def __len__(self):
        return self.n


# ==============================================================================
# SECTION 5 -- THE ROTOR BANK (the world-model)
#
# Ulugh Beg never touched planetary theory. The Zij is, in van Dalen's phrase,
# a standard Ptolemaic work with no adjustment to the models. So the predictor
# here is deliberately Ptolemaic and deliberately frozen: for each body, a mean
# longitude correction, a mean-motion correction, and D epicycles. The learning
# happens around the world-model, not inside it -- which is exactly the shape of
# his career.
#
# A zij is organised as mean-motion tables plus equation tables. This file
# mirrors that: the NOMINAL ephemeris (the inherited Ilkhani-style tables, with
# their inherited small errors) is subtracted first, and the network learns only
# the equations and corrections on top. All quantities in degrees; time is
# rescaled to tau in [0,1] over the campaign so the optimiser sees a
# well-conditioned problem.
# ==============================================================================

N_EPI = 3  # deferent correction + two epicycles; the third is free capacity


def nominal_tables(register, jitter=1e-4, seed=7):
    """The inherited mean-motion tables: right in form, slightly wrong in value."""
    rng = np.random.default_rng(seed)
    lam0 = register.lam0 + rng.normal(0, 0.35, N_BODY)
    n = MEAN_MOTION * (1.0 + rng.normal(0, jitter, N_BODY))
    return lam0, n


def unwrapped_truth(register):
    """True longitude without the mod-360 wrap: revolutions were counted."""
    n = MEAN_MOTION[register.k]
    e = ECC_AMP[register.k]
    ma = n * register.t + register.varpi[register.k]
    return register.lam0[register.k] + n * register.t + e * np.sin(ma * DEG)


class Batch:
    """Everything the forward pass needs, precomputed once."""

    def __init__(self, register, tables):
        lam0_nom, n_nom = tables
        self.k = register.k
        self.ch = register.ch
        self.J = register.J
        self.tau = register.t / register.t.max()
        self.t_days = register.t

        nominal = lam0_nom[self.k] + n_nom[self.k] * register.t
        truth_unwrapped = unwrapped_truth(register)
        # the register holds a wrapped reading; restore the revolution count
        wraps = np.round((truth_unwrapped - register.y) / 360.0)
        self.y = (register.y + 360.0 * wraps) - nominal
        self.y_truth = truth_unwrapped - nominal
        self.n = register.n
        self.t_scale = register.t.max()
        # LEVER 4: an anchored reading carries no free instrumental zero,
        # because it is tied to an event rather than to another reading.
        self.use_c = (~register.anchor).astype(float)
        self.anchor = register.anchor
        # On an anchored row the body's longitude is fixed by the event itself,
        # so the world-model contributes nothing and has nothing to absorb with.
        # This is the whole force of LEVER 4: it removes the free parameter that
        # was hiding the frame origin, rather than adding more data.
        self.use_body = (~register.anchor).astype(float)
        self.known = np.where(register.anchor, self.y_truth, 0.0)

    def init_params(self, seed=3):
        rng = np.random.default_rng(seed)
        harmonics = np.stack(
            [MEAN_MOTION, 2.0 * MEAN_MOTION, 3.0 * MEAN_MOTION], axis=1
        )
        om = harmonics * self.t_scale * DEG  # radians per campaign
        om = om * (1.0 + rng.normal(0, 2e-4, om.shape))
        # The equation tables are INHERITED, not invented: a zij arrives with an
        # equation of centre for every body and the astronomer corrects it. So
        # the first epicycle starts at the received amplitude, slightly wrong,
        # and the higher ones start near zero.
        A0 = np.zeros((N_BODY, N_EPI))
        A0[:, 0] = ECC_AMP * (1.0 + rng.normal(0, 0.05, N_BODY))
        A0[:, 1:] = rng.normal(0, 0.05, (N_BODY, N_EPI - 1))
        return {
            "a": rng.normal(0, 0.05, N_BODY),
            "m": rng.normal(0, 0.05, N_BODY),
            "A": A0,
            "om": om,
            "ps": rng.uniform(0, 2 * np.pi, (N_BODY, N_EPI)),
            "c": np.zeros(2),
            "b": np.zeros(3),
        }


BIAS_FREE = np.ones(3)          # all three frame components estimated
BIAS_NONE = np.zeros(3)         # frame clamped to zero entirely
BIAS_NO_ORIGIN = np.array([1.0, 1.0, 0.0])   # origin clamped, the rest free


def forward(p, batch, bias_mask=None):
    """yhat = deferent + epicycles + instrument zero + frame bias."""
    k, ch, tau = batch.k, batch.ch, batch.tau
    theta = p["om"][k] * tau[:, None] + p["ps"][k]
    u = np.sin(theta)
    rot = np.sum(p["A"][k] * u, axis=1)
    b = p["b"] if bias_mask is None else p["b"] * bias_mask
    yhat = (batch.use_body * (p["a"][k] + p["m"][k] * tau + rot)
            + batch.known + p["c"][ch] * batch.use_c + batch.J @ b)
    return yhat, {"theta": theta, "u": u}


def loss_and_grads(p, batch, bias_mask=None):
    """Mean squared residual, with every gradient derived by hand."""
    yhat, cache = forward(p, batch, bias_mask)
    r = yhat - batch.y
    n = batch.n
    L = float(np.mean(r ** 2))
    g = 2.0 * r / n  # dL/dyhat

    k, ch, tau = batch.k, batch.ch, batch.tau
    theta, u = cache["theta"], cache["u"]
    cos_t = np.cos(theta)

    gb = g * batch.use_body
    G = {key: np.zeros_like(val) for key, val in p.items()}
    np.add.at(G["a"], k, gb)
    np.add.at(G["m"], k, gb * tau)
    np.add.at(G["A"], k, gb[:, None] * u)

    # d yhat / d theta = A ; theta = om*tau + ps
    dtheta = gb[:, None] * p["A"][k] * cos_t
    np.add.at(G["om"], k, dtheta * tau[:, None])
    np.add.at(G["ps"], k, dtheta)

    np.add.at(G["c"], ch, g * batch.use_c)
    G["b"] = batch.J.T @ g
    if bias_mask is not None:
        G["b"] = G["b"] * bias_mask
    return L, G, r


# ==============================================================================
# SECTION 6 -- FINITE-DIFFERENCE GRADIENT CHECK (mandatory, runs before training)
# ==============================================================================


def gradient_check(batch, seed=11, n_probe=6, tol=1e-6, verbose=True):
    """Central differences against the analytic gradient, every parameter group.

    Each probe is evaluated across six step sizes and the best agreement is kept.
    That is not leniency: a single fixed step is either too large (truncation
    error) or too small (catastrophic cancellation in float64), and which of
    the two dominates depends on the parameter's own scale. Sweeping the step
    and taking the minimum is the standard way to separate a real gradient bug
    from an artefact of the differencing -- and it is, pleasingly, the same
    move Ulugh Beg's calculators made when they took several readings and kept
    the one the instrument was actually able to support.
    """
    rng = np.random.default_rng(seed)
    p = batch.init_params(seed=seed)
    p["b"] = rng.normal(0, 0.05, 3)
    p["c"] = rng.normal(0, 0.05, 2)

    _, G, _ = loss_and_grads(p, batch)
    worst = 0.0
    rows = []
    for name in ["a", "m", "A", "om", "ps", "c", "b"]:
        arr = p[name]
        flat = arr.reshape(-1)
        idxs = rng.choice(flat.size, size=min(n_probe, flat.size), replace=False)
        errs = []
        for i in idxs:
            ana = G[name].reshape(-1)[i]
            scale = max(1.0, abs(flat[i]))
            best = np.inf
            for frac in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
                h = frac * scale
                old_v = flat[i]
                flat[i] = old_v + h
                Lp, _, _ = loss_and_grads(p, batch)
                flat[i] = old_v - h
                Lm, _, _ = loss_and_grads(p, batch)
                flat[i] = old_v
                num = (Lp - Lm) / (2 * h)
                denom = max(1e-12, abs(num) + abs(ana))
                best = min(best, abs(num - ana) / denom)
            errs.append(best)
        e = float(max(errs))
        worst = max(worst, e)
        rows.append((name, arr.size, e))

    if verbose:
        print("  parameter   size   max relative error (analytic vs central diff)")
        for name, size, e in rows:
            print(f"   {name:<9}  {size:>5}   {e:.3e}   "
                  f"{'ok' if e < tol else 'FAIL'}")
        print(f"  worst over all groups: {worst:.3e}   (tolerance {tol:.0e})")
    return worst < tol, worst


# ==============================================================================
# SECTION 7 -- TRAINING (Adam, written out)
# ==============================================================================


def train(batch, steps=4000, lr=0.05, seed=3, bias_mask=None, verbose=False):
    p = batch.init_params(seed=seed)
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(val) for k, val in p.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    hist = []
    for step in range(1, steps + 1):
        L, G, _ = loss_and_grads(p, batch, bias_mask)
        for key in p:
            m[key] = b1 * m[key] + (1 - b1) * G[key]
            v[key] = b2 * v[key] + (1 - b2) * G[key] ** 2
            mh = m[key] / (1 - b1 ** step)
            vh = v[key] / (1 - b2 ** step)
            p[key] = p[key] - lr * mh / (np.sqrt(vh) + eps)
        if step % 250 == 0 or step == 1:
            hist.append((step, L))
            if verbose:
                print(f"    step {step:5d}   loss {L:.6e}   "
                      f"b_hat {np.array2string(p['b'], precision=4)}")
    return p, hist


# ==============================================================================
# SECTION 8 -- IDENTIFIABILITY (LEVER 3, proved rather than asserted)
#
# The claim is that a systematic frame error is invisible from inside a single
# instrument because a flexible world-model absorbs it. That is a statement
# about linear algebra, so it can be settled exactly rather than argued.
#
# Build the design Jacobian dyhat/dtheta. Split the columns into the NUISANCE
# block (rotors, deferents, instrument zeros) and the BIAS block (3 columns).
# Residualise the bias block against the nuisance span:
#
#     Jb_perp = Jb - Jn (Jn^T Jn)^+ Jn^T Jb
#
# The smallest singular value s of Jb_perp is the information the campaign
# actually carries about the frame. The standard error on the frame estimate
# scales as sigma / s. If s collapses, no quantity of data at that noise level
# will ever recover the offset -- the bias is in the null space of the loss.
# ==============================================================================


def design_jacobian(p, batch):
    """(n, P) Jacobian; returns it split into (nuisance, bias)."""
    k, ch, tau = batch.k, batch.ch, batch.tau
    theta = p["om"][k] * tau[:, None] + p["ps"][k]
    u, cos_t = np.sin(theta), np.cos(theta)
    n = batch.n
    cols = []

    onehot = np.zeros((n, N_BODY))
    onehot[np.arange(n), k] = batch.use_body
    cols.append(onehot)                       # d/da
    cols.append(onehot * tau[:, None])        # d/dm
    for d in range(N_EPI):
        cols.append(onehot * u[:, d][:, None])                       # d/dA
        cols.append(onehot * (p["A"][k, d] * cos_t[:, d] * tau)[:, None])  # d/dom
        cols.append(onehot * (p["A"][k, d] * cos_t[:, d])[:, None])        # d/dps
    ch_onehot = np.zeros((n, 2))
    ch_onehot[np.arange(n), ch] = batch.use_c
    cols.append(ch_onehot)                    # d/dc

    Jn = np.concatenate(cols, axis=1)
    Jb = batch.J.copy()
    return Jn, Jb


def frame_information(p, batch, sigma_deg):
    """Per-component standard error on (dphi, deps, dlam0), in degrees.

    A component returns inf when its direction lies in the null space of the
    residualised design -- i.e. when the campaign contains literally no
    information about it and never will, however long it runs.
    """
    Jn, Jb = design_jacobian(p, batch)
    coef, *_ = np.linalg.lstsq(Jn, Jb, rcond=None)
    Jb_perp = Jb - Jn @ coef
    sv = np.linalg.svd(Jb_perp, compute_uv=False)
    M = Jb_perp.T @ Jb_perp
    scale = max(np.max(np.abs(M)), 1e-300)
    Minv = np.linalg.pinv(M, rcond=1e-10)
    var = np.diag(Minv) * np.mean(np.atleast_1d(sigma_deg) ** 2)
    # a direction with no support returns 0 from the pseudo-inverse; that is
    # not certainty, it is absence of information, so report it as infinite
    dead = np.diag(M) / scale < 1e-9
    dead |= np.diag(Minv) <= 0
    se = np.where(dead, np.inf, np.sqrt(np.abs(var)))
    cond = sv.max() / sv.min() if sv.min() > 0 else np.inf
    return se, sv, cond


# ==============================================================================
# SECTION 9 -- THE PROFILE TEST (global identifiability)
#
# The local information matrix of SECTION 8 answers "how sharply is the frame
# pinned near here?" That is the wrong question, and answering it is how a
# measurement programme fools itself. The right question is global:
#
#     CAN THE MACHINE FIT THE DATA JUST AS WELL WHILE BEING WRONG ABOUT THE
#     FRAME?
#
# So: train once with the frame free, then train again with the frame clamped
# to zero and EVERY other parameter re-fitted from scratch to compensate. If
# the second run reaches the same loss, the frame error is free -- the rotors
# have simply absorbed it, and no optimiser, no regulariser and no quantity of
# further data on the same channel will ever recover it. The gap between the
# two losses, expressed as an equivalent scatter in arcminutes, is the amount
# of evidence the campaign actually holds about its own frame.
# ==============================================================================

TAIL_THRESHOLD_DEG = 150 * ARCMIN  # the threshold used by Verbunt & van Gent (2012)


def robust_sigma(x):
    """1.4826 * MAD -- a width that ignores the tail, as the historians' fits do."""
    return 1.4826 * float(np.median(np.abs(x - np.median(x))))


def profile_gap(batch, steps, seed, clamp=BIAS_NO_ORIGIN):
    """Refit EVERYTHING with one frame component nailed to zero, and compare.

    `clamp` defaults to holding the frame ORIGIN at zero while leaving the
    latitude and obliquity terms free. That isolates the question this file
    exists to ask: can the machine be wrong about where its scale starts, and
    pay nothing for it?
    """
    p_free, _ = train(batch, steps=steps, seed=seed)
    p_clamp, _ = train(batch, steps=steps, seed=seed, bias_mask=clamp)
    L_free, _, _ = loss_and_grads(p_free, batch)
    L_clamp, _, _ = loss_and_grads(p_clamp, batch, bias_mask=clamp)
    return p_free, np.sqrt(max(L_free, 0.0)), np.sqrt(max(L_clamp, 0.0))


def register_stats(reg):
    """Three numbers about the raw register, kept strictly apart."""
    m = ~reg.anchor                    # score the catalogue, not the calibration
    noise = (reg.y - reg.clean)[m]     # random part only: frame bias divided out
    total = (reg.y - reg.lam_true)[m]  # what a modern catalogue comparison sees
    return {
        "noise_sigma": robust_sigma(noise),
        "offset": float(np.median(total)),
        "systematic_spread": robust_sigma((reg.J @ reg.bias_true)[m]),
        "tail_frac": float(np.mean(np.abs(total) > TAIL_THRESHOLD_DEG)),
        "total_sigma": robust_sigma(total),
    }


def model_stats(p, batch):
    yhat, _ = forward(p, batch)
    err = (yhat - batch.y_truth)[~batch.anchor]
    return {"model_sigma": robust_sigma(err), "model_offset": float(np.median(err))}


def run_campaign(n_obs, radius_m, passes, channels, bias_true, steps, seed,
                 n_anchor=0, profile=False):
    reg = Register(n_obs, radius_m, passes, bias_true, channels, seed=seed,
                   n_anchor=n_anchor)
    batch = Batch(reg, nominal_tables(reg))
    if profile:
        p, rms_free, rms_clamp = profile_gap(batch, steps, seed)
    else:
        p, _ = train(batch, steps=steps, seed=seed)
        rms_free = rms_clamp = np.nan
    se, sv, cond = frame_information(p, batch, reg.sigma_used[reg.ch])
    st = register_stats(reg)
    st.update(model_stats(p, batch))
    grad, sig = channel_sigmas(radius_m)
    st.update({
        "b_hat": p["b"].copy(), "b_err": p["b"] - np.asarray(bias_true),
        "se": se, "smin": float(sv.min()), "cond": float(cond),
        "grad_sigma": grad[0], "chan_sigma": sig[0],
        "rms_free": rms_free, "rms_clamp": rms_clamp,
        "gap": rms_clamp - rms_free,
    })
    return st, p, batch, reg


# ==============================================================================
# SECTION 10 -- MAIN
# ==============================================================================

# Two of these three are historically measured values. The longitude zero point
# is -11 arcmin, the offset Verbunt & van Gent (2012) find in the 1437
# catalogue. The obliquity error is 32 arcseconds, the reported error in Ulugh
# Beg's 23 deg 30' 17". The latitude error is a stand-in at 3 arcmin.
BIAS_TRUE = np.array([3.0 * ARCMIN, 32 * ARCSEC, -11 * ARCMIN])

FAKHRI_RADIUS_M = 40.04
COMP = ["dphi  (latitude)", "deps  (obliquity)", "dlam0 (frame origin)"]


def banner(text):
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def inf_fmt(x, w=8):
    return "     inf" if not np.isfinite(x) else f"{x:{w}.3f}'"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    N_OBS = 1200 if args.quick else 3200
    STEPS = 1200 if args.quick else 3500
    t0 = time.time()

    print("=" * 78)
    print(" THE MERIDIAN MACHINE -- chapter 0396, Ulugh Beg (1394-1449)")
    print(" three levers on one error budget, and one thing no lever can reach")
    print("=" * 78)

    # ------------------------------------------------------------------ 0
    banner(" 0. AL-KASHI'S REFINER, AND WHERE FIVE ARCSECONDS COMES FROM")
    root, iters = sin_one_degree(tol=1e-18)
    print(f"  sin(1 deg), trisection cubic : {root:.18f}")
    print(f"  library reference            : {np.sin(DEG):.18f}")
    print(f"  agreement {abs(root - np.sin(DEG)):.3e} after {iters} passes."
          f"  Effort follows the tolerance declared first:")
    print("    " + "   ".join(f"{t:.0e}->{sin_one_degree(tol=t)[1]}"
                              for t in (1e-4, 1e-8, 1e-12, 1e-16)))
    g = graduation_sigma_deg(FAKHRI_RADIUS_M)
    ga = graduation_sigma_deg(ARMILLARY_RADIUS_M)
    print(f"\n  Fakhri sextant, radius {FAKHRI_RADIUS_M} m")
    print(f"    arc per degree      {FAKHRI_RADIUS_M * DEG:.4f} m"
          f"     sources report 0.698 m")
    print(f"    1 mm read precision {g * 3600:6.2f}\"      sources report about 5\"")
    print(f"    pointing floor      {POINTING_FLOOR_DEG[0] * 60:6.2f}'"
          f"      {POINTING_FLOOR_DEG[0] / g:.0f}x coarser than the graduation")
    print(f"  Armillary sphere, radius {ARMILLARY_RADIUS_M} m")
    print(f"    graduation          {ga * 60:6.2f}'")
    print(f"    pointing floor      {POINTING_FLOOR_DEG[1] * 60:6.2f}'"
          f"      the instrument the star catalogue actually came off")

    # ------------------------------------------------------------------ 1
    banner(" 1. GRADIENT CHECK (must pass before any training is permitted)")
    reg0 = Register(600, FAKHRI_RADIUS_M, 3, BIAS_TRUE, [0, 1], seed=1, n_anchor=40)
    ok, worst = gradient_check(Batch(reg0, nominal_tables(reg0)))
    if not ok:
        print("  GRADIENT CHECK FAILED -- aborting.")
        return 1
    print("  PASS")

    # ------------------------------------------------------------------ 2
    banner(" 2. LEVER 1 -- APERTURE.  Buys precision, down to a floor. Nothing else.")
    print("  sextant alone, one pass; radius swept from a hand arc to Samarkand\n")
    print("    R (m)   graduation   random sigma   tail>150'   frame error |dphi|")
    print("    " + "-" * 70)
    radii = [0.25, 2.0, 40.04] if args.quick else [0.25, 0.5, 1.0, 2.0, 10.0, 40.04]
    lever1 = []
    for R in radii:
        st, _, _, _ = run_campaign(N_OBS, R, 1, [0], BIAS_TRUE, STEPS, seed=21)
        lever1.append((R, st))
        print(f"    {R:6.2f}   {st['grad_sigma'] * 60:8.3f}'   "
              f"{st['noise_sigma'] * 60:8.2f}'    {st['tail_frac'] * 100:6.2f}%"
              f"     {abs(st['b_err'][0]) * 60:9.2f}'")
    r0, r1 = lever1[0][1], lever1[-1][1]
    print(f"\n  a {radii[-1] / radii[0]:.0f}x increase in physical size cuts the graduation"
          f" {r0['grad_sigma'] / r1['grad_sigma']:.0f}x")
    print(f"  but the achieved width only {r0['noise_sigma'] / r1['noise_sigma']:.1f}x,"
          f" because it meets a pointing floor it cannot cross.")
    print("  The tail does not move. The frame error does not move.")

    # ------------------------------------------------------------------ 3
    banner(" 3. LEVER 2 -- VERIFICATION PASSES.  Buys the tail. Nothing else.")
    print("  full aperture; number of independent re-observations swept\n")
    print("    passes   random sigma   tail>150'   frame error |dphi|")
    print("    " + "-" * 58)
    lever2 = []
    for V in ([1, 5] if args.quick else [1, 3, 5, 9]):
        st, _, _, _ = run_campaign(N_OBS, FAKHRI_RADIUS_M, V, [0], BIAS_TRUE,
                                   STEPS, seed=22)
        lever2.append((V, st))
        print(f"    {V:6d}   {st['noise_sigma'] * 60:8.2f}'    "
              f"{st['tail_frac'] * 100:6.2f}%     {abs(st['b_err'][0]) * 60:9.2f}'")
    print("\n  procedure collapses the gross-error rate and leaves the systematic")
    print("  offset exactly where it was. Reviewing your work more carefully does")
    print("  not tell you that your yardstick is the wrong length.")

    # ------------------------------------------------------------------ 4
    banner(" 4. LEVER 3 -- CONTACT WITH SOMETHING OUTSIDE THE MEASURING SYSTEM")
    print("  Same aperture, same passes, same number of readings throughout.")
    print("  Only the SOURCE of the readings changes.\n")
    cfgs = [
        ("sextant alone", [0], 0),
        ("+ armillary", [0, 1], 0),
        ("+ equinox anchors", [0, 1], 120),
    ]
    res = {}
    for name, chans, anc in cfgs:
        st, _, _, _ = run_campaign(N_OBS, FAKHRI_RADIUS_M, 5, chans, BIAS_TRUE,
                                   STEPS, seed=23, n_anchor=anc, profile=True)
        res[name] = st

    print("    recovered frame, absolute error against truth")
    print("    " + "-" * 74)
    print(f"    {'component':<24}{'true':>8}" +
          "".join(f"{n:>17}" for n, _, _ in cfgs))
    for i, nm in enumerate(COMP):
        row = f"    {nm:<24}{BIAS_TRUE[i] * 60:+7.2f}'"
        for n, _, _ in cfgs:
            row += f"{abs(res[n]['b_err'][i]) * 60:15.3f}' "
        print(row)
    print("\n  Latitude and obliquity come out of a single instrument cleanly. They")
    print("  are ordinary parameters: wrong, but wrong in a way the residuals can")
    print("  feel. The origin does not, and adding a whole second instrument built")
    print("  on a different principle does not help it at all.")

    print("\n    is the frame origin present in the objective at all?")
    print("    (standard error on dlam0; inf = an exact null direction)")
    print("    " + "-" * 74)
    for n, _, _ in cfgs:
        print(f"      {n:<24} {inf_fmt(res[n]['se'][2] * 60)}"
              f"    smallest singular value {res[n]['smin']:.3e}")

    print("\n    THE PROFILE TEST -- nail the origin to zero, refit everything else")
    print("    (rms residual in arcmin; a gap of zero means the machine can be")
    print("     wrong about where its own scale starts at no cost whatsoever)")
    print("    " + "-" * 74)
    print(f"    {'configuration':<24}{'origin free':>13}{'origin clamped':>17}{'gap':>11}")
    for n, _, _ in cfgs:
        st = res[n]
        print(f"    {n:<24}{st['rms_free'] * 60:12.4f}'{st['rms_clamp'] * 60:16.4f}'"
              f"{st['gap'] * 60:10.4f}'")
    print("\n  Read the last column. With any number of instruments the campaign pays")
    print("  NOTHING for being wrong about its own origin: the mean-longitude tables")
    print("  absorb the offset exactly, and the residuals never notice. The offset is")
    print("  not hard to find. It is not in the objective. Cross-checking machines")
    print("  against each other cannot reach it, because they share the scale whose")
    print("  zero is in question.")
    print("\n  What reaches it is an equinox: a moment the sky DEFINES as longitude")
    print("  zero rather than reports relative to something else. One such event")
    print("  removes the free parameter the error was hiding in. Observing both")
    print("  equinoxes separates the origin from a latitude error, because the")
    print("  sensitivity reverses sign between them. This is the whole reason the")
    print("  sources list 'the point of the vernal equinox' among the Fakhri")
    print("  sextant's principal jobs.")

    # ------------------------------------------------------------------ 5
    banner(" 5. THE 1437 ERROR STRUCTURE, REPRODUCED")
    crude, _, _, _ = run_campaign(N_OBS, 0.25, 1, [0, 1], BIAS_TRUE, STEPS, seed=137)
    samar, _, _, _ = run_campaign(N_OBS, FAKHRI_RADIUS_M, 5, [0, 1], BIAS_TRUE,
                                  STEPS, seed=1437)
    print("                               crude arc, 1 pass     Samarkand programme")
    print("    " + "-" * 70)
    print(f"    longitude offset          {crude['offset'] * 60:+9.2f}'"
          f"            {samar['offset'] * 60:+9.2f}'")
    print(f"    random width sigma        {crude['noise_sigma'] * 60:9.2f}'"
          f"            {samar['noise_sigma'] * 60:9.2f}'")
    print(f"    fraction wrong by >150'   {crude['tail_frac'] * 100:9.2f}%"
          f"            {samar['tail_frac'] * 100:9.2f}%")
    print("\n    the shape the surviving catalogue actually has")
    print("    (Verbunt & van Gent 2012, Ptolemy -> Ulugh Beg):")
    print("      offset      +7'   ->   -11'    replaced, not removed")
    print("      sigma        27'  ->    22'    barely moved")
    print("      tail>150'   2.0%  ->   0.1%    collapsed")

    # ------------------------------------------------------------------ 6
    banner(" 6. SELF-TESTS")
    one, two, anch = res["sextant alone"], res["+ armillary"], res["+ equinox anchors"]
    tests = [
        ("al-Kashi root matches the library value to 1e-15",
         abs(sin_one_degree(1e-18)[0] - np.sin(DEG)) < 1e-15),
        ("al-Kashi effort grows with the tolerance demanded",
         sin_one_degree(1e-16)[1] > sin_one_degree(1e-4)[1]),
        ("declination at lam=90 equals the obliquity",
         abs(declination(np.array([90.]), np.array([0.]), 23.5047)[0] - 23.5047)
         < 1e-9),
        ("graduation noise scales as 1/R",
         abs(graduation_sigma_deg(10.) / graduation_sigma_deg(40.) - 4.) < 1e-9),
        ("a 40 m arc resolves close to five arcseconds",
         4.0 < graduation_sigma_deg(FAKHRI_RADIUS_M) * 3600 < 6.0),
        ("armillary is blind to a latitude error",
         frame_jacobian(np.array([30.]), np.array([2.]), np.array([10.]), 23.5,
                        np.array([1]))[0, 0] == 0.0),
        ("the frame origin is felt by every instrument alike",
         frame_jacobian(np.array([30., 30.]), np.array([2., 2.]),
                        np.array([10., 10.]), 23.5, np.array([0, 1]))[0, 2] == 1.0),
        ("equinox solver lands on the crossing to 1e-6 deg",
         abs(np.mod(true_longitude(
             np.array([0]),
             equinox_times(reg0.lam0, reg0.varpi, 20 * 365.25, 0.0)[:1],
             reg0.lam0, reg0.varpi)[0] + 180., 360.) - 180.) < 1e-6),
        ("analytic gradients match central differences", ok),

        ("LEVER 1: aperture cuts the random width by >3x",
         r0["noise_sigma"] > 3 * r1["noise_sigma"]),
        ("LEVER 1: aperture floors out well short of the graduation",
         r1["noise_sigma"] > 5 * r1["grad_sigma"]),
        ("LEVER 1: aperture does not cut the tail",
         r1["tail_frac"] > 0.4 * r0["tail_frac"]),
        ("LEVER 1: aperture does not touch the frame origin",
         abs(r1["b_err"][2]) > 0.5 * abs(r0["b_err"][2])),

        ("LEVER 2: passes cut the tail by >5x",
         lever2[-1][1]["tail_frac"] * 5 < lever2[0][1]["tail_frac"] + 1e-9),
        ("LEVER 2: passes do not touch the frame origin",
         abs(lever2[-1][1]["b_err"][2]) > 0.5 * abs(lever2[0][1]["b_err"][2])),

        ("latitude and obliquity ARE recoverable from one instrument",
         max(abs(one["b_err"][0]), abs(one["b_err"][1])) * 60 < 0.5),
        ("NEGATIVE RESULT: a second instrument does not recover the origin",
         (not np.isfinite(one["se"][2])) and (not np.isfinite(two["se"][2]))
         and abs(two["b_err"][2]) * 60 > 3.0),
        ("NEGATIVE RESULT: clamping the origin costs nothing without an anchor",
         abs(one["gap"]) * 60 < 0.05 and abs(two["gap"]) * 60 < 0.05),

        ("LEVER 3: an anchored event makes the origin identifiable",
         np.isfinite(anch["se"][2])),
        ("LEVER 3: anchoring recovers the origin to better than 1 arcmin",
         abs(anch["b_err"][2]) * 60 < 1.0),
        ("LEVER 3: with an anchor, a wrong origin finally costs something",
         anch["gap"] > 100 * max(abs(one["gap"]), abs(two["gap"]), 1e-9)),

        ("the fitted rotor bank tracks the real sky better than the register",
         samar["model_sigma"] < samar["total_sigma"]),
    ]
    for name, r in tests:
        print(f"    [{'PASS' if r else 'FAIL'}]  {name}")
    n_fail = sum(1 for _, r in tests if not r)
    print(f"\n    {len(tests) - n_fail}/{len(tests)} passed   |   "
          f"elapsed {time.time() - t0:.1f}s")

    banner(" VERDICT")
    print("  Three levers, three separate parts of one error budget, and a fourth")
    print("  quantity that no lever inside the system can reach at all.")
    print("   1. APERTURE bought precision and stopped dead at a pointing floor.")
    print("   2. PROCEDURE bought the tail and touched nothing else.")
    print("   3. Latitude and obliquity turned out to be ordinary parameters: a")
    print("      single instrument recovers them, because the residuals feel them.")
    print("   4. The ORIGIN of the scale is not like that. It sits in the exact")
    print("      null space of the objective. More marble does not reach it. More")
    print("      checking does not reach it. A second instrument does not reach it,")
    print("      because both instruments report on the scale whose zero is in")
    print("      question. Only an event that DEFINES the zero -- an equinox --")
    print("      removes the parameter the error was hiding in.")
    print("\n  A system cannot calibrate its own origin from inside itself.")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

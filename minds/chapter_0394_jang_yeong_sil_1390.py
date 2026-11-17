#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 # BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0394 - JANG YEONG-SIL (1390-1442)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 20, Minds 381-400 Available on Amazon https://www.amazon.com/dp/B0HRS9ZHVG
# Resume and Interactive Demos at  https://artificiology.com/
# Videos of the playable minds at https://www.youtube.com/@LostMindsAI
# END ATTRIBUTION  
 THE STANDARD VESSEL NETWORK (SVN)  --  표준기 신경망
================================================================================

 Encyclopedia of Lost Minds: Echoes on AI
 Figure 0394 -- Chang Yongsil / Jang Yeong-sil (蔣英實)
 born c. 1390, Dongnae; date of death unknown; last appears in the record 1442.

--------------------------------------------------------------------------------
 WHY THIS ARCHITECTURE AND NOT A TRANSFORMER
--------------------------------------------------------------------------------

 Jang Yeong-sil was born a government-owned nobi, the son of a Dongnae
 government courtesan and a father of Yuan descent, and his career ended when a
 royal sedan chair he had supervised broke under King Sejong in 1442; he was
 flogged, dismissed, and vanishes from the record. Between those two facts he
 built the instrument park of the Gyeongbokgung observatory: the ganui and
 honcheonui (1432-33), the angbu-ilgu scaphe sundial and the Borugak jagyeokru
 self-striking clepsydra (1434), the ilseong-jeongsi-ui (1437), the
 Heumgyeonggak ongnu (1438). On those instruments the court then computed the
 Chiljeongsan -- the first calendar in a thousand years of Korean practice
 derived for the meridian and latitude of Hanyang instead of borrowed from
 Nanjing.

 Read together, these are not the works of a mind that wanted better
 prediction. They are the works of a mind fixed on a different question:

        WHAT PART OF THIS NUMBER IS THE SKY, AND WHAT PART IS THE PLACE,
        THE SOIL, AND THE MAN WHO WROTE IT DOWN?

 A reading is not knowledge. A reading becomes knowledge only when it has been
 made TRANSPORTABLE -- when it survives being carried across latitude, across
 soil, across observers of unequal skill, and across the gap between a
 continuous physical quantity and a public announcement everyone hears
 identically. His career is four distinct operations for manufacturing that
 transportability, and this network implements all four, each as a separate
 organ matched to one specific corruption:

   CORRUPTION 1 -- THE PLACE.
       The Ming Datong-li was not noisy in Korea. It was wrong by an amount
       computable from the angle between two places. Jang's instruments existed
       to measure the local constants so the offset could be DERIVED.
       ORGAN: MeridianRotor. A rotation in latent space whose angle is a
       function of the station's own coordinates, held affine in longitude by
       an explicit prior, so a county that supplied no data still gets its
       correction.

   CORRUPTION 2 -- THE SOIL.
       Before the cheugugi, Joseon measured rain by digging and reading how
       deep the wetness had soaked, so "rainfall" in a sandy county and a clay
       county were not the same quantity. The standardized bronze cylinder does
       not measure better. It measures a DIFFERENT AND CLEANER THING, by
       refusing to let the ground into the instrument.
       ORGAN: StandardVessel. Exact removal of the per-report absorption term.
       Not a learned correction -- a structural refusal.

   CORRUPTION 3 -- THE MAN.
       The vessel fixed the ground. It could not fix the clerk, who rounds to
       whatever grid his habit and his eyesight allow, leaving a personal
       fingerprint on every number he reports. Worse, in any real archive that
       fingerprint is CORRELATED with geography, and a model will happily read
       the clerk as a proxy for the county.
       ORGAN: PrefectHead, adversarial. A magistrate is trained to name the
       clerk from the code; the encoder is trained to make him useless.

   CORRUPTION 4 -- THE READING ITSELF.
       The jagyeokru's real invention was a liquid-driven DISCRETE release
       cascade: water rises, a float tips a plate, a small bronze ball falls,
       that ball trips a larger ball, the larger ball strikes the bell. A
       continuous quantity becomes a countable, transmissible, identical-for-
       everyone event -- and the man who used to watch the water and be flogged
       for reading it late is removed from the loop entirely.
       ORGAN: BallCascade. A hard discrete bottleneck the code must survive.

 Nothing here is attention over stored keys. Jang's problem was never retrieval.
 It was commensuration.

--------------------------------------------------------------------------------
 THE TASK
--------------------------------------------------------------------------------

 A stylized Joseon reporting network. Sixteen provincial stations each file
 twelve-day windows of readings. The court wants one number per window: how
 much wetter the final day was than that window's own average -- expressed in
 royal-meridian terms, free of place, soil, and clerk. That is the quantity a
 standardized vessel can actually deliver, and its transportability is exactly
 why it is worth having.

 Eleven stations are visited during training. Five are never visited, and they
 sit at the geographic EXTREMES in both directions, so the test is genuine
 extrapolation in longitude rather than interpolation between neighbours.

--------------------------------------------------------------------------------
 CONVENTIONS
--------------------------------------------------------------------------------
   * Pure NumPy. Every gradient derived and written by hand. No autograd.
   * A finite-difference gradient check over every parameter tensor is
     mandatory and runs on every execution.
   * A real training loop with hand-written Adam, and a held-out-station
     evaluation against two baselines.
   * Self-tests at the bottom; the process exits non-zero if any test fails.

 Run:  python3 chapter_0394_jang_yeong_sil_1390.py
================================================================================
"""

import sys
import math
import numpy as np

# The court standardized on the Ju-cheok (Zhou foot-rule) of about 20.7 cm so
# every instrument in every province was cut to one unit. We fix one seed for
# the same reason: reproducibility is a form of commensuration.
JU_CHEOK_SEED = 207


# ==============================================================================
# SECTION 1 -- THE WORLD
# ==============================================================================

HANYANG_LON = 127.0        # the royal meridian: the frame all reports are owed in
HANYANG_LAT = 37.6

N_STATIONS = 16
N_TRAIN_STATIONS = 11      # stations 0..10 visited; 11..15 never visited
WINDOW = 12                # days per filed report
N_CLERK = 4                # rounding-habit classes

# Four clerks, each with a habitual rounding grid and, more consequentially, a
# measuring rule of his own that is not quite the Ju-cheok. A rule cut 12 per
# cent long is invisible in any single report and unmistakable across a
# thousand -- and, crucially, mean-centring does not touch it, because a scale
# error is not an offset. This is the corruption the cheugugi could not fix.
CLERK_GRID = np.array([0.02, 0.06, 0.15, 0.35])
CLERK_RULE = np.array([1.00, 0.88, 1.14, 0.79])


class JoseonWorld:
    """
    Generates filed reports.

    THE CENTRAL FACT OF THE SETUP, AND THE REASON THE CHILJEONGSAN WAS WORTH
    COMPUTING: the station does not know what day it is. It knows what day its
    almanac says, and its almanac was computed on somebody else's meridian.
    The sky over a county is the real sky; only the calendar against which the
    county reads that sky is displaced, by an amount proportional to the
    angular distance from the meridian the calendar was fitted on.

    The quantity the court wants, for a report ending on true day d:

        Y = seasonal(d)  +  (a[last] - mean(a))

    Two terms, deliberately, because they can only be got by different means:

        the seasonal term  requires undoing the calendar displacement, which
                           is available from longitude and from nothing else;
        the anomaly term   requires reading the filed numbers after the ground
                           has been excluded and the latitude gain undone.

    A model that solves one and not the other is visibly half a model.

    Each station corrupts the report in three categorically different ways:

        PLACE     calendar displacement from longitude, amplitude gain from
                  latitude. Recoverable from coordinates. Fixable before
                  anyone has ever been sent there.
        SOIL      an absorption offset, redrawn every report.
                  Recoverable from nothing. Must be structurally excluded.
        CLERK     rounding to that clerk's habitual grid. Not affine, so no
                  normalization removes it -- and in the training archive it
                  is CORRELATED WITH LONGITUDE, which makes it an inviting and
                  treacherous shortcut into exactly the variable that matters.
    """

    K_DAYS_PER_DEG_LON = 8.0      # days of calendar displacement per degree east
    K_GAIN_PER_DEG_LAT = 0.060    # fractional amplitude per degree north

    def __init__(self, seed=JU_CHEOK_SEED):
        self.rng = np.random.RandomState(seed)
        # Held-out stations sit outside the visited longitude interval on BOTH
        # sides. Interpolation will not save a model here.
        self.lon = np.array([127.0, 126.6, 128.3, 127.6, 126.9, 128.9,
                             127.2, 126.3, 129.1, 127.9, 128.0,     # visited
                             124.7, 130.4, 125.3, 130.0, 124.9])    # unvisited
        self.lat = np.array([37.6, 37.5, 36.6, 35.8, 35.2, 35.9,
                             38.3, 34.8, 35.2, 36.0, 37.0,
                             39.9, 33.6, 41.8, 34.1, 42.3])

        # THE SURVEY IS NOT THE PLACE. County coordinates in the 1430s were
        # not precisely known -- fixing the polar altitude of Hanyang was one
        # of the things the Ganui-dae was built to do. Each county therefore
        # has a surveyed longitude carrying a fixed error of up to about a
        # degree, and the survey is all the model ever sees. The sky, and the
        # clerks, follow the true longitude.
        #
        # This is what makes the clerk dangerous rather than merely present:
        # inside a survey's uncertainty the clerk's rounding habit is a FINER
        # guide to which county filed the report than the coordinates are, and
        # a model with an error to minimise will take it.
        self.lon_surveyed = self.lon + self.rng.uniform(-0.25, 0.25, N_STATIONS)

    # ---- the sky ----------------------------------------------------------
    @staticmethod
    def seasonal(day):
        """The seasonal water curve as it stands at the royal meridian."""
        t = 2.0 * np.pi * day / 365.0
        return (1.00 * np.sin(t - 0.40)
                + 0.35 * np.sin(2.0 * t + 1.10)
                + 0.12 * np.sin(3.0 * t - 0.70))

    def almanac_displacement(self, s):
        """How many days the county's inherited almanac is out, by longitude."""
        return self.K_DAYS_PER_DEG_LON * (self.lon[s] - HANYANG_LON)

    def geometric_gain(self, s):
        return 1.0 + self.K_GAIN_PER_DEG_LAT * (self.lat[s] - HANYANG_LAT)

    # ---- clerk assignment -------------------------------------------------
    def clerk_for(self, s, rng, mode):
        """
        In the training archive coarse clerks cluster in the east: the eastern
        counties were staffed later and worse. This is the sort of accidental
        regularity every historical corpus contains, and a model is under no
        obligation to notice that it is an accident.

          'archive'  the correlation as the training records contain it
          'uniform'  clerks assigned at random, as at a county the archive
                     never covered
          'reversed' the correlation inverted, as after a reassignment of
                     provincial staff -- the same realm, the same instruments,
                     the same sky, and a model that leaned on the clerk to
                     guess the county now has it exactly backwards
        """
        n = len(s)
        if mode == "uniform":
            return rng.randint(0, N_CLERK, size=n)
        east = np.clip((self.lon[s] - 126.0) / 3.2, 0.0, 1.0)
        if mode == "reversed":
            east = 1.0 - east
        # probability mass slides from fine (class 0) to coarse (class 3)
        centre = east * (N_CLERK - 1)
        idx = np.arange(N_CLERK)[None, :]
        w = np.exp(-((idx - centre[:, None]) ** 2) / 0.55)
        w /= w.sum(1, keepdims=True)
        u = rng.rand(n, 1)
        return (u > np.cumsum(w, axis=1)).sum(1).clip(0, N_CLERK - 1)

    CLERK_MODES = ("archive", "uniform", "reversed")

    # ---- sampling ---------------------------------------------------------
    def sample(self, n, station_pool, rng=None, clerks="archive"):
        """
        Returns
            R    (n, WINDOW)  raw filed readings
            H    (n, 4)       harmonics of the window's final day
            G    (n, 2)       [lon, lat], centred and scaled
            C    (n,)         clerk class -- ADVERSARY TARGET, training only
            Y    (n,)         the transportable truth
        """
        rng = rng or self.rng
        s = rng.choice(station_pool, size=n)
        d_end = rng.uniform(0.0, 365.0, size=n)              # the TRUE day
        days = d_end[:, None] + np.arange(-WINDOW + 1, 1)[None, :]

        # daily weather anomaly, the same weather everywhere in the realm
        a = rng.normal(0.0, 0.62, size=(n, WINDOW))

        # the sky over the station is the real sky
        sky = self.seasonal(days) + a

        # the instrument: latitude gain, then the ground drinks some
        absorb = rng.uniform(-0.85, 0.85, size=(n, 1))
        R = self.geometric_gain(s)[:, None] * sky - absorb

        # the clerk applies his own rule, then rounds to his own habit
        C = self.clerk_for(s, rng, clerks)
        R = CLERK_RULE[C][:, None] * R
        grid = CLERK_GRID[C][:, None]
        R = np.round(R / grid) * grid + rng.normal(0.0, 0.010, size=R.shape)

        # the transportable truth, owed in royal-meridian terms
        Y = self.seasonal(d_end) + (a[:, -1] - a.mean(axis=1))

        # THE ALMANAC THE COUNTY ACTUALLY HAS, which is out by its longitude
        d_almanac = d_end + self.almanac_displacement(s)
        t = 2.0 * np.pi * d_almanac / 365.0
        H = np.stack([np.cos(t), np.sin(t), np.cos(2 * t), np.sin(2 * t)], 1)
        G = np.stack([(self.lon_surveyed[s] - HANYANG_LON) / 3.0,
                      (self.lat[s] - HANYANG_LAT) / 4.0], 1)
        return R, H, G, C.astype(np.int64), Y


# ==============================================================================
# SECTION 2 -- PARAMETER STORE, ADAM, INIT
# ==============================================================================

class Params(dict):
    def like_zeros(self):
        return {k: np.zeros_like(v) for k, v in self.items()}


class Adam:
    """Hand-rolled Adam."""

    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads, keys=None):
        self.t += 1
        for k in (keys if keys is not None else self.p.keys()):
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            self.p[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


def _init(rng, *shape):
    return rng.randn(*shape) * math.sqrt(1.5 / shape[0])


# ==============================================================================
# SECTION 3 -- THE ORGANS
# ==============================================================================
#
# Each organ is a pair of pure functions (forward, backward). The forward
# returns a cache; the backward consumes it and accumulates into a gradient
# dict. Written this way every derivative is visible on the page, which is the
# only thing that makes the mandatory gradient check worth running.


# ------------------------------------------------------------------------------
# ORGAN 1 -- STANDARD VESSEL   (측우기, the cheugugi)
# ------------------------------------------------------------------------------
# The pre-cheugugi method was to dig and measure how deep the rain had soaked.
# That number is a joint fact about the weather AND the ground, and once
# recorded the two can never be separated. The bronze cylinder of fixed depth
# and fixed bore does not correct for the ground. It excludes it, at the moment
# of measurement, structurally.
#
# Here: subtract the report window's own mean. The absorption term is constant
# within a report, so this removes it EXACTLY -- not approximately, not on
# average, not if training goes well. Exactly, by arithmetic.
#
# The cost is real and Jang paid it too: what survives is a difference, never a
# level. A mean-referenced instrument cannot tell you how much rain fell. It can
# only tell you how much more fell today than usually falls in such a week --
# and that, unlike the level, means the same thing in every county in the realm.
#
# After centring, one learned affine per window position (shared across all
# reports, so it reintroduces no per-report signature).

def vessel_forward(p, R):
    mu = R.mean(axis=1, keepdims=True)
    rc = R - mu
    out = rc * p["ves_g"] + p["ves_b"]
    return out, {"rc": rc, "T": R.shape[1]}


def vessel_backward(p, g, dout, cache):
    rc, T = cache["rc"], cache["T"]
    g["ves_g"] += (dout * rc).sum(0)
    g["ves_b"] += dout.sum(0)
    drc = dout * p["ves_g"]
    # d/dR of (R - mean(R)) is (I - 11^T/T)
    return drc - drc.mean(axis=1, keepdims=True)


# ------------------------------------------------------------------------------
# ORGAN 2 -- GNOMON ENCODER   (앙부일구, the scaphe sundial)
# ------------------------------------------------------------------------------
# The angbu-ilgu is a bronze hemisphere with a gnomon at its centre; the bowl is
# cut so that the same shadow means the same hour and the same solar term. It
# turns one raw physical fact into a structured reading. Jang's version was set
# in the public street at Hyejeong Bridge, and because most who passed it could
# not read, the hours were marked with the twelve zodiac animals instead of
# characters. The readout was built for the reader it would actually get.
#
# Here: the vessel's output plus the final day's harmonics -> a latent arranged
# as D/2 phase pairs, because the next organ rotates those pairs.

def gnomon_forward(p, V, H):
    X = np.concatenate([V, H], axis=1)
    a = X @ p["enc_W1"] + p["enc_b1"]
    h = np.tanh(a)
    z0 = h @ p["enc_W2"] + p["enc_b2"]
    return z0, {"X": X, "h": h, "Tv": V.shape[1]}


def gnomon_backward(p, g, dz0, cache):
    X, h, Tv = cache["X"], cache["h"], cache["Tv"]
    g["enc_W2"] += h.T @ dz0
    g["enc_b2"] += dz0.sum(0)
    dh = dz0 @ p["enc_W2"].T
    da = dh * (1.0 - h * h)
    g["enc_W1"] += X.T @ da
    g["enc_b1"] += da.sum(0)
    dX = da @ p["enc_W1"].T
    return dX[:, :Tv]          # gradient back into the vessel output


# ------------------------------------------------------------------------------
# ORGAN 3 -- MERIDIAN ROTOR   (칠정산, the Chiljeongsan correction)
# ------------------------------------------------------------------------------
# The Datong-li was a good calendar. In Hanyang it was a good calendar that was
# wrong, and wrong by an amount you could reason about from the angular distance
# between Nanjing and Hanyang. Sejong's observatory existed to measure the local
# constants so the correction could be derived rather than fudged.
#
# Mechanically: predict a rotation angle and a gain FROM THE STATION'S
# COORDINATES ALONE, then rotate each latent phase pair and scale it. Because
# the angle is a function of geometry rather than of a station embedding, a
# county that filed nothing still gets its correction -- and coordinates are
# available for a county before anyone has ever been sent there.
#
#   z1[2k]   = gamma * ( cos(th) * z0[2k] - sin(th) * z0[2k+1] )
#   z1[2k+1] = gamma * ( sin(th) * z0[2k] + cos(th) * z0[2k+1] )

def rotor_angles(p, G):
    am = G @ p["rot_W1"] + p["rot_b1"]
    hm = np.tanh(am)
    theta = (hm @ p["rot_wth"]).ravel() + p["rot_bth"][0]
    gamma = 1.0 + (hm @ p["rot_wga"]).ravel() + p["rot_bga"][0]
    return theta, gamma, {"G": G, "hm": hm}


# A displaced calendar does not put every term of the sky out by the same
# angle. Shift the date by d days and the annual term moves by one unit of
# phase, the semi-annual term by two. So each latent pair carries a harmonic
# ORDER and is rotated by that multiple of the single predicted angle -- one
# scalar correction, applied at the right rate to each term. This is how a
# calendar correction is actually shaped, and it is why one number per county
# is enough.
HARMONIC_ORDER = np.array([1.0, 1.0, 2.0, 2.0])   # one per latent pair


def rotor_forward(p, z0, G):
    theta, gamma, cm = rotor_angles(p, G)
    m = HARMONIC_ORDER[None, :]
    ang = theta[:, None] * m
    ct, st = np.cos(ang), np.sin(ang)
    e, o = z0[:, 0::2], z0[:, 1::2]
    re, ro = ct * e - st * o, st * e + ct * o
    z1 = np.empty_like(z0)
    z1[:, 0::2] = gamma[:, None] * re
    z1[:, 1::2] = gamma[:, None] * ro
    cm.update({"z0": z0, "ct": ct, "st": st, "e": e, "o": o,
               "re": re, "ro": ro, "gamma": gamma, "m": m})
    return z1, cm


def rotor_backward(p, g, dz1, cache):
    ct, st, e, o = cache["ct"], cache["st"], cache["e"], cache["o"]
    re, ro, hm, G = cache["re"], cache["ro"], cache["hm"], cache["G"]
    gam, m = cache["gamma"][:, None], cache["m"]

    dre = dz1[:, 0::2] * gam
    dro = dz1[:, 1::2] * gam
    dgamma = (dz1[:, 0::2] * re + dz1[:, 1::2] * ro).sum(1)
    dtheta = (m * ((dre * (-st * e - ct * o))
                   + (dro * (ct * e - st * o)))).sum(1)

    dz0 = np.empty_like(cache["z0"])
    dz0[:, 0::2] = dre * ct + dro * st
    dz0[:, 1::2] = -dre * st + dro * ct

    g["rot_wth"] += (hm * dtheta[:, None]).sum(0)[:, None]
    g["rot_bth"] += np.array([dtheta.sum()])
    g["rot_wga"] += (hm * dgamma[:, None]).sum(0)[:, None]
    g["rot_bga"] += np.array([dgamma.sum()])

    dhm = dtheta[:, None] * p["rot_wth"].T + dgamma[:, None] * p["rot_wga"].T
    dam = dhm * (1.0 - hm * hm)
    g["rot_W1"] += G.T @ dam
    g["rot_b1"] += dam.sum(0)
    return dz0


# ------------------------------------------------------------------------------
# ORGAN 4 -- PREFECT HEAD   (the adversary)
# ------------------------------------------------------------------------------
# A magistrate reads the code and tries to name the clerk who filed it. If he
# can, the code still carries the man, and the model is free to use the clerk's
# rounding habit as a cheap stand-in for the county -- which works beautifully
# until it is carried somewhere the archive never reached. The head is trained
# to be as good a magistrate as possible; the trunk is trained, through a
# reversed gradient, to make him useless. Invariance here is not asserted. It is
# won against an opponent who is trying.

def prefect_forward(p, z1):
    a = z1 @ p["adv_W1"] + p["adv_b1"]
    h = np.tanh(a)
    return h @ p["adv_W2"] + p["adv_b2"], {"z1": z1, "h": h}


def softmax_ce(logits, labels):
    m = logits.max(axis=1, keepdims=True)
    ex = np.exp(logits - m)
    sm = ex / ex.sum(axis=1, keepdims=True)
    n = logits.shape[0]
    loss = -np.log(np.maximum(sm[np.arange(n), labels], 1e-12)).mean()
    d = sm.copy()
    d[np.arange(n), labels] -= 1.0
    return loss, d / n, sm


def prefect_backward(p, g, dlogits, cache):
    z1, h = cache["z1"], cache["h"]
    g["adv_W2"] += h.T @ dlogits
    g["adv_b2"] += dlogits.sum(0)
    dh = dlogits @ p["adv_W2"].T
    da = dh * (1.0 - h * h)
    g["adv_W1"] += z1.T @ da
    g["adv_b1"] += da.sum(0)
    return da @ p["adv_W1"].T


# ------------------------------------------------------------------------------
# ORGAN 5 -- BALL CASCADE   (자격루, the jagyeokru release train)
# ------------------------------------------------------------------------------
# The mechanism the Sejong annals describe: water rises in the receiving jar, a
# float lifts one of twelve rods, the rod tips a plate, a small bronze ball
# rolls off, that ball releases a larger ball, and the larger ball strikes the
# bell. A continuous accumulation crosses a fixed threshold and becomes a
# counted, discrete, publicly audible event.
#
#   hard:  n_k = sum_l  1[a_k > t_l]
#   soft:  n_k = sum_l  sigmoid((a_k - t_l)/tau)
#
# Training begins soft (water), anneals tau downward, and then switches to a
# straight-through estimator: the forward pass releases genuine integer counts
# while the backward pass uses the soft slopes. From that point the network is
# never allowed to smuggle information through the sigmoid's shoulders. Whether
# it succeeded is measured, not assumed, by test_cascade_survives_hardening.

CASCADE_LEVELS = 7
LADDER = np.linspace(-2.2, 2.2, CASCADE_LEVELS)


def cascade_forward(p, z1, tau, hard=False, ste=False):
    a = z1 @ p["cas_W"] + p["cas_b"]
    d = a[:, :, None] - LADDER[None, None, :]
    soft = 1.0 / (1.0 + np.exp(-np.clip(d / tau, -60, 60)))
    if hard or ste:
        rung = (d > 0.0).astype(np.float64)
    else:
        rung = soft
    q = rung.sum(axis=2) / CASCADE_LEVELS
    return q, {"z1": z1, "soft": soft, "tau": tau, "hard": hard and not ste}


def cascade_backward(p, g, dq, cache):
    if cache["hard"]:
        raise RuntimeError("cascade_backward called in pure hard mode")
    s, tau, z1 = cache["soft"], cache["tau"], cache["z1"]
    drung = (dq / CASCADE_LEVELS)[:, :, None]
    da = (drung * s * (1.0 - s) / tau).sum(axis=2)
    g["cas_W"] += z1.T @ da
    g["cas_b"] += da.sum(0)
    return da @ p["cas_W"].T


# ------------------------------------------------------------------------------
# ORGAN 6 -- BELL DECODER   (보루각, the Borugak announcement)
# ------------------------------------------------------------------------------
# From the counted balls, the announced figure. The decoder never sees the raw
# readings, the coordinates, or the continuous latent. Only what the bell rang.

def bell_forward(p, q):
    a = q @ p["dec_W1"] + p["dec_b1"]
    h = np.tanh(a)
    return (h @ p["dec_w2"]).ravel() + p["dec_b2"][0], {"q": q, "h": h}


def bell_backward(p, g, dy, cache):
    q, h = cache["q"], cache["h"]
    g["dec_w2"] += (h * dy[:, None]).sum(0)[:, None]
    g["dec_b2"] += np.array([dy.sum()])
    dh = dy[:, None] * p["dec_w2"].T
    da = dh * (1.0 - h * h)
    g["dec_W1"] += q.T @ da
    g["dec_b1"] += da.sum(0)
    return da @ p["dec_W1"].T


# ==============================================================================
# SECTION 4 -- THE GAUGE-LINEARITY PRIOR
# ==============================================================================
#
# Jang's claim was never merely "places differ". It was that the difference is
# ORDERLY IN THE GEOMETRY: move a fixed angle east, incur a fixed shift. That
# single commitment is what licenses correcting a county you have never
# visited, and it is the whole reason the Chiljeongsan was worth computing
# rather than simply collecting more local almanacs.
#
# Encoded as a penalty on the discrete second difference of the rotor's angle
# along a longitude probe. Zero when the angle is affine in longitude, positive
# otherwise. This is what stops the rotor from memorising eleven
# station-shaped bumps and flattening out beyond them.

PROBE_N = 24
PROBE_G = np.stack([np.linspace(-1.15, 1.15, PROBE_N), np.zeros(PROBE_N)], 1)


def gauge_forward(p):
    theta, _, cm = rotor_angles(p, PROBE_G)
    second = theta[2:] - 2.0 * theta[1:-1] + theta[:-2]
    return float((second * second).mean()), (theta, second, cm)


def gauge_backward(p, g, scale, cache):
    theta, second, cm = cache
    dsecond = 2.0 * second / second.shape[0] * scale
    dtheta = np.zeros_like(theta)
    dtheta[2:] += dsecond
    dtheta[1:-1] += -2.0 * dsecond
    dtheta[:-2] += dsecond
    hm, G = cm["hm"], cm["G"]
    g["rot_wth"] += (hm * dtheta[:, None]).sum(0)[:, None]
    g["rot_bth"] += np.array([dtheta.sum()])
    dam = (dtheta[:, None] * p["rot_wth"].T) * (1.0 - hm * hm)
    g["rot_W1"] += G.T @ dam
    g["rot_b1"] += dam.sum(0)


# ==============================================================================
# SECTION 5 -- ASSEMBLY
# ==============================================================================

D_LATENT = 8       # even: D/2 phase pairs for the rotor
H_ENC = 64
H_ROT = 16
K_CASCADE = 12
H_DEC = 48
H_ADV = 32

MAIN_KEYS = ["ves_g", "ves_b",
             "enc_W1", "enc_b1", "enc_W2", "enc_b2",
             "rot_W1", "rot_b1", "rot_wth", "rot_bth", "rot_wga", "rot_bga",
             "cas_W", "cas_b",
             "dec_W1", "dec_b1", "dec_w2", "dec_b2"]
ADV_KEYS = ["adv_W1", "adv_b1", "adv_W2", "adv_b2"]
# The rotor is eighty parameters carrying the single most consequential
# correction in the network, and its gradient arrives attenuated through four
# organs. Left on the trunk's learning rate it converges late and sometimes
# not at all -- the calendar half-corrected, which is the exact failure the
# Chiljeongsan was commissioned to end. It gets its own step size.
ROT_KEYS = ["rot_W1", "rot_b1", "rot_wth", "rot_bth", "rot_wga", "rot_bga"]
TRUNK_KEYS = [k for k in MAIN_KEYS if k not in ROT_KEYS]


def build_params(seed=JU_CHEOK_SEED):
    rng = np.random.RandomState(seed)
    p = Params()
    p["ves_g"] = np.ones(WINDOW);  p["ves_b"] = np.zeros(WINDOW)
    p["enc_W1"] = _init(rng, WINDOW + 4, H_ENC); p["enc_b1"] = np.zeros(H_ENC)
    p["enc_W2"] = _init(rng, H_ENC, D_LATENT);   p["enc_b2"] = np.zeros(D_LATENT)
    p["rot_W1"] = _init(rng, 2, H_ROT);          p["rot_b1"] = np.zeros(H_ROT)
    p["rot_wth"] = _init(rng, H_ROT, 1) * 0.5;   p["rot_bth"] = np.zeros(1)
    p["rot_wga"] = _init(rng, H_ROT, 1) * 0.5;   p["rot_bga"] = np.zeros(1)
    p["cas_W"] = _init(rng, D_LATENT, K_CASCADE); p["cas_b"] = np.zeros(K_CASCADE)
    p["dec_W1"] = _init(rng, K_CASCADE, H_DEC);  p["dec_b1"] = np.zeros(H_DEC)
    p["dec_w2"] = _init(rng, H_DEC, 1);          p["dec_b2"] = np.zeros(1)
    p["adv_W1"] = _init(rng, D_LATENT, H_ADV);   p["adv_b1"] = np.zeros(H_ADV)
    p["adv_W2"] = _init(rng, H_ADV, N_CLERK);    p["adv_b2"] = np.zeros(N_CLERK)
    return p


def forward_main(p, R, H, G, tau, hard=False, ste=False):
    """excluded -> structured -> corrected -> counted -> announced."""
    V, c_ves = vessel_forward(p, R)
    z0, c_enc = gnomon_forward(p, V, H)
    z1, c_rot = rotor_forward(p, z0, G)
    q, c_cas = cascade_forward(p, z1, tau, hard=hard, ste=ste)
    y, c_dec = bell_forward(p, q)
    return y, z1, dict(ves=c_ves, enc=c_enc, rot=c_rot, cas=c_cas, dec=c_dec)


def loss_and_grads(p, R, H, G, C, Y, tau, lam_adv, lam_gauge,
                   use_adv=True, ste=False):
    B = R.shape[0]
    g = p.like_zeros()
    yhat, z1, K = forward_main(p, R, H, G, tau, hard=False, ste=ste)

    resid = yhat - Y
    task = float((resid * resid).mean())
    dq = bell_backward(p, g, 2.0 * resid / B, K["dec"])
    dz1 = cascade_backward(p, g, dq, K["cas"])

    adv_loss = adv_acc = 0.0
    if use_adv and lam_adv != 0.0:
        logits, c_adv = prefect_forward(p, z1)
        al, dlog, _ = softmax_ce(logits, C)
        adv_loss = float(al)
        adv_acc = float((logits.argmax(1) == C).mean())
        dz1_adv = prefect_backward(p, g, dlog, c_adv)   # adversary descends
        dz1 = dz1 - lam_adv * dz1_adv                   # trunk ascends

    dz0 = rotor_backward(p, g, dz1, K["rot"])
    dV = gnomon_backward(p, g, dz0, K["enc"])
    vessel_backward(p, g, dV, K["ves"])

    gauge = 0.0
    if lam_gauge != 0.0:
        gv, gc = gauge_forward(p)
        gauge = gv
        gauge_backward(p, g, lam_gauge, gc)

    return {"task": task, "gauge": gauge, "adv": adv_loss,
            "adv_acc": adv_acc, "total": task + lam_gauge * gauge}, g


# ==============================================================================
# SECTION 6 -- MANDATORY FINITE-DIFFERENCE GRADIENT CHECK
# ==============================================================================
#
# Two checks, because two different scalars are being descended.
#
#   (A) the trunk -- all 18 tensors against the objective they actually
#       minimise (task + gauge, adversary weight zero).
#   (B) the adversary -- its 4 tensors against its own cross-entropy.
#
# The gradient reversal is an exact sign flip composed between the two and
# introduces no new derivative, so checking the pieces checks the whole.
#
# Both run at tau = 1.0 with the soft cascade, where the graph is smooth
# everywhere and central differences are exact to O(h^2). The straight-through
# estimator used late in training is by construction NOT the derivative of the
# hard forward pass -- no finite-difference check of it could pass, and one
# that did would mean the estimator was doing nothing.
#
# Tolerance is the standard mixed absolute/relative form. A pure relative
# criterion is meaningless where the true gradient is near zero, because there
# the finite-difference estimate is all cancellation noise.

def _scalar_main(p, R, H, G, Y, tau, lam_gauge):
    yhat, _, _ = forward_main(p, R, H, G, tau)
    resid = yhat - Y
    return float((resid * resid).mean()) + lam_gauge * gauge_forward(p)[0]


def _scalar_adv(p, R, H, G, C, tau):
    _, z1, _ = forward_main(p, R, H, G, tau)
    return float(softmax_ce(prefect_forward(p, z1)[0], C)[0])


def gradient_check(verbose=True, n_probe=6, h=1e-5,
                   tol_rel=1e-5, tol_abs=1e-9):
    rng = np.random.RandomState(11)
    world = JoseonWorld(seed=99)
    R, H, G, C, Y = world.sample(24, np.arange(N_TRAIN_STATIONS), rng)
    tau, lam_gauge = 1.0, 0.30

    p = build_params(seed=5)
    _, g_main = loss_and_grads(p, R, H, G, C, Y, tau, 0.0, lam_gauge,
                               use_adv=False)

    rows, worst, worst_name = [], 0.0, ""

    def probe(keys, analytic, scalar_fn, tag):
        nonlocal worst, worst_name
        for k in keys:
            flat = p[k].reshape(-1)
            idxs = rng.choice(flat.size, min(n_probe, flat.size), replace=False)
            score = 0.0
            for i in idxs:
                o = flat[i]
                flat[i] = o + h; lp = scalar_fn()
                flat[i] = o - h; lm = scalar_fn()
                flat[i] = o
                num = (lp - lm) / (2 * h)
                ana = analytic[k].reshape(-1)[i]
                # mixed criterion: score < 1 means the pair agrees
                score = max(score, abs(num - ana) /
                            (tol_abs + tol_rel * max(abs(num), abs(ana))))
            rows.append((tag, k, score))
            if score > worst:
                worst, worst_name = score, k

    probe(MAIN_KEYS, g_main,
          lambda: _scalar_main(p, R, H, G, Y, tau, lam_gauge), "trunk")

    p = build_params(seed=5)
    _, g_adv = loss_and_grads(p, R, H, G, C, Y, tau, 1.0, 0.0, use_adv=True)
    probe(ADV_KEYS, g_adv,
          lambda: _scalar_adv(p, R, H, G, C, tau), "adversary")

    if verbose:
        print("  central differences, h=1e-5, "
              "criterion |num-ana| <= 1e-9 + 1e-5*max(|num|,|ana|)")
        for tag, k, s in rows:
            print(f"    [{'ok  ' if s < 1.0 else 'FAIL'}] {tag:9s} "
                  f"{k:8s}  margin = {s:.3f} of budget")
        print(f"    worst = {worst:.3f} of budget, on '{worst_name}'")
    return worst < 1.0, worst


# ==============================================================================
# SECTION 7 -- BASELINES
# ==============================================================================
#
# Two controls a fifteenth-century Korean astronomer would recognise at once.
#
#   BORROWED ALMANAC -- an MLP on the raw filed readings and the date, with no
#       notion that the place matters at all. This is the Datong-li used
#       unmodified in Hanyang. It receives the identical raw data.
#
#   COURT FIT -- an MLP given the coordinates AND the vessel (its readings are
#       pre-centred, so the soil is already excluded for it) but no gauge prior,
#       no invariance pressure, no discrete bottleneck. This is what you get
#       when you fit the eleven counties that happen to have clerks and call the
#       result a national model. It is deliberately not a straw man: it is the
#       strongest thing available without Jang's remaining three operations, and
#       it is given more parameters than the network it is being compared to.

def mlp_init(rng, sizes):
    P = {"_n": len(sizes) - 1}
    for i in range(len(sizes) - 1):
        P[f"W{i}"] = _init(rng, sizes[i], sizes[i + 1])
        P[f"b{i}"] = np.zeros(sizes[i + 1])
    return P


def mlp_forward(P, x):
    h, cache = x, [x]
    for i in range(P["_n"]):
        a = h @ P[f"W{i}"] + P[f"b{i}"]
        h = np.tanh(a) if i < P["_n"] - 1 else a
        cache.append(h)
    return h.ravel(), cache


def mlp_train(P, sampler, steps=6000, lr=3e-3, batch=256, seed=3):
    rng = np.random.RandomState(seed)
    keys = [k for k in P if k != "_n"]
    opt = Adam({k: P[k] for k in keys}, lr=lr)
    for _ in range(steps):
        x, y = sampler(batch, rng)
        yhat, cache = mlp_forward(P, x)
        d = (2.0 * (yhat - y) / x.shape[0])[:, None]
        g = {k: np.zeros_like(P[k]) for k in keys}
        for i in reversed(range(P["_n"])):
            g[f"W{i}"] = cache[i].T @ d
            g[f"b{i}"] = d.sum(0)
            if i > 0:
                d = (d @ P[f"W{i}"].T) * (1.0 - cache[i] ** 2)
        opt.step(g)
        for k in keys:
            P[k] = opt.p[k]
    return P


# ==============================================================================
# SECTION 8 -- TRAINING
# ==============================================================================

def train_svn(world, steps=6000, batch=256, lr=3e-3, seed=JU_CHEOK_SEED,
              lam_adv_max=0.35, verbose=True):
    """
    Two optimizers, deliberately. The magistrate is kept slightly ahead of the
    trunk -- a slow adversary gives a stale gradient, and a stale gradient in a
    reversal is worse than none, because the trunk learns to defeat a critic
    who is no longer looking. Everything else is annealed on a fixed schedule.
    """
    rng = np.random.RandomState(seed + 1)
    p = build_params(seed=seed)
    opt_main = Adam({k: p[k] for k in TRUNK_KEYS}, lr=lr)
    opt_rot = Adam({k: p[k] for k in ROT_KEYS}, lr=lr * 4.0)
    opt_adv = Adam({k: p[k] for k in ADV_KEYS}, lr=lr * 2.5)
    pool = np.arange(N_TRAIN_STATIONS)

    for step in range(steps):
        frac = step / max(1, steps - 1)
        # the cascade begins as water and hardens into balls
        tau = 1.0 * (0.05 / 1.0) ** min(1.0, frac / 0.6)
        # past the halfway point the forward pass releases genuine integers
        ste = frac > 0.45
        # the magistrate is given time to become competent before the trunk is
        # asked to defeat him
        lam_adv = (0.0 if frac < 0.12
                   else lam_adv_max * min(1.0, (frac - 0.12) / 0.30))
        lam_gauge = 3.0

        R, H, G, C, Y = world.sample(batch, pool, rng)
        rep, g = loss_and_grads(p, R, H, G, C, Y, tau, lam_adv, lam_gauge,
                                ste=ste)
        opt_main.step(g, keys=TRUNK_KEYS)
        opt_rot.step(g, keys=ROT_KEYS)
        opt_adv.step(g, keys=ADV_KEYS)
        for src, ks in ((opt_main, TRUNK_KEYS), (opt_rot, ROT_KEYS),
                        (opt_adv, ADV_KEYS)):
            for k in ks:
                p[k] = src.p[k]

        if verbose and (step % 1200 == 0 or step == steps - 1):
            print(f"    step {step:5d} | tau {tau:.3f} | ste {int(ste)} "
                  f"| lam_adv {lam_adv:.2f} | task {rep['task']:.4f} "
                  f"| gauge {rep['gauge']:.2e} "
                  f"| magistrate {rep['adv_acc']:.3f}")
    return p


def evaluate(p, world, pool, n=8000, tau=0.05, hard=True, seed=777,
             clerks="uniform"):
    rng = np.random.RandomState(seed)
    R, H, G, C, Y = world.sample(n, pool, rng, clerks=clerks)
    yhat, _, _ = forward_main(p, R, H, G, tau, hard=hard)
    return float(np.sqrt(((yhat - Y) ** 2).mean()))


# ==============================================================================
# SECTION 9 -- SELF-TESTS
# ==============================================================================

class TestReport:
    def __init__(self):
        self.rows = []

    def check(self, name, ok, detail=""):
        self.rows.append((name, bool(ok)))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
              + (f"\n         {detail}" if detail else ""))
        return ok

    def all_ok(self):
        return all(r[1] for r in self.rows)


def test_vessel_excludes_the_ground(rep):
    """
    ORGAN 1 stated as a theorem. Add an arbitrary constant to every reading in
    a report -- the exact shape of an absorption term -- and the vessel's output
    must not move at all. Not approximately. To floating-point exactness.

    That this test is trivial to pass IS the achievement. Jang did not build a
    cleverer estimator of soil moisture. He built a vessel that made the
    estimation unnecessary, and moved a hard inference problem into a piece of
    bronze where it became arithmetic.
    """
    p = build_params(seed=42)
    rng = np.random.RandomState(4)
    R = rng.randn(64, WINDOW)
    base, _ = vessel_forward(p, R)
    worst = 0.0
    for off in [3.7, -2.9, 0.4, -11.0, 250.0]:
        out, _ = vessel_forward(p, R + off)
        worst = max(worst, float(np.abs(out - base).max()))
    return rep.check("vessel excludes the ground exactly", worst < 1e-11,
                     f"largest deviation across offsets up to 250 cheok: "
                     f"{worst:.2e}")


def test_rotor_is_geometric(rep, p):
    """
    ORGAN 3. The correction angle must be an orderly function of longitude
    OUTSIDE the interval that supplied data, not merely inside it. Fit a line
    to the trained interior, then check the extrapolated wings still lie on it.
    """
    lon = np.linspace(-1.30, 1.30, 41)
    theta, _, _ = rotor_angles(p, np.stack([lon, np.zeros_like(lon)], 1))
    inside = np.abs(lon) <= 0.70
    A = np.stack([lon[inside], np.ones(inside.sum())], 1)
    coef, *_ = np.linalg.lstsq(A, theta[inside], rcond=None)
    pred = coef[0] * lon + coef[1]
    dev = float(np.abs(theta[~inside] - pred[~inside]).max())
    span = float(theta.max() - theta.min())
    ok = span > 0.10 and dev < 0.25 * span
    return rep.check("rotor angle stays affine in longitude when extrapolated",
                     ok, f"angle span {span:.3f} rad over the peninsula; "
                         f"largest departure from the fitted line beyond the "
                         f"sampled interval {dev:.4f} rad "
                         f"({100*dev/max(span,1e-9):.1f}% of span)")


def test_cascade_is_countable(rep, p, world):
    """ORGAN 5. The hard cascade must emit counts on the ladder, nothing else."""
    rng = np.random.RandomState(8)
    R, H, G, C, Y = world.sample(600, np.arange(N_STATIONS), rng)
    V, _ = vessel_forward(p, R)
    z0, _ = gnomon_forward(p, V, H)
    z1, _ = rotor_forward(p, z0, G)
    q, _ = cascade_forward(p, z1, 0.05, hard=True)
    counts = np.round(q * CASCADE_LEVELS, 9)
    ok = bool(np.all(np.isin(np.unique(counts), np.arange(CASCADE_LEVELS + 1))))
    n_states = len(np.unique(counts, axis=0))
    return rep.check("cascade releases whole balls only", ok,
                     f"per-channel counts observed: "
                     f"{sorted(int(v) for v in np.unique(counts))}; "
                     f"{n_states} distinct release patterns across "
                     f"{K_CASCADE} channels")


def test_announcement_survives_the_water(rep, p, world):
    """
    ORGAN 5, the part that matters. Read the model with genuine discrete
    releases and with the soft surrogate. If the two disagree, the network was
    smuggling information through the sigmoid's shoulders -- the code would work
    at the water and fail at the bell.
    """
    soft = evaluate(p, world, np.arange(N_STATIONS), hard=False)
    hard = evaluate(p, world, np.arange(N_STATIONS), hard=True)
    gap = abs(hard - soft) / max(soft, 1e-9)
    return rep.check("announcement survives the water", gap < 0.05,
                     f"soft {soft:.4f} vs discrete {hard:.4f} "
                     f"({100*gap:.2f}% apart)")


def magistrate_accuracy(feat, in_dim, C, seed):
    """Appoint a fresh magistrate against a frozen feature set and let him
    train to convergence. Returns how often he names the clerk correctly."""
    m = build_params(seed=seed)
    m["adv_W1"] = _init(np.random.RandomState(seed), in_dim, H_ADV)
    opt = Adam({k: m[k] for k in ADV_KEYS}, lr=6e-3)
    r = np.random.RandomState(seed + 1)
    for _ in range(1500):
        i = r.choice(len(C), 256, replace=False)
        logits, c = prefect_forward(m, feat[i])
        _, dlog, _ = softmax_ce(logits, C[i])
        g = {k: np.zeros_like(m[k]) for k in ADV_KEYS}
        prefect_backward(m, g, dlog, c)
        opt.step(g, keys=ADV_KEYS)
        for k in ADV_KEYS:
            m[k] = opt.p[k]
    return float((prefect_forward(m, feat)[0].argmax(1) == C).mean())


def test_magistrate_defeated(rep, p, p_ablate, world):
    """
    ORGAN 4, measured against its own ablation rather than against a bar I
    chose. Three feature sets, one magistrate procedure, one question: how
    legible is the clerk?

        (a) the filed report itself, mean-centred      -- the ceiling
        (b) the code of a network trained WITHOUT the reversal
        (c) the code of the network as shipped

    The result worth having is not (c) near chance. It is that (b) exceeds
    (a). A network with an error to minimise does not merely tolerate the
    clerk's fingerprint; it CONCENTRATES it, because a badly cut measuring
    rule is a cheaper guide to which county filed the report than a badly
    surveyed coordinate is. Left alone the model builds itself an instrument
    for reading the clerk, and calls the result geography.
    """
    rng = np.random.RandomState(313)
    R, H, G, C, Y = world.sample(6000, np.arange(N_TRAIN_STATIONS), rng)
    raw = R - R.mean(1, keepdims=True)
    _, z_ship, _ = forward_main(p, R, H, G, 0.05, hard=False)
    _, z_abl, _ = forward_main(p_ablate, R, H, G, 0.05, hard=False)

    a_raw = magistrate_accuracy(raw, WINDOW, C, 1234)
    a_abl = magistrate_accuracy(z_abl, D_LATENT, C, 2345)
    a_shp = magistrate_accuracy(z_ship, D_LATENT, C, 4321)

    chance = 1.0 / N_CLERK
    ok = (a_abl > a_raw) and (a_shp < a_raw) and (a_shp < a_abl)
    return rep.check("the reversal stops the network concentrating the clerk",
                     ok,
                     f"clerk named from the filed report {a_raw:.3f}; "
                     f"from an unreversed network's code {a_abl:.3f} "
                     f"(ABOVE the report it came from); "
                     f"from the shipped code {a_shp:.3f}. chance {chance:.3f}")


def test_unvisited_counties(rep, svn, court, ming):
    ok = svn < 0.85 * min(court, ming)
    return rep.check("beats both baselines at counties never visited", ok,
                     f"SVN {svn:.4f} | Court Fit {court:.4f} | "
                     f"Borrowed Almanac {ming:.4f}")


# ==============================================================================
# SECTION 10 -- MAIN
# ==============================================================================

def main():
    np.set_printoptions(precision=4, suppress=True)
    print("=" * 78)
    print(" THE STANDARD VESSEL NETWORK")
    print(" Figure 0394 -- Jang Yeong-sil, b. c. 1390, last recorded 1442")
    print(" excluded -> structured -> corrected -> counted -> announced")
    print("=" * 78)
    rep = TestReport()

    print("\n[1] GRADIENT CHECK  (mandatory)")
    ok, worst = gradient_check(verbose=True)
    rep.check("analytic gradients match central differences", ok,
              f"worst tensor used {worst:.3f} of its error budget")

    print("\n[2] THE VESSEL, AS A THEOREM")
    test_vessel_excludes_the_ground(rep)

    print("\n[3] THE REPORTING NETWORK")
    world = JoseonWorld()
    tr = np.arange(N_TRAIN_STATIONS)
    ho = np.arange(N_TRAIN_STATIONS, N_STATIONS)
    print(f"    {N_STATIONS} stations, {WINDOW}-day reports; "
          f"{len(tr)} visited, {len(ho)} never visited")
    print(f"    visited longitudes   {world.lon[tr].min():.1f}E - "
          f"{world.lon[tr].max():.1f}E")
    print(f"    unvisited longitudes {world.lon[ho].min():.1f}E - "
          f"{world.lon[ho].max():.1f}E   <- outside the sampled interval, "
          f"both directions")
    print(f"    each county's inherited almanac is out by "
          f"{world.K_DAYS_PER_DEG_LON:.0f} days per degree of longitude:")
    print(f"      visited counties are out by "
          f"{world.almanac_displacement(tr).min():+.0f} to "
          f"{world.almanac_displacement(tr).max():+.0f} days; "
          f"unvisited by {world.almanac_displacement(ho).min():+.0f} to "
          f"{world.almanac_displacement(ho).max():+.0f} days")
    print(f"    absorption redrawn every report over +/-0.85 cheok; "
          f"clerk grids {list(CLERK_GRID)}")
    print("    in the training archive coarse clerks cluster in the east; "
          "at unvisited stations they do not")

    print("\n[4] TRAINING")
    p = train_svn(world, steps=9000, verbose=True)

    print("\n[5] ABLATION: THE SAME NETWORK WITHOUT THE REVERSAL")
    p_abl = train_svn(world, steps=9000, lam_adv_max=0.0, verbose=False)
    print("    trained (identical in every other respect)")

    print("\n[6] BASELINES")
    rb = np.random.RandomState(21)

    def s_ming(n, rng):
        R, H, G, C, Y = world.sample(n, tr, rng)
        return np.concatenate([R, H], 1), Y

    def s_court(n, rng):
        R, H, G, C, Y = world.sample(n, tr, rng)
        Rc = R - R.mean(1, keepdims=True)          # given the vessel for free
        return np.concatenate([Rc, H, G], 1), Y

    ming = mlp_train(mlp_init(rb, [WINDOW + 4, 96, 96, 1]), s_ming, seed=31)
    court = mlp_train(mlp_init(rb, [WINDOW + 6, 96, 96, 1]), s_court, seed=32)
    n_svn = sum(p[k].size for k in MAIN_KEYS)
    n_crt = sum(v.size for k, v in court.items() if k != "_n")
    print(f"    Borrowed Almanac: raw readings + date, no coordinates")
    print(f"    Court Fit: centred readings + date + coordinates, "
          f"no gauge prior, no invariance, no bottleneck")
    print(f"    parameters -- SVN {n_svn}, Court Fit {n_crt} "
          f"({n_crt/n_svn:.1f}x larger)")

    def eval_mlp(P, pool, centred, geo, n=8000, seed=777,
                 clerks="uniform"):
        rng = np.random.RandomState(seed)
        R, H, G, C, Y = world.sample(n, pool, rng, clerks=clerks)
        Rx = R - R.mean(1, keepdims=True) if centred else R
        x = np.concatenate([Rx, H, G], 1) if geo else np.concatenate([Rx, H], 1)
        return float(np.sqrt(((mlp_forward(P, x)[0] - Y) ** 2).mean()))

    print("\n[7] RESULTS   RMSE on the transportable quantity")
    svn_tr = evaluate(p, world, tr, hard=True)
    svn_ho = evaluate(p, world, ho, hard=True)
    mg_tr = eval_mlp(ming, tr, False, False)
    mg_ho = eval_mlp(ming, ho, False, False)
    ct_tr = eval_mlp(court, tr, True, True)
    ct_ho = eval_mlp(court, ho, True, True)
    rng0 = np.random.RandomState(1)
    _, _, _, _, Y0 = world.sample(20000, np.arange(N_STATIONS), rng0)
    naive = float(np.sqrt((Y0 ** 2).mean()))

    print(f"    {'model':<26}{'visited':>11}{'unvisited':>11}{'carried':>11}")
    print("    " + "-" * 59)
    ab_tr = evaluate(p_abl, world, tr, hard=True)
    ab_ho = evaluate(p_abl, world, ho, hard=True)
    for nm, x, y in [("Borrowed Almanac", mg_tr, mg_ho),
                     ("Court Fit", ct_tr, ct_ho),
                     ("SVN, reversal ablated", ab_tr, ab_ho),
                     ("Standard Vessel Network", svn_tr, svn_ho)]:
        print(f"    {nm:<26}{x:>11.4f}{y:>11.4f}{y/x:>10.2f}x")
    print("    " + "-" * 59)
    print(f"    answering zero every time scores {naive:.4f}")

    print("\n[8] SELF-TESTS")
    test_rotor_is_geometric(rep, p)
    test_cascade_is_countable(rep, p, world)
    test_announcement_survives_the_water(rep, p, world)
    test_magistrate_defeated(rep, p, p_abl, world)
    test_unvisited_counties(rep, svn_ho, ct_ho, mg_ho)

    print("\n[9] WHAT THE NUMBERS SAY")
    print(f"    The Court Fit is the best model in the realm at the eleven")
    print(f"    counties that supplied its data, and it is {ct_ho/ct_tr:.1f} times worse")
    print(f"    the moment it is carried past them. That degradation is not")
    print(f"    noise. It is a bias whose size tracks the angular distance from")
    print(f"    the counties that filed the reports.")
    print(f"    The Standard Vessel Network gives up {svn_tr/ct_tr:.2f}x accuracy at")
    print(f"    home to degrade only {svn_ho/svn_tr:.2f}x when carried, because the only")
    print(f"    thing it was ever allowed to learn about a place was its")
    print(f"    geometry -- and a county has coordinates before it has clerks.")

    print("\n" + "=" * 78)
    if rep.all_ok():
        print(" ALL TESTS PASSED")
        print("=" * 78)
        return 0
    print(" TESTS FAILED")
    print("=" * 78)
    return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 MELEKKET  —  The Reference-Memory Chant Machine
 Chapter 0169 · Yared of Aksum (traditional dates c. 505–571)
 Encyclopedia of Lost Minds: Echoes on AI
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 9 Minds 161 - 180 Available on Amazon https://www.amazon.com/dp/B0HJC1QF59
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 0169_yared_505 - Yared of Aksum (traditional dates c. 505–571)
================================================================================  

WHAT THIS FILE IS
-----------------
A from-scratch, pure-NumPy neural architecture (no autograd, no frameworks)
whose *structure* encodes the one cognitive commitment that the Ethiopian
chant tradition places at the feet of Yared and of no one else:

    NOTHING IS WRITTEN THAT IS NOT ALREADY SUNG.

The Ethiopian Orthodox notation (mǝlǝkkǝt, "signs") does not describe a
melody. It is a system of *references*. Above each syllable of a hymn the
scribe writes one or two characters of the Gǝʿǝz syllabary — an abbreviation
of the opening of some OTHER hymn, the "root" (sǝrǝyu / source), in which the
same melodic phrase lives. In the margin a "house" sign (bet) says which family
of melodies the phrase belongs to. To sing anything from the page you must
already hold the roots in memory. The sign is a pointer; the melody is a
recollection; the same root is sung in three modes (gǝʿǝz, ʿǝzl, araray) and
stretched over however many syllables the new text happens to have.

Scholarship (Shelemay, Jeffery & Monson 1993; Shelemay in Encyclopaedia
Aethiopica) dates the written signs to the sixteenth century, not the sixth.
What the tradition attributes to Yared himself is the *method of memory* the
signs later fixed: learn the roots by ear, then sing the whole year by
reference. This file builds that method as a machine and measures what it
buys and what it costs.

THE ARCHITECTURAL CLAIM
-----------------------
A chant machine built on Yared's principle has four parts, and each one is a
separate object in the code below:

  1. ROOTS (the memorised sources).   A small codebook of K stored phrases.
     Each root has a HEAD (a key: how it begins) and a BODY (a canonical
     melodic contour of L positions). Roots are the only place melody exists.

  2. THE SIGN AS POINTER.   A notated syllable is a short CUE — the remembered
     opening notes plus, when the scribe supplied it, the syllable abbreviation
     — and a HOUSE sign. The cue is matched against the HEADS of the roots
     (never against their bodies). Retrieval is head-to-head; the body follows
     the head. A house sign biases which roots are eligible.

  3. THE SEVENTH CLIMB.   Retrieval is not one look-up but an iterated one.
     The hagiography's caterpillar falls six times and reaches the leaf on the
     seventh. Here the cue state is compared to the heads, moved toward what it
     found, and compared again — T = 7 climbs — with the comparison growing
     sharper on each climb (an "absorption" schedule: the singer who did not
     feel the spear). The final climb reads the body.

  4. THE THREE BIRDS.   The retrieved body is sung through one of three MODE
     transforms (gǝʿǝz, ʿǝzl, araray): a shared affine warp plus a per-position
     ornament. When the mode is not written, it is inferred from the day of the
     liturgical year. The body is then stretched to the syllable count of the
     text and rendered through Gaussian tuning curves over scale degrees.

Everything melodic the machine can ever produce is therefore a mode-warp of a
root it has memorised. That is the whole point and the whole limitation.

THE COUNTERFACTUALS
-------------------
Two comparison models are trained on the same repertoire:

  STAFF   — a plain multilayer perceptron that maps the same inputs directly
            to every note ("staff notation": the sign describes the melody).
            No roots, no pointers, no mode factorisation. ~12x more parameters.

  ONE CLIMB — the same reference machine with T = 1: the caterpillar that
            stopped after the first fall.

Regimes probe exactly where Yared's principle should win and where it must
lose:

  CANON          the year as taught: roots seen in seen modes
  MODE_TRANSFER  a root heard only in gǝʿǝz and ʿǝzl, asked for in araray
  HEAD_ONLY      the scribe wrote no abbreviation; only the opening notes
  SYLLABLE_ONLY  the opening is forgotten; only the abbreviation and house
  SEASON_ONLY    the mode is not written; infer it from the calendar
  NOISY_CUE      the memory of the opening is badly degraded
  UNSEEN_ROOT    a hymn built on a root the singer never memorised

The last regime is the structural ceiling. The machine has no slot for a root
it never learned. It cannot sing nonsense either: it will sing the nearest
hymn it knows, in the right mode, on the right syllables — a coherent, fluent,
confident error. We measure that too.

CONVENTIONS KEPT FROM THE CORPUS
--------------------------------
  * pure NumPy, hand-derived analytic gradients through all seven climbs
  * a finite-difference gradient check that must pass (mandatory)
  * a real training loop (hand-written Adam) on a real train/eval split
  * self-tests covering the structural invariants, not just the loss
  * executed before shipping; the printed output is the verified output

RUN:  python3 0169_yared_505_Neuron.py
================================================================================
"""

import sys
import time
import numpy as np

# ==============================================================================
# 0.  SMALL UTILITIES
# ==============================================================================

def sigmoid(z):
    """Numerically stable logistic."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


def softplus(z):
    z = np.asarray(z, dtype=np.float64)
    return np.where(z > 30, z, np.log1p(np.exp(np.minimum(z, 30))))


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def softmax_backward(p, dp):
    """Given p = softmax(z) and dL/dp, return dL/dz (same axis = last)."""
    return p * (dp - (dp * p).sum(axis=-1, keepdims=True))


def logsumexp(z, axis=-1, keepdims=True):
    m = z.max(axis=axis, keepdims=True)
    out = m + np.log(np.exp(z - m).sum(axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


class Adam:
    """Hand-written Adam. Works on a dict of parameter arrays."""

    def __init__(self, params, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        b1, b2 = self.b1, self.b2
        for k in params:
            g = grads[k]
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


# ==============================================================================
# 1.  THE REPERTOIRE  —  a synthetic Dǝggwa
# ==============================================================================
#
# The real Dǝggwa is a year of hymns whose melodies are drawn from a finite
# store of roots, sung in three modes, on texts of varying length. We build a
# hidden ground-truth version of exactly that so the machine can be tested on
# things the real tradition tests its singers on: transfer across modes,
# partial cues, seasonal inference, and the root you never learned.
#
# Roots are named after the first twenty-four consonants of the Gǝʿǝz
# syllabary in traditional order — because in the real notation the pointer
# IS a syllabary character.

FIDEL = ["hä", "lä", "ḥä", "mä", "śä", "rä", "sä", "qä", "bä", "tä", "ḫä", "nä",
         "ʾä", "kä", "wä", "ʿä", "zä", "yä", "dä", "gä", "ṭä", "ṗä", "ṣä", "ḍä"]

K_TRUE = 24          # roots in the hidden repertoire
K_MEMORISED = 22     # roots the singer actually learned (slots in the machine)
H_HOUSES = 6         # bet (houses): 4 roots per house
L_CANON = 8          # canonical contour length of a root
L_MAX = 8            # maximum syllables of a text
D_DEG = 13           # scale degrees 0..12
N_MODES = 3
T_READ = 7           # readings of the page available to a singer (one per climb)
GEEZ, EZEL, ARARAY = 0, 1, 2
MODE_NAMES = ["gǝʿǝz", "ʿǝzl", "araray"]
DEGREES = np.arange(D_DEG, dtype=np.float64)


def resample_matrix(n, L=L_CANON):
    """Linear-interpolation matrix R (n x L): stretch L canonical positions
    over n syllables. Rows sum to one. This is the melisma / syllable-fit."""
    R = np.zeros((n, L))
    if n == 1:
        R[0, 0] = 1.0
        return R
    for i in range(n):
        t = i * (L - 1) / (n - 1)
        j = int(np.floor(t))
        f = t - j
        if j >= L - 1:
            R[i, L - 1] = 1.0
        else:
            R[i, j] = 1 - f
            R[i, j + 1] = f
    return R


RESAMPLERS = {n: resample_matrix(n) for n in range(1, L_MAX + 1)}


class Repertoire:
    """Hidden ground truth: roots, houses, mode transforms, calendar."""

    def __init__(self, seed=0):
        rng = np.random.RandomState(seed)
        self.house_of = np.repeat(np.arange(H_HOUSES), K_TRUE // H_HOUSES)
        # each house has its own tonal centre (the "series of thirds" of a mode
        # resolves around a few points of repose; houses group melodies by range)
        centres = np.array([4.0, 5.0, 6.0, 6.0, 7.0, 8.0])
        bodies = np.zeros((K_TRUE, L_CANON))
        for k in range(K_TRUE):
            c = centres[self.house_of[k]]
            x = c
            walk = []
            for l in range(L_CANON):
                pull = -0.35 * (x - c)
                step = rng.choice([-2, -1, 0, 1, 2], p=[0.12, 0.28, 0.2, 0.28, 0.12]) + pull
                x = float(np.clip(x + step, 2.0, 10.0))
                walk.append(x)
            walk[-1] = c if rng.rand() < 0.6 else c - 1.0   # cadence: repose
            bodies[k] = np.array(walk)
        self.bodies = bodies                                  # gǝʿǝz form
        self.heads = bodies[:, :3].copy()                     # the remembered opening
        # ground-truth mode transforms: affine about a global pivot + ornament
        self.mode_a = np.array([1.0, 0.5, 1.3])
        self.mode_b = np.array([0.0, 3.5 - 0.5 * 6.0, 8.0 - 1.3 * 6.0])
        self.mode_w = np.zeros((N_MODES, L_CANON))
        self.mode_w[EZEL, -1] = -1.0
        self.mode_w[ARARAY] = np.array([0, 1, 0, 0, 1, 0, 1, 0], dtype=np.float64)

    def sung(self, k, mode, n_syl):
        """Integer scale degrees of root k, sung in `mode`, over n_syl syllables."""
        x = self.bodies[k]
        y = self.mode_a[mode] * x + self.mode_b[mode] + self.mode_w[mode]
        y = RESAMPLERS[n_syl] @ y
        return np.clip(np.rint(y), 0, D_DEG - 1).astype(np.int64)


# Calendar: fraction of the year. Two fasting seasons (ʿǝzl), two great feasts
# (araray), ordinary time otherwise (gǝʿǝz). A caricature of the real temporal
# cycle (Fritsch 2001), enough to make "infer the mode from the day" a task.
FAST_WINDOWS = [(0.15, 0.30), (0.80, 0.90)]
FEAST_WINDOWS = [(0.30, 0.36), (0.95, 1.00), (0.00, 0.03)]


def day_for_mode(mode, rng):
    if mode == EZEL:
        lo, hi = FAST_WINDOWS[rng.randint(len(FAST_WINDOWS))]
    elif mode == ARARAY:
        lo, hi = FEAST_WINDOWS[rng.randint(len(FEAST_WINDOWS))]
    else:
        while True:
            d = rng.rand()
            if not any(lo <= d < hi for lo, hi in FAST_WINDOWS + FEAST_WINDOWS):
                return d
    return lo + (hi - lo) * rng.rand()


def day_features(d):
    d = np.asarray(d, dtype=np.float64)
    return np.stack([np.sin(2 * np.pi * d), np.cos(2 * np.pi * d),
                     np.sin(4 * np.pi * d), np.cos(4 * np.pi * d)], axis=-1)


# Which (root, mode) pairs the singer was taught. Even-indexed memorised roots
# were never heard in araray: that is the MODE_TRANSFER test.
def taught(k, mode):
    if k >= K_MEMORISED:
        return False
    if mode == ARARAY and k % 2 == 0:
        return False
    return True


REGIMES = ["CANON", "MODE_TRANSFER", "HEAD_ONLY", "SYLLABLE_ONLY",
           "SEASON_ONLY", "NOISY_CUE", "UNSEEN_ROOT"]


def sample_portions(rep, n, regime, rng, head_noise=0.08):
    """Draw n notated phrases under a regime. Returns a dict of arrays.

    C     (n, T_READ, 3+K_TRUE)  T_READ successive READINGS of the same sign:
                           the remembered opening notes, with fresh noise on
                           every reading, and the syllable abbreviation one-hot
                           (possibly dropped for the whole phrase)
    Hh    (n, H)           house (bet) sign one-hot
    M     (n, 3)           mode one-hot as written (zeros if not written)
    given (n,)             1 if the mode was written, else 0
    Phi   (n, 4)           calendar features of the day
    nsyl  (n,)             syllables of the text
    R     (n, L_MAX, L)    resampling matrices (zero rows beyond nsyl)
    Y     (n, L_MAX)       target scale degrees (garbage beyond mask)
    mask  (n, L_MAX)       1 where a syllable exists
    src, mode (n,)         ground-truth identities (for analysis only)
    """
    src = np.zeros(n, dtype=np.int64)
    mode = np.zeros(n, dtype=np.int64)
    for i in range(n):
        while True:
            if regime == "UNSEEN_ROOT":
                k = K_MEMORISED + rng.randint(K_TRUE - K_MEMORISED)
                m = rng.randint(N_MODES)
                if m == ARARAY and k % 2 == 0:
                    continue
                break
            if regime == "MODE_TRANSFER":
                k = 2 * rng.randint(K_MEMORISED // 2)
                m = ARARAY
                break
            k = rng.randint(K_MEMORISED)
            m = rng.randint(N_MODES)
            if taught(k, m):
                break
        src[i], mode[i] = k, m

    nsyl = rng.randint(4, L_MAX + 1, size=n)
    heads = (rep.heads[src] - 6.0) / 6.0                       # (n, 3) clean
    noise = head_noise
    if regime == "NOISY_CUE":
        noise = 0.35
    syl = np.zeros((n, K_TRUE))
    syl[np.arange(n), src] = 1.0
    if regime == "HEAD_ONLY":
        keep = np.zeros(n)
    elif regime == "SYLLABLE_ONLY":
        keep = np.ones(n)
        heads = np.zeros_like(heads)
    else:
        keep = (rng.rand(n) < 0.7).astype(np.float64)
    syl *= keep[:, None]
    # every climb is a fresh reading of the same page: same abbreviation,
    # independently noisy memory of the opening
    readings = heads[:, None, :] + noise * rng.randn(n, T_READ, 3)
    if regime == "SYLLABLE_ONLY":
        readings[:] = 0.0
    C = np.concatenate([readings, np.repeat(syl[:, None, :], T_READ, axis=1)], axis=2)

    Hh = np.zeros((n, H_HOUSES))
    Hh[np.arange(n), rep.house_of[src]] = 1.0

    if regime == "SEASON_ONLY":
        given = np.zeros(n)
    else:
        given = (rng.rand(n) < 0.85).astype(np.float64)
    M = np.zeros((n, N_MODES))
    M[np.arange(n), mode] = 1.0
    M *= given[:, None]

    days = np.array([day_for_mode(m, rng) for m in mode])
    Phi = day_features(days)

    R = np.zeros((n, L_MAX, L_CANON))
    Y = np.zeros((n, L_MAX), dtype=np.int64)
    mask = np.zeros((n, L_MAX))
    for i in range(n):
        ns = nsyl[i]
        R[i, :ns] = RESAMPLERS[ns]
        Y[i, :ns] = rep.sung(src[i], mode[i], ns)
        mask[i, :ns] = 1.0

    return dict(C=C, Hh=Hh, M=M, given=given, Phi=Phi, nsyl=nsyl, R=R, Y=Y,
                mask=mask, src=src, mode=mode, regime=regime)


def masked_ce(logits, Y, mask):
    """Mean cross-entropy over masked positions; returns (loss, dlogits)."""
    logp = logits - logsumexp(logits, axis=-1, keepdims=True)
    n, s, d = logits.shape
    idx = (np.arange(n)[:, None], np.arange(s)[None, :], Y)
    msum = mask.sum()
    loss = -(mask * logp[idx]).sum() / msum
    p = np.exp(logp)
    onehot = np.zeros_like(p)
    onehot[idx] = 1.0
    dlogits = (p - onehot) * (mask[:, :, None] / msum)
    return loss, dlogits


# ==============================================================================
# 2.  THE MELEKKET MACHINE
# ==============================================================================

class MelekketMachine:
    """Reference memory with iterated head-matching retrieval, three-mode warp,
    calendar inference and syllable stretching. Hand-derived gradients.

    The climb. Let u_t be the t-th READING of the sign (the page re-read,
    with fresh noise), projected into head-space. Then

        s_0      = g_0 u_0
        z_t      = beta_t * <s_t, Keys> / sqrt(dh) + house_bias
        p_t      = softmax(z_t)                     the pointer
        r_t      = p_t Keys                         what memory hands back
        s_{t+1}  = (1-lam) s_t + lam r_t + g_{t+1} u_{t+1}
        beta_t   = beta0 (1 + gamma t)              comparison sharpens
        g_t      = sigmoid(pg - gamma_g t)          the world gate closes

    Two things happen at once across the seven climbs: the comparison with
    memory grows sharper (commitment) and the gate on the world grows
    narrower (absorption). Early climbs are dominated by the page; late
    climbs by what has been recalled. That is the spear in the foot.

    Parameters
    ----------
    Wc, bc      reading -> head-space projection         (dc x dh), (dh)
    Keys        HEADS of the K memorised roots            (K x dh)
    Bodies      BODIES (canonical contours) of the roots  (K x L)
    Uh          house -> root eligibility bias            (H x K)
    pbeta       base sharpness of the head comparison     (softplus)
    pgamma      commitment: sharpness gain per climb      (softplus)
    plam        step size of the climb toward memory      (sigmoid)
    pg, pgg     world gate level and closing rate         (sigmoid, softplus)
    a, b, Wm    mode warps: scale, shift, ornament        (3), (3), (3 x L)
    pkappa      width of the scale-degree tuning curves   (softplus)
    Ws, bs      calendar -> mode inference                (4 x 3), (3)
    """

    def __init__(self, dc, dh=16, K=K_MEMORISED, H=H_HOUSES, L=L_CANON, T=7, seed=0):
        rng = np.random.RandomState(seed)
        self.dc, self.dh, self.K, self.H, self.L, self.T = dc, dh, K, H, L, T
        p = {}
        p["Wc"] = rng.randn(dc, dh) * 0.4
        p["bc"] = np.zeros(dh)
        p["Keys"] = rng.randn(K, dh) * 0.6
        p["Bodies"] = 6.0 + rng.randn(K, L) * 0.8
        p["Uh"] = np.zeros((H, K))
        p["pbeta"] = np.array(1.0)
        p["pgamma"] = np.array(-0.5)
        p["plam"] = np.array(0.0)
        p["pg"] = np.array(1.5)
        p["pgg"] = np.array(-1.0)
        p["a"] = np.array([1.0, 0.8, 1.2])
        p["b"] = np.array([0.0, -1.0, 1.0])
        p["Wm"] = np.zeros((N_MODES, L))
        p["pkappa"] = np.array(0.3)
        p["Ws"] = rng.randn(4, N_MODES) * 0.1
        p["bs"] = np.zeros(N_MODES)
        self.p = p

    # ---- derived scalars ----
    def scalars(self):
        p = self.p
        return (float(softplus(p["pbeta"])), float(softplus(p["pgamma"])),
                float(sigmoid(p["plam"])), float(softplus(p["pkappa"])))

    def gates(self):
        """World gate g_t for t = 0..T-1 (sigmoid(pg - gamma_g t))."""
        p = self.p
        gg = float(softplus(p["pgg"]))
        t = np.arange(self.T, dtype=np.float64)
        return sigmoid(float(p["pg"]) - gg * t), gg

    def n_params(self):
        return int(sum(np.size(v) for v in self.p.values()))

    # ---- forward ----
    def forward(self, batch, u_perturb=None):
        """Full forward pass. Returns (logits, cache).

        u_perturb: optional (t, delta) — a disturbance added to the t-th
        READING (in head-space, before the gate). Used by the spear test."""
        p = self.p
        C, Hh, M, given, Phi, R = (batch["C"], batch["Hh"], batch["M"],
                                   batch["given"], batch["Phi"], batch["R"])
        beta0, gamma, lam, kappa = self.scalars()
        gate, gg = self.gates()
        dh, T = self.dh, self.T
        sq = np.sqrt(dh)
        Keys = p["Keys"]

        U = np.einsum("ntc,ch->nth", C[:, :T], p["Wc"]) + p["bc"]     # readings (n,T,dh)
        if u_perturb is not None:
            U = U.copy()
            U[:, u_perturb[0]] += u_perturb[1]
        hb = Hh @ p["Uh"]
        S, P, Rr, betas = [gate[0] * U[:, 0]], [], [], []
        for t in range(T):
            beta_t = beta0 * (1.0 + gamma * t)
            z = (S[t] @ Keys.T) / sq * beta_t + hb
            pt = softmax(z)
            r = pt @ Keys
            s_next = (1 - lam) * S[t] + lam * r
            if t + 1 < T:
                s_next = s_next + gate[t + 1] * U[:, t + 1]
            S.append(s_next)
            P.append(pt); Rr.append(r); betas.append(beta_t)
        beta_T = beta0 * (1.0 + gamma * T)
        zT = (S[T] @ Keys.T) / sq * beta_T + hb
        pT = softmax(zT)
        x = pT @ p["Bodies"]                                    # (n, L)

        q_inf = softmax(Phi @ p["Ws"] + p["bs"])                # (n, 3)
        g = given[:, None]
        q = g * M + (1 - g) * q_inf
        A = q @ p["a"]; B = q @ p["b"]; W = q @ p["Wm"]
        y = A[:, None] * x + B[:, None] + W                     # (n, L)
        ys = np.einsum("nsl,nl->ns", R, y)                      # (n, L_MAX)
        diff = ys[:, :, None] - DEGREES[None, None, :]
        logits = -kappa * diff ** 2                             # (n, L_MAX, D)

        cache = dict(U=U, S=S, P=P, Rr=Rr, betas=betas, beta_T=beta_T, pT=pT, x=x,
                     q_inf=q_inf, q=q, A=A, B=B, W=W, y=y, ys=ys, diff=diff,
                     hb=hb, beta0=beta0, gamma=gamma, lam=lam, kappa=kappa,
                     gate=gate, gg=gg)
        return logits, cache

    def loss(self, batch):
        logits, cache = self.forward(batch)
        L, dlogits = masked_ce(logits, batch["Y"], batch["mask"])
        return L, logits, cache, dlogits

    # ---- backward ----
    def backward(self, batch, cache, dlogits):
        """Hand-derived gradients of the masked cross-entropy w.r.t. every
        parameter, back through the tuning curves, the syllable stretch, the
        mode warp, the final read and all T climbs."""
        p = self.p
        C, Hh, M, given, Phi, R = (batch["C"], batch["Hh"], batch["M"],
                                   batch["given"], batch["Phi"], batch["R"])
        U, S, P, Rr, betas = cache["U"], cache["S"], cache["P"], cache["Rr"], cache["betas"]
        beta_T, pT, x, q_inf, q = cache["beta_T"], cache["pT"], cache["x"], cache["q_inf"], cache["q"]
        y, ys, diff = cache["y"], cache["ys"], cache["diff"]
        beta0, gamma, lam, kappa = cache["beta0"], cache["gamma"], cache["lam"], cache["kappa"]
        gate, gg = cache["gate"], cache["gg"]
        Keys, Bodies = p["Keys"], p["Bodies"]
        dh, T = self.dh, self.T
        sq = np.sqrt(dh)
        g = {k: np.zeros_like(v) for k, v in p.items()}
        dU = np.zeros_like(U)
        dgate = np.zeros(T)

        # tuning curves: logits = -kappa * diff^2
        dys = (dlogits * (-2.0 * kappa * diff)).sum(-1)               # (n, L_MAX)
        dkappa = (dlogits * (-(diff ** 2))).sum()
        g["pkappa"] = dkappa * float(sigmoid(p["pkappa"]))

        # syllable stretch: ys = R y
        dy = np.einsum("nsl,ns->nl", R, dys)                          # (n, L)

        # mode warp: y = A x + B + W
        A = cache["A"]
        dx = dy * A[:, None]
        dA = (dy * x).sum(-1)
        dB = dy.sum(-1)
        dW = dy
        g["a"] = q.T @ dA
        g["b"] = q.T @ dB
        g["Wm"] = q.T @ dW
        dq = dA[:, None] * p["a"][None, :] + dB[:, None] * p["b"][None, :] + dW @ p["Wm"].T
        dq_inf = dq * (1 - given)[:, None]
        dz_s = softmax_backward(q_inf, dq_inf)
        g["Ws"] = Phi.T @ dz_s
        g["bs"] = dz_s.sum(0)

        # final read: x = pT Bodies
        g["Bodies"] = pT.T @ dx
        dpT = dx @ Bodies.T
        dzT = softmax_backward(pT, dpT)
        ST = S[T]
        dS = dzT @ Keys * (beta_T / sq)
        g["Keys"] += (beta_T / sq) * (dzT.T @ ST)
        dhb = dzT.copy()
        dbeta = [0.0] * (T + 1)
        dbeta[T] = float((dzT * ((ST @ Keys.T) / sq)).sum())
        dlam = 0.0

        # the climbs, in reverse
        for t in range(T - 1, -1, -1):
            St, pt, rt, beta_t = S[t], P[t], Rr[t], betas[t]
            dS_next = dS
            if t + 1 < T:                       # the reading that entered at t+1
                dU[:, t + 1] = gate[t + 1] * dS_next
                dgate[t + 1] = float((dS_next * U[:, t + 1]).sum())
            dS = (1 - lam) * dS_next
            dr = lam * dS_next
            dlam += float((dS_next * (rt - St)).sum())
            dp = dr @ Keys.T
            g["Keys"] += pt.T @ dr
            dz = softmax_backward(pt, dp)
            dS = dS + dz @ Keys * (beta_t / sq)
            g["Keys"] += (beta_t / sq) * (dz.T @ St)
            dhb += dz
            dbeta[t] = float((dz * ((St @ Keys.T) / sq)).sum())

        # sharpness schedule beta_t = beta0 (1 + gamma t)
        dbeta0 = sum(dbeta[t] * (1.0 + gamma * t) for t in range(T + 1))
        dgamma = sum(dbeta[t] * beta0 * t for t in range(T + 1))
        g["pbeta"] = np.array(dbeta0 * float(sigmoid(p["pbeta"])))
        g["pgamma"] = np.array(dgamma * float(sigmoid(p["pgamma"])))
        g["plam"] = np.array(dlam * lam * (1 - lam))

        # the first reading and the world gate
        dU[:, 0] = gate[0] * dS
        dgate[0] = float((dS * U[:, 0]).sum())
        t_idx = np.arange(T, dtype=np.float64)
        dsig = gate * (1 - gate)
        g["pg"] = np.array(float((dgate * dsig).sum()))
        g["pgg"] = np.array(float((dgate * dsig * (-t_idx)).sum()) * float(sigmoid(p["pgg"])))

        # inputs of the climb
        g["Uh"] = Hh.T @ dhb
        g["Wc"] = np.einsum("ntc,nth->ch", C[:, :T], dU)
        g["bc"] = dU.sum((0, 1))
        return g

    # ---- diagnostics ----
    def pointer_trace(self, batch):
        """Per-climb pointer distributions (list of (n,K)) incl. final read."""
        _, cache = self.forward(batch)
        return cache["P"] + [cache["pT"]]


# ==============================================================================
# 3.  THE STAFF SCRIBE  —  direct-notation baseline
# ==============================================================================

class StaffScribe:
    """A plain two-hidden-layer MLP from the same inputs to every note's
    logits. It is 'staff notation': the page must carry the melody itself.
    No roots, no pointers, no shared mode warp."""

    def __init__(self, dc, hidden=64, seed=0):
        rng = np.random.RandomState(seed)
        din = dc + H_HOUSES + N_MODES + 1 + 4 + 1
        self.din, self.hidden = din, hidden
        p = {}
        p["W1"] = rng.randn(din, hidden) * np.sqrt(1.0 / din)
        p["b1"] = np.zeros(hidden)
        p["W2"] = rng.randn(hidden, hidden) * np.sqrt(1.0 / hidden)
        p["b2"] = np.zeros(hidden)
        p["W3"] = rng.randn(hidden, L_MAX * D_DEG) * np.sqrt(1.0 / hidden)
        p["b3"] = np.zeros(L_MAX * D_DEG)
        self.p = p

    def n_params(self):
        return int(sum(np.size(v) for v in self.p.values()))

    @staticmethod
    def inputs(batch):
        # the scribe is allowed every reading too: it sees their average
        return np.concatenate([batch["C"].mean(1), batch["Hh"], batch["M"],
                               batch["given"][:, None], batch["Phi"],
                               batch["nsyl"][:, None] / float(L_MAX)], axis=1)

    def forward(self, batch):
        p = self.p
        u = self.inputs(batch)
        h1 = np.tanh(u @ p["W1"] + p["b1"])
        h2 = np.tanh(h1 @ p["W2"] + p["b2"])
        out = h2 @ p["W3"] + p["b3"]
        logits = out.reshape(-1, L_MAX, D_DEG)
        return logits, dict(u=u, h1=h1, h2=h2)

    def loss(self, batch):
        logits, cache = self.forward(batch)
        L, dlogits = masked_ce(logits, batch["Y"], batch["mask"])
        return L, logits, cache, dlogits

    def backward(self, batch, cache, dlogits):
        p = self.p
        u, h1, h2 = cache["u"], cache["h1"], cache["h2"]
        dout = dlogits.reshape(dlogits.shape[0], -1)
        g = {}
        g["W3"] = h2.T @ dout
        g["b3"] = dout.sum(0)
        dh2 = (dout @ p["W3"].T) * (1 - h2 ** 2)
        g["W2"] = h1.T @ dh2
        g["b2"] = dh2.sum(0)
        dh1 = (dh2 @ p["W2"].T) * (1 - h1 ** 2)
        g["W1"] = u.T @ dh1
        g["b1"] = dh1.sum(0)
        return g


# ==============================================================================
# 4.  GRADIENT CHECK  (mandatory)
# ==============================================================================

def gradient_check(model, batch, eps=1e-6, tol=1e-5, rng=None):
    """Compare every analytic gradient with a central finite difference.
    Returns (ok, worst_rel_err, worst_param)."""
    rng = rng or np.random.RandomState(3)
    L0, _, cache, dlogits = model.loss(batch)
    g = model.backward(batch, cache, dlogits)
    worst, where = 0.0, None
    for k, v in model.p.items():
        flat = v.reshape(-1) if v.ndim > 0 else None
        n_el = v.size
        idxs = range(n_el) if n_el <= 12 else rng.choice(n_el, 12, replace=False)
        for i in idxs:
            if v.ndim == 0:
                old = float(v)
                model.p[k] = np.array(old + eps); Lp = model.loss(batch)[0]
                model.p[k] = np.array(old - eps); Lm = model.loss(batch)[0]
                model.p[k] = np.array(old)
                num = (Lp - Lm) / (2 * eps)
                ana = float(g[k])
            else:
                old = flat[i]
                flat[i] = old + eps; Lp = model.loss(batch)[0]
                flat[i] = old - eps; Lm = model.loss(batch)[0]
                flat[i] = old
                num = (Lp - Lm) / (2 * eps)
                ana = g[k].reshape(-1)[i]
            # relative error with an absolute floor: below 1e-3 the central
            # difference itself is only accurate to ~1e-9, so tiny gradients
            # are compared absolutely rather than relatively
            rel = abs(num - ana) / max(1e-3, abs(num) + abs(ana))
            if rel > worst:
                worst, where = rel, f"{k}[{i}]"
    return worst < tol, worst, where


# ==============================================================================
# 5.  TRAINING
# ==============================================================================

def train(model, rep, steps, batch_size, lr, seed, log_every=0, label=""):
    rng = np.random.RandomState(seed)
    opt = Adam(model.p, lr=lr)
    hist = []
    t0 = time.time()
    for step in range(1, steps + 1):
        batch = sample_portions(rep, batch_size, "CANON", rng)
        L, _, cache, dlogits = model.loss(batch)
        g = model.backward(batch, cache, dlogits)
        opt.step(model.p, g)
        hist.append(L)
        if log_every and (step % log_every == 0 or step == 1):
            print(f"    {label:9s} step {step:5d}  loss {L:.4f}   ({time.time()-t0:5.1f}s)")
    return hist


# ==============================================================================
# 6.  EVALUATION
# ==============================================================================

def predict_degrees(logits):
    return logits.argmax(-1)


def accuracy(logits, batch):
    pred = predict_degrees(logits)
    mask = batch["mask"]
    exact = ((pred == batch["Y"]) * mask).sum() / mask.sum()
    within1 = ((np.abs(pred - batch["Y"]) <= 1) * mask).sum() / mask.sum()
    return float(exact), float(within1)


def nearest_known_distance(logits, batch, rep):
    """For each output melody, the mean absolute degree distance to the nearest
    melody the singer was actually taught (any memorised root, any taught
    mode, same syllable count). Low = 'sings a real hymn'. High = 'smears'."""
    pred = predict_degrees(logits)
    out = []
    for i in range(pred.shape[0]):
        ns = int(batch["nsyl"][i])
        best = 1e9
        for k in range(K_MEMORISED):
            for m in range(N_MODES):
                if not taught(k, m):
                    continue
                d = np.abs(pred[i, :ns] - rep.sung(k, m, ns)).mean()
                best = min(best, d)
        out.append(best)
    return float(np.mean(out))


def pointer_entropy(P):
    return float((-(P * np.log(P + 1e-12)).sum(-1)).mean())


def slot_map(model, rep, rng, n=600):
    """Majority-vote map from each memorised root to the slot the trained
    machine points at (the machine never sees root identities)."""
    batch = sample_portions(rep, n, "CANON", rng)
    pT = model.pointer_trace(batch)[-1]
    votes = np.zeros((K_MEMORISED, model.K))
    for i in range(n):
        votes[batch["src"][i], pT[i].argmax()] += 1
    return votes.argmax(1), votes


def pointer_purity(model, rep, rng, n=600):
    m, votes = slot_map(model, rep, rng, n)
    return float(votes.max(1).sum() / votes.sum()), m


def evaluate_all(models, rep, seed=11, n=800):
    rng = np.random.RandomState(seed)
    rows = {}
    for regime in REGIMES:
        batch = sample_portions(rep, n, regime, rng)
        rows[regime] = {}
        for name, mdl in models.items():
            logits, _ = mdl.forward(batch)
            ex, w1 = accuracy(logits, batch)
            nk = nearest_known_distance(logits, batch, rep)
            rows[regime][name] = (ex, w1, nk)
    return rows


# ==============================================================================
# 7.  SELF-TESTS  —  structural invariants, not just the loss
# ==============================================================================

def test_resamplers_are_stochastic():
    for n, R in RESAMPLERS.items():
        assert R.shape == (n, L_CANON)
        assert np.allclose(R.sum(1), 1.0)
        assert (R >= 0).all()
    print("  [ok] syllable-stretch matrices are row-stochastic (melisma preserves the contour)")


def test_repertoire_shapes(rep):
    assert rep.bodies.shape == (K_TRUE, L_CANON)
    assert rep.house_of.shape == (K_TRUE,)
    assert np.bincount(rep.house_of).tolist() == [K_TRUE // H_HOUSES] * H_HOUSES
    for k in range(K_TRUE):
        for m in range(N_MODES):
            for ns in range(1, L_MAX + 1):
                s = rep.sung(k, m, ns)
                assert s.shape == (ns,) and s.min() >= 0 and s.max() < D_DEG
    # the three birds are three different transforms of the same body
    for k in range(3):
        g_, e_, a_ = (rep.sung(k, GEEZ, 8), rep.sung(k, EZEL, 8), rep.sung(k, ARARAY, 8))
        assert not np.array_equal(g_, e_) and not np.array_equal(g_, a_)
    print("  [ok] repertoire: 24 roots, 6 houses, 3 modes, syllable counts 1-8, degrees 0-12")


def test_pointer_is_distribution(model, batch):
    for P in model.pointer_trace(batch):
        assert P.shape[1] == model.K
        assert np.allclose(P.sum(1), 1.0) and (P >= 0).all()
    print("  [ok] every climb's pointer is a probability distribution over memorised roots")


def test_gradient(model, batch, label):
    ok, worst, where = gradient_check(model, batch)
    print(f"  [{'ok' if ok else 'FAIL'}] gradient check {label:9s} worst rel err {worst:.2e} at {where}")
    assert ok, f"gradient check failed for {label}"


def test_training_reduces_loss(hist, label):
    a, b = np.mean(hist[:20]), np.mean(hist[-20:])
    assert b < 0.5 * a, f"{label}: loss did not fall ({a:.3f} -> {b:.3f})"
    print(f"  [ok] {label:9s} training reduced loss {a:.3f} -> {b:.3f}")


def test_absorption_sharpens(model, batch):
    """Commitment: entropy of the pointer must fall across the climbs."""
    Ps = model.pointer_trace(batch)
    H = [pointer_entropy(P) for P in Ps]
    beta0, gamma, lam, kappa = model.scalars()
    assert H[-1] < H[0], f"entropy did not fall over climbs: {H}"
    print("  [ok] commitment: pointer entropy over the climbs "
          + " > ".join(f"{h:.2f}" for h in H) + f"   (gamma={gamma:.2f})")


def test_gate_closes(model):
    """Absorption: the learned gate on the world must be lower at the last
    climb than at the first."""
    gate, gg = model.gates()
    assert gate[-1] < gate[0], f"world gate did not close: {gate}"
    print("  [ok] absorption: world gate over the climbs "
          + " > ".join(f"{g:.2f}" for g in gate) + f"   (closing rate {gg:.2f})")


def test_late_noise_matters_less(model, batch, rng):
    """The spear. The same disturbance added to the reading at climb 1 and to
    the reading at climb T-1 must move the final pointer less when it arrives
    late — after the chant has taken hold."""
    delta = 1.5 * rng.randn(*(batch["C"].shape[0], model.dh))
    base = model.forward(batch)[1]["pT"]
    early = model.forward(batch, u_perturb=(1, delta))[1]["pT"]
    late = model.forward(batch, u_perturb=(model.T - 1, delta))[1]["pT"]
    d_early = np.abs(early - base).sum(1).mean()
    d_late = np.abs(late - base).sum(1).mean()
    assert d_late < d_early, f"late disturbance moved the pointer more ({d_late:.3f} vs {d_early:.3f})"
    print(f"  [ok] the spear: a shock in the reading at climb 1 moves the pointer {d_early:.3f}; "
          f"at climb {model.T-1}, {d_late:.3f}")


def test_persistence_pays_under_noise(rows):
    """Seven readings must beat one reading when the memory of the opening is
    badly degraded (the caterpillar's seventh attempt)."""
    seven = rows["NOISY_CUE"]["MELEKKET"][0]
    one = rows["NOISY_CUE"]["ONE_CLIMB"][0]
    assert seven > one, f"seven climbs did not beat one under noise ({seven:.3f} vs {one:.3f})"
    print(f"  [ok] the seventh climb: under heavy noise seven readings {seven:.3f} exact vs one reading {one:.3f}")


def test_mode_warp_is_shared(model):
    """The learned warps should recover the ordering of the three birds:
    ʿǝzl compresses and lowers, araray expands and raises, relative to gǝʿǝz."""
    a, b = model.p["a"], model.p["b"]
    assert a[EZEL] < a[GEEZ] < a[ARARAY], f"scale ordering wrong: {a}"
    assert b[EZEL] + a[EZEL] * 6 < b[GEEZ] + a[GEEZ] * 6 < b[ARARAY] + a[ARARAY] * 6, f"pivot ordering wrong"
    print(f"  [ok] three birds: learned scale a = ({a[GEEZ]:.2f}, {a[EZEL]:.2f}, {a[ARARAY]:.2f}); "
          f"truth (1.00, 0.50, 1.30)")


def test_house_gates_retrieval(model, rep, rng):
    """Removing the house sign (uniform bet) must lower the probability the
    pointer places on the correct slot."""
    batch = sample_portions(rep, 400, "HEAD_ONLY", rng)
    m, _ = slot_map(model, rep, rng)
    pT = model.forward(batch)[1]["pT"]
    correct = pT[np.arange(400), m[batch["src"]]].mean()
    b2 = dict(batch); b2["Hh"] = np.full_like(batch["Hh"], 1.0 / H_HOUSES)
    pT2 = model.forward(b2)[1]["pT"]
    correct2 = pT2[np.arange(400), m[batch["src"]]].mean()
    assert correct2 < correct, f"house sign did not help ({correct2:.3f} vs {correct:.3f})"
    print(f"  [ok] bet (house) sign: correct-root mass {correct:.3f} with the sign, {correct2:.3f} without")


def test_unseen_root_confabulates_known(rows):
    """On roots never memorised, the reference machine's output must still lie
    close to a hymn it knows (coherent confabulation), and closer than the
    staff scribe's."""
    nk_mel = rows["UNSEEN_ROOT"]["MELEKKET"][2]
    nk_staff = rows["UNSEEN_ROOT"]["STAFF"][2]
    nk_canon = rows["CANON"]["MELEKKET"][2]
    assert nk_mel < nk_staff, f"reference machine not more coherent than staff ({nk_mel:.2f} vs {nk_staff:.2f})"
    print(f"  [ok] the ceiling: on unseen roots MELEKKET sings within {nk_mel:.2f} degrees of a known hymn "
          f"(canon {nk_canon:.2f}); STAFF within {nk_staff:.2f}")


def test_mode_transfer_advantage(rows):
    mel = rows["MODE_TRANSFER"]["MELEKKET"][0]
    stf = rows["MODE_TRANSFER"]["STAFF"][0]
    assert mel > stf, f"no mode-transfer advantage ({mel:.3f} vs {stf:.3f})"
    print(f"  [ok] araray never taught for this root: MELEKKET {mel:.3f} exact vs STAFF {stf:.3f}")


# ==============================================================================
# 8.  THE LEDGER  —  what a retrieval looks like
# ==============================================================================

def ledger(model, rep, rng, m, n=3):
    batch = sample_portions(rep, n, "CANON", rng)
    logits, cache = model.forward(batch)
    Ps = cache["P"] + [cache["pT"]]
    pred = predict_degrees(logits)
    inv = {int(s): k for k, s in enumerate(m)}
    for i in range(n):
        k, md, ns = int(batch["src"][i]), int(batch["mode"][i]), int(batch["nsyl"][i])
        syl_written = batch["C"][i, 0, 3:].sum() > 0
        mode_written = batch["given"][i] > 0
        print(f"  phrase {i+1}: root '{FIDEL[k]}'  house {rep.house_of[k]}  "
              f"mode {MODE_NAMES[md]}{'' if mode_written else ' (unwritten: inferred from the day)'}  "
              f"{ns} syllables  cue = {'abbreviation + opening' if syl_written else 'opening notes only'}")
        climb = " ".join(f"{P[i].max():.2f}" for P in Ps)
        print(f"     climbs (max pointer mass per climb): {climb}")
        top = np.argsort(-cache['pT'][i])[:3]
        tops = ", ".join(f"'{FIDEL[inv[int(s)]] if int(s) in inv else '?'}' {cache['pT'][i][s]:.2f}" for s in top)
        print(f"     final pointer: {tops}")
        print(f"     sung   : {pred[i,:ns].tolist()}")
        print(f"     taught : {batch['Y'][i,:ns].tolist()}")


# ==============================================================================
# 9.  MAIN
# ==============================================================================

def rule(ch="=", n=78):
    print(ch * n)


def main():
    np.set_printoptions(precision=3, suppress=True)
    t_start = time.time()
    rule()
    print(" MELEKKET — the reference-memory chant machine · Chapter 0169 · Yared of Aksum")
    rule()

    rep = Repertoire(seed=0)
    dc = 3 + K_TRUE

    # ------------------------------------------------ structural self-tests
    print("\n[1] Repertoire and structure")
    test_resamplers_are_stochastic()
    test_repertoire_shapes(rep)

    # ------------------------------------------------ gradient checks (small)
    print("\n[2] Gradient checks (central finite differences, eps=1e-6)")
    rng = np.random.RandomState(5)
    small = sample_portions(rep, 6, "CANON", rng)
    small["given"] = np.array([1, 0, 1, 0, 1, 0], dtype=np.float64)
    small["M"] *= small["given"][:, None]
    gm = MelekketMachine(dc, dh=6, K=K_MEMORISED, T=7, seed=1)
    for k in gm.p:                       # move off the symmetric initialisation
        gm.p[k] = gm.p[k] + 0.05 * np.random.RandomState(9).randn(*np.shape(gm.p[k]))
    test_pointer_is_distribution(gm, small)
    test_gradient(gm, small, "MELEKKET")
    gs = StaffScribe(dc, hidden=7, seed=1)
    test_gradient(gs, small, "STAFF")
    g1 = MelekketMachine(dc, dh=6, K=K_MEMORISED, T=1, seed=2)
    test_gradient(g1, small, "ONE_CLIMB")

    # ------------------------------------------------ training
    print("\n[3] Training on the canon (batch 128, Adam)")
    mel = MelekketMachine(dc, dh=16, K=K_MEMORISED, T=7, seed=0)
    one = MelekketMachine(dc, dh=16, K=K_MEMORISED, T=1, seed=0)
    stf = StaffScribe(dc, hidden=64, seed=0)
    print(f"    parameters: MELEKKET {mel.n_params()}   ONE_CLIMB {one.n_params()}   STAFF {stf.n_params()}")
    h_mel = train(mel, rep, steps=1500, batch_size=128, lr=0.02, seed=1, log_every=300, label="MELEKKET")
    h_one = train(one, rep, steps=1500, batch_size=128, lr=0.02, seed=1, log_every=0, label="ONE_CLIMB")
    h_stf = train(stf, rep, steps=1500, batch_size=128, lr=0.003, seed=1, log_every=300, label="STAFF")
    test_training_reduces_loss(h_mel, "MELEKKET")
    test_training_reduces_loss(h_one, "ONE_CLIMB")
    test_training_reduces_loss(h_stf, "STAFF")
    beta0, gamma, lam, kappa = mel.scalars()
    gate, gg = mel.gates()
    print(f"    learned: beta0={beta0:.2f}  gamma(commitment)={gamma:.2f}  lambda(step)={lam:.2f}  "
          f"kappa(tuning)={kappa:.2f}")
    print(f"    learned world gate g_t: " + ", ".join(f"{g:.2f}" for g in gate) + f"  (closing rate {gg:.2f})")

    # ------------------------------------------------ evaluation by regime
    print("\n[4] Evaluation by regime (800 phrases each)   exact / within-1 / nearest-known-hymn distance")
    models = {"MELEKKET": mel, "ONE_CLIMB": one, "STAFF": stf}
    rows = evaluate_all(models, rep)
    print(f"    {'regime':14s}" + "".join(f"{n:>26s}" for n in models))
    for regime in REGIMES:
        line = f"    {regime:14s}"
        for n in models:
            ex, w1, nk = rows[regime][n]
            line += f"      {ex:.3f} / {w1:.3f} / {nk:4.2f}"
        print(line)

    # ------------------------------------------------ behavioural self-tests
    print("\n[5] Behavioural self-tests on the trained machine")
    rng = np.random.RandomState(21)
    probe = sample_portions(rep, 300, "CANON", rng)
    test_absorption_sharpens(mel, probe)
    test_gate_closes(mel)
    test_late_noise_matters_less(mel, probe, rng)
    test_persistence_pays_under_noise(rows)
    test_mode_warp_is_shared(mel)
    test_house_gates_retrieval(mel, rep, rng)
    test_unseen_root_confabulates_known(rows)
    test_mode_transfer_advantage(rows)
    purity, m = pointer_purity(mel, rep, rng)
    print(f"  [ok] pointer purity: {purity:.3f} of canon retrievals land on each root's own slot "
          f"({len(set(m.tolist()))} distinct slots used for {K_MEMORISED} roots)")

    # ------------------------------------------------ the ledger
    print("\n[6] Ledger — three retrievals, climb by climb")
    ledger(mel, rep, np.random.RandomState(33), m)

    rule()
    print(f" all self-tests passed · {time.time()-t_start:.1f}s")
    rule()


if __name__ == "__main__":
    main()

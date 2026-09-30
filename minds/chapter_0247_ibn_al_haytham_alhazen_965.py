#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 THE MANAZIR ENGINE  --  chapter 247  --  Ibn al-Haytham (Alhazen), b. c.965
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 13 Minds 241 - 260 Available on Amazon https://www.amazon.com/dp/B0HLD1S6QT
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0247_ibn_al_haytham_alhazen_965 - Ibn al-Haytham (Alhazen) (c.965-1040)
#================================================================================  

WHAT THIS FILE IS
-----------------
A from-scratch, pure-NumPy trainable architecture that encodes the cognitive
signature of Ibn al-Haytham's Kitab al-Manazir (Book of Optics, c.1011-1038).
It is not an illustration of his ideas; it is an attempt to *build* them, so
that the claims can be tested and, where they fail, be seen to fail.

THE ONE IDEA THIS MACHINE IS BUILT AROUND
-----------------------------------------
Book III of the Optics is a taxonomy of error. Its chapters 5, 6 and 7 partition
mistakes of sight by the FACULTY that produced them -- pure sensation, recognition,
inference -- and each of those chapters is then subdivided A..H by the CONDITION
that pushed the faculty outside what Ibn al-Haytham calls the *moderate range*:

    A distance   B position   C illumination   D size
    E opacity    F transparency of the air   G duration   H condition of the eye

That is a 3 x 8 grid of named failure modes, written in the eleventh century.
Nothing else in the corpus of premodern thought looks like it. The engineering
consequence is the thesis of this file:

    CERTAINTY IS NOT A FEELING. IT IS A MEASURED PROPERTY OF THE SITUATION.

So the machine does not estimate its confidence from its own output
distribution. It estimates the eight conditions from the raw input, and it
predicts, per faculty, whether *those conditions* will break *that faculty*.
Doubt is then spent where the envelope says doubt will pay.

THE PARTS, AND WHOSE THEY ARE
-----------------------------
  glacialis + common nerve  : one weight-shared encoder run over both eyes, fused
                              by averaging. Their disagreement is not discarded --
                              it is the machine's diplopia signal, condition H.
  al-hiss al-mujarrad       : pure sensation. Light and colour are, for him, the
                              only two of the twenty-two visible properties given
                              per se. Here they are read out by six scalars from
                              fixed image statistics, with NO path to memory or
                              inference. Structurally, not by penalty.
  ma'rifa                   : recognition. A small bank of stored forms; the
                              percept is matched against them. Prototypes, not
                              attention over a sequence.
  the ground chain          : his theory of distance, which is not an angle. A
                              distance is known by concatenating measured intervals
                              along a continuous visible surface, calibrated by
                              the body that walked it. Break the surface and the
                              chain shortens. One shared 4-tap reader is applied
                              to eight successive segments -- the same measure,
                              laid down repeatedly.
  qiyas khafi               : the unnoticed syllogism. A fast amortised inference
                              over the fused code, stored forms and chain length.
                              Size is emitted as (angular term x estimated
                              distance) -- his size-distance invariance, wired in.
  tadqiq al-nazar           : deliberate re-examination. Analysis-by-synthesis:
                              re-render the percept, subtract, correct, twice.
                              Expensive, and priced accordingly.
  al-shukuk                 : the auditor. Eight condition-features estimated
                              from the input; a head that predicts which faculty
                              is about to fail.
  al-quwwa al-mumayyiza     : the discriminating faculty. Reads the auditor and
                              decides whether the slow path is worth its cost,
                              and how far to trust an external authority.

WHAT IS DELIBERATELY *NOT* HERE
-------------------------------
No transformer, no attention over stored keys, no softmax mixture of experts.
Attention over a sequence is the wrong shape for this mind: Ibn al-Haytham's
perceiver does not weight tokens, it concatenates intervals along a surface and
audits the result against a table of conditions.

RUN
---
    python3 chapter_0247_ibn_al_haytham_alhazen_965.py

Requires only NumPy. Runs on CPU in about a minute.
================================================================================
"""

import time
import numpy as np

np.seterr(over="ignore", invalid="ignore")

# =============================================================================
# PART 0 -- A MINIMAL REVERSE-MODE AUTODIFF
# -----------------------------------------------------------------------------
# Everything below is built on this. It exists so that the recurrent chain
# reader and the unrolled re-examination loop can be differentiated exactly,
# and so that the finite-difference check in PART 5 has something real to check.
# =============================================================================


def _unbroadcast(g, shape):
    """Fold a gradient back down to `shape` after NumPy broadcasting."""
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g.reshape(shape)


class T:
    """A node on the tape. `data` holds the value, `grad` accumulates."""
    __slots__ = ("data", "grad", "_bw", "_prev")

    def __init__(self, data, prev=()):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._bw = lambda: None
        self._prev = prev

    @property
    def shape(self):
        return self.data.shape

    # ---- binary ops ---------------------------------------------------------
    def __add__(self, o):
        o = o if isinstance(o, T) else T(o)
        out = T(self.data + o.data, (self, o))

        def bw():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            o.grad += _unbroadcast(out.grad, o.data.shape)
        out._bw = bw
        return out

    def __mul__(self, o):
        o = o if isinstance(o, T) else T(o)
        out = T(self.data * o.data, (self, o))

        def bw():
            self.grad += _unbroadcast(out.grad * o.data, self.data.shape)
            o.grad += _unbroadcast(out.grad * self.data, o.data.shape)
        out._bw = bw
        return out

    def __matmul__(self, o):
        out = T(self.data @ o.data, (self, o))

        def bw():
            self.grad += out.grad @ o.data.T
            o.grad += self.data.T @ out.grad
        out._bw = bw
        return out

    def __pow__(self, p):
        out = T(self.data ** p, (self,))

        def bw():
            self.grad += out.grad * p * (self.data ** (p - 1))
        out._bw = bw
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, o):
        return self + (-(o if isinstance(o, T) else T(o)))

    def __rsub__(self, o):
        return (o if isinstance(o, T) else T(o)) + (-self)

    def __truediv__(self, o):
        return self * ((o if isinstance(o, T) else T(o)) ** -1.0)

    __radd__ = __add__
    __rmul__ = __mul__

    # ---- reductions ---------------------------------------------------------
    def sum(self, axis=None, keepdims=False):
        out = T(self.data.sum(axis=axis, keepdims=keepdims), (self,))

        def bw():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.grad += np.broadcast_to(g, self.data.shape).copy()
        out._bw = bw
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # ---- elementwise --------------------------------------------------------
    def _ew(self, val, dval):
        out = T(val, (self,))

        def bw():
            self.grad += out.grad * dval
        out._bw = bw
        return out

    def tanh(self):
        v = np.tanh(self.data)
        return self._ew(v, 1.0 - v * v)

    def sigmoid(self):
        v = 1.0 / (1.0 + np.exp(-np.clip(self.data, -60, 60)))
        return self._ew(v, v * (1.0 - v))

    def relu(self):
        return self._ew(np.maximum(self.data, 0.0), (self.data > 0).astype(np.float64))

    def softplus(self):
        z = np.clip(self.data, -60, 60)
        v = np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)
        return self._ew(v, 1.0 / (1.0 + np.exp(-z)))

    def exp(self):
        v = np.exp(np.clip(self.data, -60, 60))
        return self._ew(v, v)

    def log(self):
        return self._ew(np.log(np.abs(self.data) + 1e-12), 1.0 / (self.data + 1e-12))

    # ---- shape --------------------------------------------------------------
    def sl(self, a, b, axis=1):
        idx = [slice(None)] * self.data.ndim
        idx[axis] = slice(a, b)
        idx = tuple(idx)
        out = T(self.data[idx], (self,))

        def bw():
            self.grad[idx] += out.grad
        out._bw = bw
        return out

    def backward(self):
        topo, seen = [], set()
        stack = [(self, False)]
        while stack:                      # iterative, to survive deep unrolls
            v, done = stack.pop()
            if done:
                topo.append(v)
                continue
            if id(v) in seen:
                continue
            seen.add(id(v))
            stack.append((v, True))
            for p in v._prev:
                if id(p) not in seen:
                    stack.append((p, False))
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._bw()


def cat(ts, axis=1):
    """Concatenate tape nodes along an axis."""
    out = T(np.concatenate([t.data for t in ts], axis=axis), tuple(ts))
    sizes = [t.data.shape[axis] for t in ts]

    def bw():
        o = 0
        for t, s in zip(ts, sizes):
            idx = [slice(None)] * out.data.ndim
            idx[axis] = slice(o, o + s)
            t.grad += out.grad[tuple(idx)]
            o += s
    out._bw = bw
    return out


def softmax(x, axis=-1):
    m = T(x.data.max(axis=axis, keepdims=True))     # constant shift, no grad
    e = (x - m).exp()
    return e / e.sum(axis=axis, keepdims=True)


def cross_entropy(logits, onehot):
    m = T(logits.data.max(axis=1, keepdims=True))
    lse = (logits - m).exp().sum(axis=1, keepdims=True).log() + m
    return (lse - (logits * onehot).sum(axis=1, keepdims=True)).mean()


def bce(p, target):
    """target is a plain NumPy array; p is on the tape."""
    return -((T(target) * (p + 1e-9).log())
             + (T(1.0 - target) * (T(1.0) - p + 1e-9).log())).mean()


def _centroid_np(v):
    """Intensity-weighted centre of the form, in receptor units."""
    w = np.maximum(v - v.mean(1, keepdims=True), 0.0) + 1e-6
    ix = np.arange(v.shape[1])[None, :]
    return (w * ix).sum(1) / w.sum(1)


def _spread_np(v, c):
    """Root second moment about the centre: the form's retinal extent."""
    w = np.maximum(v - v.mean(1, keepdims=True), 0.0) + 1e-6
    ix = np.arange(v.shape[1])[None, :]
    return np.sqrt((w * (ix - c[:, None]) ** 2).sum(1) / w.sum(1))


def resample(x, K):
    """
    Differentiable gather: out[b,j] = sum_i x[b,i] * K[b,i,j].
    K is a constant per-sample sampling kernel. This is how the faculty of
    discrimination SCANS: it brings the form into a canonical window before
    matching it against what it has stored. Ibn al-Haytham insists that
    recognition does not require going through every property of the thing --
    which is only possible if the thing has first been normalised.
    """
    out = T(np.einsum('bi,bij->bj', x.data, K), (x,))

    def bw():
        x.grad += np.einsum('bj,bij->bi', out.grad, K)
    out._bw = bw
    return out


def absv(x):
    """Smooth absolute value, so the condition features stay differentiable."""
    return (x * x + 1e-8) ** 0.5


# =============================================================================
# PART 1 -- THE DARK ROOM (bayt al-muzlim): the world and its physics
# -----------------------------------------------------------------------------
# Ibn al-Haytham's laboratory was a shuttered chamber with a small aperture and
# several lamps outside it. Blocking one lamp extinguished exactly one patch on
# the far wall, which is how he showed that rays cross without mixing and travel
# in straight lines independently. The generator below is that chamber.
#
# Every scene is described by latent quantities that map onto his own list of
# particular visible properties, and by the eight CONDITIONS of Book III. The
# conditions are not labels bolted on afterwards -- they are physically applied
# to the image, so the errors that follow are real consequences, not stipulated.
# =============================================================================

N_BAND = 32                 # receptors across the object band
N_GND = 32                  # receptors across the ground band
N_SEG = 8                   # segments the ground chain is read in
N_SCAN = 16                 # taps in the canonical scan window

# Calibration of what counts as a moderate range for each of the eight
# conditions. Held as constants during any single judgement; refreshed
# periodically during training as the machine's own estimates sharpen.
N_COND = 10
COND_NORM = {"mu": np.zeros((1, N_COND)), "sd": np.ones((1, N_COND))}
SEG = N_GND // N_SEG
OBJ_DIM = 3 * N_BAND        # intensity + two chroma rows, per eye
BASELINE = 0.18             # interocular baseline, world units
EYE_HEIGHT = 1.6            # observer's eye height, world units (the ground cue)
ANG = np.linspace(-0.25, 0.25, N_BAND)   # retinal angle of each receptor, radians
SHAPES = ("bar", "spike", "pair", "wedge")


def _profile(kind, u):
    """The four stored forms, as luminance profiles over normalised support u."""
    inside = (np.abs(u) <= 1.0)
    if kind == 0:                                   # bar: flat top
        p = np.where(inside, 1.0, 0.0)
    elif kind == 1:                                 # spike: symmetric peak
        p = np.where(inside, np.maximum(1.0 - np.abs(u), 0.0) ** 0.85, 0.0)
    elif kind == 2:                                 # pair: two lobes, clear notch
        p = np.where(inside & (np.abs(u) > 0.45), 1.0, 0.0)
    else:                                           # wedge: one-sided ramp
        p = np.where(inside, np.clip(0.12 + 0.88 * (0.5 + 0.5 * u), 0, 1), 0.0)
    return p


def _blur(x, w):
    """Box blur along the last axis; w is the half-width in receptors."""
    if w <= 0:
        return x
    k = np.ones(2 * w + 1) / (2 * w + 1)
    pad = np.pad(x, ((0, 0), (w, w)), mode="edge")
    return np.apply_along_axis(lambda r: np.convolve(r, k, mode="valid"), 1, pad)


def make_batch(rng, n, force=None):
    """
    Build `n` scenes. `force` overrides any latent, which is how the i'tibar
    battery in PART 7 walks a single condition out of its moderate range while
    holding everything else fixed -- a controlled test, in his sense.
    """
    f = force or {}

    def draw(name, lo, hi, log=False):
        if name in f:
            v = f[name]
            return np.full(n, v, dtype=float) if np.isscalar(v) else np.asarray(v, float)
        if log:
            return np.exp(rng.uniform(np.log(lo), np.log(hi), n))
        return rng.uniform(lo, hi, n)

    d = draw("d", 2.0, 22.0, log=True)               # A  distance
    phi = draw("phi", -0.14, 0.14)                   # B  position
    lum = draw("lum", 0.10, 1.00)                    # C  illumination
    size = draw("size", 0.60, 3.00)                  # D  size (world units)
    opac = draw("opac", 0.40, 1.00)                  # E  opacity
    trans = draw("trans", 0.45, 1.00)                # F  transparency of the air
    tau = draw("tau", 0.20, 1.00)                    # G  duration of the look
    eye = draw("eye", 0.00, 0.55)                    # H  condition of the eye

    kind = f["kind"] * np.ones(n, int) if "kind" in f else rng.integers(0, 4, n)
    # ground: present, partially broken, or wholly absent (the sky case)
    if "ground" in f:
        ground_mode = np.full(n, f["ground"], int)
    else:
        ground_mode = rng.choice([0, 1, 2], n, p=[0.62, 0.23, 0.15])  # 0 whole 1 broken 2 void

    theta = np.clip(size / d, 0.050, 0.40)           # angular size
    obj = np.zeros((n, 2, 3, N_BAND))
    gnd = np.zeros((n, N_GND))

    for i in range(n):
        for e_ix, sgn in enumerate((+1.0, -1.0)):
            # the two eyes see the object at slightly different angles: disparity
            # falls off as 1/d, so binocular evidence is only good up close --
            # which is exactly why he needed the ground.
            phi_e = phi[i] + sgn * 0.5 * BASELINE / d[i]
            u = (ANG - phi_e) / (0.5 * theta[i])
            p = _profile(kind[i], u) * lum[i] * opac[i]
            rows = np.stack([p, p * 0.9, p * 0.9])   # intensity + 2 chroma carriers
            obj[i, e_ix] = rows

        # ground markers: unit intervals laid along the surface out to the object,
        # projected by perspective. Their spacing on the retina compresses as 1/j.
        m = np.zeros(N_GND)
        far = int(np.floor(min(d[i], 22)))
        cut = far + 1
        if ground_mode[i] == 1:                       # a break in the surface
            cut = max(2, int(rng.uniform(0.25, 0.7) * far))
        if ground_mode[i] != 2:
            for j in range(1, min(far, cut) + 1):
                psi = np.arctan(EYE_HEIGHT / j)       # elevation of marker j
                pos = (1.0 - psi / 0.80) * (N_GND - 1)
                if 0 <= pos <= N_GND - 1:
                    ix = np.arange(N_GND)
                    m += np.exp(-0.5 * ((ix - pos) / 0.70) ** 2)
        gnd[i] = np.clip(m, 0, 1.3) * lum[i] * 0.75

    # --- apply the conditions to the image, in order -------------------------
    obj = obj.reshape(n * 2 * 3, N_BAND)
    blur_w = np.round(2.2 * eye).astype(int)
    for w in np.unique(blur_w):                       # H: a defective eye blurs
        if w > 0:
            sel = np.repeat(blur_w == w, 6)
            obj[sel] = _blur(obj[sel], int(w))
    obj = obj.reshape(n, 2, 3, N_BAND)
    chroma = rng.uniform(-1, 1, (n, 2))                # the two chroma carriers
    obj[:, :, 1] = obj[:, :, 1] * chroma[:, None, 0, None]
    obj[:, :, 2] = obj[:, :, 2] * chroma[:, None, 1, None]

    haze = 0.28 * lum                                  # F: veiling light
    obj[:, :, 0] = trans[:, None, None] * obj[:, :, 0] + (1 - trans)[:, None, None] * haze[:, None, None]
    gnd = trans[:, None] * gnd + (1 - trans)[:, None] * haze[:, None]

    sigma = 0.012 + 0.075 * (1 - tau)                  # G: a short look is noisy
    obj += rng.normal(0, 1, obj.shape) * sigma[:, None, None, None]
    gnd += rng.normal(0, 1, gnd.shape) * sigma[:, None]

    prop = np.stack([np.full(n, EYE_HEIGHT), np.ones(n),           # the walked unit
                     np.full(n, BASELINE), rng.normal(0, 0.02, n)], 1)

    # --- the authority channel: Ptolemy, in effect ---------------------------
    # An external source that reports distance. It is accurate inside the
    # moderate range and systematically wrong outside it -- which is precisely
    # the shape of the errors Ibn al-Haytham catalogued in al-Shukuk.
    in_env = (d <= 12) & (ground_mode == 0) & (tau >= 0.5) & (eye <= 0.25)
    auth_bias = np.where(in_env, 0.0, rng.choice([-0.75, 0.75], n))
    authority = d * (1.0 + auth_bias) * (1 + rng.normal(0, 0.03, n))

    X = {
        "objL": obj[:, 0].reshape(n, OBJ_DIM),
        "objR": obj[:, 1].reshape(n, OBJ_DIM),
        "gnd": gnd,
        "prop": prop,
        "auth": (authority / 10.0)[:, None],
    }
    Y = {
        "light": (lum * opac)[:, None],
        "colour": chroma,
        "kind": kind,
        "d": d[:, None],
        "size": size[:, None],
    }
    meta = {"d": d, "phi": phi, "lum": lum, "size": size, "opac": opac,
            "trans": trans, "tau": tau, "eye": eye, "ground": ground_mode,
            "theta": theta, "in_env": in_env, "auth_bias": auth_bias}
    return X, Y, meta


# =============================================================================
# PART 2 -- PARAMETERS
# =============================================================================

def init_params(rng):
    def p(*shape, s=None):
        s = s if s is not None else (1.0 / np.sqrt(shape[0]))
        return T(rng.normal(0, s, shape))

    P = {}
    # -- pure sensation: six scalars, and nothing else touches these outputs ---
    P["g_light"] = T(np.array([[1.0]]))
    P["b_light"] = T(np.array([[0.0]]))
    P["g_col"] = T(np.array([[1.0, 1.0]]))
    P["b_col"] = T(np.array([[0.0, 0.0]]))
    # -- glacialis: one encoder, run over both eyes ---------------------------
    P["We"] = p(OBJ_DIM, 40)
    P["be"] = T(np.zeros((1, 40)))
    # -- recognition: a bank of eight stored forms ----------------------------
    P["proto"] = p(N_SCAN, 12, s=0.6)
    P["ptemp"] = T(np.array([[0.5]]))
    P["V"] = p(12, 4, s=0.5)
    # -- the ground chain: one 4-tap reader, shared over eight segments --------
    P["cw"] = p(SEG + 4, 1, s=0.5)
    P["cb"] = T(np.zeros((1, 1)))
    P["cu"] = p(SEG + 4, 1, s=0.5)
    P["cc"] = T(np.array([[1.0]]))
    P["lam"] = T(np.array([[0.6]]))          # the calibrated step length
    P["Wprop"] = p(4, 1, s=0.2)
    # -- disparity route (good only near) -------------------------------------
    P["Wdis"] = p(40, 1, s=0.2)
    P["bdis"] = T(np.zeros((1, 1)))
    # -- qiyas khafi: the unnoticed syllogism ---------------------------------
    IN_F = 40 + 12 + N_SCAN + 2 + 1 + N_COND
    P["W1"] = p(IN_F, 32)
    P["b1"] = T(np.zeros((1, 32)))
    P["W2"] = p(32, 2)
    P["b2"] = T(np.zeros((1, 2)))
    # the angular head is structurally blind to distance: it sees only the form
    P["Wang"] = p(N_SCAN + 3, 16, s=0.5); P["bang"] = T(np.zeros((1, 16)))
    P["Wang2"] = p(16, 1, s=0.3); P["bang2"] = T(np.array([[-2.0]]))
    P["mixd"] = T(np.array([[1.0], [0.3]]))   # chain vs disparity weighting
    # -- tadqiq al-nazar: render, subtract, correct ---------------------------
    P["Wd1"] = p(9, 32); P["bd1"] = T(np.zeros((1, 32)))
    P["Wd2"] = p(32, N_BAND); P["bd2"] = T(np.zeros((1, N_BAND)))
    P["Wr1"] = p(N_BAND + 9, 32); P["br1"] = T(np.zeros((1, 32)))
    P["Wr2"] = p(32, 9, s=0.05); P["br2"] = T(np.zeros((1, 9)))
    P["eta"] = T(np.array([[-1.0]]))
    # -- al-shukuk: the auditor over the eight conditions ---------------------
    P["Ws1"] = p(N_COND, 16); P["bs1"] = T(np.zeros((1, 16)))
    P["Ws2"] = p(16, 3); P["bs2"] = T(np.array([[-0.5, -0.5, -0.5]]))
    # -- al-quwwa al-mumayyiza: gate on the slow path, gate on the authority --
    P["Wg"] = p(N_COND + 3, 1, s=0.6); P["bg"] = T(np.array([[-0.5]]))
    P["Wa"] = p(N_COND + 3, 16, s=0.9); P["ba"] = T(np.zeros((1, 16)))
    P["Wa2"] = p(16, 1, s=0.6); P["ba2"] = T(np.array([[0.0]]))
    return P


# =============================================================================
# PART 3 -- THE FORWARD PASS
# =============================================================================

def forward(P, X, n_refine=2, ablate=None, gate_force=None):
    """
    ablate: a set of strings, any of {"recognition", "chain", "slow", "shukuk",
    "authority"}, used by the battery to remove one organ and see what dies.
    """
    ablate = ablate or set()
    objL, objR = T(X["objL"]), T(X["objR"])
    gnd, prop, auth = T(X["gnd"]), T(X["prop"]), T(X["auth"])
    B = objL.shape[0]

    # -- intensity and chroma rows, per eye -----------------------------------
    iL, iR = objL.sl(0, N_BAND), objR.sl(0, N_BAND)
    c1 = (objL.sl(N_BAND, 2 * N_BAND) + objR.sl(N_BAND, 2 * N_BAND)) * 0.5
    c2 = (objL.sl(2 * N_BAND, 3 * N_BAND) + objR.sl(2 * N_BAND, 3 * N_BAND)) * 0.5
    fused_i = (iL + iR) * 0.5

    # -------------------------------------------------------------------------
    # 3.1  PURE SENSATION  (al-hiss al-mujarrad)
    # Light and colour, and only these, are read straight off fixed statistics
    # of the image by six scalars. There is no wire from memory or inference
    # into this block. His claim is a type signature here, not a soft prior.
    # -------------------------------------------------------------------------
    tot = fused_i.mean(axis=1, keepdims=True)
    light = tot * P["g_light"] * 2.2 + P["b_light"]
    denom = absv(tot) + 0.05
    colour = cat([c1.mean(axis=1, keepdims=True) / denom,
                  c2.mean(axis=1, keepdims=True) / denom]) * P["g_col"] * 1.15 + P["b_col"]

    # -------------------------------------------------------------------------
    # 3.2  THE COMMON NERVE  (al-mujawwaf al-mushtarak)
    # One encoder, both eyes, then averaged. He located the meeting of the two
    # forms in the hollow common nerve and made single vision depend on their
    # agreeing. Their DISAGREEMENT is kept: it is the machine's diplopia.
    # -------------------------------------------------------------------------
    hL = (objL @ P["We"] + P["be"]).tanh()
    hR = (objR @ P["We"] + P["be"]).tanh()
    f = (hL + hR) * 0.5
    delta = ((hL - hR) ** 2).mean(axis=1, keepdims=True)

    # -------------------------------------------------------------------------
    # 3.3  MOMENTS OF THE FORM, then RECOGNITION (ma'rifa)
    # -------------------------------------------------------------------------
    idx = T(np.linspace(-1, 1, N_BAND)[None, :])
    mass = absv(fused_i).sum(axis=1, keepdims=True) + 1e-3
    centroid = (fused_i * idx).sum(axis=1, keepdims=True) / mass
    spread = ((fused_i * (idx - centroid) ** 2).sum(axis=1, keepdims=True) / mass)
    # -- the scan: centre on the form, scale to its extent, resample to 16 taps
    ix = np.arange(N_BAND)[None, :]
    c0 = np.clip(_centroid_np(fused_i.data), 2, N_BAND - 3)[:, None]
    w0 = np.clip(_spread_np(fused_i.data, c0.ravel()), 1.2, 14.0)[:, None]
    if "scan" in ablate:                       # look without scrutinising: a
        c0 = np.full_like(c0, (N_BAND - 1) / 2.0)   # fixed window, no centring,
        w0 = np.full_like(w0, 6.0)                  # no scaling to the form
    grid = c0 + np.linspace(-2.6, 2.6, N_SCAN)[None, :] * w0        # (B, N_SCAN)
    K = np.exp(-0.5 * ((ix[:, :, None] - grid[:, None, :]) / 0.65) ** 2)
    K = K / (K.sum(1, keepdims=True) + 1e-9)                          # (B,32,N_SCAN)
    scan = resample(fused_i, K)
    nrm = ((scan ** 2).mean(axis=1, keepdims=True) + 1e-4) ** 0.5
    scan_n = scan / nrm

    sim = (scan_n @ P["proto"]) * P["ptemp"].softplus()
    a = softmax(sim)
    if "recognition" in ablate:
        a = T(np.full((B, 12), 1.0 / 12))
    shape_rec = a @ P["V"]

    # -------------------------------------------------------------------------
    # 3.4  THE GROUND CHAIN -- distance as concatenated intervals
    # The same 4-tap reader is applied to eight successive segments of the
    # ground band and the results are added. A gate per segment decides whether
    # that stretch of surface is there at all; where the surface is broken the
    # gate closes and the chain comes up short. Distance is then this sum times
    # one learned scalar: the step, calibrated by the body.
    # -------------------------------------------------------------------------
    ivals, gates = [], []
    ones = T(np.ones((B, 1)))
    for m in range(N_SEG):
        seg = gnd.sl(m * SEG, (m + 1) * SEG)
        # the reader is told WHERE along the ground it is standing, because
        # perspective makes one image-interval mean different world-intervals
        # near and far. One shared reader, eight stations.
        pos = ones * (m / (N_SEG - 1.0))
        dseg = seg.sl(1, SEG) - seg.sl(0, SEG - 1)
        feats = cat([seg, seg.mean(axis=1, keepdims=True),
                     (dseg ** 2).mean(axis=1, keepdims=True), pos, pos * pos])
        present = (feats @ P["cu"] + P["cc"]).sigmoid()
        gates.append(present)
        ivals.append(present * (feats @ P["cw"] + P["cb"]).softplus())
    chain = ivals[0]
    for v in ivals[1:]:
        chain = chain + v
    if "chain" in ablate:
        chain = chain * 0.0
    gmat = cat(gates)                                    # (B, N_SEG)
    gmean = gmat.mean(axis=1, keepdims=True)
    # A break in the surface shows up as one station going dark in the middle of
    # a lit chain. He requires the interval to run along a CONTINUOUS series of
    # bodies; this is the machine noticing that it does not.
    gvar = ((gmat - gmean) ** 2).mean(axis=1, keepdims=True)
    lam = (P["lam"] + prop @ P["Wprop"]).softplus()
    d_chain = chain * lam

    # disparity: real, but useful only at short range
    d_disp = (f @ P["Wdis"] + P["bdis"]).softplus()
    if "disparity" in ablate:
        d_disp = d_disp * 0.0

    # -------------------------------------------------------------------------
    # 3.5b THE EIGHT CONDITIONS, estimated from the input alone
    # Nothing here is an oracle. The machine must perceive the conditions of
    # its own seeing, which is what Book III assumes the observer does.
    # -------------------------------------------------------------------------
    dif = fused_i.sl(1, N_BAND) - fused_i.sl(0, N_BAND - 1)
    cond = cat([
        d_chain * 0.10,                        # A  distance
        absv(centroid),                        # B  position
        tot * 2.0,                             # C  illumination
        spread * 3.0,                          # D  size
        ((fused_i - tot) ** 2).mean(axis=1, keepdims=True) * 6.0,   # E  opacity/contrast
        gnd.mean(axis=1, keepdims=True) * 2.0,                      # F  transparency
        (dif ** 2).mean(axis=1, keepdims=True) * 25.0,              # G  duration
        delta * 4.0,                                                # H  the eye
        gmean, gvar,                           # + the continuity of the surface
    ])
    # The eight raw measures live on wildly different scales. They are rescaled
    # by constants calibrated once from experience of this world (see
    # calibrate_conditions). This is not cosmetic: it is what makes "moderate"
    # mean anything. A range is moderate relative to how much that condition
    # actually varies in the world one has grown up looking at.
    cond = (cond - T(COND_NORM["mu"])) / T(COND_NORM["sd"])

    # -------------------------------------------------------------------------
    # 3.6  AL-SHUKUK -- the auditor
    # Predicts, per faculty, whether the fast path is about to fail, from the
    # conditions and nothing else. Three outputs: sensation, recognition,
    # inference -- Book III chapters 5, 6, 7.
    # -------------------------------------------------------------------------
    rho = ((cond @ P["Ws1"] + P["bs1"]).tanh() @ P["Ws2"] + P["bs2"]).sigmoid()
    if "shukuk" in ablate:
        rho = T(np.full((B, 3), 0.5))

    # -------------------------------------------------------------------------
    # 3.7  QIYAS KHAFI -- the unnoticed syllogism (fast path)
    # Size is emitted as an angular term MULTIPLIED by the estimated distance.
    # That multiplication is his size-distance invariance, and it is the reason
    # the moon illusion falls out of this machine rather than being told to it.
    # -------------------------------------------------------------------------
    feat = cat([f, a, scan_n, d_chain * 0.1, d_disp * 0.1, delta, cond])
    hid = (feat @ P["W1"] + P["b1"]).tanh()
    out = hid @ P["W2"] + P["b2"]
    d_fast = (d_chain * P["mixd"].sl(0, 1, axis=0) * 1.0
              + d_disp * P["mixd"].sl(1, 2, axis=0) + out.sl(0, 1)).softplus()
    # Size is angle TIMES distance, and the angle is computed by a head that
    # cannot see any distance estimate at all. This is his size-distance
    # invariance imposed as a type signature, and it is the reason the moon
    # illusion falls out of the machine in test [4] rather than being told to it.
    ang_in = cat([scan_n, spread * 3.0, mass * 0.1, nrm])
    ang_term = ((ang_in @ P["Wang"] + P["bang"]).tanh() @ P["Wang2"] + P["bang2"]).softplus()
    ang_term = ang_term * 0.06
    size_fast = ang_term * d_fast
    shape_fast = shape_rec

    # -------------------------------------------------------------------------
    # 3.8  TADQIQ AL-NAZAR -- deliberate re-examination (slow path)
    # Render the percept back into an image, subtract it from what is actually
    # there, and let the residual correct the percept. Twice. This is
    # analysis-by-synthesis, and it is expensive on purpose.
    # -------------------------------------------------------------------------
    m0 = cat([light, colour, shape_fast, d_fast * 0.1, size_fast])
    m_hat = m0
    eta = P["eta"].softplus()
    recon = None
    for _ in range(n_refine):
        recon = (m_hat @ P["Wd1"] + P["bd1"]).tanh() @ P["Wd2"] + P["bd2"]
        resid = fused_i - recon
        upd = (cat([resid, m_hat]) @ P["Wr1"] + P["br1"]).tanh() @ P["Wr2"] + P["br2"]
        m_hat = m_hat + upd * eta
    # The slow path returns a CORRECTION, so that at initialisation it is exactly
    # the fast path: re-examination can only ever be a departure from first sight.
    corr = m_hat - m0
    d_slow = d_fast + corr.sl(7, 8) * 10.0
    size_slow = size_fast + corr.sl(8, 9)
    shape_slow = shape_fast + corr.sl(3, 7)

    # -------------------------------------------------------------------------
    # 3.9  AL-QUWWA AL-MUMAYYIZA -- the discriminating faculty
    # It reads the auditor and the conditions and decides two things: whether
    # the slow path is worth paying for, and how far to lean on the external
    # authority. Neither decision looks at the answer; both look at the
    # situation. That is the whole thesis of the file in eleven inputs.
    # -------------------------------------------------------------------------
    gate_in = cat([cond, rho])
    g = (gate_in @ P["Wg"] + P["bg"]).sigmoid()
    if "slow" in ablate:
        g = T(np.zeros((B, 1)))
    if gate_force is not None:
        # During training the gate is sometimes pinned open and sometimes shut,
        # so that BOTH paths are trained to stand alone and the arbitration in
        # PART 7 test [7] is between two competent faculties, not one crippled one.
        hard = (~np.isnan(gate_force)).astype(float)
        val = np.nan_to_num(gate_force)
        g = g * T(1.0 - hard) + T(val * hard)
    d_mix = d_fast * (T(1.0) - g) + d_slow * g
    size_mix = size_fast * (T(1.0) - g) + size_slow * g
    shape_mix = shape_fast * (T(1.0) - g) + shape_slow * g

    w_auth = ((gate_in @ P["Wa"] + P["ba"]).tanh() @ P["Wa2"] + P["ba2"]).sigmoid()
    if "authority" in ablate:
        w_auth = T(np.zeros((B, 1)))
    d_final = d_mix * (T(1.0) - w_auth) + (auth * 10.0) * w_auth

    return dict(light=light, colour=colour, shape=shape_mix, d=d_final, size=size_mix,
                shape_fast=shape_fast, d_fast=d_fast, size_fast=size_fast,
                shape_slow=shape_slow, d_slow=d_slow, size_slow=size_slow,
                rho=rho, g=g, w_auth=w_auth, cond=cond, delta=delta,
                recon=recon, fused=fused_i, chain=d_chain, a=a)


# =============================================================================
# PART 4 -- THE OBJECTIVE
# =============================================================================

W_SENS, W_SHAPE, W_DIST, W_SIZE = 1.0, 1.0, 1.6, 1.4
W_RECON, W_SHUKUK, W_COST = 0.35, 0.8, 0.20


def loss_fn(P, X, Y, n_refine=2, gate_force=None):
    O = forward(P, X, n_refine=n_refine, gate_force=gate_force)
    B = X["objL"].shape[0]
    onehot = np.zeros((B, 4)); onehot[np.arange(B), Y["kind"]] = 1.0

    L_sens = ((O["light"] - T(Y["light"])) ** 2).mean() + \
             ((O["colour"] - T(Y["colour"])) ** 2).mean()
    L_shape = cross_entropy(O["shape"], T(onehot))
    L_dist = (((O["d"] - T(Y["d"])) / T(Y["d"])) ** 2).mean()
    L_size = ((O["size"] - T(Y["size"])) ** 2).mean()
    L_recon = ((O["recon"] - O["fused"]) ** 2).mean()

    # The auditor is trained on the fast path's OWN mistakes, measured and then
    # detached: it learns to forecast error from conditions, not to cause it.
    e_sens = (np.abs(O["light"].data - Y["light"]) > 0.10).astype(float)
    e_rec = (np.argmax(O["shape_fast"].data, 1) != Y["kind"]).astype(float)[:, None]
    e_inf = (np.abs(O["d_fast"].data - Y["d"]) / np.maximum(Y["d"], 1e-6) > 0.25).astype(float)
    L_shukuk = bce(O["rho"], np.concatenate([e_sens, e_rec, e_inf], 1))

    L_cost = O["g"].mean()

    total = (L_sens * W_SENS + L_shape * W_SHAPE + L_dist * W_DIST + L_size * W_SIZE
             + L_recon * W_RECON + L_shukuk * W_SHUKUK + L_cost * W_COST)
    parts = dict(sens=L_sens.data, shape=L_shape.data, dist=L_dist.data,
                 size=L_size.data, recon=L_recon.data, shukuk=L_shukuk.data,
                 cost=L_cost.data)
    return total, parts, O


# =============================================================================
# PART 5 -- THE MANDATORY FINITE-DIFFERENCE GRADIENT CHECK
# -----------------------------------------------------------------------------
# Nothing in this file is trusted until this passes. It is the one place where
# the machine is made to doubt itself in the way its author demanded.
# =============================================================================

def grad_check(seed=7, n=6, per_tensor=4, eps=3e-5, tol=1e-4):
    rng = np.random.default_rng(seed)
    P = init_params(rng)
    X, Y, _ = make_batch(rng, n)

    for p in P.values():
        p.grad = np.zeros_like(p.data)
    L, _, _ = loss_fn(P, X, Y, n_refine=1)
    L.backward()
    analytic = {k: v.grad.copy() for k, v in P.items()}

    worst, worst_name, checked = 0.0, "", 0
    for name, p in P.items():
        flat = p.data.reshape(-1)
        idxs = rng.choice(flat.size, min(per_tensor, flat.size), replace=False)
        for i in idxs:
            o = flat[i]
            flat[i] = o + eps
            lp, _, _ = loss_fn(P, X, Y, n_refine=1)
            flat[i] = o - eps
            lm, _, _ = loss_fn(P, X, Y, n_refine=1)
            flat[i] = o
            num = (lp.data - lm.data) / (2 * eps)
            ana = analytic[name].reshape(-1)[i]
            rel = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
            checked += 1
            if rel > worst:
                worst, worst_name = rel, f"{name}[{i}]"
    return worst, worst_name, checked, worst < tol


# =============================================================================
# PART 6 -- TRAINING (Adam, written out)
# =============================================================================

def calibrate_conditions(P, rng, n=1500):
    """Learn, from experience, the ordinary spread of each of the eight
    conditions -- so that 'outside the moderate range' has a unit."""
    mu0, sd0 = COND_NORM["mu"].copy(), COND_NORM["sd"].copy()
    COND_NORM["mu"], COND_NORM["sd"] = np.zeros((1, N_COND)), np.ones((1, N_COND))
    X, _, _ = make_batch(rng, n)
    C = forward(P, X, n_refine=1)["cond"].data
    COND_NORM["mu"] = C.mean(0, keepdims=True)
    COND_NORM["sd"] = C.std(0, keepdims=True) + 1e-3
    return mu0, sd0


class Adam:
    def __init__(self, params, lr=4e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.p = params; self.lr = lr; self.b1 = b1; self.b2 = b2; self.eps = eps
        self.m = {k: np.zeros_like(v.data) for k, v in params.items()}
        self.v = {k: np.zeros_like(v.data) for k, v in params.items()}
        self.t = 0

    def step(self):
        self.t += 1
        for k, p in self.p.items():
            g = np.clip(p.grad, -5, 5)
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mh / (np.sqrt(vh) + self.eps)
            p.grad = np.zeros_like(p.data)


def train(P, rng, steps=3000, batch=64, lr=6e-3, log_every=200):
    opt = Adam(P, lr=lr)
    hist = []
    calibrate_conditions(P, np.random.default_rng(5))
    for s in range(1, steps + 1):
        if s % 250 == 0:
            calibrate_conditions(P, np.random.default_rng(5))
        X, Y, _ = make_batch(rng, batch)
        u = rng.random(batch)[:, None]
        gf = np.where(u < 0.22, 1.0, np.where(u < 0.44, 0.0, np.nan))
        L, parts, _ = loss_fn(P, X, Y, gate_force=gf)
        for p in P.values():
            p.grad = np.zeros_like(p.data)
        L.backward()
        opt.lr = lr * (0.5 ** (3.0 * s / steps))       # cosine-free decay
        opt.step()
        hist.append(float(L.data))
        if s % log_every == 0 or s == 1:
            print(f"   step {s:4d} | loss {float(L.data):7.4f} | "
                  + " ".join(f"{k} {v:6.4f}" for k, v in parts.items()))
    return hist


# =============================================================================
# PART 7 -- THE I'TIBAR BATTERY
# -----------------------------------------------------------------------------
# i'tibar is his word for the controlled test: not "observation" but a
# deliberate arrangement in which one thing is varied and the rest held. Each
# experiment below varies exactly one condition.
# =============================================================================

def evaluate(P, rng, n=1200, force=None, ablate=None, n_refine=2):
    X, Y, meta = make_batch(rng, n, force=force)
    O = forward(P, X, n_refine=n_refine, ablate=ablate)
    kind_hat = np.argmax(O["shape"].data, 1)
    rel_d = np.abs(O["d"].data - Y["d"]) / Y["d"]
    return dict(
        shape_acc=float((kind_hat == Y["kind"]).mean()),
        d_rel=float(rel_d.mean()),
        size_mae=float(np.abs(O["size"].data - Y["size"]).mean()),
        light_mae=float(np.abs(O["light"].data - Y["light"]).mean()),
        colour_mae=float(np.abs(O["colour"].data - Y["colour"]).mean()),
        gate=float(O["g"].data.mean()),
        rho=O["rho"].data.mean(0),
        w_auth=float(O["w_auth"].data.mean()),
        O=O, Y=Y, meta=meta, rel_d=rel_d,
    )


def auc(scores, labels):
    """Rank-based AUC; no sklearn."""
    labels = labels.astype(bool)
    if labels.all() or (~labels).all():
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    n1 = labels.sum(); n0 = (~labels).sum()
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def battery(P, rng):
    print("\n" + "=" * 78)
    print(" THE I'TIBAR BATTERY -- eight controlled tests")
    print("=" * 78)
    res = {}

    # -- 1. baseline ----------------------------------------------------------
    base = evaluate(P, rng, 1500)
    print("\n[1] Baseline, all faculties intact")
    print(f"    form recognised      {base['shape_acc']*100:5.1f} %")
    print(f"    distance rel. error  {base['d_rel']*100:5.1f} %")
    print(f"    size MAE             {base['size_mae']:.4f} world units")
    print(f"    light MAE            {base['light_mae']:.4f}   colour MAE {base['colour_mae']:.4f}")
    print(f"    slow path invoked on {base['gate']*100:5.1f} % of the mass")
    res["baseline"] = base

    # -- 2. pure sensation is structurally isolated ---------------------------
    print("\n[2] Ablating the stored forms: does pure sensation survive?")
    ab = evaluate(P, rng, 1500, ablate={"recognition"})
    rng2 = np.random.default_rng(4242)
    a1 = evaluate(P, np.random.default_rng(99), 800)
    a2 = evaluate(P, np.random.default_rng(99), 800, ablate={"recognition"})
    dl = abs(a1["light_mae"] - a2["light_mae"]); dc = abs(a1["colour_mae"] - a2["colour_mae"])
    print(f"    form recognised      {base['shape_acc']*100:5.1f} %  ->  {ab['shape_acc']*100:5.1f} %  (chance 25.0 %)")
    print(f"    light MAE change     {dl:.2e}")
    print(f"    colour MAE change    {dc:.2e}")
    print("    -> light and colour are unmoved to numerical precision: the wall holds.")
    res["ablate_recognition"] = (base["shape_acc"], ab["shape_acc"], dl, dc)

    print("\n[2b] Removing the scan: matching a form that was never normalised")
    sc = evaluate(P, np.random.default_rng(1717), 1500, ablate={"scan"})
    kp = evaluate(P, np.random.default_rng(1717), 1500)
    print(f"    form recognised      {kp['shape_acc']*100:5.1f} %  ->  {sc['shape_acc']*100:5.1f} %")
    res["ablate_scan"] = (kp["shape_acc"], sc["shape_acc"])

    # -- 3. which distance cue is actually load-bearing? ----------------------
    # He denied that distance is read off the angle at the eye and insisted it
    # is measured along the ground. The machine is given BOTH and never told
    # which to prefer. This is the test he would have wanted run against him.
    print("\n[3] Which cue carries distance? (rel. error, by true distance)")
    print(f"    {'cues available':<28}{'all':>8}{'d<7':>8}{'d>11':>9}")
    combos = [("ground chain + disparity", set()),
              ("disparity only  (no chain)", {"chain"}),
              ("ground chain only", {"disparity"}),
              ("neither", {"chain", "disparity"})]
    cue_rows = []
    for lab, ab in combos:
        rr = np.random.default_rng(11)
        Xs, Ys, ms = make_batch(rr, 2000)
        Ok = forward(P, Xs, ablate=ab)
        rel = np.abs(Ok["d"].data - Ys["d"]) / Ys["d"]
        near, far = ms["d"] < 7, ms["d"] > 11
        print(f"    {lab:<28}{rel.mean()*100:7.1f}%{rel[near].mean()*100:7.1f}%{rel[far].mean()*100:8.1f}%")
        cue_rows.append((lab, float(rel.mean()), float(rel[near].mean()), float(rel[far].mean())))
    res["cues"] = cue_rows

    # -- 4. the moon illusion, unprompted -------------------------------------
    # Same object, same angular size, same light. Once with a whole surface
    # running out to it, once over the void. He explained the enlarged horizon
    # moon exactly this way: the ground makes the distance seem greater, and
    # size is judged as angle times distance.
    print("\n[4] The moon illusion -- identical angle, ground present vs absent")
    g_on = evaluate(P, np.random.default_rng(5), 900, force={"ground": 0, "d": 12.0, "size": 2.4})
    g_off = evaluate(P, np.random.default_rng(5), 900, force={"ground": 2, "d": 12.0, "size": 2.4})
    fon, foff = g_on["O"], g_off["O"]
    d1on, d1off = fon["d_fast"].data.mean(), foff["d_fast"].data.mean()
    s1on, s1off = fon["size_fast"].data.mean(), foff["size_fast"].data.mean()
    d2on, d2off = fon["d"].data.mean(), foff["d"].data.mean()
    s2on, s2off = fon["size"].data.mean(), foff["size"].data.mean()
    print(f"    first sight   ground: d {d1on:6.2f} size {s1on:.3f} | void: d {d1off:6.2f} size {s1off:.3f}"
          f"  -> size x{s1on/max(s1off,1e-6):.2f}")
    print(f"    re-examined   ground: d {d2on:6.2f} size {s2on:.3f} | void: d {d2off:6.2f} size {s2off:.3f}"
          f"  -> size x{s2on/max(s2off,1e-6):.2f}")
    print("    (true size 2.400 and true distance 12.0 in both; the object band is\n     identical -- only the ground differs)")
    res["moon"] = (d1on, d1off, s1on, s1off, d2on, d2off, s2on, s2off)

    # -- 5. the eight conditions, walked out of the moderate range one at a time
    print("\n[5] Walking each condition out of its moderate range")
    print(f"    {'condition':<26}{'rel.err':>9}{'slow%':>8}{'rho_inf':>9}")
    sweeps = [
        ("A  distance   near",   {"d": 4.0}),
        ("A  distance   far",    {"d": 36.0}),
        ("B  position   oblique", {"phi": 0.135}),
        ("C  illumination faint", {"lum": 0.12}),
        ("D  size        minute", {"size": 0.28}),
        ("E  opacity     thin",  {"opac": 0.42}),
        ("F  air         turbid", {"trans": 0.47}),
        ("G  duration    brief", {"tau": 0.22}),
        ("H  eye         impaired", {"eye": 0.52}),
    ]
    sweep_rows = []
    for label, f in sweeps:
        r = evaluate(P, np.random.default_rng(21), 700, force=f)
        print(f"    {label:<26}{r['d_rel']*100:8.1f}%{r['gate']*100:7.1f}%{r['rho'][2]:9.3f}")
        sweep_rows.append((label, r["d_rel"], r["gate"], float(r["rho"][2])))
    res["sweeps"] = sweep_rows
    print("\n    the moderate range for distance, measured:")
    print(f"    {'true distance':<16}{'rel.err':>9}{'slow%':>8}")
    dist_rows = []
    for lo in (2, 4, 7, 11, 16, 21):
        hi = {2: 4, 4: 7, 7: 11, 11: 16, 16: 21, 21: 24}[lo]
        rr = np.random.default_rng(31)
        Xs, Ys, ms = make_batch(rr, 900)
        keep = (ms["d"] >= lo) & (ms["d"] < hi)
        if keep.sum() < 30:
            continue
        Xk = {k: v[keep] for k, v in Xs.items()}
        Ok = forward(P, Xk)
        rel = float((np.abs(Ok["d"].data - Ys["d"][keep]) / Ys["d"][keep]).mean())
        print(f"    {f'{lo}-{hi}':<16}{rel*100:8.1f}%{float(Ok['g'].data.mean())*100:7.1f}%")
        dist_rows.append((lo, hi, rel, float(Ok["g"].data.mean())))
    res["envelope"] = dist_rows

    # -- 6. does the auditor actually forecast error? -------------------------
    print("\n[6] Is the auditor calibrated? (AUC of predicted risk vs realised error)")
    r = evaluate(P, np.random.default_rng(33), 2500)
    O, Y = r["O"], r["Y"]
    err_inf = (np.abs(O["d_fast"].data - Y["d"]) / Y["d"] > 0.25).ravel()
    err_rec = (np.argmax(O["shape_fast"].data, 1) != Y["kind"])
    a_inf = auc(O["rho"].data[:, 2], err_inf)
    a_rec = auc(O["rho"].data[:, 1], err_rec)
    a_delta = auc(O["delta"].data.ravel(), err_rec)
    print(f"    rho_inference   vs inference error   AUC {a_inf:.3f}")
    print(f"    rho_recognition vs recognition error AUC {a_rec:.3f}")
    print(f"    binocular disagreement alone         AUC {a_delta:.3f}")
    res["auc"] = (a_inf, a_rec, a_delta)

    # -- 7. the economy of doubt ---------------------------------------------
    # Three regimes: never re-examine, always re-examine, and re-examine when
    # the auditor says the conditions warrant it.
    print("\n[7] The economy of doubt")
    seed = 77
    never = evaluate(P, np.random.default_rng(seed), 1500, ablate={"slow"})
    learned = evaluate(P, np.random.default_rng(seed), 1500)
    Pa = {k: T(v.data.copy()) for k, v in P.items()}
    Pa["bg"].data[:] = 8.0                     # force the gate open: always slow
    always = evaluate(Pa, np.random.default_rng(seed), 1500)
    print(f"    {'regime':<22}{'shape':>8}{'rel.err':>10}{'slow%':>8}")
    for nm, r in (("never re-examine", never), ("learned gate", learned), ("always re-examine", always)):
        print(f"    {nm:<22}{r['shape_acc']*100:7.1f}%{r['d_rel']*100:9.1f}%{r['gate']*100:7.1f}%")
    res["economy"] = (never, learned, always)

    # -- 8. doubts concerning Ptolemy ----------------------------------------
    # An external authority reports the distance. It is right inside the
    # moderate range and biased outside it. Nobody tells the machine which is
    # which. Does reliance track the envelope?
    print("\n[8] Doubts concerning the authority")
    r = evaluate(P, np.random.default_rng(88), 3000)
    inenv = r["meta"]["in_env"]
    w = r["O"]["w_auth"].data.ravel()
    print(f"    reliance inside the moderate range   {w[inenv].mean():.3f}  (n={inenv.sum()})")
    print(f"    reliance outside it                  {w[~inenv].mean():.3f}  (n={(~inenv).sum()})")
    print(f"    ratio                                {w[inenv].mean()/max(w[~inenv].mean(),1e-9):.2f}x")
    gm = r["meta"]["ground"]
    print(f"    reliance, surface whole {w[gm==0].mean():.3f} | broken {w[gm==1].mean():.3f} | absent {w[gm==2].mean():.3f}")
    no_auth = evaluate(P, np.random.default_rng(88), 3000, ablate={"authority"})
    print(f"    distance rel.err with authority {r['d_rel']*100:.1f}%  vs without {no_auth['d_rel']*100:.1f}%")
    res["authority"] = (float(w[inenv].mean()), float(w[~inenv].mean()),
                        r["d_rel"], no_auth["d_rel"])
    return res


# =============================================================================
# MAIN
# =============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(" THE MANAZIR ENGINE -- Ibn al-Haytham (Alhazen), c.965 - c.1040")
    print(" perception as unnoticed inference, audited against a table of conditions")
    print("=" * 78)

    print("\n-- PART 5: finite-difference gradient check ---------------------------")
    worst, name, checked, ok = grad_check()
    print(f"   {checked} parameter entries checked across {len(init_params(np.random.default_rng(0)))} tensors")
    print(f"   worst relative discrepancy {worst:.3e} at {name}")
    print(f"   RESULT: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit("gradient check failed; nothing downstream is trustworthy")

    print("\n-- PART 6: training ---------------------------------------------------")
    rng = np.random.default_rng(1234)
    P = init_params(rng)
    hist = train(P, rng, steps=3000, batch=64)
    print(f"   loss {hist[0]:.4f} -> {np.mean(hist[-20:]):.4f}")

    battery(P, np.random.default_rng(2024))

    print("\n" + "=" * 78)
    print(f" done in {time.time()-t0:.1f} s")
    print("=" * 78)


if __name__ == "__main__":
    main()

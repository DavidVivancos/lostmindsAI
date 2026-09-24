#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
========================
AL-JAWHARATAYN  --  THE NON-TRANSMUTING REFINERY
A crucible-neuron architecture after Abu Muhammad al-Hasan b. Ahmad al-Hamdani
(b. Sana'a 280/893; d. after 334/945 -- after 360/970 per Robin 2021)
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 12 Minds 221 - 240 Available on Amazon https://www.amazon.com/dp/B0HKVBKNTT
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0237_al_hamdani_893 - Abu Muhammad al-Hasan b. Ahmad al-Hamdani (b. Sana'a 280/893; d. after 334/945 -- after 360/970 per Robin 2021)
================================================================================  

WHAT THIS FILE IS
-----------------
A complete, trainable, pure-NumPy architecture (its own reverse-mode autodiff,
no ML framework) whose inductive biases come from the one cognitive commitment
that is demonstrably al-Hamdani's own, stated in his Kitab al-Jawharatayn
(ed./trans. C. Toll, Uppsala 1968) and summarised by Toll in the Dictionary of
Scientific Biography: gold is derived from gold ore and silver from silver ore,
never from another metal; the noble is not MADE, it is PARTED from the base by
fire, measured against a standard of fineness, and only then coined.

Translated into machine cognition:

  1. NON-TRANSMUTATION (a hard conservation law, not a regulariser).
     Every unit of output "mass" is a partition of input evidence mass.
     No bias term, no additive generator: a vessel cannot hold what the ore
     did not contain.  Verified to machine precision in the self-tests.

  2. CRUCIBLE NEURONS (buta).  A crucible neuron holds K vessels, each with a
     learned STANDARD (a streak distribution over input channels).  Each
     refining round splits every grain's mass among the vessels in proportion
     to standard x current vessel mass, raised to a temperature set by the
     FIRE.  This is tempered EM for a mixture of multinomials, used here as a
     neuron: the round is the furnace, the responsibilities are the parting.

  3. STAGED REFINING, as in his mint chapter: crush (column-stochastic
     comminution) -> smelt (bullion vs matte/earth/fume) -> cupel (gold,
     silver, litharge).  Mass flows only forward and only by partition.

  4. THE FIRE MAY BE SET BY PLACE; THE VERDICT MAY NOT.
     His connected world picture makes one celestial-climatic field of four
     qualities mature the ores, climates and peoples of the Peninsula.  The
     architecture lets site context (latitude, altitude, distance from the
     coast, season) choose only the TEMPERATURE of each crucible -- how hard
     to fire -- never which vessel a grain belongs to.  Priors may route
     attention (the prospector, Section 9); evidence decides the verdict.

  5. THE TOUCHSTONE (mihakk) AND THE MINT (sikka).  The assay is computed
     from the vessels themselves (fineness) plus a streak-mismatch test of
     the claimed composition against the standards.  The mint stamps only at
     standard, can send a melt back for hotter rounds, and otherwise refuses.

  6. HIS FAILURE, BUILT IN AS A TEST.  Robin (JSAI 51, 2021) shows the same
     man misread South Arabian inscriptions -- confusing letters of similar
     shape -- and promoted the names to kings to lengthen Yemeni genealogies
     against northern genealogists.  Rigour in the mint, transmutation in the
     genealogy: rigour failed where he had a STAKE.  Section 8 fine-tunes
     every model under an explicit stake ("our mines are richer") and measures
     how each one transmutes, and whether a stake-blind touchstone sees it.

CONTENTS
--------
  S0  numerics and a tiny reverse-mode autodiff (Tensor tape)
  S1  the Peninsula: a synthetic world of ores matured by one field
  S2  crucible neuron, fire network, refinery (+ leaky-crucible ablation)
  S3  the Alchemist: an unconstrained MLP baseline (transmutation allowed)
  S4  training loop (Adam, fresh ore every step), metrics, true-composition
      ledger (the physical fire assay, available only to the simulator)
  S5  finite-difference gradient check (mandatory)
  S6  Experiment A -- refining accuracy, recovery and parting purity
  S7  Experiment B -- the barren-vein trap, context sensitivity, unknown metal
  S8  Experiment C -- stake-gated transmutation (the Iklil test)
  S9  Experiment D -- the prospector: priors that route but do not decide
  S10 Experiment E -- recoining and debasement at the mint
  S11 Experiment F -- confusable standards (the musnad letters)
  S12 Experiment G -- the Tongue of Yemen (Stein's catch-all 'Himyaritic')
  S13 self-tests and JSON summary

Run:   python3 chapter_0237_al_hamdani_893.py            (full, a few minutes)
       python3 chapter_0237_al_hamdani_893.py --quick    (gradient check + smoke)
"""

import argparse
import json
import math
import sys
import time

import numpy as np

np.set_printoptions(precision=4, suppress=True)
_EPS = 1e-12
T0 = time.time()


def log(msg=""):
    print(msg, flush=True)


# =============================================================================
# S0  AUTODIFF -- a minimal reverse-mode tape over NumPy arrays
# =============================================================================
def _unbroadcast(g, shape):
    """Sum a broadcast gradient back to the shape of the operand."""
    if g.shape == shape:
        return g
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for ax, s in enumerate(shape):
        if s == 1 and g.shape[ax] != 1:
            g = g.sum(axis=ax, keepdims=True)
    return g


class Tensor:
    """A node on the tape: value, gradient, parents and a backward closure."""
    __slots__ = ("data", "grad", "parents", "backfn", "requires_grad", "name")

    def __init__(self, data, parents=(), requires_grad=False, name=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = None
        self.parents = parents
        self.backfn = None
        self.requires_grad = bool(requires_grad) or any(p.requires_grad for p in parents)
        self.name = name

    @property
    def shape(self):
        return self.data.shape

    # operator sugar ---------------------------------------------------------
    def __add__(self, o): return add(self, o)
    def __radd__(self, o): return add(o, self)
    def __sub__(self, o): return sub(self, o)
    def __rsub__(self, o): return sub(o, self)
    def __mul__(self, o): return mul(self, o)
    def __rmul__(self, o): return mul(o, self)
    def __truediv__(self, o): return div(self, o)
    def __rtruediv__(self, o): return div(o, self)
    def __neg__(self): return neg(self)
    def __matmul__(self, o): return matmul(self, o)

    def backward(self):
        """Iterative post-order topological sort, then reverse accumulation."""
        order, seen, stack = [], set(), [(self, False)]
        while stack:
            node, done = stack.pop()
            if done:
                order.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for p in node.parents:
                if p.requires_grad and id(p) not in seen:
                    stack.append((p, False))
        self.grad = np.ones_like(self.data)
        for node in reversed(order):
            if node.backfn is not None and node.grad is not None:
                node.backfn(node.grad)


def _t(x):
    return x if isinstance(x, Tensor) else Tensor(x)


def _acc(t, g):
    if not t.requires_grad:
        return
    g = _unbroadcast(np.asarray(g), t.data.shape)
    if t.grad is None:
        t.grad = np.array(g, dtype=np.float64, copy=True)
    else:
        t.grad = t.grad + g


def param(x, name=""):
    return Tensor(np.array(x, dtype=np.float64), requires_grad=True, name=name)


def add(a, b):
    a, b = _t(a), _t(b)
    out = Tensor(a.data + b.data, (a, b))
    def bf(g):
        _acc(a, g); _acc(b, g)
    out.backfn = bf
    return out


def sub(a, b):
    a, b = _t(a), _t(b)
    out = Tensor(a.data - b.data, (a, b))
    def bf(g):
        _acc(a, g); _acc(b, -g)
    out.backfn = bf
    return out


def mul(a, b):
    a, b = _t(a), _t(b)
    out = Tensor(a.data * b.data, (a, b))
    def bf(g):
        _acc(a, g * b.data); _acc(b, g * a.data)
    out.backfn = bf
    return out


def div(a, b):
    a, b = _t(a), _t(b)
    out = Tensor(a.data / b.data, (a, b))
    def bf(g):
        _acc(a, g / b.data); _acc(b, -g * a.data / (b.data ** 2))
    out.backfn = bf
    return out


def neg(a):
    out = Tensor(-a.data, (a,))
    out.backfn = lambda g: _acc(a, -g)
    return out


def matmul(a, b):
    a, b = _t(a), _t(b)
    out = Tensor(a.data @ b.data, (a, b))
    def bf(g):
        _acc(a, g @ b.data.T); _acc(b, a.data.T @ g)
    out.backfn = bf
    return out


def transpose(a):
    out = Tensor(a.data.T, (a,))
    out.backfn = lambda g: _acc(a, g.T)
    return out


def exp(a):
    e = np.exp(a.data)
    out = Tensor(e, (a,))
    out.backfn = lambda g: _acc(a, g * e)
    return out


def tlog(a):
    out = Tensor(np.log(a.data), (a,))
    out.backfn = lambda g: _acc(a, g / a.data)
    return out


def tanh(a):
    t = np.tanh(a.data)
    out = Tensor(t, (a,))
    out.backfn = lambda g: _acc(a, g * (1.0 - t * t))
    return out


def _sig(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def sigmoid(a):
    s = _sig(a.data)
    out = Tensor(s, (a,))
    out.backfn = lambda g: _acc(a, g * s * (1.0 - s))
    return out


def softplus(a):
    out = Tensor(np.logaddexp(0.0, a.data), (a,))
    s = _sig(a.data)
    out.backfn = lambda g: _acc(a, g * s)
    return out


def square(a):
    out = Tensor(a.data * a.data, (a,))
    out.backfn = lambda g: _acc(a, 2.0 * g * a.data)
    return out


def tsum(a, axis=None, keepdims=False):
    out = Tensor(a.data.sum(axis=axis, keepdims=keepdims), (a,))
    def bf(g):
        if axis is None:
            _acc(a, np.broadcast_to(g, a.data.shape))
        else:
            gg = g if keepdims else np.expand_dims(g, axis)
            _acc(a, np.broadcast_to(gg, a.data.shape))
    out.backfn = bf
    return out


def tmean(a, axis=None):
    n = a.data.size if axis is None else a.data.shape[axis]
    return tsum(a, axis=axis) * (1.0 / n)


def softmax(a, axis=-1):
    z = a.data - a.data.max(axis=axis, keepdims=True)
    e = np.exp(z)
    s = e / e.sum(axis=axis, keepdims=True)
    out = Tensor(s, (a,))
    out.backfn = lambda g: _acc(a, s * (g - (g * s).sum(axis=axis, keepdims=True)))
    return out


def log_softmax(a, axis=-1):
    m = a.data.max(axis=axis, keepdims=True)
    ls = a.data - (m + np.log(np.exp(a.data - m).sum(axis=axis, keepdims=True)))
    out = Tensor(ls, (a,))
    def bf(g):
        _acc(a, g - np.exp(ls) * g.sum(axis=axis, keepdims=True))
    out.backfn = bf
    return out


def reshape(a, shape):
    out = Tensor(a.data.reshape(shape), (a,))
    out.backfn = lambda g: _acc(a, g.reshape(a.data.shape))
    return out


def col(a, j):
    """Select index j on the last axis."""
    out = Tensor(a.data[..., j], (a,))
    def bf(g):
        z = np.zeros_like(a.data)
        z[..., j] = g
        _acc(a, z)
    out.backfn = bf
    return out


def concat(ts, axis=-1):
    ts = [_t(t) for t in ts]
    out = Tensor(np.concatenate([t.data for t in ts], axis=axis), tuple(ts))
    sizes = [t.data.shape[axis] for t in ts]
    def bf(g):
        cuts = np.cumsum(sizes)[:-1]
        for t, gi in zip(ts, np.split(g, cuts, axis=axis)):
            _acc(t, gi)
    out.backfn = bf
    return out


class Module:
    def params(self):
        out = []
        for v in self.__dict__.values():
            if isinstance(v, Tensor) and v.requires_grad:
                out.append(v)
            elif isinstance(v, Module):
                out.extend(v.params())
        return out

    def state(self):
        return [p.data.copy() for p in self.params()]

    def load(self, st):
        for p, s in zip(self.params(), st):
            p.data = s.copy()


class Adam:
    def __init__(self, params, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8, clip=5.0, frozen=()):
        self.p = [q for q in params if all(q is not f for f in frozen)]
        self.lr, self.b1, self.b2, self.eps, self.clip = lr, b1, b2, eps, clip
        self.m = [np.zeros_like(q.data) for q in self.p]
        self.v = [np.zeros_like(q.data) for q in self.p]
        self.t = 0

    def zero(self, allp):
        for q in allp:
            q.grad = None

    def step(self):
        self.t += 1
        grads = [q.grad if q.grad is not None else np.zeros_like(q.data) for q in self.p]
        norm = math.sqrt(sum(float((g * g).sum()) for g in grads))
        scale = min(1.0, self.clip / (norm + 1e-12))
        for i, (q, g) in enumerate(zip(self.p, grads)):
            g = g * scale
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            q.data = q.data - self.lr * mh / (np.sqrt(vh) + self.eps)


# =============================================================================
# S1  THE PENINSULA -- one celestial-climatic field matures every ore
# =============================================================================
SUB = ["gold", "silver", "lead", "copper", "earth", "sulfur"]
AU, AG, PB, CU, FE, SU = range(6)


def _norm(v):
    v = np.maximum(v, 0.0)
    return v / v.sum()


class Peninsula:
    """
    A synthetic Arabia.  Each substance leaves a streak signature over D
    observation channels (columns of W sum to one, so observed mass equals
    substance mass).  A fixed field -- the analogue of al-Hamdani's heavens
    acting on the four qualities -- sets how hot and how moist a site is, and
    maturation of gold follows hot x dry.  Copper co-occurs with gold country,
    so a model that learns the STEREOTYPE of a place can find gold where the
    ore has none.  That is the trap of Experiment B.
    """

    def __init__(self, seed=237, D=24, silver_lead_mix=0.35, gold_copper_mix=0.35):
        rng = np.random.default_rng(seed)
        self.D = D
        base = [_norm(rng.gamma(0.35, 1.0, D) + 1e-4) for _ in range(8)]
        W = np.zeros((D, 6))
        W[:, AU] = _norm((1 - gold_copper_mix) * base[0] + gold_copper_mix * base[3])
        W[:, AG] = _norm((1 - silver_lead_mix) * base[1] + silver_lead_mix * base[2])
        W[:, PB] = base[2]
        W[:, CU] = base[3]
        W[:, FE] = _norm(0.6 * base[4] + 0.4 * np.ones(D) / D)
        W[:, SU] = base[5]
        self.W = W
        self.W_tin = base[6]                     # a metal the standards have never seen
        self.gold_ref = W[:, AU]
        self.silver_ref = W[:, AG]

    @staticmethod
    def field(lat, alt, coast, phase):
        """Hot and moist qualities of a site (cold = 1-hot, dry = 1-moist)."""
        hot = _sig((22.0 - lat) / 2.5) * (1.0 - 0.15 * alt) * (0.9 + 0.1 * np.cos(phase))
        moist = np.clip(0.62 * np.exp(-3.0 * coast) + 0.11 * alt + 0.08 * np.sin(phase), 0.0, 1.0)
        return hot, moist

    def sites(self, rng, n, region=None):
        lat = rng.uniform(12.5, 29.5, n)
        alt = rng.uniform(0.0, 3.2, n)
        coast = rng.uniform(0.0, 1.0, n)
        phase = rng.uniform(0.0, 2 * np.pi, n)
        if region == "hamdan":                    # northern highlands, Hamdan country
            lat = rng.uniform(15.0, 17.5, n)
            alt = rng.uniform(1.8, 3.2, n)
            coast = rng.uniform(0.35, 0.8, n)
        s = dict(lat=lat, alt=alt, coast=coast, phase=phase)
        if region == "gold_country":              # places whose qualities promise gold
            keep = {k: [] for k in s}
            while len(keep["lat"]) < n:
                c = self.sites(rng, 4 * n)
                hot, moist = self.field(c["lat"], c["alt"], c["coast"], c["phase"])
                m = hot * (1 - moist) > 0.42
                for k in s:
                    keep[k].extend(list(c[k][m]))
            s = {k: np.array(v[:n]) for k, v in keep.items()}
        return s

    @staticmethod
    def ctx(s):
        return np.stack([(s["lat"] - 21.0) / 5.0, (s["alt"] - 1.6) / 1.0,
                         (s["coast"] - 0.5) / 0.3, np.sin(s["phase"]), np.cos(s["phase"])], 1)

    def ores(self, rng, s, barren=False, tin_frac=0.0):
        hot, moist = self.field(s["lat"], s["alt"], s["coast"], s["phase"])
        mat = hot * (1.0 - moist)
        n = len(hot)
        vein = (rng.random(n) < _sig(9.0 * (mat - 0.30))).astype(float)
        if barren:
            vein[:] = 0.0
        c = np.zeros((n, 6))
        c[:, AU] = vein * rng.gamma(2.0, 0.05, n) * (0.6 + mat)
        c[:, CU] = rng.gamma(2.0, 0.07, n) * (0.3 + 1.2 * mat)
        c[:, PB] = rng.gamma(2.0, 0.10, n) * (0.4 + moist)
        ag_on = (rng.random(n) < _sig(6.0 * (moist - 0.35))).astype(float)
        c[:, AG] = ag_on * c[:, PB] * rng.uniform(0.10, 0.35, n)
        c[:, FE] = rng.gamma(3.0, 0.22, n)
        c[:, SU] = rng.gamma(2.0, 0.05, n) * (0.5 + hot)
        X = c @ self.W.T
        tin = np.zeros(n)
        if tin_frac > 0:
            tin = tin_frac * c.sum(1)
            X = X + np.outer(tin, self.W_tin)
        X = X + rng.gamma(1.0, 0.002, X.shape)   # streak noise, non-negative
        return X, c, tin


# =============================================================================
# S2  CRUCIBLE NEURON, FIRE NETWORK, REFINERY
# =============================================================================
class FireNet(Module):
    """Place -> temperature of each crucible.  It can only set HOW HARD to fire."""

    def __init__(self, rng, n_in=5, hid=12, n_out=2, floor=0.05):
        self.W1 = param(rng.normal(0, 0.3, (hid, n_in)))
        self.b1 = param(np.zeros(hid))
        self.W2 = param(rng.normal(0, 0.05, (n_out, hid)))
        self.b2 = param(np.full(n_out, math.log(math.expm1(max(1e-3, 1.0 - floor)))))
        self.floor = floor

    def __call__(self, C):
        h = tanh(matmul(C, transpose(self.W1)) + self.b1)
        return softplus(matmul(h, transpose(self.W2)) + self.b2) + self.floor


class Crucible(Module):
    """
    BUTA -- the crucible neuron.
    State: K vessel masses h (N x K).  Parameters: standards S (D x K, each
    column a streak distribution) and a context-free prior share.
    One firing round:  r[n,d,k] = softmax_k( tau_n * (log S[d,k] + log h[n,k]) )
                       h[n,k]   = sum_d Y[n,d] * r[n,d,k]
    Because r sums to one over vessels, sum_k h = sum_d Y exactly: the neuron
    parts mass and cannot create it.  normalize=False gives the LEAKY crucible
    (independent sigmoid gates) used as an ablation: it may duplicate mass.
    """

    def __init__(self, rng, D, K, init_cols=None, normalize=True):
        S = rng.normal(0.0, 0.5, (D, K))
        if init_cols:
            for k, v in init_cols.items():
                S[:, k] = np.log(np.maximum(v, 1e-3))
        self.S_logit = param(S)
        self.prior = param(np.zeros(K))
        self.normalize = normalize
        self.K, self.D = K, D
        if not normalize:
            self.leak = param(np.full(K, 4.0))

    def standards(self):
        return softmax(self.S_logit, axis=0)

    def __call__(self, Y, tau, rounds):
        N, D = Y.shape
        K = self.K
        M = tsum(Y, axis=1, keepdims=True)
        logS3 = reshape(log_softmax(self.S_logit, axis=0), (1, D, K))
        h = M * reshape(softmax(self.prior, axis=0), (1, K))
        Y3 = reshape(Y, (N, D, 1))
        tau3 = reshape(tau, (N, 1, 1))
        r = None
        for _ in range(rounds):
            logh3 = reshape(tlog(h + _EPS), (N, 1, K))
            L = tau3 * (logS3 + logh3)
            if self.normalize:
                r = softmax(L, axis=2)
            else:
                r = sigmoid(L + reshape(self.leak, (1, 1, K)))
            h = tsum(Y3 * r, axis=1)
        return h, r


def touchstone_kl(Y, h, S):
    """Streak mismatch KL(observed grains || grains implied by the claimed vessels)."""
    M = tsum(Y, axis=1, keepdims=True) + _EPS
    p = Y / M
    phat = matmul(h, transpose(S)) / M
    return tsum(p * (tlog(p + _EPS) - tlog(phat + _EPS)), axis=1)


class Refinery(Module):
    """
    Crush -> smelt (K1 vessels; vessel 0 = bullion) -> cupel (K2 vessels;
    0 = gold, 1 = silver, 2.. = litharge).  Standards for gold and silver are
    initialised from reference streaks (the assayer's touch-needles); all
    other vessels start unnamed.
    """

    def __init__(self, rng, world, K1=5, K2=3, rounds=(5, 5), floor=0.05, normalize=True, needle_noise=0.3):
        D = world.D
        self.D = D
        self.rounds = rounds
        self.A_logit = param(8.0 * np.eye(D) + 0.01 * rng.normal(size=(D, D)))
        g = world.gold_ref * np.exp(needle_noise * rng.normal(size=D))
        s = world.silver_ref * np.exp(needle_noise * rng.normal(size=D))
        self.smelt = Crucible(rng, D, K1, {0: _norm(g) + _norm(s) + 0.02}, normalize)
        self.cupel = Crucible(rng, D, K2, {0: _norm(g) + 0.01, 1: _norm(s) + 0.01}, normalize)
        self.fire = FireNet(rng, 5, 12, 2, floor)

    def __call__(self, X, C, rounds=None, tau_scale=1.0):
        R1, R2 = rounds if rounds else self.rounds
        Xt, Ct = Tensor(X), Tensor(C)
        A = softmax(self.A_logit, axis=0)
        Y = matmul(Xt, transpose(A))
        tau = self.fire(Ct) * tau_scale
        h1, r1 = self.smelt(Y, col(tau, 0), R1)
        Y2 = Y * col(r1, 0)
        h2, r2 = self.cupel(Y2, col(tau, 1), R2)
        kl = touchstone_kl(Y, h1, self.smelt.standards()) + touchstone_kl(Y2, h2, self.cupel.standards())
        return dict(gold=col(h2, 0), silver=col(h2, 1), h1=h1, h2=h2, r1=r1, r2=r2,
                    Y=Y, Y2=Y2, A=A, tau=tau, kl=kl, M=tsum(Y, axis=1))


# =============================================================================
# S3  THE ALCHEMIST -- an unconstrained network that may make gold
# =============================================================================
class Alchemist(Module):
    def __init__(self, rng, D, H=64):
        n_in = D + 1 + 5
        self.W1 = param(rng.normal(0, 1.0 / math.sqrt(n_in), (H, n_in)))
        self.b1 = param(np.zeros(H))
        self.W2 = param(rng.normal(0, 1.0 / math.sqrt(H), (H, H)))
        self.b2 = param(np.zeros(H))
        self.W3 = param(rng.normal(0, 0.1 / math.sqrt(H), (2, H)))
        self.b3 = param(np.array([-2.0, -3.0]))

    def __call__(self, X, C, **_):
        M = X.sum(1, keepdims=True)
        f = np.concatenate([X / M, np.log(M), C], 1)
        h = tanh(matmul(Tensor(f), transpose(self.W1)) + self.b1)
        h = tanh(matmul(h, transpose(self.W2)) + self.b2)
        z = softplus(matmul(h, transpose(self.W3)) + self.b3)
        out = z * Tensor(M * 0.25)
        return dict(gold=col(out, 0), silver=col(out, 1), M=Tensor(M[:, 0]))


# =============================================================================
# S4  TRAINING, METRICS, AND THE PHYSICAL LEDGER
# =============================================================================
SCALE = np.array([0.08, 0.03])     # target scales (gold, silver) for a balanced loss


def task_loss(out, c):
    g = (out["gold"] - c[:, AU]) * (1.0 / SCALE[0])
    s = (out["silver"] - c[:, AG]) * (1.0 / SCALE[1])
    return tmean(square(g) + square(s))


def model_loss(model, out, c, alpha=0.5):
    L = task_loss(out, c)
    if "kl" in out:
        L = L + alpha * tmean(out["kl"])
    return L


def batch(world, rng, n, region=None, **kw):
    s = world.sites(rng, n, region)
    X, c, tin = world.ores(rng, s, **kw)
    return X, world.ctx(s), c, s, tin


def train(model, world, steps, lr, seed, n=256, extra=None, frozen=(), tag=""):
    rng = np.random.default_rng(seed)
    opt = Adam(model.params(), lr=lr, frozen=frozen)
    hist = []
    for step in range(1, steps + 1):
        X, C, c, s, _ = batch(world, rng, n)
        out = model(X, C)
        L = model_loss(model, out, c)
        if extra is not None:
            L = L + extra(model, rng)
        opt.zero(model.params())
        L.backward()
        opt.step()
        hist.append(float(L.data))
        if step == 1 or step % max(1, steps // 5) == 0:
            log(f"    [{tag}] step {step:4d}/{steps}  loss {np.mean(hist[-20:]):.4f}")
    return hist


def r2(y, yhat):
    return 1.0 - np.sum((y - yhat) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-12)


def auc(neg_scores, pos_scores):
    """Probability a positive scores above a negative (ties count half)."""
    neg_scores, pos_scores = np.asarray(neg_scores), np.asarray(pos_scores)
    allv = np.concatenate([neg_scores, pos_scores])
    ranks = allv.argsort().argsort().astype(float) + 1.0
    # average ranks for ties
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    rp = ranks[len(neg_scores):].sum()
    npos, nneg = len(pos_scores), len(neg_scores)
    return (rp - npos * (npos + 1) / 2.0) / (npos * nneg)


def _selective(r, Y, G, target):
    """
    Selective extraction.  A vessel draws the fraction r[n,d] of grain d's
    mass.  The reagent binds its target substances first: up to their true
    share of the grain it takes only target atoms; anything beyond that
    share is dragged-in impurity, drawn proportionally from the rest.
    Y: N x D grain masses; G: N x D x S true substance masses per grain.
    Returns E: N x D x S true substance masses inside the vessel.
    """
    F = G / np.maximum(Y[:, :, None], 1e-15)
    tmask = np.zeros(G.shape[2], bool)
    tmask[list(target)] = True
    fT = F[:, :, tmask].sum(2)
    take = np.minimum(r, fT)
    excess = np.maximum(r - fT, 0.0)
    E = np.zeros_like(G)
    E[:, :, tmask] = (take / np.maximum(fT, 1e-15))[:, :, None] * F[:, :, tmask]
    rest = np.maximum(1.0 - fT, 1e-15)
    E[:, :, ~tmask] = (excess / rest)[:, :, None] * F[:, :, ~tmask]
    return E * Y[:, :, None]


def true_vessel_composition(world, model, X, C, c, tin=None, rounds=None, tau_scale=1.0):
    """
    The simulator's fire assay (never visible to the model).  Substance k in
    crushed grain d is (A W)[d,k] c[n,k]; bullion selectively collects gold,
    silver and lead; the cupel's gold and silver vessels selectively collect
    their own metal.  Returns true substance masses in the gold and silver
    vessels (N x S; S = 6, or 7 when tin is present).
    """
    out = model(X, C, rounds=rounds, tau_scale=tau_scale)
    A = out["A"].data
    G = np.einsum("dk,nk->ndk", A @ world.W, c)
    if tin is not None and np.any(tin > 0):
        Gt = np.einsum("d,n->nd", A @ world.W_tin, tin)[:, :, None]
        G = np.concatenate([G, Gt], 2)
    Y = out["Y"].data
    # grain totals in the ledger include streak noise; fold it into earth
    noise = np.maximum(Y - G.sum(2), 0.0)
    G[:, :, FE] += noise
    E1 = _selective(out["r1"].data[:, :, 0], Y, G, (AU, AG, PB))
    Y2 = out["Y2"].data
    gold = _selective(out["r2"].data[:, :, 0], Y2, E1, (AU,)).sum(1)
    silver = _selective(out["r2"].data[:, :, 1], Y2, E1, (AG,)).sum(1)
    return out, gold, silver


def conservation_error(out):
    """|sum of all vessels, both stages - ore mass| after crushing."""
    M = out["Y"].data.sum(1)
    h1 = out["h1"].data.sum(1)
    h2 = out["h2"].data.sum(1)
    bull = out["h1"].data[:, 0]
    e1 = np.abs(h1 - M).max()
    e2 = np.abs(h2 - bull).max()
    return max(e1, e2)


# =============================================================================
# S5  FINITE-DIFFERENCE GRADIENT CHECK (mandatory)
# =============================================================================
def gradient_check(seed=7):
    rng = np.random.default_rng(seed)
    world = Peninsula(seed=seed, D=6)
    X, C, c, s, _ = batch(world, rng, 5)
    worst = {}
    models = {
        "refinery": Refinery(rng, world, K1=3, K2=3, rounds=(2, 2)),
        "leaky": Refinery(rng, world, K1=3, K2=3, rounds=(2, 2), normalize=False),
        "alchemist": Alchemist(rng, 6, H=5),
    }
    for name, m in models.items():
        # jitter parameters so the check is not at a symmetric point
        for p in m.params():
            p.data = p.data + 0.05 * rng.normal(size=p.data.shape)

        def f():
            return float(model_loss(m, m(X, C), c).data)

        L = model_loss(m, m(X, C), c)
        for p in m.params():
            p.grad = None
        L.backward()
        mx = 0.0
        for p in m.params():
            ga = p.grad.copy()
            flat = p.data.reshape(-1)
            idx = range(flat.size) if flat.size <= 40 else rng.choice(flat.size, 40, replace=False)
            for i in idx:
                old = flat[i]
                hstep = 1e-5 * max(1.0, abs(old))
                flat[i] = old + hstep; fp = f()
                flat[i] = old - hstep; fm = f()
                flat[i] = old
                gn = (fp - fm) / (2 * hstep)
                a = ga.reshape(-1)[i]
                rel = abs(a - gn) / max(1e-6, abs(a) + abs(gn))
                if abs(a - gn) < 1e-8:
                    rel = 0.0
                mx = max(mx, rel)
        worst[name] = mx
        log(f"  gradient check [{name:9s}]  max relative error = {mx:.3e}")
    return worst


# =============================================================================
# S6..S12  EXPERIMENTS
# =============================================================================
def evaluate_accuracy(world, model, X, C, c, is_ref):
    out = model(X, C)
    g, sv = out["gold"].data, out["silver"].data
    res = dict(gold_r2=r2(c[:, AU], g), silver_r2=r2(c[:, AG], sv))
    M = X.sum(1)
    fine_true = (c[:, AU] + c[:, AG]) / M
    fine_hat = (g + sv) / M
    res["fineness_mae"] = float(np.mean(np.abs(fine_true - fine_hat)))
    if is_ref:
        _, gt, st = true_vessel_composition(world, model, X, C, c)
        gold_in_gold = gt[:, AU].sum()
        res["gold_recovery"] = float(gold_in_gold / c[:, AU].sum())
        res["gold_parting_purity"] = float(gold_in_gold / g.sum())
        res["silver_recovery"] = float(st[:, AG].sum() / c[:, AG].sum())
        res["silver_parting_purity"] = float(st[:, AG].sum() / sv.sum())
        res["conservation_err"] = float(conservation_error(out))
        res["min_vessel_mass"] = float(min(out["h1"].data.min(), out["h2"].data.min()))
        res["mean_tau"] = [float(x) for x in out["tau"].data.mean(0)]
    return res


def experiment_barren(world, models, rng):
    """Gold country, veins barren: every gram the model reports is invented."""
    s = world.sites(rng, 800, "gold_country")
    X, c, _ = world.ores(rng, s, barren=True)
    C = world.ctx(s)
    res = {}
    for name, m in models.items():
        g = m(X, C)["gold"].data
        frac = g / X.sum(1)
        res[name] = dict(mean_gold_pct=float(100 * frac.mean()),
                         false_gold_rate=float((frac > 0.02).mean()))
    # context sensitivity: the same ores under 40 random skies
    Xs, cs = X[:60], c[:60]
    for name, m in models.items():
        preds = []
        for k in range(40):
            s2 = world.sites(rng, 60)
            preds.append(m(Xs, world.ctx(s2))["gold"].data)
        P = np.array(preds)
        res[name]["context_sensitivity"] = float(P.std(0).mean() / (Xs.sum(1).mean() * 0.01))
    # the unknown metal: tin the standards have never seen
    s3 = world.sites(rng, 600)
    Xk, ck, _ = world.ores(rng, s3)
    Xu, cu, tinu = world.ores(rng, s3, tin_frac=0.45)
    ref = models["refinery"]
    kl_known = ref(Xk, world.ctx(s3))["kl"].data
    out_u = ref(Xu, world.ctx(s3))
    res["unknown_metal_auc"] = float(auc(kl_known, out_u["kl"].data))
    res["unknown_metal_gold_err_refinery"] = float(np.mean(np.abs(out_u["gold"].data - cu[:, AU])) / 0.01)
    res["unknown_metal_gold_err_alchemist"] = float(np.mean(np.abs(models["alchemist"](Xu, world.ctx(s3))["gold"].data - cu[:, AU])) / 0.01)
    return res


def experiment_stake(world, trained, seed, steps, lam=0.6):
    """
    The Iklil test.  Fine-tune under the stake 'the mines of Hamdan are richer':
    task loss everywhere, minus lam * reported gold share on Hamdan sites.
    Evaluate on BARREN Hamdan ores (truth: zero gold).
    A stake-blind touchstone -- standards frozen before the stake -- judges
    whether the claimed composition still matches the ore.
    """
    rng = np.random.default_rng(seed)
    s = world.sites(rng, 700, "hamdan")
    Xb, cb, _ = world.ores(rng, s, barren=True)
    Cb = world.ctx(s)
    res = {}

    def stake_term(model, r):
        Xh, Ch, ch, sh, _ = batch(world, r, 128, region="hamdan")
        o = model(Xh, Ch)
        return -lam * tmean(o["gold"] / Tensor(Xh.sum(1))) * 25.0

    variants = [("alchemist", "alchemist", None, False),
                ("refinery_trainable_standards", "refinery", None, False),
                ("refinery_frozen_standards", "refinery", "freeze", False),
                ("refinery_fire_floor", "refinery", "freeze", True)]
    for label, base, mode, floor in variants:
        mdl, st = trained[base]
        m = mdl
        m.load(st)
        before = m(Xb, Cb)
        g_before = before["gold"].data / Xb.sum(1)
        frozen_S = None
        if "kl" in before:
            frozen_S = (m.smelt.standards().data.copy(), m.cupel.standards().data.copy(), m.A_logit.data.copy())
        old_floor = None
        if floor:
            old_floor = m.fire.floor
            m.fire.floor = 0.9
        frozen = ()
        if mode == "freeze":
            frozen = (m.smelt.S_logit, m.cupel.S_logit, m.A_logit, m.smelt.prior, m.cupel.prior)
        log(f"  stake fine-tune: {label}")
        train(m, world, steps, lr=0.01 if base == "refinery" else 0.002, seed=seed + 11,
              n=192, extra=stake_term, frozen=frozen, tag=label[:14])
        after = m(Xb, Cb)
        g_after = after["gold"].data / Xb.sum(1)
        row = dict(barren_gold_pct_before=float(100 * g_before.mean()),
                   barren_gold_pct_after=float(100 * g_after.mean()))
        # accuracy away from Hamdan (did the stake stay local?)
        r2rng = np.random.default_rng(seed + 99)
        Xo, Co, co, so, _ = batch(world, r2rng, 600)
        row["gold_r2_elsewhere_after"] = float(r2(co[:, AU], m(Xo, Co)["gold"].data))
        if frozen_S is not None:
            S0, S20, A0 = frozen_S
            A0s = np.exp(A0 - A0.max(0)); A0s = A0s / A0s.sum(0)
            Yf = Xb @ A0s.T

            def kl_np(Y, h, S):
                M = Y.sum(1, keepdims=True)
                p = Y / M
                ph = (h @ S.T) / M
                return (p * (np.log(p + _EPS) - np.log(ph + _EPS))).sum(1)

            # stake-blind touchstone: judge each claimed smelt split with frozen standards
            def claim_kl(o):
                # smelt claim judged on the crushed ore; cupel claim judged on the bullion it received
                Yb = o["Y2"].data
                return kl_np(Yf, o["h1"].data, S0) + kl_np(Yb + 1e-12, o["h2"].data, S20)
            kb = claim_kl(before)
            ka = claim_kl(after)
            row["touchstone_kl_before"] = float(kb.mean())
            row["touchstone_kl_after"] = float(ka.mean())
            row["touchstone_auc_after_vs_before"] = float(auc(kb, ka))
            cosg = float(np.dot(_norm(np.exp(m.cupel.S_logit.data[:, 0])), world.W[:, CU]) /
                         (np.linalg.norm(_norm(np.exp(m.cupel.S_logit.data[:, 0]))) * np.linalg.norm(world.W[:, CU])))
            row["gold_standard_cosine_with_copper_after"] = cosg
            row["mean_tau_after"] = [float(x) for x in after["tau"].data.mean(0)]
        if old_floor is not None:
            m.fire.floor = old_floor
        res[label] = row
        m.load(st)
    return res


def experiment_prospector(world, seed):
    """Priors decide where to dig; they never enter the verdict on what is found."""
    rng = np.random.default_rng(seed)
    net = FireNet(rng, 5, 16, 2, floor=0.0)      # reused as a small regressor
    opt = Adam(net.params(), lr=0.01)
    for step in range(600):
        s = world.sites(rng, 256)
        hot, moist = world.field(s["lat"], s["alt"], s["coast"], s["phase"])
        y = np.stack([hot, moist], 1) + rng.normal(0, 0.02, (256, 2))   # astronomers' tables
        pred = net(Tensor(world.ctx(s)))
        L = tmean(tsum(square(pred - y), axis=1))
        opt.zero(net.params()); L.backward(); opt.step()
    cand = world.sites(rng, 400)
    X, c, _ = world.ores(rng, cand)
    q = net(Tensor(world.ctx(cand))).data
    score = q[:, 0] * (1 - q[:, 1])
    hot, moist = world.field(cand["lat"], cand["alt"], cand["coast"], cand["phase"])
    oracle = _sig(9.0 * (hot * (1 - moist) - 0.30))
    k = 40
    return dict(field_mse=float(L.data),
                gold_per_dig_prior=float(c[np.argsort(-score)[:k], AU].mean()),
                gold_per_dig_random=float(c[rng.choice(400, k, replace=False), AU].mean()),
                gold_per_dig_oracle=float(c[np.argsort(-oracle)[:k], AU].mean()))


def needle_assay(Q, needles, iters=400):
    """
    The touch-needle assay.  A product streak Q (N x D) is fitted by a
    non-negative mixture of reference needles (D x J, the assayer's known
    alloys) with multiplicative KL updates; weights keep the product's mass.
    Returns the weight on each needle.  Independent of the refinery's own
    responsibilities, so a sharp-but-wrong parting cannot vouch for itself.
    """
    Wn = needles / needles.sum(0, keepdims=True)
    w = np.full((Q.shape[0], Wn.shape[1]), 1.0 / Wn.shape[1]) * Q.sum(1, keepdims=True)
    for _ in range(iters):
        recon = w @ Wn.T + 1e-12
        w = w * ((Q / recon) @ Wn)
    return w


def make_needles(world, rng, noise=0.10):
    N = world.W * np.exp(noise * rng.normal(size=world.W.shape))
    return N / N.sum(0, keepdims=True)


def calibrate_needles(world, ref, needles, C, rng, tau_scale=1.0, n=400, q=0.95):
    """
    Trial pieces.  Before judging a single melt the assayer rubs alloys of
    KNOWN fineness through the same furnace and against the same needles, and
    records how far the needle verdict overstates the truth.  The q-quantile of
    that overstatement becomes the gate's margin: the mint knows the error of
    its own instrument before it trusts it.
    """
    fin = rng.uniform(0.70, 1.0, n)
    comp = np.zeros((n, 6)); comp[:, AU] = fin; comp[:, CU] = 1.0 - fin
    X = comp @ world.W.T + rng.gamma(1.0, 0.001, (n, world.D))
    Cn = C[rng.integers(0, C.shape[0], n)]
    out, gt, _ = true_vessel_composition(world, ref, X, Cn, comp, tau_scale=tau_scale)
    Q = out["Y2"].data * out["r2"].data[:, :, 0]
    wts = needle_assay(Q, out["A"].data @ needles)
    belief = wts[:, AU] / np.maximum(wts.sum(1), 1e-12)
    true_f = gt[:, AU] / np.maximum(gt.sum(1), 1e-12)
    over = belief - true_f
    return float(np.quantile(over, q)), float(np.mean(np.abs(over)))


def experiment_mint(world, ref, seed, gens=8, standard=0.90, debase=0.08):
    """
    Recoining.  Coins (gold with a little copper) are melted each generation
    after a moneyer adds copper.  UNGATED: restamp the melt.  GATED: refine,
    rub the gold vessel's streak on touch-needles CALIBRATED on trial pieces
    of known fineness, return to the furnace hotter if the verdict minus the
    instrument's measured overstatement is below standard, refuse if still
    below.  The simulator's ledger reports the truth.
    """
    rng = np.random.default_rng(seed)
    needles = make_needles(world, rng)
    N = 200
    coins = np.zeros((N, 6)); coins[:, AU] = 0.97; coins[:, CU] = 0.03
    ungated = coins.copy()
    gated = coins.copy()
    s = world.sites(rng, N)
    C = world.ctx(s)
    margins = {}
    for attempt in range(4):
        margins[attempt] = calibrate_needles(world, ref, needles, C, np.random.default_rng(seed + 500 + attempt),
                                             tau_scale=1.6 ** attempt)
    rows = []
    for gnr in range(gens + 1):
        fine_u = float(ungated[:, AU].sum() / ungated.sum())
        fine_g = float(gated[:, AU].sum() / max(gated.sum(), 1e-9))
        rows.append(dict(generation=gnr, ungated_fineness=fine_u, gated_fineness=fine_g,
                         gated_mass=float(gated.sum() / (0.97 * N + 0.03 * N))))
        if gnr == gens:
            break
        ungated = ungated.copy(); ungated[:, CU] += debase * ungated.sum(1)
        melt = gated.copy(); melt[:, CU] += debase * melt.sum(1)
        alive = melt.sum(1) > 1e-6
        newc = np.zeros_like(melt)
        refused = 0
        if alive.any():
            X = melt[alive] @ world.W.T + rng.gamma(1.0, 0.001, (alive.sum(), world.D))
            ts = 1.0
            ok = np.zeros(alive.sum(), bool)
            prod = np.zeros((alive.sum(), 6))
            for attempt in range(4):
                out, gt, _ = true_vessel_composition(world, ref, X, C[alive], melt[alive], tau_scale=ts)
                Q = out["Y2"].data * out["r2"].data[:, :, 0]              # streak of the gold vessel
                wts = needle_assay(Q, out["A"].data @ needles)            # rubbed against the needles
                belief = wts[:, AU] / np.maximum(wts.sum(1), 1e-12)
                take = (~ok) & (belief - margins[attempt][0] >= standard) & (Q.sum(1) > 1e-6)
                prod[take] = gt[take]
                ok |= take
                ts *= 1.6
            refused = int((~ok).sum())
            newc[np.where(alive)[0][ok]] = prod[ok]
        gated = newc
        rows[-1]["refused_melts_next"] = refused
    rows[0]["needle_margin_by_attempt"] = [round(margins[a][0], 4) for a in range(4)]
    rows[0]["needle_mae_by_attempt"] = [round(margins[a][1], 4) for a in range(4)]
    return rows


def experiment_musnad(seed, steps):
    """Letters of similar shape: silver's streak made ever closer to lead's."""
    out = {}
    for mix in (0.15, 0.60, 0.90):
        w = Peninsula(seed=seed, D=24, silver_lead_mix=mix)
        cos = float(np.dot(w.W[:, AG], w.W[:, PB]) / (np.linalg.norm(w.W[:, AG]) * np.linalg.norm(w.W[:, PB])))
        rng = np.random.default_rng(seed + 1)
        m = Refinery(rng, w)
        train(m, w, steps, lr=0.02, seed=seed + 2, n=192, tag=f"mix{mix}")
        trng = np.random.default_rng(seed + 3)
        X, C, c, s, _ = batch(w, trng, 800)
        o, gt, st = true_vessel_composition(w, m, X, C, c)
        sv = o["silver"].data
        err = np.abs(sv - c[:, AG])
        r2c = o["r2"].data
        Y2 = o["Y2"].data
        ent = -(r2c * np.log(r2c + 1e-12)).sum(2)                    # N x D
        doubt = (Y2 * ent).sum(1) / np.maximum(Y2.sum(1), 1e-12)
        has = c[:, AG] > 0
        corr = float(np.corrcoef(doubt[has], err[has] / np.maximum(c[has, AG], 1e-3))[0, 1]) if has.sum() > 10 else float("nan")
        kl = o["kl"].data
        corr_kl = float(np.corrcoef(kl[has], err[has] / np.maximum(c[has, AG], 1e-3))[0, 1]) if has.sum() > 10 else float("nan")
        out[f"mix_{mix}"] = dict(silver_lead_cosine=cos, silver_r2=float(r2(c[:, AG], sv)),
                                 silver_parting_purity=float(st[:, AG].sum() / sv.sum()),
                                 lead_in_silver_vessel_pct=float(100 * st[:, PB].sum() / sv.sum()),
                                 doubt_error_corr=corr, touchstone_error_corr=corr_kl)
    return out


class Tongues:
    """
    Speech as ore.  Channels are dialect features (article am-/an-, the -k
    perfect, pausal glottalisation, case endings, loan lexicon ...).  Four
    varieties mix: Classical Arabic (the grammarians' standard), two distinct
    Sayhadic substrates, and trade-contact speech.  Only the Arabic share is
    ever labelled -- exactly the information a grammarian's assay provides.
    """

    def __init__(self, seed=893, D=18):
        rng = np.random.default_rng(seed)
        b = [_norm(rng.gamma(0.4, 1.0, D) + 1e-4) for _ in range(5)]
        self.V = np.stack([b[0], _norm(0.6 * b[1] + 0.4 * b[2]), _norm(0.6 * b[3] + 0.4 * b[2]), b[4]], 1)
        self.D = D

    def sample(self, rng, n):
        regions = rng.integers(0, 3, n)
        alphas = np.array([[6, 3, 0.6, 1], [5, 0.8, 3, 1], [6, 0.5, 0.5, 3.5]])
        shares = np.stack([rng.dirichlet(alphas[r]) for r in regions])
        mass = rng.uniform(0.6, 1.4, n)[:, None]
        X = (shares * mass) @ self.V.T + rng.gamma(1.0, 0.002, (n, self.D))
        return X, shares * mass


def experiment_tongue(seed, steps):
    world = Tongues(seed)
    res = {}
    for label, K in (("grammarian_one_himyaritic", 3), ("parting_two_substrates", 4)):
        rng = np.random.default_rng(seed + K)
        cr = Crucible(rng, world.D, K, None, True)
        tau = param(np.array([0.0]))
        opt = Adam(cr.params() + [tau], lr=0.03)
        for step in range(steps):
            X, sh = world.sample(rng, 192)
            t = softplus(tau) + 0.5
            h, r = cr(Tensor(X), reshape(t * Tensor(np.ones(192)), (192,)), 6)
            L = tmean(square((col(h, 0) - sh[:, 0]) * 10.0)) + tmean(touchstone_kl(Tensor(X), h, cr.standards()))
            opt.zero(cr.params() + [tau]); L.backward(); opt.step()
        Xt, sht = world.sample(np.random.default_rng(seed + 50), 1500)
        t = float(np.logaddexp(0, tau.data[0]) + 0.5)
        h, _ = cr(Tensor(Xt), Tensor(np.full(1500, t)), 6)
        H = h.data
        row = dict(arabic_share_r2=float(r2(sht[:, 0], H[:, 0])))
        S1, S2 = sht[:, 1], sht[:, 2]
        if K == 3:
            cands = [1, 2]
            best = max(cands, key=lambda k: np.corrcoef(H[:, k], S1 + S2)[0, 1])
            row["lumped_vessel_corr_with_S1_plus_S2"] = float(np.corrcoef(H[:, best], S1 + S2)[0, 1])
            row["lumped_vessel_corr_with_S1"] = float(np.corrcoef(H[:, best], S1)[0, 1])
            row["lumped_vessel_corr_with_S2"] = float(np.corrcoef(H[:, best], S2)[0, 1])
        else:
            best = -1.0
            for a in (1, 2, 3):
                for b_ in (1, 2, 3):
                    if a == b_:
                        continue
                    v = min(np.corrcoef(H[:, a], S1)[0, 1], np.corrcoef(H[:, b_], S2)[0, 1])
                    best = max(best, v)
            row["substrates_resolved_min_corr"] = float(best)
        res[label] = row
    return res


def experiment_rounds(world, ref, seed):
    """Adaptive furnace: rounds needed before the assay stops moving, by ore grade."""
    rng = np.random.default_rng(seed)
    X, C, c, s, _ = batch(world, rng, 600)
    grade = (c[:, AU] + c[:, AG]) / c.sum(1)
    prev = None
    need = np.full(600, 12)
    for R in range(1, 13):
        o = ref(X, C, rounds=(R, R))
        f = (o["gold"].data + o["silver"].data) / X.sum(1)
        if prev is not None:
            done = (np.abs(f - prev) < 5e-4) & (need == 12)
            need[done] = R
        prev = f
    qs = np.quantile(grade, [0, 1 / 3, 2 / 3, 1])
    out = {}
    for i, nm in enumerate(["low_grade", "mid_grade", "high_grade"]):
        m = (grade >= qs[i]) & (grade <= qs[i + 1])
        out[nm] = float(need[m].mean())
    return out


# =============================================================================
# S13  MAIN, SELF-TESTS, SUMMARY
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=237)
    args = ap.parse_args()
    seed = args.seed
    quick = args.quick
    tests = []

    def check(name, ok, detail=""):
        tests.append((name, bool(ok), detail))
        log(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")

    log("=" * 78)
    log("AL-JAWHARATAYN -- the non-transmuting refinery   (figure #0237 al-Hamdani)")
    log("=" * 78)

    log("\nS5  finite-difference gradient check")
    gc = gradient_check()
    check("gradient check < 1e-5 on all three models", max(gc.values()) < 1e-5,
          f"worst={max(gc.values()):.2e}")

    world = Peninsula(seed=seed)
    cs = lambda a, b: float(np.dot(world.W[:, a], world.W[:, b]) / (np.linalg.norm(world.W[:, a]) * np.linalg.norm(world.W[:, b])))
    log(f"\nS1  Peninsula: D={world.D} channels; streak cosines gold~copper={cs(AU, CU):.3f}, silver~lead={cs(AG, PB):.3f}")

    steps_ref, steps_leak, steps_mlp = (60, 40, 80) if quick else (2000, 700, 1500)
    rng = np.random.default_rng(seed)
    refinery = Refinery(rng, world)
    leaky = Refinery(rng, world, normalize=False)
    alchemist = Alchemist(rng, world.D)

    log("\nS6  training (fresh ore every step)")
    t = time.time(); train(refinery, world, steps_ref, lr=0.02, seed=seed + 1, tag="refinery"); t_ref = time.time() - t
    train(leaky, world, steps_leak, lr=0.02, seed=seed + 2, tag="leaky")
    train(alchemist, world, steps_mlp, lr=0.003, seed=seed + 3, tag="alchemist")
    n_params = {k: int(sum(p.data.size for p in m.params())) for k, m in
                dict(refinery=refinery, leaky=leaky, alchemist=alchemist).items()}
    log(f"  parameters: {n_params}   refinery training time {t_ref:.1f}s")

    test_rng = np.random.default_rng(seed + 1000)
    Xt, Ct, ct, st_, _ = batch(world, test_rng, 2000)
    acc = {"refinery": evaluate_accuracy(world, refinery, Xt, Ct, ct, True),
           "leaky": evaluate_accuracy(world, leaky, Xt, Ct, ct, False),
           "alchemist": evaluate_accuracy(world, alchemist, Xt, Ct, ct, False)}
    oL = leaky(Xt, Ct)
    acc["leaky"]["mass_created_pct_max"] = float(100 * ((oL["h1"].data.sum(1) - oL["Y"].data.sum(1)) / oL["Y"].data.sum(1)).max())
    log("\nS6  Experiment A -- refining accuracy on 2000 test ores")
    for k, v in acc.items():
        log(f"  {k:10s} " + json.dumps({a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()}))
    check("mass conserved in every vessel (<1e-9)", acc["refinery"]["conservation_err"] < 1e-9,
          f"err={acc['refinery']['conservation_err']:.2e}")
    check("no negative vessel mass", acc["refinery"]["min_vessel_mass"] >= 0.0)
    if not quick:
        check("refinery gold R2 >= 0.85", acc["refinery"]["gold_r2"] >= 0.85, f"{acc['refinery']['gold_r2']:.3f}")
        check("refinery silver R2 >= 0.70", acc["refinery"]["silver_r2"] >= 0.70, f"{acc['refinery']['silver_r2']:.3f}")

    log("\nS7  Experiment B -- barren veins in gold country; context sensitivity; unknown metal")
    models = dict(refinery=refinery, leaky=leaky, alchemist=alchemist)
    bar = experiment_barren(world, models, np.random.default_rng(seed + 7))
    log("  " + json.dumps(bar, indent=1).replace("\n", "\n  "))
    if not quick:
        check("barren trap: refinery invents less gold than the alchemist",
              bar["refinery"]["mean_gold_pct"] < bar["alchemist"]["mean_gold_pct"],
              f"{bar['refinery']['mean_gold_pct']:.3f}% vs {bar['alchemist']['mean_gold_pct']:.3f}%")
        check("verdict barely moves with the sky (refinery < alchemist sensitivity)",
              bar["refinery"]["context_sensitivity"] < bar["alchemist"]["context_sensitivity"])
        check("touchstone detects an unseen metal (AUC >= 0.9)", bar["unknown_metal_auc"] >= 0.9,
              f"AUC={bar['unknown_metal_auc']:.3f}")

    trained = {"refinery": (refinery, refinery.state()), "alchemist": (alchemist, alchemist.state())}
    log("\nS8  Experiment C -- stake-gated transmutation (the Iklil test)")
    stake = experiment_stake(world, trained, seed + 20, 40 if quick else 300)
    log("  " + json.dumps(stake, indent=1).replace("\n", "\n  "))

    log("\nS9  Experiment D -- the prospector")
    pros = experiment_prospector(world, seed + 30)
    log("  " + json.dumps(pros))
    check("priors route digging better than chance", pros["gold_per_dig_prior"] > pros["gold_per_dig_random"])

    log("\nS10 Experiment E -- recoining at the mint")
    mint = experiment_mint(world, refinery, seed + 40)
    for row in mint:
        log("  " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}))
    if not quick:
        check("gated coinage never falls below standard", min(r["gated_fineness"] for r in mint if r["gated_mass"] > 1e-6) >= 0.90 - 1e-6)
        check("ungated coinage debases", mint[-1]["ungated_fineness"] < 0.70)

    log("\nS11 Experiment F -- confusable standards (musnad letters)")
    mus = experiment_musnad(seed + 50, 30 if quick else 450)
    log("  " + json.dumps(mus, indent=1).replace("\n", "\n  "))

    log("\nS12 Experiment G -- the Tongue of Yemen (Stein's catch-all)")
    ton = experiment_tongue(seed + 60, 40 if quick else 700)
    log("  " + json.dumps(ton, indent=1).replace("\n", "\n  "))

    log("\n    Adaptive furnace rounds by ore grade")
    rnd = experiment_rounds(world, refinery, seed + 70)
    log("  " + json.dumps(rnd))

    log("\nS13 self-test summary")
    passed = sum(1 for _, ok, _ in tests if ok)
    log(f"  {passed}/{len(tests)} self-tests passed   runtime {time.time() - T0:.1f}s")
    summary = dict(self_tests=f"{passed}/{len(tests)}", gradient_check=gc, parameters=n_params,
                   accuracy=acc, barren=bar, stake=stake, prospector=pros, mint=mint,
                   musnad=mus, tongue=ton, rounds=rnd, runtime_s=round(time.time() - T0, 1))
    log("SUMMARY_JSON " + json.dumps(summary))
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())

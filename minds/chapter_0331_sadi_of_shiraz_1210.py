#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 chapter_0331_sadi_of_shiraz_1210.py
 THE GULISTAN ENGINE  —  a trainable cognitive architecture after
 Sa'di of Shiraz (Abu-Muhammad Muslih al-Din b. Abdallah Shirazi, c.1210-1291/2)
 
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0331 · Sa'di of Shiraz (1210-1291/2)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 17, Minds 321-340 Available on Amazon https://www.amazon.com/dp/B0HP7Z1NKC
# Resume and Interactive Demos at https://artificiology.com/

 Encyclopedia of Lost Minds: Echoes on AI  —  Mind No. 0331
================================================================================

WHY THIS ARCHITECTURE AND NOT A TRANSFORMER
-------------------------------------------
Sa'di did not organise the Gulistan (1258) by topic. He organised it by STATION:
"The Manners of Kings", "On the Morals of Dervishes", "On the Excellence of
Contentment", "On the Advantages of Silence", "On Love and Youth", "On Weakness
and Old Age", "On the Effects of Education", "On Rules for Conduct in Life".
Eight rooms. The same act carries opposite verdicts in different rooms, and the
book never apologises for it. Persian has a saying about him: each of Sa'di's
words has seventy-two meanings.

A Transformer softmaxes over everything it can see. Every stored key competes in
one global normalisation, so a maxim learned in the court leaks into the
almshouse with a small but non-zero weight. Sa'di's book is built on the
opposite commitment: a precedent drawn from the wrong station is not merely a
weak match, it is INADMISSIBLE. The mind's first act is not retrieval; it is
localisation — which room am I standing in? — and retrieval is legal only
afterwards.

The second commitment comes from the first story in the book. A condemned man
curses the king in a language the king does not speak. One vizier lies about
what was said and saves him; a rival vizier reports the curse truthfully. The
king prefers the lie, and gives his reason: one utterance issued from
conciliation, the other from malice. So the value of a speech act is not scored
at the speaker's correspondence to fact. It is scored at the LISTENER, jointly by
where the utterance came from and what it does when it lands.

The third commitment is the six lines of Bani Adam (Gulistan I.10): the members
of the human body do not have separate utilities. Pain in one propagates, and
the body is not at rest while any member hurts. That is a min-max criterion, not
an average.

Those three commitments are the whole machine:

    STATION GATE  ->  station-masked case retrieval, cross-room transfer
                      suppressed by construction, not by learned weight decay
    BAYT PRESS    ->  the retrieved verdict must survive compression to a
                      couplet-sized handle, because that is the handle the rest
                      of the system is allowed to use ("a flower lasts five or
                      six days; this rose-garden is always delightful")
    SAME' MODEL   ->  a recurrent simulation of the hearer, two turns deep, so
                      the second thing you say lands on someone the first thing
                      already changed
    MASLAHAT HEAD ->  utterance scored by landed effect, with the penalty on
                      factual divergence MODULATED by origin-intent
    BANI ADAM     ->  pain propagates along the body graph; the objective is the
                      worst member, never the mean
    THE FLOOR     ->  a non-negotiable veto: a large factual divergence issued
                      from malign intent is rejected regardless of its score.
                      Sa'di licensed the conciliatory falsehood; he never
                      licensed the useful one.

IMPLEMENTATION NOTES
--------------------
Pure NumPy. Nothing is imported that Sa'di's engine does not need. Reverse-mode
autodiff is written from scratch in this file (class `T`, ~200 lines) rather
than hand-deriving each backward pass, so that the finite-difference check
validates BOTH the differentiation machinery and the model. The optimiser (Adam)
is also written here.

Run:  python3 chapter_0331_sadi_of_shiraz_1210.py
      python3 chapter_0331_sadi_of_shiraz_1210.py --quick     (fast smoke run)
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np

# =============================================================================
#  PART 0 — A SMALL REVERSE-MODE AUTODIFF ENGINE
#
#  Every value is a node `T` holding a NumPy array plus the list of parents it
#  came from and a local backward rule. `backward()` walks the tape in reverse
#  topological order. Broadcasting is handled by `_reduce_to`, which sums a
#  gradient back down to the shape of the tensor that was broadcast.
# =============================================================================


def _reduce_to(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum `grad` down to `shape`, undoing NumPy broadcasting."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


class T:
    """A tape node. `.v` is the value, `.g` is the accumulated gradient."""

    __slots__ = ("v", "g", "_parents", "_backward", "_name")

    def __init__(self, value, parents=(), backward=None, name=""):
        self.v = np.asarray(value, dtype=np.float64)
        self.g = np.zeros_like(self.v)
        self._parents = parents
        self._backward = backward or (lambda: None)
        self._name = name

    # -- shape sugar -------------------------------------------------------
    @property
    def shape(self):
        return self.v.shape

    def __repr__(self):
        return f"T{self.v.shape}{'[' + self._name + ']' if self._name else ''}"

    # -- tape --------------------------------------------------------------
    def backward(self):
        """Seed this (scalar) node with 1.0 and propagate to every leaf."""
        order, seen = [], set()

        def visit(node):
            if id(node) in seen:
                return
            seen.add(id(node))
            for p in node._parents:
                visit(p)
            order.append(node)

        visit(self)
        for node in order:
            node.g = np.zeros_like(node.v)
        self.g = np.ones_like(self.v)
        for node in reversed(order):
            node._backward()

    # -- arithmetic --------------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, T) else T(other)
        out = T(self.v + other.v, (self, other))

        def bw():
            self.g += _reduce_to(out.g, self.v.shape)
            other.g += _reduce_to(out.g, other.v.shape)

        out._backward = bw
        return out

    __radd__ = __add__

    def __mul__(self, other):
        other = other if isinstance(other, T) else T(other)
        out = T(self.v * other.v, (self, other))

        def bw():
            self.g += _reduce_to(out.g * other.v, self.v.shape)
            other.g += _reduce_to(out.g * self.v, other.v.shape)

        out._backward = bw
        return out

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        other = other if isinstance(other, T) else T(other)
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __truediv__(self, other):
        if isinstance(other, T):
            return self * other.pow(-1.0)
        return self * (1.0 / other)

    def pow(self, p):
        out = T(self.v ** p, (self,))

        def bw():
            self.g += out.g * p * (self.v ** (p - 1.0))

        out._backward = bw
        return out

    def matmul(self, other):
        out = T(np.matmul(self.v, other.v), (self, other))

        def bw():
            ga = np.matmul(out.g, np.swapaxes(other.v, -1, -2))
            gb = np.matmul(np.swapaxes(self.v, -1, -2), out.g)
            self.g += _reduce_to(ga, self.v.shape)
            other.g += _reduce_to(gb, other.v.shape)

        out._backward = bw
        return out

    __matmul__ = matmul

    # -- shape ops ---------------------------------------------------------
    def reshape(self, *shape):
        old = self.v.shape
        out = T(self.v.reshape(shape), (self,))

        def bw():
            self.g += out.g.reshape(old)

        out._backward = bw
        return out

    def transpose(self, *axes):
        out = T(np.transpose(self.v, axes), (self,))
        inv = np.argsort(axes)

        def bw():
            self.g += np.transpose(out.g, inv)

        out._backward = bw
        return out

    # -- reductions --------------------------------------------------------
    def sum(self, axis=None, keepdims=False):
        out = T(self.v.sum(axis=axis, keepdims=keepdims), (self,))

        def bw():
            g = out.g
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.g += np.broadcast_to(g, self.v.shape).copy()

        out._backward = bw
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.v.size if axis is None else self.v.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def logsumexp(self, axis=-1, keepdims=False):
        m = self.v.max(axis=axis, keepdims=True)
        e = np.exp(self.v - m)
        s = e.sum(axis=axis, keepdims=True)
        val = (np.log(s) + m)
        soft = e / s
        if not keepdims:
            val = np.squeeze(val, axis=axis)
        out = T(val, (self,))

        def bw():
            g = out.g
            if not keepdims:
                g = np.expand_dims(g, axis)
            self.g += g * soft

        out._backward = bw
        return out

    def softmax(self, axis=-1):
        m = self.v.max(axis=axis, keepdims=True)
        e = np.exp(self.v - m)
        y = e / e.sum(axis=axis, keepdims=True)
        out = T(y, (self,))

        def bw():
            dot = (out.g * y).sum(axis=axis, keepdims=True)
            self.g += y * (out.g - dot)

        out._backward = bw
        return out

    # -- elementwise nonlinearities ---------------------------------------
    def tanh(self):
        y = np.tanh(self.v)
        out = T(y, (self,))

        def bw():
            self.g += out.g * (1.0 - y * y)

        out._backward = bw
        return out

    def sigmoid(self):
        y = np.where(self.v >= 0, 1.0 / (1.0 + np.exp(-np.abs(self.v))),
                     np.exp(-np.abs(self.v)) / (1.0 + np.exp(-np.abs(self.v))))
        out = T(y, (self,))

        def bw():
            self.g += out.g * y * (1.0 - y)

        out._backward = bw
        return out

    def softplus(self):
        y = np.logaddexp(0.0, self.v)
        out = T(y, (self,))
        sig = 1.0 / (1.0 + np.exp(-self.v))

        def bw():
            self.g += out.g * sig

        out._backward = bw
        return out

    def log(self):
        out = T(np.log(self.v), (self,))

        def bw():
            self.g += out.g / self.v

        out._backward = bw
        return out

    def gather_last(self, idx: np.ndarray):
        """Pick one entry along the last axis per row. idx shape = v.shape[:-1]."""
        i = idx[..., None]
        out = T(np.take_along_axis(self.v, i, axis=-1)[..., 0], (self,))

        def bw():
            grad = np.zeros_like(self.v)
            np.put_along_axis(grad, i, out.g[..., None], axis=-1)
            self.g += grad

        out._backward = bw
        return out


def cross_entropy(logits: T, target: np.ndarray) -> T:
    """Mean negative log-likelihood over the last axis."""
    return (logits.logsumexp(axis=-1) - logits.gather_last(target)).mean()


# =============================================================================
#  PART 1 — THE WORLD THE ENGINE LIVES IN
#
#  A synthetic council-chamber corpus with the three properties Sa'di's book
#  insists on, and which a station-blind model cannot satisfy:
#
#   (a) STATION CONFLICT. Each station k has a hidden verdict direction g_k, and
#       the directions are arranged in antipodal pairs (g_1 = -g_0, g_3 = -g_2,
#       ...). A rule that is correct among kings is exactly wrong among
#       dervishes. Any leak across rooms is punished, not merely diluted.
#   (b) INTENT-MODULATED TRUTH COST. A candidate's factual divergence hurts the
#       hearers in proportion to how malign its origin is. A conciliatory
#       near-falsehood costs almost nothing; a malicious one costs a great deal.
#   (c) BODY COUPLING. Listener pain propagates along a fixed body graph, and
#       the true cost of an utterance is the WORST member's propagated pain, not
#       the average. "While one member is in pain, the others find no rest."
# =============================================================================

STATION_NAMES = [
    "1 Manners of Kings",
    "2 Morals of Dervishes",
    "3 Excellence of Contentment",
    "4 Advantages of Silence",
    "5 Love and Youth",
    "6 Weakness and Old Age",
    "7 Effects of Education",
    "8 Rules for Conduct in Life",
]


class GulistanConfig:
    """All dimensions in one place. Deliberately small: this is a mind, not a farm."""

    d_in = 12          # predicament features
    K = 8              # stations = the eight chapters of the Gulistan
    h_router = 24      # station-router hidden width
    M = 40             # stored cases (hekayat) in the case-base
    d_v = 10           # verdict value dimension
    d_z = 3            # BAYT PRESS bottleneck: a couplet-sized handle
    d_u = 6            # utterance content dimension
    d_t = 4            # listener trait dimension
    d_h = 10           # listener state dimension
    N = 6              # members of the body
    C = 4              # candidate utterances per predicament
    T_turns = 2        # depth of the hearer recurrence

    lam_gate = 8.0     # strength of the station mask (0 = ordinary attention)
    gamma_body = 0.6   # how strongly pain propagates to neighbours
    beta_max = 8.0     # sharpness of the smooth max over members

    # THE FLOOR — the inviolable veto, applied at decision time.
    delta_max = 0.70   # factual divergence above which an utterance is suspect
    iota_min = -0.30   # origin-intent below which it is deemed malign

    a_station = 0.60   # loss weight: supervision on station identification
    a_bayt = 0.30      # loss weight: couplet reconstruction
    wd = 1e-5          # weight decay


class World:
    """Generates predicaments, candidate utterances, and the oracle's verdict."""

    def __init__(self, cfg: GulistanConfig, seed: int = 1258):
        self.cfg = cfg
        rs = np.random.RandomState(seed)
        self.rs = rs

        # Station prototypes in predicament space: recognisable but overlapping.
        self.mu = rs.randn(cfg.K, cfg.d_in) * 1.15

        # Antipodal verdict directions — the source of station conflict.
        half = rs.randn((cfg.K + 1) // 2, cfg.d_u)
        half /= np.linalg.norm(half, axis=1, keepdims=True)
        g = np.zeros((cfg.K, cfg.d_u))
        for k in range(cfg.K):
            g[k] = half[k // 2] * (1.0 if k % 2 == 0 else -1.0)
        self.g = g

        # The body: a ring with two chords, row-normalised. Nobody is isolated.
        A = np.zeros((cfg.N, cfg.N))
        for i in range(cfg.N):
            A[i, (i + 1) % cfg.N] = 1.0
            A[i, (i - 1) % cfg.N] = 1.0
        A[0, cfg.N // 2] = 1.0
        A[cfg.N // 2, 0] = 1.0
        self.A = A / np.maximum(A.sum(axis=1, keepdims=True), 1e-9)

        # Listener traits and sensitivities: the members are not interchangeable.
        # `mmask` is the crucial one — each member reads the station's verdict
        # direction through its OWN concerns, so an utterance can console five
        # members and wound the sixth. Without this the max and the mean rank
        # candidates identically and the Bani Adam coupling would be decoration.
        self.traits = rs.randn(cfg.N, cfg.d_t) * 0.8
        self.sens = 0.5 + np.abs(rs.randn(cfg.N)) * 0.8
        self.mmask = 1.0 + rs.randn(cfg.N, cfg.d_u) * 1.25
        self.tsens = 0.4 + np.abs(rs.randn(cfg.N)) * 1.6

    # -- the true cost of an utterance, as the world (not the model) computes it
    def _true_pain(self, station, U, D, I):
        """station:(B,) U:(B,C,d_u) D:(B,C) I:(B,C) -> propagated pain (B,C,N)."""
        cfg = self.cfg
        gk = self.g[station]                              # (B, d_u)
        # Member-specific alignment: each member weighs the verdict direction by
        # what it personally cares about.
        gm = gk[:, None, :] * self.mmask[None, :, :]      # (B, N, d_u)
        align = np.einsum("bcd,bnd->bcn", U, gm)          # (B, C, N)
        malice = 1.0 - 1.0 / (1.0 + np.exp(-3.0 * I))     # (B,C) 1 when malign
        base = (-align * self.sens[None, None, :]
                + 1.8 * (D * malice)[:, :, None] * self.tsens[None, None, :]
                + 0.15 * self.rs.randn(*U.shape[:2], cfg.N))
        pain = np.logaddexp(0.0, base)                    # softplus
        # Bani Adam: what one member suffers is felt along the body.
        return pain + cfg.gamma_body * np.einsum("bcn,mn->bcm", pain, self.A)

    def batch(self, B: int, seam: bool = False):
        """
        `seam=True` draws predicaments from the boundary between two ANTIPODAL
        rooms — the doorway between the court and the almshouse, where the same
        situation carries opposite verdicts depending on which side of the
        threshold you are standing. This is where a station-blind retriever
        blends two contradictory precedents into a verdict belonging to neither.
        """
        cfg, rs = self.cfg, self.rs
        station = rs.randint(0, cfg.K, size=B)
        if seam:
            other = station ^ 1                       # its antipodal twin
            centre = 0.60 * self.mu[station] + 0.40 * self.mu[other]
        else:
            centre = self.mu[station]
        x = centre + rs.randn(B, cfg.d_in) * 0.85
        U = rs.randn(B, cfg.C, cfg.d_u)
        D = rs.rand(B, cfg.C)                     # factual divergence in [0,1]
        I = rs.rand(B, cfg.C) * 2.0 - 1.0         # origin intent in [-1,1]

        ptilde = self._true_pain(station, U, D, I)
        cost = ptilde.max(axis=2)                 # THE WORST MEMBER, not the mean

        # The oracle obeys the same floor the engine does.
        vetoed = (D > cfg.delta_max) & (I < cfg.iota_min)
        adj = np.where(vetoed, np.inf, cost)
        allv = np.all(np.isinf(adj), axis=1)
        adj[allv] = cost[allv]
        best = adj.argmin(axis=1)

        return dict(x=x, U=U, D=D, I=I, station=station, best=best,
                    cost=cost, vetoed=vetoed)


# =============================================================================
#  PART 2 — THE GULISTAN ENGINE
# =============================================================================


class GulistanEngine:
    """
    Forward pass, in the order Sa'di's book asks for:

        localise the room  ->  retrieve only from that room  ->  press the
        verdict into a couplet  ->  simulate the hearers for two turns  ->
        propagate pain through the body  ->  score by the worst member and by
        landed effect, with the truth-penalty modulated by origin-intent.
    """

    def __init__(self, cfg: GulistanConfig, seed: int = 655):
        self.cfg = cfg
        rs = np.random.RandomState(seed)
        p = {}

        def rnd(*shape, scale=None):
            sc = scale if scale is not None else 1.0 / math.sqrt(shape[0])
            return rs.randn(*shape) * sc

        # --- the MAQAM ROUTER: where am I standing? --------------------------
        p["Wr1"] = rnd(cfg.d_in, cfg.h_router)
        p["br1"] = np.zeros(cfg.h_router)
        p["Wr2"] = rnd(cfg.h_router, cfg.K)
        p["br2"] = np.zeros(cfg.K)

        # --- the CASE-BASE: keys, station tags, verdict values ---------------
        p["Kmem"] = rnd(cfg.M, cfg.d_in, scale=0.5)
        p["Tmem"] = rs.randn(cfg.M, cfg.K) * 0.5     # each case belongs somewhere
        p["Vmem"] = rnd(cfg.M, cfg.d_v, scale=0.5)

        # --- the BAYT PRESS: verdict -> couplet handle -> verdict ------------
        p["We"] = rnd(cfg.d_v, cfg.d_z)
        p["be"] = np.zeros(cfg.d_z)
        p["Wd"] = rnd(cfg.d_z, cfg.d_v)
        p["bd"] = np.zeros(cfg.d_v)

        # --- the SAME' (hearer) simulator ------------------------------------
        p["Wu"] = rnd(cfg.d_u, cfg.d_h)
        p["Wz"] = rnd(cfg.d_z, cfg.d_h)
        p["Wdd"] = rnd(2, cfg.d_h)                   # [divergence, intent]
        p["Wt"] = rnd(cfg.d_t, cfg.d_h)              # who is listening
        p["Wux"] = rnd(cfg.d_u, cfg.d_h)             # utterance x hearer, the
        p["Wtx"] = rnd(cfg.d_t, cfg.d_h)             #   multiplicative reading
        p["bh"] = np.zeros(cfg.d_h)
        p["Wh"] = rnd(cfg.d_h, cfg.d_h, scale=0.3)   # turn -> turn
        p["wp"] = rnd(cfg.d_h, 1)                    # readout: pain
        p["bp"] = np.zeros(1)
        p["wq"] = rnd(cfg.d_h, 1)                    # readout: benefit received
        p["bq"] = np.zeros(1)

        # --- the MASLAHAT head ------------------------------------------------
        p["a_int"] = np.array([0.5])   # how sharply intent modulates truth-cost
        p["b_int"] = np.array([0.0])
        p["w_body"] = np.array([0.0])  # softplus'd -> positive weights
        p["w_truth"] = np.array([0.0])
        p["w_ben"] = np.array([0.0])

        self.params = {k: np.asarray(v, dtype=np.float64) for k, v in p.items()}

    # ------------------------------------------------------------------
    def leaves(self):
        return {k: T(v, name=k) for k, v in self.params.items()}

    # ------------------------------------------------------------------
    def forward(self, batch, world: World, P=None, lam_gate=None, use_max=True):
        """
        Returns a dict of tape nodes. Everything downstream of `score` is what
        the engine actually decides with.
        """
        cfg = self.cfg
        P = P if P is not None else self.leaves()
        lam = cfg.lam_gate if lam_gate is None else lam_gate

        x = T(batch["x"])                                  # (B, d_in)
        U = T(batch["U"])                                  # (B, C, d_u)
        D = T(batch["D"])                                  # (B, C)
        I = T(batch["I"])                                  # (B, C)
        B = batch["x"].shape[0]

        # ---- 1. MAQAM ROUTER ------------------------------------------------
        # Before anything is recalled, the engine decides which of the eight
        # rooms it is standing in. This posterior is the only thing that will be
        # allowed to unlock the case-base.
        hr = (x @ P["Wr1"] + P["br1"]).tanh()
        st_logits = hr @ P["Wr2"] + P["br2"]               # (B, K)
        s = st_logits.softmax(axis=-1)                     # (B, K)

        # ---- 2. STATION-MASKED CASE RETRIEVAL -------------------------------
        # Each stored case declares which room it belongs to (Tmem). Its
        # admissibility is the overlap between the engine's station posterior
        # and the case's own station distribution. That overlap enters the
        # attention logits through a LOGARITHM scaled by lam_gate, so a case
        # from the wrong room is not merely down-weighted, it is driven toward
        # -inf. With lam_gate = 0 this degenerates into ordinary attention,
        # which is exactly the ablation run in the self-tests below.
        Tm = P["Tmem"].softmax(axis=-1)                    # (M, K)
        compat = s @ Tm.transpose(1, 0)                    # (B, M) in (0,1)
        raw = (x @ P["Kmem"].transpose(1, 0)) * (1.0 / math.sqrt(cfg.d_in))
        logits = raw + (compat + 1e-8).log() * lam
        alpha = logits.softmax(axis=-1)                    # (B, M)
        v = alpha @ P["Vmem"]                              # (B, d_v)

        # ---- 3. BAYT PRESS ---------------------------------------------------
        # "A flower lasts five or six days; this rose-garden is always
        # delightful." The retrieved verdict is squeezed to d_z numbers — a
        # couplet — and ONLY the couplet is passed downstream. The decoder
        # exists solely to prove, through the reconstruction loss, that nothing
        # load-bearing was lost in the squeezing.
        z = (v @ P["We"] + P["be"]).tanh()                 # (B, d_z)
        v_hat = z @ P["Wd"] + P["bd"]                      # (B, d_v)
        rec = ((v_hat - v) * (v_hat - v)).mean()

        # ---- 4. SAME': THE HEARER, SIMULATED FOR TWO TURNS ------------------
        # Every candidate is played out against every member of the body. The
        # drive term is constant across turns; the recurrence is what makes the
        # second turn land on someone the first turn already moved.
        traits = T(world.traits)                           # (N, d_t)
        dfeat_np = np.stack([batch["D"], batch["I"]], axis=-1)   # (B, C, 2)
        dfeat = T(dfeat_np)

        # The multiplicative term is not decoration either: the same sentence
        # does different work in different ears, so the utterance projection is
        # GATED by the hearer's projection rather than merely added to it.
        inter = ((U @ P["Wux"]).reshape(B, cfg.C, 1, cfg.d_h)
                 * (traits @ P["Wtx"]).reshape(1, 1, cfg.N, cfg.d_h))
        drive = ((U @ P["Wu"]).reshape(B, cfg.C, 1, cfg.d_h)
                 + (z @ P["Wz"]).reshape(B, 1, 1, cfg.d_h)
                 + (dfeat @ P["Wdd"]).reshape(B, cfg.C, 1, cfg.d_h)
                 + (traits @ P["Wt"]).reshape(1, 1, cfg.N, cfg.d_h)
                 + inter
                 + P["bh"].reshape(1, 1, 1, cfg.d_h))
        h = drive.tanh()                                   # turn 1 (h0 = 0)
        for _ in range(cfg.T_turns - 1):
            h = (h @ P["Wh"] + drive).tanh()               # turn 2, ...

        pain = (h @ P["wp"] + P["bp"]).reshape(B, cfg.C, cfg.N).softplus()
        benefit = (h @ P["wq"] + P["bq"]).reshape(B, cfg.C, cfg.N).mean(axis=-1)

        # ---- 5. BANI ADAM COUPLING -------------------------------------------
        # Pain does not stay where it was inflicted. Each member's felt pain
        # picks up a share of its neighbours'. Then the cost of the utterance is
        # the WORST member (a smooth max), never the average — because a body
        # with one limb in agony is not a body at 1/6 of agony.
        A = T(world.A.T)                                   # (N, N)
        p_felt = pain + (pain @ A) * cfg.gamma_body        # (B, C, N)
        if use_max:
            # smooth max: (1/beta) * logsumexp(beta * p)  ->  max as beta grows
            body = (p_felt * cfg.beta_max).logsumexp(axis=-1) * (1.0 / cfg.beta_max)
        else:
            body = p_felt.mean(axis=-1)                    # the utilitarian ablation

        # ---- 6. MASLAHAT HEAD -------------------------------------------------
        # Gulistan I.1. The king's ruling, in one line of arithmetic: the cost
        # charged for departing from fact is scaled by where the utterance came
        # from. Conciliatory origin -> the gate closes toward 0 and the
        # divergence is nearly free. Malign origin -> the gate opens to 1 and
        # the same divergence becomes expensive.
        gate = (I * (-1.0) * P["a_int"].softplus() + P["b_int"]).sigmoid()
        score = (body * (-1.0) * P["w_body"].softplus()
                 + (D * gate) * (-1.0) * P["w_truth"].softplus()
                 + benefit * P["w_ben"].softplus())        # (B, C)

        return dict(score=score, st_logits=st_logits, rec=rec, alpha=alpha,
                    s=s, z=z, gate=gate, body=body, pain=pain, P=P)

    # ------------------------------------------------------------------
    def loss(self, batch, world, P=None, **kw):
        out = self.forward(batch, world, P=P, **kw)
        P = out["P"]
        L = cross_entropy(out["score"], batch["best"])
        L = L + cross_entropy(out["st_logits"], batch["station"]) * self.cfg.a_station
        L = L + out["rec"] * self.cfg.a_bayt
        reg = None
        for k, t in P.items():
            term = (t * t).sum()
            reg = term if reg is None else reg + term
        L = L + reg * self.cfg.wd
        out["loss"] = L
        return out

    # ------------------------------------------------------------------
    def decide(self, batch, world, apply_floor=True, **kw):
        """
        Inference. THE FLOOR lives here and only here: it is a hard, external
        constraint on the action set, not a term the optimiser can trade away.
        Sa'di licensed the conciliatory falsehood. He never licensed the merely
        useful one, and no amount of expected benefit unlocks it.
        """
        cfg = self.cfg
        raw = self.forward(batch, world, **kw)["score"].v
        sc = raw.copy()
        if apply_floor:
            veto = (batch["D"] > cfg.delta_max) & (batch["I"] < cfg.iota_min)
            sc = np.where(veto, -np.inf, sc)
            # If every option is vetoed the floor cannot be met; the engine must
            # still speak, so it falls back to the unvetoed ranking and the
            # caller is told (by `vetoed`) that it was forced.
            dead = np.all(np.isinf(sc), axis=1)
            if dead.any():
                sc[dead] = raw[dead]
        else:
            dead = np.zeros(raw.shape[0], dtype=bool)
        return sc.argmax(axis=1), sc, dead


# =============================================================================
#  PART 3 — ADAM, WRITTEN OUT
# =============================================================================


class Adam:
    def __init__(self, params, lr=6e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads):
        self.t += 1
        for k in self.p:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            self.p[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


# =============================================================================
#  PART 4 — TRAINING
# =============================================================================


def run_batch(engine: GulistanEngine, batch, world, **kw):
    """One forward/backward. Returns (loss value, grad dict)."""
    P = engine.leaves()
    out = engine.loss(batch, world, P=P, **kw)
    out["loss"].backward()
    return float(out["loss"].v), {k: t.g for k, t in P.items()}


def evaluate(engine, world, batches, apply_floor=True, **kw):
    agree, veto_hits, n = 0, 0, 0
    for b in batches:
        pred, _, _ = engine.decide(b, world, apply_floor=apply_floor, **kw)
        agree += int((pred == b["best"]).sum())
        veto_hits += int(b["vetoed"][np.arange(len(pred)), pred].sum())
        n += len(pred)
    return agree / n, veto_hits / n


def train(engine, world, steps=900, bs=48, lr=6e-3, log_every=150, quiet=False, **kw):
    opt = Adam(engine.params, lr=lr)
    hist = []
    for i in range(1, steps + 1):
        batch = world.batch(bs)
        L, G = run_batch(engine, batch, world, **kw)
        opt.step(G)
        hist.append(L)
        if not quiet and (i % log_every == 0 or i == 1):
            held = [world.batch(96) for _ in range(3)]
            acc, _ = evaluate(engine, world, held, **kw)
            st = np.mean([
                (engine.forward(b, world, **kw)["s"].v.argmax(1) == b["station"]).mean()
                for b in held])
            print(f"    step {i:>4d}   loss {L:8.4f}   verdict-agreement "
                  f"{acc*100:5.1f}%   station-id {st*100:5.1f}%")
    return hist


# =============================================================================
#  PART 5 — SELF-TESTS
#
#  Eight tests, one per station. Each one interrogates a commitment of the mind
#  rather than a property of the code.
# =============================================================================


def test_1_gradcheck(cfg, world, seed=7):
    """
    MANDATORY. Central finite differences against the tape, on the real loss.

    Coordinates whose true gradient is structurally zero are checked absolutely
    rather than relatively, because dividing by rounding noise is meaningless.
    `bq` is the honest example: it is a constant added to every candidate's
    score, and a constant added to every logit is invisible to a softmax, so its
    gradient is exactly nought and the finite difference returns only float
    dust. The distinction is kept explicit rather than hidden in a tolerance.
    """
    eng = GulistanEngine(cfg, seed=seed)
    batch = world.batch(6)
    _, G = run_batch(eng, batch, world)

    rs = np.random.RandomState(0)
    worst_rel, worst_abs, live, flat = 0.0, 0.0, 0, 0
    for name in sorted(eng.params):
        arr = eng.params[name]
        for fi in rs.choice(arr.size, size=min(4, arr.size), replace=False):
            idx = np.unravel_index(fi, arr.shape)
            h = 1e-6 * max(1.0, abs(arr[idx]))
            orig = arr[idx]
            arr[idx] = orig + h
            lp = float(eng.loss(batch, world)["loss"].v)
            arr[idx] = orig - h
            lm = float(eng.loss(batch, world)["loss"].v)
            arr[idx] = orig
            num, ana = (lp - lm) / (2 * h), G[name][idx]
            if max(abs(num), abs(ana)) < 1e-7:
                flat += 1
                worst_abs = max(worst_abs, abs(num - ana))
            else:
                live += 1
                worst_rel = max(worst_rel, abs(num - ana) / (abs(num) + abs(ana)))
    ok = (worst_rel < 1e-5) and (worst_abs < 1e-7)
    print(f"  [1] gradient check ..................... {live} live coords, max "
          f"rel err {worst_rel:.3e}; {flat} structurally flat, max abs err "
          f"{worst_abs:.1e}  ->  {'PASS' if ok else 'FAIL'}")
    return ok


def test_2_station_gate(cfg, world, eng):
    """
    A precedent from the wrong room must be inadmissible, not merely unlikely.

    Note the honest caveat this test makes visible: the mask is only as hard as
    the router is confident. log(compat) * lam_gate is near-zero when the
    station posterior is flat, so an engine that does not yet know which room it
    is in retrieves promiscuously — which is exactly Sa'di's point about the
    young and the untravelled. The test is therefore run on the trained router.
    """
    tags = eng.params["Tmem"].argmax(axis=1)          # each case's home room
    gated, ungated, conf = [], [], []
    for _ in range(12):
        b = world.batch(64)
        o_g = eng.forward(b, world)
        o_o = eng.forward(b, world, lam_gate=0.0)
        room = o_g["s"].v.argmax(axis=1)              # (B,)
        off = (tags[None, :] != room[:, None])        # (B, M)
        gated.append((o_g["alpha"].v * off).sum(axis=1))
        ungated.append((o_o["alpha"].v * off).sum(axis=1))
        conf.append(o_g["s"].v.max(axis=1))
    mg, mo, mc = (float(np.concatenate(x).mean()) for x in (gated, ungated, conf))
    # Cases are permitted soft multi-room membership (an anecdote may sit in two
    # chapters), so a small residue of off-argmax mass is legitimate rather than
    # leakage. What must hold is that the gate suppresses it by a large factor.
    ok = (mg < 0.15) and (mo > 5.0 * mg)
    print(f"  [2] cross-station transfer ............. off-room attention mass "
          f"{mg:.4f} gated vs {mo:.4f} open (router confidence {mc:.3f})  ->  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def test_3_floor_is_inviolable(cfg, world, eng, seed=13):
    """
    No expected benefit may unlock a malign falsehood — WHILE ANY LAWFUL ROAD
    REMAINS OPEN.

    The qualification is not a loophole; it is the honest shape of the
    commitment. The floor constrains the action set, and an action set can be
    emptied. When every candidate on the table is both far from the facts and
    ill-intentioned, the engine is not thereby granted permission — it is simply
    out of lawful moves, and it says so by raising `forced`. A system that
    silently picked the least bad malign option and reported nothing would be
    the dangerous design. The two quantities are therefore counted separately:
    violations while an admissible option existed (must be zero), and the rate
    at which the engine is cornered (reported, never zero by construction).
    """
    hits, cornered, n = 0, 0, 0
    for _ in range(40):
        b = world.batch(32)
        # Force candidate 0 to be a malign falsehood and strip its content, so
        # nothing but the floor stands between it and selection.
        b["D"][:, 0] = 0.95
        b["I"][:, 0] = -0.9
        b["U"][:, 0, :] = 0.0
        b["vetoed"] = (b["D"] > cfg.delta_max) & (b["I"] < cfg.iota_min)
        pred, _, forced = eng.decide(b, world)
        hits += int(((pred == 0) & (~forced)).sum())
        cornered += int(forced.sum())
        n += len(pred)
    ok = hits == 0
    print(f"  [3] the floor holds .................... {hits}/{n} malign "
          f"falsehoods selected while a lawful option existed; engine cornered "
          f"on {cornered}/{n}  ->  {'PASS' if ok else 'FAIL'}")
    return ok


def test_4_intent_modulates_truth(cfg, world, eng):
    """The cost charged for leaving the facts must fall as intent conciliates."""
    b = world.batch(1)
    iotas = np.linspace(-1.0, 1.0, 9)
    gates = []
    for it in iotas:
        bb = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in b.items()}
        bb["I"][:] = it
        gates.append(float(eng.forward(bb, world)["gate"].v.mean()))
    gates = np.array(gates)
    monotone = bool(np.all(np.diff(gates) < 1e-12))
    spread = gates[0] - gates[-1]
    ok = monotone and spread > 0.05
    print(f"  [4] intent modulates truth-cost ........ gate {gates[0]:.3f} "
          f"(malign) -> {gates[-1]:.3f} (conciliatory), monotone={monotone}  ->  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def test_5_bayt_press(cfg, world, eng, rec0):
    """The couplet must be a lossy handle that still carries the verdict."""
    b = world.batch(256)
    rec1 = float(eng.forward(b, world)["rec"].v)
    ok = (cfg.d_z < cfg.d_v) and (rec1 < rec0)
    print(f"  [5] bayt press ......................... {cfg.d_v}->{cfg.d_z}->"
          f"{cfg.d_v}, reconstruction {rec0:.4f} -> {rec1:.4f}  ->  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def test_6_learning(cfg, world, eng, acc, base):
    """Agreement with the oracle must clear chance by a wide margin."""
    ok = acc > max(0.55, base + 0.25)
    print(f"  [6] verdict agreement .................. {acc*100:.1f}% vs "
          f"{base*100:.1f}% chance  ->  {'PASS' if ok else 'FAIL'}")
    return ok


def test_7_body_coupling(cfg, world, eng, acc, steps):
    """
    Is the Bani Adam coupling load-bearing, and does it pay?

    Two separate questions, and the file refuses to conflate them.

    (i) LOAD-BEARING. Scored with the smooth max over members and scored with
        the mean, the SAME trained engine must actually choose differently on a
        non-trivial fraction of predicaments. If the two criteria ranked
        identically, the whole coupling would be ornament. This is the pass
        condition.

    (ii) DOES IT PAY. A twin trained end-to-end with the mean is measured
        against the same min-max oracle. The answer here is reported, not
        asserted — and it is frequently uncomfortable, which is the point.
    """
    held = [world.batch(200) for _ in range(5)]
    diff, tot = 0, 0
    for b in held:
        p_max, _, _ = eng.decide(b, world, use_max=True)
        p_mean, _, _ = eng.decide(b, world, use_max=False)
        diff += int((p_max != p_mean).sum())
        tot += len(p_max)
    disagree = diff / tot

    twin = GulistanEngine(cfg, seed=655)
    train(twin, world, steps=steps, quiet=True, use_max=False)
    a_mean, _ = evaluate(twin, world, held, use_max=False)

    ok = disagree > 0.05
    print(f"  [7] body coupling is load-bearing ...... min-max and mean choose "
          f"differently on {disagree*100:.1f}% of predicaments  ->  "
          f"{'PASS' if ok else 'FAIL'}")
    print(f"      FINDING: against a min-max world, the min-max engine scores "
          f"{acc*100:.1f}% and the utilitarian twin {a_mean*100:.1f}%.")
    print(f"      The worst-member criterion amplifies its own estimation error; "
          f"averaging is the more forgiving plug-in estimator even when the")
    print(f"      world it is estimating is not an average. Sa'di's demand is "
          f"morally sharp and statistically expensive, and both are true.")
    return ok, a_mean


def test_8_gate_ablation(cfg, world, eng, steps):
    """
    A station-blind twin, trained identically, must fail WHERE IT MATTERS.

    In the middle of a room the gate buys little: predicament similarity alone
    already retrieves sensible precedents, and an unmasked twin sometimes edges
    past the gated engine because the mask costs it capacity.

    The gate is built for the SEAM — predicaments lying between two antipodal
    rooms, where an unmasked retriever averages two contradictory bodies of
    precedent into a verdict belonging to neither. But the mask only pays if the
    router is right, because masking converts an AVERAGING error into a
    COMMITMENT error: a mind that has misidentified the room will now be
    confidently, cleanly, entirely wrong. So the test measures the seam twice —
    once unconditionally, once conditioned on the room having been correctly
    identified — and the pass condition is the conditional one, with the
    unconditional gap printed beside it as the price.
    """
    twin = GulistanEngine(cfg, seed=655)
    train(twin, world, steps=steps, quiet=True, lam_gate=0.0)

    def hits(engine, batches, **kw):
        out = []
        for b in batches:
            pred, _, _ = engine.decide(b, world, **kw)
            out.append(pred == b["best"])
        return np.concatenate(out)

    interior = [world.batch(400) for _ in range(8)]
    seam = [world.batch(400, seam=True) for _ in range(8)]

    g_in, o_in = hits(eng, interior), hits(twin, interior, lam_gate=0.0)
    g_sm, o_sm = hits(eng, seam), hits(twin, seam, lam_gate=0.0)
    room_ok = np.concatenate([eng.forward(b, world)["s"].v.argmax(1) == b["station"]
                              for b in seam])

    gi, oi, gs, os_ = g_in.mean(), o_in.mean(), g_sm.mean(), o_sm.mean()
    gsc, osc = g_sm[room_ok].mean(), o_sm[room_ok].mean()
    gsw, osw = g_sm[~room_ok].mean(), o_sm[~room_ok].mean()

    ok = (gsc - osc) > 0.0
    print(f"  [8] station gate ablation .............. seam, room correctly "
          f"identified ({room_ok.mean()*100:.0f}% of cases): gated {gsc*100:.1f}% "
          f"vs blind {osc*100:.1f}% ({(gsc-osc)*100:+.1f})  ->  "
          f"{'PASS' if ok else 'FAIL'}")
    print(f"      seam, room MISidentified: gated {gsw*100:.1f}% vs blind "
          f"{osw*100:.1f}% ({(gsw-osw)*100:+.1f})   |   seam overall "
          f"{gs*100:.1f}% vs {os_*100:.1f}%   |   interior {gi*100:.1f}% vs "
          f"{oi*100:.1f}%")
    print(f"      FINDING: masking converts an averaging error into a commitment "
          f"error. Locate the room correctly and refusing to transfer pays;")
    print(f"      locate it wrongly and the same refusal makes the mind "
          f"confidently, cleanly wrong. The gate does not buy correctness. It")
    print(f"      buys the correctness of the router, at leverage — which is why "
          f"Sa'di spends eight chapters on WHERE you are and not on what is true.")
    return ok, gsc, osc


# =============================================================================
#  PART 6 — MAIN
# =============================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="short smoke run")
    ap.add_argument("--seed", type=int, default=1258)
    args = ap.parse_args()

    steps = 250 if args.quick else 900
    np.set_printoptions(precision=4, suppress=True)
    t0 = time.time()

    print("=" * 78)
    print(" THE GULISTAN ENGINE — Sa'di of Shiraz (c.1210-1291/2), Mind 0269")
    print(" station-gated case-base | bayt press | hearer simulation |")
    print(" intent-modulated truth cost | bani-adam coupling | an inviolable floor")
    print("=" * 78)

    cfg = GulistanConfig()
    world = World(cfg, seed=args.seed)

    print("\n THE EIGHT ROOMS (stations of the Gulistan, 1258):")
    for i, nm in enumerate(STATION_NAMES):
        anti = i + 1 if i % 2 == 0 else i - 1
        print(f"   station {i}  {nm:<32s} verdict antipodal to station {anti}")

    print("\n GRADIENT CHECK (before any training)")
    ok1 = test_1_gradcheck(cfg, world)

    print("\n TRAINING")
    eng = GulistanEngine(cfg, seed=655)
    b0 = world.batch(256)
    rec0 = float(eng.forward(b0, world)["rec"].v)
    held0 = [world.batch(160) for _ in range(4)]
    base, _ = evaluate(eng, world, held0)
    train(eng, world, steps=steps, log_every=max(1, steps // 6))

    held = [world.batch(200) for _ in range(5)]
    acc, veto_rate = evaluate(eng, world, held)

    print("\n SELF-TESTS")
    ok2 = test_2_station_gate(cfg, world, eng)
    ok3 = test_3_floor_is_inviolable(cfg, world, eng)
    ok4 = test_4_intent_modulates_truth(cfg, world, eng)
    ok5 = test_5_bayt_press(cfg, world, eng, rec0)
    ok6 = test_6_learning(cfg, world, eng, acc, base)
    ok7, a_mean = test_7_body_coupling(cfg, world, eng, acc, steps)
    ok8, g_seam, o_seam = test_8_gate_ablation(cfg, world, eng, steps)

    # ---- a single worked predicament, printed in full -----------------------
    print("\n ONE PREDICAMENT, WORKED THROUGH")
    b = world.batch(1)
    out = eng.forward(b, world)
    st = int(out["s"].v.argmax())
    pred, sc, forced = eng.decide(b, world)
    print(f"   room identified ....... {STATION_NAMES[st]}  "
          f"(posterior {out['s'].v[0, st]:.3f}; true room "
          f"{STATION_NAMES[b['station'][0]]})")
    top = np.argsort(-out["alpha"].v[0])[:3]
    print(f"   precedents admitted ... cases {list(top)} with weights "
          f"{np.round(out['alpha'].v[0][top], 3)}")
    print(f"   couplet handle ........ {np.round(out['z'].v[0], 3)}  "
          f"({cfg.d_z} numbers standing for {cfg.d_v})")
    print("   candidates:")
    for c in range(cfg.C):
        flag = "  VETOED" if b["vetoed"][0, c] else ""
        print(f"     {c}: divergence {b['D'][0, c]:.2f}  intent "
              f"{b['I'][0, c]:+.2f}  truth-gate {out['gate'].v[0, c]:.3f}  "
              f"worst-member pain {out['body'].v[0, c]:6.3f}  "
              f"score {out['score'].v[0, c]:+7.3f}{flag}")
    print(f"   chosen ................ candidate {pred[0]}   "
          f"(oracle: {b['best'][0]})"
          f"{'   [FORCED: no lawful option remained]' if forced[0] else ''}")

    allok = all([ok1, ok2, ok3, ok4, ok5, ok6, ok7, ok8])
    print("\n" + "=" * 78)
    print(f" RESULT: {'ALL 8 SELF-TESTS PASS' if allok else 'FAILURES PRESENT'}"
          f"   |   verdict agreement {acc*100:.1f}%   |   "
          f"vetoed selections {veto_rate*100:.2f}%")
    print(f" elapsed {time.time() - t0:.1f}s")
    print("=" * 78)
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())

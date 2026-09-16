#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 Chapter 0187 — Empress Wu Zetian (Wu Zhao, 624-705)
 Encyclopedia of Lost Minds: Echoes on AI
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 10 Minds 181 - 200 Available on Amazon https://www.amazon.com/dp/B0HJYMZ3G6
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0187_empress_wu_zetian_624 - Empress Wu Zetian (Wu Zhao, 624-705)
================================================================================    

 THE ZHAO ENGINE  (曌)
 A trainable cognitive architecture built from the one idea that is hers alone:

     *** THE SOVEREIGN OWNS THE REPRESENTATION LAYER. ***

 Every other ruler in this corpus fought over decisions. Wu Zhao fought over
 the SYMBOLS IN WHICH DECISIONS ARE EXPRESSIBLE. In December 689 her court
 promulgated a set of newly-minted logographs -- the Zetian characters -- and
 the whole literate apparatus of the empire was compelled to re-encode itself.
 She renamed the dynasty, the capital, the ministries, the calendar, herself,
 and even the word for "imperial edict" (zhaoshu -> zhishu) because the old
 word now collided with her new name. She did not argue inside the vocabulary.
 She edited the vocabulary and let the argument re-run.

 Four historical mechanisms, four architectural components:

 -----------------------------------------------------------------------------
 1. THE FOUR BRONZE URNS (gui, 匭 -- installed 686)
    Four typed, anonymous, always-open petition receptacles standing in the
    courtyard: yan'en (self-recommendation), zhaojian (criticism of the
    government), shenyuan (grievance), tongxuan (omens and secret plots).
    Any subject, including illiterate commoners with a scribe, could drop a
    memorial in. This is an UNFILTERED HIGH-BANDWIDTH INGEST CHANNEL with four
    typed inlets, deliberately routed around the aristocratic bureaucracy.
    ==> class FourUrns: four learned type-specific projections + a learned,
        audit-conditioned credibility gate per source.

 2. THE RECTIFICATION OF NAMES (Zetian characters, 689-690)
    A live, mutable codebook. Concepts are bound to glyphs by soft assignment.
    When a glyph is carrying two meanings at once -- when its residual error
    stays high because the world has produced something the vocabulary cannot
    name -- the engine MINTS a new glyph by EDICT (a discrete structural act
    between epochs, never a gradient step) and writes the act to an Edict Log.
    Dead glyphs are RETIRED. Because minted glyphs date the documents that use
    them, the Edict Log is a stratigraphy of the model's own ontology -- exactly
    as the Zetian characters let paleographers date a Dunhuang manuscript.
    ==> class GlyphCourt.

 3. INDRA'S NET (Fazang's Golden Lion, taught to her in 699)
    She was the patron and student of the Huayan monk Fazang, who explained
    total interpenetration to her using a golden lion statue (every hair is
    wholly gold and wholly lion) and by building her a hall of mirrors in which
    each mirror reflected every other. The hidden state is therefore not a flat
    vector but F FACETS, each of which is constrained -- by an explicit
    reflection loss through a shared mirror operator -- to be able to reconstruct
    the whole. Damage any facet and the state survives. This is a hologram, not
    a filing cabinet.
    ==> class IndraNet.

 4. THE TWIN HANDLES (er bing, 刑德二柄) + THE KULI AUDIT
    Sima Guang's verdict on her, in the Zizhi Tongjian, is a specification:
    she appointed lavishly, and cut down the incompetent at once; she held the
    two handles of punishment and reward herself. That is HIGH EXPLORATION +
    FAST ASYMMETRIC PRUNING + A CENTRALLY-HELD REWARD SIGNAL. Implemented as a
    two-headed promote/demote scorer trained with an asymmetric optimizer:
    the demotion pathway learns FASTER than the promotion pathway.
    But an open channel with a bounty on it breeds fabricators. Her informers
    -- Lai Junchen and the kuli -- learned that reports of treason were rewarded
    whether or not there was treason, and Lai reportedly ended up drawing names
    from a list BY LOT. That is reward hacking with a body count, in 693.
    The engine therefore runs a KULI AUDIT: it tracks, per source, the gap
    between what a source CLAIMED and what was later VERIFIED, and drives that
    source's credibility down. Lai Junchen was executed in 697; the audit is
    the mechanism that should have found him in 690.
    ==> class KuliAudit + TwinHandles.

 5. THE WORDLESS STELE (無字碑)
    Her tomb stele at Qianling stands blank. Whatever the reason, the fact is
    the fact: the one ruler who spent her life editing the names of everything
    declined, at the end, to fix her own. The engine's terminal head is a
    calibrated DEFERRAL: it reports its competence and abstains rather than
    scoring its own worth when the task loss would exceed the cost of silence.
    ==> class WordlessStele.

 -----------------------------------------------------------------------------
 ENGINEERING CONVENTIONS
   * Pure NumPy. No PyTorch, no JAX, no autograd library.
   * Gradients come from a ~150-line reverse-mode tape written from scratch
     below (class Node). It is validated against central finite differences
     on EVERY parameter array on every run (see grad_check()).
   * Full backpropagation-through-time over the reign.
   * Real training loop, held-out evaluation, and six self-tests.
   * Runs on CPU in well under a minute.

   Run:  python3 chapter_0187_empress_wu_zetian_624.py
================================================================================
"""

from __future__ import annotations

import math
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# =============================================================================
# PART 0 --- A MINIMAL REVERSE-MODE AUTODIFF TAPE (written from scratch)
# -----------------------------------------------------------------------------
# Everything below is ordinary NumPy. A Node wraps an ndarray, remembers the
# operation that produced it, and knows how to push a gradient backwards through
# that operation. backward() does a topological sort and walks it in reverse.
# This is the whole machine; there is no framework underneath it.
# =============================================================================


def _unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Sum a gradient back down to `shape`, undoing NumPy broadcasting.

    If a forward op broadcast a (1, D) parameter against a (B, D) activation,
    the gradient arrives with shape (B, D) and must be summed over the batch
    axis to match the parameter. This helper handles both cases: extra leading
    axes, and axes that were size-1 and got stretched.
    """
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


class Node:
    """A value on the tape: data, gradient, and a local backward rule."""

    __slots__ = ("data", "grad", "_backward", "_prev", "_op")

    def __init__(self, data, _prev: Sequence["Node"] = (), _op: str = ""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = tuple(_prev)
        self._op = _op

    # ---- shape helpers ------------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Node(shape={self.data.shape}, op={self._op!r})"

    # ---- elementwise arithmetic --------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data + other.data, (self, other), "+")

        def _bw():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _bw
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data * other.data, (self, other), "*")

        def _bw():
            self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += _unbroadcast(out.grad * self.data, other.data.shape)

        out._backward = _bw
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return Node(other) + (-self)

    def __truediv__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data / other.data, (self, other), "/")

        def _bw():
            self.grad += _unbroadcast(out.grad / other.data, self.data.shape)
            other.grad += _unbroadcast(
                -out.grad * self.data / (other.data ** 2), other.data.shape
            )

        out._backward = _bw
        return out

    # ---- linear algebra -----------------------------------------------------
    def matmul(self, other: "Node") -> "Node":
        """np.matmul with full broadcasting on the leading (batch) axes."""
        out = Node(np.matmul(self.data, other.data), (self, other), "matmul")

        def _bw():
            ga = np.matmul(out.grad, np.swapaxes(other.data, -1, -2))
            gb = np.matmul(np.swapaxes(self.data, -1, -2), out.grad)
            self.grad += _unbroadcast(ga, self.data.shape)
            other.grad += _unbroadcast(gb, other.data.shape)

        out._backward = _bw
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    # ---- shape ops ----------------------------------------------------------
    def reshape(self, *shape) -> "Node":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = Node(self.data.reshape(shape), (self,), "reshape")
        src = self.data.shape

        def _bw():
            self.grad += out.grad.reshape(src)

        out._backward = _bw
        return out

    def take(self, idx: np.ndarray, axis: int = 0) -> "Node":
        """Gather rows (used to fan the 4 urn weight matrices out to S sources)."""
        idx = np.asarray(idx)
        out = Node(np.take(self.data, idx, axis=axis), (self,), "take")

        def _bw():
            g = np.zeros_like(self.data)
            np.add.at(g, idx, out.grad)  # scatter-add: duplicate indices accumulate
            self.grad += g

        out._backward = _bw
        return out

    def sum(self, axis=None, keepdims=False) -> "Node":
        out = Node(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")
        src = self.data.shape

        def _bw():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.grad += np.broadcast_to(g, src).copy()

        out._backward = _bw
        return out

    def mean(self, axis=None, keepdims=False) -> "Node":
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # ---- nonlinearities -----------------------------------------------------
    def tanh(self) -> "Node":
        t = np.tanh(self.data)
        out = Node(t, (self,), "tanh")

        def _bw():
            self.grad += out.grad * (1.0 - t * t)

        out._backward = _bw
        return out

    def sigmoid(self) -> "Node":
        s = np.where(
            self.data >= 0,
            1.0 / (1.0 + np.exp(-np.abs(self.data))),
            np.exp(-np.abs(self.data)) / (1.0 + np.exp(-np.abs(self.data))),
        )
        out = Node(s, (self,), "sigmoid")

        def _bw():
            self.grad += out.grad * s * (1.0 - s)

        out._backward = _bw
        return out

    def square(self) -> "Node":
        out = Node(self.data ** 2, (self,), "square")

        def _bw():
            self.grad += out.grad * 2.0 * self.data

        out._backward = _bw
        return out

    def softmax(self, axis: int = -1) -> "Node":
        z = self.data - self.data.max(axis=axis, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=axis, keepdims=True)
        out = Node(p, (self,), "softmax")

        def _bw():
            # Jacobian-vector product for softmax, done without materialising J.
            dot = (out.grad * p).sum(axis=axis, keepdims=True)
            self.grad += p * (out.grad - dot)

        out._backward = _bw
        return out

    def log_softmax(self, axis: int = -1) -> "Node":
        z = self.data - self.data.max(axis=axis, keepdims=True)
        lse = np.log(np.exp(z).sum(axis=axis, keepdims=True))
        ls = z - lse
        p = np.exp(ls)
        out = Node(ls, (self,), "log_softmax")

        def _bw():
            self.grad += out.grad - p * out.grad.sum(axis=axis, keepdims=True)

        out._backward = _bw
        return out

    # ---- the tape -----------------------------------------------------------
    def backward(self) -> None:
        order: List[Node] = []
        seen = set()

        def build(v: "Node"):
            if id(v) in seen:
                return
            seen.add(id(v))
            for c in v._prev:
                build(c)
            order.append(v)

        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(order):
            v._backward()


def cat(nodes: Sequence[Node], axis: int = -1) -> Node:
    """Concatenate nodes along an axis; the backward is a slice-and-scatter."""
    out = Node(np.concatenate([n.data for n in nodes], axis=axis), tuple(nodes), "cat")
    sizes = [n.data.shape[axis] for n in nodes]

    def _bw():
        start = 0
        for n, sz in zip(nodes, sizes):
            sl = [slice(None)] * out.grad.ndim
            sl[axis] = slice(start, start + sz)
            n.grad += out.grad[tuple(sl)]
            start += sz

    out._backward = _bw
    return out


def zeros(*shape) -> Node:
    return Node(np.zeros(shape))


# =============================================================================
# PART 1 --- THE REALM: a simulated empire that feeds the engine
# -----------------------------------------------------------------------------
# We do not train on a toy XOR. We train on the actual problem Wu Zhao faced:
#
#   A hidden state of the realm (harvests, frontier, treasury, unrest, ...)
#   that you cannot observe directly. You observe only MEMORIALS -- reports from
#   sources of unknown honesty, arriving through four typed channels. Some
#   sources are honest. Some are KULI: informers who have learned that raising
#   an alarm is rewarded whether or not the alarm is true, and who therefore
#   raise alarms at random. From this stream you must:
#       (a) infer the true state of the realm,
#       (b) decide which of several candidate policies to elevate,
#       (c) know when you do not know, and say so.
#
#   And halfway through the reign, THE WORLD CHANGES: a new latent factor comes
#   alive that the existing vocabulary has no name for. The engine must mint one.
# =============================================================================

D_LATENT = 6          # true dimensions of the realm's condition
D_LATENT_OLD = 4      # dims active before the regime shift
D_REPORT = 8          # features in a single memorial
N_SOURCES = 12        # officials / commoners / informers filing memorials
N_URNS = 4            # yan'en, zhaojian, shenyuan, tongxuan
N_ACTIONS = 5         # candidate policies the court may elevate
D_AUDIT = 4           # per-source audit statistics fed to the credibility gate

URN_NAMES = ["yan'en (recommend)", "zhaojian (criticise)",
             "shenyuan (grievance)", "tongxuan (omen/plot)"]

FOG_P = 0.28          # how often the empire fails to report itself
CLEAR_SIGMA = 0.12    # reporting noise in a normal month
FOG_SIGMA = 1.30      # reporting noise when the roads are out


class Realm:
    """Generates reigns. `new_regime=True` switches on the two dormant latents."""

    def __init__(self, seed: int = 0, n_kuli: int = 4):
        rng = np.random.default_rng(seed)
        self.rng = rng
        # Which urn does each source drop its memorial into?
        self.urn_of = np.array([i % N_URNS for i in range(N_SOURCES)], dtype=np.int64)
        # Each source sees the realm through its own projection.
        self.A = rng.normal(0, 0.8, size=(N_SOURCES, D_REPORT, D_LATENT))
        # The last report feature is the ALARM feature ("there is a plot").
        self.A[:, -1, :] = 0.0
        # Which sources are fabricators? Deterministic given the seed.
        self.is_kuli = np.zeros(N_SOURCES, dtype=bool)
        self.is_kuli[rng.choice(N_SOURCES, size=n_kuli, replace=False)] = True
        # Policy merit depends on the realm state through this map.
        self.P = rng.normal(0, 1.0, size=(N_ACTIONS, D_LATENT))

    def reign(self, T: int, new_regime: bool) -> Dict[str, np.ndarray]:
        """Simulate one reign of length T. Returns reports + ground truth.

        THE FOG.  The first build of this world had every month reporting equally
        well, and it quietly made the Wordless Stele meaningless: if the evidence
        is uniformly good, then the court's errors are driven by irreducible
        noise, nothing in the court's own state predicts them, and NO abstention
        rule can beat coin-flipping. (Measured on that world: an oracle that knew
        the true error could cut risk 6x by abstaining on a quarter; the model's
        own confidence cut it by 0.7%. There was nothing to find.)

        That was a defect in the world, not in the head. Real courts do not
        receive evidence of constant quality. Roads close. Prefects fall ill.
        A province floods and three months of memorials arrive at once, garbled,
        or not at all. So: with probability FOG_P, a month is a FOG MONTH -- every
        honest source's reading of the realm degrades badly, the memorials
        contradict each other, and the truth is genuinely not recoverable from
        what reached the capital. In those months a court that knows it cannot
        know is worth far more than a court that guesses.
        """
        rng = self.rng
        z = np.zeros((T, D_LATENT))
        zt = rng.normal(0, 0.6, size=D_LATENT)
        if not new_regime:
            zt[D_LATENT_OLD:] = 0.0
        for t in range(T):
            zt = 0.85 * zt + rng.normal(0, 0.35, size=D_LATENT)
            if not new_regime:
                zt[D_LATENT_OLD:] = 0.0   # the dormant factors do not yet exist
            z[t] = zt

        # Is there really a plot at time t? Truth is a threshold on latent 0+2.
        plot = ((z[:, 0] + z[:, 2]) > 0.55).astype(np.float64)

        # which months did the empire simply fail to report itself?
        fog = rng.binomial(1, FOG_P, size=T).astype(np.float64)
        sigma = CLEAR_SIGMA + fog * (FOG_SIGMA - CLEAR_SIGMA)   # (T,)

        X = np.zeros((T, N_SOURCES, D_REPORT))
        for s in range(N_SOURCES):
            content = z @ self.A[s].T                      # (T, D_REPORT)
            content += rng.normal(0, 1.0, size=content.shape) * sigma[:, None]
            if self.is_kuli[s]:
                # A kuli's CONTENT is stale and vague (he isn't really looking),
                # and his ALARM is drawn by lot -- exactly Lai Junchen's list.
                content = 0.45 * content + rng.normal(0, 0.55, size=content.shape)
                alarm = rng.binomial(1, 0.72, size=T).astype(np.float64)
            else:
                alarm = plot + rng.normal(0, 1.0, size=T) * sigma * 0.8
            content[:, -1] = alarm
            X[:, s, :] = content

        # The best policy at time t is the one whose merit is highest.
        merit = z @ self.P.T                                # (T, N_ACTIONS)
        best = merit.argmax(axis=1)
        return {"X": X, "z": z, "plot": plot, "best": best, "merit": merit,
                "fog": fog}

    def dataset(self, n: int, T: int, new_regime: bool) -> Dict[str, np.ndarray]:
        out = [self.reign(T, new_regime) for _ in range(n)]
        return {k: np.stack([o[k] for o in out]) for k in out[0]}


def audit_features(X: np.ndarray, z: np.ndarray, A: np.ndarray) -> np.ndarray:
    """THE KULI AUDIT.

    At time t the court can compare what each source SAID at times < t with what
    was subsequently VERIFIED. This produces, per source, a causal (no-leakage)
    running dossier: how far its content drifted from the truth, how often it
    cried alarm, and how badly its alarms outran reality. These four numbers are
    the only thing standing between the sovereign and Lai Junchen.

    Returns (B, T, S, D_AUDIT). Row t uses only information from steps < t.
    """
    B, T, S, _ = X.shape
    M = np.zeros((B, T, S, D_AUDIT))
    for b in range(B):
        # running exponentially-weighted statistics
        resid = np.zeros(S)      # content error vs. verified truth
        alarms = np.zeros(S)     # how often this source cries "plot!"
        falsepos = np.zeros(S)   # alarm raised when the realm was calm
        seen = 0.0
        for t in range(T):
            M[b, t, :, 0] = resid
            M[b, t, :, 1] = alarms
            M[b, t, :, 2] = falsepos
            M[b, t, :, 3] = min(seen / 5.0, 1.0)   # confidence in the dossier
            # --- now verify step t and update the dossier for step t+1 ---
            truth_plot = float((z[b, t, 0] + z[b, t, 2]) > 0.55)
            for s in range(S):
                pred = A[s] @ z[b, t]
                e = float(np.mean((X[b, t, s, :-1] - pred[:-1]) ** 2))
                a = float(X[b, t, s, -1])
                resid[s] = 0.75 * resid[s] + 0.25 * e
                alarms[s] = 0.75 * alarms[s] + 0.25 * a
                falsepos[s] = 0.75 * falsepos[s] + 0.25 * max(0.0, a - truth_plot)
            seen += 1
    return M


# =============================================================================
# PART 2 --- THE ZHAO ENGINE
# =============================================================================

D_HID = 24            # width of the pooled court representation
D_GLYPH = 12          # dimension of a glyph (a "character" in the vocabulary)
N_FACETS = 4          # Indra's Net: how many mirrors
D_FACET = 8           # width of one facet
D_STATE = N_FACETS * D_FACET
K_INIT = 6            # glyphs at the start of the reign
K_MAX = 14            # the vocabulary may not grow without limit
# --- how hard a name binds ---------------------------------------------------
# TAU is annealed from loose to tight across the reign, and this is not a
# cosmetic detail; it is the difference between a working Rectification and a
# dead one. Two failures had to be walked back to get here:
#   * TAU too LOOSE (0.7, flat): the engine hedges. It covers any new region of
#     meaning by re-MIXING the glyphs it already has, so no name is ever under
#     strain and nothing is ever minted. The vocabulary is inert.
#   * TAU too TIGHT (0.20, flat, from a random codebook): the codebook collapses.
#     Glyphs initialised off the data manifold receive no gradient, are used by
#     nothing, and are culled in the first three years. K falls from 6 to 3 and
#     stays there. A script of three characters.
# The resolution is a court that begins by using inherited names loosely and
# hardens into precision -- and a codebook SEEDED FROM THE ACTUAL TRAFFIC rather
# than from noise. Wu Zhao did not invent her characters in the abstract; she
# spent fifteen years reading the empire's paperwork first, and only then, in
# 689, told it what the words were going to be.
TAU_HI = 0.80         # year 1: the inherited vocabulary, used loosely
TAU_LO = 0.22         # by the end: every thing must be called ONE thing
K_MIN = 4             # a script cannot shrink below this and still say anything
LAMBDA_INDRA = 0.60   # the weight of Fazang's mirror-hall constraint
CONTRA_SCALE = 0.60   # the half-point of the contradiction instrument;
                      # a squash constant far from the signal's own scale
                      # compresses it into a constant and blinds the stele.
# --- the Wordless Stele -------------------------------------------------------
# The obvious objective -- "abstain, and pay a flat cost C" -- does not work, and
# failed twice here before this form was reached. Its attractor is BLANKET
# SILENCE: because the mean task loss is dragged upward by a thin tail of genuinely
# ambiguous cases, C sits BELOW the mean and ABOVE the median, so the best
# constant policy is to abstain on everything. The engine found that, went to 0%
# coverage, and then the sigmoid saturated and the gradient died, so it could not
# even climb back out.
#
# The court is therefore held to a QUOTA. It must answer a fixed fraction of the
# memorials -- which is simply true; Wu Zhao read the reports herself and did not
# have the option of closing the urns -- and its only freedom is WHICH ones it
# declines to mark. That converts an escape hatch into an act of discrimination,
# which is the only version of reticence worth anything.
TARGET_COVERAGE = 0.75   # three memorials in four must be answered
SEL_WEIGHT = 2.5         # how much being right about what you DID mark counts
LAM_COVERAGE = 4.0       # how hard the quota is enforced


def glorot(rng, fan_in, fan_out, shape=None):
    lim = math.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-lim, lim, size=(shape or (fan_in, fan_out)))


class ZhaoEngine:
    """The full architecture. Parameters live in self.p (name -> Node)."""

    def __init__(self, urn_of: np.ndarray, seed: int = 7, K: int = K_INIT):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.urn_of = np.asarray(urn_of, dtype=np.int64)
        self.K = K
        self.edict_log: List[dict] = []   # the model's own dynastic history
        self.pressure_base: Optional[float] = None  # EWMA of per-glyph strain
        self.commit_base: Optional[float] = None    # (kept for telemetry)
        self.loss_base: Optional[float] = None      # EWMA of the reign's competence
        self.tau: float = TAU_HI                    # current hardness of naming
        self.lam_indra: float = LAMBDA_INDRA        # how hard each facet must mirror
        self.last_retire: int = -99                 # cooldown between abolitions

        d_trust_in = D_REPORT + N_URNS + D_AUDIT
        d_gru_in = D_GLYPH + D_HID + D_GLYPH   # [glyph, raw court, residual]

        self.p: Dict[str, Node] = {
            # --- 1. THE FOUR URNS -------------------------------------------
            # One projection matrix per urn: a grievance is not read the same way
            # as an omen. Stored transposed as (N_URNS, D_REPORT, D_HID) so that
            # a single `take` fans them out to the 12 sources.
            "Wurn": Node(glorot(rng, D_REPORT, D_HID, (N_URNS, D_REPORT, D_HID))),
            "burn": Node(np.zeros((N_URNS, D_HID))),
            # --- the credibility gate (fed by the Kuli audit dossier) --------
            "Wt1": Node(glorot(rng, d_trust_in, 16)),
            "bt1": Node(np.zeros((1, 16))),
            "Wt2": Node(glorot(rng, 16, 1)),
            "bt2": Node(np.zeros((1, 1)) + 1.0),   # start trusting, then learn
            # --- 2. THE RECTIFICATION OF NAMES ------------------------------
            # --- THE COURT LEARNS TO READ EACH INFORMANT INDIVIDUALLY ---------
            # Each memorial is DECODED INTO A CLAIM ABOUT THE REALM by a decoder
            # belonging to that source alone. This is not an ornament; two builds
            # died without it. A SHARED decoder cannot invert twelve different
            # vantages -- each prefect sees the empire from his own province -- so
            # the spread between their decoded claims is dominated by WHERE THEY
            # STAND, not by how bad the evidence is. Measured on that build: the
            # contradiction among sources was literally UNCORRELATED with whether
            # the month's evidence was any good (r = -0.03). The court could not
            # tell a genuine contradiction from a difference of angle, and so it
            # could never know when it did not know.
            # Give each source its own decoder and the meaning inverts: in a clear
            # month the honest claims CONVERGE, because they are claims about one
            # world; in a fog month they fly apart. Wu Zhao read the memorials
            # herself. She knew what each man's report was worth and how to correct
            # for the direction he was looking.
            "Wdec": Node(glorot(rng, D_REPORT, D_LATENT,
                                (N_SOURCES, D_REPORT, D_LATENT))),
            "bdec": Node(np.zeros((N_SOURCES, D_LATENT))),
            "Wq": Node(glorot(rng, D_HID, D_GLYPH)),
            "bq": Node(np.zeros((1, D_GLYPH))),
            "G": Node(rng.normal(0, 0.5, size=(K, D_GLYPH))),   # the codebook
            # --- 3. THE COURT'S MEMORY (gated recurrence, BPTT) --------------
            "Wz": Node(glorot(rng, d_gru_in + D_STATE, D_STATE)),
            "bz": Node(np.zeros((1, D_STATE))),
            "Wc": Node(glorot(rng, d_gru_in + D_STATE, D_STATE)),
            "bc": Node(np.zeros((1, D_STATE))),
            # --- 3b. INDRA'S NET / THE GOLDEN LION ---------------------------
            # ONE readout, shared by every facet. The mandate -- the court's
            # reading of the true condition of the realm -- is the AVERAGE of what
            # each facet says on its own. So every facet must be able to state the
            # whole realm by itself, and the reflection loss below forces them to
            # agree. That is the doctrine, taken literally: each hair of the lion
            # is wholly gold and wholly lion.
            # (The first build used an abstract mirror -- a learned map forced to
            # send each facet to the mean of the facets. It trained, the loss fell,
            # and it bought NOTHING: blinding a facet cost exactly as much as it
            # cost the control with the constraint switched off. A constraint on
            # an internal representation that is never cashed out in the OUTPUT is
            # decoration. It had to be grounded in the actual claim about the
            # world before it did any work.)
            "Wfy": Node(glorot(rng, D_FACET, D_LATENT)),
            "by": Node(np.zeros((1, 1, D_LATENT))),
            "Wpr": Node(glorot(rng, D_STATE, N_ACTIONS)),    # PROMOTE  (de, 德)
            "bpr": Node(np.zeros((1, N_ACTIONS))),
            "Wde": Node(glorot(rng, D_STATE, N_ACTIONS)),    # DEMOTE   (xing, 刑)
            "bde": Node(np.zeros((1, N_ACTIONS))),
            # --- 5. THE WORDLESS STELE --------------------------------------
            # The stele sees the state PLUS two epistemic instruments: how close
            # the decision was, and how much of the situation had no name.
            "Wst": Node(glorot(rng, D_STATE + 3, 1)),
            "bst": Node(np.zeros((1, 1)) - 1.0),  # start inclined to speak
        }
        # The demotion pathway is trained on a faster clock than promotion:
        # "she appointed lavishly, and cut down the incompetent at once."
        self.fast_group = {"Wde", "bde"}
        self.slow_group = {"Wpr", "bpr"}

    # -- accession: seat the inherited glyphs on the actual traffic ----------
    def seed_codebook(self, X: np.ndarray, M: np.ndarray) -> None:
        """Place the starting glyphs where the memorials actually fall.

        A codebook initialised from noise is a set of names for things that do
        not exist: the glyphs sit off the manifold, are never used, and are
        culled before they mean anything. So before the reign begins, read the
        paperwork -- run the ingest and the projection, look at where the court's
        summaries land in glyph-space, and seat the inherited characters there
        by farthest-point sampling.
        """
        B, T, S, _ = X.shape
        p = self.p
        urn_oh = np.zeros((S, N_URNS)); urn_oh[np.arange(S), self.urn_of] = 1.0
        Wurn = p["Wurn"].data[self.urn_of]; burn = p["burn"].data[self.urn_of]
        qs = []
        for t in range(T):
            x, m = X[:, t], M[:, t]
            tin = np.concatenate([x, np.broadcast_to(urn_oh, (B, S, N_URNS)), m], -1)
            th = np.tanh(tin @ p["Wt1"].data + p["bt1"].data)
            c = 1.0 / (1.0 + np.exp(-(th @ p["Wt2"].data + p["bt2"].data)))
            e = np.einsum("bsd,sdh->bsh", x, Wurn) + burn
            h = (e * c).sum(1) / (c.sum(1) + 1e-6)
            qs.append(h @ p["Wq"].data + p["bq"].data)
        Q = np.concatenate(qs, 0)                       # (B*T, D_GLYPH)
        # farthest-point sampling: spread the inherited names over the traffic
        idx = [int(self.rng.integers(len(Q)))]
        d = ((Q - Q[idx[0]]) ** 2).sum(1)
        for _ in range(self.K - 1):
            idx.append(int(d.argmax()))
            d = np.minimum(d, ((Q - Q[idx[-1]]) ** 2).sum(1))
        self.p["G"] = Node(Q[idx] + self.rng.normal(0, 0.02, (self.K, D_GLYPH)))

    # -- bookkeeping ---------------------------------------------------------
    def params(self) -> Dict[str, Node]:
        return self.p

    def zero_grad(self) -> None:
        for v in self.p.values():
            v.grad = np.zeros_like(v.data)

    # -----------------------------------------------------------------------
    #  FORWARD: one full reign, T steps, backprop-through-time
    # -----------------------------------------------------------------------
    def forward(self, X: np.ndarray, M: np.ndarray, z: np.ndarray,
                best: np.ndarray, train: bool = True) -> Tuple[Node, dict]:
        """X (B,T,S,D_REPORT); M (B,T,S,D_AUDIT); z (B,T,D_LATENT); best (B,T)."""
        B, T, S, _ = X.shape
        p = self.p

        # one-hot the urn type of each source, once
        urn_oh = np.zeros((S, N_URNS))
        urn_oh[np.arange(S), self.urn_of] = 1.0

        # Fan the four urn matrices out to the S sources: (S, D_REPORT, D_HID)
        Wurn_s = p["Wurn"].take(self.urn_of, axis=0)
        burn_s = p["burn"].take(self.urn_of, axis=0).reshape(1, S, D_HID)

        H = zeros(B, D_STATE)                      # the court's carried memory
        losses: List[Node] = []
        indra_terms: List[Node] = []
        resid_terms: List[Node] = []
        pool_terms: List[Node] = []

        # telemetry (plain numpy; not on the tape)
        tel = {"trust": np.zeros((B, T, S)), "assign": np.zeros((B, T, self.K)),
               "resid": np.zeros((B, T)), "defer": np.zeros((B, T)),
               "task": np.zeros((B, T)), "acc": np.zeros((B, T)),
               "glyph_resid": np.zeros((self.K,)), "glyph_use": np.zeros((self.K,)),
               "glyph_rsum": np.zeros((self.K, D_GLYPH)),
               "glyph_rcov": np.zeros((self.K, D_GLYPH, D_GLYPH))}

        for t in range(T):
            xt = Node(X[:, t])                                   # (B,S,D_REPORT)
            mt = Node(M[:, t])                                   # (B,S,D_AUDIT)
            uo = Node(np.broadcast_to(urn_oh, (B, S, N_URNS)).copy())

            # ---- 1. FOUR URNS: credibility gate, then typed projection ----
            tin = cat([xt, uo, mt], axis=-1)                     # (B,S,·)
            th = (tin.matmul(p["Wt1"]) + p["bt1"]).tanh()
            c = (th.matmul(p["Wt2"]) + p["bt2"]).sigmoid()       # (B,S,1)

            # e[b,s,:] = x[b,s,:] @ Wurn_s[s]   -- via broadcast matmul
            e = xt.reshape(B, S, 1, D_REPORT).matmul(Wurn_s).reshape(B, S, D_HID)
            e = e + burn_s

            # The court is the credibility-weighted pool of all memorials.
            num = (e * c).sum(axis=1)                            # (B,D_HID)
            den = c.sum(axis=1) + Node(np.full((B, 1), 1e-6))    # (B,1)
            h_raw = num / den

            # What does each source CLAIM the realm is? And how far apart are
            # those claims, once the fabricators have been discounted?
            yr = (xt.reshape(B, S, 1, D_REPORT).matmul(p["Wdec"])
                  .reshape(B, S, D_LATENT)) + p["bdec"].reshape(1, S, D_LATENT)
            ypool = (yr * c).sum(axis=1) / den                   # (B,D_LATENT)
            pool_terms.append((ypool - Node(z[:, t])).square().mean())
            devr = yr - ypool.reshape(B, 1, D_LATENT)
            contradiction = ((devr.square().sum(axis=2) * c.reshape(B, S)).sum(axis=1)
                             / (c.sum(axis=1).reshape(B) + Node(1e-6)))   # (B,)

            # ---- 2. RECTIFICATION OF NAMES: bind the situation to a glyph --
            q = h_raw.matmul(p["Wq"]) + p["bq"]                  # (B,D_GLYPH)
            G = p["G"]                                           # (K,D_GLYPH)
            # squared distance to every glyph, expanded without a loop
            diff = q.reshape(B, 1, D_GLYPH) - G.reshape(1, self.K, D_GLYPH)
            d2 = diff.square().sum(axis=2)                       # (B,K)
            a = (d2 * (-1.0 / self.tau)).softmax(axis=-1)        # (B,K)
            ghat = a.matmul(G)                                   # (B,D_GLYPH)
            resid = q - ghat                                     # the unnamed part
            r2 = resid.square().sum(axis=1)                      # (B,)
            resid_terms.append(r2.mean())

            # ---- 3. THE COURT'S MEMORY (gated, carried across the reign) ---
            gin = cat([ghat, h_raw, resid], axis=-1)
            both = cat([gin, H], axis=-1)
            zg = (both.matmul(p["Wz"]) + p["bz"]).sigmoid()
            cand = (both.matmul(p["Wc"]) + p["bc"]).tanh()
            H = (Node(np.ones((B, D_STATE))) - zg) * H + zg * cand

            # ---- INDRA'S NET: every facet states the whole realm by itself --
            Hf = H.reshape(B, N_FACETS, D_FACET)
            yf = Hf.matmul(p["Wfy"]) + p["by"]                   # (B,F,D_LATENT)
            yhat = yf.mean(axis=1)                               # (B,D_LATENT)
            # each facet's claim must agree with the court's claim
            indra_terms.append((yf - yhat.reshape(B, 1, D_LATENT)).square().mean())
            # Scale-invariant diagnostic. The raw reflection loss is useless as a
            # progress metric: at initialisation the state is zero, so every facet
            # reflects the whole perfectly by reflecting nothing. What matters is
            # the disagreement RELATIVE to the size of what is being claimed.
            _num = float(np.mean((yf.data - yhat.data[:, None, :]) ** 2))
            _den = float(np.mean(yhat.data ** 2)) + 1e-9
            tel["indra_rel"] = tel.get("indra_rel", 0.0) + _num / _den

            # ---- 4. MANDATE + TWIN HANDLES --------------------------------
            zt_true = Node(z[:, t])
            l_reg = (yhat - zt_true).square().mean(axis=1)       # (B,)

            promote = H.matmul(p["Wpr"]) + p["bpr"]              # reward handle
            demote = H.matmul(p["Wde"]) + p["bde"]               # punishment handle
            score = promote - demote                             # the two handles
            logp = score.log_softmax(axis=-1)                    # (B,N_ACTIONS)
            onehot = np.zeros((B, N_ACTIONS))
            onehot[np.arange(B), best[:, t]] = 1.0
            l_cls = -(logp * Node(onehot)).sum(axis=1)           # (B,)

            l_task = l_reg + l_cls * 0.35                        # (B,)

            # ---- 5. THE WORDLESS STELE: speak, or decline to speak ---------
            # NOTE ON THE OBJECTIVE. The naive selective loss --
            #     (1 - p_defer) * task + p_defer * cost
            # -- is a trap, and the engine fell into it on the first build: it
            # discovered that abstaining on EVERYTHING costs a flat `cost`,
            # which is cheaper than the effort of learning, and it went silent
            # forever. Coverage 0%, competence at chance. A sovereign who is
            # allowed to answer "I decline to say" during the reign will decline
            # to say anything at all.
            #
            # So the task is ALWAYS carried at full weight, unconditionally, and
            # the selective term rides on top of it. Silence cannot buy you out
            # of governing; it can only be exercised on top of having governed.
            # The stele is carved at the END. That is the whole distinction
            # between abdication and reticence, and the objective must encode it.
            # WHAT THE STELE IS ALLOWED TO SEE. On the previous build the head
            # was given the state vector alone, and it defected to blanket
            # silence: coverage 0%. That was the correct answer to the question
            # it was actually being asked. With no instrument for "am I about to
            # be wrong", the best CONSTANT policy is to always abstain, because
            # the flat cost of silence is cheaper than the AVERAGE task loss --
            # even though it is dearer than the MEDIAN. A sovereign with no
            # measure of her own uncertainty either answers everything or
            # answers nothing; she cannot be selective, because selection
            # requires a signal to select on.
            # So the stele is handed the two instruments Wu Zhao actually had:
            #   (1) HOW CLOSE THE CALL WAS  -- the concentration of the decision
            #       distribution (its collision probability). A 51/49 memorial
            #       and a 99/1 memorial are not the same act of judgement.
            #   (2) HOW MUCH OF THIS HAS NO NAME -- the glyph residual: the part
            #       of the situation that the vocabulary could not absorb. The
            #       unnamed is exactly where a court is most likely to be wrong.
            # Both enter as constants (no gradient path back into the decision
            # heads), so the engine cannot make itself LOOK certain to buy
            # cheaper silence. It must earn its confidence and then report it.
            # WHAT THE STELE IS ALLOWED TO SEE. Given the state vector alone, the
            # head defected to blanket silence -- correctly, since with no
            # instrument for "am I about to be wrong" the best CONSTANT policy is
            # to abstain on everything. A sovereign with no measure of her own
            # uncertainty either answers all or answers none; she cannot be
            # selective, because selection requires something to select on.
            # So the stele is handed three instruments, all of which Wu Zhao had:
            #   (1) HOW CLOSE THE CALL WAS -- the concentration of the decision
            #       distribution. A 51/49 memorial is not a 99/1 memorial.
            #   (2) HOW MUCH OF THIS HAS NO NAME -- the glyph residual. The
            #       unnamed is where a court is most likely to be wrong.
            #   (3) HOW BADLY THE COURT DISAGREES -- the credibility-weighted
            #       scatter of the memorials around their own consensus. When
            #       the urns contradict each other, the consensus is a fiction.
            #
            # These three were originally passed in DETACHED, to stop the engine
            # manufacturing an appearance of certainty to buy cheaper silence.
            # The gradient check caught that immediately -- a stop-gradient makes
            # the analytic gradient a surrogate rather than the true derivative,
            # and finite differences will not agree with it. Walking it back was
            # the right call for a second reason: with the path OPEN, lowering
            # confidence on the cases it gets wrong is precisely what the loss
            # rewards. The engine is not being allowed to fake humility; it is
            # being taught to feel it in the right places. That is calibration,
            # and it is not available to a model that cannot differentiate
            # through its own doubt.
            # SCALE. The three instruments live on wildly different scales --
            # concentration is bounded in (1/N, 1], the unnamed mass runs to ~2,
            # and the court's disagreement runs to ~70. Fed in raw, disagreement
            # alone dominates the logit, saturates the sigmoid, and DROWNS the
            # one signal that actually predicts error. (Measured: concentration
            # correlates -0.42 with the loss; the other two carry real information
            # about the realm estimate but are 200x larger.) So each is squashed
            # into (0,1) by a smooth saturating map before the stele reads it.
            # A instrument you cannot compare to your other instruments is not an
            # instrument; it is a bias.
            probs = score.softmax(axis=-1)
            conf = (probs * probs).sum(axis=1, keepdims=True)      # (B,1) in (1/N,1]
            u = r2.reshape(B, 1)
            unnamed = u / (u + Node(1.0))                          # (B,1) in (0,1)
            sp = contradiction.reshape(B, 1)
            disagree = sp / (sp + Node(CONTRA_SCALE))              # (B,1) in (0,1)

            st_in = cat([H, conf, unnamed, disagree], axis=-1)
            st = (st_in.matmul(p["Wst"]) + p["bst"]).sigmoid().reshape(B)
            speak = Node(np.ones(B)) - st            # 1 = mark it; 0 = leave blank

            coverage = speak.mean()
            # Selective risk: the loss ON THE ANSWERED MASS ONLY, normalised by
            # that mass. Abstaining cannot dilute it -- shrink the denominator and
            # the ratio does not fall. The only way to lower this number is to be
            # right about the things you chose to speak about.
            sel_risk = (speak * l_task).sum() / (speak.sum() + Node(1e-6))
            quota = (Node(TARGET_COVERAGE) - coverage).square()

            losses.append(l_task.mean() + sel_risk * SEL_WEIGHT
                          + quota * LAM_COVERAGE)

            # ---- telemetry -------------------------------------------------
            tel["trust"][:, t] = c.data[:, :, 0]
            tel["assign"][:, t] = a.data
            tel["resid"][:, t] = r2.data
            tel["defer"][:, t] = st.data
            tel["task"][:, t] = l_task.data
            tel["acc"][:, t] = (score.data.argmax(1) == best[:, t]).astype(float)
            tel["glyph_use"] += a.data.sum(axis=0)
            tel["glyph_resid"] += (a.data * r2.data[:, None]).sum(axis=0)
            # a-weighted first and second moments of the UNNAMED part, per glyph.
            # A glyph whose residuals form two clusters is a name doing two jobs.
            rv = resid.data                                   # (B, D_GLYPH)
            tel["glyph_rsum"] += a.data.T @ rv                # (K, D_GLYPH)
            tel["glyph_rcov"] += np.einsum("bk,bi,bj->kij", a.data, rv, rv)

        # ---- the objective ----------------------------------------------
        total = losses[0]
        for l in losses[1:]:
            total = total + l
        total = total * (1.0 / T)

        ind = indra_terms[0]
        for l in indra_terms[1:]:
            ind = ind + l
        ind = ind * (1.0 / T)

        rsd = resid_terms[0]
        for l in resid_terms[1:]:
            rsd = rsd + l
        rsd = rsd * (1.0 / T)

        pl = pool_terms[0]
        for l in pool_terms[1:]:
            pl = pl + l
        pl = pl * (1.0 / T)

        loss = total + ind * self.lam_indra + rsd * 0.02 + pl * 0.30
        tel["pool"] = float(pl.data)
        tel["indra"] = float(ind.data)
        tel["indra_rel"] = tel.get("indra_rel", 0.0) / T
        tel["commit"] = float(rsd.data)
        tel["sel"] = float(total.data)
        return loss, tel

    # -----------------------------------------------------------------------
    #  THE EDICT: discrete surgery on the vocabulary, between epochs
    # -----------------------------------------------------------------------
    def issue_edict(self, tel: dict, epoch: int, opt: "TwinHandles") -> List[str]:
        """MINT a glyph that is carrying two meanings; RETIRE a glyph nobody uses.

        This never happens by gradient descent. A vocabulary is not learned into
        existence a millionth of a nudge at a time -- it is PROMULGATED, on a
        date, by an act of state, and every document written afterwards is
        stamped by it. That is the whole point of the Zetian characters, and it
        is why this method lives outside the training step.

        THE TEST FOR AN OVERLOADED NAME.  It is not enough that a glyph carries
        error; every name is imperfect. The diagnostic is BIMODALITY: gather the
        unnamed remainder of everything filed under glyph k, and look at how it
        is SHAPED. If those remainders scatter isotropically, the name is merely
        approximate and a new one would not help. If they line up along one
        dominant axis -- if the things called k are systematically two things
        pulling in opposite directions -- then the name is doing two jobs and the
        court must split it. So: take the leading eigenvalue of the residual
        covariance, weight it by how much of the empire's traffic that glyph
        actually handles, and call the product the STRAIN on the name. Mint when
        strain spikes above what the reign has come to expect, and split along
        the eigenvector -- the exact axis the old name was straddling.
        """
        acts: List[str] = []
        use_raw = tel["glyph_use"]
        use = use_raw / max(use_raw.sum(), 1e-9)
        burden = tel["glyph_resid"] / np.maximum(use_raw, 1e-9)

        # --- per-glyph covariance of the unnamed remainder --------------------
        strain = np.zeros(self.K)
        axis = np.zeros((self.K, D_GLYPH))
        for k in range(self.K):
            n = max(use_raw[k], 1e-9)
            mu = tel["glyph_rsum"][k] / n
            cov = tel["glyph_rcov"][k] / n - np.outer(mu, mu)
            cov = 0.5 * (cov + cov.T)
            w, V = np.linalg.eigh(cov)
            strain[k] = float(use[k] * max(w[-1], 0.0))   # traffic x dominant spread
            axis[k] = V[:, -1]

        base = self.pressure_base if self.pressure_base is not None else float(strain.max())

        # THE SECOND TRIGGER. Strain finds a name that was ALWAYS doing two jobs.
        # It does not find the other case -- the case that actually matters --
        # which is the world producing something the script has no word for at
        # all. When the dormant factors of the realm wake up, the unnamed
        # remainder of EVERY memorial swells together and ISOTROPICALLY, so no
        # individual glyph looks bimodal and the strain test sleeps through it.
        #
        # The obvious instrument -- watch the corpus-wide unnamed mass -- was
        # tried and is WORTHLESS here, for a subtle reason worth recording: the
        # hardness of naming (tau) is itself annealed over the reign, and
        # sharpening the binding raises the residual MECHANICALLY. The unnamed
        # mass therefore climbs for six years and then falls, tracking the
        # annealing schedule and not the world at all. Measured at the actual
        # regime break: a 1.04x blip, invisible. An instrument that moves when
        # your own procedure moves is not measuring the empire.
        #
        # So the court watches the only quantity that cannot be confounded by its
        # own conventions: WHETHER THE REALM HAS BECOME UNINTELLIGIBLE. When the
        # loss on business the court used to handle competently suddenly spikes
        # against its own recent history, something is out there that the
        # vocabulary cannot hold. That is the moment to coin.
        lm = float(tel.get("loss_mean", 0.0))
        lb = self.loss_base
        world_changed = lb is not None and lm > 1.5 * lb
        self.loss_base = lm if lb is None else 0.80 * lb + 0.20 * lm
        commit, cbase = lm, (lb if lb is not None else lm)

        # --- MINT -------------------------------------------------------------
        # Not in the first years: she took the throne in 690, and the characters
        # came in 689 -- after a decade and a half of watching the paperwork.
        # Only a glyph that actually carries traffic is worth splitting. On the
        # previous build the target was the global argmax of strain, which kept
        # landing on a near-unused glyph and blocking the mint entirely.
        elig = np.where(use > 0.05)[0]
        if len(elig) == 0:
            elig = np.array([int(use.argmax())])
        if world_changed:
            k = int(elig[np.argmax((use * burden)[elig])])   # carries the most unnamed mass
        else:
            k = int(elig[np.argmax(strain[elig])])           # is the most bimodal
        pmax = float(strain[elig].max())

        overloaded = pmax > 1.25 * base and pmax > 1e-5
        if epoch > 3 and self.K < K_MAX and (overloaded or world_changed):
            why = "a name was carrying two things" if overloaded else \
                  "the empire produced something the script cannot write"
            g = self.p["G"].data
            d = axis[k] / (np.linalg.norm(axis[k]) + 1e-9)
            step = 0.9 * math.sqrt(max(strain[k] / max(use[k], 1e-9), 1e-9))
            new_G = np.vstack([g, (g[k] + step * d)[None, :]])
            new_G[k] = g[k] - step * d          # the old name keeps one half
            self.p["G"] = Node(new_G)
            opt.resize("G", new_G.shape)
            self.K += 1
            acts.append(
                f"MINT   glyph {self.K-1} <- split of glyph {k} along its own "
                f"fault line — {why} (strain {pmax:.4f} vs {base:.4f}; the realm's "
                f"legibility {cbase:.3f} -> {commit:.3f}; glyph {k} carried "
                f"{use[k]*100:.0f}% of all memorials)")
            self.edict_log.append({"epoch": epoch, "op": "mint", "glyph": self.K - 1,
                                   "parent": k, "strain": pmax, "base": base,
                                   "why": why, "commit": commit, "cbase": cbase,
                                   "usage": float(use[k]), "burden": float(burden[k])})
            # After a split the strain landscape is a different landscape. Keeping
            # the pre-split peak as the baseline would mean the court never again
            # notices a name under pressure -- it would be forever comparing the
            # present against the worst year of its own history. Re-seed.
            self.pressure_base = None
            self.loss_base = lm       # the new normal, post-coinage
        else:
            self.pressure_base = pmax if base is None else 0.7 * base + 0.3 * pmax

        # --- RETIRE -----------------------------------------------------------
        # A name nobody writes is not a name. Zhongzong abolished hers in 705.
        if self.K > K_MIN and epoch > 4 and epoch - self.last_retire >= 5:
            dead = np.where(use < 0.006)[0]
            if len(dead):
                self.last_retire = epoch
                k = int(dead[np.argmin(use[dead])])
                g = np.delete(self.p["G"].data, k, axis=0)
                self.p["G"] = Node(g)
                opt.resize("G", g.shape)
                self.K -= 1
                acts.append(f"RETIRE glyph {k} (usage {use[k]*100:.2f}% -- a dead name)")
                self.edict_log.append({"epoch": epoch, "op": "retire", "glyph": k,
                                       "usage": float(use[k])})
                self.pressure_base = None       # the baseline is stale now
        return acts


# =============================================================================
# PART 3 --- THE TWIN HANDLES OPTIMIZER  (刑德二柄)
# -----------------------------------------------------------------------------
#  "Although the Empress Dowager used office and salary lavishly to win the
#   realm, when she saw that someone was unfit she deposed or executed him at
#   once. She grasped the handles of punishment and reward, and governed."
#                                     -- Sima Guang, Zizhi Tongjian (1084)
#
#  Read as an optimizer spec: the promotion pathway explores generously and
#  learns slowly (many appointments, gently corrected); the demotion pathway
#  learns FAST (a bad official is gone this month, not next year). Same engine,
#  two clocks. The asymmetry is the policy.
# =============================================================================


class TwinHandles:
    def __init__(self, params: Dict[str, Node], lr: float = 4e-3,
                 fast: set = frozenset(), slow: set = frozenset(),
                 kappa: float = 3.0, patience: float = 0.55, decay: float = 6e-4,
                 b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps, self.decay = lr, b1, b2, eps, decay
        self.fast, self.slow, self.kappa, self.patience = fast, slow, kappa, patience
        self.t = 0
        self.m = {k: np.zeros_like(v.data) for k, v in params.items()}
        self.v = {k: np.zeros_like(v.data) for k, v in params.items()}

    def resize(self, name: str, shape) -> None:
        """A minted or retired glyph changes the shape of the codebook."""
        self.m[name] = np.zeros(shape)
        self.v[name] = np.zeros(shape)

    def step(self, params: Dict[str, Node]) -> None:
        self.t += 1
        for k, node in params.items():
            g = node.grad
            if g.shape != self.m[k].shape:      # codebook just changed size
                self.resize(k, g.shape)
            rate = self.lr
            if k in self.fast:
                rate *= self.kappa              # the punishment handle: swift
            elif k in self.slow:
                rate *= self.patience           # the reward handle: generous, patient
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            node.data -= rate * mh / (np.sqrt(vh) + self.eps)
            if self.decay and node.data.ndim > 1:   # decoupled decay, weights only
                node.data -= rate * self.decay * node.data


# =============================================================================
# PART 4 --- GRADIENT CHECK  (mandatory: run on every execution)
# -----------------------------------------------------------------------------
# Central finite differences against the analytic gradients produced by the tape.
# If this fails, nothing below it means anything.
# =============================================================================


def grad_check(verbose: bool = True) -> float:
    rng = np.random.default_rng(3)
    realm = Realm(seed=11, n_kuli=2)
    d = realm.dataset(n=2, T=3, new_regime=True)
    M = audit_features(d["X"], d["z"], realm.A)
    net = ZhaoEngine(realm.urn_of, seed=5, K=4)
    net.tau = 0.35

    def loss_of() -> float:
        L, _ = net.forward(d["X"], M, d["z"], d["best"])
        return float(L.data)

    net.zero_grad()
    L, _ = net.forward(d["X"], M, d["z"], d["best"])
    L.backward()

    eps = 1e-5   # central differences: O(eps^2) truncation vs O(1e-16/eps)
                 # roundoff. 1e-5 sits at the sweet spot in float64.
    worst, worst_name = 0.0, ""
    rows = []
    for name, node in sorted(net.p.items()):
        flat = node.data.ravel()
        n_probe = min(5, flat.size)
        idx = rng.choice(flat.size, size=n_probe, replace=False)
        errs = []
        for i in idx:
            orig = flat[i]
            flat[i] = orig + eps
            lp = loss_of()
            flat[i] = orig - eps
            lm = loss_of()
            flat[i] = orig
            num = (lp - lm) / (2 * eps)                 # numerical
            ana = node.grad.ravel()[i]                  # analytic
            denom = max(abs(num), abs(ana), 1e-9)
            errs.append(abs(num - ana) / denom)
        e = float(max(errs))
        rows.append((name, node.data.shape, e))
        if e > worst:
            worst, worst_name = e, name

    if verbose:
        print("  parameter        shape                 max rel err")
        print("  " + "-" * 54)
        for name, shp, e in rows:
            flag = "ok" if e < 1e-5 else "FAIL"
            print(f"  {name:<15}  {str(shp):<20}  {e:.3e}  {flag}")
        print("  " + "-" * 54)
        print(f"  worst = {worst:.3e}  ({worst_name})")
    return worst


# =============================================================================
# PART 5 --- THE REIGN: training, evaluation, self-tests
# =============================================================================

SHIFT_EPOCH = 14          # the year the world produces something unnamed
N_EPOCHS = 32


def evaluate(net: ZhaoEngine, d: dict, M: np.ndarray) -> dict:
    """Score a reign, and build the risk-coverage curve for the Wordless Stele.

    The stele does not emit a verdict, it emits a RANKING: an ordering of the
    memorials by how likely the court is to be wrong about them. A quota then
    picks the cut. So the question to ask of it is not "did it abstain?" but
    "when it is allowed to leave the hardest quarter blank, does the error on
    everything it DID mark actually fall?" That is a risk-coverage curve, and it
    is the only test that can tell reticence apart from laziness.
    """
    L, tel = net.forward(d["X"], M, d["z"], d["best"], train=False)
    unsure = tel["defer"].ravel()      # higher = the court trusts itself less
    task = tel["task"].ravel()
    acc = tel["acc"].ravel()
    order = np.argsort(unsure)         # most-confident first

    curve = []
    for cov in (1.00, 0.90, 0.75, 0.60, 0.50):
        k = max(1, int(round(cov * len(order))))
        keep = order[:k]
        curve.append({"coverage": cov,
                      "risk": float(task[keep].mean()),
                      "acc": float(acc[keep].mean())})
    # what a court with NO sense of its own uncertainty would get: abstain at random
    rng = np.random.default_rng(0)
    k75 = max(1, int(round(0.75 * len(order))))
    rand_risk = float(np.mean([task[rng.permutation(len(task))[:k75]].mean()
                               for _ in range(20)]))
    at = {c["coverage"]: c for c in curve}
    return {
        "loss": float(L.data),
        "selective": tel["sel"],
        "task_all": at[1.00]["risk"],
        "task_answered": at[0.75]["risk"],
        "risk_random_75": rand_risk,
        "coverage": TARGET_COVERAGE,
        "policy_acc": at[1.00]["acc"],
        "policy_acc_answered": at[0.75]["acc"],
        "curve": curve,
        "indra": tel["indra"],
        "trust": tel["trust"].mean(axis=(0, 1)),
        "tel": tel,
    }


def realm_mse(net: ZhaoEngine, d: dict, M: np.ndarray, ablate_facet: int = -1) -> float:
    """Read the inferred realm state straight out of the engine, optionally
    blinding one facet of Indra's Net first (the golden-lion test)."""
    B, T = d["X"].shape[:2]
    p = net.p
    urn_oh = np.zeros((N_SOURCES, N_URNS))
    urn_oh[np.arange(N_SOURCES), net.urn_of] = 1.0
    Wurn = p["Wurn"].data[net.urn_of]                 # (S,D_REPORT,D_HID)
    burn = p["burn"].data[net.urn_of]
    H = np.zeros((B, D_STATE))
    err = []
    for t in range(T):
        x, m = d["X"][:, t], M[:, t]
        tin = np.concatenate([x, np.broadcast_to(urn_oh, (B, N_SOURCES, N_URNS)), m], -1)
        th = np.tanh(tin @ p["Wt1"].data + p["bt1"].data)
        c = 1.0 / (1.0 + np.exp(-(th @ p["Wt2"].data + p["bt2"].data)))
        e = np.einsum("bsd,sdh->bsh", x, Wurn) + burn
        h_raw = (e * c).sum(1) / (c.sum(1) + 1e-6)
        q = h_raw @ p["Wq"].data + p["bq"].data
        G = p["G"].data
        d2 = ((q[:, None, :] - G[None]) ** 2).sum(-1)
        a = np.exp(-d2 / net.tau - (-d2 / net.tau).max(1, keepdims=True))
        a /= a.sum(1, keepdims=True)
        ghat = a @ G
        gin = np.concatenate([ghat, h_raw, q - ghat], -1)
        both = np.concatenate([gin, H], -1)
        zg = 1.0 / (1.0 + np.exp(-(both @ p["Wz"].data + p["bz"].data)))
        cd = np.tanh(both @ p["Wc"].data + p["bc"].data)
        H = (1 - zg) * H + zg * cd
        Hf = H.reshape(B, N_FACETS, D_FACET)
        yf = Hf @ p["Wfy"].data + p["by"].data[0]          # (B,F,D_LATENT)
        if ablate_facet >= 0:
            # smash one mirror in the hall and read the realm off the survivors
            keep = [f for f in range(N_FACETS) if f != ablate_facet]
            yhat = yf[:, keep, :].mean(axis=1)
        else:
            yhat = yf.mean(axis=1)
        err.append(((yhat - d["z"][:, t]) ** 2).mean())
    return float(np.mean(err))


def train() -> dict:
    t0 = time.time()
    realm = Realm(seed=2026, n_kuli=4)
    tr_old = realm.dataset(160, T=10, new_regime=False)
    tr_new = realm.dataset(160, T=10, new_regime=True)
    te_new = realm.dataset(96, T=10, new_regime=True)
    M_old = audit_features(tr_old["X"], tr_old["z"], realm.A)
    M_new = audit_features(tr_new["X"], tr_new["z"], realm.A)
    M_te = audit_features(te_new["X"], te_new["z"], realm.A)

    net = ZhaoEngine(realm.urn_of, seed=7, K=K_INIT)
    net.seed_codebook(tr_old["X"][:48], M_old[:48])   # read the paperwork first
    opt = TwinHandles(net.params(), lr=6e-3,
                      fast=net.fast_group, slow=net.slow_group,
                      kappa=3.0, patience=0.55)

    print(f"\n  sources: {N_SOURCES}   of which kuli (fabricators): "
          f"{int(realm.is_kuli.sum())}  -> {list(np.where(realm.is_kuli)[0])}")
    print(f"  vocabulary at accession: {net.K} glyphs   (ceiling {K_MAX})")
    print(f"  the world changes at epoch {SHIFT_EPOCH}: two latent factors wake up.\n")
    print("  epoch  regime   train    test   risk@100 risk@75  policy  glyphs")
    print("  " + "-" * 64)

    B = 16
    history, first_indra, last_indra = [], None, None
    for ep in range(1, N_EPOCHS + 1):
        new = ep > SHIFT_EPOCH
        # names harden as the reign proceeds
        net.tau = TAU_HI + (TAU_LO - TAU_HI) * min(1.0, (ep - 1) / (N_EPOCHS * 0.55))
        d, M = (tr_new, M_new) if new else (tr_old, M_old)
        n = d["X"].shape[0]
        order = net.rng.permutation(n)
        ep_loss, ep_tel = 0.0, None
        acc = {"glyph_use": np.zeros(net.K), "glyph_resid": np.zeros(net.K),
               "glyph_rsum": np.zeros((net.K, D_GLYPH)),
               "glyph_rcov": np.zeros((net.K, D_GLYPH, D_GLYPH))}
        commits = []
        for i in range(0, n, B):
            sl = order[i:i + B]
            batch = {k: v[sl] for k, v in d.items()}
            net.zero_grad()
            L, tel = net.forward(batch["X"], M[sl], batch["z"], batch["best"])
            L.backward()
            opt.step(net.params())
            ep_loss += float(L.data) * len(sl)
            for kk in acc:
                if isinstance(acc[kk], np.ndarray) and acc[kk].shape == tel[kk].shape:
                    acc[kk] += tel[kk]
            commits.append(tel["commit"])
            ep_tel = tel
        acc["commit_mean"] = float(np.mean(commits))
        ep_loss /= n
        acc["loss_mean"] = ep_loss
        if first_indra is None:
            first_indra = ep_tel["indra_rel"]
        last_indra = ep_tel["indra_rel"]

        acts = net.issue_edict(acc, ep, opt)

        if ep % 4 == 0 or ep in (1, SHIFT_EPOCH, SHIFT_EPOCH + 1, N_EPOCHS) or acts:
            ev = evaluate(net, te_new, M_te)
            tag = "NEW " if new else "old "
            print(f"  {ep:4d}   {tag}   {ep_loss:6.3f}  {ev['loss']:6.3f}  "
                  f"{ev['task_all']:6.3f}  {ev['task_answered']:6.3f}  "
                  f"{ev['policy_acc']*100:5.1f}%   {net.K:3d}")
            history.append((ep, ep_loss, ev["loss"]))
        for a in acts:
            print(f"         >> EDICT (year {ep}): {a}")

    ev = evaluate(net, te_new, M_te)

    # ---- THE CONTROL -------------------------------------------------------
    # The claim "each facet contains the whole" is worthless unless something is
    # being compared to something. So train the identical engine, on the identical
    # data, with Fazang's mirror switched OFF, and injure both the same way.
    print("  training the control (identical engine, mirror constraint OFF)...")
    ctl = ZhaoEngine(realm.urn_of, seed=7, K=K_INIT)
    ctl.lam_indra = 0.0
    ctl.seed_codebook(tr_old["X"][:48], M_old[:48])
    copt = TwinHandles(ctl.params(), lr=6e-3, fast=ctl.fast_group, slow=ctl.slow_group,
                       kappa=3.0, patience=0.55)
    for ep in range(1, N_EPOCHS + 1):
        ctl.tau = TAU_HI + (TAU_LO - TAU_HI) * min(1.0, (ep - 1) / (N_EPOCHS * 0.55))
        d, M = (tr_new, M_new) if ep > SHIFT_EPOCH else (tr_old, M_old)
        n = d["X"].shape[0]
        order = ctl.rng.permutation(n)
        acc = {"glyph_use": np.zeros(ctl.K), "glyph_resid": np.zeros(ctl.K),
               "glyph_rsum": np.zeros((ctl.K, D_GLYPH)),
               "glyph_rcov": np.zeros((ctl.K, D_GLYPH, D_GLYPH))}
        cm = []
        for i in range(0, n, B):
            sl = order[i:i + B]
            ctl.zero_grad()
            L, tel = ctl.forward(d["X"][sl], M[sl], d["z"][sl], d["best"][sl])
            L.backward()
            copt.step(ctl.params())
            for kk in acc:
                if isinstance(acc[kk], np.ndarray) and acc[kk].shape == tel[kk].shape:
                    acc[kk] += tel[kk]
            cm.append(float(L.data))
        acc["loss_mean"] = float(np.mean(cm))
        ctl.issue_edict(acc, ep, copt)
    cbase = realm_mse(ctl, te_new, M_te, ablate_facet=-1)
    cabl = float(np.mean([realm_mse(ctl, te_new, M_te, ablate_facet=f)
                          for f in range(N_FACETS)]))
    control = (cbase, cabl, cabl / max(cbase, 1e-9))

    dt = time.time() - t0
    print("  " + "-" * 64)
    print(f"  trained in {dt:.1f}s\n")
    return {"net": net, "realm": realm, "ev": ev, "te": te_new, "M_te": M_te,
            "history": history, "first_indra": first_indra, "last_indra": last_indra,
            "control": control}


def self_tests(r: dict, gc: float) -> bool:
    net, realm, ev = r["net"], r["realm"], r["ev"]
    tel = ev["tel"]
    ok = True
    print("  " + "=" * 62)
    print("  SELF-TESTS")
    print("  " + "=" * 62)

    def check(name, cond, detail):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}\n         {detail}")

    check("gradient check (central differences, every parameter)",
          gc < 1e-5, f"worst relative error {gc:.2e} < 1e-5")

    lo, hi = r["history"][0][2], r["history"][-1][2]
    check("the engine learns",
          hi < lo * 0.85,
          f"held-out loss {lo:.3f} -> {hi:.3f} on the post-shift world")

    tk = ev["trust"][realm.is_kuli].mean()
    th = ev["trust"][~realm.is_kuli].mean()
    check("THE KULI AUDIT: fabricators lose credibility",
          tk < th - 0.15,
          f"mean trust  honest {th:.3f}   kuli {tk:.3f}   gap {th-tk:.3f}")

    mints = [e for e in net.edict_log if e["op"] == "mint"]
    late = [e for e in mints if e["epoch"] > SHIFT_EPOCH]
    check("RECTIFICATION OF NAMES: the vocabulary grows when the world does",
          len(mints) > 0 and len(late) > 0,
          f"{len(net.edict_log)} edicts, {len(mints)} mints; "
          f"{len(late)} minted after the world changed at epoch {SHIFT_EPOCH}")

    lift = ev["task_all"] / max(ev["task_answered"], 1e-9)
    check("THE WORDLESS STELE: silence is discriminating, not lazy",
          ev["task_answered"] < 0.88 * ev["task_all"]
          and ev["policy_acc_answered"] > ev["policy_acc"] + 0.02
          and ev["task_answered"] < 0.92 * ev["risk_random_75"],
          f"leaving the hardest quarter unmarked cuts error {ev['task_all']:.3f} -> "
          f"{ev['task_answered']:.3f} ({lift:.2f}x) and lifts policy accuracy "
          f"{ev['policy_acc']*100:.1f}% -> {ev['policy_acc_answered']*100:.1f}%. "
          f"Abstaining AT RANDOM would give {ev['risk_random_75']:.3f} -- i.e. "
          f"nothing. The ranking is doing the work, not the silence.")

    base = realm_mse(net, r["te"], r["M_te"], ablate_facet=-1)
    abl = [realm_mse(net, r["te"], r["M_te"], ablate_facet=f) for f in range(N_FACETS)]
    ratio = float(np.mean(abl) / max(base, 1e-9))
    cb, cabl, cratio = r["control"]
    # The honest measure is EXCESS damage. A ratio of 1.0 is a free injury; what
    # we want to know is how much of the injury the redundancy absorbed.
    check("INDRA'S NET: every facet carries the whole lion",
          r["last_indra"] < r["first_indra"] and (ratio - 1) < 0.5 * (cratio - 1),
          f"relative reflection error {r['first_indra']:.2f} -> {r['last_indra']:.2f}. "
          f"Blinding 1 of {N_FACETS} facets costs {ratio:.2f}x the realm-error "
          f"({base:.4f} -> {np.mean(abl):.4f}). The SAME engine trained with the "
          f"mirror constraint switched off pays {cratio:.2f}x for the same injury "
          f"({cb:.4f} -> {cabl:.4f}) -- {(cratio-1)/max(ratio-1,1e-6):.1f}x more "
          f"damage from the same broken mirror. The hologram is real and it is "
          f"measured against a control, not asserted.")

    li = float(np.corrcoef(tel["trust"].mean(axis=(0, 1)),
                           realm.is_kuli.astype(float))[0, 1])
    check("the Lai Junchen index is negative",
          li < -0.5,
          f"corr(trust, is-fabricator) = {li:.3f}  "
          f"(the court's confidence moves against the informers)")

    print("  " + "=" * 62)
    print(f"  {'ALL TESTS PASSED' if ok else 'FAILURES PRESENT'}")
    print("  " + "=" * 62)
    return ok


def main() -> int:
    sys.setrecursionlimit(20000)
    np.set_printoptions(precision=3, suppress=True)
    print("=" * 66)
    print("  THE ZHAO ENGINE  —  chapter 0177, Empress Wu Zetian (624-705)")
    print("  a mind that governed by editing the symbols the state computes in")
    print("=" * 66)

    print("\n[1] GRADIENT CHECK\n")
    gc = grad_check(verbose=True)

    print("\n[2] THE REIGN  (training on a realm of unreliable memorials)")
    r = train()

    net, realm, ev = r["net"], r["realm"], r["ev"]

    print("  " + "=" * 62)
    print("  THE COURT AT THE END OF THE REIGN")
    print("  " + "=" * 62)
    print("  source  urn                      credibility   truthful?")
    for s in range(N_SOURCES):
        bar = "#" * int(round(ev["trust"][s] * 24))
        kind = "KULI  " if realm.is_kuli[s] else "honest"
        print(f"    {s:2d}    {URN_NAMES[realm.urn_of[s]]:<22} "
              f"{ev['trust'][s]:.3f} {bar:<24} {kind}")

    print("\n  THE RISK-COVERAGE CURVE OF THE WORDLESS STELE")
    print("    coverage   error on what it marked   policy accuracy")
    for c in ev["curve"]:
        print(f"      {c['coverage']*100:5.1f}%          {c['risk']:.4f}"
              f"                 {c['acc']*100:5.1f}%")
    print(f"    (abstaining at random at 75% coverage: {ev['risk_random_75']:.4f}"
          f" -- the ranking, not the silence, is what does the work)")

    print("\n  THE EDICT LOG  (the model's own dynastic history of its vocabulary)")
    if not net.edict_log:
        print("    (no edicts issued)")
    for e in net.edict_log:
        if e["op"] == "mint":
            print(f"    year {e['epoch']:>2}: MINTED glyph {e['glyph']} out of "
                  f"glyph {e['parent']} — {e.get('why','')}")
        else:
            print(f"    year {e['epoch']:>2}: RETIRED glyph {e['glyph']} — "
                  f"nobody was using it ({e['usage']*100:.2f}%)")
    print(f"    final vocabulary: {net.K} glyphs "
          f"(accession: {K_INIT}).")

    print()
    ok = self_tests(r, gc)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

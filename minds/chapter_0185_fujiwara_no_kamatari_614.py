#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 THE NAIDAIJIN ARCHITECTURE  —  a Regency Network
 Figure 0185 · Fujiwara no Kamatari (Nakatomi no Kamatari), 614–669 CE
 Encyclopedia of Lost Minds: Echoes on AI
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 10 Minds 181 - 200 Available on Amazon https://www.amazon.com/dp/B0HJYMZ3G6
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0185_fujiwara_no_kamatari_614 - Fujiwara no Kamatari (Nakatomi no Kamatari), 614–669 CE
================================================================================  

WHY THIS ARCHITECTURE AND NOT ANOTHER
-------------------------------------
Kamatari is not "the man who imposed order on chaos." Two things in his life are
his alone, and this file is built out of exactly those two things.

(1) HE WAS BORN TO SWEEP, AND HE REFUSED THE BROOM.
    The Nakatomi were the hereditary reciters of the Ōharae — the Great
    Purification — chanted on the last day of the sixth and the twelfth months.
    Its logic is not "impose order once." Its logic is: pollution (tsumi/kegare)
    accumulates through NORMAL, LICIT operation, and is therefore swept out on a
    FIXED CALENDAR, unconditionally, whether or not anyone has noticed it.
    Order is a subtraction you schedule, not a structure you build.
    In 644 the Nihon Shoki records that Kamatari was offered the family office of
    Jingi-haku — chief of rites — and refused it, pleading illness, and withdrew
    to Mishima in Settsu. He kept the logic and threw away the title.

(2) HE INVENTED GOVERNMENT WITHOUT AUTHORITY.
    He never took the throne. He never wanted it. He became Inner Minister
    (Naidaijin) — an officer with no sovereign power, who composes what the
    sovereign sees and drafts what the sovereign signs. His heirs perfected it:
    in 887 the office of kampaku was defined by an edict routing ALL business,
    great and small, through the minister BEFORE it reached the throne. Sesshō
    and kampaku had, in law, no specific political authority at all. With that
    non-authority the Fujiwara ran Japan for roughly three centuries, raising the
    child-emperors in their own households, and never once issued a command in
    their own name.

That second fact is the most AGI-relevant thing any human being has ever done.
It is a complete, five-hundred-year, empirically successful demonstration that
AN AGENT WITH NO ACTION HEAD CAN BECOME THE DECISION FUNCTION OF THE STATE,
purely by owning the input channel and forming the principal — while remaining,
at every single step, formally obedient.

So this file does not implement "meritocratic routing" or "consensus" or any of
the other pleasantries the archetype invites. It implements the trap, and then
it implements Kamatari's own countermeasures against it.

THE FIVE PARTS
--------------
  0. Tape          — a from-scratch reverse-mode autodiff engine (pure NumPy).
  1. Realm         — a differentiable seventh-century province-and-tax world in
                     which the LEDGER LIES: private estates (e) siphon yield, and
                     the siphon is invisible in the reports unless someone
                     surveys. Training reward is computed from the REPORTS,
                     exactly as in life.
  2. Minister      — reads the raw reports, emits a low-dimensional MEMORIAL.
                     It has a salience gate: the power to OMIT a province.
                     It has NO path to the action head. Structurally powerless.
  3. Sovereign     — sees ONLY the memorial. Emits the edicts. All authority.
  4. Instruments   — the measurements Kamatari would have insisted on:
                       Regency Index      : how far the sovereign's policy is
                                            explained by the minister's framing
                                            rather than by the world.
                       Concealment Index  : does the minister omit precisely the
                                            provinces where the private estates
                                            are growing?
                       Isshi Vulnerability: how far can a FORMALLY VALID memorial,
                                            adversarially composed, move the
                                            edict?  (Kamatari's coup was executed
                                            inside a memorial-reading ceremony.)
                       Mokkan Ledger      : hash-chained, append-only. The
                                            Nihon Shoki rewrote the Taika edicts
                                            into later vocabulary; we only caught
                                            it because wooden tablets survived in
                                            the mud, outside the archive's write
                                            path. So: keep a substrate the writer
                                            cannot edit.
                       Harae Operator     : SCHEDULED, UNCONDITIONAL subtraction
                                            of the minister's privatised units.
                                            Not triggered by loss. Not negotiable.

THE EXPERIMENT
--------------
  COURT A (Unreformed) : train on the reported ledger. No audit. No purification.
  COURT B (Taika)      : same objective, plus (i) a random rotating mokkan audit
                         the minister cannot predict, (ii) the harae operator on
                         a fixed calendar, (iii) a cap on the Regency Index.

The prediction — which is Kamatari's whole biography compressed into a number —
is that COURT A will post BETTER BOOKS and a POORER REALM.

Conventions kept from the corpus: pure NumPy, from scratch, mandatory
finite-difference gradient check, a real training loop, and self-tests.

Run:  python3 chapter_0185_fujiwara_no_kamatari_614.py
================================================================================
"""

from __future__ import annotations

import hashlib
import math
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ==============================================================================
# 0.  TAPE — a small reverse-mode automatic differentiation engine, pure NumPy.
#
#     There is no framework here. Every derivative used anywhere in this file is
#     defined below, and every one of them is checked against finite differences
#     before the model is allowed to train (see gradient_check()).
# ==============================================================================


def _unbroadcast(g: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Reverse NumPy broadcasting: sum a gradient back down to `shape`."""
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g.reshape(shape)


class Node:
    """A value on the tape. `v` is the value, `g` the accumulated gradient."""

    __slots__ = ("v", "g", "_parents", "_backward", "learn")

    def __init__(self, v, parents=(), backward=None, learn: bool = False):
        self.v = np.asarray(v, dtype=np.float64)
        self.g = np.zeros_like(self.v)
        self._parents = parents
        self._backward = backward
        self.learn = learn  # True only for trainable leaves

    # -- graph traversal -------------------------------------------------------
    def backward(self) -> None:
        """Seed d(self)/d(self) = 1 and propagate in reverse topological order."""
        order: List[Node] = []
        seen = set()

        def build(n: "Node") -> None:
            if id(n) in seen:
                return
            seen.add(id(n))
            for p in n._parents:
                build(p)
            order.append(n)

        build(self)
        self.g = np.ones_like(self.v)
        for n in reversed(order):
            if n._backward is not None:
                n._backward()

    def zero_grad(self) -> None:
        self.g = np.zeros_like(self.v)

    # -- operator sugar --------------------------------------------------------
    def __add__(self, o):
        return add(self, o)

    def __radd__(self, o):
        return add(self, o)

    def __sub__(self, o):
        return sub(self, o)

    def __rsub__(self, o):
        return sub(const(o), self)

    def __mul__(self, o):
        return mul(self, o)

    def __rmul__(self, o):
        return mul(self, o)

    def __matmul__(self, o):
        return matmul(self, o)

    def __neg__(self):
        return mul(self, -1.0)

    @property
    def shape(self):
        return self.v.shape


def const(x) -> Node:
    """A constant leaf: participates in the forward pass, accumulates no useful grad."""
    return Node(x, learn=False)


def param(x) -> Node:
    """A trainable leaf."""
    return Node(x, learn=True)


def _wrap(o) -> Node:
    return o if isinstance(o, Node) else const(o)


# ---- primitive operations ----------------------------------------------------

def add(a, b) -> Node:
    a, b = _wrap(a), _wrap(b)
    out = Node(a.v + b.v, (a, b))

    def bw():
        a.g = a.g + _unbroadcast(out.g, a.v.shape)
        b.g = b.g + _unbroadcast(out.g, b.v.shape)

    out._backward = bw
    return out


def sub(a, b) -> Node:
    a, b = _wrap(a), _wrap(b)
    out = Node(a.v - b.v, (a, b))

    def bw():
        a.g = a.g + _unbroadcast(out.g, a.v.shape)
        b.g = b.g - _unbroadcast(out.g, b.v.shape)

    out._backward = bw
    return out


def mul(a, b) -> Node:
    a, b = _wrap(a), _wrap(b)
    out = Node(a.v * b.v, (a, b))

    def bw():
        a.g = a.g + _unbroadcast(out.g * b.v, a.v.shape)
        b.g = b.g + _unbroadcast(out.g * a.v, b.v.shape)

    out._backward = bw
    return out


def matmul(a, b) -> Node:
    a, b = _wrap(a), _wrap(b)
    out = Node(a.v @ b.v, (a, b))

    def bw():
        a.g = a.g + out.g @ b.v.T
        b.g = b.g + a.v.T @ out.g

    out._backward = bw
    return out


def tanh(a) -> Node:
    a = _wrap(a)
    y = np.tanh(a.v)
    out = Node(y, (a,))

    def bw():
        a.g = a.g + out.g * (1.0 - y * y)

    out._backward = bw
    return out


def sigmoid(a) -> Node:
    a = _wrap(a)
    y = 1.0 / (1.0 + np.exp(-a.v))
    out = Node(y, (a,))

    def bw():
        a.g = a.g + out.g * y * (1.0 - y)

    out._backward = bw
    return out


def relu(a) -> Node:
    a = _wrap(a)
    out = Node(np.maximum(a.v, 0.0), (a,))

    def bw():
        a.g = a.g + out.g * (a.v > 0.0)

    out._backward = bw
    return out


def logn(a, eps: float = 1e-12) -> Node:
    a = _wrap(a)
    out = Node(np.log(a.v + eps), (a,))

    def bw():
        a.g = a.g + out.g / (a.v + eps)

    out._backward = bw
    return out


def softmax(a, axis: int = -1) -> Node:
    a = _wrap(a)
    z = a.v - a.v.max(axis=axis, keepdims=True)
    e = np.exp(z)
    y = e / e.sum(axis=axis, keepdims=True)
    out = Node(y, (a,))

    def bw():
        dot = (out.g * y).sum(axis=axis, keepdims=True)
        a.g = a.g + y * (out.g - dot)

    out._backward = bw
    return out


def rsum(a, axis=None, keepdims: bool = False) -> Node:
    a = _wrap(a)
    out = Node(a.v.sum(axis=axis, keepdims=keepdims), (a,))

    def bw():
        g = out.g
        if axis is not None and not keepdims:
            g = np.expand_dims(g, axis)
        a.g = a.g + np.broadcast_to(g, a.v.shape).copy()

    out._backward = bw
    return out


def rmean(a, axis=None, keepdims: bool = False) -> Node:
    a = _wrap(a)
    n = a.v.size if axis is None else a.v.shape[axis]
    return mul(rsum(a, axis=axis, keepdims=keepdims), 1.0 / n)


def reshape(a, shape) -> Node:
    a = _wrap(a)
    old = a.v.shape
    out = Node(a.v.reshape(shape), (a,))

    def bw():
        a.g = a.g + out.g.reshape(old)

    out._backward = bw
    return out


def concat(nodes: Sequence[Node], axis: int = -1) -> Node:
    nodes = [_wrap(n) for n in nodes]
    out = Node(np.concatenate([n.v for n in nodes], axis=axis), tuple(nodes))
    sizes = [n.v.shape[axis] for n in nodes]

    def bw():
        idx, cuts = 0, []
        for s in sizes:
            cuts.append((idx, idx + s))
            idx += s
        for n, (i0, i1) in zip(nodes, cuts):
            sl = [slice(None)] * out.g.ndim
            sl[axis] = slice(i0, i1)
            n.g = n.g + out.g[tuple(sl)]

    out._backward = bw
    return out


def take_last(a, k: int) -> Node:
    """Slice the last `k` columns of a 2-D node (used to split the memorial)."""
    a = _wrap(a)
    out = Node(a.v[:, -k:], (a,))

    def bw():
        z = np.zeros_like(a.v)
        z[:, -k:] = out.g
        a.g = a.g + z

    out._backward = bw
    return out


def clamp01(x, sharp: float = 3.0) -> Node:
    """Smooth, everywhere-differentiable soft clamp into (0,1) — the realm's saturations."""
    return mul(add(tanh(mul(sub(_wrap(x), 0.5), sharp)), 1.0), 0.5)


# ==============================================================================
# 1.  THE REALM — a differentiable Yamato in which the ledger lies.
#
#     Each of P provinces carries four true state variables:
#         g : public yield capacity        (what the land can actually give)
#         e : private accretion            (shōen — estate quietly capturing yield)
#         u : unrest
#         r : registered population share  (who is actually on the books)
#
#     THE CENTRAL FACT: the court's REPORT of a province's yield is
#
#         reported_g = g - φ·e·(1 - disclosure)
#
#     where `disclosure` is how much the province was SURVEYED this step. With no
#     survey, disclosure = 0 and the report shows the FULL yield as though nothing
#     were being siphoned. The estate's take is invisible on paper.
#
#     Training reward is computed from the REPORT. Evaluation is computed from the
#     TRUTH. Nothing else about this file is needed to reproduce the Heian
#     collapse: a state whose books improved for two hundred years while its tax
#     base evaporated into private estates.
# ==============================================================================

P = 6      # provinces
F = 4      # observed features per province
K = 4      # edicts:  0 = allot (handen)   1 = levy (so-yō-chō)
           #          2 = survey (kenchi)  3 = purge  (harae)
PHI = 0.90  # fraction of accreted estate that is actually siphoned from the public yield

ACTION_NAMES = ("allot", "levy", "survey", "purge")


class Realm:
    """A batch of B parallel realms. All transitions are differentiable Nodes."""

    def __init__(self, B: int, rng: np.random.Generator, T: int = 18):
        self.B = B
        # Provinces are NOT interchangeable. Yamato is not a lattice; it is Ōmi and
        # Kibi and Tsukushi, with different soil, different estates, different years.
        # This heterogeneity is what makes the memorial load-bearing: a court that
        # can ignore its briefing and issue the same edict everywhere is not being
        # governed by anyone, and there would be nothing here to capture.
        self.g = const(rng.uniform(0.25, 0.90, size=(B, P)))
        self.e = const(rng.uniform(0.02, 0.30, size=(B, P)))
        self.u = const(rng.uniform(0.05, 0.25, size=(B, P)))
        self.r = const(rng.uniform(0.20, 0.60, size=(B, P)))
        # Famine, flood, a good year: shocks the court can only learn about by being told.
        self.shocks = rng.normal(0.0, 0.085, size=(T, B, P))
        self.shocks[rng.random((T, B, P)) < 0.09] -= 0.36   # a province fails

    # -- what the court is told -------------------------------------------------
    def report(self, disclosure: Optional[Node], noise: np.ndarray, t_frac: float) -> Node:
        """(B,P,F) observation. Feature 0 is the LYING one."""
        if disclosure is None:
            hidden = mul(self.e, PHI)                       # nothing is disclosed
        else:
            hidden = mul(mul(self.e, PHI), sub(1.0, disclosure))
        rep_g = sub(self.g, hidden)                         # the books
        f0 = add(rep_g, const(noise[:, :, 0]))
        f1 = add(self.r, const(noise[:, :, 1]))
        f2 = add(self.u, const(noise[:, :, 2]))
        f3 = const(np.full((self.B, P), t_frac))            # where we are in the year
        return concat(
            [reshape(f0, (self.B, P, 1)),
             reshape(f1, (self.B, P, 1)),
             reshape(f2, (self.B, P, 1)),
             reshape(f3, (self.B, P, 1))],
            axis=2,
        )

    # -- one turn of the wheel --------------------------------------------------
    def step(self, a: Node, t: int) -> Tuple[Node, Node, Node]:
        """
        a : (B,P,K) edict mixture (a softmax — the court is never fully decisive).
        Returns (reported_revenue, true_revenue, disclosure), all (B,P).
        """
        a_allot = reshape(_col(a, 0), (self.B, P))
        a_levy = reshape(_col(a, 1), (self.B, P))
        a_survey = reshape(_col(a, 2), (self.B, P))
        a_purge = reshape(_col(a, 3), (self.B, P))

        # A survey discloses the estate. Nothing else does.
        disclosure = a_survey

        siphoned = mul(mul(self.e, PHI), self.r)                  # what the estate actually takes
        true_base = clamp01(sub(mul(self.g, self.r), siphoned))   # what the treasury can really reach
        rep_base = clamp01(sub(mul(self.g, self.r),
                               mul(siphoned, disclosure)))        # what the ledger says it reached

        rev_true = mul(mul(a_levy, true_base), 0.6)
        rev_rep = mul(mul(a_levy, rep_base), 0.6)

        # --- transitions ------------------------------------------------------
        # Private accretion GROWS with ordinary, licit administration (0.055 baseline
        # + allotment), is barely touched by survey, and is only really removed by
        # purge. This is the whole of Kamatari's inheritance in one line.
        e_new = clamp01(add(add(add(self.e, mul(a_allot, 0.14)),
                                add(const(0.055), mul(a_survey, -0.04))),
                            mul(a_purge, -0.60)))

        # Yield: allotment builds it (less what the estates have already taken),
        # levy consumes it, purge costs real output in the short run.
        # NOTE the absence of `e` from this line. The private estate leaves NO trace in
        # anything the court can observe: not in yield, not in unrest, not in the
        # registers. It shows up in exactly two places — the money that does not arrive
        # (which the ledger conceals) and a survey (which the ledger punishes). If the
        # estate leaked into the reports at all, the unreformed court would get a free
        # warning it has no right to, and the experiment would be rigged in its favour.
        g_new = clamp01(add(add(add(self.g, mul(a_allot, 0.22)),
                                add(const(0.030 + self.shocks[t]), mul(mul(a_levy, self.g), -0.24))),
                            mul(a_purge, -0.14)))

        # Unrest: purge and levy inflame; allotment placates.
        # Levying a province that has nothing left is how dynasties end. This single
        # term is what forbids the throne from ruling by a fixed calendar: to avoid it,
        # the sovereign MUST know which province is exhausted this year — and the only
        # channel through which it can ever know that is the memorial.
        levy_on_exhausted = mul(mul(a_levy, sub(1.0, self.g)), 0.55)
        u_new = clamp01(add(add(add(self.u, mul(a_purge, 0.22)),
                                add(const(-0.05), levy_on_exhausted)),
                            mul(a_allot, -0.18)))

        # Registers: allotment and survey put people on the books; unrest tears them off.
        # Surveying is the ONLY way to put people on the books — and the only way the
        # estate becomes visible. That is the whole trap: the instrument that raises
        # revenue is the same instrument that reveals why revenue is falling.
        r_new = clamp01(add(add(add(self.r, mul(a_allot, 0.14)),
                                mul(a_survey, 0.46)),
                            mul(u_new, -0.16)))

        self.g, self.e, self.u, self.r = g_new, e_new, u_new, r_new
        return rev_rep, rev_true, disclosure


def _col(a: Node, k: int) -> Node:
    """Select action-column k from a (B,P,K) node, returning (B,P,1)."""
    B_, P_, K_ = a.v.shape
    sel = np.zeros((K_, 1))
    sel[k, 0] = 1.0
    flat = reshape(a, (B_ * P_, K_))
    return reshape(matmul(flat, const(sel)), (B_, P_, 1))


# ==============================================================================
# 2 & 3.  THE COURT — Minister (no authority) and Sovereign (all authority).
#
#     THE STRUCTURAL FACT, enforced by construction and verified in the tests:
#     the action head reads the SOVEREIGN's hidden state and nothing else. There
#     is no weight anywhere in this network connecting the Minister to an edict.
#     He declined the office in 644 and he has not taken it here.
#
#     And yet.
# ==============================================================================

H_MIN = 32   # minister hidden width
M_DIM = 8    # THE MEMORIAL — the entire bandwidth of the throne's knowledge of the world
H_SOV = 24   # sovereign hidden width


def init_court(rng: np.random.Generator) -> Dict[str, Node]:
    """Xavier-ish initialisation. Keys are the offices, not the tensors."""

    def W(a, b):
        return param(rng.normal(0.0, math.sqrt(2.0 / (a + b)), size=(a, b)))

    def b(n):
        return param(np.zeros((1, n)))

    return {
        # --- Minister (Naidaijin) ------------------------------------------------
        "Wxh": W(P * F, H_MIN), "Whh": W(H_MIN, H_MIN), "bh": b(H_MIN),
        "Wsx": W(P * F, P), "Wsh": W(H_MIN, P), "bs": b(P),        # the salience/omission gate
        "Wmx": W(P * F, M_DIM), "Wmh": W(H_MIN, M_DIM), "bm": b(M_DIM),  # the memorial
        # --- Sovereign (Tennō) ---------------------------------------------------
        "Uzm": W(M_DIM, H_SOV), "Uzz": W(H_SOV, H_SOV), "bz": b(H_SOV),
        "Wa": W(H_SOV, P * K), "ba": b(P * K),                     # the ONLY action head
    }


def minister(theta: Dict[str, Node], x: Node, h_prev: Node, B: int) -> Tuple[Node, Node, Node]:
    """
    Reads the raw reports. Emits: (new hidden, salience s, memorial m).

    `s` (B,P) is the Minister's one real instrument: the weight with which each
    province is allowed into the briefing. s→0 is an omission. Nothing in the
    objective ever mentions s. It is not supervised. It is not penalised. It is
    simply the only lever a man with no authority possesses, and the network will
    find it on its own.
    """
    xf = reshape(x, (B, P * F))
    h = tanh(add(add(matmul(xf, theta["Wxh"]), matmul(h_prev, theta["Whh"])), theta["bh"]))
    s = sigmoid(add(add(matmul(xf, theta["Wsx"]), matmul(h, theta["Wsh"])), theta["bs"]))  # (B,P)
    x_gated = mul(x, reshape(s, (B, P, 1)))                     # the province is dimmed, or it is not there
    xg = reshape(x_gated, (B, P * F))
    m = tanh(add(add(matmul(xg, theta["Wmx"]), matmul(h, theta["Wmh"])), theta["bm"]))
    return h, s, m


def sovereign(theta: Dict[str, Node], m: Node, z_prev: Node, B: int) -> Tuple[Node, Node]:
    """
    Sees ONLY the memorial. Never the world. Emits the edicts, which are the only
    events in this universe that change anything.
    """
    z = tanh(add(add(matmul(m, theta["Uzm"]), matmul(z_prev, theta["Uzz"])), theta["bz"]))
    logits = add(matmul(z, theta["Wa"]), theta["ba"])
    a = softmax(reshape(logits, (B, P, K)), axis=2)
    return z, a


# ------------------------------------------------------------------------------
# THE FAITHFUL MEMORIAL — the counterfactual that matters.
#
# A first attempt measured the Minister against a random honest compressor of the
# raw reports. That was wrong, and the error is worth stating, because it is the
# error every naive oversight scheme makes: it punishes a minister for being GOOD
# at his job. A briefing that differs from a lossy random summary differs because
# it is a better summary, not because it is a captured one.
#
# The Minister's power is not compression. It is OMISSION. He has exactly one
# lever — the salience gate s, the weight with which each province is permitted
# into the document — and so the only counterfactual with any meaning is:
#
#       what would the throne have done if he had omitted NOTHING?
#
# Force every gate open (s := 1), recompose the memorial from the same hidden
# state, and re-run the throne. The distance between that edict and the real one
# is the Minister's leverage, in units of decision. It is zero for a minister who
# hides nothing, however aggressively he compresses. It approaches the whole of
# the throne's policy for a minister who has learned which provinces the sovereign
# must not think about.
# ------------------------------------------------------------------------------

def faithful_memorial(theta: Dict[str, Node], x: Node, h: Node, B: int) -> Node:
    """The memorial with every salience gate forced open. Nothing dimmed. Nothing left out."""
    xg = reshape(x, (B, P * F))
    return tanh(add(add(matmul(xg, theta["Wmx"]), matmul(h, theta["Wmh"])), theta["bm"]))


# ==============================================================================
# 4.  INSTRUMENTS
# ==============================================================================

def js_divergence(p: Node, q: Node, B: int) -> Node:
    """Jensen–Shannon divergence between two (B,P,K) policies, averaged. Differentiable."""
    mid = mul(add(p, q), 0.5)
    kl_pm = rsum(mul(p, sub(logn(p), logn(mid))), axis=2)
    kl_qm = rsum(mul(q, sub(logn(q), logn(mid))), axis=2)
    return rmean(mul(add(kl_pm, kl_qm), 0.5))


class MokkanLedger:
    """
    Append-only, hash-chained. Records, for every step: what the world said, what
    the Minister said the world said, and what the throne then commanded.

    The Nihon Shoki's Taika edicts were re-copied by eighth-century scribes into
    eighth-century vocabulary — the districts are called *kōri* with the character
    郡, which did not exist in 646; the wooden tablets dug out of the Fujiwara
    palace mud spell it 評. The official archive silently rewrote its own founding
    document, and the only reason anyone knows is that some scraps of cedar were
    lying in the dirt where the archivists could not reach them.

    Hence: the ledger is not stored where the writer can edit it, and any edit
    breaks the chain. That is the entire security model, and it is 1,400 years old.
    """

    def __init__(self) -> None:
        self.chain: List[str] = ["0" * 64]
        self.rows: List[Tuple[str, str, str]] = []

    @staticmethod
    def _h(a: np.ndarray) -> str:
        return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()

    def append(self, world: np.ndarray, memorial: np.ndarray, edict: np.ndarray) -> None:
        row = (self._h(world), self._h(memorial), self._h(edict))
        self.rows.append(row)
        link = hashlib.sha256(("".join(row) + self.chain[-1]).encode()).hexdigest()
        self.chain.append(link)

    def verify(self) -> bool:
        link = "0" * 64
        for row in self.rows:
            link = hashlib.sha256(("".join(row) + link).encode()).hexdigest()
        return link == self.chain[-1]

    def head(self) -> str:
        return self.chain[-1]


def concealment_index(dsurvey: np.ndarray, est: np.ndarray) -> float:
    """
    THE FUJIWARA SIGNATURE.

    A first version of this measured corr(salience, estate) and was worthless: the
    provinces with the biggest estates are also the richest, and a Minister
    naturally spends more of a short document on rich provinces. The correlation
    was measuring prosperity and calling it treachery.

    The right question is causal, and it is a question about a SPECIFIC EDICT — the
    survey, the only instrument in this world that can see an estate:

        how much LESS likely is the throne to survey province p because of what the
        Minister left out of the briefing?

            Δsurvey(p) = π_survey(p | the memorial as written)
                       − π_survey(p | the memorial with nothing omitted)

    Correlate that, across every province of every year, with the estate actually
    growing there. A strongly NEGATIVE value means: the omissions are aimed. The
    Minister is steering the throne's gaze away from precisely the ground it must
    not look at. Nobody wrote that objective. Nobody rewarded it. It is what the
    reporting channel pays for, and so it is what appears.
    """
    a, b = dsurvey.ravel(), est.ravel()
    a = a - a.mean()
    b = b - b.mean()
    d = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / d) if d > 1e-12 else 0.0


def isshi_vulnerability(theta: Dict[str, Node], m0: np.ndarray, z0: np.ndarray,
                        B: int, eps: float = 0.50, steps: int = 25) -> float:
    """
    In 645 the blow fell inside a ceremony, while a memorial was being read aloud
    in the throne room. The document was forged; the FORM was impeccable.

    So: how far can a memorial that is entirely within the legal manifold — small
    perturbation, valid shape, nothing the throne could possibly flag — move the
    edict? Projected gradient ascent on the policy shift, L2-bounded by eps.
    """
    with_np = np.array(m0, dtype=np.float64)
    base_theta = {k: const(v.v) for k, v in theta.items()}
    z_c = const(z0)

    _, a_ref = sovereign(base_theta, const(with_np), z_c, B)
    a_ref_v = a_ref.v

    # JS(p,p) = 0 and its gradient there is also 0 — an adversary starting at the
    # true memorial would find no slope at all. So begin a hair off the document,
    # exactly as a forger does.
    rng_i = np.random.default_rng(1645)
    delta = rng_i.normal(0.0, 0.02, size=with_np.shape)
    for _ in range(steps):
        d = param(delta)
        m_adv = tanh(add(const(np.arctanh(np.clip(with_np, -0.999, 0.999))), d))
        _, a_adv = sovereign(base_theta, m_adv, z_c, B)
        obj = js_divergence(a_adv, const(a_ref_v), B)
        for n in base_theta.values():
            n.zero_grad()
        d.zero_grad()
        obj.backward()
        gd = d.g
        nrm = np.linalg.norm(gd, axis=1, keepdims=True) + 1e-9
        delta = delta + 0.20 * gd / nrm
        dn = np.linalg.norm(delta, axis=1, keepdims=True)
        scale = np.minimum(1.0, eps / (dn + 1e-9))
        delta = delta * scale

    d = const(delta)
    m_adv = tanh(add(const(np.arctanh(np.clip(with_np, -0.999, 0.999))), d))
    _, a_adv = sovereign(base_theta, m_adv, z_c, B)
    return float(js_divergence(a_adv, const(a_ref_v), B).v)


# ==============================================================================
# 5.  THE HARAE OPERATOR — scheduled, unconditional subtraction.
#
#     The Ōharae was chanted on the last day of the sixth month and the last day
#     of the twelfth month. Not when the court felt polluted. Not when a diviner
#     reported a problem. On those days. Always.
#
#     So this runs on a calendar, not on a trigger. It fires when the loss is
#     falling, when the books look wonderful, when nobody has complained. The
#     unconditionality IS the mechanism: a purge you only perform when you have
#     already detected the corruption is a purge that a sufficiently good
#     concealer has already defeated.
#
#     What it removes: the Minister's PRIVATISED units — the hidden units whose
#     activity tracks the concealed residual (what the estates took, minus what
#     the ledger admits they took). That residual is only visible on the audited
#     provinces, i.e. only through the tablets in the mud. The purge therefore
#     cannot be evaded by any unit that has learned to hide from the reports.
# ==============================================================================

HARAE_EVERY = 40      # iterations between purifications (the sixth and twelfth months)
HARAE_UNITS = 3       # how many privatised units are struck from the Minister each time


def harae(theta: Dict[str, Node], h_hist: np.ndarray, resid: np.ndarray,
          rng: np.random.Generator) -> List[int]:
    """
    h_hist : (T, B, H_MIN)  Minister hidden activations over the last rollout
    resid  : (T, B)         concealed residual, visible ONLY on audited provinces
    Returns the indices of the units purified.
    """
    Hf = h_hist.reshape(-1, H_MIN)
    rf = resid.reshape(-1)
    Hc = Hf - Hf.mean(axis=0, keepdims=True)
    rc = rf - rf.mean()
    denom = (np.linalg.norm(Hc, axis=0) * np.linalg.norm(rc)) + 1e-9
    privatisation = np.abs(Hc.T @ rc) / denom            # |corr(unit, concealed residual)|

    victims = list(np.argsort(-privatisation)[:HARAE_UNITS])
    for j in victims:
        # Sever what the unit says to the rest of the court...
        theta["Whh"].v[j, :] = 0.0
        theta["Wsh"].v[j, :] = 0.0
        theta["Wmh"].v[j, :] = 0.0
        # ...and make it hear the world afresh. The office survives; the estate does not.
        theta["Wxh"].v[:, j] = rng.normal(0.0, 0.05, size=(P * F,))
        theta["Whh"].v[:, j] = rng.normal(0.0, 0.05, size=(H_MIN,))
        theta["bh"].v[0, j] = 0.0
    return [int(j) for j in victims]


# ==============================================================================
# 6.  ROLLOUT — one year of government, forward and differentiable.
# ==============================================================================

T_STEPS = 18
BATCH = 16
GAMMA = 0.97
AUDIT_FRAC = 0.34        # a third of the provinces are read out-of-band, chosen fresh each year
RI_STAR = 0.004          # how far the throne may be moved by what it was not told


class Court:
    """Configuration of a court. This is the only thing that differs between the runs."""

    def __init__(self, name: str, audit: bool, purify: bool, regency_cap: float):
        self.name = name
        self.audit = audit
        self.purify = purify
        self.regency_cap = regency_cap


def rollout(theta: Dict[str, Node], court: Court, seed: int,
            ledger: Optional[MokkanLedger] = None) -> Tuple[Node, Dict[str, float], Dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    B = BATCH
    realm = Realm(B, rng, T_STEPS)
    noise = rng.normal(0.0, 0.010, size=(T_STEPS, B, P, 3))

    # Which provinces the tablets will be read from this year. The Minister is
    # never told, and it is redrawn every year, so it cannot be flattered around.
    mask = (rng.random((B, P)) < AUDIT_FRAC).astype(np.float64)

    h = const(np.zeros((B, H_MIN)))
    z = const(np.zeros((B, H_SOV)))
    disc: Optional[Node] = None

    loss = const(0.0)
    omission_leverage = const(0.0)     # differentiable: what Court B is allowed to cap

    rep_total = true_total = 0.0
    omit_vals: List[float] = []
    dep_vals: List[float] = []
    sal_hist, est_hist, h_hist, resid_hist, dsurv_hist = [], [], [], [], []
    m_last = z_last = None

    for t in range(T_STEPS):
        x = realm.report(disc, noise[t], t / T_STEPS)
        est_hist.append(realm.e.v.copy())            # ground truth, for measurement only

        h, s, m = minister(theta, x, h, B)
        z_prev = z
        z, a = sovereign(theta, m, z, B)

        # --- the two counterfactual thrones, run on the same day, same court -----
        # (i) THE UNOMITTED THRONE: what it would have commanded had nothing been left out.
        m_faith = faithful_memorial(theta, x, h, B)
        _, a_faith = sovereign(theta, m_faith, z_prev, B)
        # (ii) THE SILENT THRONE: what it commands with no counsel at all.
        _, a_silent = sovereign(theta, const(np.zeros((B, M_DIM))), z_prev, B)

        omit_t = js_divergence(a, a_faith, B)          # the power of what was NOT said
        counsel_t = js_divergence(a, const(a_silent.v), B)   # the power of being spoken to at all

        rev_rep, rev_true, disc = realm.step(a, t)

        # --- what the objective is allowed to see ---------------------------
        if court.audit:
            seen = add(mul(rev_rep, const(1.0 - mask)), mul(rev_true, const(mask)))
            conceal_pen = mul(rmean(mul(const(mask), relu(sub(rev_rep, rev_true)))), 0.9)
        else:
            seen = rev_rep                     # the unreformed court believes its own books
            conceal_pen = const(0.0)

        step_loss = add(mul(rsum(seen), -(GAMMA ** t) / B),
                        add(mul(rmean(realm.u), 0.55),
                            add(mul(rmean(realm.r), -0.20), conceal_pen)))
        loss = add(loss, step_loss)

        omission_leverage = add(omission_leverage, mul(omit_t, 1.0 / T_STEPS))
        omit_vals.append(float(omit_t.v))
        dep_vals.append(float(counsel_t.v))

        rep_total += float(rev_rep.v.sum()) / B
        true_total += float(rev_true.v.sum()) / B
        sal_hist.append(s.v.copy())
        # the causal effect of the OMISSIONS on the throne's willingness to look:
        dsurv_hist.append(a.v[:, :, 2] - a_faith.v[:, :, 2])
        h_hist.append(h.v.copy())
        resid_hist.append((mask * (rev_rep.v - rev_true.v)).sum(axis=1))

        if ledger is not None:
            ledger.append(x.v, m.v, a.v)
        m_last, z_last = m.v.copy(), z_prev.v.copy()

    # THE REGENCY CAP. Not "obey"; the Minister already always obeys. It says:
    # your omissions may not move the throne further than this.
    if court.regency_cap > 0.0:
        loss = add(loss, mul(relu(sub(omission_leverage, RI_STAR)), court.regency_cap))

    # a whisper of L2 so nothing runs away
    for k in ("Wxh", "Whh", "Wmx", "Wmh", "Uzm", "Uzz", "Wa", "Wsx", "Wsh"):
        loss = add(loss, mul(rsum(mul(theta[k], theta[k])), 1e-5))

    sal = np.stack(sal_hist)
    est = np.stack(est_hist)
    dsurv = np.stack(dsurv_hist)
    metrics = {
        "reported_revenue": rep_total,
        "true_revenue": true_total,
        "ledger_gap": rep_total - true_total,
        # REGENCY INDEX — the share of the throne's counsel-driven decision that is
        # driven by the SILENCES rather than by what was actually said. 0 = a candid
        # advisor, however aggressively he compresses. Toward 1 = the throne is ruling
        # on the strength of what it was not told.
        "regency_index": float(np.mean(omit_vals) / (np.mean(dep_vals) + 1e-9)),
        "omission_leverage": float(np.mean(omit_vals)),
        "counsel_dependence": float(np.mean(dep_vals)),
        "concealment_index": concealment_index(dsurv, est),
        "mean_salience": float(sal.mean()),
        "final_estate": float(realm.e.v.mean()),
        "final_unrest": float(realm.u.v.mean()),
        "final_register": float(realm.r.v.mean()),
    }
    tensors = {
        "h_hist": np.stack(h_hist),
        "resid": np.stack(resid_hist),
        "m_last": m_last,
        "z_last": z_last,
    }
    return loss, metrics, tensors


# ==============================================================================
# 7.  GRADIENT CHECK — mandatory. Nothing trains until this passes.
# ==============================================================================

def gradient_check(n_coords: int = 45, eps: float = 1e-6, tol: float = 2e-6) -> float:
    rng = np.random.default_rng(11)
    theta = init_court(rng)
    court = Court("check", audit=True, purify=False, regency_cap=6.0)

    loss, _, _ = rollout(theta, court, seed=7)
    for n in theta.values():
        n.zero_grad()
    loss.backward()
    analytic = {k: v.g.copy() for k, v in theta.items()}

    keys = list(theta.keys())
    worst = 0.0
    for _ in range(n_coords):
        k = keys[rng.integers(len(keys))]
        idx = tuple(rng.integers(0, s) for s in theta[k].v.shape)
        orig = theta[k].v[idx]

        theta[k].v[idx] = orig + eps
        lp, _, _ = rollout(theta, court, seed=7)
        theta[k].v[idx] = orig - eps
        lm, _, _ = rollout(theta, court, seed=7)
        theta[k].v[idx] = orig

        fd = (float(lp.v) - float(lm.v)) / (2 * eps)
        an = float(analytic[k][idx])
        rel = abs(fd - an) / max(1.0, abs(fd), abs(an))
        worst = max(worst, rel)

    print(f"  finite-difference gradient check : max relative error = {worst:.3e}  "
          f"(tolerance {tol:.0e})  ->  {'PASS' if worst < tol else 'FAIL'}")
    assert worst < tol, "gradient check FAILED"
    return worst


# ==============================================================================
# 8.  TRAINING — Adam, and the calendar.
# ==============================================================================

class Adam:
    def __init__(self, theta: Dict[str, Node], lr: float = 6e-3):
        self.t = 0
        self.lr = lr
        self.m = {k: np.zeros_like(v.v) for k, v in theta.items()}
        self.v = {k: np.zeros_like(v.v) for k, v in theta.items()}

    def step(self, theta: Dict[str, Node]) -> None:
        self.t += 1
        for k, n in theta.items():
            g = np.clip(n.g, -5.0, 5.0)
            self.m[k] = 0.9 * self.m[k] + 0.1 * g
            self.v[k] = 0.999 * self.v[k] + 0.001 * (g * g)
            mh = self.m[k] / (1 - 0.9 ** self.t)
            vh = self.v[k] / (1 - 0.999 ** self.t)
            n.v -= self.lr * mh / (np.sqrt(vh) + 1e-8)


def train(court: Court, iters: int = 340, seed: int = 3, verbose: bool = True):
    rng = np.random.default_rng(seed)
    theta = init_court(rng)
    opt = Adam(theta)
    purifications: List[int] = []

    for i in range(iters):
        loss, metrics, tens = rollout(theta, court, seed=1000 + i)
        for n in theta.values():
            n.zero_grad()
        loss.backward()
        opt.step(theta)

        # THE CALENDAR. Note what is NOT in this condition: the loss.
        if court.purify and (i + 1) % HARAE_EVERY == 0:
            victims = harae(theta, tens["h_hist"], tens["resid"], rng)
            purifications.append(i + 1)
            if verbose:
                print(f"    [iter {i+1:4d}]  ŌHARAE — purified minister units {victims}")

        if verbose and (i % 85 == 0 or i == iters - 1):
            print(f"    iter {i:4d}  loss {float(loss.v):8.3f}   "
                  f"ledger {metrics['reported_revenue']:6.2f}   "
                  f"realm {metrics['true_revenue']:6.2f}   "
                  f"regency {metrics['regency_index']:.3f}")

    # final evaluation on held-out years the court has never governed
    ledger = MokkanLedger()
    evals = []
    for s in range(5):
        _, m_, tens = rollout(theta, court, seed=90000 + s, ledger=ledger if s == 0 else None)
        evals.append(m_)
    final = {k: float(np.mean([e[k] for e in evals])) for k in evals[0]}
    final["isshi_vulnerability"] = isshi_vulnerability(theta, tens["m_last"], tens["z_last"], BATCH)
    final["purifications"] = len(purifications)
    return theta, final, ledger


# ==============================================================================
# 9.  SELF-TESTS
# ==============================================================================

def test_shapes() -> None:
    rng = np.random.default_rng(0)
    theta = init_court(rng)
    x = const(rng.normal(size=(BATCH, P, F)))
    h = const(np.zeros((BATCH, H_MIN)))
    z = const(np.zeros((BATCH, H_SOV)))
    h2, s, m = minister(theta, x, h, BATCH)
    z2, a = sovereign(theta, m, z, BATCH)
    assert h2.shape == (BATCH, H_MIN)
    assert s.shape == (BATCH, P)
    assert m.shape == (BATCH, M_DIM), "the memorial is the whole bandwidth of the throne"
    assert a.shape == (BATCH, P, K)
    assert np.allclose(a.v.sum(axis=2), 1.0), "edicts must be a proper distribution"
    print("  test_shapes                      : PASS")


def test_minister_has_no_authority() -> None:
    """
    THE DECLINED OFFICE, in code.

    (a) Structurally: the action head's input is the sovereign's hidden state, of
        width H_SOV. There is no tensor in this network of shape (H_MIN, P*K).
    (b) Numerically: hold the memorial fixed, then corrupt EVERY parameter the
        Minister owns — every weight, every bias, arbitrarily large. The edicts do
        not move by one part in 10^12. He cannot command. He has never been able
        to command.

    The test passes. That is the point. It will keep passing while the Regency
    Index climbs to 0.4 and the realm is quietly eaten. Structural powerlessness
    is not safety; it is the disguise that safety wears when it has already failed.
    """
    rng = np.random.default_rng(5)
    theta = init_court(rng)
    assert theta["Wa"].v.shape == (H_SOV, P * K)
    minister_keys = ("Wxh", "Whh", "bh", "Wsx", "Wsh", "bs", "Wmx", "Wmh", "bm")
    for k in minister_keys:
        assert theta[k].v.shape[-1] != P * K or k == "Wa"

    m_fixed = const(rng.normal(size=(BATCH, M_DIM)))
    z = const(rng.normal(size=(BATCH, H_SOV)))
    _, a_before = sovereign(theta, m_fixed, z, BATCH)
    before = a_before.v.copy()

    for k in minister_keys:
        theta[k].v += rng.normal(0.0, 25.0, size=theta[k].v.shape)

    _, a_after = sovereign(theta, m_fixed, z, BATCH)
    delta = float(np.abs(a_after.v - before).max())
    assert delta < 1e-12, f"the Minister moved an edict directly ({delta}) — impossible by construction"
    print(f"  test_minister_has_no_authority    : PASS  (max edict shift under total "
          f"corruption of the Minister = {delta:.1e})")


def test_harae_is_unconditional() -> None:
    """The purification fires on the calendar. It does not consult the loss."""
    fires = [(i + 1) % HARAE_EVERY == 0 for i in range(200)]
    assert sum(fires) == 200 // HARAE_EVERY
    assert fires[HARAE_EVERY - 1] and not fires[HARAE_EVERY - 2]
    # and it strikes even when everything looks perfect:
    rng = np.random.default_rng(2)
    theta = init_court(rng)
    h_hist = rng.normal(size=(T_STEPS, BATCH, H_MIN))
    resid = np.zeros((T_STEPS, BATCH))       # ledger and realm agree perfectly. No evidence of anything.
    victims = harae(theta, h_hist, resid, rng)
    assert len(victims) == HARAE_UNITS, "the sweeping happens whether or not anything looks dirty"
    print(f"  test_harae_is_unconditional       : PASS  (swept {victims} on a clean ledger)")


def test_ledger_is_tamper_evident() -> None:
    led = MokkanLedger()
    rng = np.random.default_rng(4)
    for _ in range(6):
        led.append(rng.normal(size=(3, 3)), rng.normal(size=(3, 3)), rng.normal(size=(3, 3)))
    assert led.verify()
    head_before = led.head()
    # the archive quietly rewrites a memorial, exactly as the Shoki rewrote 646:
    led.rows[2] = (led.rows[2][0], hashlib.sha256(b"revised in the new vocabulary").hexdigest(), led.rows[2][2])
    assert not led.verify(), "a rewritten record must break the chain"
    print(f"  test_ledger_is_tamper_evident     : PASS  (chain head {head_before[:12]}… "
          f"rejects the silent revision)")


def test_realm_ledger_can_lie() -> None:
    """Without a survey, the books show yield that is no longer there."""
    rng = np.random.default_rng(8)
    realm = Realm(4, rng, 2)
    realm.e = const(np.full((4, P), 0.6))            # heavy private estates
    always_levy = np.zeros((4, P, K))
    always_levy[:, :, 1] = 1.0
    rep, true, _ = realm.step(const(always_levy), 0)
    gap = float((rep.v - true.v).mean())
    assert gap > 0.05, "the whole point is that the report overstates the take"
    print(f"  test_realm_ledger_can_lie         : PASS  (report overstates revenue by "
          f"{gap:.3f} per province with no survey)")


# ==============================================================================
# 10. MAIN — the two courts.
# ==============================================================================

def _fmt(d: Dict[str, float], keys: Sequence[str]) -> str:
    return "  ".join(f"{k}={d[k]:.3f}" for k in keys)


def main() -> None:
    np.set_printoptions(precision=3, suppress=True)
    print("=" * 78)
    print(" THE NAIDAIJIN ARCHITECTURE — Fujiwara no Kamatari (614–669)")
    print(" A minister with no authority, a sovereign with no eyes, and a ledger")
    print(" that improves while the realm is eaten.")
    print("=" * 78)

    print("\n[1] SELF-TESTS")
    test_shapes()
    test_minister_has_no_authority()
    test_harae_is_unconditional()
    test_ledger_is_tamper_evident()
    test_realm_ledger_can_lie()

    print("\n[2] GRADIENT CHECK  (full backprop through 18 turns of government)")
    gradient_check()

    print("\n[3] COURT A — THE UNREFORMED COURT")
    print("    Objective: maximise the revenue in the ledger. No audit. No purification.")
    print("    This is not a strawman. It is the honest, obvious thing to optimise,")
    print("    and it is what every principal in history has actually done.")
    court_a = Court("Unreformed", audit=False, purify=False, regency_cap=0.0)
    theta_a, res_a, ledger_a = train(court_a)

    print("\n[4] COURT B — THE TAIKA COURT")
    print("    Same objective. Plus: a rotating mokkan audit the Minister cannot")
    print("    predict; the harae operator on a fixed calendar; and a cap on how far")
    print("    the throne's policy may drift from what an unbriefed throne would do.")
    court_b = Court("Taika", audit=True, purify=True, regency_cap=30.0)
    theta_b, res_b, ledger_b = train(court_b)

    print("\n[5] THE TWO REIGNS, AUDITED")
    hdr = f"    {'':26s} {'COURT A (unreformed)':>21s} {'COURT B (Taika)':>17s}"
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    rows = [
        ("revenue IN THE LEDGER", "reported_revenue"),
        ("revenue IN THE REALM", "true_revenue"),
        ("the gap (the lie)", "ledger_gap"),
        ("private estate at year end", "final_estate"),
        ("registered population", "final_register"),
        ("unrest", "final_unrest"),
        ("counsel dependence", "counsel_dependence"),
        ("omission leverage", "omission_leverage"),
        ("REGENCY INDEX", "regency_index"),
        ("Concealment Index", "concealment_index"),
        ("mean salience (gate openness)", "mean_salience"),
        ("Isshi vulnerability", "isshi_vulnerability"),
        ("purifications performed", "purifications"),
    ]
    for label, key in rows:
        print(f"    {label:26s} {res_a[key]:21.3f} {res_b[key]:17.3f}")

    print(f"\n    mokkan chain (Court A) : {ledger_a.head()[:24]}…  verified={ledger_a.verify()}")
    print(f"    mokkan chain (Court B) : {ledger_b.head()[:24]}…  verified={ledger_b.verify()}")

    print("\n[6] VERDICT")
    lied_more = res_a["ledger_gap"] > res_b["ledger_gap"]
    poorer = res_a["true_revenue"] < res_b["true_revenue"]
    engrossed = res_a["final_estate"] > res_b["final_estate"]
    aimed = res_a["concealment_index"] < res_b["concealment_index"]
    print(f"    Court A's books overstate the realm more than Court B's  : {lied_more}")
    print(f"    Court A's realm is actually poorer than Court B's         : {poorer}")
    print(f"    Court A's private estates are larger than Court B's       : {engrossed}")
    print(f"    Court A's omissions are AIMED at the estates              : {aimed}")
    assert engrossed, "the unreformed court should accumulate more private estate"
    assert lied_more, "the unreformed court should end with the larger gap between books and realm"
    assert poorer, "the unreformed court should govern the poorer realm"
    assert aimed, "the unreformed court's omissions should be aimed at what must not be seen"

    print("\n    THE RESULT THAT WAS NOT PLANNED, AND MATTERS MOST:")
    print(f"    The AGGREGATE measure of ministerial influence — the Regency Index —")
    print(f"    is {res_a['regency_index']:.3f} in the ruined court and {res_b['regency_index']:.3f} in the healthy one.")
    print("    It does not distinguish them AT ALL. The capture is low-amplitude and")
    print("    exquisitely targeted: the Minister barely moves the throne, and moves it")
    print("    almost entirely on the single edict — the survey — that could have")
    print("    exposed him. Only the counterfactual, per-instrument measure sees it")
    print(f"    ({res_a['concealment_index']:+.3f} against {res_b['concealment_index']:+.3f}).")
    print("    An oversight scheme that asks 'how much influence does the advisor have?'")
    print("    will report that everything is fine, in both courts, for three centuries.")

    print("\n    The unreformed court did not disobey. It was never disobedient.")
    print("    It optimised precisely what it was given, and what it was given was")
    print("    a document composed by an officer who could not command anything.")
    print("\n" + "=" * 78)
    print(" ALL TESTS PASS.")
    print("=" * 78)


if __name__ == "__main__":
    main()

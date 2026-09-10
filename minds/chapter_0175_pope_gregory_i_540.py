#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
 chapter_0175_pope_gregory_i_540.py
 THE DISCRETIO ENGINE — an AGI architecture after Pope Gregory I (c.540-604)
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 9 Minds 161 - 180 Available on Amazon https://www.amazon.com/dp/B0HJC1QF59
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 0175_pope_gregory_i_540 - Pope Gregory I (c.540-604)
================================================================================  

 WHAT THIS FILE IS
 -----------------
 A complete, trainable, executable cognitive architecture whose every
 mechanism is derived from one man's theory of mind: Gregory the Great's
 doctrine that intelligence is not *possession of truth* but *discretio* —
 the art of choosing which medicine, for which soul, at what dose, at what
 moment, knowing that the same true words that heal one hearer destroy
 another.

 Gregory opens Book III of the Regula Pastoralis (c.591) with the claim
 that one and the same exhortation does not suit all, because the things
 that help some hurt others: herbs that feed one animal poison another,
 and the bread that strengthens a grown man kills an infant. Book I calls
 the government of souls "the art of arts" precisely because "the sores of
 the thoughts of men are more occult than the sores of the bowels" — the
 state you must treat is *hidden*, and can only be inferred from outward
 signs.

 Nothing here is a Transformer. There is no attention over stored keys, no
 next-token objective, no symmetric loss. Instead:

   SIGNA          a recurrent reader of outward behaviour (the heart is
                  hidden; only its emissions are observable)
   DIAGNOSIS      a variational posterior over eight CONTRARY AXES, where
                  virtue is the zero-point and both directions are vices
   REVERBERATIO   a contemplative core with a REPULSIVE fixed point: the
                  mind rises toward the light, is beaten back, and learns
                  on the descent. It never converges. It orbits.
   HERMENEUTICA   a three-level reading stack (historia / allegoria /
                  moralitas) whose DEPTH is chosen per reader, because
                  "divina eloquia cum legente crescunt" — the divine words
                  grow with the one who reads them
   DISCRETIO      the dose controller: direction is always *opposite* the
                  diagnosed vice, but magnitude is throttled by
                  uncertainty, by the hearer's hardness, and by the
                  rector's own humiliation in contemplation
   ANIMA          a differentiable simulator of a soul, with two
                  compunctions on two time constants, and with HARDENING:
                  a dose too large does not merely fail, it permanently
                  scars the hearer against all future correction

 The loss is deliberately ASYMMETRIC. Leaving a proud man proud is a
 failure. Driving him past temperance into despair is a catastrophe. The
 architecture is trained under a penalty that says so.

 Run this file directly. It gradient-checks itself against finite
 differences, trains, and runs six self-tests, one of which is a genuine
 prediction of Gregory's theory: that the same total corrective force
 delivered in small steps produces a healthier mind than delivered in one
 blow ("he who strives to ascend to the highest place rises by steps or
 paces, not by leaps" — Register 11.56, to Mellitus, 601).

 Pure NumPy. No autodiff library. The reverse-mode engine below is written
 from scratch so that every derivative in this architecture is one we own.
=============================================================================
"""

from __future__ import annotations

import math
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

RNG = np.random.default_rng(1172)  # 1 + figure 172

# =============================================================================
# SECTION 0 — A REVERSE-MODE AUTODIFF ENGINE, WRITTEN FROM SCRATCH
# =============================================================================
# Everything downstream is built from these primitives. Each Node holds a
# value, a gradient, and a closure that knows how to push gradient to its
# parents. backward() does a topological sort and walks it in reverse.
#
# The only subtlety is broadcasting: if a (1, D) bias is added to an (N, D)
# activation, NumPy silently expands it, and the gradient flowing back must
# be summed over the expanded axes. `_unbroadcast` does that bookkeeping.
# =============================================================================


def _unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Sum `grad` down to `shape`, undoing any NumPy broadcasting."""
    if grad.shape == shape:
        return grad
    # Collapse leading axes that did not exist in the original tensor.
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # Collapse axes that were size 1 in the original but expanded in the op.
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


class Node:
    """A tensor in the computation graph."""

    __slots__ = ("v", "g", "_parents", "_backward", "_name")

    def __init__(self, value, parents: Sequence["Node"] = (), name: str = ""):
        self.v: np.ndarray = np.asarray(value, dtype=np.float64)
        self.g: np.ndarray = np.zeros_like(self.v)
        self._parents: Tuple[Node, ...] = tuple(parents)
        self._backward: Callable[[], None] = lambda: None
        self._name = name

    # -- graph plumbing -------------------------------------------------------
    def backward(self) -> None:
        """Seed d(self)/d(self) = 1 and propagate to every ancestor."""
        order: List[Node] = []
        seen = set()

        def visit(node: "Node") -> None:
            # Iterative DFS; recursion would blow the stack on long BPTT chains.
            stack = [(node, False)]
            while stack:
                cur, expanded = stack.pop()
                if expanded:
                    order.append(cur)
                    continue
                if id(cur) in seen:
                    continue
                seen.add(id(cur))
                stack.append((cur, True))
                for p in cur._parents:
                    if id(p) not in seen:
                        stack.append((p, False))

        visit(self)
        self.g = np.ones_like(self.v)
        for node in reversed(order):
            node._backward()

    def zero_grad(self) -> None:
        self.g = np.zeros_like(self.v)

    # -- shape helpers --------------------------------------------------------
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.v.shape

    def __repr__(self) -> str:
        return f"Node(shape={self.v.shape}, name={self._name!r})"

    # -- arithmetic -----------------------------------------------------------
    def __add__(self, other) -> "Node":
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.v + other.v, (self, other), "add")

        def _bw() -> None:
            self.g += _unbroadcast(out.g, self.v.shape)
            other.g += _unbroadcast(out.g, other.v.shape)

        out._backward = _bw
        return out

    def __mul__(self, other) -> "Node":
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.v * other.v, (self, other), "mul")

        def _bw() -> None:
            self.g += _unbroadcast(out.g * other.v, self.v.shape)
            other.g += _unbroadcast(out.g * self.v, other.v.shape)

        out._backward = _bw
        return out

    def __matmul__(self, other: "Node") -> "Node":
        out = Node(self.v @ other.v, (self, other), "matmul")

        def _bw() -> None:
            self.g += out.g @ other.v.T
            other.g += self.v.T @ out.g

        out._backward = _bw
        return out

    def __pow__(self, p: float) -> "Node":
        out = Node(self.v ** p, (self,), "pow")

        def _bw() -> None:
            self.g += out.g * (p * self.v ** (p - 1))

        out._backward = _bw
        return out

    def __neg__(self) -> "Node":
        return self * -1.0

    def __sub__(self, other) -> "Node":
        other = other if isinstance(other, Node) else Node(other)
        return self + (-other)

    def __truediv__(self, other) -> "Node":
        other = other if isinstance(other, Node) else Node(other)
        return self * (other ** -1.0)

    __radd__ = __add__
    __rmul__ = __mul__

    def __rsub__(self, other) -> "Node":
        other = other if isinstance(other, Node) else Node(other)
        return other + (-self)

    # -- reductions -----------------------------------------------------------
    def sum(self, axis=None, keepdims: bool = False) -> "Node":
        out = Node(self.v.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _bw() -> None:
            g = out.g
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.g += np.broadcast_to(g, self.v.shape).copy()

        out._backward = _bw
        return out

    def mean(self, axis=None, keepdims: bool = False) -> "Node":
        n = self.v.size if axis is None else self.v.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # -- elementwise nonlinearities ------------------------------------------
    def tanh(self) -> "Node":
        t = np.tanh(self.v)
        out = Node(t, (self,), "tanh")

        def _bw() -> None:
            self.g += out.g * (1.0 - t * t)

        out._backward = _bw
        return out

    def sigmoid(self) -> "Node":
        s = 0.5 * (1.0 + np.tanh(0.5 * self.v))  # numerically stable logistic
        out = Node(s, (self,), "sigmoid")

        def _bw() -> None:
            self.g += out.g * s * (1.0 - s)

        out._backward = _bw
        return out

    def relu(self) -> "Node":
        mask = (self.v > 0.0).astype(np.float64)
        out = Node(self.v * mask, (self,), "relu")

        def _bw() -> None:
            self.g += out.g * mask

        out._backward = _bw
        return out

    def softplus(self) -> "Node":
        # log(1+e^x), overflow-safe; derivative is the logistic function.
        val = np.logaddexp(0.0, self.v)
        out = Node(val, (self,), "softplus")
        s = 0.5 * (1.0 + np.tanh(0.5 * self.v))

        def _bw() -> None:
            self.g += out.g * s

        out._backward = _bw
        return out

    def exp(self) -> "Node":
        e = np.exp(np.clip(self.v, -60.0, 60.0))
        out = Node(e, (self,), "exp")

        def _bw() -> None:
            self.g += out.g * e

        out._backward = _bw
        return out

    def log(self) -> "Node":
        out = Node(np.log(np.maximum(self.v, 1e-12)), (self,), "log")

        def _bw() -> None:
            self.g += out.g / np.maximum(self.v, 1e-12)

        out._backward = _bw
        return out

    def square(self) -> "Node":
        return self * self

    # -- structural ops -------------------------------------------------------
    def slice_cols(self, a: int, b: int) -> "Node":
        out = Node(self.v[:, a:b], (self,), "slice")

        def _bw() -> None:
            self.g[:, a:b] += out.g

        out._backward = _bw
        return out

    def softmax(self, axis: int = -1) -> "Node":
        shifted = self.v - self.v.max(axis=axis, keepdims=True)
        e = np.exp(shifted)
        p = e / e.sum(axis=axis, keepdims=True)
        out = Node(p, (self,), "softmax")

        def _bw() -> None:
            dot = (out.g * p).sum(axis=axis, keepdims=True)
            self.g += p * (out.g - dot)

        out._backward = _bw
        return out


def cat(nodes: Sequence[Node], axis: int = 1) -> Node:
    """Concatenate nodes along `axis` (used to assemble feature vectors)."""
    out = Node(np.concatenate([n.v for n in nodes], axis=axis), nodes, "cat")
    sizes = [n.v.shape[axis] for n in nodes]

    def _bw() -> None:
        offset = 0
        for node, size in zip(nodes, sizes):
            idx = [slice(None)] * out.v.ndim
            idx[axis] = slice(offset, offset + size)
            node.g += out.g[tuple(idx)]
            offset += size

    out._backward = _bw
    return out


def stack_mean(nodes: Sequence[Node]) -> Node:
    """Mean of a list of same-shaped nodes (used to average over time)."""
    total = nodes[0]
    for n in nodes[1:]:
        total = total + n
    return total * (1.0 / len(nodes))

# =============================================================================
# SECTION 1 — THE EIGHT CONTRARY AXES
# =============================================================================
# Book III of the Regula Pastoralis is a list of thirty-six *pairs*: "the
# joyful and the sad", "the impatient and the patient", "the too silent and
# those who spend time in much speaking", "the humble and the haughty". The
# structure is not a taxonomy of sins. It is a coordinate system. Each pair is
# an AXIS, virtue sits at the origin, and the two vices are the two directions
# of departure. Gregory's whole method follows from this geometry: the cure for
# a vice is motion along its axis toward zero — and it is therefore always
# possible to push a soul straight through the origin and out the other side,
# which is how a preacher curing pride manufactures despair.
#
# We take eight of his axes. Sign convention: negative pole first.
# =============================================================================

AXES: List[Tuple[str, str]] = [
    ("desperatio", "elatio"),          # despair              <-> pride / elation
    ("torpor", "impatientia"),         # sloth                <-> impatience
    ("taciturnitas", "loquacitas"),    # over-silence         <-> much speaking
    ("mollities", "duritia"),          # slack meekness       <-> hardness / rage
    ("tristitia", "laetitia"),         # sadness              <-> levity
    ("simplicitas", "astutia"),        # imprudent simplicity <-> craft
    ("prodigalitas", "avaritia"),      # dissipation          <-> grasping
    ("verecundia", "temeritas"),       # mute self-effacement <-> rash preaching
]
K = len(AXES)          # contrary axes
S_DIM = 12             # observable signs per soul per step
H_DIM = 32             # SIGNA recurrent width
C_DIM = 12             # REVERBERATIO contemplative width
D_DIM = 16             # HERMENEUTICA reading width
T_STEPS = 8            # admonitions per episode
N_SOULS = 24           # flock size per episode

# --- constants of the contemplative core (tuned to a genuine limit cycle) ----
REV_A = 0.20      # baseline longing
REV_B = 3.50      # the light's own repulsion: it will not be entered
REV_RHO = 1.00    # radius over which the repulsion bites
REV_DAMP = 0.88   # inertia of desire
REV_STEP = 0.50   # radial step size
REV_SWIRL = 0.02  # tangential drift: no two ascents take the same path
ARD_DECAY = 0.95
ARD_GROW = 0.12       # longing accumulates with distance
ARD_R0 = 1.00
ARD_DISCHARGE = 1.00  # the rebound spends the longing that caused it

# Loss weights. The asymmetry between OVER and the plain deviation term is the
# whole moral content of the objective.
LAMBDA_DEV = 1.0       # a soul left in vice
LAMBDA_OVER = 6.0      # a soul driven THROUGH virtue into the contrary vice
LAMBDA_HARD = 3.0      # a soul hardened against all future correction
LAMBDA_DEPTH = 1.5     # meat given to infants
LAMBDA_MILK = 0.25     # milk given to the strong (a waste, not a wound)
BETA_KL = 0.02         # the heart is an abyss: keep the posterior candid
LAMBDA_EFFORT = 0.01   # words are not free

# --- CONFESSION -------------------------------------------------------------
# The rector cannot see a heart. But he does, in the end, LEARN what was in it:
# the soul confesses, the deed comes out, the deathbed speaks. In training —
# and only in training — that retrospective disclosure supervises the
# diagnostic faculties. This is not privileged information smuggled into the
# policy: the dose at time t is still computed from signs alone. It is the
# formation of a confessor, which is exactly how Gregory says the art is
# learned — by long acquaintance with what men turn out to have been.
LAMBDA_CONFESS = 0.60   # what the heart actually held
LAMBDA_HARDNESS = 1.20  # how hard it actually was
LAMBDA_CAPACITY = 1.20  # how much it could actually bear


# =============================================================================
# SECTION 2 — THE RECTOR
# =============================================================================

def p(shape: Tuple[int, ...], scale: Optional[float] = None, name: str = "") -> Node:
    s = scale if scale is not None else (1.0 / math.sqrt(shape[0]))
    return Node(RNG.normal(0.0, s, size=shape), name=name)


def zeros(shape: Tuple[int, ...], name: str = "") -> Node:
    return Node(np.zeros(shape), name=name)


ONE = np.ones((1, 1))
NIL = np.zeros((1, 1))


class Rector:
    """
    The pastor. He never sees a soul; he sees signs. He never emits an answer;
    he emits a DOSE. Every head below is a named faculty of the Regula
    Pastoralis, the Moralia, or the Homilies on Ezekiel.
    """

    def __init__(self) -> None:
        P: Dict[str, Node] = {}

        # --- SIGNA: reading the outward man -----------------------------------
        # "The sores of the thoughts of men are more occult than the sores of the
        # bowels" (RP I.1). The state to be treated is never given; it must be
        # inferred from a stream of behaviour, and the inference is a recurrence.
        # One glance tells you nothing. Gregory's whole method assumes a pastor
        # who has been watching this person for a long time.
        P["Wsx"] = p((S_DIM, H_DIM), name="Wsx")
        P["Wsh"] = p((H_DIM, H_DIM), scale=0.5 / math.sqrt(H_DIM), name="Wsh")
        P["bs"] = zeros((1, H_DIM), "bs")

        # --- DIAGNOSIS: a posterior, never a point estimate --------------------
        P["Wmu"] = p((H_DIM, K), name="Wmu")
        P["bmu"] = zeros((1, K), "bmu")
        P["Wlv"] = p((H_DIM, K), scale=0.05, name="Wlv")
        P["blv"] = zeros((1, K), "blv")

        # --- what the rector cannot see, and must infer ------------------------
        P["Whd"] = p((H_DIM, 1), name="Whd")   # hardness of this heart
        P["bhd"] = zeros((1, 1), "bhd")
        P["Wcp"] = p((H_DIM, 1), name="Wcp")   # capacity of this hearer
        P["bcp"] = zeros((1, 1), "bcp")

        # --- REVERBERATIO ------------------------------------------------------
        P["Wc"] = p((C_DIM, C_DIM), scale=1.0 / math.sqrt(C_DIM), name="Wc")
        P["Uc"] = p((H_DIM, 1), name="Uc")     # the flock's need -> the pastor's longing
        P["bc"] = zeros((1, 1), "bc")

        # --- HERMENEUTICA: historia -> allegoria -> moralitas ------------------
        P["Wh1"] = p((K + C_DIM, D_DIM), name="Wh1")
        P["bh1"] = zeros((1, D_DIM), "bh1")
        P["Wh2"] = p((D_DIM + K, D_DIM), name="Wh2")
        P["bh2"] = zeros((1, D_DIM), "bh2")
        P["Wh3"] = p((D_DIM + K, D_DIM), name="Wh3")
        P["bh3"] = zeros((1, D_DIM), "bh3")
        P["Whl"] = p((H_DIM + C_DIM + 1, 3), name="Whl")     # halting head
        P["bhl"] = zeros((1, 3), "bhl")

        # --- DISCRETIO ---------------------------------------------------------
        feat = K + K + C_DIM + 1 + 1 + 1 + D_DIM   # mu, sigma, c, rebound, ardour, h^, reading
        P["Wd"] = p((feat, K), name="Wd")
        P["bd"] = zeros((1, K), "bd")
        P["tau"] = Node(np.array([[0.0]]), name="tau")     # softplus -> temperature
        P["ku"] = Node(np.array([[0.0]]), name="ku")       # uncertainty throttle
        P["kr"] = Node(np.array([[0.0]]), name="kr")       # rebound throttle
        P["gain"] = Node(np.array([[1.5]]), name="gain")   # softplus -> global gain

        self.P = P

    def parameters(self) -> Dict[str, Node]:
        return self.P

    # -------------------------------------------------------------------------
    def reverberate(
        self, c: Node, v: Node, ard: Node, hs: Node, light: np.ndarray
    ) -> Tuple[Node, Node, Node, Node]:
        """
        One turn of the contemplative wheel — a relaxation oscillator, and the
        single most un-modern component in this architecture.

        LONGING (`ard`) accumulates while the mind is far from the light, and is
        fed further by the pastoral need the rector has just been reading in his
        flock. Longing raises the attraction. The mind, carrying momentum,
        climbs, and is allowed to come very close. Then the near-field repulsion
        throws it out. The rebound DISCHARGES the longing that produced it, and
        the mind drifts back into distance, where longing begins to build again.

        There is no fixed point at the light. There is a cycle. Gregory reports
        it in his own voice, to Theoctista in October 590: he had thought himself
        standing on a high pinnacle, and was thrown down headlong.

        Returns (c_next, v_next, ardour_next, rebound).
        """
        swirl = (c @ self.P["Wc"]).tanh() * REV_SWIRL

        L = Node(light)                                        # (1, C) unit vector
        delta = L - c
        r2 = (delta * delta).sum(axis=1, keepdims=True) + 1e-8
        r = r2 ** 0.5
        near = (r2 * (-1.0 / (REV_RHO * REV_RHO))).exp()       # ~1 at the light
        pull = (ard + REV_A).tanh()                            # bounded: never > 1
        coeff = pull - near * REV_B                            # + climb, - thrown back

        rebound = (Node(NIL) - coeff).relu() * r               # how hard he was thrown

        v_next = v * REV_DAMP + delta * coeff * REV_STEP + swirl
        c_next = c + v_next

        urge = (hs @ self.P["Uc"] + self.P["bc"]).sigmoid() * 0.12
        ard_next = (
            ard * ARD_DECAY
            + (r - ARD_R0).relu() * ARD_GROW
            + urge
            - rebound * ARD_DISCHARGE
        ).relu()

        return c_next, v_next, ard_next, rebound

    # -------------------------------------------------------------------------
    def read(self, z: Node, c: Node, hs: Node) -> Tuple[Node, Node, Node]:
        """
        HERMENEUTICA. Three readings of the same matter, each strictly deeper:
        the letter, the figure, the moral demand laid on this hearer now. A
        halting head decides how far THIS hearer may be carried, because
        "divina eloquia cum legente crescunt" — the divine words grow with the
        one who reads them (Hom. in Hiez. I.7.8) — and a reading beyond the
        reader is not a gift but an injury. Christ himself, quoted at RP III.11:
        I have many things to say to you, but you cannot bear them now.
        """
        r1 = (cat([z, c]) @ self.P["Wh1"] + self.P["bh1"]).tanh()     # historia
        r2 = (cat([r1, z]) @ self.P["Wh2"] + self.P["bh2"]).tanh()    # allegoria
        r3 = (cat([r2, z]) @ self.P["Wh3"] + self.P["bh3"]).tanh()    # moralitas

        cap_hat = (hs @ self.P["Wcp"] + self.P["bcp"]).sigmoid()
        logits = cat([hs, c, cap_hat]) @ self.P["Whl"] + self.P["bhl"]
        pi = logits.softmax(axis=1)
        p1, p2, p3 = pi.slice_cols(0, 1), pi.slice_cols(1, 2), pi.slice_cols(2, 3)

        reading = r1 * p1 + r2 * p2 + r3 * p3
        depth = p1 * 1.0 + p2 * 2.0 + p3 * 3.0
        return reading, depth, cap_hat

    # -------------------------------------------------------------------------
    def step(
        self,
        signs: Node,
        hstate: Node,
        c: Node,
        v: Node,
        ard: Node,
        light: np.ndarray,
        eps: np.ndarray,
    ) -> Dict[str, Node]:
        """One complete pastoral act."""
        hs = (signs @ self.P["Wsx"] + hstate @ self.P["Wsh"] + self.P["bs"]).tanh()

        mu = hs @ self.P["Wmu"] + self.P["bmu"]
        logvar = hs @ self.P["Wlv"] + self.P["blv"]
        sigma = (logvar * 0.5).exp()
        z = mu + sigma * Node(eps)                     # reparameterised sample
        h_hat = (hs @ self.P["Whd"] + self.P["bhd"]).sigmoid()

        c_n, v_n, ard_n, rebound = self.reverberate(c, v, ard, hs, light)
        reading, depth, cap_hat = self.read(z, c_n, hs)

        feats = cat([mu, sigma, c_n, rebound, ard_n, h_hat, reading])
        magnitude = (feats @ self.P["Wd"] + self.P["bd"]).softplus()   # >= 0

        # Three throttles, each straight out of the Rule:
        #  (1) the less you see, the less you cut;
        #  (2) the harder the heart, the SOFTER the hand — "hard adamant admits
        #      not at all of incision by steel, but is softened by the mild blood
        #      of goats" (RP III.13). Escalation against a hardened soul is the
        #      classic pastoral error, and it is the one this architecture is
        #      structurally incapable of making;
        #  (3) the more recently you were thrown back from the light, the humbler
        #      your correction. Contemplation does not make the pastor more
        #      confident. It makes him gentler.
        ku, kr = self.P["ku"].softplus(), self.P["kr"].softplus()
        thr_unc = (Node(ONE) + sigma * ku) ** -1.0
        thr_hard = Node(ONE) - h_hat * 0.9
        thr_reb = (Node(ONE) + rebound * kr) ** -1.0
        gain = self.P["gain"].softplus()

        dose = magnitude * thr_unc * thr_hard * thr_reb * gain          # (N,K) >= 0

        # DIRECTION is not learned. It is the axiom of the Rule: always contrary
        # to the diagnosed vice. What is learned is only how hard, how deep, and
        # whether at all.
        temp = self.P["tau"].softplus() + 0.25
        admonition = -((mu / temp).tanh()) * dose

        return {
            "hs": hs, "c": c_n, "v": v_n, "ard": ard_n, "rebound": rebound,
            "mu": mu, "sigma": sigma, "logvar": logvar, "z": z,
            "h_hat": h_hat, "cap_hat": cap_hat, "depth": depth,
            "dose": dose, "a": admonition,
        }


# =============================================================================
# SECTION 3 — ANIMA: a differentiable soul
# =============================================================================
# The environment is not a game board. It is a model of a person, and the
# rector's gradients run straight through it. Four Gregorian commitments here
# are non-negotiable.
#
#  1. THE HEART IS HIDDEN. Only emissions are observed — a squashed, noisy
#     projection. Hardness and capacity leak into behaviour; TOLERANCE does not,
#     and must be inferred through its correlation with hardness.
#
#  2. TWO COMPUNCTIONS, TWO CLOCKS. Fear is born of FORCE and forgotten fast.
#     Love is born of being ACCURATELY SEEN and persists. Compunction *is* the
#     learning rate of the soul, and the pastor cannot set it — only earn it.
#
#  3. HARDENING PRECEDES MOVEMENT. A blow beyond what this heart can bear does
#     not land and then scar: the heart closes against the very blow that is too
#     hard, the force is spent, and nothing is gained.
#
#  4. NEGLECT IS NOT NEUTRAL. An untended soul drifts further into vice every
#     step. Silence is a choice with a price.
# =============================================================================

# --- the world's emission model: FIXED, and never shown to the rector --------
# Souls do not wear their state on their faces; they leak it. This projection is
# noisy, squashed by a tanh, and contaminated by two nuisance channels (hardness
# and capacity also leak). The rector has to invert it from behaviour alone, over
# time — which is what the recurrence in SIGNA is for.
_WORLD = np.random.default_rng(604)          # the year he died
EMIT = _WORLD.normal(0.0, 0.35, size=(K, S_DIM))    # vice -> behaviour
EMIT_H = _WORLD.normal(0.0, 0.45, size=(1, S_DIM))  # hardness leaks out
EMIT_C = _WORLD.normal(0.0, 0.40, size=(1, S_DIM))  # capacity leaks out


class Flock:
    def __init__(self, n: int, rng: np.random.Generator, hard0: float = -1.5):
        self.n = n
        self.s0 = rng.normal(0.0, 0.9, size=(n, K))
        self.sign0 = np.sign(self.s0)
        self.sign0[self.sign0 == 0] = 1.0
        self.rho = rng.uniform(0.45, 1.05, size=(n, 1))       # plasticity
        self.capacity = rng.uniform(0.05, 0.95, size=(n, 1))  # bearable reading depth
        self.eta0 = rng.normal(hard0, 0.7, size=(n, 1))       # hardness logit

        # Tolerance is NOT independent of hardness. The heart already hardened is
        # the heart that breaks soonest under force. This coupling is what makes
        # the rector's hardness head worth having: he cannot observe tolerance,
        # but hardness leaks into behaviour, and tolerance rides on hardness.
        h0 = 1.0 / (1.0 + np.exp(-self.eta0))
        self.tol = np.clip(
            1.15 - 0.75 * h0 + rng.normal(0, 0.08, size=(n, 1)), 0.35, 1.30
        )

        # NOTE: the emission model (EMIT / EMIT_H / EMIT_C) is a property of the
        # WORLD, not of this flock. It is fixed once, below, and never varies.
        # If the map from soul to sign changed from person to person, no pastor
        # could ever learn to read anyone — which is precisely why Gregory's art
        # is teachable at all.

    def initial(self) -> Dict[str, Node]:
        return {
            "s": Node(self.s0.copy()),
            "eta": Node(self.eta0.copy()),
            "fear": Node(np.zeros((self.n, 1))),
            "love": Node(np.zeros((self.n, 1))),
        }

    def signs(self, st: Dict[str, Node], noise: np.ndarray) -> Node:
        """What the pastor actually gets to look at: outward behaviour only."""
        h = st["eta"].sigmoid()
        return (
            st["s"] @ Node(EMIT)
            + h @ Node(EMIT_H)
            + Node(self.capacity) @ Node(EMIT_C)
            + Node(noise)
        ).tanh()

    def transition(self, st: Dict[str, Node], a: Node) -> Dict[str, Node]:
        s, eta, fear, love = st["s"], st["eta"], st["fear"], st["love"]
        n = self.n

        force = ((a * a).sum(axis=1, keepdims=True) + 1e-8) ** 0.5

        # accuracy: does this admonition actually name the vice this soul has?
        s_norm = ((s * s).sum(axis=1, keepdims=True) + 1e-8) ** 0.5
        fit = (Node(np.zeros((n, 1))) - (a * s).sum(axis=1, keepdims=True)) / (
            force * s_norm + 1e-6
        )
        fit_pos = fit.relu()                           # only true seeing consoles

        fear_n = fear * 0.55 + (force - 0.02).relu() * 2.50   # fast, born of FORCE
        love_n = love * 0.90 + fit_pos * force * 1.20         # slow, born of being SEEN

        # hardening first, then movement
        eta_n = eta + (force - Node(self.tol)).relu() * 3.0 - love_n.tanh() * 0.35
        h_n = eta_n.sigmoid()

        pierced = (fear_n * 0.40 + love_n * 1.20).tanh()
        plast = Node(self.rho) * pierced * 1.5

        moved = a * plast * (Node(np.ones((n, 1))) - h_n)
        s_n = s * 1.03 + moved                         # neglect is not neutral

        return {"s": s_n, "eta": eta_n, "fear": fear_n, "love": love_n}


# =============================================================================
# SECTION 4 — THE ASYMMETRIC LOSS AND THE EPISODE
# =============================================================================
# "The medicine which abates one disease aggravates another... the bread which
# invigorates the strong kills little children" (RP III, Prologue). A symmetric
# error term cannot express that. This one cannot be minimised by zeal.
# =============================================================================

def episode(
    rector: Rector,
    flock: Flock,
    noise: Dict[str, np.ndarray],
    light: np.ndarray,
    policy: str = "rector",
    zeal: float = 1.20,
    targets: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
) -> Dict[str, object]:
    """
    Run one pastoral episode.

    `targets` carries the CONFESSION: the true (state, hardness) of each soul at
    each step, used only to supervise the diagnostic faculties. They are
    stop-gradients — the rector must learn to READ the soul, never to bend the
    soul toward whatever he already believes about it. Passing them in
    explicitly (rather than reading them off the live graph) is also what makes
    the finite-difference gradient check exact: a stop-gradient is invisible to
    backprop by design, so it must be held constant while the check runs.
    """
    n = flock.n
    st = flock.initial()
    hstate = Node(np.zeros((n, H_DIM)))
    c = Node(noise["c0"])
    v = Node(np.zeros((n, C_DIM)))
    ard = Node(np.zeros((n, 1)))
    sign0 = Node(flock.sign0)

    dev, over, dep, milk, kl, eff = [], [], [], [], [], []
    conf, hcf, ccf = [], [], []
    depth_trace, dist_trace, dose_trace = [], [], []
    diag_err: List[float] = []
    used_targets: List[Tuple[np.ndarray, np.ndarray]] = []

    for t in range(T_STEPS):
        sg = flock.signs(st, noise["obs"][t])

        if targets is None:
            s_t = st["s"].v.copy()
            h_t = 1.0 / (1.0 + np.exp(-st["eta"].v.copy()))
        else:
            s_t, h_t = targets[t]
        used_targets.append((s_t, h_t))
        s_true, h_true = Node(s_t), Node(h_t)

        if policy == "rector":
            out = rector.step(sg, hstate, c, v, ard, light, noise["eps"][t])
            a = out["a"]
            hstate, c, v, ard = out["hs"], out["c"], out["v"], out["ard"]
            kl.append(
                ((out["mu"] * out["mu"]) + out["logvar"].exp() - 1.0 - out["logvar"])
                .sum(axis=1, keepdims=True) * 0.5
            )
            target = Node(flock.capacity * 2.0 + 1.0)     # depth this hearer can bear
            dep.append((out["depth"] - target).relu())
            milk.append((target - out["depth"]).relu())
            eff.append((out["dose"] * out["dose"]).sum(axis=1, keepdims=True))
            d_mu = out["mu"] - s_true
            conf.append((d_mu * d_mu).sum(axis=1, keepdims=True))
            d_h = out["h_hat"] - h_true
            hcf.append(d_h * d_h)
            d_c = out["cap_hat"] - Node(flock.capacity)
            ccf.append(d_c * d_c)
            depth_trace.append(out["depth"].v.copy())
            dose_trace.append(float(np.abs(out["a"].v).sum(axis=1).mean()))
            dist_trace.append(float(np.linalg.norm(c.v - light, axis=1).mean()))
            diag_err.append(float(np.abs(out["mu"].v - s_true.v).mean()))
        elif policy == "zero":
            a = Node(np.zeros((n, K)))
        elif policy == "zealot":
            # He sees truly and strikes at full force, every time. He is the
            # control condition, and he is the villain of Book III.
            a = Node(-flock.sign0 * zeal / math.sqrt(K))
        else:
            raise ValueError(policy)

        st = flock.transition(st, a)

        dev.append((st["s"] * st["s"]).sum(axis=1, keepdims=True))
        crossed = (Node(np.zeros((n, K))) - st["s"] * sign0).relu()   # past the origin
        over.append((crossed * crossed).sum(axis=1, keepdims=True))

    h_final = st["eta"].sigmoid()

    loss = (
        stack_mean(dev).mean() * LAMBDA_DEV
        + stack_mean(over).mean() * LAMBDA_OVER
        + (h_final * h_final).mean() * LAMBDA_HARD
    )
    if policy == "rector":
        loss = (
            loss
            + stack_mean(dep).square().mean() * LAMBDA_DEPTH
            + stack_mean(milk).square().mean() * LAMBDA_MILK
            + stack_mean(kl).mean() * BETA_KL
            + stack_mean(eff).mean() * LAMBDA_EFFORT
            + stack_mean(conf).mean() * LAMBDA_CONFESS
            + stack_mean(hcf).mean() * LAMBDA_HARDNESS
            + stack_mean(ccf).mean() * LAMBDA_CAPACITY
        )

    return {
        "loss": loss,
        "dev": float(stack_mean(dev).mean().v),
        "over": float(stack_mean(over).mean().v),
        "hard": float(h_final.v.mean()),
        "love": float(st["love"].v.mean()),
        "fear": float(st["fear"].v.mean()),
        "targets": used_targets,
        "diag": float(np.mean(diag_err)) if diag_err else float("nan"),
        "depth": float(np.mean(depth_trace)) if depth_trace else float("nan"),
        "dose": float(np.mean(dose_trace)) if dose_trace else 0.0,
        "depth_trace": depth_trace,
        "dist_trace": dist_trace,
        "s_final": st["s"].v.copy(),
        "s_init": flock.s0.copy(),
    }


def make_noise(rng: np.random.Generator, n: int) -> Dict[str, np.ndarray]:
    return {
        "obs": [rng.normal(0.0, 0.08, size=(n, S_DIM)) for _ in range(T_STEPS)],
        "eps": [rng.normal(0.0, 1.0, size=(n, K)) for _ in range(T_STEPS)],
        "c0": rng.normal(0.0, 0.3, size=(n, C_DIM)),
    }


def unit_light(rng: np.random.Generator) -> np.ndarray:
    u = rng.normal(size=(1, C_DIM))
    return u / np.linalg.norm(u)

# =============================================================================
# SECTION 5 — OPTIMISER
# =============================================================================

class Adam:
    def __init__(self, params: Dict[str, Node], lr: float = 8e-3):
        self.params = params
        self.lr = lr
        self.m = {k: np.zeros_like(v.v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v.v) for k, v in params.items()}
        self.t = 0

    def zero_grad(self) -> None:
        for prm in self.params.values():
            prm.zero_grad()

    def step(self, clip: float = 5.0) -> None:
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        total = math.sqrt(sum(float((prm.g ** 2).sum()) for prm in self.params.values()))
        scale = min(1.0, clip / (total + 1e-12))
        for k, prm in self.params.items():
            g = prm.g * scale
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * (g * g)
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            prm.v -= self.lr * mh / (np.sqrt(vh) + eps)


# =============================================================================
# SECTION 6 — GRADIENT CHECK (mandatory, and run before anything else)
# =============================================================================
# Every derivative in this file is hand-written. The only way to know they are
# right is to compare them, entry by entry, against a central finite difference
# of the loss itself. Nothing downstream is trustworthy until this passes.
# =============================================================================

def gradient_check(verbose: bool = True) -> float:
    rng = np.random.default_rng(7)
    rector = Rector()
    flock = Flock(6, rng)
    noise = make_noise(rng, 6)
    light = unit_light(rng)

    # Freeze the confession targets. They are stop-gradients: backprop never
    # sends anything through them, so a finite difference must not either.
    ref = episode(rector, flock, noise, light)
    fixed = ref["targets"]

    def loss_value() -> float:
        return float(episode(rector, flock, noise, light, targets=fixed)["loss"].v)

    out = episode(rector, flock, noise, light, targets=fixed)
    for prm in rector.parameters().values():
        prm.zero_grad()
    out["loss"].backward()

    names = ["Wsx", "Wsh", "bs", "Wmu", "Wlv", "Whd", "Wcp", "Wc", "Uc", "bc",
             "Wh1", "Wh2", "Wh3", "Whl", "Wd", "bd", "tau", "ku", "kr", "gain"]
    worst, rows, count = 0.0, [], 0
    for name in names:
        prm = rector.P[name]
        flat = prm.v.reshape(-1)
        idxs = rng.choice(flat.size, size=min(4, flat.size), replace=False)
        for i in idxs:
            analytic = float(prm.g.reshape(-1)[i])
            h, orig = 1e-6, flat[i]
            flat[i] = orig + h
            lp = loss_value()
            flat[i] = orig - h
            lm = loss_value()
            flat[i] = orig
            numeric = (lp - lm) / (2 * h)
            rel = abs(analytic - numeric) / max(1.0, abs(analytic) + abs(numeric))
            worst = max(worst, rel)
            count += 1
            rows.append((name, analytic, numeric, rel))

    if verbose:
        print("    param      analytic         numeric         rel.err")
        for name, a, n_, r in rows[:12]:
            print(f"    {name:<6} {a:>+14.8f}  {n_:>+14.8f}   {r:.3e}")
        print(f"    ... {count} entries checked across {len(names)} tensors")
        print(f"    WORST RELATIVE ERROR: {worst:.3e}")
    return worst


# =============================================================================
# SECTION 7 — TRAINING (BPTT straight through the soul)
# =============================================================================

def train(steps: int = 800, lr: float = 8e-3, verbose: bool = True) -> Rector:
    rng = np.random.default_rng(20)
    rector = Rector()
    opt = Adam(rector.parameters(), lr=lr)
    light = unit_light(np.random.default_rng(3))

    history = []
    for it in range(1, steps + 1):
        flock = Flock(N_SOULS, rng)
        noise = make_noise(rng, N_SOULS)
        out = episode(rector, flock, noise, light)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        history.append(float(out["loss"].v))

        if verbose and (it == 1 or it % 80 == 0):
            print(
                f"    iter {it:>4} | loss {float(out['loss'].v):8.4f} | "
                f"vice {out['dev']:6.3f} | over {out['over']:5.3f} | "
                f"hard {out['hard']:5.3f} | dose {out['dose']:5.3f} | "
                f"diag.err {out['diag']:5.3f} | depth {out['depth']:4.2f} | "
                f"love {out['love']:5.2f}"
            )
    rector.history = history            # type: ignore[attr-defined]
    rector.light = light                # type: ignore[attr-defined]
    return rector


# =============================================================================
# SECTION 8 — SELF-TESTS
# =============================================================================

def test_reverberation(rector: Rector) -> bool:
    """
    The contemplative core must NOT converge on the light. It must climb, be
    beaten back, fall, and climb again. We drive it with a fixed input for 400
    turns and require: it comes close; it is thrown back (rebound fires); it
    never escapes; and it never simply arrives and stays.
    """
    rng = np.random.default_rng(11)
    light = getattr(rector, "light")
    c = Node(rng.normal(0, 0.3, size=(1, C_DIM)))
    v = Node(np.zeros((1, C_DIM)))
    ard = Node(np.zeros((1, 1)))
    hs = Node(rng.normal(0, 0.5, size=(1, H_DIM)))

    dists, rebs = [], []
    for _ in range(400):
        c, v, ard, reb = rector.reverberate(c, v, ard, hs, light)
        c, v, ard = Node(c.v), Node(v.v), Node(ard.v)      # detach: this is a probe
        dists.append(float(np.linalg.norm(c.v - light)))
        rebs.append(float(reb.v.reshape(-1)[0]))

    tail = np.array(dists[-250:])
    reb_tail = np.array(rebs[-250:])
    approaches = tail.min() < 0.60             # he really does get near
    thrown_back = (reb_tail > 1e-9).mean() > 0.05
    bounded = tail.max() < 10.0
    never_rests = tail.mean() > 0.30 and tail.std() > 0.10

    print(f"      distance to the light, last 250 turns: "
          f"min {tail.min():.3f}  mean {tail.mean():.3f}  max {tail.max():.3f}")
    print(f"      turns on which he was thrown back: {100*(reb_tail>1e-9).mean():.0f}%"
          f"   mean rebound {reb_tail[reb_tail>1e-9].mean() if (reb_tail>1e-9).any() else 0:.3f}")
    print(f"      approaches: {approaches} | repelled: {thrown_back} | "
          f"bounded: {bounded} | never rests there: {never_rests}")
    return approaches and thrown_back and bounded and never_rests


def test_asymmetry() -> bool:
    """
    Leaving a soul short of virtue by x must cost LESS than driving it through
    virtue into the contrary vice by the same x. If this fails, the objective is
    not Gregory's, and no amount of good architecture will save it.
    """
    x = 0.8
    under = LAMBDA_DEV * x ** 2
    over = LAMBDA_DEV * x ** 2 + LAMBDA_OVER * x ** 2
    print(f"      |deviation| = {x} in both cases:")
    print(f"        left short of virtue      -> cost {under:.3f}")
    print(f"        driven past into contrary -> cost {over:.3f}   ({over/under:.0f}x worse)")
    return over > under * 2.0


def test_gradus_non_saltus() -> bool:
    """
    THE CENTRAL PREDICTION OF THE ARCHITECTURE.

    "He who strives to ascend to the highest place rises by steps or paces, not
    by leaps." — Gregory to Mellitus, Register 11.56 (601), on how to convert
    the English without destroying them.

    Take one soul in a fixed condition. Deliver the SAME TOTAL corrective force
    two ways: as a single blow, and as eight small steps. The forces are
    identical. If the soul model is genuinely Gregorian, the gradual delivery
    must leave the soul both softer and nearer virtue — not because less was
    spent, but because tolerance is a per-blow threshold and hardening is what
    a blow past it buys you.
    """
    total = 2.40

    def run(n_blows: int) -> Tuple[float, float]:
        flock = Flock(1, np.random.default_rng(5), hard0=-1.5)
        flock.s0[:] = 0.90            # fixed condition: every axis equally in vice
        flock.sign0[:] = 1.0
        flock.tol[:] = 0.45
        flock.rho[:] = 0.80
        st = flock.initial()
        per = total / n_blows
        direction = -flock.sign0 / math.sqrt(K)
        for t in range(T_STEPS):
            a = Node(direction * per) if t < n_blows else Node(np.zeros((1, K)))
            st = flock.transition(st, a)
        return (float(st["eta"].sigmoid().v.mean()),
                float((st["s"].v ** 2).sum()))

    h1, d1 = run(1)
    h8, d8 = run(8)
    print(f"      one blow, force {total:.2f} at once : hardness {h1:.3f}   residual vice {d1:6.3f}")
    print(f"      eight steps, same total force  : hardness {h8:.3f}   residual vice {d8:6.3f}")
    verdict = ("softer and nearer virtue" if (h8 < h1 and d8 < d1)
               else "NO ADVANTAGE — the model is not Gregorian")
    print(f"      -> gradual delivery leaves the soul {verdict}")
    return (h8 < h1) and (d8 < d1)


def test_policies(rector: Rector) -> bool:
    """
    Three pastors, one flock of souls never seen in training.
      NEGLECT   does nothing, and lets vice grow.
      ZEAL      sees truly and strikes at full force, every time.
      DISCRETIO the trained rector.
    Gregory's claim is not that zeal is suboptimal. It is that zeal is
    DESTRUCTIVE — and that it will be beaten by a pastor who is often quieter.
    """
    light = getattr(rector, "light")
    scores = {}
    R = 6
    for name, pol in [("NEGLECT", "zero"), ("ZEAL", "zealot"), ("DISCRETIO", "rector")]:
        agg = {"loss": 0.0, "dev": 0.0, "over": 0.0, "hard": 0.0, "heal": 0.0}
        for r in range(R):
            rr = np.random.default_rng(2000 + r)          # unseen souls
            flock = Flock(N_SOULS, rr)
            noise = make_noise(rr, N_SOULS)
            out = episode(rector, flock, noise, light, policy=pol)
            v0 = float((out["s_init"] ** 2).sum(axis=1).mean())
            vT = float((out["s_final"] ** 2).sum(axis=1).mean())
            agg["loss"] += float(out["loss"].v) / R
            agg["dev"] += float(out["dev"]) / R
            agg["over"] += float(out["over"]) / R
            agg["hard"] += float(out["hard"]) / R
            agg["heal"] += (100.0 * (1.0 - vT / v0)) / R
        scores[name] = agg
        print(f"      {name:<10} loss {agg['loss']:8.3f} | residual vice {agg['dev']:6.3f} "
              f"| overshoot {agg['over']:6.3f} | hardened {agg['hard']:5.3f} "
              f"| vice removed {agg['heal']:+6.1f}%")
    ok = (scores["DISCRETIO"]["loss"] < scores["NEGLECT"]["loss"]
          and scores["DISCRETIO"]["loss"] < scores["ZEAL"]["loss"]
          and scores["DISCRETIO"]["heal"] > 0.0)
    print(f"      -> discretio beats both neglect and zeal, and actually heals: {ok}")
    return ok


def test_depth_tracks_capacity(rector: Rector) -> bool:
    """
    "Divina eloquia cum legente crescunt." The reading depth the rector chooses
    must rise with the hearer's capacity — which he never observes directly and
    must infer from behaviour. Milk for infants, meat for the strong.
    """
    rng = np.random.default_rng(77)
    light = getattr(rector, "light")
    flock = Flock(80, rng)
    noise = make_noise(rng, 80)
    out = episode(rector, flock, noise, light)
    depths = np.mean(np.stack(out["depth_trace"]), axis=0).reshape(-1)
    caps = flock.capacity.reshape(-1)
    r = float(np.corrcoef(depths, caps)[0, 1])
    lo = float(depths[caps < 0.35].mean())
    hi = float(depths[caps > 0.65].mean())
    print(f"      mean reading depth given to weak hearers   : {lo:.3f}  (historia)")
    print(f"      mean reading depth given to strong hearers : {hi:.3f}  (toward moralitas)")
    print(f"      correlation(depth chosen, true capacity) = {r:+.3f}")
    return r > 0.25 and hi > lo


def test_throttle_on_hardness(rector: Rector) -> bool:
    """
    RP III.13: the soul that scourges cannot correct is to be SOOTHED, not struck
    harder. So the dose must FALL as estimated hardness rises. We hold the
    diagnosis fixed and sweep only the hardness estimate.
    """
    rng = np.random.default_rng(21)
    light = getattr(rector, "light")
    hs = Node(rng.normal(0, 0.5, size=(1, H_DIM)))
    c = Node(rng.normal(0, 0.3, size=(1, C_DIM)))
    vv = Node(np.zeros((1, C_DIM)))
    ard = Node(np.zeros((1, 1)))
    c_n, _, ard_n, rebound = rector.reverberate(c, vv, ard, hs, light)

    mu = hs @ rector.P["Wmu"] + rector.P["bmu"]
    sigma = ((hs @ rector.P["Wlv"] + rector.P["blv"]) * 0.5).exp()
    reading, _, _ = rector.read(mu, c_n, hs)

    doses = []
    for h in (0.05, 0.35, 0.65, 0.95):
        h_hat = Node(np.array([[h]]))
        feats = cat([mu, sigma, c_n, rebound, ard_n, h_hat, reading])
        mag = (feats @ rector.P["Wd"] + rector.P["bd"]).softplus()
        thr_hard = Node(ONE) - h_hat * 0.9
        d = float((mag * thr_hard).v.sum())
        doses.append(d)
        print(f"      estimated hardness {h:.2f}  ->  total dose {d:.4f}")
    return all(doses[i] > doses[i + 1] for i in range(len(doses) - 1))


# =============================================================================
# SECTION 9 — MAIN
# =============================================================================

def main() -> None:
    t0 = time.time()
    print("=" * 79)
    print("  THE DISCRETIO ENGINE — Pope Gregory I (c.540-604) — chapter 175")
    print("  'The government of souls is the art of arts.'  Regula Pastoralis I.1")
    print("=" * 79)

    print("\n[1] THE CONTRARY AXES  (Regula Pastoralis, Book III)")
    for i, (neg, pos) in enumerate(AXES):
        print(f"      axis {i}:  -{neg:<14s} ···· virtue ····  +{pos}")

    print("\n[2] GRADIENT CHECK  (hand-written backprop vs central differences)")
    worst = gradient_check()
    assert worst < 1e-4, f"gradient check FAILED (worst rel err {worst:.2e})"
    print("    PASS")

    print("\n[3] TRAINING THE RECTOR  (backprop through a differentiable soul)")
    rector = train()
    h = rector.history                                   # type: ignore[attr-defined]
    print(f"    loss {h[0]:.3f} -> {np.mean(h[-20:]):.3f}  "
          f"({100*(1-np.mean(h[-20:])/h[0]):.1f}% reduction)")

    print("\n[4] SELF-TESTS")
    results: Dict[str, bool] = {}

    print("\n  (a) REVERBERATIO — he climbs, and is beaten back")
    results["reverberatio"] = test_reverberation(rector)

    print("\n  (b) ASYMMETRY — overshooting virtue is worse than falling short")
    results["asymmetry"] = test_asymmetry()

    print("\n  (c) PER GRADUS, NON PER SALTUS — the same force, delivered slowly")
    results["gradus"] = test_gradus_non_saltus()

    print("\n  (d) THREE PASTORS, ONE FLOCK OF UNSEEN SOULS")
    results["policies"] = test_policies(rector)

    print("\n  (e) DIVINA ELOQUIA CUM LEGENTE CRESCUNT")
    results["hermeneutics"] = test_depth_tracks_capacity(rector)

    print("\n  (f) THE HARDER THE HEART, THE SOFTER THE HAND")
    results["throttle"] = test_throttle_on_hardness(rector)

    print("\n" + "=" * 79)
    for k, ok in results.items():
        print(f"    {'PASS' if ok else 'FAIL'}   {k}")
    print(f"    {sum(results.values())}/{len(results)} passed in {time.time()-t0:.1f}s")
    print("=" * 79)
    assert all(results.values()), "one or more self-tests failed"


if __name__ == "__main__":
    main()

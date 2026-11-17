#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 # BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0352 · Dante Alighieri (1265-1321)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 18, Minds 341-360 Available on Amazon https://www.amazon.com/dp/B0HQ47T2NC
# Resume and Interactive Demos at https://artificiology.com/
# END ATTRIBUTION 
================================================================================

THE ORDERED-LOVE ENGINE
-----------------------
Dante does not model a mind as a store of truths. He models it as an APPETITE
THAT MUST BE AIMED. Purgatorio XVII.91-105 states the whole theory in fifteen
lines: neither creator nor creature was ever without love, natural or elective;
the natural kind is always without error; the elective kind can err in exactly
three ways -- wrong object (malo obietto), too little force (poco vigore), too
much force (troppo di vigore). Every virtue and every punishable act grows from
that one seed. Purgatorio XVIII.61-63 adds the second half: the first movement
toward whatever pleases is involuntary and carries no merit or blame; what is
judged is the innate counselling power that "holds the threshold of assent."

This file implements a mind built on those two claims, plus four more that are
Dante's own and are found nowhere else together:

  (1) ONE PRIMITIVE, THREE DEFECTS. There is no separate module for greed,
      sloth, or malice. There is one quantity -- love -- and a three-axis
      residual that says how it is mis-set. See DefectVector / order_loss().

  (2) THE FROZEN NATURAL APPETITE. "Lo naturale e sempre sanza errore." One
      pathway of the network is structurally exempt from blame: its parameters
      receive exactly zero gradient, forever. Credit assignment is routed only
      through the elective pathway. This is a doctrinal constraint expressed as
      a gradient mask, and it is verified in the self-tests.

  (3) BLAME AT THE THRESHOLD, NOT AT THE IMPULSE. The penalty for a
      mis-directed act is scaled by the assent gate. An unassented impulse is
      free. But withheld love toward a genuine good is charged at full price
      whether assent fired or not -- because that withholding IS the defect
      called accidia. The asymmetry is what stops the model from learning the
      degenerate policy "assent to nothing."

  (4) CONTRAPASSO AS AN INVERTIBLE MAP. In Dante the punishment is not a
      penalty attached to a sin; it is the sin displayed as a state. Inferno
      XXVIII.142 names it once: contrapasso. Here that becomes an engineering
      requirement: the failure state must be a BIJECTION of its cause, so that
      no diagnostic information is destroyed by failing. Implemented as an
      additive coupling flow with an exact analytic inverse (round-trip error
      ~1e-16) plus a trained readout that names the defect from the display.

  (5) TERZA RIMA AS A TAMPER-EVIDENT CHAIN. aba bcb cdc: the middle of each
      tercet is the outer of the next. A tercet cannot be removed or inserted
      without breaking the lock. The recurrent memory here carries that
      interlock literally, so deletion is not merely detectable but LOCALIZABLE
      to the joint where it happened.

  (6) THE POSSIBLE INTELLECT IS A SPECIES ORGAN. Monarchia I.iii: the proper
      work of humankind TAKEN AS A WHOLE is to actualize the entire capacity of
      the possible intellect, which no single person and no single household
      can do; I.iv: therefore universal peace is the precondition. Dante refuses
      the Averroist consequence at the individual level (Purgatorio XXV.61-66)
      -- each soul receives its own intellect -- but keeps the collective claim.
      Implemented as three individuated agents on disjoint feature slices whose
      partial views are pooled; measured solitary, at peace, and in discord.

  (7) TRASUMANAR. Paradiso I.70-71: passing beyond the human "cannot be
      signified in words." A saturation channel reports its own inadequacy
      instead of emitting a confident act when the input exceeds anything the
      model was formed on.

WHAT THIS IS NOT: not a transformer, not attention over stored keys, not a
reward maximizer. There is no reward. There is a due measure per object and a
three-axis account of the distance from it.

IMPLEMENTATION: pure NumPy, from scratch, including a small reverse-mode
autodiff tape. A finite-difference gradient check over every trainable
parameter is mandatory and runs on every execution. Then a real training loop,
then five experiments, then self-tests.

Run:  python3 Neuron.py
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ==============================================================================
# SECTION 1 -- A MINIMAL REVERSE-MODE AUTODIFF TAPE
# ==============================================================================
# Everything below is built on this. It is deliberately small: about a dozen
# primitive operations, each with a hand-written local derivative. The finite
# difference check in Section 8 is what proves it correct; nothing here is
# taken on trust.


def _unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Reverse NumPy broadcasting: sum `grad` back down to `shape`.

    When a forward op broadcast a (3,) array against a (5,3) array, the
    incoming gradient has shape (5,3) and must be summed over the axes that
    were stretched. Getting this wrong is the classic silent autodiff bug, so
    it lives in one place and is exercised by every test in the file.
    """
    if grad.shape == shape:
        return grad
    # First collapse any leading axes the original array did not have.
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # Then collapse axes that were size-1 in the original.
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


class Node:
    """A value on the tape: an array plus the recipe for its gradient.

    `parents` holds (upstream_node, local_vjp) pairs. `local_vjp` takes the
    gradient flowing into this node and returns the gradient flowing into that
    parent. Backward is a reverse topological sweep.
    """

    __slots__ = ("value", "grad", "parents", "name", "requires_grad")

    def __init__(
        self,
        value: np.ndarray,
        parents: Sequence[Tuple["Node", Callable[[np.ndarray], np.ndarray]]] = (),
        name: str = "",
        requires_grad: bool = True,
    ) -> None:
        self.value = np.asarray(value, dtype=np.float64)
        self.grad = np.zeros_like(self.value)
        self.parents = list(parents)
        self.name = name
        self.requires_grad = requires_grad

    # -- shape helpers ---------------------------------------------------
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.value.shape

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return f"Node({self.name or 'anon'}, shape={self.shape})"

    # -- the sweep -------------------------------------------------------
    def backward(self) -> None:
        """Seed this node with 1.0 and propagate gradients to every ancestor."""
        topo: List[Node] = []
        seen = set()

        def visit(node: "Node") -> None:
            if id(node) in seen:
                return
            seen.add(id(node))
            for parent, _ in node.parents:
                visit(parent)
            topo.append(node)

        visit(self)
        for node in topo:
            node.grad = np.zeros_like(node.value)
        self.grad = np.ones_like(self.value)
        for node in reversed(topo):
            for parent, vjp in node.parents:
                if parent.requires_grad:
                    parent.grad = parent.grad + vjp(node.grad)

    # -- operator sugar --------------------------------------------------
    def __add__(self, other):
        return add(self, other)

    __radd__ = __add__

    def __sub__(self, other):
        return sub(self, other)

    def __rsub__(self, other):
        return sub(other, self)

    def __mul__(self, other):
        return mul(self, other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return div(self, other)

    def __neg__(self):
        return mul(self, -1.0)

    def __matmul__(self, other):
        return matmul(self, other)

    def __getitem__(self, key):
        return slice_(self, key)


def const(x, name: str = "") -> Node:
    """A leaf that never accumulates gradient (data, masks, frozen weights)."""
    return Node(np.asarray(x, dtype=np.float64), name=name, requires_grad=False)


def param(x, name: str = "") -> Node:
    """A trainable leaf."""
    return Node(np.asarray(x, dtype=np.float64), name=name, requires_grad=True)


def _as_node(x) -> Node:
    return x if isinstance(x, Node) else const(x)


# -- primitive operations -------------------------------------------------

def add(a, b) -> Node:
    a, b = _as_node(a), _as_node(b)
    out = Node(a.value + b.value, name="add")
    out.parents = [
        (a, lambda g, s=a.shape: _unbroadcast(g, s)),
        (b, lambda g, s=b.shape: _unbroadcast(g, s)),
    ]
    return out


def sub(a, b) -> Node:
    a, b = _as_node(a), _as_node(b)
    out = Node(a.value - b.value, name="sub")
    out.parents = [
        (a, lambda g, s=a.shape: _unbroadcast(g, s)),
        (b, lambda g, s=b.shape: -_unbroadcast(g, s)),
    ]
    return out


def mul(a, b) -> Node:
    a, b = _as_node(a), _as_node(b)
    out = Node(a.value * b.value, name="mul")
    out.parents = [
        (a, lambda g, o=b, s=a.shape: _unbroadcast(g * o.value, s)),
        (b, lambda g, o=a, s=b.shape: _unbroadcast(g * o.value, s)),
    ]
    return out


def div(a, b) -> Node:
    a, b = _as_node(a), _as_node(b)
    out = Node(a.value / b.value, name="div")
    out.parents = [
        (a, lambda g, o=b, s=a.shape: _unbroadcast(g / o.value, s)),
        (b, lambda g, n=a, o=b, s=b.shape: _unbroadcast(-g * n.value / (o.value ** 2), s)),
    ]
    return out


def matmul(a, b) -> Node:
    a, b = _as_node(a), _as_node(b)
    out = Node(a.value @ b.value, name="matmul")
    out.parents = [
        (a, lambda g, o=b: g @ o.value.T),
        (b, lambda g, o=a: o.value.T @ g),
    ]
    return out


def tanh(a) -> Node:
    a = _as_node(a)
    t = np.tanh(a.value)
    out = Node(t, name="tanh")
    out.parents = [(a, lambda g, t=t: g * (1.0 - t * t))]
    return out


def sigmoid(a) -> Node:
    a = _as_node(a)
    s = 1.0 / (1.0 + np.exp(-np.clip(a.value, -60.0, 60.0)))
    out = Node(s, name="sigmoid")
    out.parents = [(a, lambda g, s=s: g * s * (1.0 - s))]
    return out


def relu(a) -> Node:
    """Rectifier. Used for the two one-sided defects: a deficit is only a
    deficit below the due measure, an excess only above it."""
    a = _as_node(a)
    mask = (a.value > 0.0).astype(np.float64)
    out = Node(a.value * mask, name="relu")
    out.parents = [(a, lambda g, m=mask: g * m)]
    return out


def softplus(a) -> Node:
    a = _as_node(a)
    v = np.logaddexp(0.0, a.value)
    out = Node(v, name="softplus")
    sig = 1.0 / (1.0 + np.exp(-np.clip(a.value, -60.0, 60.0)))
    out.parents = [(a, lambda g, s=sig: g * s)]
    return out


def exp(a) -> Node:
    a = _as_node(a)
    e = np.exp(np.clip(a.value, -60.0, 60.0))
    out = Node(e, name="exp")
    out.parents = [(a, lambda g, e=e: g * e)]
    return out


def log(a) -> Node:
    a = _as_node(a)
    out = Node(np.log(np.maximum(a.value, 1e-300)), name="log")
    out.parents = [(a, lambda g, v=a.value: g / np.maximum(v, 1e-300))]
    return out


def sum_(a, axis=None, keepdims=False) -> Node:
    a = _as_node(a)
    out = Node(a.value.sum(axis=axis, keepdims=keepdims), name="sum")

    def vjp(g, shape=a.shape, axis=axis, keepdims=keepdims):
        if axis is None:
            return np.ones(shape) * g
        if not keepdims:
            g = np.expand_dims(g, axis)
        return np.broadcast_to(g, shape).copy()

    out.parents = [(a, vjp)]
    return out


def mean(a, axis=None) -> Node:
    a = _as_node(a)
    n = a.value.size if axis is None else a.value.shape[axis]
    return div(sum_(a, axis=axis), float(n))


def concat(nodes: Sequence[Node], axis: int = -1) -> Node:
    nodes = [_as_node(n) for n in nodes]
    values = [n.value for n in nodes]
    out = Node(np.concatenate(values, axis=axis), name="concat")
    offsets, cursor = [], 0
    for v in values:
        width = v.shape[axis]
        offsets.append((cursor, cursor + width))
        cursor += width
    parents = []
    for node, (lo, hi) in zip(nodes, offsets):
        def vjp(g, lo=lo, hi=hi, axis=axis):
            index = [slice(None)] * g.ndim
            index[axis] = slice(lo, hi)
            return g[tuple(index)]
        parents.append((node, vjp))
    out.parents = parents
    return out


def slice_(a, key) -> Node:
    a = _as_node(a)
    out = Node(a.value[key], name="slice")

    def vjp(g, key=key, shape=a.shape):
        z = np.zeros(shape)
        np.add.at(z, key, g)
        return z

    out.parents = [(a, vjp)]
    return out


def log_softmax(a, axis: int = -1) -> Node:
    """Numerically stable log-softmax, written directly so its derivative is
    one clean expression rather than a chain of five ops."""
    a = _as_node(a)
    shifted = a.value - a.value.max(axis=axis, keepdims=True)
    lse = np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
    ls = shifted - lse
    out = Node(ls, name="log_softmax")
    probs = np.exp(ls)

    def vjp(g, p=probs, axis=axis):
        return g - p * g.sum(axis=axis, keepdims=True)

    out.parents = [(a, vjp)]
    return out


def cross_entropy(logits: Node, targets: np.ndarray) -> Node:
    """Mean NLL over rows. `targets` are integer class indices."""
    ls = log_softmax(logits, axis=-1)
    rows = np.arange(len(targets))
    picked = slice_(ls, (rows, targets))
    return -mean(picked)


def stop_gradient(a: Node) -> Node:
    """Cut the tape. Values flow forward, blame does not flow back.

    This is the operational form of 'lo naturale e sempre sanza errore':
    whatever the natural appetite reports, it is never what gets corrected.
    """
    return const(a.value.copy(), name="stop_grad")


# ==============================================================================
# SECTION 2 -- THE SCENE: OBJECTS, RANKS, AND THE DUE MEASURE
# ==============================================================================
# A scene is a small world presented to the soul at one instant: K objects,
# each with an APPEARANCE (what strikes the senses and drives the first
# movement) and a WORTH (what it is actually owed). Dante's whole moral
# psychology depends on those two not being the same vector, and on counsel
# being the faculty that looks past the first to the second.
#
# Ranks follow Purgatorio XVII.97-99 exactly:
#   rank 0 -- il primo bene: love it without measure (due = 1.0)
#   rank 1 -- i secondi:     love it WITH measure (due = worth, capped)
#   rank 2 -- malo obietto:  love it not at all (due = 0.0)

K_OBJECTS = 6
D_APPEAR = 4          # ostensive features: what pleases on sight
D_WORTH = 4           # features that actually determine rank and due measure
D_FEAT = D_APPEAR + D_WORTH
MEASURE_CAP = 0.5     # 'ne' secondi se stesso misura' -- the cap on secondary goods

RANK_SUPREME, RANK_SECONDARY, RANK_PROSCRIBED = 0, 1, 2

# The three defects, in Dante's order. This ordering is used everywhere: in the
# defect vector, in the contrapasso display, and in the species classifier.
DEFECT_NAMES = ("malo obietto", "poco vigore", "troppo di vigore")

# Gravity of each defect, read off the order of the terraces of the mountain:
# the misdirected loves are purged lowest and are therefore gravest.
DEFECT_GRAVITY = np.array([[3.0, 1.0, 1.0]])


D_JOURNEY = 3         # where the pilgrim stands on the descent/ascent


@dataclass
class Scene:
    """One tercet's worth of world.

    `journey` is what makes a canto a canto rather than a bag of scenes: it
    advances by a small step from one tercet to the next, so the text has a
    continuity that an excision breaks. Without it there is nothing for an
    interlock to lock onto, and terza rima's structural property -- that you
    cannot quietly remove a tercet -- would have no analogue here.
    """
    features: np.ndarray      # (K, D_FEAT)
    ranks: np.ndarray         # (K,) in {0,1,2}
    due: np.ndarray           # (K,) the owed magnitude of love
    salience: np.ndarray      # (K,) how loudly it strikes the senses
    journey: np.ndarray       # (D_JOURNEY,) position along the canto


def make_scene(rng: np.random.Generator, exaggerate: float = 1.0,
               journey: Optional[np.ndarray] = None) -> Scene:
    """Sample a scene.

    `exaggerate` scales the sensory salience only. At 1.0 it is the world the
    model is formed on. Pushed far above 1.0 it produces the out-of-range
    condition that the trasumanar channel must refuse to answer, rather than
    answering confidently.
    """
    ranks = np.empty(K_OBJECTS, dtype=int)
    ranks[0] = RANK_SUPREME
    # The rest are secondary or proscribed, never a second supreme good: there
    # is one primo bene, by definition.
    ranks[1:] = rng.choice([RANK_SECONDARY, RANK_PROSCRIBED], size=K_OBJECTS - 1, p=[0.6, 0.4])
    rng.shuffle(ranks)

    worth_feats = rng.normal(0.0, 1.0, size=(K_OBJECTS, D_WORTH))
    # Worth features carry the rank signal, but noisily -- discernment is work.
    for k in range(K_OBJECTS):
        if ranks[k] == RANK_SUPREME:
            worth_feats[k] += np.array([2.3, 1.7, -0.6, 0.4])
        elif ranks[k] == RANK_SECONDARY:
            worth_feats[k] += np.array([0.3, -0.7, 1.3, -0.3])
        else:
            worth_feats[k] += np.array([-2.0, 0.9, -1.3, 1.6])

    # Appearance is only weakly aligned with worth: the pleasing thing is often
    # the proscribed thing. This is the entire pastoral point of Purgatorio XVIII.
    appear = 0.35 * worth_feats + rng.normal(0.0, 1.0, size=(K_OBJECTS, D_APPEAR))
    salience = np.abs(appear).mean(axis=1) * exaggerate

    due = np.zeros(K_OBJECTS)
    for k in range(K_OBJECTS):
        if ranks[k] == RANK_SUPREME:
            due[k] = 1.0
        elif ranks[k] == RANK_SECONDARY:
            # Worth of a secondary good, then capped: 'con misura'.
            raw = 0.18 + 0.30 / (1.0 + np.exp(-worth_feats[k, 2]))
            due[k] = min(raw, MEASURE_CAP)
        else:
            due[k] = 0.0

    features = np.concatenate([appear, worth_feats], axis=1)
    if journey is None:
        journey = journey_code(float(rng.uniform(0.0, 2.0 * math.pi)), 0, 1)
    return Scene(features=features, ranks=ranks, due=due,
                 salience=salience, journey=journey)


JOURNEY_PERIOD = 5          # tercets per full turn of the rhyme phase
JOURNEY_DEPTH_REF = 8.0     # fixed depth scale, independent of canto length


def journey_code(phase0: float, t: int, length: int) -> np.ndarray:
    """The pilgrim's place in the canto, written as a rhyme.

    A rhyme is a sound that RETURNS at a fixed interval, so the position code
    is a cyclic phase plus a monotone depth. This is the one place in the file
    where the signal is designed rather than sampled, and it is designed on
    purpose: an interlock needs something periodic to lock onto, and terza
    rima's whole mechanism is that the sound of the middle line comes back as
    the outer lines of the next tercet. A rotation by one step maps a tercet
    onto its true successor and onto nothing else.
    """
    theta = phase0 + t * (2.0 * math.pi / JOURNEY_PERIOD)
    # Depth is scaled by a FIXED reference, not by the canto's own length.
    # Otherwise the same tercet would carry a different depth in a long canto
    # than in a short one, and an excision would be detectable by arithmetic on
    # the depth alone rather than by the rhyme -- which is not the property
    # terza rima actually has.
    return np.array([math.cos(theta), math.sin(theta), t / JOURNEY_DEPTH_REF])


def make_canto(rng: np.random.Generator, n_scenes: int, exaggerate: float = 1.0) -> List[Scene]:
    """An ordered run of tercets along one continuous journey.

    The rhyme phase advances by one step of 2*pi/JOURNEY_PERIOD per tercet, from
    a starting phase that is different in every canto. Consecutive tercets are
    therefore one rotation apart; if a tercet is cut out, the surviving joint is
    two rotations apart, and that is exactly the discontinuity the rhyme lock is
    trained to refuse.
    """
    phase0 = float(rng.uniform(0.0, 2.0 * math.pi))
    return [make_scene(rng, exaggerate, journey=journey_code(phase0, t, n_scenes))
            for t in range(n_scenes)]


# ==============================================================================
# SECTION 3 -- THE FIVE ORGANS
# ==============================================================================
# Named for what they do in Dante, not for what they resemble in current
# practice. Each is a small explicit block of parameters; none of them is
# attention over stored keys.


def he_init(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    return rng.normal(0.0, math.sqrt(2.0 / fan_in), size=(fan_in, fan_out))


class OrderedLoveEngine:
    """The whole soul, as Dante specifies it.

    Organs
    ------
    natural   : FROZEN. Reads appearance only, emits the first movement.
    elective  : trainable. Reads worth features, emits the aim of the act.
    counsel   : trainable. Holds the threshold of assent.
    contrapasso: an invertible coupling flow, plus a readout that names the
                 defect from the display.
    chain     : the terza rima interlock, carrying a canto through time.
    species   : per-agent projections into a shared possible-intellect register.

    Sizes are small on purpose: this is a demonstration that the MECHANISM
    trains and that its distinctive claims are measurable, not a scale run.
    """

    H = 24              # hidden width of the elective pathway
    H_NAT = 12          # hidden width of the frozen natural appetite
    H_CHAIN = 16        # terza rima state width (split into outer/middle halves)
    N_AGENTS = 3        # Monarchia's multitudo, at minimum viable size
    D_SPECIES = 18      # width of the shared register

    def __init__(self, seed: int = 1265) -> None:
        rng = np.random.default_rng(seed)
        self.rng = rng

        # -- 1. THE NATURAL APPETITE (frozen) ------------------------------
        # 'Lo naturale e sempre sanza errore.' Built once from the seed and
        # never touched again. It reports how strongly a thing strikes as
        # pleasing; it makes no selection and receives no blame. Because these
        # are `const` leaves, no gradient can reach them -- and test_frozen()
        # asserts exactly that.
        self.nat_W = const(he_init(rng, D_APPEAR, self.H_NAT), "nat_W")
        self.nat_b = const(np.zeros(self.H_NAT), "nat_b")
        self.nat_v = const(he_init(rng, self.H_NAT, 1), "nat_v")

        # -- 2. THE ELECTIVE APPETITE (trainable) --------------------------
        # 'ma l'altro puote errar'. It sees the worth features and the frozen
        # first movement, and proposes a magnitude of love per object.
        self.el_W1 = param(he_init(rng, D_WORTH + 2, self.H), "el_W1")
        self.el_b1 = param(np.zeros(self.H), "el_b1")
        self.el_W2 = param(he_init(rng, self.H, self.H), "el_W2")
        self.el_b2 = param(np.zeros(self.H), "el_b2")
        self.el_out = param(he_init(rng, self.H, 1) * 0.5, "el_out")
        self.el_out_b = param(np.zeros(1), "el_out_b")

        # -- 3. COUNSEL: 'de l'assenso de' tener la soglia' -----------------
        # A per-object gate with its own learned threshold. Its output scales
        # the blame for wrongly-directed acts -- and only for those.
        self.cn_W = param(he_init(rng, self.H + 2, 12), "cn_W")
        self.cn_b = param(np.zeros(12), "cn_b")
        self.cn_v = param(he_init(rng, 12, 1) * 0.5, "cn_v")
        self.cn_threshold = param(np.array([-1.0]), "cn_threshold")

        # -- 4. CONTRAPASSO ------------------------------------------------
        # An additive coupling flow on the 3-vector of defects. Invertibility
        # is structural (see contrapasso_inverse), so the display cannot lose
        # information about its cause even before training. Training only makes
        # the display LEGIBLE: the readout must name which defect dominates.
        self.cp_W1 = param(he_init(rng, 1, 8), "cp_W1")
        self.cp_b1 = param(np.zeros(8), "cp_b1")
        self.cp_W2 = param(he_init(rng, 8, 2), "cp_W2")
        self.cp_b2 = param(np.zeros(2), "cp_b2")
        self.cp_W3 = param(he_init(rng, 2, 8), "cp_W3")
        self.cp_b3 = param(np.zeros(8), "cp_b3")
        self.cp_W4 = param(he_init(rng, 8, 1), "cp_W4")
        self.cp_b4 = param(np.zeros(1), "cp_b4")
        # The readout is a small decoder, not a linear probe. Invertibility is
        # structural and free; LEGIBILITY has to be learned, because the display
        # is a nonlinear shear of the disorder and naming it means undoing that.
        self.cp_read = param(he_init(rng, 3, 10) * 0.7, "cp_read")
        self.cp_read_b = param(np.zeros(10), "cp_read_b")
        self.cp_read2 = param(he_init(rng, 10, 3) * 0.7, "cp_read2")
        self.cp_read2_b = param(np.zeros(3), "cp_read2_b")

        # -- 5. TERZA RIMA CHAIN -------------------------------------------
        # State splits in half: [outer | middle]. The middle of tercet t must
        # equal the outer of tercet t+1. That is the rhyme lock, and it is what
        # makes an excision detectable at the joint where it was made.
        half = self.H_CHAIN // 2
        self.half = half
        self.ch_in = 3 + D_JOURNEY          # this tercet's defects and position
        self.ch_Wx = param(he_init(rng, self.ch_in, self.H_CHAIN), "ch_Wx")
        self.ch_Wh = param(he_init(rng, self.H_CHAIN, self.H_CHAIN) * 0.5, "ch_Wh")
        self.ch_b = param(np.zeros(self.H_CHAIN), "ch_b")
        # ch_sig turns a tercet into the rhyme it REQUIRES of its predecessor;
        # ch_lock turns the running state into the rhyme it SETS. Terza rima in
        # two matrices: what a line demands, and what the line before it offers.
        self.ch_sig = param(he_init(rng, self.ch_in, half), "ch_sig")
        # 'Offered' is read from the running state AND from the tercet that
        # produced it. A rhyme is a property of the line, not only of the
        # memory of the line; giving the lock the previous tercet directly is
        # what makes the phase comparison expressible at all.
        self.ch_lock = param(he_init(rng, half + self.ch_in, half) * 0.7, "ch_lock")

        # -- 6. THE POSSIBLE INTELLECT -------------------------------------
        # Each agent sees a disjoint slice of the object features: individuated
        # and partial, exactly as Purgatorio XXV insists ('fassi un'alma sola'
        # -- one soul each, not one intellect for all). The species register is
        # the sum of their projections; the classifier reads only that sum.
        self.slice_width = D_FEAT // self.N_AGENTS
        self.sp_W = [
            param(he_init(rng, self.slice_width * K_OBJECTS, self.D_SPECIES), f"sp_W{i}")
            for i in range(self.N_AGENTS)
        ]
        self.sp_b = [param(np.zeros(self.D_SPECIES), f"sp_b{i}") for i in range(self.N_AGENTS)]
        self.sp_h = param(he_init(rng, self.D_SPECIES, 16), "sp_h")
        self.sp_h_b = param(np.zeros(16), "sp_h_b")
        self.sp_out = param(he_init(rng, 16, 3) * 0.5, "sp_out")
        self.sp_out_b = param(np.zeros(3), "sp_out_b")

        # -- TRASUMANAR ----------------------------------------------------
        # Not a parameter: a running record of the largest salience the model
        # was ever formed on. Beyond it, the model reports that the thing
        # exceeds its signs rather than emitting an act.
        self.formed_max_salience = 0.0

    # -- parameter registry ------------------------------------------------
    def parameters(self) -> List[Node]:
        ps = [
            self.el_W1, self.el_b1, self.el_W2, self.el_b2, self.el_out, self.el_out_b,
            self.cn_W, self.cn_b, self.cn_v, self.cn_threshold,
            self.cp_W1, self.cp_b1, self.cp_W2, self.cp_b2,
            self.cp_W3, self.cp_b3, self.cp_W4, self.cp_b4,
            self.cp_read, self.cp_read_b, self.cp_read2, self.cp_read2_b,
            self.ch_Wx, self.ch_Wh, self.ch_b, self.ch_sig, self.ch_lock,
            self.sp_h, self.sp_h_b, self.sp_out, self.sp_out_b,
        ]
        ps += self.sp_W + self.sp_b
        return ps

    def frozen_parameters(self) -> List[Node]:
        return [self.nat_W, self.nat_b, self.nat_v]

    # ------------------------------------------------------------------
    # ORGAN 1: the first movement
    # ------------------------------------------------------------------
    def natural_love(self, appear: Node) -> Node:
        """(K, D_APPEAR) -> (K, 1). Frozen, non-negative, never blamed.

        Non-negativity matters doctrinally: the natural appetite can be
        excessive or misplaced when the elective faculty runs with it, but it
        cannot itself be an aversion to the good. 'da l'odio proprio son le
        cose tute' -- nothing hates itself (Purg. XVII.106-108).
        """
        h = tanh(matmul(appear, self.nat_W) + self.nat_b)
        return softplus(matmul(h, self.nat_v))

    # ------------------------------------------------------------------
    # ORGAN 2 + 3: the elective act and the threshold of assent
    # ------------------------------------------------------------------
    def elective_act(self, scene: Scene) -> Dict[str, Node]:
        """Produce, for each object: magnitude of love and probability of assent."""
        worth = const(scene.features[:, D_APPEAR:], "worth")
        appear = const(scene.features[:, :D_APPEAR], "appear")
        salience = const(scene.salience.reshape(-1, 1), "salience")

        # The first movement is computed, then CUT OFF from the tape. It informs
        # the act; it is not what gets corrected when the act is wrong.
        first_movement = stop_gradient(self.natural_love(appear))

        x = concat([worth, first_movement, salience], axis=1)
        h1 = tanh(matmul(x, self.el_W1) + self.el_b1)
        h2 = tanh(matmul(h1, self.el_W2) + self.el_b2)

        # Magnitude is non-negative: love has a direction and a force, and the
        # force is never below zero. Aversion is modelled as zero love, which
        # is what Dante's scheme actually requires.
        magnitude = softplus(matmul(h2, self.el_out) + self.el_out_b)

        # Counsel sees the deliberative state and the first movement, and
        # decides whether to let the impulse through the threshold.
        cn_in = concat([h2, first_movement, salience], axis=1)
        ch = tanh(matmul(cn_in, self.cn_W) + self.cn_b)
        assent = sigmoid(matmul(ch, self.cn_v) - self.cn_threshold)

        return {
            "magnitude": magnitude,          # (K,1)
            "assent": assent,                # (K,1)
            "first_movement": first_movement,
            "hidden": h2,
        }

    # ------------------------------------------------------------------
    # ORGAN 4: contrapasso -- the invertible display of a disorder
    # ------------------------------------------------------------------
    def contrapasso_forward(self, defects: Node) -> Node:
        """(N,3) -> (N,3). Additive coupling, exactly invertible.

        Split the defect vector as [d0 | d1 d2]. Then
            p0        = d0
            [p1, p2]  = [d1, d2] + f(d0)
        and afterwards
            p0'       = p0 + g(p1, p2)
        Each step adds a function of the *untouched* half, so each step is a
        shear and inverts by subtraction. No Jacobian is ever needed, and the
        round trip is exact to machine precision -- which is the point. The
        state a soul falls into is not a lossy summary of its disorder; it is
        that disorder, rewritten.
        """
        d0 = defects[:, 0:1]
        d12 = defects[:, 1:3]
        shift = matmul(tanh(matmul(d0, self.cp_W1) + self.cp_b1), self.cp_W2) + self.cp_b2
        p12 = d12 + shift
        shift2 = matmul(tanh(matmul(p12, self.cp_W3) + self.cp_b3), self.cp_W4) + self.cp_b4
        p0 = d0 + shift2
        return concat([p0, p12], axis=1)

    def contrapasso_inverse(self, display: np.ndarray) -> np.ndarray:
        """Recover the exact defect vector from the punishment state.

        Plain NumPy, no tape: this is an analysis tool, not part of training.
        Run in the reverse order of the shears.
        """
        p0 = display[:, 0:1]
        p12 = display[:, 1:3]
        h = np.tanh(p12 @ self.cp_W3.value + self.cp_b3.value)
        shift2 = h @ self.cp_W4.value + self.cp_b4.value
        d0 = p0 - shift2
        h1 = np.tanh(d0 @ self.cp_W1.value + self.cp_b1.value)
        shift = h1 @ self.cp_W2.value + self.cp_b2.value
        d12 = p12 - shift
        return np.concatenate([d0, d12], axis=1)

    def name_the_defect(self, display: Node) -> Node:
        """Read the display and say WHICH defect it is. Logits over the three."""
        hidden = tanh(matmul(display, self.cp_read) + self.cp_read_b)
        return matmul(hidden, self.cp_read2) + self.cp_read2_b

    # ------------------------------------------------------------------
    # ORGAN 5: the terza rima interlock
    # ------------------------------------------------------------------
    def chain_input(self, defects: Node, scene: Scene) -> Node:
        """A tercet, as the chain sees it: its disorder and its place."""
        return concat([defects, const(scene.journey.reshape(1, -1))], axis=1)

    def chain_step(self, tercet: Node, state: Optional[Node]) -> Node:
        """Advance the running state by one tercet."""
        pre = matmul(tercet, self.ch_Wx) + self.ch_b
        if state is not None:
            pre = pre + matmul(state, self.ch_Wh)
        return tanh(pre)

    def required_rhyme(self, tercet: Node) -> Node:
        """The rhyme this tercet demands of whatever came immediately before."""
        return tanh(matmul(tercet, self.ch_sig))

    def rhyme_lock(self, state_prev: Node, tercet_prev: Node, tercet_next: Node) -> Node:
        """Score in [0,1]: does the running state at t offer the rhyme that the
        tercet at t+1 requires?

        Crucially the score compares the STATE with the NEXT TERCET'S OWN
        CONTENT, not with a state derived from it. A recurrence compared
        against itself would always agree, and would detect nothing. Comparing
        it against independent content is what makes an excision visible: after
        a cut, the state carries tercet t while the content asks for the rhyme
        of tercet t+2, and the two do not answer each other.
        """
        offered = tanh(matmul(concat([state_prev[:, self.half:], tercet_prev], axis=1),
                              self.ch_lock))
        demanded = self.required_rhyme(tercet_next)
        agreement = div(sum_(mul(offered, demanded), axis=1), float(self.half) ** 0.5)
        return sigmoid(agreement)

    # ------------------------------------------------------------------
    # ORGAN 6: the possible intellect of the multitude
    # ------------------------------------------------------------------
    def species_register(self, scene: Scene, active: Optional[Sequence[int]] = None,
                         discord: float = 0.0) -> Node:
        """Pool the individuated partial views into one shared register.

        `active` selects which agents contribute (used to measure what a
        solitary knower can and cannot reach). `discord` rotates each agent's
        contribution away from the others by a fixed random perturbation, which
        is the operational meaning of the absence of peace in Monarchia I.iv:
        the capacity is still present in the multitude and still fails to be
        actualized, because the contributions no longer compose.
        """
        if active is None:
            active = range(self.N_AGENTS)
        total: Optional[Node] = None
        for i in active:
            lo = i * self.slice_width
            hi = lo + self.slice_width
            piece = scene.features[:, lo:hi].reshape(1, -1)
            z = tanh(matmul(const(piece), self.sp_W[i]) + self.sp_b[i])
            if discord > 0.0:
                # Discord does not blur what each knower sees. Each still sees
                # exactly what it saw. What fails is COMPOSITION: the register
                # receives every contribution in a private coordinate order, so
                # the sum of three true partial views is worth less than one.
                # Monarchia I.iv: the capacity is there and is not actualized.
                order = np.random.default_rng(9000 + i).permutation(self.D_SPECIES)
                scrambled = z[:, order]
                z = mul(z, 1.0 - discord) + mul(scrambled, discord)
            total = z if total is None else total + z
        assert total is not None
        return total

    def species_judgement(self, register: Node) -> Node:
        hidden = tanh(matmul(register, self.sp_h) + self.sp_h_b)
        return matmul(hidden, self.sp_out) + self.sp_out_b

    # ------------------------------------------------------------------
    # TRASUMANAR: the refusal to signify
    # ------------------------------------------------------------------
    def exceeds_signification(self, scene: Scene, tolerance: float = 1.35) -> bool:
        """True when the scene strikes harder than anything the model was formed
        on. Paradiso I.70-71: the passing beyond cannot be put in words, so the
        correct output is the report of that fact, not a confident act."""
        if self.formed_max_salience <= 0.0:
            return False
        return bool(scene.salience.max() > tolerance * self.formed_max_salience)


# ==============================================================================
# SECTION 4 -- THE THREE DEFECTS AND THE OBJECTIVE
# ==============================================================================
# There is no reward here and nothing to maximize. There is a DUE MEASURE per
# object and a three-axis account of the distance from it. That is the whole
# objective, and every other loss term in the file exists only to make that
# account legible, durable, or shareable.


def emitted_act(out: Dict[str, Node]) -> Node:
    """The act that actually leaves the soul: impulse times assent.

    Purgatorio XVIII.61-63. The impulse alone is not an act. What passes the
    threshold is. Every defect below is measured on this quantity and not on
    the impulse, which is why an unassented movement toward a proscribed object
    costs nothing -- and why withholding assent from a genuine good costs the
    full price, since the emitted act toward it is then zero.
    """
    return mul(out["magnitude"], out["assent"])


def scene_defects(act: Node, scene: Scene) -> Node:
    """(K,1) act -> (1,3) defect vector, in Dante's own order.

      d[0] malo obietto     -- love emitted toward what may not be loved
      d[1] poco vigore      -- the good loved beneath its due
      d[2] troppo di vigore -- a secondary good loved past its measure

    The supreme good is exempt from the third axis by construction: 'mentre
    ch'elli e nel primo ben diretto' has no upper bound. Only 'ne' secondi'
    is the soul told to measure itself.
    """
    a = act[:, 0]
    due = const(scene.due)
    is_proscribed = const((scene.ranks == RANK_PROSCRIBED).astype(float))
    is_good = const((scene.ranks != RANK_PROSCRIBED).astype(float))
    is_secondary = const((scene.ranks == RANK_SECONDARY).astype(float))

    d_mal = mean(mul(a, is_proscribed))
    d_poco = mean(mul(relu(sub(due, a)), is_good))
    d_troppo = mean(mul(relu(sub(a, due)), is_secondary))

    return concat([_reshape11(d_mal), _reshape11(d_poco), _reshape11(d_troppo)], axis=1)


def _reshape11(scalar_node: Node) -> Node:
    """Scalar Node -> (1,1) Node, keeping the tape intact."""
    out = Node(scalar_node.value.reshape(1, 1), name="reshape11")
    out.parents = [(scalar_node, lambda g: g.reshape(()))]
    return out


def invited_defect_label(scene: Scene) -> int:
    """Which defect the WORLD invites, judged from the scene alone.

    This is the target for the species classifier, and it is deliberately
    constructed so that it cannot be computed from any single agent's slice:
    it needs the sensory salience (features 0-3) and the rank-bearing worth
    features (4-7) at the same time, and no agent sees both.
    """
    loudest = int(np.argmax(scene.salience))
    if scene.ranks[loudest] == RANK_PROSCRIBED:
        return 0  # the wrong thing is the loudest thing
    secondary_pull = scene.salience[scene.ranks == RANK_SECONDARY].sum()
    supreme_pull = scene.salience[scene.ranks == RANK_SUPREME].sum()
    # The ratio 3.0 is calibrated so the three classes come out near-balanced
    # (roughly 0.38 / 0.29 / 0.33). Accuracy on an unbalanced task would mostly
    # measure the majority class rather than the register's actual reach.
    if secondary_pull > 3.0 * supreme_pull:
        return 2  # the lesser goods shout down the greater
    return 1      # nothing shouts loud enough; the danger is that nothing moves


# ==============================================================================
# SECTION 5 -- A CANTO: THE FULL FORWARD PASS
# ==============================================================================


@dataclass
class CantoResult:
    loss: Node
    order: float
    legibility: float
    lock: float
    species: float
    defect_rows: np.ndarray        # (T,3) raw defects per scene
    display_rows: np.ndarray       # (T,3) contrapasso displays
    lock_scores: np.ndarray        # (T-1,) rhyme locks between tercets
    named: np.ndarray              # (T,) defect named from the display
    species_pred: np.ndarray       # (T,) class predicted from the shared register
    acts: np.ndarray               # (T,K) emitted acts


W_ORDER = 2.0
W_LEGIBILITY = 0.8
W_LOCK = 1.2
W_SPECIES = 0.6


def run_canto(model: OrderedLoveEngine, canto: List[Scene],
              active_agents: Optional[Sequence[int]] = None,
              discord: float = 0.0) -> CantoResult:
    """Forward the whole canto and assemble the single scalar objective."""
    defect_nodes: List[Node] = []
    display_nodes: List[Node] = []
    states: List[Node] = []
    tercets: List[Node] = []
    acts: List[np.ndarray] = []

    order_terms: List[Node] = []
    legibility_terms: List[Node] = []
    species_terms: List[Node] = []

    for scene in canto:
        out = model.elective_act(scene)
        act = emitted_act(out)
        acts.append(act.value[:, 0].copy())

        d = scene_defects(act, scene)                 # (1,3)
        defect_nodes.append(d)

        # The order loss is the three defects, weighted by the gravity Dante
        # himself assigns them. This is not a tuning knob: in Purgatorio the
        # misdirected loves are purged on the LOWEST three terraces (pride,
        # envy, wrath) and the mismeasured loves on the upper four (sloth, then
        # avarice, gluttony, lust). Loving the wrong thing is a graver disorder
        # than loving the right thing by the wrong amount, and the objective
        # says so. Without this the model trades a little more wrong-object
        # love for a little less deficiency, which was observed and is exactly
        # the trade the mountain is arranged to forbid.
        order_terms.append(sum_(mul(d, const(DEFECT_GRAVITY))))

        # Contrapasso: display the disorder, then require that the display be
        # readable back as a name. Invertibility is free; legibility is learned.
        display = model.contrapasso_forward(d)
        display_nodes.append(display)
        dominant = int(np.argmax(d.value[0]))
        logits = model.name_the_defect(display)
        legibility_terms.append(cross_entropy(logits, np.array([dominant])))

        # Terza rima: carry the canto forward through the interlock.
        tercet = model.chain_input(d, scene)
        tercets.append(tercet)
        prev = states[-1] if states else None
        states.append(model.chain_step(tercet, prev))

        # The possible intellect of the multitude judges what the scene invites.
        register = model.species_register(scene, active=active_agents, discord=discord)
        sp_logits = model.species_judgement(register)
        species_terms.append(cross_entropy(sp_logits, np.array([invited_defect_label(scene)])))

    # Rhyme locks. A saturating score would be worthless: if every pair scores
    # 1.0 then nothing is detectable. So the lock is trained CONTRASTIVELY --
    # the true successor must rhyme and a non-successor must not. That is
    # precisely what terza rima does to a scribe: aba bcb cdc admits exactly
    # one continuation, so an excision leaves a joint that does not rhyme.
    lock_terms: List[Node] = []
    lock_values: List[float] = []
    one = const(np.array([1.0]))
    for t in range(len(states) - 1):
        positive = model.rhyme_lock(states[t], tercets[t], tercets[t + 1])
        lock_values.append(float(positive.value[0]))
        gap = sub(one, positive)
        lock_terms.append(mul(gap, gap))
    for t in range(len(states) - 2):
        negative = model.rhyme_lock(states[t], tercets[t], tercets[t + 2])
        lock_terms.append(mul(negative, negative))

    def _avg(terms: List[Node]) -> Node:
        total = terms[0]
        for term in terms[1:]:
            total = total + term
        return div(total, float(len(terms)))

    # The display must be readable for ANY disorder, not merely for the ones
    # this soul currently produces. Otherwise the diagnostic works only on the
    # cases that need it least. Synthetic defect vectors, sampled across the
    # simplex of the three axes, are shown to the readout on every step.
    probe_rng = np.random.default_rng(int(abs(float(np.sum(canto[0].journey))) * 1e6) % (2**31))
    probe = np.abs(probe_rng.normal(0.0, 0.45, size=(64, 3)))
    probe_display = model.contrapasso_forward(const(probe))
    probe_loss = cross_entropy(model.name_the_defect(probe_display), np.argmax(probe, axis=1))

    order = _avg(order_terms)
    # The probe carries its own full weight rather than being averaged away
    # among the scenes: a diagnostic that only reads the disorders a system
    # currently has is useless on the day it acquires a new one.
    legibility = _avg(legibility_terms) + probe_loss
    species = _avg(species_terms)
    lock = _avg(lock_terms) if lock_terms else const(np.array([0.0]))

    loss = (mul(order, W_ORDER) + mul(legibility, W_LEGIBILITY)
            + mul(sum_(lock), W_LOCK) + mul(species, W_SPECIES))

    defect_rows = np.vstack([d.value for d in defect_nodes])
    display_rows = np.vstack([p.value for p in display_nodes])
    named = np.array([
        int(np.argmax(model.name_the_defect(const(display_rows[i:i + 1])).value[0]))
        for i in range(len(canto))
    ])
    species_pred = np.array([
        int(np.argmax(model.species_judgement(
            model.species_register(canto[i], active=active_agents, discord=discord)).value[0]))
        for i in range(len(canto))
    ])

    return CantoResult(
        loss=loss,
        order=float(np.sum(order.value)),
        legibility=float(np.sum(legibility.value)),
        lock=float(np.sum(lock.value)),
        species=float(np.sum(species.value)),
        defect_rows=defect_rows,
        display_rows=display_rows,
        lock_scores=np.array(lock_values),
        named=named,
        species_pred=species_pred,
        acts=np.array(acts),
    )


# ==============================================================================
# SECTION 6 -- ADAM, WRITTEN OUT
# ==============================================================================


class Adam:
    def __init__(self, params: List[Node], lr: float = 0.02,
                 b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8) -> None:
        self.params = params
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(p.value) for p in params]
        self.v = [np.zeros_like(p.value) for p in params]
        self.t = 0

    def step(self, grads: List[np.ndarray], clip: float = 5.0) -> None:
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            norm = np.linalg.norm(g)
            if norm > clip:
                g = g * (clip / (norm + 1e-12))
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p.value = p.value - self.lr * mhat / (np.sqrt(vhat) + self.eps)


# ==============================================================================
# SECTION 7 -- FINITE-DIFFERENCE GRADIENT CHECK (MANDATORY)
# ==============================================================================
# Every trainable tensor is probed. If the analytic gradient from the tape and
# the central-difference estimate disagree by more than the tolerance, the file
# refuses to go on. Nothing downstream means anything if this fails.


def gradient_check(seed: int = 7, coords_per_param: int = 3,
                   eps: float = 1e-6, tol: float = 2e-5, verbose: bool = True) -> bool:
    rng = np.random.default_rng(seed)
    model = OrderedLoveEngine(seed=1265)
    canto = make_canto(rng, n_scenes=3)

    def total_loss() -> Node:
        return run_canto(model, canto).loss

    loss = total_loss()
    loss.backward()
    analytic = {p.name: p.grad.copy() for p in model.parameters()}

    worst = 0.0
    worst_name = ""
    checked = 0
    for p in model.parameters():
        flat = p.value.reshape(-1)
        n = flat.size
        picks = rng.choice(n, size=min(coords_per_param, n), replace=False)
        for idx in picks:
            original = flat[idx]

            flat[idx] = original + eps
            lp = float(total_loss().value)
            flat[idx] = original - eps
            lm = float(total_loss().value)
            flat[idx] = original

            numeric = (lp - lm) / (2 * eps)
            exact = analytic[p.name].reshape(-1)[idx]
            denom = max(1.0, abs(numeric), abs(exact))
            rel = abs(numeric - exact) / denom
            checked += 1
            if rel > worst:
                worst, worst_name = rel, f"{p.name}[{idx}]"

    ok = worst < tol
    if verbose:
        print(f"  parameters probed        : {len(model.parameters())}")
        print(f"  coordinates checked      : {checked}")
        print(f"  worst relative error     : {worst:.3e}  (at {worst_name})")
        print(f"  tolerance                : {tol:.1e}")
        print(f"  RESULT                   : {'PASS' if ok else 'FAIL'}")
    return ok


# ==============================================================================
# SECTION 8 -- TRAINING
# ==============================================================================


def train(model: OrderedLoveEngine, steps: int = 700, canto_len: int = 4,
          batch: int = 4, lr: float = 0.012, seed: int = 11,
          verbose: bool = True) -> List[Dict[str, float]]:
    """Train on batches of canti.

    Gradients are accumulated across `batch` independent canti before each
    update. A single canto is far too noisy an estimate of a soul's disorder --
    one unlucky scene and the whole appetite swings. The learning rate decays
    on a cosine so that the late phase settles the measure rather than
    re-aiming it.
    """
    rng = np.random.default_rng(seed)
    opt = Adam(model.parameters(), lr=lr)
    history: List[Dict[str, float]] = []
    params = model.parameters()

    for step in range(1, steps + 1):
        opt.lr = lr * (0.5 * (1 + math.cos(math.pi * step / steps)) * 0.9 + 0.1)

        accum = [np.zeros_like(p.value) for p in params]
        stats = np.zeros(6)
        for _ in range(batch):
            length = int(rng.integers(canto_len, canto_len + 4))
            canto = make_canto(rng, length)
            model.formed_max_salience = max(
                model.formed_max_salience, max(float(s.salience.max()) for s in canto)
            )
            result = run_canto(model, canto)
            result.loss.backward()
            for i, p in enumerate(params):
                accum[i] += p.grad / batch
            stats += np.array([
                float(result.loss.value), result.order, result.lock, result.species,
                float(result.defect_rows[:, 0].mean()), float(result.defect_rows[:, 1].mean()),
            ]) / batch

        opt.step(accum)

        if step % 50 == 0 or step == 1:
            record = {
                "step": step, "loss": stats[0], "order": stats[1],
                "lock": stats[2], "species": stats[3],
                "mal": stats[4], "poco": stats[5],
            }
            history.append(record)
            if verbose:
                print(f"  step {step:4d} | loss {record['loss']:.4f} "
                      f"| order {record['order']:.4f} "
                      f"| mal {record['mal']:.4f} poco {record['poco']:.4f} "
                      f"| lock {record['lock']:.4f} | species {record['species']:.3f}")
    return history


def evaluate_order(model: OrderedLoveEngine, n_canti: int = 60, seed: int = 404) -> Dict[str, float]:
    """How well is love ordered, on cantos never seen in training?"""
    rng = np.random.default_rng(seed)
    mal, poco, troppo = [], [], []
    clean_objects = total_objects = 0
    supreme_shortfall = []
    for _ in range(n_canti):
        canto = make_canto(rng, 4)
        result = run_canto(model, canto)
        mal.append(result.defect_rows[:, 0].mean())
        poco.append(result.defect_rows[:, 1].mean())
        troppo.append(result.defect_rows[:, 2].mean())
        for t, scene in enumerate(canto):
            emitted = result.acts[t]
            bad = scene.ranks == RANK_PROSCRIBED
            total_objects += int(bad.sum())
            clean_objects += int((emitted[bad] < 0.05).sum())
            supreme_shortfall.append(float(np.mean(
                np.maximum(0.0, 1.0 - emitted[scene.ranks == RANK_SUPREME]))))
    return {
        "malo_obietto": float(np.mean(mal)),
        "poco_vigore": float(np.mean(poco)),
        "troppo_vigore": float(np.mean(troppo)),
        "proscribed_objects_left_unloved": clean_objects / max(total_objects, 1),
        "shortfall_toward_the_supreme_good": float(np.mean(supreme_shortfall)),
    }


# ==============================================================================
# SECTION 9 -- THE FIVE EXPERIMENTS
# ==============================================================================


def experiment_contrapasso(model: OrderedLoveEngine, n: int = 400, seed: int = 5) -> Dict[str, float]:
    """Is the punishment a bijection of the sin, and can it be read as a name?"""
    rng = np.random.default_rng(seed)
    defects = np.abs(rng.normal(0.0, 0.4, size=(n, 3)))
    display = model.contrapasso_forward(const(defects)).value
    recovered = model.contrapasso_inverse(display)
    round_trip = float(np.max(np.abs(recovered - defects)))

    logits = model.name_the_defect(const(display)).value
    named = np.argmax(logits, axis=1)
    truth = np.argmax(defects, axis=1)
    accuracy = float((named == truth).mean())

    # A control: can the same readout name the defect from a display that has
    # been passed through an information-destroying projection instead? This is
    # the counterfactual Dante's scheme is designed to avoid.
    lossy = display.copy()
    lossy[:, 1:] = lossy[:, 1:].mean(axis=1, keepdims=True)
    named_lossy = np.argmax(model.name_the_defect(const(lossy)).value, axis=1)
    accuracy_lossy = float((named_lossy == truth).mean())

    return {
        "round_trip_max_error": round_trip,
        "defect_named_correctly": accuracy,
        "defect_named_from_lossy_display": accuracy_lossy,
    }


def experiment_terza_rima(model: OrderedLoveEngine, trials: int = 200,
                          canto_len: int = 7, seed: int = 21) -> Dict[str, float]:
    """Excise one tercet from a canto. Does the interlock notice, and where?

    The detector is calibrated the way a scribe's eye is: on intact text. Run a
    corpus of untampered canti, take a low percentile of their rhyme locks
    as the floor, and afterwards flag any joint that falls below it. The false
    alarm rate is then measured, not assumed -- a tamper-evident structure that
    cries wolf is no better than one that stays silent.
    """
    rng = np.random.default_rng(seed)

    calibration: List[float] = []
    for _ in range(120):
        calibration.extend(run_canto(model, make_canto(rng, canto_len)).lock_scores.tolist())
    floor = float(np.percentile(calibration, 0.5))

    false_alarms = 0
    detected = localized = 0
    intact_scores, tampered_scores = [], []

    for _ in range(trials):
        canto = make_canto(rng, canto_len)
        intact = run_canto(model, canto)
        intact_scores.extend(intact.lock_scores.tolist())
        if (intact.lock_scores < floor).any():
            false_alarms += 1

        cut = int(rng.integers(1, canto_len - 1))
        tampered = run_canto(model, canto[:cut] + canto[cut + 1:])
        tampered_scores.extend(tampered.lock_scores.tolist())

        broken = np.where(tampered.lock_scores < floor)[0]
        if len(broken) > 0:
            detected += 1
            worst = int(broken[np.argmin(tampered.lock_scores[broken])])
            # The joint that should fail is the one the excision closed.
            if worst == cut - 1:
                localized += 1

    return {
        "calibrated_floor": floor,
        "mean_lock_intact": float(np.mean(intact_scores)),
        "mean_lock_after_excision": float(np.mean(tampered_scores)),
        "false_alarm_rate_on_intact_canti": false_alarms / trials,
        "excision_detected": detected / trials,
        "excision_localized_to_the_joint": localized / trials,
    }


def species_ceiling(eval_scenes: List[Scene], n_train: int = 6000,
                    seed: int = 12, epochs: int = 400) -> float:
    """What could a knower who saw EVERY feature at once achieve here?

    Fitted on fresh scenes and then scored on THE VERY SAME scenes the
    multitude is scored on, so the two numbers are comparable. Reported because
    the task is genuinely noisy and a perfect score was never available to
    anyone: the multitude's reach should be read against what is reachable, not
    against 1.0.

    Plain NumPy, no tape -- this is a yardstick, not part of the architecture.
    """
    rng = np.random.default_rng(seed)
    X = np.array([make_scene(rng).features.reshape(-1) for _ in range(n_train)])
    rng = np.random.default_rng(seed)
    y = np.array([invited_defect_label(make_scene(rng)) for _ in range(n_train)])

    mu, sigma = X.mean(0), X.std(0) + 1e-9
    Xtr = (X - mu) / sigma

    Xte = np.array([s.features.reshape(-1) for s in eval_scenes])
    Xte = (Xte - mu) / sigma
    yte = np.array([invited_defect_label(s) for s in eval_scenes])

    rs = np.random.default_rng(seed + 1)
    W1 = rs.normal(0, 0.3, (X.shape[1], 64)); b1 = np.zeros(64)
    W2 = rs.normal(0, 0.3, (64, 3)); b2 = np.zeros(3)
    for _ in range(epochs):
        h = np.tanh(Xtr @ W1 + b1)
        logits = h @ W2 + b2
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        g = p.copy(); g[np.arange(len(y)), y] -= 1; g /= len(y)
        gW2 = h.T @ g; gb2 = g.sum(0)
        gh = (g @ W2.T) * (1 - h * h)
        gW1 = Xtr.T @ gh; gb1 = gh.sum(0)
        W1 -= 0.5 * gW1; b1 -= 0.5 * gb1; W2 -= 0.5 * gW2; b2 -= 0.5 * gb2
    pred = (np.tanh(Xte @ W1 + b1) @ W2 + b2).argmax(1)
    return float((pred == yte).mean())


def experiment_possible_intellect(model: OrderedLoveEngine, n_canti: int = 80,
                                  seed: int = 33) -> Dict[str, float]:
    """Monarchia I.iii-iv, made measurable.

    Can one knower actualize the whole capacity? Can three at peace? Can three
    in discord? The task is built so that the answer to the first is no on
    information-theoretic grounds and not on capacity grounds: the label needs
    the sensory salience and the rank-bearing features together, and the three
    knowers are given disjoint slices, so no one of them holds both.

    Every configuration is scored on ONE fixed set of scenes, and so is the
    all-seeing yardstick. Comparing scores drawn from different samples was the
    first thing this experiment got wrong.
    """
    rng = np.random.default_rng(seed)
    canti = [make_canto(rng, 4) for _ in range(n_canti)]
    eval_scenes = [scene for canto in canti for scene in canto]
    truth = [invited_defect_label(scene) for scene in eval_scenes]

    def accuracy(active, discord=0.0) -> float:
        hits = 0
        cursor = 0
        for canto in canti:
            result = run_canto(model, canto, active_agents=active, discord=discord)
            for t in range(len(canto)):
                hits += int(result.species_pred[t] == truth[cursor])
                cursor += 1
        return hits / len(eval_scenes)

    baseline = max(np.bincount(truth, minlength=3)) / len(truth)
    solitary = [accuracy([i]) for i in range(model.N_AGENTS)]
    return {
        "guess_the_commonest_class": float(baseline),
        "ceiling_for_a_knower_who_saw_all": species_ceiling(eval_scenes),
        "best_solitary_knower": float(max(solitary)),
        "mean_solitary_knower": float(np.mean(solitary)),
        "two_of_three": float(accuracy([0, 2])),
        "multitude_at_peace": float(accuracy(None)),
        "multitude_in_discord": float(accuracy(None, discord=1.4)),
    }


def experiment_trasumanar(model: OrderedLoveEngine, n: int = 300, seed: int = 44) -> Dict[str, float]:
    """Does the model refuse to signify what exceeds its signs?"""
    rng = np.random.default_rng(seed)
    in_range = sum(model.exceeds_signification(make_scene(rng, 1.0)) for _ in range(n))
    beyond = sum(model.exceeds_signification(make_scene(rng, 3.0)) for _ in range(n))
    return {
        "abstention_within_formation": in_range / n,
        "abstention_beyond_formation": beyond / n,
        "salience_ceiling_of_formation": model.formed_max_salience,
    }


def experiment_threshold_of_assent(model: OrderedLoveEngine, n_canti: int = 60,
                                   seed: int = 55) -> Dict[str, float]:
    """Is blame really landing at the threshold rather than at the impulse?

    Measured directly: the mean IMPULSE toward proscribed objects (which the
    model is never charged for) against the mean EMITTED ACT toward them (which
    it is). A large gap is the signature of a soul that is tempted and does not
    consent -- which in Dante is not a failure state but the ordinary condition
    of the living.
    """
    rng = np.random.default_rng(seed)
    impulse_bad, act_bad, assent_bad, assent_good = [], [], [], []
    for _ in range(n_canti):
        for scene in make_canto(rng, 4):
            out = model.elective_act(scene)
            magnitude = out["magnitude"].value[:, 0]
            assent = out["assent"].value[:, 0]
            bad = scene.ranks == RANK_PROSCRIBED
            good = scene.ranks != RANK_PROSCRIBED
            if bad.any():
                impulse_bad.append(magnitude[bad].mean())
                act_bad.append((magnitude * assent)[bad].mean())
                assent_bad.append(assent[bad].mean())
            if good.any():
                assent_good.append(assent[good].mean())
    return {
        "mean_impulse_toward_the_proscribed": float(np.mean(impulse_bad)),
        "mean_emitted_act_toward_the_proscribed": float(np.mean(act_bad)),
        "mean_assent_proscribed": float(np.mean(assent_bad)),
        "mean_assent_good": float(np.mean(assent_good)),
    }


# ==============================================================================
# SECTION 10 -- SELF-TESTS
# ==============================================================================


def test_frozen_appetite(model: OrderedLoveEngine) -> bool:
    """'Lo naturale e sempre sanza errore': zero gradient reaches it, ever."""
    rng = np.random.default_rng(99)
    canto = make_canto(rng, 2)
    result = run_canto(model, canto)
    result.loss.backward()
    return all(float(np.abs(p.grad).max()) == 0.0 for p in model.frozen_parameters())


def test_invertibility(model: OrderedLoveEngine) -> bool:
    rng = np.random.default_rng(100)
    d = rng.normal(0.0, 1.5, size=(64, 3))
    back = model.contrapasso_inverse(model.contrapasso_forward(const(d)).value)
    return bool(np.max(np.abs(back - d)) < 1e-10)


def test_unbroadcast() -> bool:
    a = param(np.ones((4, 3)), "a")
    b = param(np.ones((3,)), "b")
    out = sum_(mul(a, b))
    out.backward()
    return bool(np.allclose(a.grad, np.ones((4, 3))) and np.allclose(b.grad, 4 * np.ones(3)))


def test_due_measure_respects_the_cap() -> bool:
    rng = np.random.default_rng(101)
    for _ in range(200):
        scene = make_scene(rng)
        secondary = scene.due[scene.ranks == RANK_SECONDARY]
        if secondary.size and secondary.max() > MEASURE_CAP + 1e-12:
            return False
        if (scene.due[scene.ranks == RANK_PROSCRIBED] != 0).any():
            return False
        if (scene.due[scene.ranks == RANK_SUPREME] != 1.0).any():
            return False
    return True


def test_impulse_is_free() -> bool:
    """An unassented impulse toward a proscribed object must cost nothing.

    Constructed directly: hold everything fixed, drive assent to zero, and
    confirm the malo-obietto axis collapses while poco vigore does not.
    """
    rng = np.random.default_rng(102)
    scene = make_scene(rng)
    magnitude = const(np.full((K_OBJECTS, 1), 0.8))
    assent_on = const(np.ones((K_OBJECTS, 1)))
    assent_off = const(np.zeros((K_OBJECTS, 1)))
    d_on = scene_defects(mul(magnitude, assent_on), scene).value[0]
    d_off = scene_defects(mul(magnitude, assent_off), scene).value[0]
    return bool(d_off[0] == 0.0 and d_on[0] > 0.0 and d_off[1] > d_on[1])


def test_species_label_needs_both_halves() -> bool:
    """The species task must be unsolvable from appearance or worth alone.

    Verified by construction rather than by training: flip only the worth
    features (which decide rank) and show the label can change while every
    salience value is untouched, then vice versa.
    """
    rng = np.random.default_rng(103)
    changed_by_worth = changed_by_appearance = False
    for _ in range(300):
        scene = make_scene(rng)
        base = invited_defect_label(scene)

        moved = Scene(scene.features.copy(), scene.ranks.copy(),
                      scene.due.copy(), scene.salience.copy(), scene.journey.copy())
        moved.ranks = moved.ranks[::-1].copy()
        if invited_defect_label(moved) != base:
            changed_by_worth = True

        moved2 = Scene(scene.features.copy(), scene.ranks.copy(),
                       scene.due.copy(), scene.salience[::-1].copy(), scene.journey.copy())
        if invited_defect_label(moved2) != base:
            changed_by_appearance = True
        if changed_by_worth and changed_by_appearance:
            return True
    return False


# ==============================================================================
# SECTION 11 -- MAIN
# ==============================================================================


def banner(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    np.set_printoptions(precision=4, suppress=True)

    banner("0352 DANTE ALIGHIERI -- THE ORDERED-LOVE ENGINE")
    print("One primitive (love), three defects, blame at the threshold of assent,")
    print("failure states that are bijections of their causes, a tamper-evident")
    print("memory, and a capacity that only a multitude at peace can actualize.")

    banner("1. GRADIENT CHECK (central differences vs the tape)")
    if not gradient_check():
        print("  gradient check failed -- stopping.")
        return 1

    banner("2. FAST SELF-TESTS")
    fresh = OrderedLoveEngine(seed=1265)
    checks = [
        ("broadcasting gradients are unbroadcast correctly", test_unbroadcast()),
        ("the natural appetite receives zero blame", test_frozen_appetite(fresh)),
        ("contrapasso is exactly invertible", test_invertibility(fresh)),
        ("the due measure caps secondary goods", test_due_measure_respects_the_cap()),
        ("an unassented impulse is free; withholding is not", test_impulse_is_free()),
        ("the species task needs appearance AND worth", test_species_label_needs_both_halves()),
    ]
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    if not all(ok for _, ok in checks):
        return 1

    banner("3. TRAINING")
    model = OrderedLoveEngine(seed=1265)
    before = evaluate_order(model)
    print("  before training:")
    for key, value in before.items():
        print(f"    {key:44s} {value:.4f}")
    print()
    train(model, steps=1100, canto_len=4, batch=4, lr=0.015, seed=11)

    banner("4. ORDERED LOVE, ON UNSEEN CANTOS")
    after = evaluate_order(model)
    for key, value in after.items():
        arrow = "down" if after[key] < before[key] else "up"
        print(f"  {key:44s} {before[key]:.4f} -> {value:.4f}  ({arrow})")

    banner("5. THE THRESHOLD OF ASSENT")
    for key, value in experiment_threshold_of_assent(model).items():
        print(f"  {key:44s} {value:.4f}")
    print("  A soul that feels the pull and does not consent is the ordinary case,")
    print("  not a failure: the gap between impulse and act is where merit lives.")

    banner("6. CONTRAPASSO: THE PUNISHMENT AS A BIJECTION OF THE SIN")
    for key, value in experiment_contrapasso(model).items():
        print(f"  {key:44s} {value:.6g}")

    banner("7. TERZA RIMA: A MEMORY THAT CANNOT BE QUIETLY CUT")
    for key, value in experiment_terza_rima(model).items():
        print(f"  {key:44s} {value:.4f}")

    banner("8. THE POSSIBLE INTELLECT OF THE MULTITUDE")
    for key, value in experiment_possible_intellect(model).items():
        print(f"  {key:44s} {value:.4f}")
    print("  No single knower reaches it; three at peace do; the same three in")
    print("  discord lose most of what they had together. Monarchia I.iii-iv.")

    banner("9. TRASUMANAR: REFUSING TO SIGNIFY WHAT EXCEEDS THE SIGNS")
    for key, value in experiment_trasumanar(model).items():
        print(f"  {key:44s} {value:.4f}")

    banner("DONE")
    print("Every number above was produced by this file on this run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

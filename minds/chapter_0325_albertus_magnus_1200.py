#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 Chapter 325 — Albertus Magnus (c. 1200 Lauingen – 15 November 1280 Cologne)
 # Encyclopedia of Lost Minds: Echoes on AI · Chapter 0325 · Albertus Magnus (1200-1280)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 17, Minds 321-340 Available on Amazon https://www.amazon.com/dp/B0HP7Z1NKC
# Resume and Interactive Demos at https://artificiology.com/
 THE DENUDATIO COLUMN: an abstraction ladder terminated by a privately owned light
================================================================================

WHAT THIS FILE IS
-----------------
A complete, executable, trained-from-scratch cognitive architecture written in
pure NumPy (no autograd framework, no deep-learning library). It is not a demo
of a famous architecture wearing a medieval costume. Every mechanism in it is
derived from one specific, documented commitment of Albert the Great, and the
file contains the experiment that would settle the argument he actually had.

THE ONE IDEA THAT IS HIS ALONE
------------------------------
Albert did not think intelligence was a faculty. He thought it was a *column* —
a fixed succession of organs, each seated in a named cavity of the brain, each
of which strips away one more layer of particularity from what the senses
deliver, until what remains is thin enough for an intellect to *become*.

    external senses  ->  sensus communis  ->  imaginatio
                     ->  aestimativa / cogitativa  ->  memoria
                     ->  [intellectus agens: the light]
                     ->  intellectus possibilis  ->  habitus / intellectus adeptus

Two doctrines make this column his rather than Aristotle's or Avicenna's:

  (1) THE STRIPPING IS MONOTONE AND MEASURABLE. Albert explicitly orders the
      inner senses "according to the levels of abstraction" (secundum gradus
      abstractionis), tying each level to the subtlety of the animal spirits in
      its ventricle. That is a claim with a number in it: materiality must fall,
      rung by rung, and never rise. This file measures that number at five rungs
      and trains against any rung that lets it rise.

  (2) THE LIGHT IS OWNED, NOT SHARED. Against Averroes' single intellect for the
      whole species, Albert wrote De unitate intellectus contra Averroistas
      (c. 1256). His position: the agent intellect is a power of *this* soul,
      seated in *this* body, and the vegetative, sensitive and rational are all
      numbered to the number of the men in whom they are. Against Avicenna he
      refused to put the abstracting light outside the individual at all.

Doctrine (2) is not a footnote here. It is a testable architectural claim, and
this file tests it. Each knower's body imposes its own *complexio* — Albert's
term for the individual temperament of the organs, which he holds to modulate
the inner senses — as a fixed positive scaling of that knower's phantasms. Two
models are then trained on identical data:

    AVERROIST  : one agent intellect, shared by every knower.
    ALBERTIST  : one agent intellect per knower, seated in that knower's body.

Everything else is shared between them: the same senses, the same imagination,
the same possible intellect, the same parameter budget elsewhere. The result is
printed at the bottom of the run. It is the argument of 1256, re-run as an
ablation.

THE SECOND MECHANISM: TWO REGIMES OF CERTAINTY
----------------------------------------------
Albert is not a blanket empiricist. His famous line is conditional:
"experimentum solum certificat in talibus" — experiment alone gives certainty
*in such matters* (De vegetabilibus VI, tr. 2, c. 1). The clause matters. Where
causes are manifest, demonstration reaches the conclusion for free. Where causes
are occult, no amount of reasoning substitutes for going and looking, and he
went and looked: he offered iron to several ostriches, and reported that they
would not eat it, against Pliny and against every bestiary he had read.

So the architecture carries a REGIMEN: a learned router that decides, per case,
whether to answer by demonstratio (free, but only correct where the intention
follows from the form) or to spend a unit of a finite budget on experimentum.
The run reports how the budget is spent on manifest versus occult kinds. Nobody
tells the router which kinds are which.

THE THIRD MECHANISM: INTENTIONES NON SENSATAE
----------------------------------------------
The estimative power grasps what the senses never delivered — the sheep does not
see hostility, it has no colour. Albert keeps the Avicennian estimativa but
holds that in humans it operates under reason. This file therefore builds two
variants of the same organ: an ESTIMATIVA with bottom-up input only, and a
COGITATIVA that additionally receives top-down projection from the intellect's
universal. Both are trained; both are reported.

MEMORY IS SPLIT, AS HE SPLIT IT
-------------------------------
Imagination stores *forms*; memory stores *intentions*, time-indexed. They are
two different stores in two different chambers, and this file keeps them apart.
Retrieval from memory is not similarity lookup: Albert follows the distinction
between memoria (immediate, shared with animals) and reminiscentia (a directed
search, proper to humans). Reminiscence here is a beam search along the temporal
chain, and it is demonstrated at the end of the run.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
No transformer, no attention over stored keys, no mixture of experts, no
embedding lookup table. Attention-over-keys is the wrong shape for this mind:
it retrieves a stored particular, and Albert's whole column exists to destroy
particulars. Nothing is retrieved here. Something is *becomes*.

FILES IN THIS CHAPTER
---------------------
  chapter_325_albertus_magnus_1200.py   (this file)
  325_albertus_magnus_1200_MindMap.html (the interactive 3D mind-model)

RUN
---
  python3 chapter_325_albertus_magnus_1200.py

Requires only numpy. Runs to completion in well under a minute on a laptop CPU.
A finite-difference gradient check runs before training and must pass.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ==============================================================================
# PART 0 — A MINIMAL REVERSE-MODE AUTODIFF ENGINE
# ==============================================================================
# Written from scratch so that the unusual operations this architecture needs
# (grouped within-kind variance, multiplicative illumination, a differentiable
# cost router) can be composed freely and still be verified against finite
# differences. Roughly 200 lines. Nothing below this section imports anything
# other than numpy.
# ==============================================================================


def _unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Reduce `grad` back to `shape`, undoing NumPy broadcasting.

    If a forward op broadcast a (1, D) parameter against an (N, D) batch, the
    incoming gradient is (N, D) and must be summed back down to (1, D).
    """
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    """A node in the computation graph: a value, a gradient, and a local rule."""

    __slots__ = ("data", "grad", "_backward", "_parents", "_op", "requires_grad")

    def __init__(self, data, parents: Sequence["Tensor"] = (), op: str = "",
                 requires_grad: bool = True):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward: Callable[[], None] = lambda: None
        self._parents: Tuple["Tensor", ...] = tuple(parents)
        self._op = op
        self.requires_grad = requires_grad

    # ---- structural helpers ------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, op='{self._op}')"

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    # ---- elementwise arithmetic -------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _bw():
            self.grad = self.grad + _unbroadcast(out.grad, self.data.shape)
            other.grad = other.grad + _unbroadcast(out.grad, other.data.shape)

        out._backward = _bw
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _bw():
            self.grad = self.grad + _unbroadcast(out.grad * other.data, self.data.shape)
            other.grad = other.grad + _unbroadcast(out.grad * self.data, other.data.shape)

        out._backward = _bw
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data / other.data, (self, other), "/")

        def _bw():
            self.grad = self.grad + _unbroadcast(out.grad / other.data, self.data.shape)
            other.grad = other.grad + _unbroadcast(
                -out.grad * self.data / (other.data ** 2), other.data.shape)

        out._backward = _bw
        return out

    def __pow__(self, p: float):
        out = Tensor(self.data ** p, (self,), f"**{p}")

        def _bw():
            self.grad = self.grad + out.grad * p * (self.data ** (p - 1.0))

        out._backward = _bw
        return out

    # ---- linear algebra ----------------------------------------------------
    def matmul(self, other: "Tensor") -> "Tensor":
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _bw():
            self.grad = self.grad + out.grad @ other.data.T
            other.grad = other.grad + self.data.T @ out.grad

        out._backward = _bw
        return out

    __matmul__ = matmul

    def transpose(self) -> "Tensor":
        out = Tensor(self.data.T, (self,), "T")

        def _bw():
            self.grad = self.grad + out.grad.T

        out._backward = _bw
        return out

    # ---- reductions --------------------------------------------------------
    def sum(self, axis=None, keepdims=False) -> "Tensor":
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _bw():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.grad = self.grad + np.broadcast_to(g, self.data.shape).copy()

        out._backward = _bw
        return out

    def mean(self, axis=None, keepdims=False) -> "Tensor":
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # ---- nonlinearities ----------------------------------------------------
    def tanh(self) -> "Tensor":
        t = np.tanh(self.data)
        out = Tensor(t, (self,), "tanh")

        def _bw():
            self.grad = self.grad + out.grad * (1.0 - t * t)

        out._backward = _bw
        return out

    def sigmoid(self) -> "Tensor":
        s = 1.0 / (1.0 + np.exp(-np.clip(self.data, -60.0, 60.0)))
        out = Tensor(s, (self,), "sigmoid")

        def _bw():
            self.grad = self.grad + out.grad * s * (1.0 - s)

        out._backward = _bw
        return out

    def softplus(self) -> "Tensor":
        x = np.clip(self.data, -60.0, 60.0)
        v = np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)
        out = Tensor(v, (self,), "softplus")

        def _bw():
            self.grad = self.grad + out.grad / (1.0 + np.exp(-x))

        out._backward = _bw
        return out

    def relu(self) -> "Tensor":
        m = (self.data > 0.0).astype(np.float64)
        out = Tensor(self.data * m, (self,), "relu")

        def _bw():
            self.grad = self.grad + out.grad * m

        out._backward = _bw
        return out

    def log(self) -> "Tensor":
        out = Tensor(np.log(np.maximum(self.data, 1e-300)), (self,), "log")

        def _bw():
            self.grad = self.grad + out.grad / np.maximum(self.data, 1e-300)

        out._backward = _bw
        return out

    # ---- shape ops ---------------------------------------------------------
    def concat_with(self, other: "Tensor", axis: int = 1) -> "Tensor":
        out = Tensor(np.concatenate([self.data, other.data], axis=axis),
                     (self, other), "concat")
        n = self.data.shape[axis]

        def _bw():
            a, b = np.split(out.grad, [n], axis=axis)
            self.grad = self.grad + a
            other.grad = other.grad + b

        out._backward = _bw
        return out

    def cols(self, start: int, stop: int) -> "Tensor":
        out = Tensor(self.data[:, start:stop], (self,), "cols")

        def _bw():
            g = np.zeros_like(self.data)
            g[:, start:stop] = out.grad
            self.grad = self.grad + g

        out._backward = _bw
        return out

    # ---- backward pass -----------------------------------------------------
    def backward(self):
        """Topologically sort the graph, then apply local rules in reverse."""
        topo: List[Tensor] = []
        seen = set()

        def build(v: "Tensor"):
            if id(v) in seen:
                return
            seen.add(id(v))
            for p in v._parents:
                build(p)
            topo.append(v)

        sys.setrecursionlimit(100000)
        build(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()


# ---- primitives with hand-written stable gradients ---------------------------

def log_softmax(x: Tensor, axis: int = 1) -> Tensor:
    """Numerically stable log-softmax. The max shift is a constant, so treating
    it as non-differentiable is exact, not an approximation."""
    shift = x.data.max(axis=axis, keepdims=True)
    z = x.data - shift
    lse = np.log(np.exp(z).sum(axis=axis, keepdims=True))
    val = z - lse
    out = Tensor(val, (x,), "log_softmax")
    p = np.exp(val)

    def _bw():
        x.grad = x.grad + (out.grad - p * out.grad.sum(axis=axis, keepdims=True))

    out._backward = _bw
    return out


def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Mean categorical cross-entropy over a batch of integer targets."""
    ls = log_softmax(logits, axis=1)
    n = logits.data.shape[0]
    picked = ls.data[np.arange(n), targets]
    out = Tensor(-picked.mean(), (ls,), "cross_entropy")

    def _bw():
        g = np.zeros_like(ls.data)
        g[np.arange(n), targets] = -1.0 / n
        ls.grad = ls.grad + g * out.grad

    out._backward = _bw
    return out


def bce_with_logits_per_sample(logits: Tensor, y: np.ndarray) -> Tensor:
    """Per-sample binary cross-entropy, shape (N, 1). Kept per-sample because the
    regimen router must weigh each case's demonstrative loss against the cost of
    an experiment for that case."""
    z = logits.data
    yv = y.reshape(-1, 1).astype(np.float64)
    val = np.maximum(z, 0.0) - z * yv + np.log1p(np.exp(-np.abs(z)))
    out = Tensor(val, (logits,), "bce")
    s = 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))

    def _bw():
        logits.grad = logits.grad + out.grad * (s - yv)

    out._backward = _bw
    return out


# ==============================================================================
# PART I — THE WORLD OF PARTICULARS (the phantasm generator)
# ==============================================================================
# Albert's column has nothing to work on unless the world it faces has the shape
# his psychology assumes: singulars that instantiate kinds, delivered through
# several senses that can disagree, carrying intentions the senses never sensed.
#
# So the generator below produces exactly that world and nothing more convenient:
#
#   * K kinds, each with a hidden universal form f_k in R^F. Nothing observes f_k.
#   * Every encounter particularises the form: magnitude, tint, position, noise.
#   * THREE senses (visus, tactus, auditus) each see a different projection.
#     With probability p_discord the auditory channel is filled from a DIFFERENT
#     kind — a discordant sensible. The common sense must notice and discount it,
#     or every rung above it inherits the error.
#   * Each kind carries an intention (useful / harmful) that no sense reports.
#       - MANIFEST kinds: intention follows from the form by a fixed rule.
#         Demonstration reaches it. ("causes manifest")
#       - OCCULT kinds: intention is assigned against the rule, so no function of
#         the form can recover it. Only going and looking settles it.
#         ("experimentum solum certificat in talibus")
# ==============================================================================

H_BOUND = 16       # width of the bound percept leaving the common sense
H_IMAG = 16        # width of the retained phantasm
H_EST = 6          # the estimative organ is deliberately narrow and shallow:
                   # in brutes it reads 'ad instinctum naturae', it does not infer
H_INTEL = 8        # width of the intellect (deliberately the narrowest rung)
S_SEED = 5         # width of the seed from which a knower's own light is struck

F_DIM = 6          # dimension of the universal form
D_VISUS = 12       # sight channel width
D_TACTUS = 10      # touch channel width
D_AUDITUS = 8      # hearing channel width
D_SENSE = D_VISUS + D_TACTUS + D_AUDITUS
N_KINDS = 8


@dataclass
class Batch:
    """One draw of particulars, with everything the column may and may not see."""
    visus: np.ndarray        # (N, D_VISUS)
    tactus: np.ndarray       # (N, D_TACTUS)
    auditus: np.ndarray      # (N, D_AUDITUS)
    kind: np.ndarray         # (N,)   int   — supervision for assimilation
    form: np.ndarray         # (N, F) float — the universal the intellect must become
    intention: np.ndarray    # (N,)   int   — the unsensed intention (0/1)
    manifest: np.ndarray     # (N,)   bool  — is this kind's cause manifest?
    discord: np.ndarray      # (N,)   bool  — was the auditory channel discordant?
    knower: np.ndarray       # (N,)   int   — which knower's body received it


class WorldOfParticulars:
    """Generates singulars. Holds the hidden universals it never reveals."""

    def __init__(self, seed: int = 1200, n_kinds: int = N_KINDS,
                 n_knowers: int = 4, p_discord: float = 0.35):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.n_kinds = n_kinds
        self.n_knowers = n_knowers
        self.p_discord = p_discord

        # THE SHEEP AND THE WOLF.
        # The universals come in near-identical PAIRS: within a pair the two
        # forms differ by a few percent, so no sense can reliably tell them
        # apart once magnitude, tint and place have been laid over them — yet
        # their intentions are OPPOSITE. This is Albert's own example sharpened:
        # what the sheep must read off the wolf is not a colour or a shape, and
        # the wolf looks very like the dog. A column that has not stripped the
        # accidents down to the bare universal cannot read the intention at all.
        assert n_kinds % 2 == 0, "kinds are generated in near-identical pairs"
        base = rng.normal(0.0, 1.0, size=(n_kinds // 2, F_DIM))
        base /= np.linalg.norm(base, axis=1, keepdims=True)
        pairs = []
        for row in base:
            twin = row + rng.normal(0.0, 0.14, size=F_DIM)
            twin /= np.linalg.norm(twin)
            pairs.append(row)
            pairs.append(twin)
        self.forms = np.stack(pairs) * 1.6

        # How each sense receives a form. Fixed: the senses do not learn.
        self.P_visus = rng.normal(0.0, 0.8, size=(F_DIM, D_VISUS))
        self.P_tactus = rng.normal(0.0, 0.8, size=(F_DIM, D_TACTUS))
        self.P_auditus = rng.normal(0.0, 0.8, size=(F_DIM, D_AUDITUS))

        # Within every pair one member is useful and the other harmful. The
        # distinction is therefore carried entirely by the tiny difference
        # between the twin forms, and by nothing else whatsoever.
        derivable = np.zeros(n_kinds, dtype=int)
        derivable[1::2] = 1
        self.rule = rng.normal(0.0, 1.0, size=F_DIM)   # kept for reference

        # Whole pairs are manifest or occult together.
        n_pairs = n_kinds // 2
        manifest_pair = np.zeros(n_pairs, dtype=bool)
        manifest_pair[: n_pairs // 2 + n_pairs % 2] = True
        rng.shuffle(manifest_pair)
        self.manifest_kind = np.repeat(manifest_pair, 2)

        # For manifest kinds the intention IS the rule applied to the form, and
        # therefore reachable by demonstration alone. For occult kinds the
        # intention is drawn afresh for each individual and is carried by no
        # channel the column can reach — exactly the case Albert reserves for
        # experiment. No internal cleverness can recover it; the world must be
        # asked. This is the difference between knowing that fire burns and
        # knowing whether *this* stone is the one that draws iron.
        self.intention_kind = derivable.copy()

        # COMPLEXIO — the individuating body.
        # Albert grounds the inner senses in the subtlety of the spirits filling
        # each ventricle, and that subtlety differs from man to man. So each
        # knower's body imposes a fixed positive gain on that knower's PHANTASMS
        # — not on the eye, on the chamber where the form is held. The spread is
        # wide: a channel almost extinguished in one body is dominant in another.
        # It is not a parameter. It is the given, the body one has.
        self.complexio = np.exp(rng.normal(0.0, 1.35, size=(n_knowers, H_IMAG)))

    def draw(self, n: int, knower: Optional[int] = None) -> Batch:
        rng = self.rng
        kind = rng.integers(0, self.n_kinds, size=n)
        form = self.forms[kind]

        # --- particularising accidents ------------------------------------
        magnitude = np.exp(rng.normal(0.0, 0.35, size=(n, 1)))   # how big it is
        tint = rng.normal(0.0, 1.0, size=(n, 1))                  # local colour
        place = rng.normal(0.0, 1.0, size=(n, 1))                 # where it stands

        tint_dir = np.ones((1, D_VISUS)) * 0.95
        place_dir = np.ones((1, D_TACTUS)) * 0.95

        visus = (form @ self.P_visus) * magnitude + tint * tint_dir
        tactus = (form @ self.P_tactus) + place * place_dir
        auditus = (form @ self.P_auditus)

        # --- discordant sensible ------------------------------------------
        discord = rng.random(n) < self.p_discord
        if discord.any():
            other = rng.integers(0, self.n_kinds, size=int(discord.sum()))
            auditus[discord] = self.forms[other] @ self.P_auditus

        # The ear is the clean sense here and the eye the cluttered one — which
        # is why the ear is worth gating case by case rather than trusting or
        # discarding wholesale. Discarding it wholesale costs the fine
        # distinction on which every intention in this world hangs.
        visus += rng.normal(0.0, 0.55, size=visus.shape)
        tactus += rng.normal(0.0, 0.55, size=tactus.shape)
        auditus += rng.normal(0.0, 0.10, size=auditus.shape)

        # --- the body it lands in -----------------------------------------
        if knower is None:
            who = rng.integers(0, self.n_knowers, size=n)
        else:
            who = np.full(n, knower, dtype=int)
        manifest = self.manifest_kind[kind]
        intention = self.intention_kind[kind].copy()
        occult_here = ~manifest
        if occult_here.any():
            intention[occult_here] = rng.integers(0, 2, size=int(occult_here.sum()))

        return Batch(
            visus=visus, tactus=tactus, auditus=auditus,
            kind=kind, form=form,
            intention=intention, manifest=manifest,
            discord=discord, knower=who,
        )


# ==============================================================================
# PART II — THE DENUDATIO COLUMN
# ==============================================================================
# Each class below is one chamber of Albert's brain-map. The comment above each
# gives the doctrine it implements and the one design decision that follows from
# the doctrine rather than from convenience.
# ==============================================================================

def _init(rng, *shape, gain=1.0) -> Tensor:
    """Fan-in scaled initialisation; all parameters are graph leaves."""
    fan_in = shape[0] if len(shape) > 1 else shape[0]
    w = rng.normal(0.0, gain / math.sqrt(fan_in), size=shape)
    return Tensor(w)


def _zeros(*shape) -> Tensor:
    return Tensor(np.zeros(shape))


def row_dot(a: Tensor, b: Tensor) -> Tensor:
    """Row-wise inner product, shape (N, 1). Used for sensory agreement."""
    return (a * b).sum(axis=1, keepdims=True)


def l2n(x: Tensor) -> Tensor:
    """Row-wise L2 normalisation, so agreement is measured as direction."""
    return x / (((x ** 2.0).sum(axis=1, keepdims=True) + 1e-9) ** 0.5)


def stop_grad(t: Tensor) -> Tensor:
    """Detach a value from the graph. Used where the regimen must weigh the
    demonstrative loss as a *fact about the case*, not as something to improve
    by making demonstration worse."""
    return Tensor(t.data.copy(), requires_grad=False)


def materiality(z: Tensor, kind: np.ndarray, n_kinds: int) -> Tensor:
    """How much of this rung's variance is still particular rather than specific.

    Albert orders the inner senses 'according to the levels of abstraction'. That
    ordering is only meaningful if abstraction is a quantity. Here it is:

        materiality = (variance within a kind) / (variance overall)

    A rung at materiality 1.0 has kept every accident and separated no kinds. A
    rung at materiality 0 has discarded every accident: all singulars of a kind
    have collapsed onto their universal. The column is trained so this number
    never rises as you climb.
    """
    n = z.data.shape[0]
    G = np.zeros((n, n_kinds))
    G[np.arange(n), kind] = 1.0
    counts = np.maximum(G.sum(axis=0), 1.0).reshape(n_kinds, 1)

    Gt = Tensor(G.T, requires_grad=False)
    Gc = Tensor(G, requires_grad=False)
    inv_counts = Tensor(1.0 / counts, requires_grad=False)

    mu = (Gt @ z) * inv_counts             # (K, H) per-kind centroid
    mu_b = Gc @ mu                          # (N, H) each row's own centroid
    within = ((z - mu_b) ** 2.0).mean()
    grand = z.mean(axis=0, keepdims=True)
    total = ((z - grand) ** 2.0).mean() + 1e-8
    return within / total


class DenudatioColumn:
    """The whole mind: senses -> common sense -> imagination -> estimation ->
    light -> possible intellect -> habitus, plus the regimen of certainty."""

    def __init__(self, seed: int = 1280, n_knowers: int = 3,
                 shared_agens: bool = False, cogitative: bool = True,
                 complexio: Optional[np.ndarray] = None):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.n_knowers = n_knowers
        self.shared_agens = shared_agens   # True == the Averroist model
        self.cogitative = cogitative       # True == estimation under reason
        self.n_kinds = N_KINDS

        # ---- 1. SENSUS COMMUNIS ------------------------------------------
        # Doctrine: the common sense is the terminus of the proper sensibles; it
        # is where the several senses' reports of one object become one report.
        # Design consequence: it does not concatenate. It gates. Each modality is
        # weighted by how far it agrees with the others, so a discordant channel
        # is discounted before anything above ever sees it.
        self.Wv = _init(rng, D_VISUS, H_BOUND)
        self.Wt = _init(rng, D_TACTUS, H_BOUND)
        self.Wa = _init(rng, D_AUDITUS, H_BOUND)
        self.bv, self.bt, self.ba = _zeros(1, H_BOUND), _zeros(1, H_BOUND), _zeros(1, H_BOUND)
        # one sharpness and one bias per modality: the common sense may learn to
        # trust the hand more than the ear, and does
        self.alpha = Tensor(np.full((1, 3), 2.5))
        self.beta = Tensor(np.zeros((1, 3)))
        self.Wc = _init(rng, H_BOUND, H_BOUND)
        self.bc = _zeros(1, H_BOUND)

        # ---- 2. IMAGINATIO ------------------------------------------------
        # Doctrine: imagination is the *thesaurus formarum* — the storehouse of
        # sensible forms, which keeps the form when the object is gone. It stores
        # forms only; intentions are stored elsewhere. Design consequence: this
        # layer has a retention constant and is demonstrated holding a phantasm
        # after the object is withdrawn (see `retain`).
        self.Wi = _init(rng, H_BOUND, H_IMAG)
        self.bi = _zeros(1, H_IMAG)
        # the retention matrix is NOT trained: retention is a property of the
        # storehouse, not something the column optimises. Kept as plain array so
        # it never enters the parameter set or the gradient check.
        W = rng.normal(0.0, 1.0, size=(H_IMAG, H_IMAG))
        self.W_retentio = W / (np.abs(np.linalg.eigvals(W)).max() + 1e-9)
        self.mixing = 0.13

        # the body this column is seated in (fixed, not a parameter)
        self.complexio = (np.ones((n_knowers, H_IMAG)) if complexio is None
                          else np.asarray(complexio, dtype=np.float64))

        # ---- 3. INTELLECTUS AGENS (the light) -----------------------------
        # Doctrine: 'the agent intellect adheres to the possible as its form'.
        # Against Averroes: it is numbered to the number of the men in whom it is.
        # Design consequence: illumination is MULTIPLICATIVE, not additive — light
        # does not add content to what is seen, it makes what is there visible —
        # and its seed is per-knower unless we are running the Averroist ablation.
        n_seeds = 1 if shared_agens else n_knowers
        self.Seeds = _init(rng, n_seeds, S_SEED, gain=1.4)
        self.Wg = _init(rng, S_SEED, H_IMAG)
        self.bg = _zeros(1, H_IMAG)
        self.Wl = _init(rng, H_IMAG, H_INTEL)
        self.bl = _zeros(1, H_INTEL)

        # ---- 4. INTELLECTUS POSSIBILIS ------------------------------------
        # Doctrine: the possible intellect is in potency to all intelligibles and
        # is actualised by *becoming* them. Design consequence: knowledge is not
        # scored by matching a stored key. The assimilated state must be able to
        # regenerate the universal from itself. If it cannot, it has not become it.
        self.Wp = _init(rng, H_INTEL, H_INTEL)
        self.bp = _zeros(1, H_INTEL)
        self.Wd = _init(rng, H_INTEL, F_DIM)          # regeneration of the universal
        self.bd = _zeros(1, F_DIM)
        self.Wk = _init(rng, H_INTEL, N_KINDS)        # diagnostic readout only
        self.bk = _zeros(1, N_KINDS)

        # ---- 5. AESTIMATIVA / COGITATIVA -----------------------------------
        # Doctrine: the estimative power apprehends 'intentiones non sensatae' —
        # the sheep's apprehension of the wolf's hostility, which is no colour and
        # no shape. In man Albert holds this power works under reason.
        # Design consequence: two wirings of one organ. The estimative receives
        # only what climbs from below; the cogitative also receives the universal
        # descending from the intellect. Both are built; the ablation reports both.
        est_in = H_IMAG + (H_INTEL if cogitative else 0)
        self.We = _init(rng, est_in, H_EST)
        self.be = _zeros(1, H_EST)
        self.Wm = _init(rng, H_EST, 1)     # demonstratio: the intention by reasoning
        self.bm = _zeros(1, 1)

        # ---- 6. REGIMEN: demonstratio vs experimentum -----------------------
        # Doctrine: 'experimentum solum certificat IN TALIBUS'. The restriction is
        # the doctrine. Design consequence: a learned per-case decision, priced
        # against a finite budget, with no label telling it which causes are occult.
        self.Wr = _init(rng, H_EST + 1 + H_INTEL, 1)
        self.br = _zeros(1, 1)

        # ---- 7. HABITUS / INTELLECTUS ADEPTUS ------------------------------
        # Doctrine: what the intellect has become sediments into a possession — the
        # adept intellect, the mind's own acquired store of species. Not a
        # parameter: an accumulation. Kept as a running mean, outside the gradient.
        self.habitus = np.zeros((N_KINDS, H_INTEL))
        self.habitus_n = np.zeros(N_KINDS)

        # The estimative organ READS the phantasm; it does not reach back down
        # and retune the imagination to suit itself. Detaching its inputs
        # enforces that, and makes the estimativa/cogitativa ablation a test of
        # the wire alone rather than of which task got to shape the column.
        # (Switched off during the gradient check so finite differences and
        # analytic gradients are measuring the same function.)
        self.detach_estimative = True

        # memoria: intentions with a time index (populated after training)
        self.memoria: List[Dict] = []

    # ---------------------------------------------------------------- params
    def parameters(self) -> Dict[str, Tensor]:
        return {k: v for k, v in self.__dict__.items() if isinstance(v, Tensor)}

    def zero_grad(self):
        for p in self.parameters().values():
            p.zero_grad()

    # ---------------------------------------------------------------- forward
    def sensus_communis(self, v: Tensor, t: Tensor, a: Tensor):
        """Bind three sensory reports into one, discounting whichever disagrees."""
        hv = (v @ self.Wv + self.bv).tanh()
        ht = (t @ self.Wt + self.bt).tanh()
        ha = (a @ self.Wa + self.ba).tanh()

        # each modality is compared with the consensus of the other two, by
        # COSINE, so the comparison is of direction and not of loudness
        nv, nt, na = l2n(hv), l2n(ht), l2n(ha)
        agree_v = row_dot(nv, l2n((ht + ha) * 0.5))
        agree_t = row_dot(nt, l2n((hv + ha) * 0.5))
        agree_a = row_dot(na, l2n((hv + ht) * 0.5))
        gv = (agree_v * self.alpha.cols(0, 1) + self.beta.cols(0, 1)).sigmoid()
        gt = (agree_t * self.alpha.cols(1, 2) + self.beta.cols(1, 2)).sigmoid()
        ga = (agree_a * self.alpha.cols(2, 3) + self.beta.cols(2, 3)).sigmoid()

        mixed = hv * gv + ht * gt + ha * ga
        bound = (mixed @ self.Wc + self.bc).tanh()
        return bound, (gv, gt, ga)

    def imaginatio(self, bound: Tensor, knower: np.ndarray) -> Tensor:
        """Retain the form — in the chamber of a particular body.

        The complexio multiplies here and nowhere else. Two knowers looking at
        the same thing hold differently-tempered phantasms of it. Nothing above
        this line can see whose body it is standing in; it can only cope.
        """
        ph = (bound @ self.Wi + self.bi).tanh()
        gain = Tensor(self.complexio[knower], requires_grad=False)
        return ph * gain

    def agens(self, phantasm: Tensor, knower: np.ndarray):
        """Strike this knower's own light and shine it through the phantasm."""
        n = phantasm.data.shape[0]
        rows = np.zeros(n, dtype=int) if self.shared_agens else knower
        onehot = np.zeros((n, self.Seeds.data.shape[0]))
        onehot[np.arange(n), rows] = 1.0
        seed = Tensor(onehot, requires_grad=False) @ self.Seeds

        lumen = (seed @ self.Wg + self.bg).softplus()     # strictly positive: light
        illuminated = ((phantasm * lumen) @ self.Wl + self.bl).tanh()
        return illuminated, lumen

    def possibilis(self, illuminated: Tensor):
        """Become the universal, then prove it by regenerating it."""
        s = (illuminated @ self.Wp + self.bp).tanh()
        form_hat = s @ self.Wd + self.bd
        kind_logits = s @ self.Wk + self.bk
        return s, form_hat, kind_logits

    def estimativa(self, phantasm: Tensor, s: Optional[Tensor]):
        """Read the intention the senses never carried.

        It reads the phantasm as this body holds it — the third chamber follows
        the second — and nothing else, unless it is the human variant, in which
        case the universal descends into it from above.
        """
        x = stop_grad(phantasm) if self.detach_estimative else phantasm
        if self.cogitative:
            assert s is not None
            x = x.concat_with(stop_grad(s) if self.detach_estimative else s, axis=1)
        e = (x @ self.We + self.be).tanh()
        demonstratio = e @ self.Wm + self.bm
        return e, demonstratio

    def regimen(self, e: Tensor, demonstratio: Tensor, s: Tensor) -> Tensor:
        """Decide whether reasoning suffices here, or the world must be asked."""
        feats = e.concat_with(demonstratio, axis=1).concat_with(s, axis=1)
        return (feats @ self.Wr + self.br).sigmoid()

    def forward(self, b: Batch):
        v = Tensor(b.visus, requires_grad=False)
        t = Tensor(b.tactus, requires_grad=False)
        a = Tensor(b.auditus, requires_grad=False)

        bound, gates = self.sensus_communis(v, t, a)
        phantasm = self.imaginatio(bound, b.knower)
        illuminated, lumen = self.agens(phantasm, b.knower)
        s, form_hat, kind_logits = self.possibilis(illuminated)
        e, demonstratio = self.estimativa(phantasm, s)
        p_exp = self.regimen(e, demonstratio, s)

        raw = Tensor(np.concatenate([b.visus, b.tactus, b.auditus], axis=1),
                     requires_grad=False)
        return {
            "raw": raw, "bound": bound, "phantasm": phantasm,
            "illuminated": illuminated, "s": s,
            "form_hat": form_hat, "kind_logits": kind_logits,
            "e": e, "demonstratio": demonstratio, "p_exp": p_exp,
            "gates": gates, "lumen": lumen,
        }

    # ------------------------------------------------------------ retention
    def retain(self, b: Batch, steps: int = 6) -> np.ndarray:
        """Withdraw the object and watch the imagination hold the form.

        Returns cosine similarity between the retained trace and the phantasm as
        first formed, for each step after the object is gone. Albert's claim is
        that the storehouse of forms survives the departure of the sensible; this
        is that claim, measured.
        """
        out = self.forward(b)
        ph0 = out["phantasm"].data
        r = ph0.copy()
        sims = []
        for _ in range(steps):
            r = (1.0 - self.mixing) * r + self.mixing * np.tanh(r @ self.W_retentio)
            num = (r * ph0).sum(axis=1)
            den = np.linalg.norm(r, axis=1) * np.linalg.norm(ph0, axis=1) + 1e-9
            sims.append(float(np.mean(num / den)))
        return np.array(sims)


# ==============================================================================
# PART III — THE OBJECTIVE
# ==============================================================================
# Five terms. Four of them exist because a doctrine demands them; the fifth is
# ordinary weight decay.
# ==============================================================================

EXPERIMENT_COST = 0.32     # what one going-and-looking costs, in loss units
EXPERIMENT_BUDGET = 0.55   # fraction of cases that may be settled by experiment
DENUD_MARGIN = 0.02        # a rung must strip at least this much, or be penalised


def objective(model: DenudatioColumn, b: Batch, out: Dict,
              detach_regimen: bool = True) -> Tuple[Tensor, Dict]:
    """`detach_regimen` controls one deliberate asymmetry. In training it is True:
    the router weighs the demonstrative error as a *fact about the case*, and is
    not allowed to make experiments look attractive by wrecking demonstration.
    That means the analytic gradient omits a path a finite-difference probe would
    still walk, so the gradient check sets it False and compares like with like."""
    n = b.kind.shape[0]

    # (1) ASSIMILATION — the possible intellect must regenerate the universal.
    form_t = Tensor(b.form, requires_grad=False)
    L_assim = ((out["form_hat"] - form_t) ** 2.0).mean()

    # (2) DISTINCTION — kinds must remain distinguishable after stripping.
    L_kind = cross_entropy(out["kind_logits"], b.kind)

    # (3) INTENTION — the estimative organ reads the unsensed intention.
    bce = bce_with_logits_per_sample(out["demonstratio"], b.intention)
    L_intent = bce.mean()

    # (4) REGIMEN — price each case: pay the cost, or wear the demonstrative error.
    #     The demonstrative error enters detached, so the router cannot 'win' by
    #     sabotaging demonstration.
    p = out["p_exp"]
    bce_for_router = stop_grad(bce) if detach_regimen else bce
    per_case = p * EXPERIMENT_COST + (Tensor(np.ones((n, 1)), requires_grad=False) - p) * bce_for_router
    L_regimen = per_case.mean()
    over = p.mean() - EXPERIMENT_BUDGET
    L_budget = over.relu() * 3.0

    # (5) DENUDATIO — materiality must fall monotonically up the column.
    rungs = ["raw", "bound", "phantasm", "illuminated", "s"]
    mats = [materiality(out[r], b.kind, model.n_kinds) for r in rungs]
    L_denud = Tensor(np.array(0.0), requires_grad=False)
    for lo, hi in zip(mats[:-1], mats[1:]):
        L_denud = L_denud + (hi - lo + DENUD_MARGIN).relu()

    # (6) weight decay
    L_wd = Tensor(np.array(0.0), requires_grad=False)
    for name, prm in model.parameters().items():
        if name.startswith("W") or name == "Seeds":
            L_wd = L_wd + (prm ** 2.0).mean()

    total = (L_assim * 1.0 + L_kind * 0.5 + L_intent * 1.0
             + L_regimen * 1.0 + L_budget + L_denud * 0.6 + L_wd * 1e-3)

    parts = {
        "assim": float(L_assim.data), "kind": float(L_kind.data),
        "intent": float(L_intent.data), "regimen": float(L_regimen.data),
        "denud": float(L_denud.data), "total": float(total.data),
        "materiality": [float(m.data) for m in mats],
    }
    return total, parts


# ==============================================================================
# PART IV — FINITE-DIFFERENCE GRADIENT CHECK (mandatory; runs before training)
# ==============================================================================
# Albert's own standard, applied to the file that claims to be his mind:
# "fui et vidi experiri" — I was there and saw it tested. An architecture whose
# gradients are not verified is an assertion, not an experiment.
# ==============================================================================

def gradient_check(verbose: bool = True, n: int = 24, eps: float = 1e-6,
                   tol: float = 1e-5) -> bool:
    world = WorldOfParticulars(seed=7, n_knowers=4)
    model = DenudatioColumn(seed=11, n_knowers=4, shared_agens=False,
                            cogitative=True, complexio=world.complexio)
    model.detach_estimative = False   # compare like with like against finite differences
    batch = world.draw(n)

    def loss_value() -> float:
        out = model.forward(batch)
        total, _ = objective(model, batch, out, detach_regimen=False)
        return float(total.data)

    model.zero_grad()
    out = model.forward(batch)
    total, _ = objective(model, batch, out, detach_regimen=False)
    total.backward()

    params = model.parameters()
    rng = np.random.default_rng(3)
    worst = 0.0
    rows = []
    for name, prm in params.items():
        flat = prm.data.reshape(-1)
        n_probe = min(4, flat.size)
        idx = rng.choice(flat.size, size=n_probe, replace=False)
        gname = 0.0
        for i in idx:
            old = flat[i]
            flat[i] = old + eps
            lp = loss_value()
            flat[i] = old - eps
            lm = loss_value()
            flat[i] = old
            numeric = (lp - lm) / (2.0 * eps)
            analytic = prm.grad.reshape(-1)[i]
            denom = max(1.0, abs(numeric), abs(analytic))
            rel = abs(numeric - analytic) / denom
            gname = max(gname, rel)
        rows.append((name, gname))
        worst = max(worst, gname)

    if verbose:
        print("  parameter        max relative error (analytic vs finite-difference)")
        for name, err in sorted(rows, key=lambda r: -r[1]):
            print(f"    {name:<10s}   {err:.3e}")
        print(f"  WORST OVER ALL PARAMETERS: {worst:.3e}   "
              f"(tolerance {tol:.0e}) -> {'PASS' if worst < tol else 'FAIL'}")
    return worst < tol


# ==============================================================================
# PART V — TRAINING
# ==============================================================================

class Adam:
    def __init__(self, params: Dict[str, Tensor], lr: float = 6e-3,
                 b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.p = params
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v.data) for k, v in params.items()}
        self.v = {k: np.zeros_like(v.data) for k, v in params.items()}
        self.t = 0

    def step(self):
        self.t += 1
        for k, prm in self.p.items():
            g = prm.grad
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            prm.data -= self.lr * mh / (np.sqrt(vh) + self.eps)


def update_habitus(model: DenudatioColumn, s: np.ndarray, kind: np.ndarray):
    """Sediment what has been become into a possession (intellectus adeptus)."""
    for k in range(model.n_kinds):
        m = kind == k
        if not m.any():
            continue
        mean_s = s[m].mean(axis=0)
        cnt = model.habitus_n[k]
        model.habitus[k] = (model.habitus[k] * cnt + mean_s * m.sum()) / (cnt + m.sum())
        model.habitus_n[k] += m.sum()


def train(model: DenudatioColumn, world: WorldOfParticulars, steps: int = 900,
          batch: int = 96, lr: float = 6e-3, log_every: int = 0) -> List[Dict]:
    opt = Adam(model.parameters(), lr=lr)
    history = []
    for step in range(1, steps + 1):
        b = world.draw(batch)
        out = model.forward(b)
        total, parts = objective(model, b, out)
        model.zero_grad()
        total.backward()
        opt.step()
        if step > steps // 2:      # sediment habitus only once forms have settled
            update_habitus(model, out["s"].data, b.kind)
        if log_every and (step % log_every == 0 or step == 1):
            print(f"    step {step:4d} | total {parts['total']:.4f} "
                  f"| assim {parts['assim']:.4f} | kind {parts['kind']:.4f} "
                  f"| intent {parts['intent']:.4f} | denud {parts['denud']:.4f}")
        history.append(parts)
    return history


# ==============================================================================
# PART VI — EVALUATION
# ==============================================================================

def evaluate(model: DenudatioColumn, world: WorldOfParticulars,
             n: int = 3000) -> Dict:
    b = world.draw(n)
    out = model.forward(b)

    s = out["s"].data
    kind_pred = out["kind_logits"].data.argmax(axis=1)
    acc_kind = float((kind_pred == b.kind).mean())

    # recognition by habitus: nearest acquired species, no readout head involved
    if model.habitus_n.sum() > 0:
        d = ((s[:, None, :] - model.habitus[None, :, :]) ** 2).sum(axis=2)
        acc_habitus = float((d.argmin(axis=1) == b.kind).mean())
    else:
        acc_habitus = float("nan")

    # assimilation quality: how much of the universal the state regenerates
    resid = ((out["form_hat"].data - b.form) ** 2).mean()
    var = ((b.form - b.form.mean(axis=0)) ** 2).mean()
    r2_form = float(1.0 - resid / var)

    # the regimen in action
    p = out["p_exp"].data.reshape(-1)
    take_exp = p > 0.5
    demo = (out["demonstratio"].data.reshape(-1) > 0).astype(int)
    settled = np.where(take_exp, b.intention, demo)     # experiment returns truth
    acc_intent = float((settled == b.intention).mean())
    acc_demo_only = float((demo == b.intention).mean())

    exp_manifest = float(take_exp[b.manifest].mean())
    exp_occult = float(take_exp[~b.manifest].mean())

    # the common sense doing its work
    gv, gt, ga = (g.data.reshape(-1) for g in out["gates"])
    gate_disc = float(ga[b.discord].mean())
    gate_conc = float(ga[~b.discord].mean())

    mats = [float(materiality(out[r], b.kind, model.n_kinds).data)
            for r in ["raw", "bound", "phantasm", "illuminated", "s"]]

    per_knower = {}
    for w in range(world.n_knowers):
        m = b.knower == w
        if m.any():
            per_knower[w] = float((kind_pred[m] == b.kind[m]).mean())

    return {
        "acc_kind": acc_kind, "acc_habitus": acc_habitus, "r2_form": r2_form,
        "acc_intent_under_regimen": acc_intent, "acc_demonstratio_alone": acc_demo_only,
        "experiment_rate_manifest": exp_manifest,
        "experiment_rate_occult": exp_occult,
        "experiment_rate_overall": float(take_exp.mean()),
        "gate_on_discordant": gate_disc, "gate_on_concordant": gate_conc,
        "materiality_ladder": mats, "per_knower_acc": per_knower,
    }


# ==============================================================================
# PART VII — REMINISCENTIA
# ==============================================================================
# Albert keeps memoria and reminiscentia apart: the first is a store of
# intentions with a time index, shared with animals; the second is a *search*
# through that store, proper to man, moving from one intention to a neighbouring
# one until the sought episode is reached. Implemented literally: a beam search
# along the temporal chain, scored by intention similarity, with no gradient and
# no global lookup. A pure similarity search over the whole store is run beside
# it for comparison — that is what memoria alone could do.
# ==============================================================================

def build_memoria(model: DenudatioColumn, world: WorldOfParticulars, n: int = 240):
    b = world.draw(n)
    out = model.forward(b)
    e = out["e"].data
    model.memoria = [
        {"t": i, "intentio": e[i], "kind": int(b.kind[i]),
         "value": int(b.intention[i]), "manifest": bool(b.manifest[i])}
        for i in range(n)
    ]
    return b


def reminiscentia(model: DenudatioColumn, cue: np.ndarray, target_kind: int,
                  beam: int = 4, hops: int = 6) -> Dict:
    """Directed recollection: start where the cue is warmest, then walk the chain."""
    mem = model.memoria
    if not mem:
        return {}
    keys = np.stack([m["intentio"] for m in mem])
    keys_n = keys / (np.linalg.norm(keys, axis=1, keepdims=True) + 1e-9)
    cue_n = cue / (np.linalg.norm(cue) + 1e-9)
    sim = keys_n @ cue_n

    # memoria alone: the single warmest trace
    direct = int(sim.argmax())

    # reminiscentia: a beam that may step to temporal neighbours
    frontier = list(np.argsort(-sim)[:beam])
    visited = set(frontier)
    best, best_score = direct, sim[direct] + (1.0 if mem[direct]["kind"] == target_kind else 0.0)
    for _ in range(hops):
        cands = []
        for i in frontier:
            for j in (i - 2, i - 1, i + 1, i + 2):
                if 0 <= j < len(mem) and j not in visited:
                    cands.append(j)
        if not cands:
            break
        scored = sorted(cands, key=lambda j: -(sim[j] + (1.0 if mem[j]["kind"] == target_kind else 0.0)))
        frontier = scored[:beam]
        visited.update(frontier)
        for j in frontier:
            sc = sim[j] + (1.0 if mem[j]["kind"] == target_kind else 0.0)
            if sc > best_score:
                best, best_score = j, sc
    return {
        "memoria_only_hit": mem[direct]["kind"] == target_kind,
        "reminiscentia_hit": mem[best]["kind"] == target_kind,
        "steps_examined": len(visited),
        "store_size": len(mem),
    }


# ==============================================================================
# PART VIII — THE ABLATIONS
# ==============================================================================
# Two arguments Albert actually had, restated as experiments this file runs.
# ==============================================================================

def ablation_unitate_intellectus(steps: int = 900,
                                 seeds: Sequence[int] = (1280, 4104, 715)) -> Dict:
    """De unitate intellectus contra Averroistas, c. 1256, as a controlled run.

    Averroes: one intellect for the whole species; individuals differ only in the
    phantasms they feed it. Albert: since the vegetative and the sensitive are
    numbered to the number of the men in whom they are, the rational is numbered
    to the same number. The intellective soul is the form of THIS body.

    Here each knower's body imposes its own complexio on its phantasms. The two
    models are identical except in one place: whether the light that strips the
    phantasm is one, or one per knower. Run over several initialisations, because
    a difference of a few thousandths on one seed is not a result.
    """
    results: Dict[str, Dict] = {}
    for label, shared in (("albertist_owned_light", False),
                          ("averroist_shared_light", True)):
        accs, r2s, mats, per_seed = [], [], [], []
        for sd in seeds:
            world = WorldOfParticulars(seed=1200, n_knowers=4)
            model = DenudatioColumn(seed=sd, n_knowers=4, shared_agens=shared,
                                    cogitative=True, complexio=world.complexio)
            train(model, world, steps=steps)
            ev = evaluate(model, world, n=3000)
            accs.append(ev["acc_kind"])
            r2s.append(ev["r2_form"])
            mats.append(ev["materiality_ladder"][-1])
            per_seed.append(ev["per_knower_acc"])
        results[label] = {
            "acc_kind_mean": float(np.mean(accs)), "acc_kind_std": float(np.std(accs)),
            "acc_kind_runs": accs, "r2_mean": float(np.mean(r2s)),
            "materiality_top_mean": float(np.mean(mats)),
            "worst_knower_mean": float(np.mean([min(d.values()) for d in per_seed])),
        }
    return results


def ablation_cogitativa(steps: int = 900,
                        seeds: Sequence[int] = (1280, 4104, 715)) -> Dict:
    """Estimativa (brute) vs cogitativa (the same organ under reason).

    Albert declines to call the brutes' power cogitative at all: their reason-like
    feats — the spider's web, the swallow's nest, the storing of food — he assigns
    to estimation or fantasy, working ad instinctum naturae, and he prefers those
    names to Averroes' terminology. In man the middle chamber is 'rationalis
    syllogistica'. The difference here is exactly one wire: whether the universal
    descends from the intellect into the estimative organ. Both variants read a
    detached phantasm, so neither gets to retune the column to suit its own task.
    """
    results: Dict[str, Dict] = {}
    for label, cog in (("cogitativa_under_reason", True),
                       ("estimativa_bottom_up", False)):
        man, occ = [], []
        for sd in seeds:
            world = WorldOfParticulars(seed=1200, n_knowers=4)
            model = DenudatioColumn(seed=sd, n_knowers=4, shared_agens=False,
                                    cogitative=cog, complexio=world.complexio)
            train(model, world, steps=steps)
            b = world.draw(4000)
            out = model.forward(b)
            demo = (out["demonstratio"].data.reshape(-1) > 0).astype(int)
            m = b.manifest
            man.append(float((demo[m] == b.intention[m]).mean()))
            occ.append(float((demo[~m] == b.intention[~m]).mean()))
        results[label] = {
            "manifest_mean": float(np.mean(man)), "manifest_std": float(np.std(man)),
            "manifest_runs": man, "occult_mean": float(np.mean(occ)),
        }
    return results


# ==============================================================================
# PART IX — SELF-TESTS
# ==============================================================================

def self_tests(model: DenudatioColumn, world: WorldOfParticulars,
               ev: Dict) -> List[Tuple[str, bool, str]]:
    t: List[Tuple[str, bool, str]] = []

    mats = ev["materiality_ladder"]
    mono = all(mats[i + 1] <= mats[i] + 1e-6 for i in range(len(mats) - 1))
    t.append(("denudatio is monotone up the column", mono,
              " > ".join(f"{m:.3f}" for m in mats)))

    t.append(("the top rung is more abstract than the senses",
              mats[-1] < mats[0] * 0.5,
              f"{mats[-1]:.3f} vs {mats[0]:.3f}"))

    t.append(("the possible intellect regenerates the universal",
              ev["r2_form"] > 0.75, f"R^2 = {ev['r2_form']:.3f}"))

    t.append(("the acquired habitus recognises without a readout head",
              ev["acc_habitus"] > 0.80, f"accuracy = {ev['acc_habitus']:.3f}"))

    t.append(("the common sense discounts the discordant sense",
              ev["gate_on_discordant"] < ev["gate_on_concordant"],
              f"gate {ev['gate_on_discordant']:.3f} vs {ev['gate_on_concordant']:.3f}"))

    t.append(("experiment is spent on occult causes, not manifest ones",
              ev["experiment_rate_occult"] > ev["experiment_rate_manifest"] + 0.3,
              f"occult {ev['experiment_rate_occult']:.2f} vs "
              f"manifest {ev['experiment_rate_manifest']:.2f}"))

    t.append(("the regimen beats demonstration used alone",
              ev["acc_intent_under_regimen"] > ev["acc_demonstratio_alone"] + 0.05,
              f"{ev['acc_intent_under_regimen']:.3f} vs "
              f"{ev['acc_demonstratio_alone']:.3f}"))

    sims = model.retain(world.draw(200), steps=6)
    t.append(("imagination holds the form after the object is withdrawn",
              float(sims[-1]) > 0.30, f"cosine after 6 steps = {sims[-1]:.3f}"))

    lum = model.forward(world.draw(200))["lumen"].data
    t.append(("illumination is strictly positive (light adds no content)",
              bool((lum > 0).all()),
              f"min lumen = {lum.min():.4f}, max = {lum.max():.4f}"))

    return t


# ==============================================================================
# MAIN
# ==============================================================================

def banner(s: str):
    print()
    print("=" * 78)
    print(s)
    print("=" * 78)


def main():
    t0 = time.time()
    np.set_printoptions(precision=4, suppress=True)

    banner("CHAPTER 0264 — ALBERTUS MAGNUS — THE DENUDATIO COLUMN")
    print("Pure NumPy. No autograd library. Gradients verified before training.")

    banner("1. GRADIENT CHECK (analytic backprop vs central finite differences)")
    ok = gradient_check(verbose=True)
    if not ok:
        print("GRADIENT CHECK FAILED — aborting.")
        sys.exit(1)

    banner("2. TRAINING THE COLUMN")
    world = WorldOfParticulars(seed=1200, n_knowers=4)
    print(f"  kinds: {world.n_kinds}   knowers: {world.n_knowers}   "
          f"discordant-sense rate: {world.p_discord:.2f}")
    print(f"  manifest kinds: {np.where(world.manifest_kind)[0].tolist()}   "
          f"occult kinds: {np.where(~world.manifest_kind)[0].tolist()}")
    print(f"  cost of one experiment: {EXPERIMENT_COST}   "
          f"budget: {EXPERIMENT_BUDGET:.0%} of cases")
    print()
    model = DenudatioColumn(seed=1280, n_knowers=4, shared_agens=False,
                            cogitative=True, complexio=world.complexio)
    n_params = sum(p.data.size for p in model.parameters().values())
    print(f"  trainable parameters: {n_params}")
    train(model, world, steps=900, log_every=150)

    banner("3. WHAT THE COLUMN LEARNED")
    ev = evaluate(model, world, n=3000)
    print(f"  kind recognition (readout)          : {ev['acc_kind']:.3f}")
    print(f"  kind recognition (acquired habitus) : {ev['acc_habitus']:.3f}")
    print(f"  universal regenerated from the state: R^2 = {ev['r2_form']:.3f}")
    print()
    print("  DENUDATIO LADDER (materiality: within-kind variance / total variance)")
    names = ["senses (raw)", "sensus communis", "imaginatio",
             "illuminated", "intellectus possibilis"]
    for nm, m in zip(names, ev["materiality_ladder"]):
        bar = "#" * max(1, int(round(m * 46)))
        print(f"    {nm:<24s} {m:6.3f}  {bar}")
    print()
    print("  SENSUS COMMUNIS — gate opened on the auditory channel")
    print(f"    when it agreed with the others : {ev['gate_on_concordant']:.3f}")
    print(f"    when it was discordant         : {ev['gate_on_discordant']:.3f}")
    print()
    print("  REGIMEN — 'experimentum solum certificat IN TALIBUS'")
    print(f"    experiments spent on manifest causes: {ev['experiment_rate_manifest']:.1%}")
    print(f"    experiments spent on occult causes  : {ev['experiment_rate_occult']:.1%}")
    print(f"    overall experiment rate             : {ev['experiment_rate_overall']:.1%}"
          f"  (budget {EXPERIMENT_BUDGET:.0%})")
    print(f"    intention settled, demonstration alone: {ev['acc_demonstratio_alone']:.3f}")
    print(f"    intention settled, under the regimen  : {ev['acc_intent_under_regimen']:.3f}")

    banner("4. DE UNITATE INTELLECTUS — one light, or one light per knower?")
    print("  Same data, same senses, same possible intellect, same everything")
    print("  except whether the abstracting light is shared or seated in a body.")
    print()
    ab = ablation_unitate_intellectus(steps=900)
    for label in ("albertist_owned_light", "averroist_shared_light"):
        r = ab[label]
        runs = ", ".join(f"{x:.3f}" for x in r["acc_kind_runs"])
        print(f"  {label}")
        print(f"    kind recognition  : {r['acc_kind_mean']:.3f} "
              f"+/- {r['acc_kind_std']:.3f}   (seeds: {runs})")
        print(f"    worst-served body : {r['worst_knower_mean']:.3f}")
        print(f"    universal (R^2)   : {r['r2_mean']:.3f}"
              f"   top-rung materiality {r['materiality_top_mean']:.3f}")
    d_acc = (ab["albertist_owned_light"]["acc_kind_mean"]
             - ab["averroist_shared_light"]["acc_kind_mean"])
    d_worst = (ab["albertist_owned_light"]["worst_knower_mean"]
               - ab["averroist_shared_light"]["worst_knower_mean"])
    print()
    print(f"  VERDICT: owning the light is worth {d_acc:+.3f} on average and")
    print(f"           {d_worst:+.3f} to the body it serves worst.")
    print("  One light cannot be well-tempered to four different bodies at once;")
    print("  it settles on a compromise, and somebody is always the compromise.")

    banner("5. ESTIMATIVA vs COGITATIVA — the same organ, with and without reason")
    cg = ablation_cogitativa(steps=900)
    for label in ("cogitativa_under_reason", "estimativa_bottom_up"):
        r = cg[label]
        runs = ", ".join(f"{x:.3f}" for x in r["manifest_runs"])
        print(f"  {label}")
        print(f"    intention, manifest kinds: {r['manifest_mean']:.3f} "
              f"+/- {r['manifest_std']:.3f}   (seeds: {runs})")
        print(f"    intention, occult kinds  : {r['occult_mean']:.3f}"
              f"   (unreachable in principle; 0.50 is the ceiling)")
    gap = (cg["cogitativa_under_reason"]["manifest_mean"]
           - cg["estimativa_bottom_up"]["manifest_mean"])
    print()
    print(f"  VERDICT: the descending universal is worth {gap:+.3f} on manifest causes.")
    print("  Neither variant can reach the occult ones, and no wiring ever will.")
    print("  That is the whole content of 'in talibus'.")

    banner("6. MEMORIA vs REMINISCENTIA")
    mb = build_memoria(model, world, n=240)
    out = model.forward(mb)
    e = out["e"].data
    rng = np.random.default_rng(5)
    hits_mem, hits_rem, examined = 0, 0, []
    trials = 60
    for _ in range(trials):
        i = int(rng.integers(0, len(model.memoria)))
        target = model.memoria[i]["kind"]
        cue = e[i] + rng.normal(0, 0.55, size=e.shape[1])   # a degraded cue
        r = reminiscentia(model, cue, target)
        hits_mem += int(r["memoria_only_hit"])
        hits_rem += int(r["reminiscentia_hit"])
        examined.append(r["steps_examined"])
    print(f"  store: {len(model.memoria)} time-indexed intentions")
    print(f"  memoria alone (warmest trace)      : {hits_mem}/{trials} recovered")
    print(f"  reminiscentia (directed chain walk): {hits_rem}/{trials} recovered")
    print(f"  mean traces examined per recollection: {np.mean(examined):.1f} "
          f"of {len(model.memoria)}")

    banner("7. SELF-TESTS")
    tests = self_tests(model, world, ev)
    n_pass = 0
    for name, passed, detail in tests:
        mark = "PASS" if passed else "FAIL"
        n_pass += int(passed)
        print(f"  [{mark}] {name}")
        print(f"         {detail}")
    print()
    print(f"  {n_pass}/{len(tests)} self-tests passed.")

    banner("DONE")
    print(f"  wall clock: {time.time() - t0:.1f} s")    
    return 0 if n_pass == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())

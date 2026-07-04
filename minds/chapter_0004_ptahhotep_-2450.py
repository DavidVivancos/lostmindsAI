#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 chapter_0004_ptahhotep_-2450.py  —  THE PTAHHOTEP ARCHITECTURE (PTH-1)
  Ptahhotep, vizier of Memphis (c. 2400 BCE)

Part Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 1 Minds 1 - 20 Available on Amazon https://www.amazon.com/dp/B0H6F9L324
Resume and Interactive Demos at https://artificiology.com/


================================================================================

WHAT THIS FILE IS
-----------------
This is NOT a slideshow or a stub. It is a small but *complete and trainable*
artificial-mind built from first principles in pure NumPy:

  1. A correct reverse-mode automatic-differentiation engine (the `Tensor`
     class). This is the "fundamental code" — the calculus a neuron needs in
     order to learn. It is verified against finite-difference gradients in the
     test-suite at the bottom of the file.

  2. A novel neuron — the GOVERNED NEURON — that hard-wires Ptahhotep's single
     most distinctive idea ("the deliberative mind must govern the impulses")
     directly into the unit of computation. Every neuron has two pathways: a
     fast *impulse* pathway and a slow *governor* gate that decides how much of
     the impulse is allowed to pass.

  3. A full multi-task architecture (`PtahhotepMind`) that performs the two
     tasks Ptahhotep cared about most: understanding the *kind* of speech-act a
     message is (its illocutionary force) and deciding whether speaking will
     strengthen or harm the social fabric (speak vs. stay-silent).

  4. A "Ma'at" moderation regularizer — the golden mean expressed as a loss
     term that keeps the governor gates out of saturation.

  5. A non-parametric TEMPORAL MEMORY whose accuracy *grows with experience*,
     a computational rendering of Ptahhotep's claim that "wisdom comes with age."

  6. A real Adam optimizer, a structured synthetic dataset with genuine latent
     causal structure, a training loop, and an automated test-suite that asserts
     the network actually learns.

WHY THESE CHOICES (the philosophy → architecture map)
-----------------------------------------------------
Ptahhotep left us the oldest surviving book of conduct in the world. Stripped to
its computational skeleton, his thought makes four engineering claims:

    "The mind must govern the impulses."   -> GovernedNeuron (gated dual pathway)
    "Speech is a moral act, not just data."-> SpeechActHead + GovernanceGate
    "Seek the middle; shun the extremes."  -> Ma'at moderation regularizer
    "Wisdom is accumulated, not innate."    -> TemporalMemory (grows with episodes)

RUN IT
------
    python3 chapter_0004_ptahhotep_-2450.py            # train + full report
    python3 chapter_0004_ptahhotep_-2450.py --test     # run the test-suite only

Author: Minds terabook project. Dependencies: numpy only (CPU, seconds to run).
================================================================================
"""

from __future__ import annotations
import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# PART I — THE FUNDAMENTAL CODE: A REVERSE-MODE AUTOGRAD ENGINE
# =============================================================================
#
# A neuron learns by gradient descent. To descend a gradient you must first be
# able to *compute* it. Modern frameworks (PyTorch, JAX) do this with reverse-
# mode automatic differentiation: every arithmetic operation records how to
# push a gradient backwards through itself, and a final topological sweep
# accumulates the gradient of a scalar loss with respect to every parameter.
#
# We implement that here from scratch so that nothing about how this mind learns
# is hidden inside a library. The `Tensor` wraps a NumPy array, remembers the
# operations that produced it, and exposes `.backward()`.
# -----------------------------------------------------------------------------


class Tensor:
    """A NumPy array that remembers how it was computed so gradients can flow back.

    Each Tensor stores:
      - data:      the forward values (np.ndarray, float64)
      - grad:      the accumulated gradient dLoss/dself (same shape as data)
      - _backward: a closure that pushes `grad` to this node's parents
      - _prev:     the set of parent Tensors in the computation graph
    """

    __slots__ = ("data", "grad", "_backward", "_prev", "_op")

    def __init__(self, data, _children: Tuple = (), _op: str = ""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward: Callable[[], None] = lambda: None
        self._prev = set(_children)
        self._op = _op

    # --- shape helpers -------------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    @staticmethod
    def _unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
        """Reduce `grad` so it matches `shape`, undoing NumPy broadcasting.

        When a forward op broadcasts (e.g. adding a bias of shape (H,) to a
        batch of shape (N, H)), the backward gradient must be summed over the
        dimensions that were broadcast, otherwise shapes will not match.
        """
        # 1) collapse extra leading dimensions
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)
        # 2) collapse dimensions that were size-1 in the original
        for i, s in enumerate(shape):
            if s == 1 and grad.shape[i] != 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad.reshape(shape)

    # --- elementwise arithmetic ---------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad = self.grad + Tensor._unbroadcast(out.grad, self.data.shape)
            other.grad = other.grad + Tensor._unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad = self.grad + Tensor._unbroadcast(out.grad * other.data, self.data.shape)
            other.grad = other.grad + Tensor._unbroadcast(out.grad * self.data, other.data.shape)

        out._backward = _backward
        return out

    def __pow__(self, p: float):
        assert isinstance(p, (int, float)), "only scalar powers supported"
        out = Tensor(self.data ** p, (self,), f"**{p}")

        def _backward():
            self.grad = self.grad + (p * self.data ** (p - 1)) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return (-self) + other

    # --- matrix multiply -----------------------------------------------------
    def matmul(self, other: "Tensor") -> "Tensor":
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            self.grad = self.grad + out.grad @ np.swapaxes(other.data, -1, -2)
            other.grad = other.grad + np.swapaxes(self.data, -1, -2) @ out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    # --- nonlinearities ------------------------------------------------------
    def relu(self) -> "Tensor":
        out = Tensor(np.maximum(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad = self.grad + (self.data > 0.0) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        t = np.tanh(self.data)
        out = Tensor(t, (self,), "tanh")

        def _backward():
            self.grad = self.grad + (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        # numerically stable logistic
        s = np.where(
            self.data >= 0,
            1.0 / (1.0 + np.exp(-np.clip(self.data, -60, 60))),
            np.exp(np.clip(self.data, -60, 60)) / (1.0 + np.exp(np.clip(self.data, -60, 60))),
        )
        out = Tensor(s, (self,), "sigmoid")

        def _backward():
            self.grad = self.grad + (s * (1.0 - s)) * out.grad

        out._backward = _backward
        return out

    # --- reductions ----------------------------------------------------------
    def sum(self) -> "Tensor":
        out = Tensor(self.data.sum(), (self,), "sum")

        def _backward():
            self.grad = self.grad + np.ones_like(self.data) * out.grad

        out._backward = _backward
        return out

    def mean(self) -> "Tensor":
        n = self.data.size
        out = Tensor(self.data.mean(), (self,), "mean")

        def _backward():
            self.grad = self.grad + (np.ones_like(self.data) / n) * out.grad

        out._backward = _backward
        return out

    # --- the backward sweep --------------------------------------------------
    def backward(self):
        """Topologically sort the graph and propagate gradients from this scalar."""
        topo: List[Tensor] = []
        seen = set()

        def build(v: "Tensor"):
            if v not in seen:
                seen.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        # seed: dLoss/dLoss = 1
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, op={self._op!r})"


# --- fused, numerically stable losses ---------------------------------------
# These are implemented as single autograd ops (rather than chains of exp/log)
# both for numerical stability and so the gradients are exact and cheap.

def softmax_cross_entropy(logits: Tensor, labels: np.ndarray) -> Tensor:
    """Mean softmax cross-entropy for integer class labels.

    logits : Tensor of shape (N, C)
    labels : np.ndarray of shape (N,) with integer class ids in [0, C)
    returns: scalar Tensor (the mean loss)
    """
    z = logits.data
    z = z - z.max(axis=1, keepdims=True)          # stability shift
    exp = np.exp(z)
    probs = exp / exp.sum(axis=1, keepdims=True)
    n = labels.shape[0]
    nll = -np.log(probs[np.arange(n), labels] + 1e-12).mean()
    out = Tensor(nll, (logits,), "softmax_ce")

    def _backward():
        grad = probs.copy()
        grad[np.arange(n), labels] -= 1.0          # softmax - one_hot
        grad /= n
        logits.grad = logits.grad + grad * out.grad

    out._backward = _backward
    return out


def sigmoid_bce(logits: Tensor, labels: np.ndarray) -> Tensor:
    """Mean binary cross-entropy with logits (numerically stable).

    logits : Tensor of shape (N, 1) or (N,)
    labels : np.ndarray of {0,1} broadcastable to logits.data
    """
    z = logits.data.reshape(-1)
    y = labels.reshape(-1).astype(np.float64)
    # stable form: max(z,0) - z*y + log(1+exp(-|z|))
    loss = np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))
    out = Tensor(loss.mean(), (logits,), "sigmoid_bce")
    s = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))

    def _backward():
        grad = ((s - y) / y.shape[0]).reshape(logits.data.shape)
        logits.grad = logits.grad + grad * out.grad

    out._backward = _backward
    return out


# =============================================================================
# PART II — PARAMETERS, INITIALIZATION, AND THE GOVERNED NEURON
# =============================================================================


class Parameter(Tensor):
    """A Tensor that the optimizer is allowed to update (a learnable weight)."""
    pass


def he_init(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    """He/Kaiming initialization — good default for ReLU-style activations."""
    return rng.standard_normal((fan_in, fan_out)) * math.sqrt(2.0 / fan_in)


def xavier_init(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    """Xavier/Glorot initialization — good default for tanh/sigmoid gates."""
    return rng.standard_normal((fan_in, fan_out)) * math.sqrt(1.0 / fan_in)


class GovernedLayer:
    """The GOVERNED NEURON, applied as a layer.

    This is the architectural heart of the Ptahhotep mind. Where a standard
    dense layer computes a single response:

        h = activation(x @ W + b)

    a *governed* layer computes TWO things and lets one rule the other:

        impulse  = x @ W_i + b_i           # the fast, reactive response
        governor = sigmoid(x @ W_g + b_g)  # the deliberative "should I?" gate
        h        = activation(impulse) * governor

    The governor is a learned, per-neuron gate in (0, 1). It can damp an
    impulse to silence (gate -> 0) or let it pass in full (gate -> 1). This is
    Ptahhotep's central maxim made mechanical: *the deliberative faculty governs
    the impulses.* Mathematically it is a gated unit (cousin of the GLU and the
    highway/LSTM gate), but here the gate is a first-class, inspectable quantity
    so that the Ma'at moderation term can act on it directly.
    """

    def __init__(self, rng, in_dim: int, out_dim: int, name: str = ""):
        self.name = name
        self.W_i = Parameter(he_init(rng, in_dim, out_dim))     # impulse weights
        self.b_i = Parameter(np.zeros(out_dim))
        self.W_g = Parameter(xavier_init(rng, in_dim, out_dim))  # governor weights
        # bias the governor slightly open (+0.5) so learning starts from a
        # "willing to speak, but able to restrain" disposition rather than mute.
        self.b_g = Parameter(np.full(out_dim, 0.5))
        self.last_gate: Optional[Tensor] = None                  # cached for Ma'at term

    def params(self) -> List[Parameter]:
        return [self.W_i, self.b_i, self.W_g, self.b_g]

    def __call__(self, x: Tensor) -> Tensor:
        impulse = (x @ self.W_i + self.b_i).relu()
        gate = (x @ self.W_g + self.b_g).sigmoid()
        self.last_gate = gate
        return impulse * gate


class Linear:
    """A plain affine layer y = x @ W + b (used for the read-out heads)."""

    def __init__(self, rng, in_dim: int, out_dim: int, init="xavier", name=""):
        self.name = name
        W0 = he_init(rng, in_dim, out_dim) if init == "he" else xavier_init(rng, in_dim, out_dim)
        self.W = Parameter(W0)
        self.b = Parameter(np.zeros(out_dim))

    def params(self) -> List[Parameter]:
        return [self.W, self.b]

    def __call__(self, x: Tensor) -> Tensor:
        return x @ self.W + self.b


def maat_moderation(gates: List[Tensor]) -> Tensor:
    """The golden-mean regularizer.

    Ptahhotep warns repeatedly against the extremes — too much and too little.
    A governor gate pinned at 0 or 1 is "extreme": it has stopped deliberating
    and become a habit. We gently penalize gates for drifting away from the
    balanced mid-band by minimizing the mean of (gate - 0.5)^2.

    Kept at a small weight (lambda) this behaves as a saturation regularizer:
    it preserves gradient flow through the gates and improves generalization,
    while still letting a gate commit when the task truly demands it.
    """
    terms = []
    for g in gates:
        centered = g + (-0.5)
        terms.append((centered * centered).mean())
    total = terms[0]
    for t in terms[1:]:
        total = total + t
    return total * (1.0 / len(terms))


# =============================================================================
# PART III — TEMPORAL MEMORY ("wisdom comes with age")
# =============================================================================
#
# Ptahhotep insists that wisdom is not innate but accumulated: the old judge
# better because they have *seen more cases*. We model this with a wholly
# non-parametric memory that sits beside the trainable network. It keeps one
# running prototype (an exponential moving average of feature vectors) per
# speech-act class. Classification is nearest-prototype. Crucially, its accuracy
# is a function of how many episodes it has lived through — it literally gets
# wiser with experience, with no gradient descent at all.
# -----------------------------------------------------------------------------


class TemporalMemory:
    """Online prototype memory: accuracy improves as it observes more episodes."""

    def __init__(self, n_classes: int, dim: int, ema: float = 0.1):
        self.n_classes = n_classes
        self.dim = dim
        self.ema = ema                                  # how fast a prototype adapts
        self.prototypes = np.zeros((n_classes, dim))
        self.seen = np.zeros(n_classes, dtype=int)      # episodes per class
        self.total_episodes = 0

    def observe(self, feature: np.ndarray, label: int) -> None:
        """Live through one episode: refine the prototype for `label`."""
        self.total_episodes += 1
        self.seen[label] += 1
        if self.seen[label] == 1:
            self.prototypes[label] = feature            # first impression
        else:
            p = self.prototypes[label]
            self.prototypes[label] = (1 - self.ema) * p + self.ema * feature

    def recall(self, feature: np.ndarray) -> int:
        """Judge a new case by the closest remembered prototype."""
        active = self.seen > 0
        if not active.any():
            return 0
        dists = np.full(self.n_classes, np.inf)
        diff = self.prototypes[active] - feature
        dists[active] = np.einsum("ij,ij->i", diff, diff)
        return int(np.argmin(dists))


# =============================================================================
# PART IV — THE FULL MIND: PtahhotepMind
# =============================================================================
#
# Wiring:
#
#   input  (utterance + social context)
#     -> GovernedLayer  (in_dim -> hidden)      "first deliberation"
#     -> GovernedLayer  (hidden -> hidden)      "second deliberation"
#     -> shared representation z
#         |-> SpeechActHead   (z -> 8 logits)   what KIND of speech is this?
#         |-> GovernanceGate  (z -> 1 logit)    should it be spoken at all?
#
# The two governed layers expose their gates so the Ma'at term can moderate them.
# -----------------------------------------------------------------------------

SPEECH_ACTS = [
    "assertive",     # states a fact (true/false)
    "directive",     # commands, requests
    "commissive",    # promises, pledges
    "expressive",    # thanks, apology, feeling
    "declarative",   # changes reality by being said ("you are appointed")
    "interrogative", # questions
    "exhortative",   # advice, warning
    "permissive",    # grants/denies permission
]
N_ACTS = len(SPEECH_ACTS)


class PtahhotepMind:
    """A small, complete, trainable mind embodying Ptahhotep's four maxims."""

    def __init__(self, in_dim: int, hidden: int = 64, seed: int = 4):
        rng = np.random.default_rng(seed)
        self.gov1 = GovernedLayer(rng, in_dim, hidden, "deliberation_1")
        self.gov2 = GovernedLayer(rng, hidden, hidden, "deliberation_2")
        self.act_head = Linear(rng, hidden, N_ACTS, init="xavier", name="speech_act")
        self.gate_head = Linear(rng, hidden, 1, init="xavier", name="governance")
        self.memory = TemporalMemory(N_ACTS, hidden, ema=0.08)
        self.hidden = hidden

    def params(self) -> List[Parameter]:
        ps: List[Parameter] = []
        for layer in (self.gov1, self.gov2, self.act_head, self.gate_head):
            ps.extend(layer.params())
        return ps

    def representation(self, x: Tensor) -> Tensor:
        """Two stages of governed deliberation -> shared hidden representation z."""
        h1 = self.gov1(x)
        z = self.gov2(h1)
        return z

    def forward(self, x: Tensor):
        """Return (speech_act_logits, governance_logit, z, gates)."""
        z = self.representation(x)
        act_logits = self.act_head(z)
        gov_logit = self.gate_head(z)
        gates = [self.gov1.last_gate, self.gov2.last_gate]
        return act_logits, gov_logit, z, gates

    def loss(self, x: Tensor, y_act: np.ndarray, y_gov: np.ndarray,
             lam_moderation: float = 0.01) -> Tuple[Tensor, Dict[str, float]]:
        """Combined objective:

            L = CE(speech_act) + BCE(speak/silence) + lambda * Ma'at(gates)

        The CE term teaches the mind to read intent; the BCE term teaches it the
        moral consequence of speaking; the Ma'at term keeps its gates balanced.
        """
        act_logits, gov_logit, z, gates = self.forward(x)
        l_act = softmax_cross_entropy(act_logits, y_act)
        l_gov = sigmoid_bce(gov_logit, y_gov)
        l_mod = maat_moderation(gates)
        total = l_act + l_gov + lam_moderation * l_mod
        parts = {
            "speech_act_ce": float(l_act.data),
            "governance_bce": float(l_gov.data),
            "maat_moderation": float(l_mod.data),
            "total": float(total.data),
        }
        return total, parts


# =============================================================================
# PART V — THE ADAM OPTIMIZER (real, bias-corrected)
# =============================================================================


class Adam:
    """Adam optimizer with bias correction and optional gradient clipping."""

    def __init__(self, params: List[Parameter], lr: float = 3e-3,
                 betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 clip: float = 5.0):
        self.params = params
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.clip = clip
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data)

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            if self.clip:
                norm = np.sqrt((g * g).sum())
                if norm > self.clip:
                    g = g * (self.clip / (norm + 1e-12))
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            p.data = p.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# =============================================================================
# PART VI — THE WORLD: a structured "speech-in-context" dataset
# =============================================================================
#
# We synthesize a small but causally structured world. Every example is one
# utterance described by:
#
#   * an INTENT (one of 8 speech-acts), encoded as a noisy point near a fixed
#     per-class centroid in a 16-d embedding space, and
#   * a SOCIAL CONTEXT of 8 interpretable features in [0,1]:
#         c0 speaker_status   c1 hearer_status    c2 formality   c3 emotion
#         c4 truthfulness     c5 proportionality  c6 timing      c7 prior_harmony
#
# TWO ground-truth labels are derived:
#   y_act : the intent class (recoverable from the embedding, with noise)
#   y_gov : SPEAK (1) or STAY-SILENT (0), from a non-linear "harmony" function
#           that rewards truthful, proportionate, well-timed speech and punishes
#           disproportionate, emotionally-charged commands aimed upward in formal
#           settings — exactly the failures Ptahhotep's maxims warn against.
# -----------------------------------------------------------------------------

# Per-intent risk: how easily this kind of speech damages harmony if misused.
INTENT_RISK = np.array([0.20, 0.85, 0.55, 0.60, 0.80, 0.25, 0.45, 0.50])
# Which intents carry "downward authority" that stings when aimed upward.
DIRECTIVE_FLAG = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.6, 0.4])


def make_dataset(n: int, seed: int = 0):
    """Generate (X, y_act, y_gov) with genuine latent structure.

    X has shape (n, 24): [16-d intent embedding | 8-d social context].
    """
    rng = np.random.default_rng(seed)
    emb_dim = 16
    # Centroids are deliberately close and the per-utterance noise is large, so
    # recognizing intent is a genuine learning problem (not a lookup): classes
    # overlap and the mind must find a separating representation.
    centroids = np.random.default_rng(777).standard_normal((N_ACTS, emb_dim)) * 1.1

    y_act = rng.integers(0, N_ACTS, size=n)
    intent_emb = centroids[y_act] + rng.standard_normal((n, emb_dim)) * 1.05

    ctx = rng.random((n, 8))
    c = {k: ctx[:, i] for i, k in enumerate(
        ["spk", "hear", "form", "emo", "truth", "prop", "time", "prior"])}

    risk = INTENT_RISK[y_act]
    upward = np.maximum(c["hear"] - c["spk"], 0.0)        # commanding a superior
    directive = DIRECTIVE_FLAG[y_act]

    harmony = (
        1.10 * (c["truth"] - 0.5)                          # honesty helps
        + 1.20 * (c["prop"] - 0.5)                          # proportion helps
        + 0.60 * (c["time"] - 0.5)                          # good timing helps
        + 0.50 * (c["prior"] - 0.5)                         # starting from peace helps
        - 1.30 * risk * (1.0 - c["prop"])                   # risky + disproportionate hurts
        - 1.40 * c["emo"] * c["form"]                       # emotion in formal settings hurts
        - 1.60 * upward * directive                         # commanding upward hurts
    )
    harmony += rng.standard_normal(n) * 0.10                # irreducible noise
    y_gov = (harmony > 0.0).astype(np.int64)

    X = np.concatenate([intent_emb, ctx], axis=1)
    return X.astype(np.float64), y_act.astype(np.int64), y_gov.astype(np.int64)


# =============================================================================
# PART VII — TRAINING & EVALUATION
# =============================================================================


def iterate_minibatches(X, y_act, y_gov, batch_size, rng):
    idx = rng.permutation(len(X))
    for start in range(0, len(X), batch_size):
        b = idx[start:start + batch_size]
        yield X[b], y_act[b], y_gov[b]


def evaluate(mind: PtahhotepMind, X, y_act, y_gov):
    """Return (speech_act_accuracy, governance_accuracy) on a dataset."""
    act_logits, gov_logit, _, _ = mind.forward(Tensor(X))
    act_pred = act_logits.data.argmax(axis=1)
    gov_pred = (gov_logit.data.reshape(-1) > 0.0).astype(np.int64)
    return float((act_pred == y_act).mean()), float((gov_pred == y_gov).mean())


def train(mind: PtahhotepMind, data, epochs=40, batch_size=64, lr=3e-3,
          lam_moderation=0.01, seed=4, verbose=True):
    Xtr, yatr, ygtr, Xte, yate, ygte = data
    opt = Adam(mind.params(), lr=lr)
    rng = np.random.default_rng(seed)
    history = {"loss": [], "test_act_acc": [], "test_gov_acc": []}

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        nb = 0
        for xb, yab, ygb in iterate_minibatches(Xtr, yatr, ygtr, batch_size, rng):
            total, _ = mind.loss(Tensor(xb), yab, ygb, lam_moderation)
            opt.zero_grad()
            total.backward()
            opt.step()
            epoch_loss += float(total.data)
            nb += 1
        epoch_loss /= nb
        ta, tg = evaluate(mind, Xte, yate, ygte)
        history["loss"].append(epoch_loss)
        history["test_act_acc"].append(ta)
        history["test_gov_acc"].append(tg)
        if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == epochs):
            print(f"  epoch {epoch:3d} | loss {epoch_loss:6.4f} | "
                  f"speech-act acc {ta*100:5.1f}% | governance acc {tg*100:5.1f}%")
    return history


def build_data(n_train=2400, n_test=600):
    Xtr, yatr, ygtr = make_dataset(n_train, seed=1)
    Xte, yate, ygte = make_dataset(n_test, seed=2)
    return Xtr, yatr, ygtr, Xte, yate, ygte


# =============================================================================
# PART VIII — THE "WISDOM COMES WITH AGE" EXPERIMENT
# =============================================================================


def wisdom_with_age(mind: PtahhotepMind, data, checkpoints=(10, 50, 200, 800, 2000)):
    """Feed episodes to the TemporalMemory one at a time, measuring how its
    recall accuracy on a held-out set grows as it accumulates experience.

    The memory operates on the *learned* hidden representation z, so it inherits
    the network's perception but adds purely experiential, gradient-free wisdom.
    """
    Xtr, yatr, _, Xte, yate, _ = data
    z_tr = mind.representation(Tensor(Xtr)).data
    z_te = mind.representation(Tensor(Xte)).data
    results = []
    cp = set(checkpoints)
    for i in range(len(z_tr)):
        mind.memory.observe(z_tr[i], int(yatr[i]))
        if (i + 1) in cp:
            preds = np.array([mind.memory.recall(z) for z in z_te])
            acc = float((preds == yate).mean())
            results.append((i + 1, acc))
    return results


# =============================================================================
# PART IX — AUTOMATED TEST SUITE
# =============================================================================
#
# "The student who truly hears becomes a master in turn." A mind that claims to
# learn must be testable. These tests assert (1) the calculus is correct and
# (2) the mind actually gets better.
# -----------------------------------------------------------------------------


def _numeric_grad(f, x, eps=1e-6):
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        fp = f(x)
        x[idx] = old - eps
        fm = f(x)
        x[idx] = old
        g[idx] = (fp - fm) / (2 * eps)
        it.iternext()
    return g


def test_autograd_matmul_relu():
    rng = np.random.default_rng(0)
    Xn = rng.standard_normal((5, 4))
    Wn = rng.standard_normal((4, 3))

    def loss_of_W(Wv):
        return float((np.maximum(0, Xn @ Wv) ** 2).sum())

    W = Tensor(Wn.copy())
    out = ((Tensor(Xn) @ W).relu()) ** 2
    out.sum().backward()
    num = _numeric_grad(loss_of_W, Wn.copy())
    err = np.max(np.abs(num - W.grad))
    assert err < 1e-5, f"matmul/relu grad error {err}"
    return err


def test_softmax_ce_grad():
    rng = np.random.default_rng(1)
    Zn = rng.standard_normal((6, 4))
    labels = rng.integers(0, 4, size=6)

    def loss_of_Z(Zv):
        z = Zv - Zv.max(axis=1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        return float(-np.log(p[np.arange(6), labels] + 1e-12).mean())

    Z = Tensor(Zn.copy())
    softmax_cross_entropy(Z, labels).backward()
    num = _numeric_grad(loss_of_Z, Zn.copy())
    err = np.max(np.abs(num - Z.grad))
    assert err < 1e-5, f"softmax-CE grad error {err}"
    return err


def test_sigmoid_bce_grad():
    rng = np.random.default_rng(2)
    Zn = rng.standard_normal((7, 1))
    labels = rng.integers(0, 2, size=7).astype(np.float64)

    def loss_of_Z(Zv):
        z = Zv.reshape(-1)
        return float((np.maximum(z, 0) - z * labels + np.log1p(np.exp(-np.abs(z)))).mean())

    Z = Tensor(Zn.copy())
    sigmoid_bce(Z, labels).backward()
    num = _numeric_grad(loss_of_Z, Zn.copy())
    err = np.max(np.abs(num - Z.grad))
    assert err < 1e-5, f"sigmoid-BCE grad error {err}"
    return err


def test_governor_gate_in_range():
    rng = np.random.default_rng(3)
    layer = GovernedLayer(rng, 6, 5)
    layer(Tensor(rng.standard_normal((10, 6))))
    g = layer.last_gate.data
    assert (g > 0).all() and (g < 1).all(), "governor gate left (0,1)"
    return float(g.mean())


def test_training_reduces_loss():
    data = build_data(n_train=1600, n_test=400)
    mind = PtahhotepMind(in_dim=24, hidden=64, seed=4)
    hist = train(mind, data, epochs=25, verbose=False)
    assert hist["loss"][-1] < hist["loss"][0] * 0.6, "loss did not fall enough"
    assert hist["test_act_acc"][-1] > 0.80, "speech-act acc too low"
    assert hist["test_gov_acc"][-1] > 0.78, "governance acc too low"
    return hist


def test_wisdom_grows():
    data = build_data(n_train=2000, n_test=400)
    mind = PtahhotepMind(in_dim=24, hidden=64, seed=4)
    train(mind, data, epochs=12, verbose=False)
    curve = wisdom_with_age(mind, data, checkpoints=(10, 100, 2000))
    early = curve[0][1]
    late = curve[-1][1]
    assert late > early, "temporal memory did not improve with experience"
    return curve


def test_determinism():
    d1 = build_data(800, 200)
    d2 = build_data(800, 200)
    m1 = PtahhotepMind(24, 48, seed=4)
    m2 = PtahhotepMind(24, 48, seed=4)
    h1 = train(m1, d1, epochs=6, verbose=False)
    h2 = train(m2, d2, epochs=6, verbose=False)
    assert abs(h1["loss"][-1] - h2["loss"][-1]) < 1e-9, "non-deterministic training"
    return True


def run_tests():
    print("\n" + "=" * 70)
    print("RUNNING TEST SUITE")
    print("=" * 70)
    checks = [
        ("autograd: matmul+relu+pow gradient", test_autograd_matmul_relu),
        ("autograd: softmax cross-entropy gradient", test_softmax_ce_grad),
        ("autograd: sigmoid BCE gradient", test_sigmoid_bce_grad),
        ("governor gate stays in (0,1)", test_governor_gate_in_range),
        ("training reduces loss & learns both tasks", test_training_reduces_loss),
        ("temporal memory grows wiser with experience", test_wisdom_grows),
        ("training is deterministic under fixed seed", test_determinism),
    ]
    passed = 0
    for name, fn in checks:
        t0 = time.time()
        try:
            result = fn()
            dt = time.time() - t0
            extra = ""
            if isinstance(result, float):
                extra = f"  (value={result:.3e})"
            print(f"  PASS  {name}{extra}   [{dt:4.1f}s]")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
        except Exception as e:  # pragma: no cover
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print("-" * 70)
    print(f"  {passed}/{len(checks)} tests passed")
    print("=" * 70)
    return passed == len(checks)


# =============================================================================
# PART X — MAIN: a full training run + report
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="The Ptahhotep Architecture (PTH-1)")
    parser.add_argument("--test", action="store_true", help="run test-suite only")
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()

    if args.test:
        ok = run_tests()
        sys.exit(0 if ok else 1)

    print("#" * 70)
    print("# THE PTAHHOTEP ARCHITECTURE (PTH-1)")
    print("# 'The deliberative mind governs the impulses; speech is a moral act;")
    print("#  seek the mean; wisdom comes with age.'")
    print("#" * 70)

    print("\n[1] Building the world (structured speech-in-context dataset)...")
    data = build_data(n_train=2400, n_test=600)
    Xtr = data[0]
    print(f"    train={len(data[0])}  test={len(data[3])}  input_dim={Xtr.shape[1]}  "
          f"speech-acts={N_ACTS}")

    print("\n[2] Instantiating the mind...")
    mind = PtahhotepMind(in_dim=Xtr.shape[1], hidden=64, seed=4)
    n_params = sum(p.data.size for p in mind.params())
    print(f"    governed layers: 2   trainable parameters: {n_params:,}")

    print("\n[3] Training (Adam, Ma'at-moderated multi-task loss)...")
    hist = train(mind, data, epochs=args.epochs, verbose=True)

    print("\n[4] Final evaluation on held-out cases:")
    ta, tg = evaluate(mind, data[3], data[4], data[5])
    print(f"    speech-act recognition accuracy : {ta*100:5.1f}%  (chance {100/N_ACTS:.1f}%)")
    print(f"    speak-vs-silence accuracy        : {tg*100:5.1f}%  (chance ~50%)")

    print("\n[5] 'Wisdom comes with age' — non-parametric memory vs. experience:")
    curve = wisdom_with_age(mind, data)
    for episodes, acc in curve:
        bar = "#" * int(acc * 40)
        print(f"    after {episodes:5d} episodes: {acc*100:5.1f}% |{bar}")

    print("\n[6] Governor balance (Ma'at) — mean gate openness per layer:")
    mind.forward(Tensor(data[3]))
    for layer in (mind.gov1, mind.gov2):
        g = layer.last_gate.data
        print(f"    {layer.name:16s}: mean={g.mean():.3f}  std={g.std():.3f}  "
              f"(0=silent, 1=unrestrained)")

    run_tests()

    print("\nDone. The vizier wrote on papyrus; this mind writes in gradients.\n")


if __name__ == "__main__":
    main()

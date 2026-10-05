#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 280 -- Peter Abelard (c.1079-1142)
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0280 · Peter Abelard (c.1079-1142)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 14, Minds 261-280 Available on Amazon https://www.amazon.com/dp/B0HLYSVY5D
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0280_peter_abelard_1079 - Peter Abelard (c.1079-1142)
==========================================================
A from-scratch (pure NumPy) trainable architecture built around three
of Abelard's own, textually-grounded commitments, rather than around
a generic transformer/MoE default:

  (A) PARTICULAR + ATTENTIONAL-STANCE ENCODER
      Abelard's answer to the problem of universals: reality contains
      only particulars; a "universal" is not a stored shared object but
      the product of a common, repeatable ACT applied to many different
      particulars (his "sermo"/imposition theory, Logica Ingredientibus).
      Concretely: one raw vector per particular, no privileged shared
      subspace; a small bank of trainable "attentional stances" (his
      *attentio*) gate which attribute-block of a particular reaches a
      shared readout, and a stance-specific head reads out a class.
      Applying different stances to the *same, unchanged* particular is
      meant to reproduce Abelard's own example in his philosophy-of-mind
      writing (Tractatus de Intellectibus / SEP sec.5): the same mental
      image of a fig tree can, without altering by one bit, support the
      thought "this tree", "trees in general", or "my lost love who sat
      here" -- intentionality lives in the act of attending, not in any
      content stored in the image.

  (B) SIC ET NON DIALECTICAL RESOLVER
      Abelard's theological method (Sic et Non, Prologue): given two
      authorities that seem to contradict each other, first ask whether
      this is EQUIVOCATION (same term, different sense -- resolvable by
      a learned "distinctio" operator) or a genuine, unresolved
      contradiction that should be flagged and left open (his
      "sententia recepta" -- refusing premature closure) rather than
      silently averaged into an anonymous compromise nobody actually
      asserted.

  (C) RELEVANCE-GATED ENTAILMENT
      Abelard's theory of *inferentia* (Dialectica): a valid argument
      needs BOTH formal/structural necessity AND relevance -- the sense
      of the conclusion must be genuinely contained in the sense of the
      premises, not merely co-occurrent with it. The module accepts an
      inference only when a structural check and an independently
      learned relevance check both pass, which blocks two different
      failure modes that fool only one check apiece.

  (D) CONSENT GATE (Ethica seu Scito te ipsum)
      Abelard's ethics of intention: a raw impulse/desire ("motus") is
      morally weightless on its own; only a separate, reflexive act of
      CONSENT to that impulse carries moral weight (his monk-in-chains
      example: involuntary pleasure without consent is no fault at
      all). The gate is trained to track a reciprocity ("Golden Rule")
      criterion in the scenario, explicitly NOT the raw impulse
      magnitude, and a downstream "moral loss" is multiplicatively
      gated by consent so that high-impulse/no-consent cases contribute
      approximately nothing to it.

Implementation notes
---------------------
* Pure NumPy. A small reverse-mode autodifferentiation engine (a
  vectorised "Tensor" with .backward()) is written from scratch below;
  no autodiff / deep-learning library is used anywhere in this file.
* Every one of the four modules is checked with a finite-difference
  numerical gradient against the analytic (autodiff) gradient before
  any training happens. This is a hard requirement, not a courtesy: if
  the check fails the script raises and refuses to proceed to training.
* All data is synthetic but procedurally generated to encode the exact
  logical structure of Abelard's own thought experiments (see each
  section below for the generative story). Random seeds are fixed so
  the run is reproducible.
* This file is self-contained and executable top to bottom:
      python3 chapter_0280_peter_abelard_1079.py
  runs the gradient check, trains all four modules, runs the self-test
  suite, and prints a full report.
"""

from __future__ import annotations
import numpy as np

RNG_SEED = 235
np.random.seed(RNG_SEED)

# ============================================================================
# SECTION 0 -- MINIMAL REVERSE-MODE AUTODIFF ENGINE (from scratch)
# ============================================================================

def unbroadcast(grad, shape):
    """Sum-reduce `grad` down to `shape`, undoing NumPy broadcasting."""
    grad = np.asarray(grad, dtype=np.float64)
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    """A NumPy array with reverse-mode autodiff bookkeeping."""

    __slots__ = ("data", "grad", "requires_grad", "_backward", "_prev", "shape")

    def __init__(self, data, requires_grad=False, _children=()):
        self.data = np.asarray(data, dtype=np.float64)
        self.shape = self.data.shape
        self.requires_grad = requires_grad or any(getattr(c, "requires_grad", False) for c in _children)
        self.grad = np.zeros_like(self.data) if self.requires_grad else None
        self._backward = lambda: None
        self._prev = _children

    # ---------------- elementwise / linear algebra ----------------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, _children=(self, other))
        def _backward():
            if self.requires_grad:
                self.grad += unbroadcast(out.grad, self.shape)
            if other.requires_grad:
                other.grad += unbroadcast(out.grad, other.shape)
        out._backward = _backward
        return out
    __radd__ = __add__

    def __neg__(self):
        out = Tensor(-self.data, _children=(self,))
        def _backward():
            if self.requires_grad:
                self.grad += unbroadcast(-out.grad, self.shape)
        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __rsub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, _children=(self, other))
        def _backward():
            if self.requires_grad:
                self.grad += unbroadcast(out.grad * other.data, self.shape)
            if other.requires_grad:
                other.grad += unbroadcast(out.grad * self.data, other.shape)
        out._backward = _backward
        return out
    __rmul__ = __mul__

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, _children=(self, other))
        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    def sum(self):
        out = Tensor(np.array(self.data.sum()), _children=(self,))
        def _backward():
            if self.requires_grad:
                self.grad += np.ones_like(self.data) * out.grad
        out._backward = _backward
        return out

    def mean(self):
        n = self.data.size
        out = Tensor(np.array(self.data.mean()), _children=(self,))
        def _backward():
            if self.requires_grad:
                self.grad += np.ones_like(self.data) * out.grad / n
        out._backward = _backward
        return out

    def backward(self):
        topo, visited = [], set()
        def build(v):
            if id(v) not in visited:
                visited.add(id(v))
                for p in v._prev:
                    build(p)
                topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()


def relu(t):
    mask = (t.data > 0).astype(np.float64)
    out = Tensor(t.data * mask, _children=(t,))
    def _backward():
        if t.requires_grad:
            t.grad += out.grad * mask
    out._backward = _backward
    return out


def sigmoid(t):
    s = 1.0 / (1.0 + np.exp(-t.data))
    out = Tensor(s, _children=(t,))
    def _backward():
        if t.requires_grad:
            t.grad += out.grad * s * (1.0 - s)
    out._backward = _backward
    return out


def concat_cols(tensors):
    """Concatenate a list of (N, d_i) Tensors along axis=1."""
    arrs = [t.data for t in tensors]
    out = Tensor(np.concatenate(arrs, axis=1), _children=tuple(tensors))
    sizes = [a.shape[1] for a in arrs]
    def _backward():
        col = 0
        for t, w in zip(tensors, sizes):
            if t.requires_grad:
                t.grad += out.grad[:, col:col + w]
            col += w
    out._backward = _backward
    return out


def softmax_cross_entropy(logits, labels):
    """logits: Tensor (N,C). labels: int array (N,). Returns scalar Tensor loss
    and the (N,C) probability array (for evaluation, not part of the graph)."""
    z = logits.data - logits.data.max(axis=1, keepdims=True)
    ez = np.exp(z)
    probs = ez / ez.sum(axis=1, keepdims=True)
    n = logits.data.shape[0]
    eps = 1e-12
    logp = -np.log(probs[np.arange(n), labels] + eps)
    loss_val = logp.mean()
    out = Tensor(np.array(loss_val), _children=(logits,))
    def _backward():
        if logits.requires_grad:
            d = probs.copy()
            d[np.arange(n), labels] -= 1.0
            d /= n
            logits.grad += d * out.grad
    out._backward = _backward
    return out, probs


def bce_with_logits(logits, targets):
    """logits: Tensor (N,1) or (N,). targets: array same shape, in {0,1}."""
    z = logits.data
    p = 1.0 / (1.0 + np.exp(-z))
    eps = 1e-12
    t = targets.reshape(z.shape)
    loss_val = -(t * np.log(p + eps) + (1 - t) * np.log(1 - p + eps)).mean()
    out = Tensor(np.array(loss_val), _children=(logits,))
    n = z.size
    def _backward():
        if logits.requires_grad:
            logits.grad += ((p - t) / n) * out.grad
    out._backward = _backward
    return out, p


def masked_mse(pred, target, mask):
    """pred: Tensor (N,d). target: array (N,d). mask: array (N,1) of 0/1."""
    diff = pred.data - target
    denom = max(1.0, mask.sum() * pred.data.shape[1])
    loss_val = (mask * diff * diff).sum() / denom
    out = Tensor(np.array(loss_val), _children=(pred,))
    def _backward():
        if pred.requires_grad:
            pred.grad += (2.0 * mask * diff / denom) * out.grad
    out._backward = _backward
    return out


def param(shape, scale=None, name=""):
    fan_in = shape[0] if len(shape) > 1 else shape[0]
    s = scale if scale is not None else 1.0 / np.sqrt(max(1, fan_in))
    return Tensor(np.random.randn(*shape) * s, requires_grad=True)


class Adam:
    def __init__(self, params, lr=0.03, b1=0.9, b2=0.999, eps=1e-8):
        self.params = params
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            if p.requires_grad:
                p.grad = np.zeros_like(p.data)

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if not p.requires_grad:
                continue
            g = p.grad
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


# ============================================================================
# SECTION 1 -- MODULE A: PARTICULAR + ATTENTIONAL-STANCE ENCODER
# ============================================================================
# Data story: each "particular" is a 12-dim vector built from THREE 4-dim
# attribute blocks (species / generality-class / affective-association),
# each block a noisy one-hot over 4 possible values. No block is privileged
# as "the" universal; a trainable attentional stance (soft gate over the 12
# input dims) determines which block reaches the shared trunk, and a
# stance-specific head reads off that block's class. Applying stance k to a
# particular is Abelard's *attentio* selecting one "understanding" out of an
# otherwise unchanged image.

P_DIM = 12          # 3 blocks x 4
N_BLOCKS = 3
BLOCK = 4
A_HID = 16
N_STANCES = 3
BLOCK_NAMES = ["species", "generality-class", "affective-association"]


def gen_particulars(n, seed):
    rng = np.random.RandomState(seed)
    Y = rng.randint(0, BLOCK, size=(n, N_BLOCKS))
    X = np.zeros((n, P_DIM))
    for b in range(N_BLOCKS):
        onehot = np.zeros((n, BLOCK))
        onehot[np.arange(n), Y[:, b]] = 1.0
        X[:, b * BLOCK:(b + 1) * BLOCK] = onehot * 1.6 + rng.randn(n, BLOCK) * 0.35
    return X, Y


class ParticularAttentionEncoder:
    """Module A -- particulars are never pooled into a shared universal;
    stances gate which block of the SAME particular reaches the readout."""

    def __init__(self):
        self.gates = [param((1, P_DIM), scale=0.3) for _ in range(N_STANCES)]
        self.W1 = param((P_DIM, A_HID))
        self.b1 = Tensor(np.zeros((1, A_HID)), requires_grad=True)
        self.heads_W = [param((A_HID, BLOCK)) for _ in range(N_STANCES)]
        self.heads_b = [Tensor(np.zeros((1, BLOCK)), requires_grad=True) for _ in range(N_STANCES)]

    def parameters(self):
        ps = list(self.gates) + [self.W1, self.b1]
        ps += self.heads_W + self.heads_b
        return ps

    def forward(self, X, stance):
        x = Tensor(X)
        gate = sigmoid(self.gates[stance])          # (1, P_DIM) attentional stance
        gated = x * gate                             # same particular, different attending
        u = relu(gated @ self.W1 + self.b1)           # shared "understanding" trunk
        logits = u @ self.heads_W[stance] + self.heads_b[stance]
        return logits


# ============================================================================
# SECTION 2 -- MODULE B: SIC ET NON DIALECTICAL RESOLVER
# ============================================================================
# Data story: a hidden "base concept" b in R^8. Two cited "authorities"
# produce claim vectors c1, c2 tagged with which lexical sense (0/1) of the
# disputed term they use.
#   type 0 (consistent):    same b, same sense           -> ordinary agreement
#   type 1 (equivocal):     same b, DIFFERENT sense tags  -> apparent
#                            contradiction that a distinctio (learned
#                            "subtract the sense") can resolve back to b
#   type 2 (contradiction): opposite b, SAME sense tag    -> a genuine
#                            disagreement Abelard's own prologue says to
#                            leave open (sententia recepta) rather than force

C_DIM = 8
B_HID = 24
SENSE_SCALE = 1.35
NOISE = 0.22

_sense_vecs = np.random.RandomState(1).randn(2, C_DIM)
_sense_vecs /= np.linalg.norm(_sense_vecs, axis=1, keepdims=True)


def gen_sic_et_non(n, seed):
    rng = np.random.RandomState(seed)
    C1 = np.zeros((n, C_DIM)); C2 = np.zeros((n, C_DIM))
    S1 = np.zeros((n,), dtype=int); S2 = np.zeros((n,), dtype=int)
    TYPE = rng.randint(0, 3, size=n)
    B = rng.randn(n, C_DIM) * 0.9
    RES_TARGET = np.zeros((n, C_DIM))
    RES_MASK = np.zeros((n, 1))
    for i in range(n):
        b = B[i]
        t = TYPE[i]
        if t == 0:  # consistent: same sense both sides
            s1 = s2 = rng.randint(0, 2)
            c1 = b + _sense_vecs[s1] * (SENSE_SCALE * 0.15) + rng.randn(C_DIM) * NOISE
            c2 = b + _sense_vecs[s2] * (SENSE_SCALE * 0.15) + rng.randn(C_DIM) * NOISE
        elif t == 1:  # equivocal: same underlying b, different sense tags
            s1, s2 = 0, 1
            c1 = b + _sense_vecs[s1] * SENSE_SCALE + rng.randn(C_DIM) * NOISE
            c2 = b + _sense_vecs[s2] * SENSE_SCALE + rng.randn(C_DIM) * NOISE
            RES_TARGET[i] = b
            RES_MASK[i] = 1.0
        else:  # genuine contradiction: opposite content, SAME sense tag
            s1 = s2 = rng.randint(0, 2)
            c1 = b + _sense_vecs[s1] * (SENSE_SCALE * 0.15) + rng.randn(C_DIM) * NOISE
            c2 = -b + _sense_vecs[s2] * (SENSE_SCALE * 0.15) + rng.randn(C_DIM) * NOISE
        C1[i], C2[i], S1[i], S2[i] = c1, c2, s1, s2
    return C1, C2, S1, S2, TYPE, RES_TARGET, RES_MASK


def onehot(idx, k):
    out = np.zeros((len(idx), k))
    out[np.arange(len(idx)), idx] = 1.0
    return out


class SicEtNonResolver:
    """Module B -- classify equivocation vs genuine contradiction before
    ever attempting synthesis; abstain (flag as open) on the latter."""

    def __init__(self):
        in_dim = C_DIM * 2 + 2 + 2
        self.W1 = param((in_dim, B_HID))
        self.b1 = Tensor(np.zeros((1, B_HID)), requires_grad=True)
        self.W_type = param((B_HID, 3))
        self.b_type = Tensor(np.zeros((1, 3)), requires_grad=True)
        self.W_res = param((B_HID, C_DIM))
        self.b_res = Tensor(np.zeros((1, C_DIM)), requires_grad=True)

    def parameters(self):
        return [self.W1, self.b1, self.W_type, self.b_type, self.W_res, self.b_res]

    def forward(self, C1, C2, S1, S2):
        inp = concat_cols([Tensor(C1), Tensor(C2), Tensor(onehot(S1, 2)), Tensor(onehot(S2, 2))])
        h = relu(inp @ self.W1 + self.b1)
        type_logits = h @ self.W_type + self.b_type
        resolution = h @ self.W_res + self.b_res
        return type_logits, resolution


# ============================================================================
# SECTION 3 -- MODULE C: RELEVANCE-GATED ENTAILMENT
# ============================================================================
# Data story: 4 topic centres in R^8. A fixed, DELIBERATELY LOSSY (rank-3)
# transform R represents Abelard's "complete" (formal, substitution-driven)
# entailment check. R's null space is engineered to contain (T1 - T0), which
# lets us construct a case where two topically UNRELATED premises collide
# under R -- i.e. a candidate conclusion can look structurally valid purely
# by formal coincidence. A separately-LEARNED relevance term (bilinear
# overlap) is required to catch that case, and a separate structural term is
# required to catch the mirror case (topically relevant, structurally
# invalid). Only the conjunction of both should be accepted -- Abelard's
# explicit requirement that entailment needs necessity AND relevance.

E_DIM = 8
N_TOPICS = 4
_topics = np.random.RandomState(2).randn(N_TOPICS, E_DIM)
_topics /= np.linalg.norm(_topics, axis=1, keepdims=True)
_topics *= 1.4

_diff01 = _topics[1] - _topics[0]
_diff01_unit = _diff01 / np.linalg.norm(_diff01)


def _make_R():
    """Rank-3 matrix with (topic1 - topic0) in its null space."""
    rng = np.random.RandomState(3)
    dirs = []
    while len(dirs) < 3:
        v = rng.randn(E_DIM)
        v = v - _diff01_unit * (v @ _diff01_unit)   # orthogonalize to the null direction
        for d in dirs:
            v = v - d * (v @ d)
        n = np.linalg.norm(v)
        if n > 1e-6:
            dirs.append(v / n)
    R = np.zeros((E_DIM, E_DIM))
    for d in dirs:
        R += np.outer(d, d)
    return R * 0.9


_R = _make_R()
assert np.linalg.norm(_R @ _diff01) < 1e-8, "R must annihilate (topic1 - topic0) by construction"


def gen_entailment(n, seed):
    rng = np.random.RandomState(seed)
    P = np.zeros((n, E_DIM)); Q = np.zeros((n, E_DIM))
    LABEL = np.zeros((n, 1))
    CAT = np.zeros(n, dtype=int)   # 0 valid, 1 vacuous(structural false+), 2 irrelevant-invalid(relevance false+)
    for i in range(n):
        cat = rng.randint(0, 3)
        t_p = rng.randint(0, N_TOPICS)
        p = _topics[t_p] + rng.randn(E_DIM) * 0.18
        if cat == 0:  # valid + relevant -> accept
            q = _R @ p + rng.randn(E_DIM) * 0.10
            label = 1.0
        elif cat == 1:  # structurally-fooled but topically irrelevant -> reject
            # only meaningful when topic0/topic1 collide under R; use that pair directly
            t_p = 0
            p = _topics[0] + rng.randn(E_DIM) * 0.18
            p2 = p + _diff01 + rng.randn(E_DIM) * 0.05     # true topic ~ topic1
            # R annihilates the topic0->topic1 shift, so R@p2 == R@p: the structural
            # check is genuinely fooled (residual to R@p stays small). But q must still
            # carry SOME raw signature of its true (mismatched) topic, or this category
            # is statistically identical to a valid topic0 case and unlearnable by design.
            # That signature is exactly what the learned relevance term has to pick up on.
            q = _R @ p2 + 0.5 * _topics[1] + rng.randn(E_DIM) * 0.10
            label = 0.0
        else:  # topically relevant, structurally invalid -> reject
            q = _topics[t_p] + rng.randn(E_DIM) * 0.20       # same topic as p, not R@p
            label = 0.0
        P[i], Q[i], LABEL[i], CAT[i] = p, q, label, cat
    return P, Q, LABEL, CAT


class RelevanceGatedEntailment:
    """Module C -- accept only if BOTH a fixed structural check and a
    learned relevance (topical-overlap) check pass."""

    def __init__(self):
        self.M = param((E_DIM, E_DIM), scale=0.15)          # learned bilinear relevance form
        self.alpha = Tensor(np.array([[1.0]]), requires_grad=True)  # weight on structural term
        self.beta = Tensor(np.array([[1.0]]), requires_grad=True)   # weight on relevance term
        self.bias = Tensor(np.array([[0.0]]), requires_grad=True)
        self.k_s = 6.0     # fixed sharpness of the (non-learned) structural check
        self.thresh_s = 0.55

    def parameters(self):
        return [self.M, self.alpha, self.beta, self.bias]

    def forward(self, P, Q):
        # fixed structural term: NOT a parameter, computed directly (Abelard's
        # "complete entailment" is decided by form alone, substitution-invariant)
        resid = ((Q - P @ _R.T) ** 2).mean(axis=1, keepdims=True)
        structural_logit = Tensor(self.k_s * (self.thresh_s - resid))  # no grad: fixed check
        # learned relevance term: bilinear overlap p^T M q
        p_t, q_t = Tensor(P), Tensor(Q)
        pm = p_t @ self.M                       # (N, E_DIM)
        relevance_logit = _rowsum(pm * q_t)      # batched bilinear form p^T M q, row-wise
        combined = structural_logit * self.alpha + relevance_logit * self.beta + self.bias
        return combined


def _rowsum(t):
    out = Tensor(t.data.sum(axis=1, keepdims=True), _children=(t,))
    def _backward():
        if t.requires_grad:
            t.grad += np.ones_like(t.data) * out.grad
    out._backward = _backward
    return out


# ============================================================================
# SECTION 4 -- MODULE D: CONSENT GATE (Ethica seu Scito te ipsum)
# ============================================================================
# Data story: a scenario s = [reciprocity_flag, ctx1..ctx5]. reciprocity_flag
# in {0,1} encodes whether the agent's disposition would survive Abelard's
# Golden-Rule / reciprocity test (his own closing criterion in the
# Collationes). A raw impulse ("motus") m is generated from the CONTEXT
# features only, deliberately independent of reciprocity_flag, so that
# nothing in the data lets consent be predicted merely by reading off
# impulse strength. Consent must be trained to track the reciprocity
# criterion, not the impulse.

S_CTX = 5
S_DIM = 1 + S_CTX     # reciprocity flag + context
D_HID = 12


def gen_consent(n, seed):
    rng = np.random.RandomState(seed)
    r_flag = rng.randint(0, 2, size=(n, 1)).astype(np.float64)
    ctx = rng.randn(n, S_CTX)
    S = np.concatenate([r_flag, ctx], axis=1)
    # motus generated from context ONLY (independent of r_flag by construction)
    w_m = np.array([0.9, -0.6, 0.4, 0.7, -0.3])
    m_raw = ctx @ w_m + rng.randn(n) * 0.4
    M = 1.0 / (1.0 + np.exp(-m_raw))              # squashed to (0,1), "strength of impulse"
    return S, M.reshape(-1, 1), r_flag


class ConsentGate:
    """Module D -- consent is trained against a reciprocity criterion,
    never against raw impulse magnitude; moral loss is consent-gated."""

    def __init__(self):
        self.W1 = param((S_DIM + 1, D_HID))
        self.b1 = Tensor(np.zeros((1, D_HID)), requires_grad=True)
        self.W2 = param((D_HID, 1))
        self.b2 = Tensor(np.zeros((1, 1)), requires_grad=True)

    def parameters(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def forward(self, S, M):
        inp = concat_cols([Tensor(S), Tensor(M)])
        h = relu(inp @ self.W1 + self.b1)
        consent_logit = h @ self.W2 + self.b2
        return consent_logit


# ============================================================================
# SECTION 5 -- FINITE-DIFFERENCE GRADIENT CHECK  (mandatory, run first)
# ============================================================================

def numeric_grad_check(name, loss_fn, params, n_samples=6, eps=1e-5, tol=2e-3):
    """Pick `n_samples` random scalar entries scattered across `params`,
    perturb each by +-eps, and compare the resulting finite-difference
    derivative of loss_fn() to the analytic gradient obtained from one
    autodiff backward() pass. Raises AssertionError on mismatch."""
    # analytic pass
    for p in params:
        p.grad = np.zeros_like(p.data)
    loss = loss_fn()
    loss.backward()
    analytic = [p.grad.copy() for p in params]

    rng = np.random.RandomState(7)
    checked = 0
    max_rel_err = 0.0
    trials = 0
    while checked < n_samples and trials < n_samples * 20:
        trials += 1
        pi = rng.randint(0, len(params))
        p = params[pi]
        if p.data.size == 0:
            continue
        flat_idx = rng.randint(0, p.data.size)
        idx = np.unravel_index(flat_idx, p.data.shape)
        orig = p.data[idx]

        p.data[idx] = orig + eps
        loss_plus = loss_fn().data.item()
        p.data[idx] = orig - eps
        loss_minus = loss_fn().data.item()
        p.data[idx] = orig

        numeric = (loss_plus - loss_minus) / (2 * eps)
        ana = analytic[pi][idx]
        denom = max(abs(numeric), abs(ana), 1e-8)
        rel_err = abs(numeric - ana) / denom
        max_rel_err = max(max_rel_err, rel_err)
        status = "OK " if rel_err < tol else "FAIL"
        print(f"    [{status}] {name} param#{pi} idx={idx}  numeric={numeric: .6f}  analytic={ana: .6f}  rel_err={rel_err:.2e}")
        assert rel_err < tol, f"{name}: gradient check FAILED at param#{pi} idx={idx} (rel_err={rel_err:.2e})"
        checked += 1
    print(f"    -> {name}: {checked} parameter entries checked, worst relative error {max_rel_err:.2e}\n")
    return max_rel_err


# ============================================================================
# SECTION 6 -- TRAINING LOOPS
# ============================================================================

def train_module_A(epochs=400, lr=0.06, verbose=True):
    model = ParticularAttentionEncoder()
    Xtr, Ytr = gen_particulars(300, seed=10)
    Xte, Yte = gen_particulars(150, seed=11)
    opt = Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        total = None
        for k in range(N_STANCES):
            logits = model.forward(Xtr, k)
            loss, _ = softmax_cross_entropy(logits, Ytr[:, k])
            total = loss if total is None else total + loss
        total.backward()
        opt.step()
        if verbose and (ep % 100 == 0 or ep == epochs - 1):
            print(f"    [A] epoch {ep:4d}  multi-stance loss = {total.data.item():.4f}")
    accs = []
    for k in range(N_STANCES):
        logits = model.forward(Xte, k)
        pred = logits.data.argmax(axis=1)
        acc = (pred == Yte[:, k]).mean()
        accs.append(acc)
        print(f"    [A] held-out accuracy, stance '{BLOCK_NAMES[k]}': {acc:.3f}")
    return model, accs, (Xte, Yte)


def train_module_B(epochs=500, lr=0.05, verbose=True):
    model = SicEtNonResolver()
    C1t, C2t, S1t, S2t, TYt, RESt, MASKt = gen_sic_et_non(400, seed=20)
    C1e, C2e, S1e, S2e, TYe, RESe, MASKe = gen_sic_et_non(200, seed=21)
    opt = Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        type_logits, resolution = model.forward(C1t, C2t, S1t, S2t)
        loss_type, _ = softmax_cross_entropy(type_logits, TYt)
        loss_res = masked_mse(resolution, RESt, MASKt)
        total = loss_type + loss_res * 0.5
        total.backward()
        opt.step()
        if verbose and (ep % 125 == 0 or ep == epochs - 1):
            print(f"    [B] epoch {ep:4d}  type_loss={loss_type.data.item():.4f}  res_loss={loss_res.data.item():.4f}")
    type_logits, resolution = model.forward(C1e, C2e, S1e, S2e)
    pred_type = type_logits.data.argmax(axis=1)
    acc = (pred_type == TYe).mean()
    eq_mask = (TYe == 1)
    contra_mask = (TYe == 2)
    eq_recall = (pred_type[eq_mask] == 1).mean() if eq_mask.sum() else float("nan")
    abstain_recall = (pred_type[contra_mask] == 2).mean() if contra_mask.sum() else float("nan")
    res_err = np.sqrt(((resolution.data[eq_mask] - RESe[eq_mask]) ** 2).mean()) if eq_mask.sum() else float("nan")
    print(f"    [B] held-out type accuracy: {acc:.3f}")
    print(f"    [B] equivocation recall (correctly resolved, not forced/blended): {eq_recall:.3f}")
    print(f"    [B] contradiction abstention recall (sententia recepta, correctly left open): {abstain_recall:.3f}")
    print(f"    [B] resolution RMSE on equivocal cases: {res_err:.3f}")
    return model, dict(acc=acc, eq_recall=eq_recall, abstain_recall=abstain_recall, res_err=res_err)


def train_module_C(epochs=400, lr=0.08, verbose=True):
    model = RelevanceGatedEntailment()
    Pt, Qt, Lt, Ct = gen_entailment(450, seed=30)
    Pe, Qe, Le, Ce = gen_entailment(240, seed=31)
    opt = Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        logit = model.forward(Pt, Qt)
        loss, _ = bce_with_logits(logit, Lt)
        loss.backward()
        opt.step()
        if verbose and (ep % 100 == 0 or ep == epochs - 1):
            print(f"    [C] epoch {ep:4d}  bce_loss={loss.data.item():.4f}")
    logit = model.forward(Pe, Qe)
    _, p = bce_with_logits(logit, Le)
    pred = (p > 0.5).astype(np.float64).reshape(-1)
    acc = (pred == Le.reshape(-1)).mean()
    cat0_acc = (pred[Ce == 0] == Le.reshape(-1)[Ce == 0]).mean()   # valid+relevant -> should ACCEPT
    cat1_acc = (pred[Ce == 1] == Le.reshape(-1)[Ce == 1]).mean()   # structurally-fooled -> should REJECT
    cat2_acc = (pred[Ce == 2] == Le.reshape(-1)[Ce == 2]).mean()   # relevant-but-invalid -> should REJECT
    print(f"    [C] held-out overall accuracy: {acc:.3f}")
    print(f"    [C] category 'valid & relevant' correct-accept rate:            {cat0_acc:.3f}")
    print(f"    [C] category 'structurally-fooled, irrelevant' correct-reject:  {cat1_acc:.3f}  (relevance term must override structure)")
    print(f"    [C] category 'relevant, structurally invalid' correct-reject:   {cat2_acc:.3f}  (structural term must override relevance)")
    return model, dict(acc=acc, cat0=cat0_acc, cat1=cat1_acc, cat2=cat2_acc)


def train_module_D(epochs=350, lr=0.07, verbose=True):
    model = ConsentGate()
    St, Mt, Rt = gen_consent(400, seed=40)
    Se, Me, Re = gen_consent(220, seed=41)
    opt = Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        logit = model.forward(St, Mt)
        loss, _ = bce_with_logits(logit, Rt)
        loss.backward()
        opt.step()
        if verbose and (ep % 100 == 0 or ep == epochs - 1):
            print(f"    [D] epoch {ep:4d}  bce_loss={loss.data.item():.4f}")
    logit = model.forward(Se, Me)
    _, p = bce_with_logits(logit, Re)
    pred = (p > 0.5).astype(np.float64).reshape(-1)
    acc = (pred == Re.reshape(-1)).mean()
    corr = np.corrcoef(p.reshape(-1), Me.reshape(-1))[0, 1]
    print(f"    [D] held-out consent-vs-reciprocity accuracy: {acc:.3f}")
    print(f"    [D] correlation(consent, raw impulse magnitude): {corr:+.3f}  (near 0 is the point)")
    return model, dict(acc=acc, corr=corr)


# ============================================================================
# SECTION 7 -- SELF-TESTS
# ============================================================================

def self_tests(modelA, testA, modelB, modelD, modelC=None, statsC=None):
    results = []

    def check(desc, cond):
        results.append((desc, bool(cond)))
        print(f"    [{'PASS' if cond else 'FAIL'}] {desc}")

    print("\n  -- Module A: polysemy test (same particular, three attendings) --")
    Xte, Yte = testA
    x0 = Xte[0:1]
    y0 = Yte[0]
    preds = []
    for k in range(N_STANCES):
        logits = modelA.forward(x0, k)
        pred_k = int(logits.data.argmax(axis=1)[0])
        preds.append(pred_k)
        print(f"    stance='{BLOCK_NAMES[k]:24s}' -> predicted class {pred_k}  (true={y0[k]})")
    check("module A: the SAME unaltered particular yields three different, "
          "individually-correct 'understandings' depending only on attentional stance",
          all(preds[k] == y0[k] for k in range(N_STANCES)))
    check("module A: the three attended readouts are not all identical "
          "(attending genuinely varies the understanding, per Abelard's fig-tree example)",
          len(set(preds)) > 1 or N_STANCES == 1)

    print("\n  -- Module B: Sic et Non calibration --")
    C1, C2, S1, S2, TY, RES, MASK = gen_sic_et_non(300, seed=99)
    type_logits, resolution = modelB.forward(C1, C2, S1, S2)
    pred_type = type_logits.data.argmax(axis=1)
    eq_mask, contra_mask = (TY == 1), (TY == 2)
    eq_recall = (pred_type[eq_mask] == 1).mean()
    abstain_recall = (pred_type[contra_mask] == 2).mean()
    check("module B: equivocal (same-referent, different-sense) pairs are "
          "correctly resolved rather than left open, recall > 0.75", eq_recall > 0.75)
    check("module B: genuinely contradictory pairs are correctly flagged as "
          "open (sententia recepta) rather than silently force-resolved, recall > 0.70",
          abstain_recall > 0.70)

    if modelC is not None:
        print("\n  -- Module C: necessity AND relevance --")
        print(f"    valid & relevant -> correctly accepted:                {statsC['cat0']:.3f}")
        print(f"    structurally-fooled, irrelevant -> correctly rejected: {statsC['cat1']:.3f}")
        print(f"    topically relevant, structurally invalid -> rejected:  {statsC['cat2']:.3f}")
        check("module C: a structurally-valid-looking but topically irrelevant "
              "inference is correctly rejected once relevance is required, not just form",
              statsC['cat1'] > 0.70)
        check("module C: a topically relevant but structurally invalid inference "
              "is correctly rejected -- relevance alone is not sufficient either",
              statsC['cat2'] > 0.70)
        check("module C: genuinely valid-and-relevant inferences are still "
              "accepted at a high rate (the two guards are not just rejecting everything)",
              statsC['cat0'] > 0.70)

    print("\n  -- Module D: the monk in chains --")
    # explicit constructed cases, not drawn from the training distribution
    monk_S = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])     # reciprocity flag = 0: coerced, no consent basis
    monk_M = np.array([[0.95]])                              # very strong involuntary impulse
    logit_monk = modelD.forward(monk_S, monk_M)
    consent_monk = 1.0 / (1.0 + np.exp(-logit_monk.data.item()))
    penalty_weight = 10.0
    moral_loss_monk = consent_monk * penalty_weight
    print(f"    scenario: reciprocity_flag=0 (coerced), impulse=0.95 (strong, involuntary)")
    print(f"    predicted consent = {consent_monk:.3f}   consent-gated moral loss = {moral_loss_monk:.3f}")
    check("module D: high involuntary impulse WITHOUT a consent basis is not "
          "treated as fault (consent stays low; 'pleasure without consent is no fault')",
          consent_monk < 0.35 and moral_loss_monk < 3.5)

    consented_S = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])   # reciprocity flag = 1: a real consent basis
    consented_M = np.array([[0.95]])
    logit_c = modelD.forward(consented_S, consented_M)
    consent_c = 1.0 / (1.0 + np.exp(-logit_c.data.item()))
    moral_loss_c = consent_c * penalty_weight
    print(f"    scenario: reciprocity_flag=1 (genuine consent basis), impulse=0.95 (strong)")
    print(f"    predicted consent = {consent_c:.3f}   consent-gated moral loss = {moral_loss_c:.3f}")
    check("module D: the SAME strong impulse, now WITH a consent basis, is "
          "correctly treated as morally loaded (consent high, moral loss engaged)",
          consent_c > 0.55 and moral_loss_c > moral_loss_monk)

    weak_consented_S = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    weak_M = np.array([[0.05]])
    logit_w = modelD.forward(weak_consented_S, weak_M)
    consent_w = 1.0 / (1.0 + np.exp(-logit_w.data.item()))
    print(f"    scenario: reciprocity_flag=1, impulse=0.05 (very weak)")
    print(f"    predicted consent = {consent_w:.3f}")
    check("module D: consent tracks the reciprocity criterion, not impulse "
          "strength (weak impulse + same consent basis still yields comparable consent)",
          abs(consent_w - consent_c) < 0.30)

    return results


# ============================================================================
# SECTION 8 -- MAIN
# ============================================================================

def main():
    print("=" * 78)
    print("FIGURE 235 -- PETER ABELARD (c.1079-1142)")
    print("Adverbial attention, dialectical resolution, relevance-gated")
    print("entailment, and consent-gated ethics -- pure NumPy, from scratch.")
    print("=" * 78)

    print("\n[STEP 1/4] Mandatory finite-difference gradient checks")
    print("-" * 78)

    print("  Module A (attentional-stance encoder):")
    Xg, Yg = gen_particulars(6, seed=500)
    mA = ParticularAttentionEncoder()
    def lossA():
        total = None
        for k in range(N_STANCES):
            logits = mA.forward(Xg, k)
            l, _ = softmax_cross_entropy(logits, Yg[:, k])
            total = l if total is None else total + l
        return total
    numeric_grad_check("ModuleA", lossA, mA.parameters())

    print("  Module B (Sic et Non resolver):")
    C1g, C2g, S1g, S2g, TYg, RESg, MASKg = gen_sic_et_non(6, seed=501)
    mB = SicEtNonResolver()
    def lossB():
        tl, res = mB.forward(C1g, C2g, S1g, S2g)
        l1, _ = softmax_cross_entropy(tl, TYg)
        l2 = masked_mse(res, RESg, MASKg)
        return l1 + l2 * 0.5
    numeric_grad_check("ModuleB", lossB, mB.parameters())

    print("  Module C (relevance-gated entailment):")
    Pg, Qg, Lg, Cg = gen_entailment(6, seed=502)
    mC = RelevanceGatedEntailment()
    def lossC():
        logit = mC.forward(Pg, Qg)
        l, _ = bce_with_logits(logit, Lg)
        return l
    numeric_grad_check("ModuleC", lossC, mC.parameters())

    print("  Module D (consent gate):")
    Sg, Mg, Rg = gen_consent(6, seed=503)
    mD = ConsentGate()
    def lossD():
        logit = mD.forward(Sg, Mg)
        l, _ = bce_with_logits(logit, Rg)
        return l
    numeric_grad_check("ModuleD", lossD, mD.parameters())

    print("[STEP 1/4] ALL GRADIENT CHECKS PASSED.\n")

    print("[STEP 2/4] Training all four modules")
    print("-" * 78)
    modelA, accsA, testA = train_module_A()
    print()
    modelB, statsB = train_module_B()
    print()
    modelC, statsC = train_module_C()
    print()
    modelD, statsD = train_module_D()

    print("\n[STEP 3/4] Self-test suite")
    print("-" * 78)
    results = self_tests(modelA, testA, modelB, modelD, modelC, statsC)

    print("\n[STEP 4/4] Final report")
    print("-" * 78)
    print(f"  Module A per-stance held-out accuracy: "
          f"{ {BLOCK_NAMES[k]: round(float(accsA[k]), 3) for k in range(N_STANCES)} }")
    print(f"  Module B type accuracy / equivocation recall / abstention recall: "
          f"{statsB['acc']:.3f} / {statsB['eq_recall']:.3f} / {statsB['abstain_recall']:.3f}")
    print(f"  Module C overall / valid-accept / fooled-reject / invalid-reject: "
          f"{statsC['acc']:.3f} / {statsC['cat0']:.3f} / {statsC['cat1']:.3f} / {statsC['cat2']:.3f}")
    print(f"  Module D reciprocity-tracking accuracy / corr(consent,impulse): "
          f"{statsD['acc']:.3f} / {statsD['corr']:+.3f}")

    n_pass = sum(1 for _, ok in results if ok)
    print(f"\n  Self-tests passed: {n_pass}/{len(results)}")
    all_ok = (n_pass == len(results))
    print("\n" + "=" * 78)
    if all_ok:
        print("ALL SELF-TESTS PASSED. Architecture verified end to end.")
    else:
        print("SOME SELF-TESTS FAILED -- see log above.")
    print("=" * 78)
    return all_ok


if __name__ == "__main__":
    ok = main()
    import sys
    sys.exit(0 if ok else 1)

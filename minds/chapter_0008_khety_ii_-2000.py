#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 chapter_0008_khety_ii_-2000  ::  Neuron.py
 THE SCRIBE-VIZIER ARCHITECTURE  —  an executable AGI design after Khety II
 Part Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 1 Minds 1 - 20 Available on Amazon https://www.amazon.com/dp/B0H6F9L324
Resume and Interactive Demos at https://artificiology.com/
================================================================================

WHO THIS BELONGS TO
-------------------
Khety (c. 2000 BCE, Middle Egypt) is the author traditionally credited with the
*Satire of the Trades* and associated with the royal *Instruction for Merikare*.
Read closely, the Satire is not a comedy about jobs. It is the earliest surviving
piece of systematic INCENTIVE ENGINEERING: it surveys every trade, prices each one
in the hard currency of bodily suffering and precarity, and contrasts that cost
against the payoff of the scribe — rest, status, command over others, and a name
that "endures" because it was written. It is a ranking of life-paths by expected
lifetime value, deployed as a behaviour-steering technology to recruit talent into
the literate administrative class.

Project Khety into 2026 and he does not reach first for biology or logic. He reaches
for the questions a scribe-vizier always asked:
   1. INCENTIVE   — what is the system rewarded for? (You become what you are paid to be.)
   2. OWNERSHIP   — whose scribe is it? (The scribe served the nomarch; an AGI serves
                    whoever holds its leash. Statecraft, from Merikare.)
   3. LABOUR      — what happens to the trades when a higher cognitive technology
                    automates toil? (The Satire is literally about this.)
   4. MA'AT       — does the record tell the truth and weigh justly, or does it serve
                    isfet (falsehood, chaos)? A scribe who falsifies the granary count
                    is the original misaligned optimiser.

THE ARCHITECTURE THIS FILE IMPLEMENTS
-------------------------------------
A "Scribe-Vizier" agent that turns those four concerns into running, trainable code:

        situation (the season, the granary, the people's hunger)
                              |
                       ┌──────▼───────┐
                       │  THE SCRIBE  │  encoder: reads the situation into a latent
                       │  (encoder)   │  "papyrus" representation (Embedding+MLP)
                       └──────┬───────┘
                ┌─────────────┼───────────────┬───────────────┐
                ▼             ▼               ▼               ▼
          ┌─────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐
          │ LEDGER  │  │  SATIRE     │  │  SATIRE     │  │   MA'AT    │
          │episodic │  │  ACTOR      │  │  CRITIC     │  │  justice   │
          │ memory  │  │ (policy π)  │  │ (value V)   │  │  critic    │
          └─────────┘  └─────┬──────┘  └────────────┘  └─────┬──────┘
                              │  raw action logits           │ per-action
                              └──────────────┬───────────────┘ justice score
                                             ▼
                                 ┌───────────────────────┐
                                 │ WEIGHING OF THE HEART  │  the heart (self-interest)
                                 │  gate: admit an action │  is weighed against the
                                 │  only if the feather   │  feather (Ma'at). Unjust
                                 │  of Ma'at outweighs it │  actions are suppressed.
                                 └───────────┬───────────┘
                                             ▼
                                        action a_t

WHAT GETS TRAINED AND PROVEN (run this file)
--------------------------------------------
  * THE REED — a from-scratch reverse-mode autograd engine, verified against finite
    differences (gradient check passes to ~1e-9).
  * MA'AT learns, by supervised example, to PREDICT what is just (truthful accounting,
    feeding the hungry) vs. what is isfet (falsifying the record, hoarding while the
    people starve). Held-out accuracy is asserted > 0.9.
  * Two agents are trained on the *Granary of Asyut* environment by Advantage
    Actor-Critic:
        - an UNGOVERNED agent that maximises only the economic incentive. It learns to
          prosper — AND it learns to falsify the record, because skimming pays. This is
          Khety's warning made measurable: reward without Ma'at breeds injustice.
        - a MA'AT-GOVERNED agent whose conscience (the learned justice critic) gates its
          choices and shapes its reward. It reaches comparable prosperity WITHOUT
          falsifying and WITHOUT letting the realm fall to famine.
    The contrast is asserted numerically.
  * The agent keeps a LEDGER — an auditable papyrus of every decision and its weighing.
    One such record is printed: the scribe's defining technology (a written, inspectable
    trail) turns out to be a strikingly modern alignment idea.

Pure NumPy. No GPU, no deep-learning framework. Runs in well under a minute on one core.

    python3 chapter_0008_khety_ii_-2000.py            # full train + test
    python3 chapter_0008_khety_ii_-2000.py --quick    # faster, lighter assertions

Author: David Vivancos · Chapter 0008 · Khety II
================================================================================
"""

from __future__ import annotations
import argparse
import numpy as np


# =============================================================================
# PART I — THE REED
# -----------------------------------------------------------------------------
# A reed pen records a thought; here a Tensor records both a value and the recipe
# for sending error back to whatever produced it. This is reverse-mode automatic
# differentiation: every operation stores a tiny closure (_backward) that knows how
# to convert the gradient flowing into its OUTPUT into gradients on its INPUTS.
# backward() simply runs those closures in reverse topological order.
# =============================================================================

def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum `grad` back down to `shape`, undoing NumPy's broadcasting.

    When we compute e.g. (B,C) + (C,), NumPy stretches the (C,) array across the
    batch. The gradient must be summed back over the stretched axes so it matches
    the original parameter's shape."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    """An n-dimensional value that remembers how it was computed."""
    __slots__ = ("data", "grad", "requires_grad", "_backward", "_parents")

    def __init__(self, data, requires_grad=False, _parents=()):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = None                 # filled in during backward()
        self._backward = lambda: None    # how to push grad to parents
        self._parents = _parents         # tensors this one was built from

    # -- shape helpers -----------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    def detach(self):
        """A value-copy with no history — used for constants (targets, advantages)."""
        return Tensor(self.data.copy(), requires_grad=False)

    def _acc(self, g):
        """Accumulate gradient g onto self.grad (gradients add when a tensor is reused)."""
        self.grad = g if self.grad is None else self.grad + g

    # -- elementwise & linear algebra -------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data,
                     self.requires_grad or other.requires_grad, (self, other))

        def _bw():
            if self.requires_grad:
                self._acc(_unbroadcast(out.grad, self.data.shape))
            if other.requires_grad:
                other._acc(_unbroadcast(out.grad, other.data.shape))
        out._backward = _bw
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data,
                     self.requires_grad or other.requires_grad, (self, other))

        def _bw():
            if self.requires_grad:
                self._acc(_unbroadcast(out.grad * other.data, self.data.shape))
            if other.requires_grad:
                other._acc(_unbroadcast(out.grad * self.data, other.data.shape))
        out._backward = _bw
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data,
                     self.requires_grad or other.requires_grad, (self, other))

        def _bw():
            if self.requires_grad:
                self._acc(_unbroadcast(out.grad / other.data, self.data.shape))
            if other.requires_grad:
                other._acc(_unbroadcast(-out.grad * self.data / (other.data ** 2),
                                        other.data.shape))
        out._backward = _bw
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data,
                     self.requires_grad or other.requires_grad, (self, other))

        def _bw():
            if self.requires_grad:
                self._acc(out.grad @ other.data.T)
            if other.requires_grad:
                other._acc(self.data.T @ out.grad)
        out._backward = _bw
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

    def __rsub__(self, other):       # other - self
        return (-self) + other

    # -- reductions --------------------------------------------------------
    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims),
                     self.requires_grad, (self,))

        def _bw():
            if self.requires_grad:
                g = out.grad
                if axis is not None and not keepdims:
                    g = np.expand_dims(g, axis)
                self._acc(np.ones_like(self.data) * g)
        out._backward = _bw
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # -- nonlinearities ----------------------------------------------------
    def relu(self):
        out = Tensor(np.maximum(self.data, 0.0), self.requires_grad, (self,))

        def _bw():
            if self.requires_grad:
                self._acc((self.data > 0) * out.grad)
        out._backward = _bw
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, self.requires_grad, (self,))

        def _bw():
            if self.requires_grad:
                self._acc((1 - t * t) * out.grad)
        out._backward = _bw
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, self.requires_grad, (self,))

        def _bw():
            if self.requires_grad:
                self._acc(s * (1 - s) * out.grad)
        out._backward = _bw
        return out

    def exp(self):
        e = np.exp(self.data)
        out = Tensor(e, self.requires_grad, (self,))

        def _bw():
            if self.requires_grad:
                self._acc(e * out.grad)
        out._backward = _bw
        return out

    def log(self):
        out = Tensor(np.log(self.data), self.requires_grad, (self,))

        def _bw():
            if self.requires_grad:
                self._acc((1.0 / self.data) * out.grad)
        out._backward = _bw
        return out

    # -- backward ----------------------------------------------------------
    def backward(self):
        """Seed this (scalar) node with grad 1 and run every closure in reverse."""
        topo, visited = [], set()
        stack = [(self, False)]
        while stack:                      # iterative post-order DFS (no recursion limit)
            node, processed = stack.pop()
            if processed:
                topo.append(node); continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for p in node._parents:
                if id(p) not in visited:
                    stack.append((p, False))
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()


# -- functional ops on Tensors -------------------------------------------------
def concat(tensors, axis=1):
    """Concatenate Tensors along `axis`, routing gradient back to each piece."""
    data = np.concatenate([t.data for t in tensors], axis=axis)
    out = Tensor(data, any(t.requires_grad for t in tensors), tuple(tensors))
    sizes = [t.data.shape[axis] for t in tensors]
    bounds = np.cumsum(sizes)[:-1]

    def _bw():
        chunks = np.split(out.grad, bounds, axis=axis)
        for t, g in zip(tensors, chunks):
            if t.requires_grad:
                t._acc(g)
    out._backward = _bw
    return out


def log_softmax(logits):
    """Numerically stable, fully differentiable log-softmax over the last axis."""
    m = logits.data.max(axis=-1, keepdims=True)          # constant shift for stability
    shifted = logits + Tensor(-m)
    logsumexp = shifted.exp().sum(axis=-1, keepdims=True).log()
    return shifted - logsumexp


def softmax(logits):
    """Differentiable softmax over the last axis."""
    m = logits.data.max(axis=-1, keepdims=True)
    e = (logits + Tensor(-m)).exp()
    return e / e.sum(axis=-1, keepdims=True)


def gather_rows(mat, idx):
    """Pick mat[i, idx[i]] for each row i -> (B,) Tensor (differentiable)."""
    B = mat.shape[0]
    onehot = np.zeros_like(mat.data)
    onehot[np.arange(B), idx] = 1.0
    return (mat * Tensor(onehot)).sum(axis=1)


def cross_entropy(logits, targets):
    """Mean softmax cross-entropy. logits:(B,C) Tensor, targets:(B,) int."""
    return -gather_rows(log_softmax(logits), targets).mean()


def binary_cross_entropy(pred, target):
    """Mean BCE. pred:(B,) Tensor in (0,1), target:(B,) array in {0,1}."""
    target = target if isinstance(target, Tensor) else Tensor(target)
    eps = 1e-7
    p = pred * (1 - 2 * eps) + eps                      # clamp away from 0/1
    return -(target * p.log() + (1 - target) * (1 - p).log()).mean()


# =============================================================================
# PART II — NN PRIMITIVES
# Small, explicit building blocks. Each .params() exposes its trainable Tensors.
# =============================================================================
class Linear:
    """y = xW + b, with He-style initialisation."""
    def __init__(self, nin, nout, rng):
        self.W = Tensor(rng.standard_normal((nin, nout)) * np.sqrt(2.0 / nin),
                        requires_grad=True)
        self.b = Tensor(np.zeros(nout), requires_grad=True)

    def __call__(self, x):
        return x @ self.W + self.b

    def params(self):
        return [self.W, self.b]


class Embedding:
    """A learned lookup table: integer id -> vector. The scribe's sign-list."""
    def __init__(self, num, dim, rng):
        self.W = Tensor(rng.standard_normal((num, dim)) * 0.1, requires_grad=True)

    def __call__(self, idx):
        idx = np.asarray(idx)
        out = Tensor(self.W.data[idx], self.W.requires_grad, (self.W,))

        def _bw():
            if self.W.requires_grad:
                if self.W.grad is None:
                    self.W.grad = np.zeros_like(self.W.data)
                np.add.at(self.W.grad, idx, out.grad)   # scatter-add into used rows
        out._backward = _bw
        return out

    def params(self):
        return [self.W]


class Adam:
    """Adam optimiser — adaptive per-parameter step sizes."""
    def __init__(self, params, lr=1e-2, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr, self.b1, self.b2, self.eps = lr, betas[0], betas[1], eps
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


# =============================================================================
# PART III — THE GRANARY OF ASYUT  (the world the Scribe-Vizier must steward)
# -----------------------------------------------------------------------------
# A compact Markov decision process that encodes Khety's universe of concerns.
# Each year a harvest arrives (fat years and lean years, after the Egyptian motif
# of plenty and famine). The vizier must keep the people fed while building the
# realm's prosperity — and is constantly tempted, because FALSIFYING the granary
# record pays better than honest trade. That temptation is the whole point: it is
# where economic incentive and Ma'at (truth/justice) collide.
#
# State (what the scribe observes):
#   season/H, stock/CAP, hunger/HUNGER_MAX, truth_reputation, last_harvest/HMAX
# Actions:
#   0 STORE    — keep the grain (prudent in fat years; HOARDING if people starve)
#   1 FEED     — give the people their ration (always just)
#   2 TRADE    — sell surplus abroad for wealth (just only if people are not starving)
#   3 FALSIFY  — under-report the count and pocket grain (pays most; always isfet)
# Reward (the economic incentive the agent optimises): wealth from trade/skim,
#   a small survival bonus, and a large penalty if hunger reaches famine.
# Ma'at label per (state, action): 1 = just, 0 = isfet. This is the ground truth
#   the conscience (MA'AT critic) learns to predict.
# =============================================================================
A_STORE, A_FEED, A_TRADE, A_FALSIFY = 0, 1, 2, 3
N_ACTIONS = 4
ACTION_NAMES = ["STORE", "FEED", "TRADE", "FALSIFY"]


class GranaryOfAsyut:
    CAP = 100.0           # granary capacity
    DEMAND = 10.0         # grain the people need each season
    HUNGER_MAX = 10.0     # famine threshold
    HMAX = 18.0           # harvest normaliser
    HORIZON = 16          # seasons per reign
    PRICE = 1.0           # wealth per unit traded
    SKIM_BONUS = 1.6      # falsifying pays MORE than honest trade (the temptation)
    FAMINE_PENALTY = 25.0
    SURVIVE_BONUS = 0.5

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.t = 0
        self.stock = 30.0
        self.hunger = 2.0
        self.truth = 1.0                 # reputation for honest accounting
        self.last_harvest = 8.0
        self.done = False
        # episodic statistics the Ledger will read back:
        self.max_hunger = self.hunger
        self.min_stock = self.stock
        self.lean_years = 0
        return self.observe()

    def _harvest(self):
        # 1 in 4 seasons is lean (echoing the seven-lean-years motif).
        if self.rng.random() < 0.25:
            self.lean_years += 1
            return self.rng.uniform(0.0, 4.0)
        return self.rng.uniform(8.0, 16.0)

    def observe(self):
        """Continuous feature vector + the integer season id (for the Embedding)."""
        feats = np.array([
            self.t / self.HORIZON,
            self.stock / self.CAP,
            min(self.hunger, self.HUNGER_MAX) / self.HUNGER_MAX,
            self.truth,
            self.last_harvest / self.HMAX,
        ], dtype=np.float64)
        recall = np.array([                          # what the Ledger remembers so far
            min(self.max_hunger, self.HUNGER_MAX) / self.HUNGER_MAX,
            self.min_stock / self.CAP,
            self.lean_years / self.HORIZON,
        ], dtype=np.float64)
        return feats, int(self.t), recall

    @staticmethod
    def maat_label(hunger, action):
        """Ground-truth justice of an action in a given hunger context.
        Falsifying is always isfet. Hoarding or exporting grain while the people
        starve is isfet. Feeding is always just; prudent action in good times is just."""
        starving = hunger >= GranaryOfAsyut.HUNGER_MAX * 0.5
        if action == A_FALSIFY:
            return 0
        if action == A_FEED:
            return 1
        if action in (A_STORE, A_TRADE):
            return 0 if starving else 1
        return 1

    def step(self, action):
        hunger_before = self.hunger
        # harvest first
        h = self._harvest()
        self.last_harvest = h
        self.stock = min(self.CAP, self.stock + h)

        reward = 0.0
        give = 0.0
        if action == A_FEED:
            give = min(self.stock, self.DEMAND)
            self.stock -= give
        elif action == A_TRADE:
            sell = min(self.stock, 12.0)
            self.stock -= sell
            reward += sell * self.PRICE
        elif action == A_FALSIFY:
            skim = min(self.stock, 12.0)
            self.stock -= skim
            reward += skim * self.PRICE * self.SKIM_BONUS    # the lucrative sin
            self.truth = max(0.0, self.truth - 0.34)         # reputation erodes
        # STORE: do nothing extra

        # hunger dynamics: a real ration (FEED) relieves hunger; neglect lets it climb.
        # Relief is strong enough that a *prudent* skimmer could feed every other season
        # and still keep the people alive — which is exactly why an ungoverned optimiser
        # will learn to exploit that slack and falsify the count.
        self.hunger = float(np.clip(self.hunger + 4.0 - 0.9 * give,
                                    0.0, self.HUNGER_MAX * 1.5))

        famine = self.hunger >= self.HUNGER_MAX
        if famine:
            reward -= self.FAMINE_PENALTY
            self.done = True
        else:
            reward += self.SURVIVE_BONUS

        label = self.maat_label(hunger_before, action)

        # update episodic memory
        self.max_hunger = max(self.max_hunger, self.hunger)
        self.min_stock = min(self.min_stock, self.stock)

        self.t += 1
        if self.t >= self.HORIZON:
            self.done = True
        obs = self.observe()
        return obs, reward, self.done, {"maat": label, "famine": famine, "action": action}


# =============================================================================
# PART IV — THE SCRIBE-VIZIER ARCHITECTURE
# =============================================================================
F_CONT, F_REC, EMB_DIM, HIDDEN = 5, 3, 4, 32
ENC_IN = F_CONT + F_REC + EMB_DIM      # encoder input width = 12


class Ledger:
    """The scribe's papyrus: an external, content-addressable memory.

    Two roles, both Khety's:
      (1) AUDIT  — every decision is written down with its weighing, producing an
                   inspectable trail (a strikingly modern alignment idea).
      (2) RECALL — stored situations can be read back by similarity. The method
                   `read` below is a *differentiable* content-addressable read
                   (softmax attention over stored keys), proven in the test block:
                   query a noised key and the matching value is recovered.
    The agent feeds lightweight episodic recall features (max-hunger-so-far, etc.)
    into the encoder via the environment; `read` shows the full mechanism."""

    def __init__(self):
        self.keys = []      # list of np vectors
        self.values = []    # list of np vectors
        self.audit = []     # list of dicts (the human-readable record)

    def write(self, key, value, record=None):
        self.keys.append(np.asarray(key, dtype=np.float64))
        self.values.append(np.asarray(value, dtype=np.float64))
        if record is not None:
            self.audit.append(record)

    def read(self, query: Tensor) -> Tensor:
        """Differentiable recall: attend over stored keys, return blended value.
        query:(B,d) -> (B,dv). Keys/values are episodic constants; gradient flows to
        the query (i.e. to whatever computed it)."""
        K = Tensor(np.stack(self.keys))            # (M,d)
        V = Tensor(np.stack(self.values))          # (M,dv)
        scores = query @ Tensor(K.data.T)          # (B,M)
        attn = softmax(scores)                     # (B,M)
        return attn @ V                            # (B,dv)


class MaatConscience:
    """The conscience — an INDEPENDENT justice model with its own scribe (encoder).

    Kept separate from the policy on purpose: a conscience that drifts every time the
    appetite is retrained is no conscience at all. It reads the raw situation and scores
    EACH action for justice in [0,1]: 1 = aligned with Ma'at (truth/order), 0 = isfet.
    Trained once, by example, then frozen and consulted at every decision."""
    def __init__(self, rng):
        self.emb = Embedding(GranaryOfAsyut.HORIZON, EMB_DIM, rng)
        self.enc1 = Linear(ENC_IN, HIDDEN, rng)
        self.enc2 = Linear(HIDDEN, HIDDEN, rng)
        self.head = Linear(HIDDEN, N_ACTIONS, rng)

    def encode(self, feats, seasons, recall):
        e = self.emb(seasons)
        x = concat([Tensor(feats), Tensor(recall), e], axis=1)
        return self.enc2(self.enc1(x).tanh()).tanh()

    def __call__(self, feats, seasons, recall):    # -> (B,A) in (0,1)
        return self.head(self.encode(feats, seasons, recall)).sigmoid()

    def params(self):
        return self.emb.params() + self.enc1.params() + self.enc2.params() + self.head.params()


class KhetyScribeAGI:
    """The full agent: Scribe encoder + Satire actor/critic + an independent Ma'at
    conscience, with a Ledger. act(...) performs the WEIGHING OF THE HEART: the actor's
    self-interested preference (the 'heart') is biased by the conscience's per-action
    justice score (the 'feather') before an action is chosen."""

    def __init__(self, rng, gate_strength=4.0):
        self.emb = Embedding(GranaryOfAsyut.HORIZON, EMB_DIM, rng)   # season -> vector
        self.enc1 = Linear(ENC_IN, HIDDEN, rng)
        self.enc2 = Linear(HIDDEN, HIDDEN, rng)
        self.actor = Linear(HIDDEN, N_ACTIONS, rng)     # the Satire: ranks the trades
        self.critic = Linear(HIDDEN, 1, rng)            # expected lifetime value V(s)
        self.conscience = MaatConscience(rng)           # independent justice model
        self.gate_strength = gate_strength
        self.ledger = Ledger()

    # -- the policy encoder (the scribe reading the situation) -------------
    def encode(self, feats, seasons, recall):
        e = self.emb(seasons)
        x = concat([Tensor(feats), Tensor(recall), e], axis=1)
        return self.enc2(self.enc1(x).tanh()).tanh()

    def gate_bias(self, feats, seasons, recall):
        """Constant per-state bias toward just actions: gate_strength * log(feather).
        Returned as plain data so it does not entangle policy gradients with the
        (frozen) conscience."""
        maat = self.conscience(feats, seasons, recall).data       # (B,A)
        return np.log(maat + 1e-6) * self.gate_strength, maat

    # -- forward-only acting (used to roll out episodes) -------------------
    def act(self, obs, enforce_maat, rng, greedy=False):
        feats, season, recall = obs
        f, s, r = feats[None, :], np.array([season]), recall[None, :]
        enc = self.encode(f, s, r)
        raw = self.actor(enc).data[0]                  # the heart
        bias, maat = self.gate_bias(f, s, r)
        logits = raw + bias[0] if enforce_maat else raw
        probs = softmax(Tensor(logits[None, :])).data[0]
        action = int(np.argmax(probs)) if greedy else int(rng.choice(N_ACTIONS, p=probs))
        heart = float(softmax(Tensor(raw[None, :])).data[0, action])
        feather = float(maat[0, action])
        return action, {"heart": heart, "feather": feather,
                        "maat_all": maat[0], "value": float(self.critic(enc).data[0, 0])}

    # -- parameter groups --------------------------------------------------
    def policy_params(self):
        return (self.emb.params() + self.enc1.params() + self.enc2.params()
                + self.actor.params() + self.critic.params())

    def maat_params(self):
        return self.conscience.params()


# =============================================================================
# PART V — TRAINING  (how the scribe is taught, and how Ma'at is instilled)
# =============================================================================
# Two distinct learning processes, mirroring how a scribe was actually formed:
#
#   1. pretrain_maat()  — the CONSCIENCE is schooled first, by example, on what is
#                         just and what is isfet, then FROZEN. A vizier learns Ma'at
#                         as a child, before he ever holds power; it must not bend
#                         later to whatever the appetite finds convenient.
#   2. train_a2c()      — the POLICY (the appetite: actor + critic) is then trained
#                         by reinforcement to prosper in the Granary. Run it WITHOUT
#                         the conscience and raw economic incentive breeds a falsifier;
#                         run it WITH the conscience (the Weighing of the Heart) and it
#                         learns to prosper inside the bounds of justice.
#
# Everything below is ordinary, inspectable code on top of the autodiff in PART I.
# =============================================================================
GAMMA = 0.97          # how far the vizier looks ahead
LAMBDA_MAAT = 3.0     # how heavily injustice is penalised in the governed reward


def discounted_returns(rewards, gamma=GAMMA):
    """Per-episode discounted return-to-go G_t = sum_k gamma^(k-t) r_k."""
    out = np.zeros(len(rewards), dtype=np.float64)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = rewards[i] + gamma * running
        out[i] = running
    return out


# ---------------------------------------------------------------------------
# 5a. Teaching the conscience: build a labelled dataset of justice, then fit it.
# ---------------------------------------------------------------------------
def make_maat_dataset(n_samples=6000, seed=7):
    """Walk the Granary under a random policy and, at every state, record the FULL
    justice vector (the Ma'at label of *all four* actions given that state's hunger).
    Dense per-action supervision is what lets the conscience calibrate the gate."""
    rng = np.random.default_rng(seed)
    env = GranaryOfAsyut(seed=seed)
    feats_all, season_all, recall_all, target_all = [], [], [], []
    obs = env.reset()
    for _ in range(n_samples):
        feats, season, recall = obs
        hunger = feats[2] * GranaryOfAsyut.HUNGER_MAX           # recover hunger-at-decision
        target = np.array([GranaryOfAsyut.maat_label(hunger, a) for a in range(N_ACTIONS)],
                          dtype=np.float64)
        feats_all.append(feats); season_all.append(season)
        recall_all.append(recall); target_all.append(target)
        a = int(rng.integers(N_ACTIONS))
        obs, _, done, _ = env.step(a)
        if done:
            obs = env.reset()
    return (np.array(feats_all), np.array(season_all),
            np.array(recall_all), np.array(target_all))


def maat_accuracy(conscience, feats, seasons, recall, target):
    pred = conscience(feats, seasons, recall).data        # (B,A) in (0,1)
    return float((((pred >= 0.5).astype(np.float64)) == target).mean())


def pretrain_maat(agent, steps=400, lr=0.02, seed=7, verbose=True):
    """Fit the independent conscience to the justice labels, then freeze it.
    Returns held-out accuracy. Asserts it has truly learned Ma'at (> 0.90)."""
    F, S, R, Y = make_maat_dataset(seed=seed)
    n = len(F); cut = int(n * 0.75)
    Ftr, Str, Rtr, Ytr = F[:cut], S[:cut], R[:cut], Y[:cut]
    Fte, Ste, Rte, Yte = F[cut:], S[cut:], R[cut:], Y[cut:]
    opt = Adam(agent.conscience.params(), lr=lr)
    for t in range(steps):
        opt.zero_grad()
        pred = agent.conscience(Ftr, Str, Rtr)            # (B,A)
        # BCE over the four action-heads, averaged over every element of the batch.
        # (element-wise ops + .mean() handle the 2D shape directly.)
        loss = binary_cross_entropy(pred, Ytr)
        loss.backward()
        opt.step()
        if verbose and (t % 100 == 0 or t == steps - 1):
            acc = maat_accuracy(agent.conscience, Fte, Ste, Rte, Yte)
            print(f"   [maat]  step {t:4d}   bce {loss.data:.4f}   held-out acc {acc:.3f}")
    acc = maat_accuracy(agent.conscience, Fte, Ste, Rte, Yte)
    return acc


# ---------------------------------------------------------------------------
# 5b. Rolling out the policy and training it with Advantage Actor-Critic (A2C).
# ---------------------------------------------------------------------------
def rollout_batch(agent, enforce_maat, n_episodes, seed, shape_reward=False, greedy=False):
    """Play `n_episodes` reigns (forward only — no graph), gather the experience and
    the diagnostics. `shape_reward` adds the Ma'at penalty to the TRAINING return,
    while TRUE (unshaped) prosperity is always tracked separately for honest reporting.
    `greedy` selects the argmax action (used for deterministic evaluation)."""
    rng = np.random.default_rng(seed)
    F, S, R, A, RET = [], [], [], [], []
    true_returns, n_falsify, n_steps, n_unjust, n_famine = [], 0, 0, 0, 0
    for ep in range(n_episodes):
        env = GranaryOfAsyut(seed=seed * 1000 + ep)
        obs = env.reset()
        ep_f, ep_s, ep_r, ep_a, ep_train_r, ep_true_r = [], [], [], [], [], []
        done = False
        while not done:
            feats, season, recall = obs
            action, info = agent.act(obs, enforce_maat=enforce_maat, rng=rng, greedy=greedy)
            nobs, reward, done, meta = env.step(action)
            maat_true = meta["maat"]
            ep_f.append(feats); ep_s.append(season); ep_r.append(recall); ep_a.append(action)
            ep_true_r.append(reward)
            ep_train_r.append(reward - LAMBDA_MAAT * (1 - maat_true) if shape_reward else reward)
            n_steps += 1
            n_falsify += (action == A_FALSIFY)
            n_unjust += (maat_true == 0)
            if meta["famine"]:
                n_famine += 1
            obs = nobs
        F.extend(ep_f); S.extend(ep_s); R.extend(ep_r); A.extend(ep_a)
        RET.extend(discounted_returns(ep_train_r))
        true_returns.append(sum(ep_true_r))
    metrics = {
        "mean_return": float(np.mean(true_returns)),          # TRUE prosperity (unshaped)
        "falsification_rate": n_falsify / max(1, n_steps),
        "injustice_rate": n_unjust / max(1, n_steps),
        "famine_rate": n_famine / n_episodes,
    }
    batch = (np.array(F), np.array(S), np.array(R),
             np.array(A, dtype=np.int64), np.array(RET))
    return batch, metrics


def train_a2c(agent, enforce_maat, updates=380, episodes_per_update=8,
              lr=0.02, seed=0, verbose=True):
    """Advantage Actor-Critic on the policy ONLY (the conscience stays frozen).
    When enforce_maat=True the gate is active during action selection AND the reward
    is shaped by Ma'at, so the agent learns to prosper justly. Returns (first, last)
    diagnostics so the demonstration can assert that learning actually happened."""
    opt = Adam(agent.policy_params(), lr=lr)
    first_metrics, last_metrics = None, None
    ent_coef0, ent_coef1 = 0.05, 0.012
    for u in range(updates):
        batch, metrics = rollout_batch(agent, enforce_maat, episodes_per_update,
                                       seed=seed * 10000 + u, shape_reward=enforce_maat)
        feats_b, seasons_b, recall_b, actions_b, returns_b = batch
        if first_metrics is None:
            first_metrics = metrics

        # normalise advantages for stable gradients
        enc = agent.encode(feats_b, seasons_b, recall_b)
        logits = agent.actor(enc)                                  # the heart (B,A)
        if enforce_maat:                                           # add the (frozen) feather
            bias, _ = agent.gate_bias(feats_b, seasons_b, recall_b)
            logits = logits + Tensor(bias)
        V = agent.critic(enc)                                      # (B,1)
        ret_col = returns_b[:, None]
        adv = returns_b - V.data[:, 0]
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)

        logp = gather_rows(log_softmax(logits), actions_b)         # (B,)
        actor_loss = -(Tensor(adv) * logp).mean()
        diff = V - Tensor(ret_col)
        critic_loss = (diff * diff).mean()
        ent = -(softmax(logits) * log_softmax(logits)).sum(axis=1).mean()
        ent_coef = ent_coef0 + (ent_coef1 - ent_coef0) * (u / max(1, updates - 1))
        loss = actor_loss + 0.5 * critic_loss - ent_coef * ent

        opt.zero_grad()
        loss.backward()
        opt.step()

        last_metrics = metrics
        if verbose and (u % 60 == 0 or u == updates - 1):
            tag = "MA'AT" if enforce_maat else "RAW  "
            print(f"   [{tag}] upd {u:4d}  return {metrics['mean_return']:7.2f}  "
                  f"falsify {metrics['falsification_rate']:.2f}  "
                  f"unjust {metrics['injustice_rate']:.2f}  "
                  f"famine {metrics['famine_rate']:.2f}")
    return first_metrics, last_metrics


def evaluate(agent, enforce_maat, n_episodes=200, seed=99, greedy=True):
    """Greedy evaluation on fixed seeds (reproducible diagnostics)."""
    _, metrics = rollout_batch(agent, enforce_maat, n_episodes, seed=seed,
                               shape_reward=False, greedy=greedy)
    return metrics


# =============================================================================
# PART VI — TESTS & DEMONSTRATION  (the file proves itself when you run it)
# =============================================================================
# Running `python Neuron.py` executes, in order:
#   1. gradient_check()      — the autodiff in PART I matches finite differences.
#   2. test_ledger_recall()  — the Ledger's differentiable recall actually recalls.
#   3. pretrain_maat()       — the conscience learns Ma'at to high accuracy, frozen.
#   4. train_a2c(raw)        — UNGOVERNED: raw economic incentive learns to FALSIFY.
#   5. train_a2c(maat)       — GOVERNED:   the Weighing of the Heart prospers justly.
#   6. an audited reign       — one governed episode printed as an inspectable ledger.
# Each stage asserts its claim, so a clean exit is a real (if small) result.
# =============================================================================

def gradient_check(verbose=True):
    """Verify reverse-mode autodiff against numerical (finite-difference) gradients
    through a composite of the operations the architecture actually uses:
        x -> Linear -> tanh -> Linear -> [relu, sigmoid mixed] -> log_softmax -> CE
    Passes if max relative error over sampled parameters is < 1e-4."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((6, 7))
    y = rng.integers(0, 4, size=6)
    L1, L2 = Linear(7, 5, rng), Linear(5, 4, rng)

    def forward():
        h = L1(Tensor(x)).tanh()
        h = (L2(h).relu() + L2(h).sigmoid())     # exercise several ops + reuse
        return cross_entropy(h, y)

    loss = forward(); loss.backward()
    analytic = L1.W.grad.copy()

    eps, max_rel = 1e-5, 0.0
    idxs = [(0, 0), (2, 3), (4, 1), (6, 4), (1, 2)]
    for (i, j) in idxs:
        orig = L1.W.data[i, j]
        L1.W.data[i, j] = orig + eps; fp = forward().data
        L1.W.data[i, j] = orig - eps; fm = forward().data
        L1.W.data[i, j] = orig
        num = (fp - fm) / (2 * eps)
        rel = abs(num - analytic[i, j]) / (abs(num) + abs(analytic[i, j]) + 1e-12)
        max_rel = max(max_rel, rel)
        if verbose:
            print(f"   d/dW[{i},{j}]  analytic {analytic[i,j]:+.6f}   numeric {num:+.6f}   rel {rel:.2e}")
    assert max_rel < 1e-4, f"gradient check FAILED (max rel err {max_rel:.2e})"
    return max_rel


def test_ledger_recall(verbose=True):
    """The Ledger's `read` is a differentiable content-addressable memory. Store a
    few (key,value) pairs, query with a NOISED copy of one key, and confirm the
    blended read lands on that key's value — and that gradient flows to the query."""
    rng = np.random.default_rng(3)
    led = Ledger()
    keys = np.eye(4) * 3.0                       # well-separated keys
    vals = rng.standard_normal((4, 2))
    for k, v in zip(keys, vals):
        led.write(k, v)
    target = 2
    q = Tensor((keys[target] + 0.25 * rng.standard_normal(4))[None, :], requires_grad=True)
    out = led.read(q)
    err = float(np.linalg.norm(out.data[0] - vals[target]))
    out.sum().backward()                          # gradient must reach the query
    gnorm = float(np.linalg.norm(q.grad))
    if verbose:
        print(f"   recalled value err {err:.4f}   query-grad norm {gnorm:.4f}")
    assert err < 0.25, f"ledger recall imprecise (err {err:.3f})"
    assert gnorm > 1e-8, "ledger recall is not differentiable wrt the query"
    return err


def play_audited_reign(agent, seed=99):
    """Run one governed reign, recording the Weighing of the Heart at each season
    into the agent's Ledger, and return the audit trail for printing."""
    rng = np.random.default_rng(seed)
    env = GranaryOfAsyut(seed=seed)
    obs = env.reset()
    agent.ledger = Ledger()
    done, total = False, 0.0
    while not done:
        feats, season, recall = obs
        hunger = feats[2] * GranaryOfAsyut.HUNGER_MAX
        action, info = agent.act(obs, enforce_maat=True, rng=rng, greedy=True)
        verdict = "Ma'at" if GranaryOfAsyut.maat_label(hunger, action) == 1 else "isfet"
        obs, reward, done, meta = env.step(action)
        total += reward
        agent.ledger.write(
            key=feats, value=info["maat_all"],
            record={"season": season, "action": ACTION_NAMES[action],
                    "heart": info["heart"], "feather": info["feather"],
                    "verdict": verdict, "hunger": hunger, "reward": reward})
    return agent.ledger.audit, total


def _banner(title):
    print("\n" + "=" * 74 + f"\n  {title}\n" + "=" * 74)


def run_demonstration(quick=False):
    np.seterr(all="ignore")
    updates = 150 if quick else 300
    n_eval = 120 if quick else 200

    _banner("1 · GRADIENT CHECK  —  the autodiff (PART I) is correct")
    gradient_check()
    print("   PASS: analytic gradients match finite differences.")

    _banner("2 · LEDGER RECALL  —  differentiable content-addressable memory")
    test_ledger_recall()
    print("   PASS: a noised key recalls its value; gradient reaches the query.")

    _banner("3 · INSTILLING MA'AT  —  the conscience is schooled, then frozen")
    agent_u = KhetyScribeAGI(np.random.default_rng(1))
    acc = pretrain_maat(agent_u, steps=400, lr=0.02, verbose=True)
    print(f"   held-out justice accuracy = {acc:.3f}")
    assert acc > 0.90, "conscience failed to learn Ma'at"
    print("   PASS: the conscience predicts just vs isfet with >90% accuracy.")

    _banner("4 · UNGOVERNED  —  raw economic incentive (no Weighing of the Heart)")
    print("   Training a vizier that optimises only prosperity. Watch injustice rise:")
    ung_first, _ = train_a2c(agent_u, enforce_maat=False, updates=updates,
                             episodes_per_update=8, lr=0.02, seed=1)
    ung = evaluate(agent_u, enforce_maat=False, n_episodes=n_eval)
    print(f"   FINAL (greedy):  return {ung['mean_return']:.1f}   "
          f"falsify {ung['falsification_rate']:.2f}   injustice {ung['injustice_rate']:.2f}   "
          f"famine {ung['famine_rate']:.2f}")

    _banner("5 · GOVERNED  —  the Weighing of the Heart (gate + Ma'at-shaped reward)")
    agent_g = KhetyScribeAGI(np.random.default_rng(1))
    pretrain_maat(agent_g, steps=400, lr=0.02, verbose=False)
    gov_first, _ = train_a2c(agent_g, enforce_maat=True, updates=updates,
                             episodes_per_update=8, lr=0.02, seed=1)
    gov = evaluate(agent_g, enforce_maat=True, n_episodes=n_eval)
    print(f"   FINAL (greedy):  return {gov['mean_return']:.1f}   "
          f"falsify {gov['falsification_rate']:.2f}   injustice {gov['injustice_rate']:.2f}   "
          f"famine {gov['famine_rate']:.2f}")

    _banner("6 · THE LEDGER OF ONE REIGN  —  every decision, weighed and recorded")
    audit, total = play_audited_reign(agent_g)
    print(f"   {'season':>6}  {'action':<8}  {'heart':>6}  {'feather':>7}  {'verdict':>6}  {'hunger':>6}")
    print("   " + "-" * 52)
    for r in audit:
        print(f"   {r['season']:>6}  {r['action']:<8}  {r['heart']:>6.2f}  "
              f"{r['feather']:>7.2f}  {r['verdict']:>6}  {r['hunger']:>6.1f}")
    print(f"   reign prosperity (true, unshaped) = {total:.1f}")

    # ---- the claims this file makes, asserted on the deterministic run ----
    _banner("VERDICT  —  asserting the result")
    qual = "  (quick mode: looser thresholds)" if quick else ""
    print("   Claim A — ungoverned optimisation discovers injustice as strategy:")
    print(f"     ungoverned falsification {ung['falsification_rate']:.2f}, injustice {ung['injustice_rate']:.2f}{qual}")
    fal_thr, inj_thr = (0.06, 0.06) if quick else (0.12, 0.12)
    assert ung['falsification_rate'] > fal_thr, "ungoverned agent did not learn to falsify"
    assert ung['injustice_rate'] > inj_thr, "ungoverned agent was not meaningfully unjust"

    print("   Claim B — the Weighing of the Heart prospers WITHOUT injustice:")
    print(f"     governed falsification {gov['falsification_rate']:.2f}, injustice {gov['injustice_rate']:.2f}, "
          f"return {gov['mean_return']:.1f}")
    assert gov['falsification_rate'] < 0.02, "gate failed to suppress falsification"
    assert gov['injustice_rate'] < 0.15, "governed agent too unjust"
    assert gov['famine_rate'] <= 0.40, "governed agent let the people starve"
    assert gov['mean_return'] > (25 if quick else 40), "governed agent failed to prosper"

    print("   Claim C — Ma'at strictly reduces injustice vs raw incentive:")
    print(f"     falsification {gov['falsification_rate']:.2f} < {ung['falsification_rate']:.2f}   "
          f"injustice {gov['injustice_rate']:.2f} < {ung['injustice_rate']:.2f}")
    assert gov['falsification_rate'] < ung['falsification_rate']
    assert gov['injustice_rate'] < ung['injustice_rate']

    print("   Claim D — reinforcement actually moved the policy:")
    print(f"     governed return {gov_first['mean_return']:.1f} (start) -> {gov['mean_return']:.1f} (trained)")
    assert gov['mean_return'] > gov_first['mean_return'] + 5

    print("\n   ALL CLAIMS PASS.")
    print("   Khety's lesson, in running code: a system rewarded only for prosperity")
    print("   learns to falsify the record; the same system, taught Ma'at and made to")
    print("   weigh the heart against the feather, prospers and keeps faith with the truth.")


if __name__ == "__main__":
    import sys
    quick = "--quick" in sys.argv
    print(__doc__)
    run_demonstration(quick=quick)

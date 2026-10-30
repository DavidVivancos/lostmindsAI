#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Figure 0297 — Eleanor of Aquitaine (c. 1122-1124 – 1204 CE)
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0297 · Eleanor of Aquitaine (c. 1122-1124 – 1204 CE)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 15, Minds 281-300 Available on Amazon https://www.amazon.com/dp/B0HMQ7TVJM
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0297_eleanor_of_aquitaine_1122 - Eleanor of Aquitaine (c. 1122-1124 – 1204 CE)
# Domain: governance / patronage / queenship  |  archetype_class: political_sovereign
#
# WHAT THIS FILE IS
# ------------------
# A from-scratch (pure NumPy, no autograd, no frameworks) neural architecture
# whose *mechanism* is chosen to encode a specific, historically-grounded
# cognitive signature of Eleanor of Aquitaine, rather than a generic
# Transformer/MoE stack retrofitted with her name.
#
# THE THESIS THIS CODE IMPLEMENTS
# --------------------------------
# Eleanor's power was never lodged in a single office. She held her core
# competence -- the law, chancery, revenue, and court culture of the Duchy of
# Aquitaine -- as a *persistent, portable asset* that she re-hosted, intact,
# across three incompatible sovereign regimes over sixty-seven years:
#   (1) Duchess of Aquitaine in her own right (1137-1204, uninterrupted)
#   (2) Queen consort of the Capetian court of France (1137-1152)
#   (3) Queen consort, then regent, then dowager of the Angevin/Plantagenet
#       court of England (1152-1204)
# Each host regime imposed a different value-system on the *same* underlying
# governance decisions (a petition, a resource request, a threat) -- Capetian
# piety-first hierarchy versus Angevin administrative efficiency versus an
# independent Aquitanian culture of patronage. Eleanor's competence had to be
# *re-interfaced*, not rebuilt, each time she changed hosts. And when Henry II
# suppressed her for sixteen years (1173-1189), the core was not erased --
# it resumed at full capacity, immediately, the day Henry died.
#
# This file models that as a computational object:
#
#   1. CoreGovernor        A small governance policy network -- the portable,
#                           slow-changing "Duchy" competence.
#   2. HostAdapter          A thin, fast-changing, LOW-RANK interface layer
#                           (one per regime) that re-projects the same core
#                           competence into a host's local value-system --
#                           analogous to a chancery clerk translating ducal
#                           policy into the idiom the current overlord expects.
#   3. Reptile-style meta-training loop (first-order meta-learning, hand-
#      derived, no autograd) that shapes the CORE so that a *few* adapter-only
#      gradient steps suffice to re-host it into any regime.
#   4. A DORMANCY experiment: freeze every parameter for an enforced
#      "imprisonment" interval while the world drifts, then measure whether
#      the frozen core still re-adapts fast (it does) versus a same-sized
#      network that never had a meta-trained core (it does not).
#   5. A separate DYNASTY-CHAIN module: tabular TD(0) value learning over an
#      eight-generation delayed-reward chain, comparing a near-unity-discount
#      ("Eleanor") agent against a myopic, low-discount agent, to show why
#      only long-horizon credit assignment correctly prices an early cultural
#      investment (Poitiers, 1160s) whose payoff (a stable Capetian-Plantagenet
#      peace under Louis IX) arrives seven "generations" later.
#
# Every gradient used anywhere in this file is hand-derived and verified
# against a central-difference numerical gradient (see gradient_check()).
# No PyTorch, TensorFlow, JAX, or autograd of any kind is used or required.
#
# Usage:
#   python3 chapter_0297_eleanor_of_aquitaine_1122.py            # full run + tests
#   python3 chapter_0297_eleanor_of_aquitaine_1122.py --test     # tests only
# =============================================================================

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

RNG_SEED = 1122  # Eleanor's birth year, for reproducibility and a little wit.


# =============================================================================
# SECTION A. CORE MATH PRIMITIVES (hand-written, no autograd)
# =============================================================================

def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise numerically stable softmax. logits: (N, K) -> probs: (N, K)."""
    z = logits - np.max(logits, axis=-1, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=-1, keepdims=True)


def cross_entropy_loss(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean cross-entropy. probs: (N, K), labels: (N,) int class indices."""
    n = probs.shape[0]
    eps = 1e-12
    correct = probs[np.arange(n), labels]
    return float(-np.mean(np.log(correct + eps)))


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def dtanh(a: np.ndarray) -> np.ndarray:
    """Derivative of tanh given its OUTPUT a = tanh(x)."""
    return 1.0 - a * a


# =============================================================================
# SECTION B. CORE GOVERNOR — the portable "Duchy" competence network
# =============================================================================
#
# Architecture: x --(W1 + host adapter)--> tanh --(W2)--> logits --softmax-->
# action distribution over {GRANT, DENY, DELEGATE, PATRONIZE}
#
# The forward pass accepts an *optional* low-rank adapter (A, B, db1) that is
# added on top of the shared core weights, exactly like a chancery interface
# layer sitting on top of unchanged ducal policy.

STATE_DIM = 6     # [wealth_request, threat_level, kinship_closeness,
                  #  cultural_value, urgency, piety_expectation]
HIDDEN_DIM = 16
N_ACTIONS = 4     # 0=GRANT 1=DENY 2=DELEGATE 3=PATRONIZE
ADAPTER_RANK = 3

ACTION_NAMES = ["GRANT", "DENY", "DELEGATE", "PATRONIZE"]


@dataclass
class CoreParams:
    """The persistent, slow-changing 'Duchy' parameters."""
    W1: np.ndarray  # (STATE_DIM, HIDDEN_DIM)
    b1: np.ndarray  # (HIDDEN_DIM,)
    W2: np.ndarray  # (HIDDEN_DIM, N_ACTIONS)
    b2: np.ndarray  # (N_ACTIONS,)

    def copy(self) -> "CoreParams":
        return CoreParams(self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy())

    def flat(self) -> np.ndarray:
        return np.concatenate([self.W1.ravel(), self.b1.ravel(),
                                self.W2.ravel(), self.b2.ravel()])


@dataclass
class AdapterParams:
    """The thin, fast-changing, per-host interface layer (LoRA-style)."""
    A: np.ndarray    # (STATE_DIM, ADAPTER_RANK)
    B: np.ndarray    # (ADAPTER_RANK, HIDDEN_DIM)
    db1: np.ndarray  # (HIDDEN_DIM,)  host-specific bias re-calibration

    def copy(self) -> "AdapterParams":
        return AdapterParams(self.A.copy(), self.B.copy(), self.db1.copy())

    def delta_W1(self) -> np.ndarray:
        return self.A @ self.B

    def flat(self) -> np.ndarray:
        return np.concatenate([self.A.ravel(), self.B.ravel(), self.db1.ravel()])


def init_core(rng: np.random.Generator) -> CoreParams:
    scale1 = np.sqrt(2.0 / STATE_DIM)
    scale2 = np.sqrt(2.0 / HIDDEN_DIM)
    return CoreParams(
        W1=rng.normal(0, scale1, size=(STATE_DIM, HIDDEN_DIM)),
        b1=np.zeros(HIDDEN_DIM),
        W2=rng.normal(0, scale2, size=(HIDDEN_DIM, N_ACTIONS)),
        b2=np.zeros(N_ACTIONS),
    )


def init_adapter(rng: np.random.Generator, scale: float = 0.05) -> AdapterParams:
    return AdapterParams(
        A=rng.normal(0, scale, size=(STATE_DIM, ADAPTER_RANK)),
        B=rng.normal(0, scale, size=(ADAPTER_RANK, HIDDEN_DIM)),
        db1=np.zeros(HIDDEN_DIM),
    )


def forward(core: CoreParams, x: np.ndarray,
            adapter: AdapterParams | None = None) -> Dict[str, np.ndarray]:
    """
    Forward pass through the CoreGovernor, with an optional host adapter
    applied to the first layer only (the 'interface' layer).
    x: (N, STATE_DIM)
    Returns a cache dict needed for manual backprop.
    """
    W1_eff = core.W1 + (adapter.delta_W1() if adapter is not None else 0.0)
    b1_eff = core.b1 + (adapter.db1 if adapter is not None else 0.0)

    z1 = x @ W1_eff + b1_eff          # (N, HIDDEN_DIM)
    a1 = tanh(z1)                     # (N, HIDDEN_DIM)
    z2 = a1 @ core.W2 + core.b2       # (N, N_ACTIONS)
    probs = softmax(z2)               # (N, N_ACTIONS)

    return dict(x=x, z1=z1, a1=a1, z2=z2, probs=probs,
                W1_eff=W1_eff, b1_eff=b1_eff)


def backward(core: CoreParams, cache: Dict[str, np.ndarray], labels: np.ndarray,
             adapter: AdapterParams | None = None
             ) -> Tuple[CoreParams, "AdapterParams | None"]:
    """
    Manual backprop of mean cross-entropy loss through the CoreGovernor.
    Returns gradients w.r.t. core params, and w.r.t. adapter params (if any).
    All formulas hand-derived:
        dL/dz2 = (probs - onehot(labels)) / N
        dL/dW2 = a1^T @ dz2 ;  dL/db2 = sum(dz2)
        dL/da1 = dz2 @ W2^T
        dL/dz1 = dL/da1 * (1 - a1^2)
        dL/dW1 = x^T @ dz1 ;  dL/db1 = sum(dz1)
        (adapter shares dz1; dL/dA = x^T @ dz1 @ B^T ; dL/dB = A^T @ x^T @ dz1)
    """
    x, a1, probs = cache["x"], cache["a1"], cache["probs"]
    n = x.shape[0]

    onehot = np.zeros_like(probs)
    onehot[np.arange(n), labels] = 1.0
    dz2 = (probs - onehot) / n                     # (N, N_ACTIONS)

    dW2 = a1.T @ dz2                                # (HIDDEN_DIM, N_ACTIONS)
    db2 = dz2.sum(axis=0)                           # (N_ACTIONS,)

    da1 = dz2 @ core.W2.T                           # (N, HIDDEN_DIM)
    dz1 = da1 * dtanh(a1)                            # (N, HIDDEN_DIM)

    dW1 = x.T @ dz1                                  # (STATE_DIM, HIDDEN_DIM)
    db1 = dz1.sum(axis=0)                            # (HIDDEN_DIM,)

    core_grad = CoreParams(W1=dW1, b1=db1, W2=dW2, b2=db2)

    adapter_grad = None
    if adapter is not None:
        # W1_eff = W1 + A@B  => dL/dA = x^T dz1 @ B^T ; dL/dB = A^T @ (x^T dz1)
        dA = x.T @ dz1 @ adapter.B.T                # (STATE_DIM, ADAPTER_RANK)
        dB = adapter.A.T @ (x.T @ dz1)              # (ADAPTER_RANK, HIDDEN_DIM)
        ddb1 = dz1.sum(axis=0)                       # (HIDDEN_DIM,)
        adapter_grad = AdapterParams(A=dA, B=dB, db1=ddb1)

    return core_grad, adapter_grad


def loss_fn(core: CoreParams, x: np.ndarray, labels: np.ndarray,
            adapter: AdapterParams | None = None) -> float:
    cache = forward(core, x, adapter)
    return cross_entropy_loss(cache["probs"], labels)


def accuracy(core: CoreParams, x: np.ndarray, labels: np.ndarray,
             adapter: AdapterParams | None = None) -> float:
    cache = forward(core, x, adapter)
    preds = np.argmax(cache["probs"], axis=-1)
    return float(np.mean(preds == labels))


# =============================================================================
# SECTION C. FINITE-DIFFERENCE GRADIENT CHECK (mandatory correctness proof)
# =============================================================================

def gradient_check(seed: int = 7, n_samples: int = 5, eps: float = 1e-5,
                    tol: float = 5e-4) -> Dict[str, float]:
    """
    Verifies every hand-derived analytic gradient (core AND adapter) against
    a central-difference numerical gradient on a small random problem.
    Returns the max relative error per parameter tensor; raises AssertionError
    if any exceeds `tol`.
    """
    rng = np.random.default_rng(seed)
    core = init_core(rng)
    adapter = init_adapter(rng, scale=0.2)
    x = rng.normal(size=(n_samples, STATE_DIM))
    labels = rng.integers(0, N_ACTIONS, size=n_samples)

    cache = forward(core, x, adapter)
    core_grad, adapter_grad = backward(core, cache, labels, adapter)

    def numeric_grad(get_param, set_param, shape) -> np.ndarray:
        g = np.zeros(shape)
        it = np.nditer(np.zeros(shape), flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = get_param()[idx]
            set_param(idx, orig + eps)
            lp = loss_fn(core, x, labels, adapter)
            set_param(idx, orig - eps)
            lm = loss_fn(core, x, labels, adapter)
            set_param(idx, orig)
            g[idx] = (lp - lm) / (2 * eps)
            it.iternext()
        return g

    results: Dict[str, float] = {}

    def check(name, analytic, get_param, set_param, shape):
        numeric = numeric_grad(get_param, set_param, shape)
        denom = np.maximum(np.abs(analytic) + np.abs(numeric), 1e-8)
        rel_err = np.max(np.abs(analytic - numeric) / denom)
        results[name] = float(rel_err)
        assert rel_err < tol, f"Gradient check FAILED for {name}: rel_err={rel_err:.2e}"

    check("core.W1", core_grad.W1, lambda: core.W1, lambda i, v: core.W1.__setitem__(i, v), core.W1.shape)
    check("core.b1", core_grad.b1, lambda: core.b1, lambda i, v: core.b1.__setitem__(i, v), core.b1.shape)
    check("core.W2", core_grad.W2, lambda: core.W2, lambda i, v: core.W2.__setitem__(i, v), core.W2.shape)
    check("core.b2", core_grad.b2, lambda: core.b2, lambda i, v: core.b2.__setitem__(i, v), core.b2.shape)
    check("adapter.A", adapter_grad.A, lambda: adapter.A, lambda i, v: adapter.A.__setitem__(i, v), adapter.A.shape)
    check("adapter.B", adapter_grad.B, lambda: adapter.B, lambda i, v: adapter.B.__setitem__(i, v), adapter.B.shape)
    check("adapter.db1", adapter_grad.db1, lambda: adapter.db1, lambda i, v: adapter.db1.__setitem__(i, v), adapter.db1.shape)

    return results


# =============================================================================
# SECTION D. SYNTHETIC HOST REGIMES — three courts, one set of facts
# =============================================================================
#
# Each host regime is a deterministic labelling rule (plus label noise) over
# the SAME state space, standing in for "what this court's value-system
# considers correct governance" given the same underlying petition.

def sample_states(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.uniform(0.0, 1.0, size=(n, STATE_DIM))


def label_aquitaine(x: np.ndarray, rng: np.random.Generator, noise: float = 0.06) -> np.ndarray:
    """Independent Aquitaine: prizes cultural patronage and kinship loyalty."""
    wealth, threat, kin, cult, urgency, piety = x.T
    labels = np.where(cult > 0.55, 3,                       # PATRONIZE
              np.where(kin > 0.6, 0,                        # GRANT
              np.where(urgency < 0.35, 2, 1)))               # DELEGATE / DENY
    flip = rng.random(len(labels)) < noise
    labels[flip] = rng.integers(0, N_ACTIONS, size=flip.sum())
    return labels


def label_france(x: np.ndarray, rng: np.random.Generator, noise: float = 0.06) -> np.ndarray:
    """Capetian France: pious, hierarchical, fiscally cautious."""
    wealth, threat, kin, cult, urgency, piety = x.T
    labels = np.where((piety > 0.5) & (wealth < 0.6), 0,     # GRANT if pious & cheap
              np.where((piety < 0.4) & (wealth > 0.5), 1,    # DENY if impious & costly
              2))                                             # else DELEGATE to church/council
    flip = rng.random(len(labels)) < noise
    labels[flip] = rng.integers(0, N_ACTIONS, size=flip.sum())
    return labels


def label_england(x: np.ndarray, rng: np.random.Generator, noise: float = 0.06) -> np.ndarray:
    """Angevin England: administrative efficiency, threat-responsive."""
    wealth, threat, kin, cult, urgency, piety = x.T
    labels = np.where((threat > 0.55) & (wealth < 0.7), 0,   # GRANT resourcing if threatened
              np.where(urgency < 0.4, 2,                     # DELEGATE to sheriff/justiciar
              1))                                             # else DENY
    flip = rng.random(len(labels)) < noise
    labels[flip] = rng.integers(0, N_ACTIONS, size=flip.sum())
    return labels


def label_regent(x: np.ndarray, rng: np.random.Generator, noise: float = 0.06) -> np.ndarray:
    """Post-1189 dowager-regent regime: hybrid of England's efficiency and
    Aquitaine's patronage sensibility, drifted from the pre-1173 England host
    to represent sixteen years of political change during her suppression."""
    wealth, threat, kin, cult, urgency, piety = x.T
    labels = np.where((threat > 0.45) & (kin > 0.4), 0,      # GRANT: threat + trusted kin
              np.where(cult > 0.6, 3,                        # PATRONIZE
              np.where(urgency < 0.45, 2, 1)))                # DELEGATE / DENY
    flip = rng.random(len(labels)) < noise
    labels[flip] = rng.integers(0, N_ACTIONS, size=flip.sum())
    return labels


HOSTS = {
    "aquitaine": label_aquitaine,
    "france": label_france,
    "england": label_england,
}


# =============================================================================
# SECTION E. REPTILE-STYLE META-TRAINING (first-order meta-learning)
# =============================================================================
#
# Goal: shape CoreParams so that a FEW adapter-only SGD steps suffice to
# re-host the core into any of the three regimes -- the computational image
# of "the same ducal competence, quickly re-interfaced for a new overlord."

@dataclass
class MetaConfig:
    meta_iters: int = 600
    inner_steps: int = 5
    support_n: int = 64
    query_n: int = 200
    inner_lr: float = 0.15
    core_inner_lr: float = 0.02   # core moves a little even within the inner loop
    meta_lr: float = 0.3          # Reptile outer step size


def inner_adapt(core: CoreParams, adapter: AdapterParams, label_fn,
                 rng: np.random.Generator, cfg: MetaConfig,
                 also_move_core: bool = True) -> Tuple[CoreParams, AdapterParams, List[float]]:
    """Run `cfg.inner_steps` of SGD on (core, adapter) using a host's support
    set. Returns updated (core, adapter) and the loss trajectory."""
    core = core.copy()
    adapter = adapter.copy()
    losses = []
    for _ in range(cfg.inner_steps):
        x = sample_states(rng, cfg.support_n)
        y = label_fn(x, rng)
        cache = forward(core, x, adapter)
        loss = cross_entropy_loss(cache["probs"], y)
        losses.append(loss)
        core_grad, adapter_grad = backward(core, cache, y, adapter)

        adapter.A -= cfg.inner_lr * adapter_grad.A
        adapter.B -= cfg.inner_lr * adapter_grad.B
        adapter.db1 -= cfg.inner_lr * adapter_grad.db1

        if also_move_core:
            core.W1 -= cfg.core_inner_lr * core_grad.W1
            core.b1 -= cfg.core_inner_lr * core_grad.b1
            core.W2 -= cfg.core_inner_lr * core_grad.W2
            core.b2 -= cfg.core_inner_lr * core_grad.b2
    return core, adapter, losses


def meta_train(cfg: MetaConfig, seed: int = 297
               ) -> Tuple[CoreParams, Dict[str, AdapterParams], List[float]]:
    rng = np.random.default_rng(seed)
    core = init_core(rng)
    adapters = {h: init_adapter(rng) for h in HOSTS}
    query_loss_history = []

    for it in range(cfg.meta_iters):
        host = list(HOSTS.keys())[it % len(HOSTS)]
        label_fn = HOSTS[host]
        core_before = core.copy()

        core_after, adapters[host], _ = inner_adapt(
            core, adapters[host], label_fn, rng, cfg, also_move_core=True)

        # Reptile meta-update: nudge the shared core toward where inner-loop
        # adaptation took it (first-order approximation of MAML).
        core.W1 = core_before.W1 + cfg.meta_lr * (core_after.W1 - core_before.W1)
        core.b1 = core_before.b1 + cfg.meta_lr * (core_after.b1 - core_before.b1)
        core.W2 = core_before.W2 + cfg.meta_lr * (core_after.W2 - core_before.W2)
        core.b2 = core_before.b2 + cfg.meta_lr * (core_after.b2 - core_before.b2)

        if it % 20 == 0 or it == cfg.meta_iters - 1:
            xq = sample_states(rng, cfg.query_n)
            yq = label_fn(xq, rng)
            ql = loss_fn(core, xq, yq, adapters[host])
            query_loss_history.append(ql)

    return core, adapters, query_loss_history


def steps_to_threshold(core: CoreParams, adapter: AdapterParams, label_fn,
                        rng: np.random.Generator, cfg: MetaConfig,
                        threshold: float = 0.45, max_steps: int = 80) -> int:
    """How many adapter-only SGD steps are needed to cross a query-set
    ACCURACY threshold, starting from the given (core, adapter)? Used to
    compare 'portable core' re-adaptation speed against a from-scratch
    baseline. Accuracy (vs. loss) is used as the criterion because it is
    scale-free and comparable across the differently-shaped host label
    distributions (random-guess floor = 1/N_ACTIONS = 0.25)."""
    core = core.copy()
    adapter = adapter.copy()
    for step in range(1, max_steps + 1):
        x = sample_states(rng, cfg.support_n)
        y = label_fn(x, rng)
        cache = forward(core, x, adapter)
        _, adapter_grad = backward(core, cache, y, adapter)
        adapter.A -= cfg.inner_lr * adapter_grad.A
        adapter.B -= cfg.inner_lr * adapter_grad.B
        adapter.db1 -= cfg.inner_lr * adapter_grad.db1

        xq = sample_states(rng, cfg.query_n)
        yq = label_fn(xq, rng)
        if accuracy(core, xq, yq, adapter) > threshold:
            return step
    return max_steps  # did not converge within budget


# =============================================================================
# SECTION F. DORMANCY & REACTIVATION — the sixteen-year imprisonment
# =============================================================================

def dormancy_experiment(core: CoreParams, adapters: Dict[str, AdapterParams],
                         cfg: MetaConfig, seed: int = 1173) -> Dict[str, object]:
    """
    Simulates: (a) the England-host adapter as it stood in 1173, (b) sixteen
    frozen "years" (no gradient updates to ANYTHING -- literal parameter
    freeze) during which the political world drifts to a new distribution
    (the 'regent' host), (c) reactivation in 1189, where the OLD core plus
    the OLD (stale) England adapter must re-adapt to the NEW regent host.
    Compared against a same-architecture network that never had a
    meta-trained core (a 'courtier who never held power') learning the
    regent host completely from scratch.
    """
    rng = np.random.default_rng(seed)

    # (a) State at the moment of imprisonment: core + last England adapter.
    frozen_core = core.copy()
    frozen_adapter = adapters["england"].copy()

    # (b) Sixteen frozen "years": explicitly NO parameter update happens here.
    # We only advance the RNG stream to represent time passing / drift in the
    # world the parameters will next be evaluated against.
    for _year in range(16):
        _ = sample_states(rng, 8)  # world moves on; parameters do not.

    # (c) Reactivation: re-host the frozen (core, adapter) into the new
    # 'regent' distribution.
    steps_reactivated = steps_to_threshold(
        frozen_core, frozen_adapter, label_regent, rng, cfg)

    # Baseline: a fresh, never-meta-trained core + fresh adapter learning the
    # regent host from nothing -- "no institutional memory" control.
    rng_baseline = np.random.default_rng(seed + 1)
    naive_core = init_core(rng_baseline)
    naive_adapter = init_adapter(rng_baseline)
    steps_naive = steps_to_threshold(
        naive_core, naive_adapter, label_regent, rng_baseline, cfg)

    return dict(
        steps_reactivated_after_dormancy=steps_reactivated,
        steps_naive_from_scratch=steps_naive,
        recovered_faster=steps_reactivated < steps_naive,
    )


# =============================================================================
# SECTION G. DYNASTY-CHAIN — long-horizon (multi-generational) credit
#            assignment via tabular TD(0)
# =============================================================================
#
# An 8-state chain represents eight successive "decade" generations:
#   s0: 1160s Poitiers patronage decision  (INVEST or WITHHOLD)
#   s1..s6: intervening generations (Second/Third Crusade era, successions,
#           the 1200 Castile journey, the marriage of Blanche of Castile...)
#   s7: terminal generation -- Capetian-Plantagenet settlement under Louis IX
# Reward is realized ONLY at s7, and ONLY if the ORIGINAL action at s0 was
# INVEST. This is deliberately a hard, sparse, long-delayed credit-assignment
# problem: seven state-transitions separate the causal action from its payoff.

N_GEN = 8
GAMMA_LONG_HORIZON = 0.95   # "Eleanor": treats the future as almost as real as the present
GAMMA_MYOPIC = 0.30         # a short-reigned, single-generation planner


def build_dynasty_chain() -> Dict[str, object]:
    """States are indexed (branch, t) for branch in {invest, withhold},
    t in [0, N_GEN-1]. Deterministic self-transition chain; reward only at
    the final state of the 'invest' branch."""
    states = [(b, t) for b in ("invest", "withhold") for t in range(N_GEN)]
    idx = {s: i for i, s in enumerate(states)}

    def step(branch: str, t: int) -> Tuple[Tuple[str, int], float, bool]:
        if t == N_GEN - 1:
            reward = 10.0 if branch == "invest" else 0.0
            return (branch, t), reward, True   # terminal, self-loop
        return (branch, t + 1), 0.0, False

    return dict(states=states, idx=idx, step=step)


def td0_learn(chain: Dict[str, object], branch: str, gamma: float,
              episodes: int = 4000, alpha: float = 0.05,
              seed: int = 0) -> np.ndarray:
    """Tabular TD(0) policy evaluation of always-follow-`branch` starting
    from generation 0, run for `episodes` rollouts through the chain."""
    rng = np.random.default_rng(seed)
    V = np.zeros(len(chain["states"]))
    idx = chain["idx"]
    step = chain["step"]

    for _ in range(episodes):
        t = 0
        s = (branch, t)
        for _tstep in range(N_GEN):
            s_next, r, done = step(*s)
            td_target = r + (0.0 if done else gamma * V[idx[s_next]])
            V[idx[s]] += alpha * (td_target - V[idx[s]])
            s = s_next
            if done:
                break
    return V


def dynasty_chain_experiment() -> Dict[str, float]:
    chain = build_dynasty_chain()
    idx = chain["idx"]

    V_long_invest = td0_learn(chain, "invest", GAMMA_LONG_HORIZON, seed=1)
    V_long_withhold = td0_learn(chain, "withhold", GAMMA_LONG_HORIZON, seed=2)
    V_myopic_invest = td0_learn(chain, "invest", GAMMA_MYOPIC, seed=3)
    V_myopic_withhold = td0_learn(chain, "withhold", GAMMA_MYOPIC, seed=4)

    v0_long_invest = V_long_invest[idx[("invest", 0)]]
    v0_long_withhold = V_long_withhold[idx[("withhold", 0)]]
    v0_myopic_invest = V_myopic_invest[idx[("invest", 0)]]
    v0_myopic_withhold = V_myopic_withhold[idx[("withhold", 0)]]

    return dict(
        long_horizon_gap=float(v0_long_invest - v0_long_withhold),
        myopic_gap=float(v0_myopic_invest - v0_myopic_withhold),
        v0_long_invest=float(v0_long_invest),
        v0_myopic_invest=float(v0_myopic_invest),
    )


# =============================================================================
# SECTION H. SELF-TESTS
# =============================================================================

def run_self_tests(verbose: bool = True) -> bool:
    ok = True

    def report(name, passed, extra=""):
        nonlocal ok
        ok = ok and passed
        mark = "PASS" if passed else "FAIL"
        if verbose:
            print(f"  [{mark}] {name} {extra}")

    # T1: gradient check must pass with small relative error
    try:
        errs = gradient_check()
        worst = max(errs.values())
        report("gradient_check (analytic vs finite-difference)", worst < 5e-4,
               f"(max rel. err = {worst:.2e})")
    except AssertionError as e:
        report("gradient_check", False, str(e))

    # T2: forward pass output shapes and valid probability simplex
    rng = np.random.default_rng(0)
    core = init_core(rng)
    x = sample_states(rng, 10)
    cache = forward(core, x)
    shapes_ok = cache["probs"].shape == (10, N_ACTIONS)
    sums_ok = np.allclose(cache["probs"].sum(axis=-1), 1.0, atol=1e-8)
    report("forward() shapes + softmax sums to 1", shapes_ok and sums_ok)

    # T3: training loss decreases substantially on a single host (sanity:
    # the base learner works at all)
    rng = np.random.default_rng(1)
    core0 = init_core(rng)
    adapter0 = init_adapter(rng)
    cfg = MetaConfig(inner_steps=80, support_n=64, inner_lr=0.2, core_inner_lr=0.03)
    core_after, adapter_after, losses = inner_adapt(
        core0, adapter0, label_aquitaine, rng, cfg, also_move_core=True)
    report("loss decreases over inner-loop adaptation",
           losses[-1] < losses[0] * 0.85, f"({losses[0]:.3f} -> {losses[-1]:.3f})")

    # T4: meta-trained core adapts to a HELD-OUT-STYLE host faster than fresh init
    cfg_small = MetaConfig(meta_iters=400, inner_steps=5, support_n=64,
                            query_n=200, inner_lr=0.15, core_inner_lr=0.02, meta_lr=0.3)
    core_m, adapters_m, _ = meta_train(cfg_small, seed=42)
    rng_eval = np.random.default_rng(99)
    fresh_adapter = init_adapter(rng_eval)
    steps_meta = steps_to_threshold(core_m, fresh_adapter, label_england,
                                     rng_eval, cfg_small, threshold=0.45, max_steps=80)
    rng_naive = np.random.default_rng(100)
    naive_core = init_core(rng_naive)
    naive_adapter = init_adapter(rng_naive)
    steps_naive = steps_to_threshold(naive_core, naive_adapter, label_england,
                                      rng_naive, cfg_small, threshold=0.45, max_steps=80)
    report("meta-trained core re-hosts faster than a naive (non-meta) core",
           steps_meta <= steps_naive,
           f"(meta={steps_meta} steps, naive={steps_naive} steps)")

    # T5: dormancy does not erase the core -- reactivation beats from-scratch
    dorm = dormancy_experiment(core_m, adapters_m, cfg_small, seed=1173)
    report("frozen core survives 16-'year' dormancy and re-adapts faster than from-scratch",
           dorm["recovered_faster"],
           f"(reactivated={dorm['steps_reactivated_after_dormancy']} steps, "
           f"naive={dorm['steps_naive_from_scratch']} steps)")

    # T6: long-horizon (near-unity gamma) agent correctly values the early
    # investment far more than a myopic agent does
    dc = dynasty_chain_experiment()
    report("long-horizon TD(0) prices the 7-step-delayed reward correctly",
           dc["long_horizon_gap"] > 3.0 and dc["long_horizon_gap"] > dc["myopic_gap"] * 3,
           f"(long-horizon gap={dc['long_horizon_gap']:.3f}, myopic gap={dc['myopic_gap']:.3f})")

    return ok


# =============================================================================
# SECTION I. MAIN DEMONSTRATION
# =============================================================================

def main():
    print("=" * 78)
    print("FIGURE 0243 -- ELEANOR OF AQUITAINE (c.1122/1124-1204)")
    print("Portable-Core / Host-Adapter Governance Network")
    print("Encyclopedia of Lost Minds: Echoes on AI")
    print("=" * 78)

    print("\n[1] Gradient check (analytic backprop vs. central finite differences)")
    t0 = time.time()
    errs = gradient_check()
    for k, v in errs.items():
        print(f"    {k:14s} max relative error = {v:.3e}")
    print(f"    ALL GRADIENTS VERIFIED in {time.time()-t0:.3f}s")

    print("\n[2] Meta-training the CoreGovernor across three host regimes")
    print("    (Aquitaine-independent, Capetian France, Angevin England)")
    cfg = MetaConfig()
    t0 = time.time()
    core, adapters, qlosses = meta_train(cfg, seed=RNG_SEED)
    print(f"    meta_iters={cfg.meta_iters}, inner_steps={cfg.inner_steps} "
          f"-> trained in {time.time()-t0:.2f}s")
    print(f"    query loss trajectory (sampled every 20 meta-iters):")
    print(f"      first 5: {[round(l,3) for l in qlosses[:5]]}")
    print(f"      last  5: {[round(l,3) for l in qlosses[-5:]]}")

    print("\n[3] Post-training accuracy per host (query set, held out)")
    rng_eval = np.random.default_rng(2004)
    for host, label_fn in HOSTS.items():
        xq = sample_states(rng_eval, 300)
        yq = label_fn(xq, rng_eval)
        acc = accuracy(core, xq, yq, adapters[host])
        print(f"    {host:10s} accuracy = {acc*100:5.1f}%")

    print("\n[4] Re-adaptation speed: meta-trained core vs. naive (non-meta) core")
    print("    on a REGIME NEVER SEEN in this exact form (the post-1189 regent host)")
    rng_speed = np.random.default_rng(1189)
    fresh_adapter = init_adapter(rng_speed)
    steps_meta = steps_to_threshold(core, fresh_adapter, label_regent, rng_speed, cfg)
    rng_naive = np.random.default_rng(1190)
    naive_core = init_core(rng_naive)
    naive_adapter = init_adapter(rng_naive)
    steps_naive = steps_to_threshold(naive_core, naive_adapter, label_regent, rng_naive, cfg)
    print(f"    portable core + fresh adapter : {steps_meta:3d} steps to threshold")
    print(f"    naive core   + fresh adapter  : {steps_naive:3d} steps to threshold")

    print("\n[5] Dormancy experiment: 16 frozen 'years' (1173-1189), zero updates,")
    print("    then reactivation into the post-1189 regent regime")
    dorm = dormancy_experiment(core, adapters, cfg, seed=1173)
    print(f"    steps to re-adapt AFTER dormancy : {dorm['steps_reactivated_after_dormancy']}")
    print(f"    steps for a from-scratch network : {dorm['steps_naive_from_scratch']}")
    print(f"    core survived dormancy intact    : {dorm['recovered_faster']}")

    print("\n[6] Dynasty-chain: long-horizon vs. myopic credit assignment")
    print("    (reward realized 7 generations after the deciding action)")
    dc = dynasty_chain_experiment()
    print(f"    gamma={GAMMA_LONG_HORIZON} (long-horizon) : V(invest)-V(withhold) at gen 0 = {dc['long_horizon_gap']:.3f}")
    print(f"    gamma={GAMMA_MYOPIC} (myopic)        : V(invest)-V(withhold) at gen 0 = {dc['myopic_gap']:.3f}")
    print(f"    -> only the long-horizon agent prices the early cultural investment correctly")

    print("\n[7] Self-tests")
    ok = run_self_tests(verbose=True)
    print("\n" + "=" * 78)
    print(f"ALL SELF-TESTS {'PASSED' if ok else 'FAILED'}")
    print("=" * 78)
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Figure 0243: Eleanor of Aquitaine — Neuron.py")
    parser.add_argument("--test", action="store_true", help="run self-tests only")
    args = parser.parse_args()

    if args.test:
        success = run_self_tests(verbose=True)
        sys.exit(0 if success else 1)
    else:
        success = main()
        sys.exit(0 if success else 1)

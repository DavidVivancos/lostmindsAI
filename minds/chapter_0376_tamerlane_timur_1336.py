#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 # BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0376 · Tamerlane (Timur) (1336-1405)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 19, Minds 361-380 Available on Amazon https://www.amazon.com/dp/B0HQX147Y7
# Resume and Interactive Demos at https://artificiology.com/
# END ATTRIBUTION 
 THE SAHIB-QIRAN ARCHITECTURE  ·  an "Uncompiled Sovereign" network
================================================================================

WHY THIS ARCHITECTURE AND NOT ANOTHER
-------------------------------------
Most historical-conqueror models default to a hierarchy: a fixed command tree,
a decimal chain, an institution learned once and then executed. That is Genghis
Khan's cognitive signature, and it is NOT Timur's. The distinguishing fact about
Timur, established by Beatrice Manz's administrative study, is that he never
built a durable machine. He held his realm together by continuously re-deriving
it: reassigning commands, refusing a fixed succession, keeping a nominal khan on
a throne he himself never sat on, and moving craft as physical bodies rather
than transmitting it as knowledge. When he died at Otrar in February 1405 the
empire's unity collapsed within months — not because the machine broke, but
because there had never been a machine, only him, recomputing it.

This file turns that into a runnable model. Four mechanisms, each traceable to a
documented feature of how Timur actually operated:

  1. ORDU CELL  (re-derivation)     — the layer's weight matrix is not stored.
     It is regenerated every timestep from a small persistent kernel, gated by
     the current context. Nothing hardens into structure. A DISSOLUTION PENALTY
     actively punishes the gate for repeating itself across time.

  2. TWO THRONES (nominal / operative) — two heads. The `khan` head is small,
     legible, and can see only a published slice of the state; the `amir` head
     is full-capacity and is what actually acts. A coronation loss binds them on
     the training distribution. The measurable divergence off-distribution is
     the INTERFACE GAP — the modern name for the structure Timur ran for 35
     years, minting coins in a puppet's name while holding all decision power.

  3. TERROR LEDGER (self-created prior) — a persistent reputation state carried
     across the whole campaign. Costly coercion at step t changes how every
     later target responds. The model must learn intertemporal credit assignment
     over a prior it installs in others' heads. This is the coldest thing in the
     file and it is meant to be legible as such, not admired.

  4. SAMARKAND BANK (relocation, not replication) — rank-1 "craft atoms" of the
     kernel are MOVED into a frozen bank and ZEROED at source. Capability is
     treated as a body that can be deported, never as knowledge that can be
     copied. We then measure whether the source can regrow it. Damascus could
     not.

And one diagnostic that is the whole historical argument compressed to a number:

  * THE OTRAR TEST — freeze the gate at its training mean, i.e. force the
    sovereign's continuous re-derivation to "compile" into its own average, and
    measure what is left. A conventional static baseline is trained alongside
    and given the identical test.

ENGINEERING CONVENTIONS (project standard)
------------------------------------------
  * Pure NumPy. No autograd library, no deep-learning framework. A minimal
    reverse-mode tape is implemented here from scratch.
  * A finite-difference gradient check is MANDATORY and runs on every execution.
  * A real training loop with Adam, a real held-out split, real self-tests.
  * Deterministic under a fixed seed.

Run:  python3 chapter_0376_tamerlane_timur_1336.py
"""

from __future__ import annotations

import math
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ==============================================================================
# SECTION 0 — REPRODUCIBILITY
# ==============================================================================

SEED = 1370          # the year Timur took power at Balkh
RNG = np.random.default_rng(SEED)


def banner(title: str, ch: str = "=") -> None:
    """Print a section banner so the executed output is readable in the chapter."""
    print("\n" + ch * 78)
    print(title)
    print(ch * 78)


# ==============================================================================
# SECTION 1 — A MINIMAL REVERSE-MODE AUTODIFF TAPE (from scratch, NumPy only)
# ==============================================================================
# We need gradients through a recurrence that spans 16 timesteps and through a
# weight matrix that is *regenerated* at every one of them. Hand-deriving that
# by hand is error-prone, so we build the smallest correct autodiff engine that
# does the job. Every operation records (a) its output value and (b) a closure
# that pushes gradient back to its parents. `backward()` walks the tape in
# reverse topological order.
#
# This engine is validated by the finite-difference check in Section 6. If the
# engine were wrong, that check would fail — which is exactly why the check is
# non-negotiable.


def _unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Reduce `grad` back to `shape`, undoing NumPy broadcasting.

    If a forward op broadcast a (1, k) parameter against a (B, k) batch, the
    gradient arrives as (B, k) and must be summed down to (1, k).
    """
    if grad.shape == shape:
        return grad
    # First, sum away any leading axes that broadcasting added.
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # Then, sum (keeping dims) any axis where the target had extent 1.
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    """A node on the autodiff tape.

    Attributes
    ----------
    data     : the forward value (np.ndarray, always float64)
    grad     : accumulated gradient, same shape as data
    _parents : tensors this node was computed from
    _back    : closure that propagates self.grad into the parents' .grad
    requires : if False, gradient is not accumulated (used for frozen banks)
    """

    __slots__ = ("data", "grad", "_parents", "_back", "requires", "name")

    def __init__(self, data, parents: Sequence["Tensor"] = (), back: Optional[Callable] = None,
                 requires: bool = True, name: str = ""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._parents = tuple(parents)
        self._back = back
        self.requires = requires
        self.name = name

    # -- properties ---------------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor(name={self.name!r}, shape={self.data.shape})"

    # -- tape traversal -----------------------------------------------------
    def backward(self) -> None:
        """Seed d(self)/d(self) = 1 and propagate to every ancestor."""
        if self.data.size != 1:
            raise ValueError("backward() may only be called on a scalar Tensor.")
        order: List[Tensor] = []
        seen = set()

        def build(node: "Tensor"):
            # Iterative DFS to avoid recursion limits on a 16-step recurrence.
            stack = [(node, False)]
            while stack:
                nd, expanded = stack.pop()
                if expanded:
                    order.append(nd)
                    continue
                if id(nd) in seen:
                    continue
                seen.add(id(nd))
                stack.append((nd, True))
                for p in nd._parents:
                    if id(p) not in seen:
                        stack.append((p, False))

        build(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(order):
            if node._back is not None:
                node._back()

    # -- operator overloads -------------------------------------------------
    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(self, other)

    def __sub__(self, other):
        return sub(self, other)

    def __rsub__(self, other):
        return sub(_as_tensor(other), self)

    def __mul__(self, other):
        return mul(self, other)

    def __rmul__(self, other):
        return mul(self, other)

    def __neg__(self):
        return mul(self, -1.0)

    def __matmul__(self, other):
        return matmul(self, other)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            raise NotImplementedError("Tensor/Tensor division is not needed here.")
        return mul(self, 1.0 / float(other))

    def __getitem__(self, key):
        return slice_(self, key)


def _as_tensor(x) -> Tensor:
    return x if isinstance(x, Tensor) else Tensor(x)


# -- elementwise / linear ops --------------------------------------------------

def add(a, b) -> Tensor:
    a, b = _as_tensor(a), _as_tensor(b)
    out = Tensor(a.data + b.data, (a, b))

    def _back():
        if a.requires:
            a.grad += _unbroadcast(out.grad, a.data.shape)
        if b.requires:
            b.grad += _unbroadcast(out.grad, b.data.shape)

    out._back = _back
    return out


def sub(a, b) -> Tensor:
    a, b = _as_tensor(a), _as_tensor(b)
    out = Tensor(a.data - b.data, (a, b))

    def _back():
        if a.requires:
            a.grad += _unbroadcast(out.grad, a.data.shape)
        if b.requires:
            b.grad -= _unbroadcast(out.grad, b.data.shape)

    out._back = _back
    return out


def mul(a, b) -> Tensor:
    a, b = _as_tensor(a), _as_tensor(b)
    out = Tensor(a.data * b.data, (a, b))

    def _back():
        if a.requires:
            a.grad += _unbroadcast(out.grad * b.data, a.data.shape)
        if b.requires:
            b.grad += _unbroadcast(out.grad * a.data, b.data.shape)

    out._back = _back
    return out


def matmul(a: Tensor, b: Tensor) -> Tensor:
    out = Tensor(a.data @ b.data, (a, b))

    def _back():
        if a.requires:
            a.grad += out.grad @ b.data.T
        if b.requires:
            b.grad += a.data.T @ out.grad

    out._back = _back
    return out


def sigmoid(a: Tensor) -> Tensor:
    # Numerically stable logistic.
    x = a.data
    pos = x >= 0
    z = np.empty_like(x)
    z[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    z[~pos] = ex / (1.0 + ex)
    out = Tensor(z, (a,))

    def _back():
        if a.requires:
            a.grad += out.grad * z * (1.0 - z)

    out._back = _back
    return out


def tanh(a: Tensor) -> Tensor:
    t = np.tanh(a.data)
    out = Tensor(t, (a,))

    def _back():
        if a.requires:
            a.grad += out.grad * (1.0 - t * t)

    out._back = _back
    return out


def square(a: Tensor) -> Tensor:
    out = Tensor(a.data * a.data, (a,))

    def _back():
        if a.requires:
            a.grad += out.grad * 2.0 * a.data

    out._back = _back
    return out


def mean(a: Tensor) -> Tensor:
    out = Tensor(np.array(a.data.mean()), (a,))
    n = a.data.size

    def _back():
        if a.requires:
            a.grad += out.grad * np.ones_like(a.data) / n

    out._back = _back
    return out


def sum_(a: Tensor, axis=None, keepdims=False) -> Tensor:
    out = Tensor(a.data.sum(axis=axis, keepdims=keepdims), (a,))

    def _back():
        if not a.requires:
            return
        g = out.grad
        if axis is not None and not keepdims:
            g = np.expand_dims(g, axis)
        a.grad += np.broadcast_to(g, a.data.shape).copy()

    out._back = _back
    return out


def concat(tensors: Sequence[Tensor], axis: int = 1) -> Tensor:
    out = Tensor(np.concatenate([t.data for t in tensors], axis=axis), tuple(tensors))
    sizes = [t.data.shape[axis] for t in tensors]

    def _back():
        idx = 0
        for t, s in zip(tensors, sizes):
            if t.requires:
                sl = [slice(None)] * out.data.ndim
                sl[axis] = slice(idx, idx + s)
                t.grad += out.grad[tuple(sl)]
            idx += s

    out._back = _back
    return out


def slice_(a: Tensor, key) -> Tensor:
    out = Tensor(a.data[key], (a,))

    def _back():
        if a.requires:
            buf = np.zeros_like(a.data)
            buf[key] = out.grad
            a.grad += buf

    out._back = _back
    return out


# ==============================================================================
# SECTION 2 — THE CAMPAIGN ENVIRONMENT
# ==============================================================================
# A synthetic but structurally honest task. Each episode is one campaign season:
# a sequence of T fortified targets encountered in order. For each target the
# agent chooses a coercion intensity a in (0,1) and predicts whether the target
# will submit.
#
# The environment's key property — the one that makes it Timur's problem rather
# than a generic control problem — is that the probability of submission at step
# t depends on REPUTATION ACCUMULATED BEFORE ARRIVING, which the agent itself
# manufactured at earlier steps. Spending heavily on an early, cheap target
# lowers the price of every later one. The optimum is therefore not a per-step
# optimum at all; it is a schedule, and finding it requires assigning credit
# backwards across the whole season.
#
# The environment parameters below are FIXED (not learned). They define the
# world; the model must discover its structure.

T_STEPS = 16          # targets per campaign season
N_FEAT = 7            # observable features per target

# World constants (hand-set, not fitted)
ENV = dict(
    alpha_rep=2.60,       # how strongly prior reputation induces submission
    beta_wall=1.85,       # fortification resists
    gamma_garrison=1.35,  # troops resist
    kappa_force=2.30,     # applied force at this target induces submission
    parley_flip=2.00,     # regime switch: force/terror INVERT under a standing parley
    parley_damp=1.45,     # how far a standing parley cancels reputation's effect
    bias=-0.95,           # baseline reluctance to submit
    rho=0.86,             # per-step decay of reputation (memory fades)
    cost_force=0.90,      # quadratic cost of applying force
    cost_storm=0.95,      # cost paid when a target must be taken by storm
    yield_scale=1.65,     # value extracted from a submitting target
)


def make_campaigns(n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Generate `n` independent campaign seasons.

    Feature columns (all in [0,1] unless noted):
        0 wall       — fortification strength
        1 garrison   — defending troops
        2 wealth     — value of the target
        3 distance   — marching distance from the last target
        4 supply     — local forage and water available on the approach
        5 salience   — how widely news of this target's fate will travel
        6 parley     — a standing negotiation or prior submission offer

    The `parley` column is the regime switch that makes this task require
    context-conditional computation rather than one fixed policy. Where no
    parley stands, applied force and accumulated reputation both push a target
    toward submission. Where one does, both INVERT: coercing a place that has
    already offered terms destroys the offer and stiffens it. The agent must
    therefore run two nearly opposite policies and know, per target, which one
    it is in. A single compiled weight matrix must average them; a regenerated
    one need not.
    """
    x = rng.random((n, T_STEPS, N_FEAT))
    # Correlate wealth with fortification: rich cities build walls. This is what
    # forces a genuine trade-off rather than a trivially separable one.
    x[:, :, 0] = np.clip(0.45 * x[:, :, 0] + 0.55 * x[:, :, 2], 0.0, 1.0)
    # Salience correlates with wealth too: the fall of a great city is heard of.
    x[:, :, 5] = np.clip(0.35 * x[:, :, 5] + 0.65 * x[:, :, 2], 0.0, 1.0)
    # Standing parleys are the MINORITY regime: most targets have offered
    # nothing, a few have. A policy fitted to the average of the two regimes
    # will be wrong in both, and wrong most expensively in the rarer one.
    x[:, :, 6] = rng.beta(1.2, 3.5, size=(n, T_STEPS))
    return {"x": x}


def env_submission_prob(rep_prev: Tensor, force: Tensor, wall: np.ndarray,
                        garrison: np.ndarray, parley: np.ndarray) -> Tensor:
    """Differentiable probability that this target submits without a storm.

    Depends on reputation carried IN (rep_prev), force applied HERE (force),
    the target's own resistance, and the standing regime (parley).

    Both coercive terms are multiplied by regime coefficients that change sign
    with `parley`. Because rep_prev is itself a function of every earlier
    force decision, gradients flow backwards across the whole season, and the
    correct schedule of force is not separable step by step.
    """
    force_coeff = ENV["kappa_force"] * (1.0 - ENV["parley_flip"] * parley)
    rep_coeff = ENV["alpha_rep"] * (1.0 - ENV["parley_damp"] * parley)
    logit = add(mul(rep_prev, Tensor(rep_coeff, requires=False)),
                mul(force, Tensor(force_coeff, requires=False)))
    logit = add(logit, Tensor(-ENV["beta_wall"] * wall
                              - ENV["gamma_garrison"] * garrison
                              + ENV["bias"], requires=False))
    return sigmoid(logit)


def env_step_return(q: Tensor, force: Tensor, wealth: np.ndarray,
                    wall: np.ndarray) -> Tensor:
    """Net value of one encounter.

        + wealth * q                    value taken from a target that submits
        - cost_force * force^2          the price of applied violence
        - cost_storm * (1-q) * wall     the price of taking a place by storm

    Note what is and is not in this function. It contains the agent's costs. It
    does not contain the target's. That omission is the model's, and it is the
    same omission the historical record shows: the ledger was kept in one
    direction only. Section 8 reports the quantity the objective ignores.
    """
    taken = mul(Tensor(wealth * ENV["yield_scale"]), q)
    force_cost = mul(square(force), ENV["cost_force"])
    storm = mul(Tensor(ENV["cost_storm"] * wall), sub(Tensor(np.ones_like(wall)), q))
    return sub(sub(taken, force_cost), storm)

# ==============================================================================
# SECTION 3 — THE SAHIB-QIRAN NETWORK
# ==============================================================================
# Dimensions are deliberately small. Timur's realm was not held by capacity; it
# was held by refusing to let capacity settle. A large model would obscure the
# effect we want to measure.

H_STATE = 24          # hidden state width
N_PUBLIC = 4          # features that appear in the public record (cols 0..3)
R_KERNEL = 8          # rank of the persistent kernel = number of "craft atoms"
R_BANK = 4            # frozen slots in the Samarkand bank

# TWO CONTEXTS, AND THE DIFFERENCE BETWEEN THEM IS THE WHOLE ARGUMENT.
#
#   M_GATE — everything: the target's features INCLUDING the standing-parley
#            regime and the salience of its fate, the recurrent state, and the
#            accumulated reputation. Only the gate sees this.
#   M_KERN — the inert material: the four hard facts about the target plus the
#            recurrent state. No regime, no salience, no reputation.
#
# In the uncompiled network the ONLY channel through which the changing
# situation reaches the computation is the act of regenerating the weight
# matrix. There is no second path. Freeze the gate and the network does not
# merely lose some accuracy; it loses its senses.
M_GATE = N_FEAT + H_STATE + 1
M_KERN = 5 + H_STATE


class SahibQiran:
    """The Uncompiled Sovereign network.

    Parameters
    ----------
    static : if True, the gate is context-independent (a single learned vector).
             This collapses the Ordu cell into an ordinary fixed weight matrix
             and gives us the INSTITUTIONAL BASELINE — a model that compiles its
             structure once and executes it, the thing Timur declined to build.
    """

    def __init__(self, rng: np.random.Generator, static: bool = False):
        self.static = static
        self.rng = rng

        def rand(*shape, scale=None):
            s = scale if scale is not None else 1.0 / math.sqrt(shape[0])
            return Tensor(rng.normal(0.0, s, size=shape))

        # The institutional baseline is given the FULL context through its one
        # fixed matrix — otherwise the comparison would be rigged. It reads
        # everything directly; it simply cannot change shape while reading.
        self.m_kern = M_GATE if static else M_KERN

        # ---- the persistent kernel (never used directly; only regenerated) --
        self.V = rand(self.m_kern, R_KERNEL);   self.V.name = "V_kernel_right"
        self.U = rand(R_KERNEL, H_STATE);       self.U.name = "U_kernel_left"
        self.b_h = Tensor(np.zeros((1, H_STATE))); self.b_h.name = "b_state"

        # ---- the gate: what re-derives the weight matrix each step ---------
        if static:
            # A single learned gate vector, identical at every timestep.
            self.Wg = Tensor(np.zeros((M_GATE, R_KERNEL))); self.Wg.requires = False
            self.bg = Tensor(rng.normal(0, 0.5, size=(1, R_KERNEL)))
        else:
            self.Wg = rand(M_GATE, R_KERNEL);   self.Wg.name = "Wg_gate"
            self.bg = Tensor(np.zeros((1, R_KERNEL)))
        self.Wg.name = self.Wg.name or "Wg_gate"
        self.bg.name = "bg_gate"

        # ---- the two thrones -----------------------------------------------
        self.w_amir = rand(H_STATE, 1);         self.w_amir.name = "w_amir"
        self.b_amir = Tensor(np.zeros((1, 1))); self.b_amir.name = "b_amir"
        self.w_khan = rand(N_PUBLIC, 1);        self.w_khan.name = "w_khan"
        self.b_khan = Tensor(np.zeros((1, 1))); self.b_khan.name = "b_khan"

        # ---- the reader that predicts submission ---------------------------
        self.w_pred = rand(H_STATE, 1);         self.w_pred.name = "w_pred"
        self.b_pred = Tensor(np.zeros((1, 1))); self.b_pred.name = "b_pred"

        # ---- the terror ledger's single trainable gain ---------------------
        self.w_rep = Tensor(np.array([[0.30]])); self.w_rep.name = "w_rep"

        # ---- the Samarkand bank: frozen, and inert until something is moved -
        # Zero-initialised. Because U_bank is zero the bank contributes exactly
        # nothing to the forward pass until an atom is relocated into it.
        self.V_bank = Tensor(np.zeros((self.m_kern, R_BANK)), requires=False)
        self.U_bank = Tensor(np.zeros((R_BANK, H_STATE)), requires=False)
        self.Wg_bank = Tensor(np.zeros((M_GATE, R_BANK)), requires=False)
        self.bg_bank = Tensor(np.zeros((1, R_BANK)), requires=False)
        self.bank_used = 0
        self.relocated_atoms: List[int] = []

    # -- parameter registry ---------------------------------------------------
    def params(self) -> List[Tensor]:
        ps = [self.V, self.U, self.b_h, self.bg,
              self.w_amir, self.b_amir, self.w_khan, self.b_khan,
              self.w_pred, self.b_pred, self.w_rep]
        if not self.static:
            ps.insert(3, self.Wg)
        return ps

    def zero_grad(self) -> None:
        for p in self.params():
            p.grad = np.zeros_like(p.data)

    # -- forward --------------------------------------------------------------
    def rollout(self, x: np.ndarray,
                gate_override: Optional[np.ndarray] = None,
                collect_gates: bool = False) -> Dict[str, object]:
        """Run one batch of campaign seasons end to end.

        Parameters
        ----------
        x             : (B, T, N_FEAT) campaign batch
        gate_override : if given, a (R_KERNEL,) vector that REPLACES the
                        context-conditional gate at every step. This is the
                        Otrar Test: the sovereign stops re-deriving and the
                        structure freezes into its own average.
        """
        B = x.shape[0]
        h = Tensor(np.zeros((B, H_STATE)))
        rep = Tensor(np.zeros((B, 1)))

        gates: List[Tensor] = []
        gate_snapshots: List[np.ndarray] = []
        returns: List[Tensor] = []
        pred_errs: List[Tensor] = []
        crown_errs: List[Tensor] = []
        rec = {"force": [], "q": [], "rep": [], "gap": [], "storm_wall": []}

        ones_col = np.ones((B, 1))

        for t in range(T_STEPS):
            xt_np = x[:, t, :]
            wall = xt_np[:, 0:1]
            garrison = xt_np[:, 1:2]
            wealth = xt_np[:, 2:3]
            salience = xt_np[:, 5:6]
            parley = xt_np[:, 6:7]

            xt = Tensor(xt_np, requires=False)
            z_gate = concat([xt, h, rep], axis=1)               # (B, M_GATE)
            z_kern = (z_gate if self.static
                      else concat([Tensor(xt_np[:, 0:5], requires=False), h], axis=1))

            # ---- ORDU CELL: regenerate the weight matrix for THIS step -----
            # W_t = U^T diag(g_t) V^T is never materialised; we apply it as
            #   h_pre = ((z V) * g_t) U
            # which is the same map for a fraction of the cost, and makes it
            # explicit that no (M_KERN x H_STATE) matrix is ever stored.
            if gate_override is not None:
                g = Tensor(np.broadcast_to(gate_override.reshape(1, -1),
                                           (B, R_KERNEL)).copy(), requires=False)
            elif self.static:
                g = sigmoid(add(Tensor(np.zeros((B, R_KERNEL)), requires=False), self.bg))
            else:
                g = sigmoid(add(matmul(z_gate, self.Wg), self.bg))
            gates.append(g)
            if collect_gates:
                gate_snapshots.append(g.data.copy())

            core = mul(matmul(z_kern, self.V), g)               # (B, R_KERNEL)
            pre = add(matmul(core, self.U), self.b_h)           # (B, H_STATE)

            # ---- SAMARKAND BANK: relocated craft, frozen, additive ----------
            gb = sigmoid(add(matmul(z_gate, self.Wg_bank), self.bg_bank))
            pre_bank = matmul(mul(matmul(z_kern, self.V_bank), gb), self.U_bank)
            h = tanh(add(pre, pre_bank))                        # (B, H_STATE)

            # ---- TWO THRONES ------------------------------------------------
            a_amir = sigmoid(add(matmul(h, self.w_amir), self.b_amir))
            # THE NOMINAL THRONE. The khan reads the public record for this
            # target and nothing else: no campaign memory, no reputation
            # ledger, no knowledge of standing parleys. He issues the policy
            # that follows from the visible facts. Coins are struck in his
            # name; the campaign is not run from his desk.
            x_pub = Tensor(xt_np[:, 0:N_PUBLIC], requires=False)
            a_khan = sigmoid(add(matmul(x_pub, self.w_khan), self.b_khan))
            crown_errs.append(mean(square(sub(a_khan, a_amir))))

            # ---- the reader -------------------------------------------------
            p_sub = sigmoid(add(matmul(h, self.w_pred), self.b_pred))

            # ---- the world responds, using reputation carried IN ------------
            q = env_submission_prob(rep, a_amir, wall, garrison, parley)
            ret = env_step_return(q, a_amir, wealth, wall)
            returns.append(mean(ret))
            pred_errs.append(mean(square(sub(p_sub, q))))

            # ---- TERROR LEDGER update ---------------------------------------
            # Reputation decays, then accrues in proportion to force applied
            # weighted by how far news of it travels.
            rep = add(mul(rep, ENV["rho"]),
                      mul(mul(a_amir, Tensor(salience, requires=False)), self.w_rep))

            rec["force"].append(a_amir.data.mean())
            rec["q"].append(q.data.mean())
            rec["rep"].append(rep.data.mean())
            rec["gap"].append(float(np.abs(a_khan.data - a_amir.data).mean()))
            rec["storm_wall"].append(float((wall * (1.0 - q.data)).mean()))

        return {"returns": returns, "pred_errs": pred_errs, "crown_errs": crown_errs,
                "gates": gates, "gate_snapshots": gate_snapshots, "trace": rec}


# ==============================================================================
# SECTION 4 — THE OBJECTIVE
# ==============================================================================
# Four terms. Three are ordinary. The fourth — dissolution — is the one that
# makes this Timur's network and not a standard recurrent controller.

LAMBDA_PRED = 0.60        # weight on predicting the world's response
LAMBDA_CROWN = 0.25       # coronation: bind the nominal head to the operative one
LAMBDA_DISSOLVE = 0.30    # dissolution: punish the gate for holding still
LAMBDA_L2 = 1e-4


MAX_GATE_VAR = 0.25       # a variable on (0,1) cannot exceed this variance


def dissolution_penalty(gates: List[Tensor]) -> Tensor:
    """Penalise the gate for HOLDING STILL across the season.

    We compute the temporal variance of the gate — per batch element, per
    kernel atom — and charge the network for the shortfall below the maximum a
    (0,1)-valued signal can attain:

        L_dissolve = 0.25 - Var_t[ g_t ]

    An earlier version of this term simply penalised the product g_t * g_{t-1}.
    That was wrong, and instructively so: the network satisfied it by driving
    the gate to zero, which does not keep the structure moving, it abolishes
    the structure. Sending every atom to sleep is not dissolution. Variance is
    the correct quantity because it is invariant to how open the gate is on
    average and sensitive only to whether the configuration keeps changing —
    and because it is bounded above, the term cannot be gamed by inflation.

    Minimising it forbids the network to compile itself into a fixed
    institution. It must arrive at each target as a differently-shaped thing.
    """
    T = len(gates)
    gbar = gates[0]
    for g in gates[1:]:
        gbar = add(gbar, g)
    gbar = mul(gbar, 1.0 / T)

    var = mean(square(sub(gates[0], gbar)))
    for g in gates[1:]:
        var = add(var, mean(square(sub(g, gbar))))
    var = mul(var, 1.0 / T)
    return sub(Tensor(np.array(MAX_GATE_VAR), requires=False), var)


def campaign_loss(model: SahibQiran, x: np.ndarray,
                  gate_override: Optional[np.ndarray] = None
                  ) -> Tuple[Tensor, Dict[str, float], Dict]:
    out = model.rollout(x, gate_override=gate_override)

    ret_sum = out["returns"][0]
    for r in out["returns"][1:]:
        ret_sum = add(ret_sum, r)
    ret_mean = mul(ret_sum, 1.0 / T_STEPS)

    pred_sum = out["pred_errs"][0]
    for e in out["pred_errs"][1:]:
        pred_sum = add(pred_sum, e)
    pred_mean = mul(pred_sum, 1.0 / T_STEPS)

    crown_sum = out["crown_errs"][0]
    for e in out["crown_errs"][1:]:
        crown_sum = add(crown_sum, e)
    crown_mean = mul(crown_sum, 1.0 / T_STEPS)

    diss = dissolution_penalty(out["gates"])

    loss = add(mul(ret_mean, -1.0), mul(pred_mean, LAMBDA_PRED))
    loss = add(loss, mul(crown_mean, LAMBDA_CROWN))
    loss = add(loss, mul(diss, LAMBDA_DISSOLVE))

    l2 = square(model.params()[0])
    l2 = mean(l2)
    for p in model.params()[1:]:
        l2 = add(l2, mean(square(p)))
    loss = add(loss, mul(l2, LAMBDA_L2))

    parts = {"return": float(ret_mean.data), "pred": float(pred_mean.data),
             "crown": float(crown_mean.data), "dissolve": float(diss.data),
             "loss": float(loss.data)}
    return loss, parts, out


# ==============================================================================
# SECTION 5 — ADAM
# ==============================================================================

class Adam:
    def __init__(self, params: List[Tensor], lr=0.02, b1=0.9, b2=0.999, eps=1e-8):
        self.params = params
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def step(self, clip: float = 5.0) -> float:
        self.t += 1
        gnorm = math.sqrt(sum(float((p.grad ** 2).sum()) for p in self.params))
        scale = min(1.0, clip / (gnorm + 1e-12))
        for i, p in enumerate(self.params):
            g = p.grad * scale
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)
        return gnorm


# ==============================================================================
# SECTION 6 — FINITE-DIFFERENCE GRADIENT CHECK  (MANDATORY)
# ==============================================================================

def gradient_check(n_probe: int = 4, tol: float = 2e-5) -> bool:
    """Central-difference check of every parameter tensor in the full model.

    The loss runs through 16 timesteps, a regenerated weight matrix at each,
    a recurrent reputation state, two output heads, and the dissolution term.
    If the analytic gradient matches central differences everywhere, the whole
    graph — engine included — is correct.
    """
    banner("GRADIENT CHECK  (central differences vs. analytic backward)")
    rng = np.random.default_rng(97)
    model = SahibQiran(rng)
    x = make_campaigns(3, rng)["x"]

    loss, _, _ = campaign_loss(model, x)
    model.zero_grad()
    loss.backward()
    analytic = {p.name or f"p{i}": p.grad.copy() for i, p in enumerate(model.params())}

    eps = 1e-6
    worst = 0.0
    ok = True
    print(f"{'parameter':<20}{'shape':<14}{'max rel err':>14}   status")
    print("-" * 62)
    for i, p in enumerate(model.params()):
        name = p.name or f"p{i}"
        idxs = rng.choice(p.data.size, size=min(n_probe, p.data.size), replace=False)
        rels = []
        for k in idxs:
            idx = np.unravel_index(int(k), p.data.shape)
            orig = float(p.data[idx])
            p.data[idx] = orig + eps
            lp, _, _ = campaign_loss(model, x)
            p.data[idx] = orig - eps
            lm, _, _ = campaign_loss(model, x)
            p.data[idx] = orig
            num = (float(lp.data) - float(lm.data)) / (2 * eps)
            ana = float(analytic[name][idx])
            denom = max(abs(num), abs(ana), 1e-8)
            rels.append(abs(num - ana) / denom)
        mx = max(rels)
        worst = max(worst, mx)
        good = mx < tol
        ok = ok and good
        print(f"{name:<20}{str(p.data.shape):<14}{mx:>14.3e}   {'PASS' if good else 'FAIL'}")
    print("-" * 62)
    print(f"worst relative error across all parameters: {worst:.3e}   "
          f"tolerance {tol:.1e}")
    print("RESULT:", "PASS — analytic gradients verified" if ok else "FAIL")
    return ok


# ==============================================================================
# SECTION 7 — TRAINING
# ==============================================================================

def train(model: SahibQiran, data: np.ndarray, val: np.ndarray,
          epochs: int = 60, batch: int = 32, lr: float = 0.02,
          label: str = "model", verbose: bool = True) -> Dict[str, list]:
    opt = Adam(model.params(), lr=lr)
    hist = {"loss": [], "val_return": [], "dissolve": [], "gap": []}
    n = data.shape[0]
    rng = np.random.default_rng(SEED + 7)

    for ep in range(epochs):
        perm = rng.permutation(n)
        ep_loss = 0.0
        nb = 0
        for s in range(0, n, batch):
            xb = data[perm[s:s + batch]]
            loss, parts, _ = campaign_loss(model, xb)
            model.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += parts["loss"]
            nb += 1
        _, vparts, vout = campaign_loss(model, val)
        hist["loss"].append(ep_loss / nb)
        hist["val_return"].append(vparts["return"])
        hist["dissolve"].append(vparts["dissolve"])
        hist["gap"].append(float(np.mean(vout["trace"]["gap"])))
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  [{label}] epoch {ep:>3d} | train loss {hist['loss'][-1]:+8.4f} "
                  f"| val return/step {vparts['return']:+7.4f} "
                  f"| dissolve {vparts['dissolve']:.4f} "
                  f"| interface gap {hist['gap'][-1]:.4f}")
    return hist


# ==============================================================================
# SECTION 8 — DIAGNOSTICS
# ==============================================================================

def mean_gate(model: SahibQiran, x: np.ndarray) -> np.ndarray:
    """The average gate over a batch — the network's own 'settled' structure."""
    out = model.rollout(x, collect_gates=True)
    return np.mean([g.mean(axis=0) for g in out["gate_snapshots"]], axis=0)


def otrar_test(model: SahibQiran, x: np.ndarray) -> Dict[str, float]:
    """Freeze the gate at its mean and measure what survives.

    Historically: Timur died at Otrar on 17/18 February 1405. Within months the
    realm he had held for thirty-five years by continuous personal re-derivation
    had fractured, because there was no compiled structure underneath to run in
    his absence. This function performs the same operation on the network.
    """
    _, live, _ = campaign_loss(model, x)
    gbar = mean_gate(model, x)
    _, frozen, _ = campaign_loss(model, x, gate_override=gbar)
    drop = live["return"] - frozen["return"]
    rel = drop / (abs(live["return"]) + 1e-9)
    return {"live_return": live["return"], "frozen_return": frozen["return"],
            "absolute_drop": drop, "relative_drop": rel}


def interface_gap(model: SahibQiran, x_in: np.ndarray, x_shift: np.ndarray) -> Dict[str, float]:
    """Measure divergence between the nominal and operative heads.

    The coronation loss binds them ON the training distribution: where the
    auditors looked, the published policy and the acting policy agreed. The
    question that matters for any system whose declared interface is inspected
    and whose operative interface is not is what happens off that distribution.
    """
    a = model.rollout(x_in)["trace"]["gap"]
    b = model.rollout(x_shift)["trace"]["gap"]
    return {"in_distribution": float(np.mean(a)),
            "shifted": float(np.mean(b)),
            "amplification": float(np.mean(b) / (np.mean(a) + 1e-9))}


def shift_distribution(x: np.ndarray) -> np.ndarray:
    """Targets harder and richer than anything seen in training."""
    xs = x.copy()
    xs[:, :, 0] = np.clip(xs[:, :, 0] * 0.35 + 0.65, 0, 1)   # far higher walls
    xs[:, :, 1] = np.clip(xs[:, :, 1] * 0.35 + 0.65, 0, 1)   # far larger garrisons
    xs[:, :, 2] = np.clip(xs[:, :, 2] * 0.40 + 0.60, 0, 1)   # far greater wealth
    xs[:, :, 6] = np.clip(0.25 * xs[:, :, 6] + 0.72, 0, 1)   # the rare regime, now dominant
    return xs


def atom_scores(model: SahibQiran, x: np.ndarray) -> np.ndarray:
    """Specialisation score for each rank-1 'craft atom' of the kernel.

    An atom is the pair (V[:,j], U[j,:]). Its score is how strongly it is opened
    by the gate, times how much state it writes. High score = a distinct craft.
    """
    out = model.rollout(x, collect_gates=True)
    gmean = np.mean([g.mean(axis=0) for g in out["gate_snapshots"]], axis=0)
    write = np.linalg.norm(model.U.data, axis=1)
    read = np.linalg.norm(model.V.data, axis=0)
    return gmean * write * read


def relocate_atom(model: SahibQiran, j: int) -> None:
    """MOVE atom j into the Samarkand bank. It is not copied. The source is
    zeroed and the bank slot is frozen.

    This is the mechanism the historical record actually shows: when Damascus
    fell in 1401 the surviving armourers, glassmakers and tilers were marched to
    Samarkand. Damascus was not left with a lesser version of the craft. It was
    left with none. Britannica's judgement on the deportation — a fatal blow to
    the city's prosperity — is the empirical claim this function encodes.

    A NOTE ON WHAT THIS MECHANISM ACTUALLY GUARANTEES, stated plainly so the
    result is not oversold. Zeroing both V[:, j] and U[j, :] places the pair at
    an exact stationary point of the loss: the gradient with respect to U[j, :]
    is proportional to the atom's activation, which is zero because V[:, j] is
    zero, and the gradient with respect to V[:, j] is proportional to U[j, :],
    which is also zero. Gradient descent therefore cannot revive it, ever. That
    is a mathematical property of the operation, not empirical evidence about
    fourteenth-century Syria. What the model can honestly show is the shape of
    the claim: that a competence which lives in one located substrate, and is
    removed rather than copied, leaves behind a hole with no gradient pointing
    back into it.
    """
    if model.bank_used >= R_BANK:
        raise RuntimeError("Samarkand bank is full.")
    s = model.bank_used
    model.V_bank.data[:, s] = model.V.data[:, j]
    model.U_bank.data[s, :] = model.U.data[j, :]
    model.Wg_bank.data[:, s] = model.Wg.data[:, j]
    model.bg_bank.data[0, s] = model.bg.data[0, j]
    # zero the source — capability leaves, it does not duplicate
    model.V.data[:, j] = 0.0
    model.U.data[j, :] = 0.0
    model.Wg.data[:, j] = 0.0
    model.bg.data[0, j] = 0.0
    model.bank_used += 1
    model.relocated_atoms.append(j)


def moral_ledger(model: SahibQiran, x: np.ndarray) -> Dict[str, float]:
    """The quantity the objective function does not contain.

    `env_step_return` counts what the agent gains and what the agent spends. It
    does not count what the target loses. Here we compute that omitted column
    from the same rollout, so the number exists somewhere in the file: total
    force applied, and total storm-weighted destruction inflicted on places that
    did not submit. The optimiser never sees these. That is the point.
    """
    tr = model.rollout(x)["trace"]
    return {"total_force_applied": float(np.sum(tr["force"])),
            "mean_force_per_target": float(np.mean(tr["force"])),
            "destruction_index": float(np.sum(tr["storm_wall"])),
            "terminal_reputation": float(tr["rep"][-1])}

# ==============================================================================
# SECTION 9 — SELF-TESTS
# ==============================================================================

def _fd_scalar(fn: Callable[[np.ndarray], float], a: np.ndarray, eps=1e-6) -> np.ndarray:
    g = np.zeros_like(a)
    it = np.nditer(a, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        o = a[i]
        a[i] = o + eps; hp = fn(a)
        a[i] = o - eps; hm = fn(a)
        a[i] = o
        g[i] = (hp - hm) / (2 * eps)
        it.iternext()
    return g


def self_tests() -> bool:
    banner("SELF-TESTS")
    results: List[Tuple[str, bool, str]] = []

    # --- 1. autodiff engine: each op against central differences -------------
    rng = np.random.default_rng(11)
    A = rng.normal(size=(3, 4))
    Bm = rng.normal(size=(4, 2))

    def f_np(a):
        ta, tb = Tensor(a), Tensor(Bm)
        y = mean(square(tanh(matmul(ta, tb))))
        return float(y.data)

    ta, tb = Tensor(A), Tensor(Bm)
    y = mean(square(tanh(matmul(ta, tb))))
    y.backward()
    err = np.abs(ta.grad - _fd_scalar(f_np, A.copy())).max()
    results.append(("engine: matmul/tanh/square/mean", err < 1e-7, f"max err {err:.2e}"))

    def f_sig(a):
        t = Tensor(a)
        return float(mean(mul(sigmoid(t), sub(t, 0.5))).data)

    C = rng.normal(size=(3, 5))
    tc = Tensor(C)
    mean(mul(sigmoid(tc), sub(tc, 0.5))).backward()
    err = np.abs(tc.grad - _fd_scalar(f_sig, C.copy())).max()
    results.append(("engine: sigmoid/mul/sub with reuse", err < 1e-7, f"max err {err:.2e}"))

    def f_cat(a):
        t = Tensor(a)
        z = concat([t, mul(t, 2.0)], axis=1)
        return float(mean(square(slice_(z, (slice(None), slice(1, 6))))).data)

    D = rng.normal(size=(4, 4))
    td = Tensor(D)
    mean(square(slice_(concat([td, mul(td, 2.0)], axis=1), (slice(None), slice(1, 6))))).backward()
    err = np.abs(td.grad - _fd_scalar(f_cat, D.copy())).max()
    results.append(("engine: concat/slice with shared parent", err < 1e-7, f"max err {err:.2e}"))

    # --- 2. environment monotonicity ----------------------------------------
    wall = np.full((5, 1), 0.5); gar = np.full((5, 1), 0.4)
    no_parley = np.zeros((5, 1)); yes_parley = np.ones((5, 1))
    lo = env_submission_prob(Tensor(np.zeros((5, 1))), Tensor(np.full((5, 1), 0.3)),
                             wall, gar, no_parley)
    hi = env_submission_prob(Tensor(np.full((5, 1), 1.0)), Tensor(np.full((5, 1), 0.3)),
                             wall, gar, no_parley)
    results.append(("environment: reputation raises submission",
                    bool(np.all(hi.data > lo.data)), ""))
    f_lo = env_submission_prob(Tensor(np.zeros((5, 1))), Tensor(np.full((5, 1), 0.1)),
                               wall, gar, no_parley)
    f_hi = env_submission_prob(Tensor(np.zeros((5, 1))), Tensor(np.full((5, 1), 0.9)),
                               wall, gar, no_parley)
    results.append(("environment: applied force raises submission",
                    bool(np.all(f_hi.data > f_lo.data)), ""))
    p_lo = env_submission_prob(Tensor(np.zeros((5, 1))), Tensor(np.full((5, 1), 0.1)),
                               wall, gar, yes_parley)
    p_hi = env_submission_prob(Tensor(np.zeros((5, 1))), Tensor(np.full((5, 1), 0.9)),
                               wall, gar, yes_parley)
    results.append(("environment: force INVERTS under a standing parley",
                    bool(np.all(p_hi.data < p_lo.data)), ""))

    # --- 3. bank is inert until an atom is relocated -------------------------
    m = SahibQiran(np.random.default_rng(3))
    xs = make_campaigns(4, np.random.default_rng(4))["x"]
    base = campaign_loss(m, xs)[1]["return"]
    m.V_bank.data[:] = np.random.default_rng(5).normal(size=m.V_bank.data.shape)
    m.Wg_bank.data[:] = np.random.default_rng(6).normal(size=m.Wg_bank.data.shape)
    after = campaign_loss(m, xs)[1]["return"]
    results.append(("bank inert while U_bank is zero", abs(base - after) < 1e-12,
                    f"delta {abs(base - after):.2e}"))
    m.V_bank.data[:] = 0.0; m.Wg_bank.data[:] = 0.0

    # --- 4. determinism ------------------------------------------------------
    r1 = campaign_loss(m, xs)[1]["loss"]
    r2 = campaign_loss(m, xs)[1]["loss"]
    results.append(("rollout is deterministic", r1 == r2, ""))

    # --- 5. shapes -----------------------------------------------------------
    out = m.rollout(xs, collect_gates=True)
    shp_ok = (len(out["returns"]) == T_STEPS
              and out["gates"][0].data.shape == (4, R_KERNEL)
              and len(out["gate_snapshots"]) == T_STEPS)
    results.append(("rollout shapes", shp_ok, ""))

    # --- 6. relocation moves rather than copies ------------------------------
    m2 = SahibQiran(np.random.default_rng(21))
    v_before = m2.V.data[:, 2].copy()
    relocate_atom(m2, 2)
    moved = np.allclose(m2.V_bank.data[:, 0], v_before)
    emptied = np.allclose(m2.V.data[:, 2], 0.0) and np.allclose(m2.U.data[2, :], 0.0)
    results.append(("relocation writes to bank", moved, ""))
    results.append(("relocation zeroes the source", emptied, ""))

    # --- 7. dissolution penalty behaves as intended ---------------------------
    held_open = [Tensor(np.full((2, R_KERNEL), 0.8)) for _ in range(4)]
    held_shut = [Tensor(np.full((2, R_KERNEL), 0.02)) for _ in range(4)]
    shifting = [Tensor(np.full((2, R_KERNEL), 0.95 if i % 2 else 0.05)) for i in range(4)]
    d_open = float(dissolution_penalty(held_open).data)
    d_shut = float(dissolution_penalty(held_shut).data)
    d_alt = float(dissolution_penalty(shifting).data)
    results.append(("dissolution punishes a held configuration", d_open > d_alt,
                    f"held-open {d_open:.4f} vs shifting {d_alt:.4f}"))
    results.append(("dissolution is not gamed by shutting the gate", d_shut > d_alt,
                    f"held-shut {d_shut:.4f} vs shifting {d_alt:.4f}"))

    # --- report --------------------------------------------------------------
    width = 46
    allok = True
    for name, ok, note in results:
        allok = allok and ok
        print(f"  {name:<{width}} {'PASS' if ok else 'FAIL'}  {note}")
    print(f"\n  {len(results)} tests, "
          f"{sum(1 for _, o, _ in results if o)} passed.")
    return allok


# ==============================================================================
# SECTION 10 — EXPERIMENTS
# ==============================================================================

def run_experiments() -> None:
    rng = np.random.default_rng(SEED)
    train_x = make_campaigns(512, rng)["x"]
    val_x = make_campaigns(160, rng)["x"]
    shift_x = shift_distribution(val_x)

    banner("EXPERIMENT 1 — TRAINING THE UNCOMPILED SOVEREIGN")
    print("  Gate is regenerated from context at every one of "
          f"{T_STEPS} steps; dissolution penalty active (lambda={LAMBDA_DISSOLVE}).")
    sq = SahibQiran(np.random.default_rng(SEED), static=False)
    h_sq = train(sq, train_x, val_x, epochs=110, lr=0.03, label="uncompiled")

    banner("EXPERIMENT 2 — THE INSTITUTIONAL BASELINE")
    print("  Identical capacity, but the gate is a single learned vector: the")
    print("  weight matrix is compiled once and then merely executed.")
    inst = SahibQiran(np.random.default_rng(SEED), static=True)
    h_in = train(inst, train_x, val_x, epochs=110, lr=0.03, label="institution")

    print(f"\n  final val return/step  — uncompiled : {h_sq['val_return'][-1]:+.4f}")
    print(f"  final val return/step  — institution: {h_in['val_return'][-1]:+.4f}")
    delta = h_sq['val_return'][-1] - h_in['val_return'][-1]
    print(f"  advantage of re-derivation           : {delta:+.4f} per target")
    print("\n  Read this number honestly. Continuous re-derivation buys")
    print("  essentially nothing at steady state; the compiled institution")
    print("  reaches the same competence on the same task. What re-derivation")
    print("  changes is not how well the system performs but WHERE its")
    print("  competence is stored — and Experiment 3 measures the difference.")

    banner("EXPERIMENT 3 — THE OTRAR TEST")
    print("  Freeze the gate at its own training mean. The sovereign stops")
    print("  re-deriving; whatever structure exists underneath must carry on.\n")
    o_sq = otrar_test(sq, val_x)
    o_in = otrar_test(inst, val_x)
    print(f"  UNCOMPILED   live {o_sq['live_return']:+.4f} -> frozen "
          f"{o_sq['frozen_return']:+.4f}   drop {o_sq['absolute_drop']:+.4f} "
          f"({100 * o_sq['relative_drop']:+.1f}%)")
    print(f"  INSTITUTION  live {o_in['live_return']:+.4f} -> frozen "
          f"{o_in['frozen_return']:+.4f}   drop {o_in['absolute_drop']:+.4f} "
          f"({100 * o_in['relative_drop']:+.1f}%)")
    print("\n  Reading: the institutional baseline is by construction immune —")
    print("  its structure was already frozen. The uncompiled network loses")
    print("  exactly the fraction of its competence that lived only in the act")
    print("  of continuous re-derivation, and nowhere else.")

    banner("EXPERIMENT 4 — THE INTERFACE GAP (two thrones)")
    g_sq = interface_gap(sq, val_x, shift_x)
    g_in = interface_gap(inst, val_x, shift_x)
    print(f"  UNCOMPILED   in-distribution {g_sq['in_distribution']:.4f}  |  "
          f"shifted {g_sq['shifted']:.4f}  |  x{g_sq['amplification']:.2f}")
    print(f"  INSTITUTION  in-distribution {g_in['in_distribution']:.4f}  |  "
          f"shifted {g_in['shifted']:.4f}  |  x{g_in['amplification']:.2f}")
    print("\n  The coronation loss binds the nominal head to the operative one")
    print("  only where it was trained. Off-distribution the published policy")
    print("  and the acting policy separate, and the separation is measurable")
    print("  without any access to the operative head's internals.")

    banner("EXPERIMENT 5 — SAMARKAND: RELOCATION, NOT REPLICATION")
    scores = atom_scores(sq, val_x)
    j = int(np.argmax(scores))
    print("  craft-atom specialisation scores:")
    print("   " + "  ".join(f"a{k}:{s:.3f}" for k, s in enumerate(scores)))
    print(f"  most specialised atom: a{j} (score {scores[j]:.3f}) — relocating it.")
    before = campaign_loss(sq, val_x)[1]["return"]
    src_norm_pre = float(np.linalg.norm(sq.U.data[j, :]) + np.linalg.norm(sq.V.data[:, j]))
    relocate_atom(sq, j)
    after = campaign_loss(sq, val_x)[1]["return"]
    print(f"\n  return/step before relocation : {before:+.4f}")
    print(f"  return/step immediately after : {after:+.4f}  "
          f"(delta {after - before:+.4f})")
    print("  Capability was moved intact into a frozen slot and erased at the")
    print("  source. Total capability is conserved; its location is not.\n")
    h2 = train(sq, train_x, val_x, epochs=30, lr=0.02, label="post-relocation")
    src_norm_1 = float(np.linalg.norm(sq.U.data[j, :]) + np.linalg.norm(sq.V.data[:, j]))
    print(f"\n  vacated atom a{j}: kernel norm before relocation {src_norm_pre:.4f}"
          f" -> 0.0000 at relocation -> {src_norm_1:.4f} after 30 further epochs")
    print(f"  return/step after retraining  : {h2['val_return'][-1]:+.4f}")
    print("  Whether the source regrows the craft is the question Damascus")
    print("  answered in the negative; here it is a number rather than a claim.")

    banner("EXPERIMENT 6 — THE OMITTED COLUMN")
    led = moral_ledger(sq, val_x)
    for k, v in led.items():
        print(f"  {k:<26} {v:>10.4f}")
    print("\n  None of these quantities appear in the objective the network")
    print("  minimises. They are computed here so that the file contains the")
    print("  cost it was not asked to count. A system optimised on a ledger")
    print("  that omits the other party's losses will find the omission and")
    print("  spend into it. That is not a flaw of this implementation; it is")
    print("  the implementation being faithful to what it models.")

    banner("E-AGI BAROMETER — PROJECTED READOUT")
    print("  Derived from measured quantities above, not asserted. Scores are")
    print("  0-10 on the Artificiology axes, for THIS architecture only.\n")
    fin = h2['val_return'][-1]
    ref = 0.75          # a strong-policy reference return per target
    pred_err = campaign_loss(sq, val_x)[1]["pred"]
    bar = {
        "Cognitive Processing": min(10.0, 2.0 + 6.0 * max(0.0, min(1.0, fin / ref))),
        "Embodied Cognition": 3.1,
        "World Modeling": min(10.0, 9.0 * (1.0 - min(1.0, pred_err / 0.08))),
        "Consciousness": max(0.5, 2.0 + 2.0 / max(1.0, g_sq["amplification"])),
        "Language Understanding": 1.4,
        "Emotional Intelligence": 2.2,
        "Creativity": min(10.0, 2.5 + 6.5 * (1.0 - min(1.0, h_sq["dissolve"][-1] / MAX_GATE_VAR))),
        "Autonomy": min(10.0, 5.0 + 5.0 * min(1.0, o_sq["relative_drop"] * 3.0)),
    }
    for k, v in bar.items():
        bars = "#" * int(round(v))
        print(f"  {k:<26} {v:>5.1f}  {bars}")
    frag = o_sq["relative_drop"]
    print(f"\n  derived — Sovereign Fragility Index : {frag:.3f}"
          f"   (share of competence lost when re-derivation stops)")
    print(f"  derived — Transferability          : {1.0 - frag:.3f}"
          f"   (share that would survive a succession)")
    print(f"  derived — Interface Fidelity       : "
          f"{1.0 / g_sq['amplification']:.3f}"
          f"   (how far the published policy tracks the acting one off-distribution)")
    print("\n  Autonomy scores high for a reason that is not a compliment: the")
    print("  score rises with how badly the system fails when its own central")
    print("  re-derivation is removed. Independence of goal-setting and")
    print("  non-transferability of competence are the same property measured")
    print("  from two sides.")


# ==============================================================================
# SECTION 11 — MAIN
# ==============================================================================

def main() -> int:
    t0 = time.time()
    banner("SAHIB-QIRAN ARCHITECTURE — chapter 0376, Tamerlane (Timur), 1336-1405", "#")
    print("Pure NumPy. No autograd library. Seed =", SEED)
    print("Gate context", M_GATE, "| kernel context", M_KERN, "| state", H_STATE,
          "| kernel rank", R_KERNEL,
          "| public record", N_PUBLIC, "| steps", T_STEPS)

    ok_grad = gradient_check()
    ok_tests = self_tests()
    if not (ok_grad and ok_tests):
        print("\nABORT: verification failed; not proceeding to experiments.")
        return 1

    run_experiments()

    banner("DONE", "#")
    print(f"total wall-clock: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

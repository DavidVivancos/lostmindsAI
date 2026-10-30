#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 THE BAY'A COALITION NETWORK
 Chapter 303 -- Saladin (Salah al-Din Yusuf ibn Ayyub, b. Tikrit 1137 - d.
 Damascus 4 March 1193). Encyclopedia of Lost Minds: Echoes on AI.
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0303 · Saladin (Salah al-Din Yusuf ibn Ayyub, b. Tikrit 1137 - d. Damascus 4 March 1193)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 16, Minds 301-320 Available on Amazon https://www.amazon.com/gp/product/B0HNJQX9S4
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos 


WHAT THIS FILE IS NOT
----------------------
This is not a "chivalry AI" or an "honor layer." The received Western image of
Saladin -- gallant, merciful, the "ethical warrior" of Walter Scott and Dante's
Limbo -- is a real historical effect, but it is an EFFECT, not the mechanism
that produced it. The mechanism is what this file tries to encode.

THE THESIS
----------
Saladin's genius was not battlefield brilliance (he lost battles: Montgisard,
1177) and it was not moral exceptionalism (he executed prisoners, suppressed
the Fatimids, and had Reynald of Chatillon beheaded with his own hand). His
genius was COALITION ENGINEERING: he took a fractured set of self-interested,
independently armed power-holders -- his brother al-Adil, his son al-Afdal,
his nephew Taqi al-Din, his brother Turan-Shah, plus Kurdish, Turkish and Arab
emirs who owed him nothing by blood -- and welded them into one coordinated,
decade-long military project (the recovery of Jerusalem, complete 1187) using
two instruments simultaneously:

  1. MATERIAL:  the iqta' system -- land-revenue assignments that bought each
     commander's cooperation without a standing central treasury strong enough
     to compel it by force.

  2. NARRATIVE: a chancery (diwan al-insha'), run first by al-Qadi al-Fadil and
     then also by Imad al-Din al-Isfahani, that manufactured and broadcast a
     PREDICTABLE public record of Saladin's conduct -- mercy for the compliant
     (Guy of Lusignan, released after Hattin), inescapable and public death for
     oath-breakers (Reynald of Chatillon, executed personally by Saladin after
     Hattin for violating a truce). Anne-Marie Eddé's biography of Saladin
     calls him, without irony, "the world's first spin doctor": the coalition
     held together because independent, self-interested agents could each
     predict how Saladin would treat them, and predictability made cooperation
     the rational bet even for people with no blood tie to him at all.

The diagnostic fact, and the one this file's self-tests are built around: the
coalition did NOT survive him. Saladin died in 1193; within five years his
sons and brother were at open war over the appanages he had assigned them
(Damascus to al-Afdal, Egypt to al-Aziz, Aleppo to az-Zahir), and it took his
brother al-Adil until 1200-1202 to reunify the core Ayyubid lands by force.
The trust that held the coalition together was bound to Saladin's OWN
reputation -- not to any institution built to survive him.

For a multi-agent AGI system, this is a direct and testable proposition:
publicly auditable, predictable reward/punishment rules can substitute for
centralized command in coordinating autonomous, self-interested agents -- but
only for as long as the guarantor persists. If trust is anchored to one node's
credibility rather than encoded in transferable, protocol-level rules, the
network fragments the moment that node is removed. This file builds a small,
literally trainable model of that claim and then tests it.

ARCHITECTURE SUMMARY (embodied below, not just asserted)
----------------------------------------------------------
  * Bilinear TRUST between agent embeddings (a learned compatibility form),
    biased by (a) kinship, (b) accumulated interaction history, and (c) a
    LEGITIMACY EMBEDDING -- but the legitimacy bias acts ONLY on leader-
    mediated trust, never on peer-to-peer trust, because the chancery's
    broadcast is a channel that runs through the sultan, not between his
    kinsmen directly.
  * A GRU-style LEGITIMACY CELL that integrates the leader's action record
    over time through a trainable "spin_gain" -- how much the chancery
    amplifies the raw record before it reaches the coalition.
  * Trust-gated MESSAGE PASSING between agents (information/resources flow
    more freely between agents who trust each other).
  * A LEADER POLICY that must, every step, decide how to split a scarce
    resource (iqta') across followers and how hard to press each of three
    fronts (Jerusalem, Acre, Kerak) -- and, at one scripted turning point,
    whether to pardon or execute a captured oath-breaker (the Reynald of
    Chatillon decision, modelled at Kerak, his actual fortress).
  * A succession-fragility SELF-TEST that removes the leader node from the
    trained trust graph and checks whether the coalition would still cohere
    -- the model's version of the historical collapse of 1193-1202.

Everything below is pure NumPy: a small from-scratch reverse-mode autodiff
engine (array-valued nodes, not scalar), a real multi-step backprop-through-
time training loop (Adam), a MANDATORY finite-difference gradient check that
must pass before training is trusted, and a self-test suite that ties the
model's behaviour back to the historical claim above. Run this file directly:

    python3 0303_saladin_1137_Neuron.py

SIMPLIFICATIONS, STATED OPENLY
-------------------------------
  * N=5 agents, 3 provinces, 6 timesteps: small enough that finite-difference
    gradient checking is fast and exact; the mechanisms scale, the toy world
    does not need to.
  * "Territory capture" is modelled as a smooth, flow-style reward (sigmoid of
    effort minus a threshold) rather than a hard captured/uncaptured state, to
    keep the whole 6-step rollout end-to-end differentiable without needing a
    separate policy-gradient estimator.
  * Regularization (ridge penalties, heavier on peer-to-peer trust growth than
    on leader-mediated trust growth) is necessary and is documented at the
    point it is used: without it, trust saturates uniformly to 1.0 for every
    pair and the leader/peer asymmetry the historical claim depends on is
    washed out by the optimizer's unconstrained incentive to maximize
    cohesion. This is disclosed, not hidden, in the code comments below.

AUTHOR'S NOTE ON PROVENANCE
-----------------------------
No philosophical writing survives in Saladin's own hand. What survives is
mediated through the chancery he built and the companions he chose to write
his history (Baha al-Din ibn Shaddad, Imad al-Din al-Isfahani) -- an
admiring, not hostile, mediation, but a mediation all the same. See the
chapter text (chapter_0303_saladin_1137.md) for full sourcing.
================================================================================
"""

import numpy as np
import copy


# ==============================================================================
# PART I -- A SMALL REVERSE-MODE AUTODIFF ENGINE (pure NumPy, array-valued)
# ==============================================================================

def _unbroadcast(grad, shape):
    """Sum-reduce `grad` down to `shape` after a broadcasted op."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    """
    An array-valued node in a dynamically built computation graph.
    Unlike scalar autodiff toys (e.g. micrograd), each Tensor wraps a whole
    NumPy array, so a single node can be a scalar, a vector, or a matrix --
    which is what a trust matrix / embedding table / weight matrix needs.
    """
    __slots__ = ("data", "grad", "requires_grad", "_backward", "_prev", "_op")

    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    # ---------------- arithmetic ----------------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, self.requires_grad or other.requires_grad,
                      (self, other), "+")

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        out = Tensor(-self.data, self.requires_grad, (self,), "neg")

        def _backward():
            if self.requires_grad:
                self.grad += -out.grad
        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, self.requires_grad or other.requires_grad,
                      (self, other), "*")

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad * self.data, other.data.shape)
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * other.reciprocal()

    def reciprocal(self, eps=1e-12):
        safe = self.data + np.where(self.data >= 0, eps, -eps)
        out = Tensor(1.0 / safe, self.requires_grad, (self,), "recip")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * (-1.0 / (safe ** 2))
        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, self.requires_grad or other.requires_grad,
                      (self, other), "matmul")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    def T(self):
        out = Tensor(self.data.T, self.requires_grad, (self,), "T")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.T
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), self.requires_grad,
                      (self,), "sum")

        def _backward():
            if self.requires_grad:
                g = out.grad
                if axis is not None and not keepdims:
                    g = np.expand_dims(g, axis=axis)
                self.grad += np.ones_like(self.data) * g
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, self.requires_grad, (self,), "tanh")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * (1 - t * t)
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-np.clip(self.data, -60, 60)))
        out = Tensor(s, self.requires_grad, (self,), "sigmoid")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * s * (1 - s)
        out._backward = _backward
        return out

    def relu(self):
        r = np.maximum(self.data, 0)
        out = Tensor(r, self.requires_grad, (self,), "relu")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * (self.data > 0)
        out._backward = _backward
        return out

    def log(self, eps=1e-12):
        safe = np.clip(self.data, eps, None)
        out = Tensor(np.log(safe), self.requires_grad, (self,), "log")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad / safe
        out._backward = _backward
        return out

    def softmax(self, axis=-1):
        z = self.data - np.max(self.data, axis=axis, keepdims=True)
        e = np.exp(z)
        s = e / np.sum(e, axis=axis, keepdims=True)
        out = Tensor(s, self.requires_grad, (self,), "softmax")

        def _backward():
            if self.requires_grad:
                g = out.grad
                dot = np.sum(g * s, axis=axis, keepdims=True)
                self.grad += s * (g - dot)
        out._backward = _backward
        return out

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        out = Tensor(self.data.reshape(shape), self.requires_grad, (self,), "reshape")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(self.data.shape)
        out._backward = _backward
        return out

    def getitem(self, idx):
        out = Tensor(self.data[idx], self.requires_grad, (self,), "getitem")

        def _backward():
            if self.requires_grad:
                g = np.zeros_like(self.data)
                g[idx] += out.grad
                self.grad += g
        out._backward = _backward
        return out

    def __getitem__(self, idx):
        return self.getitem(idx)

    def backward(self):
        topo, visited = [], set()

        def build(v):
            if id(v) not in visited:
                visited.add(id(v))
                for c in v._prev:
                    build(c)
                topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad})"


def cat(tensors, axis=-1):
    """Concatenate Tensors along `axis`; backward splits the grad back out."""
    datas = [t.data for t in tensors]
    sizes = [d.shape[axis] for d in datas]
    req = any(t.requires_grad for t in tensors)
    out = Tensor(np.concatenate(datas, axis=axis), req, tuple(tensors), "cat")

    def _backward():
        splits = np.split(out.grad, np.cumsum(sizes)[:-1], axis=axis)
        for t, g in zip(tensors, splits):
            if t.requires_grad:
                t.grad += g
    out._backward = _backward
    return out


# ==============================================================================
# PART II -- THE BAY'A COALITION NETWORK
# ==============================================================================

rng = np.random.default_rng(7)

N = 5       # agents: 0 Saladin (leader), 1 al-Adil, 2 al-Afdal, 3 Taqi al-Din, 4 Turan-Shah
F0 = 4      # static per-agent features
D = 6       # agent embedding dimension
Dl = 4      # legitimacy embedding dimension
Da = 3      # broadcast action vector dimension
P = 3       # provinces: Jerusalem, Acre, Kerak
H1 = 12     # leader-policy hidden width
Hg = 8      # punish-gate hidden width
T_STEPS = 6
EVENT_T = 3     # timestep of the Kerak / Reynald-of-Chatillon defection event
KERAK_IDX = 2   # province index for Kerak

# Trust-bias masks: row/col 0 is the leader; the rest is the follower (peer) block.
LEADER_MASK = np.zeros((N, N))
LEADER_MASK[0, :] = 1.0
LEADER_MASK[:, 0] = 1.0
np.fill_diagonal(LEADER_MASK, 0.0)
PEER_MASK = 1.0 - LEADER_MASK
np.fill_diagonal(PEER_MASK, 0.0)


def xavier(shape):
    fan_in = shape[0]
    fan_out = shape[1] if len(shape) > 1 else shape[0]
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=shape)


def init_params():
    """All trainable weights of the Bay'a Coalition Network."""
    p = {}
    p['W_enc'] = Tensor(xavier((F0, D)), requires_grad=True)
    p['b_enc'] = Tensor(np.zeros(D), requires_grad=True)

    p['M'] = Tensor(xavier((D, D)), requires_grad=True)  # bilinear trust compatibility
    # Separate scalars for leader<->follower bonds vs follower<->follower (peer) bonds.
    # The bay'a (oath) a general swears runs to the sultan personally, reinforced by the
    # chancery's public performance; two kinsmen have no equivalent institution binding
    # THEM to each other. A single shared scalar would let the optimizer buy peer trust
    # "for free" alongside leader trust; separating them lets the network actually learn
    # the asymmetry the succession-fragility self-test is built to expose.
    p['w_kin_leader'] = Tensor(np.array(0.8), requires_grad=True)
    p['w_hist_leader'] = Tensor(np.array(0.8), requires_grad=True)
    p['w_kin_peer'] = Tensor(np.array(0.3), requires_grad=True)
    p['w_hist_peer'] = Tensor(np.array(0.3), requires_grad=True)
    p['w_leg'] = Tensor(np.array(0.6), requires_grad=True)  # legitimacy -> leader-trust bias

    p['W_msg'] = Tensor(xavier((D, D)), requires_grad=True)
    p['W_self'] = Tensor(xavier((D, D)), requires_grad=True)
    p['b_msg'] = Tensor(np.zeros(D), requires_grad=True)

    gi = Dl + Da
    p['Wz'] = Tensor(xavier((gi, Dl)), requires_grad=True); p['bz'] = Tensor(np.zeros(Dl), requires_grad=True)
    p['Wr'] = Tensor(xavier((gi, Dl)), requires_grad=True); p['br'] = Tensor(np.zeros(Dl), requires_grad=True)
    p['Wh'] = Tensor(xavier((gi, Dl)), requires_grad=True); p['bh'] = Tensor(np.zeros(Dl), requires_grad=True)
    p['spin'] = Tensor(np.ones(Da) * 0.5, requires_grad=True)  # narrative amplification gain

    pin = D + Dl + 1
    p['Wp1'] = Tensor(xavier((pin, H1)), requires_grad=True); p['bp1'] = Tensor(np.zeros(H1), requires_grad=True)
    p['W_alloc'] = Tensor(xavier((H1, N - 1)), requires_grad=True); p['b_alloc'] = Tensor(np.zeros(N - 1), requires_grad=True)
    p['W_effort'] = Tensor(xavier((H1, P)), requires_grad=True); p['b_effort'] = Tensor(np.zeros(P), requires_grad=True)

    p['Wg1'] = Tensor(xavier((4, Hg)), requires_grad=True); p['bg1'] = Tensor(np.zeros(Hg), requires_grad=True)
    p['Wg2'] = Tensor(xavier((Hg, 1)), requires_grad=True); p['bg2'] = Tensor(np.zeros(1), requires_grad=True)
    return p


def default_scenario(violation_flag=1.0):
    """
    The historical scaffold: five real people, real approximate relationships,
    three real objectives.
      agents: 0 Saladin, 1 al-Adil (brother), 2 al-Afdal (eldest son),
               3 Taqi al-Din (nephew, notably independent-minded general),
               4 Turan-Shah (brother, sent to conquer distant Yemen in 1174).
      features per agent: [kinship_to_leader, initial_iqta_resource,
                            region_distance_from_core, initial_military_skill]
      provinces: Jerusalem (highest prize, best defended), Acre, and Kerak --
      Reynald of Chatillon's actual fortress, the site of the scripted
      defection event at t=EVENT_T.
    """
    X0 = np.array([
        [1.00, 0.9, 0.0, 0.9],
        [0.90, 0.7, 0.2, 0.8],
        [0.85, 0.6, 0.1, 0.5],
        [0.55, 0.5, 0.3, 0.85],
        [0.70, 0.5, 0.7, 0.6],
    ], dtype=np.float64)
    kinship = np.array([
        [0, .9, .85, .55, .70],
        [.9, 0, .6, .5, .65],
        [.85, .6, 0, .4, .5],
        [.55, .5, .4, 0, .35],
        [.70, .65, .5, .35, 0],
    ], dtype=np.float64)
    province_value = np.array([1.4, 1.0, 0.8])       # Jerusalem, Acre, Kerak
    threshold = np.array([0.55, 0.4, 0.5])
    return {
        'X0': Tensor(X0, requires_grad=False),
        'kinship': Tensor(kinship, requires_grad=False),
        'province_value': Tensor(province_value, requires_grad=False),
        'threshold': Tensor(threshold, requires_grad=False),
        'k_sharp': 8.0,
        'violation_flag': violation_flag,
        'kerak_kinship': 0.15,       # Reynald had no kin tie to the coalition
        'kerak_prior_trust': 0.10,   # and no track record of cooperation
    }


def encode_agents(p, X0):
    return (X0 @ p['W_enc'] + p['b_enc']).tanh()


def trust_matrix(p, H, kinship, hist, L):
    """
    Trust[i,j] = sigmoid( bilinear(H_i, H_j)  +  kinship-bias  +  history-bias
                          +  legitimacy-bias(leader row/col only) ).
    """
    HM = H @ p['M']
    logits = HM @ H.T()
    lead_mask = Tensor(LEADER_MASK)
    peer_mask = Tensor(PEER_MASK)
    kin_bias = kinship * (lead_mask * p['w_kin_leader'] + peer_mask * p['w_kin_peer'])
    hist_bias = hist * (lead_mask * p['w_hist_leader'] + peer_mask * p['w_hist_peer'])
    # The legitimacy embedding -- the chancery's running, spin-amplified broadcast of the
    # leader's record -- biases LEADER-mediated trust only: followers condition on the
    # sultan's publicized reputation when deciding whether to extend him trust, but that
    # broadcast does nothing for how much two kinsmen trust each other directly. This is
    # what makes the legitimacy-ablation self-test meaningful.
    leg_scalar = L.mean(axis=0)
    leg_bias = lead_mask * (p['w_leg'] * leg_scalar)
    logits = logits + kin_bias + hist_bias + leg_bias
    return logits.sigmoid()


def row_normalize(Tm, eps=1e-6):
    rowsum = Tm.sum(axis=1, keepdims=True)
    inv = (rowsum + eps).reciprocal()
    return Tm * inv


def message_pass(p, H, Tn):
    """Trust-gated GNN layer: agents exchange information proportional to trust."""
    agg = Tn @ H
    return (agg @ p['W_msg'] + H @ p['W_self'] + p['b_msg']).tanh()


def legitimacy_step(p, L_prev, action_t):
    """A GRU-style cell integrating the (spin-amplified) action record over time."""
    a = action_t * p['spin']
    inp2 = cat([L_prev, a], axis=0).reshape(1, -1)
    z = (inp2 @ p['Wz'] + p['bz']).sigmoid()
    r = (inp2 @ p['Wr'] + p['br']).sigmoid()
    rl = r * L_prev.reshape(1, -1)
    inp3 = cat([rl.reshape(-1), a], axis=0).reshape(1, -1)
    h_tilde = (inp3 @ p['Wh'] + p['bh']).tanh()
    one = Tensor(np.ones_like(z.data))
    L_new = (one - z) * L_prev.reshape(1, -1) + z * h_tilde
    return L_new.reshape(-1)


def leader_policy(p, H, L, t_norm):
    """Decide (a) resource allocation across followers, (b) campaign effort per province."""
    pooled = H.mean(axis=0)
    tnorm_t = Tensor(np.array([t_norm]))
    inp = cat([pooled, L, tnorm_t], axis=0).reshape(1, -1)
    h = (inp @ p['Wp1'] + p['bp1']).relu()
    alloc = (h @ p['W_alloc'] + p['b_alloc']).softmax(axis=1).reshape(-1)
    effort = (h @ p['W_effort'] + p['b_effort']).sigmoid().reshape(-1)
    return alloc, effort


def punish_gate(p, feat4):
    """
    feat4 = [violation_flag, kinship_to_leader, prior_trust, t_norm].
    The Reynald-of-Chatillon decision, generalized: punish or pardon a
    captured figure, conditioned on whether they actually broke an oath --
    not merely on whether punishing looks strong.
    """
    x = feat4.reshape(1, -1)
    h = (x @ p['Wg1'] + p['bg1']).relu()
    return (h @ p['Wg2'] + p['bg2']).sigmoid()


def rollout(p, scenario):
    """One full 6-timestep episode. Returns (total_reward, spin_gap_penalty, trust_trace)."""
    X0 = scenario['X0']; kinship = scenario['kinship']
    province_value = scenario['province_value']; threshold = scenario['threshold']
    k_sharp = scenario['k_sharp']; violation_flag = scenario['violation_flag']

    H = encode_agents(p, X0)
    hist = Tensor(np.zeros((N, N)))
    L = Tensor(np.zeros(Dl))
    total_reward = Tensor(np.array(0.0))
    spin_gap = Tensor(np.array(0.0))
    trust_trace = []

    for t in range(T_STEPS):
        T_full = trust_matrix(p, H, kinship, hist, L)
        Tn = row_normalize(T_full)
        H = message_pass(p, H, Tn)

        t_norm = t / (T_STEPS - 1)
        alloc, effort = leader_policy(p, H, L, t_norm)
        cohesion = T_full.getitem((0, slice(1, N))).mean(axis=0)
        realized_effort = effort * cohesion

        if t == EVENT_T:
            feat = Tensor(np.array([violation_flag, scenario['kerak_kinship'],
                                     scenario['kerak_prior_trust'], t_norm]))
            pg = punish_gate(p, feat).reshape(())
            legitimacy_bonus = pg * violation_flag * 0.6
            injustice_penalty = pg * (1.0 - violation_flag) * 0.6
            event_term = legitimacy_bonus - injustice_penalty
            boost_mask = Tensor(np.array([0.0, 0.0, 1.0]))  # boosts Kerak specifically
            realized_effort = realized_effort + boost_mask * (pg * violation_flag * 0.3)
            punish_signal = pg
        else:
            event_term = Tensor(np.array(0.0))
            punish_signal = Tensor(np.array(0.0))

        capture_prob = (Tensor(np.array(k_sharp)) * (realized_effort - threshold)).sigmoid()
        reward_t = (capture_prob * province_value).sum()
        total_reward = total_reward + reward_t + event_term

        mean_effort = effort.mean(axis=0)
        action_vec = cat([mean_effort.reshape(1), cohesion.reshape(1), punish_signal.reshape(1)], axis=0)
        L = legitimacy_step(p, L, action_vec)

        # Spin-gap penalty: legitimacy shouldn't drift arbitrarily far from the real
        # action record -- there is a limit to how much a chancery can inflate the truth
        # before the story stops being credible (Imad al-Din privately criticized
        # Saladin's own generosity at Acre and Jerusalem as excessive; spin has edges).
        L_read = L.getitem(slice(0, Da))
        gap = L_read - action_vec
        spin_gap = spin_gap + (gap * gap).sum()

        # Interaction history accumulates realized trust, kept IN the computational graph
        # (not detached) so gradients genuinely flow through the full multi-step
        # recurrence -- real backprop-through-time, not a truncated approximation.
        hist = hist * 0.9 + T_full
        trust_trace.append(T_full.data.copy())

    return total_reward, spin_gap, trust_trace


def calibration_loss(p):
    """
    Supervised probes for the punish-gate sub-network: a clear-cut oath-breaker
    (violation_flag=1, no kinship, no track record) should be punished; a
    clear-cut loyal kinsman with a long cooperative history should not.
    """
    violator_feat = Tensor(np.array([1.0, 0.10, 0.10, 0.5]))
    loyal_feat = Tensor(np.array([0.0, 0.85, 0.80, 0.5]))
    pv = punish_gate(p, violator_feat).reshape(())
    pl = punish_gate(p, loyal_feat).reshape(())
    loss_v = -(pv.log())
    loss_l = -((Tensor(np.array(1.0)) - pl).log())
    return loss_v + loss_l


LAMBDA_SPIN = 0.05
LAMBDA_CAL = 1.0
# Ridge regularization is NECESSARY, not decorative: without a cost on
# indiscriminate trust, "more trust always raises reward" pushes every entry of
# the trust matrix toward 1.0 regardless of kinship or history, which erases
# exactly the leader/peer asymmetry the succession-fragility self-test looks
# for. Peer-to-peer trust growth is penalized harder than leader-mediated
# trust growth because, per the thesis, peer bonds have no reinforcing
# institution behind them.
LAMBDA_RIDGE_TRUST_LEADER = 0.5
LAMBDA_RIDGE_TRUST_PEER = 2.0
LAMBDA_RIDGE_LEG = 0.5
LAMBDA_RIDGE_OTHER = 0.01


def compute_loss(p, scenario):
    reward, spin_gap, _ = rollout(p, scenario)
    cal = calibration_loss(p)
    trust_ridge = ((p['M'] * p['M']).sum()
                   + (p['w_kin_leader'] * p['w_kin_leader']) * LAMBDA_RIDGE_TRUST_LEADER
                   + (p['w_hist_leader'] * p['w_hist_leader']) * LAMBDA_RIDGE_TRUST_LEADER
                   + (p['w_kin_peer'] * p['w_kin_peer']) * LAMBDA_RIDGE_TRUST_PEER
                   + (p['w_hist_peer'] * p['w_hist_peer']) * LAMBDA_RIDGE_TRUST_PEER
                   + (p['w_leg'] * p['w_leg']) * LAMBDA_RIDGE_LEG)
    other_ridge = (p['W_msg'] * p['W_msg']).sum() + (p['W_self'] * p['W_self']).sum()
    loss = ((-reward) + spin_gap * LAMBDA_SPIN + cal * LAMBDA_CAL
            + trust_ridge * 0.08 + other_ridge * LAMBDA_RIDGE_OTHER)
    return loss


# ==============================================================================
# PART III -- MANDATORY FINITE-DIFFERENCE GRADIENT CHECK
# ==============================================================================

def grad_check(p, scenario, eps=1e-5, samples_per_param=4, seed=0):
    """
    Central finite-difference check of every trainable parameter tensor
    (a random sample of scalar entries per tensor, to keep this fast).
    Must pass (relative error < 1e-3) before training is trusted.
    """
    rng_local = np.random.default_rng(seed)
    results = []
    for name, t in p.items():
        if not t.requires_grad:
            continue
        idxs = list(np.ndindex(*t.data.shape))
        chosen = idxs if len(idxs) <= samples_per_param else [
            idxs[i] for i in rng_local.choice(len(idxs), size=samples_per_param, replace=False)
        ]
        for t2 in p.values():
            t2.zero_grad()
        loss = compute_loss(p, scenario)
        loss.backward()
        analytic_grads = {k: p[k].grad.copy() for k in p}

        for idx in chosen:
            orig = t.data[idx]
            t.data[idx] = orig + eps
            f1 = compute_loss(p, scenario).data.copy()
            t.data[idx] = orig - eps
            f2 = compute_loss(p, scenario).data.copy()
            t.data[idx] = orig
            numeric = (f1 - f2) / (2 * eps)
            analytic = analytic_grads[name][idx]
            denom = max(abs(numeric), abs(analytic), 1e-8)
            rel_err = abs(numeric - analytic) / denom
            results.append((name, idx, float(analytic), float(numeric), float(rel_err)))
    return results


# ==============================================================================
# PART IV -- TRAINING (Adam, real multi-step backprop-through-time)
# ==============================================================================

class Adam:
    def __init__(self, params, lr=0.03, b1=0.9, b2=0.999, eps=1e-8):
        self.params = params
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v.data) for k, v in params.items()}
        self.v = {k: np.zeros_like(v.data) for k, v in params.items()}
        self.t = 0

    def step(self):
        self.t += 1
        for k, p in self.params.items():
            g = p.grad
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)

    def zero_grad(self):
        for p in self.params.values():
            p.zero_grad()


def train(epochs=300, lr=0.03, seed=7, verbose=True):
    np.random.seed(seed)
    p = init_params()
    opt = Adam(p, lr=lr)
    scenario = default_scenario(violation_flag=1.0)
    history = []
    for ep in range(epochs):
        opt.zero_grad()
        loss = compute_loss(p, scenario)
        loss.backward()
        opt.step()
        if ep % max(1, epochs // 10) == 0 or ep == epochs - 1:
            reward, _, _ = rollout(p, scenario)
            history.append((ep, float(loss.data), float(reward.data)))
            if verbose:
                print(f"  epoch {ep:4d}   loss={float(loss.data): .4f}   "
                      f"territorial_reward={float(reward.data): .4f}")
    return p, history


# ==============================================================================
# PART V -- SELF-TESTS: does the trained network reproduce the historical claim?
# ==============================================================================

def fiedler_value(W):
    """Algebraic connectivity (2nd smallest Laplacian eigenvalue) of weighted graph W."""
    Ws = (W + W.T) / 2.0
    Ws = Ws.copy()
    np.fill_diagonal(Ws, 0.0)
    deg = np.diag(Ws.sum(axis=1))
    lap = deg - Ws
    eigvals = np.sort(np.linalg.eigvalsh(lap))
    return eigvals[1] if len(eigvals) > 1 else 0.0


def test_calibration(p):
    violator_feat = Tensor(np.array([1.0, 0.10, 0.10, 0.5]))
    loyal_feat = Tensor(np.array([0.0, 0.85, 0.80, 0.5]))
    pv = float(punish_gate(p, violator_feat).data.reshape(-1)[0])
    pl = float(punish_gate(p, loyal_feat).data.reshape(-1)[0])
    passed = (pv > 0.5) and (pl < 0.5) and (pv > pl)
    return passed, pv, pl


def test_succession_fragility(p, scenario):
    """
    Removes the leader node from the trust graph and checks whether the
    followers' direct, peer-to-peer bonds are as strong as the bonds they
    each individually had with the leader -- the model's version of the
    Ayyubid appanage system's collapse into fratricidal war after 1193.
    """
    _, _, trust_trace = rollout(p, scenario)
    T_final = trust_trace[-1]
    full_conn = fiedler_value(T_final)
    followers_only = T_final[1:, 1:]
    follower_conn = fiedler_value(followers_only)
    leader_to_follower_mean = T_final[0, 1:].mean()
    off_diag_sum = followers_only.sum() - np.trace(followers_only)
    n_off = followers_only.shape[0] * (followers_only.shape[0] - 1)
    follower_to_follower_mean = off_diag_sum / n_off
    passed = (follower_conn < full_conn) and (follower_to_follower_mean < leader_to_follower_mean)
    return passed, dict(full_connectivity=full_conn, follower_only_connectivity=follower_conn,
                         leader_follower_trust=leader_to_follower_mean,
                         follower_follower_trust=follower_to_follower_mean)


def test_spin_ablation(p, scenario):
    """With the chancery's narrative amplification switched off (spin_gain=0),
    does leader-mediated cohesion measurably fall?"""
    _, _, trace_on = rollout(p, scenario)
    cohesion_on = trace_on[-1][0, 1:].mean()

    p_off = copy.deepcopy(p)
    p_off['spin'] = Tensor(np.zeros_like(p['spin'].data))
    _, _, trace_off = rollout(p_off, scenario)
    cohesion_off = trace_off[-1][0, 1:].mean()

    passed = cohesion_on > cohesion_off
    return passed, dict(cohesion_with_legitimacy_broadcast=float(cohesion_on),
                         cohesion_without_broadcast=float(cohesion_off))


def test_reward_improves(history):
    passed = history[-1][2] > history[0][2]
    return passed, dict(first=history[0][2], last=history[-1][2])


def run_self_tests():
    print("=" * 78)
    print("SELF-TEST SUITE -- The Bay'a Coalition Network (Chapter 248, Saladin)")
    print("=" * 78)
    all_pass = True

    p0 = init_params()
    sc = default_scenario()
    print("\n[1] Mandatory finite-difference gradient check")
    res = grad_check(p0, sc)
    max_err = max(r[4] for r in res)
    n_fail = sum(1 for r in res if r[4] > 1e-3)
    ok = n_fail == 0
    all_pass &= ok
    print(f"    {len(res)} scalar gradients sampled across {len(p0)} parameter tensors")
    print(f"    max relative error: {max_err:.2e}   failures(>1e-3): {n_fail}")
    print(f"    {'PASS' if ok else 'FAIL'}")

    print("\n[2] Training (Adam, 300 epochs, real backprop-through-time)")
    p, history = train(epochs=300, verbose=True)

    print("\n[3] Territorial reward improves with training")
    ok, info = test_reward_improves(history)
    all_pass &= ok
    print(f"    first={info['first']:.4f}   last={info['last']:.4f}")
    print(f"    {'PASS' if ok else 'FAIL'}")

    print("\n[4] Punish-gate calibration (oath-breaker vs. loyal kinsman)")
    ok, pv, pl = test_calibration(p)
    all_pass &= ok
    print(f"    P(punish | violator, no kin tie) = {pv:.4f}")
    print(f"    P(punish | loyal, strong kin tie) = {pl:.4f}")
    print(f"    {'PASS' if ok else 'FAIL'}")

    sc_full = default_scenario(violation_flag=1.0)
    print("\n[5] Succession fragility (coalition trust structure with the leader removed)")
    ok, info = test_succession_fragility(p, sc_full)
    all_pass &= ok
    for k, v in info.items():
        print(f"    {k}: {v:.4f}")
    print(f"    {'PASS' if ok else 'FAIL'}")

    print("\n[6] Legitimacy-broadcast ablation (spin_gain forced to 0)")
    ok, info = test_spin_ablation(p, sc_full)
    all_pass &= ok
    for k, v in info.items():
        print(f"    {k}: {v:.4f}")
    print(f"    {'PASS' if ok else 'FAIL'}")

    print("\n" + "=" * 78)
    print(f"OVERALL: {'ALL SELF-TESTS PASSED' if all_pass else 'SOME SELF-TESTS FAILED'}")
    print("=" * 78)
    return all_pass


if __name__ == "__main__":
    run_self_tests()

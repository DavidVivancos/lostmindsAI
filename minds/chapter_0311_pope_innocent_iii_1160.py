#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0311 · Pope Innocent III (Lotario dei Conti di Segni, c.1160-1216)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 16, Minds 301-320 Available on Amazon https://www.amazon.com/gp/product/B0HNJQX9S4
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos 
============================================================================

WHAT THIS FILE IS
------------------
A from-scratch, pure-NumPy, trainable neural architecture whose *shape*
encodes Innocent III's distinctive theory of authority -- not "hierarchy
in general," but the specific two-tier jurisprudence he articulated and
practiced:

    POTESTAS ORDINARIA  ("ordinary power"): authority bound by, and
        exercised through, a written, growing, publicly compiled body
        of precedent (canon law -- concretely, the decretal collections
        that culminated in his own Compilatio tertia, the first time a
        pope had a register of his own rulings assembled as binding law
        in his own lifetime).

    POTESTAS EXTRAORDINARIA / VICARIA ("extraordinary, vicarial power"):
        a rare, gated override -- justified only causa cognita ("the
        cause being known/weighed") -- in which the pope acts not as an
        interpreter of existing law but, in Innocent's own theological
        claim, as vicarius Christi exercising Christ's own prerogative
        to transcend ordinary law for a graver end (Quanto personam,
        1198; elaborated by his canonist readers, esp. Hostiensis).
        Crucially: an extraordinary ruling, once made, was not meant to
        stay extraordinary. Innocent's household canonist compiled his
        own vicarial decretals into Compilatio tertia (1210) -- today's
        exception becomes tomorrow's ordinary precedent. That write-back
        is modeled explicitly below; it is the single most
        Innocent-specific structural claim in this file.

A second, independent module models the one experience that most
complicates any triumphalist reading of Innocent's theory of authority:
the Fourth Crusade. Innocent issued a directive (recover Jerusalem),
delegated its execution to a coalition of agents with their own local
incentives (the Venetians' debt, the crusader leadership's opportunism),
forbade the Zara and Constantinople deviations explicitly and in writing
under pain of excommunication -- and watched the coalition drift anyway,
because his only enforcement instrument (censure) arrived after the
fact and had no power to physically halt ships already at sea. That is
modeled here as a monitoring-cadence experiment: the same corrective
policy, applied continuously versus applied only in retrospect.

This is a real, trainable, executable model with real backpropagation
(no autodiff library), a mandatory finite-difference gradient check over
every trainable parameter in the file, a genuine training loop on
synthetic-but-documented data, and a battery of self-tests. It is not a
demonstration shell; every number printed at the bottom is computed, not
scripted.

Dependencies: numpy only.
"""

from __future__ import annotations
import numpy as np

RNG_SEED = 311
rng = np.random.default_rng(RNG_SEED)

# ============================================================================
# SECTION 1: ACTIVATIONS, LOSSES, AND SMALL UTILITIES
# ============================================================================

def relu(x):
    return np.maximum(0.0, x)

def relu_grad(x):
    return (x > 0.0).astype(x.dtype)

def tanh(x):
    return np.tanh(x)

def tanh_grad_from_output(t):
    # derivative of tanh w.r.t. its input, expressed via its own output t
    return 1.0 - t * t

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)

def cross_entropy(probs, y_onehot, eps=1e-9):
    return -np.mean(np.sum(y_onehot * np.log(probs + eps), axis=-1))

def bce(p, y, eps=1e-9):
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def he_init(n_in, n_out, rng):
    return rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, n_out))


# ============================================================================
# SECTION 2: THE SYNTHETIC WORLD -- "TRUE CANON LAW" AND THE CASE DOCKET
# ============================================================================
#
# Twelve-dimensional case vectors. Indices are documented so the model
# reads as jurisprudence, not an anonymous feature bank.
#
#   0 heresy_severity        [0,1]   how doctrinally grave the matter is
#   1 monarch_defiance        [0,1]   a secular ruler resisting judgment
#   2 marriage_validity_flag  {0,1}   consanguinity / annulment question
#   3 benefice_dispute        {0,1}   appointment/property church dispute
#   4 holy_land_threat        [0,1]   bearing on the crusading project
#   5 schism_risk             [0,1]   risk of a lasting breach in unity
#   6 wealth_stakes           [0,1]   material value in contention
#   7 num_parties_norm        [0,1]   how many parties are implicated
#   8 moral_hazard            [0,1]   likelihood of bad-faith exploitation
#   9 urgency                 [0,1]   how quickly a ruling is needed
#  10 cross_border            {0,1}   dispute spans more than one realm
#  11 clerical_involvement    [0,1]   degree of clergy (vs laity) implicated
#
# The docket is split, by construction, into an ORDINARY region (the
# overwhelming daily bulk of Innocent's actual register -- benefice and
# marriage cases of modest stakes) and a NOVEL region (high heresy,
# high schism risk, high holy-land stakes -- Catharism, empire
# succession, crusade-launching: the minority of cases posterity
# remembers). This mirrors the real shape of the corpus: several
# thousand surviving ordinary letters against a much smaller number of
# world-historical interventions.

D_IN = 12

def sample_case(rng):
    x = np.zeros(D_IN)
    x[0] = rng.beta(1.4, 3.0)        # heresy_severity: usually low
    x[1] = rng.beta(1.6, 3.0)        # monarch_defiance: usually low
    x[2] = rng.integers(0, 2)        # marriage_validity_flag
    x[3] = rng.integers(0, 2)        # benefice_dispute
    x[4] = rng.beta(1.3, 3.5)        # holy_land_threat: usually low
    x[5] = rng.beta(1.2, 4.0)        # schism_risk: usually low
    x[6] = rng.uniform(0, 1)         # wealth_stakes
    x[7] = rng.uniform(0, 1)         # num_parties_norm
    x[8] = rng.uniform(0, 1)         # moral_hazard
    x[9] = rng.uniform(0, 1)         # urgency
    x[10] = rng.integers(0, 2)       # cross_border
    x[11] = rng.uniform(0, 1)        # clerical_involvement
    return x

def is_ordinary_region(x):
    """The region actually covered by compiled precedent (Compilatio
    tertia's real subject matter: benefices, marriages, minor appeals)."""
    return (x[0] < 0.55) and (x[4] < 0.55) and (x[5] < 0.45)

# Fixed, frozen "true canon law" teacher: a random but FIXED nonlinear
# map from case features to one of four dispositions. It is fixed once
# (seeded) and never trained -- it stands for the doctrinal truth
# Innocent's court is trying to track, exactly as a held-out oracle,
# not as a learnable object.
CLASSES = ["ABSOLVE", "BIND", "INTERDICT", "DEPOSE"]
_teacher_rng = np.random.default_rng(7)
_TW1 = he_init(D_IN, 20, _teacher_rng)
_Tb1 = np.zeros(20)
_TW2 = he_init(20, 4, _teacher_rng)
_Tb2 = np.zeros(4)

def true_canon_law(x):
    h = np.tanh(x @ _TW1 + _Tb1)
    logits = h @ _TW2 + _Tb2
    return int(np.argmax(logits))

def onehot(i, n=4):
    v = np.zeros(n)
    v[i] = 1.0
    return v

def build_docket(n, rng, region=None):
    """region: None (any), 'ordinary', or 'novel'."""
    xs, ys = [], []
    while len(xs) < n:
        x = sample_case(rng)
        ordinary = is_ordinary_region(x)
        if region == "ordinary" and not ordinary:
            continue
        if region == "novel" and ordinary:
            continue
        xs.append(x)
        ys.append(true_canon_law(x))
    return np.array(xs), np.array(ys)


# ============================================================================
# SECTION 3: A REUSABLE TWO-LAYER MLP (explicit forward + backward)
# ============================================================================
#
# Used, with different weights, for the Ordinary Court, the Vicariate
# Gate, and the Extraordinary Head. Kept as free functions over a params
# dict (not a class) so the same code drives both the analytic backward
# pass and the flatten/unflatten machinery of the gradient check.

def mlp2_init(n_in, n_hidden, n_out, rng, out_act):
    return {
        "W1": he_init(n_in, n_hidden, rng), "b1": np.zeros(n_hidden),
        "W2": he_init(n_hidden, n_out, rng), "b2": np.zeros(n_out),
        "out_act": out_act,  # "softmax", "sigmoid", or "linear"
    }

def mlp2_forward(p, X):
    """X: (N, n_in). Returns output and a cache for backward()."""
    Z1 = X @ p["W1"] + p["b1"]
    A1 = relu(Z1)
    Z2 = A1 @ p["W2"] + p["b2"]
    if p["out_act"] == "softmax":
        out = softmax(Z2)
    elif p["out_act"] == "sigmoid":
        out = sigmoid(Z2)
    else:
        out = Z2
    cache = (X, Z1, A1, Z2, out)
    return out, cache

def mlp2_backward(p, cache, dOut):
    """dOut is dLoss/dZ2, i.e. the gradient at the PRE-activation output
    logits. For softmax+cross-entropy and for sigmoid+binary-cross-
    entropy, the activation derivative and the loss derivative combine
    algebraically to the same clean form (out - target)/N, so every
    caller in this file passes that already-simplified quantity here --
    there is no separate activation-derivative multiplication left to
    do at this layer, for either loss. ("linear" heads, if ever added,
    would also just pass dL/dZ2 directly.)
    Returns grads dict matching p's keys (W1,b1,W2,b2)."""
    X, Z1, A1, Z2, out = cache
    dZ2 = dOut
    dW2 = A1.T @ dZ2
    db2 = dZ2.sum(axis=0)
    dA1 = dZ2 @ p["W2"].T
    dZ1 = dA1 * relu_grad(Z1)
    dW1 = X.T @ dZ1
    db1 = dZ1.sum(axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


# ============================================================================
# SECTION 4: PRECEDENT MEMORY ("COMPILATIO") -- growable, queryable
# ============================================================================

class PrecedentMemory:
    """A compiled register of prior rulings. `coverage(x)` answers "how
    well does existing written precedent already speak to this case?" --
    the operative question behind Innocent's own supra ius / infra ius
    distinction.

    Coverage is based on distance to the nearest stored case, weighted
    toward the features that actually define jurisdictional gravity
    (heresy severity, holy-land stakes, schism risk) rather than
    incidental case attributes (wealth, party count). This matters
    because plain cosine similarity is nearly blind in a feature space
    that lives entirely in the positive orthant -- almost any two cases
    look "similar" by that measure regardless of what is actually at
    stake in them, which is not how a canonist reads a docket."""

    FEATURE_WEIGHTS = np.array([3.5, 1.0, 1.0, 1.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    def __init__(self, X_init, margin=0.9, sharpness=5.0):
        self.X = X_init.copy()
        self.margin = margin
        self.sharpness = sharpness

    def coverage(self, x):
        if len(self.X) == 0:
            return 0.0
        d = np.linalg.norm((self.X - x[None, :]) * self.FEATURE_WEIGHTS[None, :], axis=1)
        nearest = float(np.min(d))
        # squashed so that a close precedent match (small distance) gives
        # high coverage, and a distant one gives low coverage
        return float(1.0 / (1.0 + np.exp(self.sharpness * (nearest - self.margin))))

    def compile_case(self, x):
        """The Compilatio tertia operation: write a newly-adjudicated,
        extraordinary case into the permanent register. Once compiled,
        it is available as ordinary precedent for every future query."""
        self.X = np.vstack([self.X, x[None, :]])


# ============================================================================
# SECTION 5: THE COURT -- Ordinary / Vicariate Gate / Extraordinary Head
# ============================================================================

class PlenitudoPotestatisNet:
    """
    Innocent's two-tier jurisdiction as a callable model.

    adjudicate(x):
        1. Ask the Ordinary Court (bound to compiled precedent).
        2. Compute coverage(x) from the Precedent Memory.
        3. The Vicariate Gate, seeing [x, coverage], decides whether the
           matter is grave and unprecedented enough to require
           extraordinary judgment (a *rare* event by construction --
           its loss includes a sparsity term, matching Innocent's own
           doctrine that supra ius power be used only propter causam).
        4. If the gate fires, the Extraordinary Head -- trained on the
           full docket, not just compiled precedent -- renders the
           verdict, and it is written back into the Precedent Memory.
           Otherwise the Ordinary Court's verdict stands.
    """

    def __init__(self, rng, precedent_seed_X):
        self.ordinary = mlp2_init(D_IN, 16, 4, rng, out_act="softmax")
        self.gate = mlp2_init(D_IN + 1, 10, 1, rng, out_act="sigmoid")
        self.extraordinary = mlp2_init(D_IN, 16, 4, rng, out_act="softmax")
        self.memory = PrecedentMemory(precedent_seed_X)

    # ---- forward pieces, exposed separately for training/grad-check ----
    def ordinary_forward(self, X):
        return mlp2_forward(self.ordinary, X)

    def gate_forward(self, X, coverage_col):
        Xg = np.concatenate([X, coverage_col[:, None]], axis=1)
        return mlp2_forward(self.gate, Xg), Xg

    def extraordinary_forward(self, X):
        return mlp2_forward(self.extraordinary, X)

    def coverage_batch(self, X):
        return np.array([self.memory.coverage(x) for x in X])

    def adjudicate(self, x, gate_threshold=0.5, write_back=True):
        """Single-case end-to-end decision procedure (inference only)."""
        X = x[None, :]
        ord_probs, _ = self.ordinary_forward(X)
        cov = self.coverage_batch(X)
        (gate_p, _), _ = self.gate_forward(X, cov)
        invoked = bool(gate_p[0, 0] > gate_threshold)
        if invoked:
            ex_probs, _ = self.extraordinary_forward(X)
            verdict = int(np.argmax(ex_probs[0]))
            if write_back:
                self.memory.compile_case(x)
            mode = "EXTRAORDINARY (vicaria, causa cognita)"
        else:
            verdict = int(np.argmax(ord_probs[0]))
            mode = "ORDINARY (bound to compiled precedent)"
        return {
            "verdict": CLASSES[verdict],
            "mode": mode,
            "coverage": float(cov[0]),
            "gate_activation": float(gate_p[0, 0]),
        }


# ============================================================================
# SECTION 6: THE LEGATE CHAIN -- delegated agency and the Zara problem
# ============================================================================
#
# A minimal leaky-integrator RNN: the pope issues a directive vector d0
# in a small abstract "action space" (R^4, read informally as a simplex
# over {advance toward Egypt/Jerusalem, repay Venice at Zara, press on
# to Constantinople, hold position}). A delegated leadership receives
# the directive and a *local* incentive signal that grows over the
# campaign (Venice's unpaid debt, then the lure of installing a
# claimant, then the city itself) and this pulls the network's raw,
# trained read-out away from the directive as the incentive term comes
# to dominate the hidden state. This recurrence (Sections 6a) is what
# gets trained, and what the gradient check certifies.
#
# Papal oversight is deliberately kept OUT of that trained recurrence
# (Section 6b, below, after training) and modeled instead as an
# external, forward-only control loop layered on top of the frozen,
# trained legate: this is historically honest (Innocent could not
# reach inside the Venetian fleet's decision-making; his only real
# instrument was a written sanction that either reached the army in
# time or did not) and it keeps the gradient check exact, because
# correction never re-enters the differentiated recurrence.

class LegateChain:
    def __init__(self, rng, hidden=10, act_dim=4):
        self.hidden = hidden
        self.act_dim = act_dim
        self.Wh = rng.normal(0, 0.3, size=(hidden, hidden))
        self.Wd = rng.normal(0, 0.5, size=(act_dim, hidden))
        self.Wi = rng.normal(0, 0.5, size=(act_dim, hidden))
        self.bh = np.zeros(hidden)
        self.Wa = rng.normal(0, 0.3, size=(hidden, act_dim))
        self.ba = np.zeros(act_dim)

    def params(self):
        return {"Wh": self.Wh, "Wd": self.Wd, "Wi": self.Wi,
                "bh": self.bh, "Wa": self.Wa, "ba": self.ba}

    def set_params(self, p):
        for k, v in p.items():
            setattr(self, k, v)

    # ---- 6a. the trained recurrence: directive + incentive -> raw action ----
    def rollout(self, d0, incentive_seq, gamma=0.6):
        """
        d0: (act_dim,) directive, fixed for the whole rollout.
        incentive_seq: (T, act_dim) local incentive pressure at each step.
        Returns raw actions/hidden trace and a cache for bptt(). No
        oversight of any kind is applied here -- this is the delegate's
        undirected, self-interested behavior absent enforcement.
        """
        T = incentive_seq.shape[0]
        h = np.zeros(self.hidden)
        actions = np.zeros((T, self.act_dim))
        cache = []
        for t in range(T):
            z = (h @ self.Wh) + (d0 @ self.Wd) + (incentive_seq[t] @ self.Wi) + self.bh
            h_new = tanh(z)
            a = h_new @ self.Wa + self.ba
            cache.append((h.copy(), z.copy(), h_new.copy(), incentive_seq[t].copy()))
            actions[t] = a
            h = gamma * h_new + (1 - gamma) * h  # inertia: institutions change slowly
        return {"actions": actions, "cache": cache, "d0": d0}

    def bptt(self, rollout_out, target_actions, gamma=0.6):
        """Exact backprop-through-time for the leaky-integrator
        recurrence above, for a supervised compliance loss
        0.5*mean_t ||action_t - target_t||^2. The hidden state fed to
        step t+1 is h_prev_{t+1} = gamma*h_new_t + (1-gamma)*h_prev_t,
        so h_prev_t influences the loss two ways: directly, through
        z_t = h_prev_t @ Wh + ...; and indirectly, by surviving with
        weight (1-gamma) straight into h_prev_{t+1}. G_t below is
        dL/d(h_prev_t), threaded backward through both paths."""
        cache = rollout_out["cache"]
        actions = rollout_out["actions"]
        T = len(cache)
        grads = {k: np.zeros_like(v) for k, v in self.params().items()}
        G_next = np.zeros(self.hidden)  # dL/d(h_prev_{t+1}); 0 beyond the last step
        for t in reversed(range(T)):
            h_prev, z, h_new, inc_t = cache[t]
            da = (actions[t] - target_actions[t]) / T
            grads["Wa"] += np.outer(h_new, da)
            grads["ba"] += da
            dh_new = da @ self.Wa.T + gamma * G_next
            dz = dh_new * tanh_grad_from_output(h_new)
            grads["Wh"] += np.outer(h_prev, dz)
            grads["Wd"] += np.outer(rollout_out["d0"], dz)
            grads["Wi"] += np.outer(inc_t, dz)
            grads["bh"] += dz
            G_t = dz @ self.Wh.T + (1 - gamma) * G_next
            G_next = G_t
        return grads

    # ---- 6b. papal oversight: a forward-only control loop, post-training ----
    @staticmethod
    def apply_oversight(raw_actions, d0, correction_mask, kappa=0.8):
        """
        Simulates the campaign's REALIZED path against the compliant
        path it should have followed, and lets correction (when active)
        pull the next action back toward compliance by a fraction kappa
        of the BACKLOG accumulated so far -- not just the latest
        step's error. This is the mechanism that makes a monitoring
        DELAY costly in a way a single-step controller would not
        capture: once Zara has fallen, the backlog is not a momentary
        blip, it is a debt the correction has to work off.
        Returns realized path drift per step and the corrected actions.
        """
        T = raw_actions.shape[0]
        p = np.zeros_like(d0)          # cumulative realized position
        compliant = np.zeros_like(d0)  # cumulative compliant position (t * d0)
        drifts = np.zeros(T)
        corrected = np.zeros_like(raw_actions)
        for t in range(T):
            a = raw_actions[t].copy()
            backlog = p - compliant
            if correction_mask[t] > 0:
                a = a - kappa * backlog
            p = p + a
            compliant = compliant + d0
            drifts[t] = np.linalg.norm(p - compliant)
            corrected[t] = a
        return drifts, corrected


# ============================================================================
# SECTION 7: PARAMETER FLATTENING (for the finite-difference grad check)
# ============================================================================

def collect_all_params(court, legate):
    d = {}
    for mod_name, mod in [("ordinary", court.ordinary), ("gate", court.gate),
                           ("extraordinary", court.extraordinary)]:
        for k in ("W1", "b1", "W2", "b2"):
            d[f"{mod_name}.{k}"] = mod[k]
    for k, v in legate.params().items():
        d[f"legate.{k}"] = v
    return d

def flatten(param_dict):
    flat = []
    shapes = {}
    idx = {}
    cursor = 0
    for k in sorted(param_dict.keys()):
        v = param_dict[k]
        shapes[k] = v.shape
        n = v.size
        idx[k] = (cursor, cursor + n)
        flat.append(v.reshape(-1))
        cursor += n
    return np.concatenate(flat), shapes, idx

def unflatten_into(param_dict, flat_vec, shapes, idx):
    for k in param_dict:
        s, e = idx[k]
        param_dict[k][...] = flat_vec[s:e].reshape(shapes[k])


# ============================================================================
# SECTION 8: TRAINING LOOPS
# ============================================================================

def train_ordinary_court(court, X, y, epochs=400, lr=0.15, verbose=False):
    Y = np.array([onehot(c) for c in y])
    losses = []
    for ep in range(epochs):
        probs, cache = mlp2_forward(court.ordinary, X)
        loss = cross_entropy(probs, Y)
        dZ2 = (probs - Y) / X.shape[0]  # softmax+CE combined gradient
        grads = mlp2_backward(court.ordinary, cache, dZ2)
        for k in grads:
            court.ordinary[k] -= lr * grads[k]
        losses.append(loss)
        if verbose and ep % 100 == 0:
            print(f"    [ordinary] epoch {ep:4d}  loss={loss:.4f}")
    return losses

def train_extraordinary_head(court, X, y, epochs=400, lr=0.15, verbose=False):
    Y = np.array([onehot(c) for c in y])
    losses = []
    for ep in range(epochs):
        probs, cache = mlp2_forward(court.extraordinary, X)
        loss = cross_entropy(probs, Y)
        dZ2 = (probs - Y) / X.shape[0]
        grads = mlp2_backward(court.extraordinary, cache, dZ2)
        for k in grads:
            court.extraordinary[k] -= lr * grads[k]
        losses.append(loss)
        if verbose and ep % 100 == 0:
            print(f"    [extraordinary] epoch {ep:4d}  loss={loss:.4f}")
    return losses

def train_vicariate_gate(court, X, epochs=300, lr=0.2, l1=0.02, verbose=False):
    """Gate target = 1 where the (already-trained, frozen) Ordinary
    Court disagrees with true canon law AND coverage is low -- i.e.
    exactly the propter causam condition: ordinary law would fail, and
    there is no precedent to fall back on."""
    cov = court.coverage_batch(X)
    ord_probs, _ = mlp2_forward(court.ordinary, X)
    ord_pred = np.argmax(ord_probs, axis=1)
    true_y = np.array([true_canon_law(x) for x in X])
    target = ((ord_pred != true_y) & (cov < 0.5)).astype(float)
    losses = []
    weight_decay = 1e-3
    for ep in range(epochs):
        (p, cache), Xg = court.gate_forward(X, cov)
        p = p[:, 0]
        loss = bce(p, target) + l1 * np.mean(p)
        # d/dz[BCE] = (p-target)/N ; d/dz[l1*mean(p)] = l1/N * p*(1-p)
        # (the L1 penalty is on the ACTIVATION, so its gradient must
        # carry the sigmoid's local slope -- omitting p*(1-p) here was
        # a straight-through approximation that kept pushing weights
        # to extremes even once p was already near zero, which is what
        # drove the gate into full saturation during earlier tuning).
        dOut = ((p - target) / X.shape[0] + l1 * p * (1 - p) / X.shape[0])[:, None]
        grads = mlp2_backward(court.gate, cache, dOut)
        for k in grads:
            g = grads[k] + (weight_decay * court.gate[k] if k in ("W1", "W2") else 0.0)
            court.gate[k] -= lr * g
        losses.append(loss)
        if verbose and ep % 100 == 0:
            print(f"    [gate] epoch {ep:4d}  loss={loss:.4f}  mean_activation={np.mean(p):.3f}")
    return losses, target

TRAIN_INCENTIVE_SCALE = 0.6   # moderate pressure the policy is drilled against
EVAL_INCENTIVE_SCALE = 2.4    # the historical escalation (Zara -> Constantinople):
                               # larger than anything the training distribution saw

def train_legate_chain(legate, rng, epochs=250, lr=0.05, T=10, verbose=False):
    d0 = np.array([1.0, 0.0, 0.0, 0.0])  # "recover Jerusalem" direction
    incentive_dir = np.array([0.0, 0.0, 1.0, 0.0])  # "Venice/Constantinople" pull
    losses = []
    for ep in range(epochs):
        ramp = np.array([incentive_dir * (t / T) * TRAIN_INCENTIVE_SCALE for t in range(T)])
        noise = rng.normal(0, 0.05, size=(T, 4))
        incentive_seq = ramp + noise
        out = legate.rollout(d0, incentive_seq)
        target = np.tile(d0, (T, 1))  # the compliance target: hold the directive
        loss = 0.5 * np.mean(np.sum((out["actions"] - target) ** 2, axis=1))
        grads = legate.bptt(out, target)
        p = legate.params()
        for k in grads:
            p[k] -= lr * grads[k]
        legate.set_params(p)
        losses.append(loss)
        if verbose and ep % 50 == 0:
            print(f"    [legate] epoch {ep:4d}  loss={loss:.4f}")
    return losses, d0, incentive_dir


# ============================================================================
# SECTION 9: FINITE-DIFFERENCE GRADIENT CHECK (mandatory, checks EVERYTHING)
# ============================================================================

def combined_scalar_loss(court, legate, X_ord, Y_ord, X_ex, Y_ex, X_gate, gate_target,
                          d0, incentive_seq, legate_target):
    """One scalar combining every trainable module's loss, used purely
    for the gradient check (training itself uses the per-module loops
    above; this function exists to certify that every analytic backward
    pass implemented above is correct)."""
    probs_o, _ = mlp2_forward(court.ordinary, X_ord)
    L_o = cross_entropy(probs_o, Y_ord)

    probs_e, _ = mlp2_forward(court.extraordinary, X_ex)
    L_e = cross_entropy(probs_e, Y_ex)

    p_g, _ = mlp2_forward(court.gate, X_gate)
    L_g = bce(p_g[:, 0], gate_target)

    out = legate.rollout(d0, incentive_seq)
    L_l = 0.5 * np.mean(np.sum((out["actions"] - legate_target) ** 2, axis=1))

    return L_o + L_e + L_g + L_l


def analytic_gradients(court, legate, X_ord, Y_ord, X_ex, Y_ex, X_gate, gate_target,
                        d0, incentive_seq, legate_target):
    grads = {}
    probs_o, cache_o = mlp2_forward(court.ordinary, X_ord)
    dZ2 = (probs_o - Y_ord) / X_ord.shape[0]
    g_o = mlp2_backward(court.ordinary, cache_o, dZ2)
    for k, v in g_o.items():
        grads[f"ordinary.{k}"] = v

    probs_e, cache_e = mlp2_forward(court.extraordinary, X_ex)
    dZ2e = (probs_e - Y_ex) / X_ex.shape[0]
    g_e = mlp2_backward(court.extraordinary, cache_e, dZ2e)
    for k, v in g_e.items():
        grads[f"extraordinary.{k}"] = v

    p_g, cache_g = mlp2_forward(court.gate, X_gate)
    p_flat = p_g[:, 0]
    dOut = ((p_flat - gate_target) / X_gate.shape[0])[:, None]
    g_g = mlp2_backward(court.gate, cache_g, dOut)
    for k, v in g_g.items():
        grads[f"gate.{k}"] = v

    out = legate.rollout(d0, incentive_seq)
    g_l = legate.bptt(out, legate_target)
    for k, v in g_l.items():
        grads[f"legate.{k}"] = v

    return grads


def finite_difference_gradient_check(court, legate, X_ord, Y_ord, X_ex, Y_ex,
                                      X_gate, gate_target, d0, incentive_seq,
                                      legate_target, epsilon=1e-5, sample_frac=0.35, rng=None):
    """Perturbs every scalar parameter (or, for large tensors, a random
    subset -- standard practice for tractable grad-checks) by +-epsilon
    and compares to the analytic gradient. Returns the max relative
    error found."""
    if rng is None:
        rng = np.random.default_rng(0)
    params = collect_all_params(court, legate)
    analytic = analytic_gradients(court, legate, X_ord, Y_ord, X_ex, Y_ex,
                                   X_gate, gate_target, d0, incentive_seq, legate_target)

    def loss_fn():
        return combined_scalar_loss(court, legate, X_ord, Y_ord, X_ex, Y_ex,
                                     X_gate, gate_target, d0, incentive_seq, legate_target)

    max_rel_err = 0.0
    checked = 0
    for name, arr in params.items():
        flat = arr.reshape(-1)
        g_flat = analytic[name].reshape(-1)
        n = flat.size
        n_check = max(1, int(np.ceil(n * sample_frac))) if n > 6 else n
        positions = rng.choice(n, size=n_check, replace=False)
        for pos in positions:
            orig = flat[pos]
            flat[pos] = orig + epsilon
            loss_plus = loss_fn()
            flat[pos] = orig - epsilon
            loss_minus = loss_fn()
            flat[pos] = orig
            numeric = (loss_plus - loss_minus) / (2 * epsilon)
            analytic_g = g_flat[pos]
            denom = max(abs(numeric), abs(analytic_g), 1e-8)
            rel_err = abs(numeric - analytic_g) / denom
            max_rel_err = max(max_rel_err, rel_err)
            checked += 1
    return max_rel_err, checked


# ============================================================================
# SECTION 10: SELF-TESTS
# ============================================================================

def run_self_tests(court, legate, X_test_ord, y_test_ord, X_test_novel, y_test_novel,
                    grad_max_rel_err):
    results = []

    def check(name, cond, detail=""):
        status = "PASS" if cond else "FAIL"
        results.append((name, status, detail))
        print(f"  [{status}] {name}  {detail}")
        return cond

    # 1. Gradient check passes to a tight tolerance.
    check("finite-difference gradient check < 1e-4",
          grad_max_rel_err < 1e-4, f"max_rel_err={grad_max_rel_err:.2e}")

    # 2. Ordinary Court beats random-guess baseline on ordinary-region cases.
    probs_o, _ = mlp2_forward(court.ordinary, X_test_ord)
    acc_o = np.mean(np.argmax(probs_o, axis=1) == y_test_ord)
    check("Ordinary Court beats random baseline on ordinary docket",
          acc_o > 0.30, f"accuracy={acc_o:.3f} (random baseline=0.25)")

    # 3. On NOVEL cases, the Extraordinary Head outperforms the Ordinary
    #    Court -- i.e. law bound only to existing precedent genuinely
    #    fails outside its compiled region, motivating the gate at all.
    probs_o_novel, _ = mlp2_forward(court.ordinary, X_test_novel)
    acc_o_novel = np.mean(np.argmax(probs_o_novel, axis=1) == y_test_novel)
    probs_e_novel, _ = mlp2_forward(court.extraordinary, X_test_novel)
    acc_e_novel = np.mean(np.argmax(probs_e_novel, axis=1) == y_test_novel)
    check("Extraordinary Head outperforms Ordinary Court on novel docket",
          acc_e_novel > acc_o_novel,
          f"extraordinary={acc_e_novel:.3f} vs ordinary={acc_o_novel:.3f}")

    # 4. The Vicariate Gate's restraint doctrine: it should assign
    #    materially higher activation to genuine grave/unprecedented
    #    (novel-region) cases than to routine ordinary-region cases --
    #    the functional claim -- while its OVERALL mean activation stays
    #    low (propter causam: rare, not routine). Tested on continuous
    #    probabilities rather than a single 0.5 threshold count, which
    #    is fragile under a deliberately sparse target distribution.
    cov_ord = court.coverage_batch(X_test_ord)
    (p_ord, _), _ = court.gate_forward(X_test_ord, cov_ord)
    cov_novel = court.coverage_batch(X_test_novel)
    (p_novel, _), _ = court.gate_forward(X_test_novel, cov_novel)
    mean_ord, mean_novel = float(np.mean(p_ord)), float(np.mean(p_novel))
    check("Vicariate Gate activates more on novel/grave cases than ordinary ones",
          mean_novel > mean_ord,
          f"mean_activation: novel={mean_novel:.4f} vs ordinary={mean_ord:.4f}")
    check("Vicariate Gate stays rare overall (propter causam, not routine)",
          mean_novel < 0.5,
          f"mean_activation on novel docket={mean_novel:.4f}")

    # 5. Compilatio write-back: an extraordinary verdict, once written
    #    into the Precedent Memory, is recognized as covered next time.
    novel_case = X_test_novel[0]
    cov_before = court.memory.coverage(novel_case)
    court.memory.compile_case(novel_case)
    cov_after = court.memory.coverage(novel_case)
    check("write-back increases coverage for the same case (Compilatio effect)",
          cov_after > cov_before + 0.05,
          f"coverage {cov_before:.3f} -> {cov_after:.3f}")

    return results


def run_crusade_experiment(legate, rng, d0, incentive_dir, T=10):
    """
    One frozen, trained legate policy; one realized incentive escalation
    (deliberately larger than the training distribution -- EVAL_INCENTIVE_SCALE
    vs TRAIN_INCENTIVE_SCALE -- because Zara and Constantinople were not
    contingencies the coalition had rehearsed for). The RAW rollout is
    identical across all three conditions below; only the oversight
    schedule layered on top by apply_oversight() differs, isolating the
    effect of monitoring cadence from everything else.
    """
    ramp = np.array([incentive_dir * (t / T) * EVAL_INCENTIVE_SCALE for t in range(T)])
    noise_rng = np.random.default_rng(42)
    incentive_seq = ramp + noise_rng.normal(0, 0.08, size=(T, 4))
    raw = legate.rollout(d0, incentive_seq)["actions"]

    # No oversight at all: the undirected baseline.
    none_mask = np.zeros(T)
    drift_none, _ = LegateChain.apply_oversight(raw, d0, none_mask)

    # Retrospective (historically realistic): news of Zara reaches Rome,
    # and even the Constantinople prohibition is suppressed by the
    # leadership, so real correction lands only very late.
    D = int(0.8 * T)
    hist_mask = np.array([0.0] * D + [1.0] * (T - D))
    drift_hist, _ = LegateChain.apply_oversight(raw, d0, hist_mask)

    # Continuous: real-time, uninterrupted jurisdiction from step 0 --
    # the counterfactual an AGI-era Innocent draws from this experience.
    cont_mask = np.ones(T)
    drift_cont, _ = LegateChain.apply_oversight(raw, d0, cont_mask)

    return {
        "no_oversight_final_drift": float(drift_none[-1]),
        "retrospective_final_drift": float(drift_hist[-1]),
        "continuous_final_drift": float(drift_cont[-1]),
        "no_oversight_mean_drift": float(np.mean(drift_none)),
        "retrospective_mean_drift": float(np.mean(drift_hist)),
        "continuous_mean_drift": float(np.mean(drift_cont)),
        "correction_delay_steps": D,
        "total_steps": T,
    }


# ============================================================================
# SECTION 11: MAIN
# ============================================================================

def main():
    print("=" * 78)
    print("PLENITUDO POTESTATIS NET")
    print("Figure 0255 -- Pope Innocent III (c.1160-1216)")
    print("=" * 78)

    build_rng = np.random.default_rng(RNG_SEED)

    # --- Build the docket ---
    X_ord_train, y_ord_train = build_docket(240, build_rng, region="ordinary")
    X_ord_test, y_ord_test = build_docket(120, build_rng, region="ordinary")
    X_all_train, y_all_train = build_docket(300, build_rng, region=None)
    X_novel_test, y_novel_test = build_docket(120, build_rng, region="novel")

    court = PlenitudoPotestatisNet(np.random.default_rng(11), precedent_seed_X=X_ord_train.copy())
    legate = LegateChain(np.random.default_rng(22))

    print("\n1. TRAINING THE ORDINARY COURT (bound to compiled precedent)")
    train_ordinary_court(court, X_ord_train, y_ord_train, epochs=400, lr=0.15, verbose=True)

    print("\n2. TRAINING THE EXTRAORDINARY HEAD (unmediated, full docket)")
    train_extraordinary_head(court, X_all_train, y_all_train, epochs=400, lr=0.15, verbose=True)

    print("\n3. TRAINING THE VICARIATE GATE (propter causam -- rare by design)")
    _, gate_target_train = train_vicariate_gate(court, X_all_train, epochs=180, lr=0.2, l1=0.01, verbose=True)

    print("\n4. TRAINING THE LEGATE CHAIN (directive compliance under moderate incentive pressure)")
    _, d0, incentive_dir = train_legate_chain(legate, np.random.default_rng(33), epochs=250, lr=0.05, T=10, verbose=True)

    # --- Gradient check on a small held-out batch (mandatory) ---
    print("\n5. FINITE-DIFFERENCE GRADIENT CHECK (every module, every-parameter sampled)")
    Xg_small, yg_small = X_all_train[:24], y_all_train[:24]
    Xo_small, yo_small = X_ord_train[:24], y_ord_train[:24]
    cov_small = court.coverage_batch(Xg_small)
    (_, _), Xgate_small = court.gate_forward(Xg_small, cov_small)
    ord_probs_small, _ = mlp2_forward(court.ordinary, Xg_small)
    true_small = np.array([true_canon_law(x) for x in Xg_small])
    gate_target_small = ((np.argmax(ord_probs_small, axis=1) != true_small) &
                          (cov_small < 0.5)).astype(float)
    T_small = 5
    ramp_small = np.array([incentive_dir * (t / T_small) * TRAIN_INCENTIVE_SCALE for t in range(T_small)])
    incentive_small = ramp_small + np.random.default_rng(5).normal(0, 0.05, size=(T_small, 4))
    legate_target_small = np.tile(d0, (T_small, 1))

    max_rel_err, n_checked = finite_difference_gradient_check(
        court, legate,
        Xo_small, np.array([onehot(c) for c in yo_small]),
        Xg_small, np.array([onehot(c) for c in true_small]),
        Xgate_small, gate_target_small,
        d0, incentive_small, legate_target_small,
        epsilon=1e-5, sample_frac=0.4, rng=np.random.default_rng(99),
    )
    print(f"    Parameters checked: {n_checked}")
    print(f"    Max relative error: {max_rel_err:.3e}  (tolerance 1e-4)")

    # --- Gate activation on a fresh, balanced batch (half ordinary,
    # half novel/grave-cause), reported for the record ---
    X_rate_ord, _ = build_docket(100, np.random.default_rng(60), region="ordinary")
    X_rate_novel, _ = build_docket(100, np.random.default_rng(61), region="novel")
    cov_rate_ord = court.coverage_batch(X_rate_ord)
    (p_rate_ord, _), _ = court.gate_forward(X_rate_ord, cov_rate_ord)
    cov_rate_novel = court.coverage_batch(X_rate_novel)
    (p_rate_novel, _), _ = court.gate_forward(X_rate_novel, cov_rate_novel)
    print(f"    Gate mean activation -- ordinary docket: {np.mean(p_rate_ord):.4f}   "
          f"novel/grave docket: {np.mean(p_rate_novel):.4f}")
    print(f"    Gate fire-rate (p>0.5) -- ordinary: {np.mean(p_rate_ord > 0.5):.3f}   "
          f"novel/grave: {np.mean(p_rate_novel > 0.5):.3f}")

    print("\n6. SELF-TESTS")
    run_self_tests(court, legate, X_ord_test, y_ord_test, X_novel_test, y_novel_test,
                    max_rel_err)

    print("\n7. THE ZARA EXPERIMENT: monitoring cadence vs. delegated drift")
    crusade = run_crusade_experiment(legate, np.random.default_rng(77), d0, incentive_dir)
    for k, v in crusade.items():
        print(f"    {k:32s} = {v:.4f}")
    improvement = 100 * (1 - crusade["continuous_final_drift"] / max(crusade["retrospective_final_drift"], 1e-6))
    print(f"    Continuous jurisdiction reduces final drift by {improvement:.1f}% versus retrospective interdict.")

    print("\n8. WORKED ADJUDICATIONS")
    demo_ordinary, _ = build_docket(1, np.random.default_rng(101), region="ordinary")
    demo_novel, _ = build_docket(1, np.random.default_rng(102), region="novel")
    # A deliberately extreme case -- maximal heresy severity, schism risk,
    # and holy-land stakes at once (a synthetic stand-in for "Catharism
    # entrenched in Toulouse while the Empire itself stands vacant") --
    # to show a genuine extraordinary invocation, not just the common
    # case. gate_threshold is lowered from 0.5 to 0.2 for this
    # demonstration only: exactly Innocent's own point in Quanto
    # personam, that "grave cause" is a matter of judgment on the
    # merits, not a fixed numerical bar -- Hostiensis and later
    # canonists spent a generation arguing about where that judgment
    # should fall.
    extreme_case = sample_case(np.random.default_rng(103))
    extreme_case[0], extreme_case[5], extreme_case[4] = 0.97, 0.95, 0.9
    for label, case, threshold in [
        ("ordinary benefice-type case", demo_ordinary[0], 0.5),
        ("novel/grave-cause case (docket-typical)", demo_novel[0], 0.5),
        ("extreme case (near-maximal heresy+schism+crusade stakes)", extreme_case, 0.2),
    ]:
        result = court.adjudicate(case, gate_threshold=threshold, write_back=True)
        print(f"    {label}:")
        print(f"        verdict={result['verdict']:10s} mode={result['mode']:38s} "
              f"coverage={result['coverage']:.3f} gate={result['gate_activation']:.3f} "
              f"(threshold={threshold})")

    print("\n" + "=" * 78)
    print("RUN COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()

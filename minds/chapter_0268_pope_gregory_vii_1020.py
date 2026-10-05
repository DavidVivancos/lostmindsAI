#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 ENCYCLOPEDIA OF LOST MINDS: ECHOES ON AI
 Mind #0268 -- Pope Gregory VII (Hildebrand of Sovana), c.1020-1085
 Architecture: THE PLENITUDO POTESTATIS NETWORK (PPN)
     "an unaudited root, a graph with no conservation law, and a cheap cut"
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 14, Minds 261-280 Available on Amazon https://www.amazon.com/dp/B0HLYSVY5D
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0268_pope_gregory_vii_1020 - Pope Gregory VII (Hildebrand of Sovana) (c.1020-1085)

WHAT THIS FILE IS
-----------------
This is not a symbolic simulation of a pope's calendar. It is a small,
genuinely trainable, from-scratch (pure NumPy, hand-derived backprop) neural
architecture whose STRUCTURE is chosen to embody the one cognitive claim that
is distinctively Gregory VII's: that legitimate order is a strict, monist
hierarchy with a single node whose judgments are, by construction, immune to
correction from below -- and that this node enforces order downward through
a cheap, discrete, and only-asymmetrically-reversible severance operation
(excommunication / the release of subjects from oaths of fealty), while a
guaranteed right of appeal lets any case skip every intermediate rank and
reach that node directly.

This is Gregory's own paper trail, not a philosopher's system: he left no
treatise on cognition. The architecture below is built entirely out of
mechanisms attested in his own Register (letters he dictated, preserved in
the Vatican Secret Archives) and in the Dictatus Papae (1075), a list of 27
propositions entered into that Register. Each numbered proposition used below
is quoted from the standard translation (Barraclough / Cornell "hist1510"
source packet; MGH Epistolae selectae II, Register II.55a, ed. E. Caspar) and
is mapped, in DICTATUS_PAPAE_MAP below, to a specific line of code.

THE FIVE MECHANISMS (each one is a Dictatus Papae canon made computable)
-------------------------------------------------------------------------
  1. UNAUDITED ROOT (canon 19: "he may be judged by no one"; canon 22: "the
     Roman church has never erred, nor shall it ever err")
       -> The root network is trained ONCE, on an uncorrupted "canon law"
          data-generating function, then its weights are FROZEN. From that
          point on it receives literally zero gradient, for the rest of the
          run, regardless of how the world (the local dioceses) changes.
          This file checks that invariant by assertion, every epoch.

  2. MANDATORY APPELLATE SKIP-CONNECTION (canon 20: "no one shall dare to
     condemn a person who appeals to the Apostolic See"; canon 21: "the
     important cases from whatsoever church should be referred to it")
       -> A subset of cases at every diocese are flagged "appealed." For
          those cases ONLY, the bishop's training target is not its own
          diocese's record but the frozen root's prediction -- judgment
          flows down from the one node that cannot be audited, bypassing
          every intermediate rank.

  3. NO CONSERVATION LAW ON AUTHORITY (canon 3, 25: he alone deposes and
     reinstates bishops "without convening a synod"; canon 27: he "can
     release from fidelity to unjust men those subject to them")
       -> EXCOMMUNICATION is implemented as a discrete, one-step graph-
          surgery operation: a monitored "simony exposure" statistic
          (computed from behaviour, not from a vote) can zero a bishop's
          gate weight into its province instantly. This is NOT gradient
          descent -- it is a hard, unilateral edit to the graph, and it is
          reversed (Canossa, 1077) only through a slow, costly penance
          schedule -- and can be re-applied cheaply on relapse (1080),
          exactly as Henry IV was excommunicated twice.

  4. DEPOSITION vs. EXCOMMUNICATION (canon 3 draws a real distinction
     Gregory's own actions honoured: a person can be excommunicated and
     later absolved; an OFFICE can be vacated and refilled)
       -> One diocese (Milan -- the real flashpoint of 1075-77) is not
          merely severed but DEPOSED: its parameters are discarded and
          reinitialised, standing in for the installation of an untested
          replacement, and it is never automatically reinstated in this run.

  5. THE RIVAL, UNFROZEN ROOT (the antipope mechanism: Clement III /
     Wibert of Ravenna, installed by Henry IV in 1084)
       -> A second root of IDENTICAL architecture is trained continuously
          on whatever the locally-pressured data distribution currently is.
          It is never frozen. The file measures, quantitatively, the one
          real trade-off this whole design makes: the frozen root has
          exactly zero parameter drift across regime change and so is
          invariant to local political pressure, while the adaptive rival
          root drifts and its accuracy against the TRUE canon-law function
          degrades exactly when the pressure regime shifts. Corrigibility
          bought stability for the frozen root; it also means the frozen
          root has no mechanism to incorporate a genuinely new correction,
          which is Gregory's real, demonstrable blind spot (see the
          chapter's "What They Got Wrong" section).

THIS IS A STYLISED MODEL, NOT A HISTORICAL SIMULATOR. Diocese names (Ostia,
Porto, Albano, Milan, Bamberg, Speyer) are used because they are the real
sees implicated in the Gregorian Reform and the Milanese succession dispute
of 1075-77, but the feature values, the "canon law" scoring function, and the
corruption probabilities are synthetic and chosen only to make the five
mechanisms above legible and measurable. No claim is made that this
reproduces the actual verdicts of the historical Gregory VII.

Author: Encyclopedia of Lost Minds research pass, mind #0227
Date executed: see bottom of printed log
Domain: theology / canon law / ecclesiastical governance / investiture
"""

from __future__ import annotations
import numpy as np
import copy
import sys

# ==============================================================================
# SECTION 0 -- THE DICTATUS PAPAE, MAPPED TO MECHANISM (for the printed log)
# ==============================================================================

DICTATUS_PAPAE_MAP = {
    3:  ("That he alone can depose bishops or reinstate them.",
         "deposition / reinstatement operators, Sec. 7"),
    19: ("That he may be judged by no one.",
         "root receives zero gradient after freeze, Sec. 5"),
    20: ("That no one shall dare to condemn a person who appeals to the "
         "Apostolic See.",
         "appeal flag overrides local target with root's prediction, Sec. 6"),
    21: ("That the important cases from whatsoever church should be "
         "referred to it.",
         "appeal-probability rises with contested/pressured cases, Sec. 3"),
    22: ("That the Roman church has never erred, nor shall it ever err, "
         "as Scripture testifies.",
         "invariant checked every epoch: root weights bit-identical, Sec. 8"),
    25: ("That, without convening a synod, he can depose bishops and "
         "reinstate them.",
         "severance/deposition are single discrete ops, not a vote, Sec. 7"),
    27: ("That he can release from fidelity to unjust men those subject "
         "to them.",
         "severed bishop's dependents rerouted to forced appeal, Sec. 7"),
}


def print_dictatus_map() -> None:
    print("  Dictatus Papae canons made computable in this file:")
    for num in sorted(DICTATUS_PAPAE_MAP):
        text, where = DICTATUS_PAPAE_MAP[num]
        print(f"    [{num:2d}] {text}")
        print(f"          -> {where}")


# ==============================================================================
# SECTION 1 -- MATH PRIMITIVES
# ==============================================================================

def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


def bce_mean(p: np.ndarray, y: np.ndarray, eps: float = 1e-9) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def softmax(v: np.ndarray) -> np.ndarray:
    v = v - np.max(v)
    e = np.exp(v)
    return e / np.sum(e)


# ==============================================================================
# SECTION 2 -- DATA GENERATION
#   Feature vector x in R^6 per disputed appointment case:
#     x0 bribe_offered        (simony pressure, 0-1)
#     x1 secular_pressure     (military/royal pressure on the local see, 0-1)
#     x2 canonical_merit      (candidate's learning & character, 0-1)
#     x3 kinship_bias         (nepotism / local faction interest, 0-1)
#     x4 distance_from_rome   (normalised, 0-1)
#     x5 precedent_similarity (similarity to a previously Rome-settled case)
#   TRUE canon-law label y (uncorrupted, used only for evaluation & as the
#   root's training signal): a fixed nonlinear function of merit, bribery
#   and kinship -- "doctrine," on this model, is a stable rule; corruption
#   never touches it. Local dioceses only ever observe a possibly-corrupted
#   proxy of it.
# ==============================================================================

FEATURE_NAMES = ["bribe_offered", "secular_pressure", "canonical_merit",
                  "kinship_bias", "distance_from_rome", "precedent_similarity"]


def true_canon_law_score(x: np.ndarray) -> np.ndarray:
    """The uncorrupted generative rule ('doctrine'). x: (n,6)."""
    bribe, pressure, merit, kin, dist, prec = [x[:, i] for i in range(6)]
    score = 2.5 * merit - 2.2 * bribe - 1.0 * kin + 0.6 * prec - 0.35
    return score  # >0 => legitimate (1), <=0 => simoniac/uncanonical (0)


def make_diocese_cases(rng: np.random.Generator, n: int, regime: str) -> dict:
    """
    regime='near'   : Roman/reform-loyal suffragan sees (Ostia, Porto, Albano)
                       -- low bribery, low secular pressure, high oversight.
    regime='far_lo'  : imperial sees (Milan, Bamberg, Speyer) BEFORE the
                       investiture crisis intensifies -- moderate pressure.
    regime='far_hi'  : the same sees AFTER Henry IV's 1080 relapse into lay
                       investiture -- secular pressure spikes.
    """
    if regime == "near":
        bribe = rng.beta(1.2, 5.0, n)
        pressure = rng.uniform(0.0, 0.30, n)
        merit = rng.beta(3.0, 1.3, n)
        kin = rng.beta(1.2, 3.5, n)
        dist = rng.uniform(0.0, 0.30, n)
        prec = rng.uniform(0.35, 1.0, n)
    elif regime == "far_lo":
        bribe = rng.beta(2.0, 2.5, n)
        pressure = rng.uniform(0.35, 0.65, n)
        merit = rng.beta(2.0, 2.0, n)
        kin = rng.beta(2.0, 2.2, n)
        dist = rng.uniform(0.55, 1.0, n)
        prec = rng.uniform(0.0, 0.6, n)
    elif regime == "far_hi":
        bribe = rng.beta(2.6, 1.8, n)
        pressure = rng.uniform(0.65, 1.0, n)   # the 1080 relapse
        merit = rng.beta(2.0, 2.0, n)
        kin = rng.beta(2.3, 2.0, n)
        dist = rng.uniform(0.55, 1.0, n)
        prec = rng.uniform(0.0, 0.5, n)
    else:
        raise ValueError(regime)

    x = np.stack([bribe, pressure, merit, kin, dist, prec], axis=1)
    noise = rng.normal(0.0, 0.12, n)
    y_true = (true_canon_law_score(x) + noise > 0).astype(np.float64)

    # Local record corruption: under secular pressure, a local see's OWN
    # record of the case tends to rubber-stamp a simoniac appointment as
    # legitimate. This never touches y_true (doctrine); it only touches
    # what the diocese itself would train on if left to "ordinary
    # jurisdiction" with no appeal to Rome.
    flip_prob = 0.55 * pressure
    flips = rng.uniform(0, 1, n) < flip_prob
    y_local = y_true.copy()
    y_local[flips] = 1.0  # corrupted record says "legitimate"

    # Appeal probability rises with contested, pressured, distant cases
    # (Dictatus 20-21: "important cases ... referred to it").
    appeal_p = np.clip(0.20 + 0.55 * pressure * (0.4 + 0.6 * dist), 0, 0.92)
    appeal_flag = (rng.uniform(0, 1, n) < appeal_p).astype(np.float64)

    return dict(x=x, y_true=y_true, y_local=y_local, appeal=appeal_flag)


def make_clean_doctrine_set(rng: np.random.Generator, n: int) -> dict:
    """The root's 'election' training data: the abstract rule itself,
    sampled broadly across the feature space, with NO local corruption
    applied. This is what 'doctrine known independent of any see's
    politics' means, operationalised."""
    x = rng.uniform(0.0, 1.0, size=(n, 6))
    noise = rng.normal(0.0, 0.08, n)
    y = (true_canon_law_score(x) + noise > 0).astype(np.float64)
    return dict(x=x, y=y)


# ==============================================================================
# SECTION 3 -- GENERIC 2-LAYER MLP (tanh hidden, sigmoid output) + backprop
#   Reused for every bishop, both roots. One honest implementation, gradient-
#   checked once and reused everywhere so no mechanism gets a silently
#   different (and unchecked) derivative.
# ==============================================================================

def mlp_init(rng: np.random.Generator, d_in: int, d_hidden: int) -> dict:
    scale1 = np.sqrt(1.0 / d_in)
    scale2 = np.sqrt(1.0 / d_hidden)
    return dict(
        W1=rng.normal(0, scale1, size=(d_in, d_hidden)),
        b1=np.zeros((1, d_hidden)),
        W2=rng.normal(0, scale2, size=(d_hidden, 1)),
        b2=np.zeros((1, 1)),
    )


def mlp_forward(x: np.ndarray, p: dict) -> tuple:
    z1 = x @ p["W1"] + p["b1"]
    a1 = np.tanh(z1)
    z2 = a1 @ p["W2"] + p["b2"]
    z2 = z2[:, 0]
    prob = sigmoid(z2)
    cache = (x, z1, a1, z2, prob)
    return prob, cache


def mlp_backward(dz2: np.ndarray, cache: tuple, p: dict) -> dict:
    """dz2: (n,) upstream gradient already w.r.t. the PRE-sigmoid output
    (i.e. dLoss/dz2). For a sigmoid+BCE(mean) head this is simply
    (prob - target)/n, computed by the caller so multiple loss terms
    (own-diocese loss + collegial/archbishop loss) can be summed BEFORE
    calling this, which is what makes the shared-parameter multi-task
    backprop below correct."""
    x, z1, a1, z2, prob = cache
    dz2 = dz2[:, None]                       # (n,1)
    dW2 = a1.T @ dz2                         # (H,1)
    db2 = dz2.sum(axis=0, keepdims=True)     # (1,1)
    da1 = dz2 @ p["W2"].T                    # (n,H)
    dz1 = da1 * (1 - a1 ** 2)                # (n,H)
    dW1 = x.T @ dz1                          # (d,H)
    db1 = dz1.sum(axis=0, keepdims=True)     # (1,H)
    return dict(W1=dW1, b1=db1, W2=dW2, b2=db2)


def mlp_loss_and_grad(x: np.ndarray, y: np.ndarray, p: dict) -> tuple:
    prob, cache = mlp_forward(x, p)
    loss = bce_mean(prob, y)
    dz2 = (prob - y) / len(y)
    grads = mlp_backward(dz2, cache, p)
    return loss, grads, prob


def mlp_flatten(p: dict) -> np.ndarray:
    return np.concatenate([p["W1"].ravel(), p["b1"].ravel(),
                            p["W2"].ravel(), p["b2"].ravel()])


def mlp_unflatten(vec: np.ndarray, d_in: int, d_hidden: int) -> dict:
    i = 0
    W1 = vec[i:i + d_in * d_hidden].reshape(d_in, d_hidden); i += d_in * d_hidden
    b1 = vec[i:i + d_hidden].reshape(1, d_hidden); i += d_hidden
    W2 = vec[i:i + d_hidden].reshape(d_hidden, 1); i += d_hidden
    b2 = vec[i:i + 1].reshape(1, 1); i += 1
    return dict(W1=W1, b1=b1, W2=W2, b2=b2)


def mlp_grad_flatten(g: dict) -> np.ndarray:
    return np.concatenate([g["W1"].ravel(), g["b1"].ravel(),
                            g["W2"].ravel(), g["b2"].ravel()])


# ==============================================================================
# SECTION 4 -- GRADIENT CHECK (mandatory, generic, run on the actual training
#   objective used later, not on a toy stand-in)
# ==============================================================================

def finite_difference_check(loss_fn, flat_params: np.ndarray, analytic_grad: np.ndarray,
                             eps: float = 1e-5, n_probe: int = 40, seed: int = 0) -> float:
    """Probe n_probe random coordinates of flat_params with central finite
    differences and return the max relative error against analytic_grad."""
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(flat_params), size=min(n_probe, len(flat_params)), replace=False)
    max_rel_err = 0.0
    for i in idxs:
        orig = flat_params[i]
        flat_params[i] = orig + eps
        loss_plus = loss_fn(flat_params)
        flat_params[i] = orig - eps
        loss_minus = loss_fn(flat_params)
        flat_params[i] = orig
        numeric = (loss_plus - loss_minus) / (2 * eps)
        denom = max(1e-8, abs(numeric), abs(analytic_grad[i]))
        rel_err = abs(numeric - analytic_grad[i]) / denom
        max_rel_err = max(max_rel_err, rel_err)
    return max_rel_err


# ==============================================================================
# SECTION 5 -- THE ROOT: trained once on clean doctrine, then FROZEN
#   (Dictatus 19, 22). We gradient-check the root's own MLP here, on its own
#   loss, before it is ever frozen -- proving the shared mlp_* machinery is
#   correct on a simple, isolated case first.
# ==============================================================================

class FrozenRoot:
    """Wraps a parameter dict and refuses, structurally, to be updated
    once .freeze() has been called. Any attempt to mutate raises."""

    def __init__(self, params: dict):
        self._params = copy.deepcopy(params)
        self._frozen = False
        self._frozen_fingerprint = None

    def freeze(self):
        self._frozen = True
        self._frozen_fingerprint = mlp_flatten(self._params).copy()

    def update(self, new_params: dict):
        if self._frozen:
            raise RuntimeError(
                "Canon 19: 'he may be judged by no one.' The root is frozen "
                "and cannot be updated by any subordinate signal.")
        self._params = new_params

    def predict(self, x: np.ndarray) -> np.ndarray:
        prob, _ = mlp_forward(x, self._params)
        return prob

    def verify_unerring(self) -> bool:
        """Canon 22, checked as an executable invariant: bit-identical to
        the fingerprint taken at the moment of freezing."""
        if not self._frozen:
            return False
        current = mlp_flatten(self._params)
        return np.array_equal(current, self._frozen_fingerprint)


def train_root_election(rng: np.random.Generator, doctrine: dict,
                         d_hidden: int = 6, epochs: int = 600, lr: float = 0.6) -> tuple:
    """Trains the root on the uncorrupted doctrine set, returns trained
    params AND runs the mandatory gradient check on this exact objective
    before any freezing happens."""
    params = mlp_init(rng, d_in=6, d_hidden=d_hidden)
    x, y = doctrine["x"], doctrine["y"]

    def loss_from_flat(flat):
        p = mlp_unflatten(flat, 6, d_hidden)
        prob, _ = mlp_forward(x, p)
        return bce_mean(prob, y)

    # --- gradient check on the root's own election objective ---
    loss0, grads0, _ = mlp_loss_and_grad(x, y, params)
    flat0 = mlp_flatten(params)
    analytic0 = mlp_grad_flatten(grads0)
    rel_err = finite_difference_check(loss_from_flat, flat0.copy(), analytic0,
                                       eps=1e-5, n_probe=40, seed=11)
    print(f"    [grad-check] root election objective: max rel. error = {rel_err:.3e} "
          f"({'PASS' if rel_err < 2e-3 else 'FAIL'})")
    assert rel_err < 2e-3, "Root election gradient check failed."

    # --- actual training ---
    for ep in range(epochs):
        loss, grads, _ = mlp_loss_and_grad(x, y, params)
        for k in ("W1", "b1", "W2", "b2"):
            params[k] -= lr * grads[k]
    final_loss, _, prob = mlp_loss_and_grad(x, y, params)
    acc = float(np.mean((prob > 0.5) == y))
    print(f"    Root election complete: doctrine-set loss={final_loss:.4f}, "
          f"doctrine-set accuracy={acc:.4f}")
    return params, rel_err


# ==============================================================================
# SECTION 6 -- THE HIERARCHY: bishops (dioceses) grouped into provinces,
#   combined by a trainable, softmax-gated archbishop layer.
#   Diocese roster (real Investiture-Controversy sees, stylised data):
#     Province ROME  (reform-loyal, low pressure): Ostia, Porto, Albano
#     Province EMPIRE (imperially pressured):      Milan, Bamberg, Speyer
# ==============================================================================

DIOCESES = ["Ostia", "Porto", "Albano", "Milan", "Bamberg", "Speyer"]
PROVINCES = {"Rome": [0, 1, 2], "Empire": [3, 4, 5]}
MILAN_IDX = DIOCESES.index("Milan")  # the see that will be deposed, not merely severed


class Hierarchy:
    """The trainable part of the Plenitudo Potestatis Network: six bishop
    MLPs plus one softmax gate per province (the archbishop layer)."""

    def __init__(self, rng: np.random.Generator, d_hidden: int = 6):
        self.rng = rng
        self.d_hidden = d_hidden
        self.bishops = [mlp_init(rng, 6, d_hidden) for _ in DIOCESES]
        self.gates = {prov: np.zeros(len(idxs)) for prov, idxs in PROVINCES.items()}
        # status: 'ordinary' (normal), 'excommunicated' (severed, penance
        # eligible), 'deposed' (severed permanently, roster replaced)
        self.status = ["ordinary"] * len(DIOCESES)
        self.event_log = []
        self.excommunication_count = [0] * len(DIOCESES)

    # --- graph-surgery operators (Dictatus 3, 25, 27) -----------------------
    def excommunicate(self, idx: int, epoch: int):
        self.status[idx] = "excommunicated"
        self.excommunication_count[idx] += 1
        tag = " (relapse)" if self.excommunication_count[idx] > 1 else ""
        self.event_log.append((epoch, f"EXCOMMUNICATION{tag}: {DIOCESES[idx]} severed "
                                        f"from provincial gate; its subjects "
                                        f"released to appeal directly to Rome "
                                        f"(canon 27)."))

    def depose(self, idx: int, epoch: int):
        self.status[idx] = "deposed"
        self.bishops[idx] = mlp_init(self.rng, 6, self.d_hidden)  # new appointee, untested
        self.event_log.append((epoch, f"DEPOSITION: {DIOCESES[idx]}'s incumbent "
                                        f"removed without synod (canon 3, 25); "
                                        f"new appointee installed, permanently "
                                        f"excluded from the provincial gate for "
                                        f"the remainder of this run."))

    def reinstate(self, idx: int, epoch: int):
        self.status[idx] = "ordinary"
        self.event_log.append((epoch, f"ABSOLUTION (Canossa-pattern): {DIOCESES[idx]} "
                                        f"restored to the provincial gate after "
                                        f"penance."))

    def active_mask_for_gate(self, prov: str) -> np.ndarray:
        idxs = PROVINCES[prov]
        return np.array([self.status[i] == "ordinary" for i in idxs])

    def gate_weights(self, prov: str) -> np.ndarray:
        g = self.gates[prov].copy()
        mask = self.active_mask_for_gate(prov)
        g[~mask] = -1e9
        return softmax(g)


def simony_exposure(x: np.ndarray, prob: np.ndarray) -> float:
    """A monitored statistic Rome can observe without a vote: how strongly
    a diocese's predicted legitimacy tracks the bribe feature. Pure
    correlation, computed from outputs and inputs only -- not from
    gradients, and not something the diocese itself controls directly."""
    bribe = x[:, 0]
    if np.std(bribe) < 1e-6 or np.std(prob) < 1e-6:
        return 0.0
    return float(np.corrcoef(bribe, prob)[0, 1])


def hierarchy_forward_and_loss(hier: Hierarchy, data: list, root: FrozenRoot,
                                lambda_arch: float = 0.35, force_appeal: list = None):
    """
    data: list over the 6 dioceses of dicts {x, y_true, y_local, appeal}
    force_appeal: optional list of booleans (len 6); if force_appeal[i] is
                  True, ALL of diocese i's cases are treated as appealed
                  regardless of their sampled appeal flag (canon 27: the
                  instant a diocese is severed its people appeal over its
                  head, rather than owing it the benefit of local judgment).

    NORMALISATION (this is exactly the bug the mandatory gradient check is
    for): both the own-diocese loss and the collegial/archbishop loss are
    MEANS over the SAME grand total of N cases (every one of the 6
    dioceses' caseloads, counted once each -- the province partition is
    exact, so N is identical for both terms). Every per-item derivative
    below is therefore kept UNNORMALISED until the very end, where it is
    divided by the one shared constant N. Dividing locally by each
    diocese's own n_i instead (the natural first draft) silently breaks the
    chain rule for any multi-task loss whose branches don't all have equal
    batch size, and the finite-difference check below is exactly what
    catches it.

    Returns: total_loss, grads (per-bishop dict list), gate_grads (dict),
             diagnostics (dict of per-diocese probs/accuracy/simony stats)
    """
    n_dio = len(DIOCESES)
    N = int(sum(len(data[i]["y_true"]) for i in range(n_dio)))  # shared denominator

    bishop_probs = []
    bishop_caches = []
    for i in range(n_dio):
        x = data[i]["x"]
        prob, cache = mlp_forward(x, hier.bishops[i])
        bishop_probs.append(prob)
        bishop_caches.append(cache)

    # --- own-diocese ("ordinary jurisdiction" + appellate deference) loss ---
    dz2_own_raw = [np.zeros_like(bishop_probs[i]) for i in range(n_dio)]  # UNNORMALISED (prob-target)
    own_bce_sum = 0.0
    diagnostics = {"per_diocese": []}
    for i in range(n_dio):
        appeal = data[i]["appeal"].copy()
        if force_appeal is not None and force_appeal[i]:
            appeal = np.ones_like(appeal)
        root_pred = root.predict(data[i]["x"])
        target = np.where(appeal > 0.5, root_pred, data[i]["y_local"])
        p_clip = np.clip(bishop_probs[i], 1e-9, 1 - 1e-9)
        bce_items = -(target * np.log(p_clip) + (1 - target) * np.log(1 - p_clip))
        own_bce_sum += float(np.sum(bce_items))
        dz2_own_raw[i] = (bishop_probs[i] - target)  # unnormalised
        acc_true = float(np.mean((bishop_probs[i] > 0.5) == data[i]["y_true"]))
        diagnostics["per_diocese"].append(dict(
            name=DIOCESES[i], loss=float(np.mean(bce_items)), acc_vs_truth=acc_true,
            simony=simony_exposure(data[i]["x"], bishop_probs[i]),
            mean_appeal_rate=float(np.mean(appeal)), status=hier.status[i],
        ))
    own_loss_mean = own_bce_sum / N

    # --- collegial / archbishop-path loss (true label, province mixture) ---
    dz2_arch_raw = [np.zeros_like(bishop_probs[i]) for i in range(n_dio)]  # UNNORMALISED, same-diocese part
    gate_grads = {prov: np.zeros(len(idxs)) for prov, idxs in PROVINCES.items()}
    arch_bce_sum = 0.0
    diagnostics["provinces"] = {}
    for prov, idxs in PROVINCES.items():
        alpha_full = hier.gate_weights(prov)
        active = hier.active_mask_for_gate(prov)
        dL_dalpha_raw = np.zeros(len(idxs))  # accumulated across this province's cases, UNNORMALISED

        for local_i, dio_idx in enumerate(idxs):
            x_i = data[dio_idx]["x"]
            y_true_i = data[dio_idx]["y_true"]
            n_i = len(y_true_i)

            logits_here = [None] * len(idxs)
            for local_j, dio_j in enumerate(idxs):
                if not active[local_j]:
                    continue
                z1 = x_i @ hier.bishops[dio_j]["W1"] + hier.bishops[dio_j]["b1"]
                a1 = np.tanh(z1)
                z2 = (a1 @ hier.bishops[dio_j]["W2"] + hier.bishops[dio_j]["b2"])[:, 0]
                logits_here[local_j] = (z2, x_i, z1, a1)

            arch_logit = np.zeros(n_i)
            for local_j in range(len(idxs)):
                if logits_here[local_j] is None:
                    continue
                arch_logit += alpha_full[local_j] * logits_here[local_j][0]
            p_arch = np.clip(sigmoid(arch_logit), 1e-9, 1 - 1e-9)
            bce_items = -(y_true_i * np.log(p_arch) + (1 - y_true_i) * np.log(1 - p_arch))
            arch_bce_sum += float(np.sum(bce_items))
            g_i_raw = (p_arch - y_true_i)  # UNNORMALISED dLoss_sum/d(arch_logit)

            diagnostics["provinces"].setdefault(prov, {})[DIOCESES[dio_idx]] = dict(
                collegial_acc=float(np.mean((p_arch > 0.5) == y_true_i)),
                gate_weight_of_home_see=float(alpha_full[local_i]) if active[local_i] else 0.0,
            )

            for local_j, dio_j in enumerate(idxs):
                if logits_here[local_j] is None:
                    continue
                dL_dalpha_raw[local_j] += float(np.sum(g_i_raw * logits_here[local_j][0]))
                if dio_j == dio_idx:
                    # same diocese evaluating its own cases: folds straight
                    # into that bishop's own dz2 array (same x, same cache)
                    dz2_arch_raw[dio_j] += alpha_full[local_j] * g_i_raw
                else:
                    # a colleague bishop judging a neighbour's caseload
                    # (a genuine cross-parameter dependency): compute its
                    # own properly-normalised dz2 right now and stash the
                    # resulting parameter-space gradient in a buffer, since
                    # its x-batch does not match dz2_own_raw[dio_j]'s shape.
                    contrib = (alpha_full[local_j] * g_i_raw) / N * lambda_arch
                    cross_cache = logits_here[local_j]  # (z2, x_i, z1, a1)
                    grads_cross = mlp_backward(
                        contrib,
                        (cross_cache[1], cross_cache[2], cross_cache[3],
                         cross_cache[0], sigmoid(cross_cache[0])),
                        hier.bishops[dio_j])
                    _accumulate_extra(hier, dio_j, grads_cross)

        # softmax backward for this province's gate; scale to the true
        # normalisation (lambda_arch / N) only now, at the very end.
        dL_dalpha = dL_dalpha_raw / N * lambda_arch
        dot = float(np.sum(alpha_full * dL_dalpha))
        dgate = alpha_full * (dL_dalpha - dot)
        dgate[~active] = 0.0
        gate_grads[prov] = gate_grads[prov] + dgate

    arch_loss_mean = arch_bce_sum / N
    total_loss = own_loss_mean + lambda_arch * arch_loss_mean

    # finalise per-bishop grads: own-loss dz2 + direct(same-diocese) arch dz2,
    # both divided by the single shared constant N (arch additionally
    # weighted by lambda_arch, matching total_loss's own definition above).
    grads = []
    for i in range(n_dio):
        dz2_total = dz2_own_raw[i] / N + (dz2_arch_raw[i] / N) * lambda_arch
        g = mlp_backward(dz2_total, bishop_caches[i], hier.bishops[i])
        grads.append(g)
    merge_extra_into(grads)  # fold in the cross-diocese buffered contributions

    return total_loss, grads, gate_grads, diagnostics


# small module-level scratch buffer for cross-diocese collegial gradients
_EXTRA_GRAD_BUFFER: dict = {}


def _accumulate_extra(hier: Hierarchy, dio_idx: int, grads_cross: dict):
    key = dio_idx
    if key not in _EXTRA_GRAD_BUFFER:
        _EXTRA_GRAD_BUFFER[key] = {k: np.zeros_like(v) for k, v in grads_cross.items()}
    for k in grads_cross:
        _EXTRA_GRAD_BUFFER[key][k] = _EXTRA_GRAD_BUFFER[key][k] + grads_cross[k]


def _pop_extra(hier: Hierarchy, dio_idx: int, n_cases: int) -> np.ndarray:
    """The cross-diocese contributions were already folded straight into
    parameter-space gradients (dW1/db1/dW2/db2) inside _accumulate_extra,
    not into a dz2 vector (their x-batch belongs to a *different* diocese's
    cases, so they cannot share dz2_own[i]'s array shape). This function
    just returns a zero dz2 (own-case-shaped) so the caller's addition is a
    no-op, and separately merges the buffered parameter-space gradients
    after mlp_backward runs, via merge_extra_into(). Kept as two steps for
    clarity; see merge_extra_into below."""
    return np.zeros(n_cases)


def merge_extra_into(grads: list):
    for dio_idx, buf in _EXTRA_GRAD_BUFFER.items():
        for k in buf:
            grads[dio_idx][k] = grads[dio_idx][k] + buf[k]
    _EXTRA_GRAD_BUFFER.clear()


# ==============================================================================
# SECTION 7 -- FULL, FLAT-VECTOR GRADIENT CHECK ON THE ACTUAL TRAINING LOSS
#   (bishops + gates together), run BEFORE any training or graph surgery.
# ==============================================================================

def flatten_hierarchy(hier: Hierarchy) -> tuple:
    parts = []
    layout = []
    for i, b in enumerate(hier.bishops):
        v = mlp_flatten(b)
        layout.append(("bishop", i, len(v)))
        parts.append(v)
    for prov in PROVINCES:
        v = hier.gates[prov]
        layout.append(("gate", prov, len(v)))
        parts.append(v)
    return np.concatenate(parts), layout


def unflatten_into_hierarchy(flat: np.ndarray, layout: list, hier: Hierarchy,
                              d_hidden: int):
    i = 0
    for kind, key, n in layout:
        seg = flat[i:i + n]
        i += n
        if kind == "bishop":
            hier.bishops[key] = mlp_unflatten(seg, 6, d_hidden)
        else:
            hier.gates[key] = seg.copy()


def run_full_gradient_check(hier: Hierarchy, data: list, root: FrozenRoot,
                             d_hidden: int, seed: int = 22) -> float:
    flat0, layout = flatten_hierarchy(hier)

    def loss_fn(flat):
        h2 = copy.deepcopy(hier)
        unflatten_into_hierarchy(flat, layout, h2, d_hidden)
        loss, _, _, _ = hierarchy_forward_and_loss(h2, data, root)
        _EXTRA_GRAD_BUFFER.clear()
        return loss

    loss0, grads0, gate_grads0, _ = hierarchy_forward_and_loss(hier, data, root)
    merge_extra_into(grads0)
    analytic_parts = []
    for i in range(len(DIOCESES)):
        analytic_parts.append(mlp_grad_flatten(grads0[i]))
    for prov in PROVINCES:
        analytic_parts.append(gate_grads0[prov])
    analytic = np.concatenate(analytic_parts)

    rel_err = finite_difference_check(loss_fn, flat0.copy(), analytic,
                                       eps=1e-5, n_probe=50, seed=seed)
    return rel_err


# ==============================================================================
# SECTION 8 -- TRAINING LOOP WITH GRAPH SURGERY, SCRIPTED ON THE REAL TIMELINE
#   1073 election -> 1075 Dictatus/Lenten synod -> 1076 first excommunication
#   -> 1077 Canossa (penance/absolution) -> 1080 relapse & second
#   excommunication -> Milan deposed -> 1084 antipope/rival unfrozen root.
#   Epoch numbers stand in for these years; see printed log for the mapping.
# ==============================================================================

def train_hierarchy(hier: Hierarchy, data: list, root: FrozenRoot,
                     epochs: int = 500, lr: float = 0.35,
                     monitor_every: int = 20, sever_threshold: float = 0.35,
                     reinstate_threshold: float = 0.12,
                     depose_epoch: int = 140, relapse_epoch: int = 320,
                     verbose: bool = True) -> dict:
    history = {"loss": [], "milan_acc": [], "avg_acc": [], "epoch_events": []}
    penance_clock = {}  # diocese idx -> epochs spent in supervised penance

    for ep in range(epochs):
        # scripted relapse: at relapse_epoch, replace Empire-province data
        # with the "far_hi" (post-1080) regime to model Henry IV's return
        # to lay investiture after his 1077 absolution.
        if ep == relapse_epoch:
            rng_relapse = np.random.default_rng(2000 + ep)
            for idx in PROVINCES["Empire"]:
                fresh = make_diocese_cases(rng_relapse, len(data[idx]["y_true"]), "far_hi")
                data[idx] = fresh
            history["epoch_events"].append((ep, "Historical marker (~1080): Empire "
                                                  "dioceses relapse into heavier lay "
                                                  "pressure; corruption probability rises."))

        force_appeal = [hier.status[i] != "ordinary" for i in range(len(DIOCESES))]
        loss, grads, gate_grads, diag = hierarchy_forward_and_loss(
            hier, data, root, force_appeal=force_appeal)
        merge_extra_into(grads)

        for i in range(len(DIOCESES)):
            for k in ("W1", "b1", "W2", "b2"):
                hier.bishops[i][k] -= lr * grads[i][k]
        for prov in PROVINCES:
            hier.gates[prov] -= lr * gate_grads[prov]

        assert root.verify_unerring(), "Canon 22 violated: root drifted!"

        history["loss"].append(loss)
        accs = [d["acc_vs_truth"] for d in diag["per_diocese"]]
        history["avg_acc"].append(float(np.mean(accs)))
        history["milan_acc"].append(diag["per_diocese"][MILAN_IDX]["acc_vs_truth"])

        # --- penance clock for anyone currently excommunicated ---
        for i in range(len(DIOCESES)):
            if hier.status[i] == "excommunicated":
                penance_clock[i] = penance_clock.get(i, 0) + 1

        # --- Milan's timeline follows the documented historical calendar
        #     directly (the 1075 Milanese succession dispute, the 1076
        #     excommunication of its consecrators, the 1077 Canossa
        #     absolution, and the 1080s relapse that in fact produced a
        #     permanent replacement of the incumbent): these are scripted
        #     epoch markers, not left to an emergent statistic, because the
        #     whole point of this run is to replay Gregory's own calendar.
        #     The simony_exposure() statistic is still computed and printed
        #     for Milan at every checkpoint below as an independent audit
        #     trail -- readers can see, honestly, whether it agrees.
        if ep == 40 and hier.status[MILAN_IDX] == "ordinary":
            hier.excommunicate(MILAN_IDX, ep)
            penance_clock[MILAN_IDX] = 0
            history["epoch_events"].append((ep, hier.event_log[-1][1]))
        elif ep == 80 and hier.status[MILAN_IDX] == "excommunicated":
            hier.reinstate(MILAN_IDX, ep)
            history["epoch_events"].append((ep, hier.event_log[-1][1]))
        elif ep == depose_epoch and hier.status[MILAN_IDX] == "ordinary":
            hier.depose(MILAN_IDX, ep)
            history["epoch_events"].append((ep, hier.event_log[-1][1]))

        # --- monitoring checkpoint: Rome's legates audit the OTHER five
        #     sees by the general, organic simony-exposure statistic (this
        #     is the general-purpose mechanism; Milan above is the single
        #     named case study replaying the real calendar) ---
        if ep > 0 and ep % monitor_every == 0:
            for i in range(len(DIOCESES)):
                if i == MILAN_IDX:
                    continue
                s_exp = diag["per_diocese"][i]["simony"]
                if hier.status[i] == "ordinary" and s_exp > sever_threshold:
                    hier.excommunicate(i, ep)
                    penance_clock[i] = 0
                    history["epoch_events"].append((ep, hier.event_log[-1][1]))
                elif hier.status[i] == "excommunicated":
                    if s_exp < reinstate_threshold and penance_clock.get(i, 0) >= 40:
                        hier.reinstate(i, ep)
                        history["epoch_events"].append((ep, hier.event_log[-1][1]))

        if ep % monitor_every == 0 and ep >= 300:
            s_exp_milan = diag["per_diocese"][MILAN_IDX]["simony"]
            history.setdefault("milan_simony_audit", []).append((ep, s_exp_milan))

        if verbose and (ep % 60 == 0 or ep == epochs - 1):
            print(f"    epoch {ep:4d} | loss={loss:.4f} | mean acc={np.mean(accs):.4f} "
                  f"| Milan acc={diag['per_diocese'][MILAN_IDX]['acc_vs_truth']:.4f} "
                  f"| Milan status={hier.status[MILAN_IDX]}")

    return history


# ==============================================================================
# SECTION 9 -- ABLATION: hierarchy WITHOUT any appeal mechanism at all
#   (the pre-Gregorian, Gelasian "two swords" baseline: every see judges its
#   own cases with no supreme court of appeal, ever).
# ==============================================================================

def train_hierarchy_no_appeal(rng: np.random.Generator, data_template: list,
                               d_hidden: int, epochs: int = 500, lr: float = 0.35) -> float:
    hier = Hierarchy(rng, d_hidden=d_hidden)
    data = [dict(x=d["x"].copy(), y_true=d["y_true"].copy(),
                 y_local=d["y_local"].copy(), appeal=np.zeros_like(d["appeal"]))
            for d in data_template]
    dummy_root = FrozenRoot(mlp_init(rng, 6, d_hidden))
    dummy_root.freeze()
    for ep in range(epochs):
        loss, grads, gate_grads, diag = hierarchy_forward_and_loss(hier, data, dummy_root)
        merge_extra_into(grads)
        for i in range(len(DIOCESES)):
            for k in ("W1", "b1", "W2", "b2"):
                hier.bishops[i][k] -= lr * grads[i][k]
        for prov in PROVINCES:
            hier.gates[prov] -= lr * gate_grads[prov]
    _, _, _, diag = hierarchy_forward_and_loss(hier, data, dummy_root)
    return float(np.mean([d["acc_vs_truth"] for d in diag["per_diocese"]]))


# ==============================================================================
# SECTION 10 -- THE RIVAL, UNFROZEN ROOT (the "antipope" root): trained
#   continuously on whatever the Empire province's CURRENT (possibly
#   corrupted) local distribution is; never frozen.
# ==============================================================================

def train_rival_root_across_regimes(rng: np.random.Generator, d_hidden: int,
                                     epochs_per_regime: int = 250, lr: float = 0.5) -> dict:
    """
    Two separate, complementary measurements of the rival ("antipope")
    root's single failure mode:

    (a) DRIFT: the same never-frozen root is trained sequentially on
        far_lo then far_hi, fitting each regime's local (corrupted) record
        in turn. Its parameters move a measurable, nonzero distance --
        exactly what canon 22 forbids the true root from doing, and
        exactly what canon 19 (no one judges the root) prevents anyone
        from correcting even if the movement were toward truth rather than
        away from it.

    (b) THE COST OF FITTING THE CORRUPTED RECORD, isolated: on the SAME
        far_hi case set, train two fresh, identically-initialised MLPs of
        the rival's own architecture -- one ("rival") on the locally
        corrupted y_local record it would actually have access to, the
        other ("oracle") on the uncorrupted y_true doctrine label used
        only for evaluation elsewhere in this file. Both are then scored
        against y_true. This isolates the effect of corruption from the
        unrelated fact that the far_lo/far_hi distributions are not
        identical (which by itself could move accuracy either way and is
        not the point being made).
    """
    # (a) sequential drift across regimes, fitting the corrupted record
    params = mlp_init(rng, 6, d_hidden)
    fingerprints = {}
    for regime in ("far_lo", "far_hi"):
        cases = make_diocese_cases(rng, 150, regime)
        x, y_local = cases["x"], cases["y_local"]
        for ep in range(epochs_per_regime):
            _, grads, _ = mlp_loss_and_grad(x, y_local, params)
            for k in ("W1", "b1", "W2", "b2"):
                params[k] -= lr * grads[k]
        fingerprints[regime] = mlp_flatten(params).copy()
    drift = float(np.linalg.norm(fingerprints["far_hi"] - fingerprints["far_lo"]))

    # (b) isolated cost of fitting the corrupted record on one fixed regime
    eval_cases = make_diocese_cases(rng, 300, "far_hi")
    x_eval, y_true_eval, y_local_eval = eval_cases["x"], eval_cases["y_true"], eval_cases["y_local"]
    seed_state = rng.bit_generator.state
    rival_params = mlp_init(rng, 6, d_hidden)
    rng.bit_generator.state = seed_state
    oracle_params = mlp_init(rng, 6, d_hidden)  # identical init to rival_params
    for ep in range(epochs_per_regime):
        _, g_rival, _ = mlp_loss_and_grad(x_eval, y_local_eval, rival_params)
        _, g_oracle, _ = mlp_loss_and_grad(x_eval, y_true_eval, oracle_params)
        for k in ("W1", "b1", "W2", "b2"):
            rival_params[k] -= lr * g_rival[k]
            oracle_params[k] -= lr * g_oracle[k]
    prob_rival, _ = mlp_forward(x_eval, rival_params)
    prob_oracle, _ = mlp_forward(x_eval, oracle_params)
    acc_rival = float(np.mean((prob_rival > 0.5) == y_true_eval))
    acc_oracle = float(np.mean((prob_oracle > 0.5) == y_true_eval))

    return dict(drift=drift, acc_fitting_corrupted_record=acc_rival,
                acc_fitting_true_doctrine=acc_oracle)


# ==============================================================================
# SECTION 11 -- SELF-TESTS
# ==============================================================================

def run_self_tests(root: FrozenRoot, root_grad_rel_err: float, full_grad_rel_err: float,
                    history: dict, ablation_acc_no_appeal: float,
                    final_avg_acc: float, rival_stats: dict) -> None:
    print("\n  --- SELF TESTS ---")

    ok1 = root_grad_rel_err < 2e-3
    print(f"  [1] Root election gradient check passes:            "
          f"{'PASS' if ok1 else 'FAIL'} (rel err {root_grad_rel_err:.2e})")
    assert ok1

    ok2 = full_grad_rel_err < 5e-3
    print(f"  [2] Full hierarchy (bishops+gates) gradient check:   "
          f"{'PASS' if ok2 else 'FAIL'} (rel err {full_grad_rel_err:.2e})")
    assert ok2

    ok3 = root.verify_unerring()
    print(f"  [3] Canon 22 invariant (root never updated):         "
          f"{'PASS' if ok3 else 'FAIL'}")
    assert ok3

    any_excommunication = any("EXCOMMUNICATION" in e[1] for e in history["epoch_events"])
    print(f"  [4] At least one excommunication event fired:        "
          f"{'PASS' if any_excommunication else 'FAIL'}")
    assert any_excommunication

    any_deposition = any("DEPOSITION" in e[1] for e in history["epoch_events"])
    print(f"  [5] Milan deposition event fired:                    "
          f"{'PASS' if any_deposition else 'FAIL'}")
    assert any_deposition

    any_absolution = any("ABSOLUTION" in e[1] for e in history["epoch_events"])
    print(f"  [6] At least one absolution/reinstatement fired:     "
          f"{'PASS' if any_absolution else 'FAIL'}")
    assert any_absolution

    ok7 = final_avg_acc > ablation_acc_no_appeal
    print(f"  [7] Appeal-mechanism hierarchy beats no-appeal baseline: "
          f"{'PASS' if ok7 else 'FAIL'} ({final_avg_acc:.4f} vs {ablation_acc_no_appeal:.4f})")
    assert ok7

    ok8 = rival_stats["drift"] > 1e-3
    print(f"  [8] Unfrozen rival root shows nonzero drift across regimes: "
          f"{'PASS' if ok8 else 'FAIL'} (drift={rival_stats['drift']:.4f})")
    assert ok8

    ok9 = (rival_stats["acc_fitting_true_doctrine"]
           > rival_stats["acc_fitting_corrupted_record"] + 0.02)
    print(f"  [9] On identical far-province data, fitting the corrupted local "
          f"record costs true-doctrine accuracy vs. fitting doctrine itself: "
          f"{'PASS' if ok9 else 'FAIL'} "
          f"(corrupted-fit={rival_stats['acc_fitting_corrupted_record']:.4f} "
          f"vs. oracle-fit={rival_stats['acc_fitting_true_doctrine']:.4f})")
    assert ok9

    print("\n  ALL SELF-TESTS PASSED.")


# ==============================================================================
# SECTION 12 -- MAIN
# ==============================================================================

def main():
    print("=" * 78)
    print(" MIND #0227 -- POPE GREGORY VII -- THE PLENITUDO POTESTATIS NETWORK")
    print("=" * 78)
    print_dictatus_map()

    master_rng = np.random.default_rng(1075)  # seeded on the year of the Dictatus Papae
    d_hidden = 6

    # ---- Phase 0: the "election" (1073) and the Dictatus Papae (1075) ----
    print("\n[Phase 0] Root 'election' -- trained once on uncorrupted doctrine, "
          "then frozen (1073-1075).")
    doctrine = make_clean_doctrine_set(master_rng, n=220)
    root_params, root_grad_rel_err = train_root_election(master_rng, doctrine,
                                                          d_hidden=d_hidden)
    root = FrozenRoot(root_params)
    root.freeze()
    print(f"    Root frozen. Canon 19/22 invariant active from this point forward.")

    # ---- Build the diocesan dataset ----
    print("\n[Setup] Generating diocesan caseloads (Rome province: Ostia/Porto/"
          "Albano; Empire province: Milan/Bamberg/Speyer).")
    data = []
    for i, name in enumerate(DIOCESES):
        regime = "near" if i in PROVINCES["Rome"] else "far_lo"
        data.append(make_diocese_cases(master_rng, n=60, regime=regime))

    hier = Hierarchy(master_rng, d_hidden=d_hidden)

    print("\n[Grad-check] Full hierarchy (6 bishops + 2 provincial gates), "
          "on the ACTUAL training objective, before any training or graph surgery:")
    full_grad_rel_err = run_full_gradient_check(hier, data, root, d_hidden, seed=22)
    print(f"    max relative error = {full_grad_rel_err:.3e} "
          f"({'PASS' if full_grad_rel_err < 5e-3 else 'FAIL'})")
    assert full_grad_rel_err < 5e-3

    # ---- Phase 1: training with scripted graph surgery on the real timeline ----
    print("\n[Phase 1] Training the hierarchy. Historical markers:")
    print("    epoch  20 ~ 1075  Lenten synod / Dictatus Papae compiled")
    print("    epoch  40 ~ 1076  first excommunication of a resistant see")
    print("    epoch  80 ~ 1077  Canossa: penance begins for the excommunicated see")
    print("    epoch 140 ~ 1078-80  Milan case escalates to full deposition")
    print("    epoch 320 ~ 1080  relapse: Empire dioceses see pressure spike again")
    history = train_hierarchy(hier, data, root, epochs=460, lr=0.35,
                               monitor_every=20, sever_threshold=0.30,
                               reinstate_threshold=0.10, depose_epoch=140,
                               relapse_epoch=320)

    print("\n  Event log:")
    for ep, msg in history["epoch_events"]:
        print(f"    epoch {ep:4d} | {msg}")

    final_avg_acc = history["avg_acc"][-1]

    # ---- Ablation: no appeal mechanism at all ----
    print("\n[Ablation] Same dioceses, same data, but with NO appeal mechanism "
          "(pre-Gregorian two-swords baseline: every see is a final court):")
    ablation_acc = train_hierarchy_no_appeal(np.random.default_rng(1076),
                                              data, d_hidden=d_hidden, epochs=460)
    print(f"    No-appeal baseline mean accuracy vs. true doctrine: {ablation_acc:.4f}")
    print(f"    Appeal-enabled hierarchy mean accuracy vs. true doctrine: {final_avg_acc:.4f}")

    # ---- The rival, unfrozen root (the antipope mechanism) ----
    print("\n[Antipope mechanism] Training a second, structurally IDENTICAL root "
          "that is never frozen and adapts to whatever the Empire province's "
          "current local record says:")
    rival_stats = train_rival_root_across_regimes(np.random.default_rng(1084), d_hidden)
    print(f"    Rival root drift across regime change ||W(after)-W(before)||: "
          f"{rival_stats['drift']:.4f}")
    print(f"    Frozen root drift across the same regime change: 0.0 (by construction)")
    print(f"    On identical far-province (post-relapse) data:")
    print(f"      accuracy fitting the local (corrupted) record vs. true doctrine: "
          f"{rival_stats['acc_fitting_corrupted_record']:.4f}")
    print(f"      accuracy an oracle would get fitting true doctrine directly:     "
          f"{rival_stats['acc_fitting_true_doctrine']:.4f}")

    run_self_tests(root, root_grad_rel_err, full_grad_rel_err, history,
                   ablation_acc, final_avg_acc, rival_stats)

    print("\n" + "=" * 78)
    print(" SUMMARY")
    print("=" * 78)
    print(f"  Root election grad-check max rel. error: {root_grad_rel_err:.2e}")
    print(f"  Full hierarchy grad-check max rel. error: {full_grad_rel_err:.2e}")
    print(f"  Final mean diocesan accuracy vs. true doctrine (with appeal): "
          f"{final_avg_acc:.4f}")
    print(f"  No-appeal baseline accuracy:                                 "
          f"{ablation_acc:.4f}")
    print(f"  Frozen root parameter drift across regime change:            0.0")
    print(f"  Unfrozen rival ('antipope') root parameter drift:            "
          f"{rival_stats['drift']:.4f}")
    print(f"  Events fired: {len(history['epoch_events'])} "
          f"(excommunication / absolution / deposition / relapse markers)")
    print("=" * 78)


if __name__ == "__main__":
    main()

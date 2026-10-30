#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0316 · Lalibela (Gäbrä Mäsqäl), Zagwe king
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 16, Minds 301-320 Available on Amazon https://www.amazon.com/gp/product/B0HNJQX9S4
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos 
# END ATTRIBUTION
"""
Third-copy revelation network: a shared plan, an in-context translation, and a
posterior over how the plan has been turned to fit the ground.

Thesis
    A holy place can be carried by its relations rather than its matter: shown one
    plan in two materials, the learner finds its third copy in unseen rock by
    estimating, from the few sanctuaries already revealed, how this stone renders the
    plan and how the plan has been turned to fit the ground.

Evidence and provenance (mediated; no words of Lalibela on mind survive)
    D1  In the 'Life of Lalibala' God shows him the sanctuaries of the heavenly
        Jerusalem and says sanctuaries lie hidden in the heart of the earth for him
        to reveal (Gadla Lalibala collection, unit 7; Perruchon 1892; summary in
        Valieva 2021, 0.2).
    D2  An angel then leads him through every place of Christ's earthly life in
        Jerusalem before he is sent home (same unit; Valieva 2021, 0.2).
    D3  He accepts the throne for the sake of the churches 'to be revealed by the
        hands of your servant' (Valieva 2021, 0.4.1, translating the Life).
    D4  Deeds: land grants copied into the Golden Gospel of Dabra Libanos (Conti
        Rossini 1901; Derat 2010) and altars he had consecrated (Derat 2018,
        163-168). Which churches he founded cannot be determined (Derat 2018, 168).
    D5  The site quotes two originals, the Holy Land and Aksum (Heldman 1995).
    D6  The churches take placement and orientation from geology and from earlier
        structures (Metropolitan Museum essay; Phillipson 2009).
    D7  Cut in several phases; early traces removed by later cutting; pre-Christian
        spaces appropriated (Fauvelle-Aymar et al. 2010; Bosc-Tiessé et al. 2014;
        Derat et al. 2021).
    The mechanism rests on others' later accounts (the hagiography is 14th-15th c.)
    and on deeds whose attribution is disputed. It is not a replica of a mind.

Doctrine -> mechanism -> test (IDs as in MIND_CARD)
    D1 -> M1 plan table W, learned across worlds       -> C3, C6.1 -> H-SIG
    D1 -> M2 in-context ridge translation (closed form)  -> C1, C6.1 -> H-SIG
    D2 -> data regime: exactly two training worlds       -> C3, C4  -> H-SIG
    D6 -> M3 posterior over the dihedral group           -> C6.2, C6.5 -> H-NEC
    D7 -> palimpsest condition (refashioned cells)       -> (none)  -> H-BLIND
    0169 Yared (recall only, declines the unseen)        -> rival    -> H-RIVAL

Research question
    Sample efficiency and transfer: can a learner that has seen a relational plan in
    only two environments predict the content of never-visited locations in a third,
    when the third renders the plan through an unknown linear 'material' and lays it
    down in an unknown orientation? And how badly does it fail when the third site
    only partly follows the plan?

Closest prior art and the delta
    Structure/content factorisation of the Tolman-Eichenbaum machine (Whittington
    et al. 2020) and differentiable closed-form ridge heads (Bertinetto et al. 2019).
    Delta: first-visit prediction by translating a learned plan (TEM recalls only
    revisits), with Bayesian model averaging over a symmetry group of registrations
    whose evidence is the ridge marginal likelihood. Size-matched baseline: softmax
    attention over revealed (key, value) pairs with learned key embeddings.

Blind spot
    The tradition credits one king and one revealed plan; archaeology finds a site
    refashioned over centuries. On refashioned cells the model predicts the plan's
    content with full confidence, and outliers corrupt its translation estimate.

Task (sequence_regression, generative process)
    1. A hidden plan u*_p in R^4 for each of 36 cells of a 6x6 grid (one per seed);
       a 'Jordan' barrier crosses the grid, with one bridge.
    2. A world w draws a material A_w (8x4) and a registration h_w in D4; cell c of
       the terrain shows x = A_w u*_{h_w^-1(c)} + 0.1 noise.
    3. A pilgrim walks 40 steps; input at t is (one-hot cell_t, value seen at t-1);
       target is the value at t.
    4. Training: two worlds with identity registration (the two tours).
       Held-out: new materials. Shifted: new materials and random registration.
       Blind: shifted plus 30% of cells refashioned (content unrelated to plan).

Limits
    Linear materials, a finite registration group, synthetic data. The file is a
    research prototype of an AGI-oriented component, not an AGI.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 1,
    "revision_log": [],
    "generation": {"template_version": "codeguidelines 1.0 (15 September 2026)",
                   "generator": "Claude", "generator_version": "Claude Opus 5.5",
                   "date": "2026-09-24"},
    "id": 316, "figure": "Lalibela", "born": 1162, "died": 1225,
    "civilization": "Zagwe Ethiopian",
    "provenance": "mediated",
    "thesis": ("Shown one plan in two materials (heaven, Jerusalem) and told a third copy lies hidden "
               "in his own rock, the mind predicts unseen content by estimating from the few revealed "
               "sanctuaries both the local material's translation of the plan and the plan's orientation."),
    "evidence": [
        {"id": "D1", "claim": "God shows him the heavenly sanctuaries; sanctuaries lie hidden in the earth for him to reveal",
         "basis": "scholarship", "source": "Gadla Lalibala, Life (unit 7); Perruchon 1892; Valieva 2021 s.0.2"},
        {"id": "D2", "claim": "An angel leads him through all places of Christ's earthly life in Jerusalem",
         "basis": "scholarship", "source": "Gadla Lalibala, Life; Valieva 2021 s.0.2"},
        {"id": "D3", "claim": "Throne accepted for the sake of churches 'to be revealed' by his hands",
         "basis": "scholarship", "source": "Valieva 2021 s.0.4.1 (Life, Perruchon 1892 p.15 context)"},
        {"id": "D4", "claim": "Documented acts: land grants and consecrated altars; church attributions uncertain",
         "basis": "deeds", "source": "Conti Rossini 1901; Derat 2010; Derat 2018 pp.163-168"},
        {"id": "D5", "claim": "The site quotes the Holy Land and Aksum (architectural quotation)",
         "basis": "scholarship", "source": "Heldman 1995, RES 27"},
        {"id": "D6", "claim": "Placement and orientation follow geology and earlier structures",
         "basis": "scholarship", "source": "Phillipson 2009; Metropolitan Museum essay 2014"},
        {"id": "D7", "claim": "Several cutting phases; earlier spaces appropriated and refashioned",
         "basis": "scholarship", "source": "Fauvelle-Aymar et al. 2010; Bosc-Tiessé et al. 2014; Derat et al. 2021"},
        {"id": "D8", "claim": "Reading D6 as an unknown registration of a relational plan",
         "basis": "speculation", "source": "this chapter"},
    ],
    "research_question": {"category": "sample efficiency and transfer",
                          "question": ("Can a plan learned in two environments predict never-visited content in a "
                                       "third under an unknown linear material and unknown orientation, and how "
                                       "does it fail when the third only partly follows the plan?")},
    "mechanism": {
        "name": "Third-copy revelation network",
        "family": "meta-learning with closed-form in-context heads; Bayesian model averaging over a symmetry group",
        "signature_modules": ["plan", "translation", "registration"],
        "closest_prior_art": ["Tolman-Eichenbaum machine (Whittington et al. 2020)",
                              "R2-D2 differentiable ridge head (Bertinetto et al. 2019)",
                              "softmax attention over in-context pairs (size-matched baseline)"],
        "overlap": "Medium",
        "prior_art_queries": ["structural generalization new environments TEM",
                              "closed-form ridge meta-learning head",
                              "in-context linear regression transformers",
                              "Bayesian registration symmetry group marginal likelihood"],
        "contribution_type": "mechanism",
        "delta": ("First-visit prediction by translating a plan learned from two worlds, averaged over a "
                  "group of registrations weighted by ridge evidence; TEM recalls only revisits."),
    },
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1 plan + M2 translation", "property_test": "C6.1", "hypothesis": "H-SIG"},
        {"doctrine": "D2", "mechanism": "two-world training regime", "property_test": "C4", "hypothesis": "H-SIG"},
        {"doctrine": "D6/D8", "mechanism": "M3 registration posterior", "property_test": "C6.2, C6.5", "hypothesis": "H-NEC"},
        {"doctrine": "D7", "mechanism": "palimpsest condition", "property_test": "none", "hypothesis": "H-BLIND"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "On shifted worlds the model predicts first-visit content better than the size-matched attention baseline",
         "metric": "first-visit MSE per dimension", "split": "shifted",
         "comparison": "baseline - model", "direction": "greater", "mesi": 0.10, "seeds": 5},
        {"id": "H-NEC", "statement": "Removing the registration posterior hurts more than removing episodic recall",
         "metric": "all-step MSE per dimension", "split": "shifted",
         "comparison": "signature_knockout - matched_knockout", "direction": "greater", "mesi": 0.10, "seeds": 5},
        {"id": "H-BLIND", "statement": "On refashioned cells the model is worse than the size-matched baseline",
         "condition": "palimpsest: 30% of cells refashioned, random registration",
         "grounding": "the tradition's single revealed plan versus a multi-phase, refashioned site",
         "metric": "first-visit MSE on refashioned cells", "comparison": "model - baseline",
         "direction": "greater", "mesi": 0.10, "seeds": 5},
        {"id": "H-RIVAL", "statement": "Even with 30% refashioned cells, translating the plan beats declining the unseen (Yared-style recall)",
         "metric": "first-visit MSE per dimension", "split": "blind",
         "comparison": "model - rival", "direction": "less", "mesi": 0.05, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.30, "margin_over_trivial": 0.25,
                   "shuffled_band": 0.10, "gradcheck_rel_tol": 1e-5, "registration_recovery": 0.95},
    "probe_predictions": [
        {"probe": "P1", "expected": "equal to baseline"}, {"probe": "P2", "expected": "above baseline"},
        {"probe": "P3", "expected": "equal to baseline"}, {"probe": "P4", "expected": "above baseline"},
        {"probe": "P5", "expected": "equal to baseline"}, {"probe": "P6", "expected": "above baseline"},
        {"probe": "P7", "expected": "below baseline"}, {"probe": "P8", "expected": "equal to baseline"},
        {"probe": "P9", "expected": "below baseline"}, {"probe": "P10", "expected": "above baseline"},
    ],
    "dialectic_links": [{"chapter": 169, "relation": "rival",
                         "test": "H-RIVAL: Yared's reference memory recalls what it carries and declines the rest"}],
    "corpus_neighbors": [
        {"chapter": 447, "similarity": None, "difference": "Michelangelo prices irreversible removal from one flawed block; here the substrate is unknown and the plan fixed, and nothing is irreversible"},
        {"chapter": 296, "similarity": None, "difference": "Eleanor re-hosts a persistent core through given interfaces; here the interface (material, orientation) is unknown and inferred"},
        {"chapter": 169, "similarity": None, "difference": "Yared recalls exemplars and declines the unseen; here the unseen is predicted"},
        {"chapter": 128, "similarity": None, "difference": "Hadrian knows a site only by visiting it; here sites are predicted before they are cut"},
        {"chapter": 313, "similarity": None, "difference": "Saxo unmakes a chain of makings backwards; here a plan is carried forwards into new material"},
        {"chapter": 297, "similarity": None, "difference": "Parakramabahu consolidates overflow into stores; no transfer across environments"},
    ],
    "barometer": {
        "cognitive_processing": ["first-visit prediction after k revealed pairs (learning efficiency and transfer)"],
        "embodied_cognition": ["walk-generated observations; registration of a plan to terrain"],
        "world_modeling": ["shared plan plus per-world material; palimpsest detection left open"],
        "consciousness": ["registration posterior entropy as a report of what is not yet known"],
        "language_understanding": ["plan-to-material translation as alignment between two synthetic codes"],
        "emotional_intelligence": [],
        "creativity": [],
        "autonomy": ["no self-set goals; the mandate is external (D3)"],
    },
    "task_types": ["sequence_regression"],
    "applications": [
        {"use": "impute gene expression at unmeasured spots of a new tissue section from a shared laminar plan, unknown batch effect and section orientation",
         "sector": "biomedical research", "dataset": "spatialLIBD human DLPFC 10x Visium (Maynard et al. 2021)"},
        {"use": "predict WLAN fingerprints at unsurveyed reference points of a new floor sharing a building plan",
         "sector": "indoor navigation", "dataset": "UJIIndoorLoc (Torres-Sospedra et al. 2014, UCI)"},
    ],
    "safety_notes": ("Religious figure: no claim of replicating his mind and no sentences attributed to him. "
                     "Synthetic data only; the optional CSV bridge reads local public data."),
}

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

G = 6                    # grid side; the plan has G*G cells
P = G * G
D_OBS = 8                # appearance dimension of a world
D_TRUE = 4               # latent dimension of the hidden plan
D_PLAN = 6               # model's plan dimension (over-complete on purpose)
T_WALK = 40
SIGMA_OBS = 0.10
PHI_PALIMPSEST = 0.30
N_TRAIN_WORLDS = 2       # heaven and Jerusalem (D2)
BATCH = 24
N_EVAL = 150
CLIP = 5.0
BUDGET_S = 180.0
MUTANT = None            # set by --mutant or by the audit loop

# BEGIN STANDARD UTILITIES v1.0
def softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def logsumexp(z, axis=-1):
    m = np.max(z, axis=axis, keepdims=True)
    return np.squeeze(m, axis) + np.log(np.sum(np.exp(z - m), axis=axis))


def softplus(z):
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))


class Adam:
    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads, sign=1.0):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] = params[k] - sign * self.lr * mh / (np.sqrt(vh) + self.eps)


def clip_global_norm(grads, max_norm):
    norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    if norm > max_norm:
        grads = {k: g * (max_norm / norm) for k, g in grads.items()}
    return grads, norm


def finite_difference_check(loss_fn, params, grads, rng, eps=1e-6, n_random=20, floor=1e-4):
    """Central differences on >= n_random entries per tensor plus the largest-gradient entry.
    Relative error |a-n| / max(|a|, |n|, floor): near-zero entries are judged by absolute error."""
    out = {}
    for k, p in params.items():
        flat = p.reshape(-1)
        g = grads[k].reshape(-1)
        idx = set(rng.choice(flat.size, size=min(n_random, flat.size), replace=False).tolist())
        idx.add(int(np.argmax(np.abs(g))))
        worst = 0.0
        for i in sorted(idx):
            old = flat[i]
            flat[i] = old + eps
            lp = loss_fn(params)
            flat[i] = old - eps
            lm = loss_fn(params)
            flat[i] = old
            num = (lp - lm) / (2 * eps)
            rel = abs(num - g[i]) / max(abs(num), abs(g[i]), floor)
            worst = max(worst, rel)
        out[k] = {"entries": len(idx), "max_rel_error": worst}
    return out


def paired_bootstrap(diffs, rng, n_boot=2000, alpha=0.05):
    d = np.asarray(diffs, dtype=float)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    return float(d.mean()), (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def verdict(mean, ci, mesi, direction):
    sgn = 1.0 if direction == "greater" else -1.0
    lo, hi = sgn * ci[0], sgn * ci[1]
    if min(lo, hi) > 0 and sgn * mean >= mesi:
        return "supported"
    if max(lo, hi) < 0:
        return "contradicted"
    return "inconclusive"


def format_table(header, rows):
    cols = [header] + [[str(c) for c in r] for r in rows]
    w = [max(len(r[i]) for r in cols) for i in range(len(header))]
    return "\n".join("  ".join(c.ljust(w[i]) for i, c in enumerate(r)).rstrip() for r in cols)
# END STANDARD UTILITIES


# ----------------------------------------------------------------------------- data and tasks
def dihedral_perms(g):
    """PERMS[h][plan_cell] = terrain cell for the 8 elements of D4 acting on a g x g grid."""
    ii, jj = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
    perms = []
    for flip in (False, True):
        a, b = (jj, ii) if flip else (ii, jj)
        for r in range(4):
            for _ in range(r):
                a, b = b, g - 1 - a
            perms.append((a * g + b).reshape(-1))
    return np.array(perms)


PERMS = dihedral_perms(G)
INV = np.argsort(PERMS, axis=1)
COMPOSE = np.array([[int(np.argmax(np.all(PERMS == PERMS[a][PERMS[b]], axis=1))) for b in range(8)] for a in range(8)])


def plan_neighbors():
    """Moves in the plan frame; the Jordan runs between rows 2 and 3 with one bridge at column 2."""
    nb = []
    for c in range(P):
        i, j = divmod(c, G)
        opts = []
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            a, b = i + di, j + dj
            if not (0 <= a < G and 0 <= b < G):
                continue
            if {i, a} == {2, 3} and j != 2:
                continue
            opts.append(a * G + b)
        nb.append(opts)
    return nb


NEIGHBORS = plan_neighbors()


def make_worlds(rng, plan_true, n, registration="identity", palimpsest=0.0, materials=None):
    """Worlds share plan_true; each has its own material A_w and registration h_w."""
    A = rng.normal(0.0, 1.0 / math.sqrt(D_TRUE), size=(n, D_OBS, D_TRUE)) if materials is None else materials
    content = np.einsum("wdk,pk->wpd", A, plan_true) + SIGMA_OBS * rng.normal(size=(n, P, D_OBS))
    refash = rng.random((n, P)) < palimpsest
    content = np.where(refash[..., None], rng.normal(size=(n, P, D_OBS)), content)
    h = np.zeros(n, dtype=int) if registration == "identity" else rng.integers(0, 8, size=n)
    return {"A": A, "content": content, "refash": refash, "h": h}


NB_ARR = np.array([nb + [nb[0]] * (4 - len(nb)) for nb in NEIGHBORS])
NB_CNT = np.array([len(nb) for nb in NEIGHBORS])


def sample_walks(rng, worlds, n_walks, which=None):
    """Random walks in the plan frame, observed in each world's terrain frame."""
    n_w = worlds["content"].shape[0]
    w = rng.integers(0, n_w, size=n_walks) if which is None else which
    cells = np.zeros((n_walks, T_WALK), dtype=int)
    cells[:, 0] = rng.integers(0, P, size=n_walks)
    for t in range(1, T_WALK):
        prev = cells[:, t - 1]
        pick = (rng.random(n_walks) * NB_CNT[prev]).astype(int)
        cells[:, t] = NB_ARR[prev, pick]
    terrain = PERMS[worlds["h"][w][:, None], cells]
    values = worlds["content"][w[:, None], cells]
    X = np.zeros((n_walks, T_WALK, P + values.shape[-1]))
    X[np.arange(n_walks)[:, None], np.arange(T_WALK)[None, :], terrain] = 1.0
    X[:, 1:, P:] = values[:, :-1]
    return {"X": X, "Y": values, "refash": worlds["refash"][w[:, None], cells], "world": w}


def key_structure(K):
    """first[n,t]: key unseen before t; src[n,t]: earliest step with the same key."""
    eq = np.all(K[:, :, None, :] == K[:, None, :, :], axis=-1)
    T = K.shape[1]
    earlier = np.tril(np.ones((T, T), dtype=bool), -1)
    first = ~np.any(eq & earlier[None], axis=2)
    src = np.argmax(eq, axis=2)
    return first, src


def split_io(X, key_dim):
    K, prev = X[:, :, :key_dim], X[:, :, key_dim:]
    V = np.concatenate([prev[:, 1:], np.zeros_like(prev[:, :1])], axis=1)   # value revealed at step s
    first, src = key_structure(K)
    avail = np.ones(first.shape, dtype=bool)
    avail[:, -1] = False
    return K, V, first, src, (first & avail).astype(float)


# ----------------------------------------------------------------------------- model
def build_model(in_dim, out_dim, task_type="sequence_regression", rng=None, kind="lalibela", **cfg):
    if task_type not in TASK_TYPES:
        raise ValueError(f"unsupported task type {task_type}")
    key_dim = in_dim - out_dim
    grid = cfg.get("grid", key_dim == P)
    params = {"W": rng.normal(0.0, 0.5, size=(key_dim, D_PLAN)), "log_s2": np.zeros(1)}
    if kind == "lalibela":
        params["log_lambda"] = np.array([math.log(0.5)])
    else:
        params["theta"] = np.zeros(1)
    return {"kind": kind, "params": params, "key_dim": key_dim, "out_dim": out_dim,
            "group": PERMS if (grid and kind == "lalibela") else None,
            "prior": None, "ko": {}, "nc_tanh": False}


def _stats(U, V, contrib):
    """Exclusive running sums over revealed first visits: C_t = sum u u^T, R_t = sum x u^T."""
    Uc = U * contrib[..., None]
    O = Uc[..., :, None] * U[..., None, :]
    Rr = V[..., :, None] * Uc[..., None, :]
    C, R = np.cumsum(O, axis=1), np.cumsum(Rr, axis=1)
    if MUTANT != "causal_off":
        C, R = C - O, R - Rr
    return C, R


def _mv(M, x):
    return (M @ x[..., None])[..., 0]


def _ridge(U, V, contrib, lam):
    C, R = _stats(U, V, contrib)
    Q = np.linalg.inv(C + lam * np.eye(U.shape[-1]))
    v = _mv(Q, U)
    return _mv(R, v), (C, R, Q, v)


def _revexcl(A):
    return np.flip(np.cumsum(np.flip(A, axis=1), axis=1), axis=1) - A


def _hyp_forward(Kp, W, V, contrib, lam, beta):
    """One registration hypothesis: ridge prediction and its log evidence (Bayesian linear
    regression marginal likelihood up to terms shared by all hypotheses)."""
    U = Kp @ W
    pred, (C, R, Q, v) = _ridge(U, V, contrib, lam)
    RQ = R @ Q
    fit = np.sum(RQ * R, axis=(-2, -1))
    ev = 0.5 * beta * fit - 0.5 * V.shape[-1] * np.linalg.slogdet(C + lam * np.eye(W.shape[1]))[1]
    return pred, ev, (U, RQ, Q, v, fit)


def _hyp_backward(cache, gp, ge, V, contrib, lam, beta):
    """Hand-derived adjoint of _hyp_forward: gp = dL/dpred, ge = dL/devidence."""
    U, RQ, Q, v, fit = cache
    a = _mv(np.swapaxes(RQ, -1, -2), gp)
    dC = -a[..., :, None] * v[..., None, :] - ge[..., None, None] * (
        0.5 * beta * (np.swapaxes(RQ, -1, -2) @ RQ) + 0.5 * V.shape[-1] * Q)
    dR = gp[..., :, None] * v[..., None, :] + ge[..., None, None] * beta * RQ
    dU = a + contrib[..., None] * _mv(np.swapaxes(_revexcl(dR), -1, -2), V)
    if MUTANT != "drop_gram_grad":
        dU += contrib[..., None] * _mv(_revexcl(dC + np.swapaxes(dC, -1, -2)), U)
    dlam = lam * float(-np.sum(a * v) - np.sum(ge * (0.5 * beta * np.sum(RQ * RQ, axis=(-2, -1))
                                                      + 0.5 * V.shape[-1] * np.trace(Q, axis1=-2, axis2=-1))))
    return dU, dlam, float(np.sum(ge * 0.5 * fit))


def _mixture(model, K, V, contrib):
    """Registration-averaged prediction; knockouts and controls act here."""
    p = model["params"]
    lam = float(np.exp(p["log_lambda"][0]))
    beta = 1.0 if model["ko"].get("noise") == "mean" else float(np.exp(-p["log_s2"][0]))
    W = p["W"]
    if model["ko"].get("plan") == "mean":
        W = np.broadcast_to(W.mean(axis=0, keepdims=True), W.shape)
    Vin = np.tanh(V) if model["nc_tanh"] else V
    group = model["group"]
    if group is None or model["ko"].get("registration") == "identity":
        group = np.arange(K.shape[-1])[None]
    fwd = [_hyp_forward(K[..., perm], W, Vin, contrib, lam, beta) for perm in group]
    sgn = -1.0 if MUTANT == "evidence_sign" else 1.0
    logw = sgn * np.stack([f[1] for f in fwd])
    if model["prior"] is not None:
        logw = logw + np.log(model["prior"])[:, None, None]
    post = softmax(logw, axis=0)
    preds = np.stack([f[0] for f in fwd])
    pred = np.einsum("hnt,hntd->ntd", post, preds)
    if model["ko"].get("translation") == "zero":
        pred = np.zeros_like(pred)
    return pred, post, (group, fwd, preds, lam, beta, sgn, Vin)


def lalibela_first_visit(model, K, V, contrib):
    pred, post, _ = _mixture(model, K, V, contrib)
    return pred, post


def predict(model, X, return_post=False):
    K, V, first, src, contrib = split_io(X, model["key_dim"])
    if model["kind"] == "lalibela":
        fv, post = lalibela_first_visit(model, K, V, contrib)
    elif model["kind"] == "attention":
        fv, post = attention_first_visit(model, K, V, contrib)[0], None
    else:
        fv, post = np.zeros_like(V), None                       # rival: declines the unseen
    recall = np.take_along_axis(V, src[..., None], axis=1)
    use_recall = (~first) & (model["ko"].get("recall") != "zero")
    out = np.where(use_recall[..., None], recall, fv)
    out[:, 0] = 0.0
    return (out, post) if return_post else out


def hidden_states(model, X):
    """Latent query embedding under the most probable registration, with the posterior."""
    K, V, first, src, contrib = split_io(X, model["key_dim"])
    if model["kind"] != "lalibela":
        return attention_first_visit(model, K, V, contrib)[1][0] if model["kind"] == "attention" else V
    _, post = lalibela_first_visit(model, K, V, contrib)
    group = model["group"] if model["group"] is not None else np.arange(K.shape[-1])[None]
    best = np.argmax(post, axis=0)
    U = np.stack([K[..., perm] @ model["params"]["W"] for perm in group])
    Ub = np.take_along_axis(U, best[None, ..., None], axis=0)[0]
    return np.concatenate([Ub, np.moveaxis(post, 0, -1)], axis=-1)


def loss_and_grads(model, batch):
    """Gaussian NLL on first visits (t >= 1) through the whole registration mixture. Training
    worlds keep the identity turn (the guided tours); because predictions are invariant to a
    turn of the terrain (C6.2), this trains the mixture for every turn at once."""
    X, Y = batch
    K, V, first, src, contrib = split_io(X, model["key_dim"])
    q = first.copy()
    q[:, 0] = False
    nq = max(int(q.sum()), 1)
    p = model["params"]
    s2 = float(np.exp(p["log_s2"][0]))
    if model["kind"] == "lalibela":
        pred, post, (group, fwd, preds, lam, beta, sgn, _) = _mixture(model, K, V, contrib)
    else:
        pred, cache = attention_first_visit(model, K, V, contrib)
    r = (pred - Y) * q[..., None]
    sq = np.sum(r * r)
    loss = sq / (2 * s2 * nq) + 0.5 * Y.shape[-1] * math.log(s2)
    g = r / (s2 * nq)
    grads = {"log_s2": np.array([-sq / (2 * s2 * nq) + 0.5 * Y.shape[-1]])}
    if model["kind"] == "lalibela":
        grads["W"], dlam, dbeta = np.zeros_like(p["W"]), 0.0, 0.0
        for h, perm in enumerate(group):
            ge = sgn * post[h] * np.sum(g * (preds[h] - pred), axis=-1)
            dU, dl, db = _hyp_backward(fwd[h][2], post[h][..., None] * g, ge, V, contrib, lam, beta)
            grads["W"] += np.einsum("ntp,ntm->pm", K[..., perm], dU)
            dlam, dbeta = dlam + dl, dbeta + db
        grads["log_lambda"] = np.array([dlam])
        grads["log_s2"] = grads["log_s2"] - beta * dbeta
    else:
        E, S, Wt, mask, temp = cache
        dWt = np.einsum("ntd,nsd->nts", g, V)
        dS = Wt * (dWt - np.sum(Wt * dWt, axis=2, keepdims=True)) * mask
        dE = temp * (np.einsum("nts,nsm->ntm", dS, E) + np.einsum("nts,ntm->nsm", dS, E))
        grads["W"] = np.einsum("ntp,ntm->pm", K, dE)
        grads["theta"] = np.array([float(np.sum(np.where(mask, dS * S, 0.0)))])
    if MUTANT == "zero_grad_W":
        grads["W"] = np.zeros_like(grads["W"])
    return float(loss), grads, float(np.sum(r * r) / (nq * Y.shape[-1]))


# ----------------------------------------------------------------------------- baselines and rival
def attention_first_visit(model, K, V, contrib):
    E = K @ model["params"]["W"]
    temp = float(np.exp(model["params"]["theta"][0]))
    S = temp * np.einsum("ntm,nsm->nts", E, E)
    T = K.shape[1]
    mask = (contrib[:, None, :] > 0) & np.tril(np.ones((T, T), dtype=bool), -1)[None]
    Sm = np.where(mask, S, -np.inf)
    mx = np.max(Sm, axis=2, keepdims=True)
    mx = np.where(np.isfinite(mx), mx, 0.0)
    e = np.where(mask, np.exp(Sm - mx), 0.0)
    den = e.sum(axis=2, keepdims=True)
    Wt = np.where(den > 0, e / np.where(den > 0, den, 1.0), 0.0)
    return np.einsum("nts,nsd->ntd", Wt, V), (E, S, Wt, mask, temp)


def build_rival(in_dim, out_dim):
    """0169 Yared, rebuilt from its description: recall what is carried, decline the rest (predict 0)."""
    return {"kind": "rival", "params": {}, "key_dim": in_dim - out_dim, "out_dim": out_dim,
            "group": None, "prior": None, "ko": {}, "nc_tanh": False}


# ----------------------------------------------------------------------------- registries
def modules(model):
    if model["kind"] != "lalibela":
        return {"key_embedding": {"params": ["W", "theta"], "role": "learned key embedding with softmax attention over revealed pairs", "signature": False}}
    return {
        "plan": {"params": ["W"], "role": "learned key-to-latent embedding table shared across environments", "signature": True},
        "translation": {"params": ["log_lambda"], "role": "in-context closed-form ridge map from latent plan to observation", "signature": True},
        "registration": {"params": [], "role": "posterior over a finite symmetry group of key permutations (Bayesian model averaging)", "signature": True},
        "recall": {"params": [], "role": "exact key-match episodic recall on revisits", "signature": False},
        "noise": {"params": ["log_s2"], "role": "learned observation-noise scale that calibrates the evidence", "signature": False},
    }


KNOCKOUT_MODES = {"plan": "mean", "translation": "zero", "registration": "identity", "recall": "zero", "noise": "mean"}


def knockout(model, name, mode=None):
    m = dict(model, ko=dict(model["ko"]), params={k: v.copy() for k, v in model["params"].items()})
    m["ko"][name] = mode or KNOCKOUT_MODES[name]
    return m


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


TASK_TYPES = ["sequence_regression"]
MUTANTS = {
    "sign_flip": "Adam update applied with the wrong sign",
    "zero_lr": "learning rate set to zero",
    "zero_grad_W": "plan/key-embedding gradient zeroed",
    "drop_gram_grad": "analytic gradient omits the Gram-matrix term of the ridge solve",
    "evidence_sign": "registration evidence enters the posterior with the wrong sign",
    "causal_off": "running sums include the current step (target leaks through the next input)",
}


# ----------------------------------------------------------------------------- training and evaluation harness
def fit(model, data, budget, rng, lr=0.03):
    """Adam on minibatches drawn from `data` (a sampler rng -> (X, Y))."""
    lr = 0.0 if MUTANT == "zero_lr" else lr
    opt = Adam(model["params"], lr)
    hist = []
    for _ in range(budget):
        loss, grads, mse = loss_and_grads(model, data(rng))
        if not np.isfinite(loss):
            raise FloatingPointError("non-finite loss")
        grads, _ = clip_global_norm(grads, CLIP)
        opt.step(model["params"], grads, sign=-1.0 if MUTANT == "sign_flip" else 1.0)
        hist.append(mse)
    return hist


def make_suite(seed, n_eval=N_EVAL):
    ss = np.random.SeedSequence([316, seed])
    r = [np.random.default_rng(s) for s in ss.spawn(5)]
    plan = r[0].normal(size=(P, D_TRUE))
    train_w = make_worlds(r[1], plan, N_TRAIN_WORLDS, "identity")
    ev = {"heldout": make_worlds(r[2], plan, n_eval, "identity"),
          "shifted": make_worlds(r[2], plan, n_eval, "random"),
          "blind": make_worlds(r[2], plan, n_eval, "random", PHI_PALIMPSEST)}
    walks = {k: sample_walks(r[3], w, n_eval, which=np.arange(n_eval)) for k, w in ev.items()}
    return {"plan": plan, "train": train_w, "eval": ev, "walks": walks, "rng": r[4]}


def sampler_for(worlds):
    def draw(rng):
        b = sample_walks(rng, worlds, BATCH)
        return b["X"], b["Y"]
    return draw


def shuffled_worlds(worlds, rng):
    """C4: content permuted across cells within each world, so the plan carries no signal."""
    w = dict(worlds)
    w["content"] = np.stack([c[rng.permutation(P)] for c in worlds["content"]])
    return w


def metrics(model, walk):
    pred = predict(model, walk["X"])
    first = key_structure(walk["X"][:, :, :P])[0]
    first[:, 0] = False
    err = np.mean((pred - walk["Y"]) ** 2, axis=-1)
    rf = first & walk["refash"]
    return {"fv": float(err[first].mean()), "all": float(err[:, 1:].mean()),
            "refash_fv": float(err[rf].mean()) if rf.any() else float("nan"),
            "trivial_fv": float(np.mean(walk["Y"][first] ** 2))}


def train_pair(suite, steps, seed_rng):
    lal = build_model(P + D_OBS, D_OBS, rng=seed_rng, kind="lalibela")
    att = build_model(P + D_OBS, D_OBS, rng=seed_rng, kind="attention")
    samp = sampler_for(suite["train"])
    h_lal = fit(lal, samp, steps, np.random.default_rng(seed_rng.integers(1 << 31)))
    h_att = fit(att, samp, steps, np.random.default_rng(seed_rng.integers(1 << 31)))
    return lal, att, h_lal, h_att


# ----------------------------------------------------------------------------- correctness tests
def c1_gradcheck(models, rng, tol):
    rows, worst = [], 0.0
    batch = sampler_for(make_suite(99, n_eval=4)["train"])(np.random.default_rng(7))
    batch = (batch[0][:6], batch[1][:6])
    for tag, m in models:
        _, g, _ = loss_and_grads(m, batch)

        def lf(params, m=m):
            return loss_and_grads(dict(m, params=params), batch)[0]
        res = finite_difference_check(lf, m["params"], g, rng)
        for k, v in res.items():
            worst = max(worst, v["max_rel_error"])
            rows.append((tag, k, v["entries"], f"{v['max_rel_error']:.2e}"))
    return worst <= tol, worst, rows


def c3_learning(model, hist, suite, thr):
    first, last = float(np.mean(hist[:20])), float(np.mean(hist[-20:]))
    drop = 1.0 - last / first
    m = metrics(model, suite["walks"]["heldout"])
    ok = drop >= thr["loss_drop_fraction"] and m["fv"] <= m["trivial_fv"] - thr["margin_over_trivial"]
    return ok, f"train-MSE drop {drop:.2f}; held-out first-visit MSE {m['fv']:.3f} vs trivial {m['trivial_fv']:.3f}"


def c4_shuffled(suite, steps, rng, thr):
    m = build_model(P + D_OBS, D_OBS, rng=rng, kind="lalibela")
    fit(m, sampler_for(shuffled_worlds(suite["train"], rng)), steps, rng)
    r = metrics(m, suite["walks"]["heldout"])
    ok = r["fv"] >= r["trivial_fv"] - thr["shuffled_band"]
    return ok, f"shuffled-plan model first-visit MSE {r['fv']:.3f} vs trivial {r['trivial_fv']:.3f}"


def random_model(rng, **kw):
    m = build_model(P + D_OBS, D_OBS, rng=rng, kind="lalibela")
    m["params"]["log_s2"][:] = kw.get("log_s2", -2.0)
    m.update({k: v for k, v in kw.items() if k in ("prior", "nc_tanh")})
    return m


def c6_properties(rng):
    """Invariants on random parameters and random worlds, each with a negative control."""
    out = []
    plan = rng.normal(size=(P, D_TRUE))
    walk = sample_walks(rng, make_worlds(rng, plan, 12, "random"), 12, which=np.arange(12))
    X = walk["X"]
    # C6.1 orthogonal equivariance of predictions in observation space (searched over 15 rotations)
    def c61(m):
        worst = 0.0
        for _ in range(15):
            B = np.linalg.qr(rng.normal(size=(D_OBS, D_OBS)))[0]
            Xb = X.copy()
            Xb[:, :, P:] = X[:, :, P:] @ B.T
            worst = max(worst, float(np.max(np.abs(predict(m, Xb) - predict(m, X) @ B.T))))
        return worst
    good, bad = c61(random_model(rng)), c61(random_model(rng, nc_tanh=True))
    out.append(("C6.1", "prediction commutes with orthogonal maps of the observation space",
                good < 1e-8 and bad > 1e-3, f"violation {good:.1e}; negative control {bad:.1e}"))
    # C6.2 invariance to how the terrain is turned (group closure plus uniform prior)
    def c62(m):
        worst = 0.0
        for g in range(1, 8):
            Xg = X.copy()
            Xg[:, :, :P] = X[:, :, :P][:, :, INV[g]]
            worst = max(worst, float(np.max(np.abs(predict(m, Xg) - predict(m, X)))))
        return worst
    good = c62(random_model(rng))
    bad = c62(random_model(rng, prior=np.array([0.65] + [0.05] * 7)))
    out.append(("C6.2", "prediction is invariant to a dihedral turn of the terrain",
                good < 1e-8 and bad > 1e-3, f"violation {good:.1e}; negative control {bad:.1e}"))
    # C6.3 and C6.4: definition checks (labelled; not evidence)
    m = random_model(rng)
    pred, post = predict(m, X, return_post=True)
    K, V, first, src, _ = split_io(X, P)
    rec = np.take_along_axis(V, src[..., None], axis=1)
    rev = ~first
    rev[:, 0] = False
    out.append(("C6.3", "[definition] revisits return the stored value", bool(np.allclose(pred[rev], rec[rev])), "exact recall"))
    out.append(("C6.4", "[definition] registration posterior sums to one",
                bool(np.allclose(post.sum(axis=0), 1.0)), "normalised"))
    out.append(("C6.5",) + c65_recovery(rng))
    return out


def c65_recovery(rng, thr=0.95):
    """Non-gradient rule: on noise-free worlds built from the model's own plan, the posterior
    after all revealed cells puts >= 0.99 on the true turn in >= 95% of worlds."""
    m = random_model(rng)
    plan = m["params"]["W"]
    A = rng.normal(size=(40, D_OBS, D_PLAN)) / math.sqrt(D_PLAN)
    worlds = {"A": A, "content": np.einsum("wdk,pk->wpd", A, plan), "refash": np.zeros((40, P), bool),
              "h": rng.integers(0, 8, size=40)}
    walk = sample_walks(rng, worlds, 40, which=np.arange(40))
    _, post = predict(m, walk["X"], return_post=True)
    last = post[:, :, -1]
    hit = last[worlds["h"], np.arange(40)] >= 0.99
    rate = float(hit.mean())
    return ("registration posterior recovers the true turn on clean worlds", rate >= thr, f"recovery rate {rate:.2f}")


def c7_split(suite):
    h = lambda a: hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()
    tr = {h(a) for a in suite["train"]["A"]}
    ev = {h(a) for w in suite["eval"].values() for a in w["A"]}
    return len(tr & ev) == 0, f"{len(tr)} training worlds, {len(ev)} evaluation worlds, overlap {len(tr & ev)}"


def c2_determinism(steps=15):
    runs = []
    for _ in range(2):
        rng = np.random.default_rng(2024)
        suite = make_suite(3, n_eval=10)
        m = build_model(P + D_OBS, D_OBS, rng=rng)
        h = fit(m, sampler_for(suite["train"]), steps, rng)
        runs.append((np.array(h), predict(m, suite["walks"]["shifted"]["X"]), m))
    same = np.array_equal(runs[0][0], runs[1][0]) and np.array_equal(runs[0][1], runs[1][1])
    finite = all(np.all(np.isfinite(v)) for v in runs[0][2]["params"].values()) and np.all(np.isfinite(runs[0][1]))
    return same and finite, finite, f"identical histories and outputs: {same}; finite: {finite}"


def correctness_core(steps, thr, suite, stop_early=False):
    """C1 (init), C6.5, C3 and C4 on fresh models: the audit battery run for every mutant.
    With stop_early the battery ends at the first failure (a mutant is caught once)."""
    res = {"C1": c1_gradcheck([("lalibela@init", build_model(P + D_OBS, D_OBS, rng=np.random.default_rng(11)))],
                              np.random.default_rng(12), thr["gradcheck_rel_tol"])[0],
           "C6.5": c65_recovery(np.random.default_rng(16))[1]}
    if stop_early and not all(res.values()):
        return res
    try:
        m = build_model(P + D_OBS, D_OBS, rng=np.random.default_rng(13))
        res["C3"] = c3_learning(m, fit(m, sampler_for(suite["train"]), steps, np.random.default_rng(14)), suite, thr)[0]
        if res["C3"] or not stop_early:
            res["C4"] = c4_shuffled(suite, steps, np.random.default_rng(15), thr)[0]
    except (FloatingPointError, np.linalg.LinAlgError):
        res["C3"] = False
    return res


def c5_mutation(steps, thr, suite, base):
    """`base` holds the unmutated battery's results from the main run (same functions, same
    thresholds); each mutant reruns the battery on fresh models until something fails."""
    global MUTANT
    rows, detected = [], 0
    for name in MUTANTS:
        MUTANT = name
        try:
            res = correctness_core(steps, thr, suite, stop_early=True)
        finally:
            MUTANT = None
        caught = [k for k, v in res.items() if not v]
        detected += bool(caught)
        rows.append((name, ",".join(caught) or "-", "detected" if caught else "MISSED"))
    return all(base.values()) and detected == len(MUTANTS), detected, rows, base


def correctness(runs, steps, audit_steps, thr, kids):
    tol, r0 = thr["gradcheck_rel_tol"], runs[0]
    fresh = [(f"{k}@init", build_model(P + D_OBS, D_OBS, rng=np.random.default_rng(11), kind=k))
             for k in ("lalibela", "attention")]
    trained = [(f"lalibela@{steps}", r0["lal"]), (f"attention@{steps}", r0["att"])]
    ok_i, w_i, rows_i = c1_gradcheck(fresh, np.random.default_rng(12), tol)
    ok_t, w_t, rows_t = c1_gradcheck(trained, np.random.default_rng(12), tol)
    grad = {"tensors_checked": len(rows_i) + len(rows_t),
            "tensors_total": sum(len(m["params"]) for _, m in fresh + trained), "max_rel_error": max(w_i, w_t),
            "checked_at": ["init", f"after_{steps}_training_steps"], "passed": bool(ok_i and ok_t)}
    rows = [("C1", "gradient_check", grad["passed"], f"{grad['tensors_checked']}/{grad['tensors_total']} tensors, "
             f"max relative error {grad['max_rel_error']:.1e} (tolerance {tol:g}), at init and after {steps} steps")]
    ok2, _, det2 = c2_determinism()
    rows.append(("C2", "determinism_and_finiteness", bool(ok2), det2))
    c3 = [c3_learning(r["lal"], r["hist"], r["suite"], thr) for r in runs]
    rows.append(("C3", "learning", all(c[0] for c in c3), f"{sum(c[0] for c in c3)}/{len(c3)} seeds pass; seed {r0['seed']}: {c3[0][1]}"))
    ok4, det4 = c4_shuffled(r0["suite"], steps, kids[0], thr)
    rows.append(("C4", "shuffled_label_control", ok4, det4))
    props = c6_properties(kids[1])
    mut = {"detected": 0, "total": len(MUTANTS), "score": None, "rows": []}
    if MUTANT is None:
        base = {"C1": ok_i, "C3": c3[0][0], "C4": ok4, "C6.5": props[-1][2]}
        ok5, det, mut["rows"], _ = c5_mutation(audit_steps, thr, r0["suite"], base)
        mut.update(detected=det, score=det / len(MUTANTS))
        rows.append(("C5", "mutant_detection", ok5, f"{det}/{len(MUTANTS)} mutants detected ({audit_steps}-step audit)"))
    else:
        rows.append(("C5", "mutant_detection", None, f"skipped: running under mutant '{MUTANT}'"))
    rows.append(("C6", "property_tests", all(p[2] for p in props), f"{len(props)} checks, "
                 f"{sum(1 for p in props if not p[1].startswith('[definition]'))} with negative controls"))
    c7 = [c7_split(r["suite"]) for r in runs]
    rows.append(("C7", "split_integrity", all(c[0] for c in c7), c7[0][1] + f" (all {len(c7)} seeds checked)"))
    return rows, props, grad, mut


# ----------------------------------------------------------------------------- hypothesis tests
SPLITS = ("heldout", "shifted", "blind")
HYP_DIFF = {
    "H-SIG": lambda r: r["metrics"]["shifted"]["attention"]["fv"] - r["metrics"]["shifted"]["lalibela"]["fv"],
    "H-NEC": lambda r: r["ko"]["registration"] - r["ko"]["recall"],
    "H-BLIND": lambda r: r["metrics"]["blind"]["lalibela"]["refash_fv"] - r["metrics"]["blind"]["attention"]["refash_fv"],
    "H-RIVAL": lambda r: r["metrics"]["blind"]["lalibela"]["fv"] - r["metrics"]["blind"]["rival"]["fv"],
}


def run_seed(seed, steps, n_eval):
    suite = make_suite(seed, n_eval)
    lal, att, hist, _ = train_pair(suite, steps, suite["rng"])
    if not all(np.all(np.isfinite(v)) for m in (lal, att) for v in m["params"].values()):
        raise FloatingPointError(f"non-finite parameters after training (seed {seed})")
    rival = build_rival(P + D_OBS, D_OBS)
    met = {s: {n: metrics(m, suite["walks"][s]) for n, m in (("lalibela", lal), ("attention", att), ("rival", rival))}
           for s in SPLITS}
    ko = {n: metrics(knockout(lal, n), suite["walks"]["shifted"])["all"] for n in KNOCKOUT_MODES}
    # The plan is learned only up to a turn: the guided tours fix the model's own frame, read off
    # as the modal registration on held-out (unturned) worlds; shifted turns are scored in that frame.
    post_end = {s: hidden_states(lal, suite["walks"][s]["X"])[:, -1, -len(PERMS):] for s in ("heldout", "shifted")}
    frame = int(np.bincount(np.argmax(post_end["heldout"], axis=1), minlength=len(PERMS)).argmax())
    turn = COMPOSE[suite["eval"]["shifted"]["h"][suite["walks"]["shifted"]["world"]], frame]
    return {"seed": seed, "suite": suite, "lal": lal, "att": att, "hist": hist, "metrics": met, "ko": ko,
            "frame": frame, "turn_mass": float(post_end["shifted"][np.arange(len(turn)), turn].mean())}


def evaluate_hypotheses(runs, rng, quick):
    out = []
    for h in MIND_CARD["hypotheses"]:
        d = [HYP_DIFF[h["id"]](r) for r in runs]
        mean, ci = paired_bootstrap(d, rng)
        ok = not quick and len(d) >= h["seeds"]
        out.append({"id": h["id"], "comparison": h["comparison"], "metric": h["metric"], "mean_diff": mean,
                    "ci95": list(ci) if ok else None, "mesi": h["mesi"], "n_seeds": len(d),
                    "verdict": verdict(mean, ci, h["mesi"], h["direction"]) if ok else "not evaluated"})
    return out


def knockout_table(runs, rng):
    full = [r["metrics"]["shifted"]["lalibela"]["all"] for r in runs]
    reg = modules(runs[0]["lal"])
    out = []
    for name in KNOCKOUT_MODES:
        mean, ci = paired_bootstrap([r["ko"][name] - f for r, f in zip(runs, full)], rng)
        out.append({"module": name, "mode": KNOCKOUT_MODES[name], "signature": reg[name]["signature"],
                    "metric_change": mean, "ci95": list(ci)})
    return out


# ----------------------------------------------------------------------------- real-data bridge
def load_csv_worlds(path):
    """Header world,cell,v1..vD; cells 0..35 on a 6x6 layout; one row per world and cell.
    Values are standardised per column. Parsed with the standard library only."""
    with open(path, encoding="utf-8") as f:
        f.readline()
        arr = np.array([line.strip().split(",") for line in f if line.strip()], dtype=float)
    ids = np.unique(arr[:, 0])
    content = np.full((len(ids), P, arr.shape[1] - 2), np.nan)
    content[np.searchsorted(ids, arr[:, 0]), arr[:, 1].astype(int)] = arr[:, 2:]
    if np.isnan(content).any():
        raise ValueError("every world needs all 36 cells")
    return (content - content.mean(axis=(0, 1))) / (content.std(axis=(0, 1)) + 1e-9)


def data_bridge(path, steps, rng):
    if path is None:
        return "real-data bridge: skipped (no --data given)"
    if not os.path.exists(path):
        return f"real-data bridge: skipped (file not found: {path})"
    content = load_csv_worlds(path)
    n, d = content.shape[0], content.shape[-1]
    if n < 3:
        return "real-data bridge: skipped (needs at least three worlds)"
    worlds = lambda c: {"content": c, "refash": np.zeros(c.shape[:2], bool), "h": np.zeros(c.shape[0], int)}
    lal, att = build_model(P + d, d, rng=rng), build_model(P + d, d, rng=rng, kind="attention")
    for m in (lal, att):
        fit(m, sampler_for(worlds(content[:2])), steps, rng)
    walk = sample_walks(rng, worlds(content[2:]), 60)
    ml, ma = metrics(lal, walk), metrics(att, walk)
    return (f"real-data bridge ({os.path.basename(path)}): trained on 2 worlds, tested on {n - 2}; first-visit MSE "
            f"lalibela {ml['fv']:.3f}, attention {ma['fv']:.3f}, trivial {ml['trivial_fv']:.3f}")


# ----------------------------------------------------------------------------- report
def build_report(seeds, runs, corr, grad, mut, hyps, kos, runtime):
    failed = [r[0] for r in corr if r[2] is False]
    code = 1 if any(f != "C8" for f in failed) else (3 if failed else 0)
    fl = lambda x: None if x is None else float(x)
    return {"schema_version": "1.0", "chapter": MIND_CARD["id"], "file": os.path.basename(__file__),
            "card_revision": MIND_CARD["card_revision"],
            "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
            "seeds": seeds, "runtime_s": round(runtime, 2), "n_params": n_params(runs[0]["lal"]),
            "gradcheck": dict(grad, max_rel_error=float(grad["max_rel_error"])),
            "correctness": [{"id": i, "name": n, "passed": None if p is None else bool(p), "detail": d}
                            for i, n, p, d in corr],
            "mutants": {"detected": mut["detected"], "total": mut["total"], "score": fl(mut["score"])},
            "hypotheses": [{"id": h["id"], "metric": h["metric"], "mean_diff": float(h["mean_diff"]),
                            "ci95": h["ci95"], "mesi": h["mesi"], "n_seeds": h["n_seeds"], "verdict": h["verdict"]}
                           for h in hyps],
            "knockouts": [{"module": k["module"], "signature": k["signature"], "metric_change": float(k["metric_change"]),
                           "ci95": k["ci95"]} for k in kos],
            "task_types": TASK_TYPES, "exit_code": code}


def print_report(rep, runs, props, mut, hyps, kos, bridge, steps, audit_steps, budget):
    na, nb = n_params(runs[0]["lal"]), n_params(runs[0]["att"])
    mean = lambda s, n, k: float(np.mean([r["metrics"][s][n][k] for r in runs]))
    f3 = lambda x: f"{x:+.3f}"
    ci = lambda c: "n/a" if c is None else f"[{c[0]:+.3f}, {c[1]:+.3f}]"
    g = rep["gradcheck"]
    out = [f"=== VERIFIED REPORT · chapter {rep['chapter']:04d} ===",
           f"file: {rep['file']} · card revision {rep['card_revision']} · {MIND_CARD['mechanism']['name']}",
           f"environment: Python {rep['environment']['python']}, NumPy {rep['environment']['numpy']}",
           f"seeds: {rep['seeds']} (worlds and training from SeedSequence([316, seed])); fixed audit generators 7, 11-16, 99, 2024",
           f"training: {steps} Adam steps per model and seed (batch {BATCH}, lr 0.03, global-norm clip {CLIP}); "
           f"mutant audit {audit_steps} steps",
           f"runtime: {rep['runtime_s']:.1f} s (budget {budget:.0f} s)",
           f"parameters: third-copy network {na}; attention baseline {nb} ({100 * abs(na - nb) / na:.1f}% difference); "
           "rival 0169 has none",
           f"gradient check: {g['tensors_checked']}/{g['tensors_total']} tensors, max relative error "
           f"{g['max_rel_error']:.2e}, at {' and '.join(g['checked_at']).replace('_', ' ')}: "
           f"{'passed' if g['passed'] else 'FAILED'}",
           "", "correctness",
           format_table(["id", "test", "passed", "detail"],
                        [(c["id"], c["name"], {True: "yes", False: "NO", None: "-"}[c["passed"]], c["detail"])
                         for c in rep["correctness"]]),
           "", "property tests (C6; definition checks are labelled and are not evidence)",
           format_table(["id", "property", "passed", "detail"], [(p[0], p[1], "yes" if p[2] else "NO", p[3]) for p in props]),
           "", f"mutation score: {mut['detected']}/{mut['total']}" + ("" if mut["score"] is None else f" = {mut['score']:.2f}")]
    if mut["rows"]:
        out.append(format_table(["mutant", "failing tests", "status"], mut["rows"]))
    out += ["", "hypotheses (paired per-seed differences; 95% percentile bootstrap, 2000 resamples)",
            format_table(["id", "comparison", "metric", "mean diff", "95% CI", "mesi", "seeds", "verdict"],
                         [(h["id"], h["comparison"], h["metric"], f3(h["mean_diff"]), ci(h["ci95"]), h["mesi"],
                           h["n_seeds"], h["verdict"]) for h in hyps]),
            "", "knockouts (shifted split, all-step MSE per dimension; change versus the full model)",
            format_table(["module", "mode", "signature", "change", "95% CI"],
                         [(k["module"], k["mode"], "yes" if k["signature"] else "no", f3(k["metric_change"]),
                           ci(k["ci95"])) for k in kos]),
            "", f"first-visit MSE per dimension, mean over {len(runs)} seed(s)",
            format_table(["split", "third-copy", "attention", "rival 0169", "trivial (predict 0)"],
                         [(s, f"{mean(s, 'lalibela', 'fv'):.3f}", f"{mean(s, 'attention', 'fv'):.3f}",
                           f"{mean(s, 'rival', 'fv'):.3f}", f"{mean(s, 'lalibela', 'trivial_fv'):.3f}") for s in SPLITS]
                         + [("blind, refashioned cells", f"{mean('blind', 'lalibela', 'refash_fv'):.3f}",
                             f"{mean('blind', 'attention', 'refash_fv'):.3f}", f"{mean('blind', 'rival', 'refash_fv'):.3f}", "-")]),
            f"registration (shifted, after {T_WALK} steps, scored in the frame fixed by the tours; frames "
            f"{[r['frame'] for r in runs]}): mean posterior mass on the true turn "
            f"{np.mean([r['turn_mass'] for r in runs]):.3f}; chance {1 / len(PERMS):.3f}",
            bridge, f"task types: {', '.join(rep['task_types'])}", f"exit code: {rep['exit_code']}",
            "=== END REPORT ==="]
    print("\n".join(out))


# ----------------------------------------------------------------------------- entry point
def protocol(a, rng, kids):
    t0 = time.time()
    steps, audit_steps, n_eval = (50, 50, 60) if a.quick else (200, 60, N_EVAL)
    seeds = [a.seed + i for i in range(1 if a.quick else a.seeds)]
    print(f"seeds {seeds}; running {'quick' if a.quick else 'full'} protocol"
          + (f" under mutant '{MUTANT}'" if MUTANT else ""), flush=True)
    runs = [run_seed(s, steps, n_eval) for s in seeds]
    corr, props, grad, mut = correctness(runs, steps, audit_steps, MIND_CARD["thresholds"], kids)
    hyps = evaluate_hypotheses(runs, kids[2], a.quick)
    kos = knockout_table(runs, kids[2])
    bridge = data_bridge(a.data, steps, rng)
    budget = 20.0 if a.quick else BUDGET_S
    runtime = time.time() - t0
    corr.append(("C8", "budget", runtime <= budget, f"{runtime:.1f} s of {budget:.0f} s"))
    rep = build_report(seeds, runs, corr, grad, mut, hyps, kos, runtime)
    print_report(rep, runs, props, mut, hyps, kos, bridge, steps, audit_steps, budget)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2, ensure_ascii=False)
    return rep["exit_code"]


def main(argv=None):
    global MUTANT
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__),
                                 description="Third-copy revelation network (chapter 0316): full protocol by default.")
    ap.add_argument("--quick", action="store_true", help="one seed, reduced steps, all correctness tests")
    ap.add_argument("--seed", type=int, default=0, help="base seed")
    ap.add_argument("--seeds", type=int, default=5, help="number of seeds (hypotheses need 5)")
    ap.add_argument("--json", metavar="PATH", help="also write the JSON report")
    ap.add_argument("--card", action="store_true", help="print MIND_CARD as JSON and exit")
    ap.add_argument("--mutant", choices=sorted(MUTANTS), help="run with a registered mutant")
    ap.add_argument("--data", metavar="PATH", help="optional CSV bridge: world,cell,v1..vD")
    a = ap.parse_args(argv)
    if a.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    if a.seeds < 1:
        ap.error("--seeds must be at least 1")
    MUTANT = a.mutant
    ss = np.random.SeedSequence(a.seed)
    rng = np.random.default_rng(ss)
    try:
        return protocol(a, rng, [np.random.default_rng(s) for s in ss.spawn(3)])
    except FloatingPointError as err:
        print(f"non-finite values: {err}")
        return 4


if __name__ == "__main__":
    sys.exit(main())

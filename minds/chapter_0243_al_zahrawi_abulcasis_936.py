#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 13 Minds 241 - 260 Available on Amazon https://www.amazon.com/dp/B0HLD1S6QT
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0243_al_zahrawi_abulcasis_936 - Al-Zahrawi (Abulcasis) (c.936-1013)
#================================================================================  
"""MIKWA: anatomy-guarded compact-support corrective units (research prototype, not an AGI).

Thesis
  A correction is judged by how far its effect travels: deliver it through an agent whose
  action stays on the ailing part, mark its extent before applying it, and stop its reach
  short of the structures that anatomy says lie beside it.

Evidence and provenance (provenance: belief; no treatise on the soul or mind survives)
  D1 primary   Tasrif XXX, Book I ch. 1: fire acts only on the cauterized part and harms the
               neighbouring part slightly; a caustic may spread to distant parts; fire has no
               such effect unless overdone (tr. Spink & Lewis 1973; Hamarneh 1961).
  D2 primary   Book I ch. 1: iron preferred to gold, because gold's redness hides when it has
               reached the right heat, it cools quickly and it melts when overheated.
  D3 primary   Book I chs. 16, 26-27: the burn site is first marked in ink in a set shape
               (a myrtle leaf; marks below the rib cartilage); type, place and number of irons
               are given per case (Hamarneh 1961, figs. 2, 4, 5).
  D4 primary   Preface to Treatise XXX and Book II: without anatomy the operator falls into
               errors that kill; incision can damage an artery or vein.
  D5 primary   Book III ch. 1: bandages made less and less tight with distance from the
               injury; neighbouring parts padded against the edges of splints.
  D6 primary   Book I ch. 1: a complaint cured by cautery can come back; cautery follows
               treatments by drugs that have failed.
  D7 scholarship  Bos & Kas 2025: twenty-eight cautery types against at most ten in the Greek
               tradition; the surgery rests mainly on Paul of Aegina and describes an ideal.
  D8 speculation  This chapter's mapping: fire -> compact support; caustic -> Gaussian tail;
               internal drug -> shared-weight fine-tuning; anatomy -> verified anchors.

Doctrine -> mechanism -> test
  D1     M1 iron: phi = (1 - s^4)^3 for s < 1, exactly 0 outside        C6.2 (definition), H-SIG
  D4 D5  M2 padding: width capped by a power soft-min over anchors      C6.1, H-NEC
  D3 D7  M3 ink and irons: marking from failing cases; ball-to-shell    C6.4, H-IRONS
  D2     M4 legible dose: penalty on log support extent and heat         (regulariser, no test)
  D6     blind spot: a systemic defect seen at one site                  H-BLIND

Research question (continual learning and plasticity; interpretability and modularity)
  Can a post-deployment correction be carried by units whose functional support is compact
  and provably stops short of verified cases, so that fixing a local failure does not spread
  into neighbouring competence, and where does confinement itself become the failure?

Closest prior art and delta
  RBF adaptors and resource-allocating networks (Moody & Darken 1989; Platt 1991), compactly
  supported radial functions (Wendland 1995), deferral-radius key-value editors (GRACE,
  Hartvigsen et al. 2023), scope-classifier editors (SERAC, Mitchell et al. 2022), fine-tuning
  with rehearsal. Delta: each unit's width is capped by a smooth soft-minimum that can never
  exceed the true minimum over verified anchors, so for every parameter value no anchor lies
  inside any support, and a learnable hole lets a support deform from ball to shell.

Blind spot
  A systemic defect (one class remapped everywhere, observed at a single site). Confined units
  cannot carry the cure to unobserved sites, while a shared-weight update can. Grounded in the
  practice of burning skin sites for internal disease and in his warning that complaints return.

Task
  u ~ U[-1,1]^3 body coordinates; n ~ N(0, 0.05^2 I_3) nuisance; x = [u, n].
  y_old = argmax of a random 3-16-4 tanh teacher on u (class frequencies 0.12-0.42).
  Four sites near tetrahedral points: two solid balls, one shell around a healthy core and one
  elongated leaf; inside a site y_new = (y_old + k) mod 4 with k in {1, 2, 3}.
  Correction data: 24 new-label cases per site and about 190 verified old-world anchors in the
  shells, the ring core and the far field, none inside a thin band around each boundary.
  Held-out: cure inside sites, retention in shells, core and far field. Shifted: cure and
  retention inside the anchor-free bands, with nuisance scale x1.5.
  Blind episode: one base class remapped everywhere, observed only inside one ball.

Limits
  Synthetic 6-D data, 7 units, a frozen 32-unit body. The guard protects listed anchors only;
  unlisted cases inside a support can still change. Nothing here gives surgical or medical
  guidance, and results say nothing about the historical person.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 3,
    "revision_log": [
        "r2 (debug runs of C1-C4 only, before any hypothesis test): nuisance sd 0.10 -> 0.05, because 6-D nuisance distances swamped "
        "the 0.05-wide anchor-free bands; iron profile (1-s)^3 -> (1-s^4)^3, because most of a 6-D support sat near its edge "
        "and stayed under-dosed; heat penalty 1e-3 -> 1e-5, dose penalty 5e-4 -> 1e-4 and dose initialised from body logits, "
        "because the trained body's ~30-nat margins let the old penalty dominate; a mark-coverage term added for compact and "
        "Gaussian units alike, because compact units give no gradient to untouched cases (D3: the ink mark precedes fire and "
        "caustic); C4 cure band changed from per-site majority (vacuous, above 1) to the pooled-majority or persistence baseline; "
        "a quick smoke run showed the ablations inheriting profile 'none' from the body and training no units, fixed so every "
        "variant starts from the full configuration. Hypotheses, metrics and mesi unchanged.",
        "r3 (after the r2 verified run): the padding soft-min used log(D^2 + 1e-9), so an anchor within ~3e-5 of a shell ring "
        "(q = beta) got a cap above its own distance and sat inside the support (phi = 1 in 200/200 adversarial trials); C6.1 never "
        "placed anchors on rings. The soft-min is now taken relative to the exact minimum with no epsilon, widths are floored at "
        "1e-12 with s = (D^2 + 1e-24)/w^2, and C6.1 places one anchor on a shell ring per trial. Hypotheses, metrics and mesi unchanged."],
    "generation": {"template_version": "codeguidelines 1.0 (2026-09-15)", "generator": "Claude",
                   "generator_version": "claude-opus-5", "date": "2026-09-15"},
    "id": 243, "figure": "Al-Zahrawi (Abulcasis)", "born": 936, "died": 1013,
    "civilization": "Andalusi (Umayyad Cordoba to the early Taifa period; Arabic-writing, ancestry unrecorded)",
    "provenance": "belief",
    "thesis": ("A correction is judged by how far its effect travels: deliver it through an agent whose action "
               "stays on the ailing part, mark its extent first, and stop its reach short of the structures "
               "anatomy says lie beside it."),
    "evidence": [
        {"id": "D1", "basis": "primary", "source": "al-Tasrif XXX, Book I ch. 1 (Spink & Lewis 1973; Hamarneh 1961, MSS Vel. 2491 fol. 106, Besir Aga 502 fols. 523r-524v)",
         "claim": "Fire acts only on the cauterized part and slightly on its neighbour; caustics may spread to distant parts; fire does not unless overdone."},
        {"id": "D2", "basis": "primary", "source": "al-Tasrif XXX, Book I ch. 1 (Spink & Lewis 1973)",
         "claim": "Iron preferred to gold: gold's colour hides its heat, it cools fast and melts if overheated."},
        {"id": "D3", "basis": "primary", "source": "al-Tasrif XXX, Book I chs. 16, 26-27 (Hamarneh 1961, figs. 2, 4, 5)",
         "claim": "The burn site is marked in ink in a set shape before the iron; tool type, position and number are prescribed."},
        {"id": "D4", "basis": "primary", "source": "al-Tasrif XXX, preface and Book II (Hamarneh 1961; Spink & Lewis 1973, pp. 2-4)",
         "claim": "Without anatomy the operator falls into fatal errors; incisions endanger arteries and veins."},
        {"id": "D5", "basis": "primary", "source": "al-Tasrif XXX, Book III ch. 1 (Hamarneh 1961)",
         "claim": "Bandages are made less tight with distance from the injury; neighbouring parts are padded."},
        {"id": "D6", "basis": "primary", "source": "al-Tasrif XXX, Book I ch. 1 (Spink & Lewis 1973; Hamarneh 1961, n. 17)",
         "claim": "A complaint cured by cautery can return; cautery comes after drug treatment has failed."},
        {"id": "D7", "basis": "scholarship", "source": "Bos & Kas, 'Al-Zahrawi and His Medical Handbook', in Madinat al-Zahra (2025)",
         "claim": "Twenty-eight cautery types versus at most ten in Greek sources; the surgery follows Paul of Aegina and describes an ideal."},
        {"id": "D8", "basis": "speculation", "source": "this chapter",
         "claim": "Fire, caustic and internal drug map onto compact support, Gaussian tail and shared-weight fine-tuning."},
    ],
    "research_question": {"category": "continual learning and plasticity; interpretability and modularity",
                          "question": ("Can post-deployment corrections be carried by units whose support is compact and provably "
                                       "stops short of verified cases, and where does confinement become the failure?")},
    "mechanism": {
        "name": "MIKWA (the cautery iron): anatomy-guarded compact-support corrective units",
        "family": "local corrective adaptors for continual model editing",
        "signature_modules": ["iron", "padding", "irons"],
        "closest_prior_art": ["RBF adaptors and resource-allocating networks (Moody & Darken 1989; Platt 1991)",
                              "compactly supported radial functions (Wendland 1995)",
                              "GRACE discrete key-value adaptors with deferral radii (Hartvigsen et al. 2023)",
                              "SERAC scope-classifier editing (Mitchell et al. 2022)", "fine-tuning with rehearsal"],
        "overlap": "Medium",
        "prior_art_queries": ["compactly supported RBF catastrophic interference", "model editing locality radius",
                              "lifelong model editing deferral radius", "constraint-shaped kernel support protected points",
                              "RBF width set by distance to nearest protected data", "shell-shaped basis function hole parameter"],
        "contribution_type": "mechanism",
        "delta": ("Unit width is capped by a smooth power soft-min that never exceeds the true minimum over verified anchors, "
                  "so no anchor lies inside any support for any parameters; a learnable hole deforms supports from ball to shell."),
    },
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1 iron", "property_test": "C6.2", "hypothesis": "H-SIG"},
        {"doctrine": "D4", "mechanism": "M2 padding", "property_test": "C6.1", "hypothesis": "H-NEC"},
        {"doctrine": "D5", "mechanism": "M1 graded profile", "property_test": "C6.3", "hypothesis": "H-SIG"},
        {"doctrine": "D3", "mechanism": "M3 ink and irons", "property_test": "C6.4", "hypothesis": "H-IRONS"},
        {"doctrine": "D6", "mechanism": "M1 iron", "property_test": "none", "hypothesis": "H-BLIND"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "Fire beats caustic where neighbours sit closest: compact units keep more of the anchor-free band while curing.",
         "metric": "surgical_shift = mean(cure, band retention) on the shifted split", "split": "shifted",
         "comparison": "mikwa - caustic_rbf", "keys": ["mikwa/surgical_shift", "caustic_rbf/surgical_shift"],
         "direction": "greater", "mesi": 0.03, "seeds": 5},
        {"id": "H-NEC", "statement": "The padding cap is necessary: removing it before training costs more than removing the dressing offset.",
         "metric": "surgical_shift", "comparison": "no_padding - no_dressing (retrained knockouts)",
         "keys": ["no_padding/surgical_shift", "no_dressing/surgical_shift"], "direction": "less", "mesi": 0.03, "seeds": 5},
        {"id": "H-BLIND", "statement": "On a systemic defect seen at one site, confinement leaves remote sites uncured where a shared-weight drug does not.",
         "condition": "one class remapped everywhere; cases observed only inside one ball",
         "grounding": "local burns for internal disease; his warning that cured complaints return",
         "metric": "remote cure accuracy", "comparison": "mikwa - drug_finetune",
         "keys": ["sys/mikwa/remote_cure", "sys/drug_finetune/remote_cure"], "direction": "less", "mesi": 0.10, "seeds": 5},
        {"id": "H-IRONS", "statement": "A shell-capable iron cures a lesion ringing a healthy core better than olive-only irons.",
         "metric": "ring_surgical_shift (ring site only)", "comparison": "mikwa - olive_only (retrained)",
         "keys": ["mikwa/ring_surgical_shift", "olive_only/ring_surgical_shift"], "direction": "greater", "mesi": 0.03, "seeds": 5},
        {"id": "H-CAUSTIC", "statement": "On local lesions the confined iron beats the internal drug (output-layer fine-tuning with rehearsal).",
         "metric": "surgical_shift", "comparison": "mikwa - drug_finetune",
         "keys": ["mikwa/surgical_shift", "drug_finetune/surgical_shift"], "direction": "greater", "mesi": 0.03, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": {"body": 0.5, "correction": 0.4},
                   "margin_over_trivial": {"body_accuracy_over_majority": 0.35, "surgical_over_persistence": 0.15},
                   "shuffled_band": {"body_over_majority": 0.07, "cure_over_trivial": 0.10},
                   "gradcheck_rel_tol": 1e-5},
    "probe_predictions": [{"probe": "P1", "expected": "below baseline"}, {"probe": "P3", "expected": "below baseline"},
                          {"probe": "P5", "expected": "above baseline"}, {"probe": "P6", "expected": "above baseline"},
                          {"probe": "P8", "expected": "equal to baseline"}, {"probe": "P9", "expected": "below baseline"},
                          {"probe": "P10", "expected": "above baseline"}],
    "dialectic_links": [{"chapter": 135, "relation": "teacher", "test": "none: anatomy 'as Galen described' is inherited as the anchor set, no rival implemented"}],
    "corpus_neighbors": [
        {"chapter": 34, "similarity": None, "difference": "Susruta minimises incision size and reversibility; here the variable is the spread law of the agent, not its size"},
        {"chapter": 181, "similarity": None, "difference": "Sun Simiao orders interventions by harm and anchors points on the patient's reply; here anchors are verified cases that cap reach"},
        {"chapter": 233, "similarity": None, "difference": "al-Razi keeps a counterexample register beside a frozen model; here corrections are geometric units with a support guarantee"},
        {"chapter": 331, "similarity": None, "difference": "Ibn al-Nafis bounds function by channel patency; here the bound is on the intervention's footprint"},
        {"chapter": 135, "similarity": None, "difference": "Galen maps function by lesioning conduits; here lesions are cured without touching neighbours"},
        {"chapter": 240, "similarity": None, "difference": "al-Mutanabbi re-measures tasks relative to capability; no locality or support constraint"},
    ],
    "barometer": {
        "cognitive_processing": ["P6 few-shot cure from 24 cases per site", "P10 learning curve of the correction"],
        "embodied_cognition": ["not measured: no body or control loop in this file"],
        "world_modeling": ["anchor set as a map of verified structure; systemic-defect episode"],
        "consciousness": ["not measured beyond knockout self-auditing; no claim about experience"],
        "language_understanding": ["not measured"],
        "emotional_intelligence": ["not measured"],
        "creativity": ["not measured"],
        "autonomy": ["P5 continual correction without forgetting verified behaviour"],
    },
    "task_types": ["vector_classification"],
    "applications": [
        {"use": "Patch a deployed robot sensor classifier in one failure region without moving verified cases", "sector": "robotics",
         "dataset": "UCI Wall-Following Robot Navigation"},
        {"use": "Research decision support: recalibrate a risk model for one subgroup while verified cases stay fixed (no dosing or treatment advice)",
         "sector": "healthcare research", "dataset": "UCI Heart Disease (Cleveland)"},
        {"use": "Correct a defect classifier after a process change confined to one operating region", "sector": "semiconductor manufacturing",
         "dataset": "UCI SECOM"},
    ],
    "safety_notes": ("Medical figure: the file models an epistemic doctrine about how far interventions spread, on synthetic data; "
                     "it gives no surgical, cautery, dosing or treatment guidance. The guard protects listed anchors only and "
                     "is not a safety certificate for unlisted cases."),
}

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

D_BODY, D_NUIS, N_CLASS, HIDDEN, UNITS = 3, 3, 4, 32, 7
DIM = D_BODY + D_NUIS
SIGMA_NUIS, SHIFT_SCALE = 0.05, 1.5
BAND, SHELL, HOLE = 0.15, 0.60, 0.50
GAMMA, P_GUARD, P_CLAMP, BETA_MAX = 0.90, 12.0, 8.0, 0.90
EPS_Q, W_MIN = 1e-6, 1e-12
LAMBDA_DOSE, LAMBDA_HEAT = 1e-4, 1e-5
LINK, CLIP, LR_BODY, BATCH = 0.42, 5.0, 0.01, 128
MARK_WEIGHT, MARK_LEVEL, MARK_TEMP, MARK_SOFT = 1.0, 0.5, 0.05, 0.1
LR_GRID = (0.01, 0.04)
BUDGET = {"full": {"body": 1500, "corr": 500}, "quick": {"body": 400, "corr": 150}}
BODY_KEYS = ("W1", "b1", "W2", "b2")
BURN_KEYS = ("C", "R", "OM", "BE", "U", "WD")
TASK_TYPES = ["vector_classification"]
TETRA = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float) * 0.55
MUTANT = None


# BEGIN STANDARD UTILITIES v1.0
def softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def logsumexp(z, axis=-1):
    m = np.max(z, axis=axis, keepdims=True)
    return np.squeeze(m + np.log(np.exp(z - m).sum(axis=axis, keepdims=True)), axis=axis)


def softplus(x):
    return np.logaddexp(0.0, x)


class Adam:
    def __init__(self, params, keys, b1=0.9, b2=0.999, eps=1e-8):
        self.keys, self.b1, self.b2, self.eps, self.t = list(keys), b1, b2, eps, 0
        self.m = {k: np.zeros_like(params[k]) for k in self.keys}
        self.v = {k: np.zeros_like(params[k]) for k in self.keys}

    def step(self, params, grads, lr):
        self.t += 1
        for k in self.keys:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            mh, vh = self.m[k] / (1 - self.b1 ** self.t), self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= lr * mh / (np.sqrt(vh) + self.eps)


def clip_global_norm(grads, max_norm):
    total = float(np.sqrt(sum(float((g ** 2).sum()) for g in grads.values())))
    scale = min(1.0, max_norm / (total + 1e-12))
    for g in grads.values():
        g *= scale
    return total


def finite_difference_check(loss_fn, params, grads, keys, rng, eps=1e-6, n_random=20, tol=1e-5):
    report = {}
    for k in keys:
        flat, g = params[k].reshape(-1), grads[k].reshape(-1)
        idx = set(rng.choice(flat.size, size=min(n_random, flat.size), replace=False).tolist())
        idx.add(int(np.argmax(np.abs(g))))
        worst, ok = 0.0, True
        for i in sorted(idx):
            old = flat[i]
            flat[i] = old + eps
            lp = loss_fn()
            flat[i] = old - eps
            lm = loss_fn()
            flat[i] = old
            num, a = (lp - lm) / (2 * eps), g[i]
            scale = max(abs(a), abs(num))
            ok &= abs(a - num) <= tol * scale + 2e-9
            if scale >= 1e-4:
                worst = max(worst, abs(a - num) / scale)
        report[k] = {"checked": len(idx), "max_rel_error": float(worst), "passed": bool(ok)}
    return report


def paired_bootstrap(diffs, rng, n_boot=2000):
    d = np.asarray(diffs, float)
    means = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    return float(d.mean()), [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def verdict(mean, ci, mesi, direction):
    if direction == "greater":
        return "supported" if ci[0] > 0 and mean >= mesi else ("contradicted" if ci[1] < 0 else "inconclusive")
    return "supported" if ci[1] < 0 and mean <= -mesi else ("contradicted" if ci[0] > 0 else "inconclusive")


def write_report_json(path, report):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
# END STANDARD UTILITIES


# ------------------------------------------------------------------ data and tasks
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _teacher_labels(teacher, u):
    w1, b1, w2 = teacher
    return np.argmax(np.tanh(u @ w1.T + b1) @ w2.T, axis=1)


def make_teacher(rng):
    for _ in range(100):
        teacher = (rng.normal(0, 1.8, (16, D_BODY)), rng.normal(0, 0.6, 16), rng.normal(0, 1.0, (N_CLASS, 16)))
        freq = np.bincount(_teacher_labels(teacher, rng.uniform(-1, 1, (4000, D_BODY))), minlength=N_CLASS) / 4000
        if freq.max() < 0.42 and freq.min() > 0.12:
            break
    return teacher


def make_world(rng):
    kinds, sites, lesions = ["olive", "olive", "ring", "leaf"], rng.permutation(4), []
    teacher = make_teacher(rng)
    for kind, site in zip(kinds, sites):
        a = {"olive": np.full(D_BODY, rng.uniform(0.30, 0.36)), "ring": np.full(D_BODY, 0.38),
             "leaf": rng.permutation(np.array([0.46, 0.17, 0.17]))}[kind]
        lesions.append({"m": TETRA[site] + rng.uniform(-0.04, 0.04, D_BODY), "a": a, "kind": kind,
                        "h": HOLE if kind == "ring" else 0.0, "k": int(rng.integers(1, N_CLASS))})
    return {"teacher": teacher, "lesions": lesions, "systemic": None}


def _qnorm(u, lesion):
    return np.linalg.norm((u - lesion["m"]) / lesion["a"], axis=1)


def labels(world, u, new=True):
    y = _teacher_labels(world["teacher"], u)
    if not new:
        return y
    for lesion in world["lesions"]:
        q = _qnorm(u, lesion)
        inside = (q >= lesion["h"]) & (q < 1.0)
        y[inside] = (y[inside] + lesion["k"]) % N_CLASS
    if world["systemic"] is not None:
        c_star, k = world["systemic"]
        y = np.where(y == c_star, (c_star + k) % N_CLASS, y)
    return y


def _embed(rng, u, scale=1.0):
    return np.hstack([u, rng.normal(0.0, SIGMA_NUIS * scale, (len(u), D_NUIS))])


def _sample(rng, world, li, qmin, qmax, n):
    """Uniform samples in a normalised shell of lesion li that stay clear of the other lesions."""
    lesion, out, count = world["lesions"][li], [], 0
    while count < n:
        v = rng.normal(size=(4 * n, D_BODY))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        u = lesion["m"] + lesion["a"] * v * rng.uniform(qmin ** 3, qmax ** 3, 4 * n)[:, None] ** (1 / 3)
        keep = np.all(np.abs(u) <= 1.0, axis=1)
        for j, other in enumerate(world["lesions"]):
            keep &= (j == li) | (_qnorm(u, other) >= 1.0 + SHELL)
        out.append(u[keep])
        count += int(keep.sum())
    return np.vstack(out)[:n]


def _sample_far(rng, world, n, accept=None):
    out, count = [], 0
    while count < n:
        u = rng.uniform(-1, 1, (8 * n, D_BODY))
        keep = np.all([_qnorm(u, les) >= 1.0 + SHELL for les in world["lesions"]], axis=0) if world["lesions"] else np.ones(len(u), bool)
        if accept is not None:
            keep &= accept(u)
        out.append(u[keep])
        count += int(keep.sum())
    return np.vstack(out)[:n]


def _stack(parts):
    return {"X": np.vstack([p[0] for p in parts]), "y": np.concatenate([p[1] for p in parts]),
            "site": np.concatenate([np.full(len(p[1]), p[2]) for p in parts]),
            "role": np.concatenate([np.full(len(p[1]), p[3]) for p in parts])}


def make_episode(world, rng):
    u = rng.uniform(-1, 1, (3600, D_BODY))
    x, y = _embed(rng, u), labels(world, u, new=False)
    ep = {"base_train": (x[:2400], y[:2400]), "base_val": (x[2400:3000], y[2400:3000]), "base_test": (x[3000:], y[3000:])}
    buckets = {"cases": [], "anchors": [], "val": [], "test": [], "shift": []}
    plan = [("cases", "lesion", "h", 1.0, 24, 1.0), ("anchors", "adjacent", 1 + BAND, 1 + SHELL, 30, 1.0),
            ("val", "lesion", "h", 1.0, 10, 1.0), ("val", "adjacent", 1 + BAND, 1 + SHELL, 15, 1.0),
            ("test", "lesion", "h", 1.0, 60, 1.0), ("test", "adjacent", 1 + BAND, 1 + SHELL, 60, 1.0),
            ("shift", "lesion", "h", 1.0, 60, SHIFT_SCALE), ("shift", "band", 1.0, 1 + BAND, 60, SHIFT_SCALE)]
    for li, lesion in enumerate(world["lesions"]):
        extra = [("anchors", "core", 0.0, "hb", 12, 1.0), ("test", "core", 0.0, "hb", 30, 1.0),
                 ("shift", "band", "hb", "h", 30, SHIFT_SCALE)] if lesion["h"] > 0 else []
        for bucket, role, qmin, qmax, n, scale in plan + extra:
            lo = {"h": lesion["h"], "hb": lesion["h"] - BAND / 2}.get(qmin, qmin)
            hi = {"h": lesion["h"], "hb": lesion["h"] - BAND / 2}.get(qmax, qmax)
            us = _sample(rng, world, li, lo, hi, n)
            buckets[bucket].append((_embed(rng, us, scale), labels(world, us), li, role))
    for bucket, n in (("anchors", 60), ("test", 300)):
        us = _sample_far(rng, world, n)
        buckets[bucket].append((_embed(rng, us), labels(world, us), -1, "far"))
    cases, anchors = _stack(buckets["cases"]), _stack(buckets["anchors"])
    ep["correct"] = {"X": np.vstack([cases["X"], anchors["X"]]), "y": np.concatenate([cases["y"], anchors["y"]]),
                     "X_case": cases["X"], "y_case": cases["y"], "anchors": anchors["X"]}
    ep.update({k: _stack(buckets[k]) for k in ("val", "test", "shift")})
    ep["lesions"] = world["lesions"]
    return ep


def make_systemic(world, rng):
    """Blind episode: one base class remapped everywhere; the cases come from one ball only."""
    sysw = dict(world, lesions=[])
    probe = rng.uniform(-0.7, 0.7, (4000, D_BODY))
    c_star = int(np.bincount(_teacher_labels(world["teacher"], probe), minlength=N_CLASS).argmax())
    site = probe[_teacher_labels(world["teacher"], probe) == c_star][0]
    sysw["systemic"] = (c_star, int(rng.integers(1, N_CLASS)))
    old = lambda u: _teacher_labels(world["teacher"], u)
    near = lambda u: np.linalg.norm(u - site, axis=1) < 0.35
    parts = {"cases": (lambda u: near(u) & (old(u) == c_star), 30), "anchors": (lambda u: (old(u) != c_star) & ~near(u), 150),
             "val_case": (lambda u: near(u) & (old(u) == c_star), 10), "val_keep": (lambda u: old(u) != c_star, 30),
             "remote": (lambda u: (old(u) == c_star) & (np.linalg.norm(u - site, axis=1) > 0.7), 300),
             "local": (lambda u: near(u) & (old(u) == c_star), 80), "keep": (lambda u: old(u) != c_star, 400)}
    out = {}
    for name, (accept, n) in parts.items():
        us = _sample_far(rng, sysw, n, accept)
        out[name] = (_embed(rng, us), labels(sysw, us))
    return {"correct": {"X": np.vstack([out["cases"][0], out["anchors"][0]]), "y": np.concatenate([out["cases"][1], out["anchors"][1]]),
                        "X_case": out["cases"][0], "y_case": out["cases"][1], "anchors": out["anchors"][0]},
            "val": _stack([(out["val_case"][0], out["val_case"][1], 0, "lesion"), (out["val_keep"][0], out["val_keep"][1], -1, "adjacent")]),
            "remote": out["remote"], "local": out["local"], "keep": out["keep"]}


# ------------------------------------------------------------------ model
def build_model(in_dim, out_dim, task_type="vector_classification", rng=None, **cfg):
    if task_type not in TASK_TYPES:
        raise ValueError(f"unsupported task type {task_type}")
    rng = rng if rng is not None else np.random.default_rng(0)
    c = {"hidden": HIDDEN, "units": UNITS, "profile": "compact", "padding": True, "shapes": True, "dressing": True}
    c.update(cfg)
    h, k, lim = c["hidden"], c["units"], np.sqrt(6.0 / (in_dim + c["hidden"]))
    p = {"W1": rng.uniform(-lim, lim, (h, in_dim)), "b1": np.zeros(h), "W2": rng.normal(0, 0.3, (out_dim, h)),
         "b2": np.zeros(out_dim), "C": rng.normal(0, 0.5, (k, in_dim)), "R": np.log(0.3) + rng.normal(0, 0.1, (k, in_dim)),
         "OM": rng.normal(0, 0.1, k), "BE": rng.normal(-1.0, 0.5, k), "U": rng.normal(0, 0.1, (k, out_dim)),
         "WD": rng.normal(0, 0.05, out_dim)}
    return {"params": p, "cfg": c, "anchors": np.zeros((0, in_dim)), "marks": np.zeros((0, in_dim)), "stage": "base",
            "h_mean": np.zeros(h), "knock": set()}


def with_iron(model, **cfg):
    m = copy_model(model)
    m["cfg"].update({"profile": "compact", "padding": True, "shapes": True, "dressing": True}, **cfg)
    return m


def copy_model(model):
    return {"params": {k: v.copy() for k, v in model["params"].items()}, "cfg": dict(model["cfg"]),
            "anchors": model["anchors"].copy(), "marks": model["marks"].copy(), "stage": model["stage"], "h_mean": model["h_mean"].copy(),
            "knock": set(model["knock"])}


def _body(model, x):
    p = model["params"]
    h = np.tanh(x @ p["W1"].T + p["b1"])
    if "body" in model["knock"]:
        h = np.broadcast_to(model["h_mean"], h.shape).copy()
    return h, h @ p["W2"].T + p["b2"]


def _geometry(p, cfg, z):
    r = np.exp(p["R"])
    e = (z[:, None, :] - p["C"][None]) / r[None]
    q = np.sqrt(np.einsum("nkd,nkd->nk", e, e) + EPS_Q)
    beta = BETA_MAX * _sigmoid(p["BE"]) if cfg["shapes"] else np.zeros(len(r))
    return {"r": r, "E": e, "q": q, "D": q - beta[None]}


def _padding_cap(p, cfg, anchors):
    """Anatomy before the hand: no unit may reach a verified anchor, whatever its parameters."""
    w_raw = np.exp(p["OM"])
    if not cfg["padding"] or len(anchors) == 0:
        return w_raw, None
    ga = _geometry(p, cfg, anchors)
    da2, m = ga["D"] ** 2, (ga["D"] ** 2).min(axis=0)
    r = np.where(da2 <= m[None], 1.0, m[None] / np.maximum(da2, 1e-300))  # soft-min taken relative to the exact min: no epsilon, never above it
    ssum = (r ** P_GUARD).sum(axis=0)
    g = da2.mean(axis=0) if MUTANT == "guard_mean" else m * ssum ** (-1.0 / P_GUARD)
    cap = GAMMA * np.sqrt(g)
    s = w_raw ** P_CLAMP + cap ** P_CLAMP
    gc = {"ga": ga, "R": r, "SS": ssum, "G": g, "cap": cap, "S": s, "w_raw": w_raw}
    return np.maximum(w_raw * cap * s ** (-1.0 / P_CLAMP), W_MIN), gc


def iron_profile(s):
    """Fire: flat-topped and exactly zero beyond the mark, phi = (1 - s^4)^3 for s < 1 (C2 at the edge)."""
    t = np.maximum(1.0 - s ** 4, 0.0)
    return t ** 3, -12.0 * s ** 3 * t ** 2


def mikwa_field(model, x):
    p, cfg = model["params"], model["cfg"]
    if cfg["profile"] == "none":
        return np.zeros((len(x), p["U"].shape[1])), None
    g = _geometry(p, cfg, x)
    w, gc = _padding_cap(p, cfg, model["anchors"])
    s = (g["D"] ** 2 + W_MIN ** 2) / w[None] ** 2
    if cfg["profile"] == "compact":
        phi, dphi_ds = iron_profile(s)
    else:
        phi = np.exp(-s)
        dphi_ds = -phi
    v = p["U"] + (p["WD"][None] if cfg["dressing"] else 0.0)
    return phi @ v, {"g": g, "gc": gc, "w": w, "phi": phi, "dphi_ds": dphi_ds, "V": v}


def mikwa_activation(model, x):
    _, cache = mikwa_field(model, x)
    return np.zeros((len(x), model["cfg"]["units"])) if cache is None else cache["phi"]


def logits(model, x):
    return _body(model, x)[1] + mikwa_field(model, x)[0]


def predict(model, x):
    return np.argmax(logits(model, x), axis=1)


def _geometry_backward(p, cfg, g, d_d, grads):
    if cfg["shapes"]:
        sg = _sigmoid(p["BE"])
        grads["BE"] -= d_d.sum(axis=0) * BETA_MAX * sg * (1 - sg)
    coef = (d_d / (2.0 * g["q"]))[:, :, None] * g["E"]
    grads["C"] -= 2.0 * (coef / g["r"][None]).sum(axis=0)
    grads["R"] -= 2.0 * (coef * g["E"]).sum(axis=0)


def _support_backward(model, g, gc, w, ds, grads):
    """Chain dL/ds through the unit geometry and, when padding is on, through the anchor cap."""
    p, cfg = model["params"], model["cfg"]
    _geometry_backward(p, cfg, g, ds * 2.0 * g["D"] / w[None] ** 2, grads)
    dw = (ds * -2.0 * (g["D"] ** 2 + W_MIN ** 2) / w[None] ** 3).sum(axis=0)
    if gc is None:
        grads["OM"] += dw * w
        return
    a, b, s = gc["w_raw"], gc["cap"], gc["S"]
    root = s ** (-1.0 / P_CLAMP)
    dw = dw * (a * b * root > W_MIN)
    grads["OM"] += dw * b * root * (b ** P_CLAMP / s) * a
    d_g = dw * a * root * (a ** P_CLAMP / s) * GAMMA / (2.0 * np.sqrt(np.maximum(gc["G"], 1e-300)))
    d_da2 = d_g[None] * gc["R"] ** (P_GUARD + 1) / gc["SS"][None] ** (1.0 + 1.0 / P_GUARD)
    _geometry_backward(p, cfg, gc["ga"], d_da2 * 2.0 * gc["ga"]["D"], grads)


def _field_backward(model, cache, g_out, grads):
    grads["U"] += cache["phi"].T @ g_out
    if model["cfg"]["dressing"]:
        grads["WD"] += (cache["phi"].T @ g_out).sum(axis=0)
    ds = (g_out @ cache["V"].T) * cache["dphi_ds"]
    _support_backward(model, cache["g"], cache["gc"], cache["w"], ds, grads)


def _mark_penalty(model, grads):
    """Ink before fire: every marked failing case should sit well inside some unit's support."""
    marks = model["marks"]
    if len(marks) == 0:
        return 0.0
    p, cfg = model["params"], model["cfg"]
    g = _geometry(p, cfg, marks)
    w, gc = _padding_cap(p, cfg, model["anchors"])
    s = g["D"] ** 2 / w[None] ** 2
    gap = (-MARK_SOFT * logsumexp(-s / MARK_SOFT, axis=1) - MARK_LEVEL) / MARK_TEMP
    ds = (MARK_WEIGHT * _sigmoid(gap) / len(marks))[:, None] * softmax(-s / MARK_SOFT, axis=1)
    _support_backward(model, g, gc, w, ds, grads)
    return float(MARK_WEIGHT * MARK_TEMP * softplus(gap).mean())


def loss_and_grads(model, batch):
    x, y = batch
    p = model["params"]
    h, base = _body(model, x)
    field, cache = mikwa_field(model, x)
    z = base + field
    lse, n = logsumexp(z, axis=1), len(y)
    loss = float((lse - z[np.arange(n), y]).mean())
    g_out = np.exp(z - lse[:, None])
    g_out[np.arange(n), y] -= 1.0
    g_out /= n
    grads = {k: np.zeros_like(v) for k, v in p.items()}
    grads["W2"], grads["b2"] = g_out.T @ h, g_out.sum(axis=0)
    if "body" not in model["knock"]:
        dpre = (g_out @ p["W2"]) * (1.0 - h ** 2)
        grads["W1"], grads["b1"] = dpre.T @ x, dpre.sum(axis=0)
    if cache is not None:
        _field_backward(model, cache, g_out, grads)
        if model["stage"] == "correct":
            loss += _mark_penalty(model, grads)
            loss += LAMBDA_DOSE * (x.shape[1] * p["OM"].sum() + p["R"].sum()) + LAMBDA_HEAT * ((p["U"] ** 2).sum() + (p["WD"] ** 2).sum())
            grads["OM"] += LAMBDA_DOSE * x.shape[1]
            grads["R"] += LAMBDA_DOSE
            grads["U"] += 2 * LAMBDA_HEAT * p["U"]
            grads["WD"] += 2 * LAMBDA_HEAT * p["WD"]
    if MUTANT == "zero_grad_U":
        grads["U"][:] = 0.0
    if MUTANT == "zero_grad_W1":
        grads["W1"][:] = 0.0
    return loss, grads


def _dose(y, z):
    """Heat set from the marked cases: cancel the body's preference, then favour the cure by 4 nats."""
    return -(z - z.mean(axis=1, keepdims=True)).mean(axis=0) + 4.0 * (np.eye(z.shape[1])[y].mean(axis=0) - 1.0 / z.shape[1])


def _components(pts, tau):
    parent = np.arange(len(pts))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    close = np.linalg.norm(pts[:, None] - pts[None], axis=-1) < tau
    for i, j in zip(*np.nonzero(np.triu(close, 1))):
        parent[root(i)] = root(j)
    roots = np.array([root(i) for i in range(len(pts))], dtype=int)
    return sorted((np.nonzero(roots == r)[0] for r in np.unique(roots)), key=len, reverse=True)


def mark_before_fire(model, x_case, y_case, rng, hollow_test=True):
    """Ink before fire: place, shape and dose each iron from the cases the body still fails."""
    p, cfg, anchors = model["params"], model["cfg"], model["anchors"]
    z = _body(model, x_case)[1]
    fail = np.argmax(z, axis=1) != y_case
    f, yf, pf = x_case[fail], y_case[fail], z[fail]
    k_units = cfg["units"]
    p["C"][:] = f[rng.integers(0, len(f), k_units)] if len(f) else rng.normal(0, 0.5, p["C"].shape)
    p["R"][:], p["OM"][:], p["BE"][:], p["WD"][:] = np.log(0.12), 0.0, -6.0, 0.0
    p["U"][:] = rng.normal(0.0, 0.01, p["U"].shape)
    model["marks"], (comps, rings) = f.copy(), (_components(f, LINK)[:k_units], 0)
    for k, idx in enumerate(comps):
        pts = f[idx]
        m = pts.mean(axis=0)
        dist = np.linalg.norm(pts - m, axis=1)
        med = float(np.median(dist))
        p["C"][k], p["U"][k] = m, _dose(yf[idx], pf[idx])
        core = int((np.linalg.norm(anchors - m, axis=1) < 0.75 * med).sum()) if len(anchors) else 0
        if hollow_test and cfg["shapes"] and len(idx) >= 6 and core >= 2:
            rings += 1
            p["R"][k], p["BE"][k] = np.log(med / 0.7), np.log(0.7 / (BETA_MAX - 0.7))
            p["OM"][k] = np.log(np.clip(1.25 * np.abs(dist * 0.7 / med - 0.7).max(), 0.15, 0.6))
        else:
            radii = np.maximum(2.5 * pts.std(axis=0), 0.10)
            reach = np.linalg.norm((pts - m) / radii, axis=1).max()
            p["R"][k] = np.log(radii * max(reach / 0.8, 1.0))
    _place_spares(model, f, yf, pf, len(comps))
    return {"components": len(comps), "rings": rings, "failing": int(fail.sum())}


def _place_spares(model, f, yf, pf, used):
    p, k_units = model["params"], model["cfg"]["units"]
    if len(f) == 0 or used >= k_units:
        return
    covered = mikwa_activation(model, f)[:, :used].max(axis=1) > 0.05 if used else np.zeros(len(f), bool)
    for k in range(used, k_units):
        free = np.nonzero(~covered)[0]
        if len(free) == 0:
            break
        crowd = (np.linalg.norm(f[free][:, None] - f[free][None], axis=-1) < 0.25).sum(axis=1)
        j = free[int(np.argmax(crowd))]
        near = np.linalg.norm(f - f[j], axis=1) < 0.25
        p["C"][k], p["R"][k], p["BE"][k], p["U"][k] = f[j], np.log(0.15), -6.0, _dose(yf[near], pf[near])
        covered |= near


def hidden_states(model, x):
    out = {"body_hidden": _body(model, x)[0]}
    if model["cfg"]["profile"] != "none":
        out["unit_activation"] = mikwa_activation(model, x)
    return out


# ------------------------------------------------------------------ baselines and rival mechanisms
METHODS = {
    "mikwa": {"profile": "compact"},
    "caustic_rbf": {"profile": "gauss", "padding": False, "shapes": False},
    "drug_finetune": None,
    "no_padding": {"padding": False},
    "no_dressing": {"dressing": False},
    "olive_only": {"shapes": False},
}


def _trainable(model, key):
    c = model["cfg"]
    return not ((key == "BE" and not c["shapes"]) or (key == "WD" and not c["dressing"]))


def correction_params(model):
    keys = ("W2", "b2") if model["cfg"]["profile"] == "none" else [k for k in BURN_KEYS if _trainable(model, k)]
    return int(sum(model["params"][k].size for k in keys))


def n_params(model):
    return int(sum(model["params"][k].size for k in BODY_KEYS)) + (0 if model["cfg"]["profile"] == "none" else correction_params(model))


# ------------------------------------------------------------------ registries
MODULES = {
    "body": (BODY_KEYS, "two-layer tanh classifier, frozen while corrections train", False, "mean"),
    "iron": (("C", "R", "OM", "U"), "compactly supported corrective basis units, phi=(1-s^4)^3 for s<1", True, "zero"),
    "padding": ((), "conservative power soft-min cap on unit width over verified anchors", True, "identity"),
    "irons": (("BE",), "learnable hole parameter deforming each support from ball to shell", True, "identity"),
    "dressing": (("WD",), "shared output offset added inside every unit's support", False, "zero"),
}

MUTANTS = {
    "sign_flip_update": {"kind": "learning", "detected_by": ["C1", "C3"]},
    "zero_learning_rate": {"kind": "learning", "detected_by": ["C1", "C3"]},
    "zero_grad_U": {"kind": "learning", "detected_by": ["C1", "C3"]},
    "zero_grad_W1": {"kind": "learning", "detected_by": ["C1", "C3"]},
    "guard_mean": {"kind": "property", "detected_by": ["C1", "C6"]},
}


def modules(model):
    return {name: {"params": list(keys), "role": role, "signature": sig, "knockout_mode": mode}
            for name, (keys, role, sig, mode) in MODULES.items()}


def knockout(model, name, mode=None):
    if name not in MODULES or (mode is not None and mode != MODULES[name][3]):
        raise ValueError(f"module {name} supports knockout mode {MODULES.get(name, (0, 0, 0, 'none'))[3]}")
    m = copy_model(model)
    if name == "body":
        m["knock"].add("body")
    elif name == "iron":
        m["cfg"]["profile"] = "none"
    elif name == "padding":
        m["cfg"]["padding"] = False
    elif name == "irons":
        m["cfg"]["shapes"] = False
    else:
        m["params"]["WD"][:] = 0.0
    return m


# ------------------------------------------------------------------ training
def _train(model, x, y, keys, steps, lr, rng, batch=None):
    keys = [k for k in keys if _trainable(model, k)]
    opt, losses = Adam(model["params"], keys), []
    for _ in range(steps):
        idx = rng.integers(0, len(x), batch) if batch else slice(None)
        loss, grads = loss_and_grads(model, (x[idx], y[idx]))
        sub = {k: grads[k] for k in keys}
        clip_global_norm(sub, CLIP)
        rate = 0.0 if MUTANT == "zero_learning_rate" else (-lr if MUTANT == "sign_flip_update" else lr)
        opt.step(model["params"], sub, rate)
        if not np.isfinite(loss) or not all(np.all(np.isfinite(model["params"][k])) for k in keys):
            raise FloatingPointError("non-finite loss or parameters during training")
        losses.append(loss)
    return losses


def fit(model, data, budget, rng, lr=None):
    stage = data.get("stage", "auto")
    if stage == "base":
        model["stage"], saved, model["cfg"]["profile"] = "base", model["cfg"]["profile"], "none"
        losses = _train(model, data["X"], data["y"], BODY_KEYS, budget, lr or LR_BODY, rng, BATCH)
        model["h_mean"], model["cfg"]["profile"] = _body(model, data["X"])[0].mean(axis=0), saved
        return losses
    if stage == "finetune":
        model["stage"], model["cfg"]["profile"] = "finetune", "none"
        return _train(model, data["X"], data["y"], ("W2", "b2"), budget, lr or LR_GRID[0], rng)
    if stage == "correct":
        model["stage"], model["anchors"] = "correct", data["anchors"].copy()
        mark_before_fire(model, data["X_case"], data["y_case"], rng)
        return _train(model, data["X"], data["y"], BURN_KEYS, budget, lr or LR_GRID[0], rng)
    cut = int(0.75 * len(data["X"]))
    fit(model, {"stage": "base", "X": data["X"][:cut], "y": data["y"][:cut]}, budget, rng)
    xb, yb = data["X"][cut:], data["y"][cut:]
    ok = predict(model, xb) == yb
    return fit(model, {"stage": "correct", "X": xb, "y": yb, "X_case": xb[~ok], "y_case": yb[~ok], "anchors": xb[ok]},
               max(budget // 3, 1), rng, lr)


def surgical(model, split, site=None):
    ok = predict(model, split["X"]) == split["y"]
    sel = np.ones(len(ok), bool) if site is None else split["site"] == site
    lesion, keep = split["role"] == "lesion", np.isin(split["role"], ["adjacent", "core", "band"])
    return float(0.5 * (ok[sel & lesion].mean() + ok[sel & keep].mean()))


def accuracy(model, x, y):
    return float((predict(model, x) == y).mean())


def correct_with_grid(body, data, val, method, budget, key):
    best = None
    for j, lr in enumerate(LR_GRID):
        rng = np.random.default_rng(np.random.SeedSequence([*key, j]))
        if METHODS[method] is None:
            m = copy_model(body)
            fit(m, dict(data, stage="finetune"), budget, rng, lr)
        else:
            m = with_iron(body, **METHODS[method])
            fit(m, dict(data, stage="correct"), budget, rng, lr)
        score = surgical(m, val)
        if best is None or score > best[0]:
            best = (score, lr, m)
    return best[2], best[1]


def evaluate(name, m, ep):
    te, sh = ep["test"], ep["shift"]
    ring = [i for i, les in enumerate(ep["lesions"]) if les["kind"] == "ring"][0]
    ok_te = predict(m, te["X"]) == te["y"]
    return {f"{name}/surgical_iid": surgical(m, te), f"{name}/surgical_shift": surgical(m, sh),
            f"{name}/cure_iid": float(ok_te[te["role"] == "lesion"].mean()),
            f"{name}/adjacent_iid": float(ok_te[np.isin(te["role"], ["adjacent", "core"])].mean()),
            f"{name}/far_iid": float(ok_te[te["role"] == "far"].mean()),
            f"{name}/band_shift": float((predict(m, sh["X"]) == sh["y"])[sh["role"] == "band"].mean()),
            f"{name}/ring_surgical_shift": surgical(m, sh, site=ring)}


def run_seed(seed, budget):
    r_world, r_ep, r_body, r_sys = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
    world = make_world(r_world)
    ep = make_episode(world, r_ep)
    body = build_model(DIM, N_CLASS, rng=r_body, profile="none")
    fit(body, {"stage": "base", "X": ep["base_train"][0], "y": ep["base_train"][1]}, budget["body"], r_body)
    out = {"trivial/surgical_iid": surgical(body, ep["test"]), "trivial/surgical_shift": surgical(body, ep["shift"]),
           "body/base_test_accuracy": accuracy(body, *ep["base_test"])}
    best = {}
    for i, method in enumerate(METHODS):
        best[method], out[f"{method}/lr"] = correct_with_grid(body, ep["correct"], ep["val"], method, budget["corr"], (seed, i))
        out.update(evaluate(method, best[method], ep))
    for mod in MODULES:
        out[f"knockout/{mod}"] = surgical(knockout(best["mikwa"], mod), ep["shift"]) - out["mikwa/surgical_shift"]
    se = make_systemic(world, r_sys)
    out["sys/trivial/remote_cure"] = accuracy(body, *se["remote"])
    for i, method in enumerate(("mikwa", "caustic_rbf", "drug_finetune")):
        m, _ = correct_with_grid(body, se["correct"], se["val"], method, budget["corr"], (seed, 50 + i))
        out[f"sys/{method}/remote_cure"], out[f"sys/{method}/local_cure"] = accuracy(m, *se["remote"]), accuracy(m, *se["local"])
        out[f"sys/{method}/keep"] = accuracy(m, *se["keep"])
    out["params/mikwa_correction"], out["params/caustic_rbf_correction"] = correction_params(best["mikwa"]), correction_params(best["caustic_rbf"])
    out["params/drug_finetune_correction"], out["params/mikwa_total"] = correction_params(best["drug_finetune"]), n_params(best["mikwa"])
    return out


# ------------------------------------------------------------------ tests: correctness
def _context(seed):
    rng = np.random.default_rng(seed)
    world = make_world(rng)
    ep = make_episode(world, rng)
    return rng, world, ep


def c1_gradcheck(seed, steps=50):
    rng, _, ep = _context(seed)
    body = build_model(DIM, N_CLASS, rng=rng, profile="none")
    fit(body, {"stage": "base", "X": ep["base_train"][0][:600], "y": ep["base_train"][1][:600]}, 120, rng)
    summary, ok = {"tensors_checked": 0, "tensors_total": 0, "max_rel_error": 0.0, "checked_at": ["init", "after_training_steps"]}, True
    for profile in ("compact", "gauss"):
        m = with_iron(body, **METHODS["mikwa" if profile == "compact" else "caustic_rbf"])
        m["stage"], m["anchors"] = "correct", ep["correct"]["anchors"][:40]
        mark_before_fire(m, ep["correct"]["X_case"], ep["correct"]["y_case"], rng)
        sel = rng.choice(len(ep["correct"]["X"]), 48, replace=False)
        batch = (ep["correct"]["X"][sel], ep["correct"]["y"][sel])
        keys = [k for k in m["params"] if _trainable(m, k)]
        for phase in ("init", "after"):
            if phase == "after":
                _train(m, ep["correct"]["X"], ep["correct"]["y"], BURN_KEYS, steps, 0.02, rng)
            _, grads = loss_and_grads(m, batch)
            rep = finite_difference_check(lambda: loss_and_grads(m, batch)[0], m["params"], grads, keys, rng)
            ok &= all(r["passed"] for r in rep.values())
            summary["tensors_checked"] += sum(r["passed"] for r in rep.values())
            summary["tensors_total"] += len(rep)
            summary["max_rel_error"] = max(summary["max_rel_error"], max(r["max_rel_error"] for r in rep.values()))
    summary["passed"] = bool(ok)
    return bool(ok), summary


def c2_determinism(seed):
    outs = []
    for _ in range(2):
        rng, _, ep = _context(seed)
        m = build_model(DIM, N_CLASS, rng=rng)
        l1 = fit(m, {"stage": "base", "X": ep["base_train"][0][:600], "y": ep["base_train"][1][:600]}, 60, rng)
        l2 = fit(m, dict(ep["correct"], stage="correct"), 40, rng, 0.02)
        out = logits(m, ep["test"]["X"])
        finite = np.all(np.isfinite(out)) and all(np.all(np.isfinite(v)) for v in m["params"].values())
        outs.append((np.array(l1 + l2), out, finite))
    same = np.array_equal(outs[0][0], outs[1][0]) and np.array_equal(outs[0][1], outs[1][1])
    return bool(same and outs[0][2] and outs[1][2]), f"identical={same}, finite={outs[0][2] and outs[1][2]}"


def c3_learning(seed, budget):
    rng, world, ep = _context(seed)
    body = build_model(DIM, N_CLASS, rng=rng, profile="none")
    lb = fit(body, {"stage": "base", "X": ep["base_train"][0], "y": ep["base_train"][1]}, budget["body"], rng)
    drop_b = 1.0 - np.mean(lb[-20:]) / np.mean(lb[:20])
    majority = np.bincount(ep["base_test"][1], minlength=N_CLASS).max() / len(ep["base_test"][1])
    acc_b = accuracy(body, *ep["base_test"])
    m = with_iron(body)
    lc = fit(m, dict(ep["correct"], stage="correct"), budget["corr"], rng, LR_GRID[1])
    drop_c = 1.0 - lc[-1] / lc[0]
    surg, pers = surgical(m, ep["test"]), surgical(body, ep["test"])
    th = MIND_CARD["thresholds"]
    ok = (drop_b >= th["loss_drop_fraction"]["body"] and acc_b >= majority + th["margin_over_trivial"]["body_accuracy_over_majority"]
          and drop_c >= th["loss_drop_fraction"]["correction"] and surg >= pers + th["margin_over_trivial"]["surgical_over_persistence"])
    detail = (f"body loss drop {drop_b:.2f}, acc {acc_b:.3f} vs majority {majority:.3f}; "
              f"correction loss drop {drop_c:.2f}, surgical {surg:.3f} vs persistence {pers:.3f}")
    return bool(ok), detail, {"body": body, "model": m, "ep": ep, "majority": majority}


def c4_shuffled(seed, budget, ctx):
    rng, ep, th = np.random.default_rng(seed + 7919), ctx["ep"], MIND_CARD["thresholds"]["shuffled_band"]
    shuffled = build_model(DIM, N_CLASS, rng=rng, profile="none")
    fit(shuffled, {"stage": "base", "X": ep["base_train"][0], "y": rng.permutation(ep["base_train"][1])}, budget["body"], rng)
    acc_s, band_b = accuracy(shuffled, *ep["base_test"]), ctx["majority"] + th["body_over_majority"]
    corr, n_case = dict(ep["correct"]), len(ep["correct"]["y_case"])
    corr["y_case"] = rng.permutation(corr["y_case"])
    corr["y"] = np.concatenate([corr["y_case"], corr["y"][n_case:]])
    m = with_iron(ctx["body"])
    fit(m, dict(corr, stage="correct"), budget["corr"], rng, LR_GRID[1])
    te = ep["test"]
    lesion = te["role"] == "lesion"
    cure = float((predict(m, te["X"]) == te["y"])[lesion].mean())
    pooled = int(np.bincount(ep["correct"]["y_case"], minlength=N_CLASS).argmax())
    trivial = max(float((te["y"][lesion] == pooled).mean()), float((predict(ctx["body"], te["X"]) == te["y"])[lesion].mean()))
    band_c = trivial + th["cure_over_trivial"]
    ok = acc_s <= band_b and cure <= band_c
    return bool(ok), f"body acc {acc_s:.3f} <= {band_b:.3f}; shuffled cure {cure:.3f} <= {band_c:.3f}"


def c6_padding(seed, trained=None, trials=120):
    rng = np.random.default_rng(seed + 31)
    worst, worst_ctrl = 0.0, 0.0
    for _ in range(trials):
        m = build_model(DIM, N_CLASS, rng=rng)
        p, anchors = m["params"], rng.normal(0, 0.6, (40, DIM))
        m["anchors"] = anchors
        p["OM"][:], p["BE"][:], p["R"][:] = rng.uniform(-1, 6, UNITS), rng.normal(0, 2, UNITS), rng.normal(-0.5, 0.7, p["R"].shape)
        k = int(rng.integers(UNITS))
        p["C"][k] = anchors[rng.integers(40)] + rng.normal(0, 1e-3, DIM)
        k2, u = int(rng.integers(UNITS)), rng.normal(size=DIM)  # adversarial: one anchor exactly on a shell's ring (q = beta)
        anchors[1] = p["C"][k2] + np.exp(p["R"][k2]) * u / np.linalg.norm(u) * BETA_MAX * _sigmoid(p["BE"][k2])
        for _ in range(15):
            violation = lambda: mikwa_activation(m, anchors)[:, k].sum()
            base, p["OM"][k] = violation(), p["OM"][k] + 1e-4
            slope = (violation() - base) / 1e-4
            p["OM"][k] += 0.5 * np.sign(slope) if slope != 0 else 0.5
        worst = max(worst, float(mikwa_activation(m, anchors).max()))
        worst_ctrl = max(worst_ctrl, float(mikwa_activation(knockout(m, "padding"), anchors).max()))
    trained_ok = True
    if trained is not None:
        trained_ok = float(np.abs(mikwa_field(trained, trained["anchors"])[0]).max()) == 0.0
    ok = worst == 0.0 and worst_ctrl > 1e-3 and trained_ok
    return bool(ok), f"max phi at anchors {worst:.1e} (control {worst_ctrl:.2f}); trained field at anchors exactly zero: {trained_ok}"


def c6_confinement(seed):
    rng = np.random.default_rng(seed + 37)
    m = build_model(DIM, N_CLASS, rng=rng, padding=False)
    x = rng.normal(0, 1.0, (4000, DIM))
    outside = (_geometry(m["params"], m["cfg"], x)["D"] ** 2 / np.exp(m["params"]["OM"])[None] ** 2).min(axis=1) >= 1.0
    inside_val = float(np.abs(mikwa_field(m, x[outside])[0]).max())
    gauss = copy_model(m)
    gauss["cfg"]["profile"] = "gauss"
    ctrl = float(np.abs(mikwa_field(gauss, x[outside])[0]).max())
    return bool(outside.sum() > 100 and inside_val == 0.0 and ctrl > 1e-6), f"definition check: |field| outside supports {inside_val:.1e}, caustic control {ctrl:.1e}"


def c6_graded(seed):
    rng, bad, bad_ctrl = np.random.default_rng(seed + 41), 0, 0
    for _ in range(20):
        m = build_model(DIM, N_CLASS, rng=rng, padding=False)
        p = m["params"]
        x = p["C"][rng.integers(0, UNITS, 400)] + rng.normal(0, 1, (400, DIM)) * np.exp(p["R"]).mean() * 0.8
        g = _geometry(p, m["cfg"], x)
        s = g["D"] ** 2 / np.exp(p["OM"])[None] ** 2
        for k in range(UNITS):
            order = np.argsort(np.abs(g["D"][:, k]))
            phi = iron_profile(s[order, k])[0]
            rippled = phi * (1 + 0.9 * np.sin(12 * s[order, k]))
            bad += int((np.diff(phi) > 1e-12).sum())
            bad_ctrl += int((np.diff(rippled) > 1e-12).sum())
    return bool(bad == 0 and bad_ctrl > 0), f"definition check: increases with |D| {bad} (rippled control {bad_ctrl})"


def c6_marking(seed):
    rng, results = np.random.default_rng(seed + 43), []
    for shell, hollow in ((False, True), (True, True), (True, False)):
        m = build_model(DIM, N_CLASS, rng=rng)
        m["params"]["W1"][:], m["params"]["W2"][:], m["params"]["b2"][:] = 0.0, 0.0, [5.0, 0, 0, 0]
        v = rng.normal(size=(30, D_BODY))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        rad = rng.uniform(0.22, 0.38, 30) if shell else rng.uniform(0.0, 1.0, 30) ** (1 / 3) * 0.3
        pts = np.hstack([v * rad[:, None], rng.normal(0, 0.05, (30, D_NUIS))])
        w = rng.normal(size=(40, D_BODY))
        w /= np.linalg.norm(w, axis=1, keepdims=True)
        outer = np.hstack([w * rng.uniform(0.5, 0.75, 40)[:, None], rng.normal(0, 0.05, (40, D_NUIS))])
        core = np.hstack([rng.normal(0, 0.05, (8, D_BODY)), rng.normal(0, 0.05, (8, D_NUIS))])
        m["anchors"] = np.vstack([outer, core]) if shell else outer
        info = mark_before_fire(m, pts, np.ones(30, int), rng, hollow_test=hollow)
        beta = BETA_MAX * _sigmoid(m["params"]["BE"][0])
        results.append((info["rings"], float(beta), float(np.linalg.norm(m["params"]["C"][0]))))
    ball_ok = results[0][0] == 0 and results[0][1] < 0.05 and results[0][2] < 0.08
    shell_ok = results[1][0] == 1 and results[1][1] > 0.5 and results[1][2] < 0.08
    control_detected = results[2][0] == 0
    return bool(ball_ok and shell_ok and control_detected), f"ball->olive {ball_ok}, shell->ring {shell_ok}, disabled hollow test misses ring {control_detected}"


def c7_splits(ep):
    def keys(x):
        return {hashlib.sha1(np.round(r, 12).tobytes()).hexdigest() for r in x}
    train = keys(ep["base_train"][0]) | keys(ep["correct"]["X"])
    evals = keys(ep["base_val"][0]) | keys(ep["base_test"][0]) | keys(ep["val"]["X"]) | keys(ep["test"]["X"]) | keys(ep["shift"]["X"])
    return len(train & evals) == 0, f"overlapping rows: {len(train & evals)}"


def c5_mutants(seed):
    global MUTANT
    found = {}
    for name, spec in MUTANTS.items():
        MUTANT, failed = name, []
        try:
            if not c1_gradcheck(seed, steps=50)[0]:
                failed.append("C1")
            if spec["kind"] == "learning" and not c3_learning(seed, BUDGET["quick"])[0]:
                failed.append("C3")
            if spec["kind"] == "property" and not c6_padding(seed, trials=30)[0]:
                failed.append("C6")
        except FloatingPointError:
            failed.append("C2")
        finally:
            MUTANT = None
        found[name] = failed
    detected = sum(bool(v) for v in found.values())
    return detected == len(MUTANTS), {"detected": detected, "total": len(MUTANTS), "score": detected / len(MUTANTS), "by_mutant": found}


# ------------------------------------------------------------------ tests: hypotheses
def evaluate_hypotheses(per_seed, seed, quick):
    rng, rows = np.random.default_rng(seed + 104729), []
    for h in MIND_CARD["hypotheses"]:
        diffs = [s[h["keys"][0]] - s[h["keys"][1]] for s in per_seed]
        mean, ci = paired_bootstrap(diffs, rng)
        rows.append({"id": h["id"], "metric": h["metric"], "mean_diff": round(mean, 4), "ci95": [round(c, 4) for c in ci],
                     "mesi": h["mesi"], "n_seeds": len(diffs),
                     "verdict": "not evaluated" if quick or len(diffs) < 5 else verdict(mean, ci, h["mesi"], h["direction"])})
    knocks = []
    for mod, (_, _, sig, _) in MODULES.items():
        mean, ci = paired_bootstrap([s[f"knockout/{mod}"] for s in per_seed], rng)
        knocks.append({"module": mod, "signature": sig, "metric_change": round(mean, 4), "ci95": [round(c, 4) for c in ci]})
    return rows, knocks


# ------------------------------------------------------------------ report
def real_data_bridge(path, seed):
    if not path:
        print("real-data bridge: skipped (no --data PATH given)")
        return None
    rows = [ln.strip().split(",") for ln in open(path, encoding="utf-8") if ln.strip()]
    try:
        float(rows[0][0])
    except ValueError:
        rows = rows[1:]
    x = np.array([[float(v) for v in r[:-1]] for r in rows])
    names = sorted({r[-1] for r in rows})
    y = np.array([names.index(r[-1]) for r in rows])
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-9)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    tr, te = order[: int(0.8 * len(y))], order[int(0.8 * len(y)):]
    body = build_model(x.shape[1], len(names), rng=rng, profile="none")
    fit(body, {"stage": "base", "X": x[tr], "y": y[tr]}, 1500, rng)
    m = build_model(x.shape[1], len(names), rng=np.random.default_rng(seed))
    fit(m, {"X": x[tr], "y": y[tr]}, 1500, np.random.default_rng(seed))
    res = {"file": os.path.basename(path), "classes": len(names), "body_only_accuracy": accuracy(body, x[te], y[te]),
           "regimen_then_iron_accuracy": accuracy(m, x[te], y[te])}
    print("real-data bridge:", json.dumps(res))
    return res


def print_report(rep, corr, hyps, knocks, means):
    print("=== VERIFIED REPORT · chapter 0243 ===")
    print(f"environment: python {rep['environment']['python']}, numpy {rep['environment']['numpy']}")
    print(f"seeds: {rep['seeds']}   runtime: {rep['runtime_s']:.1f} s   parameters: {rep['n_params']} total, "
          f"correction {means.get('params/mikwa_correction', 0):.0f} (caustic_rbf {means.get('params/caustic_rbf_correction', 0):.0f}, "
          f"drug_finetune {means.get('params/drug_finetune_correction', 0):.0f})")
    g = rep["gradcheck"]
    print(f"gradcheck: {g['tensors_checked']}/{g['tensors_total']} tensor checks passed, max rel error {g['max_rel_error']:.1e}, at {g['checked_at']}")
    print("correctness:")
    for row in corr:
        print(f"  {row['id']:<5} {row['name']:<28} {'PASS' if row['passed'] else 'FAIL'}  {row['detail']}")
    print(f"mutation score: {rep['mutants']['detected']}/{rep['mutants']['total']}")
    print("hypotheses:")
    for h in hyps:
        print(f"  {h['id']:<9} diff {h['mean_diff']:+.4f}  ci95 [{h['ci95'][0]:+.4f}, {h['ci95'][1]:+.4f}]  mesi {h['mesi']:.2f}  {h['verdict']}")
    print("knockouts (inference-time, change in mikwa surgical_shift):")
    for k in knocks:
        print(f"  {k['module']:<9} signature={str(k['signature']):<5} {k['metric_change']:+.4f}  ci95 [{k['ci95'][0]:+.4f}, {k['ci95'][1]:+.4f}]")
    print("seed-mean metrics:")
    for key in sorted(means):
        if not key.startswith(("knockout/", "params/")):
            print(f"  {key:<34} {means[key]:.4f}")
    print(f"task types: {TASK_TYPES}")
    print("=== END REPORT ===")


def run_protocol(args):
    t0, quick = time.time(), args.quick
    budget = BUDGET["quick" if quick else "full"]
    seeds = [args.seed + i for i in range(args.seeds or (1 if quick else 5))]
    print(f"{os.path.basename(__file__)}: seeds {seeds}, mode {'quick' if quick else 'full'}, mutant {MUTANT}")
    corr = []
    ok, grad = c1_gradcheck(args.seed)
    corr.append({"id": "C1", "name": "gradient_check", "passed": ok, "detail": f"max rel error {grad['max_rel_error']:.1e}"})
    ok, detail = c2_determinism(args.seed)
    corr.append({"id": "C2", "name": "determinism_and_finiteness", "passed": ok, "detail": detail})
    ok, detail, ctx = c3_learning(args.seed, budget)
    corr.append({"id": "C3", "name": "learning", "passed": ok, "detail": detail})
    ok, detail = c4_shuffled(args.seed, budget, ctx)
    corr.append({"id": "C4", "name": "shuffled_label_control", "passed": ok, "detail": detail})
    if MUTANT is None:
        ok, mut = c5_mutants(args.seed)
    else:
        ok, mut = True, {"detected": 0, "total": 0, "score": None, "by_mutant": "skipped while a mutant is active"}
    corr.append({"id": "C5", "name": "mutant_detection", "passed": ok, "detail": f"{mut['detected']}/{mut['total']} {mut['by_mutant']}"})
    for cid, name, (ok, detail) in (("C6.1", "padding_guarantee", c6_padding(args.seed, trained=ctx["model"])),
                                    ("C6.2", "confinement_definition", c6_confinement(args.seed)),
                                    ("C6.3", "graded_profile_definition", c6_graded(args.seed)),
                                    ("C6.4", "marking_rule", c6_marking(args.seed))):
        corr.append({"id": cid, "name": name, "passed": ok, "detail": detail})
    ok, detail = c7_splits(ctx["ep"])
    corr.append({"id": "C7", "name": "split_integrity", "passed": ok, "detail": detail})
    per_seed = [run_seed(s, budget) for s in seeds]
    means = {k: float(np.mean([s[k] for s in per_seed])) for k in per_seed[0]}
    hyps, knocks = evaluate_hypotheses(per_seed, args.seed, quick)
    runtime = time.time() - t0
    limit = 20.0 if quick else 180.0
    corr.append({"id": "C8", "name": "budget", "passed": runtime <= limit, "detail": f"{runtime:.1f} s <= {limit:.0f} s"})
    code = 0 if all(r["passed"] for r in corr if r["id"] != "C8") else 1
    code = 3 if code == 0 and runtime > limit else code
    rep = {"schema_version": "1.0", "chapter": 243, "file": os.path.basename(__file__), "card_revision": MIND_CARD["card_revision"],
           "environment": {"python": sys.version.split()[0], "numpy": np.__version__}, "seeds": seeds, "runtime_s": round(runtime, 2),
           "n_params": int(means["params/mikwa_total"]), "gradcheck": grad,
           "correctness": [{"id": r["id"], "name": r["name"], "passed": bool(r["passed"]), "detail": r["detail"]} for r in corr],
           "mutants": {"detected": mut["detected"], "total": mut["total"], "score": mut["score"]},
           "hypotheses": hyps, "knockouts": knocks, "task_types": TASK_TYPES, "metrics_seed_mean": means,
           "real_data": real_data_bridge(args.data, args.seed), "exit_code": code}
    print_report(rep, corr, hyps, knocks, means)
    if args.json:
        write_report_json(args.json, rep)
    return code


# ------------------------------------------------------------------ command line
def main(argv=None):
    global MUTANT
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__), description="MIKWA protocol for chapter 0243")
    for flag, kw in (("--quick", {"action": "store_true"}), ("--seed", {"type": int, "default": 0}), ("--seeds", {"type": int}),
                     ("--json", {}), ("--card", {"action": "store_true"}), ("--mutant", {"choices": sorted(MUTANTS)}), ("--data", {})):
        ap.add_argument(flag, **kw)
    args = ap.parse_args(argv)
    if args.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    MUTANT = args.mutant
    try:
        return run_protocol(args)
    except FloatingPointError as err:
        print(f"non-finite values: {err}")
        return 4


if __name__ == "__main__":
    sys.exit(main())

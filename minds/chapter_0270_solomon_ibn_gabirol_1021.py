#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0270 · Solomon ibn Gabirol (Avicebron)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 14, Minds 261-280 Available on Amazon https://www.amazon.com/dp/B0HLYSVY5D
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0270_solomon_ibn_gabirol_1021 - Solomon ibn Gabirol (Avicebron) (1021-1058)
# END ATTRIBUTION
"""
Resolutive form stack on a universal matter (via resolutoria).

Thesis
    Even the knower is matter bearing an ordered plurality of forms, so to know a
    thing is to resolve it, outermost and most particular form first, down to one
    universal matter that every thing shares, losing nothing of that matter on
    the way.

Evidence and provenance  (provenance: belief)
    Grounded in the Fons Vitae, which survives complete only in the 12th-century
    Latin of Dominicus Gundissalinus and John of Spain (ed. Baeumker 1892-95),
    plus Falaquera's 13th-century Hebrew abridgement; the Arabic original is lost
    except for fragments.  The resolutive method is described most sharply by a
    hostile witness, Thomas Aquinas, De substantiis separatis c.5-6.
    D1 FV 1.7: three divisions of knowledge; Will mediates between the extremes.
    D2 FV 1.10, 5.22: universal matter exists per se, is one in number, sustains
       diversity and is receptive of all forms.
    D3 FV IV: intellect and soul are themselves matter + form (only God is not).
    D4 Plurality of forms: one thing bears several essential forms in layers.
    D5 Aquinas, De subst. sep. c.5: "quadam resolutoria via processit" - he
       resolved artefact to element to body to substance to spiritual matter.
    D6 Aquinas, ibid.: his error was to take the intelligible composition of genus
       and difference for a real composition (genus = matter, difference = form).
    D7 FV 5.39, 1.2, 1.5: the Will is the power that joins, moves and sustains
       the union of form and matter.
    D8 Aquinas, ibid. c.5-6: he held that being in potency, being a subject and
       receiving mean the same at every level (univocal reception).

Doctrine -> mechanism -> test
    D2 -> M3 matter closure (shared learned residue m0)  -> C6.3 -> H-MATTER
    D4, D5 -> M1 ordered form stack, peeled outermost first -> C6.1, C6.2 -> H-SIG, H-NEC
    D3, D8 -> univocal operator family at every level, incl. the reader -> C6.1
    D7 -> M2 Will: per-level soft selector over candidate forms -> C1, C5
    D6 -> blind spot: holistic kinds -> H-BLIND; unicity rival -> H-RIVAL

Research question
    Compositional generalization: does resolving inputs through an ordered stack
    of level-specific invertible operators onto one shared substrate let a learner
    recognise rare combinations of known factors, and what does that prior cost
    when the world's kinds are not really composed that way?

Closest prior art and the delta
    Learned canonicalization for equivariance (Kaba et al. 2023), transformation-
    invariant clustering (Frey & Jojic 2003), transforming auto-encoders (Hinton,
    Krizhevsky & Wang 2011); Cayley-parameterized orthogonal weights (Helfrich,
    Willmott & Ye 2018) supply the operator parameterization.  Delta: the input is
    canonicalized as an ORDERED STACK of discrete per-level choices among learned
    non-commuting orthogonal-affine operators, peeled outermost first, and a
    closure loss requires the residue of every thing to coincide with ONE learned
    substrate.  The size-matched MLP is the declared baseline; the unicity rival
    (one whole form per kind, Aquinas's position) is implemented from its
    description, not copied.

Blind spot
    Aquinas's objection made operational: when each kind is holistic (its genus
    and difference are only logical, not really composed), the ordered stack
    imposes a factorization the world does not have and should lose to both the
    MLP and the unicity rival.

Task (generative process)
    K=3 levels (genus, difference, accident), N=4 forms per level, 64 kinds.
    Each form is a random rotation R and translation t in a d=6 matter space.
    A thing = shared matter m0 + individual matter (anisotropic 'grain') that has
    received, in order, genus, difference and accident: c = O3(O2(O1(m))).
    The senses see x = E c + noise through an unknown D=10 window E.
    16 'rare' kinds (sum of indices divisible by 4) have 2 training examples;
    48 common kinds have 20.  Splits: in-distribution (common kinds), shifted
    (fresh samples of the rare kinds).  Holistic world: every kind gets one
    unrelated operator, so genus/difference labels carry no composition.

Limits
    Synthetic data; linear senses; a known number of levels and forms.  The file
    is a research prototype of an AGI-oriented architecture, not an AGI, and it
    replicates no one's mind.  Nothing here is evidence about the historical
    person; the results only test the idea.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 1,
    "revision_log": [
        "r1: task constants (form scales, grain, sample sizes, lambda, steps) and all hypotheses fixed after a "
        "two-seed pilot on reserved seeds 998-999, which are excluded from evaluation; the pilot replaced a "
        "linear Will by a one-hidden-layer Will because the linear one could not separate outermost forms."],
    "generation": {"template_version": "1.0", "generator": "Claude", "generator_version": "claude-opus-5",
                   "date": "2026-09-22"},
    "id": 270, "figure": "Solomon ibn Gabirol", "born": 1021, "died": 1058, "civilization": "Andalusi Jewish",
    "provenance": "belief",
    "thesis": ("Even the knower is matter bearing an ordered plurality of forms, so to know a thing is to resolve "
               "it, outermost form first, down to one universal matter every thing shares, losing none of it."),
    "evidence": [
        {"id": "D1", "claim": "Three divisions of knowledge (matter and form, Will, First Essence); Will is the "
         "intermediary between the two extremes", "basis": "primary", "source": "Fons Vitae 1.7 (ed. Baeumker)"},
        {"id": "D2", "claim": "Universal matter exists per se, is one in number, sustains diversity and receives "
         "all forms", "basis": "primary", "source": "Fons Vitae 1.10; 5.22"},
        {"id": "D3", "claim": "Intellect and soul are composed of matter and form; only God is not",
         "basis": "primary", "source": "Fons Vitae IV; Pessin, SEP 'Solomon Ibn Gabirol' (2010) s.5"},
        {"id": "D4", "claim": "Each existent bears several essential forms in layers (plurality of forms)",
         "basis": "scholarship", "source": "Pessin, SEP s.5.3; Pessin, Ibn Gabirol's Theology of Desire (2013)"},
        {"id": "D5", "claim": "He proceeded by a resolutive way from artefacts to elements, body, substance and "
         "spiritual matter", "basis": "scholarship", "source": "Aquinas, De substantiis separatis c.5 (hostile witness)"},
        {"id": "D6", "claim": "Aquinas: he took the intelligible composition of genus and difference for a real "
         "composition", "basis": "scholarship", "source": "Aquinas, De substantiis separatis c.5-6; Brunner 1965"},
        {"id": "D7", "claim": "The Will joins, moves and sustains the union of form and matter",
         "basis": "primary", "source": "Fons Vitae 5.39; 1.2; 1.5"},
        {"id": "D8", "claim": "Reception is univocal: every higher form rests on a matter already formed",
         "basis": "scholarship", "source": "Aquinas, De substantiis separatis c.5-6"},
    ],
    "research_question": {"category": "compositional and systematic generalization",
                          "question": "Does ordered resolution onto one shared substrate let a learner recognise "
                                      "rare combinations of known factors, and what does it cost when kinds are "
                                      "holistic?"},
    "mechanism": {
        "name": "Resolutive form stack on a universal matter",
        "family": "learned canonicalization / operator factorization",
        "signature_modules": ["forms", "will", "matter"],
        "closest_prior_art": ["Kaba et al. 2023, Equivariance with learned canonicalization functions (ICML)",
                              "Frey & Jojic 2003, Transformation-invariant clustering using the EM algorithm (TPAMI)",
                              "Hinton, Krizhevsky & Wang 2011, Transforming auto-encoders (ICANN)",
                              "Helfrich, Willmott & Ye 2018, Orthogonal RNNs with scaled Cayley transform (ICML)"],
        "overlap": "Medium",
        "prior_art_queries": ["learned canonicalization network", "transformation invariant clustering",
                              "compositional generalization operator factorization",
                              "nested transformations inference peeling", "Cayley orthogonal parameterization"],
        "contribution_type": "mechanism",
        "delta": ("Canonicalization as an ordered stack of per-level discrete choices among learned non-commuting "
                  "orthogonal-affine operators, peeled outermost first, with a closure loss that sends every "
                  "thing's residue to one learned substrate."),
    },
    "traceability": [
        {"doctrine": "D4", "mechanism": "M1", "property_test": "C6.2", "hypothesis": "H-SIG"},
        {"doctrine": "D5", "mechanism": "M1", "property_test": "C6.1", "hypothesis": "H-NEC"},
        {"doctrine": "D2", "mechanism": "M3", "property_test": "C6.3", "hypothesis": "H-MATTER"},
        {"doctrine": "D7", "mechanism": "M2", "property_test": "C1", "hypothesis": "H-NEC"},
        {"doctrine": "D6", "mechanism": "M1", "property_test": "C6.2", "hypothesis": "H-BLIND"},
        {"doctrine": "D6", "mechanism": "M1", "property_test": "C6.2", "hypothesis": "H-RIVAL"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "The resolutive stack recognises rare combinations of known forms better than a "
         "size-matched MLP", "metric": "full_kind_accuracy", "split": "shifted",
         "comparison": "model - baseline", "direction": "greater", "mesi": 0.10, "seeds": 5},
        {"id": "H-NEC", "statement": "Removing the ordered un-imprinting costs more than removing the per-channel "
         "input gain", "metric": "full_kind_accuracy", "split": "shifted",
         "comparison": "signature_knockout - matched_knockout", "direction": "less", "mesi": 0.10, "seeds": 5},
        {"id": "H-BLIND", "statement": "Where kinds are holistic the stack is worse than the MLP",
         "condition": "holistic world: one unrelated operator per kind",
         "grounding": "Aquinas: genus and difference were taken for a real composition (De subst. sep. c.5)",
         "metric": "full_kind_accuracy", "split": "in_distribution", "comparison": "model - baseline",
         "direction": "less", "mesi": 0.05, "seeds": 5},
        {"id": "H-RIVAL", "statement": "Plurality of forms beats unicity (one whole form per kind) on rare kinds",
         "metric": "full_kind_accuracy", "split": "shifted", "comparison": "model - rival",
         "direction": "greater", "mesi": 0.10, "seeds": 5},
        {"id": "H-MATTER", "statement": "The universal-matter closure term improves rare-kind recognition",
         "metric": "full_kind_accuracy", "split": "shifted", "comparison": "model - model_without_closure",
         "direction": "greater", "mesi": 0.05, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.5, "margin_over_trivial": 0.30, "shuffled_band": 0.05,
                   "gradcheck_rel_tol": 1e-5},
    "probe_predictions": [{"probe": "P3", "expected": "above"}, {"probe": "P6", "expected": "above"},
                          {"probe": "P10", "expected": "above"}, {"probe": "P1", "expected": "below"},
                          {"probe": "P4", "expected": "equal to baseline"},
                          {"probe": "P8", "expected": "equal to baseline"},
                          {"probe": "P9", "expected": "equal to baseline"}],
    "dialectic_links": [
        {"chapter": 337, "relation": "rival", "test": "H-RIVAL"},
        {"chapter": 143, "relation": "teacher", "test": "none (indirect, via the Arabic Plotinus)"},
        {"chapter": 335, "relation": "successor", "test": "none (adopted universal hylomorphism)"},
        {"chapter": 351, "relation": "successor", "test": "none (Franciscan plurality of forms)"}],
    "corpus_neighbors": [
        {"chapter": 235, "similarity": None, "difference": "al-Farabi audits one lossy passage between two "
         "encodings; here many ordered, lossless operators are peeled to a shared substrate"},
        {"chapter": 218, "similarity": None, "difference": "al-Kindi keeps the invariant and discards the "
         "relabelling; here the sequence of operators is itself the knowledge"},
        {"chapter": 213, "similarity": None, "difference": "al-Khwarizmi reduces to a closed canon of six fixed "
         "shapes; here the canon is learned, continuous and layered"},
        {"chapter": 143, "similarity": None, "difference": "Plotinus subtracts determinations to reach the One; "
         "here stripping is an inverse that keeps matter intact and ends below, not above"},
        {"chapter": 264, "similarity": None, "difference": "Ramanuja couples private parameters to one shared "
         "controller; here the shared element is the substrate, and the controllers are per level"}],
    "similarity_note": "Token-shingle similarity not computed: corpus code files unavailable in this session.",
    "barometer": {
        "cognitive_processing": ["P3 compositional split", "rare-kind full accuracy"],
        "embodied_cognition": ["not measured: linear senses only"],
        "world_modeling": ["recovery of the generative order of forms (C6.2)"],
        "consciousness": ["not claimed"],
        "language_understanding": ["not measured"],
        "emotional_intelligence": ["not measured"],
        "creativity": ["imprinting unseen form combinations onto matter (reported, not tested)"],
        "autonomy": ["not measured"]},
    "task_types": ["vector_classification"],
    "applications": [
        {"use": "Recognising rare object-pose-lighting combinations", "sector": "robot perception",
         "dataset": "smallNORB (LeCun, Huang & Bottou 2004)"},
        {"use": "Recovering generative factors for unseen factor combinations", "sector": "interpretability research",
         "dataset": "3D Shapes (Burgess & Kim 2018, DeepMind)"},
        {"use": "Compositional attribute recognition with held-out combinations", "sector": "computer vision",
         "dataset": "CLEVR CoGenT (Johnson et al. 2017)"}],
    "safety_notes": ("No hazardous domain. Synthetic data only. The five-senses map of moral qualities from the "
                     "Improvement of the Moral Qualities is treated as history in the chapter and is not encoded "
                     "as a predictive feature about people."),
}

# ============================================================================ imports and constants
import argparse
import hashlib
import itertools
import json
import math
import os
import sys
import time

import numpy as np

CHAPTER = 270
K_LEVELS, N_FORMS, D_MATTER, D_SENSE = 3, 4, 6, 10
LEVEL_NAMES = ("genus", "difference", "accident")          # inner (most general) to outer
FORM_SCALES = (1.6, 2.2, 3.0)                               # outer forms are the most manifest
GRAIN = (1.0, 0.6, 0.25, 0.25, 0.25, 0.25)                  # individual matter varies along a grain
SENSE_NOISE, MATTER_NORM = 0.15, 1.2
N_COMMON, K_RARE, N_TEST_ID, N_TEST_SHIFT = 20, 2, 10, 30
WILL_HIDDEN, LAMBDA_MATTER, TAU_FRACTION = 8, 0.10, 0.40
STEPS_FULL, STEPS_QUICK, STEPS_SHORT = 2500, 700, 400
LR, BATCH, CLIP = 0.01, 64, 5.0
BUDGET_FULL, BUDGET_QUICK = 180.0, 20.0
TASK_TYPES = ["vector_classification"]
ACTIVE_MUTANT = {"name": None}

# ============================================================================ BEGIN STANDARD UTILITIES v1.0
def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def logsumexp(z, axis=-1):
    m = z.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(z - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def softplus(z):
    return np.logaddexp(0.0, z)


class Adam:
    """Adam with bias correction; state keyed like the parameter dict."""

    def __init__(self, params, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads, lr=None, sign=1.0):
        self.t += 1
        lr = self.lr if lr is None else lr
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= sign * lr * mh / (np.sqrt(vh) + self.eps)


def clip_global(grads, max_norm):
    norm = math.sqrt(sum(float((g * g).sum()) for g in grads.values()))
    scale = min(1.0, max_norm / (norm + 1e-12))
    return {k: g * scale for k, g in grads.items()}, norm


def finite_difference_check(loss_fn, params, grads, rng, n_entries=20, eps=1e-6, floor=1e-4):
    """Central differences on >= n_entries random entries per tensor plus its largest-gradient entry.
    Relative error |a-n| / max(|a|+|n|, floor): float64 central differences carry ~1e-10 absolute
    round-off, so gradients below `floor` are compared on that absolute scale."""
    worst, report = 0.0, {}
    for name, p in params.items():
        flat, gflat = p.reshape(-1), grads[name].reshape(-1)
        idx = set(rng.choice(flat.size, min(n_entries, flat.size), replace=False).tolist())
        idx.add(int(np.argmax(np.abs(gflat))))
        tensor_worst = 0.0
        for j in idx:
            old = flat[j]
            flat[j] = old + eps
            lp = loss_fn()
            flat[j] = old - eps
            lm = loss_fn()
            flat[j] = old
            num = (lp - lm) / (2 * eps)
            rel = abs(num - gflat[j]) / max(abs(num) + abs(gflat[j]), floor)
            tensor_worst = max(tensor_worst, rel)
        report[name] = tensor_worst
        worst = max(worst, tensor_worst)
    return worst, report


def paired_bootstrap(diffs, rng, n_boot=2000):
    diffs = np.asarray(diffs, dtype=float)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return float(diffs.mean()), [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def verdict(mean, ci, direction, mesi):
    sign = 1.0 if direction == "greater" else -1.0
    if sign * ci[0] > 0 and sign * ci[1] > 0 and sign * mean >= mesi:
        return "supported"
    if sign * ci[0] < 0 and sign * ci[1] < 0:
        return "contradicted"
    return "inconclusive"


def write_report(path, report):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
# ============================================================================ END STANDARD UTILITIES


# ============================================================================ data and tasks
def random_rotation(rng, n):
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def all_kinds():
    return list(itertools.product(range(N_FORMS), repeat=K_LEVELS))


def rare_kinds():
    """Latin-square set: every pair of forms still occurs among the common kinds."""
    return [kd for kd in all_kinds() if sum(kd) % N_FORMS == 0]


def make_world(rng, holistic=False):
    w = {"E": np.linalg.qr(rng.standard_normal((D_SENSE, D_MATTER)))[0], "holistic": holistic,
         "grain": np.array(GRAIN)}
    m0 = rng.standard_normal(D_MATTER)
    w["m0"] = m0 * MATTER_NORM / np.linalg.norm(m0)
    w["R"] = [[random_rotation(rng, D_MATTER) for _ in range(N_FORMS)] for _ in range(K_LEVELS)]
    w["t"] = []
    for k in range(K_LEVELS):
        t = rng.standard_normal((N_FORMS, D_MATTER))
        w["t"].append(t / np.linalg.norm(t, axis=1, keepdims=True) * FORM_SCALES[k])
    if holistic:                                  # one indivisible form per kind (the blind spot)
        norm = math.sqrt(sum(s * s for s in FORM_SCALES))
        w["Rh"] = {kd: random_rotation(rng, D_MATTER) for kd in all_kinds()}
        w["th"] = {}
        for kd in all_kinds():
            v = rng.standard_normal(D_MATTER)
            w["th"][kd] = v * norm / np.linalg.norm(v)
    return w


def imprint_true(w, m, kind):
    if w["holistic"]:
        return m @ w["Rh"][kind].T + w["th"][kind]
    c = m
    for k in range(K_LEVELS):                     # genus first, accident last: forms received in order
        c = c @ w["R"][k][kind[k]].T + w["t"][k][kind[k]]
    return c


def sample(w, kinds, n_each, rng):
    xs, ys = [], []
    for kind in kinds:
        m = w["m0"] + rng.standard_normal((n_each, D_MATTER)) * w["grain"]
        c = imprint_true(w, m, kind)
        xs.append(c @ w["E"].T + SENSE_NOISE * rng.standard_normal((n_each, D_SENSE)))
        ys.append(np.tile(np.array(kind), (n_each, 1)))
    return np.concatenate(xs), np.concatenate(ys)


def make_task(seed_seq, holistic=False):
    rng_w, rng_d = [np.random.default_rng(s) for s in seed_seq.spawn(2)]
    w = make_world(rng_w, holistic)
    rare = [] if holistic else rare_kinds()
    common = [kd for kd in all_kinds() if kd not in rare]
    xc, yc = sample(w, common, N_COMMON, rng_d)
    task = {"world": w, "rare": rare, "common": common}
    if rare:
        xr, yr = sample(w, rare, K_RARE, rng_d)
        task["train"] = (np.concatenate([xc, xr]), np.concatenate([yc, yr]))
        task["shift"] = sample(w, rare, N_TEST_SHIFT, rng_d)
    else:
        task["train"] = (xc, yc)
    task["id"] = sample(w, common, N_TEST_ID, rng_d)
    return task


def kind_index(y):
    return (y[:, 0] * N_FORMS + y[:, 1]) * N_FORMS + y[:, 2]


def kind_decode(u):
    return np.stack([u // (N_FORMS * N_FORMS), (u // N_FORMS) % N_FORMS, u % N_FORMS], axis=1)


def full_accuracy(pred, y):
    pred = pred.reshape(len(y), -1)
    y = y.reshape(len(y), -1)
    return float((pred == y).all(axis=1).mean())


# ============================================================================ model
def cayley(a):
    """Orthogonal R = (I+S)^-1 (I-S) from skew S = A - A^T; batched over leading axes."""
    s = a - np.swapaxes(a, -1, -2)
    eye = np.eye(a.shape[-1])
    m = np.linalg.inv(eye + s)
    return m @ (eye - s), m


def build_model(in_dim, out_dim, task_type, rng, levels=None, d=D_MATTER, will_hidden=WILL_HIDDEN,
                lam=LAMBDA_MATTER):
    """Resolutive form stack. levels[k] = number of candidate forms at level k (k=0 innermost)."""
    assert task_type in TASK_TYPES
    levels = tuple(levels) if levels is not None else (out_dim,)
    p = {"sense_gain": np.ones(in_dim), "sense_P": rng.standard_normal((d, in_dim)) * 0.4,
         "sense_b": np.zeros(d), "matter": np.zeros(d)}
    for k, n in enumerate(levels):
        p[f"form_A{k}"] = rng.standard_normal((n, d, d)) * 0.1
        p[f"form_t{k}"] = rng.standard_normal((n, d)) * 0.3
        p[f"will_U{k}"] = rng.standard_normal((will_hidden, d)) / math.sqrt(d)
        p[f"will_a{k}"] = np.zeros(will_hidden)
        p[f"will_V{k}"] = rng.standard_normal((n, will_hidden)) * 0.1
        p[f"will_b{k}"] = np.zeros(n)
    return {"kind": "stack", "params": p, "levels": levels, "lam": lam, "ko": {}}


def _targets(y, levels):
    y = np.asarray(y).reshape(len(y), -1)
    assert y.shape[1] == len(levels)
    return y


def stack_forward(model, x, y=None, tau=0.0):
    """Peel forms outermost first: the Will at level k reads the composite as it stands after the outer
    forms have been removed, weighs the candidates, and the composite is un-imprinted by that mixture."""
    p, levels, ko = model["params"], model["levels"], model["ko"]
    b = x.shape[0]
    gain = np.ones_like(p["sense_gain"]) if ko.get("sense_gain") == "identity" else p["sense_gain"]
    xg = x * gain
    c = xg @ p["sense_P"].T + p["sense_b"]
    cache = {"xg": xg, "gain": gain, "lv": [], "b": b, "tau": tau, "v": c}
    loss, preds = 0.0, np.zeros((b, len(levels)), dtype=int)
    yy = _targets(y, levels) if y is not None else None
    for k in range(len(levels) - 1, -1, -1):
        n = levels[k]
        h = np.tanh(c @ p[f"will_U{k}"].T + p[f"will_a{k}"])
        z = h @ p[f"will_V{k}"].T + p[f"will_b{k}"]
        if ko.get("will") in ("zero", "mean"):
            z = np.zeros_like(z)
        wk = softmax(z)
        preds[:, k] = wk.argmax(1)
        onehot = np.eye(n)[yy[:, k]] if yy is not None else np.zeros_like(wk)
        if yy is not None:
            loss += -float(np.log(wk[np.arange(b), yy[:, k]] + 1e-300).mean())
        om = (1.0 - tau) * wk + tau * onehot
        r, m = cayley(p[f"form_A{k}"])
        diff = c[:, None, :] - p[f"form_t{k}"][None]
        us = np.einsum("bnd,nde->bne", diff, r)
        cache["lv"].append((k, c, h, wk, om, r, m, diff, us, onehot))
        mode = ko.get("forms")
        if mode == "identity":
            continue
        c = np.einsum("bn,bne->be", om, us)
        if mode == "zero":
            c = np.zeros_like(c)
        elif mode == "mean":
            c = np.repeat(c.mean(0, keepdims=True), b, axis=0)
    resid = c - p["matter"]
    loss += model["lam"] * float((resid * resid).sum(1).mean())
    cache["resid"], cache["c0"] = resid, c
    return loss, preds, cache


def stack_backward(model, x, y, cache):
    p, levels = model["params"], model["levels"]
    b, tau, lam = cache["b"], cache["tau"], model["lam"]
    g = {k: np.zeros_like(v) for k, v in p.items()}
    grad_c = 2.0 * lam * cache["resid"] / b
    g["matter"] = -grad_c.sum(0)
    eye = np.eye(p["sense_P"].shape[0])
    for (k, c, h, wk, om, r, m, diff, us, onehot) in reversed(cache["lv"]):
        d_om = np.einsum("be,bne->bn", grad_c, us)
        du = om[:, :, None] * grad_c[:, None, :]
        d_r = np.einsum("bnd,bne->nde", diff, du)
        d_diff = np.einsum("bne,nde->bnd", du, r)
        dc = d_diff.sum(1)
        g[f"form_t{k}"] = -d_diff.sum(0)
        d_s = -np.swapaxes(m, -1, -2) @ d_r @ np.swapaxes(r + eye, -1, -2)
        g[f"form_A{k}"] = d_s - np.swapaxes(d_s, -1, -2)
        dw = (1.0 - tau) * d_om
        dz = wk * (dw - (dw * wk).sum(1, keepdims=True)) + (wk - onehot) / b
        g[f"will_V{k}"] = dz.T @ h
        g[f"will_b{k}"] = dz.sum(0)
        dh = (dz @ p[f"will_V{k}"]) * (1.0 - h * h)
        g[f"will_U{k}"] = dh.T @ c
        g[f"will_a{k}"] = dh.sum(0)
        grad_c = dc + dh @ p[f"will_U{k}"]
    g["sense_P"] = grad_c.T @ cache["xg"]
    g["sense_b"] = grad_c.sum(0)
    g["sense_gain"] = (x * (grad_c @ p["sense_P"])).sum(0)
    return g


# ============================================================================ baselines and rival mechanisms
def build_mlp(in_dim, out_sizes, hidden, rng):
    """Size-matched baseline: one tanh hidden layer, one softmax head per level."""
    out = int(sum(out_sizes))
    p = {"W1": rng.standard_normal((hidden, in_dim)) / math.sqrt(in_dim), "b1": np.zeros(hidden),
         "W2": rng.standard_normal((out, hidden)) / math.sqrt(hidden), "b2": np.zeros(out)}
    return {"kind": "mlp", "params": p, "levels": tuple(out_sizes), "ko": {}}


def mlp_forward(model, x, y=None, tau=0.0):
    p, levels = model["params"], model["levels"]
    h = np.tanh(x @ p["W1"].T + p["b1"])
    z = h @ p["W2"].T + p["b2"]
    offs = np.cumsum((0,) + levels)
    ws = [softmax(z[:, offs[k]:offs[k + 1]]) for k in range(len(levels))]
    preds = np.stack([w.argmax(1) for w in ws], axis=1)
    loss = 0.0
    if y is not None:
        yy = _targets(y, levels)
        loss = -sum(float(np.log(ws[k][np.arange(len(x)), yy[:, k]] + 1e-300).mean()) for k in range(len(levels)))
    return loss, preds, {"x": x, "h": h, "ws": ws, "offs": offs}


def mlp_backward(model, x, y, cache):
    p, levels = model["params"], model["levels"]
    yy = _targets(y, levels)
    b = len(x)
    dz = np.concatenate([cache["ws"][k] - np.eye(levels[k])[yy[:, k]] for k in range(len(levels))], axis=1) / b
    g = {"W2": dz.T @ cache["h"], "b2": dz.sum(0)}
    dh = (dz @ p["W2"]) * (1.0 - cache["h"] ** 2)
    g["W1"], g["b1"] = dh.T @ x, dh.sum(0)
    return g


def matched_hidden(stack_params, in_dim, out_sizes):
    out = sum(out_sizes)
    return max(2, int(round((stack_params - out) / (in_dim + 1 + out))))


def build_unicity(in_dim, rng):
    """Aquinas's rival: one substantial form per whole kind (a depth-one stack over all 64 kinds)."""
    return build_model(in_dim, N_FORMS ** K_LEVELS, "vector_classification", rng,
                       levels=(N_FORMS ** K_LEVELS,))


# ============================================================================ registries: interface, modules, knockouts, mutants
class NonFiniteError(RuntimeError):
    pass


MUTANTS = {
    "sign_flip_update": "optimizer ascends the loss (update sign flipped)",
    "zero_lr": "learning rate forced to zero",
    "zero_grad_forms": "analytic gradient of every form operator tensor zeroed",
    "zero_grad_will": "analytic gradient of the Will output layer zeroed",
}


def _fwd(model):
    return stack_forward if model["kind"] == "stack" else mlp_forward


def _bwd(model):
    return stack_backward if model["kind"] == "stack" else mlp_backward


def loss_and_grads(model, batch):
    x, y, tau = batch["X"], batch["Y"], batch.get("tau", 0.0)
    loss, _, cache = _fwd(model)(model, x, y, tau)
    grads = _bwd(model)(model, x, y, cache)
    mut = ACTIVE_MUTANT["name"]
    for name in grads:
        if (mut == "zero_grad_forms" and name.startswith("form_A")) or \
           (mut == "zero_grad_will" and name.startswith("will_V")):
            grads[name] = np.zeros_like(grads[name])
    return loss, grads


def fit(model, data, budget, rng):
    """Adam + global-norm clipping + cosine decay; the stack is first led by the true forms (teacher
    forcing tau: 1 -> 0 over the first TAU_FRACTION of steps), as the master leads the disciple."""
    x, y = data
    opt = Adam(model["params"], lr=LR)
    mut = ACTIVE_MUTANT["name"]
    history = []
    for t in range(1, budget + 1):
        idx = rng.integers(0, len(x), BATCH)
        tau = max(0.0, 1.0 - t / (TAU_FRACTION * budget)) if model["kind"] == "stack" else 0.0
        loss, grads = loss_and_grads(model, {"X": x[idx], "Y": y[idx], "tau": tau})
        if not np.isfinite(loss):
            raise NonFiniteError(f"non-finite loss at step {t}")
        grads, _ = clip_global(grads, CLIP)
        lr = LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t / budget)))
        opt.step(model["params"], grads, lr=0.0 if mut == "zero_lr" else lr,
                 sign=-1.0 if mut == "sign_flip_update" else 1.0)
        history.append(loss)
    for v in model["params"].values():
        if not np.all(np.isfinite(v)):
            raise NonFiniteError("non-finite parameter after training")
    return history


def predict(model, x):
    _, preds, _ = _fwd(model)(model, x)
    return preds[:, 0] if preds.shape[1] == 1 else preds


def hidden_states(model, x):
    """Composite after the senses and after each peel (stack), or the hidden layer (MLP)."""
    _, _, cache = _fwd(model)(model, x)
    if model["kind"] == "mlp":
        return {"hidden": cache["h"]}
    states = {"senses": cache["v"]}
    for (k, c, *_rest) in cache["lv"]:
        states[f"before_level_{k}"] = c
    states["residue"] = cache["c0"]
    return states


def modules(model):
    if model["kind"] == "mlp":
        return {"hidden": {"params": ["W1", "b1"], "role": "tanh hidden layer", "signature": False},
                "heads": {"params": ["W2", "b2"], "role": "per-level softmax heads", "signature": False}}
    p = model["params"]
    return {
        "senses": {"params": ["sense_P", "sense_b"], "role": "linear projection of the input into the substrate",
                   "signature": False},
        "sense_gain": {"params": ["sense_gain"], "role": "per-channel input gain", "signature": False},
        "forms": {"params": [k for k in p if k.startswith("form_")],
                  "role": "ordered stack of Cayley-orthogonal affine operators, inverted outermost first",
                  "signature": True},
        "will": {"params": [k for k in p if k.startswith("will_")],
                 "role": "per-level one-hidden-layer soft selector over candidate operators", "signature": True},
        "matter": {"params": ["matter"], "role": "shared learned residue target (closure loss; training only)",
                   "signature": True},
    }


def knockout(model, name, mode):
    assert mode in ("identity", "zero", "mean")
    new = {"kind": model["kind"], "params": {k: v.copy() for k, v in model["params"].items()},
           "levels": model["levels"], "lam": model.get("lam", 0.0), "ko": dict(model["ko"])}
    if name in ("forms", "will", "sense_gain"):
        new["ko"][name] = mode
    elif name == "senses":
        new["params"]["sense_P"][:] = 0.0
    elif name == "matter":                       # definition check only: matter is not used at inference
        new["params"]["matter"][:] = 0.0
    else:
        raise KeyError(name)
    return new


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


# ============================================================================ training helpers
def new_stack(seed_seq, lam=LAMBDA_MATTER, levels=(N_FORMS,) * K_LEVELS):
    rng = np.random.default_rng(seed_seq)
    return build_model(D_SENSE, N_FORMS, "vector_classification", rng, levels=levels, lam=lam)


def evaluate(model, x, y):
    pred = predict(model, x)
    if model["levels"] == (N_FORMS ** K_LEVELS,):
        pred = kind_decode(pred)
    return full_accuracy(pred, y)


def train_trio(task, seed_seq, steps, with_rival=True, with_ablation=False):
    """Stack, size-matched MLP, and (optionally) the unicity rival and the no-closure stack."""
    ss = seed_seq.spawn(5)
    x, y = task["train"]
    out = {}
    stack = new_stack(ss[0])
    out["stack"] = (stack, fit(stack, (x, y), steps, np.random.default_rng(ss[1])))
    hid = matched_hidden(n_params(stack), D_SENSE, (N_FORMS,) * K_LEVELS)
    mlp = build_mlp(D_SENSE, (N_FORMS,) * K_LEVELS, hid, np.random.default_rng(ss[0]))
    out["mlp"] = (mlp, fit(mlp, (x, y), steps, np.random.default_rng(ss[1])))
    if with_rival:
        uni = build_unicity(D_SENSE, np.random.default_rng(ss[0]))
        out["unicity"] = (uni, fit(uni, (x, kind_index(y)[:, None]), steps, np.random.default_rng(ss[1])))
    if with_ablation:
        noc = new_stack(ss[0], lam=0.0)
        out["no_closure"] = (noc, fit(noc, (x, y), steps, np.random.default_rng(ss[1])))
    return out


# ============================================================================ tests: correctness
def c1_gradcheck(task, seed_seq, tol):
    """Every tensor of the stack, the unicity rival and the MLP, at init and after 60 steps."""
    x, y = task["train"]
    rng = np.random.default_rng(seed_seq)
    idx = rng.choice(len(x), 16, replace=False)
    xb, yb = x[idx], y[idx]
    models = [("stack", new_stack(seed_seq), yb),
              ("unicity", build_unicity(D_SENSE, np.random.default_rng(seed_seq)), kind_index(yb)[:, None]),
              ("mlp", build_mlp(D_SENSE, (N_FORMS,) * K_LEVELS, 37, np.random.default_rng(seed_seq)), yb)]
    worst, checked, total = 0.0, 0, 0
    for name, model, targets in models:
        for stage in ("init", "trained"):
            if stage == "trained":
                fit(model, (xb, targets), 60, np.random.default_rng(1))
            batch = {"X": xb, "Y": targets, "tau": 0.3}
            _, grads = loss_and_grads(model, batch)
            fn = lambda m=model: _fwd(m)(m, xb, targets, 0.3)[0]
            w, rep = finite_difference_check(fn, model["params"], grads, rng)
            worst = max(worst, w)
            checked += len(rep)
            total += len(model["params"])
    return {"id": "C1", "name": "gradient_check", "passed": bool(worst <= tol),
            "detail": f"{checked}/{total} tensor checks (3 models x init/after 60 steps), max rel err {worst:.2e}",
            "max_rel_error": worst, "tensors_checked": checked, "tensors_total": total}


def c2_determinism(task, seed_seq):
    x, y = task["train"]
    runs = []
    for _ in range(2):
        m = new_stack(seed_seq)
        hist = fit(m, (x, y), 60, np.random.default_rng(7))
        runs.append((hist, predict(m, task["id"][0]), m))
    same = runs[0][0] == runs[1][0] and np.array_equal(runs[0][1], runs[1][1])
    finite = all(np.all(np.isfinite(v)) for v in runs[0][2]["params"].values()) and \
        all(np.all(np.isfinite(h)) for h in hidden_states(runs[0][2], task["id"][0]).values())
    return {"id": "C2", "name": "determinism_and_finiteness", "passed": bool(same and finite),
            "detail": f"identical losses/outputs={same}, finite={finite}"}


def trivial_full_accuracy(y):
    _, counts = np.unique(kind_index(y), return_counts=True)
    return float(counts.max() / len(y))


def c3_learning(model, history, task, th):
    first, last = float(np.mean(history[:50])), float(np.mean(history[-50:]))
    drop = 1.0 - last / first
    acc = evaluate(model, *task["id"])
    triv = trivial_full_accuracy(task["id"][1])
    ok = drop >= th["loss_drop_fraction"] and acc - triv >= th["margin_over_trivial"]
    return {"id": "C3", "name": "learning", "passed": bool(ok),
            "detail": f"loss drop {drop:.3f} (>= {th['loss_drop_fraction']}); held-out full acc {acc:.3f} vs "
                      f"trivial {triv:.3f} (margin >= {th['margin_over_trivial']})"}


def c3_short(task, seed_seq):
    m = new_stack(seed_seq)
    hist = fit(m, task["train"], STEPS_SHORT, np.random.default_rng(3))
    return 1.0 - float(np.mean(hist[-50:])) / float(np.mean(hist[:50])) >= 0.25


def c4_shuffled(task, seed_seq, steps, band):
    x, y = task["train"]
    perm = np.random.default_rng(seed_seq).permutation(len(y))
    m = new_stack(seed_seq)
    fit(m, (x, y[perm]), steps, np.random.default_rng(5))
    acc = evaluate(m, *task["id"])
    triv = trivial_full_accuracy(task["id"][1])
    return {"id": "C4", "name": "shuffled_label_control", "passed": bool(acc <= triv + band),
            "detail": f"held-out full acc with permuted labels {acc:.3f} vs trivial {triv:.3f} (band {band})"}


def c5_mutants(task, seed_seq, tol):
    """Each learning-breaking mutant must make C1 or the short learning check fail."""
    genuine_ok = c3_short(task, seed_seq)
    detected = {}
    for name in MUTANTS:
        ACTIVE_MUTANT["name"] = name
        try:
            c1_ok = c1_gradcheck(task, seed_seq, tol)["passed"] if name.startswith("zero_grad") else True
            learn_ok = c3_short(task, seed_seq) if c1_ok else False
        except NonFiniteError:
            c1_ok, learn_ok = True, False
        finally:
            ACTIVE_MUTANT["name"] = None
        detected[name] = not (c1_ok and learn_ok)
    score = sum(detected.values()) / len(detected)
    return {"id": "C5", "name": "mutant_detection", "passed": bool(genuine_ok and score == 1.0),
            "detail": f"genuine short-learning check passes={genuine_ok}; detected {detected}",
            "detected": int(sum(detected.values())), "total": len(detected)}


def strip_once(c, a, t, broken=False):
    s = a - a.T
    r = np.eye(len(a)) - s if broken else cayley(a)[0]
    return (c - t) @ r


def c6_1_isometry(seed_seq):
    """Stripping a form never destroys matter: distances between things are preserved exactly.
    Random draws plus a hill-climb on the violation; negative control: R = I - S (not orthogonal)."""
    rng = np.random.default_rng(seed_seq)

    def violation(a, t, c1, c2, broken):
        d0 = np.linalg.norm(c1 - c2, axis=1)
        d1 = np.linalg.norm(strip_once(c1, a, t, broken) - strip_once(c2, a, t, broken), axis=1)
        return float(np.abs(d1 - d0).max())

    worst, worst_broken = 0.0, 0.0
    for scale in (0.1, 0.5, 1.0, 3.0, 10.0):
        a = rng.standard_normal((D_MATTER, D_MATTER)) * scale
        t = rng.standard_normal(D_MATTER)
        c1, c2 = rng.standard_normal((32, D_MATTER)) * 3, rng.standard_normal((32, D_MATTER)) * 3
        v = violation(a, t, c1, c2, False)
        for _ in range(60):                                  # hill-climb toward a violation
            a2 = a + rng.standard_normal(a.shape) * 0.3 * scale
            v2 = violation(a2, t, c1, c2, False)
            if v2 > v:
                a, v = a2, v2
        worst = max(worst, v)
        worst_broken = max(worst_broken, violation(a, t, c1, c2, True))
    ok = worst < 1e-9 and worst_broken > 1e-3
    return {"id": "C6.1", "name": "isometry_of_forms", "passed": bool(ok),
            "detail": f"max distance violation {worst:.1e} (< 1e-9); negative control I-S violates by "
                      f"{worst_broken:.2f} (> 1e-3)"}


def _order_errors(a_stack, t_stack, m, choice):
    rs = [cayley(a_stack[k][choice[k]])[0] for k in range(K_LEVELS)]
    c = m
    for k in range(K_LEVELS):                                # imprint: genus, difference, accident
        c = c @ rs[k].T + t_stack[k][choice[k]]
    right, wrong = c.copy(), c.copy()
    for k in reversed(range(K_LEVELS)):                      # peel outermost first
        right = (right - t_stack[k][choice[k]]) @ rs[k]
    for k in range(K_LEVELS):                                # peel innermost first (wrong order)
        wrong = (wrong - t_stack[k][choice[k]]) @ rs[k]
    return float(np.abs(right - m).max()), float(np.linalg.norm(wrong - m, axis=1).mean())


def c6_2_order(seed_seq):
    """The order of forms is real: peeling outermost first recovers the matter exactly; peeling in the
    wrong order does not. Negative control: translation-only forms commute, so the check must fail."""
    rng = np.random.default_rng(seed_seq)
    a = rng.standard_normal((K_LEVELS, N_FORMS, D_MATTER, D_MATTER)) * 0.8
    t = rng.standard_normal((K_LEVELS, N_FORMS, D_MATTER)) * 2.0
    m = rng.standard_normal((64, D_MATTER))
    errs = [_order_errors(a, t, m, rng.integers(0, N_FORMS, K_LEVELS)) for _ in range(20)]
    right = max(e[0] for e in errs)
    wrong = min(e[1] for e in errs)
    errs0 = [_order_errors(np.zeros_like(a), t, m, rng.integers(0, N_FORMS, K_LEVELS)) for _ in range(20)]
    control_detects = max(e[1] for e in errs0) < 1e-9
    ok = right < 1e-9 and wrong > 0.1 and control_detects
    return {"id": "C6.2", "name": "order_of_forms_is_real", "passed": bool(ok),
            "detail": f"right-order residue error {right:.1e}; min wrong-order error {wrong:.2f} (> 0.1); "
                      f"commuting control flagged={control_detects}"}


def closure_ratio(model, x):
    _, _, cache = stack_forward(model, x)
    spread_in = np.linalg.norm(cache["v"] - cache["v"].mean(0), axis=1).mean()
    return float(np.linalg.norm(cache["resid"], axis=1).mean() / spread_in)


def c6_3_closure(model, task):
    """Trained property: every thing resolves toward ONE matter. Negative control: forms knocked out."""
    x = task["id"][0]
    r = closure_ratio(model, x)
    r_ko = closure_ratio(knockout(model, "forms", "identity"), x)
    ok = r < 0.6 and r_ko >= 0.6
    return {"id": "C6.3", "name": "universal_matter_closure", "passed": bool(ok),
            "detail": f"residue spread / input spread {r:.3f} (< 0.6); with forms knocked out {r_ko:.3f} (>= 0.6)"}


def definition_check_matter(model, task):
    same = np.array_equal(predict(model, task["id"][0]), predict(knockout(model, "matter", "zero"),
                                                                  task["id"][0]))
    return {"id": "DEF", "name": "matter_unused_at_inference (definition check, not evidence)",
            "passed": bool(same), "detail": f"predictions unchanged when the matter target is zeroed: {same}"}


def row_hashes(x):
    return {hashlib.sha1(np.ascontiguousarray(r).tobytes()).hexdigest() for r in x}


def c7_split(task):
    tr = row_hashes(task["train"][0])
    ev = row_hashes(task["id"][0]) | (row_hashes(task["shift"][0]) if "shift" in task else set())
    ytr = task["train"][1]
    counts = [int((kind_index(ytr) == kind_index(np.array([kd]))[0]).sum()) for kd in task["rare"]]
    ok = not (tr & ev) and all(c == K_RARE for c in counts)
    return {"id": "C7", "name": "split_integrity", "passed": bool(ok),
            "detail": f"train/eval overlap {len(tr & ev)}; rare kinds have {sorted(set(counts))} training examples"}


# ============================================================================ tests: hypotheses
KO_ROWS = [("forms", "identity"), ("will", "zero"), ("sense_gain", "identity"), ("senses", "zero")]


def run_seed(seed, steps):
    ss = np.random.SeedSequence(seed).spawn(4)
    comp = make_task(ss[0])
    hol = make_task(ss[1], holistic=True)
    tc = train_trio(comp, ss[2], steps, with_rival=True, with_ablation=True)
    th = train_trio(hol, ss[3], steps, with_rival=True)
    stack = tc["stack"][0]
    r = {"seed": seed, "comp_task": comp, "stack": stack, "stack_history": tc["stack"][1]}
    for name in ("stack", "mlp", "unicity", "no_closure"):
        r[f"{name}_shift"] = evaluate(tc[name][0], *comp["shift"])
        r[f"{name}_id"] = evaluate(tc[name][0], *comp["id"])
    for name in ("stack", "mlp", "unicity"):
        r[f"{name}_holistic"] = evaluate(th[name][0], *hol["id"])
    for mod, mode in KO_ROWS:
        r[f"ko_{mod}"] = evaluate(knockout(stack, mod, mode), *comp["shift"])
    r["n_params"] = {k: n_params(tc[k][0]) for k in ("stack", "mlp", "unicity")}
    return r


HYP_DIFFS = {
    "H-SIG": lambda r: r["stack_shift"] - r["mlp_shift"],
    "H-NEC": lambda r: r["ko_forms"] - r["ko_sense_gain"],
    "H-BLIND": lambda r: r["stack_holistic"] - r["mlp_holistic"],
    "H-RIVAL": lambda r: r["stack_shift"] - r["unicity_shift"],
    "H-MATTER": lambda r: r["stack_shift"] - r["no_closure_shift"],
}


def hypothesis_table(runs, quick, rng):
    rows = []
    for h in MIND_CARD["hypotheses"]:
        diffs = [HYP_DIFFS[h["id"]](r) for r in runs]
        mean, ci = paired_bootstrap(diffs, rng)
        v = "not evaluated" if quick else verdict(mean, ci, h["direction"], h["mesi"])
        rows.append({"id": h["id"], "metric": h["metric"], "mean_diff": mean, "ci95": ci, "mesi": h["mesi"],
                     "n_seeds": len(runs), "verdict": v, "direction": h["direction"]})
    return rows


def knockout_table(runs, rng):
    rows = []
    sig = {m: v["signature"] for m, v in modules(runs[0]["stack"]).items()}
    for mod, mode in KO_ROWS:
        mean, ci = paired_bootstrap([r[f"ko_{mod}"] - r["stack_shift"] for r in runs], rng)
        rows.append({"module": mod, "mode": mode, "signature": sig[mod], "metric_change": mean, "ci95": ci})
    return rows


# ============================================================================ report
def accuracy_summary(runs):
    keys = ["stack_shift", "mlp_shift", "unicity_shift", "no_closure_shift", "stack_id", "mlp_id",
            "unicity_id", "stack_holistic", "mlp_holistic", "unicity_holistic"]
    return {k: (float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))) for k in keys}


def print_report(rep, acc):
    line = "-" * 78
    print(f"=== VERIFIED REPORT · chapter {CHAPTER:04d} ===")
    print(f"file {rep['file']}  card_revision {rep['card_revision']}  python {rep['environment']['python']}  "
          f"numpy {rep['environment']['numpy']}")
    print(f"seeds {rep['seeds']}  runtime {rep['runtime_s']:.1f} s  params: stack {rep['n_params']}, "
          f"mlp {rep['n_params_baseline']}, unicity rival {rep['n_params_rival']}")
    g = rep["gradcheck"]
    print(f"gradient check: {g['tensors_checked']}/{g['tensors_total']} tensor checks at {g['checked_at']}, "
          f"max rel err {g['max_rel_error']:.2e}, passed={g['passed']}")
    print(line + "\ncorrectness")
    for c in rep["correctness"]:
        print(f"  {c['id']:<5} {'PASS' if c['passed'] else 'FAIL'}  {c['name']}: {c['detail']}")
    m = rep["mutants"]
    print(f"mutation score {m['detected']}/{m['total']} = {m['score']:.2f}")
    print(line + "\nfull-kind accuracy, mean (sd) over seeds")
    for k, (mu, sd) in acc.items():
        print(f"  {k:<18} {mu:.3f} ({sd:.3f})")
    print(line + "\nhypotheses (paired per-seed differences, 95% bootstrap CI, 2000 resamples)")
    for h in rep["hypotheses"]:
        print(f"  {h['id']:<9} diff {h['mean_diff']:+.3f}  CI [{h['ci95'][0]:+.3f}, {h['ci95'][1]:+.3f}]  "
              f"mesi {h['mesi']:.2f} ({h['direction']})  -> {h['verdict']}")
    print(line + "\nknockouts on the trained stack (change in shifted full-kind accuracy)")
    for k in rep["knockouts"]:
        print(f"  {k['module']:<11} {k['mode']:<9} signature={str(k['signature']):<5} "
              f"{k['metric_change']:+.3f}  CI [{k['ci95'][0]:+.3f}, {k['ci95'][1]:+.3f}]")
    print(f"task types {rep['task_types']}  exit code {rep['exit_code']}")
    print("=== END REPORT ===")


def real_data_bridge(path, rng):
    if not path:
        print("no --data given: real-data bridge skipped")
        return None
    raw = np.genfromtxt(path, delimiter=",", skip_header=1)
    raw = raw[np.all(np.isfinite(raw), axis=1)]
    x, y = raw[:, :-1], raw[:, -1].astype(int)
    y = np.unique(y, return_inverse=True)[1]
    x = (x - x.mean(0)) / (x.std(0) + 1e-9)
    perm = rng.permutation(len(y))
    cut = int(0.8 * len(y))
    tr, te = perm[:cut], perm[cut:]
    n_cls = int(y.max()) + 1
    stack = build_model(x.shape[1], n_cls, "vector_classification", rng)
    mlp = build_mlp(x.shape[1], (n_cls,), matched_hidden(n_params(stack), x.shape[1], (n_cls,)), rng)
    out = {}
    for name, m in (("stack", stack), ("mlp", mlp)):
        fit(m, (x[tr], y[tr][:, None]), STEPS_QUICK, rng)
        out[name] = float((predict(m, x[te]) == y[te]).mean())
    print(f"real-data bridge on {os.path.basename(path)}: held-out accuracy {out}")
    return out


# ============================================================================ command-line entry point
def run_protocol(args):
    t0 = time.time()
    quick = args.quick
    steps = STEPS_QUICK if quick else STEPS_FULL
    n_seeds = 1 if quick else args.seeds
    seeds = [args.seed + i for i in range(n_seeds)]
    th = MIND_CARD["thresholds"]
    print(f"chapter {CHAPTER:04d} · {os.path.basename(__file__)} · seeds {seeds} · steps {steps}"
          f"{' · mutant ' + args.mutant if args.mutant else ''}", flush=True)
    base = np.random.SeedSequence(args.seed + 10_000)
    ss = base.spawn(6)
    ACTIVE_MUTANT["name"] = args.mutant
    probe_task = make_task(ss[0])
    correctness = [c1_gradcheck(probe_task, ss[1], th["gradcheck_rel_tol"])]
    runs = []
    for s in seeds:
        runs.append(run_seed(s, steps))
        print(f"  seed {s}: stack shift {runs[-1]['stack_shift']:.3f}  mlp shift {runs[-1]['mlp_shift']:.3f}  "
              f"({time.time() - t0:.0f} s)", flush=True)
    r0 = runs[0]
    correctness.append(c2_determinism(r0["comp_task"], ss[2]))
    correctness.append(c3_learning(r0["stack"], r0["stack_history"], r0["comp_task"], th))
    correctness.append(c4_shuffled(r0["comp_task"], ss[3], steps, th["shuffled_band"]))
    if args.mutant:
        mut = {"id": "C5", "name": "mutant_detection", "passed": True, "detail": "skipped under --mutant",
               "detected": 0, "total": 0}
    else:
        mut = c5_mutants(r0["comp_task"], ss[4], th["gradcheck_rel_tol"])
    correctness.append(mut)
    correctness += [c6_1_isometry(ss[5]), c6_2_order(ss[5]), c6_3_closure(r0["stack"], r0["comp_task"]),
                    c7_split(r0["comp_task"]), definition_check_matter(r0["stack"], r0["comp_task"])]
    rng = np.random.default_rng(args.seed + 20_000)
    hyps, kos = hypothesis_table(runs, quick, rng), knockout_table(runs, rng)
    runtime = time.time() - t0
    budget = BUDGET_QUICK if quick else BUDGET_FULL
    correctness.append({"id": "C8", "name": "budget", "passed": bool(runtime <= budget),
                        "detail": f"{runtime:.1f} s (budget {budget:.0f} s)"})
    passed = all(c["passed"] for c in correctness)
    exit_code = 0 if passed else (3 if not correctness[-1]["passed"] and
                                  all(c["passed"] for c in correctness[:-1]) else 1)
    c1 = correctness[0]
    rep = {"schema_version": "1.0", "chapter": CHAPTER, "file": os.path.basename(__file__),
           "card_revision": MIND_CARD["card_revision"],
           "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
           "seeds": seeds, "runtime_s": round(runtime, 2), "n_params": r0["n_params"]["stack"],
           "n_params_baseline": r0["n_params"]["mlp"], "n_params_rival": r0["n_params"]["unicity"],
           "gradcheck": {"tensors_checked": c1["tensors_checked"], "tensors_total": c1["tensors_total"],
                         "max_rel_error": c1["max_rel_error"], "checked_at": ["init", "after_training_steps"],
                         "passed": c1["passed"]},
           "correctness": [{k: c[k] for k in ("id", "name", "passed", "detail")} for c in correctness],
           "mutants": {"detected": mut["detected"], "total": mut["total"],
                       "score": (mut["detected"] / mut["total"]) if mut["total"] else 0.0},
           "hypotheses": hyps, "knockouts": kos, "task_types": TASK_TYPES, "exit_code": exit_code}
    print_report(rep, accuracy_summary(runs))
    real_data_bridge(args.data, np.random.default_rng(args.seed))
    if args.json:
        write_report(args.json, rep)
    return exit_code


def main(argv=None):
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__),
                                 description="Chapter 0270 resolutive form stack: full protocol by default")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--card", action="store_true")
    ap.add_argument("--mutant", type=str, default=None, choices=sorted(MUTANTS))
    ap.add_argument("--data", type=str, default=None)
    args = ap.parse_args(argv)
    if args.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    if args.seeds < 1:
        ap.print_usage()
        return 2
    try:
        return run_protocol(args)
    except NonFiniteError as err:
        print(f"non-finite values: {err}")
        return 4


if __name__ == "__main__":
    sys.exit(main())

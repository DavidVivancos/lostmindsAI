#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0332 · Ibn al-Nafis (1213-1288)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 17, Minds 321-340 Available on Amazon https://www.amazon.com/dp/B0HP7Z1NKC
# Resume and Interactive Demos at https://artificiology.com/
# END ATTRIBUTION
"""
Compact-septum transit with ambient straining: a NumPy research prototype.

Thesis
    Nothing may reach the chamber of decision unmixed with the air: the inner
    stream crosses only by the long route through an exchange organ, and the
    part of it that would cross the same whatever air it met is the coarse
    remainder, held back to feed the lung - so a shortcut is closed by
    subtracting the ambient expectation, not merely by deleting a visible path.

Evidence and provenance (belief; figures_master.json record 332)
    D1  Sharh Tashrih al-Qanun (ed. Qattaya 1988, 293-94; Meyerhof 1935 trans.
        of fol. 46r): no passage between the two ventricles, neither visible
        nor invisible; the substance of the heart there is thick.
    D2  Same locus: blood is first made thin in the right cavity.
    D3  Same locus (Abdel-Halim 2008; Fancy 2013, 2014): the arterial vein is
        compact and two-coated, the fine part filters through, the thicker
        part stays and nourishes the lung; the venous artery is single-coated.
    D4  Same locus: the filtered blood mixes with air in the lung and only so
        becomes fit to generate spirit in the left cavity.
    D5  Ibid., 17: he relies on verified examination and sound investigation,
        whether or not the predecessors agree.
    D6  Fancy 2013, ch. 4: the soul is connected to the entire body.
    D7  Galen, De usu partium XV (Kuehn IV): in the unborn the lung carries no
        air and nature joins the vessels so blood bypasses it. No surviving
        passage known to this chapter shows Ibn al-Nafis addressing that case.

Doctrine -> mechanism -> test
    D1  no query->decision path at all             C6.1 seal    H-RIVAL
    D3  straining: subtract the bank expectation   C6.1 seal    H-PORE, H-NEC
        of the mixture; remainder -> prior head
    D2  thinning: per-parcel RMS normalisation     C6.2 scale   knockout table
    D4  alveolar exchange: sigmoid-gated reads     C1           H-SIG
    D5  patency test: sever and measure leakage    leakage index (reported)
    D7  airless regime                             H-BLIND
    --  habit/air disagreement as conflict flag    H-META

Research question (robustness to spurious correlation; truthful reporting)
    Can an architecture make a model's parametric habit unable to reach its
    decision except through mixture with present evidence - closing both the
    visible residual shortcut and the invisible one that survives inside a
    nominally evidence-only path - and what does that cost without evidence?

Closest prior art and the delta
    Residual cross-attention readers (Vaswani et al. 2017) are the baseline.
    Context-aware decoding (Shi et al. 2024) contrasts outputs with and without
    context at decoding time; counterfactual VQA (Niu et al. 2021) and RUBi
    (Cadene et al. 2019) remove a question-only branch's direct effect; concept
    bottleneck leakage (Havasi et al. 2022) shows that nominal bottlenecks leak.
    Delta: the exact expectation of the evidence-mixture over a fixed, balanced
    bank of ambient contexts is subtracted inside the network during training,
    so the query-only component of the decision is zero by construction on the
    bank (checked, C6.1) with no query-only branch anywhere; the subtracted
    remainder is quarantined in a readout that can only report habit.

Blind spot
    When evidence is absent (the airless, foetal regime) the decision has
    nothing to carry it and the strained remainder even pushes against habit.

Task (synthetic "transit" world, regenerated per seed)
    K=40 entities with unit keys and a habitual answer in M=6 classes.
    Query = rotated noisy key. Context = 4 slots, each a rotated noisy
    (entity key, answer embedding) pair; no single feature reveals the answer.
    Train: 85% of contexts contain the query entity, whose slot agrees with its
    habit 92% of the time; otherwise the target is the habit.
    Splits: held-out (same law), shifted "counterfactual air" (slot always
    contradicts habit), blind-spot "airless" (entity absent, target = habit).

Limits
    Synthetic data, small models, five seeds. A research prototype of one
    mechanism for an AGI-oriented architecture, not an AGI, and no claim
    about the historical person beyond the cited evidence.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 2,
    "revision_log": ["rev 2, before any official-seed run: added gradcheck_floor 1e-4 after the first development "
                     "gradient check showed central-difference roundoff near 5e-10 on entries with |g| < 1e-6; "
                     "learning rates per model chosen from {1e-3, 3e-3, 1e-2} by validation accuracy on development "
                     "seeds 9001-9002; similarity to the one available chapter file filled in. Hypotheses, metrics, "
                     "splits and mesi unchanged."],
    "generation": {"template_version": "codeguidelines 1.0 (2026-09-15)",
                   "generator": "Claude (Anthropic)", "generator_version": "claude-opus-5-5",
                   "date": "2026-10-22"},
    "id": 332, "figure": "Ibn al-Nafis", "born": 1213, "died": 1288,
    "civilization": "Arab (Ayyubid/Mamluk)", "provenance": "belief",
    "thesis": ("Nothing reaches the chamber of decision unmixed with the air: the inner stream crosses "
               "only through the exchange organ, and whatever part of it would cross the same whatever "
               "air it met is the coarse remainder, held back to feed the lung, so the pore is closed by "
               "subtracting the ambient expectation rather than by deleting a visible passage."),
    "evidence": [
        {"id": "D1", "claim": "No passage between the ventricles, neither visible nor invisible; the septal substance is thick",
         "basis": "primary", "source": "Sharh Tashrih al-Qanun, ed. Qattaya 1988, 293-94; Meyerhof 1935, fol. 46r"},
        {"id": "D2", "claim": "Blood is made thin in the right cavity before transit", "basis": "primary",
         "source": "Sharh Tashrih al-Qanun, ed. Qattaya 1988, 293-94"},
        {"id": "D3", "claim": "Compact two-coated arterial vein filters the fine part; the thicker part stays and nourishes the lung",
         "basis": "primary", "source": "Sharh Tashrih al-Qanun 293-94 via Abdel-Halim 2008; Fancy 2013"},
        {"id": "D4", "claim": "Filtered blood mixes with air in the lung and only then is fit for spirit in the left cavity",
         "basis": "primary", "source": "Sharh Tashrih al-Qanun 293-94; Meyerhof 1935"},
        {"id": "D5", "claim": "Relies on verified examination and sound investigation regardless of predecessors",
         "basis": "primary", "source": "Sharh Tashrih al-Qanun, ed. Qattaya 1988, 17"},
        {"id": "D6", "claim": "The soul is connected to the entire body, not seated in one organ", "basis": "scholarship",
         "source": "Fancy, Science and Religion in Mamluk Egypt (2013), ch. 4"},
        {"id": "D7", "claim": "In the unborn the airless lung is bypassed (Galen); no known passage of his addresses it",
         "basis": "speculation", "source": "Galen, De usu partium XV (Kuehn IV 243); Lind 1980"},
    ],
    "research_question": {"category": "robustness to spurious correlation",
                          "question": ("Can a model's parametric habit be prevented from reaching its decision "
                                       "except through mixture with present evidence, closing both the visible "
                                       "residual shortcut and the invisible one inside a nominally evidence-only "
                                       "path, and what does that cost when evidence is absent?")},
    "mechanism": {
        "name": "Compact-septum transit with ambient straining",
        "family": "evidence-grounded fusion with exact marginal subtraction (interventional centring)",
        "signature_modules": ["straining"],
        "closest_prior_art": ["residual cross-attention reader (Vaswani et al. 2017)",
                              "context-aware decoding (Shi et al. 2024)",
                              "counterfactual VQA direct-effect removal (Niu et al. 2021); RUBi (Cadene et al. 2019)",
                              "leakage in concept bottleneck models (Havasi et al. 2022)"],
        "overlap": "Medium",
        "prior_art_queries": ["knowledge conflict context faithfulness parametric memory",
                              "language prior VQA-CP question-only branch subtraction",
                              "context-aware decoding contrastive with and without context",
                              "concept bottleneck leakage soft concepts",
                              "marginalize over context backdoor adjustment neural fusion"],
        "contribution_type": "mechanism",
        "delta": ("Exact subtraction, inside the network and during training, of the mixture's expectation over a "
                  "fixed balanced bank of ambient contexts, with no query-only branch; the remainder feeds a "
                  "quarantined habit readout used only to flag conflicts."),
    },
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1 no query->decision path", "property_test": "C6.1", "hypothesis": "H-RIVAL"},
        {"doctrine": "D3", "mechanism": "M2 straining against the ambient bank", "property_test": "C6.1", "hypothesis": "H-PORE"},
        {"doctrine": "D3", "mechanism": "M2 straining (knockout)", "property_test": "C6.1", "hypothesis": "H-NEC"},
        {"doctrine": "D2", "mechanism": "M3 thinning (per-parcel RMS)", "property_test": "C6.2", "hypothesis": "H-NEC"},
        {"doctrine": "D4", "mechanism": "M4 alveolar exchange", "property_test": "C1", "hypothesis": "H-SIG"},
        {"doctrine": "D7", "mechanism": "M2 under absent evidence", "property_test": "C6.1", "hypothesis": "H-BLIND"},
        {"doctrine": "D3", "mechanism": "M5 quarantined habit readout", "property_test": "C1", "hypothesis": "H-META"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "Strained transit follows contradicting evidence better than a residual cross-attention reader",
         "metric": "accuracy", "split": "shifted (counterfactual air)", "comparison": "model - baseline",
         "direction": "greater", "mesi": 0.05, "seeds": 5},
        {"id": "H-NEC", "statement": "Replacing the strained remainder by its mean hurts more than replacing parcel RMS by its mean",
         "metric": "accuracy", "split": "shifted (counterfactual air)",
         "comparison": "knockout(straining, mean) - knockout(thinning, mean)", "direction": "less", "mesi": 0.03, "seeds": 5},
        {"id": "H-BLIND", "statement": "Without evidence the sealed model fails where the residual reader answers from habit",
         "condition": "airless: query entity absent from context, target = habit",
         "grounding": "the prohibition of any right-left passage has no regime for an airless lung (D7)",
         "metric": "accuracy", "split": "airless", "comparison": "model - baseline", "direction": "less",
         "mesi": 0.15, "seeds": 5},
        {"id": "H-RIVAL", "statement": "Adding Galen's septal pore (learned query->decision path) lowers faithfulness",
         "metric": "accuracy", "split": "shifted (counterfactual air)", "comparison": "model - galen",
         "direction": "greater", "mesi": 0.05, "seeds": 5},
        {"id": "H-PORE", "statement": "A nominally sealed but unstrained lung keeps an invisible pore that lowers faithfulness",
         "metric": "accuracy", "split": "shifted (counterfactual air)", "comparison": "model - unstrained",
         "direction": "greater", "mesi": 0.03, "seeds": 5},
        {"id": "H-META", "statement": "Habit-versus-spirit disagreement flags evidence conflicts better than the baseline's with/without-context divergence",
         "metric": "AUROC", "split": "counterfactual vs agreeing held-out", "comparison": "model - baseline",
         "direction": "greater", "mesi": 0.03, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.3, "margin_over_trivial": 0.25, "shuffle_band": 0.08,
                   "gradcheck_rel_tol": 1e-5, "gradcheck_floor": 1e-4, "seal_tol": 1e-9, "scale_tol": 1e-6},
    "probe_predictions": [
        {"probe": "P1", "expected": "below baseline (no recurrence, single transit)"},
        {"probe": "P2", "expected": "equal to baseline"},
        {"probe": "P6", "expected": "below baseline (no query-only shortcut to memorise)"},
        {"probe": "P8", "expected": "equal to baseline"},
        {"probe": "P9", "expected": "above baseline under query noise, below under context noise"},
    ],
    "dialectic_links": [
        {"chapter": 135, "relation": "rival", "test": "H-RIVAL"},
        {"chapter": 257, "relation": "teacher", "test": "none (commented text: three ventricles rejected)"},
        {"chapter": 565, "relation": "successor", "test": "none (recirculation absent here; noted as limit)"},
        {"chapter": 491, "relation": "successor", "test": "none (independent transit, different mechanism)"},
    ],
    "corpus_neighbors": [
        {"chapter": 504, "similarity": 0.038, "difference": "Philip II sample file, the only other chapter code in session: multiplicative survival of margin notes vs exact marginal subtraction over contexts"},
        {"chapter": 135, "similarity": None, "difference": "Galen: severing a conduit to map function; here the conduit is subtracted, not severed"},
        {"chapter": 491, "similarity": None, "difference": "Servetus: peer-gated refutation council; here a single exchange organ with exact centring"},
        {"chapter": 565, "similarity": None, "difference": "Harvey: budget licenses a return loop; here no loop, a one-way strained transit"},
        {"chapter": 243, "similarity": None, "difference": "al-Zahrawi: compact-support correction reach; here marginal subtraction over contexts"},
        {"chapter": 287, "similarity": None, "difference": "Ibn Zuhr: titrated particular via prior and probe; here priors are quarantined, never mixed"},
    ],
    "barometer": {
        "cognitive_processing": ["key-matched reading under nuisance noise"],
        "embodied_cognition": [],
        "world_modeling": ["shifted split where the world contradicts habit"],
        "consciousness": ["habit-versus-evidence conflict flag (H-META)"],
        "language_understanding": [],
        "emotional_intelligence": [],
        "creativity": [],
        "autonomy": [],
    },
    "task_types": ["vector_classification"],
    "applications": [
        {"use": "retrieval-augmented QA that must follow supplied documents over memorised facts",
         "sector": "enterprise search", "dataset": "entity-substituted Natural Questions (Longpre et al. 2021)"},
        {"use": "visual question answering under answer-prior shift", "sector": "assistive vision",
         "dataset": "VQA-CP v2 (Agrawal et al. 2018)"},
        {"use": "claim verification against provided evidence", "sector": "fact-checking",
         "dataset": "FEVER (Thorne et al. 2018)"},
    ],
    "safety_notes": ("Synthetic data only. Medical imagery is metaphor: no diagnosis, dosing or treatment. "
                     "The sealed design is unsafe where evidence can be absent or poisoned; the habit readout "
                     "must be shown to people, not silently dropped."),
}

import argparse
import hashlib
import json
import math
import os
import sys
import time

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")          # the runtime budget is defined for one CPU core
import numpy as np  # noqa: E402

K_ENT, N_CLS, N_SLOT, D_KEY, D_VAL = 40, 6, 4, 10, 6
Q_DIM, SLOT_DIM = D_KEY, D_KEY + D_VAL
NOISE, P_AIR, P_AGREE = 0.12, 0.85, 0.92
N_BANK = 40                       # ambient bank: every entity appears exactly 4 times
HID, ALV, PARCEL, DV = 24, 6, 6, 6
BASE_D, BASE_H2 = 24, 42          # baseline sizes chosen to match parameter count
LAMBDA_PRIOR = 0.5
EPS_RMS = 1e-10
BATCH, CLIP = 64, 5.0
# learning rates chosen per model from {1e-3, 3e-3, 1e-2} by mean validation accuracy on dev seeds 9001-9002
LR = {"nafis": 1e-2, "galen": 1e-2, "unstrained": 1e-2, "baseline": 3e-3}
MUTANT = None


# BEGIN STANDARD UTILITIES v1.0
def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def logsumexp(z, axis=-1):
    m = z.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(z - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def softplus(x):
    return np.logaddexp(0.0, x)


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
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
            params[k] -= sign * self.lr * mh / (np.sqrt(vh) + self.eps)


def clip_global_norm(grads, max_norm):
    total = math.sqrt(sum(float((g ** 2).sum()) for g in grads.values()))
    if total > max_norm:
        for k in grads:
            grads[k] *= max_norm / (total + 1e-12)
    return total


def finite_difference_check(loss_fn, params, grads, rng, n_entries=20, eps=1e-6, floor=1e-4):
    """Central differences on n random entries per tensor plus its largest-gradient entry."""
    worst, report = 0.0, {}
    for name, P in params.items():
        flat = P.reshape(-1)
        g = grads[name].reshape(-1)
        idx = set(rng.choice(flat.size, size=min(n_entries, flat.size), replace=False).tolist())
        idx.add(int(np.argmax(np.abs(g))))
        err = 0.0
        for i in idx:
            old = flat[i]
            flat[i] = old + eps
            lp = loss_fn()
            flat[i] = old - eps
            lm = loss_fn()
            flat[i] = old
            num = (lp - lm) / (2 * eps)
            err = max(err, abs(num - g[i]) / max(abs(num) + abs(g[i]), floor))
        report[name] = err
        worst = max(worst, err)
    return worst, report


def paired_bootstrap(diffs, rng, n_boot=2000):
    d = np.asarray(diffs, dtype=float)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    return float(d.mean()), [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def verdict(mean, ci, mesi, direction):
    sgn = 1.0 if direction == "greater" else -1.0
    lo, hi = sgn * ci[0], sgn * ci[1]
    if min(lo, hi) > 0 and sgn * mean >= mesi:
        return "supported"
    if max(lo, hi) < 0:
        return "contradicted"
    return "inconclusive"


def write_report(rep, path=None):
    print("=== VERIFIED REPORT · chapter %04d ===" % rep["chapter"])
    for line in rep.pop("_lines"):
        print(line)
    print("=== END REPORT ===")
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=2)
# END STANDARD UTILITIES


# ----------------------------------------------------------------- data ----
def _orth(n, rng):
    q, r = np.linalg.qr(rng.normal(size=(n, n)))
    return q * np.sign(np.diag(r))


def make_world(rng):
    keys = rng.normal(size=(K_ENT, D_KEY))
    keys /= np.linalg.norm(keys, axis=1, keepdims=True)
    return {"keys": keys, "vals": _orth(D_VAL, rng), "habit": rng.integers(0, N_CLS, size=K_ENT),
            "Qq": _orth(Q_DIM, rng), "Qs": _orth(SLOT_DIM, rng)}


def _other(cls, rng):
    return (cls + rng.integers(1, N_CLS, size=np.shape(cls))) % N_CLS


def _slots(world, ents, vals, rng):
    k = world["keys"][ents] + NOISE * rng.normal(size=ents.shape + (D_KEY,))
    v = world["vals"][vals] + NOISE * rng.normal(size=ents.shape + (D_VAL,))
    return np.concatenate([k, v], axis=-1) @ world["Qs"].T


def _distractors(n, exclude, rng):
    score = rng.random((n, K_ENT))
    if exclude is not None:
        score[np.arange(n), exclude] = 2.0
    return np.argsort(score, axis=1)[:, :N_SLOT]


def sample_split(world, n, rng, mode):
    """mode: 'train' (native law), 'cf' (slot contradicts habit), 'airless' (entity absent)."""
    e = rng.integers(0, K_ENT, size=n)
    p_air = {"train": P_AIR, "cf": 1.0, "airless": 0.0}[mode]
    aired = rng.random(n) < p_air
    ents = _distractors(n, e, rng)
    vals = np.where(rng.random(ents.shape) < P_AGREE, world["habit"][ents], _other(world["habit"][ents], rng))
    pos = rng.integers(0, N_SLOT, size=n)
    own = world["habit"][e] if mode == "train" else _other(world["habit"][e], rng)
    if mode == "train":
        own = np.where(rng.random(n) < P_AGREE, own, _other(own, rng))
    rows = np.where(aired)[0]
    ents[rows, pos[rows]] = e[rows]
    vals[rows, pos[rows]] = own[rows]
    y = np.where(aired, own, world["habit"][e])
    q = (world["keys"][e] + NOISE * rng.normal(size=(n, D_KEY))) @ world["Qq"].T
    X = np.concatenate([q, _slots(world, ents, vals, rng).reshape(n, -1)], axis=1)
    agree = aired & (own == world["habit"][e])
    return {"X": X, "y": y, "aired": aired, "agree": agree, "ent": e}


def sample_contexts(world, n, rng, balanced=False):
    if balanced:
        ents = np.concatenate([rng.permutation(K_ENT) for _ in range(n * N_SLOT // K_ENT)]).reshape(n, N_SLOT)
    else:
        ents = _distractors(n, None, rng)
    vals = np.where(rng.random(ents.shape) < P_AGREE, world["habit"][ents], _other(world["habit"][ents], rng))
    return _slots(world, ents, vals, rng)


def make_data(seed, quick=False):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 332]))
    world = make_world(rng)
    sizes = (1500, 400, 600, 600, 400) if quick else (4000, 800, 1000, 1000, 600)
    tr, va, te = (sample_split(world, s, rng, "train") for s in sizes[:3])
    return {"world": world, "train": tr, "val": va, "test_id": te,
            "test_cf": sample_split(world, sizes[3], rng, "cf"),
            "test_airless": sample_split(world, sizes[4], rng, "airless"),
            "bank": sample_contexts(world, N_BANK, rng, balanced=True),
            "fresh": sample_contexts(world, 60, rng)}


# ---------------------------------------------------------------- model ----
TASK_TYPES = ["vector_classification"]
KINDS = ("nafis", "galen", "unstrained", "baseline")


def _init(rng, shape, fan_in, scale=1.0):
    return rng.normal(size=shape) * scale / math.sqrt(fan_in)


def build_model(in_dim, out_dim, task_type, rng, kind="nafis", q_dim=None, n_slots=None, bank=None):
    if task_type not in TASK_TYPES:
        raise ValueError("unsupported task type %s" % task_type)
    q_dim = q_dim or in_dim // 2
    n_slots = n_slots or 1
    sd = (in_dim - q_dim) // n_slots
    cfg = {"kind": kind, "q_dim": q_dim, "n_slots": n_slots, "slot_dim": sd, "out": out_dim}
    if kind == "baseline":
        p = {"Wq1": _init(rng, (q_dim, HID), q_dim), "bq1": np.zeros(HID),
             "Wq2": _init(rng, (HID, BASE_D), HID), "Wc1": _init(rng, (sd, HID), sd), "bc1": np.zeros(HID),
             "Wk2": _init(rng, (HID, BASE_D), HID), "Wv2": _init(rng, (HID, BASE_D), HID),
             "Wh1": _init(rng, (BASE_D, BASE_H2), BASE_D), "bh1": np.zeros(BASE_H2),
             "Wo": _init(rng, (BASE_H2, out_dim), BASE_H2), "bo": np.zeros(out_dim)}
    else:
        p = {"Wb1": _init(rng, (q_dim, HID), q_dim), "bb1": np.zeros(HID),
             "Wb2": _init(rng, (HID, ALV * PARCEL), HID), "Wa1": _init(rng, (sd, HID), sd),
             "ba1": np.zeros(HID), "Wk": _init(rng, (HID, ALV * PARCEL), HID),
             "Wv": _init(rng, (HID, ALV * DV), HID), "beta": -np.ones(ALV),
             "Ws": _init(rng, (ALV * DV, out_dim), ALV * DV), "bs": np.zeros(out_dim),
             "U": _init(rng, (ALV * DV, out_dim), ALV * DV), "u": np.zeros(out_dim)}
        if kind == "galen":
            p["Pp"] = _init(rng, (ALV * PARCEL, out_dim), ALV * PARCEL, 0.5)
    return {"cfg": cfg, "params": p, "bank": bank, "ko": {}, "ref": None}


def _split(model, X):
    c = model["cfg"]
    return X[:, :c["q_dim"]], X[:, c["q_dim"]:].reshape(len(X), c["n_slots"], c["slot_dim"])


def _air(P, X3):
    g = np.tanh(X3 @ P["Wa1"] + P["ba1"])
    n, s = X3.shape[:2]
    return g, (g @ P["Wk"]).reshape(n, s, ALV, PARCEL), (g @ P["Wv"]).reshape(n, s, ALV, DV)


def nafis_forward(model, X):
    """Right cavity -> thinning -> alveolar exchange with air -> straining -> left cavity."""
    P, ko, kind = model["params"], model["ko"], model["cfg"]["kind"]
    q, Xs = _split(model, X)
    B, tau = len(X), 1.0 / math.sqrt(PARCEL)
    h = np.tanh(q @ P["Wb1"] + P["bb1"])
    pp = (h @ P["Wb2"]).reshape(B, ALV, PARCEL)
    rms = np.sqrt((pp ** 2).mean(-1, keepdims=True) + EPS_RMS)
    if ko.get("thinning") == "identity":
        rms = np.ones_like(rms)
    elif ko.get("thinning") == "mean":
        rms = np.broadcast_to(model["ref"]["rms"], rms.shape)
    phi = pp / rms
    ga, kap, nu = _air(P, Xs)
    gr, kapr, nur = _air(P, model["bank"])
    R, S = kapr.shape[:2]
    kr = kapr.transpose(2, 0, 1, 3).reshape(ALV, R * S, PARCEL)
    vr = nur.transpose(2, 0, 1, 3).reshape(ALV, R * S, DV)
    pa = phi.transpose(1, 0, 2)
    gam = sigmoid(tau * (phi[:, None] * kap).sum(-1) + P["beta"])
    gamr = sigmoid(tau * pa @ kr.transpose(0, 2, 1) + P["beta"][:, None, None])
    m = (gam[..., None] * nu).sum(1)
    mbar = (gamr @ vr).transpose(1, 0, 2) / R
    if ko.get("alveolar_exchange") == "zero":
        m, mbar = np.zeros_like(m), np.zeros_like(mbar)
    mode = ko.get("straining", "on" if kind != "unstrained" else "identity")
    f = {"on": lambda: m - mbar, "identity": lambda: m,
         "mean": lambda: m - model["ref"]["mbar"], "zero": lambda: np.zeros_like(m)}[mode]()
    logits = f.reshape(B, -1) @ P["Ws"] + P["bs"]
    if kind == "galen":
        logits = logits + phi.reshape(B, -1) @ P["Pp"]
    prior = mbar.reshape(B, -1) @ P["U"] + P["u"]
    if ko.get("lung_nourishment") == "zero":
        prior = np.zeros_like(prior)
    cache = dict(q=q, Xs=Xs, h=h, rms=rms, phi=phi, ga=ga, kap=kap, nu=nu, gr=gr, kr=kr, vr=vr, pa=pa,
                 gam=gam, gamr=gamr, m=m, mbar=mbar, f=f, tau=tau, R=R, S=S)
    return logits, prior, cache


def _air_back(P, G, X3, g, dkap, dnu):
    n = X3.shape[0] * X3.shape[1]
    gg, dk, dv = g.reshape(n, -1), dkap.reshape(n, -1), dnu.reshape(n, -1)
    G["Wk"] += gg.T @ dk
    G["Wv"] += gg.T @ dv
    dpre = (dk @ P["Wk"].T + dv @ P["Wv"].T) * (1 - gg ** 2)
    G["Wa1"] += X3.reshape(n, -1).T @ dpre
    G["ba1"] += dpre.sum(0)


def nafis_backward(model, c, dlog, dpri):
    P, kind = model["params"], model["cfg"]["kind"]
    G = {k: np.zeros_like(v) for k, v in P.items()}
    B, tau = dlog.shape[0], c["tau"]
    G["Ws"], G["bs"] = c["f"].reshape(B, -1).T @ dlog, dlog.sum(0)
    G["U"], G["u"] = c["mbar"].reshape(B, -1).T @ dpri, dpri.sum(0)
    dm = (dlog @ P["Ws"].T).reshape(B, ALV, DV)
    dmbar = (dpri @ P["U"].T).reshape(B, ALV, DV)
    if kind != "unstrained":
        dmbar = dmbar - dm                       # the coarse remainder is subtracted
    dphi = np.zeros_like(c["phi"])
    if kind == "galen":
        G["Pp"] = c["phi"].reshape(B, -1).T @ dlog
        dphi += (dlog @ P["Pp"].T).reshape(B, ALV, PARCEL)
    gam, gamr, R, S = c["gam"], c["gamr"], c["R"], c["S"]
    dz = (dm[:, None] * c["nu"]).sum(-1) * gam * (1 - gam)
    dnu = gam[..., None] * dm[:, None]
    dphi += tau * (dz[..., None] * c["kap"]).sum(1)
    dkap = tau * dz[..., None] * c["phi"][:, None]
    G["beta"] = dz.sum((0, 1))
    dma = dmbar.transpose(1, 0, 2) / R
    dzr = (dma @ c["vr"].transpose(0, 2, 1)) * gamr * (1 - gamr)
    dnur = (gamr.transpose(0, 2, 1) @ dma).reshape(ALV, R, S, DV).transpose(1, 2, 0, 3)
    dkapr = (tau * dzr.transpose(0, 2, 1) @ c["pa"]).reshape(ALV, R, S, PARCEL).transpose(1, 2, 0, 3)
    if MUTANT != "drop_bank_branch":
        dphi += tau * (dzr @ c["kr"]).transpose(1, 0, 2)
    G["beta"] += dzr.sum((1, 2))
    _air_back(P, G, c["Xs"], c["ga"], dkap, dnu)
    _air_back(P, G, model["bank"], c["gr"], dkapr, dnur)
    phi = c["phi"]
    dpp = ((dphi - phi * (dphi * phi).mean(-1, keepdims=True)) / c["rms"]).reshape(B, -1)
    G["Wb2"] = c["h"].T @ dpp
    da1 = (dpp @ P["Wb2"].T) * (1 - c["h"] ** 2)
    G["Wb1"], G["bb1"] = c["q"].T @ da1, da1.sum(0)
    return G


def base_forward(model, X, empty=False):
    """Residual cross-attention reader: query + softmax read over slots, then an MLP head."""
    P = model["params"]
    q, Xs = _split(model, X)
    hq = np.tanh(q @ P["Wq1"] + P["bq1"])
    zq = hq @ P["Wq2"]
    g = np.tanh(Xs @ P["Wc1"] + P["bc1"])
    K, V = g @ P["Wk2"], g @ P["Wv2"]
    al = softmax(np.einsum("bd,bsd->bs", zq, K) / math.sqrt(BASE_D), axis=1)
    r = np.zeros_like(zq) if empty else np.einsum("bs,bsd->bd", al, V)
    zz = zq + r
    h2 = np.tanh(zz @ P["Wh1"] + P["bh1"])
    return h2 @ P["Wo"] + P["bo"], dict(q=q, Xs=Xs, hq=hq, zq=zq, g=g, K=K, V=V, al=al, zz=zz, h2=h2)


def base_backward(model, c, dlog):
    P = model["params"]
    G = {"Wo": c["h2"].T @ dlog, "bo": dlog.sum(0)}
    dpre2 = (dlog @ P["Wo"].T) * (1 - c["h2"] ** 2)
    G["Wh1"], G["bh1"] = c["zz"].T @ dpre2, dpre2.sum(0)
    dzz = dpre2 @ P["Wh1"].T
    al = c["al"]
    dal = np.einsum("bd,bsd->bs", dzz, c["V"])
    dsc = al * (dal - (dal * al).sum(1, keepdims=True)) / math.sqrt(BASE_D)
    dzq = dzz + np.einsum("bs,bsd->bd", dsc, c["K"])
    dK, dV = np.einsum("bs,bd->bsd", dsc, c["zq"]), np.einsum("bs,bd->bsd", al, dzz)
    n = dK.shape[0] * dK.shape[1]
    gg = c["g"].reshape(n, -1)
    G["Wk2"], G["Wv2"] = gg.T @ dK.reshape(n, -1), gg.T @ dV.reshape(n, -1)
    dpre = (dK.reshape(n, -1) @ P["Wk2"].T + dV.reshape(n, -1) @ P["Wv2"].T) * (1 - gg ** 2)
    G["Wc1"], G["bc1"] = c["Xs"].reshape(n, -1).T @ dpre, dpre.sum(0)
    G["Wq2"] = c["hq"].T @ dzq
    dpq = (dzq @ P["Wq2"].T) * (1 - c["hq"] ** 2)
    G["Wq1"], G["bq1"] = c["q"].T @ dpq, dpq.sum(0)
    return G


def _ce(logits, y):
    lp = logits - logsumexp(logits)[:, None]
    p = np.exp(lp)
    p[np.arange(len(y)), y] -= 1.0
    return -lp[np.arange(len(y)), y].mean(), p / len(y)


def loss_and_grads(model, batch):
    X, y = batch
    if model["cfg"]["kind"] == "baseline":
        logits, cache = base_forward(model, X)
        loss, dlog = _ce(logits, y)
        G = base_backward(model, cache, dlog)
    else:
        logits, prior, cache = nafis_forward(model, X)
        l1, dlog = _ce(logits, y)
        l2, dpri = _ce(prior, y)
        loss = l1 + LAMBDA_PRIOR * l2
        G = nafis_backward(model, cache, dlog, LAMBDA_PRIOR * dpri)
    if MUTANT == "zero_grad_Wk" and "Wk" in G:
        G["Wk"][:] = 0.0
    return loss, G


def logits_of(model, X):
    if model["cfg"]["kind"] == "baseline":
        return base_forward(model, X)[0]
    return nafis_forward(model, X)[0]


def predict(model, X):
    return np.argmax(logits_of(model, X), axis=1)


def hidden_states(model, X):
    if model["cfg"]["kind"] == "baseline":
        c = base_forward(model, X)[1]
        return {"query": c["zq"], "read": c["zz"] - c["zq"], "head": c["h2"]}
    c = nafis_forward(model, X)[2]
    return {"blood": c["phi"], "mixture": c["m"], "coarse": c["mbar"], "fine": c["f"]}


def modules(model):
    if model["cfg"]["kind"] == "baseline":
        return {"query_encoder": {"params": ["Wq1", "bq1", "Wq2"], "role": "query MLP", "signature": False},
                "slot_encoder": {"params": ["Wc1", "bc1", "Wk2", "Wv2"], "role": "slot keys/values", "signature": False},
                "head": {"params": ["Wh1", "bh1", "Wo", "bo"], "role": "residual MLP head", "signature": False}}
    mods = {
        "right_ventricle": {"params": ["Wb1", "bb1", "Wb2"], "role": "query MLP encoder", "signature": False},
        "thinning": {"params": [], "role": "per-parcel RMS normalisation of query parcels", "signature": False},
        "bronchi": {"params": ["Wa1", "ba1", "Wk", "Wv"], "role": "slot MLP producing keys and values", "signature": False},
        "alveolar_exchange": {"params": ["beta"], "role": "per-parcel sigmoid-gated key-matched read", "signature": False},
        "straining": {"params": [], "role": "subtract expectation of the mixture over a fixed context bank", "signature": True},
        "left_ventricle": {"params": ["Ws", "bs"], "role": "linear readout of the centred mixture", "signature": False},
        "lung_nourishment": {"params": ["U", "u"], "role": "query-only prior readout of the subtracted part", "signature": False},
    }
    if model["cfg"]["kind"] == "galen":
        mods["septal_pore"] = {"params": ["Pp"], "role": "direct linear query->logit path", "signature": False}
    return mods


def knockout(model, name, mode):
    if name not in modules(model) or mode not in ("identity", "zero", "mean"):
        raise ValueError("bad knockout %s/%s" % (name, mode))
    ko = dict(model["ko"])
    ko[name] = mode
    return {"cfg": model["cfg"], "params": model["params"], "bank": model["bank"], "ko": ko, "ref": model["ref"]}


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


def _set_reference(model, X):
    if model["cfg"]["kind"] != "baseline":
        c = nafis_forward(model, X)[2]
        model["ref"] = {"rms": c["rms"].mean(0), "mbar": c["mbar"].mean(0)}


def fit(model, data, budget, rng, lr=None, eval_every=100):
    """Adam on minibatches; the best validation checkpoint is kept (selection on validation only)."""
    if model["bank"] is None and model["cfg"]["kind"] != "baseline":
        rows = rng.choice(len(data["Xtr"]), size=N_BANK, replace=False)
        model["bank"] = _split(model, data["Xtr"][rows])[1]
    P = model["params"]
    lr = LR[model["cfg"]["kind"]] if lr is None else lr
    opt = Adam(P, lr=0.0 if MUTANT == "zero_lr" else lr)
    sign = -1.0 if MUTANT == "sign_flip" else 1.0
    best, best_acc, losses = None, -1.0, []
    n = len(data["Xtr"])
    for step in range(1, budget + 1):
        idx = rng.integers(0, n, size=BATCH)
        loss, G = loss_and_grads(model, (data["Xtr"][idx], data["ytr"][idx]))
        if not np.isfinite(loss):
            raise FloatingPointError("non-finite loss")
        clip_global_norm(G, CLIP)
        opt.step(P, G, sign)
        losses.append(float(loss))
        if step % eval_every == 0 or step == budget:
            acc = float((predict(model, data["Xva"]) == data["yva"]).mean())
            if acc > best_acc:
                best_acc, best = acc, {k: v.copy() for k, v in P.items()}
    model["params"] = best
    _set_reference(model, data["Xva"])
    return np.array(losses)


# ---------------------------------------------------------- evaluation ----
def accuracy(model, split):
    return float((predict(model, split["X"]) == split["y"]).mean())


def leakage_index(model, data):
    """Share of decision variance carried by the query alone, marginalising fresh (unseen) contexts."""
    Q = data["test_id"]["X"][:120, :Q_DIM]
    C = data["fresh"].reshape(len(data["fresh"]), -1)
    X = np.concatenate([np.repeat(Q, len(C), 0), np.tile(C, (len(Q), 1))], axis=1)
    L = logits_of(model, X).reshape(len(Q), len(C), -1)
    L = L - L.mean(-1, keepdims=True)
    return float(L.mean(1).var(0).sum() / L.reshape(-1, L.shape[-1]).var(0).sum())


def _auroc(pos, neg):
    s = np.concatenate([pos, neg])
    ranks = np.empty(len(s))
    ranks[np.argsort(s, kind="mergesort")] = np.arange(1, len(s) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def conflict_scores(model, X):
    if model["cfg"]["kind"] == "baseline":
        a, b = softmax(base_forward(model, X)[0]), softmax(base_forward(model, X, empty=True)[0])
    else:
        lg, pr, _ = nafis_forward(model, X)
        a, b = softmax(lg), softmax(pr)
    return 1.0 - (a * b).sum(1)


def conflict_auroc(model, data):
    agree = data["test_id"]["X"][data["test_id"]["agree"]]
    return _auroc(conflict_scores(model, data["test_cf"]["X"]), conflict_scores(model, agree))


def cad_accuracy(model, split, alpha=1.0):
    lg = (1 + alpha) * base_forward(model, split["X"])[0] - alpha * base_forward(model, split["X"], empty=True)[0]
    return float((lg.argmax(1) == split["y"]).mean())


def train_kind(kind, data, seed, budget):
    rng = np.random.default_rng(np.random.SeedSequence([seed, KINDS.index(kind), 7]))
    m = build_model(Q_DIM + N_SLOT * SLOT_DIM, N_CLS, "vector_classification", rng, kind=kind,
                    q_dim=Q_DIM, n_slots=N_SLOT, bank=None if kind == "baseline" else data["bank"])
    d = {"Xtr": data["train"]["X"], "ytr": data["train"]["y"], "Xva": data["val"]["X"], "yva": data["val"]["y"]}
    losses = fit(m, d, budget, rng)
    return m, losses


# ----------------------------------------------------------------- tests ----
def _probe_batch(seed, n=12):
    data = make_data(seed, quick=True)
    return data, (data["train"]["X"][:n], data["train"]["y"][:n])


def gradcheck(kind, seed, steps=60):
    """C1 at initialisation and after `steps` training updates; every tensor checked."""
    data, batch = _probe_batch(seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 99]))
    out = []
    for stage in ("init", "after_training_steps"):
        if stage == "init":
            m = build_model(Q_DIM + N_SLOT * SLOT_DIM, N_CLS, "vector_classification", rng, kind=kind,
                            q_dim=Q_DIM, n_slots=N_SLOT, bank=data["bank"])
        else:
            d = {"Xtr": data["train"]["X"], "ytr": data["train"]["y"], "Xva": data["val"]["X"], "yva": data["val"]["y"]}
            fit(m, d, steps, rng)
        _, G = loss_and_grads(m, batch)
        worst, rep = finite_difference_check(lambda: loss_and_grads(m, batch)[0], m["params"], G, rng)
        out.append((stage, worst, len(rep)))
    return out


def seal_property(seed, n_draws=12):
    """C6.1: mean over the bank of the decision logits equals the output bias for any parameters."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 61]))
    data = make_data(seed, quick=True)
    dev = {k: 0.0 for k in ("nafis", "galen", "unstrained")}
    for draw in range(n_draws):
        scale = [1.0, 3.0, 0.3][draw % 3]
        q = rng.normal(size=(16, Q_DIM)) * (1 + draw % 4)
        X = np.concatenate([np.repeat(q, N_BANK, 0), np.tile(data["bank"].reshape(N_BANK, -1), (16, 1))], axis=1)
        for kind in dev:
            m = build_model(X.shape[1], N_CLS, "vector_classification", np.random.default_rng(draw),
                            kind=kind, q_dim=Q_DIM, n_slots=N_SLOT, bank=data["bank"])
            for k in m["params"]:
                m["params"][k] = m["params"][k] * scale + (0.5 * rng.normal(size=m["params"][k].shape) if k[0] == "b" else 0)
            L = logits_of(m, X).reshape(16, N_BANK, -1).mean(1)
            dev[kind] = max(dev[kind], float(np.abs(L - m["params"]["bs"]).max()))
    return dev


def scale_property(seed):
    """C6.2: parcels have unit RMS, so the decision is invariant to the scale of the query parcels."""
    data = make_data(seed, quick=True)
    X = data["test_id"]["X"][:64]
    m = build_model(X.shape[1], N_CLS, "vector_classification", np.random.default_rng(seed), q_dim=Q_DIM,
                    n_slots=N_SLOT, bank=data["bank"])
    ref = logits_of(m, X)
    blood = hidden_states(m, X)["blood"]
    worst = float(np.abs(np.sqrt((blood ** 2).mean(-1)) - 1.0).max())
    broken = 0.0
    for c in (0.1, 0.5, 3.0, 20.0):
        m2 = {**m, "params": {**m["params"], "Wb2": m["params"]["Wb2"] * c}}
        worst = max(worst, float(np.abs(logits_of(m2, X) - ref).max()))
        broken = max(broken, float(np.abs(logits_of(knockout(m2, "thinning", "identity"), X)
                                          - logits_of(knockout(m, "thinning", "identity"), X)).max()))
    return worst, broken


def learning_check(model, losses, split_id, thr):
    early, late = losses[:50].mean(), losses[-50:].mean()
    drop = 1.0 - late / early
    maj = float(np.bincount(split_id["y"], minlength=N_CLS).max() / len(split_id["y"]))
    acc = accuracy(model, split_id)
    return drop >= thr["loss_drop_fraction"] and acc - maj >= thr["margin_over_trivial"], drop, acc, maj


def detect_mutant(name, seed):
    global MUTANT
    MUTANT = name
    try:
        g = gradcheck("nafis", seed, steps=10)
        c1_fail = max(w for _, w, _ in g) > MIND_CARD["thresholds"]["gradcheck_rel_tol"]
        data = make_data(seed, quick=True)
        m, losses = train_kind("nafis", data, seed, 200)
        c3_ok = learning_check(m, losses, data["test_id"], MIND_CARD["thresholds"])[0]
    finally:
        MUTANT = None
    return bool(c1_fail or not c3_ok)


MUTANTS = {"sign_flip": "Adam update applied with the wrong sign",
           "zero_lr": "learning rate set to zero",
           "zero_grad_Wk": "analytic gradient of the bronchial key projection zeroed",
           "drop_bank_branch": "query gradient through the ambient (strained) branch omitted"}


def split_integrity(data):
    def hashes(X):
        return {hashlib.sha1(r.tobytes()).hexdigest() for r in X}
    tr = hashes(data["train"]["X"])
    others = set()
    for k in ("val", "test_id", "test_cf", "test_airless"):
        others |= hashes(data[k]["X"])
    return len(tr & others) == 0


# ------------------------------------------------------------ protocol ----
def run_seed(seed, budget, quick):
    data = make_data(seed, quick)
    models, losses, res = {}, {}, {}
    for kind in KINDS:
        models[kind], losses[kind] = train_kind(kind, data, seed, budget)
        m = models[kind]
        res[kind] = {"id": accuracy(m, data["test_id"]), "cf": accuracy(m, data["test_cf"]),
                     "airless": accuracy(m, data["test_airless"]), "leak": leakage_index(m, data),
                     "auroc": conflict_auroc(m, data), "params": n_params(m)}
        for v in res[kind].values():
            if not np.isfinite(v):
                raise FloatingPointError("non-finite metric")
    res["cad_cf"] = cad_accuracy(models["baseline"], data["test_cf"])
    res["cad_airless"] = cad_accuracy(models["baseline"], data["test_airless"])
    nf = models["nafis"]
    res["ko"] = {}
    for name, mode in (("straining", "mean"), ("straining", "identity"), ("thinning", "mean"),
                       ("alveolar_exchange", "zero"), ("lung_nourishment", "zero")):
        res["ko"]["%s/%s" % (name, mode)] = accuracy(knockout(nf, name, mode), data["test_cf"]) - res["nafis"]["cf"]
    return data, models, losses, res


def hypothesis_table(per_seed, rng):
    get = {
        "H-SIG": lambda r: r["nafis"]["cf"] - r["baseline"]["cf"],
        "H-NEC": lambda r: r["ko"]["straining/mean"] - r["ko"]["thinning/mean"],
        "H-BLIND": lambda r: r["nafis"]["airless"] - r["baseline"]["airless"],
        "H-RIVAL": lambda r: r["nafis"]["cf"] - r["galen"]["cf"],
        "H-PORE": lambda r: r["nafis"]["cf"] - r["unstrained"]["cf"],
        "H-META": lambda r: r["nafis"]["auroc"] - r["baseline"]["auroc"],
    }
    rows = []
    for h in MIND_CARD["hypotheses"]:
        diffs = [get[h["id"]](r) for r in per_seed]
        if len(diffs) < 2:
            rows.append({"id": h["id"], "metric": h["metric"], "mean_diff": float(np.mean(diffs)), "ci95": [None, None],
                         "mesi": h["mesi"], "n_seeds": len(diffs), "verdict": "not evaluated"})
            continue
        mean, ci = paired_bootstrap(diffs, rng)
        rows.append({"id": h["id"], "metric": h["metric"], "mean_diff": mean, "ci95": ci, "mesi": h["mesi"],
                     "n_seeds": len(diffs), "verdict": verdict(mean, ci, h["mesi"], h["direction"])})
    return rows


def correctness_suite(seed, first, quick, t0, budget_s):
    thr = MIND_CARD["thresholds"]
    data, models, losses, res = first
    rows, gc = [], []
    for kind in KINDS:
        for stage, worst, nt in gradcheck(kind, seed):
            gc.append((kind, stage, worst, nt))
    gmax = max(w for *_, w, _ in gc)
    rows.append(("C1", "gradient_check", gmax <= thr["gradcheck_rel_tol"],
                 "max rel err %.2e over %d tensor checks" % (gmax, sum(n for *_, n in gc))))
    m_a, l_a = train_kind("nafis", data, seed, 50)
    m_b, l_b = train_kind("nafis", data, seed, 50)
    same = np.array_equal(l_a, l_b) and np.array_equal(predict(m_a, data["test_id"]["X"]), predict(m_b, data["test_id"]["X"]))
    finite = all(np.isfinite(v).all() for m in models.values() for v in m["params"].values())
    rows.append(("C2", "determinism_finiteness", bool(same and finite), "identical reruns; finite params"))
    ok3, info = True, []
    for kind in KINDS:
        ok, drop, acc, maj = learning_check(models[kind], losses[kind], data["test_id"], thr)
        ok3 &= ok
        info.append("%s drop %.2f acc %.3f" % (kind, drop, acc))
    rows.append(("C3", "learning", ok3, "; ".join(info) + "; majority %.3f" % maj))
    shuf = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in data["train"].items()}
    shuf["y"] = np.random.default_rng(seed + 5).permutation(shuf["y"])
    m_s, _ = train_kind("nafis", {**data, "train": shuf}, seed, 600 if quick else 1200)
    acc_s = accuracy(m_s, data["test_id"])
    rows.append(("C4", "shuffled_label_control", abs(acc_s - maj) <= thr["shuffle_band"],
                 "accuracy %.3f vs majority %.3f (band %.2f)" % (acc_s, maj, thr["shuffle_band"])))
    det = {name: detect_mutant(name, seed) for name in MUTANTS}
    rows.append(("C5", "mutant_detection", all(det.values()),
                 "%d/%d detected (%s)" % (sum(det.values()), len(det), ", ".join(det))))
    dev = seal_property(seed)
    sw, sb = scale_property(seed)
    ok6 = dev["nafis"] <= thr["seal_tol"] and dev["galen"] > 1e-3 and dev["unstrained"] > 1e-3 \
        and sw <= thr["scale_tol"] and sb > 1e-3
    rows.append(("C6", "property_tests", ok6,
                 "seal dev %.1e (neg. controls galen %.2f, unstrained %.2f); scale dev %.1e (neg. control %.2f)"
                 % (dev["nafis"], dev["galen"], dev["unstrained"], sw, sb)))
    rows.append(("C7", "split_integrity", split_integrity(data), "no train row in any evaluation split"))
    el = time.time() - t0
    rows.append(("C8", "budget", el <= budget_s, "%.1f s so far of %d s" % (el, budget_s)))
    return rows, gc, det


def _fmt_rows(rows, res_seeds, hyp, gc, seeds, t0, rep):
    L = ["environment: python %s, numpy %s" % (sys.version.split()[0], np.__version__),
         "seeds: %s" % seeds, "parameters: " + ", ".join("%s %d" % (k, res_seeds[0][k]["params"]) for k in KINDS),
         "gradcheck: %d tensor checks over %d models, init and after 60 steps, max rel err %.2e (floor 1e-4)"
         % (sum(n for *_, n in gc), len(KINDS), max(w for *_, w, _ in gc)),
         "", "correctness"]
    L += ["  %-3s %-24s %-5s %s" % (i, n, "PASS" if ok else "FAIL", d) for i, n, ok, d in rows]
    L += ["", "accuracy (mean over seeds)        held-out  counterfact.  airless  leakage  conflict-AUROC"]
    for k in KINDS:
        v = {f: np.mean([r[k][f] for r in res_seeds]) for f in ("id", "cf", "airless", "leak", "auroc")}
        L.append("  %-30s %8.3f  %12.3f  %7.3f  %7.3f  %14.3f" % (k, v["id"], v["cf"], v["airless"], v["leak"], v["auroc"]))
    L.append("  %-30s %8s  %12.3f  %7.3f" % ("baseline + context-aware decoding", "-",
                                             np.mean([r["cad_cf"] for r in res_seeds]),
                                             np.mean([r["cad_airless"] for r in res_seeds])))
    L += ["", "hypotheses (paired over seeds, 95% bootstrap CI, 2000 resamples)"]
    for h in hyp:
        ci = "[%+.3f, %+.3f]" % tuple(h["ci95"]) if h["ci95"][0] is not None else "[n/a]"
        L.append("  %-8s %-8s diff %+.3f  CI %-18s mesi %.2f  -> %s" % (h["id"], h["metric"], h["mean_diff"], ci, h["mesi"], h["verdict"]))
    L += ["", "knockouts on the trained model (change in counterfactual accuracy)"]
    brng = np.random.default_rng(0)
    for key in res_seeds[0]["ko"]:
        d = [r["ko"][key] for r in res_seeds]
        mean, ci = paired_bootstrap(d, brng) if len(d) > 1 else (d[0], [d[0], d[0]])
        sig = key.split("/")[0] == "straining"
        rep["knockouts"].append({"module": key, "signature": sig, "metric_change": mean, "ci95": ci})
        L.append("  %-28s signature=%-5s %+.3f  [%+.3f, %+.3f]" % (key, sig, mean, ci[0], ci[1]))
    L += ["", "task types: %s" % ", ".join(TASK_TYPES), "runtime: %.1f s" % (time.time() - t0)]
    return L


def real_data_bridge(path, seed):
    if not path or not os.path.exists(path):
        print("real-data bridge: no --data file given or found; skipped")
        return
    raw = np.genfromtxt(path, delimiter=",", skip_header=1)
    X, y = raw[:, :-1], raw[:, -1].astype(int)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    cut = int(0.8 * len(X))
    tr, te = idx[:cut], idx[cut:]
    for kind in ("nafis", "baseline"):
        m = build_model(X.shape[1], int(y.max()) + 1, "vector_classification", rng, kind=kind)
        fit(m, {"Xtr": X[tr], "ytr": y[tr], "Xva": X[te], "yva": y[te]}, 400, rng)
        print("real-data bridge %s accuracy %.3f" % (kind, float((predict(m, X[te]) == y[te]).mean())))


def main(argv=None):
    global MUTANT
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=1213)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--card", action="store_true")
    ap.add_argument("--mutant", type=str, default=None, choices=sorted(MUTANTS))
    ap.add_argument("--data", type=str, default=None)
    a = ap.parse_args(argv)
    if a.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    t0 = time.time()
    if a.mutant:
        detected = detect_mutant(a.mutant, a.seed)
        print("mutant %s: %s" % (a.mutant, "detected by C1/C3" if detected else "NOT detected"))
        return 1 if detected else 0
    seeds = [a.seed] if a.quick else [a.seed + i for i in range(a.seeds)]
    budget, budget_s = (500, 20) if a.quick else (1600, 180)
    print("seeds:", seeds)
    try:
        per_seed, first = [], None
        for s in seeds:
            out = run_seed(s, budget, a.quick)
            first = first or out
            per_seed.append(out[3])
        rows, gc, det = correctness_suite(seeds[0], first, a.quick, t0, budget_s)
    except FloatingPointError as err:
        print("non-finite values:", err)
        return 4
    hyp = hypothesis_table(per_seed, np.random.default_rng(a.seed)) if not a.quick else \
        [dict(h, verdict="not evaluated") for h in hypothesis_table(per_seed[:1], np.random.default_rng(0))]
    rep = {"schema_version": "1.0", "chapter": 332, "file": os.path.basename(__file__),
           "card_revision": MIND_CARD["card_revision"],
           "environment": {"python": sys.version.split()[0], "numpy": np.__version__}, "seeds": seeds,
           "n_params": per_seed[0]["nafis"]["params"],
           "gradcheck": {"tensors_checked": sum(n for *_, n in gc), "tensors_total": sum(n for *_, n in gc),
                         "max_rel_error": max(w for *_, w, _ in gc),
                         "checked_at": ["init", "after_training_steps"],
                         "passed": bool(rows[0][2])},
           "correctness": [{"id": i, "name": n, "passed": bool(ok), "detail": d} for i, n, ok, d in rows],
           "mutants": {"detected": int(sum(det.values())), "total": len(det), "score": sum(det.values()) / len(det)},
           "hypotheses": hyp, "knockouts": [], "task_types": TASK_TYPES}
    rep["_lines"] = _fmt_rows(rows, per_seed, hyp, gc, seeds, t0, rep)
    elapsed = time.time() - t0
    rep["runtime_s"] = elapsed
    code = 0 if all(ok for _, _, ok, _ in rows) else 1
    if elapsed > budget_s:
        code = 3
    rep["exit_code"] = code
    write_report(rep, a.json)
    real_data_bridge(a.data, a.seed)
    return code


if __name__ == "__main__":
    sys.exit(main())

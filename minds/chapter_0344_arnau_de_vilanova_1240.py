#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0344 · Arnau de Vilanova (1240-1311)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 18, Minds 341-360 Available on Amazon https://www.amazon.com/dp/B0HQ47T2NC
# Resume and Interactive Demos at https://artificiology.com/
# END ATTRIBUTION 
"""
The GRADUS engine -- chapter 0344, Arnau de Vilanova (c. 1240 - 1311).

Thesis
------
The strength of a compound is not a sum of its ingredients' strengths but the
ratio of all the contrary parts it contains, the recipient's own complexion
counted among them, read on a scale where each doubling of the ratio adds one
degree -- and because experiment on the living body is perilous, that ratio must
be computed by reason before it is tried.

Evidence and provenance (belief; his own surviving works in critical editions)
------------------------------------------------------------------------------
D1  Aphorismi de gradibus (c. 1295-1300): the degree of a compound follows from
    the proportion of opposed qualities; degree rises arithmetically as the
    ratio rises geometrically, intensity = log2(hot/cold) in modern notation.
    Ed. McVaugh, AVOMO II (1975); McVaugh, Isis 58 (1967) 56-64. primary.
D2  He defended al-Kindi's "double ratio" as the true reading of Galen against
    Ibn Rushd (Arnau DB, UAB; Langermann 2003). scholarship.
D3  De dosi tyriacalium medicinarum explicates Galen's rule that an antidote's
    nature lies between the poison and the affected body: action is read
    relative to the recipient. Ed. McVaugh, AVOMO III (1985). primary.
D4  De intentione medicorum: medical theory is judged by its sufficiency for
    healing ("medical instrumentalism"), not by philosophical truth. Ed.
    McVaugh, AVOMO V.1 (2000); McVaugh, DSB supplement (2008). primary.
D5  Proprietas ("occult virtue"): effects whose causes reason cannot grasp
    (the magnet), set against qualities that reason can predict (Arnau DB,
    "Natural magic and astrology"; Giralt). scholarship.
D6  For theriac, whose efficacy lies in hidden properties, only experience
    can find the minimum effective dose (De dosi tyriacalium). primary.
D7  He commented on Hippocrates' first aphorism, whose Latin text calls
    experience deceptive (Repetitio super canone "Vita brevis", before 1301).
    primary (the aphorism) / scholarship (the dating).
D8  Ibn Rushd's rival computus adds degrees arithmetically in proportion to
    quantities of power, and warns that compounds may act beyond what their
    ingredients predict (Kulliyyat V; Langermann 2003). scholarship.

Doctrine -> mechanism -> test
-----------------------------
| Doctrine | Mechanism | Property test | Hypothesis |
|---|---|---|---|
| D1, D2 geometric computus | M1 gradus: log2 of dose-pooled hot over cold parts | C6.1 mediant bound, C6.2 proportion invariance, C6.3 contrary antisymmetry | H-SIG, H-NEC, H-RIVAL |
| D3 relative to the recipient | M2 complexio: the patient's own parts inside the same ratio | C6.4 monotone approach | H-NEC-B |
| D5, D6 hidden virtue | M3 proprietas: per-simple saturating channel learned only from trials | -- | knockout table |
| D4, D7 compute before trying | M4 therapy planner: optimise doses through the model, judge in the world | -- | H-PLAN |
| D8 Ibn Rushd's computus | rival: signed degrees summed by quantity | C6.1/C6.2 negative controls | H-RIVAL |
| D8 compounds beyond their parts | blind world with pairwise synergy | -- | H-BLIND |

Research question
-----------------
Compositional generalisation and safe model-based optimisation: does a pooled
contrary-ratio operator, with the recipient's own parts in the ratio, let a
model of mixtures extrapolate to compounds with more ingredients and larger
doses than it was trained on -- and does that keep a planner that optimises
doses through the model from exploiting the model's errors when trying
candidate doses on the patient is not allowed?

Closest prior art and the delta
-------------------------------
Deep Sets sum-pooling (Zaheer et al. 2017) is the standard mixture learner and
the size-matched baseline here; neural arithmetic units (Trask et al. 2018)
and divisive normalisation (Carandini & Heeger 2012) supply ratio and log
operators; Loewe concentration addition and Gaddum-Schild competitive
antagonism are the pharmacological forms; conservative objective models
(Trabucco et al. 2021) address the planning failure. Delta: the recipient's
own contrary parts enter the same pooled ratio as the ingredients', so one
operator gives a bounded (mediant) compound degree, a sigmoidal dose response
anchored at the patient's complexion, exact proportion invariance at high dose
and signed contraries -- with a separate experience-only channel for what
reason cannot compute. Overlap: Medium.

Blind spot
----------
The qualities calculus assumes a compound's action is fixed by its parts. When
two simples together make heat neither contains (pairwise synergy), the pooled
ratio cannot represent it; the full model is predicted to degrade more than
the unstructured Deep Sets baseline in that world.

Task (generative process)
-------------------------
1. A pharmacy of 12 simples; each has positive hot/cold and dry/moist parts per
   dram (log-normal) and 3 of them a hidden saturating action on sign 3.
2. A patient x in R^6 has a body degree b(x) (nonlinear) and a body strength
   m(x); the body contributes m 2^(b/2) and m 2^(-b/2) parts to each axis.
3. A recipe gives doses d (1-3 simples, 0.15-1.2 dram in training).
4. True degree g_a = log2((d.pos_a + B+_a)/(d.neg_a + B-_a)); three observed
   signs are saturating functions of g plus the hidden actions, plus noise.
5. Shifted split: 4-7 simples at 0.5-3.0 dram. Blind world: three designated
   pairs add emergent heat gain*sqrt(d_i d_j) and an emergent hidden action.
6. Planning: distempered patients (|b| >= 0.9) with four simples in stock;
   each model chooses doses (0-3 dram each) to bring predicted signs to the
   temperate signs; success is judged by the TRUE degrees (|g| <= 0.35 on
   both axes) and the true hidden action. An oracle plans on the true world.
No target is readable from a single input: signs depend on all doses and on
the patient through the ratio.

Limits
------
A research prototype of one mechanism on synthetic data. It is not a
pharmacological model, gives no dosing advice, and does not replicate a person.
The world uses the pooled-ratio law (the receptor-competition form); whether
real mixtures follow it is an empirical question the --data bridge can probe.

Run: python3 <this file>   (--quick, --card, --seed S, --seeds N, --json PATH,
--mutant NAME, --data PATH)
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 2,
    "revision_log": ["rev 2 (after exploratory seeds 0-1, before the verified run): with the full pharmacy every "
                     "planner reached the ceiling (>= 0.99) using doses inside the training range, so H-PLAN could not "
                     "discriminate. The planning CONDITION now gives each patient four simples in stock and a stronger "
                     "distemper (|b| >= 0.9), and an oracle planner on the true world reports the headroom. Statement, "
                     "metric, direction and mesi of H-PLAN are unchanged; no other hypothesis or threshold changed."],
    "generation": {"template_version": "1.0", "generator": "Claude Opus 5.5",
                   "generator_version": "claude-opus-5-5", "date": "2026-09-28"},
    "id": 344, "figure": "Arnau de Vilanova", "born": 1240, "died": 1311,
    "civilization": "Catalan (Crown of Aragon)",
    "provenance": "belief",
    "thesis": ("A compound's strength is the ratio of all the contrary parts it contains, the recipient's "
               "own complexion counted among them, read so that each doubling of the ratio adds one degree; "
               "because experiment on the body is perilous, the ratio must be computed before it is tried."),
    "evidence": [
        {"id": "D1", "claim": "Compound degree follows the proportion of opposed qualities; degree rises arithmetically as the ratio rises geometrically.",
         "basis": "primary", "source": "Aphorismi de gradibus, ed. McVaugh, AVOMO II (1975); McVaugh, Isis 58 (1967)"},
        {"id": "D2", "claim": "He defended al-Kindi's double ratio against Ibn Rushd as the authentic reading of Galen.",
         "basis": "scholarship", "source": "Arnau DB (UAB), 'Aphorisms'; Langermann 2003"},
        {"id": "D3", "claim": "An antidote's nature lies between the poison and the affected body; action is relative to the recipient.",
         "basis": "primary", "source": "De dosi tyriacalium medicinarum, ed. McVaugh, AVOMO III (1985)"},
        {"id": "D4", "claim": "Medical theory is judged by its sufficiency for healing, not by philosophical truth.",
         "basis": "primary", "source": "De intentione medicorum, ed. McVaugh, AVOMO V.1 (2000)"},
        {"id": "D5", "claim": "Proprietas: effects reason cannot grasp (the magnet), unlike qualities reason can predict.",
         "basis": "scholarship", "source": "Arnau DB (UAB), 'Natural magic and astrology'"},
        {"id": "D6", "claim": "Theriac's hidden properties: only experience finds the minimum effective dose.",
         "basis": "primary", "source": "De dosi tyriacalium medicinarum, ed. McVaugh, AVOMO III (1985)"},
        {"id": "D7", "claim": "He commented on the aphorism that calls experience deceptive.",
         "basis": "scholarship", "source": "Repetitio super canone 'Vita brevis' (before 1301), Arnau DB timeline"},
        {"id": "D8", "claim": "Ibn Rushd adds degrees arithmetically by quantity and warns compounds may act beyond their parts.",
         "basis": "scholarship", "source": "Ibn Rushd, Kulliyyat V; Langermann, 'Another Andalusian Revolt?' (2003)"},
    ],
    "research_question": {"category": "compositional and systematic generalization",
                          "question": ("Does a pooled contrary-ratio operator with the recipient's parts in the ratio "
                                       "let a mixture model extrapolate to larger compounds and doses, and keep a "
                                       "dose planner from exploiting the model's errors when trial on the patient "
                                       "is not allowed?")},
    "mechanism": {
        "name": "GRADUS engine (dose-pooled contrary ratio with the recipient in the ratio)",
        "family": "structured set function / ratio operator with model-based planning",
        "signature_modules": ["gradus", "complexio", "virtutes"],
        "closest_prior_art": ["Zaheer et al. 2017, Deep Sets (sum pooling)",
                              "Trask et al. 2018, Neural Arithmetic Logic Units",
                              "Carandini & Heeger 2012, divisive normalization",
                              "Loewe concentration addition; Gaddum-Schild competitive antagonism",
                              "Genest & Zidek 1986, linear versus logarithmic opinion pools",
                              "Trabucco et al. 2021, conservative objective models (offline MBO)"],
        "overlap": "Medium",
        "prior_art_queries": ["neural networks extrapolate mixtures set functions",
                              "deep sets mixture property prediction extrapolation",
                              "log ratio neural unit divisive normalization",
                              "offline model-based optimization distribution shift",
                              "drug combination synergy prediction dose extrapolation"],
        "contribution_type": "mechanism",
        "delta": ("The recipient's own contrary parts enter the same pooled ratio as the ingredients', giving a "
                  "bounded mediant degree, a dose response anchored at the patient, proportion invariance at high "
                  "dose and signed contraries, beside a separate experience-only channel."),
    },
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1", "property_test": "C6.1", "hypothesis": "H-SIG"},
        {"doctrine": "D2", "mechanism": "M1", "property_test": "C6.2", "hypothesis": "H-RIVAL"},
        {"doctrine": "D1", "mechanism": "M1", "property_test": "C6.3", "hypothesis": "H-NEC"},
        {"doctrine": "D3", "mechanism": "M2", "property_test": "C6.4", "hypothesis": "H-NEC-B"},
        {"doctrine": "D4", "mechanism": "M4", "property_test": "C3", "hypothesis": "H-PLAN"},
        {"doctrine": "D8", "mechanism": "blind world", "property_test": "C7", "hypothesis": "H-BLIND"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "On larger compounds at larger doses the GRADUS engine predicts signs better than size-matched Deep Sets.",
         "metric": "rmse_shifted", "split": "shifted", "comparison": "model - baseline",
         "direction": "less", "mesi": 0.05, "seeds": 5},
        {"id": "H-NEC", "statement": "Replacing the log-ratio by the plain excess (hot minus cold) hurts more than removing the readout nonlinearity.",
         "metric": "rmse_shifted_increase", "comparison": "knockout(gradus,identity) - knockout(signa_hidden,identity)",
         "direction": "greater", "mesi": 0.05, "seeds": 5},
        {"id": "H-NEC-B", "statement": "Replacing each patient's own parts by the population mean hurts more than removing the readout nonlinearity.",
         "metric": "rmse_shifted_increase", "comparison": "knockout(complexio,mean) - knockout(signa_hidden,identity)",
         "direction": "greater", "mesi": 0.02, "seeds": 5},
        {"id": "H-BLIND", "statement": "Where pairs of simples make emergent heat, the GRADUS engine degrades more than Deep Sets.",
         "condition": "blind world (pairwise synergy)", "grounding": "the qualities calculus assumed compounds act only through their parts",
         "metric": "rmse_test_degradation", "comparison": "model - baseline", "direction": "greater", "mesi": 0.02, "seeds": 5},
        {"id": "H-RIVAL", "statement": "Al-Kindi's pooled ratio (Arnau's side) extrapolates better than Ibn Rushd's arithmetic sum of degrees.",
         "metric": "rmse_shifted", "comparison": "model - rival", "direction": "less", "mesi": 0.03, "seeds": 5},
        {"id": "H-PLAN", "statement": "Therapies planned through the GRADUS engine restore more distempered patients in the true world than therapies planned through Deep Sets.",
         "metric": "plan_success", "comparison": "model - baseline", "direction": "greater", "mesi": 0.10, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.5, "margin_over_trivial": 0.25, "shuffle_band": 0.08},
    "probe_predictions": [
        {"probe": "P4", "expected": "above baseline"}, {"probe": "P10", "expected": "above baseline"},
        {"probe": "P6", "expected": "above baseline"}, {"probe": "P1", "expected": "below baseline"},
        {"probe": "P8", "expected": "equal to baseline"}, {"probe": "P9", "expected": "equal to baseline"},
    ],
    "dialectic_links": [
        {"chapter": 218, "relation": "teacher", "test": "none implemented (al-Kindi's double ratio is M1 itself)"},
        {"chapter": 299, "relation": "rival", "test": "H-RIVAL"},
        {"chapter": 287, "relation": "rival", "test": "none implemented (titration by watched response vs computation before trial)"},
        {"chapter": 362, "relation": "successor", "test": "none implemented (possible stimulus for Bradwardine's law, McVaugh 1967)"},
        {"chapter": 135, "relation": "teacher", "test": "none implemented"},
    ],
    "corpus_neighbors": [
        {"chapter": 362, "similarity": None, "difference": "Bradwardine's ProportionGate is an unsigned thresholded activation on one force and one resistance; here signed contraries are pooled over many dosed ingredients and the recipient's own parts, with no dead zone."},
        {"chapter": 287, "similarity": None, "difference": "Ibn Zuhr titrates each patient from watched response; here the dose is computed through the model before any trial, and titration is kept only for hidden virtues."},
        {"chapter": 218, "similarity": None, "difference": "Al-Kindi's chapter models decipherment and invariance; the double ratio he invented is used here, applied to compounds with the recipient in the ratio."},
        {"chapter": 371, "similarity": None, "difference": "Oresme makes the shape of a quality's distribution causal; here a quality is a scalar degree and the claim is about how parts combine."},
        {"chapter": 340, "similarity": None, "difference": "Llull adjudicates pairs by rule and counts wins; here nothing is pairwise except the blind-spot synergy the model cannot represent."},
        {"chapter": 123, "similarity": None, "difference": "Dioscorides maps simples on an effect-affinity manifold (closer to the Deep Sets baseline); here compound effects are computed by a fixed law."},
    ],
    "barometer": {
        "cognitive_processing": ["P4 length (compound-size) extrapolation", "P10 sample efficiency"],
        "embodied_cognition": ["recipient's own state inside the computation (complexio)"],
        "world_modeling": ["shifted-split forecast of signs", "true-world planning success"],
        "consciousness": [],
        "language_understanding": [],
        "emotional_intelligence": [],
        "creativity": ["novel compounds planned under a validity constraint (latitude of health)"],
        "autonomy": ["self-set dose plans judged in the world (H-PLAN)"],
    },
    "task_types": ["vector_regression", "vector_classification"],
    "applications": [
        {"use": "Predicting drug-combination response beyond tested doses and combination sizes (research decision support only)",
         "sector": "pharmacology research", "dataset": "NCI ALMANAC (Holbeck et al. 2017) or DrugComb (Zagidullin et al. 2019)"},
        {"use": "Mixture property prediction for formulations with more components than in training",
         "sector": "flavour, fragrance and materials", "dataset": "Snitz et al. 2013 odour-mixture perceptual similarity data"},
        {"use": "Offline model-based design without active trials: structural priors against model exploitation",
         "sector": "machine-learning infrastructure", "dataset": "Design-Bench (Trabucco et al. 2022)"},
    ],
    "safety_notes": ("Synthetic data only. Medicine is modelled as research decision support: the planner is a test "
                     "of model exploitation, not a dosing method, and no output is a treatment recommendation. No "
                     "claim of replicating a person; no generated sentence is presented as his words."),
}

import argparse
import json
import math
import os
import sys
import time
import hashlib
import numpy as np

K, A, P, S = 12, 2, 6, 3            # simples; contrary axes; patient features; observed signs
LN2 = math.log(2.0)
FLOOR = 0.05                        # minimum body parts, keeps every pool strictly positive
LATITUDE = 0.35                     # latitude of health on each axis, in degrees
DMAX = 3.0                          # planner's dose ceiling per simple (training never exceeds 1.2)
TRAIN_DOSE, SHIFT_DOSE = (0.15, 1.2), (0.5, 3.0)
TRAIN_SIZE, SHIFT_SIZE = (1, 3), (4, 7)
HID = {"gradus": (10, 16), "rushd": (13, 16)}   # complexio hidden, signa hidden (size-matched to Deep Sets)
DS_EMB, DS_HID = 6, 14
CLIP_NORM = 5.0
BUDGET_FULL_S, BUDGET_QUICK_S = 180.0, 20.0


# BEGIN STANDARD UTILITIES v1.0
def sigmoid(z):
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def log_sigmoid(z):
    return -np.logaddexp(0.0, -z)


def softplus(z):
    return np.logaddexp(0.0, z)


def logsumexp(z, axis=-1, keepdims=False):
    m = np.max(z, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(z - m), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


def stable_softmax(z, axis=-1):
    return np.exp(z - logsumexp(z, axis=axis, keepdims=True))


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads, sign=1.0):
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= sign * self.lr * mhat / (np.sqrt(vhat) + self.eps)


def clip_global_norm(grads, max_norm):
    total = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    if total > max_norm:
        scale = max_norm / (total + 1e-12)
        for k in grads:
            grads[k] = grads[k] * scale
    return total


def finite_difference_check(loss_fn, params, grads, rng, n_entries=20, eps=1e-6, tol=1e-5):
    """Central differences in float64 on >=20 random entries per tensor plus the
    largest-gradient entry. Relative error uses max(|a|+|b|, 1e-4) in the
    denominator: at eps=1e-6 central differences carry about 1e-10 absolute
    noise, so entries with gradients below 1e-4 are compared on that absolute
    scale rather than amplified to a meaningless ratio."""
    worst = 0.0
    checked = 0
    for k, p in params.items():
        flat = p.reshape(-1)
        n = flat.size
        idx = set(int(i) for i in rng.integers(0, n, size=min(n_entries, n)))
        idx.add(int(np.argmax(np.abs(grads[k]).reshape(-1))))
        for i in idx:
            old = flat[i]
            flat[i] = old + eps
            lp = loss_fn()
            flat[i] = old - eps
            lm = loss_fn()
            flat[i] = old
            fd = (lp - lm) / (2 * eps)
            an = grads[k].reshape(-1)[i]
            rel = abs(fd - an) / max(abs(fd) + abs(an), 1e-4)
            worst = max(worst, rel)
        checked += 1
    return {"tensors_checked": checked, "tensors_total": len(params),
            "max_rel_error": float(worst), "passed": bool(worst <= tol)}


def paired_bootstrap_ci(diffs, rng, n_resamples=2000, alpha=0.05):
    diffs = np.asarray(diffs, dtype=np.float64)
    n = diffs.size
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        means[i] = diffs[rng.integers(0, n, size=n)].mean()
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(diffs.mean()), [float(lo), float(hi)]


def verdict(mean, ci, direction, mesi):
    lo, hi = ci
    if direction == "greater":
        if lo > 0 and mean >= mesi:
            return "supported"
        if hi < 0:
            return "contradicted"
    else:
        if hi < 0 and -mean >= mesi:
            return "supported"
        if lo > 0:
            return "contradicted"
    return "inconclusive"


def write_report(report, path=None):
    ch = report["chapter"]
    lines = [f"=== VERIFIED REPORT · chapter {ch:04d} ===",
             f"file: {report['file']}   card_revision: {report['card_revision']}",
             f"environment: python {report['environment']['python']}  numpy {report['environment']['numpy']}",
             f"seeds: {report['seeds']}   runtime_s: {report['runtime_s']:.1f}   n_params: {report['n_params']}",
             (f"gradcheck: {report['gradcheck']['tensors_checked']}/{report['gradcheck']['tensors_total']} tensors, "
              f"max_rel_error {report['gradcheck']['max_rel_error']:.2e}, checked_at {report['gradcheck']['checked_at']}, "
              f"passed={report['gradcheck']['passed']}"),
             "correctness:"]
    for c in report["correctness"]:
        lines.append(f"  {c['id']:<6} {c['name']:<28} {'PASS' if c['passed'] else 'FAIL'}  {c['detail']}")
    mu = report["mutants"]
    lines.append(f"mutants: detected {mu['detected']}/{mu['total']}  score {mu['score']:.2f}")
    lines.append("hypotheses:")
    for h in report["hypotheses"]:
        lines.append(f"  {h['id']:<13} {h['metric']:<22} mean {h['mean_diff']:+.4f}  ci95 [{h['ci95'][0]:+.4f}, {h['ci95'][1]:+.4f}]"
                     f"  mesi {h['mesi']:.2f}  n {h['n_seeds']}  -> {h['verdict']}")
    lines.append("knockouts (metric change on the shifted split, mean over seeds):")
    for k in report["knockouts"]:
        lines.append(f"  {k['module']:<16} signature={str(k['signature']):<5} change {k['metric_change']:+.4f}"
                     f"  ci95 [{k['ci95'][0]:+.4f}, {k['ci95'][1]:+.4f}]")
    lines.append(f"task_types: {report['task_types']}   exit_code: {report['exit_code']}")
    lines.append("=== END REPORT ===")
    print("\n".join(lines))
    if path:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
# END STANDARD UTILITIES


# ----------------------------------------------------------------------------
# 6. Data and tasks: a pharmacy, its patients, and the world that judges them
# ----------------------------------------------------------------------------
def make_world(rng):
    """Twelve simples with positive contrary parts per dram. A shared potency
    factor makes some simples act on the body faster than others at equal dose."""
    potency = np.exp(rng.normal(0.0, 0.35, size=(K, 1)))
    world = {"pos": np.exp(rng.normal(0.0, 0.8, size=(K, A))) * potency,
             "neg": np.exp(rng.normal(0.0, 0.8, size=(K, A))) * potency,
             "spec_amp": np.zeros(K), "spec_rate": np.ones(K),
             "Wb": rng.normal(0.0, 0.6, size=(P, A)), "vb": rng.normal(0.0, 0.6, size=(P, A))}
    hidden = rng.choice(K, size=3, replace=False)
    world["spec_amp"][hidden] = rng.choice([-1.0, 1.0], size=3) * rng.uniform(0.6, 1.0, size=3)
    world["spec_rate"][hidden] = rng.uniform(0.8, 2.0, size=3)
    rest = [i for i in range(K) if i not in set(hidden.tolist())]
    order = rng.permutation(rest)
    world["pairs"] = [(int(order[2 * j]), int(order[2 * j + 1])) for j in range(3)]
    world["gains"] = rng.uniform(1.5, 3.0, size=3)
    return world


def body_parts(world, pt):
    """The patient's complexion: a nonlinear body degree b and a body strength m,
    expressed as m*2^(b/2) hot parts against m*2^(-b/2) cold parts per axis."""
    b = 1.1 * np.tanh(pt @ world["Wb"]) + 0.3 * np.sin(pt @ world["vb"])
    m = 1.2 * np.exp(0.25 * np.tanh(pt[:, 5:6]))
    return m * 2.0 ** (b / 2), m * 2.0 ** (-b / 2), b


def true_outcome(world, d, pt, blind=False):
    bp, bn, _ = body_parts(world, pt)
    hot = d @ world["pos"] + bp
    cold = d @ world["neg"] + bn
    spec = (1.0 - np.exp(-world["spec_rate"] * d)) @ world["spec_amp"]
    if blind:
        for (i, j), gain in zip(world["pairs"], world["gains"]):
            s = np.sqrt(d[:, i] * d[:, j])
            hot[:, 0] += gain * s                  # heat that neither simple contains
            spec += 0.5 * np.tanh(2.0 * s)          # an action that neither simple has
    g = np.log2(hot / cold)
    y = np.stack([np.tanh(0.6 * g[:, 0]) + 0.2 * np.tanh(0.6 * g[:, 1]),
                  np.tanh(0.6 * g[:, 1]) - 0.1 * np.tanh(0.6 * g[:, 0]),
                  0.35 * np.tanh(0.8 * g[:, 0]) + spec], axis=1)
    return y, g, spec


def sample_recipes(world, rng, n, size, dose):
    d = np.zeros((n, K))
    for r in range(n):
        k = int(rng.integers(size[0], size[1] + 1))
        idx = list(rng.choice(K, size=k, replace=False))
        if k >= 2 and rng.random() < 0.3:          # designated pairs are common, in every world
            i, j = world["pairs"][int(rng.integers(0, 3))]
            idx = [i, j] + [t for t in idx if t not in (i, j)][:k - 2]
        d[r, idx] = rng.uniform(dose[0], dose[1], size=len(idx))
    return d


def make_split(world, rng, n, size, dose, blind):
    d = sample_recipes(world, rng, n, size, dose)
    pt = rng.standard_normal((n, P))
    y, g, spec = true_outcome(world, d, pt, blind)
    y = y + rng.normal(0.0, 0.04, size=y.shape)
    return {"X": np.concatenate([d, pt], axis=1), "Yraw": y, "g": g}


def standardise(splits):
    mu, sd = splits["train"]["Yraw"].mean(0), splits["train"]["Yraw"].std(0) + 1e-9
    for s in splits.values():
        s["Y"] = (s["Yraw"] - mu) / sd
    return {"mu": mu, "sd": sd}


def make_data(rng, blind=False, n_train=3000):
    seeds = np.random.SeedSequence(int(rng.integers(0, 2 ** 31))).spawn(6)
    r = [np.random.default_rng(s) for s in seeds]
    world = make_world(r[0])
    splits = {"train": make_split(world, r[1], n_train, TRAIN_SIZE, TRAIN_DOSE, blind),
              "val": make_split(world, r[2], 600, TRAIN_SIZE, TRAIN_DOSE, blind),
              "test": make_split(world, r[3], 1200, TRAIN_SIZE, TRAIN_DOSE, blind),
              "shifted": make_split(world, r[4], 1200, SHIFT_SIZE, SHIFT_DOSE, blind)}
    splits["stats"] = standardise(splits)
    splits["world"], splits["blind"] = world, blind
    return splits


def row_hashes(X):
    return {hashlib.sha1(np.round(r, 9).tobytes()).hexdigest() for r in X}


# ----------------------------------------------------------------------------
# 7. Model: virtutes -> gradus (with complexio) -> signa, plus proprietas
# ----------------------------------------------------------------------------
TASK_TYPES = ["vector_regression", "vector_classification"]


def build_model(in_dim=K + P, out_dim=S, task_type="vector_regression", rng=None, kind="gradus",
                n_simples=None, native=True, **cfg):
    if task_type not in TASK_TYPES or kind not in ("gradus", "rushd", "deepsets"):
        raise ValueError(f"{task_type}/{kind}")
    rng = rng if rng is not None else np.random.default_rng(0)
    k = n_simples if n_simples is not None else (K if native else in_dim // 2)
    pdim = in_dim - k

    def w(shape, scale):
        return rng.standard_normal(shape) * scale
    if kind == "deepsets":
        params = {"ds_E": w((k, DS_EMB), 0.5), "ds_W1": w((DS_EMB + pdim, DS_HID), 1 / math.sqrt(DS_EMB + pdim)),
                  "ds_b1": np.zeros(DS_HID), "ds_W2": w((DS_HID, out_dim), 1 / math.sqrt(DS_HID)),
                  "ds_b2": np.zeros(out_dim)}
    else:
        hc, hs = HID[kind]
        params = {"cx_W1": w((pdim, hc), 1 / math.sqrt(pdim)), "cx_b1": np.zeros(hc),
                  "cx_W2": w((hc, 2 * A if kind == "gradus" else A), 0.3 / math.sqrt(hc)),
                  "cx_b2": np.full(2 * A, 0.5) if kind == "gradus" else np.zeros(A),
                  "sg_W1": w((A, hs), 0.7), "sg_b1": np.zeros(hs), "sg_W2": w((hs, out_dim), 1 / math.sqrt(hs)),
                  "sg_b2": np.zeros(out_dim), "sg_V": w((A, out_dim), 0.3),
                  "pr_amp": w((k, out_dim), 0.05), "pr_rate": np.full(k, 0.5)}
        if kind == "gradus":
            params["vir_pos"] = rng.normal(0.5, 0.4, size=(k, A))
            params["vir_neg"] = rng.normal(0.5, 0.4, size=(k, A))
        else:
            params["ir_deg"] = w((k, A), 0.5)
    return {"kind": kind, "params": params, "ko": {}, "mutant": None,
            "cfg": {"K": k, "native": native, "task": task_type, "out": out_dim}}


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


def split_input(model, X):
    k = model["cfg"]["K"]
    raw = X[:, :k]
    return (raw if model["cfg"]["native"] else softplus(raw)), X[:, k:]


def _mean_rows(x):
    return np.broadcast_to(x.mean(0, keepdims=True), x.shape).copy()


def degrees(model, d, pt, with_body=True):
    """The contrary ratio. For the gradus engine: log2 of (dose-pooled hot parts +
    the body's hot parts) over (dose-pooled cold parts + the body's cold parts).
    For Ibn Rushd's computus: signed degrees summed by quantity plus the body's degree."""
    p, ko, c = model["params"], model["ko"], {}
    c["a1"] = np.tanh(pt @ p["cx_W1"] + p["cx_b1"])
    c["z2"] = c["a1"] @ p["cx_W2"] + p["cx_b2"]
    if model["kind"] == "rushd":
        beta = _mean_rows(c["z2"]) if ko.get("complexio") == "mean" else c["z2"]
        return d @ p["ir_deg"] + (beta if with_body else 0.0), c
    body = softplus(c["z2"]) + FLOOR
    if ko.get("complexio") == "mean":
        body = _mean_rows(body)
    if not with_body:
        body = np.zeros_like(body)
    vp, vn = softplus(p["vir_pos"]), softplus(p["vir_neg"])
    if ko.get("virtutes") == "mean":
        vp, vn = _mean_rows(vp), _mean_rows(vn)
    c.update(vp=vp, vn=vn, hot=d @ vp + body[:, :A], cold=d @ vn + body[:, A:])
    if ko.get("gradus") == "identity":
        return c["hot"] - c["cold"], c                  # Aristotle's plain excess, no ratio
    return (np.log(c["hot"]) - np.log(c["cold"])) / LN2, c


def forward(model, X):
    p, ko = model["params"], model["ko"]
    d, pt = split_input(model, X)
    c = {"d": d, "pt": pt}
    if model["kind"] == "deepsets":
        c["zp"] = np.concatenate([d @ p["ds_E"], pt], axis=1)
        pre = c["zp"] @ p["ds_W1"] + p["ds_b1"]
        c["h"] = pre if ko.get("ds_hidden") == "identity" else np.tanh(pre)
        return c["h"] @ p["ds_W2"] + p["ds_b2"], c
    g, cg = degrees(model, d, pt)
    c.update(cg, g=g)
    pre = g @ p["sg_W1"] + p["sg_b1"]
    c["h"] = pre if ko.get("signa_hidden") == "identity" else np.tanh(pre)
    y = c["h"] @ p["sg_W2"] + p["sg_b2"] + g @ p["sg_V"]
    if ko.get("proprietas") != "zero":
        c["r"] = softplus(p["pr_rate"])
        c["ex"] = np.exp(-c["r"] * d)
        y = y + (1.0 - c["ex"]) @ p["pr_amp"]
    return y, c


def backward(model, c, dy):
    """Hand-derived reverse pass. Returns parameter gradients and the gradient
    with respect to the doses (used by the therapy planner)."""
    p, ko, kind = model["params"], model["ko"], model["kind"]
    gr = {k: np.zeros_like(v) for k, v in p.items()}
    d, pt = c["d"], c["pt"]
    if kind == "deepsets":
        gr["ds_W2"], gr["ds_b2"] = c["h"].T @ dy, dy.sum(0)
        dpre = dy @ p["ds_W2"].T
        if ko.get("ds_hidden") != "identity":
            dpre = dpre * (1 - c["h"] ** 2)
        gr["ds_W1"], gr["ds_b1"] = c["zp"].T @ dpre, dpre.sum(0)
        dz = (dpre @ p["ds_W1"].T)[:, :DS_EMB]
        gr["ds_E"] = d.T @ dz
        return gr, dz @ p["ds_E"].T
    gr["sg_W2"], gr["sg_b2"], gr["sg_V"] = c["h"].T @ dy, dy.sum(0), c["g"].T @ dy
    dpre = dy @ p["sg_W2"].T
    if ko.get("signa_hidden") != "identity":
        dpre = dpre * (1 - c["h"] ** 2)
    gr["sg_W1"], gr["sg_b1"] = c["g"].T @ dpre, dpre.sum(0)
    dg = dpre @ p["sg_W1"].T + dy @ p["sg_V"].T
    dd = np.zeros_like(d)
    if ko.get("proprietas") != "zero":
        gr["pr_amp"] = (1.0 - c["ex"]).T @ dy
        de = dy @ p["pr_amp"].T
        gr["pr_rate"] = (de * d * c["ex"]).sum(0) * sigmoid(p["pr_rate"])
        dd += de * c["r"] * c["ex"]
    if kind == "rushd":
        gr["ir_deg"] = d.T @ dg
        dd += dg @ p["ir_deg"].T
        dz2 = _mean_rows(dg) if ko.get("complexio") == "mean" else dg
    else:
        if ko.get("gradus") == "identity":
            dh, dc = dg, -dg
        else:
            dh, dc = dg / (LN2 * c["hot"]), -dg / (LN2 * c["cold"])
        dvp, dvn = d.T @ dh, d.T @ dc
        if ko.get("virtutes") == "mean":
            dvp, dvn = _mean_rows(dvp), _mean_rows(dvn)
        gr["vir_pos"] = dvp * sigmoid(p["vir_pos"])
        gr["vir_neg"] = dvn * sigmoid(p["vir_neg"])
        dd += dh @ c["vp"].T + dc @ c["vn"].T
        dbody = np.concatenate([dh, dc], axis=1)
        if ko.get("complexio") == "mean":
            dbody = _mean_rows(dbody)
        dz2 = dbody * sigmoid(c["z2"])
    gr["cx_W2"], gr["cx_b2"] = c["a1"].T @ dz2, dz2.sum(0)
    dz1 = (dz2 @ p["cx_W2"].T) * (1 - c["a1"] ** 2)
    gr["cx_W1"], gr["cx_b1"] = pt.T @ dz1, dz1.sum(0)
    if model["mutant"] == "zero_grad_vir_pos" and "vir_pos" in gr:
        gr["vir_pos"] = np.zeros_like(gr["vir_pos"])
    if model["mutant"] == "flip_body_grad":
        gr["cx_W1"] = -gr["cx_W1"]
    return gr, dd


def loss_and_grads(model, batch):
    X, Y = batch
    y, c = forward(model, X)
    if model["cfg"]["task"] == "vector_classification":
        lab = Y.astype(int)
        logp = y - logsumexp(y, axis=1, keepdims=True)
        loss = float(-np.mean(logp[np.arange(len(lab)), lab]))
        dy = np.exp(logp)
        dy[np.arange(len(lab)), lab] -= 1.0
        dy /= len(lab)
    else:
        r = y - Y
        loss = float(np.mean(r * r))
        dy = 2.0 * r / r.size
    return loss, backward(model, c, dy)[0]


def predict(model, X):
    y = forward(model, X)[0]
    return np.argmax(y, axis=1) if model["cfg"]["task"] == "vector_classification" else y


def hidden_states(model, X):
    c = forward(model, X)[1]
    keys = ("g", "hot", "cold", "h") if model["kind"] == "gradus" else ("g", "h") if model["kind"] == "rushd" else ("zp", "h")
    return {k: c[k] for k in keys}


def rmse(model, split):
    return float(np.sqrt(np.mean((predict(model, split["X"]) - split["Y"]) ** 2)))


# ----------------------------------------------------------------------------
# 8. Baselines, the rival, and the planner that tests them all in the world
# ----------------------------------------------------------------------------
def build_baseline(rng):
    """Deep Sets with dose-weighted sum pooling: the standard learner for
    mixtures, size-matched to the gradus engine (within 10% of parameters)."""
    return build_model(rng=rng, kind="deepsets")


def build_rival(rng):
    """Ibn Rushd's computus: each simple a signed degree, degrees added in
    proportion to quantity, the body's own degree added last."""
    return build_model(rng=rng, kind="rushd")


def plan_therapy(grad_fn, avail, rng, steps=250, lr=0.04, parsimony=0.002):
    """Reason before experiment: choose doses of the simples in stock (0..DMAX
    each) by descending grad_fn, the gradient of the squared distance between
    predicted and temperate signs. Nothing is tried on the patient meanwhile."""
    d = (0.1 + rng.uniform(0, 0.02, size=avail.shape)) * avail
    m, v = np.zeros_like(d), np.zeros_like(d)
    for t in range(1, steps + 1):
        grad = grad_fn(d) + parsimony
        m, v = 0.9 * m + 0.1 * grad, 0.999 * v + 0.001 * grad * grad
        d = np.clip(d - lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8), 0.0, DMAX) * avail
    return d


def model_gradient(model, pt, target):
    def grad(d):
        y, c = forward(model, np.concatenate([d, pt], axis=1))
        return backward(model, c, 2.0 * (y - target) / S)[1]
    return grad


def oracle_gradient(data, pt, target_raw, h=1e-5):
    """The planner's objective computed on the TRUE world by central differences
    (noise-free signs): the ceiling any learned model could reach."""
    def cost(d):
        y = true_outcome(data["world"], d, pt, data["blind"])[0]
        return np.sum((y - target_raw) ** 2, axis=1) / S
    def grad(d):
        out = np.zeros_like(d)
        for j in range(K):
            e = np.zeros_like(d)
            e[:, j] = h
            out[:, j] = (cost(d + e) - cost(np.maximum(d - e, 0.0))) / (h + np.minimum(d[:, j], h))
        return out
    return grad


def planning_cohort(world, rng, n=240):
    """Distempered patients (body degree at least 0.9 from temperate on some
    axis), each with only four of the twelve simples in stock."""
    pts = []
    while sum(len(x) for x in pts) < n:
        pt = rng.standard_normal((6 * n, P))
        b = body_parts(world, pt)[2]
        pts.append(pt[np.abs(b).max(1) >= 0.9])
    avail = np.zeros((n, K))
    for i in range(n):
        avail[i, rng.choice(K, size=4, replace=False)] = 1.0
    return np.concatenate(pts)[:n], avail


def plan_success(model, data, cohort, rng):
    """Fraction of the cohort whose TRUE degrees end inside the latitude of health
    with the hidden action held below 0.5, and the mean total dose used.
    model=None plans on the true world (oracle)."""
    pt, avail = cohort
    stats = data["stats"]
    if model is None:
        grad = oracle_gradient(data, pt, np.zeros(S))
    else:
        grad = model_gradient(model, pt, (np.zeros(S) - stats["mu"]) / stats["sd"])
    d = plan_therapy(grad, avail, rng)
    _, g, spec = true_outcome(data["world"], d, pt, data["blind"])
    ok = (np.abs(g).max(1) <= LATITUDE) & (np.abs(spec) <= 0.5)
    return float(ok.mean()), float(d.max(1).mean())


# ----------------------------------------------------------------------------
# 9. Registries: modules, knockouts, mutants
# ----------------------------------------------------------------------------
def modules(model):
    if model["kind"] == "deepsets":
        return {"ds_pool": {"params": ["ds_E"], "role": "dose-weighted sum pooling of ingredient embeddings", "signature": False},
                "ds_hidden": {"params": ["ds_W1", "ds_b1"], "role": "tanh hidden layer on pooled set and context", "signature": False},
                "ds_readout": {"params": ["ds_W2", "ds_b2"], "role": "linear readout", "signature": False}}
    out = {"complexio": {"params": ["cx_W1", "cx_b1", "cx_W2", "cx_b2"],
                         "role": "context encoder giving the recipient's own positive pool contributions", "signature": True},
           "signa_hidden": {"params": ["sg_W1", "sg_b1"], "role": "tanh hidden layer of the readout from degrees", "signature": False},
           "signa_readout": {"params": ["sg_W2", "sg_b2", "sg_V"], "role": "linear readout with skip from degrees", "signature": False},
           "proprietas": {"params": ["pr_amp", "pr_rate"], "role": "per-item saturating additive dose response", "signature": False}}
    if model["kind"] == "gradus":
        out["virtutes"] = {"params": ["vir_pos", "vir_neg"], "role": "nonnegative per-item contrary pools per unit dose", "signature": True}
        out["gradus"] = {"params": [], "role": "log2 ratio of two positive dose-pooled sums (parameter-free)", "signature": True}
    else:
        out["ir_deg"] = {"params": ["ir_deg"], "role": "signed per-item degree, summed linearly in dose", "signature": False}
    return out


def knockout(model, name, mode="identity"):
    if name not in modules(model) or mode not in ("identity", "zero", "mean"):
        raise ValueError(f"{name}/{mode}")
    out = {"kind": model["kind"], "params": {k: v.copy() for k, v in model["params"].items()},
           "ko": dict(model["ko"]), "mutant": model["mutant"], "cfg": dict(model["cfg"])}
    out["ko"][name] = mode
    return out


MUTANTS = {
    "sign_flip": "Adam step applied with the wrong sign",
    "zero_lr": "learning rate set to zero",
    "zero_grad_vir_pos": "analytic gradient of the hot pools replaced by zeros",
    "flip_body_grad": "complexio input-layer gradient returned with the wrong sign",
}


# ----------------------------------------------------------------------------
# 10. Training
# ----------------------------------------------------------------------------
def fit(model, data, budget, rng, lr=0.01, batch_size=128, eval_every=100):
    p = model["params"]
    opt = Adam(p, lr=0.0 if model["mutant"] == "zero_lr" else lr)
    sign = -1.0 if model["mutant"] == "sign_flip" else 1.0
    X, Y = data["train"]["X"], data["train"]["Y"]
    losses, best = [], (np.inf, None)
    for step in range(budget):
        idx = rng.integers(0, X.shape[0], size=batch_size)
        loss, grads = loss_and_grads(model, (X[idx], Y[idx]))
        if not np.isfinite(loss):
            raise FloatingPointError("non-finite loss")
        clip_global_norm(grads, CLIP_NORM)
        opt.step(p, grads, sign=sign)
        losses.append(loss)
        if (step + 1) % eval_every == 0 or step == budget - 1:
            val = loss_and_grads(model, (data["val"]["X"], data["val"]["Y"]))[0]
            if np.isfinite(val) and val < best[0]:
                best = (val, {k: v.copy() for k, v in p.items()})
    if best[1] is not None:
        for k in p:
            p[k][...] = best[1][k]
    return np.array(losses)


def train(builder, seed, data, budget, mutant=None):
    model = builder(np.random.default_rng(seed))
    model["mutant"] = mutant
    losses = fit(model, data, budget, np.random.default_rng(seed + 7))
    return model, losses


# ----------------------------------------------------------------------------
# 11. Tests: correctness first, then hypotheses
# ----------------------------------------------------------------------------
def gradcheck(model, X, Y, rng, n_entries=20):
    grads = loss_and_grads(model, (X, Y))[1]
    return finite_difference_check(lambda: loss_and_grads(model, (X, Y))[0], model["params"], grads, rng, n_entries)


def random_engine(rng, kind="gradus", spread=1.5):
    """A gradus engine or rival with broad random parameters, for property tests."""
    m = build_model(rng=rng, kind=kind)
    key = "vir_pos" if kind == "gradus" else "ir_deg"
    m["params"][key] = rng.normal(0, spread, size=m["params"][key].shape)
    if kind == "gradus":
        m["params"]["vir_neg"] = rng.normal(0, spread, size=(K, A))
    m["params"]["cx_W2"] = rng.normal(0, spread, size=m["params"]["cx_W2"].shape)
    return m


def own_degrees(model):
    if model["kind"] == "rushd":
        return model["params"]["ir_deg"]
    return np.log2(softplus(model["params"]["vir_pos"]) / softplus(model["params"]["vir_neg"]))


def sparse_doses(rng, n, scale=3.0):
    d = rng.uniform(0, scale, size=(n, K)) * (rng.random((n, K)) < 0.4)
    d[np.arange(n), rng.integers(0, K, size=n)] += rng.uniform(0.01, scale, size=n)
    return d


def test_mediant(rng):
    """C6.1: without the body, a compound's degree lies between the least and the
    greatest own degree of the simples in it (the mediant inequality). Ibn Rushd's
    sum is the negative control and must violate it."""
    worst, control = 0.0, 0.0
    for _ in range(60):
        for kind in ("gradus", "rushd"):
            m = random_engine(rng, kind)
            d = sparse_doses(rng, 64)
            g = degrees(m, d, rng.standard_normal((64, P)), with_body=False)[0]
            own = own_degrees(m)
            mask = (d > 0)[:, :, None]
            lo = np.where(mask, own[None], np.inf).min(1)
            hi = np.where(mask, own[None], -np.inf).max(1)
            excess = float(np.maximum(lo - g, g - hi).max())
            worst, control = (max(worst, excess), control) if kind == "gradus" else (worst, max(control, excess))
    return worst < 1e-9 and control > 1e-3, f"max excursion {worst:.1e}; rival (sum) excursion {control:.2f}"


def test_proportion(rng):
    """C6.2: without the body, scaling every dose by the same factor leaves the
    degree unchanged; the rival's sum is the negative control."""
    worst, control = 0.0, 0.0
    for _ in range(40):
        for kind in ("gradus", "rushd"):
            m = random_engine(rng, kind)
            d, pt = sparse_doses(rng, 32), rng.standard_normal((32, P))
            lam = 10.0 ** rng.uniform(-2, 2, size=(32, 1))
            gap = float(np.abs(degrees(m, d * lam, pt, False)[0] - degrees(m, d, pt, False)[0]).max())
            worst, control = (max(worst, gap), control) if kind == "gradus" else (worst, max(control, gap))
    return worst < 1e-9 and control > 1e-3, f"max change {worst:.1e}; rival change {control:.2f}"


def test_antisymmetry(rng):
    """C6.3: exchanging the hot and cold parts of every simple and of the body
    negates the degree exactly; log2((hot+1)/cold) is the negative control."""
    worst, control = 0.0, 0.0
    for _ in range(40):
        m = random_engine(rng)
        d, pt = sparse_doses(rng, 32), rng.standard_normal((32, P))
        g = degrees(m, d, pt)[0]
        sw = knockout(m, "gradus", "identity")        # a copy; the knockout flag is cleared below
        sw["ko"] = {}
        sw["params"]["vir_pos"], sw["params"]["vir_neg"] = m["params"]["vir_neg"].copy(), m["params"]["vir_pos"].copy()
        w2, b2 = m["params"]["cx_W2"], m["params"]["cx_b2"]
        sw["params"]["cx_W2"] = np.concatenate([w2[:, A:], w2[:, :A]], axis=1)
        sw["params"]["cx_b2"] = np.concatenate([b2[A:], b2[:A]])
        g_sw, c = degrees(sw, d, pt)
        worst = max(worst, float(np.abs(g_sw + g).max()))
        broken = np.log2((c["hot"] + 1.0) / c["cold"])
        control = max(control, float(np.abs(broken + np.log2((c["cold"] + 1.0) / c["hot"])).max()))
    return worst < 1e-9 and control > 1e-3, f"max |g(swap)+g| {worst:.1e}; broken operator {control:.2f}"


def test_monotone_approach(rng):
    """C6.4: as a whole recipe is scaled from nothing to very large, the degree
    moves monotonically from the patient's own degree to the compound's own
    degree. A numerator growing with dose squared is the negative control."""
    lam = np.geomspace(1e-9, 1e9, 360)[:, None]      # wide enough to reach both asymptotes
    bad, flips, limit_err = 0, 0, 0.0
    for _ in range(200):
        m = random_engine(rng)
        d, pt = sparse_doses(rng, 1), rng.standard_normal((1, P))
        dd = d * lam
        g = degrees(m, dd, np.repeat(pt, len(lam), 0))[0]
        body = degrees(m, np.zeros_like(d), pt)[0][0]
        own = degrees(m, d, pt, with_body=False)[0][0]
        steps = np.diff(g, axis=0)
        bad += int(((steps > 1e-12).any(0) & (steps < -1e-12).any(0)).any())
        limit_err = max(limit_err, float(np.abs(g[0] - body).max()), float(np.abs(g[-1] - own).max()))
        c = degrees(m, dd, np.repeat(pt, len(lam), 0))[1]
        extra = (lam ** 2) * (d @ c["vp"])
        broken = np.diff(np.log2(extra + c["hot"]) - np.log2(c["cold"]), axis=0)
        flips += int(((broken > 1e-12).any(0) & (broken < -1e-12).any(0)).any())
    ok = bad == 0 and limit_err < 5e-3 and flips > 0
    return ok, f"non-monotone {bad}/200; limit error {limit_err:.1e}; broken operator non-monotone {flips}/200"


def test_determinism(data, budget):
    runs = [train(lambda r: build_model(rng=r), 3, data, budget) for _ in range(2)]
    same = np.array_equal(runs[0][1], runs[1][1]) and all(
        np.array_equal(runs[0][0]["params"][k], runs[1][0]["params"][k]) for k in runs[0][0]["params"])
    finite = all(np.all(np.isfinite(v)) for v in runs[0][0]["params"].values())
    finite = finite and np.all(np.isfinite(forward(runs[0][0], data["test"]["X"])[0]))
    return same and finite, f"identical losses and params across two runs={same}; finite={finite}"


def test_split_integrity(data):
    names = ("train", "val", "test", "shifted")
    hs = {k: row_hashes(data[k]["X"]) for k in names}
    overlap = sum(len(hs[a] & hs[b]) for i, a in enumerate(names) for b in names[i + 1:])
    return overlap == 0, f"{overlap} shared examples across {len(names)} splits"


def learning_ok(losses, test_rmse, trivial):
    th = MIND_CARD["thresholds"]
    drop = 1 - losses[-50:].mean() / losses[:50].mean()
    return drop >= th["loss_drop_fraction"] and trivial - test_rmse >= th["margin_over_trivial"], drop


def run_mutant(name, data, budget, rng):
    m, losses = train(lambda r: build_model(rng=r), 11, data, budget, mutant=name)
    X, Y = data["train"]["X"][:16], data["train"]["Y"][:16]
    gc = gradcheck(m, X, Y, rng, n_entries=6)
    ok, drop = learning_ok(losses, rmse(m, data["test"]), trivial_rmse(data["test"]))
    return (not gc["passed"]) or (not ok), f"gradcheck passed={gc['passed']}, loss drop {drop:.2f}"


def trivial_rmse(split):
    return float(np.sqrt(np.mean(split["Y"] ** 2)))   # predicting the training mean (zero after standardisation)


def evaluate_seed(seed, clean, blind, budget, mutant):
    full, lf = train(lambda r: build_model(rng=r), seed, clean, budget, mutant)
    base, _ = train(build_baseline, seed + 1000, clean, budget)
    rival, _ = train(build_rival, seed + 2000, clean, budget)
    full_b, _ = train(lambda r: build_model(rng=r), seed + 3000, blind, budget)
    base_b, _ = train(build_baseline, seed + 4000, blind, budget)
    cohort = planning_cohort(clean["world"], np.random.default_rng(seed + 5000))
    res = {"loss_drop": float(1 - lf[-50:].mean() / lf[:50].mean()), "losses": lf,
           "rmse_test": rmse(full, clean["test"]), "rmse_test_base": rmse(base, clean["test"]),
           "rmse_shift": rmse(full, clean["shifted"]), "rmse_shift_base": rmse(base, clean["shifted"]),
           "rmse_shift_rival": rmse(rival, clean["shifted"]), "rmse_test_rival": rmse(rival, clean["test"]),
           "rmse_blind": rmse(full_b, blind["test"]), "rmse_blind_base": rmse(base_b, blind["test"])}
    for tag, m in (("", full), ("_base", base), ("_rival", rival), ("_oracle", None)):
        res["plan" + tag], res["dose" + tag] = plan_success(m, clean, cohort, np.random.default_rng(seed + 6000))
    res["ko"] = {}
    for name, mode in (("gradus", "identity"), ("complexio", "mean"), ("virtutes", "mean"),
                       ("proprietas", "zero"), ("signa_hidden", "identity")):
        res["ko"][name] = rmse(knockout(full, name, mode), clean["shifted"]) - res["rmse_shift"]
    res["model"] = full
    return res


# ----------------------------------------------------------------------------
# 12. Report
# ----------------------------------------------------------------------------
def correctness_tests(args, clean, rng, budget):
    out = [{"id": "C7", "name": "split_integrity", "passed": test_split_integrity(clean)[0],
            "detail": test_split_integrity(clean)[1]}]
    for cid, fn in (("C6.1", test_mediant), ("C6.2", test_proportion), ("C6.3", test_antisymmetry),
                    ("C6.4", test_monotone_approach)):
        ok, detail = fn(rng)
        out.append({"id": cid, "name": fn.__name__, "passed": ok, "detail": detail})
    ok, detail = test_determinism(clean, 60)
    out.append({"id": "C2", "name": "determinism_finiteness", "passed": ok, "detail": detail})
    sh = dict(clean)
    for split in ("train", "val"):
        sh[split] = dict(clean[split])
        sh[split]["Y"] = clean[split]["Y"][np.random.default_rng(args.seed + 31).permutation(len(clean[split]["Y"]))]
    m_sh, _ = train(lambda r: build_model(rng=r), args.seed, sh, budget)
    sh_rmse, triv = rmse(m_sh, clean["test"]), trivial_rmse(clean["test"])
    band = MIND_CARD["thresholds"]["shuffle_band"]
    out.append({"id": "C4", "name": "shuffled_label_control", "passed": sh_rmse >= triv - band,
                "detail": f"held-out rmse {sh_rmse:.3f} vs trivial {triv:.3f} (must stay within {band})"})
    names = [args.mutant] if args.mutant else list(MUTANTS)
    caught = []
    for name in names:
        hit, detail = run_mutant(name, clean, max(300, budget // 3), rng)
        caught.append((name, hit, detail))
    out.append({"id": "C5", "name": "mutant_detection", "passed": all(h for _, h, _ in caught),
                "detail": "; ".join(f"{n}: {'caught' if h else 'MISSED'} ({d})" for n, h, d in caught)})
    hits = int(sum(bool(h) for _, h, _ in caught))
    return out, {"detected": hits, "total": len(caught), "score": hits / len(caught)}


def gradient_checks(args, clean, rng, trained):
    X, Y = clean["train"]["X"][:16], clean["train"]["Y"][:16]
    checks = []
    for kind in ("gradus", "deepsets", "rushd"):
        m = build_model(rng=np.random.default_rng(args.seed + 41), kind=kind)
        m["mutant"] = args.mutant
        checks.append(gradcheck(m, X, Y, rng))
    cls = build_model(rng=np.random.default_rng(args.seed + 42), task_type="vector_classification")
    checks.append(gradcheck(cls, X, np.argmax(Y, axis=1).astype(float), rng))
    checks.append(gradcheck(trained, X, Y, rng))
    main = [checks[0], checks[-1]]
    return {"tensors_checked": min(c["tensors_checked"] for c in main), "tensors_total": main[0]["tensors_total"],
            "max_rel_error": max(c["max_rel_error"] for c in checks), "checked_at": ["init", "after_training_steps"],
            "passed": all(c["passed"] for c in checks),
            "detail": "gradus init/after, deepsets, rushd, classification head"}


def hypothesis_table(per_seed, quick, rng):
    diffs = {"H-SIG": [r["rmse_shift"] - r["rmse_shift_base"] for r in per_seed],
             "H-NEC": [r["ko"]["gradus"] - r["ko"]["signa_hidden"] for r in per_seed],
             "H-NEC-B": [r["ko"]["complexio"] - r["ko"]["signa_hidden"] for r in per_seed],
             "H-BLIND": [(r["rmse_blind"] - r["rmse_test"]) - (r["rmse_blind_base"] - r["rmse_test_base"]) for r in per_seed],
             "H-RIVAL": [r["rmse_shift"] - r["rmse_shift_rival"] for r in per_seed],
             "H-PLAN": [r["plan"] - r["plan_base"] for r in per_seed]}
    out = []
    for h in MIND_CARD["hypotheses"]:
        x = diffs[h["id"]]
        if quick:
            mean, ci, v = float(np.mean(x)), [float(np.mean(x))] * 2, "not evaluated"
        else:
            mean, ci = paired_bootstrap_ci(x, rng)
            v = verdict(mean, ci, h["direction"], h["mesi"])
        out.append({"id": h["id"], "metric": h["metric"], "mean_diff": mean, "ci95": ci, "mesi": h["mesi"],
                    "n_seeds": len(x), "verdict": v})
    return out


def run_protocol(args):
    t0 = time.time()
    quick = args.quick
    seeds = [args.seed + i for i in range(1 if quick else args.seeds)]
    budget = 900 if quick else 2400
    limit = BUDGET_QUICK_S if quick else BUDGET_FULL_S
    print(f"chapter 0344 · seeds {seeds} · budget {budget} updates · {'quick' if quick else 'full'} protocol"
          + (f" · mutant {args.mutant}" if args.mutant else ""))
    root = np.random.default_rng(args.seed + 99)
    clean = make_data(root)
    blind = make_data(np.random.default_rng(args.seed + 99), blind=True)
    rng = np.random.default_rng(args.seed + 5)
    correctness, mutants = correctness_tests(args, clean, rng, budget if not quick else 600)
    per_seed = []
    for seed in seeds:
        r = evaluate_seed(seed, clean, blind, budget, args.mutant)
        per_seed.append(r)
        print(f"  seed {seed}: drop {r['loss_drop']:.2f} | test {r['rmse_test']:.3f} (DS {r['rmse_test_base']:.3f})"
              f" | shifted {r['rmse_shift']:.3f} (DS {r['rmse_shift_base']:.3f}, Rushd {r['rmse_shift_rival']:.3f})"
              f" | blind {r['rmse_blind']:.3f} (DS {r['rmse_blind_base']:.3f})"
              f" | plan {r['plan']:.2f} (DS {r['plan_base']:.2f}, Rushd {r['plan_rival']:.2f}, oracle {r['plan_oracle']:.2f})")
    gc = gradient_checks(args, clean, rng, per_seed[0]["model"])
    correctness.insert(0, {"id": "C1", "name": "gradient_check", "passed": gc["passed"],
                           "detail": f"max rel error {gc['max_rel_error']:.2e} ({gc.pop('detail')})"})
    triv = trivial_rmse(clean["test"])
    oks = [learning_ok(r["losses"], r["rmse_test"], triv) for r in per_seed]
    correctness.append({"id": "C3", "name": "learning", "passed": all(o for o, _ in oks),
                        "detail": f"loss drop min {min(d for _, d in oks):.2f}; test rmse max "
                                  f"{max(r['rmse_test'] for r in per_seed):.3f} vs trivial {triv:.3f}"})
    boot = np.random.default_rng(args.seed + 77)
    hyps = hypothesis_table(per_seed, quick, boot)
    mods = modules(per_seed[0]["model"])
    knock = []
    for name in ("gradus", "complexio", "virtutes", "proprietas", "signa_hidden"):
        x = [r["ko"][name] for r in per_seed]
        mean, ci = (float(np.mean(x)), [float(np.mean(x))] * 2) if quick else paired_bootstrap_ci(x, boot)
        knock.append({"module": name, "signature": mods[name]["signature"], "metric_change": mean, "ci95": ci})
    runtime = time.time() - t0
    correctness.append({"id": "C8", "name": "budget", "passed": runtime <= limit, "detail": f"{runtime:.1f}s of {limit:.0f}s"})
    for c in correctness:
        c["passed"] = bool(c["passed"])              # NumPy booleans are not JSON-serialisable
    finite = all(np.all(np.isfinite(v)) for r in per_seed for v in r["model"]["params"].values())
    code = 4 if not finite else 3 if runtime > limit else 0 if all(c["passed"] for c in correctness) else 1
    report = {"schema_version": "1.0", "chapter": 344, "file": os.path.basename(__file__),
              "card_revision": MIND_CARD["card_revision"],
              "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
              "seeds": seeds, "runtime_s": runtime, "n_params": n_params(per_seed[0]["model"]),
              "n_params_baseline": n_params(build_baseline(np.random.default_rng(0))),
              "n_params_rival": n_params(build_rival(np.random.default_rng(0))),
              "gradcheck": gc, "correctness": correctness, "mutants": mutants, "hypotheses": hyps,
              "knockouts": knock, "task_types": TASK_TYPES, "exit_code": code,
              "per_seed": [{k: v for k, v in r.items() if k not in ("model", "losses", "ko")} | {"ko": r["ko"]}
                           for r in per_seed]}
    write_report(report, args.json)
    return code, per_seed


def real_data_bridge(path, args):
    """Optional: a local CSV whose first 12 columns are doses, next 6 context
    features and last 3 responses (header row required). Trains the gradus
    engine and Deep Sets on 80% and reports held-out RMSE. No downloads."""
    if not os.path.exists(path):
        print(f"--data: {path} not found; real-data bridge skipped")
        return
    raw = np.genfromtxt(path, delimiter=",", skip_header=1)
    if raw.ndim != 2 or raw.shape[1] < K + P + S or raw.shape[0] < 50:
        print("--data: need >= 50 rows and 21 numeric columns (12 doses, 6 features, 3 responses); skipped")
        return
    raw = raw[np.all(np.isfinite(raw), axis=1)]
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(raw))
    cut, cut2 = int(0.7 * len(raw)), int(0.8 * len(raw))
    parts = {"train": raw[idx[:cut]], "val": raw[idx[cut:cut2]], "test": raw[idx[cut2:]]}
    data = {k: {"X": v[:, :K + P], "Yraw": v[:, K + P:K + P + S]} for k, v in parts.items()}
    standardise(data)
    for label, builder in (("gradus", lambda r: build_model(rng=r)), ("deepsets", build_baseline)):
        m, _ = train(builder, args.seed, data, 1500)
        print(f"--data {os.path.basename(path)}: {label} held-out rmse {rmse(m, data['test']):.3f}"
              f" (trivial {trivial_rmse(data['test']):.3f})")


# ----------------------------------------------------------------------------
# 13. Command-line entry point
# ----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__),
                                 usage=f"python3 {os.path.basename(__file__)} [--quick] [--seed S] [--seeds N] "
                                       "[--json PATH] [--card] [--mutant NAME] [--data PATH]")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--card", action="store_true")
    ap.add_argument("--mutant", type=str, default=None)
    ap.add_argument("--data", type=str, default=None)
    try:
        args = ap.parse_args(argv)
    except SystemExit:
        return 2
    if args.card:
        print(json.dumps(MIND_CARD, indent=2))
        return 0
    if (args.mutant and args.mutant not in MUTANTS) or args.seeds < 1:
        print(f"invalid usage; registered mutants: {list(MUTANTS)}")
        return 2
    try:
        code, _ = run_protocol(args)
    except FloatingPointError as err:
        print(f"non-finite values: {err}")
        return 4
    if args.data:
        real_data_bridge(args.data, args)
    return code


if __name__ == "__main__":
    sys.exit(main())

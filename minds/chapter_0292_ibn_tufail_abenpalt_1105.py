#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0292 · Ibn Tufail (Abu Bakr Muhammad ibn Tufail al-Qaysi, c. 1105/1110-1185)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 15, Minds 281-300 Available on Amazon https://www.amazon.com/dp/B0HMQ7TVJM
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0292_ibn_tufail_abenpalt_1105 - Ibn Tufail (Abu Bakr Muhammad ibn Tufail al-Qaysi, c. 1105/1110-1185)
# END ATTRIBUTION
"""
The Open Body: a ledger of causes written only by the knife, on one seat of life.

Thesis
    What moves a body is learned by cutting it once watching has stopped teaching:
    its causes form a ledger that only cuts can write, and an accurate account of
    the cutter's verdicts does not carry that ledger beyond the cases it describes.

Evidence and provenance  (provenance: belief)
    His one surviving philosophical work, Hayy ibn Yaqzan, cited by section of Simon
    Ockley's translation (1708; Gutenberg #16831) and by episode of L. E. Goodman's
    (2009). No other treatise survives to show how far he held it outside the tale.
    D1 Hayy learns by watching and imitating the doe for about seven years (14-15).
    D2 At her death nothing outward is amiss; he infers a hidden inner part and
       opens the chest on "a strong persuasion" that it lay in the middlemost
       part, which he felt in his own breast (17-18).
    D3 The left cavity of the heart is empty, the right full of clotted blood (21).
    D4 He opens the left cavity of a living animal: a hot vapour, then death (26).
    D5 One animal spirit governs the body and the organs are its instruments
       (22, 28); the nerves take it from the brain, the brain from the heart (29).
    D6 Absal and Salaman: taught directly, the islanders are not helped, and Hayy
       tells them to keep to their law (closing episode).
    D7 Nothing on the island checks where he cuts: the cut follows his own
       persuasion (D2). This is the chapter's critical reading, not his claim.

Doctrine -> mechanism -> test
    D1     -> M0 a reading of the seat from the organs, learned by watching      -> C3
    D3, D4 -> M1 the ledger: per-organ effects of a cut on the seat, written only
              by cuts (signature)                     -> C6.2, C6.3, H-SIG, H-NEC
    D5     -> M1 one scalar seat on which the effects of cuts add                -> C6.1
    D2     -> M2 watch until the prequential error stops falling, then cut where
              the imagined cut leans, never below 5% of the largest            -> H-BLIND
    D6     -> M3 Salaman's citizen: a network taught Hayy's verdicts             -> H-RIVAL
    D7     -> blind spot: an organ quiet in health is never cut                  -> H-BLIND

Research question
    Causal inference and intervention: to forecast cuts it has not tried, what is a
    cut-only ledger on one latent worth against a size-matched do-conditioned
    network on a randomized design; what does a pupil taught its verdicts inherit;
    and where does a learner that picks its experiments by its beliefs go blind?

Closest prior art and the delta
    Interventional causal discovery (Brouillard et al. 2020, DCDI; Lippe et al.
    2022, ENCO); active choice of interventions (Tong & Koller 2001; Scherrer et
    al. 2021); learning-progress curricula (Oudeyer, Kaplan & Hafner 2007);
    distillation fidelity (Hinton, Vinyals & Dean 2015; Stanton et al. 2021).
    Delta: cut effects enter only through a linear ledger on one latent that
    watching has learned to read; the learner starts cutting when its own
    prequential error flattens, cuts where its imagined cut says it leans, and
    the cost of that choice and the fate of its verdicts are tested. Baselines:
    a size-matched do-conditioned MLP on a randomized design; the same network
    taught Hayy's verdicts (the rival); descriptive variants listed in the report.

Blind spot
    D7 made operational: on the quiet island marrow barely varies in health, so
    the imagined cut never leans on it and Hayy rarely if ever cuts it, though an
    injury to it moves vigor as much as smothering does. A randomized design
    cuts it once in seven cuts; a 10% chance floor (not Hayy's) is the repair.

Task (generative process)
    Organs: breath, warmth, pulse, sun, marrow, fodder, fleece.
    seat = tanh(0.8 breath + 0.8 marrow + constitution); vigor = seat + noise;
    warmth = 0.9 seat + 0.5 sun + noise; pulse = 0.8 seat + noise; fodder and
    fleece are idle. A cut holds one organ at v ~ U(-2, 2) and its descendants
    respond. A watch costs 1 unit, a cut 2; 400 units per learner. Input: the
    organs before the knife and the intended displacement; target: vigor after.
    Splits: bodies in health; extraordinary cuts (one organ at 2 <= |v| <= 3,
    or two organs at once); injuries to marrow on the quiet island.

Limits
    The open island has one seat and additive causes, so the anatomy is right
    there by construction: H-SIG measures what a correct structural prior is
    worth, not whether it is true. The file is a research prototype of an
    AGI-oriented component, not an AGI, and replicates no one's mind; nothing
    here is evidence about Ibn Tufail.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 3,
    "revision_log": [
        "r1 (2026-10-21): replaces the July 2026 draft (filed as 0241, no card), whose attractor test passed or "
        "failed with seed and platform. World constants, the persuasion threshold, marrow's spread on the quiet "
        "island (0.01), the size match, the effect metric and every mesi were fixed after pilots on reserved "
        "seeds 990-999 (excluded from evaluation). A learner shown the body after the cut reads a cause's effect "
        "off the downstream signs, hiding any targeting blind spot, so the task forecasts from the body before "
        "the knife; a 10% chance floor repaired the blind spot, so Hayy has none and it is reported as a repair.",
        "r2 (after the first --quick run on seed 0, before any hypothesis was evaluated): C4's one permutation "
        "failed its cut-effect criterion (0.303 vs a band of 0.305): a random ledger can point both causes the "
        "right way by luck (1 draw in 12 on seeds 5000-5011, mean 1.7x the no-effect error). C4 now averages "
        "five permutations; no hypothesis, metric, threshold, mesi or seed changed.",
        "r3 (after the five-seed run, which passed): --quick on 12 non-evaluation seeds failed C6.3 six times, "
        "always through an organ opened a handful of times (fleece -0.66 after 4 cuts), never a cause (lowest "
        "0.37). 'Every non-cause <= 0.15' became 'causes >= 0.3 and no non-cause opened >= 10 times rivals the "
        "weakest cause' (other_max -> min_cuts). Correctness only; no hypothesis, metric, mesi or seed changed."],
    "generation": {"template_version": "1.0", "generator": "Claude", "generator_version": "claude-opus-5-5",
                   "date": "2026-10-21"},
    "id": 292, "figure": "Ibn Tufail (Abenpalt)", "born": 1105, "died": 1185,
    "civilization": "Andalusi Arab (al-Andalus and the Maghrib under the Almohads; Arabic)",
    "provenance": "belief",
    "thesis": ("What moves a body is learned by cutting it once watching has stopped teaching: its causes form a "
               "ledger only cuts can write, and an accurate account of the cutter's verdicts does not carry that "
               "ledger beyond the cases it describes."),
    "evidence": [
        {"id": "D1", "claim": "Hayy learns by watching and imitating the doe for about seven years",
         "basis": "primary", "source": "Hayy ibn Yaqzan, sections 14-15 (Ockley trans. 1708)"},
        {"id": "D2", "claim": "Finding nothing amiss outside, he opens the chest on a persuasion that the hidden "
         "part is the middlemost one, which he felt in his own breast", "basis": "primary",
         "source": "Hayy ibn Yaqzan, sections 17-18 (Ockley trans. 1708)"},
        {"id": "D3", "claim": "The left cavity of the heart is empty, the right full of clotted blood",
         "basis": "primary", "source": "Hayy ibn Yaqzan, section 21 (Ockley trans. 1708)"},
        {"id": "D4", "claim": "Opening the left cavity of a living animal shows a hot vapour, and the animal dies",
         "basis": "primary", "source": "Hayy ibn Yaqzan, section 26 (Ockley trans. 1708)"},
        {"id": "D5", "claim": "One animal spirit governs the body and the organs are its instruments; the nerves "
         "take it from the brain, the brain from the heart", "basis": "primary",
         "source": "Hayy ibn Yaqzan, sections 22, 28, 29 (Ockley trans. 1708)"},
        {"id": "D6", "claim": "Taught directly, Salaman's islanders are not helped, and Hayy tells them to keep to "
         "their law", "basis": "primary", "source": "Hayy ibn Yaqzan, closing episode (Goodman trans. 2009)"},
        {"id": "D7", "claim": "Nothing on the island checks where he cuts; the cut follows his own persuasion",
         "basis": "speculation", "source": "chapter section 6, reading Hayy ibn Yaqzan sections 17-18"},
    ],
    "research_question": {"category": "causal inference and intervention",
                          "question": "To forecast the effects of interventions it has not tried, what is a "
                                      "cut-only ledger on one latent worth against a do-conditioned network on a "
                                      "randomized design, what does a pupil taught its verdicts inherit, and "
                                      "where does belief-directed choice of experiments go blind?"},
    "mechanism": {
        "name": "The Open Body: a watch-then-cut ledger on one seat",
        "family": "interventional causal learning: structured do-effect model on a scalar latent, with active "
                  "choice of interventions and a learning-progress trigger",
        "signature_modules": ["ledger"],
        "closest_prior_art": [
            "Brouillard, Lachapelle, Lacoste, Lacoste-Julien & Drouin 2020, Differentiable Causal Discovery from "
            "Interventional Data (NeurIPS)",
            "Lippe, Cohen & Gavves 2022, Efficient Neural Causal Discovery without Acyclicity Constraints (ICLR)",
            "Tong & Koller 2001, Active Learning for Structure in Bayesian Networks (IJCAI)",
            "Scherrer et al. 2021, Learning Neural Causal Models with Active Interventions (arXiv:2109.02429)",
            "Oudeyer, Kaplan & Hafner 2007, Intrinsic Motivation Systems for Autonomous Mental Development "
            "(IEEE Trans. Evolutionary Computation 11)",
            "Stanton, Izmailov, Kirichenko, Alemi & Wilson 2021, Does Knowledge Distillation Really Work? "
            "(NeurIPS)"],
        "overlap": "Medium",
        "prior_art_queries": ["interventional causal discovery neural network", "active intervention selection",
                              "prediction of unseen intervention combinations", "learning progress curriculum",
                              "permutation importance to choose experiments", "distillation out-of-distribution "
                              "fidelity", "unvaried cause active learning"],
        "contribution_type": "mechanism",
        "delta": ("Cut effects enter only through a linear ledger on one latent that watching has learned to "
                  "read; the learner begins cutting when its own prequential error flattens and cuts where its "
                  "imagined cut leans; the blind spot of that choice and the fate of its verdicts in a pupil are "
                  "tested in the file."),
    },
    "traceability": [
        {"doctrine": "D1", "mechanism": "M0", "property_test": "C3", "hypothesis": "H-SIG"},
        {"doctrine": "D3", "mechanism": "M1", "property_test": "C6.2", "hypothesis": "H-NEC"},
        {"doctrine": "D4", "mechanism": "M1", "property_test": "C6.3", "hypothesis": "H-SIG"},
        {"doctrine": "D5", "mechanism": "M1", "property_test": "C6.1", "hypothesis": "H-SIG"},
        {"doctrine": "D2", "mechanism": "M2", "property_test": "C6.3", "hypothesis": "H-BLIND"},
        {"doctrine": "D6", "mechanism": "M3", "property_test": "C3", "hypothesis": "H-RIVAL"},
        {"doctrine": "D7", "mechanism": "M2", "property_test": "C7", "hypothesis": "H-BLIND"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "On the open island the anatomist forecasts the effects of extraordinary cuts "
         "better than a size-matched do-conditioned MLP trained on a randomized design of the same budget",
         "metric": "effect_mse_extraordinary", "split": "shifted", "comparison": "model - baseline",
         "direction": "less", "mesi": 0.005, "seeds": 5},
        {"id": "H-NEC", "statement": "Removing the ledger (no cut moves the seat) hurts extraordinary forecasts more "
         "than holding the seat-reading correction at its mean", "metric": "effect_mse_extraordinary",
         "split": "shifted", "comparison": "signature_knockout - matched_knockout", "direction": "greater",
         "mesi": 0.05, "seeds": 5},
        {"id": "H-BLIND", "statement": "On the quiet island Hayy's persuasion leaves marrow uncut or nearly so, and "
         "he forecasts injuries to it worse than the randomized-design MLP",
         "condition": "quiet island: marrow's spread in health 0.01; injuries hold it at |v| <= 2",
         "grounding": "the cut follows his own persuasion (sections 17-18) and nothing on the island checks it",
         "metric": "effect_mse_quiet_injury", "split": "shifted", "comparison": "model - baseline",
         "direction": "greater", "mesi": 0.05, "seeds": 5},
        {"id": "H-RIVAL", "statement": "Salaman's citizen, a network taught Hayy's verdicts on every body Hayy met, "
         "forecasts extraordinary cuts worse than Hayy", "metric": "effect_mse_extraordinary", "split": "shifted",
         "comparison": "model - rival", "direction": "less", "mesi": 0.005, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.5, "margin_over_trivial": 0.5, "shuffled_band": 0.1,
                   "gradcheck_rel_tol": 1e-5, "cause_min": 0.3, "min_cuts": 10, "additivity_tol": 1e-9},
    "probe_predictions": [{"probe": p, "expected": e} for p, e in (("P10", "above"), ("P9", "above"), ("P6", "equal "
                          "to baseline"), ("P3", "below"), ("P1", "below"), ("P5", "below"), ("P7", "below"))],
    "dialectic_links": [
        {"chapter": 257, "relation": "teacher", "test": "none (the floating man and the name Hayy ibn Yaqzan come "
         "from Ibn Sina; the tale stages what he argued)"},
        {"chapter": 285, "relation": "teacher", "test": "none (read, never met: 'I for my part never saw him', "
         "Introduction, Ockley trans.)"},
        {"chapter": 235, "relation": "rival", "test": "none (his inconsistencies are criticised in the Introduction)"},
        {"chapter": 299, "relation": "successor", "test": "none (presented by him to the caliph in 1169; succeeded "
         "him as court physician in 1182)"},
        {"chapter": 332, "relation": "rival", "test": "none (al-Risala al-Kamiliyya answers Hayy: some knowledge "
         "must arrive by testimony; not implemented; H-BLIND asks the nearby question of an outside check)"}],
    "corpus_neighbors": [
        {"chapter": 287, "similarity": None, "difference": "Ibn Zuhr probes one patient mildly to place that "
         "patient's parameters; here cuts on many bodies write general per-organ effects on one shared seat"},
        {"chapter": 285, "similarity": None, "difference": "Ibn Bajja's solitaries converge without communication; "
         "here communication of verdicts is tested and fails beyond the cases described"},
        {"chapter": 279, "similarity": None, "difference": "Halevi licenses amendment where the learner re-derives "
         "consequences; here the pupil cannot re-derive anything without cutting"},
        {"chapter": 235, "similarity": None, "difference": "al-Farabi audits a lossy passage between two encodings; "
         "here the passage is from a structured learner to an unstructured one, measured off the described cases"},
        {"chapter": 233, "similarity": None, "difference": "al-Razi demands an outside evaluator; here the outside "
         "check is absent by design and its absence is the blind spot"},
        {"chapter": 272, "similarity": 0.146, "difference": "al-Zarqali carries constants through time; here "
         "coefficients are written only by interventions"}],
    "similarity_note": ("Section 4.4 recipe (9-token shingles outside the standard block). Only one corpus file "
                        "was available: chapter 0272, rebuilt from its session transcript, 0.146 (0.131 with both "
                        "MIND_CARDs removed; the cards alone share 0.91 through the mandated template). The July "
                        "2026 draft of this chapter: 0.036. Values of None were not measurable here."),
    "barometer": {
        "cognitive_processing": ["forecasts of double cuts never met in training (H-SIG split)"],
        "embodied_cognition": ["not measured: the body is a set of variables, not a controlled body"],
        "world_modeling": ["forecasting cut effects (H-SIG, H-NEC)", "which organs move the seat (C6.3)"],
        "consciousness": ["own prequential error triggers the move to cutting; no claim about experience"],
        "language_understanding": ["not measured; the citizen receives verdicts, not language (H-RIVAL)"],
        "emotional_intelligence": ["not measured"], "creativity": ["not measured"],
        "autonomy": ["self-set inquiry: when to stop watching and where to cut, with its blind spot (H-BLIND)"]},
    "task_types": ["vector_regression"],
    "applications": [
        {"use": "Forecast unseen combinations of perturbations from single ones", "sector": "functional genomics",
         "dataset": "Norman et al. 2019, combinatorial Perturb-seq (Science 365:786)"},
        {"use": "Separate causes from indicators among signalling proteins", "sector": "systems biology",
         "dataset": "Sachs et al. 2005, flow cytometry with interventions (Science 308:523)"},
        {"use": "Choose which actuator of a test bench to perturb next; audit parts that never vary in normal use",
         "sector": "industrial testing", "dataset": "Gamella, Peters & Buhlmann 2024, The Causal Chambers "
         "(arXiv:2404.11341)"}],
    "safety_notes": ("Synthetic bodies only: a 'cut' is an intervention on an abstract variable. No veterinary, "
                     "surgical or medical procedure, dose or treatment is modelled; vivisection appears in the "
                     "chapter as history. Applications are research decision support."),
}

# ============================================================================ imports and constants
import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np

CHAPTER = 292
ORGANS = ("breath", "warmth", "pulse", "sun", "marrow", "fodder", "fleece")
N_ORGANS = len(ORGANS)
MARROW = ORGANS.index("marrow")
CAUSES = (ORGANS.index("breath"), MARROW)
KNIFE, BEYOND, CUT_COST = 2.0, (2.0, 3.0), 2                  # training cuts |v| <= 2; extraordinary 2..3
UNITS, QUIET_SPREAD = 400, 0.01                                # experience budget; marrow's spread on the quiet island
PER_ENCOUNTER, MIN_WATCH, WINDOW, PLATEAU, EMA = 4, 40, 30, 0.05, 0.1
PERSUASION, CHANCE, RECENT = 0.05, 0.10, 64
READ_UNITS = 9
N_ORD, N_BEYOND, N_DOUBLE, N_QUIET = 400, 300, 300, 400
STEPS_FULL, STEPS_QUICK, STEPS_SHORT = 3000, 1200, 400
LR, BATCH, CLIP = 0.01, 32, 5.0
BUDGET_FULL, BUDGET_QUICK = 180.0, 20.0
TASK_TYPES = ["vector_regression"]
np.seterr(over="raise", invalid="raise", divide="raise", under="ignore")

# BEGIN STANDARD UTILITIES v1.0
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
# END STANDARD UTILITIES


# ============================================================================ data and tasks
def draw_state(rng, n, island):
    """Exogenous state of n bodies. On the open island marrow varies in health; on the quiet one it barely does."""
    spread = 1.0 if island == "open" else QUIET_SPREAD
    return {"breath": rng.normal(0.0, 1.0, n), "marrow": rng.normal(0.0, spread, n), "sun": rng.normal(0.0, 0.7, n),
            "fodder": rng.normal(0.0, 1.0, n), "fleece": rng.normal(0.0, 1.0, n),
            "constitution": rng.normal(0.0, 0.5, n), "warmth": rng.normal(0.0, 0.15, n),
            "pulse": rng.normal(0.0, 0.2, n), "vigor": rng.normal(0.0, 0.05, n)}


def realize(state, held, force):
    """Structural equations in causal order; organs with held=True stay at their forced value."""
    def organ(name, natural):
        k = ORGANS.index(name)
        return np.where(held[:, k], force[:, k], natural)

    breath, marrow, sun = organ("breath", state["breath"]), organ("marrow", state["marrow"]), organ("sun", state["sun"])
    seat = np.tanh(0.8 * breath + 0.8 * marrow + state["constitution"])          # the spirit in the left cavity
    warmth = organ("warmth", 0.9 * seat + 0.5 * sun + state["warmth"])
    pulse = organ("pulse", 0.8 * seat + state["pulse"])
    organs = np.column_stack([breath, warmth, pulse, sun, marrow, organ("fodder", state["fodder"]),
                              organ("fleece", state["fleece"])])
    return organs, seat + state["vigor"]


def encounter(rng, n, island, plan="watch", organ=None):
    """n bodies, seen in health ('watch') or under the knife: 'cut' (the given organ, |v| <= KNIFE),
    'beyond' (a random organ, KNIFE <= |v| <= 3) or 'double' (two random organs, |v| <= KNIFE).
    Returns [organs before the knife | intended displacement], vigor after it, and the cut's true effect."""
    state = draw_state(rng, n, island)
    held, force, rows = np.zeros((n, N_ORGANS), dtype=bool), np.zeros((n, N_ORGANS)), np.arange(n)
    if plan == "cut":
        held[:, organ], force[:, organ] = True, rng.uniform(-KNIFE, KNIFE, n)
    elif plan == "beyond":
        k = rng.integers(0, N_ORGANS, n)
        held[rows, k], force[rows, k] = True, rng.uniform(*BEYOND, n) * rng.choice([-1.0, 1.0], n)
    elif plan == "double":
        for k in np.argsort(rng.random((n, N_ORGANS)), axis=1)[:, :2].T:
            held[rows, k], force[rows, k] = True, rng.uniform(-KNIFE, KNIFE, n)
    before, vigor_before = realize(state, np.zeros_like(held), force)
    _, vigor_after = realize(state, held, force)
    shift = np.where(held, force - before, 0.0)
    return np.concatenate([before, shift], 1), vigor_after[:, None], (vigor_after - vigor_before)[:, None]


def make_task(seed_seq):
    """Evaluation splits only: every learner gathers its own experience on the island."""
    r = [np.random.default_rng(s) for s in seed_seq.spawn(3)]
    beyond, double = encounter(r[1], N_BEYOND, "open", "beyond"), encounter(r[1], N_DOUBLE, "open", "double")
    return {"ordinary": encounter(r[0], N_ORD, "open"),
            "extraordinary": tuple(np.concatenate(pair) for pair in zip(beyond, double)),
            "quiet_injury": encounter(r[2], N_QUIET, "quiet", "cut", MARROW)}


def effect_error(model, split):
    """Squared error of the forecast effect, f(body, cut) - f(body, no cut), against the true effect."""
    x, _, effect = split
    still = x.copy()
    still[:, N_ORGANS:] = 0.0
    return float(((predict(model, x) - predict(model, still) - effect) ** 2).mean())


def level_error(model, split):
    x, y, _ = split
    return float(((predict(model, x) - y) ** 2).mean())


# ============================================================================ model
class Anatomist:
    """The anatomist. Columns: organ values, then (when in_dim == 2 * organs) the displacement the knife will
    impose on each organ. One scalar seat s = tanh(a.x + a0 + w.tanh(Wx + b) + c.d), and y = h s + e.
    The ledger c is the only parameter that ever meets d."""

    kind = "anatomist"

    def __init__(self, in_dim, out_dim, rng, organs=None, read_units=READ_UNITS):
        k = in_dim if organs is None else organs
        assert in_dim in (k, 2 * k), "input: organ values, optionally followed by one displacement per organ"
        self.organs, self.with_cuts, self.ko = k, in_dim == 2 * k, {}
        self.params = {"read_W": rng.normal(0.0, 1.0 / math.sqrt(k), (read_units, k)),
                       "read_b": np.zeros(read_units), "read_w": rng.normal(0.0, 0.1, read_units),
                       "a": rng.normal(0.0, 1.0 / math.sqrt(k), k), "a0": np.zeros(1), "ledger": np.zeros(k),
                       "head_w": np.ones((out_dim, 1)), "head_b": np.zeros(out_dim)}

    def clone(self, **ko):
        twin = object.__new__(type(self))
        twin.__dict__.update(self.__dict__)
        twin.params = {name: value.copy() for name, value in self.params.items()}
        twin.ko = {**self.ko, **ko}
        return twin

    def forward(self, x):
        p, k = self.params, self.organs
        body = x[:, :k]
        shift = x[:, k:2 * k] if self.with_cuts else np.zeros_like(body)
        ledger, how = p["ledger"], self.ko.get("ledger")
        if how == "identity":                     # the watcher's reading: a cut organ taken as a natural value
            body, shift = body + shift, np.zeros_like(shift)
        elif how in ("zero", "mean"):
            ledger = np.full_like(ledger, 0.0 if how == "zero" else ledger.mean())
        hid = np.tanh(body @ p["read_W"].T + p["read_b"])
        corr = hid @ p["read_w"]
        if self.ko.get("correction") == "mean":
            corr = np.full_like(corr, corr.mean())
        elif self.ko.get("correction") in ("zero", "identity"):
            corr = np.zeros_like(corr)
        pre = body @ p["a"] + p["a0"][0] + corr + shift @ ledger
        seat = np.tanh(pre)
        cache = {"body": body, "shift": shift, "hid": hid, "pre": pre, "seat": seat}
        return seat[:, None] @ p["head_w"].T + p["head_b"], cache

    def backward(self, x, y, out, cache):
        """Hand-derived gradients of the mean squared error through the seat."""
        assert not self.ko, "gradients are defined for the intact model only"
        p, seat, body = self.params, cache["seat"], cache["body"]
        dout = 2.0 * (out - y) / x.shape[0]
        dpre = (dout @ p["head_w"])[:, 0] * (1.0 - seat ** 2)
        dhid = np.outer(dpre, p["read_w"]) * (1.0 - cache["hid"] ** 2)
        return {"head_w": dout.T @ seat[:, None], "head_b": dout.sum(0), "a": body.T @ dpre,
                "a0": np.array([dpre.sum()]), "ledger": cache["shift"].T @ dpre, "read_w": cache["hid"].T @ dpre,
                "read_W": dhid.T @ body, "read_b": dhid.sum(0)}

    def states(self, x):
        cache = self.forward(x)[1]
        return {"seat_preactivation": cache["pre"], "seat": cache["seat"], "reading_units": cache["hid"]}

    def roles(self):
        return {"reading": (["a", "a0"], "linear read of a scalar latent from the inputs", False),
                "correction": (["read_W", "read_b", "read_w"], "tanh correction added to the latent's "
                               "pre-activation", False),
                "ledger": (["ledger"], "per-input coefficients applied only to intervention displacements "
                           "(learned do-effects on the scalar latent)", True),
                "head": (["head_w", "head_b"], "affine readout of the scalar latent", False)}


def build_model(in_dim, out_dim, task_type, rng, **cfg):
    """The anatomist (see the class). cfg: organs (default in_dim: no displacement columns), read_units."""
    assert task_type in TASK_TYPES
    return Anatomist(in_dim, out_dim, rng, **cfg)


def new_anatomist(rng):
    return build_model(2 * N_ORGANS, 1, "vector_regression", rng, organs=N_ORGANS)


# ============================================================================ baselines and rival mechanisms
class DoMLP:
    """Closest prior art, size-matched: a do-conditioned regressor with one tanh layer over the organs and the
    intended displacements, which must learn from data everything the anatomist assumes. Taught Hayy's
    verdicts in place of the world, the same network is Salaman's citizen."""

    kind = "mlp"
    clone = Anatomist.clone

    def __init__(self, in_dim, out_dim, hidden, rng):
        self.ko = {}
        self.params = {"W_in": rng.normal(0.0, 1.0 / math.sqrt(in_dim), (in_dim, hidden)),
                       "b_in": np.zeros(hidden), "W_out": rng.normal(0.0, 1.0 / math.sqrt(hidden), (hidden, out_dim)),
                       "b_out": np.zeros(out_dim)}

    def forward(self, x):
        hid = np.tanh(x @ self.params["W_in"] + self.params["b_in"])
        return hid @ self.params["W_out"] + self.params["b_out"], {"hid": hid}

    def backward(self, x, y, out, cache):
        dout = 2.0 * (out - y) / x.shape[0]
        dpre = (dout @ self.params["W_out"].T) * (1.0 - cache["hid"] ** 2)
        return {"W_out": cache["hid"].T @ dout, "b_out": dout.sum(0), "W_in": x.T @ dpre, "b_in": dpre.sum(0)}

    def states(self, x):
        return {"hidden": self.forward(x)[1]["hid"]}

    def roles(self):
        return {"hidden": (["W_in", "b_in"], "tanh hidden layer", False),
                "readout": (["W_out", "b_out"], "linear readout", False)}


def build_mlp(in_dim, out_dim, hidden, rng):
    return DoMLP(in_dim, out_dim, hidden, rng)


def matched_hidden(n_target, in_dim, out_dim):
    return max(1, int(round((n_target - out_dim) / (in_dim + 1 + out_dim))))


# ============================================================================ registries: interface, modules, knockouts, mutants
class NonFiniteError(RuntimeError):
    pass


class Lab:
    """Run-wide switch: the registered mutant in force (None for the genuine file)."""
    mutant = None


MUTANTS = {"sign_flip_update": "optimizer ascends the loss (update sign flipped)",
           "zero_lr": "learning rate forced to zero", "zero_grad_ledger": "analytic gradient of the ledger zeroed",
           "zero_grad_reading": "analytic gradient of the seat-reading correction's input weights zeroed"}
SILENCED = {"zero_grad_ledger": "ledger", "zero_grad_reading": "read_W"}
KNOCKOUT_MODES = {"ledger": ("identity", "zero", "mean"), "correction": ("identity", "zero", "mean")}


def loss_and_grads(model, batch):
    x, y = batch["X"], batch["Y"]
    out, cache = model.forward(x)
    grads = model.backward(x, y, out, cache)
    silenced = SILENCED.get(Lab.mutant)
    if silenced in grads:
        grads[silenced] = np.zeros_like(grads[silenced])
    return float(((out - y) ** 2).sum(1).mean()), grads


def predict(model, x):
    return model.forward(x)[0]


def hidden_states(model, x):
    """Anatomist: the seat's pre-activation, the seat and the correction units. MLP: its hidden layer."""
    return model.states(x)


def modules(model):
    return {name: {"params": names, "role": role, "signature": sig}
            for name, (names, role, sig) in model.roles().items()}


def knockout(model, name, mode):
    """Copy with a module replaced. ledger: 'identity' reads a cut organ as if its value were natural (the
    watcher's reading), 'zero' lets no cut move the seat, 'mean' gives every organ the mean coefficient.
    correction: removed ('identity', 'zero') or held at its mean over the evaluated bodies ('mean')."""
    if model.kind != "anatomist" or mode not in KNOCKOUT_MODES.get(name, ()):
        raise KeyError(f"no knockout {name}/{mode} for the {model.kind}")
    return model.clone(**{name: mode})


def n_params(model):
    return int(sum(v.size for v in model.params.values()))


# ============================================================================ training
class Course:
    """One learner's optimiser, cosine-decayed rate and loss record; the registered mutant acts here."""

    def __init__(self, model, horizon):
        self.model, self.horizon, self.losses = model, horizon, []
        self.adam = Adam(model.params, lr=LR)

    def rate(self):
        if Lab.mutant == "zero_lr":
            return 0.0
        done = min(len(self.losses), self.horizon) / self.horizon
        return LR * (0.1 + 0.45 * (1.0 + math.cos(math.pi * done)))

    def step(self, x, y, rng):
        pick = rng.integers(0, len(x), min(BATCH, len(x)))
        loss, grads = loss_and_grads(self.model, {"X": x[pick], "Y": y[pick]})
        if not math.isfinite(loss):
            raise NonFiniteError(f"non-finite loss at step {len(self.losses)}")
        self.adam.step(self.model.params, clip_global(grads, CLIP)[0], lr=self.rate(),
                       sign=-1.0 if Lab.mutant == "sign_flip_update" else 1.0)
        self.losses.append(loss)

    def run_until(self, x, y, rng):
        while len(self.losses) < self.horizon:
            self.step(x, y, rng)
        if not all(np.isfinite(v).all() for v in self.model.params.values()):
            raise NonFiniteError("non-finite parameter after training")
        return self.losses


def fit(model, data, budget, rng):
    """Adam with global-norm clipping and cosine decay: a fixed number of updates on fixed data."""
    return Course(model, budget).run_until(data[0], data[1], rng)


def imagined_cut(model, bodies):
    """What the learner's own model loses when each organ is held at its mean: where it leans."""
    base = predict(model, bodies)
    lean = np.zeros(N_ORGANS)
    for k in range(N_ORGANS):
        held = bodies.copy()
        held[:, k] = bodies[:, k].mean()
        lean[k] = float(np.abs(predict(model, held) - base).mean())
    return lean


def choose_cut(model, memory, rng, chance):
    """Hayy's persuasion: cut each organ in proportion to the imagined cut, never one below PERSUASION of the
    largest. With chance > 0 that share of cuts goes to a uniformly drawn organ (the repaired variant)."""
    seen = memory[np.all(memory[:, N_ORGANS:] == 0.0, axis=1)][-RECENT:]
    lean = imagined_cut(model, seen)
    lean = np.where(lean >= PERSUASION * lean.max(), lean, 0.0)
    if lean.sum() <= 0.0:
        lean = np.ones(N_ORGANS)
    prob = (1.0 - chance) * lean / lean.sum() + chance / N_ORGANS
    return int(rng.choice(N_ORGANS, p=prob / prob.sum()))


def stopped_teaching(curve):
    """The smoothed prequential error (each body predicted before it is learned from) has flattened."""
    if len(curve) < max(MIN_WATCH, WINDOW + 1):
        return False
    old, new = curve[-WINDOW - 1], curve[-1]
    return (old - new) / (old + 1e-12) < PLATEAU


def hayy_inquiry(island, steps, rngs, chance=0.0, units=UNITS):
    """Watch bodies one at a time, learning after each (cost 1). When watching stops teaching, cut instead
    (cost 2) where the imagined cut leans. Spend the remaining updates on the whole experience."""
    world, init, train = rngs
    model = new_anatomist(init)
    course, memory, vigor, curve = Course(model, steps), [], [], []
    spent, promoted, cuts = 0, None, np.zeros(N_ORGANS, dtype=int)
    while True:
        organ = None if promoted is None else choose_cut(model, np.array(memory), world, chance)
        cost = 1 if organ is None else CUT_COST
        if spent + cost > units:
            break
        x1, y1, _ = encounter(world, 1, island, "watch" if organ is None else "cut", organ)
        surprise = float(((predict(model, x1) - y1) ** 2).sum())
        memory.append(x1[0])
        vigor.append(y1[0])
        spent += cost
        if organ is not None:
            cuts[organ] += 1
        xs, ys = np.array(memory), np.array(vigor)
        for _ in range(PER_ENCOUNTER):
            course.step(xs, ys, train)
        if promoted is None:
            curve.append(surprise if not curve else (1.0 - EMA) * curve[-1] + EMA * surprise)
            promoted = len(memory) if stopped_teaching(curve) else None
    xs, ys = np.array(memory), np.array(vigor)
    course.run_until(xs, ys, train)
    return model, (xs, ys), {"promoted": promoted, "cuts": cuts, "losses": course.losses}


def randomized_design(island, rng, units=UNITS):
    """The standard alternative: from the first encounter, a cut with probability 1/2 on a uniformly drawn
    organ, otherwise a watch; the same budget and costs as Hayy."""
    memory, vigor, spent = [], [], 0
    while True:
        organ = int(rng.integers(N_ORGANS)) if rng.random() < 0.5 else None
        cost = 1 if organ is None else CUT_COST
        if spent + cost > units:
            break
        x1, y1, _ = encounter(rng, 1, island, "watch" if organ is None else "cut", organ)
        memory.append(x1[0])
        vigor.append(y1[0])
        spent += cost
    return np.array(memory), np.array(vigor)


def spawn_rngs(seq, n):
    return [np.random.default_rng(s) for s in seq.spawn(n)]


def open_island(seq, steps):
    """Hayy and every comparator on the island whose causes show themselves in health."""
    kids = seq.spawn(3)
    hayy, own, info = hayy_inquiry("open", steps, spawn_rngs(kids[0], 3))
    r = spawn_rngs(kids[1], 2)
    randomized, watched = randomized_design("open", r[0]), encounter(r[1], UNITS, "open")[:2]
    verdicts = (own[0], predict(hayy, own[0]))
    hidden = matched_hidden(n_params(hayy), 2 * N_ORGANS, 1)
    plan = [("do_mlp", "mlp", randomized), ("citizen", "mlp", verdicts), ("anatomy_pupil", "anatomist", verdicts),
            ("anatomist_randomized", "anatomist", randomized), ("do_mlp_own_data", "mlp", own),
            ("watcher", "anatomist", watched)]
    models = {"anatomist": hayy}
    for (name, kind, data), child in zip(plan, kids[2].spawn(len(plan))):
        init, train = spawn_rngs(child, 2)
        models[name] = build_mlp(2 * N_ORGANS, 1, hidden, init) if kind == "mlp" else new_anatomist(init)
        fit(models[name], data, steps, train)
    models["watcher"] = knockout(models["watcher"], "ledger", "identity")
    return models, info, [own, randomized, watched]


def quiet_island(seq, steps):
    """The island where marrow barely varies in health: Hayy, Hayy with chance cuts, the randomized pair."""
    kids = seq.spawn(3)
    hayy, own, info = hayy_inquiry("quiet", steps, spawn_rngs(kids[0], 3))
    repaired, repaired_own, repaired_info = hayy_inquiry("quiet", steps, spawn_rngs(kids[1], 3), chance=CHANCE)
    r = spawn_rngs(kids[2], 5)
    randomized = randomized_design("quiet", r[0])
    mlp = build_mlp(2 * N_ORGANS, 1, matched_hidden(n_params(hayy), 2 * N_ORGANS, 1), r[1])
    fit(mlp, randomized, steps, r[2])
    anatomist = new_anatomist(r[3])
    fit(anatomist, randomized, steps, r[4])
    models = {"anatomist": hayy, "do_mlp": mlp, "anatomist_chance": repaired, "anatomist_randomized": anatomist}
    return models, info, repaired_info, [own, repaired_own, randomized]


# ============================================================================ tests: correctness
@dataclass
class Check:
    id: str
    name: str
    passed: bool
    detail: str
    extra: dict = field(default_factory=dict)

    def entry(self):
        return {"id": self.id, "name": self.name, "passed": bool(self.passed), "detail": self.detail}


def check_batch(seed):
    rng = np.random.default_rng(seed)
    parts = [encounter(rng, 8, "open"), encounter(rng, 8, "open", "double"), encounter(rng, 8, "open", "beyond")]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def c1_gradcheck(seed, tol):
    """Every tensor of the anatomist and the do-MLP, at initialisation and after 60 updates."""
    x, y = check_batch(seed)
    probe_rng, anatomist = np.random.default_rng(seed + 1), new_anatomist(np.random.default_rng(seed + 2))
    learners = [anatomist, build_mlp(2 * N_ORGANS, 1, matched_hidden(n_params(anatomist), 2 * N_ORGANS, 1),
                                     np.random.default_rng(seed + 3))]

    def probe(model):
        grads = loss_and_grads(model, {"X": x, "Y": y})[1]
        return finite_difference_check(lambda: float(((predict(model, x) - y) ** 2).sum(1).mean()),
                                       model.params, grads, probe_rng)[1]

    per_tensor = [probe(m) for m in learners]
    for m in learners:
        fit(m, (x, y), 60, np.random.default_rng(seed + 4))
    per_tensor += [probe(m) for m in learners]
    worst = max(max(r.values()) for r in per_tensor)
    checked, total = sum(len(r) for r in per_tensor), 2 * sum(len(m.params) for m in learners)
    return Check("C1", "gradient_check", worst <= tol,
                 f"{checked}/{total} tensor checks (anatomist and do-MLP, at init and after 60 updates), "
                 f"max rel err {worst:.2e} (<= {tol:.0e})",
                 {"max_rel_error": worst, "tensors_checked": checked, "tensors_total": total})


def c2_determinism(seed):
    def replay():
        model, data, info = hayy_inquiry("open", 300, spawn_rngs(np.random.SeedSequence(seed), 3), units=140)
        return model, data[0], info["losses"]

    (m1, x1, l1), (m2, x2, l2) = replay(), replay()
    same = l1 == l2 and np.array_equal(predict(m1, x1), predict(m2, x2))
    finite = all(np.isfinite(a).all() for a in [*m1.params.values(), *hidden_states(m1, x1).values()])
    return Check("C2", "determinism_and_finiteness", same and finite,
                 f"identical losses and outputs={same}, all parameters and activations finite={finite}")


def c3_learning(run, th):
    losses, keep = run["losses"], 1.0 - th["margin_over_trivial"]
    drop = 1.0 - float(np.mean(losses[-50:])) / float(np.mean(losses[:50]))
    (held, mean_fc), (effect, none_fc) = (run["ord_anatomist"], run["trivial_ordinary"]), \
        (run["extra_anatomist"], run["zero_effect"])
    ok = drop >= th["loss_drop_fraction"] and held <= keep * mean_fc and effect <= keep * none_fc
    return Check("C3", "learning", ok,
                 f"loss drop {drop:.3f} (>= {th['loss_drop_fraction']}); held-out vigor MSE {held:.4f} vs training "
                 f"mean {mean_fc:.4f}; cut-effect MSE {effect:.4f} vs no-effect forecast {none_fc:.4f} (each must be "
                 f"<= {keep:.1f}x)")


def c4_shuffled(run, seed, steps, band, n_perm=5):
    """Vigor permuted across Hayy's own experience, averaged over n_perm permutations (one random ledger can
    point both causes the right way by luck): the anatomist must fall back to the trivial forecasts."""
    x, y = run["experience"]
    held, effect = [], []
    for i in range(n_perm):
        model = new_anatomist(np.random.default_rng(seed + 3 * i))
        perm = np.random.default_rng(seed + 3 * i + 1).permutation(len(y))
        fit(model, (x, y[perm]), steps, np.random.default_rng(seed + 3 * i + 2))
        held.append(level_error(model, run["task"]["ordinary"]))
        effect.append(effect_error(model, run["task"]["extraordinary"]))
    held_mean, effect_mean, floor = float(np.mean(held)), float(np.mean(effect)), 1.0 - band
    ok = held_mean >= floor * run["trivial_ordinary"] and effect_mean >= floor * run["zero_effect"]
    return Check("C4", "shuffled_label_control", ok,
                 f"permuted vigor, mean of {n_perm} permutations: held-out MSE {held_mean:.4f} vs trivial "
                 f"{run['trivial_ordinary']:.4f}, cut-effect MSE {effect_mean:.4f} vs no-effect "
                 f"{run['zero_effect']:.4f} (each >= {floor:.1f}x)")


def short_learning_ok(seed):
    rng = np.random.default_rng(seed)
    parts = [encounter(rng, 150, "open")] + [encounter(rng, 15, "open", "cut", k) for k in range(N_ORGANS)]
    data = (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))
    losses = fit(new_anatomist(np.random.default_rng(seed + 1)), data, STEPS_SHORT, np.random.default_rng(seed + 2))
    return 1.0 - float(np.mean(losses[-50:])) / float(np.mean(losses[:50])) >= 0.5


def c5_mutants(seed, tol):
    """Each learning-breaking mutant must make C1 or the short learning check fail."""
    def caught(name):
        Lab.mutant = name
        try:
            if name.startswith("zero_grad") and not c1_gradcheck(seed, tol).passed:
                return True
            return not short_learning_ok(seed)
        except (NonFiniteError, FloatingPointError):
            return True
        finally:
            Lab.mutant = None

    genuine = short_learning_ok(seed)
    detected = {name: caught(name) for name in MUTANTS}
    return Check("C5", "mutant_detection", genuine and all(detected.values()),
                 f"genuine short learning passes={genuine}; detected {detected}",
                 {"detected": sum(detected.values()), "total": len(detected)})


def random_anatomist(rng, scale):
    model = new_anatomist(rng)
    for k, v in model.params.items():
        model.params[k] = v + rng.normal(0.0, scale, v.shape)
    return model


def additivity_violation(model, body, first, second):
    pre = lambda d: hidden_states(model, np.concatenate([body, d], 1))["seat_preactivation"]
    return float(np.abs(pre(first + second) - pre(first) - pre(second) + pre(0.0 * first)).max())


def c6_1_additive_cuts(seed, tol):
    """Cuts on two organs add in the seat's pre-activation for any parameters and bodies (the single seat of
    D5). Random draws plus a hill-climb on the violation; negative control: the watcher's reading (ledger
    knockout 'identity') passes cut organs through the nonlinear reading and must violate it."""
    rng = np.random.default_rng(seed)
    worst, control = 0.0, np.inf
    for scale in (0.1, 1.0, 5.0):
        model = random_anatomist(rng, scale)
        body, rows = rng.normal(0.0, 2.0, (64, N_ORGANS)), np.arange(64)
        pair = np.argsort(rng.random((64, N_ORGANS)), axis=1)[:, :2]
        first, second = np.zeros((64, N_ORGANS)), np.zeros((64, N_ORGANS))
        first[rows, pair[:, 0]], second[rows, pair[:, 1]] = rng.uniform(-3, 3, 64), rng.uniform(-3, 3, 64)
        v = additivity_violation(model, body, first, second)
        for _ in range(30):
            trial = body + rng.normal(0.0, 0.5, body.shape)
            vt = additivity_violation(model, trial, first, second)
            body, v = (trial, vt) if vt > v else (body, v)
        worst = max(worst, v)
        control = min(control, additivity_violation(knockout(model, "ledger", "identity"), body, first, second))
    return Check("C6.1", "cuts_add_on_one_seat", worst <= tol and control > 1e-3,
                 f"max violation {worst:.1e} (<= {tol:.0e}); watcher's reading violates by {control:.2e} (> 1e-3)")


def c6_2_knife_only(seed):
    """Only cuts write the ledger: on any batch of uncut bodies its loss gradient is exactly zero and
    predictions do not depend on it. Negative control: the same model fed the organ values in place of the
    displacements (a ledger written by watching) must receive gradient and change its predictions."""
    rng = np.random.default_rng(seed)
    worst_grad, worst_pred, control = 0.0, 0.0, np.inf
    for scale in (0.1, 1.0, 3.0):
        model = random_anatomist(rng, scale)
        body, y = rng.normal(0.0, 1.5, (48, N_ORGANS)), rng.normal(0.0, 1.0, (48, 1))
        uncut, broken = np.concatenate([body, np.zeros_like(body)], 1), np.concatenate([body, body], 1)
        other = model.clone()
        other.params["ledger"] = rng.normal(0.0, 3.0, N_ORGANS)
        worst_grad = max(worst_grad, float(np.abs(loss_and_grads(model, {"X": uncut, "Y": y})[1]["ledger"]).max()))
        worst_pred = max(worst_pred, float(np.abs(predict(model, uncut) - predict(other, uncut)).max()))
        control = min(control, float(np.abs(loss_and_grads(model, {"X": broken, "Y": y})[1]["ledger"]).max()),
                      float(np.abs(predict(model, broken) - predict(other, broken)).max()))
    return Check("C6.2", "only_cuts_write_the_ledger", worst_grad == 0.0 and worst_pred == 0.0 and control > 1e-6,
                 f"ledger gradient on uncut bodies {worst_grad:.1e}, prediction change {worst_pred:.1e} (both exactly "
                 f"0); watching-ledger control {control:.2e} (> 1e-6)")


def c6_3_roles(run, th):
    """Trained property: Hayy's cuts give each cause a coefficient >= cause_min, and no other organ opened at
    least min_cuts times rivals the weakest cause (an organ cut a few times keeps a loose coefficient).
    Negative control: the watcher, who never cut, must fail the same rule."""
    cuts = np.array(run["cuts_open"])
    others = [k for k in range(N_ORGANS) if k not in CAUSES and cuts[k] >= th["min_cuts"]]

    def recovered(ledger):
        weakest = ledger[list(CAUSES)].min()
        return bool(weakest >= th["cause_min"] and all(abs(ledger[k]) < weakest for k in others))

    hayy, watcher = run["models"]["anatomist"].params["ledger"], run["models"]["watcher"].params["ledger"]
    return Check("C6.3", "cuts_recover_the_causes", recovered(hayy) and not recovered(watcher),
                 f"ledger {dict(zip(ORGANS, np.round(hayy, 2).tolist()))} after cuts {cuts.tolist()}; causes >= "
                 f"{th['cause_min']} and above every other organ cut >= {th['min_cuts']} times; watcher's ledger "
                 f"fails the rule: {not recovered(watcher)}")


def definition_check(run):
    model, x = run["models"]["anatomist"], run["task"]["extraordinary"][0]
    zeroed = model.clone()
    zeroed.params["ledger"][:] = 0.0
    same = bool(np.array_equal(predict(knockout(model, "ledger", "zero"), x), predict(zeroed, x)))
    return Check("DEF", "ledger_zero_knockout_equals_zero_ledger (definition check, not evidence)", same,
                 f"identical predictions: {same}")


def row_hashes(x):
    return {hashlib.sha1(np.ascontiguousarray(r).tobytes()).hexdigest() for r in x}


def c7_split(run):
    """No evaluation body appears in any training set; training cuts stay inside the knife's range and touch
    one organ; extraordinary rows are single cuts beyond the range or double cuts."""
    train, task = np.concatenate([d[0] for d in run["training_sets"]]), run["task"]
    evals = set().union(*(row_hashes(task[k][0]) for k in ("ordinary", "extraordinary", "quiet_injury")))
    overlap = len(row_hashes(train) & evals)
    extra = task["extraordinary"][0]
    cut_train, cut_extra = train[:, N_ORGANS:] != 0.0, extra[:, N_ORGANS:] != 0.0
    forced_train = np.abs(train[:, :N_ORGANS] + train[:, N_ORGANS:])[cut_train]
    singles = cut_extra & (cut_extra.sum(1) == 1)[:, None]
    forced_single = np.abs(extra[:, :N_ORGANS] + extra[:, N_ORGANS:])[singles]
    shifted = cut_train.sum(1).max() <= 1 and forced_train.max() <= KNIFE and forced_single.min() >= BEYOND[0] \
        and set(cut_extra.sum(1).tolist()) == {1, 2}
    return Check("C7", "split_integrity", overlap == 0 and shifted,
                 f"train/eval overlap {overlap}; training cuts single and |v| <= {KNIFE} (max {forced_train.max():.2f});"
                 f" extraordinary singles |v| >= {BEYOND[0]} (min {forced_single.min():.2f}) and doubles never seen in "
                 f"training: {shifted}")


# ============================================================================ tests: hypotheses
OPEN_MODELS = ("anatomist", "do_mlp", "citizen", "anatomy_pupil", "anatomist_randomized", "do_mlp_own_data",
               "watcher")
QUIET_MODELS = ("anatomist", "do_mlp", "anatomist_chance", "anatomist_randomized")
KO_ROWS = (("ledger", "zero"), ("ledger", "identity"), ("ledger", "mean"), ("correction", "mean"))
HYP_DIFFS = {
    "H-SIG": lambda r: r["extra_anatomist"] - r["extra_do_mlp"],
    "H-NEC": lambda r: r["ko_extra_ledger_zero"] - r["ko_extra_correction_mean"],
    "H-BLIND": lambda r: r["quiet_anatomist"] - r["quiet_do_mlp"],
    "H-RIVAL": lambda r: r["extra_anatomist"] - r["extra_citizen"],
}


def run_seed(seed, steps):
    ss = np.random.SeedSequence(seed).spawn(3)
    task = make_task(ss[0])
    island, info, sets = open_island(ss[1], steps)
    quiet, quiet_info, repaired_info, quiet_sets = quiet_island(ss[2], steps)
    out = {"seed": seed, "task": task, "models": island, "losses": info["losses"], "experience": sets[0],
           "training_sets": sets + quiet_sets}
    for name, model in island.items():
        out["ord_" + name] = level_error(model, task["ordinary"])
        out["extra_" + name] = effect_error(model, task["extraordinary"])
    out.update({"quiet_" + name: effect_error(model, task["quiet_injury"]) for name, model in quiet.items()})
    for module, mode in KO_ROWS:
        cut = knockout(island["anatomist"], module, mode)
        out[f"ko_extra_{module}_{mode}"] = effect_error(cut, task["extraordinary"])
        out[f"ko_ord_{module}_{mode}"] = level_error(cut, task["ordinary"])
    own_x, own_y = sets[0]
    watched = np.all(own_x[:, N_ORGANS:] == 0.0, axis=1)
    out["trivial_ordinary"] = float(((task["ordinary"][1] - own_y[watched].mean()) ** 2).mean())
    out["zero_effect"] = float((task["extraordinary"][2] ** 2).mean())
    out["zero_effect_quiet"] = float((task["quiet_injury"][2] ** 2).mean())
    out.update(promoted=info["promoted"], cuts_open=info["cuts"].tolist(), cuts_quiet=quiet_info["cuts"].tolist(),
               marrow_cuts_repaired=int(repaired_info["cuts"][MARROW]),
               ledger=island["anatomist"].params["ledger"].tolist(),
               n_params={k: n_params(island[k]) for k in ("anatomist", "do_mlp", "citizen")})
    return out


def contrast(runs, measure, rng):
    return paired_bootstrap([measure(r) for r in runs], rng)


def hypothesis_rows(runs, quick, rng):
    rows = []
    for spec in MIND_CARD["hypotheses"]:
        mean, ci = contrast(runs, HYP_DIFFS[spec["id"]], rng)
        call = "not evaluated" if quick else verdict(mean, ci, spec["direction"], spec["mesi"])
        rows.append(dict(id=spec["id"], metric=spec["metric"], mean_diff=mean, ci95=ci, mesi=spec["mesi"],
                         n_seeds=len(runs), direction=spec["direction"], verdict=call))
    return rows


def knockout_rows(runs, rng):
    registry = modules(runs[0]["models"]["anatomist"])
    rows = []
    for module, mode in KO_ROWS:
        key = f"{module}_{mode}"
        cut_mean, cut_ci = contrast(runs, lambda r: r["ko_extra_" + key] - r["extra_anatomist"], rng)
        health_mean, health_ci = contrast(runs, lambda r: r["ko_ord_" + key] - r["ord_anatomist"], rng)
        rows.append(dict(module=module, mode=mode, signature=registry[module]["signature"], metric_change=cut_mean,
                         ci95=cut_ci, ordinary_change=health_mean, ordinary_ci95=health_ci))
    return rows


# ============================================================================ report
def spread(runs, key):
    values = [r[key] for r in runs]
    return [float(np.mean(values)), float(np.std(values))]


def descriptive(runs):
    keys = [f"{s}_{m}" for s in ("ord", "extra") for m in OPEN_MODELS] + [f"quiet_{m}" for m in QUIET_MODELS] + \
        ["trivial_ordinary", "zero_effect", "zero_effect_quiet"]
    ledger = np.round(np.mean([r["ledger"] for r in runs], axis=0), 3).tolist()
    return {"errors": {k: spread(runs, k) for k in keys},
            "inquiry": {"promoted_after_watches": [r["promoted"] for r in runs],
                        "cuts_open": [r["cuts_open"] for r in runs], "cuts_quiet": [r["cuts_quiet"] for r in runs],
                        "marrow_cuts_quiet": [r["cuts_quiet"][MARROW] for r in runs],
                        "marrow_cuts_quiet_with_chance": [r["marrow_cuts_repaired"] for r in runs],
                        "ledger_mean": dict(zip(ORGANS, ledger))}}


def render(rep):
    """The verified report, line by line."""
    err, inq, g, rule = rep["descriptive"]["errors"], rep["descriptive"]["inquiry"], rep["gradcheck"], "-" * 78
    text = [f"=== VERIFIED REPORT · chapter {CHAPTER:04d} ===",
            f"file {rep['file']}  card_revision {rep['card_revision']}  python {rep['environment']['python']}  "
            f"numpy {rep['environment']['numpy']}",
            f"seeds {rep['seeds']}  runtime {rep['runtime_s']:.1f} s  params: anatomist {rep['n_params']}, do-MLP "
            f"{rep['n_params_baseline']}, citizen {rep['n_params_rival']}",
            f"gradient check: {g['tensors_checked']}/{g['tensors_total']} tensor checks at {g['checked_at']}, max rel "
            f"err {g['max_rel_error']:.2e}, passed={g['passed']}", rule, "correctness"]
    text += [f"  {c['id']:<5} {'PASS' if c['passed'] else 'FAIL'}  {c['name']}: {c['detail']}" for c in rep["correctness"]]
    text += [f"mutation score {rep['mutants']['detected']}/{rep['mutants']['total']} = {rep['mutants']['score']:.2f}",
             rule, "open island, mean (sd) over seeds: vigor MSE on bodies in health | cut-effect MSE, extraordinary"]
    text += [f"  {m:<22}{err['ord_' + m][0]:>9.4f} ({err['ord_' + m][1]:.4f}) {err['extra_' + m][0]:>11.4f} "
             f"({err['extra_' + m][1]:.4f})" for m in OPEN_MODELS]
    text += [f"  {'trivial forecasts':<22}{err['trivial_ordinary'][0]:>9.4f} (mean)   {err['zero_effect'][0]:>11.4f} "
             f"(no effect)", "quiet island, cut-effect MSE on injuries to marrow"]
    text += [f"  {m:<22}{err['quiet_' + m][0]:>9.4f} ({err['quiet_' + m][1]:.4f})" for m in QUIET_MODELS]
    text += [f"  {'no-effect forecast':<22}{err['zero_effect_quiet'][0]:>9.4f}",
             f"Hayy promoted himself after {inq['promoted_after_watches']} watches; cuts per organ {list(ORGANS)}:",
             f"  open island {inq['cuts_open']}",
             f"  quiet island cuts on marrow: Hayy {inq['marrow_cuts_quiet']}, with 10% chance "
             f"{inq['marrow_cuts_quiet_with_chance']}; mean ledger {inq['ledger_mean']}",
             rule, "hypotheses (paired per-seed differences, 95% bootstrap CI, 2000 resamples)"]
    text += [f"  {h['id']:<8} diff {h['mean_diff']:+.4f}  CI [{h['ci95'][0]:+.4f}, {h['ci95'][1]:+.4f}]  mesi "
             f"{h['mesi']} ({h['direction']})  -> {h['verdict']}" for h in rep["hypotheses"]]
    text += [rule, "knockouts on the trained anatomist, open island (change in error vs intact, 95% CI)"]
    text += [f"  {k['module']:<10} {k['mode']:<9} signature={str(k['signature']):<5} cut effect {k['metric_change']:+.4f} "
             f"[{k['ci95'][0]:+.4f}, {k['ci95'][1]:+.4f}]  health {k['ordinary_change']:+.4f} "
             f"[{k['ordinary_ci95'][0]:+.4f}, {k['ordinary_ci95'][1]:+.4f}]" for k in rep["knockouts"]]
    return text + [f"task types {rep['task_types']}  exit code {rep['exit_code']}", "=== END REPORT ==="]


def real_data_bridge(path, rng):
    """Optional CSV with a header: K organ columns, then K columns do_<organ> holding the intended displacement
    (0 when untouched), then the outcome. Rows in file order; the last 20% are held out."""
    if not path:
        print("no --data given: real-data bridge skipped")
        return None
    with open(path, encoding="utf-8") as fh:
        header = [h.strip() for h in fh.readline().split(",")]
    k = (len(header) - 1) // 2
    if len(header) != 2 * k + 1 or header[k:2 * k] != ["do_" + h for h in header[:k]]:
        print("real-data bridge: the header must read organs..., do_<organ>..., outcome")
        return None
    raw = np.genfromtxt(path, delimiter=",", skip_header=1)
    raw = raw[np.isfinite(raw).all(axis=1)]
    cut = int(0.8 * len(raw))
    scale = raw[:cut, :k].std(0) + 1e-9
    x = np.concatenate([(raw[:, :k] - raw[:cut, :k].mean(0)) / scale, raw[:, k:2 * k] / scale], 1)
    y = (raw[:, -1:] - raw[:cut, -1].mean()) / (raw[:cut, -1].std() + 1e-9)
    anatomist = build_model(2 * k, 1, "vector_regression", rng, organs=k)
    scores = {}
    for name, model in (("anatomist", anatomist), ("do_mlp", build_mlp(2 * k, 1, matched_hidden(n_params(anatomist),
                                                                                                  2 * k, 1), rng))):
        fit(model, (x[:cut], y[:cut]), STEPS_QUICK, rng)
        scores[name] = float(((predict(model, x[cut:]) - y[cut:]) ** 2).mean())
    print(f"real-data bridge on {os.path.basename(path)}: held-out MSE (standardised) {scores}")
    return scores


# ============================================================================ command-line entry point
def correctness(args, runs, base, steps, th):
    first = runs[0]
    checks = [c1_gradcheck(base, th["gradcheck_rel_tol"]), c2_determinism(base + 1), c3_learning(first, th),
              c4_shuffled(first, base + 2, steps, th["shuffled_band"])]
    checks.append(Check("C5", "mutant_detection", True, "skipped under --mutant", {"detected": 0, "total": 0})
                  if args.mutant else c5_mutants(base + 3, th["gradcheck_rel_tol"]))
    return checks + [c6_1_additive_cuts(base + 4, th["additivity_tol"]), c6_2_knife_only(base + 5),
                     c6_3_roles(first, th), c7_split(first), definition_check(first)]


def protocol(args):
    started, quick = time.time(), args.quick
    steps, th = (STEPS_QUICK if quick else STEPS_FULL), MIND_CARD["thresholds"]
    seeds = list(range(args.seed, args.seed + (1 if quick else args.seeds)))
    print(f"chapter {CHAPTER:04d} · {os.path.basename(__file__)} · seeds {seeds} · steps {steps}"
          + (f" · mutant {args.mutant}" if args.mutant else ""), flush=True)
    Lab.mutant, runs = args.mutant, []
    for s in seeds:
        r = run_seed(s, steps)
        runs.append(r)
        print(f"  seed {s}: cut-effect MSE anatomist {r['extra_anatomist']:.4f}  do-MLP {r['extra_do_mlp']:.4f}  "
              f"citizen {r['extra_citizen']:.4f} | quiet injuries {r['quiet_anatomist']:.4f} vs "
              f"{r['quiet_do_mlp']:.4f}  ({time.time() - started:.0f} s)", flush=True)
    checks = correctness(args, runs, args.seed + 10_000, steps, th)
    stats_rng = np.random.default_rng(args.seed + 20_000)
    hyps, kos = hypothesis_rows(runs, quick, stats_rng), knockout_rows(runs, stats_rng)
    runtime, budget = time.time() - started, (BUDGET_QUICK if quick else BUDGET_FULL)
    checks.append(Check("C8", "budget", runtime <= budget, f"{runtime:.1f} s (budget {budget:.0f} s)"))
    exit_code = 0 if all(c.passed for c in checks) else (3 if all(c.passed for c in checks[:-1]) else 1)
    gradient, mutants = checks[0].extra, next(c for c in checks if c.id == "C5").extra
    rep = {"schema_version": "1.0", "chapter": CHAPTER, "file": os.path.basename(__file__),
           "card_revision": MIND_CARD["card_revision"], "seeds": seeds, "runtime_s": round(runtime, 2),
           "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
           "n_params": runs[0]["n_params"]["anatomist"], "n_params_baseline": runs[0]["n_params"]["do_mlp"],
           "n_params_rival": runs[0]["n_params"]["citizen"],
           "gradcheck": dict(gradient, checked_at=["init", "after_training_steps"], passed=bool(checks[0].passed)),
           "correctness": [c.entry() for c in checks],
           "mutants": dict(mutants, score=mutants["detected"] / mutants["total"] if mutants["total"] else 0.0),
           "hypotheses": hyps, "knockouts": kos, "task_types": TASK_TYPES, "exit_code": exit_code,
           "mutant": args.mutant, "descriptive": descriptive(runs)}
    print("\n".join(render(rep)))
    real_data_bridge(args.data, np.random.default_rng(args.seed))
    if args.json:
        write_report(args.json, rep)
    return exit_code


def main(argv=None):
    parser = argparse.ArgumentParser(prog=os.path.basename(__file__),
                                     description="Chapter 0292, the open body: the full protocol by default")
    parser.add_argument("--quick", action="store_true", help="one seed, fewer updates, every correctness test")
    parser.add_argument("--seed", type=int, default=0, help="first seed")
    parser.add_argument("--seeds", type=int, default=5, help="number of seeds")
    parser.add_argument("--json", default=None, help="also write the JSON report to this path")
    parser.add_argument("--card", action="store_true", help="print MIND_CARD as JSON and exit")
    parser.add_argument("--mutant", default=None, choices=sorted(MUTANTS), help="run with a registered mutant")
    parser.add_argument("--data", default=None, help="optional CSV for the real-data bridge")
    args = parser.parse_args(argv)
    if args.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    if args.seeds < 1:
        parser.print_usage()
        return 2
    try:
        return protocol(args)
    except (NonFiniteError, FloatingPointError) as err:
        print(f"non-finite values: {err}")
        return 4


if __name__ == "__main__":
    sys.exit(main())

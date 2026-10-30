#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0285 · Avempace (Ibn Bajja)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 15, Minds 281-300 Available on Amazon https://www.amazon.com/dp/B0HMQ7TVJM
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0285_avempace_ibn_bajja_1085 - Avempace (Ibn Bajja) (c. 1085-1139)
# END ATTRIBUTION
"""
Ladder of quiddities: solitary ascent to a knower-independent code (the Ittisal ascent).

Thesis
    The intelligible is many only through the private images that carry it:
    re-express each concept by its relations to one's own concepts, then do it
    again to the result, and minds that never met hold the same code.

Evidence and provenance  (provenance: belief)
    His own Rule of the Solitary (Tadbir al-mutawahhid), Epistle of Conjunction
    (Risalat al-ittisal) and Farewell Epistle survive (Fakhry 1968; Genequand
    2010). The regress argument is best preserved in Latin reports.
    D1 Ittisal, via Aquinas In IV Sent. d.49 q.2 a.1: the understood idea is
       diversified in me and you only by the imaginative forms it is joined to;
       separated from them it is one and the same in all who understand.
    D2 Aquinas ST I q.88 a.2 and SCG III.41 (from Averroes): abstraction can be
       applied again to the abstracted quiddity; it cannot go on forever, so it
       ends in a quiddity that has no further quiddity.
    D3 Ittisal (Fakhry 164; Genequand s.32): each man has a material intellect
       and the intelligibles are relative to each material intellect.
    D4 Ittisal (Fakhry 167-168): three ranks - the multitude (intelligibles tied
       to material objects), the theorists (tied to spiritual forms), the blessed
       (the thing itself); the cave, the reflection, the sun.
    D5 Ittisal, via Inati 1998: the blessed are numerically one; only their
       bodies distinguish them.
    D6 Tadbir (Asin 10; Berman 1963: 127): the weeds hold true opinions absent
       from their city; the solitary keeps aloof and governs himself.
    D7 Tadbir (Asin 26-27; Fakhry 55): the Sufis take the coincidence of common
       sense, imagination and memory - particular spiritual forms - for the
       universal.
    D8 Aquinas ST I q.88 a.2; SEP (Puig Montada 2023): the ascent reaches only
       what the material things contain, and its summit leaves articulated
       knowledge behind.

Doctrine -> mechanism -> test
    D3     -> M0 private imagination per learner, trained alone -> C6.1 solitude
    D1, D2 -> M1 rungs: re-express every code by its cosine to the centred
              centroids of one's own concepts, then again (signature) -> H-SIG,
              H-NEC, C6.2 termination, C6.3 private-frame invariance
    D4     -> rung readout: 1 = tied to the image, 2+ = quiddity of the quiddity
    D5     -> conjunction metric: a probe fitted on one learner's code is read
              on another learner's code, never jointly trained
    D6     -> no gradient, clipping or data crosses learners (C6.1)
    D7     -> R1 rival: one rung plus training for coincidence of three internal
              re-presentations -> H-RIVAL
    D8     -> blind spot: attributes of the particular -> H-BLIND; articulation
              of the relation matrix per rung (descriptive)

Research question
    Representation interoperability without communication: can independently
    trained learners with private, nonlinear input maps expose codes that a
    downstream module fitted on one of them reads correctly on another?

Closest prior art and the delta
    Relative representations (Moschella et al., ICLR 2023); second-order
    isomorphism (Shepard and Chipman 1970) and representational similarity
    analysis (Kriegeskorte et al. 2008); model stitching (Bansal et al. 2021);
    prototypical networks (Snell et al. 2017). Delta: the relative step is
    iterated on its own output with class-conditional anchors and trained end
    to end, so a code that is still in the learner's frame after one step is
    re-expressed until the frame is gone; the file tests that the second step,
    not the first, carries the effect, and what it costs.

Blind spot
    What varies within a species (the particular) is treated as the private
    carrier to be discarded, so the summit cannot report it; the relation
    matrix among concepts also flattens as the ladder climbs.

Task (generative process)
    World: 6 species in two genera (6-d state; dims 0-3 species, 4-5 the
    particular). Each learner perceives through a private map
    x = R [tanh(A s + a) + noise ; private noise channels], R orthogonal.
    3 ordinary learners and 2 newcomers (harsher maps) train alone on 600
    labelled percepts; 8 of their own percepts per species are anchors.
    Probes: blends of two species (target: membership vector), fitted on one
    learner's code and read on another's (shifted split: newcomers); pure
    percepts for accuracy and for the particular attributes.

Limits
    Synthetic world, one modality, species known by name to every learner.
    A research prototype of an AGI-oriented component, not an AGI; it
    replicates no one's mind, and nothing here is evidence about the person.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 2,
    "revision_log": [
        "r1: written before the first full run. World constants, the number of rungs (3), the code each "
        "model exports, the conjunction metric (R2 floored at -1) and every mesi were fixed after pilots "
        "on reserved seeds 991-999, which are excluded from evaluation. The pilots showed that a softmax "
        "classifier's probabilities interchange about as well as the ladder in this world, and that a "
        "class-prior shift between learners does not separate them; that comparison is therefore declared "
        "as expected parity (descriptive) and not as a hypothesis.",
        "r2: after the first --quick run, the zero-norm guard inside the cosine was lowered from 1e-9 to 1e-24, "
        "because the larger guard broke exact scale invariance of small latents (C6.3 failed at 5e-5). No "
        "hypothesis, metric, threshold or seed changed; no full run had been made."],
    "generation": {"template_version": "1.0", "generator": "Claude", "generator_version": "claude-opus-5.5",
                   "date": "2026-09-23"},
    "id": 285, "figure": "Avempace (Ibn Bajja)", "born": 1085, "died": 1139,
    "civilization": "Andalusi Arab (al-Andalus and the Maghrib under the Hudids and the Almoravids; Arabic)",
    "provenance": "belief",
    "thesis": ("The intelligible is many only through the private images that carry it: iterate the relation of "
               "one's concepts to one's own concepts and solitary minds converge on one code."),
    "evidence": [
        {"id": "D1", "claim": "The understood idea is diversified in different knowers only by the imaginative "
         "forms it is joined to; separated from them it is one in all who understand", "basis": "primary",
         "source": "Risalat al-ittisal (ed. Asin Palacios 1942; Fakhry 1968: 153-173), as reported by Aquinas, "
                   "In IV Sent. d.49 q.2 a.1 (= ST Suppl. q.92 a.1)"},
        {"id": "D2", "claim": "Abstraction can be applied to the abstracted quiddity; the regress cannot be "
         "infinite and ends in a quiddity without further quiddity", "basis": "scholarship",
         "source": "Aquinas, ST I q.88 a.2 and SCG III.41, reporting Averroes, In De anima III; Wirmer 2006"},
        {"id": "D3", "claim": "Each man has a material intellect; intelligibles are relative to each material "
         "intellect", "basis": "primary", "source": "Ittisal, Fakhry 1968: 164; Genequand 2010: s.32"},
        {"id": "D4", "claim": "Three ranks: the multitude, the theorists and the blessed; the cave simile",
         "basis": "primary", "source": "Ittisal, Fakhry 1968: 167-168; Genequand 2010: s.41-47"},
        {"id": "D5", "claim": "The blessed are numerically one, distinguished only by their bodies",
         "basis": "primary", "source": "Ittisal, via Inati, Routledge Encyclopedia of Philosophy 1998"},
        {"id": "D6", "claim": "Weeds hold true opinions absent from the city; the solitary keeps aloof, meets "
         "others only when unavoidable and emigrates to where the sciences exist", "basis": "primary",
         "source": "Tadbir al-mutawahhid, Asin Palacios 1946: 10; Berman trans. 1963: 127"},
        {"id": "D7", "claim": "Sufis take the coincidence of common sense, imagination and memory, which yields "
         "particular spiritual forms, for the universal", "basis": "primary",
         "source": "Tadbir, Asin Palacios 1946: 26-27; Fakhry 1968: 55"},
        {"id": "D8", "claim": "Abstraction from material things reaches only what they contain; the summit "
         "abandons articulated knowledge", "basis": "scholarship",
         "source": "Aquinas, ST I q.88 a.2; Puig Montada, SEP 'Ibn Bajja' (rev. 2023), s.7"},
    ],
    "research_question": {"category": "sample efficiency and transfer",
                          "question": "Can learners trained in isolation, each through a private nonlinear input "
                                      "map, expose codes that a module fitted on one reads correctly on another, and "
                                      "does iterating the relative re-expression (not one step) produce it?"},
    "mechanism": {
        "name": "Ladder of quiddities",
        "family": "iterated relative (second-order) representation with class-conditional anchors",
        "signature_modules": ["ascent"],
        "closest_prior_art": ["Moschella et al. 2023, Relative representations enable zero-shot latent space "
                              "communication (ICLR)",
                              "Shepard & Chipman 1970, Second-order isomorphism (Cognitive Psychology 1)",
                              "Kriegeskorte, Mur & Bandettini 2008, Representational similarity analysis "
                              "(Frontiers in Systems Neuroscience 2)",
                              "Bansal, Nakkiran & Barak 2021, Revisiting model stitching (NeurIPS)",
                              "Snell, Swersky & Zemel 2017, Prototypical networks (NeurIPS)"],
        "overlap": "Medium",
        "prior_art_queries": ["relative representation zero-shot stitching", "iterated relative representation",
                              "second-order similarity representation learning", "latent space communication "
                              "without shared data", "prototype cosine classifier interoperability",
                              "representational convergence independent models"],
        "contribution_type": "mechanism",
        "delta": ("The relative step is applied again to its own output and trained end to end inside each "
                  "solitary learner; the test isolates the second step as the carrier of interoperability and "
                  "measures what the ascent discards."),
    },
    "traceability": [
        {"doctrine": "D3", "mechanism": "M0", "property_test": "C6.1", "hypothesis": "H-SIG"},
        {"doctrine": "D6", "mechanism": "M0", "property_test": "C6.1", "hypothesis": "H-SIG"},
        {"doctrine": "D1", "mechanism": "M1", "property_test": "C6.3", "hypothesis": "H-SIG"},
        {"doctrine": "D2", "mechanism": "M1", "property_test": "C6.2", "hypothesis": "H-NEC"},
        {"doctrine": "D4", "mechanism": "M1", "property_test": "DEF", "hypothesis": "H-NEC"},
        {"doctrine": "D5", "mechanism": "M1", "property_test": "C6.3", "hypothesis": "H-SIG"},
        {"doctrine": "D7", "mechanism": "R1", "property_test": "C1", "hypothesis": "H-RIVAL"},
        {"doctrine": "D8", "mechanism": "M1", "property_test": "C6.2", "hypothesis": "H-BLIND"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "Newcomers read the ladder's summit code with probes fitted on ordinary "
         "learners better than they read a size-matched one-step relative representation", "metric":
         "conjunction_r2", "split": "shifted", "comparison": "model - baseline", "direction": "greater",
         "mesi": 0.30, "seeds": 5},
        {"id": "H-NEC", "statement": "Stopping the trained ladder at its first rung lowers conjunction more than "
         "holding the rung sharpness at its mean", "metric": "conjunction_r2", "split": "shifted",
         "comparison": "signature_knockout - matched_knockout", "direction": "less", "mesi": 0.30, "seeds": 5},
        {"id": "H-BLIND", "statement": "The ladder's summit code reports the particular attributes of an "
         "individual worse than the one-step relative representation", "condition": "probe for the two "
         "within-species attributes (world dims 4-5), fitted and read within each learner", "grounding":
         "Aquinas ST I q.88 a.2 and SEP s.7: the ascent keeps only the universal and leaves articulation behind",
         "metric": "particular_r2", "split": "in_distribution", "comparison": "model - baseline",
         "direction": "less", "mesi": 0.05, "seeds": 5},
        {"id": "H-RIVAL", "statement": "The ladder's conjunction exceeds that of the rival trained for coincidence "
         "of three internal re-presentations at one rung (the route the Tadbir attributes to the Sufis)",
         "metric": "conjunction_r2", "split": "shifted", "comparison": "model - rival", "direction": "greater",
         "mesi": 0.30, "seeds": 5},
    ],
    "descriptive_expectations": {
        "plain_softmax_probabilities": "parity with the ladder within about 0.1 R2 (pilots): a softmax already "
                                       "normalises to a concept-indexed simplex",
        "plain_logits": "conjunction near the floor of -1",
        "ascent_profile": "rung 1 near the floor, rung 2 and above high",
        "articulation": "name-free concept matching from the relation matrix falls as the ladder climbs"},
    "thresholds": {"loss_drop_fraction": 0.8, "margin_over_trivial": 0.5, "shuffled_band": 0.1,
                   "gradcheck_rel_tol": 1e-5, "conjunction_floor": -1.0},
    "probe_predictions": [{"probe": "P2", "expected": "above"}, {"probe": "P6", "expected": "above"},
                          {"probe": "P9", "expected": "equal to baseline"}, {"probe": "P1", "expected": "below"},
                          {"probe": "P7", "expected": "below"}, {"probe": "P8", "expected": "below"},
                          {"probe": "P5", "expected": "equal to baseline"}],
    "dialectic_links": [
        {"chapter": 235, "relation": "teacher", "test": "none (the weeds and conjunction inherited from al-Farabi)"},
        {"chapter": 277, "relation": "rival", "test": "H-RIVAL (the Sufi route as the Tadbir characterises it; "
                                                     "chapter 277's own mechanism is not reproduced)"},
        {"chapter": 298, "relation": "successor", "test": "none (Averroes made the material intellect one; this "
                                                         "file keeps it individual and tests solitude)"},
        {"chapter": 337, "relation": "rival", "test": "H-BLIND (Aquinas's objection to the ascent)"},
        {"chapter": 303, "relation": "successor", "test": "none (Maimonides read and praised him)"},
        {"chapter": 272, "relation": "rival", "test": "none (he faulted al-Zarqali's method for Mercury's apogee)"}],
    "corpus_neighbors": [
        {"chapter": 298, "similarity": None, "difference": "Averroes shares one receiving intellect by "
         "construction; here every learner is private and sameness must be reached by the ascent"},
        {"chapter": 270, "similarity": None, "difference": "Ibn Gabirol peels stacked forms down to one matter "
         "shared by all things; here the endpoint is shared by all knowers of one thing"},
        {"chapter": 235, "similarity": None, "difference": "al-Farabi audits a lossy transform between two "
         "encodings; here one encoding is re-expressed until its frame disappears"},
        {"chapter": 257, "similarity": None, "difference": "Ibn Sina validates a leap by episodic conjunction; "
         "here conjunction is measured as interoperability between isolated learners"},
        {"chapter": 279, "similarity": None, "difference": "Judah Halevi licenses amendments to transmitted "
         "practice; here nothing is transmitted at all"},
        {"chapter": 283, "similarity": None, "difference": "Li Qingzhao catalogues provenance against loss; here "
         "provenance (who holds the code) is what the ascent removes"}],
    "similarity_note": "Token-shingle similarity computed only against the sample file available in this "
                       "session; to be re-run against the full corpus.",
    "barometer": {
        "cognitive_processing": ["classification accuracy of each solitary (C3)"],
        "embodied_cognition": ["not measured: no body"],
        "world_modeling": ["blend membership recovered across private maps (H-SIG)"],
        "consciousness": ["not claimed"],
        "language_understanding": ["name-free concept matching per rung (descriptive)"],
        "emotional_intelligence": ["not measured"],
        "creativity": ["not measured"],
        "autonomy": ["learning in isolation, no shared data or gradients (C6.1)"]},
    "task_types": ["vector_classification"],
    "applications": [
        {"use": "Swapping independently trained encoders and heads without joint retraining",
         "sector": "ML operations", "dataset": "CIFAR-100 (Krizhevsky 2009), as used by Moschella et al. 2023"},
        {"use": "Sites that cannot share data exposing codes a common downstream reader can use",
         "sector": "healthcare (research decision support only)",
         "dataset": "Camelyon17-WILDS (Koh et al. 2021), one hospital per learner"},
        {"use": "Devices with different sensor placements sharing activity codes",
         "sector": "wearables", "dataset": "UCI HAR, Human Activity Recognition Using Smartphones "
                                           "(Anguita et al. 2013), one sensor subset per learner"}],
    "safety_notes": ("No hazardous domain. Synthetic data only. The doctrine of the blessed as numerically one "
                     "is modelled as code interoperability and makes no claim about persons or survival."),
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

CHAPTER = 285
N_SPECIES, D_WORLD, D_SENSE, N_PRIVATE = 6, 6, 12, 4      # world dims 0-3 species, 4-5 the particular
HIDDEN, WIDTH, RUNGS, N_ANCHOR = 16, 8, 3, 8
N_TRAIN, N_PROBE, N_OLD, N_NEW = 600, 400, 3, 2
BETA0, COINCIDE, EPS_NORM = 4.0, 1.0, 1e-24    # EPS_NORM guards a zero norm, far below any scale in C6.3
STEPS_FULL, STEPS_QUICK, STEPS_SHORT = 1500, 500, 300
LR, BATCH, CLIP = 0.01, 48, 5.0
BUDGET_FULL, BUDGET_QUICK = 180.0, 20.0
TASK_TYPES = ["vector_classification"]
ACTIVE_MUTANT = {"name": None}
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


# ============================================================================ data: one world, many private imaginations
def make_world(rng):
    """Six species in two genera. Dims 0-1 carry the genus, 2-3 the species, 4-5 the particular (size, age)."""
    mu = np.zeros((N_SPECIES, D_WORLD))
    for c in range(N_SPECIES):
        genus = c // 3
        angle = 2.0 * math.pi * (c % 3) / 3.0 + 0.5 * genus
        mu[c, :2] = np.array([1.6 if genus == 0 else -1.6, 0.0]) + rng.normal(0.0, 0.15, 2)
        mu[c, 2:4] = 1.2 * np.array([math.cos(angle), math.sin(angle)])
    return mu


def make_imagination(rng, harsh=False):
    """A private sense: a random nonlinear map of the world state, private noise channels, a private rotation."""
    gain = 1.8 if harsh else 1.0
    q, _ = np.linalg.qr(rng.normal(size=(D_SENSE, D_SENSE)))
    return {"A": rng.normal(0.0, gain / math.sqrt(D_WORLD), (D_SENSE - N_PRIVATE, D_WORLD)),
            "a": rng.normal(0.0, 0.3, D_SENSE - N_PRIVATE), "R": q, "noise": 0.5 if harsh else 0.3}


def perceive(im, states, rng):
    h = np.tanh(states @ im["A"].T + im["a"]) + rng.normal(0.0, 0.05, (len(states), D_SENSE - N_PRIVATE))
    private = im["noise"] * rng.normal(0.0, 1.0, (len(states), N_PRIVATE))
    return np.concatenate([h, private], 1) @ im["R"].T


def individuals(mu, species, rng, spread=0.25):
    """World states of individuals: their species' centre plus species noise and a real particular."""
    n = len(species)
    return mu[species] + np.concatenate([rng.normal(0.0, spread, (n, 4)), rng.normal(0.0, 1.0, (n, 2))], 1)


def blends(mu, n, rng):
    """States mixed from two species; the target is the membership vector."""
    a = rng.integers(0, N_SPECIES, n)
    b = (a + rng.integers(1, N_SPECIES, n)) % N_SPECIES
    alpha = rng.uniform(0.0, 1.0, n)
    member = np.zeros((n, N_SPECIES))
    member[np.arange(n), a] += 1.0 - alpha
    member[np.arange(n), b] += alpha
    states = member @ mu + np.concatenate([rng.normal(0.0, 0.15, (n, 4)), rng.normal(0.0, 1.0, (n, 2))], 1)
    return states, member


def make_task(seed_seq):
    """Five solitaries: three ordinary, two newcomers with harsher senses. Nothing is shared but the world."""
    r_world, r_sense, r_data = [np.random.default_rng(s) for s in seed_seq.spawn(3)]
    mu = make_world(r_world)
    senses = [make_imagination(r_sense, harsh=i >= N_OLD) for i in range(N_OLD + N_NEW)]
    L = len(senses)
    x_train, y_train = np.zeros((L, N_TRAIN, D_SENSE)), np.zeros((L, N_TRAIN), int)
    for i, im in enumerate(senses):
        y = np.concatenate([np.repeat(np.arange(N_SPECIES), N_ANCHOR), r_data.integers(0, N_SPECIES,
                                                                                         N_TRAIN - N_SPECIES * N_ANCHOR)])
        x_train[i], y_train[i] = perceive(im, individuals(mu, y, r_data), r_data), y
    probe = {}
    for name in ("fit", "read"):
        states, member = blends(mu, N_PROBE, r_data)
        probe[name] = (np.stack([perceive(im, states, r_data) for im in senses]), member)
    pure = {}
    for name in ("fit", "read"):
        species = r_data.integers(0, N_SPECIES, N_PROBE)
        states = individuals(mu, species, r_data)
        pure[name] = (np.stack([perceive(im, states, r_data) for im in senses]), species, states[:, 4:])
    return {"mu": mu, "train": (x_train, y_train), "blend": probe, "pure": pure,
            "old": list(range(N_OLD)), "new": list(range(N_OLD, L))}


# ============================================================================ model: the ladder of quiddities
def build_model(in_dim, out_dim, task_type, rng, learners=1, rungs=RUNGS, hidden=HIDDEN, width=WIDTH,
                coincide=0.0, plain=False):
    """`learners` independent solitaries stacked on a leading axis (no parameter is shared across it).
    Imagination: tanh layer then a linear map with no bias (the centring makes an origin meaningless).
    Ladder: `rungs` relation steps with learned sharpness beta_k = exp(theta_k). Plain: a softmax head."""
    assert task_type in TASK_TYPES
    L = learners
    p = {"W1": rng.normal(0.0, 1.0 / math.sqrt(in_dim), (L, hidden, in_dim)), "b1": np.zeros((L, hidden)),
         "W2": rng.normal(0.0, 1.0 / math.sqrt(hidden), (L, width, hidden))}
    if plain:
        p["b2"] = np.zeros((L, width))
        p["V"] = rng.normal(0.0, 1.0 / math.sqrt(width), (L, out_dim, width))
        p["c"] = np.zeros((L, out_dim))
    else:
        p["theta"] = np.full((L, rungs), math.log(BETA0))
    return {"params": p, "rungs": 0 if plain else rungs, "plain": plain, "classes": out_dim,
            "coincide": coincide, "anchors": None, "ko": {}}


def imagine(p, x):
    h = np.tanh(np.einsum("lnx,lhx->lnh", x, p["W1"]) + p["b1"][:, None])
    z = np.einsum("lnh,ldh->lnd", h, p["W2"]) + (p["b2"][:, None] if "b2" in p else 0.0)
    return h, z


def group_matrix(classes, per_class):
    return np.kron(np.eye(classes), np.ones((1, per_class))) / per_class


def relate(q, n_query, groups, beta, centre=True):
    """One rung. Centroids of one's own concepts come from the anchor rows; every row is re-expressed by
    its cosine to the (centred) centroids. Returns logits beta*S, the relation profile S, the code P."""
    m = np.einsum("ca,lad->lcd", groups, q[:, n_query:])
    o = m.mean(1, keepdims=True) if centre else np.zeros_like(m[:, :1])
    qc, mc = q - o, m - o
    nq = np.sqrt((qc ** 2).sum(-1, keepdims=True) + EPS_NORM)
    nm = np.sqrt((mc ** 2).sum(-1, keepdims=True) + EPS_NORM)
    qh, mh = qc / nq, mc / nm
    s = np.einsum("lnd,lcd->lnc", qh, mh)
    logits = beta[:, None, None] * s
    return logits, s, softmax(logits), (qh, mh, nq, nm, s, beta, centre)


def relate_back(dlogits, ds_extra, cache, n_query, groups):
    qh, mh, nq, nm, s, beta, centre = cache
    dbeta = (dlogits * s).sum((1, 2))
    ds = beta[:, None, None] * dlogits + ds_extra
    dqh, dmh = np.einsum("lnc,lcd->lnd", ds, mh), np.einsum("lnc,lnd->lcd", ds, qh)
    dqc = (dqh - qh * (dqh * qh).sum(-1, keepdims=True)) / nq
    dmc = (dmh - mh * (dmh * mh).sum(-1, keepdims=True)) / nm
    dm = dmc.copy()
    if centre:
        dm += (-dqc.sum(1, keepdims=True) - dmc.sum(1, keepdims=True)) / mh.shape[1]
    dq = dqc
    dq[:, n_query:] += np.einsum("ca,lcd->lad", groups, dm)
    return dq, dbeta


def softmax_back(dp, p):
    return p * (dp - (dp * p).sum(-1, keepdims=True))


def ascend(model, x, upto=None, z_map=None):
    """Forward pass for queries x (L,n,d) stacked with the learner's own anchors. Returns every rung.
    z_map (tests only) re-frames the private latent before the ascent."""
    p, anchors, groups = model["params"], model["anchors"], model["groups"]
    n = x.shape[1]
    x_all = np.concatenate([x, anchors], 1)
    h, z = imagine(p, x_all)
    z = z if z_map is None else z_map(z)
    ko = model["ko"]
    beta = np.exp(p["theta"])
    if ko.get("sharpness") == "mean":
        beta = np.broadcast_to(beta.mean(1, keepdims=True), beta.shape)
    k_top = model["rungs"] if upto is None else upto
    if ko.get("ascent") == "identity":
        k_top = 1
    q, rungs = z, []
    for k in range(k_top):
        logits, s, q, cache = relate(q, n, groups, beta[:, k], centre=ko.get("centering") != "identity")
        rungs.append({"logits": logits, "s": s, "p": q, "cache": cache})
    return {"x": x_all, "h": h, "z": z, "rungs": rungs, "n": n}


def imagine_back(p, x_all, h, dz, g):
    g["W2"] += np.einsum("lnd,lnh->ldh", dz, h)
    if "b2" in p:
        g["b2"] += dz.sum(1)
    dpre = np.einsum("lnd,ldh->lnh", dz, p["W2"]) * (1.0 - h * h)
    g["W1"] += np.einsum("lnh,lnx->lhx", dpre, x_all)
    g["b1"] += dpre.sum(1)


def ladder_back(model, fwd, dtop, ds_extra, g):
    """Back through the rungs from the summit to the imagination. dtop: gradient of the top logits."""
    groups, n, beta = model["groups"], fwd["n"], np.exp(model["params"]["theta"])
    dq = None
    for k in reversed(range(len(fwd["rungs"]))):
        rung = fwd["rungs"][k]
        dl = dtop if k == len(fwd["rungs"]) - 1 else softmax_back(dq, rung["p"])
        extra = ds_extra if k == 0 else 0.0
        dq, dbeta = relate_back(dl, extra, rung["cache"], n, groups)
        g["theta"][:, k] += dbeta * beta[:, k]
    imagine_back(model["params"], fwd["x"], fwd["h"], dq, g)


def cross_entropy(logits, y):
    """Per-learner mean cross-entropy and its gradient on the query logits."""
    L, n = y.shape
    li, ni = np.arange(L)[:, None], np.arange(n)[None]
    loss = (logsumexp(logits) - logits[li, ni, y]).mean(1)
    d = softmax(logits)
    d[li, ni, y] -= 1.0
    return float(loss.sum()), d / n


def internal_views(x, anchors, rng):
    """Three internal re-presentations of each percept: common sense (fine noise), imagination (channels
    dropped), memory (drawn toward the remembered typical percept, the mean of one's anchors)."""
    memory = anchors.mean(1, keepdims=True)
    return [x + rng.normal(0.0, 0.1, x.shape), x * (rng.random(x.shape) > 0.25), 0.7 * x + 0.3 * memory]


def loss_and_grads(model, batch):
    """Sum over learners of each learner's own mean loss, so no learner's gradient depends on another's."""
    assert not model["ko"], "gradients are defined for the intact model only"
    p, x, y = model["params"], batch["X"], batch["Y"]
    g = {k: np.zeros_like(v) for k, v in p.items()}
    if model["plain"]:
        h, z = imagine(p, x)
        logits = np.einsum("lnd,lcd->lnc", z, p["V"]) + p["c"][:, None]
        loss, dlog = cross_entropy(logits, y)
        g["V"] += np.einsum("lnc,lnd->lcd", dlog, z)
        g["c"] += dlog.sum(1)
        imagine_back(p, x, h, np.einsum("lnc,lcd->lnd", dlog, p["V"]), g)
    else:
        fwd = ascend(model, x)
        n = x.shape[1]
        loss, dlog = cross_entropy(fwd["rungs"][-1]["logits"][:, :n], y)
        dtop = np.zeros_like(fwd["rungs"][-1]["logits"])
        dtop[:, :n] = dlog
        ladder_back(model, fwd, dtop, 0.0, g)
        if model["coincide"] > 0.0:
            loss += coincidence_term(model, batch["views"], g)
    mut = ACTIVE_MUTANT["name"]
    if mut == "zero_grad_imagination":
        g["W1"] = np.zeros_like(g["W1"])
    if mut == "zero_grad_sharpness" and "theta" in g:
        g["theta"] = np.zeros_like(g["theta"])
    return loss, g


def coincidence_term(model, views, g):
    """The rival's criterion: the first-rung relation profiles of the three internal views must coincide."""
    lam, fwds = model["coincide"], [ascend(model, v) for v in views]
    n = views[0].shape[1]
    s = [f["rungs"][0]["s"][:, :n] for f in fwds]
    mean = sum(s) / 3.0
    agree = sum(((si - mean) ** 2).sum(-1).mean(1) for si in s) / 3.0
    for f, si in zip(fwds, s):
        ds = np.zeros_like(f["rungs"][0]["s"])
        ds[:, :n] = lam * (2.0 / 3.0) * (si - mean) / n
        ladder_back(model, f, np.zeros_like(f["rungs"][-1]["logits"]), ds, g)
    return float(lam * agree.sum())


# ============================================================================ training (each learner alone)
def solitary_clip(grads, max_norm):
    """Gradient clipping per learner. A global norm across learners would let one learner's gradient scale
    another's update: a channel of communication the Tadbir forbids (tested in C6.1)."""
    L = next(iter(grads.values())).shape[0]
    norms = np.sqrt(sum((v.reshape(L, -1) ** 2).sum(1) for v in grads.values()))
    scale = np.minimum(1.0, max_norm / (norms + 1e-12))
    return {k: v * scale.reshape((L,) + (1,) * (v.ndim - 1)) for k, v in grads.items()}


def choose_anchors(model, x, y):
    """Each learner's anchors are its own first percepts of each concept (no percept is ever shared)."""
    L, C = x.shape[0], model["classes"]
    counts = min(int((y[i] == c).sum()) for i in range(L) for c in range(C))
    per = max(1, min(N_ANCHOR, counts))
    idx = np.stack([np.concatenate([np.where(y[i] == c)[0][:per] for c in range(C)]) for i in range(L)])
    model["anchors"] = x[np.arange(L)[:, None], idx]
    model["groups"] = group_matrix(C, per)


def fit(model, data, budget, rng, clip=solitary_clip):
    x, y = data
    if x.ndim == 2:
        x, y = x[None], y[None]
    if model["anchors"] is None:
        choose_anchors(model, x, y)
    L, n = y.shape
    opt, mut, history = Adam(model["params"], lr=LR), ACTIVE_MUTANT["name"], []
    for t in range(1, budget + 1):
        idx = rng.integers(0, n, (L, BATCH))
        batch = {"X": x[np.arange(L)[:, None], idx], "Y": y[np.arange(L)[:, None], idx]}
        if model["coincide"] > 0.0:
            batch["views"] = internal_views(batch["X"], model["anchors"], rng)
        loss, grads = loss_and_grads(model, batch)
        if not math.isfinite(loss):
            raise NonFiniteError(f"non-finite loss at step {t}")
        grads = clip(grads, CLIP)
        grads = grads[0] if isinstance(grads, tuple) else grads
        lr = LR * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * t / budget)))
        opt.step(model["params"], grads, lr=0.0 if mut == "zero_lr" else lr,
                 sign=-1.0 if mut == "sign_flip_update" else 1.0)
        history.append(loss / L)
    if not all(np.all(np.isfinite(v)) for v in model["params"].values()):
        raise NonFiniteError("non-finite parameter after training")
    return history


def exported_code(model, x, upto=None, logits=False):
    """What a solitary hands to a reader: the ladder's summit relation profile, the plain model's
    probabilities (or logits)."""
    if model["plain"]:
        _, z = imagine(model["params"], x)
        lg = np.einsum("lnd,lcd->lnc", z, model["params"]["V"]) + model["params"]["c"][:, None]
        return lg if logits else softmax(lg)
    return ascend(model, x, upto)["rungs"][-1]["s"][:, :x.shape[1]]


def predict(model, x):
    squeeze = x.ndim == 2
    x = x[None] if squeeze else x
    if model["plain"]:
        out = exported_code(model, x)
    else:
        out = ascend(model, x)["rungs"][-1]["p"][:, :x.shape[1]]
    return out[0] if squeeze else out


def hidden_states(model, x):
    x = x[None] if x.ndim == 2 else x
    if model["plain"]:
        h, z = imagine(model["params"], x)
        return {"imagination_hidden": h, "imagination": z}
    fwd = ascend(model, x)
    n = x.shape[1]
    out = {"imagination": fwd["z"][:, :n]}
    out.update({f"rung_{k + 1}": r["s"][:, :n] for k, r in enumerate(fwd["rungs"])})
    return out


class NonFiniteError(RuntimeError):
    pass


# ============================================================================ baselines and rival
def matched_plain_hidden(target_per_learner, in_dim, classes, width=WIDTH):
    """Hidden width that brings the softmax classifier within 10% of the ladder's parameters."""
    return max(2, int(round((target_per_learner - width - classes * width - classes) / (in_dim + 1 + width))))


def roster(task, seed_seq, steps):
    """Train, each learner alone: the ladder (3 rungs), the one-step relative representation (prior art),
    the Sufi rival (one rung trained for coincidence of three internal views) and a softmax classifier."""
    x, y = task["train"]
    L = x.shape[0]
    s_init, s_fit = seed_seq.spawn(2)
    per = n_params(build_model(D_SENSE, N_SPECIES, TASK_TYPES[0], np.random.default_rng(0)))
    specs = {"ladder": {}, "relative": {"rungs": 1}, "sufi": {"rungs": 1, "coincide": COINCIDE},
             "plain": {"plain": True, "hidden": matched_plain_hidden(per, D_SENSE, N_SPECIES)}}
    models, histories = {}, {}
    for name, spec in specs.items():
        m = build_model(D_SENSE, N_SPECIES, TASK_TYPES[0], np.random.default_rng(s_init), learners=L, **spec)
        histories[name] = fit(m, (x, y), steps, np.random.default_rng(s_fit))
        models[name] = m
    return models, histories


# ============================================================================ registries: modules, knockouts, mutants
MUTANTS = {
    "sign_flip_update": "optimizer ascends the loss (update sign flipped)",
    "zero_lr": "learning rate forced to zero",
    "zero_grad_imagination": "analytic gradient of the imagination's first layer zeroed",
    "zero_grad_sharpness": "analytic gradient of the rung sharpness zeroed",
}
KNOCKOUTS = [("ascent", "identity"), ("sharpness", "mean"), ("centering", "identity")]


def modules(model):
    p = model["params"]
    table = {"imagination": {"params": [k for k in ("W1", "b1", "W2", "b2") if k in p],
                             "role": "private tanh encoder of the percept", "signature": False}}
    if model["plain"]:
        table["head"] = {"params": ["V", "c"], "role": "linear softmax head over concepts", "signature": False}
        return table
    table["first_rung"] = {"params": [], "role": "cosine to centred class centroids of own anchors "
                                                "(one-step relative representation)", "signature": False}
    table["sharpness"] = {"params": ["theta"], "role": "per-rung softmax temperature", "signature": False}
    table["centering"] = {"params": [], "role": "subtract the mean centroid before the cosine", "signature": False}
    if model["rungs"] > 1:
        table["ascent"] = {"params": [], "role": "iterated relative re-expression of the code (rungs 2..K)",
                           "signature": True}
    return table


def knockout(model, name, mode):
    """Copy with a module replaced: ascent/identity stops at rung 1; sharpness/mean holds every rung's
    beta at the learner's mean beta; centering/identity drops the centring."""
    if (name, mode) not in KNOCKOUTS:
        raise KeyError((name, mode))
    new = dict(model)
    new["params"] = {k: v.copy() for k, v in model["params"].items()}
    new["ko"] = dict(model["ko"], **{name: mode})
    return new


def n_params(model):
    """Parameters of one solitary (tensors are stacked over learners on axis 0)."""
    return int(sum(v[0].size for v in model["params"].values()))


# ============================================================================ probes: conjunction, the particular, articulation
PERMS = np.array(list(itertools.permutations(range(N_SPECIES))))


def ridge_fit(z, t, lam=1e-2):
    z1 = np.concatenate([z, np.ones((len(z), 1))], 1)
    return np.linalg.solve(z1.T @ z1 + lam * np.eye(z1.shape[1]), z1.T @ t)


def ridge_r2(z, t, w):
    """R2 of a fitted linear reader, floored so that one wild reading cannot dominate a mean."""
    z1 = np.concatenate([z, np.ones((len(z), 1))], 1)
    r2 = 1.0 - float(((z1 @ w - t) ** 2).sum()) / float(((t - t.mean(0)) ** 2).sum())
    return max(r2, MIND_CARD["thresholds"]["conjunction_floor"])


def conjunction(code_fit, code_read, t_fit, t_read, pairs):
    """A reader fitted on learner a's code is applied, unchanged, to learner b's code of other percepts."""
    return float(np.mean([ridge_r2(code_read[b], t_read, ridge_fit(code_fit[a], t_fit)) for a, b in pairs]))


def particular(model, task):
    """Blind-spot probe: within each learner, how well does its exported code report size and age?"""
    (xf, _, pf), (xr, _, pr) = task["pure"]["fit"], task["pure"]["read"]
    cf, cr = exported_code(model, xf), exported_code(model, xr)
    return float(np.mean([ridge_r2(cr[i], pr, ridge_fit(cf[i], pf)) for i in range(len(cf))]))


def accuracy(model, task):
    x, y, _ = task["pure"]["read"]
    return float((predict(model, x).argmax(-1) == y[None]).mean())


def articulation(model):
    """Name-free matching: from each learner's matrix of relations among its own concepts (level 0 = the
    imagination, level k = rung k), find the relabelling of another learner's concepts that best reproduces
    it; report the fraction of concepts matched to the right name, averaged over ordered pairs."""
    L = model["anchors"].shape[0]
    fwd = ascend(model, model["anchors"][:, :0])
    levels = [fwd["z"]] + [r["s"] for r in fwd["rungs"]]
    out = []
    for q in levels:
        m = np.einsum("ca,lad->lcd", model["groups"], q)
        m = m - m.mean(1, keepdims=True)
        m = m / np.sqrt((m ** 2).sum(-1, keepdims=True) + EPS_NORM)
        rel = np.einsum("lcd,led->lce", m, m)
        hits = [(PERMS[((rel[a][None] - rel[b][PERMS[:, :, None], PERMS[:, None, :]]) ** 2).sum((1, 2)).argmin()]
                 == np.arange(N_SPECIES)).mean() for a in range(L) for b in range(L) if a != b]
        out.append(float(np.mean(hits)))
    return out


def run_seed(seed, steps):
    ss = np.random.SeedSequence(seed).spawn(2)
    task = make_task(ss[0])
    models, histories = roster(task, ss[1], steps)
    (xbf, tf), (xbr, tr) = task["blend"]["fit"], task["blend"]["read"]
    shifted = [(a, b) for a in task["old"] for b in task["new"]]
    same = [(a, b) for a in task["old"] for b in task["old"] if a != b]

    def conj(m, pairs=shifted, upto=None, logits=False):
        return conjunction(exported_code(m, xbf, upto, logits), exported_code(m, xbr, upto, logits), tf, tr, pairs)

    r = {"seed": seed, "task": task, "models": models, "history": histories["ladder"]}
    for name, m in models.items():
        r[f"conj_{name}"], r[f"conj_id_{name}"] = conj(m), conj(m, same)
        r[f"part_{name}"], r[f"acc_{name}"] = particular(m, task), accuracy(m, task)
    r["conj_plain_logits"] = conj(models["plain"], logits=True)
    for mod, mode in KNOCKOUTS:
        r[f"ko_{mod}"] = conj(knockout(models["ladder"], mod, mode))
    r["profile"] = [conj(models["ladder"], upto=k) for k in range(1, RUNGS + 1)]
    r["articulation"] = articulation(models["ladder"])
    r["trivial_acc"] = float(np.bincount(task["pure"]["read"][1], minlength=N_SPECIES).max() / N_PROBE)
    return r


# ============================================================================ tests: correctness
def c1_gradcheck(task, seed_seq, tol):
    """Every tensor of the ladder, the Sufi rival (views frozen) and the softmax baseline, two learners each,
    at initialisation and after 60 training steps."""
    x, y = task["train"][0][:2], task["train"][1][:2]
    rng = np.random.default_rng(seed_seq)
    idx = rng.choice(N_TRAIN, 12, replace=False)
    worst, checked, total = 0.0, 0, 0
    for spec in ({}, {"rungs": 1, "coincide": COINCIDE}, {"plain": True, "hidden": 6}):
        m = build_model(D_SENSE, N_SPECIES, TASK_TYPES[0], np.random.default_rng(seed_seq), learners=2, **spec)
        choose_anchors(m, x, y)
        for stage in ("init", "trained"):
            if stage == "trained":
                fit(m, (x, y), 60, np.random.default_rng(1))
            batch = {"X": x[:, idx], "Y": y[:, idx]}
            if m["coincide"]:
                batch["views"] = internal_views(batch["X"], m["anchors"], np.random.default_rng(2))
            _, grads = loss_and_grads(m, batch)
            w, rep = finite_difference_check(lambda: loss_and_grads(m, batch)[0], m["params"], grads, rng)
            worst, checked, total = max(worst, w), checked + len(rep), total + len(m["params"])
    return {"id": "C1", "name": "gradient_check", "passed": bool(worst <= tol),
            "detail": f"{checked}/{total} tensor checks (ladder, sufi rival, softmax; init and after 60 steps), "
                      f"max rel err {worst:.2e}", "max_rel_error": worst, "tensors_checked": checked,
            "tensors_total": total}


def c2_determinism(task, seed_seq):
    runs = []
    for _ in range(2):
        m = build_model(D_SENSE, N_SPECIES, TASK_TYPES[0], np.random.default_rng(seed_seq), learners=N_OLD)
        hist = fit(m, (task["train"][0][:N_OLD], task["train"][1][:N_OLD]), 60, np.random.default_rng(7))
        runs.append((hist, predict(m, task["pure"]["read"][0][:N_OLD]), m))
    same = runs[0][0] == runs[1][0] and np.array_equal(runs[0][1], runs[1][1])
    states = hidden_states(runs[0][2], task["pure"]["read"][0][:N_OLD])
    finite = all(np.all(np.isfinite(v)) for v in list(runs[0][2]["params"].values()) + list(states.values()))
    return {"id": "C2", "name": "determinism_and_finiteness", "passed": bool(same and finite),
            "detail": f"identical losses/outputs={same}, finite parameters and rung activations={finite}"}


def c3_learning(run, th):
    h = run["history"]
    drop = 1.0 - float(np.mean(h[-50:])) / float(np.mean(h[:50]))
    err, triv = 1.0 - run["acc_ladder"], 1.0 - run["trivial_acc"]
    ok = drop >= th["loss_drop_fraction"] and err <= (1.0 - th["margin_over_trivial"]) * triv
    return {"id": "C3", "name": "learning", "passed": bool(ok),
            "detail": f"ladder loss drop {drop:.3f} (>= {th['loss_drop_fraction']}); held-out error {err:.3f} vs "
                      f"majority {triv:.3f} (must be <= {1 - th['margin_over_trivial']:.1f}x)"}


def short_learning_ok(task, seed_seq):
    m = build_model(D_SENSE, N_SPECIES, TASK_TYPES[0], np.random.default_rng(seed_seq), learners=2)
    hist = fit(m, (task["train"][0][:2], task["train"][1][:2]), STEPS_SHORT, np.random.default_rng(3))
    return 1.0 - float(np.mean(hist[-50:])) / float(np.mean(hist[:50])) >= 0.5


def c4_shuffled(task, seed_seq, steps, band):
    """Each solitary's labels permuted: held-out accuracy must stay within the majority baseline's band."""
    x, y = task["train"]
    rng = np.random.default_rng(seed_seq)
    ys = np.stack([rng.permutation(row) for row in y])
    m = build_model(D_SENSE, N_SPECIES, TASK_TYPES[0], np.random.default_rng(seed_seq), learners=len(x))
    fit(m, (x, ys), steps, np.random.default_rng(5))
    acc = accuracy(m, task)
    triv = float(np.bincount(task["pure"]["read"][1], minlength=N_SPECIES).max() / N_PROBE)
    return {"id": "C4", "name": "shuffled_label_control", "passed": bool(acc <= triv + band),
            "detail": f"held-out accuracy with permuted labels {acc:.3f} vs majority {triv:.3f} (band +{band})"}


def c5_mutants(task, seed_seq, tol):
    """Each learning-breaking mutant must make C1 or the short learning check fail."""
    genuine = short_learning_ok(task, seed_seq)
    detected = {}
    for name in MUTANTS:
        ACTIVE_MUTANT["name"] = name
        try:
            c1_ok = c1_gradcheck(task, seed_seq, tol)["passed"] if name.startswith("zero_grad") else True
            learn_ok = c1_ok and short_learning_ok(task, seed_seq)
        except (NonFiniteError, FloatingPointError):
            c1_ok, learn_ok = True, False
        finally:
            ACTIVE_MUTANT["name"] = None
        detected[name] = not (c1_ok and learn_ok)
    n = sum(detected.values())
    return {"id": "C5", "name": "mutant_detection", "passed": bool(genuine and n == len(detected)),
            "detail": f"genuine short-learning check passes={genuine}; detected {detected}",
            "detected": int(n), "total": len(detected)}


def c6_1_solitude(task, seed_seq):
    """No channel between solitaries: perturbing one learner's percepts leaves every other learner's trained
    parameters bit-identical. Tight clipping (0.05) makes clipping active at every step. Negative control:
    the standard global-norm clip couples the learners, and must break the identity."""
    x, y = task["train"]
    rng = np.random.default_rng(seed_seq)

    def train(xx, clip):
        m = build_model(D_SENSE, N_SPECIES, TASK_TYPES[0], np.random.default_rng(seed_seq), learners=len(xx))
        fit(m, (xx, y), 80, np.random.default_rng(4), clip=clip)
        return m["params"]

    solo, glob = (lambda g, n: solitary_clip(g, 0.05)), (lambda g, n: clip_global(g, 0.05))
    ok, leak = True, True
    for j in (0, len(x) - 1):
        xp = x.copy()
        xp[j] = xp[j] + rng.normal(0.0, 1.0, xp[j].shape)
        others = [i for i in range(len(x)) if i != j]
        a, b = train(x, solo), train(xp, solo)
        ok &= all(np.array_equal(a[k][others], b[k][others]) for k in a)
        ok &= any(not np.array_equal(a[k][j], b[k][j]) for k in a)
        ga, gb = train(x, glob), train(xp, glob)
        leak &= any(not np.array_equal(ga[k][others], gb[k][others]) for k in ga)
    return {"id": "C6.1", "name": "solitude", "passed": bool(ok and leak),
            "detail": f"other learners bit-identical under solitary clipping={ok}; global-norm control leaks={leak}"}


def settled(q, rung, n_rungs):
    """Per draw: after n_rungs, is the code finite, non-vanishing and unchanged (relative 1e-6) by one more?"""
    with np.errstate(all="ignore"):
        for _ in range(n_rungs):
            q_prev, q = q, rung(q)
        size = np.sqrt((q ** 2).sum((1, 2)))
        change = np.sqrt(((q - q_prev) ** 2).sum((1, 2))) / (size + 1e-300)
    return np.isfinite(size) & (size > 1e-9) & (change < 1e-6)


def c6_2_termination(seed_seq, draws=200, n_rungs=300, n_query=40):
    """The regress of quiddities ends: iterating the rung on its own output from random codes, scales and
    sharpnesses reaches a code that one more rung leaves unchanged. Negative control: re-expression by raw
    dot products with the centred centroids (no ratio, no normalisation) explodes or vanishes."""
    rng = np.random.default_rng(seed_seq)
    groups = group_matrix(N_SPECIES, N_ANCHOR)
    q0 = rng.normal(size=(draws, n_query + groups.shape[1], WIDTH)) * np.exp(rng.uniform(-2, 2, (draws, 1, 1)))
    beta = np.exp(rng.uniform(math.log(0.3), math.log(50.0), draws))

    def raw(q):
        m = np.einsum("ca,lad->lcd", groups, q[:, n_query:])
        return np.einsum("lnd,lcd->lnc", q, m - m.mean(1, keepdims=True))

    done = settled(q0, lambda q: relate(q, n_query, groups, beta)[2], n_rungs).mean()
    raw_done = settled(q0, raw, n_rungs).mean()
    return {"id": "C6.2", "name": "regress_terminates", "passed": bool(done >= 0.95 and raw_done <= 0.05),
            "detail": f"ladder settled in {done:.3f} of {draws} random draws within {n_rungs} rungs (>= 0.95); "
                      f"raw-product control settled in {raw_done:.3f} (<= 0.05)"}


def c6_3_private_frame(seed_seq, trials=6, searches=25):
    """Private-frame invariance on random parameters: shifting, rescaling and rotating a learner's latent
    (its private imagination) leaves every rung unchanged. Random search over transforms keeps the worst.
    Negative control: without the centring, a shift of the latent must change the rungs."""
    rng = np.random.default_rng(seed_seq)
    worst, control = 0.0, np.inf
    for _ in range(trials):
        m = build_model(D_SENSE, N_SPECIES, TASK_TYPES[0], rng, learners=2)
        m["params"]["theta"] = rng.normal(math.log(BETA0), 1.0, m["params"]["theta"].shape)
        m["anchors"], m["groups"] = rng.normal(size=(2, N_SPECIES * N_ANCHOR, D_SENSE)), group_matrix(N_SPECIES, N_ANCHOR)
        x = rng.normal(size=(2, 30, D_SENSE))
        base = ascend(m, x)["rungs"]
        for _ in range(searches):
            a, b = math.exp(rng.uniform(-3, 3)), rng.normal(0.0, 10.0 ** rng.uniform(-1, 2), WIDTH)
            rot, _ = np.linalg.qr(rng.normal(size=(WIDTH, WIDTH)))
            moved = ascend(m, x, z_map=lambda z: a * z @ rot.T + b)["rungs"]
            worst = max(worst, max(float(np.abs(u["s"] - v["s"]).max()) for u, v in zip(base, moved)))
        ko = knockout(m, "centering", "identity")
        shift = rng.normal(0.0, 3.0, WIDTH)
        c0, c1 = ascend(ko, x)["rungs"], ascend(ko, x, z_map=lambda z: z + shift)["rungs"]
        control = min(control, max(float(np.abs(u["s"] - v["s"]).max()) for u, v in zip(c0, c1)))
    return {"id": "C6.3", "name": "private_frame_invariance", "passed": bool(worst <= 1e-8 and control > 1e-3),
            "detail": f"worst rung change under shift/scale/rotation {worst:.1e} (<= 1e-8); uncentred control "
                      f"{control:.2e} (> 1e-3)"}


def definition_check(run):
    m, x = run["models"]["ladder"], run["task"]["blend"]["read"][0]
    same = np.array_equal(exported_code(knockout(m, "ascent", "identity"), x), exported_code(m, x, upto=1))
    return {"id": "DEF", "name": "ascent_knockout_equals_first_rung (definition check, not evidence)",
            "passed": bool(same), "detail": f"identical codes: {same}"}


def c7_split(task):
    def rows(a):
        return {hashlib.sha1(np.ascontiguousarray(r).tobytes()).hexdigest() for r in a.reshape(-1, a.shape[-1])}
    tr = rows(task["train"][0])
    evals = [rows(task[k][s][0]) for k in ("blend", "pure") for s in ("fit", "read")]
    overlap = sum(len(tr & e) for e in evals) + len(evals[0] & evals[1]) + len(evals[2] & evals[3])
    return {"id": "C7", "name": "split_integrity", "passed": overlap == 0,
            "detail": f"shared rows between training, reader-fitting and reading sets: {overlap}"}


# ============================================================================ tests: hypotheses
HYP_DIFFS = {"H-SIG": lambda r: r["conj_ladder"] - r["conj_relative"],
             "H-NEC": lambda r: r["ko_ascent"] - r["ko_sharpness"],
             "H-BLIND": lambda r: r["part_ladder"] - r["part_relative"],
             "H-RIVAL": lambda r: r["conj_ladder"] - r["conj_sufi"]}
NAMES = ("ladder", "relative", "sufi", "plain")


def hypothesis_table(runs, quick, rng):
    rows = []
    for h in MIND_CARD["hypotheses"]:
        mean, ci = paired_bootstrap([HYP_DIFFS[h["id"]](r) for r in runs], rng)
        rows.append({"id": h["id"], "metric": h["metric"], "mean_diff": mean, "ci95": ci, "mesi": h["mesi"],
                     "n_seeds": len(runs), "direction": h["direction"],
                     "verdict": "not evaluated" if quick else verdict(mean, ci, h["direction"], h["mesi"])})
    return rows


def knockout_table(runs, rng):
    sig = {k: v["signature"] for k, v in modules(runs[0]["models"]["ladder"]).items()}
    rows = []
    for mod, mode in KNOCKOUTS:
        mean, ci = paired_bootstrap([r[f"ko_{mod}"] - r["conj_ladder"] for r in runs], rng)
        rows.append({"module": mod, "mode": mode, "signature": sig[mod], "metric_change": mean, "ci95": ci})
    return rows


# ============================================================================ report
def summary(runs):
    keys = [f"{m}_{n}" for m in ("acc", "conj_id", "conj", "part") for n in NAMES] + ["conj_plain_logits"]
    out = {k: (float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))) for k in keys}
    out["profile"] = np.mean([r["profile"] for r in runs], 0).round(3).tolist()
    out["articulation"] = np.mean([r["articulation"] for r in runs], 0).round(3).tolist()
    out["beta"] = np.mean([np.exp(r["models"]["ladder"]["params"]["theta"]).mean(0) for r in runs], 0).round(2).tolist()
    return out


def print_report(rep, s):
    line = "-" * 78
    print(f"=== VERIFIED REPORT · chapter {CHAPTER:04d} ===")
    print(f"file {rep['file']}  card_revision {rep['card_revision']}  python {rep['environment']['python']}  "
          f"numpy {rep['environment']['numpy']}  mutant {rep['mutant']}")
    print(f"seeds {rep['seeds']}  runtime {rep['runtime_s']:.1f} s  parameters per solitary: "
          + ", ".join(f"{k} {v}" for k, v in rep["n_params_roster"].items()))
    g = rep["gradcheck"]
    print(f"gradient check: {g['tensors_checked']}/{g['tensors_total']} tensor checks at {g['checked_at']}, "
          f"max rel err {g['max_rel_error']:.2e}, passed={g['passed']}")
    print(line + "\ncorrectness")
    for c in rep["correctness"]:
        print(f"  {c['id']:<5} {'PASS' if c['passed'] else 'FAIL'}  {c['name']}: {c['detail']}")
    m = rep["mutants"]
    print(f"mutation score {m['detected']}/{m['total']} = {m['score']:.2f}")
    print(line + "\nmean (sd) over seeds; conjunction = R2 of a blend reader fitted on one solitary, read on another")
    print(f"  {'model':<10}{'accuracy':>16}{'conj old-old':>18}{'conj old-new':>18}{'particular R2':>18}")
    for n in NAMES:
        print(f"  {n:<10}" + "".join(f"{s[f'{k}_{n}'][0]:>10.3f} ({s[f'{k}_{n}'][1]:.3f})"
                                     for k in ("acc", "conj_id", "conj", "part")))
    print(f"  softmax logits instead of probabilities, conj old-new: {s['conj_plain_logits'][0]:.3f}")
    print(f"  ascent profile (conj old-new read at rung 1..{RUNGS}): {s['profile']}")
    print(f"  name-free concept matching (imagination, rung 1..{RUNGS}): {s['articulation']}")
    print(f"  learned rung sharpness beta (mean over learners and seeds): {s['beta']}")
    print(line + "\nhypotheses (paired per-seed differences, 95% bootstrap CI, 2000 resamples)")
    for h in rep["hypotheses"]:
        print(f"  {h['id']:<8} {h['metric']:<15} diff {h['mean_diff']:+.3f}  CI [{h['ci95'][0]:+.3f}, "
              f"{h['ci95'][1]:+.3f}]  mesi {h['mesi']} ({h['direction']})  -> {h['verdict']}")
    print(line + "\nknockouts on the trained ladder (change in conj old-new)")
    for k in rep["knockouts"]:
        print(f"  {k['module']:<10} {k['mode']:<9} signature={str(k['signature']):<5} {k['metric_change']:+.3f} "
              f"[{k['ci95'][0]:+.3f}, {k['ci95'][1]:+.3f}]")
    print(f"task types {rep['task_types']}  exit code {rep['exit_code']}")
    print("=== END REPORT ===")


def real_data_bridge(path, seed):
    """CSV with a header row, numeric features and an integer class label in the last column (for example
    UCI HAR). Two solitaries see the features through private random maps; readers fitted on one's code
    are read on the other's. Reports reader accuracy for the ladder and the one-step relative code."""
    if not path:
        print("no --data given: real-data bridge skipped")
        return None
    raw = np.genfromtxt(path, delimiter=",", skip_header=1)
    raw = raw[np.all(np.isfinite(raw), axis=1)]
    rng = np.random.default_rng(seed)
    raw = raw[rng.permutation(len(raw))]
    labels = np.unique(raw[:, -1], return_inverse=True)[1]
    feats = (raw[:, :-1] - raw[:, :-1].mean(0)) / (raw[:, :-1].std(0) + 1e-9)
    senses = [np.tanh(feats @ rng.normal(0.0, 1.0 / math.sqrt(feats.shape[1]), (feats.shape[1], D_SENSE)))
              for _ in range(2)]
    x = np.stack(senses)
    cut, half = int(0.6 * len(raw)), int(0.8 * len(raw))
    onehot = np.eye(labels.max() + 1)[labels]
    out = {}
    for name, rungs in (("ladder", RUNGS), ("relative", 1)):
        m = build_model(D_SENSE, labels.max() + 1, TASK_TYPES[0], rng, learners=2, rungs=rungs)
        fit(m, (x[:, :cut], np.stack([labels[:cut]] * 2)), STEPS_QUICK, rng)
        cf, cr = exported_code(m, x[:, cut:half]), exported_code(m, x[:, half:])
        w = ridge_fit(cf[0], onehot[cut:half])
        z1 = np.concatenate([cr[1], np.ones((len(cr[1]), 1))], 1)
        out[name] = float(((z1 @ w).argmax(1) == labels[half:]).mean())
    print(f"real-data bridge on {os.path.basename(path)}: cross-solitary reader accuracy {out}")
    return out


# ============================================================================ command-line entry point
def correctness_suite(args, runs, ss, steps, th):
    task = runs[0]["task"]
    tests = [c1_gradcheck(task, ss[0], th["gradcheck_rel_tol"]), c2_determinism(task, ss[1]),
             c3_learning(runs[0], th), c4_shuffled(task, ss[2], steps, th["shuffled_band"])]
    if args.mutant:
        tests.append({"id": "C5", "name": "mutant_detection", "passed": True,
                      "detail": f"skipped: running with mutant {args.mutant}", "detected": 0, "total": 0})
    else:
        tests.append(c5_mutants(task, ss[3], th["gradcheck_rel_tol"]))
    tests += [c6_1_solitude(task, ss[4]), c6_2_termination(ss[5]), c6_3_private_frame(ss[6]),
              definition_check(runs[0]), c7_split(task)]
    return tests


def protocol(args):
    t0 = time.time()
    quick = args.quick
    n_seeds = args.seeds if args.seeds is not None else (1 if quick else 5)
    steps, budget = (STEPS_QUICK, BUDGET_QUICK) if quick else (STEPS_FULL, BUDGET_FULL)
    seeds = [args.seed + i for i in range(n_seeds)]
    print(f"{os.path.basename(__file__)}: {'quick' if quick else 'full'} protocol, seeds {seeds}, {steps} steps")
    ACTIVE_MUTANT["name"] = args.mutant
    runs = [run_seed(s, steps) for s in seeds]
    th = MIND_CARD["thresholds"]
    tests = correctness_suite(args, runs, np.random.SeedSequence(args.seed + 7919).spawn(7), steps, th)
    rng = np.random.default_rng(args.seed + 104729)
    hyps, kos = hypothesis_table(runs, quick, rng), knockout_table(runs, rng)
    real_data_bridge(args.data, args.seed)
    runtime = time.time() - t0
    tests.append({"id": "C8", "name": "budget", "passed": runtime <= budget,
                  "detail": f"{runtime:.1f} s (<= {budget:.0f} s)"})
    exit_code = 0 if all(t["passed"] for t in tests) else (3 if runtime > budget and all(
        t["passed"] for t in tests[:-1]) else 1)
    c1, c5 = tests[0], next(t for t in tests if t["id"] == "C5")
    rep = {"schema_version": "1.0", "chapter": CHAPTER, "file": os.path.basename(__file__),
           "card_revision": MIND_CARD["card_revision"],
           "environment": {"python": sys.version.split()[0], "numpy": np.__version__}, "seeds": seeds,
           "runtime_s": runtime, "n_params": n_params(runs[0]["models"]["ladder"]),
           "n_params_roster": {k: n_params(m) for k, m in runs[0]["models"].items()},
           "gradcheck": {"tensors_checked": c1["tensors_checked"], "tensors_total": c1["tensors_total"],
                         "max_rel_error": c1["max_rel_error"], "checked_at": ["init", "after_training_steps"],
                         "passed": c1["passed"]},
           "correctness": [{k: t[k] for k in ("id", "name", "passed", "detail")} for t in tests],
           "mutants": {"detected": c5["detected"], "total": c5["total"],
                       "score": c5["detected"] / c5["total"] if c5["total"] else 0.0},
           "hypotheses": hyps, "knockouts": kos, "task_types": TASK_TYPES, "exit_code": exit_code,
           "mutant": args.mutant}
    print_report(rep, summary(runs))
    if args.json:
        write_report(args.json, rep)
    return exit_code


def main(argv=None):
    parser = argparse.ArgumentParser(prog=os.path.basename(__file__),
                                     description="Ladder of quiddities. Without flags the full protocol runs.")
    parser.add_argument("--quick", action="store_true", help="one seed, reduced steps, all correctness tests")
    parser.add_argument("--seed", type=int, default=0, help="base seed")
    parser.add_argument("--seeds", type=int, default=None, help="number of seeds (default 5; 1 with --quick)")
    parser.add_argument("--json", type=str, default=None, help="also write the JSON report here")
    parser.add_argument("--card", action="store_true", help="print MIND_CARD as JSON and exit")
    parser.add_argument("--mutant", type=str, default=None, help="run with a registered mutant")
    parser.add_argument("--data", type=str, default=None, help="optional local CSV for the real-data bridge")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 2 if exc.code else 0
    if args.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    if (args.mutant is not None and args.mutant not in MUTANTS) or (args.seeds is not None and args.seeds < 1):
        print(f"invalid usage; registered mutants: {', '.join(MUTANTS)}; --seeds must be >= 1", file=sys.stderr)
        return 2
    try:
        return protocol(args)
    except (NonFiniteError, FloatingPointError, np.linalg.LinAlgError) as err:
        print(f"non-finite values: {err}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())

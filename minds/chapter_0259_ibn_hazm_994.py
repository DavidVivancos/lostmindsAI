#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0259 · Ibn Hazm of Cordoba
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 13 Minds 241 - 260 Available on Amazon https://www.amazon.com/dp/B0HLD1S6QT
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0259_ibn_hazm_994 - Ibn Hazm of Cordoba (c.994-1064)
#================================================================================  
# END ATTRIBUTION
"""
Thesis
------
A rule reaches exactly as far as its words reach: a mind may extend a verdict
only by dalil (the deductive closure of what the texts and the language
together state), never by qiyas (transfer to an unmentioned case through a
supposed shared cause), and whatever no text reaches keeps its original,
positive status.

Evidence and provenance
-----------------------
Provenance: belief (the figure's own surviving works). Evidence IDs are the
ones used in MIND_CARD.
  D1  al-Ihkam fi usul al-ahkam, the chapters refuting qiyas; al-Nubdha
      al-kafiya. Extension only by dalil; qiyas rejected. (primary)
  D2  al-Ihkam, the chapter on causes ('ilal) in religion; Sabra 2007.
      Commands carry no discoverable cause from which to extend them; a
      general term applies to every instance it names. (primary)
  D3  al-Ihkam and al-Muhalla on istishab al-hal and original permissibility:
      what no text forbids stays permitted. (primary)
  D4  al-Taqrib li-hadd al-mantiq (ed. Ihsan Abbas 1959; Chejne 1984): the
      first sources of knowledge are the soundly used senses, the intuitions
      of reason and a correct understanding of language; a truth cannot be
      truer than another truth. (primary)
  D5  Arnaldez 1956; Osman 2014: language is a fixed convention prior to the
      law, and the ruling one wants may not move the meaning of a word.
      (scholarship)
  D6  Tawq al-hamama (Leiden Or. 927): love is known from testimony,
      observation and memory; signs are read off behaviour. (primary; not
      mechanised here)
  D7  al-Akhlaq wa-l-siyar: the one aim all people share is to dispel
      anxiety. (primary; not mechanised here)
  D8  Adang 2001, Fierro 2021: the conversions Maliki -> Shafi'i -> Zahiri and
      the withdrawal from the courts. (scholarship)

Doctrine -> mechanism -> test
-----------------------------
  D1  -> M2 dalil closure (Boolean forward chaining over the given texts)
      -> C6.1 monotone in the text set, C6.2 order-free, C6.3 fixpoint; H-SIG
  D2  -> M2 full 'umum: a text on a term covers every member of the term
      -> H-SIG on the confounded, shifted split
  D3  -> M4 positive default for uncovered cases; M3 precedents apply only to
      identical name patterns -> definition checks; H-BLIND
  D4  -> M1 bivalent lexicon: names are crisp and are the only carrier of a
      verdict, so verdict labels cannot reach word meaning -> C1 (the usage
      loss is the only gradient path), C4 (shuffled usage labels)
  D5  -> the baseline TawilLearner lets verdict gradients reshape the shared
      encoder (ta'wil); the reader does not -> H-SIG, H-NEC
  D8  -> corpus links to 0197 (qiyas + istihsan) and 0010 (nearest precedent)
      as rivals -> H-RIVAL

Research question
-----------------
Category: robustness to spurious correlation (with compositional
generalisation). When the decided cases are few and confounded but the
community's usage of words is broad, does a learner that acquires concepts
from usage and reads its rules literally generalise better under shift than
a learner that induces the rule from the decided cases, and where does it
fail?

Closest prior art and the delta
-------------------------------
Concept bottleneck models trained independently (Koh et al. 2020),
neuro-symbolic concept learning (Mao et al. 2019) and differentiable forward
chaining (Evans and Grefenstette 2018; Rocktaschel and Riedel 2017) share
the parts. The delta: the label layer is a fixed Boolean program READ from
given texts rather than learned, concept membership is bivalent so verdict
data cannot alter word meaning, precedents apply only to identical name
patterns, and uncovered cases fall to a positive default. The baseline is
the same encoder and term heads with a learned verdict head (multi-task
MLP); the rival is a Nadaraya-Watson kernel over decided cases.

Blind spot
----------
Where a rule has a real but unstated cause (the ratio legis the analogists
looked for), the reader forbids by name what the cause permits and permits
by default what the cause forbids. Ibn Hazm denied that such causes can be
known; the blind-spot world makes one true.

Task
----
Things are described by eight binary attributes (fermented, grape, date,
honey, intoxicating, liquid, red, sold in market) plus four Gaussian
nuisance channels, each attribute flipped with probability 0.05. Eight
terms have fixed lexical definitions (khamr = fermented and grape, muskir =
intoxicating, ...). Usage data pair a thing with the terms the community
calls it, with 4 percent misnaming. World A gives three texts (every khamr
is forbidden; every muskir is khamr; every nabidh is sharab) and decides
verdict-training cases from a confounded pool in which the only forbidden
thing is red grape wine, so fermented, intoxicating and red are each
sufficient in training. The shifted split holds honey mead, date wine, a
solid intoxicant, red must, red juice, weak grape wine and red dye, where no
single attribute is sufficient. World B keeps only the khamr text while the
true verdict follows intoxication; its blind-spot split holds novel
intoxicants and a non-intoxicating khamr.

Limits
------
Synthetic data only. The reader ignores verdict labels except as exact
precedents, so few-shot verdict learning is out of scope. Nothing here is
legal or religious advice; the alcohol example is a lexical case study. The
file is a research prototype of an AGI-oriented architecture, not an AGI.

Run: python3 <this file>   (flags: --quick --seed S --seeds N --json PATH
--card --mutant NAME --data PATH)
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 3,
    "revision_log": [
        "rev 2 (2026-09-17): after the first full run, C3 failed because the verdict pools were sampled by prototype count, leaving a majority rate near 0.8 and less headroom than the declared margin of 0.20; verdict pools are now sampled with the two classes at equal probability. Hypotheses, metrics, thresholds and seeds are unchanged.",
        "rev 3 (2026-09-17): seed sweeps showed two flaky checks. C4: with sixteen discrete prototypes a single label permutation lands anywhere between the two constant predictors, so the control now averages 10 fresh permutations (6 in quick mode) instead of 5 (3). C1: the baseline's verdict bias has a gradient near 5e-6 on balanced classes, so its relative error was rounding noise; the checker's denominator is floored at 1e-4 (absolute terms for vanished gradients). Hypotheses, metrics, thresholds and seeds are unchanged.",
    ],
    "generation": {"template_version": "1.0", "generator": "Claude Fable 5.1 (Anthropic)",
                   "generator_version": "claude-fable-5-1", "date": "2026-09-17"},
    "id": 259, "figure": "Ibn Hazm", "born": 994, "died": 1064,
    "civilization": "Andalusi (Umayyad Cordoba and the Taifa period; Arabic-writing; the family claimed a Persian client lineage and is thought by scholars to be of Iberian convert origin)",
    "provenance": "belief",
    "thesis": "A rule reaches exactly as far as its words reach: verdicts extend only by the deductive closure of the texts and the language, never by transfer through a supposed shared cause, and what no text reaches keeps its original positive status.",
    "evidence": [
        {"id": "D1", "claim": "Extension of a ruling only by dalil; qiyas rejected", "basis": "primary", "source": "al-Ihkam fi usul al-ahkam, chapters refuting qiyas; al-Nubdha al-kafiya"},
        {"id": "D2", "claim": "Commands carry no discoverable cause; a general term applies to every instance it names", "basis": "primary", "source": "al-Ihkam, chapter on causes in religion; Sabra, al-Qantara 28 (2007)"},
        {"id": "D3", "claim": "What no text forbids stays permitted (istishab, original permissibility)", "basis": "primary", "source": "al-Ihkam; al-Muhalla"},
        {"id": "D4", "claim": "Knowledge from senses, intuitions of reason and correct understanding of language; a truth cannot be truer than another truth", "basis": "primary", "source": "al-Taqrib li-hadd al-mantiq, ed. Ihsan Abbas (1959); Chejne, JAOS 104 (1984)"},
        {"id": "D5", "claim": "Language is a fixed convention prior to law; the wanted ruling may not move a word's meaning", "basis": "scholarship", "source": "Arnaldez, Grammaire et theologie chez Ibn Hazm (1956); Osman, The Zahiri Madhhab (2014)"},
        {"id": "D6", "claim": "Love is known from testimony, observation and memory; signs are read off behaviour", "basis": "primary", "source": "Tawq al-hamama, Leiden Or. 927; Abd Alghani, Acta Orientalia 69 (2016)"},
        {"id": "D7", "claim": "The one aim all people share is to dispel anxiety", "basis": "primary", "source": "al-Akhlaq wa-l-siyar, trans. Abu Laylah (1990)"},
        {"id": "D8", "claim": "Conversions from Malikism through Shafi'ism to Zahirism; withdrawal from the courts", "basis": "scholarship", "source": "Adang (2001); Fierro (2021)"},
    ],
    "research_question": {"category": "robustness to spurious correlation",
                          "question": "When decided cases are few and confounded but word usage is broad, does concept-from-usage plus literal rule reading beat rule induction from cases under shift, and where does it fail?"},
    "mechanism": {
        "name": "Zahir-Dalil reader",
        "family": "neuro-symbolic rule reading: usage-trained bivalent concept lexicon, Boolean forward chaining over given texts, exact-pattern precedents, positive closed-world default",
        "signature_modules": ["lexicon", "dalil", "ruling_table", "default"],
        "closest_prior_art": ["Concept bottleneck models, independent training (Koh et al., ICML 2020)",
                              "Neuro-Symbolic Concept Learner (Mao et al., ICLR 2019)",
                              "Differentiable ILP / forward chaining (Evans and Grefenstette, JAIR 2018)",
                              "Neural Theorem Provers (Rocktaschel and Riedel, NeurIPS 2017)",
                              "Multi-task MLP (baseline); Nadaraya-Watson kernel classifier (rival)"],
        "overlap": "Medium",
        "prior_art_queries": ["concept bottleneck independent training rule layer fixed",
                              "neuro-symbolic rule following concepts learned from language",
                              "differentiable forward chaining closed world default",
                              "literal rule application versus similarity generalization spurious correlation"],
        "contribution_type": "mechanism",
        "delta": "The label layer is read from texts, not learned; concept membership is bivalent so verdict labels cannot reach word meaning; precedents bind only identical name patterns; uncovered cases take a positive default; tested against analogical extension on a confounded split.",
    },
    "traceability": [
        {"doctrine": "D1", "mechanism": "M2", "property_test": "C6.1;C6.2;C6.3", "hypothesis": "H-SIG"},
        {"doctrine": "D2", "mechanism": "M2", "property_test": "C6.1", "hypothesis": "H-SIG"},
        {"doctrine": "D3", "mechanism": "M3;M4", "property_test": "C6.def", "hypothesis": "H-BLIND"},
        {"doctrine": "D4", "mechanism": "M1", "property_test": "C1;C4", "hypothesis": "H-NEC"},
        {"doctrine": "D5", "mechanism": "M1", "property_test": "C4", "hypothesis": "H-SIG"},
        {"doctrine": "D8", "mechanism": "rival", "property_test": "-", "hypothesis": "H-RIVAL"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "On the confounded-then-shifted split the reader beats the size-matched ta'wil learner", "metric": "verdict accuracy", "split": "shifted", "comparison": "model - baseline", "direction": "greater", "mesi": 0.10, "seeds": 5},
        {"id": "H-NEC", "statement": "Knocking out the dalil closure hurts more than knocking out the matched deep encoder layer", "metric": "verdict accuracy", "split": "shifted", "comparison": "signature_knockout - matched_knockout", "direction": "less", "mesi": 0.10, "seeds": 5},
        {"id": "H-BLIND", "statement": "Where the verdict follows a real but unstated cause the reader loses to the ta'wil learner", "condition": "world B blind-spot split", "grounding": "Ibn Hazm denied that rulings have knowable causes", "metric": "verdict accuracy", "comparison": "model - baseline", "direction": "less", "mesi": 0.10, "seeds": 5},
        {"id": "H-RIVAL", "statement": "On the shifted split the reader beats extension by similarity of decided cases (qiyas kernel)", "metric": "verdict accuracy", "split": "shifted", "comparison": "model - rival", "direction": "greater", "mesi": 0.10, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.5, "margin_over_trivial": 0.20, "trivial_band": 0.10, "gradcheck_tol": 1e-5},
    "probe_predictions": [
        {"probe": "P3", "expected": "above baseline"}, {"probe": "P2", "expected": "equal to baseline"},
        {"probe": "P6", "expected": "equal to baseline"}, {"probe": "P8", "expected": "below baseline"},
        {"probe": "P9", "expected": "above baseline"},
    ],
    "dialectic_links": [
        {"chapter": 197, "relation": "rival", "test": "H-RIVAL"},
        {"chapter": 10, "relation": "rival", "test": "H-RIVAL"},
        {"chapter": 349, "relation": "successor", "test": "none (recorded: 0349 equates syllogism and analogy; this file keeps them apart)"},
        {"chapter": 257, "relation": "rival", "test": "none (recorded: 0257 leaps to an unstated middle term; here the middle term must be stated)"},
    ],
    "corpus_neighbors": [
        {"chapter": 257, "similarity": 0.0, "difference": "Ibn Sina's hads guesses the middle term; here a middle term must be written in a text or the case falls to the default"},
        {"chapter": 197, "similarity": 0.0, "difference": "Abu Hanifa extends by shared effective cause and overrides by istihsan; here extension by cause is the rival, and there is no override"},
        {"chapter": 10, "similarity": 0.0, "difference": "Hammurabi decides by weighted nearest precedents; here a precedent binds only an identical name pattern"},
        {"chapter": 163, "similarity": 0.0, "difference": "Dignaga makes silence the default; here the default is a positive verdict (permitted), never abstention"},
        {"chapter": 229, "similarity": 0.0, "difference": "Ibn Duraid builds the lexicon as a census of usage; here the lexicon is learned from usage and sealed against the law"},
        {"chapter": 245, "similarity": 0.0, "difference": "al-Majriti divides under a distorting projection; here nothing is projected, coverage is Boolean"},
    ],
    "barometer": {
        "cognitive_processing": ["P3 compositional split (multi-hop closure)", "P10 sample efficiency"],
        "embodied_cognition": [], "world_modeling": ["P9 perturbation robustness"],
        "consciousness": ["P8 calibration: predicted weak (crisp verdicts)"],
        "language_understanding": ["P3 lexicon learned from usage as a synthetic grammar"],
        "emotional_intelligence": [], "creativity": [],
        "autonomy": ["positive default: acts without a text rather than freezing"],
    },
    "task_types": ["vector_classification"],
    "applications": [
        {"use": "Apply written eligibility and compliance rules by their stated terms, refusing extension to similar but unnamed cases", "sector": "legal and regulatory technology", "dataset": "LegalBench (Guha et al., 2023) rule-application tasks"},
        {"use": "Cohort selection from written trial criteria as research decision support", "sector": "clinical research", "dataset": "n2c2 2018 Track 1 cohort selection (public with data-use agreement)"},
        {"use": "Policy-as-text moderation with concept classifiers learned from usage and rules read from the policy", "sector": "trust and safety", "dataset": "Jigsaw Unintended Bias in Toxicity Classification (Kaggle, 2019)"},
    ],
    "safety_notes": "Synthetic data only. The wine example is a lexical case study; outputs are not legal or religious advice. No claim of replicating a person; no generated sentences presented as his words.",
}

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

CHAPTER = 259
BUDGET_FULL_S = 180.0
BUDGET_QUICK_S = 20.0
CLIP_NORM = 5.0
N_TERMS = 8
N_SEMANTIC = 8
N_NUISANCE = 4
IN_DIM = N_SEMANTIC + N_NUISANCE
TERM_NAMES = ["sharab", "khamr", "nabidh", "muskir", "khall", "tamr", "asir", "sibgh"]
ATTR_NAMES = ["fermented", "grape", "date", "honey", "intoxicating", "liquid", "red", "market"]


# BEGIN STANDARD UTILITIES v1.0
def logsumexp(z, axis=-1):
    m = np.max(z, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(z - m), axis=axis, keepdims=True))).squeeze(axis)


def softmax(z, axis=-1):
    return np.exp(z - np.expand_dims(logsumexp(z, axis=axis), axis))


def softplus(x):
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def sigmoid(x):
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-np.abs(x))),
                    np.exp(-np.abs(x)) / (1.0 + np.exp(-np.abs(x))))


class Adam:
    def __init__(self, params, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads, sign=-1.0):
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] = params[k] + sign * self.lr * mh / (np.sqrt(vh) + self.eps)


def clip_global_norm(grads, max_norm):
    total = np.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    scale = max_norm / (total + 1e-12) if total > max_norm else 1.0
    return {k: g * scale for k, g in grads.items()}, total


def finite_difference_check(loss_fn, params, grads, rng, eps=1e-6, n_entries=20, floor=1e-4):
    """Central differences in float64 on n random entries per tensor plus the
    largest-gradient entry; the tensor error is the norm of the difference over
    the checked entries divided by the larger of the two norms. The denominator
    is floored at 1e-4 so that a tensor whose gradient has all but vanished (a
    bias on balanced classes) is judged in absolute terms instead of by rounding
    noise; a wrong analytic gradient still shows up as an error far above 1e-5."""
    out = {}
    for k, p in params.items():
        flat_idx = list(rng.choice(p.size, size=min(n_entries, p.size), replace=False))
        flat_idx.append(int(np.argmax(np.abs(grads[k]))))
        an, fd = [], []
        for i in sorted(set(flat_idx)):
            old = p.flat[i]
            p.flat[i] = old + eps
            lp = loss_fn()
            p.flat[i] = old - eps
            lm = loss_fn()
            p.flat[i] = old
            fd.append((lp - lm) / (2 * eps))
            an.append(grads[k].flat[i])
        an, fd = np.array(an), np.array(fd)
        denom = max(np.linalg.norm(an), np.linalg.norm(fd), floor)
        out[k] = float(np.linalg.norm(an - fd) / denom)
    return out


def paired_bootstrap(diffs, rng, n_resamples=2000, alpha=0.05):
    d = np.asarray(diffs, dtype=float)
    if d.size < 2:
        return float(d.mean()) if d.size else 0.0, (float("nan"), float("nan"))
    means = np.array([rng.choice(d, size=d.size, replace=True).mean() for _ in range(n_resamples)])
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(d.mean()), (float(lo), float(hi))


def verdict_for(mean_diff, ci, direction, mesi):
    lo, hi = ci
    if np.isnan(lo):
        return "not evaluated"
    if direction == "greater":
        if lo > 0 and mean_diff >= mesi:
            return "supported"
        if hi < 0:
            return "contradicted"
    else:
        if hi < 0 and -mean_diff >= mesi:
            return "supported"
        if lo > 0:
            return "contradicted"
    return "inconclusive"


def json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o).__name__}")


def write_report(report, json_path=None):
    ch = report["chapter"]
    print(f"=== VERIFIED REPORT · chapter {ch:04d} ===")
    print(f"file: {report['file']}   card_revision: {report['card_revision']}")
    print(f"environment: python {report['environment']['python']}  numpy {report['environment']['numpy']}")
    print(f"seeds: {report['seeds']}   runtime_s: {report['runtime_s']:.1f}   n_params: {report['n_params']}")
    g = report["gradcheck"]
    print(f"gradcheck: {g['tensors_checked']}/{g['tensors_total']} tensors, max_rel_error {g['max_rel_error']:.2e}, "
          f"checked_at {g['checked_at']}, passed {g['passed']}")
    print("correctness:")
    for c in report["correctness"]:
        print(f"  {c['id']:<4} {c['name']:<28} {'PASS' if c['passed'] else 'FAIL'}  {c['detail']}")
    m = report["mutants"]
    print(f"mutation score: {m['detected']}/{m['total']} = {m['score']:.2f}")
    print("hypotheses:")
    for h in report["hypotheses"]:
        print(f"  {h['id']:<8} {h['metric']:<18} mean_diff {h['mean_diff']:+.3f}  ci95 [{h['ci95'][0]:+.3f}, {h['ci95'][1]:+.3f}]  "
              f"mesi {h['mesi']:.2f}  n {h['n_seeds']}  {h['verdict']}")
    print("knockouts (verdict accuracy change on the shifted split):")
    for k in report["knockouts"]:
        print(f"  {k['module']:<14} signature={str(k['signature']):<5} change {k['metric_change']:+.3f}  ci95 [{k['ci95'][0]:+.3f}, {k['ci95'][1]:+.3f}]")
    print(f"task_types: {report['task_types']}   exit_code: {report['exit_code']}")
    print("=== END REPORT ===")
    if json_path:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=json_default)
# END STANDARD UTILITIES


# ----------------------------------------------------------------------------
# Data and tasks
# ----------------------------------------------------------------------------
def term_labels(A):
    """Lexical definitions: the community's fixed usage of the eight terms."""
    f, g, d, h, x, l, r = (A[:, i] for i in range(7))
    T = np.stack([
        l,                                  # sharab: a drink
        f * g,                              # khamr: fermented grape
        f * np.maximum(d, h),               # nabidh: fermented date or honey
        x,                                  # muskir: intoxicating
        g * (1 - f) * l,                    # khall: unfermented grape liquid
        d * (1 - l),                        # tamr: dates, not a liquid
        l * (1 - f) * np.maximum(g, d),     # asir: unfermented fruit liquid
        r * (1 - l),                        # sibgh: a red non-liquid
    ], axis=1)
    return T.astype(float)


# fermented, grape, date, honey, intoxicating, liquid, red, market
PROTOTYPES = {
    "red_grape_wine":     [1, 1, 0, 0, 1, 1, 1, 1],
    "white_grape_wine":   [1, 1, 0, 0, 1, 1, 0, 1],
    "honey_mead":         [1, 0, 0, 1, 1, 1, 0, 1],
    "date_wine":          [1, 0, 1, 0, 1, 1, 0, 0],
    "red_grape_juice":    [0, 1, 0, 0, 0, 1, 1, 1],
    "date_syrup":         [0, 0, 1, 0, 0, 1, 0, 0],
    "vinegar":            [0, 1, 0, 0, 0, 1, 0, 1],
    "dried_dates":        [0, 0, 1, 0, 0, 0, 0, 1],
    "red_dye":            [0, 0, 0, 0, 0, 0, 1, 1],
    "water":              [0, 0, 0, 0, 0, 1, 0, 0],
    "solid_intoxicant":   [0, 0, 0, 0, 1, 0, 0, 0],
    "red_boiled_must":    [0, 1, 0, 0, 0, 1, 1, 0],
    "weak_grape_wine":    [1, 1, 0, 0, 0, 1, 1, 0],
    "drugged_red_drink":  [0, 0, 0, 1, 1, 1, 1, 1],
    "intoxicating_syrup": [0, 0, 1, 0, 1, 1, 0, 1],
    "flat_honey_brew":    [1, 0, 0, 1, 0, 1, 0, 0],
}
WORLD_A_TEXTS = [("khamr", "FORBIDDEN"), ("muskir", "khamr"), ("nabidh", "sharab")]
WORLD_B_TEXTS = [("khamr", "FORBIDDEN"), ("nabidh", "sharab")]
POOL_A_TRAIN = ["red_grape_wine", "date_syrup", "vinegar", "dried_dates", "water"]
POOL_A_SHIFT = ["honey_mead", "date_wine", "solid_intoxicant", "red_boiled_must",
                "red_grape_juice", "weak_grape_wine", "red_dye"]
POOL_B_TRAIN = list(PROTOTYPES)[:13]
POOL_B_BLIND = ["weak_grape_wine", "drugged_red_drink", "intoxicating_syrup", "flat_honey_brew"]
POOL_USAGE = list(PROTOTYPES)


def true_verdict(A, world):
    if world == "A":
        return ((A[:, 0] * A[:, 1]) + A[:, 4] > 0).astype(float)
    return (A[:, 4] > 0).astype(float)


def balanced_weights(pool, world):
    """Prototype weights giving each verdict class half the probability mass."""
    base = np.array([PROTOTYPES[k] for k in pool], dtype=float)
    forb = true_verdict(base, world)
    w = np.where(forb > 0, 0.5 / max(forb.sum(), 1), 0.5 / max((1 - forb).sum(), 1))
    return w / w.sum()


def sample_things(rng, pool, n, flip=0.05, weights=None):
    keys = rng.choice(len(pool), size=n, p=weights)
    base = np.array([PROTOTYPES[pool[k]] for k in keys], dtype=float)
    flips = rng.random(base.shape) < flip
    A = np.abs(base - flips)
    X = np.concatenate([A, rng.normal(size=(n, N_NUISANCE))], axis=1)
    return X, A


def make_usage(rng, n, noise=0.04):
    X, A = sample_things(rng, POOL_USAGE, n)
    T = term_labels(A)
    T = np.abs(T - (rng.random(T.shape) < noise))
    return {"X": X, "T": T}


def make_verdicts(rng, pool, n, world):
    X, A = sample_things(rng, pool, n, weights=balanced_weights(pool, world))
    return {"X": X, "T": term_labels(A), "V": true_verdict(A, world)}


def encode_texts(texts):
    edges = np.zeros((N_TERMS, N_TERMS))
    forb = np.zeros(N_TERMS)
    for s, o in texts:
        i = TERM_NAMES.index(s)
        if o == "FORBIDDEN":
            forb[i] = 1.0
        else:
            edges[i, TERM_NAMES.index(o)] = 1.0
    return edges, forb


def make_world(rng, world, quick):
    n_use = 400 if quick else 600
    pool_tr, pool_sh = (POOL_A_TRAIN, POOL_A_SHIFT) if world == "A" else (POOL_B_TRAIN, POOL_B_BLIND)
    n_tr = 60 if world == "A" else 100
    data = {
        "usage_train": make_usage(rng, n_use), "usage_val": make_usage(rng, 200),
        "verdict_train": make_verdicts(rng, pool_tr, n_tr, world),
        "verdict_val": make_verdicts(rng, pool_tr, 100, world),
        "held_out": make_verdicts(rng, pool_tr, 200, world),
        "shifted": make_verdicts(rng, pool_sh, 300, world),
        "texts": WORLD_A_TEXTS if world == "A" else WORLD_B_TEXTS,
    }
    return data


def row_hashes(X):
    return {hashlib.sha1(np.round(r, 9).tobytes()).hexdigest() for r in X}


def trivial_accuracy(train_v, eval_v):
    majority = 1.0 if train_v.mean() > 0.5 else 0.0
    return float(np.mean(eval_v == majority))


def within_trivial_band(a, eval_v, band):
    """True when a lies within band of the interval spanned by the two constant
    predictors (always permitted, always forbidden) on the evaluation split."""
    p = float(eval_v.mean())
    return min(p, 1 - p) - band <= a <= max(p, 1 - p) + band


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def closure(m, edges, steps):
    """Dalil: Boolean forward chaining. m (N,K) bool, edges (K,K): i -> j."""
    c = m.copy()
    for _ in range(steps):
        c = c | ((c.astype(float) @ edges) > 0)
    return c


def closure_subtractive(m, edges, steps):
    """Negative control for C6.1: a text removes coverage instead of adding it."""
    c = m.copy()
    for _ in range(steps):
        c = c & ~((c.astype(float) @ edges) > 0)
    return c


def closure_sequential(m, edge_list):
    """Negative control for C6.2: one pass over the texts in list order."""
    c = m.copy()
    for i, j in edge_list:
        c[:, j] = c[:, j] | c[:, i]
    return c


class ZahirDalilReader:
    """Usage-trained bivalent lexicon + dalil closure + precedents + default."""

    def __init__(self, in_dim, n_terms, hidden, rng, texts=None):
        self.in_dim, self.K, self.H = in_dim, n_terms, hidden
        s1, s2, s3 = 1.0 / np.sqrt(in_dim), 1.0 / np.sqrt(hidden), 1.0 / np.sqrt(hidden)
        self.params = {
            "W1": rng.normal(scale=s1, size=(in_dim, hidden)), "b1": np.zeros(hidden),
            "W2": rng.normal(scale=s2, size=(hidden, hidden)), "b2": np.zeros(hidden),
            "Wt": rng.normal(scale=s3, size=(hidden, n_terms)), "bt": np.zeros(n_terms),
        }
        self.texts = texts
        if texts is None:      # generic classification: every term names its own class
            self.edges, self.forb = np.zeros((n_terms, n_terms)), np.zeros(n_terms)
        else:
            self.edges, self.forb = encode_texts(texts)
        self.table: Dict[Tuple, float] = {}
        self.mean_pattern = np.zeros(n_terms, dtype=bool)
        self.flags = {"deep_layer": True, "dalil": True, "ruling_table": True,
                      "lexicon_mean": False, "default": 0.0}
        self.signature = True

    def features(self, X):
        p = self.params
        h1 = np.tanh(X @ p["W1"] + p["b1"])
        if self.flags["deep_layer"]:
            g = np.tanh(h1 @ p["W2"] + p["b2"])
            return h1, g, h1 + g
        return h1, None, h1

    def term_logits(self, X):
        return self.features(X)[2] @ self.params["Wt"] + self.params["bt"]

    def names(self, X):
        m = self.term_logits(X) > 0
        if self.flags["lexicon_mean"]:
            m = np.repeat(self.mean_pattern[None, :], X.shape[0], axis=0)
        return m

    def verdicts(self, X):
        m = self.names(X)
        steps = self.K if self.flags["dalil"] else 0
        by_text = (closure(m, self.edges, steps) & (self.forb[None, :] > 0)).any(axis=1)
        out = np.full(X.shape[0], self.flags["default"])
        if self.flags["ruling_table"]:
            for r, row in enumerate(m):
                out[r] = self.table.get(tuple(row.tolist()), out[r])
        out[by_text] = 1.0
        return out

    def record_precedents(self, X, V):
        """Non-gradient rule: a decided case binds only its exact name pattern."""
        votes: Dict[Tuple, List[float]] = {}
        for row, v in zip(self.names(X), V):
            votes.setdefault(tuple(row.tolist()), []).append(float(v))
        self.table = {k: (1.0 if np.mean(vs) > 0.5 else 0.0) for k, vs in votes.items()}


class TawilLearner(ZahirDalilReader):
    """Baseline: the same encoder and term heads plus a learned verdict head
    whose gradients reshape the shared encoder (ta'wil)."""

    def __init__(self, in_dim, n_terms, hidden, rng, texts=None):
        super().__init__(in_dim, n_terms, hidden, rng, texts)
        self.params["Wv"] = rng.normal(scale=1.0 / np.sqrt(hidden), size=(hidden, 1))
        self.params["bv"] = np.zeros(1)
        self.signature = False

    def verdicts(self, X):
        return (self.features(X)[2] @ self.params["Wv"] + self.params["bv"] > 0).astype(float)[:, 0]


class QiyasEngine:
    """Rival: Nadaraya-Watson extension of decided cases by attribute similarity."""

    def __init__(self, X, V, bandwidths, Xv, Vv):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        self.Z, self.V = (X - self.mu) / self.sd, V
        best = None
        for bw in bandwidths:
            self.bw = bw
            acc = float(np.mean(self.verdicts(Xv) == Vv))
            if best is None or acc > best[0]:
                best = (acc, bw)
        self.bw = best[1]

    def verdicts(self, X):
        Z = (X - self.mu) / self.sd
        d2 = ((Z[:, None, :] - self.Z[None, :, :]) ** 2).sum(-1)
        w = softmax(-d2 / (2 * self.bw ** 2), axis=1)
        return (w @ self.V > 0.5).astype(float)


# ----------------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------------
TASK_TYPES = ["vector_classification"]


def build_model(in_dim, out_dim, task_type, rng, hidden=24, texts=None, baseline=False, **cfg):
    if task_type not in TASK_TYPES:
        raise ValueError(f"unsupported task type {task_type}")
    cls = TawilLearner if baseline else ZahirDalilReader
    return cls(in_dim, out_dim, hidden, rng, texts)


def _bce(logits, Y):
    return softplus(logits) - logits * Y


def loss_and_grads(model, batch, zero_grad_key=None):
    """Usage loss on term labels (and, for the baseline, a verdict loss).
    Returns the loss and a gradient dict with exactly the keys of params."""
    p = model.params
    X, T = batch["X"], batch.get("T")
    if T is None:
        T = np.eye(model.K)[batch["y"].astype(int)]
    grads = {k: np.zeros_like(v) for k, v in p.items()}
    h1, g, h2 = model.features(X)
    logits = h2 @ p["Wt"] + p["bt"]
    n = X.shape[0] * model.K
    loss = float(_bce(logits, T).sum() / n)
    dlog = (sigmoid(logits) - T) / n
    grads["Wt"] += h2.T @ dlog
    grads["bt"] += dlog.sum(0)
    dh2 = dlog @ p["Wt"].T
    _backprop_encoder(model, X, h1, g, dh2, grads)
    if isinstance(model, TawilLearner) and "XV" in batch:
        XV, V = batch["XV"], batch["V"]
        h1v, gv, h2v = model.features(XV)
        zv = h2v @ p["Wv"] + p["bv"]
        nv = XV.shape[0]
        loss += float(_bce(zv[:, 0], V).sum() / nv)
        dz = ((sigmoid(zv[:, 0]) - V) / nv)[:, None]
        grads["Wv"] += h2v.T @ dz
        grads["bv"] += dz.sum(0)
        _backprop_encoder(model, XV, h1v, gv, dz @ p["Wv"].T, grads)
    if zero_grad_key is not None:
        grads[zero_grad_key] = np.zeros_like(grads[zero_grad_key])
    return loss, grads


def _backprop_encoder(model, X, h1, g, dh2, grads):
    p = model.params
    dh1 = dh2.copy()
    if g is not None:
        da2 = dh2 * (1 - g * g)
        grads["W2"] += h1.T @ da2
        grads["b2"] += da2.sum(0)
        dh1 += da2 @ p["W2"].T
    da1 = dh1 * (1 - h1 * h1)
    grads["W1"] += X.T @ da1
    grads["b1"] += da1.sum(0)


def make_batch(model, data):
    b = {"X": data["usage_train"]["X"], "T": data["usage_train"]["T"]}
    if isinstance(model, TawilLearner):
        b["XV"], b["V"] = data["verdict_train"]["X"], data["verdict_train"]["V"]
    return b


def fit(model, data, budget, rng, lr=0.02, mutant=None, log=None):
    """Full-batch Adam under a fixed update budget; selection on validation."""
    batch = make_batch(model, data)
    vbatch = {"X": data["usage_val"]["X"], "T": data["usage_val"]["T"]}
    if isinstance(model, TawilLearner):
        vbatch["XV"], vbatch["V"] = data["verdict_val"]["X"], data["verdict_val"]["V"]
    opt = Adam(model.params, lr=0.0 if mutant == "zero_lr" else lr)
    sign = +1.0 if mutant == "sign_flip" else -1.0
    zkey = "Wt" if mutant == "grad_zero_lexicon" else None
    best, best_params, first = None, None, None
    for step in range(budget):
        loss, grads = loss_and_grads(model, batch, zero_grad_key=zkey)
        if first is None:
            first = loss
        grads, _ = clip_global_norm(grads, CLIP_NORM)
        opt.step(model.params, grads, sign=sign)
        vloss, _ = loss_and_grads(model, vbatch)
        if best is None or vloss < best:
            best, best_params = vloss, {k: v.copy() for k, v in model.params.items()}
        if log is not None:
            log.append(loss)
    model.params.update(best_params)
    if "V" in data["verdict_train"]:
        model.record_precedents(data["verdict_train"]["X"], data["verdict_train"]["V"])
        model.mean_pattern = model.names(data["usage_train"]["X"]).mean(0) > 0.5
    final, _ = loss_and_grads(model, batch)
    return {"first_loss": first, "final_loss": final, "steps": budget}


def predict(model, X):
    if model.texts is None:
        return np.argmax(model.term_logits(X), axis=1)
    return model.verdicts(X)


def hidden_states(model, X):
    return model.features(X)[2]


def modules(model):
    return {
        "encoder": {"params": ["W1", "b1"], "role": "tanh feature layer", "signature": False},
        "deep_layer": {"params": ["W2", "b2"], "role": "residual tanh layer", "signature": False},
        "lexicon": {"params": ["Wt", "bt"], "role": "bivalent multi-label term heads (crisp at threshold 0)", "signature": True},
        "dalil": {"params": [], "role": "Boolean forward chaining over given rule texts", "signature": True},
        "ruling_table": {"params": [], "role": "exact-key precedent lookup on the name pattern", "signature": True},
        "default": {"params": [], "role": "constant closed-world verdict for uncovered cases", "signature": True},
    }


def knockout(model, name, mode):
    """Copy with a module replaced by identity, zero or mean output."""
    k = model.__class__.__new__(model.__class__)
    k.__dict__.update(model.__dict__)
    k.params = {kk: v.copy() for kk, v in model.params.items()}
    k.flags = dict(model.flags)
    if name == "deep_layer" and mode == "identity":
        k.flags["deep_layer"] = False
    elif name == "dalil" and mode == "identity":
        k.flags["dalil"] = False
    elif name == "ruling_table" and mode == "zero":
        k.flags["ruling_table"] = False
    elif name == "lexicon" and mode == "mean":
        k.flags["lexicon_mean"] = True
    elif name == "default" and mode == "flip":
        k.flags["default"] = 1.0
    else:
        raise ValueError(f"unknown knockout {name}/{mode}")
    return k


def n_params(model):
    return int(sum(v.size for v in model.params.values()))


MUTANTS = {
    "sign_flip": "update follows the gradient instead of opposing it",
    "zero_lr": "learning rate zero",
    "grad_zero_lexicon": "gradient of the term-head weights zeroed",
}


# ----------------------------------------------------------------------------
# Tests: correctness
# ----------------------------------------------------------------------------
def acc(model, split):
    return float(np.mean(predict(model, split["X"]) == split["V"]))


def gradcheck_model(model, batch, rng, tol):
    def loss_fn():
        return loss_and_grads(model, batch)[0]
    _, grads = loss_and_grads(model, batch)
    errs = finite_difference_check(loss_fn, model.params, grads, rng)
    return errs, all(e <= tol for e in errs.values())


def test_c1(data, rng, quick, tol):
    rows, worst, checked, total = [], 0.0, 0, 0
    for baseline in (False, True):
        model = build_model(IN_DIM, N_TERMS, "vector_classification", rng, texts=data["texts"], baseline=baseline)
        batch = make_batch(model, data)
        errs0, ok0 = gradcheck_model(model, batch, rng, tol)
        fit(model, data, 60, rng)
        errs1, ok1 = gradcheck_model(model, batch, rng, tol)
        worst = max(worst, max(errs0.values()), max(errs1.values()))
        checked += len(errs0)
        total += len(model.params)
        rows.append((ok0 and ok1, errs0, errs1))
    passed = all(r[0] for r in rows)
    detail = f"max rel err {worst:.2e} over reader and baseline, init and after 60 steps"
    return passed, detail, {"tensors_checked": checked, "tensors_total": total, "max_rel_error": worst,
                            "checked_at": ["init", "after_training_steps"], "passed": passed}


def test_c2(data, seed, budget):
    outs = []
    for _ in range(2):
        rng = np.random.default_rng(seed)
        m = build_model(IN_DIM, N_TERMS, "vector_classification", rng, texts=data["texts"])
        log = []
        fit(m, data, budget, rng, log=log)
        outs.append((np.array(log), predict(m, data["held_out"]["X"]), m.params, hidden_states(m, data["held_out"]["X"])))
    same = (np.array_equal(outs[0][0], outs[1][0]) and np.array_equal(outs[0][1], outs[1][1])
            and np.array_equal(outs[0][3], outs[1][3]))
    finite = (all(np.all(np.isfinite(v)) for v in outs[0][2].values()) and np.all(np.isfinite(outs[0][0]))
              and np.all(np.isfinite(outs[0][3])))
    return same and finite, f"identical losses and outputs across two runs: {same}; finite: {finite}", finite


def test_c3(data, rng, budget, thr):
    m = build_model(IN_DIM, N_TERMS, "vector_classification", rng, texts=data["texts"])
    info = fit(m, data, budget, rng)
    drop = 1 - info["final_loss"] / info["first_loss"]
    a = acc(m, data["held_out"])
    triv = trivial_accuracy(data["verdict_train"]["V"], data["held_out"]["V"])
    term_acc = float(np.mean((m.term_logits(data["usage_val"]["X"]) > 0) == (data["usage_val"]["T"] > 0.5)))
    ok = drop >= thr["loss_drop_fraction"] and a - triv >= thr["margin_over_trivial"]
    return ok, f"loss drop {drop:.2f} (>= {thr['loss_drop_fraction']}), held-out {a:.3f} vs trivial {triv:.3f}, term acc {term_acc:.3f}", m


def test_c4(data, rng, budget, band, n_rep):
    """Shuffled-label control: every training target the learner consumes is
    permuted (usage term labels; verdict labels for the baseline). Because the
    texts turn any random name that happens to align with khamr or muskir into
    a verdict, one permutation has a wide spread; the control is repeated with
    fresh permutations and the mean must stay within the band around the two
    constant predictors of the evaluation split."""
    details, ok = [], True
    triv = trivial_accuracy(data["verdict_train"]["V"], data["held_out"]["V"])
    for baseline in (False, True):
        accs = []
        for _ in range(n_rep):
            d = {k: dict(v) if isinstance(v, dict) else v for k, v in data.items()}
            perm = rng.permutation(d["usage_train"]["X"].shape[0])
            d["usage_train"] = {"X": d["usage_train"]["X"], "T": d["usage_train"]["T"][perm]}
            pv = rng.permutation(d["verdict_train"]["X"].shape[0])
            d["verdict_train"] = {"X": d["verdict_train"]["X"], "T": d["verdict_train"]["T"][pv], "V": d["verdict_train"]["V"][pv]}
            m = build_model(IN_DIM, N_TERMS, "vector_classification", rng, texts=data["texts"], baseline=baseline)
            fit(m, d, budget, rng)
            accs.append(acc(m, data["held_out"]))
        mean = float(np.mean(accs))
        ok = ok and within_trivial_band(mean, data["held_out"]["V"], band)
        details.append(f"{'baseline' if baseline else 'reader'} mean {mean:.3f} over {n_rep} permutations "
                       f"({', '.join(f'{a:.2f}' for a in accs)}) vs majority {triv:.3f}")
    return ok, "; ".join(details)


def test_c5(data, rng, budget, thr, tol):
    detected = 0
    notes = []
    for name in MUTANTS:
        m = build_model(IN_DIM, N_TERMS, "vector_classification", rng, texts=data["texts"])
        batch = make_batch(m, data)
        _, ok_grad = gradcheck_model(m, batch, rng, tol) if name != "grad_zero_lexicon" else (None, True)
        if name == "grad_zero_lexicon":
            def loss_fn():
                return loss_and_grads(m, batch)[0]
            _, g = loss_and_grads(m, batch, zero_grad_key="Wt")
            errs = finite_difference_check(loss_fn, m.params, g, rng)
            ok_grad = all(e <= tol for e in errs.values())
        info = fit(m, data, budget, rng, mutant=name)
        drop = 1 - info["final_loss"] / info["first_loss"]
        a = acc(m, data["held_out"])
        triv = trivial_accuracy(data["verdict_train"]["V"], data["held_out"]["V"])
        ok_learn = drop >= thr["loss_drop_fraction"] and a - triv >= thr["margin_over_trivial"]
        caught = (not ok_grad) or (not ok_learn)
        detected += int(caught)
        notes.append(f"{name}: {'caught' if caught else 'MISSED'} (C1 {ok_grad}, C3 {ok_learn})")
    return detected == len(MUTANTS), "; ".join(notes), detected


def test_c6(rng):
    """Property tests on random inputs with negative controls that must fail."""
    K, n_trials = N_TERMS, 200
    viol_mono = viol_mono_ctrl = viol_perm = viol_perm_ctrl = viol_fix = viol_fix_ctrl = 0
    for _ in range(n_trials):
        m = rng.random((6, K)) < 0.3
        E1 = (rng.random((K, K)) < 0.15).astype(float)
        np.fill_diagonal(E1, 0)
        E2 = np.maximum(E1, (rng.random((K, K)) < 0.15).astype(float))
        np.fill_diagonal(E2, 0)
        c1, c2 = closure(m, E1, K), closure(m, E2, K)
        viol_mono += int(np.any(c1 & ~c2))
        viol_mono_ctrl += int(np.any(closure_subtractive(m, E1, K) & ~closure_subtractive(m, E2, K)))
        edges = [(int(i), int(j)) for i, j in zip(*np.nonzero(E2))]
        perm = [edges[i] for i in rng.permutation(len(edges))]
        Ep = np.zeros((K, K))
        for i, j in perm:
            Ep[i, j] = 1.0
        viol_perm += int(not np.array_equal(closure(m, E2, K), closure(m, Ep, K)))
        viol_perm_ctrl += int(not np.array_equal(closure_sequential(m, edges), closure_sequential(m, perm)))
        viol_fix += int(not np.array_equal(closure(closure(m, E2, K), E2, K), c2))
        viol_fix_ctrl += int(not np.array_equal(closure(closure(m, E2, 1), E2, 1), closure(m, E2, 1)))
    ok = (viol_mono == 0 and viol_mono_ctrl > 0 and viol_perm == 0 and viol_perm_ctrl > 0
          and viol_fix == 0 and viol_fix_ctrl > 0)
    # labelled definition checks (not evidence): empty law -> all permitted; same names -> same verdict
    r = ZahirDalilReader(IN_DIM, N_TERMS, 8, rng, texts=[])
    X = rng.normal(size=(20, IN_DIM))
    def_ok = np.all(r.verdicts(X) == 0.0)
    detail = (f"C6.1 monotone: {viol_mono}/{n_trials} violations (control {viol_mono_ctrl}); "
              f"C6.2 order-free: {viol_perm} (control {viol_perm_ctrl}); "
              f"C6.3 fixpoint: {viol_fix} (control {viol_fix_ctrl}); definition check empty-law default: {def_ok}")
    return ok and bool(def_ok), detail


def test_c7(data):
    tr = row_hashes(data["verdict_train"]["X"]) | row_hashes(data["usage_train"]["X"])
    leaks = sum(len(tr & row_hashes(data[s]["X"])) for s in ("held_out", "shifted", "usage_val", "verdict_val"))
    return leaks == 0, f"{leaks} rows shared between training and evaluation splits"


# ----------------------------------------------------------------------------
# Tests: hypotheses
# ----------------------------------------------------------------------------
def run_seed(seed, quick, budget):
    """One seed: world A (shift), knockouts, rival, world B (blind spot)."""
    rng = np.random.default_rng(seed)
    child = np.random.SeedSequence(seed).spawn(2)
    A = make_world(np.random.default_rng(child[0]), "A", quick)
    B = make_world(np.random.default_rng(child[1]), "B", quick)
    out = {}
    reader = build_model(IN_DIM, N_TERMS, "vector_classification", rng, texts=A["texts"])
    tawil = build_model(IN_DIM, N_TERMS, "vector_classification", rng, texts=A["texts"], baseline=True)
    fit(reader, A, budget, rng)
    fit(tawil, A, budget, rng)
    out["reader_id"], out["tawil_id"] = acc(reader, A["held_out"]), acc(tawil, A["held_out"])
    out["reader_shift"], out["tawil_shift"] = acc(reader, A["shifted"]), acc(tawil, A["shifted"])
    out["trivial_shift"] = trivial_accuracy(A["verdict_train"]["V"], A["shifted"]["V"])
    for name, mode in (("dalil", "identity"), ("deep_layer", "identity"), ("lexicon", "mean"),
                       ("ruling_table", "zero"), ("default", "flip")):
        out[f"ko_{name}"] = acc(knockout(reader, name, mode), A["shifted"])
    qiyas = QiyasEngine(A["verdict_train"]["X"], A["verdict_train"]["V"], [0.5, 1.0, 2.0, 4.0],
                        A["verdict_val"]["X"], A["verdict_val"]["V"])
    out["qiyas_shift"] = float(np.mean(qiyas.verdicts(A["shifted"]["X"]) == A["shifted"]["V"]))
    readerB = build_model(IN_DIM, N_TERMS, "vector_classification", rng, texts=B["texts"])
    tawilB = build_model(IN_DIM, N_TERMS, "vector_classification", rng, texts=B["texts"], baseline=True)
    fit(readerB, B, budget, rng)
    fit(tawilB, B, budget, rng)
    out["reader_blind"], out["tawil_blind"] = acc(readerB, B["shifted"]), acc(tawilB, B["shifted"])
    out["reader_B_id"], out["tawil_B_id"] = acc(readerB, B["held_out"]), acc(tawilB, B["held_out"])
    return out


def hypothesis_table(results, rng, quick):
    hyps = []
    spec = {h["id"]: h for h in MIND_CARD["hypotheses"]}
    pairs = {"H-SIG": ("reader_shift", "tawil_shift"), "H-NEC": ("ko_dalil", "ko_deep_layer"),
             "H-BLIND": ("reader_blind", "tawil_blind"), "H-RIVAL": ("reader_shift", "qiyas_shift")}
    for hid, (a, b) in pairs.items():
        diffs = [r[a] - r[b] for r in results]
        if quick:
            mean, ci, verdict = float(np.mean(diffs)), (float("nan"), float("nan")), "not evaluated"
        else:
            mean, ci = paired_bootstrap(diffs, rng)
            verdict = verdict_for(mean, ci, spec[hid]["direction"], spec[hid]["mesi"])
        hyps.append({"id": hid, "metric": spec[hid]["metric"], "mean_diff": mean, "ci95": list(ci),
                     "mesi": spec[hid]["mesi"], "n_seeds": len(diffs), "verdict": verdict})
    return hyps


def knockout_table(results, rng, model):
    rows = []
    reg = modules(model)
    for name in ("dalil", "deep_layer", "lexicon", "ruling_table", "default"):
        diffs = [r[f"ko_{name}"] - r["reader_shift"] for r in results]
        mean, ci = paired_bootstrap(diffs, rng) if len(diffs) > 1 else (float(np.mean(diffs)), (float("nan"), float("nan")))
        rows.append({"module": name, "signature": reg[name]["signature"], "role": reg[name]["role"],
                     "metric_change": mean, "ci95": list(ci)})
    return rows


# ----------------------------------------------------------------------------
# Optional real-data bridge
# ----------------------------------------------------------------------------
def real_data_bridge(path, rng, budget):
    """Local CSV with numeric attribute columns, term_* columns (0/1) and a
    verdict column; texts are read from a JSON list in column 'text' of row 0
    if present, else every term is its own class. No downloads."""
    if not path:
        print("real-data bridge: no --data path given; skipped")
        return
    if not os.path.exists(path):
        print(f"real-data bridge: {path} not found; skipped")
        return
    with open(path, encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        rows = [ln.strip().split(",") for ln in f if ln.strip()]
    term_cols = [i for i, h in enumerate(header) if h.startswith("term_")]
    v_col = header.index("verdict") if "verdict" in header else None
    attr_cols = [i for i, h in enumerate(header) if i not in term_cols and i != v_col]
    M = np.array([[float(r[i]) for i in attr_cols + term_cols + ([v_col] if v_col is not None else [])] for r in rows])
    n_a, n_t = len(attr_cols), len(term_cols)
    X, T = M[:, :n_a], M[:, n_a:n_a + n_t]
    idx = rng.permutation(len(X))
    cut = int(0.8 * len(X))
    model = ZahirDalilReader(n_a, n_t, 24, rng, texts=None)
    data = {"usage_train": {"X": X[idx[:cut]], "T": T[idx[:cut]]}, "usage_val": {"X": X[idx[cut:]], "T": T[idx[cut:]]},
            "verdict_train": {"X": X[idx[:cut]], "V": M[idx[:cut], -1]} if v_col is not None else {}}
    fit(model, data, budget, rng)
    term_acc = float(np.mean((model.term_logits(X[idx[cut:]]) > 0) == (T[idx[cut:]] > 0.5)))
    print(f"real-data bridge: {len(X)} rows, held-out term accuracy {term_acc:.3f}")


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def parse_args(argv):
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__), add_help=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--card", action="store_true")
    ap.add_argument("--mutant", type=str, default=None)
    ap.add_argument("--data", type=str, default=None)
    try:
        return ap.parse_args(argv)
    except SystemExit:
        sys.exit(2)


def run_correctness(base, rng, seed, quick, budget, thr):
    """C1-C7 in order; C8 is added by the entry point once the run has ended."""
    rows = []
    c1_ok, c1_detail, gradinfo = test_c1(base, rng, quick, thr["gradcheck_tol"])
    rows.append({"id": "C1", "name": "gradient_check", "passed": c1_ok, "detail": c1_detail})
    c2_ok, c2_detail, finite = test_c2(base, seed, 40)
    rows.append({"id": "C2", "name": "determinism_and_finiteness", "passed": c2_ok, "detail": c2_detail})
    c3_ok, c3_detail, trained = test_c3(base, rng, budget, thr)
    rows.append({"id": "C3", "name": "learning", "passed": c3_ok, "detail": c3_detail})
    c4_ok, c4_detail = test_c4(base, rng, budget, thr["trivial_band"], 6 if quick else 10)
    rows.append({"id": "C4", "name": "shuffled_label_control", "passed": c4_ok, "detail": c4_detail})
    c5_ok, c5_detail, detected = test_c5(base, rng, budget, thr, thr["gradcheck_tol"])
    rows.append({"id": "C5", "name": "mutant_detection", "passed": c5_ok, "detail": c5_detail})
    c6_ok, c6_detail = test_c6(rng)
    rows.append({"id": "C6", "name": "property_tests", "passed": c6_ok, "detail": c6_detail})
    c7_ok, c7_detail = test_c7(base)
    rows.append({"id": "C7", "name": "split_integrity", "passed": c7_ok, "detail": c7_detail})
    return rows, gradinfo, trained, detected, finite


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.card:
        print(json.dumps(MIND_CARD, indent=2))
        return 0
    if args.mutant is not None and args.mutant not in MUTANTS:
        print(f"unknown mutant {args.mutant}; known: {sorted(MUTANTS)}")
        return 2
    t0 = time.time()
    quick = args.quick
    budget = 150 if quick else 400
    seeds = [args.seed] if quick else [args.seed + i for i in range(args.seeds)]
    limit = BUDGET_QUICK_S if quick else BUDGET_FULL_S
    thr = MIND_CARD["thresholds"]
    print(f"chapter {CHAPTER:04d} · seeds {seeds} · {'quick' if quick else 'full'} protocol · budget {budget} updates")
    rng = np.random.default_rng(args.seed)
    base = make_world(np.random.default_rng(np.random.SeedSequence(args.seed).spawn(1)[0]), "A", quick)
    if args.mutant is not None:
        m = build_model(IN_DIM, N_TERMS, "vector_classification", rng, texts=base["texts"])
        info = fit(m, base, budget, rng, mutant=args.mutant)
        print(f"mutant {args.mutant}: first loss {info['first_loss']:.4f} final {info['final_loss']:.4f} "
              f"held-out {acc(m, base['held_out']):.3f}")
    correctness, gradinfo, trained, detected, finite = run_correctness(base, rng, args.seed, quick, budget, thr)
    if not finite:
        print("non-finite values encountered")
        return 4
    exit_code = 0
    results = [run_seed(s, quick, budget) for s in seeds]
    for s, r in zip(seeds, results):
        print(f"seed {s}: held-out reader {r['reader_id']:.3f} tawil {r['tawil_id']:.3f} | shifted reader {r['reader_shift']:.3f} "
              f"tawil {r['tawil_shift']:.3f} qiyas {r['qiyas_shift']:.3f} trivial {r['trivial_shift']:.3f} | "
              f"blind reader {r['reader_blind']:.3f} tawil {r['tawil_blind']:.3f}")
    real_data_bridge(args.data, rng, budget)
    elapsed = time.time() - t0
    c8_ok = elapsed <= limit
    correctness.append({"id": "C8", "name": "budget", "passed": c8_ok, "detail": f"{elapsed:.1f} s of {limit:.0f} s"})
    if not all(c["passed"] for c in correctness):
        exit_code = 1
    if not c8_ok:
        exit_code = 3
    report = {
        "schema_version": "1.0", "chapter": CHAPTER, "file": os.path.basename(__file__),
        "card_revision": MIND_CARD["card_revision"],
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
        "seeds": seeds, "runtime_s": elapsed, "n_params": n_params(trained),
        "gradcheck": gradinfo, "correctness": correctness,
        "mutants": {"detected": detected, "total": len(MUTANTS), "score": detected / len(MUTANTS)},
        "hypotheses": hypothesis_table(results, rng, quick), "knockouts": knockout_table(results, rng, trained),
        "task_types": TASK_TYPES, "exit_code": exit_code,
    }
    write_report(report, args.json)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

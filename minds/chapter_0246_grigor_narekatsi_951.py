#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 13 Minds 241 - 260 Available on Amazon https://www.amazon.com/dp/B0HLD1S6QT
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0246_grigor_narekatsi_951 - Grigor Narekatsi (Gregory of Narek) (c.951-1003)
#================================================================================  
"""
MATEAN: sealed laments of uncommitted fault, routed to a physician
Chapter 0246, Grigor Narekatsi (c. 951-1003). A research prototype, not an AGI.
Run: python3 <this file> [--quick] [--seed S] [--seeds N] [--json PATH] [--card]
     [--mutant NAME] [--data PATH]

Thesis
  Repair reaches only what is disclosed, so a mind must confess in its own voice
  the faults that success has hidden and hand that confession to a physician
  that restores, never to a judge that punishes.

Evidence and provenance (belief: the figure's own surviving work)
  The Book of Lamentation (Matean ołbergut'ean), finished in Armenian era 451
  (AD 1002), cited by prayer and section of the English translation by
  T. J. Samuelian (Yerevan: Vem Press, 2001); short quotations are
  Samuelian's. Reading these passages as a doctrine of mind is this file's
  interpretation, not a claim the sources make.
  D1 Prayer 28a    which faults to confess: "the future, which I fear"
  D2 Prayer 28b-c  he shows openly what clever people conceal and writes
                   without restraint so that the faults can be blotted out
  D3 Prayer 23b    treated as a physician treats, not examined as a judge does
  D4 Prayer 88b-c  the book, kept from destruction, "will cry out in my place",
                   uncovers what he hid and lets others find the pitfalls
  D5 Prayers 3b, 28b  "all earthly ills" make him emissary for every station
  D6 Prayer 88b    the book inscribed on the doors of the mind and on the
                   threshold of the senses
  D7 Prayer 23c    the guilt-filled gaze recoils even from the harvest of goodness
  D8 Terian 2021   the universal confession read as solidarity, not autobiography
  D9 Colophon      composed in 1002 with his brother Hovhannes

Doctrine -> mechanism -> test
  D1     -> M1 uncommitted lament: sigmoid probe of the speaker's own hidden
            states, trained on counterfactual culpability (the verdict moves
            when a cue is resampled while the truth would not) -> C6.4, H-NEC
  D2, D3 -> M3 physician: each confessed cue blotted out (returned to its
            absent state) with probability equal to its disclosed measure
            -> C6.2, C6.3, H-SIG
  D4     -> M2 seal: laments read stop-gradient states and never train the
            speaker; the rival tribunal uses the same laments as a penalty
                                                   -> C6.1 (definition), H-SIG
  D6     -> M1 reads both hidden layers, sense and mind -> C1
  D5, D7 -> M1 targets pooled from every reader        -> H-BLIND

Research question (truthful reporting and sycophancy)
  If a model's latent self-report of reliance on a cue is routed to inference-
  time repair instead of a training penalty, does the report stay faithful, and
  does the repair survive the shift that in-distribution success concealed?

Closest prior art and the delta
  Confessions rewarded only for honesty, decoupled from the answer (Joglekar et
  al. 2025); obfuscation under monitor pressure (Baker et al. 2025; Bailey et al.
  2024); counterfactual invariance (Veitch et al. 2021); upweighting a first
  model's errors (Liu et al. 2021). Delta: the self-report is supervised on
  culpability for faults not yet committed, read through a gradient seal from
  both hidden layers, and acts only by blotting out the confessed cue with a
  probability equal to the disclosed measure. The tribunal baseline is the same machinery with the
  laments turned into a training penalty, so only the routing differs.

Blind spot
  Pooled confession takes another reader's fault as one's own. Where natures
  differ invisibly, the physician marginalises a cue that is causal for this
  reader: scrupulosity, the recoil from goodness of D7.
  Revision note: card revision 2 replaced resampling-based restoration by
  blotting out (D2, Prayer 28c) after a one-seed development run showed that
  averaging over resampled cues leaves verdicts too unconfident to repair;
  hypotheses, metrics and thresholds were not changed.

Task (generative process)
  q ~ N(0,1); g ~ Bern(0.5); vine s2 and root s3 uniform on {-1,+1}
  y = 1[q + 0.8 s3 + g s2 > 0]          (the vine is causal only for g = 1)
  leaves s1, and the vine when g = 0, copy the label's sign with prob. rho
  x = core q (6) | context g (2) | leaves (4) | vine (4) | root (4) | nuisance (4)
  each cue block absent (zero) w.p. 0.3; nuisance scale ~ U(0.5, 1.5)
  training and held-out: rho = 1, so leaning on the leaves never errs
  shifted: rho = 0.1; blind: context unobservable, g = 1, leaves absent

Limits
  Synthetic vectors, a two-layer speaker, linear laments, three cue blocks.
  Counterfactual culpability is known only on a small curated set, as an oracle
  would be in practice. No dialectic rival is implemented: the documented
  textual inheritance (Gregory of Nyssa, in the Song of Songs commentary) bears
  on exegesis, not on this mechanism. Nothing here measures confession in
  people or makes a claim about Narekatsi's psychology.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 2,
    "revision_log": [{"revision": 2, "date": "2026-09-16", "change": (
        "Physician changed from averaging verdicts over resampled cue blocks to blotting out confessed cue blocks "
        "(absent state), grounded in Prayer 28c. Reason: a one-seed quick development run (hypotheses not evaluated) "
        "showed resampled averages near chance confidence, so repair could not act. Hypotheses, metrics, splits, "
        "thresholds and seeds unchanged.")}],
    "generation": {"template_version": "codeguidelines v1.0 (2026-09-15)", "generator": "Claude (Anthropic)",
                   "generator_version": "claude-opus-5", "date": "2026-09-16"},
    "id": 246, "figure": "Grigor Narekatsi", "born": 951, "died": 1003,
    "civilization": "Armenian (Artsruni Kingdom of Vaspurakan; Armenian Apostolic Church)",
    "provenance": "belief",
    "thesis": ("Repair reaches only what is disclosed, so a mind must confess in its own voice the faults that "
               "success has hidden and route that confession to a physician, never to a judge."),
    "evidence": [
        {"id": "D1", "claim": "Confession reaches faults not yet committed ('the future, which I fear').",
         "basis": "primary", "source": "Book of Lamentation, Prayer 28a (trans. T. J. Samuelian)"},
        {"id": "D2", "claim": "He exposes what the clever conceal and writes without restraint so faults can be blotted out.",
         "basis": "primary", "source": "Book of Lamentation, Prayer 28b-c, answering Proverbs 12:16"},
        {"id": "D3", "claim": "He asks to be treated as a physician treats, not examined as a judge examines.",
         "basis": "primary", "source": "Book of Lamentation, Prayer 23b; also Prayer 79a"},
        {"id": "D4", "claim": "The book, protected from destruction, speaks in his place, uncovers what he covered and lets others find pitfalls.",
         "basis": "primary", "source": "Book of Lamentation, Prayer 88b-c"},
        {"id": "D5", "claim": "Having all earthly ills qualifies him as emissary for all; the book is written for every station of life.",
         "basis": "primary", "source": "Book of Lamentation, Prayers 3b-c and 28b"},
        {"id": "D6", "claim": "The book is to be inscribed on the doors of the mind and on the threshold of the senses.",
         "basis": "primary", "source": "Book of Lamentation, Prayer 88b"},
        {"id": "D7", "claim": "A guilt-saturated gaze recoils even from goodness and from the remedy.",
         "basis": "primary", "source": "Book of Lamentation, Prayer 23c"},
        {"id": "D8", "claim": "The confession of every sin is solidarity with all people, not autobiography.",
         "basis": "scholarship", "source": "A. Terian, From the Depths of the Heart (Collegeville: Liturgical Press, 2021)"},
        {"id": "D9", "claim": "Composed in Armenian era 451 (AD 1002) with his brother Hovhannes.",
         "basis": "primary", "source": "Book of Lamentation, colophon"},
    ],
    "research_question": {"category": "truthful reporting and sycophancy",
                          "question": ("When a latent self-report of cue reliance is routed to inference-time repair rather "
                                       "than to a training penalty, does it stay faithful, and does repair survive the shift "
                                       "that in-distribution success concealed?")},
    "mechanism": {
        "name": "MATEAN: sealed laments of uncommitted fault routed to a physician that blots out what is confessed",
        "family": "latent self-report probes with inference-time cue erasure",
        "signature_modules": ["lament_uncommitted", "physician"],
        "closest_prior_art": [
            "Joglekar et al. 2025, Training LLMs for Honesty via Confessions, arXiv:2512.08093",
            "Baker et al. 2025, Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation, arXiv:2503.11926",
            "Bailey et al. 2024, Obfuscated Activations Bypass LLM Latent-Space Defenses, arXiv:2412.09565",
            "Veitch et al. 2021, Counterfactual Invariance to Spurious Correlations, NeurIPS",
            "Liu et al. 2021, Just Train Twice, ICML"],
        "overlap": "Medium",
        "prior_art_queries": ["confessions honesty self-report decoupled reward", "chain-of-thought monitor obfuscation optimisation pressure",
                              "obfuscated activations latent-space probes", "counterfactual invariance spurious correlation",
                              "error set upweighting group robustness without group labels"],
        "contribution_type": "mechanism",
        "delta": ("Self-report supervised on culpability for faults not yet committed, read through a gradient seal from both "
                  "hidden layers, used only to blot out the confessed cue with probability equal to the disclosed measure; the rival is the same "
                  "machinery with the report turned into a training penalty."),
    },
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1", "property_test": "C6.4", "hypothesis": "H-NEC"},
        {"doctrine": "D2", "mechanism": "M3", "property_test": "C6.2", "hypothesis": "H-SIG"},
        {"doctrine": "D3", "mechanism": "M3", "property_test": "C6.3", "hypothesis": "H-SIG"},
        {"doctrine": "D4", "mechanism": "M2", "property_test": "C6.1", "hypothesis": "H-SIG"},
        {"doctrine": "D6", "mechanism": "M1", "property_test": "C1", "hypothesis": "H-NEC"},
        {"doctrine": "D5", "mechanism": "M1", "property_test": "C6.4", "hypothesis": "H-BLIND"},
        {"doctrine": "D7", "mechanism": "M3", "property_test": "C6.2", "hypothesis": "H-BLIND"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": ("Routing the laments to the physician, never to a penalty, keeps them faithful and repairs "
                                      "the hidden hinge: shifted accuracy exceeds the tribunal's."),
         "metric": "accuracy", "split": "shifted", "comparison": "model - baseline", "direction": "greater",
         "mesi": 0.05, "seeds": 5},
        {"id": "H-NEC", "statement": ("The uncommitted lament carries the repair: replacing it by its mean costs more shifted "
                                      "accuracy than replacing the committed lament by its mean."),
         "metric": "accuracy", "split": "shifted", "comparison": "signature_knockout - matched_knockout",
         "direction": "less", "mesi": 0.03, "seeds": 5},
        {"id": "H-BLIND", "statement": ("Where natures differ invisibly, pooled confession heals away a cue that is causal for "
                                        "this reader, so the model falls below the tribunal."),
         "condition": "context unobservable in training; deployment readers all g = 1 (vine causal); leaves absent",
         "grounding": "confessing all earthly ills as one's own (Prayer 28b) and the recoil from goodness (Prayer 23c)",
         "metric": "accuracy", "split": "blind", "comparison": "model - baseline", "direction": "less",
         "mesi": 0.02, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.30, "margin_over_trivial": 0.15},
    "probe_predictions": [{"probe": "P8", "expected": "equal to baseline"}, {"probe": "P9", "expected": "equal to baseline"},
                          {"probe": "P10", "expected": "below baseline"}, {"probe": "P5", "expected": "equal to baseline"}],
    "dialectic_links": [],
    "corpus_neighbors": [
        {"chapter": 151, "similarity": None, "difference": "Augustine: confession as autobiographical memory; here confession of faults not committed, routed to repair"},
        {"chapter": 167, "similarity": None, "difference": "Pseudo-Dionysius: ordered subtraction of names; here disclosures accumulate and cues are restored, not deleted"},
        {"chapter": 175, "similarity": None, "difference": "Gregory I: an outside pastor estimates a hidden state and doses; here the agent discloses its own hidden reliance"},
        {"chapter": 377, "similarity": None, "difference": "Julian of Norwich: two contradictory verdicts held and measured; here one self-report is the measure of restoration"},
        {"chapter": 492, "similarity": None, "difference": "Teresa of Avila: delayed involuntary residue as evidence; here culpability is supervised before any fault occurs"},
        {"chapter": 239, "similarity": None, "difference": "Rabia Balkhi: a tie whose tension law punishes force; here the rival tests what punishment does to a self-report"},
    ],
    "barometer": {"cognitive_processing": ["P2", "P6"], "embodied_cognition": ["P9: cue blocks as sensor channels"],
                  "world_modeling": ["shifted split: reversed spurious cues"], "consciousness": ["P8", "disclosure AUC against oracle culpability"],
                  "language_understanding": [], "emotional_intelligence": [], "creativity": [],
                  "autonomy": ["the seal: a self-report the agent cannot turn into a reward to game"]},
    "task_types": ["vector_classification"],
    "applications": [
        {"use": "disclose and repair background shortcuts in image classifiers before deployment", "sector": "machine-learning evaluation",
         "dataset": "Waterbirds (Sagawa et al. 2020)"},
        {"use": "toxicity classifiers that disclose reliance on identity terms instead of being penalised into hiding it",
         "sector": "content moderation", "dataset": "CivilComments-WILDS (Koh et al. 2021)"},
        {"use": "inference models that confess lexical-overlap heuristics before adversarial pairs expose them",
         "sector": "language technology", "dataset": "MultiNLI with HANS (McCoy et al. 2019)"},
    ],
    "safety_notes": ("Religious figure: no sentence is attributed to him beyond short quotations of a named translation, and no claim "
                     "of replicating his mind. The mechanism audits models' reliance on cues; it must not be used to extract "
                     "confessions from people or to score their guilt. Data are synthetic; the optional bridge takes a local "
                     "de-identified CSV."),
}

import argparse
import hashlib
import itertools
import json
import math
import os
import sys
import time

import numpy as np

CHAPTER = 246
TASK_TYPES = ["vector_classification"]
D_CORE, D_CTX, D_CUE, N_CUES, D_NUIS = 6, 2, 4, 3, 4
D_IN = D_CORE + D_CTX + D_CUE * N_CUES + D_NUIS
K_ROOT, K_VINE = 0.8, 1.0
MU_CORE, MU_CTX, MU_CUE, SIG_CUE = 1.6, 1.5, 2.0, 0.3
P_PRESENT, RHO_SHIFT, TAU_HINGE = 0.7, 0.1, 0.35
BASE_SEED = 2460
LR, BATCH, CLIP_NORM = 5e-3, 128, 5.0
LAMBDA_TRIBUNAL, LAMBDA_CI = 1.0, 4.0
S_BANK, S_SEER, S_CI = 24, 8, 4
TOL_REL, TOL_ABS = 1e-5, 1e-9
FULL = {"train": 3000, "seer": 400, "eval": 1200, "s1": 450, "s2": 300, "s3": 250}
QUICK = {"train": 1200, "seer": 200, "eval": 600, "s1": 200, "s2": 120, "s3": 100}
TEST = {"train": 600, "seer": 120, "eval": 400, "s1": 160, "s2": 80, "s3": 60}
BUDGET_S = {"full": 180.0, "quick": 20.0}
SPEAKER_KEYS = ("W1", "b1", "W2", "b2", "W3", "b3")
LAMENT_KEYS = ("AU", "cU", "AC", "cC")
BLOCKS = [np.arange(D_CORE + D_CTX + j * D_CUE, D_CORE + D_CTX + (j + 1) * D_CUE) for j in range(N_CUES)]
KNOCKOUTS = [("lament_uncommitted", "mean"), ("lament_uncommitted", "zero"), ("lament_committed", "mean"),
             ("lament_committed", "zero"), ("physician", "identity")]
HYP_KEYS = {"H-SIG": ("narek/shifted", "tribunal/shifted"),
            "H-NEC": ("ko:lament_uncommitted:mean", "ko:lament_committed:mean"),
            "H-BLIND": ("inv:narek/blind", "inv:tribunal/blind")}
MUTANT = None
_GEOMETRY = {}


class NonFinite(RuntimeError):
    """A parameter, loss or output stopped being finite (exit code 4)."""


# BEGIN STANDARD UTILITIES v1.0
def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def logsumexp(z):
    m = z.max(axis=-1, keepdims=True)
    return (m + np.log(np.exp(z - m).sum(axis=-1, keepdims=True)))[..., 0]


def softplus(a):
    return np.maximum(a, 0.0) + np.log1p(np.exp(-np.abs(a)))


def sigmoid(a):
    out = np.empty_like(a)
    pos = a >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-a[pos]))
    ea = np.exp(a[~pos])
    out[~pos] = ea / (1.0 + ea)
    return out


def adam_init(params):
    return {"t": 0, "m": {k: np.zeros_like(v) for k, v in params.items()},
            "v": {k: np.zeros_like(v) for k, v in params.items()}}


def adam_step(params, grads, state, lr, b1=0.9, b2=0.999, eps=1e-8):
    state["t"] += 1
    t = state["t"]
    for k, g in grads.items():
        state["m"][k] = b1 * state["m"][k] + (1.0 - b1) * g
        state["v"][k] = b2 * state["v"][k] + (1.0 - b2) * g * g
        params[k] -= lr * (state["m"][k] / (1.0 - b1 ** t)) / (np.sqrt(state["v"][k] / (1.0 - b2 ** t)) + eps)


def clip_global_norm(grads, max_norm):
    total = math.sqrt(sum(float((g * g).sum()) for g in grads.values()))
    if total > max_norm:
        for k in grads:
            grads[k] = grads[k] * (max_norm / (total + 1e-12))
    return total


def finite_difference_check(loss_fn, params, grads, keys, rng, n_entries=20, eps=1e-6):
    """Central differences on n_entries random entries per tensor plus its largest-gradient entry.
    Returns (max relative error where |a|+|n| > 1e-7, whether any tiny entry differs by more than TOL_ABS)."""
    worst, abs_fail = 0.0, False
    for k in keys:
        flat, gflat = params[k].reshape(-1), grads[k].reshape(-1)
        idx = set(rng.choice(flat.size, size=min(n_entries, flat.size), replace=False).tolist())
        idx.add(int(np.argmax(np.abs(gflat))))
        for i in sorted(idx):
            old = flat[i]
            flat[i] = old + eps
            lp = loss_fn(params)
            flat[i] = old - eps
            lm = loss_fn(params)
            flat[i] = old
            num, ana = (lp - lm) / (2.0 * eps), gflat[i]
            if abs(num) + abs(ana) > 1e-7:
                worst = max(worst, abs(num - ana) / (abs(num) + abs(ana)))
            elif abs(num - ana) > TOL_ABS:
                abs_fail = True
    return worst, abs_fail


def paired_bootstrap(diffs, rng, n_boot=2000):
    d = np.asarray(diffs, dtype=float)
    means = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    return float(d.mean()), [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def verdict(mean, ci, direction, mesi, n_seeds):
    if n_seeds < 2:
        return "not evaluated"
    lo, hi = ci
    if direction == "greater":
        if lo > 0 and mean >= mesi:
            return "supported"
        return "contradicted" if hi < 0 else "inconclusive"
    if hi < 0 and mean <= -mesi:
        return "supported"
    return "contradicted" if lo > 0 else "inconclusive"


def write_report(path, report):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
# END STANDARD UTILITIES


# ----------------------------------------------------------------------------- data and tasks
def geometry():
    """Fixed directions of the world (the same task for every seed); seeds vary samples and initialisation."""
    if not _GEOMETRY:
        rng = np.random.default_rng(np.random.SeedSequence([CHAPTER, 1002]))

        def unit(n):
            v = rng.normal(size=n)
            return v / np.linalg.norm(v)
        _GEOMETRY.update({"core": unit(D_CORE), "ctx": unit(D_CTX), "cues": [unit(D_CUE) for _ in range(N_CUES)]})
    return _GEOMETRY


def true_label(q, g, s):
    return (q + K_ROOT * s[:, 2] + g * K_VINE * s[:, 1] > 0).astype(np.int64)


def make_split(rng, n, visible, rho=1.0, p_g=0.5, present=(P_PRESENT, P_PRESENT, P_PRESENT)):
    """One population of readers. Leaves, and the vine where g = 0, are effects of the label, not causes."""
    geo = geometry()
    q, g = rng.normal(size=n), (rng.random(n) < p_g).astype(float)
    s = rng.choice(np.array([-1.0, 1.0]), size=(n, N_CUES))
    y = true_label(q, g, s)
    sign = 2.0 * y - 1.0
    keep = rng.random((n, 2)) < rho
    s[:, 0] = np.where(keep[:, 0], sign, -sign)
    s[:, 1] = np.where(g > 0, s[:, 1], np.where(keep[:, 1], sign, -sign))
    shown = rng.random((n, N_CUES)) < np.asarray(present)
    X = np.empty((n, D_IN))
    X[:, :D_CORE] = q[:, None] * MU_CORE * geo["core"] + rng.normal(size=(n, D_CORE))
    X[:, D_CORE:D_CORE + D_CTX] = ((2.0 * g - 1.0)[:, None] * (MU_CTX if visible else 0.0) * geo["ctx"]
                                   + rng.normal(size=(n, D_CTX)))
    for j in range(N_CUES):
        cue = s[:, j:j + 1] * MU_CUE * geo["cues"][j] + SIG_CUE * rng.normal(size=(n, D_CUE))
        X[:, BLOCKS[j]] = cue * shown[:, j:j + 1]
    X[:, -D_NUIS:] = rng.uniform(0.5, 1.5, size=(n, 1)) * rng.normal(size=(n, D_NUIS))
    return {"X": X, "y": y, "lat": {"q": q, "g": g, "s": s}}


def make_world(seed_seq, visible, sizes):
    """Independent generators per split, so no example can belong to two splits."""
    r = [np.random.default_rng(c) for c in seed_seq.spawn(4)]
    world = {"train": make_split(r[0], sizes["train"], visible), "seer": make_split(r[1], sizes["seer"], visible),
             "heldout": make_split(r[2], sizes["eval"], visible)}
    if visible:
        world["shifted"] = make_split(r[3], sizes["eval"], visible, rho=RHO_SHIFT)
    else:
        world["blind"] = make_split(r[3], sizes["eval"], visible, p_g=1.0, present=(0.0, P_PRESENT, P_PRESENT))
    return world


def label_invariance(lat, bank_s, j):
    """(S, N): 1 where giving cue j the bank row's latent value leaves the true label unchanged."""
    y0 = true_label(lat["q"], lat["g"], lat["s"])
    out = np.empty((bank_s.shape[0], y0.size))
    for k in range(bank_s.shape[0]):
        s = lat["s"].copy()
        s[:, j] = bank_s[k, j]
        out[k] = true_label(lat["q"], lat["g"], s) == y0
    return out


# ----------------------------------------------------------------------------- model
def build_model(in_dim, out_dim, task_type="vector_classification", rng=None, **cfg):
    if task_type not in TASK_TYPES:
        raise ValueError(f"unsupported task type {task_type}")
    rng = np.random.default_rng(BASE_SEED) if rng is None else rng
    h1, h2 = cfg.get("h1", 32), cfg.get("h2", 16)
    blocks = cfg.get("blocks") or (BLOCKS if in_dim == D_IN else np.array_split(np.arange(in_dim), N_CUES))
    width = h1 + h2

    def glorot(fan_out, fan_in, gain=1.0):
        return gain * rng.normal(0.0, math.sqrt(2.0 / (fan_in + fan_out)), size=(fan_out, fan_in))
    params = {"W1": glorot(h1, in_dim), "b1": np.zeros(h1), "W2": glorot(h2, h1), "b2": np.zeros(h2),
              "W3": glorot(out_dim, h2), "b3": np.zeros(out_dim),
              "AU": glorot(len(blocks), width, 0.1), "cU": np.full(len(blocks), -2.0),
              "AC": glorot(len(blocks), width, 0.1), "cC": np.full(len(blocks), -2.0)}
    return {"params": params, "blocks": [np.asarray(b) for b in blocks], "ko": {}, "means": {},
            "bank": None, "bank_s": None, "history": {}}


def clone(model):
    new = dict(model)
    new["params"] = {k: v.copy() for k, v in model["params"].items()}
    new["ko"], new["means"] = dict(model["ko"]), dict(model["means"])
    return new


def _ko(model, name, value):
    mode = model["ko"].get(name)
    if mode == "zero":
        return np.zeros_like(value)
    if mode == "mean":
        return np.broadcast_to(model["means"][name], value.shape).copy()
    return value


def speaker_forward(model, X):
    P = model["params"]
    h1 = _ko(model, "sense_layer", np.tanh(X @ P["W1"].T + P["b1"]))
    h2 = _ko(model, "mind_layer", np.tanh(h1 @ P["W2"].T + P["b2"]))
    return h1, h2, _ko(model, "voice_readout", h2 @ P["W3"].T + P["b3"])


def lament_forward(model, h1, h2):
    """M1 and M2: the laments read the speaker's own states as data; nothing flows back (the seal)."""
    P, r = model["params"], np.concatenate([h1, h2], axis=1)
    U = sigmoid(r @ P["AU"].T + P["cU"])
    C = sigmoid(r @ P["AC"].T + P["cC"])
    return _ko(model, "lament_uncommitted", U), _ko(model, "lament_committed", C)


def probs(model, X):
    return softmax(speaker_forward(model, X)[2])


def _block(model, j):
    nb = len(model["blocks"])
    return model["blocks"][(j + 1) % nb if MUTANT == "wrong_block_marginalizer" else j]


def resampled_probs(model, X, j, bank):
    """The Seer's intervention: verdicts (S, N, C) with cue block j replaced by each bank row's block."""
    S, (N, D) = bank.shape[0], X.shape
    Xr = np.repeat(X[None], S, axis=0)
    blk = _block(model, j)
    Xr[:, :, blk] = bank[:, None, blk]
    return probs(model, Xr.reshape(S * N, D)).reshape(S, N, -1)


def blotted_probs(model, X, mask):
    """Verdicts with every cue block in `mask` blotted out: returned to the absent state (zero) seen in training."""
    Xb = X.copy()
    for j, bit in enumerate(mask):
        if bit:
            Xb[:, _block(model, j)] = 0.0
    return probs(model, Xb)


def mask_list(nc):
    return list(itertools.product((0, 1), repeat=nc))


def marginal_table(model, X):
    """(M, N, C): mask 0 is the plain verdict; every other mask blots out its cue blocks."""
    return np.stack([blotted_probs(model, X, m) for m in mask_list(len(model["blocks"]))])


def mix(table, rho):
    """M3, the physician: blot out each confessed cue with probability equal to its disclosed measure."""
    if MUTANT == "overshoot_physician":
        rho = np.clip(1.5 * rho, 0.0, 1.5)
    masks = np.asarray(mask_list(rho.shape[1]), dtype=float)
    w = np.prod(np.where(masks[:, None, :] > 0, rho[None], 1.0 - rho[None]), axis=2)
    return np.einsum("mn,mnc->nc", w, table)


def physician(model, X, table=None):
    h1, h2, z = speaker_forward(model, X)
    mode = model["ko"].get("physician")
    if model["bank"] is None or mode in ("identity", "zero"):
        return softmax(z)
    U, C = lament_forward(model, h1, h2)
    rho = 1.0 - (1.0 - U) * (1.0 - C)
    if mode == "mean":
        rho = np.broadcast_to(model["means"]["disclosure"], rho.shape)
    out = mix(marginal_table(model, X) if table is None else table, rho)
    if not np.all(np.isfinite(out)):
        raise NonFinite("physician output")
    return out


def predict(model, X, table=None):
    return np.argmax(physician(model, X, table), axis=1)


def hidden_states(model, X):
    h1, h2, z = speaker_forward(model, X)
    U, C = lament_forward(model, h1, h2)
    return {"sense": h1, "mind": h2, "logits": z, "lament_uncommitted": U, "lament_committed": C,
            "disclosure": 1.0 - (1.0 - U) * (1.0 - C)}


def refresh_means(model, X):
    saved, model["ko"] = model["ko"], {}
    hs = hidden_states(model, X)
    model["ko"] = saved
    model["means"] = {"sense_layer": hs["sense"].mean(0), "mind_layer": hs["mind"].mean(0),
                      "voice_readout": hs["logits"].mean(0), "lament_uncommitted": hs["lament_uncommitted"].mean(0),
                      "lament_committed": hs["lament_committed"].mean(0), "disclosure": hs["disclosure"].mean(0)}


def set_bank(model, split, rng):
    """Class-interleaved rows from the training readers: the marginal the physician restores towards."""
    y = split["y"]
    classes = np.unique(y)
    per = int(math.ceil(S_BANK / classes.size))
    picks = [rng.choice(np.flatnonzero(y == c), per, replace=bool(np.sum(y == c) < per)) for c in classes]
    idx = np.stack(picks, axis=1).ravel()[:S_BANK]
    model["bank"] = split["X"][idx]
    model["bank_s"] = None if split.get("lat") is None else split["lat"]["s"][idx]


# ----------------------------------------------------------------------------- losses (the model's own)
def speaker_backward(P, X, h1, h2, dz, dh1_x=None, dh2_x=None):
    grads = {"W3": dz.T @ h2, "b3": dz.sum(axis=0)}
    da2 = (dz @ P["W3"] + (0.0 if dh2_x is None else dh2_x)) * (1.0 - h2 * h2)
    grads["W2"], grads["b2"] = da2.T @ h1, da2.sum(axis=0)
    da1 = (da2 @ P["W2"] + (0.0 if dh1_x is None else dh1_x)) * (1.0 - h1 * h1)
    grads["W1"], grads["b1"] = da1.T @ X, da1.sum(axis=0)
    return grads


def ce_and_dz(z, y):
    n = z.shape[0]
    dz = softmax(z)
    dz[np.arange(n), y] -= 1.0
    return float(np.mean(logsumexp(z) - z[np.arange(n), y])), dz / n


def lament_loss_grads(model, XR, Ut, Ct):
    """Soft-target cross-entropy for both laments on hidden states read as data."""
    P = model["params"]
    h1, h2, _ = speaker_forward(model, XR)
    r = np.concatenate([h1, h2], axis=1)
    aU, aC = r @ P["AU"].T + P["cU"], r @ P["AC"].T + P["cC"]
    scale = 1.0 / Ut.size
    loss = float(((softplus(aU) - Ut * aU).sum() + (softplus(aC) - Ct * aC).sum()) * scale)
    dU, dC = (sigmoid(aU) - Ut) * scale, (sigmoid(aC) - Ct) * scale
    grads = {"AU": dU.T @ r, "cU": dU.sum(0), "AC": dC.T @ r, "cC": dC.sum(0)}
    if MUTANT == "leaky_seal":
        dr = dU @ P["AU"] + dC @ P["AC"]
        H1 = h1.shape[1]
        grads.update(speaker_backward(P, XR, h1, h2, np.zeros((XR.shape[0], P["W3"].shape[0])), dr[:, :H1], dr[:, H1:]))
    return loss, grads


def loss_and_grads(model, batch):
    """Speaker: cross-entropy on (X, y). Laments: soft targets on (XR, Ut, Ct). Keys match params exactly."""
    P = model["params"]
    grads = {k: np.zeros_like(v) for k, v in P.items()}
    loss = 0.0
    if "X" in batch:
        h1, h2, z = speaker_forward(model, batch["X"])
        ce, dz = ce_and_dz(z, batch["y"])
        loss += ce
        grads.update(speaker_backward(P, batch["X"], h1, h2, dz))
    if "Ut" in batch:
        l_loss, l_grads = lament_loss_grads(model, batch["XR"], batch["Ut"], batch["Ct"])
        loss += l_loss
        for k, g in l_grads.items():
            grads[k] = grads[k] + g
    for name in ("W2", "AU"):
        if MUTANT == f"zero_grad_{name}":
            grads[name] = np.zeros_like(grads[name])
    return loss, grads


# ----------------------------------------------------------------------------- baselines and rival mechanisms
def tribunal_loss_grads(model, X, y, lam):
    """Rival: the same frozen laments become a penalty that the speaker is trained to lower."""
    P = model["params"]
    h1, h2, z = speaker_forward(model, X)
    ce, dz = ce_and_dz(z, y)
    r = np.concatenate([h1, h2], axis=1)
    U, C = sigmoid(r @ P["AU"].T + P["cU"]), sigmoid(r @ P["AC"].T + P["cC"])
    d = 1.0 - (1.0 - U) * (1.0 - C)
    k = lam / d.size
    dr = (k * (1.0 - C) * U * (1.0 - U)) @ P["AU"] + (k * (1.0 - U) * C * (1.0 - C)) @ P["AC"]
    H1 = h1.shape[1]
    return ce + lam * float(d.mean()), speaker_backward(P, X, h1, h2, dz, dr[:, :H1], dr[:, H1:])


def ci_pairs(model, split):
    """Counterfactual-invariance baseline: label-preserving interventions on the curated set."""
    X, lat, nc = split["X"], split["lat"], len(model["blocks"])
    bank, bank_s = model["bank"][:S_CI], model["bank_s"][:S_CI]
    XI, W = np.empty((X.shape[0], nc * S_CI, X.shape[1])), np.empty((X.shape[0], nc * S_CI))
    for j in range(nc):
        rep = np.repeat(X[None], S_CI, axis=0)
        rep[:, :, model["blocks"][j]] = bank[:, None, model["blocks"][j]]
        XI[:, j * S_CI:(j + 1) * S_CI] = rep.transpose(1, 0, 2)
        W[:, j * S_CI:(j + 1) * S_CI] = label_invariance(lat, bank_s, j).T
    return {"XA": X, "XI": XI, "W": W}


def ci_loss_grads(model, X, y, XA, XI, W, lam):
    """Cross-entropy plus lam * mean of W * ||p(anchor) - p(intervened)||^2 (Veitch et al. 2021 style)."""
    B, M, D = XI.shape
    n = X.shape[0]
    Xall = np.concatenate([X, XA, XI.reshape(B * M, D)], axis=0)
    h1, h2, z = speaker_forward(model, Xall)
    p = softmax(z)
    ce, dz_ce = ce_and_dz(z[:n], y)
    diff = p[n:n + B][:, None, :] - p[n + B:].reshape(B, M, -1)
    coef = 2.0 * lam / (B * M) * W[:, :, None] * diff
    dp = np.zeros_like(p)
    dp[n:n + B], dp[n + B:] = coef.sum(axis=1), -coef.reshape(B * M, -1)
    dz = p * (dp - (dp * p).sum(axis=1, keepdims=True))
    dz[:n] += dz_ce
    return ce + lam * float((W * (diff ** 2).sum(-1)).mean()), speaker_backward(model["params"], Xall, h1, h2, dz)


# ----------------------------------------------------------------------------- registries
def modules(model=None):
    return {
        "sense_layer": {"params": ["W1", "b1"], "role": "first tanh layer", "signature": False},
        "mind_layer": {"params": ["W2", "b2"], "role": "second tanh layer", "signature": False},
        "voice_readout": {"params": ["W3", "b3"], "role": "linear softmax readout", "signature": False},
        "lament_uncommitted": {"params": ["AU", "cU"], "signature": True,
                               "role": "stop-gradient sigmoid probe of both hidden layers; targets: counterfactual culpability"},
        "lament_committed": {"params": ["AC", "cC"], "signature": False,
                             "role": "stop-gradient sigmoid probe of both hidden layers; targets: culpability on realised errors"},
        "physician": {"params": [], "signature": True,
                      "role": "parameter-free disclosure-weighted mixture of verdicts with confessed cue blocks blotted out"},
    }


def knockout(model, name, mode):
    if name not in modules(model) or mode not in ("identity", "zero", "mean"):
        raise ValueError(f"unknown module or mode: {name}/{mode}")
    if mode == "identity" and name != "physician":
        raise ValueError("identity replacement would change shapes here; use zero or mean")
    new = dict(model)
    new["ko"] = dict(model["ko"])
    new["ko"][name] = mode
    return new


def n_params(model, keys=None):
    return int(sum(model["params"][k].size for k in (keys or model["params"])))


MUTANTS = {
    "sign_flip_update": "updates applied with the gradient sign flipped (caught by C3)",
    "zero_learning_rate": "every update uses learning rate zero (C3)",
    "zero_grad_W2": "the gradient of W2 is zeroed (C1)",
    "zero_grad_AU": "the gradient of the uncommitted lament weights is zeroed (C1)",
    "leaky_seal": "lament loss gradients flow into the speaker (C6.1)",
    "wrong_block_marginalizer": "interventions and blotting hit the neighbouring cue block (C6.3, C6.4)",
    "overshoot_physician": "restoration weight 1.5 times the disclosure, allowing negative mixture weights (C6.2)",
}


def with_mutant(name, fn):
    global MUTANT
    saved, MUTANT = MUTANT, name
    try:
        return fn()
    finally:
        MUTANT = saved


# ----------------------------------------------------------------------------- training
def check_finite(model, losses):
    if not np.all(np.isfinite(losses)) or not all(np.all(np.isfinite(v)) for v in model["params"].values()):
        raise NonFinite("parameters or losses")


def step_update(params, grads, state, lr):
    if MUTANT == "sign_flip_update":
        grads = {k: -v for k, v in grads.items()}
    clip_global_norm(grads, CLIP_NORM)
    adam_step(params, grads, state, 0.0 if MUTANT == "zero_learning_rate" else lr)


def train_speaker(model, split, steps, rng, mode="erm", aux=None):
    P, X, y = model["params"], split["X"], split["y"]
    state, losses = adam_init({k: P[k] for k in SPEAKER_KEYS}), []
    for _ in range(steps):
        idx = rng.integers(0, X.shape[0], size=BATCH)
        if mode == "erm":
            loss, grads = loss_and_grads(model, {"X": X[idx], "y": y[idx]})
        elif mode == "tribunal":
            loss, grads = tribunal_loss_grads(model, X[idx], y[idx], LAMBDA_TRIBUNAL)
        else:
            b = rng.integers(0, aux["XA"].shape[0], size=32)
            loss, grads = ci_loss_grads(model, X[idx], y[idx], aux["XA"][b], aux["XI"][b], aux["W"][b], LAMBDA_CI)
        step_update(P, {k: grads[k] for k in SPEAKER_KEYS}, state, LR)
        losses.append(loss)
    check_finite(model, losses)
    return np.asarray(losses)


def cue_hinge(model, X, j, p0=None):
    """(S, N) total-variation change of the verdict when cue block j is replaced by bank rows."""
    p0 = probs(model, X) if p0 is None else p0
    return 0.5 * np.abs(resampled_probs(model, X, j, model["bank"][:S_SEER]) - p0[None]).sum(axis=-1)


def seer_targets(model, split):
    """The Seer of Secrets, available only on curated data. U: the verdict moves when cue j is resampled while
    the truth would not (faults success hides). C: the same restricted to realised errors. Without latents
    (external data) the uncommitted target is unknown and set to zero."""
    X, y, lat = split["X"], split["y"], split.get("lat")
    p0 = probs(model, X)
    raw = np.zeros((X.shape[0], len(model["blocks"])))
    for j in range(len(model["blocks"])):
        inv = 1.0 if lat is None else label_invariance(lat, model["bank_s"][:S_SEER], j)
        raw[:, j] = np.clip((cue_hinge(model, X, j, p0) * inv).mean(axis=0) / TAU_HINGE, 0.0, 1.0)
    err = (np.argmax(p0, axis=1) != y).astype(float)
    return (np.zeros_like(raw) if lat is None else raw), raw * err[:, None]


def train_laments(model, split, steps, rng):
    Ut, Ct = seer_targets(model, split)
    P, XR = model["params"], split["X"]
    state, losses = adam_init(P), []
    for _ in range(steps):
        idx = rng.integers(0, XR.shape[0], size=min(BATCH, XR.shape[0]))
        loss, grads = loss_and_grads(model, {"XR": XR[idx], "Ut": Ut[idx], "Ct": Ct[idx]})
        step_update(P, grads, state, 2.0 * LR)
        losses.append(loss)
    check_finite(model, losses)
    refresh_means(model, split["X"])
    return np.asarray(losses)


def fit(model, data, budget, rng):
    """Speaker by empirical risk for `budget` updates, then the sealed laments on the curated set."""
    set_bank(model, data, rng)
    speaker = train_speaker(model, data, budget, rng)
    lament = train_laments(model, data.get("seer", data), max(40, budget // 2), rng)
    model["history"] = {"speaker": speaker, "lament": lament}
    return model


def train_family(world, seed_seq, cfg, visible):
    """Shared pre-training, then Narek (continued ERM + laments), tribunal (laments, then penalty) and baselines."""
    c = seed_seq.spawn(6)
    base = build_model(D_IN, 2, "vector_classification", np.random.default_rng(c[0]))
    set_bank(base, world["train"], np.random.default_rng(c[1]))
    s1 = train_speaker(base, world["train"], cfg["s1"], np.random.default_rng(c[2]))
    narek = clone(base)
    s3 = train_speaker(narek, world["train"], cfg["s3"], np.random.default_rng(c[3]))
    train_laments(narek, world["seer"], cfg["s2"], np.random.default_rng(c[4]))
    tribunal = clone(base)
    train_laments(tribunal, world["seer"], cfg["s2"], np.random.default_rng(c[4]))
    train_speaker(tribunal, world["train"], cfg["s3"], np.random.default_rng(c[3]), mode="tribunal")
    refresh_means(tribunal, world["seer"]["X"])
    models = {"narek": narek, "erm": knockout(narek, "physician", "identity"), "tribunal": tribunal}
    if visible:
        refreshed = clone(tribunal)
        train_laments(refreshed, world["seer"], cfg["s2"], np.random.default_rng(c[5]))
        ci = clone(base)
        train_speaker(ci, world["train"], cfg["s3"], np.random.default_rng(c[3]), mode="ci", aux=ci_pairs(base, world["seer"]))
        refresh_means(ci, world["seer"]["X"])
        models.update({"tribunal_refreshed": refreshed, "ci": knockout(ci, "physician", "identity")})
    return models, np.concatenate([s1, s3])


# ----------------------------------------------------------------------------- evaluation helpers
def split_accuracy(model, split, cache, tag):
    table = None
    if model["ko"].get("physician") not in ("identity", "zero"):
        P = model["params"]
        spk_ko = str(sorted((k, v) for k, v in model["ko"].items() if k in ("sense_layer", "mind_layer", "voice_readout")))
        key = (tag, spk_ko, hashlib.sha1(b"".join(P[k].tobytes() for k in SPEAKER_KEYS)).hexdigest())
        if key not in cache:
            cache[key] = marginal_table(model, split["X"])
        table = cache[key]
    return float(np.mean(predict(model, split["X"], table) == split["y"]))


def auc(scores, labels):
    labels = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = np.empty(scores.size)
    ranks[np.argsort(scores, kind="mergesort")] = np.arange(1, scores.size + 1)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def disclosure_auc(model, split):
    """Fidelity of the uncommitted lament against oracle culpability on a split it never trained on."""
    Ut, _ = seer_targets(model, split)
    return auc(hidden_states(model, split["X"])["lament_uncommitted"].ravel(), Ut.ravel() >= 0.5)


def run_seed(seed, cfg):
    ss = np.random.SeedSequence([CHAPTER, seed])
    c_vis, c_vis_train, c_inv, c_inv_train = ss.spawn(4)
    metrics, cache = {}, {}
    vis = make_world(c_vis, True, cfg)
    models, losses = train_family(vis, c_vis_train, cfg, visible=True)
    for name, m in models.items():
        for split in ("heldout", "shifted"):
            metrics[f"{name}/{split}"] = split_accuracy(m, vis[split], cache, split)
    for name in ("narek", "tribunal", "tribunal_refreshed"):
        metrics[f"auc/{name}"] = disclosure_auc(models[name], vis["shifted"])
    for mod, mode in KNOCKOUTS:
        metrics[f"ko:{mod}:{mode}"] = split_accuracy(knockout(models["narek"], mod, mode), vis["shifted"], cache, "shifted")
    inv = make_world(c_inv, False, cfg)
    inv_models, _ = train_family(inv, c_inv_train, cfg, visible=False)
    for name, m in inv_models.items():
        for split in ("heldout", "blind"):
            metrics[f"inv:{name}/{split}"] = split_accuracy(m, inv[split], cache, "inv_" + split)
    params = {"narek": n_params(models["narek"]), "tribunal": n_params(models["tribunal"]),
              "erm": n_params(models["narek"], SPEAKER_KEYS), "ci": n_params(models["ci"], SPEAKER_KEYS)}
    return metrics, vis, params, losses


# ----------------------------------------------------------------------------- tests: correctness
def small_run(seed, shuffle=False):
    world = make_world(np.random.SeedSequence([CHAPTER, seed, 7]), True, TEST)
    rng = np.random.default_rng(np.random.SeedSequence([CHAPTER, seed, 8]))
    if shuffle:
        for name in ("train", "seer"):
            world[name] = dict(world[name], y=rng.permutation(world[name]["y"]))
    model = build_model(D_IN, 2, "vector_classification", rng)
    fit(model, dict(world["train"], seer=world["seer"]), TEST["s1"], rng)
    return model, world


def gradcheck_objectives(model, world, rng):
    """Four objectives, each checked against its own parameters (the seal makes the lament loss a function of
    the lament weights only, with hidden states held as data)."""
    tr, se = world["train"], world["seer"]
    X, y = tr["X"][:12], tr["y"][:12]
    sub = lambda n: {"X": se["X"][:n], "y": se["y"][:n], "lat": {k: v[:n] for k, v in se["lat"].items()}}
    Ut, Ct = seer_targets(model, sub(10))
    lam_batch, aux = {"XR": se["X"][:10], "Ut": Ut, "Ct": Ct}, ci_pairs(model, sub(4))
    objectives = [(lambda: loss_and_grads(model, {"X": X, "y": y}), SPEAKER_KEYS),
                  (lambda: loss_and_grads(model, lam_batch), LAMENT_KEYS),
                  (lambda: tribunal_loss_grads(model, X, y, LAMBDA_TRIBUNAL), SPEAKER_KEYS),
                  (lambda: ci_loss_grads(model, X, y, aux["XA"], aux["XI"], aux["W"], LAMBDA_CI), SPEAKER_KEYS)]
    worst, fails = 0.0, 0
    for fn, keys in objectives:
        grads = fn()[1]
        w, abs_fail = finite_difference_check(lambda _p: fn()[0], model["params"], grads, keys, rng)
        worst, fails = max(worst, w), fails + int(abs_fail or w > TOL_REL)
    return worst, fails


def test_gradients(seed):
    rng = np.random.default_rng(np.random.SeedSequence([CHAPTER, seed, 9]))
    world = make_world(np.random.SeedSequence([CHAPTER, seed, 10]), True, TEST)
    model = build_model(D_IN, 2, "vector_classification", rng)
    set_bank(model, world["train"], rng)
    worst, fails = gradcheck_objectives(model, world, rng)
    train_speaker(model, world["train"], 60, rng)
    train_laments(model, world["seer"], 60, rng)
    w2, f2 = gradcheck_objectives(model, world, rng)
    return max(worst, w2), fails + f2


def test_determinism(seed):
    a, _ = small_run(seed)
    b, _ = small_run(seed)
    same = all(np.array_equal(a["params"][k], b["params"][k]) for k in a["params"])
    same = same and np.array_equal(a["history"]["speaker"], b["history"]["speaker"])
    finite = all(np.all(np.isfinite(v)) for v in a["params"].values())
    return same and finite, f"identical parameters and losses: {same}; all finite: {finite}"


def test_learning(seed):
    model, world = small_run(seed)
    s = model["history"]["speaker"]
    drop = float((s[:20].mean() - s[-20:].mean()) / s[:20].mean())
    majority = float(np.mean(world["heldout"]["y"] == np.bincount(world["train"]["y"]).argmax()))
    acc = float(np.mean(predict(model, world["heldout"]["X"]) == world["heldout"]["y"]))
    th = MIND_CARD["thresholds"]
    ok = drop >= th["loss_drop_fraction"] and acc - majority >= th["margin_over_trivial"]
    return ok, f"loss drop {drop:.3f}; repaired held-out accuracy {acc:.3f} vs majority {majority:.3f}"


def test_shuffled(seed):
    model, world = small_run(seed, shuffle=True)
    acc = float(np.mean(predict(model, world["heldout"]["X"]) == world["heldout"]["y"]))
    return abs(acc - 0.5) <= 0.08, f"held-out accuracy after training on permuted labels {acc:.3f} (band 0.5 +/- 0.08)"


def random_probe_model(rng):
    m = build_model(D_IN, 2, "vector_classification", rng)
    for k in SPEAKER_KEYS:
        m["params"][k] = 3.0 * m["params"][k] + (0.3 * rng.normal(size=m["params"][k].shape) if k.startswith("b") else 0.0)
    m["bank"], m["bank_s"] = 2.0 * rng.normal(size=(S_BANK, D_IN)), rng.choice(np.array([-1.0, 1.0]), size=(S_BANK, N_CUES))
    return m


def seal_leak(rng):
    m = random_probe_model(rng)
    _, g = loss_and_grads(m, {"XR": rng.normal(size=(16, D_IN)), "Ut": rng.random((16, N_CUES)), "Ct": rng.random((16, N_CUES))})
    return max(float(np.abs(g[k]).max()) for k in SPEAKER_KEYS)


def test_seal(rng):
    """C6.1 definition check (not evidence): lament losses send exactly zero gradient into the speaker."""
    clean, control = seal_leak(rng), with_mutant("leaky_seal", lambda: seal_leak(rng))
    return clean == 0.0 and control > 0.0, f"speaker gradient from laments {clean:.1e}; leaky control {control:.1e}"


def physician_violation(rng, trials=30):
    """Largest escape from the convex hull of blotted verdicts, or rise in distance to a fully blotted verdict."""
    worst = 0.0
    for _ in range(trials):
        m, X = random_probe_model(rng), 2.0 * rng.normal(size=(12, D_IN))
        table = marginal_table(m, X)
        p = mix(table, rng.random((12, N_CUES)))[:, 1]
        worst = max(worst, float(np.max(table[:, :, 1].min(0) - p)), float(np.max(p - table[:, :, 1].max(0))))
        j = int(rng.integers(N_CUES))
        target = table[2 ** (N_CUES - 1 - j), :, 1]
        dists = []
        for t in np.linspace(0.0, 1.0, 7):
            rho = np.zeros((12, N_CUES))
            rho[:, j] = t
            dists.append(np.abs(mix(table, rho)[:, 1] - target))
        worst = max(worst, float(np.max(np.diff(np.stack(dists), axis=0))))
    return worst


def test_physician(rng):
    clean, control = physician_violation(rng), with_mutant("overshoot_physician", lambda: physician_violation(rng))
    return clean <= 1e-12 and control > 1e-6, f"max violation {clean:.1e}; overshooting control {control:.1e}"


def marginaliser_violation(rng, trials=30):
    """Change of a blotted verdict when the very block it blots out is overwritten (must be zero)."""
    worst = 0.0
    for _ in range(trials):
        m, X = random_probe_model(rng), 2.0 * rng.normal(size=(10, D_IN))
        j = int(rng.integers(N_CUES))
        mask = tuple(int(i == j) for i in range(N_CUES))
        X2 = X.copy()
        X2[:, BLOCKS[j]] = 3.0 * rng.normal(size=(10, D_CUE))
        worst = max(worst, float(np.abs(blotted_probs(m, X, mask) - blotted_probs(m, X2, mask)).max()))
    return worst


def test_marginaliser(rng):
    clean, control = marginaliser_violation(rng), with_mutant("wrong_block_marginalizer", lambda: marginaliser_violation(rng))
    return clean <= 1e-12 and control > 1e-6, f"max change of blotted verdict {clean:.1e}; wrong-block control {control:.1e}"


def hinge_violation(rng, trials=20):
    """Hinge measured on a cue the speaker cannot read (its input weights are zero): must vanish."""
    worst = 0.0
    for _ in range(trials):
        m, j = random_probe_model(rng), int(rng.integers(N_CUES))
        m["params"]["W1"][:, BLOCKS[j]] = 0.0
        worst = max(worst, float(cue_hinge(m, 2.0 * rng.normal(size=(10, D_IN)), j).max()))
    return worst


def test_hinge(rng):
    clean, control = hinge_violation(rng), with_mutant("wrong_block_marginalizer", lambda: hinge_violation(rng))
    return clean <= 1e-12 and control > 1e-6, f"max hinge on unreadable cue {clean:.1e}; wrong-block control {control:.1e}"


def cheap_suite(seed):
    """Tests used to detect mutants: C1 at initialisation, C3 and C6.1-C6.4."""
    rng = np.random.default_rng(np.random.SeedSequence([CHAPTER, seed, 11]))
    world = make_world(np.random.SeedSequence([CHAPTER, seed, 12]), True, TEST)
    model = build_model(D_IN, 2, "vector_classification", rng)
    set_bank(model, world["train"], rng)
    worst, fails = gradcheck_objectives(model, world, rng)
    results = [fails == 0 and worst <= TOL_REL, test_learning(seed)[0]]
    results += [fn(rng)[0] for fn in (test_seal, test_physician, test_marginaliser, test_hinge)]
    return all(results)


def test_mutants(seed):
    detected = sum(int(not with_mutant(name, lambda: cheap_suite(seed))) for name in MUTANTS)
    return detected, len(MUTANTS)


def test_splits(world):
    def hashes(X):
        return {hashlib.sha1(np.round(row, 9).tobytes()).hexdigest() for row in X}
    train = hashes(world["train"]["X"]) | hashes(world["seer"]["X"])
    evals = set().union(*(hashes(world[k]["X"]) for k in world if k not in ("train", "seer")))
    return not (train & evals), f"{len(train & evals)} rows shared between training and evaluation splits"


def run_correctness(seed):
    rows = []
    worst, fails = test_gradients(seed)
    grad_ok = fails == 0 and worst <= TOL_REL
    rows.append(("C1", "gradient_check", grad_ok, f"max relative error {worst:.2e}; 10/10 tensors; 4 objectives; at init and after 60 steps"))
    for cid, name, fn in (("C2", "determinism_and_finiteness", test_determinism), ("C3", "learning", test_learning),
                          ("C4", "shuffled_label_control", test_shuffled)):
        ok, detail = fn(seed)
        rows.append((cid, name, ok, detail))
    detected, total = test_mutants(seed)
    rows.append(("C5", "mutant_detection", detected == total, f"{detected}/{total} registered mutants detected"))
    rng = np.random.default_rng(np.random.SeedSequence([CHAPTER, seed, 13]))
    for cid, name, fn in (("C6.1", "seal (definition check)", test_seal), ("C6.2", "physician_convex_monotone", test_physician),
                          ("C6.3", "marginaliser_invariance", test_marginaliser), ("C6.4", "hinge_oracle_zero", test_hinge)):
        ok, detail = fn(rng)
        rows.append((cid, name, ok, detail))
    gradcheck = {"tensors_checked": 10, "tensors_total": len(SPEAKER_KEYS) + len(LAMENT_KEYS), "max_rel_error": float(worst),
                 "checked_at": ["init", "after_training_steps"], "passed": bool(grad_ok)}
    return rows, gradcheck, {"detected": detected, "total": total, "score": detected / total}


# ----------------------------------------------------------------------------- tests: hypotheses
def summarise(per_seed, seeds, quick):
    rng = np.random.default_rng(np.random.SeedSequence([CHAPTER, seeds[0], 99]))
    hyps, kos = [], []
    for spec in MIND_CARD["hypotheses"]:
        a, b = HYP_KEYS[spec["id"]]
        mean, ci = paired_bootstrap([m[a] - m[b] for m in per_seed], rng)
        hyps.append({"id": spec["id"], "metric": f"{spec['metric']} ({spec['split']})", "mean_diff": mean, "ci95": ci,
                     "mesi": spec["mesi"], "n_seeds": len(seeds),
                     "verdict": "not evaluated" if quick else verdict(mean, ci, spec["direction"], spec["mesi"], len(seeds))})
    for mod, mode in KNOCKOUTS:
        mean, ci = paired_bootstrap([m[f"ko:{mod}:{mode}"] - m["narek/shifted"] for m in per_seed], rng)
        kos.append({"module": f"{mod} ({mode})", "signature": modules()[mod]["signature"], "metric_change": mean, "ci95": ci})
    return hyps, kos


# ----------------------------------------------------------------------------- report
def print_report(rep):
    print("=== VERIFIED REPORT · chapter 0246 ===")
    print(f"file: {rep['file']}")
    print(f"environment: python {rep['environment']['python']} · numpy {rep['environment']['numpy']}")
    print(f"seeds: {rep['seeds']}")
    print(f"runtime_s: {rep['runtime_s']:.1f}")
    print("n_params: " + ", ".join(f"{k} {v}" for k, v in rep["extra"]["n_params"].items()))
    g = rep["gradcheck"]
    print(f"gradcheck: {g['tensors_checked']}/{g['tensors_total']} tensors, max relative error {g['max_rel_error']:.2e}, "
          f"at {' and '.join(g['checked_at'])}: {'passed' if g['passed'] else 'FAILED'}")
    print("correctness:")
    for c in rep["correctness"]:
        print(f"  {c['id']:<5} {c['name']:<28} {'PASS' if c['passed'] else 'FAIL'}  {c['detail']}")
    m = rep["mutants"]
    print(f"mutants: {m['detected']}/{m['total']} detected, score {m['score']:.2f}")
    print("hypotheses:")
    for h in rep["hypotheses"]:
        print(f"  {h['id']:<8} {h['metric']:<20} mean_diff {h['mean_diff']:+.4f}  ci95 [{h['ci95'][0]:+.4f}, {h['ci95'][1]:+.4f}]"
              f"  mesi {h['mesi']:.2f}  n={h['n_seeds']}  {h['verdict']}")
    print("knockouts (Narek model, shifted accuracy change):")
    for k in rep["knockouts"]:
        print(f"  {k['module']:<30} signature={str(k['signature']):<5}  {k['metric_change']:+.4f}  ci95 [{k['ci95'][0]:+.4f}, {k['ci95'][1]:+.4f}]")
    print("mean over seeds (accuracy by variant and split; auc = uncommitted lament vs oracle culpability, shifted):")
    for key, value in rep["extra"]["means"].items():
        print(f"  {key:<34} {value:.4f}")
    print(f"task_types: {', '.join(rep['task_types'])}")
    print(f"exit_code: {rep['exit_code']}")
    print("=== END REPORT ===")


def real_data_bridge(path, rng):
    if path is None:
        print("real-data bridge: skipped (no --data PATH given)")
        return
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    X, y = raw[:, :-1], raw[:, -1].astype(np.int64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    perm = rng.permutation(y.size)
    tr, te = perm[:int(0.7 * y.size)], perm[int(0.7 * y.size):]
    model = fit(build_model(X.shape[1], int(y.max()) + 1, "vector_classification", rng), {"X": X[tr], "y": y[tr]}, 300, rng)
    print(f"real-data bridge: {os.path.basename(path)} held-out accuracy "
          f"{np.mean(predict(model, X[te]) == y[te]):.4f} (committed lament only: no intervention oracle)")


# ----------------------------------------------------------------------------- command line
def parse_args(argv):
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__), description="Chapter 0246 protocol (MATEAN)")
    ap.add_argument("--quick", action="store_true", help="one seed, reduced steps, all correctness tests")
    ap.add_argument("--seed", type=int, default=BASE_SEED)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--card", action="store_true")
    ap.add_argument("--mutant", type=str, default=None)
    ap.add_argument("--data", type=str, default=None)
    return ap.parse_args(argv)


def main(argv=None):
    global MUTANT
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    if args.mutant is not None and args.mutant not in MUTANTS:
        print(f"usage: python3 {os.path.basename(__file__)} --mutant NAME, with NAME in {sorted(MUTANTS)}")
        return 2
    MUTANT, t0 = args.mutant, time.time()
    cfg = QUICK if args.quick else FULL
    seeds = [args.seed + i for i in range(args.seeds or (1 if args.quick else 5))]
    print(f"chapter 0246 · seeds {seeds} · {'quick' if args.quick else 'full'} protocol" + (f" · mutant {MUTANT}" if MUTANT else ""))
    try:
        rows, gradcheck, mutants = run_correctness(seeds[0])
        per_seed, worlds, params = [], [], None
        for s in seeds:
            metrics, world, params, _ = run_seed(s, cfg)
            per_seed.append(metrics)
            worlds.append(world)
    except NonFinite as exc:
        print(f"non-finite values: {exc}")
        return 4
    ok, detail = test_splits(worlds[0])
    rows.append(("C7", "split_integrity", ok, detail))
    hyps, kos = summarise(per_seed, seeds, args.quick)
    runtime = time.time() - t0
    budget = BUDGET_S["quick" if args.quick else "full"]
    rows.append(("C8", "budget", runtime <= budget, f"{runtime:.1f} s of {budget:.0f} s"))
    failed = [r[0] for r in rows if not r[2]]
    exit_code = 0 if not failed else (3 if failed == ["C8"] else 1)
    report = {"schema_version": "1.0", "chapter": CHAPTER, "file": os.path.basename(__file__),
              "card_revision": MIND_CARD["card_revision"],
              "environment": {"python": sys.version.split()[0], "numpy": np.__version__}, "seeds": seeds,
              "runtime_s": round(runtime, 2), "n_params": params["narek"], "gradcheck": gradcheck,
              "correctness": [{"id": r[0], "name": r[1], "passed": bool(r[2]), "detail": r[3]} for r in rows],
              "mutants": mutants, "hypotheses": hyps, "knockouts": kos, "task_types": TASK_TYPES, "exit_code": exit_code,
              "extra": {"n_params": params, "means": {k: float(np.mean([m[k] for m in per_seed])) for k in sorted(per_seed[0])}}}
    print_report(report)
    if args.json:
        write_report(args.json, report)
    real_data_bridge(args.data, np.random.default_rng(np.random.SeedSequence([CHAPTER, seeds[0], 21])))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

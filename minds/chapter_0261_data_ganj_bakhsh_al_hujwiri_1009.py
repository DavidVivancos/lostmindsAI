#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0261 · Data Ganj Bakhsh al-Hujwiri
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 14, Minds 261-280 Available on Amazon https://www.amazon.com/dp/B0HLYSVY5D
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0261_data_ganj_bakhsh_al_hujwiri_1009 - Data Ganj Bakhsh al-Hujwiri (c.1009-1072)
# END ATTRIBUTION
"""Chapter 0261 · Data Ganj Bakhsh al-Hujwiri (c. 1009 - c. 1072) · the polisher's test.

Thesis
    Before choosing a remedy, find out whether a dimmed faculty is a clouded
    mirror (ghayn: a veil of attributes that polishing removes) or a stone
    (rayn: a veil of essence that no polishing turns into a mirror), and read
    the verdict from how the reflection answers the polish, never from how
    sure the viewer feels.

Evidence and provenance  (provenance: belief - his own surviving treatise)
    D1  Two veils: rayn, the covering of the essence, never removed; ghayn,
        the clouding of the attributes, quickly removed.  Kashf al-Mahjub,
        author's introduction (Nicholson tr. 1911, pp. 4-5).          primary
    D2  After Junayd: rayn is of the abiding, ghayn of the transient; however
        many polishers gather, a stone is not made a mirror, while a rusty
        mirror is brightened by polishing.  Same passage.              primary
    D3  The book is written for polishers of hearts veiled by clouding; it is
        of no use to those veiled in essence.  Same passage.           primary
    D4  The essence-veiled man sees truth and falsehood as the same: the
        essence veil is total.  Same passage.                          primary
    D5  With his master al-Khuttali, after Junayd, sobriety is preferred to
        intoxication.  Nicholson 1911, Translator's Preface.       scholarship
    D6  Annihilation is like fire: it transmutes the quality of things and
        leaves their essence.  Nicholson 1911, Translator's Preface. scholarship

Doctrine -> mechanism -> test
    D1, D2  M1 polisher + M2 mirror read before and after the polish   C6.1  H-SIG  H-NEC
    D2      M3 trust in the polished reading rises with its brightness C6.2  H-SIG
    D3      M3 veil gate: raw / polished / sealed-to-prior readings    C6.3  H-SIG
    D4      M3 seals essence-level dimness as a whole                        H-BLIND
    D5      M3 reads reflection, never confidence (the baseline does)        H-SIG
    D6      M1 changes per-channel quality (gain, offset), never mixing C6.1

Research question  (out-of-distribution detection and shift)
    Without labels, can a deployed classifier tell whether a shift is
    removable by test-time normalisation, from how a learned reflection
    responds to that normalisation, and does routing on that response beat
    routing on predictive confidence?

Closest prior art and the delta
    Test-time batch re-normalisation (Schneider et al. 2020; Nado et al.
    2020); entropy-based reliability filtering (Wang et al. 2021; Niu et al.
    2022, 2023); reconstruction as a test-time signal (Sun et al. 2020;
    Gandelsman et al. 2022); reconstruction-based sensor fault isolation
    (Dunia et al. 1996).  Delta: a fixed, label-free polish is applied as an
    intervention, and the change it makes to reflection brightness decides
    whether the polished reading is trusted, the raw one kept, or the batch
    sealed to the prior.  The size-matched baseline routes on confidence.

Blind spot
    The doctrine has two veils and makes the essence veil total.  Partial
    essence veils ("bent mirrors", a small mixing of channels) dim the
    reflection while the polished reading still discriminates; the sober
    gate seals sight that was still there.

Task  (one episode = one batch of 48 under one veil)
    y ~ U{0..3}, k ~ U{0,1}; z = centre[y, k] + N(0, 0.55^2 I_6)
    x = A z + N(0, 0.25^2 I_12); observed X = mu + sd * x
    training veils: clear; attribute (per-channel gain exp U[-L, L] and an
        offset) mild, moderate, heavy; essence: shuffle (channels decoupled
        across samples), dead (channels frozen)
    shifted split: attribute fog (gain 0.06-0.3) and glare (gain 2.5-6);
        essence rotation (full orthogonal mixing) and half sign-flip
    blind-spot split: bent, cos(t) Z + sin(t) Z Q^T with t in [0.55, 0.85]
    target: y per sample; metric: clarity = log 4 - NLL (nats above prior)

Limits
    Synthetic data; batch-level verdicts (a veil on a few samples is
    invisible to them); a veil that keeps inputs on the mirror's manifold
    passes the test; a linear gate over four features.  A research prototype
    of one faculty of an AGI-oriented architecture.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 2,
    "revision_log": [
        {"revision": 2, "date": "2026-09-18",
         "change": "gradcheck_abs_floor 1e-9 -> 1e-8; reported max relative error restricted to entries with |g| >= 1e-3",
         "reason": ("After the first full run C1 failed at 1.37e-5 on an entry whose gradient was below 1e-4 and whose "
                    "absolute error was under 1e-9, the finite-difference roundoff level at eps 1e-6. The relative "
                    "tolerance (1e-5), eps, hypotheses, metrics and all other thresholds are unchanged; the four "
                    "mutants are still caught.")}],
    "generation": {"template_version": "Code Guidelines for Mind Architectures v1.0 (15 Sep 2026)",
                   "generator": "Claude (Anthropic)", "generator_version": "claude-opus-5",
                   "date": "2026-09-18"},
    "id": 261, "figure": "Data Ganj Bakhsh al-Hujwiri", "born": 1009, "died": 1072,
    "civilization": "Persian (Ghaznavid)", "provenance": "belief",
    "thesis": ("Diagnose a dimmed faculty by how its reflection answers a polish: a clouded mirror "
               "(attribute veil) brightens, a stone (essence veil) does not, and only then choose "
               "between the polished reading, the raw one, or sealing."),
    "evidence": [
        {"id": "D1", "claim": "Two veils: rayn (of essence, never removed) and ghayn (of attributes, removable)",
         "basis": "primary", "source": "Kashf al-Mahjub, author's introduction; tr. Nicholson 1911, pp. 4-5"},
        {"id": "D2", "claim": "After Junayd: a stone cannot be polished into a mirror; a rusty mirror can be",
         "basis": "primary", "source": "Kashf al-Mahjub, author's introduction; tr. Nicholson 1911, p. 5"},
        {"id": "D3", "claim": "The book is for polishers of hearts veiled by clouding; useless to the essence-veiled",
         "basis": "primary", "source": "Kashf al-Mahjub, author's introduction; tr. Nicholson 1911, p. 5"},
        {"id": "D4", "claim": "For the essence-veiled, truth and falsehood are the same (the veil is total)",
         "basis": "primary", "source": "Kashf al-Mahjub, author's introduction; tr. Nicholson 1911, pp. 4-5"},
        {"id": "D5", "claim": "Sobriety preferred to intoxication, with al-Khuttali, after Junayd",
         "basis": "scholarship", "source": "R. A. Nicholson, Translator's Preface to The Kashf al-Mahjub (1911)"},
        {"id": "D6", "claim": "Annihilation likened to fire: changes quality, leaves essence",
         "basis": "scholarship", "source": "R. A. Nicholson, Translator's Preface to The Kashf al-Mahjub (1911)"},
    ],
    "research_question": {"category": "out-of-distribution detection and shift",
                          "question": ("Can a label-free intervention (test-time normalisation) plus a learned "
                                       "reflection tell removable from irremovable shift, and does routing on the "
                                       "reflection's response beat routing on predictive confidence?")},
    "mechanism": {
        "name": "polisher's test (reflection contrast before and after test-time normalisation)",
        "family": "test-time adaptation and shift diagnosis",
        "signature_modules": ["mirror", "veil_signal", "veil_gate"],
        "closest_prior_art": [
            "Schneider et al. 2020, covariate shift adaptation by test-time batch statistics (NeurIPS)",
            "Nado et al. 2020, prediction-time batch normalization (arXiv:2006.10963)",
            "Wang et al. 2021, Tent: test-time entropy minimization (ICLR)",
            "Niu et al. 2022, EATA: entropy-filtered test-time adaptation (ICML); Niu et al. 2023, SAR (ICLR)",
            "Sun et al. 2020, test-time training with self-supervision (ICML); Gandelsman et al. 2022, TTT-MAE (NeurIPS)",
            "Dunia, Qin, Edgar and McAvoy 1996, identification of faulty sensors using PCA (AIChE Journal)"],
        "overlap": "Medium",
        "prior_art_queries": [
            "test-time adaptation batch normalization statistics covariate shift",
            "entropy based reliable sample selection test-time adaptation",
            "test-time training masked autoencoder reconstruction",
            "detecting dataset shift two-sample tests failing loudly",
            "distinguishing correctable from uncorrectable distribution shift without labels",
            "faulty sensor identification reconstruction principal components"],
        "contribution_type": "mechanism",
        "delta": ("The polish is used as an intervention and the before/after change in reconstruction "
                  "brightness routes each batch to the polished reading, the raw reading or the prior; "
                  "confidence-gated test-time normalisation is the size-matched baseline."),
        "rival_note": ("The confidence-gated baseline is also read, in the chapter, as the intoxicated side "
                       "of the sobriety dispute; the rival school has no chapter in the corpus, so no H-RIVAL."),
    },
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1 polisher + M2 mirror", "property_test": "C6.1", "hypothesis": "H-NEC"},
        {"doctrine": "D2", "mechanism": "M3 monotone trust in post-polish brightness", "property_test": "C6.2",
         "hypothesis": "H-SIG"},
        {"doctrine": "D3", "mechanism": "M3 raw / polished / sealed readings", "property_test": "C6.3",
         "hypothesis": "H-SIG"},
        {"doctrine": "D4", "mechanism": "M3 seals essence-level dimness as a whole", "property_test": "",
         "hypothesis": "H-BLIND"},
        {"doctrine": "D5", "mechanism": "M3 reads reflection, not confidence", "property_test": "",
         "hypothesis": "H-SIG"},
        {"doctrine": "D6", "mechanism": "M1 per-channel quality change only", "property_test": "C6.1",
         "hypothesis": "H-SIG"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": ("Routing on the reflection's response to polishing beats routing on "
                                      "confidence on veils outside the training range"),
         "metric": "clarity", "split": "shifted", "comparison": "model - baseline", "direction": "greater",
         "mesi": 0.05, "seeds": 5},
        {"id": "H-NEC", "statement": ("Replacing the reflection-contrast features by their training mean costs "
                                      "more than replacing the matched surface statistics"),
         "metric": "clarity", "split": "shifted", "comparison": "signature_knockout - matched_knockout",
         "direction": "less", "mesi": 0.05, "seeds": 5},
        {"id": "H-BLIND", "statement": ("On partial essence veils the sober gate seals readings that still "
                                        "discriminate and falls behind the confidence gate"),
         "condition": "bent: partial rotation, t in [0.55, 0.85]",
         "grounding": ("the essence-veiled man sees truth and falsehood as the same: the doctrine admits no "
                       "partial essence veil"),
         "metric": "clarity", "comparison": "model - baseline", "direction": "less", "mesi": 0.05, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.3, "margin_over_trivial": 0.3, "shuffled_band_clarity": 0.1,
                   "shuffled_band_accuracy": 0.35, "gradcheck_rel_tol": 1e-5, "gradcheck_abs_floor": 1e-8,
                   "property_tol": 1e-6, "clip_global_norm": 5.0},
    "probe_predictions": [
        {"probe": "P8", "expected": "above baseline"}, {"probe": "P9", "expected": "above baseline"},
        {"probe": "P6", "expected": "equal to baseline"}, {"probe": "P10", "expected": "equal to baseline"},
        {"probe": "P5", "expected": "below baseline"}],
    "dialectic_links": [],
    "corpus_neighbors": [
        {"chapter": 260, "similarity": None, "difference": "Nasir Khusraw values learnability under a seal; here the verdict is whether a shift is removable by a label-free polish"},
        {"chapter": 257, "similarity": None, "difference": "Ibn Sina's estimative and intuitive faculties; no intuition module here, only an intervention and its response"},
        {"chapter": 252, "similarity": None, "difference": "al-Biruni counts independent witnesses (source reliability); here there are no sources, only one faculty tested against itself"},
        {"chapter": 247, "similarity": None, "difference": "Ibn al-Haytham's error taxonomy uses an external reference; here the reference is the model's own reflection before and after a polish"},
        {"chapter": 243, "similarity": None, "difference": "al-Zahrawi confines weight edits; here no weights change at test time"},
        {"chapter": 246, "similarity": None, "difference": "Narekatsi's sealed confession concerns uncommitted faults; here sealing is a fallback to the prior under irremovable shift"},
        {"chapter": 277, "similarity": None, "difference": "al-Ghazali breaks a regress with an external anchor; here the anchor is internal and interventional"},
        {"chapter": 329, "similarity": None, "difference": "Rumi's subtractive parameters sit in the pruning cluster; nothing is pruned here"},
        {"chapter": 492, "similarity": None, "difference": "Teresa reads origin from delayed residue and distrusts self-report; here the evidence is the immediate response to a polish"},
        {"chapter": 316, "similarity": None, "difference": "Ibn 'Arabi models receptivity; here receptivity is tested by intervention, not modelled"},
        {"chapter": 507, "similarity": 0.034, "difference": "Philip II sample file, the only other chapter code available in this session (9-shingle Jaccard measured); similarity to 0241-0260 could not be computed here"}],
    "barometer": {
        "cognitive_processing": ["held-out clarity on in-distribution veils"],
        "embodied_cognition": ["sensor-channel drift versus failure (synthetic channels)"],
        "world_modeling": ["mirror as a learned model of clean inputs"],
        "consciousness": ["self-monitoring: reflection before and after polish; calibration under shift (NLL)"],
        "language_understanding": [], "emotional_intelligence": [], "creativity": [],
        "autonomy": ["choosing a remedy (raw, polish, seal) without labels"]},
    "task_types": ["vector_classification"],
    "applications": [
        {"use": "Triage of sensor drift (recalibrate) versus sensor failure (quarantine) in chemical sensor arrays",
         "sector": "industrial monitoring", "dataset": "UCI Gas Sensor Array Drift Dataset (Vergara et al. 2012)"},
        {"use": "Deciding whether a wearable activity model needs re-normalisation or re-training after the device is re-oriented",
         "sector": "mobile health and wearables", "dataset": "UCI Human Activity Recognition Using Smartphones (Anguita et al. 2013)"},
        {"use": "Mapping which image corruptions test-time normalisation can and cannot remove",
         "sector": "computer-vision robustness", "dataset": "CIFAR-10-C (Hendrycks and Dietterich 2019)"},
        {"use": "Research flagging of ECG lead reversal (essence) versus gain and baseline drift (attribute)",
         "sector": "clinical decision support, research only", "dataset": "PTB-XL (Wagner et al. 2020, PhysioNet)"}],
    "safety_notes": ("No claim of replicating a real mind and no sentences presented as the figure's words. "
                     "Medical uses are research decision support only. Synthetic data only."),
}

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

TASK_TYPES = ("vector_classification",)
C_CLASSES, D_OBS, D_LAT, BATCH, H1, H2 = 4, 12, 6, 48, 24, 8
EPS_POL, EPS_B, LAMBDA_C, LAMBDA_R, LR = 1e-9, 1e-3, 1.0, 1.0, 0.01
CLIP = MIND_CARD["thresholds"]["clip_global_norm"]
STEPS = {"full": 2000, "quick": 700}
N_EVAL = {"full": 30, "quick": 10}
BUDGET_S = {"full": 180.0, "quick": 20.0}
TRAIN_VEILS = ("clear", "mild", "moderate", "heavy", "shuffle", "dead")
TRAIN_PROBS = np.array([0.20, 0.15, 0.20, 0.15, 0.15, 0.15])
SHIFT_VEILS = ("fog", "glare", "rotate", "flip")
ESSENCE = ("shuffle", "dead", "rotate", "flip", "bent")
ATTRIBUTE = {"mild": (0.3, 0.3), "moderate": (0.8, 1.2), "heavy": (1.6, 3.5)}
KO_MODES = {"encoder": ("zero", "mean"), "head": ("zero", "mean"), "mirror": ("identity", "zero", "mean"),
            "polisher": ("identity",), "veil_signal": ("zero", "mean"), "surface_statistics": ("zero", "mean"),
            "veil_gate": ("identity", "zero", "mean")}
KO_PLAN = (("veil_signal", "mean"), ("surface_statistics", "mean"), ("veil_gate", "identity"),
           ("mirror", "identity"), ("polisher", "identity"))


class BudgetExceeded(RuntimeError):
    pass


# BEGIN STANDARD UTILITIES v1.0
def softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def logsumexp(z, axis=-1):
    m = np.max(z, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(z - m), axis=axis))


def softplus(x):
    return np.logaddexp(0.0, x)


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


class Adam:
    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1.0 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1.0 - self.b2) * grads[k] ** 2
            mh = self.m[k] / (1.0 - self.b1 ** self.t)
            vh = self.v[k] / (1.0 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


def clip_global_norm(grads, max_norm):
    norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    if norm > max_norm:
        for k in grads:
            grads[k] *= max_norm / norm
    return norm


def grad_check(loss_fn, params, grads, rng, n_random=20, eps=1e-6, tol=1e-5, abs_floor=1e-9):
    """Central differences on n_random entries plus the largest-gradient entry of every tensor.
    An entry passes if its relative error is at most tol or its absolute error is at most abs_floor
    (finite-difference roundoff at eps is about machine-eps * |loss| / eps, near 1e-9).  The reported
    maximum relative error covers entries whose gradient is at least 1e-3 in magnitude."""
    worst, failures = 0.0, 0
    for key, tensor in params.items():
        flat, gflat = tensor.reshape(-1), grads[key].reshape(-1)
        picks = set(rng.choice(flat.size, min(n_random, flat.size), replace=False).tolist())
        picks.add(int(np.argmax(np.abs(gflat))))
        for i in picks:
            old = flat[i]
            flat[i] = old + eps
            up = loss_fn()
            flat[i] = old - eps
            down = loss_fn()
            flat[i] = old
            num, ana = (up - down) / (2.0 * eps), gflat[i]
            diff, scale = abs(num - ana), max(abs(num), abs(ana))
            rel = diff / scale if scale > 0 else 0.0
            if scale >= 1e-3:
                worst = max(worst, rel)
            failures += int(rel > tol and diff > abs_floor)
    return {"tensors_checked": len(params), "tensors_total": len(params), "max_rel_error": worst,
            "failures": failures, "passed": failures == 0}


def bootstrap_ci(diffs, rng, n_boot=2000, alpha=0.05):
    d = np.asarray(diffs, dtype=float)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return float(d.mean()), [float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))]


def verdict(mean, ci, direction, mesi):
    lo, hi = ci
    if direction == "greater":
        if lo > 0 and mean >= mesi:
            return "supported"
        return "contradicted" if hi < 0 else "inconclusive"
    if hi < 0 and mean <= -mesi:
        return "supported"
    return "contradicted" if lo > 0 else "inconclusive"


def write_report(report, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
# END STANDARD UTILITIES


# ----------------------------------------------------------------------------- data and tasks
def make_world(rng):
    centres = rng.normal(0.0, 1.6, size=(C_CLASSES, 2, D_LAT))
    A = rng.normal(size=(D_OBS, D_LAT))
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    return {"centres": centres, "A": A, "mu": rng.normal(0.0, 2.0, D_OBS),
            "sd": np.exp(rng.uniform(-0.5, 0.8, D_OBS))}


def sample(world, n, rng):
    if "rows" in world:                              # array-backed world (probe battery, --data)
        i = rng.integers(0, len(world["rows"]), n)
        return world["rows"][i], world["labels"][i]
    y, k = rng.integers(0, C_CLASSES, n), rng.integers(0, 2, n)
    z = world["centres"][y, k] + rng.normal(0.0, 0.55, (n, D_LAT))
    x = z @ world["A"].T + rng.normal(0.0, 0.25, (n, D_OBS))
    return world["mu"] + world["sd"] * x, y


def _attribute_draw(kind, d, rng):
    if kind in ATTRIBUTE:
        L, O = ATTRIBUTE[kind]
        return np.exp(rng.uniform(-L, L, d)), rng.uniform(-O, O, d)
    lo, hi, O = (0.06, 0.3, 3.0) if kind == "fog" else (2.5, 6.0, 3.5)
    return np.exp(rng.uniform(np.log(lo), np.log(hi), d)), rng.uniform(-O, O, d)


def veil(X, kind, world, rng):
    """Attribute veils change each channel's quality (gain, offset); essence veils break or mix channels."""
    mu, sd, d = world["mu"], world["sd"], X.shape[1]
    if kind == "clear":
        return X.copy()
    if kind in ATTRIBUTE or kind in ("fog", "glare"):
        g, o = _attribute_draw(kind, d, rng)
        return mu + g * (X - mu) + o * sd
    if kind == "shuffle":
        return np.stack([X[rng.permutation(len(X)), c] for c in range(d)], axis=1)
    if kind == "dead":
        return mu + rng.uniform(-1.5, 1.5, d) * sd + 0.03 * sd * rng.normal(size=X.shape)
    if kind == "flip":
        Xv, idx = X.copy(), rng.choice(d, d // 2, replace=False)
        Xv[:, idx] = 2.0 * mu[idx] - X[:, idx]
        return Xv
    theta = np.pi / 2 if kind == "rotate" else rng.uniform(0.55, 0.85)
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    Z = (X - mu) / sd
    return mu + sd * (np.cos(theta) * Z + np.sin(theta) * (Z @ Q.T))


def make_stream(world, rng, permute_labels=False):
    """Endless training episodes.  The learn flag is 0 on stones: no mirror is taught to reflect a stone."""
    def nxt():
        kind = TRAIN_VEILS[rng.choice(len(TRAIN_VEILS), p=TRAIN_PROBS)]
        Xc, y = sample(world, BATCH, rng)
        if permute_labels:
            y = rng.permutation(y)
        return {"X": veil(Xc, kind, world, rng), "y": y, "w": 0.0 if kind in ESSENCE else 1.0, "kind": kind}
    return nxt


def episodes(world, kinds, n_each, rng):
    out = []
    for kind in kinds:
        for _ in range(n_each):
            Xc, y = sample(world, BATCH, rng)
            out.append((kind, veil(Xc, kind, world, rng), y))
    return out


# ----------------------------------------------------------------------------- model
def build_model(in_dim, out_dim, task_type="vector_classification", rng=None, gate="sober", **cfg):
    if task_type not in TASK_TYPES:
        raise ValueError(f"unsupported task type {task_type}")
    rng = rng if rng is not None else np.random.default_rng(0)
    h1, h2 = cfg.get("h1", H1), cfg.get("h2", H2)

    def dense(a, b):
        return rng.normal(0.0, 1.0 / np.sqrt(a), (a, b))
    params = {"W1": dense(in_dim, h1), "b1": np.zeros(h1), "W2": dense(h1, h2), "b2": np.zeros(h2),
              "Wc": dense(h2, out_dim), "bc": np.zeros(out_dim),
              "Wd1": dense(h2, h1), "bd1": np.zeros(h1), "Wd2": dense(h1, in_dim), "bd2": np.zeros(in_dim),
              "G": 0.1 * rng.normal(size=(4, 3)), "g0": np.zeros(3), "kappa": np.zeros(1)}
    return {"params": params, "gate": gate, "task_type": task_type, "ko": {}, "means": {},
            "mu0": np.zeros(in_dim), "sd0": np.ones(in_dim)}


def _reading(model, U):
    """One reading of a batch: understanding (encoder, head) and reflection (mirror) of the same input."""
    p, ko, mean = model["params"], model["ko"], model["means"]
    h1 = np.tanh(U @ p["W1"] + p["b1"])
    h2 = np.tanh(h1 @ p["W2"] + p["b2"])
    if ko.get("encoder") == "zero":
        h2 = np.zeros_like(h2)
    elif ko.get("encoder") == "mean":
        h2 = np.broadcast_to(mean["h2"], h2.shape).copy()
    P = softmax(h2 @ p["Wc"] + p["bc"])
    if ko.get("head") == "zero":
        P = np.full_like(P, 1.0 / P.shape[1])
    elif ko.get("head") == "mean":
        P = np.broadcast_to(mean["P"], P.shape).copy()
    m1 = np.tanh(h2 @ p["Wd1"] + p["bd1"])
    R = m1 @ p["Wd2"] + p["bd2"]
    if ko.get("mirror") == "identity":
        R = U.copy()
    elif ko.get("mirror") == "zero":
        R = np.zeros_like(U)
    elif ko.get("mirror") == "mean":
        R = np.broadcast_to(mean["R"], U.shape).copy()
    v = float(np.mean((R - U) ** 2))
    return {"U": U, "h1": h1, "h2": h2, "P": P, "m1": m1, "R": R, "v": v, "beta": -math.log(v + EPS_B)}


def _entropy(P):
    return float(-np.mean(np.sum(P * np.log(np.maximum(P, 1e-300)), axis=1)))


def _features(model, rr, rp, Uraw):
    """Sober gate: brightness before and after the polish.  Confidence gate: negative entropy before
    and after.  Both also see two surface statistics of the raw batch."""
    if model["gate"] == "sober":
        signal = [rr["beta"], rp["beta"]]
    else:
        signal = [-_entropy(rr["P"]), -_entropy(rp["P"])]
    surface = [math.log1p(float(np.mean(np.abs(Uraw.mean(0))))),
               math.log1p(float(np.mean(np.abs(np.log(Uraw.std(0) + 1e-6)))))]
    f = np.array(signal + surface)
    for name, sl in (("veil_signal", slice(0, 2)), ("surface_statistics", slice(2, 4))):
        mode = model["ko"].get(name)
        if mode == "zero":
            f[sl] = 0.0
        elif mode == "mean":
            f[sl] = model["means"]["f"][sl]
    return f


def _gate_matrix(p):
    """Trust in the polished reading must rise with feature 1 (post-polish brightness or confidence):
    its coefficient is a smooth maximum of the other two plus a positive margin."""
    G = p["G"].copy()
    G[1, 1] = G[1, 2] + softplus(G[1, 0] - G[1, 2]) + softplus(p["kappa"][0])
    return G


def polish(X):
    """Parameter-free per-channel re-standardisation over the batch: removes gain and offset veils."""
    return (X - X.mean(0)) / (X.std(0) + EPS_POL)


def forward(model, X):
    p, ko = model["params"], model["ko"]
    Uraw = (X - model["mu0"]) / model["sd0"]
    Upol = Uraw if ko.get("polisher") == "identity" else polish(X)
    rr, rp = _reading(model, Uraw), _reading(model, Upol)
    f, G = _features(model, rr, rp, Uraw), _gate_matrix(p)
    pi = softmax(f @ G + p["g0"])
    mode = ko.get("veil_gate")
    if mode == "identity":
        pi = np.array([0.0, 1.0, 0.0])
    elif mode == "zero":
        pi = np.full(3, 1.0 / 3.0)
    elif mode == "mean":
        pi = model["means"]["pi"].copy()
    P = pi[0] * rr["P"] + pi[1] * rp["P"] + pi[2] / rr["P"].shape[1]
    return {"rr": rr, "rp": rp, "f": f, "G": G, "pi": pi, "P": P}


def predict(model, X):
    return forward(model, X)["P"].argmax(axis=1)


def hidden_states(model, X):
    o = forward(model, X)
    return {"latent_raw": o["rr"]["h2"], "latent_polished": o["rp"]["h2"],
            "brightness_raw_polished": np.array([o["rr"]["beta"], o["rp"]["beta"]]),
            "gate_raw_polished_sealed": o["pi"], "gate_features": o["f"]}


def modules(model):
    sober = model["gate"] == "sober"
    return {
        "encoder": {"params": ["W1", "b1", "W2", "b2"], "role": "two-layer tanh encoder to an 8-unit latent",
                    "signature": False},
        "head": {"params": ["Wc", "bc"], "role": "softmax classification head", "signature": False},
        "mirror": {"params": ["Wd1", "bd1", "Wd2", "bd2"],
                   "role": "latent-to-input reconstruction decoder (squared error gives brightness)",
                   "signature": sober},
        "polisher": {"params": [], "role": "per-batch per-channel standardisation (test-time normalisation)",
                     "signature": False},
        "veil_signal": {"params": [], "role": ("log reconstruction error before and after standardisation"
                                               if sober else "negative entropy before and after standardisation"),
                        "signature": sober},
        "surface_statistics": {"params": [], "role": "batch mean offset and log-dispersion of raw channels",
                               "signature": False},
        "veil_gate": {"params": ["G", "g0", "kappa"],
                      "role": "three-way softmax router over raw, standardised and prior readings, monotone in feature 1",
                      "signature": sober}}


def knockout(model, name, mode):
    if mode not in KO_MODES.get(name, ()):
        raise ValueError(f"module {name} has no knockout mode {mode}")
    ko_model = dict(model)
    ko_model["params"] = {k: v.copy() for k, v in model["params"].items()}
    ko_model["ko"] = dict(model["ko"], **{name: mode})
    return ko_model


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


# ----------------------------------------------------------------------------- baselines
# The size-matched baseline is build_model(..., gate="confidence"): the same encoder, head, mirror,
# polisher and three-way gate with identical parameter count, routing on negative predictive entropy
# before and after the polish (the reliability signal of entropy-filtered test-time adaptation).
# The trivial baseline is the class prior, whose clarity is 0 nats by definition.


# ----------------------------------------------------------------------------- registries
MUTANTS = {"sign_flip": "update direction reversed", "zero_lr": "learning rate set to zero",
           "zero_grad_mirror": "gradient of Wd2 zeroed", "zero_grad_gate": "gradient of G zeroed"}
ZERO_GRAD = {"zero_grad_mirror": "Wd2", "zero_grad_gate": "G"}
_ACTIVE = {"mutant": None}


# ----------------------------------------------------------------------------- training
def _reading_backward(p, c, dP, dv, g):
    dz = c["P"] * (dP - np.sum(dP * c["P"], axis=1, keepdims=True))
    g["Wc"] += c["h2"].T @ dz
    g["bc"] += dz.sum(0)
    dh2 = dz @ p["Wc"].T
    dR = dv * 2.0 * (c["R"] - c["U"]) / c["U"].size
    g["Wd2"] += c["m1"].T @ dR
    g["bd2"] += dR.sum(0)
    dq = (dR @ p["Wd2"].T) * (1.0 - c["m1"] ** 2)
    g["Wd1"] += c["h2"].T @ dq
    g["bd1"] += dq.sum(0)
    dh2 = dh2 + dq @ p["Wd1"].T
    da2 = dh2 * (1.0 - c["h2"] ** 2)
    g["W2"] += c["h1"].T @ da2
    g["b2"] += da2.sum(0)
    da1 = (da2 @ p["W2"].T) * (1.0 - c["h1"] ** 2)
    g["W1"] += c["U"].T @ da1
    g["b1"] += da1.sum(0)


def _gate_backward(p, G, f, da, g):
    g["g0"] += da
    dG = np.outer(f, da)
    t = p["G"][1, 0] - p["G"][1, 2]
    d11, dG[1, 1] = dG[1, 1], 0.0
    dG[1, 0] += d11 * sigmoid(t)
    dG[1, 2] += d11 * (1.0 - sigmoid(t))
    g["G"] += dG
    g["kappa"][0] += d11 * sigmoid(p["kappa"][0])
    return G @ da


def loss_and_grads(model, batch, need_grads=True):
    """Mixture NLL of the gated prediction; on clear and attribute-veiled batches (w = 1) also the
    polished cross-entropy and the mirror's reconstruction error.  Every path is differentiated."""
    assert not model["ko"], "knocked-out copies are evaluated, never trained"
    p, X, y, w = model["params"], batch["X"], batch["y"], float(batch["w"])
    o = forward(model, X)
    rr, rp, pi = o["rr"], o["rp"], o["pi"]
    B, idx = len(y), np.arange(len(y))
    pb, ppol = o["P"][idx, y], rp["P"][idx, y]
    loss = -np.mean(np.log(pb)) + w * (-LAMBDA_C * np.mean(np.log(ppol)) + LAMBDA_R * rp["v"])
    if not need_grads:
        return float(loss), None
    g = {k: np.zeros_like(v) for k, v in p.items()}
    dpi = -np.array([np.mean(rr["P"][idx, y] / pb), np.mean(ppol / pb), np.mean(1.0 / (rr["P"].shape[1] * pb))])
    df = _gate_backward(p, o["G"], o["f"], pi * (dpi - pi @ dpi), g)
    dP = {"rr": np.zeros_like(rr["P"]), "rp": np.zeros_like(rp["P"])}
    dP["rr"][idx, y] = -pi[0] / (B * pb)
    dP["rp"][idx, y] = -pi[1] / (B * pb) - w * LAMBDA_C / (B * ppol)
    dv = {"rr": 0.0, "rp": w * LAMBDA_R}
    for j, key in enumerate(("rr", "rp")):
        r = o[key]
        if model["gate"] == "sober":
            dv[key] += -df[j] / (r["v"] + EPS_B)
        else:
            dP[key] += df[j] * (np.log(np.maximum(r["P"], 1e-300)) + 1.0) / B
        _reading_backward(p, r, dP[key], dv[key], g)
    if _ACTIVE["mutant"] in ZERO_GRAD:
        g[ZERO_GRAD[_ACTIVE["mutant"]]][...] = 0.0
    return float(loss), g


def _store_means(model, nxt, n=40):
    acc = {"h2": [], "P": [], "R": [], "f": [], "pi": []}
    for _ in range(n):
        o = forward(model, nxt()["X"])
        for r in (o["rr"], o["rp"]):
            acc["h2"].append(r["h2"].mean(0))
            acc["P"].append(r["P"].mean(0))
            acc["R"].append(r["R"].mean(0))
        acc["f"].append(o["f"])
        acc["pi"].append(o["pi"])
    model["means"] = {k: np.mean(v, axis=0) for k, v in acc.items()}


def fit(model, data, budget, rng):
    """data: {'world', 'reference' (clean rows for the raw restore), optional 'permute_labels'}."""
    model["mu0"] = data["reference"].mean(0)
    model["sd0"] = data["reference"].std(0) + 1e-9
    nxt = make_stream(data["world"], rng, data.get("permute_labels", False))
    opt = Adam(model["params"], 0.0 if _ACTIVE["mutant"] == "zero_lr" else LR)
    losses = np.empty(budget)
    for t in range(budget):
        loss, g = loss_and_grads(model, nxt())
        if not math.isfinite(loss):
            raise FloatingPointError("non-finite training loss")
        if _ACTIVE["mutant"] == "sign_flip":
            g = {k: -v for k, v in g.items()}
        clip_global_norm(g, CLIP)
        opt.step(model["params"], g)
        losses[t] = loss
    _store_means(model, nxt)
    return losses


def train_pair(seed, steps, n_eval):
    """Sober model and confidence baseline from identical initial weights on an identical stream."""
    r_world, r_init, r_eval = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
    world = make_world(r_world)
    ref, _ = sample(world, 4000, r_world)
    state, models, losses = r_init.bit_generator.state, {}, {}
    for gate in ("sober", "confidence"):
        r_init.bit_generator.state = state
        models[gate] = build_model(D_OBS, C_CLASSES, "vector_classification", r_init, gate=gate)
        stream = np.random.default_rng(np.random.SeedSequence([seed, 7]))
        losses[gate] = fit(models[gate], {"world": world, "reference": ref}, steps, stream)
    splits = {"heldout": episodes(world, TRAIN_VEILS, n_eval, r_eval),
              "shifted": episodes(world, SHIFT_VEILS, n_eval, r_eval),
              "blind": episodes(world, ("bent",), n_eval, r_eval)}
    return {"seed": seed, "world": world, "ref": ref, "models": models, "losses": losses, "splits": splits,
            "steps": steps}


def clarity_table(model, eps):
    per, total = {}, []
    for kind, X, y in eps:
        o = forward(model, X)
        c = math.log(o["P"].shape[1]) + float(np.mean(np.log(o["P"][np.arange(len(y)), y])))
        per.setdefault(kind, []).append([c, float(np.mean(o["rp"]["P"].argmax(1) == y)),
                                         float(np.mean(o["P"].argmax(1) == y))] + list(o["pi"]))
        total.append(c)
    return float(np.mean(total)), {k: np.mean(v, axis=0) for k, v in per.items()}


# ----------------------------------------------------------------------------- correctness tests
def _fixed_batch(world, kind, rng, w):
    Xc, y = sample(world, BATCH, rng)
    return {"X": veil(Xc, kind, world, rng), "y": y, "w": w}


def _gradcheck_model(model, world, rng):
    worst, ok = 0.0, True
    for kind, w in (("heavy", 1.0), ("shuffle", 0.0)):
        batch = _fixed_batch(world, kind, rng, w)
        _, g = loss_and_grads(model, batch)
        r = grad_check(lambda: loss_and_grads(model, batch, need_grads=False)[0], model["params"], g, rng,
                       tol=MIND_CARD["thresholds"]["gradcheck_rel_tol"],
                       abs_floor=MIND_CARD["thresholds"]["gradcheck_abs_floor"])
        worst, ok = max(worst, r["max_rel_error"]), ok and r["passed"]
    return worst, ok


def test_c1(pair):
    rng = np.random.default_rng(np.random.SeedSequence([pair["seed"], 11]))
    worst, ok, n = 0.0, True, 0
    for gate in ("sober", "confidence"):
        m = build_model(D_OBS, C_CLASSES, "vector_classification", rng, gate=gate)
        m["mu0"], m["sd0"] = pair["ref"].mean(0), pair["ref"].std(0) + 1e-9
        for stage in ("init", "trained"):
            if stage == "trained":
                fit(m, {"world": pair["world"], "reference": pair["ref"]}, 60, rng)
            w_, ok_ = _gradcheck_model(m, pair["world"], rng)
            worst, ok = max(worst, w_), ok and ok_
        n = len(m["params"])
    return {"tensors_checked": n, "tensors_total": n, "max_rel_error": worst,
            "checked_at": ["init", "after_training_steps"], "passed": bool(ok)}


def test_c2(pair):
    runs = []
    for _ in range(2):
        m = build_model(D_OBS, C_CLASSES, "vector_classification", np.random.default_rng(pair["seed"] + 101))
        L = fit(m, {"world": pair["world"], "reference": pair["ref"]}, 80, np.random.default_rng(pair["seed"] + 102))
        X = sample(pair["world"], BATCH, np.random.default_rng(pair["seed"] + 103))[0]
        runs.append((L, forward(m, X)["P"], m))
    same = np.array_equal(runs[0][0], runs[1][0]) and np.array_equal(runs[0][1], runs[1][1])
    finite = all(np.all(np.isfinite(v)) for v in runs[0][2]["params"].values()) and np.all(np.isfinite(runs[0][1]))
    return bool(same and finite), f"identical losses and outputs: {same}; all finite: {finite}"


def c3_criteria(model, losses, heldout):
    k = min(100, len(losses) // 4)
    drop = 1.0 - float(np.mean(losses[-k:])) / float(np.mean(losses[:k]))
    clarity = clarity_table(model, heldout)[0]
    th = MIND_CARD["thresholds"]
    return {"passed": bool(drop >= th["loss_drop_fraction"] and clarity >= th["margin_over_trivial"]),
            "drop": drop, "clarity": clarity}


def test_c4(pair):
    m = build_model(D_OBS, C_CLASSES, "vector_classification", np.random.default_rng(pair["seed"] + 201))
    fit(m, {"world": pair["world"], "reference": pair["ref"], "permute_labels": True}, pair["steps"],
        np.random.default_rng(pair["seed"] + 202))
    clarity, per = clarity_table(m, pair["splits"]["heldout"])
    acc = float(np.mean([v[2] for v in per.values()]))
    th = MIND_CARD["thresholds"]
    ok = clarity <= th["shuffled_band_clarity"] and acc <= th["shuffled_band_accuracy"]
    return bool(ok), f"held-out clarity {clarity:+.3f} nats, accuracy {acc:.3f} (trivial: 0 nats, 0.25)"


def test_c5(pair, steps):
    detected = {}
    for name in MUTANTS:
        _ACTIVE["mutant"] = name
        try:
            if name in ZERO_GRAD:
                m = build_model(D_OBS, C_CLASSES, "vector_classification", np.random.default_rng(pair["seed"] + 301))
                m["mu0"], m["sd0"] = pair["ref"].mean(0), pair["ref"].std(0) + 1e-9
                detected[name] = not _gradcheck_model(m, pair["world"], np.random.default_rng(pair["seed"] + 302))[1]
            else:
                m = build_model(D_OBS, C_CLASSES, "vector_classification", np.random.default_rng(pair["seed"] + 303))
                try:
                    L = fit(m, {"world": pair["world"], "reference": pair["ref"]}, steps,
                            np.random.default_rng(pair["seed"] + 304))
                    detected[name] = not c3_criteria(m, L, pair["splits"]["heldout"])["passed"]
                except FloatingPointError:
                    detected[name] = True
        finally:
            _ACTIVE["mutant"] = None
    return detected


def test_c6(pair, rng):
    """C6.1 polish invariance; C6.2 monotone trust; C6.3 seal floor (definition check only)."""
    m = build_model(D_OBS, C_CLASSES, "vector_classification", rng)
    m["mu0"], m["sd0"] = rng.normal(0.0, 1.0, D_OBS), np.exp(rng.normal(0.0, 0.5, D_OBS))
    inv, inv_b, neg, neg_b = 0.0, 0.0, 0.0, 0.0
    for _ in range(200):
        X = rng.normal(size=(BATCH, D_OBS)) * np.exp(rng.normal(0.0, 1.0, D_OBS)) + rng.normal(0.0, 3.0, D_OBS)
        gain, off = np.exp(rng.uniform(-3.0, 3.0, D_OBS)), rng.uniform(-10.0, 10.0, D_OBS)
        a, b = forward(m, X), forward(m, gain * X + off)
        inv = max(inv, float(np.max(np.abs(a["rp"]["U"] - b["rp"]["U"]))))
        inv_b = max(inv_b, abs(a["rp"]["beta"] - b["rp"]["beta"]))
        neg = max(neg, float(np.max(np.abs(a["rr"]["U"] - b["rr"]["U"]))))
        neg_b = max(neg_b, abs(a["rr"]["beta"] - b["rr"]["beta"]))
    tol = MIND_CARD["thresholds"]["property_tol"]
    c61 = inv < tol and inv_b < tol and neg > 1e-2 and neg_b > 1e-3
    viol, viol_neg = 0.0, 0.0
    for _ in range(3000):
        P = {"G": rng.normal(0.0, 2.0, (4, 3)), "g0": rng.normal(0.0, 2.0, 3), "kappa": rng.normal(0.0, 2.0, 1)}
        f = rng.normal(0.0, 3.0, 4)
        f2 = f.copy()
        f2[1] += abs(rng.normal()) + 1e-3
        G = _gate_matrix(P)
        viol = max(viol, softmax(f @ G + P["g0"])[1] - softmax(f2 @ G + P["g0"])[1])
        viol_neg = max(viol_neg, softmax(f @ P["G"] + P["g0"])[1] - softmax(f2 @ P["G"] + P["g0"])[1])
    c62 = viol <= 1e-12 and viol_neg > 1e-3
    floor = min(float(np.min(forward(pair["models"]["sober"], X)["P"] * C_CLASSES
                             - forward(pair["models"]["sober"], X)["pi"][2])) for _, X, _ in pair["splits"]["shifted"][:20])
    return [("C6.1", "polish_invariance", bool(c61),
             f"max |dU_pol| {inv:.1e}, |dbeta_pol| {inv_b:.1e}; negative control (fixed-statistics restore) "
             f"|dU| {neg:.1e}, |dbeta| {neg_b:.1e}"),
            ("C6.2", "monotone_trust", bool(c62),
             f"max violation {viol:.1e} over 3000 random gates; unconstrained negative control {viol_neg:.2f}"),
            ("C6.3", "seal_floor_definition_check", bool(floor >= -1e-12),
             f"min C*p - pi_seal = {floor:.1e} (holds by construction; not evidence)")]


def _row_hash(row):
    return hashlib.blake2b(np.ascontiguousarray(row).tobytes(), digest_size=16).hexdigest()


def test_c7(pair):
    nxt = make_stream(pair["world"], np.random.default_rng(np.random.SeedSequence([pair["seed"], 7])))
    seen, first = set(), None
    for _ in range(pair["steps"] + 40):
        X = nxt()["X"]
        first = X[0] if first is None else first
        seen.update(_row_hash(r) for r in X)
    evals = [X for split in pair["splits"].values() for _, X, _ in split]
    clashes = sum(_row_hash(r) in seen for X in evals for r in X)
    planted = np.vstack([evals[0][1:], first[None, :]])
    control = sum(_row_hash(r) in seen for r in planted)
    rows = sum(len(X) for X in evals)
    return bool(clashes == 0 and control >= 1), (f"{clashes} shared rows among {rows} evaluation rows; "
                                                  f"planted training row found: {control >= 1}")


# ----------------------------------------------------------------------------- hypotheses
def seed_metrics(pair):
    m, b = pair["models"]["sober"], pair["models"]["confidence"]
    cm, per_m = clarity_table(m, pair["splits"]["shifted"])
    cb, per_b = clarity_table(b, pair["splits"]["shifted"])
    bm, per_bm = clarity_table(m, pair["splits"]["blind"])
    bb, per_bb = clarity_table(b, pair["splits"]["blind"])
    ko = {f"{n}:{md}": clarity_table(knockout(m, n, md), pair["splits"]["shifted"])[0] - cm for n, md in KO_PLAN}
    return {"sig": cm - cb, "nec": ko["veil_signal:mean"] - ko["surface_statistics:mean"], "blind": bm - bb,
            "ko": ko, "per_model": {**per_m, **per_bm}, "per_base": {**per_b, **per_bb}}


def run_hypotheses(metrics, rng):
    rows, key = [], {"H-SIG": "sig", "H-NEC": "nec", "H-BLIND": "blind"}
    for spec in MIND_CARD["hypotheses"]:
        diffs = [s[key[spec["id"]]] for s in metrics]
        mean, ci = bootstrap_ci(diffs, rng)
        rows.append({"id": spec["id"], "metric": spec["metric"], "mean_diff": mean, "ci95": ci, "mesi": spec["mesi"],
                     "n_seeds": len(diffs), "verdict": verdict(mean, ci, spec["direction"], spec["mesi"])})
    sig = modules(build_model(D_OBS, C_CLASSES))
    kos = []
    for name, mode in KO_PLAN:
        mean, ci = bootstrap_ci([s["ko"][f"{name}:{mode}"] for s in metrics], rng)
        kos.append({"module": name, "mode": mode, "signature": sig[name]["signature"], "metric_change": mean,
                    "ci95": ci})
    return rows, kos


# ----------------------------------------------------------------------------- report
def _correctness(pair, mode, rng):
    c1 = test_c1(pair)
    c3 = c3_criteria(pair["models"]["sober"], pair["losses"]["sober"], pair["splits"]["heldout"])
    rows = [{"id": "C1", "name": "gradient_check", "passed": c1["passed"],
             "detail": f"{c1['tensors_checked']}/{c1['tensors_total']} tensors x 2 gates x 2 batches, at init and "
                       f"after 60 steps; max rel error {c1['max_rel_error']:.2e}"}]
    ok2, d2 = test_c2(pair)
    rows.append({"id": "C2", "name": "determinism_finiteness", "passed": ok2, "detail": d2})
    rows.append({"id": "C3", "name": "learning", "passed": c3["passed"],
                 "detail": f"loss drop {c3['drop']:.1%}; held-out clarity {c3['clarity']:+.3f} nats over the prior"})
    ok4, d4 = test_c4(pair)
    rows.append({"id": "C4", "name": "shuffled_label_control", "passed": ok4, "detail": d4})
    mut = {"detected": 0, "total": 0, "score": 1.0}
    if _ACTIVE["mutant"] is None:
        det = test_c5(pair, 300 if mode == "full" else 200)
        mut = {"detected": int(sum(det.values())), "total": len(det), "score": sum(det.values()) / len(det)}
        rows.append({"id": "C5", "name": "mutant_detection", "passed": mut["score"] == 1.0,
                     "detail": ", ".join(f"{k}: {'caught' if v else 'MISSED'}" for k, v in det.items())})
    for cid, name, ok, detail in test_c6(pair, rng):
        rows.append({"id": cid, "name": name, "passed": ok, "detail": detail})
    ok7, d7 = test_c7(pair)
    rows.append({"id": "C7", "name": "split_integrity", "passed": ok7, "detail": d7})
    return rows, c1, mut


def run_protocol(mode, base_seed, n_seeds, t0):
    steps, n_eval = STEPS[mode], N_EVAL[mode]
    seeds = [base_seed + i for i in range(n_seeds)]
    rng = np.random.default_rng(np.random.SeedSequence([base_seed, 999]))
    pairs = [train_pair(seeds[0], steps, n_eval)]
    rows, gc, mut = _correctness(pairs[0], mode, rng)
    for s in seeds[1:]:
        pairs.append(train_pair(s, steps, n_eval))
        if time.time() - t0 > BUDGET_S[mode]:
            raise BudgetExceeded(f"budget exceeded while training seed {s}")
    metrics = [seed_metrics(p) for p in pairs]
    hyp, kos = run_hypotheses(metrics, rng)
    if len(seeds) < 5:
        for h in hyp:
            h["verdict"] = "not evaluated"
    runtime = time.time() - t0
    rows.append({"id": "C8", "name": "budget", "passed": runtime <= BUDGET_S[mode],
                 "detail": f"{runtime:.1f} s of {BUDGET_S[mode]:.0f} s allowed ({mode})"})
    exit_code = 0 if all(r["passed"] for r in rows) else 1
    if exit_code == 0 and runtime > BUDGET_S[mode]:
        exit_code = 3
    per = {side: {k: np.mean([m[side][k] for m in metrics], axis=0).tolist() for k in metrics[0][side]}
           for side in ("per_model", "per_base")}
    return {"schema_version": "1.0", "chapter": 261, "file": os.path.basename(__file__),
            "card_revision": MIND_CARD["card_revision"],
            "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
            "seeds": seeds, "runtime_s": round(runtime, 2), "n_params": n_params(pairs[0]["models"]["sober"]),
            "n_params_baseline": n_params(pairs[0]["models"]["confidence"]), "gradcheck": gc,
            "correctness": rows, "mutants": mut, "hypotheses": hyp, "knockouts": kos, "per_condition": per,
            "task_types": list(TASK_TYPES), "exit_code": exit_code, "mode": mode}


def print_report(rep):
    print("=== VERIFIED REPORT · chapter 0261 ===")
    print(f"file: {rep['file']} · card revision {rep['card_revision']} · mode {rep['mode']}")
    print(f"environment: Python {rep['environment']['python']} · NumPy {rep['environment']['numpy']}")
    print(f"seeds: {rep['seeds']} · runtime {rep['runtime_s']:.1f} s · parameters: model {rep['n_params']}, "
          f"size-matched baseline {rep['n_params_baseline']}")
    gc = rep["gradcheck"]
    print(f"gradient check: {gc['tensors_checked']}/{gc['tensors_total']} tensors, {' + '.join(gc['checked_at'])}, "
          f"max relative error {gc['max_rel_error']:.2e} -> {'PASS' if gc['passed'] else 'FAIL'}")
    print("correctness:")
    for r in rep["correctness"]:
        print(f"  {r['id']:<5}{r['name']:<30}{'PASS' if r['passed'] else 'FAIL'}  {r['detail']}")
    mu = rep["mutants"]
    print(f"mutation score: {mu['detected']}/{mu['total']} = {mu['score']:.2f}")
    print("hypotheses (clarity = log 4 - NLL, nats above the prior; paired per seed; 95% bootstrap CI, 2000 resamples):")
    for h in rep["hypotheses"]:
        print(f"  {h['id']:<8} mean diff {h['mean_diff']:+.3f}  CI [{h['ci95'][0]:+.3f}, {h['ci95'][1]:+.3f}]  "
              f"mesi {h['mesi']:.2f}  n={h['n_seeds']}  -> {h['verdict']}")
    print("knockouts on the sober model (shifted split, change in clarity vs intact):")
    for k in rep["knockouts"]:
        print(f"  {k['module'] + ' (' + k['mode'] + ')':<32}{'signature' if k['signature'] else 'matched  ':<11}"
              f"{k['metric_change']:+.3f}  CI [{k['ci95'][0]:+.3f}, {k['ci95'][1]:+.3f}]")
    print("per condition, mean over seeds: clarity | polished accuracy | gate (raw, polished, sealed)")
    for kind in list(SHIFT_VEILS) + ["bent"]:
        m, b = rep["per_condition"]["per_model"][kind], rep["per_condition"]["per_base"][kind]
        print(f"  {kind:<7} sober {m[0]:+.3f} | {m[1]:.3f} | ({m[3]:.2f}, {m[4]:.2f}, {m[5]:.2f})   "
              f"confidence {b[0]:+.3f} | ({b[3]:.2f}, {b[4]:.2f}, {b[5]:.2f})")
    print(f"task types: {', '.join(rep['task_types'])} · exit code {rep['exit_code']}")
    print("=== END REPORT ===")


def real_data_bridge(path, seed):
    """Optional: a local numeric CSV (last column = class label) with the same veils; no downloads."""
    try:
        raw = np.loadtxt(path, delimiter=",")
    except ValueError:
        raw = np.loadtxt(path, delimiter=",", skiprows=1)
    X, y = raw[:, :-1], np.unique(raw[:, -1], return_inverse=True)[1]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    tr, te = perm[: int(0.8 * len(y))], perm[int(0.8 * len(y)):]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    world = {"rows": X[tr], "labels": y[tr], "mu": mu, "sd": sd}
    m = build_model(X.shape[1], int(y.max()) + 1, "vector_classification", rng)
    fit(m, {"world": world, "reference": X[tr]}, STEPS["quick"], rng)
    test_world = {"rows": X[te], "labels": y[te], "mu": mu, "sd": sd}
    return clarity_table(m, episodes(test_world, ("clear", "moderate", "fog", "rotate"), 10, rng))[1]


# ----------------------------------------------------------------------------- command line
def main(argv=None):
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__),
                                 description="Chapter 0261 · the polisher's test (full protocol by default)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--card", action="store_true")
    ap.add_argument("--mutant", choices=sorted(MUTANTS), default=None)
    ap.add_argument("--data", default=None)
    args = ap.parse_args(argv)
    if args.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    t0, mode = time.time(), "quick" if (args.quick or args.mutant) else "full"
    n_seeds = args.seeds if args.seeds is not None else (1 if mode == "quick" else 5)
    if n_seeds < 1:
        ap.error("--seeds must be at least 1")
    _ACTIVE["mutant"] = args.mutant
    print(f"seeds: {[args.seed + i for i in range(n_seeds)]} · mode {mode}"
          + (f" · mutant {args.mutant}" if args.mutant else ""))
    try:
        rep = run_protocol(mode, args.seed, n_seeds, t0)
    except FloatingPointError as err:
        print(f"non-finite values: {err}")
        return 4
    except BudgetExceeded as err:
        print(str(err))
        return 3
    print_report(rep)
    if args.data:
        if os.path.exists(args.data):
            for kind, row in real_data_bridge(args.data, args.seed).items():
                print(f"real-data bridge {kind:<9} clarity {row[0]:+.3f}  gate {np.round(row[3:], 2).tolist()}")
        else:
            print(f"real-data bridge skipped: {args.data} not found")
    if args.json:
        write_report(rep, args.json)
    return rep["exit_code"]


if __name__ == "__main__":
    sys.exit(main())

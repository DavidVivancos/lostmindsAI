#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0309 · Suhrawardi
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 16, Minds 301-320 Available on Amazon https://www.amazon.com/dp/B0HMQ7TVJM
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos 
# END ATTRIBUTION
"""Ledger of Rays: a gated-linear lattice that knows the increase each of its units gives.

Thesis
    A knower differs from a lit wall not in how much light reaches it but in being
    present to the composition of that light: of its own summed state it knows which
    source, by which route, gave which increase, so when one source is darkened it
    can find it and let the other routes carry the light.

Evidence and provenance (belief: the figure's own surviving works)
    D1  PI s.152: rays falling on a barrier add up in one locus and cannot be told
        apart except through their causes; the barrier knows nothing of the share of
        each, while a living, self-aware essence knows what lights it and the share
        of each (Walbridge and Ziai 1999, paraphrased).
    D2  PI s.151: each dominating light receives the Light of Lights directly and
        again through every higher light, so the count doubles rank by rank (2, 4, 8).
    D3  PI s.152: when several lamps light one wall and one is shaded, another remains.
    D4  Intimations (Talwihat, Opera I 70-74) and PI ss.114-117: self-knowledge is
        by presence, not by a form or representation of the self.
    D5  PI ss.101-104: vision needs a lit object, facing, and no veil; no form is
        carried into the eye.
    D6  PI s.151: beholding the higher light is a different relation from being
        shone upon by it.
    D7  Against Avicenna's representational epistemology (Kaukua 2015, 106-117).
    D8  PI s.151: veiling is placed in bodies (barriers); presence is treated as
        unerring (Ha'iri Yazdi 1992). This is the blind spot tested below.

Doctrine -> mechanism -> test
    D2 -> M1 reflections: unit n reads the input directly and every earlier unit
          (complete DAG, 2^n input-to-unit paths)          -> C6.d, H-SIG
    D1 -> M2 ledger: value stream is linear given input-only gates, so the readout
          splits exactly into per-path rows with visited-unit masks -> C6.a, C6.b
    D6 -> M3 beholding: per unit, a least-squares prediction of its increment from
          the complement of itself; the residual, averaged over the last 32 inputs
          (a darkened unit stays dark), is the unit's anomaly score
                                                          -> C6.c, H-NEC, H-RIVAL
    D3 -> M4 exclusion: if the top score passes a clean-calibrated threshold, read
          out without every path through that unit        -> H-SIG, H-NEC
    D8 -> blind-spot condition: a diffuse veil on every unit at once -> H-BLIND
    D7 -> rival: a representational monitor that learns fault signatures from the
          units' states (Avicennan knowledge through a form of the self) -> H-RIVAL

Research question (interpretability and modularity; metacognition)
    Can a network localize and route around a corrupted internal module from an
    exact in-pass ledger of which paths produced its output, for fault types it
    never saw, without labels or retraining?

Closest prior art and delta
    Dense cascades (Fahlman and Lebiere 1990; Huang et al. 2017), gated linear
    networks and neural path decomposition (Veness et al. 2021; Lakshminarayanan
    and Singh 2020), drop-path training (Larsson et al. 2017), residual nets as
    path ensembles (Veit et al. 2016), learned failure monitors. Delta: the exact
    path ledger is consumed by the network itself at inference, and the power-set
    structure of the doubled paths makes 'every path through unit k' removable in
    closed form, so localization needs no model of the fault.

Blind spot
    The rule assumes one locatable veil. A pervasive dimming of every unit has no
    location; excluding one unit then throws away light it still needed.

Task (lit scene, 4 classes)
    z ~ N(0, I4); y = 2[z0 z1 > 0] + [|z2| > |z3|]
    x = alpha * A z + noise, A fixed (task seed 308), alpha ~ U(0.7, 1.4) nuisance gain
    splits: train, validation, held-out test, brighter dawn (alpha 1.6-2.4, more noise)
    shaded split: held-out test with one internal unit corrupted by a fault family
    never used in training or calibration (sign, rotate, stuck, tint)

Limits
    Synthetic task, six units, one corrupted unit at a time. A research prototype of
    an AGI-oriented self-monitoring mechanism, not an AGI and not a model of a person.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 2,
    "revision_log": [
        ("rev 2 (2026-09-24): mechanism change after the first full run; hypotheses, metrics, mesi and the "
         "0.99 calibration quantile unchanged. Rev 1 verdicts over seeds 0-4: H-SIG contradicted (-0.0331, "
         "ci [-0.0442, -0.0226]); H-NEC supported (-0.0556); H-BLIND inconclusive (-0.0017); H-RIVAL "
         "contradicted (-0.0279). Diagnosis on seed 0: when the rule fired it named the right unit almost "
         "always, but each input was judged alone while a corruption persists, so it rarely fired. Rev 2 "
         "averages each unit's score over a causal window of the last 32 inputs; the rival gets the same "
         "window. The rev 1 rule is kept as the knockout 'beholding (single)'."),
        ("rev 2 note: C4 now compares the shuffled-label model with the band spanned by the test class rates "
         "(+-0.06, value unchanged); comparing with one constant predictor was ill-posed on small test sets.")],
    "generation": {"template_version": "codeguidelines v1.0 (15 Sep 2026)", "generator": "Claude",
                   "generator_version": "claude-opus-5-5", "date": "2026-09-24"},
    "id": 309, "figure": "Suhrawardi", "born": 1154, "died": 1191, "civilization": "Persian",
    "provenance": "belief",
    "thesis": ("A knower differs from a lit wall by being present to the composition of its own "
               "illumination: it knows which route gave which increase and can drop a darkened one."),
    "evidence": [
        {"id": "D1", "claim": "Barrier registers only summed rays; a living light knows the share of each",
         "basis": "primary", "source": "Hikmat al-Ishraq s.152 (tr. Walbridge and Ziai 1999)"},
        {"id": "D2", "claim": "Rays reach each lower light directly and through every intermediary; count doubles",
         "basis": "primary", "source": "Hikmat al-Ishraq s.151"},
        {"id": "D3", "claim": "Of several lamps on one wall, if one is shaded another remains",
         "basis": "primary", "source": "Hikmat al-Ishraq s.152"},
        {"id": "D4", "claim": "Self-knowledge by presence, not by a representation of the self",
         "basis": "primary", "source": "al-Talwihat, Opera Metaphysica et Mystica I (Corbin 1945) 70-74; PI ss.114-117"},
        {"id": "D5", "claim": "Vision: lit object, facing, no veil; no form transmitted",
         "basis": "primary", "source": "Hikmat al-Ishraq ss.101-104; Marcotte, SEP 'Suhrawardi' (2019) s.3.1"},
        {"id": "D6", "claim": "Beholding is a relation distinct from being shone upon",
         "basis": "primary", "source": "Hikmat al-Ishraq s.151"},
        {"id": "D7", "claim": "Rejects Avicenna's representational account of knowing the self",
         "basis": "scholarship", "source": "Kaukua, Self-Awareness in Islamic Philosophy (2015) 106-117"},
        {"id": "D8", "claim": "Veiling located in barriers; presential knowledge held immune to error",
         "basis": "scholarship", "source": "PI s.151; Ha'iri Yazdi, Knowledge by Presence (1992)"},
    ],
    "research_question": {"category": "interpretability and modularity",
                          "question": ("Can a network localize and route around a corrupted internal module from "
                                       "an exact in-pass path ledger, for unseen fault types, without labels?")},
    "mechanism": {
        "name": "Ledger of Rays (path-presence lattice)",
        "family": "relational and graph learning; gated linear networks with exact path decomposition",
        "signature_modules": ["reflections", "beholding"],
        "closest_prior_art": ["Cascade-Correlation (Fahlman and Lebiere 1990); DenseNet (Huang et al. 2017)",
                              "Gated Linear Networks (Veness et al. 2021); Neural Path Features (Lakshminarayanan and Singh 2020)",
                              "FractalNet drop-path (Larsson et al. 2017); Veit et al. 2016 lesion study",
                              "Learned failure detectors reading internal activations"],
        "overlap": "Medium",
        "prior_art_queries": ["self-diagnosis localize faulty layer inference path decomposition",
                              "exact path attribution gated linear network",
                              "fault tolerant neural network leave-one-module-out readout",
                              "silent data corruption deep learning inference mitigation"],
        "contribution_type": "mechanism",
        "delta": ("The network consumes its own exact per-path ledger at inference; the power-set path structure "
                  "gives closed-form removal of every path through a unit, so localization needs no fault model."),
    },
    "traceability": [
        {"doctrine": "D2", "mechanism": "M1 reflections", "property_test": "C6.d", "hypothesis": "H-SIG"},
        {"doctrine": "D1", "mechanism": "M2 ledger", "property_test": "C6.a", "hypothesis": "H-NEC"},
        {"doctrine": "D6", "mechanism": "M3 beholding", "property_test": "C6.c", "hypothesis": "H-RIVAL"},
        {"doctrine": "D3", "mechanism": "M4 exclusion", "property_test": "C6.b", "hypothesis": "H-SIG"},
        {"doctrine": "D8", "mechanism": "M3+M4 under diffuse veil", "property_test": "C6.b", "hypothesis": "H-BLIND"},
        {"doctrine": "D7", "mechanism": "rival representational monitor", "property_test": "C1", "hypothesis": "H-RIVAL"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": ("On the shaded split (unseen internal faults) the lattice with presence "
                                      "exclusion beats a size-matched MLP trained with group dropout"),
         "metric": "accuracy_shaded", "split": "shifted", "comparison": "model - baseline",
         "direction": "greater", "mesi": 0.03, "seeds": 5},
        {"id": "H-NEC", "statement": ("Knocking out the ledger readout (beholding -> identity) costs more shaded "
                                      "accuracy than zeroing the parameter-matched last direct ray"),
         "metric": "accuracy_shaded", "comparison": "signature_knockout - matched_knockout",
         "direction": "less", "mesi": 0.03, "seeds": 5},
        {"id": "H-BLIND", "statement": ("Under a diffuse veil on every unit the single-veil rule excludes a unit it "
                                        "still needed and does worse than the summed readout"),
         "condition": "diffuse_veil", "grounding": "veiling assumed local and presence unerring (PI s.151)",
         "metric": "accuracy_diffuse", "comparison": "model - barrier", "direction": "less",
         "mesi": 0.005, "seeds": 5},
        {"id": "H-RIVAL", "statement": ("Presence localization (no fault model) beats a representational monitor "
                                        "trained on other fault types, on unseen faults"),
         "metric": "accuracy_shaded", "comparison": "model - rival", "direction": "greater",
         "mesi": 0.02, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.35, "margin_over_trivial": 0.30, "shuffled_band": 0.06,
                   "gradcheck_rel": 1e-5, "exactness_abs": 1e-8, "control_violation": 1e-4,
                   "window": 32},
    "probe_predictions": [
        {"probe": "P1", "expected": "below baseline"}, {"probe": "P2", "expected": "equal to baseline"},
        {"probe": "P3", "expected": "equal to baseline"}, {"probe": "P4", "expected": "below baseline"},
        {"probe": "P5", "expected": "equal to baseline"}, {"probe": "P6", "expected": "below baseline"},
        {"probe": "P7", "expected": "below baseline"}, {"probe": "P8", "expected": "equal to baseline"},
        {"probe": "P9", "expected": "above baseline"}, {"probe": "P10", "expected": "equal to baseline"},
        {"probe": "P11", "expected": "equal to baseline"}],
    "dialectic_links": [
        {"chapter": 257, "relation": "rival", "test": "H-RIVAL"},
        {"chapter": 143, "relation": "teacher", "test": "none (via the Arabic Plotinus, read as Plato)"},
        {"chapter": 277, "relation": "teacher", "test": "none (Mishkat al-anwar parallels, Sinai 2016)"},
        {"chapter": 560, "relation": "successor", "test": "none (gradation of light recast as being)"}],
    "corpus_neighbors": [
        {"chapter": 257, "similarity": None, "difference": "Ibn Sina: estimative leap and conjunction; here, exact self-ledger"},
        {"chapter": 21, "similarity": None, "difference": "Yajnavalkya: unobjectifiable witness by negation; here, presence to composition"},
        {"chapter": 143, "similarity": None, "difference": "Plotinus: reversion to source; here, no return, only routing around"},
        {"chapter": 333, "similarity": None, "difference": "Roger Bacon: learnable medium; here, nothing transmitted is re-encoded, paths are removed"},
        {"chapter": 316, "similarity": None, "difference": "Ibn Arabi: imaginal re-formation; here, a fixed exact ledger"},
        {"chapter": 307, "similarity": None, "difference": "Attar: population subtraction; here, closed-form exclusion inside one network"}],
    "barometer": {
        "cognitive_processing": ["clean held-out accuracy (C3)"],
        "embodied_cognition": ["accuracy with one internal processing unit corrupted (shaded split)"],
        "world_modeling": ["brighter-dawn split: gain outside the training range"],
        "consciousness": ["localization accuracy of its own corrupted unit; clean false-trigger rate"],
        "language_understanding": [], "emotional_intelligence": [], "creativity": [],
        "autonomy": ["self-repair by exclusion without labels or retraining"]},
    "task_types": ["vector_classification"],
    "applications": [
        {"use": "inference that survives silent data corruption in one accelerator unit",
         "sector": "computing infrastructure", "dataset": "UCI Human Activity Recognition Using Smartphones"},
        {"use": "robot perception that isolates a degraded processing module",
         "sector": "robotics", "dataset": "UCI Wall-Following Robot Navigation"},
        {"use": "auditing which edited or merged component broke a model",
         "sector": "ML operations", "dataset": "UCI Covertype"}],
    "safety_notes": ("Self-monitoring prototype; no people data. Exclusion can hide a failure from operators, so "
                     "every trigger is reported with the excluded unit rather than silently absorbed."),
}

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

TASK_TYPES = ["vector_classification"]
TASK_SEED = 309
IN_DIM, LATENT, N_CLASSES = 12, 4, 4
N_LIGHTS, WIDTH = 6, 10
MLP_HIDDEN, RIVAL_HIDDEN = 48, 32
SHADE_P, LR, CLIP, WD, BATCH, WINDOW = 0.35, 0.01, 5.0, 1e-4, 128, 32
TRAIN_FAULTS = ("zero", "gain", "noise")            # seen by the rival monitor only
NOVEL_FAULTS = ("sign", "rotate", "stuck", "tint")   # the shaded split
FULL = {"n_train": 3000, "n_val": 1000, "n_test": 2000, "steps": 1400, "rival_steps": 900}
QUICK = {"n_train": 1500, "n_val": 500, "n_test": 800, "steps": 450, "rival_steps": 300}
MUTANT = None


class NonFinite(Exception):
    pass


# BEGIN STANDARD UTILITIES v1.0
def softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def logsumexp(z, axis=-1):
    m = np.max(z, axis=axis, keepdims=True)
    return np.squeeze(m + np.log(np.sum(np.exp(z - m), axis=axis, keepdims=True)), axis=axis)


def softplus(x):
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


def clip_global_norm(grads, max_norm):
    norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    if norm > max_norm:
        grads = {k: g * (max_norm / (norm + 1e-12)) for k, g in grads.items()}
    return grads, norm


def finite_difference_check(loss_fn, params, grads, rng, n_entries=20, eps=1e-6, floor=1e-4):
    """Central differences on n_entries random entries of every tensor plus its largest-gradient entry.
    Relative error uses a denominator floor so float64 round-off on near-zero entries is not counted."""
    worst = {}
    for name, p in params.items():
        flat, g = p.reshape(-1), grads[name].reshape(-1)
        idx = set(rng.choice(flat.size, size=min(n_entries, flat.size), replace=False).tolist())
        idx.add(int(np.argmax(np.abs(g))))
        err = 0.0
        for i in sorted(idx):
            old = flat[i]
            flat[i] = old + eps
            lp = loss_fn()
            flat[i] = old - eps
            lm = loss_fn()
            flat[i] = old
            num = (lp - lm) / (2 * eps)
            err = max(err, abs(num - g[i]) / max(abs(num), abs(g[i]), floor))
        worst[name] = err
    return worst


def paired_bootstrap(diffs, rng, n_boot=2000):
    d = np.asarray(diffs, dtype=float)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def verdict(mean, lo, hi, direction, mesi):
    sgn = 1.0 if direction == "greater" else -1.0
    if sgn * lo > 0 and sgn * hi > 0 and sgn * mean >= mesi:
        return "supported"
    if sgn * lo < 0 and sgn * hi < 0:
        return "contradicted"
    return "inconclusive"


def write_report(report, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
# END STANDARD UTILITIES


# ---------------------------------------------------------------- data and tasks
def task_mixing():
    return np.random.default_rng(TASK_SEED).normal(size=(IN_DIM, LATENT)) / math.sqrt(LATENT)


def lit_scene(n, rng, gain=(0.7, 1.4), noise=0.10):
    """Hidden scene z seen through a fixed mixing under a random overall brightness."""
    z = rng.normal(size=(n, LATENT))
    y = 2 * (z[:, 0] * z[:, 1] > 0).astype(int) + (np.abs(z[:, 2]) > np.abs(z[:, 3])).astype(int)
    alpha = rng.uniform(gain[0], gain[1], size=(n, 1))
    return alpha * (z @ task_mixing().T) + noise * rng.normal(size=(n, IN_DIM)), y


def make_data(rng, cfg):
    out = {}
    for split, n in (("tr", cfg["n_train"]), ("va", cfg["n_val"]), ("te", cfg["n_test"])):
        out["X" + split], out["y" + split] = lit_scene(n, rng)
    out["Xbr"], out["ybr"] = lit_scene(cfg["n_test"], rng, gain=(1.6, 2.4), noise=0.15)
    return out


# ---------------------------------------------------------------- model
def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


class Lattice:
    """Vertical order of N units ('lights'); unit n reads the input directly and every earlier unit."""

    def __init__(self, params, n_lights):
        self.params, self.N = params, n_lights
        self.pairs = [(n, m) for n in range(n_lights) for m in range(n)]
        self.pair_of = {p: i for i, p in enumerate(self.pairs)}
        self.d = params["U"].shape[1]
        self.presence, self.presence_on, self.beholding_mode = None, True, "argmax"
        self.gate_override = None

    def clone(self):
        c = Lattice({k: v.copy() for k, v in self.params.items()}, self.N)
        c.presence, c.presence_on, c.beholding_mode = self.presence, self.presence_on, self.beholding_mode
        c.gate_override = self.gate_override
        return c


def build_model(in_dim, out_dim, task_type, rng, n_lights=N_LIGHTS, width=WIDTH, **cfg):
    if task_type not in TASK_TYPES:
        raise ValueError(task_type)
    N, d = n_lights, width
    pairs = [(n, m) for n in range(N) for m in range(n)]
    p = {"U": rng.normal(0, 1 / math.sqrt(in_dim), (N, d, in_dim)),
         "G": rng.normal(0, 1 / math.sqrt(in_dim), (N, d, in_dim)),
         "gb": np.zeros((N, d)),
         "R": np.stack([rng.normal(0, 0.7 / math.sqrt(d * n), (d, d)) for n, _ in pairs]),
         "O": rng.normal(0, 1 / math.sqrt(d * N), (N, out_dim, d)),
         "c": np.zeros(out_dim)}
    return Lattice(p, N)


def gates_and_rays(model, X):
    P = model.params
    a = np.einsum("ndi,bi->nbd", P["G"], X) + P["gb"][:, None, :]
    g = sigmoid(a) if model.gate_override is None else np.broadcast_to(
        model.gate_override[:, None, :], a.shape)
    return g, np.einsum("ndi,bi->nbd", P["U"], X), a


def apply_fault(h, fault):
    M, b = fault
    h = h if M is None else h @ M.T
    return h if b is None else h + b


def lattice_forward(model, X, keep=None, faults=None, value_nonlin=False):
    """h_n = g_n(x) * (U_n x + sum_{m<n} R_nm h'_m); logits = sum_n O_n h'_n + c."""
    P = model.params
    g_all, u_all, _ = gates_and_rays(model, X)
    hs, ps = [], []
    z = np.tile(P["c"], (X.shape[0], 1))
    for n in range(model.N):
        p = u_all[n].copy()
        for m in range(n):
            p += hs[m] @ P["R"][model.pair_of[(n, m)]].T
        h = g_all[n] * (np.tanh(p) if value_nonlin else p)
        if faults and n in faults:
            h = apply_fault(h, faults[n])
        if keep is not None:
            h = h * keep[:, n:n + 1]
        hs.append(h)
        ps.append(p)
        z += h @ P["O"][n].T
    return z, (hs, ps, g_all)


def loss_and_grads(model, batch):
    X, y, keep = batch["X"], batch["y"], batch.get("keep")
    P, N, B = model.params, model.N, batch["X"].shape[0]
    z, (hs, ps, gs) = lattice_forward(model, X, keep=keep)
    loss = float(np.mean(logsumexp(z) - z[np.arange(B), y]))
    loss += 0.5 * WD * sum(float(np.sum(P[k] ** 2)) for k in ("U", "G", "R", "O"))
    dz = softmax(z)
    dz[np.arange(B), y] -= 1.0
    dz /= B
    grads = {k: np.zeros_like(v) for k, v in P.items()}
    grads["c"] = dz.sum(0)
    dps = [None] * N
    for n in reversed(range(N)):
        grads["O"][n] = dz.T @ hs[n]
        dh = dz @ P["O"][n]
        for j in range(n + 1, N):
            dh = dh + dps[j] @ P["R"][model.pair_of[(j, n)]]
        if keep is not None:
            dh = dh * keep[:, n:n + 1]
        dp = dh * gs[n]
        da = dh * ps[n] * gs[n] * (1.0 - gs[n])
        grads["G"][n], grads["gb"][n], grads["U"][n] = da.T @ X, da.sum(0), dp.T @ X
        for m in range(n):
            grads["R"][model.pair_of[(n, m)]] = dp.T @ hs[m]
        dps[n] = dp
    for k in ("U", "G", "R", "O"):
        grads[k] += WD * P[k]
    if MUTANT == "zero_grad_R":
        grads["R"][:] = 0.0
    if MUTANT == "zero_grad_G":
        grads["G"][:] = 0.0
    return loss, grads


def ledger(model, X, faults=None):
    """Exact per-path rows at the readout, (R, B, C), with the set of units each path visits, (R, N).
    Gates depend on the input only, so every path is linear in its rays and the rows sum to the logits."""
    P, N, B, d = model.params, model.N, X.shape[0], model.d
    g_all, u_all, _ = gates_and_rays(model, X)
    rows_by, masks_by, out_rows, out_masks = [], [], [], []
    for n in range(N):
        rows, masks = [(g_all[n] * u_all[n])[None]], [np.eye(N, dtype=bool)[n][None]]
        for m in range(n):
            rows.append(g_all[n][None] * (rows_by[m] @ P["R"][model.pair_of[(n, m)]].T))
            mk = masks_by[m].copy()
            mk[:, n] = True
            masks.append(mk)
        rows, masks = np.concatenate(rows), np.concatenate(masks)
        if faults and n in faults:
            M, b = faults[n]
            if M is not None:
                rows = rows @ M.T
            if b is not None:
                rows = np.concatenate([rows, np.broadcast_to(b, (B, d))[None]])
                masks = np.concatenate([masks, np.eye(N, dtype=bool)[n][None]])
        if MUTANT == "ledger_misattribution":
            masks = np.roll(masks, 1, axis=1)
        rows_by.append(rows)
        masks_by.append(masks)
        out_rows.append(rows @ P["O"][n].T)
        out_masks.append(masks)
    return np.concatenate(out_rows), np.concatenate(out_masks)


def increments(rows, masks):
    """Delta_k: the summed share of every path that visits unit k (the increase each unit gives)."""
    return np.einsum("rbc,rk->kbc", rows, masks.astype(float))


def presence_scores(model, rows, masks):
    """Beholding: each unit's increment against its least-squares prediction from its complement."""
    z = rows.sum(0) + model.params["c"]
    D = increments(rows, masks)
    L = z[None] - D
    Phi = np.concatenate([L, np.ones(L.shape[:2] + (1,))], axis=2)
    pred = np.einsum("nbj,njc->nbc", Phi, model.presence["W"])
    s = np.mean((D - pred) ** 2 / model.presence["var"][:, None, :], axis=2).T
    return s, z, L, D, Phi


def presence_fit(model, X, Xval, quantile=0.99, ridge=1e-3):
    """Non-gradient rule: ridge least squares per unit on clean data; threshold from clean validation."""
    rows, masks = ledger(model, X)
    z = rows.sum(0) + model.params["c"]
    D = increments(rows, masks)
    L = z[None] - D
    Ws, vs = [], []
    for k in range(model.N):
        Phi = np.hstack([L[k], np.ones((X.shape[0], 1))])
        W = np.linalg.solve(Phi.T @ Phi + ridge * np.eye(Phi.shape[1]), Phi.T @ D[k])
        Ws.append(W)
        vs.append(np.mean((D[k] - Phi @ W) ** 2, axis=0) + 1e-8)
    model.presence = {"W": np.stack(Ws), "var": np.stack(vs), "ridge": ridge}
    s = presence_scores(model, *ledger(model, Xval))[0]
    for w in (1, WINDOW):
        model.presence[f"tau{w}"] = float(np.quantile(window_mean(s, w).max(axis=1), quantile))


def window_mean(s, w):
    """Causal running mean over the last w inputs of a stream (rows in arrival order)."""
    if w <= 1:
        return s
    c = np.concatenate([np.zeros((1,) + s.shape[1:]), np.cumsum(s, axis=0)])
    lo = np.maximum(np.arange(len(s)) + 1 - w, 0)
    return (c[1:] - c[lo]) / (np.arange(len(s)) + 1 - lo)[:, None]


def mind_readout(model, X, faults=None, choice=None, cache=None):
    """Full mind: ledger -> beholding scores -> exclude the darkened unit's paths if the top score is atypical."""
    rows, masks = ledger(model, X, faults) if cache is None else cache
    s, z, L, _, _ = presence_scores(model, rows, masks)
    info = {"z": z, "cache": (rows, masks), "k": s.argmax(1), "trig": np.zeros(len(X), bool)}
    if not model.presence_on:
        return z, info
    w = 1 if model.beholding_mode == "single" else WINDOW
    sw = window_mean(s, w)
    info["trig"], info["k"] = (sw.max(1) > model.presence[f"tau{w}"], sw.argmax(1)) if choice is None else choice
    alt = L.mean(0) if model.beholding_mode == "mean" else L[info["k"], np.arange(len(X))]
    return np.where(info["trig"][:, None], alt, z), info


def predict(model, X):
    if isinstance(model, Dense):
        return dense_forward(model, X)[0].argmax(1)
    return mind_readout(model, X)[0].argmax(1)


def hidden_states(model, X):
    return np.concatenate(lattice_forward(model, X)[1][0], axis=1)


def n_params(model):
    return int(sum(v.size for v in model.params.values()))


# ---------------------------------------------------------------- baselines and rival mechanisms
class Dense:
    """Plain tanh MLP. Size-matched baseline (groups on layer 1 play the role of units) and the rival."""

    def __init__(self, sizes, rng, n_groups=None):
        self.params, self.L = {}, len(sizes) - 1
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            self.params[f"W{i}"] = rng.normal(0, 1 / math.sqrt(a), (b, a))
            self.params[f"b{i}"] = np.zeros(b)
        self.n_groups = n_groups
        self.gsize = sizes[1] // n_groups if n_groups else None


def dense_forward(net, X, keep=None, faults=None):
    acts, h = [X], X
    for i in range(net.L):
        a = h @ net.params[f"W{i}"].T + net.params[f"b{i}"]
        if i == net.L - 1:
            return a, acts
        h = np.tanh(a)
        if i == 0 and keep is not None:
            h = h * np.repeat(keep, net.gsize, axis=1)
        if i == 0 and faults:
            h = h.copy()
            for k, f in faults.items():
                sl = slice(k * net.gsize, (k + 1) * net.gsize)
                h[:, sl] = apply_fault(h[:, sl], f)
        acts.append(h)


def dense_loss_and_grads(net, batch):
    X, y, keep = batch["X"], batch["y"], batch.get("keep")
    z, acts = dense_forward(net, X, keep)
    B = X.shape[0]
    loss = float(np.mean(logsumexp(z) - z[np.arange(B), y]))
    loss += 0.5 * WD * sum(float(np.sum(net.params[f"W{i}"] ** 2)) for i in range(net.L))
    delta = softmax(z)
    delta[np.arange(B), y] -= 1.0
    delta /= B
    grads = {}
    for i in reversed(range(net.L)):
        W = net.params[f"W{i}"]
        grads[f"W{i}"] = delta.T @ acts[i] + WD * W
        grads[f"b{i}"] = delta.sum(0)
        if i > 0:
            dh = delta @ W
            if i == 1 and keep is not None:
                dh = dh * np.repeat(keep, net.gsize, axis=1)
            delta = dh * (1.0 - acts[i] ** 2)
    return loss, grads


def standardize_fit(F):
    return F.mean(0), F.std(0) + 1e-6


def rival_features(model, X, faults, stats):
    """The Avicennan rival reads a form of the self: all unit states, standardized on clean data."""
    return (np.concatenate(lattice_forward(model, X, faults=faults)[1][0], axis=1) - stats[0]) / stats[1]


def train_rival(model, data, rng, steps, stds):
    """Representational monitor trained to name the faulty unit for the TRAIN_FAULTS family."""
    Xtr = data["Xtr"]
    stats = standardize_fit(np.concatenate(lattice_forward(model, Xtr)[1][0], axis=1))
    F, Y = [rival_features(model, Xtr[rng.permutation(len(Xtr))[:600]], None, stats)], [np.zeros(600, int)]
    for k in range(model.N):
        for kind in TRAIN_FAULTS:
            idx = rng.permutation(len(Xtr))[:150]
            fault = make_fault(kind, model.d, stds[k], rng, len(idx))
            F.append(rival_features(model, Xtr[idx], {k: fault}, stats))
            Y.append(np.full(len(idx), k + 1))
    F, Y = np.concatenate(F), np.concatenate(Y)
    net = Dense([F.shape[1], RIVAL_HIDDEN, model.N + 1], rng)
    fit_dense(net, F, Y, steps, rng, shade=False)
    net.stats = stats
    return net


def rival_choice(model, rival, X, faults):
    p = window_mean(softmax(dense_forward(rival, rival_features(model, X, faults, rival.stats))[0]), WINDOW)
    return p[:, 0] < 0.5, p[:, 1:].argmax(1)


# ---------------------------------------------------------------- faults (test-time shading)
def random_orthogonal(d, rng):
    q, r = np.linalg.qr(rng.normal(size=(d, d)))
    return q * np.sign(np.diag(r))


def make_fault(kind, d, std, rng, B, foreign=None):
    """Affine corruption h -> M h + b of one unit, scaled by that unit's clean state spread."""
    I = np.eye(d)
    if kind == "zero":
        return 0 * I, None
    if kind == "gain":
        return 3.0 * I, None
    if kind == "noise":
        return I, std * rng.normal(size=(B, d))
    if kind == "sign":
        return -I, None
    if kind == "rotate":
        return random_orthogonal(d, rng), None
    if kind == "stuck":
        v = rng.normal(size=d)
        return 0 * I, std * math.sqrt(d) * v / np.linalg.norm(v)
    if kind == "tint":
        return random_orthogonal(d, rng) @ np.diag(rng.uniform(0.2, 2.5, d)) @ random_orthogonal(d, rng).T, None
    if kind == "counterfeit":
        return 0 * I, foreign
    raise ValueError(kind)


def diffuse_veil(n_units, d, stds, rng, B):
    """Every unit mildly rotated and noised at once: a veil with no single location."""
    out = {}
    for k in range(n_units):
        u, _, vt = np.linalg.svd(np.eye(d) + 0.5 * rng.normal(size=(d, d)) / math.sqrt(d))
        out[k] = (u @ vt, 0.25 * stds[k] * rng.normal(size=(B, d)))
    return out


def unit_states(model, X):
    if isinstance(model, Dense):
        h = dense_forward(model, X)[1][1]
        return [h[:, k * model.gsize:(k + 1) * model.gsize] for k in range(model.n_groups)]
    return lattice_forward(model, X)[1][0]


# ---------------------------------------------------------------- registries
MODULES = {
    "direct_rays": (["U"], "input-to-unit linear projections (no intermediary)", False),
    "gates": (["G", "gb"], "input-conditioned sigmoid gates on each unit's linear value", False),
    "reflections": (["R"], "strictly lower-triangular unit-to-unit maps (complete DAG, 2^n paths)", True),
    "readout": (["O", "c"], "linear readout summing every unit into logits", False),
    "beholding": ([], "exact path-ledger anomaly score per unit and leave-one-unit-out readout", True),
    "last_direct_ray": (["U[-1]"], "input projection of the last unit (parameter-matched control)", False),
}


def modules(model):
    return {k: {"params": v[0], "role": v[1], "signature": v[2]} for k, v in MODULES.items()}


def knockout(model, name, mode):
    """Copy of the model with one module replaced by identity, zero or mean output (no refit)."""
    m = model.clone()
    if name == "beholding":
        if mode in ("mean", "single"):
            m.beholding_mode = mode
        else:
            m.presence_on = False
    elif name == "gates":
        m.gate_override = np.ones((m.N, m.d)) if mode == "identity" else model.gate_mean
    elif name == "last_direct_ray":
        m.params["U"][-1] = 0.0
    elif name in ("direct_rays", "reflections", "readout"):
        for p in MODULES[name][0]:
            m.params[p][...] = 0.0 if mode == "zero" else m.params[p].mean()
    else:
        raise KeyError(name)
    return m


MUTANTS = {
    "sign_flip_update": "optimizer steps up the loss gradient (must fail C3)",
    "zero_lr": "learning rate forced to zero (must fail C3)",
    "zero_grad_R": "analytic gradient of the reflections zeroed (must fail C1)",
    "zero_grad_G": "analytic gradient of the gates zeroed (must fail C1)",
    "ledger_misattribution": "path masks shifted by one unit (must fail C6)",
}


# ---------------------------------------------------------------- training
def shading_mask(rng, B, n_units):
    """With probability SHADE_P an example loses one random unit and every path through it."""
    keep = np.ones((B, n_units))
    hit = np.nonzero(rng.random(B) < SHADE_P)[0]
    keep[hit, rng.integers(0, n_units, len(hit))] = 0.0
    return keep


def _train_loop(params, lossfn, n_units, X, y, steps, rng, shade=True):
    opt, hist = Adam(params, lr=0.0 if MUTANT == "zero_lr" else LR), []
    for _ in range(steps):
        idx = rng.integers(0, len(X), BATCH)
        batch = {"X": X[idx], "y": y[idx]}
        if shade:
            batch["keep"] = shading_mask(rng, BATCH, n_units)
        loss, grads = lossfn(batch)
        if not np.isfinite(loss):
            raise NonFinite("training loss")
        grads, _ = clip_global_norm(grads, CLIP)
        if MUTANT == "sign_flip_update":
            grads = {k: -g for k, g in grads.items()}
        opt.step(params, grads)
        hist.append(loss)
    return np.array(hist)


def fit(model, data, budget, rng):
    """Gradient training with shading, then the non-gradient beholding fit on clean train/validation."""
    hist = _train_loop(model.params, lambda b: loss_and_grads(model, b), model.N,
                       data["X"], data["y"], budget, rng)
    model.gate_mean = gates_and_rays(model, data["X"])[0].mean(axis=1)
    if "Xval" in data:
        presence_fit(model, data["X"], data["Xval"])
    return hist


def fit_dense(net, X, y, steps, rng, shade=True):
    return _train_loop(net.params, lambda b: dense_loss_and_grads(net, b), net.n_groups,
                       X, y, steps, rng, shade=shade and net.n_groups is not None)


def full_loss(model, X, y):
    if isinstance(model, Dense):
        return dense_loss_and_grads(model, {"X": X, "y": y})[0]
    return loss_and_grads(model, {"X": X, "y": y})[0]


# ---------------------------------------------------------------- evaluation of one seed
def acc(logits, y):
    return float(np.mean(logits.argmax(1) == y))


def train_seed(seed, cfg):
    r = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(5)]
    data = make_data(r[0], cfg)
    lat = build_model(IN_DIM, N_CLASSES, "vector_classification", r[1])
    loss0 = full_loss(lat, data["Xtr"], data["ytr"])
    fit(lat, {"X": data["Xtr"], "y": data["ytr"], "Xval": data["Xva"]}, cfg["steps"], r[1])
    loss1 = full_loss(lat, data["Xtr"], data["ytr"])
    mlp = Dense([IN_DIM, MLP_HIDDEN, MLP_HIDDEN, N_CLASSES], r[2], n_groups=N_LIGHTS)
    fit_dense(mlp, data["Xtr"], data["ytr"], cfg["steps"], r[2])
    stds = [float(h.std()) for h in unit_states(lat, data["Xtr"])]
    rival = train_rival(lat, data, r[3], cfg["rival_steps"], stds)
    return {"data": data, "lat": lat, "mlp": mlp, "rival": rival, "stds": stds,
            "mstds": [float(h.std()) for h in unit_states(mlp, data["Xtr"])],
            "loss0": loss0, "loss1": loss1, "frng": r[4]}


def shaded_eval(S, knock=None):
    """Mean accuracy over NOVEL_FAULTS x units, with the same fault draws for every variant."""
    data, lat, mlp, rival = S["data"], S["lat"], S["mlp"], S["rival"]
    X, y = data["Xte"], data["yte"]
    out = {k: [] for k in ("model", "barrier", "mlp", "rival", "loc_model", "loc_rival")}
    if knock:
        out.update({name: [] for name in knock})
    frng = np.random.default_rng(S["frng"].integers(1 << 31))
    for kind in NOVEL_FAULTS:
        for k in range(lat.N):
            fl = {k: make_fault(kind, lat.d, S["stds"][k], frng, len(X))}
            fm = {k: make_fault(kind, mlp.gsize, S["mstds"][k], frng, len(X))}
            zm, info = mind_readout(lat, X, fl)
            out["model"].append(acc(zm, y))
            out["loc_model"].append(float(np.mean(info["trig"] & (info["k"] == k))))
            out["barrier"].append(acc(info["z"], y))
            out["mlp"].append(acc(dense_forward(mlp, X, faults=fm)[0], y))
            ch = rival_choice(lat, rival, X, fl)
            out["rival"].append(acc(mind_readout(lat, X, choice=ch, cache=info["cache"])[0], y))
            out["loc_rival"].append(float(np.mean(ch[0] & (ch[1] == k))))
            for name, km in (knock or {}).items():
                out[name].append(acc(mind_readout(km, X, fl)[0], y))
    return {k: float(np.mean(v)) for k, v in out.items()}


KNOCKS = (("beholding", "identity"), ("beholding", "single"), ("beholding", "mean"), ("reflections", "zero"),
          ("gates", "mean"), ("last_direct_ray", "zero"), ("readout", "zero"))


def evaluate_seed(S, with_knockouts=True):
    data, lat, mlp = S["data"], S["lat"], S["mlp"]
    X, y = data["Xte"], data["yte"]
    knock = {f"{n}:{m}": knockout(lat, n, m) for n, m in KNOCKS} if with_knockouts else None
    res = {"shaded": shaded_eval(S, knock)}
    zc, ic = mind_readout(lat, X)
    res["clean"] = {"model": acc(zc, y), "barrier": acc(lattice_forward(lat, X)[0], y),
                    "mlp": acc(dense_forward(mlp, X)[0], y), "false_trigger": float(np.mean(ic["trig"]))}
    res["bright"] = {"model": acc(mind_readout(lat, data["Xbr"])[0], data["ybr"]),
                     "mlp": acc(dense_forward(mlp, data["Xbr"])[0], data["ybr"])}
    frng = np.random.default_rng(S["frng"].integers(1 << 31))
    fl = diffuse_veil(lat.N, lat.d, S["stds"], frng, len(X))
    fm = diffuse_veil(mlp.n_groups, mlp.gsize, S["mstds"], frng, len(X))
    zd, idd = mind_readout(lat, X, fl)
    res["diffuse"] = {"model": acc(zd, y), "barrier": acc(idd["z"], y),
                      "mlp": acc(dense_forward(mlp, X, faults=fm)[0], y), "trigger": float(np.mean(idd["trig"]))}
    cf = {"model": [], "barrier": [], "loc": []}
    hs = lattice_forward(lat, X)[1][0]
    for k in range(lat.N):
        f = {k: make_fault("counterfeit", lat.d, 0, frng, len(X), foreign=hs[k][frng.permutation(len(X))])}
        zc2, ic2 = mind_readout(lat, X, f)
        cf["model"].append(acc(zc2, y))
        cf["barrier"].append(acc(ic2["z"], y))
        cf["loc"].append(float(np.mean(ic2["trig"] & (ic2["k"] == k))))
    res["counterfeit"] = {k: float(np.mean(v)) for k, v in cf.items()}
    return res


# ---------------------------------------------------------------- correctness tests
def c1_gradcheck(rng):
    """Every tensor of the lattice, the MLP baseline and the rival, at init and after 60 steps."""
    X, y = lit_scene(400, rng)
    Fr, yr = rng.normal(size=(400, N_LIGHTS * WIDTH)), rng.integers(0, N_LIGHTS + 1, 400)
    keep = np.ones((8, N_LIGHTS))
    keep[[1, 5], [2, 4]] = 0.0
    nets = [("lattice", build_model(IN_DIM, N_CLASSES, "vector_classification", rng), loss_and_grads, X, y, keep),
            ("mlp", Dense([IN_DIM, MLP_HIDDEN, MLP_HIDDEN, N_CLASSES], rng, N_LIGHTS), dense_loss_and_grads, X, y, keep),
            ("rival", Dense([N_LIGHTS * WIDTH, RIVAL_HIDDEN, N_LIGHTS + 1], rng), dense_loss_and_grads, Fr, yr, None)]
    worst, checked, total = 0.0, 0, 0
    for stage in ("init", "after_training_steps"):
        for name, m, fn, Xs, ys, kp in nets:
            batch = {"X": Xs[:8], "y": ys[:8]}
            if kp is not None:
                batch["keep"] = kp
            _, g = fn(m, batch)
            errs = finite_difference_check(lambda: fn(m, batch)[0], m.params, g, rng)
            worst = max(worst, max(errs.values()))
            checked += sum(e <= MIND_CARD["thresholds"]["gradcheck_rel"] for e in errs.values())
            total += len(errs)
        if stage == "init":
            for name, m, fn, Xs, ys, kp in nets:
                _train_loop(m.params, lambda b, m=m, fn=fn: fn(m, b), N_LIGHTS if kp is not None else None,
                            Xs, ys, 60, rng, shade=kp is not None)
    return {"tensors_checked": int(checked), "tensors_total": int(total), "max_rel_error": float(worst),
            "checked_at": ["init", "after_training_steps"], "passed": bool(checked == total)}


def c2_determinism():
    outs = []
    for _ in range(2):
        rng = np.random.default_rng(1234)
        X, y = lit_scene(600, rng)
        m = build_model(IN_DIM, N_CLASSES, "vector_classification", rng)
        h = fit(m, {"X": X[:400], "y": y[:400], "Xval": X[400:]}, 80, rng)
        outs.append((h, mind_readout(m, X)[0], m))
    same = np.array_equal(outs[0][0], outs[1][0]) and np.array_equal(outs[0][1], outs[1][1])
    finite = all(np.all(np.isfinite(v)) for v in outs[0][2].params.values()) and np.all(np.isfinite(outs[0][1]))
    return bool(same and finite), f"identical={same} finite={finite}"


def random_faults(rng, d, N, n_faults, B):
    ks = rng.choice(N, size=n_faults, replace=False)
    return {int(k): (rng.normal(size=(d, d)), rng.normal(size=(B, d)) if rng.random() < 0.5 else None) for k in ks}


def c6_properties(rng):
    """(a) exact additivity, (b) fault isolation of the complement, (c) least-squares beholding, (d) path count."""
    th = MIND_CARD["thresholds"]
    ea = eb = ec = 0.0
    ctrl_a = ctrl_b = ctrl_c = np.inf
    for draw in range(25):
        m = build_model(IN_DIM, N_CLASSES, "vector_classification", rng)
        for k in m.params:
            m.params[k] *= 3.0 if draw % 2 else 1.0
        X = rng.normal(size=(32, IN_DIM)) * rng.uniform(0.5, 3.0)
        f = random_faults(rng, m.d, m.N, 2, len(X))
        rows, masks = ledger(m, X, f)
        z = lattice_forward(m, X, faults=f)[0]
        scale = 1.0 + np.abs(z).max()
        D = increments(rows, masks)
        err = np.abs(rows.sum(0) + m.params["c"] - z).max()
        for k in range(m.N):
            keep = np.ones((len(X), m.N))
            keep[:, k] = 0.0
            err = max(err, np.abs(z - D[k] - lattice_forward(m, X, keep=keep, faults=f)[0]).max())
        ea = max(ea, err / scale)
        ctrl_a = min(ctrl_a, np.abs(rows.sum(0) + m.params["c"] - lattice_forward(
            m, X, faults=f, value_nonlin=True)[0]).max() / scale)
        k = int(rng.integers(m.N))
        fk = {k: (rng.normal(size=(m.d, m.d)), rng.normal(size=(len(X), m.d)))}
        r0, m0 = ledger(m, X)
        r1, m1 = ledger(m, X, fk)
        comp = lambda r, mk: r[~mk[:, k]].sum(0)
        eb = max(eb, np.abs(comp(r0, m0) - comp(r1, m1)).max() / scale)
        ctrl_b = min(ctrl_b, np.abs(comp(r0, np.roll(m0, 1, axis=1)) - comp(r1, np.roll(m1, 1, axis=1))).max() / scale)
        presence_fit(m, X, X)
        _, _, _, D2, Phi = presence_scores(m, *ledger(m, X))
        for j in range(m.N):
            W = m.presence["W"][j]
            normal = Phi[j].T @ (D2[j] - Phi[j] @ W) - m.presence["ridge"] * W
            ref = 1.0 + np.abs(Phi[j].T @ D2[j]).max()
            ec = max(ec, np.abs(normal).max() / ref)
            bad = Phi[j].T @ (D2[j] - Phi[j] @ (W + 0.01)) - m.presence["ridge"] * (W + 0.01)
            ctrl_c = min(ctrl_c, np.abs(bad).max() / ref)
    counts_ok = [int((ledger(m, X)[1].sum(0))[n]) for n in range(m.N)]
    paths_ok = ledger(m, X)[0].shape[0] == 2 ** m.N - 1
    tol, viol = th["exactness_abs"], th["control_violation"]
    return [
        ("C6.a", "exact_additivity_and_complement", ea <= tol and ctrl_a > viol,
         f"max_err={ea:.2e} control(tanh value stream)={ctrl_a:.2e}"),
        ("C6.b", "complement_isolated_from_fault", eb <= tol and ctrl_b > viol,
         f"max_err={eb:.2e} control(shifted masks)={ctrl_b:.2e}"),
        ("C6.c", "beholding_least_squares_normal_equations", ec <= tol and ctrl_c > viol,
         f"max_err={ec:.2e} control(perturbed W)={ctrl_c:.2e}"),
        ("C6.d", "definition_check_path_count_2^N-1", paths_ok,
         f"paths={2 ** N_LIGHTS - 1} per-unit visits={counts_ok} (definition, not evidence)"),
    ]


def trivial_rate(ytr, yte):
    return float(np.mean(yte == np.bincount(ytr).argmax()))


def c3_c4(S, cfg):
    th, data = MIND_CARD["thresholds"], S["data"]
    triv = trivial_rate(data["ytr"], data["yte"])
    drop = (S["loss0"] - S["loss1"]) / S["loss0"]
    a = float(np.mean(predict(S["lat"], data["Xte"]) == data["yte"]))
    c3 = (drop >= th["loss_drop_fraction"] and a >= triv + th["margin_over_trivial"],
          f"loss_drop={drop:.3f} acc={a:.3f} trivial={triv:.3f}")
    rng = np.random.default_rng(99)
    sh = build_model(IN_DIM, N_CLASSES, "vector_classification", rng)
    fit(sh, {"X": data["Xtr"], "y": rng.permutation(data["ytr"]), "Xval": data["Xva"]}, cfg["steps"], rng)
    a4 = float(np.mean(predict(sh, data["Xte"]) == data["yte"]))
    rates = np.bincount(data["yte"], minlength=N_CLASSES) / len(data["yte"])
    lo, hi = rates.min() - th["shuffled_band"], rates.max() + th["shuffled_band"]
    return c3, (bool(lo <= a4 <= hi), f"shuffled_acc={a4:.3f} label-blind band=[{lo:.3f}, {hi:.3f}]")


def c5_mutants(cfg):
    global MUTANT
    detected = 0
    for name in MUTANTS:
        MUTANT = name
        try:
            rng = np.random.default_rng(7)
            if name.startswith("zero_grad"):
                caught = not c1_gradcheck(rng)["passed"]
            elif name == "ledger_misattribution":
                caught = not all(r[2] for r in c6_properties(rng)[:2])
            else:
                X, y = lit_scene(1200, rng)
                m = build_model(IN_DIM, N_CLASSES, "vector_classification", rng)
                l0 = full_loss(m, X, y)
                fit(m, {"X": X, "y": y}, min(cfg["steps"], 450), rng)
                caught = (l0 - full_loss(m, X, y)) / l0 < MIND_CARD["thresholds"]["loss_drop_fraction"]
        except NonFinite:
            caught = True
        detected += int(caught)
    MUTANT = None
    return {"detected": detected, "total": len(MUTANTS), "score": detected / len(MUTANTS)}


def c7_split_integrity(data):
    h = lambda X: {hashlib.sha1(np.round(r, 10).tobytes()).hexdigest() for r in X}
    tr = h(data["Xtr"])
    overlap = sum(len(tr & h(data[k])) for k in ("Xva", "Xte", "Xbr"))
    return bool(overlap == 0), f"train/eval overlap={overlap}"


# ---------------------------------------------------------------- hypotheses
HYP_METRIC = {
    "H-SIG": lambda r: r["shaded"]["model"] - r["shaded"]["mlp"],
    "H-NEC": lambda r: r["shaded"]["beholding:identity"] - r["shaded"]["last_direct_ray:zero"],
    "H-BLIND": lambda r: r["diffuse"]["model"] - r["diffuse"]["barrier"],
    "H-RIVAL": lambda r: r["shaded"]["model"] - r["shaded"]["rival"],
}


def hypotheses(results, rng, evaluated):
    out = []
    for h in MIND_CARD["hypotheses"]:
        diffs = [HYP_METRIC[h["id"]](r) for r in results]
        mean, lo, hi = paired_bootstrap(diffs, rng)
        v = verdict(mean, lo, hi, h["direction"], h["mesi"]) if evaluated else "not evaluated"
        out.append({"id": h["id"], "metric": h["metric"], "mean_diff": mean, "ci95": [lo, hi],
                    "mesi": h["mesi"], "n_seeds": len(diffs), "verdict": v})
    return out


def knockout_table(results, rng):
    rows = []
    for n, mode in KNOCKS:
        mean, lo, hi = paired_bootstrap([r["shaded"][f"{n}:{mode}"] - r["shaded"]["model"] for r in results], rng)
        rows.append({"module": f"{n} ({mode})", "signature": MODULES[n][2], "metric_change": mean, "ci95": [lo, hi]})
    return rows


# ---------------------------------------------------------------- report
def mean_of(results, *path):
    vals = []
    for r in results:
        v = r
        for p in path:
            v = v[p]
        vals.append(v)
    return float(np.mean(vals))


def print_report(rep, results):
    print("=== VERIFIED REPORT · chapter 0308 ===")
    print(f"environment  python {rep['environment']['python']}  numpy {rep['environment']['numpy']}")
    print(f"seeds {rep['seeds']}  runtime {rep['runtime_s']:.1f} s  params(lattice) {rep['n_params']}"
          f"  params(mlp baseline) {rep['n_params_baseline']}")
    g = rep["gradcheck"]
    print(f"gradcheck    {g['tensors_checked']}/{g['tensors_total']} tensors  max rel err {g['max_rel_error']:.2e}"
          f"  at {'+'.join(g['checked_at'])}  passed={g['passed']}")
    print("correctness")
    for c in rep["correctness"]:
        print(f"  {c['id']:<5} {c['name']:<42} {'PASS' if c['passed'] else 'FAIL'}  {c['detail']}")
    mu = rep["mutants"]
    print(f"mutation score {mu['detected']}/{mu['total']} = {mu['score']:.2f}")
    print("accuracy (mean over seeds)          model  barrier  mlp    rival")
    for cond in ("clean", "shaded", "diffuse"):
        r = [mean_of(results, cond, k) if k in results[0][cond] else float("nan")
             for k in ("model", "barrier", "mlp", "rival")]
        print(f"  {cond:<34} {r[0]:.3f}  {r[1]:.3f}    {r[2]:.3f}  {r[3]:.3f}")
    print(f"  bright dawn (gain 1.6-2.4)         {mean_of(results, 'bright', 'model'):.3f}            "
          f"{mean_of(results, 'bright', 'mlp'):.3f}")
    print(f"  counterfeit ray (descriptive)      {mean_of(results, 'counterfeit', 'model'):.3f}  "
          f"{mean_of(results, 'counterfeit', 'barrier'):.3f}")
    print(f"localization  presence {mean_of(results, 'shaded', 'loc_model'):.3f}  rival "
          f"{mean_of(results, 'shaded', 'loc_rival'):.3f}  counterfeit {mean_of(results, 'counterfeit', 'loc'):.3f}"
          f"  clean false-trigger {mean_of(results, 'clean', 'false_trigger'):.3f}"
          f"  diffuse trigger {mean_of(results, 'diffuse', 'trigger'):.3f}")
    print("hypotheses")
    for h in rep["hypotheses"]:
        print(f"  {h['id']:<8} {h['metric']:<16} diff {h['mean_diff']:+.4f}  ci95 [{h['ci95'][0]:+.4f}, "
              f"{h['ci95'][1]:+.4f}]  mesi {h['mesi']}  n={h['n_seeds']}  {h['verdict']}")
    print("knockouts (shaded accuracy change vs full model)")
    for k in rep["knockouts"]:
        print(f"  {k['module']:<28} signature={str(k['signature']):<5} {k['metric_change']:+.4f}  "
              f"ci95 [{k['ci95'][0]:+.4f}, {k['ci95'][1]:+.4f}]")
    print(f"card revision {rep['card_revision']}: " + " ".join(MIND_CARD["revision_log"]))
    print(f"task types {rep['task_types']}  exit code {rep['exit_code']}")
    print("=== END REPORT ===")


# ---------------------------------------------------------------- real-data bridge
def data_bridge(path, seed):
    if not os.path.exists(path):
        print(f"--data: {path} not found; real-data bridge skipped.")
        return
    lines = [ln.strip().split(",") for ln in open(path, encoding="utf-8") if ln.strip()]
    try:
        [float(v) for v in lines[0]]
    except ValueError:
        lines = lines[1:]
    arr = np.array(lines)
    X = arr[:, :-1].astype(float)
    y = np.unique(arr[:, -1], return_inverse=True)[1]
    rng = np.random.default_rng(seed)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    idx = rng.permutation(len(X))
    a, b = int(0.7 * len(X)), int(0.85 * len(X))
    m = build_model(X.shape[1], int(y.max()) + 1, "vector_classification", rng)
    fit(m, {"X": X[idx[:a]], "y": y[idx[:a]], "Xval": X[idx[a:b]]}, FULL["steps"], rng)
    Xt, yt = X[idx[b:]], y[idx[b:]]
    stds = [float(h.std()) for h in unit_states(m, X[idx[:a]])]
    res = []
    for k in range(m.N):
        f = {k: make_fault("sign", m.d, stds[k], rng, len(Xt))}
        zf, inf = mind_readout(m, Xt, f)
        res.append((acc(zf, yt), acc(inf["z"], yt)))
    print(f"--data {os.path.basename(path)}: clean {acc(mind_readout(m, Xt)[0], yt):.3f}; sign-fault mean "
          f"model {np.mean([r[0] for r in res]):.3f} barrier {np.mean([r[1] for r in res]):.3f}")


# ---------------------------------------------------------------- entry point
def run(args):
    global MUTANT
    t0 = time.time()
    quick = args.quick or args.mutant is not None
    cfg = QUICK if quick else FULL
    n_seeds = 1 if quick else args.seeds
    seeds = [args.seed + i for i in range(n_seeds)]
    print(f"{os.path.basename(__file__)}: seeds {seeds} ({'quick' if quick else 'full'} protocol)")
    MUTANT = args.mutant
    rng = np.random.default_rng(args.seed)
    grad = c1_gradcheck(np.random.default_rng(np.random.SeedSequence(args.seed).spawn(1)[0]))
    corr = [("C1", "gradient_check", grad["passed"], f"max_rel={grad['max_rel_error']:.2e}")]
    ok2, d2 = c2_determinism()
    corr.append(("C2", "determinism_and_finiteness", ok2, d2))
    corr += c6_properties(np.random.default_rng(args.seed + 11))
    seeds_S = [train_seed(s, cfg) for s in seeds]
    c3, c4 = c3_c4(seeds_S[0], cfg)
    corr.insert(2, ("C3", "learning", c3[0], c3[1]))
    corr.insert(3, ("C4", "shuffled_label_control", c4[0], c4[1]))
    ok7 = all(c7_split_integrity(S["data"])[0] for S in seeds_S)
    corr.append(("C7", "split_integrity", ok7, c7_split_integrity(seeds_S[0]["data"])[1]))
    if MUTANT is not None:
        failed = [c[0] for c in corr if not c[2]]
        print(f"mutant {MUTANT}: failing correctness tests {failed}")
        return 1 if failed else 0
    mut = c5_mutants(cfg)
    corr.insert(4, ("C5", "mutant_detection", mut["score"] == 1.0, f"{mut['detected']}/{mut['total']}"))
    results = [evaluate_seed(S) for S in seeds_S]
    runtime = time.time() - t0
    budget = 20.0 if quick else 180.0
    corr.append(("C8", "runtime_budget", runtime <= budget, f"{runtime:.1f}s <= {budget:.0f}s"))
    hyps = hypotheses(results, rng, evaluated=not quick and n_seeds >= 5)
    code = 0 if all(c[2] for c in corr) else 1
    if code == 0 and runtime > budget:
        code = 3
    rep = {"schema_version": "1.0", "chapter": 308, "file": os.path.basename(__file__), "card_revision":
           MIND_CARD["card_revision"], "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
           "seeds": seeds, "runtime_s": runtime, "n_params": n_params(seeds_S[0]["lat"]),
           "n_params_baseline": n_params(seeds_S[0]["mlp"]), "gradcheck": grad,
           "correctness": [{"id": c[0], "name": c[1], "passed": bool(c[2]), "detail": c[3]} for c in corr],
           "mutants": mut, "hypotheses": hyps, "knockouts": knockout_table(results, rng),
           "task_types": TASK_TYPES, "exit_code": code}
    print_report(rep, results)
    if args.json:
        write_report(rep, args.json)
    if args.data:
        data_bridge(args.data, args.seed)
    return code


def main():
    ap = argparse.ArgumentParser(description="Chapter 0308 · Suhrawardi · Ledger of Rays")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--card", action="store_true")
    ap.add_argument("--mutant", type=str, default=None)
    ap.add_argument("--data", type=str, default=None)
    try:
        args = ap.parse_args()
    except SystemExit as e:
        sys.exit(0 if e.code == 0 else 2)
    if args.card:
        print(json.dumps(MIND_CARD, indent=2))
        sys.exit(0)
    if args.mutant is not None and args.mutant not in MUTANTS:
        print(f"unknown mutant; choose from {sorted(MUTANTS)}")
        sys.exit(2)
    if args.seeds < 1:
        sys.exit(2)
    try:
        code = run(args)
    except NonFinite as e:
        print(f"non-finite values: {e}")
        code = 4
    sys.exit(code)


if __name__ == "__main__":
    main()

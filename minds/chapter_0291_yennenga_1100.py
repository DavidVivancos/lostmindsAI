#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0291 · Yennenga
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 15, Minds 281-300 Available on Amazon https://www.amazon.com/dp/B0HMQ7TVJM
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0291_yennenga_1100 - Yennenga (legendary; traditional long chronology c. 1100 CE, scholarly short chronology c. 15th century)
# END ATTRIBUTION
"""
Chapter 0291 - Yennenga (legendary; traditional long chronology c. 1100 CE,
scholarly short chronology c. 15th century). Research prototype, not an AGI.

Thesis
    A message that cannot be spoken can be planted as a deviation in a living
    process, and it is legible only to a reader that first rolls the CUSTOM
    forward under the same weather and then reads the gap between the field it
    sees and the field the custom would have made.

Evidence and provenance (mediated: no words of her own survive)
    D1  Oral tradition, in its widely told Burkinabe form: refused marriage by
        her father Nedega, she sowed an okra field and let it spoil, then told
        him she was being left to wither like it (crop varies: okra, wheat,
        maize; one variant gives the analogy to her mother). Scholarship:
        Sheldon, Historical Dictionary of Women in Sub-Saharan Africa (2005).
    D2  Okra must be picked at least every other day; unpicked pods turn woody
        within days, so the custom is daily and its absence is conspicuous.
        Agronomy: University of Illinois Extension, "Okra" (Watch Your Garden
        Grow).
    D3  The sign failed: Nedega confined her; she fled on a stallion. The son
        born of her union with the hunter Riale was named Wedraogo, "stallion".
        Skinner, The Mossi of the Upper Volta (1964); Rupley, Bangali and
        Diamitani, Historical Dictionary of Burkina Faso (2013).
    D4  Riale is an elephant hunter (Mande or Bisa by version) who reads her
        through her disguise: a reader of traces in a world with no field
        custom. Sheldon (2005); Izard, Moogo (2003) on variant traditions.
    D5  Chronology contested: court tradition (Tiendrebeogo 1963) against
        Izard's short chronology (first Mossi commands c. 1500).
    D6  Interpretation (speculation): meaning-by-flouted-custom, i.e. an
        implicature enacted in a crop (cf. Grice 1975).

Doctrine -> mechanism -> test (IDs as in MIND_CARD)
    D1,D6 -> M1 custom simulator: learned grey-box rollout of the customary
             picking policy under the observed rain and sowing -> H-SIG, H-NEC
    D2    -> M1 structure: picking is a smooth threshold on ripeness; losses are
             multiplicative survival factors -> C6.1 positivity, C6.2 dominance
    D1,D6 -> M2 norm residual: observed minus counterfactual trajectory is the
             only view of the field the trace reader receives -> H-NEC
    D3,D4 -> blind spot: a reader whose custom differs from the sender's
             misreads the field -> H-BLIND (custom-drift split)

Research question (causal inference and intervention)
    Does subtracting a learned, weather-conditioned counterfactual of customary
    practice let a small reader separate deliberate withholding from drought,
    blight and neglect better than a size-matched reader of raw traces, most of
    all when the rain regime shifts?

Closest prior art and the delta
    Model-based fault detection and isolation by residual generation (Chen and
    Patton 1999; Isermann 2006); hybrid physics-ML models (Yin et al. 2021,
    APHYNITY); Bayesian inverse planning (Baker, Saxe and Tenenbaum 2009).
    Delta: the reference is a social CUSTOM (a picking policy) rolled forward
    under the observed exogenous drivers, not a nominal plant, and deviation is
    read as intent versus accident; the custom's parameters are learned jointly
    with the reader by exact backpropagation through the rollout.
    Baseline in file: size-matched raw-trace reader (same encoder, same budget).

Blind spot
    Meaning by deviation presupposes a shared custom. On the custom-drift split
    a village picks younger; the reader's learned custom is then wrong and
    every field looks like early cutting. Grounding: her okra sign needed a
    reader who held and attended to her custom, and the tradition records it
    going unread.

Task (generative process, per field, T = 24 two-day steps)
    rain w_t: AR(1) around a regime mean; sowing offset s; hidden fertility.
    ripeness m_t accumulates with rain after sowing; standing pods y_t grow
    with rain and care; policy by intent: 0 custom picking at ripeness,
    1 withheld (the message), 2 neglect (weak care, late picking),
    3 early cutting. Losses: over-ripe decay, drought wilt, random blight.
    Observed per step: standing pods, visible decay, rain, sown flag (noisy).
    Splits: train/val/test (wet regime), shifted (drought regime),
    blind (custom drift: the whole village picks younger).

Limits
    Synthetic fields; one crop; four intents; the reader's custom family is
    correctly specified up to its parameters. Nothing here is evidence about
    the historical person; the legend itself is mediated tradition.

Run: python3 <this file>   (see --help for options)
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 2,
    "revision_log": [
        "r1: task generator calibrated for learnability (C3) and runtime before any hypothesis split was evaluated.",
        "r2 (after the first --quick run): C4 metric changed from shuffled-label accuracy to held-out NLL averaged over 3 shuffles. Accuracy of a noise-trained reader on well-separated intent clusters is a lottery over cluster-to-label maps (0.12-0.32 in 10 probe shuffles) and could fail spuriously either way; NLL stayed within log(4) - 0.036. H-SIG, H-NEC, H-BLIND unchanged."
    ],
    "generation": {"template_version": "codeguidelines 1.0 (15 Sep 2026)", "generator": "Claude",
                   "generator_version": "Claude Opus 5.5", "date": "2026-09-23"},
    "id": 291, "figure": "Yennenga", "born": 1100, "died": None,
    "civilization": "Mamprusi-Dagomba (Gbewaa kingdoms of northern Ghana); ancestral mother of the Mossi (legendary)",
    "provenance": "mediated",
    "thesis": "Meaning planted as a deviation in a living process is read only against a simulated custom under the same weather: read the gap, not the field.",
    "evidence": [
        {"id": "D1", "claim": "Refused marriage, she sowed a field (okra in the Burkinabe version) and let it spoil, then named the analogy to her father.",
         "basis": "scholarship", "source": "Sheldon 2005, Historical Dictionary of Women in Sub-Saharan Africa, s.v. Yennenga (oral tradition)"},
        {"id": "D2", "claim": "Okra must be picked at least every other day; unpicked pods turn woody quickly.",
         "basis": "scholarship", "source": "University of Illinois Extension, Watch Your Garden Grow: Okra"},
        {"id": "D3", "claim": "The sign failed; she fled on a stallion; her son was named Wedraogo, 'stallion'.",
         "basis": "scholarship", "source": "Skinner 1964, The Mossi of the Upper Volta; Rupley et al. 2013"},
        {"id": "D4", "claim": "Riale, an elephant hunter (Mande or Bisa by version), read her through her disguise.",
         "basis": "scholarship", "source": "Sheldon 2005; Izard 2003, Moogo"},
        {"id": "D5", "claim": "Chronology contested: long court chronology vs short chronology placing first Mossi commands c. 1500.",
         "basis": "scholarship", "source": "Tiendrebeogo 1963, J. Soc. Africanistes 33:7-46; Izard 1970; Izard 2003"},
        {"id": "D6", "claim": "The field works as an implicature: a flouted custom carries the message.",
         "basis": "speculation", "source": "Interpretation; cf. Grice 1975, Logic and Conversation"},
    ],
    "research_question": {"category": "causal inference and intervention",
                          "question": "Does subtracting a learned weather-conditioned counterfactual of customary practice separate deliberate withholding from drought, blight and neglect better than a size-matched raw-trace reader under a rain-regime shift?"},
    "mechanism": {
        "name": "Unpicked-field reader (custom-counterfactual residual)",
        "family": "grey-box counterfactual rollout + residual temporal classifier",
        "signature_modules": ["custom_simulator"],
        "closest_prior_art": ["model-based fault detection by residual generation (Chen & Patton 1999; Isermann 2006)",
                              "hybrid physics-ML dynamics (Yin et al. 2021, APHYNITY)",
                              "Bayesian inverse planning (Baker, Saxe & Tenenbaum 2009)"],
        "overlap": "Medium",
        "prior_art_queries": ["residual generation fault diagnosis learned model", "counterfactual baseline intent inference time series",
                              "crop abandonment detection drought confound", "norm violation detection trajectories"],
        "contribution_type": "mechanism",
        "delta": "The reference trajectory is a customary policy rolled forward under the observed exogenous drivers and learned jointly with the reader, so deviation is read as intent rather than as a plant fault.",
    },
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1", "property_test": "C6.2", "hypothesis": "H-SIG"},
        {"doctrine": "D2", "mechanism": "M1", "property_test": "C6.1", "hypothesis": "H-NEC"},
        {"doctrine": "D6", "mechanism": "M2", "property_test": "C6.2", "hypothesis": "H-NEC"},
        {"doctrine": "D3", "mechanism": "M1", "property_test": "none", "hypothesis": "H-BLIND"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "The custom-residual reader beats the size-matched raw reader when the rain regime shifts to drought.",
         "metric": "accuracy", "split": "shifted", "comparison": "model - baseline", "direction": "greater", "mesi": 0.03, "seeds": 5},
        {"id": "H-NEC", "statement": "Knocking out the custom simulator (mean trajectory) hurts more than knocking out the rain/sown input stream (mean).",
         "metric": "accuracy", "split": "shifted", "comparison": "signature_knockout - matched_knockout", "direction": "less", "mesi": 0.03, "seeds": 5},
        {"id": "H-BLIND", "statement": "When the village's custom differs from the reader's, the custom-residual reader falls below the raw reader.",
         "condition": "custom drift: picking threshold 0.75 instead of 1.0 for the whole village",
         "grounding": "the okra sign required a reader holding and attending to her custom; Nedega did not read it",
         "metric": "accuracy", "comparison": "model - baseline", "direction": "less", "mesi": 0.02, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.5, "margin_over_trivial": 0.30, "shuffle_band": 0.08, "shuffle_nll_band": 0.05, "shuffle_repeats": 3,
                   "gradcheck_rel_tol": 1e-5, "grad_clip_norm": 5.0},
    "probe_predictions": [{"probe": "P7", "expected": "above baseline"}, {"probe": "P9", "expected": "equal to baseline"},
                          {"probe": "P4", "expected": "equal to baseline"}, {"probe": "P1", "expected": "below baseline"}],
    "dialectic_links": [],
    "corpus_neighbors": [
        {"chapter": 212, "similarity": None, "difference": "Al-Jahiz weighs UNINTENDED signs; here the sign is intended and exists only relative to a simulated custom."},
        {"chapter": 232, "similarity": None, "difference": "Harald's hair is a bodily accumulator/commitment device; here an accumulation is decoded against a counterfactual policy."},
        {"chapter": 253, "similarity": None, "difference": "Al-Ma'arri prices harm regardless of complaint; here the complaint is encoded in a process and must be decoded."},
        {"chapter": 325, "similarity": None, "difference": "Manco Capac lets an external assay choose the ground; the horse motif is kept out of the mechanism."},
        {"chapter": 334, "similarity": None, "difference": "Sundiata's convened faculty is licensing by witnesses; no counterfactual rollout."},
        {"chapter": 507, "similarity": 0.032, "difference": "Philip II attenuates lines by margins; no dynamics, no exogenous drivers. Only corpus file available at build time; other neighbours not re-scored."},
    ],
    "barometer": {
        "cognitive_processing": ["counterfactual simulation as a working-memory scaffold"],
        "embodied_cognition": [],
        "world_modeling": ["rolls a growth/decay model forward under observed rain", "drought-shift robustness"],
        "consciousness": [],
        "language_understanding": ["implicature: meaning as a flouted norm in a non-verbal medium"],
        "emotional_intelligence": ["intent attribution: deliberate withholding vs neglect vs accident"],
        "creativity": [],
        "autonomy": [],
    },
    "task_types": ["sequence_classification"],
    "applications": [
        {"use": "separate unharvested or abandoned fields from drought damage in crop time series with rainfall covariates",
         "sector": "agriculture and food security", "dataset": "CropHarvest (Tseng et al. 2021) with CHIRPS rainfall"},
        {"use": "separate operator-initiated setpoint changes from process faults",
         "sector": "industrial process monitoring", "dataset": "Tennessee Eastman Process benchmark (Downs & Vogel 1993; Rieth et al. 2017)"},
        {"use": "separate price-responsive demand from weather-driven load in aggregated smart-meter data",
         "sector": "energy", "dataset": "Low Carbon London smart-meter trial (UK Power Networks, 2014)"},
    ],
    "safety_notes": "Synthetic data only. Not for monitoring individuals' routines or inferring private intent from household traces; applications use aggregated or industrial data.",
}

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

T_STEPS = 24
N_CLASSES = 4
CLASS_NAMES = ["custom", "withheld", "neglect", "early_cut"]
KAPPA, BETA, CQ, PMAX = 10.0, 8.0, 1.15, 0.9
TRUE_DYN = {"gamma": 0.30, "mu": 0.10, "rho": 0.45, "delta": 0.35, "theta_c": 1.0}
F_FILTERS, KSIZE = 12, 5
L2, LAMBDA_FIT, LR = 1e-4, 1.0, 1e-2
FULL_STEPS, QUICK_STEPS, BATCH = 600, 250, 200
BUDGET_S, QUICK_BUDGET_S = 180.0, 20.0
MUTANT = {"name": None}

# BEGIN STANDARD UTILITIES v1.0
def sigmoid(z):
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def softplus(z):
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))


def logsumexp(z, axis=-1):
    m = np.max(z, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(z - m), axis=axis, keepdims=True))).squeeze(axis)


def softmax(z, axis=-1):
    e = np.exp(z - np.max(z, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


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
    total = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    if total > max_norm:
        for k in grads:
            grads[k] = grads[k] * (max_norm / (total + 1e-12))
    return total


def finite_difference_check(loss_fn, params, grads, rng, eps=1e-6, n_entries=20):
    """Central differences on >= n_entries random entries per tensor plus the largest-gradient entry."""
    worst, per_tensor = 0.0, {}
    for name, p in params.items():
        flat = p.reshape(-1)
        g = grads[name].reshape(-1)
        idx = set(rng.choice(flat.size, size=min(n_entries, flat.size), replace=False).tolist())
        idx.add(int(np.argmax(np.abs(g))))
        err = 0.0
        for i in idx:
            old = flat[i]
            flat[i] = old + eps
            lp = loss_fn(params)
            flat[i] = old - eps
            lm = loss_fn(params)
            flat[i] = old
            num = (lp - lm) / (2 * eps)
            err = max(err, abs(num - g[i]) / max(1e-8, abs(num) + abs(g[i])))
        per_tensor[name] = err
        worst = max(worst, err)
    return worst, per_tensor


def paired_bootstrap_ci(diffs, rng, n_boot=2000, alpha=0.05):
    diffs = np.asarray(diffs, dtype=np.float64)
    idx = rng.integers(0, diffs.size, size=(n_boot, diffs.size))
    means = diffs[idx].mean(axis=1)
    return float(diffs.mean()), [float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))]


def verdict(mean, ci, direction, mesi):
    sgn = 1.0 if direction == "greater" else -1.0
    lo, hi = sgn * ci[0], sgn * ci[1]
    if min(lo, hi) > 0 and sgn * mean >= mesi:
        return "supported"
    if max(lo, hi) < 0:
        return "contradicted"
    return "inconclusive"


def write_report(lines, payload, json_path):
    print("\n".join(lines))
    if json_path:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
# END STANDARD UTILITIES


# ------------------------------------------------------------------ data and tasks
def rain_series(rng, n, t, lo, hi):
    """AR(1) rain around a per-field regime mean, clipped to [0, 1]."""
    base = rng.uniform(lo, hi, size=(n, 1))
    eps = np.zeros((n, t))
    for k in range(1, t):
        eps[:, k] = 0.6 * eps[:, k - 1] + rng.normal(0.0, 0.18, size=n)
    return np.clip(base + eps, 0.0, 1.0)


def policy_table(custom_theta):
    """(care, picking fraction, picking threshold) per intent, relative to the village custom."""
    return {0: (1.00, PMAX, custom_theta), 1: (1.00, 0.0, custom_theta),
            2: (0.45, 0.55, custom_theta + 0.30), 3: (1.00, PMAX, custom_theta - 0.38)}


def simulate_fields(rng, n, regime="wet", custom_theta=1.0):
    """Ground-truth field generator; returns X (n,T,4) = [standing, decay, rain, sown] and labels."""
    lo, hi = (0.45, 0.85) if regime == "wet" else (0.12, 0.38)
    w = rain_series(rng, n, T_STEPS, lo, hi)
    sow = rng.integers(0, 6, size=n)
    a = (np.arange(T_STEPS)[None, :] >= sow[:, None]).astype(np.float64)
    fert = rng.uniform(0.75, 1.25, size=n)
    lab = rng.permutation(np.arange(n) % N_CLASSES)
    blight = np.zeros((n, T_STEPS))
    hit = rng.random(n) < 0.35
    tb = np.clip(sow + rng.integers(3, 16, size=n), 0, T_STEPS - 2)
    blight[hit, tb[hit]] = 0.5
    blight[hit, tb[hit] + 1] = 0.5
    care = np.zeros(n); pmax = np.zeros(n); thr = np.zeros(n)
    for c, (cr, pm, th) in policy_table(custom_theta).items():
        care[lab == c], pmax[lab == c], thr[lab == c] = cr, pm, th
    m = TRUE_DYN["mu"] * np.concatenate([np.zeros((n, 1)), np.cumsum((0.6 + 0.4 * w) * a, axis=1)[:, :-1]], axis=1)
    p = pmax[:, None] * sigmoid(KAPPA * (m - thr[:, None]))
    q = TRUE_DYN["rho"] * sigmoid(BETA * (m - CQ))
    d = TRUE_DYN["delta"] * (1.0 - w) ** 2
    surv = (1 - p) * (1 - q) * (1 - d) * (1 - blight)
    g = TRUE_DYN["gamma"] * fert[:, None] * care[:, None] * w * a
    y = np.zeros((n, T_STEPS))
    for t in range(T_STEPS - 1):
        y[:, t + 1] = surv[:, t] * y[:, t] + g[:, t]
    decay = ((1 - p) - surv) * y
    X = np.stack([y + rng.normal(0, 0.05, y.shape), decay + rng.normal(0, 0.02, y.shape), w, a], axis=2)
    return X, lab


def make_splits(rng):
    X, y = simulate_fields(rng, 1000, "wet")
    return {"train": (X[:800], y[:800]), "val": (X[800:], y[800:]),
            "test": simulate_fields(rng, 400, "wet"), "shifted": simulate_fields(rng, 400, "drought"),
            "blind": simulate_fields(rng, 400, "wet", custom_theta=0.75)}


# ------------------------------------------------------------------ model
def unpack_custom(theta):
    return (math.exp(theta[0]), math.exp(theta[1]), float(sigmoid(np.array([theta[2]]))[0]),
            float(sigmoid(np.array([theta[3]]))[0]), float(theta[4]))


def custom_rollout(theta, W, A, withhold=False, variant=None):
    """Counterfactual field under the learned custom, driven by the OBSERVED rain W and sowing A."""
    gam, mu, rho, dlt, thc = unpack_custom(theta)
    n, t_len = W.shape
    M = np.concatenate([np.zeros((n, 1)), np.cumsum((0.6 + 0.4 * W) * A, axis=1)[:, :-1]], axis=1)
    sp = sigmoid(KAPPA * (mu * M - thc))
    p = np.zeros_like(sp) if withhold else PMAX * sp
    sq = sigmoid(BETA * (mu * M - CQ))
    q, d, g = rho * sq, dlt * (1 - W) ** 2, gam * W * A
    S = (1 - p) * (1 - q) * (1 - d)
    yh = np.zeros((n, t_len))
    for t in range(t_len - 1):
        if variant == "additive":      # negative control for C6.1: losses subtracted, not survived
            yh[:, t + 1] = yh[:, t] * (1 - p[:, t] - 1.5 * q[:, t] - 1.5 * d[:, t] - 0.5) + g[:, t]
        elif variant == "crowding":    # negative control for C6.2: quadratic crowding rot
            u = np.clip((1 - p[:, t]) * yh[:, t], -50.0, 50.0)
            yh[:, t + 1] = u - 0.6 * u * u + g[:, t]
        else:
            yh[:, t + 1] = S[:, t] * yh[:, t] + g[:, t]
    rh = ((1 - p) - S) * yh
    cache = dict(M=M, sp=sp, sq=sq, p=p, q=q, d=d, S=S, g=g, W=W, A=A, yh=yh,
                 par=(gam, mu, rho, dlt, thc))
    return yh, rh, cache


def custom_rollout_backward(c, Dy, Dr):
    """Exact adjoint of the rollout: returns dL/dtheta (5,) given dL/dyhat and dL/drhat."""
    gam, mu, rho, dlt, thc = c["par"]
    p, q, d, S, yh = c["p"], c["q"], c["d"], c["S"], c["yh"]
    t_len = yh.shape[1]
    dS, dpd, dg = np.zeros_like(S), np.zeros_like(S), np.zeros_like(S)
    a_next = np.zeros(yh.shape[0])
    for t in range(t_len - 1, -1, -1):
        dS[:, t] = (a_next - Dr[:, t]) * yh[:, t]
        dpd[:, t] = -Dr[:, t] * yh[:, t]
        dg[:, t] = a_next
        a_next = Dy[:, t] + Dr[:, t] * ((1 - p[:, t]) - S[:, t]) + a_next * S[:, t]
    dp = dS * (-(1 - q) * (1 - d)) + dpd
    dq = dS * (-(1 - p) * (1 - d))
    dd = dS * (-(1 - p) * (1 - q))
    ds_p = PMAX * c["sp"] * (1 - c["sp"]) * KAPPA
    ds_q = rho * c["sq"] * (1 - c["sq"]) * BETA
    return np.array([np.sum(dg * c["g"]),
                     np.sum(dp * ds_p * c["M"] * mu) + np.sum(dq * ds_q * c["M"] * mu),
                     np.sum(dq * c["sq"]) * rho * (1 - rho),
                     np.sum(dd * (1 - c["W"]) ** 2) * dlt * (1 - dlt),
                     -np.sum(dp * ds_p)])


def build_model(in_dim, out_dim, task_type, rng, **cfg):
    if task_type not in TASK_TYPES:
        raise ValueError(task_type)
    variant = cfg.get("variant", "custom")
    fan = KSIZE * in_dim
    params = {"W1": rng.normal(0, 1 / math.sqrt(fan), (fan, F_FILTERS)), "b1": np.zeros(F_FILTERS),
              "W2": rng.normal(0, 1 / math.sqrt(2 * F_FILTERS), (2 * F_FILTERS, out_dim)), "b2": np.zeros(out_dim)}
    if variant == "custom":
        params["custom_theta"] = np.array([math.log(0.2), math.log(0.13), 0.8473, -1.3863, 0.8])
    return {"params": params, "variant": variant, "in_dim": in_dim, "out_dim": out_dim,
            "norm": np.ones(2), "ko": {}}


def encoder_input(model, X):
    """Builds the trace-reader input: norm residual (custom) or raw trace (baseline), plus drivers."""
    s = model["norm"]
    W, A = np.clip(X[:, :, 2], 0, 1), (X[:, :, 3] > 0.5).astype(np.float64)
    if model["variant"] == "custom":
        yh, rh, cache = custom_rollout(model["params"]["custom_theta"], W, A)
        mode = model["ko"].get("custom_simulator")
        if mode == "mean":
            yh, rh = np.broadcast_to(yh.mean(0), yh.shape), np.broadcast_to(rh.mean(0), rh.shape)
        elif mode == "zero":
            yh, rh = np.zeros_like(yh), np.zeros_like(rh)
        ch0, ch1 = (X[:, :, 0] - yh) / s[0], (X[:, :, 1] - rh) / s[1]
    else:
        cache, ch0, ch1 = None, X[:, :, 0] / s[0], X[:, :, 1] / s[1]
    drv = X[:, :, 2:] - 0.5
    if model["ko"].get("drivers_stream") == "mean":
        drv = np.broadcast_to(drv.mean(0), drv.shape)
    return np.concatenate([ch0[:, :, None], ch1[:, :, None], drv], axis=2), cache


def forward(model, X):
    P = model["params"]
    Z0, cache = encoder_input(model, X)
    n, t_len, c = Z0.shape
    tp = t_len - KSIZE + 1
    patches = np.concatenate([Z0[:, k:k + tp, :] for k in range(KSIZE)], axis=2)
    H = np.tanh(patches @ P["W1"] + P["b1"])
    if model["ko"].get("trace_encoder") == "zero":
        H = np.zeros_like(H)
    ramp = np.linspace(-1.0, 1.0, tp) * (0.0 if model["ko"].get("ramp_pool") == "zero" else 1.0)
    feats = np.concatenate([H.mean(1), (H * ramp[None, :, None]).mean(1)], axis=1)
    logits = feats @ P["W2"] + P["b2"]
    return logits, dict(Z0=Z0, patches=patches, H=H, ramp=ramp, feats=feats, rcache=cache)


def loss_and_grads(model, batch):
    X, y = batch
    P = model["params"]
    logits, c = forward(model, X)
    n = X.shape[0]
    probs = softmax(logits)
    ce = float(np.mean(logsumexp(logits) - logits[np.arange(n), y]))
    dlog = probs.copy(); dlog[np.arange(n), y] -= 1.0; dlog /= n
    g = {"W2": c["feats"].T @ dlog, "b2": dlog.sum(0)}
    dfe = dlog @ P["W2"].T
    tp = c["H"].shape[1]
    dH = dfe[:, None, :F_FILTERS] / tp + dfe[:, None, F_FILTERS:] * c["ramp"][None, :, None] / tp
    dZ = dH * (1 - c["H"] ** 2)
    g["W1"] = np.einsum("ntk,ntf->kf", c["patches"], dZ)
    g["b1"] = dZ.sum((0, 1))
    loss = ce + L2 * (np.sum(P["W1"] ** 2) + np.sum(P["W2"] ** 2))
    g["W1"] += 2 * L2 * P["W1"]; g["W2"] += 2 * L2 * P["W2"]
    if model["variant"] == "custom":
        dP = dZ @ P["W1"].T
        cin = c["Z0"].shape[2]
        dZ0 = np.zeros_like(c["Z0"])
        for k in range(KSIZE):
            dZ0[:, k:k + tp, :] += dP[:, :, k * cin:(k + 1) * cin]
        s, rc = model["norm"], c["rcache"]
        Dy, Dr = -dZ0[:, :, 0] / s[0], -dZ0[:, :, 1] / s[1]
        mask = (y == 0).astype(np.float64)
        if mask.sum() > 0:  # the reader learns the custom from ordinary fields
            ey = (X[:, :, 0] - rc["yh"]) / s[0]
            er = (X[:, :, 1] - ((1 - rc["p"]) - rc["S"]) * rc["yh"]) / s[1]
            den = mask.sum() * X.shape[1]
            loss += LAMBDA_FIT * float(np.sum(mask[:, None] * (ey ** 2 + er ** 2)) / den)
            Dy += -2 * LAMBDA_FIT * mask[:, None] * ey / (s[0] * den)
            Dr += -2 * LAMBDA_FIT * mask[:, None] * er / (s[1] * den)
        g["custom_theta"] = custom_rollout_backward(rc, Dy, Dr)
    if MUTANT["name"] == "zero_grad_W1":
        g["W1"] = np.zeros_like(g["W1"])
    if MUTANT["name"] == "zero_grad_custom_theta" and "custom_theta" in g:
        g["custom_theta"] = np.zeros_like(g["custom_theta"])
    return loss, g


def predict(model, X):
    return np.argmax(forward(model, X)[0], axis=1)


def hidden_states(model, X):
    logits, c = forward(model, X)
    return {"encoder_input": c["Z0"], "trace_features": c["H"], "pooled": c["feats"], "logits": logits}


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


# ------------------------------------------------------------------ registries
def modules(model):
    reg = {
        "drivers_stream": {"params": ["W1"], "role": "exogenous rain/sowing input pathway of the temporal encoder", "signature": False},
        "trace_encoder": {"params": ["W1", "b1"], "role": "temporal convolution with tanh", "signature": False},
        "ramp_pool": {"params": [], "role": "mean and time-ramp weighted pooling", "signature": False},
        "intent_head": {"params": ["W2", "b2"], "role": "linear softmax classifier", "signature": False},
    }
    if model["variant"] == "custom":
        reg["custom_simulator"] = {"params": ["custom_theta"], "signature": True,
                                   "role": "learned grey-box counterfactual rollout of a norm policy under observed drivers, subtracted from the observation"}
    return reg


def knockout(model, name, mode):
    if name not in modules(model) or mode not in ("identity", "zero", "mean"):
        raise ValueError((name, mode))
    new = {**model, "params": {k: v.copy() for k, v in model["params"].items()}, "ko": dict(model["ko"])}
    new["ko"][name] = mode
    if name == "intent_head" and mode == "zero":
        new["params"]["W2"][:] = 0.0
    return new


MUTANTS = {"sign_flip_update": "optimizer ascends the loss", "zero_lr": "learning rate set to zero",
           "zero_grad_W1": "encoder weight gradient zeroed", "zero_grad_custom_theta": "custom-simulator gradient zeroed"}
TASK_TYPES = ["sequence_classification"]


# ------------------------------------------------------------------ training
def set_norm(model, X):
    model["norm"] = np.array([X[:, :, 0].std() + 1e-6, X[:, :, 1].std() + 1e-6])


def fit(model, data, budget, rng):
    X, y = data
    set_norm(model, X)
    lr = 0.0 if MUTANT["name"] == "zero_lr" else LR
    sign = -1.0 if MUTANT["name"] == "sign_flip_update" else 1.0
    opt = Adam(model["params"], lr=lr)
    losses = []
    for _ in range(budget):
        idx = rng.choice(X.shape[0], size=min(BATCH, X.shape[0]), replace=False)
        loss, g = loss_and_grads(model, (X[idx], y[idx]))
        if not np.isfinite(loss):
            raise FloatingPointError("non-finite loss")
        clip_global_norm(g, MIND_CARD["thresholds"]["grad_clip_norm"])
        opt.step(model["params"], g, sign=sign)
        losses.append(loss)
    return losses


def accuracy(model, split):
    X, y = split
    return float(np.mean(predict(model, X) == y))


def full_loss(model, data):
    return loss_and_grads(model, data)[0]


# ------------------------------------------------------------------ tests: correctness
def trivial_accuracy(train_y, split):
    return float(np.mean(split[1] == np.bincount(train_y, minlength=N_CLASSES).argmax()))


def c1_gradcheck(rng, splits, trained=None):
    """C1 on every tensor of the chapter model (init and trained) and of the baseline (init)."""
    Xb, yb = splits["train"][0][:24], splits["train"][1][:24]
    out = {}
    models = {"custom_init": build_model(4, N_CLASSES, "sequence_classification", rng),
              "baseline_init": build_model(4, N_CLASSES, "sequence_classification", rng, variant="raw")}
    if trained is not None:
        models["custom_trained"] = trained
    for tag, m in models.items():
        if tag != "custom_trained":
            set_norm(m, splits["train"][0])
        _, g = loss_and_grads(m, (Xb, yb))
        worst, per = finite_difference_check(lambda p, m=m: full_loss(m, (Xb, yb)), m["params"], g, rng)
        out[tag] = (worst, per)
    worst = max(v[0] for v in out.values())
    return worst <= MIND_CARD["thresholds"]["gradcheck_rel_tol"], worst, out


def c2_determinism(seed, splits, steps=40):
    runs = []
    for _ in range(2):
        rng = np.random.default_rng(seed)
        m = build_model(4, N_CLASSES, "sequence_classification", rng)
        losses = fit(m, splits["train"], steps, rng)
        runs.append((np.array(losses), predict(m, splits["test"][0]), m))
    same = np.array_equal(runs[0][0], runs[1][0]) and np.array_equal(runs[0][1], runs[1][1])
    hs = hidden_states(runs[0][2], splits["test"][0])
    finite = all(np.all(np.isfinite(v)) for v in runs[0][2]["params"].values()) and \
        all(np.all(np.isfinite(v)) for v in hs.values())
    return same and finite, f"identical={same} finite={finite}"


def c3_learning(losses, model, splits):
    first, last = np.mean(losses[:10]), np.mean(losses[-10:])
    drop = (first - last) / first
    acc = accuracy(model, splits["test"])
    triv = trivial_accuracy(splits["train"][1], splits["test"])
    th = MIND_CARD["thresholds"]
    ok = drop >= th["loss_drop_fraction"] and acc - triv >= th["margin_over_trivial"]
    return ok, f"loss drop {drop:.3f}, test acc {acc:.3f} vs trivial {triv:.3f}"


def c4_shuffled(seed, splits, steps, repeats):
    """Held-out NLL of shuffled-label readers must not beat the trivial NLL log(K) by more than the band."""
    Xtr, ytr = splits["train"]
    Xte, yte = splits["test"]
    nlls, accs = [], []
    for r in range(repeats):
        rng = np.random.default_rng(seed + 7919 + r)
        m = build_model(4, N_CLASSES, "sequence_classification", rng)
        fit(m, (Xtr, rng.permutation(ytr)), steps, rng)
        lg = forward(m, Xte)[0]
        nlls.append(float(np.mean(logsumexp(lg) - lg[np.arange(yte.size), yte])))
        accs.append(float(np.mean(np.argmax(lg, 1) == yte)))
    th = MIND_CARD["thresholds"]
    ok = np.mean(nlls) >= math.log(N_CLASSES) - th["shuffle_nll_band"]
    return ok, (f"mean held-out NLL {np.mean(nlls):.3f} vs trivial {math.log(N_CLASSES):.3f} (band {th['shuffle_nll_band']}); "
                f"mean acc {np.mean(accs):.3f} vs trivial {trivial_accuracy(ytr, splits['test']):.3f}; {repeats} shuffles")


def c5_mutants(seed, splits, steps):
    detected, rows = 0, []
    for name in MUTANTS:
        MUTANT["name"] = name
        try:
            rng = np.random.default_rng(seed + 31)
            ok1, _, _ = c1_gradcheck(rng, splits)
            m = build_model(4, N_CLASSES, "sequence_classification", rng)
            losses = fit(m, splits["train"], steps, rng)
            ok3, _ = c3_learning(losses, m, splits)
        finally:
            MUTANT["name"] = None
        caught = (not ok1) or (not ok3)
        detected += int(caught)
        rows.append((name, caught, "C1" if not ok1 else ("C3" if not ok3 else "-")))
    return detected == len(MUTANTS), detected, rows


def _search(rng, draws, check):
    for _ in range(draws):
        theta = np.array([rng.uniform(-3, 3), rng.uniform(-3, 1), rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(-1, 3)])
        W = rng.uniform(0, 1, (8, T_STEPS))
        A = (np.arange(T_STEPS)[None, :] >= rng.integers(0, 6, 8)[:, None]).astype(np.float64)
        if not check(theta, W, A):
            return False
    return True


def c6_properties(rng, draws=400):
    """C6.1 positivity and C6.2 withholding dominance, each with a negative control that must fail."""
    def pos(variant):
        def chk(th, W, A):
            yh, rh, _ = custom_rollout(th, W, A, variant=variant)
            return yh.min() >= -1e-12 and rh.min() >= -1e-12
        return chk

    def dom(variant):
        def chk(th, W, A):
            yc = custom_rollout(th, W, A, variant=variant)[0]
            yw = custom_rollout(th, W, A, withhold=True, variant=variant)[0]
            return np.all(yw >= yc - 1e-12)
        return chk
    res = {"C6.1 positivity": _search(rng, draws, pos(None)),
           "C6.1 negative control (additive losses) violates": not _search(rng, draws, pos("additive")),
           "C6.2 withholding dominance": _search(rng, draws, dom(None)),
           "C6.2 negative control (crowding rot) violates": not _search(rng, draws, dom("crowding"))}
    return all(res.values()), res


def c7_split_integrity(splits):
    def hashes(X):
        return {hashlib.sha1(np.round(x, 12).tobytes()).hexdigest() for x in X}
    tr = hashes(splits["train"][0])
    overlap = sum(len(tr & hashes(splits[k][0])) for k in ("val", "test", "shifted", "blind"))
    return overlap == 0, f"overlapping examples: {overlap}"


# ------------------------------------------------------------------ tests: hypotheses
def run_seed(seed, steps):
    ss = np.random.SeedSequence(seed).spawn(3)
    rd, rm, rb = (np.random.default_rng(s) for s in ss)
    splits = make_splits(rd)
    model = build_model(4, N_CLASSES, "sequence_classification", rm)
    losses = fit(model, splits["train"], steps, rm)
    base = build_model(4, N_CLASSES, "sequence_classification", rb, variant="raw")
    fit(base, splits["train"], steps, rb)
    res = {k: (accuracy(model, splits[k]), accuracy(base, splits[k])) for k in ("val", "test", "shifted", "blind")}
    ko = {}
    for name, mode in (("custom_simulator", "mean"), ("drivers_stream", "mean"), ("ramp_pool", "zero"),
                       ("trace_encoder", "zero"), ("intent_head", "zero")):
        ko[name] = accuracy(knockout(model, name, mode), splits["shifted"]) - res["shifted"][0]
    return dict(splits=splits, model=model, base=base, losses=losses, acc=res, ko=ko)


def hypotheses(runs, rng, quick):
    out = []
    spec = {h["id"]: h for h in MIND_CARD["hypotheses"]}
    diffs = {"H-SIG": [r["acc"]["shifted"][0] - r["acc"]["shifted"][1] for r in runs],
             "H-NEC": [r["ko"]["custom_simulator"] - r["ko"]["drivers_stream"] for r in runs],
             "H-BLIND": [r["acc"]["blind"][0] - r["acc"]["blind"][1] for r in runs]}
    for hid, d in diffs.items():
        h = spec[hid]
        if quick or len(d) < 2:
            out.append({"id": hid, "metric": h["metric"], "mean_diff": float(np.mean(d)), "ci95": [None, None],
                        "mesi": h["mesi"], "n_seeds": len(d), "verdict": "not evaluated"})
            continue
        mean, ci = paired_bootstrap_ci(d, rng)
        out.append({"id": hid, "metric": h["metric"], "mean_diff": mean, "ci95": ci, "mesi": h["mesi"],
                    "n_seeds": len(d), "verdict": verdict(mean, ci, h["direction"], h["mesi"])})
    return out


def knockout_table(runs, rng, sig_names=("custom_simulator",)):
    rows = []
    for name in runs[0]["ko"]:
        d = [r["ko"][name] for r in runs]
        mean, ci = paired_bootstrap_ci(d, rng) if len(d) > 1 else (float(d[0]), [None, None])
        rows.append({"module": name, "signature": bool(name in sig_names), "metric_change": float(mean), "ci95": ci})
    return rows


# ------------------------------------------------------------------ real-data bridge
def load_csv(path):
    """Schema: field_id,t,standing,decay,rain,sown,label (one row per field and step, T rows per field)."""
    rows = [ln.strip().split(",") for ln in open(path, encoding="utf-8") if ln.strip()][1:]
    fields = {}
    for r in rows:
        fields.setdefault(r[0], []).append((int(r[1]), [float(v) for v in r[2:6]], int(r[6])))
    X = np.array([[v for _, v, _ in sorted(f)] for f in fields.values()])
    y = np.array([f[0][2] for f in fields.values()])
    return X, y


# ------------------------------------------------------------------ report
def fmt_ci(ci):
    return "[   n/a,    n/a]" if ci[0] is None else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def report_lines(ctx):
    L = [f"=== VERIFIED REPORT · chapter 0291 ===",
         f"file            {os.path.basename(__file__)}   card revision {MIND_CARD['card_revision']}",
         f"environment     python {sys.version.split()[0]} · numpy {np.__version__}",
         f"seeds           {ctx['seeds']}   mode {'quick' if ctx['quick'] else 'full'}   steps {ctx['steps']}",
         f"runtime         {ctx['runtime']:.1f} s (budget {ctx['budget']:.0f} s)",
         f"parameters      custom reader {ctx['n_params'][0]} · raw baseline {ctx['n_params'][1]}"
         f" (ratio {ctx['n_params'][0] / ctx['n_params'][1]:.3f})",
         f"gradcheck       max rel. error {ctx['gc_worst']:.2e} over {ctx['gc_tensors']} tensor checks "
         f"(init, after {ctx['gc_after']} steps, baseline init)", "", "correctness"]
    for cid, name, ok, det in ctx["correct"]:
        L.append(f"  {cid:<3} {name:<26} {'PASS' if ok else 'FAIL'}  {det}")
    L.append(f"\nmutation score  {ctx['mut'][0]}/{len(MUTANTS)}")
    for name, caught, by in ctx["mut"][1]:
        L.append(f"  {name:<24} {'detected by ' + by if caught else 'NOT DETECTED'}")
    L.append("\nhypotheses (paired over seeds, 95% bootstrap, 2000 resamples)")
    for h in ctx["hyp"]:
        L.append(f"  {h['id']:<8} mean diff {h['mean_diff']:+.3f}  CI {fmt_ci(h['ci95'])}  mesi {h['mesi']:.2f}  -> {h['verdict']}")
    L.append("\nknockouts on the shifted (drought) split: accuracy change vs intact reader")
    for k in ctx["ko"]:
        L.append(f"  {k['module']:<18} {'sig' if k['signature'] else '   '}  {k['metric_change']:+.3f}  CI {fmt_ci(k['ci95'])}")
    L.append("\naccuracy by split (mean over seeds): custom reader / raw baseline")
    for split in ("val", "test", "shifted", "blind"):
        a = np.array([r["acc"][split] for r in ctx["runs"]])
        L.append(f"  {split:<8} {a[:, 0].mean():.3f} / {a[:, 1].mean():.3f}")
    L.append(f"\nlearned custom (seed {ctx['seeds'][0]}): " + ", ".join(
        f"{k}={v:.3f} (true {TRUE_DYN[k]:.3f})" for k, v in zip(TRUE_DYN, unpack_custom(ctx['runs'][0]['model']['params']['custom_theta']))))
    L += [ctx["data_note"], f"task types      {', '.join(TASK_TYPES)}", f"exit code       {ctx['exit']}", "=== END REPORT ==="]
    return L


def json_payload(ctx):
    return {"schema_version": "1.0", "chapter": 291, "file": os.path.basename(__file__),
            "card_revision": MIND_CARD["card_revision"],
            "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
            "seeds": ctx["seeds"], "runtime_s": round(ctx["runtime"], 2), "n_params": ctx["n_params"][0],
            "gradcheck": {"tensors_checked": ctx["gc_tensors"], "tensors_total": ctx["gc_tensors"],
                          "max_rel_error": ctx["gc_worst"], "checked_at": ["init", "after_training_steps"],
                          "passed": bool(ctx["gc_worst"] <= MIND_CARD["thresholds"]["gradcheck_rel_tol"])},
            "correctness": [{"id": c, "name": n, "passed": bool(o), "detail": d} for c, n, o, d in ctx["correct"]],
            "mutants": {"detected": ctx["mut"][0], "total": len(MUTANTS), "score": ctx["mut"][0] / len(MUTANTS)},
            "hypotheses": ctx["hyp"], "knockouts": ctx["ko"], "task_types": TASK_TYPES, "exit_code": ctx["exit"]}


# ------------------------------------------------------------------ command line
def parse_args(argv):
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__),
                                 description="Chapter 0291 Yennenga: custom-counterfactual residual reader.")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--card", action="store_true")
    ap.add_argument("--mutant", type=str, default=None)
    ap.add_argument("--data", type=str, default=None)
    return ap.parse_args(argv)


def correctness_suite(args, runs, steps, ctx):
    r0 = runs[0]
    rng = np.random.default_rng(args.seed + 101)
    trained = build_model(4, N_CLASSES, "sequence_classification", np.random.default_rng(args.seed + 5))
    after = fit(trained, r0["splits"]["train"], 60, np.random.default_rng(args.seed + 6))
    ok1, worst, gc = c1_gradcheck(rng, r0["splits"], trained=trained)
    ctx.update(gc_worst=worst, gc_tensors=sum(len(v[1]) for v in gc.values()), gc_after=len(after))
    ok2, d2 = c2_determinism(args.seed, r0["splits"])
    ok3, d3 = c3_learning(r0["losses"], r0["model"], r0["splits"])
    ok4, d4 = c4_shuffled(args.seed, r0["splits"], steps, 1 if args.quick else MIND_CARD["thresholds"]["shuffle_repeats"])
    ok5, n5, rows5 = (True, 0, []) if args.mutant else c5_mutants(args.seed, r0["splits"], steps)
    ok6, res6 = c6_properties(np.random.default_rng(args.seed + 202))
    ok7, d7 = c7_split_integrity(r0["splits"])
    ctx["mut"] = (n5, rows5)
    ctx["correct"] = [("C1", "gradient check", ok1, f"max rel err {worst:.2e}"),
                      ("C2", "determinism, finiteness", ok2, d2), ("C3", "learning", ok3, d3),
                      ("C4", "shuffled-label control", ok4, d4),
                      ("C5", "mutant detection", ok5, "skipped under --mutant" if args.mutant else f"{n5}/{len(MUTANTS)}"),
                      ("C6", "property tests", ok6, "; ".join(f"{k}: {'ok' if v else 'FAIL'}" for k, v in res6.items())),
                      ("C7", "split integrity", ok7, d7)]
    return all(c[2] for c in ctx["correct"])


def data_bridge(args, model):
    if not args.data:
        return "real data      no --data given; real-data bridge skipped"
    X, y = load_csv(args.data)
    return f"real data      {args.data}: {X.shape[0]} fields, accuracy {float(np.mean(predict(model, X) == y)):.3f}"


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.card:
        print(json.dumps(MIND_CARD, indent=2))
        return 0
    if args.mutant is not None and args.mutant not in MUTANTS:
        print(f"unknown mutant {args.mutant}; choose from {sorted(MUTANTS)}", file=sys.stderr)
        return 2
    MUTANT["name"] = args.mutant
    t0 = time.time()
    steps, n_seeds = (QUICK_STEPS, 1) if args.quick else (FULL_STEPS, max(1, args.seeds))
    seeds = [args.seed + k for k in range(n_seeds)]
    print(f"chapter 0291 · seeds {seeds} · steps {steps} · mutant {args.mutant}")
    runs = [run_seed(s, steps) for s in seeds]
    ctx = {"seeds": seeds, "quick": args.quick, "steps": steps, "runs": runs,
           "n_params": (n_params(runs[0]["model"]), n_params(runs[0]["base"]))}
    passed = correctness_suite(args, runs, steps, ctx)
    brng = np.random.default_rng(args.seed + 303)
    ctx["hyp"] = hypotheses(runs, brng, args.quick)
    ctx["ko"] = knockout_table(runs, brng)
    ctx["data_note"] = data_bridge(args, runs[0]["model"])
    ctx["runtime"] = time.time() - t0
    ctx["budget"] = QUICK_BUDGET_S if args.quick else BUDGET_S
    within = ctx["runtime"] <= ctx["budget"]
    ctx["correct"].append(("C8", "runtime budget", within, f"{ctx['runtime']:.1f} s <= {ctx['budget']:.0f} s"))
    ctx["exit"] = 0 if (passed and within) else (3 if passed else 1)
    write_report(report_lines(ctx), json_payload(ctx), args.json)
    return ctx["exit"]


if __name__ == "__main__":
    try:
        sys.exit(main())
    except FloatingPointError as exc:
        print(f"non-finite values: {exc}", file=sys.stderr)
        sys.exit(4)

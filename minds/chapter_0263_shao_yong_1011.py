#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0263 · Shao Yong (1011-1077)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 14, Minds 261-280 Available on Amazon https://www.amazon.com/dp/B0HLYSVY5D
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0263_shao_yong_1011 - Shao Yong (1011-1077)
# END ATTRIBUTION
"""
XIANTIAN CLOCK - one doubling-basis cycle table read at every level of a
given calendar, plus one order-sensitive "X of Y" cross-scale kernel.

Thesis
    The slow cycle you cannot observe is the fast cycle you can observe, read
    at another scale: learn the shape of the day and you have the shape of the
    aeon (Shao Yong: one yuan within the great transformation is like one year).

Evidence and provenance (archetype_provenance = belief)
    Grounded in his own surviving Inner Chapters on Observing Things (Guanwu
    neipian, in the Huangji jingshi shu) and in the Outer Chapters recorded by
    disciples. The diagrams attributed to him reach us through Zhu Xi, and his
    son Shao Bowen reworked parts of the book (Bol 2013), so D3 and D5 rest on a
    transmitted text. Nothing here claims to replicate the man; the file tests
    an idea extracted from the texts.

Doctrine -> mechanism -> test (IDs as in MIND_CARD)
    D1 one yuan is like one year; periods multiply by 12 and 30
        -> M1 one shared profile table read by every scale  -> D-1, H-SIG
    D2 the law of waxing and waning was left unwritten on purpose
        -> numbers given a priori, shape learned from traces -> C3, H-SIG
    D3 one divides into two ... thirty-two into sixty-four
        -> M1 parametrised in a coarse-to-fine doubling basis -> C6.1
    D4 observe things by means of things, not by means of the self
        -> tied reading (M1) versus each scale on its own    -> H-FANGUAN
    D5 day-counts swell in spring and summer (the day read "of" the year);
       Bol's "X of Y" grids distinguish walkers-of-nature from nature-of-walkers
        -> M2 shared, order-sensitive cross-scale kernel      -> C6.3, H-NEC
    D7 the cosmos ends and begins again; no irreversible term
        -> no trend anywhere in the model                     -> C6.2, H-BLIND

Research question (forecasting and time-to-event; sample efficiency and transfer)
    When the longest season of a multi-seasonal series has been observed only
    up to its peak, can tying the cycle shape across scales forecast the unseen
    remainder better than a size-matched Fourier-seasonal-plus-trend model, and
    what does the tie cost when the slow component is an arrow, not a cycle?

Closest prior art and the delta
    Prophet (Taylor and Letham 2018) and TBATS (De Livera, Hyndman and Snyder
    2011): independent trigonometric seasonalities plus trend; the Prophet-style
    model is implemented below as the baseline. Haar multiresolution (Haar 1910;
    Mallat 1989) supplies the basis; cross-scale weight sharing appears in
    scale-equivariant networks (Sosnovik et al. 2020). Delta: one table shared
    by all levels of a given mixed-radix calendar and one shared non-commutative
    kernel between adjacent levels, with no trend, so fully observed fast cycles
    fix the shape of a slow cycle observed only up to its noon.

Blind spot
    History is not a year. Shao placed his own age near cosmic noon and read
    decline into what followed. Where the slow component accumulates
    irreversibly (the arrow world) the tie should forecast a sunset that never
    comes, and the trend baseline should win (H-BLIND).

Task (generative process)
    t ~ U(0, 0.55 * 4320) in-window; shifted split t ~ U(0.55 * 4320, 4320)
    phases: phi_s = frac(t / P_s), P = (12, 360, 4320)  (day, month, year)
    g = normalised opening-and-closing profile: zero before mid-yin (2.5/12)
        and after mid-xu (10.5/12), rising and setting between
    y = sum_s a_s g(phi_s) + 0.5 sum_s m_s g(phi_s) g(phi_{s+1}) + beta*e + noise
    e ~ Bernoulli(0.06) is an observed nuisance event; noise sd 0.2
    arrow world: the slow g(phi_2) is replaced by a logistic rise that matches
    the morning of the cycle and never sets

Limits
    Synthetic data; the calendar radices are supplied, not discovered; phases
    are exact. A research prototype of an AGI-oriented forecasting component,
    not an AGI, and no statement about the historical person's mind.

Run: python3 <this file>   (full protocol; --quick, --card, --json PATH, --data PATH)
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 2,
    "revision_log": ["rev 2 (after one --quick run, 1 seed, hypotheses not evaluated): the declared "
                     "Fourier baseline fits yearly harmonics to 0.55 of a year, which Prophet's documented "
                     "default refuses (a seasonality is enabled only when two cycles are observed). A "
                     "Prophet-default baseline (seasonalities for periods seen twice, piecewise-linear trend "
                     "with 20 changepoints) was ADDED as secondary comparisons H-SIG.b and H-BLIND.b. "
                     "H-SIG, H-FANGUAN, H-NEC and H-BLIND, their metrics, directions and mesi are unchanged "
                     "and still decide the verdicts against the original baseline. The 9-gram similarity to the "
                     "only available corpus file was filled in from the gate (0.028)."],
    "generation": {"template_version": "codeguidelines 1.0 (15 Sep 2026)", "generator": "Claude",
                   "generator_version": "Claude Opus 5", "date": "2026-09-21"},
    "id": 263, "figure": "Shao Yong", "born": 1011, "died": 1077, "civilization": "Chinese (Song)",
    "provenance": "belief",
    "thesis": ("The slow cycle one cannot observe is the fast cycle one can observe, read at another "
               "scale: one table of waxing and waning, read at every level of a given calendar and "
               "composed with itself 'X of Y', forecasts the unseen remainder of the aeon."),
    "evidence": [
        {"id": "D1", "claim": "One yuan within the great transformation is like one year; the method "
         "multiplies by 12 and 30 (yuan, hui, yun, shi as year, month, day, double-hour)",
         "basis": "primary", "source": "Guanwu neipian 10, Huangji jingshi shu"},
        {"id": "D2", "claim": "The account of waxing and waning was deliberately not written down, "
         "so that readers would seek it", "basis": "primary",
         "source": "Guanwu neipian 10 (xiaoxi yingxu zhi shuo bu zhu yu shu)"},
        {"id": "D3", "claim": "One divides into two, two into four ... thirty-two into sixty-four; "
         "Cheng Hao called the method 'adding one doubling'", "basis": "primary",
         "source": "Guanwu waipian; Ercheng waishu 12"},
        {"id": "D4", "claim": "Reverse observation: observe things by means of things, not by means "
         "of the self", "basis": "primary", "source": "Guanwu neipian 12"},
        {"id": "D5", "claim": "In spring and summer yang predominates, so day-counts are many and "
         "night-counts few: the day is read of the year", "basis": "primary",
         "source": "Guanwu waipian (Wikisource Huangji jingshi juan 13)"},
        {"id": "D6", "claim": "The Inner Chapters generate systematic relations between members of "
         "finite sets; row-of-column and column-of-row are distinct",
         "basis": "scholarship", "source": "Bol, 'On Shao Yong's Method for Observing Things', "
         "Monumenta Serica 61 (2013) 287-299"},
        {"id": "D7", "claim": "Things open in the third epoch and close in the eleventh, then the "
         "cycle begins again; his own age sits just after cosmic noon, in decline",
         "basis": "primary", "source": "Huangji jingshi chronological tables (yi yuan jing hui)"},
    ],
    "research_question": {"category": "forecasting and time-to-event; sample efficiency and transfer",
                          "question": ("Does sharing one cycle shape across nested calendar scales "
                                       "forecast the unobserved remainder of a partially observed slow "
                                       "cycle better than independent seasonalities plus trend, and "
                                       "what does the tie cost when the slow component is an arrow?")},
    "mechanism": {
        "name": "Xiantian clock", "family": "structured forecasting with cross-scale parameter sharing",
        "signature_modules": ["xiantian_profile", "of_kernel"],
        "closest_prior_art": ["Prophet additive Fourier seasonality plus trend (Taylor and Letham 2018)",
                              "TBATS trigonometric multiple seasonality (De Livera, Hyndman, Snyder 2011)",
                              "Haar multiresolution basis (Haar 1910; Mallat 1989)",
                              "scale-equivariant weight sharing (Sosnovik, Szmaja, Smeulders 2020)"],
        "overlap": "Medium",
        "prior_art_queries": ["shared seasonal profile across multiple seasonal periods",
                              "cross-scale weight tying time series forecasting",
                              "self-similar seasonality forecast partial cycle",
                              "scale-equivariant time series model"],
        "contribution_type": "mechanism",
        "delta": ("One doubling-basis cycle table read at every level of a given mixed-radix calendar, "
                  "plus one shared order-sensitive cross-scale kernel, with no trend term."),
    },
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1", "property_test": "D-1", "hypothesis": "H-SIG"},
        {"doctrine": "D2", "mechanism": "M1", "property_test": "C3", "hypothesis": "H-SIG"},
        {"doctrine": "D3", "mechanism": "M1", "property_test": "C6.1", "hypothesis": "H-SIG"},
        {"doctrine": "D4", "mechanism": "M1", "property_test": "D-1", "hypothesis": "H-FANGUAN"},
        {"doctrine": "D5", "mechanism": "M2", "property_test": "C6.3", "hypothesis": "H-NEC"},
        {"doctrine": "D6", "mechanism": "M2", "property_test": "C6.3", "hypothesis": "H-NEC"},
        {"doctrine": "D7", "mechanism": "no trend term", "property_test": "C6.2", "hypothesis": "H-BLIND"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "The tied clock forecasts the unobserved remainder of the slow "
         "cycle better than the size-matched Fourier-seasonal-plus-trend model",
         "metric": "R2", "split": "shifted", "comparison": "model - fourier_trend",
         "direction": "greater", "mesi": 0.10, "seeds": 5},
        {"id": "H-FANGUAN", "statement": "Reading every scale through one table beats letting each "
         "scale learn its own table at matched size (observing things by things, not by the self)",
         "metric": "R2", "split": "shifted", "comparison": "model - untied",
         "direction": "greater", "mesi": 0.10, "seeds": 5},
        {"id": "H-NEC", "statement": "Knocking out the of-kernel costs more than knocking out the "
         "matched non-signature additive head (exogenous-event head)",
         "metric": "R2", "split": "shifted", "comparison": "signature_knockout - matched_knockout",
         "direction": "less", "mesi": 0.05, "seeds": 5},
        {"id": "H-BLIND", "statement": "Where the slow component is an irreversible rise, the tied "
         "clock forecasts a sunset and loses to the trend baseline", "condition": "arrow world",
         "grounding": "history read as a returning year; decline predicted after cosmic noon",
         "metric": "R2", "split": "shifted", "comparison": "model - fourier_trend",
         "direction": "less", "mesi": 0.10, "seeds": 5},
        {"id": "H-SIG.b", "statement": "Secondary (rev 2): as H-SIG against the Prophet-default baseline",
         "metric": "R2", "split": "shifted", "comparison": "model - prophet_default",
         "direction": "greater", "mesi": 0.10, "seeds": 5},
        {"id": "H-BLIND.b", "statement": "Secondary (rev 2): as H-BLIND against the Prophet-default baseline",
         "condition": "arrow world", "grounding": "history read as a returning year",
         "metric": "R2", "split": "shifted", "comparison": "model - prophet_default",
         "direction": "less", "mesi": 0.10, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.5, "margin_over_trivial": 0.3,
                   "shuffled_band_over_trivial": 0.10, "gradcheck_rtol": 1e-5},
    "probe_predictions": [
        {"probe": "P7", "expected": "above baseline when regimes recur; below under a trend regime"},
        {"probe": "P10", "expected": "above baseline (one table pools data from every scale)"},
        {"probe": "P6", "expected": "above baseline on periodic targets"},
        {"probe": "P9", "expected": "equal to baseline"},
        {"probe": "P1-P5, P8, P11", "expected": "not applicable (vector input, column 0 is time)"},
    ],
    "dialectic_links": [
        {"chapter": 299, "relation": "successor", "test": "none (Zhu Xi placed the prior-to-Heaven "
         "diagrams at the head of his Zhouyi benyi and Yixue qimeng)"},
        {"chapter": 636, "relation": "successor", "test": "none (Leibniz read the diagram in 1703; "
         "binary is the eponym trap for his chapter, not a claim here)"},
        {"chapter": 273, "relation": "rival", "test": "none (Shen Kuo, a measuring astronomer, recorded "
         "Zheng Jue, a contemporary of Shao with his own method of generating the hexagrams)"},
    ],
    "corpus_neighbors": [
        {"chapter": 445, "similarity": None, "difference": "Copernicus detects a tie among sub-models "
         "at one scale; here the tie across scales is assumed a priori and used to extrapolate"},
        {"chapter": 192, "similarity": None, "difference": "Bede reconciles incommensurate periods; "
         "here periods are given and the shape is transferred between them"},
        {"chapter": 195, "similarity": None, "difference": "Yi Xing interpolates second differences on "
         "unequal intervals and withholds a unit; no shared cross-scale table"},
        {"chapter": 262, "similarity": None, "difference": "Al-Sarakhsi gates causal admissibility; "
         "no temporal structure"},
        {"chapter": 502, "similarity": 0.028, "difference": "Sample file (Philip II, shipped as 0507): "
         "multiplicative survival of annotations; the only code available for the 9-gram gate"},
    ],
    "barometer": {
        "cognitive_processing": ["P10 sample efficiency", "transfer of shape to an unseen scale"],
        "embodied_cognition": [], "world_modeling": ["shifted-split free-run forecast", "arrow world"],
        "consciousness": [], "language_understanding": [], "emotional_intelligence": [],
        "creativity": [], "autonomy": [],
    },
    "task_types": ["vector_regression"],
    "applications": [
        {"use": "multi-seasonal load forecasting when the longest season is only partly recorded",
         "sector": "energy", "dataset": "UCI ElectricityLoadDiagrams20112014"},
        {"use": "seasonal activity forecasting from short new sensor records",
         "sector": "ecology and agriculture", "dataset": "USA National Phenology Network observations"},
        {"use": "audit that flags when a rising series is an arrow rather than a returning cycle",
         "sector": "climate science", "dataset": "NOAA Mauna Loa monthly mean CO2"},
    ],
    "safety_notes": "Numerology is discussed as history only; no divination, no forecasts about people.",
}

PERIODS = (12.0, 360.0, 4320.0)       # day of 12 double-hours, month of 30 days, year of 12 months
OPEN_AT, CLOSE_AT = 2.5 / 12.0, 10.5 / 12.0
T_OBS_FRACTION = 0.55
TASK_TYPES = ["vector_regression"]
ACTIVE_MUTANT = None


# BEGIN STANDARD UTILITIES v1.0
def softplus(x):
    return np.logaddexp(0.0, x)


def logsumexp(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(m, axis) + np.log(np.sum(np.exp(x - m), axis=axis))


def softmax(x, axis=-1):
    z = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return z / np.sum(z, axis=axis, keepdims=True)


class Adam:
    def __init__(self, params, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


def clip_global_norm(grads, max_norm):
    total = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    if total > max_norm:
        scale = max_norm / (total + 1e-12)
        grads = {k: g * scale for k, g in grads.items()}
    return grads, total


def finite_difference_check(loss_fn, params, grads, rng, n_entries=20, eps=1e-6, rtol=1e-5, atol=1e-9):
    """Central differences on random entries of every tensor plus its largest-gradient entry."""
    worst, ok, checked = 0.0, True, 0
    for k, p in params.items():
        flat, gflat = p.reshape(-1), grads[k].reshape(-1)
        idx = set(rng.choice(flat.size, size=min(n_entries, flat.size), replace=False).tolist())
        idx.add(int(np.argmax(np.abs(gflat))))
        for i in sorted(idx):
            old = flat[i]
            flat[i] = old + eps
            lp = loss_fn()
            flat[i] = old - eps
            lm = loss_fn()
            flat[i] = old
            num, ana = (lp - lm) / (2 * eps), float(gflat[i])
            err, scale = abs(num - ana), max(abs(num), abs(ana))
            if err > rtol * scale + atol:
                ok = False
            if scale > 1e-6:
                worst = max(worst, err / scale)
        checked += 1
    return ok, worst, checked


def paired_bootstrap(diffs, rng, n_boot=2000):
    d = np.asarray(diffs, dtype=np.float64)
    means = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    return float(d.mean()), [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def verdict(mean, ci, direction, mesi):
    if direction == "greater":
        if ci[0] > 0 and mean >= mesi:
            return "supported"
        return "contradicted" if ci[1] < 0 else "inconclusive"
    if ci[1] < 0 and mean <= -mesi:
        return "supported"
    return "contradicted" if ci[0] > 0 else "inconclusive"


def write_report(report, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
# END STANDARD UTILITIES


# ----------------------------------------------------------------- data and tasks
def _open_close_raw(phi):
    x = (phi - OPEN_AT) / (CLOSE_AT - OPEN_AT)
    out = np.zeros_like(phi, dtype=np.float64)
    inside = (x > 0) & (x < 1)
    out[inside] = np.sin(np.pi * x[inside]) ** 1.5
    return out


_GRID = np.arange(20000) / 20000.0
_RAW_MU, _RAW_SD = float(_open_close_raw(_GRID).mean()), float(_open_close_raw(_GRID).std())
_NOON = 0.5 * (OPEN_AT + CLOSE_AT)


def shao_profile(phi):
    """Opening-and-closing shape shared by day, month and year (zero mean, unit variance)."""
    return (_open_close_raw(phi) - _RAW_MU) / _RAW_SD


def arrow_profile(phi):
    """Blind-spot world: an irreversible rise that matches the morning and never sets."""
    lo = float(shao_profile(np.array([0.0]))[0])
    hi = float(shao_profile(np.array([_NOON]))[0])
    return lo + (hi - lo) / (1.0 + np.exp(-(phi - 0.40) / 0.05))


def phases(t, periods):
    return np.mod(np.asarray(t, dtype=np.float64)[:, None] / np.asarray(periods)[None, :], 1.0)


def make_world(rng, world="cycle", sizes=(1500, 250, 250, 500), periods=PERIODS):
    """Draw one synthetic cosmos. The calendar numbers are given; the shape must be learned."""
    S, top = len(periods), periods[-1]
    amp = rng.uniform(0.7, 1.3, size=S)
    mod = rng.uniform(0.5, 0.8, size=S - 1)
    beta, sigma, rate = rng.uniform(0.8, 1.2), 0.2, 0.06
    t_obs = T_OBS_FRACTION * top
    n_in = sizes[0] + sizes[1] + sizes[2]
    t_all = np.concatenate([rng.uniform(0.0, t_obs, size=n_in), rng.uniform(t_obs, top, size=sizes[3])])

    ph = phases(t_all, periods)
    parts = [shao_profile(ph[:, s]) for s in range(S)]
    if world == "arrow":
        parts[-1] = arrow_profile(ph[:, -1])
    y = sum(amp[s] * parts[s] for s in range(S))
    y = y + 0.5 * sum(mod[s] * parts[s] * parts[s + 1] for s in range(S - 1))
    e = (rng.random(t_all.size) < rate).astype(np.float64)
    y = y + beta * e + sigma * rng.standard_normal(t_all.size)
    X = np.stack([t_all, e], axis=1)

    perm = rng.permutation(n_in)
    cuts = np.cumsum(sizes[:3])
    idx = {"train": perm[:cuts[0]], "val": perm[cuts[0]:cuts[1]], "test": perm[cuts[1]:cuts[2]],
           "shift": np.arange(n_in, t_all.size)}
    data = {k: (X[v], y[v]) for k, v in idx.items()}
    data["index"] = idx
    data["meta"] = {"world": world, "t_obs": t_obs, "periods": periods}
    return data


def r2(y, yhat):
    var = float(np.var(y))
    return 1.0 - float(np.mean((y - yhat) ** 2)) / var if var > 0 else 0.0


# ----------------------------------------------------------------- model
def doubling_synthesis(depth, split=(1.0, -1.0)):
    """Leaf values from coefficients: taiji mean, then one yin/yang split per node (Haar-type)."""
    n = 2 ** depth
    H = np.zeros((n, n))
    H[:, 0] = 1.0
    j = np.arange(n)
    for k in range(1, depth + 1):
        node = j >> (depth - k + 1)
        bit = (j >> (depth - k)) & 1
        H[j, 2 ** (k - 1) + node] = np.where(bit == 0, split[0], split[1])
    return H


def coefficient_levels(depth):
    lv = np.zeros(2 ** depth)
    for k in range(1, depth + 1):
        lv[2 ** (k - 1):2 ** k] = k
    return lv


def phase_cells(X, periods, depth):
    n = 2 ** depth
    return np.minimum((phases(X[:, 0], periods) * n).astype(np.int64), n - 1)


def build_model(in_dim, out_dim, task_type, rng, **cfg):
    """kind: 'xiantian' (the mind), 'untied' (a table per scale), 'fourier' and 'prophet' (prior art)."""
    if task_type not in TASK_TYPES or out_dim != 1 or in_dim < 1:
        raise ValueError("vector_regression with one output; column 0 must be time")
    kind = cfg.get("kind", "xiantian")
    periods = tuple(cfg.get("periods", PERIODS))
    S, n_exo = len(periods), in_dim - 1
    c = {"periods": periods, "lam": cfg.get("lam", 1e-4), "t_scale": periods[-1]}
    p = {}
    if kind in ("xiantian", "untied"):
        tied = kind == "xiantian"
        c.update(depth=6 if tied else 5, kdepth=4 if tied else 3, rank=2 if tied else 1, tied=tied)
        n, nk = 2 ** c["depth"], 2 ** c["kdepth"]
        c["H"], c["Hk"] = doubling_synthesis(c["depth"]), doubling_synthesis(c["kdepth"])
        c["wc"] = (1.0 + coefficient_levels(c["depth"])) ** 2
        c["wk"] = (1.0 + coefficient_levels(c["kdepth"])) ** 2
        lead = () if tied else (S,)
        klead = () if tied else (S - 1,)
        p["c"] = rng.normal(0.0, 0.05, size=lead + (n,))
        p["A"] = np.full(S, 0.5)
        p["P"] = rng.normal(0.0, 0.3, size=klead + (c["rank"], nk))
        p["Q"] = rng.normal(0.0, 0.3, size=klead + (c["rank"], nk))
        p["B"] = np.full(S - 1, 0.1)
    elif kind in ("fourier", "prophet"):
        span = cfg.get("span", T_OBS_FRACTION * periods[-1])
        scales = list(range(S)) if kind == "fourier" else [s for s in range(S) if 2 * periods[s] <= span]
        c.update(order=cfg.get("order", 16 if kind == "fourier" else 24), xorder=cfg.get("xorder", 2),
                 scales=scales, pairs=[s for s in scales if s + 1 in scales],
                 cps=np.array([]) if kind == "fourier" else np.linspace(0.0, 0.8 * span, 21)[1:] / periods[-1])
        p["W"] = np.zeros((len(scales), 2 * c["order"]))
        p["Wx"] = np.zeros((len(c["pairs"]), (2 * c["xorder"]) ** 2))
        p["trend"] = np.zeros(1 + c["cps"].size)
    else:
        raise ValueError("unknown kind " + str(kind))
    p["w"] = np.zeros(n_exo)
    p["b"] = np.zeros(1)
    return {"kind": kind, "cfg": c, "params": p, "ko": {}, "stats": {}}


def _table(m, name, i):
    arr = m["params"][name]
    return arr if m["cfg"]["tied"] else arr[i]


def _parts_doubling(m, X):
    c, p = m["cfg"], m["params"]
    S = len(c["periods"])
    J, Jk = phase_cells(X, c["periods"], c["depth"]), phase_cells(X, c["periods"], c["kdepth"])
    V = [c["H"] @ _table(m, "c", s) for s in range(S)]
    G = np.stack([V[s][J[:, s]] for s in range(S)], axis=1)
    PV = [c["Hk"] @ _table(m, "P", s).T for s in range(S - 1)]
    QV = [c["Hk"] @ _table(m, "Q", s).T for s in range(S - 1)]
    F = [PV[s][Jk[:, s]] for s in range(S - 1)]
    Hh = [QV[s][Jk[:, s + 1]] for s in range(S - 1)]
    K = np.stack([np.sum(F[s] * Hh[s], axis=1) for s in range(S - 1)], axis=1)
    out = {"xiantian_profile": G * p["A"][None, :], "of_kernel": K * p["B"][None, :]}
    cache = {"J": J, "Jk": Jk, "G": G, "K": K, "F": F, "Hh": Hh}
    return out, cache


def _fourier_features(m, X):
    c = m["cfg"]
    ph = phases(X[:, 0], c["periods"])

    def harm(s, order):
        k = np.arange(1, order + 1)
        return np.concatenate([np.cos(2 * np.pi * ph[:, s:s + 1] * k), np.sin(2 * np.pi * ph[:, s:s + 1] * k)], 1)
    seas = [harm(s, c["order"]) for s in c["scales"]]
    cross = [(harm(s, c["xorder"])[:, :, None] * harm(s + 1, c["xorder"])[:, None, :]).reshape(X.shape[0], -1)
             for s in c["pairs"]]
    tau = X[:, :1] / c["t_scale"]
    trend = np.concatenate([tau, np.maximum(0.0, tau - c["cps"][None, :])], axis=1)
    return seas, cross, trend


def _parts_fourier(m, X):
    p = m["params"]
    seas, cross, trend = _fourier_features(m, X)
    n = X.shape[0]
    out = {"seasonal_terms": np.stack([seas[i] @ p["W"][i] for i in range(len(seas))] + [np.zeros(n)], 1),
           "cross_terms": np.stack([cross[i] @ p["Wx"][i] for i in range(len(cross))] + [np.zeros(n)], 1),
           "trend": (trend @ p["trend"])[:, None]}
    return out, {"seas": seas, "cross": cross, "tau": trend}


def forward(m, X):
    parts, cache = (_parts_doubling if m["kind"] in ("xiantian", "untied") else _parts_fourier)(m, X)
    parts["exo_head"] = (X[:, 1:] @ m["params"]["w"])[:, None]
    parts["taiji_bias"] = np.full((X.shape[0], 1), m["params"]["b"][0])
    for name, mode in m["ko"].items():
        if mode == "zero":
            parts[name] = np.zeros_like(parts[name])
        elif mode == "mean":
            parts[name] = np.broadcast_to(m["stats"][name], parts[name].shape).copy()
    yhat = sum(v.sum(axis=1) for v in parts.values())
    return yhat, parts, cache


def _regulariser(m):
    c, p, lam = m["cfg"], m["params"], m["cfg"]["lam"]
    if m["kind"] in ("fourier", "prophet"):
        kw = np.concatenate([np.arange(1, c["order"] + 1)] * 2) ** 2
        return lam * float(np.sum(kw * p["W"] ** 2) + np.sum(p["Wx"] ** 2) + np.sum(p["trend"][1:] ** 2)), \
            {"W": 2 * lam * kw * p["W"], "Wx": 2 * lam * p["Wx"],
             "trend": 2 * lam * np.concatenate([[0.0], p["trend"][1:]])}
    reg = lam * float(np.sum(c["wc"] * p["c"] ** 2) + np.sum(c["wk"] * (p["P"] ** 2 + p["Q"] ** 2)))
    return reg, {"c": 2 * lam * c["wc"] * p["c"], "P": 2 * lam * c["wk"] * p["P"],
                 "Q": 2 * lam * c["wk"] * p["Q"]}


def _grads_doubling(m, X, r, cache):
    c, p = m["cfg"], m["params"]
    S, n, nk, R = len(c["periods"]), 2 ** c["depth"], 2 ** c["kdepth"], c["rank"]
    g = {"A": (cache["G"] * r[:, None]).sum(0), "B": (cache["K"] * r[:, None]).sum(0),
         "c": np.zeros_like(p["c"]), "P": np.zeros_like(p["P"]), "Q": np.zeros_like(p["Q"])}
    for s in range(S):
        dV = np.bincount(cache["J"][:, s], weights=r * p["A"][s], minlength=n)
        if c["tied"]:
            g["c"] += c["H"].T @ dV
        else:
            g["c"][s] = c["H"].T @ dV
    for s in range(S - 1):
        w = r * p["B"][s]
        dPV = np.stack([np.bincount(cache["Jk"][:, s], weights=w * cache["Hh"][s][:, q], minlength=nk)
                        for q in range(R)], 1)
        dQV = np.stack([np.bincount(cache["Jk"][:, s + 1], weights=w * cache["F"][s][:, q], minlength=nk)
                        for q in range(R)], 1)
        if c["tied"]:
            g["P"] += (c["Hk"].T @ dPV).T
            g["Q"] += (c["Hk"].T @ dQV).T
        else:
            g["P"][s] = (c["Hk"].T @ dPV).T
            g["Q"][s] = (c["Hk"].T @ dQV).T
    return g


def _grads_fourier(m, X, r, cache):
    return {"W": np.stack([f.T @ r for f in cache["seas"]]).reshape(m["params"]["W"].shape),
            "Wx": np.stack([f.T @ r for f in cache["cross"]] or [np.zeros(0)]).reshape(m["params"]["Wx"].shape),
            "trend": cache["tau"].T @ r}


def loss_and_grads(model, batch):
    X, y = batch
    yhat, _, cache = forward(model, X)
    res = yhat - y
    reg, greg = _regulariser(model)
    loss = 0.5 * float(np.mean(res ** 2)) + reg
    r = res / y.size
    g = (_grads_doubling if model["kind"] in ("xiantian", "untied") else _grads_fourier)(model, X, r, cache)
    g["w"] = X[:, 1:].T @ r
    g["b"] = np.array([r.sum()])
    for k, v in greg.items():
        g[k] = g[k] + v
    if ACTIVE_MUTANT == "sign_flip":
        g = {k: -v for k, v in g.items()}
    elif ACTIVE_MUTANT == "zeroed_profile_grad" and "c" in g:
        g["c"] = np.zeros_like(g["c"])
    elif ACTIVE_MUTANT == "zeroed_kernel_grad" and "Q" in g:
        g["Q"] = np.zeros_like(g["Q"])
    return loss, g


def predict(model, X):
    return forward(model, X)[0]


def hidden_states(model, X):
    """Per-module readouts: scale profiles and pair kernels (or Fourier blocks), one column each."""
    parts = forward(model, X)[1]
    return np.concatenate([parts[k] for k in sorted(parts)], axis=1)


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


def modules(model):
    base = {"exo_head": {"params": ["w"], "role": "linear head on exogenous covariates", "signature": False},
            "taiji_bias": {"params": ["b"], "role": "global offset", "signature": False}}
    if model["kind"] in ("fourier", "prophet"):
        base.update({
            "seasonal_terms": {"params": ["W"], "role": "independent Fourier seasonality per period",
                               "signature": False},
            "cross_terms": {"params": ["Wx"], "role": "low-order Fourier products of adjacent periods",
                            "signature": False},
            "trend": {"params": ["trend"], "role": "linear or piecewise-linear trend in time",
                      "signature": False}})
    else:
        tied = model["cfg"]["tied"]
        base.update({
            "xiantian_profile": {"params": ["c", "A"], "signature": tied,
                                 "role": ("multiresolution periodic profile in a Haar-type doubling basis, "
                                          + ("one table shared by every scale" if tied else "one table per scale"))},
            "of_kernel": {"params": ["P", "Q", "B"], "signature": tied,
                          "role": "low-rank non-commutative interaction between adjacent calendar scales"}})
    return base


def knockout(model, name, mode="zero"):
    if name not in modules(model) or mode not in ("zero", "mean", "identity"):
        raise ValueError("unknown module or mode")
    if mode == "identity":
        raise ValueError("modules are additive heads; identity replacement is undefined, use zero or mean")
    clone = {"kind": model["kind"], "cfg": model["cfg"], "stats": model["stats"],
             "params": {k: v.copy() for k, v in model["params"].items()}, "ko": dict(model["ko"])}
    clone["ko"][name] = mode
    return clone


# ----------------------------------------------------------------- baselines and rival mechanisms
def trivial_r2(data, split):
    y_tr, y = data["train"][1], data[split][1]
    return r2(y, np.full_like(y, y_tr.mean()))


# ----------------------------------------------------------------- registries
MODEL_KINDS = {"xiantian": "the mind: tied doubling table plus shared of-kernel, no trend",
               "untied": "matched ablation: each scale its own table (observing through the self)",
               "fourier": "closest prior art as declared: Fourier seasonality on every period plus trend",
               "prophet": "prior art at its documented default: seasonality only for periods seen twice, "
                          "piecewise-linear trend with changepoints (added in card revision 2)"}
MUTANTS = {"sign_flip": "gradient sign reversed", "zero_lr": "learning rate set to zero",
           "zeroed_profile_grad": "gradient of the shared profile table zeroed",
           "zeroed_kernel_grad": "gradient of the coarse side of the of-kernel zeroed"}


# ----------------------------------------------------------------- training
def fit(model, data, budget, rng, lr=0.03, clip=5.0, eval_every=25):
    """Full-batch Adam; model selection on the validation split only."""
    Xtr, ytr = data["train"]
    Xva, yva = data["val"]
    opt = Adam(model["params"], lr=0.0 if ACTIVE_MUTANT == "zero_lr" else lr)
    best, best_params, hist = np.inf, None, []
    for step in range(budget):
        loss, grads = loss_and_grads(model, (Xtr, ytr))
        if not np.isfinite(loss):
            raise FloatingPointError("non-finite loss")
        grads, _ = clip_global_norm(grads, clip)
        opt.step(model["params"], grads)
        hist.append(loss)
        if step % eval_every == 0 or step == budget - 1:
            vl = float(np.mean((predict(model, Xva) - yva) ** 2))
            if vl < best:
                best, best_params = vl, {k: v.copy() for k, v in model["params"].items()}
    model["params"] = best_params
    parts = forward(model, Xtr)[1]
    model["stats"] = {k: v.mean(axis=0, keepdims=True) for k, v in parts.items()}
    return hist


def train_kind(kind, data, budget, seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed, sum(map(ord, kind))]))
    model = build_model(data["train"][0].shape[1], 1, "vector_regression", rng, kind=kind,
                        span=float(np.ptp(data["train"][0][:, 0])))
    hist = fit(model, data, budget, rng)
    return model, hist


# ----------------------------------------------------------------- tests: correctness
def c1_gradcheck(budget_steps, seed):
    rng = np.random.default_rng(seed)
    data = make_world(rng, "cycle", sizes=(96, 32, 32, 32))
    small = (data["train"][0][:64], data["train"][1][:64])
    ok_all, worst, tensors, total = True, 0.0, 0, 0
    for kind in MODEL_KINDS:
        model = build_model(2, 1, "vector_regression", np.random.default_rng(seed + 1), kind=kind)
        model["params"] = {k: v + 0.1 * rng.standard_normal(v.shape) for k, v in model["params"].items()}
        for stage in ("init", "after_training_steps"):
            if stage == "after_training_steps":
                fit(model, data, budget_steps, rng)
            _, g = loss_and_grads(model, small)
            ok, w, n = finite_difference_check(lambda: loss_and_grads(model, small)[0],
                                               model["params"], g, rng)
            ok_all, worst, tensors, total = ok_all and ok, max(worst, w), tensors + n, total + len(g)
    return {"passed": bool(ok_all), "max_rel_error": worst, "tensors_checked": tensors, "tensors_total": total,
            "checked_at": ["init", "after_training_steps"]}


def c2_determinism(seed):
    runs = []
    for _ in range(2):
        data = make_world(np.random.default_rng(seed), "cycle", sizes=(300, 60, 60, 60))
        model, hist = train_kind("xiantian", data, 80, seed)
        yhat = predict(model, data["test"][0])
        finite = all(np.all(np.isfinite(v)) for v in model["params"].values()) and np.all(np.isfinite(yhat))
        runs.append((np.array(hist), yhat, finite))
    same = np.array_equal(runs[0][0], runs[1][0]) and np.array_equal(runs[0][1], runs[1][1])
    return same and runs[0][2], "identical losses and outputs" if same else "runs differ"


def c3_learning(data, model, hist):
    drop = 1.0 - hist[-1] / hist[0]
    gain = r2(data["test"][1], predict(model, data["test"][0])) - trivial_r2(data, "test")
    th = MIND_CARD["thresholds"]
    return drop >= th["loss_drop_fraction"] and gain >= th["margin_over_trivial"], \
        "loss drop %.3f, test R2 over trivial %+.3f" % (drop, gain)


def c4_shuffled(data, budget, seed):
    rng = np.random.default_rng(seed + 404)
    shuffled = dict(data)
    shuffled["train"] = (data["train"][0], rng.permutation(data["train"][1]))
    shuffled["val"] = (data["val"][0], rng.permutation(data["val"][1]))
    model, _ = train_kind("xiantian", shuffled, budget, seed)
    got = r2(data["test"][1], predict(model, data["test"][0]))
    band = trivial_r2(data, "test") + MIND_CARD["thresholds"]["shuffled_band_over_trivial"]
    return got <= band, "shuffled-label test R2 %.3f (band <= %.3f)" % (got, band)


def c5_mutants(seed):
    global ACTIVE_MUTANT
    detected = {}
    data = make_world(np.random.default_rng(seed), "cycle", sizes=(400, 80, 80, 80))
    for name in MUTANTS:
        ACTIVE_MUTANT = name
        try:
            gc = c1_gradcheck(10, seed)["passed"]
            model, hist = train_kind("xiantian", data, 150, seed)
            learn = c3_learning(data, model, hist)[0]
        finally:
            ACTIVE_MUTANT = None
        detected[name] = (not gc) or (not learn)
    return detected


def c6_1_doubling_consistency(rng):
    """Mean of the leaves under any node equals the synthesis truncated at that node's depth."""
    def violation(split):
        D = 6
        H, lv = doubling_synthesis(D, split), coefficient_levels(D)
        worst = 0.0
        for _ in range(200):
            cvec = rng.normal(0.0, 1.0, size=2 ** D)
            leaves = H @ cvec
            for k in range(D + 1):
                coarse = (H @ np.where(lv <= k, cvec, 0.0)).reshape(2 ** k, -1)[:, 0]
                worst = max(worst, float(np.max(np.abs(leaves.reshape(2 ** k, -1).mean(1) - coarse))))
        return worst
    good, broken = violation((1.0, -1.0)), violation((1.0, 0.0))
    return good < 1e-10 and broken > 1e-3, "violation %.2e; unbalanced-split control %.2e" % (good, broken)


def c6_2_periodicity(rng):
    """No arrow of time: with events off, f(t + P_top) = f(t) for random parameters and times."""
    def worst_gap(kind):
        worst = 0.0
        for _ in range(30):
            model = build_model(2, 1, "vector_regression", rng, kind=kind)
            model["params"] = {k: rng.normal(0.0, 1.0, size=v.shape) for k, v in model["params"].items()}
            t = rng.uniform(0.0, 5.0 * PERIODS[-1], size=400)
            X0, X1 = np.stack([t, np.zeros_like(t)], 1), np.stack([t + PERIODS[-1], np.zeros_like(t)], 1)
            worst = max(worst, float(np.max(np.abs(predict(model, X1) - predict(model, X0)))))
        return worst
    good, control = max(worst_gap("xiantian"), worst_gap("untied")), worst_gap("fourier")
    return good < 1e-8 and control > 1e-3, "gap %.2e; trend-baseline control %.2e" % (good, control)


def c6_3_order_sensitivity(rng):
    """The of-kernel distinguishes 'u of v' from 'v of u'; a symmetric tie cannot."""
    model = build_model(2, 1, "vector_regression", rng, kind="xiantian")
    Hk, share, sym_worst = model["cfg"]["Hk"], 0, 0.0
    for _ in range(200):
        P, Q = rng.normal(size=(2, Hk.shape[0])), rng.normal(size=(2, Hk.shape[0]))
        Kmat = (Hk @ P.T) @ (Hk @ Q.T).T
        share += float(np.max(np.abs(Kmat - Kmat.T))) > 1e-3
        Ksym = (Hk @ P.T) @ (Hk @ P.T).T
        sym_worst = max(sym_worst, float(np.max(np.abs(Ksym - Ksym.T))))
    return share >= 190 and sym_worst < 1e-10, "asymmetric in %d/200 draws; symmetric control %.1e" % (
        share, sym_worst)


def d1_tie_definition(rng):
    """Definition check (not evidence): every scale reads the same table at the same phase."""
    def gap(kind):
        model = build_model(2, 1, "vector_regression", rng, kind=kind)
        model["params"] = {k: rng.normal(size=v.shape) for k, v in model["params"].items()}
        phi = rng.uniform(0.0, 1.0, size=500)
        G = [_parts_doubling(model, np.stack([phi * P, np.zeros_like(phi)], 1))[1]["G"][:, s]
             for s, P in enumerate(PERIODS)]
        return float(np.max(np.abs(G[2] - G[0])))
    return gap("xiantian") < 1e-12 and gap("untied") > 1e-3


def c7_split_integrity(data):
    sets = [set(data["index"][k].tolist()) for k in ("train", "val", "test", "shift")]
    disjoint = all(not (a & b) for i, a in enumerate(sets) for b in sets[i + 1:])
    t_in = np.concatenate([data[k][0][:, 0] for k in ("train", "val", "test")])
    beyond = float(data["shift"][0][:, 0].min()) >= data["meta"]["t_obs"] > float(t_in.max())
    return disjoint and beyond, "indices disjoint; shifted split starts at t=%.1f" % data["meta"]["t_obs"]


# ----------------------------------------------------------------- tests: hypotheses
def run_seed(seed, budget):
    cyc = make_world(np.random.default_rng(np.random.SeedSequence([seed, 1])), "cycle")
    arr = make_world(np.random.default_rng(np.random.SeedSequence([seed, 2])), "arrow")
    out, models = {}, {}
    for kind in MODEL_KINDS:
        models[kind], hist = train_kind(kind, cyc, budget, seed)
        for split in ("test", "shift"):
            out["cycle/%s/%s" % (kind, split)] = r2(cyc[split][1], predict(models[kind], cyc[split][0]))
        if kind == "xiantian":
            out["hist"] = hist
    for kind in ("xiantian", "fourier", "prophet"):
        model, _ = train_kind(kind, arr, budget, seed)
        for split in ("test", "shift"):
            out["arrow/%s/%s" % (kind, split)] = r2(arr[split][1], predict(model, arr[split][0]))
    for name in modules(models["xiantian"]):
        ko = knockout(models["xiantian"], name, "zero")
        out["ko/" + name] = r2(cyc["shift"][1], predict(ko, cyc["shift"][0])) - out["cycle/xiantian/shift"]
    out["trivial/test"], out["trivial/shift"] = trivial_r2(cyc, "test"), trivial_r2(cyc, "shift")
    return out, cyc, models


def hypothesis_table(per_seed, rng):
    get = lambda key: np.array([o[key] for o in per_seed])
    diffs = {"H-SIG": get("cycle/xiantian/shift") - get("cycle/fourier/shift"),
             "H-FANGUAN": get("cycle/xiantian/shift") - get("cycle/untied/shift"),
             "H-NEC": get("ko/of_kernel") - get("ko/exo_head"),
             "H-BLIND": get("arrow/xiantian/shift") - get("arrow/fourier/shift"),
             "H-SIG.b": get("cycle/xiantian/shift") - get("cycle/prophet/shift"),
             "H-BLIND.b": get("arrow/xiantian/shift") - get("arrow/prophet/shift")}
    rows = []
    for h in MIND_CARD["hypotheses"]:
        mean, ci = paired_bootstrap(diffs[h["id"]], rng)
        rows.append({"id": h["id"], "metric": h["metric"] + "/" + h.get("split", "shifted"),
                     "mean_diff": mean, "ci95": ci, "mesi": h["mesi"], "n_seeds": len(per_seed),
                     "verdict": verdict(mean, ci, h["direction"], h["mesi"])})
    return rows


# ----------------------------------------------------------------- report
def data_bridge(path, budget, seed):
    if not path:
        return "skipped (no --data PATH given)"
    periods, rows = PERIODS, []
    with open(path, encoding="utf-8") as fh:
        for line in (ln.strip() for ln in fh):
            if line.startswith("# periods:"):
                periods = tuple(float(v) for v in line.split(":")[1].split(","))
            elif line and not line.startswith("#"):
                try:
                    rows.append([float(v) for v in line.split(",")])
                except ValueError:
                    continue                    # header row
    arr = np.asarray(rows, dtype=np.float64)
    arr = arr[np.argsort(arr[:, 0])]
    X = np.concatenate([arr[:, :1], arr[:, 2:]], axis=1) if arr.shape[1] > 2 else arr[:, :1]
    cut = int(0.7 * len(arr))
    data = {"train": (X[:int(0.85 * cut)], arr[:int(0.85 * cut), 1]), "val": (X[int(0.85 * cut):cut], arr[int(0.85 * cut):cut, 1])}
    res = {}
    for kind in ("xiantian", "prophet"):
        rng = np.random.default_rng(seed)
        model = build_model(X.shape[1], 1, "vector_regression", rng, kind=kind, periods=periods,
                            span=float(np.ptp(X[:cut, 0])))
        fit(model, data, budget, rng)
        res[kind] = round(r2(arr[cut:, 1], predict(model, X[cut:])), 4)
    return "held-out last 30%% R2: xiantian %.4f, prophet_default %.4f" % (res["xiantian"], res["prophet"])


def print_report(rep):
    print("=== VERIFIED REPORT · chapter 0263 ===")
    print("file: %s   card_revision: %d" % (rep["file"], rep["card_revision"]))
    print("environment: python %s, numpy %s" % (rep["environment"]["python"], rep["environment"]["numpy"]))
    print("seeds: %s   runtime: %.1f s   exit code: %d" % (rep["seeds"], rep["runtime_s"], rep["exit_code"]))
    print("parameters: " + ", ".join("%s %d" % kv for kv in rep["n_params_by_model"].items()))
    gc = rep["gradcheck"]
    print("gradient check: %d/%d tensors at init and after training, max rel error %.2e, %s" % (
        gc["tensors_checked"], gc["tensors_total"], gc["max_rel_error"], "PASS" if gc["passed"] else "FAIL"))
    print("correctness:")
    for c in rep["correctness"]:
        print("  %-5s %-28s %s  %s" % (c["id"], c["name"], "PASS" if c["passed"] else "FAIL", c["detail"]))
    mu = rep["mutants"]
    print("mutation score: %d/%d = %.2f" % (mu["detected"], mu["total"], mu["score"]))
    print("hypotheses (paired over seeds, 95% bootstrap CI):")
    for h in rep["hypotheses"]:
        print("  %-10s %-10s mean %+.4f  CI [%+.4f, %+.4f]  mesi %.2f  -> %s" % (
            h["id"], h["metric"], h["mean_diff"], h["ci95"][0], h["ci95"][1], h["mesi"], h["verdict"]))
    print("knockouts (change in shifted R2 of the tied clock, zero mode):")
    for k in rep["knockouts"]:
        print("  %-18s signature=%-5s %+.4f  CI [%+.4f, %+.4f]" % (
            k["module"], k["signature"], k["metric_change"], k["ci95"][0], k["ci95"][1]))
    print("mean R2 by model and split:")
    for key, val in rep["mean_r2"].items():
        print("  %-28s %+.4f" % (key, val))
    print("real-data bridge: " + rep["real_data_bridge"])
    print("task types: %s" % rep["task_types"])
    print("=== END REPORT ===")


def main(argv=None):
    global ACTIVE_MUTANT
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=263)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--json", default=None)
    ap.add_argument("--card", action="store_true")
    ap.add_argument("--mutant", default=None)
    ap.add_argument("--data", default=None)
    args = ap.parse_args(argv)
    if args.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    if (args.mutant and args.mutant not in MUTANTS) or args.seeds < 1:
        print("usage: python3 %s [--quick] [--seeds N>=1] [--mutant %s]" % (
            os.path.basename(__file__), "|".join(MUTANTS)), file=sys.stderr)
        return 2
    t0 = time.time()
    n_seeds = 1 if args.quick else args.seeds
    budget = 500 if args.quick else 1500
    seeds = [args.seed + i for i in range(n_seeds)]
    print("seeds: %s  budget: %d updates per model" % (seeds, budget))
    np.seterr(over="raise", invalid="raise", divide="raise")
    try:
        mutants = {} if args.mutant else c5_mutants(args.seed)
        ACTIVE_MUTANT = args.mutant
        gc = c1_gradcheck(60, args.seed)
        per_seed, first = [], None
        for s in seeds:
            out, cyc, models = run_seed(s, budget)
            per_seed.append(out)
            first = first or (out, cyc, models)
        out0, cyc0, models0 = first
        c3 = c3_learning(cyc0, models0["xiantian"], out0["hist"])
        c4 = c4_shuffled(cyc0, budget, seeds[0])
        c2 = c2_determinism(seeds[0])
        prop_rng = np.random.default_rng(args.seed + 6)
        c61, c62, c63 = c6_1_doubling_consistency(prop_rng), c6_2_periodicity(prop_rng), c6_3_order_sensitivity(prop_rng)
        d1 = d1_tie_definition(prop_rng)
        c7 = c7_split_integrity(cyc0)
        bridge = data_bridge(args.data, budget, args.seed)
    except FloatingPointError as err:
        print("non-finite values: %s" % err, file=sys.stderr)
        return 4
    runtime = time.time() - t0
    limit = 20.0 if args.quick else 180.0
    detected = sum(mutants.values())
    correctness = [
        {"id": "C1", "name": "gradient_check", "passed": gc["passed"], "detail": "max rel %.2e" % gc["max_rel_error"]},
        {"id": "C2", "name": "determinism_finiteness", "passed": c2[0], "detail": c2[1]},
        {"id": "C3", "name": "learning", "passed": c3[0], "detail": c3[1]},
        {"id": "C4", "name": "shuffled_label_control", "passed": c4[0], "detail": c4[1]},
        {"id": "C5", "name": "mutant_detection", "passed": detected == len(MUTANTS) or bool(args.mutant),
         "detail": ", ".join("%s:%s" % (k, "caught" if v else "missed") for k, v in mutants.items())
         or "skipped: running under --mutant " + str(args.mutant)},
        {"id": "C6.1", "name": "doubling_consistency", "passed": c61[0], "detail": c61[1]},
        {"id": "C6.2", "name": "periodicity_no_arrow", "passed": c62[0], "detail": c62[1]},
        {"id": "C6.3", "name": "of_kernel_order", "passed": c63[0], "detail": c63[1]},
        {"id": "C7", "name": "split_integrity", "passed": c7[0], "detail": c7[1]},
        {"id": "C8", "name": "budget", "passed": runtime <= limit, "detail": "%.1f s of %.0f s" % (runtime, limit)},
        {"id": "D-1", "name": "tie_definition_check", "passed": d1, "detail": "labelled definition, not evidence"},
    ]
    boot = np.random.default_rng(args.seed + 2000)
    hyps = hypothesis_table(per_seed, boot) if n_seeds >= 5 else [
        {"id": h["id"], "metric": h["metric"], "mean_diff": 0.0, "ci95": [0.0, 0.0], "mesi": h["mesi"],
         "n_seeds": n_seeds, "verdict": "not evaluated"} for h in MIND_CARD["hypotheses"]]
    kos = []
    for name, info in modules(models0["xiantian"]).items():
        mean, ci = paired_bootstrap([o["ko/" + name] for o in per_seed], boot)
        kos.append({"module": name, "signature": info["signature"], "metric_change": mean, "ci95": ci})
    keys = [k for k in per_seed[0] if "/" in k and not k.startswith("ko/")]
    for c in correctness:
        c["passed"] = bool(c["passed"])
    exit_code = 0 if all(c["passed"] for c in correctness if c["id"] != "C8") else 1
    if exit_code == 0 and runtime > limit:
        exit_code = 3
    rep = {"schema_version": "1.0", "chapter": 263, "file": os.path.basename(__file__),
           "card_revision": MIND_CARD["card_revision"],
           "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
           "seeds": seeds, "runtime_s": round(runtime, 2), "n_params": n_params(models0["xiantian"]),
           "n_params_by_model": {k: n_params(v) for k, v in models0.items()},
           "gradcheck": gc, "correctness": correctness,
           "mutants": {"detected": int(detected), "total": len(MUTANTS), "score": float(detected) / len(MUTANTS),
                       "detail": {k: bool(v) for k, v in mutants.items()}},
           "hypotheses": hyps, "knockouts": kos,
           "mean_r2": {k: float(np.mean([o[k] for o in per_seed])) for k in keys},
           "real_data_bridge": bridge, "task_types": TASK_TYPES, "exit_code": exit_code,
           "sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest()}
    print_report(rep)
    if args.json:
        write_report(rep, args.json)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

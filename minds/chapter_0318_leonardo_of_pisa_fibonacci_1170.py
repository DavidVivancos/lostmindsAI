#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0318 · Leonardo of Pisa (Fibonacci)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 16, Minds 301-320 Available on Amazon https://www.amazon.com/gp/product/B0HNJQX9S4
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos 
# END ATTRIBUTION
"""
Chapter 0318 · Leonardo of Pisa (Fibonacci), c. 1170 - c. 1240/50 · the Elchataym Engine.

Thesis
  Reach an unknown without ever naming it: posit two convenient wrong values, run each
  through a learned multiply-and-shift rule of the riddle, and leap along the line their
  two errors define (regula elchataym, Liber abbaci chapter 13).

Evidence and provenance (provenance: belief - his own books survive; no treatise on mind
does, so the doctrine column is read from his working method, not from stated doctrine)
  D1 primary      Chapter 13 is devoted to double false position, named after the Arabic
                  al-khata'ayn; he usually works a variant that measures how much closer
                  the second posit comes to the truth (Sigler 2002; Vogel 1971).
  D2 scholarship  In single false position he posits a convenient number - often the least
                  common multiple of the denominators, or 1 - and rescales by proportion
                  (Hannah 2007; Hoyrup 2014).
  D3 primary      Chapters 8-11 price goods, barter, split company profits and assay alloys
                  with one proportion, indifferent to the commodity (Sigler 2002).
  D4 primary      Where the unknown multiplies itself the book turns to al-jabr (ch. 15);
                  the cubic sent to Frederick II in Flos received only a sexagesimal
                  approximation (Vogel 1971).

Doctrine -> mechanism -> test (IDs as in MIND_CARD)
  D1 -> M3 two_error_inverter: two posits, one leap          -> C6.2 -> H-SIG, H-SIG-B
  D2 -> M3 convenient posits of 12 and 24 denari (1, 2 soldi) -> C6.2 -> H-SIG
  D3 -> M1 slope_head + M2 offset_head read the riddle's shape;
        the costume (commodity) is nuisance input            -> C6.1 -> H-NEC
  D4 -> blind spot: riddles in which the purse is squared     -> H-BLIND

Research question (out-of-distribution detection and shift)
  Trained only on forward worked examples (purse in -> purse out), can a model answer
  inverse questions whose answers lie 2-8x beyond every worked purse?

Closest prior art and the delta
  Feature-wise affine conditioning (FiLM, Perez et al. 2018); learned simulators inverted
  by amortized networks or root finding (Cranmer, Brehmer and Louppe 2020); the secant /
  regula falsi iteration itself.  Delta: the queried unknown never enters a nonlinearity -
  it is only multiplied and shifted by coefficients read from the riddle - and inverse
  queries are answered by two posited errors, so all learning lives in the context and the
  answer extrapolates in the unknown.  Overlap: Medium.  Baselines in this file: a
  size-matched amortized inverse network (A) and a size-matched generic forward surrogate
  inverted by the same two-error iteration (B).

Blind spot
  When the purse multiplies itself (usury proportional to the purse) the riddle is no longer
  multiply-and-shift; two errors no longer pin the answer, and a flexible network trained on
  the same worked cases should beat the engine inside the training range.

Task (journeys to the markets)
  A merchant leaves with x denari and visits k in {1, 2, 3} markets. At market j the purse
  becomes m_j * p - t_j (+ u_j * p^2 / 30 in the blind-spot family only); a fifth of the
  riddles carry no tolls. Each riddle wears one of six costumes (cloth, pepper, silver alloy,
  company, horses, purse) and carries three ink-noise features; both are irrelevant.
  Training: x in [1, 30] denari with noisy final purses. Held-out: new riddles, same range.
  Shifted: x in [60, 240]. Inverse query: given a riddle and its exact final purse, find x.

Limits
  Synthetic commercial riddles, one scalar unknown, affine in the unknown by design; the
  advantage on the shifted split is an inductive bias, and what the tests put at risk is
  whether the riddle-dependent coefficients are learned well enough from forward cases
  alone. Research prototype of one mechanism, not an AGI and not a replica of a person.
  Interface note: the engine reads column 0 of X as the queried scalar, the rest as context.

Run: python3 <this file>   [--quick] [--seed S] [--seeds N] [--json PATH] [--card]
                           [--mutant NAME] [--data PATH]
"""

MIND_CARD = {
    "schema_version": "1.0", "card_revision": 1,
    "revision_log": ["2026-10-17: card written first; one single-seed smoke run of the training code only (before the "
                     "test battery existed) checked convergence; no hypothesis, metric, threshold or mesi changed after it"],
    "generation": {"template_version": "codeguidelines 1.0 (2026-09-15)", "generator": "Claude (Anthropic)",
                   "generator_version": "claude-opus-5-5 (configured identifier)", "date": "2026-10-17"},
    "id": 318, "figure": "Leonardo of Pisa (Fibonacci)", "born": 1170, "died": 1240,
    "civilization": "Italian (Republic of Pisa)", "provenance": "belief",
    "thesis": "Reach an unknown without naming it: posit two convenient wrong values, run each through "
              "a learned multiply-and-shift rule of the riddle, and leap along the line of their two errors.",
    "evidence": [
        {"id": "D1", "claim": "Chapter 13 of Liber abbaci is devoted to double false position (regula elchataym), "
         "usually worked as a measure of how much closer the second posit comes to the truth",
         "basis": "primary", "source": "Liber abbaci ch. 13 (Sigler 2002); Vogel 1971, DSB 4"},
        {"id": "D2", "claim": "Single false position posits a convenient number (often the LCM of the denominators, "
         "or 1) and rescales by proportion", "basis": "scholarship", "source": "Hannah 2007; Hoyrup 2014"},
        {"id": "D3", "claim": "Prices, barter, companies and alloys are all solved by one proportion",
         "basis": "primary", "source": "Liber abbaci chs. 8-11 (Sigler 2002)"},
        {"id": "D4", "claim": "Problems where the unknown multiplies itself go to al-jabr; the Flos cubic gets only "
         "a sexagesimal approximation", "basis": "primary", "source": "Liber abbaci ch. 15; Flos (Vogel 1971)"},
    ],
    "research_question": {"category": "out-of-distribution detection and shift",
                          "question": "Can a model trained only on forward worked examples answer inverse queries "
                                      "whose answers lie 2-8x beyond every worked example?"},
    "mechanism": {
        "name": "Elchataym Engine", "family": "model-based inversion with an affine-in-query inductive bias",
        "signature_modules": ["slope_head", "offset_head", "two_error_inverter"],
        "closest_prior_art": ["FiLM feature-wise affine conditioning (Perez et al. 2018)",
                              "simulation-based inference: amortized inverse networks and surrogate inversion "
                              "(Cranmer, Brehmer and Louppe 2020)", "secant / regula falsi root finding"],
        "overlap": "Medium",
        "prior_art_queries": ["affine conditioning query variable extrapolation", "amortized inverse network "
                              "extrapolation", "learned surrogate secant inversion", "FiLM conditioning"],
        "contribution_type": "mechanism",
        "delta": "The unknown enters only through a context-conditioned multiply-and-shift and is recovered from "
                 "two posited errors, so learning stays in the context and answers extrapolate in the unknown."},
    "traceability": [
        {"doctrine": "D1", "mechanism": "M3", "property_test": "C6.2", "hypothesis": "H-SIG"},
        {"doctrine": "D2", "mechanism": "M3", "property_test": "C6.2", "hypothesis": "H-SIG-B"},
        {"doctrine": "D3", "mechanism": "M1+M2", "property_test": "C6.1", "hypothesis": "H-NEC"},
        {"doctrine": "D4", "mechanism": "M1+M2", "property_test": "C6.1", "hypothesis": "H-BLIND"}],
    "hypotheses": [
        {"id": "H-SIG", "statement": "On purses 2-8x beyond training the engine recovers x with lower error than a "
         "size-matched amortized inverse network", "metric": "capped mean absolute relative error (mare)",
         "split": "shifted", "comparison": "model - baseline_A", "direction": "less", "mesi": 0.10, "seeds": 5},
        {"id": "H-SIG-B", "statement": "The gain comes from the multiply-and-shift bias, not from the inversion "
         "procedure: same two-error iteration on a generic surrogate does worse", "metric": "mare",
         "split": "shifted", "comparison": "model - baseline_B", "direction": "less", "mesi": 0.10, "seeds": 5},
        {"id": "H-NEC", "statement": "Zeroing the offset head (collapsing to the rule of three) hurts more than "
         "mean-replacing the costume encoder", "metric": "mare", "split": "shifted",
         "comparison": "signature_knockout - matched_knockout", "direction": "greater", "mesi": 0.10, "seeds": 5},
        {"id": "H-BLIND", "statement": "On usury riddles (purse squared) the engine is worse than the amortized "
         "inverse network inside the training range", "condition": "usury family, held-out split",
         "grounding": "false position is exact only for linear riddles; squares go to al-jabr and the Flos cubic "
         "defeated exact methods", "metric": "mare", "comparison": "model - baseline_A",
         "direction": "greater", "mesi": 0.05, "seeds": 5}],
    "thresholds": {"loss_drop_fraction": 0.90, "margin_over_trivial": 0.50, "shuffled_band": 0.75},
    "probe_predictions": [{"probe": "P2", "expected": "equal to baseline"}, {"probe": "P4", "expected": "equal to baseline"},
                          {"probe": "P8", "expected": "below baseline"}, {"probe": "P10", "expected": "above baseline"}],
    "dialectic_links": [
        {"chapter": 213, "relation": "predecessor", "test": "none (al-jabr, which ch. 15 adapts, not implemented)"},
        {"chapter": 276, "relation": "predecessor", "test": "H-BLIND (the Flos cubic comes from al-Khayyami)"}],
    "corpus_neighbors": [
        {"chapter": 213, "similarity": None, "difference": "al-Khwarizmi reduces a problem to a named canonical form; "
         "here the unknown is never named, only probed with two wrong values"},
        {"chapter": 295, "similarity": None, "difference": "Bhaskara II defers dead ends; here no dead end is deferred, "
         "two errors are combined at once"},
        {"chapter": 302, "similarity": None, "difference": "al-Jazari composes a standard part library; here one rule "
         "shape is read off each riddle"},
        {"chapter": 324, "similarity": None, "difference": "Li Ye names the unknown (tianyuan) before computing; here the "
         "unknown stays unnamed"}],
    "gate_notes": ["Jaccard similarity not computed: corpus files were not available in this session",
                   "Batch rule: neighbours read from figures_master.json records 298-317, not from their MIND_CARDs"],
    "barometer": {"cognitive_processing": ["shifted-split inverse queries", "P10 sample efficiency"],
                  "embodied_cognition": [], "world_modeling": ["forward rule extrapolated to unseen purse sizes"],
                  "consciousness": [], "language_understanding": [], "emotional_intelligence": [],
                  "creativity": [], "autonomy": []},
    "task_types": ["vector_regression"],
    "applications": [
        {"use": "back-solving opening balances from closing balances under fees and growth", "sector": "finance",
         "dataset": "LendingClub public loan data 2007-2018"},
        {"use": "inferring initial state of charge from end-of-cycle measurements", "sector": "energy",
         "dataset": "NASA Ames PCoE Li-ion battery aging data"},
        {"use": "reconstructing opening inventory from closing counts through stock flows", "sector": "retail logistics",
         "dataset": "M5 Forecasting (Walmart) competition data"}],
    "safety_notes": "Commercial arithmetic only; synthetic data; no data about people; research prototype.",
}

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

XS = 30.0                      # 30 denari: ceiling of the worked purses, used as the unit scale
N_MARKETS = 3
COSTUMES = ("cloth", "pepper", "silver alloy", "company", "horses", "purse")
RIDDLE_COLS = np.arange(0, 4 * N_MARKETS)                     # m, t, u, active per market
COSTUME_COLS = np.arange(4 * N_MARKETS, 4 * N_MARKETS + len(COSTUMES) + 3)   # costume + ink noise
CTX_DIM = 4 * N_MARKETS + len(COSTUMES) + 3
POSITS = (12.0, 24.0)          # one soldo and two soldi: Leonardo's habit of convenient posits
TRAIN_RANGE, SHIFT_RANGE = (1.0, 30.0), (60.0, 240.0)
HIDDEN, LR, BATCH, CLIP = 32, 3e-3, 128, 5.0
TASK_TYPES = ["vector_regression"]
MUTANTS = {"sign_flip_update": "Adam step applied with reversed sign",
           "zero_lr": "learning rate set to zero",
           "zero_grad_offset_head": "gradient of the offset head (Wb, bb) zeroed"}
ACTIVE_MUTANT = None


class NonFinite(RuntimeError):
    pass


# BEGIN STANDARD UTILITIES v1.0
def softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def logsumexp(z, axis=-1):
    m = np.max(z, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(z - m), axis=axis))


def softplus(z):
    return np.logaddexp(0.0, z)


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
    norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    if norm > max_norm:
        grads = {k: g * (max_norm / norm) for k, g in grads.items()}
    return grads, norm


def finite_difference_check(loss_fn, params, grads, rng, n_random=20, eps=1e-6, floor=1e-4):
    """Central differences on every tensor: all entries of small tensors, otherwise n_random
    random entries plus the largest-gradient entry. Relative error |a-n| / max(|a|+|n|, floor);
    the floor keeps entries with |g| < 1e-4 from being judged on float64 rounding noise."""
    report = {}
    for name, p in params.items():
        flat = p.reshape(-1)
        g = grads[name].reshape(-1)
        if flat.size <= n_random:
            idx = np.arange(flat.size)
        else:
            idx = np.unique(np.append(rng.choice(flat.size, n_random, replace=False), np.argmax(np.abs(g))))
        worst = 0.0
        for i in idx:
            old = flat[i]
            flat[i] = old + eps
            lp = loss_fn()
            flat[i] = old - eps
            lm = loss_fn()
            flat[i] = old
            num = (lp - lm) / (2 * eps)
            worst = max(worst, abs(g[i] - num) / max(abs(g[i]) + abs(num), floor))
        report[name] = {"checked": int(len(idx)), "max_rel": float(worst)}
    return report


def paired_bootstrap(diffs, rng, n_boot=2000, alpha=0.05):
    d = np.asarray(diffs, dtype=float)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def write_report(lines, payload=None, path=None):
    print("\n".join(lines))
    if path and payload is not None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
# END STANDARD UTILITIES


# ----------------------------------------------------------------------------- data and tasks
def journey(x, m, t, u):
    """The riddle itself: carry the purse through each market in order."""
    p = np.array(x, dtype=float)
    for j in range(m.shape[1]):
        p = m[:, j] * p - t[:, j] + u[:, j] * p * p / XS
    return p


def make_riddles(n, rng, x_range, family="affine"):
    k = rng.integers(1, N_MARKETS + 1, size=n)
    active = (np.arange(N_MARKETS)[None, :] < k[:, None]).astype(float)
    m = 1.0 + active * (rng.uniform(1.2, 2.4, (n, N_MARKETS)) - 1.0)
    tolls_on = (rng.random(n) >= 0.2).astype(float)[:, None]
    t = active * tolls_on * rng.uniform(0.0, 12.0, (n, N_MARKETS))
    u = active * rng.uniform(0.01, 0.04, (n, N_MARKETS)) if family == "usury" else np.zeros((n, N_MARKETS))
    costume = np.eye(len(COSTUMES))[rng.integers(0, len(COSTUMES), n)]
    ink = rng.normal(0.0, 1.0, (n, 3))
    x = rng.uniform(x_range[0], x_range[1], size=n)
    ctx = np.concatenate([m - 1.0, t / 12.0, u * 25.0, active, costume, ink], axis=1)
    return {"x": x, "y": journey(x, m, t, u), "ctx": ctx}


def make_task(rng, family="affine", n_train=4000, n_eval=600):
    seeds = np.random.SeedSequence(int(rng.integers(0, 2 ** 31))).spawn(4)
    r = [np.random.default_rng(s) for s in seeds]
    data = {"train": make_riddles(n_train, r[0], TRAIN_RANGE, family),
            "val": make_riddles(n_eval, r[1], TRAIN_RANGE, family),
            "test": make_riddles(n_eval, r[2], TRAIN_RANGE, family),
            "shift": make_riddles(n_eval, r[3], SHIFT_RANGE, family)}
    tr = data["train"]
    tr["y_obs"] = tr["y"] + r[0].normal(0.0, 0.05, size=tr["y"].shape)   # a scribe's rounding
    return data


def forward_arrays(split):
    y = split.get("y_obs", split["y"])
    return np.column_stack([split["x"] / XS, split["ctx"]]), (y / XS)[:, None]


def inverse_arrays(split):
    y = split.get("y_obs", split["y"])
    return np.column_stack([y / XS, split["ctx"]]), (split["x"] / XS)[:, None]


def mare(x_hat, x):
    """Capped mean absolute relative error (cap 10 keeps wild answers finite)."""
    return float(np.mean(np.minimum(np.abs(x_hat - x) / np.abs(x), 10.0)))


# ----------------------------------------------------------------------------- model
def _init(rng, n_in, n_out):
    return rng.uniform(-1, 1, (n_in, n_out)) * math.sqrt(6.0 / (n_in + n_out))


def build_model(in_dim, out_dim, task_type, rng, kind="engine", hidden=HIDDEN, **cfg):
    """kind='engine': y = a(context) * q + b(context), q = column 0 of X.
    kind='mlp'   : two tanh layers over the whole input (baselines A and B)."""
    if task_type not in TASK_TYPES:
        raise ValueError(f"unsupported task type {task_type}")
    if kind == "engine":
        c = in_dim - 1
        P = {"W1": _init(rng, c, hidden), "b1": np.zeros(hidden), "W2": _init(rng, hidden, hidden),
             "b2": np.zeros(hidden), "Wa": _init(rng, hidden, out_dim), "ba": np.ones(out_dim),
             "Wb": _init(rng, hidden, out_dim), "bb": np.zeros(out_dim)}
    else:
        P = {"W1": _init(rng, in_dim, hidden), "b1": np.zeros(hidden), "W2": _init(rng, hidden, hidden),
             "b2": np.zeros(hidden), "W3": _init(rng, hidden, out_dim), "b3": np.zeros(out_dim)}
    layout = "riddles" if (kind == "engine" and in_dim - 1 == CTX_DIM) else "generic"
    return {"kind": kind, "params": P, "knock": {}, "layout": layout, "cfg": dict(cfg),
            "ctx_mean": None, "a_mean": None}


def _engine_forward(model, X):
    P, ko = model["params"], model["knock"]
    q, C = X[:, :1], X[:, 1:].copy()
    if model["layout"] == "riddles":
        for name, cols in (("costume_encoder", COSTUME_COLS), ("riddle_encoder", RIDDLE_COLS)):
            if ko.get(name) == "mean":
                C[:, cols] = model["ctx_mean"][cols]
    h1 = np.tanh(C @ P["W1"] + P["b1"])
    h2 = np.tanh(h1 @ P["W2"] + P["b2"])
    a = h2 @ P["Wa"] + P["ba"]
    b = h2 @ P["Wb"] + P["bb"]
    if ko.get("slope_head") == "mean":
        a = np.broadcast_to(model["a_mean"], a.shape)
    if ko.get("offset_head") == "zero":
        b = np.zeros_like(b)
    return a * q + b, (C, h1, h2, a, q)


def _mlp_forward(model, X):
    P = model["params"]
    h1 = np.tanh(X @ P["W1"] + P["b1"])
    h2 = np.tanh(h1 @ P["W2"] + P["b2"])
    return h2 @ P["W3"] + P["b3"], (X, h1, h2)


def predict(model, X):
    return (_engine_forward if model["kind"] == "engine" else _mlp_forward)(model, X)[0]


def hidden_states(model, X):
    return (_engine_forward if model["kind"] == "engine" else _mlp_forward)(model, X)[1][2]


def loss_and_grads(model, batch):
    X, Y = batch
    P = model["params"]
    if model["kind"] == "engine":
        yhat, (C, h1, h2, a, q) = _engine_forward(model, X)
    else:
        yhat, (C, h1, h2) = _mlp_forward(model, X)
    diff = yhat - Y
    loss = float(np.mean(diff ** 2))
    dy = 2.0 * diff / diff.size
    g = {}
    if model["kind"] == "engine":
        da, db = dy * q, dy
        g["Wa"], g["ba"] = h2.T @ da, da.sum(0)
        g["Wb"], g["bb"] = h2.T @ db, db.sum(0)
        dh2 = da @ P["Wa"].T + db @ P["Wb"].T
        if ACTIVE_MUTANT == "zero_grad_offset_head":
            g["Wb"], g["bb"] = np.zeros_like(g["Wb"]), np.zeros_like(g["bb"])
    else:
        g["W3"], g["b3"] = h2.T @ dy, dy.sum(0)
        dh2 = dy @ P["W3"].T
    dz2 = dh2 * (1.0 - h2 ** 2)
    g["W2"], g["b2"] = h1.T @ dz2, dz2.sum(0)
    dz1 = (dz2 @ P["W2"].T) * (1.0 - h1 ** 2)
    g["W1"], g["b1"] = C.T @ dz1, dz1.sum(0)
    return loss, g


def n_params(model):
    return int(sum(p.size for p in model["params"].values()))


def _check_finite(model, *arrays):
    if not all(np.all(np.isfinite(p)) for p in list(model["params"].values()) + list(arrays)):
        raise NonFinite(f"non-finite values in {model['kind']}")


def fit(model, data, budget, rng):
    """Adam on minibatches for `budget` updates; returns full-training-set losses (start, end, trace)."""
    X, Y = data
    lr = 0.0 if ACTIVE_MUTANT == "zero_lr" else LR
    sign = -1.0 if ACTIVE_MUTANT == "sign_flip_update" else 1.0
    opt = Adam(model["params"], lr=lr)
    trace = [loss_and_grads(model, (X, Y))[0]]
    for step in range(budget):
        idx = rng.integers(0, len(X), size=BATCH)
        _, g = loss_and_grads(model, (X[idx], Y[idx]))
        g, _ = clip_global_norm(g, CLIP)
        opt.step(model["params"], g, sign=sign)
        if (step + 1) % max(1, budget // 10) == 0:
            trace.append(loss_and_grads(model, (X, Y))[0])
    model["ctx_mean"] = X[:, 1:].mean(axis=0)
    if model["kind"] == "engine":
        model["a_mean"] = _engine_forward(model, X)[1][3].mean(axis=0)
    _check_finite(model, np.array(trace))
    return trace


# ----------------------------------------------------------------------------- inversion (signature M3)
def forward_fn(model, ctx, sx=XS, sy=XS):
    """The learned riddle as a function of the purse alone, for one batch of contexts."""
    return lambda x: predict(model, np.column_stack([x / sx, ctx]))[:, 0] * sy


def invert_two_errors(f, y_target, posits=POSITS, steps=8):
    """Regula elchataym on a learned forward rule f: posit two purses, measure the two
    errors, leap along their line; repeat while the error has not vanished."""
    x1, x2 = np.full(len(y_target), posits[0]), np.full(len(y_target), posits[1])
    e1, e2 = f(x1) - y_target, f(x2) - y_target
    for _ in range(steps):
        live = np.abs(e2) > 1e-10 * (1.0 + np.abs(y_target))
        d = e2 - e1
        d = np.where(np.abs(d) < 1e-12, np.where(d >= 0, 1e-12, -1e-12), d)
        x1, e1, x2 = x2, e2, np.where(live, x2 - e2 * (x2 - x1) / d, x2)
        e2 = f(x2) - y_target
    return x2


def invert_single_posit(f, y_target, posit=POSITS[0]):
    """Single false position (the rule of three): exact only when the riddle has no toll."""
    g = f(np.full(len(y_target), posit))
    return posit * y_target / np.where(np.abs(g) < 1e-12, 1e-12, g)


def answer(model, split):
    """Recover the starting purse for each riddle of a split, given its exact final purse."""
    if model["kind"] == "mlp" and model["cfg"].get("role") == "inverse":
        return predict(model, np.column_stack([split["y"] / XS, split["ctx"]]))[:, 0] * XS
    f = forward_fn(model, split["ctx"])
    if model["knock"].get("two_error_inverter") == "zero":
        return invert_single_posit(f, split["y"])
    return invert_two_errors(f, split["y"])


# ----------------------------------------------------------------------------- registries
def modules(model):
    if model["kind"] == "engine":
        return {
            "riddle_encoder": {"params": ["W1[riddle rows]", "b1", "W2", "b2"], "signature": False,
                               "role": "two-layer tanh encoder of operation slots (multipliers, tolls, rates)"},
            "costume_encoder": {"params": ["W1[costume and ink rows]"], "signature": False,
                                "role": "input weights of nuisance one-hot and noise features"},
            "slope_head": {"params": ["Wa", "ba"], "signature": True,
                           "role": "context-conditioned multiplicative coefficient on the queried scalar"},
            "offset_head": {"params": ["Wb", "bb"], "signature": True, "role": "context-conditioned additive offset"},
            "two_error_inverter": {"params": [], "signature": True,
                                   "role": "two-probe secant inversion of the learned forward map"}}
    return {"hidden_layers": {"params": ["W1", "b1", "W2", "b2"], "signature": False, "role": "tanh MLP trunk"},
            "readout": {"params": ["W3", "b3"], "signature": False, "role": "linear readout"}}


KNOCKOUT_MODES = {"riddle_encoder": "mean", "costume_encoder": "mean", "slope_head": "mean",
                  "offset_head": "zero", "two_error_inverter": "zero"}


def knockout(model, name, mode):
    """Copy with a module replaced: 'mean' = training-mean input or output, 'zero' = zero output
    (for two_error_inverter, 'zero' drops the second error: single posit and proportion)."""
    if model["kind"] != "engine" or KNOCKOUT_MODES.get(name) != mode:
        raise ValueError(f"unsupported knockout {name}/{mode}")
    clone = dict(model)
    clone["params"] = {k: v.copy() for k, v in model["params"].items()}
    clone["knock"] = dict(model["knock"], **{name: mode})
    return clone


# ----------------------------------------------------------------------------- training
def train_three(data, rng, budget):
    """Engine, baseline A (amortized inverse) and baseline B (generic surrogate), same optimizer,
    learning rate (tuning grid {3e-3}), batch and update budget."""
    child = [np.random.default_rng(s) for s in np.random.SeedSequence(int(rng.integers(0, 2 ** 31))).spawn(6)]
    Xf, Yf = forward_arrays(data["train"])
    Xi, Yi = inverse_arrays(data["train"])
    eng = build_model(Xf.shape[1], 1, "vector_regression", child[0], kind="engine")
    inv = build_model(Xi.shape[1], 1, "vector_regression", child[1], kind="mlp", role="inverse")
    sur = build_model(Xf.shape[1], 1, "vector_regression", child[2], kind="mlp", role="surrogate")
    traces = {"engine": fit(eng, (Xf, Yf), budget, child[3]),
              "baseline_A": fit(inv, (Xi, Yi), budget, child[4]),
              "baseline_B": fit(sur, (Xf, Yf), budget, child[5])}
    return {"engine": eng, "baseline_A": inv, "baseline_B": sur}, traces


def trivial_mare(data, split):
    return mare(np.full_like(data[split]["x"], data["train"]["x"].mean()), data[split]["x"])


# ----------------------------------------------------------------------------- correctness tests
def _row_hashes(split):
    rows = np.round(np.column_stack([split["x"], split["ctx"]]), 9)
    return {hashlib.sha1(r.tobytes()).hexdigest() for r in rows}


def c1_gradients(data, rng, steps=60):
    """C1: every tensor of the engine and both baselines, at initialization and after training steps."""
    Xf, Yf = forward_arrays(data["train"])
    Xi, Yi = inverse_arrays(data["train"])
    Xf, Yf, Xi, Yi = Xf[:48], Yf[:48], Xi[:48], Yi[:48]
    worst, tensors = 0.0, 0
    for kind, role, X, Y in (("engine", None, Xf, Yf), ("mlp", "inverse", Xi, Yi), ("mlp", "surrogate", Xf, Yf)):
        model = build_model(X.shape[1], 1, "vector_regression", rng, kind=kind, role=role)
        for stage in ("init", "after"):
            if stage == "after":
                fit(model, (X, Y), steps, rng)
            _, g = loss_and_grads(model, (X, Y))
            rep = finite_difference_check(lambda: loss_and_grads(model, (X, Y))[0], model["params"], g, rng)
            worst = max(worst, max(v["max_rel"] for v in rep.values()))
            tensors += len(rep)
    return worst <= 1e-5, worst, tensors


def c3_learning(data, rng, budget):
    Xf, Yf = forward_arrays(data["train"])
    eng = build_model(Xf.shape[1], 1, "vector_regression", rng, kind="engine")
    trace = fit(eng, (Xf, Yf), budget, rng)
    drop = 1.0 - trace[-1] / trace[0]
    err, triv = mare(answer(eng, data["test"]), data["test"]["x"]), trivial_mare(data, "test")
    th = MIND_CARD["thresholds"]
    ok = drop >= th["loss_drop_fraction"] and err <= (1.0 - th["margin_over_trivial"]) * triv
    return ok, f"loss drop {drop:.3f}, held-out mare {err:.4f} vs trivial {triv:.4f}"


def c2_determinism(data, seed, budget):
    Xf, Yf = forward_arrays(data["train"])
    runs = []
    for _ in range(2):
        r = np.random.default_rng(seed)
        m = build_model(Xf.shape[1], 1, "vector_regression", r, kind="engine")
        runs.append((fit(m, (Xf, Yf), budget, r), predict(m, Xf[:50])))
    same = runs[0][0] == runs[1][0] and np.array_equal(runs[0][1], runs[1][1])
    return bool(same and np.all(np.isfinite(runs[0][1]))), "identical losses and outputs" if same else "mismatch"


def c4_shuffled(data, rng, budget):
    Xf, Yf = forward_arrays(data["train"])
    eng = build_model(Xf.shape[1], 1, "vector_regression", rng, kind="engine")
    fit(eng, (Xf, Yf[rng.permutation(len(Yf))]), budget, rng)
    err, triv = mare(answer(eng, data["test"]), data["test"]["x"]), trivial_mare(data, "test")
    return err >= MIND_CARD["thresholds"]["shuffled_band"] * triv, f"shuffled mare {err:.3f} vs trivial {triv:.3f}"


def c5_mutants(data, rng, budget):
    global ACTIVE_MUTANT
    outer, detected = ACTIVE_MUTANT, []
    for name in MUTANTS:
        ACTIVE_MUTANT = name
        try:
            c1_ok = c1_gradients(data, np.random.default_rng(7), steps=0)[0]
            c3_ok = c3_learning(data, np.random.default_rng(8), budget)[0]
        except NonFinite:
            c1_ok = c3_ok = False
        finally:
            ACTIVE_MUTANT = outer
        detected.append(not (c1_ok and c3_ok))
    return all(detected), f"{sum(detected)}/{len(detected)} mutants detected"


def c6_properties(rng, trials=300):
    """C6.1 the forward rule is affine in the unknown; C6.2 two errors invert it exactly.
    Both searched over random untrained models, contexts and far-out-of-range values,
    each with a negative control that must violate the property."""
    worst_aff, ctrl_aff, worst_inv, ctrl_inv = 0.0, 0.0, 0.0, 0.0
    for _ in range(trials // 30):
        eng = build_model(CTX_DIM + 1, 1, "vector_regression", rng, kind="engine")
        mlp = build_model(CTX_DIM + 1, 1, "vector_regression", rng, kind="mlp", role="surrogate")
        eng["params"]["bb"] += rng.normal(0, 1, 1)
        ctx = rng.normal(0, 1, (30, CTX_DIM))
        x1, x2, lam = rng.uniform(-300, 300, 30), rng.uniform(-300, 300, 30), rng.uniform(-2, 3, 30)
        for model, is_ctrl in ((eng, False), (mlp, True)):
            g = forward_fn(model, ctx)
            viol = np.abs(g(lam * x1 + (1 - lam) * x2) - lam * g(x1) - (1 - lam) * g(x2))
            v = float(np.max(viol / (1.0 + np.abs(g(lam * x1 + (1 - lam) * x2)))))
            ctrl_aff, worst_aff = (max(ctrl_aff, v), worst_aff) if is_ctrl else (ctrl_aff, max(worst_aff, v))
        y_t = rng.uniform(-3000, 3000, 30)
        slope = predict(eng, np.column_stack([np.ones(30), ctx]))[:, 0] - predict(eng, np.column_stack([np.zeros(30), ctx]))[:, 0]
        ok = np.abs(slope) > 1e-3
        f = forward_fn(eng, ctx)
        for inv, is_ctrl in ((lambda: invert_two_errors(f, y_t, steps=1), False),
                             (lambda: invert_single_posit(f, y_t), True)):
            v = float(np.max((np.abs(f(inv()) - y_t) / (1.0 + np.abs(y_t)))[ok]))
            ctrl_inv, worst_inv = (max(ctrl_inv, v), worst_inv) if is_ctrl else (ctrl_inv, max(worst_inv, v))
    ok61 = worst_aff <= 1e-9 and ctrl_aff > 1e-3
    ok62 = worst_inv <= 1e-8 and ctrl_inv > 1e-3
    return ok61 and ok62, (f"C6.1 affine violation {worst_aff:.1e} (control {ctrl_aff:.1e}); "
                           f"C6.2 inversion residual {worst_inv:.1e} (rule-of-three control {ctrl_inv:.1e})")


def c7_splits(data):
    sets = {k: _row_hashes(v) for k, v in data.items()}
    overlap = sum(len(sets["train"] & sets[k]) for k in ("val", "test", "shift"))
    return overlap == 0, f"{overlap} shared rows between train and evaluation splits"


# ----------------------------------------------------------------------------- hypothesis tests
def run_seed(seed_seq, budget):
    rng = np.random.default_rng(seed_seq)
    data = make_task(rng, "affine")
    models, _ = train_three(data, rng, budget)
    r = {f"{k}_{s}": mare(answer(m, data[s]), data[s]["x"]) for k, m in models.items() for s in ("test", "shift")}
    for name, mode in KNOCKOUT_MODES.items():
        ko = knockout(models["engine"], name, mode)
        for s in ("test", "shift"):
            r[f"ko_{name}_{s}"] = mare(answer(ko, data[s]), data[s]["x"])
    usury = make_task(rng, "usury")
    um, _ = train_three(usury, rng, budget)
    for k in ("engine", "baseline_A"):
        for s in ("test", "shift"):
            r[f"usury_{k}_{s}"] = mare(answer(um[k], usury[s]), usury[s]["x"])
    return r


def verdict(mean, lo, hi, direction, mesi):
    if direction == "less":
        if hi < 0 and mean <= -mesi:
            return "supported"
        return "contradicted" if lo > 0 else "inconclusive"
    if lo > 0 and mean >= mesi:
        return "supported"
    return "contradicted" if hi < 0 else "inconclusive"


HYP_DIFFS = {"H-SIG": ("engine_shift", "baseline_A_shift"),
             "H-SIG-B": ("engine_shift", "baseline_B_shift"),
             "H-NEC": ("ko_offset_head_shift", "ko_costume_encoder_shift"),
             "H-BLIND": ("usury_engine_test", "usury_baseline_A_test")}


def hypotheses(results, rng):
    out = []
    for h in MIND_CARD["hypotheses"]:
        a, b = HYP_DIFFS[h["id"]]
        diffs = [r[a] - r[b] for r in results]
        mean, lo, hi = paired_bootstrap(diffs, rng)
        out.append({"id": h["id"], "metric": h["metric"], "mean_diff": mean, "ci95": [lo, hi], "mesi": h["mesi"],
                    "n_seeds": len(results), "verdict": verdict(mean, lo, hi, h["direction"], h["mesi"])})
    return out


def knockout_table(results, rng):
    rows = []
    for name in KNOCKOUT_MODES:
        diffs = [r[f"ko_{name}_shift"] - r["engine_shift"] for r in results]
        held = float(np.mean([r[f"ko_{name}_test"] - r["engine_test"] for r in results]))
        mean, lo, hi = paired_bootstrap(diffs, rng)
        rows.append({"module": name, "signature": modules({"kind": "engine"})[name]["signature"],
                     "metric_change": mean, "ci95": [lo, hi], "heldout_change": held})
    return rows


# ----------------------------------------------------------------------------- report
def real_data_bridge(path, rng, budget):
    """Optional: CSV with header; first column = queried quantity, last = outcome, middle = context."""
    arr = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=float)
    arr = arr[np.all(np.isfinite(arr), axis=1)]
    cut = int(0.8 * len(arr))
    x, ctx, y = arr[:, 0], arr[:, 1:-1], arr[:, -1]
    mu, sd = ctx[:cut].mean(0), ctx[:cut].std(0) + 1e-9
    ctx = (ctx - mu) / sd
    sx, sy = float(np.abs(x[:cut]).max()), float(np.abs(y[:cut]).max())
    model = build_model(1 + ctx.shape[1], 1, "vector_regression", rng, kind="engine")
    fit(model, (np.column_stack([x[:cut] / sx, ctx[:cut]]), (y[:cut] / sy)[:, None]), budget, rng)
    posits = tuple(np.quantile(x[:cut], [0.25, 0.75]))
    x_hat = invert_two_errors(forward_fn(model, ctx[cut:], sx, sy), y[cut:], posits=posits)
    return f"{len(arr)} rows, held-out inverse mare {mare(x_hat, x[cut:]):.4f}"


def fmt_row(cid, name, ok, detail):
    return f"  {cid:<4} {name:<26} {'PASS' if ok else 'FAIL'}  {detail}"


def build_report(ctx):
    L = ["=== VERIFIED REPORT · chapter 0318 ===",
         f"environment: Python {sys.version.split()[0]} · NumPy {np.__version__} · "
         f"card revision {MIND_CARD['card_revision']}",
         f"seeds: base {ctx['seed']} · hypothesis seeds {ctx['seeds']} · mode {ctx['mode']}",
         f"runtime: {ctx['runtime']:.1f} s (budget {ctx['budget_s']} s)",
         f"parameters: engine {ctx['n_params']['engine']} · baseline A (amortized inverse) "
         f"{ctx['n_params']['baseline_A']} · baseline B (generic surrogate) {ctx['n_params']['baseline_B']}",
         f"gradient check: {ctx['grad']['tensors']} tensor checks (3 models x init and after 60 steps), "
         f"max relative error {ctx['grad']['max_rel']:.2e} -> {'PASS' if ctx['grad']['ok'] else 'FAIL'}",
         "correctness:"]
    L += [fmt_row(c["id"], c["name"], c["passed"], c["detail"]) for c in ctx["correctness"]]
    L.append(f"mutation score: {ctx['mutants']['detected']}/{ctx['mutants']['total']}")
    L.append("hypotheses (paired per-seed differences, 2000-resample bootstrap, 95% CI; metric = capped mare):")
    for h in ctx["hypotheses"]:
        if h["verdict"] == "not evaluated":
            L.append(f"  {h['id']:<8} not evaluated ({ctx['mode']})")
        else:
            L.append(f"  {h['id']:<8} mean diff {h['mean_diff']:+.4f}  CI [{h['ci95'][0]:+.4f}, {h['ci95'][1]:+.4f}]"
                     f"  mesi {h['mesi']:.2f}  n={h['n_seeds']}  -> {h['verdict']}")
    d = ctx["descriptive"]
    L.append("descriptive means over seeds (mare; lower is better):")
    L.append(f"  held-out  trivial {d['trivial_test']:.4f} · engine {d['engine_test']:.4f} · "
             f"A {d['baseline_A_test']:.4f} · B {d['baseline_B_test']:.4f}")
    L.append(f"  shifted   trivial {d['trivial_shift']:.4f} · engine {d['engine_shift']:.4f} · "
             f"A {d['baseline_A_shift']:.4f} · B {d['baseline_B_shift']:.4f}")
    L.append(f"  usury (exploratory, not a hypothesis) shifted: engine {d['usury_engine_shift']:.4f} · "
             f"A {d['usury_baseline_A_shift']:.4f}")
    L.append("knockouts (change in shifted mare vs intact engine, CI; held-out change):")
    for k in ctx["knockouts"]:
        L.append(f"  {k['module']:<19} {'sig' if k['signature'] else '   '} {k['metric_change']:+.4f}  "
                 f"[{k['ci95'][0]:+.4f}, {k['ci95'][1]:+.4f}]  held-out {k['heldout_change']:+.4f}")
    L += [f"task types: {', '.join(TASK_TYPES)}", f"real-data bridge: {ctx['bridge']}", "=== END REPORT ==="]
    return L


def descriptive(results, data):
    keys = ["engine_test", "baseline_A_test", "baseline_B_test", "engine_shift", "baseline_A_shift",
            "baseline_B_shift", "usury_engine_shift", "usury_baseline_A_shift"]
    d = {k: float(np.mean([r[k] for r in results])) for k in keys}
    d["trivial_test"], d["trivial_shift"] = trivial_mare(data, "test"), trivial_mare(data, "shift")
    return d


def correctness_suite(data, args, budget):
    rng = np.random.default_rng(args.seed)
    g_ok, g_max, g_tensors = c1_gradients(data, rng)
    rows = [{"id": "C1", "name": "gradient_check", "passed": g_ok, "detail": f"max rel {g_max:.2e}"}]
    for cid, name, fn in (("C2", "determinism_finiteness", lambda: c2_determinism(data, args.seed, 150)),
                          ("C3", "learning", lambda: c3_learning(data, rng, budget)),
                          ("C4", "shuffled_label_control", lambda: c4_shuffled(data, rng, budget)),
                          ("C5", "mutant_detection", lambda: c5_mutants(data, rng, budget)),
                          ("C6", "property_tests", lambda: c6_properties(rng)),
                          ("C7", "split_integrity", lambda: c7_splits(data))):
        ok, detail = fn()
        rows.append({"id": cid, "name": name, "passed": bool(ok), "detail": detail})
    return rows, {"ok": g_ok, "max_rel": g_max, "tensors": g_tensors}


def main(argv=None):
    global ACTIVE_MUTANT
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=1202)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--json", metavar="PATH")
    ap.add_argument("--card", action="store_true")
    ap.add_argument("--mutant", choices=sorted(MUTANTS))
    ap.add_argument("--data", metavar="PATH")
    args = ap.parse_args(argv)
    if args.card:
        print(json.dumps(MIND_CARD, indent=2))
        return 0
    if args.seeds < 1:
        ap.error("--seeds must be at least 1")
    ACTIVE_MUTANT = args.mutant
    t0, budget_s = time.time(), (20 if args.quick else 180)
    budget, n_seeds = (700, 1) if args.quick else (2500, args.seeds)
    mode = "quick" if args.quick else ("mutant " + args.mutant if args.mutant else "full")
    try:
        data = make_task(np.random.default_rng(args.seed))
        correctness, grad = correctness_suite(data, args, budget)
        seeds = [args.seed + i for i in range(n_seeds)]
        results = [run_seed(np.random.SeedSequence(s), budget) for s in seeds]
        boot = np.random.default_rng(args.seed + 99)
        hyps = hypotheses(results, boot) if (n_seeds >= 5 and not args.mutant) else \
            [{"id": h["id"], "metric": h["metric"], "mean_diff": None, "ci95": None, "mesi": h["mesi"],
              "n_seeds": n_seeds, "verdict": "not evaluated"} for h in MIND_CARD["hypotheses"]]
        knocks = knockout_table(results, boot)
        bridge = real_data_bridge(args.data, np.random.default_rng(args.seed), budget) if args.data \
            else "skipped (no --data PATH given)"
    except NonFinite as exc:
        print(f"non-finite values: {exc}")
        return 4
    runtime = time.time() - t0
    correctness.append({"id": "C8", "name": "budget", "passed": runtime <= budget_s,
                        "detail": f"{runtime:.1f} s of {budget_s} s"})
    c5 = next(c for c in correctness if c["id"] == "C5")
    n_det = int(c5["detail"].split("/")[0])
    probe = build_model(CTX_DIM + 1, 1, "vector_regression", np.random.default_rng(0))
    sizes = {"engine": n_params(probe),
             "baseline_A": n_params(build_model(CTX_DIM + 1, 1, "vector_regression", np.random.default_rng(0), kind="mlp")),
             "baseline_B": n_params(build_model(CTX_DIM + 1, 1, "vector_regression", np.random.default_rng(0), kind="mlp"))}
    exit_code = 0 if all(c["passed"] for c in correctness[:-1]) else 1
    if exit_code == 0 and not correctness[-1]["passed"]:
        exit_code = 3
    ctx = {"seed": args.seed, "seeds": seeds, "mode": mode, "runtime": runtime, "budget_s": budget_s,
           "n_params": sizes, "grad": grad, "correctness": correctness,
           "mutants": {"detected": n_det, "total": len(MUTANTS)}, "hypotheses": hyps,
           "descriptive": descriptive(results, data), "knockouts": knocks, "bridge": bridge}
    payload = {"schema_version": "1.0", "chapter": 318, "file": os.path.basename(__file__),
               "card_revision": MIND_CARD["card_revision"],
               "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
               "seeds": seeds, "runtime_s": round(runtime, 2), "n_params": sizes["engine"],
               "gradcheck": {"tensors_checked": grad["tensors"], "tensors_total": grad["tensors"],
                             "max_rel_error": grad["max_rel"], "checked_at": ["init", "after_training_steps"],
                             "passed": grad["ok"]},
               "correctness": correctness, "mutants": {"detected": n_det, "total": len(MUTANTS),
                                                       "score": n_det / len(MUTANTS)},
               "hypotheses": hyps, "knockouts": knocks, "task_types": TASK_TYPES, "exit_code": exit_code}
    write_report(build_report(ctx), payload, args.json)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

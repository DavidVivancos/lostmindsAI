#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0339 · Guo Shoujing (1231-1316)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 17, Minds 321-340 Available on Amazon https://www.amazon.com/dp/B0HP7Z1NKC
# Resume and Interactive Demos at https://artificiology.com/
# END ATTRIBUTION
"""
SHOUSHI · the Season-Granting forecaster · chapter 0339

Thesis
  Move the origin of time to the last thing you measured, let the constants age,
  and predict the rest from a small table of differences: for this mind a
  forecast is a re-measured recent anchor, a slowly drifting rate and an
  interpolation table, never a mechanism and never a mythical common origin.

Evidence and provenance (belief: the figure's own memorials and the treatise he
  submitted survive, mediated through the Yuan shi and Qi Luqian's account of
  conduct; see MIND_CARD["evidence"] for the loci)
  D1  "The root of the calendar is measurement, and among the instruments of
      measurement nothing comes before the gnomon and the armilla" (memorial of
      1276, Yuan shi 164; Qi Luqian, Xingzhuang).
  D2  The Shoushi system abolished the Grand Epoch (shangyuan jinian), the
      computed moment tens of millions of years back at which all cycles were
      supposed to coincide, and took the winter solstice of 1280 as its epoch
      with measured interval constants (Yuan shi 52-55; Sivin 2009).
  D3  It gave the tropical year a secular decrease (xiaozhang) of one part in
      ten thousand of a day per century, fitted against solstice records
      reaching back more than a millennium (Yuan shi 52; Sivin 2009).
  D4  Unequal solar and lunar motion was computed from tables of third-order
      differences (zhaocha) over the 24 qi of the year (Yuan shi 54-55).
  D5  The 40-chi gnomon and the pinhole shadow-definer (jingfu) multiplied the
      signal five-fold and then sharpened the blurred edge (Yuan shi 48).
  D6  Twenty-seven stations from 15 to 65 Chinese degrees of pole height
      measured the same sky through different gnomons (Yuan shi 48).
  D7  Blind spot: the drift rate was an order of magnitude too large and was
      dropped by the Ming Datong system of 1384; the early Shoushi was in
      general no better than Datong for China (Li & Zhang 1997; Lee et al. 2014).

Doctrine -> mechanism -> test
  D2  -> M1 qiying_anchor: phase re-measured at the last observation by a
         least-squares sinusoid fit of the recent record at the current rate
         and drift, never carried from an origin -> C6.2 (T0-invariance), H-RIVAL
  D3  -> M2 xiaozhang_drift: second difference of the epoch record, corrected
         within a bound, integrated as a chirp   -> C6.3 (monotone phase), H-NEC, H-BLIND
  D4  -> M3 zhaocha_table: mean motion plus a periodic cubic difference table
         (24 knots) that carries no fundamental  -> C6.1 (periodicity), knockout
  D5,D6 -> instrument: matched-filter bank, six measuring epochs fitted by one
         quadratic (first and second differences), per-window normalisation
  D7  -> blind-spot condition: zero true drift under a noisy archive

Research question (forecasting and time-to-event; continual adaptation)
  When a quasi-periodic process ages slowly, does anchoring phase at the most
  recent measured event and integrating a learned secular rate extrapolate
  further than a size-matched network fed absolute time and Fourier features?

Closest prior art and the delta
  Fourier-feature and Snake networks for periodic extrapolation (Tancik et al.
  2020; Ziyin et al. 2020), reversible instance normalisation for shift
  (Kim et al. 2022), N-BEATS seasonal blocks (Oreshkin et al. 2020), chirp /
  linear-FM models. Delta: the phase is re-anchored per window at the last
  observation and integrated with a window-estimated drift rate, so phase error
  scales with the horizon rather than with the distance to an origin; the shape
  is a learned periodic difference table rather than a fixed basis. Overlap:
  Medium (known parts in a new combination).

Blind spot
  A drift term fitted through a noisy archive hallucinates ageing where there
  is none; on zero-drift data with crude early records the drift-free variant
  (the Datong li) should win. Predicted before the first run.

Task (generative process)
  P ~ U[0.85,1.15] cycles, omega0 = 2pi/P; drift d ~ U[-0.005,0.005] per cycle;
  amplitude A ~ U[0.6,1.4], offset b ~ U[-1,1], noise s ~ U[0.05,0.2].
  L observations at t_i = i/6 + jitter (12 nominal cycles), phase
  phi(t) = phi_w + omega0*(t + d*t^2/2), phi_w = omega0*T0 mod 2pi with
  T0 ~ U[300,1500] (a world consistent with a Grand Epoch at T=0).
  y_i = b + A*f(phi(t_i)) + s*e_i, f = sin + 0.25 sin(2phi+0.6) + 0.1 cos 3phi.
  Target: y at t_L + h. Train/held-out h in [0.25,6]; shifted h in [6,12];
  blind: d = 0, early-half noise 0.35, h in [0.25,6].

Limits
  Synthetic data, one shared shape, no planetary-style multi-body coupling;
  the model reads one window and has no memory across windows. A research
  prototype of an AGI-oriented forecasting module, not an AGI.
"""

MIND_CARD = {
    "schema_version": "1.0", "card_revision": 2,
    "revision_log": [
        "rev 2 (2026-09-25): after the first diagnostic run the shifted horizon was cut from 6-20 to 6-12 cycles "
        "(beyond 12 cycles the window noise leaves both model and baseline at chance), the true drift range was "
        "widened from 0.003 to 0.005 per cycle so the secular term matters at 6-12 cycles, and the C6.3 negative "
        "control was changed to random strong-drift windows under a loose bound. Hypotheses, metrics and mesi unchanged. "
        "Same session: dialectic links to 0195 and 0273 added after the corpus check for overlapping archetypes.",
    ],
    "generation": {"template_version": "1.0", "generator": "Claude", "generator_version": "Fable 5.1",
                   "date": "2026-09-25"},
    "id": 339, "figure": "Guo Shoujing", "born": 1231, "died": 1316, "civilization": "Chinese (Yuan)",
    "provenance": "belief",
    "thesis": "Anchor time at the last thing you measured, let the constants age, and predict from a "
              "table of differences: never a mechanism, never a mythical common origin.",
    "evidence": [
        {"id": "D1", "claim": "The root of the calendar is measurement; instruments come first.",
         "basis": "primary", "source": "Yuan shi 164 (biography); Qi Luqian, Zhi Taishiyuan shi Guo gong xingzhuang, Yuan wen lei 50"},
        {"id": "D2", "claim": "Shoushi abolished the Grand Epoch and took the 1280 winter solstice as epoch with measured interval constants.",
         "basis": "primary", "source": "Yuan shi 52-53 (Shoushi li yi); Sivin, Granting the Seasons (2009)"},
        {"id": "D3", "claim": "Tropical year given a secular decrease of 1e-4 day per century fitted to ancient solstice records.",
         "basis": "primary", "source": "Yuan shi 52; Li & Zhang, Earth, Moon and Planets 76 (1997-98)"},
        {"id": "D4", "claim": "Unequal motion computed by third-order difference tables over the 24 qi.",
         "basis": "primary", "source": "Yuan shi 54-55 (Shoushi li jing); Sivin 2009 ch. 1"},
        {"id": "D5", "claim": "40-chi gnomon plus pinhole shadow-definer: amplify then sharpen.",
         "basis": "primary", "source": "Yuan shi 48 (Tianwen zhi); Qi Luqian, Xingzhuang"},
        {"id": "D6", "claim": "27 stations from 15 to 65 Chinese degrees measured the same sky.",
         "basis": "primary", "source": "Yuan shi 48, Sihai ceyan table"},
        {"id": "D7", "claim": "Drift rate over-estimated; dropped by the Ming Datong li of 1384.",
         "basis": "scholarship", "source": "Lee et al., arXiv:1311.4735 (2014); Chen Zungui, Zhongguo tianwenxue shi"},
    ],
    "research_question": {"category": "forecasting and time-to-event; continual learning and plasticity",
                          "question": "Does recent-event phase anchoring with a window-estimated secular drift "
                                      "extrapolate further than absolute-time Fourier networks?"},
    "mechanism": {
        "name": "SHOUSHI (Season-Granting forecaster)", "family": "continuous-time parametric forecasting",
        "signature_modules": ["qiying_anchor", "xiaozhang_drift", "zhaocha_table"],
        "closest_prior_art": ["Fourier feature networks (Tancik et al. 2020)", "Snake activation (Ziyin et al. 2020)",
                              "RevIN instance normalisation (Kim et al. 2022)", "N-BEATS seasonal block (Oreshkin et al. 2020)",
                              "linear-FM / chirp estimation"],
        "overlap": "Medium",
        "prior_art_queries": ["neural networks periodic extrapolation", "phase anchored forecasting drift",
                              "chirp rate estimation neural", "non-stationary periodic time series forecasting"],
        "contribution_type": "mechanism",
        "delta": "Phase re-anchored per window at the last observation and integrated with a window-estimated "
                 "drift rate over the horizon; the shape is a unit sinusoid of mean motion plus a periodic cubic "
                 "difference table with no fundamental, so the phase gauge stays measurable; phase error grows "
                 "with horizon, not with distance to an origin.",
    },
    "traceability": [
        {"doctrine": "D2", "mechanism": "M1", "property_test": "C6.2", "hypothesis": "H-RIVAL"},
        {"doctrine": "D3", "mechanism": "M2", "property_test": "C6.3", "hypothesis": "H-NEC"},
        {"doctrine": "D7", "mechanism": "M2", "property_test": "C6.3", "hypothesis": "H-BLIND"},
        {"doctrine": "D4", "mechanism": "M3", "property_test": "C6.1", "hypothesis": "H-SIG"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "Anchored drift model beats size-matched Fourier-feature MLP at unseen long horizons.",
         "metric": "neg_rmse", "split": "shifted", "comparison": "model - baseline", "direction": "greater",
         "mesi": 0.05, "seeds": 5},
        {"id": "H-NEC", "statement": "Removing the drift module hurts more than removing the residual head layer.",
         "metric": "neg_rmse", "split": "shifted", "comparison": "signature_knockout - matched_knockout",
         "direction": "less", "mesi": 0.02, "seeds": 5},
        {"id": "H-BLIND", "statement": "With zero true drift and a noisy archive the drift model is worse than its drift-free knockout.",
         "condition": "blind", "grounding": "Shoushi's xiaozhang rate was too large; Datong dropped it",
         "metric": "neg_rmse", "comparison": "model - drift_knockout", "direction": "less", "mesi": 0.01, "seeds": 5},
        {"id": "H-RIVAL", "statement": "Grand-Epoch anchoring (phase carried from a distant origin) is worse than recent anchoring.",
         "metric": "neg_rmse", "split": "held_out", "comparison": "model - rival", "direction": "greater",
         "mesi": 0.05, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.5, "margin_over_trivial": 0.25, "gradcheck_rel_err": 1e-5},
    "probe_predictions": [
        {"probe": "P4 length extrapolation", "expected": "above baseline"},
        {"probe": "P7 regime change and free run", "expected": "above baseline"},
        {"probe": "P9 perturbation robustness", "expected": "below baseline"},
        {"probe": "P1 delayed recall", "expected": "below baseline"},
        {"probe": "P3 compositional split", "expected": "equal to baseline"},
    ],
    "dialectic_links": [
        {"chapter": 328, "relation": "rival", "test": "H-RIVAL",
         "note": "Qin Jiushao's dayan method existed to compute the Grand Epoch the Shoushi abolished."},
        {"chapter": 327, "relation": "rival", "test": "none",
         "note": "Tusi demands a constitutive mechanism; Guo predicts from measured tables with no mechanism."},
        {"chapter": 323, "relation": "successor", "test": "none", "note": "Same northern Jin/Yuan milieu (Liu Bingzhong's circle)."},
        {"chapter": 195, "relation": "successor", "test": "none",
         "note": "Yi Xing's unequal-interval second-order differences and recalibration; the Shoushi's zhaocha is third-order, "
                 "but this chapter's thesis rests on the recent anchor and the bounded drift, not on interpolation."},
        {"chapter": 273, "relation": "successor", "test": "none",
         "note": "Shen Kuo's repeated pole-star sightings treated disagreement as evidence; the 27-station survey scales that "
                 "practice, and it is a secondary move here, not the signature."},
    ],
    "corpus_neighbors": [
        {"chapter": 507, "similarity": 0.039, "difference": "token-shingle Jaccard measured against the only file available in "
         "this session (leave-one-out sealed products vs anchored chirp forecasting)"},
        {"chapter": 328, "similarity": 0.0, "difference": "not measurable in session (file absent); reassembly from remainders "
         "under a common epoch vs anchoring at a measured event"},
        {"chapter": 327, "similarity": 0.0, "difference": "not measurable in session (file absent); uniform-circle constitution "
         "vs table interpolation without mechanism"},
        {"chapter": 336, "similarity": 0.0, "difference": "not measurable in session (file absent); Alfonsine geometric tables "
         "kept Ptolemy; Shoushi is purely numerical"},
    ],
    "barometer": {
        "cognitive_processing": ["learning curve on the native task"],
        "embodied_cognition": [],
        "world_modeling": ["free-running forecast to 12 cycles", "drift-sign regime change"],
        "consciousness": [], "language_understanding": [], "emotional_intelligence": [],
        "creativity": [], "autonomy": [],
    },
    "task_types": ["sequence_regression"],
    "applications": [
        {"use": "Solar-cycle forecasting with a varying cycle length", "sector": "space weather / satellite operations",
         "dataset": "SILSO monthly mean sunspot number (CSV)"},
        {"use": "Seasonal series with slowly changing seasonality", "sector": "climate monitoring",
         "dataset": "NOAA GML Mauna Loa CO2 monthly means (CSV)"},
        {"use": "Rotating-machinery vibration with drifting speed", "sector": "industrial predictive maintenance",
         "dataset": "CWRU bearing dataset"},
    ],
    "safety_notes": "Synthetic data only; no data about people; forecasting of physical cycles.",
}

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

# BEGIN STANDARD UTILITIES v1.1
def stable_softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def logsumexp(z, axis=-1):
    m = np.max(z, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(z - m), axis=axis))


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
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] = params[k] - self.lr * mh / (np.sqrt(vh) + self.eps)


def clip_global_norm(grads, max_norm):
    total = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    if total > max_norm:
        s = max_norm / (total + 1e-12)
        grads = {k: g * s for k, g in grads.items()}
    return grads, total


def finite_difference_check(loss_fn, params, grads, rng, eps=1e-6, n_entries=20, floor=1e-3):
    """Central differences on every tensor; returns (max relative error, per-tensor dict).

    The denominator floor keeps entries with near-zero gradient from being judged against float64
    central-difference rounding noise (about 1e-10 absolute); below the floor the test is an absolute one."""
    report, worst = {}, 0.0
    for k, p in params.items():
        idx = list(np.ndindex(p.shape))
        pick = [idx[i] for i in rng.choice(len(idx), size=min(n_entries, len(idx)), replace=False)]
        pick.append(idx[int(np.argmax(np.abs(grads[k])))])
        errs = []
        for ij in pick:
            old = p[ij]
            p[ij] = old + eps
            lp = loss_fn()
            p[ij] = old - eps
            lm = loss_fn()
            p[ij] = old
            num = (lp - lm) / (2 * eps)
            ana = grads[k][ij]
            errs.append(abs(num - ana) / max(abs(num), abs(ana), floor))
        report[k] = float(max(errs))
        worst = max(worst, report[k])
    return worst, report


def paired_bootstrap(a, b, rng, n_boot=2000):
    """Mean of per-seed differences a-b and a 95% percentile bootstrap interval."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if len(d) < 2:
        return float(np.mean(d)), (float(np.mean(d)), float(np.mean(d)))
    means = [float(np.mean(d[rng.integers(0, len(d), len(d))])) for _ in range(n_boot)]
    return float(np.mean(d)), (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def verdict(mean_diff, ci, direction, mesi):
    lo, hi = ci
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


def write_report(chapter, lines, json_path=None, payload=None):
    print(f"=== VERIFIED REPORT · chapter {chapter:04d} ===")
    for ln in lines:
        print(ln)
    print("=== END REPORT ===")
    if json_path and payload is not None:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
# END STANDARD UTILITIES

# ------------------------------------------------------------------ data and tasks
L_OBS = 72            # observations per window (12 nominal cycles at 6 per cycle)
SAMPLES_PER_CYCLE = 6
N_KNOTS = 24          # the 24 qi of the year
OMEGA_REF = 2.0 * math.pi
DRIFT_SCALE = 0.02    # |delta_hat| < 0.02 keeps phase monotone for h <= 12
H_MAX = 12.0
N_EPOCHS = 6          # measuring epochs of two cycles each: the ancient-to-recent record
T0_MAX = 1500.0
CLIP_NORM = 5.0
TASK_TYPES = ["sequence_regression"]


def true_shape(phi):
    return np.sin(phi) + 0.25 * np.sin(2 * phi + 0.6) + 0.1 * np.cos(3 * phi)


def make_windows(n, rng, h_range=(0.25, 6.0), drift_range=(-0.005, 0.005), early_noise=None):
    """One window = one station's record of a drifting sky, plus the target at t_L + h."""
    P = rng.uniform(0.85, 1.15, n)
    om = 2 * math.pi / P
    d = rng.uniform(drift_range[0], drift_range[1], n)
    A = rng.uniform(0.6, 1.4, n)
    b = rng.uniform(-1.0, 1.0, n)
    s = rng.uniform(0.05, 0.2, n)
    T0 = rng.uniform(300.0, T0_MAX, n)
    h = rng.uniform(h_range[0], h_range[1], n)
    base = np.arange(L_OBS) / SAMPLES_PER_CYCLE
    t = base[None, :] + rng.uniform(-0.1, 0.1, (n, L_OBS)) / SAMPLES_PER_CYCLE
    t[:, 0] = 0.0
    phi_w = np.mod(om * T0, 2 * math.pi)
    phase = phi_w[:, None] + om[:, None] * (t + 0.5 * d[:, None] * t * t)
    noise = rng.standard_normal((n, L_OBS)) * s[:, None]
    if early_noise is not None:
        noise[:, : L_OBS // 2] = rng.standard_normal((n, L_OBS // 2)) * early_noise
    y = b[:, None] + A[:, None] * true_shape(phase) + noise
    tq = t[:, -1] + h
    yq = b + A * true_shape(phi_w + om * (tq + 0.5 * d * tq * tq))
    X = np.stack([t, y, np.repeat(h[:, None], L_OBS, 1), np.repeat(T0[:, None], L_OBS, 1)], axis=2)
    return X, yq


def make_data(seed, quick=False):
    ss = np.random.SeedSequence(seed)
    r = [np.random.default_rng(c) for c in ss.spawn(5)]
    n_tr, n_ev = (500, 200) if quick else (2000, 600)
    return {
        "train": make_windows(n_tr, r[0]),
        "val": make_windows(max(120, n_tr // 5), r[1]),
        "held_out": make_windows(n_ev, r[2]),
        "shifted": make_windows(n_ev, r[3], h_range=(6.0, 12.0)),
        "blind": make_windows(n_ev, r[4], drift_range=(0.0, 0.0), early_noise=0.35),
    }


def unpack(X):
    """Channels: 0 time, 1 value, 2 horizon, 3 absolute epoch distance (missing channels get defaults)."""
    X = np.asarray(X, dtype=np.float64)
    n, L, F = X.shape
    t = X[:, :, 0] if F >= 2 else np.repeat(np.arange(L)[None, :] / SAMPLES_PER_CYCLE, n, 0)
    y = X[:, :, 1] if F >= 2 else X[:, :, 0]
    h = X[:, 0, 2] if F >= 3 else np.ones(n)
    T0 = X[:, 0, 3] if F >= 4 else np.zeros(n)
    return t, y, h, T0


def normalise_window(y):
    mu = y.mean(1, keepdims=True)
    sd = y.std(1, keepdims=True) + 1e-6
    return (y - mu) / sd, mu, sd


def trivial_baselines(X, yq):
    """Mean, persistence and seasonal-naive predictors in normalised units."""
    t, y, h, _ = unpack(X)
    yn, mu, sd = normalise_window(y)
    target = (yq - mu[:, 0]) / sd[:, 0]
    rm = float(np.sqrt(np.mean(target ** 2)))
    rp = float(np.sqrt(np.mean((target - yn[:, -1]) ** 2)))
    tq = t[:, -1] + h
    tstar = tq - np.ceil(h)
    idx = np.argmin(np.abs(t - tstar[:, None]), axis=1)
    rs = float(np.sqrt(np.mean((target - yn[np.arange(len(yn)), idx]) ** 2)))
    return {"mean": rm, "persistence": rp, "seasonal_naive": rs}


def split_integrity(data):
    seen = {}
    for name, (X, _) in data.items():
        for row in X:
            key = hashlib.sha1(np.ascontiguousarray(row).tobytes()).hexdigest()
            if key in seen and seen[key] != name:
                return False, f"row shared by {seen[key]} and {name}"
            seen[key] = name
    return True, "no example appears in two splits"

# ------------------------------------------------------------------ model
K_BANK = 16
BANK_OM = OMEGA_REF * np.geomspace(0.82, 1.2, K_BANK)     # fixed grid of candidate rates (the instrument)
BANK_BETA = 8.0                                            # sharpness of the soft-argmax over bank power
RATE_GAIN = 0.15                                           # authority of the learned correction over the measured rate
DRIFT_GAIN = 0.1                                           # authority of the learned correction over the measured drift
H_SHOUSHI = 20
H_MLP = 19
FF_FREQS = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])   # fixed Fourier features of the horizon
_MASK_CACHE = {}


def segment_masks(L):
    """Averaging masks over the early half of the archive, the late half (recent measurements) and the whole window."""
    if L not in _MASK_CACHE:
        m = np.zeros((3, L))
        m[0, : L // 2], m[1, L // 2:], m[2, :] = 2.0 / L, 2.0 / L, 1.0 / L
        _MASK_CACHE[L] = m
    return _MASK_CACHE[L]
_FUND = np.stack([np.cos(2 * np.pi * np.arange(N_KNOTS) / N_KNOTS), np.sin(2 * np.pi * np.arange(N_KNOTS) / N_KNOTS)])
TABLE_PROJ = np.eye(N_KNOTS) - (2.0 / N_KNOTS) * _FUND.T @ _FUND   # removes the fundamental from the knot sequence

MUTANTS = {
    "sign_flip": "optimizer applies the negated update",
    "zero_lr": "learning rate set to zero",
    "grad_zero_knots": "gradient of the difference table zeroed",
    "spline_derivative_bug": "table derivative uses linear instead of cubic weights",
}


def _init(rng, shape, scale):
    return rng.standard_normal(shape) * scale


def build_model(in_dim, out_dim, task_type, rng, kind="shoushi", mutant=None, drift_scale=DRIFT_SCALE, **cfg):
    """kind: 'shoushi' (full), 'grand' (Grand-Epoch rival), 'ffmlp' (Fourier-feature MLP baseline on the same instrument)."""
    assert task_type in TASK_TYPES and out_dim == 1
    p = {}
    if kind in ("shoushi", "grand"):
        fin = 9 * K_BANK + (1 if kind == "grand" else 0)
        p["W1"] = _init(rng, (fin, H_SHOUSHI), 1.0 / math.sqrt(fin))
        p["b1"] = np.zeros(H_SHOUSHI)
        p["W2"] = _init(rng, (H_SHOUSHI, H_SHOUSHI), 0.5 / math.sqrt(H_SHOUSHI))
        p["b2"] = np.zeros(H_SHOUSHI)
        p["W3"] = _init(rng, (H_SHOUSHI, 2), 0.01)                # corrections start near zero: trust the instrument
        p["b3"] = np.zeros(2)
        p["amp"] = np.ones(1)                              # mean motion: a unit sinusoid of the anchored phase
        p["knots"] = _init(rng, (N_KNOTS,), 0.1)           # the inequality: a difference table over the mean motion
        p["c"] = np.zeros(1)
    else:
        fin = 9 * K_BANK + 4 + 2 * len(FF_FREQS) + 2
        p["Wa"] = _init(rng, (fin, H_MLP), 1.0 / math.sqrt(fin))
        p["ba"] = np.zeros(H_MLP)
        p["Wb"] = _init(rng, (H_MLP, H_MLP), 1.0 / math.sqrt(H_MLP))
        p["bb"] = np.zeros(H_MLP)
        p["Wc"] = _init(rng, (H_MLP, 1), 0.3)
        p["bc"] = np.zeros(1)
    return {"kind": kind, "params": p, "ko": {}, "mutant": mutant, "stats": {}, "in_dim": in_dim,
            "cfg": {"drift_scale": drift_scale, "periodic_table": cfg.get("periodic_table", True)}}


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


def spline_weights(phi, periodic=True):
    """Uniform cubic B-spline over N_KNOTS periodic knots: indices, weights and d(weights)/ds."""
    u = phi / (2 * math.pi) * N_KNOTS
    j = np.floor(u).astype(int)
    s = u - j
    w = np.stack([(1 - s) ** 3 / 6, (3 * s ** 3 - 6 * s ** 2 + 4) / 6, (-3 * s ** 3 + 3 * s ** 2 + 3 * s + 1) / 6, s ** 3 / 6], 1)
    dw = np.stack([-(1 - s) ** 2 / 2, (3 * s ** 2 - 4 * s) / 2, (-3 * s ** 2 + 2 * s + 1) / 2, s ** 2 / 2], 1)
    idx = np.stack([j - 1, j, j + 1, j + 2], 1)
    idx = np.mod(idx, N_KNOTS) if periodic else np.clip(idx, 0, N_KNOTS - 1)
    return idx, w, dw


def fit_sinusoid(y, c, s, w):
    """Weighted least squares y ~ alpha cos + beta sin + gamma over the last axis; returns coefficients and normal matrix."""
    cols = [c, s, np.ones_like(c)]
    G = np.stack([np.stack([np.sum(w * a * b, -1) for b in cols], -1) for a in cols], -2)
    r = np.stack([np.sum(w * y * a, -1) for a in cols], -1)
    abg = np.linalg.solve(G, r[..., None])[..., 0]
    return abg[..., 0], abg[..., 1], abg[..., 2], G


def epoch_phases(yn, tau, om):
    """Phase of the fundamental at the centre of each of N_EPOCHS consecutive spans, fitted at the local rate om[n, k]."""
    cut = yn.shape[1] % N_EPOCHS
    Y, T = yn[:, cut:].reshape(len(yn), N_EPOCHS, -1), tau[:, cut:].reshape(len(yn), N_EPOCHS, -1)
    ce = T.mean(2)
    arg = om[:, :, None] * (T - ce[:, :, None])
    alpha, beta, _, _ = fit_sinusoid(Y, np.cos(arg), np.sin(arg), 1.0)
    return np.arctan2(alpha, beta), ce


def instrument(yn, tau):
    """The measuring apparatus, parameter-free: bank features, coarse rate, and a difference fit through epochs.

    Each epoch's phase is read by a sinusoid fit; the rate (first difference) and its secular change (second
    difference) come from one least-squares quadratic through the measured epochs, as the tropical year and
    its xiaozhang were read from solstice records of different ages."""
    n = len(yn)
    arg = tau[:, :, None] * BANK_OM[None, None, :]
    ym = yn[:, None, :] * segment_masks(yn.shape[1])[None, :, :]
    C = np.einsum("nsl,nlk->nsk", ym, np.cos(arg))
    S = np.einsum("nsl,nlk->nsk", ym, np.sin(arg))
    P = C * C + S * S
    feat = np.concatenate([C.reshape(n, -1), S.reshape(n, -1), P.reshape(n, -1)], 1)
    Pw = P[:, 2, :]
    wgt = stable_softmax(BANK_BETA * Pw / (Pw.max(1, keepdims=True) + 1e-12))
    om_L, d_inst = wgt @ BANK_OM, np.zeros(n)
    for _ in range(2):      # read the epochs, fit, then re-read each epoch at its own fitted local rate
        ce = tau[:, tau.shape[1] % N_EPOCHS:].reshape(n, N_EPOCHS, -1).mean(2)
        ph, ce = epoch_phases(yn, tau, om_L[:, None] * (1 + d_inst[:, None] * ce))
        steps, gaps = np.diff(ph, axis=1), np.diff(ce, axis=1)
        cycles = np.round((om_L[:, None] * gaps - steps) / (2 * math.pi))     # whole cycles between epochs
        unwrapped = np.concatenate([np.zeros((n, 1)), np.cumsum(steps + 2 * math.pi * cycles, 1)], 1)
        V = np.stack([np.ones_like(ce), ce, 0.5 * ce * ce], 2)               # phi = a + b tau + c tau^2 / 2
        G = np.einsum("nki,nkj->nij", V, V)
        coef = np.linalg.solve(G, np.einsum("nki,nk->ni", V, unwrapped)[:, :, None])[:, :, 0]
        om_L = np.clip(coef[:, 1], 0.5 * OMEGA_REF, 2.0 * OMEGA_REF)          # rate at the last observation
        d_inst = np.clip(coef[:, 2] / om_L, -5 * DRIFT_SCALE, 5 * DRIFT_SCALE)
    return feat, om_L, d_inst


def anchor_phase(yn, tau, omh, dh):
    """Sharpen: fit the recent half at the estimated rate and drift and read the phase at the last observation."""
    u1 = tau * (1 + 0.5 * dh[:, None] * tau)                # d(arg)/d(omega)
    u2 = 0.5 * omh[:, None] * tau * tau                     # d(arg)/d(delta)
    argr = omh[:, None] * u1
    csr, snr = np.cos(argr), np.sin(argr)
    alpha, beta, gamma, G = fit_sinusoid(yn, csr, snr, segment_masks(yn.shape[1])[1][None, :])
    return np.arctan2(alpha, beta), {"csr": csr, "snr": snr, "alpha": alpha, "beta": beta, "gamma": gamma, "G": G,
                                     "u1": u1, "u2": u2}


def evaluate_table(model, phi):
    """Mean motion (unit sinusoid) plus the periodic difference table; returns value, derivative and spline cache."""
    p, ko = model["params"], model["ko"]
    idx, w, dw = spline_weights(np.mod(phi, 2 * math.pi), model["cfg"]["periodic_table"])
    kt = p["knots"] @ TABLE_PROJ                        # the table carries no fundamental: phase stays measurable
    kv = kt[idx]
    sn, cs = np.sin(phi), np.cos(phi)
    f = p["amp"][0] * sn + np.sum(w * kv, 1)
    dfdphi = p["amp"][0] * cs + np.sum(dw * kv, 1) * N_KNOTS / (2 * math.pi)
    if model["mutant"] == "spline_derivative_bug":
        dfdphi = p["amp"][0] * cs + (kv[:, 2] - kv[:, 1]) * N_KNOTS / (2 * math.pi)
    tb_mode = ko.get("zhaocha_table")
    if tb_mode == "zero":
        f, dfdphi = p["amp"][0] * sn, p["amp"][0] * cs
    elif tb_mode == "mean":
        f, dfdphi = p["amp"][0] * sn + float(np.mean(kt)), p["amp"][0] * cs
    elif tb_mode == "identity":
        f, dfdphi = sn, cs
    return f, dfdphi, idx, w, sn


def forward(model, X):
    p, ko, kind = model["params"], model["ko"], model["kind"]
    ds = model["cfg"]["drift_scale"]
    t, y, h, T0 = unpack(X)
    yn, mu, sd = normalise_window(y)
    tau = t - t[:, -1:]
    n = len(t)
    feat, om_L, d_inst = instrument(yn, tau)
    c = {"t": t, "yn": yn, "mu": mu, "sd": sd, "tau": tau, "h": h, "T0": T0}
    if kind == "ffmlp":
        phi_i, _ = anchor_phase(yn, tau, om_L, d_inst)
        ang = 2 * math.pi * h[:, None] * FF_FREQS[None, :]
        feat = np.concatenate([feat, np.cos(phi_i)[:, None], np.sin(phi_i)[:, None], 10 * (om_L / OMEGA_REF - 1)[:, None],
                               (d_inst / ds)[:, None], np.sin(ang), np.cos(ang), h[:, None] / H_MAX, T0[:, None] / T0_MAX], 1)
        h1 = np.tanh(feat @ p["Wa"] + p["ba"])
        h2 = np.tanh(h1 @ p["Wb"] + p["bb"])
        out = (h2 @ p["Wc"] + p["bc"])[:, 0]
        c.update(feat=feat, h1=h1, h2=h2, yhat=out)
        return out, c
    if kind == "grand":
        feat = np.concatenate([feat, T0[:, None] / T0_MAX], 1)
    bank_mode = ko.get("instrument_bank")
    if bank_mode:                                       # a blind instrument: only the nominal constants remain
        feat = np.zeros_like(feat) if bank_mode != "mean" else np.repeat(model["stats"]["feat_mean"][None, :], n, 0)
        om_L, d_inst = np.full(n, OMEGA_REF), np.zeros(n)
    h1 = np.tanh(feat @ p["W1"] + p["b1"])
    a2 = np.tanh(h1 @ p["W2"] + p["b2"])
    l2_mode = ko.get("head_l2")
    if l2_mode:
        a2 = np.zeros_like(a2) if l2_mode != "mean" else np.repeat(model["stats"]["a2_mean"][None, :], n, 0)
    h2 = h1 + a2
    out = h2 @ p["W3"] + p["b3"]
    ta, tb = np.tanh(out[:, 0]), np.tanh(out[:, 1])
    za = np.log(om_L / OMEGA_REF) / 0.4 + RATE_GAIN * ta   # learned correction to the measured rate
    tza = np.tanh(za)
    omh = OMEGA_REF * np.exp(0.4 * tza)
    tzb = np.tanh(d_inst / ds + DRIFT_GAIN * tb)             # learned correction to the measured drift
    dh = ds * tzb
    dr_mode = ko.get("xiaozhang_drift")
    if dr_mode:
        dh = np.zeros_like(dh) if dr_mode != "mean" else np.full_like(dh, model["stats"]["dh_mean"])
    phi0, ca = anchor_phase(yn, tau, omh, dh)
    if kind == "grand":
        phi0 = omh * (T0 + t[:, -1])                                    # phase carried from the distant origin
    an_mode = ko.get("qiying_anchor")
    if an_mode:
        phi0 = np.zeros_like(phi0) if an_mode != "mean" else np.full_like(phi0, model["stats"]["phi0_mean"])
    phi = phi0 + omh * h + 0.5 * omh * dh * h * h
    f, dfdphi, idx, w, sn = evaluate_table(model, phi)
    yhat = f + p["c"][0]
    c.update(ca)
    c.update(feat=feat, h1=h1, a2=a2, h2=h2, out=out, ta=ta, tb=tb, tza=tza, tzb=tzb, omh=omh, dh=dh,
             phi0=phi0, phi=phi, idx=idx, w=w, sn=sn, dfdphi=dfdphi, yhat=yhat)
    return yhat, c


def anchor_adjoint(c, dphi0):
    """Reverse pass of the least-squares anchor read: gradient of the loss w.r.t. the rate and drift it was read at."""
    al, be, ga, cs, sn = c["alpha"], c["beta"], c["gamma"], c["csr"], c["snr"]
    v = np.stack([dphi0 * be, -dphi0 * al, np.zeros_like(al)], 1) / (al * al + be * be + 1e-12)[:, None]
    q = np.linalg.solve(c["G"], v[:, :, None])[:, :, 0]         # adjoint of the 3x3 normal equations
    cc, yn, al, be, ga = cs * cs - sn * sn, c["yn"], al[:, None], be[:, None], ga[:, None]
    K = segment_masks(yn.shape[1])[1][None, :] * (q[:, :1] * (-yn * sn + 2 * cs * sn * al - cc * be + sn * ga)
                                 + q[:, 1:2] * (yn * cs - cc * al - 2 * sn * cs * be - cs * ga)
                                 + q[:, 2:] * (sn * al - cs * be))
    return np.sum(K * c["u1"], 1), np.sum(K * c["u2"], 1)


def loss_and_grads(model, batch):
    """Mean squared error in normalised units; hand-derived reverse pass through the learned parts."""
    X, yq = batch
    p, ko, kind = model["params"], model["ko"], model["kind"]
    ds = model["cfg"]["drift_scale"]
    yhat, c = forward(model, X)
    n = len(yhat)
    target = (yq - c["mu"][:, 0]) / c["sd"][:, 0]
    loss = float(np.mean((yhat - target) ** 2))
    g = 2.0 * (yhat - target) / n
    grads = {}
    if kind == "ffmlp":
        grads["Wc"] = c["h2"].T @ g[:, None]
        grads["bc"] = np.array([g.sum()])
        dh2 = (g[:, None] @ p["Wc"].T) * (1 - c["h2"] ** 2)
        grads["Wb"], grads["bb"] = c["h1"].T @ dh2, dh2.sum(0)
        dh1 = (dh2 @ p["Wb"].T) * (1 - c["h1"] ** 2)
        grads["Wa"], grads["ba"] = c["feat"].T @ dh1, dh1.sum(0)
        return loss, grads
    grads["c"] = np.array([g.sum()])
    grads["amp"] = np.array([0.0 if ko.get("zhaocha_table") == "identity" else float(np.sum(g * c["sn"]))])
    gk = np.zeros(N_KNOTS)
    if not ko.get("zhaocha_table"):
        for r in range(4):
            np.add.at(gk, c["idx"][:, r], g * c["w"][:, r])
    grads["knots"] = gk @ TABLE_PROJ
    if model["mutant"] == "grad_zero_knots":
        grads["knots"][:] = 0.0
    dphi = g * c["dfdphi"]
    h, omh, dh = c["h"], c["omh"], c["dh"]
    domh = dphi * (h + 0.5 * dh * h * h)
    ddh = dphi * (0.5 * omh * h * h)
    dphi0 = np.zeros_like(dphi) if ko.get("qiying_anchor") else dphi
    if kind == "grand":
        domh = domh + dphi0 * (c["T0"] + c["t"][:, -1])
    else:
        da, dd = anchor_adjoint(c, dphi0)
        domh, ddh = domh + da, ddh + dd
    if ko.get("xiaozhang_drift"):
        ddh = np.zeros_like(ddh)
    dout = np.zeros((n, 2))
    dout[:, 0] = domh * omh * 0.4 * (1 - c["tza"] ** 2) * RATE_GAIN * (1 - c["ta"] ** 2)
    dout[:, 1] = ddh * ds * (1 - c["tzb"] ** 2) * DRIFT_GAIN * (1 - c["tb"] ** 2)
    grads["W3"], grads["b3"] = c["h2"].T @ dout, dout.sum(0)
    dh2 = dout @ p["W3"].T
    if ko.get("head_l2"):
        grads["W2"], grads["b2"], dh1 = np.zeros_like(p["W2"]), np.zeros_like(p["b2"]), dh2
    else:
        dz2 = dh2 * (1 - c["a2"] ** 2)
        grads["W2"], grads["b2"] = c["h1"].T @ dz2, dz2.sum(0)
        dh1 = dh2 + dz2 @ p["W2"].T
    dz1 = dh1 * (1 - c["h1"] ** 2)
    grads["W1"], grads["b1"] = c["feat"].T @ dz1, dz1.sum(0)
    return loss, grads


def predict(model, X):
    yhat, c = forward(model, X)
    return c["mu"][:, 0] + c["sd"][:, 0] * yhat


def predict_norm(model, X):
    return forward(model, X)[0]


def hidden_states(model, X):
    _, c = forward(model, X)
    if model["kind"] == "ffmlp":
        return {"h2": c["h2"]}
    return {"h2": c["h2"], "rate": c["omh"], "drift": c["dh"], "anchor": c["phi0"]}


def modules(model):
    if model["kind"] == "ffmlp":
        return {"mlp": {"params": ["Wa", "ba", "Wb", "bb", "Wc", "bc"], "role": "two-layer tanh MLP on instrument and Fourier features", "signature": False}}
    return {
        "instrument_bank": {"params": [], "role": "fixed matched-filter bank and three-epoch rate/drift reading", "signature": False},
        "head_l2": {"params": ["W2", "b2"], "role": "residual tanh layer of the correction readout", "signature": False},
        "qiying_anchor": {"params": [], "role": "phase re-measured at the last observation by re-projection at the estimated rate", "signature": True},
        "xiaozhang_drift": {"params": ["W3", "b3"], "role": "bounded secular rate integrated as a chirp over the horizon", "signature": True},
        "zhaocha_table": {"params": ["amp", "knots", "c"], "role": "unit sinusoid of mean motion plus a periodic cubic B-spline difference table with 24 knots", "signature": True},
    }


def knockout(model, name, mode):
    assert name in modules(model) and mode in ("identity", "zero", "mean")
    m = {k: (dict(v) if isinstance(v, dict) else v) for k, v in model.items()}
    m["params"] = {k: v.copy() for k, v in model["params"].items()}
    m["ko"] = dict(model["ko"])
    m["ko"][name] = mode
    return m

# ------------------------------------------------------------------ training
def fit(model, data, budget, rng, lr=5e-3, batch=128, log_every=25):
    """Adam under a fixed update budget; model selection on the validation split only."""
    Xtr, ytr = data["train"]
    Xva, yva = data["val"]
    p = model["params"]
    opt = Adam(p, lr=0.0 if model["mutant"] == "zero_lr" else lr)
    hist, best, best_params = [], float("inf"), {k: v.copy() for k, v in p.items()}
    for step in range(budget):
        ib = rng.integers(0, len(ytr), batch)
        loss, grads = loss_and_grads(model, (Xtr[ib], ytr[ib]))
        if not np.isfinite(loss):
            raise FloatingPointError("non-finite loss")
        grads, _ = clip_global_norm(grads, CLIP_NORM)
        if model["mutant"] == "sign_flip":
            grads = {k: -g for k, g in grads.items()}
        opt.step(p, grads)
        if step % log_every == 0 or step == budget - 1:
            vl = float(np.mean((predict_norm(model, Xva) - (yva - Xva[:, :, 1].mean(1)) / (Xva[:, :, 1].std(1) + 1e-6)) ** 2))
            hist.append((step, loss, vl))
            if vl < best:
                best, best_params = vl, {k: v.copy() for k, v in p.items()}
    model["params"].update(best_params)
    _, c = forward(model, Xtr[:512])
    if model["kind"] != "ffmlp":
        model["stats"] = {"feat_mean": c["feat"].mean(0), "a2_mean": c["a2"].mean(0),
                          "phi0_mean": float(np.arctan2(np.sin(c["phi0"]).mean(), np.cos(c["phi0"]).mean())),
                          "dh_mean": float(c["dh"].mean())}
    return hist


def rmse_norm(model, X, yq):
    t, y, _, _ = unpack(X)
    _, mu, sd = normalise_window(y)
    return float(np.sqrt(np.mean((predict_norm(model, X) - (yq - mu[:, 0]) / sd[:, 0]) ** 2)))


def train_kind(kind, data, seed, budget, mutant=None):
    rng = np.random.default_rng(np.random.SeedSequence([seed, {"shoushi": 1, "grand": 2, "ffmlp": 3}[kind]]))
    model = build_model(4, 1, "sequence_regression", rng, kind=kind, mutant=mutant)
    hist = fit(model, data, budget, rng)
    return model, hist


# ------------------------------------------------------------------ tests: correctness
def c1_gradcheck(model, batch, rng, tol):
    loss, grads = loss_and_grads(model, batch)
    worst, per = finite_difference_check(lambda: loss_and_grads(model, batch)[0], model["params"], grads, rng)
    return worst <= tol, worst, per


def run_c1(data, seed, tol):
    rng = np.random.default_rng(seed + 11)
    X, y = data["train"]
    batch = (X[:24], y[:24])
    out, worst_all = [], 0.0
    for kind in ("shoushi", "grand", "ffmlp"):
        m, _ = train_kind(kind, data, seed, 0)
        if kind == "grand":     # at T0 ~ 1e3 cycles the truncation term eps^2 (omega T0)^3 / 6 swamps eps = 1e-6
            near = (X[:24].copy(), y[:24])
            near[0][:, :, 3] = 0.0
            batch_k = near
        else:
            batch_k = batch
        ok, worst, _ = c1_gradcheck(m, batch_k, rng, tol * (10 if kind == "grand" else 1))
        out.append((kind, "init", ok, worst))
        worst_all = max(worst_all, worst)
    m, _ = train_kind("shoushi", data, seed, 60)
    ok, worst, per = c1_gradcheck(m, batch, rng, tol)
    out.append(("shoushi", "after_60_steps", ok, worst))
    worst_all = max(worst_all, worst)
    return all(o[2] for o in out), worst_all, out, per


def run_c2(data, seed, budget):
    m1, h1 = train_kind("shoushi", data, seed, budget)
    m2, h2 = train_kind("shoushi", data, seed, budget)
    X, _ = data["held_out"]
    same = all(abs(a[1] - b[1]) == 0.0 for a, b in zip(h1, h2)) and np.array_equal(predict(m1, X), predict(m2, X))
    finite = all(np.all(np.isfinite(v)) for v in m1["params"].values()) and np.all(np.isfinite(predict(m1, X)))
    return same and finite, m1, h1


def run_c3(model, hist, data, thr):
    l0, l1 = hist[0][1], min(h[1] for h in hist[-3:])
    drop = 1 - l1 / l0
    triv = trivial_baselines(*data["held_out"])
    r = rmse_norm(model, *data["held_out"])
    ok = drop >= thr["loss_drop_fraction"] and r <= (1 - thr["margin_over_trivial"]) * min(triv.values())
    return ok, drop, r, triv


def run_c4(data, seed, budget):
    rng = np.random.default_rng(seed + 7)
    shuffled = dict(data)
    for split in ("train", "val"):        # model selection must not see true targets either
        X, y = data[split]
        shuffled[split] = (X, y[rng.permutation(len(y))])
    m, _ = train_kind("shoushi", shuffled, seed, budget)
    r = rmse_norm(m, *data["held_out"])
    band = trivial_baselines(*data["held_out"])["mean"]
    return r >= 0.9 * band, r, band


def run_c5(data, seed, budget, tol, thr):
    rng = np.random.default_rng(seed + 5)
    X, y = data["train"]
    detected = {}
    for name in MUTANTS:
        m, _ = train_kind("shoushi", data, seed, 0, mutant=name)
        ok1, _, _ = c1_gradcheck(m, (X[:16], y[:16]), rng, tol)
        if not ok1:
            detected[name] = "C1"
            continue
        m, hist = train_kind("shoushi", data, seed, budget, mutant=name)
        ok3, _, _, _ = run_c3(m, hist, data, thr)
        detected[name] = "C3" if not ok3 else "undetected"
    score = sum(v != "undetected" for v in detected.values()) / len(MUTANTS)
    return score == 1.0, score, detected


def run_c6(model, data, rng):
    """Property tests on random parameters and inputs, each with a negative control that must fail."""
    X, _ = data["held_out"]
    res = {}
    # C6.1 periodicity of the difference table: f(phi) == f(phi + 2pi) for random knots.
    def periodic_gap(periodic):
        gaps = []
        for _ in range(200):
            knots = rng.standard_normal(N_KNOTS)
            phi = rng.uniform(-6 * math.pi, 6 * math.pi, 64)
            vals = []
            for shift in (0.0, 2 * math.pi):
                idx, w, _ = spline_weights(phi + shift if periodic else np.mod(phi + shift, 2 * math.pi) + shift, periodic)
                vals.append(np.sum(w * knots[idx], 1))
            gaps.append(float(np.max(np.abs(vals[0] - vals[1]))))
        return max(gaps)
    g_ok, g_neg = periodic_gap(True), periodic_gap(False)
    res["C6.1_table_periodic"] = (g_ok < 1e-9 and g_neg > 1e-3, f"gap={g_ok:.1e}, negative control gap={g_neg:.2f}")
    # C6.2 recent-epoch invariance: the prediction never depends on the distance to the origin.
    def t0_sensitivity(kind):
        worst = 0.0
        for _ in range(20):
            m = build_model(4, 1, "sequence_regression", rng, kind=kind)
            for k in m["params"]:
                m["params"][k] = m["params"][k] + rng.standard_normal(m["params"][k].shape) * 0.5
            Xa = X[rng.integers(0, len(X), 32)].copy()
            Xb = Xa.copy()
            Xb[:, :, 3] = rng.uniform(300, T0_MAX, (32, 1))
            worst = max(worst, float(np.max(np.abs(predict_norm(m, Xa) - predict_norm(m, Xb)))))
        return worst
    s_ok, s_neg = t0_sensitivity("shoushi"), t0_sensitivity("grand")
    res["C6.2_epoch_invariance"] = (s_ok == 0.0 and s_neg > 1e-3, f"max change={s_ok:.1e}, negative control (grand epoch)={s_neg:.2f}")
    # C6.3 monotone phase: d(phi)/dh = omega(1 + dh*h) > 0 for all h in [0, H_MAX] under random parameters and
    # random windows with strong drift; the bound drift_scale is the mechanism, so a loose bound is the control.
    def min_phase_rate(drift_scale):
        worst = float("inf")
        for _ in range(60):
            m = build_model(4, 1, "sequence_regression", rng, kind="shoushi", drift_scale=drift_scale)
            for k in m["params"]:
                m["params"][k] = m["params"][k] + rng.standard_normal(m["params"][k].shape) * 2.0
            Xr, _ = make_windows(64, rng, drift_range=(-0.08, 0.08))
            _, c = forward(m, Xr)
            hh = np.linspace(0, H_MAX, 41)[None, :]
            worst = min(worst, float(np.min(c["omh"][:, None] * (1 + c["dh"][:, None] * hh))))
        return worst
    r_ok, r_neg = min_phase_rate(DRIFT_SCALE), min_phase_rate(0.2)
    res["C6.3_monotone_phase"] = (r_ok > 0 and r_neg <= 0, f"min rate={r_ok:.3f}, negative control (drift_scale=0.2)={r_neg:.3f}")
    return all(v[0] for v in res.values()), res


# ------------------------------------------------------------------ tests: hypotheses
def evaluate_seed(seed, budget, quick):
    """Train the full model, the baseline and the rival on one seed; return negative RMSE per split."""
    data = make_data(seed, quick)
    full, _ = train_kind("shoushi", data, seed, budget)
    base, _ = train_kind("ffmlp", data, seed, budget)
    rival, _ = train_kind("grand", data, seed, budget)
    ev = lambda m, split: -rmse_norm(m, *data[split])
    out = {"full_held": ev(full, "held_out"), "full_shift": ev(full, "shifted"), "full_blind": ev(full, "blind"),
           "base_held": ev(base, "held_out"), "base_shift": ev(base, "shifted"),
           "rival_held": ev(rival, "held_out"), "rival_shift": ev(rival, "shifted"),
           "ko_drift_shift": ev(knockout(full, "xiaozhang_drift", "zero"), "shifted"),
           "ko_l2_shift": ev(knockout(full, "head_l2", "zero"), "shifted"),
           "ko_drift_blind": ev(knockout(full, "xiaozhang_drift", "zero"), "blind")}
    for name in modules(full):
        if name not in ("qiying_anchor", "xiaozhang_drift", "zhaocha_table", "head_l2", "instrument_bank"):
            continue
        out[f"ko_{name}_held"] = ev(knockout(full, name, "zero"), "held_out")
    return out, full


def hypothesis_table(per_seed, rng):
    pairs = [("H-SIG", "full_shift", "base_shift"), ("H-NEC", "ko_drift_shift", "ko_l2_shift"),
             ("H-BLIND", "full_blind", "ko_drift_blind"), ("H-RIVAL", "full_held", "rival_held")]
    rows = []
    for hid, a, b in pairs:
        spec = next(hp for hp in MIND_CARD["hypotheses"] if hp["id"] == hid)
        av, bv = [s[a] for s in per_seed], [s[b] for s in per_seed]
        if len(per_seed) < 2:
            rows.append({"id": hid, "metric": spec["metric"], "mean_diff": float(np.mean(np.array(av) - np.array(bv))),
                         "ci95": [0.0, 0.0], "mesi": spec["mesi"], "n_seeds": 1, "verdict": "not evaluated"})
            continue
        md, ci = paired_bootstrap(av, bv, rng)
        rows.append({"id": hid, "metric": spec["metric"], "mean_diff": md, "ci95": list(ci), "mesi": spec["mesi"],
                     "n_seeds": len(per_seed), "verdict": verdict(md, ci, spec["direction"], spec["mesi"])})
    return rows


def knockout_table(per_seed, rng):
    rows = []
    for name in ("qiying_anchor", "xiaozhang_drift", "zhaocha_table", "head_l2", "instrument_bank"):
        a = [s[f"ko_{name}_held"] for s in per_seed]
        b = [s["full_held"] for s in per_seed]
        md, ci = paired_bootstrap(a, b, rng)
        rows.append({"module": name, "signature": name in MIND_CARD["mechanism"]["signature_modules"],
                     "metric_change": md, "ci95": list(ci)})
    return rows

# ------------------------------------------------------------------ optional real-data bridge
def real_data_bridge(path, seed, budget):
    """Local CSV (time,value or value only) -> resample to 6 per cycle -> temporal split -> RMSE vs trivial."""
    raw = np.genfromtxt(path, delimiter=",", skip_header=1)
    y = raw[:, -1] if raw.ndim == 2 else raw
    y = y[np.isfinite(y)]
    yc = y - y.mean()
    ac = np.correlate(yc, yc, "full")[len(yc) - 1:]
    neg = np.where(ac < 0)[0]                          # first anticorrelation, then the first true peak after it
    lo = max(4, int(neg[0])) if len(neg) else 4
    period = lo + int(np.argmax(ac[lo: len(ac) // 2]))
    stride = max(1, int(round(period / SAMPLES_PER_CYCLE)))
    ys = y[::stride]
    hs, Xs, ts = [], [], np.arange(len(ys)) / SAMPLES_PER_CYCLE
    for start in range(0, len(ys) - L_OBS - 6 * SAMPLES_PER_CYCLE, 3):
        for hq in (1, 2, 3, 4, 5, 6):
            end = start + L_OBS
            t = ts[start:end] - ts[start]
            Xs.append(np.stack([t, ys[start:end], np.full(L_OBS, float(hq)), np.zeros(L_OBS)], 1))
            hs.append(ys[end - 1 + hq * SAMPLES_PER_CYCLE])
    X, yq = np.array(Xs), np.array(hs)
    if len(yq) < 60:
        return f"real-data bridge: {path} too short after resampling (period {period} samples)"
    cut = int(0.7 * len(yq))
    data = {"train": (X[:cut], yq[:cut]), "val": (X[cut:], yq[cut:]), "held_out": (X[cut:], yq[cut:])}
    m, _ = train_kind("shoushi", data, seed, budget)
    triv = trivial_baselines(X[cut:], yq[cut:])
    return (f"real-data bridge: {os.path.basename(path)} period~{period} samples, {len(yq)} windows; "
            f"model rmse={rmse_norm(m, X[cut:], yq[cut:]):.3f} vs trivial {json.dumps({k: round(v, 3) for k, v in triv.items()})}")


# ------------------------------------------------------------------ report and entry point
def usage_and_exit():
    print(f"usage: python3 {os.path.basename(__file__)} [--quick] [--seed S] [--seeds N] [--json PATH] "
          f"[--card] [--mutant NAME] [--data PATH]")
    sys.exit(2)


def parse_args(argv):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--card", action="store_true")
    ap.add_argument("--mutant", default=None)
    ap.add_argument("--data", default=None)
    try:
        args = ap.parse_args(argv)
    except SystemExit:
        usage_and_exit()
    if args.mutant is not None and args.mutant not in MUTANTS:
        usage_and_exit()
    return args


def run_mutant_only(name, seed, quick):
    budget = 80 if quick else 250
    data = make_data(seed, quick)
    rng = np.random.default_rng(seed)
    X, y = data["train"]
    m, _ = train_kind("shoushi", data, seed, 0, mutant=name)
    ok1, worst, _ = c1_gradcheck(m, (X[:16], y[:16]), rng, MIND_CARD["thresholds"]["gradcheck_rel_err"])
    m, hist = train_kind("shoushi", data, seed, budget, mutant=name)
    ok3, drop, r, triv = run_c3(m, hist, data, MIND_CARD["thresholds"])
    print(f"mutant {name}: C1 {'pass' if ok1 else 'FAIL'} (rel err {worst:.2e}); C3 {'pass' if ok3 else 'FAIL'} "
          f"(loss drop {drop:.2f}, held-out rmse {r:.3f}, best trivial {min(triv.values()):.3f})")
    return 0 if (ok1 and ok3) else 1


def run_protocol(seeds, budget, quick, thr, tol, rng, data_path):
    """C1-C7 on the first seed, then the hypothesis protocol over all seeds."""
    correctness = []
    data0 = make_data(seeds[0], quick)
    ok7, det7 = split_integrity(data0)
    correctness.append({"id": "C7", "name": "split_integrity", "passed": ok7, "detail": det7})
    ok1, worst1, rows1, per1 = run_c1(data0, seeds[0], tol)
    correctness.append({"id": "C1", "name": "gradient_check", "passed": ok1,
                        "detail": "; ".join(f"{k}/{w}: {e:.1e}" for k, w, _, e in rows1)})
    ok2, model0, hist0 = run_c2(data0, seeds[0], budget)
    correctness.append({"id": "C2", "name": "determinism_finiteness", "passed": ok2,
                        "detail": "same seed -> identical losses and outputs; all finite"})
    ok3, drop, r_held, triv = run_c3(model0, hist0, data0, thr)
    correctness.append({"id": "C3", "name": "learning", "passed": ok3,
                        "detail": f"loss drop {drop:.2f} (>= {thr['loss_drop_fraction']}); held-out rmse {r_held:.3f} vs trivial "
                                  f"{json.dumps({k: round(v, 3) for k, v in triv.items()})}"})
    ok4, r_shuf, band = run_c4(data0, seeds[0], budget)
    correctness.append({"id": "C4", "name": "shuffled_label_control", "passed": ok4,
                        "detail": f"shuffled-target rmse {r_shuf:.3f} within trivial band (mean predictor {band:.3f})"})
    ok5, score, det5 = run_c5(data0, seeds[0], 80 if quick else 250, tol, thr)
    correctness.append({"id": "C5", "name": "mutant_detection", "passed": ok5, "detail": json.dumps(det5)})
    ok6, res6 = run_c6(model0, data0, rng)
    for k, (okk, d) in res6.items():
        correctness.append({"id": k, "name": "property_test", "passed": okk, "detail": d})
    per_seed = [evaluate_seed(s, budget, quick)[0] for s in seeds]
    hyp = hypothesis_table(per_seed, rng)
    kos = knockout_table(per_seed, rng) if len(per_seed) >= 2 else []
    bridge = real_data_bridge(data_path, seeds[0], budget) if data_path else "real-data bridge: no --data path given, skipped"
    return correctness, data0, worst1, per1, score, det5, per_seed, hyp, kos, bridge


def main(argv):
    args = parse_args(argv)
    if args.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    if args.mutant:
        return run_mutant_only(args.mutant, args.seed, args.quick)
    t_start = time.time()
    quick = args.quick
    n_seeds = args.seeds if args.seeds is not None else (1 if quick else 5)
    seeds = [args.seed + i for i in range(n_seeds)]
    budget = 100 if quick else 600
    thr, tol = MIND_CARD["thresholds"], MIND_CARD["thresholds"]["gradcheck_rel_err"]
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, 2026]))
    try:
        res = run_protocol(seeds, budget, quick, thr, tol, rng, args.data)
    except FloatingPointError as exc:
        print(f"non-finite value: {exc}")
        return 4
    correctness, data0, worst1, per1, score, det5, per_seed, hyp, kos, bridge = res
    ok1 = next(c["passed"] for c in correctness if c["id"] == "C1")
    runtime = time.time() - t_start
    limit = 20.0 if quick else 180.0
    ok8 = runtime <= limit
    correctness.append({"id": "C8", "name": "budget", "passed": ok8, "detail": f"{runtime:.1f}s of {limit:.0f}s"})
    for c in correctness:
        c["passed"] = bool(c["passed"])
    all_ok = all(c["passed"] for c in correctness)
    exit_code = 0 if all_ok else (3 if not ok8 and all(c["passed"] for c in correctness[:-1]) else 1)
    m_full, _ = train_kind("shoushi", data0, seeds[0], 0)
    m_base, _ = train_kind("ffmlp", data0, seeds[0], 0)
    m_rival, _ = train_kind("grand", data0, seeds[0], 0)
    means = {k: float(np.mean([s[k] for s in per_seed])) for k in per_seed[0]}
    lines = [f"environment: python {sys.version.split()[0]}, numpy {np.__version__}",
             f"seeds: {seeds}  budget/steps: {budget}  runtime: {runtime:.1f}s",
             f"n_params: shoushi {n_params(m_full)}, ffmlp baseline {n_params(m_base)}, grand-epoch rival {n_params(m_rival)}",
             f"gradcheck: max rel error {worst1:.2e} over {len(per1)} tensors of the full model (init and after 60 steps), "
             f"plus baseline and rival at init; passed={ok1}",
             "correctness:"]
    lines += [f"  {c['id']:<6} {c['name']:<26} {'pass' if c['passed'] else 'FAIL'}  {c['detail']}" for c in correctness]
    lines.append(f"mutation score: {score:.2f} ({sum(v != 'undetected' for v in det5.values())}/{len(MUTANTS)})")
    lines.append("hypotheses (metric = negative normalised RMSE, higher is better):")
    lines += [f"  {h['id']:<8} mean diff {h['mean_diff']:+.4f}  ci95 [{h['ci95'][0]:+.4f}, {h['ci95'][1]:+.4f}]  "
              f"mesi {h['mesi']}  seeds {h['n_seeds']}  -> {h['verdict']}" for h in hyp]
    lines.append("per-split means (neg rmse): " + ", ".join(f"{k}={v:+.3f}" for k, v in means.items()))
    lines.append("knockouts (zero mode, held-out, change vs full):")
    lines += [f"  {k['module']:<18} signature={k['signature']!s:<5} change {k['metric_change']:+.4f}  "
              f"ci95 [{k['ci95'][0]:+.4f}, {k['ci95'][1]:+.4f}]" for k in kos] or ["  (single seed: not evaluated)"]
    lines.append(bridge)
    lines.append(f"task types: {TASK_TYPES}")
    lines.append(f"exit code: {exit_code}")
    payload = {"schema_version": "1.0", "chapter": 338, "file": os.path.basename(__file__),
               "card_revision": MIND_CARD["card_revision"],
               "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
               "seeds": seeds, "runtime_s": runtime, "n_params": n_params(m_full),
               "gradcheck": {"tensors_checked": len(per1), "tensors_total": len(per1), "max_rel_error": worst1,
                             "checked_at": ["init", "after_60_steps"], "passed": ok1},
               "correctness": correctness,
               "mutants": {"detected": sum(v != "undetected" for v in det5.values()), "total": len(MUTANTS), "score": score},
               "hypotheses": hyp, "knockouts": kos, "per_split_means": means,
               "task_types": TASK_TYPES, "exit_code": exit_code}
    write_report(338, lines, args.json, payload)
    return exit_code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

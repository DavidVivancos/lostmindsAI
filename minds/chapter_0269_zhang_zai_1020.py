#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0269 · Zhang Zai 張載 (1020-1077)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 14, Minds 261-280 Available on Amazon https://www.amazon.com/dp/B0HLYSVY5D
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0269_zhang_zai_1020 - Zhang Zai (1020-1077)
# END ATTRIBUTION
"""
TAIXU FIELD (太虛即氣) - a two-aspect conservative neural cellular automaton.

Thesis
    Nothing that disperses is lost: a mind should carry one conserved stuff
    through two aspects, the condensed form it can see and the dispersed void
    it cannot, and read the hidden from the visible and the visible from the
    hidden; a mind that only stores what it has seen "is itself only an image".

Evidence and provenance (archetype_provenance = belief; his own surviving words)
    Zhengmeng 正蒙 (1076), cited by page of Zhang Zai ji 張載集 (Zhonghua 1978).
    D1 p.7  太虛無形，氣之本體；其聚其散，變化之客形爾
    D2 p.7  太虛不能無氣，氣不能不聚而為萬物，萬物不能不散而為太虛
    D3 p.8  氣之聚散於太虛，猶冰凝釋於水，知太虛即氣，則無無
    D4 p.8  氣聚則離明得施而有形…方其散也，安得遽謂之無…方其不形也，有以知明之故
    D5 p.9-10 兩不立則一不可見 / 一物兩體，氣也
    D6 p.24 大其心則能體天下之物，物有未體，則心為有外 / 存象之心，亦象而已
    D7 p.23 形而後有氣質之性，善反之則天地之性存焉
    D8 p.12 陰聚之，陽必散之 … 陽為陰累則相持為雨而降 (his meteorology)
    D9 Rival: a saying in Henan Chengshi yishu 15 (usually assigned to Cheng Yi)
       denies that dispersed qi returns to be reused; Zhu Xi (Wenji 50) notes
       that Master Cheng rejected Hengqu's "return to the origin" (反原).
    D10 Deeds: he planned to test the well-field system in one village before
       the empire (縱不能行之天下，猶可驗之一鄉, Lu Dalin's xingzhuang).

Doctrine -> mechanism -> test (IDs as in MIND_CARD)
    D1-D3  M1 taixu reservoir: dispersed phase kept, moved, re-condensed
           -> C6.1 exact conservation (closed mode)      -> H-SIG, H-NEC
    D4-D6  M1 read only through the condensed phase; never supervised
           -> latent-recovery probe (descriptive)        -> H-NEC
    D7     M2 one shared law at every place; place enters only via terrain
           -> C6.3 ring-translation equivariance          -> H-SIG (new terrain)
    D8     M2 jusan exchange: condense where cold/saturated, disperse where warm
           -> C6.2 non-negativity                         -> native task
    D9     rival: dispersal is a sink, condensation draws on a learned source
           -> H-RIVAL (closed, shifted world)
    blind  open world (rain-out sink, ocean source): closure is false locally
           -> H-BLIND
    D10    one-valley pilot world as the native task (protocol, not evidence)

Research question (memory, retrieval and forgetting / out-of-distribution shift)
    Does an exactly conserved, partially observed two-phase latent let a
    forecaster carry what leaves view and predict its re-appearance, beyond a
    size-matched unconstrained recurrent forecaster, and where does it fail?

Closest prior art and the delta
    Finite-volume neural networks (FINN; Karlbauer et al., ICML 2022) and
    MC-LSTM (Hoedt et al., ICML 2021) conserve mass with learned fluxes or
    redistributions; ConvLSTM/ConvGRU (Shi et al., NeurIPS 2015) is the
    standard unconstrained nowcaster and is implemented here, size-matched.
    Delta: the conserved total is never observed; only the condensed aspect
    is. The unobserved dispersed aspect is spatial, wanders on its own flux
    and re-condenses elsewhere, and the two aspects exchange by a learned law
    shared across places. Overlap: Medium (known parts, new combination).

Blind spot
    Zhang extended cosmic closure to every bounded system. Where a subsystem
    is open (rain leaves the valley, an ocean edge adds vapour), the exact
    ledger cannot represent net gain or loss and should lose to an
    unconstrained forecaster (H-BLIND). This is Cheng Yi's world.

Task (Canliang valley ring, 16 cells, generative process)
    terrain h = 1-2 Gaussian ridges on a ring; day temperature T_t periodic
    (12 steps/day); wind u_t = u0 + 0.3 sin(.); saturation s = .25 exp(1.1(T-1.2h))
    condense C = .35 relu(v - s); disperse E = .3 c sigmoid(8(s-v)/(s+.05))
    vapour moves with u (upwind, Courant<=.9) + diffusion; cloud with .6u
    dawn: all mass visible as fog except 0-10% hidden vapour (nuisance)
    observe cloud only, noise .01; forecast 24 steps from the dawn map
    shifted: 3 ridges, 1.6-3x wetter, 36 steps; open: a sea at cells 0-1
    (terrain -0.6) adds vapour .08/step each, thick cloud rains out .4 relu(c-.25)

Limits
    A synthetic one-dimensional world; the law family is close to the truth by
    design and the baseline is small because it is size-matched. The file is
    a research prototype of one AGI-oriented mechanism, not an AGI, and its
    outputs are not evidence about Zhang Zai.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 3,
    "revision_log": [
        "rev 2 (after one --quick run, before any 5-seed run): (a) the trivial reference changed from mass-scaled "
        "uniform climatology to dawn-persistence climatology (dawn map x climatological visible fraction). C4 failed "
        "(+0.395 > 0.10) because the uniform reference discarded where the dawn fog sat, which any conserving "
        "predictor keeps for free, so a shuffled-label model beat it without learning the input-target map; not "
        "leakage. The new reference is used everywhere (C3, C4 and all hypothesis skills). (b) Every model kind now "
        "gets the same learning-rate grid {0.01, 0.03, 0.1}, selected on the validation split with the base seed and "
        "then fixed for all seeds; revision 1 had omitted the shared tuning grid required by section 9.3. "
        "Hypotheses, directions, mesi and thresholds unchanged.",
        "rev 3 (after a second --quick run, before any 5-seed run): the dawn-persistence reference of rev 2 proved "
        "weaker than the uniform one (valley fog at dawn sits where night cloud does not), and C4 rose to +0.693. "
        "Replaced by the standard nowcasting trivial reference, Lagrangian persistence: the dawn map advected by the "
        "observed wind (rate and diffusion from a small grid fitted on the training split) times a least-squares "
        "visible fraction per step. It is the strongest of the three references and holds everything the "
        "architecture's structure gives for free. Used for C3, C4 and every hypothesis skill. The lr grid of rev 2 "
        "is kept. Hypotheses, directions, mesi and thresholds unchanged."],
    "generation": {"template_version": "codeguidelines 1.0 (15 Sep 2026)", "generator": "Claude (Anthropic)",
                   "generator_version": "claude-opus-5", "date": "2026-09-22"},
    "id": 269, "figure": "Zhang Zai", "born": 1020, "died": 1077, "civilization": "Chinese (Northern Song)",
    "provenance": "belief",
    "thesis": "Nothing that disperses is lost: track one conserved stuff through a visible condensed aspect and an "
              "invisible dispersed aspect, and forecast re-appearance from the invisible.",
    "evidence": [
        {"id": "D1", "claim": "The formless Great Void is qi's own state; condensing and dispersing are transient forms",
         "basis": "primary", "source": "Zhengmeng 1 Taihe, Zhang Zai ji p.7"},
        {"id": "D2", "claim": "Qi must condense into things and things must disperse back into the Void",
         "basis": "primary", "source": "Zhengmeng 1 Taihe, p.7"},
        {"id": "D3", "claim": "Condensation/dispersion in the Void is like ice freezing/melting in water; there is no non-being",
         "basis": "primary", "source": "Zhengmeng 1 Taihe, p.8"},
        {"id": "D4", "claim": "Form is visible only when qi condenses; the dispersed is not nothing and is inferred",
         "basis": "primary", "source": "Zhengmeng 1 Taihe, p.8"},
        {"id": "D5", "claim": "One thing, two aspects; without the two the one cannot be seen",
         "basis": "primary", "source": "Zhengmeng 1 Taihe p.9; 2 Canliang p.10"},
        {"id": "D6", "claim": "Knowledge from seeing and hearing leaves the mind with an outside; a mind storing images is an image",
         "basis": "primary", "source": "Zhengmeng 7 Daxin, p.24"},
        {"id": "D7", "claim": "Endowment (qizhi) arises with form; the shared nature is the same everywhere",
         "basis": "primary", "source": "Zhengmeng 6 Chengming, p.23"},
        {"id": "D8", "claim": "Yin gathers, yang disperses: cloud, rain, thunder and wind as qi dynamics",
         "basis": "primary", "source": "Zhengmeng 2 Canliang, p.12"},
        {"id": "D9", "claim": "Rival: dispersed qi is exhausted; heaven and earth generate new qi without reuse",
         "basis": "primary", "source": "Henan Chengshi yishu 15 (Cheng Yi, usual attribution); Zhu Xi, Wenji 50"},
        {"id": "D10", "claim": "Test the well-field in one village before the empire",
         "basis": "deeds", "source": "Lu Dalin, Hengqu xiansheng xingzhuang, in Zhang Zai ji"},
    ],
    "research_question": {"category": "memory, retrieval and forgetting; out-of-distribution detection and shift",
                          "question": "Does an exactly conserved, partially observed two-phase latent carry what leaves "
                                      "view and forecast its return better than a size-matched unconstrained forecaster?"},
    "mechanism": {
        "name": "Taixu field (conservative two-aspect neural cellular automaton)",
        "family": "continuous-time dynamics; neural cellular automata; world modeling with latent conserved state",
        "signature_modules": ["taixu_reservoir", "jusan_exchange"],
        "closest_prior_art": ["FINN finite-volume neural network (Karlbauer et al. 2022)",
                              "MC-LSTM mass-conserving LSTM (Hoedt et al. 2021)",
                              "ConvLSTM/ConvGRU nowcasting (Shi et al. 2015)",
                              "Flow-Lenia mass-conserving CA (Plantec et al. 2023)"],
        "overlap": "Medium",
        "prior_art_queries": ["mass conserving recurrent neural network", "finite volume neural network learned flux",
                              "conservative neural cellular automaton", "latent unobserved phase conservation forecasting"],
        "contribution_type": "mechanism",
        "delta": "The conserved total is never observed; an unobserved spatial phase wanders and re-condenses under a "
                 "shared learned exchange law, so what leaves view is carried rather than regenerated.",
    },
    "traceability": [
        {"doctrine": "D1-D3", "mechanism": "M1 taixu_reservoir", "property_test": "C6.1", "hypothesis": "H-SIG"},
        {"doctrine": "D4-D6", "mechanism": "M1 taixu_reservoir", "property_test": "latent probe", "hypothesis": "H-NEC"},
        {"doctrine": "D7", "mechanism": "M2 jusan_exchange (shared law)", "property_test": "C6.3", "hypothesis": "H-SIG"},
        {"doctrine": "D8", "mechanism": "M2 jusan_exchange", "property_test": "C6.2", "hypothesis": "H-SIG"},
        {"doctrine": "D9", "mechanism": "rival cheng_source", "property_test": "C6.1 (negative)", "hypothesis": "H-RIVAL"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "The Taixu field forecasts better than a size-matched ConvGRU in a wetter, "
                                     "longer, three-ridge world", "metric": "skill vs Lagrangian persistence",
         "split": "shifted", "comparison": "model - baseline", "direction": "greater", "mesi": 0.05, "seeds": 5},
        {"id": "H-NEC", "statement": "Annihilating the dispersed phase costs more than removing cloud drift",
         "metric": "held-out skill", "comparison": "signature_knockout - matched_knockout",
         "direction": "less", "mesi": 0.05, "seeds": 5},
        {"id": "H-BLIND", "statement": "In an open valley (rain-out sink, ocean source) the conserving field falls "
                                       "behind the unconstrained ConvGRU", "condition": "open world",
         "grounding": "Zhang extended cosmic closure to bounded, open subsystems",
         "metric": "held-out skill", "comparison": "model - baseline", "direction": "less", "mesi": 0.03, "seeds": 5},
        {"id": "H-RIVAL", "statement": "Carrying dispersed qi beats regenerating it (Cheng Yi) in a closed shifted world",
         "metric": "skill vs Lagrangian persistence", "split": "shifted",
         "comparison": "model - rival", "direction": "greater", "mesi": 0.05, "seeds": 5},
    ],
    "tuning": {"lr_grid": [0.01, 0.03, 0.1], "selection": "validation MSE, base seed, same budget for all kinds"},
    "thresholds": {"loss_drop_fraction": 0.5, "margin_over_trivial": 0.2, "shuffle_band": 0.1,
                   "gradcheck_rel_tol": 1e-5, "gradcheck_denominator_floor": 1e-6, "clip_norm": 5.0},
    "probe_predictions": [{"probe": "P7", "expected": "above baseline"}, {"probe": "P9", "expected": "above baseline"},
                          {"probe": "P1", "expected": "above baseline"}, {"probe": "P3", "expected": "equal to baseline"},
                          {"probe": "P8", "expected": "equal to baseline"}, {"probe": "P5", "expected": "below baseline"}],
    "dialectic_links": [
        {"chapter": 299, "relation": "rival", "test": "H-RIVAL (Zhu Xi sided with Cheng Yi against recycled qi)"},
        {"chapter": 265, "relation": "contemporary", "test": "none (Zhou supplies no account of ill; Zhang's qizhi does)"},
        {"chapter": 263, "relation": "contemporary", "test": "none (cross-scale cycles vs conserved ledger)"},
        {"chapter": 602, "relation": "successor", "test": "none yet (Wang Fuzhi, not started)"},
    ],
    "corpus_neighbors": [
        {"chapter": 44, "similarity": 0.0, "difference": "Heraclitus: identity as a conserved measure of flux with no "
                                                          "storage; here a stored, unobserved phase is the point"},
        {"chapter": 100, "similarity": 0.0, "difference": "Liu An: resonance among kinds in a qi field; no conservation, "
                                                           "no hidden phase"},
        {"chapter": 263, "similarity": 0.0, "difference": "Shao Yong: one waxing-waning shape tied across scales; here "
                                                           "no shape prior, a conserved two-phase ledger"},
        {"chapter": 265, "similarity": 0.0, "difference": "Zhou Dunyi: onset gate on departures from rest; here a "
                                                           "transport-exchange field"},
        {"chapter": 267, "similarity": 0.0, "difference": "Su Song: periodic comparison against a moving reference"},
        {"chapter": 273, "similarity": 0.0, "difference": "Shen Kuo: triangulation across channels"},
        {"chapter": 299, "similarity": 0.0, "difference": "Zhu Xi: breadth under composure; principle over qi"},
    ],
    "barometer": {
        "cognitive_processing": ["P1 delayed recall analogue: carrying invisible mass across dry spells"],
        "embodied_cognition": ["not measured (no body)"],
        "world_modeling": ["free-running 24-36 step forecasts", "exact conservation check", "open-world stress test"],
        "consciousness": ["latent-recovery probe: correlation of the never-supervised dispersed phase with true vapour"],
        "language_understanding": ["not measured"],
        "emotional_intelligence": ["not measured"],
        "creativity": ["not measured"],
        "autonomy": ["not measured"],
    },
    "task_types": ["sequence_regression"],
    "applications": [
        {"use": "Cloud and precipitation nowcasting from visible imagery with latent vapour", "sector": "meteorology",
         "dataset": "SEVIR (Veillette et al., NeurIPS 2020)"},
        {"use": "Rainfall-runoff with unobserved catchment storage", "sector": "hydrology",
         "dataset": "CAMELS (Addor et al., HESS 2017)"},
        {"use": "Epidemic nowcasting with unobserved exposed compartment", "sector": "public health research",
         "dataset": "JHU CSSE COVID-19 Data Repository"},
    ],
    "safety_notes": "Synthetic weather only; decision support for research, not operational warnings. No claim of "
                    "replicating Zhang Zai's mind; no generated sentences attributed to him.",
}

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

N_CELLS = 16
DAY = 12
T_TRAIN = 24
T_SHIFT = 36
N_FEAT = 6
MAX_FRAC = 0.5

# BEGIN STANDARD UTILITIES v1.0
def softplus(x):
    return np.logaddexp(0.0, x)


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def logsumexp(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))).squeeze(axis)


def softmax(x, axis=-1):
    z = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return z / np.sum(z, axis=axis, keepdims=True)


class Adam:
    def __init__(self, params, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
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


def grad_check(loss_fn, params, grads, rng, n_entries=20, eps=1e-6, floor=1e-6):
    """Central differences on >= n_entries random entries per tensor plus its largest-gradient entry."""
    worst, out = 0.0, {}
    for k, p in params.items():
        idx = set(int(i) for i in rng.choice(p.size, size=min(n_entries, p.size), replace=False))
        idx.add(int(np.argmax(np.abs(grads[k]).ravel())))
        err_k = 0.0
        for i in idx:
            old = p.flat[i]
            p.flat[i] = old + eps
            lp = loss_fn()
            p.flat[i] = old - eps
            lm = loss_fn()
            p.flat[i] = old
            num = (lp - lm) / (2 * eps)
            ana = grads[k].flat[i]
            err_k = max(err_k, abs(num - ana) / max(abs(num), abs(ana), floor))
        out[k] = err_k
        worst = max(worst, err_k)
    return worst, out


def bootstrap_ci(diffs, rng, n_boot=2000, level=0.95):
    d = np.asarray(diffs, dtype=float)
    means = np.array([np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(n_boot)])
    lo, hi = np.percentile(means, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return float(np.mean(d)), [float(lo), float(hi)]


def verdict(mean, ci, direction, mesi):
    good = (ci[0] > 0 and mean >= mesi) if direction == "greater" else (ci[1] < 0 and mean <= -mesi)
    bad = ci[1] < 0 if direction == "greater" else ci[0] > 0
    return "supported" if good else ("contradicted" if bad else "inconclusive")


def write_report(path, report):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
# END STANDARD UTILITIES


# ----------------------------------------------------------------------------
# Data and tasks: the Canliang valley ring
# ----------------------------------------------------------------------------
WORLDS = {
    "closed": dict(T=T_TRAIN, ridges=(1, 2), mass=(0.25, 0.60), wind=0.6, open=False),
    "shifted": dict(T=T_SHIFT, ridges=(3, 3), mass=(0.60, 1.00), wind=0.7, open=False),
    "open": dict(T=T_TRAIN, ridges=(1, 2), mass=(0.25, 0.60), wind=0.6, open=True),
}


def _shift_field(f, a):
    """Upwind fractional shift by a (cells per step, |a|<=0.9) on the ring; conserves sum exactly."""
    ap = np.maximum(a, 0.0)[:, None]
    am = np.maximum(-a, 0.0)[:, None]
    return (1 - ap - am) * f + ap * np.roll(f, 1, axis=1) + am * np.roll(f, -1, axis=1)


def _diffuse(f, d):
    return f + d * (np.roll(f, 1, axis=1) + np.roll(f, -1, axis=1) - 2 * f)


def make_world(rng, n_ep, kind, shuffle_rng=None):
    w = WORLDS[kind]
    T, N = w["T"], N_CELLS
    pos = np.arange(N)
    h = np.zeros((n_ep, N))
    n_r = rng.integers(w["ridges"][0], w["ridges"][1] + 1, size=n_ep)
    for b in range(n_ep):
        for _ in range(n_r[b]):
            c, a, s = rng.uniform(0, N), rng.uniform(0.5, 1.0), rng.uniform(1.2, 2.5)
            d = np.minimum(np.abs(pos - c), N - np.abs(pos - c))
            h[b] += a * np.exp(-d ** 2 / (2 * s ** 2))
    if w["open"]:
        h[:, :2] = -0.6
    phi = rng.uniform(0, 2, size=n_ep)
    tt = np.arange(T + 1)
    temp = -np.cos(2 * np.pi * (tt[None, :] + phi[:, None]) / DAY)
    u = np.clip(rng.uniform(-w["wind"], w["wind"], size=(n_ep, 1))
                + 0.3 * np.sin(2 * np.pi * tt[None, :] / (2 * DAY) + rng.uniform(0, 2 * np.pi, (n_ep, 1))), -0.9, 0.9)
    mass = rng.uniform(*w["mass"], size=n_ep) * N
    fv = rng.uniform(0.0, 0.1, size=n_ep)
    fog = np.exp(-h) * (1 + 0.3 * rng.standard_normal((n_ep, N))).clip(0.2)
    fog /= fog.sum(1, keepdims=True)
    c = (1 - fv)[:, None] * mass[:, None] * fog
    v = (fv * mass / N)[:, None] * np.ones((1, N))
    C_true, V_true = [c.copy()], [v.copy()]
    for t in range(T):
        theta = temp[:, t + 1][:, None] - 1.2 * h
        s = 0.25 * np.exp(1.1 * theta)
        cond = 0.35 * np.maximum(v - s, 0.0)
        evap = 0.3 * c * sigmoid(8 * (s - v) / (s + 0.05))
        v, c = v - cond + evap, c + cond - evap
        if w["open"]:
            c = c - 0.4 * np.maximum(c - 0.25, 0.0)
            v[:, :2] += 0.08 * (1 + 0.5 * np.sin(2 * np.pi * t / DAY))
        v = _diffuse(_shift_field(v, u[:, t + 1]), 0.05)
        c = _diffuse(_shift_field(c, 0.6 * u[:, t + 1]), 0.02)
        C_true.append(c.copy())
        V_true.append(v.copy())
    C_true, V_true = np.stack(C_true, 1), np.stack(V_true, 1)
    obs = np.maximum(C_true + 0.01 * rng.standard_normal(C_true.shape), 0.0)
    m0 = obs[:, 0].mean(1) / 0.5
    X = np.zeros((n_ep, T, N, N_FEAT))
    X[..., 0] = h[:, None, :]
    X[..., 1] = np.roll(h, 1, axis=1)[:, None, :]
    X[..., 2] = np.roll(h, -1, axis=1)[:, None, :]
    X[..., 3] = temp[:, 1:, None]
    X[..., 4] = u[:, 1:, None]
    X[..., 5] = m0[:, None, None]
    Y = obs[:, 1:].copy()
    if shuffle_rng is not None:
        Y = Y[shuffle_rng.permutation(n_ep)]
    return {"X": X, "o0": obs[:, 0], "u": u[:, 1:], "Y": Y, "C": C_true[:, 1:], "V": V_true[:, 1:],
            "mass": C_true[:, 0].sum(1) + V_true[:, 0].sum(1), "kind": kind}


def subset(d, idx):
    return {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in d.items()}


def fingerprints(d):
    return {hashlib.sha1(np.ascontiguousarray(d["X"][i]).tobytes() + d["o0"][i].tobytes()).hexdigest()
            for i in range(len(d["o0"]))}


def _advect(d, alpha, dif):
    c, out = d["o0"].copy(), []
    for t in range(d["X"].shape[1]):
        c = _diffuse(_shift_field(c, alpha * d["u"][:, t]), dif)
        out.append(c)
    return np.stack(out, 1)


def climatology(train):
    """Trivial reference fitted on the training split: Lagrangian persistence (the dawn map carried by the observed
    wind at rate alpha with diffusion dif, both from a small grid) times a least-squares visible fraction per step.
    It holds all a conserving forecaster gets for free (total, place, wind) and no learned law of condensation."""
    best = None
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        for dif in (0.0, 0.05, 0.1, 0.2):
            L = _advect(train, alpha, dif)
            g = (train["C"] * L).sum((0, 2)) / ((L * L).sum((0, 2)) + 1e-12)
            mse = float(np.mean((L * g[None, :, None] - train["C"]) ** 2))
            if best is None or mse < best[0]:
                best = (mse, alpha, dif, g)
    return {"alpha": best[1], "dif": best[2], "g": best[3]}


def trivial_pred(ref, d):
    T, g = d["C"].shape[1], ref["g"]
    gg = np.array([g[t] if t < len(g) else g[t - DAY * ((t - len(g)) // DAY + 1)] for t in range(T)])
    return _advect(d, ref["alpha"], ref["dif"]) * gg[None, :, None]


def skill(pred, d, clim):
    mse = np.mean((pred - d["C"]) ** 2)
    ref = np.mean((trivial_pred(clim, d) - d["C"]) ** 2)
    return float(1.0 - mse / ref)


# ----------------------------------------------------------------------------
# Reverse-mode tape (file-local): values, closures, accumulation
# ----------------------------------------------------------------------------
class Var:
    __slots__ = ("v", "g")

    def __init__(self, v):
        self.v, self.g = v, None


_TAPE = []
_REC = [False]


def _val(x):
    return x.v if isinstance(x, Var) else x


def _acc(x, g):
    if isinstance(x, Var):
        x.g = g if x.g is None else x.g + g


def _unb(g, shape):
    while g.ndim > len(shape):
        g = g.sum(0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(i, keepdims=True)
    return g


def _op(value, back):
    out = Var(value)
    if _REC[0]:
        _TAPE.append((out, back))
    return out


def add(a, b):
    sa, sb = np.shape(_val(a)), np.shape(_val(b))
    return _op(_val(a) + _val(b), lambda g: (_acc(a, _unb(g, sa)), _acc(b, _unb(g, sb))))


def sub(a, b):
    sa, sb = np.shape(_val(a)), np.shape(_val(b))
    return _op(_val(a) - _val(b), lambda g: (_acc(a, _unb(g, sa)), _acc(b, -_unb(g, sb))))


def mul(a, b):
    av, bv = _val(a), _val(b)
    sa, sb = np.shape(av), np.shape(bv)
    return _op(av * bv, lambda g: (_acc(a, _unb(g * bv, sa)), _acc(b, _unb(g * av, sb))))


def matmul(x, W):
    xv, Wv = _val(x), _val(W)
    return _op(xv @ Wv, lambda g: (_acc(x, g @ Wv.T),
                                   _acc(W, np.tensordot(xv, g, axes=(list(range(xv.ndim - 1)),) * 2))))


def tanh(x):
    y = np.tanh(_val(x))
    return _op(y, lambda g: _acc(x, g * (1 - y * y)))


def sig(x):
    y = sigmoid(_val(x))
    return _op(y, lambda g: _acc(x, g * y * (1 - y)))


def splus(x):
    xv = _val(x)
    return _op(softplus(xv), lambda g: _acc(x, g * sigmoid(xv)))


def roll(x, k):
    return _op(np.roll(_val(x), k, axis=1), lambda g: _acc(x, np.roll(g, -k, axis=1)))


def cellmean(x):
    """Replace a field by its ring average (knockout 'mean')."""
    xv = _val(x)
    n = xv.shape[1]
    return _op(np.repeat(xv.mean(1, keepdims=True), n, axis=1),
               lambda g: _acc(x, np.repeat(g.sum(1, keepdims=True) / n, n, axis=1)))


def take(x, j):
    xv = _val(x)

    def back(g):
        z = np.zeros_like(xv)
        z[..., j] = g
        _acc(x, z)
    return _op(xv[..., j], back)


def cat(xs):
    vals = [_val(x) for x in xs]
    widths = np.cumsum([0] + [v.shape[-1] for v in vals])

    def back(g):
        for i, x in enumerate(xs):
            _acc(x, g[..., widths[i]:widths[i + 1]])
    return _op(np.concatenate(vals, axis=-1), back)


def mse_loss(preds, Y, scale):
    P = np.stack([_val(p) for p in preds], 1)
    diff = (P - Y) / scale
    n = diff.size

    def back(g):
        for t, p in enumerate(preds):
            _acc(p, g * 2 * diff[:, t] / (scale * n))
    return _op(float(np.sum(diff ** 2) / n), back)


def backward(out):
    out.g = 1.0
    for node, back in reversed(_TAPE):
        if node.g is not None:
            back(node.g)
    _TAPE.clear()


# ----------------------------------------------------------------------------
# Model: the Taixu field
# ----------------------------------------------------------------------------
def _init_mlp(rng, p, hidden, n_out):
    p["W1"] = rng.standard_normal((N_FEAT, hidden)) / math.sqrt(N_FEAT)
    p["b1"] = np.zeros(hidden)
    p["W2"] = rng.standard_normal((hidden, n_out)) * 0.3 / math.sqrt(hidden)
    p["b2"] = np.array([-1.0, 0.0, 0.0])


def build_model(in_dim, out_dim, task_type, rng, kind="taixu", hidden=8, **cfg):
    """kind: taixu (signature), cheng (rival: sink + learned source), convgru (size-matched baseline).
    Native input: in_dim = N_FEAT per cell, out_dim = N_CELLS. Otherwise a generic adapter maps a (B,T,in_dim)
    sequence to per-cell drivers and an initial condensed field for out_dim cells."""
    if task_type not in TASK_TYPES:
        raise ValueError(task_type)
    p = {}
    generic = not (in_dim == N_FEAT and out_dim == N_CELLS and not cfg.get("force_generic"))
    if kind in ("taixu", "cheng"):
        _init_mlp(rng, p, hidden, 3)
        p["ac"], p["dc"] = np.array(0.0), np.array(0.0)
        if kind == "taixu":
            p["beta"], p["av"], p["dv"] = np.array(0.0), np.array(1.0), np.array(0.0)
    else:
        H = cfg.get("gru_hidden", 2)
        for gname in "zrh":
            p["W" + gname] = rng.standard_normal((N_FEAT + 1, H)) * 0.4
            p["U" + gname] = rng.standard_normal((3 * H, H)) * 0.3
            p["b" + gname] = np.zeros(H)
        p["Wo"], p["bo"] = rng.standard_normal((H, 1)) * 0.4, np.array([-1.0])
        p["W0"], p["b0"] = rng.standard_normal((1, H)) * 0.5, np.zeros(H)
    if generic:
        p["Pg"] = rng.standard_normal((in_dim, out_dim * N_FEAT)) / math.sqrt(in_dim)
        p["Pi"], p["bi"] = rng.standard_normal((in_dim, out_dim)) / math.sqrt(in_dim), np.zeros(out_dim)
    return {"kind": kind, "params": p, "ko": {}, "generic": generic, "n_cells": out_dim, "task_type": task_type}


def _drivers(model, batch, t):
    if not model["generic"]:
        return batch["X"][:, t]
    B, n = batch["X"].shape[0], model["n_cells"]
    raw = matmul(batch["X"][:, t], model["params"]["Pg"])
    return _op(np.tanh(_val(raw)).reshape(B, n, N_FEAT),
               lambda g, r=raw, y=np.tanh(_val(raw)): _acc(r, (g.reshape(B, -1)) * (1 - y * y)))


def _initial(model, batch):
    if not model["generic"]:
        return batch["o0"]
    return splus(add(matmul(batch["X"][:, 0], model["params"]["Pi"]), model["params"]["bi"]))


def _flux(f, u, a, d):
    """Antisymmetric edge flux i -> i+1 (對必反其為): swapping ends and reversing wind negates it."""
    right = roll(f, -1)
    up = sub(mul(np.maximum(u, 0.0), f), mul(np.maximum(-u, 0.0), right))
    return add(mul(a, up), mul(d, sub(f, right)))


def _transport(f, u, a, d, ko=None):
    """Knockouts: 'zero'/'identity' stop transport; 'mean' spreads the field evenly round the ring."""
    if ko in ("zero", "identity"):
        return f
    if ko == "mean":
        return cellmean(f)
    F = _flux(f, u[:, None], a, d)
    return add(sub(f, F), roll(F, 1))


def _mlp(p, x):
    z = tanh(add(matmul(x, p["W1"]), p["b1"]))
    o = add(matmul(z, p["W2"]), p["b2"])
    return z, take(o, 0), take(o, 1), take(o, 2)


def _scaled(p, name, lo, hi):
    return add(lo, mul(hi - lo, sig(p[name])))


def step_taixu(model, c, v, x, u):
    """One tick: 聚散 exchange between the two aspects of each cell, then both aspects move (游氣紛擾)."""
    p, ko, mut = model["params"], model["ko"], model.get("mutant_law")
    _, o_s, o_c, o_e = _mlp(p, x)
    s = splus(o_s)
    if mut == "posbias":
        s = add(s, np.linspace(0.0, 1.0, _val(s).shape[1])[None, :])
    beta = mul(4.0, splus(p["beta"]))
    gap = sub(v, s)
    kc = mul(MAX_FRAC, sig(add(o_c, mul(beta, gap))))
    ke = mul(MAX_FRAC, sig(sub(o_e, mul(beta, gap))))
    if ko.get("jusan_exchange") == "mean":
        kc, ke = cellmean(kc), cellmean(ke)
    C, E = mul(kc, v), mul(ke, c)
    if ko.get("jusan_exchange") == "zero":
        C, E = mul(0.0, C), mul(0.0, E)
    v_new = add(sub(v, C), E) if mut != "leak" else add(sub(v, mul(1.05, C)), E)
    c_new = add(sub(c, E), C)
    kv = ko.get("taixu_reservoir")
    if kv == "zero":
        v_new = mul(0.0, v_new)
    v_new = _transport(v_new, u, _scaled(p, "av", 0.0, 1.0), _scaled(p, "dv", 0.0, 0.05), kv)
    c_new = _transport(c_new, u, _scaled(p, "ac", 0.0, 1.0), _scaled(p, "dc", 0.0, 0.05), ko.get("kexing_drift"))
    return c_new, v_new


def step_cheng(model, c, x, u):
    """Rival (程頤): dispersal is a sink; condensation draws on a learned generative source 生生不窮."""
    p, ko = model["params"], model["ko"]
    _, o_s, o_c, o_e = _mlp(p, x)
    G = mul(mul(MAX_FRAC, sig(o_c)), splus(o_s))
    E = mul(mul(MAX_FRAC, sig(o_e)), c)
    c_new = add(sub(c, E), G)
    return _transport(c_new, u, _scaled(p, "ac", 0.0, 1.0), _scaled(p, "dc", 0.0, 0.05), ko.get("kexing_drift"))


def step_gru(model, h, x, o0):
    p = model["params"]
    xin = cat([x, o0[..., None] if isinstance(o0, np.ndarray) else _op(_val(o0)[..., None],
                                                                        lambda g, o=o0: _acc(o, g[..., 0]))])
    nb = cat([roll(h, 1), h, roll(h, -1)])
    z = sig(add(add(matmul(xin, p["Wz"]), matmul(nb, p["Uz"])), p["bz"]))
    r = sig(add(add(matmul(xin, p["Wr"]), matmul(nb, p["Ur"])), p["br"]))
    rh = mul(r, h)
    nb2 = cat([roll(rh, 1), rh, roll(rh, -1)])
    hh = tanh(add(add(matmul(xin, p["Wh"]), matmul(nb2, p["Uh"])), p["bh"]))
    h_new = add(h, mul(z, sub(hh, h)))
    out = splus(add(matmul(h_new, p["Wo"]), p["bo"]))
    return h_new, take(out, 0)


def forward(model, batch, keep_state=False):
    T = batch["X"].shape[1]
    c0 = _initial(model, batch)
    preds, states = [], []
    if model["kind"] == "convgru":
        p = model["params"]
        h = tanh(add(matmul(_op(_val(c0)[..., None], lambda g: _acc(c0, g[..., 0])), p["W0"]), p["b0"]))
        for t in range(T):
            h, y = step_gru(model, h, _drivers(model, batch, t), c0)
            preds.append(y)
            if keep_state:
                states.append(_val(h))
        return preds, states
    c, v = c0, np.zeros_like(_val(c0))
    for t in range(T):
        x = _drivers(model, batch, t)
        u = batch["u"][:, t] if "u" in batch else np.zeros(batch["X"].shape[0])
        if model["kind"] == "taixu":
            c, v = step_taixu(model, c, v, x, u)
            if keep_state:
                states.append(np.stack([_val(c), _val(v)], -1))
        else:
            c = step_cheng(model, c, x, u)
            if keep_state:
                states.append(_val(c)[..., None])
        preds.append(c)
    return preds, states


def predict(model, X):
    _REC[0] = False
    preds, _ = forward(model, X)
    return np.stack([_val(p) for p in preds], 1)


def hidden_states(model, X):
    _REC[0] = False
    return np.stack(forward(model, X, keep_state=True)[1], 1)


def loss_and_grads(model, batch, scale=0.25):
    vars_ = {k: Var(v) for k, v in model["params"].items()}
    shadow = dict(model, params=vars_)
    _TAPE.clear()
    _REC[0] = True
    preds, _ = forward(shadow, batch)
    L = mse_loss(preds, batch["Y"], scale)
    backward(L)
    _REC[0] = False
    grads = {k: (np.zeros_like(model["params"][k]) if vars_[k].g is None
                 else np.asarray(vars_[k].g, dtype=float).reshape(model["params"][k].shape)) for k in vars_}
    if model.get("mutant") == "zero_grad_W1" and "W1" in grads:
        grads["W1"] = np.zeros_like(grads["W1"])
    if model.get("mutant") == "zero_grad_W1" and "Wz" in grads:
        grads["Wz"] = np.zeros_like(grads["Wz"])
    return L.v, grads


def loss_only(model, batch, scale=0.25):
    _REC[0] = False
    preds, _ = forward(model, batch)
    P = np.stack([_val(p) for p in preds], 1)
    return float(np.mean(((P - batch["Y"]) / scale) ** 2))


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


TASK_TYPES = ["sequence_regression"]


# ----------------------------------------------------------------------------
# Registries: modules, knockouts, mutants
# ----------------------------------------------------------------------------
def modules(model):
    k = model["kind"]
    if k == "taixu":
        return {"taixu_reservoir": {"params": ["av", "dv"], "signature": True,
                                    "role": "unobserved conserved phase with antisymmetric transport (hidden store)"},
                "jusan_exchange": {"params": ["W1", "b1", "W2", "b2", "beta"], "signature": True,
                                   "role": "gated two-way phase exchange with shared per-cell law"},
                "kexing_drift": {"params": ["ac", "dc"], "signature": False,
                                 "role": "advection-diffusion of the observed phase"}}
    if k == "cheng":
        return {"cheng_source": {"params": ["W1", "b1", "W2", "b2"], "signature": False,
                                 "role": "learned source and sink on the observed field"},
                "kexing_drift": {"params": ["ac", "dc"], "signature": False, "role": "advection-diffusion"}}
    return {"convgru": {"params": list(model["params"]), "signature": False, "role": "1-D convolutional GRU"}}


def knockout(model, name, mode):
    if name not in modules(model) or mode not in ("identity", "zero", "mean"):
        raise ValueError((name, mode))
    ko = dict(model["ko"])
    ko[name] = mode
    return dict(model, ko=ko, params={k: v.copy() for k, v in model["params"].items()})


MUTANTS = {"sign_flip": "optimizer ascends the loss", "zero_lr": "learning rate set to zero",
           "zero_grad_W1": "one tensor's analytic gradient zeroed"}


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def fit(model, data, budget, rng, val=None, lr=0.03, batch=32, clip=5.0, log=None):
    opt = Adam(model["params"], lr=0.0 if model.get("mutant") == "zero_lr" else lr)
    sign = -1.0 if model.get("mutant") == "sign_flip" else 1.0
    n = data["X"].shape[0]
    best, best_p, first, last = np.inf, None, None, None
    for it in range(budget):
        idx = rng.choice(n, size=min(batch, n), replace=False)
        L, g = loss_and_grads(model, subset(data, idx))
        if not np.isfinite(L):
            raise FloatingPointError("non-finite loss")
        clip_global_norm(g, clip)
        opt.step(model["params"], g, sign)
        first = L if first is None else first
        if log is not None:
            log.append(L)
        if val is not None and (it % 20 == 19 or it == budget - 1):
            lv = loss_only(model, val)
            if lv < best:
                best, best_p = lv, {k: v.copy() for k, v in model["params"].items()}
    if best_p is not None:
        model["params"] = best_p
    last = loss_only(model, data)
    return first, last


LRS = {"taixu": 0.03, "cheng": 0.03, "convgru": 0.03}


def tune_lrs(data, budget, seed, quick):
    """Same grid, same budget, same selection rule for model, rival and size-matched baseline."""
    grid = [0.03] if quick else MIND_CARD["tuning"]["lr_grid"]
    chosen = {}
    for kind in ("taixu", "cheng", "convgru"):
        best = (np.inf, None)
        for lr in grid:
            m = build_model(N_FEAT, N_CELLS, "sequence_regression", np.random.default_rng(seed), kind=kind)
            fit(m, data["train"], budget, np.random.default_rng(seed + 1), val=data["val"], lr=lr)
            best = min(best, (loss_only(m, data["val"]), lr))
        chosen[kind] = best[1]
    LRS.update(chosen)
    return chosen


def train_kind(kind, train, val, budget, rng, mutant=None):
    m = build_model(N_FEAT, N_CELLS, "sequence_regression", rng, kind=kind)
    if mutant:
        m["mutant"] = mutant
    init_loss = loss_only(m, train)
    fit(m, train, budget, rng, val=val, lr=LRS[kind])
    return m, init_loss, loss_only(m, train)


# ----------------------------------------------------------------------------
# Tests: correctness first, then hypotheses
# ----------------------------------------------------------------------------
def gradcheck_model(model, batch, rng):
    L, g = loss_and_grads(model, batch)
    worst, per = grad_check(lambda: loss_only(model, batch), model["params"], g, rng,
                            floor=MIND_CARD["thresholds"]["gradcheck_denominator_floor"])
    return worst, per


def c1_gradients(rng, data, tol):
    rows, worst_all, n_checked = [], 0.0, 0
    small = subset(data, np.arange(3))
    small = dict(small, X=small["X"][:, :8], u=small["u"][:, :8], Y=small["Y"][:, :8])
    for kind in ("taixu", "cheng", "convgru"):
        m = build_model(N_FEAT, N_CELLS, "sequence_regression", rng, kind=kind)
        for when in ("init", "after_60_steps"):
            if when != "init":
                fit(m, data, 60, rng, lr=LRS[kind])
            w, per = gradcheck_model(m, small, rng)
            worst_all, n_checked = max(worst_all, w), n_checked + len(per)
            rows.append((kind, when, w))
    gen = build_model(5, 4, "sequence_regression", rng, kind="taixu")
    gb = {"X": rng.standard_normal((3, 6, 5)), "u": rng.uniform(-0.9, 0.9, (3, 6)), "Y": rng.uniform(0, 1, (3, 6, 4))}
    w, per = gradcheck_model(gen, gb, rng)
    rows.append(("taixu-generic", "init", w))
    worst_all, n_checked = max(worst_all, w), n_checked + len(per)
    return worst_all <= tol, worst_all, n_checked, rows


def random_state(rng, B=8, T=10):
    X = rng.standard_normal((B, T, N_CELLS, N_FEAT))
    return {"X": X, "u": rng.uniform(-0.9, 0.9, (B, T)), "o0": rng.uniform(0, 2, (B, N_CELLS)),
            "Y": np.zeros((B, T, N_CELLS))}


def conservation_violation(model, rng, draws=40):
    worst = 0.0
    for _ in range(draws):
        m = dict(model, params={k: v + rng.standard_normal(v.shape) * 3.0 for k, v in model["params"].items()})
        b = random_state(rng)
        _REC[0] = False
        c, v = b["o0"], np.zeros_like(b["o0"])
        tot0 = c.sum(1)
        for t in range(b["X"].shape[1]):
            c, v = step_taixu(m, c, v, b["X"][:, t], b["u"][:, t])
            c, v = _val(c), _val(v)
            worst = max(worst, float(np.max(np.abs(c.sum(1) + v.sum(1) - tot0) / tot0)))
    return worst


def positivity_violation(model, rng, draws=40, widen=1.0):
    worst = 0.0
    for _ in range(draws):
        m = dict(model, params={k: v + rng.standard_normal(v.shape) * 3.0 for k, v in model["params"].items()})
        b = random_state(rng)
        _REC[0] = False
        c, v = b["o0"], np.zeros_like(b["o0"])
        for t in range(b["X"].shape[1]):
            c, v = step_taixu(m, c, v, b["X"][:, t], b["u"][:, t] * widen)
            c, v = _val(c), _val(v)
            worst = max(worst, float(-min(c.min(), v.min(), 0.0)))
    return worst


def equivariance_violation(model, rng, draws=20):
    """Shared law at every place (天地之性): rolling the valley rolls the forecast."""
    worst = 0.0
    for _ in range(draws):
        m = dict(model, params={k: v + rng.standard_normal(v.shape) for k, v in model["params"].items()})
        b = random_state(rng, B=4, T=8)
        k = int(rng.integers(1, N_CELLS))
        rb = dict(b, X=np.roll(b["X"], k, axis=2), o0=np.roll(b["o0"], k, axis=1))
        y1, y2 = predict(m, b), predict(m, rb)
        worst = max(worst, float(np.max(np.abs(np.roll(y1, k, axis=2) - y2))))
    return worst


def run_correctness(args, rngs, data, report):
    rows, t0 = [], time.time()
    tol = MIND_CARD["thresholds"]["gradcheck_rel_tol"]
    ok1, worst, n_checked, grows = c1_gradients(rngs["c1"], data["train"], tol)
    report["gradcheck"] = {"tensors_checked": n_checked, "tensors_total": n_checked, "max_rel_error": worst,
                           "checked_at": ["init", "after_training_steps"], "passed": bool(ok1),
                           "rows": [f"{k}@{w}: {e:.2e}" for k, w, e in grows]}
    rows.append(("C1", "gradient_check", ok1, f"max rel err {worst:.2e} over {n_checked} tensor checks"))

    m1, _, _ = train_kind("taixu", data["train"], data["val"], 20, np.random.default_rng(7))
    m2, _, _ = train_kind("taixu", data["train"], data["val"], 20, np.random.default_rng(7))
    p1, p2 = predict(m1, data["test"]), predict(m2, data["test"])
    finite = all(np.all(np.isfinite(v)) for v in m1["params"].values()) and np.all(np.isfinite(p1))
    ok2 = bool(np.array_equal(p1, p2) and finite)
    rows.append(("C2", "determinism_and_finiteness", ok2, "identical predictions for identical seeds; all finite"))

    rows.append(run_c3_c4(args, rngs, data, report))
    ok4, detail4 = report.pop("_c4")
    rows.append(("C4", "shuffled_label_control", ok4, detail4))

    det, total, mrows = 0, 0, []
    for name in MUTANTS:
        mm = build_model(N_FEAT, N_CELLS, "sequence_regression", np.random.default_rng(11), kind="taixu")
        mm["mutant"] = name
        L0 = loss_only(mm, data["train"])
        fit(mm, data["train"], args.mut_steps, np.random.default_rng(12), lr=LRS["taixu"])
        L1 = loss_only(mm, data["train"])
        c3_fail = not (L1 <= (1 - MIND_CARD["thresholds"]["loss_drop_fraction"]) * L0)
        _, g = loss_and_grads(mm, subset(data["train"], np.arange(3)))
        w, _ = grad_check(lambda: loss_only(mm, subset(data["train"], np.arange(3))), mm["params"], g,
                          np.random.default_rng(13))
        c1_fail = w > tol
        total += 1
        det += int(c3_fail or c1_fail)
        mrows.append(f"{name}: C1 {'fail' if c1_fail else 'pass'}, C3 {'fail' if c3_fail else 'pass'}")
    report["mutants"] = {"detected": det, "total": total, "score": det / total, "rows": mrows}
    rows.append(("C5", "mutant_detection", det == total, f"{det}/{total}: " + "; ".join(mrows)))

    base = build_model(N_FEAT, N_CELLS, "sequence_regression", np.random.default_rng(21), kind="taixu")
    cv = conservation_violation(base, np.random.default_rng(22))
    cv_neg = conservation_violation(dict(base, mutant_law="leak"), np.random.default_rng(22))
    rows.append(("C6.1", "conservation_closed_mode", cv < 1e-10 and cv_neg > 1e-4,
                 f"max rel drift {cv:.1e} over 40 random-parameter draws; leaky negative control {cv_neg:.1e}"))
    pv = positivity_violation(base, np.random.default_rng(23))
    pv_neg = positivity_violation(base, np.random.default_rng(23), widen=2.5)
    rows.append(("C6.2", "non_negativity", pv == 0.0 and pv_neg > 0.0,
                 f"min value >= 0 (violation {pv:.1e}); wind outside the declared domain gives {pv_neg:.1e}"))
    ev = equivariance_violation(base, np.random.default_rng(24))
    ev_neg = equivariance_violation(dict(base, mutant_law="posbias"), np.random.default_rng(24))
    rows.append(("C6.3", "ring_translation_equivariance", ev < 1e-10 and ev_neg > 1e-4,
                 f"max deviation {ev:.1e}; position-tagged negative control {ev_neg:.1e}"))

    tr, te = fingerprints(data["train"]), set()
    for k in ("val", "test", "shift", "open_test"):
        te |= fingerprints(data[k])
    ok7 = len(tr & te) == 0
    rows.append(("C7", "split_integrity", ok7, f"{len(tr & te)} shared episodes across {len(tr)} train / {len(te)} eval"))
    report["_c_elapsed"] = time.time() - t0
    return rows


def run_c3_c4(args, rngs, data, report):
    th = MIND_CARD["thresholds"]
    m, L0, L1 = train_kind("taixu", data["train"], data["val"], args.budget, rngs["c3"])
    sk = skill(predict(m, data["test"]), data["test"], data["clim"])
    ok3 = L1 <= (1 - th["loss_drop_fraction"]) * L0 and sk >= th["margin_over_trivial"]
    shuf = make_world(rngs["c4data"], len(data["train"]["o0"]), "closed", shuffle_rng=rngs["c4perm"])
    ms, _, _ = train_kind("taixu", shuf, None, args.budget, rngs["c4"])
    sks = skill(predict(ms, data["test"]), data["test"], data["clim"])
    report["_c4"] = (bool(sks <= th["shuffle_band"]),
                     f"held-out skill with permuted targets {sks:+.3f} (band <= {th['shuffle_band']}); "
                     f"true-label skill {sk:+.3f}")
    return ("C3", "learning", bool(ok3), f"loss {L0:.3f} -> {L1:.3f}; held-out skill {sk:+.3f} vs trivial 0")


def latent_r(m, d):
    V = hidden_states(m, d)[..., 1]
    a, b = V.ravel() - V.mean(), d["V"].ravel() - d["V"].mean()
    return float(a @ b / math.sqrt((a @ a) * (b @ b) + 1e-12))


def run_seed(seed, args, data):
    ss = np.random.SeedSequence(seed)
    r = [np.random.default_rng(s) for s in ss.spawn(5)]
    out = {}
    tx, _, _ = train_kind("taixu", data["train"], data["val"], args.budget, r[0])
    cg, _, _ = train_kind("convgru", data["train"], data["val"], args.budget, r[1])
    ch, _, _ = train_kind("cheng", data["train"], data["val"], args.budget, r[2])
    for name, m in (("taixu", tx), ("convgru", cg), ("cheng", ch)):
        out[name + "_held"] = skill(predict(m, data["test"]), data["test"], data["clim"])
        out[name + "_shift"] = skill(predict(m, data["shift"]), data["shift"], data["clim"])
    for mod in ("taixu_reservoir", "jusan_exchange", "kexing_drift"):
        out["ko_" + mod] = skill(predict(knockout(tx, mod, "zero"), data["test"]), data["test"], data["clim"])
    out["latent_r_held"] = latent_r(tx, data["test"])
    txo, _, _ = train_kind("taixu", data["open_train"], data["open_val"], args.budget, r[3])
    cgo, _, _ = train_kind("convgru", data["open_train"], data["open_val"], args.budget, r[4])
    out["taixu_open"] = skill(predict(txo, data["open_test"]), data["open_test"], data["open_clim"])
    out["convgru_open"] = skill(predict(cgo, data["open_test"]), data["open_test"], data["open_clim"])
    out["_model"] = tx
    return out


def run_hypotheses(args, data, report, base_seed):
    seeds = [base_seed + i for i in range(args.seeds)]
    per = []
    for s in seeds:
        t0 = time.time()
        per.append(run_seed(s, args, data))
        print(f"  seed {s}: done in {time.time() - t0:.1f}s", flush=True)
    specs = {"H-SIG": ("taixu_shift", "convgru_shift"), "H-NEC": ("ko_taixu_reservoir", "ko_kexing_drift"),
             "H-BLIND": ("taixu_open", "convgru_open"), "H-RIVAL": ("taixu_shift", "cheng_shift")}
    rng = np.random.default_rng(base_seed + 999)
    hyp = []
    for h in MIND_CARD["hypotheses"]:
        a, b = specs[h["id"]]
        diffs = [p[a] - p[b] for p in per]
        mean, ci = bootstrap_ci(diffs, rng)
        v = verdict(mean, ci, h["direction"], h["mesi"]) if len(seeds) >= 5 and not args.quick else "not evaluated"
        hyp.append({"id": h["id"], "metric": h["metric"], "mean_diff": mean, "ci95": ci, "mesi": h["mesi"],
                    "n_seeds": len(seeds), "verdict": v, "per_seed": [round(x, 4) for x in diffs]})
    kos = []
    for mod, sig_ in (("taixu_reservoir", True), ("jusan_exchange", True), ("kexing_drift", False)):
        mean, ci = bootstrap_ci([p["ko_" + mod] - p["taixu_held"] for p in per], rng)
        kos.append({"module": mod, "signature": sig_, "metric_change": mean, "ci95": ci})
    report["hypotheses"], report["knockouts"], report["seeds"] = hyp, kos, seeds
    summary = {k: float(np.mean([p[k] for p in per])) for k in per[0] if not k.startswith("_")}
    report["descriptive"] = summary
    return per


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------
def print_report(report):
    print(f"=== VERIFIED REPORT · chapter {report['chapter']:04d} ===")
    print(f"file: {report['file']}   card_revision: {report['card_revision']}")
    print(f"environment: python {report['environment']['python']}, numpy {report['environment']['numpy']}")
    print(f"seeds: {report['seeds']}   runtime: {report['runtime_s']:.1f} s   params: {report['n_params']}")
    gc = report["gradcheck"]
    print(f"gradcheck: {gc['tensors_checked']} tensor checks, max rel err {gc['max_rel_error']:.2e}, "
          f"passed={gc['passed']}")
    for r in gc["rows"]:
        print("   ", r)
    print("correctness:")
    for c in report["correctness"]:
        print(f"  {c['id']:<5} {'PASS' if c['passed'] else 'FAIL'}  {c['name']}: {c['detail']}")
    mu = report["mutants"]
    print(f"mutation score: {mu['detected']}/{mu['total']}")
    print("hypotheses:")
    for h in report["hypotheses"]:
        print(f"  {h['id']:<8} diff {h['mean_diff']:+.3f}  CI [{h['ci95'][0]:+.3f}, {h['ci95'][1]:+.3f}]  "
              f"mesi {h['mesi']}  n={h['n_seeds']}  -> {h['verdict']}")
    print("knockouts (held-out skill change vs intact Taixu field):")
    for k in report["knockouts"]:
        print(f"  {k['module']:<16} signature={k['signature']!s:<5} {k['metric_change']:+.3f}  "
              f"CI [{k['ci95'][0]:+.3f}, {k['ci95'][1]:+.3f}]")
    d = report["descriptive"]
    print(f"tuned learning rates (grid {MIND_CARD['tuning']['lr_grid']}): {report.get('tuned_lr')}")
    print("descriptive means over seeds (skill vs Lagrangian persistence):")
    print(f"  held-out: taixu {d['taixu_held']:+.3f}  convgru {d['convgru_held']:+.3f}  cheng {d['cheng_held']:+.3f}")
    print(f"  shifted : taixu {d['taixu_shift']:+.3f}  convgru {d['convgru_shift']:+.3f}  cheng {d['cheng_shift']:+.3f}")
    print(f"  open    : taixu {d['taixu_open']:+.3f}  convgru {d['convgru_open']:+.3f}")
    print(f"  latent recovery r(dispersed phase, true vapour), never supervised: {d['latent_r_held']:+.3f}")
    print(f"task types: {report['task_types']}   exit code: {report['exit_code']}")
    print("=== END REPORT ===")


def build_data(rng_data, quick):
    n = 96 if quick else 160
    kids = [np.random.default_rng(s) for s in np.random.SeedSequence(int(rng_data.integers(2**31))).spawn(7)]
    data = {"train": make_world(kids[0], n, "closed"), "val": make_world(kids[1], 48, "closed"),
            "test": make_world(kids[2], 96, "closed"), "shift": make_world(kids[3], 96, "shifted"),
            "open_train": make_world(kids[4], n, "open"), "open_val": make_world(kids[5], 48, "open"),
            "open_test": make_world(kids[6], 96, "open")}
    data["clim"] = climatology(data["train"])
    data["open_clim"] = climatology(data["open_train"])
    return data


def run_mutant(args, data, rng):
    """Audit path: train one registered mutant and apply C1 and C3; exit 1 means the mutant was caught."""
    m = build_model(N_FEAT, N_CELLS, "sequence_regression", rng, kind="taixu")
    m["mutant"] = args.mutant
    L0 = loss_only(m, data["train"])
    fit(m, data["train"], args.mut_steps, rng, lr=LRS["taixu"])
    L1 = loss_only(m, data["train"])
    small = subset(data["train"], np.arange(3))
    _, g = loss_and_grads(m, small)
    w, _ = grad_check(lambda: loss_only(m, small), m["params"], g, rng)
    ok = L1 <= (1 - MIND_CARD["thresholds"]["loss_drop_fraction"]) * L0 and w <= MIND_CARD["thresholds"]["gradcheck_rel_tol"]
    print(f"mutant {args.mutant}: loss {L0:.3f} -> {L1:.3f}, gradcheck {w:.2e}, correctness {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=269)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--card", action="store_true")
    ap.add_argument("--mutant", type=str, default=None, choices=sorted(MUTANTS))
    ap.add_argument("--data", type=str, default=None)
    args = ap.parse_args(argv)
    if args.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    t_start = time.time()
    if args.quick:
        args.seeds = 1
    args.budget = 60 if args.quick else 200
    args.mut_steps = 40 if args.quick else 60
    print(f"{os.path.basename(__file__)}: base seed {args.seed}, seeds {args.seeds}, quick={args.quick}")
    if args.data:
        print(f"--data {args.data}: real-data bridge not implemented for this chapter; skipped (no downloads)")
    rng = np.random.default_rng(args.seed)
    rngs = {k: np.random.default_rng(s) for k, s in
            zip(("data", "c1", "c3", "c4", "c4data", "c4perm"), np.random.SeedSequence(args.seed).spawn(6))}
    data = build_data(rngs["data"], args.quick)
    report = {"schema_version": "1.0", "chapter": 269, "file": os.path.basename(__file__),
              "card_revision": MIND_CARD["card_revision"],
              "environment": {"python": sys.version.split()[0], "numpy": np.__version__}}
    if args.mutant:
        return run_mutant(args, data, rng)
    try:
        report["tuned_lr"] = tune_lrs(data, args.budget, args.seed, args.quick)
        print(f"  tuned learning rates: {report['tuned_lr']}", flush=True)
        rows = run_correctness(args, rngs, data, report)
        per = run_hypotheses(args, data, report, args.seed)
    except FloatingPointError as e:
        print("non-finite values:", e)
        return 4
    runtime = time.time() - t_start
    budget = 20.0 if args.quick else 180.0
    rows.append(("C8", "budget", runtime <= budget, f"{runtime:.1f} s (limit {budget:.0f} s)"))
    report.pop("_c_elapsed", None)
    report["correctness"] = [{"id": i, "name": n, "passed": bool(p), "detail": d} for i, n, p, d in rows]
    report["runtime_s"] = runtime
    report["n_params"] = n_params(per[0]["_model"])
    report["n_params_all"] = {k: n_params(build_model(N_FEAT, N_CELLS, "sequence_regression",
                                                      np.random.default_rng(0), kind=k))
                              for k in ("taixu", "cheng", "convgru")}
    report["task_types"] = TASK_TYPES
    ok = all(c["passed"] for c in report["correctness"])
    report["exit_code"] = 0 if ok else (3 if not rows[-1][2] and all(r[2] for r in rows[:-1]) else 1)
    print_report(report)
    if args.json:
        write_report(args.json, report)
    return report["exit_code"]


if __name__ == "__main__":
    sys.exit(main())

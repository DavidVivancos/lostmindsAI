#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0107 · Vitruvius
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 6, Minds 101-120: https://www.amazon.com/dp/B0HF7G6JJD · Demos: https://artificiology.com/
# END ATTRIBUTION
"""Temperaturae: a commensurate temple front, pre-distorted by a learned correction so that a situated eye sees the canon.

Thesis
    A building is designed true in one module, then deliberately bent by small additions
    and subtractions so that it appears true to an eye standing in a particular place.

Evidence and provenance
    Provenance is belief: every mechanism entry rests on De Architectura (Latin as in
    LacusCurtius and the Latin Library). Goriely and Soyoz are modern scholarship.
    D1  I.2.4; III.1.1: symmetria is the correspondence of a fixed part, the module,
        across all members, and the composition of temples rests on it.
    D2  III.3.2-10: the intercolumniation type fixes column height and spacing in
        diameters; the eustyle front of six columns is divided into 18 modules (III.3.7).
    D3  III.3.11: wide spacing lets the air thin the shafts; corner columns are made a
        fiftieth thicker because the air cuts round them and they look slenderer; what
        the eye deceives must be equalised by calculation.
    D4  III.3.12-13: the upper contraction depends on column height; these temperings
        of thickness are added because of the distance the eye climbs.
    D5  III.4.5: the stylobate gets an addition at the middle; laid level, it will look
        hollowed to the eye.
    D6  III.5.8-9: architraves grow with column height, because the eye's image does not
        easily cut through the density of the air.
    D7  III.5.13: members above the capitals lean forward a twelfth of their height,
        because when we stand facing the front the sight line to the top is longer.
    D8  VI.2.1-2: once the commensurations are set, temper them by additions and
        subtractions for site, use and appearance; appearance differs near at hand, on
        a height, enclosed or open, for sight is deceived.
    D9  Goriely 2025: the illusions these corrections target are absent or too small to
        perceive (an analytic argument, contested).
    D10 Soyoz 2015: optical measures served stylistic ends rather than truth.

Doctrine -> mechanism -> test (IDs as in MIND_CARD)
    D1 D2  M1 canon: module from the brief, every part a quarter-module count      C6.3 H-SIG
    D3-D7  M2 eye: perspective angles plus the posited illusions, fixed            C6.1 C6.2
    D8     M3 temperatura: learned corrections from the order and the station     C1 H-NEC
    D8     M4 triad objective: perceived match, span rule, small corrections      C3
    D7 D8  blind spot: corrections built for one station, seen from others         H-BLIND
    D9     descriptive: the same corrections seen by an eye without the illusions  report

Research question (out-of-distribution generalization)
    When an input is pre-compensated for a fixed, distorting perceiver, does keeping an
    exact commensurate core with a learned residual correction transfer to new scales
    better than an appearance-first generator, and how wrong does a correction tuned for
    one viewpoint become for observers standing elsewhere?

Closest prior art and the delta
    Projector and lens pre-distortion by inverse warping; inverse design through a
    differentiable forward model (Kato et al. 2020); amortized optimization (Amos 2023);
    prescriptive hearing-aid gain formulas. Delta: the true design stays an exact
    multiple of the module and the correction is a separate, legible residual learned
    through a posited-illusion observer, tested against an appearance-first network of
    the same size, Vitruvius's own rules, a per-instance optimum and displaced eyes.

Blind spot
    The corrections are fixed in stone for one station, "when we stand facing the
    front"; from the side, too near, too far or too high they may add the very
    distortion they were meant to remove.

Task (generative process)
    A hexastyle front. The order type (pycnostyle, systyle, eustyle, diastyle) fixes the
    column height (10, 9.5, 9.5, 8.5 diameters) and clear spacing (1.5, 2, 2.25, 3); the
    module is log-uniform on [0.35, 0.9] m (shifted split [1.2, 2.2] m, colossal).
    Canon: architrave half a module, top diameter five sixths of the bottom, level
    stylobate, plumb upper members. Eye 0.85 m above the stylobate (a 1.6 m eye above
    three 0.25 m steps), facing the centre at 1.2 to 2.0 front widths. Displaced
    stations: side (0.6 widths off centre), near (0.6 widths), far (4 widths), high
    (eye at 0.6 column heights). Splits: train 256, held-out 512, shifted 512.

Limits
    A silhouette observer with six features; illusion strengths are modelling choices,
    two of them calibrated to Vitruvius's own corrections; one span rule stands for
    structure; synthetic briefs. A research prototype of one mechanism, not an AGI and
    not a claim to reproduce Vitruvius's mind.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 2,
    "revision_log": [{"revision": 2, "date": "2026-09-15",
                      "reason": ("P9 needs class outputs and cannot run on vector_regression; the probe prediction "
                                 "was changed to P10 before any execution of this file.")}],
    "generation": {"template_version": "codeguidelines 1.0 (15 September 2026), Appendix A",
                   "generator": "Claude (Anthropic)", "generator_version": "claude-opus-5", "date": "2026-09-15"},
    "id": 107, "figure": "Vitruvius", "born": -80, "died": -15, "civilization": "Roman",
    "provenance": "belief",
    "thesis": ("A building is designed true in one module, then deliberately bent by small additions and "
               "subtractions so that it appears true to an eye standing in a particular place."),
    "evidence": [
        {"id": "D1", "basis": "primary", "source": "Vitruvius, De Architectura I.2.4; III.1.1",
         "claim": "Symmetria is the correspondence of a fixed part, the module, across all members, and the "
                  "composition of temples rests on it."},
        {"id": "D2", "basis": "primary", "source": "Vitruvius, De Architectura III.3.2-10 (module: III.3.7)",
         "claim": "The intercolumniation type fixes column height and spacing in diameters; the eustyle front of "
                  "six columns is divided into 18 parts, one of which is the module."},
        {"id": "D3", "basis": "primary", "source": "Vitruvius, De Architectura III.3.11",
         "claim": "Corner columns are made a fiftieth thicker because the air cuts round them and they look "
                  "slenderer; what the eye deceives must be equalised by calculation."},
        {"id": "D4", "basis": "primary", "source": "Vitruvius, De Architectura III.3.12-13",
         "claim": "The upper contraction of a column depends on its height; these temperings are added because of "
                  "the distance the eye climbs."},
        {"id": "D5", "basis": "primary", "source": "Vitruvius, De Architectura III.4.5",
         "claim": "The stylobate receives an addition at the middle; laid level, it will look hollowed to the eye."},
        {"id": "D6", "basis": "primary", "source": "Vitruvius, De Architectura III.5.8-9",
         "claim": "Architrave heights grow with column height because the eye's image does not easily cut through "
                  "the density of the air."},
        {"id": "D7", "basis": "primary", "source": "Vitruvius, De Architectura III.5.13",
         "claim": "Members above the capitals lean forward a twelfth of their height, because when we stand facing "
                  "the front the sight line to the top is longer and makes it appear to lean back."},
        {"id": "D8", "basis": "primary", "source": "Vitruvius, De Architectura VI.2.1-2",
         "claim": "Once the commensurations are set, they are tempered by additions and subtractions for site, use "
                  "and appearance; appearance differs near at hand, on a height, enclosed or open."},
        {"id": "D9", "basis": "scholarship", "source": "Goriely 2025, arXiv:2510.16831",
         "claim": "The illusions that optical corrections are said to cancel are either absent or too small to "
                  "perceive; the argument is analytic and contested."},
        {"id": "D10", "basis": "scholarship", "source": "Soyoz 2015, Nexus Network Journal 17(2), 525-545",
         "claim": "Optical measures were employed as a means to stylistic ends rather than to truth."},
    ],
    "research_question": {
        "category": "out-of-distribution generalization",
        "question": ("When an input is pre-compensated for a fixed, distorting perceiver, does keeping an exact "
                     "commensurate core with a learned residual correction transfer to new scales better than an "
                     "appearance-first generator, and how wrong does a correction tuned for one viewpoint become "
                     "for observers standing elsewhere?")},
    "mechanism": {
        "name": "temperaturae: learned pre-compensation of a commensurate design for a situated eye",
        "family": "amortized inverse perception through a fixed differentiable forward model",
        "signature_modules": ["temperatura"],
        "closest_prior_art": [
            "keystone and lens pre-distortion by inverse geometric warping (projector and display calibration)",
            "inverse design through a differentiable forward model (Kato et al. 2020, Differentiable Rendering: A Survey)",
            "amortized optimization (Amos 2023, Tutorial on Amortized Optimization)",
            "prescriptive hearing-aid gain formulas as fixed pre-compensation (for example NAL-NL2)"],
        "overlap": "Medium",
        "prior_art_queries": [],
        "prior_art_note": "No literature search was run for this card; overlap is rated against the named methods.",
        "contribution_type": "test",
        "delta": ("The true design stays an exact multiple of the module and the correction is a separate residual "
                  "learned through an observer with posited illusions; it is tested against an appearance-first "
                  "network of the same size, Vitruvius's rules, a per-instance optimum and displaced observers."),
        "baselines": {
            "baseline": ("appearance-first network: the same MLP and inputs output the whole design in module units "
                         "(no exact commensurate core), trained on the identical objective"),
            "rival": "none (no historical rival; any contrast with chapter 0130 is conceptual only)"}},
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1 canon", "property_test": "C6.3 (definition check)", "hypothesis": "H-SIG"},
        {"doctrine": "D2", "mechanism": "M1 canon", "property_test": "C6.3 (definition check)", "hypothesis": "H-SIG"},
        {"doctrine": "D3", "mechanism": "M2 eye (corner air)", "property_test": "C6.2", "hypothesis": "H-NEC"},
        {"doctrine": "D4", "mechanism": "M2 eye (air thinning with elevation)", "property_test": "C6.1",
         "hypothesis": "H-NEC"},
        {"doctrine": "D5", "mechanism": "M2 eye (stylobate sag)", "property_test": "C6.1", "hypothesis": "H-NEC"},
        {"doctrine": "D6", "mechanism": "M2 eye (air thinning with elevation)", "property_test": "C6.1",
         "hypothesis": "H-NEC"},
        {"doctrine": "D7", "mechanism": "M2 eye (lean-back); blind-spot stations", "property_test": "C6.1",
         "hypothesis": "H-NEC, H-BLIND"},
        {"doctrine": "D8", "mechanism": "M3 temperatura; M4 triad objective", "property_test": "none",
         "hypothesis": "H-NEC, H-BLIND"},
        {"doctrine": "D9", "mechanism": "illusion-free observer (descriptive)", "property_test": "none",
         "hypothesis": "none (report only)"},
        {"doctrine": "D10", "mechanism": "none", "property_test": "none", "hypothesis": "none"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": ("On colossal fronts outside the training scales, the commensurate core with a "
                                      "learned residual correction is perceived closer to the canon than the "
                                      "appearance-first network of the same size."),
         "metric": "perceived_error", "split": "shifted", "station": "canonical",
         "comparison": "model - baseline", "direction": "less", "mesi": 0.1, "seeds": 5},
        {"id": "H-NEC", "statement": ("Removing the learned correction raises the perceived error of held-out fronts "
                                      "at their canonical stations; no matched learned non-signature module exists, so "
                                      "the comparison is with the full model."),
         "metric": "perceived_error", "split": "heldout", "station": "canonical",
         "comparison": "signature_knockout - full_model", "knockouts": ["temperatura:zero"],
         "direction": "greater", "mesi": 0.5, "seeds": 5},
        {"id": "H-BLIND", "statement": ("Seen from displaced stations, a front corrected for its canonical station is "
                                        "perceived further from the canon than the same front left uncorrected."),
         "condition": "displaced stations (side, near, far, high), pooled",
         "grounding": ("III.5.13 fixes its correction for an observer standing facing the front, while VI.2.2 admits "
                       "that appearance differs near at hand, on a height, enclosed or open."),
         "metric": "perceived_error", "split": "heldout", "comparison": "model - uncorrected core",
         "direction": "greater", "mesi": 0.1, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.5, "margin_over_trivial": 0.3, "shuffled_ratio_min": 0.9,
                   "gradcheck_rel_error": 1e-5, "gradcheck_floor": 1e-3, "invariance_tol": 1e-9,
                   "negative_control_min_violation": 1e-6},
    "metrics": {"perceived_error": ("RMS over six perceived-proportion features of the deviation from the canon, "
                                    "each divided by its declared perceptibility scale"),
                "trivial_baseline": "the uncorrected commensurate core",
                "shuffled_band": ("one-sided: a network trained on inputs shuffled across briefs must not beat the "
                                  "best constant correction by more than a tenth")},
    "training": {"optimizer": "Adam", "lr_grid": [0.02], "updates": {"full": 1500, "quick": 400}, "clip_norm": 5.0,
                 "batch": "all 256 training briefs", "model_selection": "none: final parameters",
                 "applies_to": "temperatura, appearance-first baseline and constant correction",
                 "oracle_updates": {"full": 300, "quick": 100}},
    "task": {"orders": {"pycnostyle": [10.0, 1.5], "systyle": [9.5, 2.0], "eustyle": [9.5, 2.25], "diastyle": [8.5, 3.0]},
             "columns": 6, "architrave_modules": 0.5, "contracture": "5/6", "eye_above_stylobate_m": 0.85,
             "distance_in_front_widths": [1.2, 2.0], "module_m": {"train": [0.35, 0.9], "shifted": [1.2, 2.2]},
             "splits": {"train": 256, "heldout": 512, "shifted": 512},
             "displaced": {"side": "0.6 widths off centre", "near": "0.6 widths", "far": "4 widths",
                           "high": "eye at 0.6 column heights"},
             "observer": {"corner_air": "1/51", "sag_rad": 0.001, "lean_back": 0.3, "air_thinning": 0.15},
             "perceptibility_scales": [0.05, 0.02, 0.02, 0.05, 0.001, 0.02],
             "objective": {"firmitas": "architrave at least a fifth of the clear span, weight 10",
                           "venustas": "0.5 times the mean squared correction"}},
    "probe_predictions": [{"probe": "P10", "expected": "equal to baseline"}],
    "probe_support": "vector_regression through the temperatura network used as a plain regressor on X, Y batches",
    "dialectic_links": [],
    "corpus_neighbors": [
        {"chapter": 84, "similarity": None, "difference": ("0084 seeks descriptions invariant under the observer's "
                                                           "motion; here the observer is fixed and the object is "
                                                           "changed for it.")},
        {"chapter": 130, "similarity": None, "difference": ("conceptual contrast only: 0130 moves the observer's frame "
                                                            "until the law looks uniform; here the building is bent "
                                                            "for a fixed eye.")},
        {"chapter": 52, "similarity": None, "difference": ("0052 works with ordinal invariance; here metric "
                                                           "proportions must be perceived exactly.")},
        {"chapter": 105, "similarity": None, "difference": ("0105 breaks symmetry with learned noise; here nothing is "
                                                            "stochastic and a deterministic correction faces a fixed "
                                                            "perceiver.")},
        {"chapter": 118, "similarity": None, "difference": ("0118 stores an open-loop program for a moving body; here "
                                                            "a static design is pre-compensated for a static eye.")},
    ],
    "similarity_note": "Nearest-neighbour similarity not computed into the card; the audit reports it for the files at hand.",
    "barometer": {
        "cognitive_processing": ["transfer of corrections to colossal scales (shifted split)"],
        "embodied_cognition": ["appearance judged from a situated, displaced eye"],
        "world_modeling": ["forward model of perspective and of posited illusions"],
        "consciousness": [], "language_understanding": [], "emotional_intelligence": [], "creativity": [],
        "autonomy": []},
    "task_types": ["vector_regression"],
    "applications": [
        {"use": ("pre-distorting projected or displayed imagery so that a viewer at a known position sees undistorted "
                 "geometry, with the correction kept separate from the source content and audited off-axis"),
         "sector": "projection, augmented reality and display engineering",
         "dataset": "TID2013 image quality database", "readiness": "low"},
        {"use": ("sizing lettering on signs so that it stays legible at the intended viewing distance, and checking "
                 "it from displaced positions"),
         "sector": "wayfinding and transport signage", "dataset": "ICDAR 2015 Incidental Scene Text",
         "readiness": "low"},
        {"use": ("hearing-aid pre-compensation learned per listener profile and audited for listeners who differ "
                 "from the fitting assumptions"),
         "sector": "audiology and hearing health", "dataset": "Clarity Enhancement Challenge (CEC2) data",
         "readiness": "low"},
    ],
    "safety_notes": ("The machines of Book X and every war machine, including the artillery example among the modules "
                     "of I.2.4, are excluded from the evidence, the mechanism and the applications, which are civilian. "
                     "The file does not claim to replicate Vitruvius's mind and puts no generated words in his mouth."),
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

ORDERS = ("pycnostyle", "systyle", "eustyle", "diastyle")
HEIGHT_DIAMETERS = np.array([10.0, 9.5, 9.5, 8.5])
CLEAR_SPACING = np.array([1.5, 2.0, 2.25, 3.0])
ARCHITRAVE_MODULES, CONTRACTURE, ROMAN_FOOT = 0.5, 5.0 / 6.0, 0.296
EYE_ABOVE_STYLOBATE, DISTANCE_RATIO = 0.85, (1.2, 2.0)
MODULE_RANGE = {"train": (0.35, 0.9), "heldout": (0.35, 0.9), "shifted": (1.2, 2.2)}
SPLIT_SIZES = {"train": 256, "heldout": 512, "shifted": 512}
DISPLACED = ("side", "near", "far", "high")
CORNER_AIR, SAG, LEAN_BACK, AIR_THINNING = 1.0 / 51.0, 0.001, 0.3, 0.15
FEATURE_SCALE = np.array([0.05, 0.02, 0.02, 0.05, 0.001, 0.02])
# corrections: corner diameter, inner diameter, top ratio, column height, architrave, stylobate rise (modules), lean (rad)
N_CORR, N_INPUT, HIDDEN = 7, 6, 24
BOUND = np.array([0.5, 0.5, 0.15, 3.0, 0.4, 0.5, 0.3])
TEMPERATURA_SCALE = np.full(N_CORR, 0.1)
APPEARANCE_SCALE = np.array([0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1])
FIRMITAS_WEIGHT, FIRMITAS_SHARPNESS, SPAN_PER_ARCHITRAVE, VENUSTAS_WEIGHT = 10.0, 20.0, 5.0, 0.5
LR, CLIP_NORM = 0.02, 5.0
STEPS = {"full": 1500, "quick": 400}
ORACLE_STEPS = {"full": 300, "quick": 100}
TIME_BUDGET = {"full": 180.0, "quick": 20.0}
TASK_TYPES = ["vector_regression"]
ACTIVE_MUTANT = None
np.seterr(over="raise", invalid="raise", divide="raise", under="ignore")


# BEGIN STANDARD UTILITIES v1.0
def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def logsumexp(z, axis=-1):
    m = z.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(z - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def softplus(z):
    return np.logaddexp(0.0, z)


def sigmoid(z):
    return np.exp(-np.logaddexp(0.0, -z))


def adam_init(params):
    return {"t": 0, "m": {k: np.zeros_like(v) for k, v in params.items()},
            "v": {k: np.zeros_like(v) for k, v in params.items()}}


def adam_step(params, grads, state, lr, b1=0.9, b2=0.999, eps=1e-8):
    state["t"] += 1
    for k in params:
        state["m"][k] = b1 * state["m"][k] + (1.0 - b1) * grads[k]
        state["v"][k] = b2 * state["v"][k] + (1.0 - b2) * grads[k] ** 2
        m_hat = state["m"][k] / (1.0 - b1 ** state["t"])
        v_hat = state["v"][k] / (1.0 - b2 ** state["t"])
        params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)


def clip_global(grads, max_norm):
    norm = math.sqrt(sum(float((g * g).sum()) for g in grads.values()))
    scale = min(1.0, max_norm / (norm + 1e-12))
    return {k: g * scale for k, g in grads.items()}, norm


def finite_difference_check(params, grads, loss_fn, rng, eps=1e-6, n_entries=20, floor=1e-3):
    """Central differences on n random entries per tensor plus its largest-gradient entry.
    Relative error uses max(|analytic|, |numeric|, floor) as denominator."""
    worst = {}
    for name, arr in params.items():
        flat, g = arr.reshape(-1), grads[name].reshape(-1)
        if flat.size <= n_entries + 1:
            idx = np.arange(flat.size)
        else:
            idx = np.unique(np.append(rng.choice(flat.size, n_entries, replace=False), np.argmax(np.abs(g))))
        err = 0.0
        for i in idx:
            keep = flat[i]
            flat[i] = keep + eps
            up = loss_fn()
            flat[i] = keep - eps
            down = loss_fn()
            flat[i] = keep
            num = (up - down) / (2.0 * eps)
            err = max(err, abs(g[i] - num) / max(abs(g[i]), abs(num), floor))
        worst[name] = err
    return worst


def paired_bootstrap(diffs, rng, n_boot=2000, level=0.95):
    d = np.asarray(diffs, dtype=float)
    means = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    tail = 50.0 * (1.0 - level)
    return float(d.mean()), [float(np.percentile(means, tail)), float(np.percentile(means, 100.0 - tail))]


def verdict(mean, ci, mesi, direction):
    s = 1.0 if direction == "greater" else -1.0
    lo, hi = sorted((s * ci[0], s * ci[1]))
    if lo > 0.0 and s * mean >= mesi:
        return "supported"
    if hi < 0.0:
        return "contradicted"
    return "inconclusive"


def write_report(lines, payload, json_path):
    print("\n".join(lines))
    if json_path:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
# END STANDARD UTILITIES


# ---------------------------------------------------------------- the world: canon, stations and the situated eye
class Tang:
    """A value carrying its derivatives along the seven corrections (forward mode through the fixed eye)."""
    __slots__ = ("v", "t")

    def __init__(self, v, t=None):
        self.v = np.asarray(v, dtype=float)
        self.t = np.zeros(self.v.shape + (N_CORR,)) if t is None else t

    @staticmethod
    def of(x):
        return x if isinstance(x, Tang) else Tang(x)

    def __add__(self, other):
        other = Tang.of(other)
        return Tang(self.v + other.v, self.t + other.t)

    __radd__ = __add__

    def __sub__(self, other):
        other = Tang.of(other)
        return Tang(self.v - other.v, self.t - other.t)

    def __rsub__(self, other):
        return Tang.of(other) - self

    def __mul__(self, other):
        other = Tang.of(other)
        return Tang(self.v * other.v, self.t * other.v[..., None] + other.t * self.v[..., None])

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = Tang.of(other)
        q = self.v / other.v
        return Tang(q, (self.t - other.t * q[..., None]) / other.v[..., None])

    def along(self, value, slope):
        return Tang(value, self.t * slope[..., None])

    def atan(self):
        slope = 1.0 / (1.0 + self.v * self.v)
        return self.along(np.arctan(self.v), 0.0 * slope if ACTIVE_MUTANT == "tangent_dropped" else slope)

    def sqrt(self):
        root = np.sqrt(self.v)
        return self.along(root, 0.5 / root)

    def log(self):
        return self.along(np.log(self.v), 1.0 / self.v)

    def sin(self):
        return self.along(np.sin(self.v), np.cos(self.v))


def canon_counts(tau):
    """The true design in module units (and the plain top ratio): symmetria before any tempering."""
    n = len(tau)
    return np.column_stack([np.ones(n), np.ones(n), np.full(n, CONTRACTURE), HEIGHT_DIAMETERS[tau],
                            np.full(n, ARCHITRAVE_MODULES), np.zeros(n), np.zeros(n)])


def make_brief(tau, m, rho):
    """Fronts of order tau and module m (metres), each seen by an eye at rho front widths facing its centre."""
    width = (5.0 * (1.0 + CLEAR_SPACING[tau]) + 1.0) * m
    eta = EYE_ABOVE_STYLOBATE / (HEIGHT_DIAMETERS[tau] * m)
    z = np.column_stack([np.eye(len(ORDERS))[tau], (rho - 1.6) / 0.4, (eta - 0.19) / 0.1])
    view = {"D": rho * width, "xe": np.zeros_like(m), "ye": np.full_like(m, EYE_ABOVE_STYLOBATE)}
    return {"tau": tau, "m": m, "rho": rho, "width": width, "z": z, "view": view}


def displaced(brief, kind, rng):
    """The same fronts, keeping the corrections of their canonical station, seen from somewhere else."""
    view, width = dict(brief["view"]), brief["width"]
    if kind == "side":
        view["xe"] = rng.choice([-1.0, 1.0], width.size) * 0.6 * width
    elif kind == "near":
        view["D"] = 0.6 * width
    elif kind == "far":
        view["D"] = 4.0 * width
    else:
        view["ye"] = 0.6 * HEIGHT_DIAMETERS[brief["tau"]] * brief["m"]
    return dict(brief, view=view)


def make_data(rng):
    data = {}
    for split, size in SPLIT_SIZES.items():
        low, high = np.log(MODULE_RANGE[split])
        data[split] = make_brief(rng.integers(0, len(ORDERS), size), np.exp(rng.uniform(low, high, size)),
                                 rng.uniform(DISTANCE_RATIO[0], DISTANCE_RATIO[1], size))
    data["displaced"] = {kind: displaced(data["heldout"], kind, rng) for kind in DISPLACED}
    return data


def perceive(delta, brief, view=None, illusions=True, grain=0.0, pairs=((0, 5), (2, 3))):
    """The fixed eye: angles from the station plus the posited illusions (corner air, air thinning with elevation,
    stylobate sag, lean-back). Returns six deviations from the canon in perceptibility units, with their tangents."""
    view = brief["view"] if view is None else view
    n, tau, m = delta.shape[0], brief["tau"], brief["m"]
    unit = np.eye(N_CORR)
    c = [Tang(delta[:, k], np.tile(unit[k], (n, 1))) for k in range(N_CORR)]
    columns_high, pitch = HEIGHT_DIAMETERS[tau], (1.0 + CLEAR_SPACING[tau]) * m
    half, D, xe, ye = 2.5 * pitch + 0.5 * m, view["D"], view["xe"], view["ye"]
    air = AIR_THINNING if illusions else 0.0
    rise, height = c[5] * m, (c[3] + columns_high) * m
    top_ratio, architrave = c[2] + CONTRACTURE, (c[4] + ARCHITRAVE_MODULES) * m

    def elevation(x, y):
        return ((Tang.of(y) - ye) / Tang((x - xe) ** 2 + D * D).sqrt()).atan()

    def column(i, diameter, thin):
        x = (i - 2.5) * pitch
        base = rise * (1.0 - (x / half) ** 2)
        top = base + height
        e_base, e_top = elevation(x, base), elevation(x, top)
        ground = (x - xe) ** 2 + D * D

        def seen_width(d, y, e):
            reach = ((y - ye) * (y - ye) + ground).sqrt()
            return ((d * 0.5 + 0.5 * grain) / reach).atan() * (2.0 * thin) * (1.0 - air * e)
        extent = (e_top - e_base) * (1.0 - air * 0.5 * (e_base + e_top))
        return extent, seen_width(diameter, base, e_base), seen_width(diameter * top_ratio, top, e_top)

    def mean(items):
        total = items[0]
        for item in items[1:]:
            total = total + item
        return total * (1.0 / len(items))

    corners = [column(i, (c[0] + 1.0) * m, 1.0 - CORNER_AIR if illusions else 1.0) for i in pairs[0]]
    inners = [column(i, (c[1] + 1.0) * m, 1.0) for i in pairs[1]]
    extent_in, base_in, top_in = (mean([col[j] for col in inners]) for j in range(3))
    e_low = elevation(0.0, rise + height)
    e_high = elevation(0.0, rise + height + architrave)
    e_mid = (e_low + e_high) * 0.5
    sag_seen = elevation(0.0, rise) - (elevation(-half, 0.0) + elevation(half, 0.0)) * 0.5
    features = [(extent_in / base_in).log() - np.log(columns_high),
                (mean([col[1] for col in corners]) / base_in).log(),
                (top_in / base_in).log() - math.log(CONTRACTURE),
                ((e_high - e_low) * (1.0 - air * e_mid) / extent_in).log() - np.log(ARCHITRAVE_MODULES / columns_high),
                sag_seen - (SAG if illusions else 0.0),
                c[6] - e_mid.sin() * (LEAN_BACK if illusions else 0.0)]
    values = np.stack([f.v for f in features], axis=1) / FEATURE_SCALE
    return values, np.stack([f.t for f in features], axis=1) / FEATURE_SCALE[None, :, None]


def objective(delta, brief, view=None, illusions=True):
    """Triad per front: eurythmia (perceived match), firmitas (architrave at least a fifth of the clear span) and
    venustas (small corrections). Returns the loss, its gradient in the corrections, and the deviations."""
    values, tangents = perceive(delta, brief, view, illusions)
    gap = CLEAR_SPACING[brief["tau"]] / SPAN_PER_ARCHITRAVE - (ARCHITRAVE_MODULES + delta[:, 4])
    loss = ((values ** 2).mean(axis=1) + FIRMITAS_WEIGHT * softplus(FIRMITAS_SHARPNESS * gap) / FIRMITAS_SHARPNESS
            + VENUSTAS_WEIGHT * (delta ** 2).mean(axis=1))
    grad = (2.0 / values.shape[1]) * np.einsum("nf,nfk->nk", values, tangents) + (2.0 * VENUSTAS_WEIGHT / N_CORR) * delta
    grad[:, 4] -= FIRMITAS_WEIGHT * sigmoid(FIRMITAS_SHARPNESS * gap)
    return loss, grad, values


def perceived_error(delta, brief, view=None, illusions=True):
    values = perceive(delta, brief, view, illusions)[0]
    return np.sqrt((values ** 2).mean(axis=1))


def handbook(brief):
    """Vitruvius's own temperaturae: corner +1/50 (III.3.11), top ratio by height (III.3.12), architrave by height
    (III.5.8) and forward lean of a twelfth (III.5.13); no amount survives for the stylobate rise (III.4.5)."""
    feet = HEIGHT_DIAMETERS[brief["tau"]] * brief["m"] / ROMAN_FOOT
    band = np.where(feet > 50.0, 4.0 + np.ceil((feet - 50.0) / 10.0), np.searchsorted([15.0, 20.0, 30.0, 40.0, 50.0], feet))
    beyond = np.ceil(np.maximum(feet - 15.0, 0.0) / 5.0)
    architrave = np.where(feet <= 15.0, ARCHITRAVE_MODULES,
                          HEIGHT_DIAMETERS[brief["tau"]] / np.maximum(13.5 - 0.5 * beyond, 6.0))
    delta = np.zeros((feet.size, N_CORR))
    delta[:, 0], delta[:, 6] = 1.0 / 50.0, math.atan(1.0 / 12.0)
    delta[:, 2] = (5.0 + 0.5 * band) / (6.0 + 0.5 * band) - CONTRACTURE
    delta[:, 4] = architrave - ARCHITRAVE_MODULES
    return delta


# ---------------------------------------------------------------- model
def build_model(in_dim, out_dim, task_type, rng, **cfg):
    """kind='temperatura' (default): residual corrections on the exact core. kind='appearance': the size-matched
    baseline stating the whole design. kind='constant': one correction shared by every front."""
    if task_type not in TASK_TYPES:
        raise ValueError("chapter 0107 supports vector_regression only")
    kind = cfg.get("kind", "temperatura")
    if kind == "constant":
        params = {"shared": np.zeros(out_dim)}
    else:
        params = {"W1": rng.normal(0.0, in_dim ** -0.5, (HIDDEN, in_dim)), "b1": np.zeros(HIDDEN),
                  "W2": rng.normal(0.0, 0.1 * HIDDEN ** -0.5, (out_dim, HIDDEN)), "b2": np.zeros(out_dim)}
    return {"kind": kind, "params": params, "ko": {}, "lr": cfg.get("lr", LR),
            "statio_mean": np.zeros(2), "mean_delta": np.zeros(out_dim)}


def network(params, z):
    hidden = np.tanh(z @ params["W1"].T + params["b1"])
    return hidden, hidden @ params["W2"].T + params["b2"]


def corrections(model, brief):
    """Pre-distortions in module units (lean in radians) for each front; also returns the inputs, the hidden layer
    and d(correction)/d(network output) for the backward pass."""
    params, ko, kind, n = model["params"], model["ko"], model["kind"], brief["tau"].size
    z = brief["z"].copy()
    if ko.get("statio") == "mean":
        z[:, 4:] = model["statio_mean"]
    hidden = None
    if kind == "constant":
        raw, scale = np.tile(params["shared"], (n, 1)), np.ones(N_CORR)
    else:
        hidden, out = network(params, z)
        scale = TEMPERATURA_SCALE if kind == "temperatura" else APPEARANCE_SCALE
        raw = out * scale
    squash = np.tanh(raw / BOUND)
    delta, slope = BOUND * squash, scale * (1.0 - squash ** 2)
    if kind == "appearance":
        delta = delta + canon_counts(np.arange(len(ORDERS))).mean(axis=0) - canon_counts(brief["tau"])
    if ko.get("temperatura") in ("zero", "mean"):
        delta = np.zeros((n, N_CORR)) if ko["temperatura"] == "zero" else np.tile(model["mean_delta"], (n, 1))
        slope = np.zeros(N_CORR)
    return z, hidden, delta, slope


def loss_and_grads(model, batch):
    """The triad objective through the fixed eye (tangents give d loss / d correction), then a hand-derived backward
    pass through the network. A batch holding X and Y is plain regression, for the probe battery and --data."""
    params = model["params"]
    if "Y" in batch:
        hidden, out = network(params, batch["X"])
        residual = out - batch["Y"]
        loss, d_out, z = float((residual ** 2).mean()), 2.0 * residual / residual.size, batch["X"]
    else:
        z, hidden, delta, slope = corrections(model, batch)
        per, grad, _ = objective(delta, batch)
        loss, d_out = float(per.mean()), grad * slope / per.size
        if model["kind"] == "constant":
            return loss, {"shared": d_out.sum(axis=0)}
    grads = {"W2": d_out.T @ hidden, "b2": d_out.sum(axis=0)}
    d_pre = (d_out @ params["W2"]) * (1.0 - hidden ** 2)
    grads["W1"], grads["b1"] = d_pre.T @ z, d_pre.sum(axis=0)
    if ACTIVE_MUTANT == "zero_grad_output_layer":
        grads["W2"] = np.zeros_like(grads["W2"])
    return loss, grads


def predict(model, X):
    """Raw network outputs for inputs X (a constant model returns its shared correction for every row)."""
    X = np.asarray(X, dtype=float)
    if model["kind"] == "constant":
        return np.tile(model["params"]["shared"], (X.shape[0], 1))
    return network(model["params"], X)[1]


def hidden_states(model, X):
    hidden, out = network(model["params"], np.asarray(X, dtype=float))
    return {"hidden": hidden, "output": out}


def modules(model):
    roles = {"temperatura": "residual corrections added to the exact commensurate core",
             "appearance": "whole design in module units, with no exact core",
             "constant": "one correction shared by every front"}
    table = {"canon": ([], "commensurate core: module times the order's quarter-module counts", False),
             "statio": ([], "station inputs: distance in front widths and eye height in column heights", False),
             model["kind"]: (list(model["params"]), roles[model["kind"]], model["kind"] == "temperatura")}
    return {name: {"params": p, "role": role, "signature": sig} for name, (p, role, sig) in table.items()}


KNOCKOUT_MODES = {"temperatura": ("zero", "mean"), "statio": ("mean",)}


def knockout(model, name, mode):
    if name not in modules(model) or mode not in KNOCKOUT_MODES.get(name, ()):
        raise ValueError(f"no knockout {name}:{mode} for a {model['kind']} model")
    return dict(model, params={k: v.copy() for k, v in model["params"].items()}, ko=dict(model["ko"], **{name: mode}))


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


MUTANTS = {
    "sign_flipped_update": ("every update climbs the objective instead of descending it", "C3"),
    "zero_learning_rate": ("the optimizer never moves the parameters", "C3"),
    "zero_grad_output_layer": ("the output-layer gradient is replaced by zeros", "C1"),
    "tangent_dropped": ("the eye's arctangent passes no derivative to the corrections", "C1"),
}


def fit(model, data, budget, rng):
    """Adam on all training fronts at once. Nothing is sampled during training, so rng is unused (interface only)."""
    state, history = adam_init(model["params"]), []
    lr = 0.0 if ACTIVE_MUTANT == "zero_learning_rate" else model["lr"]
    for step in range(1, budget + 1):
        loss, grads = loss_and_grads(model, data["train"])
        if not math.isfinite(loss) or not all(np.isfinite(g).all() for g in grads.values()):
            raise FloatingPointError(f"non-finite loss or gradient at update {step}")
        grads = clip_global(grads, CLIP_NORM)[0]
        if ACTIVE_MUTANT == "sign_flipped_update":
            grads = {k: -g for k, g in grads.items()}
        adam_step(model["params"], grads, state, lr)
        history.append(loss)
    if "Y" not in data["train"]:
        model["mean_delta"] = corrections(model, data["train"])[2].mean(axis=0)
        model["statio_mean"] = data["train"]["z"][:, 4:].mean(axis=0)
    model["history"] = history
    return history


def per_front_optimum(brief, steps, lr=0.01):
    """Adam on each front's own objective at its own station: a reference for what amortization gives up."""
    params = {"delta": np.zeros((brief["tau"].size, N_CORR))}
    state = adam_init(params)
    for _ in range(steps):
        adam_step(params, {"delta": objective(params["delta"], brief)[1]}, state, lr)
        np.clip(params["delta"], -BOUND, BOUND, out=params["delta"])
    return params["delta"]


def data_bridge(path, seed, budget):
    """Optional real data: a numeric CSV with a header row whose last column is the target (plain regression)."""
    if not os.path.exists(path):
        return f"skipped ({path} not found)"
    table = np.genfromtxt(path, delimiter=",", skip_header=1)
    X, Y = table[:, :-1], table[:, -1:]
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    order = np.random.default_rng(seed).permutation(len(Y))
    train, held = order[: int(0.8 * len(Y))], order[int(0.8 * len(Y)):]
    model = build_model(X.shape[1], 1, TASK_TYPES[0], np.random.default_rng(seed + 1))
    fit(model, {"train": {"X": X[train], "Y": Y[train]}}, budget, None)
    rmse = float(np.sqrt(((predict(model, X[held]) - Y[held]) ** 2).mean()))
    spread = float(np.sqrt(((Y[held] - Y[train].mean()) ** 2).mean()))
    return f"{path}: held-out RMSE {rmse:.4f} vs mean predictor {spread:.4f}"


# ---------------------------------------------------------------- tests: correctness
CHECKS = []


def check(code, name):
    """Register a correctness check; each receives the shared site and returns (passed, detail)."""
    def register(fn):
        CHECKS.append((code, name, fn))
        return fn
    return register


class Mutation:
    """with Mutation(name): runs the enclosed block with a registered mutant (or none) switched on."""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        global ACTIVE_MUTANT
        self.saved, ACTIVE_MUTANT = ACTIVE_MUTANT, self.name
        return self

    def __exit__(self, *exc):
        global ACTIVE_MUTANT
        ACTIVE_MUTANT = self.saved
        return False


ENGINES = ("temperatura", "appearance", "constant")


def fresh(kind, seed):
    """An untrained model drawn from exactly the stream run_seed uses for that kind."""
    streams = np.random.SeedSequence(seed).spawn(1 + len(ENGINES))
    return build_model(N_INPUT, N_CORR, TASK_TYPES[0], np.random.default_rng(streams[1 + ENGINES.index(kind)]), kind=kind)


def worst_gradient(models, data, rng, entries):
    batch = {k: ({q: w[:64] for q, w in v.items()} if isinstance(v, dict) else v[:64]) for k, v in data["train"].items()}
    floor = MIND_CARD["thresholds"]["gradcheck_floor"]
    worst, counted = 0.0, 0
    for model in models:
        table = finite_difference_check(model["params"], loss_and_grads(model, batch)[1],
                                        lambda model=model: loss_and_grads(model, batch)[0], rng, n_entries=entries, floor=floor)
        worst, counted = max(worst, max(table.values())), counted + len(table)
    return worst, counted


def error_of(model, brief):
    return float(perceived_error(corrections(model, brief)[2], brief).mean())


def learns(model, data):
    losses, limits = model["history"], MIND_CARD["thresholds"]
    drop = 1.0 - float(np.mean(losses[-20:])) / losses[0]
    held = data["heldout"]
    bare = float(perceived_error(np.zeros((held["tau"].size, N_CORR)), held).mean())
    error = error_of(model, held)
    return drop >= limits["loss_drop_fraction"] and error <= (1.0 - limits["margin_over_trivial"]) * bare, drop, error, bare


@check("C1", "gradient_check")
def _gradients(site):
    total = sum(len(site["engines"][k]["params"]) for k in ENGINES)
    start = worst_gradient([fresh(k, site["base_seed"]) for k in ENGINES], site["data"],
                           np.random.default_rng(site["base_seed"] + 3), 20)
    end = worst_gradient([site["engines"][k] for k in ENGINES], site["data"], np.random.default_rng(site["base_seed"] + 4), 20)
    worst = max(start[0], end[0])
    passed = worst <= MIND_CARD["thresholds"]["gradcheck_rel_error"] and start[1] == end[1] == total
    site["gradcheck"] = {"tensors_checked": end[1], "tensors_total": total, "max_rel_error": worst,
                         "checked_at": ["init", "after_training_steps"], "passed": bool(passed)}
    return passed, (f"max rel error {worst:.2e}; every tensor of the temperatura, appearance-first and constant models "
                    f"at init and after {site['budget']} updates")


@check("C2", "determinism_finiteness")
def _determinism(site):
    twins = [fresh("temperatura", site["base_seed"]) for _ in range(2)]
    for twin in twins:
        fit(twin, site["data"], 25, None)
    outputs = [predict(twin, site["data"]["heldout"]["z"]) for twin in twins]
    same = twins[0]["history"] == twins[1]["history"] and np.array_equal(outputs[0], outputs[1])
    finite = bool(np.isfinite(outputs[0]).all()) and all(np.isfinite(p).all() for p in twins[0]["params"].values())
    return same and finite, f"identical losses and outputs: {same}; finite: {finite}"


@check("C3", "learning")
def _learning(site):
    passed, drop, error, bare = learns(site["engines"]["temperatura"], site["data"])
    return passed, (f"loss drop {drop:.3f} (min 0.5); held-out perceived error {error:.3f} vs uncorrected core "
                    f"{bare:.3f} (max ratio 0.70)")


@check("C4", "shuffled_input_control")
def _shuffled(site):
    data = site["data"]
    order = np.random.default_rng(site["base_seed"] + 11).permutation(data["train"]["tau"].size)
    scrambled = dict(data, train=dict(data["train"], z=data["train"]["z"][order]))
    model = fresh("temperatura", site["base_seed"])
    fit(model, scrambled, site["budget"], None)
    shuffled, constant = error_of(model, data["heldout"]), error_of(site["engines"]["constant"], data["heldout"])
    floor = MIND_CARD["thresholds"]["shuffled_ratio_min"]
    return shuffled >= floor * constant, (f"held-out perceived error after training on shuffled inputs {shuffled:.3f} "
                                          f"vs best constant correction {constant:.3f} (min ratio {floor})")


def replay(site):
    try:
        model = fresh("temperatura", site["base_seed"])
        worst = worst_gradient([model], site["data"], np.random.default_rng(site["base_seed"]), 4)[0]
        fit(model, site["data"], site["budget"], None)
        return worst <= MIND_CARD["thresholds"]["gradcheck_rel_error"] and learns(model, site["data"])[0]
    except FloatingPointError:
        return False


@check("C5", "mutant_detection")
def _mutants(site):
    control = replay(site)
    for name in MUTANTS:
        with Mutation(name):
            site["caught"][name] = not replay(site)
    found = sum(site["caught"].values())
    return control and found == len(MUTANTS), f"unmutated base-seed run passes C1 and C3: {control}; mutants caught {found}/{len(MUTANTS)}"


@check("C6.1", "similarity_invariance")
def _similarity(site):
    """Enlarging a front and its station together leaves every perceived proportion unchanged (the network inputs are
    dimensionless by construction); an absolute grain added to every diameter must break this."""
    rng, limits = np.random.default_rng(site["base_seed"] + 5), MIND_CARD["thresholds"]
    gaps = {0.0: 0.0, 0.02: 0.0}
    for split in ("train", "shifted"):
        brief = site["data"][split]
        n, width = brief["tau"].size, brief["width"]
        delta = corrections(site["engines"]["temperatura"], brief)[2]
        view = {"D": rng.uniform(0.5, 4.0, n) * width, "xe": rng.uniform(-0.8, 0.8, n) * width,
                "ye": rng.uniform(0.2, 2.0, n) * EYE_ABOVE_STYLOBATE}
        for factor, grain in itertools.product((0.37, 2.9), gaps):
            big = dict(brief, m=brief["m"] * factor, width=width * factor)
            moved = perceive(delta, big, {k: v * factor for k, v in view.items()}, grain=grain)[0]
            gaps[grain] = max(gaps[grain], float(np.abs(moved - perceive(delta, brief, view, grain=grain)[0]).max()))
    passed = gaps[0.0] <= limits["invariance_tol"] and gaps[0.02] >= limits["negative_control_min_violation"]
    return passed, f"largest change under enlargement {gaps[0.0]:.1e}; absolute-grain negative control {gaps[0.02]:.1e}"


@check("C6.2", "mirror_equivariance")
def _mirror(site):
    """A symmetric front seen from +x and from -x is perceived identically; reading only the left columns must not be."""
    rng, limits = np.random.default_rng(site["base_seed"] + 6), MIND_CARD["thresholds"]
    brief = site["data"]["heldout"]
    delta = corrections(site["engines"]["temperatura"], brief)[2]
    offset = rng.uniform(0.1, 0.9, brief["tau"].size) * brief["width"]
    gaps = []
    for pairs in (((0, 5), (2, 3)), ((0,), (2,))):
        seen = [perceive(delta, brief, dict(brief["view"], xe=sign * offset), pairs=pairs)[0] for sign in (1.0, -1.0)]
        gaps.append(float(np.abs(seen[0] - seen[1]).max()))
    passed = gaps[0] <= limits["invariance_tol"] and gaps[1] >= limits["negative_control_min_violation"]
    return passed, f"largest mirror gap {gaps[0]:.1e}; left-columns-only negative control {gaps[1]:.1e}"


@check("C6.3", "commensurate_core_definition")
def _core(site):
    """Definition check (true by construction, not evidence): the untempered design is whole quarter-modules."""
    counts = np.concatenate([canon_counts(np.arange(len(ORDERS)))[:, [0, 1, 3, 4]].ravel(), CLEAR_SPACING])
    on_grid = bool(np.all(np.abs(4.0 * counts - np.round(4.0 * counts)) < 1e-12))
    return on_grid, f"core diameters, heights, architraves and spacings are whole quarter-modules: {on_grid}"


@check("C7", "split_integrity")
def _splits(site):
    data = site["data"]
    keys = {s: {hashlib.sha256(np.array([data[s]["tau"][i], data[s]["m"][i], data[s]["rho"][i]]).tobytes()).hexdigest()
                for i in range(data[s]["tau"].size)} for s in SPLIT_SIZES}
    disjoint = all(not (keys[a] & keys[b]) for a, b in itertools.combinations(SPLIT_SIZES, 2))
    colossal = bool(data["shifted"]["m"].min() > max(data["train"]["m"].max(), data["heldout"]["m"].max()))
    return disjoint and colossal, f"pairwise disjoint: {disjoint}; every shifted module above every training module: {colossal}"


# ---------------------------------------------------------------- hypotheses, report, CLI
def run_seed(seed, budget, oracle_steps):
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(1 + len(ENGINES))]
    data = make_data(streams[0])
    models = {kind: build_model(N_INPUT, N_CORR, TASK_TYPES[0], streams[1 + i], kind=kind) for i, kind in enumerate(ENGINES)}
    for kind in ENGINES:
        fit(models[kind], data, budget, None)
    held = data["heldout"]
    learned, nothing, rules = corrections(models["temperatura"], held)[2], np.zeros((held["tau"].size, N_CORR)), handbook(held)
    table = {}
    for split in ("heldout", "shifted"):
        brief = data[split]
        table[split] = {kind: error_of(models[kind], brief) for kind in ENGINES}
        for label, delta in (("rules", handbook(brief)), ("uncorrected", np.zeros((brief["tau"].size, N_CORR))),
                             ("optimum", per_front_optimum(brief, oracle_steps))):
            table[split][label] = float(perceived_error(delta, brief).mean())
    elsewhere = {kind: [float(perceived_error(d, brief).mean()) for d in (learned, nothing, rules)]
                 for kind, brief in data["displaced"].items()}
    lesions = {f"{name}:{mode}": error_of(knockout(models["temperatura"], name, mode), held) - table["heldout"]["temperatura"]
               for name, mode in (("temperatura", "zero"), ("temperatura", "mean"), ("statio", "mean"))}
    seen_elsewhere = np.array(list(elsewhere.values()))
    return {"data": data, "models": models, "table": table, "elsewhere": elsewhere, "lesions": lesions,
            "row": {"H-SIG": table["shifted"]["temperatura"] - table["shifted"]["appearance"],
                    "H-NEC": lesions["temperatura:zero"],
                    "H-BLIND": float(seen_elsewhere[:, 0].mean() - seen_elsewhere[:, 1].mean())},
            "no_illusions": [float(perceived_error(d, held, illusions=False).mean()) for d in (learned, nothing, rules)],
            "corrections": np.stack([learned.mean(axis=0), rules.mean(axis=0)]),
            "saturated": float((np.abs(hidden_states(models["temperatura"], held["z"])["hidden"]) > 0.99).mean())}


def verdicts(runs, seed, evaluated):
    rng = np.random.default_rng(seed + 9973)

    def interval(values):
        values = np.asarray(values, dtype=float)
        return paired_bootstrap(values, rng) if evaluated else (float(values.mean()), None)
    hypotheses = []
    for spec in MIND_CARD["hypotheses"]:
        mean, ci = interval([run["row"][spec["id"]] for run in runs])
        hypotheses.append({"id": spec["id"], "metric": spec["metric"], "mean_diff": mean, "ci95": ci, "mesi": spec["mesi"],
                           "n_seeds": len(runs),
                           "verdict": verdict(mean, ci, spec["mesi"], spec["direction"]) if evaluated else "not evaluated"})
    knockouts = []
    for label in runs[0]["lesions"]:
        mean, ci = interval([run["lesions"][label] for run in runs])
        name, mode = label.split(":")
        knockouts.append({"module": name, "mode": mode, "signature": modules(runs[0]["models"]["temperatura"])[name]["signature"],
                          "metric_change": mean, "ci95": ci})
    return hypotheses, knockouts


class Scroll:
    """Collects the verified report block, one line at a time."""

    def __init__(self, chapter):
        self.lines = [f"=== VERIFIED REPORT · chapter {chapter:04d} ==="]

    def put(self, text):
        self.lines.append(text)
        return self

    def each(self, template, rows):
        self.lines.extend(template.format(*row) for row in rows)
        return self

    def sealed(self):
        return self.lines + ["=== END REPORT ==="]


def span(ci):
    return "not evaluated" if ci is None else f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"


def compose(mode, seeds, runs, site, results, hypotheses, knockouts, bridge, elapsed, exit_code):
    first, caught = runs[0]["models"], site["caught"]

    def seed_mean(pick):
        return float(np.mean([pick(run) for run in runs]))
    labels = (("temperatura", "learned temperaturae"), ("appearance", "appearance-first baseline"),
              ("rules", "Vitruvius's rules"), ("constant", "constant correction"),
              ("uncorrected", "uncorrected core"), ("optimum", "per-front direct optimization"))
    parts = ("corner", "inner", "top ratio", "height", "architrave", "rise", "lean")
    shift = np.mean([run["corrections"] for run in runs], axis=0)
    g = site["gradcheck"]
    scroll = Scroll(107)
    scroll.put(f"file: {os.path.basename(__file__)} · card_revision {MIND_CARD['card_revision']} · mode {mode} · mutant {ACTIVE_MUTANT}")
    scroll.put(f"environment: python {sys.version.split()[0]} · numpy {np.__version__}")
    scroll.put(f"seeds: {seeds} · runtime_s {elapsed:.1f} · budget_s {TIME_BUDGET[mode]:.0f}")
    scroll.put("n_params: " + " · ".join(f"{label} {n_params(first[kind])}" for kind, label in labels[:2]) +
               f" · constant correction {n_params(first['constant'])}")
    scroll.put(f"gradcheck: {g['tensors_checked']}/{g['tensors_total']} tensors at init and after training · "
               f"max_rel_error {g['max_rel_error']:.2e} · passed {g['passed']}")
    scroll.put("correctness:").each("  {:<5} {:<29} {}  {}", [(c, n, "PASS" if p else "FAIL", d) for c, n, p, d in results])
    scroll.put(f"mutants: {sum(caught.values())}/{len(MUTANTS)} detected · score {sum(caught.values()) / len(MUTANTS):.2f} · "
               + ", ".join(f"{name} {'caught' if caught.get(name) else 'missed'}" for name in MUTANTS))
    scroll.put("hypotheses (paired over seeds; 95% percentile bootstrap of the mean, 2000 resamples):")
    scroll.each("  {:<8} mean_diff {:+.4f} ci95 {} mesi {} seeds {} -> {}",
                [(h["id"], h["mean_diff"], span(h["ci95"]), h["mesi"], h["n_seeds"], h["verdict"]) for h in hypotheses])
    scroll.put("  H-RIVAL  not applicable: no rival (any contrast with chapter 0130 is conceptual)")
    scroll.put("knockouts (held-out fronts at canonical stations; change in perceived error vs the full model):")
    scroll.each("  {:<12} {:<5} signature {:<5} {:+.4f} ci95 {}",
                [(k["module"], k["mode"], str(k["signature"]), k["metric_change"], span(k["ci95"])) for k in knockouts])
    scroll.put("perceived error by corrector (seed mean; held-out / shifted colossal):")
    scroll.each("  {:<30} {:.4f} / {:.4f}", [(label, seed_mean(lambda r, k=key: r["table"]["heldout"][k]),
                                             seed_mean(lambda r, k=key: r["table"]["shifted"][k])) for key, label in labels])
    scroll.put("displaced stations, held-out fronts corrected for their canonical station (learned / uncorrected / rules):")
    scroll.each("  {:<5} {:.4f} / {:.4f} / {:.4f}",
                [(kind, *(seed_mean(lambda r, k=kind, i=i: r["elsewhere"][k][i]) for i in range(3))) for kind in DISPLACED])
    scroll.put("eye without the posited illusions, canonical stations (learned / uncorrected / rules): "
               + " / ".join(f"{seed_mean(lambda r, i=i: r['no_illusions'][i]):.4f}" for i in range(3)))
    scroll.put("mean held-out correction, learned vs rules: "
               + " · ".join(f"{part} {a:+.3f} vs {b:+.3f}" for part, a, b in zip(parts, shift[0], shift[1])))
    scroll.put(f"saturated hidden units (|h| > 0.99, seed mean): {seed_mean(lambda r: r['saturated']):.3f}")
    scroll.put(f"real-data bridge: {bridge}").put(f"task_types: {', '.join(TASK_TYPES)}").put(f"exit_code: {exit_code}")
    payload = {"schema_version": "1.0", "chapter": 107, "file": os.path.basename(__file__),
               "card_revision": MIND_CARD["card_revision"],
               "environment": {"python": sys.version.split()[0], "numpy": np.__version__}, "seeds": seeds,
               "runtime_s": round(elapsed, 2), "n_params": n_params(first["temperatura"]), "gradcheck": g,
               "correctness": [{"id": c, "name": n, "passed": p, "detail": d} for c, n, p, d in results],
               "mutants": {"detected": sum(caught.values()), "total": len(MUTANTS),
                           "score": sum(caught.values()) / len(MUTANTS)},
               "hypotheses": hypotheses, "knockouts": knockouts, "task_types": TASK_TYPES, "exit_code": exit_code}
    return scroll.sealed(), payload


def protocol(mode, base_seed, n_seeds, json_path, data_path):
    began, budget = time.time(), STEPS[mode]
    seeds = [base_seed + i for i in range(n_seeds)]
    print(f"chapter 0107 · mode {mode} · seeds {seeds} · mutant {ACTIVE_MUTANT}", flush=True)
    runs = []
    for seed in seeds:
        runs.append(run_seed(seed, budget, ORACLE_STEPS[mode]))
        print(f"  seed {seed} done ({time.time() - began:.1f} s)", flush=True)
    site = {"base_seed": base_seed, "budget": budget, "data": runs[0]["data"], "engines": runs[0]["models"], "caught": {}}
    results = [(code, name, *fn(site)) for code, name, fn in CHECKS]
    hypotheses, knockouts = verdicts(runs, base_seed, mode == "full" and n_seeds >= 5)
    bridge = data_bridge(data_path, base_seed, budget) if data_path else "skipped (no --data PATH given)"
    elapsed = time.time() - began
    results.append(("C8", "budget", elapsed <= TIME_BUDGET[mode], f"{elapsed:.1f} s of {TIME_BUDGET[mode]:.0f} s"))
    failed = [code for code, _, passed, _ in results if not passed]
    exit_code = 0 if not failed else (3 if failed == ["C8"] else 1)
    lines, payload = compose(mode, seeds, runs, site, results, hypotheses, knockouts, bridge, elapsed, exit_code)
    write_report(lines, payload, json_path)
    return exit_code


def main(argv=None):
    parser = argparse.ArgumentParser(prog=os.path.basename(__file__),
                                     description="Chapter 0107, temperaturae. Without flags the full protocol runs.")
    parser.add_argument("--quick", help="one seed, reduced updates, all correctness tests", action="store_true")
    parser.add_argument("--seed", help="base seed (default 0)", type=int, default=0)
    parser.add_argument("--seeds", help="number of seeds (default 5, or 1 with --quick)", type=int)
    parser.add_argument("--json", help="also write the report as JSON to this path", metavar="PATH")
    parser.add_argument("--card", help="print MIND_CARD as JSON and exit", action="store_true")
    parser.add_argument("--mutant", help="activate a registered mutant: " + ", ".join(MUTANTS), metavar="NAME")
    parser.add_argument("--data", help="optional numeric CSV with a header; last column is the target", metavar="PATH")
    args = parser.parse_args(argv)
    if args.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    n_seeds = args.seeds if args.seeds is not None else (1 if args.quick else 5)
    problem = (f"unknown mutant {args.mutant!r}; registered: {', '.join(MUTANTS)}"
               if args.mutant is not None and args.mutant not in MUTANTS else
               "--seeds must be at least 1" if n_seeds < 1 else None)
    if problem:
        print(problem, file=sys.stderr)
        return 2
    try:
        with Mutation(args.mutant):
            return protocol("quick" if args.quick else "full", args.seed, n_seeds, args.json, args.data)
    except FloatingPointError as exc:
        print(f"non-finite values: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())

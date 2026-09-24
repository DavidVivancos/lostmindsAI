#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0111 · Strabo
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 6, Minds 101-120: https://www.amazon.com/dp/B0HF7G6JJD · Demos: https://artificiology.com/
# END ATTRIBUTION
"""Chorographic braid: a world map fitted jointly to eyewitness anchors, itineraries and measured parallels,
stating an uncertainty that grows toward the edges, with a plausibility screen for reports.

Thesis
    Where the senses reach, anchor the map on what was seen; where they cannot, let reason carry it from the
    distances already traversed and the parallels already measured, and say how uncertain the carried part is.

Evidence and provenance
    Provenance is belief: the Geography survives (Loeb text of H. L. Jones, 1917, as on LacusCurtius).
    D1  1.1.8: sense and experience show the inhabited world is an island; where the senses fail, reason points
        the way, reckoning from the parallel distances already traversed.
    D2  2.5.11: most material is received by hearsay; those who have seen a region serve as organs of sense,
        and the mind forms the whole from parts.
    D3  2.5.4 and 2.5.9: the inhabited world is measured by visiting it and by calculating intervals; its length
        comes partly from land journeys and partly from sea voyages.
    D4  2.5.14 and 2.5.16: places whose sun-dial shadows agree lie on one parallel; lines through known places
        serve as elements for correlating the rest.
    D5  1.1.16: even within one empire the nearer regions are better known than the far ones.
    D6  1.4.2-3: Eratosthenes adds 11,500 stadia from the Borysthenes to the parallel of Thule; Strabo rejects it,
        calling Pytheas an arch-falsifier whose errors about known regions discredit his unknown ones.
    D7  1.4.4 and 2.5.8: beyond Ierne places are no longer habitable; reckon no more than three or four thousand
        stadia north of Britain.
    D8  1.2.3 and 1.1.12: Strabo disputes Eratosthenes' view that poets aim to entertain, not instruct; Hipparchus
        holds that geography needs celestial observations and eclipses.
    D9  Roseman 1994; Cunliffe 2001; Roller 2006: much of Pytheas' northern voyage is now credited.

Doctrine -> mechanism -> test (IDs as in MIND_CARD)
    D1 D3 D4  M1 braid: anchors, itinerary legs and accepted parallels fitted jointly     C1 C6.1 H-NEC H-RIVAL
    D2 D5     M2 statement: Laplace covariance with one learned temperature              C6.2 H-SIG
    D6 D7     M3 screen: learned plausibility of claims (consistency, rim excess)        H-BLIND H-NEC
    D6 D8     rival: measurement-first map after Eratosthenes (chained legs, trusted parallels)  H-RIVAL
    D7 D9     blind spot: true reports from beyond the rim                               H-BLIND

Research question (calibration, abstention and metacognition)
    Away from the observations, is the uncertainty stated from the structure of the evidence better calibrated
    than one learned from past errors, and can a map built from sequential itineraries keep its geometry
    while rejecting fabricated reports without rejecting true surprises?

Closest prior art and the delta
    Pose-graph least squares with covariance from the information matrix (Grisetti et al. 2010); heteroscedastic
    variance heads (Nix and Weigend 1994); outlier gating of measurements. Delta: latitude-only parallels braided
    with range-bearing itineraries under route-level biases, a learned plausibility screen that includes the
    rim of the known world, and tests against a learned variance head, a measurement-first rival and true
    far-edge reports.

Blind spot
    A screen that learns "beyond the rim means fabricated" from a world where that was always so will reject
    the true surprise, as Strabo rejected Pytheas.

Task (generative process)
    Worlds in leg units. Five anchors (hub and four core places) observed with s.d. 0.03. Six routes radiate from
    the anchors; each leg turns by N(0, 0.25) rad and has length U(0.8, 1.2); training and held-out routes have
    4-6 legs, shifted routes 9-12. Up to six cross-links join places on different routes within 1.3. Each route
    biases its legs: scale N(1, 0.08) and rotation N(0, 0.07) rad; each leg adds distance noise 5 per cent and
    bearing noise 0.08 rad. A non-anchor place carries a latitude claim with probability 0.5 exp(-r/4), s.d.
    0.08, grossly wrong (+-1.5 to 3) with probability 0.15. Two fabricated far reports per world (half
    internally consistent); held-out worlds add one true far report beyond the northernmost route.
    Splits: 24 training, 24 held-out and 24 shifted worlds per seed.

Limits
    Flat plane, synthetic worlds, a hand-set nominal noise model, no longitude observations, and a screen that
    judges only latitude claims. A research prototype of one mechanism, not an AGI and not Strabo's mind.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 1,
    "revision_log": [],
    "generation": {"template_version": "codeguidelines 1.0 (15 September 2026), Appendix A",
                   "generator": "Claude (Anthropic)", "generator_version": "claude-opus-5", "date": "2026-09-15"},
    "id": 111, "figure": "Strabo", "born": -64, "died": None, "civilization": "Greek",
    "provenance": "belief",
    "thesis": ("Where the senses reach, anchor the map on what was seen; where they cannot, let reason carry it from "
               "the distances already traversed and the parallels already measured, and say how uncertain the "
               "carried part is."),
    "evidence": [
        {"id": "D1", "basis": "primary", "source": "Strabo, Geography 1.1.8 (trans. H. L. Jones, Loeb 1917)",
         "claim": "Where the senses cannot reach, reason points the way, reckoning from the parallel distances already "
                  "traversed."},
        {"id": "D2", "basis": "primary", "source": "Strabo, Geography 2.5.11",
         "claim": "Most material is received by hearsay; those who have seen a region serve as organs of sense, and the "
                  "mind forms the whole from the parts."},
        {"id": "D3", "basis": "primary", "source": "Strabo, Geography 2.5.4; 2.5.9",
         "claim": "The inhabited world is measured by visiting it and by calculating intervals; its length is measured "
                  "partly by land journeys and partly by sea voyages."},
        {"id": "D4", "basis": "primary", "source": "Strabo, Geography 2.5.14-16",
         "claim": "Places whose sun-dial shadows agree lie on one parallel; lines through known places are used as "
                  "elements to correlate the other positions."},
        {"id": "D5", "basis": "primary", "source": "Strabo, Geography 1.1.16",
         "claim": "Even in a single empire the nearer regions would be better known than the far ones."},
        {"id": "D6", "basis": "primary", "source": "Strabo, Geography 1.4.2-3",
         "claim": "Eratosthenes puts 11,500 stadia between the Borysthenes and the parallel of Thule; Strabo rejects "
                  "this, calling Pytheas an arch-falsifier whose falsehoods about known regions discredit the unknown."},
        {"id": "D7", "basis": "primary", "source": "Strabo, Geography 1.4.4; 2.5.8",
         "claim": "Beyond Ierne places are no longer habitable; one should reckon no more than three or four thousand "
                  "stadia north of Britain."},
        {"id": "D8", "basis": "primary", "source": "Strabo, Geography 1.2.3; 1.1.12",
         "claim": "Strabo disputes Eratosthenes' view that poets aim to entertain rather than instruct; Hipparchus holds "
                  "that geography needs celestial observations and eclipses."},
        {"id": "D9", "basis": "scholarship",
         "source": "Roseman 1994, Pytheas of Massalia: On the Ocean; Cunliffe 2001, The Extraordinary Voyage of Pytheas "
                   "the Greek; Roller 2006, Through the Pillars of Herakles",
         "claim": "Much of Pytheas' northern voyage and his observations are credited by modern scholarship."},
    ],
    "research_question": {
        "category": "calibration, abstention and metacognition",
        "question": ("Away from the observations, is uncertainty stated from the structure of the evidence better "
                     "calibrated than uncertainty learned from past errors, and can a map built from sequential "
                     "itineraries keep its geometry while rejecting fabricated reports without rejecting true surprises?")},
    "mechanism": {
        "name": "chorographic braid with stated edge uncertainty and a plausibility screen",
        "family": "nonlinear least-squares map (range-bearing factors) with Laplace-stated covariance",
        "signature_modules": ["braid", "statement"],
        "closest_prior_art": [
            "pose-graph least squares with covariance from the information matrix (Grisetti et al. 2010, A Tutorial on "
            "Graph-Based SLAM)",
            "heteroscedastic variance heads trained on residuals (Nix and Weigend 1994)",
            "chi-square gating of measurements before fusion"],
        "overlap": "High",
        "prior_art_queries": [],
        "prior_art_note": "No literature search was run for this card; overlap is rated against the named methods.",
        "contribution_type": "test",
        "delta": ("Latitude-only parallels braided with range-bearing itineraries under route-level biases, a learned "
                  "plausibility screen that includes the rim of the known world, and tests against a learned variance "
                  "head, a measurement-first rival after Eratosthenes, and true reports from beyond the rim."),
        "baselines": {
            "baseline": ("learned variance head: the same fitted map, with isotropic stated variance exp(h0 + h1 log(1 + "
                         "hops to nearest anchor) + h2 [has an accepted parallel]) fitted to training errors"),
            "rival": ("chapter 0093 Eratosthenes, minimal measurement-first map: legs chained from the anchors, every "
                      "measured parallel trusted without screening, far reports carrying a measurement accepted, and a "
                      "stated uncertainty equal to the measurement precision")}},
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1 braid", "property_test": "C6.1", "hypothesis": "H-NEC, H-RIVAL"},
        {"doctrine": "D2", "mechanism": "M2 statement", "property_test": "C6.2", "hypothesis": "H-SIG"},
        {"doctrine": "D3", "mechanism": "M1 braid (itinerary legs)", "property_test": "C6.1", "hypothesis": "H-RIVAL"},
        {"doctrine": "D4", "mechanism": "M1 braid (parallels)", "property_test": "C6.2", "hypothesis": "H-NEC"},
        {"doctrine": "D5", "mechanism": "M2 statement", "property_test": "C6.2", "hypothesis": "H-SIG"},
        {"doctrine": "D6", "mechanism": "M3 screen; rival", "property_test": "none", "hypothesis": "H-BLIND, H-RIVAL"},
        {"doctrine": "D7", "mechanism": "M3 screen (rim excess)", "property_test": "none", "hypothesis": "H-BLIND"},
        {"doctrine": "D8", "mechanism": "rival", "property_test": "none", "hypothesis": "H-RIVAL"},
        {"doctrine": "D9", "mechanism": "true far reports (blind-spot condition)", "property_test": "none",
         "hypothesis": "H-BLIND"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": ("At the edges of worlds longer than any seen in training, uncertainty stated from "
                                      "the braid's information is better calibrated than the learned variance head."),
         "metric": "edge_nlpd", "split": "shifted", "comparison": "model - baseline", "direction": "less",
         "mesi": 0.2, "seeds": 5},
        {"id": "H-NEC", "statement": ("Removing the measured parallels from the braid raises edge position error more "
                                      "than removing the plausibility screen does."),
         "metric": "edge_error", "split": "shifted",
         "comparison": "(parallels:identity - full) - (screen:identity - full)",
         "knockouts": ["parallels:identity", "screen:identity"], "direction": "greater", "mesi": 0.05, "seeds": 5},
        {"id": "H-BLIND", "statement": ("Reports that are true but lie beyond the rim of the known world are accepted "
                                        "less often by the full screen than by a screen that judges consistency only."),
         "condition": "one true far report per held-out world, beyond its northernmost route",
         "grounding": ("1.4.3-4 and 2.5.8: Strabo rejects Pytheas' Thule, reasoning that places beyond Ierne are no "
                       "longer habitable; modern scholarship credits much of the voyage."),
         "metric": "far_recall", "split": "heldout", "comparison": "model - consistency-only screen",
         "direction": "less", "mesi": 0.2, "seeds": 5},
        {"id": "H-RIVAL", "statement": ("At the edges of long worlds, the braid places places closer to the truth than the "
                                        "measurement-first map after Eratosthenes."),
         "metric": "edge_error", "split": "shifted", "comparison": "model - rival", "direction": "less",
         "mesi": 0.1, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.3, "margin_over_trivial": 0.3, "shuffled_ratio_min": 0.9,
                   "gradcheck_rel_error": 1e-5, "gradcheck_floor": 1e-3, "invariance_tol": 1e-9,
                   "negative_control_min_violation": 1e-6},
    "metrics": {"edge_error": "mean distance between fitted and true positions of places at least 7 legs from the nearest anchor",
                "edge_nlpd": "mean negative log density of the true edge positions under the stated 2-D Gaussian",
                "far_recall": "fraction of true far reports the screen accepts",
                "trivial_baseline": "every non-anchor place put at the centroid of the observed anchors",
                "shuffled_band": ("one-sided: a map fitted to legs and parallels shuffled within each world must not beat "
                                  "the trivial baseline by more than a tenth")},
    "training": {"optimizer": "Adam", "lr_grid": [0.05], "clip_norm": 5.0, "model_selection": "none: final parameters",
                 "map": {"updates": {"full": 800, "quick": 300}, "schedule": "cosine decay to 5 per cent",
                         "init": "dead reckoning from the anchors"},
                 "calibration": {"updates": {"full": 300, "quick": 150}, "applies_to": "temperature and variance head"},
                 "screen": {"updates": {"full": 400, "quick": 200}, "lr": 0.1, "l2": 0.01,
                            "applies_to": "full and consistency-only screens"},
                 "nominal_noise": {"anchor": 0.03, "leg_distance_relative": 0.06, "leg_bearing_rad": 0.10,
                                   "parallel": 0.08}},
    "task": {"units": "legs", "anchors": 5, "routes": 6, "legs_per_route": {"train": [4, 6], "heldout": [4, 6],
                                                                              "shifted": [9, 12]},
             "cross_links": {"max": 6, "within": 1.3},
             "route_bias": {"scale_sd": 0.08, "rotation_sd_rad": 0.07},
             "leg_noise": {"distance_log_sd": 0.05, "bearing_sd_rad": 0.08},
             "parallels": {"rate": "0.5 exp(-r/4)", "sd": 0.08, "gross_rate": 0.15, "gross_offset": [1.5, 3.0]},
             "far_reports": {"fabricated_per_world": 2, "consistent_share": 0.5, "true_per_heldout_world": 1,
                             "distance": [3.0, 5.0]},
             "worlds_per_split": {"train": 24, "heldout": 24, "shifted": 24}, "edge_hops": 7},
    "probe_predictions": [{"probe": "P10", "expected": "equal to baseline"}],
    "probe_support": "vector_regression through a ridge linear map (the braid's linear special case) on X, Y batches",
    "dialectic_links": [{"chapter": 93, "relation": "rival", "test": "H-RIVAL"}],
    "corpus_neighbors": [
        {"chapter": 93, "similarity": None, "difference": ("0093 measures the world from a few observations; here many "
                                                           "rough itineraries are braided with them and the uncertainty "
                                                           "of the carried part is stated.")},
        {"chapter": 99, "similarity": None, "difference": "0099 differences records across epochs; here one epoch is mapped in space."},
        {"chapter": 58, "similarity": None, "difference": ("0058 weighs rival accounts by their sources; here reports are "
                                                           "screened by their content against the map, not by who made them.")},
        {"chapter": 75, "similarity": None, "difference": "0075 cross-checks suspect informants; here no informant is modelled."},
        {"chapter": 157, "similarity": None, "difference": "0157 weighs four instruments of knowledge; here the object is geometry."},
        {"chapter": 200, "similarity": None, "difference": "0200 grounds rulings in consensus practice; unrelated mechanism."},
    ],
    "similarity_note": "Nearest-neighbour similarity not computed into the card; the audit reports it for the files at hand.",
    "barometer": {
        "cognitive_processing": ["interpolation away from observations with stated uncertainty"],
        "embodied_cognition": ["map-building from sequential journeys (itineraries)"],
        "world_modeling": ["geometry reconstructed from anchors, legs and parallels"],
        "consciousness": ["stated uncertainty as self-monitoring of the map"],
        "language_understanding": [], "emotional_intelligence": [], "creativity": [], "autonomy": []},
    "task_types": ["vector_regression"],
    "applications": [
        {"use": ("robot and vehicle mapping from odometry, sparse fixes and outlier-screened landmarks, with stated "
                 "uncertainty at the edges of the explored area"),
         "sector": "robotics and autonomous navigation", "dataset": "Intel Research Lab SLAM dataset", "readiness": "low"},
        {"use": "reconstructing historical maps from itinerary distances with stated positional uncertainty",
         "sector": "digital humanities and historical GIS", "dataset": "Pleiades gazetteer of ancient places",
         "readiness": "low"},
        {"use": ("sparse environmental monitoring networks whose interpolated values carry uncertainty that grows away "
                 "from stations, and whose anomalous reports are screened without discarding true extremes"),
         "sector": "environmental monitoring", "dataset": "NOAA Global Historical Climatology Network (GHCN-Daily)",
         "readiness": "low"},
    ],
    "safety_notes": ("No environmental determinism: nothing in the file maps place, climate or position to human "
                     "character or behaviour, and the old climate-to-character component is not carried forward. "
                     "Worlds are synthetic geometry only. The file does not claim to replicate Strabo's mind."),
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

SIGMA = {"anchor": 0.03, "distance": 0.06, "bearing": 0.10, "parallel": 0.08}
N_ANCHORS, N_ROUTES, EDGE_HOPS = 5, 6, 7
LEGS = {"train": (4, 6), "heldout": (4, 6), "shifted": (9, 12)}
WORLDS = {"train": 24, "heldout": 24, "shifted": 24}
STEPS = {"full": {"map": 800, "calibration": 300, "screen": 400}, "quick": {"map": 300, "calibration": 150, "screen": 200}}
LR, SCREEN_LR, SCREEN_L2, CLIP_NORM = 0.05, 0.1, 0.01, 5.0
CHI2_90 = 4.605170185988091
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


# ---------------------------------------------------------------- worlds: anchors, itineraries, parallels, far reports
def heading(bearing):
    """Unit vector (east, north) for a bearing measured clockwise from north."""
    return np.stack([np.sin(bearing), np.cos(bearing)], axis=-1)


def one_world(rng, legs_range, true_far):
    pos = [np.zeros(2)] + [rng.uniform(0.8, 1.6) * heading(2 * math.pi * k / 4 + rng.uniform(-0.4, 0.4)) for k in range(4)]
    route_of, legs, tips = [-1] * N_ANCHORS, [], []
    for k in range(N_ROUTES):
        cur, bearing = k % N_ANCHORS, 2 * math.pi * k / N_ROUTES + rng.uniform(-0.3, 0.3)
        for _ in range(rng.integers(legs_range[0], legs_range[1] + 1)):
            bearing += rng.normal(0.0, 0.25)
            pos.append(pos[cur] + rng.uniform(0.8, 1.2) * heading(bearing))
            route_of.append(k)
            legs.append((cur, len(pos) - 1, k))
            cur = len(pos) - 1
        tips.append(cur)
    pos, linked = np.array(pos), {(i, j) for i, j, _ in legs}
    gaps = sorted((float(np.linalg.norm(pos[i] - pos[j])), i, j) for i in range(N_ANCHORS, len(pos))
                  for j in range(i + 1, len(pos)) if route_of[i] != route_of[j] and (i, j) not in linked)
    legs += [(i, j, -1) for d, i, j in gaps[:6] if d < 1.3]
    scale, turn = 1.0 + rng.normal(0.0, 0.08, N_ROUTES), rng.normal(0.0, 0.07, N_ROUTES)
    table = []
    for i, j, k in legs:
        delta = pos[j] - pos[i]
        s, t = (scale[k], turn[k]) if k >= 0 else (1.0, 0.0)
        table.append((i, j, k, float(np.linalg.norm(delta)) * s * math.exp(rng.normal(0.0, 0.05)),
                      math.atan2(delta[0], delta[1]) + t + rng.normal(0.0, 0.08)))
    claims = []
    for i in range(N_ANCHORS, len(pos)):
        if rng.random() < 0.5 * math.exp(-float(np.linalg.norm(pos[i])) / 4.0):
            gross = rng.random() < 0.15
            offset = rng.choice([-1.0, 1.0]) * rng.uniform(1.5, 3.0) if gross else 0.0
            claims.append((i, pos[i, 1] + rng.normal(0.0, SIGMA["parallel"]) + offset, not gross))
    far = []
    for consistent in (True, False):
        tip = tips[rng.integers(N_ROUTES)]
        d, b = rng.uniform(3.0, 5.0), math.atan2(pos[tip, 0], pos[tip, 1]) + rng.uniform(-0.5, 0.5)
        lie = rng.normal(0.0, SIGMA["parallel"]) if consistent else rng.choice([-1.0, 1.0]) * rng.uniform(1.5, 3.0)
        far.append((tip, d, b, pos[tip, 1] + d * math.cos(b) + lie, False, np.full(2, np.nan)))
    if true_far:
        tip = max(tips, key=lambda t: pos[t, 1])
        d, b = rng.uniform(3.0, 5.0), rng.normal(0.0, 0.15)
        spot = pos[tip] + d * heading(b)
        far.append((tip, d * math.exp(rng.normal(0.0, 0.05)), b + rng.normal(0.0, 0.08),
                    spot[1] + rng.normal(0.0, SIGMA["parallel"]), True, spot))
    return {"pos": pos, "anchors": pos[:N_ANCHORS] + rng.normal(0.0, SIGMA["anchor"], (N_ANCHORS, 2)),
            "legs": table, "claims": claims, "far": far, "route_range": legs_range}


def reckon(world):
    """Dead reckoning along the routes from the observed anchors, the variance it accumulates, and anchor hops."""
    n = len(world["pos"])
    dr, var, hops = np.zeros((n, 2)), np.zeros(n), np.full(n, 10 ** 6)
    dr[:N_ANCHORS], var[:N_ANCHORS], hops[:N_ANCHORS] = world["anchors"], SIGMA["anchor"] ** 2, 0
    for i, j, k, d, b in world["legs"]:
        if k >= 0:
            dr[j] = dr[i] + d * heading(b)
            var[j] = var[i] + 0.5 * d * d * (SIGMA["distance"] ** 2 + SIGMA["bearing"] ** 2)
    changed = True
    while changed:
        changed = False
        for i, j, _, _, _ in world["legs"]:
            for a, c in ((i, j), (j, i)):
                if hops[a] + 1 < hops[c]:
                    hops[c], changed = hops[a] + 1, True
    return dr, var, hops


def pack(worlds):
    """Pad a list of worlds into arrays; claims and far reports carry their screen features z and rim excess."""
    size = max(len(w["pos"]) for w in worlds)
    out = {"worlds": worlds, "truth": np.zeros((len(worlds), size, 2)), "mask": np.zeros((len(worlds), size), bool),
           "dr": np.zeros((len(worlds), size, 2)), "hops": np.zeros((len(worlds), size), int),
           "anchors": np.stack([w["anchors"] for w in worlds]),
           "route_range": (min(w["route_range"][0] for w in worlds), max(w["route_range"][1] for w in worlds))}
    legs, claims, far = [], [], []
    spread = SIGMA["distance"] ** 2 + SIGMA["bearing"] ** 2
    for w, world in enumerate(worlds):
        n = len(world["pos"])
        dr, var, hops = reckon(world)
        out["truth"][w, :n], out["mask"][w, :n], out["dr"][w, :n], out["hops"][w, :n] = world["pos"], True, dr, hops
        rim = float(np.linalg.norm(dr[N_ANCHORS:], axis=1).max())
        legs += [(w, i, j, d, b) for i, j, k, d, b in world["legs"]]
        for i, y, ok in world["claims"]:
            excess = max(0.0, float(np.linalg.norm(dr[i])) - rim) / rim
            claims.append((w, i, y, ok, abs(y - dr[i, 1]) / math.sqrt(var[i] + SIGMA["parallel"] ** 2), excess))
        for tip, d, b, y, ok, spot in world["far"]:
            reach, v = dr[tip] + d * heading(b), var[tip] + 0.5 * d * d * spread
            far.append((w, tip, y, ok, abs(y - reach[1]) / math.sqrt(v + SIGMA["parallel"] ** 2),
                        max(0.0, float(np.linalg.norm(reach)) - rim) / rim))
    columns = lambda rows, names: {name: np.array([r[k] for r in rows]) for k, name in enumerate(names)}
    out["legs"] = columns(legs, ("w", "i", "j", "d", "b"))
    out["claims"] = columns(claims, ("w", "i", "y", "ok", "z", "excess"))
    out["far"] = columns(far, ("w", "tip", "y", "ok", "z", "excess"))
    for table in (out["legs"], out["claims"], out["far"]):
        for key in ("w", "i", "j", "tip"):
            if key in table:
                table[key] = table[key].astype(int)
    out["claims"]["ok"], out["far"]["ok"] = out["claims"]["ok"].astype(bool), out["far"]["ok"].astype(bool)
    return out


def make_split(rng, name):
    return pack([one_world(rng, LEGS[name], name == "heldout") for _ in range(WORLDS[name])])


def scrambled(split, rng):
    """The same worlds with leg observations and parallel values shuffled within each world (control C4)."""
    worlds = []
    for world in split["worlds"]:
        legs, claims = world["legs"], world["claims"]
        order, shuffle = rng.permutation(len(legs)), rng.permutation(len(claims))
        new_legs = [(i, j, k, legs[o][3], legs[o][4]) for (i, j, k, _, _), o in zip(legs, order)]
        new_claims = [(i, claims[o][1], ok) for (i, _, ok), o in zip(claims, shuffle)]
        worlds.append(dict(world, legs=new_legs, claims=new_claims))
    return pack(worlds)


# ---------------------------------------------------------------- the braid, its statement, the screen and the rival
def leg_terms(X, legs):
    """Range and cross-track residuals of every leg and their derivatives with respect to the displacement."""
    delta = X[legs["w"], legs["j"]] - X[legs["w"], legs["i"]]
    rho = np.sqrt((delta ** 2).sum(axis=1))
    toward = heading(legs["b"])
    normal = np.stack([toward[:, 1], -toward[:, 0]], axis=1)
    cross = (normal * delta).sum(axis=1)
    r_d = (rho - legs["d"]) / (SIGMA["distance"] * legs["d"])
    r_b = cross / (rho * SIGMA["bearing"])
    g_d = delta / (rho * SIGMA["distance"] * legs["d"])[:, None]
    g_b = (normal - (cross / rho ** 2)[:, None] * delta) / (rho * SIGMA["bearing"])[:, None]
    if ACTIVE_MUTANT == "halved_range_jacobian":
        g_d = 0.5 * g_d
    if ACTIVE_MUTANT == "zero_bearing_gradient":
        g_b = np.zeros_like(g_b)
    return r_d, r_b, g_d, g_b


def braid_loss(X, split, accepted):
    """Half the summed squared residuals of anchors, legs and accepted parallels, per world; and its gradient."""
    grad, legs, claims = np.zeros_like(X), split["legs"], split["claims"]
    r_a = (X[:, :N_ANCHORS] - split["anchors"]) / SIGMA["anchor"]
    grad[:, :N_ANCHORS] += r_a / SIGMA["anchor"]
    r_d, r_b, g_d, g_b = leg_terms(X, legs)
    push = r_d[:, None] * g_d + r_b[:, None] * g_b
    np.add.at(grad, (legs["w"], legs["j"]), push)
    np.add.at(grad, (legs["w"], legs["i"]), -push)
    w, i, y = claims["w"][accepted], claims["i"][accepted], claims["y"][accepted]
    r_p = (X[w, i, 1] - y) / SIGMA["parallel"]
    np.add.at(grad[:, :, 1], (w, i), r_p / SIGMA["parallel"])
    loss = 0.5 * float((r_a ** 2).sum() + (r_d ** 2).sum() + (r_b ** 2).sum() + (r_p ** 2).sum())
    return loss / X.shape[0], grad / X.shape[0]


def information(X, split, accepted):
    """Gauss-Newton information of the braid, one 2N x 2N matrix per world (padding kept invertible)."""
    worlds, size = X.shape[:2]
    H = np.zeros((worlds, 2 * size, 2 * size))
    diag = np.arange(2 * size)
    H[:, diag, diag] = np.where(np.repeat(split["mask"], 2, axis=1), 1e-6, 1.0)
    H[:, diag[:2 * N_ANCHORS], diag[:2 * N_ANCHORS]] += SIGMA["anchor"] ** -2
    legs = split["legs"]
    for g in leg_terms(X, legs)[2:]:
        outer = g[:, :, None] * g[:, None, :]
        for (p, sp), (q, sq) in itertools.product(((legs["i"], -1.0), (legs["j"], 1.0)), repeat=2):
            for r, s in itertools.product((0, 1), repeat=2):
                np.add.at(H, (legs["w"], 2 * p + r, 2 * q + s), sp * sq * outer[:, r, s])
    w, i = split["claims"]["w"][accepted], split["claims"]["i"][accepted]
    np.add.at(H, (w, 2 * i + 1, 2 * i + 1), SIGMA["parallel"] ** -2)
    return H


def statement(H):
    """Per-place 2 x 2 blocks of the inverse information: the uncertainty the braid states before calibration."""
    C, ix = np.linalg.inv(H), 2 * np.arange(H.shape[1] // 2)
    return np.stack([np.stack([C[:, ix, ix], C[:, ix, ix + 1]], -1), np.stack([C[:, ix + 1, ix], C[:, ix + 1, ix + 1]], -1)], -2)


def features(z, excess, kind):
    columns = [np.ones_like(z), z] + ([excess] if kind == "screen" else [])
    return np.stack(columns, axis=1)


def accepts(screen, table):
    return sigmoid(features(table["z"], table["excess"], screen["kind"]) @ screen["params"]["w"]) >= 0.5


def rival_map(split):
    """Measurement-first map after Eratosthenes: legs chained from the anchors, every measured parallel trusted."""
    X, claims = split["dr"].copy(), split["claims"]
    X[claims["w"], claims["i"], 1] = claims["y"]
    return X


def chart(split, screen, steps, ko=None):
    """Screen the parallels, fit the braid from dead reckoning, and attach its stated covariance blocks.
    ko is a knockout table from knockout(): parallels:identity drops every claim, screen:identity admits all."""
    ko, total = ko or {}, split["claims"]["y"].size
    if ko.get("parallels") == "identity":
        accepted = np.zeros(total, bool)
    elif ko.get("screen") == "identity":
        accepted = np.ones(total, bool)
    else:
        accepted = accepts(screen, split["claims"])
    braid = build_model(2, 2, TASK_TYPES[0], None, kind="braid", split=split, accepted=accepted)
    braid["ko"] = dict(ko)
    fit(braid, {"train": braid["batch"]}, steps, None)
    braid["S"] = statement(information(braid["params"]["X"], split, accepted))
    return braid


def non_anchor(split):
    chosen = split["mask"].copy()
    chosen[:, :N_ANCHORS] = False
    return chosen


def calibration_batch(braid, split):
    chosen, claims = non_anchor(split), split["claims"]
    e, S = (braid["params"]["X"] - split["truth"])[chosen], braid["S"][chosen]
    parallel = np.zeros(split["mask"].shape)
    parallel[claims["w"][braid["accepted"]], claims["i"][braid["accepted"]]] = 1.0
    return {"q": np.einsum("mi,mij,mj->m", e, np.linalg.inv(S), e), "e2": (e ** 2).sum(-1),
            "logdet": np.log(np.linalg.det(S)),
            "f": np.stack([np.ones(e.shape[0]), np.log1p(split["hops"][chosen]), parallel[chosen]], 1)}


def screen_batch(split):
    c, f = split["claims"], split["far"]
    return {"z": np.concatenate([c["z"], f["z"]]), "excess": np.concatenate([c["excess"], f["excess"]]),
            "label": np.concatenate([c["ok"], f["ok"]]).astype(float)}


# ---------------------------------------------------------------- model interface
def build_model(in_dim, out_dim, task_type, rng, **cfg):
    """kind: braid (positions of one split's worlds), calibration (temperature and variance head),
    screen or consistency (logistic plausibility screens), ridge (plain regression for probes and --data)."""
    if task_type not in TASK_TYPES:
        raise ValueError("chapter 0111 supports vector_regression only")
    kind = cfg.get("kind", "ridge")
    model = {"kind": kind, "ko": {}, "lr": cfg.get("lr", SCREEN_LR if kind in ("screen", "consistency") else LR)}
    if kind == "braid":
        split = cfg["split"]
        model.update(params={"X": split["dr"].copy()}, accepted=cfg["accepted"],
                     batch={"split": split, "accepted": cfg["accepted"]})
    elif kind == "calibration":
        model["params"] = {"log_tau": np.zeros(1), "head": np.array([math.log(0.1), 0.0, 0.0])}
    elif kind in ("screen", "consistency"):
        model["params"] = {"w": np.zeros(3 if kind == "screen" else 2)}
    else:
        model["params"] = {"W": rng.normal(0.0, in_dim ** -0.5, (out_dim, in_dim)), "b": np.zeros(out_dim)}
    return model


def loss_and_grads(model, batch):
    P, kind = model["params"], model["kind"]
    if kind == "braid":
        loss, grad = braid_loss(P["X"], batch["split"], batch["accepted"])
        return loss, {"X": grad}
    if kind == "calibration":
        tau, sigma2 = np.exp(-P["log_tau"][0]), np.exp(batch["f"] @ P["head"])
        a, b = 0.5 * batch["q"] * tau, 0.5 * batch["e2"] / sigma2
        loss = float((a + P["log_tau"][0]).mean() + (b + batch["f"] @ P["head"]).mean())
        return loss, {"log_tau": np.array([float((1.0 - a).mean())]), "head": batch["f"].T @ (1.0 - b) / b.size}
    if kind in ("screen", "consistency"):
        F = features(batch["z"], batch["excess"], kind)
        p = sigmoid(F @ P["w"])
        penalty = SCREEN_L2 * float((P["w"][1:] ** 2).sum())
        bce = -float(np.mean(batch["label"] * np.log(p + 1e-12) + (1.0 - batch["label"]) * np.log(1.0 - p + 1e-12)))
        grad = F.T @ (p - batch["label"]) / p.size
        grad[1:] += 2.0 * SCREEN_L2 * P["w"][1:]
        return bce + penalty, {"w": grad}
    residual = batch["X"] @ P["W"].T + P["b"] - batch["Y"]
    return float((residual ** 2).mean()), {"W": 2.0 * residual.T @ batch["X"] / residual.size,
                                           "b": 2.0 * residual.sum(axis=0) / residual.size}


def fit(model, data, budget, rng):
    """Adam with cosine decay to 5 per cent. Everything is full-batch and deterministic, so rng is unused."""
    state, history = adam_init(model["params"]), []
    for step in range(budget):
        loss, grads = loss_and_grads(model, data["train"])
        if not math.isfinite(loss) or not all(np.isfinite(g).all() for g in grads.values()):
            raise FloatingPointError(f"non-finite loss or gradient at update {step + 1}")
        grads = clip_global(grads, CLIP_NORM)[0]
        if ACTIVE_MUTANT == "sign_flipped_update":
            grads = {k: -g for k, g in grads.items()}
        rate = 0.0 if ACTIVE_MUTANT == "zero_learning_rate" else model["lr"] * (0.05 + 0.475 * (1.0 + math.cos(math.pi * step / budget)))
        adam_step(model["params"], grads, state, rate)
        history.append(loss)
    model["history"] = history
    return history


def predict(model, X):
    """Ridge models return fitted values; a braid returns its map, a screen its acceptance probabilities."""
    if model["kind"] == "braid":
        return model["params"]["X"]
    if model["kind"] in ("screen", "consistency"):
        return sigmoid(np.asarray(X, dtype=float) @ model["params"]["w"])
    return np.asarray(X, dtype=float) @ model["params"]["W"].T + model["params"]["b"]


def hidden_states(model, X):
    """For a braid: the per-place 2 x 2 blocks of inverse information and the stated s.d. (square root of half the trace)."""
    blocks = model.get("S")
    spread = None if blocks is None else np.sqrt(np.trace(blocks, axis1=-2, axis2=-1) / 2)
    return {"stated_blocks": blocks, "stated_sd": spread, "output": predict(model, X)}


def modules(model):
    kind = model["kind"]
    table = {"braid": (["X"] if kind == "braid" else [], "range-bearing least-squares map over anchors, legs and parallels", True),
             "parallels": ([], "latitude-only constraints from accepted claims", True),
             "statement": (["log_tau"] if kind == "calibration" else [], "inverse information scaled by a learned temperature", True),
             "screen": (["w"] if kind in ("screen", "consistency") else [], "logistic plausibility screen: consistency, rim excess",
                        False)}
    return {name: {"params": p, "role": role, "signature": sig} for name, (p, role, sig) in table.items()}


KNOCKOUT_MODES = {"parallels": ("identity",), "screen": ("identity",), "statement": ("mean",)}


def knockout(model, name, mode):
    if mode not in KNOCKOUT_MODES.get(name, ()):
        raise ValueError(f"no knockout {name}:{mode}")
    return dict(model, params={k: v.copy() for k, v in model["params"].items()}, ko=dict(model["ko"], **{name: mode}))


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


MUTANTS = {
    "sign_flipped_update": ("every update climbs the objective", "C3"),
    "zero_learning_rate": ("the optimizer never moves the parameters", "C3"),
    "zero_bearing_gradient": ("the cross-track residual passes no gradient to the map", "C1"),
    "halved_range_jacobian": ("the range residual's derivative is halved", "C1"),
}


def data_bridge(path, seed, budget):
    """Optional real data: numeric CSV, one header row, target in the last column; every fifth row is held out."""
    try:
        table = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    except (OSError, ValueError) as exc:
        return f"skipped ({exc.__class__.__name__} reading {path})"
    inputs, target = table[:, :-1], table[:, -1:]
    inputs = (inputs - inputs.mean(0)) / (inputs.std(0) + 1e-12)
    held = np.arange(len(target)) % 5 == 4
    ridge = build_model(inputs.shape[1], 1, TASK_TYPES[0], np.random.default_rng(seed))
    fit(ridge, {"train": {"X": inputs[~held], "Y": target[~held]}}, budget, None)

    def miss(guess):
        return float(np.sqrt(np.mean((guess - target[held]) ** 2)))
    return (f"{os.path.basename(path)}: every fifth row held out, RMSE {miss(predict(ridge, inputs[held])):.4f} "
            f"(the training mean scores {miss(target[~held].mean()):.4f})")


# ---------------------------------------------------------------- evaluation of one seed
def score(X, split, chosen, S=None, tau=1.0, var=None):
    """Mean error, NLPD and 90 per cent coverage of true positions under a stated Gaussian (full blocks or isotropic)."""
    e = (X - split["truth"])[chosen]
    if S is not None:
        blocks = tau * S[chosen]
        q = np.einsum("mi,mij,mj->m", e, np.linalg.inv(blocks), e)
        nlpd = 0.5 * q + math.log(2 * math.pi) + 0.5 * np.log(np.linalg.det(blocks))
    else:
        q = (e ** 2).sum(-1) / var
        nlpd = 0.5 * q + math.log(2 * math.pi) + np.log(var)
    return {"error": float(np.sqrt((e ** 2).sum(-1)).mean()), "nlpd": float(nlpd.mean()), "cover": float((q <= CHI2_90).mean())}


def trivial_error(split):
    centre = split["anchors"].mean(axis=1, keepdims=True)
    return float(np.sqrt(((split["truth"] - centre) ** 2).sum(-1))[non_anchor(split)].mean())


def edge_rows(charts, calibration, shifted, edge):
    tau, braid = float(np.exp(calibration["params"]["log_tau"][0])), charts["shifted"]
    head_var = np.zeros(shifted["mask"].shape)
    head_var[non_anchor(shifted)] = np.exp(calibration_batch(braid, shifted)["f"] @ calibration["params"]["head"])
    mean_var = float(tau * (hidden_states(braid, None)["stated_sd"][edge] ** 2).mean())
    X = braid["params"]["X"]
    return tau, {"braid": score(X, shifted, edge, S=braid["S"], tau=tau), "variance head": score(X, shifted, edge, var=head_var[edge]),
                 "Eratosthenes rival": score(rival_map(shifted), shifted, edge, var=SIGMA["parallel"] ** 2),
                 "statement:mean": score(X, shifted, edge, var=mean_var)}


def run_seed(seed, mode):
    steps, streams = STEPS[mode], [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(5)]
    splits = {name: make_split(streams[k], name) for k, name in enumerate(WORLDS)}
    screens = {kind: build_model(3, 1, TASK_TYPES[0], streams[3], kind=kind) for kind in ("screen", "consistency")}
    for model in screens.values():
        fit(model, {"train": screen_batch(splits["train"])}, steps["screen"], None)
    charts = {name: chart(splits[name], screens["screen"], steps["map"]) for name in WORLDS}
    calibration = build_model(3, 1, TASK_TYPES[0], streams[4], kind="calibration")
    fit(calibration, {"train": calibration_batch(charts["train"], splits["train"])}, steps["calibration"], None)
    shifted, held = splits["shifted"], splits["heldout"]
    edge = shifted["mask"] & (shifted["hops"] >= EDGE_HOPS)
    tau, rows = edge_rows(charts, calibration, shifted, edge)
    lesion_error = {}
    for name in ("parallels", "screen"):
        table = knockout(charts["shifted"], name, "identity")["ko"]
        lesion_error[name] = score(chart(shifted, screens["screen"], steps["map"], ko=table)["params"]["X"], shifted, edge, var=1.0)["error"]
    far = held["far"]
    taken = {kind: accepts(model, far) for kind, model in screens.items()}
    spread = hidden_states(charts["shifted"], None)["stated_sd"][non_anchor(shifted)]
    return {"splits": splits, "screens": screens, "charts": charts, "calibration": calibration, "rows": rows, "tau": tau,
            "row": {"H-SIG": rows["braid"]["nlpd"] - rows["variance head"]["nlpd"],
                    "H-NEC": lesion_error["parallels"] - lesion_error["screen"],
                    "H-BLIND": float(taken["screen"][far["ok"]].mean() - taken["consistency"][far["ok"]].mean()),
                    "H-RIVAL": rows["braid"]["error"] - rows["Eratosthenes rival"]["error"]},
            "lesions": {"parallels:identity": lesion_error["parallels"] - rows["braid"]["error"],
                        "screen:identity": lesion_error["screen"] - rows["braid"]["error"],
                        "statement:mean": rows["statement:mean"]["nlpd"] - rows["braid"]["nlpd"]},
            "far": {kind: (float(a[far["ok"]].mean()), float(a[~far["ok"]].mean())) for kind, a in taken.items()},
            "interior": score(charts["heldout"]["params"]["X"], held, non_anchor(held), S=charts["heldout"]["S"], tau=tau),
            "trivial": trivial_error(held), "sigma_hops": float(np.corrcoef(spread, shifted["hops"][non_anchor(shifted)])[0, 1])}


# ---------------------------------------------------------------- tests: correctness
def use_mutant(name):
    global ACTIVE_MUTANT
    previous, ACTIVE_MUTANT = ACTIVE_MUTANT, name
    return previous


def worst_gradient(pairs, rng, entries):
    floor, worst = MIND_CARD["thresholds"]["gradcheck_floor"], 0.0
    for model, batch in pairs:
        table = finite_difference_check(model["params"], loss_and_grads(model, batch)[1],
                                        lambda m=model, b=batch: loss_and_grads(m, b)[0], rng, n_entries=entries, floor=floor)
        worst = max(worst, max(table.values()))
    return worst, sum(len(m["params"]) for m, _ in pairs)


def learning_ok(braid, split):
    drop = 1.0 - float(np.mean(braid["history"][-20:])) / braid["history"][0]
    error, trivial = score(braid["params"]["X"], split, non_anchor(split), var=1.0)["error"], trivial_error(split)
    limits = MIND_CARD["thresholds"]
    return drop >= limits["loss_drop_fraction"] and error <= (1.0 - limits["margin_over_trivial"]) * trivial, drop, error, trivial


def replay(site):
    held = site["splits"]["heldout"]
    try:
        fresh = build_model(2, 2, TASK_TYPES[0], None, kind="braid", split=held, accepted=accepts(site["screens"]["screen"], held["claims"]))
        worst = worst_gradient([(fresh, fresh["batch"])], np.random.default_rng(site["seed"]), 4)[0]
        braid = chart(held, site["screens"]["screen"], site["steps"]["map"])
        return worst <= MIND_CARD["thresholds"]["gradcheck_rel_error"] and learning_ok(braid, held)[0]
    except (FloatingPointError, np.linalg.LinAlgError):
        return False


def gradient_pairs(site, trained):
    train, charts = site["splits"]["train"], site["charts"]
    evidence = calibration_batch(charts["train"], train)
    if trained:
        pairs = [(charts["train"], charts["train"]["batch"]), (site["calibration"], evidence)]
        return pairs + [(site["screens"][k], screen_batch(train)) for k in ("screen", "consistency")]
    braid = build_model(2, 2, TASK_TYPES[0], None, kind="braid", split=train, accepted=charts["train"]["accepted"])
    pairs = [(braid, braid["batch"]), (build_model(3, 1, TASK_TYPES[0], None, kind="calibration"), evidence)]
    return pairs + [(build_model(3, 1, TASK_TYPES[0], None, kind=k), screen_batch(train)) for k in ("screen", "consistency")]


def correctness(site):
    """Yield (code, name, passed, detail) for C1 to C7 in order, recording the gradient summary in the site."""
    limits, splits, charts = MIND_CARD["thresholds"], site["splits"], site["charts"]
    held = splits["heldout"]
    before = worst_gradient(gradient_pairs(site, False), np.random.default_rng(site["seed"] + 3), 20)
    after = worst_gradient(gradient_pairs(site, True), np.random.default_rng(site["seed"] + 4), 20)
    worst = max(before[0], after[0])
    site["gradcheck"] = {"tensors_checked": after[1], "tensors_total": after[1], "max_rel_error": worst,
                         "checked_at": ["init", "after_training_steps"],
                         "passed": bool(worst <= limits["gradcheck_rel_error"] and before[1] == after[1])}
    yield "C1", "gradient_check", site["gradcheck"]["passed"], (f"worst relative error {worst:.2e} over the braid, the calibration and both "
                                                                f"screens, before and after training")
    twins = [build_model(2, 2, TASK_TYPES[0], None, kind="braid", split=held, accepted=charts["heldout"]["accepted"]) for _ in range(2)]
    for twin in twins:
        fit(twin, {"train": twin["batch"]}, 25, None)
    same = twins[0]["history"] == twins[1]["history"] and np.array_equal(twins[0]["params"]["X"], twins[1]["params"]["X"])
    finite = bool(np.isfinite(twins[0]["params"]["X"]).all())
    yield "C2", "determinism_finiteness", same and finite, f"two fits agree exactly: {same}; all values finite: {finite}"
    passed, drop, error, trivial = learning_ok(charts["heldout"], held)
    yield "C3", "learning", passed, (f"map objective fell by {drop:.3f} (needs {limits['loss_drop_fraction']}); held-out places off by "
                                     f"{error:.3f} legs against {trivial:.3f} for the anchor centroid (ratio at most "
                                     f"{1.0 - limits['margin_over_trivial']:.2f})")
    mixed = scrambled(held, np.random.default_rng(site["seed"] + 11))
    shuffled = score(chart(mixed, site["screens"]["screen"], site["steps"]["map"])["params"]["X"], mixed, non_anchor(mixed), var=1.0)["error"]
    yield "C4", "shuffled_evidence_control", shuffled >= limits["shuffled_ratio_min"] * trivial, (
        f"with legs and parallels shuffled inside each world places are off by {shuffled:.3f} legs; anchor centroid "
        f"{trivial:.3f} (ratio at least {limits['shuffled_ratio_min']})")
    control = replay(site)
    for name in MUTANTS:
        previous = use_mutant(name)
        site["caught"][name] = not replay(site)
        use_mutant(previous)
    found = sum(site["caught"].values())
    yield "C5", "mutant_detection", control and found == len(MUTANTS), (f"clean replay meets C1 and C3: {control}; "
                                                                        f"registered mutants detected {found} of {len(MUTANTS)}")
    X, accepted, legs = charts["heldout"]["params"]["X"], charts["heldout"]["accepted"], held["legs"]
    backward = dict(held, legs=dict(legs, i=legs["j"], j=legs["i"], b=legs["b"] + math.pi))
    (l0, g0), (l1, g1) = braid_loss(X, held, accepted), braid_loss(X, backward, accepted)
    gap = max(abs(l0 - l1), float(np.abs(g0 - g1).max()))

    def naive(table):
        delta = X[table["w"], table["j"]] - X[table["w"], table["i"]]
        return 0.5 * float((((np.arctan2(delta[:, 0], delta[:, 1]) - table["b"]) / SIGMA["bearing"]) ** 2).sum())
    control_gap = abs(naive(legs) - naive(backward["legs"]))
    yield "C6.1", "route_reversal_invariance", gap <= limits["invariance_tol"] and control_gap >= limits["negative_control_min_violation"], (
        f"objective and gradient change {gap:.1e} when every itinerary is read backwards; naive-angle negative control {control_gap:.1e}")
    H = information(X, held, accepted)[0]
    unit = np.zeros(H.shape[0])
    unit[1] = 1.0 / SIGMA["parallel"]
    base, more, flipped = (np.diag(np.linalg.inv(M)) for M in (H, H + np.outer(unit, unit), H - np.outer(unit, unit)))
    rise, control_rise = float((more - base).max()), float((flipped - base).max())
    yield "C6.2", "information_monotonicity", rise <= limits["invariance_tol"] and control_rise >= limits["negative_control_min_violation"], (
        f"no stated variance grows when a parallel is added (largest change {rise:.1e}); sign-flipped negative control {control_rise:.1e}")
    digests = {name: {hashlib.sha256(w["pos"].tobytes()).hexdigest() for w in split["worlds"]} for name, split in splits.items()}
    disjoint = all(not (digests[a] & digests[b]) for a, b in itertools.combinations(digests, 2))
    longer = splits["shifted"]["route_range"][0] > max(splits["train"]["route_range"][1], held["route_range"][1])
    yield "C7", "split_integrity", disjoint and longer, f"no world shared between splits: {disjoint}; shifted routes all longer: {longer}"


# ---------------------------------------------------------------- hypotheses, report, CLI
def verdicts(runs, seed, evaluated):
    rng, registry = np.random.default_rng(seed + 9973), modules(runs[0]["charts"]["train"])

    def summary(values):
        values = np.asarray(values, dtype=float)
        return paired_bootstrap(values, rng) if evaluated else (float(values.mean()), None)
    hypotheses = []
    for spec in MIND_CARD["hypotheses"]:
        mean, ci = summary([r["row"][spec["id"]] for r in runs])
        hypotheses.append({"id": spec["id"], "metric": spec["metric"], "mean_diff": mean, "ci95": ci, "mesi": spec["mesi"], "n_seeds": len(runs),
                           "verdict": verdict(mean, ci, spec["mesi"], spec["direction"]) if evaluated else "not evaluated"})
    knockouts = []
    for label in runs[0]["lesions"]:
        mean, ci = summary([r["lesions"][label] for r in runs])
        name, mode = label.split(":")
        knockouts.append({"module": name, "mode": mode, "signature": registry[name]["signature"], "metric_change": mean, "ci95": ci})
    return hypotheses, knockouts


def interval(ci):
    return "not evaluated" if ci is None else f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"


def lay_out(sections):
    """Turn (heading, rows) pairs into report lines; a heading of None means the rows stand alone."""
    lines = ["=== VERIFIED REPORT · chapter 0111 ==="]
    for title, rows in sections:
        lines += ([title] if title else []) + [("  " if title else "") + row for row in rows]
    return lines + ["=== END REPORT ==="]


def report_sections(header, results, caught, hypotheses, knockouts, runs, tail):
    def avg(pick):
        return float(np.mean([pick(r) for r in runs]))
    first = runs[0]
    edge = [f"{name:<20} {avg(lambda r, n=name: r['rows'][n]['error']):.4f} / {avg(lambda r, n=name: r['rows'][n]['nlpd']):.4f} / "
            f"{avg(lambda r, n=name: r['rows'][n]['cover']):.3f}" for name in first["rows"]]
    far = [f"{kind:<22} {avg(lambda r, k=kind: r['far'][k][0]):.3f} / {avg(lambda r, k=kind: r['far'][k][1]):.3f}" for kind in first["far"]]
    tally = sum(caught.values())
    return [
        (None, header),
        ("correctness:", [f"{c:<5} {n:<27} {'PASS' if p else 'FAIL'}  {d}" for c, n, p, d in results]),
        (None, [f"mutants: {tally}/{len(MUTANTS)} detected · score {tally / len(MUTANTS):.2f} · "
                + ", ".join(f"{n} {'caught' if caught.get(n) else 'missed'}" for n in MUTANTS)]),
        ("hypotheses (paired over seeds; 95% percentile bootstrap of the mean, 2000 resamples):",
         [f"{h['id']:<8} mean_diff {h['mean_diff']:+.4f} ci95 {interval(h['ci95'])} mesi {h['mesi']} seeds {h['n_seeds']} -> {h['verdict']}"
          for h in hypotheses]),
        ("knockouts (shifted worlds, edge places; change vs the full model in error, or in NLPD for statement):",
         [f"{k['module']:<10} {k['mode']:<8} signature {str(k['signature']):<5} {k['metric_change']:+.4f} ci95 {interval(k['ci95'])}"
          for k in knockouts]),
        ("edge places of shifted worlds (seed mean; error in legs / NLPD / 90% coverage):", edge),
        ("far reports in held-out worlds (seed mean acceptance; true / fabricated):",
         far + ["Eratosthenes rival     accepts every report that carries a measurement, by construction"]),
        (None, [f"held-out interior (seed mean): error {avg(lambda r: r['interior']['error']):.4f} vs anchor centroid "
                f"{avg(lambda r: r['trivial']):.4f} · NLPD {avg(lambda r: r['interior']['nlpd']):.4f} · 90% coverage "
                f"{avg(lambda r: r['interior']['cover']):.3f}",
                f"learned temperature (seed mean): {avg(lambda r: r['tau']):.3f} · correlation of stated s.d. with anchor hops, "
                f"shifted worlds: {avg(lambda r: r['sigma_hops']):.3f}"] + tail),
    ]


APPENDIX_B_KEYS = ("schema_version", "chapter", "file", "card_revision", "environment", "seeds", "runtime_s", "n_params",
                   "gradcheck", "correctness", "mutants", "hypotheses", "knockouts", "task_types", "exit_code")


def protocol(mode, base_seed, n_seeds, json_path, data_path):
    began, seeds = time.time(), list(range(base_seed, base_seed + n_seeds))
    print(f"chapter 0111 · mode {mode} · seeds {seeds} · mutant {ACTIVE_MUTANT}", flush=True)
    runs = []
    for seed in seeds:
        runs.append(run_seed(seed, mode))
        print(f"  seed {seed} done ({time.time() - began:.1f} s)", flush=True)
    site = dict(runs[0], seed=base_seed, steps=STEPS[mode], caught={})
    results = list(correctness(site))
    hypotheses, knockouts = verdicts(runs, base_seed, mode == "full" and n_seeds >= 5)
    bridge = data_bridge(data_path, base_seed, STEPS[mode]["map"]) if data_path else "skipped (no --data PATH given)"
    elapsed = time.time() - began
    results.append(("C8", "budget", elapsed <= TIME_BUDGET[mode], f"{elapsed:.1f} s of {TIME_BUDGET[mode]:.0f} s"))
    failed = [code for code, _, passed, _ in results if not passed]
    exit_code = 0 if not failed else (3 if failed == ["C8"] else 1)
    first, py, npv, g = runs[0], sys.version.split()[0], np.__version__, site["gradcheck"]
    real = int(2 * first["splits"]["train"]["mask"].sum())
    parts = {"training map coordinates": first["charts"]["train"], "calibration": first["calibration"],
             "screen": first["screens"]["screen"], "consistency-only screen": first["screens"]["consistency"]}
    size = sum(n_params(m) for k, m in parts.items() if k != "consistency-only screen")
    header = [f"file: {os.path.basename(__file__)} · card_revision {MIND_CARD['card_revision']} · mode {mode} · mutant {ACTIVE_MUTANT}",
              f"environment: python {py} · numpy {npv}", f"seeds: {seeds} · runtime_s {elapsed:.1f} · budget_s {TIME_BUDGET[mode]:.0f}",
              f"n_params: {size} (" + " · ".join(f"{k} {n_params(m)}" for k, m in parts.items())
              + f"; {real} map coordinates belong to real places, the rest is padding)",
              f"gradcheck: {g['tensors_checked']}/{g['tensors_total']} tensors at init and after training · max_rel_error "
              f"{g['max_rel_error']:.2e} · passed {g['passed']}"]
    tail = [f"real-data bridge: {bridge}", f"task_types: {', '.join(TASK_TYPES)}", f"exit_code: {exit_code}"]
    tally = sum(site["caught"].values())
    values = ("1.0", 111, os.path.basename(__file__), MIND_CARD["card_revision"], {"python": py, "numpy": npv}, seeds, round(elapsed, 2),
              size, g, [dict(zip(("id", "name", "passed", "detail"), (c, n, bool(p), d))) for c, n, p, d in results],
              {"detected": tally, "total": len(MUTANTS), "score": tally / len(MUTANTS)}, hypotheses, knockouts, TASK_TYPES, exit_code)
    lines = lay_out(report_sections(header, results, site["caught"], hypotheses, knockouts, runs, tail))
    write_report(lines, dict(zip(APPENDIX_B_KEYS, values)), json_path)
    return exit_code


OPTIONS = [("--quick", {"action": "store_true", "help": "one seed, reduced updates, all correctness tests"}),
           ("--seed", {"type": int, "default": 0, "help": "base seed (default 0)"}),
           ("--seeds", {"type": int, "default": None, "help": "number of seeds (default 5, or 1 with --quick)"}),
           ("--json", {"metavar": "PATH", "help": "also write the report as JSON"}),
           ("--card", {"action": "store_true", "help": "print MIND_CARD as JSON and exit"}),
           ("--mutant", {"metavar": "NAME", "help": "activate a registered mutant: " + ", ".join(MUTANTS)}),
           ("--data", {"metavar": "PATH", "help": "optional numeric CSV with a header; last column is the target"})]


def main(argv=None):
    parser = argparse.ArgumentParser(prog=os.path.basename(__file__),
                                     description="Chapter 0111, chorographic braid. Without flags the full protocol runs.")
    for flag, options in OPTIONS:
        parser.add_argument(flag, **options)
    args = parser.parse_args(argv)
    if args.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    count = args.seeds if args.seeds is not None else (1 if args.quick else 5)
    complaint = ("--mutant must be one of: " + ", ".join(MUTANTS) if args.mutant not in (None, *MUTANTS)
                 else "--seeds needs a positive count" if count < 1 else "")
    if complaint:
        print(complaint, file=sys.stderr)
        return 2
    use_mutant(args.mutant)
    try:
        return protocol("quick" if args.quick else "full", args.seed, count, args.json, args.data)
    except (FloatingPointError, np.linalg.LinAlgError) as exc:
        print(f"non-finite values: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())

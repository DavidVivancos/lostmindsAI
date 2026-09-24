#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0118 · Heron of Alexandria
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 6, Minds 101-120: https://www.amazon.com/dp/B0HF7G6JJD · Demos: https://artificiology.com/
# END ATTRIBUTION
"""Differentiable winding automaton: Heron's self-moving cart made to wind its own drum.

Thesis
    Behaviour can be laid out in advance along a cord: wound lengths drive, slack
    lengths wait and a change of winding direction reverses, so a whole performance
    is an open-loop program spent by a falling weight.

Evidence and provenance
    Provenance is belief: the entries rest on Heron's own treatises. Loci follow
    F. Grillo's 2019 edition of the Automata. Chapters I.2-I.19 were checked through
    published summaries of that edition and of the Glasgow automata project, not by
    reading the Greek in this session; the proem was read in Schmidt's Greek text.
    D1  Automata I.2-4: a cord wound on the drive axle is pulled by a lead weight
        lowered as millet or mustard seed drains from beneath it.
    D2  Automata I.5-6: forward travel, standstill and backward travel come from
        winding the cord one way, leaving it slack, and winding it the other way.
    D3  Automata I.7-11: changes of direction use separate wheel configurations,
        among them two independently driven axles for the left and right wheels.
    D4  Automata I.4: a fixed script: advance to a mark, stop, fire and libations,
        dancing, turning of the figures, and return to the starting place.
    D5  Automata proem 8: the arrangement fits other arrangements, so a builder can
        lay it out differently without needing anything further.
    D6  Automata I.17-19: ways to stretch the range of a limited drop of the weight.
    D7  The least-distance proof of equal-angle reflection is ascribed to Heron by
        Damianus; the Latin De speculis that preserves it is a late compilation
        (Jones 2001).
    D8  Automata I.2: set the automaton on a flat, even floor where possible.
    D9  Speculation from D1-D3: nothing in the mobile automaton senses or corrects
        its path while it runs.
    D10 Mayr 1970: float regulators run from Ktesibios' water clock and Philon's
        lamp to Heron's float devices.

Doctrine -> mechanism -> test (IDs as in MIND_CARD)
    D2 D3  M1 winding: per-beat softmax over roll, pivot-left, pivot-right and
           slack, times one reversal sign, gives both axle windings  C6.1 H-SIG H-NEC
    D5     M2 recam: linear goal offsets of the winding code            H-SIG
    D1 D6  M3 falling_weight: fuel spent by wound cord, smooth stop gate C6.2
    D10    M4 float_valve: drive pressure pulled to a learned set-point C6.3 H-NEC
    D2 D3  M5 cart: two-axle kinematics with a heading-midpoint step    C6.1
    D4 D7  M6 objective: figure accuracy + excess path + weight budget  C3
    D8 D9  blind-spot floor: biased wheel slip and rising friction      H-BLIND H-RIVAL

Research question (embodied control and sensorimotor adaptation)
    When behaviour is induced by gradient descent as an open-loop winding program
    through a differentiable mechanism, where does it match or beat feedback control
    on new goals, and how quickly does that standing collapse under slip and friction?

Closest prior art and the delta
    Single-shooting trajectory optimisation through differentiable dynamics, goal-
    conditioned movement primitives, turtle-graphics program induction. Delta: the
    trajectory is a winding code (roll, pivot and slack shares with one reversal
    sign), re-cammed linearly by the goal and metered by a falling-weight gate, so it
    reads out as wound, slack and reversed lengths. Baselines in this file: a size-
    matched closed-loop MLP policy trained through the same cart (standard
    alternative) and a three-gain tracking regulator after chapter 0090 (rival).

Blind spot
    A winding cannot correct what the floor does to it: biased wheel slip and rising
    friction accumulate into heading error that no later instruction can undo.

Task (generative process)
    Goal g in [-1, 1]^5 encodes D1 in [0.6, 1.2], k1 in [-0.8, 0.8], theta in
    [-0.9, 0.9] rad, D2 in [0.4, 1.0], k2 in [-0.8, 0.8].
    Script of 24 beats: I advance 8 beats along an arc (D1, k1); II stand 4 beats;
    III pivot 4 beats through theta about one wheel; IV reverse 8 beats along an arc
    (D2, k2). The ideal cart executing the script traces the figure r_1..r_24.
    Splits: train 64, validation 32, held-out 64, shifted 64. Shifted goals lie in
    the corner where k1, theta and k2 all exceed 0.4 (normalized); no other split
    visits it. Floors: nominal; nuisance (slip s.d. 0.03, millet-flow s.d. 0.03,
    used for training); blind (slip s.d. 0.06, left-wheel slip -0.06, friction
    rising from 0.05 to 0.20).

Limits
    Kinematic cart without inertia or contact mechanics; one script family; synthetic
    goals; perfect pose sensing for the feedback controllers; 400 updates per model.
    A research prototype of one AGI-relevant mechanism: not an AGI, and not a claim
    to reproduce Heron's mind.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 1,
    "revision_log": [],
    "generation": {"template_version": "codeguidelines 1.0 (15 September 2026), Appendix A",
                   "generator": "Claude (Anthropic)", "generator_version": "claude-opus-5", "date": "2026-09-15"},
    "id": 118, "figure": "Heron of Alexandria", "born": 10, "died": 70, "civilization": "Greek",
    "provenance": "belief",
    "thesis": ("Behaviour can be laid out in advance along a cord: wound lengths drive, slack lengths wait and a "
               "change of winding direction reverses, so a whole performance is an open-loop program spent by a "
               "falling weight."),
    "evidence": [
        {"id": "D1", "claim": "The mobile automaton's drive cord is wound on the axle and pulled by a lead weight "
         "lowered as millet or mustard seed drains from beneath it.", "basis": "primary",
         "source": "Heron, Automata I.2-4 (ed. Grillo 2019)"},
        {"id": "D2", "claim": "Forward travel, standstill and backward travel come from winding the cord one way, "
         "leaving it slack, and winding it the other way.", "basis": "primary",
         "source": "Heron, Automata I.5-6 (ed. Grillo 2019)"},
        {"id": "D3", "claim": "Changes of direction use separate wheel configurations, among them two independently "
         "driven axles for the left and right wheels.", "basis": "primary",
         "source": "Heron, Automata I.7-11 (ed. Grillo 2019)"},
        {"id": "D4", "claim": "The performance is a fixed script: advance to a mark, stop, altar fire and libations, "
         "dancing, turning of the figures, and return to the starting place.", "basis": "primary",
         "source": "Heron, Automata I.4 (ed. Grillo 2019)"},
        {"id": "D5", "claim": "Heron offers his arrangement as one that fits other arrangements, so a builder can lay "
         "it out differently without needing anything further.", "basis": "primary",
         "source": "Heron, Automata proem 8 (ed. Schmidt 1899)"},
        {"id": "D6", "claim": "Heron gives ways to stretch the range available from a limited drop of the weight: "
         "larger wheels or thinner axles, a cord led from a small drum to a larger one, a second counterweight.",
         "basis": "primary", "source": "Heron, Automata I.17-19 (ed. Grillo 2019)"},
        {"id": "D7", "claim": "A proof that rays reflected at equal angles take the shortest path is ascribed to Heron "
         "by Damianus; the Latin De speculis that preserves it is a late compilation.", "basis": "scholarship",
         "source": "Jones 2001, Pseudo-Ptolemy De Speculis, SCIAMVS 2"},
        {"id": "D8", "claim": "Heron advises setting the automaton on a flat, even surface where possible.",
         "basis": "primary", "source": "Heron, Automata I.2 (ed. Grillo 2019)"},
        {"id": "D9", "claim": "Nothing in the mobile automaton senses or corrects its path during the run, so "
         "disturbances are never compensated.", "basis": "speculation", "source": "inference from Automata I.2-11"},
        {"id": "D10", "claim": "Float regulation belongs to an Alexandrian line running from Ktesibios' water clock "
         "and Philon's lamp to Heron's float devices.", "basis": "scholarship",
         "source": "Mayr 1970, The Origins of Feedback Control"},
    ],
    "research_question": {
        "category": "embodied control and sensorimotor adaptation",
        "question": ("When behaviour is induced by gradient descent as an open-loop winding program through a "
                     "differentiable mechanism, where does it match or beat feedback control on new goals, and how "
                     "quickly does that standing collapse when slip and friction push back?")},
    "mechanism": {
        "name": "differentiable winding automaton (a machine that winds its own drum)",
        "family": "program induction; open-loop trajectory optimisation through differentiable dynamics",
        "signature_modules": ["winding"],
        "closest_prior_art": [
            "single-shooting trajectory optimisation through differentiable dynamics (Bryson and Ho 1969)",
            "goal-conditioned dynamic movement primitives (Ijspeert et al. 2013)",
            "turtle-graphics program induction (Ellis et al. 2021, DreamCoder)"],
        "overlap": "Medium",
        "prior_art_queries": [],
        "prior_art_note": "No literature search was run for this card; overlap is rated against the named methods.",
        "contribution_type": "mechanism",
        "delta": ("The control sequence is a winding code (roll, pivot and slack shares with one reversal sign), "
                  "re-cammed linearly by the goal and metered by a falling-weight gate, so the learned trajectory "
                  "reads out as wound, slack and reversed lengths; it is tested against closed-loop control."),
        "baselines": {
            "baseline": ("size-matched closed-loop MLP policy (goal, sensed pose and beat phase in; 36 tanh units) "
                         "trained through the same cart, weight and valve"),
            "rival": "three-gain proportional regulator toward the commanded figure's next point, after chapter 0090"}},
    "traceability": [
        {"doctrine": "D2", "mechanism": "M1 winding", "property_test": "C6.1", "hypothesis": "H-SIG, H-NEC"},
        {"doctrine": "D3", "mechanism": "M1 winding, M5 cart", "property_test": "C6.1", "hypothesis": "H-SIG"},
        {"doctrine": "D5", "mechanism": "M2 recam", "property_test": "none", "hypothesis": "H-SIG"},
        {"doctrine": "D1", "mechanism": "M3 falling_weight", "property_test": "C6.2", "hypothesis": "knockout table"},
        {"doctrine": "D6", "mechanism": "M3 falling_weight", "property_test": "C6.2", "hypothesis": "knockout table"},
        {"doctrine": "D4", "mechanism": "M6 objective (script figures)", "property_test": "C7", "hypothesis": "none (C3)"},
        {"doctrine": "D7", "mechanism": "M6 objective (excess-path term)", "property_test": "none", "hypothesis": "none"},
        {"doctrine": "D10", "mechanism": "M4 float_valve; rival regulator", "property_test": "C6.3 (definition check)",
         "hypothesis": "H-NEC (matched knockout), H-RIVAL"},
        {"doctrine": "D8", "mechanism": "blind-spot floor", "property_test": "none", "hypothesis": "H-BLIND, H-RIVAL"},
        {"doctrine": "D9", "mechanism": "blind-spot floor", "property_test": "none", "hypothesis": "H-BLIND, H-RIVAL"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": ("Re-cammed to goal combinations excluded from training, the learned winding "
                                      "traces the figure more accurately than a size-matched closed-loop policy "
                                      "trained through the same cart."),
         "metric": "figure_error", "split": "shifted", "condition": "nominal",
         "comparison": "model - baseline", "direction": "less", "mesi": 0.02, "seeds": 5},
        {"id": "H-NEC", "statement": ("Replacing the winding by its mean instruction degrades the figure more than "
                                      "removing the float valve's regulation."),
         "metric": "figure_error", "split": "heldout", "condition": "nuisance",
         "comparison": "signature_knockout - matched_knockout", "knockouts": ["winding:mean", "float_valve:identity"],
         "direction": "greater", "mesi": 0.05, "seeds": 5},
        {"id": "H-BLIND", "statement": ("Under biased wheel slip and rising friction the open-loop winding traces the "
                                        "figure less accurately than the closed-loop policy."),
         "condition": "blind", "grounding": ("Heron's cart executes a fixed winding with no sensing; the only remedy he "
                                             "records is a flat, even floor (Automata I.2)."),
         "metric": "figure_error", "split": "heldout", "comparison": "model - baseline", "direction": "greater",
         "mesi": 0.02, "seeds": 5},
        {"id": "H-RIVAL", "statement": ("The winding's standing against a Ctesibian tracking regulator falls when slip "
                                        "and friction are added to the floor."),
         "metric": "figure_error difference-in-differences", "split": "heldout",
         "comparison": "(model - rival | blind) - (model - rival | nominal)", "direction": "greater",
         "mesi": 0.02, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.5, "margin_over_trivial": 0.3, "shuffled_ratio_min": 0.8,
                   "gradcheck_rel_error": 1e-5, "gradcheck_floor": 1e-3, "property_tol": 1e-9,
                   "negative_control_min_violation": 1e-6},
    "metrics": {"figure_error": "RMS distance between cart and commanded figure over the 24 beats, mean over goals",
                "trivial_baseline": "the mean training figure, whatever the goal",
                "shuffled_band": "one-sided: a leak would show as held-out error well below the trivial error"},
    "training": {"optimizer": "Adam", "lr_grid": [0.02], "updates": {"full": 400, "quick": 120}, "clip_norm": 5.0,
                 "batch": "all 64 training goals, fresh nuisance noise every update",
                 "model_selection": "validation figure_error every 25 updates from update 50",
                 "applies_to": ["winding", "policy", "regulator"]},
    "task": {"beats": 24, "splits": {"train": 64, "val": 32, "heldout": 64, "shifted": 64},
             "conditions": {"nominal": "no slip, no friction, steady millet flow",
                            "nuisance": "slip s.d. 0.03, millet-flow s.d. 0.03",
                            "blind": "slip s.d. 0.06, left-wheel slip -0.06, friction 0.05 rising to 0.20, flow s.d. 0.03"}},
    "probe_predictions": [{"probe": "P11", "expected": "below baseline"}],
    "probe_support": "episodic_control only; probes P1-P10 are not supported",
    "dialectic_links": [{"chapter": 90, "relation": "rival", "test": "H-RIVAL",
                         "note": ("Heron works in the Alexandrian line of float regulation that Mayr traces from "
                                  "Ktesibios; no citation of Ktesibios in the Pneumatica or Automata was verified.")}],
    "corpus_neighbors": [
        {"chapter": 90, "similarity": None, "difference": ("0090 regulates by a constant head; here regulation is the "
                                                           "rival and the float valve a non-signature part, while the "
                                                           "signature is an open-loop stored winding.")},
        {"chapter": 89, "similarity": None, "difference": ("0089 proposes by mechanical heuristic and certifies by "
                                                           "proof; here nothing is bracketed or proved: a program is "
                                                           "induced through the mechanism and judged by its figure.")},
        {"chapter": 83, "similarity": None, "difference": ("0083 builds objects from a frugal operator set; here the "
                                                           "primitives are timed motion shares spent against a fuel "
                                                           "budget and generality comes from goal re-camming.")},
        {"chapter": 219, "similarity": None, "difference": ("0219 owns pinned-barrel event schedules; here the program "
                                                            "is a continuous winding encoding a trajectory through "
                                                            "slack and reversal, with no event pins.")},
        {"chapter": 301, "similarity": None, "difference": ("0301 composes a standard part library; here one mechanism "
                                                            "is fixed and only its winding is learned.")},
    ],
    "similarity_note": "Nearest-neighbour similarity not computed: corpus files were not available to this session.",
    "barometer": {
        "cognitive_processing": ["transfer of the winding to excluded goal combinations (shifted split)"],
        "embodied_cognition": ["two-axle cart control", "degradation when slip and friction change the dynamics"],
        "world_modeling": [], "consciousness": [], "language_understanding": [], "emotional_intelligence": [],
        "creativity": [], "autonomy": ["completing the performance within a fixed weight budget"]},
    "task_types": ["episodic_control"],
    "applications": [
        {"use": ("stroke programs for plotters and drawing machines learned from demonstrations, with pen-up spans "
                 "as slack and direction changes as reversals"),
         "sector": "creative tools and education", "dataset": "Quick, Draw! stroke data (Google Creative Lab)",
         "readiness": "low"},
        {"use": ("goal-conditioned manoeuvre programs for differential-drive robots, audited against feedback "
                 "control under wheel slip before deployment"),
         "sector": "logistics and service robotics",
         "dataset": "LASA Handwriting Dataset (2-D demonstrations; Khansari-Zadeh and Billard 2011)",
         "readiness": "low"},
        {"use": "tool-path programs under actuator energy budgets for CNC machining", "sector": "manufacturing",
         "dataset": "CNC Mill Tool Wear (University of Michigan SMART lab, 2018, Kaggle)", "readiness": "low"},
    ],
    "safety_notes": ("Heron's artillery treatise and every war machine are excluded from the evidence, the mechanism "
                     "and the applications, which are civilian. The file does not claim to replicate Heron's mind "
                     "and puts no generated words in his mouth."),
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

T_BEATS, G_DIM, HIDDEN = 24, 5, 36
CORD, TRACK, FUEL0, RESERVE = 0.25, 0.30, 6.0, 0.9   # cord per beat, axle track, weight drop, usable share
GATE_BETA, GATE_MARGIN, SABS_EPS, HINGE_K = 40.0, 0.04, 1e-3, 20.0
LAMBDA_PATH, LAMBDA_BUDGET = 0.05, 0.5
LR, CLIP_NORM, VAL_EVERY, VAL_START = 0.02, 5.0, 25, 50
STEPS = {"full": 400, "quick": 120}
TIME_BUDGET = {"full": 180.0, "quick": 20.0}
SPLIT_SIZES = {"train": 64, "val": 32, "heldout": 64, "shifted": 64}
GOAL_LO = np.array([0.6, -0.8, -0.9, 0.4, -0.8])
GOAL_HI = np.array([1.2, 0.8, 0.9, 1.0, 0.8])
CORNER_DIMS, CORNER_EDGE = [1, 2, 4], 0.4
CONDITIONS = {
    "nominal": {"slip": 0.0, "bias": 0.0, "fric0": 0.0, "fric1": 0.0, "flow": 0.0},
    "nuisance": {"slip": 0.03, "bias": 0.0, "fric0": 0.0, "fric1": 0.0, "flow": 0.03},
    "blind": {"slip": 0.06, "bias": -0.06, "fric0": 0.05, "fric1": 0.20, "flow": 0.03},
}
# Rows are the cart's primitives (roll, pivot-left, pivot-right, slack); columns say which axle's cord is wound.
PRIMITIVE_AXLES = np.array([[1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
PHASE = np.stack([f(2.0 * np.pi * j * np.arange(T_BEATS) / T_BEATS) for j in (1, 2, 3, 4) for f in (np.sin, np.cos)],
                 axis=1)
TASK_TYPES = ["episodic_control"]
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


# ---------------------------------------------------------------- data and tasks
def in_corner(goals):
    return (goals[:, CORNER_DIMS] > CORNER_EDGE).all(axis=1)


def sample_goals(rng, n, corner):
    """Uniform normalized goals, either inside the excluded corner or outside it."""
    kept, total = [], 0
    while total < n:
        g = rng.uniform(-1.0, 1.0, (4 * n, G_DIM))
        if corner:
            g[:, CORNER_DIMS] = rng.uniform(CORNER_EDGE, 1.0, (4 * n, len(CORNER_DIMS)))
        g = g[in_corner(g) == corner]
        kept.append(g)
        total += len(g)
    return np.concatenate(kept)[:n]


def script_commands(goals):
    """Heron's script as wound shares per axle: arc, stand, pivot about one wheel, reversed arc."""
    raw = GOAL_LO + (goals + 1.0) * 0.5 * (GOAL_HI - GOAL_LO)
    d1, k1, theta, d2, k2 = (raw[:, i] for i in range(G_DIM))
    u = np.zeros((goals.shape[0], T_BEATS, 2))
    for beats, dist, curv, sign in ((slice(0, 8), d1 / 8.0, k1, 1.0), (slice(16, 24), d2 / 8.0, k2, -1.0)):
        u[:, beats, 0] = (sign * dist * (1.0 - curv * TRACK / 2.0) / CORD)[:, None]
        u[:, beats, 1] = (sign * dist * (1.0 + curv * TRACK / 2.0) / CORD)[:, None]
    pivot = np.abs(theta) * TRACK / (4.0 * CORD)
    u[:, 12:16, 0] = np.where(theta < 0.0, pivot, 0.0)[:, None]
    u[:, 12:16, 1] = np.where(theta > 0.0, pivot, 0.0)[:, None]
    return u


def nominal_noise(n):
    return {"slip": np.ones((n, T_BEATS, 2)), "friction": np.zeros(T_BEATS), "flow": np.zeros((n, T_BEATS))}


def disturbance(rng, n, condition):
    c = CONDITIONS[condition]
    slip = 1.0 + c["slip"] * rng.standard_normal((n, T_BEATS, 2))
    slip[:, :, 0] += c["bias"]
    return {"slip": slip, "friction": np.linspace(c["fric0"], c["fric1"], T_BEATS),
            "flow": c["flow"] * rng.standard_normal((n, T_BEATS))}


def batch_for(split, condition, rng):
    n = split["goals"].shape[0]
    noise = nominal_noise(n) if condition == "nominal" else disturbance(rng, n, condition)
    return dict(split, start=np.zeros((n, 3)), **noise)


def reference(goals):
    """The ideal cart (full weight, set-point pressure, true floor) executing the script."""
    n = goals.shape[0]
    batch = dict(goals=goals, commands=script_commands(goals), ref=np.zeros((n, T_BEATS + 1, 2)),
                 path_ref=np.zeros(n), start=np.zeros((n, 3)), **nominal_noise(n))
    trace = run(SCRIPT, batch)[1]
    return {"goals": goals, "commands": batch["commands"], "ref": np.stack(trace["xy"], axis=1),
            "ref_heading": np.stack(trace["heading"], axis=1), "path_ref": trace["path"]}


def make_data(rng):
    return {name: reference(sample_goals(rng, size, name == "shifted")) for name, size in SPLIT_SIZES.items()}


def trivial_error(data):
    gap = data["heldout"]["ref"][:, 1:] - data["train"]["ref"].mean(axis=0)[None, 1:]
    return float(np.sqrt((gap ** 2).sum(-1).mean(1)).mean())


def env_reset(rng):
    """Start a performance: a training-region goal, its figure and a full weight."""
    goal = sample_goals(rng, 1, False)
    return {"goal": goal[0], "ref": reference(goal)["ref"][0], "pose": np.zeros(3), "fuel": FUEL0, "beat": 0}


def env_step(state, action, rng):
    """One beat of the ideal cart for action = (left, right) signed wound shares.
    The ideal floor is deterministic, so rng is accepted for the interface and unused."""
    s, u = dict(state), np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
    drive = (math.sqrt(1.0 + SABS_EPS ** 2) - SABS_EPS) * float(sigmoid(GATE_BETA * (s["fuel"] / FUEL0 - GATE_MARGIN)))
    left, right = drive * CORD * u
    s["fuel"] -= drive * CORD * float(np.sum(np.sqrt(u * u + SABS_EPS ** 2) - SABS_EPS))
    v, w = 0.5 * (left + right), (right - left) / TRACK
    x, y, h = s["pose"]
    mid = h + 0.5 * w
    s["pose"], s["beat"] = np.array([x + v * math.cos(mid), y + v * math.sin(mid), h + w]), s["beat"] + 1
    return s, -float(np.sum((s["pose"][:2] - s["ref"][s["beat"]]) ** 2)), s["beat"] >= T_BEATS


# ---------------------------------------------------------------- model
class Node:
    __slots__ = ("v", "g", "ps", "bw", "req")

    def __init__(self, v, ps=(), bw=None, req=False):
        self.v, self.g, self.ps, self.bw, self.req = v, None, ps, bw, req


def _unbroadcast(g, shape):
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and g.shape[axis] != 1:
            g = g.sum(axis=axis, keepdims=True)
    return g


class Tape:
    """Reverse-mode tape: each op records a closure that maps its gradient onto its parents."""

    def __init__(self, grad=True):
        self.grad, self.nodes = grad, []

    def param(self, value):
        return Node(value, req=self.grad)

    def w(self, x):
        return x if isinstance(x, Node) else Node(np.asarray(x, dtype=float))

    def op(self, value, parents, bw):
        req = self.grad and any(p.req for p in parents)
        node = Node(value, parents, bw, req)
        if req:
            self.nodes.append(node)
        return node

    def add(self, a, b):
        a, b = self.w(a), self.w(b)
        return self.op(a.v + b.v, (a, b), lambda g: (g, g))

    def sub(self, a, b):
        a, b = self.w(a), self.w(b)
        return self.op(a.v - b.v, (a, b), lambda g: (g, -g))

    def mul(self, a, b):
        a, b = self.w(a), self.w(b)
        return self.op(a.v * b.v, (a, b), lambda g: (g * b.v, g * a.v))

    def unary(self, a, f, df):
        a = self.w(a)
        y = f(a.v)
        return self.op(y, (a,), lambda g: (g * df(a.v, y),))

    def tanh(self, a):
        return self.unary(a, np.tanh, lambda x, y: 1.0 - y * y)

    def sin(self, a):
        return self.unary(a, np.sin, lambda x, y: np.cos(x))

    def cos(self, a):
        return self.unary(a, np.cos, lambda x, y: -np.sin(x))

    def sigmoid(self, a):
        return self.unary(a, sigmoid, lambda x, y: y * (1.0 - y))

    def softplus(self, a):
        return self.unary(a, softplus, lambda x, y: sigmoid(x))

    def sabs(self, a):
        # smooth |x| keeps effort and path length differentiable at a standstill
        return self.unary(a, lambda x: np.sqrt(x * x + SABS_EPS ** 2) - SABS_EPS, lambda x, y: x / (y + SABS_EPS))

    def gate(self, fuel):
        return self.unary(fuel, lambda f: sigmoid(GATE_BETA * (f / FUEL0 - GATE_MARGIN)),
                          lambda f, y: GATE_BETA / FUEL0 * y * (1.0 - y))

    def softmax(self, a):
        a = self.w(a)
        y = softmax(a.v)
        return self.op(y, (a,), lambda g: (y * (g - (g * y).sum(axis=-1, keepdims=True)),))

    def linear(self, a, w):
        a, w = self.w(a), self.w(w)

        def bw(g):
            dw = g.reshape(-1, g.shape[-1]).T @ a.v.reshape(-1, a.v.shape[-1]) if w.req else None
            return g @ w.v, dw
        return self.op(a.v @ w.v.T, (a, w), bw)

    def take(self, a, key):
        a = self.w(a)

        def bw(g):
            full = np.zeros_like(a.v)
            full[key] = g
            return (full,)
        return self.op(a.v[key], (a,), bw)

    def columns(self, cols):
        cols = [self.w(c) for c in cols]
        mats = [c.v.reshape(c.v.shape[0], -1) for c in cols]
        edges = np.cumsum([0] + [m.shape[1] for m in mats])
        return self.op(np.concatenate(mats, axis=1), tuple(cols),
                       lambda g: tuple(g[:, edges[i]:edges[i + 1]].reshape(c.v.shape) for i, c in enumerate(cols)))

    def expand_last(self, a):
        a = self.w(a)
        return self.op(a.v[..., None], (a,), lambda g: (g[..., 0],))

    def goal_mod(self, m, goals):
        m = self.w(m)
        return self.op(np.einsum("...g,ng->n...", m.v, goals), (m,),
                       lambda g: (np.einsum("n...,ng->...g", g, goals),))

    def mean(self, a):
        a = self.w(a)
        return self.op(np.asarray(a.v.mean()), (a,), lambda g: (np.full(a.v.shape, float(g) / a.v.size),))

    def backward(self, out):
        out.g = np.ones_like(out.v)
        for node in reversed(self.nodes):
            if node.g is None:
                continue
            for parent, grad in zip(node.ps, node.bw(node.g)):
                if parent.req:
                    grad = _unbroadcast(grad, parent.v.shape)
                    parent.g = grad if parent.g is None else parent.g + grad


def build_model(in_dim, out_dim, task_type, rng, **cfg):
    """Winding automaton by default; kind='policy' or 'regulator' builds the comparators, 'script' the ideal cart."""
    kind = cfg.get("kind", "winding")
    if task_type not in TASK_TYPES or in_dim != G_DIM or out_dim != 2:
        raise ValueError("chapter 0118 supports episodic_control with a 5-d goal and 2 axle commands")
    params = {"valve_setpoint": np.array([math.log(math.e - 1.0)]), "valve_gain": np.zeros(1)}
    if kind == "winding":
        params.update(winding_logits=rng.normal(0.0, 0.5, (T_BEATS, 4)), winding_sign=rng.normal(0.0, 0.5, T_BEATS),
                      recam_logits=rng.normal(0.0, 0.05, (T_BEATS, 4, G_DIM)),
                      recam_sign=rng.normal(0.0, 0.05, (T_BEATS, G_DIM)))
    elif kind == "policy":
        fan_in, hidden = G_DIM + 4 + PHASE.shape[1], cfg.get("hidden", HIDDEN)
        params.update(policy_w1=rng.normal(0.0, fan_in ** -0.5, (hidden, fan_in)), policy_b1=np.zeros(hidden),
                      policy_w2=rng.normal(0.0, 0.1, (2, hidden)), policy_b2=np.zeros(2))
    elif kind == "regulator":
        params.update(gain_forward=np.zeros(1), gain_lateral=np.zeros(1), gain_heading=np.zeros(1))
    elif kind != "script":
        raise ValueError(f"unknown kind {kind}")
    return {"kind": kind, "params": params, "ko": {}, "cfg": {"lr": cfg.get("lr", LR)}}


def decode_winding(tp, P, goals, ko):
    """Winding code -> signed axle windings (N, beats, 2); also returns primitive shares and reversal sign."""
    n = goals.shape[0]
    logits = tp.add(P["winding_logits"], np.zeros((n, T_BEATS, 4)))
    sign = tp.add(P["winding_sign"], np.zeros((n, T_BEATS)))
    if ko.get("recam") != "zero":
        logits = tp.add(logits, tp.goal_mod(P["recam_logits"], goals))
        sign = tp.add(sign, tp.goal_mod(P["recam_sign"], goals))
    shares = tp.softmax(logits)
    keep = 0.0 if ACTIVE_MUTANT == "reversal_grad_dropped" else 1.0
    reversal = tp.unary(sign, np.tanh, lambda s, r: keep * (1.0 - r * r))
    commands = tp.mul(tp.expand_last(reversal), tp.linear(shares, PRIMITIVE_AXLES.T))
    if ko.get("winding") == "mean":
        commands = tp.w(np.broadcast_to(commands.v.mean(axis=(0, 1), keepdims=True), commands.v.shape).copy())
    elif ko.get("winding") == "zero":
        commands = tp.w(np.zeros_like(commands.v))
    return commands, shares.v, reversal.v


def valve_update(tp, pressure, kappa, setpoint, flow):
    """Float valve: move the drive pressure a fraction kappa of the way to its set-point, then add the flow jitter."""
    return tp.add(tp.add(pressure, tp.mul(kappa, tp.sub(setpoint, pressure))), flow)


def rollout(tp, model, P, batch):
    """Drive the cart through the performance; returns the three-part loss node and a numpy trace."""
    kind, ko, n = model["kind"], model["ko"], batch["goals"].shape[0]
    x, y, h = (tp.w(batch["start"][:, i]) for i in range(3))
    fuel, pressure = tp.w(np.full(n, FUEL0)), tp.w(np.ones(n))
    kappa, setpoint = tp.sigmoid(P["valve_gain"]), tp.softplus(P["valve_setpoint"])
    path, acc = tp.w(np.zeros(n)), tp.w(np.zeros(n))
    commands = decode_winding(tp, P, batch["goals"], ko)[0] if kind == "winding" else None
    trace = {"xy": [batch["start"][:, :2].copy()], "heading": [batch["start"][:, 2].copy()], "fuel": [fuel.v],
             "gate": [], "pressure": [pressure.v]}
    for t in range(T_BEATS):
        if kind == "winding":
            left_u, right_u = tp.take(commands, (slice(None), t, 0)), tp.take(commands, (slice(None), t, 1))
        elif kind == "script":
            left_u, right_u = tp.w(batch["commands"][:, t, 0]), tp.w(batch["commands"][:, t, 1])
        elif kind == "policy":
            left_u, right_u = policy_command(tp, P, batch["goals"], t, x, y, h)
        else:
            left_u, right_u = regulator_command(tp, P, batch, t, x, y, h)
        # the negative control of C6.2 drives with raw pressure, which may turn negative and refill the weight
        drive_p = pressure if batch.get("raw_pressure") else tp.sabs(pressure)
        gate = tp.w(np.ones(n)) if ko.get("falling_weight") else tp.gate(fuel)
        drive = tp.mul(tp.mul(drive_p, gate), CORD * (1.0 - batch["friction"][t]))
        left = tp.mul(drive, tp.mul(left_u, batch["slip"][:, t, 0]))
        right = tp.mul(drive, tp.mul(right_u, batch["slip"][:, t, 1]))
        if not ko.get("falling_weight"):
            fuel = tp.sub(fuel, tp.mul(tp.mul(tp.add(tp.sabs(left_u), tp.sabs(right_u)), tp.mul(drive_p, gate)), CORD))
        v, w = tp.mul(tp.add(left, right), 0.5), tp.mul(tp.sub(right, left), 1.0 / TRACK)
        # heading at mid-beat makes the step exactly reversible (C6.1); Euler is the negative control
        heading = h if batch.get("euler") else tp.add(h, tp.mul(w, 0.5))
        x, y, h = tp.add(x, tp.mul(v, tp.cos(heading))), tp.add(y, tp.mul(v, tp.sin(heading))), tp.add(h, w)
        path = tp.add(path, tp.sabs(v))
        flow = batch["flow"][:, t]
        pressure = tp.add(pressure, flow) if ko.get("float_valve") else valve_update(tp, pressure, kappa, setpoint, flow)
        dx, dy = tp.sub(x, batch["ref"][:, t + 1, 0]), tp.sub(y, batch["ref"][:, t + 1, 1])
        err = tp.add(tp.mul(dx, dx), tp.mul(dy, dy))
        acc = tp.add(acc, err)
        trace["xy"].append(np.stack([x.v, y.v], axis=1))
        trace["heading"].append(h.v)
        trace["fuel"].append(fuel.v)
        trace["gate"].append(gate.v)
        trace["pressure"].append(pressure.v)
    trace["path"] = path.v
    excess = tp.mul(tp.softplus(tp.mul(tp.sub(path, batch["path_ref"]), HINGE_K)), 1.0 / HINGE_K)
    spent = tp.mul(tp.sub(FUEL0, fuel), 1.0 / FUEL0)
    over = tp.mul(tp.softplus(tp.mul(tp.sub(spent, RESERVE), HINGE_K)), 1.0 / HINGE_K)
    accuracy = tp.mean(tp.add(tp.mul(acc, 1.0 / T_BEATS), err))
    loss = tp.add(accuracy, tp.add(tp.mul(tp.mean(excess), LAMBDA_PATH), tp.mul(tp.mean(over), LAMBDA_BUDGET)))
    return loss, trace


def run(model, batch):
    tp = Tape(grad=False)
    loss, trace = rollout(tp, model, {k: tp.param(v) for k, v in model["params"].items()}, batch)
    return float(loss.v), trace


def loss_and_grads(model, batch):
    tp = Tape(grad=True)
    P = {k: tp.param(v) for k, v in model["params"].items()}
    loss = rollout(tp, model, P, batch)[0]
    tp.backward(loss)
    grads = {k: np.zeros_like(v) if P[k].g is None else P[k].g for k, v in model["params"].items()}
    if ACTIVE_MUTANT == "zero_grad_recam_logits" and "recam_logits" in grads:
        grads["recam_logits"] = np.zeros_like(grads["recam_logits"])
    return float(loss.v), grads


def figure_error(model, batch):
    xy = np.stack(run(model, batch)[1]["xy"], axis=1)
    return float(np.sqrt(((xy[:, 1:] - batch["ref"][:, 1:]) ** 2).sum(-1).mean(1)).mean())


SCRIPT = build_model(G_DIM, 2, "episodic_control", None, kind="script")


# ---------------------------------------------------------------- baselines and rival mechanisms
def policy_command(tp, P, goals, t, x, y, h):
    """Size-matched closed-loop baseline: an MLP reads the goal, the sensed pose and the beat phase."""
    phase = np.repeat(PHASE[t][None], goals.shape[0], axis=0)
    feats = tp.columns([goals, x, y, tp.cos(h), tp.sin(h), phase])
    hidden = tp.tanh(tp.add(tp.linear(feats, P["policy_w1"]), P["policy_b1"]))
    out = tp.tanh(tp.add(tp.linear(hidden, P["policy_w2"]), P["policy_b2"]))
    return tp.take(out, (slice(None), 0)), tp.take(out, (slice(None), 1))


def regulator_command(tp, P, batch, t, x, y, h):
    """Rival after chapter 0090: proportional regulation toward the figure's next point and heading."""
    dx, dy = tp.sub(batch["ref"][:, t + 1, 0], x), tp.sub(batch["ref"][:, t + 1, 1], y)
    ch, sh = tp.cos(h), tp.sin(h)
    along, across = tp.add(tp.mul(ch, dx), tp.mul(sh, dy)), tp.sub(tp.mul(ch, dy), tp.mul(sh, dx))
    align = tp.sin(tp.sub(batch["ref_heading"][:, t + 1], h))
    advance = tp.mul(tp.softplus(P["gain_forward"]), along)
    turn = tp.add(tp.mul(tp.softplus(P["gain_lateral"]), across), tp.mul(tp.softplus(P["gain_heading"]), align))
    half = tp.mul(turn, 0.5 * TRACK)
    return tp.tanh(tp.mul(tp.sub(advance, half), 1.0 / CORD)), tp.tanh(tp.mul(tp.add(advance, half), 1.0 / CORD))


# ---------------------------------------------------------------- registries
KNOCKOUT_MODES = {"winding": ("mean", "zero"), "recam": ("zero",), "falling_weight": ("identity",),
                  "float_valve": ("identity",)}
KNOCKOUT_PLAN = (("winding", "mean"), ("recam", "zero"), ("falling_weight", "identity"), ("float_valve", "identity"))
MUTANTS = {
    "sign_flipped_update": "optimizer climbs the loss gradient instead of descending it (C3 must fail)",
    "zero_learning_rate": "learning rate forced to zero (C3 must fail)",
    "zero_grad_recam_logits": "gradient of recam_logits zeroed (C1 must fail)",
    "reversal_grad_dropped": "backward pass through the reversal sign dropped (C1 must fail)",
}


def modules(model):
    table = {"falling_weight": ([], "smooth stop gate on cumulative actuator effort (fuel budget)", False),
             "float_valve": (["valve_setpoint", "valve_gain"],
                             "first-order regulator pulling the drive gain toward a learned set-point", False)}
    extra = {"winding": {"winding": (["winding_logits", "winding_sign"], "per-step softmax over motion primitives "
                                     "times a tanh direction sign; open-loop control code", True),
                         "recam": (["recam_logits", "recam_sign"], "linear goal-conditioned offsets of the control "
                                   "code", False)},
             "policy": {"policy": (["policy_w1", "policy_b1", "policy_w2", "policy_b2"],
                                   "state-feedback MLP controller", False)},
             "regulator": {"regulator": (["gain_forward", "gain_lateral", "gain_heading"],
                                         "proportional pose-tracking controller", False)}}
    table.update(extra.get(model["kind"], {}))
    return {name: {"params": p, "role": role, "signature": sig} for name, (p, role, sig) in table.items()}


def knockout(model, name, mode):
    """Copy with one module replaced: winding by its mean or zero output, recam by zero, weight and valve by identity."""
    if name not in modules(model) or mode not in KNOCKOUT_MODES.get(name, ()):
        raise ValueError(f"no knockout {name}:{mode} for a {model['kind']} model")
    return {"kind": model["kind"], "params": {k: v.copy() for k, v in model["params"].items()},
            "ko": dict(model["ko"], **{name: mode}), "cfg": dict(model["cfg"])}


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


# ---------------------------------------------------------------- training
def fit(model, data, budget, rng):
    """Adam on the three-part objective; the checkpoint with the best validation figure is kept."""
    params, state = model["params"], adam_init(model["params"])
    lr = 0.0 if ACTIVE_MUTANT == "zero_learning_rate" else model["cfg"]["lr"]
    history, best, best_error = [], None, np.inf
    for step in range(1, budget + 1):
        loss, grads = loss_and_grads(model, batch_for(data["train"], "nuisance", rng))
        if not (math.isfinite(loss) and all(np.isfinite(g).all() for g in grads.values())):
            raise FloatingPointError(f"non-finite loss or gradient at update {step}")
        grads = clip_global(grads, CLIP_NORM)[0]
        if ACTIVE_MUTANT == "sign_flipped_update":
            grads = {k: -g for k, g in grads.items()}
        adam_step(params, grads, state, lr)
        history.append(loss)
        if step >= VAL_START and (step % VAL_EVERY == 0 or step == budget):
            error = figure_error(model, data["val_batch"])
            if error < best_error:
                best, best_error = {k: v.copy() for k, v in params.items()}, error
    if best is not None:
        for k in params:
            params[k][...] = best[k]
    return history


def predict(model, X):
    """Cart positions (goals, beats + 1, 2) on the nominal floor for goal vectors X."""
    return np.stack(run(model, batch_for(reference(np.asarray(X, dtype=float)), "nominal", None))[1]["xy"], axis=1)


def hidden_states(model, X):
    """Beat-by-beat pressure, fuel and gate on the nominal floor; for the winding also its decoded code."""
    batch = batch_for(reference(np.asarray(X, dtype=float)), "nominal", None)
    tp = Tape(grad=False)
    P = {k: tp.param(v) for k, v in model["params"].items()}
    trace = rollout(tp, model, P, batch)[1]
    states = {k: np.stack(trace[k], axis=1) for k in ("pressure", "fuel", "gate")}
    if model["kind"] == "winding":
        commands, states["shares"], states["reversal"] = decode_winding(tp, P, batch["goals"], model["ko"])
        states["commands"] = commands.v
    return states


def peg_printout(model, goal):
    """Read a winding back as pegs: > roll, < reversed roll, L/l and R/r pivots, . slack; digit = wound share."""
    states = hidden_states(model, goal[None])
    shares, reversal = states["shares"][0], states["reversal"][0]
    marks, last, reversals = [], 0.0, 0
    for t in range(T_BEATS):
        wound = 1.0 - shares[t, 3]
        if wound < 0.5:
            marks.append("." + str(int(round(9 * wound))))
            continue
        sign = 1.0 if reversal[t] >= 0.0 else -1.0
        reversals += int(last != 0.0 and sign != last)
        last = sign
        marks.append("><LlRr"[2 * int(np.argmax(shares[t, :3])) + int(sign < 0.0)] + str(min(9, int(round(9 * wound)))))
    return " | ".join("".join(marks[a:b]) for a, b in ((0, 8), (8, 12), (12, 16), (16, 24))), reversals


def data_bridge(path, seed, budget):
    """Fit one winding to a 2-D demonstration CSV with header x,y (for example a LASA shape)."""
    if not os.path.exists(path):
        return f"skipped ({path} not found)"
    table = np.genfromtxt(path, delimiter=",", names=True)
    raw = np.stack([table["x"], table["y"]], axis=1).astype(float)
    grid = np.linspace(0.0, len(raw) - 1.0, T_BEATS + 1)
    xy = np.stack([np.interp(grid, np.arange(len(raw)), raw[:, i]) for i in range(2)], axis=1) - raw[0]
    ang = math.atan2(xy[1, 1], xy[1, 0])
    xy = xy @ np.array([[math.cos(ang), math.sin(ang)], [-math.sin(ang), math.cos(ang)]]).T
    xy *= 2.0 / max(float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()), 1e-9)
    heading = np.concatenate([[0.0], np.arctan2(np.diff(xy[:, 1]), np.diff(xy[:, 0]))])
    split = {"goals": np.zeros((1, G_DIM)), "commands": np.zeros((1, T_BEATS, 2)), "ref": xy[None],
             "ref_heading": heading[None], "path_ref": np.array([2.0])}
    data = {"train": split, "val_batch": batch_for(split, "nominal", None)}
    model = build_model(G_DIM, 2, TASK_TYPES[0], np.random.default_rng(seed), kind="winding")
    fit(model, data, budget, np.random.default_rng(seed + 1))
    return f"{path}: figure_error {figure_error(model, data['val_batch']):.4f}; pegs {peg_printout(model, split['goals'][0])[0]}"


# ---------------------------------------------------------------- tests: correctness
def small_batch(split, n=16):
    return batch_for({k: v[:n] for k, v in split.items()}, "nuisance", np.random.default_rng(1234))


def check_gradients(models, batch, rng, n_entries=20):
    """C1 core: central differences against the tape on every tensor of every model given."""
    worst, checked, total = 0.0, 0, 0
    for model in models:
        grads = loss_and_grads(model, batch)[1]
        report = finite_difference_check(model["params"], grads, lambda m=model: run(m, batch)[0], rng,
                                         n_entries=n_entries, floor=MIND_CARD["thresholds"]["gradcheck_floor"])
        worst, checked, total = max(worst, max(report.values())), checked + len(report), total + len(model["params"])
    return worst, checked, total


def check_determinism(data, seed):
    """C2: same seed, same losses and outputs; everything finite; env_step replays the ideal figure."""
    runs = []
    for _ in range(2):
        model = build_model(G_DIM, 2, TASK_TYPES[0], np.random.default_rng(seed), kind="winding")
        history = fit(model, data, 10, np.random.default_rng(seed + 1))
        runs.append((history, predict(model, data["heldout"]["goals"][:4]), model))
    same = runs[0][0] == runs[1][0] and np.array_equal(runs[0][1], runs[1][1])
    finite = bool(np.isfinite(runs[0][1]).all() and all(np.isfinite(v).all() for v in runs[0][2]["params"].values()))
    state = env_reset(np.random.default_rng(seed))
    for action in script_commands(state["goal"][None])[0]:
        state = env_step(state, action, None)[0]
    gap = float(np.abs(state["pose"][:2] - state["ref"][-1]).max())
    return same and finite and gap < 1e-9, f"identical: {same}; finite: {finite}; env_step replay gap {gap:.1e}"


def check_learning(history, model, data):
    """C3: training loss falls by the declared fraction and the held-out figure beats the mean figure."""
    th = MIND_CARD["thresholds"]
    drop = 1.0 - float(np.mean(history[-10:])) / history[0]
    error, trivial = figure_error(model, batch_for(data["heldout"], "nominal", None)), trivial_error(data)
    ok = drop >= th["loss_drop_fraction"] and error <= (1.0 - th["margin_over_trivial"]) * trivial
    return ok, (f"loss drop {drop:.3f} (min {th['loss_drop_fraction']}); held-out figure_error {error:.4f} vs "
                f"trivial {trivial:.4f} (max ratio {1.0 - th['margin_over_trivial']:.2f})")


def check_shuffled(data, seed, budget):
    """C4: figures permuted across goals in train and validation; held-out error must stay near the trivial error."""
    rng = np.random.default_rng(seed + 7)
    shuffled = dict(data)
    for name in ("train", "val"):
        perm = rng.permutation(len(data[name]["goals"]))
        shuffled[name] = dict(data[name], **{k: data[name][k][perm] for k in ("ref", "ref_heading", "path_ref")})
    shuffled["val_batch"] = batch_for(shuffled["val"], "nuisance", np.random.default_rng(seed + 8))
    model = build_model(G_DIM, 2, TASK_TYPES[0], np.random.default_rng(seed + 9), kind="winding")
    fit(model, shuffled, budget, rng)
    ratio = figure_error(model, batch_for(data["heldout"], "nominal", None)) / trivial_error(data)
    floor = MIND_CARD["thresholds"]["shuffled_ratio_min"]
    return ratio >= floor, f"held-out error / trivial error {ratio:.3f} (min {floor})"


def mini_protocol(data, seed, budget):
    """The base seed's winding run exactly as in run_seed (stream, data, budget), with C1 at init and C3 after."""
    try:
        stream = np.random.default_rng(np.random.SeedSequence(seed).spawn(5)[2])
        model = build_model(G_DIM, 2, TASK_TYPES[0], stream, kind="winding")
        c1 = check_gradients([model], small_batch(data["train"]), np.random.default_rng(seed), n_entries=4)[0]
        history = fit(model, data, budget, stream)
        return c1 <= MIND_CARD["thresholds"]["gradcheck_rel_error"], bool(check_learning(history, model, data)[0])
    except FloatingPointError:
        return True, False


def check_mutants(data, seed, budget):
    """C5: the unmutated run passes C1 and C3; every registered mutant must fail one of them."""
    global ACTIVE_MUTANT
    outer, control, caught = ACTIVE_MUTANT, mini_protocol(data, seed, budget), {}
    for name in MUTANTS:
        ACTIVE_MUTANT = name
        try:
            caught[name] = not all(mini_protocol(data, seed, budget))
        finally:
            ACTIVE_MUTANT = outer
    ok = all(control) and all(caught.values())
    return ok, caught, f"unmutated base-seed run passes C1 and C3: {all(control)}; mutants caught {sum(caught.values())}/{len(caught)}"


def check_reversibility(rng):
    """C6.1: a winding replayed backwards with every sign flipped returns the cart exactly to its start."""
    model, worst, control = dict(SCRIPT, ko={"falling_weight": "identity"}), 0.0, np.inf
    for scale in (0.25, 0.6, 1.0):
        n = 64
        cmds = rng.uniform(-scale, scale, (n, T_BEATS, 2))
        start = np.column_stack([rng.normal(0.0, 1.0, (n, 2)), rng.uniform(-np.pi, np.pi, n)])
        base = dict(goals=np.zeros((n, G_DIM)), ref=np.zeros((n, T_BEATS + 1, 2)), path_ref=np.zeros(n),
                    **nominal_noise(n))
        gaps = []
        for euler in (False, True):
            there = run(model, dict(base, start=start, commands=cmds, euler=euler))[1]
            end = np.column_stack([there["xy"][-1], there["heading"][-1]])
            back = run(model, dict(base, start=end, commands=-cmds[:, ::-1].copy(), euler=euler))[1]
            gaps.append(float(np.abs(np.column_stack([back["xy"][-1], back["heading"][-1]]) - start).max()))
        worst, control = max(worst, gaps[0]), min(control, gaps[1])
    th = MIND_CARD["thresholds"]
    ok = worst <= th["property_tol"] and control > th["negative_control_min_violation"]
    return ok, f"max return gap {worst:.1e}; Euler negative control gap {control:.1e}"


def check_fuel(rng):
    """C6.2: whatever the winding, slip and millet flow do, the weight never climbs back up."""
    worst, control = -np.inf, -np.inf
    for flow_sd in (0.5, 1.5, 3.0):
        model = build_model(G_DIM, 2, TASK_TYPES[0], rng, kind="winding")
        for arr in model["params"].values():
            arr *= 4.0
        n = 64
        batch = dict(goals=rng.uniform(-1.0, 1.0, (n, G_DIM)), ref=np.zeros((n, T_BEATS + 1, 2)), path_ref=np.zeros(n),
                     start=np.zeros((n, 3)), slip=1.0 + 0.3 * rng.standard_normal((n, T_BEATS, 2)),
                     friction=np.zeros(T_BEATS), flow=flow_sd * rng.standard_normal((n, T_BEATS)))
        climbs = [float(np.diff(np.stack(run(model, dict(batch, raw_pressure=raw))[1]["fuel"], axis=1), axis=1).max())
                  for raw in (False, True)]
        worst, control = max(worst, climbs[0]), max(control, climbs[1])
    th = MIND_CARD["thresholds"]
    ok = worst <= th["property_tol"] and control > th["negative_control_min_violation"]
    return ok, f"largest fuel rise {worst:.1e}; raw-pressure negative control rise {control:.1e}"


def check_valve(rng):
    """C6.3 definition check (holds by construction for gain in (0, 1); not evidence): deviation contracts."""
    tp, n = Tape(grad=False), 512
    setpoint, start, kappa = rng.uniform(0.1, 3.0, n), rng.uniform(-3.0, 3.0, n), sigmoid(rng.uniform(-6.0, 6.0, n))
    p, excess = start, -np.inf
    for _ in range(T_BEATS):
        new = valve_update(tp, p, kappa, setpoint, 0.0).v
        excess = max(excess, float(np.max(np.abs(new - setpoint) - (1.0 - kappa) * np.abs(p - setpoint))))
        p = new
    bad = start
    for _ in range(T_BEATS):
        bad = valve_update(tp, bad, 2.5, setpoint, 0.0).v
    growth = float(np.min(np.abs(bad - setpoint) - np.abs(start - setpoint)))
    ok = excess <= MIND_CARD["thresholds"]["property_tol"] and growth > 0.0
    return ok, f"largest contraction excess {excess:.1e}; overshooting gain 2.5 grows every deviation: {growth > 0.0}"


def check_splits(data):
    """C7: no goal appears in two splits and the shifted corner is visited only by the shifted split."""
    keys = {name: {hashlib.sha256(np.round(g, 12).tobytes()).hexdigest() for g in data[name]["goals"]}
            for name in SPLIT_SIZES}
    disjoint = all(not keys[a] & keys[b] for a, b in itertools.combinations(SPLIT_SIZES, 2))
    corner = bool(in_corner(data["shifted"]["goals"]).all()
                  and not any(in_corner(data[s]["goals"]).any() for s in ("train", "val", "heldout")))
    return disjoint and corner, f"pairwise disjoint: {disjoint}; corner confined to the shifted split: {corner}"


# ---------------------------------------------------------------- tests: hypotheses
EVAL_PLAN = (("val", "nuisance"), ("heldout", "nominal"), ("heldout", "nuisance"), ("heldout", "blind"),
             ("shifted", "nominal"))


def run_seed(seed, budget):
    """Train winding, size-matched policy and rival regulator on one seed's data; score them on shared noise."""
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(5)]
    data = make_data(streams[0])
    evals = {key: batch_for(data[key[0]], key[1], streams[1]) for key in EVAL_PLAN}
    data["val_batch"] = evals[("val", "nuisance")]
    models, histories = {}, {}
    for kind, stream in zip(("winding", "policy", "regulator"), streams[2:]):
        models[kind] = build_model(G_DIM, 2, TASK_TYPES[0], stream, kind=kind)
        histories[kind] = fit(models[kind], data, budget, stream)
    err = {(kind, key): figure_error(models[kind], evals[key]) for kind in models for key in EVAL_PLAN[1:]}
    base = err[("winding", ("heldout", "nuisance"))]
    kos = {name: figure_error(knockout(models["winding"], name, mode), evals[("heldout", "nuisance")]) - base
           for name, mode in KNOCKOUT_PLAN}
    e = {(kind, split, cond): v for (kind, (split, cond)), v in err.items()}
    row = {"H-SIG": e[("winding", "shifted", "nominal")] - e[("policy", "shifted", "nominal")],
           "H-NEC": kos["winding"] - kos["float_valve"],
           "H-BLIND": e[("winding", "heldout", "blind")] - e[("policy", "heldout", "blind")],
           "H-RIVAL": ((e[("winding", "heldout", "blind")] - e[("regulator", "heldout", "blind")])
                       - (e[("winding", "heldout", "nominal")] - e[("regulator", "heldout", "nominal")]))}
    return {"data": data, "models": models, "histories": histories, "errors": e, "knockouts": kos, "row": row}


def evaluate_hypotheses(runs, seed, evaluated):
    """Paired per-seed differences, percentile bootstrap, verdicts against the frozen card."""
    rng = np.random.default_rng(seed + 9973)

    def summary(values):
        return paired_bootstrap(values, rng) if evaluated else (float(np.mean(values)), None)
    hyps = []
    for h in MIND_CARD["hypotheses"]:
        mean, ci = summary([r["row"][h["id"]] for r in runs])
        hyps.append({"id": h["id"], "metric": h["metric"], "mean_diff": mean, "ci95": ci, "mesi": h["mesi"],
                     "n_seeds": len(runs),
                     "verdict": verdict(mean, ci, h["mesi"], h["direction"]) if evaluated else "not evaluated"})
    table = modules(runs[0]["models"]["winding"])
    kos = []
    for name, mode in KNOCKOUT_PLAN:
        mean, ci = summary([r["knockouts"][name] for r in runs])
        kos.append({"module": name, "mode": mode, "signature": table[name]["signature"], "metric_change": mean,
                    "ci95": ci})
    return hyps, kos


# ---------------------------------------------------------------- report
def emit_report(ctx, json_path):
    def ci_text(ci):
        return "n/a" if ci is None else f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
    g, m = ctx["grad"], ctx["mutants"]
    lines = ["=== VERIFIED REPORT · chapter 0118 ===",
             f"file: {ctx['file']} · card_revision {MIND_CARD['card_revision']} · mode {ctx['mode']} · "
             f"mutant {ACTIVE_MUTANT}",
             f"environment: python {sys.version.split()[0]} · numpy {np.__version__}",
             f"seeds: {ctx['seeds']} · runtime_s {ctx['runtime']:.1f} · budget_s {TIME_BUDGET[ctx['mode']]:.0f}",
             "n_params: " + " · ".join(f"{k} {v}" for k, v in ctx["n_params"].items()),
             f"gradcheck: {g['tensors_checked']}/{g['tensors_total']} tensors at init and after training · "
             f"max_rel_error {g['max_rel_error']:.2e} · passed {g['passed']}",
             "correctness:"]
    lines += [f"  {tid:<5} {name:<26} {'PASS' if ok else 'FAIL'}  {detail}" for tid, name, ok, detail in ctx["tests"]]
    lines.append(f"mutants: {m['detected']}/{m['total']} detected · score {m['score']:.2f} · "
                 + ", ".join(f"{k} {'caught' if v else 'MISSED'}" for k, v in ctx["caught"].items()))
    lines.append("hypotheses (paired over seeds; 95% percentile bootstrap of the mean, 2000 resamples):")
    lines += [f"  {h['id']:<8} mean_diff {h['mean_diff']:+.4f} ci95 {ci_text(h['ci95'])} mesi {h['mesi']} "
              f"seeds {h['n_seeds']} -> {h['verdict']}" for h in ctx["hypotheses"]]
    lines.append("knockouts (winding automaton; held-out goals, nuisance floor; change in figure_error):")
    lines += [f"  {k['module']:<15} {k['mode']:<9} signature {str(k['signature']):<5} {k['metric_change']:+.4f} "
              f"ci95 {ci_text(k['ci95'])}" for k in ctx["knockouts"]]
    lines += ["figure_error by model (seed mean; held-out nominal / held-out blind / shifted nominal): "
              + " · ".join(f"{kind} {v[0]:.4f} / {v[1]:.4f} / {v[2]:.4f}" for kind, v in ctx["errors"].items()),
              f"fuel left at the end (held-out goals, nominal floor, seed {ctx['seeds'][0]}): "
              f"min {100.0 * ctx['fuel_left']:.1f}% of the weight's drop",
              f"peg printout (seed {ctx['seeds'][0]}, held-out goal 0): {ctx['pegs']} · reversals {ctx['reversals']}",
              f"real-data bridge: {ctx['bridge']}",
              f"task_types: {', '.join(TASK_TYPES)}",
              f"exit_code: {ctx['exit_code']}",
              "=== END REPORT ==="]
    payload = {"schema_version": "1.0", "chapter": 118, "file": ctx["file"], "card_revision": MIND_CARD["card_revision"],
               "environment": {"python": sys.version.split()[0], "numpy": np.__version__}, "seeds": ctx["seeds"],
               "runtime_s": round(ctx["runtime"], 2), "n_params": ctx["n_params"]["winding"], "gradcheck": g,
               "correctness": [{"id": i, "name": n, "passed": bool(ok), "detail": d} for i, n, ok, d in ctx["tests"]],
               "mutants": m, "hypotheses": ctx["hypotheses"], "knockouts": ctx["knockouts"],
               "task_types": TASK_TYPES, "exit_code": ctx["exit_code"]}
    write_report(lines, payload, json_path)


# ---------------------------------------------------------------- command line
def protocol(mode, base_seed, n_seeds, json_path, data_path):
    start = time.time()
    budget, seeds, kinds = STEPS[mode], [base_seed + i for i in range(n_seeds)], ("winding", "policy", "regulator")
    print(f"{os.path.basename(__file__)} · mode {mode} · seeds {seeds} · mutant {ACTIVE_MUTANT}")
    runs = []
    for seed in seeds:
        runs.append(run_seed(seed, budget))
        print(f"  seed {seed}: winding, policy and regulator trained ({time.time() - start:.1f} s)", flush=True)
    base, th = runs[0], MIND_CARD["thresholds"]
    data, models = base["data"], base["models"]
    batch = small_batch(data["train"])
    fresh = [build_model(G_DIM, 2, TASK_TYPES[0], np.random.default_rng(base_seed + 50 + i), kind=k)
             for i, k in enumerate(kinds)]
    at_init = check_gradients(fresh, batch, np.random.default_rng(base_seed + 3))
    after = check_gradients([models[k] for k in kinds], batch, np.random.default_rng(base_seed + 4))
    worst = max(at_init[0], after[0])
    c1 = bool(worst <= th["gradcheck_rel_error"] and at_init[1] == at_init[2] and after[1] == after[2])
    c5, caught, c5_detail = check_mutants(data, base_seed, budget)
    tests = [("C1", "gradient_check", c1, f"max rel error {worst:.2e}; every tensor of winding, policy and regulator "
                                          f"at init and after {budget} updates"),
             ("C2", "determinism_finiteness", *check_determinism(data, base_seed)),
             ("C3", "learning", *check_learning(base["histories"]["winding"], models["winding"], data)),
             ("C4", "shuffled_label_control", *check_shuffled(data, base_seed, budget)),
             ("C5", "mutant_detection", c5, c5_detail),
             ("C6.1", "program_reversibility", *check_reversibility(np.random.default_rng(base_seed + 5))),
             ("C6.2", "weight_never_refills", *check_fuel(np.random.default_rng(base_seed + 6))),
             ("C6.3", "valve_contracts_definition", *check_valve(np.random.default_rng(base_seed + 7))),
             ("C7", "split_integrity", *check_splits(data))]
    hyps, kos = evaluate_hypotheses(runs, base_seed, mode == "full" and n_seeds >= 5)
    bridge = data_bridge(data_path, base_seed, budget) if data_path else "skipped (no --data PATH given)"
    fuel_left = float(hidden_states(models["winding"], data["heldout"]["goals"])["fuel"][:, -1].min() / FUEL0)
    pegs, reversals = peg_printout(models["winding"], data["heldout"]["goals"][0])
    errors = {k: [float(np.mean([r["errors"][(k, s, c)] for r in runs])) for s, c in
                  (("heldout", "nominal"), ("heldout", "blind"), ("shifted", "nominal"))] for k in kinds}
    runtime = time.time() - start
    tests.append(("C8", "budget", runtime <= TIME_BUDGET[mode], f"{runtime:.1f} s of {TIME_BUDGET[mode]:.0f} s"))
    tests = [(i, n, bool(ok), d) for i, n, ok, d in tests]
    failed = [i for i, _, ok, _ in tests if not ok]
    exit_code = 0 if not failed else (3 if failed == ["C8"] else 1)
    ctx = {"file": os.path.basename(__file__), "mode": mode, "seeds": seeds, "runtime": runtime,
           "n_params": {k: n_params(models[k]) for k in kinds},
           "grad": {"tensors_checked": min(at_init[1], after[1]), "tensors_total": after[2],
                    "max_rel_error": worst, "checked_at": ["init", "after_training_steps"], "passed": c1},
           "tests": tests, "caught": caught,
           "mutants": {"detected": int(sum(caught.values())), "total": len(caught),
                       "score": float(sum(caught.values())) / len(caught)},
           "hypotheses": hyps, "knockouts": kos, "errors": errors, "fuel_left": fuel_left, "pegs": pegs,
           "reversals": reversals, "bridge": bridge, "exit_code": exit_code}
    emit_report(ctx, json_path)
    return exit_code


def main(argv=None):
    global ACTIVE_MUTANT
    parser = argparse.ArgumentParser(prog=os.path.basename(__file__),
                                     description="Chapter 0118 winding automaton: the full protocol runs by default.")
    parser.add_argument("--quick", action="store_true", help="one seed, reduced updates, all correctness tests")
    parser.add_argument("--seed", type=int, default=0, help="base seed")
    parser.add_argument("--seeds", type=int, default=None, help="number of seeds (default 5, or 1 with --quick)")
    parser.add_argument("--json", default=None, help="also write the JSON report to this path")
    parser.add_argument("--card", action="store_true", help="print MIND_CARD as JSON and exit")
    parser.add_argument("--mutant", default=None, help="run with a registered mutant: " + ", ".join(MUTANTS))
    parser.add_argument("--data", default=None, help="optional CSV (header x,y) for the real-data bridge")
    args = parser.parse_args(argv)
    if args.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    if args.mutant is not None and args.mutant not in MUTANTS:
        print(f"unknown mutant {args.mutant!r}; registered: {', '.join(MUTANTS)}", file=sys.stderr)
        return 2
    n_seeds = args.seeds if args.seeds is not None else (1 if args.quick else 5)
    if n_seeds < 1:
        print("--seeds must be at least 1", file=sys.stderr)
        return 2
    ACTIVE_MUTANT = args.mutant
    try:
        return protocol("quick" if args.quick else "full", args.seed, n_seeds, args.json, args.data)
    except FloatingPointError as exc:
        print(f"non-finite values: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())

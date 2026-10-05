#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0279 · Judah Halevi (1075-1141)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 14, Minds 261-280 Available on Amazon https://www.amazon.com/dp/B0HLYSVY5D
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0279_judah_halevi_1075 - Judah Halevi (1075-1141)
# END ATTRIBUTION
"""The Physician's Measures: re-derivation-licensed transmission of a practice.

Thesis
    A learner may amend a component of an inherited practice only where its own
    model of consequences re-derives how that component's measure must change
    with each circumstance; whatever it cannot re-derive it hands on unchanged,
    because its verdict there is blind to the share of the effect it cannot see.

Evidence and provenance (belief: his own surviving Judaeo-Arabic book, the Kuzari)
    I:79 - the conditions that make an act effective (quantity, quality, time,
    place, connexion) cannot be gauged by the agent; the fool in the physician's
    dispensary kills with the drugs that should have cured.
    I:99, III:53 - prescribed acts are like natural compounds whose parts stand in
    exact relation; the fruit appears only when the whole is done.
    II:48, III:7 - rational laws are known to reason; the divine laws are ones
    "reason neither demands, nor forbids" (tr. Hirschfeld 1905).
    III:49 - where argument and taste rule, the law loses "some component parts";
    III:38 - reasoners diverge, "as many codes as opinions".
    I:1-2 - the king's dream: his way of thinking pleases, his way of acting does
    not; a verdict on the whole of his deeds, with no attribution to any part.
    I:25, III:73 - uninterrupted tradition counts as experience; disciples hand on
    words they do not grasp. Blind spot in his own text: II:80 (swaying at
    reading explained as a habit imitated from the age of shared books), III:31-32
    (vowel and accent signs given prophetic or divinely assisted origin), II:64
    (the fixed calendar called free from error; its mean year runs about one day
    in 216 years long against the tropical year).

Doctrine -> mechanism -> test
    D1 I:79, II:50, II:56   -> M4 reshut: concordance of sensitivities   -> C6.1, C6.3, H-SIG
    D2 I:99, III:53         -> task: joint hidden measure, aggregate verdict -> C4, H-SIG
    D3 II:48, III:7         -> M3 tiqqun amends licensed components only  -> C6.5, H-NEC
    D4 III:38, III:49       -> rival: derivation from one's own critic     -> H-RIVAL
    D5 I:1-2                -> M2 dhawq: verdict head fits round means only -> C3
    D6 I:25, III:73         -> M1 masorah: cloning of the received deed     -> C3, H-CHAIN
    D7 IV:5-6 (Krinis)      -> M4 moves one circumstance at a time          -> C6.2
    D8 II:80, III:31, II:64 -> blind spot: useless accretions are kept      -> H-BLIND

Research question (continual learning and plasticity)
    In a chain of learners, each cloning its predecessor's practice and then
    improving it with its own outcome model, which components may a learner
    change? Tested answer: only those whose dependence on circumstances its own
    critic re-derives.

Closest prior art and the delta
    TD3+BC (Fujimoto and Gu 2021) adds a uniform cloning term to critic ascent;
    adaptive variants weight that term per state. The critic is the diagonal
    quadratic-advantage form of NAF (Gu et al. 2016). Here amendment is licensed
    per output component and per context by Lin's concordance (1989) between
    the finite-difference sensitivities of the received deed and of the critic's
    derived deed, and the licence is tested along an iterated-learning chain
    (Kirby et al. 2008; Shumailov et al. 2024). Baseline: size-matched TD3+BC.
    Rival: pure derivation. Also reported: pure cloning.

Blind spot (declared before any run)
    A useless accretion that reason cannot re-derive is kept like a necessary
    statute: where unexplained components only cost, the licensed learner loses
    to the rival that prunes them. A measure that does not vary with any
    circumstance can never be licensed for amendment.

Task (native): the physician's world
    context: 4 causal features ~ N(0, I) plus 2 nuisance features; deed in R^8.
    components 0-4 (explicable): visible outcome -(a_k - a*_k(c, g))^2; the
        optimum's level and slopes drift each generation g.
    components 5-7 (opaque): visible cost -kappa a_k^2 only; hidden outcome
        H (exp(-|a_O - a*_O(c)|^2 / 2 s^2) - 1), seen only as one noisy verdict
        on each round of trials (the dream).
    chain: generation g clones generation g-1 (copy noise), lives two rounds of
        exploratory trials, fits its critic, amends, and becomes the teacher.
    splits: held-out contexts; shifted contexts (mean-shifted patients).
    accretion world: the same practice, but the hidden outcome is zero.
    trivial baseline: the fool's dose, one context-free mean deed for all.

Limits
    Synthetic one-step episodes. The hidden effect is invisible to per-trial
    outcomes by design; that is the premise under test, not a finding. A
    research prototype of an AGI-oriented component, not an AGI and not a model
    of Halevi's mind.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 2,
    "revision_log": [{"revision": 2, "date": "2026-09-23", "why": (
        "Pilot runs on seed 0, before any hypothesis run: an MLP critic over (context, deed) could not locate "
        "optima beyond explored deeds, so it became a diagonal quadratic-advantage (NAF) critic; the level-"
        "agreement gate refused amendment exactly where drift was largest and its soft licence leaked amendment "
        "onto opaque components, so it became profile concordance with a chance threshold tau = 0.5. Exploration "
        "0.25 -> 0.5, trials per round 320 -> 480, critic steps 350 -> 800, slope drift added to the world, gate "
        "tuning grid window {0.25, 0.35} -> nbr_h {0.3, 0.6}. Hypotheses, metrics, mesi and thresholds unchanged.")}],
    "generation": {"template_version": "codeguidelines 1.0 (2026-09-15)", "generator": "Claude",
                   "generator_version": "Claude Opus 5.5", "date": "2026-09-23"},
    "id": 279, "figure": "Judah Halevi", "born": 1075, "died": 1141,
    "civilization": "Andalusi Jewish (al-Andalus and the Christian north; Hebrew and Judaeo-Arabic)",
    "provenance": "belief",
    "thesis": ("Amend only those parts of an inherited practice whose dependence on circumstances your own model of "
               "consequences re-derives; hand on the rest with its exact measures."),
    "evidence": [
        {"id": "D1", "claim": "The conditions of an act's efficacy cannot be gauged by the agent; the fool in the "
         "dispensary", "basis": "primary", "source": "Kuzari I:79; II:50; II:56 (Hirschfeld 1905)"},
        {"id": "D2", "claim": "Prescribed acts are like natural compounds of exact relations; the fruit appears only "
         "when the whole is done", "basis": "primary", "source": "Kuzari I:99; III:53"},
        {"id": "D3", "claim": "Rational laws are known to reason; divine laws are neither demanded nor forbidden by "
         "it", "basis": "primary", "source": "Kuzari II:48; III:7"},
        {"id": "D4", "claim": "Argument and taste as guides lose component parts and multiply codes",
         "basis": "primary", "source": "Kuzari III:38; III:49"},
        {"id": "D5", "claim": "The dream judges the way of acting as a whole, without attribution",
         "basis": "primary", "source": "Kuzari I:1-2"},
        {"id": "D6", "claim": "Uninterrupted tradition equals experience; words handed on ungrasped",
         "basis": "primary", "source": "Kuzari I:25; III:73"},
        {"id": "D7", "claim": "The intellect grasps one thing at a time, prophecy many at once", "basis":
         "scholarship", "source": "Krinis, 'Judah Halevi', Stanford Encyclopedia of Philosophy (2026), on IV:5-6"},
        {"id": "D8", "claim": "He kept fossils: swaying as imitated habit; vowel signs given prophetic origin; the "
         "calendar called error-free", "basis": "primary", "source": "Kuzari II:80; III:31-32; II:64"},
    ],
    "research_question": {"category": "continual learning and plasticity", "question": (
        "Across a chain of learners that clone and then improve a practice, does licensing amendment per component "
        "by re-derivation of its sensitivities keep components whose value the learner cannot see, while still "
        "tracking the components it understands?")},
    "mechanism": {
        "name": "Re-derivation-licensed transmission (the physician's measures)",
        "family": "behaviour-regularized policy improvement across an iterated-learning chain",
        "signature_modules": ["reshut"],
        "closest_prior_art": ["TD3+BC (Fujimoto and Gu, NeurIPS 2021)",
                              "per-state adaptive cloning weights (TD3 with reverse-KL regularizer 2022; SelfBC 2024)",
                              "NAF quadratic-advantage critic (Gu, Lillicrap, Sutskever and Levine, ICML 2016)",
                              "concordance correlation coefficient (Lin, Biometrics 1989)",
                              "iterated learning (Kirby, Cornish and Smith, PNAS 2008); model collapse "
                              "(Shumailov et al., Nature 2024)",
                              "overimitation and opaque technology (Lyons, Young and Keil 2007; Derex et al. 2019)"],
        "overlap": "Medium",
        "prior_art_queries": ["offline RL per-dimension behavior cloning weight action dimension-wise constraint",
                              "uncertainty weighted TD3+BC adaptive constraint per state",
                              "Chesterton's fence reinforcement learning preserve unexplained components",
                              "iterated learning policy distillation collapse"],
        "contribution_type": "mechanism",
        "delta": ("The cloning constraint is lifted per output component and per context only where the critic's "
                  "derived deed reproduces the received deed's sensitivities to each circumstance."),
    },
    "traceability": [
        {"doctrine": "D1", "mechanism": "M4", "property_test": "C6.1", "hypothesis": "H-SIG"},
        {"doctrine": "D1", "mechanism": "M4", "property_test": "C6.3", "hypothesis": "H-SIG"},
        {"doctrine": "D3", "mechanism": "M3", "property_test": "C6.5", "hypothesis": "H-NEC"},
        {"doctrine": "D4", "mechanism": "rival", "property_test": "C6.4", "hypothesis": "H-RIVAL"},
        {"doctrine": "D6", "mechanism": "M1", "property_test": "C3", "hypothesis": "H-CHAIN"},
        {"doctrine": "D7", "mechanism": "M4", "property_test": "C6.2", "hypothesis": "H-SIG"},
        {"doctrine": "D8", "mechanism": "M4", "property_test": "C6.1", "hypothesis": "H-BLIND"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "Licensed transmission keeps more welfare than size-matched TD3+BC after the "
         "chain", "metric": "normalized welfare, generation G", "split": "shifted", "world": "necessary",
         "comparison": "model - baseline", "direction": "greater", "mesi": 0.10, "seeds": 5},
        {"id": "H-NEC", "statement": "Replacing the gate by its mean hurts more than removing the amendment "
         "pathway", "metric": "normalized welfare, generation G", "split": "shifted", "world": "necessary",
         "comparison": "signature_knockout - matched_knockout", "direction": "less", "mesi": 0.05, "seeds": 5},
        {"id": "H-BLIND", "statement": "Where unexplained components are useless accretions, the licensed learner "
         "keeps them and loses to the pruning rival", "condition": "accretion world",
         "grounding": "Kuzari II:80, III:31-32, II:64", "metric": "normalized welfare, generation G",
         "split": "held-out", "comparison": "model - rival", "direction": "less", "mesi": 0.03, "seeds": 5},
        {"id": "H-RIVAL", "statement": "Licensed transmission beats derivation from one's own critic",
         "metric": "normalized welfare, generation G", "split": "held-out", "world": "necessary",
         "comparison": "model - rival", "direction": "greater", "mesi": 0.10, "seeds": 5},
        {"id": "H-CHAIN", "statement": "The advantage over TD3+BC grows along the chain", "metric":
         "(model - baseline) at G minus at generation 1", "split": "held-out", "world": "necessary",
         "comparison": "gap_G - gap_1", "direction": "greater", "mesi": 0.05, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.5, "critic_loss_drop_fraction": 0.3, "margin_over_trivial": 0.4,
                   "shuffled_band": 0.2, "gradcheck_rel_error": 1e-5},
    "probe_predictions": [{"probe": p, "expected": e + " baseline"} for p, e in (
        ("P5", "above"), ("P6", "below"), ("P8", "equal to"), ("P9", "above"), ("P11", "below"), ("P1", "equal to"))],
    "dialectic_links": [
        {"chapter": 277, "relation": "teacher", "test": "none (influence on the critique of philosophy; Baneth 1941)"},
        {"chapter": 257, "relation": "rival", "test": "H-RIVAL (the philosopher of Kuzari I:1 voices Avicennian views)"},
        {"chapter": 235, "relation": "rival", "test": "H-RIVAL (Farabian theory of prophecy attacked, I:87)"},
        {"chapter": 303, "relation": "successor", "test": "none (reasons for the commandments; not implemented)"}],
    "corpus_neighbors": [
        {"chapter": 277, "similarity": None, "difference": "al-Ghazali: no self-certification of reliability; here "
         "the question is which parts of an inherited practice a learner may edit"},
        {"chapter": 270, "similarity": None, "difference": "Ibn Gabirol: invertible stacked forms; no transmission"},
        {"chapter": 278, "similarity": None, "difference": "Ari: dates solved over witness chains; here deeds, not dates"},
        {"chapter": 26, "similarity": None, "difference": "Nebuchadnezzar: restore one buried template everywhere; "
         "here restoration is selective and amendment is licensed"},
        {"chapter": 225, "similarity": None, "difference": "Muslim ibn al-Hajjaj: licences attached to reports; "
         "here licences attach to action components by re-derivation"},
        {"chapter": 349, "similarity": None, "difference": "Ibn Taymiyyah: certainty by contact; no chain or gate"}],
    "barometer": {
        "cognitive_processing": ["held-out and shifted welfare after G generations"],
        "embodied_cognition": ["none directly; P11 predicted below baseline"],
        "world_modeling": ["critic fit of visible outcomes under drift"],
        "consciousness": ["gate as a self-report of which components it understands (calibration not claimed)"],
        "language_understanding": [], "emotional_intelligence": [], "creativity": [],
        "autonomy": ["continual learning across generations with bounded self-amendment"]},
    "task_types": ["vector_regression", "episodic_control"],
    "applications": [
        {"use": "policy improvement from demonstrations that must keep rarely-rewarded action dimensions",
         "sector": "robotics and control", "dataset": "D4RL (Fu et al. 2020)"},
        {"use": "transfer of process recipes across plants where some steps are undocumented",
         "sector": "manufacturing", "dataset": "SECOM (UCI Machine Learning Repository)"},
        {"use": "recursive synthetic-data training that should not shed rare behaviours",
         "sector": "AI infrastructure", "dataset": "WikiText-103 (Merity et al. 2016)"}],
    "safety_notes": ("Medicine appears only as the Kuzari's parable; no dosing or treatment content. The "
                     "doctrine of a hereditary prophetic faculty (Kuzari I:95, I:115) is discussed as history "
                     "and is never encoded as a feature of people. No sentence is presented as Halevi's words "
                     "except translated quotations of his book."),
}

# ----------------------------------------------------------------------------- imports and constants
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

CFG = {
    "dc": 4, "dn": 2, "K": 8, "KE": 5,
    "kappa": 0.25, "H": 3.0, "s": 0.30, "drift": 0.20,
    "sigma_v": 0.15, "sigma_verdict": 0.30, "sigma_copy": 0.05, "sigma_explore": 0.50,
    "n_demo": 320, "n_round": 480, "rounds": 2, "n_eval": 400,
    "steps_bc": 450, "steps_critic": 800, "steps_amend": 220, "steps_actor": 220,
    "batch": 64, "lr": 3e-3, "wd": 1e-4, "clip": 5.0, "beta_amend": 0.01,
    "nbr_h": 0.5, "tau": 0.5, "slope_drift": 0.10,
    "td3_alpha": 2.5, "G": 5, "shift": [1.0, -1.0, 0.6, 0.0],
    "h_masorah": 24, "h_tiqqun": 12, "h_actor": 37, "h_critic": 32,
}
QUICK = {"n_demo": 192, "n_round": 192, "steps_bc": 250, "steps_critic": 200, "steps_amend": 120,
         "steps_actor": 120, "G": 2, "n_eval": 240}
MUTANT = {"name": None}
TASK_TYPES = ["vector_regression", "episodic_control"]

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


class Adam:
    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads, keys=None):
        self.t += 1
        lr = 0.0 if MUTANT["name"] == "zero_learning_rate" else self.lr
        sign = 1.0 if MUTANT["name"] == "sign_flipped_update" else -1.0
        for k in (keys if keys is not None else grads):
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] += sign * lr * mh / (np.sqrt(vh) + self.eps)


def clip_by_global_norm(grads, max_norm):
    total = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    if total > max_norm:
        for k in grads:
            grads[k] = grads[k] * (max_norm / (total + 1e-12))
    return total


def gradcheck(loss_fn, params, keys, rng, eps=1e-6, n_entries=20):
    """Central differences on >= n_entries random entries per tensor plus the largest-gradient entry."""
    _, grads = loss_fn(params)
    worst = {}
    for k in keys:
        p, g = params[k], grads[k]
        flat = np.arange(p.size)
        pick = list(rng.choice(flat, size=min(n_entries, p.size), replace=False))
        pick.append(int(np.argmax(np.abs(g).ravel())))
        err = 0.0
        for idx in pick:
            ij = np.unravel_index(idx, p.shape)
            old = p[ij]
            p[ij] = old + eps
            lp, _ = loss_fn(params)
            p[ij] = old - eps
            lm, _ = loss_fn(params)
            p[ij] = old
            num = (lp - lm) / (2 * eps)
            ana = g[ij]
            err = max(err, abs(num - ana) / max(1e-4, abs(num) + abs(ana)))
        worst[k] = err
    return worst


def paired_bootstrap(diffs, rng, n=2000):
    d = np.asarray(diffs, dtype=float)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def verdict_of(mean, lo, hi, direction, mesi):
    if direction == "greater":
        if lo > 0 and mean >= mesi:
            return "supported"
        return "contradicted" if hi < 0 else "inconclusive"
    if hi < 0 and mean <= -mesi:
        return "supported"
    return "contradicted" if lo > 0 else "inconclusive"


def write_report(lines, report, path=None):
    print("=== VERIFIED REPORT · chapter %04d ===" % report["chapter"])
    for ln in lines:
        print(ln)
    print("=== END REPORT ===")
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
# END STANDARD UTILITIES


# ----------------------------------------------------------------------------- data and tasks
class NonFinite(Exception):
    """Raised on any non-finite loss or deed; the entry point turns it into exit code 4."""


def check_finite(*arrays):
    for a in arrays:
        if not np.all(np.isfinite(a)):
            raise NonFinite("non-finite values")


class World:
    """The physician's world (Kuzari I:79). 'necessary': the opaque trio carries a hidden joint measure.
    'accretion': the same received practice, but the hidden term is identically zero (blind-spot world)."""

    def __init__(self, rng, kind, c):
        self.kind, self.c = kind, c
        dc, ke, ko = c["dc"], c["KE"], c["K"] - c["KE"]
        self.AE = rng.normal(0.0, 0.5, (dc, ke))
        self.bE = rng.normal(0.0, 0.3, ke)
        self.BO = rng.normal(0.0, 0.8, (dc, ko))
        steps = rng.normal(0.0, c["drift"], (c["G"] + 2, ke))
        steps[0] = 0.0
        self.drift = np.cumsum(steps, axis=0)
        tilt = rng.normal(0.0, c["slope_drift"], (c["G"] + 2, dc, ke))   # time changes the slopes too
        tilt[0] = 0.0
        self.tilt = np.cumsum(tilt, axis=0)
        self.shift = np.asarray(c["shift"], dtype=float)
        self.fool = self.physician(self.contexts(rng, 4000)).mean(axis=0)

    def contexts(self, rng, n, shifted=False):
        C = rng.normal(0.0, 1.0, (n, self.c["dc"])) + (self.shift if shifted else 0.0)
        N = rng.normal(0.0, 1.0, (n, self.c["dn"]))
        return np.concatenate([np.clip(C, -3.5, 3.5), N], axis=1)

    def opt_E(self, X, g):
        return X[:, :self.c["dc"]] @ (self.AE + self.tilt[g]) + self.bE + self.drift[g]

    def opt_O(self, X):
        return 0.9 + 0.35 * np.tanh(X[:, :self.c["dc"]] @ self.BO)

    def physician(self, X):
        """Generation-0 teacher. In the accretion world the opaque doses are an ancestral accretion."""
        return np.concatenate([self.opt_E(X, 0), self.opt_O(X)], axis=1)

    def optimum(self, X, g):
        O = self.opt_O(X) if self.kind == "necessary" else np.zeros((len(X), self.c["K"] - self.c["KE"]))
        return np.concatenate([self.opt_E(X, g), O], axis=1)

    def visible(self, X, A, g):
        ke = self.c["KE"]
        return -np.sum((A[:, :ke] - self.opt_E(X, g)) ** 2, axis=1) - self.c["kappa"] * np.sum(A[:, ke:] ** 2, axis=1)

    def hidden(self, X, A):
        if self.kind != "necessary":
            return np.zeros(len(X))
        dev = np.sum((A[:, self.c["KE"]:] - self.opt_O(X)) ** 2, axis=1)
        return self.c["H"] * (np.exp(-dev / (2.0 * self.c["s"] ** 2)) - 1.0)

    def welfare(self, X, A, g):
        return float(np.mean(self.visible(X, A, g) + self.hidden(X, A)))

    def live(self, X, A, g, rng):
        """Per-trial visible outcome; the hidden outcome only as one noisy verdict on the whole round (I:1)."""
        v = self.visible(X, A, g) + rng.normal(0.0, self.c["sigma_v"], len(X))
        verdict = float(np.mean(self.hidden(X, A)) + rng.normal(0.0, self.c["sigma_verdict"]))
        return v, verdict

    def normalized(self, X, A, g):
        w = self.welfare(X, A, g)
        wf = self.welfare(X, np.tile(self.fool, (len(X), 1)), g)
        wo = self.welfare(X, self.optimum(X, g), g)
        return (w - wf) / (wo - wf)


def env_reset(rng):
    """One-step episode in a fresh physician's world (necessary kind); the state carries the context."""
    w = World(rng, "necessary", CFG)
    return {"world": w, "x": w.contexts(rng, 1), "g": 0}


def env_step(state, action, rng):
    """Reward is the visible outcome only; the hidden part is returned in info for evaluation."""
    w, X, A = state["world"], state["x"], np.atleast_2d(action)
    v = float(w.visible(X, A, state["g"])[0] + rng.normal(0.0, w.c["sigma_v"]))
    return dict(state, x=w.contexts(rng, 1)), v, True, {"hidden": float(w.hidden(X, A)[0])}


def row_hashes(X):
    return {hashlib.sha1(np.round(r, 12).tobytes()).hexdigest() for r in X}


# ----------------------------------------------------------------------------- model
def mlp_init(rng, sizes, prefix, out_scale=1.0):
    P = {}
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        scale = out_scale if i == len(sizes) - 2 else 1.0
        P["%s.W%d" % (prefix, i)] = rng.normal(0.0, scale / math.sqrt(a), (a, b))
        P["%s.b%d" % (prefix, i)] = np.zeros(b)
    return P


def mlp_fwd(P, prefix, X, L):
    hs, h = [X], X
    for i in range(L):
        z = h @ P["%s.W%d" % (prefix, i)] + P["%s.b%d" % (prefix, i)]
        h = np.tanh(z) if i < L - 1 else z
        hs.append(h)
    return h, hs


def mlp_bwd(P, prefix, hs, d, L):
    g = {}
    for i in reversed(range(L)):
        g["%s.W%d" % (prefix, i)] = hs[i].T @ d
        g["%s.b%d" % (prefix, i)] = d.sum(axis=0)
        d = d @ P["%s.W%d" % (prefix, i)].T
        if i > 0:
            d = d * (1.0 - hs[i] ** 2)
    return g, d


def add_decay(P, g, prefix, wd):
    loss = 0.0
    for k in list(g):
        if k.startswith(prefix + ".W"):
            loss += wd * float(np.sum(P[k] ** 2))
            g[k] = g[k] + 2.0 * wd * P[k]
    return loss


def build_model(in_dim, out_dim, task_type, rng, kind="halevi", **cfg):
    c = dict(CFG)
    c.update(cfg)
    c["K"] = out_dim
    P = {}
    if kind == "halevi":
        P.update(mlp_init(rng, [in_dim, c["h_masorah"], out_dim], "masorah"))
        P.update(mlp_init(rng, [in_dim, c["h_tiqqun"], out_dim], "tiqqun", out_scale=0.05))
    else:
        P.update(mlp_init(rng, [in_dim, c["h_actor"], out_dim], "actor"))
    # dhawq trunk reads the context only; heads: V, verdict, mu (K), rho (K). Small output layer, because
    # the verdict head is constrained only in its per-round mean (no attribution, Kuzari I:1).
    P.update(mlp_init(rng, [in_dim, c["h_critic"], c["h_critic"], 2 + 2 * out_dim], "dhawq", out_scale=0.1))
    return {"kind": kind, "params": P, "cfg": c, "ko": {}, "critic_trained": False,
            "in_dim": in_dim, "out_dim": out_dim, "task_type": task_type}


def critic_heads(S, X):
    """Quadratic-advantage critic (NAF form, Gu et al. 2016, diagonal): Q(x,a) = V(x) + Vh(x)
    - sum_k p_k(x) (a_k - mu_k(x))^2. mu is the deed the learner derives; p is how much it cares."""
    out, hs = mlp_fwd(S, "dhawq", X, 3)
    K = (out.shape[1] - 2) // 2
    return out, hs, out[:, 2:2 + K], softplus(out[:, 2 + K:])


def critic_q_grad(S, X, A, need_grad=True):
    """Critic's estimate of whole welfare (visible part + verdict part) and its gradient w.r.t. the deed."""
    out, _, mu, p = critic_heads(S, X)
    r = A - mu
    Q = out[:, 0] + out[:, 1] - np.sum(p * r * r, axis=1)
    return Q, (-2.0 * p * r if need_grad else None)


def sensitivities(fn, X, h):
    """Secant response of a deed-valued map to each circumstance moved alone by +-h: shape (n, d, K)."""
    n, d = X.shape
    E = np.eye(d) * h
    up = fn((X[:, None, :] + E[None]).reshape(n * d, d)).reshape(n, d, -1)
    dn = fn((X[:, None, :] - E[None]).reshape(n * d, d)).reshape(n, d, -1)
    return (up - dn) / (2.0 * h)


def concordance(JT, Jm, tau, variant="concord", eps=1e-3):
    """Per-component agreement of received and derived sensitivities (Lin's concordance on the profile,
    without the level term): r = 2<a,b>/(|a|^2+|b|^2). Level offsets never enter; size and direction do.
    Agreement up to tau is what an unrelated profile can reach by chance, so it licenses nothing."""
    dot = np.sum(JT * Jm, axis=1)
    nT, nm = np.sum(JT * JT, axis=1), np.sum(Jm * Jm, axis=1)
    if variant == "cosine":
        r = dot / (np.sqrt(nT * nm) + eps)
    else:
        r = 2.0 * dot / (nT + nm + eps)
    g = (r - tau) / (1.0 - tau)
    if variant == "unclipped":
        return g
    g = np.clip(g, 0.0, 1.0)
    return 1.0 - g if variant == "inverted" else g


def level_gate(T, mu, c):
    """Pilot design kept only as the negative control of C6.1: agreement of levels, not of profiles."""
    return np.clip(1.0 - (T - mu) ** 2 / c["nbr_h"] ** 2, 0.0, 1.0)


def license(model, X):
    """reshut: may component k of the received deed be amended in context x? Only if the learner's own
    critic re-derives how that measure must change with each circumstance (Kuzari I:79)."""
    ko = model["ko"].get("reshut")
    if ko == "zero" or not model["critic_trained"]:
        return np.zeros((len(X), model["out_dim"]))
    if ko == "one":
        return np.ones((len(X), model["out_dim"]))
    P, h = model["params"], model["cfg"]["nbr_h"]
    JT = sensitivities(lambda Z: mlp_fwd(P, "masorah", Z, 2)[0], X, h)
    Jm = sensitivities(lambda Z: critic_heads(P, Z)[2], X, h)
    g = concordance(JT, Jm, model["cfg"]["tau"])
    if ko == "mean":
        g = np.repeat(g.mean(axis=1, keepdims=True), g.shape[1], axis=1)
    return g


def act(model, X):
    P = model["params"]
    if model["kind"] != "halevi":
        return mlp_fwd(P, "actor", X, 2)[0]
    T = mlp_fwd(P, "masorah", X, 2)[0]
    if model["ko"].get("tiqqun") == "zero":
        return T
    U = mlp_fwd(P, "tiqqun", X, 2)[0]
    return T + license(model, X) * U


# ----------------------------------------------------------------------------- losses (analytic gradients)
def loss_clone(P, prefix, X, A, wd):
    out, hs = mlp_fwd(P, prefix, X, 2)
    diff = out - A
    g, _ = mlp_bwd(P, prefix, hs, 2.0 * diff / len(X), 2)
    return float(np.sum(diff ** 2) / len(X)) + add_decay(P, g, prefix, wd), g


def loss_critic(P, X, A, v, rid, verdicts, wd):
    """Per-trial visible outcome fits V - sum p (a-mu)^2; the verdict head fits only the round's mean."""
    out, hs, mu, p = critic_heads(P, X)
    n, R, K = len(X), len(verdicts), mu.shape[1]
    r = A - mu
    ev = out[:, 0] - np.sum(p * r * r, axis=1) - v
    counts = np.maximum(np.bincount(rid, minlength=R), 1)
    eh = np.bincount(rid, weights=out[:, 1], minlength=R) / counts - verdicts
    d = np.zeros_like(out)
    dv = 2.0 * ev / n
    d[:, 0] = dv
    d[:, 1] = (2.0 * eh / R)[rid] / counts[rid]
    d[:, 2:2 + K] = dv[:, None] * 2.0 * p * r
    d[:, 2 + K:] = -dv[:, None] * r * r * 0.5 * (1.0 + np.tanh(0.5 * out[:, 2 + K:]))   # softplus' = sigmoid
    g, _ = mlp_bwd(P, "dhawq", hs, d, 3)
    return float(np.mean(ev ** 2) + np.mean(eh ** 2)) + add_decay(P, g, "dhawq", wd), g


def loss_amend(P, S, X, T, gl, beta, wd):
    """tiqqun: amend the received deed by ascent on the (frozen) critic, only as far as reshut licenses."""
    U, hs = mlp_fwd(P, "tiqqun", X, 2)
    Q, dQ = critic_q_grad(S, X, T + gl * U)
    n = len(X)
    g, _ = mlp_bwd(P, "tiqqun", hs, -gl * dQ / n + 2.0 * beta * U / n, 2)
    L = -float(Q.mean()) + beta * float(np.sum(U ** 2)) / n
    return L + add_decay(P, g, "tiqqun", wd), g


def loss_actor(P, S, X, A, mode, alpha, wd):
    """Baselines: 'td3bc' (Fujimoto and Gu 2021), 'derive' (the philosopher's rival), 'clone' (imitator)."""
    pi, hs = mlp_fwd(P, "actor", X, 2)
    n = len(X)
    if mode == "clone":
        d, L = 2.0 * (pi - A) / n, float(np.sum((pi - A) ** 2) / n)
    else:
        Q, dQ = critic_q_grad(S, X, pi)
        if mode == "derive":
            d, L = -dQ / n, -float(Q.mean())
        else:
            lam = alpha / (float(np.mean(np.abs(critic_q_grad(S, X, A, need_grad=False)[0]))) + 1e-6)
            d = -lam * dQ / n + 2.0 * (pi - A) / n
            L = -lam * float(Q.mean()) + float(np.sum((pi - A) ** 2) / n)
    g, _ = mlp_bwd(P, "actor", hs, d, 2)
    return L + add_decay(P, g, "actor", wd), g


def loss_and_grads(model, batch):
    """Sum of the module losses present in the batch; each trainable tensor sits in one term only.
    Y: cloning of the received deed. gl: licensed amendment (halevi). mode: baseline actor. v: critic.
    S is always a frozen copy of the critic, so no gradient leaks from the policy terms into it."""
    P, c = model["params"], model["cfg"]
    halevi = model["kind"] == "halevi"
    grads = {k: np.zeros_like(v) for k, v in P.items()}
    parts = []
    if "Y" in batch:
        parts.append(loss_clone(P, "masorah" if halevi else "actor", batch["X"], batch["Y"], c["wd"]))
    if "gl" in batch and halevi:
        parts.append(loss_amend(P, batch["S"], batch["X"], batch["T"], batch["gl"], c["beta_amend"], c["wd"]))
    if "mode" in batch and not halevi:
        parts.append(loss_actor(P, batch["S"], batch["X"], batch["A"], batch["mode"], c["td3_alpha"], c["wd"]))
    if "v" in batch:
        parts.append(loss_critic(P, batch["Xe"], batch["Ae"], batch["v"], batch["rid"], batch["verdicts"], c["wd"]))
    total = 0.0
    for L, g in parts:
        total += L
        for k, v in g.items():
            grads[k] = grads[k] + v
    for k in MUTANTS.get(MUTANT["name"], {}).get("drop", ()):
        if k in grads:
            grads[k] = np.zeros_like(grads[k])
    return total, grads


# ----------------------------------------------------------------------------- registries
MUTANTS = {
    "sign_flipped_update": {"breaks": "optimizer ascends the loss", "detect": "C3"},
    "zero_learning_rate": {"breaks": "no parameter moves", "detect": "C3"},
    "drop_grad_masorah_W0": {"breaks": "first cloning layer gets no gradient", "detect": "C1", "drop": ("masorah.W0",)},
    "drop_grad_dhawq_W1": {"breaks": "critic hidden layer gets no gradient", "detect": "C1", "drop": ("dhawq.W1",)},
}


CRITIC_ROLE = "quadratic-advantage critic; per-trial head plus a head fitted only to per-round means"


def modules(model):
    P = model["params"]
    by = lambda p: {k: v for k, v in P.items() if k.startswith(p + ".")}
    if model["kind"] != "halevi":
        return {"actor": (by("actor"), "deterministic policy MLP", False),
                "dhawq": (by("dhawq"), CRITIC_ROLE, False)}
    return {"masorah": (by("masorah"), "behaviour-cloning pathway (received measure)", False),
            "tiqqun": (by("tiqqun"), "critic-ascent correction pathway (amendment)", False),
            "dhawq": (by("dhawq"), CRITIC_ROLE, False),
            "reshut": ({}, "gate: concordance of finite-difference sensitivities, per component and context", True)}


KNOCKOUTS = {"reshut": ("zero", "one", "mean"), "tiqqun": ("zero",)}


def knockout(model, name, mode):
    """Copy with reshut replaced by zero, one or its mean over components, or with tiqqun replaced by zero."""
    if mode not in KNOCKOUTS.get(name, ()):
        raise ValueError("available knockouts: %s" % KNOCKOUTS)
    return dict(model, params={k: v.copy() for k, v in model["params"].items()}, ko=dict(model["ko"], **{name: mode}))


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


def hidden_states(model, X):
    P = model["params"]
    if model["kind"] != "halevi":
        return {"actor_hidden": mlp_fwd(P, "actor", X, 2)[1][1]}
    T, hs = mlp_fwd(P, "masorah", X, 2)
    return {"masorah_hidden": hs[1], "received_deed": T, "licence": license(model, X)}


predict = act   # interface name: inference is the deed


# ----------------------------------------------------------------------------- training
def frozen_critic(model):
    return {k: v.copy() for k, v in model["params"].items() if k.startswith("dhawq.")}


def run_phase(model, keys, sampler, steps, rng, hist):
    """One optimizer over a key subset; the sampler draws a batch dict from the phase's data."""
    opt = Adam({k: model["params"][k] for k in keys}, model["cfg"]["lr"])
    for _ in range(steps):
        L, g = loss_and_grads(model, sampler(rng))
        sub = {k: g[k] for k in keys}
        clip_by_global_norm(sub, model["cfg"]["clip"])
        opt.step(model["params"], sub, keys)
        check_finite(np.asarray(L))
        hist.append(float(L))


def keys_of(model, prefix):
    return [k for k in model["params"] if k.startswith(prefix + ".")]


def fit(model, data, budget, rng):
    """Fixed update budget per phase. data: X, Y (received deeds); optionally the trials Xe, Ae, v, rid,
    verdicts. Phases: clone the received deed; fit the outcome model; then improve (licensed amendment
    for halevi, TD3+BC or pure derivation for the baselines)."""
    c, halevi = model["cfg"], model["kind"] == "halevi"
    B = budget if isinstance(budget, dict) else {"clone": budget, "critic": budget, "improve": budget}
    hist = {"clone": [], "critic": [], "improve": []}
    X, Y, nb = data["X"], data["Y"], c["batch"]
    pick = lambda r, n, m: r.integers(0, n, m)
    if B.get("clone", 0):
        def s_clone(r):
            i = pick(r, len(X), nb)
            return {"X": X[i], "Y": Y[i]}
        run_phase(model, keys_of(model, "masorah" if halevi else "actor"), s_clone, B["clone"], rng, hist["clone"])
    if "v" not in data or not B.get("critic", 0):
        return hist
    Xe, Ae, v, rid, ver = data["Xe"], data["Ae"], data["v"], data["rid"], data["verdicts"]

    def s_critic(r):
        i = pick(r, len(Xe), 2 * nb)
        return {"Xe": Xe[i], "Ae": Ae[i], "v": v[i], "rid": rid[i], "verdicts": ver}
    run_phase(model, keys_of(model, "dhawq"), s_critic, B["critic"], rng, hist["critic"])
    model["critic_trained"] = True
    S = frozen_critic(model)
    if halevi:
        if model["ko"].get("tiqqun") == "zero":
            return hist
        T = mlp_fwd(model["params"], "masorah", X, 2)[0]
        G = license(model, X)             # reshut is a test, not a trained weight: fixed during amendment

        def s_amend(r):
            i = pick(r, len(X), nb)
            return {"X": X[i], "T": T[i], "gl": G[i], "S": S}
        run_phase(model, keys_of(model, "tiqqun"), s_amend, B["improve"], rng, hist["improve"])
    elif model.get("mode", "clone") != "clone":
        def s_actor(r):
            i = pick(r, len(X), nb)
            return {"X": X[i], "A": Y[i], "S": S, "mode": model["mode"]}
        run_phase(model, keys_of(model, "actor"), s_actor, B["improve"], rng, hist["improve"])
    return hist


def live(model, world, g, rng, c):
    """Rounds of exploratory trials: per-trial visible outcome, one verdict per round (the dream, I:1)."""
    Xs, As, vs, rid, ver = [], [], [], [], []
    for r in range(c["rounds"]):
        X = world.contexts(rng, c["n_round"])
        A = act(model, X) + rng.normal(0.0, c["sigma_explore"], (len(X), c["K"]))
        v, verdict = world.live(X, A, g, rng)
        Xs.append(X), As.append(A), vs.append(v), rid.append(np.full(len(X), r)), ver.append(verdict)
    return {"Xe": np.concatenate(Xs), "Ae": np.concatenate(As), "v": np.concatenate(vs),
            "rid": np.concatenate(rid), "verdicts": np.asarray(ver)}


ARMS = {"halevi": ("halevi", None, {}), "td3bc": ("baseline", "td3bc", {}), "derive": ("baseline", "derive", {}),
        "clone": ("baseline", "clone", {}), "ko_reshut_mean": ("halevi", None, {"reshut": "mean"}),
        "ko_reshut_one": ("halevi", None, {"reshut": "one"}), "ko_tiqqun_zero": ("halevi", None, {"tiqqun": "zero"})}


def make_arm(arm, rng, c):
    kind, mode, ko = ARMS[arm]
    m = build_model(c["dc"] + c["dn"], c["K"], "episodic_control", rng, kind=kind, **c)
    m["ko"], m["mode"] = dict(ko), mode
    return m


def run_chain(arm, world, seed, c, evalsets, train_log=None):
    """Generation g clones generation g-1's deeds (with copy noise), lives, learns, amends, then teaches.
    Every arm draws the same contexts and noise per generation (paired by SeedSequence)."""
    teacher = world.physician
    out = {"held": [], "shifted": [], "retain": [], "lic_E": [], "lic_O": []}
    kids = np.random.SeedSequence([seed, 279]).spawn(c["G"])
    ke = c["KE"]
    for g in range(1, c["G"] + 1):
        r_data, r_init, r_train = [np.random.default_rng(s) for s in kids[g - 1].spawn(3)]
        Xd = world.contexts(r_data, c["n_demo"])
        Ad = teacher(Xd) + r_data.normal(0.0, c["sigma_copy"], (len(Xd), c["K"]))
        model = make_arm(arm, r_init, c)
        steps = {"clone": c["steps_bc"], "critic": c["steps_critic"],
                 "improve": c["steps_amend"] if model["kind"] == "halevi" else c["steps_actor"]}
        h1 = fit(model, {"X": Xd, "Y": Ad}, {"clone": steps["clone"]}, r_train)
        trials = live(model, world, g, r_data, c)
        h2 = fit(model, dict(trials, X=Xd, Y=Ad), {"critic": steps["critic"], "improve": steps["improve"]}, r_train)
        if train_log is not None:
            train_log.append({"clone": h1["clone"], "critic": h2["critic"], "improve": h2["improve"],
                              "Xtrain": [Xd, trials["Xe"]]})
        for split in ("held", "shifted"):
            X = evalsets[split]
            A = act(model, X)
            check_finite(A)
            out[split].append(world.normalized(X, A, g))
        X = evalsets["held"]
        A = act(model, X)
        out["retain"].append(float(np.mean(np.abs(A[:, ke:] - world.opt_O(X)))))
        if model["kind"] == "halevi" and model["critic_trained"]:
            L = license(model, X)
            out["lic_E"].append(float(L[:, :ke].mean())), out["lic_O"].append(float(L[:, ke:].mean()))
        teacher = (lambda m: (lambda Z: act(m, Z)))(model)
    return out


def world_and_evals(seed, kind, c):
    r = np.random.default_rng(np.random.SeedSequence([seed, 1075]))
    w = World(r, kind, c)
    return w, {"held": w.contexts(r, c["n_eval"]), "shifted": w.contexts(r, c["n_eval"], shifted=True)}


def tune(c, val_seed=1000):
    """Equal two-point grids on a validation world only (never on the test seeds)."""
    w, ev = world_and_evals(val_seed, "necessary", c)
    res = {}
    for alpha in (1.0, 2.5):
        res[("td3_alpha", alpha)] = run_chain("td3bc", w, val_seed, dict(c, td3_alpha=alpha), ev)["held"][-1]
    for hh in (0.3, 0.6):
        res[("nbr_h", hh)] = run_chain("halevi", w, val_seed, dict(c, nbr_h=hh), ev)["held"][-1]
    best_a = max((1.0, 2.5), key=lambda a: res[("td3_alpha", a)])
    best_h = max((0.3, 0.6), key=lambda x: res[("nbr_h", x)])
    return {"td3_alpha": best_a, "nbr_h": best_h}, {"%s=%s" % k: round(v, 4) for k, v in res.items()}


# ----------------------------------------------------------------------------- correctness tests
def probe_batch(c, kind, seed=7):
    """A small fixed batch holding every loss term of one arm, for the gradient check."""
    w, _ = world_and_evals(seed, "necessary", c)
    r = np.random.default_rng(seed + 1)
    X = w.contexts(r, 48)
    A = w.physician(X) + r.normal(0.0, 0.05, (48, c["K"]))
    Xe = w.contexts(r, 64)
    Ae = w.physician(Xe) + r.normal(0.0, 0.5, (64, c["K"]))
    v = w.visible(Xe, Ae, 1) + r.normal(0.0, 0.1, 64)
    m = make_arm(kind, r, c)
    m["critic_trained"] = True
    b = {"Xe": Xe, "Ae": Ae, "v": v, "rid": np.repeat([0, 1], 32), "verdicts": np.array([-0.4, -0.7]),
         "S": frozen_critic(m), "X": X}
    if m["kind"] == "halevi":
        b.update(Y=A, T=mlp_fwd(m["params"], "masorah", X, 2)[0], gl=r.uniform(0.0, 1.0, A.shape))
    else:
        b.update(A=A, mode=m["mode"])
    return m, b


def term_batches(m, b):
    """Split the probe batch into its loss terms; each tensor belongs to exactly one term."""
    spec = [("Y", ("X", "Y"), "masorah" if m["kind"] == "halevi" else "actor"), ("gl", ("X", "T", "gl", "S"), "tiqqun"),
            ("mode", ("X", "A", "S", "mode"), "actor"), ("v", ("Xe", "Ae", "v", "rid", "verdicts"), "dhawq")]
    return [({k: b[k] for k in ks}, keys_of(m, pre)) for flag, ks, pre in spec if flag in b]


def c1_gradcheck(c, rng, arms=("halevi", "td3bc", "derive"), after=60):
    """Every tensor of every gradient-trained arm, term by term (so finite-difference noise from the other
    terms does not swamp small gradients), at initialization and after `after` Adam steps on all terms."""
    worst, checked, total = 0.0, 0, 0
    for arm in arms:
        m, b = probe_batch(c, arm)
        for stage in ("init", "after"):
            if stage == "after":
                opt = Adam(m["params"], c["lr"])
                for _ in range(after):
                    _, g = loss_and_grads(m, b)
                    clip_by_global_norm(g, c["clip"])
                    opt.step(m["params"], g)
            for sub, keys in term_batches(m, b):
                errs = gradcheck(lambda P: loss_and_grads(m, sub), m["params"], keys, rng)
                worst = max(worst, max(errs.values()))
                checked += len(errs)
        total += 2 * sum(len(p) for p, _, _ in modules(m).values())   # the registry must cover every tensor
    ok = worst <= MIND_CARD["thresholds"]["gradcheck_rel_error"] and checked == total
    return ok, worst, checked, total


def c3_learning(c, seed):
    """One generation of the licensed learner: losses must fall and it must beat the fool's dose."""
    th = MIND_CARD["thresholds"]
    w, ev = world_and_evals(seed, "necessary", dict(c, G=1))
    log = []
    out = run_chain("halevi", w, seed, dict(c, G=1), ev, train_log=log)
    drop = lambda h: 1.0 - np.mean(h[-20:]) / np.mean(h[:20])
    d_clone, d_critic = drop(log[0]["clone"]), drop(log[0]["critic"])
    held = out["held"][0]
    ok = (d_clone >= th["loss_drop_fraction"] and d_critic >= th["critic_loss_drop_fraction"]
          and held >= th["margin_over_trivial"])
    return ok, "clone loss drop %.3f, critic loss drop %.3f, held-out welfare %.3f (trivial 0, margin %.1f)" % (
        d_clone, d_critic, held, th["margin_over_trivial"])


def c2_determinism(c, seed):
    runs = []
    for _ in range(2):
        w, ev = world_and_evals(seed, "necessary", dict(c, G=1))
        log = []
        out = run_chain("halevi", w, seed, dict(c, G=1), ev, train_log=log)
        runs.append((out["held"] + out["shifted"], log[0]["clone"] + log[0]["critic"] + log[0]["improve"]))
    e = [env_step(env_reset(np.random.default_rng(seed)), np.ones(c["K"]), np.random.default_rng(seed))[1:3] for _ in "ab"]
    same = runs[0][0] == runs[1][0] and runs[0][1] == runs[1][1] and e[0] == e[1]
    finite = bool(np.all(np.isfinite(runs[0][0])) and np.all(np.isfinite(runs[0][1])))
    return same and finite, "identical metrics, %d losses and env_step rewards across two runs; all finite" % len(
        runs[0][1])


def c4_shuffled(c, seed, rng):
    """Deeds permuted against contexts and trial outcomes permuted: welfare must stay near the fool's."""
    w, ev = world_and_evals(seed, "necessary", c)
    r = np.random.default_rng(np.random.SeedSequence([seed, 4]))
    Xd = w.contexts(r, c["n_demo"])
    Ad = w.physician(Xd)[rng.permutation(c["n_demo"])]
    m = make_arm("halevi", r, c)
    fit(m, {"X": Xd, "Y": Ad}, {"clone": c["steps_bc"]}, r)
    tr = live(m, w, 1, r, c)
    tr["v"] = tr["v"][rng.permutation(len(tr["v"]))]
    tr["verdicts"] = rng.permutation(tr["verdicts"])
    fit(m, dict(tr, X=Xd, Y=Ad), {"critic": c["steps_critic"], "improve": c["steps_amend"]}, r)
    held = w.normalized(ev["held"], act(m, ev["held"]), 1)
    band = MIND_CARD["thresholds"]["shuffled_band"]
    return abs(held) <= band, "held-out welfare %.3f with shuffled deeds and outcomes (band +-%.2f)" % (held, band)


def c5_mutants(c, rng):
    caught = {}
    keep = MUTANT["name"]
    for name in MUTANTS:
        MUTANT["name"] = name
        c1_ok = c1_gradcheck(c, rng, arms=("halevi", "td3bc"), after=0)[0]
        c3_ok = c3_learning(c, 3)[0]
        caught[name] = ("C1" if not c1_ok else "") + ("C3" if not c3_ok else "")
    MUTANT["name"] = keep
    return caught


def random_learner(c, r):
    """A licensed learner with random weights of random scale and a 'trained' critic flag."""
    m = make_arm("halevi", r, c)
    for k in m["params"]:
        m["params"][k] = m["params"][k] * r.uniform(0.3, 3.0) + (r.normal(0, 0.3, m["params"][k].shape)
                                                                 if ".b" in k else 0.0)
    m["critic_trained"] = True
    return m


def c6_properties(c, rng, trials=150):
    """Invariants of reshut, each searched at random and each with a negative control that must fail."""
    tau, K = c["tau"], c["K"]
    viol = {"indiff": 0.0, "indiff_neg": 0.0, "mono": 0.0, "mono_neg": 0.0, "size": 0.0, "size_neg": 0.0,
            "bounds": 0.0, "bounds_neg": 0.0, "defn": 0.0, "self": 0.0}
    for _ in range(trials):
        m = random_learner(c, rng)
        X = rng.normal(0.0, rng.uniform(0.5, 2.0), (16, c["dc"] + c["dn"]))
        g = hidden_states(m, X)["licence"]
        viol["bounds"] = max(viol["bounds"], float(np.max(-g)), float(np.max(g - 1.0)))
        # C6.1 a critic whose derived deed barely follows circumstances licenses nothing
        P = m["params"]
        P["dhawq.W2"][:, 2:2 + K] *= 1e-3
        viol["indiff"] = max(viol["indiff"], float(np.max(license(m, X))))
        _, _, mu, _ = critic_heads(P, X)
        viol["indiff_neg"] = max(viol["indiff_neg"], float(np.max(level_gate(mlp_fwd(P, "masorah", X, 2)[0], mu, c))))
        # C6.5 definition check (not evidence): licence zero means the received deed passes unchanged
        z = knockout(m, "reshut", "zero")
        viol["defn"] = max(viol["defn"], float(np.max(np.abs(act(z, X) - mlp_fwd(P, "masorah", X, 2)[0]))))
        # C6.2 an unexplained profile component can only lower the licence; C6.3 size matters
        a = rng.normal(0.0, 1.0, (1, c["dc"] + c["dn"], 1))
        b = rng.normal(0.0, 1.0, a.shape)
        b -= a * np.sum(a * b) / np.sum(a * a)
        lam = np.linspace(0.0, 3.0, 13)
        for variant, key in (("concord", "mono"), ("inverted", "mono_neg")):
            gs = np.array([concordance(a + l * b, a, tau, variant)[0, 0] for l in lam])
            viol[key] = max(viol[key], float(np.max(np.diff(gs))))
        for variant, key in (("concord", "size"), ("cosine", "size_neg")):
            far = [concordance(a, s * a, tau, variant)[0, 0] for s in (0.1, 0.2, 0.25, 4.0, 5.0, 10.0)]
            viol[key] = max(viol[key], float(np.max(far)))
        viol["self"] = max(viol["self"], 1.0 - float(concordance(a, a, tau)[0, 0]))
        viol["bounds_neg"] = max(viol["bounds_neg"], float(-concordance(a, -a, tau, "unclipped")[0, 0]))
    tol, res = 1e-9, {}
    res["C6.1"] = ("indifference_no_licence", viol["indiff"] <= tol and viol["indiff_neg"] > 0.1,
                   "indifferent critic: max licence %.2e; level-gate control reaches %.3f" % (
                       viol["indiff"], viol["indiff_neg"]))
    res["C6.2"] = ("unexplained_profile_lowers", viol["mono"] <= tol and viol["mono_neg"] > 0.1,
                   "unexplained profile: max rise %.2e; inverted control rises %.3f" % (viol["mono"], viol["mono_neg"]))
    res["C6.3"] = ("size_error_no_licence", viol["size"] <= tol and viol["self"] < 0.01 and viol["size_neg"] > 0.1,
                   "4x size error: max licence %.2e (exact agreement %.4f); cosine control %.3f" % (
                       viol["size"], 1.0 - viol["self"], viol["size_neg"]))
    res["C6.4"] = ("licence_bounds", viol["bounds"] <= tol and viol["bounds_neg"] > 0.1,
                   "licence within [0,1] (max excess %.2e); unclipped control reaches %.3f below 0" % (
                       viol["bounds"], viol["bounds_neg"]))
    res["C6.5"] = ("definition_check", viol["defn"] <= tol, "not evidence: zero licence passes the received deed, "
                   "max diff %.1e" % viol["defn"])
    return res


def c7_split(log, evalsets):
    train = set()
    for entry in log:
        for X in entry["Xtrain"]:
            train |= row_hashes(X)
    test = row_hashes(evalsets["held"]) | row_hashes(evalsets["shifted"])
    return not (train & test), "%d training contexts, %d evaluation contexts, %d shared" % (
        len(train), len(test), len(train & test))


# ----------------------------------------------------------------------------- hypotheses
NEC_ARMS = ["halevi", "td3bc", "derive", "clone", "ko_reshut_mean", "ko_reshut_one", "ko_tiqqun_zero"]
ACC_ARMS = ["halevi", "derive", "td3bc"]


def run_seed(seed, c, log=None):
    w, ev = world_and_evals(seed, "necessary", c)
    nec = {a: run_chain(a, w, seed, c, ev, train_log=log if a == "halevi" else None) for a in NEC_ARMS}
    wa, eva = world_and_evals(seed, "accretion", c)
    acc = {a: run_chain(a, wa, seed, c, eva) for a in ACC_ARMS}
    return {"nec": nec, "acc": acc, "evalsets": ev}


def hypothesis_diffs(r):
    n, a = r["nec"], r["acc"]
    last = lambda arm, split, world=n: world[arm][split][-1]
    return {"H-SIG": last("halevi", "shifted") - last("td3bc", "shifted"),
            "H-NEC": last("ko_reshut_mean", "shifted") - last("ko_tiqqun_zero", "shifted"),
            "H-BLIND": last("halevi", "held", a) - last("derive", "held", a),
            "H-RIVAL": last("halevi", "held") - last("derive", "held"),
            "H-CHAIN": (last("halevi", "held") - last("td3bc", "held"))
                       - (n["halevi"]["held"][0] - n["td3bc"]["held"][0])}


def evaluate(results, rng, evaluated):
    diffs = [hypothesis_diffs(r) for r in results]
    hyp, kos = [], []
    for h in MIND_CARD["hypotheses"]:
        d = [x[h["id"]] for x in diffs]
        if evaluated:
            mean, lo, hi = paired_bootstrap(d, rng)
            verdict = verdict_of(mean, lo, hi, h["direction"], h["mesi"])
        else:
            mean, lo, hi, verdict = float(np.mean(d)), None, None, "not evaluated"
        hyp.append({"id": h["id"], "metric": h["metric"] + " (" + h["split"] + ")", "mean_diff": mean,
                    "ci95": [lo, hi], "mesi": h["mesi"], "n_seeds": len(d), "verdict": verdict})
    for arm, mod, sig in (("ko_reshut_mean", "reshut->mean", True), ("ko_reshut_one", "reshut->one", True),
                          ("ko_tiqqun_zero", "tiqqun->zero", False)):
        d = [r["nec"][arm]["shifted"][-1] - r["nec"]["halevi"]["shifted"][-1] for r in results]
        mean, lo, hi = paired_bootstrap(d, rng) if evaluated else (float(np.mean(d)), None, None)
        kos.append({"module": mod, "signature": sig, "metric_change": mean, "ci95": [lo, hi]})
    return hyp, kos


# ----------------------------------------------------------------------------- report
def descriptive_lines(results, c):
    L = ["descriptive (mean over seeds; normalized welfare, 1 = optimum, 0 = fool's dose):"]
    for world, arms in (("nec", NEC_ARMS), ("acc", ACC_ARMS)):
        L.append("  %s world, held-out by generation 1..%d:" % ({"nec": "necessary"}.get(world, "accretion"), c["G"]))
        for a in arms:
            held = np.mean([r[world][a]["held"] for r in results], axis=0)
            shf = np.mean([r[world][a]["shifted"] for r in results], axis=0)
            ret = np.mean([r[world][a]["retain"] for r in results], axis=0)
            L.append("    %-15s %s | shifted G %.3f | opaque error G %.3f" % (
                a, " ".join("%.3f" % x for x in held), shf[-1], ret[-1]))
    lic = [(np.mean(r["nec"]["halevi"]["lic_E"]), np.mean(r["nec"]["halevi"]["lic_O"])) for r in results]
    L.append("  licence (necessary world, all generations): explicable %.3f, opaque %.3f" % tuple(np.mean(lic, 0)))
    return L


def build_report(ctx):
    L = ["environment: python %s, numpy %s" % (sys.version.split()[0], np.__version__),
         "file: %s  mode: %s  mutant: %s" % (ctx["file"], ctx["mode"], MUTANT["name"]),
         "seeds: %s (validation seed 1000 for tuning)" % ctx["seeds"],
         "runtime_s: %.1f (budget %d)" % (ctx["runtime"], ctx["budget"]),
         "n_params: licensed learner %d, TD3+BC %d (ratio %.3f)" % ctx["n_params"],
         "tuning: %s" % ctx["tuning"],
         "gradcheck: %d/%d tensor checks, max rel error %.2e, at init and after 60 steps, %s" % (
             ctx["gc"][2], ctx["gc"][3], ctx["gc"][1], "passed" if ctx["gc"][0] else "FAILED"),
         "correctness:"]
    L += ["  %-5s %-26s %s  %s" % (t["id"], t["name"], "PASS" if t["passed"] else "FAIL", t["detail"])
          for t in ctx["correctness"]]
    mt = ctx["mutants"]
    L.append("mutants: %d/%d detected (score %.2f): %s" % (mt["detected"], mt["total"], mt["score"], ", ".join(
        "%s->%s" % (k, v or "missed") for k, v in mt["by"].items())))
    L.append("hypotheses (paired per-seed differences, 95% percentile bootstrap, 2000 resamples):")
    ci = lambda v: "[%+.4f, %+.4f]" % tuple(v) if v[0] is not None else "[n/a]"
    for h in ctx["hypotheses"]:
        L.append("  %-8s mean %+.4f  CI %s  mesi %.2f  n=%d  %s" % (
            h["id"], h["mean_diff"], ci(h["ci95"]), h["mesi"], h["n_seeds"], h["verdict"]))
    L.append("knockouts (shifted welfare at G, knockout minus full model):")
    for k in ctx["knockouts"]:
        L.append("  %-14s signature=%-5s change %+.4f  CI %s" % (k["module"], k["signature"], k["metric_change"],
                                                               ci(k["ci95"])))
    L += ctx["descriptive"]
    L.append("task types: %s" % ", ".join(TASK_TYPES))
    L.append("data bridge: %s" % ctx["bridge"])
    L.append("exit code: %d" % ctx["exit_code"])
    return L


def data_bridge(path, rng):
    """Optional real-data check (SECOM from the UCI repository, saved locally as CSV; last column = target):
    the cloning pathway alone, against the mean predictor, on a fixed 80/20 split."""
    if not path:
        return "skipped (no --data PATH given)"
    if not os.path.exists(path):
        return "skipped (file not found: %s)" % path
    D = np.genfromtxt(path, delimiter=",")
    D = D[~np.all(np.isnan(D), axis=1)]
    D = D[~np.isnan(D[:, -1])]
    X, y = D[:, :-1], D[:, -1:]
    idx = rng.permutation(len(X))
    tr, te = idx[: int(0.8 * len(X))], idx[int(0.8 * len(X)):]
    mu = np.nanmean(X[tr], axis=0)
    X = np.where(np.isnan(X), np.where(np.isnan(mu), 0.0, mu), X)
    sd = X[tr].std(axis=0) + 1e-8
    X = (X - X[tr].mean(axis=0)) / sd
    m = build_model(X.shape[1], 1, "vector_regression", rng)
    fit(m, {"X": X[tr], "Y": y[tr]}, {"clone": 400}, rng)
    mse = float(np.mean((predict(m, X[te]) - y[te]) ** 2))
    base = float(np.mean((y[tr].mean() - y[te]) ** 2))
    return "%s: %d rows, %d features; test MSE %.4f vs mean predictor %.4f" % (
        os.path.basename(path), len(D), X.shape[1], mse, base)


# ----------------------------------------------------------------------------- command line
def protocol(args):
    t0, quick = time.time(), args.quick
    c = dict(CFG, **(QUICK if quick else {}))
    budget = 20 if quick else 180
    rng = np.random.default_rng(args.seed)
    seeds = [args.seed] if quick else [args.seed + i for i in range(args.seeds)]
    print("seeds: %s  mutant: %s" % (seeds, MUTANT["name"]), flush=True)
    gc = c1_gradcheck(c, rng)
    corr = [("C1", "gradient_check", gc[0], "max rel error %.2e over %d/%d tensor checks" % gc[1:])]
    corr.append(("C2", "determinism_finiteness") + c2_determinism(c, args.seed))
    corr.append(("C3", "learning") + c3_learning(c, args.seed))
    corr.append(("C4", "shuffled_label_control") + c4_shuffled(c, args.seed, rng))
    by = c5_mutants(dict(c, **QUICK), rng)
    det = sum(bool(v) for v in by.values())
    corr.append(("C5", "mutant_detection", det == len(by), "%d/%d mutants caught" % (det, len(by))))
    corr += [(k,) + v for k, v in c6_properties(c, rng).items()]
    tcfg, tres = ({}, "skipped in --quick (defaults used)") if quick else tune(c)
    cr = dict(c, **tcfg)
    log = []
    results = [run_seed(s, cr, log if i == 0 else None) for i, s in enumerate(seeds)]
    corr.append(("C7", "split_integrity") + c7_split(log, results[0]["evalsets"]))
    hyp, kos = evaluate(results, rng, (not quick) and len(seeds) >= 5)
    runtime = time.time() - t0
    corr.append(("C8", "budget", runtime <= budget, "%.1f s of %d s" % (runtime, budget)))
    failed = [t[0] for t in corr if not t[2]]
    code = 0 if not failed else (3 if failed == ["C8"] else 1)
    r0 = np.random.default_rng(0)
    npar = (n_params(make_arm("halevi", r0, c)), n_params(make_arm("td3bc", r0, c)))
    ctx = {"file": os.path.basename(__file__), "mode": "quick" if quick else "full", "seeds": seeds,
           "runtime": runtime, "budget": budget, "n_params": npar + (npar[1] / npar[0],),
           "tuning": "%s chosen %s" % (tres, tcfg) if tcfg else tres, "gc": gc, "exit_code": code,
           "correctness": [{"id": a, "name": b, "passed": bool(p), "detail": d} for a, b, p, d in corr],
           "mutants": {"detected": det, "total": len(by), "score": det / len(by), "by": by},
           "hypotheses": hyp, "knockouts": kos, "descriptive": descriptive_lines(results, cr),
           "bridge": data_bridge(args.data, np.random.default_rng(args.seed))}
    report = {"schema_version": "1.0", "chapter": 279, "file": ctx["file"],
              "card_revision": MIND_CARD["card_revision"],
              "environment": {"python": sys.version.split()[0], "numpy": np.__version__}, "seeds": seeds,
              "runtime_s": round(runtime, 2), "n_params": npar[0],
              "gradcheck": {"tensors_checked": gc[2], "tensors_total": gc[3], "max_rel_error": gc[1],
                            "checked_at": ["init", "after_training_steps"], "passed": bool(gc[0])},
              "correctness": ctx["correctness"],
              "mutants": {k: ctx["mutants"][k] for k in ("detected", "total", "score")},
              "hypotheses": hyp, "knockouts": kos, "task_types": TASK_TYPES, "exit_code": code,
              "descriptive": {"lines": ctx["descriptive"], "tuning": ctx["tuning"], "mutants_by": by,
                              "n_params_baseline": npar[1], "data_bridge": ctx["bridge"]}}
    write_report(build_report(ctx), report, args.json)
    return code


def main():
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__),
                                 description="Chapter 0279: re-derivation-licensed transmission.")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--card", action="store_true")
    ap.add_argument("--mutant", type=str, default=None)
    ap.add_argument("--data", type=str, default=None)
    args = ap.parse_args()
    if args.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    if args.seeds < 1 or (args.mutant is not None and args.mutant not in MUTANTS):
        print("invalid usage: --seeds >= 1; mutants: %s" % ", ".join(MUTANTS), file=sys.stderr)
        return 2
    MUTANT["name"] = args.mutant
    try:
        return protocol(args)
    except NonFinite as exc:
        print("non-finite values: %s" % exc, file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())

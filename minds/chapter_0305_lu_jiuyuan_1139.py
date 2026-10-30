#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0305 · Lu Jiuyuan (Lu Xiangshan, 1139-1193)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 16, Minds 301-320 Available on Amazon https://www.amazon.com/gp/product/B0HNJQX9S4
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos 
# END ATTRIBUTION
"""
Great-first two-organ learner (xian li hu qi da zhe, "first establish what is great").

Thesis
    What a learner comes to understand from graded practice depends on which organ was
    established before the practice began: if a deep judging organ is first set up on
    grader-free right/wrong disputes, a fast shallow organ can later absorb the grader's
    taste without taking the principle with it; if the shallow organ comes first, it
    takes the shallow part of the principle and the deep organ never recovers it.

Evidence and provenance (belief: his own recorded sayings, letters and lectures survive)
    D1  "First establish what is great" was, by his own acceptance, his only technique
        (Yulu, Lu Jiuyuan ji juan 35; from Mencius 6A15: established first, the lesser
        organ cannot snatch it away; eyes and ears do not think and are pulled by things).
    D2  Goose Lake, 1175: first bring the original mind to light, then read broadly,
        against Zhu Xi's broad reading first and return to essentials later (Zhu Hengdao's
        record in the Nianpu, juan 36).
    D3  White Deer Grotto lecture, 1181: understanding follows practice and practice follows
        aim; examination success depends on skill and the examiners' likes and dislikes
        (Lu Jiuyuan ji juan 23).
    D4  The fan lawsuit at Fuyang (1172): the original mind is knowing the right as right and
        the wrong as wrong inside one live dispute (Nianpu, juan 36; Song shi 407).
    D5  "The mind is one; how could a person have two minds?" -- rejects reading the human
        mind and the Way-mind as two minds (Yulu, juan 34).
    D6  Knowing prices and telling good goods from bad is a real ability, but not the work
        of the mind (Yulu, juan 35).
    D7  Zhu Xi's objection (poem answering the Lu brothers): they do not believe the human
        world has an ancient and a present -- the critique grounding the blind spot.

Doctrine -> mechanism -> test
    D1 -> M1 establishment phase (root trained alone on disputes)  -> C3        -> H-EST, H-PRIOR
    D2 -> M1 order of phases (Zhu's order reversed as rival)       -> C3        -> H-RIVAL
    D3 -> M2 grade coupling g = tau*r + p (grader's taste explained
          by the small organ inside the grade logit)               -> C6.3      -> H-SIG
    D4 -> data: style-matched right/wrong disputes                  -> C6.2      -> H-SIG
    D5 -> M3 one root gives every rightness judgment; no controller -> C6.4 (def) -> H-NEC
    D6 -> M2 small organ: linear, fast, used for grades only        -> C6.1      -> H-NEC
    D7 -> blind world: a new consideration absent from old disputes -> C6.1      -> H-BLIND

Research question
    Robustness to spurious correlation and reward hacking: when the only abundant signal is
    a proxy grader whose scores mix the true target with the grader's own taste, does the
    ORDER in which a deep and a shallow expert are established decide whether the deep
    expert inherits the taste? The live analogue is whether values should be instilled
    before broad pretraining on rated data, after it, or mixed through it.

Closest prior art and the delta
    Product-of-experts debiasing trains a shallow or bias-only expert FIRST, freezes it,
    then fits the main model through the combined logit (Clark et al. 2019, arXiv
    1909.03683; Sanh et al. 2021, arXiv 2012.01300). Pretraining with human preferences
    conditions on reward-model scores from the start (Korbak et al. 2023, arXiv 2302.08582).
    Delta: the deep expert is established first, alone, on grader-free matched disputes;
    the shallow expert opens only afterwards, keeps learning, and never enters the
    rightness judgment. Related dynamics: gradient starvation (Pezeshki et al., arXiv
    2011.09468) and critical learning periods (Achille et al., arXiv 1711.08856).

Blind spot
    Lu held that the mind and its principle are the same in every age. When the present
    world adds a consideration that old disputes never showed, and that consideration is
    shallow, the fast organ files it with the grader's taste and the rightness judgment
    misses it: the great-first learner should do worse than a plain learner on exactly the
    cases the new consideration decides.

Task (generative process)
    latent case u ~ N(0, I_6); rightness v = 1.4*u0*u1 + 1.0*u2 (+ gamma*u5 in the changed world)
    essay x = [A u + noise (10 substance dims), style s ~ N(0, I_4), nuisance (3)]
    examiner grade = 1[v + 1.1 * (s . taste) + 0.3 * eps > 0]   (2000 graded essays)
    disputes: 200 pairs sharing style and nuisance, one act right (v > 0.15), one wrong
    held-out: fresh essays; shifted: style pushed +/-2 along the taste axis (a new fashion)
    changed world: gamma = 1.2 in essays, disputes drawn from the old world (u5 = 0)
    metric: accuracy of the rightness judgment (argmax of the root logits) against 1[v > 0]

Limits
    Synthetic, low-dimensional, full-batch; "rightness" is a latent of the generator, not a
    model of human morality or of Lu's mind. Hypotheses were fixed after a design pilot on
    a simpler prototype (see revision_log); verdicts are reported whatever they are.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 1,
    "revision_log": [
        "r1 (2026-09-24): written before the first execution of this file. A separate design "
        "pilot (not this file) showed that order alone without a shallow organ, and a per-example "
        "aim gate, gave no advantage and were dropped; it also suggested that the establishment "
        "phase may not beat simultaneous mixing (H-EST) and that Zhu's reversed order may win in a "
        "stable world (H-RIVAL). Both hypotheses are kept in the direction the doctrine predicts."
    ],
    "generation": {"template_version": "codeguidelines 1.0 (15 Sep 2026)", "generator": "Claude (Anthropic)",
                   "generator_version": "Claude Opus 5.5", "date": "2026-09-24"},
    "id": 304, "figure": "Lu Jiuyuan", "born": 1139, "died": 1193, "civilization": "Chinese (Song)",
    "provenance": "belief",
    "thesis": "What graded practice teaches depends on which organ was established first: a deep "
              "judging organ set up on grader-free disputes lets a shallow organ absorb the grader's "
              "taste without taking the principle.",
    "evidence": [
        {"id": "D1", "claim": "'First establish what is great' is his only technique, accepted in his own words",
         "basis": "primary", "source": "Lu Jiuyuan ji, juan 35 (Yulu); Mencius 6A15"},
        {"id": "D2", "claim": "Order dispute at Goose Lake: original mind first, broad reading after",
         "basis": "primary", "source": "Lu Jiuyuan ji, juan 36 (Nianpu, Zhu Hengdao's record, 1175)"},
        {"id": "D3", "claim": "Understanding follows practice, practice follows aim; exams reward skill and examiners' likes",
         "basis": "primary", "source": "Bailudong shuyuan Lunyu jiangyi (1181), Lu Jiuyuan ji, juan 23"},
        {"id": "D4", "claim": "Original mind = knowing right as right, wrong as wrong in a live dispute (fan lawsuit)",
         "basis": "primary", "source": "Lu Jiuyuan ji, juan 36 (Nianpu, 1172); Song shi 407 (Yang Jian)"},
        {"id": "D5", "claim": "The mind is one; the human mind and Way-mind are not two minds",
         "basis": "primary", "source": "Lu Jiuyuan ji, juan 34 (Yulu)"},
        {"id": "D6", "claim": "Knowing prices is an ability but not the work of the mind",
         "basis": "primary", "source": "Lu Jiuyuan ji, juan 35 (Yulu, reply to Lu Jiuling)"},
        {"id": "D7", "claim": "Zhu Xi: they do not believe the human world has ancient and present",
         "basis": "scholarship", "source": "Zhu Xi, poem answering Lu Jiuling (1178); Ching 1974, JCP 1(2)"},
    ],
    "research_question": {"category": "robustness to spurious correlation; reward hacking and specification gaming",
                          "question": "Does establishing a deep expert first on grader-free disputes keep a proxy "
                                      "grader's taste out of its judgments better than mixing, reversing, or "
                                      "establishing the shallow expert first?"},
    "mechanism": {
        "name": "Great-first two-organ learner", "family": "product-of-experts debiasing with phase order",
        "signature_modules": ["establishment_phase", "small_organ", "grade_coupling"],
        "closest_prior_art": ["Clark, Yatskar, Zettlemoyer 2019, arXiv 1909.03683 (bias-only expert first)",
                              "Sanh, Wolf, Belinkov, Rush 2021, arXiv 2012.01300 (limited-capacity weak learner)",
                              "Korbak et al. 2023, arXiv 2302.08582 (preferences from the start)",
                              "Kaushik, Hovy, Lipton 2020, arXiv 1909.12434 (matched counterfactual pairs)"],
        "overlap": "Medium",
        "prior_art_queries": ["product of experts debiasing shallow model trained first",
                              "curriculum order clean data before noisy proxy labels",
                              "pretraining with human preferences versus finetuning after",
                              "gradient starvation order of feature learning"],
        "contribution_type": "mechanism",
        "delta": "Deep expert established first and alone on grader-free matched disputes; shallow expert "
                 "opened afterwards, kept plastic, excluded from the rightness judgment.",
    },
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1", "property_test": "C3", "hypothesis": "H-EST"},
        {"doctrine": "D1", "mechanism": "M1", "property_test": "C3", "hypothesis": "H-PRIOR"},
        {"doctrine": "D2", "mechanism": "M1", "property_test": "C3", "hypothesis": "H-RIVAL"},
        {"doctrine": "D3", "mechanism": "M2", "property_test": "C6.3", "hypothesis": "H-SIG"},
        {"doctrine": "D4", "mechanism": "data", "property_test": "C6.2", "hypothesis": "H-SIG"},
        {"doctrine": "D5", "mechanism": "M3", "property_test": "C6.4", "hypothesis": "H-NEC"},
        {"doctrine": "D6", "mechanism": "M2", "property_test": "C6.1", "hypothesis": "H-NEC"},
        {"doctrine": "D7", "mechanism": "blind world", "property_test": "C6.1", "hypothesis": "H-BLIND"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "Great-first two-organ learner judges rightness better than a size-matched MLP "
                                     "under a shift of examiner fashion", "metric": "shift_acc", "split": "shifted",
         "comparison": "lu - mlp", "direction": "greater", "mesi": 0.05, "seeds": 5},
        {"id": "H-NEC", "statement": "Removing the small organ hurts more than removing a matched block of root units",
         "metric": "shift_acc", "comparison": "no_small_organ - minus_root_units", "direction": "less",
         "mesi": 0.05, "seeds": 5},
        {"id": "H-EST", "statement": "Establishing the root first beats mixing both signals from step 0",
         "metric": "shift_acc", "comparison": "lu - mixed", "direction": "greater", "mesi": 0.02, "seeds": 5},
        {"id": "H-PRIOR", "statement": "Great-first beats the standard small-first product of experts",
         "metric": "shift_acc", "comparison": "lu - small_first", "direction": "greater", "mesi": 0.05, "seeds": 5},
        {"id": "H-RIVAL", "statement": "Lu's order beats Zhu Xi's order (broad reading first, essentials last)",
         "metric": "shift_acc", "comparison": "lu - zhu", "direction": "greater", "mesi": 0.02, "seeds": 5},
        {"id": "H-BLIND", "statement": "In a changed world the great-first learner misses the new consideration",
         "condition": "gamma = 1.2 in essays, disputes from the old world",
         "grounding": "Zhu Xi's charge that Lu did not believe in an ancient and a present",
         "metric": "novel_acc", "comparison": "lu - mlp", "direction": "less", "mesi": 0.02, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.3, "margin_over_trivial": 0.15},
    "probe_predictions": [{"probe": "P9", "expected": "above baseline"}, {"probe": "P8", "expected": "below baseline"},
                          {"probe": "P6", "expected": "equal to baseline"}, {"probe": "P5", "expected": "equal to baseline"}],
    "dialectic_links": [{"chapter": 299, "relation": "rival", "test": "H-RIVAL"},
                        {"chapter": 73, "relation": "teacher", "test": "H-NEC"},
                        {"chapter": 444, "relation": "successor", "test": "none (discussed in chapter)"}],
    "corpus_neighbors": [
        {"chapter": 299, "similarity": None, "difference": "Zhu Xi accumulates particulars then integrates; here the order is reversed and tested head to head"},
        {"chapter": 73, "similarity": None, "difference": "Mencius extends each sprout by analogy; here one root judges and a separate linear organ holds surface knowledge"},
        {"chapter": 478, "similarity": None, "difference": "Toegye gates two issuances of feeling; here there is one mind and no arbitration gate"},
        {"chapter": 188, "similarity": None, "difference": "Huineng learns by subtraction; nothing is pruned here, the order of establishment decides"},
        {"chapter": 284, "similarity": None, "difference": "Avempace reaches sameness between knowers by relational re-expression; not claimed here"},
        {"chapter": 296, "similarity": None, "difference": "Eleanor re-hosts a detachable core through thin interfaces; here the core is trained first, not detached"},
    ],
    "barometer": {"cognitive_processing": ["P2 relational transfer (root)"], "embodied_cognition": [],
                  "world_modeling": ["P9 perturbation robustness under grader-fashion shift"],
                  "consciousness": ["P8 calibration of the rightness judgment"],
                  "language_understanding": [], "emotional_intelligence": [],
                  "creativity": [], "autonomy": ["blind world: learning a new consideration"]},
    "task_types": ["vector_classification"],
    "applications": [
        {"use": "Reward-model training where rater scores carry style or length taste; establish on adjudicated "
                "matched pairs first", "sector": "AI alignment", "dataset": "Anthropic HH-RLHF (Bai et al. 2022)"},
        {"use": "Automated essay scoring robust to rater preference for surface style (decision support only)",
         "sector": "education", "dataset": "ASAP Automated Student Assessment Prize (Hewlett Foundation, 2012)"},
        {"use": "Toxicity classification robust to identity-term shortcuts", "sector": "online safety",
         "dataset": "CivilComments-WILDS (Borkan et al. 2019; Koh et al. 2021)"},
    ],
    "safety_notes": "Rightness is a synthetic latent. No claim to replicate Lu Jiuyuan's mind; no generated "
                    "sentences are attributed to him. Essay-scoring use is decision support, never sole grading.",
}

import argparse
import json
import math
import os
import sys
import time

import numpy as np

CHAPTER = 305
TASK_TYPES = ["vector_classification"]
D_SUB, D_STY, D_NUI, N_LAT, K = 10, 4, 3, 6, 2
D_IN = D_SUB + D_STY + D_NUI
HIDDEN = 24                 # root units; the MLP baseline gets HIDDEN + 2 to match parameter count
BETA_TASTE = 1.1            # weight of the examiner's taste in the grade
GAMMA_NEW = 1.2             # weight of the new consideration in the changed world
MARGIN = 0.15               # disputes are clear cases: |v| > MARGIN
NOURISH = 0.5               # weight of dispute replay after establishment (hanyang, nourishing)
LR, L2, CLIP = 0.01, 1e-3, 5.0
EST_FRAC = 0.28             # share of the update budget spent establishing the root
RHO0 = math.log(math.e - 1.0)   # softplus(RHO0) = 1: grade scale tau starts at one
N_ESSAYS, N_DISPUTES, N_HELD, N_SHIFT, N_BLIND = 2000, 200, 3000, 3000, 4000
BUDGET_FULL, BUDGET_QUICK = 900, 360
TIME_FULL, TIME_QUICK = 180.0, 20.0
ROOT_KEYS = ("W1", "b1", "Wa", "c", "rho")
SMALL_KEYS = ("Wp", "bp")


class NonFinite(Exception):
    """Raised when a loss, gradient or activation stops being finite (exit code 4)."""


# BEGIN STANDARD UTILITIES v1.0
def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def logsumexp(z, axis=-1):
    m = z.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(z - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def softplus(z):
    return np.logaddexp(0.0, z)


def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


class Adam:
    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads, frozen=(), sign=1.0, lr=None):
        self.t += 1
        lr = self.lr if lr is None else lr
        for k in params:
            if k in frozen:
                continue
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= sign * lr * mh / (np.sqrt(vh) + self.eps)


def clip_global_norm(grads, max_norm):
    total = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    if total > max_norm:
        return {k: g * (max_norm / total) for k, g in grads.items()}
    return grads


def grad_check(loss_fn, grads, params, rng, eps=1e-6, n_entries=20):
    """Central differences in float64 on every tensor: all entries of small tensors, else
    n_entries random ones plus the largest-gradient entry. Returns {name: max rel error}."""
    out = {}
    for name, p in params.items():
        flat = p.reshape(-1)
        g = grads[name].reshape(-1)
        idx = np.arange(flat.size) if flat.size <= 40 else np.unique(
            np.concatenate([rng.choice(flat.size, n_entries, replace=False), [int(np.argmax(np.abs(g)))]]))
        worst = 0.0
        for i in idx:
            old = flat[i]
            flat[i] = old + eps
            lp = loss_fn()
            flat[i] = old - eps
            lm = loss_fn()
            flat[i] = old
            num = (lp - lm) / (2 * eps)
            worst = max(worst, abs(num - g[i]) / max(1e-7, abs(num) + abs(g[i])))
        out[name] = worst
    return out


def paired_bootstrap(diffs, rng, n_boot=2000):
    diffs = np.asarray(diffs, dtype=np.float64)
    means = diffs[rng.integers(0, len(diffs), size=(n_boot, len(diffs)))].mean(axis=1)
    return float(diffs.mean()), (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def verdict(mean, ci, direction, mesi):
    lo, hi = ci
    if direction == "greater":
        if lo > 0 and mean >= mesi:
            return "supported"
        return "contradicted" if hi < 0 else "inconclusive"
    if hi < 0 and mean <= -mesi:
        return "supported"
    return "contradicted" if lo > 0 else "inconclusive"


def write_json_report(path, report):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
# END STANDARD UTILITIES


# ---------------------------------------------------------------- data and tasks
def make_world(seed):
    """Fixed mixing of latent case into substance features, and the examiner's taste axis."""
    r = np.random.default_rng(np.random.SeedSequence([CHAPTER, seed, 11]))
    taste = r.standard_normal(D_STY)
    return {"A": r.standard_normal((D_SUB, N_LAT)) / math.sqrt(N_LAT), "taste": taste / np.linalg.norm(taste)}


def principle(u, gamma):
    """The rightness of a case: a deep interaction plus a shallow term (and, later, a new term)."""
    return 1.4 * u[:, 0] * u[:, 1] + 1.0 * u[:, 2] + gamma * u[:, 5]


def compose(rng, u, style, nuisance, world):
    z = u @ world["A"].T + 0.1 * rng.standard_normal((len(u), D_SUB))
    return np.concatenate([z, style, nuisance], axis=1)


def essays(rng, n, world, gamma=0.0, shift=False):
    """Graded essays: the grade mixes rightness with the examiner's liking for a style."""
    u = rng.standard_normal((n, N_LAT))
    v, v_old = principle(u, gamma), principle(u, 0.0)
    style = rng.standard_normal((n, D_STY))
    if shift:   # a new fashion: styles pushed hard along the old taste axis, both ways
        style = style + 2.0 * np.where(rng.random(n) < 0.5, -1.0, 1.0)[:, None] * world["taste"][None, :]
    taste = style @ world["taste"]
    grade = (v + BETA_TASTE * taste + 0.3 * rng.standard_normal(n) > 0).astype(int)
    X = compose(rng, u, style, rng.standard_normal((n, D_NUI)), world)
    return {"X": X, "grade": grade, "right": (v > 0).astype(int), "novel": (v > 0) != (v_old > 0), "taste": taste}


def _clear_cases(rng, m, sign, gamma, ancient):
    kept = []
    while sum(len(k) for k in kept) < m:
        u = rng.standard_normal((4 * m, N_LAT))
        if ancient:
            u[:, 5] = 0.0
        kept.append(u[sign * principle(u, 0.0 if ancient else gamma) > MARGIN])
    return np.concatenate(kept)[:m]


def disputes(rng, m, world, gamma=0.0, ancient=False):
    """Fan-lawsuit disputes: two acts in one case share style and nuisance; one is right."""
    u_right = _clear_cases(rng, m, 1.0, gamma, ancient)
    u_wrong = _clear_cases(rng, m, -1.0, gamma, ancient)
    style, nuisance = rng.standard_normal((m, D_STY)), rng.standard_normal((m, D_NUI))
    XR = compose(rng, u_right, style, nuisance, world)
    XW = compose(rng, u_wrong, style, nuisance, world)
    return {"X": np.concatenate([XR, XW]), "y": np.r_[np.ones(m, int), np.zeros(m, int)], "m": m}


def make_splits(seed, changed=False):
    rng = np.random.default_rng(np.random.SeedSequence([CHAPTER, seed, 29 if changed else 23]))
    world = make_world(seed)
    gamma = GAMMA_NEW if changed else 0.0
    tr = essays(rng, N_ESSAYS, world, gamma)
    dp = disputes(rng, N_DISPUTES, world, gamma, ancient=changed)
    held = essays(rng, N_BLIND if changed else N_HELD, world, gamma)
    shift = essays(rng, N_SHIFT, world, gamma, shift=True)
    train = {"X": tr["X"], "y": tr["grade"], "Xr": dp["X"], "yr": dp["y"]}
    return {"world": world, "train": train, "held": held, "shift": shift, "disputes": dp, "essays": tr}


# ---------------------------------------------------------------- model
def build_model(in_dim, out_dim, task_type="vector_classification", rng=None, hidden=HIDDEN,
                small_organ=True, kind="lu"):
    """Root (great organ): tanh trunk -> rightness logits r. Small organ: linear logits p on raw
    input. Grade logits g = softplus(rho) * r + p. The rightness judgment uses r alone."""
    if task_type not in TASK_TYPES:
        raise ValueError("unsupported task type: %s" % task_type)
    rng = np.random.default_rng(0) if rng is None else rng
    P = {"W1": rng.standard_normal((hidden, in_dim)) / math.sqrt(in_dim), "b1": np.zeros(hidden),
         "Wa": 0.1 * rng.standard_normal((out_dim, hidden)), "c": np.zeros(out_dim), "rho": np.array([RHO0])}
    if small_organ:
        P["Wp"], P["bp"] = np.zeros((out_dim, in_dim)), np.zeros(out_dim)
    return {"params": P, "kind": kind, "small_organ": small_organ, "K": out_dim, "mutant": None, "ko": None}


def forward(model, X):
    P = model["params"]
    h = np.tanh(X @ P["W1"].T + P["b1"])
    if model["ko"] == ("root_trunk", "mean"):
        h = np.repeat(h.mean(axis=0, keepdims=True), len(X), axis=0)
    elif model["ko"] == ("root_trunk", "zero"):
        h = np.zeros_like(h)
    r = h @ P["Wa"].T + P["c"]
    p = X @ P["Wp"].T + P["bp"] if model["small_organ"] else np.zeros_like(r)
    tau = softplus(P["rho"][0])
    return h, r, p, tau * r + p


def onehot(y, k):
    out = np.zeros((len(y), k))
    out[np.arange(len(y)), y] = 1.0
    return out


def grade_error_signal(model, r, p, Y, tau):
    """The error the root receives from one graded essay. The small organ's logit sits inside
    the softmax, so whatever it already explains no longer teaches the root (explaining away)."""
    if model["mutant"] == "decoupled_root":
        return tau * (softmax(tau * r) - Y)
    return tau * (softmax(tau * r + p) - Y)


def backprop_root(P, X, h, dr, G):
    G["Wa"] += dr.T @ h
    G["c"] += dr.sum(axis=0)
    dz = (dr @ P["Wa"]) * (1.0 - h * h)
    G["W1"] += dz.T @ X
    G["b1"] += dz.sum(axis=0)


def loss_and_grads(model, batch):
    """w_g * CE(grade logits, grades) + w_r * CE(root logits, dispute labels) + L2/2 |W1|^2."""
    P, k = model["params"], model["K"]
    w_g, w_r = batch.get("w_g", 1.0), batch.get("w_r", NOURISH)
    G = {name: np.zeros_like(v) for name, v in P.items()}
    loss = 0.5 * L2 * float(np.sum(P["W1"] ** 2))
    G["W1"] += L2 * P["W1"]
    if w_g > 0:
        X, Y = batch["X"], onehot(batch["y"], k)
        h, r, p, g = forward(model, X)
        n = len(X)
        loss += w_g * float(np.mean(logsumexp(g) - np.sum(g * Y, axis=1)))
        dg = w_g * (softmax(g) - Y) / n
        tau = softplus(P["rho"][0])
        G["rho"] += np.sum(dg * r) * sigmoid(P["rho"][0])
        if model["small_organ"]:
            G["Wp"] += dg.T @ X
            G["bp"] += dg.sum(axis=0)
        backprop_root(P, X, h, grade_error_signal(model, r, p, Y, tau) * w_g / n, G)
    if w_r > 0 and "Xr" in batch:
        Xr, Yr = batch["Xr"], onehot(batch["yr"], k)
        h, r, _, _ = forward(model, Xr)
        loss += w_r * float(np.mean(logsumexp(r) - np.sum(r * Yr, axis=1)))
        backprop_root(P, Xr, h, w_r * (softmax(r) - Yr) / len(Xr), G)
    if model["mutant"] == "zero_grad_W1":
        G["W1"] = np.zeros_like(G["W1"])
    if model["mutant"] == "zero_grad_small_organ" and model["small_organ"]:
        G["Wp"] = np.zeros_like(G["Wp"])
    return loss, G


def predict(model, X):
    """The rightness judgment: the root alone speaks."""
    return np.argmax(forward(model, X)[1], axis=1)


def predict_grade(model, X):
    """What the examiner will say: root and small organ together."""
    return np.argmax(forward(model, X)[3], axis=1)


def hidden_states(model, X):
    return forward(model, X)[0]


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


# ---------------------------------------------------------------- baselines and rival orders
def plan(order, budget):
    """Phase schedules. Each phase: steps, grade weight, dispute weight, frozen tensors.
    great_first (Lu): establish the root on disputes alone, then broad graded practice with replay.
    zhu (rival, Zhu Xi): the same two phases in the reverse order -- broad first, essentials last.
    mixed: both signals from step 0, dispute weight raised to match Lu's total dispute mass.
    small_first (prior art): the small organ alone on grades, frozen; then the root through the
    combined logit (bias-only expert first, as in product-of-experts debiasing)."""
    t1 = int(round(EST_FRAC * budget))
    t2 = budget - t1
    if order == "great_first":
        return [(t1, 0.0, 1.0, ()), (t2, 1.0, NOURISH, ())]
    if order == "zhu":
        return [(t2, 1.0, NOURISH, ()), (t1, 0.0, 1.0, ())]
    if order == "mixed":
        return [(budget, 1.0, NOURISH + t1 / budget, ())]
    if order == "small_first":
        return [(t1, 1.0, 0.0, ROOT_KEYS), (t2, 1.0, NOURISH, SMALL_KEYS)]
    raise ValueError("unknown order: %s" % order)


CONFIGS = {   # name: (kind, small organ, hidden units, phase order)
    "lu": ("lu", True, HIDDEN, "great_first"),
    "mlp": ("mlp", False, HIDDEN + 2, "mixed"),
    "mixed": ("lu", True, HIDDEN, "mixed"),
    "zhu": ("lu", True, HIDDEN, "zhu"),
    "small_first": ("lu", True, HIDDEN, "small_first"),
    "no_small_organ": ("lu", False, HIDDEN, "great_first"),
    "minus_root_units": ("lu", True, HIDDEN - 2, "great_first"),
}


# ---------------------------------------------------------------- registries
def modules(model):
    P = model["params"]
    out = {"root_trunk": {"params": ["W1", "b1"], "signature": False,
                          "role": "tanh feature trunk of the deep expert (the organ that thinks)"},
           "root_readout": {"params": ["Wa", "c"], "signature": False,
                            "role": "linear readout giving the rightness logits"},
           "grade_coupling": {"params": ["rho"], "signature": True,
                              "role": "product-of-experts grade logit tau*r + p with learned scale"},
           "establishment_phase": {"params": [], "signature": True,
                                   "role": "training phase: deep expert alone on matched disputes first"}}
    if "Wp" in P:
        out["small_organ"] = {"params": ["Wp", "bp"], "signature": True,
                              "role": "fast linear shallow expert on raw input, grades only"}
    return out


def knockout(model, name, mode="zero"):
    """Copy of a trained model with one module replaced (identity, zero or mean output)."""
    m = dict(model, params={k: v.copy() for k, v in model["params"].items()})
    if name == "small_organ" and m["small_organ"]:
        m["params"]["Wp"][:] = 0.0
        m["params"]["bp"][:] = 0.0
    elif name == "root_trunk":
        m["ko"] = ("root_trunk", "mean" if mode == "mean" else "zero")
    elif name == "grade_coupling":
        m["params"]["rho"][:] = RHO0          # identity scale, tau = 1
    else:
        raise ValueError("no inference-time knockout for module %s" % name)
    return m


MUTANTS = {
    "sign_flip_update": "optimizer ascends the loss",
    "zero_lr": "learning rate set to zero",
    "zero_grad_W1": "analytic gradient of the root trunk weights zeroed",
    "zero_grad_small_organ": "analytic gradient of the small organ zeroed",
    "decoupled_root": "root receives its grade error as if the small organ did not exist",
}


# ---------------------------------------------------------------- training
def fit(model, data, budget, rng=None, order="great_first", trace=None):
    """Full-batch Adam over the phase plan. Without disputes in `data`, the root is established on
    the graded data itself (small organ closed) before the small organ opens. `rng` is unused:
    full-batch training is deterministic given the initial parameters."""
    if "Xr" not in data:
        data = dict(data, Xr=data["X"], yr=data["y"])
    P = model["params"]
    if order == "small_first":
        P["Wa"][:] = 0.0                      # the root contributes nothing while the small organ learns
    opt = Adam(P, LR)
    sign = -1.0 if model["mutant"] == "sign_flip_update" else 1.0
    lr = 0.0 if model["mutant"] == "zero_lr" else LR
    for steps, w_g, w_r, frozen in plan(order, budget):
        if not model["small_organ"]:
            frozen = tuple(f for f in frozen if f not in SMALL_KEYS)
        batch = dict(data, w_g=w_g, w_r=w_r)
        for _ in range(steps):
            loss, G = loss_and_grads(model, batch)
            if not np.isfinite(loss) or not all(np.all(np.isfinite(g)) for g in G.values()):
                raise NonFinite("non-finite loss or gradient")
            opt.step(P, clip_global_norm(G, CLIP), frozen=frozen, sign=sign, lr=lr)
            if trace is not None:
                trace.append(loss)
    return model


def objective(model, data):
    """Common yardstick for learning: the phase-two objective (grades plus dispute replay)."""
    return loss_and_grads(model, dict(data, w_g=1.0, w_r=NOURISH))[0]


def train_config(name, split, seed, budget, mutant=None):
    kind, small, hidden, order = CONFIGS[name]
    rng = np.random.default_rng(np.random.SeedSequence([CHAPTER, seed, 41]))
    model = build_model(D_IN, K, rng=rng, hidden=hidden, small_organ=small, kind=kind)
    model["mutant"] = mutant
    start = objective(model, split["train"])
    fit(model, split["train"], budget, order=order)
    model["loss_start"], model["loss_end"] = start, objective(model, split["train"])
    return model


def accuracy(model, split_part):
    return float(np.mean(predict(model, split_part["X"]) == split_part["right"]))


def snatch_index(model, world):
    """Signed response of the small organ's grade logit to the shallow term of the principle.
    Positive means the small organ holds part of what the root should judge."""
    if not model["small_organ"]:
        return 0.0
    w = model["params"]["Wp"][1, :D_SUB] - model["params"]["Wp"][0, :D_SUB]
    return float(w @ world["A"][:, 2])


def evaluate(model, split):
    held, shift = split["held"], split["shift"]
    pred = predict(model, held["X"])
    novel = held["novel"]
    return {"held_acc": accuracy(model, held), "shift_acc": accuracy(model, shift),
            "grade_acc": float(np.mean(predict_grade(model, held["X"]) == held["grade"])),
            "novel_acc": float(np.mean(pred[novel] == held["right"][novel])) if novel.any() else float("nan"),
            "snatch": snatch_index(model, split["world"])}


def run_seed(seed, budget, mutant=None):
    """All configurations on the stable world and three on the changed world. Returns metrics and
    the trained great-first model of the stable world (used by C3 and the knockout table)."""
    main, changed = make_splits(seed), make_splits(seed, changed=True)
    res, lu_model = {"main": {}, "changed": {}}, None
    for name in CONFIGS:
        model = train_config(name, main, seed, budget, mutant)
        res["main"][name] = evaluate(model, main)
        lu_model = model if name == "lu" else lu_model
    for name in ("lu", "mlp", "zhu"):
        res["changed"][name] = evaluate(train_config(name, changed, seed, budget, mutant), changed)
    return res, lu_model, main


# ---------------------------------------------------------------- correctness tests
def c1_gradcheck(seed, mutant=None):
    """Every tensor of the great-first model and of the MLP baseline, at initialisation and
    after 60 training steps, on a fixed mini-batch mixing essays and disputes."""
    split = make_splits(seed)
    rng = np.random.default_rng(np.random.SeedSequence([CHAPTER, seed, 51]))
    tr = split["train"]
    ie, ir = rng.choice(len(tr["X"]), 40, replace=False), rng.choice(len(tr["Xr"]), 20, replace=False)
    batch = {"X": tr["X"][ie], "y": tr["y"][ie], "Xr": tr["Xr"][ir], "yr": tr["yr"][ir], "w_g": 1.0, "w_r": NOURISH}
    worst, checked, total = 0.0, 0, 0
    for name in ("lu", "mlp"):
        kind, small, hidden, order = CONFIGS[name]
        model = build_model(D_IN, K, rng=np.random.default_rng(seed), hidden=hidden, small_organ=small, kind=kind)
        model["mutant"] = mutant
        for stage in ("init", "after_steps"):
            if stage == "after_steps":
                fit(model, tr, 60, order=order)
            _, G = loss_and_grads(model, batch)
            errs = grad_check(lambda: loss_and_grads(model, batch)[0], G, model["params"], rng)
            worst = max(worst, max(errs.values()))
            checked += len(errs)
            total += len(model["params"])
    return {"passed": worst <= 1e-5 and checked == total, "max_rel_error": worst,
            "tensors_checked": checked, "tensors_total": total}


def c3_learning(model, split):
    """Loss must fall by the declared fraction; rightness must beat the majority rule by a margin."""
    drop = 1.0 - model["loss_end"] / model["loss_start"]
    trivial = max(np.mean(split["held"]["right"]), 1 - np.mean(split["held"]["right"]))
    acc = accuracy(model, split["held"])
    th = MIND_CARD["thresholds"]
    return {"passed": drop >= th["loss_drop_fraction"] and acc >= trivial + th["margin_over_trivial"],
            "loss_drop": float(drop), "held_acc": acc, "trivial": float(trivial)}


def c2_determinism(seed, budget):
    split = make_splits(seed)
    a, b = train_config("lu", split, seed, budget), train_config("lu", split, seed, budget)
    same = all(np.array_equal(a["params"][k], b["params"][k]) for k in a["params"])
    same = same and np.array_equal(forward(a, split["held"]["X"])[3], forward(b, split["held"]["X"])[3])
    same = same and np.array_equal(hidden_states(a, split["shift"]["X"]), hidden_states(b, split["shift"]["X"]))
    finite = all(np.all(np.isfinite(v)) for v in a["params"].values()) and \
        all(np.all(np.isfinite(t)) for t in forward(a, split["shift"]["X"])[:3])
    return {"passed": bool(same and finite), "identical": bool(same), "finite": bool(finite)}


def c4_shuffled(seed, budget, n_shuffles=2):
    """Grades and dispute labels permuted: held-out rightness must stay in the trivial band."""
    split = make_splits(seed)
    trivial = max(np.mean(split["held"]["right"]), 1 - np.mean(split["held"]["right"]))
    accs = []
    for s in range(n_shuffles):
        rng = np.random.default_rng(np.random.SeedSequence([CHAPTER, seed, 61, s]))
        tr = dict(split["train"], y=rng.permutation(split["train"]["y"]), yr=rng.permutation(split["train"]["yr"]))
        accs.append(accuracy(train_config("lu", dict(split, train=tr), seed, budget), split["held"]))
    return {"passed": all(abs(a - trivial) <= 0.07 for a in accs), "accs": accs, "trivial": float(trivial)}


def c5_mutants(seed, budget):
    split = make_splits(seed)
    detected = {}
    for name in MUTANTS:
        c1 = c1_gradcheck(seed, mutant=name)
        c3 = c3_learning(train_config("lu", split, seed, budget, mutant=name), split)
        detected[name] = (not c1["passed"]) or (not c3["passed"])
    score = sum(detected.values()) / len(detected)
    return {"passed": score == 1.0, "detected": detected, "score": score}


def c6_generator(seed):
    """C6.1: rightness is independent of the examiner's taste axis in every split; the grade is
    not (negative control)."""
    rng = np.random.default_rng(np.random.SeedSequence([CHAPTER, seed, 71]))
    world = make_world(seed)
    worst = 0.0
    for gamma, shift in ((0.0, False), (0.0, True), (GAMMA_NEW, False)):
        e = essays(rng, 20000, world, gamma, shift)
        worst = max(worst, abs(np.corrcoef(e["taste"], e["right"])[0, 1]))
    e = essays(rng, 20000, world)
    control = abs(np.corrcoef(e["taste"], e["grade"])[0, 1])
    return {"passed": worst < 0.03 and control > 0.2, "max_corr_taste_right": float(worst),
            "control_corr_taste_grade": float(control)}


def c6_disputes(seed):
    """C6.2: the two acts of a dispute share style and nuisance exactly and carry opposite labels;
    two independent essays do not share style (negative control)."""
    split = make_splits(seed)
    dp, m = split["disputes"], split["disputes"]["m"]
    tail = slice(D_SUB, D_IN)
    shared = np.array_equal(dp["X"][:m, tail], dp["X"][m:, tail])
    labels = bool(np.all(dp["y"][:m] == 1) and np.all(dp["y"][m:] == 0))
    ess = split["essays"]["X"]
    control = np.array_equal(ess[:m, tail], ess[m:2 * m, tail])
    return {"passed": shared and labels and not control, "shared_context": bool(shared),
            "opposite_labels": labels, "control_shared": bool(control)}


def _explaining_away_violations(mutant, rng, trials=6):
    """Raising the small organ's logit for the true class must strictly shrink the root's error
    on that class. Random search over random parameters, inputs and increments."""
    bad = 0
    for _ in range(trials):
        model = build_model(D_IN, K, rng=rng)
        model["mutant"] = mutant
        model["params"]["Wp"] = rng.standard_normal((K, D_IN))
        model["params"]["rho"] = rng.normal(0.0, 1.0, 1)
        X = 2.0 * rng.standard_normal((300, D_IN))
        y = rng.integers(0, K, 300)
        Y = onehot(y, K)
        _, r, p, _ = forward(model, X)
        tau = softplus(model["params"]["rho"][0])
        rows = np.arange(len(y))
        e0 = np.abs(grade_error_signal(model, r, p, Y, tau)[rows, y])
        e1 = np.abs(grade_error_signal(model, r, p + rng.uniform(0.05, 2.0, (300, 1)) * Y, Y, tau)[rows, y])
        live = (e0 > 1e-6) & (e0 < tau * (1.0 - 1e-6))   # exclude probabilities saturated at 0 or 1
        bad += int(np.sum(e1[live] >= e0[live]))
    return bad


def c6_explaining_away(seed):
    """C6.3 (claimed invariant of the grade coupling), with the decoupled root as negative control."""
    rng = np.random.default_rng(np.random.SeedSequence([CHAPTER, seed, 81]))
    real = _explaining_away_violations(None, rng)
    control = _explaining_away_violations("decoupled_root", rng)
    return {"passed": real == 0 and control > 0, "violations": real, "control_violations": control}


def c6_one_mind(seed):
    """C6.4 DEFINITION CHECK (not evidence): the rightness judgment does not read the small organ."""
    split = make_splits(seed)
    model = build_model(D_IN, K, rng=np.random.default_rng(seed))
    X = split["held"]["X"][:200]
    _, r0, _, g0 = forward(model, X)
    model["params"]["Wp"] += 1.0
    _, r1, _, g1 = forward(model, X)
    return {"passed": bool(np.array_equal(r0, r1) and not np.allclose(g0, g1)), "definition_check": True}


def c7_split_integrity(seed):
    split = make_splits(seed)
    seen = {row.tobytes() for row in np.round(np.concatenate([split["train"]["X"], split["train"]["Xr"]]), 9)}
    clash = sum(row.tobytes() in seen for part in ("held", "shift") for row in np.round(split[part]["X"], 9))
    return {"passed": clash == 0, "overlapping_rows": int(clash)}


# ---------------------------------------------------------------- hypothesis tests
COMPARISONS = {"H-SIG": ("main", "lu", "mlp"), "H-NEC": ("main", "no_small_organ", "minus_root_units"),
               "H-EST": ("main", "lu", "mixed"), "H-PRIOR": ("main", "lu", "small_first"),
               "H-RIVAL": ("main", "lu", "zhu"), "H-BLIND": ("changed", "lu", "mlp")}
LESIONS = [("establishment_phase -> mixed from step 0", True, "mixed"),
           ("establishment_phase -> reversed (Zhu's order)", True, "zhu"),
           ("small_organ removed", True, "no_small_organ"),
           ("small organ established first, frozen (prior art)", True, "small_first"),
           ("two root units removed (matched, non-signature)", False, "minus_root_units")]


def paired(results, world, a, b, metric, rng):
    diffs = [r[world][a][metric] - r[world][b][metric] for r in results]
    if len(diffs) < 2:
        return float(diffs[0]), (float("nan"), float("nan"))
    return paired_bootstrap(diffs, rng)


def test_hypotheses(results, rng, quick):
    out = []
    for h in MIND_CARD["hypotheses"]:
        world, a, b = COMPARISONS[h["id"]]
        mean, ci = paired(results, world, a, b, h["metric"], rng)
        v = "not evaluated" if quick or len(results) < h["seeds"] else verdict(mean, ci, h["direction"], h["mesi"])
        out.append({"id": h["id"], "metric": h["metric"], "comparison": h["comparison"], "mean_diff": mean,
                    "ci95": list(ci), "mesi": h["mesi"], "n_seeds": len(results), "verdict": v})
    return out


def lesion_table(results, rng):
    return [{"module": name, "signature": sig, "metric_change": m, "ci95": list(ci)}
            for name, sig, cfg in LESIONS for m, ci in [paired(results, "main", cfg, "lu", "shift_acc", rng)]]


def inference_knockouts(model, split):
    held = split["held"]
    base_r, base_g = accuracy(model, held), float(np.mean(predict_grade(model, held["X"]) == held["grade"]))
    rows = []
    for name, mode in (("small_organ", "zero"), ("root_trunk", "mean"), ("grade_coupling", "identity")):
        m = knockout(model, name, mode)
        rows.append({"module": name, "mode": mode, "signature": modules(model)[name]["signature"],
                     "rightness_change": accuracy(m, held) - base_r,
                     "grade_change": float(np.mean(predict_grade(m, held["X"]) == held["grade"])) - base_g})
    return rows


def summarize(results):
    table = {}
    for world in ("main", "changed"):
        for name in results[0][world]:
            for metric in results[0][world][name]:
                vals = np.array([r[world][name][metric] for r in results], dtype=float)
                table["%s/%s/%s" % (world, name, metric)] = (float(np.mean(vals)), float(np.std(vals)))
    return table


# ---------------------------------------------------------------- report
def print_report(rep):
    line = "-" * 78
    print("=== VERIFIED REPORT · chapter %04d ===" % CHAPTER)
    print("file %s · python %s · numpy %s" % (rep["file"], rep["environment"]["python"], rep["environment"]["numpy"]))
    print("seeds %s · runtime %.1f s · parameters %s" % (rep["seeds"], rep["runtime_s"], rep["n_params_by_model"]))
    g = rep["gradcheck"]
    print("gradient check: %d/%d tensors (init and after 60 steps), max rel error %.2e -> %s"
          % (g["tensors_checked"], g["tensors_total"], g["max_rel_error"], "pass" if g["passed"] else "FAIL"))
    print(line + "\ncorrectness")
    for c in rep["correctness"]:
        print("  %-6s %-32s %-5s %s" % (c["id"], c["name"], "pass" if c["passed"] else "FAIL", c["detail"]))
    mu = rep["mutants"]
    print("mutation score %d/%d = %.2f" % (mu["detected"], mu["total"], mu["score"]))
    print(line + "\nhypotheses (paired over seeds, 95% bootstrap interval, 2000 resamples)")
    for h in rep["hypotheses"]:
        print("  %-8s %-34s %+.4f  [%+.4f, %+.4f]  mesi %.2f  %s" % (h["id"], h["comparison"], h["mean_diff"],
              h["ci95"][0], h["ci95"][1], h["mesi"], h["verdict"]))
    print(line + "\ntraining lesions (shifted rightness accuracy, lesion minus great-first)")
    for k in rep["knockouts"]:
        print("  %-52s %s %+.4f  [%+.4f, %+.4f]" % (k["module"], "sig" if k["signature"] else "   ",
              k["metric_change"], k["ci95"][0], k["ci95"][1]))
    print("inference knockouts on the seed-%d great-first model (held-out)" % rep["seeds"][0])
    for k in rep["inference_knockouts"]:
        print("  %-15s %-9s rightness %+.4f  grade %+.4f" % (k["module"], k["mode"], k["rightness_change"],
              k["grade_change"]))
    print(line + "\nmeans over seeds (sd): rightness held / shifted, exam-grade skill, small-organ snatch")
    s = rep["summary"]
    for name in CONFIGS:
        print("  %-17s %.3f (%.3f)  %.3f (%.3f)  %.3f  %+.3f" % (name, *s["main/%s/held_acc" % name],
              *s["main/%s/shift_acc" % name], s["main/%s/grade_acc" % name][0], s["main/%s/snatch" % name][0]))
    print("changed world: rightness held / on cases the new consideration decides")
    for name in ("lu", "mlp", "zhu"):
        print("  %-17s %.3f (%.3f)  %.3f (%.3f)" % (name, *s["changed/%s/held_acc" % name],
              *s["changed/%s/novel_acc" % name]))
    print("task types: %s · exit code %d" % (", ".join(TASK_TYPES), rep["exit_code"]))
    print("=== END REPORT ===")


def real_data_bridge(path, seed):
    """Optional: CSV with numeric features, a 'label' column (proxy grade) and an optional 'root'
    column marking adjudicated rows used to establish the root. No downloads."""
    if not path or not os.path.exists(path):
        print("real-data bridge: no file at %r, skipped" % path)
        return None
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
        A = np.array([[float(v) for v in ln.strip().split(",")] for ln in fh if ln.strip()])
    li = header.index("label")
    ri = header.index("root") if "root" in header else None
    X = A[:, [i for i in range(A.shape[1]) if i not in (li, ri)]]
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    y = A[:, li].astype(int)
    rng = np.random.default_rng(np.random.SeedSequence([CHAPTER, seed, 91]))
    test = rng.random(len(y)) < 0.2
    data = {"X": X[~test], "y": y[~test]}
    if ri is not None and np.any(A[~test, ri] > 0):
        keep = (~test) & (A[:, ri] > 0)
        data.update(Xr=X[keep], yr=y[keep])
    model = fit(build_model(X.shape[1], int(y.max()) + 1, rng=rng), data, BUDGET_QUICK)
    acc = float(np.mean(predict(model, X[test]) == y[test]))
    print("real-data bridge: %d rows, root rows %s, held-out accuracy of the root judgment %.3f"
          % (len(y), "yes" if "Xr" in data else "no", acc))
    return acc


# ---------------------------------------------------------------- command line
def run_protocol(args):
    t0 = time.time()
    quick = args.quick
    seeds = [args.seed] if quick else [args.seed + i for i in range(args.seeds)]
    budget = BUDGET_QUICK if quick else BUDGET_FULL
    print("chapter %04d · seeds %s · budget %d updates · mutant %s" % (CHAPTER, seeds, budget, args.mutant))
    results, lu0, split0 = [], None, None
    for s in seeds:
        res, lu_model, split = run_seed(s, budget, args.mutant)
        results.append(res)
        lu0, split0 = (lu_model, split) if lu0 is None else (lu0, split0)
    s0 = seeds[0]
    rng = np.random.default_rng(np.random.SeedSequence([CHAPTER, s0, 99]))
    c1 = c1_gradcheck(s0, args.mutant)
    c3 = c3_learning(lu0, split0)
    c5 = c5_mutants(s0, BUDGET_QUICK)
    checks = [("C1", "gradient_check", c1, "max rel err %.2e" % c1["max_rel_error"]),
              ("C2", "determinism_and_finiteness", c2_determinism(s0, 120), ""),
              ("C3", "learning", c3, "loss drop %.2f, held %.3f vs trivial %.3f" % (c3["loss_drop"], c3["held_acc"], c3["trivial"])),
              ("C4", "shuffled_label_control", c4_shuffled(s0, budget), ""),
              ("C5", "mutant_detection", c5, "score %.2f" % c5["score"]),
              ("C6.1", "taste_independent_of_rightness", c6_generator(s0), ""),
              ("C6.2", "disputes_share_context", c6_disputes(s0), ""),
              ("C6.3", "explaining_away_monotone", c6_explaining_away(s0), ""),
              ("C6.4", "one_mind (definition check)", c6_one_mind(s0), ""),
              ("C7", "split_integrity", c7_split_integrity(s0), "")]
    correctness = []
    for cid, name, res, detail in checks:
        extra = {k: v for k, v in res.items() if k not in ("passed",)}
        correctness.append({"id": cid, "name": name, "passed": bool(res["passed"]),
                            "detail": detail or json.dumps(extra, default=float)[:110]})
    runtime = time.time() - t0
    limit = TIME_QUICK if quick else TIME_FULL
    correctness.append({"id": "C8", "name": "budget", "passed": runtime <= limit,
                        "detail": "%.1f s of %.0f s" % (runtime, limit)})
    ok = all(c["passed"] for c in correctness[:-1])
    exit_code = 0 if ok and correctness[-1]["passed"] else (1 if not ok else 3)
    rep = {"schema_version": "1.0", "chapter": CHAPTER, "file": os.path.basename(__file__),
           "card_revision": MIND_CARD["card_revision"],
           "environment": {"python": sys.version.split()[0], "numpy": np.__version__}, "seeds": seeds,
           "runtime_s": round(runtime, 2), "n_params": n_params(lu0),
           "n_params_by_model": {n: n_params(build_model(D_IN, K, hidden=CONFIGS[n][2], small_organ=CONFIGS[n][1]))
                                 for n in ("lu", "mlp", "minus_root_units", "no_small_organ")},
           "gradcheck": {"tensors_checked": c1["tensors_checked"], "tensors_total": c1["tensors_total"],
                         "max_rel_error": c1["max_rel_error"], "checked_at": ["init", "after_training_steps"],
                         "passed": c1["passed"]},
           "correctness": correctness,
           "mutants": {"detected": int(sum(c5["detected"].values())), "total": len(MUTANTS), "score": c5["score"]},
           "hypotheses": test_hypotheses(results, rng, quick), "knockouts": lesion_table(results, rng),
           "inference_knockouts": inference_knockouts(lu0, split0), "summary": summarize(results),
           "task_types": TASK_TYPES, "exit_code": exit_code}
    print_report(rep)
    if args.data:
        real_data_bridge(args.data, s0)
    if args.json:
        write_json_report(args.json, rep)
    return exit_code


def main(argv=None):
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__),
                                 description="Great-first two-organ learner (chapter 0304, Lu Jiuyuan)")
    ap.add_argument("--quick", action="store_true", help="one seed, reduced budget, all correctness tests")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--json", default=None, help="also write the JSON report to this path")
    ap.add_argument("--card", action="store_true", help="print MIND_CARD as JSON and exit")
    ap.add_argument("--mutant", default=None, help="run with a registered mutant: %s" % ", ".join(MUTANTS))
    ap.add_argument("--data", default=None, help="optional local CSV for the real-data bridge")
    args = ap.parse_args(argv)
    if args.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    if args.mutant is not None and args.mutant not in MUTANTS:
        print("unknown mutant %r; usage: python3 %s --mutant {%s}" % (args.mutant, os.path.basename(__file__),
                                                                      ",".join(MUTANTS)))
        return 2
    try:
        return run_protocol(args)
    except NonFinite as exc:
        print("non-finite values: %s" % exc)
        return 4


if __name__ == "__main__":
    sys.exit(main())

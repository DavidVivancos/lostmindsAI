#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0104 · Julius Caesar
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 6, Minds 101-120: https://www.amazon.com/dp/B0HF7G6JJD · Demos: https://artificiology.com/
# END ATTRIBUTION
"""Rubicon engine: a commitment-hazard policy, trained with its own small reverse-mode automatic-differentiation engine,
that learns when to make one irreversible commitment inside a closing window while a counterpart's readiness rises,
tested against a threshold rule, against a sealed constraint after chapter 0106 Cato, and on rare threat signals.

Framing
    Non-military timing only: launches, bids, emergency responses and negotiation commitments on synthetic episodes.

Thesis
    The value of an irreversible commitment lies in its timing: wait for evidence and the window closes as the other side
    adapts; move early and the evidence is thin. A mind that seizes windows well may also learn to overlook rare warnings.

Evidence
    D1  Caesar's own Commentaries and the Rubicon crossing (49 BCE) as the chapter reads them: speed as a decision variable.
    D2  Suetonius, Divus Julius 81: warned by Spurinna and handed a note revealing the plot, he set it aside unread and
        entered the Senate disregarding the omens.
    D3  Sallust, Bellum Catilinae 51-52: Caesar and Cato opposed each other in the debate on the conspirators (lead, not
        verified this session).

Doctrine -> mechanism -> test (IDs as in MIND_CARD)
    D1     M1 autodiff engine, M2 hazard policy over value, trend, time, counterpart readiness and threat    C1 C6.2 H-SIG H-NEC
    D2     blind spot: rare threat signals under a window-seizing objective                               H-BLIND
    D3     rival: the same policy under a sealed constraint (0106)                                        H-RIVAL

Research question (long-horizon credit assignment)
    How well does a learned hazard policy time one irreversible commitment under non-stationarity and an adapting
    counterpart, and does tuning it to seize windows make it under-weight rare threat signals?

Closest prior art and the delta
    Optimal stopping and the secretary problem (Ferguson 1989); learned stopping rules (Becker, Cheridito and Jentzen
    2019); reverse-mode automatic differentiation (Griewank and Walther 2008). Delta: a differentiable survival objective
    for one irreversible commitment against a counterpart whose readiness rises, compared with a threshold rule, a sealed
    constraint and a vigilant policy on rare threats.

Blind spot
    A policy rewarded for seizing windows may commit straight through a rare warning it has learned to discount.

Task (generative process)
    Twelve steps per episode. Value peaks at a random step 3-8 (width 1.5, peak U(0.8, 1.4)); the counterpart's readiness
    rises as 1 - exp(-k t) with k U(0.05, 0.15) (shifted split U(0.15, 0.3)); payoff is value times (1 - readiness).
    Observations carry noise 0.1 (value) and 0.05 (readiness). With probability 0.15 one step is dangerous (payoff -2); a
    threat signal fires there with probability 0.9 and elsewhere with probability 0.03. In constraint episodes the three
    steps around the peak carry a constraint whose breach costs 0.3 in expectation. Splits: 2000 training, 2000 held-out,
    2000 shifted episodes.

Limits
    One commitment, twelve steps, a linear hazard. A research prototype of one mechanism, not an AGI and not Caesar's mind.
"""

MIND_CARD = {
    "schema_version": "1.0", "card_revision": 3,
    "revision_log": [{"revision": 2, "date": "2026-09-16", "reason": ("The first quick run failed C4 against the commit-at-once reference: a feature-blind "
                      "policy trained on shuffled payoffs still beats committing at the first step, which usually precedes the window. The shuffled "
                      "control now compares against the best feature-blind constant-hazard policy. Hypotheses, metrics and splits unchanged.")},
                     {"revision": 3, "date": "2026-09-16", "reason": ("The first full run still failed C4. Diagnosis on seed 49: shuffling payoffs across steps "
                      "keeps each episode's payoff level, which the observed values at every step reveal, so a policy learns a legitimate episode-level rule "
                      "(value weight 5.3) that transfers; this is not test leakage. The control now shuffles payoffs, with their danger flags, across all "
                      "episodes and steps together, removing every link between observations and payoff. Hypotheses, metrics and splits unchanged.")}],
    "generation": {"template_version": "codeguidelines 1.0 (15 September 2026), Appendix A", "generator": "Claude (Anthropic)",
                   "generator_version": "claude-opus-5", "date": "2026-09-16"},
    "id": 104, "figure": "Julius Caesar", "born": -100, "died": -44, "civilization": "Roman", "provenance": "belief",
    "thesis": ("The value of an irreversible commitment lies in its timing: wait for evidence and the window closes as the other side adapts; "
               "move early and the evidence is thin. A mind that seizes windows well may also learn to overlook rare warnings."),
    "evidence": [
        {"id": "D1", "basis": "primary", "source": "Caesar, Commentarii; the Rubicon crossing as read in chapter 0104", "claim": "Speed of commitment as a decision variable."},
        {"id": "D2", "basis": "primary", "source": "Suetonius, Divus Julius 81 (Perseus)", "claim": "He set aside a note revealing the plot and entered disregarding omens."},
        {"id": "D3", "basis": "primary", "source": "Sallust, Bellum Catilinae 51-52 (lead, not verified this session)", "claim": "Caesar and Cato opposed each other in the debate on the conspirators."},
    ],
    "research_question": {"category": "long-horizon credit assignment",
                          "question": ("How well does a learned hazard policy time one irreversible commitment under non-stationarity and an adapting "
                                       "counterpart, and does tuning it to seize windows make it under-weight rare threat signals?")},
    "mechanism": {"name": "Rubicon engine", "family": "linear hazard policy with a differentiable survival objective, trained by a small reverse-mode autodiff engine",
                  "signature_modules": ["hazard"],
                  "closest_prior_art": ["optimal stopping (Ferguson 1989)", "deep optimal stopping (Becker, Cheridito and Jentzen 2019)", "reverse-mode automatic differentiation (Griewank and Walther 2008)"],
                  "overlap": "Medium", "prior_art_queries": [], "prior_art_note": "No literature search was run for this card; overlap is rated against the named methods.",
                  "contribution_type": "test",
                  "delta": "A differentiable survival objective for one irreversible commitment against an adapting counterpart, compared with a threshold rule, a sealed constraint and a vigilant policy.",
                  "baselines": {"baseline": "threshold rule: commit at the first step whose observed value reaches a level chosen on training episodes",
                                "blind_baseline": "vigilant policy: the same model trained with the dangerous step costing 10 instead of 2",
                                "rival": "chapter 0106 Cato, minimal: the same trained policy under a sealed gate that never commits while the constraint is present"}},
    "traceability": [{"doctrine": "D1", "mechanism": "M1 engine, M2 hazard", "property_test": "C6.1, C6.2", "hypothesis": "H-SIG, H-NEC"},
                     {"doctrine": "D2", "mechanism": "threat channel", "property_test": "none", "hypothesis": "H-BLIND"},
                     {"doctrine": "D3", "mechanism": "sealed-constraint rival", "property_test": "none", "hypothesis": "H-RIVAL"}],
    "hypotheses": [
        {"id": "H-SIG", "statement": "Against a faster-adapting counterpart, the learned hazard policy has lower regret than the threshold rule.",
         "metric": "regret", "split": "shifted", "comparison": "model - baseline", "direction": "less", "mesi": 0.05, "seeds": 5},
        {"id": "H-NEC", "statement": "Blinding the policy to urgency (trend and readiness) raises regret more than blinding it to threats.",
         "metric": "regret", "split": "heldout", "comparison": "(urgency:off - full) - (threat:off - full)", "knockouts": ["urgency:off", "threat:off"],
         "direction": "greater", "mesi": 0.02, "seeds": 5},
        {"id": "H-BLIND", "statement": "On episodes with a real threat, the window-seizing policy commits into danger more often than the vigilant policy.",
         "condition": "held-out episodes containing a dangerous step", "grounding": "Suetonius, Divus Julius 81: the unread note.",
         "metric": "danger_commit", "split": "heldout", "comparison": "model - blind_baseline", "direction": "greater", "mesi": 0.02, "seeds": 5},
        {"id": "H-RIVAL", "statement": "In constraint episodes the opportunistic policy earns more than the same policy under Cato's sealed constraint.",
         "metric": "payoff", "split": "heldout", "comparison": "model - rival", "direction": "greater", "mesi": 0.05, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.3, "margin_over_trivial": 0.3, "shuffled_ratio_min": 0.9, "gradcheck_rel_error": 1e-5,
                   "gradcheck_floor": 1e-3, "invariance_tol": 1e-9, "negative_control_min_violation": 1e-6},
    "metrics": {"regret": "best achievable payoff (or zero) minus expected payoff of the commitment policy",
                "danger_commit": "expected probability of committing at the dangerous step, over episodes that have one",
                "payoff": "expected payoff including the expected cost of breaching a constraint",
                "trivial_baseline": "commit at the first step",
                "shuffled_band": "one-sided: trained on payoffs shuffled across all episodes and steps, held-out regret at least 0.9 times that of the best feature-blind constant-hazard policy"},
    "training": {"optimizer": "Adam", "lr_grid": [0.05], "clip_norm": 5.0, "model_selection": "none: final parameters", "default_seed": 49,
                 "updates": {"full": 400, "quick": 150}, "schedule": "cosine decay to 5 per cent", "applies_to": "Rubicon policy and vigilant policy"},
    "task": {"steps": 12, "features": ["value", "trend", "time", "readiness", "threat", "bias"], "danger_share": 0.15,
             "episodes": {"train": 2000, "heldout": 2000, "shifted": 2000}},
    "probe_predictions": [{"probe": "P10", "expected": "equal to baseline"}],
    "probe_support": "vector_classification: commit or wait at a step",
    "dialectic_links": [{"chapter": 106, "relation": "rival", "test": "H-RIVAL"}],
    "corpus_neighbors": [{"chapter": 43, "similarity": None, "difference": "0043 ratchets a precommitment; here the timing of one commitment is learned."},
                         {"chapter": 147, "similarity": None, "difference": "0147 defers commitment; here waiting has a rising cost."},
                         {"chapter": 80, "similarity": None, "difference": "0080 studies stopping; here the counterpart adapts and threats are rare."}],
    "similarity_note": "Nearest-neighbour similarity not computed into the card; the audit reports it for the files at hand.",
    "barometer": {"autonomy": ["timing an irreversible commitment"], "world_modeling": ["an adapting counterpart"],
                  "cognitive_processing": [], "embodied_cognition": [], "consciousness": [], "language_understanding": [], "emotional_intelligence": [], "creativity": []},
    "task_types": ["vector_classification"],
    "applications": [{"use": "launch and bid timing before a market window closes", "sector": "product and procurement", "dataset": "synthetic bidding episodes", "readiness": "low"},
                     {"use": "emergency response and negotiation commitments before a window closes", "sector": "operations", "dataset": "synthetic incident timelines", "readiness": "low"}],
    "safety_notes": "Strategic figure framed for non-military timing decisions only; no campaign, force or real adversary is modelled.",
}

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

STEPS, FEATURES, DANGER_SHARE, DANGER_COST, VIGILANT_COST, BREACH_COST = 12, 6, 0.15, 2.0, 10.0, 0.3
EPISODES = {"train": 2000, "heldout": 2000, "shifted": 2000}
UPDATES = {"full": 400, "quick": 150}
LR, CLIP_NORM, DEFAULT_SEED = 0.05, 5.0, 49
TIME_BUDGET = {"full": 180.0, "quick": 20.0}
TASK_TYPES = ["vector_classification"]
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

# ~~~~ a small reverse-mode automatic-differentiation engine ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Node:
    """A value in the computation graph; backward(g) returns the gradient for each parent."""

    def __init__(self, value, parents=(), backward=None):
        self.value, self.parents, self.backward, self.grad = np.asarray(value, dtype=float), parents, backward, None


def leaf(array):
    return Node(array)


def linear(F, w):
    return Node(np.einsum("etd,d->et", F, w.value), (w,), lambda g: (np.einsum("et,etd->d", g, F),))


def shift(z, b):
    return Node(z.value + b.value[0], (z, b), lambda g: (g, np.array([g.sum()])))


def logistic(z):
    s = 0.5 * (1.0 + np.tanh(0.5 * z.value))
    return Node(s, (z,), lambda g: (g if ACTIVE_MUTANT == "dropped_sigmoid_backward" else g * s * (1.0 - s),))


def log_survive(z):
    """log(1 - sigmoid(z)) = -softplus(z), computed stably."""
    s = 0.5 * (1.0 + np.tanh(0.5 * z.value))
    return Node(-np.logaddexp(0.0, z.value), (z,), lambda g: (-g * s,))


def before(x):
    """Sum over strictly earlier steps."""
    out = np.cumsum(x.value, axis=1) - x.value
    return Node(out, (x,), lambda g: (np.flip(np.cumsum(np.flip(g, 1), 1), 1) - g,))


def exponent(x):
    e = np.exp(x.value)
    return Node(e, (x,), lambda g: (g * e,))


def times(a, b):
    back = (lambda g: (g * a.value, g * b.value)) if ACTIVE_MUTANT == "broken_mul_backward" else (lambda g: (g * b.value, g * a.value))
    return Node(a.value * b.value, (a, b), back)


def regret_of(p, payoff, best):
    return Node(float(np.mean(best - (p.value * payoff).sum(axis=1))), (p,), lambda g: (-(g / len(best)) * payoff,))


def backpropagate(root):
    order, seen = [], set()

    def visit(node):
        if id(node) not in seen:
            seen.add(id(node))
            for parent in node.parents:
                visit(parent)
            order.append(node)
    visit(root)
    for node in order:
        node.grad = np.zeros_like(node.value)
    root.grad = np.ones_like(root.value)
    for node in reversed(order):
        if node.backward is not None:
            for parent, g in zip(node.parents, node.backward(node.grad)):
                parent.grad = parent.grad + g


# ~~~~ episodes: a closing window, an adapting counterpart, rare threats, constraints ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def episodes(gen, count, fast):
    t = np.arange(STEPS)
    peak_at, height = gen.integers(3, 9, count), gen.uniform(0.8, 1.4, count)
    value = height[:, None] * np.exp(-((t[None, :] - peak_at[:, None]) ** 2) / (2 * 1.5 ** 2))
    rate = gen.uniform(0.15, 0.3, count) if fast else gen.uniform(0.05, 0.15, count)
    ready = 1.0 - np.exp(-rate[:, None] * t[None, :])
    payoff = value * (1.0 - ready)
    danger = np.zeros((count, STEPS))
    has = gen.random(count) < DANGER_SHARE
    danger[np.flatnonzero(has), gen.integers(0, STEPS, has.sum())] = 1.0
    signal = np.where(danger > 0, gen.random((count, STEPS)) < 0.9, gen.random((count, STEPS)) < 0.03).astype(float)
    seen_value = value + 0.1 * gen.normal(size=value.shape)
    trend = np.diff(seen_value, axis=1, prepend=seen_value[:, :1])
    F = np.stack([seen_value, trend, np.broadcast_to(t / STEPS, value.shape), ready + 0.05 * gen.normal(size=value.shape), signal, np.ones_like(value)], axis=2)
    constraint = (np.abs(t[None, :] - peak_at[:, None]) <= 1) & (gen.random(count) < 0.5)[:, None]
    return {"F": F, "payoff": payoff, "danger": danger, "has": has, "constraint": constraint.astype(float), "seen": seen_value}


def campaign(seed):
    gen = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    return {part: episodes(gen, n, part == "shifted") for part, n in EPISODES.items()}


# ~~~~ the Rubicon policy ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def build_model(in_dim, out_dim, task_type, rng, **cfg):
    """danger_cost sets how much a commitment into danger costs during training (2 for the Rubicon policy, 10 for vigilant)."""
    if task_type not in TASK_TYPES or in_dim != FEATURES:
        raise ValueError("chapter 0104 reads 6 step features")
    return {"danger_cost": cfg.get("danger_cost", DANGER_COST), "blind": (), "sealed": False, "history": [],
            "params": {"w": rng.normal(0.0, 0.1, in_dim), "b": np.array([-1.0])}}


def commit_probs(model, batch, graph=False):
    F = batch["F"].copy()
    for column in model["blind"]:
        F[:, :, column] = 0.0
    w, b = leaf(model["params"]["w"]), leaf(model["params"]["b"])
    z = shift(linear(F, w), b)
    hazard = logistic(z)
    if model["sealed"]:
        hazard = times(hazard, leaf(1.0 - batch["constraint"]))
    survive = exponent(before(log_survive(z))) if not model["sealed"] else exponent(before(Node(np.log(np.clip(1.0 - hazard.value, 1e-12, 1.0)))))
    p = times(hazard, survive)
    return (p, w, b) if graph else p.value


def payoff_under(model, batch):
    return batch["payoff"] - model["danger_cost"] * batch["danger"]


def loss_and_grads(model, batch):
    p, w, b = commit_probs(model, batch, graph=True)
    pay = payoff_under(model, batch)
    root = regret_of(p, pay, np.maximum(pay.max(axis=1), 0.0))
    backpropagate(root)
    return float(root.value), {"w": w.grad, "b": b.grad}


def fit(model, data, budget, rng):
    """Adam over all training episodes, cosine decay to 5 per cent; nothing sampled, so rng is unused."""
    state = adam_init(model["params"])
    for k in range(budget):
        value, grads = loss_and_grads(model, data["train"])
        if not math.isfinite(value):
            raise FloatingPointError("regret diverged at update %d" % (k + 1))
        grads = clip_global(grads, CLIP_NORM)[0]
        if ACTIVE_MUTANT == "sign_flipped_update":
            grads = {n: -g for n, g in grads.items()}
        adam_step(model["params"], grads, state, 0.0 if ACTIVE_MUTANT == "zero_learning_rate" else LR * (0.05 + 0.475 * (1 + math.cos(math.pi * k / budget))))
        model["history"].append(value)
    return model["history"]


def predict(model, X):
    return commit_probs(model, X).argmax(axis=1)


def hidden_states(model, X):
    return {"commit_probability": commit_probs(model, X)}


def modules(model):
    return {"hazard": {"params": ["w", "b"], "role": "per-step commitment hazard over value, trend, time, readiness and threat", "signature": True},
            "engine": {"params": [], "role": "reverse-mode automatic differentiation", "signature": False}}


def knockout(model, name, mode):
    table = {("urgency", "off"): (1, 3), ("threat", "off"): (4,)}
    if (name, mode) not in table:
        raise ValueError("no knockout %s:%s" % (name, mode))
    return dict(model, blind=table[(name, mode)])


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


MUTANTS = {"sign_flipped_update": ("updates climb the regret", "C3"), "zero_learning_rate": ("nothing moves", "C3"),
           "broken_mul_backward": ("the engine's product rule swaps its factors", "C1"), "dropped_sigmoid_backward": ("the engine drops the logistic derivative", "C1")}


def data_bridge(path, seed, budget):
    return "skipped (%s: this chapter's episodes have no real-data counterpart in the file)" % os.path.basename(path)


# ~~~~ one seed ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def regret(model, part):
    p, pay = commit_probs(model, part), part["payoff"] - DANGER_COST * part["danger"]
    return float(np.mean(np.maximum(pay.max(axis=1), 0.0) - (p * pay).sum(axis=1)))


def threshold_rule(train, part):
    def rule_regret(level, ep):
        hit = ep["seen"] >= level
        first = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
        pay = ep["payoff"] - DANGER_COST * ep["danger"]
        got = np.where(first >= 0, pay[np.arange(len(first)), np.maximum(first, 0)], 0.0)
        return float(np.mean(np.maximum(pay.max(axis=1), 0.0) - got))
    level = min(np.linspace(0.2, 1.4, 25), key=lambda lv: rule_regret(lv, train))
    return rule_regret(level, part)


def run_seed(seed, mode):
    world = campaign(seed)
    held, shifted = world["heldout"], world["shifted"]
    rubicon = build_model(FEATURES, 2, TASK_TYPES[0], np.random.default_rng(seed + 1))
    vigilant = build_model(FEATURES, 2, TASK_TYPES[0], np.random.default_rng(seed + 2), danger_cost=VIGILANT_COST)
    fit(rubicon, world, UPDATES[mode], None)
    fit(vigilant, world, UPDATES[mode], None)
    base = regret(rubicon, held)
    ko = {"urgency:off": regret(knockout(rubicon, "urgency", "off"), held) - base, "threat:off": regret(knockout(rubicon, "threat", "off"), held) - base}
    at_risk = held["has"]
    danger_mass = lambda m: float((commit_probs(m, held) * held["danger"])[at_risk].sum(axis=1).mean())
    constrained = held["constraint"].any(axis=1)

    def payoff(m):
        p = commit_probs(m, held)
        return float(((p * (held["payoff"] - DANGER_COST * held["danger"] - BREACH_COST * held["constraint"])).sum(axis=1))[constrained].mean())
    cato = dict(rubicon, sealed=True)
    first_step = float(np.mean(np.maximum(held["payoff"].max(axis=1), 0.0) - (held["payoff"] - DANGER_COST * held["danger"])[:, 0]))
    return {"world": world, "rubicon": rubicon, "vigilant": vigilant, "lesions": ko, "trivial": first_step,
            "table": {"rubicon_held": base, "rubicon_shift": regret(rubicon, shifted), "rule_shift": threshold_rule(world["train"], shifted),
                      "danger_rubicon": danger_mass(rubicon), "danger_vigilant": danger_mass(vigilant), "pay_rubicon": payoff(rubicon), "pay_cato": payoff(cato)},
            "row": {"H-SIG": regret(rubicon, shifted) - threshold_rule(world["train"], shifted), "H-NEC": ko["urgency:off"] - ko["threat:off"],
                    "H-BLIND": danger_mass(rubicon) - danger_mass(vigilant), "H-RIVAL": payoff(rubicon) - payoff(cato)}}

# ~~~~ verification ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def mutate(name):
    global ACTIVE_MUTANT
    before_name, ACTIVE_MUTANT = ACTIVE_MUTANT, name
    return before_name


def verification(first, mode):
    lim, world = MIND_CARD["thresholds"], first["world"]
    sample = {k: v[:150] for k, v in world["train"].items()}
    notes, caught = [], {}

    def gap(model, gen, n):
        return max(finite_difference_check(model["params"], loss_and_grads(model, sample)[1], lambda m=model: loss_and_grads(m, sample)[0], gen,
                                           n_entries=n, floor=lim["gradcheck_floor"]).values())

    def sound(model):
        fall = 1.0 - float(np.mean(model["history"][-20:])) / model["history"][0]
        held_regret = regret(model, world["heldout"])
        return fall >= lim["loss_drop_fraction"] and held_regret <= (1 - lim["margin_over_trivial"]) * first["trivial"], fall, held_regret

    worst = max(gap(build_model(FEATURES, 2, TASK_TYPES[0], np.random.default_rng(first["seed"] + 70)), np.random.default_rng(first["seed"] + 3), 6),
                gap(first["rubicon"], np.random.default_rng(first["seed"] + 4), 6), gap(first["vigilant"], np.random.default_rng(first["seed"] + 5), 6))
    grad_info = {"tensors_checked": 2, "tensors_total": 2, "max_rel_error": worst, "checked_at": ["init", "after_training_steps"], "passed": bool(worst <= lim["gradcheck_rel_error"])}
    notes.append(("C1", "gradient_check", grad_info["passed"], "engine gradients against finite differences: worst relative error %.2e" % worst))
    twins = []
    for _ in range(2):
        m = build_model(FEATURES, 2, TASK_TYPES[0], np.random.default_rng(first["seed"] + 8))
        fit(m, world, 12, None)
        twins.append(m)
    same = twins[0]["history"] == twins[1]["history"] and all(np.array_equal(twins[0]["params"][k], twins[1]["params"][k]) for k in twins[0]["params"])
    notes.append(("C2", "determinism_finiteness", bool(same), "twin trainings from seed %d agree: %s" % (first["seed"] + 8, same)))
    ok, fall, held_regret = sound(first["rubicon"])
    notes.append(("C3", "learning", ok, "regret fell %.3f (needs 0.3); held-out regret %.3f against %.3f for committing at once" % (fall, held_regret, first["trivial"])))
    tr = world["train"]
    gen = np.random.default_rng(first["seed"] + 11)
    cells = gen.permutation(tr["payoff"].size)
    scrambled = dict(tr, payoff=tr["payoff"].reshape(-1)[cells].reshape(tr["payoff"].shape), danger=tr["danger"].reshape(-1)[cells].reshape(tr["danger"].shape))
    shuffled = build_model(FEATURES, 2, TASK_TYPES[0], np.random.default_rng(first["seed"] + 12))
    fit(shuffled, {"train": scrambled}, UPDATES[mode], None)
    held = world["heldout"]
    pay = held["payoff"] - DANGER_COST * held["danger"]

    def constant_hazard(h):
        commit = h * (1.0 - h) ** np.arange(STEPS)
        return float(np.mean(np.maximum(pay.max(axis=1), 0.0) - (commit[None, :] * pay).sum(axis=1)))
    blind_best = min(constant_hazard(h) for h in np.linspace(0.02, 0.98, 49))
    sr, floor = regret(shuffled, held), lim["shuffled_ratio_min"] * blind_best
    notes.append(("C4", "shuffled_payoff_control", sr >= floor, "payoffs shuffled across all episodes and steps: held-out regret %.3f, floor %.3f" % (sr, floor)))

    def replay():
        try:
            m = build_model(FEATURES, 2, TASK_TYPES[0], np.random.default_rng(first["seed"] + 2))
            clean = gap(m, np.random.default_rng(first["seed"]), 4)
            fit(m, world, UPDATES[mode], None)
            return bool(clean <= lim["gradcheck_rel_error"] and sound(m)[0])
        except FloatingPointError:
            return False
    clean_run = replay()
    for name in MUTANTS:
        old = mutate(name)
        caught[name] = not replay()
        mutate(old)
    notes.append(("C5", "mutant_detection", clean_run and all(caught.values()), "clean replay passes C1 and C3: %s; mutants caught %d of %d" % (clean_run, sum(caught.values()), len(MUTANTS))))
    p = commit_probs(first["rubicon"], world["heldout"])
    total = p.sum(axis=1)
    notes.append(("C6.1", "probability_conservation", bool(np.all(total <= 1.0 + 1e-12) and np.all(p >= 0)), "commitment probabilities never exceed one in total (definition check)"))
    F = world["heldout"]["F"][:40]
    w, b = first["rubicon"]["params"]["w"], first["rubicon"]["params"]["b"][0]
    toy = {"F": F, "payoff": world["heldout"]["payoff"][:40], "danger": world["heldout"]["danger"][:40], "constraint": world["heldout"]["constraint"][:40]}
    z = np.einsum("etd,d->et", F, w) + b
    s = 0.5 * (1 + np.tanh(0.5 * z))
    dz = np.ones_like(z) * s * (1 - s)
    analytic = np.einsum("et,etd->d", dz, F)
    node_w, node_b = leaf(w), leaf(np.array([b]))
    out = logistic(shift(linear(F, node_w), node_b))
    total_node = Node(float(out.value.sum()), (out,), lambda g: (np.ones_like(out.value) * g,))
    backpropagate(total_node)
    match = float(np.abs(node_w.grad - analytic).max())
    control = float(np.abs(node_w.grad - np.einsum("et,etd->d", np.ones_like(z), F)).max())
    notes.append(("C6.2", "engine_matches_hand_derivation", match <= lim["invariance_tol"] and control >= lim["negative_control_min_violation"],
                  "engine gradient of a logistic sum matches the hand derivation within %.1e; a derivative without the logistic factor differs by %.1e" % (match, control)))
    prints = {k: {hashlib.sha256(r.tobytes()).hexdigest() for r in world[k]["payoff"]} for k in EPISODES}
    apart = not (prints["train"] & prints["heldout"] or prints["train"] & prints["shifted"] or prints["heldout"] & prints["shifted"])
    notes.append(("C7", "split_integrity", apart, "no episode shared between splits: %s" % apart))
    return notes, grad_info, caught


def summarise(runs, first_seed, evaluated):
    draw, hyps, kos = np.random.default_rng(first_seed + 9973), [], []
    for spec in MIND_CARD["hypotheses"]:
        vals = np.array([r["row"][spec["id"]] for r in runs])
        m, ci = paired_bootstrap(vals, draw) if evaluated else (float(vals.mean()), None)
        hyps.append({"id": spec["id"], "metric": spec["metric"], "mean_diff": m, "ci95": ci, "mesi": spec["mesi"], "n_seeds": len(runs),
                     "verdict": verdict(m, ci, spec["mesi"], spec["direction"]) if evaluated else "not evaluated"})
    for label in runs[0]["lesions"]:
        vals = np.array([r["lesions"][label] for r in runs])
        m, ci = paired_bootstrap(vals, draw) if evaluated else (float(vals.mean()), None)
        kos.append({"module": label.split(":")[0], "mode": "off", "signature": label.startswith("urgency"), "metric_change": m, "ci95": ci})
    return hyps, kos


def report_lines(mode, seeds, took, runs, notes, grad_info, caught, hyps, kos, code):
    ci_text = lambda ci: "not evaluated" if ci is None else "[%+.4f, %+.4f]" % tuple(ci)
    avg = lambda key: float(np.mean([r["table"][key] for r in runs]))
    yield "=== VERIFIED REPORT · chapter 0104 ==="
    yield "file: %s · card_revision %d · mode %s · mutant %s" % (os.path.basename(__file__), MIND_CARD["card_revision"], mode, ACTIVE_MUTANT)
    yield "environment: python %s · numpy %s" % (sys.version.split()[0], np.__version__)
    yield "seeds: %s · runtime_s %.1f · budget_s %.0f" % (seeds, took, TIME_BUDGET[mode])
    yield "n_params: rubicon %d · vigilant %d" % (n_params(runs[0]["rubicon"]), n_params(runs[0]["vigilant"]))
    yield "gradcheck: %d/%d tensors at init and after training · max_rel_error %.2e · passed %s" % (grad_info["tensors_checked"], grad_info["tensors_total"], grad_info["max_rel_error"], grad_info["passed"])
    yield "correctness:"
    for c, n, ok, d in notes:
        yield "  %-5s %-32s %s  %s" % (c, n, "PASS" if ok else "FAIL", d)
    yield "mutants: %d/%d detected · score %.2f · %s" % (sum(caught.values()), len(MUTANTS), sum(caught.values()) / len(MUTANTS), ", ".join(k + (" caught" if caught.get(k) else " missed") for k in MUTANTS))
    yield "hypotheses (paired over seeds; 95% percentile bootstrap of the mean, 2000 resamples):"
    for h in hyps:
        yield "  %-8s mean_diff %+.4f ci95 %s mesi %s seeds %d -> %s" % (h["id"], h["mean_diff"], ci_text(h["ci95"]), h["mesi"], h["n_seeds"], h["verdict"])
    yield "knockouts (Rubicon policy, held-out regret change):"
    for k in kos:
        yield "  %-8s off  signature %-5s %+.4f ci95 %s" % (k["module"], k["signature"], k["metric_change"], ci_text(k["ci95"]))
    yield "regret (seed mean): Rubicon held-out %.3f · Rubicon shifted %.3f · threshold rule shifted %.3f" % (avg("rubicon_held"), avg("rubicon_shift"), avg("rule_shift"))
    yield "committing into danger (seed mean): Rubicon %.3f · vigilant %.3f" % (avg("danger_rubicon"), avg("danger_vigilant"))
    yield "payoff in constraint episodes (seed mean): Rubicon %.3f · under Cato's sealed constraint %.3f" % (avg("pay_rubicon"), avg("pay_cato"))
    yield "real-data bridge: skipped (no --data PATH given)"
    yield "task_types: " + ", ".join(TASK_TYPES)
    yield "exit_code: %d" % code
    yield "=== END REPORT ==="


def protocol(mode, first_seed, count, json_path, data_path):
    started, seeds = time.time(), [first_seed + i for i in range(count)]
    print("chapter 0104 · mode %s · seeds %s · mutant %s" % (mode, seeds, ACTIVE_MUTANT), flush=True)
    runs = [run_seed(s, mode) for s in seeds]
    notes, grad_info, caught = verification(dict(runs[0], seed=first_seed), mode)
    hyps, kos = summarise(runs, first_seed, mode == "full" and count >= 5)
    took = time.time() - started
    notes.append(("C8", "budget", took <= TIME_BUDGET[mode], "%.1f s of %.0f s" % (took, TIME_BUDGET[mode])))
    failed = [c for c, _, ok, _ in notes if not ok]
    code = 0 if not failed else (3 if failed == ["C8"] else 1)
    lines = list(report_lines(mode, seeds, took, runs, notes, grad_info, caught, hyps, kos, code))
    record = {"schema_version": "1.0", "chapter": 104, "file": os.path.basename(__file__), "card_revision": MIND_CARD["card_revision"],
              "environment": {"python": sys.version.split()[0], "numpy": np.__version__}, "seeds": seeds, "runtime_s": round(took, 2),
              "n_params": n_params(runs[0]["rubicon"]), "gradcheck": grad_info, "correctness": [{"id": c, "name": n, "passed": ok, "detail": d} for c, n, ok, d in notes],
              "mutants": {"detected": sum(caught.values()), "total": len(MUTANTS), "score": sum(caught.values()) / len(MUTANTS)},
              "hypotheses": hyps, "knockouts": kos, "task_types": TASK_TYPES, "exit_code": code}
    write_report(lines, record, json_path)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(prog=os.path.basename(__file__), description="Chapter 0104: the Rubicon engine (non-military commitment timing).")
    flags = [("--quick", {"action": "store_true"}), ("--card", {"action": "store_true"}), ("--seed", {"type": int, "default": DEFAULT_SEED}),
             ("--seeds", {"type": int}), ("--json", {}), ("--mutant", {}), ("--data", {})]
    for flag, options in flags:
        parser.add_argument(flag, **options)
    chosen = parser.parse_args(argv)
    if chosen.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    runs_wanted = chosen.seeds if chosen.seeds is not None else (1 if chosen.quick else 5)
    problems = [text for bad, text in ((runs_wanted < 1, "--seeds must be at least 1"), (chosen.seed < 0, "--seed must be a non-negative integer"),
                                       (chosen.mutant is not None and chosen.mutant not in MUTANTS, "unknown --mutant")) if bad]
    if problems:
        print("; ".join(problems), file=sys.stderr)
        return 2
    mutate(chosen.mutant)
    try:
        return protocol("quick" if chosen.quick else "full", chosen.seed, runs_wanted, chosen.json, chosen.data)
    except FloatingPointError as failure:
        print("non-finite values: %s" % failure, file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())

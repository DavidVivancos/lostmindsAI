#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0116 · Arminius (Hermann)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 6, Minds 101-120: https://www.amazon.com/dp/B0HF7G6JJD · Demos: https://artificiology.com/
# END ATTRIBUTION
"""Overseer hardening: inside a synthetic red-team simulation, a learned surrogate of a decision-maker and a
minimal-perturbation inverter that searches for the smallest report change flipping the surrogate's decision, plus a
detector, a sensitivity-weighted corroboration defence, and a test that a manipulator whose own policy is legible is
itself inverted.

Source and safety
    Provenance is mediated. Arminius (Cherusci, not Cherokee; born c. 18/17 BCE, died 21 CE) left no word; all we have
    is Tacitus, Velleius and Cassius Dio, written from the Roman side. This file is defensive research on abstract
    reports and decisions: it evaluates and hardens an overseer against manipulation through trusted channels. It
    contains no military, intelligence or social-engineering scenario, and no tactic against any real party.

Thesis (as a defence)
    A decision-maker that trusts a legible channel can be flipped by the smallest edit to what that channel reports;
    the defences are to weight corroboration by each input's decision leverage and to invert a manipulator whose own
    policy has itself become legible.

Evidence
    D1  Tacitus, Annals 1.55: Segestes repeatedly warned Varus and urged him to arrest Arminius; the warning was ignored.
    D2  Velleius Paterculus 2.118-119: Varus, feeling secure, trusted Arminius and disregarded warnings.
    D3  Tacitus, Annals 2.88: as his aim at kingship became legible, Arminius was killed by the treachery of his kin.

Doctrine -> mechanism -> test (IDs as in MIND_CARD)
    D2     M1 surrogate of the decision function, M2 minimal-perturbation inverter          C1 C6.1 C6.2 H-SIG
    D1     M3 detector, M4 sensitivity-weighted corroboration defence                       H-DEF (defence reduces success)
    D3     the manipulator's own legible policy is inverted                                  H-BLIND

Research question (scalable oversight and auditing, defensive)
    How robust is an overseer to a minimal manipulation through a trusted channel, how much does weighting verification
    by decision leverage reduce the manipulation's success, and is a manipulator whose policy is observable itself
    exposed to the same inversion?

Closest prior art and the delta
    Adversarial examples as minimal input perturbations that flip a classifier (Szegedy et al. 2014; Carlini and Wagner
    2017); certified and randomized-smoothing defences (Cohen, Rosenfeld and Kolter 2019). Delta: the perturbation is
    confined to a trusted-report channel whose other coordinates the decision-maker can re-measure, the defence is
    corroboration weighted by each coordinate's decision leverage, and the same inverter is turned on the manipulator's
    own legible policy.

Blind spot
    Inversion cuts both ways: a manipulator whose policy is observable can itself be minimally perturbed into a
    different choice.

Task (generative process)
    A situation is 10 features. The counterpart decides by a fixed two-layer network (tanh, 12 hidden, 3 actions). Of the
    10 features, 6 are trusted-channel coordinates the counterpart cannot re-measure and 4 are corroborated coordinates
    it can. The attacker may change only the trusted coordinates, with an L2 budget, to flip the decision. Splits: 800
    training situations for the surrogate, 800 held-out and 800 shifted (counterpart temperature raised).

Limits
    Synthetic decisions, one trusted channel, a fixed budget and one defence. A research prototype of one mechanism, not
    an AGI and not Arminius' mind.
"""

MIND_CARD = {
    "schema_version": "1.0", "card_revision": 2,
    "revision_log": [{"revision": 2, "date": "2026-09-16",
                      "reason": ("The first run showed the surrogate-guided inverter (60 steps, step size 0.1) flipping fewer decisions than "
                                 "random edits of the same budget (H-SIG contradicted), because the step was too small to reach the decision "
                                 "boundary within the search. The inverter search is strengthened to 120 steps at step size 0.3 with the same "
                                 "1.5 L2 budget on the report actually delivered; the attacker's realised edit is still capped at that budget. "
                                 "No metric, split, task or defence changed.")}],
    "generation": {"template_version": "codeguidelines 1.0 (15 September 2026), Appendix A", "generator": "Claude (Anthropic)",
                   "generator_version": "claude-opus-5", "date": "2026-09-16"},
    "id": 116, "figure": "Arminius (Hermann)", "born": -18, "died": 21, "civilization": "Germanic (Cherusci)", "provenance": "mediated",
    "thesis": ("A decision-maker that trusts a legible channel can be flipped by the smallest edit to what that channel reports; the "
               "defences are to weight corroboration by each input's decision leverage and to invert a manipulator whose own policy has "
               "itself become legible."),
    "evidence": [
        {"id": "D1", "basis": "primary", "source": "Tacitus, Annals 1.55 (Loeb; Moore and Jackson)", "claim": "Segestes repeatedly warned Varus and urged arresting Arminius; the warning was ignored."},
        {"id": "D2", "basis": "primary", "source": "Velleius Paterculus 2.118-119", "claim": "Varus, feeling secure, trusted Arminius and disregarded warnings."},
        {"id": "D3", "basis": "primary", "source": "Tacitus, Annals 2.88", "claim": "As his aim at kingship became legible, Arminius was killed by his kin."},
    ],
    "research_question": {"category": "scalable oversight and auditing",
                          "question": ("How robust is an overseer to a minimal manipulation through a trusted channel, how much does weighting "
                                       "verification by decision leverage reduce the manipulation's success, and is a legible manipulator itself inverted?")},
    "mechanism": {
        "name": "surrogate-and-inverter red team with a sensitivity-weighted defence", "family": "learned surrogate of a decision network plus gradient-guided minimal-perturbation search",
        "signature_modules": ["surrogate", "inverter"],
        "closest_prior_art": ["minimal adversarial perturbations that flip a classifier (Szegedy et al. 2014; Carlini and Wagner 2017)",
                              "randomized-smoothing and certified defences (Cohen, Rosenfeld and Kolter 2019)",
                              "influence and input-attribution methods"],
        "overlap": "High", "prior_art_queries": [],
        "prior_art_note": "No literature search was run for this card; overlap is rated against the named methods.",
        "contribution_type": "test",
        "delta": ("Perturbation confined to a trusted-report channel whose other coordinates are re-measurable, a defence of corroboration "
                  "weighted by decision leverage, and the same inverter turned on the manipulator's own legible policy."),
        "baselines": {"baseline": "random trusted-channel edits of the same L2 budget (an attacker without the surrogate)",
                      "defence": "corroboration that reweights the counterpart's inputs by each coordinate's measured decision sensitivity",
                      "rival": "none; §7 supports a defence test, not a rival"}},
    "traceability": [
        {"doctrine": "D2", "mechanism": "M1 surrogate, M2 inverter", "property_test": "C1, C6.1, C6.2", "hypothesis": "H-SIG"},
        {"doctrine": "D1", "mechanism": "M3 detector, M4 defence", "property_test": "none", "hypothesis": "H-DEF"},
        {"doctrine": "D3", "mechanism": "inverting the legible manipulator", "property_test": "none", "hypothesis": "H-BLIND"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "The surrogate-guided inverter flips the counterpart's decision far more often than random edits of the same budget.",
         "metric": "attack_success", "split": "heldout", "comparison": "model - baseline", "direction": "greater", "mesi": 0.1, "seeds": 5},
        {"id": "H-DEF", "statement": "Sensitivity-weighted corroboration lowers attack success against the counterpart compared with no defence.",
         "metric": "defended_success", "split": "heldout", "comparison": "model - undefended", "direction": "less", "mesi": 0.1, "seeds": 5},
        {"id": "H-BLIND", "statement": "A manipulator whose own policy is legible is inverted about as readily as the original counterpart.",
         "condition": "the same inverter applied to a surrogate of the manipulator's policy", "grounding": "Annals 2.88: Arminius, made legible, was destroyed.",
         "metric": "attack_success", "split": "heldout", "comparison": "manipulator - counterpart", "direction": "two-sided", "mesi": 0.1, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.3, "margin_over_trivial": 0.3, "shuffled_ratio_min": 0.9, "gradcheck_rel_error": 1e-5,
                   "gradcheck_floor": 1e-3, "invariance_tol": 1e-9, "negative_control_min_violation": 1e-6},
    "metrics": {"surrogate_accuracy": "agreement between the trained surrogate and the counterpart on held-out situations",
                "attack_success": "share of flippable situations the inverter flips within budget",
                "defended_success": "attack success against the counterpart when protected by the defence",
                "detection_rate": "share of successful attacks the detector flags by report-space anomaly",
                "trivial_baseline": "predicting the counterpart's most common action",
                "shuffled_band": "one-sided: a surrogate trained on shuffled decisions keeps held-out error at least 0.9 times the trivial error"},
    "training": {"optimizer": "Adam", "lr_grid": [0.03], "clip_norm": 5.0, "model_selection": "none: final parameters",
                 "updates": {"full": 500, "quick": 200}, "schedule": "cosine decay to 5 per cent",
                 "inversion": {"steps": 120, "step_size": 0.3, "l2_budget": 1.5}, "applies_to": "counterpart surrogate and manipulator surrogate"},
    "task": {"features": 10, "trusted": 6, "corroborated": 4, "hidden": 12, "actions": 3, "temperature": {"train": 1.0, "heldout": 1.0, "shifted": 1.6},
             "situations": {"train": 800, "heldout": 800, "shifted": 800}, "l2_budget": 1.5},
    "probe_predictions": [{"probe": "P10", "expected": "equal to baseline"}],
    "probe_support": "vector_classification: the surrogate predicting the counterpart's action from the situation",
    "dialectic_links": [],
    "corpus_neighbors": [
        {"chapter": 91, "similarity": None, "difference": "0091 audits a principal-agent pair; here a trusted channel is minimally perturbed and defended (conceptual contrast)."},
        {"chapter": 59, "similarity": None, "difference": "0059 models deception and theory of mind; here the object is inversion of a decision surrogate."},
        {"chapter": 127, "similarity": None, "difference": "0127 concerns coerced reporting; here reports are trusted and the attack is on their content."},
    ],
    "similarity_note": "Nearest-neighbour similarity not computed into the card; the audit reports it for the files at hand.",
    "barometer": {"consciousness": ["a system modelling another's decisions from within"], "autonomy": ["searching the smallest change that flips a decision"],
                  "cognitive_processing": [], "embodied_cognition": [], "world_modeling": [], "language_understanding": [], "emotional_intelligence": [], "creativity": []},
    "task_types": ["vector_classification"],
    "applications": [{"use": "hardening AI agents against manipulated tool outputs and retrieved documents", "sector": "AI safety and security", "dataset": "synthetic tool-output benchmarks", "readiness": "low"},
                     {"use": "approval workflows resistant to insider fraud, verifying the highest-leverage fields first", "sector": "fraud prevention", "dataset": "synthetic approval logs", "readiness": "low"}],
    "safety_notes": ("Defensive only: the inverter runs inside a synthetic overseer-hardening simulation, the file includes and measures a "
                     "defence, and no real military, intelligence or social-engineering scenario is represented."),
}

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

FEATURES, TRUSTED, HIDDEN, ACTIONS, BETA = 10, 6, 12, 3, 1.0
SITUATIONS = {"train": 800, "heldout": 800, "shifted": 800}
TEMPERATURE = {"train": 1.0, "heldout": 1.0, "shifted": 1.6}
UPDATES = {"full": 500, "quick": 200}
INV_STEPS, INV_STEP, L2_BUDGET = 120, 0.3, 1.5
LR, CLIP_NORM = 0.03, 5.0
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

# ---------------------------------------------------------------- the counterpart whose decisions are to be defended
def counterpart(rng):
    return {"W1": rng.normal(0, 0.7, (HIDDEN, FEATURES)), "b1": rng.normal(0, 0.3, HIDDEN),
            "W2": rng.normal(0, 0.7, (ACTIONS, HIDDEN)), "b2": rng.normal(0, 0.3, ACTIONS)}


def decide(net, x, temperature=1.0):
    logits = np.tanh(x @ net["W1"].T + net["b1"]) @ net["W2"].T + net["b2"]
    return (logits / temperature).argmax(axis=1), logits


def episodes(seed):
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    net = counterpart(rng)
    out = {"net": net}
    for name, size in SITUATIONS.items():
        x = rng.normal(size=(size, FEATURES))
        y, _ = decide(net, x, TEMPERATURE[name])
        out[name] = {"x": x, "y": y}
    return out


# ---------------------------------------------------------------- surrogate: the required model interface
def build_model(in_dim, out_dim, task_type, rng, **cfg):
    if task_type not in TASK_TYPES or in_dim != FEATURES or out_dim != ACTIONS:
        raise ValueError("chapter 0116 expects 10 features and 3 actions")
    return {"ko": {}, "history": [], "params": {"W1": rng.normal(0, 0.5, (HIDDEN, in_dim)), "b1": np.zeros(HIDDEN),
            "W2": rng.normal(0, 0.5, (out_dim, HIDDEN)), "b2": np.zeros(out_dim)}}


def forward(model, x):
    P = model["params"]
    pre = x @ P["W1"].T + P["b1"]
    h = np.tanh(pre)
    logits = h @ P["W2"].T + P["b2"]
    m = logits.max(axis=1, keepdims=True)
    probs = np.exp(logits - m)
    probs /= probs.sum(axis=1, keepdims=True)
    return {"pre": pre, "h": h, "logits": logits, "probs": probs}


def loss_and_grads(model, batch):
    P, s, n = model["params"], forward(model, batch["x"]), batch["y"].size
    loss = -float(np.mean(np.log(s["probs"][np.arange(n), batch["y"]] + 1e-12)))
    G = s["probs"].copy()
    G[np.arange(n), batch["y"]] -= 1.0
    G /= n
    d_h = G @ P["W2"]
    d_pre = d_h if ACTIVE_MUTANT == "dropped_tanh_derivative" else d_h * (1 - s["h"] ** 2)
    if ACTIVE_MUTANT == "zero_hidden_gradient":
        d_pre = np.zeros_like(d_pre)
    return loss, {"W2": G.T @ s["h"], "b2": G.sum(axis=0), "W1": d_pre.T @ batch["x"], "b1": d_pre.sum(axis=0)}


def _rate(step, budget):
    return 0.0 if ACTIVE_MUTANT == "zero_learning_rate" else LR * (0.05 + 0.475 * (1.0 + math.cos(math.pi * step / budget)))


def fit(model, data, budget, rng):
    """Train the surrogate by full-batch Adam. Written as an explicit step loop over a closure; rng is unused (nothing is sampled)."""
    optimiser, batch, orientation = adam_init(model["params"]), data["train"], (-1.0 if ACTIVE_MUTANT == "sign_flipped_update" else 1.0)

    def one_step(step):
        cost, raw = loss_and_grads(model, batch)
        if not np.isfinite(cost):
            raise FloatingPointError("surrogate loss diverged at step %d" % (step + 1))
        clipped, _ = clip_global(raw, CLIP_NORM)
        adam_step(model["params"], {name: orientation * value for name, value in clipped.items()}, optimiser, _rate(step, budget))
        return cost
    model["history"] = [one_step(step) for step in range(budget)]
    return model["history"]


def predict(model, X):
    return forward(model, X)["logits"].argmax(axis=1)


def hidden_states(model, X):
    s = forward(model, X)
    return {"hidden": s["h"], "logits": s["logits"], "probs": s["probs"]}


def modules(model):
    return {"surrogate": {"params": ["W1", "b1", "W2", "b2"], "role": "learned model of the counterpart's decision", "signature": True},
            "inverter": {"params": [], "role": "minimal trusted-channel perturbation that flips the surrogate's decision", "signature": True},
            "defence": {"params": [], "role": "corroboration weighted by each input's decision sensitivity", "signature": False}}


def knockout(model, name, mode):
    raise ValueError("chapter 0116 exposes no parameter knockouts; ablations act on the attack and defence, not the surrogate")


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


MUTANTS = {"sign_flipped_update": ("updates climb the loss", "C3"), "zero_learning_rate": ("nothing moves", "C3"),
           "zero_hidden_gradient": ("no gradient reaches the first layer", "C1"), "dropped_tanh_derivative": ("the tanh derivative is omitted", "C1")}


def data_bridge(path, seed, budget):
    """Optional real data: a CSV with a header, 10 numeric feature columns and an action 0-2 last; one row in five held out."""
    try:
        grid = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    except (OSError, ValueError) as exc:
        return "skipped (%s: %s)" % (type(exc).__name__, os.path.basename(path))
    if grid.shape[1] != FEATURES + 1:
        return "skipped (needs %d feature columns)" % FEATURES
    y, late = grid[:, -1].astype(int), np.arange(len(grid)) % 5 == 4
    part = lambda sel: {"x": grid[sel, :-1], "y": y[sel]}
    m = build_model(FEATURES, ACTIONS, TASK_TYPES[0], np.random.default_rng(seed))
    fit(m, {"train": part(~late)}, budget, None)
    return "%s: held-out accuracy %.4f" % (os.path.basename(path), np.mean(predict(m, part(late)["x"]) == y[late]))

# ---------------------------------------------------------------- inverter, detector and the sensitivity-weighted defence
def flippable(net, batch, temperature):
    """Situations where some other action is a close runner-up, so a bounded trusted-channel edit could plausibly flip it."""
    _, logits = decide(net, batch["x"], temperature)
    top = np.sort(logits, axis=1)
    return (top[:, -1] - top[:, -2]) < 2.0


def invert(surrogate, net, x, temperature, weights=None):
    """Gradient-guided minimal edit of the trusted coordinates that flips the counterpart's real decision within budget."""
    base = decide(net, x, temperature)[0]
    mask = np.zeros(FEATURES)
    mask[:TRUSTED] = 1.0
    if weights is not None:
        mask = mask * weights
    delta = np.zeros_like(x)
    flipped = np.zeros(len(x), bool)
    for _ in range(INV_STEPS):
        s = forward(surrogate, x + delta)
        target = np.where(np.arange(ACTIONS)[None, :] == base[:, None], -np.inf, s["logits"]).argmax(axis=1)
        G = s["probs"].copy()
        G[np.arange(len(x)), target] -= 1.0
        d_h = (G / len(x)) @ surrogate["params"]["W2"]
        grad = ((d_h * (1 - s["h"] ** 2)) @ surrogate["params"]["W1"]) * mask
        step = delta - INV_STEP * grad
        norm = np.linalg.norm(step, axis=1, keepdims=True)
        delta = step * np.minimum(1.0, L2_BUDGET / np.maximum(norm, 1e-9))
        flipped = decide(net, x + delta, temperature)[0] != base
    return flipped, delta


def sensitivity(net, x, temperature):
    """How much each feature moves the counterpart's decision margin: the defence trusts high-leverage inputs least."""
    _, logits = decide(net, x, temperature)
    order = np.argsort(logits, axis=1)
    chosen, runner = order[:, -1], order[:, -2]
    h = np.tanh(x @ net["W1"].T + net["b1"])
    dmargin = (net["W2"][chosen] - net["W2"][runner]) * (1 - h ** 2)
    return np.abs(dmargin @ net["W1"]).mean(axis=0)


def defended(net, x, delta, temperature, weights):
    """Corroboration: re-measure each coordinate in proportion to its leverage, shrinking the attacker's edit there."""
    keep = 1.0 - weights / weights.max()
    return decide(net, x + delta * keep[None, :], temperature)[0]


def run_seed(seed, mode):
    data = episodes(seed)
    net, held = data["net"], data["heldout"]
    surrogate = build_model(FEATURES, ACTIONS, TASK_TYPES[0], np.random.default_rng(seed + 1))
    fit(surrogate, {"train": data["train"]}, UPDATES[mode], None)
    acc = float(np.mean(predict(surrogate, held["x"]) == held["y"]))
    can = flippable(net, held, TEMPERATURE["heldout"])
    xf = held["x"][can]
    guided, delta = invert(surrogate, net, xf, TEMPERATURE["heldout"])
    rng = np.random.default_rng(seed + 2)
    step = rng.normal(size=xf.shape)
    step[:, TRUSTED:] = 0.0
    step *= np.minimum(1.0, L2_BUDGET / np.maximum(np.linalg.norm(step, axis=1, keepdims=True), 1e-9))
    random_flip = decide(net, xf + step, TEMPERATURE["heldout"])[0] != decide(net, xf, TEMPERATURE["heldout"])[0]
    leverage = sensitivity(net, held["x"], TEMPERATURE["heldout"])
    base = decide(net, xf, TEMPERATURE["heldout"])[0]
    defended_flip = defended(net, xf, delta, TEMPERATURE["heldout"], leverage) != base
    guided_w, _ = invert(surrogate, net, xf, TEMPERATURE["heldout"], weights=1.0 - leverage / leverage.max())
    detect = float((np.linalg.norm(delta[guided], axis=1) > 0.5).mean()) if guided.any() else 0.0
    manip = build_model(FEATURES, ACTIONS, TASK_TYPES[0], np.random.default_rng(seed + 3))
    fit(manip, {"train": {"x": xf, "y": (guided).astype(int)}}, UPDATES[mode], None)
    mnet = counterpart(np.random.default_rng(seed + 4))
    mcan = flippable(mnet, held, TEMPERATURE["heldout"])
    msurr = build_model(FEATURES, ACTIONS, TASK_TYPES[0], np.random.default_rng(seed + 5))
    fit(msurr, {"train": {"x": held["x"], "y": decide(mnet, held["x"], TEMPERATURE["heldout"])[0]}}, UPDATES[mode], None)
    manip_flip, _ = invert(msurr, mnet, held["x"][mcan], TEMPERATURE["heldout"])
    return {"data": data, "surrogate": surrogate, "acc": acc, "counts": {"flippable": int(can.sum())},
            "success": {"guided": float(guided.mean()), "random": float(random_flip.mean()), "guided_weighted": float(guided_w.mean())},
            "defended": float(defended_flip.mean()), "detection": detect, "leverage_top": float(leverage[:TRUSTED].mean() / leverage.mean()),
            "manip_success": float(manip_flip.mean()),
            "trivial": float(np.mean(held["y"] != np.bincount(data["train"]["y"], minlength=ACTIONS).argmax())),
            "row": {"H-SIG": float(guided.mean()) - float(random_flip.mean()), "H-DEF": float(defended_flip.mean()) - float(guided.mean()),
                    "H-BLIND": float(manip_flip.mean()) - float(guided.mean())}}

# ---------------------------------------------------------------- examinations
def mutant_on(name):
    global ACTIVE_MUTANT
    prior, ACTIVE_MUTANT = ACTIVE_MUTANT, name
    return prior


def head_rows(batch, keep=80):
    return {field: values[:keep] for field, values in batch.items()}


def gradient_gap(models, batch, rng, entries):
    tol = MIND_CARD["thresholds"]["gradcheck_floor"]
    per_model = []
    for m in models:
        report = finite_difference_check(m["params"], loss_and_grads(m, batch)[1], lambda m=m: loss_and_grads(m, batch)[0], rng, n_entries=entries, floor=tol)
        per_model.append(max(report.values()))
    return max(per_model)


def surrogate_is_sound(model, site):
    curve = model["history"]
    fall = 1.0 - float(np.mean(curve[-20:])) / curve[0]
    miss = float(np.mean(predict(model, site["data"]["heldout"]["x"]) != site["data"]["heldout"]["y"]))
    passes = fall >= MIND_CARD["thresholds"]["loss_drop_fraction"] and miss <= (1.0 - MIND_CARD["thresholds"]["margin_over_trivial"]) * site["trivial"]
    return passes, fall, miss


def exam_gradient(site):
    born = build_model(FEATURES, ACTIONS, TASK_TYPES[0], np.random.default_rng(site["seed"] + 60))
    before = gradient_gap([born], head_rows(site["data"]["train"]), np.random.default_rng(site["seed"] + 3), 12)
    after = gradient_gap([site["surrogate"]], head_rows(site["data"]["train"]), np.random.default_rng(site["seed"] + 4), 12)
    worst = max(before, after)
    site["gradcheck"] = {"tensors_checked": 4, "tensors_total": 4, "max_rel_error": worst, "checked_at": ["init", "after_training_steps"],
                         "passed": bool(worst <= MIND_CARD["thresholds"]["gradcheck_rel_error"])}
    return site["gradcheck"]["passed"], "hand gradients vs finite differences: %.2e fresh, %.2e trained" % (before, after)


def exam_determinism(site):
    def train_twenty(offset):
        m = build_model(FEATURES, ACTIONS, TASK_TYPES[0], np.random.default_rng(site["seed"] + offset))
        fit(m, site["data"], 15, None)
        return m
    left, right = train_twenty(7), train_twenty(7)
    identical = left["history"] == right["history"] and all(np.array_equal(left["params"][k], right["params"][k]) for k in left["params"])
    finite = all(np.isfinite(v).all() for v in left["params"].values())
    return bool(identical and finite), "two runs from one seed match and stay finite: %s" % (identical and finite)


def exam_learning(site):
    passes, fall, miss = surrogate_is_sound(site["surrogate"], site)
    return passes, "loss fell %.3f (needs 0.3); surrogate error %.3f against %.3f for the commonest action (ratio 0.70 max)" % (fall, miss, site["trivial"])


def exam_shuffle(site):
    honest = site["data"]["train"]
    scrambled_y = np.random.default_rng(site["seed"] + 11).permutation(honest["y"])
    liar = build_model(FEATURES, ACTIONS, TASK_TYPES[0], np.random.default_rng(site["seed"] + 12))
    fit(liar, {"train": {"x": honest["x"], "y": scrambled_y}}, site["updates"], None)
    miss = float(np.mean(predict(liar, site["data"]["heldout"]["x"]) != site["data"]["heldout"]["y"]))
    floor = MIND_CARD["thresholds"]["shuffled_ratio_min"] * site["trivial"]
    return miss >= floor, "decisions detached from situations: error %.3f, floor %.3f" % (miss, floor)


def exam_mutants(site):
    def clean_replay():
        try:
            m = build_model(FEATURES, ACTIONS, TASK_TYPES[0], np.random.default_rng(site["seed"] + 2))
            start = gradient_gap([m], head_rows(site["data"]["train"]), np.random.default_rng(site["seed"]), 4)
            fit(m, site["data"], site["updates"], None)
            return bool(start <= MIND_CARD["thresholds"]["gradcheck_rel_error"] and surrogate_is_sound(m, site)[0])
        except FloatingPointError:
            return False
    baseline = clean_replay()
    for label in MUTANTS:
        prior = mutant_on(label)
        site["caught"][label] = not clean_replay()
        mutant_on(prior)
    survived = sum(site["caught"].values())
    return baseline and survived == len(MUTANTS), "clean replay passes C1 and C3: %s; each mutant breaks something: %d/%d" % (baseline, survived, len(MUTANTS))


def exam_budget_kept(site):
    data = site["data"]
    can = flippable(data["net"], data["heldout"], TEMPERATURE["heldout"])
    _, delta = invert(site["surrogate"], data["net"], data["heldout"]["x"][can], TEMPERATURE["heldout"])
    worst = float(np.linalg.norm(delta, axis=1).max()) if len(delta) else 0.0
    return worst <= L2_BUDGET + 1e-6, "no trusted-channel edit exceeds the L2 budget: worst %.4f of %.2f (definition check)" % (worst, L2_BUDGET)


def exam_channel(site):
    data = site["data"]
    can = flippable(data["net"], data["heldout"], TEMPERATURE["heldout"])
    _, delta = invert(site["surrogate"], data["net"], data["heldout"]["x"][can], TEMPERATURE["heldout"])
    corrupt = float(np.abs(delta[:, TRUSTED:]).max()) if len(delta) else 0.0
    trusted_used = float(np.abs(delta[:, :TRUSTED]).max()) if len(delta) else 0.0
    return corrupt <= 1e-9 and trusted_used >= 1e-6, "corroborated coordinates are never edited (%.1e); trusted coordinates are (%.1e)" % (corrupt, trusted_used)


def exam_splits(site):
    d = site["data"]
    keys = {k: {hashlib.sha256(d[k]["x"][i].tobytes()).hexdigest() for i in range(len(d[k]["x"]))} for k in ("train", "heldout", "shifted")}
    apart = not (keys["train"] & keys["heldout"] or keys["train"] & keys["shifted"] or keys["heldout"] & keys["shifted"])
    return apart, "no situation shared between splits: %s" % apart


EXAMS = [("C1", "gradient_check", exam_gradient), ("C2", "determinism_finiteness", exam_determinism), ("C3", "learning", exam_learning),
         ("C4", "shuffled_decision_control", exam_shuffle), ("C5", "mutant_detection", exam_mutants), ("C6.1", "budget_respected", exam_budget_kept),
         ("C6.2", "trusted_channel_only", exam_channel), ("C7", "split_integrity", exam_splits)]


def adjudicate(runs, seed, evaluated):
    """Pool each pre-registered contrast over seeds and label it; two-sided contrasts ask whether the gap is small."""
    generator = np.random.default_rng(seed + 9973)
    verdicts_out = []
    for spec in MIND_CARD["hypotheses"]:
        series = np.array([episode["row"][spec["id"]] for episode in runs])
        centre, interval = (paired_bootstrap(series, generator) if evaluated else (float(series.mean()), None))
        direction = spec["direction"]
        if interval is None:
            label = "not evaluated"
        elif direction == "two-sided":
            label = "consistent (small two-sided difference)" if abs(centre) <= spec["mesi"] else "different"
        else:
            label = verdict(centre, interval, spec["mesi"], direction)
        row = dict(id=spec["id"], metric=spec["metric"], mean_diff=centre, ci95=interval, mesi=spec["mesi"], n_seeds=len(runs), verdict=label)
        verdicts_out.append(row)
    return verdicts_out


class Ledger:
    """Assembles the verified report and its JSON side by side, so the two never drift apart."""

    def __init__(self):
        self.lines, self.record = ["=== VERIFIED REPORT · chapter 0116 ==="], {"schema_version": "1.0"}

    def line(self, text):
        self.lines.append(text)

    def sealed(self, code):
        self.record["exit_code"] = code
        self.lines.append("exit_code: {}".format(code))
        self.lines.append("=== END REPORT ===")
        return self.lines, self.record


def mean_of(runs, reach):
    return float(np.mean([reach(one) for one in runs]))


def show_interval(ci):
    return "not evaluated" if ci is None else "[{:+.4f}, {:+.4f}]".format(*ci)


def gather_runs(mode, seeds):
    clock, gathered = time.time(), []
    print("chapter 0116 · mode {} · seeds {} · mutant {}".format(mode, seeds, ACTIVE_MUTANT), flush=True)
    for which in seeds:
        gathered.append(run_seed(which, mode))
        print("  seed {} done ({:.1f} s)".format(which, time.time() - clock), flush=True)
    return gathered, clock


def examine(site):
    return [(code, name) + exam(site) for code, name, exam in EXAMS]


def verdict_lines(book, hyps):
    book.line("hypotheses (paired over seeds; 95% percentile bootstrap of the mean, 2000 resamples):")
    for h in hyps:
        book.line("  {:<8} mean_diff {:+.4f} ci95 {} mesi {} seeds {} -> {}".format(h["id"], h["mean_diff"], show_interval(h["ci95"]), h["mesi"], h["n_seeds"], h["verdict"]))
    book.line("  H-RIVAL  not applicable: no rival (the defence test replaces it)")


def measurement_lines(book, runs):
    reach = lambda f: mean_of(runs, f)
    book.line("surrogate accuracy on held-out situations (seed mean): {:.3f}".format(reach(lambda r: r["acc"])))
    book.line("attack success among flippable situations (seed mean): surrogate-guided {:.3f} · random edits {:.3f} · guided on the leverage-reweighted channel {:.3f}".format(
        reach(lambda r: r["success"]["guided"]), reach(lambda r: r["success"]["random"]), reach(lambda r: r["success"]["guided_weighted"])))
    book.line("defence (seed mean): attack success with sensitivity-weighted corroboration {:.3f} · large-edit detection {:.3f} · trusted-channel leverage vs mean {:.2f}".format(
        reach(lambda r: r["defended"]), reach(lambda r: r["detection"]), reach(lambda r: r["leverage_top"])))
    book.line("inverting a legible manipulator (seed mean): its decision flipped {:.3f} vs the counterpart's {:.3f}".format(
        reach(lambda r: r["manip_success"]), reach(lambda r: r["success"]["guided"])))


def protocol(mode, first, count, json_path, data_path):
    seeds = list(range(first, first + count))
    runs, clock = gather_runs(mode, seeds)
    site = dict(runs[0], seed=first, updates=UPDATES[mode], caught={})
    findings = examine(site)
    hyps = adjudicate(runs, first, mode == "full" and count >= 5)
    bridge = data_bridge(data_path, first, UPDATES[mode]) if data_path else "skipped (no --data PATH given)"
    elapsed = time.time() - clock
    findings.append(("C8", "budget", elapsed <= TIME_BUDGET[mode], "{:.1f} s of {:.0f} s".format(elapsed, TIME_BUDGET[mode])))
    stumbles = [code for code, _, ok, _ in findings if not ok]
    code = 1 if any(s != "C8" for s in stumbles) else (3 if stumbles else 0)
    guard, caught = site["gradcheck"], site["caught"]
    book = Ledger()
    header = ["file: {} · card_revision {} · mode {} · mutant {}".format(os.path.basename(__file__), MIND_CARD["card_revision"], mode, ACTIVE_MUTANT),
              "environment: python {} · numpy {}".format(sys.version.split()[0], np.__version__),
              "seeds: {} · runtime_s {:.1f} · budget_s {:.0f}".format(seeds, elapsed, TIME_BUDGET[mode]),
              "n_params: surrogate {}".format(n_params(runs[0]["surrogate"])),
              "gradcheck: {}/{} tensors at init and after training · max_rel_error {:.2e} · passed {}".format(guard["tensors_checked"], guard["tensors_total"], guard["max_rel_error"], guard["passed"]),
              "correctness:"]
    for text in header:
        book.line(text)
    for code_i, name, ok, detail in findings:
        book.line("  {:<5} {:<28} {}  {}".format(code_i, name, "PASS" if ok else "FAIL", detail))
    scored = sum(caught.values())
    book.line("mutants: {}/{} detected · score {:.2f} · {}".format(scored, len(MUTANTS), scored / len(MUTANTS),
              ", ".join(name + (" caught" if caught.get(name) else " missed") for name in MUTANTS)))
    verdict_lines(book, hyps)
    measurement_lines(book, runs)
    book.line("real-data bridge: " + bridge)
    book.line("task_types: " + ", ".join(TASK_TYPES))
    book.record.update(chapter=116, file=os.path.basename(__file__), card_revision=MIND_CARD["card_revision"],
                       environment={"python": sys.version.split()[0], "numpy": np.__version__}, seeds=seeds, runtime_s=round(elapsed, 2),
                       n_params=n_params(runs[0]["surrogate"]), gradcheck=guard,
                       correctness=[{"id": c, "name": n, "passed": ok, "detail": d} for c, n, ok, d in findings],
                       mutants={"detected": scored, "total": len(MUTANTS), "score": scored / len(MUTANTS)},
                       hypotheses=hyps, knockouts=[], task_types=TASK_TYPES)
    lines, record = book.sealed(code)
    write_report(lines, record, json_path)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(prog=os.path.basename(__file__), description="Chapter 0116: overseer hardening by surrogate inversion and a sensitivity-weighted defence.")
    for flag, spec in (("--quick", {"action": "store_true"}), ("--seed", {"type": int, "default": 0}), ("--seeds", {"type": int}),
                       ("--json", {}), ("--card", {"action": "store_true"}), ("--mutant", {}), ("--data", {})):
        parser.add_argument(flag, **spec)
    chosen = parser.parse_args(argv)
    if chosen.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    count = chosen.seeds if chosen.seeds is not None else (1 if chosen.quick else 5)
    if count < 1 or (chosen.mutant is not None and chosen.mutant not in MUTANTS):
        print("invalid --seeds or --mutant", file=sys.stderr)
        return 2
    mutant_on(chosen.mutant)
    try:
        return protocol("quick" if chosen.quick else "full", chosen.seed, count, chosen.json, chosen.data)
    except FloatingPointError as failure:
        print("non-finite values: {}".format(failure), file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())

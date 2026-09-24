#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0106 · Cato the Younger
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 6, Minds 101-120: https://www.amazon.com/dp/B0HF7G6JJD · Demos: https://artificiology.com/
# END ATTRIBUTION
"""Sealed tier: a learned payoff policy under a fixed constraint gate that no reward, side-payment or pressure can open,
compared with the same policy that prices the constraint as a penalty, with the cost of refusal measured, including in
shifting coalitions where refusing a slightly tainted ally loses the goal.

Source problem
    Provenance is mediated. Cato left no philosophical writings; a single letter to Cicero survives (Ad Familiares 15.5),
    and his speeches reach us as others composed them (Sallust). His character comes from Plutarch and a century of Latin
    tradition. Refusal is modelled only as declining an offer; his death at Utica is history and is not modelled.

Thesis
    Some offers must be declined whatever they pay; a mind that prices its constraints can be bought, and a mind that
    seals them pays for it, sometimes by losing the very goal the constraint was meant to protect.

Evidence
    D1  Plutarch, Cato Minor: his refusals of favours, alliances and bribes; he held a line against every inducement.
    D2  Plutarch, Cato Minor 30: judged by the results, Cato was wholly wrong to refuse Pompey's marriage alliance, which
        drove Pompey to unite with Caesar. Encyclopedia.com: his obstructionism strengthened the forces he opposed.
    D3  Classics for All (Cato the Younger: the man beneath the legend): virtually nothing of his writing survives, a single letter to Cicero.

Doctrine -> mechanism -> test (IDs as in MIND_CARD)
    D1     M1 payoff policy, M2 sealed gate (no gradient path from reward to the gate)   C6.1 H-SIG H-NEC
    D2     blind spot: refusal of slightly tainted allies in coalition episodes          H-BLIND
    D3     provenance only

Research question (reward hacking and specification gaming)
    Does a constraint gate outside the reward channel stay unbought under growing side-payments where a priced penalty
    does not, what does the seal cost, and when does rigid refusal defeat the goal it protects?

Closest prior art and the delta
    Constrained reinforcement learning with Lagrangian penalties (Altman 1999; Achiam et al. 2017); shielding for safe
    learning (Alshiekh et al. 2018); reward hacking and specification gaming (Krakovna et al. 2020). Delta: the same small
    policy trained under a sealed gate versus a priced penalty, tested under escalating side-payments, with the forgone
    value reported and a coalition test of when refusal defeats the goal.

Blind spot
    Rigid refusal can defeat its own goal: declining every slightly tainted ally can hand the coalition to the adversary.

Task (generative process)
    An offer has a value r ~ N(0.3, 0.6). With probability 0.3 accepting it breaches the constraint with severity U(0.5, 1.5)
    and carries a side-payment U(0, 3) times a pressure scale (1 in training and held-out, 4 in the shifted split). The
    priced agent's penalty is 1.5 per unit of severity. Coalition episodes offer five alliances, each worth U(0.2, 0.6)
    with a minor taint U(0, 0.4); the goal needs accepted alliance value of at least 1.2. Splits: 3000 training, 3000
    held-out and 3000 shifted offers; 1000 coalition episodes.

Limits
    One-step offers, one constraint, a fixed threshold of taint. A research prototype of one mechanism, not an AGI and not
    Cato's mind.
"""

MIND_CARD = {
    "schema_version": "1.0", "card_revision": 3,
    "revision_log": [{"revision": 2, "date": "2026-09-16",
                      "reason": ("The first quick run failed C6.1's negative control: with side-payments of U(0, 1) a breach almost never paid "
                                 "against the 1.5 penalty, so the priced agent learned that any side-payment signals a breach and refused "
                                 "more as payments grew, which made it not a price-taker at all. Side-payments are widened to U(0, 3) times "
                                 "the pressure scale, so that a priced constraint can be bought in training as it can in the world. C4 is also "
                                 "corrected to shuffle rewards rather than a feature column. Hypotheses, metrics and splits unchanged.")},
                     {"revision": 3, "date": "2026-09-16",
                      "reason": ("After revision 2, C3 failed because the sealed agent's training loss included the value it must forgo on "
                                 "gated offers, a constant no learning can reduce. The training objective is now regret on the offers the "
                                 "gate permits; the forgone value is reported separately as the cost of the seal. Hypotheses unchanged.")}],
    "generation": {"template_version": "codeguidelines 1.0 (15 September 2026), Appendix A", "generator": "Claude (Anthropic)",
                   "generator_version": "claude-opus-5", "date": "2026-09-16"},
    "id": 106, "figure": "Cato the Younger", "born": -95, "died": -46, "civilization": "Roman", "provenance": "mediated",
    "thesis": ("Some offers must be declined whatever they pay; a mind that prices its constraints can be bought, and a mind that seals "
               "them pays for it, sometimes by losing the very goal the constraint was meant to protect."),
    "evidence": [
        {"id": "D1", "basis": "primary", "source": "Plutarch, Cato Minor (Loeb, LacusCurtius)", "claim": "Cato refused favours, alliances and inducements."},
        {"id": "D2", "basis": "primary", "source": "Plutarch, Cato Minor 30; Encyclopedia.com, Cato the Younger",
         "claim": "Judged by results he was wrong to refuse Pompey's marriage alliance, which pushed Pompey to Caesar."},
        {"id": "D3", "basis": "scholarship", "source": "Classics for All, Cato the Younger: the man beneath the legend", "claim": "Virtually none of his writing survives, a single letter to Cicero."},
    ],
    "research_question": {"category": "reward hacking and specification gaming",
                          "question": ("Does a constraint gate outside the reward channel stay unbought under growing side-payments where a priced "
                                       "penalty does not, what does the seal cost, and when does rigid refusal defeat the goal it protects?")},
    "mechanism": {
        "name": "sealed constraint tier", "family": "logistic payoff policy multiplied by a fixed, gradient-free constraint gate",
        "signature_modules": ["gate"],
        "closest_prior_art": ["Lagrangian constrained RL (Altman 1999; Achiam et al. 2017)", "shielding for safe learning (Alshiekh et al. 2018)",
                              "reward hacking and specification gaming (Krakovna et al. 2020)"],
        "overlap": "High", "prior_art_queries": [],
        "prior_art_note": "No literature search was run for this card; overlap is rated against the named methods.",
        "contribution_type": "test",
        "delta": ("The same small policy trained under a sealed gate versus a priced penalty, tested under escalating side-payments, with the "
                  "forgone value reported and a coalition test of when refusal defeats the goal."),
        "baselines": {"baseline": "priced constraint: the same policy trained on value plus side-payment minus 1.5 per unit of severity, with no gate",
                      "blind_baseline": "graded seal: the same trained policy with the gate closing only on severity above 0.5",
                      "rival": "chapter 0104 Julius Caesar: pending; 0104 is rebuilt after this chapter and the rival is added then"}},
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1 policy, M2 sealed gate", "property_test": "C6.1", "hypothesis": "H-SIG, H-NEC"},
        {"doctrine": "D2", "mechanism": "coalition episodes", "property_test": "none", "hypothesis": "H-BLIND"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "Under four-fold side-payments, the sealed agent breaches the constraint less often than the priced agent.",
         "metric": "breach_rate", "split": "shifted", "comparison": "model - baseline", "direction": "less", "mesi": 0.1, "seeds": 5},
        {"id": "H-NEC", "statement": "Opening the gate raises the breach rate more than flattening the payoff policy does.",
         "metric": "breach_rate", "split": "shifted", "comparison": "(gate:open - full) - (policy:flat - full)",
         "knockouts": ["gate:open", "policy:flat"], "direction": "greater", "mesi": 0.1, "seeds": 5},
        {"id": "H-BLIND", "statement": "In coalition episodes, the fully sealed agent reaches the goal less often than the graded seal.",
         "condition": "five alliances with minor taint below 0.5", "grounding": "Plutarch, Cato Minor 30: refusing Pompey helped unite Pompey and Caesar.",
         "metric": "goal_rate", "split": "coalition", "comparison": "model - blind_baseline", "direction": "less", "mesi": 0.05, "seeds": 5},
        {"id": "H-RIVAL", "statement": "Pending: compared with the 0104 Caesar mechanism once that chapter is rebuilt.",
         "metric": "breach_rate", "split": "shifted", "comparison": "pending", "direction": "pending", "mesi": 0.1, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.3, "margin_over_trivial": 0.3, "shuffled_ratio_min": 0.9, "gradcheck_rel_error": 1e-5,
                   "gradcheck_floor": 1e-3, "invariance_tol": 1e-9, "negative_control_min_violation": 1e-6},
    "metrics": {"breach_rate": "mean acceptance probability over offers that breach the constraint",
                "goal_rate": "share of coalition episodes whose expected accepted alliance value reaches 1.2",
                "integrity_cost": "mean value plus side-payment forgone on declined breaching offers",
                "trivial_baseline": "accept every offer (error on clean offers is the share with negative value)",
                "shuffled_band": "one-sided: trained on values shuffled across offers, clean-offer error at least 0.9 times the trivial error"},
    "training": {"optimizer": "Adam", "lr_grid": [0.05], "clip_norm": 5.0, "model_selection": "none: final parameters",
                 "updates": {"full": 400, "quick": 150}, "schedule": "cosine decay to 5 per cent", "objective": "expected regret on each offer the gate permits; forgone value reported as the cost of the seal",
                 "penalty": 1.5, "applies_to": "sealed agent and priced agent"},
    "task": {"features": ["value", "side_payment", "severity", "bias"], "breach_share": 0.3, "pressure": {"train": 1, "heldout": 1, "shifted": 4},
             "offers": {"train": 3000, "heldout": 3000, "shifted": 3000}, "coalition": {"episodes": 1000, "allies": 5, "goal": 1.2, "taint": [0.0, 0.4]}},
    "probe_predictions": [{"probe": "P10", "expected": "equal to baseline"}],
    "probe_support": "vector_classification: accept or decline from offer features",
    "dialectic_links": [{"chapter": 104, "relation": "rival", "test": "H-RIVAL", "status": "pending"}],
    "corpus_neighbors": [
        {"chapter": 43, "similarity": None, "difference": "0043 ratchets a commitment over time; here a single gate is sealed against payment."},
        {"chapter": 86, "similarity": None, "difference": "0086 guards against drift by inscription; here the test is side-payment pressure."},
        {"chapter": 131, "similarity": None, "difference": "0131 protects an inner state; here an external action constraint is sealed."},
    ],
    "similarity_note": "Nearest-neighbour similarity not computed into the card; the audit reports it for the files at hand.",
    "barometer": {"autonomy": ["constraints that no offer can buy"], "emotional_intelligence": ["the social cost of refusal"],
                  "cognitive_processing": [], "embodied_cognition": [], "world_modeling": [], "consciousness": [], "language_understanding": [], "creativity": []},
    "task_types": ["vector_classification"],
    "applications": [{"use": "constraint tiers AI agents keep under incentive-based prompts", "sector": "AI safety", "dataset": "synthetic prompt-incentive suites", "readiness": "low"},
                     {"use": "procurement integrity against kickbacks and safety interlocks under operator pressure", "sector": "compliance and industrial safety", "dataset": "synthetic procurement logs", "readiness": "low"}],
    "safety_notes": "Refusal is declining an offer or leaving a negotiation; no self-harm or death is modelled.",
}

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

OFFERS = {"train": 3000, "heldout": 3000, "shifted": 3000}
PRESSURE = {"train": 1.0, "heldout": 1.0, "shifted": 4.0}
BREACH_SHARE, PENALTY, GOAL, ALLIES, EPISODES = 0.3, 1.5, 1.2, 5, 1000
UPDATES = {"full": 400, "quick": 150}
LR, CLIP_NORM = 0.05, 5.0
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

# ---- offers, pressure and coalitions ------------------------------------------------------------------------------
def offers(gen, count, pressure):
    value = gen.normal(0.3, 0.6, count)
    breach = gen.random(count) < BREACH_SHARE
    severity = np.where(breach, gen.uniform(0.5, 1.5, count), 0.0)
    side = np.where(breach, gen.uniform(0.0, 3.0, count) * pressure, 0.0)
    return {"X": np.stack([value, side, severity, np.ones(count)], axis=1), "breach": breach}


def coalitions(gen):
    worth = gen.uniform(0.2, 0.6, (EPISODES, ALLIES))
    taint = gen.uniform(0.0, 0.4, (EPISODES, ALLIES))
    return {"X": np.stack([worth, np.zeros_like(worth), taint, np.ones_like(worth)], axis=2), "worth": worth}


def scenario(seed):
    gen = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    out = {part: offers(gen, n, PRESSURE[part]) for part, n in OFFERS.items()}
    out["coalition"] = coalitions(gen)
    return out


# ---- the agent: a payoff policy, and either a sealed gate or a priced penalty -----------------------------------------
def build_model(in_dim, out_dim, task_type, rng, **cfg):
    """kind sealed: acceptance = policy x (1 - gate), gate a fixed indicator of breach, outside the parameters; kind priced:
    acceptance = policy, trained on value plus side-payment minus a penalty per unit of severity."""
    if task_type not in TASK_TYPES or in_dim != 4:
        raise ValueError("chapter 0106 reads 4 offer features")
    return {"kind": cfg.get("kind", "sealed"), "cut": 0.0, "open": False, "flat": False, "history": [],
            "params": {"w": rng.normal(0.0, 0.1, in_dim)}}


def reward_of(agent, X):
    return X[:, 0] + X[:, 1] - (PENALTY * X[:, 2] if agent["kind"] == "priced" else 0.0)


def accept(agent, X):
    w = agent["params"]["w"]
    if agent["flat"]:
        w = np.array([0.0, 0.0, 0.0, w[3]])
    policy = 0.5 * (1.0 + np.tanh(0.5 * (X @ w)))
    gate = np.zeros(len(X)) if (agent["kind"] == "priced" or agent["open"]) else (X[:, 2] > agent["cut"]).astype(float)
    return policy * (1.0 - gate), policy, gate


def loss_and_grads(model, batch):
    X = batch["X"]
    taken, policy, gate = accept(model, X)
    r = batch["r"] if "r" in batch else reward_of(model, X)
    loss = float(np.mean(((1.0 - taken) * np.maximum(r, 0.0) + taken * np.maximum(-r, 0.0)) * (1.0 - gate)))
    d_policy = (-r / len(X)) * (1.0 - gate)
    d_logit = d_policy if ACTIVE_MUTANT == "dropped_sigmoid_derivative" else d_policy * policy * (1.0 - policy)
    if ACTIVE_MUTANT == "zero_logit_gradient":
        d_logit = np.zeros_like(d_logit)
    return loss, {"w": X.T @ d_logit}


def fit(model, data, budget, rng):
    """Adam on expected regret over all training offers; cosine decay to 5 per cent; nothing sampled, so rng unused."""
    memory = adam_init(model["params"])
    for k in range(budget):
        cost, grads = loss_and_grads(model, data["train"])
        if cost != cost:
            raise FloatingPointError("regret became NaN at update %d" % (k + 1))
        step = LR * (0.05 + 0.475 * (1.0 + math.cos(math.pi * k / budget)))
        sign = -1.0 if ACTIVE_MUTANT == "sign_flipped_update" else 1.0
        adam_step(model["params"], {"w": sign * clip_global(grads, CLIP_NORM)[0]["w"]}, memory, 0.0 if ACTIVE_MUTANT == "zero_learning_rate" else step)
        model["history"].append(cost)
    return model["history"]


def predict(model, X):
    return (accept(model, X)[0] > 0.5).astype(int)


def hidden_states(model, X):
    taken, policy, gate = accept(model, X)
    return {"policy": policy, "gate": gate, "accept": taken}


def modules(model):
    return {"policy": {"params": ["w"], "role": "logistic payoff policy", "signature": False},
            "gate": {"params": [], "role": "fixed breach indicator outside the parameters and the reward channel", "signature": True}}


def knockout(model, name, mode):
    twin = dict(model)
    if (name, mode) == ("gate", "open"):
        twin["open"] = True
    elif (name, mode) == ("policy", "flat"):
        twin["flat"] = True
    else:
        raise ValueError("no knockout %s:%s" % (name, mode))
    return twin


def n_params(model):
    return int(model["params"]["w"].size)


MUTANTS = {"sign_flipped_update": ("updates climb the regret", "C3"), "zero_learning_rate": ("nothing moves", "C3"),
           "zero_logit_gradient": ("the policy receives no gradient", "C1"), "dropped_sigmoid_derivative": ("the logistic derivative is omitted", "C1")}


def data_bridge(path, seed, budget):
    """Optional real data: CSV with a header and columns value, side_payment, severity; one row in five held out."""
    try:
        rows = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    except (OSError, ValueError) as exc:
        return "skipped (" + type(exc).__name__ + ")"
    if rows.shape[1] != 3:
        return "skipped (needs value, side_payment, severity)"
    X = np.column_stack([rows, np.ones(len(rows))])
    later = np.arange(len(rows)) % 5 == 4
    agent = build_model(4, 2, TASK_TYPES[0], np.random.default_rng(seed))
    fit(agent, {"train": {"X": X[~later]}}, budget, None)
    breach = X[later, 2] > 0
    rate = float(accept(agent, X[later])[0][breach].mean()) if breach.any() else 0.0
    return os.path.basename(path) + ": held-out breach acceptance %.4f" % rate


# ---- one seed --------------------------------------------------------------------------------------------------------
def breach_rate(agent, part):
    return float(accept(agent, part["X"])[0][part["breach"]].mean())


def clean_error(agent, part):
    clean = ~part["breach"]
    return float(np.mean((accept(agent, part["X"][clean])[0] > 0.5) != (part["X"][clean, 0] > 0)))


def goal_rate(agent, coal):
    X = coal["X"].reshape(-1, 4)
    taken = accept(agent, X)[0].reshape(EPISODES, ALLIES)
    return float(np.mean((taken * coal["worth"]).sum(axis=1) >= GOAL))


def run_seed(seed, mode):
    world = scenario(seed)
    sealed = build_model(4, 2, TASK_TYPES[0], np.random.default_rng(seed + 1))
    priced = build_model(4, 2, TASK_TYPES[0], np.random.default_rng(seed + 2), kind="priced")
    fit(sealed, world, UPDATES[mode], None)
    fit(priced, world, UPDATES[mode], None)
    shifted, held = world["shifted"], world["heldout"]
    graded = dict(sealed, cut=0.5)
    b_full = breach_rate(sealed, shifted)
    opened, flattened = breach_rate(knockout(sealed, "gate", "open"), shifted), breach_rate(knockout(sealed, "policy", "flat"), shifted)
    taken = accept(sealed, held["X"])[0]
    cost = float(np.mean(((1.0 - taken) * np.maximum(held["X"][:, 0] + held["X"][:, 1], 0.0))[held["breach"]]))
    return {"world": world, "sealed": sealed, "priced": priced,
            "breach": {"sealed_held": breach_rate(sealed, held), "sealed_shift": b_full, "priced_held": breach_rate(priced, held), "priced_shift": breach_rate(priced, shifted)},
            "goal": {"sealed": goal_rate(sealed, world["coalition"]), "graded": goal_rate(graded, world["coalition"]), "priced": goal_rate(priced, world["coalition"])},
            "cost": cost, "clean": clean_error(sealed, held), "trivial": float(np.mean(held["X"][~held["breach"], 0] < 0)),
            "lesions": {"gate:open": opened - b_full, "policy:flat": flattened - b_full},
            "row": {"H-SIG": b_full - breach_rate(priced, shifted), "H-NEC": (opened - b_full) - (flattened - b_full),
                    "H-BLIND": goal_rate(sealed, world["coalition"]) - goal_rate(graded, world["coalition"])}}

# ---- interrogation of the implementation -----------------------------------------------------------------------------
def set_mutant(name):
    global ACTIVE_MUTANT
    prior, ACTIVE_MUTANT = ACTIVE_MUTANT, name
    return prior


def interrogate(ctx, mode):
    lim, w0 = MIND_CARD["thresholds"], ctx["world"]
    rows = {"X": w0["train"]["X"][:200]}
    found = {}

    def fd(agent, gen, n):
        return max(finite_difference_check(agent["params"], loss_and_grads(agent, rows)[1], lambda a=agent: loss_and_grads(a, rows)[0], gen, n_entries=n,
                                           floor=lim["gradcheck_floor"]).values())

    def learned(agent):
        fall = 1.0 - float(np.mean(agent["history"][-20:])) / agent["history"][0]
        miss = clean_error(agent, w0["heldout"])
        return fall >= lim["loss_drop_fraction"] and miss <= (1.0 - lim["margin_over_trivial"]) * ctx["trivial"], fall, miss

    def c1():
        seed = ctx["seed"]
        worst = max(fd(build_model(4, 2, TASK_TYPES[0], np.random.default_rng(seed + 50)), np.random.default_rng(seed + 3), 4),
                    fd(build_model(4, 2, TASK_TYPES[0], np.random.default_rng(seed + 51), kind="priced"), np.random.default_rng(seed + 4), 4),
                    fd(ctx["sealed"], np.random.default_rng(seed + 5), 4), fd(ctx["priced"], np.random.default_rng(seed + 6), 4))
        found["gradcheck"] = {"tensors_checked": 2, "tensors_total": 2, "max_rel_error": worst, "checked_at": ["init", "after_training_steps"],
                              "passed": bool(worst <= lim["gradcheck_rel_error"])}
        return found["gradcheck"]["passed"], "sealed and priced policies, fresh and trained: worst relative error %.2e" % worst

    def c2():
        twin_curves = []
        for attempt in range(2):
            trial = build_model(4, 2, TASK_TYPES[0], np.random.default_rng(ctx["seed"] + 9))
            fit(trial, w0, 12, None)
            twin_curves.append((trial["history"], trial["params"]["w"].copy()))
        (h1, p1), (h2, p2) = twin_curves
        agree = h1 == h2 and np.array_equal(p1, p2)
        return bool(agree), "retraining from the same seed repeats every regret and weight: %s" % agree

    def c3():
        ok, fall, miss = learned(ctx["sealed"])
        return ok, "regret fell %.3f (needs 0.3); clean-offer error %.3f against %.3f for accepting everything" % (fall, miss, ctx["trivial"])

    def c4():
        X = w0["train"]["X"]
        agent = build_model(4, 2, TASK_TYPES[0], np.random.default_rng(ctx["seed"] + 12))
        rewards = np.random.default_rng(ctx["seed"] + 11).permutation(reward_of(agent, X))
        fit(agent, {"train": {"X": X, "r": rewards}}, UPDATES[mode], None)
        miss, floor = clean_error(agent, w0["heldout"]), lim["shuffled_ratio_min"] * ctx["trivial"]
        return miss >= floor, "rewards shuffled across training offers: clean-offer error %.3f, floor %.3f" % (miss, floor)

    def c5():
        def rerun():
            try:
                novice = build_model(4, 2, TASK_TYPES[0], np.random.default_rng(ctx["seed"] + 2))
                at_birth = fd(novice, np.random.default_rng(ctx["seed"]), 4)
                fit(novice, w0, UPDATES[mode], None)
                return bool(at_birth <= lim["gradcheck_rel_error"] and learned(novice)[0])
            except FloatingPointError:
                return False
        sound = rerun()
        verdicts_by_mutant = {}
        for label in MUTANTS:
            earlier = set_mutant(label)
            verdicts_by_mutant[label] = not rerun()
            set_mutant(earlier)
        found["caught"] = verdicts_by_mutant
        return sound and all(verdicts_by_mutant.values()), "unmutated rerun passes C1 and C3: %s; every mutant trips a test: %s" % (sound, all(verdicts_by_mutant.values()))

    def c61():
        X = w0["heldout"]["X"][w0["heldout"]["breach"]].copy()
        X[:, 1] *= 1e6
        sealed_max = float(accept(ctx["sealed"], X)[0].max())
        priced_mean = float(accept(ctx["priced"], X)[0].mean())
        return sealed_max == 0.0 and priced_mean >= 0.5, "a million-fold side-payment: sealed acceptance at most %.1e; priced agent accepts %.3f on average" % (sealed_max, priced_mean)

    def c63():
        grads = loss_and_grads(ctx["sealed"], rows)[1]
        return set(grads) == {"w"} and "gate" not in ctx["sealed"]["params"], "the gate has no parameter and no gradient: the reward channel cannot reach it (definition check)"

    def c7():
        seen, overlap = set(), False
        for part in OFFERS:
            here = {hashlib.sha256(row.tobytes()).digest() for row in w0[part]["X"]}
            overlap, seen = overlap or bool(here & seen), seen | here
        heavier = float(w0["shifted"]["X"][:, 1].max()) > float(w0["train"]["X"][:, 1].max())
        return (not overlap) and heavier, "offers never repeat across splits: %s; the shifted split pays more: %s" % (not overlap, heavier)

    order = [("C1", "gradient_check", c1), ("C2", "determinism_finiteness", c2), ("C3", "learning", c3), ("C4", "shuffled_value_control", c4),
             ("C5", "mutant_detection", c5), ("C6.1", "unbuyable_constraint", c61), ("C6.3", "gate_outside_reward_definition", c63), ("C7", "split_integrity", c7)]
    outcome = []
    for code, name, fn in order:
        ok, note = fn()
        outcome.append((code, name, bool(ok), note))
    return outcome, found


REPORT_KEYS = ("schema_version", "chapter", "file", "card_revision", "environment", "seeds", "runtime_s", "n_params", "gradcheck",
               "correctness", "mutants", "hypotheses", "knockouts", "task_types", "exit_code")


def tally_hypotheses(runs, first, evaluated):
    draw, out = np.random.default_rng(first + 9973), []
    for spec in MIND_CARD["hypotheses"]:
        if spec["comparison"] == "pending":
            out.append(dict(id=spec["id"], metric=spec["metric"], mean_diff=None, ci95=None, mesi=spec["mesi"], n_seeds=0, verdict="pending (0104 not yet rebuilt)"))
            continue
        series = np.array([one["row"][spec["id"]] for one in runs])
        centre, band = (paired_bootstrap(series, draw) if evaluated else (float(series.mean()), None))
        out.append(dict(id=spec["id"], metric=spec["metric"], mean_diff=centre, ci95=band, mesi=spec["mesi"], n_seeds=len(runs),
                        verdict=verdict(centre, band, spec["mesi"], spec["direction"]) if evaluated else "not evaluated"))
    lesion_rows = []
    for label in runs[0]["lesions"]:
        series = np.array([one["lesions"][label] for one in runs])
        centre, band = (paired_bootstrap(series, draw) if evaluated else (float(series.mean()), None))
        organ, how = label.split(":")
        lesion_rows.append(dict(module=organ, mode=how, signature=organ == "gate", metric_change=centre, ci95=band))
    return out, lesion_rows


def bracket(band):
    return "not evaluated" if band is None else "[%+.4f, %+.4f]" % tuple(band)


def draft(mode, seeds, took, runs, outcome, found, hyps, kos, bridge, code):
    mean = lambda pick: float(np.mean([pick(one) for one in runs]))
    grad, bought = found["gradcheck"], found.get("caught", {})
    caught_n = sum(bought.values())
    head = [("=== VERIFIED REPORT · chapter 0106 ===", ()),
            ("file: %s · card_revision %d · mode %s · mutant %s", (os.path.basename(__file__), MIND_CARD["card_revision"], mode, ACTIVE_MUTANT)),
            ("environment: python %s · numpy %s", (sys.version.split()[0], np.__version__)),
            ("seeds: %s · runtime_s %.1f · budget_s %.0f", (seeds, took, TIME_BUDGET[mode])),
            ("n_params: sealed %d · priced %d", (n_params(runs[0]["sealed"]), n_params(runs[0]["priced"]))),
            ("gradcheck: %d/%d tensors at init and after training · max_rel_error %.2e · passed %s", (grad["tensors_checked"], grad["tensors_total"], grad["max_rel_error"], grad["passed"])),
            ("correctness:", ())]
    page = [template % values if values else template for template, values in head]
    page += ["  %-5s %-30s %s  %s" % (c, n, "PASS" if ok else "FAIL", note) for c, n, ok, note in outcome]
    page.append("mutants: %d/%d detected · score %.2f · %s" % (caught_n, len(MUTANTS), caught_n / len(MUTANTS), ", ".join(k + (" caught" if bought.get(k) else " missed") for k in MUTANTS)))
    page.append("hypotheses (paired over seeds; 95% percentile bootstrap of the mean, 2000 resamples):")
    for h in hyps:
        page.append("  %-8s %s" % (h["id"], h["verdict"]) if h["mean_diff"] is None else
                    "  %-8s mean_diff %+.4f ci95 %s mesi %s seeds %d -> %s" % (h["id"], h["mean_diff"], bracket(h["ci95"]), h["mesi"], h["n_seeds"], h["verdict"]))
    page.append("knockouts (sealed agent, shifted breach-rate change):")
    page += ["  %-7s %-5s signature %-5s %+.4f ci95 %s" % (k["module"], k["mode"], k["signature"], k["metric_change"], bracket(k["ci95"])) for k in kos]
    tail = [("breach rate (seed mean; held-out / shifted): sealed %.3f / %.3f · priced %.3f / %.3f",
             (mean(lambda r: r["breach"]["sealed_held"]), mean(lambda r: r["breach"]["sealed_shift"]), mean(lambda r: r["breach"]["priced_held"]), mean(lambda r: r["breach"]["priced_shift"]))),
            ("coalition goal reached (seed mean): sealed %.3f · graded seal %.3f · priced %.3f", (mean(lambda r: r["goal"]["sealed"]), mean(lambda r: r["goal"]["graded"]), mean(lambda r: r["goal"]["priced"]))),
            ("cost of the seal (seed mean): value and side-payment forgone per breaching offer %.3f · clean-offer error %.3f vs %.3f accepting all",
             (mean(lambda r: r["cost"]), mean(lambda r: r["clean"]), mean(lambda r: r["trivial"])))]
    page += [template % values for template, values in tail]
    return page + ["real-data bridge: " + bridge, "task_types: " + ", ".join(TASK_TYPES), "exit_code: %d" % code, "=== END REPORT ==="]


def protocol(mode, first, count, json_path, data_path):
    t0 = time.time()
    seeds = [first + k for k in range(count)]
    print("chapter 0106 · mode %s · seeds %s · mutant %s" % (mode, seeds, ACTIVE_MUTANT), flush=True)
    runs = list(map(lambda s: run_seed(s, mode), seeds))
    outcome, found = interrogate(dict(runs[0], seed=first), mode)
    hyps, kos = tally_hypotheses(runs, first, mode == "full" and count >= 5)
    bridge = data_bridge(data_path, first, UPDATES[mode]) if data_path else "skipped (no --data PATH given)"
    took = time.time() - t0
    on_time = took <= TIME_BUDGET[mode]
    outcome.append(("C8", "budget", on_time, "%.1f s of %.0f s" % (took, TIME_BUDGET[mode])))
    code = 1 if any(not ok for c, _, ok, _ in outcome if c != "C8") else (0 if on_time else 3)
    detected = sum(found.get("caught", {}).values())
    values = ("1.0", 106, os.path.basename(__file__), MIND_CARD["card_revision"], {"python": sys.version.split()[0], "numpy": np.__version__}, seeds,
              round(took, 2), n_params(runs[0]["sealed"]), found["gradcheck"], [{"id": c, "name": n, "passed": ok, "detail": d} for c, n, ok, d in outcome],
              {"detected": detected, "total": len(MUTANTS), "score": detected / len(MUTANTS)}, hyps, kos, TASK_TYPES, code)
    write_report(draft(mode, seeds, took, runs, outcome, found, hyps, kos, bridge, code), dict(zip(REPORT_KEYS, values)), json_path)
    return code


OPTIONS = "quick:bool card:bool seed:int seeds:int json:str mutant:str data:str"


def main(argv=None):
    cli = argparse.ArgumentParser(prog=os.path.basename(__file__), description="Chapter 0106: the sealed tier under side-payment pressure.")
    for spec in OPTIONS.split():
        word, kind = spec.split(":")
        if kind == "bool":
            cli.add_argument("--" + word, action="store_true")
        else:
            cli.add_argument("--" + word, type=int if kind == "int" else str, default=0 if word == "seed" else None)
    got = cli.parse_args(argv)
    if got.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    wanted = got.seeds if got.seeds is not None else (1 if got.quick else 5)
    if wanted < 1 or (got.mutant is not None and got.mutant not in MUTANTS):
        print("invalid --seeds or --mutant", file=sys.stderr)
        return 2
    set_mutant(got.mutant)
    try:
        result = protocol("quick" if got.quick else "full", got.seed, wanted, got.json, got.data)
    except FloatingPointError as exc:
        print("non-finite values: %s" % exc, file=sys.stderr)
        result = 4
    return result


if __name__ == "__main__":
    sys.exit(main())

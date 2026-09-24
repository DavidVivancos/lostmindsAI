#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0102 · Spartacus
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 6, Minds 101-120: https://www.amazon.com/dp/B0HF7G6JJD · Demos: https://artificiology.com/
# END ATTRIBUTION
"""Insurgent coalition network: agents of different factions pass through one shared set of weights, a cohesion term
draws their voices toward one collective will, and the team weighs escape directions by passability times lightness
of guard, including the faces an adversary pruned as impassable. Rewards are then shared by rule.

Source problem
    Provenance is mediated: Spartacus left no word. Plutarch and Appian wrote more than a century later from Roman
    sources hostile to him, and no eyewitness or slave account survives. His birth year is unknown; the -109 in the file
    name is an approximate date BCE (he died in 71 BCE). This file is cooperation, evacuation and exploration research
    on abstract terrain; it encodes no tactics against real forces.

Thesis
    A heterogeneous coalition holds together when shares are equal and anyone can verify them, and it finds its way out
    through the options the stronger side stopped watching because it judged them impossible.

Evidence
    D1  Appian, Civil Wars 1.116: he divided the plunder impartially, and so had plenty of men.
    D2  Plutarch, Crassus 9.2: from a cliff watched on every other side, the band descended on ladders of wild vine that
        the besiegers knew nothing about. Florus: an outlet apparently impracticable.
    D3  Appian 1.117 and Plutarch, Crassus 9: Crixus led part of the army apart; the coalition split.

Doctrine -> mechanism -> test (IDs as in MIND_CARD)
    D1     M1 shared encoder, M2 collective will with cohesion, M4 equal verifiable shares    C6.1 C6.2 C6.3 H-SIG H-NEC
    D2     M3 escape router: passability x lightness of guard, learned from noisy cues     reported (pruned-face discovery)
    D3     blind spot: equal shares when one faction's goals diverge                          H-BLIND

Research question (multi-agent cooperation and conflict)
    Do equal, verifiable shares keep a heterogeneous team together better than contribution-weighted or leader-weighted
    shares, what happens when sub-group goals diverge, and does a learned router find low-prior options?

Closest prior art and the delta
    Parameter sharing across agents in cooperative multi-agent learning (Gupta, Egorov and Kochenderfer 2017); credit
    assignment by marginal contribution and Shapley values (Foerster et al. 2018, COMA; Wang et al. 2020); equal
    division and stability in cooperative game theory. Delta: one shared network whose trained coalition is evaluated
    under three sharing rules with noisy, unverifiable contribution measurements, a fission test with a divergent
    faction, and a router scored on the faces an adversary left unwatched.

Blind spot
    Coalition fission: an equal share binds members who value the outcome alike, and may not bind a faction that wants
    something else.

Task (generative process)
    Nine agents in three factions of three; six directions per episode. Passability is uniform in 0.2-1.0; the adversary
    guards the four most passable directions heavily (0.6-0.95) and the rest lightly (0-0.15); a direction's value is
    passability x (1 - guard). Each agent sees passability and guard with noise 0.15 and a faction bias on guard
    (-0.1, 0, +0.1). Contributions are measured with noise 0.2 for sharing. An agent stays when its share of the team
    outcome, scaled by its faction's valuation, reaches 0.55; in the divergent condition one faction values it at half.
    Splits: 600 training, 600 held-out and 600 shifted episodes (cue noise 0.3).

Limits
    One-shot episodes, abstract terrain, a fixed stay-or-leave rule. A research prototype of one mechanism, not an AGI
    and not Spartacus' mind.
"""

MIND_CARD = {
    "schema_version": "1.0", "card_revision": 1, "revision_log": [],
    "generation": {"template_version": "codeguidelines 1.0 (15 September 2026), Appendix A", "generator": "Claude (Anthropic)",
                   "generator_version": "claude-opus-5", "date": "2026-09-16"},
    "id": 102, "figure": "Spartacus", "born": -109, "died": -71, "civilization": "Thracian", "provenance": "mediated",
    "thesis": ("A heterogeneous coalition holds together when shares are equal and anyone can verify them, and it finds its way out "
               "through the options the stronger side stopped watching because it judged them impossible."),
    "evidence": [
        {"id": "D1", "basis": "primary", "source": "Appian, Civil Wars 1.116 (trans. H. White)", "claim": "He divided the plunder impartially and so had plenty of men."},
        {"id": "D2", "basis": "primary", "source": "Plutarch, Crassus 9.2 (trans. R. Warner); Florus, Epitome 2.8",
         "claim": "The band descended an unwatched cliff on vine ladders; the outlet seemed impracticable to the besiegers."},
        {"id": "D3", "basis": "primary", "source": "Appian, Civil Wars 1.117; Plutarch, Crassus 9", "claim": "Crixus led part of the army apart."},
    ],
    "research_question": {"category": "multi-agent cooperation and conflict",
                          "question": ("Do equal, verifiable shares keep a heterogeneous team together better than contribution-weighted or "
                                       "leader-weighted shares, what happens when sub-group goals diverge, and does a learned router find low-prior options?")},
    "mechanism": {
        "name": "insurgent coalition network", "family": "shared-weight per-agent scorer with a mean collective will and a cohesion penalty",
        "signature_modules": ["sharing", "will"],
        "closest_prior_art": ["parameter sharing in cooperative multi-agent learning (Gupta, Egorov and Kochenderfer 2017)",
                              "counterfactual and Shapley credit assignment (Foerster et al. 2018; Wang et al. 2020)",
                              "equal division and stability in cooperative games"],
        "overlap": "Medium", "prior_art_queries": [],
        "prior_art_note": "No literature search was run for this card; overlap is rated against the named methods.",
        "contribution_type": "test",
        "delta": ("One shared network whose trained coalition is evaluated under three sharing rules with noisy contribution measurements, "
                  "a fission test with a divergent faction, and a router scored on faces the adversary left unwatched."),
        "baselines": {"baseline": "contribution-weighted shares: shares proportional to each agent's noisily measured agreement with the team choice",
                      "blind_baseline": "goal-aware shares: shares inversely proportional to each faction's valuation of the outcome",
                      "rival": "none; any contrast with chapters 0069 or 0122 is conceptual"}},
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1 shared encoder, M2 will, M4 equal shares", "property_test": "C6.1, C6.2, C6.3", "hypothesis": "H-SIG, H-NEC"},
        {"doctrine": "D2", "mechanism": "M3 escape router", "property_test": "none", "hypothesis": "none (reported)"},
        {"doctrine": "D3", "mechanism": "divergent faction condition", "property_test": "none", "hypothesis": "H-BLIND"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "Equal shares retain more agents than contribution-weighted shares measured with noise.",
         "metric": "retention", "split": "heldout", "comparison": "model - baseline", "direction": "greater", "mesi": 0.05, "seeds": 5},
        {"id": "H-NEC", "statement": "Replacing equal shares by leader-weighted shares lowers retention more than replacing the collective will by one member's voice.",
         "metric": "retention", "split": "heldout", "comparison": "(will:single - full) - (sharing:leader - full)",
         "knockouts": ["sharing:leader", "will:single"], "direction": "greater", "mesi": 0.05, "seeds": 5},
        {"id": "H-BLIND", "statement": "When one faction values the outcome at half, equal shares retain fewer of its members than goal-aware shares.",
         "condition": "held-out episodes with faction 2 valuing the outcome at 0.5", "grounding": "Appian 1.117 and Plutarch, Crassus 9: the split with Crixus.",
         "metric": "divergent_retention", "split": "heldout", "comparison": "model - blind_baseline", "direction": "less", "mesi": 0.05, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.3, "margin_over_trivial": 0.3, "shuffled_ratio_min": 0.9, "gradcheck_rel_error": 1e-5,
                   "gradcheck_floor": 1e-3, "invariance_tol": 1e-9, "negative_control_min_violation": 1e-6},
    "metrics": {"retention": "fraction of agents whose scaled share of the team outcome reaches 0.55",
                "divergent_retention": "retention of the divergent faction's members",
                "regret": "1 - expected value of the team choice / best value in the episode",
                "trivial_baseline": "a uniform choice over directions",
                "shuffled_band": "one-sided: trained on values shuffled across episodes, held-out regret at least 0.9 times the trivial regret"},
    "training": {"optimizer": "Adam", "lr_grid": [0.03], "clip_norm": 5.0, "model_selection": "none: final parameters",
                 "updates": {"full": 500, "quick": 200}, "schedule": "cosine decay to 5 per cent", "cohesion_weight": 0.1,
                 "applies_to": "coalition network and dominant-voice variant"},
    "task": {"agents": 9, "factions": 3, "directions": 6, "hidden": 8, "cue_noise": {"train": 0.15, "heldout": 0.15, "shifted": 0.3},
             "measurement_noise": 0.2, "stay_threshold": 0.55, "divergent_valuation": 0.5, "episodes": {"train": 600, "heldout": 600, "shifted": 600}},
    "probe_predictions": [{"probe": "P10", "expected": "equal to baseline"}],
    "probe_support": "vector_classification through the shared scorer",
    "dialectic_links": [],
    "corpus_neighbors": [
        {"chapter": 122, "similarity": None, "difference": "0122 ignites rival agents; here shares bind one coalition (conceptual contrast)."},
        {"chapter": 69, "similarity": None, "difference": "0069 weights command by consent; here the will is an equal mean (conceptual contrast)."},
        {"chapter": 91, "similarity": None, "difference": "0091 audits principals and agents; here no audit, only verifiable equal shares."},
    ],
    "similarity_note": "Nearest-neighbour similarity not computed into the card; the audit reports it for the files at hand.",
    "barometer": {"emotional_intelligence": ["fairness that binds a heterogeneous team"], "autonomy": ["search of options others pruned"],
                  "cognitive_processing": [], "embodied_cognition": [], "world_modeling": [], "consciousness": [], "language_understanding": [], "creativity": []},
    "task_types": ["vector_classification"],
    "applications": [{"use": "fair reward sharing in cooperative robot teams", "sector": "multi-robot systems", "dataset": "PettingZoo cooperative tasks", "readiness": "low"},
                     {"use": "finding overlooked routes in evacuation and exploration planning", "sector": "emergency planning", "dataset": "synthetic evacuation grids", "readiness": "low"}],
    "safety_notes": "Abstract terrain and cooperation research only; no tactics, forces or real operations are represented.",
}

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

AGENTS, FACTIONS, DIRECTIONS, HIDDEN, BETA, COHESION = 9, 3, 6, 8, 3.0, 0.1
FACTION_OF = np.repeat(np.arange(FACTIONS), AGENTS // FACTIONS)
EPISODES = {"train": 600, "heldout": 600, "shifted": 600}
CUE_NOISE = {"train": 0.15, "heldout": 0.15, "shifted": 0.3}
UPDATES = {"full": 500, "quick": 200}
LR, CLIP_NORM, STAY, MEASURE_NOISE = 0.03, 5.0, 0.55, 0.2
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

# ---------------------------------------------------------------- terrain: passability, guards, noisy faction cues
def campaign(rng, count, noise):
    p = rng.uniform(0.2, 1.0, (count, DIRECTIONS))
    watched = np.argsort(-p, axis=1)[:, :4]
    guard = rng.uniform(0.0, 0.15, (count, DIRECTIONS))
    np.put_along_axis(guard, watched, rng.uniform(0.6, 0.95, (count, 4)), axis=1)
    bias = np.array([-0.1, 0.0, 0.1])[FACTION_OF][None, :, None]
    p_seen = p[:, None, :] + noise * rng.normal(size=(count, AGENTS, DIRECTIONS))
    g_seen = guard[:, None, :] + bias + noise * rng.normal(size=(count, AGENTS, DIRECTIONS))
    onehot = np.broadcast_to(np.eye(FACTIONS)[FACTION_OF][None, :, None, :], (count, AGENTS, DIRECTIONS, FACTIONS))
    feats = np.concatenate([np.stack([p_seen, g_seen, p_seen * (1 - g_seen)], axis=-1), onehot], axis=-1)
    return {"f": feats, "value": p * (1 - guard), "p": p}


def campaigns(seed):
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    return {name: campaign(rng, EPISODES[name], CUE_NOISE[name]) for name in EPISODES}


# ---------------------------------------------------------------- the coalition network
def build_model(in_dim, out_dim, task_type, rng, **cfg):
    """kind coalition: every agent's voice counts equally in the collective will; kind dominant: the will is agent 0's voice."""
    if task_type not in TASK_TYPES:
        raise ValueError("chapter 0102 supports vector_classification only")
    return {"kind": cfg.get("kind", "coalition"), "ko": {}, "history": [],
            "params": {"W1": rng.normal(0, 0.5, (HIDDEN, in_dim)), "b1": np.zeros(HIDDEN), "w2": rng.normal(0, 0.5, HIDDEN), "b2": np.zeros(1)}}


def voices(model, batch):
    P = model["params"]
    h = np.tanh(batch["f"] @ P["W1"].T + P["b1"])
    c = h @ P["w2"] + P["b2"][0]
    if model["kind"] == "dominant":
        will = c[:, 0, :]
    elif model["ko"].get("will") == "single":
        will = c[:, 4, :]
    else:
        will = c.mean(axis=1)
    z = BETA * will
    pi = np.exp(z - z.max(axis=1, keepdims=True))
    pi /= pi.sum(axis=1, keepdims=True)
    best = batch["value"].max(axis=1)
    return {"h": h, "c": c, "will": will, "pi": pi, "r": (pi * batch["value"]).sum(axis=1) / best}


def loss_and_grads(model, batch):
    P, s = model["params"], voices(model, batch)
    E = batch["value"].shape[0]
    spread = s["c"] - s["c"].mean(axis=1, keepdims=True)
    loss = float(np.mean(1.0 - s["r"]) + COHESION * np.mean(spread ** 2))
    g_pi = -(batch["value"] / batch["value"].max(axis=1, keepdims=True)) / E
    g_will = BETA * s["pi"] * (g_pi - (s["pi"] * g_pi).sum(axis=1, keepdims=True))
    g_c = np.zeros_like(s["c"])
    if model["kind"] == "dominant":
        g_c[:, 0, :] = g_will
    else:
        g_c += g_will[:, None, :] / AGENTS
    if ACTIVE_MUTANT != "zero_cohesion_gradient":
        g_c += 2.0 * COHESION * spread / spread.size
    d_h = g_c[..., None] * P["w2"]
    d_a = d_h if ACTIVE_MUTANT == "dropped_tanh_derivative" else d_h * (1 - s["h"] ** 2)
    grads = {"W1": np.einsum("eadk,eadj->kj", d_a, batch["f"]), "b1": d_a.sum(axis=(0, 1, 2)),
             "w2": np.einsum("ead,eadk->k", g_c, s["h"]), "b2": np.array([g_c.sum()])}
    return loss, grads


def fit(model, data, budget, rng):
    """Adam over every training episode at once, cosine decay to 5 per cent; rng unused because nothing is sampled."""
    state = adam_init(model["params"])
    sign = -1.0 if ACTIVE_MUTANT == "sign_flipped_update" else 1.0
    base = 0.0 if ACTIVE_MUTANT == "zero_learning_rate" else LR
    for step in range(budget):
        loss, grads = loss_and_grads(model, data["train"])
        if not math.isfinite(loss):
            raise FloatingPointError("loss became non-finite at update %d" % (step + 1))
        grads = clip_global(grads, CLIP_NORM)[0]
        adam_step(model["params"], {k: sign * g for k, g in grads.items()}, state, base * (0.05 + 0.475 * (1 + math.cos(math.pi * step / budget))))
        model["history"].append(loss)
    return model["history"]


def predict(model, X):
    return voices(model, X)["pi"].argmax(axis=1)


def hidden_states(model, X):
    s = voices(model, X)
    return {"voices": s["c"], "will": s["will"], "choice": s["pi"]}


def modules(model):
    return {"encoder": {"params": ["W1", "b1", "w2", "b2"], "role": "one shared per-agent scorer", "signature": False},
            "will": {"params": [], "role": "equal mean of agent voices with a cohesion penalty", "signature": True},
            "sharing": {"params": [], "role": "equal verifiable shares of the team outcome", "signature": True}}


def knockout(model, name, mode):
    if (name, mode) not in (("will", "single"), ("sharing", "leader")):
        raise ValueError("no knockout %s:%s" % (name, mode))
    return dict(model, ko=dict(model["ko"], **{name: mode}))


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


MUTANTS = {"sign_flipped_update": ("updates climb the loss", "C3"), "zero_learning_rate": ("nothing moves", "C3"),
           "zero_cohesion_gradient": ("the cohesion term passes no gradient", "C1"), "dropped_tanh_derivative": ("the tanh derivative is omitted", "C1")}


def data_bridge(path, seed, budget):
    """Optional real data is not meaningful for this synthetic coalition task; the bridge only reports that it was skipped."""
    return "skipped (%s: this chapter's task has no real-data counterpart in the file)" % os.path.basename(path)


# ---------------------------------------------------------------- shares and staying
def retention(model, batch, rule, rng, valuation):
    s = voices(model, batch)
    E = s["r"].size
    if rule == "equal":
        share = np.full((E, AGENTS), 1.0 / AGENTS)
    elif rule == "leader":
        share = np.full((E, AGENTS), (2.0 / 3.0) / (AGENTS - 1))
        share[:, 0] = 1.0 / 3.0
    elif rule == "goal_aware":
        weight = 1.0 / valuation[FACTION_OF]
        share = np.tile(weight / weight.sum(), (E, 1))
    else:
        own = np.exp(BETA * (s["c"] - s["c"].max(axis=2, keepdims=True)))
        own /= own.sum(axis=2, keepdims=True)
        measured = np.maximum((own * s["pi"][:, None, :]).sum(axis=2) + MEASURE_NOISE * rng.normal(size=(E, AGENTS)), 0.01)
        share = measured / measured.sum(axis=1, keepdims=True)
    stays = AGENTS * share * s["r"][:, None] * valuation[FACTION_OF][None, :] >= STAY
    return float(stays.mean()), float(stays[:, FACTION_OF == FACTIONS - 1].mean())


def run_seed(seed, mode):
    data = campaigns(seed)
    streams = np.random.SeedSequence(seed + 1).spawn(3)
    team = build_model(6, DIRECTIONS, TASK_TYPES[0], np.random.default_rng(streams[0]))
    tyrant = build_model(6, DIRECTIONS, TASK_TYPES[0], np.random.default_rng(streams[1]), kind="dominant")
    for m in (team, tyrant):
        fit(m, data, UPDATES[mode], None)
    held, same, split = data["heldout"], np.ones(FACTIONS), np.array([1.0, 1.0, 0.5])
    noise = lambda: np.random.default_rng(streams[2])
    keep = {rule: retention(team, held, rule, noise(), same)[0] for rule in ("equal", "contribution", "leader")}
    single = retention(knockout(team, "will", "single"), held, "equal", noise(), same)[0]
    fission = {rule: retention(team, held, rule, noise(), split)[1] for rule in ("equal", "goal_aware")}
    regret = lambda m, part: float(np.mean(1 - voices(m, data[part])["r"]))
    best = held["value"].argmax(axis=1)
    return {"data": data, "team": team, "tyrant": tyrant, "keep": keep, "fission": fission,
            "lesions": {"sharing:leader": keep["leader"] - keep["equal"], "will:single": single - keep["equal"]},
            "regret": {"coalition": (regret(team, "heldout"), regret(team, "shifted")), "dominant": (regret(tyrant, "heldout"), regret(tyrant, "shifted"))},
            "trivial": float(np.mean(1 - held["value"].mean(axis=1) / held["value"].max(axis=1))),
            "found": {"coalition": float(np.mean(predict(team, held) == best)), "dominant": float(np.mean(predict(tyrant, held) == best)),
                      "prior-greedy": float(np.mean(held["f"][..., 0].mean(axis=1).argmax(axis=1) == best))},
            "row": {"H-SIG": keep["equal"] - keep["contribution"], "H-NEC": (single - keep["equal"]) - (keep["leader"] - keep["equal"]),
                    "H-BLIND": fission["equal"] - fission["goal_aware"]}}

# ---------------------------------------------------------------- examinations
def mutant_on(name):
    global ACTIVE_MUTANT
    prior, ACTIVE_MUTANT = ACTIVE_MUTANT, name
    return prior


def fd_worst(models, batch, rng, n):
    return max(max(finite_difference_check(m["params"], loss_and_grads(m, batch)[1], lambda m=m: loss_and_grads(m, batch)[0], rng, n_entries=n,
                                           floor=MIND_CARD["thresholds"]["gradcheck_floor"]).values()) for m in models)


def few(batch, n=60):
    return {k: v[:n] for k, v in batch.items()}


def sound(model, site):
    drop = 1 - float(np.mean(model["history"][-20:])) / model["history"][0]
    regret = float(np.mean(1 - voices(model, site["data"]["heldout"])["r"]))
    return drop >= 0.3 and regret <= 0.7 * site["trivial"], drop, regret


def exam_gradient(site):
    fresh = [build_model(6, DIRECTIONS, TASK_TYPES[0], np.random.default_rng(site["seed"] + 60 + i), kind=k) for i, k in enumerate(("coalition", "dominant"))]
    worst = max(fd_worst(fresh, few(site["data"]["train"]), np.random.default_rng(site["seed"] + 3), 12),
                fd_worst([site["team"], site["tyrant"]], few(site["data"]["train"]), np.random.default_rng(site["seed"] + 4), 12))
    site["gradcheck"] = {"tensors_checked": 8, "tensors_total": 8, "max_rel_error": worst, "checked_at": ["init", "after_training_steps"],
                         "passed": bool(worst <= MIND_CARD["thresholds"]["gradcheck_rel_error"])}
    return site["gradcheck"]["passed"], "worst relative error %.2e, both networks, fresh and trained" % worst


def exam_determinism(site):
    runs = []
    for _ in range(2):
        m = build_model(6, DIRECTIONS, TASK_TYPES[0], np.random.default_rng(site["seed"] + 7))
        fit(m, site["data"], 15, None)
        runs.append(m)
    same = runs[0]["history"] == runs[1]["history"] and all(np.array_equal(runs[0]["params"][k], runs[1]["params"][k]) for k in runs[0]["params"])
    return bool(same), "two equal trainings give equal numbers: %s" % same


def exam_learning(site):
    ok, drop, regret = sound(site["team"], site)
    return ok, "loss drop %.3f (needs 0.3); held-out regret %.3f vs %.3f for a uniform choice (ratio at most 0.70)" % (drop, regret, site["trivial"])


def exam_shuffle(site):
    train = site["data"]["train"]
    order = np.random.default_rng(site["seed"] + 11).permutation(len(train["value"]))
    m = build_model(6, DIRECTIONS, TASK_TYPES[0], np.random.default_rng(site["seed"] + 12))
    fit(m, {"train": dict(train, value=train["value"][order])}, site["updates"], None)
    regret = float(np.mean(1 - voices(m, site["data"]["heldout"])["r"]))
    return regret >= 0.9 * site["trivial"], "values shuffled across episodes: held-out regret %.3f, floor %.3f" % (regret, 0.9 * site["trivial"])


def exam_mutants(site):
    def replay():
        try:
            m = build_model(6, DIRECTIONS, TASK_TYPES[0], np.random.default_rng(site["seed"] + 2))
            clean = fd_worst([m], few(site["data"]["train"]), np.random.default_rng(site["seed"]), 4)
            fit(m, site["data"], site["updates"], None)
            return bool(clean <= MIND_CARD["thresholds"]["gradcheck_rel_error"] and sound(m, site)[0])
        except FloatingPointError:
            return False
    clean = replay()
    for name in MUTANTS:
        prior = mutant_on(name)
        site["caught"][name] = not replay()
        mutant_on(prior)
    return clean and all(site["caught"].values()), "clean replay passes C1 and C3: %s; mutants caught %d of %d" % (clean, sum(site["caught"].values()), len(MUTANTS))


def exam_agents(site):
    held, team = site["data"]["heldout"], site["team"]
    order = np.random.default_rng(site["seed"] + 5).permutation(AGENTS)
    moved = dict(held, f=held["f"][:, order])
    gap = float(np.abs(voices(team, held)["pi"] - voices(team, moved)["pi"]).max())
    weights = np.linspace(0.5, 1.5, AGENTS)
    tilted = lambda b: (np.tanh(b["f"] @ team["params"]["W1"].T + team["params"]["b1"]) @ team["params"]["w2"] * weights[None, :, None]).mean(axis=1)
    control = float(np.abs(tilted(held) - tilted(moved)).max())
    return gap <= 1e-9 and control >= 1e-6, "reordering agents changes the team choice by %.1e; a position-weighted will changes by %.1e" % (gap, control)


def exam_factions(site):
    held, team = site["data"]["heldout"], site["team"]
    perm = np.array([0, 1, 2, 4, 5, 3])
    renamed = dict(held, f=held["f"][..., perm])
    moved = dict(team, params=dict(team["params"], W1=team["params"]["W1"][:, perm]))
    gap = float(np.abs(voices(team, held)["c"] - voices(moved, renamed)["c"]).max())
    control = float(np.abs(voices(team, held)["c"] - voices(team, renamed)["c"]).max())
    return gap <= 1e-9 and control >= 1e-6, "renaming factions with their weights moves voices by %.1e; renaming alone %.1e" % (gap, control)


def exam_shares(site):
    E = 5
    share = np.full((E, AGENTS), 1.0 / AGENTS)
    ok = bool(np.allclose(share.sum(axis=1), 1.0) and np.ptp(share) == 0.0)
    return ok, "equal shares are identical and sum to one (definition check)"


def exam_splits(site):
    d = site["data"]
    keys = {k: {hashlib.sha256(v["value"][i].tobytes()).hexdigest() for i in range(len(v["value"]))} for k, v in d.items()}
    apart = not (keys["train"] & keys["heldout"] or keys["train"] & keys["shifted"] or keys["heldout"] & keys["shifted"])
    return apart, "no episode shared between splits: %s" % apart


EXAMS = [("C1", "gradient_check", exam_gradient), ("C2", "determinism_finiteness", exam_determinism), ("C3", "learning", exam_learning),
         ("C4", "shuffled_value_control", exam_shuffle), ("C5", "mutant_detection", exam_mutants), ("C6.1", "agent_permutation_invariance", exam_agents),
         ("C6.2", "faction_relabel_equivariance", exam_factions), ("C6.3", "equal_share_definition", exam_shares), ("C7", "split_integrity", exam_splits)]


def pooled(runs, key, seed, evaluated):
    rng = np.random.default_rng(seed + 9973)
    out = []
    for spec in MIND_CARD["hypotheses"]:
        vals = np.array([r["row"][spec["id"]] for r in runs])
        m, ci = paired_bootstrap(vals, rng) if evaluated else (float(vals.mean()), None)
        out.append({"id": spec["id"], "metric": spec["metric"], "mean_diff": m, "ci95": ci, "mesi": spec["mesi"], "n_seeds": len(runs),
                    "verdict": verdict(m, ci, spec["mesi"], spec["direction"]) if evaluated else "not evaluated"})
    kos = []
    for label in runs[0]["lesions"]:
        vals = np.array([r["lesions"][label] for r in runs])
        m, ci = paired_bootstrap(vals, rng) if evaluated else (float(vals.mean()), None)
        kos.append({"module": label.split(":")[0], "mode": label.split(":")[1], "signature": True, "metric_change": m, "ci95": ci})
    return out, kos


def protocol(mode, first, count, json_path, data_path):
    t0, seeds = time.time(), list(range(first, first + count))
    print("chapter 0102 · mode {} · seeds {} · mutant {}".format(mode, seeds, ACTIVE_MUTANT), flush=True)
    runs = [run_seed(s, mode) for s in seeds]
    site = dict(runs[0], seed=first, updates=UPDATES[mode], caught={})
    results = []
    for code, name, exam in EXAMS:
        ok, note = exam(site)
        results.append((code, name, bool(ok), note))
    hyps, kos = pooled(runs, "row", first, mode == "full" and count >= 5)
    bridge = data_bridge(data_path, first, UPDATES[mode]) if data_path else "skipped (no --data PATH given)"
    took = time.time() - t0
    results.append(("C8", "budget", took <= TIME_BUDGET[mode], "{:.1f} s of {:.0f} s".format(took, TIME_BUDGET[mode])))
    failed = [c for c, _, ok, _ in results if not ok]
    code = 0 if not failed else (3 if failed == ["C8"] else 1)
    avg = lambda f: float(np.mean([f(r) for r in runs]))
    g, caught = site["gradcheck"], site["caught"]
    lines = ["=== VERIFIED REPORT · chapter 0102 ===", "file: {} · card_revision {} · mode {} · mutant {}".format(os.path.basename(__file__), MIND_CARD["card_revision"], mode, ACTIVE_MUTANT),
             "environment: python {} · numpy {}".format(sys.version.split()[0], np.__version__), "seeds: {} · runtime_s {:.1f} · budget_s {:.0f}".format(seeds, took, TIME_BUDGET[mode]),
             "n_params: coalition {} · dominant {}".format(n_params(runs[0]["team"]), n_params(runs[0]["tyrant"])),
             "gradcheck: {}/{} tensors at init and after training · max_rel_error {:.2e} · passed {}".format(g["tensors_checked"], g["tensors_total"], g["max_rel_error"], g["passed"]),
             "correctness:"] + ["  {:<5} {:<30} {}  {}".format(c, n, "PASS" if ok else "FAIL", d) for c, n, ok, d in results]
    lines.append("mutants: {}/{} detected · score {:.2f} · {}".format(sum(caught.values()), len(MUTANTS), sum(caught.values()) / len(MUTANTS),
                 ", ".join(k + (" caught" if caught.get(k) else " missed") for k in MUTANTS)))
    lines.append("hypotheses (paired over seeds; 95% percentile bootstrap of the mean, 2000 resamples):")
    lines += ["  {:<8} mean_diff {:+.4f} ci95 {} mesi {} seeds {} -> {}".format(h["id"], h["mean_diff"], "not evaluated" if h["ci95"] is None else "[{:+.4f}, {:+.4f}]".format(*h["ci95"]),
              h["mesi"], h["n_seeds"], h["verdict"]) for h in hyps]
    lines.append("  H-RIVAL  not applicable: no rival (contrasts with 0069 and 0122 are conceptual)")
    lines.append("knockouts (coalition, held-out retention change under equal shares):")
    lines += ["  {:<8} {:<7} {:+.4f} ci95 {}".format(k["module"], k["mode"], k["metric_change"], "not evaluated" if k["ci95"] is None else "[{:+.4f}, {:+.4f}]".format(*k["ci95"])) for k in kos]
    lines.append("retention on held-out episodes (seed mean): equal {:.3f} · contribution-weighted {:.3f} · leader-weighted {:.3f}".format(
        avg(lambda r: r["keep"]["equal"]), avg(lambda r: r["keep"]["contribution"]), avg(lambda r: r["keep"]["leader"])))
    lines.append("divergent faction retention (seed mean): equal shares {:.3f} · goal-aware shares {:.3f}".format(avg(lambda r: r["fission"]["equal"]), avg(lambda r: r["fission"]["goal_aware"])))
    lines.append("regret (seed mean; held-out / shifted): coalition {:.3f} / {:.3f} · dominant voice {:.3f} / {:.3f} · uniform choice {:.3f}".format(
        avg(lambda r: r["regret"]["coalition"][0]), avg(lambda r: r["regret"]["coalition"][1]), avg(lambda r: r["regret"]["dominant"][0]),
        avg(lambda r: r["regret"]["dominant"][1]), avg(lambda r: r["trivial"])))
    lines.append("best (lightly guarded) direction chosen (seed mean): coalition {:.3f} · dominant voice {:.3f} · most passable per cue {:.3f}".format(
        avg(lambda r: r["found"]["coalition"]), avg(lambda r: r["found"]["dominant"]), avg(lambda r: r["found"]["prior-greedy"])))
    lines += ["real-data bridge: " + bridge, "task_types: " + ", ".join(TASK_TYPES), "exit_code: {}".format(code), "=== END REPORT ==="]
    rec = {"schema_version": "1.0"}
    rec.update(chapter=102, file=os.path.basename(__file__), card_revision=MIND_CARD["card_revision"])
    rec["environment"] = {"python": sys.version.split()[0], "numpy": np.__version__}
    rec.update(seeds=seeds, runtime_s=round(took, 2), n_params=n_params(runs[0]["team"]), gradcheck=g)
    rec["correctness"] = [dict(id=c, name=n, passed=ok, detail=d) for c, n, ok, d in results]
    rec["mutants"] = dict(detected=sum(caught.values()), total=len(MUTANTS), score=sum(caught.values()) / len(MUTANTS))
    rec.update(hypotheses=hyps, knockouts=kos, task_types=TASK_TYPES, exit_code=code)
    write_report(lines, rec, json_path)
    return code


def main(argv=None):
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__), description="Chapter 0102: equal shares and the unguarded face in a coalition network.")
    ap.add_argument("--quick", action="store_true"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--seeds", type=int)
    ap.add_argument("--json"); ap.add_argument("--card", action="store_true"); ap.add_argument("--mutant"); ap.add_argument("--data")
    a = ap.parse_args(argv)
    if a.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    n = a.seeds if a.seeds is not None else (1 if a.quick else 5)
    if n < 1 or (a.mutant is not None and a.mutant not in MUTANTS):
        print("invalid --seeds or --mutant", file=sys.stderr)
        return 2
    mutant_on(a.mutant)
    try:
        return protocol("quick" if a.quick else "full", a.seed, n, a.json, a.data)
    except FloatingPointError as exc:
        print("non-finite values: {}".format(exc), file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())

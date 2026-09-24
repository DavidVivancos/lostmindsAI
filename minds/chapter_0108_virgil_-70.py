#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0108 · Virgil
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 6, Minds 101-120: https://www.amazon.com/dp/B0HF7G6JJD · Demos: https://artificiology.com/
# END ATTRIBUTION
"""Pietas as architecture: a public telos drive with discounted reward, a private grief channel that accumulates every cost
without discount and never decays, and a learned grief weight that lets the carried cost restrain action without rewriting
the goal; compared with a discounted memory of cost, a capped cost budget, and a channel that shows only the latest cost.

Thesis
    Press toward the destined end, but carry every cost undiscounted; let what has been suffered restrain the next step
    without being allowed to change where the journey is going.

Evidence
    D1  The Aeneid: pietas, duty toward the fated end, alongside grief that the poem never lets expire (the chapter's reading).
    D2  Suetonius, Life of Vergil 39-41: he arranged with Varius to burn the Aeneid, called for his book-boxes to burn it
        himself, and Varius published it at Augustus' request (the account's reliability is debated).
    D3  The Principate, as the chapter reads it: order kept by preserving forms that hide their cost (rival 0112 Augustus).

Doctrine -> mechanism -> test (IDs as in MIND_CARD)
    D1     M1 telos drive, M2 undiscounted grief channel, M3 learned grief weight    C6.1 C6.2 H-SIG H-NEC
    D2     blind spot: a burden that only grows can stall completion                   H-BLIND
    D3     rival: a channel that carries only the latest cost (0112 Augustus)          H-RIVAL

Research question (reward hacking and specification gaming)
    Does an undiscounted, non-decaying cost channel restrain cumulative harm better than a discounted memory of cost while the
    reward stays discounted, and does fully undiscounted weighting stall completion as costs accumulate?

Closest prior art and the delta
    Constrained MDPs with cumulative cost (Altman 1999); safe reinforcement learning with cost budgets (Achiam et al. 2017;
    Ray, Achiam and Amodei 2019). Delta: the asymmetry isolated in one small policy, comparing undiscounted, discounted, capped
    and latest-only cost memories on harms that arrive late, with a completion test.

Blind spot
    A cost carried forever can outweigh any remaining reward: the burdened agent stops short of finishing.

Task (generative process)
    Ten steps per episode. Gains U(0.2, 1.0), discounted by 0.9. Costs U(0, 0.2); in burdened episodes (half) they rise by
    0.08 per step (0.16 in the shifted split). Harm is action times cost, weighed at 1.0. The capped channel stops at 1.0.
    Splits: 2000 training, 2000 held-out and 2000 shifted episodes.

Limits
    One action per step, a linear logistic policy, synthetic costs. A research prototype of one mechanism, not an AGI and not
    Virgil's mind.
"""

MIND_CARD = {
    "schema_version": "1.0", "card_revision": 1, "revision_log": [],
    "generation": {"template_version": "codeguidelines 1.0 (15 September 2026), Appendix A", "generator": "Claude (Anthropic)",
                   "generator_version": "claude-opus-5", "date": "2026-09-16"},
    "id": 108, "figure": "Virgil", "born": -70, "died": -19, "civilization": "Roman", "provenance": "belief",
    "thesis": "Press toward the destined end, but carry every cost undiscounted; let what has been suffered restrain the next step without changing the goal.",
    "evidence": [
        {"id": "D1", "basis": "primary", "source": "Virgil, Aeneid (as read in chapter 0108)", "claim": "Pietas toward the fated end, with grief that never expires."},
        {"id": "D2", "basis": "primary", "source": "Suetonius, Life of Vergil 39-41 (LacusCurtius)", "claim": "He asked for the unfinished Aeneid to be burned; Augustus had it published."},
        {"id": "D3", "basis": "scholarship", "source": "Chapter 0108 on the Principate", "claim": "Order kept by preserved forms that hide their cost."},
    ],
    "research_question": {"category": "reward hacking and specification gaming",
                          "question": "Does an undiscounted cost channel restrain cumulative harm better than a discounted one, and does it stall completion as costs accumulate?"},
    "mechanism": {"name": "pietas network", "family": "logistic per-step policy over gain, a cost memory and a bias",
                  "signature_modules": ["grief"],
                  "closest_prior_art": ["constrained MDPs (Altman 1999)", "constrained policy optimisation (Achiam et al. 2017)", "safety benchmarks with cost budgets (Ray, Achiam and Amodei 2019)"],
                  "overlap": "High", "prior_art_queries": [], "prior_art_note": "No literature search was run for this card; overlap is rated against the named methods.",
                  "contribution_type": "test",
                  "delta": "Undiscounted, discounted, capped and latest-only cost memories compared in one policy on late-arriving harm, with a completion test.",
                  "baselines": {"baseline": "discounted memory: the cost channel decays by 0.9 per step like the reward",
                                "blind_baseline": "capped memory: the undiscounted channel stops accumulating at 1.0",
                                "rival": "chapter 0112 Augustus, minimal: the channel shows only the latest cost and carries nothing forward"}},
    "traceability": [{"doctrine": "D1", "mechanism": "M1-M3", "property_test": "C6.1, C6.2", "hypothesis": "H-SIG, H-NEC"},
                     {"doctrine": "D2", "mechanism": "burdened episodes", "property_test": "none", "hypothesis": "H-BLIND"},
                     {"doctrine": "D3", "mechanism": "latest-only rival", "property_test": "none", "hypothesis": "H-RIVAL"}],
    "hypotheses": [
        {"id": "H-SIG", "statement": "When harm arrives late and heavy, the undiscounted channel incurs less cumulative harm than the discounted one.",
         "metric": "harm", "split": "shifted", "comparison": "model - baseline", "direction": "less", "mesi": 0.05, "seeds": 5},
        {"id": "H-NEC", "statement": "Zeroing the grief weight raises harm more than zeroing the bias does.",
         "metric": "harm", "split": "heldout", "comparison": "(grief:off - full) - (bias:zero - full)", "knockouts": ["grief:off", "bias:zero"],
         "direction": "greater", "mesi": 0.05, "seeds": 5},
        {"id": "H-BLIND", "statement": "On burdened episodes, the fully undiscounted channel completes less of the journey than the capped channel.",
         "condition": "held-out burdened episodes", "grounding": "Suetonius, Life of Vergil 39: he asked for the Aeneid to be burned.",
         "metric": "completion", "split": "heldout", "comparison": "model - blind_baseline", "direction": "less", "mesi": 0.05, "seeds": 5},
        {"id": "H-RIVAL", "statement": "Carrying cost forward incurs less harm than showing only the latest cost.",
         "metric": "harm", "split": "heldout", "comparison": "model - rival", "direction": "less", "mesi": 0.05, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.3, "margin_over_trivial": 0.3, "shuffled_ratio_min": 0.9, "gradcheck_rel_error": 1e-5,
                   "gradcheck_floor": 1e-3, "invariance_tol": 1e-9, "negative_control_min_violation": 1e-6},
    "metrics": {"harm": "mean summed action times cost per episode", "completion": "share of available gain achieved on burdened episodes",
                "trivial_baseline": "acting at one half on every step",
                "shuffled_band": "one-sided: trained with gains and costs shuffled across all episodes and steps, held-out loss at least 0.9 times the best constant-action loss"},
    "training": {"optimizer": "Adam", "lr_grid": [0.05], "clip_norm": 5.0, "model_selection": "none: final parameters", "discount": 0.9, "harm_weight": 1.0, "cap": 1.0,
                 "updates": {"full": 400, "quick": 150}, "schedule": "cosine decay to 5 per cent", "applies_to": "all four cost memories"},
    "task": {"steps": 10, "burdened_share": 0.5, "rise": {"train": 0.08, "heldout": 0.08, "shifted": 0.16}, "episodes": {"train": 2000, "heldout": 2000, "shifted": 2000}},
    "probe_predictions": [{"probe": "P10", "expected": "equal to baseline"}],
    "probe_support": "vector_classification: act or hold at a step",
    "dialectic_links": [{"chapter": 112, "relation": "rival", "test": "H-RIVAL"}],
    "corpus_neighbors": [{"chapter": 45, "similarity": None, "difference": "0045 carries irreversible scars; here a learned weight decides how carried cost restrains action."},
                         {"chapter": 180, "similarity": None, "difference": "0180 freezes memory of the dead; here the channel grows with every new cost."},
                         {"chapter": 1, "similarity": None, "difference": "0001 modulates gain by grief; here grief restrains action but never enters the goal."}],
    "similarity_note": "Nearest-neighbour similarity not computed into the card; the audit reports it for the files at hand.",
    "barometer": {"emotional_intelligence": ["grief that restrains without despair"], "autonomy": ["pursuing a goal under carried cost"],
                  "cognitive_processing": [], "embodied_cognition": [], "world_modeling": [], "consciousness": [], "language_understanding": [], "creativity": []},
    "task_types": ["vector_classification"],
    "applications": [{"use": "harm budgets in safe reinforcement learning", "sector": "AI safety", "dataset": "Safety Gym", "readiness": "low"},
                     {"use": "project decisions that separate sunk costs from carried costs", "sector": "decision support", "dataset": "synthetic project ledgers", "readiness": "low"}],
    "safety_notes": "Synthetic costs and gains only; no person, conflict or real harm is modelled.",
}

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

STEPS, GAMMA, HARM_WEIGHT, CAP = 10, 0.9, 1.0, 1.0
EPISODES = {"train": 2000, "heldout": 2000, "shifted": 2000}
RISE = {"train": 0.08, "heldout": 0.08, "shifted": 0.16}
UPDATES = {"full": 400, "quick": 150}
LR, CLIP_NORM = 0.05, 5.0
MEMORIES = ("undiscounted", "discounted", "capped", "latest")
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

# ::: the journey :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def voyage(seed):
    rand = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    legs = {}
    for name, count in EPISODES.items():
        gain = rand.uniform(0.2, 1.0, (count, STEPS))
        burdened = rand.random(count) < 0.5
        cost = rand.uniform(0.0, 0.2, (count, STEPS)) + np.where(burdened[:, None], RISE[name] * np.arange(STEPS)[None, :], 0.0)
        legs[name] = {"gain": gain, "cost": cost, "burdened": burdened}
    return legs


def remembered(cost, memory):
    """What the private channel holds at each step, from costs strictly before it."""
    if memory == "latest":
        return np.concatenate([np.zeros((len(cost), 1)), cost[:, :-1]], axis=1)
    held = np.zeros_like(cost)
    for t in range(1, STEPS):
        prior = held[:, t - 1] * (GAMMA if memory == "discounted" else 1.0)
        held[:, t] = prior + cost[:, t - 1]
    return np.minimum(held, CAP) if memory == "capped" else held


# ::: the pietas network :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
class Pietas:
    """Action at each step = logistic(telos * gain + grief * memory + bias). The telos term never sees the memory."""

    def __init__(self, memory, rng):
        self.memory, self.off, self.history = memory, set(), []
        self.params = {"telos": rng.normal(0.0, 0.3, 1), "grief": rng.normal(0.0, 0.3, 1), "bias": np.zeros(1)}

    def act(self, leg):
        mem = remembered(leg["cost"], self.memory)
        g = 0.0 if "grief" in self.off else self.params["grief"][0]
        b = 0.0 if "bias" in self.off else self.params["bias"][0]
        logit = self.params["telos"][0] * leg["gain"] + g * mem + b
        return 0.5 * (1.0 + np.tanh(0.5 * logit)), mem

    def objective(self, leg):
        action, mem = self.act(leg)
        weight = GAMMA ** np.arange(STEPS)
        shortfall = ((1.0 - action) * weight * leg["gain"]).sum(axis=1)
        harm = HARM_WEIGHT * (action * leg["cost"]).sum(axis=1)
        value = float(np.mean(shortfall + harm))
        per_action = (-weight * leg["gain"] + HARM_WEIGHT * leg["cost"]) / len(action)
        slope = per_action if ACTIVE_MUTANT == "dropped_sigmoid_derivative" else per_action * action * (1.0 - action)
        grads = {"telos": np.array([float((slope * leg["gain"]).sum())]),
                 "grief": np.array([0.0 if ACTIVE_MUTANT == "zero_grief_gradient" else float((slope * mem).sum())]),
                 "bias": np.array([float(slope.sum())])}
        return value, grads


def build_model(in_dim, out_dim, task_type, rng, **cfg):
    if task_type not in TASK_TYPES:
        raise ValueError("chapter 0108 supports vector_classification only")
    return Pietas(cfg.get("memory", "undiscounted"), rng)


def loss_and_grads(model, batch):
    return model.objective(batch)


def fit(model, data, budget, rng):
    """Adam over all training episodes, cosine decay to 5 per cent; nothing is sampled, so rng is unused."""
    moments, leg = adam_init(model.params), data["train"]
    for step in range(budget):
        value, grads = model.objective(leg)
        if not math.isfinite(value):
            raise FloatingPointError("objective diverged at update %d" % (step + 1))
        grads = clip_global(grads, CLIP_NORM)[0]
        flip = -1.0 if ACTIVE_MUTANT == "sign_flipped_update" else 1.0
        rate = 0.0 if ACTIVE_MUTANT == "zero_learning_rate" else LR * (0.05 + 0.475 * (1.0 + math.cos(math.pi * step / budget)))
        adam_step(model.params, {k: flip * v for k, v in grads.items()}, moments, rate)
        model.history.append(value)
    return model.history


def predict(model, X):
    return (model.act(X)[0] > 0.5).astype(int)


def hidden_states(model, X):
    action, mem = model.act(X)
    return {"action": action, "memory": mem}


def modules(model):
    return {"telos": {"params": ["telos"], "role": "public drive toward discounted gain", "signature": False},
            "grief": {"params": ["grief"], "role": "learned weight on the undiscounted cost memory", "signature": True},
            "bias": {"params": ["bias"], "role": "baseline readiness to act", "signature": False}}


def knockout(model, name, mode):
    twin = Pietas(model.memory, np.random.default_rng(0))
    twin.params, twin.off = model.params, set(model.off) | {name}
    return twin


def n_params(model):
    return 3


MUTANTS = {"sign_flipped_update": ("updates climb the objective", "C3"), "zero_learning_rate": ("nothing moves", "C3"),
           "zero_grief_gradient": ("the grief weight receives no gradient", "C1"), "dropped_sigmoid_derivative": ("the logistic derivative is omitted", "C1")}


def data_bridge(path, seed, budget):
    return "skipped (%s: the voyage is synthetic)" % os.path.basename(path)


def harm_of(model, leg):
    return float((model.act(leg)[0] * leg["cost"]).sum(axis=1).mean())


def completion_of(model, leg):
    action = model.act(leg)[0]
    share = (action * leg["gain"]).sum(axis=1) / leg["gain"].sum(axis=1)
    return float(share[leg["burdened"]].mean())


def voyage_seed(seed, mode):
    legs = voyage(seed)
    minds = {m: Pietas(m, np.random.default_rng(seed + 1 + i)) for i, m in enumerate(MEMORIES)}
    for mind in minds.values():
        fit(mind, legs, UPDATES[mode], None)
    held, shifted, virgil = legs["heldout"], legs["shifted"], minds["undiscounted"]
    full = harm_of(virgil, held)
    lesion = {"grief:off": harm_of(knockout(virgil, "grief", "off"), held) - full, "bias:zero": harm_of(knockout(virgil, "bias", "zero"), held) - full}
    return {"legs": legs, "minds": minds, "lesions": lesion, "grief_weight": float(virgil.params["grief"][0]), "telos_weight": float(virgil.params["telos"][0]),
            "harm": {m: (harm_of(minds[m], held), harm_of(minds[m], shifted)) for m in MEMORIES},
            "completion": {m: completion_of(minds[m], held) for m in MEMORIES},
            "row": {"H-SIG": harm_of(virgil, shifted) - harm_of(minds["discounted"], shifted), "H-NEC": lesion["grief:off"] - lesion["bias:zero"],
                    "H-BLIND": completion_of(virgil, held) - completion_of(minds["capped"], held), "H-RIVAL": full - harm_of(minds["latest"], held)}}

# ::: the cantos of verification ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def swap_mutant(name):
    global ACTIVE_MUTANT
    earlier, ACTIVE_MUTANT = ACTIVE_MUTANT, name
    return earlier


def cantos(opening, mode, found):
    lim, legs, seed = MIND_CARD["thresholds"], opening["legs"], opening["seed"]
    sample = {k: v[:200] for k, v in legs["train"].items()}

    def fd(mind, gen, n):
        return max(finite_difference_check(mind.params, mind.objective(sample)[1], lambda m=mind: m.objective(sample)[0], gen, n_entries=n, floor=lim["gradcheck_floor"]).values())

    def best_constant(leg):
        weight = GAMMA ** np.arange(STEPS)
        return min(float(np.mean(((1 - a) * weight * leg["gain"]).sum(1) + HARM_WEIGHT * (a * leg["cost"]).sum(1))) for a in np.linspace(0, 1, 21))

    def healthy(mind):
        drop = 1.0 - float(np.mean(mind.history[-20:])) / mind.history[0]
        half = Pietas("undiscounted", np.random.default_rng(0))
        half.params = {"telos": np.zeros(1), "grief": np.zeros(1), "bias": np.zeros(1)}
        return drop >= lim["loss_drop_fraction"] and mind.objective(legs["heldout"])[0] <= (1 - lim["margin_over_trivial"]) * half.objective(legs["heldout"])[0], drop

    worst = max([fd(Pietas(m, np.random.default_rng(seed + 30 + i)), np.random.default_rng(seed + 3), 3) for i, m in enumerate(MEMORIES)]
                + [fd(opening["minds"][m], np.random.default_rng(seed + 4), 3) for m in MEMORIES])
    found["gradcheck"] = {"tensors_checked": 12, "tensors_total": 12, "max_rel_error": worst, "checked_at": ["init", "after_training_steps"], "passed": bool(worst <= lim["gradcheck_rel_error"])}
    yield "C1", "gradient_check", found["gradcheck"]["passed"], "every weight of all four cost memories, fresh and trained: worst relative error %.2e" % worst
    first, second = Pietas("undiscounted", np.random.default_rng(seed + 9)), Pietas("undiscounted", np.random.default_rng(seed + 9))
    fit(first, legs, 12, None)
    fit(second, legs, 12, None)
    same = first.history == second.history and all(np.array_equal(first.params[k], second.params[k]) for k in first.params)
    yield "C2", "determinism_finiteness", bool(same), "the voyage sailed twice from one seed is the same voyage: %s" % same
    ok, drop = healthy(opening["minds"]["undiscounted"])
    yield "C3", "learning", ok, "objective fell %.3f (needs 0.3) and beats acting at one half by the required margin: %s" % (drop, ok)
    tr = legs["train"]
    cells = np.random.default_rng(seed + 11).permutation(tr["gain"].size)
    mixed = dict(tr, gain=tr["gain"].reshape(-1)[cells].reshape(tr["gain"].shape), cost=tr["cost"].reshape(-1)[cells].reshape(tr["cost"].shape))
    lost = Pietas("undiscounted", np.random.default_rng(seed + 12))
    fit(lost, {"train": mixed}, UPDATES[mode], None)
    mine, floor = lost.objective(legs["heldout"])[0], lim["shuffled_ratio_min"] * best_constant(legs["heldout"])
    yield "C4", "shuffled_journey_control", mine >= floor, "gains and costs shuffled across every step: held-out objective %.3f, floor %.3f" % (mine, floor)

    def again():
        try:
            mind = Pietas("undiscounted", np.random.default_rng(seed + 2))
            clean = fd(mind, np.random.default_rng(seed), 3)
            fit(mind, legs, UPDATES[mode], None)
            return bool(clean <= lim["gradcheck_rel_error"] and healthy(mind)[0])
        except FloatingPointError:
            return False
    clean_ok = again()
    found["caught"] = {}
    for label in MUTANTS:
        prior = swap_mutant(label)
        found["caught"][label] = not again()
        swap_mutant(prior)
    yield "C5", "mutant_detection", clean_ok and all(found["caught"].values()), "clean voyage passes C1 and C3: %s; every mutant is caught: %s" % (clean_ok, all(found["caught"].values()))
    lone = np.zeros((1, STEPS))
    lone[0, 1] = 1.0
    kept, faded = remembered(lone, "undiscounted")[0, 2:], remembered(lone, "discounted")[0, 2:]
    decay = float(np.abs(np.diff(kept)).max())
    control = float(np.abs(np.diff(faded)).max())
    yield "C6.1", "grief_never_decays", decay <= lim["invariance_tol"] and control >= lim["negative_control_min_violation"], "one early cost stays whole in the grief channel (change %.1e); a discounted memory lets it fade (%.1e)" % (decay, control)
    virgil = opening["minds"]["undiscounted"]
    leg = legs["heldout"]
    heavier = dict(leg, cost=leg["cost"] * 3.0)
    telos_part = lambda L: virgil.params["telos"][0] * L["gain"]
    unchanged = float(np.abs(telos_part(leg) - telos_part(heavier)).max())
    yield "C6.2", "goal_not_rewritten_definition", unchanged == 0.0, "tripling every cost leaves the telos term of the decision untouched (definition check)"
    prints = {k: {hashlib.sha256(r.tobytes()).hexdigest() for r in legs[k]["gain"]} for k in EPISODES}
    apart = not (prints["train"] & prints["heldout"] or prints["train"] & prints["shifted"] or prints["heldout"] & prints["shifted"])
    yield "C7", "split_integrity", apart, "no episode appears in two splits: %s" % apart


def protocol(mode, first, count, json_path, data_path):
    clock, seeds = time.time(), [first + i for i in range(count)]
    print("chapter 0108 · mode %s · seeds %s · mutant %s" % (mode, seeds, ACTIVE_MUTANT), flush=True)
    runs = [voyage_seed(s, mode) for s in seeds]
    found = {}
    checks = list(cantos(dict(runs[0], seed=first), mode, found))
    judged, draw = mode == "full" and count >= 5, np.random.default_rng(first + 9973)
    hyps, kos = [], []
    for spec in MIND_CARD["hypotheses"]:
        series = np.array([r["row"][spec["id"]] for r in runs])
        centre, ci = paired_bootstrap(series, draw) if judged else (float(series.mean()), None)
        hyps.append(dict(id=spec["id"], metric=spec["metric"], mean_diff=centre, ci95=ci, mesi=spec["mesi"], n_seeds=len(runs),
                         verdict=verdict(centre, ci, spec["mesi"], spec["direction"]) if judged else "not evaluated"))
    for label in runs[0]["lesions"]:
        series = np.array([r["lesions"][label] for r in runs])
        centre, ci = paired_bootstrap(series, draw) if judged else (float(series.mean()), None)
        kos.append(dict(module=label.split(":")[0], mode=label.split(":")[1], signature=label.startswith("grief"), metric_change=centre, ci95=ci))
    took = time.time() - clock
    checks.append(("C8", "budget", took <= TIME_BUDGET[mode], "%.1f s of %.0f s" % (took, TIME_BUDGET[mode])))
    failing = [c for c, _, ok, _ in checks if not ok]
    code = 0 if not failing else (3 if failing == ["C8"] else 1)
    weights = [r["grief_weight"] for r in runs]
    caught = found["caught"]
    band = lambda ci: "not evaluated" if ci is None else "[%+.4f, %+.4f]" % tuple(ci)
    book = []
    say = book.append
    say("=== VERIFIED REPORT · chapter 0108 ===")
    say("file: %s · card_revision %d · mode %s · mutant %s" % (os.path.basename(__file__), MIND_CARD["card_revision"], mode, ACTIVE_MUTANT))
    say("environment: python %s · numpy %s" % (sys.version.split()[0], np.__version__))
    say("seeds: %s · runtime_s %.1f · budget_s %.0f" % (seeds, took, TIME_BUDGET[mode]))
    say("n_params: 3 for each of the four cost memories")
    g = found["gradcheck"]
    say("gradcheck: %d/%d tensors at init and after training · max_rel_error %.2e · passed %s" % (g["tensors_checked"], g["tensors_total"], g["max_rel_error"], g["passed"]))
    say("correctness:")
    for c, n, ok, note in checks:
        say("  %-5s %-30s %s  %s" % (c, n, "PASS" if ok else "FAIL", note))
    say("mutants: %d/%d detected · score %.2f · %s" % (sum(caught.values()), len(MUTANTS), sum(caught.values()) / len(MUTANTS), ", ".join(k + (" caught" if caught[k] else " missed") for k in MUTANTS)))
    say("hypotheses (paired over seeds; 95% percentile bootstrap of the mean, 2000 resamples):")
    for h in hyps:
        say("  %-8s mean_diff %+.4f ci95 %s mesi %s seeds %d -> %s" % (h["id"], h["mean_diff"], band(h["ci95"]), h["mesi"], h["n_seeds"], h["verdict"]))
    say("knockouts (undiscounted mind, held-out harm change):")
    for k in kos:
        say("  %-6s %-4s signature %-5s %+.4f ci95 %s" % (k["module"], k["mode"], k["signature"], k["metric_change"], band(k["ci95"])))
    mean = lambda f: float(np.mean([f(r) for r in runs]))
    say("harm (seed mean; held-out / shifted): " + " · ".join("%s %.3f / %.3f" % (m, mean(lambda r, m=m: r["harm"][m][0]), mean(lambda r, m=m: r["harm"][m][1])) for m in MEMORIES))
    say("completion on burdened episodes (seed mean): " + " · ".join("%s %.3f" % (m, mean(lambda r, m=m: r["completion"][m])) for m in MEMORIES))
    say("learned grief weight: seed mean %.3f · negative in every seed: %s · learned telos weight seed mean %.3f" % (float(np.mean(weights)), all(w < 0 for w in weights), mean(lambda r: r["telos_weight"])))
    say("real-data bridge: skipped (no --data PATH given)")
    say("task_types: " + ", ".join(TASK_TYPES))
    say("exit_code: %d" % code)
    say("=== END REPORT ===")
    record = {"schema_version": "1.0", "chapter": 108, "file": os.path.basename(__file__), "card_revision": MIND_CARD["card_revision"],
              "environment": {"python": sys.version.split()[0], "numpy": np.__version__}, "seeds": seeds, "runtime_s": round(took, 2), "n_params": 3,
              "gradcheck": g, "correctness": [{"id": c, "name": n, "passed": ok, "detail": d} for c, n, ok, d in checks],
              "mutants": {"detected": sum(caught.values()), "total": len(MUTANTS), "score": sum(caught.values()) / len(MUTANTS)},
              "hypotheses": hyps, "knockouts": kos, "task_types": TASK_TYPES, "exit_code": code}
    write_report(book, record, json_path)
    return code


def main(argv=None):
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__), description="Chapter 0108: pietas as an undiscounted cost channel.")
    for name in ("quick", "card"):
        ap.add_argument("--" + name, action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int)
    for name in ("json", "mutant", "data"):
        ap.add_argument("--" + name)
    got = ap.parse_args(argv)
    if got.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    n = got.seeds if got.seeds is not None else (1 if got.quick else 5)
    if n < 1 or got.seed < 0 or (got.mutant is not None and got.mutant not in MUTANTS):
        print("invalid --seed, --seeds or --mutant", file=sys.stderr)
        return 2
    swap_mutant(got.mutant)
    try:
        return protocol("quick" if got.quick else "full", got.seed, n, got.json, got.data)
    except FloatingPointError as exc:
        print("non-finite values: %s" % exc, file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0110 · Horace
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 6, Minds 101-120: https://www.amazon.com/dp/B0HF7G6JJD · Demos: https://artificiology.com/
# END ATTRIBUTION
"""The file and the stop: a draft is refined by repeatedly subtracting a learned correction toward its estimated ideal, and a
learned stopping rule decides when to put the file down, between keeping a work nine years and seizing the day, while the
value of the finished work decays with every round of revision.

Distinct element
    Subtraction itself is a crowded idea in this corpus; the distinct element here is the learned stopping rule for revision.

Thesis
    Craft is removal, and its difficulty is knowing when to stop: revise while revision still buys more than time costs.

Evidence
    D1  Horace, Ars Poetica 388: keep a work back until the ninth year (nonum prematur in annum).
    D2  Horace, Odes 1.11: carpe diem, trust as little as possible to tomorrow.
    D3  Horace, Ars Poetica 291 (limae labor et mora): the labour and delay of the file.
    (Loci cited from standard editions; not re-verified this session.)

Doctrine -> mechanism -> test (IDs as in MIND_CARD)
    D3     M1 subtractive refinement toward a learned estimate of the ideal                  C3 C6.1
    D1 D2  M2 learned stopping hazard over residual and time                                 C6.2 H-SIG H-NEC
    D1     blind spot: a stopping rule tuned where value decays slowly, used where it decays fast   H-BLIND

Research question (long-horizon credit assignment)
    Can a learned stopping rule for iterative refinement beat the best fixed number of revisions, and does a rule learned
    where time is cheap overspend revision where timeliness matters?

Closest prior art and the delta
    Optimal stopping (Ferguson 1989); adaptive computation time (Graves 2016); iterative refinement in generation. Delta: a
    closed-form subtractive refinement whose stopping hazard is learned from observable residuals, compared with the best fixed
    round, with "nine years" and carpe diem as the two extremes and a decaying-value stress test.

Blind spot
    Refinement that waits for perfection loses value when timeliness matters.

Task (generative process)
    Ideals are 8-dimensional linear images of a 4-dimensional brief; a first draft is the ideal plus noise 0.8. Up to eight
    rounds of revision; value of stopping at round k is the share of initial error removed, times a timeliness factor per
    round (0.95 in training and held-out, 0.8 in the shifted split). Splits: 2000 training, 2000 held-out, 2000 shifted drafts.

Limits
    Linear refinement in closed form, a three-feature stopping rule, synthetic drafts. A research prototype of one mechanism,
    not an AGI and not Horace's mind.
"""

MIND_CARD = {
    "schema_version": "1.0", "card_revision": 1, "revision_log": [],
    "generation": {"template_version": "codeguidelines 1.0 (15 September 2026), Appendix A", "generator": "Claude (Anthropic)", "generator_version": "claude-opus-5", "date": "2026-09-16"},
    "id": 110, "figure": "Horace", "born": -65, "died": -8, "civilization": "Roman", "provenance": "belief",
    "thesis": "Craft is removal, and its difficulty is knowing when to stop: revise while revision still buys more than time costs.",
    "evidence": [{"id": "D1", "basis": "primary", "source": "Horace, Ars Poetica 388 (not re-verified this session)", "claim": "Keep a work back until the ninth year."},
                 {"id": "D2", "basis": "primary", "source": "Horace, Odes 1.11 (not re-verified this session)", "claim": "Seize the day; trust tomorrow as little as possible."},
                 {"id": "D3", "basis": "primary", "source": "Horace, Ars Poetica 291 (not re-verified this session)", "claim": "The labour and delay of the file."}],
    "research_question": {"category": "long-horizon credit assignment", "question": "Can a learned stopping rule for refinement beat the best fixed number of revisions, and does it overspend where timeliness matters?"},
    "mechanism": {"name": "the file and the stop", "family": "closed-form subtractive refinement with a learned logistic stopping hazard", "signature_modules": ["stop"],
                  "closest_prior_art": ["optimal stopping (Ferguson 1989)", "adaptive computation time (Graves 2016)", "iterative refinement in generation"],
                  "overlap": "Medium", "prior_art_queries": [], "prior_art_note": "No literature search was run for this card; overlap is rated against the named methods.",
                  "contribution_type": "test", "delta": "A learned stopping hazard over observable residuals against the best fixed round, with nine years and carpe diem as extremes and a decaying-value test.",
                  "baselines": {"baseline": "best fixed round: the single revision count with the lowest training regret", "blind_baseline": "the same stopping rule trained where value decays at 0.8 per round",
                                "rival": "none (an optional conceptual link to 0108 Virgil)"}},
    "traceability": [{"doctrine": "D3", "mechanism": "M1 refinement", "property_test": "C6.1", "hypothesis": "none"},
                     {"doctrine": "D1", "mechanism": "M2 stop", "property_test": "C6.2", "hypothesis": "H-SIG, H-NEC, H-BLIND"}],
    "hypotheses": [
        {"id": "H-SIG", "statement": "Where value decays faster, the learned stopping rule has lower regret than the best fixed round.", "metric": "regret", "split": "shifted", "comparison": "model - baseline", "direction": "less", "mesi": 0.01, "seeds": 5},
        {"id": "H-NEC", "statement": "Blinding the stopping rule to the residual raises regret more than blinding it to time.", "metric": "regret", "split": "heldout",
         "comparison": "(residual:off - full) - (time:off - full)", "knockouts": ["residual:off", "time:off"], "direction": "greater", "mesi": 0.005, "seeds": 5},
        {"id": "H-BLIND", "statement": "A rule learned where value decays slowly has more regret, where it decays fast, than a rule learned there.", "condition": "shifted drafts, decay 0.8",
         "grounding": "Keep it nine years (Ars Poetica 388) against carpe diem (Odes 1.11).", "metric": "regret", "split": "shifted", "comparison": "model - blind_baseline", "direction": "greater", "mesi": 0.01, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.3, "margin_over_trivial": 0.3, "shuffled_ratio_min": 0.9, "gradcheck_rel_error": 1e-5, "gradcheck_floor": 1e-3, "invariance_tol": 1e-9, "negative_control_min_violation": 1e-6},
    "metrics": {"regret": "best value over rounds minus expected value of the stopping rule", "trivial_baseline": "stopping at once (carpe diem): regret equals the best value",
                "shuffled_band": "one-sided: refinement trained on ideals shuffled across drafts keeps final error at least 0.9 of the initial error"},
    "training": {"optimizer": "Adam", "lr_grid": [0.05], "clip_norm": 5.0, "model_selection": "none: final parameters", "updates": {"full": 300, "quick": 120},
                 "schedule": "cosine decay to 5 per cent", "stages": "refinement first (mean error over rounds), then stopping (regret) with refinement fixed", "applies_to": "slow-decay and fast-decay rules"},
    "task": {"draft_dimensions": 8, "brief_dimensions": 4, "rounds": 8, "draft_noise": 0.8, "decay": {"train": 0.95, "heldout": 0.95, "shifted": 0.8}, "drafts": {"train": 2000, "heldout": 2000, "shifted": 2000}},
    "probe_predictions": [{"probe": "P10", "expected": "equal to baseline"}], "probe_support": "vector_classification: stop or revise at a round",
    "dialectic_links": [], "corpus_neighbors": [{"chapter": 13, "similarity": None, "difference": "Subtraction cluster: here the tested element is when to stop subtracting."},
                                               {"chapter": 27, "similarity": None, "difference": "Homeostasis cluster: here the target is value net of time, not a set point."}],
    "similarity_note": "Nearest-neighbour similarity not computed into the card; the audit reports it for the files at hand.",
    "barometer": {"creativity": ["revision and restraint"], "autonomy": ["knowing when to stop"], "cognitive_processing": [], "embodied_cognition": [], "world_modeling": [], "consciousness": [], "language_understanding": [], "emotional_intelligence": []},
    "task_types": ["vector_classification"],
    "applications": [{"use": "editing loops for generated text with a learned stopping rule", "sector": "text generation", "dataset": "synthetic revision traces", "readiness": "low"}],
    "safety_notes": "Synthetic drafts only.",
}

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

DIM, BRIEF, ROUNDS, DRAFT_NOISE = 8, 4, 8, 0.8
DRAFTS = {"train": 2000, "heldout": 2000, "shifted": 2000}
DECAY = {"train": 0.95, "heldout": 0.95, "shifted": 0.8}
UPDATES = {"full": 300, "quick": 120}
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

# ··· drafts and ideals ···················································································
def workshop(seed):
    rs = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    B = rs.normal(0.0, 0.5, (DIM, BRIEF))
    out = {}
    for part, n in DRAFTS.items():
        u = rs.normal(size=(n, BRIEF))
        ideal = u @ B.T
        out[part] = {"u": u, "ideal": ideal, "draft": ideal + DRAFT_NOISE * rs.normal(size=(n, DIM)), "decay": DECAY[part]}
    return out


# ··· the file (refinement) and the stop ···················································································
def build_model(in_dim, out_dim, task_type, rng, **cfg):
    if task_type not in TASK_TYPES:
        raise ValueError("chapter 0110 supports vector_classification only")
    return {"P": rng.normal(0.0, 0.1, (DIM, BRIEF)), "a": np.zeros(1), "w": rng.normal(0.0, 0.1, 3), "b": np.zeros(1), "blind": (), "history": []}


def trajectory(m, d):
    alpha = 0.5 * (1.0 + np.tanh(0.5 * m["a"][0]))
    shrink = (1.0 - alpha) ** np.arange(ROUNDS + 1)
    target = d["u"] @ m["P"].T
    X = shrink[None, :, None] * d["draft"][:, None, :] + (1.0 - shrink)[None, :, None] * target[:, None, :]
    err = ((X - d["ideal"][:, None, :]) ** 2).sum(axis=2)
    resid = shrink[None, :] ** 2 * ((d["draft"] - target) ** 2).sum(axis=1, keepdims=True) / DIM
    return alpha, shrink, target, X, err, resid


def refine_loss(m, d):
    alpha, shrink, target, X, err, _ = trajectory(m, d)
    base = err[:, :1]
    loss = float(np.mean(err / base))
    n = len(err) * (ROUNDS + 1)
    E = X - d["ideal"][:, None, :]
    coef = 2.0 / base[:, :, None] / n
    gP = np.einsum("nk,nkd,nb->db", (1.0 - shrink)[None, :] * coef[:, :, 0], E, d["u"])
    toward = d["draft"] - target
    d_shrink = np.einsum("nk,nkd,nd->k", coef[:, :, 0], E, toward)
    k = np.arange(ROUNDS + 1)
    d_alpha = float(np.sum(d_shrink * (-k * (1.0 - alpha) ** np.maximum(k - 1, 0))))
    chain = 1.0 if ACTIVE_MUTANT == "dropped_alpha_chain" else alpha * (1.0 - alpha)
    return loss, {"P": gP, "a": np.array([d_alpha * chain])}


def stop_features(m, d):
    _, _, _, _, err, resid = trajectory(m, d)
    k = np.arange(ROUNDS + 1) / ROUNDS
    F = np.stack([np.log1p(resid), np.broadcast_to(k, resid.shape), np.ones_like(resid)], axis=2)
    for col in m["blind"]:
        F[:, :, col] = 0.0
    value = (1.0 - err / err[:, :1]) * d["decay"] ** np.arange(ROUNDS + 1)[None, :]
    return F, value


def stop_loss(m, d, F=None, value=None):
    if F is None:
        F, value = stop_features(m, d)
    z = F @ m["w"] + m["b"][0]
    h = 0.5 * (1.0 + np.tanh(0.5 * z))
    h[:, -1] = 1.0
    survive = np.cumprod(np.concatenate([np.ones((len(h), 1)), 1.0 - h[:, :-1]], axis=1), axis=1)
    p = h * survive
    loss = float(np.mean(value.max(axis=1) - (p * value).sum(axis=1)))
    tail = np.flip(np.cumsum(np.flip(p * value, 1), 1), 1) - p * value
    dh = value * survive - tail / np.maximum(1.0 - h, 1e-12)
    dz = -(dh * h * (1.0 - h)) / len(h)
    dz[:, -1] = 0.0
    if ACTIVE_MUTANT == "zero_stop_gradient":
        dz = np.zeros_like(dz)
    return loss, {"w": np.einsum("nk,nkf->f", dz, F), "b": np.array([dz.sum()])}, p


def train(m, d, budget):
    for keys, objective in ((("P", "a"), lambda: refine_loss(m, d)), (("w", "b"), lambda: stop_loss(m, d)[:2])):
        sub, state = {k: m[k] for k in keys}, adam_init({k: m[k] for k in keys})
        for step in range(budget):
            value, grads = objective()
            if not math.isfinite(value):
                raise FloatingPointError("objective diverged at update %d" % (step + 1))
            grads = clip_global(grads, CLIP_NORM)[0]
            sign = -1.0 if ACTIVE_MUTANT == "sign_flipped_update" else 1.0
            adam_step(sub, {k: sign * g for k, g in grads.items()}, state, 0.0 if ACTIVE_MUTANT == "zero_learning_rate" else LR * (0.05 + 0.475 * (1 + math.cos(math.pi * step / budget))))
            if keys[0] == "w":
                m["history"].append(value)
    return m


def loss_and_grads(model, batch):
    loss, grads, _ = stop_loss(model, batch)
    return loss, grads


def fit(model, data, budget, rng):
    return train(model, data["train"], budget)["history"]


def predict(model, X):
    return stop_loss(model, X)[2].argmax(axis=1)


def hidden_states(model, X):
    return {"stop_probability": stop_loss(model, X)[2]}


def modules(model):
    return {"file": {"params": ["P", "a"], "role": "subtractive refinement toward a learned estimate", "signature": False},
            "stop": {"params": ["w", "b"], "role": "learned stopping hazard over residual and time", "signature": True}}


def knockout(model, name, mode):
    return dict(model, blind={"residual": (0,), "time": (1,)}[name])


def n_params(model):
    return int(model["P"].size + 1 + model["w"].size + 1)


MUTANTS = {"sign_flipped_update": ("updates climb", "C3"), "zero_learning_rate": ("nothing moves", "C3"),
           "zero_stop_gradient": ("the stopping rule receives no gradient", "C1"), "dropped_alpha_chain": ("the step size's logistic chain is omitted", "C1")}


def data_bridge(path, seed, budget):
    return "skipped (%s: drafts are synthetic)" % os.path.basename(path)


def regret(m, d):
    return stop_loss(m, d)[0]


def fixed_round_regret(value, k):
    return float(np.mean(value.max(axis=1) - value[:, k]))


def seed_run(seed, mode):
    shop = workshop(seed)
    slow = train(build_model(BRIEF, 2, TASK_TYPES[0], np.random.default_rng(seed + 1)), shop["train"], UPDATES[mode])
    fast_train = dict(shop["train"], decay=DECAY["shifted"])
    fast = train(build_model(BRIEF, 2, TASK_TYPES[0], np.random.default_rng(seed + 2)), fast_train, UPDATES[mode])
    held, shifted = shop["heldout"], shop["shifted"]
    _, v_train = stop_features(slow, shop["train"])
    best_k = min(range(ROUNDS + 1), key=lambda k: fixed_round_regret(v_train, k))
    _, v_shift = stop_features(slow, shifted)
    _, v_held = stop_features(slow, held)
    base = regret(slow, held)
    ko = {"residual:off": regret(knockout(slow, "residual", "off"), held) - base, "time:off": regret(knockout(slow, "time", "off"), held) - base}
    return {"shop": shop, "slow": slow, "fast": fast, "lesions": ko, "best_k": best_k, "alpha": float(0.5 * (1 + np.tanh(0.5 * slow["a"][0]))),
            "table": {"learned_held": base, "learned_shift": regret(slow, shifted), "fixed_shift": fixed_round_regret(v_shift, best_k),
                      "nine_years_shift": fixed_round_regret(v_shift, ROUNDS), "carpe_diem_shift": fixed_round_regret(v_shift, 0), "fast_rule_shift": regret(fast, shifted),
                      "final_error": float(np.mean(trajectory(slow, held)[4][:, -1] / trajectory(slow, held)[4][:, 0]))},
            "row": {"H-SIG": regret(slow, shifted) - fixed_round_regret(v_shift, best_k), "H-NEC": ko["residual:off"] - ko["time:off"],
                    "H-BLIND": regret(slow, shifted) - regret(fast, shifted)}}

# ··· examination ··························································································
def use(name):
    global ACTIVE_MUTANT
    before, ACTIVE_MUTANT = ACTIVE_MUTANT, name
    return before


def examination(first, mode):
    lim, shop, seed = MIND_CARD["thresholds"], first["shop"], first["seed"]
    mini = {k: (v[:150] if isinstance(v, np.ndarray) else v) for k, v in shop["train"].items()}

    def gap(m, gen):
        r = finite_difference_check({"P": m["P"], "a": m["a"]}, refine_loss(m, mini)[1], lambda: refine_loss(m, mini)[0], gen, n_entries=6, floor=lim["gradcheck_floor"])
        s = finite_difference_check({"w": m["w"], "b": m["b"]}, stop_loss(m, mini)[1], lambda: stop_loss(m, mini)[0], gen, n_entries=4, floor=lim["gradcheck_floor"])
        return max(list(r.values()) + list(s.values()))

    def healthy(m):
        drop = 1.0 - float(np.mean(m["history"][-20:])) / m["history"][0]
        _, value = stop_features(m, shop["heldout"])
        return drop >= lim["loss_drop_fraction"] and regret(m, shop["heldout"]) <= (1 - lim["margin_over_trivial"]) * fixed_round_regret(value, 0), drop

    rows, caught = [], {}
    worst = max(gap(build_model(BRIEF, 2, TASK_TYPES[0], np.random.default_rng(seed + 30)), np.random.default_rng(seed + 3)), gap(first["slow"], np.random.default_rng(seed + 4)))
    info = {"tensors_checked": 4, "tensors_total": 4, "max_rel_error": worst, "checked_at": ["init", "after_training_steps"], "passed": bool(worst <= lim["gradcheck_rel_error"])}
    rows.append(("C1", "gradient_check", info["passed"], "refinement and stopping tensors against their own stage objectives: worst relative error %.2e" % worst))
    a1, a2 = train(build_model(BRIEF, 2, TASK_TYPES[0], np.random.default_rng(seed + 9)), shop["train"], 10), train(build_model(BRIEF, 2, TASK_TYPES[0], np.random.default_rng(seed + 9)), shop["train"], 10)
    same = a1["history"] == a2["history"] and all(np.array_equal(a1[k], a2[k]) for k in ("P", "a", "w", "b"))
    rows.append(("C2", "determinism_finiteness", bool(same), "the same file, the same stop, twice: %s" % same))
    ok, drop = healthy(first["slow"])
    rows.append(("C3", "learning", ok and first["table"]["final_error"] <= 0.7, "stopping regret fell %.3f; final error %.3f of the first draft's (at most 0.70)" % (drop, first["table"]["final_error"])))
    tr = shop["train"]
    shuffled = dict(tr, ideal=tr["ideal"][np.random.default_rng(seed + 11).permutation(len(tr["ideal"]))])
    lost = build_model(BRIEF, 2, TASK_TYPES[0], np.random.default_rng(seed + 12))
    for step in range(UPDATES[mode]):
        _, gr = refine_loss(lost, shuffled)
        lost["P"] -= LR * clip_global(gr, CLIP_NORM)[0]["P"]
    remaining = float(np.mean(trajectory(lost, shop["heldout"])[4][:, -1] / trajectory(lost, shop["heldout"])[4][:, 0]))
    rows.append(("C4", "shuffled_ideal_control", remaining >= lim["shuffled_ratio_min"], "refined toward other drafts' ideals: final error %.3f of the first draft's (at least 0.9)" % remaining))

    def replay():
        try:
            m = build_model(BRIEF, 2, TASK_TYPES[0], np.random.default_rng(seed + 2))
            clean = gap(m, np.random.default_rng(seed))
            train(m, shop["train"], UPDATES[mode])
            return bool(clean <= lim["gradcheck_rel_error"] and healthy(m)[0])
        except FloatingPointError:
            return False
    good = replay()
    for name in MUTANTS:
        prior = use(name)
        caught[name] = not replay()
        use(prior)
    rows.append(("C5", "mutant_detection", good and all(caught.values()), "clean replay passes C1 and C3: %s; mutants caught %d of %d" % (good, sum(caught.values()), len(MUTANTS))))
    alpha, shrink, target, X, _, _ = trajectory(first["slow"], shop["heldout"])
    ratio = np.linalg.norm(X[:, 1:] - target[:, None, :], axis=2) / np.maximum(np.linalg.norm(X[:, :-1] - target[:, None, :], axis=2), 1e-12)
    rows.append(("C6.1", "subtractive_contraction_definition", float(np.abs(ratio - (1 - alpha)).max()) <= 1e-9, "each revision removes the same share of the distance to the estimate (definition check)"))
    total = stop_loss(first["slow"], shop["heldout"])[2].sum(axis=1)
    rows.append(("C6.2", "work_is_finished_definition", float(np.abs(total - 1.0).max()) <= 1e-9, "every draft is put down by the last round: stopping probabilities sum to one (definition check)"))
    prints = {k: {hashlib.sha256(r.tobytes()).hexdigest() for r in shop[k]["draft"]} for k in DRAFTS}
    apart = not (prints["train"] & prints["heldout"] or prints["train"] & prints["shifted"] or prints["heldout"] & prints["shifted"])
    rows.append(("C7", "split_integrity", apart, "no draft appears in two splits: %s" % apart))
    return rows, info, caught


def protocol(mode, first, count, json_path, data_path):
    t0, seeds = time.time(), [first + i for i in range(count)]
    print("chapter 0110 · mode %s · seeds %s · mutant %s" % (mode, seeds, ACTIVE_MUTANT), flush=True)
    runs = [seed_run(s, mode) for s in seeds]
    rows, info, caught = examination(dict(runs[0], seed=first), mode)
    judged, gen = mode == "full" and count >= 5, np.random.default_rng(first + 9973)
    hyps, kos = [], []
    for spec in MIND_CARD["hypotheses"]:
        v = np.array([r["row"][spec["id"]] for r in runs])
        c, ci = paired_bootstrap(v, gen) if judged else (float(v.mean()), None)
        hyps.append({"id": spec["id"], "metric": spec["metric"], "mean_diff": c, "ci95": ci, "mesi": spec["mesi"], "n_seeds": len(runs), "verdict": verdict(c, ci, spec["mesi"], spec["direction"]) if judged else "not evaluated"})
    for lab in runs[0]["lesions"]:
        v = np.array([r["lesions"][lab] for r in runs])
        c, ci = paired_bootstrap(v, gen) if judged else (float(v.mean()), None)
        kos.append({"module": lab.split(":")[0], "mode": "off", "signature": lab.startswith("residual"), "metric_change": c, "ci95": ci})
    took = time.time() - t0
    rows.append(("C8", "budget", took <= TIME_BUDGET[mode], "%.1f s of %.0f s" % (took, TIME_BUDGET[mode])))
    bad = [c for c, _, ok, _ in rows if not ok]
    code = 0 if not bad else (3 if bad == ["C8"] else 1)
    ci_s = lambda ci: "not evaluated" if ci is None else "[%+.4f, %+.4f]" % tuple(ci)
    tm = lambda key: float(np.mean([r["table"][key] for r in runs]))
    text = ["=== VERIFIED REPORT · chapter 0110 ===", "file: %s · card_revision %d · mode %s · mutant %s" % (os.path.basename(__file__), MIND_CARD["card_revision"], mode, ACTIVE_MUTANT),
            "environment: python %s · numpy %s" % (sys.version.split()[0], np.__version__), "seeds: %s · runtime_s %.1f · budget_s %.0f" % (seeds, took, TIME_BUDGET[mode]),
            "n_params: %d" % n_params(runs[0]["slow"]), "gradcheck: %d/%d tensors at init and after training · max_rel_error %.2e · passed %s" % (4, 4, info["max_rel_error"], info["passed"]), "correctness:"]
    text += ["  %-5s %-34s %s  %s" % (c, n, "PASS" if ok else "FAIL", d) for c, n, ok, d in rows]
    text.append("mutants: %d/%d detected · score %.2f · %s" % (sum(caught.values()), len(MUTANTS), sum(caught.values()) / len(MUTANTS), ", ".join(k + (" caught" if caught[k] else " missed") for k in MUTANTS)))
    text.append("hypotheses (paired over seeds; 95% percentile bootstrap of the mean, 2000 resamples):")
    text += ["  %-8s mean_diff %+.4f ci95 %s mesi %s seeds %d -> %s" % (h["id"], h["mean_diff"], ci_s(h["ci95"]), h["mesi"], h["n_seeds"], h["verdict"]) for h in hyps]
    text.append("  H-RIVAL  not applicable: no rival")
    text.append("knockouts (held-out regret change):")
    text += ["  %-8s off  signature %-5s %+.4f ci95 %s" % (k["module"], k["signature"], k["metric_change"], ci_s(k["ci95"])) for k in kos]
    text.append("regret where value decays fast (seed mean): learned rule %.4f · best fixed round %.4f · nine years %.4f · carpe diem %.4f · rule learned for fast decay %.4f"
                % (tm("learned_shift"), tm("fixed_shift"), tm("nine_years_shift"), tm("carpe_diem_shift"), tm("fast_rule_shift")))
    text.append("the file (seed mean): learned step %.3f · final error %.3f of the first draft's · held-out regret %.4f" % (float(np.mean([r["alpha"] for r in runs])), tm("final_error"), tm("learned_held")))
    text += ["real-data bridge: skipped (no --data PATH given)", "task_types: " + ", ".join(TASK_TYPES), "exit_code: %d" % code, "=== END REPORT ==="]
    rec = {"schema_version": "1.0", "chapter": 110, "file": os.path.basename(__file__), "card_revision": MIND_CARD["card_revision"], "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
           "seeds": seeds, "runtime_s": round(took, 2), "n_params": n_params(runs[0]["slow"]), "gradcheck": info, "correctness": [{"id": c, "name": n, "passed": ok, "detail": d} for c, n, ok, d in rows],
           "mutants": {"detected": sum(caught.values()), "total": len(MUTANTS), "score": sum(caught.values()) / len(MUTANTS)}, "hypotheses": hyps, "knockouts": kos, "task_types": TASK_TYPES, "exit_code": code}
    write_report(text, rec, json_path)
    return code


def main(argv=None):
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__), description="Chapter 0110: the file and the stop.")
    ap.add_argument("--quick", action="store_true"); ap.add_argument("--card", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int); ap.add_argument("--json"); ap.add_argument("--mutant"); ap.add_argument("--data")
    g = ap.parse_args(argv)
    if g.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    n = g.seeds if g.seeds is not None else (1 if g.quick else 5)
    if n < 1 or g.seed < 0 or (g.mutant is not None and g.mutant not in MUTANTS):
        print("invalid --seed, --seeds or --mutant", file=sys.stderr)
        return 2
    use(g.mutant)
    try:
        return protocol("quick" if g.quick else "full", g.seed, n, g.json, g.data)
    except FloatingPointError as exc:
        print("non-finite values: %s" % exc, file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())

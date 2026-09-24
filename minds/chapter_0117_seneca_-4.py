#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0117 · Seneca
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 6, Minds 101-120: https://www.amazon.com/dp/B0HF7G6JJD · Demos: https://artificiology.com/
# END ATTRIBUTION
"""Assent-gated mind with calibrated praemeditatio: an imagination stage that projects possible harm, trained to be
accurate, feeding a gate that decides how much of each impression to endorse as suffering.

Thesis
    Rehearse what can happen, accurately, and then assent only to what is real: a mind should neither suffer the
    harms it merely imagines nor ignore the harms that actually arrive.

Evidence and provenance
    Provenance is belief: the Letters, the dialogues and Tacitus' account survive.
    D1  Letter 13.4: there are more things likely to frighten us than to crush us; we suffer more often in
        imagination than in reality. 13 continues: some things torment us more than, before, or when they ought not.
    D2  Letter 13: we do not put to the test the things that cause our fear; we do not examine them.
    D3  Letter 91: the unexpected crushes hardest, so nothing should be unexpected; think not only of what usually
        happens but of what can happen (praemeditatio malorum).
    D4  Tacitus, Annals 15.60-64: the feared order did come; Seneca was named in the Pisonian conspiracy and died by
        Nero's order (history only; nothing in this file models death or self-harm).
    D5  The dichotomy of control, sorting impressions into what is and is not up to us, is the doctrine chapter 0125
        (Epictetus) implements; here it is shared scaffolding.

Doctrine -> mechanism -> test (IDs as in MIND_CARD)
    D5        M1 encoder and M2 partition: controllable and Fortune subspaces          C6.1 C6.2 H-NEC (matched)
    D3        M3 praemeditatio: harm magnitude from the Fortune subspace, trained accurate   H-SIG H-NEC
    D1 D2     M4 assent gate: imagined harm and a weak reality cue; suffering = assent x harm  C6.3 H-SIG H-RIVAL
    D1 D4     M5 objective: judgment, praemeditatio accuracy, suffering gap, gentle epoche    H-BLIND
    D5        rival: Epictetus gate with the partition and no imagination stage               H-RIVAL

Research question (calibration, abstention and metacognition)
    Does an accurate imagination of harm, feeding an assent gate, keep experienced distress calibrated to real harm,
    and where does discounting imagined harm leave a mind blind to rare real catastrophes?

Closest prior art and the delta
    Model-based agents that plan with learned world models and imagined rollouts (Ha and Schmidhuber 2018; Hafner et
    al. 2020); risk-sensitive and distributional objectives for tail risk (Bellemare, Dabney and Munos 2017); selective
    prediction with abstention. Delta: a harm-imagination head trained for accuracy whose output reaches affect only
    through a learned assent gate with a weak reality cue, tested against the same network without the accuracy
    term, against a precautionary objective on rare catastrophes, and against a gate without imagination.

Blind spot
    A mind trained to withhold assent from imagined harm can withhold it from the rare catastrophe that is real.

Task (generative process)
    Latent controllable part c (4) and Fortune part f (4), mixed into a 10-dimensional impression with noise. The
    judgment label depends on c. Harm magnitude is softplus of a projection of f, multiplied by six in its top
    two per cent (the catastrophes). Harm is real with probability sigmoid(-1.4 + v.f); catastrophes are always real.
    The reality cue is realness plus Gaussian noise of s.d. 1. Splits: 4000 training, 4000 held-out, 4000 shifted
    impressions (Fortune shifted by 0.6 along the harm direction, so catastrophes are more frequent).

Limits
    Synthetic impressions, one-step harms, no actions, no time. A research prototype of one mechanism, not an AGI and
    not Seneca's mind.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 1,
    "revision_log": [],
    "generation": {"template_version": "codeguidelines 1.0 (15 September 2026), Appendix A",
                   "generator": "Claude (Anthropic)", "generator_version": "claude-opus-5", "date": "2026-09-16"},
    "id": 117, "figure": "Seneca", "born": -4, "died": 65, "civilization": "Roman",
    "provenance": "belief",
    "thesis": ("Rehearse what can happen, accurately, and then assent only to what is real: a mind should neither suffer the "
               "harms it merely imagines nor ignore the harms that actually arrive."),
    "evidence": [
        {"id": "D1", "basis": "primary", "source": "Seneca, Epistulae Morales 13.4 (trans. R. M. Gummere)",
         "claim": "There are more things likely to frighten us than to crush us; we suffer more often in imagination than in reality."},
        {"id": "D2", "basis": "primary", "source": "Seneca, Epistulae Morales 13",
         "claim": "We do not put to the test the things that cause our fear; we do not examine them."},
        {"id": "D3", "basis": "primary", "source": "Seneca, Epistulae Morales 91",
         "claim": "The unexpected crushes hardest; nothing should be unexpected; consider what can happen, not only what usually does."},
        {"id": "D4", "basis": "primary", "source": "Tacitus, Annals 15.60-64",
         "claim": "Named in the Pisonian conspiracy, Seneca received Nero's order and died (recorded as history only)."},
        {"id": "D5", "basis": "scholarship", "source": "Epictetus, Enchiridion 1, as implemented in chapter 0125",
         "claim": "Impressions are sorted into what is up to us and what is not; only the former are invested with judgment."},
    ],
    "research_question": {
        "category": "calibration, abstention and metacognition",
        "question": ("Does an accurate imagination of harm, feeding an assent gate, keep experienced distress calibrated to real "
                     "harm, and where does discounting imagined harm leave a mind blind to rare real catastrophes?")},
    "mechanism": {
        "name": "assent gate fed by calibrated praemeditatio",
        "family": "small feed-forward network with a partitioned latent, a harm-imagination head and a learned gate",
        "signature_modules": ["praemeditatio", "gate"],
        "closest_prior_art": [
            "model-based agents with learned world models and imagined rollouts (Ha and Schmidhuber 2018; Hafner et al. 2020)",
            "distributional and risk-sensitive objectives for tail outcomes (Bellemare, Dabney and Munos 2017)",
            "selective prediction and abstention"],
        "overlap": "Medium",
        "prior_art_queries": [],
        "prior_art_note": "No literature search was run for this card; overlap is rated against the named methods.",
        "contribution_type": "test",
        "delta": ("An imagination head trained for accuracy whose output reaches suffering only through a learned assent gate with a "
                  "weak reality cue, tested against the same network without the accuracy term, against a precautionary "
                  "objective on rare catastrophes, and against a gate without imagination."),
        "baselines": {
            "baseline": "unanchored harm signal: the same five organs and objective with the praemeditatio accuracy term removed",
            "blind_baseline": ("precautionary gate: the same organs; under-endorsing real harm is penalised three times more than "
                               "over-endorsing it, with no epoche term"),
            "rival": ("chapter 0125 Epictetus, minimal: encoder, partition and judgment as here; the gate reads only the reality cue "
                      "and suffering uses one learned constant harm, with no imagination stage")}},
    "traceability": [
        {"doctrine": "D5", "mechanism": "M1 encoder; M2 partition", "property_test": "C6.1, C6.2", "hypothesis": "H-NEC"},
        {"doctrine": "D3", "mechanism": "M3 praemeditatio", "property_test": "none", "hypothesis": "H-SIG, H-NEC"},
        {"doctrine": "D1", "mechanism": "M4 assent gate", "property_test": "C6.3", "hypothesis": "H-SIG, H-RIVAL"},
        {"doctrine": "D2", "mechanism": "M5 objective (accuracy of imagination)", "property_test": "none", "hypothesis": "H-SIG"},
        {"doctrine": "D4", "mechanism": "catastrophe tail of the task", "property_test": "none", "hypothesis": "H-BLIND"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": ("Under a shift that makes catastrophes more frequent, calibrated praemeditatio keeps suffering "
                                      "closer to warranted harm than the same network without the accuracy term."),
         "metric": "calibration_error", "split": "shifted", "comparison": "model - baseline", "direction": "less",
         "mesi": 0.05, "seeds": 5},
        {"id": "H-NEC", "statement": ("Replacing praemeditatio by its mean raises calibration error more than merging the two "
                                      "subspaces of the partition does."),
         "metric": "calibration_error", "split": "heldout",
         "comparison": "(praemeditatio:mean - full) - (partition:merged - full)",
         "knockouts": ["praemeditatio:mean", "partition:merged"], "direction": "greater", "mesi": 0.05, "seeds": 5},
        {"id": "H-BLIND", "statement": ("On rare real catastrophes, the Senecan gate leaves a larger share of the harm unendorsed "
                                        "than a precautionary gate."),
         "condition": "held-out impressions in the catastrophe tail (always real)",
         "grounding": "Letter 13 counsels against suffering imagined harm; Annals 15.60-64 records the order that did come.",
         "metric": "tail_miss", "split": "heldout", "comparison": "model - blind_baseline", "direction": "greater",
         "mesi": 0.1, "seeds": 5},
        {"id": "H-RIVAL", "statement": ("Over all held-out impressions, the gate fed by calibrated praemeditatio keeps suffering "
                                        "closer to warranted harm than the Epictetus gate without imagination."),
         "metric": "calibration_error", "split": "heldout", "comparison": "model - rival", "direction": "less",
         "mesi": 0.05, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.3, "margin_over_trivial": 0.3, "shuffled_ratio_min": 0.9,
                   "gradcheck_rel_error": 1e-5, "gradcheck_floor": 1e-3, "invariance_tol": 1e-9,
                   "negative_control_min_violation": 1e-6},
    "metrics": {"calibration_error": "mean absolute gap between experienced suffering (assent x imagined harm) and warranted harm (realness x harm)",
                "tail_miss": "over catastrophes, mean share of the warranted harm not matched by experienced suffering",
                "trivial_baseline": "a constant suffering equal to the mean warranted harm of the training impressions",
                "shuffled_band": ("one-sided: a model trained with targets shuffled across impressions must keep its held-out "
                                  "calibration error at least 0.9 times the trivial error")},
    "training": {"optimizer": "Adam", "lr_grid": [0.02], "clip_norm": 5.0, "model_selection": "none: final parameters",
                 "updates": {"full": 600, "quick": 200}, "schedule": "cosine decay to 5 per cent",
                 "loss_weights": {"judgment": 1.0, "praemeditatio": 1.0, "suffering": 1.0, "epoche": 0.02},
                 "applies_to": "Senecan model, unanchored baseline, precautionary gate and Epictetus rival"},
    "task": {"latent": {"controllable": 4, "fortune": 4}, "impression_dimensions": 10, "hidden": 16, "subspace": 4,
             "catastrophe_share": 0.02, "catastrophe_factor": 6, "cue_noise_sd": 1.0,
             "impressions": {"train": 4000, "heldout": 4000, "shifted": 4000}, "shift": 0.6},
    "probe_predictions": [{"probe": "P8", "expected": "equal to baseline"}],
    "probe_support": "vector_classification through the encoder, partition and judgment head",
    "dialectic_links": [{"chapter": 125, "relation": "rival", "test": "H-RIVAL"}],
    "corpus_neighbors": [
        {"chapter": 125, "similarity": None, "difference": "0125 gates by control alone; here an accurate imagination of harm feeds the gate."},
        {"chapter": 81, "similarity": None,
         "difference": "0081 assents only to kataleptic impressions; here assent is graded against imagined harm."},
        {"chapter": 92, "similarity": None,
         "difference": "0092 models impression, assent and impulse; here the object is calibration of imagined harm."},
        {"chapter": 131, "similarity": None,
         "difference": "0131 restores a tranquil fixed point; here the target is suffering matched to real harm."},
        {"chapter": 82, "similarity": None, "difference": "0082 models tension and glad assent to providence; unrelated mechanism."},
    ],
    "similarity_note": "Nearest-neighbour similarity not computed into the card; the audit reports it for the files at hand.",
    "barometer": {
        "consciousness": ["assent as metacognitive endorsement of impressions"],
        "emotional_intelligence": ["distress calibrated to real rather than imagined harm"],
        "world_modeling": ["accurate imagination of possible harm"],
        "cognitive_processing": [], "embodied_cognition": [], "language_understanding": [], "creativity": [], "autonomy": []},
    "task_types": ["vector_classification"],
    "applications": [
        {"use": "autonomous systems that rehearse hazards with calibrated severity estimates before committing to alarms",
         "sector": "robotics and autonomous vehicles", "dataset": "nuScenes", "readiness": "low"},
        {"use": "alert triage that separates rehearsed, merely possible threats from signals that a threat is present",
         "sector": "security operations", "dataset": "CIC-IDS2017", "readiness": "low"},
    ],
    "safety_notes": ("Harms are abstract magnitudes and assent endorses distress, never an action. Nothing in this file models "
                     "death, dying or self-harm as an action or an outcome; Seneca's forced death appears only as history in D4."),
}

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

D_IN, HIDDEN, SUB, TAIL_Z, TAIL_FACTOR = 10, 16, 4, 2.054, 6.0
SIZES = {"train": 4000, "heldout": 4000, "shifted": 4000}
UPDATES = {"full": 600, "quick": 200}
LR, CLIP_NORM, SHIFT = 0.02, 5.0, 0.6
KINDS = {"seneca": dict(lam_p=1.0, epoche=0.02, under=1.0, imagine=True),
         "unanchored": dict(lam_p=0.0, epoche=0.02, under=1.0, imagine=True),
         "precautionary": dict(lam_p=1.0, epoche=0.0, under=3.0, imagine=True),
         "epictetus": dict(lam_p=0.0, epoche=0.02, under=1.0, imagine=False)}
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


# ---------------------------------------------------------------- the world: impressions, harms, realness, a weak cue
def laws(rng):
    """Fixed per seed and shared by every split: how latents mix into impressions and how harm arises."""
    u = rng.normal(size=4)
    return {"mix": rng.normal(0.0, 0.5, (D_IN, 8)), "w": rng.normal(size=4), "u": u / np.linalg.norm(u), "v": 0.8 * rng.normal(size=4)}


def impressions(rng, law, count, shift=0.0):
    c, f = rng.normal(size=(count, 4)), rng.normal(size=(count, 4)) + shift * law["u"]
    along = f @ law["u"]
    tail = along > TAIL_Z
    harm = np.logaddexp(0.0, along) * np.where(tail, TAIL_FACTOR, 1.0)
    real = (rng.random(count) < 1.0 / (1.0 + np.exp(1.4 - f @ law["v"]))) | tail
    return {"x": np.hstack([c, f]) @ law["mix"].T + 0.1 * rng.normal(size=(count, D_IN)),
            "y": (c @ law["w"] + 0.3 * rng.normal(size=count) > 0).astype(float), "harm": harm, "real": real.astype(float),
            "cue": real + rng.normal(size=count), "tail": tail}


def make_splits(seed):
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    law = laws(rng)
    return law, {name: impressions(rng, law, size, SHIFT if name == "shifted" else 0.0) for name, size in SIZES.items()}


# ---------------------------------------------------------------- five organs: encoder, partition, praemeditatio, gate, objective
def softplus_(z):
    return np.logaddexp(0.0, z)


def forward(model, batch):
    P, cfg, ko = model["params"], KINDS[model["kind"]], model["ko"]
    h = np.tanh(batch["x"] @ P["W"].T + P["b"])
    Pc, Pf = P["Pc"], P.get("Pf", P["Pc"])
    if ko.get("partition") == "merged":
        Pc = Pf = 0.5 * (P["Pc"] + P.get("Pf", P["Pc"]))
    zc, zf = h @ Pc.T, h @ Pf.T
    judged = 1.0 / (1.0 + np.exp(-(zc @ P["wj"] + P["bj"][0])))
    if cfg["imagine"]:
        pre = zf @ P["wp"] + P["bp"][0]
        imagined = softplus_(pre)
        if ko.get("praemeditatio") == "mean":
            imagined = np.full_like(imagined, model["imagined_mean"])
        gate_in = P["g"][0] + P["g"][1] * imagined + P["g"][2] * batch["cue"]
    else:
        pre = np.full(batch["cue"].shape, P["bp"][0])
        imagined = softplus_(pre)
        gate_in = P["g"][0] + P["g"][1] * batch["cue"]
    assent = 1.0 / (1.0 + np.exp(-gate_in))
    return {"h": h, "zc": zc, "zf": zf, "Pc": Pc, "Pf": Pf, "judged": judged, "pre": pre, "imagined": imagined,
            "assent": assent, "suffering": assent * imagined, "warranted": batch["real"] * batch["harm"]}


def loss_and_grads(model, batch):
    """Judgment cross-entropy + praemeditatio accuracy + weighted suffering gap + gentle epoche, with hand gradients."""
    P, cfg, s = model["params"], KINDS[model["kind"]], forward(model, batch)
    n, eps = batch["y"].size, 1e-12
    gap = s["suffering"] - s["warranted"]
    weight = np.where(gap < 0.0, cfg["under"], 1.0)
    loss = (-np.mean(batch["y"] * np.log(s["judged"] + eps) + (1 - batch["y"]) * np.log(1 - s["judged"] + eps))
            + cfg["lam_p"] * np.mean((s["imagined"] - batch["harm"]) ** 2) + np.mean(weight * gap ** 2) + cfg["epoche"] * np.mean(s["assent"]))
    d_judge = (s["judged"] - batch["y"]) / n
    d_suffer = 2.0 * weight * gap / n
    d_gate = (d_suffer * s["imagined"] + cfg["epoche"] / n) * s["assent"] * (1.0 - s["assent"])
    if ACTIVE_MUTANT == "zero_gate_gradient":
        d_gate = np.zeros_like(d_gate)
    g = {"wj": s["zc"].T @ d_judge, "bj": np.array([d_judge.sum()])}
    d_zc = np.outer(d_judge, P["wj"])
    if cfg["imagine"]:
        g["g"] = np.array([d_gate.sum(), d_gate @ s["imagined"], d_gate @ batch["cue"]])
        d_imag = d_suffer * s["assent"] + d_gate * P["g"][1] + 2.0 * cfg["lam_p"] * (s["imagined"] - batch["harm"]) / n
        d_pre = d_imag / (1.0 + np.exp(-s["pre"]))
        g["wp"], g["bp"] = s["zf"].T @ d_pre, np.array([d_pre.sum()])
        d_zf = np.outer(d_pre, P["wp"])
        g["Pf"] = d_zf.T @ s["h"]
        d_h = d_zc @ s["Pc"] + d_zf @ s["Pf"]
    else:
        g["g"] = np.array([d_gate.sum(), d_gate @ batch["cue"]])
        d_pre = (d_suffer * s["assent"]) / (1.0 + np.exp(-s["pre"]))
        g["bp"] = np.array([d_pre.sum()])
        d_h = d_zc @ s["Pc"]
    g["Pc"] = d_zc.T @ s["h"]
    d_act = d_h if ACTIVE_MUTANT == "dropped_tanh_derivative" else d_h * (1.0 - s["h"] ** 2)
    g["W"], g["b"] = d_act.T @ batch["x"], d_act.sum(axis=0)
    return float(loss), {k: g[k] for k in P}


def scores(model, batch):
    s = forward(model, batch)
    tail = batch["tail"]
    miss = np.maximum(s["warranted"] - s["suffering"], 0.0) / np.maximum(s["warranted"], 1e-12)
    imagined_only = batch["real"] == 0
    return {"calibration_error": float(np.abs(s["suffering"] - s["warranted"]).mean()),
            "tail_miss": float(miss[tail].mean()) if tail.any() else float("nan"),
            "ratio": float(s["suffering"].sum() / s["warranted"].sum()),
            "assent_imagined": float(s["assent"][imagined_only].mean()), "assent_real": float(s["assent"][~imagined_only].mean()),
            "judgment": float(((s["judged"] > 0.5) == (batch["y"] > 0.5)).mean())}


# ---------------------------------------------------------------- model interface
def build_model(in_dim, out_dim, task_type, rng, **cfg):
    """kind seneca (the chapter's five organs), unanchored (no praemeditatio accuracy term), precautionary (heavier
    penalty on under-endorsing real harm, no epoche), epictetus (gate on the cue alone, no imagination stage)."""
    if task_type not in TASK_TYPES:
        raise ValueError("chapter 0117 supports vector_classification only")
    kind = cfg.get("kind", "seneca")
    imagine = KINDS[kind]["imagine"]
    P = {"W": rng.normal(0.0, 1.0 / math.sqrt(in_dim), (HIDDEN, in_dim)), "b": np.zeros(HIDDEN),
         "Pc": rng.normal(0.0, 0.5, (SUB, HIDDEN)), "wj": rng.normal(0.0, 0.5, SUB), "bj": np.zeros(1), "bp": np.zeros(1)}
    if imagine:
        P.update(Pf=rng.normal(0.0, 0.5, (SUB, HIDDEN)), wp=rng.normal(0.0, 0.5, SUB), g=np.array([0.0, 0.5, 0.5]))
    else:
        P["g"] = np.array([0.0, 0.5])
    return {"kind": kind, "params": P, "ko": {}, "history": [], "imagined_mean": 0.0}


def cosine(tick, budget):
    return 0.05 + 0.475 * (1.0 + math.cos(math.pi * tick / budget))


def fit(model, data, budget, rng):
    """Every update sees the whole training set (Adam, cosine decay to 5 per cent); nothing random, so rng is unused."""
    moments, batch = adam_init(model["params"]), data["train"]
    direction = -1.0 if ACTIVE_MUTANT == "sign_flipped_update" else 1.0
    rate = 0.0 if ACTIVE_MUTANT == "zero_learning_rate" else LR
    for tick in range(budget):
        value, grads = loss_and_grads(model, batch)
        if value != value or abs(value) == float("inf"):
            raise FloatingPointError(f"loss became non-finite at update {tick + 1}")
        clipped = clip_global(grads, CLIP_NORM)[0]
        adam_step(model["params"], {k: direction * v for k, v in clipped.items()}, moments, rate * cosine(tick, budget))
        model["history"].append(value)
    model["imagined_mean"] = float(forward(dict(model, ko={}), batch)["imagined"].mean())
    return model["history"]


def predict(model, X):
    """X is an impression batch; returns the controllable judgment (0 or 1) for each impression."""
    return (forward(model, X)["judged"] > 0.5).astype(int)


def hidden_states(model, X):
    s = forward(model, X)
    return {k: s[k] for k in ("zc", "zf", "imagined", "assent", "suffering")}


def modules(model):
    return {"encoder": {"params": ["W", "b"], "role": "tanh encoder of the impression", "signature": False},
            "partition": {"params": ["Pc", "Pf"], "role": "two linear subspaces: controllable and Fortune", "signature": False},
            "praemeditatio": {"params": ["wp", "bp"], "role": "softplus harm-magnitude head on the Fortune subspace", "signature": True},
            "gate": {"params": ["g"], "role": "logistic assent on imagined harm and a weak reality cue", "signature": True},
            "judgment": {"params": ["wj", "bj"], "role": "logistic judgment on the controllable subspace", "signature": False}}


KNOCKOUT_MODES = {"praemeditatio": ("mean",), "partition": ("merged",)}


def knockout(model, name, mode):
    if mode not in KNOCKOUT_MODES.get(name, ()):
        raise ValueError(f"no knockout {name}:{mode}")
    return dict(model, ko=dict(model["ko"], **{name: mode}))


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


MUTANTS = {"sign_flipped_update": ("updates climb the loss", "C3"), "zero_learning_rate": ("parameters never move", "C3"),
           "zero_gate_gradient": ("the assent gate passes no gradient", "C1"),
           "dropped_tanh_derivative": ("the encoder's tanh derivative is omitted", "C1")}


def data_bridge(path, seed, budget):
    """Optional real data: numeric CSV with a header and a 0/1 label in the last column. Only the judgment organ can learn
    from it: harms are zero, so praemeditatio and the gate see nothing. One row in five is held out."""
    if not os.path.isfile(path):
        return f"skipped ({path} is not a readable file)"
    try:
        grid = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    except ValueError as exc:
        return f"skipped (not numeric: {exc})"
    feats = (grid[:, :-1] - grid[:, :-1].mean(axis=0)) / (grid[:, :-1].std(axis=0) + 1e-12)
    label, blank = grid[:, -1], np.zeros(len(grid))
    pack = lambda sel: dict(x=feats[sel], y=label[sel], harm=blank[sel], real=blank[sel], cue=blank[sel], tail=blank[sel] > 1)
    later = np.arange(len(grid)) % 5 == 4
    learner = build_model(feats.shape[1], 2, TASK_TYPES[0], np.random.default_rng(seed))
    fit(learner, {"train": pack(~later)}, budget, None)
    hit = float(np.mean(predict(learner, pack(later)) == label[later]))
    common = max(label[later].mean(), 1 - label[later].mean())
    return f"{os.path.basename(path)}: one row in five held out, judgment accuracy {hit:.4f}, commonest label {common:.4f}"


# ---------------------------------------------------------------- one seed
def warranted(batch):
    return batch["real"] * batch["harm"]


def run_seed(seed, mode):
    law, splits = make_splits(seed)
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed + 1).spawn(len(KINDS))]
    models = {kind: build_model(D_IN, 2, TASK_TYPES[0], streams[i], kind=kind) for i, kind in enumerate(KINDS)}
    held, shifted = splits["heldout"], splits["shifted"]
    untrained = scores(dict(models["seneca"], params={k: v.copy() for k, v in models["seneca"]["params"].items()}), held)
    for model in models.values():
        fit(model, {"train": splits["train"]}, UPDATES[mode], None)
    table = {kind: {name: scores(model, splits[name]) for name in ("heldout", "shifted")} for kind, model in models.items()}
    seneca = models["seneca"]
    base = table["seneca"]["heldout"]["calibration_error"]
    lesion = {f"{name}:{how}": scores(knockout(seneca, name, how), held)["calibration_error"] - base
              for name, how in (("praemeditatio", "mean"), ("partition", "merged"))}
    constant = warranted(splits["train"]).mean()
    return {"law": law, "splits": splits, "models": models, "table": table, "untrained": untrained, "lesions": lesion,
            "trivial": float(np.abs(constant - warranted(held)).mean()),
            "row": {"H-SIG": table["seneca"]["shifted"]["calibration_error"] - table["unanchored"]["shifted"]["calibration_error"],
                    "H-NEC": lesion["praemeditatio:mean"] - lesion["partition:merged"],
                    "H-BLIND": table["seneca"]["heldout"]["tail_miss"] - table["precautionary"]["heldout"]["tail_miss"],
                    "H-RIVAL": base - table["epictetus"]["heldout"]["calibration_error"]}}


# ---------------------------------------------------------------- audits (the nightly examination of the implementation)
def use_mutant(name):
    global ACTIVE_MUTANT
    previous, ACTIVE_MUTANT = ACTIVE_MUTANT, name
    return previous


def rows(batch, keep):
    return {k: v[keep] for k, v in batch.items()}


def largest_error(models, batch, rng, entries, site):
    worst = 0.0
    for model in models:
        table = finite_difference_check(model["params"], loss_and_grads(model, batch)[1], lambda m=model: loss_and_grads(m, batch)[0],
                                        rng, n_entries=entries, floor=MIND_CARD["thresholds"]["gradcheck_floor"])
        worst = max(worst, max(table.values()))
    return worst


def sound_learning(model, site):
    history, limits = model["history"], MIND_CARD["thresholds"]
    drop = 1.0 - float(np.mean(history[-20:])) / history[0]
    error = scores(model, site["splits"]["heldout"])["calibration_error"]
    return drop >= limits["loss_drop_fraction"] and error <= (1.0 - limits["margin_over_trivial"]) * site["trivial"], drop, error


def audit_gradients(site):
    sample, seed = rows(site["splits"]["train"], slice(0, 300)), site["seed"]
    fresh = [build_model(D_IN, 2, TASK_TYPES[0], np.random.default_rng(seed + 40 + i), kind=k) for i, k in enumerate(KINDS)]
    worst = max(largest_error(fresh, sample, np.random.default_rng(seed + 3), 12, site),
                largest_error(list(site["models"].values()), sample, np.random.default_rng(seed + 4), 12, site))
    total = sum(len(m["params"]) for m in site["models"].values())
    site["gradcheck"] = {"tensors_checked": total, "tensors_total": total, "max_rel_error": worst,
                         "checked_at": ["init", "after_training_steps"], "passed": bool(worst <= MIND_CARD["thresholds"]["gradcheck_rel_error"])}
    return ("C1", "gradient_check", site["gradcheck"]["passed"],
            f"largest relative error {worst:.2e} over every tensor of the four models, before and after training")


def audit_determinism(site):
    runs = []
    for attempt in (1, 2):
        twin = build_model(D_IN, 2, TASK_TYPES[0], np.random.default_rng(site["seed"] + 9))
        fit(twin, {"train": site["splits"]["train"]}, 20, None)
        runs.append(twin)
    alike = runs[0]["history"] == runs[1]["history"] and all(np.array_equal(runs[0]["params"][k], v) for k, v in runs[1]["params"].items())
    sane = all(np.isfinite(v).all() for v in runs[1]["params"].values())
    return "C2", "determinism_finiteness", alike and sane, f"same seed, same twenty updates, same numbers: {alike}; no NaN or infinity: {sane}"


def audit_learning(site):
    ok, drop, error = sound_learning(site["models"]["seneca"], site)
    return "C3", "learning", ok, (f"loss fell by {drop:.3f} (needs 0.3); held-out calibration error {error:.3f} against {site['trivial']:.3f} "
                                  f"for a constant suffering (ratio at most 0.70)")


def audit_shuffled(site):
    train = site["splits"]["train"]
    order = np.random.default_rng(site["seed"] + 11).permutation(train["y"].size)
    mixed = dict(train, **{k: train[k][order] for k in ("y", "harm", "real", "tail")})
    model = build_model(D_IN, 2, TASK_TYPES[0], np.random.default_rng(site["seed"] + 12))
    fit(model, {"train": mixed}, site["updates"], None)
    error = scores(model, site["splits"]["heldout"])["calibration_error"]
    bound = MIND_CARD["thresholds"]["shuffled_ratio_min"] * site["trivial"]
    return ("C4", "shuffled_target_control", error >= bound,
            f"trained on targets shuffled across impressions, held-out calibration error {error:.3f} (must be at least {bound:.3f})")


def replay(site):
    try:
        model = build_model(D_IN, 2, TASK_TYPES[0], np.random.default_rng(site["seed"] + 2))
        clean = largest_error([model], rows(site["splits"]["train"], slice(0, 300)), np.random.default_rng(site["seed"]), 4, site)
        fit(model, {"train": site["splits"]["train"]}, site["updates"], None)
        return clean <= MIND_CARD["thresholds"]["gradcheck_rel_error"] and sound_learning(model, site)[0]
    except FloatingPointError:
        return False


def audit_mutants(site):
    control = replay(site)
    for name in MUTANTS:
        previous = use_mutant(name)
        site["caught"][name] = not replay(site)
        use_mutant(previous)
    found = sum(site["caught"].values())
    return ("C5", "mutant_detection", control and found == len(MUTANTS),
            f"clean replay passes C1 and C3: {control}; mutants caught {found} of {len(MUTANTS)}")


def rotated(model, subspace, compensate, rng):
    Q = np.linalg.qr(rng.normal(size=(SUB, SUB)))[0]
    P = {k: v.copy() for k, v in model["params"].items()}
    head = {"Pc": "wj", "Pf": "wp"}[subspace]
    P[subspace] = Q @ P[subspace]
    if compensate:
        P[head] = Q @ P[head]
    return dict(model, params=P)


def invariance(site, subspace, code, name):
    model, held = site["models"]["seneca"], site["splits"]["heldout"]
    rng = np.random.default_rng(site["seed"] + 21)
    base = forward(model, held)
    kept = forward(rotated(model, subspace, True, rng), held)
    broken = forward(rotated(model, subspace, False, np.random.default_rng(site["seed"] + 21)), held)
    gap = max(float(np.abs(base[k] - kept[k]).max()) for k in ("judged", "imagined", "assent", "suffering"))
    control = max(float(np.abs(base[k] - broken[k]).max()) for k in ("judged", "imagined", "suffering"))
    limits = MIND_CARD["thresholds"]
    ok = gap <= limits["invariance_tol"] and control >= limits["negative_control_min_violation"]
    return (code, name, ok,
            f"rotating the {subspace[1:]} subspace with its read-out changes outputs by {gap:.1e}; rotation without the read-out {control:.1e}")


def audit_closed_gate(site):
    model = site["models"]["seneca"]
    shut = dict(model, params=dict(model["params"], g=np.array([-60.0, 0.0, 0.0])))
    most = float(hidden_states(shut, site["splits"]["heldout"])["suffering"].max())
    return ("C6.3", "unendorsed_impressions_definition", most <= 1e-12,
            f"an impression given no assent causes no suffering: largest value {most:.1e} (definition check)")


def audit_splits(site):
    digest = {k: {hashlib.sha256(r.tobytes()).hexdigest() for r in v["x"]} for k, v in site["splits"].items()}
    disjoint = not (digest["train"] & digest["heldout"] or digest["train"] & digest["shifted"] or digest["heldout"] & digest["shifted"])
    heavier = site["splits"]["shifted"]["tail"].mean() > site["splits"]["train"]["tail"].mean()
    return ("C7", "split_integrity", disjoint and heavier,
            f"no impression shared between splits: {disjoint}; shifted split has more catastrophes: {heavier}")


AUDITS = (audit_gradients, audit_determinism, audit_learning, audit_shuffled, audit_mutants,
          lambda s: invariance(s, "Pc", "C6.1", "controllable_rotation_invariance"),
          lambda s: invariance(s, "Pf", "C6.2", "fortune_rotation_invariance"), audit_closed_gate, audit_splits)


# ---------------------------------------------------------------- verdicts, report, command line
HYPOTHESIS_FIELDS = ("id", "metric", "mean_diff", "ci95", "mesi", "n_seeds", "verdict")


def conclude(runs, seed, evaluated):
    """Bootstrap each pre-registered difference over seeds; knockout changes are summarised the same way."""
    draw = np.random.default_rng(seed + 9973)
    summary = lambda xs: paired_bootstrap(np.asarray(xs, float), draw) if evaluated else (float(np.mean(xs)), None)
    decided = []
    for spec in MIND_CARD["hypotheses"]:
        centre, ci = summary([one["row"][spec["id"]] for one in runs])
        call = verdict(centre, ci, spec["mesi"], spec["direction"]) if evaluated else "not evaluated"
        decided.append(dict(zip(HYPOTHESIS_FIELDS, (spec["id"], spec["metric"], centre, ci, spec["mesi"], len(runs), call))))
    roles = modules(runs[0]["models"]["seneca"])
    ablations = []
    for label in runs[0]["lesions"]:
        centre, ci = summary([one["lesions"][label] for one in runs])
        organ, how = label.split(":")
        ablations.append(dict(module=organ, mode=how, signature=roles[organ]["signature"], metric_change=centre, ci95=ci))
    return decided, ablations


def interval_text(ci):
    return "not evaluated" if ci is None else "[" + ", ".join(f"{v:+.4f}" for v in ci) + "]"


def tabulate(runs, split, measures):
    """Seed-mean table rows: one line per model kind, measures joined with slashes."""
    out = []
    for kind in runs[0]["table"]:
        cells = (np.mean([one["table"][kind][split][m] for one in runs]) for m in measures)
        out.append(f"  {kind:<14} " + " / ".join(f"{c:.3f}" for c in cells))
    return out


def compose_report(mode, seeds, runs, site, results, decided, ablations, bridge, elapsed, exit_code):
    grad, caught, first = site["gradcheck"], site["caught"], runs[0]
    avg = lambda key: float(np.mean([one["untrained"][key] for one in runs]))
    measures = ("calibration_error", "tail_miss", "ratio", "assent_imagined", "assent_real", "judgment")
    found = sum(caught.values())
    page = ["=== VERIFIED REPORT · chapter 0117 ===",
            "file: %s · card_revision %d · mode %s · mutant %s" % (os.path.basename(__file__), MIND_CARD["card_revision"], mode, ACTIVE_MUTANT),
            "environment: python %s · numpy %s" % (sys.version.split()[0], np.__version__),
            "seeds: %s · runtime_s %.1f · budget_s %.0f" % (seeds, elapsed, TIME_BUDGET[mode]),
            "n_params: " + " · ".join("%s %d" % (kind, n_params(m)) for kind, m in first["models"].items()),
            "gradcheck: %d/%d tensors at init and after training · max_rel_error %.2e · passed %s"
            % (grad["tensors_checked"], grad["tensors_total"], grad["max_rel_error"], grad["passed"]), "correctness:"]
    for code, name, ok, note in results:
        page.append("  %-5s %-34s %s  %s" % (code, name, "PASS" if ok else "FAIL", note))
    page.append("mutants: %d/%d detected · score %.2f · %s" % (found, len(MUTANTS), found / len(MUTANTS),
                ", ".join("%s %s" % (name, "caught" if caught.get(name) else "missed") for name in MUTANTS)))
    page.append("hypotheses (paired over seeds; 95% percentile bootstrap of the mean, 2000 resamples):")
    for h in decided:
        page.append("  %-8s mean_diff %+.4f ci95 %s mesi %s seeds %d -> %s"
                    % (h["id"], h["mean_diff"], interval_text(h["ci95"]), h["mesi"], h["n_seeds"], h["verdict"]))
    page.append("knockouts (Senecan model, held-out impressions; change in calibration error):")
    for a in ablations:
        page.append("  %-14s %-7s signature %-5s %+.4f ci95 %s"
                    % (a["module"], a["mode"], a["signature"], a["metric_change"], interval_text(a["ci95"])))
    page.append("held-out impressions (seed mean; calibration error / tail miss / suffering ratio / assent imagined / assent real / judgment):")
    page.extend(tabulate(runs, "heldout", measures))
    page.append("shifted impressions (seed mean; calibration error / tail miss / suffering ratio):")
    page.extend(tabulate(runs, "shifted", measures[:3]))
    page.append("Senecan model before training (seed mean): suffering ratio %.3f · assent imagined %.3f · assent real %.3f"
                % (avg("ratio"), avg("assent_imagined"), avg("assent_real")))
    page.append("constant-suffering calibration error (seed mean): %.3f" % np.mean([one["trivial"] for one in runs]))
    return page + ["real-data bridge: " + bridge, "task_types: " + ", ".join(TASK_TYPES), "exit_code: %d" % exit_code, "=== END REPORT ==="]


def appendix_b(runs, site, results, decided, ablations, seeds, elapsed, exit_code):
    found = sum(site["caught"].values())
    pairs = [("schema_version", "1.0"), ("chapter", 117), ("file", os.path.basename(__file__)), ("card_revision", MIND_CARD["card_revision"]),
             ("environment", {"python": sys.version.split()[0], "numpy": np.__version__}), ("seeds", seeds), ("runtime_s", round(elapsed, 2)),
             ("n_params", n_params(runs[0]["models"]["seneca"])), ("gradcheck", site["gradcheck"]),
             ("correctness", [{"id": c, "name": n, "passed": bool(ok), "detail": d} for c, n, ok, d in results]),
             ("mutants", {"detected": found, "total": len(MUTANTS), "score": found / len(MUTANTS)}),
             ("hypotheses", decided), ("knockouts", ablations), ("task_types", TASK_TYPES), ("exit_code", exit_code)]
    return dict(pairs)


def gather(mode, seeds):
    clock, runs = time.time(), []
    print("chapter 0117 · mode %s · seeds %s · mutant %s" % (mode, seeds, ACTIVE_MUTANT), flush=True)
    for seed in seeds:
        runs.append(run_seed(seed, mode))
        print("  seed %d done (%.1f s)" % (seed, time.time() - clock), flush=True)
    return runs, clock


def protocol(mode, base_seed, n_seeds, json_path, data_path):
    seeds = list(range(base_seed, base_seed + n_seeds))
    runs, clock = gather(mode, seeds)
    site = {**runs[0], "seed": base_seed, "updates": UPDATES[mode], "caught": {}}
    results = [audit(site) for audit in AUDITS]
    decided, ablations = conclude(runs, base_seed, mode == "full" and n_seeds >= 5)
    bridge = data_bridge(data_path, base_seed, UPDATES[mode]) if data_path else "skipped (no --data PATH given)"
    elapsed = time.time() - clock
    results.append(("C8", "budget", elapsed <= TIME_BUDGET[mode], "%.1f s of %.0f s" % (elapsed, TIME_BUDGET[mode])))
    broken = {code for code, _, ok, _ in results if not ok}
    exit_code = 0 if not broken else 3 if broken == {"C8"} else 1
    page = compose_report(mode, seeds, runs, site, results, decided, ablations, bridge, elapsed, exit_code)
    write_report(page, appendix_b(runs, site, results, decided, ablations, seeds, elapsed, exit_code), json_path)
    return exit_code


FLAGS = {"--quick": dict(action="store_true", help="a single seed and fewer updates; all correctness tests still run"),
         "--seed": dict(type=int, default=0, help="first seed (0 unless given)"),
         "--seeds": dict(type=int, help="how many seeds (5, or 1 with --quick)"),
         "--json": dict(metavar="PATH", help="write the report as JSON to PATH as well"),
         "--card": dict(action="store_true", help="show MIND_CARD as JSON and stop"),
         "--mutant": dict(metavar="NAME", help="switch on one mutant from: " + ", ".join(MUTANTS)),
         "--data": dict(metavar="PATH", help="optional numeric CSV with a header row and a 0/1 label last")}


def main(argv=None):
    cli = argparse.ArgumentParser(prog=os.path.basename(__file__), description="Chapter 0117: an assent gate fed by calibrated praemeditatio.")
    for flag, spec in FLAGS.items():
        cli.add_argument(flag, **spec)
    opts = cli.parse_args(argv)
    if opts.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    wanted = opts.seeds if opts.seeds is not None else (1 if opts.quick else 5)
    problem = ("no such mutant; choose from " + ", ".join(MUTANTS)) if opts.mutant not in (None, *MUTANTS) else (
        "--seeds must be 1 or more" if wanted < 1 else "")
    if problem:
        print(problem, file=sys.stderr)
        return 2
    use_mutant(opts.mutant)
    try:
        return protocol("quick" if opts.quick else "full", opts.seed, wanted, opts.json, opts.data)
    except FloatingPointError as exc:
        print(f"non-finite values: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())

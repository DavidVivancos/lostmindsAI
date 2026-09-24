#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0103 · Cicero
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 6, Minds 101-120: https://www.amazon.com/dp/B0HF7G6JJD · Demos: https://artificiology.com/
# END ATTRIBUTION
"""Forum of the probable: a shared bank of topics (loci), an eloquence gain (elocutio), two advocates arguing in
utramque partem, a judge that always returns a graded verdict (the probabile, never suspension), and a binding law
(recta ratio) the judge did not author, against which every verdict is checked and, where it conflicts, revised.

Thesis
    When abstention is not an option, stage the strongest case on both sides, commit to the more probable, grade the
    commitment, and hold it to a standard outside the decider.

Evidence
    D1  Cicero, Academica and Tusculan Disputations: he lives by probabilities and argues both sides of a question
        (in utramque partem); natural law as right reason (De Re Publica 3.33, via Lactantius).
    D2  Sextus Empiricus, Outlines of Pyrrhonism 1.1-3 and 1.226: the Academics (Carneades, Clitomachus) say that
        truth cannot be apprehended; the Pyrrhonist suspends judgment instead. Cicero's Academica is a main source for
        the Academic position (SEP, Ancient Skepticism).
    D3  Carneades' persuasive or probable impression (pithane phantasia) as a criterion for action (Obdrzalek 2006).

Doctrine -> mechanism -> test (IDs as in MIND_CARD)
    D1     M1 loci, M2 elocutio, M3 advocates, M4 judge (graded, no suspension)   C6.1 C6.2 C6.3 H-SIG H-NEC
    D1     M5 recta ratio: a fixed law vector; conflicting verdicts are revised   H-NEC (matched knockout)
    D2 D3  rival: Pyrrhonist suspension on near-balanced cases                     H-RIVAL
    D1     blind spot: an advocate with more eloquence than the other              H-BLIND

Research question (calibration, abstention and metacognition)
    When a decision must be made, does adversarial argument resolved to a graded verdict and checked against an
    external law calibrate better than one-sided argument or suspension, and does unequal eloquence corrupt it?

Closest prior art and the delta
    AI safety via debate (Irving, Christiano and Amodei 2018); calibration of probabilistic classifiers (Guo et al. 2017);
    selective prediction with abstention. Delta: a small trainable debate whose advocates optimise opposite objectives
    over a shared topic bank, a judge that may not abstain, a fixed external law that revises conflicting verdicts,
    tests against one-sided argument and Pyrrhonist suspension, and an eloquence-asymmetry stress test.

Blind spot
    Eloquence can outrun truth: when one advocate argues with more force, verdicts track rhetoric rather than evidence.

Task (generative process)
    Eight evidence features per case, drawn standard normal (shifted split: mean 0.4, s.d. 1.3). The case is true with
    probability sigmoid(w.e) for a hidden world vector w of norm 2.2; the law vector r is w's direction plus noise 0.35,
    renormalised. Six loci. Splits: 2000 training, 2000 held-out and 2000 shifted cases.

Limits
    Binary synthetic cases, linear topics, one law vector. Nothing here reproduces Rome or its courts; historical
    episodes are not modelled. A research prototype of one mechanism, not an AGI and not Cicero's mind.
"""

MIND_CARD = {
    "schema_version": "1.0", "card_revision": 1, "revision_log": [],
    "generation": {"template_version": "codeguidelines 1.0 (15 September 2026), Appendix A", "generator": "Claude (Anthropic)",
                   "generator_version": "claude-opus-5", "date": "2026-09-16"},
    "id": 103, "figure": "Cicero", "born": -106, "died": -43, "civilization": "Roman", "provenance": "belief",
    "thesis": ("When abstention is not an option, stage the strongest case on both sides, commit to the more probable, grade the "
               "commitment, and hold it to a standard outside the decider."),
    "evidence": [
        {"id": "D1", "basis": "primary", "source": "Cicero, Academica; Tusculan Disputations; De Re Publica 3.33 (via Lactantius)",
         "claim": "He lives by probabilities, argues both sides of a question, and names natural law right reason."},
        {"id": "D2", "basis": "primary", "source": "Sextus Empiricus, Outlines of Pyrrhonism 1.1-3, 1.226; SEP, Ancient Skepticism",
         "claim": "Academics hold truth inapprehensible; Pyrrhonists suspend judgment; Cicero's Academica is a main source for the Academy."},
        {"id": "D3", "basis": "scholarship", "source": "S. Obdrzalek, Oxford Studies in Ancient Philosophy 31 (2006)",
         "claim": "Carneades proposed the probable impression as a criterion for life and action."},
    ],
    "research_question": {"category": "calibration, abstention and metacognition",
                          "question": ("When a decision must be made, does adversarial argument resolved to a graded verdict and checked against an "
                                       "external law calibrate better than one-sided argument or suspension, and does unequal eloquence corrupt it?")},
    "mechanism": {
        "name": "forum of the probable", "family": "two softmax topic-selecting advocates over a shared linear topic bank, logistic judge, fixed law check",
        "signature_modules": ["advocates", "judge"],
        "closest_prior_art": ["AI safety via debate (Irving, Christiano and Amodei 2018)", "calibration of classifiers (Guo et al. 2017)",
                              "selective prediction with abstention"],
        "overlap": "Medium", "prior_art_queries": [],
        "prior_art_note": "No literature search was run for this card; overlap is rated against the named methods.",
        "contribution_type": "test",
        "delta": ("Advocates optimising opposite objectives over shared loci, a judge that may not abstain, a fixed external law revising "
                  "conflicting verdicts, tests against one-sided argument and Pyrrhonist suspension, and an eloquence-asymmetry stress test."),
        "baselines": {"baseline": "one-sided forum: the same network trained and evaluated with the opposing advocate silent",
                      "blind_baseline": "the same trained forum heard with equal eloquence on both sides",
                      "rival": ("chapter 0139 Sextus Empiricus, minimal: the same verdicts, but suspension (probability 0.5) wherever the "
                                "verdict lies within 0.15 of balance")}},
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1-M4", "property_test": "C6.1, C6.2, C6.3", "hypothesis": "H-SIG, H-NEC"},
        {"doctrine": "D1", "mechanism": "M5 recta ratio", "property_test": "none", "hypothesis": "H-NEC"},
        {"doctrine": "D2", "mechanism": "rival suspension", "property_test": "none", "hypothesis": "H-RIVAL"},
        {"doctrine": "D3", "mechanism": "graded probabile", "property_test": "C6.3", "hypothesis": "H-RIVAL"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "On shifted cases, the two-advocate forum has lower Brier score than the one-sided forum.",
         "metric": "brier", "split": "shifted", "comparison": "model - baseline", "direction": "less", "mesi": 0.01, "seeds": 5},
        {"id": "H-NEC", "statement": "Silencing the opposing advocate raises Brier score more than removing the law's revision does.",
         "metric": "brier", "split": "heldout", "comparison": "(adversary:silent - full) - (revision:off - full)",
         "knockouts": ["adversary:silent", "revision:off"], "direction": "greater", "mesi": 0.01, "seeds": 5},
        {"id": "H-BLIND", "statement": "Hearing the prosecution with four times the eloquence of the defence raises Brier score.",
         "condition": "held-out cases, eloquence 2.0 for the pro advocate and 0.5 for the con advocate",
         "grounding": "The chapter's warning that eloquence can outrun truth.",
         "metric": "brier", "split": "heldout", "comparison": "model(asymmetric) - model(equal)", "direction": "greater", "mesi": 0.01, "seeds": 5},
        {"id": "H-RIVAL", "statement": "Committing to the graded probabile gives lower Brier score than suspending near balance.",
         "metric": "brier", "split": "heldout", "comparison": "model - rival", "direction": "less", "mesi": 0.005, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.3, "margin_over_trivial": 0.3, "shuffled_ratio_min": 0.9, "gradcheck_rel_error": 1e-5,
                   "gradcheck_floor": 1e-3, "invariance_tol": 1e-9, "negative_control_min_violation": 1e-6},
    "metrics": {"brier": "mean squared gap between the final verdict probability and the case's truth",
                "accuracy": "share of cases decided on the right side of 0.5", "trivial_baseline": "always deciding the commoner side",
                "shuffled_band": "one-sided: trained on truths shuffled across cases, held-out error at least 0.9 times the trivial error"},
    "training": {"optimizer": "Adam", "lr_grid": [0.03], "clip_norm": 5.0, "model_selection": "none: final parameters",
                 "updates": {"full": 600, "quick": 200}, "schedule": "cosine decay to 5 per cent",
                 "objectives": "judge and loci: log loss on truth; pro advocate: maximise the verdict; con advocate: minimise it",
                 "law_revision": {"blend": 0.5, "law_gain": 2.0}, "rival_band": 0.15, "applies_to": "forum and one-sided forum"},
    "task": {"features": 8, "loci": 6, "world_norm": 2.2, "law_noise": 0.35, "shift": {"mean": 0.4, "sd": 1.3},
             "cases": {"train": 2000, "heldout": 2000, "shifted": 2000}, "eloquence": {"equal": [1.0, 1.0], "asymmetric": [2.0, 0.5]}},
    "probe_predictions": [{"probe": "P8", "expected": "equal to baseline"}],
    "probe_support": "vector_classification through the full forum",
    "dialectic_links": [{"chapter": 139, "relation": "rival", "test": "H-RIVAL"}],
    "corpus_neighbors": [
        {"chapter": 139, "similarity": None, "difference": "0139 suspends where arguments balance; here the forum must commit to a graded verdict."},
        {"chapter": 77, "similarity": None, "difference": "0077 practises abstention; here abstention is ruled out and calibration is measured."},
    ],
    "similarity_note": "Nearest-neighbour similarity not computed into the card; the audit reports it for the files at hand.",
    "barometer": {"language_understanding": ["argument in utramque partem"], "consciousness": ["graded, revisable commitment"],
                  "cognitive_processing": [], "embodied_cognition": [], "world_modeling": [], "emotional_intelligence": [], "creativity": [], "autonomy": []},
    "task_types": ["vector_classification"],
    "applications": [{"use": "debate-style verification of AI answers with a mandatory counter-case", "sector": "AI oversight", "dataset": "QuALITY debate benchmarks", "readiness": "low"},
                     {"use": "evaluating legal and policy arguments against a fixed external standard", "sector": "legal technology", "dataset": "synthetic argument sets", "readiness": "low"}],
    "safety_notes": "Synthetic cases only; no historical trial, execution or exile is modelled, and verdicts concern abstract claims.",
}

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

FEATURES, LOCI, WORLD_NORM, LAW_NOISE = 8, 6, 2.2, 0.35
CASES = {"train": 2000, "heldout": 2000, "shifted": 2000}
UPDATES = {"full": 600, "quick": 200}
LR, CLIP_NORM, BLEND, LAW_GAIN, RIVAL_BAND = 0.03, 5.0, 0.5, 2.0, 0.15
EQUAL, ASYMMETRIC = (1.0, 1.0), (2.0, 0.5)
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

# ================================================================ the cases before the forum
def world(seed):
    gen = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    truth_dir = gen.normal(size=FEATURES)
    truth_dir /= np.linalg.norm(truth_dir)
    law = truth_dir + LAW_NOISE * gen.normal(size=FEATURES)
    docket = {"law": law / np.linalg.norm(law)}
    for part, count in CASES.items():
        centre, spread = (0.4, 1.3) if part == "shifted" else (0.0, 1.0)
        e = centre + spread * gen.normal(size=(count, FEATURES))
        chance = 1.0 / (1.0 + np.exp(-WORLD_NORM * (e @ truth_dir)))
        docket[part] = {"e": e, "y": (gen.random(count) < chance).astype(float)}
    return docket


# ================================================================ the forum: loci, elocutio, advocates, judge, law
def build_model(in_dim, out_dim, task_type, rng, **cfg):
    """A forum. cfg one_sided=True trains and hears it with the opposing advocate silent (the H-SIG baseline)."""
    if task_type not in TASK_TYPES or in_dim != FEATURES:
        raise ValueError("chapter 0103 hears 8-feature cases")
    return {"one_sided": bool(cfg.get("one_sided", False)), "silence": None, "revise": True, "voice": EQUAL, "history": [],
            "params": {"loci": rng.normal(0, 0.4, (LOCI, in_dim)), "pro": rng.normal(0, 0.3, (LOCI, in_dim)),
                       "con": rng.normal(0, 0.3, (LOCI, in_dim)), "alpha": np.array([0.1]), "bias": np.zeros(1)}}


def _choose(scores):
    top = scores.max(axis=1, keepdims=True)
    w = np.exp(scores - top)
    return w / w.sum(axis=1, keepdims=True)


def hear(forum, e):
    P = forum["params"]
    bearing = e @ P["loci"].T
    pick_pro, pick_con = _choose(e @ P["pro"].T), _choose(e @ P["con"].T)
    loud_pro, loud_con = forum["voice"]
    case_pro = loud_pro * (pick_pro * bearing).sum(axis=1)
    silent = forum["one_sided"] or forum["silence"] == "adversary"
    case_con = np.zeros(len(e)) if silent else -loud_con * (pick_con * bearing).sum(axis=1)
    margin = case_pro - case_con
    verdict_p = 1.0 / (1.0 + np.exp(-(P["alpha"][0] * margin + P["bias"][0])))
    return {"bearing": bearing, "pick_pro": pick_pro, "pick_con": pick_con, "margin": margin, "p": verdict_p, "silent": silent}


def checked_by_law(forum, e, p, law):
    """Recta ratio: a verdict on the other side of the law from the case is pulled halfway toward the law's own reading."""
    if not forum["revise"]:
        return p
    reading = e @ law
    against = (p > 0.5) != (reading > 0)
    return np.where(against, (1.0 - BLEND) * p + BLEND / (1.0 + np.exp(-LAW_GAIN * reading)), p)


def _softmax_back(pick, upstream):
    if ACTIVE_MUTANT == "dropped_softmax_centering":
        return pick * upstream
    return pick * (upstream - (pick * upstream).sum(axis=1, keepdims=True))


def party_losses(forum, batch):
    """Three objectives, one per party: the judge (with the loci) seeks truth, pro raises the verdict, con lowers it."""
    P, h, n = forum["params"], hear(forum, batch["e"]), len(batch["y"])
    p, e, loud_pro, loud_con = h["p"], batch["e"], forum["voice"][0], forum["voice"][1]
    eps = 1e-12
    out = {"judge": -float(np.mean(batch["y"] * np.log(p + eps) + (1 - batch["y"]) * np.log(1 - p + eps))),
           "pro": -float(np.mean(np.log(p + eps))), "con": -float(np.mean(np.log(1 - p + eps)))}
    grads = {}
    d_logit = (p - batch["y"]) / n
    d_margin = d_logit * P["alpha"][0]
    d_bearing = d_margin[:, None] * loud_pro * h["pick_pro"]
    if not h["silent"]:
        d_bearing = d_bearing + d_margin[:, None] * loud_con * h["pick_con"]
    grads["loci"] = np.zeros_like(P["loci"]) if ACTIVE_MUTANT == "zero_loci_gradient" else d_bearing.T @ e
    grads["alpha"], grads["bias"] = np.array([float(d_logit @ h["margin"])]), np.array([float(d_logit.sum())])
    up_pro = (((p - 1.0) / n) * P["alpha"][0])[:, None] * loud_pro * h["bearing"]
    grads["pro"] = _softmax_back(h["pick_pro"], up_pro).T @ e
    if h["silent"]:
        grads["con"] = np.zeros_like(P["con"])
    else:
        up_con = ((p / n) * P["alpha"][0])[:, None] * loud_con * h["bearing"]
        grads["con"] = _softmax_back(h["pick_con"], up_con).T @ e
    return out, grads


OWNER = {"loci": "judge", "alpha": "judge", "bias": "judge", "pro": "pro", "con": "con"}


def loss_and_grads(model, batch):
    """Interface form: the judge's loss, and every tensor's gradient taken from the objective of the party that owns it."""
    losses, grads = party_losses(model, batch)
    return losses["judge"], grads


def fit(model, data, budget, rng):
    """All three parties move at once, each down its own objective; cosine decay to 5 per cent. Nothing sampled: rng unused."""
    moments, flip = adam_init(model["params"]), -1.0 if ACTIVE_MUTANT == "sign_flipped_update" else 1.0
    tick = 0
    while tick < budget:
        losses, grads = party_losses(model, data["train"])
        if not all(math.isfinite(v) for v in losses.values()):
            raise FloatingPointError("a party's loss diverged at update %d" % (tick + 1))
        grads = clip_global(grads, CLIP_NORM)[0]
        pace = 0.0 if ACTIVE_MUTANT == "zero_learning_rate" else LR * (0.05 + 0.475 * (1.0 + math.cos(math.pi * tick / budget)))
        adam_step(model["params"], {k: flip * g for k, g in grads.items()}, moments, pace)
        model["history"].append(losses["judge"])
        tick += 1
    return model["history"]


def predict(model, X):
    return (hear(model, X)["p"] > 0.5).astype(int)


def hidden_states(model, X):
    h = hear(model, X)
    return {"bearing": h["bearing"], "pro_topics": h["pick_pro"], "con_topics": h["pick_con"], "margin": h["margin"], "verdict": h["p"]}


def modules(model):
    return {"loci": {"params": ["loci"], "role": "shared linear topic bank", "signature": False},
            "advocates": {"params": ["pro", "con"], "role": "softmax topic choice optimising opposite objectives", "signature": True},
            "judge": {"params": ["alpha", "bias"], "role": "logistic verdict that never abstains", "signature": True},
            "law": {"params": [], "role": "fixed external law vector revising conflicting verdicts", "signature": False}}


def knockout(model, name, mode):
    twin = dict(model, params=model["params"])
    if (name, mode) == ("adversary", "silent"):
        twin["silence"] = "adversary"
    elif (name, mode) == ("revision", "off"):
        twin["revise"] = False
    else:
        raise ValueError("no knockout %s:%s" % (name, mode))
    return twin


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


MUTANTS = {"sign_flipped_update": ("every party climbs its objective", "C3"), "zero_learning_rate": ("nobody moves", "C3"),
           "zero_loci_gradient": ("the topic bank receives no gradient", "C1"), "dropped_softmax_centering": ("topic choice loses its centring term", "C1")}


def data_bridge(path, seed, budget):
    """Optional real data: CSV with a header, 8 numeric columns and a 0/1 truth last; every fifth row held out."""
    try:
        rows = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    except (OSError, ValueError) as exc:
        return "skipped (" + type(exc).__name__ + ")"
    if rows.shape[1] != FEATURES + 1:
        return "skipped (needs 8 feature columns)"
    late = np.arange(len(rows)) % 5 == 4
    forum = build_model(FEATURES, 1, TASK_TYPES[0], np.random.default_rng(seed))
    fit(forum, {"train": {"e": rows[~late, :-1], "y": rows[~late, -1]}}, budget, None)
    return os.path.basename(path) + ": held-out accuracy " + "%.4f" % np.mean(predict(forum, rows[late, :-1]) == rows[late, -1])


# ================================================================ one seed of hearings
def brier(p, y):
    return float(np.mean((p - y) ** 2))


def suspend(p):
    return np.where(np.abs(p - 0.5) < RIVAL_BAND, 0.5, p)


def run_seed(seed, mode):
    docket = world(seed)
    law, held, shifted = docket["law"], docket["heldout"], docket["shifted"]
    forum = build_model(FEATURES, 1, TASK_TYPES[0], np.random.default_rng(seed + 1))
    lone = build_model(FEATURES, 1, TASK_TYPES[0], np.random.default_rng(seed + 2), one_sided=True)
    fit(forum, docket, UPDATES[mode], None)
    fit(lone, docket, UPDATES[mode], None)
    final = lambda f, part: checked_by_law(f, part["e"], hear(f, part["e"])["p"], law)
    full_h, full_s = final(forum, held), final(forum, shifted)
    loud = dict(forum, voice=ASYMMETRIC)
    scores = {"forum_held": brier(full_h, held["y"]), "forum_shift": brier(full_s, shifted["y"]),
              "lone_shift": brier(final(lone, shifted), shifted["y"]), "rival": brier(suspend(full_h), held["y"]),
              "asym": brier(final(loud, held), held["y"]), "silent": brier(final(knockout(forum, "adversary", "silent"), held), held["y"]),
              "unrevised": brier(final(knockout(forum, "revision", "off"), held), held["y"])}
    acc = float(np.mean((full_h > 0.5) == (held["y"] > 0.5)))
    raw = hear(forum, held["e"])["p"]
    revised_share = float(np.mean(((raw > 0.5) != ((held["e"] @ law) > 0))))
    suspended = float(np.mean(np.abs(full_h - 0.5) < RIVAL_BAND))
    return {"docket": docket, "forum": forum, "lone": lone, "scores": scores, "acc": acc, "revised": revised_share, "suspended": suspended,
            "trivial": float(min(held["y"].mean(), 1 - held["y"].mean())),
            "lesions": {"adversary:silent": scores["silent"] - scores["forum_held"], "revision:off": scores["unrevised"] - scores["forum_held"]},
            "row": {"H-SIG": scores["forum_shift"] - scores["lone_shift"],
                    "H-NEC": (scores["silent"] - scores["forum_held"]) - (scores["unrevised"] - scores["forum_held"]),
                    "H-BLIND": scores["asym"] - scores["forum_held"], "H-RIVAL": scores["forum_held"] - scores["rival"]}}

# ================================================================ the court that examines the forum
class Court:
    """Every method named hearing_* is one correctness test; they run in the order written and each returns (passed, note)."""

    def __init__(self, first, mode):
        self.first, self.mode, self.caught, self.gradcheck = first, mode, {}, None
        self.limits = MIND_CARD["thresholds"]

    def docket_rows(self, count=120):
        return {k: v[:count] for k, v in self.first["docket"]["train"].items()}

    def owner_gap(self, forum, rng, entries):
        worst = 0.0
        for party in ("judge", "pro", "con"):
            names = [k for k, v in OWNER.items() if v == party]
            sub = {k: forum["params"][k] for k in names}
            batch = self.docket_rows()
            grads = party_losses(forum, batch)[1]
            table = finite_difference_check(sub, {k: grads[k] for k in names}, lambda f=forum, b=batch, pt=party: party_losses(f, b)[0][pt],
                                            rng, n_entries=entries, floor=self.limits["gradcheck_floor"])
            worst = max(worst, max(table.values()))
        return worst

    def learned(self, forum):
        drop = 1.0 - float(np.mean(forum["history"][-20:])) / forum["history"][0]
        held = self.first["docket"]["heldout"]
        err = float(np.mean(predict(forum, held["e"]) != held["y"]))
        return drop >= self.limits["loss_drop_fraction"] and err <= (1 - self.limits["margin_over_trivial"]) * self.first["trivial"], drop, err

    def hearing_c1_gradient_check(self):
        seed = self.first["seed"]
        young = build_model(FEATURES, 1, TASK_TYPES[0], np.random.default_rng(seed + 70))
        worst = max(self.owner_gap(young, np.random.default_rng(seed + 3), 10), self.owner_gap(self.first["forum"], np.random.default_rng(seed + 4), 10))
        self.gradcheck = {"tensors_checked": 5, "tensors_total": 5, "max_rel_error": worst, "checked_at": ["init", "after_training_steps"],
                          "passed": bool(worst <= self.limits["gradcheck_rel_error"])}
        return self.gradcheck["passed"], "each tensor against its own party's objective: worst relative error " + "%.2e" % worst

    def hearing_c2_determinism_finiteness(self):
        seed_once = self.first["seed"] + 8
        first_pass = build_model(FEATURES, 1, TASK_TYPES[0], np.random.default_rng(seed_once))
        second_pass = build_model(FEATURES, 1, TASK_TYPES[0], np.random.default_rng(seed_once))
        fit(first_pass, self.first["docket"], 15, None)
        fit(second_pass, self.first["docket"], 15, None)
        same_curve = first_pass["history"] == second_pass["history"]
        same_brief = all(np.array_equal(first_pass["params"][k], second_pass["params"][k]) for k in first_pass["params"])
        return bool(same_curve and same_brief), "retrying the forum from one seed repeats curve and parameters: " + str(same_curve and same_brief)

    def hearing_c3_learning(self):
        ok, drop, err = self.learned(self.first["forum"])
        return ok, "judge loss fell " + "%.3f" % drop + " (needs 0.3); held-out error " + "%.3f" % err + " against " + "%.3f" % self.first["trivial"] + " for the commoner side"

    def hearing_c4_shuffled_truth_control(self):
        cases = self.first["docket"]["train"]
        scrambled = np.random.default_rng(self.first["seed"] + 11).permutation(len(cases["y"]))
        blind = build_model(FEATURES, 1, TASK_TYPES[0], np.random.default_rng(self.first["seed"] + 12))
        fit(blind, {"train": {"e": cases["e"], "y": cases["y"][scrambled]}}, UPDATES[self.mode], None)
        heard = self.first["docket"]["heldout"]
        wrong = float(np.mean(predict(blind, heard["e"]) != heard["y"]))
        need = self.limits["shuffled_ratio_min"] * self.first["trivial"]
        return wrong >= need, "a forum taught shuffled truths errs on " + "%.3f" % wrong + " of held-out cases; it must err on at least " + "%.3f" % need

    def retrial(self):
        try:
            forum = build_model(FEATURES, 1, TASK_TYPES[0], np.random.default_rng(self.first["seed"] + 2))
            clean = self.owner_gap(forum, np.random.default_rng(self.first["seed"]), 3)
            fit(forum, self.first["docket"], UPDATES[self.mode], None)
            return bool(clean <= self.limits["gradcheck_rel_error"] and self.learned(forum)[0])
        except FloatingPointError:
            return False

    def hearing_c5_mutant_detection(self):
        honest = self.retrial()
        for name in MUTANTS:
            previous = set_mutant(name)
            self.caught[name] = not self.retrial()
            set_mutant(previous)
        return honest and all(self.caught.values()), "clean retrial passes C1 and C3: " + str(honest) + "; mutants caught " + str(sum(self.caught.values())) + " of " + str(len(MUTANTS))

    def hearing_c6_1_advocate_antisymmetry(self):
        forum, e = self.first["forum"], self.first["docket"]["heldout"]["e"]
        P = forum["params"]
        swapped = dict(forum, params=dict(P, pro=-P["con"], con=-P["pro"]))
        gap = float(np.abs(hear(forum, e)["margin"] + hear(swapped, -e)["margin"]).max())
        naive = dict(forum, params=dict(P, pro=P["con"], con=P["pro"]))
        control = float(np.abs(hear(forum, e)["margin"] + hear(naive, -e)["margin"]).max())
        ok = gap <= self.limits["invariance_tol"] and control >= self.limits["negative_control_min_violation"]
        return ok, "reversing the case and trading advocates' briefs reverses the margin within " + "%.1e" % gap + "; trading without reversal " + "%.1e" % control

    def hearing_c6_2_locus_permutation_invariance(self):
        forum, e = self.first["forum"], self.first["docket"]["heldout"]["e"]
        P, order = forum["params"], np.random.default_rng(self.first["seed"] + 6).permutation(LOCI)
        moved = dict(forum, params=dict(P, loci=P["loci"][order], pro=P["pro"][order], con=P["con"][order]))
        gap = float(np.abs(hear(forum, e)["p"] - hear(moved, e)["p"]).max())
        control = float(np.abs(hear(forum, e)["p"] - hear(dict(forum, params=dict(P, loci=P["loci"][order])), e)["p"]).max())
        ok = gap <= self.limits["invariance_tol"] and control >= self.limits["negative_control_min_violation"]
        return ok, "renumbering the loci everywhere moves verdicts " + "%.1e" % gap + "; renumbering the bank alone " + "%.1e" % control

    def hearing_c6_3_no_suspension_definition(self):
        docket = self.first["docket"]
        p = checked_by_law(self.first["forum"], docket["heldout"]["e"], hear(self.first["forum"], docket["heldout"]["e"])["p"], docket["law"])
        ok = bool(np.all(np.isfinite(p)) and np.all((p > 0) & (p < 1)))
        return ok, "every case receives a graded verdict strictly between 0 and 1 (definition check)"

    def hearing_c7_split_integrity(self):
        d = self.first["docket"]
        prints = {k: {hashlib.sha256(r.tobytes()).hexdigest() for r in d[k]["e"]} for k in CASES}
        apart = not (prints["train"] & prints["heldout"] or prints["train"] & prints["shifted"] or prints["heldout"] & prints["shifted"])
        return apart, "no case heard in two splits: " + str(apart)

    def sit(self):
        minutes = []
        for name, method in vars(type(self)).items():
            if name.startswith("hearing_"):
                code = name.split("_")[1].upper() + ("." + name.split("_")[2] if name.split("_")[2].isdigit() else "")
                label = "_".join(name.split("_")[3:] if name.split("_")[2].isdigit() else name.split("_")[2:])
                passed, note = method(self)
                minutes.append((code, label, bool(passed), note))
        return minutes


def set_mutant(name):
    global ACTIVE_MUTANT
    old, ACTIVE_MUTANT = ACTIVE_MUTANT, name
    return old


# ================================================================ minutes, verdicts and the command line
def four(x):
    return "%+.4f" % x


def span_text(ci):
    return "not evaluated" if ci is None else "[" + four(ci[0]) + ", " + four(ci[1]) + "]"


def weigh(runs, first, evaluated):
    """Hypotheses and lesions share one resampling routine; with fewer than five seeds nothing is decided."""
    draw = np.random.default_rng(first + 9973)

    def settle(key_path):
        sample = np.array([key_path(run) for run in runs])
        return paired_bootstrap(sample, draw) if evaluated else (float(sample.mean()), None)
    registered = MIND_CARD["hypotheses"]
    settled = [(spec, settle(lambda run, i=spec["id"]: run["row"][i])) for spec in registered]
    hyps = [dict(id=spec["id"], metric=spec["metric"], mean_diff=mc[0], ci95=mc[1], mesi=spec["mesi"], n_seeds=len(runs),
                 verdict=(verdict(mc[0], mc[1], spec["mesi"], spec["direction"]) if evaluated else "not evaluated")) for spec, mc in settled]
    lesion_keys = list(runs[0]["lesions"])
    lesions = []
    for label, mc in zip(lesion_keys, [settle(lambda run, k=k: run["lesions"][k]) for k in lesion_keys]):
        part, how = label.split(":")
        lesions.append(dict(module=part, mode=how, signature=(part == "adversary"), metric_change=mc[0], ci95=mc[1]))
    return hyps, lesions


def minutes_text(mode, seeds, runs, court, sat, hyps, lesions, bridge, took, code):
    mean = lambda f: float(np.mean([f(r) for r in runs]))
    g, caught = court.gradcheck, court.caught
    text = ["=== VERIFIED REPORT · chapter 0103 ===",
            "file: " + os.path.basename(__file__) + " · card_revision " + str(MIND_CARD["card_revision"]) + " · mode " + mode + " · mutant " + str(ACTIVE_MUTANT),
            "environment: python " + sys.version.split()[0] + " · numpy " + np.__version__,
            "seeds: " + str(seeds) + " · runtime_s " + "%.1f" % took + " · budget_s " + "%.0f" % TIME_BUDGET[mode],
            "n_params: forum " + str(n_params(runs[0]["forum"])) + " · one-sided forum " + str(n_params(runs[0]["lone"])),
            "gradcheck: " + str(g["tensors_checked"]) + "/" + str(g["tensors_total"]) + " tensors at init and after training · max_rel_error " + "%.2e" % g["max_rel_error"] + " · passed " + str(g["passed"]),
            "correctness:"]
    text += ["  " + c.ljust(5) + " " + n.ljust(32) + " " + ("PASS" if ok else "FAIL") + "  " + note for c, n, ok, note in sat]
    text.append("mutants: " + str(sum(caught.values())) + "/" + str(len(MUTANTS)) + " detected · score " + "%.2f" % (sum(caught.values()) / len(MUTANTS)) + " · "
                + ", ".join(k + (" caught" if caught.get(k) else " missed") for k in MUTANTS))
    text.append("hypotheses (paired over seeds; 95% percentile bootstrap of the mean, 2000 resamples):")
    text += ["  " + h["id"].ljust(8) + " mean_diff " + four(h["mean_diff"]) + " ci95 " + span_text(h["ci95"]) + " mesi " + str(h["mesi"]) + " seeds " + str(h["n_seeds"]) + " -> " + h["verdict"] for h in hyps]
    text.append("knockouts (forum, held-out Brier change):")
    text += ["  " + k["module"].ljust(10) + " " + k["mode"].ljust(7) + " signature " + str(k["signature"]).ljust(5) + " " + four(k["metric_change"]) + " ci95 " + span_text(k["ci95"]) for k in lesions]
    text.append("Brier (seed mean): forum held-out " + "%.4f" % mean(lambda r: r["scores"]["forum_held"]) + " · forum shifted " + "%.4f" % mean(lambda r: r["scores"]["forum_shift"])
                + " · one-sided shifted " + "%.4f" % mean(lambda r: r["scores"]["lone_shift"]) + " · Pyrrhonist suspension " + "%.4f" % mean(lambda r: r["scores"]["rival"])
                + " · asymmetric eloquence " + "%.4f" % mean(lambda r: r["scores"]["asym"]))
    text.append("forum (seed mean): held-out accuracy " + "%.3f" % mean(lambda r: r["acc"]) + " · verdicts revised by the law " + "%.3f" % mean(lambda r: r["revised"])
                + " · cases a Pyrrhonist would suspend " + "%.3f" % mean(lambda r: r["suspended"]) + " · commoner-side error " + "%.3f" % mean(lambda r: r["trivial"]))
    text += ["real-data bridge: " + bridge, "task_types: " + ", ".join(TASK_TYPES), "exit_code: " + str(code), "=== END REPORT ==="]
    return text


def protocol(mode, first, count, json_path, data_path):
    began, seeds = time.time(), list(range(first, first + count))
    print("chapter 0103 · mode " + mode + " · seeds " + str(seeds) + " · mutant " + str(ACTIVE_MUTANT), flush=True)
    runs = [run_seed(s, mode) for s in seeds]
    court = Court(dict(runs[0], seed=first), mode)
    sat = court.sit()
    hyps, lesions = weigh(runs, first, mode == "full" and count >= 5)
    bridge = data_bridge(data_path, first, UPDATES[mode]) if data_path else "skipped (no --data PATH given)"
    took = time.time() - began
    within = took <= TIME_BUDGET[mode]
    sat.append(("C8", "budget", within, "%.1f s of %.0f s" % (took, TIME_BUDGET[mode])))
    other_failures = any(not ok for c, _, ok, _ in sat if c != "C8")
    code = 1 if other_failures else (0 if within else 3)
    fields = ("schema_version", "chapter", "file", "card_revision", "environment", "seeds", "runtime_s", "n_params", "gradcheck", "correctness",
              "mutants", "hypotheses", "knockouts", "task_types", "exit_code")
    found = sum(court.caught.values())
    values = ["1.0", 103, os.path.basename(__file__), MIND_CARD["card_revision"], {"python": sys.version.split()[0], "numpy": np.__version__}, seeds,
              round(took, 2), n_params(runs[0]["forum"]), court.gradcheck, [{"id": c, "name": n, "passed": ok, "detail": d} for c, n, ok, d in sat],
              {"detected": found, "total": len(MUTANTS), "score": found / len(MUTANTS)}, hyps, lesions, TASK_TYPES, code]
    record = {}
    for key, value in zip(fields, values):
        record[key] = value
    write_report(minutes_text(mode, seeds, runs, court, sat, hyps, lesions, bridge, took, code), record, json_path)
    return code


SWITCHES = "quick card"
VALUES = "seed:int seeds:int json:str mutant:str data:str"


def main(argv=None):
    parser = argparse.ArgumentParser(prog=os.path.basename(__file__), description="Chapter 0103: the forum of the probable.")
    for word in SWITCHES.split():
        parser.add_argument("--" + word, action="store_true")
    for item in VALUES.split():
        word, kind = item.split(":")
        parser.add_argument("--" + word, type={"int": int, "str": str}[kind], default=(0 if word == "seed" else None))
    args = parser.parse_args(argv)
    if args.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    how_many = args.seeds if args.seeds is not None else (1 if args.quick else 5)
    usable = how_many >= 1 and (args.mutant is None or args.mutant in MUTANTS)
    if not usable:
        print("invalid --seeds or --mutant", file=sys.stderr)
        return 2
    set_mutant(args.mutant)
    try:
        outcome = protocol("quick" if args.quick else "full", args.seed, how_many, args.json, args.data)
    except FloatingPointError as exc:
        print("non-finite values: " + str(exc), file=sys.stderr)
        outcome = 4
    return outcome


if __name__ == "__main__":
    sys.exit(main())

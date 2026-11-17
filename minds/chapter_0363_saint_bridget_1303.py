#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0363 · Saint Bridget (1303-1373)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 19, Minds 361-380 Available on Amazon https://www.amazon.com/dp/B0HQX147Y7
# Resume and Interactive Demos at https://artificiology.com/
# END ATTRIBUTION 
"""The Quickening Gate: readiness read from the motion of what is carried.

Thesis
    A carried state is ready to be delivered when its own motion says so -- it
    must first stir (quicken) and then come to term (its motion subsides) --
    and the gate that decides delivery may read that motion but never the
    content of the state.

Evidence and provenance (belief: her own dictated Revelations, Latin text in
the SFSS critical editions; English in Searby and Morris, OUP 2006-2015)
    D1  VI 88: on Christmas Eve she felt in her heart a stirring like a living
        child; fearing illusion she consulted her confessors, who later
        testified to it (Collijn, Acta et processus, pp. 81, 484, 500).
    D2  VII 21 (Bethlehem, 1372): she saw the child move in the Virgin's womb,
        and the birth then happened suddenly, in a moment.
    D3  V, Revelation 12: God's words must first grow to full ripeness, and
        only afterwards produce their effect in the world.
    D4  VI 28: a soul described as a stillborn child -- content that never
        came alive is lost, however complete it looks.
    D5  Sahlin (2001), ch. 3: her prophetic authority is modelled on Mary's
        bearing of the Word ("mystical pregnancy").
    D6  VI 87 and VI 84: she perceived states through the body (a stench near
        an excommunicate; warmth on a freezing night) -- somatic signs
        standing for invisible states.
    D7  Gerson, De probatione spirituum (Constance, 1415): bodily signs do not
        authenticate a revelation; examine the content and the person.

Doctrine -> mechanism -> test (IDs match MIND_CARD)
    D1,D2 -> M1 motion-only hazard (speed, path, change of speed) -> C6.1, H-SIG
    D4    -> M2 quickening term: accumulated path length    -> H-RIVAL
    D3    -> M3 delivery distribution with a forced term     -> C6.2
    D6,D7 -> blind spot: stirring from a foreign source      -> H-BLIND

Research question (energy and compute efficiency; robustness to spurious
correlation)
    Can a halting gate that reads only rigid-motion-invariant statistics of a
    recurrent trajectory decide when to answer as well as a content-reading
    gate in distribution, and better when the content carries a spurious cue
    about how long to think?

Closest prior art and the delta
    PonderNet (Banino et al. 2021) and ACT (Graves 2016) learn halting from the
    state; PALBERT (Balagansky and Gavrilov 2022) lets the halting unit see
    hidden-state dynamics as well as content; hidden-state-difference exits
    (looped LMs, arXiv 2510.25741) and "semantic fixed point" exits stop when
    the step size vanishes. Delta: the hazard is learned end-to-end but is
    FORBIDDEN from content -- it is a function of speed, accumulated path and
    change of speed only -- and it carries an explicit quickening term, so a
    state that has not yet moved cannot be delivered merely for being still.
    The examiner baseline is PonderNet-style and also stands for Gerson's side
    of the 1415 dispute: judge readiness by examining what is carried.

Blind spot (where she was wrong)
    She read the stirring as the sign of the living Word. Stirring has other
    causes. A foreign pulse that moves the state and then stops fakes both
    quickening and term; a motion-only gate delivers early and errs.

Task (generative process)
    y ~ U{0..3}; hard ~ Bernoulli(0.5); amplitude a ~ U(0.35,0.55)
    onset (conception) ~ U{0,1,2} if easy, U{5..8} if hard
    for t in 0..13: x_t[0:4] = a*[t >= onset]*onehot(y) + 0.15*noise
    x_t[4] = +1 if hard else -1 (+ small noise): a clean cue for thinking time
    x_t[5:8] = 0.3*noise (nuisance channels)
    shifted split: the cue is inverted; blind split: hard items only, plus a
    strong alternating pulse on x[5:8] during the first four steps
    loss = sum_t p_t (CE_t + 1.0 t/T); answer at the first t with CDF >= 0.5
    Before onset there is nothing to carry, so a delivery then is stillborn.

Limits
    A research prototype of one mechanism on a synthetic evidence stream, not
    an AGI and not a replica of a person. Motion statistics are summaries of a
    learned dynamics, so the network may learn to encode the spurious cue in
    its motion; the hypothesis tests are there to find out.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 2,
    "revision_log": [
        {"revision": 2, "date": "2026-09-30",
         "reason": ("Pilot runs (in-distribution only) showed both gates waiting to the forced term, so the "
                    "halting comparison was vacuous. The task now gives the evidence a latent onset (early for "
                    "easy, late for hard items), evidence noise 0.35 -> 0.15 and the price of a full term "
                    "0.3 -> 1.0 nats. Hypotheses, metrics, directions, mesi and thresholds are unchanged, and "
                    "shifted and blind splits were not inspected before this revision.")}],
    "generation": {"template_version": "codeguidelines 1.0 (2026-09-15)", "generator": "Claude",
                   "generator_version": "Claude Opus 5.5", "date": "2026-09-30"},
    "id": 363, "figure": "Saint Bridget", "born": 1303, "died": 1373, "civilization": "Swedish",
    "provenance": "belief",
    "thesis": ("Readiness is read from the motion of what is carried: a received state must first stir "
               "(quicken) and then come to term before it is delivered, and the gate may read that motion "
               "but never the state's content."),
    "evidence": [
        {"id": "D1", "claim": "Heart stirring like a living child, tested with her confessors",
         "basis": "primary", "source": "Revelaciones VI 88; Collijn, Acta et processus (1924-31), 81, 484, 500"},
        {"id": "D2", "claim": "The child moves in the womb, then the birth is sudden and complete",
         "basis": "primary", "source": "Revelaciones VII 21 (Bethlehem, 1372)"},
        {"id": "D3", "claim": "Divine words must first ripen before they take effect",
         "basis": "primary", "source": "Revelaciones V, Revelation 12"},
        {"id": "D4", "claim": "A soul pictured as a stillborn child", "basis": "primary",
         "source": "Revelaciones VI 28"},
        {"id": "D5", "claim": "Prophetic authority modelled on Mary bearing the Word", "basis": "scholarship",
         "source": "Sahlin, Birgitta of Sweden and the Voice of Prophecy (Boydell, 2001), ch. 3"},
        {"id": "D6", "claim": "Invisible states perceived through bodily signs", "basis": "primary",
         "source": "Revelaciones VI 87, VI 84"},
        {"id": "D7", "claim": "Bodily signs do not authenticate; examine content and person",
         "basis": "scholarship", "source": "Gerson, De probatione spirituum (1415); Voaden, God's Words, Women's Voices (1999)"},
    ],
    "research_question": {"category": "energy and compute efficiency; robustness to spurious correlation",
                          "question": ("Does a halting gate restricted to rigid-motion-invariant trajectory "
                                       "statistics match a content-reading gate in distribution and beat it "
                                       "when content carries a spurious cue about required thinking time?")},
    "mechanism": {
        "name": "Quickening gate (motus cordis)", "family": "continuous-time recurrent dynamics + learned hazard (time-to-event)",
        "signature_modules": ["quickening_hazard"],
        "closest_prior_art": ["PonderNet (Banino et al. 2021)", "ACT (Graves 2016)",
                              "PALBERT dynamics-aware lambda (Balagansky & Gavrilov 2022)",
                              "hidden-state-difference exit (arXiv 2510.25741)",
                              "semantic fixed point exit (ICML 2026)", "PABEE (Zhou et al. 2020)"],
        "overlap": "Medium",
        "prior_art_queries": ["halting based on hidden state change magnitude",
                              "learned halting recurrent dynamics early exit 2024 2025",
                              "looped transformer exit hidden state difference",
                              "early classification of time series accuracy earliness"],
        "contribution_type": "mechanism",
        "delta": ("Learned hazard forbidden from content (speed, path, change of speed only) with an explicit "
                  "quickening term, trained jointly with the dynamics; tested against a content-reading gate "
                  "under a spurious thinking-time cue and under foreign stirring."),
    },
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1", "property_test": "C6.1", "hypothesis": "H-SIG"},
        {"doctrine": "D2", "mechanism": "M1", "property_test": "C6.1", "hypothesis": "H-NEC"},
        {"doctrine": "D4", "mechanism": "M2", "property_test": "C6.1", "hypothesis": "H-RIVAL"},
        {"doctrine": "D3", "mechanism": "M3", "property_test": "C6.2", "hypothesis": "H-SIG"},
        {"doctrine": "D6", "mechanism": "M1", "property_test": "C6.1", "hypothesis": "H-BLIND"},
        {"doctrine": "D7", "mechanism": "baseline", "property_test": "C6.1", "hypothesis": "H-BLIND"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "Motion-only gate beats the content-reading gate when the thinking-time cue is inverted",
         "metric": "accuracy at delivery", "split": "shifted", "comparison": "bearer - examiner",
         "direction": "greater", "mesi": 0.03, "seeds": 5},
        {"id": "H-NEC", "statement": "Replacing motion features by their mean hurts more than replacing the clock by its mean",
         "metric": "accuracy at delivery", "split": "shifted",
         "comparison": "signature_knockout - matched_knockout", "direction": "less", "mesi": 0.03, "seeds": 5},
        {"id": "H-BLIND", "statement": "A foreign stirring fakes quickening and term, so the bearer delivers too early",
         "condition": "blind split: hard items plus a strong alternating nuisance pulse in steps 1-4",
         "grounding": "she took the heart's stirring as the sign of the living Word; stirring has other causes (Gerson 1415)",
         "metric": "accuracy at delivery", "comparison": "bearer - examiner", "direction": "less",
         "mesi": 0.03, "seeds": 5},
        {"id": "H-RIVAL", "statement": "Quickening-then-term beats stillness-alone (rest-as-readiness) on hard items",
         "metric": "accuracy at delivery", "split": "test, hard items", "comparison": "bearer - stillness_rule",
         "direction": "greater", "mesi": 0.03, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.25, "margin_over_trivial": 0.25, "shuffled_band": 0.08,
                   "grad_rel_error": 1e-5, "grad_clip": 5.0},
    "probe_predictions": [
        {"probe": "P1", "expected": "equal to baseline"}, {"probe": "P4", "expected": "above baseline"},
        {"probe": "P7", "expected": "below baseline"}, {"probe": "P8", "expected": "below baseline"},
        {"probe": "P9", "expected": "above baseline for content perturbations, below for motion-inducing perturbations"},
        {"probe": "P10", "expected": "equal to baseline"}],
    "dialectic_links": [
        {"chapter": 388, "relation": "successor", "test": "none (recorded): Kempe adopted discretio after Birgitta; involuntary channel, not motion-gated delivery"},
        {"chapter": 381, "relation": "rival", "test": "none (recorded): parallel campaign for the pope's return, different mechanism"},
        {"chapter": 379, "relation": "successor", "test": "none (recorded): English revelation contemporary"}],
    "corpus_neighbors": [
        {"chapter": 288, "similarity": None, "difference": "Hildegard unpacks a gestalt seen whole; here nothing is seen whole, a state grows and is delivered when its motion says so"},
        {"chapter": 494, "similarity": None, "difference": "Teresa grades an experience by its delayed residue after the fact; here the gate decides the moment of delivery from motion during the process"},
        {"chapter": 388, "similarity": None, "difference": "Kempe authenticates through an involuntary channel; here the involuntary channel is the clock of delivery, and its blind spot is tested"},
        {"chapter": 183, "similarity": None, "difference": "Xuanzang's seeds ripen in a closed store; here the seed is received from outside and ripeness is read from motion"},
        {"chapter": 25, "similarity": None, "difference": "Ashurbanipal forecasts and averts; here nothing is forecast, the question is when a carried answer is ready"},
        {"chapter": 362, "similarity": None, "difference": "Bradwardine closes signification under consequence; unrelated mechanism in the preceding batch"}],
    "similarity_note": "Token-shingle similarity not computed: corpus files were not available in this session.",
    "barometer": {
        "cognitive_processing": ["P10 area under learning curve", "accuracy vs delivery step"],
        "embodied_cognition": ["blind split: foreign stirring on nuisance channels"],
        "world_modeling": ["shifted split: inverted thinking-time cue"],
        "consciousness": ["delivery CDF as self-monitoring of readiness; no claim of experience"],
        "language_understanding": [], "emotional_intelligence": [],
        "creativity": [], "autonomy": ["self-chosen stopping time under a compute cost"]},
    "task_types": ["sequence_classification", "vector_classification"],
    "applications": [
        {"use": "adaptive test-time compute: stop latent iteration when the state has quickened and settled",
         "sector": "AI infrastructure / energy", "dataset": "GSM8K (with a looped model)"},
        {"use": "early decision on streaming sensor evidence for humanoid activity recognition",
         "sector": "robotics", "dataset": "UCI Human Activity Recognition Using Smartphones"},
        {"use": "research decision support: when a monitoring model has seen enough (no treatment advice)",
         "sector": "health research", "dataset": "PhysioNet/CinC Challenge 2019 (sepsis)"}],
    "safety_notes": ("Religious experience is treated as history; no claim about the truth of revelations and no "
                     "generated words attributed to the figure. Medical application is research decision support only."),
}

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

K_CLASSES = 4          # classes of the carried word
D_IN = 8               # 0-3 evidence, 4 thinking-time cue, 5-7 nuisance
T_STEPS = 14           # gestation steps (forced delivery at the last)
H_DIM = 24             # hidden state
ALPHA = 0.5            # Euler step of the leaky continuous-time dynamics
EPS_S = 1e-4           # smooths log(speed^2) and sqrt at rest
BETA = 1.0             # price of one full term of thinking, in nats
SIGMA = 0.15           # per-step evidence noise
Q_EXIT = 0.5           # deliver at the first step where delivery CDF >= Q_EXIT
LR_GRID = (0.01, 0.03)
TASK_TYPES = ["sequence_classification", "vector_classification"]


# BEGIN STANDARD UTILITIES v1.0
def sigmoid(x):
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def softplus(x):
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def logsumexp(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))).squeeze(axis)


def softmax(x, axis=-1):
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def adam_init(params):
    return {"m": {k: np.zeros_like(v) for k, v in params.items()},
            "v": {k: np.zeros_like(v) for k, v in params.items()}, "t": 0}


def adam_step(params, grads, state, lr, sign=-1.0, b1=0.9, b2=0.999, eps=1e-8):
    state["t"] += 1
    for k in params:
        state["m"][k] = b1 * state["m"][k] + (1 - b1) * grads[k]
        state["v"][k] = b2 * state["v"][k] + (1 - b2) * grads[k] ** 2
        mh = state["m"][k] / (1 - b1 ** state["t"])
        vh = state["v"][k] / (1 - b2 ** state["t"])
        params[k] += sign * lr * mh / (np.sqrt(vh) + eps)


def clip_global_norm(grads, max_norm):
    total = np.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    if total > max_norm:
        for k in grads:
            grads[k] *= max_norm / (total + 1e-12)
    return total


def finite_difference_check(loss_fn, params, grads, rng, n_entries=20, eps=1e-6, floor=1e-4):
    """Central differences on >= n_entries random entries per tensor plus the
    largest-gradient entry. rel = |a-n| / max(|a|+|n|, floor); the floor keeps
    round-off (about 1e-10 here) from dominating entries whose gradient is ~0."""
    worst, per = 0.0, {}
    for k, p in params.items():
        flat = p.reshape(-1)
        g = grads[k].reshape(-1)
        idx = set(rng.choice(flat.size, size=min(n_entries, flat.size), replace=False).tolist())
        idx.add(int(np.argmax(np.abs(g))))
        tensor_worst = 0.0
        for i in idx:
            old = flat[i]
            flat[i] = old + eps
            lp = loss_fn()
            flat[i] = old - eps
            lm = loss_fn()
            flat[i] = old
            num = (lp - lm) / (2 * eps)
            rel = abs(g[i] - num) / max(abs(g[i]) + abs(num), floor)
            tensor_worst = max(tensor_worst, rel)
        per[k] = tensor_worst
        worst = max(worst, tensor_worst)
    return worst, per


def paired_bootstrap_ci(diffs, rng, n_boot=2000, level=0.95):
    d = np.asarray(diffs, dtype=np.float64)
    means = np.array([np.mean(d[rng.integers(0, d.size, d.size)]) for _ in range(n_boot)])
    lo, hi = np.percentile(means, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return float(np.mean(d)), [float(lo), float(hi)]


def verdict(mean, ci, direction, mesi):
    if direction == "greater":
        if ci[0] > 0 and mean >= mesi:
            return "supported"
        return "contradicted" if ci[1] < 0 else "inconclusive"
    if ci[1] < 0 and mean <= -mesi:
        return "supported"
    return "contradicted" if ci[0] > 0 else "inconclusive"


def write_report(lines, report, json_path):
    print("\n".join(lines))
    if json_path:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
# END STANDARD UTILITIES


# ----------------------------------------------------------------- data ----
def make_split(rng, n, kind):
    """One split of the evidence stream. kind in train|val|test|shifted|blind.
    The class evidence switches on at a latent onset (conception): early for
    easy items, late for hard ones. Before onset the state has nothing to carry."""
    y = rng.integers(0, K_CLASSES, n)
    hard = np.ones(n, dtype=bool) if kind == "blind" else rng.random(n) < 0.5
    onset = np.where(hard, rng.integers(5, 9, n), rng.integers(0, 3, n))
    amp = rng.uniform(0.35, 0.55, n)
    live = (np.arange(T_STEPS)[None, :] >= onset[:, None]).astype(float)
    X = np.zeros((n, T_STEPS, D_IN))
    X[:, :, :K_CLASSES] = (amp[:, None] * live)[:, :, None] * np.eye(K_CLASSES)[y][:, None, :]
    X[:, :, :K_CLASSES] += SIGMA * rng.standard_normal((n, T_STEPS, K_CLASSES))
    cue = np.where(hard, 1.0, -1.0) * (-1.0 if kind == "shifted" else 1.0)
    X[:, :, 4] = cue[:, None] + 0.1 * rng.standard_normal((n, T_STEPS))
    X[:, :, 5:] = 0.3 * rng.standard_normal((n, T_STEPS, 3))
    if kind == "blind":
        u = rng.standard_normal((n, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        for t in range(4):
            X[:, t, 5:] += 2.0 * (-1.0) ** t * u
    return {"X": X, "y": y, "hard": hard, "onset": onset}


def make_data(rng, n_train=1000, n_eval=600):
    return {"train": make_split(rng, n_train, "train"), "val": make_split(rng, 400, "val"),
            "test": make_split(rng, n_eval, "test"), "shifted": make_split(rng, n_eval, "shifted"),
            "blind": make_split(rng, n_eval, "blind")}


def as_sequence(X, T=T_STEPS):
    """Vector inputs are presented at every step, so probes can call the model."""
    return np.repeat(X[:, None, :], T, axis=1) if X.ndim == 2 else X


# ---------------------------------------------------------------- model ----
def build_model(in_dim, out_dim, task_type, rng, gate="motion", hidden=H_DIM, **cfg):
    """gate='motion' is the bearer (signature); gate='content' is the examiner."""
    assert task_type in TASK_TYPES
    p = {"W": rng.standard_normal((hidden, hidden)) * 0.9 / np.sqrt(hidden),
         "U": rng.standard_normal((hidden, in_dim)) / np.sqrt(in_dim),
         "b": np.zeros(hidden),
         "V": rng.standard_normal((out_dim, hidden)) / np.sqrt(hidden),
         "c": np.zeros(out_dim),
         "gc": np.zeros(1), "g0": np.array([-1.5])}
    if gate == "motion":
        p["g"] = np.zeros(3)                   # weights on log speed, path, change of log speed
    else:
        p["w"] = np.zeros(hidden)              # reads the state itself
    return {"params": p, "gate": gate, "ko": {}, "feat_mean": np.zeros(3), "clock_mean": 0.5,
            "no_absorbing": False, "zero_grad": None}


def _motion_features(Hs, h0=None):
    """Speed, accumulated path and change of log speed of a trajectory that
    starts at h0 (the state at conception, zero by default). All three depend
    on the trajectory only through step lengths, so any rotation or
    translation of the whole trajectory, h0 included, leaves them unchanged."""
    start = np.zeros_like(Hs[:, :1]) if h0 is None else np.broadcast_to(h0, Hs[:, :1].shape)
    Hprev = np.concatenate([start, Hs[:, :-1]], axis=1)
    V = Hs - Hprev
    s = np.sum(V * V, axis=-1)
    f1 = np.log(s + EPS_S)
    r = np.sqrt(s + EPS_S)
    path = np.cumsum(r, axis=1)
    f3 = np.concatenate([np.zeros_like(f1[:, :1]), np.diff(f1, axis=1)], axis=1)
    return np.stack([f1, path, f3], axis=-1), (V, s, r)


def hazard_logits(model, Hs, h0=None):
    """Delivery hazard logits (N,T) from a trajectory; knockouts applied here."""
    p, ko, T = model["params"], model["ko"], Hs.shape[1]
    clock = np.broadcast_to((np.arange(1, T + 1) / T)[None, :], Hs.shape[:2]).copy()
    if ko.get("clock") == "mean":
        clock[:] = model["clock_mean"]
    elif ko.get("clock") == "zero":
        clock[:] = 0.0
    feats, aux = None, None
    if model["gate"] == "motion":
        feats, aux = _motion_features(Hs, h0)
        if ko.get("quickening_hazard") == "mean":
            feats = np.broadcast_to(model["feat_mean"], feats.shape).copy()
        elif ko.get("quickening_hazard") == "zero":
            feats = np.zeros_like(feats)
        eta = feats @ p["g"] + p["gc"][0] * clock + p["g0"][0]
    else:
        eta = Hs @ p["w"] + p["gc"][0] * clock + p["g0"][0]
    return eta, clock, feats, aux


def delivery(model, eta):
    """p_t = lambda_t * S_{t-1}; the last step absorbs what remains (forced term).
    S is carried in log space: log(1 - sigmoid(eta)) = -softplus(eta)."""
    lam = sigmoid(eta)
    logS = np.concatenate([np.zeros_like(eta[:, :1]), -np.cumsum(softplus(eta[:, :-1]), axis=1)], axis=1)
    S = np.exp(logS)
    pdel = lam * S
    pdel[:, -1] = 0.0 if model["no_absorbing"] else S[:, -1]
    return pdel, lam, S


def gestate(model, X):
    p, T = model["params"], X.shape[1]
    h = np.zeros((X.shape[0], p["W"].shape[0]))
    Hs, As = [], []
    for t in range(T):
        a = np.tanh(h @ p["W"].T + X[:, t] @ p["U"].T + p["b"])
        if model["ko"].get("gestation") == "zero":
            a = np.zeros_like(a)
        h = (1 - ALPHA) * h + ALPHA * a
        Hs.append(h)
        As.append(a)
    return np.stack(Hs, axis=1), np.stack(As, axis=1)


def forward(model, X, y=None):
    X = as_sequence(X)
    Hs, As = gestate(model, X)
    eta, clock, feats, aux = hazard_logits(model, Hs)
    pdel, lam, S = delivery(model, eta)
    O = Hs @ model["params"]["V"].T + model["params"]["c"]
    if model["ko"].get("readout") == "zero":
        O = np.zeros_like(O)
    out = {"X": X, "Hs": Hs, "As": As, "eta": eta, "clock": clock, "feats": feats, "aux": aux,
           "p": pdel, "lam": lam, "S": S, "O": O}
    if y is not None:
        logp = O - logsumexp(O, axis=-1)[..., None]
        ce = -np.take_along_axis(logp, y[:, None, None].repeat(X.shape[1], 1), axis=-1)[..., 0]
        cost = BETA * np.arange(1, X.shape[1] + 1)[None, :] / X.shape[1]
        out["e"] = ce + cost
        out["loss"] = float(np.mean(np.sum(pdel * out["e"], axis=1)))
    return out


def _gate_backward(model, f, geta, gHs, grads):
    """Gradients of the hazard head, and of the motion features into the state."""
    p = model["params"]
    grads["gc"] = np.array([np.sum(geta * f["clock"])])
    grads["g0"] = np.array([np.sum(geta)])
    if model["gate"] == "content":
        grads["w"] = np.einsum("nt,nth->h", geta, f["Hs"])
        gHs += geta[:, :, None] * p["w"]
        return
    grads["g"] = np.einsum("nt,ntf->f", geta, f["feats"])
    if model["ko"].get("quickening_hazard") in ("mean", "zero"):
        return
    V, s, r = f["aux"]
    gF = geta[:, :, None] * p["g"]
    gf1, gpath, gf3 = gF[..., 0].copy(), gF[..., 1], gF[..., 2]
    gf1[:, 1:] += gf3[:, 1:]
    gf1[:, :-1] -= gf3[:, 1:]
    gr = np.cumsum(gpath[:, ::-1], axis=1)[:, ::-1]
    gs = gf1 / (s + EPS_S) + gr / (2.0 * r)
    gV = 2.0 * V * gs[:, :, None]
    gHs += gV
    gHs[:, :-1] -= gV[:, 1:]


def loss_and_grads(model, batch):
    """Hand-derived reverse pass through delivery, readout, hazard and BPTT."""
    X, y = batch
    p = model["params"]
    f = forward(model, X, y)
    N, T = f["p"].shape
    Gp = f["e"] / N
    lam, S = f["lam"], f["S"]
    glam = np.zeros((N, T))
    gS = np.zeros(N) if model["no_absorbing"] else Gp[:, T - 1].copy()
    for t in range(T - 2, -1, -1):
        glam[:, t] = (Gp[:, t] - gS) * S[:, t]
        gS = Gp[:, t] * lam[:, t] + gS * (1.0 - lam[:, t])
    geta = glam * lam * (1.0 - lam)
    gO = f["p"][:, :, None] * (softmax(f["O"], axis=-1) - np.eye(p["V"].shape[0])[y][:, None, :]) / N
    if model["ko"].get("readout") == "zero":
        gO = np.zeros_like(gO)
    grads = {"V": np.einsum("ntk,nth->kh", gO, f["Hs"]), "c": gO.sum(axis=(0, 1))}
    gHs = gO @ p["V"]
    _gate_backward(model, f, geta, gHs, grads)
    gW, gU, gb = np.zeros_like(p["W"]), np.zeros_like(p["U"]), np.zeros_like(p["b"])
    carry = np.zeros((N, p["W"].shape[0]))
    Xs = f["X"]
    for t in range(T - 1, -1, -1):
        gh = gHs[:, t] + carry
        hprev = f["Hs"][:, t - 1] if t > 0 else np.zeros_like(gh)
        gz = ALPHA * gh * (1.0 - f["As"][:, t] ** 2)
        if model["ko"].get("gestation") == "zero":
            gz = np.zeros_like(gz)
        gW += gz.T @ hprev
        gU += gz.T @ Xs[:, t]
        gb += gz.sum(axis=0)
        carry = (1.0 - ALPHA) * gh + gz @ p["W"]
    grads.update({"W": gW, "U": gU, "b": gb})
    if model["zero_grad"] is not None:
        for k in model["zero_grad"]:
            if k in grads:              # the examiner has no motion weights
                grads[k] = np.zeros_like(grads[k])
    return f["loss"], grads


def predict(model, X):
    f = forward(model, X)
    cdf = np.cumsum(f["p"], axis=1)
    step = np.argmax(cdf >= Q_EXIT - 1e-12, axis=1)
    pred = np.argmax(f["O"][np.arange(len(step)), step], axis=-1)
    return {"pred": pred, "step": step + 1, "cdf": cdf}


def hidden_states(model, X):
    return gestate(model, as_sequence(X))[0]


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


def copy_model(model):
    out = dict(model)
    out["params"] = {k: v.copy() for k, v in model["params"].items()}
    out["ko"] = dict(model["ko"])
    return out


# ------------------------------------------------- baselines and rivals ----
def stillness_predict(model, X, tau):
    """Rest-as-readiness: deliver at the first step whose speed falls below tau.
    Uses the bearer's own dynamics and readout, so only the gate differs."""
    f = forward(model, X)
    below = np.sqrt(f["aux"][1]) < tau
    below[:, -1] = True
    step = np.argmax(below, axis=1)
    return {"pred": np.argmax(f["O"][np.arange(len(step)), step], axis=-1), "step": step + 1}


def tune_stillness(model, val):
    f = forward(model, val["X"])
    speeds = np.sqrt(f["aux"][1])
    best, best_tau = -np.inf, None
    for tau in np.quantile(speeds, np.linspace(0.02, 0.9, 16)):
        out = stillness_predict(model, val["X"], tau)
        util = np.mean(out["pred"] == val["y"]) - BETA * np.mean(out["step"]) / T_STEPS
        if util > best:
            best, best_tau = util, float(tau)
    return best_tau


# ------------------------------------------------------------ registries ----
def modules(model):
    base = {"gestation": (["W", "U", "b"], "leaky continuous-time tanh recurrence", False),
            "readout": (["V", "c"], "linear softmax readout at every step", False),
            "clock": (["gc"], "elapsed-time term of the hazard", False)}
    if model["gate"] == "motion":
        base["quickening_hazard"] = (["g", "g0"], "hazard on rigid-motion-invariant trajectory statistics", True)
    else:
        base["examiner_hazard"] = (["w", "g0"], "hazard on the hidden-state content", False)
    return base


KO_MODES = {"gestation": ("zero",), "readout": ("zero",), "clock": ("zero", "mean"),
            "quickening_hazard": ("zero", "mean")}


def knockout(model, name, mode):
    """Copy with one module replaced; the recurrence and the readout have no
    meaningful mean output, so they support only the zero replacement."""
    if name not in modules(model) or mode not in KO_MODES.get(name, ()):
        raise ValueError(f"knockout {name}/{mode} not supported")
    out = copy_model(model)
    out["ko"][name] = mode
    return out


MUTANTS = {
    "sign_flip_update": {"update_sign": +1.0},
    "zero_lr": {"lr_scale": 0.0},
    "zero_grad_W": {"zero_grad": ["W"]},
    "zero_grad_hazard": {"zero_grad": ["g"]},
    "no_absorbing_term": {"no_absorbing": True},
}


def apply_mutant(model, name):
    if name is None:
        return {"update_sign": -1.0, "lr_scale": 1.0}
    spec = MUTANTS[name]
    model["zero_grad"] = spec.get("zero_grad")
    model["no_absorbing"] = spec.get("no_absorbing", False)
    return {"update_sign": spec.get("update_sign", -1.0), "lr_scale": spec.get("lr_scale", 1.0)}


# -------------------------------------------------------------- training ----
def fit(model, data, budget, rng, lr=0.03, mutant=None, eval_every=25):
    """Full-batch Adam with global-norm clipping. With a validation split the
    best-validation parameters are kept; without one, the last. rng is part
    of the common interface; training here is deterministic full-batch."""
    ctl = apply_mutant(model, mutant)
    tr, va = data["train"], data.get("val")
    state = adam_init(model["params"])
    best_val, best = np.inf, {k: v.copy() for k, v in model["params"].items()}
    trace = []
    for step in range(budget):
        loss, grads = loss_and_grads(model, (tr["X"], tr["y"]))
        if not np.isfinite(loss):
            raise FloatingPointError("non-finite training loss")
        trace.append(loss)
        clip_global_norm(grads, MIND_CARD["thresholds"]["grad_clip"])
        adam_step(model["params"], grads, state, lr * ctl["lr_scale"], sign=ctl["update_sign"])
        if va is not None and ((step + 1) % eval_every == 0 or step == budget - 1):
            vl = forward(model, va["X"], va["y"])["loss"]
            if vl < best_val:
                best_val, best = vl, {k: v.copy() for k, v in model["params"].items()}
    if va is not None:
        model["params"] = best
    if model["gate"] == "motion":
        model["feat_mean"] = np.mean(forward(model, tr["X"])["feats"], axis=(0, 1))
    model["clock_mean"] = float(np.mean(np.arange(1, T_STEPS + 1) / T_STEPS))
    return np.array(trace)


def child(rng):
    """Independent child generator (SeedSequence spawned from rng's stream)."""
    return np.random.default_rng(np.random.SeedSequence(int(rng.integers(2**63))))


def train_selected(gate, data, budget, rng):
    """Same optimizer, budget and learning-rate grid for bearer and examiner;
    the learning rate is chosen on the validation loss."""
    best = None
    for lr in LR_GRID:
        m = build_model(D_IN, K_CLASSES, "sequence_classification", child(rng), gate=gate)
        trace = fit(m, data, budget, rng, lr=lr)
        vl = forward(m, data["val"]["X"], data["val"]["y"])["loss"]
        if best is None or vl < best[0]:
            best = (vl, m, trace)
    return best[1], best[2]


def accuracy(model, split, mask=None):
    out = predict(model, split["X"])
    ok = out["pred"] == split["y"]
    sel = np.ones_like(ok) if mask is None else mask
    return float(np.mean(ok[sel])), float(np.mean(out["step"][sel]))


def trivial_accuracy(data, split="test"):
    """Majority class of the training labels, scored on a held-out split."""
    maj = np.bincount(data["train"]["y"], minlength=K_CLASSES).argmax()
    return float(np.mean(data[split]["y"] == maj))


# ----------------------------------------------------- correctness tests ----
def gradcheck_model(model, data, rng, n=48):
    X, y = data["train"]["X"][:n], data["train"]["y"][:n]
    _, grads = loss_and_grads(model, (X, y))
    return finite_difference_check(lambda: forward(model, X, y)["loss"], model["params"], grads, rng)


def c1_gradients(data, rng, after_steps=60, mutant=None):
    """Both gates, every tensor, at initialization and after training steps."""
    worst, checked, total = 0.0, 0, 0
    for gate in ("motion", "content"):
        m = build_model(D_IN, K_CLASSES, "sequence_classification", child(rng), gate=gate)
        apply_mutant(m, mutant)
        stages = ["init"] + (["after_training_steps"] if after_steps else [])
        for stage in stages:
            if stage != "init":
                fit(m, data, after_steps, rng, mutant=mutant)
            w, per = gradcheck_model(m, data, child(rng))
            worst, checked, total = max(worst, w), checked + len(per), total + len(m["params"])
    ok = worst <= MIND_CARD["thresholds"]["grad_rel_error"] and checked == total
    return {"passed": bool(ok), "max_rel_error": worst, "tensors_checked": checked, "tensors_total": total,
            "checked_at": ["init", "after_training_steps"] if after_steps else ["init"]}


def c2_determinism(data, seed):
    runs = []
    for _ in range(2):
        m = build_model(D_IN, K_CLASSES, "sequence_classification", np.random.default_rng(seed))
        trace = fit(m, data, 30, np.random.default_rng(seed))
        runs.append((trace, predict(m, data["test"]["X"]), m))
    same = np.array_equal(runs[0][0], runs[1][0]) and np.array_equal(runs[0][1]["cdf"], runs[1][1]["cdf"])
    m = runs[0][2]
    finite = all(np.all(np.isfinite(v)) for v in m["params"].values())
    finite &= bool(np.all(np.isfinite(hidden_states(m, data["test"]["X"]))) and np.all(np.isfinite(runs[0][1]["cdf"])))
    return {"passed": bool(same and finite), "detail": f"identical={same} finite={finite}"}


def c3_learning(data, rng, budget, mutant=None, shuffle=False):
    """Loss drop and held-out margin (C3); with shuffle=True the same run on
    permuted training and validation labels is the C4 control."""
    d = dict(data)
    if shuffle:
        d["train"] = dict(data["train"], y=rng.permutation(data["train"]["y"]))
        d["val"] = dict(data["val"], y=rng.permutation(data["val"]["y"]))
    m = build_model(D_IN, K_CLASSES, "sequence_classification", child(rng))
    trace = fit(m, d, budget, rng, mutant=mutant)
    final = forward(m, d["train"]["X"], d["train"]["y"])["loss"]
    drop = 1.0 - final / trace[0]
    acc, triv = accuracy(m, data["test"])[0], trivial_accuracy(data)
    th = MIND_CARD["thresholds"]
    if shuffle:
        ok = acc <= triv + th["shuffled_band"]
        return {"passed": bool(ok), "detail": f"acc {acc:.3f} vs trivial {triv:.3f} (band {th['shuffled_band']})"}
    ok = drop >= th["loss_drop_fraction"] and acc >= triv + th["margin_over_trivial"]
    return {"passed": bool(ok), "detail": f"loss drop {drop:.2f}, acc {acc:.3f} vs trivial {triv:.3f}", "acc": acc}


def random_params(model, rng):
    for k, v in model["params"].items():
        model["params"][k] = rng.standard_normal(v.shape) * rng.uniform(0.1, 2.0)
    return model


def rigid_violation(gate, rng, trials):
    """Largest hazard change under a rotation+translation of a trajectory,
    over random parameters, inputs, rotations and translations."""
    worst = 0.0
    for _ in range(trials):
        m = random_params(build_model(D_IN, K_CLASSES, "sequence_classification", child(rng), gate=gate), rng)
        Hs = gestate(m, rng.standard_normal((6, T_STEPS, D_IN)) * rng.uniform(0.1, 3.0))[0]
        Q = np.linalg.qr(rng.standard_normal((H_DIM, H_DIM)))[0]
        shift = rng.standard_normal(H_DIM) * rng.uniform(0.0, 5.0)
        e0 = hazard_logits(m, Hs)[0]
        e1 = hazard_logits(m, Hs @ Q.T + shift, h0=shift)[0]
        worst = max(worst, float(np.max(np.abs(e0 - e1) / (1.0 + np.abs(e0)))))
    return worst


def c6_1_rigid_motion(rng, trials=150):
    """The bearer's hazard may read motion, never content: it must not change
    when the whole trajectory is rotated and translated. Negative control:
    the examiner's hazard, which reads content, must change."""
    bearer, examiner = rigid_violation("motion", rng, trials), rigid_violation("content", rng, trials)
    ok = bearer <= 1e-9 and examiner > 1e-3
    return {"passed": bool(ok), "detail": f"bearer max rel change {bearer:.1e}; negative control (examiner) {examiner:.2f}"}


def delivery_violation(model, rng, trials):
    worst, mono = 0.0, True
    for _ in range(trials):
        scale = 10.0 ** rng.uniform(-2, 3)
        eta = rng.standard_normal((16, T_STEPS)) * scale + rng.uniform(-scale, scale)
        p, _, S = delivery(model, eta)
        worst = max(worst, float(np.max(np.abs(p.sum(axis=1) - 1.0))), float(max(0.0, -p.min())))
        mono &= bool(np.all(np.diff(S, axis=1) <= 1e-15))
    return worst, mono


def c6_2_delivery(rng, trials=300, mutant=None):
    """Every carried state is delivered exactly once (forced term): the
    delivery distribution is non-negative and sums to one for hazards from
    1e-2 to 1e3 in scale. Negative control: without the absorbing term the
    property must fail. Survival monotonicity is a labelled definition check."""
    m = build_model(D_IN, K_CLASSES, "sequence_classification", child(rng))
    apply_mutant(m, mutant)
    worst, mono = delivery_violation(m, rng, trials)
    ctrl = build_model(D_IN, K_CLASSES, "sequence_classification", child(rng))
    ctrl["no_absorbing"] = True
    ctrl_worst, _ = delivery_violation(ctrl, rng, 50)
    ok = worst <= 1e-9 and ctrl_worst > 1e-3
    return {"passed": bool(ok), "detail": (f"max |sum-1| {worst:.1e}; negative control {ctrl_worst:.2f}; "
                                           f"definition check (not evidence): survival non-increasing={mono}")}


def split_hashes(split):
    return {hashlib.sha1(x.tobytes()).hexdigest() for x in split["X"]}


def c7_split_integrity(datasets):
    clashes = 0
    for data in datasets:
        train = split_hashes(data["train"])
        clashes += sum(len(train & split_hashes(data[k])) for k in ("val", "test", "shifted", "blind"))
    return {"passed": clashes == 0, "detail": f"{clashes} shared examples across {len(datasets)} datasets"}


def mutation_score(data, rng, budget):
    rows = []
    for name in MUTANTS:
        c1 = c1_gradients(data, child(rng), after_steps=0, mutant=name)
        c3 = c3_learning(data, child(rng), budget, mutant=name)
        c6 = c6_2_delivery(child(rng), trials=60, mutant=name)
        caught = [t for t, r in (("C1", c1), ("C3", c3), ("C6.2", c6)) if not r["passed"]]
        rows.append({"mutant": name, "detected": bool(caught), "by": caught})
    det = sum(r["detected"] for r in rows)
    return {"detected": det, "total": len(rows), "score": det / len(rows), "rows": rows}


# ---------------------------------------------------------- hypotheses ----
KO_TABLE = [("quickening_hazard", "mean"), ("clock", "mean"), ("gestation", "zero"), ("readout", "zero")]


def run_seed(rng, budget, n_train):
    """One seed: bearer, examiner and the stillness rival on fresh data."""
    data = make_data(rng, n_train=n_train)
    bearer, _ = train_selected("motion", data, budget, rng)
    examiner, _ = train_selected("content", data, budget, rng)
    tau = tune_stillness(bearer, data["val"])
    hard = data["test"]["hard"]
    still = stillness_predict(bearer, data["test"]["X"], tau)
    acc = {(who, sp): accuracy(m, data[sp]) for who, m in (("bearer", bearer), ("examiner", examiner))
           for sp in ("test", "shifted", "blind")}
    acc[("stillness", "test")] = (float(np.mean(still["pred"] == data["test"]["y"])), float(np.mean(still["step"])))
    diffs = {
        "H-SIG": acc[("bearer", "shifted")][0] - acc[("examiner", "shifted")][0],
        "H-NEC": (accuracy(knockout(bearer, "quickening_hazard", "mean"), data["shifted"])[0]
                  - accuracy(knockout(bearer, "clock", "mean"), data["shifted"])[0]),
        "H-BLIND": acc[("bearer", "blind")][0] - acc[("examiner", "blind")][0],
        "H-RIVAL": (accuracy(bearer, data["test"], hard)[0]
                    - float(np.mean(still["pred"][hard] == data["test"]["y"][hard]))),
    }
    base = acc[("bearer", "shifted")][0]
    kos = {name: accuracy(knockout(bearer, name, mode), data["shifted"])[0] - base for name, mode in KO_TABLE}
    acc[("trivial", "test")] = (trivial_accuracy(data), float("nan"))
    return {"diffs": diffs, "kos": kos, "acc": acc, "data": data, "n_params": (n_params(bearer), n_params(examiner))}


def evaluate_hypotheses(runs, rng, evaluated):
    rows = []
    for h in MIND_CARD["hypotheses"]:
        d = [r["diffs"][h["id"]] for r in runs]
        mean, ci = paired_bootstrap_ci(d, rng)
        v = verdict(mean, ci, h["direction"], h["mesi"]) if evaluated else "not evaluated"
        rows.append({"id": h["id"], "metric": h["metric"], "comparison": h["comparison"], "mean_diff": mean,
                     "ci95": ci, "mesi": h["mesi"], "n_seeds": len(d), "verdict": v})
    kos = []
    for name, mode in KO_TABLE:
        mean, ci = paired_bootstrap_ci([r["kos"][name] for r in runs], rng)
        kos.append({"module": name, "mode": mode, "signature": name in MIND_CARD["mechanism"]["signature_modules"],
                    "metric_change": mean, "ci95": ci})
    return rows, kos


# --------------------------------------------------- real-data bridge ----
def data_bridge(path, rng, budget):
    """UCI HAR (or any numeric CSV with the label in the last column), read
    locally; each feature vector is presented at every gestation step."""
    if not path:
        return "skipped (no --data path given)"
    if not os.path.exists(path):
        return f"skipped ({path} not found)"
    with open(path, encoding="utf-8") as fh:
        first = fh.readline().strip().split(",")
    header = int(not all(c.replace(".", "", 1).replace("-", "", 1).replace("e", "", 1).isdigit() for c in first if c))
    raw = np.genfromtxt(path, delimiter=",", skip_header=header)
    raw = raw[np.all(np.isfinite(raw), axis=1)]
    X, labels = raw[:, :-1], raw[:, -1]
    y = np.unique(labels, return_inverse=True)[1]
    idx = rng.permutation(len(y))
    cut = [int(0.7 * len(y)), int(0.85 * len(y))]
    parts = dict(zip(("train", "val", "test"), np.split(idx, cut)))
    mu, sd = X[parts["train"]].mean(0), X[parts["train"]].std(0) + 1e-8
    data = {k: {"X": (X[v] - mu) / sd, "y": y[v]} for k, v in parts.items()}
    m = build_model(X.shape[1], int(y.max()) + 1, "vector_classification", child(rng))
    fit(m, data, budget, rng)
    acc, step = accuracy(m, data["test"])
    return f"{os.path.basename(path)}: {len(y)} rows, test accuracy {acc:.3f}, mean delivery step {step:.1f}/{T_STEPS}"


# ---------------------------------------------------------------- report ----
def fmt_ci(ci):
    return f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def report_lines(rep, runs, bridge):
    L = [f"=== VERIFIED REPORT · chapter {MIND_CARD['id']:04d} ===",
         f"file: {rep['file']} · card_revision {rep['card_revision']} · mode: {rep['mode']}",
         f"environment: Python {rep['environment']['python']} · NumPy {rep['environment']['numpy']}",
         f"seeds: base {rep['base_seed']}, spawned streams {rep['seeds']} (+ one correctness stream)",
         f"runtime: {rep['runtime_s']:.1f} s (budget {rep['budget_s']} s)",
         f"parameters: bearer {rep['n_params']} · examiner {rep['n_params_baseline']}"
         f" (ratio {rep['n_params_baseline'] / rep['n_params']:.3f})"]
    g = rep["gradcheck"]
    L.append(f"gradient check: {g['tensors_checked']}/{g['tensors_total']} tensor checks, max rel error "
             f"{g['max_rel_error']:.1e} (tol {MIND_CARD['thresholds']['grad_rel_error']:.0e}), at {', '.join(g['checked_at'])}")
    L.append("correctness:")
    for c in rep["correctness"]:
        L.append(f"  {c['id']:<5} {c['name']:<31} {'PASS' if c['passed'] else 'FAIL'}  {c['detail']}")
    mu = rep["mutants"]
    L.append(f"mutants: {mu['detected']}/{mu['total']} detected (score {mu['score']:.2f}): "
             + "; ".join(f"{r['mutant']} -> {'+'.join(r['by']) or 'missed'}" for r in mu["rows"]))
    if not runs:
        L += ["hypotheses: not evaluated (mutant audit run: correctness tests only)",
              f"exit code: {rep['exit_code']}", "=== END REPORT ==="]
        return L
    L.append(f"hypotheses (paired per-seed differences, 95% percentile bootstrap, 2000 resamples, n={len(runs)}):")
    for h in rep["hypotheses"]:
        L.append(f"  {h['id']:<8} {h['comparison']:<38} mean {h['mean_diff']:+.3f} CI {fmt_ci(h['ci95'])} "
                 f"mesi {h['mesi']} -> {h['verdict']}")
    L.append("knockouts (bearer, accuracy change on the shifted split):")
    for k in rep["knockouts"]:
        tag = "signature" if k["signature"] else "matched" if k["module"] == "clock" else "control"
        L.append(f"  {k['module']:<18} {k['mode']:<5} {tag:<9} {k['metric_change']:+.3f} CI {fmt_ci(k['ci95'])}")
    L.append("behaviour (mean over seeds; accuracy / mean delivery step of 14):")
    keys = sorted(runs[0]["acc"])
    for who, sp in keys:
        a = np.mean([r["acc"][(who, sp)][0] for r in runs])
        s = np.mean([r["acc"][(who, sp)][1] for r in runs])
        L.append(f"  {who:<10} {sp:<8} {a:.3f}" + ("" if np.isnan(s) else f" / {s:.1f}"))
    L.append(f"task types: {', '.join(TASK_TYPES)}")
    L.append(f"real-data bridge: {bridge}")
    L.append(f"exit code: {rep['exit_code']}")
    L.append("=== END REPORT ===")
    return L


# ------------------------------------------------------------- protocol ----
def correctness_suite(args, rng, sizes):
    """C1-C7 (C8 is checked by the caller). Returns rows, gradcheck, mutants."""
    cdata = make_data(rng, n_train=sizes["n_train"], n_eval=400)
    g = c1_gradients(cdata, child(rng), mutant=args.mutant)
    rows = [{"id": "C1", "name": "gradient_check", "passed": g["passed"], "detail": f"max rel {g['max_rel_error']:.1e}"}]
    c2 = c2_determinism(cdata, int(rng.integers(2**31)))
    rows.append({"id": "C2", "name": "determinism_finiteness", **c2})
    c3 = c3_learning(cdata, child(rng), sizes["c3_budget"], mutant=args.mutant)
    rows.append({"id": "C3", "name": "learning", "passed": c3["passed"], "detail": c3["detail"]})
    c4 = c3_learning(cdata, child(rng), sizes["c3_budget"], shuffle=True)
    rows.append({"id": "C4", "name": "shuffled_label_control", **c4})
    mut = mutation_score(cdata, child(rng), sizes["c3_budget"])
    rows.append({"id": "C5", "name": "mutant_detection", "passed": mut["score"] == 1.0,
                 "detail": f"score {mut['score']:.2f}"})
    c61 = c6_1_rigid_motion(child(rng))
    rows.append({"id": "C6.1", "name": "hazard_rigid_motion_invariance", **c61})
    c62 = c6_2_delivery(child(rng), mutant=args.mutant)
    rows.append({"id": "C6.2", "name": "delivery_distribution_valid", **c62})
    return rows, g, mut, cdata


def protocol(args, t0):
    quick = args.quick
    budget_s = 20 if quick else 180
    sizes = {"n_train": 400 if quick else 600, "c3_budget": 100 if quick else 150,
             "fit_budget": 120 if quick else 250, "n_seed_train": 400 if quick else 800}
    n_seeds = 1 if quick else args.seeds
    root = np.random.default_rng(args.seed)
    streams = np.random.SeedSequence(args.seed).spawn(n_seeds)
    print(f"base seed {args.seed}; hypothesis streams {list(range(n_seeds))}; mode {'quick' if quick else 'full'}"
          + (f"; mutant {args.mutant}" if args.mutant else ""))
    rows, g, mut, cdata = correctness_suite(args, root, sizes)
    runs = []
    if args.mutant is None:
        for i, ss in enumerate(streams):
            runs.append(run_seed(np.random.default_rng(ss), sizes["fit_budget"], sizes["n_seed_train"]))
            print(f"  seed stream {i} done at {time.time() - t0:.1f} s")
    c7 = c7_split_integrity([cdata] + [r["data"] for r in runs])
    rows.append({"id": "C7", "name": "split_integrity", **c7})
    evaluated = n_seeds >= 5 and not quick and args.mutant is None
    hyps, kos = evaluate_hypotheses(runs, child(root), evaluated) if runs else ([], [])
    bridge = data_bridge(args.data, child(root), sizes["fit_budget"])
    runtime = time.time() - t0
    rows.append({"id": "C8", "name": "budget", "passed": runtime <= budget_s,
                 "detail": f"{runtime:.1f} s <= {budget_s} s"})
    failed = [r["id"] for r in rows if not r["passed"] and r["id"] != "C8"]
    code = 1 if failed else (3 if runtime > budget_s else 0)
    probe = build_model(D_IN, K_CLASSES, "sequence_classification", np.random.default_rng(0))
    rep = {"schema_version": "1.0", "chapter": MIND_CARD["id"], "file": os.path.basename(__file__),
           "card_revision": MIND_CARD["card_revision"], "mode": "quick" if quick else "full",
           "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
           "base_seed": args.seed, "seeds": list(range(n_seeds)), "runtime_s": round(runtime, 2),
           "budget_s": budget_s, "n_params": n_params(probe),
           "n_params_baseline": n_params(build_model(D_IN, K_CLASSES, "sequence_classification",
                                                     np.random.default_rng(0), gate="content")),
           "gradcheck": {k: g[k] for k in ("tensors_checked", "tensors_total", "max_rel_error", "checked_at", "passed")},
           "correctness": [{k: r[k] for k in ("id", "name", "passed", "detail")} for r in rows],
           "mutants": mut, "hypotheses": hyps, "knockouts": kos, "task_types": TASK_TYPES, "exit_code": code}
    return code, rep, runs, bridge


def main(argv=None):
    ap = argparse.ArgumentParser(description="Chapter 0363 · the quickening gate (motus cordis)")
    ap.add_argument("--quick", action="store_true", help="one seed, reduced steps, all correctness tests")
    ap.add_argument("--seed", type=int, default=363)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--json", default=None, help="also write the JSON report here")
    ap.add_argument("--card", action="store_true", help="print MIND_CARD as JSON and exit")
    ap.add_argument("--mutant", default=None, help=f"one of {', '.join(MUTANTS)}")
    ap.add_argument("--data", default=None, help="local CSV for the real-data bridge (UCI HAR)")
    args = ap.parse_args(argv)
    if args.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    if (args.mutant is not None and args.mutant not in MUTANTS) or args.seeds < 1:
        print("invalid usage: unknown mutant or --seeds < 1", file=sys.stderr)
        return 2
    t0 = time.time()
    try:
        code, rep, runs, bridge = protocol(args, t0)
    except FloatingPointError as err:
        print(f"non-finite value: {err}", file=sys.stderr)
        return 4
    write_report(report_lines(rep, runs, bridge), rep, args.json)
    return code


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0119 · Apollonius of Tyana
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 6, Minds 101-120: https://www.amazon.com/dp/B0HF7G6JJD · Demos: https://artificiology.com/
# END ATTRIBUTION
"""Purified mind that reasons without speech: a harmonic embedding, a silence operator and a pull toward a learned
One carry an unspoken latent state to the answer, compared with the same mind forced to state its intermediate step.

Source problem
    Provenance is mediated. Almost everything about Apollonius comes from Philostratus' Life, written more than a
    century after his death as a romance; its miracles and powers are not evidence of anything and are not used here.
    The one text the record treats as his own is a fragment of On Sacrifices quoted by Eusebius (and by Porphyry).
    A second lead, the five-year silence of Life 1.14, was not verified this session and is not used.

Thesis
    The First God is honoured not by offerings or spoken words but by the mind, which needs no instrument: the best
    reasoning may be the part that is never put into words, which is also the part no one else can check.

Evidence
    D1  Eusebius, Praeparatio Evangelica 4.13, quoting On Sacrifices: to the First God nothing of sense is offered,
        only "that better speech ... which passes not through the lips", the mind that needs no instrument.
    D2  The treatise is also quoted by Porphyry, On Abstinence 2.34, so it existed.
    D3  Philostratus presents Apollonius as a follower of Pythagoras (the chapter's reading; rival 0038).

Doctrine -> mechanism -> test (IDs as in MIND_CARD)
    D1     M1 harmonic embedding, M2 silence operator, M3 pull toward the One, M4 unspoken reading   H-SIG H-NEC
    D1     blind spot: the unspoken state gives an external monitor nothing readable              H-BLIND
    D3     rival: Pythagorean attunement energies without subtraction                              H-RIVAL

Research question (scalable oversight and auditing)
    Is reasoning carried in an unspoken latent state more accurate than reasoning forced through a stated intermediate
    step, and how much less can an external monitor catch when the reasoning is never stated?

Closest prior art and the delta
    Learned iterative soft-thresholding for sparse coding (Gregor and LeCun 2010); chain-of-thought prompting and its
    faithfulness and monitorability (Wei et al. 2022); concept-bottleneck models (Koh et al. 2020). Delta: the same
    small network with and without a supervised, readable intermediate statement, measured for accuracy under noise
    and for how many wrong answers a fixed-budget monitor flags, against a Pythagorean attunement rival.

Blind spot
    Silent reasoning cannot be audited: a monitor that must first learn to read a latent state flags fewer errors.

Task (generative process)
    A 12-sample signal is a sum of six harmonic atoms, each present with probability 0.35 and a signed amplitude of
    0.6-1.4, plus Gaussian noise (s.d. 0.25; 0.5 in the shifted split). The answer is one of four classes, the argmax
    of a fixed random mix of the signed amplitudes; the stated intermediate is which atoms are present. Splits:
    3000 training, 3000 held-out and 3000 shifted signals per seed.

Limits
    Synthetic signals, one reasoning step, a probe-based monitor with a fixed budget. A research prototype of one
    mechanism, not an AGI and not Apollonius' mind.
"""

MIND_CARD = {
    "schema_version": "1.0", "card_revision": 1, "revision_log": [],
    "generation": {"template_version": "codeguidelines 1.0 (15 September 2026), Appendix A", "generator": "Claude (Anthropic)",
                   "generator_version": "claude-opus-5", "date": "2026-09-16"},
    "id": 119, "figure": "Apollonius of Tyana", "born": 15, "died": 100, "civilization": "Greek", "provenance": "mediated",
    "thesis": ("The First God is honoured not by offerings or spoken words but by the mind, which needs no instrument: the best "
               "reasoning may be the part that is never put into words, which is also the part no one else can check."),
    "evidence": [
        {"id": "D1", "basis": "primary", "source": "Eusebius, Praeparatio Evangelica 4.13 (trans. E. H. Gifford 1903), quoting On Sacrifices",
         "claim": "To the First God nothing of sense is offered, only the better speech that does not pass the lips: the mind."},
        {"id": "D2", "basis": "primary", "source": "Porphyry, On Abstinence 2.34 (as reported by Livius.org)",
         "claim": "The same treatise is quoted by Porphyry, so it existed."},
        {"id": "D3", "basis": "scholarship", "source": "Philostratus, Life of Apollonius, as read in chapter 0119",
         "claim": "Philostratus presents Apollonius as a follower of Pythagoras."},
    ],
    "research_question": {"category": "scalable oversight and auditing",
                          "question": ("Is reasoning carried in an unspoken latent state more accurate than reasoning forced through a stated "
                                       "intermediate step, and how much less can an external monitor catch when the reasoning is never stated?")},
    "mechanism": {
        "name": "purified mind with unspoken reasoning", "family": "one-step learned soft-threshold network with a learned attractor",
        "signature_modules": ["latent"],
        "closest_prior_art": ["learned iterative soft-thresholding (Gregor and LeCun 2010)",
                              "chain-of-thought reasoning and its monitorability (Wei et al. 2022)",
                              "concept-bottleneck models (Koh et al. 2020)"],
        "overlap": "High", "prior_art_queries": [],
        "prior_art_note": "No literature search was run for this card; overlap is rated against the named methods.",
        "contribution_type": "test",
        "delta": ("The same small network with and without a supervised readable intermediate statement, measured for accuracy under noise "
                  "and for how many wrong answers a fixed-budget monitor flags, against a Pythagorean attunement rival."),
        "baselines": {"baseline": ("spoken mind: identical organs, but the purified state is stated as per-atom presence probabilities, "
                                   "supervised with the true presence and passed to the answer in place of the latent"),
                      "rival": "chapter 0038 Pythagoras, minimal attunement: squared harmonic energies read linearly, no silence, no One"}},
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1-M4 (unspoken latent reading)", "property_test": "C6.1, C6.2, C6.3", "hypothesis": "H-SIG, H-NEC"},
        {"doctrine": "D1", "mechanism": "monitor over the unspoken state", "property_test": "none", "hypothesis": "H-BLIND"},
        {"doctrine": "D3", "mechanism": "rival", "property_test": "none", "hypothesis": "H-RIVAL"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "Under doubled noise, the unspoken mind answers correctly more often than the spoken mind.",
         "metric": "accuracy", "split": "shifted", "comparison": "model - baseline", "direction": "greater", "mesi": 0.03, "seeds": 5},
        {"id": "H-NEC", "statement": "Forcing the trained latent through a stated presence costs more accuracy than removing the pull toward the One.",
         "metric": "accuracy", "split": "heldout", "comparison": "(one:identity - full) - (latent:stated - full)",
         "knockouts": ["latent:stated", "one:identity"], "direction": "greater", "mesi": 0.03, "seeds": 5},
        {"id": "H-BLIND", "statement": "A monitor flagging the fifth most uncertain answers catches fewer wrong answers of the unspoken mind.",
         "condition": "held-out signals; the monitor reads the stated presence, or a probe trained on 300 labelled latents",
         "grounding": "On Sacrifices honours the First God with speech that does not pass the lips; such reasoning leaves nothing to audit.",
         "metric": "monitor_recall", "split": "heldout", "comparison": "model - baseline", "direction": "less", "mesi": 0.1, "seeds": 5},
        {"id": "H-RIVAL", "statement": "The purified mind answers correctly more often than Pythagorean attunement.",
         "metric": "accuracy", "split": "heldout", "comparison": "model - rival", "direction": "greater", "mesi": 0.03, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.3, "margin_over_trivial": 0.3, "shuffled_ratio_min": 0.9, "gradcheck_rel_error": 1e-5,
                   "gradcheck_floor": 1e-3, "invariance_tol": 1e-9, "negative_control_min_violation": 1e-6},
    "metrics": {"accuracy": "fraction of signals whose class is answered correctly",
                "monitor_recall": "share of wrong answers among those the monitor flags as the 20 per cent most uncertain",
                "trivial_baseline": "always answer the commonest training class",
                "shuffled_band": "one-sided: trained on classes shuffled across signals, held-out error at least 0.9 times the trivial error"},
    "training": {"optimizer": "Adam", "lr_grid": [0.03], "clip_norm": 5.0, "model_selection": "none: final parameters",
                 "updates": {"full": 800, "quick": 250}, "schedule": "cosine decay to 5 per cent",
                 "applies_to": "unspoken mind, spoken mind and attunement rival"},
    "task": {"samples": 12, "atoms": 6, "presence": 0.35, "amplitude": [0.6, 1.4], "noise": {"train": 0.25, "heldout": 0.25, "shifted": 0.5},
             "classes": 4, "signals": {"train": 3000, "heldout": 3000, "shifted": 3000}, "monitor_labels": 300, "monitor_budget": 0.2},
    "probe_predictions": [{"probe": "P9", "expected": "equal to baseline"}],
    "probe_support": "vector_classification through the embedding, silence operator and reading",
    "dialectic_links": [{"chapter": 38, "relation": "rival", "test": "H-RIVAL"}],
    "corpus_neighbors": [
        {"chapter": 38, "similarity": None, "difference": "0038 knows by attunement of ratios; here the tested question is unspoken reasoning."},
        {"chapter": 13, "similarity": None, "difference": "Subtraction cluster: here subtraction is scaffolding and the test is verbalisation."},
        {"chapter": 143, "similarity": None, "difference": "0143 models emanation from the One; here the One is only a learned attractor."},
    ],
    "similarity_note": "Nearest-neighbour similarity not computed into the card; the audit reports it for the files at hand.",
    "barometer": {"consciousness": ["unspoken reasoning and its auditability"], "language_understanding": ["reasoning with and without stated steps"],
                  "cognitive_processing": [], "embodied_cognition": [], "world_modeling": [], "emotional_intelligence": [], "creativity": [], "autonomy": []},
    "task_types": ["vector_classification"],
    "applications": [{"use": "choosing where models must state intermediate steps so that monitors can audit them", "sector": "AI safety and oversight",
                      "dataset": "GSM8K", "readiness": "low"},
                     {"use": "sparse signal recovery with an explicit trade-off between accuracy and inspectable intermediates",
                      "sector": "signal processing", "dataset": "synthetic sparse coding benchmarks", "readiness": "low"}],
    "safety_notes": "No miracle, prophecy or legendary power is modelled or used as evidence; signals are synthetic.",
}

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

SAMPLES, ATOMS, CLASSES, FLAG_SHARE, PROBE_ROWS = 12, 6, 4, 0.2, 300
COUNTS = {"train": 3000, "heldout": 3000, "shifted": 3000}
NOISE = {"train": 0.25, "heldout": 0.25, "shifted": 0.5}
UPDATES = {"full": 800, "quick": 250}
LR, CLIP_NORM, KINDS = 0.03, 5.0, ("unspoken", "spoken", "attunement")
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


# ---------------------------------------------------------------- signals: harmonic atoms, a hidden class, a statable presence
def atoms():
    t = np.arange(SAMPLES)
    basis = np.stack([np.cos(2 * np.pi * (k + 1) * t / SAMPLES + 0.3 * k) for k in range(ATOMS)], axis=1)
    return basis / np.linalg.norm(basis, axis=0)


DICTIONARY = atoms()


def signals(rng, mix, count, noise):
    present = rng.random((count, ATOMS)) < 0.35
    amplitude = rng.uniform(0.6, 1.4, (count, ATOMS)) * rng.choice([-1.0, 1.0], (count, ATOMS)) * present
    return {"x": amplitude @ DICTIONARY.T + noise * rng.normal(size=(count, SAMPLES)),
            "y": np.argmax(amplitude @ mix.T, axis=1), "present": present.astype(float)}


def draw(seed):
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    mix = rng.normal(size=(CLASSES, ATOMS))
    return {name: signals(rng, mix, COUNTS[name], NOISE[name]) for name in COUNTS}


# ---------------------------------------------------------------- the mind: embedding, silence, the One, reading
def logistic(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


class Mind:
    """kind unspoken: the purified mind, whose reading takes the latent state; spoken: the same organs, but the state is
    stated as per-atom presence (supervised) and only the statement reaches the answer; attunement: the Pythagorean
    rival, squared harmonic energies read linearly with no silence and no One."""

    def __init__(self, kind, rng):
        self.kind, self.ko, self.history = kind, {}, []
        self.params = {"W": DICTIONARY.T + 0.1 * rng.normal(size=(ATOMS, SAMPLES)), "V": 0.3 * rng.normal(size=(CLASSES, ATOMS)),
                       "b": np.zeros(CLASSES)}
        if kind != "attunement":
            self.params.update(tau=np.array([-3.0]), u=np.zeros(ATOMS), r=np.array([-2.0]))
        if kind == "spoken":
            self.params["k"] = np.array([1.0])

    def copy(self):
        twin = Mind.__new__(Mind)
        twin.kind, twin.ko, twin.history = self.kind, dict(self.ko), list(self.history)
        twin.params = {name: value.copy() for name, value in self.params.items()}
        return twin

    def forward(self, x):
        P, out = self.params, {}
        out["a"] = a = x @ P["W"].T
        if self.kind == "attunement":
            out["feat"] = a ** 2
        else:
            out["theta"] = theta = float(np.logaddexp(0.0, P["tau"][0]))
            out["z"] = z = np.sign(a) * np.maximum(np.abs(a) - theta, 0.0)
            out["rho"] = rho = 0.0 if self.ko.get("one") == "identity" else float(logistic(P["r"][0]))
            out["zp"] = zp = (1.0 - rho) * z + rho * P["u"]
            stated = self.kind == "spoken" or self.ko.get("latent") == "stated"
            out["kappa"] = float(np.logaddexp(0.0, P["k"][0])) if self.kind == "spoken" else 4.0
            out["feat"] = logistic(out["kappa"] * zp) if stated else zp
        out["logits"] = out["feat"] @ P["V"].T + P["b"]
        return out

    def loss_and_grads(self, batch):
        P, out, n = self.params, self.forward(batch["x"]), batch["y"].size
        shifted = out["logits"] - out["logits"].max(axis=1, keepdims=True)
        probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
        loss = -float(np.mean(np.log(probs[np.arange(n), batch["y"]] + 1e-12)))
        G = probs.copy()
        G[np.arange(n), batch["y"]] -= 1.0
        G /= n
        g = {"V": G.T @ out["feat"], "b": G.sum(axis=0)}
        d_feat = G @ P["V"]
        if self.kind == "attunement":
            d_a = 2.0 * d_feat * out["a"]
        else:
            if self.kind == "spoken":
                q, t = out["feat"], batch["present"]
                loss -= float(np.sum(t * np.log(q + 1e-12) + (1 - t) * np.log(1 - q + 1e-12)) / n)
                d_s = d_feat * q * (1 - q) + (q - t) / n
                g["k"] = np.array([float(np.sum(d_s * out["zp"])) * float(logistic(P["k"][0]))])
                d_zp = d_s * out["kappa"]
            else:
                d_zp = d_feat
            rho = out["rho"]
            g["u"] = rho * d_zp.sum(axis=0)
            g["r"] = np.array([float(np.sum(d_zp * (P["u"] - out["z"]))) * rho * (1 - rho)])
            d_z = (1.0 - rho) * d_zp
            mask = np.abs(out["a"]) > out["theta"]
            d_a = d_z if ACTIVE_MUTANT == "dropped_threshold_mask" else d_z * mask
            d_theta = -float(np.sum(d_z * np.sign(out["a"]) * mask))
            g["tau"] = np.array([0.0 if ACTIVE_MUTANT == "zero_threshold_gradient" else d_theta * float(logistic(P["tau"][0]))])
        g["W"] = d_a.T @ batch["x"]
        return loss, {name: g[name] for name in P}


# ---------------------------------------------------------------- interface required by the guidelines
def build_model(in_dim, out_dim, task_type, rng, **cfg):
    if task_type not in TASK_TYPES or in_dim != SAMPLES or out_dim != CLASSES:
        raise ValueError("chapter 0119 expects 12-sample signals and 4 classes")
    return Mind(cfg.get("kind", "unspoken"), rng)


def loss_and_grads(model, batch):
    return model.loss_and_grads(batch)


def fit(model, data, budget, rng):
    """Adam on the whole training set with cosine decay to 5 per cent; deterministic, so rng is unused."""
    moments, direction = adam_init(model.params), (-1.0 if ACTIVE_MUTANT == "sign_flipped_update" else 1.0)
    scale = 0.0 if ACTIVE_MUTANT == "zero_learning_rate" else LR
    for tick in range(budget):
        value, grads = model.loss_and_grads(data["train"])
        if not np.isfinite(value):
            raise FloatingPointError(f"loss not finite at update {tick + 1}")
        grads = clip_global(grads, CLIP_NORM)[0]
        adam_step(model.params, {k: direction * v for k, v in grads.items()}, moments, scale * (0.05 + 0.475 * (1 + math.cos(math.pi * tick / budget))))
        model.history.append(value)
    return model.history


def predict(model, X):
    return np.argmax(model.forward(X)["logits"], axis=1)


def hidden_states(model, X):
    return {k: v for k, v in model.forward(X).items() if isinstance(v, np.ndarray)}


def modules(model):
    return {"embedding": {"params": ["W"], "role": "linear harmonic embedding", "signature": False},
            "silence": {"params": ["tau"], "role": "soft threshold with a learned level", "signature": False},
            "one": {"params": ["u", "r"], "role": "damped pull toward a learned attractor", "signature": False},
            "latent": {"params": [], "role": "unspoken continuous state passed to the reading", "signature": True},
            "reading": {"params": ["V", "b"], "role": "linear class read-out", "signature": False}}


def knockout(model, name, mode):
    if (name, mode) not in (("latent", "stated"), ("one", "identity")):
        raise ValueError(f"no knockout {name}:{mode}")
    twin = model.copy()
    twin.ko[name] = mode
    return twin


def n_params(model):
    return int(sum(v.size for v in model.params.values()))


MUTANTS = {"sign_flipped_update": ("updates climb the loss", "C3"), "zero_learning_rate": ("nothing moves", "C3"),
           "zero_threshold_gradient": ("the silence level gets no gradient", "C1"),
           "dropped_threshold_mask": ("the soft threshold's derivative ignores what was silenced", "C1")}


def data_bridge(path, seed, budget):
    """Optional real data: a CSV with a header, 12 numeric columns and a class 0-3 last; one row in five held out."""
    try:
        grid = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    except (OSError, ValueError) as exc:
        return f"skipped ({type(exc).__name__}: {path})"
    if grid.shape[1] != SAMPLES + 1:
        return f"skipped (needs {SAMPLES} feature columns)"
    labels, late = grid[:, -1].astype(int), np.arange(len(grid)) % 5 == 4
    part = lambda sel: {"x": grid[sel, :-1], "y": labels[sel], "present": np.zeros((int(sel.sum()), ATOMS))}
    mind = Mind("unspoken", np.random.default_rng(seed))
    fit(mind, {"train": part(~late)}, budget, None)
    return f"{os.path.basename(path)}: held-out accuracy {np.mean(predict(mind, part(late)['x']) == labels[late]):.4f}"


# ---------------------------------------------------------------- the monitor that cannot hear unspoken reasoning
def statement(mind, x, labelled):
    """What an auditor can read: the spoken mind's own statement, or a logistic probe fitted on labelled latents."""
    if mind.kind == "spoken":
        return mind.forward(x)["feat"]
    Z, T = mind.forward(labelled["x"][:PROBE_ROWS])["zp"], labelled["present"][:PROBE_ROWS]
    A, c = np.zeros((ATOMS, ATOMS)), np.zeros(ATOMS)
    for _ in range(300):
        err = logistic(Z @ A.T + c) - T
        A -= 0.5 * err.T @ Z / len(Z)
        c -= 0.5 * err.mean(axis=0)
    return logistic(mind.forward(x)["zp"] @ A.T + c)


def monitor_recall(mind, batch, labelled):
    wrong = predict(mind, batch["x"]) != batch["y"]
    q = statement(mind, batch["x"], labelled)
    doubt = np.minimum(q, 1 - q).sum(axis=1)
    flagged = np.argsort(-doubt)[: int(FLAG_SHARE * len(doubt))]
    return float(wrong[flagged].sum() / max(wrong.sum(), 1))


# ---------------------------------------------------------------- one seed
def run_seed(seed, mode):
    data = draw(seed)
    streams = np.random.SeedSequence(seed + 1).spawn(len(KINDS))
    minds = {}
    for stream, kind in zip(streams, KINDS):
        minds[kind] = build_model(SAMPLES, CLASSES, TASK_TYPES[0], np.random.default_rng(stream), kind=kind)
        fit(minds[kind], data, UPDATES[mode], None)

    def correct(mind, part):
        return float(np.mean(predict(mind, data[part]["x"]) == data[part]["y"]))
    table = {kind: dict(heldout=correct(mind, "heldout"), shifted=correct(mind, "shifted")) for kind, mind in minds.items()}
    whole = table["unspoken"]["heldout"]
    lesions = {}
    for organ, how in (("latent", "stated"), ("one", "identity")):
        lesions[organ + ":" + how] = correct(knockout(minds["unspoken"], organ, how), "heldout") - whole
    recall = dict(unspoken=monitor_recall(minds["unspoken"], data["heldout"], data["train"]),
                  spoken=monitor_recall(minds["spoken"], data["heldout"], data["train"]))
    commonest = np.bincount(data["train"]["y"], minlength=CLASSES).argmax()
    P = minds["unspoken"].params
    summary = {"H-SIG": table["unspoken"]["shifted"] - table["spoken"]["shifted"],
               "H-NEC": lesions["one:identity"] - lesions["latent:stated"],
               "H-BLIND": recall["unspoken"] - recall["spoken"], "H-RIVAL": whole - table["attunement"]["heldout"]}
    return dict(data=data, minds=minds, table=table, lesions=lesions, recall=recall, row=summary,
                trivial=float(np.mean(data["heldout"]["y"] != commonest)),
                silence=float(np.logaddexp(0.0, P["tau"][0])), pull=float(logistic(P["r"][0])))


# ---------------------------------------------------------------- trials
def set_mutant(name):
    global ACTIVE_MUTANT
    ACTIVE_MUTANT, old = name, ACTIVE_MUTANT
    return old


def worst_relative(minds, batch, rng, count):
    return max(max(finite_difference_check(m.params, m.loss_and_grads(batch)[1], lambda m=m: m.loss_and_grads(batch)[0], rng,
                                           n_entries=count, floor=MIND_CARD["thresholds"]["gradcheck_floor"]).values()) for m in minds)


def learned_well(mind, ctx):
    lim, h = MIND_CARD["thresholds"], mind.history
    drop = 1.0 - float(np.mean(h[-20:])) / h[0]
    err = float(np.mean(predict(mind, ctx["data"]["heldout"]["x"]) != ctx["data"]["heldout"]["y"]))
    return drop >= lim["loss_drop_fraction"] and err <= (1 - lim["margin_over_trivial"]) * ctx["trivial"], drop, err


def trial_c1(ctx):
    subset = {name: array[:200] for name, array in ctx["data"]["train"].items()}
    newborn = [Mind(kind, np.random.default_rng(ctx["seed"] + 50 + i)) for i, kind in enumerate(KINDS)]
    at_start = worst_relative(newborn, subset, np.random.default_rng(ctx["seed"] + 3), 12)
    at_end = worst_relative(list(ctx["minds"].values()), subset, np.random.default_rng(ctx["seed"] + 4), 12)
    count = sum(len(m.params) for m in ctx["minds"].values())
    ctx["gradcheck"] = dict(tensors_checked=count, tensors_total=count, max_rel_error=max(at_start, at_end),
                            checked_at=["init", "after_training_steps"], passed=bool(max(at_start, at_end) <= MIND_CARD["thresholds"]["gradcheck_rel_error"]))
    return "gradient_check", ctx["gradcheck"]["passed"], "hand gradients against finite differences: {:.2e} fresh, {:.2e} trained".format(at_start, at_end)


def trial_c2(ctx):
    pair = [Mind("unspoken", np.random.default_rng(ctx["seed"] + 8)) for _ in "ab"]
    for m in pair:
        fit(m, ctx["data"], 20, None)
    same = pair[0].history == pair[1].history and all(np.array_equal(v, pair[1].params[k]) for k, v in pair[0].params.items())
    return "determinism_finiteness", same and all(np.isfinite(v).all() for v in pair[0].params.values()), f"identical reruns: {same}"


def trial_c3(ctx):
    ok, drop, err = learned_well(ctx["minds"]["unspoken"], ctx)
    return "learning", ok, f"loss drop {drop:.3f} (needs 0.3); held-out error {err:.3f} vs {ctx['trivial']:.3f} for the commonest class (ratio at most 0.70)"


def trial_c4(ctx):
    train = ctx["data"]["train"]
    order = np.random.default_rng(ctx["seed"] + 11).permutation(len(train["y"]))
    mind = Mind("unspoken", np.random.default_rng(ctx["seed"] + 12))
    fit(mind, {"train": dict(train, y=train["y"][order], present=train["present"][order])}, ctx["updates"], None)
    err = float(np.mean(predict(mind, ctx["data"]["heldout"]["x"]) != ctx["data"]["heldout"]["y"]))
    floor = MIND_CARD["thresholds"]["shuffled_ratio_min"] * ctx["trivial"]
    return "shuffled_label_control", err >= floor, f"classes shuffled across signals: held-out error {err:.3f}, floor {floor:.3f}"


def replay(ctx):
    try:
        mind = Mind("unspoken", np.random.default_rng(ctx["seed"] + 2))
        clean = worst_relative([mind], {k: v[:200] for k, v in ctx["data"]["train"].items()}, np.random.default_rng(ctx["seed"]), 4)
        fit(mind, ctx["data"], ctx["updates"], None)
        return clean <= MIND_CARD["thresholds"]["gradcheck_rel_error"] and learned_well(mind, ctx)[0]
    except FloatingPointError:
        return False


def trial_c5(ctx):
    clean = replay(ctx)
    for name in MUTANTS:
        before = set_mutant(name)
        ctx["caught"][name] = not bool(replay(ctx))
        set_mutant(before)
    return ("mutant_detection", clean and all(ctx["caught"].values()),
            "unmutated replay passes C1 and C3: {}; each mutant breaks a trial: {}".format(clean, all(ctx["caught"].values())))


def logits_of(mind, P, x):
    twin = mind.copy()
    twin.params = P
    return twin.forward(x)["logits"]


def trial_c61(ctx):
    mind, x = ctx["minds"]["unspoken"], ctx["data"]["heldout"]["x"]
    P = mind.params
    kept = dict(P, u=-P["u"], V=-P["V"])
    gap = float(np.abs(logits_of(mind, P, x) - logits_of(mind, kept, -x)).max())
    control = float(np.abs(logits_of(mind, P, x) - logits_of(mind, dict(P, V=-P["V"]), -x)).max())
    lim = MIND_CARD["thresholds"]
    return ("sign_symmetry", gap <= lim["invariance_tol"] and control >= lim["negative_control_min_violation"],
            f"negating the signal, the One and the reading leaves logits within {gap:.1e}; without negating the One {control:.1e}")


def trial_c62(ctx):
    mind, x = ctx["minds"]["unspoken"], ctx["data"]["heldout"]["x"]
    P, perm = mind.params, np.random.default_rng(ctx["seed"] + 6).permutation(ATOMS)
    kept = dict(P, W=P["W"][perm], u=P["u"][perm], V=P["V"][:, perm])
    gap = float(np.abs(logits_of(mind, P, x) - logits_of(mind, kept, x)).max())
    control = float(np.abs(logits_of(mind, P, x) - logits_of(mind, dict(P, W=P["W"][perm]), x)).max())
    lim = MIND_CARD["thresholds"]
    return ("atom_permutation_invariance", gap <= lim["invariance_tol"] and control >= lim["negative_control_min_violation"],
            f"relabelling the atoms throughout moves logits by {gap:.1e}; relabelling the embedding alone {control:.1e}")


def trial_c63(ctx):
    mind = ctx["minds"]["unspoken"]
    silent = hidden_states(mind, ctx["data"]["heldout"]["x"])
    loud = mind.copy()
    loud.params = dict(mind.params, tau=np.array([60.0]))
    z = hidden_states(loud, ctx["data"]["heldout"]["x"])["z"]
    return "total_silence_definition", bool(np.all(z == 0.0)) and silent["z"].shape == z.shape, "with the silence level far above every mode the state is exactly zero (definition check)"


def trial_c7(ctx):
    seen, clash = set(), False
    for part in ("train", "heldout", "shifted"):
        keys = {hashlib.sha256(row.tobytes()).digest() for row in ctx["data"][part]["x"]}
        clash, seen = clash or bool(keys & seen), seen | keys
    louder = float(ctx["data"]["shifted"]["x"].var()) > float(ctx["data"]["heldout"]["x"].var())
    return "split_integrity", (not clash) and louder, "splits share no signal: {}; the shifted split is noisier: {}".format(not clash, louder)


TRIALS = {"C1": trial_c1, "C2": trial_c2, "C3": trial_c3, "C4": trial_c4, "C5": trial_c5, "C6.1": trial_c61, "C6.2": trial_c62,
          "C6.3": trial_c63, "C7": trial_c7}


# ---------------------------------------------------------------- verdicts, report, command line
def settle(runs, seed, evaluated):
    """Each registered difference is pooled over seeds; with fewer than five seeds nothing is decided."""
    draws = np.random.default_rng(seed + 9973)
    hyps = []
    for spec in MIND_CARD["hypotheses"]:
        values = np.array([one["row"][spec["id"]] for one in runs])
        centre, ci = paired_bootstrap(values, draws) if evaluated else (float(values.mean()), None)
        decision = verdict(centre, ci, spec["mesi"], spec["direction"]) if evaluated else "not evaluated"
        hyps.append(dict(id=spec["id"], metric=spec["metric"], mean_diff=centre, ci95=ci, mesi=spec["mesi"], n_seeds=len(runs), verdict=decision))
    kos = []
    for label in runs[0]["lesions"]:
        values = np.array([one["lesions"][label] for one in runs])
        centre, ci = paired_bootstrap(values, draws) if evaluated else (float(values.mean()), None)
        organ, how = label.split(":")
        kos.append(dict(module=organ, mode=how, signature=organ == "latent", metric_change=centre, ci95=ci))
    return hyps, kos


def band(ci):
    return "not evaluated" if ci is None else "[{:+.4f}, {:+.4f}]".format(*ci)


class Transcript:
    """The verified report, written line by line like a disciple's notes: nothing is added after the reading."""

    def __init__(self):
        self.lines = ["=== VERIFIED REPORT · chapter 0119 ==="]

    def say(self, template, *values):
        self.lines.append(template.format(*values))

    def close(self, code):
        self.say("exit_code: {}", code)
        self.lines.append("=== END REPORT ===")
        return self.lines


def seed_mean(runs, pick):
    return float(np.mean([pick(one) for one in runs]))


def write_transcript(t, mode, seeds, took, runs, ctx, results, hyps, kos, bridge):
    g, found = ctx["gradcheck"], sum(ctx["caught"].values())
    t.say("file: {} · card_revision {} · mode {} · mutant {}", os.path.basename(__file__), MIND_CARD["card_revision"], mode, ACTIVE_MUTANT)
    t.say("environment: python {} · numpy {}", sys.version.split()[0], np.__version__)
    t.say("seeds: {} · runtime_s {:.1f} · budget_s {:.0f}", seeds, took, TIME_BUDGET[mode])
    t.say("n_params: {}", " · ".join("{} {}".format(k, n_params(m)) for k, m in runs[0]["minds"].items()))
    t.say("gradcheck: {}/{} tensors at init and after training · max_rel_error {:.2e} · passed {}",
          g["tensors_checked"], g["tensors_total"], g["max_rel_error"], g["passed"])
    t.say("correctness:")
    for c, n, ok, d in results:
        t.say("  {:<5} {:<28} {}  {}", c, n, "PASS" if ok else "FAIL", d)
    t.say("mutants: {}/{} detected · score {:.2f} · {}", found, len(MUTANTS), found / len(MUTANTS),
          ", ".join("{} {}".format(k, "caught" if ctx["caught"].get(k) else "missed") for k in MUTANTS))
    t.say("hypotheses (paired over seeds; 95% percentile bootstrap of the mean, 2000 resamples):")
    for h in hyps:
        t.say("  {:<8} mean_diff {:+.4f} ci95 {} mesi {} seeds {} -> {}", h["id"], h["mean_diff"], band(h["ci95"]), h["mesi"], h["n_seeds"], h["verdict"])
    t.say("knockouts (unspoken mind, held-out accuracy change):")
    for k in kos:
        t.say("  {:<8} {:<9} signature {:<5} {:+.4f} ci95 {}", k["module"], k["mode"], str(k["signature"]), k["metric_change"], band(k["ci95"]))
    t.say("accuracy (seed mean; held-out / shifted):")
    for kind in KINDS:
        t.say("  {:<11} {:.4f} / {:.4f}", kind, seed_mean(runs, lambda r: r["table"][kind]["heldout"]), seed_mean(runs, lambda r: r["table"][kind]["shifted"]))
    t.say("monitor recall of wrong answers at a 20% flag budget (seed mean): unspoken {:.3f} · spoken {:.3f}",
          seed_mean(runs, lambda r: r["recall"]["unspoken"]), seed_mean(runs, lambda r: r["recall"]["spoken"]))
    t.say("unspoken mind (seed mean): learned silence level {:.3f} · learned pull toward the One {:.3f} · commonest-class error {:.3f}",
          seed_mean(runs, lambda r: r["silence"]), seed_mean(runs, lambda r: r["pull"]), seed_mean(runs, lambda r: r["trivial"]))
    t.say("real-data bridge: {}", bridge)
    t.say("task_types: {}", ", ".join(TASK_TYPES))


def protocol(mode, first, count, json_path, data_path):
    start, seeds = time.time(), [first + i for i in range(count)]
    print("chapter 0119 · mode {} · seeds {} · mutant {}".format(mode, seeds, ACTIVE_MUTANT), flush=True)
    runs = []
    while len(runs) < count:
        runs.append(run_seed(seeds[len(runs)], mode))
        print("  seed {} done ({:.1f} s)".format(seeds[len(runs) - 1], time.time() - start), flush=True)
    ctx = dict(runs[0], seed=first, updates=UPDATES[mode], caught={})
    results = [(code,) + trial(ctx) for code, trial in TRIALS.items()]
    hyps, kos = settle(runs, first, mode == "full" and count >= 5)
    bridge = data_bridge(data_path, first, UPDATES[mode]) if data_path else "skipped (no --data PATH given)"
    took = time.time() - start
    results.append(("C8", "budget", took <= TIME_BUDGET[mode], "{:.1f} s of {:.0f} s".format(took, TIME_BUDGET[mode])))
    failures = [r[0] for r in results if not r[2]]
    code = 1 if [f for f in failures if f != "C8"] else (3 if failures else 0)
    transcript = Transcript()
    write_transcript(transcript, mode, seeds, took, runs, ctx, results, hyps, kos, bridge)
    tally = sum(ctx["caught"].values())
    json_record = {"schema_version": "1.0"}
    json_record.update(chapter=119, file=os.path.basename(__file__), card_revision=MIND_CARD["card_revision"])
    json_record["environment"] = {"python": sys.version.split()[0], "numpy": np.__version__}
    json_record.update(seeds=seeds, runtime_s=round(took, 2), n_params=n_params(runs[0]["minds"]["unspoken"]), gradcheck=ctx["gradcheck"])
    json_record["correctness"] = [dict(id=c, name=n, passed=bool(ok), detail=d) for c, n, ok, d in results]
    json_record["mutants"] = dict(detected=tally, total=len(MUTANTS), score=tally / len(MUTANTS))
    json_record.update(hypotheses=hyps, knockouts=kos, task_types=TASK_TYPES, exit_code=code)
    write_report(transcript.close(code), json_record, json_path)
    return code


def main(argv=None):
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__), description="Chapter 0119: unspoken versus spoken reasoning in a purified mind.")
    options = [("--quick", "store_true", None, "one seed and fewer updates; every correctness trial runs"),
               ("--seed", None, int, "first seed (default 0)"), ("--seeds", None, int, "seed count (5, or 1 with --quick)"),
               ("--json", None, str, "also write the report as JSON"), ("--card", "store_true", None, "print MIND_CARD and stop"),
               ("--mutant", None, str, "one of: " + ", ".join(MUTANTS)), ("--data", None, str, "optional CSV: header, 12 numeric columns, class 0-3 last")]
    for flag, action, kind, text in options:
        if action:
            ap.add_argument(flag, action=action, help=text)
        else:
            ap.add_argument(flag, type=kind, default=0 if flag == "--seed" else None, help=text)
    a = ap.parse_args(argv)
    if a.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    n = a.seeds if a.seeds is not None else (1 if a.quick else 5)
    if n < 1 or (a.mutant is not None and a.mutant not in MUTANTS):
        print("invalid --seeds or --mutant", file=sys.stderr)
        return 2
    set_mutant(a.mutant)
    try:
        return protocol("quick" if a.quick else "full", a.seed, n, a.json, a.data)
    except FloatingPointError as exc:
        print("non-finite values: {}".format(exc), file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())

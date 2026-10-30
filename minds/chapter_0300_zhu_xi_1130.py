#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0300 · Zhu Xi
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 15, Minds 281-300 Available on Amazon https://www.amazon.com/dp/B0HMQ7TVJM
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0300_zhu_xi_1130 - Zhu Xi (1130-1200)
# END ATTRIBUTION
"""
Chapter 0300 · Zhu Xi (1130-1200) · known-principle-first investigation (chapter 0300, Zhu Xi (1130-1200)).

Thesis
    Each new thing is first read through the principle already known, and only
    what that reading leaves unexplained is allowed to deepen the principle
    itself (因其已知之理而益窮之), one thing after another, so that a shared
    structure accumulates which can read things never investigated.

Evidence and provenance (provenance: belief; ids as in MIND_CARD)
    D1 primary  Daxue zhangju, supplementary commentary to the gewu chapter:
                即物而窮其理 ... 因其已知之理而益窮之 ... 一旦豁然貫通.
    D2 primary  理一分殊: one principle, many shares (Zhuzi yulei 1; Ximing jie).
    D3 primary  Reading order and method: Great Learning first "to fix the
                frame" (Zhuzi yulei 14); greed for reading much is the great
                malady (Zhuzi yulei 104).
    D4 schol.   Goose Lake, 1175: inquiry and study (Zhu) against "the mind is
                principle" (Lu Jiuyuan).
    D5 schol.   1169 resolution of the zhonghe problem: composure (jing) held
                before and after feelings arise (Adler 2014).
    D6 schol.   Dai Zhen (Mengzi ziyi shuzheng, 1777): the Cheng-Zhu li was
                imposed as one pattern on things that do not share it.

Doctrine -> mechanism -> test
    D1 -> M1 reading_phase (head-only fit, shared trunk frozen, before any
          joint update of each task)        -> C6.2 -> H-SIG, H-NEC
    D2 -> M2 one shared feature bank + per-task input FiLM ("qi") and
          per-task readout ("share")       -> C6.1 -> H-SIG
    D4 -> rival "innate": trunk frozen at birth, only readouts learn
                                            ->         H-RIVAL
    D5 -> M3 global-norm clipping (bounded update), matched non-signature
          knockout                          ->         H-NEC
    D6 -> blind world: tasks share no components ->     H-BLIND

Research question (sample efficiency and transfer; continual learning)
    In a no-replay stream of regression tasks that share a hidden set of
    components, does fitting each task's private head on frozen shared
    features before joint fine-tuning leave a trunk that reads novel
    combinations of components (never co-occurring in training) better than
    plain sequential joint fine-tuning, and does that advantage vanish when
    the tasks share nothing?

Closest prior art and the delta
    LP-FT (Kumar et al., 2022, "Fine-Tuning can Distort Pretrained Features
    and Underperform Out-of-Distribution", ICLR) is the same two-step rule
    applied to one transfer. ELLA (Ruvolo and Eaton, 2013) builds a shared
    basis over a task stream. Overlap is High, so the contribution is the
    test: the rule is applied per task inside a lifelong stream, judged on a
    compositional split, against an innatist rival and in a world with no
    shared principle.

Blind spot
    Zhu Xi assumed one li runs through all things. Where tasks share no
    structure, reading by known principle has nothing to read with; the
    predicted advantage over joint fine-tuning should shrink (H-BLIND).

Task (generative process)
    Six hidden ridge components c_j(x) = sin(w_j . x + p_j), x in [-2, 2]^2.
    A task ("thing") mixes two components with signed weights, after its own
    affine distortion of the input (scale, offset), plus Gaussian noise.
    Training things use nine of the fifteen component pairs (every component
    appears at least twice); in-distribution new things reuse those pairs;
    shifted new things use the six pairs never seen together. New things are
    read from k=10 examples with the shared trunk frozen. The blind world
    draws two fresh private components for every thing.

Limits
    A small research prototype of one learning procedure, not an AGI and not
    a replica of a person. Synthetic task; no replay; one trunk size; the
    huoran guantong "sudden penetration" is described in the chapter, not
    claimed here. Code results are not evidence about the historical figure.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 2,
    "revision_log": ["r2 (after first --quick run): C3 failed because its held-out metric was "
                     "end-of-stream retention, which mixes learning with forgetting in a no-replay "
                     "stream. C3 now measures each thing just before and just after its own "
                     "investigation. Hypotheses, thresholds and hyperparameters unchanged; no "
                     "hypothesis had been evaluated."],
    "generation": {"template_version": "1.0", "generator": "Claude",
                   "generator_version": "Claude Opus 5.5", "date": "2026-10-06",
                   "notes": "Standard utilities block written to the section 5.4 "
                            "specification; the canonical shared block was not "
                            "available in this session. Neighbour MIND_CARDs were "
                            "not available; neighbours compared at master-record level."},
    "id": 300, "figure": "Zhu Xi", "born": 1130, "died": 1200, "civilization": "Chinese",
    "provenance": "belief",
    "thesis": "Read every new thing through the principle already known, and let only "
              "the residue deepen the principle (因其已知之理而益窮之).",
    "evidence": [
        {"id": "D1", "claim": "Gewu extends knowledge by probing each encountered thing, "
                              "building on principles already known, until integration.",
         "basis": "primary", "source": "Daxue zhangju, supplementary commentary (buzhuan)"},
        {"id": "D2", "claim": "One principle, many shares (li yi fen shu).",
         "basis": "primary", "source": "Zhuzi yulei 1; Ximing jie"},
        {"id": "D3", "claim": "Ordered reading; greed for breadth without depth is a malady.",
         "basis": "primary", "source": "Zhuzi yulei 14 and 104"},
        {"id": "D4", "claim": "Goose Lake 1175: inquiry and study versus the mind is principle.",
         "basis": "scholarship", "source": "Baidu Baike, Goose Lake Debate; de Bary, Sources of Chinese Tradition"},
        {"id": "D5", "claim": "1169 zhonghe resolution: composure before and after arousal.",
         "basis": "scholarship", "source": "Adler, Reconstructing the Confucian Dao (2014)"},
        {"id": "D6", "claim": "Critique that one li was imposed on unlike things.",
         "basis": "scholarship", "source": "Dai Zhen, Mengzi ziyi shuzheng (1777); IEP 'Dai Zhen'"},
    ],
    "research_question": {"category": "sample efficiency and transfer",
                          "question": "Does per-task linear-probe-then-fine-tune in a no-replay "
                                      "stream build a trunk that reads novel component "
                                      "combinations better than sequential joint fine-tuning?"},
    "mechanism": {
        "name": "known-principle-first investigation", "family": "continual multi-task transfer",
        "signature_modules": ["reading_phase"],
        "closest_prior_art": ["LP-FT, Kumar et al. 2022 (ICLR)", "ELLA, Ruvolo and Eaton 2013 (ICML)",
                              "multi-head sequential fine-tuning"],
        "overlap": "High",
        "prior_art_queries": ["linear probing then fine-tuning feature distortion",
                              "lifelong learning shared basis task stream",
                              "multi-head continual regression compositional split"],
        "contribution_type": "test",
        "delta": "LP-FT applied per task inside a lifelong stream, judged on a compositional "
                 "split, against an innatist rival and in a world with no shared principle.",
    },
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1 reading_phase", "property_test": "C6.2", "hypothesis": "H-SIG"},
        {"doctrine": "D2", "mechanism": "M2 shared bank + private qi/share", "property_test": "C6.1", "hypothesis": "H-SIG"},
        {"doctrine": "D4", "mechanism": "rival innate trunk", "property_test": "C6.2", "hypothesis": "H-RIVAL"},
        {"doctrine": "D5", "mechanism": "M3 clipping (matched knockout)", "property_test": "C6.3", "hypothesis": "H-NEC"},
        {"doctrine": "D6", "mechanism": "blind world", "property_test": "C7", "hypothesis": "H-BLIND"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "The full model reads shifted new things better than "
                                     "size-matched sequential joint fine-tuning.",
         "metric": "mean few-shot R2 on shifted new things", "split": "shifted",
         "comparison": "model - baseline", "direction": "greater", "mesi": 0.03, "seeds": 5},
        {"id": "H-NEC", "statement": "Knocking out the reading phase costs more than knocking "
                                    "out clipping (matched non-signature procedure).",
         "metric": "mean few-shot R2 on shifted new things",
         "comparison": "signature_knockout - matched_knockout", "direction": "less", "mesi": 0.03, "seeds": 5},
        {"id": "H-BLIND", "statement": "The advantage over joint fine-tuning shrinks in a world "
                                       "whose things share no components.",
         "condition": "blind world", "grounding": "one li assumed for all things (Dai Zhen's critique)",
         "metric": "(model - baseline) R2, blind minus native shifted",
         "comparison": "model - baseline", "direction": "less", "mesi": 0.03, "seeds": 5},
        {"id": "H-RIVAL", "statement": "Investigation beats an innate frozen trunk with learned "
                                       "readouts only (Lu Jiuyuan's side at Goose Lake).",
         "metric": "mean few-shot R2 on shifted new things",
         "comparison": "model - rival", "direction": "greater", "mesi": 0.05, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.5, "margin_over_trivial": 0.2, "shuffled_band": 0.05,
                   "gradcheck_rel_tol": 1e-5, "clip_norm": 1.0},
    "probe_predictions": [
        {"probe": "P6", "expected": "above baseline"}, {"probe": "P5", "expected": "above baseline"},
        {"probe": "P10", "expected": "above baseline"}, {"probe": "P2", "expected": "equal to baseline"},
        {"probe": "P4", "expected": "equal to baseline"}, {"probe": "P8", "expected": "equal to baseline"},
    ],
    "dialectic_links": [
        {"chapter": 305, "relation": "rival", "test": "H-RIVAL"},
        {"chapter": 265, "relation": "teacher", "test": "none"},
        {"chapter": 269, "relation": "teacher", "test": "none"},
        {"chapter": 73, "relation": "teacher", "test": "none"},
        {"chapter": 446, "relation": "successor", "test": "none"},
        {"chapter": 480, "relation": "successor", "test": "none"},
        {"chapter": 718, "relation": "successor", "test": "H-BLIND"},
    ],
    "corpus_neighbors": [
        {"chapter": 299, "similarity": None, "difference": "Ibn Rushd gates one common substrate by "
         "capacity; here the substrate is built task by task and the signature is the order of fitting."},
        {"chapter": 305, "similarity": None, "difference": "Lu Jiuyuan establishes a deep organ first and "
         "opens a linear organ later; here, per thing, the linear reading comes first and deepening after."},
        {"chapter": 480, "similarity": None, "difference": "Toegye routes which origin issues a feeling "
         "through a standing gate; no gate here, only a fitting order."},
        {"chapter": 269, "similarity": None, "difference": "Zhang Zai forecasts an unseen conserved phase; "
         "here nothing is conserved, components are recovered across tasks."},
        {"chapter": 85, "similarity": None, "difference": "Xunzi clears one-sided fixation by xu-yi-jing; "
         "here partiality belongs to the things, not the mind."},
        {"chapter": 285, "similarity": None, "difference": "Avempace reaches shared understanding without "
         "communication; here sharing is a trained trunk."},
        {"chapter": 292, "similarity": None, "difference": "Ibn Tufail learns causes by intervention on one "
         "island; here breadth across many things is the point."},
        {"chapter": 288, "similarity": None, "difference": "Hildegard decompresses a whole seen at once; "
         "here the whole is assembled from parts."},
    ],
    "barometer": {
        "cognitive_processing": ["P6 few-shot adaptation", "P2 relational transfer", "P10 sample efficiency"],
        "embodied_cognition": ["P9 perturbation robustness (input channel noise)"],
        "world_modeling": ["shifted compositional split", "P7 regime change"],
        "consciousness": ["P8 calibration (not targeted)"],
        "language_understanding": ["P3 compositional split (not targeted)"],
        "emotional_intelligence": [], "creativity": [],
        "autonomy": ["P5 task sequence forgetting"],
    },
    "task_types": ["vector_regression"],
    "applications": [
        {"use": "Onboard a new engine unit into a fleet degradation model: read it on frozen shared "
                "features before fine-tuning", "sector": "aerospace maintenance",
         "dataset": "NASA C-MAPSS turbofan degradation simulation"},
        {"use": "Add new buildings to a shared load-forecasting model one at a time", "sector": "energy",
         "dataset": "Building Data Genome Project 2"},
        {"use": "Add new monitoring stations to a multi-site regression model",
         "sector": "environmental research", "dataset": "UCI Beijing Multi-Site Air-Quality Data"},
    ],
    "safety_notes": "Philosophy-of-mind figure; no hazardous content. Models a learning procedure, "
                    "makes no claim to replicate the person and attributes no generated words to him.",
}

import argparse
import hashlib
import itertools
import json
import os
import sys
import time

import numpy as np

CHAPTER = 300
TASK_TYPES = ["vector_regression"]
TASK = {"n_comp": 6, "n_train_pairs": 9, "n_train_things": 12, "n_new": 6, "noise": 0.1,
        "n_train": 64, "n_test": 128, "k_shot": 10, "n_query": 200}
TRAIN = {"budget": 200, "quick_budget": 100, "adapt_steps": 150, "adapt_lr": 0.03, "adapt_l2": 1e-3}
LI_KEYS = ("li_W1", "li_b1", "li_W2", "li_b2")
HEAD_KEYS = ("qi_gamma", "qi_beta", "share_w", "share_c")
ACTIVE_MUTANT = [None]


class NonFiniteError(RuntimeError):
    pass


# BEGIN STANDARD UTILITIES v1.0
def softmax(x, axis=-1):
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def logsumexp(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(x - m), axis=axis))


def softplus(x):
    return np.logaddexp(0.0, x)


def adam_step(state, key, param, grad, lr, b1=0.9, b2=0.999, eps=1e-8):
    m, v, t = state.get(key, (np.zeros_like(param), np.zeros_like(param), 0))
    t += 1
    m = b1 * m + (1.0 - b1) * grad
    v = b2 * v + (1.0 - b2) * grad * grad
    param -= lr * (m / (1.0 - b1 ** t)) / (np.sqrt(v / (1.0 - b2 ** t)) + eps)
    state[key] = (m, v, t)


def clip_global_norm(grads, max_norm):
    norm = float(np.sqrt(sum(float(np.sum(g * g)) for g in grads.values())))
    scale = 1.0 if (not np.isfinite(max_norm) or norm <= max_norm) else max_norm / (norm + 1e-12)
    return {k: g * scale for k, g in grads.items()}, norm


def finite_difference_check(loss_fn, params, grads, rng, eps=1e-6, n_entries=20, floor=1e-6):
    """Central differences on every tensor: all entries of small tensors, otherwise
    n_entries random entries plus the largest-gradient entry. Relative error uses
    |a - n| / max(|a| + |n|, floor); the floor keeps float64 round-off on
    near-zero entries from dominating."""
    report = {}
    for name, p in params.items():
        flat = p.reshape(-1)
        gflat = grads[name].reshape(-1)
        if flat.size <= n_entries + 1:
            idx = np.arange(flat.size)
        else:
            idx = np.unique(np.append(rng.choice(flat.size, n_entries, replace=False),
                                      int(np.argmax(np.abs(gflat)))))
        worst = 0.0
        for i in idx:
            old = flat[i]
            flat[i] = old + eps
            lp = loss_fn()
            flat[i] = old - eps
            lm = loss_fn()
            flat[i] = old
            num = (lp - lm) / (2.0 * eps)
            worst = max(worst, abs(num - gflat[i]) / max(abs(num) + abs(gflat[i]), floor))
        report[name] = (worst, int(idx.size))
    return report


def paired_bootstrap_ci(diffs, rng, n_boot=2000, alpha=0.05):
    d = np.asarray(diffs, dtype=float)
    boots = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    return float(d.mean()), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def verdict(mean, lo, hi, direction, mesi):
    if direction == "greater":
        if lo > 0 and mean >= mesi:
            return "supported"
        return "contradicted" if hi < 0 else "inconclusive"
    if hi < 0 and mean <= -mesi:
        return "supported"
    return "contradicted" if lo > 0 else "inconclusive"


def write_report(chapter, lines, json_path=None, payload=None):
    print(f"=== VERIFIED REPORT · chapter {chapter:04d} ===")
    for line in lines:
        print(line)
    print("=== END REPORT ===")
    if json_path:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
# END STANDARD UTILITIES


# ----------------------------------------------------------------- data and tasks
def _component(rng):
    ang, mag = rng.uniform(0, 2 * np.pi), rng.uniform(0.9, 1.6)
    return np.array([mag * np.cos(ang), mag * np.sin(ang)]), float(rng.uniform(0, 2 * np.pi))


def _thing(rng, comps, pair, tag):
    signs = rng.choice([-1.0, 1.0], size=2)
    return {"comps": comps, "pair": pair, "tag": tag,
            "a": signs * rng.uniform(0.7, 1.3, size=2),
            "scale": rng.uniform(0.75, 1.25, size=2), "offset": rng.uniform(-0.3, 0.3, size=2)}


def _train_pairs(rng, n_comp, n_pairs):
    pairs = list(itertools.combinations(range(n_comp), 2))
    for _ in range(2000):
        chosen = [pairs[i] for i in rng.permutation(len(pairs))[:n_pairs]]
        if np.bincount(np.array(chosen).ravel(), minlength=n_comp).min() >= 2:
            return chosen, [p for p in pairs if p not in chosen]
    raise RuntimeError("could not draw a covering set of training pairs")


def make_world(rng, blind=False, cfg=TASK):
    """Native world: things share six components. Blind world: no sharing at all."""
    comps = [_component(rng) for _ in range(cfg["n_comp"])]
    train_pairs, held_pairs = _train_pairs(rng, cfg["n_comp"], cfg["n_train_pairs"])
    extra = [train_pairs[i] for i in rng.integers(0, len(train_pairs), cfg["n_train_things"] - len(train_pairs))]

    def spawn(pairs, tag):
        out = []
        for p in pairs:
            cs = [_component(rng), _component(rng)] if blind else [comps[p[0]], comps[p[1]]]
            out.append(_thing(rng, cs, None if blind else p, tag))
        return out
    id_pairs = [train_pairs[i] for i in rng.integers(0, len(train_pairs), cfg["n_new"])]
    shifted_pairs = [held_pairs[i % len(held_pairs)] for i in range(cfg["n_new"])]
    return {"blind": blind, "train_pairs": train_pairs, "held_pairs": held_pairs,
            "train": spawn(train_pairs + extra, "train"), "id": spawn(id_pairs, "id"),
            "shifted": spawn(shifted_pairs, "shifted")}


def thing_fn(thing, X):
    Xd = X * thing["scale"] + thing["offset"]
    return sum(a * np.sin(Xd @ w + ph) for a, (w, ph) in zip(thing["a"], thing["comps"]))


def sample(thing, n, rng, noise=TASK["noise"]):
    X = rng.uniform(-2.0, 2.0, size=(n, 2))
    return X, thing_fn(thing, X) + noise * rng.standard_normal(n)


def build_stream(world, rng, n_train=TASK["n_train"], n_test=TASK["n_test"]):
    stream = []
    for th in world["train"]:
        Xtr, ytr = sample(th, n_train, rng)
        Xte, yte = sample(th, n_test, rng)
        stream.append({"X_tr": Xtr, "y_tr": ytr, "X_te": Xte, "y_te": yte})
    return stream


# ----------------------------------------------------------------- model
def build_model(in_dim, out_dim, task_type, rng, n_things=1, hidden=16, n_art=8, **cfg):
    if task_type not in TASK_TYPES:
        raise ValueError(f"unsupported task type {task_type}")
    s = 1.0 / np.sqrt(n_art)
    params = {"li_W1": rng.normal(0, 1 / np.sqrt(in_dim), (in_dim, hidden)), "li_b1": np.zeros(hidden),
              "li_W2": rng.normal(0, 1 / np.sqrt(hidden), (hidden, n_art)), "li_b2": np.zeros(n_art),
              "qi_gamma": np.ones((n_things, in_dim)), "qi_beta": np.zeros((n_things, in_dim)),
              "share_w": rng.uniform(-s, s, (n_things, n_art, out_dim)),
              "share_c": np.zeros((n_things, out_dim))}
    config = {"protocol": "gewu", "read_frac": 0.3, "lr": 0.01, "batch": 32,
              "clip": MIND_CARD["thresholds"]["clip_norm"], "l2": 1e-4, "knocked": {}}
    config.update(cfg)
    return {"params": params, "cfg": config}


def _knock(out, mode):
    if mode == "zero":
        return np.zeros_like(out)
    if mode == "mean":
        return np.broadcast_to(out.mean(axis=0, keepdims=True), out.shape).copy()
    return out


def forward(model, X, things):
    P, kn = model["params"], model["cfg"]["knocked"]
    G, B = P["qi_gamma"][things], P["qi_beta"][things]
    z = G * X + B if "qi" not in kn else (X if kn["qi"] == "identity" else _knock(G * X + B, kn["qi"]))
    h = np.tanh(z @ P["li_W1"] + P["li_b1"])
    if "li_layer1" in kn:
        h = _knock(h, kn["li_layer1"])
    phi = np.tanh(h @ P["li_W2"] + P["li_b2"])
    if "li_layer2" in kn:
        phi = _knock(phi, kn["li_layer2"])
    yhat = np.einsum("nk,nko->no", phi, P["share_w"][things]) + P["share_c"][things]
    if "share" in kn:
        yhat = _knock(yhat, kn["share"])
    return yhat, (X, z, h, phi, G)


def loss_and_grads(model, batch):
    P, l2, kn = model["params"], model["cfg"]["l2"], model["cfg"]["knocked"]
    X, t = batch["X"], batch["t"]
    y = batch["y"].reshape(len(X), -1)
    yhat, (X, z, h, phi, G) = forward(model, X, t)
    r = yhat - y
    loss = float(np.mean(r * r) + 0.5 * l2 * (np.sum(P["li_W1"] ** 2) + np.sum(P["li_W2"] ** 2)))
    d = 2.0 * r / r.size
    gr = {k: np.zeros_like(v) for k, v in P.items()}
    np.add.at(gr["share_w"], t, phi[:, :, None] * d[:, None, :])
    np.add.at(gr["share_c"], t, d)
    da2 = np.einsum("nko,no->nk", P["share_w"][t], d) * (1 - phi ** 2)
    gr["li_W2"], gr["li_b2"] = h.T @ da2 + l2 * P["li_W2"], da2.sum(0)
    da1 = (da2 @ P["li_W2"].T) * (1 - h ** 2)
    gr["li_W1"], gr["li_b1"] = z.T @ da1 + l2 * P["li_W1"], da1.sum(0)
    if "qi" not in kn:
        dz = da1 @ P["li_W1"].T
        np.add.at(gr["qi_gamma"], t, dz * X)
        np.add.at(gr["qi_beta"], t, dz)
    if ACTIVE_MUTANT[0] == "zero_grad_li_W2":
        gr["li_W2"][:] = 0.0
    return loss, gr


def predict(model, X, things=None):
    t = np.zeros(len(X), dtype=int) if things is None else np.broadcast_to(things, (len(X),))
    return forward(model, X, t)[0]


def hidden_states(model, X, things=None):
    t = np.zeros(len(X), dtype=int) if things is None else np.broadcast_to(things, (len(X),))
    return forward(model, X, t)[1][3]


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


# ----------------------------------------------------------------- baselines and rival mechanisms
PROTOCOLS = {
    "gewu": "per task: head-only reading (trunk frozen) for read_frac of the budget, then joint",
    "joint": "baseline: per task joint fine-tuning for the whole budget (multi-head sequential FT)",
    "innate": "rival (Lu Jiuyuan): trunk frozen at birth; only task heads ever learn",
}


def frozen_steps(cfg, budget):
    if cfg["protocol"] == "innate":
        return budget
    return int(round(cfg["read_frac"] * budget)) if cfg["protocol"] == "gewu" else 0


# ----------------------------------------------------------------- registries
def modules(model):
    return {
        "qi": {"params": ["qi_gamma", "qi_beta"], "signature": False,
               "role": "per-task feature-wise affine input modulation (input FiLM)"},
        "li_layer1": {"params": ["li_W1", "li_b1"], "signature": False, "role": "shared tanh hidden layer"},
        "li_layer2": {"params": ["li_W2", "li_b2"], "signature": False, "role": "shared tanh feature bank"},
        "share": {"params": ["share_w", "share_c"], "signature": False,
                  "role": "per-task linear readout over shared features"},
        "reading_phase": {"params": [], "signature": True,
                          "role": "procedure: per-task linear probe on frozen trunk before joint fine-tuning"},
        "clip": {"params": [], "signature": False, "role": "procedure: global-norm gradient clipping"},
    }


def knockout(model, name, mode="identity"):
    """Procedure knockouts change training; forward-module knockouts are for evaluation,
    except 'qi' identity, whose gradient (zero) stays exact and so can be trained."""
    new = {"params": {k: v.copy() for k, v in model["params"].items()},
           "cfg": dict(model["cfg"], knocked=dict(model["cfg"]["knocked"]))}
    if name == "reading_phase":
        new["cfg"].update(protocol="joint", read_frac=0.0)
    elif name == "clip":
        new["cfg"]["clip"] = float("inf")
    elif name in ("qi", "li_layer1", "li_layer2", "share"):
        if mode == "identity" and name != "qi":
            raise ValueError(f"identity knockout is shape-incompatible for {name}")
        new["cfg"]["knocked"][name] = mode
    else:
        raise KeyError(name)
    return new


MUTANTS = {"sign_flip": "update moves up the gradient",
           "zero_lr": "learning rate forced to zero",
           "zero_grad_li_W2": "gradient of the shared feature bank zeroed"}


# ----------------------------------------------------------------- training
def apply_updates(model, opt, upd, lr):
    grads, _ = clip_global_norm({k: g for k, (_, g) in upd.items()}, model["cfg"]["clip"])
    if ACTIVE_MUTANT[0] == "sign_flip":
        grads = {k: -g for k, g in grads.items()}
    step_lr = 0.0 if ACTIVE_MUTANT[0] == "zero_lr" else lr
    for k, (view, _) in upd.items():
        adam_step(opt, k, view, grads[k], step_lr)


def investigate(model, t, X, y, budget, rng, opt):
    """Gewu on one thing. Reading steps touch only row t of the head tensors."""
    P, cfg = model["params"], model["cfg"]
    n_frozen, losses = frozen_steps(cfg, budget), []
    for s in range(budget):
        idx = rng.choice(len(X), size=min(cfg["batch"], len(X)), replace=False)
        loss, g = loss_and_grads(model, {"X": X[idx], "y": y[idx], "t": np.full(len(idx), t)})
        if not np.isfinite(loss):
            raise NonFiniteError(f"non-finite loss on thing {t}")
        upd = {f"{k}#{t}": (P[k][t], g[k][t]) for k in HEAD_KEYS}
        if s >= n_frozen:
            upd.update({k: (P[k], g[k]) for k in LI_KEYS})
        apply_updates(model, opt, upd, cfg["lr"])
        losses.append(loss)
    return losses


def fit(model, data, budget, rng):
    opt, history = {}, []
    for t, d in enumerate(data):
        history.append(investigate(model, t, d["X_tr"], d["y_tr"], budget, rng, opt))
    return history


def r2(y, yhat):
    sst = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - float(np.sum((y - yhat) ** 2)) / max(sst, 1e-12)


def read_new_thing(model, Xs, ys, Xq, yq, rng):
    """Read a never-investigated thing: fresh head, trunk frozen, identical for every protocol."""
    n_art, out = model["params"]["share_w"].shape[1:]
    s = 1.0 / np.sqrt(n_art)
    head = {"qi_gamma": np.ones((1, Xs.shape[1])), "qi_beta": np.zeros((1, Xs.shape[1])),
            "share_w": rng.uniform(-s, s, (1, n_art, out)), "share_c": np.zeros((1, out))}
    tmp = {"params": dict({k: model["params"][k] for k in LI_KEYS}, **head),
           "cfg": dict(model["cfg"], clip=float("inf"))}
    opt, t0 = {}, np.zeros(len(Xs), dtype=int)
    for _ in range(TRAIN["adapt_steps"]):
        loss, g = loss_and_grads(tmp, {"X": Xs, "y": ys, "t": t0})
        g["share_w"] = g["share_w"] + TRAIN["adapt_l2"] * head["share_w"]
        for k in HEAD_KEYS:
            adam_step(opt, k, head[k], g[k], TRAIN["adapt_lr"])
    return r2(yq, predict(tmp, Xq).ravel())


def transfer_score(model, things, seed):
    scores = []
    for i, th in enumerate(things):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 900, i]))
        Xs, ys = sample(th, TASK["k_shot"], rng)
        Xq, yq = sample(th, TASK["n_query"], rng)
        scores.append(read_new_thing(model, Xs, ys, Xq, yq, rng))
    return float(np.mean(scores))


def run_condition(seed, blind, protocol, budget, knock=None):
    world = make_world(np.random.default_rng(np.random.SeedSequence([seed, 1, int(blind)])), blind=blind)
    model = build_model(2, 1, "vector_regression", np.random.default_rng(np.random.SeedSequence([seed, 2])),
                        n_things=len(world["train"]), protocol=protocol)
    if knock:
        model = knockout(model, *knock)
    stream = build_stream(world, np.random.default_rng(np.random.SeedSequence([seed, 3, int(blind)])))
    fit(model, stream, budget, np.random.default_rng(np.random.SeedSequence([seed, 4])))
    return {"shifted": transfer_score(model, world["shifted"], seed), "id": transfer_score(model, world["id"], seed)}


# ----------------------------------------------------------------- tests: correctness
def _small_setup(seed, n_things=3, n=24):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 11]))
    world = make_world(rng)
    stream = build_stream(world, rng, n_train=n, n_test=8)[:n_things]
    model = build_model(2, 1, "vector_regression", rng, n_things=n_things)
    batch = {"X": np.vstack([d["X_tr"] for d in stream]), "y": np.concatenate([d["y_tr"] for d in stream]),
             "t": np.repeat(np.arange(n_things), n)}
    return model, stream, batch, rng


def c1_gradcheck(seed):
    model, stream, batch, rng = _small_setup(seed)
    out = {}
    for when in ("init", "after_training_steps"):
        if when != "init":
            fit(model, stream, 20, rng)   # 3 things x 20 = 60 training steps
        _, g = loss_and_grads(model, batch)
        out[when] = finite_difference_check(lambda: loss_and_grads(model, batch)[0], model["params"], g, rng)
    worst = max(v[0] for rep in out.values() for v in rep.values())
    tensors = len(out["init"])
    ok = worst <= MIND_CARD["thresholds"]["gradcheck_rel_tol"]
    return ok, {"max_rel_error": worst, "tensors_checked": tensors, "tensors_total": len(model["params"]),
                "checked_at": list(out)}


def _thing_loss(model, d, t):
    return float(np.mean((predict(model, d["X_tr"], t).ravel() - d["y_tr"]) ** 2))


def c3_learning(seed, budget, n_things=None, shuffle=False):
    """Learning is judged per thing, just before and just after its own investigation;
    end-of-stream retention is a studied property of the stream, not a learning test."""
    world = make_world(np.random.default_rng(np.random.SeedSequence([seed, 21])))
    stream = build_stream(world, np.random.default_rng(np.random.SeedSequence([seed, 22])))[:n_things]
    if shuffle:
        rs = np.random.default_rng(np.random.SeedSequence([seed, 23]))
        for d in stream:
            d["y_tr"] = d["y_tr"][rs.permutation(len(d["y_tr"]))]
    model = build_model(2, 1, "vector_regression", np.random.default_rng(np.random.SeedSequence([seed, 24])),
                        n_things=len(stream))
    rng, opt, before, after, held = np.random.default_rng(np.random.SeedSequence([seed, 25])), {}, [], [], []
    for t, d in enumerate(stream):
        before.append(_thing_loss(model, d, t))
        investigate(model, t, d["X_tr"], d["y_tr"], budget, rng, opt)
        after.append(_thing_loss(model, d, t))
        held.append(r2(d["y_te"], predict(model, d["X_te"], t).ravel()))
    b, a, h = float(np.mean(before)), float(np.mean(after)), float(np.mean(held))
    th = MIND_CARD["thresholds"]
    ok = (a <= (1 - th["loss_drop_fraction"]) * b) and h >= th["margin_over_trivial"]
    return ok, {"loss_before": b, "loss_after": a, "heldout_r2": h, "trivial_r2": 0.0}


def c2_determinism(seed):
    runs = []
    for _ in range(2):
        model, stream, batch, rng = _small_setup(seed)
        hist = fit(model, stream, 15, rng)
        runs.append((np.array(hist), predict(model, batch["X"], batch["t"])))
    same = np.array_equal(runs[0][0], runs[1][0]) and np.array_equal(runs[0][1], runs[1][1])
    finite = all(np.all(np.isfinite(a)) for r in runs for a in r)
    return same and finite, {"identical": bool(same), "finite": bool(finite)}


def c4_shuffled(seed, budget):
    _, info = c3_learning(seed, budget, shuffle=True)
    ok = info["heldout_r2"] <= MIND_CARD["thresholds"]["shuffled_band"]
    return ok, {"heldout_r2_shuffled": info["heldout_r2"], "band": MIND_CARD["thresholds"]["shuffled_band"]}


def c5_mutants(seed):
    results = {}
    for name in MUTANTS:
        ACTIVE_MUTANT[0] = name
        try:
            g_ok, _ = c1_gradcheck(seed)
            l_ok, _ = c3_learning(seed, 60, n_things=4)
        except NonFiniteError:
            g_ok, l_ok = True, False
        finally:
            ACTIVE_MUTANT[0] = None
        results[name] = not (g_ok and l_ok)
    return all(results.values()), {"detected": sum(results.values()), "total": len(results), "per_mutant": results}


def c6_properties(seed):
    """C6.1 feature-bank permutation equivariance; C6.2 reading isolation;
    C6.3 clipping preserves direction and bounds the norm. Each with a negative control."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 31]))
    model, stream, batch, _ = _small_setup(seed)
    viol, ctrl = 0.0, 0.0
    for _ in range(20):
        for k in model["params"]:
            model["params"][k] = model["params"][k] + rng.normal(0, 0.3, model["params"][k].shape)
        perm = rng.permutation(model["params"]["li_W2"].shape[1])
        base = predict(model, batch["X"], batch["t"])
        pm = knockout(model, "clip")
        pm["params"]["li_W2"], pm["params"]["li_b2"] = model["params"]["li_W2"][:, perm], model["params"]["li_b2"][perm]
        neg = predict(pm, batch["X"], batch["t"])
        pm["params"]["share_w"] = model["params"]["share_w"][:, perm, :]
        viol = max(viol, float(np.max(np.abs(predict(pm, batch["X"], batch["t"]) - base))))
        ctrl = max(ctrl, float(np.max(np.abs(neg - base))))
    c61 = viol < 1e-10 and ctrl > 1e-3
    fit(model, stream[:2], 10, rng)
    snap = {k: v.copy() for k, v in model["params"].items()}
    others = predict(model, batch["X"], batch["t"])[batch["t"] != 2]
    reading = knockout(model, "clip")
    reading["cfg"]["read_frac"] = 1.0            # every step is a reading step
    investigate(reading, 2, stream[2]["X_tr"], stream[2]["y_tr"], 5, rng, {})
    li_fixed = all(np.array_equal(snap[k], reading["params"][k]) for k in LI_KEYS)
    iso = np.array_equal(others, predict(reading, batch["X"], batch["t"])[batch["t"] != 2])
    leak = knockout(reading, "reading_phase")     # negative control: joint steps must move the trunk
    investigate(leak, 2, stream[2]["X_tr"], stream[2]["y_tr"], 1, rng, {})
    leak_detected = not all(np.array_equal(reading["params"][k], leak["params"][k]) for k in LI_KEYS)
    c62 = li_fixed and iso and leak_detected
    g = {"a": rng.normal(0, 3, 5), "b": rng.normal(0, 3, (2, 2))}
    cl, n0 = clip_global_norm(g, 1.0)
    n1 = float(np.sqrt(sum(np.sum(v * v) for v in cl.values())))
    cos = sum(float(np.sum(cl[k] * g[k])) for k in g) / (n1 * n0)
    per_tensor = {k: v / max(1.0, float(np.linalg.norm(v))) for k, v in g.items()}
    cos_bad = sum(float(np.sum(per_tensor[k] * g[k])) for k in g) / (
        float(np.sqrt(sum(np.sum(v * v) for v in per_tensor.values()))) * n0)
    c63 = n1 <= 1.0 + 1e-9 and abs(cos - 1) < 1e-12 and abs(cos_bad - 1) > 1e-6
    return c61 and c62 and c63, {"C6.1_equivariance": c61, "C6.2_reading_isolation": c62,
                                 "C6.3_clip_direction": c63, "max_violation": viol}


def _hash_rows(X):
    return {hashlib.sha1(np.ascontiguousarray(r).tobytes()).hexdigest() for r in X}


def c7_splits(seed):
    world = make_world(np.random.default_rng(np.random.SeedSequence([seed, 1, 0])))
    stream = build_stream(world, np.random.default_rng(np.random.SeedSequence([seed, 3, 0])))
    rows_ok = all(not (_hash_rows(d["X_tr"]) & _hash_rows(d["X_te"])) for d in stream)
    pairs_ok = not (set(world["held_pairs"]) & set(world["train_pairs"]))
    shifted_ok = all(th["pair"] in world["held_pairs"] for th in world["shifted"])
    rng = np.random.default_rng(np.random.SeedSequence([seed, 900, 0]))
    Xs, _ = sample(world["shifted"][0], TASK["k_shot"], rng)
    Xq, _ = sample(world["shifted"][0], TASK["n_query"], rng)
    support_ok = not (_hash_rows(Xs) & _hash_rows(Xq))
    ok = rows_ok and pairs_ok and shifted_ok and support_ok
    return ok, {"train_test_disjoint": rows_ok, "pairs_disjoint": pairs_ok,
                "shifted_novel": shifted_ok, "support_query_disjoint": support_ok}


# ----------------------------------------------------------------- tests: hypotheses
CONDITIONS = {"full": (False, "gewu", None), "baseline": (False, "joint", None),
              "no_clip": (False, "gewu", ("clip",)), "rival": (False, "innate", None),
              "no_qi": (False, "gewu", ("qi", "identity")),
              "blind_full": (True, "gewu", None), "blind_baseline": (True, "joint", None)}


def run_hypotheses(seeds, budget, rng):
    per = {name: [] for name in CONDITIONS}
    for s in seeds:
        for name, (blind, proto, knock) in CONDITIONS.items():
            per[name].append(run_condition(s, blind, proto, budget, knock))
    sh = {k: np.array([r["shifted"] for r in v]) for k, v in per.items()}
    diffs = {"H-SIG": sh["full"] - sh["baseline"],
             "H-NEC": sh["baseline"] - sh["no_clip"],
             "H-BLIND": (sh["blind_full"] - sh["blind_baseline"]) - (sh["full"] - sh["baseline"]),
             "H-RIVAL": sh["full"] - sh["rival"]}
    rows = []
    for h in MIND_CARD["hypotheses"]:
        mean, lo, hi = paired_bootstrap_ci(diffs[h["id"]], rng)
        rows.append({"id": h["id"], "metric": h["metric"], "mean_diff": mean, "ci95": [lo, hi],
                     "mesi": h["mesi"], "n_seeds": len(seeds), "verdict": verdict(mean, lo, hi, h["direction"], h["mesi"])})
    knock_rows = []
    for name, cond, sig in (("reading_phase", "baseline", True), ("clip", "no_clip", False), ("qi", "no_qi", False)):
        mean, lo, hi = paired_bootstrap_ci(sh[cond] - sh["full"], rng)
        knock_rows.append({"module": name, "signature": sig, "metric_change": mean, "ci95": [lo, hi]})
    summary = {k: (float(v.mean()), float(np.mean([r["id"] for r in per[k]]))) for k, v in sh.items()}
    return rows, knock_rows, summary


# ----------------------------------------------------------------- report
def data_bridge(path, seed):
    if not path:
        return "skipped (no --data PATH given)"
    if not os.path.exists(path):
        return f"skipped ({path} not found)"
    arr = np.genfromtxt(path, delimiter=",", skip_header=1)
    groups, X, y = arr[:, 0].astype(int), arr[:, 1:-1], arr[:, -1]
    X, y = (X - X.mean(0)) / (X.std(0) + 1e-9), (y - y.mean()) / (y.std() + 1e-9)
    ids = np.unique(groups)
    cut = max(1, int(0.7 * len(ids)))
    stream = [{"X_tr": X[groups == g], "y_tr": y[groups == g]} for g in ids[:cut]]
    rng = np.random.default_rng(seed)
    model = build_model(X.shape[1], 1, "vector_regression", rng, n_things=cut)
    fit(model, stream, TRAIN["quick_budget"], rng)
    scores = []
    for g in ids[cut:]:
        Xg, yg = X[groups == g], y[groups == g]
        if len(Xg) > TASK["k_shot"] + 5:
            p = rng.permutation(len(Xg))
            ks, qs = p[:TASK["k_shot"]], p[TASK["k_shot"]:]
            scores.append(read_new_thing(model, Xg[ks], yg[ks], Xg[qs], yg[qs], rng))
    return f"fitted {cut} groups; few-shot R2 on {len(scores)} held-out groups = {np.mean(scores):.4f}" \
        if scores else "no held-out group large enough"


def build_report(args, seeds, t_start, correctness, grad, mut, hyp, knocks, summary, bridge):
    n_p = n_params(build_model(2, 1, "vector_regression", np.random.default_rng(0), n_things=TASK["n_train_things"]))
    lines = [f"environment: python {sys.version.split()[0]} · numpy {np.__version__}",
             f"seeds: {seeds} (base {args.seed})", f"parameters: {n_p}",
             f"gradient check: {grad['tensors_checked']}/{grad['tensors_total']} tensors at {grad['checked_at']}, "
             f"max relative error {grad['max_rel_error']:.2e}",
             "correctness:"]
    lines += [f"  {cid} {name:<22s} {'PASS' if ok else 'FAIL'}  {json.dumps(det, default=float)}"
              for cid, name, ok, det in correctness]
    lines.append(f"mutation score: {mut['detected']}/{mut['total']}")
    lines.append("hypotheses (mean diff, 95% bootstrap CI, mesi, verdict):")
    lines += [f"  {h['id']:<8s} {h['mean_diff']:+.4f}  [{h['ci95'][0]:+.4f}, {h['ci95'][1]:+.4f}]  "
              f"mesi {h['mesi']:.2f}  {h['verdict']}" for h in hyp]
    lines.append("knockouts (shifted R2 change vs full model):")
    lines += [f"  {k['module']:<14s} signature={k['signature']!s:<5s} {k['metric_change']:+.4f}  "
              f"[{k['ci95'][0]:+.4f}, {k['ci95'][1]:+.4f}]" for k in knocks]
    if summary:
        lines.append("condition means (shifted R2, in-distribution R2):")
        lines += [f"  {k:<15s} {v[0]:+.4f}  {v[1]:+.4f}" for k, v in summary.items()]
    lines += [f"real-data bridge: {bridge}", f"task types: {TASK_TYPES}",
              f"runtime: {time.time() - t_start:.1f} s"]
    return lines, n_p


def main(argv=None):
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--json", default=None)
    ap.add_argument("--card", action="store_true")
    ap.add_argument("--mutant", choices=sorted(MUTANTS), default=None)
    ap.add_argument("--data", default=None)
    args = ap.parse_args(argv)
    if args.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    budget = TRAIN["quick_budget"] if args.quick else TRAIN["budget"]
    seeds = [args.seed + i for i in range(1 if args.quick else args.seeds)]
    print(f"{os.path.basename(__file__)} · seeds {seeds} · budget {budget} steps per thing")
    ACTIVE_MUTANT[0] = args.mutant
    try:
        g_ok, grad = c1_gradcheck(args.seed)
        tests = [("C1", "gradient_check", g_ok, {"max_rel_error": grad["max_rel_error"]}),
                 ("C2", "determinism_finite", *c2_determinism(args.seed)),
                 ("C3", "learning", *c3_learning(args.seed, budget)),
                 ("C4", "shuffled_label_control", *c4_shuffled(args.seed, budget))]
        ACTIVE_MUTANT[0] = None
        mut_ok, mut = c5_mutants(args.seed)
        tests += [("C5", "mutant_detection", mut_ok, {"score": mut["detected"] / mut["total"]}),
                  ("C6", "property_tests", *c6_properties(args.seed)),
                  ("C7", "split_integrity", *c7_splits(args.seed))]
        if args.quick or args.mutant:
            hyp = [{"id": h["id"], "metric": h["metric"], "mean_diff": float("nan"), "ci95": [float("nan")] * 2,
                    "mesi": h["mesi"], "n_seeds": len(seeds), "verdict": "not evaluated"}
                   for h in MIND_CARD["hypotheses"]]
            knocks, summary = [], {}
        else:
            hyp, knocks, summary = run_hypotheses(seeds, budget, rng)
    except NonFiniteError as exc:
        print(f"non-finite values: {exc}")
        return 4
    runtime = time.time() - t0
    limit = 20.0 if args.quick else 180.0
    tests.append(("C8", "budget", runtime <= limit, {"runtime_s": round(runtime, 1), "limit_s": limit}))
    bridge = data_bridge(args.data, args.seed)
    lines, n_p = build_report(args, seeds, t0, tests, grad, mut, hyp, knocks, summary, bridge)
    all_ok = all(t[2] for t in tests)
    exit_code = 0 if all_ok else (3 if not tests[-1][2] and all(t[2] for t in tests[:-1]) else 1)
    payload = {"schema_version": "1.0", "chapter": CHAPTER, "file": os.path.basename(__file__),
               "card_revision": MIND_CARD["card_revision"],
               "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
               "seeds": seeds, "runtime_s": round(time.time() - t0, 2), "n_params": n_p,
               "gradcheck": dict(grad, passed=bool(g_ok)),
               "correctness": [{"id": c, "name": n, "passed": bool(ok), "detail": json.dumps(d, default=float)}
                               for c, n, ok, d in tests],
               "mutants": {"detected": mut["detected"], "total": mut["total"], "score": mut["detected"] / mut["total"]},
               "hypotheses": hyp, "knockouts": knocks, "task_types": TASK_TYPES, "exit_code": exit_code}
    write_report(CHAPTER, lines, args.json, payload)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

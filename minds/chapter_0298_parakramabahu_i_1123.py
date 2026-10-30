#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0298 · Parakramabahu I
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 15, Minds 281-300 Available on Amazon https://www.amazon.com/dp/B0HMQ7TVJM
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0298_parakramabahu_i_1123 - Parakramabahu I (c. 1123-1186)
# END ATTRIBUTION
"""Wana-Samudra synapse: fill-and-spill consolidation for continual learning.

Thesis
    Consolidate only the surplus: a learning signal that fits inside a small,
    evaporating shallow store is used while it lasts and then lost to the sky;
    only what overflows that store crosses the spillway (Sinhala wana) into
    larger, slower storage, ending in a reservoir that never drains to the sea.

Evidence and provenance (archetype_provenance = mediated)
    No words of Parakramabahu I (c.1123-1186) survive in his own hand. The
    water maxim is a speech the 13th-century chronicler Dhammakitti puts in his
    mouth (Culavamsa ch. 68, trans. Geiger 1929); Geiger's own introduction
    warns that the youthful king is drawn as an ideal Arthasastra ruler, so the
    espionage episodes (Cv 66.129 ff.) are not used here. The hydraulic record
    (enlargement of reservoirs around older tanks, river diversions on the
    Jajjara/Deduru Oya, the Parakrama Samudra built around the older Topa Wewa)
    is deeds, attested by the chronicle's tallies and by standing works. The
    fill-and-spill cascade is the dry-zone tank tradition he worked inside
    (Madduma Bandara 1985), not his invention.

Doctrine -> mechanism -> test (IDs as in MIND_CARD)
    D1 maxim: rain must not reach the sea unused
       -> M1 wana_spillway: dead-zone transfer; the excess over a tank's
          capacity is routed one way downstream, never clipped or discarded
       -> C6.1 conservation, C6.3 sub-capacity containment; H-SIG
    D2 cascade of tanks of rising size ending in a 'samudra'
       -> M2 samudra_cascade: K stores, rising capacity, falling evaporation,
          last store never leaks; visible weight = total stored water
       -> C6.2 capacity bound, C6.4 evaporation contraction; H-NEC
    D5/D7 treasury exhaustion and slow post-reign decay
       -> blind-spot condition: genuine change arriving as sub-threshold
          drizzle evaporates before it can consolidate; H-BLIND

Research question (continual learning and plasticity)
    Does consolidating only the overflow of a local, evaporating cushion make
    a continual learner robust to short, intense label-poisoning storms,
    relative to linear multi-timescale consolidation, and what does it cost
    when genuine change arrives as drizzle below the cushion?

Closest prior art and the delta
    Benna & Fusi (2016) complex synapse: a chain of communicating beakers with
    exponentially growing sizes and shrinking tubes; the first beaker is the
    visible weight and flow is linear and two-way (Kaplanis, Shanahan and
    Clopath 2018 used it for continual RL). Implemented here as the
    size-matched baseline with the same number of stores per weight.
    Delta: transfer is thresholded and one-way (fill-and-spill), the visible
    weight is the whole stored mass, and forgetting acts only on the shallow
    store, so a burst smaller than the cushion never reaches deep storage.

Blind spot
    A surplus-capturing apparatus registers floods, not drizzle. If a real
    change in the world arrives as weak, intermittent evidence whose shallow
    equilibrium stays below capacity, it evaporates every dry spell and the
    learner keeps answering the old rule. Grounded in the reign's documented
    exhaustion (heavy taxation, vanishing high-value coin, failed Pandyan
    expedition) and the slow, debated decay of the dry-zone reservoir state
    after 1186 (Indrapala ed. 1971; Murphey 1957).

Task (generative process)
    Three 'fields' (core tasks): binary rules given by random softplus teacher
    networks on 12 Gaussian features plus a task one-hot; multiplicative sensor
    jitter on the observed features is nuisance. A stream of years: each year
    has three seasons of clean mixed batches from all fields, each followed by
    storms (short bursts from one random field with every label flipped), each
    storm followed by a dry spell (no data; the stores evolve on their own).
    Held-out split: moderate storms. Shifted split: longer, more frequent
    storms. Blind-spot split: field 0's rule drifts to a new teacher whose
    data arrive as sparse, small batches (drizzle).

Limits
    Synthetic tasks, a two-layer network, three stores per weight and a fixed
    rain rule (clipped SGD). A research prototype of one mechanism for an
    AGI-oriented learner, not an AGI and not a model of the historical king.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 2,
    "revision_log": [
        "r2 (after quick runs and calibration on exploration seeds 50-52, before any evaluation-seed "
        "hypothesis run): (a) seasons water all three fields in mixed batches; single-field seasons made "
        "task interference, not storms, the main forgetting (storm-free plain SGD only ~0.62); "
        "(b) storm presets recalibrated with a model-neutral criterion: storms must cost plain SGD about "
        "10 points (moderate) and over 30 points (heavy) against a storm-free stream; "
        "(c) tuning grid widened from tau_f alone to 3x3 for each store model (tau_f x cushion capacity; "
        "tau_f x beaker ratio) because both are sensitive to their shape knob; "
        "(d) C4 uses the mean of five label permutations: a network fitted to shuffled labels is a random "
        "smooth function whose agreement with a smooth teacher varies widely (single permutations gave "
        "0.434 and 0.610 against a majority rate of 0.528), so one permutation cannot separate variance "
        "from leakage; a leak would raise every permutation and the mean; "
        "(e) storm persistence is measured against a counterfactual twin living through the same dry "
        "spell. Hypotheses, directions, splits and mesi unchanged; exploration showed Benna-Fusi ahead "
        "on the heavy split and the primary H-SIG was kept there."],
    "generation": {"template_version": "codeguidelines v1.0 (2026-09-15)", "generator": "Claude",
                   "generator_version": "Claude Opus 5.5", "date": "2026-09-24"},
    "id": 298, "figure": "Parakramabahu I", "born": 1123, "died": 1186, "civilization": "Sinhalese",
    "provenance": "mediated",
    "thesis": ("Capture the surplus, not the stream: only what overflows a small evaporating store is "
               "routed down a one-way cascade into ever larger storage that never drains to the sea."),
    "evidence": [
        {"id": "D1", "claim": "Water maxim: no rain may reach the ocean without first serving people; "
                              "a speech attributed to him by the 13th-c. chronicler",
         "basis": "scholarship", "source": "Culavamsa ch. 68 (Dhammakitti), trans. W. Geiger 1929"},
        {"id": "D2", "claim": "Reservoirs enlarged and joined around older tanks (Parakrama Samudra around "
                              "Topa Wewa); dams on the Jajjara (Deduru Oya) at three points",
         "basis": "deeds", "source": "Nicholas, Historical Topography (1963); Brohier, Ancient Irrigation "
                                     "Works in Ceylon (1934-35)"},
        {"id": "D3", "claim": "Dry-zone tanks work as fill-and-spill cascades: upstream surplus feeds "
                              "downstream tanks; tradition predates him",
         "basis": "scholarship", "source": "Madduma Bandara 1985, in Lundqvist et al. (eds), "
                                          "Strategies for River Basin Management"},
        {"id": "D4", "claim": "Chronicle tallies of works (165 dams, 3,910 canals, 163 major and 2,376 "
                              "minor tanks) are chronicle claims, largely restorations",
         "basis": "scholarship", "source": "De Silva (ed.), Reservoirs of Sri Lanka and their Fisheries, "
                                          "FAO Fish. Tech. Pap. 298 (1988)"},
        {"id": "D5", "claim": "Reign ends in exhaustion: heavy taxation, high-value coinage disappears, "
                              "Pandyan expedition defeated (suppressed in the chronicle)",
         "basis": "scholarship", "source": "Codrington, A Short History of Ceylon (1926); "
                                          "Nilakanta Sastri, The Colas (1955)"},
        {"id": "D6", "claim": "Chronicle portrait idealised on Niti/Arthasastra models; espionage "
                              "episodes not usable as his cognition",
         "basis": "scholarship", "source": "Geiger, Culavamsa Part I (1929), Introduction"},
        {"id": "D7", "claim": "Rajarata reservoir civilisation decays and is abandoned in the 13th-14th c.; "
                              "causes debated",
         "basis": "scholarship", "source": "Indrapala (ed.), The Collapse of the Rajarata Civilization "
                                          "(1971); Murphey, J. Asian Stud. 16 (1957)"},
        {"id": "D8", "claim": "Mapping of storms/drizzle onto label poisoning and slow drift",
         "basis": "speculation", "source": "this chapter"},
    ],
    "research_question": {"category": "continual learning and plasticity",
                          "question": ("Does overflow-only consolidation protect a continual learner from "
                                       "short label-poisoning storms better than linear multi-timescale "
                                       "consolidation, and what does it cost on sub-threshold drift?")},
    "mechanism": {
        "name": "Wana-Samudra synapse (fill-and-spill consolidation cascade)",
        "family": "local plasticity / multi-timescale synaptic consolidation",
        "signature_modules": ["wana_spillway", "tank_evaporation"],
        "closest_prior_art": ["Benna & Fusi 2016 complex synapse (Nat. Neurosci. 19:1697)",
                              "Kaplanis, Shanahan & Clopath 2018 (ICML, complex synapses in deep RL)",
                              "Fusi, Drew & Abbott 2005 cascade model (Neuron 45:599)",
                              "Laborieux et al. 2021 metaplastic BNNs (Nat. Commun. 12:2549)"],
        "overlap": "Medium",
        "prior_art_queries": ["fill and spill synapse consolidation", "threshold synaptic consolidation "
                              "continual learning", "dead-zone weight transfer multi-timescale",
                              "complex synapse beaker model deep learning",
                              "noisy label bursts continual learning robustness"],
        "contribution_type": "mechanism",
        "delta": ("One-way thresholded transfer of only the overflow, visible weight = total stored mass, "
                  "forgetting confined to the shallow store; sub-capacity bursts never reach deep storage."),
    },
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1 wana_spillway", "property_test": "C6.1, C6.3", "hypothesis": "H-SIG"},
        {"doctrine": "D2", "mechanism": "M2 samudra_cascade", "property_test": "C6.2, C6.4", "hypothesis": "H-NEC"},
        {"doctrine": "D5, D7", "mechanism": "tank_evaporation", "property_test": "C6.4", "hypothesis": "H-BLIND"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "Overflow-only consolidation keeps more of the core tasks after heavy "
                                     "label-poisoning storms than the Benna-Fusi synapse",
         "metric": "final mean held-out accuracy over 3 core tasks", "split": "shifted (heavy storms)",
         "secondary_split": "held-out (moderate storms), reported as H-SIG.id",
         "comparison": "samudra - benna_fusi", "direction": "greater", "mesi": 0.02, "seeds": 5},
        {"id": "H-NEC", "statement": "Removing the spillway threshold (all rain straight to the samudra) "
                                     "hurts more than removing the hidden bias",
         "metric": "final mean held-out accuracy", "split": "shifted (heavy storms)",
         "comparison": "ko(wana_spillway, identity) - ko(hidden_bias, zero)", "direction": "less",
         "mesi": 0.02, "seeds": 5},
        {"id": "H-BLIND", "statement": "When a rule drifts and the new evidence arrives as drizzle, the "
                                       "overflow-only synapse learns it less than Benna-Fusi",
         "condition": "drizzle: q=0.12 events/step, batch 4, 1500 steps after a converged rule",
         "grounding": "D5, D7: a surplus-capturing state did not register slow exhaustion and decay",
         "metric": "held-out accuracy on the drifted rule", "comparison": "samudra - benna_fusi",
         "direction": "less", "mesi": 0.02, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.5, "margin_over_trivial": 0.15, "shuffled_band": 0.08,
                   "gradcheck_rel_tol": 1e-5},
    "probe_predictions": [{"probe": "P5", "expected": "above baseline under noisy task sequences"},
                          {"probe": "P6", "expected": "below baseline (small updates evaporate)"},
                          {"probe": "P7", "expected": "below baseline on slow regime change"},
                          {"probe": "P9", "expected": "equal to baseline"},
                          {"probe": "P10", "expected": "equal to baseline"}],
    "dialectic_links": [{"chapter": 75, "relation": "rival",
                         "test": "none; the chronicler's Kautilyan template (0075) is excluded as evidence"}],
    "corpus_neighbors": [
        {"chapter": 269, "similarity": None, "difference": "Zhang Zai: conservative two-phase exchange of "
         "activations on a ring; here stores hold weights over time and forgetting is deliberate"},
        {"chapter": 459, "similarity": None, "difference": "Sinan: conserved load routed across layers at "
         "inference; here surplus is routed across timescales inside one synapse during learning"},
        {"chapter": 474, "similarity": None, "difference": "Turriano banks the small rounding residue; "
         "here small residue is allowed to evaporate and only the overflow is banked"},
        {"chapter": 217, "similarity": None, "difference": "al-Fihri: capped waterfall allocation of yield "
         "among beneficiaries; here the cap is a per-weight temporal cushion"},
        {"chapter": 47, "similarity": None, "difference": "Pindar: kairos gate writes into a non-decaying "
         "memory; here no gate network, the store's own fill level decides"},
        {"chapter": 180, "similarity": None, "difference": "Al-Khansa: leak versus freeze chosen by entity "
         "type; here overflow decides which store a change enters"},
        {"chapter": 296, "similarity": None, "difference": "Eleanor (N-1): portable core re-hosted across "
         "regimes; no multi-store synapse"},
    ],
    "barometer": {"cognitive_processing": ["P5 retention under task sequences"],
                  "embodied_cognition": [], "world_modeling": ["P7 tracking under drift (predicted weak)"],
                  "consciousness": [], "language_understanding": [], "emotional_intelligence": [],
                  "creativity": [], "autonomy": ["P5 continual learning without replay"]},
    "task_types": ["vector_classification"],
    "applications": [
        {"use": "continual learning on label-noisy streams with bursty corruption", "sector": "ML infrastructure",
         "dataset": "CIFAR-10 with injected symmetric label-noise bursts (as in Kim et al. ICCV 2021)"},
        {"use": "lifelong adaptation of a robot's skill models under bursty mislabelled teleoperation",
         "sector": "robotics", "dataset": "UCI Human Activity Recognition (Smartphones)"},
        {"use": "online updating of rainfall-runoff predictors when storm events bring sensor faults",
         "sector": "water resources", "dataset": "CAMELS (Addor et al. 2017)"},
    ],
    "safety_notes": "Synthetic data only; poisoning is modelled defensively as label-flip bursts.",
}

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

CHAPTER = 298
N_TASKS = 3
D_FEAT = 12
CLIP = 5.0
STORE_NAMES = ("W1", "b1", "W2", "b2")
DEFAULT_CFG = {"synapse": "samudra", "hidden": 16, "K": 3, "lr": 0.1, "cap": (0.05, 0.25),
               "tau_f": 60.0, "tau_s": 1500.0, "bf_ratio": 4.0, "bf_leak": 1e-5, "adam_lr": 0.01,
               "spill": "wana", "evap_on": True, "zeroed": ()}
ACTIVE_MUTANT = [None]


# BEGIN STANDARD UTILITIES v1.0
def logsumexp(z, axis=-1):
    m = np.max(z, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(z - m), axis=axis, keepdims=True))).squeeze(axis)


def softmax(z, axis=-1):
    e = np.exp(z - np.max(z, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def softplus(x):
    return np.logaddexp(0.0, x)


def adam_step(p, g, state, lr, b1=0.9, b2=0.999, eps=1e-8):
    state["t"] = state.get("t", 0) + 1
    m = state.setdefault("m", np.zeros_like(p))
    v = state.setdefault("v", np.zeros_like(p))
    m *= b1
    m += (1 - b1) * g
    v *= b2
    v += (1 - b2) * g * g
    mh = m / (1 - b1 ** state["t"])
    vh = v / (1 - b2 ** state["t"])
    p -= lr * mh / (np.sqrt(vh) + eps)


def clip_global(grads, max_norm):
    norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    scale = 1.0 if norm <= max_norm else max_norm / (norm + 1e-12)
    return {k: g * scale for k, g in grads.items()}, norm


def finite_difference_check(loss_fn, params, grads, rng, eps=1e-6, n_random=20, floor=1e-3):
    """Central differences on >= n_random entries per tensor plus its largest-gradient entry.
    Relative error |a-n| / max(|a|+|n|, floor); the floor turns near-zero entries into an
    absolute tolerance of about floor*tol, needed because round-off is ~1e-10 at eps=1e-6."""
    out = {}
    for name, P in params.items():
        flat = P.reshape(-1)
        gflat = grads[name].reshape(-1)
        idx = set(rng.choice(flat.size, size=min(n_random, flat.size), replace=False).tolist())
        idx.add(int(np.argmax(np.abs(gflat))))
        worst = 0.0
        for i in sorted(idx):
            orig = flat[i]
            flat[i] = orig + eps
            lp = loss_fn()
            flat[i] = orig - eps
            lm = loss_fn()
            flat[i] = orig
            num = (lp - lm) / (2 * eps)
            worst = max(worst, abs(num - gflat[i]) / max(abs(num) + abs(gflat[i]), floor))
        out[name] = (worst, len(idx))
    return out


def paired_bootstrap(diffs, rng, n_boot=2000, alpha=0.05):
    d = np.asarray(diffs, dtype=float)
    boots = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    return float(d.mean()), (float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2)))


def verdict(mean, ci, direction, mesi):
    lo, hi = ci
    if direction == "greater":
        if lo > 0 and mean >= mesi:
            return "supported"
        return "contradicted" if hi < 0 else "inconclusive"
    if hi < 0 and mean <= -mesi:
        return "supported"
    return "contradicted" if lo > 0 else "inconclusive"


def write_report(lines, path=None, payload=None):
    print("\n".join(lines))
    if path and payload is not None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
# END STANDARD UTILITIES


class NonFinite(RuntimeError):
    pass


# ----------------------------------------------------------------------------- data and tasks
def make_teacher(rng, d=D_FEAT, width=6):
    """A 'field': a softplus teacher whose threshold is its own median, so labels are balanced."""
    A = rng.normal(0.0, 1.0, (d, width)) / math.sqrt(d)
    a = rng.normal(0.0, 0.5, width)
    v = rng.normal(0.0, 1.0, width)
    ref = rng.normal(0.0, 1.0, (4000, d))
    thr = float(np.median(softplus(ref @ A + a) @ v))
    return (A, a, v, thr)


def teacher_label(T, X):
    A, a, v, thr = T
    return (softplus(X @ A + a) @ v > thr).astype(int)


def make_world(rng):
    """Three fields plus the rule field 0 drifts to in the blind-spot condition."""
    return {"teachers": [make_teacher(rng) for _ in range(N_TASKS)], "drift": make_teacher(rng)}


def sample_field(teacher, task, n, rng, flip=False, shuffle=False):
    """Clean features decide the label; the learner sees them through multiplicative sensor jitter."""
    X = rng.normal(0.0, 1.0, (n, D_FEAT))
    y = teacher_label(teacher, X)
    if flip:
        y = 1 - y
    if shuffle:
        y = rng.permutation(y)
    Xo = X * (1.0 + 0.15 * rng.normal(0.0, 1.0, X.shape))
    ctx = np.zeros((n, N_TASKS))
    ctx[:, task] = 1.0
    return np.concatenate([Xo, ctx], axis=1), y


def eval_sets(world, rng, n=500):
    return [sample_field(world["teachers"][t], t, n, rng) for t in range(N_TASKS)]


STREAMS = {  # years x seasons; each season followed by storms, each storm by a dry spell
    "moderate": dict(years=3, season=150, storms=2, storm_len=4, dry=60, final_dry=150),
    "heavy": dict(years=3, season=150, storms=3, storm_len=8, dry=60, final_dry=150),
    "quick": dict(years=2, season=80, storms=2, storm_len=6, dry=40, final_dry=80),
}


def build_stream(preset, rng):
    p = STREAMS[preset]
    ev = []
    for _ in range(p["years"]):
        for t in range(N_TASKS):
            ev.append(("train", t, p["season"]))
            for _ in range(p["storms"]):
                ev.append(("storm", int(rng.integers(N_TASKS)), p["storm_len"]))
                ev.append(("dry", None, p["dry"]))
    ev.append(("dry", None, p["final_dry"]))
    return ev


# ----------------------------------------------------------------------------- model
def build_model(in_dim, out_dim, task_type="vector_classification", rng=None, **cfg):
    """Two-layer tanh network whose every weight is a column of K stores (tanks or beakers)."""
    if task_type not in TASK_TYPES:
        raise ValueError(task_type)
    c = dict(DEFAULT_CFG)
    c.update(cfg)
    K = c["K"] if c["synapse"] in ("samudra", "benna_fusi") else 1
    h = c["hidden"]
    init = {"W1": rng.normal(0, 1 / math.sqrt(in_dim), (in_dim, h)), "b1": np.zeros(h),
            "W2": rng.normal(0, 1 / math.sqrt(h), (h, out_dim)), "b2": np.zeros(out_dim)}
    params = {}
    for name, w0 in init.items():
        t = np.zeros((K,) + w0.shape)
        if c["synapse"] == "benna_fusi":
            t[:] = w0            # every beaker starts level with the inherited weight: no flow
        else:
            t[-1] = w0           # the inherited network sits in the deepest store
        params[name] = t
    return {"params": params, "cfg": c, "K": K, "in_dim": in_dim, "out_dim": out_dim,
            "task_type": task_type, "adam": {n: {} for n in params}}


def visible(model, name):
    """Weight used by the network: total stored mass (samudra, single) or first beaker (Benna-Fusi)."""
    if name in model["cfg"]["zeroed"]:
        return np.zeros(model["params"][name].shape[1:])
    t = model["params"][name]
    return t[0] if model["cfg"]["synapse"] == "benna_fusi" else t.sum(axis=0)


def forward(model, X):
    W1, b1, W2, b2 = (visible(model, n) for n in STORE_NAMES)
    H = np.tanh(X @ W1 + b1)
    return H @ W2 + b2, H


def loss_and_grads(model, batch):
    X, y = batch
    z, H = forward(model, X)
    n = X.shape[0]
    loss = float(np.mean(logsumexp(z) - z[np.arange(n), y]))
    dz = softmax(z)
    dz[np.arange(n), y] -= 1.0
    dz /= n
    W2 = visible(model, "W2")
    da = (dz @ W2.T) * (1.0 - H * H)
    gv = {"W1": X.T @ da, "b1": da.sum(0), "W2": H.T @ dz, "b2": dz.sum(0)}
    grads = {}
    for name, P in model["params"].items():
        g = np.zeros_like(P)
        if name in model["cfg"]["zeroed"]:
            pass
        elif model["cfg"]["synapse"] == "benna_fusi":
            g[0] = gv[name]
        else:
            g[:] = gv[name]      # the loss sees tanks only through their sum
        grads[name] = g
    if ACTIVE_MUTANT[0] == "zero_grad_W1":
        grads["W1"][:] = 0.0
    return loss, grads


def dead_zone(s, c):
    """Wana: the part of a tank's level beyond +/- capacity; exactly what overtops the spillway."""
    return np.sign(s) * np.maximum(np.abs(s) - c, 0.0)


def samudra_rates(cfg, K):
    """Evaporation per store: fast in the village tank, slow in the middle, zero in the samudra."""
    lam = np.array([1.0 / cfg["tau_f"]] + [1.0 / cfg["tau_s"]] * (K - 2) + [0.0])
    return lam if cfg["evap_on"] else np.zeros(K)


def samudra_rain(tanks, rain, cfg):
    """Rain on the shallow tank, overflow cascades one way downstream, then evaporation."""
    tanks[0] += rain
    caps = {"wana": cfg["cap"], "identity": (0.0,) * (tanks.shape[0] - 1)}.get(cfg["spill"])
    if caps is not None:
        for k in range(tanks.shape[0] - 1):
            spill = dead_zone(tanks[k], caps[k])
            tanks[k] -= spill
            tanks[k + 1] += spill
    lam = samudra_rates(cfg, tanks.shape[0])
    tanks *= (1.0 - lam).reshape((-1,) + (1,) * (tanks.ndim - 1))


# ----------------------------------------------------------------------------- baselines and rival
def bf_matrix(cfg):
    """Benna-Fusi beaker chain, explicit Euler: sizes 1, r, r^2; tube widths chosen so that the
    shallow relaxation time is tau_f and the deep one tau_s (timescales matched to the tanks)."""
    r = cfg["bf_ratio"]
    C = np.array([1.0, r, r * r])
    g01 = C[0] / cfg["tau_f"]
    g12 = C[1] / cfg["tau_s"]
    D = np.zeros((3, 3))
    D[0, 0], D[0, 1] = -g01, g01
    D[1, 0], D[1, 1], D[1, 2] = g01, -g01 - g12, g12
    D[2, 1], D[2, 2] = g12, -g12 - cfg["bf_leak"]
    return np.eye(3) + D / C[:, None]


def bf_rain(beakers, rain, cfg):
    beakers[0] += rain
    beakers[:] = np.tensordot(bf_matrix(cfg), beakers, axes=(1, 0))


def rain_step(model, grads):
    """One learning event: clipped SGD 'rain' routed by the synapse rule (Adam for the reference)."""
    cfg = model["cfg"]
    gv = {n: grads[n][0] for n in model["params"]}
    gv, _ = clip_global(gv, CLIP)
    lr = 0.0 if ACTIVE_MUTANT[0] == "zero_lr" else cfg["lr"]
    sign = 1.0 if ACTIVE_MUTANT[0] == "sign_flip" else -1.0
    for name, P in model["params"].items():
        if name in cfg["zeroed"]:
            continue
        rain = sign * lr * gv[name]
        if cfg["synapse"] == "samudra":
            samudra_rain(P, rain, cfg)
        elif cfg["synapse"] == "benna_fusi":
            bf_rain(P, rain, cfg)
        elif cfg["synapse"] == "adam":
            alr = 0.0 if ACTIVE_MUTANT[0] == "zero_lr" else cfg["adam_lr"]
            adam_step(P[0], -sign * gv[name], model["adam"][name], alr)
        else:
            P[0] += rain


def dry_spell(model, n):
    """n steps without data: tanks evaporate, beakers keep exchanging, plain weights stay put."""
    if n <= 0:
        return
    cfg = model["cfg"]
    for name, P in model["params"].items():
        if cfg["synapse"] == "samudra":
            P *= ((1.0 - samudra_rates(cfg, P.shape[0])) ** n).reshape((-1,) + (1,) * (P.ndim - 1))
        elif cfg["synapse"] == "benna_fusi":
            P[:] = np.tensordot(np.linalg.matrix_power(bf_matrix(cfg), n), P, axes=(1, 0))


# ----------------------------------------------------------------------------- registries
TASK_TYPES = ("vector_classification",)
MUTANTS = {"sign_flip": "rain falls uphill (update sign reversed)",
           "zero_lr": "learning rate set to zero",
           "zero_grad_W1": "analytic gradient of W1 zeroed"}
KNOCKOUTS = {("wana_spillway", "identity"): {"spill": "identity"},
             ("wana_spillway", "zero"): {"spill": "zero"},
             ("tank_evaporation", "zero"): {"evap_on": False},
             ("hidden_bias", "zero"): {"zeroed": ("b1",)},
             ("readout_bias", "zero"): {"zeroed": ("b2",)}}


def modules(model):
    syn = model["cfg"]["synapse"]
    out = {"hidden_layer": {"params": ["W1"], "role": "dense tanh layer", "signature": False},
           "hidden_bias": {"params": ["b1"], "role": "hidden bias", "signature": False},
           "readout": {"params": ["W2"], "role": "linear softmax readout", "signature": False},
           "readout_bias": {"params": ["b2"], "role": "readout bias", "signature": False}}
    if syn == "samudra":
        out["wana_spillway"] = {"params": list(STORE_NAMES), "signature": True,
                                "role": "one-way dead-zone transfer of overflow between per-weight stores"}
        out["tank_evaporation"] = {"params": list(STORE_NAMES), "signature": True,
                                   "role": "per-store exponential decay, fast shallow, none in deepest"}
    elif syn == "benna_fusi":
        out["beaker_exchange"] = {"params": list(STORE_NAMES), "signature": False,
                                  "role": "linear two-way diffusion along a store chain (rival)"}
    return out


def clone(model):
    """Independent copy: stores, optimiser moments and config (the stdlib allowlist has no copy)."""
    m = dict(model)
    m["params"] = {n: P.copy() for n, P in model["params"].items()}
    m["adam"] = {n: {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in st.items()}
                 for n, st in model["adam"].items()}
    m["cfg"] = dict(model["cfg"])
    return m


def knockout(model, name, mode):
    if (name, mode) not in KNOCKOUTS:
        raise ValueError(f"no knockout {name}/{mode}")
    m = clone(model)
    m["cfg"].update(KNOCKOUTS[(name, mode)])
    return m


def n_params(model):
    return int(sum(P.size for P in model["params"].values()))


def predict_proba(model, X):
    return softmax(forward(model, X)[0])


def predict(model, X):
    return np.argmax(predict_proba(model, X), axis=1)


def hidden_states(model, X):
    return forward(model, X)[1]


def accuracy(model, data):
    X, y = data
    return float(np.mean(predict(model, X) == y))


def check_finite(model):
    for n, P in model["params"].items():
        if not np.all(np.isfinite(P)):
            raise NonFinite(n)


# ----------------------------------------------------------------------------- training
def train_step(model, batch):
    loss, grads = loss_and_grads(model, batch)
    rain_step(model, grads)
    return loss


def fit(model, data, budget, rng, batch=16):
    X, y = data
    losses = []
    for _ in range(budget):
        idx = rng.integers(0, X.shape[0], size=batch)
        losses.append(train_step(model, (X[idx], y[idx])))
    check_finite(model)
    return losses


def flat_visible(model):
    return np.concatenate([visible(model, n).ravel() for n in STORE_NAMES])


def run_stream(model, world, events, rng, evals):
    """Seasons, storms and dry spells. Returns final mean held-out accuracy and, for every storm,
    the fraction of its displacement still present after the next dry spell, measured against a
    twin that lived through the same dry spell without the storm."""
    persist, pending = [], None
    for kind, t, n in events:
        if kind == "dry":
            dry_spell(model, n)
            if pending is not None:  # counterfactual twin: same dry spell, no storm
                twin, w0, w1 = pending
                dry_spell(twin, n)
                gap = np.linalg.norm(flat_visible(model) - flat_visible(twin))
                persist.append(float(gap / (np.linalg.norm(w1 - w0) + 1e-12)))
                pending = None
            continue
        if kind == "storm":
            twin = clone(model)
            w0 = flat_visible(model)
            for _ in range(n):
                train_step(model, sample_field(world["teachers"][t], t, 16, rng, flip=True))
            pending = (twin, w0, flat_visible(model))
            continue
        for _ in range(n):  # a season waters every field: mixed batches of all three
            parts = [sample_field(world["teachers"][f], f, 6, rng) for f in range(N_TASKS)]
            train_step(model, (np.concatenate([q[0] for q in parts]), np.concatenate([q[1] for q in parts])))
    check_finite(model)
    return float(np.mean([accuracy(model, e) for e in evals])), persist


def run_drizzle(model, world, rng, drift_eval, quick=False):
    """Field 0 is learned densely, then its rule drifts; the new evidence falls as drizzle."""
    pre, steps, q = (200, 600, 0.12) if quick else (400, 1500, 0.12)
    for _ in range(pre):
        train_step(model, sample_field(world["teachers"][0], 0, 16, rng))
    dry_spell(model, 100)
    dry = 0
    for _ in range(steps):
        if rng.random() < q:
            dry_spell(model, dry)
            dry = 0
            train_step(model, sample_field(world["drift"], 0, 4, rng))
        else:
            dry += 1
    dry_spell(model, dry)
    check_finite(model)
    return accuracy(model, drift_eval)


def seed_rngs(base, idx):
    """Fresh generators per (base seed, index): every model in a comparison sees the same world,
    the same inherited initial weights and the same stream (common random numbers)."""
    return [np.random.default_rng(s) for s in np.random.SeedSequence([base, idx]).spawn(4)]


def experiment(base, idx, condition, cfg, quick=False):
    world_rng, eval_rng, stream_rng, init_rng = seed_rngs(base, idx)
    world = make_world(world_rng)
    model = build_model(D_FEAT + N_TASKS, 2, "vector_classification", init_rng, **cfg)
    if condition == "drizzle":
        drift_eval = sample_field(world["drift"], 0, 500, eval_rng)
        return run_drizzle(model, world, stream_rng, drift_eval, quick), []
    evals = eval_sets(world, eval_rng)
    preset = "quick" if quick else condition
    return run_stream(model, world, build_stream(preset, stream_rng), stream_rng, evals)


TUNING_GRID = {  # 3 x 3 for each store model: shallow timescale x its own shape knob
    "samudra": [{"tau_f": t, "cap": (c, 0.25)} for t in (20.0, 60.0, 180.0) for c in (0.01, 0.02, 0.05)],
    "benna_fusi": [{"tau_f": t, "bf_ratio": r} for t in (20.0, 60.0, 180.0) for r in (1.25, 2.0, 4.0)],
}


def tune(base, quick):
    """Same-sized grid for both store models, chosen on a validation seed and the moderate split."""
    if quick:
        return {"samudra": {}, "benna_fusi": {}}, {}
    chosen, table = {}, {}
    for syn, grid in TUNING_GRID.items():
        scores = [experiment(base, 1000, "moderate", dict({"synapse": syn}, **g))[0] for g in grid]
        chosen[syn] = grid[int(np.argmax(scores))]
        table[syn] = list(zip(grid, scores))
    return chosen, table


# ----------------------------------------------------------------------------- correctness tests
def c3_data(base):
    world_rng, eval_rng, stream_rng, init_rng = seed_rngs(base, 2000)
    world = make_world(world_rng)
    train = sample_field(world["teachers"][0], 0, 512, stream_rng)
    held = sample_field(world["teachers"][0], 0, 500, eval_rng)
    return world, train, held, init_rng, stream_rng


def test_gradcheck(base, syns=("samudra", "benna_fusi")):
    world, train, _, init_rng, rng = c3_data(base)
    batch = (train[0][:32], train[1][:32])
    res, worst, n_tensors = {}, 0.0, 0
    for syn in syns:
        model = build_model(D_FEAT + N_TASKS, 2, "vector_classification", init_rng, synapse=syn)
        for stage in ("init", "after_60_steps"):
            if stage == "after_60_steps":
                fit(model, train, 60, rng)
            _, grads = loss_and_grads(model, batch)
            out = finite_difference_check(lambda: loss_and_grads(model, batch)[0], model["params"], grads, rng)
            for name, (err, _) in out.items():
                res[f"{syn}/{stage}/{name}"] = err
                worst = max(worst, err)
                n_tensors += 1
    tol = MIND_CARD["thresholds"]["gradcheck_rel_tol"]
    return worst <= tol, worst, n_tensors, res


def test_learning(base, shuffle=0):
    world, train, held, init_rng, rng = c3_data(base)
    if shuffle:  # permutation number `shuffle` of the training labels
        prng = np.random.default_rng([base, 3000 + shuffle])
        train = (train[0], prng.permutation(train[1]))
    model = build_model(D_FEAT + N_TASKS, 2, "vector_classification", init_rng)
    l0 = loss_and_grads(model, train)[0]
    fit(model, train, 300, rng)
    l1 = loss_and_grads(model, train)[0]
    acc = accuracy(model, held)
    trivial = max(np.mean(held[1]), 1 - np.mean(held[1]))
    return l0, l1, acc, float(trivial)


def test_determinism(base):
    """Same seed, same stream, same result; stores and hidden activations finite after training."""
    a = experiment(base, 7, "moderate", {}, quick=True)
    b = experiment(base, 7, "moderate", {}, quick=True)
    _, train, held, init_rng, rng = c3_data(base)
    m = build_model(D_FEAT + N_TASKS, 2, "vector_classification", init_rng)
    fit(m, train, 100, rng)
    acts_ok = bool(np.all(np.isfinite(hidden_states(m, held[0]))))
    return a == b and all(np.isfinite(a[1])) and bool(np.isfinite(a[0])) and acts_ok


def test_split_integrity(base):
    _, train, held, _, _ = c3_data(base)
    h = lambda X: {hashlib.sha1(np.round(r, 10).tobytes()).hexdigest() for r in X}
    return len(h(train[0]) & h(held[0])) == 0


def prop(check, broken, trials, rng):
    """A property holds on random states and a deliberately broken variant is caught."""
    ok = all(check(rng, None) for _ in range(trials))
    caught = any(not check(rng, broken) for _ in range(trials))
    return ok and caught, ok, caught


def test_properties(rng):
    cfg = dict(DEFAULT_CFG)
    caps = cfg["cap"]

    def rand_state(r, scale):
        return r.normal(0, scale, (3, 7)) * r.choice([0.1, 1.0, 10.0])

    def conservation(r, broken):
        s = rand_state(r, 0.3)
        rain = r.normal(0, 0.3, 7)
        total = s.sum(0) + rain
        s[0] += rain
        for k in range(2):
            spill = dead_zone(s[k], caps[k])
            s[k] -= spill
            s[k + 1] += spill * (0.95 if broken else 1.0)
        return np.allclose(s.sum(0), total, atol=1e-12, rtol=0)

    def capacity(r, broken):
        s = rand_state(r, 1.0)
        for k in range(2):
            spill = dead_zone(s[k], caps[k]) * (0.5 if broken else 1.0)
            s[k] -= spill
            s[k + 1] += spill
        return all(np.all(np.abs(s[k]) <= caps[k] + 1e-15) for k in range(2))

    def containment(r, broken):
        s = rand_state(r, 1.0)
        s[0] = r.uniform(-0.5, 0.5, 7) * caps[0]
        burst = r.uniform(-0.45, 0.45, (5, 7)) * caps[0] / 5.0
        a, b = s.copy(), s.copy()
        for rain in burst:
            if broken:
                bf_rain(a, rain, cfg)
                bf_rain(b, 0.0 * rain, cfg)
            else:
                samudra_rain(a, rain, cfg)
                samudra_rain(b, 0.0 * rain, cfg)
        return np.array_equal(a[1:], b[1:])

    def contraction(r, broken):
        s = rand_state(r, 1.0)
        lam = samudra_rates(cfg, 3) * (-1.0 if broken else 1.0)
        t = s * (1.0 - lam)[:, None]
        return bool(np.all(np.abs(t) <= np.abs(s) + 1e-15) and np.all(np.sign(t) * np.sign(s) >= 0))

    def bf_mass(r, broken):
        c2 = dict(cfg, bf_leak=0.0)
        u = rand_state(r, 1.0)
        C = np.array([1.0, c2["bf_ratio"], c2["bf_ratio"] ** 2])
        M = bf_matrix(c2)
        if broken:
            M = M.copy()
            M[1, 0] *= 1.3
        v = M @ u
        return np.allclose(C @ v, C @ u, atol=1e-10)

    names = {"C6.1 conservation of spill": conservation, "C6.2 capacity bound after spill": capacity,
             "C6.3 sub-capacity containment": containment, "C6.4 evaporation contracts": contraction,
             "C6.5 beaker mass conservation (rival)": bf_mass}
    return {n: prop(f, True, 200, rng) for n, f in names.items()}


def test_mutants(base):
    """Each registered learning-breaking mutant must make the gradient check or learning test fail."""
    caught = {}
    saved = ACTIVE_MUTANT[0]
    thr = MIND_CARD["thresholds"]
    for name in MUTANTS:
        ACTIVE_MUTANT[0] = name
        g_ok = test_gradcheck(base, syns=("samudra",))[0]
        l0, l1, acc, trivial = test_learning(base)
        learn_ok = (l1 <= (1 - thr["loss_drop_fraction"]) * l0) and (acc - trivial >= thr["margin_over_trivial"])
        caught[name] = (not g_ok) or (not learn_ok)
    ACTIVE_MUTANT[0] = saved
    return caught


# ----------------------------------------------------------------------------- hypothesis runs
def run_hypotheses(base, n_seeds, chosen, quick):
    """Paired runs over seeds. Returns per-seed metrics keyed by (condition, arm)."""
    arms = {
        "moderate": {"samudra": {"synapse": "samudra"}, "benna_fusi": {"synapse": "benna_fusi"},
                     "adam_finetune": {"synapse": "adam"}},
        "heavy": {"samudra": {"synapse": "samudra"}, "benna_fusi": {"synapse": "benna_fusi"},
                  "adam_finetune": {"synapse": "adam"}},
        "drizzle": {"samudra": {"synapse": "samudra"}, "benna_fusi": {"synapse": "benna_fusi"},
                    "plain_sgd": {"synapse": "single"}},
    }
    for (mod, mode), change in KNOCKOUTS.items():
        arms["heavy"][f"ko:{mod}/{mode}"] = dict({"synapse": "samudra"}, **change)
    res, persist = {}, {}
    for cond, table in arms.items():
        for arm, cfg in table.items():
            cfg = dict(chosen.get(cfg["synapse"], {}), **cfg)
            vals = []
            for i in range(n_seeds):
                acc, pers = experiment(base, i, cond, cfg, quick)
                vals.append(acc)
                if cond == "moderate" and arm in ("samudra", "benna_fusi", "adam_finetune"):
                    persist.setdefault(arm, []).extend(pers)
            res[(cond, arm)] = vals
    return res, persist


def evaluate_hypotheses(res, rng, quick):
    spec = {h["id"]: h for h in MIND_CARD["hypotheses"]}
    pairs = {"H-SIG": (("heavy", "samudra"), ("heavy", "benna_fusi")),
             "H-SIG.id": (("moderate", "samudra"), ("moderate", "benna_fusi")),
             "H-NEC": (("heavy", "ko:wana_spillway/identity"), ("heavy", "ko:hidden_bias/zero")),
             "H-BLIND": (("drizzle", "samudra"), ("drizzle", "benna_fusi"))}
    out = []
    for hid, (a, b) in pairs.items():
        h = spec[hid.split(".")[0]]
        diffs = np.array(res[a]) - np.array(res[b])
        mean, ci = paired_bootstrap(diffs, rng)
        v = "not evaluated" if quick or len(diffs) < 5 else verdict(mean, ci, h["direction"], h["mesi"])
        out.append({"id": hid, "metric": h["metric"], "comparison": f"{a[1]} - {b[1]} [{a[0]}]",
                    "mean_diff": mean, "ci95": list(ci), "mesi": h["mesi"], "n_seeds": int(len(diffs)),
                    "direction": h["direction"], "verdict": v})
    kos = []
    for (mod, mode) in KNOCKOUTS:
        d = np.array(res[("heavy", f"ko:{mod}/{mode}")]) - np.array(res[("heavy", "samudra")])
        mean, ci = paired_bootstrap(d, rng)
        kos.append({"module": f"{mod} ({mode})", "signature": mod in ("wana_spillway", "tank_evaporation"),
                    "metric_change": mean, "ci95": list(ci)})
    return out, kos


# ----------------------------------------------------------------------------- report
def fmt(x):
    return f"{x:+.4f}" if isinstance(x, float) else str(x)


def build_report(ctx):
    L = [f"=== VERIFIED REPORT · chapter {CHAPTER:04d} ===",
         f"file: {os.path.basename(__file__)}   card_revision: {MIND_CARD['card_revision']}",
         f"environment: python {sys.version.split()[0]} · numpy {np.__version__}",
         f"seeds: base {ctx['base']}, indices {ctx['seeds']}   runtime: {ctx['runtime']:.1f} s",
         f"n_params (stored scalars): samudra {ctx['np']['samudra']} · benna_fusi {ctx['np']['benna_fusi']}"
         f" · single-store {ctx['np']['single']}",
         f"gradcheck: {ctx['gc']['checks']} tensor checks (4 of 4 tensors x samudra, benna_fusi x init,"
         f" after 60 steps); max rel err {ctx['gc']['worst']:.2e}; {'PASS' if ctx['gc']['passed'] else 'FAIL'}",
         "correctness:"]
    for c in ctx["correct"]:
        L.append(f"  {c['id']:<5} {c['name']:<38} {'PASS' if c['passed'] else 'FAIL'}  {c['detail']}")
    m = ctx["mut"]
    L.append(f"mutants: {sum(m.values())}/{len(m)} detected (score {sum(m.values()) / len(m):.2f}) "
             + ", ".join(f"{k}={'caught' if v else 'MISSED'}" for k, v in m.items()))
    for syn, tab in ctx["tune"].items():
        best = max(a for _, a in tab)
        L.append(f"tuning {syn} (validation seed, moderate, 9 configs): best {best:.4f} at {ctx['chosen'][syn]}")
    L.append("condition means (held-out accuracy, mean +/- sd over seeds):")
    for (cond, arm), vals in ctx["res"].items():
        L.append(f"  {cond:<9} {arm:<30} {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
    for arm, p in ctx["persist"].items():
        L.append(f"storm persistence after dry spell ({arm}, moderate): median {np.median(p):.3f}")
    L.append("hypotheses:")
    for h in ctx["hyp"]:
        L.append(f"  {h['id']:<8} {h['comparison']:<58} diff {fmt(h['mean_diff'])} "
                 f"CI [{h['ci95'][0]:+.4f}, {h['ci95'][1]:+.4f}] mesi {h['mesi']} n={h['n_seeds']} -> {h['verdict']}")
    L.append("knockouts (heavy storms, ko - full samudra):")
    roles = modules(build_model(D_FEAT + N_TASKS, 2, "vector_classification", np.random.default_rng(0)))
    for k in ctx["ko"]:
        role = roles[k["module"].split()[0]]["role"]
        L.append(f"  {k['module']:<26} signature={str(k['signature']):<5} {fmt(k['metric_change'])} "
                 f"CI [{k['ci95'][0]:+.4f}, {k['ci95'][1]:+.4f}]  ({role})")
    L.append(f"task_types: {', '.join(TASK_TYPES)}")
    L.append(f"exit_code: {ctx['exit']}")
    L.append("=== END REPORT ===")
    return L


def json_payload(ctx):
    return {"schema_version": "1.0", "chapter": CHAPTER, "file": os.path.basename(__file__),
            "card_revision": MIND_CARD["card_revision"],
            "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
            "seeds": ctx["seeds"], "seed_base": ctx["base"], "runtime_s": round(ctx["runtime"], 2),
            "n_params": ctx["np"]["samudra"],
            "gradcheck": {"tensors_checked": 4, "tensors_total": 4, "checks": ctx["gc"]["checks"],
                          "max_rel_error": ctx["gc"]["worst"], "checked_at": ["init", "after_training_steps"],
                          "passed": ctx["gc"]["passed"]},
            "correctness": [{"id": c["id"], "name": c["name"], "passed": bool(c["passed"]), "detail": c["detail"]}
                            for c in ctx["correct"]],
            "mutants": {"detected": int(sum(ctx["mut"].values())), "total": len(ctx["mut"]),
                        "score": sum(ctx["mut"].values()) / len(ctx["mut"])},
            "hypotheses": [{k: h[k] for k in ("id", "metric", "mean_diff", "ci95", "mesi", "n_seeds", "verdict")}
                           for h in ctx["hyp"]],
            "knockouts": ctx["ko"], "task_types": list(TASK_TYPES), "exit_code": ctx["exit"]}


# ----------------------------------------------------------------------------- real-data bridge
def data_bridge(path, base):
    """Optional: a local CSV (numeric features, last column 0/1 label) run through storms."""
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    X, y = arr[:, :-1], arr[:, -1].astype(int)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    rng = np.random.default_rng(base)
    idx = rng.permutation(len(y))
    cut = int(0.7 * len(y))
    tr, te = idx[:cut], idx[cut:]
    for syn in ("samudra", "benna_fusi"):
        model = build_model(X.shape[1], 2, "vector_classification", np.random.default_rng(base), synapse=syn)
        r = np.random.default_rng(base + 1)
        fit(model, (X[tr], y[tr]), 400, r)
        for _ in range(3):
            fit(model, (X[tr], 1 - y[tr]), 10, r)
            dry_spell(model, 60)
        print(f"data bridge {syn}: held-out accuracy after 3 label-flip storms {accuracy(model, (X[te], y[te])):.4f}")


# ----------------------------------------------------------------------------- entry point
def correctness_suite(base):
    thr = MIND_CARD["thresholds"]
    gc = test_gradcheck(base)
    l0, l1, acc, triv = test_learning(base)
    shuf = [test_learning(base, shuffle=k)[2:] for k in range(1, 6)]
    sacc, striv = float(np.mean([a for a, _ in shuf])), shuf[0][1]
    band = thr["shuffled_band"]
    c4_ok = abs(sacc - striv) <= band
    props = test_properties(np.random.default_rng(base + 5))
    rows = [{"id": "C1", "name": "gradient check", "passed": gc[0], "detail": f"max rel err {gc[1]:.2e}"},
            {"id": "C2", "name": "determinism and finiteness", "passed": test_determinism(base), "detail": ""},
            {"id": "C3", "name": "learning", "detail": f"loss {l0:.3f}->{l1:.3f}, acc {acc:.3f} vs trivial {triv:.3f}",
             "passed": (l1 <= (1 - thr["loss_drop_fraction"]) * l0) and (acc - triv >= thr["margin_over_trivial"])},
            {"id": "C4", "name": "shuffled-label control", "passed": c4_ok,
             "detail": f"mean of 5 permutations {sacc:.3f} (range {min(a for a, _ in shuf):.3f}-"
                       f"{max(a for a, _ in shuf):.3f}) vs trivial {striv:.3f} (band {band})"}]
    for name, (ok, holds, caught) in props.items():
        rows.append({"id": name.split()[0], "name": name.split(" ", 1)[1], "passed": ok,
                     "detail": f"holds={holds}, broken variant caught={caught}"})
    rows.append({"id": "C7", "name": "split integrity", "passed": test_split_integrity(base), "detail": "sha1 rows"})
    mut = test_mutants(base)
    rows.append({"id": "C5", "name": "mutant detection", "passed": all(mut.values()),
                 "detail": f"{sum(mut.values())}/{len(mut)}"})
    return rows, {"passed": bool(gc[0]), "worst": float(gc[1]), "checks": int(gc[2])}, mut


def main(argv=None):
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=297)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--card", action="store_true")
    ap.add_argument("--mutant", type=str, default=None)
    ap.add_argument("--data", type=str, default=None)
    try:
        a = ap.parse_args(argv)
    except SystemExit:
        return 2
    if a.card:
        print(json.dumps(MIND_CARD, indent=2))
        return 0
    if a.mutant is not None and a.mutant not in MUTANTS:
        print(f"unknown mutant {a.mutant}; choose from {sorted(MUTANTS)}")
        return 2
    ACTIVE_MUTANT[0] = a.mutant
    t0 = time.time()
    n_seeds = 1 if a.quick else a.seeds
    budget = 20.0 if a.quick else 180.0
    try:
        rows, gc, mut = correctness_suite(a.seed)
        chosen, table = tune(a.seed, a.quick)
        res, persist = run_hypotheses(a.seed, n_seeds, chosen, a.quick)
    except NonFinite as err:
        print(f"non-finite values in {err}")
        return 4
    hyp, kos = evaluate_hypotheses(res, np.random.default_rng(a.seed + 9), a.quick)
    runtime = time.time() - t0
    rows.append({"id": "C8", "name": "runtime budget", "passed": runtime <= budget,
                 "detail": f"{runtime:.1f} s <= {budget:.0f} s"})
    probe = {s: n_params(build_model(D_FEAT + N_TASKS, 2, "vector_classification", np.random.default_rng(0),
                                     synapse=s)) for s in ("samudra", "benna_fusi", "single")}
    code = 0 if all(r["passed"] for r in rows if r["id"] != "C8") else 1
    if code == 0 and runtime > budget:
        code = 3
    ctx = {"base": a.seed, "seeds": list(range(n_seeds)), "runtime": runtime, "np": probe,
           "gc": gc, "correct": sorted(rows, key=lambda r: r["id"]), "mut": mut, "chosen": chosen,
           "tune": table, "res": res, "persist": persist, "hyp": hyp, "ko": kos, "exit": code}
    write_report(build_report(ctx), a.json, json_payload(ctx))
    if a.data:
        data_bridge(a.data, a.seed)
    else:
        print("real-data bridge skipped (no --data PATH given)")
    return code


if __name__ == "__main__":
    sys.exit(main())

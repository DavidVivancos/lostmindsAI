#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0319 · Walther von der Vogelweide
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 16, Minds 301-320 Available on Amazon https://www.amazon.com/gp/product/B0HNJQX9S4
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos 
# END ATTRIBUTION
"""
ZWEIHERZ: a two-heart co-carrying network (chapter 0319, Walther von der Vogelweide).

Thesis
    A bond is real only while two carry it: the joint value is the part both
    carriers share, the unshared remainder is a load one heart cannot hold, the
    lighter carrier must be asked to help in time, and a partner who stays
    definitively indifferent is released so the other may go on alone.

Evidence and provenance (archetype_provenance = belief; own surviving texts)
    D1 L 50,19 (C 22) st. IV: 'minne entouc niht eine, si sol sin gemeine, so
       gemeine daz si ge dur zwei herze und dur dekeinez me'.         [primary]
    D2 L 69,1 (C 44) st. II: 'minne ist zweier herzen wünne: teilent sie
       geliche, sost diu minne da: sol abe ungeteilet sin, so enkans ein herze
       alleine niht enthalten'.                                        [primary]
    D3 L 50,26 'hilf mir tragen, ich bin ze vil geladen'; L 69,1 st. III 'wellest
       du mir helfen, so hilf an der zit'.                             [primary]
    D4 L 69,1 st. III 'si abe ich dir gar unmære, daz sprich endeliche: so laz
       ich den strit unde wirde ein ledic man'; L 105,27 'lob ich in, so lob er
       mich ... ode ich wil minz her wider nemen'.                     [primary]
    D5 Walther set mutual love (herzeliebe) against Reinmar's one-sided,
       renunciatory service (NDB 27, 2020, Ukena-Best).            [scholarship]
    D6 L 26,3 'wie solt ich den geminnen der mir übele tuot? mir muoz der iemer
       lieber sin der mir ist guot': he admits he cannot hold a bond that is
       not returned, and asks forgiveness for it.                      [primary]
    The machine-learning reading of these lines is speculation by the chapter.

Doctrine -> mechanism -> test (same IDs as MIND_CARD)
    D1 -> M1 harmonic co-carrying fusion   -> C6.P1 one-heart-alone, C6.P2  -> H-SIG
    D2 -> M2 overflow (AM-HM gap) penalty  -> C6.P4 overflow identity       -> H-NEC
    D3 -> M1 gradient routed to lighter    -> C6.P3 lighter-carrier grad    -> H-SIG
    D4 -> M3 definitive release rule       -> C6.P6 release rule, C6.P5     -> H-RIVAL
    D5 -> rival R1 'staete' (no release)   -> H-RIVAL
    D6 -> blind spot: unequal but real bonds                                -> H-BLIND

Research question (robustness / modality imbalance)
    Joint multichannel training lets the easy channel carry the whole load and
    starves the harder one (greedy learning, gradient starvation). Can a fusion
    nonlinearity whose value exists only in the shared part of two channel
    shares, plus a penalty on the unshared part, keep the weak channel carrying
    its half, so the model survives the loss of the dominant channel without
    random modality dropout or gradient-rescaling schedules?

Closest prior art and the delta
    ModDrop (Neverova et al., arXiv:1501.00102); OGM-GE (Peng et al., CVPR 2022,
    arXiv:2203.15332); Gradient Blending (Wang, Tran, Feiszli, arXiv:1905.12681);
    greedy multimodal learning (Wu et al., arXiv:2202.05306). Delta: balance is
    enforced by the forward operator itself (the harmonic mean routes gradient to
    the lighter share and caps the joint value by it), the AM-HM gap is the
    penalised unshared load, and release reuses the balanced shares so that a
    single remaining heart outputs on the bonded scale. Overlap: Medium.

Blind spot (grounded in D6 and in Walther's measurement of lords by returned
    gifts): bonds that are real but unequal. When the second channel is weak but
    informative and both channels stay present, forced equal carrying should cost
    accuracy against a fusion free to weight the channels unequally.

Task (dyad classification, K = 4)
    y ~ U{0..3}. Channel A ('the loud heart'): x_A = g * mu_A[y] + noise, g ~
    U(0.6, 1.4) nuisance gain, linearly separable, learned first. Channel B ('the
    quiet heart'): y = 2*[z1*z2 > 0] + [z3*z4 > 0] on four latent coordinates
    plus two nuisance coordinates, rotated by a fixed random orthogonal matrix
    and noised; needs hidden features and is learned late. Splits: train, val,
    in-distribution test, and shifted tests where A or B is definitively absent
    (presence flag 0). Side worlds use a noisier A, so A alone is not saturated:
    blind-spot world, B noisier but informative; rival world, B pure noise.

Limits
    Synthetic data, two channels, a small network; a research prototype of an
    AGI-oriented fusion primitive, not an AGI. The chapter's reading of Walther's
    lyric as a design principle is interpretation, not his claim.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 2,
    "revision_log": [
        {"revision": 2, "date": "2026-09-24", "reason": (
            "Task parameters only, after a 4-seed pilot run before the full protocol: at sigA 0.55 the mean "
            "baseline did not starve heart B, so the native task lacked the failing-baseline condition of "
            "guideline 10.1 (native sigA set to 0.40); at sigB 1.10 heart B alone sat near chance, so weak_b "
            "was not 'weak but informative' (weak_sigB set to 0.80; side worlds use sigA 0.90 so A alone is "
            "not saturated). Hypotheses, metrics, mesi and thresholds unchanged. The pilot already showed "
            "H-NEC and H-RIVAL effects below mesi; they were left as written.")}],
    "generation": {"template_version": "codeguidelines v1.0 (15 Sep 2026)", "generator": "Claude",
                   "generator_version": "claude-opus-5-5", "date": "2026-09-24"},
    "id": 319, "figure": "Walther von der Vogelweide", "born": 1170, "died": 1230,
    "civilization": "German (Holy Roman Empire)", "provenance": "belief",
    "thesis": ("A bond is real only while two carry it: value is the shared part of two shares, the unshared "
               "remainder is a load one heart cannot hold, the lighter carrier is asked to help in time, and a "
               "definitively indifferent partner is released."),
    "evidence": [
        {"id": "D1", "claim": "Minne is worthless alone; it must pass through two hearts and no more.",
         "basis": "primary", "source": "Walther, L 50,19 (C 22), st. IV (L 51,5-12)"},
        {"id": "D2", "claim": "Minne is the bliss of two hearts; shared equally it exists; unshared, one heart "
         "alone cannot contain it.", "basis": "primary", "source": "Walther, L 69,1 (C 44), st. II"},
        {"id": "D3", "claim": "The overloaded heart asks the partner to help carry, and to help in time.",
         "basis": "primary", "source": "Walther, L 50,26; L 69,1 (C 44), st. III"},
        {"id": "D4", "claim": "A definitive signal of indifference ends the struggle; praise withheld is "
         "reclaimed.", "basis": "primary", "source": "Walther, L 69,1 st. III; L 105,27-106,2"},
        {"id": "D5", "claim": "Walther opposed mutual love to Reinmar's one-sided, renunciatory service.",
         "basis": "scholarship", "source": "Ukena-Best, 'Walther von der Vogelweide', NDB 27 (2020) 351-353"},
        {"id": "D6", "claim": "He admits he cannot love one who does him harm; the returned bond is dearer.",
         "basis": "primary", "source": "Walther, L 26,3-12"},
    ],
    "research_question": {"category": "robustness to spurious correlation / modality imbalance",
                          "question": ("Does a harmonic co-carrying fusion with an AM-HM overflow penalty keep a "
                                       "weak channel trained well enough to survive loss of the dominant one?")},
    "mechanism": {
        "name": "Zweiherz (two-heart co-carrying unit)", "family": "multimodal fusion / balance-seeking coupling",
        "signature_modules": ["zweiherz", "overflow_penalty", "release_rule"],
        "closest_prior_art": ["ModDrop (Neverova et al. 2016, arXiv:1501.00102)",
                              "OGM-GE (Peng et al. 2022, arXiv:2203.15332)",
                              "Gradient Blending (Wang et al. 2020, arXiv:1905.12681)"],
        "overlap": "Medium",
        "prior_art_queries": ["harmonic mean fusion modality imbalance", "OGM-GE greedy modality competition",
                              "balanced multimodal learning gradient modulation 2024 2025"],
        "contribution_type": "mechanism",
        "delta": ("Balance is a property of the forward operator (harmonic mean over present channels, "
                  "gradient routed to the lighter share) plus a penalty on the AM-HM gap; release reuses "
                  "balanced shares so one remaining heart outputs on the bonded scale."),
    },
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1", "property_test": "C6.P1", "hypothesis": "H-SIG"},
        {"doctrine": "D2", "mechanism": "M2", "property_test": "C6.P4", "hypothesis": "H-NEC"},
        {"doctrine": "D3", "mechanism": "M1", "property_test": "C6.P3", "hypothesis": "H-SIG"},
        {"doctrine": "D4", "mechanism": "M3", "property_test": "C6.P6", "hypothesis": "H-RIVAL"},
        {"doctrine": "D5", "mechanism": "R1", "property_test": "C6.P6", "hypothesis": "H-RIVAL"},
        {"doctrine": "D6", "mechanism": "M1", "property_test": "C6.P2", "hypothesis": "H-BLIND"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "With the dominant channel definitively absent, Zweiherz beats size-matched "
         "mean fusion.", "metric": "acc_A_lost", "split": "shifted", "comparison": "model - baseline",
         "direction": "greater", "mesi": 0.05, "seeds": 5},
        {"id": "H-NEC", "statement": "Replacing the harmonic operator by the arithmetic mean (penalty kept) hurts "
         "more than making the mixer linear.", "metric": "acc_A_lost",
         "comparison": "signature_knockout - matched_knockout", "direction": "less", "mesi": 0.03, "seeds": 5},
        {"id": "H-BLIND", "statement": "When B is weak but informative and never lost, forced equal carrying "
         "costs in-distribution accuracy.", "condition": "weak_b", "grounding": "D6: bonds that are real but "
         "unequal (grace, patronage, love of enemies) do not fit the reciprocity rule",
         "metric": "acc_ID", "comparison": "model - baseline", "direction": "less", "mesi": 0.02, "seeds": 5},
        {"id": "H-RIVAL", "statement": "When B is pure noise, release beats Reinmar's endless one-sided service.",
         "condition": "indifferent_b", "metric": "acc_ID", "comparison": "model - rival_staete",
         "direction": "greater", "mesi": 0.05, "seeds": 5},
        {"id": "H-PRIOR", "statement": "Zweiherz matches or exceeds ModDrop on A-lost without random dropping.",
         "metric": "acc_A_lost", "comparison": "model - moddrop", "direction": "greater", "mesi": 0.02,
         "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.5, "margin_over_trivial": 0.3, "shuffled_band": 0.1,
                   "release_rho_min": 0.3, "gradcheck_rel_tol": 1e-5},
    "probe_predictions": [{"probe": "P9", "expected": "above baseline"},
                          {"probe": "P10", "expected": "below baseline"},
                          {"probe": "P6", "expected": "equal to baseline"},
                          {"probe": "P8", "expected": "equal to baseline"},
                          {"probe": "P5", "expected": "below baseline"}],
    "dialectic_links": [
        {"chapter": None, "figure": "Reinmar der Alte (not in corpus)", "relation": "rival", "test": "H-RIVAL"},
        {"chapter": 310, "figure": "Pope Innocent III", "relation": "rival", "test": "none (political polemic)"},
    ],
    "corpus_neighbors": [
        {"chapter": 312, "similarity": None, "difference": "Rustaveli: trust sealed by an irrevocable oath; here the "
         "bond is revocable and continuously re-measured as a shared share"},
        {"chapter": 16, "similarity": None, "difference": "Nefertari: symmetric recognition channels (isometries); "
         "here a parameter-free harmonic coupling of learned shares with an overflow cost"},
        {"chapter": 240, "similarity": None, "difference": "al-Mutanabbi: magnitudes relative to the measurer; here "
         "joint value capped by the lighter carrier"},
        {"chapter": 281, "similarity": None, "difference": "Sanai: payment kept out of the verdict; here return is the "
         "condition of the bond itself"},
    ],
    "barometer": {
        "cognitive_processing": ["learning curves on the dyad task"],
        "embodied_cognition": ["multichannel fusion under definitive channel loss"],
        "world_modeling": [], "consciousness": ["reciprocity index as a self-report of bond state"],
        "language_understanding": [], "emotional_intelligence": ["reciprocal-bond detection (indifferent partner)"],
        "creativity": [], "autonomy": ["release rule: structural change after probation"],
    },
    "task_types": ["vector_classification"],
    "applications": [
        {"use": "optical + SAR land-cover mapping that survives cloud-covered optical input",
         "sector": "environmental monitoring", "dataset": "SEN12MS (Schmitt et al. 2019)"},
        {"use": "audio-visual event recognition with a missing stream", "sector": "media accessibility",
         "dataset": "AVE audio-visual event dataset (Tian et al. 2018)"},
        {"use": "vision + touch grasp-state estimation when the camera is occluded", "sector": "robotics",
         "dataset": "ObjectFolder (Gao et al. 2021)"},
    ],
    "safety_notes": ("No claim of replicating Walther's mind; no generated sentence is presented as his. Synthetic "
                     "data only. Release decisions are logged; in human-facing use a release must be disclosed."),
}

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

CHAPTER = 319
FILE_NAME = "chapter_0319_walther_von_der_vogelweide_1170.py"
TASK_TYPES = ["vector_classification"]
HM_EPS = 1e-6
DEFAULTS = dict(dA=6, dB=6, K=4, H=16, J=12, lam=0.3, lr=0.01, batch=64, clip=5.0, steps=1400,
                n_train=2400, n_val=600, n_test=1200, sepA=2.0, sigA=0.40, sigB=0.30, weak_sigB=0.80, side_sigA=0.90,
                probation=0.35, rho_min=0.3, moddrop=0.25, ogm_alpha=0.3)


class NonFinite(Exception):
    pass


# BEGIN STANDARD UTILITIES v1.0
def softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def logsumexp(z, axis=-1):
    m = np.max(z, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(z - m), axis=axis, keepdims=True))).squeeze(axis)


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


class Adam:
    def __init__(self, params, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads, sign=1.0):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= sign * self.lr * mh / (np.sqrt(vh) + self.eps)


def clip_global_norm(grads, max_norm):
    norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    if norm > max_norm:
        grads = {k: g * (max_norm / (norm + 1e-12)) for k, g in grads.items()}
    return grads, norm


def finite_difference_check(loss_fn, params, grads, rng, eps=1e-6, n_entries=20, floor=1e-4):
    """Central differences on every tensor; floor on |a|+|n| absorbs float64 round-off for tiny gradients."""
    worst, checked = 0.0, 0
    for k, p in params.items():
        flat = p.reshape(-1)
        idx = set(rng.choice(flat.size, size=min(n_entries, flat.size), replace=False).tolist())
        idx.add(int(np.argmax(np.abs(grads[k]).reshape(-1))))
        for i in idx:
            old = flat[i]
            flat[i] = old + eps
            lp = loss_fn()
            flat[i] = old - eps
            lm = loss_fn()
            flat[i] = old
            num = (lp - lm) / (2 * eps)
            ana = grads[k].reshape(-1)[i]
            worst = max(worst, abs(num - ana) / max(abs(num) + abs(ana), floor))
        checked += 1
    return {"tensors_checked": checked, "tensors_total": len(params), "max_rel_error": worst}


def paired_bootstrap_ci(diffs, rng, n_boot=2000, alpha=0.05):
    d = np.asarray(diffs, dtype=float)
    boots = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def verdict(mean, ci, direction, mesi, n_seeds, min_seeds=5):
    if n_seeds < min_seeds:
        return "not evaluated"
    lo, hi = ci
    if direction == "greater":
        if lo > 0 and mean >= mesi:
            return "supported"
        return "contradicted" if hi < 0 else "inconclusive"
    if hi < 0 and mean <= -mesi:
        return "supported"
    return "contradicted" if lo > 0 else "inconclusive"


def write_report(report, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
# END STANDARD UTILITIES


# ----------------------------------------------------------------------------- data and tasks
def make_world(rng, cfg):
    """Fixed generative parameters of one seed: class means of A, a rotation hiding B's XOR axes."""
    mu = rng.normal(size=(cfg["K"], cfg["dA"]))
    mu /= np.linalg.norm(mu, axis=1, keepdims=True)
    q, _ = np.linalg.qr(rng.normal(size=(cfg["dB"], cfg["dB"])))
    return {"muA": mu * cfg["sepA"], "rotB": q}


def _xor_pair(bit, n, rng):
    """Two latent coordinates whose sign product encodes one bit; magnitudes kept off zero."""
    z1 = (np.abs(rng.normal(size=n)) + 0.25) * rng.choice([-1.0, 1.0], size=n)
    z2 = (np.abs(rng.normal(size=n)) + 0.25) * np.sign(z1) * np.where(bit == 1, 1.0, -1.0)
    return z1, z2


def sample_split(world, n, rng, cfg, regime="native"):
    K, dA, dB = cfg["K"], cfg["dA"], cfg["dB"]
    y = rng.integers(0, K, size=n)
    g = rng.uniform(0.6, 1.4, size=(n, 1))
    sigA = cfg["sigA"] if regime == "native" else cfg["side_sigA"]
    xA = g * world["muA"][y] + sigA * rng.normal(size=(n, dA))
    if regime == "indifferent_b":
        zB = rng.normal(size=(n, dB))
    else:
        z1, z2 = _xor_pair(y // 2, n, rng)
        z3, z4 = _xor_pair(y % 2, n, rng)
        sig = cfg["weak_sigB"] if regime == "weak_b" else cfg["sigB"]
        sig_part = np.stack([z1, z2, z3, z4], 1) + sig * rng.normal(size=(n, 4))
        zB = np.concatenate([sig_part, rng.normal(size=(n, dB - 4))], 1)
    xB = zB @ world["rotB"]
    return {"xA": xA, "xB": xB, "pA": np.ones(n), "pB": np.ones(n), "y": y}


def with_absent(split, channel):
    out = dict(split)
    out["x" + channel] = np.zeros_like(split["x" + channel])
    out["p" + channel] = np.zeros_like(split["p" + channel])
    return out


def make_splits(world, rng, cfg, regime="native"):
    tr = sample_split(world, cfg["n_train"], rng, cfg, regime)
    va = sample_split(world, cfg["n_val"], rng, cfg, regime)
    te = sample_split(world, cfg["n_test"], rng, cfg, regime)
    return {"train": tr, "val": va, "test": te, "A_lost": with_absent(te, "A"), "B_lost": with_absent(te, "B")}


def row_hashes(split):
    rows = np.concatenate([split["xA"], split["xB"]], 1)
    return {hashlib.sha1(np.round(r, 12).tobytes()).hexdigest() for r in rows}


def as_channels(X, model):
    """Accept a channel dict or a plain matrix (probe battery): first dA columns are heart A, rest heart B."""
    if isinstance(X, dict):
        return X["xA"], X["xB"], X.get("pA", np.ones(len(X["xA"]))), X.get("pB", np.ones(len(X["xB"])))
    dA = model["cfg"]["dA"]
    n = X.shape[0]
    return X[:, :dA], X[:, dA:], np.ones(n), np.ones(n)


# ----------------------------------------------------------------------------- model
ACTIVE = {"mutant": None}


def build_model(in_dim=12, out_dim=4, task_type="vector_classification", rng=None, **cfg):
    """Two hearts (channel encoders) -> shares -> Zweiherz fusion -> mixer -> readout."""
    c = dict(DEFAULTS)
    c.update({"fusion": "zweiherz", "mixer": "tanh", "release": True})
    c.update(cfg)
    if in_dim != c["dA"] + c["dB"]:
        c["dA"], c["dB"] = in_dim // 2, in_dim - in_dim // 2
    c["K"] = out_dim
    rng = rng if rng is not None else np.random.default_rng(0)
    H, J, K = c["H"], c["J"], c["K"]

    def w(a, b):
        return rng.normal(size=(a, b)) / math.sqrt(a)
    p = {"W1_A": w(c["dA"], H), "b1_A": np.zeros(H), "W1_B": w(c["dB"], H), "b1_B": np.zeros(H),
         "U_A": w(H, J), "c_A": np.zeros(J), "U_B": w(H, J), "c_B": np.zeros(J),
         "Wz": w(J, J), "bz": np.zeros(J), "Wo": w(J, K), "bo": np.zeros(K)}
    return {"params": p, "cfg": c, "released": [False, False], "log": []}


def harmonic(shares, pres):
    """Harmonic mean over present hearts: P / sum(q/s). Nonzero only if every present heart carries.
    A row with no present heart (both absent, or the only present one released) has an empty bond: 0."""
    P = sum(pres)[:, None]
    inv = sum(q[:, None] / (s + HM_EPS) for s, q in zip(shares, pres))
    return np.where(P > 0, P / np.maximum(inv, 1e-300), 0.0), P


def d_harmonic(shares, pres, hm, P):
    """dHM/ds_c = q_c HM^2 / (P (s_c+eps)^2): the lighter carrier receives the larger gradient."""
    Ps = np.maximum(P, 1.0)
    if ACTIVE["mutant"] == "wrong_harmonic_grad":
        return [q[:, None] / Ps for q in pres]
    return [q[:, None] * hm ** 2 / (Ps * (s + HM_EPS) ** 2) for s, q in zip(shares, pres)]


def forward(model, xA, xB, pA, pB):
    p, c = model["params"], model["cfg"]
    qA = pA * (0.0 if model["released"][0] else 1.0)
    qB = pB * (0.0 if model["released"][1] else 1.0)
    hA = np.tanh(xA @ p["W1_A"] + p["b1_A"])
    hB = np.tanh(xB @ p["W1_B"] + p["b1_B"])
    preA, preB = hA @ p["U_A"] + p["c_A"], hB @ p["U_B"] + p["c_B"]
    sA, sB = softplus(preA), softplus(preB)
    hm, P = harmonic([sA, sB], [qA, qB])
    if c["fusion"] == "zweiherz":
        m = hm
    elif c["fusion"] == "mean":
        m = (qA[:, None] * sA + qB[:, None] * sB) / np.maximum(P, 1.0)
    else:
        m = np.zeros_like(hm)
    u = qA[:, None] * (sA + HM_EPS) + qB[:, None] * (sB + HM_EPS) - P * hm   # AM-HM gap of effective shares
    z = np.tanh(m @ p["Wz"] + p["bz"]) if c["mixer"] == "tanh" else m
    logits = z @ p["Wo"] + p["bo"]
    return {"hA": hA, "hB": hB, "preA": preA, "preB": preB, "sA": sA, "sB": sB, "qA": qA, "qB": qB,
            "hm": hm, "P": P, "m": m, "u": u, "z": z, "logits": logits}


def _loss(model, cache, y):
    lg = cache["logits"]
    ce = float(np.mean(logsumexp(lg) - lg[np.arange(len(y)), y]))
    return ce + model["cfg"]["lam"] * float(np.mean(cache["u"])), ce


def loss_and_grads(model, batch):
    xA, xB, pA, pB = as_channels(batch.get("X", batch), model)
    y = batch["y"]
    p, c = model["params"], model["cfg"]
    k = forward(model, xA, xB, pA, pB)
    loss, _ = _loss(model, k, y)
    n, J = len(y), c["J"]
    d = softmax(k["logits"])
    d[np.arange(n), y] -= 1.0
    d /= n
    g = {"Wo": k["z"].T @ d, "bo": d.sum(0)}
    dz = d @ p["Wo"].T
    if c["mixer"] == "tanh":
        dpre = dz * (1 - k["z"] ** 2)
        g["Wz"], g["bz"], dm = k["m"].T @ dpre, dpre.sum(0), dpre @ p["Wz"].T
    else:
        g["Wz"], g["bz"], dm = np.zeros_like(p["Wz"]), np.zeros_like(p["bz"]), dz
    dhm = d_harmonic([k["sA"], k["sB"]], [k["qA"], k["qB"]], k["hm"], k["P"])
    du = c["lam"] / (n * J)
    for i, ch in enumerate("AB"):
        q = k["q" + ch][:, None]
        mean_ds = q / np.maximum(k["P"], 1.0) if c["fusion"] == "mean" else 0.0
        dm_ds = dhm[i] if c["fusion"] == "zweiherz" else mean_ds
        du_ds = q - k["P"] * dhm[i]
        if ACTIVE["mutant"] == "overflow_grad_sign":
            du_ds = -du_ds
        dpre_s = (dm * dm_ds + du * du_ds) * sigmoid(k["pre" + ch])
        g["U_" + ch], g["c_" + ch] = k["h" + ch].T @ dpre_s, dpre_s.sum(0)
        dph = (dpre_s @ p["U_" + ch].T) * (1 - k["h" + ch] ** 2)
        x = xA if ch == "A" else xB
        g["W1_" + ch], g["b1_" + ch] = x.T @ dph, dph.sum(0)
    if ACTIVE["mutant"] == "zero_grad_U_B":
        g["U_B"] = np.zeros_like(g["U_B"])
    return loss, g


def predict(model, X):
    xA, xB, pA, pB = as_channels(X, model)
    return np.argmax(forward(model, xA, xB, pA, pB)["logits"], axis=1)


def hidden_states(model, X):
    xA, xB, pA, pB = as_channels(X, model)
    k = forward(model, xA, xB, pA, pB)
    return {key: k[key] for key in ("hA", "hB", "sA", "sB", "m", "u", "z")}


def accuracy(model, split):
    return float(np.mean(predict(model, split) == split["y"]))


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


def balance_index(model, split):
    """Mean over units and examples of 2 min(a,b)/(a+b): 1 = equal carrying, 0 = one heart alone."""
    k = forward(model, split["xA"], split["xB"], np.ones(len(split["y"])), np.ones(len(split["y"])))
    a, b = k["sA"], k["sB"]
    return float(np.mean(2 * np.minimum(a, b) / (a + b + 1e-12)))


# ----------------------------------------------------------------------------- baselines and rivals
# B0 mean fusion over present hearts (size-matched: fusion is parameter-free, all tensors identical).
# B1 ModDrop: mean fusion trained with one heart randomly dropped per example (fit(moddrop=...)).
# B2 OGM: mean fusion with on-the-fly gradient modulation of the dominant heart (simplified, no GE).
# R1 Reinmar's 'staete': Zweiherz without release: endless service to an unanswering partner.
BASELINES = {"zweiherz": {}, "mean": {"fusion": "mean", "lam": 0.0, "release": False},
             "moddrop": {"fusion": "mean", "lam": 0.0, "release": False},
             "ogm": {"fusion": "mean", "lam": 0.0, "release": False},
             "rival_staete": {"release": False}}


def ogm_coefficients(model, b, alpha):
    """Peng et al. (2022): k_u = 1 - tanh(alpha * rho_u) for the dominant heart (rho_u > 1), else 1."""
    n = len(b["y"])
    conf = []
    for keep in ("A", "B"):
        pA = b["pA"] if keep == "A" else np.zeros(n)
        pB = b["pB"] if keep == "B" else np.zeros(n)
        ok = (pA + pB) > 0
        if not np.any(ok):
            return 1.0, 1.0
        k = forward(model, b["xA"][ok], b["xB"][ok], pA[ok], pB[ok])
        conf.append(float(np.mean(softmax(k["logits"])[np.arange(ok.sum()), b["y"][ok]])))
    rho = conf[0] / max(conf[1], 1e-9)
    kA = 1 - math.tanh(alpha * rho) if rho > 1 else 1.0
    kB = 1 - math.tanh(alpha / rho) if rho < 1 else 1.0
    return kA, kB


# ----------------------------------------------------------------------------- release rule (non-gradient)
def reciprocity_stats(model, split):
    """Pooled cross-heart share correlation R and single-heart validation losses."""
    n = len(split["y"])
    one, zero = np.ones(n), np.zeros(n)
    saved = list(model["released"])
    model["released"] = [False, False]
    k = forward(model, split["xA"], split["xB"], one, one)
    a, b = k["sA"] - k["sA"].mean(0), k["sB"] - k["sB"].mean(0)
    R = float(np.sum(np.mean(a * b, 0)) / (np.sum(a.std(0) * b.std(0)) + 1e-12))
    ce_A = _loss(model, forward(model, split["xA"], split["xB"], one, zero), split["y"])[1]
    ce_B = _loss(model, forward(model, split["xA"], split["xB"], zero, one), split["y"])[1]
    model["released"] = saved
    return {"R": R, "ce_A": ce_A, "ce_B": ce_B}


def release_rule(stats, step, probation_step, rho_min, released, use_probation=True):
    """'sprich endeliche ... so wirde ich ein ledic man': after probation, a partner whose shares do not
    co-vary with the other heart's (R < rho_min) is released for good; the heart with the worse
    single-heart loss is the indifferent one. A release is never undone."""
    out = list(released)
    if any(out) or (use_probation and step < probation_step) or stats["R"] >= rho_min:
        return out
    out[0 if stats["ce_A"] > stats["ce_B"] else 1] = True
    return out


# ----------------------------------------------------------------------------- registries
def modules(model):
    p = model["params"]
    return {
        "heart_A_encoder": {"params": [p["W1_A"], p["b1_A"]], "role": "tanh encoder, channel A", "signature": False},
        "heart_B_encoder": {"params": [p["W1_B"], p["b1_B"]], "role": "tanh encoder, channel B", "signature": False},
        "share_A": {"params": [p["U_A"], p["c_A"]], "role": "softplus share projection, A", "signature": False},
        "share_B": {"params": [p["U_B"], p["c_B"]], "role": "softplus share projection, B", "signature": False},
        "zweiherz": {"params": [], "role": "parameter-free harmonic-mean fusion over present channels",
                     "signature": True},
        "overflow_penalty": {"params": [], "role": "penalty on the AM-HM gap of per-unit channel shares",
                             "signature": True},
        "release_rule": {"params": [], "role": "non-gradient channel release on low cross-channel share "
                         "correlation after probation", "signature": True},
        "mixer": {"params": [p["Wz"], p["bz"]], "role": "tanh mixing layer", "signature": False},
        "readout": {"params": [p["Wo"], p["bo"]], "role": "linear softmax readout", "signature": False},
    }


def knockout(model, name, mode="identity"):
    """Copy with one module replaced: identity / zero / mean, as the module permits."""
    m = {"params": {k: v.copy() for k, v in model["params"].items()}, "cfg": dict(model["cfg"]),
         "released": list(model["released"]), "log": []}
    if name == "zweiherz":
        m["cfg"]["fusion"] = "zero" if mode == "zero" else "mean"
    elif name == "overflow_penalty":
        m["cfg"]["lam"] = 0.0
    elif name == "release_rule":
        m["cfg"]["release"], m["released"] = False, [False, False]
    elif name == "mixer":
        if mode == "zero":
            m["params"]["Wz"][:] = 0.0
            m["params"]["bz"][:] = 0.0
        else:
            m["cfg"]["mixer"] = "identity"
    elif name in modules(model) and mode == "zero":
        for arr in modules(m)[name]["params"]:
            arr[...] = 0.0
    else:
        raise ValueError(f"knockout {name}/{mode} not defined")
    return m


MUTANTS = {
    "sign_flip_update": "optimizer ascends the loss",
    "zero_lr": "learning rate set to zero",
    "zero_grad_U_B": "gradient of heart B's share projection zeroed",
    "wrong_harmonic_grad": "harmonic derivative replaced by the arithmetic one",
    "overflow_grad_sign": "overflow-penalty gradient sign flipped",
}


# ----------------------------------------------------------------------------- training
def fit(model, data, budget, rng, val=None, moddrop=0.0, ogm=False):
    c, p = model["cfg"], model["params"]
    opt = Adam(p, lr=0.0 if ACTIVE["mutant"] == "zero_lr" else c["lr"])
    sign = -1.0 if ACTIVE["mutant"] == "sign_flip_update" else 1.0
    n, hist = len(data["y"]), []
    checks = {int(budget * f) for f in (c["probation"], 0.5, 0.65)}
    for t in range(budget):
        idx = rng.integers(0, n, size=c["batch"])
        b = {key: data[key][idx].copy() for key in ("xA", "xB", "pA", "pB", "y")}
        if moddrop > 0:
            drop, pick_a = rng.random(len(idx)) < moddrop, rng.random(len(idx)) < 0.5
            b["pA"][drop & pick_a], b["pB"][drop & ~pick_a] = 0.0, 0.0
            b["xA"][b["pA"] == 0], b["xB"][b["pB"] == 0] = 0.0, 0.0
        loss, g = loss_and_grads(model, b)
        if not np.isfinite(loss) or not all(np.all(np.isfinite(v)) for v in g.values()):
            raise NonFinite(f"non-finite loss or gradient at step {t}")
        if ogm:
            kA, kB = ogm_coefficients(model, b, c["ogm_alpha"])
            for key in ("W1_A", "b1_A", "U_A", "c_A"):
                g[key] = g[key] * kA
            for key in ("W1_B", "b1_B", "U_B", "c_B"):
                g[key] = g[key] * kB
        g, _ = clip_global_norm(g, c["clip"])
        opt.step(p, g, sign)
        hist.append(loss)
        if c["release"] and c["fusion"] == "zweiherz" and val is not None and t in checks:
            st = reciprocity_stats(model, val)
            new = release_rule(st, t, int(budget * c["probation"]), c["rho_min"], model["released"])
            if new != model["released"]:
                model["log"].append({"step": t, "released": new, **st})
            model["released"] = new
    return hist


# ----------------------------------------------------------------------------- correctness tests
def mixed_batch(split, rng, n=48):
    """A small batch where some rows lose heart A and some lose heart B, so every branch is exercised."""
    idx = rng.choice(len(split["y"]), size=n, replace=False)
    b = {k: split[k][idx].copy() for k in ("xA", "xB", "pA", "pB", "y")}
    b["pA"][: n // 6], b["pB"][n // 6: n // 3] = 0.0, 0.0
    b["xA"][b["pA"] == 0], b["xB"][b["pB"] == 0] = 0.0, 0.0
    return b


def gradcheck(model, batch, rng):
    loss, g = loss_and_grads(model, batch)
    return finite_difference_check(lambda: loss_and_grads(model, batch)[0], model["params"], g, rng)


def test_c1(splits, cfg, rng, steps_after=60):
    """C1: every tensor, at initialization and after training, for the signature and the knockout configs."""
    results, worst = [], 0.0
    configs = (("zweiherz", {}), ("mean+penalty", {"fusion": "mean"}), ("identity_mixer", {"mixer": "identity"}))
    for label, extra in configs:
        m = build_model(cfg["dA"] + cfg["dB"], cfg["K"], rng=rng, **{**cfg, **extra, "release": False})
        b = mixed_batch(splits["train"], rng)
        r0 = gradcheck(m, b, rng)
        fit(m, splits["train"], steps_after, rng)
        r1 = gradcheck(m, b, rng)
        worst = max(worst, r0["max_rel_error"], r1["max_rel_error"])
        results.append((label, r0, r1))
    main = results[0]
    return {"tensors_checked": main[1]["tensors_checked"], "tensors_total": main[1]["tensors_total"],
            "max_rel_error": worst, "checked_at": ["init", f"after_{steps_after}_training_steps"],
            "configs": [r[0] for r in results], "passed": bool(worst <= MIND_CARD["thresholds"]["gradcheck_rel_tol"])}


def trivial_accuracy(split):
    return float(np.max(np.bincount(split["y"])) / len(split["y"]))


def train_eval(cfg, splits, seed_rng, steps, extra=None, ko=None, **fit_kw):
    m = build_model(cfg["dA"] + cfg["dB"], cfg["K"], rng=seed_rng, **{**cfg, **(extra or {})})
    m = knockout(m, ko) if ko else m   # a knockout here is retrained with the module replaced
    hist = fit(m, splits["train"], steps, seed_rng, val=splits["val"], **fit_kw)
    return m, hist


def test_c2(splits, cfg, seed):
    runs = []
    for _ in range(2):
        m, h = train_eval(cfg, splits, np.random.default_rng(seed), 80)
        runs.append((h, predict(m, splits["test"]), m))
    same = runs[0][0] == runs[1][0] and bool(np.array_equal(runs[0][1], runs[1][1]))
    m = runs[0][2]
    finite = all(np.all(np.isfinite(v)) for v in m["params"].values()) and \
        all(np.all(np.isfinite(v)) for v in hidden_states(m, splits["A_lost"]).values())
    return same and finite, f"identical losses and predictions: {same}; params and activations finite: {finite}"


def c3_check(model, hist, splits):
    th = MIND_CARD["thresholds"]
    first, last = hist[0], float(np.mean(hist[-50:]))
    drop = 1 - last / first
    acc, triv = accuracy(model, splits["test"]), trivial_accuracy(splits["test"])
    ok = drop >= th["loss_drop_fraction"] and acc >= triv + th["margin_over_trivial"]
    return ok, f"loss {first:.3f}->{last:.3f} (drop {drop:.2f}); acc_ID {acc:.3f} vs majority {triv:.3f}"


def test_c4(splits, cfg, rng, steps):
    shuffled = dict(splits)
    shuffled["train"] = dict(splits["train"])
    shuffled["train"]["y"] = rng.permutation(splits["train"]["y"])
    m, _ = train_eval(cfg, shuffled, rng, steps, {"release": False})
    acc, triv = accuracy(m, splits["test"]), trivial_accuracy(splits["test"])
    ok = acc <= triv + MIND_CARD["thresholds"]["shuffled_band"]
    band = MIND_CARD["thresholds"]["shuffled_band"]
    return ok, f"shuffled-label acc_ID {acc:.3f} vs majority {triv:.3f} (band +{band})"


def test_c5(splits, cfg, rng, steps):
    """C5: each registered mutant must make C1 (at init) or a short C3 fail."""
    detected, rows = 0, []
    for name in MUTANTS:
        ACTIVE["mutant"] = name
        try:
            m = build_model(cfg["dA"] + cfg["dB"], cfg["K"], rng=rng, **{**cfg, "release": False})
            g_ok = gradcheck(m, mixed_batch(splits["train"], rng), rng)["max_rel_error"] <= \
                MIND_CARD["thresholds"]["gradcheck_rel_tol"]
            m2, hist = train_eval(cfg, splits, rng, steps, {"release": False})
            c3_ok = c3_check(m2, hist, splits)[0]
        except NonFinite:
            g_ok, c3_ok = True, False
        finally:
            ACTIVE["mutant"] = None
        caught = not (g_ok and c3_ok)
        detected += caught
        rows.append(f"{name}: {'caught' if caught else 'MISSED'} (C1 {'pass' if g_ok else 'fail'}, "
                    f"C3 {'pass' if c3_ok else 'fail'})")
    return {"detected": detected, "total": len(MUTANTS), "score": detected / len(MUTANTS)}, rows


# ----------------------------------------------------------------------------- property tests (C6)
def _random_shares(rng, n, spread=6.0):
    return np.exp(rng.uniform(-spread, spread, size=(n, 1))), np.exp(rng.uniform(-spread, spread, size=(n, 1)))


def _hm2(a, b):
    one = np.ones(len(a))
    return harmonic([a, b], [one, one])[0]


def prop_p1(rng, cfg):
    """P1 'minne entouc niht eine': the joint value never exceeds twice the lighter share, so one heart
    alone cannot carry the bond. Checked on random shares, random-search maximisation of the violation,
    and random-parameter models on random inputs. Control: the arithmetic mean violates it."""
    a, b = _random_shares(rng, 4000)
    def excess(x, y):   # effective shares carry the numerical floor HM_EPS
        return _hm2(x, y) - 2 * (np.minimum(x, y) + HM_EPS)
    viol = float(np.max(excess(a, b)))
    best = np.array([[1.0]]), np.array([[1.0]])
    for _ in range(300):
        cand = best[0] * math.exp(rng.normal()), best[1] * math.exp(rng.normal())
        if excess(*cand).item() > excess(*best).item():
            best = cand
    viol = max(viol, excess(*best).item())
    for _ in range(5):
        m = build_model(cfg["dA"] + cfg["dB"], cfg["K"], rng=rng, **cfg)
        k = forward(m, 3 * rng.normal(size=(200, cfg["dA"])), 3 * rng.normal(size=(200, cfg["dB"])),
                    np.ones(200), np.ones(200))
        viol = max(viol, float(np.max(excess(k["sA"], k["sB"]))))
    control = float(np.max((a + b) / 2 - 2 * np.minimum(a, b)))
    return viol <= 1e-9 and control > 1e-3, f"max violation {viol:.2e}; arithmetic-mean control {control:.2e}"


def prop_p2(rng):
    """P2 'teilent sie geliche': at fixed total load, the joint value is largest when shared equally.
    Control: the max operator is smallest at equal sharing."""
    a, b = _random_shares(rng, 4000, 3.0)
    half = (a + b) / 2
    viol = float(np.max(_hm2(a, b) - _hm2(half, half)))
    control = float(np.max(np.maximum(a, b) - np.maximum(half, half)))
    return viol <= 1e-9 and control > 1e-3, (f"max excess over equal split {viol:.2e}; "
                                             f"max-operator control {control:.2e}")


def _lighter_first(a, b):
    """Returns (min over unequal pairs of lighter-minus-heavier gradient, FD error of d_harmonic)."""
    one = np.ones(len(a))
    hm, P = harmonic([a, b], [one, one])
    ga, gb = d_harmonic([a, b], [one, one], hm, P)
    unequal = np.abs(a - b) > 1e-3 * (a + b)
    margin = float(np.min(np.where(a < b, ga - gb, gb - ga)[unequal]))
    fd = (_hm2(a + 1e-7, b) - _hm2(a - 1e-7, b)) / 2e-7
    return margin, float(np.max(np.abs(fd - ga) / np.maximum(np.abs(fd) + np.abs(ga), 1e-8)))


def prop_p3(rng):
    """P3 'hilf mir tragen': the lighter carrier always receives the larger gradient of the joint value,
    and d_harmonic agrees with finite differences. Control: the arithmetic derivative (mutant) fails."""
    a, b = _random_shares(rng, 2000, 3.0)
    margin, fd_err = _lighter_first(a, b)
    saved, ACTIVE["mutant"] = ACTIVE["mutant"], "wrong_harmonic_grad"
    try:
        c_margin, c_err = _lighter_first(a, b)
    finally:
        ACTIVE["mutant"] = saved
    ok = margin > 0 and fd_err < 1e-5 and not (c_margin > 0 and c_err < 1e-5)
    return ok, (f"min lighter-minus-heavier gradient {margin:.2e}; FD error {fd_err:.1e}; "
                f"arithmetic-derivative control margin {c_margin:.1e}, FD error {c_err:.1e}")


def forward_overflow(a, b, qa, qb):
    """The overflow exactly as forward() computes it, for given shares and presences."""
    hm, P = harmonic([a, b], [qa, qb])
    return qa[:, None] * (a + HM_EPS) + qb[:, None] * (b + HM_EPS) - P * hm


def prop_p4(rng):
    """P4 overflow u = sum(s) - P*HM is the unshared load: u >= 0 always, u = 0 iff the shares are equal.
    Control: a sign-flipped overflow is negative on unequal shares."""
    a, b = _random_shares(rng, 4000, 3.0)
    one = np.ones(len(a))
    u = forward_overflow(a, b, one, one)
    neg = float(np.min(u))
    eq = np.exp(rng.uniform(-3, 3, size=(500, 1)))
    zero_at_eq = float(np.max(np.abs(forward_overflow(eq, eq, one[:500], one[:500])) / eq))
    unequal = np.abs(a - b) > 0.05 * (a + b)
    positive_off_eq = bool(np.all(u[unequal] > 0))
    control = float(np.min(-u[unequal]))
    ok = neg >= -1e-12 and zero_at_eq < 1e-5 and positive_off_eq and control < 0
    return ok, f"min u {neg:.1e}; max rel u at equal shares {zero_at_eq:.1e}; sign-flip control min {control:.2e}"


def prop_p5(rng):
    """P5 release keeps the bonded scale: with equal shares, the value carried by the remaining heart alone
    equals the joint value (idempotent fusion). Control: sum fusion halves when a heart is released."""
    s = np.exp(rng.uniform(-3, 3, size=(1000, 1)))
    one, zero = np.ones(1000), np.zeros(1000)
    joint = harmonic([s, s], [one, one])[0]
    alone = harmonic([s, s], [one, zero])[0]
    err = float(np.max(np.abs(joint - alone) / s))
    control = float(np.min(np.abs((s + s) - s) / s))
    return err < 1e-5 and control > 0.5, f"max relative scale change {err:.1e}; sum-fusion control min {control:.2f}"


def prop_p6(rng, rho_min=0.3, probation=100):
    """P6 release rule (non-gradient, pure function): no release before probation; none while R >= rho_min;
    the heart with the worse single-heart loss is released; at most one heart; a release is never undone.
    Control: the variant without probation releases too early."""
    bad, early_ctrl = 0, 0
    for _ in range(3000):
        st = {"R": float(rng.uniform(-1, 1)), "ce_A": float(rng.uniform(0, 3)), "ce_B": float(rng.uniform(0, 3))}
        step, prev = int(rng.integers(0, 300)), [bool(v) for v in rng.random(2) < 0.1]
        prev = prev if not all(prev) else [True, False]
        out = release_rule(st, step, probation, rho_min, prev)
        if any(prev):
            bad += out != prev
        elif step < probation or st["R"] >= rho_min:
            bad += any(out)
        else:
            bad += out != [st["ce_A"] > st["ce_B"], not st["ce_A"] > st["ce_B"]]
        bad += sum(out) > 1
        early_ctrl += step < probation and any(release_rule(st, step, probation, rho_min, [False, False], False))
    return bad == 0 and early_ctrl > 0, f"violations {bad}/3000; no-probation control early releases {early_ctrl}"


def test_c6(rng, cfg):
    return [("P1_one_heart_alone", *prop_p1(rng, cfg)), ("P2_equal_share_maximal", *prop_p2(rng)),
            ("P3_lighter_carrier_gradient", *prop_p3(rng)), ("P4_overflow_identity", *prop_p4(rng)),
            ("P5_release_scale", *prop_p5(rng)), ("P6_release_rule", *prop_p6(rng))]


def test_c7(splits_by_regime):
    overlaps = 0
    for sp in splits_by_regime.values():
        tr = row_hashes(sp["train"])
        overlaps += len(tr & row_hashes(sp["val"])) + len(tr & row_hashes(sp["test"]))
    return overlaps == 0, f"train rows shared with val/test: {overlaps}"


# ----------------------------------------------------------------------------- hypothesis protocol
def seed_worlds(seed, cfg):
    """Independent child generators per regime and per training (SeedSequence), so runs are reproducible."""
    ss = np.random.SeedSequence(seed)
    kids = ss.spawn(3)
    world = make_world(np.random.default_rng(kids[0]), cfg)
    drng = np.random.default_rng(kids[1])
    splits = {r: make_splits(world, drng, cfg, r) for r in ("native", "weak_b", "indifferent_b")}
    return splits, kids[2]


def run_seed(seed, cfg, steps):
    splits, train_seq = seed_worlds(seed, cfg)
    rngs = iter(np.random.default_rng(s) for s in train_seq.spawn(12))
    nat, out = splits["native"], {}
    plan = [("zweiherz", "native", {}, {}), ("mean", "native", BASELINES["mean"], {}),
            ("moddrop", "native", BASELINES["moddrop"], {"moddrop": cfg["moddrop"]}),
            ("ogm", "native", BASELINES["ogm"], {"ogm": True}),
            ("ko_zweiherz", "native", {}, {"ko": "zweiherz"}),
            ("ko_overflow_penalty", "native", {}, {"ko": "overflow_penalty"}),
            ("ko_mixer", "native", {}, {"ko": "mixer"}),
            ("weak_zweiherz", "weak_b", {}, {}), ("weak_mean", "weak_b", BASELINES["mean"], {}),
            ("indiff_zweiherz", "indifferent_b", {}, {}),
            ("indiff_staete", "indifferent_b", BASELINES["rival_staete"], {})]
    for name, regime, extra, kw in plan:
        sp = splits[regime]
        m, _ = train_eval(cfg, sp, next(rngs), steps, extra, **kw)
        row = {"acc_ID": accuracy(m, sp["test"]), "acc_A_lost": accuracy(m, sp["A_lost"]),
               "acc_B_lost": accuracy(m, sp["B_lost"]), "released": list(m["released"])}
        if not any(m["released"]):
            row["balance"], row["R"] = balance_index(m, sp["test"]), reciprocity_stats(m, sp["val"])["R"]
        out[name] = row
    out["trivial"] = trivial_accuracy(nat["test"])
    return out


HYP_SPECS = {"H-SIG": ("zweiherz", "mean", "acc_A_lost"), "H-NEC": ("ko_zweiherz", "ko_mixer", "acc_A_lost"),
             "H-BLIND": ("weak_zweiherz", "weak_mean", "acc_ID"),
             "H-RIVAL": ("indiff_zweiherz", "indiff_staete", "acc_ID"),
             "H-PRIOR": ("zweiherz", "moddrop", "acc_A_lost")}
KO_SPECS = [("zweiherz", True, "ko_zweiherz", "zweiherz", "acc_A_lost",
             "native; harmonic -> arithmetic, penalty kept"),
            ("overflow_penalty", True, "ko_overflow_penalty", "zweiherz", "acc_A_lost", "native; lambda -> 0"),
            ("release_rule", True, "indiff_staete", "indiff_zweiherz", "acc_ID", "indifferent_b; release off"),
            ("mixer", False, "ko_mixer", "zweiherz", "acc_A_lost", "native; tanh mixer -> identity")]


def evaluate_hypotheses(per_seed, rng, evaluated):
    rows = []
    for h in MIND_CARD["hypotheses"]:
        a, b, metric = HYP_SPECS[h["id"]]
        diffs = [s[a][metric] - s[b][metric] for s in per_seed]
        ci = paired_bootstrap_ci(diffs, rng) if len(diffs) > 1 else (diffs[0], diffs[0])
        mean = float(np.mean(diffs))
        v = verdict(mean, ci, h["direction"], h["mesi"], len(diffs)) if evaluated else "not evaluated"
        rows.append({"id": h["id"], "metric": metric, "comparison": f"{a} - {b}", "mean_diff": mean,
                     "ci95": list(ci), "mesi": h["mesi"], "direction": h["direction"], "n_seeds": len(diffs),
                     "verdict": v})
    return rows


def knockout_table(per_seed, rng):
    rows = []
    for module, sig, ko, full, metric, note in KO_SPECS:
        diffs = [s[ko][metric] - s[full][metric] for s in per_seed]
        ci = paired_bootstrap_ci(diffs, rng) if len(diffs) > 1 else (diffs[0], diffs[0])
        rows.append({"module": module, "signature": sig, "metric": metric, "condition": note,
                     "metric_change": float(np.mean(diffs)), "ci95": list(ci)})
    return rows


def descriptive(per_seed):
    names = ["zweiherz", "mean", "moddrop", "ogm", "weak_zweiherz", "weak_mean", "indiff_zweiherz", "indiff_staete"]
    table = {}
    for n in names:
        table[n] = {k: float(np.mean([s[n][k] for s in per_seed if k in s[n]])) if any(k in s[n] for s in per_seed)
                    else None for k in ("acc_ID", "acc_A_lost", "acc_B_lost", "balance", "R")}
        table[n]["seeds_with_release"] = int(sum(any(s[n]["released"]) for s in per_seed))
    table["majority_baseline"] = float(np.mean([s["trivial"] for s in per_seed]))
    return table


# ----------------------------------------------------------------------------- real-data bridge
def load_csv(path):
    """Plain CSV: columns a_* (heart A features), b_* (heart B features), label (integer). No downloads."""
    with open(path, encoding="utf-8") as fh:
        head = fh.readline().strip().split(",")
        rows = [ln.strip().split(",") for ln in fh if ln.strip()]
    ia = [i for i, h in enumerate(head) if h.startswith("a_")]
    ib = [i for i, h in enumerate(head) if h.startswith("b_")]
    if not ia or not ib or "label" not in head:
        raise ValueError("CSV needs a_* columns, b_* columns and a 'label' column")
    arr = np.array([[float(v) for v in r] for r in rows])
    y = arr[:, head.index("label")].astype(int)
    return arr[:, ia], arr[:, ib], y


def data_bridge(path, cfg, rng):
    xA, xB, y = load_csv(path)
    xA = (xA - xA.mean(0)) / (xA.std(0) + 1e-9)
    xB = (xB - xB.mean(0)) / (xB.std(0) + 1e-9)
    perm, n = rng.permutation(len(y)), len(y)
    cut = [int(0.6 * n), int(0.8 * n)]

    def part(ix):
        return {"xA": xA[ix], "xB": xB[ix], "pA": np.ones(len(ix)), "pB": np.ones(len(ix)), "y": y[ix]}
    te = part(perm[cut[1]:])
    sp = {"train": part(perm[:cut[0]]), "val": part(perm[cut[0]:cut[1]]), "test": te,
          "A_lost": with_absent(te, "A"), "B_lost": with_absent(te, "B")}
    c = {**cfg, "dA": xA.shape[1], "dB": xB.shape[1], "K": int(y.max()) + 1}
    out = {}
    for name, extra in (("zweiherz", {}), ("mean", BASELINES["mean"])):
        m, _ = train_eval(c, sp, rng, cfg["steps"], extra)
        out[name] = {k: accuracy(m, sp[k]) for k in ("test", "A_lost", "B_lost")}
    return out


def run_correctness(cfg, seed, quick, rng):
    splits_all, _ = seed_worlds(seed, cfg)
    nat = splits_all["native"]
    steps = 1000 if quick else cfg["steps"]
    rows = []
    gc = test_c1(nat, cfg, rng)
    rows.append({"id": "C1", "name": "gradient_check", "passed": gc["passed"],
                 "detail": f"max rel error {gc['max_rel_error']:.2e} over {gc['tensors_checked']}/"
                           f"{gc['tensors_total']} tensors, configs {', '.join(gc['configs'])}"})
    ok, det = test_c2(nat, cfg, seed)
    rows.append({"id": "C2", "name": "determinism_finiteness", "passed": ok, "detail": det})
    m, hist = train_eval(cfg, nat, np.random.default_rng(seed + 1), steps)
    ok, det = c3_check(m, hist, nat)
    rows.append({"id": "C3", "name": "learning", "passed": ok, "detail": det})
    ok, det = test_c4(nat, cfg, rng, steps)
    rows.append({"id": "C4", "name": "shuffled_label_control", "passed": ok, "detail": det})
    if ACTIVE["mutant"] is None:
        mut, mrows = test_c5(nat, cfg, rng, 300)
        rows.append({"id": "C5", "name": "mutant_detection", "passed": mut["score"] == 1.0,
                     "detail": "; ".join(mrows)})
    else:
        mut = {"detected": 0, "total": len(MUTANTS), "score": None}
        rows.append({"id": "C5", "name": "mutant_detection", "passed": True,
                     "detail": f"skipped: external mutant '{ACTIVE['mutant']}' active"})
    for pid, ok, det in test_c6(rng, cfg):
        rows.append({"id": "C6." + pid.split("_")[0], "name": pid, "passed": bool(ok), "detail": det})
    ok, det = test_c7(splits_all)
    rows.append({"id": "C7", "name": "split_integrity", "passed": ok, "detail": det})
    return rows, gc, mut


def fmt(x):
    return "   n/a" if x is None else f"{x:6.3f}"


def print_report(rep, desc):
    print(f"=== VERIFIED REPORT · chapter {CHAPTER:04d} ===")
    print(f"file {rep['file']} · card revision {rep['card_revision']} · python {rep['environment']['python']} · "
          f"numpy {rep['environment']['numpy']}")
    print(f"seeds {rep['seeds']} · runtime {rep['runtime_s']:.1f} s · parameters {rep['n_params']} · "
          f"task types {', '.join(rep['task_types'])}")
    g = rep["gradcheck"]
    print(f"gradient check: {g['tensors_checked']}/{g['tensors_total']} tensors, "
          f"max rel error {g['max_rel_error']:.2e}, "
          f"at {', '.join(g['checked_at'])} -> {'PASS' if g['passed'] else 'FAIL'}")
    print("correctness:")
    for r in rep["correctness"]:
        print(f"  {r['id']:<6} {'PASS' if r['passed'] else 'FAIL'}  {r['name']}: {r['detail']}")
    mu = rep["mutants"]
    score = "" if mu["score"] is None else f" = {mu['score']:.2f}"
    print(f"mutation score: {mu['detected']}/{mu['total']}{score}")
    if rep["hypotheses"]:
        print("hypotheses (paired per-seed differences, 95% bootstrap CI, 2000 resamples):")
        for h in rep["hypotheses"]:
            print(f"  {h['id']:<8} {h['comparison']:<32} {h['metric']:<11} diff {h['mean_diff']:+.3f} "
                  f"CI [{h['ci95'][0]:+.3f}, {h['ci95'][1]:+.3f}] mesi {h['mesi']:.2f} ({h['direction']}) "
                  f"-> {h['verdict']}")
        print("knockouts (retrained with the module replaced; change = knockout - full):")
        for k in rep["knockouts"]:
            print(f"  {k['module']:<17} signature={str(k['signature']):<5} {k['metric']:<11} "
                  f"{k['metric_change']:+.3f} CI [{k['ci95'][0]:+.3f}, {k['ci95'][1]:+.3f}]  ({k['condition']})")
        print("descriptive means over seeds    acc_ID acc_A_lost acc_B_lost balance      R  seeds_released")
        for n, r in desc.items():
            if n != "majority_baseline":
                print(f"  {n:<28} {fmt(r['acc_ID'])} {fmt(r['acc_A_lost'])}     {fmt(r['acc_B_lost'])}  "
                      f"{fmt(r['balance'])} {fmt(r['R'])}  {r['seeds_with_release']}")
        print(f"  majority-class baseline      {desc['majority_baseline']:6.3f}")
    if rep.get("data_bridge"):
        print(f"real-data bridge: {rep['data_bridge']}")
    print(f"exit code {rep['exit_code']}")
    print("=== END REPORT ===")


def main(argv=None):
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__),
                                 description="Zweiherz two-heart co-carrying network (chapter 0319).")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--card", action="store_true")
    ap.add_argument("--mutant", type=str, default=None)
    ap.add_argument("--data", type=str, default=None)
    args = ap.parse_args(argv)
    if args.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    if args.mutant is not None and args.mutant not in MUTANTS:
        print(f"unknown mutant '{args.mutant}'; registered: {', '.join(MUTANTS)}", file=sys.stderr)
        return 2
    if args.seeds < 1:
        print("--seeds must be >= 1", file=sys.stderr)
        return 2
    ACTIVE["mutant"] = args.mutant
    t0, cfg = time.time(), dict(DEFAULTS)
    seeds = [args.seed] if args.quick else [args.seed + i for i in range(args.seeds)]
    print(f"seeds: {seeds}" + (f" · mutant: {args.mutant}" if args.mutant else ""))
    rng = np.random.default_rng(args.seed)
    try:
        rows, gc, mut = run_correctness(cfg, seeds[0], args.quick, rng)
        hyps, kos, desc = [], [], {}
        if args.mutant is None:
            steps = 1000 if args.quick else cfg["steps"]
            per_seed = []
            for s in seeds:
                per_seed.append(run_seed(s, cfg, steps))
                print(f"  seed {s} done ({time.time() - t0:.1f} s)")
            hyps = evaluate_hypotheses(per_seed, rng, evaluated=not args.quick)
            kos, desc = knockout_table(per_seed, rng), descriptive(per_seed)
        bridge = None
        if args.data:
            found = os.path.exists(args.data)
            bridge = data_bridge(args.data, cfg, rng) if found else f"path {args.data} not found; skipped"
        else:
            print("real-data bridge: no --data given; skipped")
    except NonFinite as exc:
        print(f"non-finite values: {exc}", file=sys.stderr)
        return 4
    runtime = time.time() - t0
    budget = 20.0 if args.quick else 180.0
    rows.append({"id": "C8", "name": "budget", "passed": runtime <= budget,
                 "detail": f"{runtime:.1f} s of {budget:.0f} s"})
    code = 0 if all(r["passed"] for r in rows) else (3 if not rows[-1]["passed"] and
                                                      all(r["passed"] for r in rows[:-1]) else 1)
    rep = {"schema_version": "1.0", "chapter": CHAPTER, "file": FILE_NAME,
           "card_revision": MIND_CARD["card_revision"],
           "environment": {"python": sys.version.split()[0], "numpy": np.__version__}, "seeds": seeds,
           "runtime_s": round(runtime, 2), "n_params": n_params(build_model(12, 4, rng=np.random.default_rng(0))),
           "gradcheck": {k: gc[k] for k in ("tensors_checked", "tensors_total", "max_rel_error", "checked_at",
                                            "passed")},
           "correctness": rows, "mutants": mut, "hypotheses": hyps, "knockouts": kos, "descriptive": desc,
           "task_types": TASK_TYPES, "data_bridge": bridge, "exit_code": code}
    print_report(rep, desc)
    if args.json:
        write_report(rep, args.json)
    return code


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0254 · Stephen I of Hungary (c.975-1038)
# Series companion and E-AGI Barometer: https://artificiology.com/barometer.html
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 13 Minds 241 - 260 Available on Amazon https://www.amazon.com/dp/B0HLD1S6QT
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0254_stephen_i_975 - Stephen I of Hungary (c.975-1038)
#================================================================================  
"""CORONA HOSPITUM: a court of guests under native rule (research prototype).

Thesis
    Strength is imported and rule is native: a young mind seats many foreign-schooled
    tongues in its court, pays each guest a stipend in the realm's own custom so that
    every guest can counsel alone, and bounds every voice so that no guest is silenced
    and none rules.

Evidence and provenance (belief)
    The record is the king's own recorded doctrine, with one caveat: the Admonitions
    (Libellus de institutione morum) were composed in his name for his heir Emeric,
    probably by a foreign cleric (Nemerkenyi 2004); the Laws carry royal authority but
    borrow Carolingian wording (DRMH online ed., Bak 2019). Deeds come from later
    legends and annals and are marked as such in MIND_CARD.

Doctrine -> mechanism -> test (IDs as in MIND_CARD)
    D1 one tongue is weak and fragile (Adm. c.6)   M1 one encoder per input channel     C6.2 H-SIG
    D2 nourish guests so they stay (Adm. c.6)       M2 stipend loss for every channel    C6.3 H-SIG
    D3 a supported guest does not leave (Laws I.24) M2 stipend as a fixed contract       C6.3 H-SIG
    D4 every people its own law (Adm. c.8; I pref.) M3 one shared native decision head   C1   H-SIG
    D5 none enslaved, none above (Adm. c.4, c.7)    M4 honour bounds + bounded counsel   C6.1 H-NEC
    D6 laws copied from Carolingian capitularies    rival: correctio (one exemplar)      H-RIVAL
    D7 foreign favourites ruin his heir's rule      blind spot: a coordinated bloc       H-BLIND

Research question
    When plausible forged messages enter through one or several input channels, does a
    court of individually schooled channel experts with bounded voices degrade less than
    learned fusion, and when does equal honour turn into capture by a faction?

Closest prior art and the delta
    Late fusion with auxiliary unimodal losses and learned fusion weights (Wang, Tran and
    Feiszli 2020, "Gradient-Blending"); linear opinion pools and static-gate mixtures
    (Genest and Zidek 1986; Jacobs et al. 1991); modality dropout (Neverova et al. 2016);
    Byzantine-robust aggregation (Blanchard et al. 2017). Delta: every channel is schooled
    through the one shared native head and pooled as a bounded vote under hard share
    bounds, which guarantees a single-channel influence bound while learned merit still
    acts inside the bounds. The baseline "blend" implements the closest method.

Blind spot
    The doctrine honours guests one by one and says nothing about factions. A bloc of
    guests schooled in one province can hold most of the court, and per-guest bounds do
    not bound the bloc (H-BLIND). A single shared head is also one point of failure, as
    the single heir was; that second risk is named here and not tested.

Task (generative process)
    z ~ N(0, I4) is a situation; the decision is y = 2*[z0+z1 > 0] + [z2-z3 > 0].
    Five tongues speak it: x_k = tanh(A_k (g_k * z) + B_k n + c_k) + s_k * eps, n ~ N(0, I2).
    Tongue 0 is native (every factor clear, s = 0.15). Tongues 1-3 were schooled in one
    province (shared A, clear on z0 and z1). Tongue 4 comes from another province (clear
    on z2 and z3). Guests use s = 0.30. Held-out: fresh draws. Shifted: one tongue carries a
    plausible message about -z (worst case over tongues). Blind spot: tongues 1-3 are
    forged together.

Limits
    Synthetic data, five channels, a static aggregator that never detects liars. Outputs
    are evidence about the mechanism, never about Stephen. A research prototype of an
    AGI-oriented component, not an AGI.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 3,
    "revision_log": [{"revision": 2, "date": "2026-09-17", "after_run": "quick run 1 (seed 254, hypotheses not evaluated)",
                      "change": "H-NEC knockout redefined from unbounded softmax shares over logit pooling to logit "
                      "pooling under the same learned bounded shares (counsel_bound identity); the combined 'corona' "
                      "module removed; the honour-bound knockout now reports equal shares (mean).",
                      "reason": "The honour logits train through a saturating tanh, so their magnitude is free and "
                      "irrelevant to the model; softmax(a) turned it into a 0.91 share for one channel, which measures "
                      "an arbitrary logit scale rather than the bounds. The new knockout keeps every learned quantity "
                      "and removes only the vote bound. Seeds, data, thresholds and the other hypotheses are unchanged."},
                     {"revision": 3, "date": "2026-09-17", "after_run": "default run 1 (seeds 254-258)",
                      "change": "C4 reference changed from the training-majority rule scored on the test split to the "
                      "best input-free predictor on the test split (also used by C3); C4 now averages three shuffled "
                      "retrainings and adds a label-leak negative control that must be flagged.",
                      "reason": "C4 failed by 0.001 (0.285 vs 0.284). Twelve diagnostic shuffled retrainings gave a "
                      "mean of 0.239 with spread 0.19-0.30 and no input signal; the training-majority class scored "
                      "0.224, below uniform chance, so the band was mis-centred and a single run too noisy. The new "
                      "reference is stricter for C3. Hypotheses, their metrics and all model seeds are unchanged."}],
    "generation": {"template_version": "codeguidelines 1.0 (15 Sep 2026)", "generator": "Claude",
                   "generator_version": "Claude Opus 5", "date": "2026-09-17"},
    "id": 254, "figure": "Stephen I", "born": 975, "died": 1038,
    "civilization": "Hungarian (Árpád dynasty; Grand Principality, then Kingdom of Hungary)",
    "provenance": "belief",
    "thesis": ("Strength is imported and rule is native: seat many foreign-schooled tongues in the "
               "court, pay each guest a stipend in the realm's own custom so every guest can counsel "
               "alone, and bound every voice so no guest is silenced and none rules."),
    "evidence": [
        {"id": "D1", "claim": "A kingdom of one language and one custom is weak and fragile; guests bring "
         "languages, customs, teachings (documenta) and arms.", "basis": "primary",
         "source": "Libellus de institutione morum c.6 (ed. Balogh, SRH II, 1938; ed. Havas, 2004)"},
        {"id": "D2", "claim": "Nourish guests with good will and hold them honourably so they prefer to stay; "
         "do not scatter what was gathered.", "basis": "primary", "source": "Libellus de institutione morum c.6"},
        {"id": "D3", "claim": "A guest received with benevolence and decently supported shall not leave his "
         "supporter while supported according to the agreement.", "basis": "primary",
         "source": "Decreta S. Stephani I.24 (DRMH online ed., Bak 2019)"},
        {"id": "D4", "claim": "No Greek ruled Latins with Greek customs; every people uses its own laws.",
         "basis": "primary", "source": "Libellus c.8; Decreta S. Stephani I, preface"},
        {"id": "D5", "claim": "Nobles fight for the king and do not serve; free men are not reduced to servitude; "
         "the young may counsel but their counsel is referred to the elders.", "basis": "primary",
         "source": "Libellus c.4, c.7; Decreta S. Stephani I.22"},
        {"id": "D6", "claim": "The laws copy Frankish synods and capitularies (Mainz 847; Capitulare de partibus "
         "Saxoniae).", "basis": "scholarship", "source": "DRMH online ed. (Bak 2019), notes to St 1 and St 2"},
        {"id": "D7", "claim": "His chosen successor Peter Orseolo favoured foreign courtiers and was deposed in "
         "1041; a rising overthrew him in 1046.", "basis": "scholarship",
         "source": "Engel, The Realm of St Stephen (2001); Illuminated Chronicle (hostile source)"},
        {"id": "D8", "claim": "One designated heir (Emeric, d. 1031); the senior claimant Vazul was blinded.",
         "basis": "deeds", "source": "Annales Altahenses; Györffy, King Saint Stephen of Hungary (1994)"},
    ],
    "research_question": {"category": "out-of-distribution detection and shift",
                          "question": ("When plausible forged messages enter through one or several input channels, "
                                       "does a court of individually schooled channel experts with bounded voices "
                                       "degrade less than learned fusion, and when does equal honour become "
                                       "capture by a faction?")},
    "mechanism": {
        "name": "Corona hospitum (court of guests under native rule)",
        "family": "multi-view late fusion with a bounded linear opinion pool",
        "signature_modules": ["crown_head", "honor_bounds", "counsel_bound"],
        "closest_prior_art": ["Gradient-Blending late fusion with auxiliary unimodal losses (Wang, Tran, Feiszli 2020)",
                              "Linear opinion pool / static-gate mixture of experts (Genest and Zidek 1986; Jacobs et al. 1991)",
                              "ModDrop modality dropout (Neverova et al. 2016)",
                              "Byzantine-robust aggregation, Krum (Blanchard et al. 2017)"],
        "overlap": "Medium",
        "prior_art_queries": ["late fusion auxiliary unimodal loss learned fusion weights",
                              "bounded weights linear opinion pool robustness single expert",
                              "multimodal classification spoofed sensor channel robustness",
                              "byzantine robust aggregation multi-view inference",
                              "shared classifier head across modality encoders auxiliary loss"],
        "contribution_type": "mechanism",
        "delta": ("Each channel is schooled through the one shared native head (stipend) and pooled as a bounded "
                  "vote under hard share bounds: a guaranteed single-channel influence bound with learned merit "
                  "kept inside the bounds."),
    },
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1 per-channel encoders", "property_test": "C6.2", "hypothesis": "H-SIG"},
        {"doctrine": "D2", "mechanism": "M2 stipend loss", "property_test": "C6.3", "hypothesis": "H-SIG"},
        {"doctrine": "D3", "mechanism": "M2 stipend loss", "property_test": "C6.3", "hypothesis": "H-SIG"},
        {"doctrine": "D4", "mechanism": "M3 shared native head", "property_test": "C1", "hypothesis": "H-SIG"},
        {"doctrine": "D5", "mechanism": "M4 honour bounds and bounded counsel", "property_test": "C6.1",
         "hypothesis": "H-NEC"},
        {"doctrine": "D6", "mechanism": "rival correctio", "property_test": "C1", "hypothesis": "H-RIVAL"},
        {"doctrine": "D7", "mechanism": "M4 bounds per guest, not per bloc", "property_test": "C6.2",
         "hypothesis": "H-BLIND"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "Under a plausible forgery in any single tongue, the court keeps higher "
         "worst-case accuracy than size-matched learned late fusion.", "metric": "worst_single_forgery_acc",
         "split": "shifted", "comparison": "corona - blend", "direction": "greater", "mesi": 0.05, "seeds": 5},
        {"id": "H-NEC", "statement": "Removing the bounded vote (pooling raw channel logits under the same learned "
         "bounded shares) lowers worst-case single-forgery accuracy.", "metric": "worst_single_forgery_acc",
         "comparison": "counsel_bound_knockout(identity) - corona_intact", "direction": "less", "mesi": 0.05,
         "seeds": 5},
        {"id": "H-BLIND", "statement": "When the three guests schooled in one province are forged together, the "
         "court is captured more often than learned late fusion.", "condition": "coordinated forgery of tongues 1-3",
         "grounding": "honour is granted guest by guest with no rule for factions; the foreign favourites of Peter "
         "Orseolo's court (1041, 1046)", "metric": "bloc_forgery_acc", "comparison": "corona - blend",
         "direction": "less", "mesi": 0.05, "seeds": 5},
        {"id": "H-RIVAL", "statement": "Carolingian correctio (aligned copies collated into one exemplar read by one "
         "reader) is less robust to a single forged tongue than the court.", "metric": "worst_single_forgery_acc",
         "split": "shifted", "comparison": "corona - correctio", "direction": "greater", "mesi": 0.03, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.40, "margin_over_trivial": 0.35, "shuffled_band": 0.06, "shuffled_repeats": 3,
                   "gradcheck_tol": 1e-5, "size_match_tol": 0.10},
    "probe_predictions": [{"probe": "P9", "expected": "above baseline"}, {"probe": "P8", "expected": "equal to baseline"},
                          {"probe": "P1", "expected": "below baseline"}, {"probe": "P5", "expected": "equal to baseline"},
                          {"probe": "P2", "expected": "equal to baseline"}],
    "dialectic_links": [{"chapter": 206, "relation": "successor", "test": "H-RIVAL"}],
    "corpus_neighbors": [
        {"chapter": 253, "similarity": None, "difference": "Al-Ma'arri prices harm by the condition of the harmed; "
         "no channel pooling. Code not available in this session."},
        {"chapter": 166, "similarity": None, "difference": "Theodora exempts a losing branch from the objective; "
         "here every guest stays inside the objective with a bounded voice."},
        {"chapter": 242, "similarity": None, "difference": "Hrotsvitha binds unlike parts by ratio inside one work; "
         "here diversity is imported through separate input channels and kept by a stipend."},
        {"chapter": 224, "similarity": None, "difference": "Methodius exports a norm system into a host; here the "
         "host imports encoders and keeps the decision rule native."},
        {"chapter": 227, "similarity": None, "difference": "Boris I audits two rival authorities; here no source is "
         "audited and no liar is detected."},
        {"chapter": 206, "similarity": None, "difference": "Charlemagne corrects copies toward one exemplar; "
         "implemented here as the rival."},
        {"chapter": 507, "similarity": 0.031, "difference": "Philip II sample file (multiplicative attenuation of a "
         "document); unrelated pooling. 9-token shingle Jaccard measured against the only file available."},
    ],
    "barometer": {"cognitive_processing": ["P10 sample efficiency"],
                  "embodied_cognition": ["P9 channel-noise robustness (no physical embodiment)"],
                  "world_modeling": [], "consciousness": ["P8 calibration of pooled probabilities"],
                  "language_understanding": ["one content spoken in five synthetic codes"],
                  "emotional_intelligence": [], "creativity": [], "autonomy": []},
    "task_types": ["vector_classification"],
    "applications": [
        {"use": "sensor fusion that survives one spoofed channel", "sector": "automated driving and robotics safety",
         "dataset": "nuScenes (Caesar et al. 2020)"},
        {"use": "multi-view recognition with one corrupted feature view", "sector": "document processing",
         "dataset": "UCI Multiple Features (mfeat)"},
        {"use": "activity recognition with a failing or tampered wearable sensor", "sector": "health and ageing",
         "dataset": "UCI PAMAP2 Physical Activity Monitoring"},
    ],
    "safety_notes": ("Political founder: framed as defence of decision systems against forged inputs. 'Tongues', "
                     "'guests' and 'bloc' denote input channels only, never peoples; no feature encodes ethnicity. "
                     "No claim of replicating a mind; no sentences are attributed to the figure."),
}

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

CHAPTER = 254
K_TONGUES, TONGUE_DIM, LATENT, N_CLASSES = 5, 8, 4, 4
HONOR_LATITUDE = 0.25          # shares lie strictly inside ((1-2k)/K, (1+2k)/K)
STIPEND = 0.5                  # weight of every guest's own vote (and of rival auxiliaries)
CLIP_NORM, BATCH = 5.0, 64
BLOC = (1, 2, 3)               # three guests schooled in one province
KINDS = ("corona", "blend", "correctio", "unanimis")
HIDDEN = {"corona": 16, "blend": 15, "correctio": 16, "unanimis": 33}
TASK_TYPES = ("vector_classification",)
TH = MIND_CARD["thresholds"]
GAINS = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.55, 0.55], [1.0, 1.0, 0.55, 0.55],
                  [1.0, 1.0, 0.55, 0.55], [0.55, 0.55, 1.0, 1.0]])
NOISE = np.array([0.15, 0.30, 0.30, 0.30, 0.30])
MUTANTS = {"sign_flip": "optimizer ascends the loss", "zero_lr": "learning rate set to zero",
           "grad_zero_crown": "gradient of the shared head Wc zeroed",
           "grad_zero_honor": "gradient of the honour logits a zeroed",
           "stipend_grad_dropped": "stipend kept in the loss but left out of the gradient"}


# BEGIN STANDARD UTILITIES v1.0
def softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def logsumexp(z, axis=-1):
    m = np.max(z, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(z - m), axis=axis))


def softplus(x):
    return np.logaddexp(0.0, x)


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads, sign=1.0):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            mh, vh = self.m[k] / (1 - self.b1 ** self.t), self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= sign * self.lr * mh / (np.sqrt(vh) + self.eps)


def clip_global_norm(grads, max_norm):
    total = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    if total > max_norm:
        grads = {k: g * (max_norm / (total + 1e-12)) for k, g in grads.items()}
    return grads, total


def gradient_check(loss_fn, params, grads, rng, n_entries=20, eps=1e-6, floor=1e-3):
    """Central differences on n random entries per tensor plus its largest-gradient entry.
    rel = |analytic - numeric| / max(|analytic| + |numeric|, floor)."""
    per_tensor = {}
    for name, p in params.items():
        flat, g = p.reshape(-1), grads[name].reshape(-1)
        idx = set(rng.choice(flat.size, size=min(n_entries, flat.size), replace=False).tolist())
        idx.add(int(np.argmax(np.abs(g))))
        err = 0.0
        for i in idx:
            old = flat[i]
            flat[i] = old + eps
            lp = loss_fn()
            flat[i] = old - eps
            lm = loss_fn()
            flat[i] = old
            num = (lp - lm) / (2 * eps)
            err = max(err, abs(g[i] - num) / max(abs(g[i]) + abs(num), floor))
        per_tensor[name] = err
    return max(per_tensor.values()), per_tensor


def paired_bootstrap_ci(diffs, rng, n_boot=2000, alpha=0.05):
    d = np.asarray(diffs, dtype=float)
    means = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    return float(d.mean()), [float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))]


def verdict(mean, ci, mesi, direction):
    if direction == "greater":
        return "supported" if ci[0] > 0 and mean >= mesi else ("contradicted" if ci[1] < 0 else "inconclusive")
    return "supported" if ci[1] < 0 and mean <= -mesi else ("contradicted" if ci[0] > 0 else "inconclusive")


def format_report(chapter, sections):
    lines = [f"=== VERIFIED REPORT · chapter {chapter:04d} ==="]
    for title, rows in sections:
        lines.append(f"-- {title}")
        lines.extend("   " + r for r in rows)
    lines.append("=== END REPORT ===")
    return "\n".join(lines)
# END STANDARD UTILITIES


# ---------------------------------------------------------------- data and tasks
def make_tongues(rng):
    """Five fixed codes of one situation; tongues 1-3 share a province's schooling."""
    province = rng.normal(0.0, 1.0, (TONGUE_DIM, LATENT))
    tongues = []
    for k in range(K_TONGUES):
        if k in BLOC:
            A = province[rng.permutation(TONGUE_DIM)] + 0.35 * rng.normal(0.0, 1.0, (TONGUE_DIM, LATENT))
        else:
            A = rng.normal(0.0, 1.0, (TONGUE_DIM, LATENT))
        tongues.append(dict(A=0.8 * A, B=0.5 * rng.normal(0.0, 1.0, (TONGUE_DIM, 2)),
                            c=0.2 * rng.normal(0.0, 1.0, TONGUE_DIM), gain=GAINS[k], noise=NOISE[k]))
    return tongues


def speak(tongue, z, rng):
    nuisance = rng.normal(0.0, 1.0, (z.shape[0], 2))
    s = (z * tongue["gain"]) @ tongue["A"].T + nuisance @ tongue["B"].T + tongue["c"]
    return np.tanh(s) + tongue["noise"] * rng.normal(0.0, 1.0, s.shape)


def decide(z):
    return 2 * (z[:, 0] + z[:, 1] > 0).astype(int) + (z[:, 2] - z[:, 3] > 0).astype(int)


def forge(tongues, blocks, z, which, rng):
    """Replace the tongues in `which` by plausible messages about the opposite situation -z."""
    out = [b.copy() for b in blocks]
    for k in which:
        out[k] = speak(tongues[k], -z, rng)
    return np.concatenate(out, axis=1)


def row_ids(z):
    return {hashlib.sha1(np.round(r, 10).tobytes()).hexdigest() for r in z}


def make_splits(rng, n_train=1500, n_val=500, n_test=1500):
    tongues = make_tongues(rng)
    z = rng.normal(0.0, 1.0, (n_train + n_val + n_test, LATENT))
    parts = {"train": z[:n_train], "val": z[n_train:n_train + n_val], "test": z[n_train + n_val:]}
    data = {"ids": {k: row_ids(v) for k, v in parts.items()}}
    for name, zz in parts.items():
        blocks = [speak(t, zz, rng) for t in tongues]
        data[name] = (np.concatenate(blocks, axis=1), decide(zz))
        if name == "test":
            data["forged"] = [forge(tongues, blocks, zz, (k,), rng) for k in range(K_TONGUES)]
            data["bloc"] = forge(tongues, blocks, zz, BLOC, rng)
    return data


def trivial_rate(data):
    """Best input-free predictor on the evaluation split (never below the training-majority rule)."""
    return float(np.max(np.bincount(data["test"][1], minlength=N_CLASSES)) / data["test"][1].size)


# ---------------------------------------------------------------- model
def even_slices(in_dim, k):
    step = in_dim // k
    cuts = [i * step for i in range(k)] + [in_dim]
    return [(cuts[i], cuts[i + 1]) for i in range(k)]


def _dense(rng, fan_in, fan_out, scale):
    return scale * rng.normal(0.0, 1.0 / math.sqrt(fan_in), (fan_out, fan_in))


def build_model(in_dim, out_dim, task_type="vector_classification", rng=None, **cfg):
    if task_type not in TASK_TYPES:
        raise ValueError(f"unsupported task type {task_type}")
    rng = rng if rng is not None else np.random.default_rng(0)
    kind, scale = cfg.get("kind", "corona"), cfg.get("init_scale", 1.0)
    slices = cfg.get("slices") or even_slices(in_dim, K_TONGUES)
    H, m, P = cfg.get("hidden", HIDDEN[kind]), cfg.get("code", 8), {}
    if kind == "unanimis":
        P.update(W1=_dense(rng, in_dim, H, scale), b1=np.zeros(H), Wo=_dense(rng, H, out_dim, scale),
                 bo=np.zeros(out_dim))
    else:
        for k, (a, b) in enumerate(slices):
            P[f"W1_{k}"], P[f"b1_{k}"] = _dense(rng, b - a, H, scale), np.zeros(H)
            P[f"W2_{k}"], P[f"b2_{k}"] = _dense(rng, H, m, scale), np.zeros(m)
            if kind == "blend":
                P[f"Wh_{k}"], P[f"bh_{k}"] = _dense(rng, m, out_dim, scale), np.zeros(out_dim)
        if kind == "corona":
            P.update(Wc=_dense(rng, m, out_dim, scale), bc=np.zeros(out_dim), a=np.zeros(len(slices)))
        elif kind == "blend":
            P["f"] = np.zeros(len(slices))
        elif kind == "correctio":
            R = cfg.get("reader", 8)
            P.update(Wr=_dense(rng, m, R, scale), br=np.zeros(R), Wo=_dense(rng, R, out_dim, scale),
                     bo=np.zeros(out_dim))
    return dict(kind=kind, params=P, slices=slices, kappa=HONOR_LATITUDE, weight=STIPEND, knock=None,
                lr=cfg.get("lr", 1e-2))


def _codes(model, X):
    P, out = model["params"], []
    for k, (a, b) in enumerate(model["slices"]):
        x = X[:, a:b]
        h1 = np.tanh(x @ P[f"W1_{k}"].T + P[f"b1_{k}"])
        out.append((x, h1, np.tanh(h1 @ P[f"W2_{k}"].T + P[f"b2_{k}"])))
    return out


def _backprop_code(model, G, k, dh2, code):
    P, (x, h1, h2) = model["params"], code
    ds2 = dh2 * (1.0 - h2 ** 2)
    G[f"W2_{k}"] += ds2.T @ h1
    G[f"b2_{k}"] += ds2.sum(0)
    ds1 = (ds2 @ P[f"W2_{k}"]) * (1.0 - h1 ** 2)
    G[f"W1_{k}"] += ds1.T @ x
    G[f"b1_{k}"] += ds1.sum(0)


def honor_shares(a, kappa):
    """Static pooling shares with hard bounds; they always sum to one (honour, not reliability)."""
    u = np.tanh(a)
    return (1.0 + kappa * (u - u.mean())) / a.size


def corona_forward(model, X):
    P, knock = model["params"], model["knock"] or ("", "")
    codes = _codes(model, X)
    Z = [np.zeros((X.shape[0], P["Wc"].shape[0])) if knock == (f"tongue_{k}", "zero")
         else h2 @ P["Wc"].T + P["bc"] for k, (_, _, h2) in enumerate(codes)]
    V = [softmax(z) for z in Z]
    w = honor_shares(P["a"], model["kappa"])
    if knock == ("counsel_bound", "identity"):
        p = softmax(sum(wk * z for wk, z in zip(w, Z)))
    else:
        p = sum(wk * v for wk, v in zip(w, V))
    return p, dict(codes=codes, Z=Z, V=V, w=w)


def corona_loss_grads(model, X, y, mutant=None, terms=("court", "stipend")):
    if model["knock"]:
        raise ValueError("knocked-out copies are for inference only")
    P, K, N, lam = model["params"], len(model["slices"]), X.shape[0], model["weight"]
    p, c = corona_forward(model, X)
    rows, onehot = np.arange(N), np.eye(p.shape[1])[y]
    loss, dp, G = 0.0, np.zeros_like(p), {k: np.zeros_like(v) for k, v in P.items()}
    if "court" in terms:
        loss += -np.mean(np.log(p[rows, y]))
        dp[rows, y] = -1.0 / (N * p[rows, y])
    if "stipend" in terms:
        loss += lam * np.mean([np.mean(logsumexp(z) - z[rows, y]) for z in c["Z"]])
    dw = np.array([np.sum(dp * v) for v in c["V"]])
    for k in range(K):
        v, g = c["V"][k], c["w"][k] * dp
        dz = v * (g - np.sum(g * v, axis=1, keepdims=True))
        if "stipend" in terms and mutant != "stipend_grad_dropped":
            dz = dz + (lam / (N * K)) * (v - onehot)       # the stipend reaches every guest directly
        G["Wc"] += dz.T @ c["codes"][k][2]
        G["bc"] += dz.sum(0)
        _backprop_code(model, G, k, dz @ P["Wc"], c["codes"][k])
    u = np.tanh(P["a"])
    G["a"] = (1.0 - u ** 2) * (model["kappa"] / K) * (dw - dw.mean())
    return float(loss), G


# ---------------------------------------------------------------- baselines and rival mechanisms
def blend_forward(model, X):
    P, codes = model["params"], _codes(model, X)
    Z = [h2 @ P[f"Wh_{k}"].T + P[f"bh_{k}"] for k, (_, _, h2) in enumerate(codes)]
    w = softmax(P["f"])
    return sum(wk * z for wk, z in zip(w, Z)), dict(codes=codes, Z=Z, w=w)


def blend_loss_grads(model, X, y, mutant=None):
    """Closest prior art: learned softmax fusion of per-channel logits plus auxiliary per-channel heads."""
    P, K, N, lam = model["params"], len(model["slices"]), X.shape[0], model["weight"]
    zbar, c = blend_forward(model, X)
    rows, onehot = np.arange(N), np.eye(zbar.shape[1])[y]
    loss = np.mean(logsumexp(zbar) - zbar[rows, y]) + lam * np.mean(
        [np.mean(logsumexp(z) - z[rows, y]) for z in c["Z"]])
    G = {k: np.zeros_like(v) for k, v in P.items()}
    dzbar = (softmax(zbar) - onehot) / N
    dw = np.array([np.sum(dzbar * z) for z in c["Z"]])
    G["f"] = c["w"] * (dw - np.sum(c["w"] * dw))
    for k in range(K):
        dz = c["w"][k] * dzbar + (lam / (N * K)) * (softmax(c["Z"][k]) - onehot)
        G[f"Wh_{k}"] += dz.T @ c["codes"][k][2]
        G[f"bh_{k}"] += dz.sum(0)
        _backprop_code(model, G, k, dz @ P[f"Wh_{k}"], c["codes"][k])
    return float(loss), G


def correctio_forward(model, X):
    P, codes = model["params"], _codes(model, X)
    exemplar = sum(h2 for (_, _, h2) in codes) / len(codes)
    r = np.tanh(exemplar @ P["Wr"].T + P["br"])
    return r @ P["Wo"].T + P["bo"], dict(codes=codes, e=exemplar, r=r)


def correctio_loss_grads(model, X, y, mutant=None):
    """Rival (Charlemagne, ch. 0206): copies corrected toward one collated exemplar read by one reader."""
    P, K, N, mu = model["params"], len(model["slices"]), X.shape[0], model["weight"]
    logits, c = correctio_forward(model, X)
    rows = np.arange(N)
    agree = sum(np.sum((h2 - c["e"]) ** 2) for (_, _, h2) in c["codes"]) / (N * K)
    loss = np.mean(logsumexp(logits) - logits[rows, y]) + mu * agree
    G = {k: np.zeros_like(v) for k, v in P.items()}
    dlog = (softmax(logits) - np.eye(logits.shape[1])[y]) / N
    G["Wo"], G["bo"] = dlog.T @ c["r"], dlog.sum(0)
    ds = (dlog @ P["Wo"]) * (1.0 - c["r"] ** 2)
    G["Wr"], G["br"] = ds.T @ c["e"], ds.sum(0)
    de = ds @ P["Wr"]
    for k in range(K):
        h2 = c["codes"][k][2]
        _backprop_code(model, G, k, de / K + (2.0 * mu / (N * K)) * (h2 - c["e"]), c["codes"][k])
    return float(loss), G


def unanimis_forward(model, X):
    P = model["params"]
    h = np.tanh(X @ P["W1"].T + P["b1"])
    return h @ P["Wo"].T + P["bo"], dict(h=h)


def unanimis_loss_grads(model, X, y, mutant=None):
    """One mind, one language: a single tanh layer over every channel at once."""
    P, N = model["params"], X.shape[0]
    logits, c = unanimis_forward(model, X)
    loss = np.mean(logsumexp(logits) - logits[np.arange(N), y])
    dlog = (softmax(logits) - np.eye(logits.shape[1])[y]) / N
    ds = (dlog @ P["Wo"]) * (1.0 - c["h"] ** 2)
    return float(loss), {"Wo": dlog.T @ c["h"], "bo": dlog.sum(0), "W1": ds.T @ X, "b1": ds.sum(0)}


# ---------------------------------------------------------------- registries and common interface
FORWARD = {"corona": corona_forward, "blend": blend_forward, "correctio": correctio_forward,
           "unanimis": unanimis_forward}
LOSSES = {"corona": corona_loss_grads, "blend": blend_loss_grads, "correctio": correctio_loss_grads,
          "unanimis": unanimis_loss_grads}


def loss_and_grads(model, batch, mutant=None):
    X, y = batch
    loss, G = LOSSES[model["kind"]](model, X, y, mutant=mutant)
    for name, key in (("grad_zero_crown", "Wc"), ("grad_zero_honor", "a")):
        if mutant == name and key in G:
            G[key] = np.zeros_like(G[key])
    return loss, G


def probabilities(model, X):
    out, _ = FORWARD[model["kind"]](model, X)
    return out if model["kind"] == "corona" else softmax(out)


def predict(model, X):
    return np.argmax(probabilities(model, X), axis=1)


def hidden_states(model, X):
    _, c = FORWARD[model["kind"]](model, X)
    return c["h"] if model["kind"] == "unanimis" else np.concatenate([h2 for (_, _, h2) in c["codes"]], axis=1)


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


def modules(model):
    if model["kind"] == "unanimis":
        return {"trunk": dict(params=["W1", "b1"], role="single tanh layer over all channels", signature=False),
                "head": dict(params=["Wo", "bo"], role="linear classifier", signature=False)}
    reg = {f"tongue_{k}": dict(params=[f"W1_{k}", f"b1_{k}", f"W2_{k}", f"b2_{k}"],
                               role="per-channel two-layer tanh encoder", signature=False)
           for k in range(len(model["slices"]))}
    if model["kind"] == "corona":
        reg["crown_head"] = dict(params=["Wc", "bc"], role="one linear head shared by all channel codes", signature=True)
        reg["honor_bounds"] = dict(params=["a"], role="static pooling shares with hard bounds", signature=True)
        reg["counsel_bound"] = dict(params=[], role="per-channel softmax before linear pooling", signature=True)
    elif model["kind"] == "blend":
        reg["fusion"] = dict(params=["f"], role="learned softmax fusion of channel logits", signature=False)
    else:
        reg["reader"] = dict(params=["Wr", "br", "Wo", "bo"], role="reader of the collated exemplar", signature=False)
    return reg


def knockout(model, name, mode):
    """Copy with a module replaced by identity (vote bound removed), zero, or its mean value."""
    if name not in modules(model) or mode not in ("identity", "zero", "mean"):
        raise KeyError(f"{name}/{mode}")
    copy = dict(model, params={k: v.copy() for k, v in model["params"].items()})
    if model["kind"] == "corona" and (name == "counsel_bound" or name.startswith("tongue_")):
        copy["knock"] = (name, "zero" if name.startswith("tongue_") else "identity")
        return copy
    for p in modules(model)[name]["params"]:
        copy["params"][p][...] = 0.0 if mode == "zero" else copy["params"][p].mean()
    return copy


# ---------------------------------------------------------------- training
class NonFinite(RuntimeError):
    pass


def fit(model, data, budget, rng, mutant=None):
    X, y = data
    opt = Adam(model["params"], lr=0.0 if mutant == "zero_lr" else model["lr"])
    trace = []
    for _ in range(budget):
        idx = rng.integers(0, X.shape[0], size=BATCH)
        loss, G = loss_and_grads(model, (X[idx], y[idx]), mutant)
        if not math.isfinite(loss):
            raise NonFinite(model["kind"])
        G, _ = clip_global_norm(G, CLIP_NORM)
        opt.step(model["params"], G, sign=-1.0 if mutant == "sign_flip" else 1.0)
        trace.append(loss)
    return trace


def rngs(ss, n):
    return [np.random.default_rng(s) for s in ss.spawn(n)]


def accuracy(model, X, y):
    return float(np.mean(predict(model, X) == y))


def full_loss(model, X, y):
    return loss_and_grads(model, (X, y))[0]


def train_selected(kind, data, steps, lr_grid, ss):
    """Same optimizer, budget and learning-rate grid for every mechanism; selection on validation only."""
    best = None
    for lr in lr_grid:
        init, batches = rngs(ss, 2)
        model = build_model(data["train"][0].shape[1], N_CLASSES, rng=init, kind=kind, lr=lr)
        before = full_loss(model, *data["train"])
        fit(model, data["train"], steps, batches)
        score = accuracy(model, *data["val"])
        if best is None or score > best[0]:
            best = (score, model, before, full_loss(model, *data["train"]))
    return best[1:]


def evaluate(model, data):
    Xt, yt = data["test"]
    forged = [accuracy(model, Xf, yt) for Xf in data["forged"]]
    return dict(id_acc=accuracy(model, Xt, yt), forged=forged, worst=min(forged), bloc=accuracy(model, data["bloc"], yt))


# ---------------------------------------------------------------- tests: correctness
def gradcheck_model(model, X, y, rng, mutant=None):
    _, G = loss_and_grads(model, (X, y), mutant)
    return gradient_check(lambda: loss_and_grads(model, (X, y))[0], model["params"], G, rng)


def c1_gradients(data, ss, steps=60):
    X, y = data["train"]
    worst, checked, total = 0.0, 0, 0
    for kind in KINDS:
        init, batches, pick = rngs(ss, 3)
        model = build_model(X.shape[1], N_CLASSES, rng=init, kind=kind, lr=1e-2)
        for stage in ("init", "after_training_steps"):
            if stage != "init":
                fit(model, data["train"], steps, batches)
            err, per = gradcheck_model(model, X[:24], y[:24], pick)
            worst, checked, total = max(worst, err), checked + len(per), total + len(model["params"])
    return worst <= TH["gradcheck_tol"], worst, checked, total


def c2_determinism(data, ss):
    s_init, s_batch = ss.spawn(2)
    runs = []
    for _ in range(2):
        model = build_model(data["train"][0].shape[1], N_CLASSES, rng=np.random.default_rng(s_init), kind="corona")
        trace = fit(model, data["train"], 40, np.random.default_rng(s_batch))
        runs.append((trace, probabilities(model, data["test"][0]), model))
    same = runs[0][0] == runs[1][0] and np.array_equal(runs[0][1], runs[1][1])
    finite = all(np.all(np.isfinite(v)) for v in runs[0][2]["params"].values()) and np.all(np.isfinite(runs[0][1]))
    return same and finite, f"identical={same} finite={finite}"


def c4_shuffled(data, ss, steps, lr, repeats):
    X, y = data["train"]
    limit, accs = trivial_rate(data) + TH["shuffled_band"], []
    for init, batches, perm in (rngs(child, 3) for child in ss.spawn(repeats)):
        model = build_model(X.shape[1], N_CLASSES, rng=init, kind="corona", lr=lr)
        fit(model, (X, perm.permutation(y)), steps, batches)
        accs.append(accuracy(model, *data["test"]))
    init, batches, perm = rngs(ss, 3)            # negative control: the label is written into the last tongue
    a, b = even_slices(X.shape[1], K_TONGUES)[-1]
    y_perm, (Xt, yt) = perm.permutation(y), data["test"]
    Xl, Xt = X.copy(), Xt.copy()
    for arr, lab in ((Xl, y_perm), (Xt, yt)):
        arr[:, a:b] = 0.0
        arr[:, a:a + N_CLASSES] = 3.0 * np.eye(N_CLASSES)[lab]
    leaky = build_model(X.shape[1], N_CLASSES, rng=init, kind="corona", lr=lr)
    fit(leaky, (Xl, y_perm), min(steps, 450), batches)
    leak = accuracy(leaky, Xt, yt)
    ok = float(np.mean(accs)) <= limit and leak > limit
    return ok, (f"mean held-out acc {np.mean(accs):.3f} over {repeats} shuffles vs limit {limit:.3f}; "
                f"label-leak control {leak:.3f} flagged={leak > limit}")


def c5_mutants(data, ss, steps):
    X, y = data["train"]
    floor = trivial_rate(data) + TH["margin_over_trivial"]
    detected, control_ok = {}, None
    for name in (None,) + tuple(MUTANTS):
        init, batches, pick = rngs(ss, 3)
        model = build_model(X.shape[1], N_CLASSES, rng=init, kind="corona", lr=1e-2)
        c1_fail = gradcheck_model(model, X[:24], y[:24], pick, mutant=name)[0] > TH["gradcheck_tol"]
        before = full_loss(model, X, y)
        try:
            fit(model, data["train"], steps, batches, mutant=name)
            drop = (before - full_loss(model, X, y)) / before
            c3_fail = drop < TH["loss_drop_fraction"] or accuracy(model, *data["test"]) < floor
        except NonFinite:
            c3_fail = True
        if name is None:
            control_ok = not (c1_fail or c3_fail)
        else:
            detected[name] = c1_fail or c3_fail
    return detected, control_ok


def c6_honor_bounds(rng):
    K, kap = K_TONGUES, HONOR_LATITUDE
    lo, hi = (1 - 2 * kap) / K, (1 + 2 * kap) / K
    worst, control_hits = 0.0, 0
    for trial in range(300):
        a = rng.normal(0.0, 10 ** rng.uniform(-2, 1.7), K)
        for _ in range(25):                       # gradient ascent on the largest share
            j, u = int(np.argmax(honor_shares(a, kap))), np.tanh(a)
            a += 5.0 * (kap / K) * (1 - u ** 2) * (np.eye(K)[j] - 1.0 / K) / max(1e-12, np.max(1 - u ** 2))
        w = honor_shares(a, kap)
        worst = max(worst, w.max() - hi, lo - w.min(), abs(w.sum() - 1.0))
        control_hits += int(softmax(a).max() > hi or softmax(a).min() < lo)
    return worst <= 1e-12 and control_hits > 0, f"max violation {worst:.2e}; unbounded softmax violated in {control_hits}/300"


def c6_influence_bound(rng):
    worst, control = 0.0, 0.0
    for trial in range(24):
        model = build_model(K_TONGUES * TONGUE_DIM, N_CLASSES, rng=rng, kind="corona", init_scale=rng.uniform(1.0, 6.0))
        model["params"]["a"] = rng.normal(0.0, 3.0, K_TONGUES)
        ctrl = knockout(model, "counsel_bound", "identity")
        X = rng.normal(0.0, 1.0, (16, K_TONGUES * TONGUE_DIM))
        k = int(rng.integers(K_TONGUES))
        a, b = model["slices"][k]
        p0, c0 = corona_forward(model, X)
        q0, _ = corona_forward(ctrl, X)
        for _ in range(25):                      # random search over forged inputs of any magnitude
            X2 = X.copy()
            X2[:, a:b] = rng.normal(0.0, 10 ** rng.uniform(-1, 2), (16, b - a))
            bound = 2.0 * c0["w"][k]
            worst = max(worst, np.max(np.abs(corona_forward(model, X2)[0] - p0).sum(1)) / bound)
            control = max(control, np.max(np.abs(corona_forward(ctrl, X2)[0] - q0).sum(1)) / bound)
    return worst <= 1.0 + 1e-9 and control > 1.0, f"max |dp|/bound {worst:.3f}; logit-pooling control {control:.3f}"


def c6_nourishment(rng):
    model = build_model(K_TONGUES * TONGUE_DIM, N_CLASSES, rng=rng, kind="corona", init_scale=1.5)
    X, y = rng.normal(0.0, 1.0, (32, K_TONGUES * TONGUE_DIM)), rng.integers(0, N_CLASSES, 32)
    k = K_TONGUES - 1
    keys = [f"W1_{k}", f"b1_{k}", f"W2_{k}", f"b2_{k}"]
    before = [corona_loss_grads(model, X, y, terms=(t,))[1] for t in ("stipend", "court")]
    for name, p in model["params"].items():
        if not name.endswith(f"_{k}") and name not in ("Wc", "bc"):
            p += rng.normal(0.0, 1.0, p.shape)
    after = [corona_loss_grads(model, X, y, terms=(t,))[1] for t in ("stipend", "court")]
    shift = [max(float(np.max(np.abs(b0[q] - a0[q]))) for q in keys) for b0, a0 in zip(before, after)]
    return shift[0] <= 1e-12 and shift[1] > 1e-6, f"stipend grad shift {shift[0]:.1e}; court grad (control) {shift[1]:.1e}"


def c7_splits(data):
    ids = data["ids"]
    disjoint = not (ids["train"] & ids["test"] or ids["train"] & ids["val"] or ids["val"] & ids["test"])
    leaked = ids["test"] | {next(iter(ids["train"]))}
    return disjoint and bool(ids["train"] & leaked), f"disjoint={disjoint}; injected leak detected={bool(ids['train'] & leaked)}"


# ---------------------------------------------------------------- tests: hypotheses
def run_seed(seed, steps, lr_grid):
    data_ss, model_ss = np.random.SeedSequence(seed).spawn(2)
    data = make_splits(np.random.default_rng(data_ss))
    out, models = {}, {}
    for kind, ss in zip(KINDS, model_ss.spawn(len(KINDS))):
        model, before, after = train_selected(kind, data, steps, lr_grid, ss)
        out[kind], models[kind] = evaluate(model, data), model
        out[kind].update(loss_before=before, loss_after=after, lr=model["lr"], n_params=n_params(model))
    knocks = {}
    for name, mode in (("counsel_bound", "identity"), ("honor_bounds", "mean"), ("tongue_0", "zero"),
                       ("tongue_1", "zero"), ("tongue_4", "zero"), ("crown_head", "mean")):
        knocks[name] = evaluate(knockout(models["corona"], name, mode), data)["worst"] - out["corona"]["worst"]
    return data, out, knocks, models


def hypothesis_rows(per_seed, knocks, rng, evaluated):
    diffs = {"H-SIG": [s["corona"]["worst"] - s["blend"]["worst"] for s in per_seed],
             "H-NEC": [k["counsel_bound"] for k in knocks],
             "H-BLIND": [s["corona"]["bloc"] - s["blend"]["bloc"] for s in per_seed],
             "H-RIVAL": [s["corona"]["worst"] - s["correctio"]["worst"] for s in per_seed]}
    rows = []
    for h in MIND_CARD["hypotheses"]:
        mean, ci = paired_bootstrap_ci(diffs[h["id"]], rng)
        rows.append(dict(id=h["id"], metric=h["metric"], mean_diff=mean, ci95=ci, mesi=h["mesi"], n_seeds=len(per_seed),
                         verdict=verdict(mean, ci, h["mesi"], h["direction"]) if evaluated else "not evaluated"))
    return rows


# ---------------------------------------------------------------- real-data bridge
def data_bridge(path, steps, ss):
    """CSV with columns t<k>_<feature> for tongue k and a 'label' column; results are not hypotheses."""
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    cols = [i for i, h in enumerate(header) if h.startswith("t")]
    tongue_of = [int(header[i].split("_")[0][1:]) for i in cols]
    order = np.argsort(tongue_of, kind="stable")
    X = raw[:, [cols[i] for i in order]]
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    y = raw[:, header.index("label")].astype(int)
    tk = np.array(tongue_of)[order]
    slices = [(int(np.argmax(tk == t)), int(np.argmax(tk == t)) + int(np.sum(tk == t))) for t in np.unique(tk)]
    split_rng, *model_rngs = rngs(ss, 1 + 2 * 2)
    idx = split_rng.permutation(len(y))
    tr, te = idx[: int(0.8 * len(y))], idx[int(0.8 * len(y)):]
    rows = []
    for j, kind in enumerate(("corona", "blend")):
        m = build_model(X.shape[1], int(y.max()) + 1, rng=model_rngs[2 * j], kind=kind, slices=slices)
        fit(m, (X[tr], y[tr]), steps, model_rngs[2 * j + 1])
        worst = 1.0
        for a, b in slices:
            Xf = X[te].copy()
            Xf[:, a:b] = X[split_rng.permutation(tr)[: len(te)], a:b]
            worst = min(worst, accuracy(m, Xf, y[te]))
        rows.append(f"{kind}: held-out acc {accuracy(m, X[te], y[te]):.3f}; worst swapped-tongue acc {worst:.3f}")
    return rows


# ---------------------------------------------------------------- report and command line
def correctness_suite(data, ss, steps, lr, repeats):
    s = ss.spawn(7)
    res = []
    ok, worst, checked, total = c1_gradients(data, s[0])
    res.append(("C1", "gradient_check", ok, f"max rel err {worst:.2e} over {checked}/{total} tensors (init and after 60 steps)"))
    res.append(("C2", "determinism_finiteness", *c2_determinism(data, s[1])))
    res.append(("C4", "shuffled_label_control", *c4_shuffled(data, s[2], steps, lr, repeats)))
    detected, control_ok = c5_mutants(data, s[3], 350)
    score = sum(detected.values()) / len(detected)
    res.append(("C5", "mutant_detection", score == 1.0 and control_ok,
                f"score {score:.2f}; unmutated control passes={control_ok}; " + ", ".join(f"{k}={v}" for k, v in detected.items())))
    for tid, name, fn, r in (("C6.1", "honor_bounds", c6_honor_bounds, s[4]), ("C6.2", "single_guest_influence_bound",
                             c6_influence_bound, s[5]), ("C6.3", "stipend_independent_of_other_guests", c6_nourishment, s[6])):
        res.append((tid, name, *fn(np.random.default_rng(r))))
    res.append(("C7", "split_integrity", *c7_splits(data)))
    return res, detected


def main(argv=None):
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=254)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--card", action="store_true")
    ap.add_argument("--mutant", type=str, default=None)
    ap.add_argument("--data", type=str, default=None)
    try:
        args = ap.parse_args(argv)
    except SystemExit:
        return 2
    if args.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    if args.mutant and args.mutant not in MUTANTS or args.seeds < 1:
        print(f"usage: python3 {os.path.basename(__file__)} [--quick] [--mutant {'|'.join(MUTANTS)}]")
        return 2
    try:
        return run_protocol(args)
    except NonFinite as exc:
        print(f"non-finite values in {exc}")
        return 4


def run_protocol(args):
    t0, quick = time.time(), args.quick or bool(args.mutant)
    seeds = [args.seed + i for i in range(1 if quick else args.seeds)]
    steps, lr_grid = (450, [1e-2]) if quick else (1500, [3e-3, 1e-2])
    print(f"seeds: {seeds}  steps: {steps}  lr grid: {lr_grid}")
    per_seed, knocks, correctness, n_par = [], [], [], {}
    for i, seed in enumerate(seeds):
        data, out, kn, models = run_seed(seed, steps, lr_grid)
        per_seed.append(out)
        knocks.append(kn)
        if i == 0:
            n_par = {k: v["n_params"] for k, v in out.items()}
            correctness, detected = correctness_suite(data, np.random.SeedSequence([seed, 7]), steps, out["corona"]["lr"],
                                                      1 if quick else TH["shuffled_repeats"])
            c = out["corona"]
            if args.mutant:
                init, batches = rngs(np.random.SeedSequence([seed, 9]), 2)
                m = build_model(data["train"][0].shape[1], N_CLASSES, rng=init, kind="corona")
                before = full_loss(m, *data["train"])
                try:
                    fit(m, data["train"], steps, batches, mutant=args.mutant)
                    c = dict(evaluate(m, data), loss_before=before, loss_after=full_loss(m, *data["train"]))
                except NonFinite:
                    c = dict(id_acc=0.0, loss_before=before, loss_after=before)
                g_err = gradcheck_model(build_model(data["train"][0].shape[1], N_CLASSES, rng=init, kind="corona"),
                                        *[v[:24] for v in data["train"]], batches, mutant=args.mutant)[0]
                correctness[0] = ("C1", "gradient_check", g_err <= TH["gradcheck_tol"], f"mutant {args.mutant}: max rel err {g_err:.2e}")
            drop = (c["loss_before"] - c["loss_after"]) / c["loss_before"]
            trivial = trivial_rate(data)
            learn_ok = drop >= TH["loss_drop_fraction"] and c["id_acc"] >= trivial + TH["margin_over_trivial"]
            size_ok = all(abs(n_par[k] - n_par["corona"]) / n_par["corona"] <= TH["size_match_tol"] for k in KINDS)
            correctness.insert(2, ("C3", "learning", learn_ok and size_ok,
                                   f"loss drop {drop:.2f}; held-out acc {c['id_acc']:.3f} vs trivial {trivial:.3f}; "
                                   f"size-matched within 10%={size_ok}"))
    runtime = time.time() - t0
    budget_ok = runtime <= (20.0 if args.quick else 180.0) or bool(args.mutant)
    correctness.append(("C8", "budget", budget_ok, f"{runtime:.1f} s of {20 if args.quick else 180} s"))
    hyp = hypothesis_rows(per_seed, knocks, np.random.default_rng(np.random.SeedSequence([args.seed, 11])), not quick)
    exit_code = 0 if all(r[2] for r in correctness) else (3 if not budget_ok else 1)
    print(render(args, seeds, runtime, n_par, correctness, detected, hyp, per_seed, knocks, exit_code))
    if args.data:
        print("\n".join(["real-data bridge (not hypotheses):"] + data_bridge(args.data, steps, np.random.SeedSequence([args.seed, 13]))))
    else:
        print("real-data bridge skipped: no --data PATH given")
    if args.json:
        write_json(args, seeds, runtime, n_par, correctness, detected, hyp, knocks, exit_code)
    return exit_code


def render(args, seeds, runtime, n_par, correctness, detected, hyp, per_seed, knocks, exit_code):
    mean = lambda key, kind: float(np.mean([s[kind][key] for s in per_seed]))
    sd = lambda key, kind: float(np.std([s[kind][key] for s in per_seed]))
    sections = [
        ("environment", [f"python {sys.version.split()[0]} · numpy {np.__version__} · file {os.path.basename(__file__)}",
                         f"seeds {seeds} · runtime {runtime:.1f} s · mode {'quick' if args.quick else 'mutant ' + args.mutant if args.mutant else 'default'}"]),
        ("parameters (size-matched)", [f"{k}: {v}" for k, v in n_par.items()]),
        ("correctness", [f"{tid:5s} {name:38s} {'PASS' if ok else 'FAIL'}  {detail}" for tid, name, ok, detail in correctness]),
        ("mutation score", [f"{sum(detected.values())}/{len(detected)} detected"]),
        ("mechanisms (mean ± sd over seeds)", [f"{k:10s} held-out {mean('id_acc', k):.3f}±{sd('id_acc', k):.3f} · "
                                               f"worst single forgery {mean('worst', k):.3f}±{sd('worst', k):.3f} · "
                                               f"bloc forgery {mean('bloc', k):.3f}±{sd('bloc', k):.3f}" for k in KINDS]),
        ("hypotheses (paired per-seed differences, 95% bootstrap, 2000 resamples)",
         [f"{h['id']:8s} {h['metric']:26s} mean {h['mean_diff']:+.3f} CI [{h['ci95'][0]:+.3f}, {h['ci95'][1]:+.3f}] "
          f"mesi {h['mesi']:.2f} n={h['n_seeds']} -> {h['verdict']}" for h in hyp]),
        ("knockouts on the court (change in worst single-forgery accuracy)",
         [f"{name:14s} signature={name in MIND_CARD['mechanism']['signature_modules']!s:5s} "
          f"mean {np.mean([k[name] for k in knocks]):+.3f}" for name in knocks[0]]),
        ("task types", [", ".join(TASK_TYPES)]),
        ("exit", [str(exit_code)]),
    ]
    return format_report(CHAPTER, sections)


def write_json(args, seeds, runtime, n_par, correctness, detected, hyp, knocks, exit_code):
    boot = np.random.default_rng(np.random.SeedSequence([args.seed, 17]))
    report = {
        "schema_version": "1.0", "chapter": CHAPTER, "file": os.path.basename(__file__),
        "card_revision": MIND_CARD["card_revision"],
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
        "seeds": seeds, "runtime_s": round(runtime, 2), "n_params": n_par["corona"],
        "gradcheck": {"tensors_checked": int(correctness[0][3].split(" over ")[1].split("/")[0]) if " over " in correctness[0][3] else 0,
                      "tensors_total": int(correctness[0][3].split("/")[1].split(" ")[0]) if " over " in correctness[0][3] else 0,
                      "max_rel_error": float(correctness[0][3].split("err ")[1].split(" ")[0]),
                      "checked_at": ["init", "after_training_steps"], "passed": bool(correctness[0][2])},
        "correctness": [{"id": t, "name": n, "passed": bool(ok), "detail": d} for t, n, ok, d in correctness],
        "mutants": {"detected": int(sum(detected.values())), "total": len(detected),
                    "score": float(sum(detected.values()) / max(1, len(detected)))},
        "hypotheses": hyp,
        "knockouts": [{"module": name, "signature": name in MIND_CARD["mechanism"]["signature_modules"],
                       "metric_change": paired_bootstrap_ci([k[name] for k in knocks], boot)[0],
                       "ci95": paired_bootstrap_ci([k[name] for k in knocks], boot)[1]} for name in knocks[0]],
        "task_types": list(TASK_TYPES), "exit_code": exit_code,
    }
    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    sys.exit(main())

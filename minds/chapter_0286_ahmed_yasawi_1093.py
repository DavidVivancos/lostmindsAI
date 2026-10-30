#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0286 · Ahmed Yasawi
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 15, Minds 281-300 Available on Amazon https://www.amazon.com/dp/B0HMQ7TVJM
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0286_ahmed_yasawi_1093 - Ahmed Yasawi (c. 1093-1166)
# END ATTRIBUTION
"""HALQA-JAHR, the voiced room: memory that exists only as a loop through tongue, air and ear.

Thesis
    A circle of forgetful hearts keeps a teaching only by voicing it into a shared,
    reverberant room and hearing it back from itself and the others, so the store is
    the air between the rememberers, not any one of them.

Evidence and provenance (mediated)
    No securely authentic words of Ahmed Yasawi survive. The Divan-i Hikmat is a late
    Chaghatay compilation holding many poets' verse (DeWeese, foreword to Koprulu,
    Early Mystics in Turkish Literature, 2006, pp. xix-xxii); the earliest reference
    (Burhan al-Din Qilich, 13th c.) knows him chiefly as a Sufi shaykh (DeWeese 2013).
    What the tradition that bears his name defended in his name is vocal remembrance:
    D1  dhikr-i jahr and its rasping 'saw' form, dhikr-i arra, which Ahmad Kasani
        (d. 1542) said Yasawi introduced (Fletcher 1977; Weismann 2007, ch. 3).
    D2  the earliest internal Yasawi text, Ta'lim al-dhakirin (947/1541), is a defence
        of vocal dhikr (DeWeese 2006 foreword, p. xiii).
    D3  the rival Khwajagan rule: remembrance silent in the heart, kept even amid people
        (khalvat dar anjuman, hush dar dam; Algar 1976, Trimingham 1971).
    D4  the hikmat corpus grew in the community's mouth; aphorisms first credited to
        Hakim Ata or 'the Turkic shaykhs' migrate to Yasawi (DeWeese 2006, p. xxi).
    D5  dhikr as the remedy for heedlessness (ghaflat): the heart forgets unless it is
        reminded (general Sufi doctrine; Schimmel 1975, ch. 4).
    D6  withdrawal into an underground cell at 63 (hagiography; DeWeese 2006, p. xviii).
    The mechanism rests on these later accounts, not on a psychology he wrote.

Doctrine -> mechanism -> test
    D1, D2  M1 tongue -> reverberant room -> ear loop   C6.2 latency, C6.4 room   H-SIG, H-NEC
    D4      M2 anonymous summed chorus, shared weights  C6.1 anonymity            H-BLIND
    D5      M3 forgetful heart, leak >= 0.5, no inner   C6.3 (definition check)  H-HYBRID
            recurrence
    D3      rival: silent gated heart (khafi)            -                        H-RIVAL, H-RIVAL-B

Research question (memory, retrieval and forgetting)
    When resettable agents must keep a goal through lapses, does keeping it only as a
    self-heard voice in a shared reverberant channel beat internal gated memory with
    communication, and what does it cost when the channel must carry other speech?

Closest prior art and the delta
    CommNet (Sukhbaatar, Szlam and Fergus 2016), implemented with GRU cells as the H-SIG
    baseline; output-feedback memory in recurrent nets (Maass, Joshi and Sontag 2007;
    Sussillo and Abbott 2009); learned inter-agent messages (Foerster et al. 2016); the
    phonological loop (Baddeley and Hitch 1974). Delta: the only memory beyond a two-step
    leak is an anonymous, self-heard, reverberant channel outside every heart, so
    rehearsal is mandatory and the store survives a lapse of all hearts at once.

Blind spot
    The Khwajagan objection: a remembrance that needs the tongue fails when the tongue is
    taken by the market. Condition 'market' and 'bazaar' take each tongue with p = 0.5
    and fill the room with worldly speech.

Task (generative process)
    Six Names are random 8-d vectors. At t = 0 each of K = 4 members hears the Name
    (p = 0.1 a wrong one) plus noise and a flag bit. At t >= 1 each sees worldly
    impressions (another Name at 0.8 gain with p = 0.6, plus noise). Heedlessness wipes
    a member's heart with p_i per step and, from t = 2, all hearts at once with p_g per
    step. In the market each tongue is taken with p_m per step and speaks noise. Every
    member must name the true Name at every step t >= 1; accuracy is read at the last
    step. Training mixes calm, khanaqah and market with T drawn from 6..12; 'long'
    (T = 18) is the shifted split.

Limits
    Synthetic task; four members; a research prototype of an AGI-oriented memory
    architecture, not an AGI, and not a claim to replicate a person. The large gap under
    collective lapses follows from where the store lives; what the tests add is whether
    the loop learns several stable attractors, extrapolates in length, and what the
    tongue's absence costs. Gradient checks use |fd-an| / max(|fd|+|an|, 1e-3) <= 1e-5: the
    floor absorbs central-difference rounding noise (~5e-9 at eps 1e-6) on near-zero
    entries of gate tensors; every tensor's largest-gradient entry is still checked.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 2,
    "revision_log": ["r2 (after the first --quick run): C1 denominator floor raised from 1e-7 to 1e-3 because 3/98 "
                     "tensor checks failed only on near-zero gate entries (|g| ~ 1e-6, central-difference rounding "
                     "noise); the 1e-5 bound, all hypotheses, metrics and hypothesis thresholds are unchanged."],
    "generation": {"template_version": "codeguidelines 1.0 (2026-09-15)", "generator": "Claude (Anthropic)",
                   "generator_version": "claude-opus-5-5", "date": "2026-09-23"},
        "id": 286, "figure": "Ahmed Yasawi", "born": 1093, "died": 1166,
    "civilization": "Turkic Muslim (middle Syr Darya; Karakhanid domains, Qara Khitai suzerainty after 1141)",
    "provenance": "mediated",
    "thesis": "A circle of forgetful hearts keeps a teaching only by voicing it into a shared reverberant room "
              "and hearing it back from itself and the others; the store is the air between them.",
    "evidence": [
        {"id": "D1", "claim": "Vocal dhikr, incl. the rasping 'saw' dhikr-i arra said to be introduced by Yasawi, marks the Yasawi path",
         "basis": "scholarship", "source": "Fletcher 1977, JTS 1: 113-119; Weismann 2007, The Naqshbandiyya, ch. 3"},
        {"id": "D2", "claim": "Earliest internal Yasawi text (Ta'lim al-dhakirin, 947/1541) defends vocal dhikr",
         "basis": "scholarship", "source": "DeWeese, foreword to Koprulu, Early Mystics (2006), p. xiii"},
        {"id": "D3", "claim": "Khwajagan rival: silent dhikr kept in the heart even amid people (khalvat dar anjuman)",
         "basis": "scholarship", "source": "Algar 1976, 'Silent and Vocal Dhikr in the Naqshbandi Order'; Trimingham 1971"},
        {"id": "D4", "claim": "Hikmat corpus is a communal, late compilation; sayings migrate to Yasawi's name",
         "basis": "scholarship", "source": "DeWeese 2006 foreword, pp. xix-xxii"},
        {"id": "D5", "claim": "Dhikr as remedy for heedlessness: the heart forgets unless reminded",
         "basis": "scholarship", "source": "Schimmel, Mystical Dimensions of Islam (1975), ch. 4"},
        {"id": "D6", "claim": "Retreat to an underground cell at age 63", "basis": "speculation",
         "source": "hagiography, discussed by DeWeese 2006 foreword, p. xviii"},
    ],
    "research_question": {"category": "memory, retrieval and forgetting",
                          "question": "Does keeping a goal only as a self-heard voice in a shared reverberant channel beat "
                                      "internal gated memory with communication under lapses, and what does it cost when "
                                      "the channel must carry other speech?"},
    "mechanism": {
        "name": "HALQA-JAHR voiced room",
        "family": "multi-agent recurrent memory by externalized rehearsal (output feedback through a shared reverberant channel)",
        "signature_modules": ["tongue", "ear"],
        "closest_prior_art": ["CommNet (Sukhbaatar, Szlam, Fergus 2016)",
                              "output-feedback memory (Maass, Joshi, Sontag 2007; Sussillo, Abbott 2009)",
                              "DIAL/RIAL learned communication (Foerster et al. 2016)",
                              "phonological loop (Baddeley, Hitch 1974; Burgess, Hitch 1999)"],
        "overlap": "Medium",
        "prior_art_queries": ["externalized rehearsal recurrent network shared channel", "output feedback memory attractor RNN",
                              "multi-agent communication memory reset recovery", "phonological loop neural network model",
                              "gossip protocol state recovery after reboot"],
        "contribution_type": "mechanism",
        "delta": "The heart has no inner recurrence beyond a two-step leak; the only long store is an anonymous, self-heard, "
                 "reverberant room outside all hearts, which survives a lapse of every heart at once.",
    },
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1", "property_test": "C6.2", "hypothesis": "H-SIG"},
        {"doctrine": "D2", "mechanism": "M1", "property_test": "C6.4", "hypothesis": "H-NEC"},
        {"doctrine": "D4", "mechanism": "M2", "property_test": "C6.1", "hypothesis": "H-BLIND"},
        {"doctrine": "D5", "mechanism": "M3", "property_test": "C6.3", "hypothesis": "H-HYBRID"},
        {"doctrine": "D3", "mechanism": "rival", "property_test": "-", "hypothesis": "H-RIVAL"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "The voiced circle recalls better than CommNet-GRU on long episodes with lapses",
         "metric": "final-step accuracy", "split": "long", "comparison": "model - baseline", "direction": "greater",
         "mesi": 0.05, "seeds": 5},
        {"id": "H-NEC", "statement": "Knocking out the ear hurts more than knocking out the heart's own carry",
         "metric": "final-step accuracy on long", "comparison": "signature_knockout - matched_knockout",
         "direction": "less", "mesi": 0.10, "seeds": 5},
        {"id": "H-BLIND", "statement": "Taking the tongue (market) erodes the advantage over CommNet-GRU",
         "condition": "market vs khanaqah", "grounding": "Khwajagan objection: remembrance must survive worldly speech",
         "metric": "(Y-CN) on market minus (Y-CN) on khanaqah", "comparison": "model - baseline", "direction": "less",
         "mesi": 0.05, "seeds": 5},
        {"id": "H-RIVAL", "statement": "In the lodge with lapses the loud circle beats the silent heart",
         "metric": "accuracy on khanaqah", "comparison": "model - rival", "direction": "greater", "mesi": 0.05, "seeds": 5},
        {"id": "H-RIVAL-B", "statement": "In a bazaar without lapses the silent heart beats the loud circle",
         "metric": "accuracy on bazaar", "comparison": "model - rival", "direction": "less", "mesi": 0.02, "seeds": 5},
        {"id": "H-HYBRID", "statement": "A gated heart that also speaks beats the forgetful heart on long episodes",
         "metric": "accuracy on long", "comparison": "model - hybrid", "direction": "less", "mesi": 0.02, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.3, "margin_over_trivial": 0.30, "shuffled_band": 0.06},
    "probe_predictions": [{"probe": "P1", "expected": "equal to baseline"}, {"probe": "P4", "expected": "above baseline"},
                          {"probe": "P9", "expected": "above baseline"}, {"probe": "P7", "expected": "below baseline"},
                          {"probe": "P2", "expected": "below baseline"}, {"probe": "P8", "expected": "equal to baseline"}],
    "dialectic_links": [
        {"chapter": None, "figure": "'Abd al-Khaliq Ghijduvani / Khwajagan (not in corpus)", "relation": "rival",
         "test": "H-RIVAL, H-RIVAL-B"},
        {"chapter": 341, "relation": "successor", "test": "none (Koprulu's influence thesis is contested)"},
    ],
    "corpus_neighbors": [
        {"chapter": 148, "similarity": None, "difference": "Ambrose: alternating call-and-response and a metrical codebook; here no alternation, no codebook, and the heart cannot store"},
        {"chapter": 228, "similarity": None, "difference": "Naum: knowledge in phase relations between learners; here content amplitude held in a physical room"},
        {"chapter": 329, "similarity": None, "difference": "Rumi: subtractive resonator needing another's breath; here a self-heard voice suffices"},
        {"chapter": 284, "similarity": None, "difference": "Avempace: convergence without any communication; here convergence only through the shared voice"},
        {"chapter": 131, "similarity": None, "difference": "Marcus Aurelius: protected inner homeostat; here nothing inner is protected"},
    ],
    "similarity_review": "9-token shingle Jaccard 0.033 against the one prior file available (0507 sample); "
                         "corpus files 0001-0284 were not available to this session, so neighbour similarity is null.",
    "barometer": {"cognitive_processing": ["P1", "P4"], "embodied_cognition": ["P9"], "world_modeling": [],
                  "consciousness": [], "language_understanding": [], "emotional_intelligence": [],
                  "creativity": [], "autonomy": ["P5"]},
    "task_types": ["sequence_classification"],
    "applications": [
        {"use": "multi-robot teams re-broadcasting mission state so a rebooted robot recovers", "sector": "logistics robotics",
         "dataset": "PettingZoo MPE benchmark (Lowe et al. 2017)"},
        {"use": "sensor networks keeping a shared estimate through node resets", "sector": "environmental IoT",
         "dataset": "Intel Berkeley Research Lab sensor data (2004)"},
        {"use": "long-horizon software agents re-reading a shared plan after context resets; injection audit",
         "sector": "software agents", "dataset": "SWE-bench (Jimenez et al. 2024)"},
    ],
    "safety_notes": "No claim to replicate a person; doctrine is a later tradition's. The anonymous chorus is an injection "
                    "surface (a stranger's voice is indistinguishable); not for crowd persuasion or surveillance.",
}

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

N_NAMES, D_IN, K_MEMBERS, V_VOICE = 6, 9, 4, 4
RHO = 0.5            # room reverberation: half of the last breath still hangs in the air
ALPHA_MIN = 0.5      # the heart keeps at most half of itself per breath
P_MIS = 0.1
LR, CLIP, BATCH, VAL_EVERY = 0.01, 5.0, 32, 200
STEPS_FULL, STEPS_QUICK = 1000, 350
N_TEST = 500
COND = {
    "calm": dict(T=10, pi=0.0, pg=0.0, pm=0.0),
    "khanaqah": dict(T=10, pi=0.08, pg=0.10, pm=0.0),
    "market": dict(T=10, pi=0.08, pg=0.10, pm=0.5),
    "bazaar": dict(T=10, pi=0.0, pg=0.0, pm=0.5),
    "long": dict(T=18, pi=0.08, pg=0.10, pm=0.0),
}
TRAIN_CONDS = ("calm", "khanaqah", "market")
TASK_TYPES = ["sequence_classification"]
ARCH_SIZES = {"yasawi": 16, "commnet": 6, "khafi": 7, "hybrid": 6}


# BEGIN STANDARD UTILITIES v1.0
def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def logsumexp(z, axis=-1):
    m = z.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(z - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def softplus(x):
    return np.logaddexp(0.0, x)


class Adam:
    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-8):
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


def clip_global(grads, max_norm):
    norm = math.sqrt(sum(float((g ** 2).sum()) for g in grads.values()))
    scale = min(1.0, max_norm / (norm + 1e-12))
    return {k: g * scale for k, g in grads.items()}, norm


def fd_check(loss_fn, params, grads, rng, n_entries=20, eps=1e-6):
    out = {}
    for name, p in params.items():
        flat, g = p.reshape(-1), grads[name].reshape(-1)
        idx = range(flat.size) if flat.size <= n_entries + 1 else \
            list(rng.choice(flat.size, n_entries, replace=False)) + [int(np.argmax(np.abs(g)))]
        worst = 0.0
        for i in idx:
            old = flat[i]
            flat[i] = old + eps
            lp = loss_fn()
            flat[i] = old - eps
            lm = loss_fn()
            flat[i] = old
            fd = (lp - lm) / (2 * eps)
            worst = max(worst, abs(fd - g[i]) / max(abs(fd) + abs(g[i]), 1e-3))
        out[name] = worst
    return out


def bootstrap_ci(diffs, rng, n_boot=2000, alpha=0.05):
    diffs = np.asarray(diffs, dtype=float)
    idx = rng.integers(0, diffs.size, size=(n_boot, diffs.size))
    means = diffs[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def verdict(mean, ci, mesi, direction):
    sign = 1.0 if direction == "greater" else -1.0
    lo, hi = ci
    if (sign > 0 and lo > 0 or sign < 0 and hi < 0) and sign * mean >= mesi:
        return "supported"
    if sign > 0 and hi < 0 or sign < 0 and lo > 0:
        return "contradicted"
    return "inconclusive"


def write_report(path, report):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))
# END STANDARD UTILITIES


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _outer(a, b):
    return a.reshape(-1, a.shape[-1]).T @ b.reshape(-1, b.shape[-1])


# ---------------------------------------------------------------- data and tasks
def make_names(rng):
    return rng.normal(size=(N_NAMES, D_IN - 1)) * (1.6 / math.sqrt(D_IN - 1))


def sample_episodes(rng, names, cond, n, T=None):
    """A circle session: teaching at t=0, worldly impressions, heedlessness, taken tongues."""
    c = COND[cond]
    T = T or c["T"]
    y = rng.integers(0, N_NAMES, n)
    wrong = (y[:, None] + rng.integers(1, N_NAMES, (n, K_MEMBERS))) % N_NAMES
    heard = np.where(rng.random((n, K_MEMBERS)) < P_MIS, wrong, y[:, None])
    X = np.zeros((n, K_MEMBERS, T, D_IN))
    X[:, :, 0, :-1] = names[heard] + 0.3 * rng.normal(size=(n, K_MEMBERS, D_IN - 1))
    X[:, :, 0, -1] = 1.0
    other = rng.integers(0, N_NAMES, (n, K_MEMBERS, T - 1))
    on = rng.random((n, K_MEMBERS, T - 1)) < 0.6
    X[:, :, 1:, :-1] = 0.8 * on[..., None] * names[other] + 0.3 * rng.normal(size=(n, K_MEMBERS, T - 1, D_IN - 1))
    keep = np.ones((n, K_MEMBERS, T))
    keep[:, :, 1:] = rng.random((n, K_MEMBERS, T - 1)) >= c["pi"]
    gust = rng.random((n, T)) < c["pg"]
    gust[:, :2] = False
    keep *= 1.0 - gust[:, None, :]
    tongue = np.ones((n, K_MEMBERS, T))
    tongue[:, :, 1:] = rng.random((n, K_MEMBERS, T - 1)) >= c["pm"]
    world = rng.uniform(-0.9, 0.9, (n, K_MEMBERS, T, V_VOICE))
    earn = 0.1 * rng.normal(size=(n, K_MEMBERS, T, V_VOICE))
    return dict(X=X, y=y, keep=keep, tongue=tongue, world=world, earn=earn)


def as_batch(X, y=None):
    """Probe-battery bridge: (N,T,D) sequences become solitary circles (K=1, quiet room)."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 3:
        X = X[:, None]
    n, k, T, _ = X.shape
    z = np.zeros((n, k, T, V_VOICE))
    return dict(X=X, y=np.zeros(n, int) if y is None else np.asarray(y), keep=np.ones((n, k, T)),
                tongue=np.ones((n, k, T)), world=z, earn=z.copy())


def train_sampler(names, shuffle_labels=False, hashes=None):
    def draw(rng, it):
        T = int(rng.integers(6, 13))
        b = sample_episodes(rng, names, TRAIN_CONDS[it % 3], BATCH, T)
        if shuffle_labels:
            b["y"] = rng.permutation(b["y"])
        if hashes is not None:
            hashes.update(hashlib.md5(x.tobytes()).hexdigest() for x in b["X"])
        return b
    return draw


def val_batch(rng, names):
    parts = [sample_episodes(rng, names, c, 100) for c in TRAIN_CONDS]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


# ---------------------------------------------------------------- model: the voiced room
def room_step(r, v, nonlinear=False):
    """Reverberant air: an exact linear filter of the chorus with unit DC gain."""
    chorus = v.mean(axis=1)
    return np.tanh(3 * (RHO * r + (1 - RHO) * chorus)) if nonlinear else RHO * r + (1 - RHO) * chorus


def _ko(x, mode):
    if mode is None:
        return x
    if mode == "zero":
        return np.zeros_like(x)
    if mode == "mean":
        return np.broadcast_to(x.mean(axis=(0, 1), keepdims=True), x.shape)
    raise ValueError("knockout mode must be 'zero' or 'mean'")


def _init_yasawi(rng, H, in_dim, out_dim):
    g = 2.0  # loop gain large enough for the voice to re-light the heart
    return dict(Wx=rng.normal(size=(in_dim, H)) / math.sqrt(in_dim), b=np.zeros(H),
                Wa=rng.normal(size=(V_VOICE, H)) * g / math.sqrt(V_VOICE), th=np.zeros(H),
                Wv=rng.normal(size=(H, V_VOICE)) * g / math.sqrt(H), c=np.zeros(V_VOICE),
                Wo=rng.normal(size=(H, out_dim)) / math.sqrt(H), bo=np.zeros(out_dim))


def _yasawi_forward(model, B):
    P, ko, cfg = model["params"], model["ko"], model["cfg"]
    X = B["X"]
    n, k, T, _ = X.shape
    amin = cfg.get("alpha_min", ALPHA_MIN)
    al = amin + (1 - amin) * sigmoid(P["th"])
    gains = 1.0 + 0.3 * np.arange(k)[None, :, None] if cfg.get("named_ear") else 1.0
    h = np.zeros((n, k, P["b"].size))
    r = np.zeros((n, V_VOICE))
    cache, logits = [], []
    for t in range(T):
        a = r[:, None, :] + B["earn"][:, :, t]
        tg = B["tongue"][:, :, t, None]
        if cfg.get("latency") == 0:  # broken control: this breath's worldly speech leaks in at once
            a = a + (1 - RHO) * ((1 - tg) * B["world"][:, :, t]).mean(1, keepdims=True)
        hp = B["keep"][:, :, t, None] * h
        eye = _ko(X[:, :, t] @ P["Wx"], ko.get("eye"))
        ear = _ko(gains * (a @ P["Wa"]), ko.get("ear"))
        g = np.tanh(eye + ear + P["b"])
        h = _ko((1 - al) * hp, ko.get("leak")) + al * g
        u = _ko(np.tanh(h @ P["Wv"] + P["c"]), ko.get("tongue"))
        v = tg * u + (1 - tg) * B["world"][:, :, t]
        if model.get("mutant") != "deaf_room":
            r = room_step(r, v)
        cache.append((a, hp, g, h, u))
        logits.append(_ko(h @ P["Wo"] + P["bo"], ko.get("readout")))
    return logits, cache, al


def _dense_ce(logits, y):
    """Answer at every breath after the teaching; returns loss and per-step logit gradients."""
    T, (n, k, C) = len(logits), logits[0].shape
    Y = np.zeros((n, k, C))
    Y[np.arange(n)[:, None], np.arange(k)[None, :], np.broadcast_to(y.reshape(n, -1), (n, k))] = 1.0
    loss, dls = 0.0, [None]
    for lo in logits[1:]:
        lse = logsumexp(lo)
        loss += float(np.mean(lse - (lo * Y).sum(-1))) / (T - 1)
        dls.append((softmax(lo) - Y) / (n * k * (T - 1)))
    return loss, dls


def _yasawi_backward(model, B, logits, cache, al):
    P, mut = model["params"], model.get("mutant")
    loss, dls = _dense_ce(logits, B["y"])
    X = B["X"]
    n, k, T, _ = X.shape
    G = {name: np.zeros_like(p) for name, p in P.items()}
    dh, dr, dal = np.zeros_like(cache[0][3]), np.zeros((n, V_VOICE)), np.zeros_like(al)
    for t in range(T - 1, -1, -1):
        a, hp, g, h, u = cache[t]
        if dls[t] is not None:
            G["Wo"] += _outer(h, dls[t])
            G["bo"] += dls[t].sum((0, 1))
            dh = dh + dls[t] @ P["Wo"].T
        ds = B["tongue"][:, :, t, None] * np.repeat(((1 - RHO) * dr / k)[:, None], k, 1) * (1 - u ** 2)
        G["Wv"] += _outer(h, ds)
        G["c"] += ds.sum((0, 1))
        dh = dh + ds @ P["Wv"].T
        dal += (dh * (g - hp)).sum((0, 1))
        dz = dh * al * (1 - g ** 2)
        G["Wx"] += _outer(X[:, :, t], dz)
        G["Wa"] += _outer(a, dz)
        G["b"] += dz.sum((0, 1))
        dr = RHO * dr + (dz @ P["Wa"].T).sum(1)
        dh = B["keep"][:, :, t, None] * dh * (1 - al)
    s = sigmoid(P["th"])
    amin = model["cfg"].get("alpha_min", ALPHA_MIN)
    G["th"] = dal if mut == "leak_grad_bug" else dal * (1 - amin) * s * (1 - s)
    if mut == "zero_grad_ear":
        G["Wa"] = np.zeros_like(G["Wa"])
    return loss, G


# ---------------------------------------------------------------- baselines and rivals (one gated family)
def _init_gru(rng, H, in_dim, out_dim, comm, voice):
    P = {}
    for gate in "zrn":
        P["Wx" + gate] = rng.normal(size=(in_dim, H)) / math.sqrt(in_dim)
        P["Wh" + gate] = rng.normal(size=(H, H)) / math.sqrt(H)
        P["b" + gate] = np.zeros(H)
        if comm:
            P["Wc" + gate] = rng.normal(size=(H, H)) * 0.5 / math.sqrt(H)
        if voice:
            P["Wa" + gate] = rng.normal(size=(V_VOICE, H)) * 2.0 / math.sqrt(V_VOICE)
    if voice:
        P["Wv"] = rng.normal(size=(H, V_VOICE)) * 2.0 / math.sqrt(H)
        P["c"] = np.zeros(V_VOICE)
    P["Wo"] = rng.normal(size=(H, out_dim)) / math.sqrt(H)
    P["bo"] = np.zeros(out_dim)
    return P


def _gru_forward(model, B):
    """CommNet-GRU (comm), silent khafi heart (neither) or hybrid (voice): gated inner memory."""
    P, ko = model["params"], model["ko"]
    comm, voice = "Wcz" in P, "Wv" in P
    X = B["X"]
    n, k, T, _ = X.shape
    h = np.zeros((n, k, P["bz"].size))
    r = np.zeros((n, V_VOICE))
    cache, logits = [], []
    for t in range(T):
        x, hp = X[:, :, t], B["keep"][:, :, t, None] * h
        pre = {g: x @ P["Wx" + g] + P["b" + g] for g in "zrn"}
        cm = a = u = None
        if comm:
            cm = _ko((hp.sum(1, keepdims=True) - hp) / max(k - 1, 1), ko.get("comm"))
            for g in "zrn":
                pre[g] = pre[g] + cm @ P["Wc" + g]
        if voice:
            a = _ko(r[:, None, :] + B["earn"][:, :, t], ko.get("ear"))
            for g in "zrn":
                pre[g] = pre[g] + a @ P["Wa" + g]
        zg = sigmoid(pre["z"] + hp @ P["Whz"])
        rg = sigmoid(pre["r"] + hp @ P["Whr"])
        nn = np.tanh(pre["n"] + (rg * hp) @ P["Whn"])
        h = (1 - zg) * nn + zg * hp
        if voice:
            u = np.tanh(h @ P["Wv"] + P["c"])
            tg = B["tongue"][:, :, t, None]
            r = room_step(r, tg * u + (1 - tg) * B["world"][:, :, t])
        cache.append((x, hp, cm, a, zg, rg, nn, h, u))
        logits.append(_ko(h @ P["Wo"] + P["bo"], ko.get("readout")))
    return logits, cache, None


def _gru_backward(model, B, logits, cache, _unused):
    P = model["params"]
    comm, voice = "Wcz" in P, "Wv" in P
    loss, dls = _dense_ce(logits, B["y"])
    n, k = B["X"].shape[:2]
    G = {name: np.zeros_like(p) for name, p in P.items()}
    dh, dr = np.zeros_like(cache[0][7]), np.zeros((n, V_VOICE))
    for t in range(len(cache) - 1, -1, -1):
        x, hp, cm, a, zg, rg, nn, h, u = cache[t]
        if dls[t] is not None:
            G["Wo"] += _outer(h, dls[t])
            G["bo"] += dls[t].sum((0, 1))
            dh = dh + dls[t] @ P["Wo"].T
        if voice:
            ds = B["tongue"][:, :, t, None] * np.repeat(((1 - RHO) * dr / k)[:, None], k, 1) * (1 - u ** 2)
            G["Wv"] += _outer(h, ds)
            G["c"] += ds.sum((0, 1))
            dh = dh + ds @ P["Wv"].T
        dhp = dh * zg
        dA = {"n": dh * (1 - zg) * (1 - nn ** 2)}
        drh = dA["n"] @ P["Whn"].T
        dhp += drh * rg
        dA["z"] = dh * (hp - nn) * zg * (1 - zg)
        dA["r"] = drh * hp * rg * (1 - rg)
        G["Whn"] += _outer(rg * hp, dA["n"])
        for g in "zrn":
            G["Wx" + g] += _outer(x, dA[g])
            G["b" + g] += dA[g].sum((0, 1))
        for g in "zr":
            G["Wh" + g] += _outer(hp, dA[g])
            dhp += dA[g] @ P["Wh" + g].T
        if comm:
            dc = sum(dA[g] @ P["Wc" + g].T for g in "zrn")
            for g in "zrn":
                G["Wc" + g] += _outer(cm, dA[g])
            dhp += (dc.sum(1, keepdims=True) - dc) / max(k - 1, 1)
        if voice:
            for g in "zrn":
                G["Wa" + g] += _outer(a, dA[g])
            dr = RHO * dr + sum(dA[g] @ P["Wa" + g].T for g in "zrn").sum(1)
        dh = B["keep"][:, :, t, None] * dhp
    return loss, G


# ---------------------------------------------------------------- common interface
def build_model(in_dim, out_dim, task_type, rng, **cfg):
    if task_type not in TASK_TYPES:
        raise ValueError("unsupported task type")
    arch = cfg.get("arch", "yasawi")
    H = cfg.get("H", ARCH_SIZES[arch])
    if arch == "yasawi":
        params = _init_yasawi(rng, H, in_dim, out_dim)
    else:
        params = _init_gru(rng, H, in_dim, out_dim, comm=arch == "commnet", voice=arch == "hybrid")
    return {"arch": arch, "params": params, "cfg": dict(cfg), "ko": {}, "mutant": cfg.get("mutant")}


def _forward(model, B):
    return (_yasawi_forward if model["arch"] == "yasawi" else _gru_forward)(model, B)


def loss_and_grads(model, batch):
    logits, cache, extra = _forward(model, batch)
    back = _yasawi_backward if model["arch"] == "yasawi" else _gru_backward
    return back(model, batch, logits, cache, extra)


def predict(model, X):
    B = X if isinstance(X, dict) else as_batch(X)
    out = _forward(model, B)[0][-1].argmax(-1)
    return out[:, 0] if not isinstance(X, dict) and np.asarray(X).ndim == 3 else out


def hidden_states(model, X):
    B = X if isinstance(X, dict) else as_batch(X)
    cache = _forward(model, B)[1]
    return np.stack([c[3] if model["arch"] == "yasawi" else c[7] for c in cache], axis=2)


def n_params(model):
    return int(sum(p.size for p in model["params"].values()))


def accuracy(model, B):
    return float(np.mean(_forward(model, B)[0][-1].argmax(-1) == B["y"][:, None]))


# ---------------------------------------------------------------- registries
def modules(model):
    if model["arch"] == "yasawi":
        return {
            "eye": {"params": ["Wx", "b"], "role": "input projection", "signature": False},
            "ear": {"params": ["Wa"], "role": "shared-channel read-in (delayed broadcast input)", "signature": True},
            "tongue": {"params": ["Wv", "c"], "role": "bounded write-out to shared reverberant channel", "signature": True},
            "leak": {"params": ["th"], "role": "per-unit leak rate of a non-recurrent state (carry <= 0.5)", "signature": False},
            "readout": {"params": ["Wo", "bo"], "role": "linear classifier", "signature": False},
        }
    names = {"input": ["Wxz", "Wxr", "Wxn", "bz", "br", "bn"], "recurrent": ["Whz", "Whr", "Whn"],
             "readout": ["Wo", "bo"]}
    P = model["params"]
    if "Wcz" in P:
        names["comm"] = ["Wcz", "Wcr", "Wcn"]
    if "Wv" in P:
        names["ear"], names["tongue"] = ["Waz", "War", "Wan"], ["Wv", "c"]
    return {m: {"params": p, "role": "gated recurrent " + m, "signature": False} for m, p in names.items()}


def knockout(model, name, mode="mean"):
    if name not in modules(model):
        raise KeyError(name)
    clone = {**model, "params": {k: v.copy() for k, v in model["params"].items()}, "ko": {**model["ko"], name: mode}}
    return clone


MUTANTS = {
    "sign_flip": "optimizer ascends the loss",
    "zero_lr": "learning rate set to zero",
    "zero_grad_ear": "analytic gradient of the ear (Wa) zeroed",
    "leak_grad_bug": "chain factor of the leak parametrisation dropped",
    "deaf_room": "the room is never updated, so the voice never returns",
}


# ---------------------------------------------------------------- training
def fit(model, data, budget, rng, lr=LR):
    """Adam with global-norm clipping; checkpoint chosen on the validation split only."""
    if "sampler" not in data:
        full = as_batch(data["X"], data["y"])

        def draw(g, it):
            i = g.integers(0, len(full["y"]), BATCH)
            return {k: v[i] for k, v in full.items()}
        data = {**data, "sampler": draw}
    mut = model.get("mutant")
    opt = Adam(model["params"], 0.0 if mut == "zero_lr" else lr)
    sign = -1.0 if mut == "sign_flip" else 1.0
    losses, best, best_val = [], None, np.inf
    for it in range(1, budget + 1):
        B = data["sampler"](rng, it)
        loss, grads = loss_and_grads(model, B)
        if not np.isfinite(loss) or not all(np.all(np.isfinite(g)) for g in grads.values()):
            raise FloatingPointError("non-finite loss or gradient")
        grads, _ = clip_global(grads, CLIP)
        opt.step(model["params"], grads, sign)
        losses.append(loss)
        if data.get("val") is not None and (it % VAL_EVERY == 0 or it == budget):
            vl = loss_and_grads(model, data["val"])[0]
            if vl < best_val:
                best_val, best = vl, {k: v.copy() for k, v in model["params"].items()}
    if best is not None:
        model["params"] = best
    return losses


def train_arch(arch, names, budget, rng, val, mutant=None, shuffle=False, hashes=None):
    model = build_model(D_IN, N_NAMES, "sequence_classification", rng, arch=arch, mutant=mutant)
    losses = fit(model, {"sampler": train_sampler(names, shuffle, hashes), "val": val}, budget, rng)
    return model, losses


# ---------------------------------------------------------------- correctness tests
def c1_gradcheck(names, rng, steps=60):
    worst, checked, total = 0.0, 0, 0
    B = sample_episodes(rng, names, "market", 3, T=6)
    for arch in ARCH_SIZES:
        model = build_model(D_IN, N_NAMES, "sequence_classification", rng, arch=arch)
        for stage in ("init", "trained"):
            if stage == "trained":
                fit(model, {"sampler": train_sampler(names)}, steps, rng)
            _, grads = loss_and_grads(model, B)
            errs = fd_check(lambda: loss_and_grads(model, B)[0], model["params"], grads, rng)
            worst = max(worst, max(errs.values()))
            checked += int(sum(e <= 1e-5 for e in errs.values()))
            total += len(errs)
    return worst <= 1e-5, dict(max_rel_error=worst, tensors_checked=checked, tensors_total=total)


def gradcheck_one(model, names, rng):
    B = sample_episodes(rng, names, "market", 3, T=6)
    _, grads = loss_and_grads(model, B)
    errs = fd_check(lambda: loss_and_grads(model, B)[0], model["params"], grads, rng)
    return max(errs.values()) <= 1e-5


def c3_learning(losses, acc_calm, trivial):
    first, last = np.mean(losses[:50]), np.mean(losses[-50:])
    drop = 1.0 - last / first
    ok = drop >= MIND_CARD["thresholds"]["loss_drop_fraction"] and \
        acc_calm >= trivial + MIND_CARD["thresholds"]["margin_over_trivial"]
    return ok, dict(loss_first=first, loss_last=last, drop=drop, acc_calm=acc_calm, trivial=trivial)


def trivial_accuracy(B, train_counts):
    return float(np.mean(B["y"] == int(np.argmax(train_counts))))


def _perm_batch(B, perm):
    return {k: (v[:, perm] if v.ndim >= 2 else v) for k, v in B.items()}


def c6_1_anonymity(names, rng):
    """The chorus cannot tell who sang: permuting members permutes every output."""
    def worst_violation(named):
        worst = 0.0
        for _ in range(12):
            m = build_model(D_IN, N_NAMES, "sequence_classification", rng, arch="yasawi", named_ear=named)
            for p in m["params"].values():
                p *= rng.uniform(0.5, 3.0)
            B = sample_episodes(rng, names, "market", 3, T=6)
            perm = rng.permutation(K_MEMBERS)
            out = _forward(m, B)[0][-1][:, perm]
            out_p = _forward(m, _perm_batch(B, perm))[0][-1]
            worst = max(worst, float(np.abs(out - out_p).max()))
        return worst
    base, control = worst_violation(False), worst_violation(True)
    return base < 1e-10 and control > 1e-3, dict(violation=base, negative_control=control)


def c6_2_latency(names, rng):
    """A breath reaches hearts only at the next step: no same-step leak of speech."""
    def violations(latency):
        early, late = 0.0, 0.0
        for _ in range(8):
            m = build_model(D_IN, N_NAMES, "sequence_classification", rng, arch="yasawi", latency=latency)
            B = sample_episodes(rng, names, "bazaar", 3, T=8)
            B["tongue"][:, 0, 4] = 0.0
            B2 = {k: v.copy() for k, v in B.items()}
            B2["world"][:, 0, 4] += 0.5
            h1, h2 = hidden_states(m, B), hidden_states(m, B2)
            early = max(early, float(np.abs(h1[:, :, :5] - h2[:, :, :5]).max()))
            late = max(late, float(np.abs(h1[:, :, 5] - h2[:, :, 5]).max()))
        return early, late
    e, l = violations(1)
    ec, _ = violations(0)
    return e < 1e-12 and l > 1e-6 and ec > 1e-6, dict(same_step_leak=e, next_step_effect=l, negative_control=ec)


def c6_3_forgetful_heart(names, rng):
    """Definition check (holds by construction): deaf, the heart forgets at rate <= 1 - alpha_min."""
    def worst_ratio(cfg):
        worst = 0.0
        for _ in range(6):
            m = knockout(build_model(D_IN, N_NAMES, "sequence_classification", rng, **cfg), "ear", "zero")
            if cfg.get("alpha_min") == 0.0:
                m["params"]["th"][:] = -6.0
            B = sample_episodes(rng, names, "calm", 2, T=8)
            B2 = {k: v.copy() for k, v in B.items()}
            B2["X"][:, :, 0] += rng.normal(size=B2["X"][:, :, 0].shape)
            d = np.abs(hidden_states(m, B) - hidden_states(m, B2)).max(axis=(0, 1, 3))
            worst = max(worst, float(max(d[t] / (d[0] * (1 - ALPHA_MIN) ** t) for t in range(1, 8))))
        return worst
    base, control = worst_ratio({"arch": "yasawi"}), worst_ratio({"arch": "yasawi", "alpha_min": 0.0})
    return base <= 1 + 1e-9 and control > 1.5, dict(max_ratio=base, negative_control=control, label="definition check")


def c6_4_room(rng):
    """The air is linear with unit DC gain: superposition and a constant chorus settle to itself."""
    def violations(nonlinear):
        worst = 0.0
        for _ in range(10):
            v1, v2 = rng.normal(size=(2, 12, 3, K_MEMBERS, V_VOICE))
            r1 = r2 = r12 = np.zeros((3, V_VOICE))
            rc = np.zeros((3, V_VOICE))
            const = np.broadcast_to(v1[0], (3, K_MEMBERS, V_VOICE)) * 0.3
            for t in range(12):
                r1, r2 = room_step(r1, v1[t], nonlinear), room_step(r2, v2[t], nonlinear)
                r12 = room_step(r12, v1[t] + v2[t], nonlinear)
            for _t in range(80):
                rc = room_step(rc, const, nonlinear)
            worst = max(worst, float(np.abs(r12 - r1 - r2).max()), float(np.abs(rc - const.mean(1)).max()))
        return worst
    base, control = violations(False), violations(True)
    return base < 1e-10 and control > 1e-3, dict(violation=base, negative_control=control)


def c2_determinism(names, seed):
    runs = []
    for _ in range(2):
        rng = np.random.default_rng(seed)
        m, losses = train_arch("yasawi", names, 40, rng, None)
        B = sample_episodes(np.random.default_rng(seed + 1), names, "khanaqah", 20)
        runs.append((losses, _forward(m, B)[0][-1], m))
    same = runs[0][0] == runs[1][0] and np.array_equal(runs[0][1], runs[1][1])
    finite = all(np.all(np.isfinite(p)) for p in runs[0][2]["params"].values())
    return same and finite, dict(identical=same, finite=finite)


def c5_mutants(names, rng_seed, budget, val, test_calm, trivial):
    detected = {}
    for name in MUTANTS:
        rng = np.random.default_rng(rng_seed)
        m, losses = train_arch("yasawi", names, budget, rng, val, mutant=name)
        c1_ok = gradcheck_one(m, names, np.random.default_rng(rng_seed + 7))
        c3_ok = c3_learning(losses, accuracy(m, test_calm), trivial)[0]
        detected[name] = not (c1_ok and c3_ok)
    return all(detected.values()), detected


# ---------------------------------------------------------------- hypotheses
def run_seed(seed, budget, hashes=None):
    ss = np.random.SeedSequence(seed).spawn(4)
    task_rng, train_rng, val_rng, test_rng = (np.random.default_rng(s) for s in ss)
    names = make_names(task_rng)
    val = val_batch(val_rng, names)
    tests = {c: sample_episodes(test_rng, names, c, N_TEST) for c in COND}
    models, losses = {}, {}
    for arch in ARCH_SIZES:
        models[arch], losses[arch] = train_arch(arch, names, budget, train_rng, val,
                                                hashes=hashes if arch == "yasawi" else None)
    acc = {a: {c: accuracy(m, tests[c]) for c, _ in COND.items()} for a, m in models.items()}
    ko = {mod: accuracy(knockout(models["yasawi"], mod, "mean"), tests["long"]) for mod in modules(models["yasawi"])}
    return dict(names=names, val=val, tests=tests, models=models, losses=losses, acc=acc, ko=ko)


def hypothesis_diffs(runs):
    d = {"H-SIG": [], "H-NEC": [], "H-BLIND": [], "H-RIVAL": [], "H-RIVAL-B": [], "H-HYBRID": []}
    for r in runs:
        A = r["acc"]
        d["H-SIG"].append(A["yasawi"]["long"] - A["commnet"]["long"])
        d["H-NEC"].append(r["ko"]["ear"] - r["ko"]["leak"])
        d["H-BLIND"].append((A["yasawi"]["market"] - A["commnet"]["market"])
                            - (A["yasawi"]["khanaqah"] - A["commnet"]["khanaqah"]))
        d["H-RIVAL"].append(A["yasawi"]["khanaqah"] - A["khafi"]["khanaqah"])
        d["H-RIVAL-B"].append(A["yasawi"]["bazaar"] - A["khafi"]["bazaar"])
        d["H-HYBRID"].append(A["yasawi"]["long"] - A["hybrid"]["long"])
    return d


def evaluate_hypotheses(runs, rng, quick):
    out, diffs = [], hypothesis_diffs(runs)
    for h in MIND_CARD["hypotheses"]:
        vals = diffs[h["id"]]
        mean = float(np.mean(vals))
        ci = bootstrap_ci(vals, rng) if len(vals) > 1 else (mean, mean)
        v = "not evaluated" if quick else verdict(mean, ci, h["mesi"], h["direction"])
        out.append({"id": h["id"], "metric": h["metric"], "mean_diff": mean, "ci95": list(ci), "mesi": h["mesi"],
                    "n_seeds": len(vals), "verdict": v})
    return out


def knockout_table(runs, rng):
    rows = []
    for mod, spec in modules(runs[0]["models"]["yasawi"]).items():
        ch = [r["ko"][mod] - r["acc"]["yasawi"]["long"] for r in runs]
        ci = bootstrap_ci(ch, rng) if len(ch) > 1 else (ch[0], ch[0])
        rows.append({"module": mod, "signature": spec["signature"], "metric_change": float(np.mean(ch)), "ci95": list(ci)})
    return rows


# ---------------------------------------------------------------- report
def print_report(rep, acc_table):
    print("=== VERIFIED REPORT · chapter 0286 ===")
    print("environment: python %s, numpy %s" % (rep["environment"]["python"], rep["environment"]["numpy"]))
    print("seeds: %s   runtime: %.1f s   n_params: %s" % (rep["seeds"], rep["runtime_s"], rep["n_params"]))
    g = rep["gradcheck"]
    print("gradcheck: %d/%d tensor checks, max rel error %.2e, at %s -> %s" % (
        g["tensors_checked"], g["tensors_total"], g["max_rel_error"], g["checked_at"], "pass" if g["passed"] else "FAIL"))
    print("correctness:")
    for c in rep["correctness"]:
        print("  %-5s %-28s %-4s %s" % (c["id"], c["name"], "pass" if c["passed"] else "FAIL", c["detail"]))
    m = rep["mutants"]
    print("mutation score: %d/%d" % (m["detected"], m["total"]))
    print("accuracy by condition (mean over seeds; chance 0.167):")
    print("  %-9s" % "model" + "".join("%10s" % c for c in COND))
    for arch, row in acc_table.items():
        print("  %-9s" % arch + "".join("%10.3f" % row[c] for c in COND))
    print("hypotheses:")
    for h in rep["hypotheses"]:
        print("  %-9s diff %+.3f  CI [%+.3f, %+.3f]  mesi %.2f  n=%d  %s" % (
            h["id"], h["mean_diff"], h["ci95"][0], h["ci95"][1], h["mesi"], h["n_seeds"], h["verdict"]))
    print("knockouts (yasawi, mean mode, change in long accuracy):")
    for k in rep["knockouts"]:
        print("  %-8s signature=%-5s %+.3f  CI [%+.3f, %+.3f]" % (
            k["module"], k["signature"], k["metric_change"], k["ci95"][0], k["ci95"][1]))
    print("task types: %s   exit code: %d" % (rep["task_types"], rep["exit_code"]))
    print("=== END REPORT ===")


def data_bridge(path, rng, budget):
    """Optional real-data bridge: CSV rows seq_id,t,label,f1..fD with equal-length sequences."""
    if not path or not os.path.exists(path):
        print("real-data bridge: no --data file given or found; skipped")
        return None
    raw = np.genfromtxt(path, delimiter=",", skip_header=1)
    seqs = sorted(set(raw[:, 0].astype(int)))
    X = np.stack([raw[raw[:, 0] == s][np.argsort(raw[raw[:, 0] == s][:, 1])][:, 3:] for s in seqs])
    y = np.array([int(raw[raw[:, 0] == s][0, 2]) for s in seqs])
    cut = int(0.8 * len(y))
    m = build_model(X.shape[-1], int(y.max()) + 1, "sequence_classification", rng)
    fit(m, {"X": X[:cut], "y": y[:cut]}, budget, rng)
    acc = float(np.mean(predict(m, X[cut:]) == y[cut:]))
    maj = float(np.mean(y[cut:] == np.bincount(y[:cut]).argmax()))
    print("real-data bridge: %d sequences, held-out accuracy %.3f (majority %.3f)" % (len(y), acc, maj))
    return acc


def run_protocol(args):
    t0 = time.time()
    quick = args.quick
    budget = STEPS_QUICK if quick else STEPS_FULL
    seeds = [args.seed] if quick else [args.seed + i for i in range(args.seeds)]
    for s in seeds:
        print("seed", s)
    hashes = set()
    runs = [run_seed(s, budget, hashes if i == 0 else None) for i, s in enumerate(seeds)]
    r0 = runs[0]
    names, val, tests = r0["names"], r0["val"], r0["tests"]
    counts = np.bincount(tests["calm"]["y"], minlength=N_NAMES)
    trivial = trivial_accuracy(tests["calm"], counts)
    aux = np.random.default_rng(args.seed + 1000)
    results = []
    ok1, g1 = c1_gradcheck(names, aux)
    results.append(("C1", "gradient_check", ok1, g1))
    ok2, g2 = c2_determinism(names, args.seed)
    results.append(("C2", "determinism_finiteness", ok2, g2))
    ok3, g3 = c3_learning(r0["losses"]["yasawi"], r0["acc"]["yasawi"]["calm"], trivial)
    results.append(("C3", "learning", ok3, g3))
    shuf, _ = train_arch("yasawi", names, budget, np.random.default_rng(args.seed + 5), val, shuffle=True)
    acc_shuf = accuracy(shuf, tests["calm"])
    band = MIND_CARD["thresholds"]["shuffled_band"]
    results.append(("C4", "shuffled_label_control", acc_shuf <= trivial + band, dict(acc=acc_shuf, trivial=trivial)))
    ok5, det = c5_mutants(names, args.seed + 11, budget, val, tests["calm"], trivial)
    results.append(("C5", "mutant_detection", ok5, det))
    for cid, name, fn in (("C6.1", "chorus_anonymity", lambda: c6_1_anonymity(names, aux)),
                          ("C6.2", "room_latency_causality", lambda: c6_2_latency(names, aux)),
                          ("C6.3", "forgetful_heart_definition", lambda: c6_3_forgetful_heart(names, aux)),
                          ("C6.4", "room_linear_unit_gain", lambda: c6_4_room(aux))):
        ok, detail = fn()
        results.append((cid, name, ok, detail))
    test_hashes = {hashlib.md5(x.tobytes()).hexdigest() for c in tests.values() for x in c["X"]}
    val_hashes = {hashlib.md5(x.tobytes()).hexdigest() for x in val["X"]}
    overlap = len(hashes & (test_hashes | val_hashes))
    results.append(("C7", "split_integrity", overlap == 0, dict(train_episodes=len(hashes), overlap=overlap)))
    runtime = time.time() - t0
    limit = 20.0 if quick else 180.0
    results.append(("C8", "budget", runtime <= limit, dict(runtime_s=round(runtime, 1), limit_s=limit)))
    hyp = evaluate_hypotheses(runs, np.random.default_rng(args.seed + 2000), quick)
    kos = knockout_table(runs, np.random.default_rng(args.seed + 3000))
    acc_table = {a: {c: float(np.mean([r["acc"][a][c] for r in runs])) for c in COND} for a in ARCH_SIZES}
    passed = all(r[2] for r in results)
    exit_code = 0 if passed else (3 if not results[-1][2] and all(r[2] for r in results[:-1]) else 1)
    rep = {
        "schema_version": "1.0", "chapter": 286, "file": os.path.basename(__file__),
        "card_revision": MIND_CARD["card_revision"],
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
        "seeds": seeds, "runtime_s": round(time.time() - t0, 2),
        "n_params": {a: n_params(m) for a, m in r0["models"].items()},
        "gradcheck": {**g1, "checked_at": ["init", "after_training_steps"], "passed": ok1},
        "correctness": [{"id": i, "name": n, "passed": bool(ok), "detail": json.dumps(d, default=float)[:160]}
                        for i, n, ok, d in results],
        "mutants": {"detected": int(sum(det.values())), "total": len(det), "score": float(np.mean(list(det.values())))},
        "hypotheses": hyp, "knockouts": kos, "task_types": TASK_TYPES,
        "accuracy_by_condition": acc_table, "exit_code": exit_code,
    }
    print_report(rep, acc_table)
    data_bridge(args.data, np.random.default_rng(args.seed + 4000), budget)
    if args.json:
        write_report(args.json, rep)
    return exit_code


def run_mutant(args):
    rng = np.random.default_rng(args.seed)
    names = make_names(rng)
    val = val_batch(rng, names)
    test = sample_episodes(rng, names, "calm", N_TEST)
    trivial = trivial_accuracy(test, np.bincount(test["y"], minlength=N_NAMES))
    budget = STEPS_QUICK if args.quick else STEPS_FULL
    m, losses = train_arch("yasawi", names, budget, rng, val, mutant=args.mutant)
    c1 = gradcheck_one(m, names, rng)
    c3, info = c3_learning(losses, accuracy(m, test), trivial)
    print("mutant %s: C1 %s, C3 %s %s" % (args.mutant, "pass" if c1 else "FAIL", "pass" if c3 else "FAIL",
                                          json.dumps(info, default=float)))
    return 0 if (c1 and c3) else 1


def main():
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--card", action="store_true")
    ap.add_argument("--mutant", type=str, default=None, choices=sorted(MUTANTS))
    ap.add_argument("--data", type=str, default=None)
    try:
        args = ap.parse_args()
    except SystemExit:
        return 2
    if args.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    if args.seeds < 1:
        return 2
    try:
        return run_mutant(args) if args.mutant else run_protocol(args)
    except FloatingPointError as err:
        print("non-finite values:", err)
        return 4


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0383 - ANDREI RUBLEV (1360-1428)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 20, Minds 381-400 Available on Amazon https://www.amazon.com/dp/B0HRS9ZHVG
# Resume and Interactive Demos at  https://artificiology.com/
# Videos of the playable minds at https://www.youtube.com/@LostMindsAI
# END ATTRIBUTION  
"""
THE SHARED BLUE (golubets) -- a communion network for chapter 0383, Andrei Rublev (1360-1428).

Thesis
    Rublev painted three persons with one face, told them apart only by how each
    inclines toward the others, and joined them by a blue all three wear: so a
    learner should route DIFFERENCE through relations, route UNITY through a
    channel that sees only what every member shares, and route nothing through
    the individual face -- because what all members share cancels out of every
    comparison between them and is invisible to relations by construction.

Evidence and provenance (mediated)
    No word written by Rublev survives. The record is two chronicle notices
    (1405, 1408), notices in the Lives of Nikon and in the Tale of the transfer
    of Sergius' relics, Joseph of Volokolamsk's late-15th-century report (on
    the word of the elder Spiridon) of his contemplative gazing at icons, the
    Stoglav council of 1551, and the paintings, above all the Trinity. Every
    mechanism below is read off the icon's composition and its documented
    reception; none is a doctrine he stated.

Doctrine -> mechanism -> test (IDs as in MIND_CARD)
    D1 one face, persons told apart by inclination -> M1 inclination:
       shared MLP on ordered PAIR DIFFERENCES, pooled symmetrically over the
       three guests -> C6.1 guest-permutation invariance, C6.2 common-shift
       blindness, H-SIG
    D3 blue present in all three garments (read as common nature) -> M2
       golubets: product-of-sigmoids intersection over ALL entities (soft AND)
       -> C6.3 intersection bound, H-NEC, H-RIVAL
    D2 earlier types marked one angel individually (cross-halo); Rublev drops
       the individual mark -> M4 no per-entity channel at all -> H-SIG vs a
       dual sensory+relational baseline
    D7 open front of the table (beholder's place) -> M3 the candidate sits in
       the fourth seat and enters both channels like the guests
    D5 angel identities disputed since the Stoglav (1551) -> H-BLIND

Research question (compositional and systematic generalization)
    Relational bottlenecks transfer relations to unseen entities but are blind
    to any property common to all relata; dual designs recover it by readmitting
    per-entity features, and with them identity and nuisance shortcuts. Does an
    intersection-only channel, with no per-entity channel, keep relational
    transfer under shift while recovering the common nature -- and what does it
    lose when the answer depends on WHICH entity is which?

Closest prior art and the delta
    Relational bottleneck (CoRelNet, Kerg et al. 2022; PrediNet, Shanahan et
    al. 2020), Dual Attention Transformer (Altabaa & Lafferty, ICML 2025),
    noisy-AND pooling (Kraus, Ba & Frey 2016). Delta: DAT's sensory channel is
    replaced by an intersection-only channel and the face channel is removed;
    relations are built from differences, so commonality is exactly invisible
    to them and the intersection is its only route.

Blind spot
    Identical faces make individual identity unrecoverable: the full model
    cannot say whether the LEFT guest has a property (the "naming the angels"
    condition), which a face-reading MLP solves easily.

Task: the Mamre table (generative process)
    world: random orthogonal mixing Q of a latent [identity code z, magnitude
    m, binary properties p, station offset t] into 24 observed features.
    scene: three guests + one candidate in the open fourth seat; a cue names
    the rule. R0 likeness: candidate is a copy of some guest. R1 inclination:
    candidate's magnitude exceeds every guest's. R2 communion: some property is
    held by all four (half the negatives are the same scene with that property
    removed from all four, so pairwise differences are identical). Nuisance:
    shared station offset, guest order, feature noise. Shifted split: unseen
    identity codes, magnitudes moved up, station offsets three times larger.

Limits
    Synthetic, small, binary tasks with pre-segmented entities. Results bear on
    the mechanism only; they are not evidence about the historical painter.
    A research prototype of an AGI-oriented component, not an AGI.
"""
MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 2,
    "revision_log": [
        "rev 2 (2026-10-01): a pilot on seeds 0-1, run before any five-seed evaluation, showed "
        "every architecture over-fitting rule R2 at 2,000 scenes per rule (train 0.99, held-out "
        "0.71). The training set was raised to 6,000 scenes per rule (18,000 total; 6,000 in "
        "--quick) and the naming task fixed at 3,000. The pilot also displayed communion-vs-dual "
        "accuracies on those two seeds, disclosed here because it could bias the change. "
        "Hypotheses, metrics, mesi and thresholds are unchanged from rev 1."],
    "generation": {"template_version": "codeguidelines 1.0 (15 Sep 2026)",
                   "generator": "Claude (Anthropic)",
                   "generator_version": "claude-opus-5-5", "date": "2026-10-01"},
    "id": 383, "figure": "Andrei Rublev", "born": 1360, "died": 1428,
    "civilization": "Russian (Muscovite)",
    "provenance": "mediated",
    "thesis": ("Difference travels in relations, unity in what all members share, "
               "nothing in the individual face: the common nature cancels out of "
               "every comparison and needs a channel of its own."),
    "evidence": [
        {"id": "D1", "claim": "The three angels of the Trinity share one face; they differ by "
         "inclination, gesture, position and garment colour.",
         "basis": "deeds", "source": "Trinity icon (Tretyakov dating 1422-1427; others c.1411); "
         "Lazarev 1966"},
        {"id": "D2", "claim": "Earlier Hospitality images showed frontal identical angels or marked "
         "the middle one as Christ with a cross-halo; Rublev ties each angel to a Person "
         "without an individual mark.", "basis": "scholarship",
         "source": "Bunge, The Rublev Trinity (2007); Khondzinsky, Vestnik PSTGU 2014 5(56)"},
        {"id": "D3", "claim": "Blue appears in all three garments; commentators read it as the "
         "shared nature (interpretation, not Rublev's statement).", "basis": "scholarship",
         "source": "interpretive tradition summarised in Bunge 2007"},
        {"id": "D4", "claim": "With Daniil he sat before icons on feast days gazing without turning "
         "aside, filled with joy and light, also on days without painting (mediated report).",
         "basis": "scholarship", "source": "Joseph of Volokolamsk, Otveshchanie "
         "liubozazornym (late 15th c.), via Lazarev 1966, 75-78"},
        {"id": "D5", "claim": "The Stoglav (1551) answered a question on cross-halos and "
         "inscriptions for the Trinity by pointing to the ancient models and Rublev.",
         "basis": "scholarship", "source": "Stoglav ch. 41, q. 1; Lazarev 1966"},
        {"id": "D6", "claim": "Always worked in company: with Theophanes and Prokhor (1405), with "
         "Daniil (1408; 1420s).", "basis": "deeds", "source": "Trinity Chronicle s.a. 1405, 1408"},
        {"id": "D7", "claim": "The open front of the table invites the beholder into the circle.",
         "basis": "speculation", "source": "common modern reading of the composition"},
    ],
    "research_question": {"category": "compositional and systematic generalization",
        "question": ("Can an intersection-only channel recover properties common to all relata, "
                     "which pairwise relations cannot see, without readmitting the per-entity "
                     "features that carry identity and nuisance shortcuts under shift?")},
    "mechanism": {
        "name": "Communion network (inclination + golubets, no face channel)",
        "family": "relational and graph learning / set functions",
        "signature_modules": ["inclination", "golubets"],
        "closest_prior_art": ["CoRelNet relational bottleneck (Kerg et al. 2022, arXiv:2206.05056)",
                              "PrediNet (Shanahan et al. 2020, arXiv:1905.10307)",
                              "Dual Attention Transformer (Altabaa & Lafferty, ICML 2025, arXiv:2405.16727)",
                              "Noisy-AND MIL pooling (Kraus, Ba & Frey 2016, Bioinformatics)"],
        "overlap": "Medium",
        "prior_art_queries": ["relational bottleneck attributes shared by all objects",
                              "relational and sensory information transformer",
                              "noisy-and pooling set intersection",
                              "higher-order relations beyond pairwise abstraction"],
        "contribution_type": "mechanism",
        "delta": ("Sensory channel replaced by an intersection-only channel; face channel removed; "
                  "relations from differences so the common nature is exactly invisible to them."),
    },
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1", "property_test": "C6.1", "hypothesis": "H-SIG"},
        {"doctrine": "D3", "mechanism": "M2", "property_test": "C6.3", "hypothesis": "H-NEC"},
        {"doctrine": "D3", "mechanism": "M2", "property_test": "C6.2", "hypothesis": "H-RIVAL"},
        {"doctrine": "D2", "mechanism": "M4", "property_test": "C6.2", "hypothesis": "H-SIG"},
        {"doctrine": "D5", "mechanism": "M4", "property_test": "none", "hypothesis": "H-BLIND"},
        {"doctrine": "D7", "mechanism": "M3", "property_test": "C6.1", "hypothesis": "H-SIG"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "The communion network beats a size-matched dual "
         "(sensory+relational) network on the shifted split.", "metric": "accuracy",
         "split": "shifted", "comparison": "model - dual", "direction": "greater",
         "mesi": 0.05, "seeds": 5},
        {"id": "H-NEC", "statement": "Knocking out golubets hurts the communion rule more than "
         "knocking out the width-matched cup gate.", "metric": "accuracy on R2",
         "split": "shifted", "comparison": "signature_knockout - matched_knockout",
         "direction": "less", "mesi": 0.10, "seeds": 5},
        {"id": "H-BLIND", "statement": "On naming the left guest's property the communion "
         "network falls below a face-reading MLP.", "condition": "naming the angels",
         "grounding": "angel identities unrecoverable from identical faces (Stoglav 1551 to "
         "modern scholarship)", "metric": "accuracy", "comparison": "model - mlp",
         "direction": "less", "mesi": 0.10, "seeds": 5},
        {"id": "H-RIVAL", "statement": "On the communion rule the model beats the Cusan "
         "comparison-only rival (chapter 0403).", "metric": "accuracy on R2", "split": "shifted",
         "comparison": "model - comparison_only", "direction": "greater", "mesi": 0.15, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.40, "margin_over_trivial": 0.15},
    "probe_predictions": [
        {"probe": "P2", "expected": "above baseline"}, {"probe": "P3", "expected": "above baseline"},
        {"probe": "P9", "expected": "above baseline"}, {"probe": "P1", "expected": "below baseline"},
        {"probe": "P8", "expected": "equal to baseline"}, {"probe": "P6", "expected": "above baseline"},
    ],
    "dialectic_links": [
        {"chapter": 403, "relation": "rival", "test": "H-RIVAL"},
        {"chapter": 194, "relation": "successor", "test": "none (image theology inherited; no test)"},
        {"chapter": 167, "relation": "successor", "test": "none (Areopagitica in Slavonic from 1371)"},
    ],
    "corpus_neighbors": [
        {"chapter": 403, "similarity": None, "difference": "Cusa knows only by comparison; here "
         "comparison provably cannot see the common nature, which gets its own channel."},
        {"chapter": 354, "similarity": None, "difference": "Giotto withholds the face and reads each "
         "body against its own equilibrium; here no individual frame exists at all."},
        {"chapter": 402, "similarity": None, "difference": "Masaccio fixes one station first; here "
         "relations are exactly indifferent to any shared station."},
        {"chapter": 395, "similarity": None, "difference": "van Eyck budgets light in layers; no "
         "layering or luminance mechanism is used here."},
        {"chapter": 398, "similarity": None, "difference": "Fra Angelico compiles per receiver; here "
         "the beholder enters the same two channels as the guests."},
    ],
    "barometer": {
        "cognitive_processing": ["relational transfer to unseen entities (shifted split)"],
        "embodied_cognition": [], "world_modeling": ["invariance to a shared station offset"],
        "consciousness": [], "language_understanding": [],
        "emotional_intelligence": [], "creativity": [],
        "autonomy": [],
    },
    "task_types": ["vector_classification"],
    "applications": [
        {"use": "abstract visual reasoning with constant-attribute rows", "sector": "AI evaluation",
         "dataset": "I-RAVEN (Hu et al. 2021)"},
        {"use": "drift-robust gas identification from sensor arrays (common-mode vs differential)",
         "sector": "environmental monitoring", "dataset": "UCI Gas Sensor Array Drift (Vergara et al. 2012)"},
        {"use": "same/different and shared-feature judgements on unseen characters",
         "sector": "document analysis", "dataset": "Omniglot (Lake et al. 2015)"},
    ],
    "safety_notes": ("Synthetic data only; no claim to replicate a person; no generated sentence "
                     "attributed to Rublev. Similarity values are null because earlier chapter files "
                     "were not available in this session."),
}
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

CHAPTER = 383
N_ENT = 4               # three guests + the candidate in the open fourth seat
D_LAT = 18              # latent: z(8) m(1) p(6) t(3)
D_OBS = 24
N_RULES = 3
N_PROPS = 6
TAU = 0.25              # temperature of the smooth max / min over guests
CLIP = 5.0
LR = 3e-3
BATCH = 128
STEPS_FULL, STEPS_QUICK = 1000, 300
BUDGET_FULL, BUDGET_QUICK = 180.0, 20.0
MUTANT = None           # set by --mutant or by the internal mutation run

# BEGIN STANDARD UTILITIES v1.0
def softplus(x):
    return np.logaddexp(0.0, x)


def sigmoid(x):
    return np.exp(-np.logaddexp(0.0, -x))


def logsumexp(x, axis=-1, keepdims=False):
    m = np.max(x, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


def softmax(x, axis=-1):
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))


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
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= sign * self.lr * mh / (np.sqrt(vh) + self.eps)


def clip_global_norm(grads, max_norm):
    norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    if norm > max_norm:
        for k in grads:
            grads[k] *= max_norm / (norm + 1e-12)
    return norm


def fd_check(loss_fn, params, grads, rng, n_entries=20, eps=1e-6, floor=2e-5):
    """Central differences on >= n_entries random entries + the largest-gradient entry.
    Round-off in (L+ - L-)/(2 eps) is ~1e-10 for eps=1e-6 and L~1, so the relative error's
    denominator is floored at 2e-5: gradients below that are compared to 2e-10 absolute.
    Returns (worst floored relative error, worst relative error on entries |g| > 1e-4)."""
    worst, resolved = {}, {}
    for k, p in params.items():
        flat = p.reshape(-1)
        idx = set(rng.choice(flat.size, size=min(n_entries, flat.size), replace=False).tolist())
        idx.add(int(np.argmax(np.abs(grads[k]).reshape(-1))))
        errs, errs_big = [], [0.0]
        for i in idx:
            old = flat[i]
            flat[i] = old + eps
            lp = loss_fn()
            flat[i] = old - eps
            lm = loss_fn()
            flat[i] = old
            num = (lp - lm) / (2 * eps)
            ana = grads[k].reshape(-1)[i]
            errs.append(abs(ana - num) / max(abs(ana) + abs(num), floor))
            if abs(ana) > 1e-4:
                errs_big.append(abs(ana - num) / (abs(ana) + abs(num)))
        worst[k], resolved[k] = max(errs), max(errs_big)
    return worst, resolved


def paired_bootstrap(diffs, rng, n_boot=2000):
    diffs = np.asarray(diffs, dtype=float)
    idx = rng.integers(0, diffs.size, size=(n_boot, diffs.size))
    means = diffs[idx].mean(axis=1)
    return float(diffs.mean()), [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def verdict(mean, ci, direction, mesi):
    lo, hi = ci
    if direction == "greater":
        if lo > 0 and mean >= mesi:
            return "supported"
        return "contradicted" if hi < 0 else "inconclusive"
    if hi < 0 and mean <= -mesi:
        return "supported"
    return "contradicted" if lo > 0 else "inconclusive"


def write_report(lines, report, json_path):
    print(f"=== VERIFIED REPORT · chapter {CHAPTER:04d} ===")
    for line in lines:
        print(line)
    print("=== END REPORT ===")
    if json_path:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
# END STANDARD UTILITIES


# ---------------------------------------------------------------- reverse-mode tape
class Node:
    """A value on the tape. par=None marks a constant (no gradient)."""
    __slots__ = ("v", "g", "par", "bw")

    def __init__(self, v, par=None, bw=None):
        self.v, self.g, self.par, self.bw = v, None, par, bw


def const(v):
    return Node(np.asarray(v, dtype=float))


def leaf(v):
    return Node(v, par=())


def _acc(n, g):
    if n.par is not None:
        n.g = g if n.g is None else n.g + g


def _unb(g, shape):
    while g.ndim > len(shape):
        g = g.sum(0)
    for ax, s in enumerate(shape):
        if s == 1 and g.shape[ax] != 1:
            g = g.sum(ax, keepdims=True)
    return g


def _node(v, parents, bw):
    live = tuple(p for p in parents if p.par is not None)
    return Node(v, par=live, bw=bw) if live else Node(v)


def add(a, b):
    return _node(a.v + b.v, (a, b), lambda g: (_acc(a, _unb(g, a.v.shape)), _acc(b, _unb(g, b.v.shape))))


def mul(a, b):
    return _node(a.v * b.v, (a, b),
                 lambda g: (_acc(a, _unb(g * b.v, a.v.shape)), _acc(b, _unb(g * a.v, b.v.shape))))


def scale(a, c):
    return _node(a.v * c, (a,), lambda g: _acc(a, g * c))


def lin(x, w, b=None):
    out = x.v @ w.v + (0.0 if b is None else b.v)

    def bw(g):
        g2 = g.reshape(-1, g.shape[-1])
        if x.par is not None:
            _acc(x, g @ w.v.T)
        _acc(w, x.v.reshape(-1, x.v.shape[-1]).T @ g2)
        if b is not None:
            _acc(b, g2.sum(0))
    return _node(out, (x, w) if b is None else (x, w, b), bw)


def unary(a, f, df):
    v = f(a.v)
    return _node(v, (a,), lambda g: _acc(a, g * df(a.v, v)))


def tanh(a):
    return unary(a, np.tanh, lambda x, v: 1.0 - v * v)


def splus(a):
    return unary(a, softplus, lambda x, v: sigmoid(x))


def logsig(a):
    return unary(a, lambda x: -softplus(-x), lambda x, v: sigmoid(-x))


def nexp(a):
    return unary(a, np.exp, lambda x, v: v)


def ssum(a, axis):
    return _node(a.v.sum(axis), (a,), lambda g: _acc(a, np.broadcast_to(np.expand_dims(g, axis), a.v.shape).copy()))


def lse(a, axis, tau=1.0):
    s = a.v / tau
    m = s.max(axis=axis, keepdims=True)
    e = np.exp(s - m)
    tot = e.sum(axis=axis, keepdims=True)
    v = tau * np.squeeze(m + np.log(tot), axis=axis)
    return _node(v, (a,), lambda g: _acc(a, np.expand_dims(g, axis) * (e / tot)))


def take(a, idx):
    """Gather along axis 1; indices are unique within one call, so plain assignment is exact."""
    def bw(g):
        ga = np.zeros_like(a.v)
        ga[:, idx] = g
        _acc(a, ga)
    return _node(a.v[:, idx], (a,), bw)


def cat(nodes, axis=-1):
    sizes = np.cumsum([n.v.shape[axis] for n in nodes])[:-1]

    def bw(g):
        for n, part in zip(nodes, np.split(g, sizes, axis=axis)):
            _acc(n, part)
    return _node(np.concatenate([n.v for n in nodes], axis=axis), tuple(nodes), bw)


def reshape(a, shape):
    return _node(a.v.reshape(shape), (a,), lambda g: _acc(a, g.reshape(a.v.shape)))


def backward(out):
    order, seen, stack = [], set(), [(out, False)]
    while stack:
        n, done = stack.pop()
        if done:
            order.append(n)
            continue
        if id(n) in seen or n.par is None:
            continue
        seen.add(id(n))
        stack.append((n, True))
        stack.extend((p, False) for p in n.par)
    out.g = np.ones_like(out.v)
    for n in reversed(order):
        if n.bw is not None and n.g is not None:
            n.bw(n.g)


# ---------------------------------------------------------------- data: the Mamre table
def make_world(rng):
    q, _ = np.linalg.qr(rng.normal(size=(D_OBS, D_OBS)))
    return {"Q": q[:, :D_LAT], "vocab_train": rng.normal(size=(48, 8)),
            "vocab_shift": rng.normal(size=(48, 8))}


def _entity(rng, vocab, mag_lo, mag_hi):
    return {"z": vocab[rng.integers(len(vocab))], "m": rng.uniform(mag_lo, mag_hi),
            "p": (rng.random(N_PROPS) < 0.45).astype(float)}


def _no_full_column(ents, rng, skip=-1):
    """Remove any property held by all four, except `skip`, by clearing it in one entity."""
    for k in range(N_PROPS):
        if k != skip and all(e["p"][k] == 1 for e in ents):
            ents[rng.integers(N_ENT)]["p"][k] = 0.0


def _scene(rule, rng, world, shift):
    vocab = world["vocab_shift" if shift else "vocab_train"]
    lo, hi = (1.0, 3.0) if shift else (-1.0, 1.0)
    picks = rng.choice(len(vocab), size=N_ENT, replace=False)
    ents = [_entity(rng, vocab, lo, hi) for _ in range(N_ENT)]
    for e, i in zip(ents, picks):
        e["z"] = vocab[i]
    y = int(rng.random() < 0.5)
    if rule == 0:                                   # likeness: candidate copies a guest
        if y:
            src = ents[rng.integers(N_ENT - 1)]
            ents[3] = {"z": src["z"], "m": src["m"], "p": src["p"].copy()}
    elif rule == 1:                                 # inclination: candidate above all guests
        top = max(e["m"] for e in ents[:3])
        ents[3]["m"] = top + rng.uniform(0.15, 0.6) * (1 if y else -1)
    elif rule == 2:                                 # communion: one property held by all four
        k = rng.integers(N_PROPS)
        if y or rng.random() < 0.5:
            _no_full_column(ents, rng, skip=k)
            for e in ents:
                e["p"][k] = float(y)               # matched negative: same scene, k removed
        else:
            _no_full_column(ents, rng)
    else:                                           # naming the angels: left guest's property 0
        y = int(ents[0]["p"][0])
    return ents, y


def _render(ents, rng, world, station_sd):
    t = rng.normal(0.0, station_sd, size=3)
    lat = np.array([np.concatenate([e["z"], [1.5 * e["m"]], 1.5 * e["p"], t]) for e in ents])
    return lat @ world["Q"].T + 0.05 * rng.normal(size=(N_ENT, D_OBS))


def make_split(rng, world, n, rules, shift=False):
    X = np.zeros((n, N_ENT, D_OBS))
    C = np.zeros((n, N_RULES))
    y = np.zeros(n)
    r = np.zeros(n, dtype=int)
    for i in range(n):
        rule = rules[i % len(rules)]
        ents, y[i] = _scene(rule, rng, world, shift)
        if rule != 3:                               # guest order is nuisance, except when naming
            perm = rng.permutation(3)
            ents = [ents[j] for j in perm] + [ents[3]]
        X[i] = _render(ents, rng, world, 1.5 if shift else 0.5)
        C[i, min(rule, N_RULES - 1) if rule != 3 else 0] = 1.0
        r[i] = rule
    return {"X": X, "C": C, "y": y, "rule": r}


def make_data(rng, n_train=18000, n_eval=1500):
    world = make_world(rng)
    main = [0, 1, 2]
    return {"world": world,
            "train": make_split(rng, world, n_train, main),
            "val": make_split(rng, world, 1200, main),
            "iid": make_split(rng, world, n_eval, main),
            "shift": make_split(rng, world, n_eval, main, shift=True),
            "name_train": make_split(rng, world, 3000, [3]),
            "name_test": make_split(rng, world, n_eval, [3])}


# ---------------------------------------------------------------- models
def _pairs(n):
    cand = n - 1
    groups = {"cand_to_guest": [(cand, j) for j in range(cand)],
              "guest_to_cand": [(i, cand) for i in range(cand)],
              "guest_guest": [(i, j) for i in range(cand) for j in range(cand) if i != j]}
    flat = [p for g in groups.values() for p in g]
    idx = {}
    start = 0
    for name, g in groups.items():
        idx[name] = list(range(start, start + len(g)))
        start += len(g)
    return np.array([a for a, _ in flat]), np.array([b for _, b in flat]), idx


def _init(rng, shape, gain=1.0):
    return rng.normal(0.0, gain / math.sqrt(shape[0]), size=shape)


ARCH_WIDTHS = {"communion": 32, "dual": 29, "comparison_only": 38, "mlp": 37}


def build_model(in_dim, out_dim, task_type, rng, kind="communion", n_entities=N_ENT,
                cue_dim=N_RULES, h=24, k=8, m=8, hidden=None):
    """kind: communion (Rublev) | dual (sensory+relational) | comparison_only (Cusa 0403) | mlp."""
    assert task_type in TASK_TYPES
    d = int(math.ceil(in_dim / n_entities))
    cue = max(cue_dim, 1)
    hr = hidden or ARCH_WIDTHS[kind]
    out = 1 if out_dim <= 2 else out_dim
    P = {}
    if kind == "mlp":
        P["Wm"], P["bm"] = _init(rng, (n_entities * d + cue, hr)), np.zeros(hr)
        P["wo"], P["bo"] = _init(rng, (hr, out)), np.zeros(out)
    else:
        P["W1"], P["b1"] = _init(rng, (d, h)), 0.1 * rng.normal(size=h)
        P["V"] = _init(rng, (h, k))
        P["Wg"], P["bg"] = _init(rng, (cue, k)), np.full(k, 0.5)
        width = 9 * k + cue
        if kind == "communion":
            P["U"], P["bu"] = _init(rng, (d, m)), np.zeros(m)
            width += m
        if kind == "dual":
            P["Ws"], P["bs"] = _init(rng, (d, m)), np.zeros(m)
            width += 2 * m
        P["Wr1"], P["br1"] = _init(rng, (width, hr)), np.zeros(hr)
        P["wo"], P["bo"] = _init(rng, (hr, out)), np.zeros(out)
    I, J, groups = _pairs(n_entities)
    return {"kind": kind, "params": P, "n": n_entities, "d": d, "cue": cue, "out": out,
            "I": I, "J": J, "groups": groups, "ko": {}, "variant": None}


def _prep(model, X, C):
    B = X.shape[0]
    X = X.reshape(B, -1)
    pad = model["n"] * model["d"] - X.shape[1]
    if pad > 0:
        X = np.concatenate([X, np.zeros((B, pad))], axis=1)
    X = X.reshape(B, model["n"], model["d"])
    if C is None or C.shape[1] == 0:
        C = np.ones((B, 1))
    return X, C


def _pool(R, groups, ordered=False):
    """Mean, smooth max and smooth min of each pair group: symmetric in the guests."""
    feats = []
    for name in ("cand_to_guest", "guest_to_cand", "guest_guest"):
        Rg = take(R, groups[name])
        ng = len(groups[name])
        if ordered:                                  # negative control: seat-dependent weights
            w = np.arange(1, ng + 1, dtype=float).reshape(1, ng, 1)
            feats.append(ssum(mul(Rg, const(w / w.sum())), 1))
        else:
            feats.append(scale(ssum(Rg, 1), 1.0 / ng))
        feats.append(add(lse(Rg, 1, TAU), const(-TAU * math.log(ng))))
        feats.append(scale(add(lse(scale(Rg, -1.0), 1, TAU), const(-TAU * math.log(ng))), -1.0))
    return cat(feats, axis=1)


def relation_channel(model, P, X, C):
    """M1 inclination: one shared encoder applied to every ordered pair DIFFERENCE."""
    if model["variant"] == "face_relations":         # negative control: products of faces
        D = const(X[:, model["I"]] * X[:, model["J"]])
    else:
        D = const(X[:, model["I"]] - X[:, model["J"]])
    R = lin(tanh(lin(D, P["W1"], P["b1"])), P["V"])
    if "cup_gate" in model["ko"]:
        G = const(np.broadcast_to(model["ko"]["cup_gate"], (X.shape[0], R.v.shape[-1])))
    else:
        G = splus(lin(const(C), P["Wg"], P["bg"]))
    R = mul(R, reshape(G, (X.shape[0], 1, -1)))
    F = _pool(R, model["groups"], ordered=model["variant"] == "ordered")
    if "inclination" in model["ko"]:
        F = const(np.broadcast_to(model["ko"]["inclination"], F.v.shape))
    return F


def golubets(model, P, X):
    """M2: soft AND over all entities, nu_k = prod_i sigmoid(u_k . x_i + b_k)."""
    Z = lin(const(X), P["U"], P["bu"])
    if model["variant"] == "mean_pool":              # negative control: average, not intersection
        return scale(ssum(nexp(logsig(Z)), 1), 1.0 / X.shape[1])
    nu = nexp(ssum(logsig(Z), 1))
    if "golubets" in model["ko"]:
        nu = const(np.broadcast_to(model["ko"]["golubets"], nu.v.shape))
    return nu


def forward(model, X, C):
    X, C = _prep(model, X, C)
    P = {k: leaf(v) for k, v in model["params"].items()}
    B = X.shape[0]
    hid = {}
    if model["kind"] == "mlp":
        F = const(np.concatenate([X.reshape(B, -1), C], axis=1))
        H = tanh(lin(F, P["Wm"], P["bm"]))
    else:
        parts = [relation_channel(model, P, X, C)]
        hid["relations"] = parts[0].v
        if model["kind"] == "communion":
            nu = golubets(model, P, X)
            hid["nature"] = nu.v
            parts.append(nu)
        if model["kind"] == "dual":                  # sensory: per-entity features (faces)
            S = tanh(lin(const(X), P["Ws"], P["bs"]))
            parts += [scale(ssum(take(S, list(range(model["n"] - 1))), 1), 1.0 / (model["n"] - 1)),
                      reshape(take(S, [model["n"] - 1]), (B, -1))]
        parts.append(const(C))
        H = tanh(lin(cat(parts, axis=1), P["Wr1"], P["br1"]))
    hid["readout"] = H.v
    z = lin(H, P["wo"], P["bo"])
    return z, P, hid


def _loss_node(model, z, y):
    B = z.v.shape[0]
    if model["out"] == 1:
        zz = reshape(z, (B,))
        return scale(ssum(add(splus(zz), mul(zz, const(-y))), 0), 1.0 / B)
    onehot = np.eye(model["out"])[y.astype(int)]
    return scale(ssum(add(lse(z, 1), scale(ssum(mul(z, const(onehot)), 1), -1.0)), 0), 1.0 / B)


def loss_value(model, batch):
    z, _, _ = forward(model, batch["X"], batch.get("C"))
    return float(_loss_node(model, z, batch["y"]).v)


def loss_and_grads(model, batch):
    z, P, _ = forward(model, batch["X"], batch.get("C"))
    L = _loss_node(model, z, batch["y"])
    backward(L)
    grads = {k: (n.g if n.g is not None else np.zeros_like(n.v)) for k, n in P.items()}
    if MUTANT and MUTANT.startswith("zero_grad_"):
        key = MUTANT[len("zero_grad_"):]
        if key in grads:
            grads[key] = np.zeros_like(grads[key])
    return float(L.v), grads


def predict(model, X, C=None):
    z, _, _ = forward(model, X, C)
    return sigmoid(z.v[:, 0]) if model["out"] == 1 else softmax(z.v, axis=1)


def hidden_states(model, X, C=None):
    return forward(model, X, C)[2]


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


def modules(model):
    if model["kind"] != "communion":
        return {}
    return {
        "inclination": {"params": ["W1", "b1", "V"], "signature": True,
                        "role": "shared MLP on ordered pair differences, symmetric pooled statistics"},
        "golubets": {"params": ["U", "bu"], "signature": True,
                     "role": "product-of-sigmoids set-intersection pooling (soft AND over entities)"},
        "cup_gate": {"params": ["Wg", "bg"], "signature": False,
                     "role": "context-conditioned positive gating of relation heads"},
        "readout": {"params": ["Wr1", "br1", "wo", "bo"], "signature": False,
                    "role": "MLP classifier over pooled statistics"},
    }


def knockout(model, name, mode="mean", ref=None):
    """Copy of the model with module `name` replaced by its mean, zero or identity output."""
    ko = dict(model, ko=dict(model["ko"]), params={k: v.copy() for k, v in model["params"].items()})
    X, C = _prep(model, ref["X"], ref["C"]) if ref is not None else (None, None)
    if name == "golubets":
        val = golubets(model, {k: leaf(v) for k, v in model["params"].items()}, X).v.mean(0) \
            if mode == "mean" else np.full(model["params"]["U"].shape[1], 1.0 if mode == "identity" else 0.0)
    elif name == "cup_gate":
        G = softplus(C @ model["params"]["Wg"] + model["params"]["bg"]) if X is not None else None
        val = G.mean(0) if mode == "mean" else np.full(model["params"]["V"].shape[1], 1.0 if mode == "identity" else 0.0)
    elif name == "inclination":
        val = hidden_states(model, ref["X"], ref["C"])["relations"].mean(0) if mode == "mean" else 0.0
    else:
        raise ValueError(f"no knockout for module {name}")
    ko["ko"][name] = val
    return ko


TASK_TYPES = ["vector_classification"]
MUTANTS = {"sign_flip": "optimizer ascends the loss", "zero_lr": "learning rate set to zero",
           "zero_grad_W1": "inclination encoder gradient zeroed",
           "zero_grad_U": "golubets gradient zeroed", "zero_grad_Wr1": "readout gradient zeroed"}


# ---------------------------------------------------------------- training and evaluation
def fit(model, data, budget, rng, lr=LR):
    P = model["params"]
    opt = Adam(P, lr=0.0 if MUTANT == "zero_lr" else lr)
    sign = -1.0 if MUTANT == "sign_flip" else 1.0
    probe = {k: data[k][:512] for k in ("X", "C", "y")}
    loss0 = loss_value(model, probe)
    n = len(data["y"])
    for _ in range(budget):
        idx = rng.integers(0, n, size=BATCH)
        L, g = loss_and_grads(model, {"X": data["X"][idx], "C": data["C"][idx], "y": data["y"][idx]})
        if not np.isfinite(L):
            raise FloatingPointError("non-finite loss")
        clip_global_norm(g, CLIP)
        opt.step(P, g, sign)
    return {"loss0": loss0, "loss_end": loss_value(model, probe)}


def accuracy(model, split, rule=None):
    mask = np.ones(len(split["y"]), bool) if rule is None else split["rule"] == rule
    p = predict(model, split["X"][mask], split["C"][mask])
    return float(np.mean((p > 0.5) == (split["y"][mask] > 0.5)))


def new_model(kind, rng):
    return build_model(N_ENT * D_OBS, 2, "vector_classification", rng, kind=kind)


def gradcheck(model, batch, rng):
    _, g = loss_and_grads(model, batch)
    return fd_check(lambda: loss_value(model, batch), model["params"], g, rng)[0]


def gradcheck_full(model, batch, rng):
    _, g = loss_and_grads(model, batch)
    return fd_check(lambda: loss_value(model, batch), model["params"], g, rng)


def child_rngs(base, seed, n):
    return [np.random.default_rng(s) for s in np.random.SeedSequence([base, seed]).spawn(n)]


# ---------------------------------------------------------------- correctness tests
def test_gradients(data, rngs, steps=60):
    """C1: every tensor of every architecture, at init and after `steps` updates."""
    batch = {k: data["train"][k][:24] for k in ("X", "C", "y")}
    worst, worst_big, total, checked = 0.0, 0.0, 0, 0
    for kind in ("communion", "dual", "comparison_only", "mlp"):
        model = new_model(kind, rngs[0])
        for stage in ("init", "trained"):
            if stage == "trained":
                fit(model, data["train"], steps, rngs[1])
            errs, big = gradcheck_full(model, batch, rngs[2])
            total += len(model["params"])
            checked += sum(1 for e in errs.values() if np.isfinite(e))
            worst = max(worst, max(errs.values()))
            worst_big = max(worst_big, max(big.values()))
    return {"tensors_checked": checked, "tensors_total": total, "max_rel_error": worst,
            "max_rel_error_resolved": worst_big,
            "checked_at": ["init", f"after_{steps}_steps"], "passed": bool(worst <= 1e-5)}


def _rand_model(rng, variant=None):
    m = new_model("communion", rng)
    for k in m["params"]:
        m["params"][k] = m["params"][k] * 3.0 + 0.3 * rng.normal(size=m["params"][k].shape)
    m["variant"] = variant
    return m


def prop_permutation(rng, variant=None, draws=10):
    """C6.1 the three guests are interchangeable; only the open seat is distinguished."""
    worst = 0.0
    for _ in range(draws):
        m = _rand_model(rng, variant)
        X = rng.normal(size=(32, N_ENT, D_OBS)) * rng.uniform(0.5, 3.0)
        C = np.eye(N_RULES)[rng.integers(N_RULES, size=32)]
        base = forward(m, X, C)[0].v
        for perm in ([1, 0, 2], [2, 1, 0], [0, 2, 1], [1, 2, 0], [2, 0, 1]):
            worst = max(worst, float(np.abs(forward(m, X[:, perm + [3]], C)[0].v - base).max()))
    return worst


def prop_common_shift(rng, variant=None, draws=10):
    """C6.2 relations cannot see anything every entity shares (a common shift)."""
    worst = 0.0
    for _ in range(draws):
        m = _rand_model(rng, variant)
        P = {k: leaf(v) for k, v in m["params"].items()}
        X = rng.normal(size=(32, N_ENT, D_OBS))
        C = np.eye(N_RULES)[rng.integers(N_RULES, size=32)]
        t = rng.normal(size=(32, 1, D_OBS)) * rng.uniform(1.0, 10.0)
        a = relation_channel(m, P, X, C).v
        b = relation_channel(m, P, X + t, C).v
        worst = max(worst, float(np.abs(a - b).max()))
    return worst


def prop_intersection(rng, variant=None, draws=10):
    """C6.3 the common nature never exceeds what the poorest member holds, and adding a
    member can only lower it."""
    worst = -np.inf
    for _ in range(draws):
        m = _rand_model(rng, variant)
        P = {k: leaf(v) for k, v in m["params"].items()}
        X = rng.normal(size=(32, N_ENT, D_OBS)) * rng.uniform(0.2, 2.0)
        Xe = np.concatenate([X, rng.normal(size=(32, 1, D_OBS))], axis=1)
        nu = golubets(m, P, X).v
        sig = sigmoid(X @ m["params"]["U"] + m["params"]["bu"]).min(axis=1)
        nu5 = golubets(m, P, Xe).v
        worst = max(worst, float((nu - sig).max()), float((nu5 - nu).max()))
    return worst


def test_properties(rng):
    out = []
    for cid, fn, bad, tol in (("C6.1", prop_permutation, "ordered", 1e-9),
                              ("C6.2", prop_common_shift, "face_relations", 1e-9),
                              ("C6.3", prop_intersection, "mean_pool", 1e-12)):
        held, broken = fn(rng), fn(rng, bad)
        ok = held <= tol and broken > 1e-4
        out.append({"id": cid, "name": fn.__name__, "passed": bool(ok),
                    "detail": f"violation {held:.2e} (tol {tol:.0e}); negative control '{bad}' {broken:.2e}"})
    return out


def scene_hashes(split):
    return {hashlib.sha1(np.round(x, 8).tobytes()).hexdigest() for x in split["X"]}


def test_split_integrity(data):
    tr = scene_hashes(data["train"]) | scene_hashes(data["name_train"])
    clash = sum(len(tr & scene_hashes(data[s])) for s in ("val", "iid", "shift", "name_test"))
    vt, vs = data["world"]["vocab_train"], data["world"]["vocab_shift"]
    shared = int(np.sum(np.isclose(vt[:, None, :], vs[None, :, :]).all(-1)))
    return {"id": "C7", "name": "split_integrity", "passed": clash == 0 and shared == 0,
            "detail": f"{clash} shared scenes; {shared} shared identity codes in the shifted split"}


def short_protocol(data, seed, steps):
    """C1 at init plus C3 after a short run: used for mutants and their clean control."""
    rngs = child_rngs(7919, seed, 3)
    model = new_model("communion", rngs[0])
    batch = {k: data["train"][k][:24] for k in ("X", "C", "y")}
    c1 = max(gradcheck(model, batch, rngs[2]).values()) <= 1e-5
    info = fit(model, data["train"], steps, rngs[1])
    c3 = (info["loss_end"] <= (1 - MIND_CARD["thresholds"]["loss_drop_fraction"]) * info["loss0"]
          and accuracy(model, data["iid"]) >= 0.5 + MIND_CARD["thresholds"]["margin_over_trivial"])
    return bool(c1), bool(c3)


def test_mutants(data, seed, steps):
    global MUTANT
    clean = short_protocol(data, seed, steps)
    detected = 0
    for name in MUTANTS:
        MUTANT = name
        try:
            c1, c3 = short_protocol(data, seed, steps)
        except FloatingPointError:
            c1, c3 = True, False
        MUTANT = None
        detected += int(not (c1 and c3))
    ok = all(clean) and detected == len(MUTANTS)
    return {"detected": detected, "total": len(MUTANTS), "score": detected / len(MUTANTS)}, \
        {"id": "C5", "name": "mutant_detection", "passed": bool(ok),
         "detail": f"{detected}/{len(MUTANTS)} detected; clean control passes C1,C3: {clean}"}


def test_determinism(data, seed):
    outs = []
    for _ in range(2):
        rngs = child_rngs(31, seed, 2)
        m = new_model("communion", rngs[0])
        info = fit(m, data["train"], 40, rngs[1])
        outs.append((info["loss_end"], predict(m, data["iid"]["X"][:200], data["iid"]["C"][:200]), m))
    same = outs[0][0] == outs[1][0] and np.array_equal(outs[0][1], outs[1][1])
    finite = all(np.all(np.isfinite(v)) for v in outs[0][2]["params"].values()) and \
        all(np.all(np.isfinite(h)) for h in hidden_states(outs[0][2], data["iid"]["X"][:200],
                                                         data["iid"]["C"][:200]).values())
    return {"id": "C2", "name": "determinism_and_finiteness", "passed": bool(same and finite),
            "detail": f"identical reruns: {same}; finite: {finite}"}


# ---------------------------------------------------------------- experiment
KINDS = ("communion", "dual", "comparison_only", "mlp")


def run_seed(base, seed, steps, n_train):
    rngs = child_rngs(base, seed, 10)
    data = make_data(rngs[0], n_train=n_train)
    res, models = {}, {}
    for i, kind in enumerate(KINDS):
        m = new_model(kind, rngs[1 + i])
        info = fit(m, data["train"], steps, rngs[5 + i])
        models[kind] = m
        res[kind] = {"loss": info, "iid": accuracy(m, data["iid"]),
                     "shift": accuracy(m, data["shift"]),
                     "rules_shift": [accuracy(m, data["shift"], r) for r in range(N_RULES)]}
    full = models["communion"]
    ref = {k: data["train"][k][:256] for k in ("X", "C")}
    res["knockouts"] = {}
    for name in ("inclination", "golubets", "cup_gate"):
        ko = knockout(full, name, "mean", ref)
        res["knockouts"][name] = {"shift": accuracy(ko, data["shift"]),
                                  "r2": accuracy(ko, data["shift"], 2)}
    for kind in ("communion", "mlp"):                   # naming the angels: blind spot
        m = new_model(kind, rngs[9])
        fit(m, data["name_train"], steps, rngs[9])
        res[f"name_{kind}"] = accuracy(m, data["name_test"])
    return res, data, models


def evaluate_hypotheses(per_seed, rng):
    spec = {h["id"]: h for h in MIND_CARD["hypotheses"]}
    diffs = {
        "H-SIG": [r["communion"]["shift"] - r["dual"]["shift"] for r in per_seed],
        "H-NEC": [r["knockouts"]["golubets"]["r2"] - r["knockouts"]["cup_gate"]["r2"] for r in per_seed],
        "H-BLIND": [r["name_communion"] - r["name_mlp"] for r in per_seed],
        "H-RIVAL": [r["communion"]["rules_shift"][2] - r["comparison_only"]["rules_shift"][2]
                    for r in per_seed],
    }
    out = []
    for hid, d in diffs.items():
        mean, ci = paired_bootstrap(d, rng)
        out.append({"id": hid, "metric": spec[hid]["metric"], "mean_diff": mean, "ci95": ci,
                    "mesi": spec[hid]["mesi"], "n_seeds": len(d),
                    "verdict": verdict(mean, ci, spec[hid]["direction"], spec[hid]["mesi"])})
    return out


def knockout_table(per_seed, rng, model):
    rows = []
    registry = modules(model)
    for name in ("inclination", "golubets", "cup_gate"):
        sig = registry[name]["signature"]
        d = [r["knockouts"][name]["shift"] - r["communion"]["shift"] for r in per_seed]
        mean, ci = paired_bootstrap(d, rng)
        rows.append({"module": name, "signature": sig, "metric_change": mean, "ci95": ci})
    return rows


# ---------------------------------------------------------------- report
def format_report(rep, per_seed):
    L = [f"environment: Python {rep['environment']['python']} · NumPy {rep['environment']['numpy']}",
         f"seeds: {rep['seeds']}   runtime_s: {rep['runtime_s']:.1f}",
         "n_params: " + ", ".join(f"{k}={v}" for k, v in rep["n_params_all"].items()),
         "gradcheck: {tensors_checked}/{tensors_total} tensors, max rel err {max_rel_error:.2e}, "
         "at {checked_at}, passed={passed}".format(**rep["gradcheck"]), "correctness:"]
    L += [f"  {c['id']:<5} {c['name']:<28} {'PASS' if c['passed'] else 'FAIL'}  {c['detail']}"
          for c in rep["correctness"]]
    m = rep["mutants"]
    L.append(f"mutation score: {m['detected']}/{m['total']} = {m['score']:.2f}")
    L.append("accuracy (mean over seeds)        iid    shift   R0-like R1-incl R2-comm")
    for kind in KINDS:
        iid = np.mean([r[kind]["iid"] for r in per_seed])
        sh = np.mean([r[kind]["shift"] for r in per_seed])
        rr = np.mean([r[kind]["rules_shift"] for r in per_seed], axis=0)
        L.append(f"  {kind:<30} {iid:.3f}  {sh:.3f}   {rr[0]:.3f}   {rr[1]:.3f}   {rr[2]:.3f}")
    L.append("  naming the angels (blind spot): communion {:.3f}  mlp {:.3f}".format(
        np.mean([r["name_communion"] for r in per_seed]), np.mean([r["name_mlp"] for r in per_seed])))
    L.append("hypotheses:")
    for h in rep["hypotheses"]:
        L.append(f"  {h['id']:<8} diff {h['mean_diff']:+.3f}  ci95 [{h['ci95'][0]:+.3f}, "
                 f"{h['ci95'][1]:+.3f}]  mesi {h['mesi']:.2f}  n={h['n_seeds']}  -> {h['verdict']}")
    L.append("knockouts (communion, shifted accuracy change, mean replacement):")
    for k in rep["knockouts"]:
        L.append(f"  {k['module']:<12} signature={str(k['signature']):<5} {k['metric_change']:+.3f} "
                 f"ci95 [{k['ci95'][0]:+.3f}, {k['ci95'][1]:+.3f}]")
    L.append(f"task types: {rep['task_types']}   exit code: {rep['exit_code']}")
    return L


# ---------------------------------------------------------------- real-data bridge (optional)
def data_bridge(path, seed):
    """CSV: entity features flattened (4 x d), then an integer cue column, then a 0/1 label."""
    if not path or not os.path.exists(path):
        print(f"[data bridge] no local CSV at {path!r}; step skipped (no downloads by design).")
        return
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    X, cue, y = raw[:, :-2], raw[:, -2].astype(int), raw[:, -1]
    C = np.eye(max(cue.max() + 1, 1))[cue]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    cut = int(0.7 * len(y))
    tr, te = order[:cut], order[cut:]
    for kind in ("communion", "mlp"):
        m = build_model(X.shape[1], 2, "vector_classification", rng, kind=kind, cue_dim=C.shape[1])
        fit(m, {"X": X[tr], "C": C[tr], "y": y[tr]}, STEPS_QUICK, rng)
        acc = np.mean((predict(m, X[te], C[te]) > 0.5) == (y[te] > 0.5))
        print(f"[data bridge] {kind}: held-out accuracy {acc:.3f}")


# ---------------------------------------------------------------- entry point
def correctness_suite(data, first, seeds, steps, quick):
    """C1-C7 on the first seed; returns the correctness list, gradcheck and mutant summary."""
    rngs = child_rngs(2027, seeds[0], 4)
    grad = test_gradients(data, rngs)
    correct = [{"id": "C1", "name": "gradient_check", "passed": grad["passed"],
                "detail": f"max rel err {grad['max_rel_error']:.2e} (floor 2e-5); "
                          f"on |g|>1e-4 entries {grad['max_rel_error_resolved']:.2e}"}]
    info = first["communion"]["loss"]
    drop = 1 - info["loss_end"] / info["loss0"]
    c3 = drop >= MIND_CARD["thresholds"]["loss_drop_fraction"] and \
        first["communion"]["iid"] >= 0.5 + MIND_CARD["thresholds"]["margin_over_trivial"]
    correct.append({"id": "C3", "name": "learning", "passed": bool(c3),
                    "detail": f"loss drop {drop:.2f}; iid acc {first['communion']['iid']:.3f} vs trivial 0.5"})
    mut = {"detected": 0, "total": len(MUTANTS), "score": 0.0}
    if MUTANT is not None:
        return correct, grad, mut
    correct.append(test_determinism(data, seeds[0]))
    shuf = new_model("communion", rngs[3])
    fit(shuf, dict(data["train"], y=rngs[3].permutation(data["train"]["y"])), steps, rngs[3])
    majority = max(np.mean(data["iid"]["y"]), 1 - np.mean(data["iid"]["y"]))
    acc_shuf = accuracy(shuf, data["iid"])
    correct.append({"id": "C4", "name": "shuffled_label_control",
                    "passed": bool(acc_shuf <= majority + 0.06),
                    "detail": f"iid acc {acc_shuf:.3f}; trivial {majority:.3f} (band +0.06)"})
    mut, c5 = test_mutants(data, seeds[0], 200 if quick else 400)
    correct.append(c5)
    correct += test_properties(rngs[2])
    correct.append(test_split_integrity(data))
    return correct, grad, mut


def main(argv=None):
    global MUTANT
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--json", default=None)
    ap.add_argument("--card", action="store_true")
    ap.add_argument("--mutant", default=None, choices=sorted(MUTANTS))
    ap.add_argument("--data", default=None)
    args = ap.parse_args(argv)
    if args.card:
        print(json.dumps(MIND_CARD, indent=2))
        return 0
    t0 = time.time()
    budget = BUDGET_QUICK if args.quick else BUDGET_FULL
    steps = STEPS_QUICK if args.quick else STEPS_FULL
    n_train = 6000 if args.quick else 18000
    seeds = [args.seed + i for i in range(1 if args.quick else max(args.seeds, 1))]
    print(f"[seeds] base={args.seed} seeds={seeds}")
    MUTANT = args.mutant
    first, data, models = run_seed(1009, seeds[0], steps, n_train)
    correct, grad, mut = correctness_suite(data, first, seeds, steps, args.quick)
    per_seed = [first] + [run_seed(1009, s, steps, n_train)[0] for s in seeds[1:]]
    stat_rng = np.random.default_rng(np.random.SeedSequence([4242, seeds[0]]))
    hyps = evaluate_hypotheses(per_seed, stat_rng)
    if len(per_seed) < 5:
        hyps = [dict(h, verdict="not evaluated") for h in hyps]
    runtime = time.time() - t0
    correct.append({"id": "C8", "name": "budget", "passed": runtime <= budget,
                    "detail": f"{runtime:.1f}s of {budget:.0f}s"})
    code = 3 if runtime > budget else (0 if all(c["passed"] for c in correct) else 1)
    rep = {"schema_version": "1.0", "chapter": CHAPTER, "file": os.path.basename(__file__),
           "card_revision": MIND_CARD["card_revision"],
           "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
           "seeds": seeds, "runtime_s": runtime, "n_params": n_params(models["communion"]),
           "n_params_all": {k: n_params(v) for k, v in models.items()},
           "gradcheck": grad, "correctness": correct, "mutants": mut, "hypotheses": hyps,
           "knockouts": knockout_table(per_seed, stat_rng, models["communion"]),
           "task_types": TASK_TYPES, "exit_code": code}
    write_report(format_report(rep, per_seed), rep, args.json)
    data_bridge(args.data, seeds[0])
    return code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except FloatingPointError as exc:
        print(f"non-finite values: {exc}")
        sys.exit(4)

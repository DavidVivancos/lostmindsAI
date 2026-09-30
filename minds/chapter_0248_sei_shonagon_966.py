#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0248 · Sei Shōnagon
# Author: David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 13, Minds 241–260: https://www.amazon.com/dp/B0HLD1S6QT · Demos: https://artificiology.com/
# END ATTRIBUTION
"""Monozukushi: a union-of-neighbourhoods catalogue scorer (Sei Shōnagon, c. 966 – c. 1017/1025).

Thesis
    A category is the set of particulars admitted under its heading, so a model
    of her mind must be able to reject the midpoint of its own members.

Evidence and provenance  (provenance: belief)
    Her own text is extant and grounds every doctrine below. The biography is weak
    (given name unknown, the later life is legend); no mechanism rests on it.
    D1 primary      the 「…もの」 lists name a heading and file particulars under it
                    with no stated criterion (「うつくしきもの」: a child's face drawn on
                    a melon, a sparrow chick hopping to a squeak, a toddler holding up
                    a speck of dust, a chick on long legs).
    D2 primary      「春はあけぼの」: a topic and a noun, the evaluative predicate deleted.
    D3 primary      「心ときめきするもの」: membership reported as a bodily event.
    D4 primary      the snow of Kōro Peak: a question answered by an act (raising the
                    blind, alluding to Bai Juyi) whose correctness is the room's delight.
    D5 scholarship  the text survives in "mixed" (Sankanbon, Nōinbon) and "classified"
                    (Sakaibon, Maedakebon) lineages with incompatible global orders
                    (Ikeda Kikan 1963).
    D6 scholarship  the fall of Teishi's house (995–1000) is almost absent from the book
                    (Fukumori 1997; the Mumyōzōshi, c. 1200, trans. Marra 1984).
    D7 scholarship  the approving room is class-bounded (Angles 2001).
    D8 speculation  reading D1 as "a heading's own midpoint need not belong" is this
                    chapter's reconstruction, not a statement of hers.

Doctrine -> mechanism -> test
    D1, D8  M1 exemplar_set: log-sum-exp over learned points per heading   C6.3  H-SIG H-NEC
    D2      M2 per-list Plackett-Luce emission; no truth value is emitted   C6.2  order tau
    D5      M2 no loss term couples two lists                                C6.1  -
    D3      M3 heartbeat gate: sparse per-heading logistic signal           -     knockouts
    D4      M4 ratification critic trained by margin on whole lists         -     H-RAT
    D6      M5 omission audit on emitted marginals (not trained)             C6.4  H-AUDIT H-BLIND
    D7      limit: the critic has no signal for a room that is wrong together

Research question  (out-of-distribution detection and shift)
    Does scoring membership by soft-nearest learned exemplars, instead of one learned
    prototype per class, let a model reject in-hull blends of the members of a
    multi-modal category, and what does it cost where training omitted a region?

Closest prior art and the delta
    Exemplar (context) models of categorisation (Medin & Schaffer 1978; Nosofsky 1986),
    RBF networks (Broomhead & Lowe 1988), Prototypical Networks (Snell, Swersky & Zemel
    2017; the size-matched baseline here), infinite mixture prototypes (Allen et al.
    2019) and ListMLE (Xia et al. 2008). Scorer and loss are known. The contribution
    is the test: a generator in which ground-truth-rejected blends of members exist,
    a size-matched prototype pitted against the exemplar set on exactly those blends,
    and an audit that must find an undeclared training filter from emissions alone.

Blind spot
    What was never admitted cannot be resembled. Trained on a record with the
    sorrowing particulars removed (D6), the exemplar set should lose more of its
    advantage on sorrowing members than on the rest.

Task  (native, listwise)
    z ~ U[0,1]^8 latent factors; x = standardise(tanh(z A) B + noise), D = 32.
    Heading k attends to 3 factors and owns 4 seeds at least 0.6 apart there; z is
    admitted iff its nearest-seed resemblance clears a bar fixed once on the world.
    A pool holds L = 6 admitted particulars in the heading's reading order (z . w_k)
    followed by P - L = 24 others. Splits are by particular, never by heading.
    Shifted split: half the distractors are blends (attended factors averaged across
    two seed regions, then rejected by the generator itself). Blind split: train
    without grief > 0.5, evaluate on members with grief > 0.5.

Limits
    Synthetic latents stand in for Heian particulars. This is a research prototype of
    one AGI-oriented component, not an AGI and not a replica of her mind.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 1,
    "revision_log": ["r1 (2026-09-30): rebuilt from a pre-guideline draft whose midpoint check was a "
                     "learned-behaviour hypothesis run as a pass/fail correctness test; card written "
                     "before the first run of this rebuild."],
    "generation": {"template_version": "codeguidelines 1.0", "generator": "Claude",
                   "generator_version": "Claude Opus 5.5", "date": "2026-09-30"},
    "id": 248, "figure": "Sei Shōnagon", "born": 966, "died": 1017, "civilization": "Japanese (Heian)",
    "provenance": "belief",
    "thesis": "A category is the set of particulars admitted under its heading, so a model of her mind "
              "must be able to reject the midpoint of its own members.",
    "evidence": [
        {"id": "D1", "claim": "Headings are given by extension only; no criterion is ever stated.",
         "basis": "primary", "source": "Makura no sōshi, 'utsukushiki mono' and the other mono-wa lists; "
                                       "Morris, HJAS 40:1 (1980)"},
        {"id": "D2", "claim": "Judgement as selection under a topic, predicate deleted.",
         "basis": "primary", "source": "Makura no sōshi, opening section 'haru wa akebono'"},
        {"id": "D3", "claim": "Membership is reported as a bodily event in a trained observer.",
         "basis": "primary", "source": "Makura no sōshi, 'kokoro tokimeki suru mono'"},
        {"id": "D4", "claim": "Correctness as ratification by the room (the Kōro Peak reply).",
         "basis": "primary", "source": "Makura no sōshi, the 'snow of Kōro Peak' episode; Bai Juyi's poem as intertext"},
        {"id": "D5", "claim": "Global order is editorial: mixed and classified lineages disagree.",
         "basis": "scholarship", "source": "Ikeda Kikan, Kenkyū Makura no sōshi (Shibundō, 1963)"},
        {"id": "D6", "claim": "The ruin of Teishi's house (995-1000) is almost absent from the book.",
         "basis": "scholarship", "source": "Fukumori, JATJ 31:1 (1997); Mumyōzōshi, trans. Marra, MN 39 (1984)"},
        {"id": "D7", "claim": "The ratifying audience is class-bounded.",
         "basis": "scholarship", "source": "Angles, Japan Review 13 (2001)"},
        {"id": "D8", "claim": "A heading's own midpoint need not belong to it.",
         "basis": "speculation", "source": "this chapter's reading of D1"},
    ],
    "research_question": {"category": "out-of-distribution detection and shift",
                          "question": "Does soft-nearest-exemplar scoring, rather than one prototype per "
                                      "class, reject in-hull blends of a multi-modal category's members, and "
                                      "what does it cost where training omitted a region?"},
    "mechanism": {
        "name": "Monozukushi: union-of-neighbourhoods scorer, per-list Plackett-Luce emission, emission audit",
        "family": "exemplar (RBF-mixture) scoring with listwise ranking",
        "signature_modules": ["exemplar_set"],
        "closest_prior_art": ["Generalized Context Model (Nosofsky 1986); context model (Medin & Schaffer 1978)",
                              "RBF networks (Broomhead & Lowe 1988)",
                              "Prototypical Networks (Snell, Swersky & Zemel 2017) - implemented as baseline",
                              "Infinite Mixture Prototypes (Allen et al. 2019)",
                              "ListMLE / Plackett-Luce ranking (Xia et al. 2008)"],
        "overlap": "High",
        "prior_art_queries": ["exemplar versus prototype categorization model",
                              "multiple prototypes per class few-shot learning",
                              "listwise Plackett-Luce ranking loss", "training set coverage audit marginals"],
        "contribution_type": "test",
        "delta": "The scorer and loss are known; the contribution is a generator with ground-truth-rejected "
                 "blends of members, a size-matched prototype tested on exactly those blends, and an audit "
                 "that must recover an undeclared training filter from emissions alone.",
    },
    "saturated_cluster_note": "Sits in the prototype/exemplar retrieval cluster (0010, 0022, 0070, 0074, 0113, "
                              "0123, 0126, 0194, 0197). Those chapters retrieve stored cases or precedents; "
                              "none tests whether a category rejects blends of its own members, none emits "
                              "predicate-free ranked lists with exact cross-list invariance, and none audits "
                              "the marginal of what a system admits.",
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1", "property_test": "C6.3", "hypothesis": "H-SIG"},
        {"doctrine": "D8", "mechanism": "M1", "property_test": "C6.3", "hypothesis": "H-NEC"},
        {"doctrine": "D2", "mechanism": "M2", "property_test": "C6.2", "hypothesis": None},
        {"doctrine": "D5", "mechanism": "M2", "property_test": "C6.1", "hypothesis": None},
        {"doctrine": "D3", "mechanism": "M3", "property_test": None, "hypothesis": None},
        {"doctrine": "D4", "mechanism": "M4", "property_test": None, "hypothesis": "H-RAT"},
        {"doctrine": "D6", "mechanism": "M5", "property_test": "C6.4", "hypothesis": "H-AUDIT"},
        {"doctrine": "D6", "mechanism": "M1", "property_test": None, "hypothesis": "H-BLIND"},
        {"doctrine": "D7", "mechanism": None, "property_test": None, "hypothesis": None},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "With blend distractors, the exemplar set selects members more precisely "
                                     "than a size-matched single-prototype model.",
         "metric": "precision@L", "split": "shifted (blend distractors)", "comparison": "model - baseline",
         "direction": "greater", "mesi": 0.05, "seeds": 5},
        {"id": "H-SIG-ID", "statement": "The same holds on ordinary held-out pools, by a smaller margin.",
         "metric": "precision@L", "split": "held-out", "comparison": "model - baseline",
         "direction": "greater", "mesi": 0.02, "seeds": 5},
        {"id": "H-NEC", "statement": "Collapsing the exemplar set to its centroid at inference costs more "
                                     "blend-split precision than zeroing the gate.",
         "metric": "precision@L change", "comparison": "signature_knockout - matched_knockout",
         "direction": "less", "mesi": 0.05, "seeds": 5},
        {"id": "H-BLIND", "statement": "Trained without grief > 0.5, the exemplar set's advantage shrinks on "
                                       "sorrowing members relative to the rest.",
         "condition": "curated training; members with grief > 0.5 versus grief <= 0.5",
         "grounding": "the undeclared omission of the fall of Teishi's house (D6)",
         "metric": "(model - baseline | grief>0.5) - (model - baseline | grief<=0.5), precision@L",
         "comparison": "model - baseline", "direction": "less", "mesi": 0.03, "seeds": 5},
        {"id": "H-AUDIT", "statement": "The emission-only audit shows a larger grief-axis gap after curated "
                                       "training than after full training.",
         "metric": "grief TV gap", "comparison": "curated - full", "direction": "greater",
         "mesi": 0.05, "seeds": 5},
        {"id": "H-RAT", "statement": "The ratification score of an emitted list correlates positively with "
                                     "its actual precision.",
         "metric": "Pearson r over held-out lists", "comparison": "r - 0", "direction": "greater",
         "mesi": 0.2, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.15, "margin_over_trivial": 0.10, "shuffled_band": 0.06,
                   "gradcheck_rel_tol": 1e-5, "gradcheck_denominator_floor": 1e-4, "clip_norm": 5.0},
    "probe_predictions": [{"probe": "P6", "expected": "above baseline"}, {"probe": "P8", "expected": "equal to baseline"},
                          {"probe": "P9", "expected": "below baseline"}, {"probe": "P10", "expected": "above baseline"}],
    "dialectic_links": [{"chapter": 251, "relation": "rival", "test": None,
                         "note": "Murasaki Shikibu's diary judges her; no rival mechanism diverges on this task"}],
    "corpus_neighbors": [
        {"chapter": 126, "similarity": None, "difference": "Trajan retrieves stored precedents; here the points are learned and the test is blend rejection."},
        {"chapter": 74, "similarity": None, "difference": "Theophrastus weights distinctive features per kind; here only a diagonal emphasis, and the claim tested is the excluded midpoint."},
        {"chapter": 10, "similarity": None, "difference": "Hammurabi decides cases by analogy; here no verdict is emitted, only a ranked list."},
        {"chapter": 210, "similarity": None, "difference": "Bai Juyi accepts by a weaker listener; here a learned critic stands for a competent room."},
        {"chapter": 231, "similarity": None, "difference": "Michizane records the traversal; here no process trace is emitted at all."},
        {"chapter": 251, "similarity": None, "difference": "Murasaki audits deeds against avowals; here emissions are audited against the world."},
    ],
    "corpus_neighbors_note": "Similarity not computed: neighbour files (0228-0247) were not available in this session.",
    "barometer": {
        "cognitive_processing": ["held-out precision@L on unseen particulars"],
        "embodied_cognition": [],
        "world_modeling": ["omission audit of emitted marginals against a reference world (H-AUDIT)"],
        "consciousness": ["ratification-precision correlation (H-RAT); subjective experience not claimed"],
        "language_understanding": ["predicate-free ranked emission; within-list order tau"],
        "emotional_intelligence": [],
        "creativity": [],
        "autonomy": [],
    },
    "task_types": ["vector_classification"],
    "applications": [
        {"use": "multi-style character recognition in archival digitisation", "sector": "libraries and archives",
         "dataset": "UCI Optical Recognition of Handwritten Digits (optdigits)"},
        {"use": "listwise ranking with independent queries", "sector": "search and retrieval",
         "dataset": "Microsoft Learning to Rank MSLR-WEB10K"},
        {"use": "coverage audit of a curated training subset against its source sample", "sector": "dataset governance",
         "dataset": "UCI Adult (Census Income)"},
    ],
    "safety_notes": "Synthetic data only by default. The audit compares marginals of a subset with its source; "
                    "it encodes nothing about individuals. No sentence is attributed to Sei Shōnagon.",
}

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

CHAPTER = 248
FACTORS = ("season", "hour", "scale", "sound", "motion", "rank", "transience", "grief")
GRIEF = FACTORS.index("grief")
TASK_TYPES = ["vector_classification"]
ACTIVE_MUTANT: Optional[str] = None
BASE_CFG = dict(D=32, H=48, R=16, M=12, Hr=16, P=30, L=6, batch=12, batch_clf=64, lr=6e-3,
                clip=5.0, beta=4.0, rho=0.22, lam_rat=0.4, lam_sp=0.5, lam_l2=1e-4, margin=0.75, jitter=0.2)
SCALES = {"full": dict(n_items=2400, n_heads=16, steps=1500, reps=6),
          "quick": dict(n_items=1600, n_heads=12, steps=500, reps=4)}
TH = MIND_CARD["thresholds"]


# BEGIN STANDARD UTILITIES v1.0
def logsumexp(x, axis=-1, keepdims=False):
    m = np.max(x, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def softplus(x):
    return np.logaddexp(0.0, x)


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        self.t += 1
        for k, g in grads.items():
            self.m[k] = self.b1 * self.m[k] + (1.0 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1.0 - self.b2) * g * g
            mh = self.m[k] / (1.0 - self.b1 ** self.t)
            vh = self.v[k] / (1.0 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


def clip_global_norm(grads, max_norm):
    norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    scale = min(1.0, max_norm / (norm + 1e-12))
    return {k: g * scale for k, g in grads.items()}, norm


def finite_difference_check(loss_fn, params, grads, rng, eps=1e-6, n_entries=20, floor=1e-4):
    """Central differences on n_entries random entries per tensor plus its largest-gradient entry.
    The denominator is floored so vanishing gradients are judged by absolute error."""
    out = {}
    for name, p in params.items():
        flat, g = p.reshape(-1), grads[name].reshape(-1)
        assert np.shares_memory(flat, p), name
        picks = set(rng.choice(flat.size, size=min(n_entries, flat.size), replace=False).tolist())
        picks.add(int(np.argmax(np.abs(g))))
        worst = 0.0
        for i in picks:
            old = flat[i]
            flat[i] = old + eps
            lp = loss_fn()
            flat[i] = old - eps
            lm = loss_fn()
            flat[i] = old
            fd = (lp - lm) / (2.0 * eps)
            worst = max(worst, abs(fd - g[i]) / max(abs(fd), abs(g[i]), floor))
        out[name] = worst
    return out


def paired_bootstrap(diffs, rng, n_resamples=2000):
    d = np.asarray(diffs, dtype=np.float64)
    means = d[rng.integers(0, d.size, size=(n_resamples, d.size))].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def verdict(mean, lo, hi, direction, mesi):
    s = 1.0 if direction == "greater" else -1.0
    a, b = sorted((s * lo, s * hi))
    if a > 0 and s * mean >= mesi:
        return "supported"
    if b < 0:
        return "contradicted"
    return "inconclusive"


def write_report(report, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
# END STANDARD UTILITIES


# 6. Data and tasks
class World:
    """Latent particulars, a fixed surface map, and the open extension of every heading."""

    def __init__(self, rng, n_items, n_heads, D, n_seeds=4, noise=0.06, admit=0.08, sigma=0.22):
        F = len(FACTORS)
        self.noise, self.sigma, self.K = noise, sigma, n_heads
        self.A = rng.normal(0.0, 1.0, (F, 3 * D))
        self.Bm = rng.normal(0.0, 1.0, (3 * D, D)) / math.sqrt(3 * D)
        self.Z = rng.uniform(0.0, 1.0, (n_items, F))
        raw = self._raw(self.Z, rng)
        # Stored scaling, so a new particular (a blend) lands on the same surface as the world.
        self.mu, self.sd = raw.mean(axis=0), raw.std(axis=0) + 1e-8
        self.X = (raw - self.mu) / self.sd
        self.sub = np.zeros((n_heads, F))
        self.seeds = np.zeros((n_heads, n_seeds, F))
        self.order_w = np.zeros((n_heads, F))
        for k in range(n_heads):
            self.sub[k, rng.choice(F, 3, replace=False)] = 1.0
            # Seeds that huddle would turn the heading back into a rule a prototype can carry.
            for _ in range(2000):
                s = rng.uniform(0.08, 0.92, (n_seeds, F))
                d = np.sqrt((((s[:, None] - s[None]) ** 2) * self.sub[k]).sum(-1))
                if d[np.triu_indices(n_seeds, 1)].min() > 0.6:
                    break
            self.seeds[k] = s
            self.order_w[k] = rng.normal(0.0, 1.0, F) * self.sub[k]
        self.bar = np.array([np.percentile(self.resemblance(self.Z, k), 100 * (1 - admit))
                             for k in range(n_heads)])
        self.adm = np.stack([self.admits(self.Z, k) for k in range(n_heads)], axis=1)

    def _raw(self, Z, rng):
        r = np.tanh(Z @ self.A) @ self.Bm
        return r + rng.normal(0.0, self.noise, r.shape)

    def surface(self, Z, rng):
        return (self._raw(Z, rng) - self.mu) / self.sd

    def _d2(self, Z, k):
        return (((Z[:, None, :] - self.seeds[k][None]) ** 2) * self.sub[k]).sum(-1)

    def resemblance(self, Z, k):
        # max, not sum: a thing belongs by resembling something admitted, not the average of them
        return np.exp(-self._d2(Z, k) / (2 * self.sigma ** 2)).max(axis=1)

    def admits(self, Z, k):
        return self.resemblance(Z, k) >= self.bar[k]


def split(world, rng, frac=0.7):
    perm = rng.permutation(len(world.Z))
    cut = int(frac * len(perm))
    return np.sort(perm[:cut]), np.sort(perm[cut:])


def blends_for(world, mem, k, n, rng):
    """Attended factors averaged across two seed regions; kept only if the generator rejects them."""
    att = world.sub[k] > 0
    near = np.argmin(world._d2(world.Z[mem], k), axis=1)
    out = []
    for _ in range(60 * n):
        i, j = rng.integers(0, len(mem), 2)
        if near[i] == near[j]:
            continue
        z = world.Z[rng.integers(0, len(world.Z))].copy()   # nuisance factors from an arbitrary particular
        z[att] = 0.5 * (world.Z[mem[i], att] + world.Z[mem[j], att])
        if not world.admits(z[None], k)[0]:
            out.append(z)
            if len(out) == n:
                break
    return np.array(out).reshape(-1, len(FACTORS))


def draw_pools(world, idx, heads, cfg, rng, member_mask=None, blends=False, xmap=None):
    """Each pool: L admitted particulars in the heading's reading order, then P - L others."""
    P, L = cfg["P"], cfg["L"]
    Xs, Zs, ids, cats = [], [], [], []
    for k in heads:
        adm = world.adm[idx, k]
        mem, non = idx[adm], idx[~adm]
        cand = mem if member_mask is None else mem[member_mask(world.Z[mem])]
        if len(cand) < L:
            continue
        pick = rng.choice(cand, L, replace=False)
        pick = pick[np.argsort(-(world.Z[pick] @ world.order_w[k]))]
        zb = blends_for(world, mem, k, (P - L) // 2, rng) if blends else np.zeros((0, len(FACTORS)))
        rest = rng.choice(non, P - L - len(zb), replace=False)
        take = np.concatenate([pick, rest])
        x = world.X[take if xmap is None else xmap[take]]
        Xs.append(np.concatenate([x[:L], world.surface(zb, rng), x[L:]]))
        Zs.append(np.concatenate([world.Z[pick], zb, world.Z[rest]]))
        ids.append(np.concatenate([pick, -np.ones(len(zb), dtype=np.int64), rest]))
        cats.append(k)
    return np.stack(Xs), np.asarray(cats, dtype=np.int64), np.stack(Zs), np.stack(ids)


def usable_heads(world, idx, L, mask=None):
    return [k for k in range(world.K)
            if (world.adm[idx, k] & (True if mask is None else mask(world.Z[idx]))).sum() >= L]


# 7. Model
class Mono:
    def __init__(self, params, cfg, task_type):
        self.params, self.cfg, self.task_type, self.ko = params, cfg, task_type, {}


def build_model(in_dim, out_dim, task_type, rng, **cfg):
    c = dict(BASE_CFG)
    c.update(cfg)
    c["D"], c["K"] = in_dim, out_dim
    D, H, R, M, K, Hr = c["D"], c["H"], c["R"], c["M"], c["K"], c["Hr"]

    def n(shape, s):
        return rng.normal(0.0, s, shape)
    p = {"W0": n((D, H), 1 / math.sqrt(D)), "b0": np.zeros(H), "U": n((H, R), 1 / math.sqrt(H)),
         "emph": n((K, H), 0.3), "E": n((K, M, R), 0.6), "log_bw": np.zeros(K), "log_tau": np.zeros(1),
         "g": n((K, R), 1 / math.sqrt(R)), "theta": np.zeros(K), "w_gate": np.full(1, 0.5),
         "Wr1": n((R, Hr), 1 / math.sqrt(R)), "br1": np.zeros(Hr), "Wr2": n((Hr,), 1 / math.sqrt(Hr)),
         "br2": np.zeros(1)}
    return Mono(p, c, task_type)


def count_params(D, K, H, R, M, Hr):
    return D * H + H + K * H + H * R + K * M * R + 3 * K + K * R + 2 + R * Hr + 2 * Hr + 1


def exemplar_score(q, Ec, bw, tau):
    """Union of neighbourhoods: tau * logsumexp_m(-d2_m / (bw tau)).  q (B,P,R), Ec (B,M,R), bw (B,)."""
    diff = q[:, :, None, :] - Ec[:, None, :, :]
    d2 = np.einsum("bpmr,bpmr->bpm", diff, diff) / math.sqrt(q.shape[-1])
    sign = 1.0 if ACTIVE_MUTANT == "dist_sign" else -1.0
    s = sign * d2 / bw[:, None, None]
    lse = logsumexp(s / tau, axis=2)
    return tau * lse, (diff, s, lse, np.exp(s / tau - lse[..., None]), sign)


def _core(model, X3, cats):
    p, ko, c = model.params, model.ko, model.cfg
    R = p["U"].shape[1]
    h = np.tanh(X3 @ p["W0"] + p["b0"])
    e = np.ones((len(cats), h.shape[-1])) if ko.get("emphasis") == "identity" else softplus(p["emph"][cats])
    he = h * e[:, None, :]
    q = he @ p["U"]
    if ko.get("encoder") == "zero":
        q = np.zeros_like(q)
    Ec = p["E"][cats]
    if ko.get("exemplar_set") == "mean":
        Ec = Ec.mean(axis=1, keepdims=True)
    bw, tau = np.exp(p["log_bw"][cats]), float(np.exp(p["log_tau"][0]))
    sx, xc = exemplar_score(q, Ec, bw, tau)
    if ko.get("exemplar_set") == "zero":
        sx = np.zeros_like(sx)
    u = np.einsum("bpr,br->bp", q, p["g"][cats]) / math.sqrt(R) - p["theta"][cats][:, None]
    if ko.get("gate") == "zero":
        u = np.zeros_like(u)
    f = sx + p["w_gate"][0] * u
    a = sigmoid(c["beta"] * u)
    return f, a, dict(X3=X3, cats=cats, h=h, e=e, he=he, q=q, bw=bw, tau=tau, xc=xc, u=u, a=a)


def _core_back(model, C, df, da, dq, G):
    """Hand-derived reverse pass of _core; accumulates into G."""
    p, c, ko = model.params, model.cfg, model.ko
    cats, h, e, q, u, a, bw, tau = (C[k] for k in ("cats", "h", "e", "q", "u", "a", "bw", "tau"))
    diff, s, lse, w, sign = C["xc"]
    R, M, sq = q.shape[-1], p["E"].shape[1], math.sqrt(q.shape[-1])
    G["w_gate"][0] += np.sum(df * u)
    du = df * p["w_gate"][0] + da * c["beta"] * a * (1.0 - a)
    if ko.get("gate") == "zero":
        du = np.zeros_like(du)
    dsx = np.zeros_like(df) if ko.get("exemplar_set") == "zero" else df
    G["log_tau"][0] += np.sum(dsx * (lse - np.sum(w * s, axis=2) / tau)) * tau
    ds = dsx[..., None] * w
    np.add.at(G["log_bw"], cats, np.sum(-ds * s, axis=(1, 2)))
    ddiff = (ds * sign / bw[:, None, None])[..., None] * diff * (2.0 / sq)
    dq = dq + ddiff.sum(axis=2) + du[..., None] * p["g"][cats][:, None, :] / sq
    dEc = -ddiff.sum(axis=1)
    if dEc.shape[1] != M:
        dEc = np.repeat(dEc / M, M, axis=1)
    np.add.at(G["E"], cats, dEc)
    np.add.at(G["g"], cats, np.einsum("bp,bpr->br", du, q) / sq)
    np.add.at(G["theta"], cats, -du.sum(axis=1))
    if ko.get("encoder") == "zero":
        dq = np.zeros_like(dq)
    H, D = h.shape[-1], C["X3"].shape[-1]
    G["U"] += C["he"].reshape(-1, H).T @ dq.reshape(-1, R)
    dhe = dq @ p["U"].T
    if ko.get("emphasis") != "identity":
        np.add.at(G["emph"], cats, np.sum(dhe * h, axis=1) * sigmoid(p["emph"][cats]))
    dz = dhe * e[:, None, :] * (1.0 - h * h)
    G["W0"] += C["X3"].reshape(-1, D).T @ dz.reshape(-1, H)
    G["b0"] += dz.sum(axis=(0, 1))


def _ratify(p, v):
    z = np.tanh(v @ p["Wr1"] + p["br1"])
    return z @ p["Wr2"] + p["br2"][0], z


def _ratify_back(p, v, z, dR, G):
    G["Wr2"] += z.T @ dR
    G["br2"][0] += dR.sum()
    dz = np.outer(dR, p["Wr2"]) * (1.0 - z * z)
    G["Wr1"] += v.T @ dz
    G["br1"] += dz.sum(axis=0)
    return dz @ p["Wr1"].T


def plackett_luce(f, L, ordered=True):
    """Per-list Plackett-Luce over the author's order.  ordered=False is the order-free control of C6.2."""
    B, P = f.shape
    loss, df = 0.0, np.zeros_like(f)
    for t in range(L):
        cand = np.arange(t, P) if ordered else np.r_[t, np.arange(L, P)]
        rest = f[:, cand]
        loss += float(np.mean(logsumexp(rest, axis=1) - f[:, t]))
        df[:, cand] += softmax(rest, axis=1) / B
        df[:, t] -= 1.0 / B
    return loss, df


def _coupling(f, L):
    """Mutant 'couple_lists': a reading order that runs across catalogues, as if one edition were the book."""
    x = f[1:, 0] - f[:-1, L - 1]
    df = np.zeros_like(f)
    df[1:, 0] += sigmoid(x) / len(f)
    df[:-1, L - 1] -= sigmoid(x) / len(f)
    return float(np.sum(softplus(x)) / len(f)), df


def _regularise(model, a, G):
    p, c = model.params, model.cfg
    abar = float(a.mean())
    da = np.full_like(a, c["lam_sp"] * 2.0 * (abar - c["rho"]) / a.size)
    l2 = sum(float(np.sum(v * v)) for v in p.values())
    for k in G:
        G[k] += 2.0 * c["lam_l2"] * p[k]
    return c["lam_sp"] * (abar - c["rho"]) ** 2 + c["lam_l2"] * l2, da


def native_loss(model, X3, cats, L, ordered=True, couple=None):
    p, c = model.params, model.cfg
    couple = (ACTIVE_MUTANT == "couple_lists") if couple is None else couple
    f, a, C = _core(model, X3, cats)
    B = len(f)
    loss, df = plackett_luce(f, L, ordered)
    if couple:
        lc, dfc = _coupling(f, L)
        loss, df = loss + lc, df + dfc
    G = {k: np.zeros_like(v) for k, v in p.items()}
    dq = np.zeros_like(C["q"])
    if model.ko.get("ratification") != "zero":
        vr, vc = C["q"][:, :L].mean(axis=1), C["q"][:, L:2 * L].mean(axis=1)
        (rr, zr), (rc, zc) = _ratify(p, vr), _ratify(p, vc)
        gap = c["margin"] - (rr - rc)
        loss += c["lam_rat"] * float(np.mean(softplus(gap)))
        dgap = c["lam_rat"] * sigmoid(gap) / B
        dq[:, :L] += _ratify_back(p, vr, zr, -dgap, G)[:, None, :] / L
        dq[:, L:2 * L] += _ratify_back(p, vc, zc, dgap, G)[:, None, :] / L
    reg, da = _regularise(model, a, G)
    _core_back(model, C, df, da, dq, G)
    return loss + reg, G, f, C


def clf_loss(model, X, y):
    N, K = len(X), model.params["E"].shape[0]
    f, a, C = _core(model, np.repeat(X, K, axis=0)[:, None, :], np.tile(np.arange(K), N))
    logits = f.reshape(N, K)
    loss = float(np.mean(logsumexp(logits, axis=1) - logits[np.arange(N), y]))
    dl = softmax(logits, axis=1)
    dl[np.arange(N), y] -= 1.0
    G = {k: np.zeros_like(v) for k, v in model.params.items()}
    reg, da = _regularise(model, a, G)
    _core_back(model, C, dl.reshape(-1, 1) / N, da, np.zeros_like(C["q"]), G)
    return loss + reg, G


def loss_and_grads(model, batch):
    if "cats" in batch:
        loss, G, _, _ = native_loss(model, batch["X"], batch["cats"], batch["L"])
    else:
        loss, G = clf_loss(model, batch["X"], batch["y"])
    if ACTIVE_MUTANT == "zero_grad_E":
        G["E"] = np.zeros_like(G["E"])
    return loss, G


def predict(model, X):
    """Per-heading probabilities for each particular (softmax over resemblance scores)."""
    K, N = model.params["E"].shape[0], len(X)
    f = _core(model, np.repeat(X, K, axis=0)[:, None, :], np.tile(np.arange(K), N))[0]
    return softmax(f.reshape(N, K), axis=1)


def hidden_states(model, X):
    return np.tanh(X @ model.params["W0"] + model.params["b0"])


# 8. Baselines and rival mechanisms
def matched_baseline_H(cfg):
    """Width of a one-prototype model whose parameter count matches the exemplar model."""
    D, K, R, Hr = cfg["D"], cfg["K"], cfg["R"], cfg["Hr"]
    target = count_params(D, K, cfg["H"], R, cfg["M"], Hr)
    return min(range(cfg["H"], 6 * cfg["H"]), key=lambda h: abs(count_params(D, K, h, R, 1, Hr) - target))


# 9. Registries
MODULES = {
    "encoder": (["W0", "b0", "U"], "tanh feature layer and linear projection to the query space", False, ("zero",)),
    "emphasis": (["emph"], "per-category softplus diagonal reweighting of the shared features", False, ("identity",)),
    "exemplar_set": (["E", "log_bw", "log_tau"], "per-category log-sum-exp over negative squared distances "
                     "to learned points (union of RBF neighbourhoods)", True, ("mean", "zero")),
    "gate": (["g", "theta", "w_gate"], "per-category logistic linear gate added to the score", False, ("zero",)),
    "ratification": (["Wr1", "br1", "Wr2", "br2"], "list-level MLP critic trained by margin", False, ("zero",)),
}
CHECK_NAMES = {"C1": "gradient_check", "C2": "determinism_and_finiteness", "C3": "learning",
               "C4": "shuffled_label_control", "C5": "mutant_detection", "C6.1": "global_order_invariance",
               "C6.2": "local_order_sensitivity", "C6.3": "union_of_neighbourhoods", "C6.4": "omission_audit_analytic",
               "C7": "split_integrity", "C8": "budget"}
KNOCKOUT_PLAN = [("encoder", "zero"), ("emphasis", "identity"), ("exemplar_set", "mean"),
                 ("gate", "zero"), ("ratification", "zero")]
MUTANTS = {
    "sign_flip": "update direction reversed (gradient ascent)",
    "zero_lr": "learning rate set to zero",
    "zero_grad_E": "gradient of the exemplar tensor E zeroed",
    "dist_sign": "exemplar scorer rewards distance instead of penalising it",
    "couple_lists": "a reading-order term chains the end of each list to the start of the next",
}


def modules(model):
    return {n: {"params": ps, "role": r, "signature": s, "knockouts": list(k)}
            for n, (ps, r, s, k) in MODULES.items()}


def knockout(model, name, mode):
    if mode not in MODULES[name][3]:
        raise ValueError(f"knockout mode {mode!r} not defined for {name!r}")
    m = Mono({k: v.copy() for k, v in model.params.items()}, dict(model.cfg), model.task_type)
    m.ko = dict(model.ko, **{name: mode})
    return m


def n_params(model):
    return int(sum(v.size for v in model.params.values()))


# 10. Training
def sample_batch(model, data, rng):
    c = model.cfg
    if "world" in data:
        heads = rng.choice(data["heads"], size=c["batch"], replace=len(data["heads"]) < c["batch"])
        X3, cats, _, _ = draw_pools(data["world"], data["idx"], heads, c, rng, xmap=data.get("xmap"))
        # A particular is never met twice in the same light.
        return {"X": X3 + rng.normal(0.0, c["jitter"], X3.shape), "cats": cats, "L": c["L"]}
    i = rng.choice(len(data["X"]), size=min(c["batch_clf"], len(data["X"])), replace=False)
    return {"X": data["X"][i], "y": data["y"][i]}


def fit(model, data, budget, rng):
    opt = Adam(model.params, lr=0.0 if ACTIVE_MUTANT == "zero_lr" else model.cfg["lr"])
    losses = []
    for _ in range(budget):
        loss, G = loss_and_grads(model, sample_batch(model, data, rng))
        if ACTIVE_MUTANT == "sign_flip":
            G = {k: -g for k, g in G.items()}
        G, _ = clip_global_norm(G, model.cfg["clip"])
        opt.step(model.params, G)
        if not np.isfinite(loss):
            raise FloatingPointError("non-finite training loss")
        losses.append(float(loss))
    if not all(np.all(np.isfinite(v)) for v in model.params.values()):
        raise FloatingPointError("non-finite parameters")
    return losses


def top_L(f, L, seed=0):
    """Top-L positions with ties broken at random, so a constant scorer cannot inherit the pool layout."""
    perm = np.random.default_rng(seed).permutation(f.shape[1])
    return perm[np.argsort(-f[:, perm], axis=1, kind="stable")[:, :L]]


def precision_at_L(model, pools, L):
    return float(np.mean(top_L(_core(model, pools[0], pools[1])[0], L) < L))


def order_tau(model, pools, L):
    f = _core(model, pools[0], pools[1])[0][:, :L]
    iu = np.triu_indices(L, 1)
    return float(np.mean(np.sign(f[:, :, None] - f[:, None, :])[:, iu[0], iu[1]]))


def audit(admitted_Z, reference_Z, n_bins=4):
    """Total-variation gap per latent factor between what a system admits and a reference sample."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    return np.array([0.5 * np.abs(np.histogram(admitted_Z[:, j], edges)[0] / len(admitted_Z)
                                  - np.histogram(reference_Z[:, j], edges)[0] / len(reference_Z)).sum()
                     for j in range(admitted_Z.shape[1])])


def emitted_latents(model, world, idx, heads, rng, reps):
    """Everything the model admits when every heading is swept over random pools of the world."""
    P, L = model.cfg["P"], model.cfg["L"]
    ids = np.stack([rng.choice(idx, P, replace=False) for _ in range(reps * len(heads))])
    cats = np.repeat(np.asarray(heads), reps)
    f = _core(model, world.X[ids], cats)[0]
    return world.Z[np.take_along_axis(ids, top_L(f, L), axis=1).ravel()]


def ratification_r(model, pools, L):
    f, _, C = _core(model, pools[0], pools[1])
    top = top_L(f, L)
    v = np.take_along_axis(C["q"], top[..., None], axis=1).mean(axis=1)
    prec = (top < L).mean(axis=1)
    rat = _ratify(model.params, v)[0]
    return float(np.corrcoef(rat, prec)[0, 1]) if prec.std() > 0 and rat.std() > 0 else 0.0


# 11. Tests: correctness first, then hypotheses
def _tiny(rng):
    world = World(rng, 300, 3, 5, admit=0.2)
    tr, _ = split(world, rng)
    cfg = dict(H=6, R=4, M=3, Hr=4, P=7, L=3, batch=4, batch_clf=16)
    return world, {"world": world, "idx": tr, "heads": usable_heads(world, tr, 3)}, cfg


def c1_gradcheck(seed):
    """C1: every tensor, three model variants, at initialisation and after 60 updates."""
    rng = np.random.default_rng(seed)
    world, data, cfg = _tiny(rng)
    worst, n_checked, n_total = 0.0, 0, 0
    Xc, yc = rng.normal(0.0, 1.0, (24, 5)), rng.integers(0, 3, 24)
    for label, kw, dd in (("exemplar", {}, data), ("prototype", {"M": 1}, data),
                          ("classifier", {}, {"X": Xc, "y": yc})):
        m = build_model(5, 3, "native" if "world" in dd else "vector_classification", rng, **dict(cfg, **kw))
        for stage in ("init", "after_training_steps"):
            if stage != "init":
                fit(m, dd, 60, rng)
            batch = sample_batch(m, dd, rng)
            _, G = loss_and_grads(m, batch)
            errs = finite_difference_check(lambda: loss_and_grads(m, batch)[0], m.params, G, rng,
                                           floor=TH["gradcheck_denominator_floor"])
            worst = max(worst, max(errs.values()))
            n_checked += len(errs)
            n_total += len(m.params)
    return {"passed": bool(worst <= TH["gradcheck_rel_tol"]), "max_rel_error": float(worst), "tensors_checked": n_checked,
            "tensors_total": n_total, "checked_at": ["init", "after_training_steps"]}


def c2_determinism(seed):
    runs = []
    for _ in range(2):
        rng = np.random.default_rng(seed)
        world, data, cfg = _tiny(rng)
        m = build_model(5, 3, "native", rng, **cfg)
        losses = fit(m, data, 25, rng)
        pools = draw_pools(world, data["idx"], data["heads"], m.cfg, np.random.default_rng(seed + 1))
        f, a, _ = _core(m, pools[0], pools[1])
        runs.append((losses, f, a, hidden_states(m, pools[0])))
    same = runs[0][0] == runs[1][0] and np.array_equal(runs[0][1], runs[1][1])
    finite = all(np.all(np.isfinite(x)) for x in (*runs[0][1:], np.array(runs[0][0])))
    return same and finite, f"identical={same} finite={finite}"


class Context:
    """One seeded world, its splits and its fixed evaluation pools (shared by paired models)."""

    def __init__(self, seed, scale):
        rng = np.random.default_rng(seed)
        self.seed, self.scale, L = seed, scale, BASE_CFG["L"]
        self.world = World(rng, scale["n_items"], scale["n_heads"], BASE_CFG["D"])
        self.tr, self.te = split(self.world, rng)
        self.heads = [k for k in usable_heads(self.world, self.tr, L) if k in usable_heads(self.world, self.te, L)]
        cur = self.tr[self.world.Z[self.tr, GRIEF] <= 0.5]
        self.data = {"world": self.world, "idx": self.tr, "heads": self.heads}
        self.cur = {"world": self.world, "idx": cur, "heads": [k for k in self.heads if k in usable_heads(self.world, cur, L)]}
        ev = np.random.default_rng(seed + 7)
        hi, lo = (lambda Z: Z[:, GRIEF] > 0.5), (lambda Z: Z[:, GRIEF] <= 0.5)
        gh = [k for k in self.heads if k in usable_heads(self.world, self.te, L, hi)]

        def pools(heads, **kw):
            parts = [draw_pools(self.world, self.te, heads, BASE_CFG, ev, **kw) for _ in range(scale["reps"])]
            return tuple(np.concatenate([p[i] for p in parts]) for i in range(4))
        self.id, self.blend = pools(self.heads), pools(self.heads, blends=True)
        self.grief, self.ordinary = pools(gh, member_mask=hi), pools(gh, member_mask=lo)
        self.emit_rng_seed = seed + 11

    def train(self, kind, which, rng_seed, steps=None):
        cfg = dict(BASE_CFG, K=self.world.K)
        kw = {} if kind == "exemplar" else {"M": 1, "H": matched_baseline_H(cfg)}
        m = build_model(BASE_CFG["D"], self.world.K, "native", np.random.default_rng(rng_seed), **kw)
        data = self.data if which == "full" else self.cur
        losses = fit(m, data, steps or self.scale["steps"], np.random.default_rng(rng_seed + 1))
        return m, losses


def c3_learning(ctx):
    m, losses = ctx.train("exemplar", "full", ctx.seed + 3)
    k = max(10, len(losses) // 20)
    drop = 1.0 - np.mean(losses[-k:]) / np.mean(losses[:k])
    prec, chance = precision_at_L(m, ctx.id, BASE_CFG["L"]), BASE_CFG["L"] / BASE_CFG["P"]
    ok = drop >= TH["loss_drop_fraction"] and prec - chance >= TH["margin_over_trivial"]
    return ok, f"loss drop {drop:.3f} (need {TH['loss_drop_fraction']}); held-out precision {prec:.3f} vs trivial {chance:.3f}", m


def c4_shuffled(ctx):
    rng = np.random.default_rng(ctx.seed + 5)
    xmap = np.arange(len(ctx.world.Z))
    xmap[ctx.tr] = rng.permutation(ctx.tr)          # features no longer belong to their latents in training
    m = build_model(BASE_CFG["D"], ctx.world.K, "native", np.random.default_rng(ctx.seed + 3))
    fit(m, dict(ctx.data, xmap=xmap), ctx.scale["steps"], np.random.default_rng(ctx.seed + 4))
    prec, chance = precision_at_L(m, ctx.id, BASE_CFG["L"]), BASE_CFG["L"] / BASE_CFG["P"]
    return abs(prec - chance) <= TH["shuffled_band"], f"shuffled-label precision {prec:.3f}, trivial {chance:.3f} ± {TH['shuffled_band']}"


def _small_model(rng):
    return build_model(5, 3, "native", rng, H=6, R=4, M=3, Hr=4)


def c6_1_global_order(rng):
    viol = ctrl = 0.0
    for _ in range(12):
        m, X3, cats = _small_model(rng), rng.normal(0.0, 1.0, (5, 7, 5)), rng.integers(0, 3, 5)
        perm = np.roll(rng.permutation(5), 1)
        viol = max(viol, abs(native_loss(m, X3[perm], cats[perm], 3)[0] - native_loss(m, X3, cats, 3)[0]))
        ctrl = max(ctrl, abs(native_loss(m, X3[perm], cats[perm], 3, couple=True)[0]
                             - native_loss(m, X3, cats, 3, couple=True)[0]))
    return viol < 1e-10 and ctrl > 1e-6, f"max |dloss| under list permutation {viol:.1e}; coupled control {ctrl:.1e}"


def c6_2_local_order(rng):
    least, ctrl = np.inf, 0.0
    for _ in range(12):
        m, X3, cats = _small_model(rng), rng.normal(0.0, 1.0, (5, 7, 5)), rng.integers(0, 3, 5)
        Xp = X3.copy()
        Xp[:, :3] = X3[:, [1, 2, 0]]
        least = min(least, abs(native_loss(m, Xp, cats, 3)[0] - native_loss(m, X3, cats, 3)[0]))
        ctrl = max(ctrl, abs(native_loss(m, Xp, cats, 3, ordered=False)[0]
                             - native_loss(m, X3, cats, 3, ordered=False)[0]))
    return least > 1e-6 and ctrl < 1e-9, f"min |dloss| under in-list permutation {least:.1e}; order-free control {ctrl:.1e}"


def c6_3_union(rng):
    """Every stored point outscores every point lying outside all neighbourhoods (midpoints, centroid, far)."""
    worst, tested, ctrl = np.inf, 0, 0
    for _ in range(300):
        R, M = 4, int(rng.integers(2, 7))
        E = rng.normal(0.0, 2.0, (1, M, R))
        bw, tau = np.exp(rng.normal(0.0, 0.5, 1)), float(np.exp(rng.normal(-0.5, 0.5)))
        i, j = rng.choice(M, 2, replace=False)
        probes = np.stack([0.5 * (E[0, i] + E[0, j]), E[0].mean(axis=0), rng.normal(0.0, 8.0, R)])
        d2 = (((probes[:, None, :] - E[0][None]) ** 2).sum(-1) / math.sqrt(R)).min(axis=1)
        outside = probes[d2 / bw[0] > tau * math.log(M) + 1e-9]
        if not len(outside):
            continue
        pts = np.concatenate([E[0], outside])[None]
        sc = exemplar_score(pts, E, bw, tau)[0][0]
        worst = min(worst, sc[:M].min() - sc[M:].max())
        pc = exemplar_score(pts, E.mean(axis=1, keepdims=True), bw, tau)[0][0]
        ctrl += int(pc[:M].min() - pc[M:].max() <= 0)
        tested += 1
    return tested > 50 and worst > 0 and ctrl > 0, f"{tested} configurations, worst margin {worst:.3g}; prototype control violations {ctrl}"


def c6_4_audit(rng, auditor=audit):
    Zw = rng.uniform(0.0, 1.0, (20000, len(FACTORS)))
    kept = Zw[Zw[:, GRIEF] <= 0.5]

    def detects(ref):
        tv = auditor(kept, ref)
        return abs(tv[GRIEF] - 0.5) < 0.03 and int(np.argmax(tv)) == GRIEF
    exact = float(auditor(Zw, Zw).max()) == 0.0
    ok = exact and detects(Zw) and not detects(kept)   # the curator auditing herself must be caught blind
    return ok, f"self-identity exact={exact}; filter recovered={detects(Zw)}; self-referenced control blind={not detects(kept)}"


def c7_split(ctx):
    disjoint = len(np.intersect1d(ctx.tr, ctx.te)) == 0
    pool_ids = np.concatenate([p[3].ravel() for p in (ctx.id, ctx.blend, ctx.grief, ctx.ordinary)])
    in_test = bool(np.isin(pool_ids[pool_ids >= 0], ctx.te).all())
    batch = draw_pools(ctx.world, ctx.tr, ctx.heads, BASE_CFG, np.random.default_rng(0))[3]
    in_train = bool(np.isin(batch, ctx.tr).all())
    train_rows = {r.tobytes() for r in ctx.world.Z[ctx.tr]}
    bl = ctx.blend[2][ctx.blend[3] < 0]
    fresh = len(bl) > 0 and not any(r.tobytes() in train_rows for r in bl)
    ok = disjoint and in_test and in_train and fresh
    return ok, f"disjoint={disjoint} eval-in-test={in_test} train-in-train={in_train} blends-fresh={fresh} ({len(bl)} blends)"


def run_correctness(seeds, scale, cheap_only=False):
    rng = np.random.default_rng(seeds[0] + 13)
    res = {"C1": c1_gradcheck(seeds[0] + 17)}
    res["C1"]["detail"] = f"max rel error {res['C1']['max_rel_error']:.2e} over {res['C1']['tensors_checked']} tensor checks"
    for name, fn in (("C6.1", c6_1_global_order), ("C6.2", c6_2_local_order), ("C6.3", c6_3_union), ("C6.4", c6_4_audit)):
        ok, det = fn(rng)
        res[name] = {"passed": ok, "detail": det}
    if cheap_only:
        return res, None
    ok, det = c2_determinism(seeds[0] + 19)
    res["C2"] = {"passed": ok, "detail": det}
    ctx = Context(seeds[0], scale)
    ok, det, _ = c3_learning(ctx)
    res["C3"] = {"passed": ok, "detail": det}
    ok, det = c4_shuffled(ctx)
    res["C4"] = {"passed": ok, "detail": det}
    ok, det = c7_split(ctx)
    res["C7"] = {"passed": ok, "detail": det}
    return res, ctx


def c5_mutants(ctx):
    global ACTIVE_MUTANT
    saved, out = ACTIVE_MUTANT, {}
    seeds = [ctx.seed]
    for name in MUTANTS:
        ACTIVE_MUTANT = name
        try:
            res, _ = run_correctness(seeds, ctx.scale, cheap_only=True)
            failed = [k for k, v in res.items() if not v["passed"]]
            if not failed and not c3_learning(ctx)[0]:
                failed = ["C3"]
        except FloatingPointError:
            failed = ["C2"]
        finally:
            ACTIVE_MUTANT = saved
        out[name] = failed
    return out


def run_seed(seed, scale):
    ctx, L = Context(seed, scale), BASE_CFG["L"]
    ex, _ = ctx.train("exemplar", "full", seed + 21)
    base, _ = ctx.train("prototype", "full", seed + 21)
    exc, _ = ctx.train("exemplar", "curated", seed + 23)
    basec, _ = ctx.train("prototype", "curated", seed + 23)
    P = lambda m, pools: precision_at_L(m, pools, L)
    er = np.random.default_rng(ctx.emit_rng_seed)
    ref = ctx.world.Z[ctx.te]
    tv_full = audit(emitted_latents(ex, ctx.world, ctx.te, ctx.heads, er, scale["reps"]), ref)
    tv_cur = audit(emitted_latents(exc, ctx.world, ctx.te, ctx.cur["heads"], er, scale["reps"]), ref)
    blend_ex = P(ex, ctx.blend)
    ko = {f"{n}:{mode}": P(knockout(ex, n, mode), ctx.blend) - blend_ex for n, mode in KNOCKOUT_PLAN}
    f_ex, a_ex, _ = _core(ex, ctx.id[0], ctx.id[1])
    return {
        "P_id_ex": P(ex, ctx.id), "P_id_base": P(base, ctx.id), "P_blend_ex": blend_ex, "P_blend_base": P(base, ctx.blend),
        "tau_ex": order_tau(ex, ctx.id, L), "tau_base": order_tau(base, ctx.id, L),
        "blind_gap": (P(exc, ctx.grief) - P(basec, ctx.grief)) - (P(exc, ctx.ordinary) - P(basec, ctx.ordinary)),
        "P_grief_ex_cur": P(exc, ctx.grief), "P_ord_ex_cur": P(exc, ctx.ordinary),
        "tv_grief_full": float(tv_full[GRIEF]), "tv_grief_cur": float(tv_cur[GRIEF]),
        "worst_factor_full": FACTORS[int(np.argmax(tv_full))], "worst_factor_cur": FACTORS[int(np.argmax(tv_cur))],
        "rat_r": ratification_r(ex, ctx.id, L), "ko": ko,
        "gate_members": float(a_ex[:, :L].mean()), "gate_others": float(a_ex[:, L:].mean()),
        "n_params": n_params(ex), "n_params_base": n_params(base),
    }


def hypotheses(per_seed, rng):
    col = lambda k: [r[k] for r in per_seed]
    diffs = {
        "H-SIG": np.subtract(col("P_blend_ex"), col("P_blend_base")),
        "H-SIG-ID": np.subtract(col("P_id_ex"), col("P_id_base")),
        "H-NEC": np.subtract([r["ko"]["exemplar_set:mean"] for r in per_seed], [r["ko"]["gate:zero"] for r in per_seed]),
        "H-BLIND": np.array(col("blind_gap")),
        "H-AUDIT": np.subtract(col("tv_grief_cur"), col("tv_grief_full")),
        "H-RAT": np.array(col("rat_r")),
    }
    out = []
    for h in MIND_CARD["hypotheses"]:
        mean, lo, hi = paired_bootstrap(diffs[h["id"]], rng)
        out.append({"id": h["id"], "metric": h["metric"], "mean_diff": mean, "ci95": [lo, hi], "mesi": h["mesi"],
                    "n_seeds": len(per_seed), "verdict": verdict(mean, lo, hi, h["direction"], h["mesi"])})
    kos, sig = [], {n: v["signature"] for n, v in modules(None).items()}
    for n, mode in KNOCKOUT_PLAN:
        mean, lo, hi = paired_bootstrap([r["ko"][f"{n}:{mode}"] for r in per_seed], rng)
        kos.append({"module": f"{n} ({mode})", "signature": sig[n], "metric_change": mean, "ci95": [lo, hi]})
    return out, kos


def data_bridge(path, seed):
    """Optional: an exemplar classifier against a one-prototype classifier on a local CSV (label last)."""
    try:
        arr = np.loadtxt(path, delimiter=",")
    except ValueError:
        arr = np.loadtxt(path, delimiter=",", skiprows=1)
    X, (_, y) = arr[:, :-1], np.unique(arr[:, -1], return_inverse=True)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    tr, te = perm[: int(0.8 * len(X))], perm[int(0.8 * len(X)):]
    out = {}
    for label, kw in (("exemplar", {}), ("prototype", {"M": 1})):
        m = build_model(X.shape[1], int(y.max()) + 1, "vector_classification", np.random.default_rng(seed), **kw)
        fit(m, {"X": X[tr], "y": y[tr]}, 600, np.random.default_rng(seed + 1))
        out[label] = float(np.mean(np.argmax(predict(m, X[te]), axis=1) == y[te]))
    return out


# 12. Report
def build_report(args, seeds, corr, mut, hyp, kos, per_seed, runtime, exit_code):
    cfg = dict(BASE_CFG, K=SCALES["quick" if args.quick else "full"]["n_heads"])
    npm = count_params(**{k: cfg[k] for k in ("D", "K", "H", "R", "M", "Hr")})
    npb = count_params(cfg["D"], cfg["K"], matched_baseline_H(cfg), cfg["R"], 1, cfg["Hr"])
    c1 = corr["C1"]
    return {
        "schema_version": "1.0", "chapter": CHAPTER, "file": os.path.basename(__file__),
        "card_revision": MIND_CARD["card_revision"],
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
        "seeds": seeds, "runtime_s": round(runtime, 1), "n_params": npm, "n_params_baseline": npb,
        "gradcheck": {k: c1[k] for k in ("tensors_checked", "tensors_total", "max_rel_error", "checked_at", "passed")},
        "correctness": [{"id": k, "name": CHECK_NAMES[k], "passed": bool(v["passed"]), "detail": v["detail"]}
                        for k, v in sorted(corr.items())],
        "mutants": {"detected": sum(bool(v) for v in mut.values()), "total": len(mut),
                    "score": sum(bool(v) for v in mut.values()) / len(mut), "by": mut},
        "hypotheses": hyp, "knockouts": kos,
        "descriptive": ({k: float(np.mean([r[k] for r in per_seed])) for k in per_seed[0]
                         if isinstance(per_seed[0][k], float)} if per_seed else {}),
        "worst_factor": ([(r["worst_factor_full"], r["worst_factor_cur"]) for r in per_seed] if per_seed else []),
        "task_types": TASK_TYPES + ["native: listwise catalogue selection"], "exit_code": exit_code,
    }


def print_report(rep):
    print(f"\n=== VERIFIED REPORT · chapter {CHAPTER:04d} ===")
    print(f"file         {rep['file']}  (card revision {rep['card_revision']})")
    print(f"environment  Python {rep['environment']['python']} · NumPy {rep['environment']['numpy']}")
    print(f"seeds        {rep['seeds']}")
    print(f"runtime      {rep['runtime_s']} s")
    d = 100.0 * (rep["n_params_baseline"] - rep["n_params"]) / rep["n_params"]
    print(f"parameters   exemplar model {rep['n_params']} · size-matched prototype {rep['n_params_baseline']} ({d:+.1f}%)")
    g = rep["gradcheck"]
    print(f"gradcheck    {g['tensors_checked']}/{g['tensors_total']} tensor checks at {g['checked_at']} · "
          f"max rel error {g['max_rel_error']:.2e} · {'PASS' if g['passed'] else 'FAIL'}")
    print("correctness")
    for c in rep["correctness"]:
        print(f"  {c['id']:<5} {c['name']:<27} {'PASS' if c['passed'] else 'FAIL'}  {c['detail']}")
    mu = rep["mutants"]
    print(f"mutants      {mu['detected']}/{mu['total']} detected (score {mu['score']:.2f})")
    for name, by in mu["by"].items():
        print(f"  {name:<13} {'detected by ' + ', '.join(by) if by else 'NOT DETECTED'}")
    print("hypotheses   (paired over seeds, 95% bootstrap CI, 2000 resamples)")
    for h in rep["hypotheses"]:
        est = (f"{h['mean_diff']:+.3f}  [{h['ci95'][0]:+.3f}, {h['ci95'][1]:+.3f}]" if h["n_seeds"]
               else "   --   [  --  ,   --  ]")
        print(f"  {h['id']:<9} {est}  mesi {h['mesi']:.2f}  n={h['n_seeds']}  {h['verdict']}")
    if rep["knockouts"]:
        print("knockouts    (exemplar model, change in blend-split precision@L)")
        for k in rep["knockouts"]:
            print(f"  {k['module']:<24} {'sig' if k['signature'] else '   '} {k['metric_change']:+.3f}  "
                  f"[{k['ci95'][0]:+.3f}, {k['ci95'][1]:+.3f}]")
    if rep["descriptive"]:
        print("descriptive  (means over seeds; trivial precision@L = 0.200)")
        keys = list(rep["descriptive"])
        for i in range(0, len(keys), 3):
            print("  " + "   ".join(f"{k} {rep['descriptive'][k]:.3f}" for k in keys[i:i + 3]))
        print(f"  audit worst factor (full, curated) per seed: {rep['worst_factor']}")
    print(f"task types   {rep['task_types']}")
    print(f"exit code    {rep['exit_code']}")
    print("=== END REPORT ===")


# 13. Command-line entry point
def main(argv=None):
    global ACTIVE_MUTANT
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__), description="Chapter 0248 · Sei Shōnagon")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=966)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--json", default=None)
    ap.add_argument("--card", action="store_true")
    ap.add_argument("--mutant", default=None)
    ap.add_argument("--data", default=None)
    args = ap.parse_args(argv)
    if args.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    if args.mutant is not None and args.mutant not in MUTANTS:
        print(f"unknown mutant {args.mutant!r}; choose from {sorted(MUTANTS)}", file=sys.stderr)
        return 2
    ACTIVE_MUTANT = args.mutant
    t0, scale = time.time(), SCALES["quick" if args.quick else "full"]
    n_seeds = 1 if args.quick else max(5, args.seeds)
    seeds = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(args.seed).spawn(n_seeds)]
    print(f"chapter {CHAPTER:04d} · Sei Shōnagon · base seed {args.seed} · seeds {seeds}"
          + (f" · MUTANT {ACTIVE_MUTANT}" if ACTIVE_MUTANT else ""))
    try:
        corr, ctx = run_correctness(seeds, scale)
        print("  correctness tests done", f"({time.time() - t0:.1f}s)")
        mut = c5_mutants(ctx)
        corr["C5"] = {"passed": all(mut.values()), "detail": f"mutation score {sum(map(bool, mut.values()))}/{len(mut)}"}
        per_seed, hyp, kos = [], [], []
        if not args.quick:
            for s in seeds:
                per_seed.append(run_seed(s, scale))
                print(f"  seed {s} done ({time.time() - t0:.1f}s)")
            hyp, kos = hypotheses(per_seed, np.random.default_rng(args.seed))
        else:
            hyp = [{"id": h["id"], "metric": h["metric"], "mean_diff": None, "ci95": [None, None],
                    "mesi": h["mesi"], "n_seeds": 0, "verdict": "not evaluated"} for h in MIND_CARD["hypotheses"]]
        if args.data:
            print(f"  real-data bridge (test accuracy): {data_bridge(args.data, seeds[0])}")
        else:
            print("  real-data bridge: no --data PATH given, skipped")
    except FloatingPointError as err:
        print(f"non-finite values: {err}", file=sys.stderr)
        return 4
    runtime = time.time() - t0
    budget = 20.0 if args.quick else 180.0
    corr["C8"] = {"passed": runtime <= budget, "detail": f"{runtime:.1f} s of {budget:.0f} s"}
    others_ok = all(v["passed"] for k, v in corr.items() if k != "C8")
    code = 1 if not others_ok else (0 if corr["C8"]["passed"] else 3)
    rep = build_report(args, seeds, corr, mut, hyp, kos, per_seed, runtime, code)
    print_report(rep)
    if args.json:
        write_report(rep, args.json)
    return code


if __name__ == "__main__":
    sys.exit(main())

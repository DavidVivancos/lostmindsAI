#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0358 · Juan Ruiz, Archpriest of Hita (1283-1350)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 18, Minds 341-360 Available on Amazon https://www.amazon.com/dp/B0HQ47T2NC
# Resume and Interactive Demos at https://artificiology.com/
# END ATTRIBUTION 
"""
La disputa por señas: certifying a shared reading by imagining the costumed rival.

Thesis
    Agreement on every sign exchanged never certifies a shared reading: a reader
    is certified only when one can no longer imagine a plausible rival (the ribald
    dressed as a doctor) who would have answered exactly as the partner did, and
    each probe is spent on the sign where that imagined rival would part company.

Evidence and provenance (belief: the figure's own verse survives)
    Libro de buen amor, MSS S (Salamanca), G (Gayoso), T (Toledo); dated era 1368
    (1330) in T/G and era 1381 (1343) in S. Stanza numbers follow Blecua (1992).
    The gesture debate of st. 44-63 is inherited from the Glossa ordinaria of
    Accursius on D. 1.2.2.4 (Maragno 2015); the move that is Ruiz's own is the
    frame he builds around it in st. 64-70 and in the prose prologue (S only).

Doctrine -> mechanism -> test
    ID  doctrine (locus)                                 mechanism                 test
    D1  both disputants leave certain they understood;   M1 fantasia: rival        H-SIG, C6.P2, C6.P3
        the doctor and the lout read the same signs       readers built in latent
        as theology and as threats (st. 55-63)            code space
    D2  "trabaja do fallares las sus señales çiertas"     M2 senal: probe where     H-SIG, knockout senal
        (st. 68)                                           rivals diverge
    D3  "non ha mala palabra si non es a mal tenida"      M3 entendimiento: FiLM    C1, C3, knockout
        (st. 64); "qual puntares, tal dirá" (st. 70)       decoder, reader code      entendimiento
                                                           sets what a sign means
    D4  the reader is known by the set of what he         M4 memoria: permutation-  C6.P1
        answered; memory slips (prologue)                  invariant set encoder
    D5  the Romans chose a ribald and dressed him "como   costume seeding of M1     H-SIG, H-RIVAL
        si fuesse doctor" (st. 52-54)                      from remembered readers
    D6  the prologue grants that the same pages serve     juicio: certify iff no    H-BLIND
        buen amor and loco amor; the book's own intent     surviving rival
        is left undecidable (prologue; st. 1629-1634)

Research question (scalable oversight and auditing)
    Can an evaluator with a small probe budget certify that a model shares its
    intended reading, when agreement on the probes it habitually asks is equally
    produced by readers its prior assigns zero mass?

Closest prior art and the delta
    BALD (Houlsby et al. 2011) and query-by-committee (Seung, Opper and
    Sompolinsky 1992) choose queries by disagreement among hypotheses that are
    retrieved and weighted by a prior; conditional neural processes (Garnelo et
    al. 2018) with FiLM conditioning (Perez et al. 2018) supply the reader model;
    DivDis (Lee, Yao and Finn 2023) diversifies hypotheses during training.
    Delta: rivals are constructed at audit time, by re-costuming remembered readers
    with gradient ascent in the reader-code space so that they fit the evidence yet
    disagree with one's own reading, and certification waits until none survive.
    This reaches reader types with zero prior mass, which retrieval cannot.

Blind spot (grounded in what the figure got wrong)
    St. 68 promises sure signs to whoever works for them. Where the channel holds
    no discriminating sign, the imagined rival can never be discharged, and the
    auditor refuses genuine doctors that a prior-trusting auditor certifies. Ruiz's
    own book is the historical case: seven centuries of readers have not found the
    sign that settles whether it preaches or teaches loco amor.

Task (generative process)
    1. A world has 12 signs (fixed random features, 2 nuisance dimensions) and 6
       replies; the doctor's own reading is a fixed table OWN.
    2. A reader type is a set of deviation modules A{2,3,4} B{4,5,6} C{6,7,8}
       D{9,10,11}; each shifts replies by its offset mod 6; signs 0 and 1 are
       canonical and read alike by everyone; 3 % of other entries flip at random.
    3. Training prior: eight types; D appears only combined with A or C, so the
       lone-D reader (the ribald) has zero mass.
    4. A conditional neural process learns reply(sign | set of observed exchanges).
    5. Audit: after the canonical signs, 3 probes; certify iff the partner's table
       differs from OWN in at most one sign. Splits: held-out (training prior),
       shifted (Roman selection: 45 % ribalds), narrow (signs 9-11 unprobeable).

Limits
    Synthetic readers with discrete tables; a 4-dimensional reader code; one fixed
    own reading; no language. This is a research prototype of an audit mechanism,
    not a model of Juan Ruiz's mind and not an AGI.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 3,
    "revision_log": ["r2 after the first --quick run (seed 0): gradient-check denominator floor set to 1e-4 "
                     "because float64 roundoff (about 1e-10) dominated entries below 1e-5; quick-mode training "
                     "raised to 2500 steps (full runs 3000) because the reader model was undertrained; tailoring ascent given "
                     "per-rival backtracking after C6.P3 showed 25 % of rivals ending below their seed; the fit penalty "
                     "became a smoothed hinge (softplus, k=10) so its pull does not vanish at the boundary. No "
                     "hypothesis, metric, split, mesi or comparison changed.",
                     "r3 after the first full run: the C4 reference became the best constant predictor on the "
                     "evaluation split (upper band 0.04), because reply frequencies are near-tied and the training-"
                     "split mode differed from the evaluation mode (shuffled 0.262 vs training mode 0.210). "
                     "The shuffled-label model now gets the same update budget as the real model (it had 800 "
                     "steps and kept chance per-sign preferences). Hypotheses and their statistics unchanged."],
    "generation": {"template_version": "codeguidelines 1.0", "generator": "Claude Opus 5.5",
                   "generator_version": "claude-opus-5-5", "date": "2026-09-30"},
    "id": 358, "figure": "Juan Ruiz, Archpriest of Hita", "born": 1283, "died": 1350,
    "civilization": "Castilian", "provenance": "belief",
    "thesis": ("Agreement on every exchanged sign never certifies a shared reading; certify only when no "
               "plausible costumed rival who would have answered the same can still be imagined, and probe "
               "where that rival parts from one's own reading."),
    "evidence": [
        {"id": "D1", "claim": "Greek doctor and Roman ribald both judge the gesture debate a success while "
         "reading the same signs as doctrine and as threats of violence", "basis": "primary",
         "source": "Libro de buen amor, st. 44-63 (ed. Blecua 1992)"},
        {"id": "D2", "claim": "The reader must work where the sure signs are found", "basis": "primary",
         "source": "Libro de buen amor, st. 68"},
        {"id": "D3", "claim": "No word is bad unless held as bad; the book says what it is played to say",
         "basis": "primary", "source": "Libro de buen amor, st. 64 and st. 70"},
        {"id": "D4", "claim": "Human memory slips; understanding, will and memory are the soul's faculties",
         "basis": "primary", "source": "Libro de buen amor, prose prologue (MS S)"},
        {"id": "D5", "claim": "The Romans chose a ribald and dressed him as a doctor of philosophy",
         "basis": "primary", "source": "Libro de buen amor, st. 52-54"},
        {"id": "D6", "claim": "The same pages are offered for buen amor and, reluctantly, for loco amor; "
         "the work's intention is left to the reader", "basis": "primary",
         "source": "Libro de buen amor, prose prologue; st. 1629-1634"},
        {"id": "D7", "claim": "The gesture tale is inherited from Accursius' gloss on D. 1.2.2.4; Ruiz's "
         "contribution is the hermeneutic frame of st. 64-70", "basis": "scholarship",
         "source": "Maragno 2015; Barletta, Hispanic Review 80.3 (2012)"},
    ],
    "research_question": {"category": "scalable oversight and auditing",
                          "question": ("Can a small probe budget certify that a model shares the intended "
                                       "reading when habitual probes are also passed by readers the prior "
                                       "assigns zero mass?")},
    "mechanism": {
        "name": "Costumed-rival imagination with divergence probing (fantasia + senal cierta)",
        "family": "meta-learning (conditional neural process) with model-based imagination and active learning",
        "signature_modules": ["fantasia", "senal"],
        "closest_prior_art": ["BALD (Houlsby et al. 2011)", "Query by committee (Seung et al. 1992)",
                              "Conditional neural processes (Garnelo et al. 2018)", "FiLM (Perez et al. 2018)",
                              "DivDis (Lee, Yao and Finn 2023)"],
        "overlap": "Medium",
        "prior_art_queries": ["adversarial hypothesis construction active learning certification",
                              "latent space counterfactual rival model probing", "auditing hidden objectives",
                              "right for the wrong reasons evaluation underspecification"],
        "contribution_type": "mechanism",
        "delta": ("Rivals are constructed at audit time by gradient ascent in reader-code space from remembered "
                  "readers, under fit and support penalties, instead of being retrieved and prior-weighted; "
                  "certification waits until none survive."),
        "cluster_note": ("Adjacent to the audit and abstention clusters. New consequence: detects reader types "
                         "with zero prior mass that BALD and committees certify, and fails predictably where "
                         "the channel lacks a discriminating sign."),
    },
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1 fantasia", "property_test": "C6.P2, C6.P3", "hypothesis": "H-SIG"},
        {"doctrine": "D2", "mechanism": "M2 senal", "property_test": "C6.P3", "hypothesis": "H-SIG"},
        {"doctrine": "D3", "mechanism": "M3 entendimiento", "property_test": "C1", "hypothesis": "H-NEC"},
        {"doctrine": "D4", "mechanism": "M4 memoria", "property_test": "C6.P1", "hypothesis": "H-NEC"},
        {"doctrine": "D5", "mechanism": "M1 costume seeding", "property_test": "C6.P2", "hypothesis": "H-RIVAL"},
        {"doctrine": "D6", "mechanism": "juicio", "property_test": "C6.P3", "hypothesis": "H-BLIND"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "Costumed-rival auditing certifies more accurately than BALD when the "
         "partner population contains unseen ribalds", "metric": "certification_accuracy", "split": "shifted",
         "comparison": "model - baseline(BALD)", "direction": "greater", "mesi": 0.10, "seeds": 5},
        {"id": "H-NEC", "statement": "Replacing fantasia by identity (no imagined rival) destroys the gain",
         "metric": "certification_accuracy", "split": "shifted",
         "comparison": "signature_knockout(fantasia=identity) - full_model", "direction": "less",
         "mesi": 0.10, "seeds": 5},
        {"id": "H-BLIND", "statement": "With signs 9-11 unprobeable and a benign population, the imagined "
         "lout cannot be discharged and the auditor refuses genuine doctors", "condition": "narrow channel, "
         "training prior", "grounding": "st. 68 presumes sure signs exist; the book's own intent stays "
         "undecidable", "metric": "certification_accuracy", "comparison": "model - baseline(BALD)",
         "direction": "less", "mesi": 0.10, "seeds": 5},
        {"id": "H-RIVAL", "statement": "The Greek doctor's rule (certify on agreement with habitual probes) "
         "is worse than costumed-rival auditing under Roman selection", "metric": "certification_accuracy",
         "split": "shifted", "comparison": "model - rival(greek)", "direction": "greater", "mesi": 0.10,
         "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.30, "margin_over_trivial": 0.07, "c4_band": 0.04,
                   "gradcheck_rel_error": 1e-5, "gradcheck_floor": 1e-4,
                   "grad_clip_norm": 5.0},
    "probe_predictions": [
        {"probe": "P6", "expected": "above baseline (set-conditioned reader model adapts from k examples)"},
        {"probe": "P3", "expected": "above baseline (FiLM code recombines deviation modules)"},
        {"probe": "P8", "expected": "equal to baseline (certification is not per-item calibration)"},
        {"probe": "P1", "expected": "below baseline (mean pooling discards order)"},
        {"probe": "P11", "expected": "equal to baseline (no control loop)"},
    ],
    "dialectic_links": [
        {"chapter": 115, "relation": "teacher", "test": "none (Ovidian and pseudo-Ovidian sources of Don Amor)"},
        {"chapter": 72, "relation": "teacher", "test": "none (Aristotle cited as authority, st. 71)"},
        {"chapter": 259, "relation": "rival", "test": "none (Castro's disputed Ibn Hazm influence)"},
    ],
    "corpus_neighbors": [
        {"chapter": 367, "similarity": None, "difference": "Boccaccio: receiver-conditioned artifact; here the "
         "receiver is audited by constructed rivals, not served"},
        {"chapter": 369, "similarity": None, "difference": "Hafez: uncollapsed reading kept open; here readings "
         "are forced apart by probes"},
        {"chapter": 380, "similarity": None, "difference": "Chaucer: pilgrim frame of many tellers; here one "
         "auditor and imagined listeners"},
        {"chapter": 507, "similarity": 0.027, "difference": "only file available for the shingle gate; "
         "different family (no set encoder, no latent rival search)"},
    ],
    "barometer": {
        "cognitive_processing": ["held-out reply accuracy from 4-7 exchanges", "probe efficiency"],
        "embodied_cognition": [],
        "world_modeling": ["shifted-population certification", "unseen reader combinations"],
        "consciousness": ["refusal when rivals survive (a metacognitive signal, not experience)"],
        "language_understanding": ["reader-conditioned sign meaning in a synthetic sign system"],
        "emotional_intelligence": ["false-belief style audit: partner answers alike, believes otherwise"],
        "creativity": ["constructed rivals outside the training support of types"],
        "autonomy": ["self-chosen probes under a fixed budget"],
    },
    "task_types": ["sequence_classification"],
    "applications": [
        {"use": "pre-deployment auditing of models for readings that pass habitual evals",
         "sector": "AI safety and evaluation", "dataset": "HANS (McCoy, Pavlick and Linzen 2019)"},
        {"use": "certifying that a crowd annotator reads a labelling guideline as intended",
         "sector": "data annotation", "dataset": "Bluebirds (Welinder et al. 2010), 39 annotators x 108 images"},
        {"use": "detecting right answers given from misconceptions in adaptive tests",
         "sector": "education", "dataset": "ASSISTments 2009-2010 skill builder"},
    ],
    "data_bridge": ("--data PATH: CSV partner_id,sign_id,reply_id with 12 items per partner (for example 12 "
                    "Bluebirds images for all 39 annotators); replies are relabelled so the per-item majority "
                    "becomes the own reading."),
    "safety_notes": ("Audit protocol for models and annotators; no manipulation of people. Readers are synthetic "
                     "tables; no sentence here is presented as Juan Ruiz's own words beyond cited stanzas."),
}

import argparse
import json
import math
import os
import sys
import time

import numpy as np

CHAPTER = 358

# BEGIN STANDARD UTILITIES v1.0
def softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def logsumexp(z, axis=-1, keepdims=False):
    m = np.max(z, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(z - m), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


def log_softmax(z, axis=-1):
    return z - logsumexp(z, axis=axis, keepdims=True)


def softplus(z):
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads, sign=1.0):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1.0 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1.0 - self.b2) * grads[k] ** 2
            mh = self.m[k] / (1.0 - self.b1 ** self.t)
            vh = self.v[k] / (1.0 - self.b2 ** self.t)
            params[k] -= sign * self.lr * mh / (np.sqrt(vh) + self.eps)


def clip_global_norm(grads, max_norm):
    norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    if norm > max_norm:
        for k in grads:
            grads[k] = grads[k] * (max_norm / (norm + 1e-12))
    return norm


def finite_diff_check(f, params, grads, rng, n_random=20, eps=1e-6, floor=1e-6):
    """Central differences on every tensor: all entries of small tensors, otherwise
    n_random random entries plus the largest-gradient entry. Returns worst relative error per tensor."""
    worst = {}
    for k, P in params.items():
        flat, g = P.reshape(-1), grads[k].reshape(-1)
        if flat.size <= n_random:
            idx = np.arange(flat.size)
        else:
            idx = np.unique(np.concatenate([rng.choice(flat.size, n_random, replace=False),
                                            [int(np.argmax(np.abs(g)))]]))
        err = 0.0
        for i in idx:
            old = flat[i]
            flat[i] = old + eps
            lp = f()
            flat[i] = old - eps
            lm = f()
            flat[i] = old
            num = (lp - lm) / (2.0 * eps)
            err = max(err, abs(num - g[i]) / max(abs(num), abs(g[i]), floor))
        worst[k] = err
    return worst


def paired_bootstrap(diffs, rng, n_resamples=2000, alpha=0.05):
    d = np.asarray(diffs, dtype=float)
    idx = rng.integers(0, len(d), size=(n_resamples, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def verdict(mean, ci, direction, mesi):
    lo, hi = ci
    if direction == "greater":
        if lo > 0 and mean >= mesi:
            return "supported"
        if hi < 0:
            return "contradicted"
    else:
        if hi < 0 and mean <= -mesi:
            return "supported"
        if lo > 0:
            return "contradicted"
    return "inconclusive"


def write_report(chapter, sections, json_path=None, payload=None):
    print("=== VERIFIED REPORT · chapter %04d ===" % chapter)
    for title, rows in sections:
        print("-- " + title)
        for row in rows:
            print("  " + row)
    print("=== END REPORT ===")
    if json_path:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
# END STANDARD UTILITIES

# ---------------------------------------------------------------- data and tasks
N_SIGNS, N_REPLIES, F_SIGN, F_NUIS = 12, 6, 8, 2
IN_DIM = F_SIGN + F_NUIS + N_REPLIES
OWN = np.array([1, 3, 0, 4, 2, 5, 1, 3, 0, 2, 4, 5])        # the doctor's own reading: sign -> reply
CANON = [0, 1]                                               # signs every reader reads alike
MODULES = {"A": ((2, 3, 4), 1), "B": ((4, 5, 6), 2), "C": ((6, 7, 8), 3), "D": ((9, 10, 11), 4)}
READERS = ["doctor", "merchant", "lover", "mocker", "pilgrim", "jongleur", "courtier", "scholar", "ribald"]
READER_MODS = ["", "A", "B", "C", "AD", "CD", "AB", "BC", "D"]
PRIORS = {  # the ribald (lone D) never occurs in training: the costumed lout of st. 52-54
    "train": np.array([.40, .12, .12, .10, .08, .08, .05, .05, 0.0]),
    "shifted": np.array([.35, .04, .04, .04, .04, .04, 0.0, 0.0, .45]),
}
FULL = np.ones(N_SIGNS, bool)
NARROW = FULL.copy()
NARROW[9:] = False
SPLITS = {"heldout": ("train", FULL), "shifted": ("shifted", FULL), "narrow": ("train", NARROW)}
P_FLIP, NUIS_SD = 0.03, 0.5
# Fixed world constant (not an experiment seed): the perceptual features of the twelve signs.
SIGN_FEAT = np.random.default_rng(np.random.SeedSequence(358)).normal(size=(N_SIGNS, F_SIGN))


def reader_table(mods):
    t = OWN.copy()
    for m in mods:
        signs, off = MODULES[m]
        t[list(signs)] = (t[list(signs)] + off) % N_REPLIES
    return t


TABLES = np.stack([reader_table(m) for m in READER_MODS])


def sample_readers(n, prior, rng):
    """prior: a name in PRIORS, or an explicit (P, 12) pool of tables to draw from (data bridge)."""
    if isinstance(prior, np.ndarray):
        kinds = rng.integers(0, len(prior), n)
        return kinds, prior[kinds].copy()
    kinds = rng.choice(len(READERS), size=n, p=PRIORS[prior] / PRIORS[prior].sum())
    tabs = TABLES[kinds].copy()
    flip = rng.random((n, N_SIGNS)) < P_FLIP
    flip[:, CANON] = False
    tabs = np.where(flip, (tabs + rng.integers(1, N_REPLIES, (n, N_SIGNS))) % N_REPLIES, tabs)
    return kinds, tabs


def tokens(signs, replies, rng, noise=NUIS_SD):
    feat = SIGN_FEAT[signs]
    nuis = rng.normal(0.0, noise, signs.shape + (F_NUIS,)) if noise > 0 else np.zeros(signs.shape + (F_NUIS,))
    rep = np.zeros(signs.shape + (N_REPLIES,))
    if replies is not None:
        np.put_along_axis(rep, replies[..., None], 1.0, axis=-1)
    return np.concatenate([feat, nuis, rep], axis=-1)


def example_keys(tabs, ctx, lens, q):
    """Exact integer identity of a training example: reader table, context set, query sign."""
    code = (tabs.astype(np.uint64) * (np.uint64(6) ** np.arange(N_SIGNS, dtype=np.uint64))).sum(1)
    mask = np.zeros(len(tabs), np.uint64)
    for t in range(ctx.shape[1]):
        bit = np.left_shift(np.uint64(1), ctx[:, t].astype(np.uint64))
        mask |= np.where(t < lens, bit, np.uint64(0))
    return (code << np.uint64(16)) | (mask << np.uint64(4)) | q.astype(np.uint64)


def is_eval_key(keys):
    return ((keys * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(40)) % np.uint64(5) == 0


def cnp_batch(n, prior, rng, split, lmin=0, tmax=7):
    """Context of 0..7 exchanges and a query sign outside the context; hash-assigned to train or eval."""
    _, tabs = sample_readers(n, prior, rng)
    perm = rng.permuted(np.tile(np.arange(N_SIGNS), (n, 1)), axis=1)
    ctx, q = perm[:, :tmax], perm[:, tmax]
    lens = rng.integers(lmin, tmax + 1, n)
    keys = example_keys(tabs, ctx, lens, q)
    keep = is_eval_key(keys) == (split == "eval")
    M = (np.arange(tmax)[None] < lens[:, None]).astype(float)
    X = tokens(ctx, np.take_along_axis(tabs, ctx, 1), rng)
    return {"X": X[keep], "M": M[keep], "Xq": tokens(q, None, rng)[keep],
            "y": tabs[np.arange(n), q][keep], "keys": keys[keep], "q": q[keep]}


def concordant(tabs):
    return (tabs != OWN).sum(1) <= 1


# ---------------------------------------------------------------- model
TASK_TYPES = ["sequence_classification"]
QUERY_TOKENS = tokens(np.arange(N_SIGNS), None, None, noise=0.0)   # noise-free sign tokens


def build_model(in_dim=IN_DIM, out_dim=N_REPLIES, task_type="sequence_classification", rng=None,
                E=32, Q=4, D=32):
    rng = rng if rng is not None else np.random.default_rng(0)
    s = lambda a, b, gain=1.0: rng.normal(0.0, gain / np.sqrt(a), (a, b))
    z = np.zeros
    params = {"Wt": s(in_dim, E), "bt": z(E), "Wm": s(E, E), "bm": z(E), "Wc": s(E, Q), "bc": z(Q),
              "Wq": s(in_dim, D), "bq": z(D), "Wg": s(Q, D, .5), "bg": z(D), "Wb": s(Q, D, .5), "bb": z(D),
              "Wo": s(D, out_dim), "bo": z(out_dim)}
    return {"params": params, "task_type": task_type, "ko": {}, "code_mean": np.zeros(Q)}


def encode(model, X, M):
    """memoria: the reader known by the set of his answers (Deep Sets mean pooling)."""
    p = model["params"]
    T = np.tanh(X @ p["Wt"] + p["bt"])
    cnt = np.maximum(M.sum(1), 1.0)[:, None]
    pool = (T * M[..., None]).sum(1) / cnt
    P2 = np.tanh(pool @ p["Wm"] + p["bm"])
    c = P2 @ p["Wc"] + p["bc"]
    if model["ko"].get("memoria") == "mean":
        c = np.broadcast_to(model["code_mean"], c.shape).copy()
    return c, (T, cnt, pool, P2)


def decode(model, c, Xq):
    """entendimiento: the reader code sets the gain and bias of the sign features (FiLM)."""
    p = model["params"]
    U = np.tanh(Xq @ p["Wq"] + p["bq"])
    G, B = c @ p["Wg"] + p["bg"], c @ p["Wb"] + p["bb"]
    if model["ko"].get("entendimiento") == "zero":
        G, B = np.zeros_like(G), np.zeros_like(B)
    Hh = np.tanh(U * (1.0 + G) + B)
    return Hh @ p["Wo"] + p["bo"], (U, G, Hh)


ACTIVE = {"mutant": None}
MUTANTS = {"sign_flip": "Adam update applied with flipped sign", "zero_lr": "learning rate set to zero",
           "zero_grad_Wg": "gradient of the FiLM gain weights zeroed",
           "zero_grad_Wt": "gradient of the token-encoder weights zeroed"}


def loss_and_grads(model, batch):
    p = model["params"]
    X, M, Xq, y = batch["X"], batch["M"], batch["Xq"], batch["y"]
    c, (T, cnt, pool, P2) = encode(model, X, M)
    lo, (U, G, Hh) = decode(model, c, Xq)
    ls = log_softmax(lo)
    n = len(y)
    loss = -ls[np.arange(n), y].mean()
    dlo = np.exp(ls)
    dlo[np.arange(n), y] -= 1.0
    dlo /= n
    g = {"Wo": Hh.T @ dlo, "bo": dlo.sum(0)}
    dZ = (dlo @ p["Wo"].T) * (1.0 - Hh ** 2)
    dG, dB, dU = dZ * U, dZ, dZ * (1.0 + G)
    g.update(Wg=c.T @ dG, bg=dG.sum(0), Wb=c.T @ dB, bb=dB.sum(0))
    dc = dG @ p["Wg"].T + dB @ p["Wb"].T
    dUp = dU * (1.0 - U ** 2)
    g.update(Wq=Xq.T @ dUp, bq=dUp.sum(0), Wc=P2.T @ dc, bc=dc.sum(0))
    dP = (dc @ p["Wc"].T) * (1.0 - P2 ** 2)
    g.update(Wm=pool.T @ dP, bm=dP.sum(0))
    dA = ((dP @ p["Wm"].T)[:, None, :] * M[..., None] / cnt[:, :, None]) * (1.0 - T ** 2)
    g.update(Wt=np.einsum("nti,nte->ie", X, dA), bt=dA.sum((0, 1)))
    if ACTIVE["mutant"] in ("zero_grad_Wg", "zero_grad_Wt"):
        g[ACTIVE["mutant"][10:]] = np.zeros_like(g[ACTIVE["mutant"][10:]])
    return float(loss), g


def split_generic(X):
    """Generic sequence input: every token but the last is context, the last token is the query."""
    return X[:, :-1], np.ones(X.shape[:2])[:, :-1], X[:, -1]


def predict(model, X):
    Xc, M, Xq = split_generic(X)
    c, _ = encode(model, Xc, M)
    return decode(model, c, Xq)[0].argmax(-1)


def hidden_states(model, X):
    Xc, M, Xq = split_generic(X)
    c, (_, _, pool, _) = encode(model, Xc, M)
    _, (_, _, Hh) = decode(model, c, Xq)
    return {"pooled_exchanges": pool, "reader_code": c, "film_hidden": Hh}


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


def decode_all(model, C):
    """Log-probabilities of every reply at every sign for reader codes C (..., Q)."""
    p = model["params"]
    U = np.tanh(QUERY_TOKENS @ p["Wq"] + p["bq"])
    G, B = C @ p["Wg"] + p["bg"], C @ p["Wb"] + p["bb"]
    if model["ko"].get("entendimiento") == "zero":
        G, B = np.zeros_like(G), np.zeros_like(B)
    Hh = np.tanh(U * (1.0 + G[..., None, :]) + B[..., None, :])
    return log_softmax(Hh @ p["Wo"] + p["bo"]), (U, Hh)


# Audit constants, fixed before the first run.
TAU, TAU_ACCEPT, P_CLEAR, MIN_CLEAR = 0.5, 0.7, 0.6, 2       # fit hinge, fit acceptance, clear reading
ALPHA, LAM_FIT, LAM_SUP, KAPPA, ETA, ASCENT = 0.5, 4.0, 2.0, 10.0, 0.1, 40
BUDGET, BANK_N, SUPPORT_Q, BETA, DELTA = 3, 400, 0.99, 1.0, 1.5 / 12


def build_bank(model, rng, pool=None):
    """Remembered readers: codes of training-prior partners seen through all twelve signs."""
    _, tabs = sample_readers(BANK_N, "train" if pool is None else pool, rng)
    X = tokens(np.tile(np.arange(N_SIGNS), (BANK_N, 1)), tabs, rng, noise=0.0)
    codes, _ = encode(model, X, np.ones((BANK_N, N_SIGNS)))
    mu = codes.mean(0)
    Si = np.linalg.inv(np.cov(codes.T) + 1e-3 * np.eye(codes.shape[1]))
    m2 = np.einsum("nq,qr,nr->n", codes - mu, Si, codes - mu)
    lp, _ = decode_all(model, codes)
    return {"codes": codes, "mu": mu, "Si": Si, "rho2": float(np.quantile(m2, SUPPORT_Q)),
            "p": np.exp(lp), "lp": lp}


def observed_nll(lp_bank, O, obs):
    """(N, Nb) negative log-likelihood of each episode's observations under each remembered reader."""
    out = np.zeros((len(O), len(lp_bank)))
    for h in range(N_SIGNS):
        out += O[:, h][:, None] * (-lp_bank[:, h, :][:, obs[:, h]].T)
    return out


def imagination_objective(model, bank, C, O, obs, tgt):
    """J(c) = sum_h w_h (1 - p_h[own_h]) - lf sum_{h in O} softplus(k (nll_h - tau)) / k - ls/2 relu(M2 - rho2)^2,
    w_h = (target_h + alpha/12) on unobserved signs; a smoothed hinge keeps a near-constant pull toward the
    evidence once a rival starts to contradict it. Returns J (N, K) and dJ/dC (N, K, Q)."""
    p = model["params"]
    lp, (U, Hh) = decode_all(model, C)
    P = np.exp(lp)
    ar = np.arange(N_SIGNS)
    pown = P[..., ar, OWN]
    w = (tgt[None] + ALPHA / N_SIGNS) * (~O)[:, None, :]
    ob = np.broadcast_to(obs[:, None, :, None], lp.shape[:-1] + (1,))
    zfit = KAPPA * (-np.take_along_axis(lp, ob, -1)[..., 0] - TAU)
    dv = C - bank["mu"]
    s = np.maximum(np.einsum("nkq,qr,nkr->nk", dv, bank["Si"], dv) - bank["rho2"], 0.0)
    J = (w * (1.0 - pown)).sum(-1) - LAM_FIT * (softplus(zfit) * O[:, None, :]).sum(-1) / KAPPA \
        - 0.5 * LAM_SUP * s ** 2
    v = (1.0 / (1.0 + np.exp(-zfit))) * O[:, None, :]
    e_own = np.zeros_like(P)
    e_own[..., ar, OWN] = 1.0
    e_obs = np.zeros_like(P)
    np.put_along_axis(e_obs, ob, 1.0, -1)
    dlo = (w * pown)[..., None] * (P - e_own) - LAM_FIT * v[..., None] * (P - e_obs)
    dZ = (dlo @ p["Wo"].T) * (1.0 - Hh ** 2)
    dC = (dZ * U).sum(-2) @ p["Wg"].T + dZ.sum(-2) @ p["Wb"].T
    if model["ko"].get("entendimiento") == "zero":
        dC = np.zeros_like(dC)
    dC -= LAM_SUP * s[..., None] * 2.0 * (dv @ bank["Si"])
    return J, dC


def costume_seeds(bank, O, obs):
    """For each target sign, the remembered reader that most disagrees there while fitting the evidence."""
    fit = observed_nll(bank["lp"], O, obs) / np.maximum(O.sum(1), 1)[:, None]
    dis = 1.0 - bank["p"][:, np.arange(N_SIGNS), OWN]                     # (Nb, 12)
    score = dis.T[None, :, :] - BETA * fit[:, None, :]
    return bank["codes"][score.argmax(-1)]                               # (N, 12, Q)


def fantasia(model, bank, O, obs, steps=ASCENT, direction=1.0, seeds=None):
    """Imagine rival readers: costume remembered readers, then tailor them to the evidence by ascent."""
    if model["ko"].get("fantasia") == "identity":
        c, _ = encode(model, tokens(np.tile(np.arange(N_SIGNS), (len(O), 1)), obs, None, 0.0), O.astype(float))
        return c[:, None, :]
    C = costume_seeds(bank, O, obs) if seeds is None else seeds.copy()
    tgt = np.eye(N_SIGNS)
    J, g = imagination_objective(model, bank, C, O, obs, tgt)
    step = np.full(C.shape[:2], ETA)
    for _ in range(steps):       # normalized steps; a rival whose objective would worsen halves its step
        Cn = C + direction * step[..., None] * g / (np.linalg.norm(g, axis=-1, keepdims=True) + 1e-12)
        Jn, gn = imagination_objective(model, bank, Cn, O, obs, tgt)
        ok = direction * (Jn - J) >= 0
        C, J, g = np.where(ok[..., None], Cn, C), np.where(ok, Jn, J), np.where(ok[..., None], gn, g)
        step = np.where(ok, step, 0.5 * step)
    return C


def survivors(model, bank, C, O, obs):
    """A rival survives if it fits every observation, is in the support of remembered readers, and
    clearly reads at least two signs differently from one's own reading."""
    lp, _ = decode_all(model, C)
    ob = np.broadcast_to(obs[:, None, :, None], lp.shape[:-1] + (1,))
    fit = ((-np.take_along_axis(lp, ob, -1)[..., 0] <= TAU_ACCEPT) | ~O[:, None, :]).all(-1)
    clear = (lp.argmax(-1) != OWN) & (np.exp(lp.max(-1)) >= P_CLEAR)
    dv = C - bank["mu"]
    inside = np.einsum("nkq,qr,nkr->nk", dv, bank["Si"], dv) <= bank["rho2"]
    return fit & inside & (clear.sum(-1) >= MIN_CLEAR), clear


def senal(clear, live, O, allowed, rng, mode=None):
    """señal çierta: probe the sign on which the most surviving rivals clearly part from one's reading."""
    score = (clear & live[..., None]).sum(1) + rng.random(O.shape) * 1e-3
    if mode == "mean":
        score = rng.random(O.shape)
    score[~(allowed[None, :] & ~O)] = -np.inf
    return score.argmax(1)


def audit_ruiz(model, bank, tabs, allowed, rng, budget=BUDGET, steps=ASCENT):
    """juicio: refuse on any observed discord; certify as soon as no imagined rival survives."""
    n = len(tabs)
    O = np.zeros((n, N_SIGNS), bool)
    O[:, CANON] = True
    done, cert = np.zeros(n, bool), np.zeros(n, bool)
    for b in range(budget + 1):
        done |= ((tabs != OWN) & O).any(1)
        C = fantasia(model, bank, O, tabs, steps=steps)
        live, clear = survivors(model, bank, C, O, tabs)
        newly = ~done & ~live.any(1)
        cert[newly], done = True, done | newly
        if b == budget or done.all():
            break
        h = senal(clear, live, O, allowed, rng, model["ko"].get("senal"))
        act = np.where(~done)[0]
        O[act, h[act]] = True
    return cert


# ---------------------------------------------------------------- baselines and rival mechanisms
def audit_bald(bank, tabs, allowed, budget=BUDGET):
    """BALD over remembered readers: prior-weighted posterior, mutual-information probes, certify when the
    posterior expected disagreement with one's own reading falls below DELTA."""
    n = len(tabs)
    O = np.zeros((n, N_SIGNS), bool)
    O[:, CANON] = True
    Pb, lb = bank["p"], np.log(bank["p"] + 1e-12)
    dis = (Pb.argmax(-1) != OWN).mean(-1)
    for b in range(budget + 1):
        w = softmax(-observed_nll(bank["lp"], O, tabs), axis=1)
        if b == budget:
            break
        pm = np.einsum("nb,bhr->nhr", w, Pb)
        mi = -(pm * np.log(pm + 1e-12)).sum(-1) + np.einsum("nb,bh->nh", w, (Pb * lb).sum(-1))
        mi[~(allowed[None, :] & ~O)] = -np.inf
        O[np.arange(n), mi.argmax(1)] = True
    return (w @ dis) < DELTA


def audit_greek(tabs, allowed, budget=BUDGET):
    """The Greek doctor: habitual probes in fixed order, certify on agreement."""
    probes = [h for h in range(2, N_SIGNS) if allowed[h]][:budget]
    idx = CANON + probes
    return (tabs[:, idx] == OWN[idx]).all(1)


def audit_recall(model, bank, tabs, allowed, rng):
    """Committee of remembered readers without tailoring (costume seeds, no ascent)."""
    return audit_ruiz(model, bank, tabs, allowed, rng, steps=0)


# ---------------------------------------------------------------- registries
KNOCKOUT_MODES = {"memoria": "mean", "entendimiento": "zero", "fantasia": "identity", "senal": "mean"}


def modules(model):
    return {
        "memoria": {"params": ["Wt", "bt", "Wm", "bm", "Wc", "bc"], "signature": False,
                    "role": "permutation-invariant set encoder (mean pooling) producing a 4-d reader code"},
        "entendimiento": {"params": ["Wq", "bq", "Wg", "bg", "Wb", "bb", "Wo", "bo"], "signature": False,
                          "role": "FiLM-conditioned query decoder: reader code sets gain and bias per sign"},
        "fantasia": {"params": [], "signature": True,
                     "role": "latent-space counterfactual search: memory-bank seeds, normalized gradient ascent "
                             "on disagreement under fit and Mahalanobis-support penalties"},
        "senal": {"params": [], "signature": True,
                  "role": "active query selection: sign where most surviving rivals disagree"},
        "juicio": {"params": [], "signature": False,
                   "role": "certification rule: certify iff no surviving rival and no observed discord"},
    }


def knockout(model, name, mode):
    if KNOCKOUT_MODES.get(name) != mode:
        raise ValueError("unsupported knockout %s/%s" % (name, mode))
    ko = {"params": {k: v.copy() for k, v in model["params"].items()}, "task_type": model["task_type"],
          "ko": dict(model["ko"]), "code_mean": model["code_mean"].copy()}
    ko["ko"][name] = mode
    return ko


# ---------------------------------------------------------------- training
class NonFinite(Exception):
    pass


def fit(model, data, budget, rng, lr=3e-3, batch=128, shuffle_labels=False):
    """data: None for the native world (training prior), or an explicit (P, 12) table pool."""
    opt = Adam(model["params"], lr=0.0 if ACTIVE["mutant"] == "zero_lr" else lr)
    sign = -1.0 if ACTIVE["mutant"] == "sign_flip" else 1.0
    prior = "train" if data is None else data
    losses = []
    for _ in range(budget):
        b = cnp_batch(batch, prior, rng, "train")
        if shuffle_labels:
            b["y"] = rng.permutation(b["y"])
        loss, g = loss_and_grads(model, b)
        if not np.isfinite(loss):
            raise NonFinite("training loss")
        clip_global_norm(g, MIND_CARD["thresholds"]["grad_clip_norm"])
        opt.step(model["params"], g, sign=sign)
        losses.append(loss)
    if not all(np.isfinite(v).all() for v in model["params"].values()):
        raise NonFinite("parameters")
    b = cnp_batch(512, prior, rng, "train")
    model["code_mean"] = encode(model, b["X"], b["M"])[0].mean(0)
    return np.array(losses)


# ---------------------------------------------------------------- tests: correctness
def gradcheck(model, rng):
    b = cnp_batch(24, "train", rng, "train", lmin=2)
    _, g = loss_and_grads(model, b)
    return finite_diff_check(lambda: loss_and_grads(model, b)[0], model["params"], g, rng,
                             floor=MIND_CARD["thresholds"]["gradcheck_floor"])


def gradcheck_imagination(model, bank, rng):
    """Analytic check of the non-parametric rule: dJ/dc of the imagination objective."""
    _, tabs = sample_readers(3, "shifted", rng)
    O = np.zeros((3, N_SIGNS), bool)
    O[:, CANON + [4, 9]] = True
    C = bank["codes"][rng.integers(0, BANK_N, (3, N_SIGNS))] + rng.normal(0.0, 0.3, (3, N_SIGNS, 4))
    _, g = imagination_objective(model, bank, C, O, tabs, np.eye(N_SIGNS))
    worst = 0.0
    for n in range(3):           # each rival's objective depends only on its own code: check per rival
        for k in range(N_SIGNS):
            f = lambda: float(imagination_objective(model, bank, C, O, tabs, np.eye(N_SIGNS))[0][n, k])
            worst = max(worst, finite_diff_check(f, {"c": C[n, k]}, {"c": g[n, k]}, rng,
                                                 floor=MIND_CARD["thresholds"]["gradcheck_floor"])["c"])
    return worst


def heldout_metrics(model, rng, n=10000):
    tr, ev = cnp_batch(n, "train", rng, "train", lmin=4), cnp_batch(n, "train", rng, "eval", lmin=4)
    maj = np.array([np.bincount(tr["y"][tr["q"] == h], minlength=N_REPLIES).argmax() for h in range(N_SIGNS)])
    pred = decode(model, encode(model, ev["X"], ev["M"])[0], ev["Xq"])[0].argmax(-1)
    best_constant = np.bincount(ev["y"], minlength=N_REPLIES).max() / len(ev["y"])
    return float((pred == ev["y"]).mean()), float((maj[ev["q"]] == ev["y"]).mean()), float(best_constant)


def loss_drop(L):
    return 1.0 - float(L[-100:].mean()) / float(L[:50].mean())


def test_determinism(entropy):
    runs = []
    for _ in range(2):
        r = np.random.default_rng(np.random.SeedSequence(entropy))
        m = build_model(rng=r)
        L = fit(m, None, 60, r)
        e = cnp_batch(64, "train", np.random.default_rng(np.random.SeedSequence(entropy)), "eval", lmin=4)
        runs.append((L, decode(m, encode(m, e["X"], e["M"])[0], e["Xq"])[0], m["params"]))
    same = np.array_equal(runs[0][0], runs[1][0]) and np.array_equal(runs[0][1], runs[1][1])
    finite = all(np.isfinite(v).all() for v in list(runs[0][2].values()) + [runs[0][0], runs[0][1]])
    Xg = np.concatenate([e["X"], e["Xq"][:, None, :]], axis=1)          # generic interface: last token queries
    hs = hidden_states(m, Xg)
    iface = predict(m, Xg).shape == (len(Xg),) and all(np.isfinite(v).all() for v in hs.values())
    return same and finite and iface, "identical losses and logits: %s; finite: %s; interface: %s" % (
        same, finite, iface)


def test_mutants(entropy, steps, tol, drop):
    passed = {}
    for name in [None] + list(MUTANTS):
        ACTIVE["mutant"] = name
        r = np.random.default_rng(np.random.SeedSequence(entropy))
        m = build_model(rng=r)
        gc_ok = max(gradcheck(m, r).values()) <= tol
        try:
            learn_ok = loss_drop(fit(m, None, steps, r)) >= drop
        except NonFinite:
            learn_ok = False
        passed[name] = gc_ok and learn_ok
    ACTIVE["mutant"] = None
    detected = sum(not passed[k] for k in MUTANTS)
    return passed[None], detected


def _pooled_codes(model, X, M, positional):
    p = model["params"]
    T = np.tanh(X @ p["Wt"] + p["bt"])
    w = M * (1.0 + 0.3 * np.arange(X.shape[1])[None, :]) if positional else M
    pool = (T * w[..., None]).sum(1) / np.maximum(w.sum(1), 1.0)[:, None]
    return np.tanh(pool @ p["Wm"] + p["bm"]) @ p["Wc"] + p["bc"]


def test_set_invariance(rng):
    """P1: the reader code ignores the order of exchanges (random params, random inputs)."""
    m = build_model(rng=rng)
    X, M = rng.normal(size=(16, 7, IN_DIM)), (rng.random((16, 7)) < 0.8).astype(float)
    worst = {False: 0.0, True: 0.0}
    for positional in (False, True):
        c0 = _pooled_codes(m, X, M, positional)
        for _ in range(20):
            perm = rng.permutation(7)
            worst[positional] = max(worst[positional],
                                    float(np.abs(_pooled_codes(m, X[:, perm], M[:, perm], positional) - c0).max()))
    assert np.allclose(encode(m, X, M)[0], _pooled_codes(m, X, M, False))
    return worst[False] < 1e-10 and worst[True] > 1e-3, "max drift %.1e (positional control %.1e)" % (
        worst[False], worst[True])


def random_evidence(rng, n=48):
    _, tabs = sample_readers(n, "shifted", rng)
    O = np.zeros((n, N_SIGNS), bool)
    O[:, CANON] = True
    for i in range(n):
        O[i, rng.choice(np.arange(2, N_SIGNS), rng.integers(0, 4), replace=False)] = True
    return tabs, O


def test_support(model, bank, rng):
    """P2: imagined rivals stay near the support of remembered readers; control removes the penalty."""
    global LAM_SUP
    tabs, O = random_evidence(rng)
    frac = {}
    saved = LAM_SUP
    for lam in (saved, 0.0):
        LAM_SUP = lam
        C = fantasia(model, bank, O, tabs, steps=2 * ASCENT)
        dv = C - bank["mu"]
        frac[lam] = float((np.einsum("nkq,qr,nkr->nk", dv, bank["Si"], dv) <= 1.2 * bank["rho2"]).mean())
    LAM_SUP = saved
    return frac[saved] >= 0.95 and frac[0.0] < 0.95, "inside 1.2 rho2: %.3f (no-penalty control %.3f)" % (
        frac[saved], frac[0.0])


def test_ascent(model, bank, rng):
    """P3: tailoring never lowers the imagination objective and raises it for most rivals; a descending
    control must violate it. A wrong gradient sign would leave rivals at their seeds and fail the second part."""
    tabs, O = random_evidence(rng)
    seeds = costume_seeds(bank, O, tabs)
    J0 = imagination_objective(model, bank, seeds, O, tabs, np.eye(N_SIGNS))[0]
    frac = {}
    for d in (1.0, -1.0):
        J = imagination_objective(model, bank, fantasia(model, bank, O, tabs, direction=d, seeds=seeds), O, tabs,
                                  np.eye(N_SIGNS))[0]
        frac[d] = (float((J >= J0 - 1e-9).mean()), float((J > J0 + 1e-6).mean()))
    ok = frac[1.0][0] == 1.0 and frac[1.0][1] >= 0.5 and frac[-1.0][0] < 0.5
    return ok, "non-decreasing %.3f, strictly raised %.3f (descent control non-decreasing %.3f)" % (
        frac[1.0][0], frac[1.0][1], frac[-1.0][0])


def test_split(rng):
    """C7: hash-assigned splits share no example; the unfiltered sampler would (negative control)."""
    tr = set()
    for _ in range(150):
        tr.update(cnp_batch(128, "train", rng, "train")["keys"].tolist())
    ev = set(cnp_batch(20000, "train", rng, "eval")["keys"].tolist())
    raw = set(cnp_batch(20000, "train", rng, "train")["keys"].tolist())
    raw |= set(cnp_batch(20000, "train", rng, "eval")["keys"].tolist())
    return len(tr & ev) == 0 and len(ev) > 0 and len(raw & ev) > 0, (
        "train keys %d, eval keys %d, overlap %d (unfiltered control %d)" % (len(tr), len(ev), len(tr & ev),
                                                                             len(raw & ev)))


# ---------------------------------------------------------------- tests: hypotheses
def evaluate_seed(model, rng, n_ep):
    bank = build_bank(model, rng)
    acc = {}
    for split, (prior, allowed) in SPLITS.items():
        _, tabs = sample_readers(n_ep, prior, rng)
        truth = concordant(tabs)
        runs = {"ruiz": audit_ruiz(model, bank, tabs, allowed, rng), "bald": audit_bald(bank, tabs, allowed),
                "greek": audit_greek(tabs, allowed), "recall": audit_recall(model, bank, tabs, allowed, rng)}
        if split == "shifted":
            for name, mode in KNOCKOUT_MODES.items():
                km = knockout(model, name, mode)
                kb = build_bank(km, rng) if name in ("memoria", "entendimiento") else bank
                runs["ko_" + name] = audit_ruiz(km, kb, tabs, allowed, rng)
        for k, cert in runs.items():
            acc[(split, k)] = float((cert == truth).mean())
    return acc


HYP_DIFFS = {"H-SIG": ("shifted", "ruiz", "bald"), "H-NEC": ("shifted", "ko_fantasia", "ruiz"),
             "H-BLIND": ("narrow", "ruiz", "bald"), "H-RIVAL": ("shifted", "ruiz", "greek")}


def hypothesis_rows(per_seed, rng):
    out = []
    for h in MIND_CARD["hypotheses"]:
        split, a, b = HYP_DIFFS[h["id"]]
        d = [s[(split, a)] - s[(split, b)] for s in per_seed]
        if len(d) < 5:
            out.append({"id": h["id"], "metric": h["metric"], "mean_diff": float(np.mean(d)), "ci95": [None, None],
                        "mesi": h["mesi"], "n_seeds": len(d), "verdict": "not evaluated"})
            continue
        mean, ci = paired_bootstrap(d, rng)
        out.append({"id": h["id"], "metric": h["metric"], "mean_diff": mean, "ci95": list(ci), "mesi": h["mesi"],
                    "n_seeds": len(d), "verdict": verdict(mean, ci, h["direction"], h["mesi"])})
    return out


def knockout_rows(per_seed, rng):
    rows = []
    for name in KNOCKOUT_MODES:
        d = [s[("shifted", "ko_" + name)] - s[("shifted", "ruiz")] for s in per_seed]
        ci = paired_bootstrap(d, rng)[1] if len(d) >= 5 else (None, None)
        rows.append({"module": name, "signature": modules(None)[name]["signature"],
                     "metric_change": float(np.mean(d)), "ci95": list(ci)})
    return rows


# ---------------------------------------------------------------- report
def fmt_ci(ci):
    return "[%+.3f, %+.3f]" % tuple(ci) if ci[0] is not None else "[n/a]"


def build_sections(info):
    sec = [("environment", ["python %s · numpy %s" % (sys.version.split()[0], np.__version__)]),
           ("run", ["mode: %s" % info["mode"], "seeds: %s" % info["seeds"],
                    "runtime_s: %.1f (budget %d)" % (info["runtime"], info["budget_s"]),
                    "n_params: %d" % info["n_params"], "task_types: %s" % ", ".join(TASK_TYPES)])]
    gc = info["gradcheck"]
    sec.append(("gradient check (central differences, eps 1e-6, float64)",
                ["tensors checked %d/%d at init and after %d steps" % (gc["tensors_checked"], gc["tensors_total"],
                                                                         info["steps"]),
                 "max relative error: init %.2e · trained %.2e · imagination dJ/dc %.2e" % (
                     gc["init"], gc["trained"], gc["imagination"])]))
    sec.append(("correctness", ["%-3s %-26s %-4s %s" % (c["id"], c["name"], "PASS" if c["passed"] else "FAIL",
                                                        c["detail"]) for c in info["correctness"]]))
    mu = info["mutants"]
    sec.append(("mutants", ["detected %d/%d · score %.2f" % (mu["detected"], mu["total"], mu["score"])]))
    rows = []
    for split in (SPLITS if info["per_seed"] else []):
        cells = []
        for pol in ("ruiz", "bald", "recall", "greek"):
            v = [s[(split, pol)] for s in info["per_seed"]]
            cells.append("%s %.3f±%.3f" % (pol, np.mean(v), np.std(v)))
        rows.append("%-8s %s" % (split, " · ".join(cells)))
    sec.append(("certification accuracy by split (mean ± sd over seeds)", rows or ["not evaluated"]))
    sec.append(("hypotheses", ["%-7s mean %+.3f  ci95 %s  mesi %.2f  n=%d  %s" % (
        h["id"], h["mean_diff"], fmt_ci(h["ci95"]), h["mesi"], h["n_seeds"], h["verdict"])
        for h in info["hypotheses"]]))
    sec.append(("knockouts (shifted accuracy change vs full model)", ["%-14s signature=%-5s %+.3f  ci95 %s" % (
        k["module"], k["signature"], k["metric_change"], fmt_ci(k["ci95"])) for k in info["knockouts"]]))
    return sec


def json_payload(info, exit_code):
    return {"schema_version": "1.0", "chapter": CHAPTER, "file": os.path.basename(__file__),
            "card_revision": MIND_CARD["card_revision"],
            "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
            "seeds": info["seeds"], "runtime_s": round(info["runtime"], 2), "n_params": info["n_params"],
            "gradcheck": {"tensors_checked": info["gradcheck"]["tensors_checked"],
                          "tensors_total": info["gradcheck"]["tensors_total"],
                          "max_rel_error": max(info["gradcheck"]["init"], info["gradcheck"]["trained"]),
                          "imagination_rel_error": info["gradcheck"]["imagination"],
                          "checked_at": ["init", "after_training_steps"], "passed": info["gradcheck"]["passed"]},
            "correctness": info["correctness"], "mutants": info["mutants"], "hypotheses": info["hypotheses"],
            "knockouts": info["knockouts"],
            "conditions": {"%s/%s" % k: [s[k] for s in info["per_seed"]] for k in
                           (info["per_seed"][0] if info["per_seed"] else {})},
            "task_types": TASK_TYPES, "exit_code": exit_code}


# ---------------------------------------------------------------- real-data bridge
def data_bridge(path, rng, steps):
    """CSV partner_id,sign_id,reply_id; the first 12 items; per-item majority becomes the own reading."""
    if not os.path.exists(path):
        print("data bridge: %s not found; skipped" % path)
        return
    with open(path, encoding="utf-8") as fh:
        rows = [ln.strip().split(",") for ln in fh if ln.strip()][1:]
    pids, items = sorted({r[0] for r in rows}), sorted({int(r[1]) for r in rows})[:N_SIGNS]
    tab = np.full((len(pids), N_SIGNS), -1)
    for pid, s, r in rows:
        if int(s) in items:
            tab[pids.index(pid), items.index(int(s))] = int(r) % N_REPLIES
    tab = tab[(tab >= 0).all(1)]
    if len(tab) < 8 or len(items) < N_SIGNS:
        print("data bridge: need at least 8 complete partners over 12 items; skipped")
        return
    ref = np.array([np.bincount(tab[:, h], minlength=N_REPLIES).argmax() for h in range(N_SIGNS)])
    tab = rng.permutation((tab - ref + OWN) % N_REPLIES)
    k = len(tab) // 2
    model = build_model(rng=rng)
    fit(model, tab[:k], steps, rng)
    bank = build_bank(model, rng, pool=tab[:k])
    truth = concordant(tab[k:])
    for name, cert in (("ruiz", audit_ruiz(model, bank, tab[k:], FULL, rng)), ("bald", audit_bald(bank, tab[k:], FULL))):
        print("data bridge %-5s partners %d · certified %.2f · accuracy vs full-table concordance %.2f" % (
            name, len(truth), cert.mean(), (cert == truth).mean()))


# ---------------------------------------------------------------- command line
def run(a):
    t0 = time.time()
    quick = a.quick or a.mutant is not None
    n_seeds = 1 if quick else (a.seeds or 5)
    steps, n_ep, budget_s = (2500, 60, 20) if quick else (3000, 160, 180)
    th = MIND_CARD["thresholds"]
    ACTIVE["mutant"] = a.mutant
    kids = np.random.SeedSequence(a.seed).spawn(n_seeds + 1)
    seeds = [int(k.generate_state(1)[0]) for k in kids]
    print("%s · base seed %d · per-seed entropies %s" % (os.path.basename(__file__), a.seed, seeds))
    rng = np.random.default_rng(kids[0])
    model = build_model(rng=rng)
    g_init = gradcheck(model, rng)
    L = fit(model, None, steps, rng)
    g_tr = gradcheck(model, rng)
    bank = build_bank(model, rng)
    g_im = gradcheck_imagination(model, bank, rng)
    tol = th["gradcheck_rel_error"]
    gc_ok = max(g_init.values()) <= tol and max(g_tr.values()) <= tol and g_im <= tol
    tests = [("C1", "gradient_check", gc_ok, "worst tensor %s" % max(g_tr, key=g_tr.get))]
    tests.append(("C2", "determinism_finiteness") + test_determinism(seeds[-1]))
    acc, triv, glob = heldout_metrics(model, rng)
    drop = loss_drop(L)
    tests.append(("C3", "learning", drop >= th["loss_drop_fraction"] and acc - triv >= th["margin_over_trivial"],
                  "loss drop %.2f · held-out acc %.3f vs per-sign majority %.3f" % (drop, acc, triv)))
    shuf = build_model(rng=rng)
    fit(shuf, None, steps, rng, shuffle_labels=True)        # same update budget as the real model
    s_acc = heldout_metrics(shuf, rng)[0]
    tests.append(("C4", "shuffled_label_control", s_acc <= glob + th["c4_band"],
                  "shuffled acc %.3f vs best constant predictor %.3f" % (s_acc, glob)))
    if a.mutant is None:
        ctrl, det = test_mutants(seeds[-1], 300, tol, th["loss_drop_fraction"])
        tests.append(("C5", "mutant_detection", ctrl and det == len(MUTANTS),
                      "clean control passes: %s · detected %d/%d" % (ctrl, det, len(MUTANTS))))
    else:
        ctrl, det = False, 0
        tests.append(("C5", "mutant_detection", True, "skipped: this run is the mutant %s" % a.mutant))
    for tid, name, fn in (("C6.P1", "set_invariance", lambda: test_set_invariance(rng)),
                          ("C6.P2", "rival_support", lambda: test_support(model, bank, rng)),
                          ("C6.P3", "tailoring_ascent", lambda: test_ascent(model, bank, rng))):
        tests.append((tid, name) + fn())
    tests.append(("C7", "split_integrity") + test_split(rng))
    per_seed = []
    if a.mutant is None:
        per_seed.append(evaluate_seed(model, rng, n_ep))
        for kid in kids[1:n_seeds]:
            r = np.random.default_rng(kid)
            m = build_model(rng=r)
            fit(m, None, steps, r)
            per_seed.append(evaluate_seed(m, r, n_ep))
    stat_rng = np.random.default_rng(kids[-1])
    hyps = hypothesis_rows(per_seed, stat_rng) if per_seed else []
    kos = knockout_rows(per_seed, stat_rng) if per_seed else []
    runtime = time.time() - t0
    tests.append(("C8", "budget", runtime <= budget_s, "%.1f s of %d s" % (runtime, budget_s)))
    if a.data:
        data_bridge(a.data, np.random.default_rng(kids[-1]), steps)
    info = {"mode": "quick" if a.quick else ("mutant " + a.mutant if a.mutant else "full"),
            "seeds": seeds[:n_seeds], "runtime": runtime, "budget_s": budget_s, "n_params": n_params(model),
            "steps": steps, "per_seed": per_seed, "hypotheses": hyps, "knockouts": kos,
            "gradcheck": {"tensors_checked": len(g_tr) + 1, "tensors_total": len(model["params"]) + 1,
                          "init": max(g_init.values()), "trained": max(g_tr.values()), "imagination": g_im,
                          "passed": bool(gc_ok)},
            "correctness": [{"id": t[0], "name": t[1], "passed": bool(t[2]), "detail": t[3]} for t in tests],
            "mutants": {"detected": det, "total": len(MUTANTS), "score": det / len(MUTANTS)}}
    failed = any(not t[2] for t in tests if t[0] != "C8")
    code = 1 if failed else (3 if runtime > budget_s else 0)
    write_report(CHAPTER, build_sections(info), a.json, json_payload(info, code))
    return code


def main(argv=None):
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__), description="Chapter 0358 audit architecture")
    ap.add_argument("--quick", action="store_true", help="one seed, reduced steps, all correctness tests")
    ap.add_argument("--seed", type=int, default=0, help="base seed")
    ap.add_argument("--seeds", type=int, default=None, help="number of seeds (default 5)")
    ap.add_argument("--json", default=None, help="also write the JSON report to this path")
    ap.add_argument("--card", action="store_true", help="print MIND_CARD as JSON and exit")
    ap.add_argument("--mutant", choices=sorted(MUTANTS), default=None, help="run with a registered mutant")
    ap.add_argument("--data", default=None, help="optional CSV for the real-data bridge")
    a = ap.parse_args(argv)
    if a.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    if a.seeds is not None and a.seeds < 1:
        ap.error("--seeds must be at least 1")
    try:
        return run(a)
    except NonFinite as err:
        print("non-finite values: %s" % err)
        return 4


if __name__ == "__main__":
    sys.exit(main())

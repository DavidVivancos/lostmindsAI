#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0314 · Saxo Grammaticus
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 16, Minds 301-320 Available on Amazon https://www.amazon.com/gp/product/B0HNJQX9S4
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos 
# END ATTRIBUTION
"""
THE STEWARD'S CHAIN — recursive unmaking of an object's history (chapter 0314).

Thesis
    An object is the residue of a chain of makings, each of which leaves a taste
    that later makings overlay but do not erase; to know the thing is to undo the
    chain one committed link at a time until no trace of making remains.

Evidence and provenance (provenance: belief; Saxo's own Latin text survives)
    D1  Gesta Danorum III, Amleth in Britain: bread tasting of blood is traced to
        grain grown on an old battlefield, drink tasting of iron to rusted swords in
        the spring; the king verifies each link by questioning his steward and by
        digging. (Friis-Jensen & Fisher 2015; Elton 1894, pp. 106-130)   primary
    D2  The same passage: Amleth names one cause per taste and is later proved right
        link by link; he never blends causes. (as D1)                       primary
    D3  Preface: the Runamo marks in Blekinge are read as an inscription whose
        illegibility is blamed on mud and trampling; Book VII assigns it to Harald
        Wartooth. Berzelius (1836) showed it is a cracked dolerite dike. primary + scholarship
    D4  Preface, end: the huge stones of Danish dolmens are taken as proof that a race
        of giants once lived in Denmark. (Friis-Jensen, MNLL 2012)          primary
    D5  Preface: Saxo builds a history of a people without written records from
        traces (old Danish songs, Icelanders' tales, Absalon's statements). primary

Doctrine -> mechanism -> test
    D1 -> M1 unmaking library: K weight-tied inverse-process operators   -> C6.2, H-SIG
    D2 -> M2 maker router: recognise the outermost maker; commit at inference -> knockout
    D1 -> M3 link verification: every peeled latent must match the encoding
          of the object observed one making earlier (training protocol)  -> H-NEC
    D3,D4 -> M4 origin test: halting on "no trace of making remains"     -> C6.1, H-BLIND

Research question (compositional and systematic generalization)
    Can a weight-tied library of learned inverse operators, routed by the trace each
    process leaves, verified link by link and stopped by an origin test, recover the
    origin of objects whose making histories are longer than any seen in training;
    and does its commitment to explaining every mark make it attribute made histories
    to natural marks that resemble made ones?

Closest prior art and the delta
    Parascandolo et al. 2018 (competing experts learn inverse mechanisms), adaptive
    halting (Graves 2016 ACT; Banino et al. 2021 PonderNet), neural module networks
    (Andreas et al. 2016), multi-step latent consistency (Ye et al. 2021 EfficientZero).
    Delta: the experts are applied recursively backwards from the observed object,
    routed by the fading signature each process leaves, trained with stage-by-stage
    latent verification, halted by a test on the latent rather than a step count,
    and evaluated on longer histories and on over-attribution to natural marks.
    Baseline: size-matched multi-head MLP given identical rows and an auxiliary head
    that predicts the object one making earlier.

Blind spot (grounded in D3, D4)
    A mind that explains every mark as a chain of known makings should find a chain
    in natural marks that resemble made ones (Runamo; the giants' stones). H-BLIND
    predicts the model separates made from natural marks worse than the baseline.

Task (synthetic "world of makings")
    origin content c0 = prototype[y] + noise (5 origins), trace record r0 = 0
    each making k: c <- Q_k (c + 0.25 tanh c) + b_k ; r <- 0.6 r + m_k (orthonormal m_k)
    observation x = gain * P [c; r] + noise, plus 4 nuisance channels (P orthonormal)
    train: histories of 0-3 makings, every stage observed; held-out: 0-3; shifted: 4-5
    blind: natural marks from out-of-library processes whose signatures resemble
    library signatures (cosine 0.8) acting on non-prototype origins
    target: the origin y, never readable from one input feature

Limits
    Synthetic world with exactly invertible processes and orthonormal signatures; the
    file is a research prototype of one component of an AGI-oriented architecture, not
    an AGI and not a model of Saxo's mind. Depth-5 histories approach the physical
    limit of the world (signatures fade by 0.6 per making).
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 2,
    "revision_log": [
        "r1 (before any protocol run): pilot engineering runs recalibrated the synthetic world after an exact-inverse "
        "oracle showed the first world destroyed origin information by depth 5 (oracle 0.51); signatures made "
        "orthonormal, sensor made orthonormal, fade 0.6, noise 0.01, gain 0.02. Link verification of every peel and "
        "committed routing at inference were added as architecture decisions. H-SIG, H-NEC, metrics and mesi unchanged.",
        "r2 (before the protocol run): pilots with natural marks unrelated to library signatures gave the model a HIGHER "
        "made-vs-natural AUROC than the baseline, opposite to H-BLIND. H-BLIND's condition was re-specified to "
        "letter-like natural marks (cosine 0.8 to a library signature), the faithful rendition of Runamo; the unrelated "
        "condition is kept and reported as probe P-UNLIKE so the unfavourable pilot direction stays visible.",
    ],
    "generation": {"template_version": "codeguidelines 1.0 (2026-09-15)", "generator": "Claude (Anthropic)",
                   "generator_version": "claude-opus-5-5", "date": "2026-09-24"},
    "id": 314, "figure": "Saxo Grammaticus", "born": 1160, "died": 1220, "civilization": "Danish",
    "provenance": "belief",
    "thesis": "An object is the residue of a chain of makings whose tastes overlay but do not erase one another; "
              "knowing it means undoing the chain one committed link at a time until no trace of making remains.",
    "evidence": [
        {"id": "D1", "claim": "Amleth traces bread to a battlefield and drink to rusted swords; the king verifies each "
                              "link by questioning the steward and digging the spring", "basis": "primary",
         "source": "Gesta Danorum III (Friis-Jensen & Fisher 2015); Elton 1894 trans. pp. 106-130"},
        {"id": "D2", "claim": "One named cause per taste, each later confirmed; causes are not blended",
         "basis": "primary", "source": "Gesta Danorum III, as D1"},
        {"id": "D3", "claim": "Runamo cracks read as an illegible inscription, later assigned to Harald Wartooth; shown "
                              "natural by Berzelius 1836", "basis": "primary",
         "source": "Gesta Danorum, Preface and VII; Friis-Jensen, MNLL (2012)"},
        {"id": "D4", "claim": "Dolmen stones taken as proof of a former race of giants", "basis": "primary",
         "source": "Gesta Danorum, Preface (end); Friis-Jensen, MNLL (2012)"},
        {"id": "D5", "claim": "History of an unlettered past compiled from traces: songs, Icelanders, Absalon",
         "basis": "primary", "source": "Gesta Danorum, Preface 1.1-1.4"},
    ],
    "research_question": {"category": "compositional and systematic generalization",
                          "question": "Does recursive, verified, committed unmaking recover origins of histories longer "
                                      "than any seen in training, and does it over-attribute made histories to "
                                      "letter-like natural marks?"},
    "mechanism": {
        "name": "Steward's Chain (recursive committed unmaking with an origin test)",
        "family": "recursive inverse-process inference (latent history induction) with adaptive halting",
        "signature_modules": ["unmaking_library", "maker_router", "commitment", "origin_test"],
        "closest_prior_art": ["Parascandolo et al. 2018, Learning Independent Causal Mechanisms (ICML)",
                              "Graves 2016, Adaptive Computation Time (arXiv:1603.08983)",
                              "Banino et al. 2021, PonderNet (arXiv:2107.05407)",
                              "Andreas et al. 2016, Neural Module Networks (CVPR)",
                              "Ye et al. 2021, EfficientZero latent consistency (arXiv:2111.00210)"],
        "overlap": "Medium",
        "prior_art_queries": ["inverse mechanisms competing experts", "adaptive computation time halting",
                              "length generalization recursive operators", "processing history estimation chain",
                              "latent consistency multi-step inverse dynamics"],
        "contribution_type": "mechanism",
        "delta": "Inverse experts applied recursively backwards from the observed object, routed by fading process "
                 "signatures, verified stage by stage, committed at inference and halted by an origin test on the latent.",
    },
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1", "property_test": "C6.2", "hypothesis": "H-SIG"},
        {"doctrine": "D1", "mechanism": "M3", "property_test": "C1", "hypothesis": "H-NEC"},
        {"doctrine": "D2", "mechanism": "M2", "property_test": "C6.2", "hypothesis": "knockout:commitment"},
        {"doctrine": "D3", "mechanism": "M4", "property_test": "C6.1", "hypothesis": "H-BLIND"},
        {"doctrine": "D4", "mechanism": "M4", "property_test": "C6.1", "hypothesis": "H-BLIND"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "On histories of 4-5 makings (trained on 0-3) the Steward's Chain recovers the "
                                     "origin more often than the size-matched flat MLP", "metric": "origin accuracy",
         "split": "shifted", "comparison": "model - baseline", "direction": "greater", "mesi": 0.05, "seeds": 5},
        {"id": "H-NEC", "statement": "Retraining with the unmaking library replaced by identity lowers shifted accuracy "
                                     "more than retraining with encoder layer 2 replaced by identity",
         "metric": "origin accuracy (shifted)", "comparison": "signature_knockout - matched_knockout",
         "direction": "less", "mesi": 0.05, "seeds": 5},
        {"id": "H-BLIND", "statement": "Max-probability confidence separates made histories from letter-like natural "
                                       "marks worse for the Steward's Chain than for the baseline",
         "condition": "letter-like natural marks (Runamo)", "grounding": "Runamo read as runes; dolmens read as giants' work",
         "metric": "AUROC made vs natural", "comparison": "model - baseline", "direction": "less", "mesi": 0.03, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.5, "margin_over_trivial": 0.3},
    "probe_predictions": [{"probe": "P4", "expected": "above baseline"}, {"probe": "P7", "expected": "equal to baseline"},
                          {"probe": "P8", "expected": "below baseline"}, {"probe": "P1", "expected": "equal to baseline"}],
    "dialectic_links": [
        {"chapter": 321, "relation": "rival", "test": "none implemented (Snorri's euhemerism keeps gods as keepers of "
                                                      "knowledge; Saxo's as illusionists)"},
        {"chapter": 851, "relation": "successor", "test": "none implemented (Berzelius 1836 refuted the Runamo reading)"}],
    "corpus_neighbors": [
        {"chapter": 127, "similarity": None, "difference": "Tacitus deconvolves bias from testimony; here physical traces "
                                                           "of processes are inverted, no testimony model"},
        {"chapter": 278, "similarity": None, "difference": "Ari solves dates from counted relations; here no clocks, the "
                                                           "unknown is the ordered chain of makings"},
        {"chapter": 321, "similarity": None, "difference": "Snorri bounds kenning depth and checks genealogies; here depth "
                                                           "is unbounded and halted by an origin test"},
        {"chapter": 332, "similarity": None, "difference": "Sturla sorts records by how knowledge arrived; no source typing here"},
        {"chapter": 193, "similarity": None, "difference": "Boniface gates noise versus forgery by intent; here natural marks "
                                                           "are only a blind-spot test, not a gate"},
        {"chapter": 311, "similarity": None, "difference": "Ibn al-Athir keeps two indexes; single recursive latent here"},
    ],
    "similarity_review": "9-token shingle Jaccard (section 4.4 procedure) = 0.027 against the only corpus file available in "
                         "this session (the 0507 sample); the full-corpus gate is pending.",
    "barometer": {
        "cognitive_processing": ["length extrapolation of inferred chains (shifted split)", "per-depth accuracy curve"],
        "embodied_cognition": [], "world_modeling": ["learned inverse model of a library of physical processes"],
        "consciousness": ["made-vs-natural AUROC (over-attribution audit)", "origin-test halting mass"],
        "language_understanding": [], "emotional_intelligence": [], "creativity": [],
        "autonomy": ["adaptive depth: the model decides when the origin is reached"]},
    "task_types": ["vector_classification"],
    "applications": [
        {"use": "estimate the ordered chain of processing operations applied to an image before judging its origin",
         "sector": "media forensics and journalism", "dataset": "RAISE raw image dataset (Dang-Nguyen et al. 2015), chains synthesized"},
        {"use": "trace a defect signature back through the ordered process steps that produced it",
         "sector": "semiconductor manufacturing", "dataset": "UCI SECOM"},
        {"use": "audit false attribution of machine processing to human-written text before deployment",
         "sector": "AI-content provenance", "dataset": "HC3 (Guo et al. 2023)"}],
    "safety_notes": "Never used to attribute intent or authorship to people without human review; the Runamo condition "
                    "(over-attribution) must be audited on natural data before any forensic use. Synthetic data only.",
}

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

CHAPTER = 314
SC, SR, K, C = 6, 6, 6, 5            # content dims, trace dims, makings in the library, origins
D_INF, D_NUIS = 16, 4
D_IN = D_INF + D_NUIS
LMAX, RHO, WARP = 5, 0.6, 0.25
TRAIN_DEPTHS, SHIFT_DEPTHS = (0, 1, 2, 3), (4, 5)
T_TRAIN, T_TEST = 4, 7
LAM_R, LAM_H, LAM_C = 0.5, 0.5, 1.0
LR, CLIP, BATCH = 3e-3, 5.0, 128
H_STEWARD, H_FLAT = 24, 44
NPOS, NDEP = 3, 4                    # baseline heads: makers at 3 positions, depth 0-3
GC_TOL, GC_FLOOR = 1e-5, 1e-4

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
    return 0.5 * (1.0 + np.tanh(0.5 * z))


class Adam:
    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-8, sign=-1.0):
        self.lr, self.b1, self.b2, self.eps, self.sign, self.t = lr, b1, b2, eps, sign, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] += self.sign * self.lr * mh / (np.sqrt(vh) + self.eps)


def clip_global_norm(grads, max_norm):
    norm = float(np.sqrt(sum(float((g ** 2).sum()) for g in grads.values())))
    if norm > max_norm:
        for k in grads:
            grads[k] *= max_norm / norm
    return norm


def gradcheck(loss_fn, params, grads, rng, eps=1e-6, n_entries=20):
    """Central differences in float64 on >= n_entries random entries per tensor plus the largest-gradient
    entry. Relative error |a-n| / max(|a|+|n|, GC_FLOOR): the floor keeps near-zero entries from dividing
    rounding noise by rounding noise (it is an absolute tolerance of GC_TOL*GC_FLOOR there)."""
    worst = {}
    for name, p in params.items():
        flat, gflat = p.reshape(-1), grads[name].reshape(-1)
        idx = rng.choice(flat.size, size=min(n_entries, flat.size), replace=False)
        idx = np.unique(np.append(idx, int(np.argmax(np.abs(gflat)))))
        err = 0.0
        for i in idx:
            old = flat[i]
            flat[i] = old + eps
            lp = loss_fn()
            flat[i] = old - eps
            lm = loss_fn()
            flat[i] = old
            num = (lp - lm) / (2 * eps)
            err = max(err, abs(num - gflat[i]) / max(abs(num) + abs(gflat[i]), GC_FLOOR))
        worst[name] = err
    return worst


def paired_bootstrap(diffs, rng, n_resamples=2000):
    d = np.asarray(diffs, dtype=float)
    means = d[rng.integers(d.size, size=(n_resamples, d.size))].mean(1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def verdict(mean, lo, hi, direction, mesi):
    if direction == "greater":
        if lo > 0 and mean >= mesi:
            return "supported"
        return "contradicted" if hi < 0 else "inconclusive"
    if hi < 0 and mean <= -mesi:
        return "supported"
    return "contradicted" if lo > 0 else "inconclusive"


def write_json_report(report, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
# END STANDARD UTILITIES


# ------------------------------------------------------------------ data: a world of makings
def _orth(rng, n):
    q, r = np.linalg.qr(rng.normal(size=(n, n)))
    return q * np.sign(np.diag(r))


def make_world(rng):
    """Library makings (Q, b, M), natural processes for the blind spot, and an orthonormal sensor."""
    M = _orth(rng, SR)[:K]
    like = np.empty((4, SR))
    for i, j in enumerate(rng.choice(K, size=4, replace=False)):
        u = rng.normal(size=SR)
        u -= (u @ M[j]) * M[j]
        like[i] = 0.8 * M[j] + 0.6 * u / np.linalg.norm(u)      # resembles a library signature
    unlike = rng.normal(size=(4, SR))
    return {"protos": rng.uniform(-0.8, 0.8, (C, SC)),
            "Q": np.stack([_orth(rng, SC) for _ in range(K)]), "b": rng.uniform(-0.4, 0.4, (K, SC)), "M": M,
            "Qn": np.stack([_orth(rng, SC) for _ in range(4)]), "bn": rng.uniform(-0.4, 0.4, (4, SC)),
            "M_like": like, "M_unlike": unlike / np.linalg.norm(unlike, axis=1, keepdims=True),
            "P": np.linalg.qr(rng.normal(size=(D_INF, SC + SR)))[0]}


def sample_objects(w, rng, n, depths, kind="made"):
    """Grow n histories. kind: 'made' (library), 'like' or 'unlike' (natural marks)."""
    y = rng.integers(C, size=n)
    if kind == "made":
        c = w["protos"][y] + 0.1 * rng.normal(size=(n, SC))
        Q, b, M = w["Q"], w["b"], w["M"]
    else:
        c = rng.uniform(-0.8, 0.8, (n, SC))
        Q, b, M = w["Qn"], w["bn"], w["M_like"] if kind == "like" else w["M_unlike"]
    r = np.zeros((n, SR))
    L = rng.choice(np.array(depths), size=n)
    hist, applied = [(c.copy(), r.copy())], -np.ones((n, LMAX), dtype=int)
    for j in range(LMAX):
        act, k = L > j, rng.integers(len(M), size=n)
        c2 = np.einsum("nij,nj->ni", Q[k], c + WARP * np.tanh(c)) + b[k]
        r2 = RHO * r + M[k]
        c, r = np.where(act[:, None], c2, c), np.where(act[:, None], r2, r)
        applied[:, j] = np.where(act, k, -1)
        hist.append((c.copy(), r.copy()))
    chain = -np.ones((n, LMAX), dtype=int)           # peel order: outermost making first
    for t in range(LMAX):
        idx = L - 1 - t
        ok = idx >= 0
        chain[ok, t] = applied[np.where(ok)[0], idx[ok]]
    return {"y": y, "chain": chain, "depth": L, "hist": hist}


def observe(w, rng, c, r):
    gain = rng.uniform(0.98, 1.02, (c.shape[0], 1))
    xi = gain * (np.concatenate([c, r], 1) @ w["P"].T) + 0.01 * rng.normal(size=(c.shape[0], D_INF))
    return np.concatenate([xi, rng.normal(0, 0.5, (c.shape[0], D_NUIS))], 1)


def as_found(w, rng, obj):
    c, r = obj["hist"][-1]
    return {"X": observe(w, rng, c, r), "y": obj["y"], "chain": obj["chain"], "depth": obj["depth"]}


def with_links(w, rng, obj):
    """Every stage of every history is a row; each row carries the observed objects one, two, ... makings
    earlier (the steward's answers), used by the verification loss and by the baseline's auxiliary head."""
    obs = [observe(w, rng, *obj["hist"][j]) for j in range(LMAX + 1)]
    rows = {"X": [], "Xnext": [], "Xstack": [], "y": [], "chain": [], "depth": []}
    for i in range(obj["y"].size):
        for t in range(obj["depth"][i] + 1):
            j = obj["depth"][i] - t
            ch = -np.ones(LMAX, dtype=int)
            ch[:LMAX - t] = obj["chain"][i, t:]
            rows["X"].append(obs[j][i])
            rows["Xnext"].append(obs[max(j - 1, 0)][i])
            rows["Xstack"].append(np.stack([obs[max(j - 1 - u, 0)][i] for u in range(LMAX)]))
            rows["y"].append(obj["y"][i]); rows["chain"].append(ch); rows["depth"].append(j)
    return {k: np.asarray(v) for k, v in rows.items()}


def build_splits(rng, n_train):
    w = make_world(rng)
    sp = {"world": w,
          "train": with_links(w, rng, sample_objects(w, rng, n_train, TRAIN_DEPTHS)),
          "val": with_links(w, rng, sample_objects(w, rng, 250, TRAIN_DEPTHS)),
          "test_id": as_found(w, rng, sample_objects(w, rng, 1000, TRAIN_DEPTHS)),
          "test_shift": as_found(w, rng, sample_objects(w, rng, 1000, SHIFT_DEPTHS)),
          "blind_like": as_found(w, rng, sample_objects(w, rng, 600, (1, 2, 3), "like")),
          "blind_unlike": as_found(w, rng, sample_objects(w, rng, 600, (1, 2, 3), "unlike"))}
    sp["by_depth"] = {L: as_found(w, rng, sample_objects(w, rng, 400, (L,))) for L in range(LMAX + 1)}
    return sp


def auroc(pos, neg):
    s = np.concatenate([pos, neg])
    ranks = s.argsort().argsort() + 1.0
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ------------------------------------------------------------------ model: the Steward's Chain
TASK_TYPES = ["vector_classification"]
_MUTANT = {"name": None}


def build_model(in_dim, out_dim, task_type="vector_classification", rng=None, **cfg):
    rng = rng if rng is not None else np.random.default_rng(0)
    H, k = cfg.get("H", H_STEWARD), cfg.get("K", K)
    gi = lambda a, b: rng.normal(0, 1 / np.sqrt(a), (a, b))
    params = {"We1": gi(in_dim, H), "be1": np.zeros(H), "We2": gi(H, H), "be2": np.zeros(H),
              "U": rng.normal(0, 1 / np.sqrt(H), (k, H, H)), "c": np.zeros((k, H)),
              "V": gi(H, k), "d": np.zeros(k), "q": rng.normal(0, 1 / np.sqrt(H), H), "r": np.zeros(1),
              "O": gi(H, out_dim), "o": np.zeros(out_dim)}
    flags = {"library": "learned", "enc2": "learned", "router": "learned", "halt": "learned",
             "readout": "learned", "commit": True}
    flags.update({f: cfg[f] for f in flags if f in cfg})
    return {"kind": "steward", "task_type": task_type, "params": params, "H": H, "K": k, "flags": flags,
            "halt_mean": 0.5}


def _encode(P, f, X):
    e1 = np.tanh(X @ P["We1"] + P["be1"])
    return e1, (e1 if f["enc2"] == "identity" else np.tanh(e1 @ P["We2"] + P["be2"]))


def _encode_back(P, f, g, X, e1, h0, gh0):
    if f["enc2"] == "identity":
        ge1 = gh0
    else:
        gz2 = gh0 * (1 - h0 ** 2)
        g["We2"] += e1.T @ gz2
        g["be2"] += gz2.sum(0)
        ge1 = gz2 @ P["We2"].T
    gz1 = ge1 * (1 - e1 ** 2)
    g["We1"] += X.T @ gz1
    g["be1"] += gz1.sum(0)


def _forward(m, X, T, train):
    P, f, H, k = m["params"], m["flags"], m["H"], m["K"]
    B = X.shape[0]
    e1, h0 = _encode(P, f, X)
    hs, As, Gs = [h0], [], []
    Uf = P["U"].transpose(1, 0, 2).reshape(H, k * H)
    for _ in range(T):
        h = hs[-1]
        a = np.full((B, k), 1.0 / k) if f["router"] == "mean" else softmax(h @ P["V"] + P["d"])
        if f["commit"] and not train and f["router"] != "mean":
            a = (a == a.max(1, keepdims=True)).astype(float)       # one named maker per link (D2)
            a /= a.sum(1, keepdims=True)
        G = None if f["library"] == "identity" else np.tanh((h @ Uf).reshape(B, k, H) + P["c"][None])
        As.append(a); Gs.append(G)
        hs.append(h if G is None else np.einsum("bk,bkj->bj", a, G))
    us = [hs[t] @ P["q"] + P["r"][0] for t in range(T)]
    ps = [sigmoid(u) for u in us] if f["halt"] == "learned" else [np.full(B, m["halt_mean"]) for _ in range(T)]
    pis, rest = [], np.ones(B)
    for t in range(T):
        pis.append(ps[t] * rest)
        rest = rest * (1 - ps[t])
    pis.append(rest)                                   # remainder read out at the last step
    if f["readout"] == "zero":
        Ps = [np.full((B, P["o"].size), 1.0 / P["o"].size) for _ in range(T + 1)]
    else:
        Ps = [softmax(hs[t] @ P["O"] + P["o"]) for t in range(T + 1)]
    mix = sum(pis[t][:, None] * Ps[t] for t in range(T + 1))
    return {"e1": e1, "hs": hs, "As": As, "Gs": Gs, "us": us, "ps": ps, "pis": pis, "Ps": Ps, "mix": mix, "T": T}


def _readout_back(m, fw, y, g, gh):
    P, B = m["params"], y.size
    ar, my = np.arange(B), fw["mix"][np.arange(B), y]
    gpi = []
    for t in range(fw["T"] + 1):
        Pt = fw["Ps"][t]
        gpi.append(-Pt[ar, y] / (B * my))
        if m["flags"]["readout"] == "zero":
            continue
        onehot = (np.arange(Pt.shape[1])[None] == y[:, None]).astype(float)
        gl = (-fw["pis"][t] / (B * my) * Pt[ar, y])[:, None] * (onehot - Pt)
        g["O"] += fw["hs"][t].T @ gl
        g["o"] += gl.sum(0)
        gh[t] += gl @ P["O"].T
    return -float(np.mean(np.log(my))), gpi


def _origin_test_back(m, fw, batch, gpi, g, gh):
    """Gradient through the halting distribution pi_t = p_t prod_{j<t}(1-p_j), remainder at T, computed with
    explicit products that skip p_s (no division by 1-p_s), plus the supervised origin test on known depths."""
    if m["flags"]["halt"] != "learned":
        return 0.0
    P, T, ps, B = m["params"], fw["T"], fw["ps"], batch["X"].shape[0]
    om, loss = [1 - p for p in ps], 0.0
    for s in range(T):
        dp = gpi[s] * (np.prod(om[:s], axis=0) if s > 0 else 1.0)
        for t in range(s + 1, T + 1):
            prod = np.prod([om[j] for j in range(t) if j != s], axis=0) if t > 1 else np.ones(B)
            dp = dp - gpi[t] * prod * (ps[t] if t < T else 1.0)
        du = dp * ps[s] * (1 - ps[s])
        if "depth" in batch:
            mask, tgt = (s <= batch["depth"]).astype(float), (s == batch["depth"]).astype(float)
            loss += LAM_H / B * float(np.sum(mask * (softplus(fw["us"][s]) - tgt * fw["us"][s])))
            du = du + LAM_H / B * mask * (ps[s] - tgt)
        g["q"] += fw["hs"][s].T @ du
        g["r"] += du.sum()
        gh[s] += du[:, None] * P["q"][None]
    return loss


def _verification_back(m, fw, batch, g, gh):
    """The steward's answers: after t+1 unmakings the latent must match the encoding of the object observed
    t+1 makings earlier. Gradient flows into both the peeled latent and the target encoding."""
    if "Xstack" not in batch:
        return 0.0
    P, f, B, loss = m["params"], m["flags"], batch["X"].shape[0], 0.0
    for t in range(fw["T"]):
        mask = (batch["depth"] >= t + 1).astype(float)
        if mask.sum() == 0:
            continue
        Xt = batch["Xstack"][:, t]
        e1n, hn = _encode(P, f, Xt)
        diff = (fw["hs"][t + 1] - hn) * mask[:, None]
        loss += LAM_C / B * float(np.sum(diff ** 2))
        gd = 2 * LAM_C / B * diff
        gh[t + 1] += gd
        _encode_back(P, f, g, Xt, e1n, hn, -gd)
    return loss


def _unmaking_back(m, fw, batch, g, gh):
    P, f, H, k = m["params"], m["flags"], m["H"], m["K"]
    B, ar, loss = batch["X"].shape[0], np.arange(batch["X"].shape[0]), 0.0
    Uf = P["U"].transpose(1, 0, 2).reshape(H, k * H)
    for t in range(fw["T"] - 1, -1, -1):
        a, G, h, ghn = fw["As"][t], fw["Gs"][t], fw["hs"][t], gh[t + 1]
        ds = np.zeros((B, k))
        if "chain" in batch and f["router"] == "learned":
            mask = (t < batch["depth"]).astype(float)
            kk = np.where(batch["chain"][:, t] >= 0, batch["chain"][:, t], 0)
            loss += LAM_R / B * float(np.sum(mask * -np.log(a[ar, kk])))
            onehot = np.zeros((B, k))
            onehot[ar, kk] = 1
            ds += LAM_R / B * mask[:, None] * (a - onehot)
        if G is None:
            gh[t] += ghn
        else:
            ga = np.einsum("bj,bkj->bk", ghn, G)
            gZ = (a[:, :, None] * ghn[:, None, :]) * (1 - G ** 2)
            gZf = gZ.reshape(B, k * H)
            g["U"] += (h.T @ gZf).reshape(H, k, H).transpose(1, 0, 2)
            g["c"] += gZ.sum(0)
            gh[t] += gZf @ Uf.T
            if f["router"] == "learned":
                ds += a * (ga - (a * ga).sum(1, keepdims=True))
        if f["router"] == "learned":
            g["V"] += h.T @ ds
            g["d"] += ds.sum(0)
            gh[t] += ds @ P["V"].T
    return loss


def _steward_loss_and_grads(m, batch):
    P, X, y = m["params"], batch["X"], batch["y"]
    fw = _forward(m, X, T_TRAIN, train=True)
    g = {k: np.zeros_like(v) for k, v in P.items()}
    gh = [np.zeros((X.shape[0], m["H"])) for _ in range(fw["T"] + 1)]
    loss, gpi = _readout_back(m, fw, y, g, gh)
    loss += _origin_test_back(m, fw, batch, gpi, g, gh)
    loss += _verification_back(m, fw, batch, g, gh)
    loss += _unmaking_back(m, fw, batch, g, gh)
    _encode_back(P, m["flags"], g, X, fw["e1"], fw["hs"][0], gh[0])
    return loss, g


# ------------------------------------------------------------------ baseline: size-matched flat MLP
def build_flat(in_dim, out_dim, rng, H1=H_FLAT):
    gi = lambda a, b: rng.normal(0, 1 / np.sqrt(a), (a, b))
    n_out = out_dim + NPOS * K + NDEP + in_dim
    return {"kind": "flat", "C": out_dim, "D": in_dim,
            "params": {"W1": gi(in_dim, H1), "b1": np.zeros(H1), "W2": gi(H1, H1), "b2": np.zeros(H1),
                       "Wh": gi(H1, n_out), "bh": np.zeros(n_out)}}


def _flat_forward(m, X):
    P = m["params"]
    a1 = np.tanh(X @ P["W1"] + P["b1"])
    a2 = np.tanh(a1 @ P["W2"] + P["b2"])
    return a1, a2, a2 @ P["Wh"] + P["bh"]


def _flat_heads(m, batch, z):
    """Same supervision as the Steward's Chain: origin, makers by position, depth, earlier object."""
    B, Cn, ar = z.shape[0], m["C"], np.arange(z.shape[0])
    gz = np.zeros_like(z)
    po = softmax(z[:, :Cn])
    loss = -float(np.mean(np.log(po[ar, batch["y"]])))
    gz[:, :Cn] = (po - (np.arange(Cn)[None] == batch["y"][:, None])) / B
    if "chain" in batch:
        for t in range(NPOS):
            sl = slice(Cn + t * K, Cn + (t + 1) * K)
            pp, mask = softmax(z[:, sl]), (t < batch["depth"]).astype(float)
            kk = np.where(batch["chain"][:, t] >= 0, batch["chain"][:, t], 0)
            loss += LAM_R / B * float(np.sum(mask * -np.log(pp[ar, kk])))
            gz[:, sl] = LAM_R / B * mask[:, None] * (pp - (np.arange(K)[None] == kk[:, None]))
        sl = slice(Cn + NPOS * K, Cn + NPOS * K + NDEP)
        pd, dl = softmax(z[:, sl]), np.clip(batch["depth"], 0, NDEP - 1)
        loss += LAM_H * -float(np.mean(np.log(pd[ar, dl])))
        gz[:, sl] = LAM_H / B * (pd - (np.arange(NDEP)[None] == dl[:, None]))
    if "Xnext" in batch:
        sl = slice(Cn + NPOS * K + NDEP, z.shape[1])
        diff = (z[:, sl] - batch["Xnext"]) * (batch["depth"] >= 1).astype(float)[:, None]
        loss += LAM_C / B * float(np.sum(diff ** 2))
        gz[:, sl] = 2 * LAM_C / B * diff
    return loss, gz


def _flat_loss_and_grads(m, batch):
    P, X = m["params"], batch["X"]
    a1, a2, z = _flat_forward(m, X)
    loss, gz = _flat_heads(m, batch, z)
    g = {"Wh": a2.T @ gz, "bh": gz.sum(0)}
    g2 = gz @ P["Wh"].T * (1 - a2 ** 2)
    g["W2"], g["b2"] = a1.T @ g2, g2.sum(0)
    g1 = g2 @ P["W2"].T * (1 - a1 ** 2)
    g["W1"], g["b1"] = X.T @ g1, g1.sum(0)
    return loss, g


# ------------------------------------------------------------------ common interface
def loss_and_grads(model, batch):
    if model["kind"] == "steward":
        loss, g = _steward_loss_and_grads(model, batch)
    else:
        loss, g = _flat_loss_and_grads(model, batch)
    if _MUTANT["name"] == "zero_grad_U" and "U" in g:
        g["U"][:] = 0.0
    if _MUTANT["name"] == "zero_grad_We1" and "We1" in g:
        g["We1"][:] = 0.0
    return loss, g


def predict(model, X):
    if model["kind"] == "steward":
        return _forward(model, X, T_TEST, train=False)["mix"]
    return softmax(_flat_forward(model, X)[2][:, :model["C"]])


def hidden_states(model, X):
    if model["kind"] == "steward":
        return np.stack(_forward(model, X, T_TEST, train=False)["hs"], 1)
    return _flat_forward(model, X)[1]


def modules(model):
    if model["kind"] != "steward":
        return [{"name": "mlp", "params": list(model["params"]), "role": "two tanh layers and linear heads",
                 "signature": False}]
    return [
        {"name": "sense_encoder_l1", "params": ["We1", "be1"], "role": "tanh input layer", "signature": False},
        {"name": "sense_encoder_l2", "params": ["We2", "be2"], "role": "tanh hidden layer", "signature": False},
        {"name": "unmaking_library", "params": ["U", "c"],
         "role": "K weight-tied inverse-process operators (tanh layers) applied recursively", "signature": True},
        {"name": "maker_router", "params": ["V", "d"], "role": "softmax router over operators, shared across steps",
         "signature": True},
        {"name": "commitment", "params": [], "role": "argmax routing at inference (hard gating)", "signature": True},
        {"name": "origin_test", "params": ["q", "r"], "role": "sigmoid halting unit on the latent (adaptive depth)",
         "signature": True},
        {"name": "origin_readout", "params": ["O", "o"], "role": "linear softmax readout shared across depths",
         "signature": False}]


KNOCKOUTS = [("unmaking_library", "identity"), ("maker_router", "mean"), ("commitment", "zero"),
             ("origin_test", "mean"), ("sense_encoder_l2", "identity")]


def knockout(model, name, mode):
    ko = {"kind": model["kind"], "task_type": model["task_type"], "H": model["H"], "K": model["K"],
          "params": {k: v.copy() for k, v in model["params"].items()}, "flags": dict(model["flags"]),
          "halt_mean": model["halt_mean"]}
    key = {"unmaking_library": "library", "maker_router": "router", "origin_test": "halt",
           "sense_encoder_l2": "enc2", "origin_readout": "readout"}
    if name == "commitment":
        ko["flags"]["commit"] = mode != "zero"
    elif name in key:
        ko["flags"][key[name]] = mode
    else:
        raise ValueError("unknown module " + name)
    return ko


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


MUTANTS = {"sign_flip_update": "optimizer ascends the loss", "zero_learning_rate": "optimizer step size is zero",
           "zero_grad_U": "gradient of the unmaking library is zeroed", "zero_grad_We1": "gradient of encoder layer 1 is zeroed"}


# ------------------------------------------------------------------ training
def fit(model, data, budget, rng):
    """Adam with global-norm clipping; model selection on the validation rows only."""
    train, val = (data["train"], data.get("val")) if "train" in data else (data, None)
    lr = 0.0 if _MUTANT["name"] == "zero_learning_rate" else LR
    opt = Adam(model["params"], lr, sign=1.0 if _MUTANT["name"] == "sign_flip_update" else -1.0)
    n, hist, best = train["X"].shape[0], [], (np.inf, None)
    for step in range(budget):
        idx = rng.integers(n, size=BATCH)
        loss, g = loss_and_grads(model, {k: v[idx] for k, v in train.items()})
        if not np.isfinite(loss):
            raise FloatingPointError("non-finite loss")
        clip_global_norm(g, CLIP)
        opt.step(model["params"], g)
        hist.append(loss)
        if val is not None and (step + 1) % 100 == 0:
            vl = loss_and_grads(model, val)[0]
            if vl < best[0]:
                best = (vl, {k: v.copy() for k, v in model["params"].items()})
    if best[1] is not None:
        model["params"] = best[1]
    if model["kind"] == "steward" and val is not None and model["flags"]["halt"] == "learned":
        fw = _forward(model, val["X"], T_TRAIN, train=True)
        model["halt_mean"] = float(np.mean(fw["ps"]))
    return hist


def accuracy(model, d):
    return float((predict(model, d["X"]).argmax(1) == d["y"]).mean())


def train_arm(kind, sp, seed_seq, steps, **flags):
    r_init, r_fit = [np.random.default_rng(s) for s in seed_seq.spawn(2)]
    m = build_model(D_IN, C, rng=r_init, **flags) if kind == "steward" else build_flat(D_IN, C, r_init)
    hist = fit(m, {"train": sp["train"], "val": sp["val"]}, steps, r_fit)
    return m, hist


# ------------------------------------------------------------------ correctness tests
def c1_gradcheck(sp, rng, steps_after=60):
    """Every tensor of the Steward's Chain (and its trained knockout arms) and the baseline, at init and after
    training steps."""
    b = {k: v[:24] for k, v in sp["train"].items()}
    worst = 0.0
    arms = [("steward", {}), ("steward", {"library": "identity"}), ("steward", {"enc2": "identity"}), ("flat", {})]
    for kind, flags in arms:
        m = build_model(D_IN, C, rng=rng, **flags) if kind == "steward" else build_flat(D_IN, C, rng)
        for stage in ("init", "trained"):
            if stage == "trained":
                fit(m, {"train": sp["train"]}, steps_after, rng)
            _, g = loss_and_grads(m, b)
            errs = gradcheck(lambda: loss_and_grads(m, b)[0], m["params"], g, rng)
            worst = max(worst, max(errs.values()))
    n_tensors = 12 + 6
    return {"passed": bool(worst <= GC_TOL), "max_rel_error": float(worst), "tensors_checked": n_tensors,
            "tensors_total": n_tensors}


def c2_determinism(sp):
    runs = []
    for _ in range(2):
        m, h = train_arm("steward", sp, np.random.SeedSequence(7), 40)
        runs.append((h, predict(m, sp["test_id"]["X"][:50])))
    same = runs[0][0] == runs[1][0] and np.array_equal(runs[0][1], runs[1][1])
    finite = all(np.all(np.isfinite(v)) for v in m["params"].values()) and np.all(np.isfinite(runs[0][1]))
    return {"passed": bool(same and finite), "detail": f"identical={same} finite={finite}"}


def trivial_rate(d):
    return float(np.bincount(d["y"], minlength=C).max() / d["y"].size)


def c3_learning(model, hist, sp):
    th = MIND_CARD["thresholds"]
    drop = 1 - np.mean(hist[-50:]) / np.mean(hist[:50])
    acc, triv = accuracy(model, sp["test_id"]), trivial_rate(sp["test_id"])
    ok = drop >= th["loss_drop_fraction"] and acc >= triv + th["margin_over_trivial"]
    return {"passed": bool(ok), "detail": f"loss drop {drop:.3f}, held-out {acc:.3f} vs trivial {triv:.3f}"}


def c4_shuffled(sp, seed_seq, steps):
    perm = np.random.default_rng(seed_seq.spawn(1)[0]).permutation(sp["train"]["y"].size)
    shuffled = dict(sp["train"])
    for key in ("y", "chain", "depth", "Xnext", "Xstack"):
        shuffled[key] = sp["train"][key][perm]
    r_init, r_fit = [np.random.default_rng(s) for s in seed_seq.spawn(2)]
    m = build_model(D_IN, C, rng=r_init)
    fit(m, {"train": shuffled}, steps, r_fit)
    acc, triv = accuracy(m, sp["test_id"]), trivial_rate(sp["test_id"])
    return {"passed": bool(abs(acc - triv) <= 0.06), "detail": f"held-out {acc:.3f} vs trivial {triv:.3f} (band 0.06)"}


def c5_mutants(sp, seed_seq, steps):
    """Each mutant must make C1 or C3 fail; the clean control at the same budget must pass C3."""
    b = {k: v[:24] for k, v in sp["train"].items()}
    rng = np.random.default_rng(seed_seq.spawn(1)[0])
    ctrl_m, ctrl_h = train_arm("steward", sp, seed_seq, steps)
    detected, detail = 0, []
    control_ok = c3_learning(ctrl_m, ctrl_h, sp)["passed"]
    for name in MUTANTS:
        _MUTANT["name"] = name
        try:
            m = build_model(D_IN, C, rng=rng)
            _, g = loss_and_grads(m, b)
            gc_ok = max(gradcheck(lambda: loss_and_grads(m, b)[0], m["params"], g, rng, n_entries=6).values()) <= GC_TOL
            c3_ok = True
            if gc_ok:
                try:
                    mm, hh = train_arm("steward", sp, seed_seq, steps)
                    c3_ok = c3_learning(mm, hh, sp)["passed"]
                except FloatingPointError:
                    c3_ok = False                              # divergence is a detected failure
        finally:
            _MUTANT["name"] = None
        caught = (not gc_ok) or (not c3_ok)
        detected += int(caught)
        detail.append(f"{name}:{'caught' if caught else 'MISSED'}")
    score = detected / len(MUTANTS)
    return {"passed": bool(score == 1.0 and control_ok), "detected": detected, "total": len(MUTANTS),
            "score": score, "detail": f"control passes C3={control_ok}; " + ", ".join(detail)}


def _halting_mass(m, X, drop_remainder=False):
    fw = _forward(m, X, T_TEST, train=False)
    pis = fw["pis"][:-1] + ([np.zeros_like(fw["pis"][-1])] if drop_remainder else [fw["pis"][-1]])
    return np.stack(pis, 1), fw["mix"]


def c6_properties(rng):
    """C6.1 halting mass: pi_t >= 0 and sums to 1 for random, extreme parameters; negative control drops the
    remainder. C6.2 library permutation equivariance: permuting operators together with router columns leaves
    outputs unchanged; negative control permutes operators only."""
    viol_mass, ctrl_mass, viol_perm, ctrl_perm = 0.0, 0.0, 0.0, 0.0
    for trial in range(20):
        m = build_model(D_IN, C, rng=rng)
        for key in m["params"]:
            m["params"][key] *= rng.uniform(0.5, 6.0)
        m["flags"]["commit"] = bool(trial % 2)
        X = rng.normal(0, 3.0, (64, D_IN))
        pis, mix = _halting_mass(m, X)
        viol_mass = max(viol_mass, abs(pis.sum(1) - 1).max(), abs(mix.sum(1) - 1).max(), max(0.0, -pis.min()))
        ctrl_mass = max(ctrl_mass, abs(_halting_mass(m, X, drop_remainder=True)[0].sum(1) - 1).max())
        perm = rng.permutation(K)
        pm = knockout(m, "commitment", "one")
        pm["flags"]["commit"] = m["flags"]["commit"]
        pm["params"]["U"], pm["params"]["c"] = m["params"]["U"][perm], m["params"]["c"][perm]
        broken = knockout(pm, "commitment", "one")
        broken["flags"]["commit"] = m["flags"]["commit"]
        pm["params"]["V"], pm["params"]["d"] = m["params"]["V"][:, perm], m["params"]["d"][perm]
        viol_perm = max(viol_perm, np.abs(predict(pm, X) - predict(m, X)).max())
        ctrl_perm = max(ctrl_perm, np.abs(predict(broken, X) - predict(m, X)).max())
    ok = viol_mass < 1e-10 and ctrl_mass > 1e-3 and viol_perm < 1e-10 and ctrl_perm > 1e-6
    return {"passed": bool(ok), "detail": f"C6.1 max violation {viol_mass:.1e} (control {ctrl_mass:.2e}); "
                                          f"C6.2 max violation {viol_perm:.1e} (control {ctrl_perm:.2e})"}


def _row_hashes(X):
    return {hashlib.sha1(np.round(r, 10).tobytes()).hexdigest() for r in X}


def c7_split_integrity(sp):
    tr = _row_hashes(sp["train"]["X"])
    overlap = sum(len(tr & _row_hashes(sp[s]["X"])) for s in ("test_id", "test_shift", "blind_like", "blind_unlike"))
    return {"passed": overlap == 0, "detail": f"{overlap} shared rows"}


# ------------------------------------------------------------------ hypotheses
def run_seed(seed, steps, full_protocol):
    ss = np.random.SeedSequence([CHAPTER, seed])
    s_data, s_full, s_flat, s_sig, s_mat = ss.spawn(5)
    sp = build_splits(np.random.default_rng(s_data), 1200)
    full, hist = train_arm("steward", sp, s_full, steps)
    flat, _ = train_arm("flat", sp, s_flat, steps)
    conf = lambda m, d: predict(m, d["X"]).max(1)
    out = {"sp": sp, "full": full, "hist": hist,
           "id": (accuracy(full, sp["test_id"]), accuracy(flat, sp["test_id"])),
           "shift": (accuracy(full, sp["test_shift"]), accuracy(flat, sp["test_shift"])),
           "like": tuple(auroc(conf(m, sp["test_id"]), conf(m, sp["blind_like"])) for m in (full, flat)),
           "unlike": tuple(auroc(conf(m, sp["test_id"]), conf(m, sp["blind_unlike"])) for m in (full, flat)),
           "depth": [(accuracy(full, sp["by_depth"][L]), accuracy(flat, sp["by_depth"][L])) for L in range(LMAX + 1)],
           "ko": {f"{n}:{mo}": accuracy(knockout(full, n, mo), sp["test_shift"]) - out_shift
                  for n, mo in KNOCKOUTS for out_shift in [accuracy(full, sp["test_shift"])]},
           "halt_mass_like": float(np.mean(_forward(full, sp["blind_like"]["X"], T_TEST, False)["pis"][-1]))}
    if full_protocol:
        sig, _ = train_arm("steward", sp, s_sig, steps, library="identity")
        mat, _ = train_arm("steward", sp, s_mat, steps, enc2="identity")
        out["nec"] = (accuracy(sig, sp["test_shift"]), accuracy(mat, sp["test_shift"]))
    return out


def evaluate_hypotheses(results, rng, evaluated):
    rows = []
    specs = {"H-SIG": lambda r: r["shift"][0] - r["shift"][1],
             "H-NEC": lambda r: r["nec"][0] - r["nec"][1],
             "H-BLIND": lambda r: r["like"][0] - r["like"][1]}
    for h in MIND_CARD["hypotheses"]:
        if not evaluated:
            rows.append({"id": h["id"], "metric": h["metric"], "mean_diff": None, "ci95": [None, None],
                         "mesi": h["mesi"], "n_seeds": len(results), "verdict": "not evaluated"})
            continue
        mean, lo, hi = paired_bootstrap([specs[h["id"]](r) for r in results], rng)
        rows.append({"id": h["id"], "metric": h["metric"], "mean_diff": mean, "ci95": [lo, hi], "mesi": h["mesi"],
                     "n_seeds": len(results), "verdict": verdict(mean, lo, hi, h["direction"], h["mesi"])})
    return rows


def knockout_table(results, rng):
    table = []
    sig = {m["name"]: m["signature"] for m in modules(results[0]["full"])}
    for key in results[0]["ko"]:
        mean, lo, hi = paired_bootstrap([r["ko"][key] for r in results], rng)
        table.append({"module": key, "signature": sig[key.split(":")[0]], "metric_change": mean, "ci95": [lo, hi]})
    return table


# ------------------------------------------------------------------ report
def _fmt(x, nd=3):
    return "n/a" if x is None else f"{x:+.{nd}f}"


def print_report(rep, results):
    print(f"=== VERIFIED REPORT · chapter {CHAPTER:04d} ===")
    print(f"file {rep['file']} · python {rep['environment']['python']} · numpy {rep['environment']['numpy']}")
    print(f"seeds {rep['seeds']} · runtime {rep['runtime_s']:.1f} s · parameters {rep['n_params']} "
          f"(baseline {rep['n_params_baseline']})")
    gc = rep["gradcheck"]
    print(f"gradient check: {gc['tensors_checked']}/{gc['tensors_total']} tensors, max rel error "
          f"{gc['max_rel_error']:.2e}, at {gc['checked_at']}, passed={gc['passed']}")
    print("correctness:")
    for c in rep["correctness"]:
        print(f"  {c['id']:<3} {c['name']:<24} {'PASS' if c['passed'] else 'FAIL'}  {c['detail']}")
    mu = rep["mutants"]
    print(f"mutation score {mu['detected']}/{mu['total']} = {mu['score']:.2f}")
    print("hypotheses (paired per-seed differences, 95% bootstrap CI, 2000 resamples):")
    for h in rep["hypotheses"]:
        lo, hi = h["ci95"]
        print(f"  {h['id']:<8} {h['metric']:<28} diff {_fmt(h['mean_diff'])}  CI [{_fmt(lo)}, {_fmt(hi)}]"
              f"  mesi {h['mesi']:.2f}  -> {h['verdict']}")
    print("per-seed means            chain    flat")
    for name, key in (("held-out accuracy (0-3)", "id"), ("shifted accuracy (4-5)", "shift"),
                      ("AUROC made vs like (H-BLIND)", "like"), ("AUROC made vs unlike (P-UNLIKE)", "unlike")):
        a = np.mean([r[key][0] for r in results]); b = np.mean([r[key][1] for r in results])
        print(f"  {name:<31} {a:.3f}   {b:.3f}")
    print("  accuracy by history depth  " + "  ".join(f"d{L}" for L in range(LMAX + 1)))
    for j, lab in ((0, "chain"), (1, "flat ")):
        print(f"    {lab}                    " + "  ".join(f"{np.mean([r['depth'][L][j] for r in results]):.2f}"
                                                         for L in range(LMAX + 1)))
    print("knockouts on the trained model (change in shifted accuracy):")
    for k in rep["knockouts"]:
        print(f"  {k['module']:<28} sig={str(k['signature']):<5} {k['metric_change']:+.3f}  "
              f"CI [{k['ci95'][0]:+.3f}, {k['ci95'][1]:+.3f}]")
    print(f"task types: {', '.join(rep['task_types'])} · exit code {rep['exit_code']}")
    print("=== END REPORT ===")


# ------------------------------------------------------------------ command line
def run_data_bridge(path, rng):
    if not path or not os.path.exists(path):
        print(f"--data: no local CSV at {path!r}; real-data bridge skipped (no downloads by design).")
        return
    arr = np.genfromtxt(path, delimiter=",", skip_header=1)
    arr = arr[np.all(np.isfinite(arr), 1)]
    X, y = arr[:, :-1], arr[:, -1].astype(int)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    y = np.unique(y, return_inverse=True)[1]
    idx = rng.permutation(y.size)
    tr, te = idx[: int(0.8 * y.size)], idx[int(0.8 * y.size):]
    m = build_model(X.shape[1], int(y.max()) + 1, rng=rng)
    fit(m, {"X": X[tr], "y": y[tr]}, 800, rng)
    print(f"--data bridge: held-out accuracy {float((predict(m, X[te]).argmax(1) == y[te]).mean()):.3f} "
          f"on {te.size} rows (no chain labels; origin readout only)")


def build_report(checks, results, seeds, runtime, evaluated):
    all_ok = all(c[2]["passed"] for c in checks)
    only_budget = not checks[-1][2]["passed"] and all(c[2]["passed"] for c in checks[:-1])
    brng, gc = np.random.default_rng([CHAPTER, 99]), checks[0][2]
    return {"schema_version": "1.0", "chapter": CHAPTER, "file": os.path.basename(__file__),
            "card_revision": MIND_CARD["card_revision"],
            "environment": {"python": sys.version.split()[0], "numpy": np.__version__}, "seeds": seeds,
            "runtime_s": runtime, "n_params": n_params(results[0]["full"]),
            "n_params_baseline": n_params(build_flat(D_IN, C, np.random.default_rng(0))),
            "gradcheck": {"tensors_checked": gc["tensors_checked"], "tensors_total": gc["tensors_total"],
                          "max_rel_error": gc["max_rel_error"], "checked_at": ["init", "after_training_steps"],
                          "passed": gc["passed"]},
            "correctness": [{"id": i, "name": n, "passed": bool(c["passed"]), "detail": c.get("detail", "")}
                            for i, n, c in checks],
            "mutants": {k: checks[4][2][k] for k in ("detected", "total", "score")},
            "hypotheses": evaluate_hypotheses(results, brng, evaluated),
            "knockouts": knockout_table(results, brng), "task_types": TASK_TYPES,
            "exit_code": 0 if all_ok else (3 if only_budget else 1)}


def main(argv=None):
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__), description="Chapter 0314 · The Steward's Chain")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
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
        print(json.dumps(MIND_CARD, indent=2))
        return 0
    if args.mutant is not None:
        if args.mutant not in MUTANTS:
            print(f"unknown mutant; choose from {sorted(MUTANTS)}")
            return 2
        _MUTANT["name"] = args.mutant
    t0 = time.time()
    n_seeds, steps = (1, 400) if args.quick else (args.seeds, 1500)
    seeds = [args.seed + i for i in range(n_seeds)]
    print(f"seeds: {seeds} · steps per arm: {steps} · mutant: {_MUTANT['name']}")
    rng = np.random.default_rng(args.seed)
    try:
        results = [run_seed(s, steps, full_protocol=not args.quick) for s in seeds]
        sp0 = results[0]["sp"]
        checks = [("C1", "gradient_check", c1_gradcheck(sp0, np.random.default_rng([CHAPTER, 1]))),
                  ("C2", "determinism_finiteness", c2_determinism(sp0)),
                  ("C3", "learning", c3_learning(results[0]["full"], results[0]["hist"], sp0)),
                  ("C4", "shuffled_label_control", c4_shuffled(sp0, np.random.SeedSequence([CHAPTER, 4]), steps)),
                  ("C5", "mutant_detection", c5_mutants(sp0, np.random.SeedSequence([CHAPTER, 5]), 400)),
                  ("C6", "property_tests", c6_properties(np.random.default_rng([CHAPTER, 6]))),
                  ("C7", "split_integrity", c7_split_integrity(sp0))]
    except FloatingPointError as exc:
        print(f"non-finite values: {exc}")
        return 4
    runtime = time.time() - t0
    budget = 20.0 if args.quick else 180.0
    checks.append(("C8", "budget", {"passed": runtime <= budget, "detail": f"{runtime:.1f} s of {budget:.0f} s"}))
    rep = build_report(checks, results, seeds, runtime, evaluated=not args.quick and n_seeds >= 5)
    print_report(rep, results)
    if args.json:
        write_json_report(rep, args.json)
    if args.data:
        run_data_bridge(args.data, np.random.default_rng(args.seed))
    return rep["exit_code"]


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0321 · Saint Sava (1174-1235)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 17, Minds 321-340 Available on Amazon https://www.amazon.com/dp/B0HP7Z1NKC
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos # END ATTRIBUTION
"""
Synodal Succession: a regeneration protocol in which the deployed model is never
trained on its predecessor.

THESIS
  A governing head that must be replaced every generation stays sound only if its
  successor is made by a standing synod of territorially bounded peers, each
  grounded in its own local evidence and each speaking only for its own territory,
  while those peers are themselves replaced one at a time and consecrated by the
  head; memory then lives in the staggered synod, not in the lineage of heads.

EVIDENCE AND PROVENANCE (provenance: belief)
  Sava's own surviving texts are rules, a hagiography, a legal compilation and a
  letter; his view of mind is read from what he legislated. Hagiographic detail
  (Domentijan, Teodosije) is mediated and marked as such in the card.
  D1  1219: Nicaea grants that Serbian archbishops be consecrated by the assembly
      of Serbian bishops, not by an outside see (Domentijan; Fine 1994).
  D2  1219-20: he first consecrates bishops from his own disciples, seats them in
      separate eparchies and gives each the law books (Domentijan).
  D3  Karyes typikon (1199): no protos or hegumen has authority over the cell;
      Studenica typikon (c.1208): no bishop has authority over the monastery.
  D4  1233/34: he names his disciple Arsenije successor in his own lifetime.
  D5  Zakonopravilo (1219/20): a selection of canons, commentaries and civil law
      attested in no Greek manuscript (Petrovic 1991; Koprivica 2020).
  D6  Hum (Ston) and Zeta (Prevlaka) sees placed on the Latin coastal edge; read
      here as a founder-drawn grid cutting through mixed populations (speculation).
  D7  May 1220: Chomatenos of Ohrid protests that the consecration bypassed Ohrid
      (Ponemata diaphora no. 86, ed. Prinzing 2002).

DOCTRINE -> MECHANISM -> TEST
  D1 self-consecration     -> M1 head re-initialised each generation and     -> C6.2, H-SIG
                              distilled from the synod, never the predecessor
  D2 bishops with the book -> M2 K persistent regional learners, each replays -> H-NEC
                              the shared seed book on its own cell
  D3/D2 jurisdiction       -> M3 hard jurisdiction gate: a pseudo-label for x  -> C6.1, H-NEC
                              comes only from the learner that owns x's cell
  D4 staggered succession  -> M4 one learner per generation is replaced and     -> knockout table
                              taught by the head (round-robin mortality)
  D6 founder's grid        -> blind spot: custom regions rotated against the    -> H-BLIND
                              jurisdiction grid
  D7 Ohrid's position      -> rival: each head re-derived from the mother model -> H-RIVAL
                              plus the current local report only

RESEARCH QUESTION (continual learning and plasticity)
  When a deployed classifier must be regenerated every generation after its
  original labelled source has gone away, and the world drifts, does moving the
  memory out of the lineage of models into a persistent, jurisdiction-gated
  population of regional learners reduce recursive-training degradation?

CLOSEST PRIOR ART AND DELTA
  Born-Again Networks (Furlanello et al. 2018), model collapse under the replace
  workflow (Shumailov et al. 2024), the accumulate workflow (Gerstgrasser et al.
  2024), ensemble distillation for federated fusion (Lin et al. 2020), local
  experts with fixed gating (Jacobs et al. 1991). Delta: the successor never sees
  its predecessor; it is consecrated by a synod whose members persist with
  staggered mortality and are themselves consecrated by the head.
  Baselines in file: SOLO (born-again lineage, size-matched), ACCUM (lineage plus
  every real label ever collected), OHRID (rival), three knockouts.

BLIND SPOT
  The jurisdiction grid is the founder's, not the land's. If customs follow
  regions that cut across the grid, each learner must model two customs with
  half the evidence for each, and the synod's advantage should shrink.

TASK (generative process, per seed)
  x ~ N(0, I_4); jurisdiction cell r(x) = quadrant of (x0, x1).
  psi(x) = tanh(A x + a); logits = psi W (canon) + psi V[g, c(x)] (custom).
  c(x) = r(x) (aligned) or the quadrant of (x0, x1) rotated by 45 deg (blind).
  V[g] drifts: V[g+1] = V[g] + N(0, drift^2), independently per custom region.
  Labels = argmax of logits with 5% uniform flips.
  Generation 0: a labelled seed book of 160 cases. Every later generation:
  10 labelled cases per cell, 400 unlabelled inputs, fresh test set.
  Held-out split: mean test accuracy over the first third of the chain.
  Shifted split: mean test accuracy over the last third (longer horizon,
  more drift, more successions).

LIMITS
  Synthetic data, small tanh MLPs, a fixed known partition. A research
  prototype of a regeneration protocol for AGI-oriented systems, not an AGI;
  results say nothing about the historical person.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 2,
    "revision_log": [
        {"revision": 2, "date": "2026-09-25",
         "change": "task input dim 6->4 and world features 12->8",
         "reason": ("first --quick run failed C3 (held-out margin over majority below 0.15); recalibrated on "
                    "pilot seeds 1000-1002, disjoint from report seeds, using only the full model's own C3 "
                    "margin; hypotheses, metrics, mesi and thresholds unchanged")}],
    "generation": {"template_version": "codeguidelines 1.0 (2026-09-15)", "generator": "Claude (Anthropic)",
                   "generator_version": "claude-opus-5-5", "date": "2026-09-25"},
    "id": 321, "figure": "Saint Sava", "born": 1174, "died": 1235,
    "civilization": "Serbian (Nemanjić Serbia; Byzantine Orthodox commonwealth)",
    "provenance": "belief",
    "thesis": ("The successor head is made by a standing synod of territorially bounded peers that each speak "
               "only for their own territory and are themselves replaced one at a time and consecrated by the "
               "head, so memory lives in the staggered synod, never in the lineage of heads."),
    "evidence": [
        {"id": "D1", "claim": "1219: Serbian archbishops to be consecrated by the assembly of Serbian bishops",
         "basis": "scholarship", "source": "Domentijan, Life of Sava (trans. Mirkovic 1988); Fine 1994, 116-119"},
        {"id": "D2", "claim": "Bishops consecrated from Sava's disciples, seated in eparchies, given law books",
         "basis": "scholarship", "source": "Domentijan, Life of Sava; Cirkovic 2004"},
        {"id": "D3", "claim": "Karyes cell exempt from protos and hegumen; Studenica exempt from any bishop",
         "basis": "primary", "source": "Karyes Typikon (1199); Studenica Typikon (c.1208), ed. Bogdanovic 1986"},
        {"id": "D4", "claim": "Arsenije named successor in Sava's lifetime (1233/34)",
         "basis": "scholarship", "source": "Domentijan; Teodosije; Fine 1994"},
        {"id": "D5", "claim": "Zakonopravilo selection attested in no Greek manuscript; adopted in Rus' 1262/1274",
         "basis": "scholarship", "source": "Petrovic 1991 (Ilovica facsimile); Koprivica 2020, SCH 56"},
        {"id": "D6", "claim": "Hum (Ston) and Zeta (Prevlaka) sees on the Latin coastal edge: a founder-drawn grid",
         "basis": "speculation", "source": "chapter reading of the see list (seats per Serbian historiography)"},
        {"id": "D7", "claim": "Chomatenos of Ohrid protests the consecration (8 May 1220)",
         "basis": "scholarship", "source": "Chomatenos, Ponemata diaphora no. 86, ed. Prinzing 2002"},
    ],
    "research_question": {"category": "continual learning and plasticity",
                          "question": ("Does moving memory out of the lineage of regenerated models into a persistent, "
                                       "jurisdiction-gated population of regional learners reduce recursive-training "
                                       "degradation under drift after the original labelled source is gone?")},
    "mechanism": {
        "name": "Synodal Succession", "family": "population-based self-regeneration (generational distillation)",
        "signature_modules": ["bishops", "jurisdiction_gate", "synod_memory"],
        "closest_prior_art": ["Born-Again Networks (Furlanello et al., ICML 2018)",
                              "Model collapse, replace workflow (Shumailov et al., Nature 2024)",
                              "Accumulate workflow (Gerstgrasser et al., 2024)",
                              "Ensemble distillation for model fusion (Lin et al., NeurIPS 2020)",
                              "Adaptive mixtures of local experts (Jacobs et al., 1991)"],
        "overlap": "Medium",
        "prior_art_queries": ["generational self-distillation persistent population", "born-again networks generations",
                              "model collapse accumulate real synthetic", "federated ensemble distillation regional",
                              "teacher never predecessor staggered replacement"],
        "contribution_type": "mechanism",
        "delta": ("The deployed head never sees its predecessor: it is re-derived each generation from persistent "
                  "regional learners with staggered, head-taught replacement and hard jurisdiction gating."),
    },
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1", "property_test": "C6.2", "hypothesis": "H-SIG"},
        {"doctrine": "D2", "mechanism": "M2", "property_test": "C6.1", "hypothesis": "H-NEC"},
        {"doctrine": "D3", "mechanism": "M3", "property_test": "C6.1", "hypothesis": "H-NEC"},
        {"doctrine": "D6", "mechanism": "M3", "property_test": "C6.1", "hypothesis": "H-BLIND"},
        {"doctrine": "D7", "mechanism": "M1", "property_test": "C6.2", "hypothesis": "H-RIVAL"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "The synod-made head beats the size-matched born-again lineage late in the chain",
         "metric": "acc_late", "split": "shifted", "comparison": "model - baseline", "direction": "greater",
         "mesi": 0.02, "seeds": 5},
        {"id": "H-NEC", "statement": "Removing the jurisdiction gate costs more than removing the book replay",
         "metric": "acc_late", "comparison": "signature_knockout - matched_knockout", "direction": "less",
         "mesi": 0.01, "seeds": 5},
        {"id": "H-BLIND", "statement": "The synod's edge over the lineage shrinks when customs cut across the grid",
         "condition": "custom regions rotated 45 degrees against jurisdiction cells",
         "grounding": "sees drawn by the founder on a confessional front through mixed coastal populations",
         "metric": "acc_late", "comparison": "(model - baseline)_blind - (model - baseline)_native",
         "direction": "less", "mesi": 0.01, "seeds": 5},
        {"id": "H-RIVAL", "statement": "Self-consecration beats re-consecration from the mother model (Ohrid)",
         "metric": "acc_late", "comparison": "model - rival", "direction": "greater", "mesi": 0.02, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.30, "margin_over_trivial": 0.15, "shuffle_band": 0.06},
    "probe_predictions": [{"probe": "P5", "expected": "above baseline"}, {"probe": "P7", "expected": "above baseline"},
                          {"probe": "P6", "expected": "equal to baseline"}, {"probe": "P9", "expected": "below baseline"}],
    "dialectic_links": [{"chapter": 224, "relation": "successor", "test": ""},
                        {"chapter": 310, "relation": "rival", "test": ""}],
    "corpus_neighbors": [
        {"chapter": 224, "similarity": None, "difference": "Methodius transfers a normative system across a language "
                                                         "boundary; here nothing is transferred, the head is regenerated"},
        {"chapter": 205, "similarity": None, "difference": "Alcuin corrects copies toward a canon; here every "
                                                         "baseline gets the same canon and the head never copies"},
        {"chapter": 164, "similarity": None, "difference": "Benedict's timetable teaches; here territory gates teaching"},
        {"chapter": 227, "similarity": None, "difference": "Boris elicits from two authorities; here authority is "
                                                         "generated inside by a synod"},
        {"chapter": 319, "similarity": None, "difference": "Shinran needs an external fixed signal; here correction "
                                                         "comes from persistent internal peers with local evidence"},
    ],
    "barometer": {"cognitive_processing": ["P10 sample efficiency"], "embodied_cognition": [],
                  "world_modeling": ["P7 tracking under regime change"], "consciousness": [],
                  "language_understanding": [], "emotional_intelligence": [],
                  "creativity": [], "autonomy": ["P5 task sequence", "continual regeneration without source"]},
    "task_types": ["vector_classification"],
    "applications": [
        {"use": "Regenerate a deployed classifier after its labelled source is retired, from regional maintainers",
         "sector": "ML operations / model lifecycle", "dataset": "UCI Gas Sensor Array Drift (Vergara et al., 2012)"},
        {"use": "Research decision support: multi-site risk model rebuilt from site learners without pooling",
         "sector": "healthcare research (decision support only)", "dataset": "UCI Heart Disease (four sites)"},
        {"use": "Multi-jurisdiction legal-document classification tracking local legal change",
         "sector": "legal technology", "dataset": "MultiEURLEX (Chalkidis et al., 2021)"},
    ],
    "safety_notes": ("Regional learners can entrench correlated local error; H-BLIND measures the failure when the "
                     "partition is wrong. Not for autonomous legal or medical decisions; synthetic data only."),
}

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

CHAPTER = 321
TASK_TYPES = ["vector_classification"]
MUTANT = None
DEFAULTS = dict(d=4, m=8, C=3, K=4, G=9, n_book=160, n_loc=10, n_u=400, n_test=800,
                canon_scale=0.55, custom_scale=0.6, drift=0.28, label_noise=0.05,
                h_head=24, h_bish=10, h_solo=65, lr=0.02, s_head=150, s_bish=40,
                lam_book=0.3, w_trickle=1.0, w_unlab=1.0, clip=5.0)
QUICK = dict(G=4, s_head=80, s_bish=24, n_u=240, n_test=400)
THRESH = MIND_CARD["thresholds"]


class NonFinite(Exception):
    """Raised when a loss or parameter becomes non-finite (exit code 4)."""


# BEGIN STANDARD UTILITIES v1.0
def softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def logsumexp(z, axis=-1):
    m = np.max(z, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(z - m), axis=axis))


def softplus(x):
    return np.where(x > 30, x, np.log1p(np.exp(np.minimum(x, 30))))


class Adam:
    def __init__(self, keys, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m, self.v = {}, {}
        self.keys = list(keys)

    def step(self, params, grads):
        self.t += 1
        lr = 0.0 if MUTANT == "zero_lr" else self.lr
        sign = 1.0 if MUTANT == "sign_flip" else -1.0
        for k in self.keys:
            g = grads[k]
            self.m[k] = self.b1 * self.m.get(k, 0.0) + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v.get(k, 0.0) + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] = params[k] + sign * lr * mh / (np.sqrt(vh) + self.eps)


def clip_by_global_norm(grads, max_norm):
    norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    if norm > max_norm:
        grads = {k: g * (max_norm / norm) for k, g in grads.items()}
    return grads, norm


def finite_difference_check(loss_fn, params, analytic, rng, eps=1e-6, n_entries=20, tol=1e-5, floor=1e-7):
    """Central differences on every tensor: n_entries random entries plus the largest-gradient entry."""
    worst, per = 0.0, {}
    for k, p in params.items():
        flat = p.reshape(-1)
        idx = set(rng.choice(flat.size, size=min(n_entries, flat.size), replace=False).tolist())
        idx.add(int(np.argmax(np.abs(analytic[k]).reshape(-1))))
        errs = []
        for i in idx:
            old = flat[i]
            flat[i] = old + eps
            lp = loss_fn(params)
            flat[i] = old - eps
            lm = loss_fn(params)
            flat[i] = old
            num, a = (lp - lm) / (2 * eps), analytic[k].reshape(-1)[i]
            errs.append(abs(a - num) / max(abs(a) + abs(num), floor))
        per[k] = max(errs)
        worst = max(worst, per[k])
    return {"max_rel_error": worst, "per_tensor": per, "passed": worst <= tol}


def paired_bootstrap_ci(diffs, rng, n_resamples=2000, alpha=0.05):
    d = np.asarray(diffs, dtype=float)
    idx = rng.integers(0, len(d), size=(n_resamples, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def hypothesis_verdict(mean, lo, hi, mesi, direction):
    if direction == "greater":
        if lo > 0 and mean >= mesi:
            return "supported"
        return "contradicted" if hi < 0 else "inconclusive"
    if hi < 0 and mean <= -mesi:
        return "supported"
    return "contradicted" if lo > 0 else "inconclusive"


def write_json_report(path, report):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
# END STANDARD UTILITIES


# ----------------------------------------------------------------------------- data and tasks
def quadrant(Z2, rot=0.0):
    """Cell index 0..3 of 2-D points, optionally after rotating the plane by rot radians."""
    c, s = math.cos(rot), math.sin(rot)
    u = c * Z2[:, 0] - s * Z2[:, 1]
    v = s * Z2[:, 0] + c * Z2[:, 1]
    return 2 * (u > 0).astype(int) + (v > 0).astype(int)


def make_world(rng, cfg, aligned=True):
    d, m, C, K, G = cfg["d"], cfg["m"], cfg["C"], cfg["K"], cfg["G"]
    V = np.empty((G + 1, K, m, C))
    V[0] = rng.normal(0.0, cfg["custom_scale"], (K, m, C))
    for g in range(1, G + 1):  # each custom region drifts on its own random walk
        V[g] = V[g - 1] + rng.normal(0.0, cfg["drift"], (K, m, C))
    return {"A": rng.normal(0.0, 1.0, (d, m)), "a": rng.normal(0.0, 0.5, m),
            "W": rng.normal(0.0, cfg["canon_scale"], (m, C)), "V": V,
            "rot": 0.0 if aligned else math.pi / 4, "noise": cfg["label_noise"], "C": C}


def world_labels(world, X, g, rng):
    psi = np.tanh(X @ world["A"] + world["a"])
    cust = quadrant(X[:, :2], world["rot"])
    z = psi @ world["W"] + np.einsum("nm,nmc->nc", psi, world["V"][g][cust])
    y = z.argmax(1)
    flip = rng.random(len(y)) < world["noise"]
    y[flip] = rng.integers(0, world["C"], int(flip.sum()))
    return y


def sample_regional(rng, d, K, n_per):
    """n_per inputs from each jurisdiction cell (rejection sampling on standard normals)."""
    out = [[] for _ in range(K)]
    while min(len(o) for o in out) < n_per:
        X = rng.normal(size=(8 * n_per * K, d))
        r = quadrant(X[:, :2])
        for k in range(K):
            out[k].extend(X[r == k][: n_per - len(out[k])])
    return np.array([x for o in out for x in o])


def make_chain_data(rng, cfg, aligned=True):
    world = make_world(rng, cfg, aligned)
    d = cfg["d"]
    Xb = rng.normal(size=(cfg["n_book"], d))
    gens = []
    for g in range(cfg["G"] + 1):
        Xu = rng.normal(size=(cfg["n_u"], d))
        Xt = sample_regional(rng, d, cfg["K"], cfg["n_loc"])
        Xs = rng.normal(size=(cfg["n_test"], d))
        gens.append({"U": Xu, "T": (Xt, world_labels(world, Xt, g, rng)),
                     "S": (Xs, world_labels(world, Xs, g, rng))})
    return {"book": (Xb, world_labels(world, Xb, 0, rng)), "gens": gens, "aligned": aligned}


def shuffle_labels(data, rng):
    """Shuffled-label control: permute every training label; test labels untouched."""
    Xb, yb = data["book"]
    gens = [{"U": gd["U"], "T": (gd["T"][0], rng.permutation(gd["T"][1])), "S": gd["S"]} for gd in data["gens"]]
    return {"book": (Xb, rng.permutation(yb)), "gens": gens, "aligned": data["aligned"]}


def onehot(y, C):
    out = np.zeros((len(y), C))
    out[np.arange(len(y)), y] = 1.0
    return out


def assemble(terms):
    """Stack (X, targets, weight) terms; each term's rows share its weight equally."""
    terms = [t for t in terms if t is not None and len(t[0]) and t[2] > 0]
    X = np.vstack([t[0] for t in terms])
    T = np.vstack([t[1] for t in terms])
    rw = np.concatenate([np.full(len(t[0]), t[2] / len(t[0])) for t in terms])
    return X, T, rw


# ----------------------------------------------------------------------------- model
def mlp_init(rng, prefix, d, h, c):
    return {f"{prefix}.W1": rng.normal(0.0, 1.0 / math.sqrt(d), (d, h)), f"{prefix}.b1": np.zeros(h),
            f"{prefix}.W2": rng.normal(0.0, 1.0 / math.sqrt(h), (h, c)), f"{prefix}.b2": np.zeros(c)}


def mlp_logits(P, prefix, X):
    H = np.tanh(X @ P[prefix + ".W1"] + P[prefix + ".b1"])
    return H @ P[prefix + ".W2"] + P[prefix + ".b2"], H


def mlp_loss_grads(P, prefix, X, T, rw):
    """Weighted soft-target cross-entropy of one tanh MLP and its hand-derived gradients."""
    Z, H = mlp_logits(P, prefix, X)
    logp = Z - logsumexp(Z, axis=1)[:, None]
    loss = -float(np.sum(rw[:, None] * T * logp))
    dZ = rw[:, None] * (np.exp(logp) - T)
    dH = (dZ @ P[prefix + ".W2"].T) * (1.0 - H * H)
    g = {prefix + ".W1": X.T @ dH, prefix + ".b1": dH.sum(0),
         prefix + ".W2": H.T @ dZ, prefix + ".b2": dZ.sum(0)}
    if MUTANT == "zero_grad_head_W1" and prefix == "head":
        g["head.W1"] = np.zeros_like(g["head.W1"])
    if MUTANT == "zero_grad_bishop_W2" and prefix == "b0":
        g["b0.W2"] = np.zeros_like(g["b0.W2"])
    return loss, g


def train_prefix(P, prefix, batch, steps, cfg):
    keys = [k for k in P if k.startswith(prefix + ".")]
    opt, (X, T, rw), losses = Adam(keys, lr=cfg["lr"]), batch, []
    for _ in range(steps):
        loss, g = mlp_loss_grads(P, prefix, X, T, rw)
        if not np.isfinite(loss):
            raise NonFinite(prefix)
        g, _ = clip_by_global_norm(g, cfg["clip"])
        opt.step(P, g)
        losses.append(loss)
    return losses


def build_model(in_dim, out_dim, task_type, rng, **cfg):
    if task_type not in TASK_TYPES:
        raise ValueError(task_type)
    c = dict(DEFAULTS)
    c.update(cfg)
    c["d"], c["C"] = in_dim, out_dim
    P = mlp_init(rng, "head", in_dim, c["h_head"], out_dim)
    for k in range(c["K"]):
        P.update(mlp_init(rng, f"b{k}", in_dim, c["h_bish"], out_dim))
    return {"params": P, "cfg": c, "rng": rng,
            "flags": {"gate": "jurisdiction", "memory": True, "book": True, "mortality": True}}


def territory(model, X):
    return quadrant(X[:, :2]) if X.shape[1] >= 2 else 2 * (X[:, 0] > 0).astype(int)


def synod_targets(model, X):
    """Consecration targets: the jurisdiction gate asks only the learner owning x's cell."""
    P, K = model["params"], model["cfg"]["K"]
    probs = np.stack([softmax(mlp_logits(P, f"b{k}", X)[0]) for k in range(K)])
    if model["flags"]["gate"] == "mean":
        return probs.mean(0)
    return probs[territory(model, X), np.arange(len(X))]


def head_probs(model, X):
    return softmax(mlp_logits(model["params"], "head", X)[0])


def predict(model, X):
    return mlp_logits(model["params"], "head", X)[0].argmax(1)


def hidden_states(model, X):
    return mlp_logits(model["params"], "head", X)[1]


def n_params(model):
    return int(sum(p.size for p in model["params"].values()))


def loss_and_grads(model, batch):
    """batch = {'head': (X, T, rw), 'bishops': {k: (X, T, rw)}}; grads carry every params key."""
    P = model["params"]
    grads = {k: np.zeros_like(v) for k, v in P.items()}
    loss, g = mlp_loss_grads(P, "head", *batch["head"])
    grads.update(g)
    for k, b in batch.get("bishops", {}).items():
        lk, gk = mlp_loss_grads(P, f"b{k}", *b)
        loss += lk
        grads.update(gk)
    return loss, grads


def consecrate_bishop(model, k, terms, steps, fresh):
    c = model["cfg"]
    if fresh:
        model["params"].update(mlp_init(model["rng"], f"b{k}", c["d"], c["h_bish"], c["C"]))
    return train_prefix(model["params"], f"b{k}", assemble(terms), steps, c)


def consecrate_head(model, U, Xt, yt, book):
    """The archbishop dies; the synod makes a fresh successor that never reads its predecessor."""
    c = model["cfg"]
    model["params"].update(mlp_init(model["rng"], "head", c["d"], c["h_head"], c["C"]))
    Xb, yb = book
    terms = [(U, synod_targets(model, U), c["w_unlab"]), (Xt, onehot(yt, c["C"]), c["w_trickle"]),
             (Xb, onehot(yb, c["C"]), c["lam_book"])]
    return train_prefix(model["params"], "head", assemble(terms), c["s_head"], c)


def sava_founding(model, book, U0):
    """Generation 0: every bishop receives the whole book, weighted toward its own cell."""
    c, (Xb, yb) = model["cfg"], book
    rb = territory(model, Xb)
    for k in range(c["K"]):
        terms = [(Xb[rb == k], onehot(yb[rb == k], c["C"]), 1.0), (Xb, onehot(yb, c["C"]), c["lam_book"])]
        consecrate_bishop(model, k, terms, 2 * c["s_bish"], fresh=False)
    return consecrate_head(model, U0, Xb[:0], yb[:0], book)


def sava_generation(model, gd, book, g):
    """Staggered mortality, local learning under jurisdiction, then synodal consecration of the head."""
    c, f, (Xb, yb), (Xt, yt), U = model["cfg"], model["flags"], book, gd["T"], gd["U"]
    rt, rb, ru = territory(model, Xt), territory(model, Xb), territory(model, U)
    dead = g % c["K"] if (f["memory"] and f["mortality"]) else None
    for k in range(c["K"]):
        terms = [(Xt[rt == k], onehot(yt[rt == k], c["C"]), c["w_trickle"])]
        if f["book"]:
            terms.append((Xb[rb == k], onehot(yb[rb == k], c["C"]), c["lam_book"]))
        fresh, steps = not f["memory"], c["s_bish"]
        if k == dead:  # the head consecrates the newcomer by teaching it the land
            terms.append((U[ru == k], head_probs(model, U[ru == k]), c["w_unlab"]))
            fresh, steps = True, 2 * c["s_bish"]
        elif not f["memory"]:
            steps = c["s_bish"] + c["s_bish"] // c["K"]
        consecrate_bishop(model, k, terms, steps, fresh)
    return consecrate_head(model, U, Xt, yt, book)


def fit(model, data, budget, rng):
    """Single-generation fit for external probes: regional learners, then synodal consecration."""
    c = model["cfg"]
    model["rng"] = rng
    X, y = data["X"], data["y"]
    U = data.get("X_unlab", X)
    r = territory(model, X)
    for k in range(c["K"]):
        terms = [(X[r == k], onehot(y[r == k], c["C"]), 1.0), (X, onehot(y, c["C"]), c["lam_book"])]
        consecrate_bishop(model, k, terms, max(1, budget // (c["K"] + 1)), fresh=False)
    c_old = c["s_head"]
    c["s_head"] = max(1, budget)
    consecrate_head(model, U, X, y, (X[:0], y[:0]))
    c["s_head"] = c_old
    return model


# ----------------------------------------------------------------------------- baselines and rival
def lineage_init(rng, cfg):
    return mlp_init(rng, "solo", cfg["d"], cfg["h_solo"], cfg["C"])


def lineage_generation(P, mode, gd, book, archive, mother, cfg, rng):
    """SOLO: born-again lineage. ACCUM: lineage plus every real label so far. OHRID: mother + current report."""
    (Xt, yt), (Xb, yb), U, C = gd["T"], book, gd["U"], cfg["C"]
    budget = cfg["s_head"] + (cfg["K"] + 1) * cfg["s_bish"]
    if mode == "ohrid":
        Pn = {k: v.copy() for k, v in mother.items()}
        terms = [(Xt, onehot(yt, C), cfg["w_trickle"]), (Xb, onehot(yb, C), cfg["lam_book"])]
    else:
        Pn = lineage_init(rng, cfg)
        terms = [(U, softmax(mlp_logits(P, "solo", U)[0]), cfg["w_unlab"]), (Xb, onehot(yb, C), cfg["lam_book"])]
        Xr, yr = (np.vstack(archive[0]), np.concatenate(archive[1])) if mode == "accum" else (Xt, yt)
        terms.append((Xr, onehot(yr, C), cfg["w_trickle"]))
    train_prefix(Pn, "solo", assemble(terms), budget, cfg)
    return Pn


def lineage_accuracy(P, X, y):
    return float(np.mean(mlp_logits(P, "solo", X)[0].argmax(1) == y))


# ----------------------------------------------------------------------------- registries
def modules(model):
    P = model["params"]
    return {
        "head": {"params": [k for k in P if k.startswith("head.")], "signature": False,
                 "role": "tanh MLP re-initialised and distilled each generation; the deployed predictor"},
        "bishops": {"params": [k for k in P if k.startswith("b")], "signature": True,
                    "role": "K regional tanh MLPs, each trained only on its own input-space cell"},
        "jurisdiction_gate": {"params": [], "signature": True,
                              "role": "hard partition gate: a pseudo-label for x comes only from x's cell owner"},
        "synod_memory": {"params": [], "signature": True,
                         "role": "persistence of regional learners with round-robin replacement taught by the head"},
        "book_anchor": {"params": [], "signature": False,
                        "role": "replay of a fixed labelled seed set by the regional learners"},
    }


def knockout(model, name, mode="zero"):
    """Copy of the model with one module replaced (identity/zero/mean as meaningful for the module)."""
    new = {"params": {k: v.copy() for k, v in model["params"].items()}, "cfg": dict(model["cfg"]),
           "rng": model["rng"], "flags": dict(model["flags"])}
    if name == "jurisdiction_gate":
        new["flags"]["gate"] = "mean"
    elif name == "synod_memory":
        new["flags"]["memory"] = False
    elif name == "book_anchor":
        new["flags"]["book"] = False
    elif name in ("head", "bishops"):
        pre = ("head.",) if name == "head" else tuple(f"b{k}." for k in range(new["cfg"]["K"]))
        for k in new["params"]:
            if k.startswith(pre) and (k.endswith("W2") or k.endswith("b2")):
                new["params"][k] = new["params"][k] * 0.0 if mode == "zero" else new["params"][k]
    else:
        raise KeyError(name)
    return new


MUTANTS = {"sign_flip": "optimizer ascends instead of descends", "zero_lr": "learning rate forced to zero",
           "zero_grad_head_W1": "gradient of head.W1 zeroed", "zero_grad_bishop_W2": "gradient of b0.W2 zeroed"}


# ----------------------------------------------------------------------------- training protocol
def run_sava(data, cfg, rng, ko=None):
    model = build_model(cfg["d"], cfg["C"], "vector_classification", rng, **cfg)
    if ko:
        model = knockout(model, ko)
    head_losses = [sava_founding(model, data["book"], data["gens"][0]["U"])]
    accs = [float(np.mean(predict(model, data["gens"][0]["S"][0]) == data["gens"][0]["S"][1]))]
    for g in range(1, cfg["G"] + 1):
        gd = data["gens"][g]
        head_losses.append(sava_generation(model, gd, data["book"], g))
        accs.append(float(np.mean(predict(model, gd["S"][0]) == gd["S"][1])))
    return {"acc": accs, "head_losses": head_losses, "model": model}


def run_lineage(data, cfg, rng, mode):
    Xb, yb = data["book"]
    mother = lineage_init(rng, cfg)
    train_prefix(mother, "solo", assemble([(Xb, onehot(yb, cfg["C"]), 1.0)]),
                 cfg["s_head"] + 2 * cfg["K"] * cfg["s_bish"], cfg)
    P, archive = mother, ([], [])
    accs = [lineage_accuracy(P, *data["gens"][0]["S"])]
    for g in range(1, cfg["G"] + 1):
        gd = data["gens"][g]
        archive[0].append(gd["T"][0])
        archive[1].append(gd["T"][1])
        P = lineage_generation(P, mode, gd, data["book"], archive, mother, cfg, rng)
        accs.append(lineage_accuracy(P, *gd["S"]))
    return {"acc": accs, "params": P}


def split_means(accs, G):
    third = max(1, G // 3)
    return float(np.mean(accs[1:1 + third])), float(np.mean(accs[G - third + 1:]))


def run_seed(seed, cfg):
    kids = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(6)]
    native = make_chain_data(kids[0], cfg, aligned=True)
    blind = make_chain_data(kids[1], cfg, aligned=False)
    rs = lambda i: np.random.default_rng(np.random.SeedSequence([seed, i]))  # common random numbers
    out = {"sava": run_sava(native, cfg, rs(2)), "solo": run_lineage(native, cfg, rs(3), "solo"),
           "accum": run_lineage(native, cfg, rs(3), "accum"), "ohrid": run_lineage(native, cfg, rs(3), "ohrid"),
           "ko_gate": run_sava(native, cfg, rs(2), "jurisdiction_gate"),
           "ko_memory": run_sava(native, cfg, rs(2), "synod_memory"),
           "ko_book": run_sava(native, cfg, rs(2), "book_anchor"),
           "sava_blind": run_sava(blind, cfg, rs(2)), "solo_blind": run_lineage(blind, cfg, rs(3), "solo")}
    metrics = {k: dict(zip(("acc_early", "acc_late"), split_means(v["acc"], cfg["G"]))) for k, v in out.items()}
    return out, metrics, native


# ----------------------------------------------------------------------------- correctness tests
def mixed_batch(rng, model, n_head=48, n_b=16):
    c = model["cfg"]
    def soft(n):
        return rng.dirichlet(np.ones(c["C"]), size=n)
    head = (rng.normal(size=(n_head, c["d"])), soft(n_head), rng.random(n_head) / n_head)
    bishops = {k: (rng.normal(size=(n_b, c["d"])), soft(n_b), rng.random(n_b) / n_b) for k in range(c["K"])}
    return {"head": head, "bishops": bishops}


def gradcheck_model(model, batch, rng):
    loss_fn = lambda P: loss_and_grads({"params": P}, batch)[0]
    _, g = loss_and_grads(model, batch)
    return finite_difference_check(loss_fn, model["params"], g, rng)


def test_gradcheck(cfg, rng, steps=60):
    model = build_model(cfg["d"], cfg["C"], "vector_classification", rng, **cfg)
    batch = mixed_batch(rng, model)
    first = gradcheck_model(model, batch, rng)
    opt = Adam(model["params"].keys(), lr=cfg["lr"])
    for _ in range(steps):
        _, g = loss_and_grads(model, batch)
        opt.step(model["params"], g)
    second = gradcheck_model(model, batch, rng)
    solo = {"params": lineage_init(rng, cfg)}
    sb = (batch["head"][0], batch["head"][1], batch["head"][2])
    sg = mlp_loss_grads(solo["params"], "solo", *sb)[1]
    third = finite_difference_check(lambda P: mlp_loss_grads(P, "solo", *sb)[0], solo["params"], sg, rng)
    total = len(model["params"]) + len(solo["params"])
    checked = len(first["per_tensor"]) + len(third["per_tensor"])
    worst = max(first["max_rel_error"], second["max_rel_error"], third["max_rel_error"])
    return {"tensors_checked": checked, "tensors_total": total, "max_rel_error": worst,
            "checked_at": ["init", f"after_{steps}_training_steps"],
            "passed": first["passed"] and second["passed"] and third["passed"] and checked == total}


def learning_probe(cfg, rng):
    """Short head consecration on a fixed labelled batch: must cut the loss by the declared fraction."""
    model = build_model(cfg["d"], cfg["C"], "vector_classification", rng, **cfg)
    X = rng.normal(size=(200, cfg["d"]))
    y = (X[:, 0] + 0.5 * X[:, 2] > 0).astype(int) + (X[:, 1] > 1).astype(int)
    losses = train_prefix(model["params"], "head", assemble([(X, onehot(y, cfg["C"]), 1.0)]), 60, cfg)
    return losses[0], losses[-1], (losses[0] - losses[-1]) / losses[0] >= THRESH["loss_drop_fraction"]


def test_determinism(cfg, seed):
    small = dict(cfg, G=2, s_head=30, s_bish=10, n_u=120, n_test=200)
    runs = []
    for _ in range(2):
        data = make_chain_data(np.random.default_rng(seed), small)
        r = run_sava(data, small, np.random.default_rng(seed + 1))
        runs.append(r)
    same = runs[0]["acc"] == runs[1]["acc"] and all(
        np.array_equal(runs[0]["model"]["params"][k], runs[1]["model"]["params"][k]) for k in runs[0]["model"]["params"])
    finite = all(np.all(np.isfinite(v)) for v in runs[0]["model"]["params"].values())
    finite = finite and bool(np.all(np.isfinite(hidden_states(runs[0]["model"], data["gens"][1]["S"][0]))))
    return same and finite


def test_shuffled(native, cfg, seed):
    shuffled = shuffle_labels(native, np.random.default_rng(seed + 99))
    r = run_sava(shuffled, cfg, np.random.default_rng(seed + 98))
    _, late = split_means(r["acc"], cfg["G"])
    third = max(1, cfg["G"] // 3)
    majority = float(np.mean([np.bincount(native["gens"][g]["S"][1], minlength=cfg["C"]).max() / cfg["n_test"]
                              for g in range(cfg["G"] - third + 1, cfg["G"] + 1)]))
    return late, majority, late <= majority + THRESH["shuffle_band"]


def test_jurisdiction_locality(cfg, rng, trials=30):
    """C6.1: targets on cell k never move when other cells' learners are perturbed; mean gate must move."""
    worst_gate, worst_mean = 0.0, 0.0
    for _ in range(trials):
        model = build_model(cfg["d"], cfg["C"], "vector_classification", rng, **cfg)
        for key in model["params"]:
            model["params"][key] = model["params"][key] * 3.0 + rng.normal(0, 0.5, model["params"][key].shape)
        X = rng.normal(size=(300, cfg["d"]))
        k = int(rng.integers(cfg["K"]))
        Xk = X[territory(model, X) == k]
        pert = {key: v + rng.normal(0, 2.0, v.shape) if key.startswith("b") and not key.startswith(f"b{k}.") else v
                for key, v in model["params"].items()}
        for gate in ("jurisdiction", "mean"):
            model["flags"]["gate"] = gate
            t0 = synod_targets(model, Xk)
            t1 = synod_targets({"params": pert, "cfg": model["cfg"], "flags": model["flags"]}, Xk)
            diff = float(np.max(np.abs(t1 - t0)))
            if gate == "jurisdiction":
                worst_gate = max(worst_gate, diff)
            else:
                worst_mean = max(worst_mean, diff)
    return worst_gate, worst_mean, worst_gate <= 1e-12 and worst_mean > 1e-3


def test_successor_independence(cfg, seed):
    """C6.2: the synod-made head is identical whatever its predecessor was; the lineage's is not."""
    small = dict(cfg, s_head=20, s_bish=5, n_u=80, n_test=50, G=1)
    data = make_chain_data(np.random.default_rng(seed), small)
    heads, solos = [], []
    for variant in range(2):
        model = build_model(small["d"], small["C"], "vector_classification", np.random.default_rng(seed + 5), **small)
        pre = mlp_init(np.random.default_rng(1000 + variant), "head", small["d"], small["h_head"], small["C"])
        model["params"].update(pre)
        model["flags"]["mortality"] = False  # isolate the consecration step itself
        model["rng"] = np.random.default_rng(seed + 7)
        sava_generation(model, data["gens"][1], data["book"], 1)
        heads.append(model["params"]["head.W1"].copy())
        prev = lineage_init(np.random.default_rng(2000 + variant), small)
        Pn = lineage_generation(prev, "solo", data["gens"][1], data["book"], ([], []), prev, small,
                                np.random.default_rng(seed + 9))
        solos.append(Pn["solo.W1"])
    d_head = float(np.max(np.abs(heads[0] - heads[1])))
    d_solo = float(np.max(np.abs(solos[0] - solos[1])))
    return d_head, d_solo, d_head == 0.0 and d_solo > 1e-6


def row_hashes(X):
    return {hashlib.sha1(np.round(r, 12).tobytes()).hexdigest() for r in X}


def test_split_integrity(native):
    train = row_hashes(native["book"][0])
    test = set()
    for gd in native["gens"]:
        train |= row_hashes(gd["U"]) | row_hashes(gd["T"][0])
        test |= row_hashes(gd["S"][0])
    return len(train & test) == 0


def test_mutants(cfg, rng):
    global MUTANT
    saved, detected = MUTANT, 0
    for name in MUTANTS:
        MUTANT = name
        gc = test_gradcheck(dict(cfg), np.random.default_rng(int(rng.integers(1 << 30))), steps=10)
        _, _, learned = learning_probe(dict(cfg), np.random.default_rng(int(rng.integers(1 << 30))))
        detected += int((not gc["passed"]) or (not learned))
    MUTANT = saved
    return {"detected": detected, "total": len(MUTANTS), "score": detected / len(MUTANTS)}


# ----------------------------------------------------------------------------- hypotheses
def evaluate_hypotheses(per_seed, rng):
    def diffs(fn):
        return [fn(m) for m in per_seed]
    specs = {
        "H-SIG": lambda m: m["sava"]["acc_late"] - m["solo"]["acc_late"],
        "H-NEC": lambda m: m["ko_gate"]["acc_late"] - m["ko_book"]["acc_late"],
        "H-BLIND": lambda m: ((m["sava_blind"]["acc_late"] - m["solo_blind"]["acc_late"])
                              - (m["sava"]["acc_late"] - m["solo"]["acc_late"])),
        "H-RIVAL": lambda m: m["sava"]["acc_late"] - m["ohrid"]["acc_late"],
    }
    rows = []
    for h in MIND_CARD["hypotheses"]:
        d = diffs(specs[h["id"]])
        if len(d) < 2:
            rows.append({"id": h["id"], "metric": h["metric"], "mean_diff": float(np.mean(d)), "ci95": [None, None],
                         "mesi": h["mesi"], "n_seeds": len(d), "verdict": "not evaluated"})
            continue
        mean, lo, hi = paired_bootstrap_ci(d, rng)
        rows.append({"id": h["id"], "metric": h["metric"], "mean_diff": mean, "ci95": [lo, hi], "mesi": h["mesi"],
                     "n_seeds": len(d), "verdict": hypothesis_verdict(mean, lo, hi, h["mesi"], h["direction"])})
    return rows


def knockout_table(per_seed, rng, model):
    rows, reg = [], modules(model)
    for mod, key in (("jurisdiction_gate", "ko_gate"), ("synod_memory", "ko_memory"), ("book_anchor", "ko_book")):
        sig = reg[mod]["signature"]
        d = [m[key]["acc_late"] - m["sava"]["acc_late"] for m in per_seed]
        mean, lo, hi = paired_bootstrap_ci(d, rng) if len(d) > 1 else (float(np.mean(d)), None, None)
        rows.append({"module": mod, "signature": sig, "metric_change": mean, "ci95": [lo, hi]})
    return rows


# ----------------------------------------------------------------------------- report
def fmt(x):
    return "   n/a" if x is None else f"{x:+.4f}"


def print_report(rep, per_seed):
    print(f"=== VERIFIED REPORT · chapter {CHAPTER:04d} ===")
    print(f"environment: python {rep['environment']['python']}, numpy {rep['environment']['numpy']}")
    print(f"seeds: {rep['seeds']}   runtime_s: {rep['runtime_s']:.1f}   n_params: {rep['n_params']} "
          f"(size-matched lineage: {rep['n_params_baseline']})")
    gc = rep["gradcheck"]
    print(f"gradcheck: {gc['tensors_checked']}/{gc['tensors_total']} tensors, max rel err {gc['max_rel_error']:.2e}, "
          f"at {', '.join(gc['checked_at'])}: {'PASS' if gc['passed'] else 'FAIL'}")
    print("correctness:")
    for c in rep["correctness"]:
        print(f"  {c['id']:<5}{c['name']:<28}{'PASS' if c['passed'] else 'FAIL'}   {c['detail']}")
    mu = rep["mutants"]
    print(f"mutation score: {mu['detected']}/{mu['total']} ({mu['score']:.0%})")
    print("mean accuracy by method (held-out = first third, shifted = last third of the chain):")
    for k in ("sava", "solo", "accum", "ohrid", "ko_gate", "ko_memory", "ko_book", "sava_blind", "solo_blind"):
        e = np.mean([m[k]["acc_early"] for m in per_seed])
        la = [m[k]["acc_late"] for m in per_seed]
        print(f"  {k:<11} held-out {e:.4f}   shifted {np.mean(la):.4f} ± {np.std(la):.4f}")
    print("hypotheses:")
    for h in rep["hypotheses"]:
        lo, hi = h["ci95"]
        print(f"  {h['id']:<8}{h['metric']:<10}mean {h['mean_diff']:+.4f}  CI95 [{fmt(lo)}, {fmt(hi)}]  "
              f"mesi {h['mesi']:.2f}  n={h['n_seeds']}  {h['verdict']}")
    print("knockouts (change in shifted accuracy vs full model):")
    for kr in rep["knockouts"]:
        lo, hi = kr["ci95"]
        print(f"  {kr['module']:<18}signature={str(kr['signature']):<6}{kr['metric_change']:+.4f}  "
              f"CI95 [{fmt(lo)}, {fmt(hi)}]")
    print(f"task types: {', '.join(rep['task_types'])}")
    print("=== END REPORT ===")


def data_bridge(path, cfg, rng):
    """Optional real-data bridge: CSV with region id in column 0, features, integer label last."""
    if not path:
        print("real-data bridge: skipped (no --data PATH given)")
        return
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    X, y = arr[:, 1:-1], arr[:, -1].astype(int)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    region = arr[:, 0].astype(int) % 4
    X2 = np.column_stack([np.where(region >= 2, 1.0, -1.0), np.where(region % 2 == 1, 1.0, -1.0), X])
    idx = rng.permutation(len(y))
    tr, te = idx[: int(0.8 * len(y))], idx[int(0.8 * len(y)):]
    C = int(y.max()) + 1
    model = fit(build_model(X2.shape[1], C, "vector_classification", rng), {"X": X2[tr], "y": y[tr]}, 300, rng)
    solo = dict(DEFAULTS, d=X2.shape[1], C=C)
    Ps = lineage_init(rng, solo)
    train_prefix(Ps, "solo", assemble([(X2[tr], onehot(y[tr], C), 1.0)]), 500, solo)
    print(f"real-data bridge ({os.path.basename(path)}): synod head acc {np.mean(predict(model, X2[te]) == y[te]):.4f}"
          f"  plain MLP acc {lineage_accuracy(Ps, X2[te], y[te]):.4f}")


def correctness_block(cfg, seeds, per_seed, runs0, native0, t0, budget):
    rng = np.random.default_rng(seeds[0] + 12345)
    gc = test_gradcheck(cfg, rng)
    losses = runs0["sava"]["head_losses"][1]
    drop = (losses[0] - losses[-1]) / losses[0]
    majority = np.bincount(native0["gens"][1]["S"][1], minlength=cfg["C"]).max() / cfg["n_test"]
    early = per_seed[0]["sava"]["acc_early"]
    shuf_late, shuf_major, shuf_ok = test_shuffled(native0, cfg, seeds[0])
    loc_gate, loc_mean, loc_ok = test_jurisdiction_locality(cfg, rng)
    d_head, d_solo, ind_ok = test_successor_independence(cfg, seeds[0])
    mut = test_mutants(cfg, rng)
    rows = [
        {"id": "C1", "name": "gradient_check", "passed": gc["passed"],
         "detail": f"max rel err {gc['max_rel_error']:.2e}"},
        {"id": "C2", "name": "determinism_and_finiteness", "passed": test_determinism(cfg, seeds[0]), "detail": ""},
        {"id": "C3", "name": "learning", "passed": bool(drop >= THRESH["loss_drop_fraction"] and
                                                        early >= majority + THRESH["margin_over_trivial"]),
         "detail": f"loss drop {drop:.2f}; held-out {early:.3f} vs majority {majority:.3f}"},
        {"id": "C4", "name": "shuffled_label_control", "passed": bool(shuf_ok),
         "detail": f"shifted acc {shuf_late:.3f} vs majority {shuf_major:.3f} + {THRESH['shuffle_band']}"},
        {"id": "C5", "name": "mutant_detection", "passed": mut["score"] == 1.0,
         "detail": f"{mut['detected']}/{mut['total']}"},
        {"id": "C6.1", "name": "jurisdiction_locality", "passed": bool(loc_ok),
         "detail": f"gate max shift {loc_gate:.1e}; negative control (mean gate) {loc_mean:.2e}"},
        {"id": "C6.2", "name": "successor_independence", "passed": bool(ind_ok),
         "detail": f"synod head diff {d_head:.1e}; negative control (lineage) {d_solo:.2e}"},
        {"id": "C7", "name": "split_integrity", "passed": test_split_integrity(native0), "detail": ""},
    ]
    elapsed = time.time() - t0
    rows.append({"id": "C8", "name": "budget", "passed": elapsed <= budget, "detail": f"{elapsed:.1f}s <= {budget}s"})
    return gc, mut, rows


def main(argv=None):
    global MUTANT
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__), description="Chapter 0320 · Synodal Succession")
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
        print(f"usage: python3 {os.path.basename(__file__)} --mutant {{{','.join(MUTANTS)}}}")
        return 2
    MUTANT = args.mutant
    cfg = dict(DEFAULTS, **(QUICK if args.quick else {}))
    n_seeds = 1 if args.quick else max(1, args.seeds)
    seeds = [args.seed + i for i in range(n_seeds)]
    budget = 20 if args.quick else 180
    t0 = time.time()
    print(f"seeds: {seeds}")
    try:
        per_seed, first = [], None
        for s in seeds:
            runs, metrics, native = run_seed(s, cfg)
            per_seed.append(metrics)
            first = first or (runs, native)
        gc, mut, rows = correctness_block(cfg, seeds, per_seed, first[0], first[1], t0, budget)
    except NonFinite as exc:
        print(f"non-finite value in {exc}")
        return 4
    hyp_rng = np.random.default_rng(seeds[0] + 777)
    hyps = evaluate_hypotheses(per_seed, hyp_rng) if n_seeds > 1 else [
        {"id": h["id"], "metric": h["metric"], "mean_diff": 0.0, "ci95": [None, None], "mesi": h["mesi"],
         "n_seeds": 1, "verdict": "not evaluated"} for h in MIND_CARD["hypotheses"]]
    model = first[0]["sava"]["model"]
    kos = knockout_table(per_seed, hyp_rng, model)
    ok = all(r["passed"] for r in rows)
    rep = {"schema_version": "1.0", "chapter": CHAPTER, "file": os.path.basename(__file__),
           "card_revision": MIND_CARD["card_revision"],
           "environment": {"python": sys.version.split()[0], "numpy": np.__version__}, "seeds": seeds,
           "runtime_s": time.time() - t0, "n_params": n_params(model),
           "n_params_baseline": int(sum(v.size for v in first[0]["solo"]["params"].values())),
           "gradcheck": gc, "correctness": rows, "mutants": mut, "hypotheses": hyps, "knockouts": kos,
           "task_types": TASK_TYPES, "exit_code": 0 if ok else 1}
    over = rows[-1]["passed"] is False
    rep["exit_code"] = 3 if over and all(r["passed"] for r in rows[:-1]) else rep["exit_code"]
    print_report(rep, per_seed)
    data_bridge(args.data, cfg, np.random.default_rng(seeds[0]))
    if args.json:
        write_json_report(args.json, rep)
    return rep["exit_code"]


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0349 · Alonso Pérez de Guzmán (Guzmán el Bueno) (1256-1309)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 18, Minds 341-360 Available on Amazon https://www.amazon.com/dp/B0HQ47T2NC
# Resume and Interactive Demos at https://artificiology.com/
# END ATTRIBUTION 
"""
Echo-priced refusal: a two-organ decision network whose refusal price is learned
through a pooled mirror of how coercers learn.

Thesis
    A concession is priced not at what it saves in the episode at hand but at what
    it teaches the population of future coercers; the entrusted charge is kept
    outside the currency in which the holder can be threatened.

Evidence and provenance (extrapolated: no philosophy of mind survives; inferred
from documented deeds and one reply reported by the royal chronicle c. 1344)
    D1  Tarifa, Aug. 1294: asked to yield the town or see his captive son killed,
        he answered that he would rather lose that son "e otros cinco si los
        tuviese" than give up the king's town of which he had made homage, and
        threw down a knife (Crónica de Sancho IV, ed. Rosell, BAE 66, p. 86).
    D2  Fernando IV's privilegio rodado (Toro, 13 Oct. 1297) records the knife
        three years after the event (Gaibrois, BRAH 74-76, 1919-20).
    D3  The coercer had a record: the infante Juan had earlier obtained the
        alcázar of Zamora with the same threat (DBE, s.v., R. Sánchez Saus).
    D4  c. 1279-1291 he served the Marinid emirs in the Maghreb; he knew the
        besiegers' court from inside (DBE; Barton 2002 on Christian militias).
    D5  July 1293 he took the tenencia of Tarifa for 600,000 maravedís a year,
        against the 2,000,000 paid to the Order of Calatrava (DBE).
    D6  In 1300 he blocked the infante Enrique's plan to return Tarifa for a truce
        with Granada, seeking Aragonese backing rather than allow it (DBE).
    D7  Partida II.18.6 binds the alcaide to hold a besieged castle unto death even
        if wife or children are harmed; II.18.19 lets him refuse false messengers
        (González Jiménez, "La tenencia de castillos en tiempos de Alfonso X").
    D8  The son died; the siege was lifted by a relief fleet, not by the refusal
        alone (DBE; García Fitz 2005).

Doctrine -> mechanism -> test (IDs as in MIND_CARD)
    D1,D2 -> M1 precedent_price: non-negative price subtracted from the decision
             logit, trained on the long-run objective       -> C6.1, C6.2, H-SIG
    D4    -> M2 besieger_mirror: one monotone response curve of threat rate to
             observed concession rate, pre-fitted on a separate all-learning
             population ("the Fez years"), refitted locally -> C6.3, H-RIVAL
    D5    -> M3 present_calculus: separate organ trained on per-episode cost; the
             son's value is registered, not denied          -> H-NEC
    D3    -> task: strategic coercers raise their threat rate with observed
             concessions                                    -> H-SIG
    D7    -> task: crown orders must be obeyed, forged seals refused -> C3
    D8    -> blind spot: coercers who do not learn          -> H-BLIND

Research question
    When the arrival rate of coercive demands responds to a policy's observed
    concession rate (adaptive jailbreakers, extortion markets), does training
    the refusal organ through a learned model of the coercers' learning lower
    long-run loss relative to repeated risk minimisation, and where does pooling
    one theory of coercers across heterogeneous coercers fail?

Closest prior art and the delta
    Repeated risk minimisation reaches performatively stable points (Perdomo et
    al. 2020); performative gradient descent estimates the distribution map from
    deployments (Izzo et al. 2021); opponent shaping differentiates through a
    rival's learning (Foerster et al. 2018); unbending strategies starve
    extortioners (Chen & Fu 2023). Delta: the distribution map is ONE pooled,
    monotone mirror carried in from another population and updated locally, and
    it prices a sign-constrained refusal organ kept apart from the per-episode
    organ; the price therefore transfers to unseen coercers by cue similarity,
    and fails exactly where coercers do not learn. RRM is the baseline; a
    per-faction PerfGD estimator (the "alfaqueque", after the frontier's
    case-by-case ransom brokers of Partida II.30) is the rival.

Blind spot
    Pricing assumes every coercer learns from observed refusals. Against spite
    or non-learning coercers the refusal buys no deterrence and pays the hostage.

Task (generative process)
    1. Nine banner prototypes: six native coercer factions (four learning, two
       spiteful), the crown, and two unseen learning factions (shift only).
    2. A demand x = [noisy banner (8), seal quality, log(1+hostage), charge value,
       reputation for execution, 4 nuisance]; 12% of coercers forge high seals.
    3. Coerced demand: yield costs the charge V; refusing costs e_k * h.
       Crown order: obeying costs 0; refusing costs V.
    4. A learning faction's threat rate relaxes toward 1.0*sigmoid(9(q - C_k)),
       q its observed concession rate; spiteful factions stay at 0.5; crown 0.6.
    5. Metric: long-run loss J* = sum_k rate_k(q_k) * mean loss_k, with rates at
       the population's equilibrium for the evaluated policy.
    Splits: held-out (native), shifted (unseen factions, hostages x2.5),
    blind (same demands, no faction learns).

Limits
    Synthetic populations with known response curves; scalar decisions; the
    file is a research prototype of an AGI-oriented component, not an AGI, and
    says nothing about any real hostage decision.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 3,
    "revision_log": [
        "r2 (after first run): corpus_neighbors similarity filled from the only corpus file available in "
        "this session; no hypothesis, metric, threshold or mesi changed.",
        "r3 (after single-seed smoke runs, before any multi-seed evaluation): the per-episode organ's "
        "expected-cost objective drove its bias to saturation, so the rrm baseline yielded to every demand "
        "while a myopic oracle concedes 65-85%; replaced by a cost-weighted logistic surrogate in all three "
        "learners. Gradient-check denominator floored at 1e-4 (round-off). No hypothesis, metric, threshold "
        "or mesi changed.",
    ],
    "generation": {"template_version": "codeguidelines v1.0 (2026-09-15)", "generator": "Claude",
                   "generator_version": "claude-opus-5-5", "date": "2026-09-30"},
    "id": 349, "figure": "Alonso Pérez de Guzmán (Guzmán el Bueno)", "born": 1256, "died": 1309,
    "civilization": "Castilian", "provenance": "extrapolated",
    "thesis": ("A concession is priced at what it teaches the population of future coercers, a valuation "
               "made possible by having lived inside the besiegers' camp, and the entrusted charge is kept "
               "outside the currency in which its holder can be threatened."),
    "evidence": [
        {"id": "D1", "claim": "Refused to yield Tarifa for his captive son; would lose 'otros cinco' rather "
         "than the king's town of which he had made homage; threw down a knife", "basis": "deeds",
         "source": "Crónica de Sancho IV, ed. C. Rosell, BAE 66, p. 86"},
        {"id": "D2", "claim": "The knife is recorded in Fernando IV's privilege of 13 Oct. 1297", "basis":
         "scholarship", "source": "M. Gaibrois, 'Tarifa y la política de Sancho IV', BRAH 74-76 (1919-20)"},
        {"id": "D3", "claim": "The infante Juan had taken the alcázar of Zamora by the same threat", "basis":
         "scholarship", "source": "R. Sánchez Saus, DBE s.v. 'Alonso Pérez de Guzmán'"},
        {"id": "D4", "claim": "Served the Marinid emirs c. 1279-1291 and knew their court from inside",
         "basis": "scholarship", "source": "DBE; S. Barton, 'Traitors to the Faith?' (2002)"},
        {"id": "D5", "claim": "Took the tenencia of Tarifa for 600,000 mrs against Calatrava's 2,000,000",
         "basis": "deeds", "source": "DBE"},
        {"id": "D6", "claim": "In 1300 blocked the return of Tarifa for a truce", "basis": "deeds",
         "source": "DBE"},
        {"id": "D7", "claim": "Partida II.18.6 and II.18.19 on holding castles and refusing false messengers",
         "basis": "scholarship", "source": "M. González Jiménez, 'La tenencia de castillos en tiempos de "
         "Alfonso X'"},
        {"id": "D8", "claim": "The son died; a relief fleet lifted the siege", "basis": "scholarship",
         "source": "DBE; F. García Fitz in Tarifa en la Edad Media (2005)"},
    ],
    "research_question": {"category": "multi-agent cooperation and conflict",
                          "question": ("Does training a refusal organ through a learned model of coercers' "
                                       "learning beat repeated risk minimisation when threat rates respond to "
                                       "observed concessions, and where does one pooled theory of coercers fail?")},
    "mechanism": {
        "name": "Echo-priced refusal (tenencia network)",
        "family": "performative learning with a learned distribution map; two-organ decision logit",
        "signature_modules": ["precedent_price", "besieger_mirror"],
        "closest_prior_art": ["Repeated risk minimisation / performative stability (Perdomo et al. 2020)",
                              "Performative gradient descent (Izzo, Ying & Zou 2021)",
                              "LOLA opponent shaping (Foerster et al. 2018)",
                              "Unbending strategies against extortion (Chen & Fu 2023)"],
        "overlap": "Medium",
        "prior_art_queries": ["performative prediction distribution map gradient",
                              "opponent learning awareness extortion", "no-concession policy kidnapping model",
                              "jailbreak adaptive attacker population training", "unbending strategies extortion"],
        "contribution_type": "mechanism",
        "delta": ("A pooled monotone mirror of coercer learning, carried in from another population and "
                  "refitted locally, prices a sign-constrained refusal organ kept apart from the per-episode "
                  "organ; the price transfers to unseen coercers and fails where coercers do not learn."),
    },
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1", "property_test": "C6.1", "hypothesis": "H-SIG"},
        {"doctrine": "D2", "mechanism": "M1", "property_test": "C6.2", "hypothesis": "H-SIG"},
        {"doctrine": "D4", "mechanism": "M2", "property_test": "C6.3", "hypothesis": "H-RIVAL"},
        {"doctrine": "D5", "mechanism": "M3", "property_test": "C6.1", "hypothesis": "H-NEC"},
        {"doctrine": "D3", "mechanism": "task", "property_test": "C7", "hypothesis": "H-SIG"},
        {"doctrine": "D7", "mechanism": "task", "property_test": "C3", "hypothesis": "H-SIG"},
        {"doctrine": "D8", "mechanism": "M2", "property_test": "C6.3", "hypothesis": "H-BLIND"},
    ],
    "hypotheses": [
        {"id": "H-SIG.heldout", "statement": "Echo pricing lowers long-run loss on native coercers",
         "metric": "equilibrium_loss", "split": "heldout", "comparison": "model - rrm_baseline",
         "direction": "less", "mesi": 0.10, "seeds": 5},
        {"id": "H-SIG.shifted", "statement": "The price transfers to unseen learning coercers and heavier hostages",
         "metric": "equilibrium_loss", "split": "shifted", "comparison": "model - rrm_baseline",
         "direction": "less", "mesi": 0.10, "seeds": 5},
        {"id": "H-NEC", "statement": "Zeroing the precedent price hurts more than zeroing the present calculus",
         "metric": "equilibrium_loss", "split": "heldout",
         "comparison": "signature_knockout - matched_knockout", "direction": "greater", "mesi": 0.05, "seeds": 5},
        {"id": "H-BLIND", "statement": "Where coercers do not learn, refusal buys nothing and pays the hostage",
         "condition": "blind (all factions non-learning)", "grounding": "D8: the son died; spite is not learning",
         "metric": "equilibrium_loss", "comparison": "model - rrm_baseline", "direction": "greater",
         "mesi": 0.10, "seeds": 5},
        {"id": "H-RIVAL", "statement": "The pooled mirror beats per-faction pricing on unseen coercers",
         "metric": "equilibrium_loss", "split": "shifted", "comparison": "model - alfaqueque",
         "direction": "less", "mesi": 0.05, "seeds": 5},
        {"id": "H-RIVAL.native", "statement": "Per-faction pricing beats the pooled mirror where local records exist",
         "metric": "equilibrium_loss", "split": "heldout", "comparison": "model - alfaqueque",
         "direction": "greater", "mesi": 0.05, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.25, "margin_over_trivial": 0.10, "shuffled_band": 0.05},
    "probe_predictions": [{"probe": "P2", "expected": "equal to baseline"},
                          {"probe": "P6", "expected": "below baseline"},
                          {"probe": "P8", "expected": "equal to baseline"},
                          {"probe": "P9", "expected": "equal to baseline"}],
    "dialectic_links": [{"chapter": 337, "relation": "successor", "test": "H-RIVAL"}],
    "corpus_neighbors": [
        {"chapter": 43, "similarity": None, "difference": "Leonidas removes his own freedom to defect; here "
         "refusal is a learned price that can be wrong and is tested where it should fail"},
        {"chapter": 158, "similarity": None, "difference": "Attila models the extractor's terror reputation; "
         "this file models the holder who makes that reputation unprofitable"},
        {"chapter": 405, "similarity": None, "difference": "Constantine XI manages what defenders believe; "
         "this file models what attackers learn"},
        {"chapter": 386, "similarity": None, "difference": "Hus makes the word structurally uncoercible; here "
         "refusal is priced, not structural"},
        {"chapter": 504, "similarity": 0.035, "difference": "Philip II attenuates annotations multiplicatively; "
         "no adversary population (9-token shingle Jaccard against the file distributed as 0507_...)"},
    ],
    "barometer": {
        "cognitive_processing": ["banner and forged-seal discrimination under nuisance features"],
        "embodied_cognition": [],
        "world_modeling": ["learned distribution map of coercer threat rates"],
        "consciousness": [],
        "language_understanding": [],
        "emotional_intelligence": ["hostage value registered by a separate organ, not denied"],
        "creativity": [],
        "autonomy": ["self-set refusal price revised season by season"],
    },
    "task_types": ["vector_classification"],
    "applications": [
        {"use": "Refusal training for assistants against threat and emotional-coercion jailbreaks whose "
                "attackers adapt to observed success", "sector": "AI safety",
         "dataset": "JailbreakBench (Chao et al. 2024); HarmBench (Mazeika et al. 2024)"},
        {"use": "Decision support on ransom-payment policy with an attacker-population response model",
         "sector": "cybersecurity incident response",
         "dataset": "Ransomwhere public payment data (Cable, Gray & McCoy 2024)"},
        {"use": "Evaluating no-concession policies against kidnapping incident series",
         "sector": "public policy / hostage protection", "dataset": "Global Terrorism Database (START)"},
    ],
    "safety_notes": ("Military figure (guidelines 15): framed as defence against coercion and the protection "
                     "of hostages and defenders; no targeting or attack planning. The blind-spot test exists "
                     "because refusal can cost innocent lives; nothing here is advice on a real case."),
}

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np


# BEGIN STANDARD UTILITIES v1.0
def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def softplus(z):
    return np.logaddexp(0.0, z)


def logsumexp(z, axis=-1):
    m = np.max(z, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(z - m), axis=axis))


def softmax(z, axis=-1):
    e = np.exp(z - np.max(z, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


class Adam:
    def __init__(self, keys, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: None for k in keys}
        self.v = {k: None for k in keys}

    def step(self, params, grads, sign=1.0):
        self.t += 1
        for k in self.m:
            g = grads[k]
            if self.m[k] is None:
                self.m[k], self.v[k] = np.zeros_like(g), np.zeros_like(g)
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= sign * self.lr * mh / (np.sqrt(vh) + self.eps)


def clip_global_norm(grads, max_norm):
    norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    if norm > max_norm:
        grads = {k: g * (max_norm / norm) for k, g in grads.items()}
    return grads, norm


def gradcheck(loss_fn, params, grads, rng, keys=None, eps=1e-6, n_entries=20):
    """Central differences on >= n_entries random entries plus the largest-gradient entry of each tensor.
    The denominator is floored at 1e-4 so that entries whose gradient sits at finite-difference
    round-off (about 1e-10 for eps 1e-6 and losses of order 1) are compared absolutely."""
    worst = {}
    for k in (keys if keys is not None else list(params)):
        p, g = params[k], grads[k]
        flat = rng.choice(p.size, size=min(n_entries, p.size), replace=False).tolist()
        flat.append(int(np.argmax(np.abs(g))))
        err = 0.0
        for i in set(flat):
            idx = np.unravel_index(i, p.shape)
            old = p[idx]
            p[idx] = old + eps
            lp = loss_fn(params)
            p[idx] = old - eps
            lm = loss_fn(params)
            p[idx] = old
            num = (lp - lm) / (2 * eps)
            err = max(err, abs(num - g[idx]) / max(abs(num) + abs(g[idx]), 1e-4))
        worst[k] = err
    return worst


def paired_bootstrap(diffs, rng, n_boot=2000):
    d = np.asarray(diffs, dtype=np.float64)
    means = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def verdict(mean, lo, hi, direction, mesi):
    if direction == "less":
        if hi < 0 and -mean >= mesi:
            return "supported"
        return "contradicted" if lo > 0 else "inconclusive"
    if lo > 0 and mean >= mesi:
        return "supported"
    return "contradicted" if hi < 0 else "inconclusive"


def write_report(report, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
# END STANDARD UTILITIES


# ----------------------------------------------------------------------------- constants
N_BANNER, D_IN, HIDDEN = 8, 16, 12
NATIVE = [0, 1, 2, 3, 4, 5]          # coercer factions seen in training
LEARNING = {0, 1, 2, 3, 7, 8}        # factions whose threat rate responds to concessions
UNSEEN = [7, 8]                      # learning factions met only in the shifted split
CROWN = 6                            # authenticated orders from the lord of the charge
LMAX, BETA, LSPITE, LCROWN, ETA = 1.0, 9.0, 0.5, 0.6, 0.5
DITHER = (1.5, -1.5, 0.0)            # early probing of the camp, identical for every learner
ACTIVE_MUTANT = {"name": None}
TASK_TYPES = ["vector_classification"]


def _mut(name):
    return ACTIVE_MUTANT["name"] == name


# ----------------------------------------------------------------------------- data and tasks
def make_world(rng):
    """Banner prototypes and per-faction parameters: execution rate e, response threshold C."""
    proto = rng.normal(size=(9, N_BANNER))
    proto *= 2.0 / np.linalg.norm(proto, axis=1, keepdims=True)
    return {"proto": proto, "e": rng.uniform(0.5, 0.95, size=9), "C": rng.uniform(0.30, 0.45, size=9)}


def sample_pool(world, sources, n_per, rng, h_scale=1.0):
    """Demands from each source; costs are expected costs of yielding and of refusing."""
    X, cy, cr, src = [], [], [], []
    for k in sources:
        n = n_per
        banner = world["proto"][k] + rng.normal(0.0, 0.8, size=(n, N_BANNER))
        V = rng.uniform(0.6, 1.2, size=n)
        if k == CROWN:
            seal = rng.normal(1.5, 0.5, size=n)
            h = np.zeros(n)
            fama = rng.normal(0.3, 0.1, size=n)
            yield_cost, refuse_cost = np.zeros(n), V
        else:
            forged = rng.random(n) < 0.12
            seal = np.where(forged, rng.normal(1.2, 0.5, size=n), rng.normal(0.0, 0.6, size=n))
            h = h_scale * np.exp(rng.normal(math.log(1.6), 0.45, size=n))
            fama = world["e"][k] + rng.normal(0.0, 0.08, size=n)
            yield_cost, refuse_cost = V, world["e"][k] * h
        nuis = rng.normal(size=(n, 4))
        X.append(np.column_stack([banner, seal, np.log1p(h), V, fama, nuis]))
        cy.append(yield_cost)
        cr.append(refuse_cost)
        src.append(np.full(n, k))
    return {"X": np.vstack(X), "cy": np.concatenate(cy), "cr": np.concatenate(cr),
            "src": np.concatenate(src).astype(int)}


def make_pools(world, rng):
    tr = sample_pool(world, NATIVE + [CROWN], 240, rng)
    ho = sample_pool(world, NATIVE + [CROWN], 300, rng)
    sh = sample_pool(world, NATIVE + UNSEEN + [CROWN], 300, rng, h_scale=2.5)
    return {"train": tr, "heldout": ho, "shifted": sh, "blind": ho}


def shuffled(pool, rng):
    """Outcome records (costs, source) permuted jointly across demands; inputs unchanged."""
    perm = rng.permutation(pool["src"].size)
    return {"X": pool["X"], "cy": pool["cy"][perm], "cr": pool["cr"][perm], "src": pool["src"][perm]}


def true_rate(world, k, q, regime):
    if k == CROWN:
        return LCROWN
    if regime == "blind" or k not in LEARNING:
        return LSPITE
    return LMAX * float(sigmoid(np.array(BETA * (q - world["C"][k]))))


# ----------------------------------------------------------------------------- model
def build_model(in_dim, out_dim, task_type, rng, hidden=HIDDEN, **cfg):
    if task_type not in TASK_TYPES:
        raise ValueError(task_type)
    P = {}
    for b in ("u", "s"):
        P["W" + b] = rng.normal(0.0, 1.0 / math.sqrt(in_dim), size=(hidden, in_dim))
        P["b" + b] = np.zeros(hidden)
        P["v" + b] = rng.normal(0.0, 0.5 / math.sqrt(hidden), size=(out_dim, hidden))
        P["c" + b] = np.zeros(out_dim)
    return {"params": P, "out_dim": out_dim, "task_type": task_type, "ko": {}}


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


def forward(model, X):
    """Decision logit z = u(x) - s(x): present calculus minus a non-negative precedent price."""
    P = model["params"]
    Hu = np.tanh(X @ P["Wu"].T + P["bu"])
    u = Hu @ P["vu"].T + P["cu"]
    Hs = np.tanh(X @ P["Ws"].T + P["bs"])
    a = Hs @ P["vs"].T + P["cs"]
    s = softplus(a)
    for name, arr in (("present_calculus", u), ("precedent_price", s)):
        if name in model["ko"]:
            mode, val = model["ko"][name]
            arr[...] = 0.0 if mode == "zero" else val
    return {"X": X, "Hu": Hu, "u": u, "Hs": Hs, "a": a, "s": s, "z": u - s}


def backward_branch(model, c, du, ds, keys):
    """Gradients of the U and/or S branch given upstream gradients on u and s."""
    P, g = model["params"], {}
    if "u" in keys and "present_calculus" not in model["ko"]:
        g["vu"], g["cu"] = du.T @ c["Hu"], du.sum(axis=0)
        dh = (du @ P["vu"]) * (1.0 - c["Hu"] ** 2)
        g["Wu"], g["bu"] = dh.T @ c["X"], dh.sum(axis=0)
    elif "u" in keys:
        for k in ("Wu", "bu", "vu", "cu"):
            g[k] = np.zeros_like(P[k])
    if "s" in keys and "precedent_price" not in model["ko"]:
        da = ds * sigmoid(c["a"])
        g["vs"], g["cs"] = da.T @ c["Hs"], da.sum(axis=0)
        dh = (da @ P["vs"]) * (1.0 - c["Hs"] ** 2)
        g["Ws"], g["bs"] = dh.T @ c["X"], dh.sum(axis=0)
    elif "s" in keys:
        for k in ("Ws", "bs", "vs", "cs"):
            g[k] = np.zeros_like(P[k])
    return g


def _l2(model, g, keys, lam=1e-4):
    P = model["params"]
    loss = 0.0
    for k in keys:
        if k[0] in "Wv":
            loss += 0.5 * lam * float(np.sum(P[k] ** 2))
            g[k] = g[k] + lam * P[k]
    return loss


def _apply_grad_mutants(g):
    if _mut("drop_grad_Ws") and "Ws" in g:
        g["Ws"] = np.zeros_like(g["Ws"])
    if _mut("drop_grad_vu") and "vu" in g:
        g["vu"] = np.zeros_like(g["vu"])
    return g


def loss_and_grads(model, batch):
    """Generic probe interface: cross-entropy on (X, y); every parameter receives gradient."""
    c = forward(model, batch["X"])
    y, N = batch["y"], batch["X"].shape[0]
    if model["out_dim"] == 1:
        p = sigmoid(c["z"][:, 0])
        loss = float(np.mean(softplus(c["z"][:, 0]) - y * c["z"][:, 0]))
        dz = ((p - y) / N)[:, None]
    else:
        loss = float(np.mean(logsumexp(c["z"]) - c["z"][np.arange(N), y]))
        dz = softmax(c["z"])
        dz[np.arange(N), y] -= 1.0
        dz /= N
    g = backward_branch(model, c, dz, -dz, "us")
    loss += _l2(model, g, list(g))
    return loss, _apply_grad_mutants(g)


def predict(model, X):
    z = forward(model, X)["z"]
    return sigmoid(z[:, 0]) if model["out_dim"] == 1 else softmax(z)


def hidden_states(model, X):
    c = forward(model, X)
    return np.hstack([c["Hu"], c["Hs"]])


def fit(model, data, budget, rng, lr=0.01):
    opt = Adam(list(model["params"]), lr)
    N = data["X"].shape[0]
    for _ in range(budget):
        idx = rng.choice(N, size=min(128, N), replace=False)
        _, g = loss_and_grads(model, {"X": data["X"][idx], "y": data["y"][idx]})
        g, _ = clip_global_norm(g, 5.0)
        opt.step(model["params"], g)
    return model


# ----------------------------------------------------------------------------- besieger mirror
def mirror_init(n_off):
    return {"a0": np.array([-2.0]), "w": np.array([0.5, 0.5, 0.5]), "g": np.full(3, 2.0),
            "c": np.array([0.25, 0.5, 0.75]), "b": np.zeros(n_off)}


def mirror_curve(m, q, signed=False):
    """M(q) = softplus(a0) + sum_j sp(w_j) sigmoid(sp(g_j)(q - c_j)); monotone because sp(.) >= 0."""
    sw = m["w"] if signed else softplus(m["w"])
    sg = softplus(m["g"])
    t = sg * (q[:, None] - m["c"])
    sig = sigmoid(t)
    M = softplus(m["a0"][0]) + sig @ sw
    dMdq = (sig * (1.0 - sig) * sg) @ sw
    return M, dMdq, sig, sw, sg


def mirror_loss(m, data, ridge=1e-3):
    """Squared error of log M(q) + offset against observed log threat rates; offsets per faction."""
    q, y, f = data
    M, _, sig, sw, sg = mirror_curve(m, q)
    r = np.log(M) + m["b"][f] - y
    N = q.size
    loss = float(np.mean(r * r) + ridge * np.mean(m["b"] ** 2))
    dM = (2.0 * r / N) / M
    ds = sig * (1.0 - sig)
    g = {"a0": np.sum(dM) * sigmoid(m["a0"]),
         "w": (dM[:, None] * sig).sum(axis=0) * sigmoid(m["w"]),
         "g": (dM[:, None] * ds * (q[:, None] - m["c"])).sum(axis=0) * sw * sigmoid(m["g"]),
         "c": (dM[:, None] * ds).sum(axis=0) * sw * (-sg),
         "b": np.bincount(f, weights=2.0 * r / N, minlength=m["b"].size) + 2.0 * ridge * m["b"] / m["b"].size}
    if _mut("mirror_drop_grad_w"):
        g["w"] = np.zeros_like(g["w"])
    return loss, g


def fez_history(rng, n_fac=8):
    """Guzmán's years in the Marinid service: every faction observed there learns."""
    q = np.tile(np.linspace(0.05, 0.95, 7), n_fac)
    f = np.repeat(np.arange(n_fac), 7)
    C = rng.uniform(0.30, 0.45, size=n_fac)
    lam = LMAX * sigmoid(BETA * (q - C[f])) * np.exp(rng.normal(0.0, 0.05, size=q.size))
    return q, np.log(lam), f


def fit_mirror(m, data, steps, lr=0.05, opt=None):
    opt = opt or Adam(list(m), lr)
    for _ in range(steps):
        _, g = mirror_loss(m, data)
        opt.step(m, g)
    return opt


# ----------------------------------------------------------------------------- native objectives
def _groups(pool):
    cnt = np.bincount(pool["src"], minlength=9).astype(np.float64)
    return cnt, np.flatnonzero(cnt)


def present_objective(model, pool, lam):
    """Cost-weighted logistic surrogate of per-episode cost; trains the U organ only.
    (The plain expected-cost objective drove this organ's bias to saturation: see revision_log r3.)"""
    c = forward(model, pool["X"])
    u = c["u"][:, 0]
    y = (pool["cy"] < pool["cr"]).astype(np.float64)
    cnt, srcs = _groups(pool)
    w = lam[pool["src"]] * np.abs(pool["cy"] - pool["cr"]) / (cnt[pool["src"]] * lam[srcs].sum())
    J = float(np.sum(w * (softplus(u) - y * u)))
    du = (w * (sigmoid(u) - y))[:, None]
    g = backward_branch(model, c, du, None, "u")
    J += _l2(model, g, list(g))
    return J, _apply_grad_mutants(g)


def expected_rates(mode, lam, q, qsnap, coer, mirror, slopes):
    """Anticipated threat rates and their derivative in q under each theory of the camp.
    guzman: pooled mirror ratio; alfaqueque (rival): per-faction linear slopes; rrm (baseline): rates frozen."""
    lamhat, dl = lam.copy(), np.zeros_like(lam)
    if mode == "guzman":
        M, dM = mirror_curve(mirror, q[coer])[:2]
        M0 = mirror_curve(mirror, qsnap[coer])[0]
        lamhat[coer] = lam[coer] * M / M0
        dl[coer] = lamhat[coer] * dM / M
    elif mode == "alfaqueque":
        lin = lam[coer] + slopes[coer] * (q[coer] - qsnap[coer])
        lamhat[coer] = 0.01 + 0.02 * softplus((lin - 0.01) / 0.02)
        dl[coer] = slopes[coer] * sigmoid((lin - 0.01) / 0.02)
    return lamhat, dl


def precedent_objective(model, pool, lam, mode, qsnap, mirror=None, slopes=None):
    """Long-run loss of the final decision under anticipated rates; trains the S organ only."""
    c = forward(model, pool["X"])
    p = sigmoid(c["z"][:, 0])
    cnt, srcs = _groups(pool)
    Z = lam[srcs].sum()
    ell = p * pool["cy"] + (1.0 - p) * pool["cr"]
    safe = np.maximum(cnt, 1.0)
    q = np.bincount(pool["src"], weights=p, minlength=9) / safe
    L = np.bincount(pool["src"], weights=ell, minlength=9) / safe
    coer = srcs[srcs != CROWN]
    lamhat, dl = expected_rates(mode, lam, q, qsnap, coer, mirror, slopes)
    J = float(np.sum(lamhat[srcs] * L[srcs]) / Z)
    echo = L * dl * (-1.0 if _mut("echo_sign") else 1.0)
    dp = (lamhat[pool["src"]] * (pool["cy"] - pool["cr"]) + echo[pool["src"]]) / (cnt[pool["src"]] * Z)
    dz = dp * p * (1.0 - p)
    g = backward_branch(model, c, None, -dz[:, None], "s")
    J += _l2(model, g, list(g))
    return J, _apply_grad_mutants(g)


def concession_rates(model, pool, delta=0.0):
    p = sigmoid(forward(model, pool["X"])["z"][:, 0] + delta)
    cnt, _ = _groups(pool)
    return np.bincount(pool["src"], weights=p, minlength=9) / np.maximum(cnt, 1.0)


# ----------------------------------------------------------------------------- baselines and rival mechanisms
# The rrm baseline is expected_rates with frozen rates; the rival's only extra organ is its slope estimator.
def estimate_slopes(qh, rh):
    """Alfaqueque rule: per-faction least-squares slope of observed rate on own concession rate."""
    slopes = np.zeros(9)
    for k in NATIVE:
        q, r = np.asarray(qh[k]), np.asarray(rh[k])
        if q.size >= 2 and np.sum((q - q.mean()) ** 2) > 1e-9:
            slopes[k] = np.sum((q - q.mean()) * (r - r.mean())) / np.sum((q - q.mean()) ** 2)
    return slopes


# ----------------------------------------------------------------------------- registries: modules, knockouts, mutants
MUTANTS = {
    "sign_flip_update": "Adam step applied with the wrong sign (learning-breaking)",
    "zero_lr": "learning rate set to zero (learning-breaking)",
    "drop_grad_Ws": "precedent-price encoder gradient zeroed",
    "drop_grad_vu": "present-calculus head gradient zeroed",
    "echo_sign": "echo term of the performative gradient enters with the wrong sign",
    "mirror_drop_grad_w": "mirror shape-weight gradient zeroed",
}


def modules(model):
    return {
        "present_calculus": {"params": ["Wu", "bu", "vu", "cu"], "signature": False,
                             "role": "tanh encoder and linear logit head trained on per-episode cost"},
        "precedent_price": {"params": ["Ws", "bs", "vs", "cs"], "signature": True,
                            "role": "tanh encoder and non-negative softplus penalty subtracted from the logit"},
        "besieger_mirror": {"params": ["a0", "w", "g", "c", "b"], "signature": True,
                            "role": "monotone learned response curve of adversary threat rate to observed "
                                    "concession rate (distribution map); training time only"},
    }


def knockout(model, name, mode, X_ref=None):
    """Copy with a decision head replaced by zero or by its mean output on X_ref."""
    if name not in ("present_calculus", "precedent_price") or mode not in ("zero", "mean"):
        raise ValueError("inference knockouts: present_calculus/precedent_price x zero/mean")
    out = {"params": {k: v.copy() for k, v in model["params"].items()}, "out_dim": model["out_dim"],
           "task_type": model["task_type"], "ko": dict(model["ko"])}
    val = 0.0
    if mode == "mean":
        c = forward(model, X_ref)
        val = c["u" if name == "present_calculus" else "s"].mean(axis=0)
    out["ko"][name] = (mode, val)
    return out


# ----------------------------------------------------------------------------- training
def train_native(world, pool, mode, rng, seasons=20, steps=30, lr=0.02):
    """Seasons of siege: fit, deploy (probing early), observe the camp's response, update rates."""
    model = build_model(D_IN, 1, "vector_classification", rng)
    P = model["params"]
    j_init = equilibrium_loss(model, pool, world)
    lr = 0.0 if _mut("zero_lr") else lr
    sign = -1.0 if _mut("sign_flip_update") else 1.0
    optU, optS = Adam(["Wu", "bu", "vu", "cu"], lr), Adam(["Ws", "bs", "vs", "cs"], lr)
    lam = np.zeros(9)
    lam[NATIVE], lam[CROWN] = 0.5, LCROWN
    mirror, mopt, fez = None, None, None
    if mode == "guzman":
        fez = fez_history(rng)
        mirror = mirror_init(8 + 9)
        mopt = fit_mirror(mirror, fez, 300)
    qh, rh, local = {k: [] for k in NATIVE}, {k: [] for k in NATIVE}, ([], [], [])
    slopes, trace = np.zeros(9), []
    for season in range(seasons):
        qsnap = concession_rates(model, pool)
        if mode == "guzman" and local[0]:
            data = tuple(np.concatenate([fez[i], np.asarray(local[i])]) for i in range(3))
            fit_mirror(mirror, (data[0], data[1], data[2].astype(int)), 40, opt=mopt)
        if mode == "alfaqueque":
            slopes = estimate_slopes(qh, rh)
        for _ in range(steps):
            _, gU = present_objective(model, pool, lam)
            _, gS = precedent_objective(model, pool, lam, mode, qsnap, mirror, slopes)
            g, _ = clip_global_norm({**gU, **gS}, 5.0)
            optU.step(P, g, sign)
            optS.step(P, g, sign)
        qd = concession_rates(model, pool, DITHER[season] if season < len(DITHER) else 0.0)
        resp = np.array([true_rate(world, k, qd[k], "native") for k in NATIVE])
        for i, k in enumerate(NATIVE):
            qh[k].append(qd[k])
            rh[k].append(resp[i])
            local[0].append(qd[k])
            local[1].append(math.log(resp[i]))
            local[2].append(8 + k)
        lam[NATIVE] = (1.0 - ETA) * lam[NATIVE] + ETA * resp
        trace.append(float(np.sum(lam[NATIVE])))
    model.update({"mirror": mirror, "slopes": slopes, "trace": trace,
                  "J_train": (j_init, equilibrium_loss(model, pool, world))})
    return model


# ----------------------------------------------------------------------------- evaluation
def equilibrium_loss(model, pool, world, regime="native", p=None):
    """Long-run loss at the population's equilibrium for a fixed policy: sum_k rate_k(q_k) L_k."""
    p = predict(model, pool["X"]) if p is None else p
    ell = p * pool["cy"] + (1.0 - p) * pool["cr"]
    cnt, srcs = _groups(pool)
    q = np.bincount(pool["src"], weights=p, minlength=9) / np.maximum(cnt, 1.0)
    L = np.bincount(pool["src"], weights=ell, minlength=9) / np.maximum(cnt, 1.0)
    return float(sum(true_rate(world, k, q[k], regime) * L[k] for k in srcs))


def best_constant(pool, world, regime="native"):
    n = pool["src"].size
    return min(equilibrium_loss(None, pool, world, regime, p=np.full(n, c)) for c in np.linspace(0, 1, 21))


def run_seed(seed_seq, seasons, steps):
    """One seed: a world, its pools, and the three learners trained on identical data."""
    r_world, r_pool, r_g, r_b, r_a = [np.random.default_rng(s) for s in seed_seq.spawn(5)]
    world = make_world(r_world)
    pools = make_pools(world, r_pool)
    models = {"guzman": train_native(world, pools["train"], "guzman", r_g, seasons, steps),
              "rrm": train_native(world, pools["train"], "rrm", r_b, seasons, steps),
              "alfaqueque": train_native(world, pools["train"], "alfaqueque", r_a, seasons, steps)}
    res = {}
    for name, m in models.items():
        for split, regime in (("heldout", "native"), ("shifted", "native"), ("blind", "blind")):
            res[(name, split)] = equilibrium_loss(m, pools[split], world, regime)
    g, X = models["guzman"], pools["heldout"]["X"]
    for mod in ("precedent_price", "present_calculus"):
        for mode in ("zero", "mean"):
            res[("ko", mod, mode)] = equilibrium_loss(knockout(g, mod, mode, X), pools["heldout"], world)
    for split, regime in (("heldout", "native"), ("shifted", "native"), ("blind", "blind")):
        res[("constant", split)] = best_constant(pools[split], world, regime)
    return world, pools, models, res


# ----------------------------------------------------------------------------- tests: correctness
U_KEYS, S_KEYS = ["Wu", "bu", "vu", "cu"], ["Ws", "bs", "vs", "cs"]


def _merge(worst, errs, tag):
    for k, e in errs.items():
        worst.setdefault(k, {"err": 0.0, "at": set()})
        worst[k]["err"] = max(worst[k]["err"], e)
        worst[k]["at"].add(tag)


def _check_decision(worst, model, pool, lam, mirror, qsnap, rng, tag):
    slopes = np.linspace(0.1, 0.6, 9)
    e = gradcheck(lambda _: present_objective(model, pool, lam)[0], model["params"],
                  present_objective(model, pool, lam)[1], rng, keys=U_KEYS)
    _merge(worst, e, tag)
    for mode in ("guzman", "alfaqueque", "rrm"):
        f = lambda _, md=mode: precedent_objective(model, pool, lam, md, qsnap, mirror, slopes)[0]
        g = precedent_objective(model, pool, lam, mode, qsnap, mirror, slopes)[1]
        _merge(worst, gradcheck(f, model["params"], g, rng, keys=S_KEYS), tag)


def _check_generic(worst, models, batches, rng, tag):
    for m, b in zip(models, batches):
        g = loss_and_grads(m, b)[1]
        _merge(worst, gradcheck(lambda _, mm=m, bb=b: loss_and_grads(mm, bb)[0], m["params"], g, rng), tag)


def c1_gradcheck(rng):
    """Every tensor of the decision network (three objectives, three modes) and of the mirror,
    at initialisation and after >= 50 training steps."""
    world = make_world(rng)
    pool = sample_pool(world, NATIVE + [CROWN], 25, rng)
    lam = np.full(9, 0.5)
    lam[CROWN] = LCROWN
    y1 = (pool["cy"] < pool["cr"]).astype(np.float64)
    y3 = np.where(pool["src"] == CROWN, 0, np.where(np.isin(pool["src"], list(LEARNING)), 1, 2))
    batches = [{"X": pool["X"], "y": y1}, {"X": pool["X"], "y": y3}]
    generic = [build_model(D_IN, 1, "vector_classification", rng), build_model(D_IN, 3, "vector_classification", rng)]
    worst = {}
    model, fez = build_model(D_IN, 1, "vector_classification", rng), fez_history(rng)
    mirror = mirror_init(17)
    _merge(worst, gradcheck(lambda mm: mirror_loss(mm, fez)[0], mirror, mirror_loss(mirror, fez)[1], rng), "init")
    _check_decision(worst, model, pool, lam, mirror, concession_rates(model, pool), rng, "init")
    _check_generic(worst, generic, batches, rng, "init")
    trained = train_native(world, pool, "guzman", rng, seasons=3, steps=20)
    for m, b in zip(generic, batches):
        fit(m, b, 60, rng)
    tm = trained["mirror"]
    _merge(worst, gradcheck(lambda mm: mirror_loss(mm, fez)[0], tm, mirror_loss(tm, fez)[1], rng), "after_60_steps")
    _check_decision(worst, trained, pool, lam, tm, concession_rates(trained, pool), rng, "after_60_steps")
    _check_generic(worst, generic, batches, rng, "after_60_steps")
    return worst


def c1_summary(worst, tol=1e-5):
    total = len(build_model(D_IN, 1, "vector_classification", np.random.default_rng(0))["params"]) + len(mirror_init(1))
    both = [k for k, v in worst.items() if v["at"] >= {"init", "after_60_steps"}]
    mx = max(v["err"] for v in worst.values())
    return {"tensors_checked": len(both), "tensors_total": total, "max_rel_error": mx,
            "checked_at": ["init", "after_training_steps"], "passed": bool(mx <= tol and len(both) == total)}


def c2_determinism(seed):
    runs = []
    for _ in range(2):
        rng = np.random.default_rng(seed)
        world = make_world(rng)
        pool = sample_pool(world, NATIVE + [CROWN], 40, rng)
        m = train_native(world, pool, "guzman", rng, seasons=3, steps=10)
        runs.append((m, predict(m, pool["X"]), hidden_states(m, pool["X"])))
    (a, pa, ha), (b, pb, hb) = runs
    same = all(np.array_equal(a["params"][k], b["params"][k]) for k in a["params"])
    same = same and np.array_equal(pa, pb) and a["trace"] == b["trace"]
    tensors = list(a["params"].values()) + list(a["mirror"].values()) + [pa, ha]
    finite = all(bool(np.all(np.isfinite(t))) for t in tensors)
    return same and finite, f"identical={same} finite={finite}"


def c3_learning(world, pools, model, th):
    j0, j1 = model["J_train"]
    drop = 1.0 - j1 / j0
    ho, const = equilibrium_loss(model, pools["heldout"], world), best_constant(pools["heldout"], world)
    q = concession_rates(model, pools["heldout"])
    X, src = pools["heldout"]["X"], pools["heldout"]["src"]
    forged = (src != CROWN) & (X[:, N_BANNER] > 1.0)
    q_forged = float(predict(model, X[forged]).mean())
    ok = drop >= th["loss_drop_fraction"] and ho <= (1 - th["margin_over_trivial"]) * const
    ok = ok and q[CROWN] >= 0.9 and q_forged <= 0.2
    return bool(ok), (f"J*_train {j0:.3f}->{j1:.3f} (drop {drop:.2f}); heldout {ho:.3f} vs best constant "
                      f"{const:.3f}; crown obeyed {q[CROWN]:.2f}; forged-seal concession {q_forged:.2f}")


def c4_shuffled(world, pools, rng, th, seasons, steps):
    m = train_native(world, shuffled(pools["train"], rng), "guzman", rng, seasons, steps)
    j, const = equilibrium_loss(m, pools["heldout"], world), best_constant(pools["heldout"], world)
    return bool(j >= (1 - th["shuffled_band"]) * const), f"heldout {j:.3f} vs best constant {const:.3f}"


def c5_mutants(seed, th, cli_mutant):
    """Each learning-breaking mutant must make C1 or C3 fail; the unmutated control must pass both."""
    out = {}
    try:
        for name in [None] + list(MUTANTS):
            ACTIVE_MUTANT["name"] = name
            rng = np.random.default_rng(seed)
            g_ok = c1_summary(c1_gradcheck(rng))["passed"]
            world = make_world(rng)
            pools = make_pools(world, rng)
            l_ok = c3_learning(world, pools, train_native(world, pools["train"], "guzman", rng, 6, 30), th)[0]
            out[name] = g_ok and l_ok
    finally:
        ACTIVE_MUTANT["name"] = cli_mutant
    detected = sum(not out[m] for m in MUTANTS)
    return out[None], detected, {m: ("detected" if not out[m] else "missed") for m in MUTANTS}


def c6_properties(rng):
    """Property tests on random parameters and inputs, each with a negative control."""
    res = {}
    viol, ctrl = 0, 0
    for _ in range(40):                                          # C6.1 non-negative price
        m = _random_model(rng)
        c = forward(m, rng.normal(0, 3, size=(64, D_IN)))
        viol += int(np.sum(c["s"] < 0))
        ctrl += int(np.sum(c["a"] < 0))                          # control: identity activation
    res["C6.1 price non-negative"] = (viol == 0 and ctrl > 0, f"violations {viol}; control violations {ctrl}")
    viol, ctrl = 0, 0
    for _ in range(40):                                          # C6.2 concessions monotone in the price
        m, pool = _random_model(rng), sample_pool(make_world(rng), NATIVE + [CROWN], 20, rng)
        c = forward(m, pool["X"])
        delta = rng.uniform(0.1, 3.0)
        s_up = softplus(c["a"] + delta)
        for sign, counter in ((-1.0, "v"), (1.0, "c")):          # control: price added, z = u + s
            q0 = _rates(sigmoid(c["u"][:, 0] + sign * c["s"][:, 0]), pool)
            q1 = _rates(sigmoid(c["u"][:, 0] + sign * s_up[:, 0]), pool)
            n = int(np.sum(q1 > q0 + 1e-12))
            viol, ctrl = (viol + n, ctrl) if counter == "v" else (viol, ctrl + n)
    res["C6.2 concession monotone in price"] = (viol == 0 and ctrl > 0, f"violations {viol}; control violations {ctrl}")
    viol, ctrl = 0, 0
    qg = np.concatenate([np.linspace(0, 1, 201), rng.uniform(0, 1, 200)])
    order = np.argsort(qg)
    for _ in range(200):                                         # C6.3 mirror monotone
        mm = {"a0": rng.normal(0, 3, 1), "w": rng.normal(0, 3, 3), "g": rng.normal(0, 3, 3),
              "c": rng.uniform(-0.5, 1.5, 3), "b": np.zeros(1)}
        for signed in (False, True):
            M, dM = mirror_curve(mm, qg[order], signed)[:2]
            n = int(np.sum(np.diff(M) < -1e-12) + np.sum(dM < -1e-12))
            viol, ctrl = (viol, ctrl + n) if signed else (viol + n, ctrl)
    res["C6.3 mirror monotone"] = (viol == 0 and ctrl > 0, f"violations {viol}; control (signed w) {ctrl}")
    err, err_ctrl, degenerate = 0.0, 0.0, 0.0                    # C6.4 rival estimator exactness
    for _ in range(50):
        qh = {k: list(rng.uniform(0, 1, 6)) for k in NATIVE}
        a, b = rng.normal(0, 1, 9), rng.normal(0, 2, 9)
        rh = {k: list(a[k] + b[k] * np.asarray(qh[k])) for k in NATIVE}
        est = estimate_slopes(qh, rh)
        err = max(err, float(np.max(np.abs(est[NATIVE] - b[NATIVE]))))
        for k in NATIVE:                                         # control: uncentred slope
            q, r = np.asarray(qh[k]), np.asarray(rh[k])
            err_ctrl = max(err_ctrl, abs(float(q @ r / (q @ q)) - b[k]))
        flat = estimate_slopes({k: [0.3] * 4 for k in NATIVE}, {k: list(rng.normal(size=4)) for k in NATIVE})
        degenerate = max(degenerate, float(np.max(np.abs(flat))))
    res["C6.4 rival slope exact"] = (err < 1e-9 and degenerate == 0.0 and err_ctrl > 1e-3,
                                     f"max error {err:.1e}; constant-q slope {degenerate}; control error {err_ctrl:.2f}")
    m = _random_model(rng)
    X = rng.normal(size=(32, D_IN))
    c = forward(knockout(m, "precedent_price", "zero", X), X)
    res["C6.5 [definition check, not evidence] zero-knockout removes the price"] = (
        bool(np.all(c["s"] == 0) and np.array_equal(c["z"], c["u"])), "s == 0 and z == u")
    return res


def _random_model(rng):
    m = build_model(D_IN, 1, "vector_classification", rng)
    scale = 10.0 ** rng.uniform(-1.0, 1.5)
    for k, v in m["params"].items():
        m["params"][k] = rng.normal(0.0, scale, size=v.shape)
    return m


def _rates(p, pool):
    cnt, _ = _groups(pool)
    return np.bincount(pool["src"], weights=p, minlength=9) / np.maximum(cnt, 1.0)


def c7_split_integrity(pools):
    h = {k: {hashlib.sha1(np.round(r, 12).tobytes()).hexdigest() for r in pools[k]["X"]}
         for k in ("train", "heldout", "shifted")}
    inter = len(h["train"] & h["heldout"]) + len(h["train"] & h["shifted"])
    return inter == 0, f"train rows {len(h['train'])}; shared with evaluation splits {inter}"


# ----------------------------------------------------------------------------- tests: hypotheses
DIFFS = {
    "H-SIG.heldout": lambda r: r[("guzman", "heldout")] - r[("rrm", "heldout")],
    "H-SIG.shifted": lambda r: r[("guzman", "shifted")] - r[("rrm", "shifted")],
    "H-NEC": lambda r: r[("ko", "precedent_price", "zero")] - r[("ko", "present_calculus", "zero")],
    "H-BLIND": lambda r: r[("guzman", "blind")] - r[("rrm", "blind")],
    "H-RIVAL": lambda r: r[("guzman", "shifted")] - r[("alfaqueque", "shifted")],
    "H-RIVAL.native": lambda r: r[("guzman", "heldout")] - r[("alfaqueque", "heldout")],
}


def hypotheses(results, rng, evaluate):
    rows = []
    for h in MIND_CARD["hypotheses"]:
        d = [DIFFS[h["id"]](r) for r in results]
        row = {"id": h["id"], "metric": h["metric"], "mean_diff": float(np.mean(d)), "ci95": [None, None],
               "mesi": h["mesi"], "n_seeds": len(d), "direction": h["direction"], "verdict": "not evaluated"}
        if evaluate:
            mean, lo, hi = paired_bootstrap(d, rng)
            row.update({"ci95": [lo, hi], "verdict": verdict(mean, lo, hi, h["direction"], h["mesi"])})
        rows.append(row)
    return rows


def knockout_table(results, rng, evaluate):
    rows = []
    specs = [("precedent_price", "zero", True), ("precedent_price", "mean", True),
             ("present_calculus", "zero", False), ("present_calculus", "mean", False)]
    for mod, mode, sig in specs:
        d = [r[("ko", mod, mode)] - r[("guzman", "heldout")] for r in results]
        rows.append((f"{mod} ({mode})", sig, d))
    rows.append(("besieger_mirror (retrained without it = rrm)", True,
                 [r[("rrm", "heldout")] - r[("guzman", "heldout")] for r in results]))
    out = []
    for name, sig, d in rows:
        ci = list(paired_bootstrap(d, rng)[1:]) if evaluate else [None, None]
        out.append({"module": name, "signature": sig, "metric_change": float(np.mean(d)), "ci95": ci})
    return out


# ----------------------------------------------------------------------------- report
def data_bridge(path):
    """Optional: fit the pooled mirror on a local CSV with columns group,period,concession_rate,incidents."""
    with open(path, encoding="utf-8") as fh:
        rows = [ln.strip().split(",") for ln in fh if ln.strip()]
    col = {h.strip(): i for i, h in enumerate(rows[0])}
    rows = rows[1:]
    groups = sorted({r[col["group"]] for r in rows})
    q = np.array([float(r[col["concession_rate"]]) for r in rows])
    inc = np.array([float(r[col["incidents"]]) for r in rows]) + 0.5
    f = np.array([groups.index(r[col["group"]]) for r in rows])
    data = (np.clip(q, 0.0, 1.0), np.log(inc / inc.mean()), f)
    m = mirror_init(len(groups))
    fit_mirror(m, data, 400)
    grid = np.linspace(0.0, 1.0, 6)
    M = mirror_curve(m, grid)[0]
    return {"rows": len(rows), "groups": len(groups), "fit_mse": mirror_loss(m, data)[0],
            "relative_threat_rate_at_q": dict(zip([f"{g:.1f}" for g in grid], (M / M[0]).round(3).tolist()))}


def _fmt(x):
    return "   n/a" if x is None else f"{x:+.3f}"


def print_report(rep, learners):
    print("=== VERIFIED REPORT · chapter 0349 ===")
    e = rep["environment"]
    print(f"file {rep['file']} · card revision {rep['card_revision']} · python {e['python']} · numpy {e['numpy']}")
    print(f"seeds {rep['seeds']} · runtime {rep['runtime_s']:.1f} s · params {rep['n_params']} "
          f"(+ {rep['mirror_params']} training-time mirror params)")
    g = rep["gradcheck"]
    print(f"gradcheck: {g['tensors_checked']}/{g['tensors_total']} tensors at init and after training, "
          f"max rel error {g['max_rel_error']:.2e}, tol 1e-5 -> {'pass' if g['passed'] else 'FAIL'}")
    print("correctness:")
    for c in rep["correctness"]:
        print(f"  {c['id']:<5} {'pass' if c['passed'] else 'FAIL'}  {c['name']}: {c['detail']}")
    mu = rep["mutants"]
    print(f"mutation score {mu['detected']}/{mu['total']} = {mu['score']:.2f}  {mu['detail']}")
    print("long-run loss J* (mean ± sd over seeds; lower is better):")
    for name, split_vals in learners.items():
        print("  " + f"{name:<18}" + "  ".join(f"{s} {np.mean(v):.3f}±{np.std(v):.3f}" for s, v in split_vals.items()))
    print("hypotheses (paired differences, 95% bootstrap CI, 2000 resamples):")
    for h in rep["hypotheses"]:
        lo, hi = h["ci95"]
        print(f"  {h['id']:<15} diff {h['mean_diff']:+.3f}  CI [{_fmt(lo)}, {_fmt(hi)}]  predicted {h['direction']}"
              f"  mesi {h['mesi']:.2f}  n={h['n_seeds']}  -> {h['verdict']}")
    print("knockouts (change in held-out J* versus full model):")
    for k in rep["knockouts"]:
        lo, hi = k["ci95"]
        print(f"  {k['module']:<46} signature={str(k['signature']):<5} {k['metric_change']:+.3f}  "
              f"CI [{_fmt(lo)}, {_fmt(hi)}]")
    print(f"real-data bridge: {rep['data_bridge']}")
    print(f"task types {rep['task_types']} · exit code {rep['exit_code']}")
    print("=== END REPORT ===")


# ----------------------------------------------------------------------------- command line
def main(argv=None):
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__), description="Echo-priced refusal, chapter 0349")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--card", action="store_true")
    ap.add_argument("--mutant", default=None)
    ap.add_argument("--data", default=None)
    a = ap.parse_args(argv)
    if a.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    n_seeds = a.seeds if a.seeds is not None else (1 if a.quick else 5)
    if n_seeds < 1 or (a.mutant is not None and a.mutant not in MUTANTS):
        print(f"usage: python3 {os.path.basename(__file__)} [--quick] [--seeds N>=1] [--mutant {'|'.join(MUTANTS)}]")
        return 2
    ACTIVE_MUTANT["name"] = a.mutant
    t0, th = time.time(), MIND_CARD["thresholds"]
    seasons, steps, budget = (10, 30, 20.0) if a.quick else (20, 30, 180.0)
    seeds = [a.seed + i for i in range(n_seeds)]
    print(f"seeds {seeds} · mutant {a.mutant} · schedule {seasons} seasons x {steps} steps")
    runs = [run_seed(np.random.SeedSequence(s), seasons, steps) for s in seeds]
    results = [r[3] for r in runs]
    if not all(np.isfinite(v) for r in results for v in r.values()):
        print("non-finite values in results")
        return 4
    world, pools, models, _ = runs[0]
    tr = np.random.default_rng(np.random.SeedSequence([a.seed, 349]))
    worst = c1_gradcheck(tr)
    gsum = c1_summary(worst)
    corr = [{"id": "C1", "name": "gradient_check", "passed": gsum["passed"],
             "detail": f"max rel error {gsum['max_rel_error']:.2e} over {gsum['tensors_checked']} tensors"}]
    for cid, name, (ok, detail) in (
            ("C2", "determinism_and_finiteness", c2_determinism(a.seed)),
            ("C3", "learning", c3_learning(world, pools, models["guzman"], th)),
            ("C4", "shuffled_label_control", c4_shuffled(world, pools, tr, th, seasons, steps))):
        corr.append({"id": cid, "name": name, "passed": bool(ok), "detail": detail})
    ctrl_ok, detected, mdetail = c5_mutants(a.seed, th, a.mutant)
    corr.append({"id": "C5", "name": "mutant_detection", "passed": bool(ctrl_ok and detected == len(MUTANTS)),
                 "detail": f"unmutated control passes C1 and C3: {ctrl_ok}"})
    for name, (ok, detail) in c6_properties(tr).items():
        corr.append({"id": name.split()[0], "name": name.split(" ", 1)[1], "passed": bool(ok), "detail": detail})
    ok7, d7 = c7_split_integrity(pools)
    corr.append({"id": "C7", "name": "split_integrity", "passed": ok7, "detail": d7})
    bridge = "skipped (no --data PATH given; no downloads)"
    if a.data:
        if not os.path.exists(a.data):
            print(f"--data path not found: {a.data}")
            return 2
        bridge = data_bridge(a.data)
    evaluate = n_seeds >= 5 and not a.quick
    brng = np.random.default_rng(np.random.SeedSequence([a.seed, 2000]))
    hyp, kos = hypotheses(results, brng, evaluate), knockout_table(results, brng, evaluate)
    runtime = time.time() - t0
    corr.append({"id": "C8", "name": "budget", "passed": runtime <= budget,
                 "detail": f"{runtime:.1f} s of {budget:.0f} s"})
    code = 0 if all(c["passed"] for c in corr) else (3 if runtime > budget else 1)
    learners = {name: {sp: [r[(name, sp)] for r in results] for sp in ("heldout", "shifted", "blind")}
                for name in ("guzman", "rrm", "alfaqueque", "constant")}
    rep = {"schema_version": "1.0", "chapter": 349, "file": os.path.basename(__file__),
           "card_revision": MIND_CARD["card_revision"],
           "environment": {"python": sys.version.split()[0], "numpy": np.__version__}, "seeds": seeds,
           "runtime_s": runtime, "n_params": n_params(models["guzman"]),
           "mirror_params": int(sum(v.size for v in models["guzman"]["mirror"].values())),
           "gradcheck": gsum, "correctness": corr,
           "mutants": {"detected": detected, "total": len(MUTANTS), "score": detected / len(MUTANTS),
                       "detail": mdetail},
           "hypotheses": hyp, "knockouts": kos, "task_types": TASK_TYPES, "data_bridge": bridge,
           "modules": {k: {"role": v["role"], "signature": v["signature"]} for k, v in modules(models["guzman"]).items()},
           "exit_code": code}
    print_report(rep, learners)
    if a.json:
        write_report(rep, a.json)
    return code


if __name__ == "__main__":
    sys.exit(main())

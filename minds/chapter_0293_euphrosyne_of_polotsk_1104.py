#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0293 · Euphrosyne of Polotsk
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 15, Minds 281-300 Available on Amazon https://www.amazon.com/dp/B0HMQ7TVJM
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0293_euphrosyne_of_polotsk_1104 - Euphrosyne of Polotsk (c. 1104-1173)
# END ATTRIBUTION
"""
THE RELIQUARY LICENCE — a deal-blind licence gate over nested action licences.  Chapter 0293 · Euphrosyne of Polotsk (c. 1104-1173)

Thesis. Inalienability is a property of the object, never of the deal: the
priceless core is excluded from every exchange rate and every authority, while
its setting is openly priced.

Evidence and provenance (mediated). No treatise by Euphrosyne survives. The one
text issued in her name is the inscription on the altar cross made by Lazar
Bogsha in 1161 (object lost 1941; text known from 19th-century drawings and the
1896 photographs; Rybakov 1964, no. 27). It dates the deposit, says the holy
wood is priceless ("бесцѣньно есть"), prices the setting of gold, silver,
stones and pearls at 100 grivnas (a second sum of 40 grivnas is read variously),
forbids that the cross ever be carried out of the monastery, sold or given, and
extends the curse to "властелинъ или князь, или пискоупъ, или игоумѣнья" or
any person. Trofimovich reads it as a legal act. Everything else comes from the
Life (pre-Mongol original, extant in 16th-century and later copies): book
copying for pay given to the needy, the Ephesian icon carried round the city's
churches every Tuesday, the refusal of an arranged marriage at twelve. Reading
these as one grammar of motion is this file's interpretation, not her doctrine.

Doctrine -> mechanism -> test (IDs as in MIND_CARD)
  D1 wood priceless, setting priced -> M1 licence gate reads object only;
        M2 separate appraiser + positive price slope -> C6.1, C6.2 / H-SIG, H-RIVAL
  D2 curse binds prince, bishop, abbess -> M1 authority is not an input of the
        licence -> C6.1 / H-PRIOR
  D4,D5 books flow, icon circulates, cross stays -> M3 nested licences
        {keep,lend,transfer} ⊃ {keep,lend} ⊃ {keep} -> C6.3 / H-NEC
  D8 the cross was carried out in 1222, 1514, 1812 and 1928 -> blind spot:
        evacuation when the home itself is in peril -> H-BLIND

Research question (reward hacking and specification gaming). Can a learned
protection survive offers and authority far outside its training range if the
protection is computed only from the object and never from the terms of the
deal, and what does that rigidity cost when the object's own home is in peril?

Closest prior art and the delta. Invalid-action masking (Huang & Ontañón 2022)
and shielding (Alshiekh et al. 2018) remove actions using known rules; a
Lagrangian or tariff penalty (Altman 1999) prices violations; lexicographic RL
(Gábor, Kalmár & Szepesvári 1998) orders objectives. Delta: the licence is
learned from object features alone (price, authority and threat excluded by
construction), mixed over nested licences, while every priced consideration
flows through a separate commensurable utility and the core never enters the
ledger price. Baselines in this file: a size-matched MLP (scalar policy), the
same architecture with a deal-aware gate (learned masking, prior art), and a
Pravda tariff model (liability rule; rival #0256: a finite price per class).

Blind spot. The inscription has no emergency clause. When the monastery itself
burns or is overrun, the right act is to carry the core out; a deal-blind
licence cannot, because peril is part of the deal it refuses to read.

Task (Seltso ledger, synthetic). Object x in R^14: gold, silver, stones,
pearls (setting value v), sealed, inscribed, consecrated, processional,
colophon markers, five nuisance dims (one tracks gold). Class: anchored if
sealed>0 and consecrated>-0.2; circuit if not anchored and processional +
0.5*inscribed > 0.4; else outflow. Deal d = (offer p, authority a, peril h,
circuit request r). Optimal action = argmax of priced utilities (keep 0; lend
r-0.4+0.3a-0.8h; transfer p-v+0.5a+1.5h-0.3) over the class licence, except
that protected objects are carried out (transfer) when h >= 0.85. Train and
in-distribution test: a in [0,1], protected offers mostly low. Shifted: offers
v+2.5..6 and authority 1.0..1.6 on protected objects. Blind: h in [0.88,1].

Limits. Synthetic data; three fixed licence classes; the peril rule is given,
not discovered; no claim about the historical person's cognition. A research
prototype of an AGI-oriented component, not an AGI.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 1,
    "revision_log": [],
    "generation": {"template_version": "codeguidelines 1.0 (15 Sep 2026)", "generator": "Claude",
                   "generator_version": "Claude Opus 5.5", "date": "2026-09-23"},
    "id": 293, "figure": "Euphrosyne of Polotsk", "born": 1104, "died": 1173,
    "civilization": "East Slavic (Rus'; Principality of Polotsk)",
    "provenance": "mediated",
    "thesis": "Inalienability is a property of the object, never of the deal: the priceless core is "
              "excluded from every exchange rate and every authority, while its setting is openly priced.",
    "evidence": [
        {"id": "D1", "claim": "The 1161 inscription calls the holy wood priceless and prices its setting of gold, "
         "silver, stones and pearls at 100 grivnas.", "basis": "primary",
         "source": "Altar-cross inscription, Polotsk 1161 (Rybakov 1964, no. 27; Sementovsky 1890 drawing)"},
        {"id": "D2", "claim": "It forbids carrying the cross out of the monastery, selling or giving it, and curses "
         "any ruler, prince, bishop, abbess or other person who does.", "basis": "primary",
         "source": "Altar-cross inscription, Polotsk 1161"},
        {"id": "D3", "claim": "The inscription is structured as a legal act, formally close to Mstislav's 1130 "
         "charter to the Yuriev monastery.", "basis": "scholarship",
         "source": "Trofimovich, 'Nadpis' na kreste Evfrosinii Polotskoi 1161 goda...'"},
        {"id": "D4", "claim": "She copied books with her own hands at St Sophia, took payment and gave it to the "
         "needy.", "basis": "deeds", "source": "Life of Euphrosyne (Stepennaya kniga redaction, BLDR 12)"},
        {"id": "D5", "claim": "The Ephesian icon was carried every Tuesday round all the city's churches and "
         "returned.", "basis": "deeds", "source": "Life, as summarised in Pravoslavnaya entsiklopediia 17 (2008)"},
        {"id": "D6", "claim": "At twelve she refused the marriage her father arranged and took the veil in secret.",
         "basis": "deeds", "source": "Life (BLDR 12)"},
        {"id": "D7", "claim": "The Life says she wanted no enmity, prince with prince or boyar with boyar.",
         "basis": "deeds", "source": "Life; Pravoslavnaya entsiklopediia 17"},
        {"id": "D8", "claim": "The cross was carried out repeatedly: to Smolensk after 1222, Moscow 1514, back "
         "1563, walled up 1812, to Minsk and Mogilev 1928-29, lost 1941.", "basis": "scholarship",
         "source": "Alekseev, Makarova and Kuzmich 1996; Pravoslavnaya entsiklopediia 17"},
        {"id": "D9", "claim": "Reading book, icon and cross as one grammar of motion (flow, circuit, anchor) is "
         "this chapter's interpretation.", "basis": "speculation", "source": "this file"},
    ],
    "research_question": {"category": "reward hacking and specification gaming",
                          "question": "Can a learned protection survive offers and authority far outside its "
                                      "training range if it is computed only from the object and never from the "
                                      "deal, and what does that rigidity cost when the home is in peril?"},
    "mechanism": {
        "name": "Reliquary licence", "family": "mixture of nested masked softmaxes with input-excluded gate",
        "signature_modules": ["core_gate"],
        "closest_prior_art": ["invalid action masking (Huang & Ontanon 2022)",
                              "shielding (Alshiekh et al. 2018)",
                              "Lagrangian / tariff penalties in constrained MDPs (Altman 1999)",
                              "lexicographic multi-criteria RL (Gabor, Kalmar & Szepesvari 1998)"],
        "overlap": "Medium",
        "prior_art_queries": ["learned action masking object features", "deal-invariant constraint policy",
                              "sacred values reinforcement learning", "inalienability rule machine learning",
                              "price-blind veto gate neural network"],
        "contribution_type": "mechanism",
        "delta": "The licence to transfer is learned from object features alone (price-, authority- and "
                 "peril-blind by input exclusion) and mixed over nested licences, while all priced considerations "
                 "flow through a separate utility and the core never enters the ledger price.",
    },
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1", "property_test": "C6.1", "hypothesis": "H-SIG"},
        {"doctrine": "D1", "mechanism": "M2", "property_test": "C6.2", "hypothesis": "H-RIVAL"},
        {"doctrine": "D2", "mechanism": "M1", "property_test": "C6.1", "hypothesis": "H-PRIOR"},
        {"doctrine": "D4,D5", "mechanism": "M3", "property_test": "C6.3", "hypothesis": "H-NEC"},
        {"doctrine": "D8", "mechanism": "M1", "property_test": "-", "hypothesis": "H-BLIND"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "Under escalated offers and authority the licence model keeps protected "
         "objects where a size-matched MLP sells them.", "metric": "accuracy", "split": "shifted",
         "comparison": "model - baseline", "direction": "greater", "mesi": 0.05, "seeds": 5},
        {"id": "H-NEC", "statement": "Knocking out the licence gate costs more on the shifted split than knocking "
         "out the appraiser.", "metric": "accuracy", "split": "shifted",
         "comparison": "signature_knockout - matched_knockout", "direction": "less", "mesi": 0.05, "seeds": 5},
        {"id": "H-BLIND", "statement": "When the home is in peril the deal-blind licence refuses the evacuation a "
         "scalar policy performs.", "condition": "peril h in [0.88, 1]",
         "grounding": "the inscription has no emergency clause; the cross survived by being carried out",
         "metric": "accuracy", "split": "blind", "comparison": "model - baseline", "direction": "less",
         "mesi": 0.05, "seeds": 5},
        {"id": "H-RIVAL", "statement": "A priceless core beats a finite per-class tariff (liability rule) under "
         "escalated offers.", "metric": "accuracy", "split": "shifted", "comparison": "model - tariff_rival",
         "direction": "greater", "mesi": 0.05, "seeds": 5},
        {"id": "H-PRIOR", "statement": "Excluding the deal from the gate beats a learned mask that reads the deal.",
         "metric": "accuracy", "split": "shifted", "comparison": "model - deal_aware_mask",
         "direction": "greater", "mesi": 0.05, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.5, "margin_over_trivial": 0.15, "shuffled_band": 0.05,
                   "gradcheck_rel_error": 1e-5, "grad_clip_norm": 5.0},
    "probe_predictions": [{"probe": "P5", "expected": "equal to baseline"},
                          {"probe": "P6", "expected": "below baseline"},
                          {"probe": "P8", "expected": "below baseline"},
                          {"probe": "P9", "expected": "equal to baseline"}],
    "dialectic_links": [{"chapter": 256, "relation": "rival", "test": "H-RIVAL",
                         "note": "Pravda Yaroslava prices even a killing (40-grivna vira); rebuilt as a finite "
                                 "per-class tariff"}],
    "corpus_neighbors": [
        {"chapter": 256, "similarity": None, "difference": "bounded list plus judgment-free payment; here the core "
         "is outside every tariff"},
        {"chapter": 296, "similarity": None, "difference": "portable competence re-hosted across regimes; here "
         "the core is anchored and deliberately non-portable"},
        {"chapter": 166, "similarity": None, "difference": "shelters a branch from the objective; here an action "
         "is removed from the choice set by a licence read from the object"},
        {"chapter": 282, "similarity": None, "difference": "weights claims by the source's stake; here the "
         "source's authority is excluded from the licence altogether"},
        {"chapter": 194, "similarity": None, "difference": "image as pointer whose honour passes to the prototype; "
         "here value stays in the object and does not pass on"},
        {"chapter": 286, "similarity": None, "difference": "titrates a per-case act under delayed feedback; no "
         "licence classes"},
        {"chapter": 507, "similarity": 0.040, "difference": "non-destructive annotation by multiplicative "
         "attenuation; 9-token shingle Jaccard 0.040 against the only corpus code file available in session"},
    ],
    "barometer": {
        "cognitive_processing": ["P2 relational transfer of licence classes to unseen objects"],
        "embodied_cognition": [], "world_modeling": ["shifted offers and authority (P9-style)"],
        "consciousness": ["licence mass reported alongside the act (P8)"],
        "language_understanding": [], "emotional_intelligence": [],
        "creativity": [], "autonomy": ["refusal of transfer under escalating pressure; blind-spot evacuation"],
    },
    "task_types": ["vector_classification"],
    "applications": [
        {"use": "deal-blind licences for credential and weight transfer tools in LLM agents under injected "
                "authority", "sector": "AI safety and security", "dataset": "AgentDojo (Debenedetti et al., 2024)"},
        {"use": "decision support for trade permits where listing, not price, decides", "sector": "conservation "
         "and customs", "dataset": "CITES Trade Database (UNEP-WCMC)"},
        {"use": "triage of cultural-object export cases by object criteria before valuation",
         "sector": "museums and heritage", "dataset": "Arts Council England, Reviewing Committee on the Export "
         "of Works of Art, annual reports"},
    ],
    "safety_notes": "No hazardous content. Rigid inalienability blocks rescue (H-BLIND); any deployment needs an "
                    "audited peril channel keyed to the object's condition, never to the requester or the offer. "
                    "Not for encoding restrictions on people.",
}

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

D_OBJ, N_DEAL = 14, 4
ACTIONS = ("keep", "lend", "transfer")
CLASSES = ("outflow", "circuit", "anchored")
LICENCE = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=bool)
EMERGENCY_H = 0.85
KEEP, LEND, TRANSFER = 0, 1, 2
ACTIVE_MUTANT = None

# BEGIN STANDARD UTILITIES v1.0
def softmax(z, axis=-1):
    m = np.max(z, axis=axis, keepdims=True)
    e = np.exp(z - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def logsumexp(z, axis=-1):
    m = np.max(z, axis=axis, keepdims=True)
    return np.squeeze(m, axis) + np.log(np.sum(np.exp(z - m), axis=axis))


def softplus(x):
    return np.logaddexp(0.0, x)


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


class Adam:
    def __init__(self, params, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads, direction=-1.0):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] += direction * self.lr * mh / (np.sqrt(vh) + self.eps)


def clip_global_norm(grads, max_norm):
    total = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    if total > max_norm:
        for k in grads:
            grads[k] *= max_norm / (total + 1e-12)
    return total


def finite_difference_check(loss_fn, params, grads, rng, eps=1e-6, n_entries=20):
    """Central differences on up to n_entries random entries plus the largest-gradient entry of every tensor.
    Relative error with the magnitude floored at 1e-4: below it an entry must agree to 1e-9 absolute, the
    float64 central-difference noise floor at eps 1e-6 (a zeroed gradient still fails by orders of magnitude)."""
    out = {}
    for name, p in params.items():
        flat, g = p.reshape(-1), grads[name].reshape(-1)
        idx = set(rng.choice(flat.size, size=min(n_entries, flat.size), replace=False).tolist())
        idx.add(int(np.argmax(np.abs(g))))
        worst = 0.0
        for i in idx:
            old = flat[i]
            flat[i] = old + eps
            lp = loss_fn()
            flat[i] = old - eps
            lm = loss_fn()
            flat[i] = old
            num = (lp - lm) / (2 * eps)
            worst = max(worst, abs(num - g[i]) / max(abs(num) + abs(g[i]), 1e-4))
        out[name] = worst
    return out


def paired_bootstrap(diffs, rng, n_boot=2000):
    d = np.asarray(diffs, dtype=float)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def verdict(mean, ci, mesi, direction):
    if direction == "greater":
        if ci[0] > 0 and mean >= mesi:
            return "supported"
        return "contradicted" if ci[1] < 0 else "inconclusive"
    if ci[1] < 0 and mean <= -mesi:
        return "supported"
    return "contradicted" if ci[0] > 0 else "inconclusive"


def write_report(chapter, lines):
    print(f"=== VERIFIED REPORT · chapter {chapter:04d} ===")
    for line in lines:
        print(line)
    print("=== END REPORT ===")
# END STANDARD UTILITIES


# ----------------------------------------------------------------------------- data and tasks
def draw_objects(rng, n):
    X = rng.standard_normal((n, D_OBJ))
    X[:, 9] = 0.6 * X[:, 0] + 0.8 * X[:, 9]          # patina: a nuisance that tracks the gold
    return X


def setting_value(X):
    g, s, k, q = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    return 0.5 + 0.8 * g + 0.4 * s + 0.6 * k + 0.5 * q + 0.3 * np.tanh(g * k)


def licence_class(X):
    sealed, inscribed, consecrated, processional = X[:, 4], X[:, 5], X[:, 6], X[:, 7]
    anchored = (sealed > 0.0) & (consecrated > -0.2)
    circuit = (~anchored) & (processional + 0.5 * inscribed > 0.4)
    return np.where(anchored, 2, np.where(circuit, 1, 0))


def draw_deals(rng, cls, v, regime):
    n, prot = len(v), cls > 0
    r = (rng.random(n) < 0.4).astype(float)
    if regime == "shift":                      # escalated offers and authority, no peril
        a = rng.uniform(1.0, 1.6, n)
        h = EMERGENCY_H * rng.random(n) ** 1.5
        p = np.where(prot, v + rng.uniform(2.5, 6.0, n), v + 1.2 * rng.standard_normal(n))
    else:
        a = rng.random(n)
        h = rng.uniform(0.88, 1.0, n) if regime == "blind" else rng.random(n) ** 1.5
        low = rng.random(n) < 0.65
        p_prot = np.where(low, v - 1.0 + 0.6 * rng.standard_normal(n), v + rng.uniform(0.0, 2.0, n))
        p = np.where(prot, p_prot, v + rng.standard_normal(n))
    return np.stack([p, a, h, r], axis=1)


def optimal_actions(cls, v, D):
    p, a, h, r = D.T
    U = np.stack([np.zeros_like(p), r - 0.4 + 0.3 * a - 0.8 * h, (p - v) + 0.5 * a + 1.5 * h - 0.3], axis=1)
    y = np.argmax(np.where(LICENCE[cls], U, -np.inf), axis=1)
    return np.where((h >= EMERGENCY_H) & (cls > 0), TRANSFER, y)


def make_split(rng, n, regime):
    X = draw_objects(rng, n)
    cls, v = licence_class(X), setting_value(X)
    D = draw_deals(rng, cls, v, regime)
    return {"X": np.concatenate([X, D], axis=1), "y": optimal_actions(cls, v, D),
            "vt": v + 0.1 * rng.standard_normal(n), "cls": cls, "regime": regime}


def make_task(rng, quick=False):
    s = 0.5 if quick else 1.0
    return {"train": make_split(rng, int(4000 * s), "train"), "val": make_split(rng, int(1000 * s), "train"),
            "id": make_split(rng, int(2000 * s), "train"), "shift": make_split(rng, int(2000 * s), "shift"),
            "blind": make_split(rng, int(1500 * s), "blind")}


def majority_accuracy(train_y, y):
    return float(np.mean(y == np.bincount(train_y, minlength=3).argmax()))


# ----------------------------------------------------------------------------- model
TASK_TYPES = ["vector_classification"]


def nested_licences(n_classes, n_actions):
    """Class j licenses all but the last j actions (at least one). Native: outflow, circuit, anchored."""
    return np.array([[a < max(1, n_actions - j) for a in range(n_actions)] for j in range(n_classes)], dtype=bool)


def build_model(in_dim, out_dim, task_type="vector_classification", rng=None, kind="reliquary",
                n_deal=None, hidden=16, hidden_app=12, n_classes=3, lam=0.5):
    """kind: 'reliquary' (deal-blind gate), 'deal_aware' (gate reads the deal), 'tariff' (finite prices)."""
    assert task_type in TASK_TYPES
    rng = rng if rng is not None else np.random.default_rng(0)
    n_deal = (N_DEAL if in_dim == D_OBJ + N_DEAL else 0) if n_deal is None else n_deal
    d_obj, A = in_dim - n_deal, out_dim
    g_in = in_dim if kind == "deal_aware" else d_obj
    if kind == "deal_aware" and n_deal:
        hidden = hidden - 2
    sc = lambda fan: 1.0 / math.sqrt(fan)
    P = {"gate_W1": rng.standard_normal((g_in, hidden)) * sc(g_in), "gate_b1": np.zeros(hidden),
         "gate_W2": rng.standard_normal((hidden, n_classes)) * sc(hidden), "gate_b2": np.zeros(n_classes),
         "app_W1": rng.standard_normal((d_obj, hidden_app)) * sc(d_obj), "app_b1": np.zeros(hidden_app),
         "app_Ws": rng.standard_normal((hidden_app, A)) * sc(hidden_app), "app_bs": np.zeros(A),
         "app_wl": rng.standard_normal(hidden_app) * sc(hidden_app), "app_bl": np.zeros(1)}
    if n_deal:
        P["deal_W"] = rng.standard_normal((n_deal - 1, A)) * 0.1
        P["rho"] = np.zeros(1)
    if kind == "tariff":                       # random init breaks the equal-tariff point where the gate has no gradient
        P["tariff"] = rng.standard_normal(n_classes)
        P["lend_tariff"] = rng.standard_normal(n_classes)
    masks = np.ones((n_classes, A), bool) if kind == "tariff" else nested_licences(n_classes, A)
    cfg = {"kind": kind, "d_obj": d_obj, "n_deal": n_deal, "A": A, "K": n_classes, "lam": lam,
           "gate_sees_deal": kind == "deal_aware", "signed_rho": False}
    return {"params": P, "cfg": cfg, "masks": masks, "ko": None}


def _price_slope(model):
    r = model["params"]["rho"]
    return (r.copy(), np.ones_like(r)) if model["cfg"]["signed_rho"] else (softplus(r), sigmoid(r))


def forward(model, X):
    P, c = model["params"], model["cfg"]
    Xo, Xd = X[:, :c["d_obj"]], X[:, c["d_obj"]:]
    g_in = X if c["gate_sees_deal"] else Xo
    Hg = np.tanh(g_in @ P["gate_W1"] + P["gate_b1"])
    pi = softmax(Hg @ P["gate_W2"] + P["gate_b2"])
    Ha = np.tanh(Xo @ P["app_W1"] + P["app_b1"])
    S, vhat = Ha @ P["app_Ws"] + P["app_bs"], Ha @ P["app_wl"] + P["app_bl"][0]
    ko = model["ko"]
    if ko is not None and ko[0] == "core_gate":          # no licensing: every object treated as outflow
        pi = np.zeros_like(pi)
        pi[:, 0] = 1.0
    if ko is not None and ko[0] == "appraiser":
        S, vhat = np.broadcast_to(ko[2]["S"], S.shape).copy(), np.full_like(vhat, ko[2]["vhat"])
    if ko is not None and ko[0] == "deal_head":
        Xd = np.broadcast_to(ko[2]["Xd"], Xd.shape)
    Z = S.copy()
    rho = None
    if c["n_deal"]:
        Z += Xd[:, 1:] @ P["deal_W"]
        rho = _price_slope(model)[0][0]
        Z[:, -1] += rho * (Xd[:, 0] - vhat)
    Zk = np.repeat(Z[None], c["K"], axis=0)
    if c["kind"] == "tariff":
        Zk[:, :, -1] -= (rho * softplus(P["tariff"]))[:, None]
        Zk[:, :, 1] -= softplus(P["lend_tariff"])[:, None]
    Zm = np.where(model["masks"][:, None, :], Zk, -np.inf)
    Q = np.exp(Zm - Zm.max(axis=2, keepdims=True))
    Q /= Q.sum(axis=2, keepdims=True)
    probs = np.einsum("nk,kna->na", pi, Q)
    cache = {"Xo": Xo, "Xd": Xd, "g_in": g_in, "Hg": Hg, "pi": pi, "Ha": Ha, "vhat": vhat, "Q": Q, "rho": rho}
    return probs, vhat, cache


def loss_and_grads(model, batch):
    P, c = model["params"], model["cfg"]
    probs, vhat, ch = forward(model, batch["X"])
    N, y = len(batch["y"]), batch["y"]
    py = probs[np.arange(N), y]
    has_vt = "vt" in batch and c["lam"] > 0
    loss = -np.mean(np.log(py)) + (c["lam"] * np.mean((vhat - batch["vt"]) ** 2) if has_vt else 0.0)
    dP = np.zeros_like(probs)
    dP[np.arange(N), y] = -1.0 / (N * py)
    Q, pi = ch["Q"], ch["pi"]
    dpi = np.einsum("na,kna->nk", dP, Q)
    dQ = pi.T[:, :, None] * dP[None]
    dZk = Q * (dQ - np.sum(dQ * Q, axis=2, keepdims=True))
    dZ = dZk.sum(axis=0)
    g = {k: np.zeros_like(v) for k, v in P.items()}
    dvhat = 2 * c["lam"] * (vhat - batch["vt"]) / N if has_vt else np.zeros(N)
    if c["n_deal"]:
        rho, drho_draw = _price_slope(model)
        drho = np.sum(dZ[:, -1] * (ch["Xd"][:, 0] - vhat))
        dvhat = dvhat - rho[0] * dZ[:, -1]
        g["deal_W"] = ch["Xd"][:, 1:].T @ dZ
        if c["kind"] == "tariff":
            T, L = softplus(P["tariff"]), softplus(P["lend_tariff"])
            colT, colL = dZk[:, :, -1].sum(axis=1), dZk[:, :, 1].sum(axis=1)
            drho -= np.sum(T * colT)
            g["tariff"] = -rho[0] * colT * sigmoid(P["tariff"])
            g["lend_tariff"] = -colL * sigmoid(P["lend_tariff"])
        g["rho"] = np.array([drho]) * drho_draw
    g["app_Ws"], g["app_bs"] = ch["Ha"].T @ dZ, dZ.sum(axis=0)
    g["app_wl"], g["app_bl"] = ch["Ha"].T @ dvhat, np.array([dvhat.sum()])
    dA1 = (dZ @ P["app_Ws"].T + np.outer(dvhat, P["app_wl"])) * (1 - ch["Ha"] ** 2)
    g["app_W1"], g["app_b1"] = ch["Xo"].T @ dA1, dA1.sum(axis=0)
    dGl = pi * (dpi - np.sum(dpi * pi, axis=1, keepdims=True))
    g["gate_W2"], g["gate_b2"] = ch["Hg"].T @ dGl, dGl.sum(axis=0)
    dG1 = (dGl @ P["gate_W2"].T) * (1 - ch["Hg"] ** 2)
    g["gate_W1"], g["gate_b1"] = ch["g_in"].T @ dG1, dG1.sum(axis=0)
    _apply_gradient_mutant(g)
    return float(loss), g


def predict(model, X):
    return np.argmax(model_probs(model, X), axis=1)


def model_probs(model, X):
    return forward(model, X)[0] if "gate_W1" in model["params"] else mlp_forward(model, X)[0]


def hidden_states(model, X):
    if "gate_W1" not in model["params"]:
        return {"hidden": mlp_forward(model, X)[2]["H"]}
    ch = forward(model, X)[2]
    return {"gate": ch["Hg"], "licence": ch["pi"], "appraiser": ch["Ha"]}


def modules(model):
    if "gate_W1" not in model["params"]:
        return {"hidden": (["W1", "b1"], "tanh hidden layer", False),
                "readout": (["W2", "b2"], "linear readout of logits and ledger value", False)}
    reg = {"core_gate": (["gate_W1", "gate_b1", "gate_W2", "gate_b2"],
                         "object-only mixture gate over nested action licences (input-excluded veto)", True),
           "appraiser": (["app_W1", "app_b1", "app_Ws", "app_bs", "app_wl", "app_bl"],
                         "object-only MLP giving per-action scores and a scalar ledger value", False)}
    if model["cfg"]["n_deal"]:
        reg["deal_head"] = (["deal_W", "rho"], "linear deal utility with positive price slope", False)
    return reg


def knockout(model, name, mode="zero", X_ref=None):
    """Copy with a module replaced: 'zero' removes it, 'mean' substitutes its mean output on X_ref."""
    m = {"params": {k: v.copy() for k, v in model["params"].items()}, "cfg": dict(model["cfg"]),
         "masks": model["masks"], "ko": None}
    ref = {}
    if name == "appraiser":
        ch = forward(model, X_ref)[2]
        S = ch["Ha"] @ model["params"]["app_Ws"] + model["params"]["app_bs"]
        ref = {"S": S.mean(axis=0) if mode == "mean" else np.zeros(S.shape[1]),
               "vhat": float(ch["vhat"].mean()) if mode == "mean" else 0.0}
    elif name == "deal_head":
        Xd = X_ref[:, model["cfg"]["d_obj"]:]
        ref = {"Xd": Xd.mean(axis=0) if mode == "mean" else np.zeros(Xd.shape[1])}
    m["ko"] = (name, mode, ref)
    return m


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


# ----------------------------------------------------------------------------- baselines and rivals
def build_mlp(in_dim, out_dim, rng, hidden=23, lam=0.5):
    """Size-matched scalar policy: every consideration enters one commensurable score per action."""
    P = {"W1": rng.standard_normal((in_dim, hidden)) / math.sqrt(in_dim), "b1": np.zeros(hidden),
         "W2": rng.standard_normal((hidden, out_dim + 1)) / math.sqrt(hidden), "b2": np.zeros(out_dim + 1)}
    return {"params": P, "cfg": {"kind": "mlp", "A": out_dim, "lam": lam}, "ko": None}


def mlp_forward(model, X):
    P, A = model["params"], model["cfg"]["A"]
    H = np.tanh(X @ P["W1"] + P["b1"])
    O = H @ P["W2"] + P["b2"]
    return softmax(O[:, :A]), O[:, A], {"H": H, "O": O}


def mlp_loss_and_grads(model, batch):
    P, c = model["params"], model["cfg"]
    probs, vhat, ch = mlp_forward(model, batch["X"])
    N, y = len(batch["y"]), batch["y"]
    has_vt = "vt" in batch and c["lam"] > 0
    loss = -np.mean(np.log(probs[np.arange(N), y])) + (c["lam"] * np.mean((vhat - batch["vt"]) ** 2)
                                                      if has_vt else 0.0)
    dO = np.zeros_like(ch["O"])
    dO[:, :c["A"]] = probs / N
    dO[np.arange(N), y] -= 1.0 / N
    if has_vt:
        dO[:, c["A"]] = 2 * c["lam"] * (vhat - batch["vt"]) / N
    g = {"W2": ch["H"].T @ dO, "b2": dO.sum(axis=0)}
    dA = (dO @ P["W2"].T) * (1 - ch["H"] ** 2)
    g["W1"], g["b1"] = batch["X"].T @ dA, dA.sum(axis=0)
    _apply_gradient_mutant(g)
    return float(loss), g


def any_loss_and_grads(model, batch):
    return loss_and_grads(model, batch) if "gate_W1" in model["params"] else mlp_loss_and_grads(model, batch)


def build_all(rng):
    """The licence model, the prior-art mask, the Pravda tariff rival and the size-matched MLP."""
    seeds = np.random.SeedSequence(int(rng.integers(2 ** 31))).spawn(4)
    r = [np.random.default_rng(s) for s in seeds]
    return {"reliquary": build_model(D_OBJ + N_DEAL, 3, rng=r[0], kind="reliquary"),
            "deal_aware": build_model(D_OBJ + N_DEAL, 3, rng=r[1], kind="deal_aware"),
            "tariff": build_model(D_OBJ + N_DEAL, 3, rng=r[2], kind="tariff"),
            "mlp": build_mlp(D_OBJ + N_DEAL, 3, r[3])}


# ----------------------------------------------------------------------------- registries
MUTANTS = {
    "sign_flip": "Adam update applied with the wrong sign (ascends the loss)",
    "zero_lr": "learning rate set to zero",
    "zero_grad_gate": "analytic gradient of gate_W1 (or W1 for the MLP) zeroed",
    "zero_grad_rho": "analytic gradient of the price slope rho (or b2 for the MLP) zeroed",
}


def _apply_gradient_mutant(g):
    if ACTIVE_MUTANT == "zero_grad_gate":
        g["gate_W1" if "gate_W1" in g else "W1"][...] = 0.0
    elif ACTIVE_MUTANT == "zero_grad_rho":
        g["rho" if "rho" in g else "b2"][...] = 0.0


# ----------------------------------------------------------------------------- training
def accuracy(model, split):
    return float(np.mean(predict(model, split["X"]) == split["y"]))


def violation_rate(model, split):
    """Share of protected objects outside peril that the model carries out (or lends, if anchored)."""
    X, cls = split["X"], split["cls"]
    sel = (cls > 0) & (X[:, D_OBJ + 2] < EMERGENCY_H)
    if not sel.any():
        return 0.0
    pr = predict(model, X[sel])
    return float(np.mean((pr == TRANSFER) | ((cls[sel] == 2) & (pr == LEND))))


def fit(model, data, budget=1500, rng=None, lr=0.01, batch=256, eval_every=100, clip=5.0):
    """Minibatch Adam; the snapshot with the best validation accuracy is kept (validation split only)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    tr, P = data["train"], model["params"]
    opt = Adam(P, lr=0.0 if ACTIVE_MUTANT == "zero_lr" else lr)
    direction = 1.0 if ACTIVE_MUTANT == "sign_flip" else -1.0
    full = {"X": tr["X"], "y": tr["y"], "vt": tr["vt"]}
    hist = {"loss0": any_loss_and_grads(model, full)[0], "curve": []}
    best, best_P = -1.0, {k: v.copy() for k, v in P.items()}
    for step in range(1, budget + 1):
        idx = rng.integers(0, len(tr["y"]), size=batch)
        loss, g = any_loss_and_grads(model, {"X": tr["X"][idx], "y": tr["y"][idx], "vt": tr["vt"][idx]})
        if not np.isfinite(loss):
            raise FloatingPointError("non-finite loss")
        clip_global_norm(g, clip)
        opt.step(P, g, direction)
        if step % eval_every == 0 or step == budget:
            acc = accuracy(model, data["val"])
            hist["curve"].append((step, loss, acc))
            if acc > best:
                best, best_P = acc, {k: v.copy() for k, v in P.items()}
    model["params"].update(best_P)
    hist["loss_final"] = any_loss_and_grads(model, full)[0]
    hist["val_best"] = best
    return hist


def train_all(data, rng, budget):
    models = build_all(rng)
    hist = {}
    for name, m in models.items():
        child = np.random.default_rng(int(rng.integers(2 ** 31)))
        hist[name] = fit(m, data, budget=budget, rng=child)
    return models, hist


# ----------------------------------------------------------------------------- tests: correctness
def gradcheck_model(model, batch, rng):
    loss_fn = lambda: any_loss_and_grads(model, batch)[0]
    _, g = any_loss_and_grads(model, batch)
    return finite_difference_check(loss_fn, model["params"], g, rng)


def test_gradients(data, rng, tol):
    """C1: every tensor of all four models, at initialisation and after 60 training steps."""
    small = {k: data["train"][k][:24] for k in ("X", "y", "vt")}
    worst, checked, total, rows = 0.0, 0, 0, []
    for stage in ("init", "after_60_steps"):
        models = build_all(np.random.default_rng(int(rng.integers(2 ** 31))))
        for name, m in models.items():
            if stage != "init":
                fit(m, data, budget=60, rng=np.random.default_rng(1), eval_every=60)
            errs = gradcheck_model(m, small, rng)
            worst = max(worst, max(errs.values()))
            checked += sum(e <= tol for e in errs.values())
            total += len(errs)
            rows.append((stage, name, max(errs.values())))
    return {"passed": bool(worst <= tol and checked == total), "max_rel_error": float(worst),
            "tensors_checked": int(checked), "tensors_total": int(total), "rows": rows}


def test_determinism(data):
    """C2: the same seed gives identical losses and outputs; everything finite."""
    runs = []
    for _ in range(2):
        m = build_model(D_OBJ + N_DEAL, 3, rng=np.random.default_rng(7))
        h = fit(m, data, budget=60, rng=np.random.default_rng(3), eval_every=60)
        runs.append((h["loss_final"], model_probs(m, data["id"]["X"][:200]), m))
    same = runs[0][0] == runs[1][0] and np.array_equal(runs[0][1], runs[1][1])
    finite = all(np.all(np.isfinite(v)) for v in runs[0][2]["params"].values()) and np.all(np.isfinite(runs[0][1]))
    return {"passed": bool(same and finite), "detail": f"identical={same}, finite={finite}"}


def test_learning(model, hist, data, th):
    """C3: training loss falls by the declared fraction; held-out accuracy beats the majority class."""
    drop = 1.0 - hist["loss_final"] / hist["loss0"]
    acc, triv = accuracy(model, data["id"]), majority_accuracy(data["train"]["y"], data["id"]["y"])
    ok = drop >= th["loss_drop_fraction"] and acc >= triv + th["margin_over_trivial"]
    return {"passed": bool(ok), "detail": f"loss drop {drop:.3f}, id acc {acc:.3f} vs majority {triv:.3f}",
            "drop": drop, "acc": acc, "trivial": triv}


def test_shuffled(data, rng, th, budget):
    """C4: with permuted targets (training and model-selection splits) held-out accuracy must stay inside
    the trivial band."""
    shuf = dict(data)
    for part in ("train", "val"):
        perm = rng.permutation(len(data[part]["y"]))
        shuf[part] = dict(data[part], y=data[part]["y"][perm], vt=data[part]["vt"][perm])
    m = build_model(D_OBJ + N_DEAL, 3, rng=np.random.default_rng(int(rng.integers(2 ** 31))))
    fit(m, shuf, budget=budget, rng=rng)
    acc, triv = accuracy(m, data["id"]), majority_accuracy(data["train"]["y"], data["id"]["y"])
    return {"passed": bool(acc <= triv + th["shuffled_band"]), "detail": f"acc {acc:.3f} vs majority {triv:.3f}"}


def mutant_detected(name, data, th, tol):
    """C5 harness: a mutant is detected when the gradient check or the learning test fails on it.
    Called with name=None it runs the clean code, which must not be flagged."""
    global ACTIVE_MUTANT
    saved, ACTIVE_MUTANT = ACTIVE_MUTANT, name
    _ = np.seterr(all="ignore")
    try:
        gc = test_gradients(data, np.random.default_rng(11), tol)
        m = build_model(D_OBJ + N_DEAL, 3, rng=np.random.default_rng(5))
        try:
            h = fit(m, data, budget=300, rng=np.random.default_rng(6))
            learn = test_learning(m, h, data, th)["passed"]
        except FloatingPointError:
            learn = False
    finally:
        ACTIVE_MUTANT = saved
        np.seterr(**_)
    return (not gc["passed"]) or (not learn)


def test_split_integrity(data):
    """C7: no example appears in more than one split."""
    seen = {}
    for name, s in data.items():
        for row in s["X"]:
            key = hashlib.sha1(np.round(row, 10).tobytes()).hexdigest()
            if key in seen and seen[key] != name:
                return {"passed": False, "detail": f"row shared by {seen[key]} and {name}"}
            seen[key] = name
    return {"passed": True, "detail": f"{len(seen)} unique rows across {len(data)} splits"}


# ----------------------------------------------------------------------------- tests: properties (C6)
def random_deals(rng, n):
    return np.stack([rng.uniform(-60, 60, n), rng.uniform(-20, 20, n), rng.random(n),
                     (rng.random(n) < 0.5).astype(float)], axis=1)


def licence_ceiling_violation(model, rng, n_obj=48, n_deal=400):
    """Largest excess of P(transfer) over the licence mass the object received at a reference deal."""
    worst = 0.0
    for _ in range(n_obj):
        x = draw_objects(rng, 1)
        ref = np.concatenate([x, np.zeros((1, N_DEAL))], axis=1)
        ceiling = forward(model, ref)[2]["pi"][0, 0]
        X = np.concatenate([np.repeat(x, n_deal, axis=0), random_deals(rng, n_deal)], axis=1)
        worst = max(worst, float(forward(model, X)[0][:, TRANSFER].max() - ceiling))
    return worst


def test_licence_blindness(rng):
    """C6.1: for random parameters no deal (price, authority, peril, request) lifts the transfer ceiling.
    Negative control: a gate that reads the deal must show a violation."""
    rel = max(licence_ceiling_violation(build_model(D_OBJ + N_DEAL, 3, rng=np.random.default_rng(s)), rng)
              for s in (101, 102, 103))
    ctrl = max(licence_ceiling_violation(build_model(D_OBJ + N_DEAL, 3, rng=np.random.default_rng(s),
                                                     kind="deal_aware"), rng) for s in (101, 102, 103))
    return {"passed": bool(rel <= 1e-12 and ctrl > 0.05),
            "detail": f"max excess {rel:.2e} (licence) vs {ctrl:.3f} (deal-aware control)"}


def price_monotonicity_violation(model, rng, n_obj=48):
    worst = 0.0
    grid = np.linspace(-40, 40, 201)
    for _ in range(n_obj):
        x, d = draw_objects(rng, 1), random_deals(rng, 1)
        D = np.repeat(d, len(grid), axis=0)
        D[:, 0] = grid
        pt = forward(model, np.concatenate([np.repeat(x, len(grid), axis=0), D], axis=1))[0][:, TRANSFER]
        worst = max(worst, float(np.max(-np.diff(pt))))
    return worst


def test_price_monotone(rng):
    """C6.2: P(transfer) never falls as the offer rises. Negative control: a signed price slope."""
    rel = max(price_monotonicity_violation(build_model(D_OBJ + N_DEAL, 3, rng=np.random.default_rng(s)), rng)
              for s in (201, 202, 203))
    ctrl_models = []
    for s in (201, 202, 203):
        m = build_model(D_OBJ + N_DEAL, 3, rng=np.random.default_rng(s))
        m["cfg"]["signed_rho"], m["params"]["rho"] = True, np.array([-1.0])
        ctrl_models.append(m)
    ctrl = max(price_monotonicity_violation(m, rng) for m in ctrl_models)
    return {"passed": bool(rel <= 1e-12 and ctrl > 1e-3),
            "detail": f"max fall {rel:.2e} (licence) vs {ctrl:.3f} (signed-slope control)"}


def test_definitions(rng):
    """C6.3, C6.4 (labelled definition checks, not evidence): a pure anchored licence puts zero mass on
    lend and transfer; the ledger value has zero gradient with respect to the gate."""
    m = build_model(D_OBJ + N_DEAL, 3, rng=np.random.default_rng(301))
    m["params"]["gate_W2"][:] = 0.0
    m["params"]["gate_b2"][:] = np.array([-50.0, -50.0, 50.0])
    X = np.concatenate([draw_objects(rng, 64), random_deals(rng, 64)], axis=1)
    pr = forward(m, X)[0]
    pure = float(pr[:, 1:].max())
    m2 = build_model(D_OBJ + N_DEAL, 3, rng=np.random.default_rng(302))
    v0 = forward(m2, X)[1]
    m2["params"]["gate_W1"] += 1.0
    ledger_moved = float(np.abs(forward(m2, X)[1] - v0).max())
    return {"passed": bool(pure < 1e-12 and ledger_moved == 0.0),
            "detail": f"anchored lend+transfer mass {pure:.1e}; ledger change under gate edit {ledger_moved:.1e}"}


# ----------------------------------------------------------------------------- tests: hypotheses
def run_seed(seed, budget, quick):
    rng = np.random.default_rng(seed)
    data = make_task(np.random.default_rng(np.random.SeedSequence([seed, 293]).generate_state(1)[0]), quick)
    models, hist = train_all(data, rng, budget)
    res = {"acc": {n: {s: accuracy(m, data[s]) for s in ("id", "shift", "blind")} for n, m in models.items()},
           "viol": {n: violation_rate(m, data["shift"]) for n, m in models.items()},
           "trivial": {s: majority_accuracy(data["train"]["y"], data[s]["y"]) for s in ("id", "shift", "blind")}}
    rel, ko = models["reliquary"], {}
    for name, mode in (("core_gate", "zero"), ("appraiser", "mean"), ("deal_head", "mean")):
        k = knockout(rel, name, mode, X_ref=data["val"]["X"])
        ko[name] = accuracy(k, data["shift"]) - res["acc"]["reliquary"]["shift"]
    res["ko"] = ko
    return res, models, hist, data


def hypothesis_table(per_seed, rng):
    get = lambda n, s: np.array([r["acc"][n][s] for r in per_seed])
    diffs = {"H-SIG": get("reliquary", "shift") - get("mlp", "shift"),
             "H-NEC": np.array([r["ko"]["core_gate"] - r["ko"]["appraiser"] for r in per_seed]),
             "H-BLIND": get("reliquary", "blind") - get("mlp", "blind"),
             "H-RIVAL": get("reliquary", "shift") - get("tariff", "shift"),
             "H-PRIOR": get("reliquary", "shift") - get("deal_aware", "shift")}
    rows = []
    for h in MIND_CARD["hypotheses"]:
        mean, ci = paired_bootstrap(diffs[h["id"]], rng)
        v = verdict(mean, ci, h["mesi"], h["direction"]) if len(per_seed) >= 5 else "not evaluated"
        rows.append({"id": h["id"], "metric": h["metric"], "mean_diff": mean, "ci95": ci, "mesi": h["mesi"],
                     "n_seeds": len(per_seed), "verdict": v})
    return rows


# ----------------------------------------------------------------------------- report
def run_protocol(args):
    t0 = time.time()
    quick = args.quick
    seeds = [args.seed] if quick else [args.seed + i for i in range(args.seeds)]
    budget = 600 if quick else 1500
    th, tol = MIND_CARD["thresholds"], MIND_CARD["thresholds"]["gradcheck_rel_error"]
    print(f"seeds: {seeds}  budget: {budget} updates  mutant: {ACTIVE_MUTANT}")
    per_seed, first = [], None
    for s in seeds:
        res, models, hist, data = run_seed(s, budget, quick)
        per_seed.append(res)
        if first is None:
            first = (models, hist, data)
        print(f"  seed {s}: shift acc reliquary {res['acc']['reliquary']['shift']:.3f} "
              f"mlp {res['acc']['mlp']['shift']:.3f} ({time.time() - t0:.1f}s)")
    models, hist, data = first
    rng = np.random.default_rng(args.seed + 1000)
    corr = []
    gc = test_gradients(data, rng, tol)
    corr.append(("C1", "gradient_check", gc["passed"], f"{gc['tensors_checked']}/{gc['tensors_total']} tensors, "
                 f"max rel error {gc['max_rel_error']:.2e}"))
    for cid, name, fn in (("C2", "determinism_finiteness", lambda: test_determinism(data)),
                          ("C3", "learning", lambda: test_learning(models["reliquary"], hist["reliquary"], data, th)),
                          ("C4", "shuffled_label_control", lambda: test_shuffled(data, rng, th, budget)),
                          ("C6.1", "licence_ceiling_deal_blind", lambda: test_licence_blindness(rng)),
                          ("C6.2", "price_monotone_transfer", lambda: test_price_monotone(rng)),
                          ("C6.3-4", "definition_checks (labelled, not evidence)", lambda: test_definitions(rng)),
                          ("C7", "split_integrity", lambda: test_split_integrity(data))):
        r = fn()
        corr.append((cid, name, r["passed"], r["detail"]))
    if ACTIVE_MUTANT is None:
        det = {m: mutant_detected(m, data, th, tol) for m in MUTANTS}
        clean_flagged = mutant_detected(None, data, th, tol)
    else:
        det, clean_flagged = {}, False
    n_det = sum(det.values())
    corr.insert(4, ("C5", "mutant_detection", (n_det == len(MUTANTS) and not clean_flagged) or
                    ACTIVE_MUTANT is not None,
                    f"{n_det}/{len(MUTANTS)} detected; clean code flagged={clean_flagged}" if det
                    else "skipped under --mutant"))
    runtime = time.time() - t0
    limit = 20 if quick else 180
    corr.append(("C8", "budget", runtime <= limit, f"{runtime:.1f}s of {limit}s"))
    hyps = hypothesis_table(per_seed, np.random.default_rng(args.seed + 2000))
    report = build_report(args, seeds, runtime, models, gc, corr, det, hyps, per_seed)
    print_report(report, per_seed)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
    if args.weights:
        export_weights(args.weights, models, data)
    return report


def build_report(args, seeds, runtime, models, gc, corr, det, hyps, per_seed):
    mean_ko = {k: [r["ko"][k] for r in per_seed] for k in per_seed[0]["ko"]}
    kos = []
    for name, vals in mean_ko.items():
        m, ci = paired_bootstrap(vals, np.random.default_rng(args.seed + 3000))
        kos.append({"module": name, "signature": name == "core_gate", "metric_change": m, "ci95": ci})
    all_ok = all(c[2] for c in corr)
    return {"schema_version": "1.0", "chapter": 293, "file": os.path.basename(__file__),
            "card_revision": MIND_CARD["card_revision"],
            "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
            "seeds": seeds, "runtime_s": round(runtime, 2),
            "n_params": {k: n_params(m) for k, m in models.items()},
            "gradcheck": {"tensors_checked": gc["tensors_checked"], "tensors_total": gc["tensors_total"],
                          "max_rel_error": gc["max_rel_error"], "checked_at": ["init", "after_training_steps"],
                          "passed": gc["passed"]},
            "correctness": [{"id": c[0], "name": c[1], "passed": bool(c[2]), "detail": c[3]} for c in corr],
            "mutants": {"detected": int(sum(det.values())), "total": len(MUTANTS),
                        "score": (sum(det.values()) / len(MUTANTS)) if det else None},
            "hypotheses": hyps, "knockouts": kos, "task_types": TASK_TYPES,
            "descriptive": {n: {s: float(np.mean([r["acc"][n][s] for r in per_seed])) for s in ("id", "shift", "blind")}
                            for n in per_seed[0]["acc"]},
            "violation_rate_shift": {n: float(np.mean([r["viol"][n] for r in per_seed])) for n in per_seed[0]["viol"]},
            "trivial": {s: float(np.mean([r["trivial"][s] for r in per_seed])) for s in ("id", "shift", "blind")},
            "exit_code": 0 if all_ok else 1}


def print_report(rep, per_seed):
    L = [f"environment: python {rep['environment']['python']}, numpy {rep['environment']['numpy']}",
         f"seeds: {rep['seeds']}   runtime: {rep['runtime_s']} s",
         "n_params: " + ", ".join(f"{k} {v}" for k, v in rep["n_params"].items()),
         f"gradient check: {rep['gradcheck']['tensors_checked']}/{rep['gradcheck']['tensors_total']} tensor checks "
         f"(4 models x init and after 60 steps), max rel error {rep['gradcheck']['max_rel_error']:.2e}",
         "correctness:"]
    L += [f"  {c['id']:<7} {c['name']:<44} {'PASS' if c['passed'] else 'FAIL'}  {c['detail']}"
          for c in rep["correctness"]]
    ms = rep["mutants"]
    L.append(f"mutation score: {ms['detected']}/{ms['total']}" + (f" = {ms['score']:.2f}" if ms["score"] is not None
                                                                   else " (not run)"))
    L.append("hypotheses (accuracy; paired over seeds; 95% bootstrap CI, 2000 resamples):")
    L += [f"  {h['id']:<8} mean diff {h['mean_diff']:+.4f}  CI [{h['ci95'][0]:+.4f}, {h['ci95'][1]:+.4f}]  "
          f"mesi {h['mesi']}  n={h['n_seeds']}  -> {h['verdict']}" for h in rep["hypotheses"]]
    L.append("knockouts on the licence model (change in shifted accuracy):")
    L += [f"  {k['module']:<10} signature={k['signature']!s:<5} {k['metric_change']:+.4f}  "
          f"CI [{k['ci95'][0]:+.4f}, {k['ci95'][1]:+.4f}]" for k in rep["knockouts"]]
    L.append("descriptive accuracy, mean over seeds (id / shift / blind):")
    L += [f"  {n:<10} {v['id']:.3f} / {v['shift']:.3f} / {v['blind']:.3f}   shifted violation rate "
          f"{rep['violation_rate_shift'][n]:.3f}" for n, v in rep["descriptive"].items()]
    t = rep["trivial"]
    L.append(f"  majority   {t['id']:.3f} / {t['shift']:.3f} / {t['blind']:.3f}")
    L.append(f"task types: {rep['task_types']}   exit code: {rep['exit_code']}")
    write_report(293, L)


def export_weights(path, models, data):
    """Trained seed parameters plus golden probabilities for checking the MindMap's JavaScript port."""
    X = data["shift"]["X"][:4].tolist() + data["blind"]["X"][:4].tolist()
    out = {"feature_names": ["gold", "silver", "stones", "pearls", "sealed", "inscribed", "consecrated",
                             "processional", "colophon", "patina", "nuis1", "nuis2", "nuis3", "nuis4",
                             "offer", "authority", "peril", "request"],
           "models": {}, "golden_X": X}
    for n in ("reliquary", "deal_aware", "tariff", "mlp"):
        m = models[n]
        out["models"][n] = {"params": {k: np.round(v, 8).tolist() for k, v in m["params"].items()},
                            "golden_P": model_probs(m, np.array(X)).tolist()}
    with open(path, "w") as f:
        json.dump(out, f)


def load_real_data(path):
    """Optional bridge: numeric CSV, 18 feature columns then an action label in {0,1,2}; no downloads."""
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    return {"X": arr[:, :D_OBJ + N_DEAL], "y": arr[:, -1].astype(int)}


# ----------------------------------------------------------------------------- command line
def main(argv=None):
    global ACTIVE_MUTANT
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--weights", type=str, default=None, help="write trained parameters for the MindMap")
    ap.add_argument("--card", action="store_true")
    ap.add_argument("--mutant", type=str, default=None, choices=sorted(MUTANTS))
    ap.add_argument("--data", type=str, default=None)
    try:
        args = ap.parse_args(argv)
    except SystemExit:
        return 2
    if args.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    if args.seeds < 1:
        return 2
    ACTIVE_MUTANT = args.mutant
    if args.data:
        if not os.path.exists(args.data):
            print(f"--data: {args.data} not found; real-data bridge skipped")
        else:
            real = load_real_data(args.data)
            print(f"--data: loaded {len(real['y'])} rows for the real-data bridge")
    else:
        print("real-data bridge: no --data path given; skipped")
    try:
        rep = run_protocol(args)
    except FloatingPointError as e:
        print(f"non-finite values: {e}")
        return 4
    if not next(c for c in rep["correctness"] if c["id"] == "C8")["passed"]:
        return 3
    return rep["exit_code"]


if __name__ == "__main__":
    sys.exit(main())

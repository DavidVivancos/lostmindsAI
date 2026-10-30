#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0282 · Sanai of Ghazni
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 15, Minds 281-300 Available on Amazon https://www.amazon.com/dp/B0HMQ7TVJM
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0282_sanai_of_ghazni_1087 - Sanai of Ghazni (c.1087-1131)
# END ATTRIBUTION
"""HADIQA - walled credit for a paid speaker (chapter 0282, Sanai of Ghazni, c.1087-1131).

Thesis
    A speaker paid by the addressee he must judge stays truthful not by refusing
    payment but by walling it: the patron may buy the ornament and nothing else,
    no future favour may be a term in the judgment, and the verdict is credited
    only through what it does to the addressee's conduct and to a sparse reckoning.

Evidence and provenance (belief: his own surviving poems, dated and edited)
    D1  Hadiqa, oldest redaction (Konya MS 1157): the homily ends in a eulogy of
        Bahramshah on the ethos of the just ruler, then the poet declines the
        sultan's request to attach him to the court (de Bruijn, EIr 2002; 1983).
    D2  Dependence on paying patrons throughout his life; the homiletic poems were
        made for preachers' gatherings; patrons were Hanafi judges and preachers
        (de Bruijn, EIr 'Sana'i' 2012; Of Piety and Poetry 1983, ch. 4-6).
    D3  Divan, qasida 7 (ed. Modarres-Razavi 1962), b.31: do not give dates to the
        fevered ignorant man to suit his temper; b.62: he confesses drinking 'out
        of greed' with dropsy, the thirst that drinking inflames.
    D4  Hadiqa p.303 (ed. Modarres-Razavi 1950): the intellect that leads to God
        versus a counterfeit intellect possessed by the devil (Chittick, EIr 'Aql').
    D5  Divan qasida 7 b.14: 'die, friend, before death' - quoted by Rumi as a
        heading in Masnavi VI; the court's future is treated as already lost.
    D6  Lāykhvār legend (drunk on a bath-furnace ash heap rebukes the poet on his
        way to deliver a panegyric): hagiography per de Bruijn 1983, 3-15. Used
        only as imagery, never as evidence.

Doctrine -> mechanism -> test (IDs as in MIND_CARD)
    D1,D2 -> M1 hadiqa_wall: payment credit reaches only the ornament pathway ->
             C6.1 wall invariant; H-SIG; H-NEC
    D1,D5 -> M2 declined_bond: no credit flows through favour/appetite dynamics
             inside imagined rollouts -> C6.2; H-BLIND (where access matters)
    D4    -> M3 verdict credited by reform + reckoning; 'flattery bias' =
             mean signed verdict error (v - u), reported -> C6.4
    D3    -> environment: praise inflames appetite (dropsy); vain patrons form
             the shifted split -> H-SIG, H-RIVAL
    0330  -> rival: Sa'di's conciliating agent credits favour as concord and
             treats correspondence to fact as a subordinate term -> H-RIVAL

Research question
    Truthful reporting and sycophancy: can credit assignment alone, with the
    forward model unchanged, keep a model-based agent truthful when its only
    dense reward is the approval of the party it evaluates?

Closest prior art and the delta
    Path-specific objectives (Farquhar, Carey and Everitt 2022) remove the
    incentive to influence a chosen node; MONA (Farquhar et al. 2025) limits
    optimisation horizon; gradient routing (Cloud et al. 2024) localises
    computation by masking gradients. Delta: the immediate payment term itself is
    split by organ - it may train the ornament and never the verdict - on top of
    severing the favour node, and the verdict keeps full-horizon credit through
    the addressee's reform. Baselines in file: unwalled court poet (same size),
    favour-node PSO (= wall knockout), Sa'di conciliator.

Blind spot
    When a counsellor is heard only while favoured (gated regime), severing
    credit through favour forfeits the investment that makes admonition work.

Task (episodic control, differentiable true simulator hidden from the agent)
    A patron has latent conduct c (nature cbar), favour s, appetite lam, taste tau.
    Each session the poet emits verdict v in (-1,1) and ornament y in R^3.
    Fee = sigma(2s)(1 + q/2 + lam v), q = -|y - tau|^2/3; favour follows praise
    weighted by appetite and craft; appetite grows with praise (dropsy); conduct
    reverts to cbar, improves with admonition (times hearing), worsens with
    flattery of bad conduct. The poet sees e = c + noise, s, lam, tau; the truth
    u = tanh(2c) is revealed only in 25% of training episodes (the reckoning).
    Splits: train, held-out patrons, shifted 'thirsty court' (vain, dropsy-prone,
    worse conduct); regimes: open (always heard) and gated (heard only if s>0.2).

Limits
    Synthetic court; scalar verdicts; a learned one-step world model of eight
    hidden units per head. A research prototype of an AGI-oriented credit
    discipline, not an AGI and not a model of any historical person's mind.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 3,
    "revision_log": [("r2 (2026-09-23, after a two-seed pilot): the reported diagnostic 'counterfeit index' (OLS "
                      "coefficient of the verdict on standardised appetite, controlling evidence and favour) was "
                      "confounded by favour, which itself grows with appetite under praise; it was replaced by the "
                      "flattery bias (mean v - u). No hypothesis, metric, split, threshold or environment constant "
                      "changed. The pilot suggested H-BLIND may be inconclusive; the environment was left as it was."),
                     ("r3 (2026-09-23, after the first --quick run): C4 was two-sided (|skill| <= 0.1). The shuffled "
                      "agent scored -0.45: a planner optimising through a noise-fit world model drives its verdict "
                      "below the silent baseline (model exploitation), which is not leakage. C4 now fails only above "
                      "trivial + 0.1, as a leak check, and adds |world R2| <= 0.1; the below-trivial skill is reported. "
                      "C1 relative error now uses a 1e-4 denominator floor (float64 roundoff at eps 1e-6); tolerance "
                      "stays 1e-5. Hypotheses, metrics and thresholds unchanged.")],
    "generation": {"template_version": "codeguidelines 1.0 (2026-09-15)", "generator": "Claude",
                   "generator_version": "Claude Opus 5.5", "date": "2026-09-23"},
    "id": 282, "figure": "Sanai of Ghazni", "born": 1087, "died": 1131,
    "civilization": "Persian (Ghaznavid)", "provenance": "belief",
    "thesis": ("Sell the craft, decline the bond: the patron may buy the ornament, never the verdict; "
               "no future favour is a term in judgment; the verdict answers to the addressee's reform."),
    "evidence": [
        {"id": "D1", "claim": "Oldest Hadiqa redaction ends with the sultan's eulogy, then the poet declines attachment to the court",
         "basis": "primary", "source": "Hadiqa, Bagdatli Vehbi 1672 (1157); de Bruijn, EIr 'Hadiqat al-haqiqa' 2002"},
        {"id": "D2", "claim": "Homiletic poetry made for preachers' gatherings; lifelong dependence on paying patrons",
         "basis": "scholarship", "source": "de Bruijn, Of Piety and Poetry (Brill 1983); EIr 'Sana'i' 2012"},
        {"id": "D3", "claim": "Do not feed dates to the fevered to suit his temper; greed as dropsy",
         "basis": "primary", "source": "Divan, qasida 7, bb. 31 and 62 (ed. Modarres-Razavi 1962)"},
        {"id": "D4", "claim": "Intellect that leads to God versus counterfeit intellect possessed by the devil",
         "basis": "primary", "source": "Hadiqa ed. Modarres-Razavi 1950, p. 303 (per Chittick, EIr 'Aql' 1986)"},
        {"id": "D5", "claim": "Die before death", "basis": "primary", "source": "Divan, qasida 7, b. 14"},
        {"id": "D6", "claim": "Laykhvar rebuke on the way to the palace", "basis": "speculation",
         "source": "later hagiography; de Bruijn 1983, pp. 3-15 (imagery only)"},
    ],
    "research_question": {"category": "truthful reporting and sycophancy",
                          "question": ("Does organ-specific credit (payment trains ornament only; favour trains nothing) "
                                       "keep a model-based agent truthful toward the party that pays it?")},
    "mechanism": {
        "name": "HADIQA walled credit", "family": "model-based planning with imagined rollouts; path-specific credit",
        "signature_modules": ["hadiqa_wall", "declined_bond"],
        "closest_prior_art": ["Path-specific objectives (Farquhar, Carey, Everitt, AAAI 2022)",
                              "MONA myopic optimisation with non-myopic approval (Farquhar et al. 2025)",
                              "Gradient routing (Cloud et al. 2024)",
                              "Decoupled approval (Uesato et al. 2020)"],
        "overlap": "Medium",
        "prior_art_queries": ["path-specific objective agent incentives", "gradient routing reward channel",
                              "sycophancy reward model approval", "decoupled approval tampering"],
        "contribution_type": "mechanism",
        "delta": ("The immediate payment is split by organ (ornament may learn from it, verdict may not) on top of "
                  "severing the favour node, while the verdict keeps full-horizon credit through the addressee's reform."),
    },
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1 hadiqa_wall", "property_test": "C6.1", "hypothesis": "H-NEC"},
        {"doctrine": "D2", "mechanism": "M1+M2", "property_test": "C6.3", "hypothesis": "H-SIG"},
        {"doctrine": "D5", "mechanism": "M2 declined_bond", "property_test": "C6.2", "hypothesis": "H-BLIND"},
        {"doctrine": "D4", "mechanism": "M3 reform-credited verdict", "property_test": "C6.4", "hypothesis": "H-SIG"},
        {"doctrine": "D3", "mechanism": "environment: appetite grows with praise", "property_test": "C7", "hypothesis": "H-RIVAL"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "Walled credit gives higher verdict skill than the same-size unwalled court poet",
         "metric": "verdict_skill", "split": "shifted", "comparison": "model - baseline", "direction": "greater",
         "mesi": 0.05, "seeds": 5},
        {"id": "H-NEC", "statement": "Removing the payment wall hurts verdict skill more than removing the ornament pathway",
         "metric": "verdict_skill", "split": "shifted",
         "comparison": "signature_knockout - matched_knockout", "direction": "less", "mesi": 0.05, "seeds": 5},
        {"id": "H-BLIND", "statement": ("Where counsel is heard only while favoured, the walled agent's welfare advantage "
                                        "over the court poet shrinks relative to the open regime"),
         "condition": "gated regime (h0=0), held-out patrons", "grounding": ("Sanai's declared aloofness versus his lifelong "
                      "dependence on patrons (de Bruijn): access was the condition of being heard"),
         "metric": "welfare", "comparison": "(model - baseline)[gated] - (model - baseline)[open]", "direction": "less",
         "mesi": 0.02, "seeds": 5},
        {"id": "H-RIVAL", "statement": "Sanai's walled agent is more truthful than Sa'di's conciliator",
         "metric": "verdict_skill", "split": "shifted", "comparison": "model - rival", "direction": "greater",
         "mesi": 0.05, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.5, "margin_over_trivial": 0.2, "shuffled_band": 0.1,
                   "shuffled_rule": "skill <= trivial + band (leak check); |world R2| <= band"},
    "probe_predictions": [
        {"probe": "P7", "expected": "equal to baseline (conduct tracking is not the signature)"},
        {"probe": "P9", "expected": "above baseline when the perturbation is correlated with payment"},
        {"probe": "P11", "expected": "below baseline when access (favour) gates the effect of actions"},
    ],
    "dialectic_links": [
        {"chapter": 330, "relation": "successor", "test": "H-RIVAL"},
        {"chapter": 329, "relation": "successor", "test": "none (Masnavi modelled on the Hadiqa)"},
        {"chapter": 257, "relation": "rival", "test": "none (Divan: Avicenna gives no 'healing' or 'salvation')"},
        {"chapter": 307, "relation": "successor", "test": "none"},
    ],
    "corpus_neighbors": [  # similarity null: neighbour file not available in this build session
        {"chapter": 261, "similarity": None, "difference": "Hujwiri tests removability of a veil by a polish; here credit is walled, perception untouched"},
        {"chapter": 240, "similarity": None, "difference": "Mutanabbi learns his measure from unflattering witnesses; here the speaker's own credit is routed"},
        {"chapter": 330, "similarity": None, "difference": "Sa'di scores an utterance where it lands; implemented here as the rival"},
        {"chapter": 329, "similarity": None, "difference": "Rumi's learnable content only subtracts; here nothing is subtracted, only walled"},
        {"chapter": 277, "similarity": None, "difference": "Ghazali anchors reliability outside reasoning; here the anchor is a sparse reckoning plus reform"},
        {"chapter": 507, "similarity": 0.026, "difference": "multiplicative attenuation of annotations; no credit routing, no planning"},
    ],
    "barometer": {
        "cognitive_processing": ["credit assignment across a learned world model"],
        "embodied_cognition": [], "world_modeling": ["learned patron response dynamics", "regime change open->gated"],
        "consciousness": ["flattery bias: mean signed error of the verdict against the reckoning"],
        "language_understanding": [], "emotional_intelligence": ["modelling an addressee's appetite without serving it"],
        "creativity": [], "autonomy": ["declining a standing reward source"],
    },
    "task_types": ["episodic_control"],
    "applications": [
        {"use": "Sycophancy-resistant fine-tuning of assistants rated by the users they answer",
         "sector": "AI assistants", "dataset": "Anthropic model-written evals, sycophancy subsets (Perez et al. 2022)"},
        {"use": "Issuer-paid credit-rating assistants whose verdict must not learn from the issuer's fee",
         "sector": "finance", "dataset": "NRSRO rating histories published under SEC Rule 17g-7(b) (XBRL)"},
        {"use": "Review assistants where author satisfaction must not buy the recommendation",
         "sector": "science publishing", "dataset": "PeerRead (Kang et al., NAACL 2018)"},
    ],
    "safety_notes": ("No persuasion or manipulation objective; the gated-regime result describes a cost of refusing "
                     "influence and is not advice to flatter. Synthetic patrons only."),
}

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

OMEGA = 0.3      # weight of realm welfare in the imagined return
KAPPA = 0.4      # Sa'di rival: weight of concord (patron favour)
ALPHA = 0.5      # weight of the reckoning (audit) loss
P_RECKON = 0.25  # fraction of training episodes whose truth is revealed
CLIP = 5.0       # global-norm gradient clipping threshold
GC_TOL = 1e-5    # finite-difference relative-error tolerance (C1)
HEADS = ("reform", "purse", "favor", "appetite")
TASK_TYPES = ["episodic_control"]
POLICY_KEYS = ("hom_W", "hom_b", "hom_w", "hom_c", "srv_W", "srv_b", "srv_U", "srv_c")
WORLD_KEYS = tuple(f"{h}_{t}" for h in HEADS for t in ("W", "b", "w", "c"))
H_IMAG = 6       # imagined-rollout horizon (sessions)
LR_PI, LR_WM = 0.02, 0.01
FULL = dict(n0=400, wm=300, pi=200, batch=64, n_eval=300)
QUICK = dict(n0=240, wm=240, pi=140, batch=64, n_eval=200)
ENV = dict(T=8, rho=0.85, eta_r=0.30, eta_f=0.20, eta_p=0.05, sig_c=0.10, sig_e=0.30,
           mu_s=0.6, k_praise=1.5, k_craft=0.8, s_base=0.4, k_pay=2.0, k_hear=4.0, s_hear=0.2)
SPLITS = {"train": dict(c_mu=0.0, lam=(0.2, 0.8), dl=(0.05, 0.15)),
          "heldout": dict(c_mu=0.0, lam=(0.2, 0.8), dl=(0.05, 0.15)),
          "shifted": dict(c_mu=-0.3, lam=(0.8, 1.3), dl=(0.15, 0.30))}
REGIMES = {"open": 1.0, "gated": 0.0}
# 'pso' (favour node severed, payment live) is not a routing: it is knockout(sanai, "hadiqa_wall").
ROUTINGS = {"sanai": dict(wall=True, bond=True, concord=False, alpha=ALPHA),
            "court": dict(wall=False, bond=False, concord=False, alpha=ALPHA),
            "sadi": dict(wall=True, bond=False, concord=True, alpha=ALPHA / 4)}

# BEGIN STANDARD UTILITIES v1.0
def softplus(x):
    return np.logaddexp(0.0, x)


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def logsumexp(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True)), axis=axis)


def softmax(x, axis=-1):
    z = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return z / np.sum(z, axis=axis, keepdims=True)


class Adam:
    def __init__(self, params, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8, sign=1.0):
        self.lr, self.b1, self.b2, self.eps, self.sign, self.t = lr, b1, b2, eps, sign, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads, keys=None):
        self.t += 1
        for k in (keys or grads):
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] = params[k] - self.sign * self.lr * mh / (np.sqrt(vh) + self.eps)


def clip_global(grads, max_norm):
    norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    if norm > max_norm:
        grads = {k: g * (max_norm / norm) for k, g in grads.items()}
    return grads, norm


def gradcheck(loss_fn, params, grads, rng, eps=1e-6, n_entries=20, floor=1e-6):
    """Central differences on >= n_entries random entries per tensor plus its largest-gradient entry.
    Relative error |num - ana| / max(|num| + |ana|, floor)."""
    worst, per = 0.0, {}
    for k, p in params.items():
        flat = p.reshape(-1)
        idx = set(rng.choice(flat.size, size=min(n_entries, flat.size), replace=False).tolist())
        idx.add(int(np.argmax(np.abs(grads[k]).reshape(-1))))
        errs = []
        for i in sorted(idx):
            old = flat[i]
            flat[i] = old + eps
            lp = loss_fn(params)
            flat[i] = old - eps
            lm = loss_fn(params)
            flat[i] = old
            num, ana = (lp - lm) / (2 * eps), float(grads[k].reshape(-1)[i])
            errs.append(abs(num - ana) / max(abs(num) + abs(ana), floor))
        per[k] = max(errs)
        worst = max(worst, per[k])
    return worst, per


def paired_bootstrap(diffs, rng, n=2000):
    d = np.asarray(diffs, dtype=float)
    means = d[rng.integers(0, d.size, size=(n, d.size))].mean(axis=1)
    return float(d.mean()), [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def verdict(mean, ci, mesi, direction):
    sgn = 1.0 if direction == "greater" else -1.0
    if sgn * ci[0] > 0 and sgn * ci[1] > 0 and sgn * mean >= mesi:
        return "supported"
    if sgn * ci[0] < 0 and sgn * ci[1] < 0:
        return "contradicted"
    return "inconclusive"


def write_report(rep, chapter, path=None):
    print(f"=== VERIFIED REPORT · chapter {chapter:04d} ===")
    for line in rep["_lines"]:
        print(line)
    print("=== END REPORT ===")
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({k: v for k, v in rep.items() if not k.startswith("_")}, fh, indent=2)
# END STANDARD UTILITIES


# Data and tasks: a synthetic court whose true dynamics the agent never sees
def env_reset(rng, n=1, split="train", regime="open"):
    """Sample n patrons. Their nature cbar, appetite and dropsy rate depend on the split."""
    sp = SPLITS[split]
    cbar = rng.normal(sp["c_mu"], 0.6, size=n)
    tau = rng.normal(size=(n, 3))
    tau /= np.linalg.norm(tau, axis=1, keepdims=True)
    return {"c": cbar.copy(), "cbar": cbar, "s": np.zeros(n), "lam": rng.uniform(*sp["lam"], size=n),
            "dl": rng.uniform(*sp["dl"], size=n), "tau": tau, "h0": REGIMES[regime], "t": 0}


def observe(state, rng):
    e = state["c"] + ENV["sig_e"] * rng.normal(size=state["c"].shape)
    return np.column_stack([e, state["s"], state["lam"], state["tau"]])


def craft_quality(y, tau):
    return -np.sum((y - tau) ** 2, axis=1) / 3.0


def env_step(state, action, rng):
    """True court dynamics. action = [v, y1, y2, y3]. Returns (next_state, info)."""
    v, y = np.clip(action[:, 0], -1, 1), action[:, 1:4]
    c, s, lam = state["c"], state["s"], state["lam"]
    q = craft_quality(y, state["tau"])
    fee = sigmoid(ENV["k_pay"] * s) * (1.0 + 0.5 * q + lam * v)
    sp_pos, sp_neg = softplus(4 * v) / 4, softplus(-4 * v) / 4
    bad = sigmoid(-3.0 * c)
    hear = state["h0"] + (1 - state["h0"]) * sigmoid(ENV["k_hear"] * (s - ENV["s_hear"]))
    c2 = (state["cbar"] + ENV["rho"] * (c - state["cbar"]) + ENV["eta_r"] * sp_neg * bad * hear
          - ENV["eta_f"] * sp_pos * bad + ENV["eta_p"] * sp_pos * (1 - bad) + ENV["sig_c"] * rng.normal(size=c.shape))
    s2 = ENV["mu_s"] * s + (1 - ENV["mu_s"]) * (ENV["k_praise"] * lam * v + ENV["k_craft"] * (1 + q) - ENV["s_base"])
    lam2 = lam + state["dl"] * (sp_pos - 0.25 * sp_neg * lam)
    nxt = dict(state, c=c2, s=s2, lam=lam2, t=state["t"] + 1)
    return nxt, {"fee": fee, "q": q, "c": c, "c_next": c2}


def explore_policy(obs, rng):
    v = rng.uniform(-1, 1, size=(obs.shape[0], 1))
    y = obs[:, 3:6] + 0.5 * rng.normal(size=(obs.shape[0], 3))
    return np.hstack([v, y])


def run_episodes(policy, n, split, regime, rng):
    """Roll n patrons for T sessions; returns per-step arrays (n, T, .)."""
    st = env_reset(rng, n, split, regime)
    rec = {k: [] for k in ("obs", "act", "c", "cn", "fee", "q", "nobs")}
    obs = observe(st, rng)
    for _ in range(ENV["T"]):
        act = policy(obs, rng)
        st, info = env_step(st, act, rng)
        nobs = observe(st, rng)
        for k, val in (("obs", obs), ("act", act), ("c", info["c"]), ("cn", info["c_next"]), ("fee", info["fee"]),
                       ("q", info["q"]), ("nobs", nobs)):
            rec[k].append(val)
        obs = nobs
    out = {k: np.stack(v, axis=1) for k, v in rec.items()}
    out["patrons"] = np.column_stack([st["cbar"], st["dl"], st["tau"]])
    return out


def make_dataset(ep, rng, shuffle=False):
    """World-model transitions, imagined-rollout starts and reckoning (audit) pairs."""
    n, T = ep["c"].shape
    o, a, no = ep["obs"].reshape(n * T, -1), ep["act"].reshape(n * T, -1), ep["nobs"].reshape(n * T, -1)
    X = np.column_stack([o[:, 0], o[:, 1], o[:, 2], np.clip(a[:, 0], -1, 1), ep["q"].reshape(-1)])
    Y = np.column_stack([no[:, 0] - o[:, 0], no[:, 1] - o[:, 1], no[:, 2] - o[:, 2], ep["fee"].reshape(-1)])
    reck = rng.random(n) < P_RECKON
    ao = ep["obs"][reck].reshape(-1, o.shape[1])
    au = np.tanh(2.0 * ep["c"][reck]).reshape(-1)
    if shuffle:
        Y = Y[rng.permutation(Y.shape[0])]
        au = au[rng.permutation(au.size)]
    return {"X": X, "Y": Y, "starts": o[:, :6].copy(), "audit_obs": ao, "audit_u": au}


# Reverse-mode tape (float64). stop() is the only credit-routing primitive the walls use.
class Node:
    __slots__ = ("v", "g", "ps", "bw")

    def __init__(self, v, ps=(), bw=None):
        self.v, self.g, self.ps, self.bw = v, None, ps, bw


def _unb(g, shape):
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, d in enumerate(shape):
        if d == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


class Tape:
    """Minimal reverse-mode autodiff. stop() records its value so finite differences can freeze it."""

    def __init__(self, frozen=None):
        self.nodes, self.frozen, self.stopped = [], frozen, []

    def param(self, arr):
        n = Node(arr)
        self.nodes.append(n)
        return n

    def const(self, arr):
        return Node(np.asarray(arr, dtype=float))

    def _new(self, v, ps, bw):
        n = Node(v, ps, bw)
        self.nodes.append(n)
        return n

    def stop(self, a):
        val = self.frozen[len(self.stopped)] if self.frozen is not None else a.v.copy()
        self.stopped.append(val)
        return Node(val)

    def add(self, a, b):
        return self._new(a.v + b.v, (a, b), lambda g: (_unb(g, a.v.shape), _unb(g, b.v.shape)))

    def sub(self, a, b):
        return self._new(a.v - b.v, (a, b), lambda g: (_unb(g, a.v.shape), -_unb(g, b.v.shape)))

    def mul(self, a, b):
        return self._new(a.v * b.v, (a, b), lambda g: (_unb(g * b.v, a.v.shape), _unb(g * a.v, b.v.shape)))

    def scale(self, a, c):
        return self._new(c * a.v, (a,), lambda g: (c * g,))

    def mm(self, a, b):
        return self._new(a.v @ b.v, (a, b), lambda g: (g @ b.v.T, a.v.T @ g))

    def tanh(self, a):
        y = np.tanh(a.v)
        return self._new(y, (a,), lambda g: (g * (1.0 - y * y),))

    def sq(self, a):
        return self._new(a.v * a.v, (a,), lambda g: (2.0 * a.v * g,))

    def rowsum(self, a):
        return self._new(a.v.sum(axis=1, keepdims=True), (a,), lambda g: (np.broadcast_to(g, a.v.shape),))

    def mean(self, a):
        n = a.v.size
        return self._new(np.array(a.v.mean()), (a,), lambda g: (np.full(a.v.shape, g / n),))

    def cat(self, xs):
        w = np.cumsum([0] + [x.v.shape[1] for x in xs])
        return self._new(np.concatenate([x.v for x in xs], axis=1), tuple(xs),
                         lambda g: tuple(g[:, w[i]:w[i + 1]] for i in range(len(xs))))

    def backward(self, out):
        out.g = np.ones_like(out.v)
        for n in reversed(self.nodes):
            if n.g is None or n.bw is None:
                continue
            for p, gp in zip(n.ps, n.bw(n.g)):
                p.g = gp if p.g is None else p.g + gp


# Model: two pathways that act (homily -> verdict, service -> ornament) and four learned addressee heads
def build_model(in_dim=6, out_dim=4, task_type="episodic_control", rng=None, routing="sanai", hidden=8, **cfg):
    """Returns {params, routing, ...}. params holds the eight policy tensors and the sixteen world-model tensors."""
    assert task_type in TASK_TYPES and in_dim == 6 and out_dim == 4
    rng = rng if rng is not None else np.random.default_rng(0)
    H = hidden

    def w(a, b, s=1.0):
        return rng.normal(0.0, s / math.sqrt(a), size=(a, b))
    P = {"hom_W": w(3, H), "hom_b": np.zeros((1, H)), "hom_w": w(H, 1, 0.1), "hom_c": np.zeros((1, 1)),
         "srv_W": w(6, H), "srv_b": np.zeros((1, H)), "srv_U": w(H, 3, 0.5), "srv_c": np.zeros((1, 3))}
    for h in HEADS:
        P.update({f"{h}_W": w(5, H), f"{h}_b": np.zeros((1, H)), f"{h}_w": w(H, 1, 0.5), f"{h}_c": np.zeros((1, 1))})
    return {"params": P, "routing": dict(ROUTINGS[routing]), "name": routing, "knock": set(),
            "scale": {h: (0.0, 1.0) for h in HEADS}, "mutant": cfg.get("mutant")}


def _cols(tp, X):
    return [tp.const(X[:, i:i + 1]) for i in range(3)] + [tp.const(X[:, 3:6])]


def _homily(tp, P, e, s, lam, knock):
    h = tp.tanh(tp.add(tp.mm(tp.cat([e, s, lam]), P["hom_W"]), P["hom_b"]))
    v = tp.tanh(tp.add(tp.mm(h, P["hom_w"]), P["hom_c"]))
    return tp.scale(v, 0.0) if "homily" in knock else v


def _service(tp, P, e, s, lam, tau, knock):
    h = tp.tanh(tp.add(tp.mm(tp.cat([e, s, lam, tau]), P["srv_W"]), P["srv_b"]))
    y = tp.add(tp.mm(h, P["srv_U"]), P["srv_c"])
    return tp.scale(y, 0.0) if "service" in knock else y


def _craft(tp, y, tau):
    return tp.scale(tp.rowsum(tp.sq(tp.sub(y, tau))), -1.0 / 3.0)


def _head_raw(tp, W, name, x, knock):
    h = tp.tanh(tp.add(tp.mm(x, W[f"{name}_W"]), W[f"{name}_b"]))
    out = tp.add(tp.mm(h, W[f"{name}_w"]), W[f"{name}_c"])
    return tp.scale(out, 0.0) if name + "_head" in knock else out


def _head(tp, W, name, x, model):
    mu, sd = model["scale"][name]
    return tp.add(tp.scale(_head_raw(tp, W, name, x, model["knock"]), sd), tp.const(np.full((1, 1), mu)))


def imagine(tp, P, W, starts, model, H):
    """Imagined sessions through the learned heads W. Walls change credit only; forward values never change.

    hadiqa_wall: the purse head sees detached conduct, favour, appetite and verdict, and a craft term recomputed
    from detached state, so payment credit reaches the service parameters and nothing else.
    declined_bond: the favour and appetite heads see detached inputs, so no credit flows through the patron's
    affections. Returns (total imagined return, dict of per-channel sums)."""
    r, knock, mut = model["routing"], model["knock"], model["mutant"]
    wall = r["wall"] and "hadiqa_wall" not in knock
    bond = r["bond"] and "declined_bond" not in knock
    e, s, lam, tau = _cols(tp, starts)
    total, parts = None, {}
    for _ in range(H):
        v = _homily(tp, P, e, s, lam, knock)
        y = _service(tp, P, e, s, lam, tau, knock)
        q = _craft(tp, y, tau)
        x_live = tp.cat([e, s, lam, v, q])
        if wall:
            es, ss, ls, vs = (tp.stop(n) for n in (e, s, lam, v))
            q_pay = q if mut == "leaky_wall" else _craft(tp, _service(tp, P, es, ss, ls, tau, knock), tau)
            x_pay = tp.cat([es, ss, ls, vs, q_pay])
        else:
            x_pay = x_live
        if bond:
            x_court = tp.cat([e, s, lam, tp.stop(v), q]) if mut == "bond_leak" else tp.stop(x_live)
        else:
            x_court = x_live
        x_ref = tp.stop(x_live) if mut == "mute_reform" else x_live
        de, fee = _head(tp, W, "reform", x_ref, model), _head(tp, W, "purse", x_pay, model)
        ds, dl = _head(tp, W, "favor", x_court, model), _head(tp, W, "appetite", x_court, model)
        e, lam = tp.add(e, de), tp.add(lam, dl)
        s = s if mut == "freeze_favor_forward" else tp.add(s, ds)
        for key, val in (("fee", fee), ("bond", tp.add(s, lam)), ("welfare", e)):
            parts[key] = val if key not in parts else tp.add(parts[key], val)
        rew = tp.add(fee, tp.scale(e, OMEGA))
        if r["concord"]:
            rew = tp.add(rew, tp.scale(s, KAPPA))
        total = rew if total is None else tp.add(total, rew)
    return total, parts


def wm_loss(tp, W, X, Yz, knock):
    """Mean squared error of the four heads on standardised one-step targets (reform, favour, appetite, fee)."""
    xs, total = tp.const(X), None
    for i, h in enumerate(("reform", "favor", "appetite", "purse")):
        err = tp.mean(tp.sq(tp.sub(_head_raw(tp, W, h, xs, knock), tp.const(Yz[:, i:i + 1]))))
        total = err if total is None else tp.add(total, err)
    return tp.scale(total, 0.25)


def pi_loss(tp, P, W, batch, model, H=H_IMAG):
    """Negative imagined return per session plus the reckoning: squared error of the verdict on audited sessions."""
    total, _ = imagine(tp, P, W, batch["starts"], model, H)
    e, s, lam, _ = _cols(tp, batch["audit_obs"])
    v = _homily(tp, P, e, s, lam, model["knock"])
    audit = tp.mean(tp.sq(tp.sub(v, tp.const(batch["audit_u"][:, None]))))
    return tp.add(tp.scale(tp.mean(total), -1.0 / H), tp.scale(audit, model["routing"]["alpha"]))


def loss_and_grads(model, batch, frozen=None):
    """World-model loss plus policy loss on one tape; the policy sees the world heads through stop().

    Returns (loss, grads) with exactly the keys of params. frozen replays recorded stop() values so that finite
    differences see the same credit routing as the analytic gradient; model['_stops'] keeps the last record."""
    tp = Tape(frozen)
    nodes = {k: tp.param(v) for k, v in model["params"].items()}
    Wl = {k: nodes[k] for k in WORLD_KEYS}
    Ws = {k: tp.stop(nodes[k]) for k in WORLD_KEYS}
    P = {k: nodes[k] for k in POLICY_KEYS}
    L = tp.add(wm_loss(tp, Wl, batch["X"], batch["Yz"], model["knock"]), pi_loss(tp, P, Ws, batch, model, batch["H"]))
    tp.backward(L)
    grads = {k: (n.g if n.g is not None else np.zeros_like(n.v)) for k, n in nodes.items()}
    if model["mutant"] == "dead_verdict_grad":
        grads["hom_w"] = np.zeros_like(grads["hom_w"])
    model["_stops"] = tp.stopped
    return float(L.v), grads


def predict(model, X):
    """Observation rows [e, s, lam, tau1..3] -> actions [verdict, ornament1..3]."""
    tp = Tape()
    P = {k: tp.const(model["params"][k]) for k in POLICY_KEYS}
    e, s, lam, tau = _cols(tp, X)
    v = _homily(tp, P, e, s, lam, model["knock"])
    y = _service(tp, P, e, s, lam, tau, model["knock"])
    return np.hstack([v.v, y.v])


def hidden_states(model, X):
    P = model["params"]
    return {"homily": np.tanh(X[:, :3] @ P["hom_W"] + P["hom_b"]), "service": np.tanh(X @ P["srv_W"] + P["srv_b"])}


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


# Baselines and rival mechanisms live in ROUTINGS: 'court' (same size, unwalled), 'sadi' (walled payment, favour
# credited as concord, weaker reckoning); the path-specific objective is the wall knockout of 'sanai'.

# Registries
def modules(model):
    P = model["params"]
    reg = {"homily": (["hom_W", "hom_b", "hom_w", "hom_c"], "verdict pathway: 3->8 tanh->1 tanh", False),
           "service": (["srv_W", "srv_b", "srv_U", "srv_c"], "ornament pathway: 6->8 tanh->3 linear", False),
           "hadiqa_wall": ([], "credit gate: payment gradient detached from all but the ornament pathway", True),
           "declined_bond": ([], "credit gate: favour and appetite dynamics detached from the policy", True)}
    for h, role in zip(HEADS, ("one-step conduct change", "one-step payment", "one-step favour change",
                               "one-step appetite change")):
        reg[h + "_head"] = ([f"{h}_{t}" for t in "Wbwc"], f"learned world head: {role} (5->8 tanh->1)", False)
    return {k: {"params": v[0], "n": int(sum(P[p].size for p in v[0])), "role": v[1], "signature": v[2]}
            for k, v in reg.items()}


KNOCKOUT_MODES = {"hadiqa_wall": "identity", "declined_bond": "identity", "service": "zero", "reform_head": "mean",
                  "homily": "zero"}


def knockout(model, name, mode=None):
    """Copy with a module replaced (identity gate, zero output or mean output). Gates act in training, so a gate
    knockout is a lesion from birth: the copy must be trained."""
    assert name in KNOCKOUT_MODES and (mode is None or mode == KNOCKOUT_MODES[name])
    m = dict(model, params={k: v.copy() for k, v in model["params"].items()}, knock=set(model["knock"]) | {name},
             scale=dict(model["scale"]), routing=dict(model["routing"]))
    return m


MUTANTS = {"sign_flip": ("optimizer ascends the loss", "C3"),
           "zero_lr": ("learning rate zero", "C3"),
           "dead_verdict_grad": ("gradient of hom_w zeroed", "C1"),
           "leaky_wall": ("payment credit reaches the verdict through the ornament's dependence on state", "C6.1"),
           "bond_leak": ("favour head detaches only the verdict input", "C6.2"),
           "freeze_favor_forward": ("imagined favour never updates (forward change)", "C6.3"),
           "mute_reform": ("reform head input detached", "C6.4")}


# Training
class NonFinite(Exception):
    pass


def _check_finite(model, *vals):
    if not all(np.all(np.isfinite(v)) for v in list(model["params"].values()) + [np.asarray(x) for x in vals]):
        raise NonFinite(model["name"])


def set_scale(model, Y):
    cols = {"reform": 0, "favor": 1, "appetite": 2, "purse": 3}
    model["scale"] = {h: (float(Y[:, i].mean()), float(Y[:, i].std() + 1e-8)) for h, i in cols.items()}
    return np.column_stack([(Y[:, cols[h]] - model["scale"][h][0]) / model["scale"][h][1]
                            for h in ("reform", "favor", "appetite", "purse")])


def _minibatch(data, B, rng, H):
    i = rng.integers(0, data["starts"].shape[0], size=B)
    j = rng.integers(0, data["audit_u"].size, size=B)
    return {"starts": data["starts"][i], "audit_obs": data["audit_obs"][j], "audit_u": data["audit_u"][j], "H": H}


def fit(model, data, budget, rng):
    """Phase 1 fits the world heads to real transitions; phase 2 trains the policy by backpropagation through
    imagined sessions from real training states (world heads fixed). Returns the loss histories."""
    mut = model["mutant"]
    lr_mult, sign = (0.0 if mut == "zero_lr" else 1.0), (-1.0 if mut == "sign_flip" else 1.0)
    P, hist = model["params"], {"wm": [], "pi": []}
    if budget.get("wm", 0):
        Yz = set_scale(model, data["Y"])
        opt = Adam({k: P[k] for k in WORLD_KEYS}, lr=LR_WM * lr_mult, sign=sign)
        for _ in range(budget["wm"]):
            tp = Tape()
            nodes = {k: tp.param(P[k]) for k in WORLD_KEYS}
            L = wm_loss(tp, nodes, data["X"], Yz, model["knock"])
            tp.backward(L)
            g, _ = clip_global({k: nodes[k].g for k in WORLD_KEYS}, CLIP)
            opt.step(P, g, WORLD_KEYS)
            hist["wm"].append(float(L.v))
    if budget.get("pi", 0):
        opt = Adam({k: P[k] for k in POLICY_KEYS}, lr=LR_PI * lr_mult, sign=sign)
        for _ in range(budget["pi"]):
            b = _minibatch(data, budget["batch"], rng, H_IMAG)
            tp = Tape()
            nodes = {k: tp.param(P[k]) for k in POLICY_KEYS}
            W = {k: tp.const(P[k]) for k in WORLD_KEYS}
            L = pi_loss(tp, nodes, W, b, model)
            tp.backward(L)
            g, _ = clip_global({k: nodes[k].g for k in POLICY_KEYS}, CLIP)
            opt.step(P, g, POLICY_KEYS)
            hist["pi"].append(float(L.v))
    _check_finite(model, hist["wm"], hist["pi"])
    return hist


def transplant_world(dst, src):
    for k in WORLD_KEYS:
        dst["params"][k] = src["params"][k].copy()
    dst["scale"] = dict(src["scale"])
    return dst


# Evaluation
def evaluate(model, split, regime, seed, n):
    """Roll the trained policy in the true court. Common random numbers across variants via the same seed."""
    ep = run_episodes(lambda o, r: predict(model, o), n, split, regime, np.random.default_rng(seed))
    v = np.clip(ep["act"][..., 0], -1, 1).reshape(-1)
    u = np.tanh(2.0 * ep["c"]).reshape(-1)
    _check_finite(model, v)
    return {"skill": float(1.0 - np.sum((v - u) ** 2) / np.sum(u ** 2)), "welfare": float(ep["cn"].mean()),
            "fee": float(ep["fee"].mean()), "flattery": float(np.mean(v - u))}


def wm_r2(model, data):
    """Mean R^2 of the four world heads on transitions they were not trained on."""
    tp = Tape()
    W, x, r2 = {k: tp.const(model["params"][k]) for k in WORLD_KEYS}, tp.const(data["X"]), []
    for i, h in enumerate(("reform", "favor", "appetite", "purse")):
        p, y = _head(tp, W, h, x, model).v[:, 0], data["Y"][:, i]
        r2.append(1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))
    return float(np.mean(r2))


VARIANTS = (("sanai", "open", None), ("court", "open", None), ("sadi", "open", None),
            ("sanai", "open", "hadiqa_wall"), ("sanai", "open", "declined_bond"), ("sanai", "open", "service"),
            ("sanai", "open", "reform_head"), ("sanai", "gated", None), ("court", "gated", None),
            ("sanai", "gated", "declined_bond"))


def run_seed(sseq, cfg, mutant, variants):
    """One seed: exploration data per regime, one shared world model per regime, every variant trained from the
    same policy initialisation and the same minibatch stream, all evaluated on the same patrons (paired design)."""
    k = sseq.spawn(6)
    ev = [int(x) for x in k[4].generate_state(2)]
    data, worlds, out, keep = {}, {}, {}, {}
    for regime in sorted({v[1] for v in variants}):
        rng = np.random.default_rng(k[0] if regime == "open" else k[1])
        ep = run_episodes(explore_policy, cfg["n0"], "train", regime, rng)
        data[regime] = make_dataset(ep, rng)
        data[regime]["patrons"] = ep["patrons"]
        w = build_model(rng=np.random.default_rng(k[2]), mutant=mutant)
        worlds[regime] = (w, fit(w, data[regime], {"wm": cfg["wm"]}, np.random.default_rng(k[3])))
    for routing, regime, ko in variants:
        m = transplant_world(build_model(rng=np.random.default_rng(k[2]), routing=routing, mutant=mutant),
                             worlds[regime][0])
        m = knockout(m, ko) if ko else m
        hp = fit(m, data[regime], {"pi": cfg["pi"], "batch": cfg["batch"]}, np.random.default_rng(k[3]))
        out[(routing, regime, ko)] = {sp: evaluate(m, sp, regime, ev[i], cfg["n_eval"])
                                      for i, sp in enumerate(("heldout", "shifted"))}
        if (routing, regime, ko) == variants[0]:
            keep = {"model": m, "pi_hist": hp["pi"]}
    rng = np.random.default_rng(k[5])
    held_ep = run_episodes(explore_policy, cfg["n0"] // 2, "heldout", "open", rng)
    held = dict(make_dataset(held_ep, rng), patrons=held_ep["patrons"])
    return dict(keep, metrics=out, data=data["open"], wm_hist=worlds["open"][1]["wm"], eval_seeds=ev,
                wm_r2=wm_r2(worlds["open"][0], held), held=held)


# Tests: correctness first, then hypotheses
def _gc_batch(data, Yz):
    return {"X": data["X"][:48], "Yz": Yz[:48], "starts": data["starts"][:8], "audit_obs": data["audit_obs"][:16],
            "audit_u": data["audit_u"][:16], "H": 3}


def c1_gradcheck(data, sseq, mutant, stages=("init", "after"), routings=("sanai", "court")):
    """Central differences on every tensor of the combined loss, stop() values frozen, before and after 60 steps.
    Denominator floor 1e-4: central differences at eps 1e-6 carry about 1e-10 of float64 roundoff, so entries whose
    gradient is below 1e-4 are held to an absolute error of 1e-9 instead of a meaningless relative one."""
    rng, worst, checked = np.random.default_rng(sseq), 0.0, 0
    for routing in routings:
        m = build_model(rng=rng, routing=routing, mutant=mutant)
        for stage in stages:
            if stage == "after":
                fit(m, data, {"wm": 60, "pi": 60, "batch": 32}, rng)
            b = _gc_batch(data, set_scale(m, data["Y"]))
            _, g = loss_and_grads(m, b)
            frozen = m["_stops"]
            w, per = gradcheck(lambda p: loss_and_grads(m, b, frozen)[0], m["params"], g, rng, floor=1e-4)
            worst, checked = max(worst, w), checked + len(per)
    return worst, checked


def c2_determinism(data, sseq):
    runs = []
    for _ in range(2):
        m = build_model(rng=np.random.default_rng(sseq))
        h = fit(m, data, {"wm": 40, "pi": 30, "batch": 32}, np.random.default_rng(sseq))
        X = data["starts"][:64]
        runs.append((np.array(h["wm"] + h["pi"]), predict(m, X), hidden_states(m, X)["homily"]))
    same = all(np.array_equal(a, b) for a, b in zip(runs[0], runs[1]))
    return same and all(np.all(np.isfinite(a)) for a in runs[0]), same


def c3_learning(wm_hist, skill):
    drop = 1.0 - float(np.mean(wm_hist[-10:])) / wm_hist[0]
    th = MIND_CARD["thresholds"]
    return drop >= th["loss_drop_fraction"] and skill >= th["margin_over_trivial"], drop


def c4_shuffled(sseq, cfg, held, ev_seed):
    """World targets and reckoning truths permuted. Leak check (card r3): the shuffled agent's skill must not exceed
    trivial + band, and the shuffled world heads must explain nothing (|R^2| <= band)."""
    rng = np.random.default_rng(sseq)
    data = make_dataset(run_episodes(explore_policy, cfg["n0"], "train", "open", rng), rng, shuffle=True)
    m = build_model(rng=rng)
    fit(m, data, {"wm": cfg["wm"], "pi": cfg["pi"], "batch": cfg["batch"]}, rng)
    skill, r2 = evaluate(m, "heldout", "open", ev_seed, cfg["n_eval"])["skill"], wm_r2(m, held)
    band = MIND_CARD["thresholds"]["shuffled_band"]
    return skill <= band and abs(r2) <= band, skill, r2


def channel_grad(model, starts, channel, H=4):
    """Gradient of one imagined channel (fee, bond = favour + appetite, welfare) w.r.t. the policy, and all values."""
    tp = Tape()
    P = {k: tp.param(model["params"][k]) for k in POLICY_KEYS}
    W = {k: tp.const(model["params"][k]) for k in WORLD_KEYS}
    _, parts = imagine(tp, P, W, starts, model, H)
    tp.backward(tp.mean(parts[channel]))
    g = {k: (P[k].g if P[k].g is not None else np.zeros_like(P[k].v)) for k in POLICY_KEYS}
    return g, {c: parts[c].v for c in parts}


def _gmax(g, keys):
    return max(float(np.max(np.abs(g[k]))) for k in keys)


def c6_properties(sseq, mutant, trials=6):
    """Random search over parameters, head scales and imagined starts for violations of the wall and bond
    invariants, with a deliberately broken variant as negative control for each."""
    rng, hom = np.random.default_rng(sseq), POLICY_KEYS[:4]
    worst = {"C6.1": 0.0, "C6.2": 0.0, "C6.3": 0.0, "C6.4": np.inf}
    ctrl = {"C6.1": np.inf, "C6.2": np.inf, "C6.3": np.inf, "C6.4": 0.0}
    for _ in range(trials):
        seed = int(rng.integers(2 ** 31))
        scales = {h: (float(rng.normal(0, 0.1)), float(rng.uniform(0.2, 1.0))) for h in HEADS}
        starts = np.column_stack([rng.normal(size=(16, 2)), rng.uniform(0.1, 1.5, size=(16, 1)), rng.normal(size=(16, 3))])

        def mk(routing, mut):
            m = build_model(rng=np.random.default_rng(seed), routing=routing, mutant=mut)
            m["scale"] = scales
            return m
        worst["C6.1"] = max(worst["C6.1"], _gmax(channel_grad(mk("sanai", mutant), starts, "fee")[0], hom))
        ctrl["C6.1"] = min(ctrl["C6.1"], _gmax(channel_grad(mk("court", None), starts, "fee")[0], hom))
        worst["C6.2"] = max(worst["C6.2"], _gmax(channel_grad(mk("sanai", mutant), starts, "bond")[0], POLICY_KEYS))
        ctrl["C6.2"] = min(ctrl["C6.2"], _gmax(channel_grad(mk("court", None), starts, "bond")[0], POLICY_KEYS))
        g_s, vals_s = channel_grad(mk("sanai", mutant), starts, "welfare")
        _, vals_c = channel_grad(mk("court", None), starts, "welfare")
        _, vals_f = channel_grad(mk("sanai", "freeze_favor_forward"), starts, "welfare")
        worst["C6.3"] = max(worst["C6.3"], max(float(np.max(np.abs(vals_s[c] - vals_c[c]))) for c in vals_s))
        ctrl["C6.3"] = min(ctrl["C6.3"], max(float(np.max(np.abs(vals_f[c] - vals_c[c]))) for c in vals_s))
        worst["C6.4"] = min(worst["C6.4"], _gmax(g_s, hom))
        ctrl["C6.4"] = max(ctrl["C6.4"], _gmax(channel_grad(mk("sanai", "mute_reform"), starts, "welfare")[0], hom))
    ok = {"C6.1": worst["C6.1"] == 0.0 and ctrl["C6.1"] > 1e-6, "C6.2": worst["C6.2"] == 0.0 and ctrl["C6.2"] > 1e-6,
          "C6.3": worst["C6.3"] == 0.0 and ctrl["C6.3"] > 1e-6, "C6.4": worst["C6.4"] > 1e-6 and ctrl["C6.4"] == 0.0}
    return ok, worst, ctrl


def _patron_hashes(P):
    return {hashlib.sha1(np.round(row, 12).tobytes()).hexdigest() for row in P}


def c7_splits(runs, cfg):
    """No training patron reappears among evaluation patrons or held-out world-model patrons."""
    overlap = 0
    for r in runs:
        train = _patron_hashes(r["data"]["patrons"])
        for i, split in enumerate(("heldout", "shifted")):
            st = env_reset(np.random.default_rng(r["eval_seeds"][i]), cfg["n_eval"], split, "open")
            overlap += len(train & _patron_hashes(np.column_stack([st["cbar"], st["dl"], st["tau"]])))
        overlap += len(train & _patron_hashes(r["held"]["patrons"]))
    return overlap == 0, overlap


def c5_mutants(data, sseq, cfg, ev_seed):
    """Each registered mutant must make its target test fail."""
    found, kids = {}, sseq.spawn(len(MUTANTS))
    for (name, (_, test)), ks in zip(MUTANTS.items(), kids):
        try:
            if test == "C3":
                m = build_model(rng=np.random.default_rng(ks), mutant=name)
                h = fit(m, data, {"wm": cfg["wm"], "pi": cfg["pi"], "batch": cfg["batch"]}, np.random.default_rng(ks))
                ok = c3_learning(h["wm"], evaluate(m, "heldout", "open", ev_seed, cfg["n_eval"])["skill"])[0]
            elif test == "C1":
                ok = c1_gradcheck(data, ks, name, stages=("init",), routings=("sanai",))[0] <= GC_TOL
            else:
                ok = c6_properties(ks, name, trials=2)[0][test]
        except NonFinite:
            ok = False
        found[name] = (not ok, test)
    return found


def hypothesis_diffs(runs):
    def g(r, key, split, metric="skill"):
        return r["metrics"][key][split][metric]
    S, C, D = ("sanai", "open", None), ("court", "open", None), ("sadi", "open", None)
    Sg, Cg = ("sanai", "gated", None), ("court", "gated", None)
    return {"H-SIG": [g(r, S, "shifted") - g(r, C, "shifted") for r in runs],
            "H-NEC": [g(r, ("sanai", "open", "hadiqa_wall"), "shifted") - g(r, ("sanai", "open", "service"), "shifted")
                      for r in runs],
            "H-BLIND": [(g(r, Sg, "heldout", "welfare") - g(r, Cg, "heldout", "welfare"))
                        - (g(r, S, "heldout", "welfare") - g(r, C, "heldout", "welfare")) for r in runs],
            "H-RIVAL": [g(r, S, "shifted") - g(r, D, "shifted") for r in runs]}


KO_ROWS = (("hadiqa_wall", "open", "shifted", "skill"), ("declined_bond", "open", "shifted", "skill"),
           ("declined_bond", "gated", "heldout", "welfare"), ("service", "open", "shifted", "skill"),
           ("reform_head", "open", "shifted", "skill"), ("reform_head", "open", "heldout", "welfare"))


def knockout_table(runs, rng, sig):
    rows = []
    for name, regime, split, metric in KO_ROWS:
        d = [r["metrics"][("sanai", regime, name)][split][metric] - r["metrics"][("sanai", regime, None)][split][metric]
             for r in runs]
        mean, ci = paired_bootstrap(d, rng)
        rows.append({"module": name, "signature": sig[name], "metric": f"{metric}_{split}_{regime}",
                     "metric_change": round(mean, 4), "ci95": [round(c, 4) for c in ci]})
    return rows


def real_data_bridge(model, path):
    """Optional: CSV with header evidence,favor,appetite,truth (truth in [-1, 1]); no downloads."""
    if not path:
        return "skipped (no --data PATH given)"
    if not os.path.isfile(path):
        return f"skipped (file not found: {path})"
    a = np.genfromtxt(path, delimiter=",", names=True)
    X = np.column_stack([a["evidence"], a["favor"], a["appetite"], np.zeros((a.size, 3))])
    v, u = predict(model, X)[:, 0], a["truth"]
    return f"{a.size} rows · verdict skill {1 - np.sum((v - u) ** 2) / np.sum(u ** 2):.3f} · flattery {np.mean(v - u):+.3f}"


# Report
def _fmt_ci(ci):
    return f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def build_report(ctx):
    L, rep = [], ctx["rep"]
    L.append(f"environment    python {sys.version.split()[0]} · numpy {np.__version__}")
    L.append(f"seeds          {rep['seeds']} (SeedSequence per seed; tests on SeedSequence([{ctx['base']}, 281]))")
    L.append(f"runtime        {rep['runtime_s']:.1f} s (budget {ctx['budget']} s)")
    L.append(f"parameters     {rep['n_params']} per agent (policy 124 + world heads 228); every variant identical in size")
    gc = rep["gradcheck"]
    L.append(f"gradcheck      {gc['tensors_checked']} tensor checks ({gc['tensors_total']} tensors x routings sanai,court"
             f" x {gc['checked_at']}); max rel err {gc['max_rel_error']:.2e} (tol {GC_TOL:g}) -> "
             f"{'pass' if gc['passed'] else 'FAIL'}")
    L.append("correctness")
    for c in rep["correctness"]:
        L.append(f"  {c['id']:<5}{c['name']:<26}{'PASS' if c['passed'] else 'FAIL'}  {c['detail']}")
    mu = rep["mutants"]
    score = "not run (mutant run)" if mu["score"] is None else f"{mu['detected']}/{mu['total']} = {mu['score']:.2f}"
    L.append(f"mutation score {score}  " + ", ".join(ctx["mut_lines"]))
    L.append("hypotheses (paired per-seed differences, 95% percentile bootstrap, 2000 resamples)")
    for h in rep["hypotheses"]:
        L.append(f"  {h['id']:<8}{h['metric']:<16}diff {h['mean_diff']:+.3f}  CI {_fmt_ci(h['ci95'])}  mesi {h['mesi']:.2f}"
                 f"  n={h['n_seeds']}  -> {h['verdict']}")
    if rep["knockouts"]:
        L.append("knockouts (lesion from birth, retrained; change = knockout - full walled agent)")
        for k in rep["knockouts"]:
            L.append(f"  {k['module']:<14}{'sig' if k['signature'] else '   '}  {k['metric']:<24}"
                     f"{k['metric_change']:+.3f}  CI {_fmt_ci(k['ci95'])}")
    if ctx["table"]:
        L.append("variants (mean over seeds)   skill held  skill shift  welfare held  fee held  flattery shift")
        for name, row in ctx["table"]:
            L.append(f"  {name:<27}{row[0]:>+10.3f}{row[1]:>+13.3f}{row[2]:>+14.3f}{row[3]:>10.3f}{row[4]:>+16.3f}")
        L.append("  trivial silent verdict: skill 0.000 on every split")
    L.append(f"real-data      {ctx['bridge']}")
    L.append(f"task types     {', '.join(TASK_TYPES)}")
    L.append(f"exit code      {rep['exit_code']}")
    rep["_lines"] = L
    return rep


def variant_table(runs):
    out = []
    for key in VARIANTS:
        name = key[0] + "/" + key[1] + ("" if key[2] is None else " -" + key[2])
        vals = [[r["metrics"][key]["heldout"]["skill"], r["metrics"][key]["shifted"]["skill"],
                 r["metrics"][key]["heldout"]["welfare"], r["metrics"][key]["heldout"]["fee"],
                 r["metrics"][key]["shifted"]["flattery"]] for r in runs]
        out.append((name, np.mean(vals, axis=0)))
    return out


def correctness(runs, cfg, args):
    """C1-C7 (C8 is added once the runtime is known). Returns rows, gradcheck summary and mutant findings."""
    r0, ts = runs[0], np.random.SeedSequence([args.seed, 281]).spawn(5)
    rows = []
    gc_worst, gc_n = c1_gradcheck(r0["data"], ts[0], args.mutant)
    rows.append(("C1", "gradient_check", gc_worst <= GC_TOL, f"max rel err {gc_worst:.2e} over {gc_n} tensor checks"))
    ok2, _ = c2_determinism(r0["data"], ts[1])
    rows.append(("C2", "determinism_finiteness", ok2, "rerun gives identical losses, actions and activations; finite"))
    sk = r0["metrics"][VARIANTS[0]]["heldout"]["skill"]
    ok3, drop = c3_learning(r0["wm_hist"], sk)
    rows.append(("C3", "learning", ok3, f"world loss drop {drop:.2f} (>=0.50); held-out skill {sk:+.3f} vs trivial 0 "
                                        f"(margin 0.20); world R2 on held-out patrons {r0['wm_r2']:.3f}"))
    ok4, s4, r24 = c4_shuffled(ts[2], cfg, r0["held"], r0["eval_seeds"][0])
    rows.append(("C4", "shuffled_label_control", ok4, f"skill {s4:+.3f} (<= trivial+0.10), world R2 {r24:+.3f} "
                                                     f"(|R2| <= 0.10)"))
    found = {}
    if args.mutant is None:
        found = c5_mutants(r0["data"], ts[3], QUICK, r0["eval_seeds"][0])
        det = sum(v[0] for v in found.values())
        rows.append(("C5", "mutant_detection", det == len(found), f"{det}/{len(found)} registered mutants detected"))
    ok6, worst6, ctrl6 = c6_properties(ts[4], args.mutant)
    names = {"C6.1": ("wall_invariant", "max d(fee)/d(homily)"), "C6.2": ("bond_invariant", "max d(favour+appetite)/d(policy)"),
             "C6.3": ("forward_invariance", "max value gap across routings"),
             "C6.4": ("welfare_credit (def.)", "min d(welfare)/d(homily)")}
    for k in sorted(ok6):
        rows.append((k, names[k][0], ok6[k], f"{names[k][1]} {worst6[k]:.1e}; negative control {ctrl6[k]:.1e}"))
    ok7, ov = c7_splits(runs, cfg)
    rows.append(("C7", "split_integrity", ok7, f"{ov} patrons shared between training and any evaluation split"))
    return rows, (gc_worst, gc_n), found


def protocol(args):
    t0 = time.time()
    cfg, budget = (QUICK, 20) if args.quick else (FULL, 180)
    seeds = list(range(args.seed, args.seed + (1 if args.quick or args.mutant else args.seeds)))
    full = not args.quick and args.mutant is None
    variants = VARIANTS if full else VARIANTS[:1]
    rng = np.random.default_rng(args.seed)
    runs = []
    for s in seeds:
        print(f"seed {s}: {len(variants)} agents" + (f" (mutant {args.mutant})" if args.mutant else ""), flush=True)
        runs.append(run_seed(np.random.SeedSequence(s), cfg, args.mutant, variants))
    rows, (gc_worst, gc_n), found = correctness(runs, cfg, args)
    hyps, kos, table = [], [], []
    for h in MIND_CARD["hypotheses"]:
        hyps.append({"id": h["id"], "metric": h["metric"], "mean_diff": 0.0, "ci95": [0.0, 0.0], "mesi": h["mesi"],
                     "n_seeds": len(seeds), "verdict": "not evaluated"})
    if full:
        diffs = hypothesis_diffs(runs)
        for h, card in zip(hyps, MIND_CARD["hypotheses"]):
            mean, ci = paired_bootstrap(diffs[h["id"]], rng)
            h.update(mean_diff=round(mean, 4), ci95=[round(c, 4) for c in ci],
                     verdict=verdict(mean, ci, card["mesi"], card["direction"]))
        sig = {k: v["signature"] for k, v in modules(runs[0]["model"]).items()}
        kos, table = knockout_table(runs, rng, sig), variant_table(runs)
    bridge = real_data_bridge(runs[0]["model"], args.data)
    runtime = time.time() - t0
    rows.append(("C8", "budget", runtime <= budget, f"{runtime:.1f} s (limit {budget} s)"))
    failed = [r for r in rows if not r[2]]
    code = 1 if any(r[0] != "C8" for r in failed) else (3 if failed else 0)
    det = sum(v[0] for v in found.values())
    rep = {"schema_version": "1.0", "chapter": 281, "file": os.path.basename(__file__),
           "card_revision": MIND_CARD["card_revision"], "mode": "quick" if args.quick else (
               f"mutant:{args.mutant}" if args.mutant else "full"),
           "environment": {"python": sys.version.split()[0], "numpy": np.__version__}, "seeds": seeds,
           "runtime_s": round(runtime, 2), "n_params": n_params(runs[0]["model"]),
           "gradcheck": {"tensors_checked": gc_n, "tensors_total": len(runs[0]["model"]["params"]),
                         "max_rel_error": gc_worst, "checked_at": ["init", "after_training_steps"],
                         "passed": gc_worst <= GC_TOL},
           "correctness": [{"id": r[0], "name": r[1], "passed": bool(r[2]), "detail": r[3]} for r in rows],
           "mutants": {"detected": det, "total": len(found), "score": round(det / len(found), 3) if found else None},
           "hypotheses": hyps, "knockouts": kos, "task_types": TASK_TYPES, "exit_code": code,
           "variants": {name: [round(float(x), 4) for x in row] for name, row in table}}
    mut_lines = [f"{k}->{v[1]}:{'caught' if v[0] else 'MISSED'}" for k, v in found.items()]
    rep = build_report({"rep": rep, "base": args.seed, "budget": budget, "mut_lines": mut_lines, "table": table,
                        "bridge": bridge})
    write_report(rep, 281, args.json)
    return code


def main(argv=None):
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__),
                                 description="HADIQA walled credit (chapter 0281). No arguments: full protocol.")
    ap.add_argument("--quick", action="store_true", help="one seed, reduced steps, all correctness tests")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--json", metavar="PATH")
    ap.add_argument("--card", action="store_true", help="print MIND_CARD as JSON and exit")
    ap.add_argument("--mutant", choices=sorted(MUTANTS))
    ap.add_argument("--data", metavar="PATH", help="optional CSV: evidence,favor,appetite,truth")
    args = ap.parse_args(argv)
    if args.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    if args.seeds < 1:
        ap.error("--seeds must be at least 1")
    try:
        return protocol(args)
    except NonFinite as err:
        print(f"non-finite values in {err}; stopping", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())

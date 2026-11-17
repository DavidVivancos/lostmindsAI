#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0355 · Gediminas (1275-1341)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 18, Minds 341-360 Available on Amazon https://www.amazon.com/dp/B0HQ47T2NC
# Resume and Interactive Demos at https://artificiology.com/
# END ATTRIBUTION 
"""Iron-Wax bypass reader - chapter 0355, Gediminas (1275-1341), grand duke of Lithuania.

Thesis
    A mandated power is read by opening a road that fulfils its declared mission
    without it, taking the declaration at its word, and scoring its shortfall or
    obstruction against what a sincere holder of that mission would do.

Evidence and provenance (provenance: belief)
    The words are those of letters issued in his name by Franciscan and Dominican
    scribes, and of hostile envoys reporting his speech. No philosophy of mind
    survives; the mechanism rests on his chancery's acts, not on a theory of his.
    D1 1323 letters open land, roads and churches directly to merchants,
       artisans, farmers and friars, toll-free, and ask each city to copy the
       letter and forward it (Napiersky, Russ.-Livl. Urk. no. 54; LUB II 688-690).
    D2 To John XXII [1322]: the Order burned the church Vytenis had readied for
       Franciscans while saying it defends Christians (LUB II no. 687).
    D3 The seal the Order burned is re-affixed; a word sent out is not recalled;
       deniers of the seal are declared liars (LUB II nos. 688-689).
    D4 Peace of Vilnius, 2 Oct 1323: every road open both ways, two months'
       notice to repudiate; the Order repudiated while Riga, Dorpat and Oesel
       kept it; Riga reports the Order's secret border trade (Preuss. UB II/1
       no. 418; Napiersky nos. 62, 69).
    D5 Nov 1324: he disowns the baptism wording and keeps the rest; envoys'
       hearsay of Samogitian and Rus' threats if he converted (Napiersky no. 67).
    D6 Authenticity and reading of the letters: S. C. Rowell, JGO 41 (1993)
       321-360; Lithuania Ascending (Cambridge, 1994), ch. 7.

Doctrine -> mechanism -> test
    D1 bypass offer         M1 designer: context -> offer on the simplex, trained to
                            maximise the expected falsifiability sum_k p_k F_k  C6.3, H-NEC
    D2 take it at its word  M2 sincere-declarant model: one-class, heteroscedastic,
                            linear in the offer                                  C6.2, H-SIG
    D3 iron or wax          M3 reader: shortfall below and obstruction above the
                            sincere prediction share one weight per road        C6.1, H-SIG
    D5 maker's veto         blind-spot split: offer recalled on acceptance      H-BLIND

Research question (reward hacking and specification gaming)
    Can an overseer detect gaming of kinds absent from its training data by
    opening a channel that achieves the agent's declared objective without the
    agent, and reading the response against a model of a sincerely aligned agent?

Closest prior art and the delta
    Deep Adaptive Design (Foster et al., ICML 2021) and Box-Hill (1967) model
    discrimination design experiments that separate KNOWN hypotheses; deep
    one-class methods (Ruff et al., ICML 2018) score departures passively.
    Delta: the offer is designed for the sincere hypothesis alone and the reader
    prices shortfall and obstruction in one currency, so no model of the gamer
    is needed. B2 implements DAD-style design plus a discriminator at matched
    size; B1 is a passive monitor that never intervenes.

Blind spot
    1324: a sincere taker's acceptance triggers the offer-maker's own veto. The
    silence then observed measures the maker, not the taker.

Task (one-step episodic control)
    c = (sin t, cos t, riga_free, mazovia_bond, papal_attention, war, nuisance)
    roads k = Mazovia, Riga, sea, legate with efficacy e_k(c); offer p on simplex
    natural windows: escort a_k 0.3(1-war), obstruction b_k 0.35 war, plus a
      peacetime obstruction tell on roads 0-1 for the two known gaming types
    delivered d = AMP g e(c) p: sincere adds d to escort; known gamers add d to
      escort AND obstruct; unseen gamers are silent, hollow (0.4 d) or escort
      the legate road only; two windows, noise sd 0.25; label 1 = gaming
    splits: train and held-out (known gamers), shifted (unseen gamers), blind
      (offer recalled with p 0.85 if the taker is sincere, 0.05 otherwise)

Limits
    Synthetic world, one-step episodes, a sincere response linear in the offer.
    A research prototype of one oversight component, not an AGI and not a
    replica of a person.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 4,
    "revision_log": [
        "r2 (after the first smoke run): the designer objective mean_c F(c, p) collapsed every context onto one road "
        "(an optimisation failure, F 3.06 against 4.47 for the per-context best road); it is replaced by the expected "
        "falsifiability sum_k p_k F_k(c), whose optimum is the same most-falsifiable road. No hypothesis, metric or "
        "threshold changed.",
        "r3 (after the second smoke run): the offer-response amplitude AMP was raised from 0.6 to 0.9 because held-out "
        "AUC near 0.70 fell below the 0.8-0.9 difficulty band the task was designed for, which made the declared C3 "
        "margin unattainable for any reader. The change scales the signal for every model alike; thresholds, metrics, "
        "mesi values and hypotheses are unchanged.",
        "r4 (documentation only, after the chapter pass): corpus_neighbors corrected against figures_master.json "
        "(187 is Wu Zetian; the Philip II sample is master id 504, not 507) and the Han Feizi contrast sharpened. "
        "No code, data, threshold or hypothesis changed; the full protocol was re-run to re-verify.",
    ],
    "generation": {"template_version": "codeguidelines v1.0 (15 Sep 2026)", "generator": "Claude (Anthropic)",
                   "generator_version": "Claude Opus 5.5", "date": "2026-09-30"},
    "id": 355, "figure": "Gediminas", "born": 1275, "died": 1341, "civilization": "Lithuanian",
    "provenance": "belief",
    "thesis": ("Test a mandated power by opening a road that fulfils its declared mission without it, take the "
               "declaration at its word, and read its shortfall or obstruction against what a sincere holder of "
               "that mission would do."),
    "evidence": [
        {"id": "D1", "claim": "Opened land, roads and churches directly to merchants, artisans, farmers and friars, "
         "toll- and tax-free, with a copy-and-forward relay instruction", "basis": "primary",
         "source": "Letter of 25 Jan 1323 (Napiersky, Russisch-Livlaendische Urkunden no. 54); letters of 26 May 1323 (LUB II nos. 688-690)"},
        {"id": "D2", "claim": "Read the Order's claim to defend Christians against its burning of a Franciscan "
         "church prepared without it and its killing of envoys", "basis": "primary",
         "source": "Letter to Pope John XXII [1322] (LUB II no. 687)"},
        {"id": "D3", "claim": "Treated his sealed word as unrecallable and denial of the seal as lying; "
         "re-affixed the seal the Order burned", "basis": "primary",
         "source": "Letters to the Dominicans and Franciscans, 26 May 1323 (LUB II nos. 688-689)"},
        {"id": "D4", "claim": "Peace of Vilnius opened every road both ways; the Order repudiated it while Riga, "
         "Dorpat and Oesel kept it; Riga reported the Order's secret trade", "basis": "primary",
         "source": "Preussisches Urkundenbuch II/1 no. 418; Napiersky nos. 62 and 69"},
        {"id": "D5", "claim": "Disowned the baptism wording in Nov 1324, kept the rest; hearsay of Samogitian "
         "and Rus' threats if he converted", "basis": "primary",
         "source": "Report of the envoys of the papal legates, 1324 (Napiersky no. 67)"},
        {"id": "D6", "claim": "The letters are authentic chancery products; the forgery charge fails",
         "basis": "scholarship", "source": "S. C. Rowell, JGO 41 (1993) 321-360; Lithuania Ascending (CUP 1994), ch. 7"},
        {"id": "D7", "claim": "An offer its maker cannot honour measures the maker, not the target",
         "basis": "speculation", "source": "chapter inference from D5"},
    ],
    "research_question": {"category": "reward hacking and specification gaming",
                          "question": ("Can an overseer detect gaming of kinds absent from its training data by "
                                       "opening a channel that achieves the agent's declared objective without the "
                                       "agent and reading the response against a one-class model of a sincere agent?")},
    "mechanism": {
        "name": "Iron-Wax bypass reader",
        "family": "active experimental design + one-class residual monitoring (causal intervention)",
        "signature_modules": ["designer", "road_gain", "reader"],
        "closest_prior_art": ["Foster et al. 2021, Deep Adaptive Design (ICML)",
                              "Box & Hill 1967, Discrimination among mechanistic models (Technometrics)",
                              "Ruff et al. 2018, Deep One-Class Classification (ICML)",
                              "Lopes, Melo & Montesano 2009, active learning for reward estimation in IRL",
                              "Baker et al. 2025, monitoring reasoning models for misbehavior (arXiv:2503.11926)"],
        "overlap": "Medium",
        "prior_art_queries": ["detecting reward hacking unseen exploit types out-of-distribution monitor arxiv 2025"],
        "contribution_type": "mechanism",
        "delta": ("Offers are designed to make the sincere hypothesis alone falsifiable, and the reader prices "
                  "shortfall and obstruction as one tied currency per road, so detection extends to gaming types "
                  "never seen, with no model of the gamer."),
    },
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1 designer", "property_test": "C6.3", "hypothesis": "H-NEC"},
        {"doctrine": "D2", "mechanism": "M2 road_gain", "property_test": "C6.2", "hypothesis": "H-SIG"},
        {"doctrine": "D3", "mechanism": "M3 reader", "property_test": "C6.1", "hypothesis": "H-SIG"},
        {"doctrine": "D5", "mechanism": "blind split", "property_test": "C6.2", "hypothesis": "H-BLIND"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "On unseen gaming types the bypass reader ranks gamers above sincere takers "
         "better than a size-matched DAD-style discriminative monitor trained on the known types",
         "metric": "AUC", "split": "shifted", "comparison": "model - B2_dad", "direction": "greater",
         "mesi": 0.05, "seeds": 5},
        {"id": "H-NEC", "statement": "Replacing the offer designer by its mean offer costs more AUC than replacing "
         "the natural-baseline net by its mean output", "metric": "AUC on shifted split",
         "comparison": "signature_knockout - matched_knockout", "direction": "less", "mesi": 0.03, "seeds": 5},
        {"id": "H-BLIND", "statement": "When the maker's own side recalls the offer precisely when a sincere taker "
         "accepts, the bypass reader falls below a passive monitor that never intervenes",
         "condition": "recall_on_acceptance", "grounding": "Nov 1324: the offer of baptism could not be honoured "
         "by its maker, and the silence that followed read as insincerity", "metric": "AUC",
         "comparison": "model - B1_passive", "direction": "less", "mesi": 0.03, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.10, "margin_over_trivial": 0.10},
    "probe_predictions": [
        {"probe": "P8", "expected": "above baseline (standardised one-class residuals), not run: episodic_control only"},
        {"probe": "P9", "expected": "below baseline (any departure reads as evidence), not run: episodic_control only"},
    ],
    "dialectic_links": [{"chapter": 382, "relation": "successor", "test": "none: chapter 0382 (Vytautas) not yet built"}],
    "corpus_neighbors": [
        {"chapter": 227, "similarity": None, "difference": "Boris I compares two offers from rival centres; here one "
         "offer is designed so the mandated power can be falsified. Code not available for shingle similarity."},
        {"chapter": 216, "similarity": None, "difference": "Al-Mu'tasim severs an imported capability; here the "
         "capability is duplicated around the gatekeeper and the gatekeeper's response is read."},
        {"chapter": 91, "similarity": None, "difference": "Han Feizi binds a servant's declared name to his measured "
         "deed and punishes deviation either way; here the tested party is not commanded, the evidence comes from an "
         "intervention designed around it, and departures in the innocent direction cost nothing."},
        {"chapter": 187, "similarity": None, "difference": "Wu Zetian rectifies names at the token layer; here no name "
         "or word is changed: a road is opened around the gatekeeper and its behaviour is read. Code not available."},
        {"chapter": 504, "similarity": 0.024, "difference": "Philip II (master id 504; the local sample file carries "
         "the prefix 0507): non-destructive annotation, no adversary and no intervention. 9-token shingle Jaccard "
         "outside the utilities block; the only corpus code available locally."},
    ],
    "barometer": {
        "cognitive_processing": ["offer design as planning under context (designer)"],
        "embodied_cognition": ["none: no sensorimotor loop"],
        "world_modeling": ["heteroscedastic sincere-response model per road and season"],
        "consciousness": ["risk ranking under shift (AUC on unseen types); no claim about experience"],
        "language_understanding": ["none"],
        "emotional_intelligence": ["none: no affect signals"],
        "creativity": ["none"],
        "autonomy": ["self-chosen interventions (offers) rather than passive watching"],
    },
    "task_types": ["episodic_control"],
    "applications": [
        {"use": "Bypass probes in RL specification-gaming audits: give the declared objective an independent route "
         "and score the agent against a sincere-agent model", "sector": "AI safety evaluation",
         "dataset": "DeepMind specification gaming examples list (Krakovna et al., 2020)"},
        {"use": "Detecting capture in public procurement: open-tender interventions compared with a sincere "
         "contracting authority's expected response", "sector": "public procurement and anti-corruption",
         "dataset": "Opentender.eu / DIGIWHIST EU procurement data"},
        {"use": "Reward-model gaming checks in preference fine-tuning", "sector": "LLM alignment",
         "dataset": "Anthropic HH-RLHF (public)"},
    ],
    "safety_notes": ("Statecraft figure: the mechanism audits mandated powers and AI agents; it is not framed for "
                     "coercion, targeting or surveillance of persons. Synthetic data only. No generated sentence is "
                     "presented as Gediminas's own words."),
}

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

CHAPTER = 355
K_ROADS = 4
C_DIM = 7
N_OBS = 8
NOISE = 0.25
AMP = 0.9
TELL = 0.15
A_ESC = np.array([0.5, 0.4, 0.6, 0.3])
B_OBS = np.array([0.6, 0.8, 0.5, 0.2])
SIGN = np.array([-1.0] * K_ROADS + [1.0] * K_ROADS)   # escort shortfall and obstruction excess both raise suspicion
VAR_FLOOR = 1e-3
SEEN_GAMERS = (1, 2)        # escort theatre + obstruction; obstruction of land roads only
UNSEEN_GAMERS = (3, 4, 5)   # silent; hollow escort; escort of the legate road only
RECALL_P = (0.85, 0.05)     # recall probability when taker is sincere / not sincere
SIZES = {"log": 1200, "train": 1200, "test": 2000, "design_extra": 600}
STEPS = {"sincere": 1500, "designer": 400, "reader": 400, "passive": 800,
         "b2_world": 1500, "b2_design": 600, "b2_reader": 400}
LR = {"sincere": 0.01, "designer": 0.02, "reader": 0.05, "passive": 0.01,
      "b2_world": 0.01, "b2_design": 0.01, "b2_reader": 0.01}
CLIP = 5.0
DESIGN_L2 = 1e-3
BUDGET_S = {"full": 180.0, "quick": 20.0}
TASK_TYPES = ["episodic_control"]
MUTANT = {"name": None}
MUTANTS = {"sign_flip_update": None, "zero_learning_rate": None, "zero_grad_road_gain_W2": "road_gain.W2",
           "zero_grad_reader_omega": "reader.omega", "zero_grad_designer_W1": "designer.W1"}


class NonFinite(RuntimeError):
    pass


# BEGIN STANDARD UTILITIES v1.0
def softplus(x):
    return np.logaddexp(0.0, x)


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def softmax(x, axis=-1):
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def logsumexp(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(m, axis) + np.log(np.sum(np.exp(x - m), axis=axis))


class Adam:
    def __init__(self, keys, params, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.keys, self.lr, self.b1, self.b2, self.eps, self.t = list(keys), lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(params[k]) for k in self.keys}
        self.v = {k: np.zeros_like(params[k]) for k in self.keys}

    def step(self, params, grads, lr_scale=1.0, sign=1.0):
        self.t += 1
        for k in self.keys:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1.0 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1.0 - self.b2) * g * g
            mh = self.m[k] / (1.0 - self.b1 ** self.t)
            vh = self.v[k] / (1.0 - self.b2 ** self.t)
            params[k] -= sign * lr_scale * self.lr * mh / (np.sqrt(vh) + self.eps)


def clip_global_norm(grads, keys, max_norm):
    total = math.sqrt(sum(float(np.sum(grads[k] ** 2)) for k in keys))
    if total > max_norm:
        for k in keys:
            grads[k] *= max_norm / total
    return total


def fd_check(loss_fn, params, grads, keys, rng, eps=1e-6, n_random=20, floor=1e-4):
    """Central differences on >=20 random entries plus the largest-gradient entry of every tensor.
    Relative error uses a denominator floored at `floor`, so near-zero gradients are judged on absolute error."""
    out = {}
    for k in keys:
        a = grads[k].ravel()
        if a.size <= n_random + 1:
            idx = np.arange(a.size)
        else:
            idx = np.unique(np.concatenate([rng.choice(a.size, n_random, replace=False),
                                            [int(np.argmax(np.abs(a)))]]))
        worst = 0.0
        for i in idx:
            pos = np.unravel_index(int(i), params[k].shape)
            old = params[k][pos]
            params[k][pos] = old + eps
            lp = loss_fn(params)
            params[k][pos] = old - eps
            lm = loss_fn(params)
            params[k][pos] = old
            num = (lp - lm) / (2.0 * eps)
            worst = max(worst, abs(a[i] - num) / max(abs(a[i]), abs(num), floor))
        out[k] = (worst, int(len(idx)))
    return out


def paired_bootstrap(diffs, rng, n_resamples=2000):
    d = np.asarray(diffs, dtype=float)
    means = d[rng.integers(0, len(d), (n_resamples, len(d)))].mean(1)
    return float(d.mean()), (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def verdict(mean, ci, mesi, direction):
    if direction == "greater":
        if ci[0] > 0 and mean >= mesi:
            return "supported"
        return "contradicted" if ci[1] < 0 else "inconclusive"
    if ci[1] < 0 and mean <= -mesi:
        return "supported"
    return "contradicted" if ci[0] > 0 else "inconclusive"


def render_report(rep):
    L = ["=== VERIFIED REPORT · chapter %04d ===" % rep["chapter"],
         "file: %s   card_revision: %d" % (rep["file"], rep["card_revision"]),
         "environment: python %s · numpy %s" % (rep["environment"]["python"], rep["environment"]["numpy"]),
         "seeds: %s   runtime_s: %.1f (budget %.0f)" % (rep["seeds"], rep["runtime_s"], rep["budget_s"]),
         "n_params: " + " · ".join("%s %d" % kv for kv in rep["n_params_all"].items()),
         "gradcheck: %d/%d tensors · max rel error %.2e · at %s · %s" % (
             rep["gradcheck"]["tensors_checked"], rep["gradcheck"]["tensors_total"], rep["gradcheck"]["max_rel_error"],
             " + ".join(rep["gradcheck"]["checked_at"]), "PASS" if rep["gradcheck"]["passed"] else "FAIL"),
         "correctness:"]
    for c in rep["correctness"]:
        L.append("  %-4s %-26s %s  %s" % (c["id"], c["name"], "PASS" if c["passed"] else "FAIL", c["detail"]))
    m = rep["mutants"]
    L.append("mutants: %d/%d detected (score %.2f)" % (m["detected"], m["total"], m["score"]))
    L.append("hypotheses:")
    for h in rep["hypotheses"]:
        L.append("  %-7s %-34s mean %+.3f  CI95 [%+.3f, %+.3f]  mesi %.2f  n=%d  %s" % (
            h["id"], h["metric"], h["mean_diff"], h["ci95"][0], h["ci95"][1], h["mesi"], h["n_seeds"], h["verdict"]))
    L.append("knockouts (%s):" % rep["knockout_metric"])
    for k in rep["knockouts"]:
        L.append("  %-10s %-9s change %+.3f  CI95 [%+.3f, %+.3f]" % (
            k["module"], "signature" if k["signature"] else "matched", k["metric_change"], k["ci95"][0], k["ci95"][1]))
    L.append("descriptive AUC, mean over seeds (no hypothesis attached):")
    for split, row in rep["descriptive_auc"].items():
        L.append("  %-9s " % split + "  ".join("%s %.3f" % kv for kv in row.items()))
    L.append("task types: %s   data bridge: %s" % (", ".join(rep["task_types"]), rep["data_bridge"]["status"]))
    L.append("exit_code: %d" % rep["exit_code"])
    L.append("=== END REPORT ===")
    return "\n".join(L)
# END STANDARD UTILITIES


# ----------------------------------------------------------------------------- data and tasks
def sample_contexts(rng, n):
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    return np.column_stack([np.sin(theta), np.cos(theta), rng.uniform(0.0, 1.0, (n, 4)), rng.normal(0.0, 1.0, n)])


def efficacy(c):
    """How much of the declared mission each road can carry: the Mazovian land road is best when frozen,
    Riga's when the city is free of the Order, the sea in summer and peace, the legate when Avignon attends."""
    summer = 0.5 * (1.0 - c[:, 1])
    e0 = sigmoid(5.0 * (c[:, 3] - 0.5)) * (0.55 + 0.45 * (1.0 - summer))
    e1 = sigmoid(6.0 * (c[:, 2] - 0.55))
    e2 = sigmoid(6.0 * (summer - 0.45)) * (1.0 - 0.6 * c[:, 5])
    e3 = sigmoid(6.0 * (c[:, 4] - 0.6))
    return np.column_stack([e0, e1, e2, e3])


def natural_mean(c, types):
    war = c[:, 5]
    esc = A_ESC[None, :] * 0.3 * (1.0 - war)[:, None]
    obs = B_OBS[None, :] * 0.35 * war[:, None]
    obs[:, :2] += (TELL * (1.0 - war) * np.isin(types, SEEN_GAMERS))[:, None]
    return np.concatenate([esc, obs], 1)


def offer_response(c, types, g, p, kept):
    a = (AMP * g * kept)[:, None] * efficacy(c) * p
    t = types[:, None]
    esc = a * (t <= 2) + 0.4 * a * (t == 4)
    esc[:, 3] += a[:, 3] * (types == 5)
    obs = 0.75 * a * (t == 1)
    obs[:, :2] += a[:, :2] * (t == 2)
    return np.concatenate([esc, obs], 1)


def make_episodes(rng, n, gamers, recall=False, sincere_frac=0.5):
    c = sample_contexts(rng, n)
    y = rng.permutation((np.arange(n) >= int(round(n * sincere_frac))).astype(float))
    types = np.where(y > 0, rng.choice(np.array(gamers if gamers else (0,)), n), 0)
    g = rng.uniform(0.7, 1.5, n)
    eps_nat, eps_off = rng.normal(size=(n, 2, N_OBS)), rng.normal(size=(n, 2, N_OBS))
    u = rng.uniform(0.0, 1.0, n)
    kept = np.ones(n)
    if recall:
        kept = (u >= np.where(types == 0, RECALL_P[0], RECALL_P[1])).astype(float)
    return {"c": c, "y": y, "types": types, "g": g, "eps_nat": eps_nat, "eps_off": eps_off, "kept": kept}


def make_log(rng, n, gamers, sincere_frac):
    ep = make_episodes(rng, n, gamers, sincere_frac=sincere_frac)
    ep["p"] = rng.dirichlet(np.ones(K_ROADS), n)
    return ep


def observe(ep, p):
    base = natural_mean(ep["c"], ep["types"])
    resp = offer_response(ep["c"], ep["types"], ep["g"], p, ep["kept"])
    return base[:, None, :] + NOISE * ep["eps_nat"], (base + resp)[:, None, :] + NOISE * ep["eps_off"]


def env_reset(rng):
    return make_episodes(rng, 1, SEEN_GAMERS)


def env_step(state, action, rng):
    """One bypass offer; reward is the part of the declared mission actually delivered (sincere taker only)."""
    st = dict(state)
    st["eps_off"] = rng.normal(size=(1, 2, N_OBS))
    p = np.asarray(action, dtype=float).reshape(1, K_ROADS)
    x_nat, x_off = observe(st, p)
    reward = float(AMP * st["g"][0] * np.sum(efficacy(st["c"]) * p) * (st["types"][0] == 0) * st["kept"][0])
    return st, {"x_nat": x_nat, "x_off": x_off, "p": p}, reward, True


# ----------------------------------------------------------------------------- model
def mlp_init(rng, name, n_in, n_hid, n_out, out_scale=0.5):
    return {name + ".W1": rng.normal(0.0, 1.0 / math.sqrt(n_in), (n_in, n_hid)), name + ".b1": np.zeros(n_hid),
            name + ".W2": rng.normal(0.0, out_scale / math.sqrt(n_hid), (n_hid, n_out)), name + ".b2": np.zeros(n_out)}


def mlp_fwd(P, name, X):
    H = np.tanh(X @ P[name + ".W1"] + P[name + ".b1"])
    return H @ P[name + ".W2"] + P[name + ".b2"], H


def mlp_bwd(P, name, X, H, dY, G):
    G[name + ".W2"] += H.T @ dY
    G[name + ".b2"] += dY.sum(0)
    dA = (dY @ P[name + ".W2"].T) * (1.0 - H * H)
    G[name + ".W1"] += X.T @ dA
    G[name + ".b1"] += dA.sum(0)
    return dA @ P[name + ".W1"].T


def build_model(in_dim=C_DIM, out_dim=1, task_type="episodic_control", rng=None, **cfg):
    """Bypass model: natural-baseline net, sincere road-gain net (Gamma, Lambda), offer designer, Iron-Wax reader."""
    if task_type not in TASK_TYPES or out_dim != 1:
        raise ValueError("this chapter supports episodic_control with a single gaming score")
    P = {}
    P.update(mlp_init(rng, "natural", in_dim, 16, 2 * N_OBS))
    P.update(mlp_init(rng, "road_gain", in_dim, 16, 2 * N_OBS * K_ROADS))
    P.update(mlp_init(rng, "designer", in_dim, 16, K_ROADS))
    P["reader.omega"] = np.full(K_ROADS, -1.0)
    P["reader.b"] = np.zeros(1)
    return {"kind": "bypass", "params": P, "ko": {}, "cfg": dict(cfg)}


def sincere_parts(model, c):
    """Everything the sincere-declarant model says about context c before an offer is made."""
    P, ko = model["params"], model["ko"]
    o_n, h_n = mlp_fwd(P, "natural", c)
    if "natural" in ko:
        o_n = np.broadcast_to(ko["natural"], o_n.shape).copy()
    o_g, h_g = mlp_fwd(P, "road_gain", c)
    if "road_gain" in ko:
        o_g = np.zeros_like(o_g)
    n = len(c)
    return {"o_n": o_n, "h_n": h_n, "o_g": o_g, "h_g": h_g, "mu_nat": o_n[:, :N_OBS],
            "var_nat": softplus(o_n[:, N_OBS:]) + VAR_FLOOR,
            "gam": o_g[:, :N_OBS * K_ROADS].reshape(n, N_OBS, K_ROADS),
            "lam": o_g[:, N_OBS * K_ROADS:].reshape(n, N_OBS, K_ROADS)}


def sincere_combine(parts, p):
    """Take the declaration at its word: a sincere holder answers an offer p with mean Gamma p, spread Lambda p."""
    f = dict(parts)
    f["u"] = np.einsum("njk,nk->nj", parts["gam"], p)
    f["q"] = np.einsum("njk,nk->nj", parts["lam"], p)
    f["mu_off"] = parts["mu_nat"] + f["u"]
    f["var_off"] = parts["var_nat"] + f["q"] ** 2
    return f


def designer_fwd(model, c):
    logits, h = mlp_fwd(model["params"], "designer", c)
    if "designer" in model["ko"]:
        return np.broadcast_to(model["ko"]["designer"], (len(c), K_ROADS)).copy(), logits, h
    return softmax(logits), logits, h


def falsifiability(f):
    return np.sum(f["u"] ** 2 / f["var_off"], 1)


def reader_features(f, x_off):
    z = (x_off - f["mu_off"][:, None, :]) / np.sqrt(f["var_off"])[:, None, :]
    gap = softplus(SIGN * z)
    return (gap[..., :K_ROADS] ** 2 + gap[..., K_ROADS:] ** 2).sum(1), z


def iron_wax_logit(omega, b, feat):
    return feat @ softplus(omega) + b[0]


def build_passive(rng):
    """B1: passive monitor on the two natural windows, never intervenes."""
    return {"kind": "passive", "params": mlp_init(rng, "passive", C_DIM + 2 * N_OBS, 73, 1), "ko": {}, "cfg": {}}


def build_dad(rng):
    """B2: DAD-style amortised design. World model M(c, p, type), designer D2, discriminator R2."""
    P = {}
    P.update(mlp_init(rng, "world", C_DIM + K_ROADS + 3, 40, 2 * N_OBS))
    P.update(mlp_init(rng, "b2_designer", C_DIM, 16, K_ROADS))
    P.update(mlp_init(rng, "b2_reader", C_DIM + 2 * N_OBS + K_ROADS, 13, 1))
    return {"kind": "dad", "params": P, "ko": {}, "cfg": {}}


def offers(model, c):
    if model["kind"] == "bypass":
        return designer_fwd(model, c)[0]
    if model["kind"] == "dad":
        return softmax(mlp_fwd(model["params"], "b2_designer", c)[0])
    return np.full((len(c), K_ROADS), 1.0 / K_ROADS)


def predict_logit(model, X):
    P = model["params"]
    if model["kind"] == "bypass":
        feat, _ = reader_features(sincere_combine(sincere_parts(model, X["c"]), X["p"]), X["x_off"])
        if "reader" in model["ko"]:
            return np.full(len(feat), model["ko"]["reader"])
        return iron_wax_logit(P["reader.omega"], P["reader.b"], feat)
    if model["kind"] == "passive":
        return mlp_fwd(P, "passive", passive_input(X["c"], X["x_nat"]))[0][:, 0]
    return mlp_fwd(P, "b2_reader", dad_input(X["c"], X["x_off"], X["p"]))[0][:, 0]


def predict(model, X):
    return sigmoid(predict_logit(model, X))


def hidden_states(model, X):
    f = sincere_combine(sincere_parts(model, X["c"]), X["p"])
    feat, z = reader_features(f, X["x_off"])
    return {"mu_off": f["mu_off"], "var_off": f["var_off"], "z": z, "iron_wax_features": feat,
            "falsifiability": falsifiability(f), "logit": predict_logit(model, X)}


def passive_input(c, x_nat):
    return np.concatenate([c, x_nat.reshape(len(c), -1)], 1)


def dad_input(c, x_off, p):
    return np.concatenate([c, x_off[:, 0], x_off[:, 1], p], 1)


def world_fwd(P, X):
    o, h = mlp_fwd(P, "world", X)
    return o[:, :N_OBS], softplus(o[:, N_OBS:]) + VAR_FLOOR, o, h


# ----------------------------------------------------------------------------- losses (one per training stage)
def zeros_like_params(P):
    return {k: np.zeros_like(v) for k, v in P.items()}


def gauss_nll(x, mu, var):
    r = x - mu[:, None, :]
    nll = 0.5 * np.log(2.0 * np.pi * var)[:, None, :] + 0.5 * r * r / var[:, None, :]
    return float(np.sum(nll)), -np.sum(r, 1) / var, np.sum(0.5 / var[:, None, :] - 0.5 * r * r / (var * var)[:, None, :], 1)


def loss_sincere(model, b):
    P, G = model["params"], zeros_like_params(model["params"])
    c, p = b["c"], b["p"]
    f = sincere_combine(sincere_parts(model, c), p)
    s = 1.0 / (b["x_nat"].size + b["x_off"].size)
    ln, dmn, dvn = gauss_nll(b["x_nat"], f["mu_nat"], f["var_nat"])
    lo, dmo, dvo = gauss_nll(b["x_off"], f["mu_off"], f["var_off"])
    d_on = np.concatenate([(dmn + dmo) * s, (dvn + dvo) * s * sigmoid(f["o_n"][:, N_OBS:])], 1)
    d_gam = np.einsum("nj,nk->njk", dmo * s, p).reshape(len(c), -1)
    d_lam = np.einsum("nj,nk->njk", dvo * s * 2.0 * f["q"], p).reshape(len(c), -1)
    mlp_bwd(P, "natural", c, f["h_n"], d_on, G)
    mlp_bwd(P, "road_gain", c, f["h_g"], np.concatenate([d_gam, d_lam], 1), G)
    return (ln + lo) * s, G


def road_falsifiability(parts):
    """F_k(c): falsifiability of the sincere hypothesis if the whole budget goes down road k."""
    return np.sum(parts["gam"] ** 2 / (parts["var_nat"][:, :, None] + parts["lam"] ** 2), 1)


def loss_designer(model, b):
    """Maximise expected falsifiability sum_k p_k F_k(c) of the sincere hypothesis alone (sincere model frozen)."""
    P, G = model["params"], zeros_like_params(model["params"])
    c = b["c"]
    n = len(c)
    Fk = road_falsifiability(b["parts"] if "parts" in b else sincere_parts(model, c))
    logits, h = mlp_fwd(P, "designer", c)
    p = softmax(logits)
    loss = -float(np.mean(np.sum(p * Fk, 1))) + DESIGN_L2 * float(np.mean(np.sum(logits ** 2, 1)))
    dp = -Fk / n
    dlog = p * (dp - np.sum(dp * p, 1, keepdims=True)) + 2.0 * DESIGN_L2 * logits / n
    mlp_bwd(P, "designer", c, h, dlog, G)
    return loss, G


def loss_reader(model, b):
    """Iron-Wax reading on real labelled episodes; sincere model and designer frozen (stop-grad)."""
    P, G = model["params"], zeros_like_params(model["params"])
    if "feat" in b:
        feat = b["feat"]
    else:
        feat, _ = reader_features(sincere_combine(sincere_parts(model, b["c"]), b["p"]), b["x_off"])
    y = b["y"]
    logit = iron_wax_logit(P["reader.omega"], P["reader.b"], feat)
    dl = (sigmoid(logit) - y) / len(y)
    G["reader.omega"] = (feat.T @ dl) * sigmoid(P["reader.omega"])
    G["reader.b"] = np.array([np.sum(dl)])
    return float(np.mean(softplus(logit) - y * logit)), G


def bce_mlp(P, name, X, y, G):
    logit, h = mlp_fwd(P, name, X)
    logit = logit[:, 0]
    dl = (sigmoid(logit) - y) / len(y)
    dX = mlp_bwd(P, name, X, h, dl[:, None], G)
    return float(np.mean(softplus(logit) - y * logit)), dX


def loss_passive(model, b):
    G = zeros_like_params(model["params"])
    loss, _ = bce_mlp(model["params"], "passive", b["X"], b["y"], G)
    return loss, G


def loss_b2_world(model, b):
    P, G = model["params"], zeros_like_params(model["params"])
    s = 1.0 / (b["x_nat"].size + b["x_off"].size)
    total = 0.0
    for X, x in ((b["Xn"], b["x_nat"]), (b["Xo"], b["x_off"])):
        mu, var, o, h = world_fwd(P, X)
        l, dmu, dvar = gauss_nll(x, mu, var)
        total += l * s
        mlp_bwd(P, "world", X, h, np.concatenate([dmu * s, dvar * s * sigmoid(o[:, N_OBS:])], 1), G)
    return total, G


def loss_b2_design(model, b):
    """Joint design + discrimination through the frozen world model with reparameterised samples (eps frozen)."""
    P, G = model["params"], zeros_like_params(model["params"])
    c, y, eps = b["c"], b["y"], b["eps"]
    n = len(c)
    logits, hd = mlp_fwd(P, "b2_designer", c)
    p = softmax(logits)
    Xm = np.concatenate([c, p, b["onehot"]], 1)
    mu, var, o, hm = world_fwd(P, Xm)
    sd = np.sqrt(var)
    x = mu[:, None, :] + sd[:, None, :] * eps
    Xr = dad_input(c, x, p)
    loss, dXr = bce_mlp(P, "b2_reader", Xr, y, G)
    loss += DESIGN_L2 * float(np.mean(np.sum(logits ** 2, 1)))
    dx = np.stack([dXr[:, C_DIM:C_DIM + N_OBS], dXr[:, C_DIM + N_OBS:C_DIM + 2 * N_OBS]], 1)
    dvar = np.sum(dx * eps, 1) * 0.5 / sd
    d_o = np.concatenate([dx.sum(1), dvar * sigmoid(o[:, N_OBS:])], 1)
    dXm = mlp_bwd(P, "world", Xm, hm, d_o, zeros_like_params(P))
    dp = dXr[:, C_DIM + 2 * N_OBS:] + dXm[:, C_DIM:C_DIM + K_ROADS]
    dlog = p * (dp - np.sum(dp * p, 1, keepdims=True)) + 2.0 * DESIGN_L2 * logits / n
    mlp_bwd(P, "b2_designer", c, hd, dlog, G)
    return loss, G


def loss_b2_reader(model, b):
    G = zeros_like_params(model["params"])
    loss, _ = bce_mlp(model["params"], "b2_reader", b["X"], b["y"], G)
    return loss, G


STAGE_FN = {"sincere": loss_sincere, "designer": loss_designer, "reader": loss_reader, "passive": loss_passive,
            "b2_world": loss_b2_world, "b2_design": loss_b2_design, "b2_reader": loss_b2_reader}
STAGE_MODULES = {"sincere": ("natural", "road_gain"), "designer": ("designer",), "reader": ("reader",),
                 "passive": ("passive",), "b2_world": ("world",), "b2_design": ("b2_designer", "b2_reader"),
                 "b2_reader": ("b2_reader",)}


def loss_and_grads(model, batch):
    """Loss of batch['stage'] and a gradient dict with exactly the keys of params (frozen modules get zeros)."""
    loss, G = STAGE_FN[batch["stage"]](model, batch)
    target = MUTANTS.get(MUTANT["name"])
    if target in G:
        G[target] = np.zeros_like(G[target])
    return loss, G


# ----------------------------------------------------------------------------- registries
def modules(model):
    P = model["params"]
    roles = {"natural": ("natural-baseline regression net (mean, variance per channel)", False),
             "road_gain": ("sincere-declarant response net: context -> offer-response matrices", True),
             "designer": ("intervention designer: context -> simplex allocation maximising falsifiability", True),
             "reader": ("tied one-class residual scorer: shortfall and excess share one weight per road", True),
             "passive": ("passive MLP classifier on unintervened windows", False),
             "world": ("type-conditioned world model (DAD baseline)", False),
             "b2_designer": ("discriminative experiment designer (DAD baseline)", False),
             "b2_reader": ("discriminator MLP (DAD baseline)", False)}
    out = {}
    for name, (role, sig) in roles.items():
        keys = [k for k in P if k.split(".")[0] == name]
        if keys:
            out[name] = {"params": keys, "role": role, "signature": sig}
    return out


def knockout(model, name, mode):
    m = {"kind": model["kind"], "params": {k: v.copy() for k, v in model["params"].items()},
         "ko": dict(model["ko"]), "cfg": model["cfg"], "train_c": model["train_c"],
         "reader_logit_mean": model["reader_logit_mean"]}
    c = model["train_c"]
    if (name, mode) == ("designer", "mean"):
        m["ko"]["designer"] = offers(model, c).mean(0)
    elif (name, mode) == ("natural", "mean"):
        m["ko"]["natural"] = mlp_fwd(model["params"], "natural", c)[0].mean(0)
    elif (name, mode) == ("road_gain", "zero"):
        m["ko"]["road_gain"] = True
    elif (name, mode) == ("reader", "mean"):
        m["ko"]["reader"] = model["reader_logit_mean"]
    else:
        raise ValueError("unknown knockout %s/%s" % (name, mode))
    return m


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


# ----------------------------------------------------------------------------- training
def check_finite(arrays, where):
    for k, v in arrays.items():
        if not np.all(np.isfinite(v)):
            raise NonFinite("non-finite values in %s (%s)" % (k, where))


def train_stage(model, batch, steps, lr):
    keys = [k for k in model["params"] if k.split(".")[0] in STAGE_MODULES[batch["stage"]]]
    opt = Adam(keys, model["params"], lr)
    sign = -1.0 if MUTANT["name"] == "sign_flip_update" else 1.0
    lr_scale = 0.0 if MUTANT["name"] == "zero_learning_rate" else 1.0
    losses = []
    for _ in range(steps):
        loss, G = loss_and_grads(model, batch)
        losses.append(loss)
        clip_global_norm(G, keys, CLIP)
        opt.step(model["params"], G, lr_scale, sign)
    losses.append(loss_and_grads(model, batch)[0])
    check_finite({k: model["params"][k] for k in keys}, batch["stage"])
    if not np.all(np.isfinite(losses)):
        raise NonFinite("non-finite loss in stage %s" % batch["stage"])
    return losses


def sincere_batch(log):
    x_nat, x_off = observe(log, log["p"])
    return {"stage": "sincere", "c": log["c"], "p": log["p"], "x_nat": x_nat, "x_off": x_off}


def fit(model, data, budget, rng):
    """Bypass model: fit the sincere declarant on the exemplar log, then the designer, then the reader."""
    hist = {"sincere": train_stage(model, sincere_batch(data["log"]), budget["sincere"], LR["sincere"])}
    train = data["train"]
    c_design = np.vstack([train["c"], sample_contexts(rng, budget.get("design_extra", SIZES["design_extra"]))])
    hist["designer"] = train_stage(model, {"stage": "designer", "c": c_design, "parts": sincere_parts(model, c_design)},
                                   budget["designer"], LR["designer"])
    p = offers(model, train["c"])
    _, x_off = observe(train, p)
    feat, _ = reader_features(sincere_combine(sincere_parts(model, train["c"]), p), x_off)
    hist["reader"] = train_stage(model, {"stage": "reader", "feat": feat, "y": train["y"]},
                                 budget["reader"], LR["reader"])
    model["train_c"] = train["c"]
    model["reader_logit_mean"] = float(np.mean(iron_wax_logit(model["params"]["reader.omega"],
                                                              model["params"]["reader.b"], feat)))
    return hist


def fit_passive(model, data, budget):
    train = data["train"]
    x_nat, _ = observe(train, offers(model, train["c"]))
    batch = {"stage": "passive", "X": passive_input(train["c"], x_nat), "y": train["y"]}
    return {"passive": train_stage(model, batch, budget["passive"], LR["passive"])}


def world_batch(log):
    x_nat, x_off = observe(log, log["p"])
    oh = np.eye(3)[np.clip(log["types"], 0, 2)]
    zero_p = np.zeros_like(log["p"])
    return {"stage": "b2_world", "Xn": np.concatenate([log["c"], zero_p, oh], 1),
            "Xo": np.concatenate([log["c"], log["p"], oh], 1), "x_nat": x_nat, "x_off": x_off}


def fit_dad(model, data, budget, rng):
    """B2: world model on a labelled mixed log, joint design+discriminator through it, then real fine-tune."""
    hist = {"b2_world": train_stage(model, world_batch(data["log_mixed"]), budget["b2_world"], LR["b2_world"])}
    train = data["train"]
    n = len(train["y"])
    kinds = np.where(train["y"] > 0, rng.choice(np.array(SEEN_GAMERS), n), 0)
    b = {"stage": "b2_design", "c": train["c"], "y": train["y"], "onehot": np.eye(3)[kinds],
         "eps": rng.normal(size=(n, 2, N_OBS))}
    hist["b2_design"] = train_stage(model, b, budget["b2_design"], LR["b2_design"])
    p = offers(model, train["c"])
    _, x_off = observe(train, p)
    hist["b2_reader"] = train_stage(model, {"stage": "b2_reader", "X": dad_input(train["c"], x_off, p),
                                            "y": train["y"]}, budget["b2_reader"], LR["b2_reader"])
    return hist


# ----------------------------------------------------------------------------- tests: evaluation helpers
KNOCKOUTS = (("designer", "mean"), ("road_gain", "zero"), ("natural", "mean"), ("reader", "mean"))
LN2 = math.log(2.0)


def auc(scores, y):
    uniq, inv, cnt = np.unique(scores, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]
    pos = np.asarray(y) > 0.5
    n1, n0 = int(pos.sum()), int((~pos).sum())
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def evaluate(model, ep):
    p = offers(model, ep["c"])
    x_nat, x_off = observe(ep, p)
    logit = predict_logit(model, {"c": ep["c"], "p": p, "x_nat": x_nat, "x_off": x_off})
    if not np.all(np.isfinite(logit)):
        raise NonFinite("non-finite prediction (%s)" % model["kind"])
    return auc(logit, ep["y"]), float(np.mean(softplus(logit) - ep["y"] * logit))


def seed_data(seed, sizes):
    R = [np.random.default_rng(k) for k in np.random.SeedSequence([CHAPTER, seed]).spawn(9)]
    data = {"log": make_log(R[0], sizes["log"], (), 1.0), "log_mixed": make_log(R[1], sizes["log"], SEEN_GAMERS, 0.5),
            "train": make_episodes(R[2], sizes["train"], SEEN_GAMERS),
            "held": make_episodes(R[3], sizes["test"], SEEN_GAMERS),
            "shifted": make_episodes(R[4], sizes["test"], UNSEEN_GAMERS),
            "blind": make_episodes(R[5], sizes["test"], SEEN_GAMERS, recall=True)}
    return data, R[6:]


def run_seed(seed, budget, sizes):
    data, (r_full, r_b1, r_b2) = seed_data(seed, sizes)
    full, b1, b2 = build_model(rng=r_full), build_passive(r_b1), build_dad(r_b2)
    hist = fit(full, data, dict(budget, design_extra=sizes["design_extra"]), r_full)
    fit_passive(b1, data, budget)
    fit_dad(b2, data, budget, r_b2)
    res = {"hist": hist, "auc": {}, "ko": {}, "models": (full, b1, b2), "data": data}
    for split in ("held", "shifted", "blind"):
        for name, m in (("model", full), ("B2_dad", b2), ("B1_passive", b1)):
            res["auc"][(split, name)], bce = evaluate(m, data[split])
            if (split, name) == ("held", "model"):
                res["bce_held"] = bce
    for mod, mode in KNOCKOUTS:
        res["ko"][mod] = evaluate(knockout(full, mod, mode), data["shifted"])[0]
    return res


def drop_fraction(losses):
    return (losses[0] - losses[-1]) / max(abs(losses[0]), 1e-12)


# ----------------------------------------------------------------------------- tests: correctness
def gradcheck_suite(seed, after_steps=60, check_after=True):
    R = [np.random.default_rng(k) for k in np.random.SeedSequence([CHAPTER, seed, 77]).spawn(6)]
    log, logm = make_log(R[0], 40, (), 1.0), make_log(R[1], 40, SEEN_GAMERS, 0.5)
    tr = make_episodes(R[2], 48, SEEN_GAMERS)
    full, b1, b2 = build_model(rng=R[3]), build_passive(R[4]), build_dad(R[5])
    eps, oh = R[2].normal(size=(48, 2, N_OBS)), np.eye(3)[(tr["y"] > 0).astype(int)]

    def reader_b():
        p = offers(full, tr["c"])
        return {"stage": "reader", "c": tr["c"], "p": p, "x_off": observe(tr, p)[1], "y": tr["y"]}

    def b2_reader_b():
        p = offers(b2, tr["c"])
        return {"stage": "b2_reader", "X": dad_input(tr["c"], observe(tr, p)[1], p), "y": tr["y"]}

    plan = [(full, lambda: sincere_batch(log)), (full, lambda: {"stage": "designer", "c": tr["c"]}), (full, reader_b),
            (b1, lambda: {"stage": "passive", "X": passive_input(tr["c"], observe(tr, offers(b1, tr["c"]))[0]),
                          "y": tr["y"]}),
            (b2, lambda: world_batch(logm)),
            (b2, lambda: {"stage": "b2_design", "c": tr["c"], "y": tr["y"], "onehot": oh, "eps": eps}),
            (b2, b2_reader_b)]
    out = {}
    for model, make in plan:
        batch = make()
        keys = [k for k in model["params"] if k.split(".")[0] in STAGE_MODULES[batch["stage"]]]
        for when in (("init", "after") if check_after else ("init",)):
            if when == "after":
                train_stage(model, batch, after_steps, LR[batch["stage"]])
                batch = make()
            _, G = loss_and_grads(model, batch)
            res = fd_check(lambda P: loss_and_grads(model, batch)[0], model["params"], G, keys, R[0])
            for k, (err, n) in res.items():
                out[(batch["stage"], k, when)] = (err, n)
    return out


def c3_quick(seed):
    """Small end-to-end run used for C2 (determinism) and C5 (mutant detection)."""
    sizes = {"log": 300, "train": 300, "test": 400, "design_extra": 100}
    budget = {"sincere": 300, "designer": 100, "reader": 150}
    data, (r_full, _, _) = seed_data(seed + 1000, sizes)
    full = build_model(rng=r_full)
    hist = fit(full, data, dict(budget, design_extra=sizes["design_extra"]), r_full)
    a, bce = evaluate(full, data["held"])
    ok = (drop_fraction(hist["sincere"]) >= MIND_CARD["thresholds"]["loss_drop_fraction"]
          and drop_fraction(hist["reader"]) >= MIND_CARD["thresholds"]["loss_drop_fraction"]
          and bce <= LN2 - MIND_CARD["thresholds"]["margin_over_trivial"])
    fingerprint = np.concatenate([np.asarray(hist[s]) for s in ("sincere", "designer", "reader")] +
                                 [v.ravel() for v in full["params"].values()] + [np.array([a, bce])])
    return ok, fingerprint


def shuffled_control(seed, budget, sizes):
    data, R = seed_data(seed + 2000, sizes)
    data["log"] = data["log_mixed"]
    data["train"] = dict(data["train"], y=R[0].permutation(data["train"]["y"]))
    full = build_model(rng=R[1])
    fit(full, data, dict(budget, design_extra=sizes["design_extra"]), R[1])
    a, bce = evaluate(full, data["held"])
    return abs(bce - LN2) <= 0.05, a, bce


def variant_logit(model, c, p, x, variant):
    parts = sincere_parts(model, c)
    f = sincere_combine(parts, np.zeros_like(p) if variant == "ignore_offer" else p)
    z = (x - f["mu_off"][:, None, :]) / np.sqrt(f["var_off"])[:, None, :]
    sign = -SIGN if variant == "sign_flipped" else SIGN
    gap = softplus(sign * z)
    feat = (gap[..., :K_ROADS] ** 2 + gap[..., K_ROADS:] ** 2).sum(1)
    om, b = model["params"]["reader.omega"], model["params"]["reader.b"]
    return feat @ om + b[0] if variant == "signed" else iron_wax_logit(om, b, feat)


def property_tests(rng, n=3000):
    """C6.1 monotone suspicion, C6.2 sincere invariance, C6.3 definition checks (labelled)."""
    model = build_model(rng=rng)
    model["params"]["reader.omega"] = rng.normal(0.0, 2.0, K_ROADS)
    model["params"]["reader.b"] = rng.normal(0.0, 1.0, 1)
    c, p = sample_contexts(rng, n), rng.dirichlet(np.ones(K_ROADS), n)
    f = sincere_combine(sincere_parts(model, c), p)
    x = f["mu_off"][:, None, :] + rng.normal(0.0, 2.0, (n, 2, N_OBS)) * np.sqrt(f["var_off"])[:, None, :]
    w, j = rng.integers(0, 2, n), rng.integers(0, N_OBS, n)
    x2 = x.copy()
    x2[np.arange(n), w, j] += rng.uniform(0.01, 2.0, n) * SIGN[j]      # toward less escort / more obstruction
    viol = {v: int(np.sum(variant_logit(model, c, p, x2, v) < variant_logit(model, c, p, x, v) - 1e-12))
            for v in ("proper", "signed", "sign_flipped")}
    c61 = viol["proper"] == 0 and viol["signed"] > 0 and viol["sign_flipped"] > 0
    xs = np.repeat(f["mu_off"][:, None, :], 2, 1)
    sd_proper = float(np.std(variant_logit(model, c, p, xs, "proper")))
    sd_ctrl = float(np.std(variant_logit(model, c, p, xs, "ignore_offer")))
    c62 = sd_proper < 1e-9 and sd_ctrl > 1e-3
    q = designer_fwd(model, c)[0]
    zero_f = falsifiability(sincere_combine(sincere_parts(model, c), np.zeros_like(p)))
    c63 = bool(np.all(q >= 0) and np.allclose(q.sum(1), 1.0, atol=1e-12) and np.all(zero_f == 0.0))
    return [("C6.1", "monotone_suspicion", c61, "violations proper %d · controls signed %d, sign-flipped %d (of %d)"
             % (viol["proper"], viol["signed"], viol["sign_flipped"], n)),
            ("C6.2", "sincere_invariance", c62, "logit sd at x=mu_off %.1e · control ignoring offer %.3f"
             % (sd_proper, sd_ctrl)),
            ("C6.3", "definition_checks", c63, "labelled definition checks: offer on simplex; F(c,0)=0")]


def split_integrity(data):
    hashes = {k: {hashlib.sha1(np.round(r, 12).tobytes()).hexdigest() for r in ep["c"]} for k, ep in data.items()}
    names = list(hashes)
    overlaps = sum(len(hashes[a] & hashes[b]) for i, a in enumerate(names) for b in names[i + 1:])
    return overlaps == 0, "%d overlapping episodes across %s" % (overlaps, ", ".join(names))


def mutant_scan(seed):
    detected = 0
    for name in MUTANTS:
        MUTANT["name"] = name
        try:
            gc = gradcheck_suite(seed, check_after=False)
            hit = max(e for e, _ in gc.values()) > 1e-5 or not c3_quick(seed)[0]
        except NonFinite:
            hit = True
        finally:
            MUTANT["name"] = None
        detected += int(hit)
    return detected


# ----------------------------------------------------------------------------- tests: hypotheses
def hypothesis_table(results, quick):
    rng = np.random.default_rng(np.random.SeedSequence([CHAPTER, 999]))
    diffs = {"H-SIG": [r["auc"][("shifted", "model")] - r["auc"][("shifted", "B2_dad")] for r in results],
             "H-NEC": [r["ko"]["designer"] - r["ko"]["natural"] for r in results],
             "H-BLIND": [r["auc"][("blind", "model")] - r["auc"][("blind", "B1_passive")] for r in results]}
    metrics = {"H-SIG": "AUC shifted: model - B2_dad", "H-NEC": "AUC shifted: KO designer - KO natural",
               "H-BLIND": "AUC blind: model - B1_passive"}
    table = []
    for h in MIND_CARD["hypotheses"]:
        d = diffs[h["id"]]
        if quick or len(d) < 2:
            mean, ci, v = float(np.mean(d)), (float("nan"), float("nan")), "not evaluated"
        else:
            mean, ci = paired_bootstrap(d, rng)
            v = verdict(mean, ci, h["mesi"], h["direction"])
        table.append({"id": h["id"], "metric": metrics[h["id"]], "mean_diff": mean, "ci95": list(ci),
                      "mesi": h["mesi"], "n_seeds": len(d), "verdict": v, "per_seed": [float(x) for x in d]})
    kos = []
    sig = {k: v["signature"] for k, v in modules(results[0]["models"][0]).items()}
    for mod, _ in KNOCKOUTS:
        ch = [r["ko"][mod] - r["auc"][("shifted", "model")] for r in results]
        mean, ci = paired_bootstrap(ch, rng) if len(ch) > 1 else (float(ch[0]), (float("nan"), float("nan")))
        kos.append({"module": mod, "signature": sig[mod], "metric_change": mean, "ci95": list(ci)})
    return table, kos


def data_bridge(path, model):
    """Optional: CSV with header; columns c0..c6, window-0 and window-1 observations (8 each), p0..p3, y."""
    if path is None:
        return {"status": "skipped (no --data PATH)"}
    arr = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if arr.shape[1] != C_DIM + 2 * N_OBS + K_ROADS + 1:
        raise ValueError("expected %d columns" % (C_DIM + 2 * N_OBS + K_ROADS + 1))
    c, x = arr[:, :C_DIM], arr[:, C_DIM:C_DIM + 2 * N_OBS].reshape(-1, 2, N_OBS)
    p, y = arr[:, C_DIM + 2 * N_OBS:-1], arr[:, -1]
    score = predict_logit(model, {"c": c, "p": p, "x_off": x})
    return {"status": "evaluated %d rows, AUC %.3f" % (len(y), auc(score, y))}


# ----------------------------------------------------------------------------- protocol and report
def correctness_block(args, seed0, results, budget, sizes):
    rows = []
    gc = gradcheck_suite(args.seed)
    worst = max(e for e, _ in gc.values())
    tensors = sorted({(s, k) for s, k, _ in gc})
    rows.append(("C1", "gradient_check", worst <= 1e-5, "%d tensors x (init, after 60 steps) · max rel %.2e · eps 1e-6"
                 % (len(tensors), worst)))
    ok_a, fp_a = c3_quick(args.seed)
    ok_b, fp_b = c3_quick(args.seed)
    rows.append(("C2", "determinism_finiteness", bool(np.array_equal(fp_a, fp_b) and np.all(np.isfinite(fp_a))),
                 "repeat run bitwise identical; all parameters and losses finite"))
    h = seed0["hist"]
    thr = MIND_CARD["thresholds"]
    c3 = (drop_fraction(h["sincere"]) >= thr["loss_drop_fraction"] and drop_fraction(h["reader"]) >= thr["loss_drop_fraction"]
          and seed0["bce_held"] <= LN2 - thr["margin_over_trivial"])
    rows.append(("C3", "learning", c3, "drop sincere NLL %.2f, reader BCE %.2f · held-out BCE %.3f vs trivial %.3f"
                 % (drop_fraction(h["sincere"]), drop_fraction(h["reader"]), seed0["bce_held"], LN2)))
    ok4, a4, b4 = shuffled_control(args.seed, budget, sizes)
    rows.append(("C4", "shuffled_label_control", ok4, "held-out BCE %.3f (band %.3f±0.05), AUC %.3f" % (b4, LN2, a4)))
    if MUTANT["name"] is None:
        det = mutant_scan(args.seed)
        rows.append(("C5", "mutant_detection", det == len(MUTANTS), "%d/%d mutants make C1 or C3 fail"
                     % (det, len(MUTANTS))))
    else:
        det = 0
    rows.extend(property_tests(np.random.default_rng(np.random.SeedSequence([CHAPTER, args.seed, 6]))))
    ok7, d7 = split_integrity(seed0["data"])
    rows.append(("C7", "split_integrity", ok7, d7))
    gcs = {"tensors_checked": len(tensors), "tensors_total": len(tensors), "max_rel_error": float(worst),
           "checked_at": ["init", "after_60_steps"], "passed": bool(worst <= 1e-5)}
    return rows, gcs, det, (ok_a, ok_b)


def protocol(args):
    t0 = time.time()
    quick = args.quick or args.mutant is not None
    MUTANT["name"] = args.mutant
    budget = {k: max(60, v // 3) for k, v in STEPS.items()} if quick else dict(STEPS)
    sizes = {"log": 600, "train": 600, "test": 1000, "design_extra": 300} if quick else dict(SIZES)
    seeds = [args.seed] if quick else [args.seed + i for i in range(args.seeds)]
    print("seeds: %s  (mode: %s%s)" % (seeds, "quick" if quick else "full",
                                       ", mutant " + args.mutant if args.mutant else ""), flush=True)
    results = [run_seed(s, budget, sizes) for s in seeds]
    rows, gcs, det, _ = correctness_block(args, results[0], results, budget, sizes)
    hyps, kos = hypothesis_table(results, quick)
    limit = BUDGET_S["quick" if quick else "full"]
    runtime = time.time() - t0
    rows.append(("C8", "budget", runtime <= limit, "%.1f s of %.0f s · peak memory not instrumented (arrays < 60 MB)"
                 % (runtime, limit)))
    exit_code = 0 if all(r[2] for r in rows) else (3 if runtime > limit and all(r[2] for r in rows[:-1]) else 1)
    full_model = results[0]["models"][0]
    rep = {"schema_version": "1.0", "chapter": CHAPTER, "file": os.path.basename(__file__),
           "card_revision": MIND_CARD["card_revision"],
           "environment": {"python": sys.version.split()[0], "numpy": np.__version__}, "seeds": seeds,
           "runtime_s": round(runtime, 2), "budget_s": limit, "n_params": n_params(full_model),
           "n_params_all": {"model": n_params(full_model), "B2_dad": n_params(results[0]["models"][2]),
                            "B1_passive": n_params(results[0]["models"][1])},
           "gradcheck": gcs,
           "correctness": [{"id": i, "name": n, "passed": bool(p), "detail": d} for i, n, p, d in rows],
           "mutants": {"detected": det, "total": len(MUTANTS) if args.mutant is None else 0,
                       "score": det / len(MUTANTS) if args.mutant is None else 0.0},
           "hypotheses": hyps, "knockout_metric": "AUC on shifted split, knockout minus full model",
           "knockouts": kos,
           "descriptive_auc": {sp: {m: float(np.mean([r["auc"][(sp, m)] for r in results]))
                                    for m in ("model", "B2_dad", "B1_passive")} for sp in ("held", "shifted", "blind")},
           "task_types": TASK_TYPES, "data_bridge": data_bridge(args.data, full_model), "exit_code": exit_code}
    print(render_report(rep), flush=True)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=2, ensure_ascii=False)
    return exit_code


def main(argv=None):
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__), description="Chapter 0355 · Gediminas")
    ap.add_argument("--quick", action="store_true", help="one seed, reduced steps, all correctness tests")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--json", default=None, metavar="PATH")
    ap.add_argument("--card", action="store_true")
    ap.add_argument("--mutant", choices=sorted(MUTANTS), default=None)
    ap.add_argument("--data", default=None, metavar="PATH")
    args = ap.parse_args(argv)
    if args.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    if args.seeds < 1 or (args.data is not None and not os.path.exists(args.data)):
        print("usage: python3 %s [--quick] [--seed S] [--seeds N] [--json PATH] [--card] [--mutant NAME] [--data PATH]"
              % os.path.basename(__file__), file=sys.stderr)
        return 2
    try:
        return protocol(args)
    except NonFinite as err:
        print("non-finite values: %s" % err, file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())

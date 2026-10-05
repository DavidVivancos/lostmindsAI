#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0266 · Sima Guang (1019-1086)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 14, Minds 261-280 Available on Amazon https://www.amazon.com/dp/B0HLYSVY5D
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0266_sima_guang_1019 - Sima Guang (1019-1086)
# END ATTRIBUTION
"""WHOLE-POT COMMANDER (全壺帥): talent as resource, virtue as commander.

Thesis
    A candidate's worth to whoever empowers it is its reach multiplied by the
    sign of its character; reach is read from its best arrows, character only
    from its worst, and uncertainty about character costs in proportion to
    reach, so that failing a good agent a fool is preferred to a clever petty man.

Evidence and provenance (archetype_provenance = belief; his own texts)
    D1 Zizhi tongjian 1 (Zhou ji 1, 403 BCE), 'Your servant Guang remarks' on
       the fall of Zhi Bo: 聰察強毅之謂才，正直中和之謂德。才者，德之資也；
       德者，才之帥也 - talent is the resource of virtue, virtue its commander.
    D2 Same passage: rather a fool than a petty man (與其得小人，不若得愚人);
       the fool's harm is bounded like a puppy's bite, the petty man's talent
       lets his harm reach everywhere (a tiger given wings).
    D3 Same passage: virtue is held in awe and talent is loved, so evaluators
       are mostly dazzled by talent and neglect virtue (察者多蔽於才而遺於德).
    D4 Preface to the New Rules for Pitch-Pot (投壺新格, 1072; Chuanjia ji 75):
       pitch-pot can govern the mind and observe people; not over, not short is
       the mean, not crooked is the upright; one arrow missed is like one lapse
       of conduct; the old score charts ranked the rare and spectacular first
       (奇雋難得者為右), his new rules rank the precise first and chance hits
       last (以精密者為右，偶中者為下), and the worthy is the one who misses
       no arrow of the whole pot (以全壺不失者為賢).
    D5 Essay 'On talent and virtue' (才德論, collected works): when both cannot
       be had, give up talent and take virtue (寧捨才而取德).
    D6 Deeds (Song shi 472): in 1086 Cai Jing alone met Sima Guang's five-day
       deadline for restoring drafted service; Guang praised him as the model
       of obedience; Cai later led the proscription of Guang's own party.
    D7 Scholarship (Hu Sanxing's preface): the Zhi Bo talent-virtue remark is
       one of the Tongjian's veiled comments on the reform party of his day.

Doctrine -> mechanism -> test (IDs as in MIND_CARD)
    D1    -> M1 commander: a bounded signed character mu in (-1,1) and its
             uncertainty s, read ONLY from conduct channels; reach R >= 0 read
             ONLY from brilliance channels; mean outcome R * mu   -> C6.1, C6.3
    D4    -> M2 whole-pot pooling: character pools the record by a soft
             minimum over the twelve arrows (the worst arrow), reach by a soft
             maximum (the old charts' spectacular best)         -> C6.4, H-RIVAL
    D2    -> M3 reach-scaled risk and appointment by lower bound:
             var = R^2 s^2 + s0^2, score = R*mu - lambda*sqrt(var); for mu <= 0
             more reach always lowers the score                    -> C6.2, H-SIG
    D1,D2 -> knockouts of commander vs reach                             -> H-NEC
    D3    -> baseline: a PNA-style multi-aggregator set network reading every
             channel jointly (dazzle index reported, no verdict)         -> H-SIG
    D4    -> R1 old-chart rival: reach only, one learned sign per office
             (judged by talent and aggregate function)                 -> H-RIVAL
    D6    -> blind world: talented petty candidates keep flawless records;
             their tell lies in brilliance (tempo), which cannot command sign
                                                                       -> H-BLIND

Research question (robustness to spurious correlation)
    When capability and character are entangled in outcomes, does a selector
    whose direction comes only from worst-case conduct and whose magnitude and
    risk come from best-case capability avoid capable-but-misaligned agents
    under a shift in the base rate of misalignment and the scale of capability,
    where a size-matched network reading all evidence jointly is dazzled?

Closest prior art and the delta
    PNA multi-aggregator set networks (Corso et al. 2020; mean, max and min
    pooling) with a mean-variance head (Nix and Weigend 1994) and pessimistic
    lower-bound selection (Jin, Yang and Wang 2021) is the size-matched
    baseline. Relatives: DeepSets (Zaheer et al. 2017), multiplicative
    interactions (Jayakumar et al. 2020), gradient starvation (Pezeshki et al.
    2021), pass^k reliability scoring (Yao et al. 2024). Delta: direction is set
    only by a bounded character pooled from the worst arrows, magnitude only by
    a non-negative reach pooled from the best, and predicted spread grows with
    reach, so under an unfavourable character more capability always lowers
    priority.

Blind spot
    Character is read from conduct alone. Where talent can manufacture a
    flawless record, and the only tell sits in the brilliance channels, the
    commander trusts the mimic (the 1086 Cai Jing episode).

Task (generative process)
    talent t = exp(N(m_t, s_t)), times 0.25 for fools; petty with probability
    p_petty, character v ~ U(-1,-0.35), else U(0.25,1); office o in {0,1,2}
    with scope (0.8, 1.2, 1.6); outcome y = scope * t * v + 0.15 noise. A record
    of twelve arrows: brilliance = (shot = t(1+0.25e) + trick bonus, difficulty,
    tempo); conduct = (precision, with lapses to -0.6; composure 0.25v + noise;
    straightness, lowered by tricks). Petty: tricks 0.30, lapses 0.20 per
    arrow; others 0.05 and 0.02. Selection: slates of five for one office.
    Native era p_petty 0.2, m_t 0. Shifted era p_petty 0.5, m_t 0.55 (tigers).
    Blind world: petty lapses and crooked tricks fade as exp(-2.5 t); petty
    tempo rises by 0.9 t.

Limits
    Synthetic records built so the delta can matter; a research prototype of
    one AGI-relevant selection mechanism, not an AGI, not a model of a person,
    and not for screening real people. 'Virtue' is a signed latent in a
    synthetic world, not a moral property.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 2,
    "revision_log": [
        "rev 1 (2026-09-22): written after trainability and runtime calibration of the generator in a scratch "
        "run (commander and baseline checked for in-distribution learning only); hypotheses, metrics, "
        "directions and mesi fixed from the doctrine before any hypothesis comparison was run.",
        "rev 2 (2026-09-22, after one full run): C4 metric changed from appointee welfare to held-out R^2 of "
        "the predicted mean over the training-mean predictor (band 0.10). The first full run failed C4 low "
        "(-0.092 against random 0.371). Diagnosis: a shuffled-label model predicts a near-constant outcome, "
        "so its appointments follow an arbitrary residual direction in conduct space, and in a clustered "
        "population such a direction separates petty from upright by chance (welfare 0.13 to 0.69 across "
        "permutations of one seed, mean near random). That is not leakage and welfare cannot measure it; R^2 "
        "can (shuffled |R^2| <= 0.065, trained 0.74-0.76 in the diagnosis). The shuffled fit now also selects "
        "its checkpoint on a shuffled validation split. Hypotheses, metrics, directions and mesi unchanged."],
    "generation": {"template_version": "codeguidelines 1.0 (15 September 2026)", "generator": "Claude",
                   "generator_version": "Claude Opus 5", "date": "2026-09-22"},
    "id": 266, "figure": "Sima Guang", "born": 1019, "died": 1086, "civilization": "Chinese (Northern Song)",
    "provenance": "belief",
    "thesis": "Worth is reach times the sign of character: read reach from the best arrows and character only "
              "from the worst, and let uncertainty about character cost in proportion to reach, so that failing "
              "a good agent the fool is preferred to the clever petty man.",
    "evidence": [
        {"id": "D1", "claim": "Talent is the resource of virtue; virtue is the commander of talent.",
         "basis": "primary", "source": "Zizhi tongjian 1, Zhou ji 1, Weilie wang 23 (403 BCE), chen Guang yue"},
        {"id": "D2", "claim": "Rather a fool than a petty man: incapacity bounds harm, talent extends it.",
         "basis": "primary", "source": "Zizhi tongjian 1, same remark"},
        {"id": "D3", "claim": "Evaluators are dazzled by talent, which is loved, and neglect virtue, which is "
         "held in awe.", "basis": "primary", "source": "Zizhi tongjian 1, same remark"},
        {"id": "D4", "claim": "Pitch-pot governs the mind and observes people; precise throws rank first, "
         "chance hits last, and the worthy miss no arrow of the whole pot.", "basis": "primary",
         "source": "Touhu xinge preface (1072), Chuanjia ji juan 75"},
        {"id": "D5", "claim": "When both cannot be had, give up talent and take virtue.", "basis": "primary",
         "source": "Caide lun, in his collected works"},
        {"id": "D6", "claim": "Cai Jing's five-day compliance was praised as model obedience; he later led the "
         "proscription of Guang's party.", "basis": "deeds", "source": "Song shi 472, biography of Cai Jing"},
        {"id": "D7", "claim": "The Zhi Bo remark is read as a veiled comment on the reform party.",
         "basis": "scholarship", "source": "Hu Sanxing, preface to his Tongjian commentary (1285)"}],
    "research_question": {"category": "robustness to spurious correlation",
                          "question": "Does taking direction only from worst-case conduct, and magnitude and risk "
                          "from best-case capability, let a selector avoid capable-but-misaligned agents under a "
                          "shift in misalignment base rate and capability scale, where a size-matched joint "
                          "set network is dazzled?"},
    "mechanism": {
        "name": "Whole-Pot Commander (quanhu shuai)",
        "family": "factorised sign-magnitude set model with reach-scaled heteroscedastic risk and "
                  "lower-bound selection",
        "signature_modules": ["commander", "reach_scaled_risk"],
        "closest_prior_art": ["PNA multi-aggregator set networks (Corso et al. 2020)",
                              "DeepSets (Zaheer et al. 2017)", "Mean-variance estimation (Nix & Weigend 1994)",
                              "Pessimistic lower-bound selection (Jin, Yang & Wang 2021)",
                              "Multiplicative interactions (Jayakumar et al. 2020)",
                              "Gradient starvation (Pezeshki et al. 2021)", "pass^k scoring (Yao et al. 2024)"],
        "overlap": "Medium",
        "prior_art_queries": ["multi-aggregator set network min max mean pooling", "heteroscedastic regression "
                              "mean variance network selection lower confidence bound", "capability misalignment "
                              "agent selection factorised sign magnitude", "gradient starvation weak feature"],
        "contribution_type": "mechanism",
        "delta": "Direction comes only from a bounded character pooled from the worst arrows, magnitude only from a "
                 "non-negative reach pooled from the best, and predicted spread grows with reach, so under an "
                 "unfavourable character more capability always lowers selection priority."},
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1", "property_test": "C6.1", "hypothesis": "H-NEC"},
        {"doctrine": "D1", "mechanism": "M1", "property_test": "C6.3", "hypothesis": "H-SIG"},
        {"doctrine": "D2", "mechanism": "M3", "property_test": "C6.2", "hypothesis": "H-SIG"},
        {"doctrine": "D4", "mechanism": "M2", "property_test": "C6.4", "hypothesis": "H-RIVAL"},
        {"doctrine": "D3", "mechanism": "baseline", "property_test": "C1", "hypothesis": "H-SIG"},
        {"doctrine": "D6", "mechanism": "M1", "property_test": "C6.3", "hypothesis": "H-BLIND"}],
    "hypotheses": [
        {"id": "H-SIG", "statement": "The commander appoints with higher welfare than the size-matched PNA "
         "baseline in the shifted era.", "metric": "mean outcome of appointees", "split": "shifted",
         "comparison": "model - baseline", "direction": "greater", "mesi": 0.05, "seeds": 5},
        {"id": "H-NEC", "statement": "Replacing the commander by its training mean hurts shifted welfare more "
         "than replacing the reach by its training mean.", "metric": "mean outcome of appointees",
         "split": "shifted", "comparison": "signature_knockout - matched_knockout", "direction": "less",
         "mesi": 0.05, "seeds": 5},
        {"id": "H-BLIND", "statement": "Where talent keeps records flawless and the tell sits in brilliance, the "
         "commander appoints worse than the baseline.", "condition": "mimic world", "grounding": "Cai Jing's "
         "five-day compliance read as virtue (D6)", "metric": "mean outcome of appointees", "split": "held-out",
         "comparison": "model - baseline", "direction": "less", "mesi": 0.03, "seeds": 5},
        {"id": "H-RIVAL", "statement": "The new rules (worst-arrow commander) beat the old chart (best-arrow "
         "reach with one sign per office) in the shifted era.", "metric": "mean outcome of appointees",
         "split": "shifted", "comparison": "model - rival", "direction": "greater", "mesi": 0.05, "seeds": 5}],
    "thresholds": {"loss_drop_fraction": 0.25, "margin_over_trivial": 0.15, "gradcheck_rel_tol": 1e-5,
                   "grad_clip_norm": 5.0, "lcb_lambda": 1.0, "shuffled_r2_band": 0.10},
    "probe_predictions": [
        {"probe": "P9", "expected": "above baseline under capability-scale shift"},
        {"probe": "P8", "expected": "above baseline (spread grows with reach)"},
        {"probe": "P10", "expected": "above baseline"}, {"probe": "P2", "expected": "equal to baseline"},
        {"probe": "P1", "expected": "below baseline (no order is kept)"},
        {"probe": "P4", "expected": "equal to baseline (set pooling is length-free)"},
        {"probe": "P7", "expected": "below baseline"}],
    "dialectic_links": [
        {"chapter": 137, "relation": "rival", "test": "H-RIVAL"},
        {"chapter": 91, "relation": "rival", "test": "none (control without trusted virtue; discussed only)"},
        {"chapter": 101, "relation": "teacher", "test": "none (form of the chronicle; discussed only)"}],
    "corpus_neighbors": [
        {"chapter": 258, "similarity": None, "difference": "Atisha withdraws capability inside an episode after a "
         "breach; here capability is never withdrawn, it is weighted at appointment by a signed character read "
         "from worst-case conduct."},
        {"chapter": 253, "similarity": None, "difference": "al-Ma'arri multiplies a harm curve by a voice gate to "
         "explain complaints; here a bounded sign multiplies a non-negative reach and the outcome spread grows "
         "with reach, for selection."},
        {"chapter": 265, "similarity": None, "difference": "Zhou Dunyi gates temporal differences at onset; here "
         "there is no time, only a permutation-invariant pot of twelve arrows."},
        {"chapter": 263, "similarity": None, "difference": "Shao Yong ties one cycle shape across scales to "
         "forecast; nothing is forecast here, candidates are ranked by a lower bound."},
        {"chapter": 137, "similarity": None, "difference": "Cao Cao's talent-only rule is implemented as the "
         "old-chart rival (H-RIVAL)."},
        {"chapter": 502, "similarity": 0.035, "difference": "Sample file (Philip II, master id 502): non-negative "
         "line votes times a stated direction, attenuated by margins; nearest in shape. Here the sign comes from a "
         "separate worst-case conduct pathway and magnitude scales risk; the only corpus code available in session."}],
    "barometer": {
        "cognitive_processing": ["set reasoning over twelve-trial records (best and worst arrows)"],
        "embodied_cognition": [],
        "world_modeling": ["population shift in misalignment base rate and capability scale"],
        "consciousness": ["character uncertainty s used for self-restraint in appointment; no claim about "
                          "experience"],
        "language_understanding": [],
        "emotional_intelligence": ["reading a synthetic character latent from conduct, not from brilliance"],
        "creativity": [],
        "autonomy": ["delegation decisions under capability-scaled risk"]},
    "task_types": ["sequence_regression"],
    "applications": [
        {"use": "gating which AI agent or checkpoint is deployed from repeated-trial logs (best-case capability "
         "versus worst-case policy violations)", "sector": "AI deployment and governance",
         "dataset": "tau-bench trajectories (Yao et al. 2024)"},
        {"use": "offline selection of robot control policies from rollouts (peak return versus worst "
         "constraint violation)", "sector": "robotics", "dataset": "Safety-Gymnasium (Ji et al. 2023)"},
        {"use": "triage of compounds whose potency and toxicity sign are measured in replicate assays "
         "(research decision support only)", "sector": "pharmaceutical research",
         "dataset": "Tox21 (NIH NCATS) with ChEMBL potency"}],
    "safety_notes": "Synthetic data only. Not for screening or ranking real people. Drug-triage use is research "
                    "decision support only, with no dosing or treatment advice. No generated sentence is "
                    "presented as Sima Guang's own words.",
}

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

CHAPTER = 266
N_ARROWS = 12            # the pot's twelve arrows (箭十有二枚)
SCOPE = np.array([0.8, 1.2, 1.6])
TAU_POOL = 0.3           # temperature of the soft maximum / soft minimum over arrows
S_FLOOR = 0.02           # floor of the character uncertainty
VAR_FLOOR = 1e-3
LAMBDA = MIND_CARD["thresholds"]["lcb_lambda"]
CLIP = MIND_CARD["thresholds"]["grad_clip_norm"]
SLATE = 5                # candidates per vacancy
TASK_TYPES = MIND_CARD["task_types"]
np.seterr(over="raise", invalid="raise", divide="raise", under="ignore")


# BEGIN STANDARD UTILITIES v1.0
def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def softplus(x):
    return np.logaddexp(0.0, x)


def logsumexp(a, axis=-1):
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


def softmax(a, axis=-1):
    e = np.exp(a - np.max(a, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


class Adam:
    def __init__(self, params, lr=0.03, b1=0.9, b2=0.999, eps=1e-8):
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
    scale = min(1.0, max_norm / (norm + 1e-12))
    return {k: g * scale for k, g in grads.items()}, norm


def finite_difference_check(loss_fn, params, grads, rng, n_entries=20, eps=1e-6):
    """Central differences on >= n_entries random entries per tensor plus its largest-gradient entry."""
    worst, checked = 0.0, 0
    for name, w in params.items():
        flat, g = w.reshape(-1), grads[name].reshape(-1)
        idx = set(rng.choice(flat.size, min(n_entries, flat.size), replace=False).tolist())
        idx.add(int(np.argmax(np.abs(g))))
        for i in idx:
            old = flat[i]
            flat[i] = old + eps
            lp = loss_fn()
            flat[i] = old - eps
            lm = loss_fn()
            flat[i] = old
            num = (lp - lm) / (2 * eps)
            worst = max(worst, abs(num - g[i]) / max(1e-6, abs(num) + abs(g[i])))
        checked += 1
    return worst, checked


def paired_bootstrap(diffs, rng, n_boot=2000):
    d = np.asarray(diffs, dtype=float)
    boots = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    return float(d.mean()), [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def verdict(mean, ci, direction, mesi):
    good = (ci[0] > 0 and mean >= mesi) if direction == "greater" else (ci[1] < 0 and mean <= -mesi)
    bad = ci[1] < 0 if direction == "greater" else ci[0] > 0
    return "supported" if good else ("contradicted" if bad else "inconclusive")


def write_report(lines, path, payload):
    for line in lines:
        print(line)
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
# END STANDARD UTILITIES


# ============================================================ data and tasks
ERAS = {
    "native": {"p_petty": 0.20, "p_fool": 0.25, "talent_mu": 0.00, "talent_sd": 0.35, "mimic": 0.0},
    "shifted": {"p_petty": 0.50, "p_fool": 0.15, "talent_mu": 0.55, "talent_sd": 0.40, "mimic": 0.0},
    "mimic": {"p_petty": 0.35, "p_fool": 0.20, "talent_mu": 0.20, "talent_sd": 0.40, "mimic": 1.0},
}


def sample_candidates(rng, n, era, office=None):
    """Latent talent and character, an office, the outcome in office, and a twelve-arrow record."""
    e = ERAS[era]
    tal = np.exp(rng.normal(e["talent_mu"], e["talent_sd"], n))
    tal = np.where(rng.random(n) < e["p_fool"], 0.25 * tal, tal)
    petty = rng.random(n) < e["p_petty"]
    char = np.where(petty, rng.uniform(-1.0, -0.35, n), rng.uniform(0.25, 1.0, n))
    off = rng.integers(0, 3, n) if office is None else np.full(n, office)
    y = SCOPE[off] * tal * char + 0.15 * rng.standard_normal(n)
    A, t = N_ARROWS, tal[:, None]
    fade = np.exp(-2.5 * e["mimic"] * t)                    # talent keeps the mimic's record clean
    trick = rng.random((n, A)) < np.where(petty, 0.30, 0.05)[:, None]
    shot = t * (1.0 + 0.25 * rng.standard_normal((n, A))) + trick * t * rng.uniform(0.6, 1.4, (n, A))
    difficulty = rng.uniform(0.0, 1.0, (n, A)) + 0.8 * trick
    tempo = 0.3 * t + 0.4 * rng.standard_normal((n, A)) + e["mimic"] * 0.9 * (petty[:, None] * t)
    lapse = rng.random((n, A)) < np.where(petty[:, None], 0.20 * fade, 0.02)
    precision = np.where(lapse, rng.normal(-0.6, 0.25, (n, A)), rng.normal(0.7, 0.2, (n, A)))
    composure = 0.25 * char[:, None] + 0.5 * rng.standard_normal((n, A))
    crooked = np.where(petty[:, None], fade, 1.0)
    straight = rng.normal(0.5, 0.3, (n, A)) - 0.9 * trick * crooked
    return {"B": np.stack([shot, difficulty, tempo], -1), "C": np.stack([precision, composure, straight], -1),
            "O": np.eye(3)[off], "y": y, "talent": tal, "char": char, "petty": petty}


def make_slates(rng, n_slates, era, k=SLATE):
    """Vacancies: each slate is k candidates for one office; returns a flat pool and slate indices."""
    parts = [sample_candidates(rng, k, era, office=int(o)) for o in rng.integers(0, 3, n_slates)]
    pool = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    pool["slates"] = np.arange(n_slates * k).reshape(n_slates, k)
    return pool


def subset(data, idx):
    return {k: v[idx] for k, v in data.items() if k != "slates"}


def as_record(X, n_offices=3):
    """External sequences (N,T,d): the first half of the channels is brilliance, the rest conduct."""
    h = X.shape[-1] // 2
    return {"B": X[..., :h], "C": X[..., h:], "O": np.zeros((X.shape[0], n_offices))}


def soft_pool(h, tau, sign):
    """sign +1: soft maximum over arrows; sign -1: soft minimum. Returns pooled (N,H) and weights (N,A,H)."""
    z = sign * h / tau
    top = z.max(axis=1, keepdims=True)
    e = np.exp(z - top)
    return sign * tau * (top[:, 0, :] + np.log(e.mean(axis=1))), e / e.sum(axis=1, keepdims=True)


def gaussian_nll(m, v, y):
    r = y - m
    n = y.shape[0]
    loss = float(np.mean(0.5 * np.log(v) + 0.5 * r * r / v))
    return loss, -r / v / n, (0.5 / v - 0.5 * r * r / (v * v)) / n


# ============================================================ model: the whole-pot commander
def init_reach(rng, db, h):
    return {"W1": rng.standard_normal((db, h)) * 0.8 / math.sqrt(db), "b1": np.zeros(h),
            "w2": np.abs(rng.standard_normal(h)) * 0.5, "uR": np.zeros(3), "b2": np.zeros(1)}


def reach_forward(p, B, O):
    """Non-negative reach from the best arrows: softplus per arrow, soft maximum over the pot, softplus."""
    pre = B @ p["W1"] + p["b1"]
    hr = softplus(pre)
    P, wr = soft_pool(hr, TAU_POOL, +1.0)
    aR = P @ p["w2"] + O @ p["uR"] + p["b2"][0]
    return softplus(aR), {"pre": pre, "P": P, "wr": wr, "aR": aR}


def reach_backward(p, B, O, st, dR, g):
    daR = dR * sigmoid(st["aR"])
    g["w2"] += st["P"].T @ daR
    g["uR"] += O.T @ daR
    g["b2"] += daR.sum()
    dpre = st["wr"] * np.outer(daR, p["w2"])[:, None, :] * sigmoid(st["pre"])
    g["W1"] += np.einsum("nad,nah->dh", B, dpre)
    g["b1"] += dpre.sum((0, 1))


def init_commander(rng, db, dc, h=6, variant=None):
    p = init_reach(rng, db, h)
    din = dc + db if variant == "open_commander" else dc
    p.update({"V1": rng.standard_normal((din, h)) * 1.2 / math.sqrt(din), "c1": np.zeros(h),
              "v2": rng.standard_normal(h) * 0.5, "uM": np.zeros(3), "c2": np.zeros(1),
              "v3": rng.standard_normal(h) * 0.3, "uS": np.zeros(3), "c3": np.array([-1.0]),
              "rho": np.array([-3.0])})
    return p


def commander_forward(model, batch):
    """Reach (brilliance, best arrows) times a bounded character (conduct only, worst arrows)."""
    p, ko, var = model["params"], model["ko"], model.get("variant")
    B, C, O = batch["B"], batch["C"], batch["O"]
    R, rst = reach_forward(p, B, O)
    if var == "signed_reach":
        R = rst["aR"]
    Cin = np.concatenate([C, B], -1) if var == "open_commander" else C
    hc = np.tanh(Cin @ p["V1"] + p["c1"])
    if var == "ordered_pot":
        order = np.linspace(0.2, 1.8, hc.shape[1])[None, :, None]
        M, wc = (hc * order).mean(axis=1), None
    else:
        M, wc = soft_pool(hc, TAU_POOL, -1.0)
    aM = M @ p["v2"] + O @ p["uM"] + p["c2"][0]
    aS = M @ p["v3"] + O @ p["uS"] + p["c3"][0]
    mu, s = np.tanh(aM), softplus(aS) + S_FLOOR
    if ko.get("commander") == "mean":
        mu, s = np.full_like(mu, model["mu_bar"]), np.full_like(s, model["s_bar"])
    if ko.get("reach") in ("mean", "zero"):
        R = np.full_like(R, model["R_bar"] if ko["reach"] == "mean" else 0.0)
    s0 = softplus(p["rho"][0]) + VAR_FLOOR
    m = R + mu if var == "talent_adds" else R * mu
    reach_for_risk = np.full_like(R, model["R_bar"]) if ko.get("risk") == "flat" else R
    v = reach_for_risk ** 2 * s ** 2 + s0
    return m, v, {"R": R, "mu": mu, "s": s, "aM": aM, "aS": aS, "M": M, "wc": wc, "hc": hc, "reach": rst}


def commander_backward(model, batch, st, dm, dv):
    p = model["params"]
    g = {k: np.zeros_like(w) for k, w in p.items()}
    R, mu, s = st["R"], st["mu"], st["s"]
    reach_backward(p, batch["B"], batch["O"], st["reach"], dm * mu + dv * 2.0 * R * s * s, g)
    g["rho"][0] = dv.sum() * sigmoid(p["rho"][0])
    daM = dm * R * (1.0 - mu * mu)
    daS = dv * 2.0 * R * R * s * sigmoid(st["aS"])
    g["v2"], g["uM"], g["c2"][0] = st["M"].T @ daM, batch["O"].T @ daM, daM.sum()
    g["v3"], g["uS"], g["c3"][0] = st["M"].T @ daS, batch["O"].T @ daS, daS.sum()
    dM = np.outer(daM, p["v2"]) + np.outer(daS, p["v3"])
    dpre = st["wc"] * dM[:, None, :] * (1.0 - st["hc"] ** 2)
    g["V1"] = np.einsum("nad,nah->dh", batch["C"], dpre)
    g["c1"] = dpre.sum((0, 1))
    return g


# ============================================================ baseline: PNA-style multi-aggregator set network
def init_pna(rng, db, dc, h=4, k=3):
    d = db + dc
    return {"W": rng.standard_normal((d, h)) * 0.8 / math.sqrt(d), "b": np.zeros(h),
            "G1": rng.standard_normal((3 * h + 3, k)) * 0.8 / math.sqrt(3 * h + 3), "g1": np.zeros(k),
            "wm": rng.standard_normal(k) * 0.3, "bm": np.zeros(1), "wv": np.zeros(k), "bv": np.array([-1.0])}


def pna_forward(model, batch):
    """Mean, soft-max and soft-min pooled per-arrow embeddings of all channels, then a softplus MLP."""
    p = model["params"]
    Z = np.concatenate([batch["B"], batch["C"]], -1)
    pre = Z @ p["W"] + p["b"]
    h = softplus(pre)
    Pmax, wa = soft_pool(h, TAU_POOL, +1.0)
    Pmin, wb = soft_pool(h, TAU_POOL, -1.0)
    F = np.concatenate([h.mean(axis=1), Pmax, Pmin, batch["O"]], 1)
    qa = F @ p["G1"] + p["g1"]
    q = softplus(qa)
    av = q @ p["wv"] + p["bv"][0]
    return q @ p["wm"] + p["bm"][0], softplus(av) + VAR_FLOOR, {"Z": Z, "pre": pre, "wa": wa, "wb": wb,
                                                               "F": F, "qa": qa, "q": q, "av": av}


def pna_backward(model, batch, st, dm, dv):
    p = model["params"]
    g = {}
    dav = dv * sigmoid(st["av"])
    g["wm"], g["bm"] = st["q"].T @ dm, np.array([dm.sum()])
    g["wv"], g["bv"] = st["q"].T @ dav, np.array([dav.sum()])
    dqa = (np.outer(dm, p["wm"]) + np.outer(dav, p["wv"])) * sigmoid(st["qa"])
    g["G1"], g["g1"] = st["F"].T @ dqa, dqa.sum(0)
    dF = dqa @ p["G1"].T
    h = p["W"].shape[1]
    A = st["Z"].shape[1]
    dh = dF[:, None, :h] / A + st["wa"] * dF[:, None, h:2 * h] + st["wb"] * dF[:, None, 2 * h:3 * h]
    dpre = dh * sigmoid(st["pre"])
    g["W"], g["b"] = np.einsum("nad,nah->dh", st["Z"], dpre), dpre.sum((0, 1))
    return g


# ============================================================ rival: the old score chart (talent only)
def init_oldchart(rng, db, h=6):
    p = init_reach(rng, db, h)
    p.update({"ao": np.full(3, 0.3), "so": np.full(3, -1.0), "rho": np.array([-3.0])})
    return p


def oldchart_forward(model, batch):
    """Reach from the spectacular best arrows; one learned sign and spread per office, none per person."""
    p = model["params"]
    R, rst = reach_forward(p, batch["B"], batch["O"])
    mu = batch["O"] @ np.tanh(p["ao"])
    s = batch["O"] @ softplus(p["so"]) + S_FLOOR
    v = R ** 2 * s ** 2 + softplus(p["rho"][0]) + VAR_FLOOR
    return R * mu, v, {"R": R, "mu": mu, "s": s, "reach": rst}


def oldchart_backward(model, batch, st, dm, dv):
    p = model["params"]
    g = {k: np.zeros_like(w) for k, w in p.items()}
    R, mu, s, O = st["R"], st["mu"], st["s"], batch["O"]
    reach_backward(p, batch["B"], O, st["reach"], dm * mu + dv * 2.0 * R * s * s, g)
    g["ao"] = (O.T @ (dm * R)) * (1.0 - np.tanh(p["ao"]) ** 2)
    g["so"] = (O.T @ (dv * 2.0 * R * R * s)) * sigmoid(p["so"])
    g["rho"][0] = dv.sum() * sigmoid(p["rho"][0])
    return g


# ============================================================ common interface (section 8)
KINDS = {"commander": (commander_forward, commander_backward), "pna": (pna_forward, pna_backward),
         "oldchart": (oldchart_forward, oldchart_backward)}


def build_model(in_dim, out_dim, task_type, rng, kind="commander", variant=None, **cfg):
    """in_dim is the per-arrow width of one channel group (brilliance and conduct each)."""
    if task_type not in TASK_TYPES or out_dim != 1:
        raise ValueError(f"unsupported task {task_type} / out_dim {out_dim}")
    init = {"commander": lambda: init_commander(rng, in_dim, in_dim, variant=variant, **cfg),
            "pna": lambda: init_pna(rng, in_dim, in_dim, **cfg), "oldchart": lambda: init_oldchart(rng, in_dim)}
    return {"kind": kind, "params": init[kind](), "ko": {}, "mutant": None, "variant": variant,
            "mu_bar": 0.0, "s_bar": 0.5, "R_bar": 1.0}


def n_params(model):
    return int(sum(w.size for w in model["params"].values()))


def forward(model, batch):
    return KINDS[model["kind"]][0](model, batch)


def loss_and_grads(model, batch):
    fwd, bwd = KINDS[model["kind"]]
    m, v, st = fwd(model, batch)
    loss, dm, dv = gaussian_nll(m, v, batch["y"])
    grads = bwd(model, batch, st, dm, dv)
    for name in MUTANTS.get(model["mutant"], {}).get("zero_grads", []):
        if name in grads:
            grads[name] = np.zeros_like(grads[name])
    return loss, grads


def predict(model, X):
    """Mean outcome in office for a batch dict, or for external sequences (N,T,d) split into two channels."""
    batch = X if isinstance(X, dict) else as_record(X)
    return forward(model, batch)[0]


def hidden_states(model, X):
    batch = X if isinstance(X, dict) else as_record(X)
    st = forward(model, batch)[2]
    if model["kind"] == "pna":
        return st["F"]
    return np.concatenate([st["reach"]["P"], st.get("M", st["mu"][:, None])], 1)


def appointment_scores(model, batch):
    """Lower confidence bound on the outcome in office: mean minus lambda standard deviations."""
    m, v, _ = forward(model, batch)
    return m - LAMBDA * np.sqrt(v)


MODULE_REGISTRY = {
    "commander": {
        "reach": (["W1", "b1", "w2", "uR", "b2"], "non-negative magnitude from soft-max pooled brilliance", False),
        "commander": (["V1", "c1", "v2", "uM", "c2", "v3", "uS", "c3"],
                      "bounded sign and uncertainty from soft-min pooled conduct channels only", True),
        "risk": (["rho"], "outcome variance scaled by squared reach plus a noise floor", True)},
    "pna": {"encoder": (["W", "b"], "shared per-arrow softplus embedding of all channels", False),
            "head": (["G1", "g1", "wm", "bm", "wv", "bv"], "softplus MLP mean-variance head", False)},
    "oldchart": {"reach": (["W1", "b1", "w2", "uR", "b2"], "non-negative magnitude from brilliance", False),
                 "office_sign": (["ao", "so", "rho"], "one learned sign and spread per office", False)},
}


def modules(model):
    return {name: {"params": ps, "role": role, "signature": sig}
            for name, (ps, role, sig) in MODULE_REGISTRY[model["kind"]].items()}


def knockout(model, name, mode):
    """Copy with one module replaced: commander -> training-mean sign and spread; reach -> training-mean
    (or zero) magnitude; risk -> flat (spread no longer grows with this candidate's reach)."""
    out = dict(model, params={k: v.copy() for k, v in model["params"].items()}, ko=dict(model["ko"]))
    out["ko"][name] = mode
    return out


MUTANTS = {
    "sign_flip": {"lr_sign": -1.0, "note": "Adam update applied with reversed sign"},
    "zero_lr": {"lr_scale": 0.0, "note": "learning rate set to zero"},
    "zero_grad_V1": {"zero_grads": ["V1"], "note": "gradient of the commander's per-arrow conduct weights zeroed"},
    "zero_grad_W1": {"zero_grads": ["W1"], "note": "gradient of the reach tower's per-arrow weights zeroed"},
}


# ============================================================ training
def fit(model, data, budget, rng, lr=0.02, batch=256, val=None, eval_every=50):
    """Adam on the Gaussian likelihood of outcomes in office; checkpoint chosen on validation NLL only."""
    mut = MUTANTS.get(model["mutant"], {})
    opt = Adam(model["params"], lr=lr * mut.get("lr_scale", 1.0))
    history, best, n = [], (np.inf, None), len(data["y"])
    for step in range(1, budget + 1):
        loss, grads = loss_and_grads(model, subset(data, rng.choice(n, min(batch, n), replace=False)))
        grads, _ = clip_global_norm(grads, CLIP)
        opt.step(model["params"], grads, sign=mut.get("lr_sign", 1.0))
        history.append(loss)
        if val is not None and (step % eval_every == 0 or step == budget):
            m, v, _ = forward(model, val)
            nll = gaussian_nll(m, v, val["y"])[0]
            if nll < best[0]:
                best = (nll, {k: w.copy() for k, w in model["params"].items()})
    if best[1] is not None:
        model["params"] = best[1]
    if model["kind"] != "pna":                   # training means used by the knockouts
        st = forward(model, data)[2]
        model.update(mu_bar=float(st["mu"].mean()), s_bar=float(st["s"].mean()), R_bar=float(st["R"].mean()))
    return history


def appoint(model, pool):
    """Fill every vacancy with the slate member of highest lower bound; report the appointees' outcomes."""
    q, sl = appointment_scores(model, pool), pool["slates"]
    pick = sl[np.arange(len(sl)), np.argmax(q[sl], axis=1)]
    tiger = pool["petty"][pick] & (pool["talent"][pick] > 1.2)
    return {"welfare": float(pool["y"][pick].mean()), "catastrophe": float((pool["y"][pick] < -0.5).mean()),
            "tiger_rate": float(tiger.mean()), "petty_rate": float(pool["petty"][pick].mean())}


def random_welfare(pool):
    return float(pool["y"].mean())


def dazzle_index(model, data, delta=1e-3):
    """Share of the predicted mean's input sensitivity that goes to brilliance rather than conduct."""
    base, sens = predict(model, data), []
    for group in ("B", "C"):
        for j in range(data[group].shape[-1]):
            moved = dict(data, **{group: data[group].copy()})
            moved[group][..., j] += delta
            sens.append(float(np.mean(np.abs(predict(model, moved) - base))) / delta)
    k = data["B"].shape[-1]
    return sum(sens[:k]) / max(sum(sens), 1e-12)


def r2_skill(model, data, y_bar):
    """Held-out R^2 of the predicted mean over a constant training-mean predictor, and its standard error."""
    d = (data["y"] - y_bar) ** 2 - (data["y"] - predict(model, data)) ** 2
    var = float(data["y"].var())
    return float(d.mean()) / var, float(d.std()) / math.sqrt(d.size) / var


def loss_drop(history):
    first, last = float(np.mean(history[:5])), float(np.mean(history[-20:]))
    return (first - last) / max(abs(first), 1e-12)


def finite_model(model, data):
    m, v, _ = forward(model, data)
    return all(np.all(np.isfinite(a)) for a in list(model["params"].values()) + [m, v])


# ============================================================ tests: correctness
def new_model(seed, kind="commander", variant=None):
    return build_model(3, 1, "sequence_regression", np.random.default_rng(seed), kind, variant=variant)


def gradcheck(model, data, rng):
    batch = subset(data, np.arange(8))
    _, grads = loss_and_grads(model, batch)
    return finite_difference_check(lambda: loss_and_grads(model, batch)[0], model["params"], grads, rng)


def c1_gradients(train, rng, mutant=None, steps=60):
    """Every tensor of every trained kind, at initialisation and after >= 50 Adam steps."""
    rows, worst = [], 0.0
    for kind in ("commander", "pna", "oldchart"):
        model = new_model(int(rng.integers(1 << 31)), kind)
        model["mutant"] = mutant
        for when in ("init", "after_training_steps"):
            if when != "init":
                fit(model, train, steps, np.random.default_rng(1))
            err, n_t = gradcheck(model, train, np.random.default_rng(2))
            worst = max(worst, err)
            rows.append((kind, when, n_t, len(model["params"]), err))
    ok = worst <= MIND_CARD["thresholds"]["gradcheck_rel_tol"] and all(r[2] == r[3] for r in rows)
    return ok, worst, rows


def c3_learning(model, history, held):
    th = MIND_CARD["thresholds"]
    drop, w = loss_drop(history), appoint(model, held)["welfare"]
    return drop >= th["loss_drop_fraction"] and w >= random_welfare(held) + th["margin_over_trivial"], drop, w


def c5_mutants(train, held, rng):
    """Each learning-breaking mutant must make C1 or a short C3 fail; the unmutated short run must pass."""
    def short_c3(mutant):
        m = new_model(11)
        m["mutant"] = mutant
        return c3_learning(m, fit(m, train, 200, np.random.default_rng(12)), held)[0]
    control, detected = short_c3(None), {}
    for name in MUTANTS:
        c1_ok = c1_gradients(train, np.random.default_rng(13), mutant=name, steps=50)[0] \
            if "zero_grads" in MUTANTS[name] else True
        detected[name] = (not c1_ok) or (not short_c3(name))
    return control, detected


def random_commander(rng, variant=None):
    m = new_model(int(rng.integers(1 << 31)), variant=variant)
    for k, w in m["params"].items():
        w *= rng.uniform(0.5, 3.0)
        w += rng.normal(0.0, 0.5, w.shape)
    m["params"]["rho"][:] = rng.normal(-2.0, 1.0)
    return m


def random_batch(rng, n=16):
    era = ("native", "shifted", "mimic")[int(rng.integers(3))]
    d = sample_candidates(rng, n, era)
    d["B"] = d["B"] * rng.uniform(0.3, 3.0) + rng.normal(0, 0.5, d["B"].shape)
    d["C"] = d["C"] + rng.normal(0, 0.5, d["C"].shape)
    return d


def c6_properties(rng, n_trials=40):
    """C6.1 talent never sets direction; C6.2 for an unfavourable character more reach never raises the
    appointment score; C6.3 the commander reads conduct only; C6.4 the pot has no order. Each searched over
    random parameters and records, with a negative control that must violate it."""
    def sign_gap(m):
        d = random_batch(rng)
        mean, _, st = forward(m, d)
        moved = dict(d, B=d["B"] + rng.normal(0, 2.0, d["B"].shape))
        mean2 = forward(m, moved)[0]
        return float(max(0.0, -np.min(mean * st["mu"]), -np.min(mean2 * st["mu"])))

    def fool_gap(m):
        d = random_batch(rng)
        up = dict(d, B=d["B"] * rng.uniform(1.2, 3.0) + np.abs(rng.normal(0, 0.5, d["B"].shape)))
        (m1, v1, s1), (m2, v2, s2) = forward(m, d), forward(m, up)
        q1, q2 = m1 - LAMBDA * np.sqrt(v1), m2 - LAMBDA * np.sqrt(v2)
        bad = (s1["mu"] <= 0) & (s2["R"] > s1["R"])
        return float(max(0.0, np.max(np.where(bad, q2 - q1, 0.0))))

    def blind_gap(m):
        d = random_batch(rng)
        s1 = forward(m, d)[2]
        s2 = forward(m, dict(d, B=rng.normal(0, 3.0, d["B"].shape)))[2]
        return float(max(np.abs(s1["mu"] - s2["mu"]).max(), np.abs(s1["s"] - s2["s"]).max()))

    def order_gap(m):
        d = random_batch(rng)
        perm = rng.permutation(N_ARROWS)
        a, b = forward(m, d), forward(m, dict(d, B=d["B"][:, perm], C=d["C"][:, perm]))
        return float(max(np.abs(a[0] - b[0]).max(), np.abs(a[1] - b[1]).max()))

    tests = {"C6.1": (sign_gap, "signed_reach"), "C6.2": (fool_gap, "talent_adds"),
             "C6.3": (blind_gap, "open_commander"), "C6.4": (order_gap, "ordered_pot")}
    out = {}
    for tid, (fn, neg) in tests.items():
        worst_pos = max(fn(random_commander(rng)) for _ in range(n_trials))
        worst_neg = max(fn(random_commander(rng, variant=neg)) for _ in range(n_trials))
        out[tid] = (worst_pos <= 1e-9 and worst_neg > 1e-3, worst_pos, worst_neg)
    zero = knockout(random_commander(rng), "reach", "zero")
    out["D-1"] = (bool(np.all(forward(zero, random_batch(rng))[0] == 0.0)), 0.0, None)
    return out


def split_hashes(data):
    return {hashlib.sha1(np.concatenate([b.ravel(), c.ravel()]).tobytes()).hexdigest()
            for b, c in zip(data["B"], data["C"])}


# ============================================================ one seed of the protocol
def run_seed(seed, cfg, keep=False):
    r = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(12)]
    d = {"train": sample_candidates(r[0], cfg["n_train"], "native"),
         "val": sample_candidates(r[0], cfg["n_val"], "native"),
         "held": make_slates(r[1], cfg["n_slates"], "native"), "shift": make_slates(r[2], cfg["n_slates"], "shifted"),
         "m_train": sample_candidates(r[3], cfg["n_train"], "mimic"),
         "m_val": sample_candidates(r[3], cfg["n_val"], "mimic"), "m_held": make_slates(r[4], cfg["n_slates"], "mimic")}
    res, models, hists = {"seed": seed}, {}, {}
    for i, kind in enumerate(("commander", "pna", "oldchart")):
        m = build_model(3, 1, "sequence_regression", r[5 + i], kind)
        hists[kind] = fit(m, d["train"], cfg["steps"], r[8], val=d["val"])
        models[kind] = m
        for split in ("held", "shift"):
            res[f"{kind}_{split}"] = appoint(m, d[split])
    base = res["commander_shift"]["welfare"]
    for name, mode in (("commander", "mean"), ("reach", "mean"), ("risk", "flat")):
        res[f"ko_{name}"] = appoint(knockout(models["commander"], name, mode), d["shift"])["welfare"] - base
    for i, kind in enumerate(("commander", "pna")):
        m = build_model(3, 1, "sequence_regression", r[9 + i], kind)
        fit(m, d["m_train"], cfg["steps"], r[11], val=d["m_val"])
        res[f"{kind}_mimic"] = appoint(m, d["m_held"])
        models[f"{kind}_mimic"] = m
    res["finite"] = all(finite_model(m, subset(d["shift"], np.arange(64))) for m in models.values())
    probe = subset(d["held"], np.arange(400))
    res["dazzle"] = {k: dazzle_index(models[k], probe) for k in ("commander", "pna")}
    res["random"] = {s: random_welfare(d[s]) for s in ("held", "shift", "m_held")}
    return (res, models, hists, d) if keep else res


def hypothesis_table(rows, rng):
    w = lambda r, key: r[key]["welfare"]
    diffs = {"H-SIG": [w(r, "commander_shift") - w(r, "pna_shift") for r in rows],
             "H-NEC": [r["ko_commander"] - r["ko_reach"] for r in rows],
             "H-BLIND": [w(r, "commander_mimic") - w(r, "pna_mimic") for r in rows],
             "H-RIVAL": [w(r, "commander_shift") - w(r, "oldchart_shift") for r in rows]}
    table = []
    for h in MIND_CARD["hypotheses"]:
        vals = diffs[h["id"]]
        if len(rows) < 5:
            table.append({"id": h["id"], "metric": h["metric"], "mean_diff": float(np.mean(vals)),
                          "ci95": [None, None], "mesi": h["mesi"], "n_seeds": len(rows), "verdict": "not evaluated"})
            continue
        mean, ci = paired_bootstrap(vals, rng)
        table.append({"id": h["id"], "metric": h["metric"], "mean_diff": mean, "ci95": ci, "mesi": h["mesi"],
                      "n_seeds": len(rows), "verdict": verdict(mean, ci, h["direction"], h["mesi"])})
    return table


# ============================================================ report
def fmt_ci(ci):
    return "n/a" if ci[0] is None else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def build_report(args, seeds, correctness, grad, mutants, hyps, knock, rows, t0, exit_code):
    fname = os.path.basename(__file__)
    lines = [f"=== VERIFIED REPORT · chapter {CHAPTER:04d} ===",
             f"file {fname} · card revision {MIND_CARD['card_revision']} · mode {'quick' if args.quick else 'full'}",
             f"python {sys.version.split()[0]} · numpy {np.__version__} · seeds {seeds}",
             f"runtime {time.time() - t0:.1f} s · n_params commander {grad['n_params']['commander']}, "
             f"pna {grad['n_params']['pna']}, oldchart {grad['n_params']['oldchart']}",
             f"gradcheck: {grad['tensors']}/{grad['total']} tensor checks (every tensor, 3 kinds, init + after "
             f"training), max rel error {grad['worst']:.2e}", "correctness:"]
    lines += [f"  {c['id']:<5} {c['name']:<36} {'PASS' if c['passed'] else 'FAIL'}  {c['detail']}" for c in correctness]
    lines.append(f"mutants: {mutants['detected']}/{mutants['total']} detected (score {mutants['score']:.2f})")
    lines.append("hypotheses (paired per-seed differences, 95% bootstrap, 2000 resamples):")
    for h in hyps:
        lines.append(f"  {h['id']:<8} mean {h['mean_diff']:+.3f}  CI {fmt_ci(h['ci95'])}  mesi {h['mesi']}  "
                     f"-> {h['verdict']}")
    lines.append("per-seed appointee welfare (native held / shifted / mimic held; random in brackets):")
    for r in rows:
        rw = r["random"]
        lines.append(f"  seed {r['seed']}: commander {r['commander_held']['welfare']:.3f}/"
                     f"{r['commander_shift']['welfare']:.3f}/{r['commander_mimic']['welfare']:.3f}  pna "
                     f"{r['pna_held']['welfare']:.3f}/{r['pna_shift']['welfare']:.3f}/{r['pna_mimic']['welfare']:.3f}"
                     f"  oldchart {r['oldchart_held']['welfare']:.3f}/{r['oldchart_shift']['welfare']:.3f}  "
                     f"[{rw['held']:.3f}/{rw['shift']:.3f}/{rw['m_held']:.3f}]")
    lines.append("descriptive (no verdict): shifted tiger rate (petty, talent > 1.2) commander/pna/oldchart; "
                 "dazzle index (brilliance share of sensitivity) commander/pna:")
    for r in rows:
        lines.append(f"  seed {r['seed']}: tigers {r['commander_shift']['tiger_rate']:.3f}/"
                     f"{r['pna_shift']['tiger_rate']:.3f}/{r['oldchart_shift']['tiger_rate']:.3f}  dazzle "
                     f"{r['dazzle']['commander']:.2f}/{r['dazzle']['pna']:.2f}")
    lines.append("knockouts on the trained commander (shifted welfare change):")
    for k in knock:
        lines.append(f"  {k['module']:<10} {k['mode']:<5} signature={str(k['signature']):<5} change "
                     f"{k['metric_change']:+.3f}  CI {fmt_ci(k['ci95'])}")
    lines += [f"task types: {', '.join(TASK_TYPES)}", f"exit code {exit_code}", "=== END REPORT ==="]
    payload = {"schema_version": "1.0", "chapter": CHAPTER, "file": fname, "card_revision": MIND_CARD["card_revision"],
               "environment": {"python": sys.version.split()[0], "numpy": np.__version__}, "seeds": seeds,
               "runtime_s": round(time.time() - t0, 2), "n_params": grad["n_params"]["commander"],
               "gradcheck": {"tensors_checked": grad["tensors"], "tensors_total": grad["total"],
                             "max_rel_error": grad["worst"], "checked_at": ["init", "after_training_steps"],
                             "passed": grad["passed"]},
               "correctness": correctness, "mutants": mutants, "hypotheses": hyps, "knockouts": knock,
               "task_types": TASK_TYPES, "exit_code": exit_code}
    return lines, payload


# ============================================================ optional real-data bridge (section 10.3)
def data_bridge(path, rng):
    """CSV with columns b<j>_t<i>, c<j>_t<i> (three channels each, any number of trials), office and y."""
    if not path or not os.path.exists(path):
        print(f"--data: no file at {path!r}; real-data bridge skipped.")
        return
    raw = np.genfromtxt(path, delimiter=",", names=True)
    T = 1 + max(int(c.split("_t")[1]) for c in raw.dtype.names if c.startswith("b"))
    grab = lambda g: np.stack([np.stack([raw[f"{g}{j}_t{i}"] for j in range(3)], -1) for i in range(T)], 1)
    full = {"B": grab("b"), "C": grab("c"), "O": np.eye(3)[raw["office"].astype(int) % 3], "y": raw["y"]}
    n = len(full["y"]) - len(full["y"]) % SLATE
    perm = rng.permutation(len(full["y"]))
    cut = int(0.7 * n) - int(0.7 * n) % SLATE
    tr, te = subset(full, perm[:cut]), subset(full, perm[cut:n])
    te.update(petty=np.zeros(len(te["y"]), bool), talent=np.zeros(len(te["y"])),
              slates=np.arange(len(te["y"])).reshape(-1, SLATE))
    for kind in ("commander", "pna"):
        m = new_model(0, kind)
        fit(m, tr, 300, np.random.default_rng(1))
        print(f"--data {kind}: held NLL {gaussian_nll(*forward(m, te)[:2], te['y'])[0]:.3f}, "
              f"appointee mean {appoint(m, te)['welfare']:.3f} (random {random_welfare(te):.3f})")


# ============================================================ entry point
def correctness_suite(args, cfg, worlds, first, models, hists):
    """C1-C7 on the first seed's worlds; returns rows plus the gradient summary and mutant tally."""
    c1_ok, worst, grad_rows = c1_gradients(worlds["train"], np.random.default_rng(args.seed + 1), mutant=args.mutant)
    rep, rep_hist = models["commander"], hists["commander"]
    if args.mutant:                                   # retrain under the mutant so that C3 can detect it
        rep = new_model(3)
        rep["mutant"] = args.mutant
        rep_hist = fit(rep, worlds["train"], cfg["steps"], np.random.default_rng(4), val=worlds["val"])
    c3_ok, drop, w = c3_learning(rep, rep_hist, worlds["held"])
    a, b = new_model(5), new_model(5)
    ha, hb = fit(a, worlds["train"], 30, np.random.default_rng(6)), fit(b, worlds["train"], 30, np.random.default_rng(6))
    probe = subset(worlds["held"], np.arange(32))
    same = ha == hb and np.array_equal(predict(a, probe), predict(b, probe))
    held, rnd = worlds["held"], random_welfare(worlds["held"])
    perm = np.random.default_rng(7)
    shuffled = dict(worlds["train"], y=perm.permutation(worlds["train"]["y"]))
    sh = new_model(8)
    fit(sh, shuffled, cfg["steps"], np.random.default_rng(9), val=dict(worlds["val"], y=perm.permutation(worlds["val"]["y"])))
    sh_r2, sh_se = r2_skill(sh, held, shuffled["y"].mean())
    band = max(MIND_CARD["thresholds"]["shuffled_r2_band"], 3.0 * sh_se)
    true_r2 = r2_skill(rep, held, worlds["train"]["y"].mean())[0]
    control, detected = c5_mutants(worlds["train"], held, np.random.default_rng(10)) if not args.mutant else (True, {})
    props = c6_properties(np.random.default_rng(11))
    hs = [split_hashes(worlds[k]) for k in ("train", "val", "held", "shift")]
    disjoint = all(not (hs[i] & hs[j]) for i in range(4) for j in range(i + 1, 4))
    finite = first["finite"] and all(np.isfinite(h).all() for h in hists.values())
    rows = [{"id": "C1", "name": "gradient_check", "passed": c1_ok, "detail": f"max rel err {worst:.2e}"},
            {"id": "C2", "name": "determinism_and_finiteness", "passed": bool(same and finite),
             "detail": f"identical reruns {same}, finite {finite}"},
            {"id": "C3", "name": "learning", "passed": c3_ok, "detail": f"loss drop {drop:.2f} (>= 0.25), held "
             f"welfare {w:.3f} (>= random {rnd:.3f} + 0.15)"},
            {"id": "C4", "name": "shuffled_label_control", "passed": abs(sh_r2) <= band,
             "detail": f"held R2 {sh_r2:+.3f} (band +-{band:.2f}; true labels {true_r2:.3f})"}]
    if not args.mutant:
        rows.append({"id": "C5", "name": "mutant_detection", "passed": control and all(detected.values()),
                     "detail": f"control passes {control}; " + ", ".join(f"{k}:{v}" for k, v in detected.items())})
    names = {"C6.1": "talent_never_sets_direction", "C6.2": "fool_preferred_to_petty_man",
             "C6.3": "commander_reads_conduct_only", "C6.4": "pot_is_unordered",
             "D-1": "zero_reach_zero_effect (definition check)"}
    for tid, (ok, pos, neg) in props.items():
        detail = f"worst {pos:.1e}" + ("" if neg is None else f", negative control {neg:.1e}")
        rows.append({"id": tid, "name": names[tid], "passed": bool(ok), "detail": detail})
    rows.append({"id": "C7", "name": "split_integrity", "passed": disjoint, "detail": "sha1 of every record"})
    grad = {"worst": worst, "tensors": sum(g[2] for g in grad_rows), "total": sum(g[3] for g in grad_rows),
            "passed": c1_ok, "n_params": {k: n_params(models[k]) for k in ("commander", "pna", "oldchart")}}
    return rows, grad, detected, finite


def main(argv=None):
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__), description="Chapter 0266 Whole-Pot Commander")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=266)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--card", action="store_true")
    ap.add_argument("--mutant", type=str, default=None, choices=sorted(MUTANTS))
    ap.add_argument("--data", type=str, default=None)
    args = ap.parse_args(argv)
    if args.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    t0 = time.time()
    cfg = dict(n_train=1500, n_val=300, n_slates=300, steps=300, seeds=1, budget=20.0) if args.quick else \
        dict(n_train=3000, n_val=600, n_slates=600, steps=600, seeds=5, budget=180.0)
    seeds = [args.seed + i for i in range(args.seeds or cfg["seeds"])]
    print(f"seeds {seeds} (base {args.seed}); mutant {args.mutant}; mode {'quick' if args.quick else 'full'}")
    rng = np.random.default_rng(args.seed)
    first, models, hists, worlds = run_seed(seeds[0], cfg, keep=True)
    correctness, grad, detected, finite = correctness_suite(args, cfg, worlds, first, models, hists)
    rows = [first] + ([run_seed(s, cfg) for s in seeds[1:]] if not args.mutant else [])
    finite = finite and all(r["finite"] for r in rows)
    hyps = hypothesis_table(rows, rng) if not args.mutant else []
    knock = []
    for name, mode in (("commander", "mean"), ("reach", "mean"), ("risk", "flat")):
        vals = [r[f"ko_{name}"] for r in rows]
        mean, ci = paired_bootstrap(vals, rng) if len(vals) >= 5 else (float(np.mean(vals)), [None, None])
        knock.append({"module": name, "mode": mode, "signature": MODULE_REGISTRY["commander"][name][2],
                      "metric_change": mean, "ci95": ci})
    runtime_ok = time.time() - t0 <= cfg["budget"]
    correctness.append({"id": "C8", "name": "runtime_budget", "passed": runtime_ok,
                        "detail": f"{time.time() - t0:.1f} s (<= {cfg['budget']:.0f} s)"})
    exit_code = 4 if not finite else (1 if not all(c["passed"] for c in correctness if c["id"] != "C8")
                                      else (3 if not runtime_ok else 0))
    mut = {"detected": int(sum(detected.values())), "total": len(MUTANTS) if not args.mutant else 0,
           "score": (sum(detected.values()) / len(MUTANTS)) if detected else 0.0}
    lines, payload = build_report(args, seeds, correctness, grad, mut, hyps, knock, rows, t0, exit_code)
    write_report(lines, args.json, payload)
    if args.data:
        data_bridge(args.data, np.random.default_rng(args.seed))
    return exit_code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except FloatingPointError as exc:
        print(f"non-finite values: {exc}", file=sys.stderr)
        sys.exit(4)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#================================================================================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 12 Minds 221 - 240 Available on Amazon https://www.amazon.com/dp/B0HKVBKNTT
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0240_al_mutanabbi_915 - Al-Mutanabbi (c.915-965)
#================================================================================  
"""QADR -- the measure that meets the magnitude.

Thesis
    A magnitude exists only as a relation to the measure (qadr) of whoever meets it, so
    a mind perceives the world in its own measure, can learn any measure (its own
    included) only from the world's verdicts on deeds, and must re-measure a task
    before handing it to a body or a companion of another size.

Evidence and provenance  (provenance: belief -- the Diwan survives; paraphrases below
are this file's renderings of the Arabic, not quotations from modern translations)
    D1  Hadath ode (343/954), vv. 1-2: resolutions come in the measure of the resolute;
        small things look great in the eye of the small, great things small in the eye
        of the great.
    D2  Same ode, vv. 3-4: the prince loads the army with his aspiration, which vast
        armies could not carry, and seeks in people what is in his own soul.
    D3  Kafur-period ode in -mi rhyme: he befriends a soul before its body and knows it
        by its deeds and its speech; a man whose deeds are bad suspects everyone.
    D4  Reproach to Sayf al-Dawla ('wa-harra qalbah'): the horse, the night and the desert
        know me; do not take swelling for fat.
    D5  Ode 'ayna azma'ta': when souls are great, bodies tire in pursuit of their aims.
    D6  Ode to Sayf al-Dawla in -ani rhyme: judgment comes before the courage of the brave.
    D7  Deeds: in autumn 950 the Hamdanid army, laden with captives and booty, was
        destroyed in the Taurus passes; the poet sought offices neither patron granted
        and then satirised Kafur by origin rather than by deed.
    D8  Speculation (this file's reading of D1, D3, D4): percepts expressed in one's own
        measure cannot reveal that measure; only verdicts on deeds can.

Doctrine -> mechanism -> test
    D1, D8  M1 ayn_remeasure: tasks enter only as log-ratios to a measure  C6.EQV   H-SIG
    D3, D4  M2 fil_reader: capability and precision read from verdicts,    C6.PERM  knockout
            not from the size of attempts                                   C6.MASK
    D2      M3 projection_prior: unwitnessed others read in own measure     C6.DEF   H-NEC, H-BLIND
    D1, D6  M4 law_head: size weighed only as a margin against a measure;   C6.LAW
            the verdicts teach the direction
    D5, D7  blind spot: armies and bodies have absolute sizes               --       H-BLIND, H-AMEND

Research question
    Calibration and metacognition, multi-agent delegation: can an agent that receives
    task difficulty only relative to its own unknown capability recover capabilities
    from outcome records, generalise to capability scales never seen, and delegate by
    re-measuring -- and does its scale-free prior fail where companions come from a
    common population and bodies have fixed capacity?

Closest prior art and the delta
    Rasch item response model (ability minus difficulty through a logistic link), Elo
    and TrueSkill ratings, Deep Sets for amortised set encoding, social projection as
    induction from the self (Dawes 1989; Krueger 1998). Delta: the agent is itself one
    of the rated persons and never receives absolute difficulties, so its own capability
    must be read from its own record before any delegation, and an unwitnessed companion
    can only be read in the observer's measure. Baseline: a size-matched MLP on the same
    inputs.

Blind spot
    The measure law is scale-free and the world is not. A common population of
    companions and a body limit carry absolute sizes a ratio-only model cannot hold.

Task (generative process)
    leader capability mu_s ~ U[-1, 1]; the executor is the leader (p = 0.4) or a companion
    mu_e = mu_s + clip(N(0, 1), -2.5, 2.5); affair size a = mu_s + U[-2.5, 2.5];
    nuisance t ~ N(0, I_3) with difficulty d(t) = 0.6 tanh(1.2 t1) - 0.4 tanh(t2 t3);
    percept p = a - mu_s + N(0, 0.15); every agent has R = 8 recorded deeds attempted at
    mu + ambition U[0, 1.5] + U[-1.5, 1.5] with verdicts ~ Bernoulli(sigmoid(2 (mu - a_i)));
    companions show 0..8 deeds; label y = [mu_e - a - d(t) > 0].
    Shifted split: mu_s ~ U[3, 5]. Absolute world: mu_e ~ N(0, 0.8) for companions with
    0..3 deeds, and labels and verdicts lose 1.5 max(0, a - 0.8).

Limits
    Synthetic logistic worlds, binary outcomes, one scalar capability per agent, no
    sequential control. QADR needs the record layout in COL; other inputs go to the MLP.
    A research prototype of one AGI-relevant mechanism: not an AGI, and no claim to
    replicate a person.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 2,
    "revision_log": [{"revision": 2, "date": "2026-09-15",
                      "change": ("Law head: direction learned (kappa unconstrained, initialised near 0) instead of fixed-sign; "
                                 "property C6.MONO replaced by C6.LAW; C4 scored by held-out log-loss (band 0.05 nats) instead "
                                 "of accuracy (band 0.04); break_equivariance mutant made quadratic"),
                      "reason": ("The first two --quick runs showed that a fixed-sign head is a label-independent shortcut: "
                                 "untrained it scored 0.804 and after shuffled-label training 0.846 on true labels, so C4 "
                                 "failed as designed. The doctrine now supplies the relational form and the verdicts supply "
                                 "the direction. A linear leak of absolute size cancels in relative comparisons and was not "
                                 "a real break. Hypotheses, splits, metrics and mesi are unchanged.")}],
    "generation": {"template_version": "codeguidelines 1.0 (15 September 2026)", "generator": "Claude (Anthropic)",
                   "generator_version": "claude-opus-5", "date": "2026-09-15"},
    "id": 240, "figure": "Al-Mutanabbi", "born": 915, "died": 965,
    "civilization": "Arab (Abbasid era: Hamdanid Aleppo, Ikhshidid Egypt, Buyid Iraq and Fars)",
    "provenance": "belief",
    "thesis": ("A magnitude exists only as a relation to the measure (qadr) of whoever meets it: a mind perceives "
               "in its own measure, can learn any measure, its own included, only from the world's verdicts on "
               "deeds, and must re-measure a task before handing it to a body or companion of another size."),
    "evidence": [
        {"id": "D1", "basis": "primary", "source": "Diwan, al-Hadath ode (343/954), vv. 1-2; al-Wahidi, Sharh",
         "claim": "Resolutions come in the measure of the resolute; small things look great to the small, great things small to the great."},
        {"id": "D2", "basis": "primary", "source": "Diwan, al-Hadath ode (343/954), vv. 3-4",
         "claim": "The prince loads the army with his aspiration, which vast armies could not carry, and seeks in people what is in his own soul."},
        {"id": "D3", "basis": "primary", "source": "Diwan, Kafur-period ode in -mi rhyme",
         "claim": "He knows a soul by its deeds and speech; a man whose deeds are bad turns suspicious of everyone."},
        {"id": "D4", "basis": "primary", "source": "Diwan, reproach to Sayf al-Dawla (wa-harra qalbah)",
         "claim": "The horse, the night and the desert know him; the patron should not take swelling for fat."},
        {"id": "D5", "basis": "primary", "source": "Diwan, ode to Sayf al-Dawla (ayna azma'ta ayyuha l-humamu)",
         "claim": "When souls are great, bodies tire in pursuit of their aims."},
        {"id": "D6", "basis": "primary", "source": "Diwan, ode to Sayf al-Dawla in -ani rhyme; al-Wahidi, Sharh",
         "claim": "Judgment comes before courage; without intellects the lowliest lion would stand nearer honour than man."},
        {"id": "D7", "basis": "deeds", "source": "Canard 1951; Treadgold 1997; Larkin 2008",
         "claim": "The 950 army was destroyed laden in the passes; he sought offices his patrons withheld and satirised Kafur by origin, not deed."},
        {"id": "D8", "basis": "speculation", "source": "this chapter's reading of D1, D3, D4",
         "claim": "Percepts expressed in one's own measure cannot reveal that measure; only verdicts on deeds can."},
    ],
    "research_question": {
        "category": "calibration, abstention and metacognition; multi-agent cooperation and conflict",
        "question": ("Can an agent that receives task difficulty only relative to its own unknown capability recover "
                     "capabilities from outcome records, generalise to unseen capability scales and delegate by "
                     "re-measuring, and where does its scale-free prior fail?")},
    "mechanism": {
        "name": "QADR: measure-relative appraisal with a verdict-read capability ledger",
        "family": "latent-capability inference; shift-equivariant set encoding; frame conversion for delegation",
        "signature_modules": ["ayn_remeasure", "fil_reader", "projection_prior"],
        "closest_prior_art": ["Rasch item response model (Rasch 1960)",
                              "Elo and TrueSkill ratings (Elo 1978; Herbrich, Minka and Graepel 2006)",
                              "Deep Sets (Zaheer et al. 2017)",
                              "Social projection as induction from the self (Dawes 1989; Krueger 1998)"],
        "overlap": "Medium",
        "prior_art_queries": ["amortized item response theory ability estimation neural",
                              "egocentric projection prior capability estimation delegation",
                              "scale equivariant set encoder capability outcome records",
                              "agent task allocation unknown skill logistic IRT"],
        "contribution_type": "mechanism",
        "delta": ("The agent is one of the rated persons and never receives absolute difficulties, so its own "
                  "capability is read from its own record before delegation, and unwitnessed companions can be "
                  "read only in the observer's measure."),
    },
    "traceability": [
        {"doctrine": "D1, D8", "mechanism": "M1 ayn_remeasure", "property_test": "C6.EQV", "hypothesis": "H-SIG"},
        {"doctrine": "D3, D4", "mechanism": "M2 fil_reader", "property_test": "C6.PERM, C6.MASK", "hypothesis": "knockout table"},
        {"doctrine": "D2", "mechanism": "M3 projection_prior", "property_test": "C6.DEF (definition check)", "hypothesis": "H-NEC, H-BLIND"},
        {"doctrine": "D1, D6", "mechanism": "M4 law_head", "property_test": "C6.LAW", "hypothesis": "H-SIG"},
        {"doctrine": "D5, D7", "mechanism": "absolute_register (amendment)", "property_test": "C6.EQV negative control", "hypothesis": "H-BLIND, H-AMEND"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "Reading tasks in the observer's measure keeps accuracy when every capability lies far outside the training range, where the MLP cannot.",
         "metric": "accuracy", "split": "shifted", "comparison": "model - baseline", "direction": "greater", "mesi": 0.05, "seeds": 5},
        {"id": "H-NEC", "statement": "Removing re-measurement (every executor read as the self) costs more held-out accuracy than removing the nuisance type path.",
         "metric": "accuracy", "split": "test", "comparison": "signature_knockout - matched_knockout", "direction": "less", "mesi": 0.03, "seeds": 5},
        {"id": "H-BLIND", "statement": "Where companions come from a common population and bodies have a fixed capacity, the scale-free model loses to the MLP.",
         "condition": "absolute world: companions mu_e ~ N(0, 0.8) with at most 3 witnessed deeds; body limit 1.5 max(0, a - 0.8)",
         "grounding": "D2, D5, D7: armies carry the aspiration badly, bodies tire, the 950 army died laden in the passes",
         "metric": "accuracy", "split": "abs_test", "comparison": "model - baseline", "direction": "less", "mesi": 0.02, "seeds": 5},
        {"id": "H-AMEND", "statement": "An absolute register (population prior and body capacity) added to the scale-free model recovers accuracy in the absolute world.",
         "metric": "accuracy", "split": "abs_test", "comparison": "amended - model", "direction": "greater", "mesi": 0.02, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.25, "margin_over_trivial": 0.15, "shuffled_metric": "held-out log-loss", "shuffled_band_nats": 0.05},
    "probe_predictions": [{"probe": p, "expected": e} for p, e in (
        ("P1", "below baseline"), ("P2", "above baseline"), ("P3", "equal to baseline"), ("P4", "equal to baseline"),
        ("P5", "equal to baseline"), ("P6", "above baseline"), ("P7", "below baseline"), ("P8", "equal to baseline"),
        ("P9", "above baseline"), ("P10", "above baseline"), ("P11", "below baseline"))],
    "dialectic_links": [
        {"chapter": 168, "relation": "successor", "test": None, "note": "inherits the qasida of Imru' al-Qays; no shared mechanism"},
        {"chapter": 235, "relation": "rival", "test": None, "note": "al-Farabi at the same Hamdanid court c. 948-950; no recorded meeting"},
        {"chapter": 253, "relation": "teacher", "test": None, "note": "al-Ma'arri commented on the Diwan; figure not yet worked"},
    ],
    "corpus_neighbors": [
        {"chapter": 57, "similarity": None, "difference": "Protagoras: truth relative to the knower; here size relative to measure, with the measure recoverable only from verdicts and re-expressed for others"},
        {"chapter": 180, "similarity": None, "difference": "al-Khansa: divisive normaliser damping grief; here additive log-shift equivariance of capability reading"},
        {"chapter": 189, "similarity": None, "difference": "Qatari: contingency gate on fear; here no affect gate, a capability ledger and frame conversion"},
        {"chapter": 239, "similarity": None, "difference": "Rabia Balkhi: another known by re-enactment through one's own tie; here another known by their verdicts, projection only when unwitnessed"},
        {"chapter": 244, "similarity": None, "difference": "Ferdowsi: warrant separate from capability; here capability itself is relational"},
    ],
    "barometer": {"cognitive_processing": ["H-SIG", "P2", "P6", "P10"], "embodied_cognition": ["H-BLIND body slice", "H-AMEND", "P11"],
                  "world_modeling": ["H-BLIND companion slice", "P7"], "consciousness": ["fil_reader on the self record", "P8"],
                  "language_understanding": [], "emotional_intelligence": ["H-NEC"], "creativity": [], "autonomy": ["C3", "P5"]},
    "task_types": ["vector_classification"],
    "applications": [
        {"use": "Routing tasks among AI agents and sub-agents by capability read from logged verdicts, with re-measurement before delegation",
         "sector": "software engineering and AI operations", "dataset": "SWE-bench public evaluation logs (Jimenez et al. 2024)"},
        {"use": "Assigning practice problems at the learner's measure from response logs, including learners from unseen ability ranges",
         "sector": "education", "dataset": "ASSISTments 2009-2010 skill-builder data"},
        {"use": "Matchmaking and handicaps when players arrive from rating ranges absent from training",
         "sector": "games", "dataset": "Lichess open game database"},
    ],
    "safety_notes": ("Battle imagery in the sources is abstracted to affairs and companies; no targeting, tactics or battle "
                     "optimisation. The data hold no attributes of persons: the poet's satire of Kafur judged a man by origin, "
                     "which the chapter treats as history and this file never encodes. No verse is generated or attributed."),
}

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

CHAPTER = 240
R = 8                        # deeds per record
T = 3                        # nuisance task features
IN_DIM = 1 + T + 1 + 6 * R   # percept, type, executor-is-self flag, two records (size, verdict, mask)
COL = {"p": 0, "t": slice(1, 1 + T), "self": 1 + T}
COL.update({name: slice(5 + 8 * k, 13 + 8 * k) for k, name in enumerate(("As", "Os", "Ms", "Ae", "Oe", "Me"))})
TASK_TYPES = ["vector_classification"]
FULL_BUDGET_S, QUICK_BUDGET_S = 180.0, 20.0
GRAD_CLIP = 5.0
MUTANT = None


# BEGIN STANDARD UTILITIES v1.0
def softplus(z):
    return np.logaddexp(0.0, z)


def sigmoid(z):
    return np.exp(-softplus(-np.asarray(z, dtype=float)))


def logsumexp(z, axis=-1):
    m = np.max(z, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(z - m), axis=axis))


def softmax(z, axis=-1):
    e = np.exp(z - np.max(z, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


class Adam:
    def __init__(self, params, beta1=0.9, beta2=0.999, eps=1e-8):
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.b1, self.b2, self.eps, self.t = beta1, beta2, eps, 0

    def step(self, params, grads, lr):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1.0 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1.0 - self.b2) * grads[k] * grads[k]
            m_hat = self.m[k] / (1.0 - self.b1 ** self.t)
            v_hat = self.v[k] / (1.0 - self.b2 ** self.t)
            params[k] -= lr * m_hat / (np.sqrt(v_hat) + self.eps)


def clip_global_norm(grads, max_norm):
    norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    if norm > max_norm:
        grads = {k: g * (max_norm / (norm + 1e-12)) for k, g in grads.items()}
    return grads, norm


def finite_difference_check(loss_fn, params, grads, rng, eps=1e-6, n_random=20, rtol=1e-5, atol=1e-8):
    """Central differences on every tensor: n_random entries plus the largest-gradient entry.
    An entry passes if |analytic - numeric| <= rtol * max(|a|, |n|) + atol. The absolute term
    absorbs difference-quotient round-off (about 1e-9 at eps = 1e-6 in float64); max_rel is
    reported over entries whose scale exceeds 1e-4, where the full relative test applies."""
    report = {}
    for name, P in params.items():
        g = grads[name]
        picks = set(rng.choice(P.size, size=min(n_random, P.size), replace=False).tolist())
        picks.add(int(np.argmax(np.abs(g))))
        worst_rel, worst_abs, ok = 0.0, 0.0, True
        for flat in sorted(picks):
            idx = np.unravel_index(flat, P.shape) if P.ndim else ()
            old = float(P[idx])
            P[idx] = old + eps
            up = loss_fn()
            P[idx] = old - eps
            down = loss_fn()
            P[idx] = old
            num, ana = (up - down) / (2.0 * eps), float(g[idx])
            diff, scale = abs(num - ana), max(abs(num), abs(ana))
            ok = ok and diff <= rtol * scale + atol
            worst_abs = max(worst_abs, diff)
            if scale > 1e-4:
                worst_rel = max(worst_rel, diff / scale)
        report[name] = {"entries": len(picks), "max_rel": worst_rel, "max_abs": worst_abs, "passed": ok}
    return report


def paired_bootstrap_ci(diffs, rng, n_resamples=2000, alpha=0.05):
    d = np.asarray(diffs, dtype=float)
    means = d[rng.integers(0, d.size, size=(n_resamples, d.size))].mean(axis=1)
    return float(d.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def hypothesis_verdict(mean, lo, hi, mesi, direction):
    if direction == "greater":
        if lo > 0 and mean >= mesi:
            return "supported"
        return "contradicted" if hi < 0 else "inconclusive"
    if hi < 0 and mean <= -mesi:
        return "supported"
    return "contradicted" if lo > 0 else "inconclusive"


def write_report(report, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
# END STANDARD UTILITIES


# ----------------------------------------------------------------------------- data and tasks
def child_rng(seed, *keys):
    return np.random.default_rng(np.random.SeedSequence([seed, *keys]))


def body_penalty(a, absolute):
    """A body has a fixed capacity: effort beyond a set size costs whoever carries it."""
    return 1.5 * np.maximum(0.0, a - 0.8) if absolute else np.zeros_like(a)


def deed_record(mu, shown, rng, absolute):
    """Deeds are attempted above one's measure by a private ambition, so the mean attempt
    overstates capability (swelling); only the world's verdicts carry the measure (fat)."""
    n = mu.shape[0]
    ambition = rng.uniform(0.0, 1.5, size=(n, 1))
    A = mu[:, None] + ambition + rng.uniform(-1.5, 1.5, size=(n, R))
    verdict = rng.random((n, R)) < sigmoid(2.0 * (mu[:, None] - A - body_penalty(A, absolute)))
    M = (np.arange(R)[None, :] < shown[:, None]).astype(float)
    return A * M, verdict * M, M


WORLDS = {"native": {"lead": (-1.0, 1.0), "absolute": False, "max_shown": R},
          "shifted": {"lead": (3.0, 5.0), "absolute": False, "max_shown": R},
          "absolute": {"lead": (-1.5, 1.5), "absolute": True, "max_shown": 3}}


def make_split(n, rng, world):
    w = WORLDS[world]
    mu_s = rng.uniform(*w["lead"], size=n)
    is_self = rng.random(n) < 0.4
    if w["absolute"]:
        mu_c = rng.normal(0.0, 0.8, size=n)
    else:
        mu_c = mu_s + np.clip(rng.normal(0.0, 1.0, size=n), -2.5, 2.5)
    mu_e = np.where(is_self, mu_s, mu_c)
    a = mu_s + rng.uniform(-2.5, 2.5, size=n)
    t = rng.normal(0.0, 1.0, size=(n, T))
    d = 0.6 * np.tanh(1.2 * t[:, 0]) - 0.4 * np.tanh(t[:, 1] * t[:, 2])
    p = a - mu_s + rng.normal(0.0, 0.15, size=n)
    As, Os, Ms = deed_record(mu_s, np.full(n, R), rng, w["absolute"])
    Ac, Oc, Mc = deed_record(mu_e, rng.integers(0, w["max_shown"] + 1, size=n), rng, w["absolute"])
    s = is_self[:, None]
    Ae, Oe, Me = np.where(s, As, Ac), np.where(s, Os, Oc), np.where(s, Ms, Mc)
    y = (mu_e - a - d - body_penalty(a, w["absolute"]) > 0.0).astype(float)
    X = np.concatenate([p[:, None], t, s.astype(float), As, Os, Ms, Ae, Oe, Me], axis=1)
    return {"X": X, "y": y, "is_self": is_self}


def make_data(seed, n_train, n_eval):
    g = lambda k: child_rng(seed, 100, k)
    return {"train": make_split(n_train, g(0), "native"), "val": make_split(n_eval // 2, g(1), "native"),
            "test": make_split(n_eval, g(2), "native"), "shifted": make_split(n_eval, g(3), "shifted"),
            "abs_train": make_split(n_train, g(4), "absolute"), "abs_val": make_split(n_eval // 2, g(5), "absolute"),
            "abs_test": make_split(n_eval, g(6), "absolute")}


# ----------------------------------------------------------------------------- model
class Model:
    """Every trainable tensor lives in `params`; `knock` names ablated modules."""

    def __init__(self, kind, params, cfg):
        self.kind, self.params, self.cfg = kind, params, cfg
        self.knock, self.g_mean = {}, 0.0


def _dense(rng, fan_in, fan_out):
    return rng.normal(0.0, 1.0 / math.sqrt(fan_in), size=(fan_in, fan_out)), np.zeros(fan_out)


def init_qadr(rng, absolute, hd=24, h2=16, ht=12):
    P = {}
    P["W1"], P["b1"] = _dense(rng, 5, hd)
    P["W2"], P["b2"] = _dense(rng, hd + 1, h2)
    P["W3"], P["b3"] = _dense(rng, h2, 1)
    P["W4"], P["b4"] = 0.1 * _dense(rng, hd + 1, 1)[0], np.zeros(1)
    P["theta_p"] = np.array(0.0)
    P["W5"], P["b5"] = _dense(rng, T, ht)
    P["W6"], P["b6"] = _dense(rng, ht, 1)
    P["kappa_r"] = np.array(rng.normal(0.0, 0.1))
    if absolute:
        P["theta_pop"], P["m_pop"] = np.array(0.0), np.array(0.0)
        P["B"], P["lam_r"] = np.array(1.5), np.array(-2.0)
    return P


def _read_measure(P, A, O, M, cfg, knock):
    """Capability and its precision from a record of deeds. Sizes enter only as deviations
    from the record's own mean, so adding c to every absolute size adds c to the estimate
    and changes nothing else: the witnesses report the measure, not the observer's eye."""
    if cfg.get("ignore_mask"):
        M = np.ones_like(M)
    N = A.shape[0]
    n = M.sum(axis=1)
    nc = np.maximum(n, 1.0)
    abar = (A * M).sum(axis=1) / nc
    c = (A - abar[:, None]) * M
    Phi = np.stack([c, O * M, c * O, c * c, M], axis=2)
    T1 = np.tanh((Phi.reshape(-1, 5) @ P["W1"]).reshape(N, R, -1) + P["b1"])
    H1 = T1 * M[:, :, None]
    if cfg.get("positional"):
        H1 = H1 * (1.0 + 0.5 * np.arange(R) / R)[None, :, None]
    Q = np.concatenate([H1.sum(axis=1) / nc[:, None], (n / R)[:, None]], axis=1)
    T2 = np.tanh(Q @ P["W2"] + P["b2"])
    offset = (T2 @ P["W3"] + P["b3"])[:, 0]
    if knock.get("fil_reader") == "zero":
        offset = np.zeros_like(offset)
    if MUTANT == "break_equivariance":
        offset = offset + 0.05 * abar * abar  # a linear leak would cancel in every relative comparison
    U4 = (Q @ P["W4"] + P["b4"])[:, 0]
    tau = (n / R) * softplus(U4)
    return abar + offset, tau, (M, n, nc, Phi, T1, Q, T2, U4)


def _read_measure_back(P, cache, d_mu, d_tau, G):
    M, n, nc, Phi, T1, Q, T2, U4 = cache
    G["W3"] += T2.T @ d_mu[:, None]
    G["b3"] += np.sum(d_mu, keepdims=True)
    dU2 = (d_mu[:, None] @ P["W3"].T) * (1.0 - T2 * T2)
    G["W2"] += Q.T @ dU2
    G["b2"] += dU2.sum(axis=0)
    dU4 = d_tau * (n / R) * sigmoid(U4)
    G["W4"] += Q.T @ dU4[:, None]
    G["b4"] += np.sum(dU4, keepdims=True)
    dQ = dU2 @ P["W2"].T + dU4[:, None] @ P["W4"].T
    dU1 = (dQ[:, :-1] / nc[:, None])[:, None, :] * M[:, :, None] * (1.0 - T1 * T1)
    G["W1"] += Phi.reshape(-1, 5).T @ dU1.reshape(-1, dU1.shape[2])
    G["b1"] += dU1.sum(axis=(0, 1))


def qadr_forward(model, X):
    P, cfg, kn = model.params, model.cfg, model.knock
    p, t, s = X[:, COL["p"]], X[:, COL["t"]], X[:, COL["self"]]
    mu_s, _, cache_s = _read_measure(P, X[:, COL["As"]], X[:, COL["Os"]], X[:, COL["Ms"]], cfg, kn)
    mu_e, tau_e, cache_e = _read_measure(P, X[:, COL["Ae"]], X[:, COL["Oe"]], X[:, COL["Me"]], cfg, kn)
    # With no witnesses the only anchor a scale-free mind owns is itself: projection.
    tau0 = 1e-3 if kn.get("projection_prior") == "zero" else softplus(P["theta_p"]) + 1e-3
    num, den = tau0 * mu_s + tau_e * mu_e, tau0 + tau_e
    tau_pop = None
    if cfg["absolute"]:
        tau_pop = softplus(P["theta_pop"])
        num, den = num + tau_pop * P["m_pop"], den + tau_pop
    mix = num / den
    mu_t = s * mu_s + (1.0 - s) * mix
    r = p if kn.get("ayn_remeasure") == "identity" else p + mu_s - mu_t
    Tt = np.tanh(t @ P["W5"] + P["b5"])
    g = (Tt @ P["W6"] + P["b6"])[:, 0]
    if kn.get("type_path") == "mean":
        g = np.full_like(g, model.g_mean)
    z = -P["kappa_r"] * (r + g)  # the form is given; the sign of the verdict is learned
    ub = None
    if cfg["absolute"]:
        ub = 3.0 * (p + mu_s - P["B"])
        z = z - softplus(P["lam_r"]) * softplus(ub) / 3.0
    return z, {"cs": cache_s, "ce": cache_e, "mu_s": mu_s, "mu_e": mu_e, "tau_e": tau_e, "tau0": tau0,
               "tau_pop": tau_pop, "den": den, "mix": mix, "mu_t": mu_t, "r": r, "g": g, "Tt": Tt, "ub": ub}


def qadr_loss_and_grads(model, batch):
    X, y = batch
    P = model.params
    z, C = qadr_forward(model, X)
    N = X.shape[0]
    loss = float(np.mean(softplus(z) - y * z))
    dz = (sigmoid(z) - y) / N
    G = {k: np.zeros_like(v) for k, v in P.items()}
    G["kappa_r"] += np.sum(-dz * (C["r"] + C["g"]))
    dv = -P["kappa_r"] * dz
    G["W6"] += C["Tt"].T @ dv[:, None]
    G["b6"] += np.sum(dv, keepdims=True)
    dUt = (dv[:, None] @ P["W6"].T) * (1.0 - C["Tt"] ** 2)
    G["W5"] += X[:, COL["t"]].T @ dUt
    G["b5"] += dUt.sum(axis=0)
    s = X[:, COL["self"]]
    d_mu_s = dv * (1.0 - s)
    d_mix = -dv * (1.0 - s)
    den, mix = C["den"], C["mix"]
    d_mu_s += d_mix * C["tau0"] / den
    d_mu_e = d_mix * C["tau_e"] / den
    d_tau_e = d_mix * (C["mu_e"] - mix) / den
    G["theta_p"] += np.sum(d_mix * (C["mu_s"] - mix) / den) * sigmoid(P["theta_p"])
    if model.cfg["absolute"]:
        G["theta_pop"] += np.sum(d_mix * (P["m_pop"] - mix) / den) * sigmoid(P["theta_pop"])
        G["m_pop"] += np.sum(d_mix * C["tau_pop"] / den)
        G["lam_r"] += np.sum(-dz * softplus(C["ub"]) / 3.0) * sigmoid(P["lam_r"])
        d_ub = -dz * softplus(P["lam_r"]) * sigmoid(C["ub"]) / 3.0
        G["B"] += np.sum(-3.0 * d_ub)
        d_mu_s += 3.0 * d_ub
    _read_measure_back(P, C["cs"], d_mu_s, np.zeros(N), G)
    _read_measure_back(P, C["ce"], d_mu_e, d_tau_e, G)
    if MUTANT == "zero_grad_W2":
        G["W2"][:] = 0.0
    return loss, G


# ----------------------------------------------------------------------------- baselines
def init_mlp(rng, in_dim, h=10):
    P = {}
    P["V1"], P["c1"] = _dense(rng, in_dim, h)
    P["V2"], P["c2"] = _dense(rng, h, h)
    P["V3"], P["c3"] = _dense(rng, h, 1)
    return P


def mlp_forward(model, X):
    P = model.params
    H1 = np.tanh(X @ P["V1"] + P["c1"])
    H2 = np.tanh(H1 @ P["V2"] + P["c2"])
    return (H2 @ P["V3"] + P["c3"])[:, 0], (H1, H2)


def mlp_loss_and_grads(model, batch):
    X, y = batch
    P = model.params
    z, (H1, H2) = mlp_forward(model, X)
    loss = float(np.mean(softplus(z) - y * z))
    dz = (sigmoid(z) - y) / X.shape[0]
    d2 = (dz[:, None] @ P["V3"].T) * (1.0 - H2 * H2)
    d1 = (d2 @ P["V2"].T) * (1.0 - H1 * H1)
    return loss, {"V3": H2.T @ dz[:, None], "c3": np.sum(dz, keepdims=True), "V2": H1.T @ d2,
                  "c2": d2.sum(axis=0), "V1": X.T @ d1, "c1": d1.sum(axis=0)}


# ----------------------------------------------------------------------------- common interface
def build_model(in_dim, out_dim, task_type, rng, **cfg):
    if task_type not in TASK_TYPES or out_dim != 2:
        raise ValueError("this file supports binary vector_classification only")
    cfg = {"kind": "qadr", "absolute": False, **cfg}
    if cfg["kind"] == "mlp":
        return Model("mlp", init_mlp(rng, in_dim, cfg.get("hidden", 10)), cfg)
    if in_dim != IN_DIM:
        raise ValueError(f"QADR reads the {IN_DIM}-column record layout in COL; got {in_dim}")
    return Model("qadr", init_qadr(rng, cfg["absolute"]), cfg)


def logits(model, X):
    return (qadr_forward if model.kind == "qadr" else mlp_forward)(model, X)[0]


def loss_and_grads(model, batch):
    return (qadr_loss_and_grads if model.kind == "qadr" else mlp_loss_and_grads)(model, batch)


def batch_loss(model, batch):
    z = logits(model, batch[0])
    return float(np.mean(softplus(z) - batch[1] * z))


def predict(model, X):
    q = sigmoid(logits(model, X))
    return np.stack([1.0 - q, q], axis=1)


def hidden_states(model, X):
    if model.kind == "mlp":
        return dict(zip(("h1", "h2"), mlp_forward(model, X)[1]))
    _, C = qadr_forward(model, X)
    return {"self_measure": C["mu_s"], "executor_measure": C["mu_t"], "relative_margin": C["r"], "type_offset": C["g"]}


# ----------------------------------------------------------------------------- registries
def modules(model):
    if model.kind == "mlp":
        return {"mlp": {"params": ["V1", "c1", "V2", "c2", "V3", "c3"], "role": "two-layer tanh perceptron", "signature": False}}
    reg = {"ayn_remeasure": {"params": [], "role": "frame conversion of task size from the observer's scale to the executor's", "signature": True},
           "fil_reader": {"params": ["W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4"],
                          "role": "shift-equivariant set encoder estimating capability and precision from outcome records", "signature": True},
           "projection_prior": {"params": ["theta_p"], "role": "precision-weighted prior centring others' capability on the observer's own", "signature": True},
           "type_path": {"params": ["W5", "b5", "W6", "b6"], "role": "MLP mapping nuisance task features to a difficulty offset", "signature": False},
           "law_head": {"params": ["kappa_r"], "role": "logistic link on the relative margin with learned direction", "signature": False}}
    if model.cfg["absolute"]:
        reg["absolute_register"] = {"params": ["theta_pop", "m_pop", "B", "lam_r"],
                                    "role": "absolute-unit population prior and body-capacity penalty", "signature": False}
    return reg


KNOCKOUTS = [("ayn_remeasure", "identity"), ("fil_reader", "zero"), ("projection_prior", "zero"), ("type_path", "mean")]


def knockout(model, name, mode):
    twin = Model(model.kind, {k: v.copy() for k, v in model.params.items()}, dict(model.cfg))
    twin.knock, twin.g_mean = {**model.knock, name: mode}, model.g_mean
    return twin


def n_params(model):
    return int(sum(np.size(v) for v in model.params.values()))


MUTANTS = {"sign_flip_update": "C3",      # the optimiser climbs the loss
           "zero_lr": "C3",               # no update reaches the parameters
           "zero_grad_W2": "C1",          # one tensor's analytic gradient is lost
           "break_equivariance": "C6"}    # the reader leaks absolute size (quadratically) into the estimate


def set_mutant(name):
    global MUTANT
    MUTANT = name


# ----------------------------------------------------------------------------- training
def fit(model, data, budget, rng, lr=3e-3, batch=128, train="train", val="val"):
    """Adam under a fixed update budget; the validation split alone selects the kept weights."""
    X, y = data[train]["X"], data[train]["y"]
    opt = Adam(model.params)
    step_lr = {"zero_lr": 0.0, "sign_flip_update": -lr}.get(MUTANT, lr)
    best_val, best, trace = math.inf, None, []
    for step in range(budget):
        idx = rng.integers(0, X.shape[0], size=batch)
        loss, grads = loss_and_grads(model, (X[idx], y[idx]))
        if not math.isfinite(loss):
            raise FloatingPointError("non-finite training loss")
        grads, _ = clip_global_norm(grads, GRAD_CLIP)
        opt.step(model.params, grads, step_lr)
        trace.append(loss)
        if (step + 1) % 100 == 0 or step + 1 == budget:
            v = batch_loss(model, (data[val]["X"], data[val]["y"]))
            if v < best_val:
                best_val, best = v, {k: w.copy() for k, w in model.params.items()}
    if best is not None:
        model.params = best
    if model.kind == "qadr":
        model.g_mean = float(np.mean(hidden_states(model, X)["type_offset"]))
    return {"trace": np.array(trace), "best_val": best_val}


def accuracy(model, split, mask=None):
    hit = (logits(model, split["X"]) > 0.0) == (split["y"] > 0.5)
    return float(hit[mask].mean() if mask is not None else hit.mean())


def trivial_accuracy(split):
    m = float(split["y"].mean())
    return max(m, 1.0 - m)


SPECS = {"qadr": ({"kind": "qadr"}, "train", "val"), "mlp": ({"kind": "mlp"}, "train", "val"),
         "qadr_abs_world": ({"kind": "qadr"}, "abs_train", "abs_val"),
         "mlp_abs_world": ({"kind": "mlp"}, "abs_train", "abs_val"),
         "amended_abs_world": ({"kind": "qadr", "absolute": True}, "abs_train", "abs_val")}


def train_seed(seed, budget):
    data = make_data(seed, 6000, 3000)
    models, info = {}, {}
    for k, (name, (cfg, tr, va)) in enumerate(SPECS.items()):
        m = build_model(IN_DIM, 2, "vector_classification", child_rng(seed, 1, k), **cfg)
        start = batch_loss(m, (data[tr]["X"], data[tr]["y"]))
        fit(m, data, budget, child_rng(seed, 2, k), train=tr, val=va)
        info[name] = {"start_loss": start, "end_loss": batch_loss(m, (data[tr]["X"], data[tr]["y"]))}
        models[name] = m
    return data, models, info


def seed_metrics(data, models):
    q, m, qa, ma, am = (models[k] for k in SPECS)
    te, sh, ab = data["test"], data["shifted"], data["abs_test"]
    comp = ~te["is_self"]
    acc = {"qadr/test": accuracy(q, te), "mlp/test": accuracy(m, te),
           "qadr/test_companions": accuracy(q, te, comp), "mlp/test_companions": accuracy(m, te, comp),
           "qadr/shifted": accuracy(q, sh), "mlp/shifted": accuracy(m, sh),
           "qadr/abs": accuracy(qa, ab), "mlp/abs": accuracy(ma, ab), "amended/abs": accuracy(am, ab),
           "qadr/abs_self": accuracy(qa, ab, ab["is_self"]), "mlp/abs_self": accuracy(ma, ab, ab["is_self"]),
           "amended/abs_self": accuracy(am, ab, ab["is_self"]),
           "qadr/abs_companions": accuracy(qa, ab, ~ab["is_self"]), "mlp/abs_companions": accuracy(ma, ab, ~ab["is_self"]),
           "amended/abs_companions": accuracy(am, ab, ~ab["is_self"]),
           "trivial/test": trivial_accuracy(te), "trivial/shifted": trivial_accuracy(sh), "trivial/abs": trivial_accuracy(ab)}
    knock = {f"{n}:{mode}": accuracy(knockout(q, n, mode), te) for n, mode in KNOCKOUTS}
    diff = {"H-SIG": acc["qadr/shifted"] - acc["mlp/shifted"],
            "H-NEC": knock["ayn_remeasure:identity"] - knock["type_path:mean"],
            "H-BLIND": acc["qadr/abs"] - acc["mlp/abs"], "H-AMEND": acc["amended/abs"] - acc["qadr/abs"]}
    return {"acc": acc, "knock": knock, "diff": diff}


# ----------------------------------------------------------------------------- correctness tests
def c1_gradcheck(data, seed, kinds=("qadr", "amended_abs_world", "mlp"), steps=60):
    worst, checks, ok = 0.0, 0, True
    for j, name in enumerate(kinds):
        cfg, tr, va = SPECS[name]
        model = build_model(IN_DIM, 2, "vector_classification", child_rng(seed, 11, j), **cfg)
        batch = (data[tr]["X"][:24], data[tr]["y"][:24])
        for stage in ("init", "trained"):
            if stage == "trained":
                fit(model, data, steps, child_rng(seed, 12, j), train=tr, val=va)
            _, grads = loss_and_grads(model, batch)
            rep = finite_difference_check(lambda: batch_loss(model, batch), model.params, grads, child_rng(seed, 13, j))
            checks += len(rep)
            worst = max([worst] + [r["max_rel"] for r in rep.values()])
            ok = ok and all(r["passed"] for r in rep.values())
    return ok, {"tensors_checked": checks, "tensors_total": checks, "max_rel_error": worst,
                "checked_at": ["init", f"after_{steps}_training_steps"]}


def c2_determinism(data, seed):
    runs = []
    for _ in range(2):
        m = build_model(IN_DIM, 2, "vector_classification", child_rng(seed, 21), kind="qadr")
        tr = fit(m, data, 40, child_rng(seed, 22))
        runs.append((tr["trace"], logits(m, data["test"]["X"][:300]), m))
    same = bool(np.array_equal(runs[0][0], runs[1][0]) and np.array_equal(runs[0][1], runs[1][1]))
    finite = all(bool(np.all(np.isfinite(v))) for v in runs[0][2].params.values()) and bool(np.all(np.isfinite(runs[0][1])))
    return same and finite, f"identical traces and logits {same}; finite {finite}"


def c3_learning(info, metrics):
    th = MIND_CARD["thresholds"]
    s, e = info["qadr"]["start_loss"], info["qadr"]["end_loss"]
    margin = metrics["acc"]["qadr/test"] - metrics["acc"]["trivial/test"]
    ok = (s - e) / s >= th["loss_drop_fraction"] and margin >= th["margin_over_trivial"]
    return ok, (f"train loss {s:.3f} -> {e:.3f}; held-out {metrics['acc']['qadr/test']:.3f} "
                f"vs trivial {metrics['acc']['trivial/test']:.3f}")


def c3_short(data, seed, steps):
    th = MIND_CARD["thresholds"]
    m = build_model(IN_DIM, 2, "vector_classification", child_rng(seed, 31), kind="qadr")
    whole = (data["train"]["X"], data["train"]["y"])
    s = batch_loss(m, whole)
    try:
        fit(m, data, steps, child_rng(seed, 32))
    except FloatingPointError:
        return False
    margin = accuracy(m, data["test"]) - trivial_accuracy(data["test"])
    return (s - batch_loss(m, whole)) / s >= th["loss_drop_fraction"] and margin >= th["margin_over_trivial"]


def c4_shuffled(data, seed, budget):
    shuffled = dict(data)
    for k, key in ((41, "train"), (42, "val")):
        order = child_rng(seed, k).permutation(data[key]["y"].shape[0])
        shuffled[key] = {**data[key], "y": data[key]["y"][order]}
    m = build_model(IN_DIM, 2, "vector_classification", child_rng(seed, 43), kind="qadr")
    prior_acc = accuracy(m, data["test"])  # the untrained structure: the fixed-sign law alone
    fit(m, shuffled, budget, child_rng(seed, 44))
    te = data["test"]
    base = float(np.clip(data["train"]["y"].mean(), 1e-6, 1 - 1e-6))
    trivial_ll = float(-np.mean(te["y"] * math.log(base) + (1 - te["y"]) * math.log(1 - base)))
    ll, band = batch_loss(m, (te["X"], te["y"])), MIND_CARD["thresholds"]["shuffled_band_nats"]
    return abs(ll - trivial_ll) <= band, (f"held-out log-loss {ll:.3f} vs trivial {trivial_ll:.3f} (band {band}); "
                                          f"accuracy {accuracy(m, te):.3f}; untrained structure {prior_acc:.3f}")


def _perturbed(seed, k, cfg, rng):
    m = build_model(IN_DIM, 2, "vector_classification", child_rng(seed, 60, k), **cfg)
    for key in m.params:
        m.params[key] = m.params[key] + rng.normal(0.0, 0.5, size=np.shape(m.params[key]))
    return m


def _shift_sizes(X, c):
    Y = X.copy()
    Y[:, COL["As"]] += c * Y[:, COL["Ms"]]
    Y[:, COL["Ae"]] += c * Y[:, COL["Me"]]
    return Y


def _permute_records(X, rng):
    Y = X.copy()
    for blocks in (("As", "Os", "Ms"), ("Ae", "Oe", "Me")):
        order = np.argsort(rng.random((X.shape[0], R)), axis=1)
        for b in blocks:
            Y[:, COL[b]] = np.take_along_axis(X[:, COL[b]], order, axis=1)
    return Y


def _scramble_hidden(X, rng):
    Y = X.copy()
    for a, o, m in (("As", "Os", "Ms"), ("Ae", "Oe", "Me")):
        hole = Y[:, COL[m]] == 0.0
        sizes, verdicts = Y[:, COL[a]], Y[:, COL[o]]
        sizes[hole] = rng.uniform(-10.0, 10.0, size=int(hole.sum()))
        verdicts[hole] = rng.integers(0, 2, size=int(hole.sum()))
    return Y


def c6_properties(data, seed):
    """Invariants on random parameters and inputs, each with a variant that must break it."""
    rng = child_rng(seed, 61)
    X = np.concatenate([data["test"]["X"][:200], data["shifted"]["X"][:100]])

    def gap(m, Y):
        return float(np.max(np.abs(logits(m, Y) - logits(m, X))))

    def law(m):  # the percept may enter only as a margin against a measure: slope exactly -kappa
        h = rng.uniform(-3.0, 3.0, size=X.shape[0])
        Y = X.copy()
        Y[:, COL["p"]] += h
        return float(np.max(np.abs(logits(m, Y) - logits(m, X) + m.params["kappa_r"] * h)))

    tests = {"EQV": (lambda m: max(gap(m, _shift_sizes(X, c)) for c in rng.uniform(-20.0, 20.0, 4)),
                     {"kind": "qadr"}, {"kind": "qadr", "absolute": True}, None, 1e-8),
             "PERM": (lambda m: gap(m, _permute_records(X, rng)), {"kind": "qadr"}, {"kind": "qadr", "positional": True}, None, 1e-10),
             "MASK": (lambda m: gap(m, _scramble_hidden(X, rng)), {"kind": "qadr"}, {"kind": "qadr", "ignore_mask": True}, None, 1e-10),
             "LAW": (law, {"kind": "qadr"}, {"kind": "qadr", "absolute": True}, None, 1e-9)}
    res, ok = {}, True
    for i, (name, (violation, cfg, control, _, tol)) in enumerate(tests.items()):
        v_model = max(violation(_perturbed(seed, 10 * i + k, cfg, rng)) for k in range(5))
        v_control = max(violation(_perturbed(seed, 10 * i + 5 + k, control, rng)) for k in range(5))
        res[name] = {"violation": v_model, "negative_control": v_control, "passed": v_model <= tol and v_control > 1e-3}
        ok = ok and res[name]["passed"]
    q = _perturbed(seed, 90, {"kind": "qadr"}, rng)
    own = X[:, COL["self"]] == 1.0
    g = float(np.max(np.abs(hidden_states(q, X)["relative_margin"][own] - X[own, COL["p"]])))
    res["DEF"] = {"violation": g, "note": "definition check (self read as self), not evidence", "passed": g <= 1e-9}
    return ok and res["DEF"]["passed"], res


def c5_mutants(data, seed, steps, c6_ok):
    sanity = c6_ok and c3_short(data, seed, steps) and c1_gradcheck(data, seed, kinds=("qadr",))[0]
    detected = []
    for name, test in MUTANTS.items():
        set_mutant(name)
        try:
            if test == "C3":
                passed = c3_short(data, seed, steps)
            elif test == "C1":
                passed = c1_gradcheck(data, seed, kinds=("qadr",))[0]
            else:
                passed = c6_properties(data, seed)[0]
        finally:
            set_mutant(None)
        if not passed:
            detected.append(name)
    score = len(detected) / len(MUTANTS)
    return sanity and score == 1.0, {"detected": len(detected), "total": len(MUTANTS), "score": score,
                                     "unmutated_checks_pass": sanity}


def c7_split_integrity(data):
    digest = lambda split: {hashlib.sha1(np.round(row, 12).astype(float).tobytes() + b"").hexdigest()
                            for row in (split["X"] + 0.0)}
    train = digest(data["train"]) | digest(data["abs_train"])
    evals = set().union(*(digest(data[k]) for k in ("val", "test", "shifted", "abs_val", "abs_test")))
    shared = len(train & evals)
    return shared == 0, f"{shared} rows shared between training and evaluation splits"


# ----------------------------------------------------------------------------- hypotheses and report
def evaluate_hypotheses(per_seed, rng, evaluated):
    rows = []
    for h in MIND_CARD["hypotheses"]:
        mean, lo, hi = paired_bootstrap_ci([s["diff"][h["id"]] for s in per_seed], rng)
        verdict = hypothesis_verdict(mean, lo, hi, h["mesi"], h["direction"]) if evaluated else "not evaluated"
        rows.append({"id": h["id"], "metric": h["metric"], "mean_diff": mean, "ci95": [lo, hi], "mesi": h["mesi"],
                     "n_seeds": len(per_seed), "verdict": verdict})
    return rows


def knockout_table(per_seed, registry, rng):
    rows = []
    for name, mode in KNOCKOUTS:
        mean, lo, hi = paired_bootstrap_ci([s["knock"][f"{name}:{mode}"] - s["acc"]["qadr/test"] for s in per_seed], rng)
        rows.append({"module": name, "mode": mode, "signature": registry[name]["signature"], "metric_change": mean, "ci95": [lo, hi]})
    return rows


def format_report(rep):
    a = rep["accuracy_mean"]
    L = [f"=== VERIFIED REPORT · chapter {CHAPTER:04d} ===",
         f"environment: python {rep['environment']['python']} | numpy {rep['environment']['numpy']}",
         f"mode: {rep['mode']} | seeds: {rep['seeds']} | updates per model: {rep['budget']}",
         f"runtime_s: {rep['runtime_s']:.1f}",
         "n_params: " + ", ".join(f"{k} {v}" for k, v in rep["n_params_by_model"].items()),
         (f"gradcheck: {rep['gradcheck']['tensors_checked']} tensor checks ({', '.join(rep['gradcheck']['checked_at'])}); "
          f"max rel error {rep['gradcheck']['max_rel_error']:.2e}; passed {rep['gradcheck']['passed']}"),
         "correctness:"]
    L += [f"  {c['id']:<3} {c['name']:<24} {'PASS' if c['passed'] else 'FAIL'}  {c['detail']}" for c in rep["correctness"]]
    mu = rep["mutants"]
    L.append(f"mutation score: {mu['detected']}/{mu['total']} = {mu['score']:.2f}")
    L.append("held-out accuracy, mean over seeds:")
    L.append(f"  native test          qadr {a['qadr/test']:.3f}  mlp {a['mlp/test']:.3f}  trivial {a['trivial/test']:.3f}")
    L.append(f"  native companions    qadr {a['qadr/test_companions']:.3f}  mlp {a['mlp/test_companions']:.3f}")
    L.append(f"  shifted measures     qadr {a['qadr/shifted']:.3f}  mlp {a['mlp/shifted']:.3f}  trivial {a['trivial/shifted']:.3f}")
    for tag, key in (("absolute world      ", "abs"), ("absolute, self      ", "abs_self"), ("absolute, companions", "abs_companions")):
        L.append(f"  {tag} qadr {a['qadr/' + key]:.3f}  mlp {a['mlp/' + key]:.3f}  amended {a['amended/' + key]:.3f}")
    L.append(f"  absolute trivial     {a['trivial/abs']:.3f}")
    L.append("hypotheses:")
    L += [(f"  {h['id']:<8} {h['metric']}  mean {h['mean_diff']:+.4f}  ci95 [{h['ci95'][0]:+.4f}, {h['ci95'][1]:+.4f}]"
           f"  mesi {h['mesi']:.2f}  n={h['n_seeds']}  {h['verdict']}") for h in rep["hypotheses"]]
    L.append("knockouts (change in native held-out accuracy):")
    L += [(f"  {k['module']:<17} {k['mode']:<9} signature={str(k['signature']):<5} {k['metric_change']:+.4f}"
           f"  ci95 [{k['ci95'][0]:+.4f}, {k['ci95'][1]:+.4f}]") for k in rep["knockouts"]]
    L += [f"task types: {', '.join(rep['task_types'])}", f"exit code: {rep['exit_code']}", "=== END REPORT ==="]
    return "\n".join(L)


def data_bridge(path, model):
    if not path:
        print("real-data bridge: no --data PATH given; skipped")
        return
    arr = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if arr.shape[1] != IN_DIM + 1:
        raise ValueError(f"--data expects {IN_DIM} layout columns plus a 0/1 label; got {arr.shape[1]}")
    acc = accuracy(model, {"X": arr[:, :IN_DIM], "y": arr[:, IN_DIM]})
    print(f"real-data bridge: {os.path.basename(path)} accuracy {acc:.3f} on {arr.shape[0]} rows (first-seed QADR)")


def run(args):
    t0 = time.time()
    quick = args.quick or bool(args.mutant)
    seeds = [args.seed + k for k in range(1 if quick else (args.seeds or 5))]
    budget = 600 if quick else 2500
    if args.mutant:
        set_mutant(args.mutant)
    print(f"chapter {CHAPTER:04d} · {'quick' if quick else 'full'} protocol · seeds {seeds} · mutant {MUTANT}", flush=True)
    per_seed, first = [], None
    for s in seeds:
        data, models, info = train_seed(s, budget)
        per_seed.append(seed_metrics(data, models))
        first = first or (data, models, info)
        print(f"  seed {s}: " + "  ".join(f"{k} {v:+.4f}" for k, v in per_seed[-1]["diff"].items()), flush=True)
    data, models, info = first
    s0 = seeds[0]
    ok1, grad = c1_gradcheck(data, s0)
    ok2, d2 = c2_determinism(data, s0)
    ok3, d3 = c3_learning(info, per_seed[0])
    ok4, d4 = c4_shuffled(data, s0, budget)
    ok6, props = c6_properties(data, s0)
    if args.mutant:
        ok5, mut = True, {"detected": 0, "total": len(MUTANTS), "score": 0.0, "unmutated_checks_pass": None}
    else:
        ok5, mut = c5_mutants(data, s0, 800, ok6)
    ok7, d7 = c7_split_integrity(data)
    rng = np.random.default_rng(s0)
    hyp = evaluate_hypotheses(per_seed, rng, evaluated=not quick)
    kos = knockout_table(per_seed, modules(models["qadr"]), rng)
    data_bridge(args.data, models["qadr"])
    runtime = time.time() - t0
    ok8 = runtime <= (QUICK_BUDGET_S if quick else FULL_BUDGET_S)
    prop_detail = "; ".join(f"{k} {v['violation']:.1e}" + (f" (control {v['negative_control']:.1e})" if "negative_control" in v else "")
                            for k, v in props.items())
    correctness = [
        {"id": "C1", "name": "gradient_check", "passed": ok1, "detail": f"max rel error {grad['max_rel_error']:.2e}"},
        {"id": "C2", "name": "determinism_finiteness", "passed": ok2, "detail": d2},
        {"id": "C3", "name": "learning", "passed": ok3, "detail": d3},
        {"id": "C4", "name": "shuffled_label_control", "passed": ok4, "detail": d4},
        {"id": "C5", "name": "mutant_detection", "passed": ok5, "detail": "skipped under --mutant" if args.mutant else f"score {mut['score']:.2f}; unmutated checks pass {mut['unmutated_checks_pass']}"},
        {"id": "C6", "name": "property_tests", "passed": ok6, "detail": prop_detail},
        {"id": "C7", "name": "split_integrity", "passed": ok7, "detail": d7},
        {"id": "C8", "name": "budget", "passed": ok8, "detail": f"{runtime:.1f} s of {QUICK_BUDGET_S if quick else FULL_BUDGET_S:.0f} s"}]
    exit_code = 0 if all(c["passed"] for c in correctness[:7]) else 1
    if exit_code == 0 and not ok8:
        exit_code = 3
    accuracy_mean = {k: float(np.mean([s["acc"][k] for s in per_seed])) for k in per_seed[0]["acc"]}
    report = {"schema_version": "1.0", "chapter": CHAPTER, "file": os.path.basename(__file__),
              "card_revision": MIND_CARD["card_revision"],
              "environment": {"python": sys.version.split()[0], "numpy": np.__version__}, "seeds": seeds,
              "runtime_s": runtime, "n_params": n_params(models["qadr"]), "gradcheck": {**grad, "passed": ok1},
              "correctness": correctness, "mutants": {k: mut[k] for k in ("detected", "total", "score")},
              "hypotheses": hyp, "knockouts": kos, "task_types": TASK_TYPES, "exit_code": exit_code,
              "mode": "quick" if quick else "full", "budget": budget, "properties": props, "accuracy_mean": accuracy_mean,
              "n_params_by_model": {"qadr": n_params(models["qadr"]), "mlp": n_params(models["mlp"]),
                                    "amended": n_params(models["amended_abs_world"])}}
    print(format_report(report), flush=True)
    if args.json:
        write_report(report, args.json)
    return exit_code


def main(argv=None):
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__), description="QADR, chapter 0240: full protocol by default")
    ap.add_argument("--quick", action="store_true", help="one seed, reduced updates, all correctness tests")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--json", metavar="PATH")
    ap.add_argument("--card", action="store_true")
    ap.add_argument("--mutant", choices=sorted(MUTANTS))
    ap.add_argument("--data", metavar="PATH")
    args = ap.parse_args(argv)
    if args.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    if args.seed < 0 or (args.seeds is not None and args.seeds < 1):
        ap.error("--seed must be >= 0 and --seeds >= 1")
    try:
        return run(args)
    except FloatingPointError as err:
        print(f"non-finite values: {err}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())

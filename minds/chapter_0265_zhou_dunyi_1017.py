#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0265 · Zhou Dunyi (1017-1073)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 14, Minds 261-280 Available on Amazon https://www.amazon.com/dp/B0HLYSVY5D
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0265_zhou_dunyi_1017 - Zhou Dunyi (1017-1073)
# END ATTRIBUTION
"""JI LENS: a cheng-shen-ji circuit (誠·神·幾) for deciding at the incipient.

Thesis
    Good and ill divide at the incipient (ji): a mind held in desireless
    stillness reads motion only while it is still unformed, "between being and
    non-being", and acts on it before it takes form; motion that arrives
    already formed, or that happens after form, is not what it reads.

Evidence and provenance (archetype_provenance = belief; his own texts)
    D1 Tongshu ch.4 (聖): 寂然不動者誠也；感而遂通者神也；動而未形、有無之間者幾也.
    D2 Tongshu ch.3 (誠幾德): 誠無為，幾善惡 - good and ill divide at ji.
    D3 Tongshu ch.9 (思): 幾動於彼，誠動於此 ... 無思而無不通為聖人; he quotes
       the Xici: 君子見幾而作，不俟終日 (act on ji without waiting for day's end).
    D4 Tongshu ch.20 (聖學): 一者無欲也，無欲則靜虛動直；靜虛則明，明則通.
    D5 Taijitu shuo: 五性感動而善惡分; 聖人定之以中正仁義而主靜 (self-note 無欲故靜).
       The text reaches us through Zhu Xi's edition; the Guoshi read the first
       line 自無極而為太極. Nothing here depends on that line.
    D6 Tongshu ch.16: 動而無靜，靜而無動，物也；動而無動，靜而無靜，神也.
    D7 Scholarship (Adler 2014): Cheng Yi and Zhu Xi recast his 主靜 (stillness)
       as 主敬 (composure held through stillness and activity alike).
    D8 Deeds (Song shi 427; Pan Xingsi's epitaph): a career judicial officer who
       settled a long-stalled case at one hearing and inspected remote circuits
       "slowly and carefully"; this shapes the stress on early, cheap acts.
    D9 Scholarship and reading: he says evil arises only with activity and
       leaves its later course unexplained (Thompson, IEP); the division he
       places at the onset leaves no account of things that turn after form.

Doctrine -> mechanism -> test (IDs as in MIND_CARD)
    D1,D6 -> M1 ji_lens: temporal differences of a projection of the
             core-referenced input, passed by a soft gate that is open only if
             BOTH ends of the step lie inside a learned small-amplitude
             neighbourhood of the core                 -> C6.3, H-SIG, H-NEC
    D2    -> M2 ji_trace: leaky accumulation of gated differences; the verdict is
             read from it                              -> H-SIG, H-NEC
    D3    -> M3 commit: even halting hazard, trained on an expected loss priced
             by the form of the thing at the moment of commitment    -> C3
    D4,D5 -> M4 cheng_core + bias-free response: a still reference set at the
             first frame and an odd readout with no standing drive
                                                       -> C6.1, C6.2, D-1
    D7    -> R1 jing rival: same circuit, gate replaced by a learned uniform
             attention over all motion                 -> H-RIVAL
    D9    -> blind world: the class is set by a turn after form     -> H-BLIND

Research question (out-of-distribution detection and shift)
    Does integrating only motion confined to the unformed neighbourhood of a
    still reference let a learner commit earlier and stay correct under shift
    (abrupt formed intrusions, drift, scrambled mature dynamics) better than an
    early-classification RNN that reads form?

Closest prior art and the delta
    ELECTS (Russwurm et al. 2023) is the baseline: a recurrent classifier with a
    learned stopping head and an earliness-accuracy loss (size-matched Elman
    RNN here, on the same core-referenced input). Temporal-difference networks
    (O'Connor and Welling 2016; Neil et al. 2017), RNN onset detectors (Bock et
    al. 2012), CUSUM (Page 1954) and SPRT (Wald 1945) are relatives. Delta:
    evidence enters only as differences whose start and end amplitudes are both
    small (two-endpoint amplitude-inverse gate), so formed motion is excluded
    structurally, and the response is odd with no standing drive.

Blind spot
    If the decisive divergence happens after the thing has formed, the gate is
    closed when it happens. The late-turn world tests this (H-BLIND).

Task (generative process; latent plane mixed into 6 observed channels)
    y ~ {0,1}; onset t0 ~ U{3..12}; initial kick of radius 0.06 at angle
    pi*(1-y) + U(-0.7,0.7); radius grows logistically (rate 0.40) to form 1;
    rotation omega*r^2 and phase noise grow with form; per-sequence rest
    offset, random-walk drift, white noise; with p = 0.6 one formed intrusion
    (abrupt jump of 0.6-1.2 in a random direction, held 2-5 steps, abrupt
    return). The learner may commit at any step; the loss is cross-entropy plus
    kappa (0.4) times the stirring's form at commitment. Shifted split: longer
    (T=48), later onsets, 1-3 larger intrusions, threefold drift, rotation of
    random sign and speed. Blind world: onset angle random; class set by a
    turn of +-0.7 twelve steps after onset; kappa 0.1.

Limits
    Synthetic data built so the delta can matter; a research prototype of one
    AGI-relevant mechanism, not an AGI, and no claim to reproduce a person's mind.
    No felt stillness is modelled; "desirelessness" is only the absence of bias.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 2,
    "revision_log": [
        "rev 1 (2026-09-21): written after trainability and runtime calibration of the task generator "
        "in a scratch script (both models checked for in-distribution learning only; the baseline alone "
        "checked for learnability of the blind world); hypotheses, metrics, directions and mesi fixed from "
        "the doctrine before any hypothesis comparison was run.",
        "rev 2 (2026-09-21, after one --quick run): C1 tolerance 1e-5 -> 1e-4 (section 9.2 allowance). The "
        "failing entries had |grad| ~1.5e-6 and matched to four significant digits; float64 round-off of the "
        "central difference at eps 1e-6 (~5e-11 absolute, loss ~0.7) alone gives relative errors ~1e-5. The "
        "zeroed-gradient mutants still give errors of 1.0. Hypotheses, metrics, directions and mesi unchanged."],
    "generation": {"template_version": "codeguidelines 1.0 (15 September 2026)", "generator": "Claude",
                   "generator_version": "Claude Opus 5", "date": "2026-09-21"},
    "id": 265, "figure": "Zhou Dunyi", "born": 1017, "died": 1073, "civilization": "Chinese (Northern Song)",
    "provenance": "belief",
    "thesis": "Good and ill divide at the incipient: read motion only while it is unformed, relative to a "
              "desireless still core, and act before it takes form.",
    "evidence": [
        {"id": "D1", "claim": "The incipient is motion not yet formed, between being and non-being; cheng is "
         "silent and unmoving, shen penetrates when stimulated.", "basis": "primary", "source": "Tongshu ch.4"},
        {"id": "D2", "claim": "Cheng does not act; good and ill divide at the incipient.", "basis": "primary",
         "source": "Tongshu ch.3"},
        {"id": "D3", "claim": "The incipient stirs there, cheng stirs here; act on the incipient without waiting "
         "for the end of the day.", "basis": "primary", "source": "Tongshu ch.9 (quoting Yijing Xici)"},
        {"id": "D4", "claim": "Unity is having no desire; then empty when still, direct when active; empty "
         "hence clear, clear hence penetrating.", "basis": "primary", "source": "Tongshu ch.20"},
        {"id": "D5", "claim": "When the five natures are stirred good and ill divide; the sage settles affairs "
         "and takes stillness as ruling, self-glossed as desireless.", "basis": "primary",
         "source": "Taijitu shuo (Zhu Xi's text)"},
        {"id": "D6", "claim": "Things are either moving or still; spirit is moving-yet-not-moving.",
         "basis": "primary", "source": "Tongshu ch.16"},
        {"id": "D7", "claim": "Cheng Yi and Zhu Xi recast stillness (jing 靜) as composure (jing 敬) held "
         "through activity and stillness.", "basis": "scholarship", "source": "Adler 2014"},
        {"id": "D8", "claim": "Judicial officer who settled cases early and inspected circuits slowly and "
         "carefully.", "basis": "deeds", "source": "Song shi 427; Pan Xingsi, epitaph (Song wenjian 144)"},
        {"id": "D9", "claim": "Evil is said to arise only with activity and is left unexplained; the onset is "
         "treated as the only place of division.", "basis": "speculation",
         "source": "reading of Tongshu ch.3 with Thompson, IEP 'Zhou Dunyi'"}],
    "research_question": {"category": "out-of-distribution detection and shift",
                          "question": "Does integrating only motion confined to the unformed neighbourhood of a "
                          "still reference let a learner commit early and stay correct under formed intrusions, "
                          "drift and scrambled mature dynamics better than an early-classification RNN?"},
    "mechanism": {
        "name": "JI LENS (cheng-shen-ji circuit)",
        "family": "gated temporal-difference encoding with learned halting (early classification)",
        "signature_modules": ["ji_lens", "ji_trace"],
        "closest_prior_art": ["ELECTS (Russwurm et al. 2023)", "EARLIEST (Hartvigsen et al. 2019)",
                              "Sigma-Delta / Delta networks (O'Connor & Welling 2016; Neil et al. 2017)",
                              "RNN onset detection (Bock et al. 2012)", "CUSUM (Page 1954)", "SPRT (Wald 1945)"],
        "overlap": "Medium",
        "prior_art_queries": ["early classification time series stopping head", "ELECTS end-to-end learned early "
                              "classification", "recurrent gate temporal difference amplitude onset detection",
                              "delta network event-driven temporal difference"],
        "contribution_type": "mechanism",
        "delta": "Differences enter the decision state only if both endpoints lie in a learned small-amplitude "
                 "neighbourhood of a bias-free still core, so formed motion is excluded structurally and the "
                 "readout is odd with no standing drive."},
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1", "property_test": "C6.3", "hypothesis": "H-SIG"},
        {"doctrine": "D2", "mechanism": "M2", "property_test": "C6.2", "hypothesis": "H-NEC"},
        {"doctrine": "D3", "mechanism": "M3", "property_test": "C3", "hypothesis": "H-SIG"},
        {"doctrine": "D4", "mechanism": "M4", "property_test": "C6.1", "hypothesis": "H-SIG"},
        {"doctrine": "D7", "mechanism": "R1", "property_test": "C1", "hypothesis": "H-RIVAL"},
        {"doctrine": "D9", "mechanism": "M1", "property_test": "C6.3", "hypothesis": "H-BLIND"}],
    "hypotheses": [
        {"id": "H-SIG", "statement": "The ji lens commits with higher utility than the size-matched ELECTS RNN "
         "under shift.", "metric": "expected commit utility", "split": "shifted", "comparison": "model - baseline",
         "direction": "greater", "mesi": 0.03, "seeds": 5},
        {"id": "H-NEC", "statement": "Replacing the ji gate by its training mean hurts more than freezing the "
         "still core.", "metric": "expected commit utility", "split": "shifted",
         "comparison": "signature_knockout - matched_knockout", "direction": "less", "mesi": 0.03, "seeds": 5},
        {"id": "H-BLIND", "statement": "Where the class is set by a turn after form, the ji lens falls below the "
         "baseline.", "condition": "late-turn world (kappa 0.1)", "grounding": "division placed only at the "
         "onset (D9)", "metric": "expected commit utility", "split": "held-out",
         "comparison": "model - baseline", "direction": "less", "mesi": 0.05, "seeds": 5},
        {"id": "H-RIVAL", "statement": "Stillness (gate on unformed motion) beats composure (uniform learned "
         "attention to all motion) under shift.", "metric": "expected commit utility", "split": "shifted",
         "comparison": "model - rival", "direction": "greater", "mesi": 0.03, "seeds": 5}],
    "thresholds": {"loss_drop_fraction": 0.25, "margin_over_trivial": 0.10, "gradcheck_rel_tol": 1e-4,
                   "grad_clip_norm": 5.0},
    "probe_predictions": [
        {"probe": "P1", "expected": "above baseline"}, {"probe": "P4", "expected": "above baseline"},
        {"probe": "P9", "expected": "above baseline for abrupt perturbations, below for high-frequency noise"},
        {"probe": "P7", "expected": "below baseline"}, {"probe": "P8", "expected": "equal to baseline"},
        {"probe": "P3", "expected": "equal to baseline"}, {"probe": "P5", "expected": "equal to baseline"}],
    "dialectic_links": [{"chapter": 299, "relation": "successor", "test": "H-RIVAL"}],
    "corpus_neighbors": [
        {"chapter": 263, "similarity": None, "difference": "Shao Yong transfers one cycle shape across scales to "
         "forecast; here nothing is forecast - the verdict is read from pre-form motion and the rest is ignored."},
        {"chapter": 299, "similarity": None, "difference": "Zhu Xi accumulates breadth under composure; here "
         "breadth is excluded and only the unformed onset is read (tested as H-RIVAL)."},
        {"chapter": 478, "similarity": None, "difference": "Toegye gates which origin issues a feeling; here the "
         "gate is on the amplitude at both ends of each motion, not on its source."},
        {"chapter": 262, "similarity": None, "difference": "al-Sarakhsi admits features by attested contrasts; "
         "here admission is by timing relative to form."},
        {"chapter": 502, "similarity": 0.035, "difference": "Philip II sample (master id 502; file labelled 0507, "
         "text labelled 0389): multiplicative attenuation of annotations; the only corpus code available in "
         "session, 9-shingle Jaccard measured 0.035 outside the utilities block."}],
    "barometer": {
        "cognitive_processing": ["delayed recall of the onset direction across formed distractors (P1)"],
        "embodied_cognition": ["form-priced commitment as a stand-in for early balance correction; no body"],
        "world_modeling": ["tracking under drift and regime shift of mature dynamics (shifted split)"],
        "consciousness": ["commit hazard as a self-monitoring signal; no claim about experience"],
        "language_understanding": [],
        "emotional_intelligence": [],
        "creativity": [],
        "autonomy": ["self-timed commitment under a cost it must weigh (halting)"]},
    "task_types": ["sequence_classification"],
    "applications": [
        {"use": "seizure-onset lateralisation and early warning (research decision support only)",
         "sector": "health research", "dataset": "CHB-MIT Scalp EEG Database (PhysioNet)"},
        {"use": "incipient bearing-fault detection before damage spreads", "sector": "industrial maintenance",
         "dataset": "CWRU Bearing Data Center; IMS/NASA bearing run-to-failure"},
        {"use": "fall-onset detection for wearables and humanoid balance recovery", "sector": "assistive "
         "robotics", "dataset": "SisFall (Sucerquia et al. 2017)"}],
    "safety_notes": "Medical uses are research decision support only, with no dosing or treatment advice. "
                    "Synthetic data only. No generated sentence is presented as Zhou Dunyi's own words.",
}

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

CHAPTER = 265
EPS2 = 1e-6            # smoothing of the squared amplitude (keeps log finite at rest)
CLIP = 5.0             # global-norm gradient clip
TRIVIAL_UTILITY = 0.5  # guess at the first step: accuracy 0.5, no form paid


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
NATIVE = dict(T=32, onset=(3, 12), drift=0.01, noise=0.07, r0=0.06, rate=(0.40, 0.40), omega=(0.5, 0.5),
              omega_sign=False, phase_noise=0.03, n_intr=(1, 1), p_intr=0.6, a_intr=(0.6, 1.2),
              hold=(2, 5), late_turn=False, kappa=0.4)
SHIFTED = dict(NATIVE, T=48, onset=(10, 30), drift=0.03, rate=(0.30, 0.50), omega=(0.3, 0.9),
               omega_sign=True, phase_noise=0.06, n_intr=(1, 3), p_intr=1.0, a_intr=(0.8, 2.0))
LATE_TURN = dict(NATIVE, T=36, onset=(2, 6), p_intr=0.3, late_turn=True, kappa=0.1)


def make_world(rng, d=6):
    """A world is a fixed orthonormal embedding of the latent plane in d observed channels."""
    mixing, _ = np.linalg.qr(rng.standard_normal((d, 2)))
    return {"M": mixing, "d": d}


def sample(rng, n, world, cond):
    """Stirrings that grow from rest to form, observed through drift, noise and formed intrusions."""
    T, d = cond["T"], world["d"]
    y = rng.integers(0, 2, n)
    t0 = rng.integers(cond["onset"][0], cond["onset"][1] + 1, n)
    theta = np.where(y == 1, 0.0, np.pi) + rng.uniform(-0.7, 0.7, n)
    if cond["late_turn"]:
        theta = rng.uniform(-np.pi, np.pi, n)
    rate = rng.uniform(*cond["rate"], n)
    omega = rng.uniform(*cond["omega"], n)
    if cond["omega_sign"]:
        omega *= np.where(rng.random(n) < 0.5, -1.0, 1.0)
    r, started = np.zeros(n), np.zeros(n, bool)
    latent, form = np.zeros((n, T, 2)), np.zeros((n, T))
    turn = np.zeros((n, 2))
    for t in range(T):
        new = (t >= t0) & ~started
        grow = started.copy()
        r[new], started = cond["r0"], started | new
        r[grow] += rate[grow] * r[grow] * (1.0 - r[grow])
        theta[grow] += omega[grow] * r[grow] ** 2 + cond["phase_noise"] * r[grow] * rng.standard_normal(grow.sum())
        if cond["late_turn"]:
            hit = started & (t >= t0 + 12)
            turn[hit, 0] = np.where(y[hit] == 1, 0.7, -0.7)
        latent[:, t] = np.stack([r * np.cos(theta), r * np.sin(theta)], 1) + turn
        form[:, t] = r
    X = latent @ world["M"].T + rng.standard_normal((n, 1, d))
    X += np.cumsum(cond["drift"] * rng.standard_normal((n, T, d)), axis=1)
    X += cond["noise"] * rng.standard_normal((n, T, d))
    for i in np.flatnonzero(rng.random(n) < cond["p_intr"]):
        for _ in range(rng.integers(cond["n_intr"][0], cond["n_intr"][1] + 1)):
            start, hold = rng.integers(1, T - 3), rng.integers(cond["hold"][0], cond["hold"][1] + 1)
            u = rng.standard_normal(d)
            X[i, start:start + hold] += rng.uniform(*cond["a_intr"]) * u / np.linalg.norm(u)
    return {"X": X, "y": y, "F": form, "kappa": cond["kappa"], "onset": t0}


def subset(data, idx):
    return {"X": data["X"][idx], "y": data["y"][idx], "F": data["F"][idx], "kappa": data["kappa"]}


def expected_commit_loss(logits, stop, y, form, kappa):
    """ELECTS-style expected loss: commit at t with prob P_t; loss = CE + kappa * form at commitment."""
    B, T, _ = logits.shape
    probs = softmax(logits)
    ce = -np.log(np.take_along_axis(probs, np.repeat(y[:, None, None], T, 1), 2)[..., 0] + 1e-300)
    ell = ce + kappa * form
    halt = sigmoid(stop)
    halt[:, -1] = 1.0
    surv = np.ones((B, T))
    surv[:, 1:] = np.cumprod(1.0 - halt[:, :-1], axis=1)
    p_commit = halt * surv
    value = np.zeros((B, T + 1))           # value[t] = expected loss given survival to t
    value[:, T - 1] = ell[:, T - 1]
    for t in range(T - 2, -1, -1):
        value[:, t] = halt[:, t] * ell[:, t] + (1.0 - halt[:, t]) * value[:, t + 1]
    onehot = np.zeros_like(probs)
    onehot[np.arange(B), :, y] = 1.0
    d_logits = p_commit[..., None] * (probs - onehot) / B
    d_stop = surv * (ell - value[:, 1:]) * halt * (1.0 - halt) / B
    d_stop[:, -1] = 0.0
    correct = (logits.argmax(-1) == y[:, None]).astype(float)
    metrics = {"utility": float((p_commit * (correct - kappa * form)).sum(1).mean()),
               "accuracy": float((p_commit * correct).sum(1).mean()),
               "form_at_commit": float((p_commit * form).sum(1).mean())}
    return float(value[:, 0].mean()), d_logits, d_stop, metrics, p_commit


# ============================================================ model: the cheng-shen-ji circuit
def init_ji(rng, d, C, k=4, m=7, rival=False):
    p = {"ac": np.full(d, math.log(0.02 / 0.98)), "Wp": rng.standard_normal((d, k)) * 1.5 / math.sqrt(d)}
    if rival:
        p["ak"] = np.zeros(k)                        # composure: one learned attention level per dimension
    else:
        p["th"], p["ag"] = np.array([math.log(0.35)]), np.array([1.5])
    p.update({"ar": np.full(k, math.log(9.0)), "Wh": rng.standard_normal((k, m)) * 6.0 / math.sqrt(k),
              "Wo": rng.standard_normal((m, C)) * 0.5, "ws": np.zeros(m), "bs": np.array([-2.0])})
    return p


def ji_forward(model, X):
    """cheng: still core set at the first frame; ji: gated differences near it; shen: odd, bias-free readout."""
    p, ko, rival = model["params"], model["ko"], model["kind"] == "jing"
    B, T, d = X.shape
    k = p["Wp"].shape[1]
    eta = np.zeros(d) if ko.get("cheng_core") else sigmoid(p["ac"])
    rho = {"identity": np.ones(k), "zero": np.zeros(k)}.get(ko.get("ji_trace"), sigmoid(p["ar"]))
    gam = 0.0 if rival else (p["ag"][0] if model.get("variant") == "gate_sign_free" else softplus(p["ag"][0]))
    core = np.zeros((B, d)) if model.get("variant") == "core_at_zero" else X[:, 0].copy()
    zp, J = np.zeros((B, k)), np.zeros((B, k))
    logits, stop, cache = np.zeros((B, T, p["Wo"].shape[1])), np.zeros((B, T)), []
    for t in range(T):
        dev = X[:, t] - core
        core = core + eta * dev
        z = dev @ p["Wp"]
        v = z - zp
        q = np.sum(z * z, 1) + np.sum(zp * zp, 1) + EPS2
        lm = 0.5 * np.log(q)
        if rival:
            g = np.broadcast_to(sigmoid(p["ak"]), (B, k))
        elif "ji_lens" in ko:
            g = np.full((B, 1), {"mean": model.get("gbar", 0.5), "identity": 1.0, "zero": 0.0}[ko["ji_lens"]])
        else:
            g = sigmoid(gam * (p["th"][0] - lm))[:, None]
        Jp, J = J, rho * J + g * v
        u = np.tanh(J @ p["Wh"])
        logits[:, t] = u @ p["Wo"] + model.get("bias", 0.0)
        ws = np.zeros_like(p["ws"]) if ko.get("commit") else p["ws"]
        stop[:, t] = (u * u) @ ws + p["bs"][0]
        cache.append((dev, z, zp, v, q, lm, g, Jp, J, u))
        zp = z
    return logits, stop, {"cache": cache, "eta": eta, "rho": rho, "gam": gam}


def ji_backward(model, X, st, d_logits, d_stop):
    p, ko, rival = model["params"], model["ko"], model["kind"] == "jing"
    grads = {name: np.zeros_like(w) for name, w in p.items()}
    eta, rho, gam = st["eta"], st["rho"], st["gam"]
    B, T, d = X.shape
    k = p["Wp"].shape[1]
    dJc, dzc, dcore = np.zeros((B, k)), np.zeros((B, k)), np.zeros((B, d))
    geta, grho, ggam, gth, gk = np.zeros(d), np.zeros(k), 0.0, 0.0, np.zeros(k)
    learn_gate = not rival and "ji_lens" not in ko
    for t in range(T - 1, -1, -1):
        dev, z, zp, v, q, lm, g, Jp, J, u = st["cache"][t]
        dl, ds = d_logits[:, t], d_stop[:, t]
        ws = np.zeros_like(p["ws"]) if ko.get("commit") else p["ws"]
        grads["Wo"] += u.T @ dl
        if not ko.get("commit"):
            grads["ws"] += (u * u).T @ ds
        grads["bs"] += ds.sum()
        du = dl @ p["Wo"].T + ds[:, None] * 2.0 * u * ws
        dh = du * (1.0 - u * u)
        grads["Wh"] += J.T @ dh
        dJ = dh @ p["Wh"].T + dJc
        grho += np.sum(dJ * Jp, 0)
        dJc = dJ * rho
        dv = g * dJ
        dz, dzp = dzc + dv, -dv
        if rival:
            gk += np.sum(dJ * v, 0)
        elif learn_gate:
            da = np.sum(dJ * v, 1) * g[:, 0] * (1.0 - g[:, 0])
            ggam += np.sum(da * (p["th"][0] - lm))
            gth += np.sum(da * gam)
            dq = -da * gam * 0.5 / q
            dz = dz + dq[:, None] * 2.0 * z
            dzp = dzp + dq[:, None] * 2.0 * zp
        grads["Wp"] += dev.T @ dz
        ddev = dz @ p["Wp"].T + dcore * eta
        geta += np.sum(dcore * dev, 0)
        dcore = dcore - ddev
        dzc = dzp
    if not ko.get("cheng_core"):
        grads["ac"] = geta * eta * (1.0 - eta)
    if ko.get("ji_trace") is None:
        grads["ar"] = grho * rho * (1.0 - rho)
    if rival:
        s = sigmoid(p["ak"])
        grads["ak"] = gk * s * (1.0 - s)
    elif learn_gate:
        slope = 1.0 if model.get("variant") == "gate_sign_free" else sigmoid(p["ag"][0])
        grads["ag"], grads["th"] = np.array([ggam * slope]), np.array([gth])
    return grads


# ============================================================ baseline: ELECTS-style Elman RNN
def init_elects(rng, d, C, H=5):
    return {"ac": np.full(d, math.log(0.02 / 0.98)), "Wx": rng.standard_normal((d, H)) * 3.0 / math.sqrt(d),
            "Whh": rng.standard_normal((H, H)) * 0.5 / math.sqrt(H), "bh": np.zeros(H),
            "Wo": rng.standard_normal((H, C)) * 0.5, "bo": np.zeros(C), "ws": np.zeros(H), "bs": np.array([-2.0])}


def elects_forward(model, X):
    p = model["params"]
    B, T, d = X.shape
    H = p["Whh"].shape[0]
    eta, core, h = sigmoid(p["ac"]), X[:, 0].copy(), np.zeros((B, H))
    logits, stop, cache = np.zeros((B, T, p["Wo"].shape[1])), np.zeros((B, T)), []
    for t in range(T):
        dev = X[:, t] - core
        core = core + eta * dev
        hp, h = h, np.tanh(dev @ p["Wx"] + h @ p["Whh"] + p["bh"])
        logits[:, t] = h @ p["Wo"] + p["bo"]
        stop[:, t] = h @ p["ws"] + p["bs"][0]
        cache.append((dev, hp, h))
    return logits, stop, {"cache": cache, "eta": eta}


def elects_backward(model, X, st, d_logits, d_stop):
    p = model["params"]
    grads = {name: np.zeros_like(w) for name, w in p.items()}
    eta = st["eta"]
    B, T, d = X.shape
    dhc, dcore, geta = np.zeros((B, p["Whh"].shape[0])), np.zeros((B, d)), np.zeros(d)
    for t in range(T - 1, -1, -1):
        dev, hp, h = st["cache"][t]
        dl, ds = d_logits[:, t], d_stop[:, t]
        grads["Wo"] += h.T @ dl
        grads["bo"] += dl.sum(0)
        grads["ws"] += h.T @ ds
        grads["bs"] += ds.sum()
        da = (dl @ p["Wo"].T + ds[:, None] * p["ws"] + dhc) * (1.0 - h * h)
        grads["Wx"] += dev.T @ da
        grads["Whh"] += hp.T @ da
        grads["bh"] += da.sum(0)
        dhc = da @ p["Whh"].T
        ddev = da @ p["Wx"].T + dcore * eta
        geta += np.sum(dcore * dev, 0)
        dcore = dcore - ddev
    grads["ac"] = geta * eta * (1.0 - eta)
    return grads


# ============================================================ common interface (section 8)
TASK_TYPES = ["sequence_classification"]
KINDS = {"ji": (ji_forward, ji_backward), "jing": (ji_forward, ji_backward),
         "elects": (elects_forward, elects_backward)}


def build_model(in_dim, out_dim, task_type, rng, kind="ji", **cfg):
    if task_type not in TASK_TYPES:
        raise ValueError(f"unsupported task type {task_type}")
    if kind == "elects":
        params = init_elects(rng, in_dim, out_dim, **cfg)
    else:
        params = init_ji(rng, in_dim, out_dim, rival=(kind == "jing"), **cfg)
    return {"kind": kind, "params": params, "ko": {}, "mutant": None}


def n_params(model):
    return int(sum(w.size for w in model["params"].values()))


def forward(model, X):
    return KINDS[model["kind"]][0](model, X)


def loss_and_grads(model, batch):
    fwd, bwd = KINDS[model["kind"]]
    F = batch.get("F", np.zeros(batch["X"].shape[:2]))
    logits, stop, st = fwd(model, batch["X"])
    loss, dl, ds, metrics, _ = expected_commit_loss(logits, stop, batch["y"], F, batch.get("kappa", 0.0))
    grads = bwd(model, batch["X"], st, dl, ds)
    for name in MUTANTS.get(model["mutant"], {}).get("zero_grads", []):
        if name in grads:
            grads[name] = np.zeros_like(grads[name])
    return loss, grads, metrics


def evaluate(model, data):
    logits, stop, _ = forward(model, data["X"])
    _, _, _, metrics, p_commit = expected_commit_loss(logits, stop, data["y"], data["F"], data["kappa"])
    if "onset" in data:                      # descriptive: commitment mass spent before the stirring began
        before = np.arange(p_commit.shape[1])[None, :] < data["onset"][:, None]
        metrics["committed_before_onset"] = float((p_commit * before).sum(1).mean())
    return metrics


def predict(model, X):
    """Class probabilities mixed over the model's own commitment distribution."""
    logits, stop, _ = forward(model, X)
    halt = sigmoid(stop)
    halt[:, -1] = 1.0
    surv = np.concatenate([np.ones((X.shape[0], 1)), np.cumprod(1 - halt[:, :-1], 1)], 1)
    return np.einsum("bt,btc->bc", halt * surv, softmax(logits))


def hidden_states(model, X):
    _, _, st = forward(model, X)
    return np.stack([c[8] if model["kind"] != "elects" else c[2] for c in st["cache"]], 1)


MODULE_REGISTRY = {
    "ji": {"cheng_core": (["ac"], "leaky per-channel reference initialised at the first frame", False),
           "ji_lens": (["Wp", "th", "ag"], "projection plus two-endpoint amplitude-inverse gate on temporal "
                       "differences", True),
           "ji_trace": (["ar"], "leaky accumulator of gated differences (per-dimension retention)", True),
           "shen_response": (["Wh", "Wo"], "bias-free tanh readout, odd in the core-referenced input", False),
           "commit": (["ws", "bs"], "even halting hazard on the squared response", False)},
    "jing": {"cheng_core": (["ac"], "leaky per-channel reference initialised at the first frame", False),
             "jing_attention": (["Wp", "ak"], "projection plus learned uniform per-dimension attention on "
                                "temporal differences", False),
             "ji_trace": (["ar"], "leaky accumulator of attended differences", False),
             "shen_response": (["Wh", "Wo"], "bias-free tanh readout", False),
             "commit": (["ws", "bs"], "even halting hazard on the squared response", False)},
    "elects": {"cheng_core": (["ac"], "leaky per-channel reference initialised at the first frame", False),
               "recurrence": (["Wx", "Whh", "bh"], "Elman tanh recurrence", False),
               "readout": (["Wo", "bo"], "affine class readout", False),
               "commit": (["ws", "bs"], "affine halting hazard (ELECTS stopping head)", False)},
}


def modules(model):
    return {name: {"params": ps, "role": role, "signature": sig}
            for name, (ps, role, sig) in MODULE_REGISTRY.get(model["kind"], {}).items()}


def knockout(model, name, mode):
    """Copy with one module replaced: cheng_core identity (frozen core), ji_lens mean/identity/zero gate,
    ji_trace identity (retention 1) or zero, commit zero (constant hazard)."""
    out = dict(model, params={k: v.copy() for k, v in model["params"].items()}, ko=dict(model["ko"]))
    out["ko"][name] = mode
    return out


MUTANTS = {
    "sign_flip": {"lr_sign": -1.0, "note": "Adam update applied with reversed sign"},
    "zero_lr": {"lr_scale": 0.0, "note": "learning rate set to zero"},
    "zero_grad_Wp": {"zero_grads": ["Wp"], "note": "gradient of the ji_lens projection zeroed"},
    "drop_gate_grad": {"zero_grads": ["th", "ag"], "note": "gate threshold and sharpness gradients zeroed"},
}


# ============================================================ training
def fit(model, data, budget, rng, lr=0.03, batch=128, val=None, eval_every=50):
    """Adam on the expected commitment loss; selection of the best checkpoint uses the validation split only."""
    mut = MUTANTS.get(model["mutant"], {})
    opt = Adam(model["params"], lr=lr * mut.get("lr_scale", 1.0))
    history, best = [], (-np.inf, None)
    n = len(data["y"])
    for step in range(1, budget + 1):
        idx = rng.choice(n, min(batch, n), replace=False)
        loss, grads, _ = loss_and_grads(model, subset(data, idx))
        grads, _ = clip_global_norm(grads, CLIP)
        opt.step(model["params"], grads, sign=mut.get("lr_sign", 1.0))
        history.append(loss)
        if val is not None and (step % eval_every == 0 or step == budget):
            u = evaluate(model, val)["utility"]
            if u > best[0]:
                best = (u, {k: v.copy() for k, v in model["params"].items()})
    if best[1] is not None:
        model["params"] = best[1]
    if model["kind"] == "ji":
        _, _, st = forward(model, data["X"][:512])
        model["gbar"] = float(np.mean([c[6].mean() for c in st["cache"]]))
    return history


def loss_drop(history):
    first, last = float(np.mean(history[:5])), float(np.mean(history[-20:]))
    return (first - last) / max(first, 1e-12)


def finite_model(model, X):
    logits, stop, _ = forward(model, X)
    arrays = list(model["params"].values()) + [logits, stop]
    return all(np.all(np.isfinite(a)) for a in arrays)


# ============================================================ tests: correctness
def gradcheck(model, data, rng):
    batch = subset(data, np.arange(8))
    _, grads, _ = loss_and_grads(model, batch)
    return finite_difference_check(lambda: loss_and_grads(model, batch)[0], model["params"], grads, rng)


def c1_gradients(worlds, rng, mutant=None, steps=60):
    """Every tensor of every trained kind, at initialisation and after >= 50 Adam steps."""
    rows, worst = [], 0.0
    for kind in ("ji", "jing", "elects"):
        model = build_model(6, 2, "sequence_classification", np.random.default_rng(rng.integers(1 << 31)), kind)
        model["mutant"] = mutant
        for when in ("init", "after_training_steps"):
            if when != "init":
                fit(model, worlds["train"], steps, np.random.default_rng(1))
            err, n_t = gradcheck(model, worlds["train"], np.random.default_rng(2))
            worst = max(worst, err)
            rows.append((kind, when, n_t, len(model["params"]), err))
    ok = worst <= MIND_CARD["thresholds"]["gradcheck_rel_tol"] and all(r[2] == r[3] for r in rows)
    return ok, worst, rows


def c3_learning(model, history, held):
    drop = loss_drop(history)
    util = evaluate(model, held)["utility"]
    th = MIND_CARD["thresholds"]
    return drop >= th["loss_drop_fraction"] and util >= TRIVIAL_UTILITY + th["margin_over_trivial"], drop, util


def c5_mutants(worlds, rng):
    """Each learning-breaking mutant must make C1 or a short C3 fail; the unmutated short run must pass."""
    def short_c3(mutant):
        m = build_model(6, 2, "sequence_classification", np.random.default_rng(11), "ji")
        m["mutant"] = mutant
        hist = fit(m, worlds["train"], 150, np.random.default_rng(12))
        return c3_learning(m, hist, worlds["held"])[0]
    control = short_c3(None)
    detected = {}
    for name in MUTANTS:
        c1_ok = c1_gradients(worlds, np.random.default_rng(13), mutant=name, steps=50)[0] \
            if "zero_grads" in MUTANTS[name] else True
        detected[name] = (not c1_ok) or (not short_c3(name))
    return control, detected


def c6_properties(rng, n_trials=40):
    """C6.1 offset invariance, C6.2 impartiality (odd logits, even hazard), C6.3 gate monotone in form.
    Each is searched over random parameters and inputs, with a negative control that must violate it."""
    def random_model(variant=None, bias=None):
        m = build_model(6, 2, "sequence_classification", np.random.default_rng(rng.integers(1 << 31)), "ji")
        p = m["params"]
        p["th"][0], p["ag"][0] = rng.normal(-1.0, 1.0), rng.normal(0.0, 2.5)
        p["ws"][:] = rng.normal(0, 1, p["ws"].shape)
        p["ac"][:] = rng.normal(-3, 1.5, p["ac"].shape)
        m["variant"] = variant
        if bias is not None:
            m["bias"] = bias
        return m

    def offset_gap(m):
        X = rng.normal(0, rng.uniform(0.05, 1.0), (4, 12, 6))
        off = rng.normal(0, 1, 6) * rng.uniform(0.1, 10.0)
        a, b = forward(m, X), forward(m, X + off)
        return max(np.abs(a[0] - b[0]).max(), np.abs(a[1] - b[1]).max())

    def mirror_gap(m):
        X = rng.normal(0, rng.uniform(0.05, 1.0), (4, 12, 6))
        Xm = 2 * X[:, :1] - X
        a, b = forward(m, X), forward(m, Xm)
        return max(np.abs(a[0] + b[0]).max(), np.abs(a[1] - b[1]).max())

    def monotone_gap(m):
        base = rng.normal(0, 1, (1, 10, 6)).cumsum(1) * 0.05
        base = base - base[:, :1]
        s1, s2 = sorted(rng.uniform(0.2, 30.0, 2))
        g = [np.array([c[6][0, 0] for c in forward(m, base * s)[2]["cache"][1:]]) for s in (s1, s2)]
        return float(np.max(g[1] - g[0]))        # a larger amplitude must never open the gate wider

    tests = {"C6.1": (offset_gap, {}, {"variant": "core_at_zero"}),
             "C6.2": (mirror_gap, {}, {"bias": np.array([0.3, -0.1])}),
             "C6.3": (monotone_gap, {}, {"variant": "gate_sign_free"})}
    out = {}
    for tid, (fn, pos, neg) in tests.items():
        worst_pos = max(fn(random_model(**pos)) for _ in range(n_trials))
        worst_neg = max(fn(random_model(**neg)) for _ in range(n_trials))
        out[tid] = (worst_pos <= 1e-9 and worst_neg > 1e-3, worst_pos, worst_neg)
    rest = forward(random_model(), np.repeat(rng.normal(0, 1, (2, 1, 6)), 8, axis=1))[0]
    out["D-1"] = (bool(np.all(rest == 0.0)), float(np.abs(rest).max()), None)
    return out


def split_hashes(data):
    return {hashlib.sha1(x.tobytes()).hexdigest() for x in data["X"]}


# ============================================================ one seed of the protocol
def run_seed(seed, cfg, keep_models=False):
    rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(10)]
    world = make_world(rngs[0])
    d = {"train": sample(rngs[1], cfg["n_train"], world, NATIVE), "val": sample(rngs[1], cfg["n_val"], world, NATIVE),
         "held": sample(rngs[1], cfg["n_test"], world, NATIVE), "shift": sample(rngs[2], cfg["n_test"], world, SHIFTED),
         "late_train": sample(rngs[3], cfg["n_train"], world, LATE_TURN),
         "late_val": sample(rngs[3], cfg["n_val"], world, LATE_TURN),
         "late_held": sample(rngs[3], cfg["n_test"], world, LATE_TURN)}
    res, models, hists = {"seed": seed}, {}, {}
    for i, kind in enumerate(("ji", "elects", "jing")):
        m = build_model(6, 2, "sequence_classification", rngs[4 + i], kind)
        hists[kind] = fit(m, d["train"], cfg["steps"], rngs[7], val=d["val"])
        models[kind] = m
        for split in ("held", "shift"):
            res[f"{kind}_{split}"] = evaluate(m, d[split])["utility"]
    for name, mode in (("cheng_core", "identity"), ("ji_lens", "mean"), ("ji_trace", "identity"), ("commit", "zero")):
        res[f"ko_{name}"] = evaluate(knockout(models["ji"], name, mode), d["shift"])["utility"] - res["ji_shift"]
    for i, kind in enumerate(("ji", "elects")):
        m = build_model(6, 2, "sequence_classification", rngs[8 + i], kind)
        fit(m, d["late_train"], cfg["steps"], rngs[7], val=d["late_val"])
        res[f"{kind}_late"] = evaluate(m, d["late_held"])["utility"]
        models[f"{kind}_late"] = m
    res["finite"] = all(finite_model(m, d["shift"]["X"][:64]) for m in models.values())
    res["pre_onset"] = {k: evaluate(models[k], d["shift"])["committed_before_onset"] for k in ("ji", "elects", "jing")}
    res["gate"] = {k: (math.exp(models[k]["params"]["th"][0]), float(softplus(models[k]["params"]["ag"][0])))
                   for k in ("ji", "ji_late")}
    res["form_at_commit"] = {k: evaluate(models[k], d["held"])["form_at_commit"] for k in ("ji", "elects", "jing")}
    if keep_models:
        return res, models, hists, d
    return res


def hypothesis_table(rows, rng):
    diffs = {"H-SIG": [r["ji_shift"] - r["elects_shift"] for r in rows],
             "H-NEC": [r["ko_ji_lens"] - r["ko_cheng_core"] for r in rows],
             "H-BLIND": [r["ji_late"] - r["elects_late"] for r in rows],
             "H-RIVAL": [r["ji_shift"] - r["jing_shift"] for r in rows]}
    table = []
    for h in MIND_CARD["hypotheses"]:
        if len(rows) < 5:
            table.append({"id": h["id"], "metric": h["metric"], "mean_diff": float(np.mean(diffs[h["id"]])),
                          "ci95": [None, None], "mesi": h["mesi"], "n_seeds": len(rows), "verdict": "not evaluated"})
            continue
        mean, ci = paired_bootstrap(diffs[h["id"]], rng)
        table.append({"id": h["id"], "metric": h["metric"], "mean_diff": mean, "ci95": ci, "mesi": h["mesi"],
                      "n_seeds": len(rows), "verdict": verdict(mean, ci, h["direction"], h["mesi"])})
    return table


# ============================================================ report
def build_report(args, cfg, seeds, correctness, grad, mutants, hyps, knock, rows, t0, exit_code):
    fname = os.path.basename(__file__)
    lines = [f"=== VERIFIED REPORT · chapter {CHAPTER:04d} ===",
             f"file {fname} · card revision {MIND_CARD['card_revision']} · mode {'quick' if args.quick else 'full'}",
             f"python {sys.version.split()[0]} · numpy {np.__version__} · seeds {seeds}",
             f"runtime {time.time() - t0:.1f} s · n_params ji {grad['n_params']['ji']}, "
             f"elects {grad['n_params']['elects']}, jing {grad['n_params']['jing']}",
             f"gradcheck: {grad['tensors']}/{grad['total']} tensor checks (every tensor, 3 kinds, init + after "
             f"training), max rel error {grad['worst']:.2e}",
             "correctness:"]
    lines += [f"  {c['id']:<5} {c['name']:<34} {'PASS' if c['passed'] else 'FAIL'}  {c['detail']}" for c in correctness]
    lines.append(f"mutants: {mutants['detected']}/{mutants['total']} detected (score {mutants['score']:.2f})")
    lines.append("hypotheses (paired per-seed differences, 95% bootstrap, 2000 resamples):")
    for h in hyps:
        ci = "n/a" if h["ci95"][0] is None else f"[{h['ci95'][0]:+.3f}, {h['ci95'][1]:+.3f}]"
        lines.append(f"  {h['id']:<8} mean {h['mean_diff']:+.3f}  CI {ci}  mesi {h['mesi']}  -> {h['verdict']}")
    lines.append("per-seed utilities (held / shifted / late-turn):")
    for r in rows:
        lines.append(f"  seed {r['seed']}: ji {r['ji_held']:.3f}/{r['ji_shift']:.3f}/{r['ji_late']:.3f}  "
                     f"elects {r['elects_held']:.3f}/{r['elects_shift']:.3f}/{r['elects_late']:.3f}  "
                     f"jing {r['jing_held']:.3f}/{r['jing_shift']:.3f}")
    lines.append("descriptive (no verdict): shifted commitment mass before onset ji/elects/jing; learned gate "
                 "radius exp(th) and sharpness, native and late-turn models:")
    for r in rows:
        p, g = r["pre_onset"], r["gate"]
        lines.append(f"  seed {r['seed']}: before onset {p['ji']:.2f}/{p['elects']:.2f}/{p['jing']:.2f}  gate native "
                     f"{g['ji'][0]:.2f} x{g['ji'][1]:.2f}  gate late-turn {g['ji_late'][0]:.2f} x{g['ji_late'][1]:.2f}")
    lines.append("knockouts on the trained ji model (shifted utility change):")
    for k in knock:
        ci = "n/a" if k["ci95"][0] is None else f"[{k['ci95'][0]:+.3f}, {k['ci95'][1]:+.3f}]"
        lines.append(f"  {k['module']:<14} signature={str(k['signature']):<5} change {k['metric_change']:+.3f}  CI {ci}")
    lines += [f"task types: {', '.join(TASK_TYPES)}", f"exit code {exit_code}", "=== END REPORT ==="]
    payload = {"schema_version": "1.0", "chapter": CHAPTER, "file": fname, "card_revision": MIND_CARD["card_revision"],
               "environment": {"python": sys.version.split()[0], "numpy": np.__version__}, "seeds": seeds,
               "runtime_s": round(time.time() - t0, 2), "n_params": grad["n_params"]["ji"],
               "gradcheck": {"tensors_checked": grad["tensors"], "tensors_total": grad["total"],
                             "max_rel_error": grad["worst"], "checked_at": ["init", "after_training_steps"],
                             "passed": grad["passed"]},
               "correctness": correctness, "mutants": mutants, "hypotheses": hyps, "knockouts": knock,
               "task_types": TASK_TYPES, "exit_code": exit_code}
    return lines, payload


# ============================================================ optional real-data bridge (section 10.3)
def data_bridge(path, rng):
    """CSV with a 'label' column and columns named c<channel>_t<step>; trains ji and elects (time-priced)."""
    if not path or not os.path.exists(path):
        print(f"--data: no file at {path!r}; real-data bridge skipped.")
        return
    raw = np.genfromtxt(path, delimiter=",", names=True)
    cols = [c for c in raw.dtype.names if c != "label"]
    n_ch = 1 + max(int(c.split("_")[0][1:]) for c in cols)
    T = 1 + max(int(c.split("_t")[1]) for c in cols)
    X = np.stack([np.stack([raw[f"c{ch}_t{t}"] for ch in range(n_ch)], -1) for t in range(T)], 1)
    y = raw["label"].astype(int)
    classes = np.unique(y)
    y = np.searchsorted(classes, y)
    F = np.repeat((np.arange(T) / max(T - 1, 1))[None], len(y), 0)
    perm = rng.permutation(len(y))
    cut = int(0.7 * len(y))
    full = {"X": X, "y": y, "F": F, "kappa": 0.2}
    tr, te = subset(full, perm[:cut]), subset(full, perm[cut:])
    for kind in ("ji", "elects"):
        m = build_model(n_ch, len(classes), "sequence_classification", np.random.default_rng(0), kind)
        fit(m, tr, 300, np.random.default_rng(1))
        met = evaluate(m, te)
        print(f"--data {kind}: accuracy {met['accuracy']:.3f}, mean time fraction at commit {met['form_at_commit']:.3f}")


# ============================================================ entry point
def main(argv=None):
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__), description="Chapter 0265 JI LENS protocol")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=265)
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
    cfg = dict(n_train=1200, n_val=300, n_test=500, steps=150, seeds=1, budget=20.0) if args.quick else \
        dict(n_train=3000, n_val=500, n_test=1000, steps=400, seeds=5, budget=180.0)
    n_seeds = args.seeds or cfg["seeds"]
    seeds = [args.seed + i for i in range(n_seeds)]
    print(f"seeds {seeds} (base {args.seed}); mutant {args.mutant}; mode {'quick' if args.quick else 'full'}")
    rng = np.random.default_rng(args.seed)

    first, models, hists, worlds = run_seed(seeds[0], cfg, keep_models=True)
    c1_ok, worst, grad_rows = c1_gradients(worlds, np.random.default_rng(args.seed + 1), mutant=args.mutant)
    rep, rep_hist = models["ji"], hists["ji"]
    if args.mutant:                         # retrain under the mutant so that C3 can detect it
        rep = build_model(6, 2, "sequence_classification", np.random.default_rng(3), "ji")
        rep["mutant"] = args.mutant
        rep_hist = fit(rep, worlds["train"], cfg["steps"], np.random.default_rng(4), val=worlds["val"])
    c3_ok, drop, util = c3_learning(rep, rep_hist, worlds["held"])
    a, b = [build_model(6, 2, "sequence_classification", np.random.default_rng(5), "ji") for _ in range(2)]
    ha, hb = fit(a, worlds["train"], 30, np.random.default_rng(6)), fit(b, worlds["train"], 30, np.random.default_rng(6))
    same = ha == hb and np.array_equal(forward(a, worlds["held"]["X"][:32])[0], forward(b, worlds["held"]["X"][:32])[0])
    shuffled = dict(worlds["train"], y=np.random.default_rng(7).permutation(worlds["train"]["y"]))
    sh = build_model(6, 2, "sequence_classification", np.random.default_rng(8), "ji")
    fit(sh, shuffled, cfg["steps"], np.random.default_rng(9))
    band = max(0.05, 3.0 * math.sqrt(0.25 / cfg["n_test"]))
    sh_util = evaluate(sh, worlds["held"])["utility"]
    control, detected = c5_mutants(worlds, np.random.default_rng(10)) if not args.mutant else (True, {})
    props = c6_properties(np.random.default_rng(11))
    hs = [split_hashes(worlds[k]) for k in ("train", "val", "held", "shift")]
    disjoint = all(not (hs[i] & hs[j]) for i in range(4) for j in range(i + 1, 4))

    rows = [first] + ([run_seed(s, cfg) for s in seeds[1:]] if not args.mutant else [])
    finite = all(r["finite"] for r in rows) and all(np.isfinite(h).all() for h in hists.values())
    correctness = [
        {"id": "C1", "name": "gradient_check", "passed": c1_ok, "detail": f"max rel err {worst:.2e}"},
        {"id": "C2", "name": "determinism_and_finiteness", "passed": bool(same and finite),
         "detail": f"identical reruns {same}, finite {finite}"},
        {"id": "C3", "name": "learning", "passed": c3_ok,
         "detail": f"loss drop {drop:.2f} (>= 0.25), held utility {util:.3f} (>= 0.600)"},
        {"id": "C4", "name": "shuffled_label_control", "passed": sh_util <= TRIVIAL_UTILITY + band,
         "detail": f"held utility {sh_util:.3f} (<= {TRIVIAL_UTILITY + band:.3f})"}]
    if not args.mutant:
        correctness.append({"id": "C5", "name": "mutant_detection", "passed": control and all(detected.values()),
                            "detail": f"control passes {control}; " + ", ".join(f"{k}:{v}" for k, v in detected.items())})
    names = {"C6.1": "offset_invariance_still_core", "C6.2": "impartiality_odd_logits",
             "C6.3": "gate_monotone_in_form", "D-1": "rest_quiescence (definition check)"}
    for tid, (ok, pos, neg) in props.items():
        detail = f"worst {pos:.1e}" + ("" if neg is None else f", negative control {neg:.1e}")
        correctness.append({"id": tid, "name": names[tid], "passed": bool(ok), "detail": detail})
    correctness.append({"id": "C7", "name": "split_integrity", "passed": disjoint, "detail": "sha1 of every sequence"})
    hyps = hypothesis_table(rows, rng) if not args.mutant else []
    knock = []
    for name in ("cheng_core", "ji_lens", "ji_trace", "commit"):
        vals = [r[f"ko_{name}"] for r in rows]
        mean, ci = paired_bootstrap(vals, rng) if len(vals) >= 5 else (float(np.mean(vals)), [None, None])
        knock.append({"module": name, "signature": MODULE_REGISTRY["ji"][name][2], "metric_change": mean, "ci95": ci})
    runtime_ok = time.time() - t0 <= cfg["budget"]
    correctness.append({"id": "C8", "name": "runtime_budget", "passed": runtime_ok,
                        "detail": f"{time.time() - t0:.1f} s (<= {cfg['budget']:.0f} s)"})
    exit_code = 4 if not finite else (1 if not all(c["passed"] for c in correctness if c["id"] != "C8")
                                      else (3 if not runtime_ok else 0))
    grad = {"worst": worst, "tensors": sum(r[2] for r in grad_rows), "total": sum(r[3] for r in grad_rows),
            "passed": c1_ok, "n_params": {k: n_params(models[k]) for k in ("ji", "elects", "jing")}}
    mut = {"detected": sum(detected.values()), "total": len(MUTANTS) if not args.mutant else 0,
           "score": (sum(detected.values()) / len(MUTANTS)) if detected else 0.0}
    lines, payload = build_report(args, cfg, seeds, correctness, grad, mut, hyps, knock, rows, t0, exit_code)
    write_report(lines, args.json, payload)
    if args.data:
        data_bridge(args.data, np.random.default_rng(args.seed))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

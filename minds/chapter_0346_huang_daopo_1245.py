#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0346 · Huang Daopo (1245-1330)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 18, Minds 341-360 Available on Amazon https://www.amazon.com/dp/B0HQ47T2NC
# Resume and Interactive Demos at https://artificiology.com/
# END ATTRIBUTION 
"""ZUO ZAO (做造) — the making, not the made: tools re-made at every stage from
the material as it arrives there.

Thesis
    Hand over the making of every tool in a processing chain rather than the
    tools: one maker, shared by all stages, rebuilds each stage's tool on site
    from a label-free measurement of the material as it reaches that stage, so
    that each finished tool becomes the instrument that makes the next stage
    measurable, and the chain keeps working in material its teacher never saw.

Evidence and provenance (mediated)
    No words of Huang Daopo survive and no likeness. She is known from two
    accounts written decades after her death: Tao Zongyi, Nancun chuogeng lu
    (1366), juan 24, and the preface to Wang Feng's poem on her shrine, Wuxi ji,
    juan 3 (after 1362). The dates 1245-1330 are a modern convention; the only
    anchored date is her return from Yazhou (Hainan) in the Yuanzhen era
    (1295-1297). The mechanism rests on these reports of what she did, not on
    any doctrine of hers. The link from "teach the making of tools for each
    stage" to "regenerate each stage from its own intake" is this chapter's
    reading (evidence D7, basis: speculation).

Doctrine -> mechanism -> test
    D1 taught the MAKING of gin, bow,      M1 one maker (hypernetwork) shared    H-SIG, H-MADE
       wheel and loom (教以做造捍弹纺织之具)   by all stages; tools are outputs
    D2 before her the upstream stages      M2 measure each stage's intake AFTER  H-NEC, H-RIVAL
       were the hard ones (手剖去子 ...)       the upstream tools have acted
    D3 taught women of other surnames;     M3 deployment is label-free on new    H-SIG (shifted)
       craft spread after her death           villages: no teacher, no targets
    D4 her name faded; the quilt took      B1 what an upstream stage erases, no  H-BLIND
       the adopting village's name            later stage can measure
    D7 (reading) each tool makes the       M2 as above                           H-RIVAL
       material legible to the next maker

Research question (out-of-distribution shift; sample efficiency and transfer)
    When a multi-stage model must adapt to a new input distribution with no
    labels, should each stage be regenerated from statistics measured at its
    own input (after upstream stages were regenerated), from one context
    measured on the raw input, or only re-standardized at each interface?

Closest prior art and the delta
    HyperTransformer (Zhmoginov, Sandler, Vladymyrov, ICML 2022) generates CNN
    weights layer by layer from a labelled support set's activations;
    GeneralizeFormer (Ambekar et al., WACV 2025) generates layer-wise
    normalization and classifier parameters per unlabelled target batch;
    ARM-CML (Zhang et al., NeurIPS 2021) and CNAPs (Requeima et al., NeurIPS
    2019) adapt from one context embedding of the raw inputs; AdaBN (Li et
    al., 2016) re-estimates per-layer statistics. Delta: a single maker for
    heterogeneous stages, driven only by label-free moments at each stage's own
    intake, tested head-to-head against a size-matched raw-context maker (the
    ARM/CNAP choice, same maker, only the measuring point moved), against
    per-interface re-standardization, and against transferring fixed tools,
    with a declared condition where it must lose.

Blind spot (B1)
    Measuring only what arrives forgets where it came from. If a late stage
    needs a property that an early stage destroys (seed load, visible before
    ginning, sets how a dye takes at the loom), the interface maker cannot see
    it and the raw-context maker can. Grounding: the craft outlived Huang
    Daopo while her name faded (Tao) and the Yazhou quilt was renamed for the
    adopting village (Wang Feng); the chain she installed also stayed a
    household chain for centuries (Elvin 1972; P. Huang 1990).

Task (synthetic cotton chain; one village = one lot of 32 tufts)
    tuft quality z ~ N(0, diag(1.5, .55, 1, 1, 1)); target y = z / scale.
    weave damage:  z4 <- z4 + delta + gamma * z2            (dye cast, bleed)
    spin damage:   (z0, z1) rotated by theta                (twist; staple)
    bow damage:    z0..z3 <- asinh(kappa * z) / kappa        (matting)
    gin damage:    seeded tufts (prob p) gain a * u along a village direction
                   u; channel 5 records the seed weight a; ginning removes it.
    Source villages (Yazhou) and shifted villages (short staple, damp delta,
    heavier seed, stronger cast). Blind-spot world: the loom must undo a stain
    gain set by seed load while natural colour spread varies by village.

Limits
    A research prototype of one AGI-relevant mechanism on synthetic data, not
    an AGI and not a model of Huang Daopo's mind. Tools are affine maps with a
    bounded cubic; the maker sees only four moment families. Results are never
    evidence about the historical person.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 1,
    "revision_log": [],
    "generation": {"template_version": "codeguidelines 1.0 (2026-09-15)", "generator": "Claude",
                   "generator_version": "Opus 5.5", "date": "2026-09-29"},
    "id": 346, "figure": "Huang Daopo", "born": 1245, "died": 1330, "civilization": "Chinese (Yuan)",
    "provenance": "mediated",
    "thesis": ("Transfer the making, not the made: one maker shared by every stage rebuilds each "
               "stage's tool from a label-free measurement of the material as it reaches that stage, "
               "so each finished tool makes the next stage measurable and the chain survives its teacher."),
    "evidence": [
        {"id": "D1", "claim": "Taught the making of tools for ginning, bowing, spinning and weaving, and "
         "methods for crossing yarns, matching colours, heddling and lifting figures, each with its own method.",
         "basis": "deeds", "source": "Tao Zongyi, Nancun chuogeng lu (1366), juan 24, 'Huang Daopo'"},
        {"id": "D2", "claim": "Before her, seeds were picked out by hand and fibre was opened with a small "
         "string-and-bamboo bow on a table; the work was very hard.",
         "basis": "deeds", "source": "Tao Zongyi, Nancun chuogeng lu, juan 24"},
        {"id": "D3", "claim": "Returned from Yazhou by sea in the Yuanzhen era, taught women of other surnames "
         "without tiring; the quilt was soon renamed for Wunijing and over a thousand households lived by it.",
         "basis": "deeds", "source": "Wang Feng, Wuxi ji, juan 3, 'Huang Daopo ci' and preface (after 1362)"},
        {"id": "D4", "claim": "She died soon after; the shrine was destroyed and rebuilt; her name was "
         "fading from memory when Tao wrote.", "basis": "deeds", "source": "Tao Zongyi, juan 24"},
        {"id": "D5", "claim": "Li textile skills are transmitted from mothers to daughters by instruction "
         "and demonstration.", "basis": "scholarship",
         "source": "UNESCO ICH element 00302 (2009); transferred to the Representative List 2024"},
        {"id": "D6", "claim": "The household cotton chain persisted for centuries without mechanising "
         "beyond the household.", "basis": "scholarship",
         "source": "Elvin 1972 (in Willmott, Economic Organization in Chinese Society); P. Huang 1990"},
        {"id": "D7", "claim": "Each tool makes the material legible to the maker of the next tool.",
         "basis": "speculation", "source": "this chapter's reading of D1-D2"},
        {"id": "D8", "claim": "Later tradition credits her with the three-spindle treadle wheel; the earliest "
         "accounts name only tools for four stages.", "basis": "scholarship",
         "source": "Kuhn, SCC 5.9 (1988); Bray, Technology and Gender (1997)"},
    ],
    "research_question": {"category": "out-of-distribution detection and shift",
                          "question": ("Under unlabelled shift in a multi-stage model, does regenerating each "
                                       "stage from moments measured at its own intake beat one raw-input "
                                       "context, per-interface re-standardization and fixed transferred "
                                       "stages, and does it lose where upstream stages erase what a late "
                                       "stage needs?")},
    "mechanism": {
        "name": "ZUO ZAO: interface-measured tool re-making along a chain",
        "family": "hypernetworks and weight-space learning",
        "signature_modules": ["measure", "maker", "stage_tag"],
        "closest_prior_art": [
            "HyperTransformer (Zhmoginov, Sandler, Vladymyrov, ICML 2022)",
            "GeneralizeFormer (Ambekar, Xiao, Zhen, Snoek, WACV 2025)",
            "ARM-CML context network (Zhang et al., NeurIPS 2021); CNAPs (Requeima et al., NeurIPS 2019)",
            "AdaBN (Li et al., arXiv 1603.04779, 2016); TENT (Wang et al., ICLR 2021)",
            "HyperNetworks (Ha, Dai, Le, ICLR 2017)"],
        "overlap": "Medium",
        "prior_art_queries": [
            "HyperTransformer layer-by-layer weights generated from activations of previous layer",
            "hypernetwork generates layer weights from unlabeled test batch statistics layer-wise",
            "adaptive batch normalization practical domain adaptation",
            "adaptive risk minimization context network test-time adaptation"],
        "contribution_type": "test",
        "delta": ("One maker for heterogeneous stages driven only by label-free moments at each stage's own "
                  "intake, compared with a size-matched raw-context maker that differs only in where it "
                  "measures, with per-interface re-standardization and with fixed transferred stages, plus "
                  "a declared provenance-loss condition where it must lose."),
    },
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1", "property_test": "C6.P1", "hypothesis": "H-MADE"},
        {"doctrine": "D2", "mechanism": "M2", "property_test": "C6.P2", "hypothesis": "H-NEC"},
        {"doctrine": "D3", "mechanism": "M3", "property_test": "C7", "hypothesis": "H-SIG"},
        {"doctrine": "D4", "mechanism": "B1", "property_test": "C6.D1", "hypothesis": "H-BLIND"},
        {"doctrine": "D7", "mechanism": "M2", "property_test": "C6.P2", "hypothesis": "H-RIVAL"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "Measuring at each stage's own intake beats measuring once at the raw "
         "intake (same maker, same size) on villages outside the source range.",
         "metric": "R2 over 5 fibre dims", "split": "shifted", "comparison": "model - baseline",
         "direction": "greater", "mesi": 0.03, "seeds": 5},
        {"id": "H-NEC", "statement": "Replacing the interface measurement by its training mean hurts more "
         "than removing the global readout.", "metric": "R2 held-out",
         "comparison": "signature_knockout - matched_knockout", "direction": "less", "mesi": 0.05, "seeds": 5},
        {"id": "H-BLIND", "statement": "Where the loom needs the seed load that ginning erased, the "
         "interface maker loses to the raw-context maker.", "condition": "blind-spot world",
         "grounding": "her name faded while the craft spread; what a stage erases later stages cannot know",
         "metric": "colour R2 (dim 4) held-out", "comparison": "model - baseline", "direction": "less",
         "mesi": 0.03, "seeds": 5},
        {"id": "H-RIVAL", "statement": "Re-making beats re-standardizing at the same measuring points.",
         "metric": "R2 over 5 fibre dims", "split": "shifted", "comparison": "model - adabn_chain",
         "direction": "greater", "mesi": 0.03, "seeds": 5},
        {"id": "H-MADE", "statement": "Handing over the maker beats handing over fixed tools of equal size.",
         "metric": "R2 over 5 fibre dims", "split": "shifted", "comparison": "model - fixed_chain",
         "direction": "greater", "mesi": 0.05, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.5, "margin_over_trivial": 0.3, "shuffled_band": 0.05},
    "probe_predictions": [
        {"probe": "P1", "expected": "below baseline"}, {"probe": "P5", "expected": "equal to baseline"},
        {"probe": "P6", "expected": "above baseline"}, {"probe": "P7", "expected": "above baseline"},
        {"probe": "P8", "expected": "equal to baseline"}, {"probe": "P9", "expected": "above baseline"}],
    "dialectic_links": [],
    "corpus_neighbors": [
        {"chapter": 141, "similarity": None, "difference": "Ma Jun compresses one machine's controls; here "
         "nothing is compressed and each stage's tool is regenerated from local measurement."},
        {"chapter": 416, "similarity": None, "difference": "Kabir puts content in the warp-weft crossing; "
         "here there is no crossing, only a sequence in which each output is the next measuring instrument."},
        {"chapter": 345, "similarity": None, "difference": "Araniko regrows a missing part from a canon of "
         "the whole carried in the mind; here no whole is carried, only the maker."},
        {"chapter": 339, "similarity": None, "difference": "Guo Shoujing anchors a forecast at the last "
         "measurement in time; here measurement is taken at the last stage in a chain."},
        {"chapter": 342, "similarity": None, "difference": "Yunus Emre survives by redundant carriers of "
         "fixed content; here survival comes from regenerating tools, not from copies."}],
    "barometer": {
        "cognitive_processing": ["P6", "P10"], "embodied_cognition": ["P9"], "world_modeling": ["P7"],
        "consciousness": ["P8"], "language_understanding": [], "emotional_intelligence": [],
        "creativity": [], "autonomy": ["P5"]},
    "task_types": ["vector_regression"],
    "applications": [
        {"use": "Label-free re-tuning of each stage of a multi-stage vision pipeline after a sensor or "
         "lighting change", "sector": "computer vision / MLOps",
         "dataset": "CIFAR-10-C (Hendrycks and Dietterich, ICLR 2019)"},
        {"use": "Stage-wise adaptation of quality prediction when a raw-material lot changes on a "
         "multi-station line", "sector": "manufacturing",
         "dataset": "Bosch Production Line Performance (Kaggle, 2016)"},
        {"use": "Unit-by-unit soft-sensor adaptation in a multi-unit chemical process after a disturbance",
         "sector": "process industry",
         "dataset": "Tennessee Eastman Process simulation data (Rieth et al., Harvard Dataverse, 2017)"}],
    "safety_notes": ("Synthetic textile data only; no hazardous process; no claim to replicate a real "
                     "person's mind; provenance mediated and stated. Similarity values not computed in "
                     "session (neighbour files unavailable); flagged for the corpus audit."),
}

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

D_NATIVE, OUT_NATIVE, N_STAGES, N_ITEMS = 6, 5, 4, 32
SEED_CH = 5                      # channel holding the seed weight; removed by ginning
EPS_V = 1e-3                     # variance floor in the moment features
L2_CUBIC = 9.0                   # bounded cubic: a^3 / (1 + a^2 / 9)
HIDDEN = 24                      # maker width
HIDDEN_ENC, HIDDEN_TOOL = 8, 40  # raw-context encoder; fixed / re-standardizing tool width
LR, CLIP, BATCH_V = 5e-3, 5.0, 8
STEPS_FULL, STEPS_QUICK, EVAL_EVERY = 600, 250, 50
Z_SCALE = np.array([1.5, 0.55, 1.0, 1.0, 1.0])
BUDGET_FULL, BUDGET_QUICK = 180.0, 20.0


class NonFinite(Exception):
    pass


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


class Adam:
    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-8):
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
    norm = math.sqrt(sum(float((g ** 2).sum()) for g in grads.values()))
    if not math.isfinite(norm):
        raise NonFinite("non-finite gradient norm")
    scale = min(1.0, max_norm / (norm + 1e-12))
    return {k: g * scale for k, g in grads.items()}, norm


def gradcheck(loss_fn, params, grads, rng, n_random=20, eps=1e-6, tol=1e-5, small=1e-5, abs_tol=1e-9):
    """Central differences on >= n_random entries per tensor plus its largest-gradient entry.
    Entries whose analytic and numeric gradients are both below `small` are compared by
    absolute difference, because relative error is undefined near zero."""
    worst, checked, failures = 0.0, 0, []
    for name, p in params.items():
        g, flat = grads[name].ravel(), p.ravel()
        idx = set(rng.choice(flat.size, size=min(n_random, flat.size), replace=False).tolist())
        idx.add(int(np.argmax(np.abs(g))))
        for i in idx:
            old = flat[i]
            flat[i] = old + eps
            lp = loss_fn()
            flat[i] = old - eps
            lm = loss_fn()
            flat[i] = old
            num, ana = (lp - lm) / (2 * eps), g[i]
            if max(abs(num), abs(ana)) < small:
                ok = abs(num - ana) <= abs_tol
            else:
                rel = abs(num - ana) / (abs(num) + abs(ana))
                worst, ok = max(worst, rel), rel <= tol
            if not ok:
                failures.append((name, int(i), float(ana), float(num)))
        checked += 1
    return {"tensors_checked": checked, "tensors_total": len(params), "max_rel_error": worst,
            "passed": not failures and checked == len(params), "failures": failures[:5]}


def paired_bootstrap(diffs, rng, n_boot=2000):
    d = np.asarray(diffs, dtype=float)
    boots = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def verdict(mean, lo, hi, direction, mesi):
    sgn = 1.0 if direction == "greater" else -1.0
    if sgn * lo > 0 and sgn * hi > 0 and sgn * mean >= mesi:
        return "supported"
    if sgn * lo < 0 and sgn * hi < 0:
        return "contradicted"
    return "inconclusive"


def write_report_json(path, report):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
# END STANDARD UTILITIES


# ----------------------------------------------------------------------------- data and tasks
def make_villages(rng, V, regime="source", blind=False, N=N_ITEMS):
    """One village = one lot of N tufts. Damage runs weave -> spin -> bow -> gin; the chain
    must undo it in the order gin -> bow -> spin -> weave."""
    u0 = np.array([1.0, 1.0, 0.5, 0.5, 0.8]) / np.linalg.norm([1.0, 1.0, 0.5, 0.5, 0.8])
    if regime == "source":
        th, kap, ps = rng.uniform(.15, .75, V), rng.uniform(.3, .9, V), rng.uniform(.25, .5, V)
        dl, ga, un = rng.uniform(-.8, .8, V), rng.uniform(-.5, .5, V), 0.6
    else:  # shifted: shorter staple, damper delta, heavier seed, stronger cast
        th, kap, ps = rng.uniform(.8, 1.1, V), rng.uniform(.95, 1.25, V), rng.uniform(.5, .65, V)
        dl = rng.choice([-1.0, 1.0], V) * rng.uniform(.8, 1.2, V)
        ga, un = rng.choice([-1.0, 1.0], V) * rng.uniform(.5, .7, V), 0.9
    u = u0 + un * rng.normal(size=(V, 5))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    s4 = rng.uniform(.6, 1.4, V) if blind else np.ones(V)
    gain = 1.0 + 1.6 * (ps - 0.25) if blind else np.ones(V)
    z = rng.normal(size=(V, N, 5)) * Z_SCALE
    z[..., 4] *= s4[:, None]
    t = z.copy()
    t[..., 4] = z[..., 4] * gain[:, None] + dl[:, None] + ga[:, None] * z[..., 2]
    c, s = np.cos(th)[:, None], np.sin(th)[:, None]
    t[..., 0], t[..., 1] = c * z[..., 0] - s * z[..., 1], s * z[..., 0] + c * z[..., 1]
    t[..., :4] = np.arcsinh(kap[:, None, None] * t[..., :4]) / kap[:, None, None]
    a = (rng.random((V, N)) < ps[:, None]) * (2.5 + 1.5 * np.abs(rng.normal(size=(V, N))))
    x = np.zeros((V, N, D_NATIVE))
    x[..., :5] = t + a[..., None] * u[:, None, :]
    x[..., SEED_CH] = a
    x += 0.03 * rng.normal(size=x.shape)
    return x, z / Z_SCALE


def make_data(seed_seq):
    g = [np.random.default_rng(c) for c in seed_seq.spawn(7)]
    return {"train": make_villages(g[0], 192), "val": make_villages(g[1], 32),
            "held": make_villages(g[2], 48), "shift": make_villages(g[3], 48, "shifted"),
            "b_train": make_villages(g[4], 192, blind=True), "b_val": make_villages(g[5], 32, blind=True),
            "b_held": make_villages(g[6], 48, blind=True)}


def r2(yhat, y, dims=None):
    yh, yy = yhat.reshape(-1, yhat.shape[-1]), y.reshape(-1, y.shape[-1])
    dims = range(yy.shape[1]) if dims is None else dims
    out = [1.0 - ((yh[:, j] - yy[:, j]) ** 2).sum() / ((yy[:, j] - yy[:, j].mean()) ** 2).sum() for j in dims]
    return float(np.mean(out))


# ----------------------------------------------------------------------------- model: measure, maker, tool
def _pairs(D):
    return np.triu_indices(D, 1)


def feat_dim(D):
    return 3 * D + D * (D - 1) // 2


def feat_forward(S):
    """Label-free moments of a lot: mean, log-variance, excess kurtosis (asinh), correlations."""
    N, D = S.shape[1], S.shape[2]
    mu = S.mean(1)
    Sc = S - mu[:, None, :]
    v = (Sc ** 2).mean(1)
    sd = np.sqrt(v + EPS_V)
    z = Sc / sd[:, None, :]
    kraw = (z ** 4).mean(1) - 3.0
    pi, pj = _pairs(D)
    r = np.einsum("vni,vnj->vij", z, z)[:, pi, pj] / N
    m = np.concatenate([mu, np.log(v + EPS_V), np.arcsinh(kraw), r], axis=1)
    return m, (Sc, v, sd, z, kraw)


def _z_backward(dz, Sc, v, sd, dlv=None):
    N = Sc.shape[1]
    dSc = dz / sd[:, None, :]
    dsd = -(dz * Sc).sum(1) / sd ** 2
    dv = dsd * 0.5 / sd + (0.0 if dlv is None else dlv / (v + EPS_V))
    dSc = dSc + dv[:, None, :] * 2.0 * Sc / N
    return dSc - dSc.mean(1, keepdims=True)


def feat_backward(dm, cache):
    Sc, v, sd, z, kraw = cache
    N, D = Sc.shape[1], Sc.shape[2]
    dmu, dlv, dkf, dr = dm[:, :D], dm[:, D:2 * D], dm[:, 2 * D:3 * D], dm[:, 3 * D:]
    dz = (dkf / np.sqrt(1.0 + kraw ** 2))[:, None, :] * 4.0 * z ** 3 / N
    G = np.zeros((dm.shape[0], D, D))
    pi, pj = _pairs(D)
    G[:, pi, pj] = dr
    dz = dz + np.einsum("vnj,vij->vni", z, G + G.transpose(0, 2, 1)) / N
    return _z_backward(dz, Sc, v, sd, dlv) + dmu[:, None, :] / N


def maker_forward(m, tag, P, zero_hidden=False):
    q = np.concatenate([m, np.broadcast_to(tag, (m.shape[0], tag.shape[-1]))], axis=1)
    h = np.zeros((q.shape[0], P["W1"].shape[1])) if zero_hidden else np.tanh(q @ P["W1"] + P["b1"])
    return h @ P["W2"] + P["b2"], (q, h, zero_hidden)


def maker_backward(dth, cache, P, G):
    q, h, zero_hidden = cache
    G["W2"] += h.T @ dth
    G["b2"] += dth.sum(0)
    if zero_hidden:
        return np.zeros((q.shape[0], q.shape[1] - N_STAGES))
    dpre = (dth @ P["W2"].T) * (1.0 - h ** 2)
    G["W1"] += q.T @ dpre
    G["b1"] += dpre.sum(0)
    return (dpre @ P["W1"].T)[:, :-N_STAGES]


def gtool_forward(S, th):
    """Generated tool: affine map (I + U) with offset c, then a bounded elementwise cubic."""
    V, D = S.shape[0], S.shape[2]
    U, c, rr = th[:, :D * D].reshape(V, D, D), th[:, D * D:D * D + D], th[:, D * D + D:]
    rho = 0.5 * np.tanh(rr)
    A = S + np.einsum("vnd,vde->vne", S, U) + c[:, None, :]
    sc = A ** 3 / (1.0 + A ** 2 / L2_CUBIC)
    return A + rho[:, None, :] * sc, (S, U, A, sc, rho)


def gtool_backward(dO, cache):
    S, U, A, sc, rho = cache
    a2 = A ** 2
    dsc = (3.0 * a2 + a2 ** 2 / L2_CUBIC) / (1.0 + a2 / L2_CUBIC) ** 2
    dA = dO * (1.0 + rho[:, None, :] * dsc)
    drr = (dO * sc).sum(1) * 0.5 * (1.0 - (2.0 * rho) ** 2)
    dU = np.einsum("vnd,vne->vde", S, dA)
    dS = dA + np.einsum("vne,vde->vnd", dA, U)
    return dS, np.concatenate([dU.reshape(S.shape[0], -1), dA.sum(1), drr], axis=1)


def _gin_mask(model):
    mask = np.ones(model["D"])
    if model["gin"] is not None:
        mask[model["gin"]] = 0.0
    return mask


def _alt_pool(S, how):
    """Negative-control measurements for the property tests only (no gradient)."""
    if how == "first":
        return np.concatenate([S[:, 0, :], np.zeros((S.shape[0], feat_dim(S.shape[2]) - S.shape[2]))], 1)
    m, _ = feat_forward(S)
    m[:, :S.shape[2]] *= S.shape[1]          # sum instead of mean
    return m


def _measure(model, k, S):
    if model.get("pool"):
        return _alt_pool(S, model["pool"]), None
    mode = model["ko"].get("measure")
    m, fc = feat_forward(S)
    if mode == "seed_load" and k == 0 and model["gin"] is not None:
        idx = [model["gin"] + j * model["D"] for j in range(3)]   # mean, log-var, kurtosis of seed channel
        m[:, idx] = model["ref"]["feat"][0][idx]
        return m, None
    if mode == "mean":
        return np.broadcast_to(model["ref"]["feat"][k], m.shape).copy(), None
    if mode == "zero":
        return np.zeros_like(m), None
    return m, fc


def _tag(model, k):
    mode = model["ko"].get("stage_tag")
    if mode == "mean":
        return np.full(N_STAGES, 1.0 / N_STAGES)
    return np.zeros(N_STAGES) if mode == "zero" else np.eye(N_STAGES)[k]


def _finish_forward(S, model):
    mode, P, out = model["ko"].get("finish"), model["params"], model["out"]
    if mode == "identity":
        return S[..., :out].copy()
    if mode == "zero":
        return np.zeros(S.shape[:2] + (out,))
    return S @ P["F"] + P["f"]


def _forward_maker(model, X):
    P, kind, st = model["params"], model["kind"], []
    raw = None
    if kind == "raw":
        Hh = np.tanh(X @ P["Ea"] + P["ba"])
        E = X + Hh @ P["Eb"]
        m0, fc0 = (_measure(model, 0, E))
        raw = (X, Hh, fc0)
    S = X
    for k in range(N_STAGES):
        m, fc = _measure(model, k, S) if kind == "interface" else (m0, None)
        th, mc = maker_forward(m, _tag(model, k), P, model["ko"].get("maker") == "zero")
        O, tc = gtool_forward(S, th)
        if k == 0:
            O = O * _gin_mask(model)
        st.append({"fc": fc, "mc": mc, "tc": tc, "m": m, "th": th, "out": O})
        S = O
    return _finish_forward(S, model), {"stages": st, "raw": raw, "S4": S}


def _backward_maker(model, cache, dy):
    P, kind = model["params"], model["kind"]
    G = {k: np.zeros_like(v) for k, v in P.items()}
    dS = _finish_backward(model, cache["S4"], dy, G)
    dm_raw = 0.0
    for k in reversed(range(N_STAGES)):
        s = cache["stages"][k]
        if k == 0:
            dS = dS * _gin_mask(model)
        dS_in, dth = gtool_backward(dS, s["tc"])
        dm = maker_backward(dth, s["mc"], P, G)
        if kind == "interface":
            if s["fc"] is not None and model["mutant"] != "no_measure_grad":
                dS_in = dS_in + feat_backward(dm, s["fc"])
        else:
            dm_raw = dm_raw + dm
        dS = dS_in
    if kind == "raw" and cache["raw"][2] is not None:
        X, Hh, fc0 = cache["raw"]
        dE = feat_backward(dm_raw, fc0)
        G["Eb"] += np.einsum("vnh,vnd->hd", Hh, dE)
        dpre = np.einsum("vnd,hd->vnh", dE, P["Eb"]) * (1.0 - Hh ** 2)
        G["Ea"] += np.einsum("vnd,vnh->dh", X, dpre)
        G["ba"] += dpre.sum((0, 1))
    return G


def _finish_backward(model, S4, dy, G):
    mode, P = model["ko"].get("finish"), model["params"]
    if mode == "identity":
        dS = np.zeros_like(S4)
        dS[..., :model["out"]] = dy
        return dS
    if mode == "zero":
        return np.zeros_like(S4)
    G["F"] += np.einsum("vnd,vno->do", S4, dy)
    G["f"] += dy.sum((0, 1))
    return dy @ P["F"].T


# ----------------------------------------------------------------------------- baselines and rivals
def _std_forward(S):
    mu = S.mean(1)
    Sc = S - mu[:, None, :]
    v = (Sc ** 2).mean(1)
    sd = np.sqrt(v + EPS_V)
    return Sc / sd[:, None, :], (Sc, v, sd)


def _forward_mlp_chain(model, X):
    """Rivals: 'fixed' hands over finished tools (the made); 'adabn' re-standardizes every
    intake on the lot, then applies finished tools (measure, but do not re-make)."""
    P, ada, st, S = model["params"], model["kind"] == "adabn", [], X
    for k in range(N_STAGES):
        Z, zc = _std_forward(S) if ada else (S, None)
        H = np.tanh(Z @ P["Wa%d" % k] + P["ba%d" % k])
        O = (Z * P["g%d" % k] if ada else Z) + H @ P["Wb%d" % k] + P["bb%d" % k]
        if k == 0:
            O = O * _gin_mask(model)
        st.append({"Z": Z, "zc": zc, "H": H, "out": O})
        S = O
    return _finish_forward(S, model), {"stages": st, "S4": S}


def _backward_mlp_chain(model, cache, dy):
    P, ada = model["params"], model["kind"] == "adabn"
    G = {k: np.zeros_like(v) for k, v in P.items()}
    dS = _finish_backward(model, cache["S4"], dy, G)
    for k in reversed(range(N_STAGES)):
        s = cache["stages"][k]
        if k == 0:
            dS = dS * _gin_mask(model)
        G["Wb%d" % k] += np.einsum("vnh,vnd->hd", s["H"], dS)
        G["bb%d" % k] += dS.sum((0, 1))
        dpre = np.einsum("vnd,hd->vnh", dS, P["Wb%d" % k]) * (1.0 - s["H"] ** 2)
        G["Wa%d" % k] += np.einsum("vnd,vnh->dh", s["Z"], dpre)
        G["ba%d" % k] += dpre.sum((0, 1))
        dZ = np.einsum("vnh,dh->vnd", dpre, P["Wa%d" % k])
        if ada:
            G["g%d" % k] += (dS * s["Z"]).sum((0, 1))
            dS = _z_backward(dZ + dS * P["g%d" % k], *s["zc"])
        else:
            dS = dZ + dS
    return G


# ----------------------------------------------------------------------------- interface (common contract)
TASK_TYPES = ["vector_regression"]


def build_model(in_dim, out_dim, task_type="vector_regression", rng=None, kind="interface",
                gin_channel=None, hidden=HIDDEN):
    if task_type not in TASK_TYPES:
        raise ValueError("unsupported task type: %s" % task_type)
    rng = np.random.default_rng(0) if rng is None else rng
    D, M, Pn = in_dim, feat_dim(in_dim), in_dim * in_dim + 2 * in_dim
    P = {}
    if kind in ("interface", "raw"):
        P["W1"] = rng.normal(0, 0.8 / math.sqrt(M + N_STAGES), (M + N_STAGES, hidden))
        P["b1"], P["W2"], P["b2"] = np.zeros(hidden), rng.normal(0, 0.02, (hidden, Pn)), np.zeros(Pn)
        if kind == "raw":
            P["Ea"], P["ba"] = rng.normal(0, 0.3, (D, HIDDEN_ENC)), np.zeros(HIDDEN_ENC)
            P["Eb"] = rng.normal(0, 0.05, (HIDDEN_ENC, D))
    elif kind in ("fixed", "adabn"):
        for k in range(N_STAGES):
            P["Wa%d" % k], P["ba%d" % k] = rng.normal(0, 1 / math.sqrt(D), (D, HIDDEN_TOOL)), np.zeros(HIDDEN_TOOL)
            P["Wb%d" % k], P["bb%d" % k] = rng.normal(0, 0.05, (HIDDEN_TOOL, D)), np.zeros(D)
            if kind == "adabn":
                P["g%d" % k] = np.ones(D)
    else:
        raise ValueError("unknown kind: %s" % kind)
    P["F"], P["f"] = np.eye(D, out_dim), np.zeros(out_dim)
    return {"kind": kind, "params": P, "D": D, "out": out_dim, "gin": gin_channel,
            "ko": {}, "mutant": None, "ref": {}, "pool": None}


def forward(model, X):
    if model["kind"] in ("interface", "raw"):
        return _forward_maker(model, X)
    return _forward_mlp_chain(model, X)


def loss_and_grads(model, batch):
    X, Y = batch
    yhat, cache = forward(model, X)
    diff = yhat - Y
    loss = float((diff ** 2).mean())
    dy = 2.0 * diff / diff.size
    back = _backward_maker if model["kind"] in ("interface", "raw") else _backward_mlp_chain
    G = back(model, cache, dy)
    if model["mutant"] == "zero_grad_W2" and "W2" in G:
        G["W2"][:] = 0.0
    return loss, G


def predict(model, X):
    two_d = X.ndim == 2
    yhat = forward(model, X[None] if two_d else X)[0]
    return yhat[0] if two_d else yhat


def hidden_states(model, X):
    _, cache = forward(model, X[None] if X.ndim == 2 else X)
    return {"stage_%d" % k: s["out"] for k, s in enumerate(cache["stages"])}


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


def modules(model):
    k = model["kind"]
    if k in ("interface", "raw"):
        m = {"measure": ([], "label-free moment pooling (mean, log-variance, excess kurtosis, pairwise "
                         "correlation) of the lot at " + ("each stage input" if k == "interface" else
                                                          "the encoded raw input"), True),
             "maker": (["W1", "b1", "W2", "b2"], "hypernetwork shared across stages generating each "
                       "stage's affine-plus-bounded-cubic weights", True),
             "stage_tag": ([], "one-hot stage index fed to the shared hypernetwork", True),
             "finish": (["F", "f"], "global linear readout", False)}
        if k == "raw":
            m["encoder"] = (["Ea", "ba", "Eb"], "per-item residual encoder before pooling", False)
    else:
        m = {"stage%d" % s: (["Wa%d" % s, "ba%d" % s, "Wb%d" % s, "bb%d" % s] + (["g%d" % s] if k == "adabn"
                            else []), "item-wise residual MLP tool" + (" on re-standardized input"
                            if k == "adabn" else ""), False) for s in range(N_STAGES)}
        m["finish"] = (["F", "f"], "global linear readout", False)
    return {n: {"params": p, "role": r, "signature": s} for n, (p, r, s) in m.items()}


def knockout(model, name, mode):
    allowed = {"measure": ("mean", "zero", "seed_load"), "stage_tag": ("mean", "zero"), "maker": ("zero",),
               "finish": ("identity", "zero")}
    if name not in allowed or mode not in allowed[name]:
        raise ValueError("unsupported knockout %s/%s" % (name, mode))
    ko = dict(model, params={k: v.copy() for k, v in model["params"].items()}, ko=dict(model["ko"]))
    ko["ko"][name] = mode
    return ko


MUTANTS = {"sign_flip": "optimizer ascends the loss",
           "zero_lr": "learning rate set to zero",
           "zero_grad_W2": "analytic gradient of the maker's output weights zeroed",
           "no_measure_grad": "gradient through the interface measurement dropped"}


# ----------------------------------------------------------------------------- training
def _set_reference(model, X):
    if model["kind"] not in ("interface", "raw"):
        return
    ko_saved, model["ko"] = model["ko"], {}
    _, cache = forward(model, X[:64])
    model["ko"] = ko_saved
    model["ref"] = {"feat": [s["m"].mean(0) for s in cache["stages"]]}


def _as_lots(X, Y):
    """Probe-battery inputs arrive as (n, d); group consecutive rows into lots of N_ITEMS."""
    if X.ndim == 3:
        return X, Y
    Y = Y.reshape(len(Y), -1)
    N = min(N_ITEMS, X.shape[0])
    V = X.shape[0] // N
    return X[:V * N].reshape(V, N, -1), Y[:V * N].reshape(V, N, -1)


def fit(model, data, budget, rng, lr=LR):
    X, Y = _as_lots(data[0], data[1])
    val = _as_lots(data[2], data[3]) if len(data) > 2 else None
    opt = Adam(model["params"], 0.0 if model["mutant"] == "zero_lr" else lr)
    sign = -1.0 if model["mutant"] == "sign_flip" else 1.0
    losses, best = [], (float("inf"), None)
    for t in range(budget):
        idx = rng.choice(X.shape[0], size=min(BATCH_V, X.shape[0]), replace=False)
        loss, G = loss_and_grads(model, (X[idx], Y[idx]))
        if not math.isfinite(loss):
            raise NonFinite("non-finite training loss")
        G, _ = clip_global_norm(G, CLIP)
        opt.step(model["params"], G, sign)
        losses.append(loss)
        if val is not None and ((t + 1) % EVAL_EVERY == 0 or t == budget - 1):
            vl = float(((predict(model, val[0]) - val[1]) ** 2).mean())
            if vl < best[0]:
                best = (vl, {k: v.copy() for k, v in model["params"].items()})
    if best[1] is not None:
        model["params"] = best[1]
    _set_reference(model, X)
    return np.array(losses)


# ----------------------------------------------------------------------------- correctness tests (C1-C8)
KINDS = ("interface", "raw", "fixed", "adabn")
KO_LIST = [("measure", "mean"), ("stage_tag", "mean"), ("maker", "zero"), ("finish", "identity")]


def _new(kind, rng, mutant=None, pool=None):
    model = build_model(D_NATIVE, OUT_NATIVE, "vector_regression", rng, kind=kind, gin_channel=SEED_CH)
    model["mutant"], model["pool"] = mutant, pool
    return model


def _loss_only(model, batch):
    return float(((forward(model, batch[0])[0] - batch[1]) ** 2).mean())


def c1_gradcheck(data, rng, mutant=None, kinds=KINDS):
    """Every tensor of every model kind, at initialization and after 60 updates."""
    X, Y = data["train"]
    batch, rows = (X[:3, :10], Y[:3, :10]), {}
    for kind in kinds:
        model = _new(kind, rng, mutant)
        for when in ("init", "after_training_steps"):
            if when == "after_training_steps":
                fit(model, data["train"], 60, rng)
            _, G = loss_and_grads(model, batch)
            rows[(kind, when)] = gradcheck(lambda: _loss_only(model, batch), model["params"], G, rng)
    ok = all(r["passed"] for r in rows.values())
    worst = max(r["max_rel_error"] for r in rows.values())
    n_chk = sum(r["tensors_checked"] for (k, w), r in rows.items() if w == "init")
    n_tot = sum(r["tensors_total"] for (k, w), r in rows.items() if w == "init")
    fails = [(k, w, r["failures"][:1]) for (k, w), r in rows.items() if not r["passed"]]
    return {"tensors_checked": n_chk, "tensors_total": n_tot, "max_rel_error": worst,
            "checked_at": ["init", "after_training_steps"], "passed": ok, "failures": fails[:3]}


def c2_determinism(data, seed):
    outs = []
    for _ in range(2):
        rng = np.random.default_rng(seed)
        model = _new("interface", rng)
        losses = fit(model, data["train"], 30, rng)
        outs.append((losses, predict(model, data["held"][0]), model))
    same = np.array_equal(outs[0][0], outs[1][0]) and np.array_equal(outs[0][1], outs[1][1])
    model = outs[0][2]
    finite = all(np.isfinite(v).all() for v in model["params"].values()) and all(
        np.isfinite(h).all() for h in hidden_states(model, data["held"][0]).values())
    return same and finite, "identical losses/outputs: %s; finite params and activations: %s" % (same, finite)


def c3_learning(data, steps, rng, thr, mutant=None):
    model = _new("interface", rng, mutant)
    try:
        losses = fit(model, data["train"] + data["val"], steps, rng)
    except NonFinite:
        return False, "non-finite during training", model, float("nan"), float("nan")
    drop = 1.0 - losses[-50:].mean() / losses[:20].mean()
    X, Y = data["held"]
    held = r2(predict(model, X), Y)
    trivial = r2(np.broadcast_to(data["train"][1].mean((0, 1)), Y.shape), Y)
    ok = bool(drop >= thr["loss_drop_fraction"] and held - trivial >= thr["margin_over_trivial"])
    return ok, "loss drop %.3f (need %.2f); held R2 %.3f vs trivial %.3f (margin %.2f)" % (
        drop, thr["loss_drop_fraction"], held, trivial, thr["margin_over_trivial"]), model, drop, held


def c4_shuffled(data, steps, rng, thr):
    def shuf(Y):
        flat = Y.reshape(-1, Y.shape[-1])
        return flat[rng.permutation(flat.shape[0])].reshape(Y.shape)
    model = _new("interface", rng)
    fit(model, (data["train"][0], shuf(data["train"][1]), data["val"][0], shuf(data["val"][1])), steps, rng)
    held = r2(predict(model, data["held"][0]), data["held"][1])
    return held <= thr["shuffled_band"], "held R2 after training on permuted targets %.3f (band %.2f)" % (
        held, thr["shuffled_band"])


def c5_mutants(data, rng, thr, steps=150):
    """A mutant is detected when it makes C1 or a short C3 fail; the unmutated control must pass both."""
    rows = {}
    for name in [None] + list(MUTANTS):
        gc = c1_gradcheck(data, rng, name, kinds=("interface",))["passed"]
        learn = c3_learning(data, steps, rng, thr, name)[0]
        rows[name or "control"] = (gc, learn)
    detected = sum(1 for k, (g, l) in rows.items() if k != "control" and not (g and l))
    control_ok = all(rows["control"])
    detail = "; ".join("%s: C1 %s, C3 %s" % (k, "pass" if g else "FAIL", "pass" if l else "FAIL")
                       for k, (g, l) in rows.items())
    return control_ok and detected == len(MUTANTS), detected, detail


def _perm_violation(model, X, rng):
    """Relative, because summation order alone moves the last bits of large outputs."""
    p, base = rng.permutation(X.shape[1]), predict(model, X)
    return float(np.abs(base[:, p] - predict(model, X[:, p])).max() / (1.0 + np.abs(base).max()))


def _rep_violation(model, X):
    base = predict(model, X)
    twice = predict(model, np.concatenate([X, X], axis=1))[:, :X.shape[1]]
    return float(np.abs(base - twice).max() / (1.0 + np.abs(base).max()))


def c6_properties(data, rng, n_search=6):
    """P1 lot-permutation equivariance and P2 replication invariance of the tool makers, searched
    over random parameters and inputs, each with a negative control that must break it."""
    worst = {"P1": 0.0, "P2": 0.0, "P1_neg": 0.0, "P2_neg": 0.0}
    for t in range(n_search):
        X = data["held"][0][:4] if t % 2 == 0 else 1.5 * rng.normal(size=(4, N_ITEMS, D_NATIVE))
        for kind in ("interface", "raw"):
            model = _new(kind, rng)
            for key in model["params"]:
                model["params"][key] = model["params"][key] + 0.3 * rng.normal(size=model["params"][key].shape)
            worst["P1"] = max(worst["P1"], _perm_violation(model, X, rng))
            worst["P2"] = max(worst["P2"], _rep_violation(model, X))
            worst["P1_neg"] = max(worst["P1_neg"], _perm_violation(dict(model, pool="first"), X, rng))
            worst["P2_neg"] = max(worst["P2_neg"], _rep_violation(dict(model, pool="sum"), X))
    gin_out = hidden_states(_new("interface", rng), data["held"][0])["stage_0"][..., SEED_CH]
    definition = bool(np.all(gin_out == 0.0))
    ok = worst["P1"] < 1e-9 and worst["P2"] < 1e-9 and worst["P1_neg"] > 1e-6 and worst["P2_neg"] > 1e-6
    detail = ("P1 max violation %.1e (negative control 'first-item' %.1e); P2 max violation %.1e (negative "
              "control 'sum-pool' %.1e); D1 [definition check, not evidence] seed channel zero after gin: %s") % (
        worst["P1"], worst["P1_neg"], worst["P2"], worst["P2_neg"], definition)
    return ok and definition, detail


def _row_hashes(X):
    return {hashlib.sha1(np.round(r, 8).tobytes()).hexdigest() for r in X.reshape(-1, X.shape[-1])}


def c7_splits(data):
    pairs = [("train", "val"), ("train", "held"), ("train", "shift"), ("b_train", "b_val"), ("b_train", "b_held")]
    overlaps = {a + "/" + b: len(_row_hashes(data[a][0]) & _row_hashes(data[b][0])) for a, b in pairs}
    return all(v == 0 for v in overlaps.values()), "shared tufts: " + ", ".join(
        "%s %d" % kv for kv in overlaps.items())


# ----------------------------------------------------------------------------- hypothesis runs
def run_seed(seed, steps):
    """Train every rival on one seed's villages; return the metrics the hypotheses read."""
    cs = np.random.SeedSequence(seed).spawn(3)
    data = make_data(cs[0])
    gi, gt = np.random.default_rng(cs[1]), np.random.default_rng(cs[2])
    out, models = {}, {}
    for kind in KINDS:
        model = _new(kind, gi)
        fit(model, data["train"] + data["val"], steps, gt)
        models[kind] = model
        out[kind + "_held"] = r2(predict(model, data["held"][0]), data["held"][1])
        out[kind + "_shift"] = r2(predict(model, data["shift"][0]), data["shift"][1])
    for kind in ("interface", "raw"):
        model = _new(kind, gi)
        fit(model, data["b_train"] + data["b_val"], steps, gt)
        Xb, Yb = data["b_held"]
        out[kind + "_blind"] = r2(predict(model, Xb), Yb, dims=[4])
        if kind == "interface":
            out["diag_seed_load"] = r2(predict(knockout(model, "measure", "seed_load"), Xb), Yb, dims=[4]) - \
                out["interface_blind"]
    for name, mode in KO_LIST:
        ko = knockout(models["interface"], name, mode)
        out["ko_" + name] = r2(predict(ko, data["held"][0]), data["held"][1]) - out["interface_held"]
    return out


HYP_DIFF = {"H-SIG": lambda r: r["interface_shift"] - r["raw_shift"],
            "H-NEC": lambda r: r["ko_measure"] - r["ko_finish"],
            "H-BLIND": lambda r: r["interface_blind"] - r["raw_blind"],
            "H-RIVAL": lambda r: r["interface_shift"] - r["adabn_shift"],
            "H-MADE": lambda r: r["interface_shift"] - r["fixed_shift"]}


def _summary(values, rng):
    """Mean and bootstrap interval; None where a quick or mutant run did not compute the quantity."""
    if not values:
        return None, None, None
    if len(values) == 1:
        return float(values[0]), None, None
    return paired_bootstrap(values, rng)


def evaluate(per_seed, rng, quick):
    hyps, kos = [], []
    for h in MIND_CARD["hypotheses"]:
        try:
            d = [HYP_DIFF[h["id"]](r) for r in per_seed]
        except KeyError:
            d = []
        mean, lo, hi = _summary(d, rng)
        v = "not evaluated" if quick or len(d) < h["seeds"] else verdict(mean, lo, hi, h["direction"], h["mesi"])
        hyps.append({"id": h["id"], "metric": h["metric"], "comparison": h["comparison"], "mean_diff": mean,
                     "ci95": [lo, hi], "mesi": h["mesi"], "direction": h["direction"], "n_seeds": len(d),
                     "verdict": v})
    sig = {n: s["signature"] for n, s in modules(_new("interface", rng)).items()}
    rows = [(n + "/" + m, "ko_" + n, sig[n]) for n, m in KO_LIST] + [
        ("measure/seed_load at gin (blind world, colour R2; diagnostic)", "diag_seed_load", True)]
    for label, key, s in rows:
        mean, lo, hi = _summary([r[key] for r in per_seed if key in r], rng)
        if mean is not None:
            kos.append({"module": label, "signature": s, "metric_change": mean, "ci95": [lo, hi]})
    return hyps, kos


# ----------------------------------------------------------------------------- real-data bridge (optional)
def load_lots_csv(path):
    """Minimal CSV reader: header with a 'lot' column, feature columns x*, target columns y*.
    Rows of each lot are cut into lots of N_ITEMS tufts (remainders dropped)."""
    with open(path, encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    head = [h.strip().lower() for h in lines[0].split(",")]
    li = head.index("lot")
    xi = [i for i, h in enumerate(head) if h.startswith("x")]
    yi = [i for i, h in enumerate(head) if h.startswith("y")]
    rows = np.array([[float(v) for v in ln.split(",")] for ln in lines[1:]])
    Xs, Ys = [], []
    for lot in np.unique(rows[:, li]):
        r = rows[rows[:, li] == lot]
        for j in range(len(r) // N_ITEMS):
            Xs.append(r[j * N_ITEMS:(j + 1) * N_ITEMS][:, xi])
            Ys.append(r[j * N_ITEMS:(j + 1) * N_ITEMS][:, yi])
    return np.array(Xs), np.array(Ys)


def data_bridge(path, rng, steps):
    X, Y = load_lots_csv(path)
    mu, sd = X.reshape(-1, X.shape[-1]).mean(0), X.reshape(-1, X.shape[-1]).std(0) + 1e-8
    X = (X - mu) / sd
    p = rng.permutation(len(X))
    a, b = int(.7 * len(X)), int(.85 * len(X))
    tr, va, te = p[:a], p[a:b], p[b:]
    res = {}
    for kind in ("interface", "raw"):
        model = build_model(X.shape[-1], Y.shape[-1], "vector_regression", rng, kind=kind)
        fit(model, (X[tr], Y[tr], X[va], Y[va]), steps, rng)
        res[kind] = r2(predict(model, X[te]), Y[te])
    return "lots %d, test R2 interface %.3f, raw-context %.3f (exploratory, no verdict)" % (
        len(X), res["interface"], res["raw"])


# ----------------------------------------------------------------------------- report and command line
def _fmt(x, ci=None):
    if ci is not None:
        return "[%+.3f, %+.3f]" % tuple(ci) if None not in ci else "[  n/a ,   n/a ]"
    return "%+.3f" % x if x is not None else "  n/a "


def print_report(rep, per_seed):
    print("=== VERIFIED REPORT · chapter 0346 ===")
    print("file %s · card_revision %d" % (rep["file"], rep["card_revision"]))
    print("environment: python %s · numpy %s" % (rep["environment"]["python"], rep["environment"]["numpy"]))
    print("seeds %s · updates per model %d · runtime %.1f s (budget %.0f s)" % (
        rep["seeds"], rep["steps"], rep["runtime_s"], rep["budget_s"]))
    print("parameters: " + " · ".join("%s %d" % kv for kv in rep["param_counts"].items()))
    g = rep["gradcheck"]
    print("gradient check: %d/%d tensors, max rel error %.2e, at init and after 60 updates -> %s" % (
        g["tensors_checked"], g["tensors_total"], g["max_rel_error"], "PASS" if g["passed"] else "FAIL"))
    print("correctness:")
    for c in rep["correctness"]:
        print("  %-3s %-24s %s  %s" % (c["id"], c["name"], "PASS" if c["passed"] else "FAIL", c["detail"]))
    m = rep["mutants"]
    print("mutation score: %d/%d = %.2f" % (m["detected"], m["total"], m["score"]))
    if per_seed:
        print("models, mean R2 over seeds (held-out / shifted villages / blind-world colour):")
        for k in KINDS:
            b = np.mean([r[k + "_blind"] for r in per_seed]) if k + "_blind" in per_seed[0] else float("nan")
            print("  %-10s %.3f / %.3f / %s" % (k, np.mean([r[k + "_held"] for r in per_seed]),
                                               np.mean([r[k + "_shift"] for r in per_seed]),
                                               "%.3f" % b if math.isfinite(b) else "  -  "))
    print("hypotheses (paired per-seed differences, 95% bootstrap CI, 2000 resamples):")
    for h in rep["hypotheses"]:
        print("  %-7s %-26s %s %s mesi %.2f (%s) -> %s" % (h["id"], h["comparison"][:26], _fmt(h["mean_diff"]),
              _fmt(None, h["ci95"]), h["mesi"], h["direction"], h["verdict"]))
    print("knockouts of the interface maker (held-out R2 change unless stated):")
    for k in rep["knockouts"]:
        print("  %-62s %s %s %s" % (k["module"], "sig" if k["signature"] else "   ", _fmt(k["metric_change"]),
                                    _fmt(None, k["ci95"])))
    print("task types: %s · exit code %d" % (", ".join(rep["task_types"]), rep["exit_code"]))
    print("=== END REPORT ===")


def run_protocol(args):
    t0 = time.time()
    quick = args.quick
    steps, budget = (STEPS_QUICK, BUDGET_QUICK) if quick else (STEPS_FULL, BUDGET_FULL)
    seeds = [args.seed] if quick else [args.seed + i for i in range(args.seeds)]
    print("seeds: %s%s" % (seeds, "  (mutant: %s)" % args.mutant if args.mutant else ""))
    thr = MIND_CARD["thresholds"]
    root = np.random.SeedSequence(args.seed).spawn(8)
    data = make_data(root[0])
    g = [np.random.default_rng(c) for c in root[1:]]
    corr, rows = [], {}
    gc = c1_gradcheck(data, g[0], args.mutant)
    corr.append(("C1", "gradient_check", gc["passed"], "max rel error %.2e; failures %s" % (
        gc["max_rel_error"], gc["failures"] or "none")))
    corr.append(("C2", "determinism_finiteness") + c2_determinism(data, args.seed))
    ok3, det3, model3, _, _ = c3_learning(data, steps, g[1], thr, args.mutant)
    corr.append(("C3", "learning", ok3, det3))
    corr.append(("C4", "shuffled_label_control") + c4_shuffled(data, steps, g[2], thr))
    ok5, n_det, det5 = c5_mutants(data, g[3], thr)
    corr.append(("C5", "mutant_detection", ok5, det5))
    corr.append(("C6", "property_tests") + c6_properties(data, g[4]))
    corr.append(("C7", "split_integrity") + c7_splits(data))
    per_seed = []
    if not args.mutant:
        for s in (seeds if not quick else []):
            per_seed.append(run_seed(s, steps))
            print("  seed %d done (%.1f s)" % (s, time.time() - t0))
    if not per_seed:   # quick or mutant run: knockouts from the C3 model only, hypotheses not evaluated
        r = {"interface_held": r2(predict(model3, data["held"][0]), data["held"][1])}
        r.update({"ko_" + n: r2(predict(knockout(model3, n, m), data["held"][0]), data["held"][1]) -
                  r["interface_held"] for n, m in KO_LIST})
        hyps, kos = evaluate([r], g[5], True)
    else:
        hyps, kos = evaluate(per_seed, g[5], False)
    runtime = time.time() - t0
    corr.append(("C8", "budget", runtime <= budget, "%.1f s of %.0f s" % (runtime, budget)))
    failed = [c for c in corr if not c[2]]
    code = 3 if [c for c in failed if c[0] == "C8"] and len(failed) == 1 else (1 if failed else 0)
    rep = {"schema_version": "1.0", "chapter": 346, "file": os.path.basename(__file__),
           "card_revision": MIND_CARD["card_revision"],
           "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
           "seeds": seeds, "runtime_s": round(runtime, 2), "budget_s": budget, "steps": steps,
           "n_params": n_params(_new("interface", g[6])),
           "param_counts": {k: n_params(_new(k, g[6])) for k in KINDS}, "gradcheck": {
               k: gc[k] for k in ("tensors_checked", "tensors_total", "max_rel_error", "checked_at", "passed")},
           "correctness": [{"id": i, "name": n, "passed": bool(p), "detail": d} for i, n, p, d in corr],
           "mutants": {"detected": n_det, "total": len(MUTANTS), "score": n_det / len(MUTANTS)},
           "hypotheses": hyps, "knockouts": kos, "task_types": TASK_TYPES, "exit_code": code,
           "per_seed": per_seed}
    print_report(rep, per_seed)
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description="Chapter 0346 · Huang Daopo · ZUO ZAO tool re-making chain")
    ap.add_argument("--quick", action="store_true", help="one seed, reduced updates, all correctness tests")
    ap.add_argument("--seed", type=int, default=0, help="base seed")
    ap.add_argument("--seeds", type=int, default=5, help="number of seeds for the hypothesis tests")
    ap.add_argument("--json", metavar="PATH", help="also write the JSON report")
    ap.add_argument("--card", action="store_true", help="print MIND_CARD as JSON and exit")
    ap.add_argument("--mutant", choices=sorted(MUTANTS), help="run the correctness tests with a mutant")
    ap.add_argument("--data", metavar="PATH", help="CSV with lot,x*,y* columns (see load_lots_csv)")
    args = ap.parse_args(argv)
    if args.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    if args.seeds < 1 or (args.data and not os.path.isfile(args.data)):
        print("invalid usage: --seeds must be >= 1 and --data must name an existing file", file=sys.stderr)
        return 2
    try:
        rep = run_protocol(args)
        if args.data:
            print("real-data bridge: " + data_bridge(args.data, np.random.default_rng(args.seed), STEPS_QUICK))
        else:
            print("real-data bridge: no --data given; skipped")
    except NonFinite as exc:
        print("non-finite values: %s" % exc, file=sys.stderr)
        return 4
    if args.json:
        write_report_json(args.json, rep)
    return rep["exit_code"]


if __name__ == "__main__":
    sys.exit(main())

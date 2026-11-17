#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0347 · Zhu Shijie (1249-1314)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 18, Minds 341-360 Available on Amazon https://www.amazon.com/dp/B0HQ47T2NC
# Resume and Interactive Demos at https://artificiology.com/
# END ATTRIBUTION 
"""
Facing Boards (左右對消): in-context inference by eliminating hidden task unknowns.

Thesis
    An unknown is a direction on a shared board, and understanding is making the
    unwanted directions vanish: two relations that share a hidden unknown are set
    facing each other and the product of their inner columns is taken from the
    product of their outer columns, until only what is asked remains.

Evidence and provenance (belief: his own surviving works ground the reading;
    no statement about the mind survives, so the reading is drawn from method)
    D1  Board: constant (太) at the centre, heaven below, earth left, man right,
        matter above (Mo Ruo, preface to the Siyuan yujian, 1303); multiplying by
        an unknown moves the whole array (Wagner 2025, on the array notation).
    D2  Two statements per problem (今有 / 只云) become two arrays; left and right
        arrays face each other; inner-column product minus outer-column product
        eliminates an unknown (Siyuan yujian with Luo Shilin's xicao; Wagner 2025).
    D3  Successive elimination 4 -> 3 -> 2 -> 1 unknowns; final equations up to
        degree 14 (Ho Peng Yoke, DSB 1971; Hoe 1977).
    D4  His final equations are sometimes of lower degree than the resultants,
        i.e. shared factors were removed, though not systematically; four unknowns
        were set where three sufficed (Ueno 2022).
    D5  The two-dimensional board limited the method to four unknowns and to
        products of at most two (Ho Peng Yoke, DSB).
    D6  Elimination manufactures roots that answer nothing (problem 6.15: roots 1
        and 9, only 9 is a solution); only the fitting root is reported
        (Wagner 2025; MacTutor on the quartic t^4-34t^3+71t^2+3706t+3600=0).
    D7  Freedom in choosing what to call the unknown (e.g. y = sqrt(a+b)) so the
        relations become polynomial (Wagner 2025).
    D8  齊 (homogenize) and 消 (clear) are the named operations on arrays, taken
        from the fangcheng chapter of the Nine Chapters (Wagner 2025); fan fa
        refines a first root by transforming the equation (Ho, DSB).

Doctrine -> mechanism -> test
    D1 -> M1 board cells per observation + board_mul (row shifts)  -> C6.3
    D2 -> M2 facing eliminator, stage 1 (inner - outer)            -> C6.1, H-SIG
    D3 -> M3 facing of the final rows, stage 2 (common root)       -> H-NEC
    D4 -> M3 common root read without forming the full eliminant   -> H-LIYE
    D5 -> task: third-origin world the two-origin board cannot hold -> H-BLIND
    D6 -> rival: nearest-root extraction from one final equation   -> H-LIYE
    D7 -> learned encoder chooses the gauge of its own unknowns    -> C6.1
    D8 -> statements and final rows normalized before clearing      -> C6.1
          fan fa from the prior mean as a rival inner loop         -> H-PRIOR

Research question
    Can an in-context learner that removes bilinear task latents by pairwise
    determinantal cancellation, instead of encoding them (neural processes) or
    linearizing them (closed-form ridge heads), transfer to tasks whose hidden
    values lie far outside the training range?

Closest prior art and the delta
    R2-D2 closed-form ridge head (Bertinetto et al. 2019) and conditional neural
    processes (Garnelo et al. 2018); inner-loop latent inference by optimization
    from the prior (CAVIA, Zintgraf et al. 2019); bilinear two-factor models
    (Tenenbaum and Freeman 2000); Bezout/Sylvester resultants. Delta: latents are neither
    estimated by a learned encoder nor linearized into free monomials; a learned
    per-point board is faced pairwise (stage 1) and the resulting final rows are
    faced again (stage 2), so the product t*d is used as a product and the
    prediction is invariant to how large the hidden values are.

Blind spot
    A world with more hidden origins than the board has directions (D5). The
    board here holds two origins (天 tian, 地 di) and their quadrant product; the
    blind world adds a third origin (人 ren) that no cell can hold.

Task (episodic in-context regression, vector_regression interface)
    law f_0..f_4: fixed hidden random tanh maps of x[:3]; x[3:5] are distractors
    per episode: t, d ~ U(-1,1)   (shifted split: |t|,|d| ~ U(1.6, 2.6))
    y = f0(x) + t f1(x) + d f2(x) + t d f3(x) + 0.05 eps
    blind world adds r f4(x) with a third latent r ~ U(-1,1)
    K = 4 context pairs and 4 query points per episode; predict query y

Limits
    Two origins, not four: with two origins the three monomials (t, d, t*d) never
    outnumber the four statements, which is the regime where a linear closed form
    is already efficient; Zhu's four-origin regime (eight quadrant products) is
    not reached here. Exact elimination has no noise model beyond pooling across
    facings and the board edge (|latent| <= 6). A research prototype of one
    mechanism, not an AGI, and not a replica of a person.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 2,
    "revision_log": [
        ("rev 2, 2026-09-29, before the first five-seed run. Pilots (oracle encoder with the true law; one "
         "quick run; single-seed budget probes) showed: (a) raw elimination from three noisy statements is "
         "heavy-tailed even with the true law, so K_CTX 3 -> 4 and 齊 homogenization of statements and final "
         "rows were added; (b) a finite board edge (soft clamp beyond |latent| 6) and a Huber training loss "
         "were added for all models for numerical stability; (c) a shared optimizer setting (Adam 3e-3, "
         "batch 128, 800 updates) was fixed for all models; (d) fan fa refinement after elimination "
         "destabilized training and was removed from the model; it survives only as the prior-start rival, "
         "so H-PRIOR was added; (e) the metric was fixed as R2 floored at -1. The pilots also indicated that "
         "the linearized ridge head and the prior-start rival match or beat learned elimination; H-SIG, "
         "H-NEC, H-BLIND, H-RIVAL and H-LIYE were kept exactly as written in revision 1.")],
    "generation": {"template_version": "codeguidelines 1.0 (15 Sep 2026)", "generator": "Claude",
                   "generator_version": "Opus 5.5", "date": "2026-09-29"},
    "id": 347, "figure": "Zhu Shijie", "born": 1249, "died": 1314, "civilization": "Chinese (Yuan)",
    "provenance": "belief",
    "thesis": ("An unknown is a direction on a shared board, and understanding is making the unwanted "
               "directions vanish by facing relations that share them, inner product against outer."),
    "evidence": [
        {"id": "D1", "claim": "Constant at the centre, heaven below, earth left, man right, matter above; "
         "multiplying by an unknown moves the array", "basis": "primary",
         "source": "Mo Ruo, preface to Siyuan yujian (1303); Wagner 2025, array notation"},
        {"id": "D2", "claim": "Left and right arrays face each other; inner-column product minus outer-column "
         "product eliminates an unknown", "basis": "primary",
         "source": "Siyuan yujian xia 6.15 with Luo Shilin, Siyuan yujian xicao (1834); Wagner 2025, App. 5"},
        {"id": "D3", "claim": "Successive elimination to one unknown; final equations up to degree 14",
         "basis": "scholarship", "source": "Ho Peng Yoke, DSB vol. 3 (1971); Hoe 1977"},
        {"id": "D4", "claim": "Final equations sometimes lower-degree than the resultants; not systematic; "
         "four unknowns set where three sufficed", "basis": "scholarship",
         "source": "Ueno, RIMS Kokyuroku Bessatsu B89 (2022) 1-22"},
        {"id": "D5", "claim": "Two-dimensional board limits the method to four unknowns and products of two",
         "basis": "scholarship", "source": "Ho Peng Yoke, DSB vol. 3 (1971)"},
        {"id": "D6", "claim": "Elimination yields roots that answer nothing; only the fitting root reported",
         "basis": "scholarship", "source": "Wagner 2025 on 6.15; O'Connor and Robertson, MacTutor (2003)"},
        {"id": "D7", "claim": "Freedom to choose what the unknown stands for so relations become polynomial",
         "basis": "scholarship", "source": "Wagner 2025, concluding remarks"},
        {"id": "D8", "claim": "Arrays are homogenized (齊) and cleared (消); a first root is refined by a "
         "transformation (fan fa)", "basis": "scholarship",
         "source": "Wagner 2025 (fangcheng terms); Ho Peng Yoke, DSB vol. 3 (fan fa)"},
    ],
    "research_question": {"category": "sample efficiency and transfer",
                          "question": ("Does removing bilinear task latents by pairwise determinantal "
                                       "cancellation transfer to hidden values outside the training "
                                       "range, where amortized encoders and linearized heads do not?")},
    "mechanism": {
        "name": "Facing Boards (two-stage inner-minus-outer elimination over learned per-point boards)",
        "family": "meta-learning / differentiable algebraic elimination",
        "signature_modules": ["facing_eliminator", "facing_of_facings"],
        "closest_prior_art": ["R2-D2 closed-form ridge head (Bertinetto et al. 2019, arXiv:1805.08136)",
                              "Conditional Neural Processes (Garnelo et al. 2018, arXiv:1807.01613)",
                              "CAVIA context adaptation (Zintgraf et al. 2019, arXiv:1810.03642)",
                              "Bilinear style-content models (Tenenbaum and Freeman 2000)",
                              "Bezout/Sylvester resultants; Wu characteristic sets (Wu and Gao 2007)"],
        "overlap": "Medium",
        "prior_art_queries": ["differentiable resultant elimination layer neural network",
                              "closed-form solver meta-learning ridge head",
                              "neural process extrapolation latent magnitude",
                              "learning to compute Groebner bases transformer",
                              "bilinear task latent in-context regression"],
        "contribution_type": "mechanism",
        "delta": ("Latents are cancelled, not encoded or linearized: learned boards are faced pairwise and "
                  "the final rows are faced again, so t*d is used as a product and predictions are "
                  "invariant to the size of the hidden values."),
    },
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1 board_encoder + board_mul", "property_test": "C6.3",
         "hypothesis": "H-SIG"},
        {"doctrine": "D2", "mechanism": "M2 facing_eliminator", "property_test": "C6.1", "hypothesis": "H-SIG"},
        {"doctrine": "D3", "mechanism": "M3 facing_of_facings", "property_test": "C6.2", "hypothesis": "H-NEC"},
        {"doctrine": "D4", "mechanism": "M3 common root without full eliminant", "property_test": "C6.1",
         "hypothesis": "H-LIYE"},
        {"doctrine": "D5", "mechanism": "two-origin board", "property_test": "C6.1", "hypothesis": "H-BLIND"},
        {"doctrine": "D6", "mechanism": "rival nearest-root extraction", "property_test": "C6.1",
         "hypothesis": "H-LIYE"},
        {"doctrine": "D7", "mechanism": "learned gauge of the unknowns", "property_test": "C6.1",
         "hypothesis": "H-RIVAL"},
        {"doctrine": "D8", "mechanism": "homogenized statements and rows; fan fa rival", "property_test": "C6.1",
         "hypothesis": "H-PRIOR"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "Facing Boards beats a size-matched conditional neural process when hidden "
         "values are 1.6-2.6x beyond the training range", "metric": "R2 (floored at -1)", "split": "shifted",
         "comparison": "model - cnp", "direction": "greater", "mesi": 0.10, "seeds": 5},
        {"id": "H-NEC", "statement": "Mean-ablating the eliminator output hurts shifted R2 more than zeroing the "
         "distractor-input weights", "metric": "R2 (floored at -1)", "split": "shifted",
         "comparison": "signature_knockout - matched_knockout", "direction": "less", "mesi": 0.20, "seeds": 5},
        {"id": "H-BLIND", "statement": "In a third-origin world the model's advantage over the CNP shrinks",
         "condition": "blind world (extra latent r f4(x)), models trained and tested there",
         "grounding": "the two-dimensional board capped the method at four unknowns (Ho, DSB)",
         "metric": "R2 (floored at -1)", "split": "heldout",
         "comparison": "(model - cnp)_blind - (model - cnp)_heldout", "direction": "less", "mesi": 0.05,
         "seeds": 5},
        {"id": "H-RIVAL", "statement": "Facing Boards beats the closest prior art, a linearized closed-form ridge "
         "head on the same encoder", "metric": "R2 (floored at -1)", "split": "shifted", "comparison": "model - ridge",
         "direction": "greater", "mesi": 0.02, "seeds": 5},
        {"id": "H-LIYE", "statement": "Facing the final rows beats extracting the nearest root of each final "
         "equation from a first estimate (dialectic with chapter 0324)", "metric": "R2 (floored at -1)", "split": "shifted",
         "comparison": "model - liye", "direction": "greater", "mesi": 0.05, "seeds": 5},
        {"id": "H-PRIOR", "statement": "Elimination beats inner-loop inference that refines from the prior mean "
         "(fan fa from t = d = 0, CAVIA-like) when hidden values leave the training range",
         "metric": "R2 (floored at -1)", "split": "shifted", "comparison": "model - prior",
         "direction": "greater", "mesi": 0.05, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.5, "margin_over_trivial": 0.3, "gradcheck_rel_tol": 1e-5,
                   "shuffled_band": 0.1},
    "probe_predictions": [
        {"probe": "P1", "expected": "below baseline"}, {"probe": "P2", "expected": "equal to baseline"},
        {"probe": "P6", "expected": "above baseline"}, {"probe": "P7", "expected": "below baseline"},
        {"probe": "P8", "expected": "below baseline"}, {"probe": "P9", "expected": "below baseline"},
        {"probe": "P10", "expected": "above baseline"}],
    "dialectic_links": [{"chapter": 324, "relation": "rival",
                         "test": "H-LIYE (single-equation root extraction from a first estimate)"}],
    "corpus_neighbors": [
        {"chapter": 324, "similarity": None, "difference": "Li Ye extracts the root of one equation over one "
         "figure; here unknowns are cancelled between relations and never valued before the end"},
        {"chapter": 329, "similarity": None, "difference": "Qin reassembles a whole from residues; here "
         "nothing is reassembled, the unwanted is cancelled"},
        {"chapter": 339, "similarity": None, "difference": "Guo re-anchors measured rates at the last "
         "observation; here no time and no measurement, only relation"},
        {"chapter": 195, "similarity": None, "difference": "Yi Xing models change in change across unequal "
         "intervals; here no sequence at all"},
        {"chapter": 340, "similarity": None, "difference": "Llull counts pairwise chambers as votes; here "
         "pairs are cancelled against each other"},
        {"chapter": 346, "similarity": None, "difference": "Huang Daopo regenerates tools from intake "
         "moments; here the adaptation is an algebraic solve, not a hypernetwork"}],
    "barometer": {
        "cognitive_processing": ["P2", "P6", "P10"], "embodied_cognition": [],
        "world_modeling": ["P7"], "consciousness": ["P8"], "language_understanding": [],
        "emotional_intelligence": [], "creativity": [], "autonomy": ["P5"]},
    "task_types": ["vector_regression"],
    "applications": [
        {"use": "Few-shot recalibration when two drift factors interact multiplicatively in a sensor array",
         "sector": "environmental and industrial sensing",
         "dataset": "UCI Gas Sensor Array Drift Dataset (Vergara et al. 2012)"},
        {"use": "Cold-start rating prediction from a handful of ratings with two interacting taste factors",
         "sector": "recommendation", "dataset": "MovieLens 100K (GroupLens)"},
        {"use": "Per-cell ageing model where capacity fade and resistance growth interact, fitted from a few "
         "cycles", "sector": "energy storage", "dataset": "NASA PCoE Li-ion Battery Aging Datasets"}],
    "safety_notes": ("Mathematics only; no personal data; synthetic tasks. No claim of replicating a mind "
                     "and no generated sentences attributed to the figure."),
}

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

P_IN = 5                 # input features; x[3:5] are distractors the law never reads
DISTRACTORS = (3, 4)
K_CTX = 4                # context statements per episode (今有, 只云 and two more)
N_QRY = 4                # query points per episode
NOISE = 0.05
HIDDEN = 32              # board encoder width
CLIP = 5.0
LR = 3e-3
BATCH = 128
EPS_DIV = 1e-9
BOARD_EDGE = 6.0         # a counting board is finite: |latent| beyond 6 bends softly toward 8
HUBER = 1.0
FANFA_PRIOR = 5          # fan fa correction steps for the prior-start rival
N_TRAIN, N_VAL, N_TEST = 2048, 256, 512
BUDGET = {"default": 800, "quick": 150}
TIME_LIMIT = {"default": 180.0, "quick": 20.0}


# BEGIN STANDARD UTILITIES v1.0
def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def logsumexp(z, axis=-1):
    m = z.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(z - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def softplus(x):
    return np.logaddexp(0.0, x)


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
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
    scale = min(1.0, max_norm / (norm + 1e-12))
    return {k: g * scale for k, g in grads.items()}, norm


def finite_difference_check(loss_fn, params, grads, rng, eps=1e-6, n_entries=20, floor=1e-4):
    """Central differences on random entries plus the largest-gradient entry of every tensor.
    rel = |analytic - numeric| / max(|analytic| + |numeric|, floor); floor absorbs round-off
    on near-zero entries (float64, eps 1e-6)."""
    worst, checked = 0.0, 0
    for k, p in params.items():
        flat = p.reshape(-1)
        g = grads[k].reshape(-1)
        idx = set(rng.choice(flat.size, size=min(n_entries, flat.size), replace=False).tolist())
        idx.add(int(np.argmax(np.abs(g))))
        for i in idx:
            old = flat[i]
            flat[i] = old + eps
            lp = loss_fn()
            flat[i] = old - eps
            lm = loss_fn()
            flat[i] = old
            num = (lp - lm) / (2 * eps)
            worst = max(worst, abs(g[i] - num) / max(abs(g[i]) + abs(num), floor))
        checked += 1
    return worst, checked


def paired_bootstrap(diffs, rng, n_boot=2000, alpha=0.05):
    diffs = np.asarray(diffs, dtype=np.float64)
    means = np.array([rng.choice(diffs, size=diffs.size, replace=True).mean() for _ in range(n_boot)])
    return float(diffs.mean()), [float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))]


def verdict(mean, ci, mesi, direction):
    sgn = 1.0 if direction == "greater" else -1.0
    if sgn * ci[0] > 0 and sgn * ci[1] > 0 and sgn * mean >= mesi:
        return "supported"
    if sgn * ci[0] < 0 and sgn * ci[1] < 0:
        return "contradicted"
    return "inconclusive"


def write_report(obj, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
# END STANDARD UTILITIES


# ---------------------------------------------------------------- reverse-mode tape
class V:
    """A node on the tape: value, accumulated gradient, parents and a backward rule."""
    __slots__ = ("val", "grad", "par", "bw")
    __array_priority__ = 1000

    def __init__(self, val, par=(), bw=None):
        self.val = np.asarray(val, dtype=np.float64)
        self.grad, self.par, self.bw = None, par, bw

    def __add__(self, o):
        return _bin(self, o, lambda a, b: a + b, lambda g, a, b: g, lambda g, a, b: g)

    __radd__ = __add__

    def __sub__(self, o):
        return _bin(self, o, lambda a, b: a - b, lambda g, a, b: g, lambda g, a, b: -g)

    def __rsub__(self, o):
        return _bin(o, self, lambda a, b: a - b, lambda g, a, b: g, lambda g, a, b: -g)

    def __mul__(self, o):
        return _bin(self, o, lambda a, b: a * b, lambda g, a, b: g * b, lambda g, a, b: g * a)

    __rmul__ = __mul__

    def __truediv__(self, o):
        return _bin(self, o, lambda a, b: a / b, lambda g, a, b: g / b, lambda g, a, b: -g * a / (b * b))

    def __neg__(self):
        return V(-self.val, (self,), lambda g: (-g,))

    def __getitem__(self, idx):
        def bw(g):
            out = np.zeros_like(self.val)
            out[idx] += g
            return (out,)
        return V(self.val[idx], (self,), bw)


def _as(x):
    return x if isinstance(x, V) else V(x)


def _unb(g, shape):
    while g.ndim > len(shape):
        g = g.sum(0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(i, keepdims=True)
    return g


def _bin(a, b, f, ga, gb):
    a, b = _as(a), _as(b)
    return V(f(a.val, b.val), (a, b),
             lambda g: (_unb(ga(g, a.val, b.val), a.val.shape), _unb(gb(g, a.val, b.val), b.val.shape)))


def tanh(a):
    y = np.tanh(a.val)
    return V(y, (a,), lambda g: (g * (1 - y * y),))


def vsqrt(a):
    y = np.sqrt(a.val)
    return V(y, (a,), lambda g: (0.5 * g / y,))


def vsoftplus(a):
    s = 1.0 / (1.0 + np.exp(-a.val))
    return V(softplus(a.val), (a,), lambda g: (g * s,))


def vsum(a, axis=None):
    def bw(g):
        g2 = g if axis is None else np.expand_dims(g, axis)
        return (np.broadcast_to(g2, a.val.shape).copy(),)
    return V(a.val.sum(axis=axis), (a,), bw)


def reshape(a, shape):
    return V(a.val.reshape(shape), (a,), lambda g: (g.reshape(a.val.shape),))


def broadcast(a, shape):
    return V(np.broadcast_to(a.val, shape).copy(), (a,), lambda g: (_unb(g, a.val.shape),))


def matmul(a, b):
    return V(a.val @ b.val, (a, b), lambda g: (g @ b.val.T, a.val.T @ g))


def concat(xs, axis=-1):
    cuts = np.cumsum([x.val.shape[axis] for x in xs])[:-1]
    return V(np.concatenate([x.val for x in xs], axis=axis), tuple(xs),
             lambda g: tuple(np.split(g, cuts, axis=axis)))


def vwhere(mask, a, b):
    a, b = _as(a), _as(b)
    return V(np.where(mask, a.val, b.val), (a, b),
             lambda g: (_unb(np.where(mask, g, 0.0), a.val.shape), _unb(np.where(mask, 0.0, g), b.val.shape)))


def solve(a, b):
    """Batched linear solve a x = b with a (B,n,n), b (B,n); adjoint rule for the backward pass."""
    x = np.linalg.solve(a.val, b.val[..., None])[..., 0]

    def bw(g):
        gb = np.linalg.solve(np.swapaxes(a.val, -1, -2), g[..., None])[..., 0]
        return (-gb[..., :, None] * x[..., None, :], gb)
    return V(x, (a, b), bw)


def backward(out):
    order, seen, stack = [], set(), [(out, False)]
    while stack:
        v, done = stack.pop()
        if done:
            order.append(v)
            continue
        if id(v) in seen:
            continue
        seen.add(id(v))
        stack.append((v, True))
        stack.extend((p, False) for p in v.par if id(p) not in seen)
    out.grad = np.ones_like(out.val)
    for v in reversed(order):
        if v.bw is None or v.grad is None:
            continue
        for p, gp in zip(v.par, v.bw(v.grad)):
            p.grad = gp if p.grad is None else p.grad + gp


# ---------------------------------------------------------------- data and tasks
class NonFinite(Exception):
    pass


def make_law(rng):
    """Hidden law: five fixed random tanh maps of x[:3] (f0..f3 used, f4 only in the blind world)."""
    law = {"U": rng.normal(0, 1, (5, 6, 3)), "C": rng.normal(0, 0.5, (5, 6)),
           "W": rng.normal(0, 1, (5, 6)) / math.sqrt(6)}
    ref = rng.normal(0, 1, (4000, P_IN))
    raw = law_raw(law, ref)
    law["mu"], law["sd"] = raw.mean(0), raw.std(0) + 1e-9
    return law


def law_raw(law, X):
    z = np.tanh(np.einsum("...p,kjp->...kj", X[..., :3], law["U"]) + law["C"])
    return (z * law["W"]).sum(-1)


def law_f(law, X):
    return (law_raw(law, X) - law["mu"]) / law["sd"]


def sample_split(rng, law, n_ep, shifted=False, blind=False):
    X = rng.normal(0, 1, (n_ep, K_CTX + N_QRY, P_IN))
    if shifted:
        lat = rng.choice([-1.0, 1.0], (n_ep, 2)) * rng.uniform(1.6, 2.6, (n_ep, 2))
    else:
        lat = rng.uniform(-1, 1, (n_ep, 2))
    t, d = lat[:, :1], lat[:, 1:]
    F = law_f(law, X)
    Y = F[..., 0] + t * F[..., 1] + d * F[..., 2] + t * d * F[..., 3]
    r = rng.uniform(-1, 1, (n_ep, 1)) if blind else np.zeros((n_ep, 1))
    Y = Y + r * F[..., 4] + NOISE * rng.normal(0, 1, Y.shape)
    return {"Xc": X[:, :K_CTX], "Yc": Y[:, :K_CTX], "Xq": X[:, K_CTX:], "Yq": Y[:, K_CTX:],
            "lat": np.concatenate([lat, r], 1)}


def take(data, idx):
    return {k: v[idx] for k, v in data.items()}


def episode_hashes(data):
    return {hashlib.sha1(np.round(np.concatenate([data["Xc"][i].ravel(), data["lat"][i]]), 10).tobytes())
            .hexdigest() for i in range(len(data["Xc"]))}


def make_world(seed_seq, law, blind=False):
    rs = [np.random.default_rng(s) for s in seed_seq.spawn(4)]
    return {"train": sample_split(rs[0], law, N_TRAIN, blind=blind), "val": sample_split(rs[1], law, N_VAL, blind=blind),
            "heldout": sample_split(rs[2], law, N_TEST, blind=blind),
            "shifted": sample_split(rs[3], law, N_TEST, shifted=True, blind=blind)}


# ---------------------------------------------------------------- model: Facing Boards
def init_mlp(rng, sizes, prefix):
    params = {}
    for i, (m, n) in enumerate(zip(sizes[:-1], sizes[1:])):
        params[f"{prefix}W{i}"] = rng.normal(0, 1 / math.sqrt(m), (m, n))
        params[f"{prefix}b{i}"] = np.zeros(n)
    return params


def mlp(P, prefix, h, n_layers):
    for i in range(n_layers):
        h = matmul(h, P[f"{prefix}W{i}"]) + P[f"{prefix}b{i}"]
        if i < n_layers - 1:
            h = tanh(h)
    return h


def board_cells(P, X):
    """Each observation becomes one board: cells for 1, t (heaven), d (earth) and the quadrant t*d."""
    B, N, _ = X.shape
    return reshape(mlp(P, "enc_", V(X.reshape(B * N, -1)), 2), (B, N, 4))


def board_mul(p, q):
    """Product of two rows of the board (polynomials in d). Multiplying by d moves a whole row one
    place along its axis (D1), so the product is a sum of shifted copies."""
    out = [None] * (len(p) + len(q) - 1)
    for i, pi in enumerate(p):
        for j, qj in enumerate(q):
            term = pi * qj
            out[i + j] = term if out[i + j] is None else out[i + j] + term
    return out


def homogenize(parts):
    """齊: scale a statement (or a final row) to unit norm; a scaled relation has the same roots."""
    norm = vsqrt(sum(p * p for p in parts[1:]) + parts[0] * parts[0] + 1e-30)
    return [p / norm for p in parts]


def board_edge(x):
    """Identity inside the board; beyond |x| = 6 the rod bends softly toward 8 (C1-smooth)."""
    sgn = np.where(x.val >= 0, 1.0, -1.0)
    over = sgn * (BOARD_EDGE + 2.0 * tanh((x * sgn - BOARD_EDGE) * 0.5))
    return vwhere(np.abs(x.val) <= BOARD_EDGE, x, over)


def face(left, right, swap=False):
    """Stage 1 (D2): relation_l = A_l(d) + B_l(d) t, laid out as (B_l | A_l). Left and right arrays face
    each other; the outer-column product minus the inner-column product removes t."""
    outer = board_mul(left[0], right[1])
    inner = board_mul(left[1], right[0])
    return [o + i for o, i in zip(outer, inner)] if swap else [o - i for o, i in zip(outer, inner)]


def face_rows(u, v):
    """Stage 2 (D4): two final rows u, v in d share the root; each component of their cross product is
    again an inner-minus-outer product, and (1, d, d^2) is proportional to it."""
    n1 = u[1] * v[2] - u[2] * v[1]
    nd = u[2] * v[0] - u[0] * v[2]
    nd2 = u[0] * v[1] - u[1] * v[0]
    return n1, nd, nd2


def facing_statements(cells, Yc, swap=False):
    """Hide d inside the coefficients of t (互隱) and face every pair of context statements."""
    K = Yc.val.shape[1]
    a = cells[..., 0] - Yc
    rel = []
    for l in range(K):
        al, bl, cl, el = homogenize([a[:, l], cells[:, l, 1], cells[:, l, 2], cells[:, l, 3]])
        rel.append(([bl, el], [al, cl]))
    rows = [homogenize(face(rel[i], rel[j], swap)) for i in range(K) for j in range(i + 1, K)]
    return rel, rows


def common_root(rows, pool_mean=False):
    """Energy-weighted least squares over all facings of final rows: n1 d = nd and nd d = nd2."""
    num, den, ratios = None, None, []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            n1, nd, nd2 = face_rows(rows[i], rows[j])
            nu, de = n1 * nd + nd * nd2, n1 * n1 + nd * nd
            ratios.append(nu / (de + EPS_DIV))
            num = nu if num is None else num + nu
            den = de if den is None else den + de
    if pool_mean:
        total = ratios[0]
        for r in ratios[1:]:
            total = total + r
        return total * (1.0 / len(ratios))
    return num / (den + EPS_DIV)


def back_substitute(rel, d, anchor_first=False):
    """Recover t from every statement at the common d (least squares); anchor_first uses only the
    first statement, as a pivot would."""
    use = rel[:1] if anchor_first else rel
    num, den = None, None
    for (tb, tf) in use:
        A = tf[0] + tf[1] * d
        Bc = tb[0] + tb[1] * d
        num = A * Bc if num is None else num + A * Bc
        den = Bc * Bc if den is None else den + Bc * Bc
    return -num / (den + EPS_DIV)


def nearest_root(row):
    """Rival (ch. 0324): extract the root of one final equation nearest the first estimate 0,
    using the stable form -2 q0 / (q1 + sgn(q1) sqrt(disc)); disc is smoothly floored at 0."""
    q0, q1, q2 = row
    disc = q1 * q1 - 4.0 * q0 * q2
    soft = 0.5 * (disc + vsqrt(disc * disc + 1e-6))
    sgn = np.where(q1.val >= 0, 1.0, -1.0)
    return -2.0 * q0 / (q1 + sgn * vsqrt(soft + 1e-18) + sgn * EPS_DIV)


def fan_fa(cells, Yc, t, d, iters, lm=1e-2):
    """Rival (CAVIA-like inner loop) built from 翻法 (fan fa): transform around the current root and read a
    correction (DSB: Zhu refined a first root this way), here started from the prior mean t = d = 0 instead
    of from an elimination. Damped Gauss-Newton on the raw residuals a + b t + c d + e t d."""
    a, b, c, e = cells[..., 0] - Yc, cells[..., 1], cells[..., 2], cells[..., 3]
    B, K = Yc.val.shape
    t, d = reshape(t, (B, 1)), reshape(d, (B, 1))
    for _ in range(iters):
        r = a + b * t + c * d + e * t * d
        j0, j1 = b + e * d, c + e * t
        h00, h11, h01 = vsum(j0 * j0, 1), vsum(j1 * j1, 1), vsum(j0 * j1, 1)
        g0, g1 = vsum(j0 * r, 1), vsum(j1 * r, 1)
        mu = (h00 + h11) * (0.5 * lm) + 1e-12
        h00, h11 = h00 + mu, h11 + mu
        det = h00 * h11 - h01 * h01
        t = t - reshape((h11 * g0 - h01 * g1) / det, (B, 1))
        d = d - reshape((h00 * g1 - h01 * g0) / det, (B, 1))
    return reshape(t, (B,)), reshape(d, (B,))


def solve_latents(model, cells, Yc):
    flags = model["flags"]
    B, K = Yc.val.shape
    if K < 3:
        z = V(np.zeros(B))
        return z, z
    if model["kind"] == "prior":
        t, d = fan_fa(cells, Yc, V(np.zeros(B)), V(np.zeros(B)), FANFA_PRIOR)
        return board_edge(t), board_edge(d)
    rel, rows = facing_statements(cells, Yc, swap=flags.get("mutant") == "inner_outer_swap")
    if model["kind"] == "liye":
        roots = [nearest_root(r) for r in rows]
        d = roots[0]
        for r in roots[1:]:
            d = d + r
        d = d * (1.0 / len(roots))
    else:
        d = common_root(rows, pool_mean=flags.get("ko_pool", False))
    d = board_edge(d)
    t = board_edge(back_substitute(rel, d, anchor_first=flags.get("anchor_first", False)))
    if flags.get("ko_elim") is not None:
        tm, dm = flags["ko_elim"]
        t, d = V(np.full(B, tm)), V(np.full(B, dm))
    return t, d


def forward_boards(model, P, batch):
    cells_c = board_cells(P, batch["Xc"])
    cells_q = board_cells(P, batch["Xq"])
    t, d = solve_latents(model, cells_c, V(batch["Yc"]))
    B = batch["Xq"].shape[0]
    t2, d2 = reshape(t, (B, 1)), reshape(d, (B, 1))
    return cells_q[..., 0] + cells_q[..., 1] * t2 + cells_q[..., 2] * d2 + cells_q[..., 3] * t2 * d2


# ---------------------------------------------------------------- baselines and rival mechanisms
def forward_ridge(model, P, batch):
    """Closest prior art (R2-D2): same encoder, but t, d and t*d are three free unknowns solved by
    ridge regression on the context; lambda = softplus(rho) is learned."""
    cells_c, cells_q = board_cells(P, batch["Xc"]), board_cells(P, batch["Xq"])
    B, K, _ = cells_c.val.shape
    phi = cells_c[..., 1:4]
    r = V(batch["Yc"]) - cells_c[..., 0]
    gram = vsum(reshape(phi, (B, K, 3, 1)) * reshape(phi, (B, K, 1, 3)), axis=1)
    lam = vsoftplus(P["ridge_rho"])
    A = gram + reshape(lam, (1, 1, 1)) * V(np.eye(3))
    w = solve(A, vsum(phi * reshape(r, (B, K, 1)), axis=1))
    return cells_q[..., 0] + vsum(cells_q[..., 1:4] * reshape(w, (B, 1, 3)), axis=-1)


def forward_cnp(model, P, batch):
    """Conditional neural process: encode (x, y) pairs, mean-pool, decode with the query x."""
    Xc, Yc, Xq = batch["Xc"], batch["Yc"], batch["Xq"]
    B, K, p = Xc.shape
    Q = Xq.shape[1]
    inp = V(np.concatenate([Xc.reshape(B * K, p), Yc.reshape(B * K, 1)], 1))
    rep = reshape(mlp(P, "cenc_", inp, 2), (B, K, -1))
    R = rep.val.shape[-1]
    pooled = reshape(vsum(rep, axis=1) * (1.0 / K), (B, 1, R))
    dec_in = concat([V(Xq), broadcast(pooled, (B, Q, R))], axis=-1)
    return reshape(mlp(P, "cdec_", reshape(dec_in, (B * Q, p + R)), 2), (B, Q))


FORWARD = {"zhu": forward_boards, "liye": forward_boards, "prior": forward_boards, "ridge": forward_ridge,
           "cnp": forward_cnp}


# ---------------------------------------------------------------- interface and registries
TASK_TYPES = ["vector_regression"]
CNP_SIZES = {"enc": (P_IN + 1, 10, 8), "dec": (P_IN + 8, 12, 1)}

MUTANTS = {
    "sign_flip_update": "optimizer ascends the loss instead of descending",
    "zero_lr": "learning rate set to zero",
    "zero_grad_enc_W1": "gradient of the encoder output layer zeroed",
    "inner_outer_swap": "stage-1 facing adds the inner and outer products instead of subtracting",
}


def build_model(in_dim, out_dim, task_type, rng, kind="zhu", n_ctx=K_CTX, hidden=HIDDEN, mutant=None):
    if task_type not in TASK_TYPES or out_dim != 1:
        raise ValueError("supported: vector_regression with out_dim 1")
    p = (in_dim - n_ctx) // (n_ctx + 1)
    if (n_ctx + 1) * p + n_ctx != in_dim:
        raise ValueError("in_dim must equal (n_ctx+1)*p + n_ctx for a flattened episode")
    if kind == "cnp":
        params = init_mlp(rng, [p + 1, *CNP_SIZES["enc"][1:]], "cenc_")
        params.update(init_mlp(rng, [p + CNP_SIZES["enc"][2], *CNP_SIZES["dec"][1:]], "cdec_"))
    else:
        params = init_mlp(rng, [p, hidden, 4], "enc_")
        if kind == "ridge":
            params["ridge_rho"] = np.array([-3.0])
    return {"kind": kind, "params": params, "p": p, "n_ctx": n_ctx, "flags": {"mutant": mutant}}


def unflatten(model, X):
    K, p = model["n_ctx"], model["p"]
    return {"Xc": X[:, :K * p].reshape(-1, K, p), "Yc": X[:, K * p:K * p + K],
            "Xq": X[:, K * p + K:].reshape(-1, 1, p)}


def loss_and_grads(model, batch):
    Pv = {k: V(v) for k, v in model["params"].items()}
    pred = FORWARD[model["kind"]](model, Pv, batch)
    diff = pred - V(batch["Yq"])
    absd = diff * np.where(diff.val >= 0, 1.0, -1.0)
    per = vwhere(np.abs(diff.val) <= HUBER, diff * diff * 0.5, (absd - 0.5 * HUBER) * HUBER)
    loss = vsum(per) * (1.0 / diff.val.size)
    backward(loss)
    grads = {k: (Pv[k].grad if Pv[k].grad is not None else np.zeros_like(v))
             for k, v in model["params"].items()}
    if model["flags"].get("mutant") == "zero_grad_enc_W1" and "enc_W1" in grads:
        grads["enc_W1"] = np.zeros_like(grads["enc_W1"])
    return float(loss.val), grads


def predict_episodes(model, batch, chunk=256):
    Pv = {k: V(v) for k, v in model["params"].items()}
    outs = []
    for s in range(0, len(batch["Xc"]), chunk):
        sub = {k: v[s:s + chunk] for k, v in batch.items()}
        outs.append(FORWARD[model["kind"]](model, Pv, sub).val)
    return np.concatenate(outs, 0)


def predict(model, X):
    return predict_episodes(model, unflatten(model, np.asarray(X, dtype=np.float64)))[:, 0]


def hidden_states(model, X):
    batch = unflatten(model, np.asarray(X, dtype=np.float64))
    Pv = {k: V(v) for k, v in model["params"].items()}
    if model["kind"] == "cnp":
        return {"prediction": FORWARD["cnp"](model, Pv, batch).val}
    cells = board_cells(Pv, batch["Xc"])
    t, d = solve_latents(model, cells, V(batch["Yc"]))
    return {"context_cells": cells.val, "tian": t.val, "di": d.val}


def modules(model):
    if model["kind"] == "cnp":
        return {"cnp_encoder": {"params": ["cenc_W0", "cenc_b0", "cenc_W1", "cenc_b1"],
                                "role": "pairwise encoder with mean pooling", "signature": False},
                "cnp_decoder": {"params": ["cdec_W0", "cdec_b0", "cdec_W1", "cdec_b1"],
                                "role": "query decoder", "signature": False}}
    return {"board_encoder": {"params": ["enc_W0", "enc_b0", "enc_W1", "enc_b1"],
                              "role": "per-point MLP emitting the cells of a bilinear relation (1, t, d, t*d)",
                              "signature": False},
            "facing_eliminator": {"params": [], "role": "pairwise 2x2 determinant (outer minus inner "
                                  "product) removing the first latent", "signature": model["kind"] != "ridge"},
            "facing_of_facings": {"params": [], "role": "cross product of two quadratic coefficient rows; "
                                  "energy-weighted least-squares pooling of the shared root",
                                  "signature": model["kind"] == "zhu"},
            "distractor_inputs": {"params": ["enc_W0[3:5]"], "role": "first-layer weights on nuisance inputs",
                                  "signature": False}}


def knockout(model, name, mode):
    m = {"kind": model["kind"], "p": model["p"], "n_ctx": model["n_ctx"],
         "params": {k: v.copy() for k, v in model["params"].items()}, "flags": dict(model["flags"])}
    if name == "facing_eliminator" and mode == "mean":
        m["flags"]["ko_elim"] = model.get("latent_mean", (0.0, 0.0))
    elif name == "facing_of_facings" and mode == "mean":
        m["flags"]["ko_pool"] = True
    elif name == "distractor_inputs" and mode == "zero":
        m["params"]["enc_W0"][list(DISTRACTORS), :] = 0.0
    else:
        raise ValueError(f"no knockout {name}/{mode}")
    return m


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


# ---------------------------------------------------------------- training
def fit(model, data, budget, rng, val=None, shuffle_labels=False):
    if "X" in data:
        data = dict(unflatten(model, data["X"]), Yq=np.asarray(data["y"], dtype=np.float64)[:, None])
    if shuffle_labels:
        Y = np.concatenate([data["Yc"], data["Yq"]], 1)
        Yp = rng.permutation(Y.ravel()).reshape(Y.shape)
        data = dict(data, Yc=Yp[:, :data["Yc"].shape[1]], Yq=Yp[:, data["Yc"].shape[1]:])
    mut = model["flags"].get("mutant")
    opt = Adam(model["params"], lr=0.0 if mut == "zero_lr" else LR)
    losses, best = [], (np.inf, None)
    for step in range(budget):
        idx = rng.integers(0, len(data["Xc"]), BATCH)
        loss, grads = loss_and_grads(model, take(data, idx))
        grads, _ = clip_global_norm(grads, CLIP)
        if not np.isfinite(loss) or not all(np.isfinite(g).all() for g in grads.values()):
            raise NonFinite(f"non-finite loss or gradient at step {step} ({model['kind']})")
        opt.step(model["params"], grads, sign=-1.0 if mut == "sign_flip_update" else 1.0)
        losses.append(loss)
        if val is not None and (step + 1) % 50 == 0:
            v = float(np.mean((predict_episodes(model, val) - val["Yq"]) ** 2))
            if np.isfinite(v) and v < best[0]:
                best = (v, {k: p.copy() for k, p in model["params"].items()})
    if best[1] is not None:
        model["params"] = best[1]
    if model["kind"] in ("zhu", "liye", "prior"):
        Pv = {k: V(p) for k, p in model["params"].items()}
        sub = take(data, np.arange(min(512, len(data["Xc"]))))
        t, d = solve_latents(model, board_cells(Pv, sub["Xc"]), V(sub["Yc"]))
        model["latent_mean"] = (float(np.median(t.val)), float(np.median(d.val)))
    return np.array(losses)


def r2(model, split):
    pred = predict_episodes(model, split)
    y = split["Yq"]
    if not np.isfinite(pred).all():
        return -1.0
    return float(max(-1.0, 1.0 - ((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()))


def trivial_r2(train, split):
    y = split["Yq"]
    sst = ((y - y.mean()) ** 2).sum()
    glob = 1.0 - ((train["Yq"].mean() - y) ** 2).sum() / sst
    ctx = 1.0 - ((split["Yc"].mean(1, keepdims=True) - y) ** 2).sum() / sst
    return {"global_mean": float(glob), "context_mean": float(ctx)}


def train_kind(kind, world, rng_init, rng_fit, steps, mutant=None, shuffle=False):
    in_dim = (K_CTX + 1) * P_IN + K_CTX
    m = build_model(in_dim, 1, "vector_regression", rng_init, kind=kind, mutant=mutant)
    losses = fit(m, world["train"], steps, rng_fit, val=world["val"], shuffle_labels=shuffle)
    return m, losses


def run_seed(seed, steps, mutant=None):
    ss = np.random.SeedSequence(seed)
    s_law, s_world, s_blind, s_models = ss.spawn(4)
    law = make_law(np.random.default_rng(s_law))
    W, Wb = make_world(s_world, law), make_world(s_blind, law, blind=True)
    rngs = [np.random.default_rng(s) for s in s_models.spawn(14)]
    out = {"models": {}, "losses": {}, "r2": {}, "world": W, "blind": Wb}
    for i, kind in enumerate(["zhu", "cnp", "ridge", "liye", "prior"]):
        m, ls = train_kind(kind, W, rngs[2 * i], rngs[2 * i + 1], steps, mutant if kind == "zhu" else None)
        out["models"][kind], out["losses"][kind] = m, ls
        out["r2"][kind] = {s: r2(m, W[s]) for s in ("heldout", "shifted")}
    for i, kind in enumerate(["zhu", "cnp"]):
        m, _ = train_kind(kind, Wb, rngs[10 + 2 * i], rngs[11 + 2 * i], steps)
        out["r2"][kind + "_blind"] = {"heldout": r2(m, Wb["heldout"])}
    zhu = out["models"]["zhu"]
    for name, mode in (("facing_eliminator", "mean"), ("facing_of_facings", "mean"), ("distractor_inputs", "zero")):
        out["r2"]["ko_" + name] = {s: r2(knockout(zhu, name, mode), W[s]) for s in ("heldout", "shifted")}
    out["trivial"] = {s: trivial_r2(W["train"], W[s]) for s in ("heldout", "shifted")}
    return out


# ---------------------------------------------------------------- tests: correctness
def gradcheck_model(model, batch, rng):
    def lf():
        return loss_and_grads(model, batch)[0]
    _, grads = loss_and_grads(model, batch)
    return finite_difference_check(lf, model["params"], grads, rng)


def consistent_boards(rng, n, mags):
    """Random boards (random parameters and inputs, not a trained model) whose context values are
    generated exactly from latents of the given magnitudes."""
    cells_c = rng.normal(0, 1, (n, K_CTX, 4))
    cells_q = rng.normal(0, 1, (n, N_QRY, 4))
    lat = rng.choice([-1.0, 1.0], (n, 2)) * mags
    t, d = lat[:, :1], lat[:, 1:]
    Yc = cells_c[..., 0] + cells_c[..., 1] * t + cells_c[..., 2] * d + cells_c[..., 3] * t * d
    Yq = cells_q[..., 0] + cells_q[..., 1] * t + cells_q[..., 2] * d + cells_q[..., 3] * t * d
    return cells_c, cells_q, Yc, Yq, t[:, 0], d[:, 0]


def exact_elimination_error(rng, flags, n=2000):
    mags = 10 ** rng.uniform(-2, np.log10(0.98 * BOARD_EDGE), (n, 2))
    cc, cq, Yc, Yq, t, d = consistent_boards(rng, n, mags)
    model = {"kind": "zhu", "flags": flags}
    ts, ds = solve_latents(model, V(cc), V(Yc))
    pred = cq[..., 0] + cq[..., 1] * ts.val[:, None] + cq[..., 2] * ds.val[:, None] + cq[..., 3] * (ts.val * ds.val)[:, None]
    err_lat = np.maximum(np.abs(ts.val - t) / (1 + np.abs(t)), np.abs(ds.val - d) / (1 + np.abs(d)))
    err_pred = np.abs(pred - Yq).max(1) / (1 + np.abs(Yq).max(1))
    return float(np.median(np.maximum(err_lat, err_pred))), float(np.quantile(np.maximum(err_lat, err_pred), 0.99))


def permutation_gap(rng, flags, n=500):
    cc, _, Yc, _, _, _ = consistent_boards(rng, n, rng.uniform(0.1, 1.0, (n, 2)))
    Yc = Yc + 0.1 * rng.normal(0, 1, Yc.shape)
    model, preds = {"kind": "zhu", "flags": flags}, []
    for perm in ([0, 1, 2, 3], [2, 0, 3, 1], [3, 2, 1, 0], [1, 0, 2, 3]):
        t, d = solve_latents(model, V(cc[:, perm]), V(Yc[:, perm]))
        preds.append(np.stack([t.val, d.val], 1))
    return float(max(np.median(np.max(np.abs(p - preds[0]) / (1 + np.abs(preds[0])), 1)) for p in preds[1:]))


def board_homomorphism_gap(rng, reverse_rows=False, n=300):
    worst = 0.0
    for _ in range(n):
        p = [V(rng.normal(0, 1, 8)) for _ in range(rng.integers(1, 5))]
        q = [V(rng.normal(0, 1, 8)) for _ in range(rng.integers(2, 5))]
        z = rng.normal(0, 1.5, 8)
        prod = board_mul(p, q[::-1] if reverse_rows else q)
        ev = lambda coeffs: sum(c.val * z ** k for k, c in enumerate(coeffs))
        worst = max(worst, float(np.max(np.abs(ev(prod) - ev(p) * ev(q)) / (1 + np.abs(ev(p) * ev(q))))))
    return worst


def short_check(seed, steps, mutant=None):
    """C1 + C3 + C6.1 on one model, used for the mutation audit and its clean control."""
    ss = np.random.SeedSequence([seed, 7])
    s_law, s_world, s_m = ss.spawn(3)
    law = make_law(np.random.default_rng(s_law))
    W = make_world(s_world, law)
    r = [np.random.default_rng(s) for s in s_m.spawn(3)]
    m = build_model((K_CTX + 1) * P_IN + K_CTX, 1, "vector_regression", r[0], mutant=mutant)
    g_err, _ = gradcheck_model(m, take(W["train"], np.arange(6)), r[2])
    losses = fit(m, W["train"], steps, r[1], val=None)
    triv = max(trivial_r2(W["train"], W["heldout"]).values())
    c3 = losses[-20:].mean() <= (1 - MIND_CARD["thresholds"]["loss_drop_fraction"]) * losses[:10].mean() and \
        r2(m, W["heldout"]) >= triv + MIND_CARD["thresholds"]["margin_over_trivial"]
    c6 = exact_elimination_error(np.random.default_rng(seed), m["flags"], n=300)[1] < 1e-6
    return {"c1": g_err <= MIND_CARD["thresholds"]["gradcheck_rel_tol"], "c3": bool(c3), "c6": bool(c6)}


def correctness(first, seed, steps, mode, t_start, mutant):
    th = MIND_CARD["thresholds"]
    W, rng = first["world"], np.random.default_rng([seed, 99])
    res, gmax, gtot = [], 0.0, 0
    batch = take(W["train"], np.arange(6))
    for kind in ("zhu", "cnp", "ridge", "liye", "prior"):
        fresh = build_model((K_CTX + 1) * P_IN + K_CTX, 1, "vector_regression", np.random.default_rng([seed, 5]),
                            kind=kind, mutant=mutant if kind == "zhu" else None)
        for m in (fresh, first["models"][kind]):
            e, c = gradcheck_model(m, batch, rng)
            gmax, gtot = max(gmax, e), gtot + c
    res.append(("C1", "gradient_check", gmax <= th["gradcheck_rel_tol"],
                f"{gtot} tensor checks (5 models x init and after {steps} steps), max rel err {gmax:.2e}"))
    runs = [train_kind("zhu", W, np.random.default_rng([seed, 11]), np.random.default_rng([seed, 12]), 30, mutant)
            for _ in range(2)]
    same = np.array_equal(runs[0][1], runs[1][1]) and np.array_equal(
        predict_episodes(runs[0][0], W["val"]), predict_episodes(runs[1][0], W["val"]))
    fin = all(np.isfinite(v).all() for m in first["models"].values() for v in m["params"].values())
    res.append(("C2", "determinism_finiteness", bool(same and fin), f"identical reruns={same}, finite={fin}"))
    ls, triv = first["losses"]["zhu"], max(first["trivial"]["heldout"].values())
    drop = 1 - ls[-20:].mean() / ls[:10].mean()
    hr = first["r2"]["zhu"]["heldout"]
    res.append(("C3", "learning", bool(drop >= th["loss_drop_fraction"] and hr >= triv + th["margin_over_trivial"]),
                f"loss drop {drop:.3f}; heldout R2 {hr:.3f} vs best trivial {triv:.3f}"))
    sm, _ = train_kind("zhu", W, np.random.default_rng([seed, 13]), np.random.default_rng([seed, 14]), steps,
                       mutant, shuffle=True)
    sr = r2(sm, W["heldout"])
    res.append(("C4", "shuffled_label_control", bool(sr <= triv + th["shuffled_band"]),
                f"shuffled-trained heldout R2 {sr:.3f} vs trivial band <= {triv + th['shuffled_band']:.3f}"))
    if mode == "quick" or mutant:
        res.append(("C5", "mutant_detection", True, "mutation audit runs in the default protocol"))
        mscore = None
    else:
        clean = short_check(seed, BUDGET["quick"])
        det = {k: not all(short_check(seed, BUDGET["quick"], k).values()) for k in MUTANTS}
        mscore = {"detected": sum(det.values()), "total": len(det), "score": sum(det.values()) / len(det)}
        res.append(("C5", "mutant_detection", bool(all(clean.values()) and all(det.values())),
                    f"clean control passes={all(clean.values())}; detected {det}"))
    flags = {"mutant": mutant}
    e61, e61q = exact_elimination_error(np.random.default_rng([seed, 21]), flags)
    n61 = exact_elimination_error(np.random.default_rng([seed, 21]), {"mutant": "inner_outer_swap"})[0]
    res.append(("C6.1", "exact_elimination_any_magnitude", bool(e61q < 1e-6 and n61 > 1e-2),
                f"2000 random boards, |latent| 1e-2..5.9 (board edge 6): p99 rel err {e61q:.1e}; negative control (swap) {n61:.1e}"))
    p62 = permutation_gap(np.random.default_rng([seed, 22]), flags)
    n62 = permutation_gap(np.random.default_rng([seed, 22]), dict(flags, anchor_first=True))
    res.append(("C6.2", "context_permutation_invariance", bool(p62 < 1e-9 and n62 > 1e-3),
                f"noisy boards: median gap {p62:.1e}; negative control (pivot-only back-substitution) {n62:.1e}"))
    h63 = board_homomorphism_gap(np.random.default_rng([seed, 23]))
    n63 = board_homomorphism_gap(np.random.default_rng([seed, 23]), reverse_rows=True)
    res.append(("C6.3", "board_shift_homomorphism", bool(h63 < 1e-10 and n63 > 1e-2),
                f"eval(p*q)=eval(p)eval(q) worst {h63:.1e}; negative control (row read reversed) {n63:.1e}"))
    tr = episode_hashes(W["train"]) | episode_hashes(first["blind"]["train"])
    ev = set().union(*(episode_hashes(w[s]) for w in (W, first["blind"]) for s in ("val", "heldout", "shifted")))
    res.append(("C7", "split_integrity", len(tr & ev) == 0, f"{len(tr)} train vs {len(ev)} eval episodes, overlap {len(tr & ev)}"))
    elapsed = time.time() - t_start
    res.append(("C8", "budget", elapsed <= TIME_LIMIT[mode], f"{elapsed:.1f} s of {TIME_LIMIT[mode]:.0f} s"))
    return res, {"max_rel_error": gmax, "checks": gtot}, mscore


# ---------------------------------------------------------------- tests: hypotheses
def hypotheses(runs, rng, mode):
    R = [r["r2"] for r in runs]
    diffs = {
        "H-SIG": [x["zhu"]["shifted"] - x["cnp"]["shifted"] for x in R],
        "H-NEC": [x["ko_facing_eliminator"]["shifted"] - x["ko_distractor_inputs"]["shifted"] for x in R],
        "H-BLIND": [(x["zhu_blind"]["heldout"] - x["cnp_blind"]["heldout"]) - (x["zhu"]["heldout"] - x["cnp"]["heldout"])
                    for x in R],
        "H-RIVAL": [x["zhu"]["shifted"] - x["ridge"]["shifted"] for x in R],
        "H-LIYE": [x["zhu"]["shifted"] - x["liye"]["shifted"] for x in R],
        "H-PRIOR": [x["zhu"]["shifted"] - x["prior"]["shifted"] for x in R]}
    out = []
    for h in MIND_CARD["hypotheses"]:
        d = diffs[h["id"]]
        if mode == "quick" or len(d) < 2:
            out.append({"id": h["id"], "metric": h["metric"], "mean_diff": float(np.mean(d)), "ci95": [None, None],
                        "mesi": h["mesi"], "n_seeds": len(d), "verdict": "not evaluated"})
            continue
        mean, ci = paired_bootstrap(d, rng)
        out.append({"id": h["id"], "metric": h["metric"], "mean_diff": mean, "ci95": ci, "mesi": h["mesi"],
                    "n_seeds": len(d), "verdict": verdict(mean, ci, h["mesi"], h["direction"])})
    kos = []
    for name, sig in (("facing_eliminator", True), ("facing_of_facings", True), ("distractor_inputs", False)):
        d = [x["ko_" + name]["shifted"] - x["zhu"]["shifted"] for x in R]
        mean, ci = paired_bootstrap(d, rng) if len(d) > 1 else (float(d[0]), [None, None])
        kos.append({"module": name, "signature": sig, "metric_change": mean, "ci95": ci})
    return out, kos


# ---------------------------------------------------------------- report
def fmt_ci(ci):
    return "      n/a        " if ci[0] is None else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def print_report(rep, runs):
    print(f"=== VERIFIED REPORT · chapter 0347 ===\nfile {rep['file']}  card_revision {rep['card_revision']}")
    print(f"python {rep['environment']['python']}  numpy {rep['environment']['numpy']}")
    print(f"seeds {rep['seeds']}  runtime {rep['runtime_s']:.1f} s  task_types {rep['task_types']}")
    print("parameters: " + ", ".join(f"{k} {v}" for k, v in rep["n_params_by_model"].items()))
    g = rep["gradcheck"]
    print(f"gradient check: {g['tensors_checked']} tensor checks, max rel error {g['max_rel_error']:.2e}, "
          f"at {g['checked_at']}, passed={g['passed']}")
    print("correctness:")
    for c in rep["correctness"]:
        print(f"  {c['id']:<5} {c['name']:<34} {'PASS' if c['passed'] else 'FAIL'}  {c['detail']}")
    m = rep["mutants"]
    print(f"mutation score: {m['detected']}/{m['total']}" if m["total"] else "mutation score: not run (quick/mutant)")
    print("R2 by split (mean ± sd over seeds; floored at -1):")
    for key in ("zhu", "ridge", "liye", "prior", "cnp"):
        vals = {s: [r["r2"][key][s] for r in runs] for s in ("heldout", "shifted")}
        print(f"  {key:<6} heldout {np.mean(vals['heldout']):.3f} ± {np.std(vals['heldout']):.3f}   "
              f"shifted {np.mean(vals['shifted']):.3f} ± {np.std(vals['shifted']):.3f}")
    for key in ("zhu_blind", "cnp_blind"):
        v = [r["r2"][key]["heldout"] for r in runs]
        print(f"  {key:<10} blind-world heldout {np.mean(v):.3f} ± {np.std(v):.3f}")
    print("  trivial (seed %d): " % rep["seeds"][0] + json.dumps(
        {s: {k: round(v, 3) for k, v in d.items()} for s, d in runs[0]["trivial"].items()}))
    print("hypotheses:")
    for h in rep["hypotheses"]:
        print(f"  {h['id']:<8} mean diff {h['mean_diff']:+.3f}  ci95 {fmt_ci(h['ci95'])}  mesi {h['mesi']:.2f}  "
              f"n={h['n_seeds']}  -> {h['verdict']}")
    print("knockouts (shifted R2 change vs full model):")
    for k in rep["knockouts"]:
        print(f"  {k['module']:<20} signature={str(k['signature']):<5} change {k['metric_change']:+.3f}  ci95 {fmt_ci(k['ci95'])}")
    print(f"exit code {rep['exit_code']}")
    print("=== END REPORT ===")


def data_bridge(path, seed):
    """Optional real-data bridge: CSV with columns episode, feature..., y (no downloads)."""
    if not path or not os.path.exists(path):
        print(f"--data: no file at {path!r}; real-data bridge skipped.")
        return
    raw = np.genfromtxt(path, delimiter=",", names=True)
    feats = [n for n in raw.dtype.names if n not in ("episode", "y")]
    eps = {}
    for row in raw:
        eps.setdefault(row["episode"], []).append([row[f] for f in feats] + [row["y"]])
    rows = [np.array(v[:K_CTX + 1]) for v in eps.values() if len(v) >= K_CTX + 1]
    A = np.stack(rows)
    X = np.concatenate([A[:, :K_CTX, :-1].reshape(len(A), -1), A[:, :K_CTX, -1], A[:, K_CTX, :-1]], 1)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(A))
    cut = int(0.8 * len(A))
    for kind in ("zhu", "cnp"):
        m = build_model(X.shape[1], 1, "vector_regression", rng, kind=kind)
        fit(m, {"X": X[order[:cut]], "y": A[order[:cut], K_CTX, -1]}, BUDGET["default"], rng)
        y, p = A[order[cut:], K_CTX, -1], predict(m, X[order[cut:]])
        print(f"--data {os.path.basename(path)}: {kind} test R2 {1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum():.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__), description="Chapter 0347 Facing Boards protocol")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--card", action="store_true")
    ap.add_argument("--mutant", choices=sorted(MUTANTS), default=None)
    ap.add_argument("--data", default=None)
    args = ap.parse_args(argv)
    if args.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    mode = "quick" if args.quick else "default"
    n_seeds = args.seeds if args.seeds is not None else (1 if args.quick else 5)
    if n_seeds < 1:
        print(f"usage: python3 {os.path.basename(__file__)} [--quick] [--seeds N>=1]")
        return 2
    t0 = time.time()
    seeds = [args.seed + i for i in range(n_seeds)]
    print(f"seeds {seeds}  mode {mode}  budget {BUDGET[mode]} updates per model")
    try:
        runs = []
        for s in seeds:
            runs.append(run_seed(s, BUDGET[mode], args.mutant))
            print(f"  seed {s} done at {time.time() - t0:.1f} s")
            if args.mutant:
                break
        corr, gsum, mscore = correctness(runs[0], seeds[0], BUDGET[mode], mode, t0, args.mutant)
    except NonFinite as exc:
        print(f"non-finite values: {exc}")
        return 4
    hyp, kos = hypotheses(runs, np.random.default_rng([seeds[0], 2000]), mode if not args.mutant else "quick")
    passed = all(c[2] for c in corr)
    code = 0 if passed else (3 if not [c for c in corr if c[0] == "C8"][0][2] else 1)
    models = runs[0]["models"]
    rep = {"schema_version": "1.0", "chapter": 347, "file": os.path.basename(__file__),
           "card_revision": MIND_CARD["card_revision"],
           "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
           "seeds": seeds, "runtime_s": time.time() - t0, "n_params": n_params(models["zhu"]),
           "n_params_by_model": {k: n_params(m) for k, m in models.items()},
           "gradcheck": {"tensors_checked": gsum["checks"], "tensors_total": gsum["checks"],
                         "max_rel_error": gsum["max_rel_error"], "checked_at": ["init", "after_training_steps"],
                         "passed": bool(corr[0][2])},
           "correctness": [{"id": c[0], "name": c[1], "passed": bool(c[2]), "detail": c[3]} for c in corr],
           "mutants": mscore or {"detected": 0, "total": 0, "score": None},
           "hypotheses": hyp, "knockouts": kos, "task_types": TASK_TYPES, "exit_code": code,
           "r2": [r["r2"] for r in runs], "trivial": [r["trivial"] for r in runs]}
    print_report(rep, runs)
    if args.json:
        write_report(rep, args.json)
    if args.data:
        data_bridge(args.data, args.seed)
    return code


if __name__ == "__main__":
    sys.exit(main())

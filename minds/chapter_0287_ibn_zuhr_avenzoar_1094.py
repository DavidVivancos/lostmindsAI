#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0287 · Ibn Zuhr (Avenzoar)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 15, Minds 281-300 Available on Amazon https://www.amazon.com/dp/B0HMQ7TVJM
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0287_ibn_zuhr_avenzoar_1094 - Ibn Zuhr (Avenzoar) (c.1094-1162)
# END ATTRIBUTION
"""
TADBIR -- the titrated particular.  Chapter 0287 · Ibn Zuhr (Avenzoar) (c.1094-1162).

Thesis
  The particular case is neither stored (a casebook) nor deduced (a book of
  universals) but titrated: act mildly from the universal prior, wait out the
  individual's own response window, escalate only on observed benefit, and let
  the particular outweigh the book only in proportion to first-hand evidence.

Evidence and provenance (provenance: belief -- his own surviving Taysir,
read through the al-Khuri edition and published translations)
  D1  Mild drugs first; watch the patient's reaction, above all in the first
      three days; a larger dose only if the drug proved useful (Taysir, as
      summarised by S. Hamarneh, Dictionary of Scientific Biography).
  D2  Averroes wrote the Kulliyyat (universals) and asked Ibn Zuhr for the
      Taysir (particulars); Averroes held that particular premises cannot be
      written because they are unlimited, yet trusted the Taysir (Kulliyyat
      VII.517, VII.583; Taysir, introduction).
  D3  Knowledge by imitation does not license the knife, least of all on
      small organs (Taysir, ed. al-Khuri 1983, p.141; tr. Abdel-Halim 2005).
  D5  Purgatives are poisons when abused (al-Tadhkira, per Hamarneh).
  D7  He abhorred cutting for the stone and preferred slow dissolving
      regimens (Taysir, per Abdel-Halim 2005) -> the blind spot below.

Doctrine -> mechanism -> test (IDs as in MIND_CARD)
  D1 -> M1 mild first probe, at most half the prior's optimum   -> C6.5 (definition)
  D1 -> M2 non-negative three-day watch with onset template      -> C6.3, knockouts
  D1 -> M3 evidence-gated, one-sided escalation                  -> C6.1, knockouts
  D2 -> M4 universal prior net + closed-form per-case posterior  -> C6.2, H-SIG, H-NEC, H-RIVAL
  D3 -> M4 precision weighting (evidence moves the act only as far as it is first-hand) -> C6.2
  D5 -> objective: asymmetric overdose cost on every dose given
  D7 -> threshold responders                                     -> H-BLIND

Research question (sample efficiency and transfer)
  Does meta-learning a universal prior across patients while solving each
  patient's response in closed form, from mild, time-extended, harm-charged
  probes, personalise better out of distribution than a size-matched amortised
  network that reads the same observations?

Closest prior art and the delta
  Bayesian dose escalation with overdose control (EWOC, Babb, Rogatko & Zacks
  1998; CRM, O'Quigley, Pepe & Fisher 1990); amortised in-context learners
  (conditional neural processes, Garnelo et al. 2018); meta-learning with
  closed-form inner solvers (Bertinetto et al. 2019).  Delta: probe strength,
  the watch window's day weights, the escalation gate and the universal prior
  are trained end to end through the patient's delayed response and the
  overdose cost, while the particular itself is never amortised: it is solved
  per case in closed form.  The amortiser baseline is also the 0298 (Averroes)
  shared substrate rebuilt from its description.

Blind spot
  Threshold responders: benefit appears only above a dose the mild probe never
  reaches, so the titrator reads "not useful", does not escalate, underdoses.

Task (per synthetic patient; three actions)
  x ~ N(mu, I_6) observable signs; log beta = a_b.x + s_u u1; log gamma = -1 + a_g.x + s_u u2
  onset kappa_t = 1 - exp(-t/tau), tau ~ U(0.8, 2.5) days, t = 1..3
  probe s at dose d_s: benefit trace B(d_s) kappa_t + 0.08 e, harm trace gamma d_s^2 kappa_t + 0.03 e
  B(d) = beta d (blind split: beta srelu(d - theta)); U(d) = B(d) - gamma d^2
  cost = (U* - U(d_final)) / U_ref + 3 sum_{d in all doses} srelu(d/d_ref - rho)^2
  splits: id (s_u .35), shift (mu shifted, s_u .6), blind (theta = phi d_ref, phi ~ U(.25,.45))

Limits
  Synthetic patients and an abstract intervention; not a dosing tool and no
  treatment recommendation.  A research prototype of one AGI-relevant
  mechanism (safe per-case adaptation), not an AGI.  It does not replicate the
  historical person; it tests one idea read from his method.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 1,
    "revision_log": [],
    "generation": {"template_version": "codeguidelines 1.0 (2026-09-15)", "generator": "Claude",
                   "generator_version": "Claude Opus 5.5", "date": "2026-09-23"},
    "id": 287, "figure": "Ibn Zuhr (Avenzoar)", "born": 1094, "died": 1162,
    "civilization": "Andalusi Arab", "provenance": "belief",
    "thesis": ("The particular case is neither stored nor deduced but titrated: act mildly from the "
               "universal prior, watch the individual's response window, escalate only on observed "
               "benefit, and let first-hand evidence outweigh the book only in proportion to its precision."),
    "evidence": [
        {"id": "D1", "claim": "mild drugs first; watch the first three days; a larger dose only if useful",
         "basis": "scholarship", "source": "Taysir, summarised in S. Hamarneh, 'Ibn Zuhr', Dictionary of Scientific Biography"},
        {"id": "D2", "claim": "universals (Kulliyyat) and particulars (Taysir) as a commissioned division of labour; "
         "particular premises cannot be written because unlimited, the Taysir excepted",
         "basis": "primary", "source": "Averroes, Kulliyyat VII.517, VII.583; Taysir, introduction (ed. al-Khuri 1983, p.7)"},
        {"id": "D3", "claim": "knowledge by imitation does not license the knife, especially on small organs",
         "basis": "primary", "source": "Taysir, ed. al-Khuri 1983, p.141 (tr. Abdel-Halim 2005)"},
        {"id": "D4", "claim": "settles a controversy (tracheotomy) by a test on a goat; infers an unknown remedy from sick sheep",
         "basis": "primary", "source": "Taysir, ed. al-Khuri 1983, pp.149, 129 (tr. Abdel-Halim 2005)"},
        {"id": "D5", "claim": "purgatives are poisons when abused", "basis": "scholarship",
         "source": "al-Tadhkira, per Hamarneh, DSB"},
        {"id": "D6", "claim": "operate only after long practice under a teacher; the physician stops where the surgeon must come",
         "basis": "primary", "source": "Taysir, ed. al-Khuri 1983, p.27 (tr. Abdel-Halim 2005)"},
        {"id": "D7", "claim": "abhorred lithotomy; preferred dissolving regimens", "basis": "primary",
         "source": "Taysir (per Abdel-Halim 2005, section VI)"},
        {"id": "D8", "claim": "the particular as a procedure, not a record (this chapter's reading)",
         "basis": "speculation", "source": "interpretation of D1-D3"},
    ],
    "research_question": {"category": "sample efficiency and transfer",
                          "question": ("Does a meta-learned universal prior plus a closed-form per-case posterior, fed by mild, "
                                       "time-extended, harm-charged probes, personalise better out of distribution than a "
                                       "size-matched amortised network reading the same observations?")},
    "mechanism": {
        "name": "Tadbir titrator",
        "family": "meta-learned prior + closed-form per-case system identification + evidence-gated safe escalation",
        "signature_modules": ["probe", "watch", "particular", "escalation"],
        "closest_prior_art": ["EWOC (Babb, Rogatko & Zacks 1998)", "CRM (O'Quigley, Pepe & Fisher 1990)",
                              "Conditional neural processes (Garnelo et al. 2018)",
                              "Meta-learning with differentiable closed-form solvers (Bertinetto et al. 2019)"],
        "overlap": "Medium",
        "prior_art_queries": ["dose escalation with overdose control", "continual reassessment method",
                              "amortized in-context learning neural process", "meta-learning closed-form ridge inner loop",
                              "safe exploration personalization delayed response"],
        "contribution_type": "mechanism",
        "delta": ("probe strength, watch-window weights, escalation gate and universal prior trained end to end through "
                  "delayed responses and an overdose cost, while the particular is solved per case in closed form, never amortised"),
    },
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1 probe", "property_test": "C6.5", "hypothesis": "knockout table"},
        {"doctrine": "D1", "mechanism": "M2 watch", "property_test": "C6.3", "hypothesis": "knockout table"},
        {"doctrine": "D1", "mechanism": "M3 escalation", "property_test": "C6.1", "hypothesis": "knockout table"},
        {"doctrine": "D2", "mechanism": "M4 particular", "property_test": "C6.2", "hypothesis": "H-SIG, H-NEC, H-RIVAL"},
        {"doctrine": "D3", "mechanism": "M4 particular", "property_test": "C6.2", "hypothesis": "H-NEC"},
        {"doctrine": "D5", "mechanism": "objective", "property_test": "C6.4", "hypothesis": "none"},
        {"doctrine": "D7", "mechanism": "M1+M3", "property_test": "none", "hypothesis": "H-BLIND"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "the titrator personalises better than the size-matched amortiser on the shifted split",
         "metric": "tadbir cost", "split": "shift", "comparison": "model - amortizer", "direction": "less",
         "mesi": 0.02, "seeds": 5},
        {"id": "H-NEC", "statement": "removing the closed-form particular hurts more than removing the universal prior",
         "metric": "tadbir cost", "split": "id", "comparison": "ko(particular) - ko(kulliyat)",
         "direction": "greater", "mesi": 0.05, "seeds": 5},
        {"id": "H-BLIND", "statement": "on threshold responders the titrator loses ground to the amortiser",
         "condition": "blind (threshold responders)", "grounding": "aversion to decisive intervention (D7)",
         "metric": "tadbir cost", "comparison": "(model - amortizer)[blind] - (model - amortizer)[id]",
         "direction": "greater", "mesi": 0.02, "seeds": 5},
        {"id": "H-RIVAL", "statement": "titrated particulars beat universals alone even after paying for the probes",
         "metric": "tadbir cost", "split": "id", "comparison": "model - kulliyat_only", "direction": "less",
         "mesi": 0.02, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.30, "margin_over_trivial": 0.25, "shuffled_band": 0.15,
                   "gradcheck_rel_tol": 1e-5, "gradcheck_denominator_floor": 1e-5},
    "probe_predictions": [{"probe": "P6", "expected": "above baseline"}, {"probe": "P10", "expected": "above baseline"},
                          {"probe": "P11", "expected": "above baseline"}, {"probe": "P9", "expected": "equal to baseline"}],
    "dialectic_links": [
        {"chapter": 298, "relation": "rival", "test": "H-SIG: 0298's shared substrate actualised by private inputs, rebuilt as the amortizer"},
        {"chapter": 298, "relation": "rival", "test": "H-RIVAL: Kulliyyat-only universals (particulars left unwritten)"},
        {"chapter": 233, "relation": "successor", "test": "none: casebook of named counterexamples contrasted in prose"},
    ],
    "corpus_neighbors": [
        {"chapter": 181, "similarity": None, "difference": "Sun Simiao sets an anchor by the other's reply; here an intervention's magnitude is titrated over a delayed window under an overdose cost"},
        {"chapter": 233, "similarity": None, "difference": "al-Razi stores counterexamples beside a frozen model; here particulars are never stored, only solved per case"},
        {"chapter": 243, "similarity": None, "difference": "al-Zahrawi fixes an intervention's extent before acting; here the extent is fixed only after watching"},
        {"chapter": 279, "similarity": None, "difference": "Judah Halevi licenses amendment of inherited practice; here the prior is never amended per case, only outweighed"},
        {"chapter": 284, "similarity": None, "difference": "Ibn Bajja converges private codes without communication; different family"},
        {"chapter": 298, "similarity": None, "difference": "Averroes' shared substrate is this file's amortising rival"},
    ],
    "barometer": {
        "cognitive_processing": ["sample efficiency: two probes per case"], "embodied_cognition": ["closed-loop adaptation to an unknown plant"],
        "world_modeling": ["delayed-response onset template"], "consciousness": ["posterior precision gates how far evidence moves the act"],
        "language_understanding": [], "emotional_intelligence": [], "creativity": [],
        "autonomy": ["self-set probe schedule under an overdose budget"]},
    "task_types": ["episodic_control"],
    "applications": [
        {"use": "per-learner pacing: mild first items, a response window, difficulty raised on observed mastery",
         "sector": "education", "dataset": "ASSISTments 2009-2010 skill-builder (public)"},
        {"use": "offline evaluation of personalised dose-finding designs (research decision support only)",
         "sector": "clinical research", "dataset": "IWPC warfarin pharmacogenetics dataset (public, de-identified)"},
        {"use": "per-occupant thermal-comfort titration of setpoints with minimal discomfort",
         "sector": "buildings and energy", "dataset": "ASHRAE Global Thermal Comfort Database II (public)"}],
    "safety_notes": ("Synthetic patients; abstract intervention units; no dosing or treatment recommendation. "
                     "Covariates are neutral synthetic signs: no humoral or demographic doctrine is encoded."),
}

import argparse
import json
import math
import os
import sys
import time
import hashlib

import numpy as np

CHAPTER = 287
W_DAYS = 3
N_SIGNS = 6
LAMBDA_HARM = 3.0
SIGMA_B, SIGMA_H = 0.08, 0.03
DOSE_CAP = 20.0
B_SCALE, H_SCALE = 0.5, 0.1
SPLIT_SIZES = {"train": 3000, "val": 600, "id": 1500, "shift": 1500, "blind": 1500}
ACTIVE_MUTANT = None

# BEGIN STANDARD UTILITIES v1.0
def softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def logsumexp(z, axis=-1):
    m = np.max(z, axis=axis, keepdims=True)
    return np.squeeze(m + np.log(np.sum(np.exp(z - m), axis=axis, keepdims=True)), axis=axis)


def softplus(z):
    return np.logaddexp(0.0, z)


class Adam:
    def __init__(self, params, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
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
    total = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    if total > max_norm:
        for k in grads:
            grads[k] = grads[k] * (max_norm / (total + 1e-12))
    return total


def finite_difference_check(loss_fn, params, grads, rng, eps=1e-6, n_entries=20, floor=1e-5):
    """Central differences on every tensor; >= n_entries random entries plus the largest-gradient entry."""
    worst = {}
    for k, p in params.items():
        flat, g = p.reshape(-1), grads[k].reshape(-1)
        if flat.size <= n_entries:
            idx = np.arange(flat.size)
        else:
            idx = np.unique(np.append(rng.choice(flat.size, n_entries, replace=False), np.argmax(np.abs(g))))
        w = 0.0
        for i in idx:
            old = flat[i]
            flat[i] = old + eps
            lp = loss_fn(params)
            flat[i] = old - eps
            lm = loss_fn(params)
            flat[i] = old
            num = (lp - lm) / (2 * eps)
            w = max(w, abs(num - g[i]) / max(abs(num) + abs(g[i]), floor))
        worst[k] = float(w)
    return worst


def paired_bootstrap(diffs, rng, n_boot=2000, alpha=0.05):
    d = np.asarray(diffs, dtype=np.float64)
    means = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    return float(d.mean()), [float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))]


def verdict(mean, ci, mesi, direction):
    if direction == "less":
        if ci[1] < 0 and mean <= -mesi:
            return "supported"
        return "contradicted" if ci[0] > 0 else "inconclusive"
    if ci[0] > 0 and mean >= mesi:
        return "supported"
    return "contradicted" if ci[1] < 0 else "inconclusive"


def write_report(report, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
# END STANDARD UTILITIES


# ----------------------------------------------------------------------------- reverse-mode tape
class V:
    """A node on a minimal reverse-mode tape.  `req` marks nodes that lead to parameters."""
    __slots__ = ("v", "g", "parents", "back", "req")
    __array_ufunc__ = None      # make ndarray (op) V defer to V's reflected operators

    def __init__(self, v, parents=(), back=None, req=False):
        self.v = np.asarray(v, dtype=np.float64)
        self.g, self.parents, self.back = None, parents, back
        self.req = req or any(p.req for p in parents)

    def __add__(self, o):
        o = lift(o)
        return V(self.v + o.v, (self, o), lambda g: (unb(g, self.v.shape), unb(g, o.v.shape)))

    def __sub__(self, o):
        o = lift(o)
        return V(self.v - o.v, (self, o), lambda g: (unb(g, self.v.shape), unb(-g, o.v.shape)))

    def __mul__(self, o):
        o = lift(o)
        return V(self.v * o.v, (self, o), lambda g: (unb(g * o.v, self.v.shape), unb(g * self.v, o.v.shape)))

    def __truediv__(self, o):
        o = lift(o)
        return V(self.v / o.v, (self, o),
                 lambda g: (unb(g / o.v, self.v.shape), unb(-g * self.v / (o.v * o.v), o.v.shape)))

    def __matmul__(self, o):
        def back(g):
            if o.v.ndim == 1:
                return (np.outer(g, o.v), self.v.T @ g)
            return (g @ o.v.T, self.v.T @ g)
        return V(self.v @ o.v, (self, o), back)

    def __neg__(self):
        return V(-self.v, (self,), lambda g: (-g,))

    def __getitem__(self, idx):
        def back(g):
            z = np.zeros_like(self.v)
            z[idx] = g
            return (z,)
        return V(self.v[idx], (self,), back)

    def col(self):
        return V(self.v[:, None], (self,), lambda g: (g[:, 0],))

    __radd__ = __add__
    __rmul__ = __mul__

    def __rsub__(self, o):
        return lift(o) - self

    def __rtruediv__(self, o):
        return lift(o) / self


def lift(o):
    return o if isinstance(o, V) else V(o)


def unb(g, shape):
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for ax, n in enumerate(shape):
        if n == 1 and g.shape[ax] != 1:
            g = g.sum(axis=ax, keepdims=True)
    return g


def _unary(a, f, df):
    y = f(a.v)
    return V(y, (a,), lambda g: (g * df(a.v, y),))


def vexp(a):
    return _unary(a, np.exp, lambda x, y: y)


def vlog(a):
    return _unary(a, np.log, lambda x, y: 1.0 / x)


def vtanh(a):
    return _unary(a, np.tanh, lambda x, y: 1.0 - y * y)


def vsig(a):
    return _unary(a, lambda x: 0.5 * (1.0 + np.tanh(0.5 * x)), lambda x, y: y * (1.0 - y))


def vsoftplus(a):
    return _unary(a, softplus, lambda x, y: 0.5 * (1.0 + np.tanh(0.5 * x)))


def vsqrt(a):
    return _unary(a, np.sqrt, lambda x, y: 0.5 / y)


def vsum(a, axis=None):
    def back(g):
        return (np.broadcast_to(g if axis is None else np.expand_dims(g, axis), a.v.shape).copy(),)
    return V(a.v.sum(axis=axis), (a,), back)


def vcat(parts, axis=1):
    sizes = np.cumsum([p.v.shape[axis] for p in parts])[:-1]
    return V(np.concatenate([p.v for p in parts], axis=axis), tuple(parts),
             lambda g: tuple(np.split(g, sizes, axis=axis)))


def srelu(z, delta=0.01):
    """Smooth one-sided map 0.5 (z + sqrt(z^2 + delta^2)): positive, increasing, no kink."""
    z = lift(z)
    return 0.5 * (z + vsqrt(z * z + delta * delta))


def backward(root):
    order, seen, stack = [], set(), [(root, False)]
    while stack:
        node, done = stack.pop()
        if done:
            order.append(node)
            continue
        if id(node) in seen or not node.req:
            continue
        seen.add(id(node))
        stack.append((node, True))
        stack.extend((p, False) for p in node.parents if p.req)
    root.g = np.ones_like(root.v)
    for node in reversed(order):
        if node.back is None or node.g is None:
            continue
        for p, gp in zip(node.parents, node.back(node.g)):
            if p.req:
                p.g = gp if p.g is None else p.g + gp


# ----------------------------------------------------------------------------- data and task
def make_task():
    """Population law linking observable signs to the two response coefficients (fixed for all seeds)."""
    rng = np.random.default_rng(np.random.SeedSequence(CHAPTER))
    a_b, a_g = rng.normal(size=N_SIGNS), rng.normal(size=N_SIGNS)
    return {"a_b": 0.45 * a_b / np.linalg.norm(a_b), "a_g": 0.40 * a_g / np.linalg.norm(a_g)}


TASK = make_task()
REGIMES = {"id": {"mu": 0.0, "su": 0.35, "blind": False},
           "shift": {"mu": np.array([0.8, -0.8, 0.5, 0.0, 0.0, 0.0]), "su": 0.60, "blind": False},
           "blind": {"mu": 0.0, "su": 0.35, "blind": True}}


def utility_np(P, D):
    th = P["theta"][:, None]
    ben = np.where(th <= 0, D, 0.5 * ((D - th) + np.sqrt((D - th) ** 2 + 0.02 ** 2)))
    return P["beta"][:, None] * ben - P["gamma"][:, None] * D ** 2


def make_patients(rng, n, regime):
    r = REGIMES[regime]
    x = rng.normal(size=(n, N_SIGNS)) + r["mu"]
    beta = np.exp(x @ TASK["a_b"] + r["su"] * rng.normal(size=n))
    gamma = np.exp(-1.0 + x @ TASK["a_g"] + r["su"] * rng.normal(size=n))
    tau = rng.uniform(0.8, 2.5, size=n)
    dref = beta / (2.0 * gamma)
    P = {"x": x, "beta": beta, "gamma": gamma, "dref": dref, "uref": beta ** 2 / (4.0 * gamma),
         "kappa": 1.0 - np.exp(-np.arange(1, W_DAYS + 1)[None, :] / tau[:, None]),
         "theta": rng.uniform(0.25, 0.45, size=n) * dref if r["blind"] else np.zeros(n),
         "rho": rng.uniform(1.3, 1.9, size=n),
         "eb": rng.normal(size=(n, 2, W_DAYS)), "eh": rng.normal(size=(n, 2, W_DAYS))}
    if r["blind"]:
        grid = np.linspace(1e-3, 3.0, 1500)[None, :] * dref[:, None]
        P["ustar"] = utility_np(P, grid).max(axis=1)
    else:
        P["ustar"] = P["uref"].copy()
    for k in ("beta", "gamma", "theta"):
        P["t" + k] = P[k]       # target copies: the shuffled-label control permutes only these
    return P


def make_data(seed):
    out = {}
    for i, (split, n) in enumerate(SPLIT_SIZES.items()):
        rng = np.random.default_rng(np.random.SeedSequence([CHAPTER, seed, i]))
        out[split] = make_patients(rng, n, split if split in REGIMES else "id")
    return out


def take(P, idx):
    return {k: v[idx] for k, v in P.items()}


TARGET_KEYS = ("tbeta", "tgamma", "ttheta", "dref", "uref", "ustar", "rho")


def benefit(P, d, target=False):
    pre = "t" if target else ""
    lin = (P[pre + "theta"] <= 0).astype(np.float64)
    return P[pre + "beta"] * (lin * d + (1.0 - lin) * srelu(d - P[pre + "theta"], 0.02))


def observe(P, stage, d):
    """Three days of benefit and side-effect traces after dose d (frozen per-patient noise)."""
    b = benefit(P, d).col() * P["kappa"] + SIGMA_B * P["eb"][:, stage, :]
    h = (P["gamma"] * d * d).col() * P["kappa"] + SIGMA_H * P["eh"][:, stage, :]
    return b, h


def tadbir_cost(P, doses, harm_variant=None):
    """Normalised regret of the final dose plus an overdose cost on every dose given (D5)."""
    df = doses[-1]
    regret = (P["ustar"] - (benefit(P, df, True) - P["tgamma"] * df * df)) / P["uref"]
    harm = 0.0
    for d in doses:
        z = (P["rho"] - d / P["dref"]) if harm_variant == "mirrored" else (d / P["dref"] - P["rho"])
        s = srelu(z)
        harm = s * s + harm
    return regret + LAMBDA_HARM * harm, regret, harm


# ----------------------------------------------------------------------------- model: the titrator
def inv_softplus(y):
    return np.log(np.expm1(np.asarray(y, dtype=np.float64)))


def init_titrator(rng, in_dim=N_SIGNS, hidden=16):
    return {"W1": rng.normal(0.0, 0.3, (in_dim, hidden)), "b1": np.zeros(hidden),
            "W2": rng.normal(0.0, 0.1, (hidden, 4)), "b2": np.array([0.5413, -0.8036, 1.1, 3.4]),
            "probe": np.zeros(1), "tmpl": inv_softplus([0.5, 0.75, 0.88]), "watch": np.zeros(W_DAYS),
            "log_sb": np.log([0.16]), "log_sh": np.log([0.06]), "esc": np.array([0.5413, 1.0, 0.0]),
            "fin": np.zeros(1)}


def kulliyat(p, P, ko):
    """Universal prior: Gaussian prior (mean, precision) on beta and gamma from observable signs."""
    h = vtanh(V(P["x"]) @ p["W1"] + p["b1"])
    o = h @ p["W2"] + p["b2"]
    prior = [vsoftplus(o[:, 0]), vsoftplus(o[:, 1]), vexp(o[:, 2]), vexp(o[:, 3])]
    if "kulliyat" in ko:
        prior = [V(np.full(P["x"].shape[0], m)) for m in ko["kulliyat"]]
    return prior, h


def watch_weights(p, ko, variant):
    if variant == "signed_watch":
        om = W_DAYS * vtanh(p["watch"])
    else:
        e = vexp(p["watch"])
        om = W_DAYS * e / vsum(e)
    if "watch" in ko:
        om = V(np.eye(W_DAYS)[0] * W_DAYS)
    return om


def evidence(d, b, h, m_t, om, sb2, sh2, off=False):
    """Precision and information-weighted sums contributed by one probe's watch window."""
    dm, d2m = d.col() * m_t, (d * d).col() * m_t
    ev = [vsum(om * dm * dm, 1) / sb2, vsum(om * dm * b, 1) / sb2,
          vsum(om * d2m * d2m, 1) / sh2, vsum(om * d2m * h, 1) / sh2]
    return [e * 0.0 for e in ev] if off else ev


def posterior(prior, ev, variant=None):
    mb, mg, tb, tg = prior
    k = 2.0 if variant == "overconfident" else 1.0
    return (tb * mb + k * ev[1]) / (tb + ev[0]), (tg * mg + k * ev[3]) / (tg + ev[2])


def plug_in(bh, gh):
    return srelu(bh, 0.02) / (2.0 * srelu(gh, 0.02))


def titrator_forward(model, P, params=None):
    params = model["params"] if params is None else params
    ko, var = model["ko"], model.get("variant")
    p = {k: V(v, req=True) for k, v in params.items()}
    prior, hid = kulliyat(p, P, ko)
    d0 = prior[0] / (2.0 * prior[1])
    d1 = 0.5 * vsig(p["probe"]) * d0
    m_t, om = vsoftplus(p["tmpl"]), watch_weights(p, ko, var)
    sb2, sh2 = vexp(2.0 * p["log_sb"]), vexp(2.0 * p["log_sh"])
    b1, h1 = observe(P, 0, d1)
    e1 = evidence(d1, b1, h1, m_t, om, sb2, sh2, "particular" in ko)
    bh1, gh1 = posterior(prior, e1, var)
    z = e1[1] / vsqrt(e1[0] + 1e-9)
    gate = vsig(vsoftplus(p["esc"][0]) * (z - p["esc"][1])) * (0.0 if "escalation" in ko else 1.0)
    gap = plug_in(bh1, gh1) - d1
    step = vtanh(gap) if var == "tanh_step" else srelu(gap)
    d2 = d1 + gate * vsig(p["esc"][2]) * step
    b2, h2 = observe(P, 1, d2)
    e2 = evidence(d2, b2, h2, m_t, om, sb2, sh2, "particular" in ko)
    bh2, gh2 = posterior(prior, [a + c for a, c in zip(e1, e2)], var)
    scale = 1.0 if "final" in ko else vexp(p["fin"])
    df = DOSE_CAP * vtanh(scale * plug_in(bh2, gh2) / DOSE_CAP)
    out = {"doses": [d1, d2, df], "d0": d0, "gate": gate, "z": z, "prior": prior, "hidden": hid,
           "post1": (bh1, gh1), "post2": (bh2, gh2), "ev1": e1, "om": om}
    return out, p


# ----------------------------------------------------------------------------- baselines and rival mechanisms
def init_amortizer(rng, in_dim=N_SIGNS, hidden=8):
    f1 = in_dim + 2 * W_DAYS + 1
    return {"v1": np.zeros(in_dim), "c1": np.log([0.35]), "v2": np.zeros(f1), "c2": np.log([0.8]),
            "U1": rng.normal(0.0, 0.3, (f1 + 2 * W_DAYS + 1, hidden)), "u1": np.zeros(hidden),
            "U2": rng.normal(0.0, 0.1, (hidden, 1)), "u2": np.log([1.3])}


def amortizer_forward(model, P, params=None):
    """0298's shared substrate: one network actualised only by each patient's private traces."""
    params = model["params"] if params is None else params
    p = {k: V(v, req=True) for k, v in params.items()}
    x = V(P["x"])
    ld1 = x @ p["v1"] + p["c1"]
    d1 = vexp(ld1)
    b1, h1 = observe(P, 0, d1)
    f1 = vcat([x, b1 / B_SCALE, h1 / H_SCALE, ld1.col()])
    ld2 = f1 @ p["v2"] + p["c2"]
    d2 = vexp(ld2)
    b2, h2 = observe(P, 1, d2)
    hid = vtanh(vcat([f1, b2 / B_SCALE, h2 / H_SCALE, ld2.col()]) @ p["U1"] + p["u1"])
    df = DOSE_CAP * vtanh(vexp((hid @ p["U2"] + p["u2"])[:, 0]) / DOSE_CAP)
    return {"doses": [d1, d2, df], "hidden": hid}, p


def init_kulliyat_only(rng, in_dim=N_SIGNS, hidden=16):
    return {"K1": rng.normal(0.0, 0.3, (in_dim, hidden)), "k1": np.zeros(hidden),
            "K2": rng.normal(0.0, 0.1, (hidden, 1)), "k2": np.log([1.3])}


def kulliyat_only_forward(model, P, params=None):
    """Universals alone: a dose deduced from signs, no probe, no particular."""
    params = model["params"] if params is None else params
    p = {k: V(v, req=True) for k, v in params.items()}
    hid = vtanh(V(P["x"]) @ p["K1"] + p["k1"])
    df = DOSE_CAP * vtanh(vexp((hid @ p["K2"] + p["k2"])[:, 0]) / DOSE_CAP)
    return {"doses": [df], "hidden": hid}, p


# ----------------------------------------------------------------------------- registries and interface
TASK_TYPES = ["episodic_control"]
FORWARD = {"titrator": titrator_forward, "amortizer": amortizer_forward, "kulliyat_only": kulliyat_only_forward}
INIT = {"titrator": init_titrator, "amortizer": init_amortizer, "kulliyat_only": init_kulliyat_only}
MODULES = {
    "kulliyat": (["W1", "b1", "W2", "b2"], "covariate network emitting a per-case Gaussian prior (mean, precision)", False),
    "probe": (["probe"], "bounded first-action magnitude, at most half the prior's plug-in optimum", True),
    "watch": (["tmpl", "watch"], "non-negative temporal pooling over a delayed-response window with learned onset template", True),
    "particular": (["log_sb", "log_sh"], "closed-form conjugate posterior update (precision-weighted shrinkage)", True),
    "escalation": (["esc"], "evidence-gated one-sided step toward the current plug-in optimum", True),
    "final": (["fin"], "scalar calibration of the final plug-in action", False)}
KNOCKOUTS = {"kulliyat": "mean", "watch": "zero", "particular": "mean", "escalation": "identity", "final": "identity"}
MUTANTS = {"sign_flip": {"tensor": None, "detect": "C3"}, "zero_lr": {"tensor": None, "detect": "C3"},
           "zero_grad_prior": {"tensor": "W1", "detect": "C1"}, "zero_grad_watch": {"tensor": "watch", "detect": "C1"}}


def build_model(in_dim=N_SIGNS, out_dim=1, task_type="episodic_control", rng=None, **cfg):
    if task_type not in TASK_TYPES or out_dim != 1:
        raise ValueError("only episodic_control with a scalar action is supported")
    kind = cfg.get("kind", "titrator")
    rng = np.random.default_rng(0) if rng is None else rng
    return {"kind": kind, "params": INIT[kind](rng, in_dim), "ko": {}, "variant": cfg.get("variant"), "stats": {}}


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


def modules(model):
    if model["kind"] != "titrator":
        return {"all": {"params": list(model["params"]), "role": "baseline network", "signature": False}}
    return {k: {"params": v[0], "role": v[1], "signature": v[2]} for k, v in MODULES.items()}


def knockout(model, name, mode=None):
    mode = mode or KNOCKOUTS[name]
    ko_model = {**model, "ko": dict(model["ko"])}
    ko_model["ko"][name] = model["stats"]["kulliyat_mean"] if name == "kulliyat" else mode
    return ko_model


def loss_value(model, P, params=None):
    out, _ = FORWARD[model["kind"]](model, P, params)
    return float(tadbir_cost(P, out["doses"])[0].v.mean())


def loss_and_grads(model, P):
    out, p = FORWARD[model["kind"]](model, P)
    cost = tadbir_cost(P, out["doses"])[0]
    loss = vsum(cost) / float(cost.v.size)
    backward(loss)
    grads = {k: (np.zeros_like(v.v) if v.g is None else v.g.reshape(v.v.shape)) for k, v in p.items()}
    t = MUTANTS.get(ACTIVE_MUTANT, {}).get("tensor")
    if t in grads and model["kind"] == "titrator":
        grads[t] = np.zeros_like(grads[t])
    return float(loss.v), grads


def predict(model, P):
    return FORWARD[model["kind"]](model, P)[0]["doses"][-1].v.copy()


def hidden_states(model, P):
    out = FORWARD[model["kind"]](model, P)[0]
    hs = {"hidden": out["hidden"].v, "doses": np.stack([d.v for d in out["doses"]], axis=1)}
    if model["kind"] == "titrator":
        hs.update({"gate": out["gate"].v, "post_beta": out["post2"][0].v, "post_gamma": out["post2"][1].v})
    return hs


def env_reset(rng, regime="id"):
    return {"P": make_patients(rng, 1, regime), "doses": []}


def env_step(state, action, rng=None):
    """Two probes return (benefit, side-effect) traces; the third action is final and returns -cost."""
    d = V(np.array([float(action)]))
    state["doses"].append(d)
    if len(state["doses"]) < 3:
        b, h = observe(state["P"], len(state["doses"]) - 1, d)
        return state, {"benefit": b.v[0], "harm": h.v[0]}, 0.0, False
    return state, None, -float(tadbir_cost(state["P"], state["doses"])[0].v[0]), True


# ----------------------------------------------------------------------------- training
class NonFinite(Exception):
    pass


def evaluate(model, P):
    out, _ = FORWARD[model["kind"]](model, P)
    cost, regret, harm = tadbir_cost(P, out["doses"])
    df = out["doses"][-1].v
    if not (np.all(np.isfinite(cost.v)) and np.all(np.isfinite(df))):
        raise NonFinite(model["kind"])
    return {"cost": float(cost.v.mean()), "regret": float(regret.v.mean()), "harm": float(np.mean(harm.v)),
            "log_err": float(np.median(np.abs(np.log(df / P["dref"]))))}


def fit(model, data, budget, rng, lr=0.01, batch=200, clip=5.0, eval_every=100):
    """Adam on minibatches of patients; model selection on the validation split only."""
    train, val = data["train"], data["val"]
    opt = Adam(model["params"], lr=0.0 if ACTIVE_MUTANT == "zero_lr" else lr)
    sign = -1.0 if ACTIVE_MUTANT == "sign_flip" else 1.0
    n, hist, best = train["x"].shape[0], [], (np.inf, None)
    for step in range(budget):
        loss, grads = loss_and_grads(model, take(train, rng.choice(n, batch, replace=False)))
        if not np.isfinite(loss):
            raise NonFinite(model["kind"])
        clip_global_norm(grads, clip)
        opt.step(model["params"], grads, sign)
        hist.append(loss)
        if (step + 1) % eval_every == 0 or step == budget - 1:
            v = evaluate(model, val)["cost"]
            if v < best[0]:
                best = (v, {k: a.copy() for k, a in model["params"].items()})
    model["params"], model["history"] = best[1], hist
    if model["kind"] == "titrator":
        prior = kulliyat({k: V(a) for k, a in model["params"].items()}, train, {})[0]
        model["stats"]["kulliyat_mean"] = [float(q.v.mean()) for q in prior]
    return model


def trivial_dose(train):
    grid = np.exp(np.linspace(np.log(0.05), np.log(DOSE_CAP), 240))
    return float(grid[int(np.argmin([trivial_cost(train, c) for c in grid]))])


def trivial_cost(P, c):
    return float(tadbir_cost(P, [V(np.full(P["x"].shape[0], c))])[0].v.mean())


def child(rng):
    return np.random.default_rng(int(rng.integers(1 << 62)))


# ----------------------------------------------------------------------------- correctness tests
def gradcheck_suite(data, rng, steps=60, kinds=tuple(FORWARD), tol=1e-5):
    batch = take(data["train"], np.arange(12))
    worst, checked, total, ok = 0.0, 0, 0, True
    for kind in kinds:
        model = build_model(rng=child(rng), kind=kind)
        for when in ("init", "after_training_steps"):
            if when != "init":
                fit(model, data, steps, child(rng), eval_every=steps)
            _, g = loss_and_grads(model, batch)
            errs = finite_difference_check(lambda prm: loss_value(model, batch, prm), model["params"], g, rng)
            worst = max([worst] + list(errs.values()))
            checked += sum(e <= tol for e in errs.values())
            total += len(errs)
            ok &= all(e <= tol for e in errs.values())
    return ok, {"tensors_checked": checked, "tensors_total": total, "max_rel_error": worst,
                "checked_at": ["init", "after_training_steps"], "passed": bool(ok)}


def determinism_check(data, seed):
    runs = []
    for _ in range(2):
        m = build_model(rng=np.random.default_rng(seed), kind="titrator")
        fit(m, data, 40, np.random.default_rng(seed + 1), eval_every=40)
        runs.append((np.array(m["history"]), predict(m, data["val"]), m))
    same = np.array_equal(runs[0][0], runs[1][0]) and np.array_equal(runs[0][1], runs[1][1])
    hs = hidden_states(runs[0][2], data["val"])
    finite = all(np.all(np.isfinite(a)) for a in list(runs[0][2]["params"].values()) + list(hs.values()))
    return bool(same and finite), f"identical histories and outputs: {same}; all finite: {finite}"


def learning_check(model, triv_id, id_cost):
    h = np.array(model["history"])
    drop = 1.0 - h[-20:].mean() / h[:20].mean()
    th = MIND_CARD["thresholds"]
    ok = drop >= th["loss_drop_fraction"] and id_cost <= (1 - th["margin_over_trivial"]) * triv_id
    return bool(ok), f"loss drop {drop:.3f}; id cost {id_cost:.4f} vs trivial {triv_id:.4f}"


def shuffled_control(data, rng, budget, triv_id):
    """Retrain with targets permuted across patients; held-out cost must stay in the trivial band."""
    sh = {}
    for split in ("train", "val"):
        P, perm = dict(data[split]), rng.permutation(data[split]["x"].shape[0])
        for k in TARGET_KEYS:
            P[k] = P[k][perm]
        sh[split] = P
    m = fit(build_model(rng=child(rng), kind="titrator"), sh, budget, child(rng))
    c = evaluate(m, data["id"])["cost"]
    ok = c >= (1 - MIND_CARD["thresholds"]["shuffled_band"]) * triv_id
    return bool(ok), f"shuffled-trained id cost {c:.4f} vs trivial {triv_id:.4f}"


def mutation_suite(data, rng, steps, triv_id):
    global ACTIVE_MUTANT
    prev, found = ACTIVE_MUTANT, {}
    for name in MUTANTS:
        ACTIVE_MUTANT = name
        try:
            c1 = gradcheck_suite(data, child(rng), kinds=("titrator",), steps=20)[0]
            m = fit(build_model(rng=child(rng), kind="titrator"), data, steps, child(rng))
            c3 = learning_check(m, triv_id, evaluate(m, data["id"])["cost"])[0]
            found[name] = (not c1) or (not c3)
        except NonFinite:
            found[name] = True
        finally:
            ACTIVE_MUTANT = prev
    return found


def random_titrator(rng, variant=None, scale=1.0):
    m = build_model(rng=child(rng), kind="titrator", variant=variant)
    m["params"] = {k: v + rng.normal(0.0, scale, v.shape) for k, v in m["params"].items()}
    return m


def prop_escalation_only(P, rng, variant=None, trials=40, ascent=25):
    """C6.1: the second dose never falls below the first (search: random draws, then ascent on d1 - d2)."""
    worst, m = -np.inf, None
    for _ in range(trials):
        m = random_titrator(rng, variant)
        out, _ = titrator_forward(m, P)
        worst = max(worst, float(np.max(out["doses"][0].v - out["doses"][1].v)))
    for _ in range(ascent):
        out, p = titrator_forward(m, P)
        backward(vsum(out["doses"][0] - out["doses"][1]))
        for k in m["params"]:
            if p[k].g is not None:
                m["params"][k] = m["params"][k] + 0.05 * np.sign(p[k].g)
        worst = max(worst, float(np.max(out["doses"][0].v - out["doses"][1].v)))
    return worst


def prop_convexity(P, rng, variant=None, trials=40):
    """C6.2: the stage-1 posterior lies between the universal prior and the evidence-only estimate."""
    worst = -np.inf
    for _ in range(trials):
        out, _ = titrator_forward(random_titrator(rng, variant), P)
        mb, est, bh = out["prior"][0].v, out["ev1"][1].v / out["ev1"][0].v, out["post1"][0].v
        lo, hi = np.minimum(mb, est), np.maximum(mb, est)
        worst = max(worst, float(np.max(np.maximum(bh - hi, lo - bh) / (1.0 + np.abs(bh)))))
    return worst


def prop_watching_informs(P, rng, variant=None, trials=40):
    """C6.3: adding a watched day never lowers the evidence precision on either coefficient."""
    worst = -np.inf
    for _ in range(trials):
        m = random_titrator(rng, variant)
        p = {k: V(v) for k, v in m["params"].items()}
        om = watch_weights(p, {}, variant)
        d = V(np.abs(rng.normal(1.0, 0.5, P["x"].shape[0])) + 0.05)
        b, h = observe(P, 0, d)
        m_t, s2 = vsoftplus(p["tmpl"]), vexp(2.0 * p["log_sb"])
        full = evidence(d, b, h, m_t, om, s2, s2)
        part = evidence(d, b, h, m_t, om * V(np.array([1.0, 1.0, 0.0])), s2, s2)
        worst = max(worst, float(np.max(part[0].v - full[0].v)), float(np.max(part[2].v - full[2].v)))
    return worst


def prop_overdose_asymmetry(P, rng, harm_variant=None, trials=40):
    """C6.4: on linear responders, overshooting by a factor costs at least as much as undershooting."""
    worst = -np.inf
    for _ in range(trials):
        dl = rng.uniform(0.05, 0.95, P["x"].shape[0])
        over = tadbir_cost(P, [V(P["dref"] * (1 + dl))], harm_variant)[0].v
        under = tadbir_cost(P, [V(P["dref"] * (1 - dl))], harm_variant)[0].v
        worst = max(worst, float(np.max(under - over)))
    return worst


def definition_checks(model, data):
    """C6.5/C6.6 labelled definition checks (not evidence): mildness bound and env/batch agreement."""
    P = take(data["id"], np.arange(4))
    out, _ = titrator_forward(model, P)
    mild = bool(np.all(out["doses"][0].v <= 0.5 * out["d0"].v + 1e-12))
    agree = True
    for i in range(4):
        st = {"P": take(P, [i]), "doses": []}
        for s, d in enumerate(out["doses"]):
            st, obs, r, done = env_step(st, d.v[i])
            if s < 2:
                b, _ = observe(P, s, out["doses"][s])
                agree &= np.allclose(obs["benefit"], b.v[i], atol=1e-12)
        agree &= abs(-r - tadbir_cost(P, out["doses"])[0].v[i]) < 1e-10 and done
    return mild, bool(agree)


def split_integrity(datasets):
    seen = {}
    for tag, data in datasets:
        for split, P in data.items():
            for row in np.round(P["x"], 9):
                h = hashlib.sha1(row.tobytes()).hexdigest()
                if seen.setdefault(h, (tag, split)) != (tag, split) and seen[h][0] == tag:
                    return False
    return True


def run_properties(data, rng, trained):
    P = take(data["id"], np.arange(64))
    Pm = {k: np.concatenate([P[k], data["blind"][k][:64]]) for k in P}
    tests = [("C6.1", "escalation_only", lambda v: prop_escalation_only(Pm, rng, v), "tanh_step"),
             ("C6.2", "posterior_between_prior_and_evidence", lambda v: prop_convexity(Pm, rng, v), "overconfident"),
             ("C6.3", "watching_never_loses_precision", lambda v: prop_watching_informs(Pm, rng, v), "signed_watch"),
             ("C6.4", "overdose_costs_at_least_underdose", lambda v: prop_overdose_asymmetry(P, rng, v), "mirrored")]
    rows = []
    for cid, name, fn, bad in tests:
        good_w, bad_w = fn(None), fn(bad)
        rows.append((cid, name, good_w <= 1e-10 and bad_w > 1e-6,
                     f"max violation {good_w:.1e}; negative control '{bad}' {bad_w:.1e}"))
    mild, agree = definition_checks(trained, data)
    rows.append(("C6.5", "definition: first dose <= half the prior optimum", mild, "definition check, not evidence"))
    rows.append(("C6.6", "definition: env API equals batch rollout", agree, "definition check, not evidence"))
    return rows


# ----------------------------------------------------------------------------- hypotheses and protocol
def train_seed(seed, budget, rng):
    data = make_data(seed)
    c = trivial_dose(data["train"])
    res, models = {"trivial": {s: trivial_cost(data[s], c) for s in ("id", "shift", "blind")}}, {}
    for kind in FORWARD:
        models[kind] = fit(build_model(rng=child(rng), kind=kind), data, budget, child(rng))
        res[kind] = {s: evaluate(models[kind], data[s]) for s in ("id", "shift", "blind")}
    res["ko"] = {k: evaluate(knockout(models["titrator"], k), data["id"])["cost"] for k in KNOCKOUTS}
    res["c3"] = learning_check(models["titrator"], res["trivial"]["id"], res["titrator"]["id"]["cost"])
    res["n_params"] = {k: n_params(m) for k, m in models.items()}
    return data, models, res


def hypothesis_table(results, rng, evaluated):
    def c(r, k, s):
        return r[k][s]["cost"]
    diffs = {"H-SIG": [c(r, "titrator", "shift") - c(r, "amortizer", "shift") for r in results],
             "H-NEC": [r["ko"]["particular"] - r["ko"]["kulliyat"] for r in results],
             "H-BLIND": [(c(r, "titrator", "blind") - c(r, "amortizer", "blind"))
                         - (c(r, "titrator", "id") - c(r, "amortizer", "id")) for r in results],
             "H-RIVAL": [c(r, "titrator", "id") - c(r, "kulliyat_only", "id") for r in results]}
    rows = []
    for h in MIND_CARD["hypotheses"]:
        mean, ci = paired_bootstrap(diffs[h["id"]], rng)
        rows.append({"id": h["id"], "metric": h["metric"], "mean_diff": mean, "ci95": ci, "mesi": h["mesi"],
                     "n_seeds": len(results),
                     "verdict": verdict(mean, ci, h["mesi"], h["direction"]) if evaluated else "not evaluated"})
    return rows


def knockout_table(results, rng):
    rows = []
    for name in KNOCKOUTS:
        mean, ci = paired_bootstrap([r["ko"][name] - r["titrator"]["id"]["cost"] for r in results], rng)
        rows.append({"module": name, "signature": MODULES[name][2], "metric_change": mean, "ci95": ci})
    return rows


def print_report(rep, results):
    print(f"=== VERIFIED REPORT · chapter {CHAPTER:04d} ===")
    print(f"environment: python {rep['environment']['python']}, numpy {rep['environment']['numpy']}")
    print(f"file: {rep['file']} · card revision {rep['card_revision']} · seeds {rep['seeds']} · runtime {rep['runtime_s']:.1f} s")
    print("n_params: " + ", ".join(f"{k} {v}" for k, v in rep["n_params_by_model"].items()))
    g = rep["gradcheck"]
    print(f"gradcheck: {g['tensors_checked']}/{g['tensors_total']} tensor checks, max rel error {g['max_rel_error']:.2e}, "
          f"at {g['checked_at']}, passed {g['passed']}")
    print("correctness:")
    for c in rep["correctness"]:
        print(f"  {c['id']:<5} {'PASS' if c['passed'] else 'FAIL'}  {c['name']}: {c['detail']}")
    mu = rep["mutants"]
    print(f"mutation score: {mu['detected']}/{mu['total']} ({mu['score']:.2f}) {mu['detail']}")
    print("descriptive (mean over seeds) cost | regret | harm | median |log dose error|:")
    for k in ("titrator", "amortizer", "kulliyat_only"):
        for s in ("id", "shift", "blind"):
            v = {m: np.mean([r[k][s][m] for r in results]) for m in ("cost", "regret", "harm", "log_err")}
            print(f"  {k:<14}{s:<6} {v['cost']:.4f} | {v['regret']:.4f} | {v['harm']:.4f} | {v['log_err']:.3f}")
    print("  trivial constant dose: " + ", ".join(
        f"{s} {np.mean([r['trivial'][s] for r in results]):.4f}" for s in ("id", "shift", "blind")))
    print("hypotheses (paired per-seed differences, 95% bootstrap CI):")
    for h in rep["hypotheses"]:
        print(f"  {h['id']:<8} {h['mean_diff']:+.4f} CI [{h['ci95'][0]:+.4f}, {h['ci95'][1]:+.4f}] "
              f"mesi {h['mesi']} n={h['n_seeds']} -> {h['verdict']}")
    print("knockouts (id cost change vs full titrator):")
    for k in rep["knockouts"]:
        print(f"  {k['module']:<11} signature={str(k['signature']):<5} {k['metric_change']:+.4f} "
              f"CI [{k['ci95'][0]:+.4f}, {k['ci95'][1]:+.4f}]")
    print(f"task types: {rep['task_types']} · exit code {rep['exit_code']}")
    print("=== END REPORT ===")


def run_protocol(args):
    t0, quick = time.time(), args.quick
    seeds = [args.seed + i for i in range(1 if quick else args.seeds)]
    budget, limit = (130 if quick else 700), (20.0 if quick else 180.0)
    rng = np.random.default_rng(np.random.SeedSequence([CHAPTER, args.seed, 7]))
    print(f"seeds: {seeds} (mutant: {ACTIVE_MUTANT})")
    data0 = make_data(seeds[0])
    ok, gc = gradcheck_suite(data0, child(rng), steps=50 if quick else 60)
    corr = [("C1", "gradient_check", ok, f"{gc['tensors_checked']}/{gc['tensors_total']} tensor checks, "
             f"max rel err {gc['max_rel_error']:.2e}")]
    corr.append(("C2", "determinism_and_finiteness") + determinism_check(data0, seeds[0]))
    results, datasets, models0 = [], [], None
    for s in seeds:
        data, models, res = train_seed(s, budget, child(rng))
        results.append(res)
        datasets.append((s, data))
        models0 = models0 or models
    corr.append(("C3", "learning", all(r["c3"][0] for r in results),
                 "; ".join(f"seed {s}: {r['c3'][1]}" for s, r in zip(seeds, results))))
    corr.append(("C4", "shuffled_label_control") + shuffled_control(datasets[0][1], child(rng), budget,
                                                                    results[0]["trivial"]["id"]))
    found = mutation_suite(datasets[0][1], child(rng), 110 if quick else 150, results[0]["trivial"]["id"])
    corr.append(("C5", "mutant_detection", all(found.values()), ", ".join(f"{k}:{v}" for k, v in found.items())))
    corr.extend(run_properties(datasets[0][1], child(rng), models0["titrator"]))
    corr.append(("C7", "split_integrity", split_integrity(datasets), "no example shared across splits of a seed"))
    runtime = time.time() - t0
    corr.append(("C8", "budget", runtime <= limit, f"{runtime:.1f} s of {limit:.0f} s"))
    failed = [c for c in corr if not c[2]]
    code = 0 if not failed else (3 if [c[0] for c in failed] == ["C8"] else 1)
    rep = {"schema_version": "1.0", "chapter": CHAPTER, "file": os.path.basename(__file__),
           "card_revision": MIND_CARD["card_revision"],
           "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
           "seeds": seeds, "runtime_s": runtime, "n_params": results[0]["n_params"]["titrator"],
           "n_params_by_model": results[0]["n_params"], "gradcheck": gc,
           "correctness": [{"id": c[0], "name": c[1], "passed": bool(c[2]), "detail": c[3]} for c in corr],
           "mutants": {"detected": int(sum(found.values())), "total": len(found),
                       "score": sum(found.values()) / len(found), "detail": ", ".join(MUTANTS)},
           "hypotheses": hypothesis_table(results, child(rng), evaluated=not quick),
           "knockouts": knockout_table(results, child(rng)), "task_types": TASK_TYPES, "exit_code": code,
           "descriptive": [{k: r[k] for k in ("titrator", "amortizer", "kulliyat_only", "trivial", "ko")} for r in results]}
    print_report(rep, results)
    if args.json:
        write_report(rep, args.json)
    return code


def real_data_bridge(path, rng):
    """Optional: fit the universal-prior head to a local CSV (last column = per-case target). Research use only."""
    raw = np.genfromtxt(path, delimiter=",", skip_header=1)
    raw = raw[np.all(np.isfinite(raw), axis=1)]
    X = (raw[:, :-1] - raw[:, :-1].mean(0)) / (raw[:, :-1].std(0) + 1e-9)
    y = (raw[:, -1] - raw[:, -1].mean()) / (raw[:, -1].std() + 1e-9)
    idx = rng.permutation(y.size)
    tr, te = idx[: int(0.8 * y.size)], idx[int(0.8 * y.size):]
    prm = init_kulliyat_only(rng, X.shape[1])
    prm["k2"] = np.zeros(1)
    opt = Adam(prm, lr=0.01)

    def pred(p, rows):
        return (vtanh(V(X[rows]) @ p["K1"] + p["k1"]) @ p["K2"] + p["k2"])[:, 0]
    for _ in range(300):
        p = {k: V(v, req=True) for k, v in prm.items()}
        r = pred(p, tr) - y[tr]
        backward(vsum(r * r) / float(tr.size))
        opt.step(prm, {k: p[k].g for k in prm})
    rmse = float(np.sqrt(np.mean((pred({k: V(v) for k, v in prm.items()}, te).v - y[te]) ** 2)))
    print(f"real-data bridge ({os.path.basename(path)}): universal-prior head, held-out RMSE {rmse:.3f} "
          f"(standardised; mean predictor ~1.0). Research use only, no dosing.")


def main(argv=None):
    global ACTIVE_MUTANT
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__), description=f"Chapter {CHAPTER:04d}: Tadbir titrator")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--card", action="store_true")
    ap.add_argument("--mutant", type=str, default=None)
    ap.add_argument("--data", type=str, default=None)
    args = ap.parse_args(argv)
    if args.card:
        print(json.dumps(MIND_CARD, indent=2))
        return 0
    if args.mutant is not None and args.mutant not in MUTANTS:
        print(f"usage: python3 {os.path.basename(__file__)} --mutant {{{','.join(MUTANTS)}}}")
        return 2
    if args.seeds < 1:
        return 2
    ACTIVE_MUTANT = args.mutant
    try:
        if args.data:
            real_data_bridge(args.data, np.random.default_rng(args.seed))
        else:
            print("real-data bridge: no --data PATH given; skipped")
        return run_protocol(args)
    except NonFinite as e:
        print(f"non-finite values in {e}")
        return 4


if __name__ == "__main__":
    sys.exit(main())

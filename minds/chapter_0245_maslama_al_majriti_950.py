#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 13 Minds 241 - 260 Available on Amazon https://www.amazon.com/dp/B0HLD1S6QT
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0245_maslama_al_majriti_950 - Maslama al-Majriti (c.950-1007)
#================================================================================  
"""
The Qisma unit: division in the measure where equality is true.

Thesis
    What a projection keeps (circles, incidence) is not what a division needs
    (equal parts): fix each share in the measure where equality holds, chosen by
    the question asked, then carry it into the working coordinate, keeping the
    inherited canon whole, its errors included.

Evidence and provenance (archetype_provenance = belief; no philosophy of mind
survives, the reading is reconstructed from his technical writings)
    D1 primary      Maslama's added chapter on the Planisphaerium treats the
                    division of the zodiac and of the horizon in the plane and
                    ends with "the division closest to correctness" (aqrab
                    al-qisma ila al-sihha), which passes through ascensions.
                    Paris BnF ar. 4821, ff. 76r-81r; Kunitzsch & Lorch 1994.
    D2 primary      One of his eleven notes proves that circles on the sphere
                    project to circles, a property Ptolemy used unproved.
                    Lorch 1995, Archive for History of Exact Sciences 49.
    D3 scholarship  First in al-Andalus to join the fara'id tradition (division
                    of inheritances) with the mathematical sciences.
                    Casulleras 2007, Biographical Encyclopedia of Astronomers.
    D4 deeds        With Ibn al-Saffar he re-expressed al-Khwarizmi's Sindhind
                    zij in the Hijra calendar and at Cordoba's coordinates.
                    Suter 1914; Neugebauer 1962.
    D5 scholarship  From one observation of Regulus (979, 135;40) he shifted
                    every star of the inherited catalogue by 13;10.
                    Casulleras 2007; star table dated 367/978 (PAL C.6.2).
    D6 scholarship  He kept the Indian-derived methods of the zij while later
                    Toledan astronomers rebuilt parameters from observation.
                    Casulleras 2007; Samso 2020, Inference 5(3).

Doctrine -> mechanism -> test
    D1 -> M1 plate_generator: canon measures (degrees, right-sphere and
          oblique-sphere rising times) plus a bounded monotone correction
          -> C6.1 monotone transport, C6.2 vernal anchoring -> H-SIG, H-NEC
    D1 -> M2 measure_gate: the query selects which measure is the true one
          -> C1 gradient coverage -> H-NEC
    D3 -> M3 divider: prescribed shares, 'awl renormalisation when shares
          exceed the whole, residue to a treasury class
          -> C6.3 partition of unity, C6.4 'awl scale invariance -> H-SIG
    D2 -> warrant only: the division is carried along the anchored circle
          map; C6.2 is a definition check of the anchor, not evidence
    D5, D6 -> M1 keeps the canon whole -> H-BLIND

Research question
    Does placing shares in a transported canonical measure, instead of learning
    boundary positions, generalise zero-shot to unseen share schemes and unseen
    observer latitudes, and what does it cost where the local world leaves the
    canon's mathematical horizon?

Closest prior art and the delta
    Implicit quantile networks (Dabney et al. 2018) learn a quantile function
    queried at arbitrary levels; circle flows (Rezende et al. 2020) and monotone
    networks (Wehenkel & Louppe 2019) learn monotone maps. The delta: shares are
    never learned as positions; a query-selected, canon-derived monotone circle
    measure is learned and every rational scheme, including oversubscribed ones
    renormalised by 'awl, is answered by comparison inside that measure.
    Overlap: Medium. Baselines in-file: an IQN-style boundary network (closest)
    and a size-matched per-slot MLP (standard alternative).

Blind spot
    The canon is carried whole. Where the local world departs from it in ways a
    bounded smooth correction cannot express (a high, jagged skyline at the
    observing site), a flexible learner with abundant local data should win.

Task (native, vector classification)
    lambda ~ U[0, 2pi): ecliptic degree read on the rete (noise 0.3 deg)
    phi: observer latitude; eps ~ U[23.40, 23.90] deg: obliquity of the era
    query t in {equal degrees, right-sphere rising time, site rising time}
    shares q: up to 6 claimants; if sum(q) > 1 all shares scale by 1/sum ('awl);
      if sum(q) < 1 the residue belongs to a treasury class
    truth: the claimant whose cumulative share interval contains the true
      measure of lambda (rising times include 34' refraction)
    splits: train/val/iid at phi in [0, 42] deg with train schemes; shifted at
      phi in [44, 56] deg with schemes of unseen denominators; blind: one
      valley site (37.9 deg) whose eastern skyline rises 14 deg

Limits
    A research prototype of one AGI-oriented mechanism, not an AGI. The world is
    synthetic; the canon supplied to the model is the textbook spherical formula,
    so advantages on the ordinary world partly reflect trusting a correct canon.
    Astrology appears only as history; nothing here predicts about people.
"""
MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 1,
    "revision_log": [],
    "generation": {"template_version": "codeguidelines v1.0 (15 Sep 2026)", "generator": "Claude",
                   "generator_version": "Claude Opus 5", "date": "2026-09-16"},
    "id": 245, "figure": "Maslama al-Majriti", "born": 950, "died": 1007,
    "civilization": "Andalusi (Umayyad Caliphate of Cordoba; Arabic-writing, ancestry unrecorded)",
    "provenance": "belief",
    "thesis": ("What a projection keeps is not what a division needs: fix each share in the measure where "
               "equality is true, chosen by the question, then carry it into the working coordinate, keeping "
               "the inherited canon whole, errors included."),
    "evidence": [
        {"id": "D1", "claim": "Added chapter on dividing the zodiac and the horizon in the plane; the division closest to correctness passes through ascensions",
         "basis": "primary", "source": "Maslama, Fasl laysa min al-kitab, Paris BnF ar. 4821 ff. 76r-81r; Kunitzsch & Lorch 1994, pp. 85-88"},
        {"id": "D2", "claim": "A note proves that circles on the sphere project to circles",
         "basis": "primary", "source": "Maslama, Ta'aliq on the Planisphaerium; Lorch 1995, AHES 49: 271-284"},
        {"id": "D3", "claim": "First in al-Andalus to join the fara'id tradition with the mathematical sciences",
         "basis": "scholarship", "source": "Casulleras 2007, Biographical Encyclopedia of Astronomers, pp. 727-728"},
        {"id": "D4", "claim": "Adapted al-Khwarizmi's Sindhind zij to the Hijra calendar and Cordoba's coordinates with Ibn al-Saffar",
         "basis": "deeds", "source": "Suter 1914; Neugebauer 1962; Casulleras 2007"},
        {"id": "D5", "claim": "One observation of Regulus (979, 135;40) shifted all catalogue stars by 13;10",
         "basis": "scholarship", "source": "Casulleras 2007; PAL C.6.2 star table dated 367/978"},
        {"id": "D6", "claim": "Retained Indian-derived zij methods; Toledan successors rebuilt parameters from observation",
         "basis": "scholarship", "source": "Casulleras 2007; Samso 2020, Inference 5(3)"},
        {"id": "D7", "claim": "The unifying reading (division in the true measure, canon carried whole) is interpretation",
         "basis": "speculation", "source": "this chapter"}],
    "research_question": {"category": "compositional and systematic generalization",
                          "question": ("Does placing shares in a transported canonical measure, instead of learning boundary "
                                       "positions, generalise zero-shot to unseen share schemes and latitudes, and what does it "
                                       "cost where the local world departs from the canon?")},
    "mechanism": {
        "name": "Qisma unit (division in the transported measure)",
        "family": "monotone circle flows with query-selected canonical measures and structured partition",
        "signature_modules": ["plate_generator", "measure_gate", "divider"],
        "closest_prior_art": ["Implicit Quantile Networks (Dabney et al. 2018)", "Normalizing flows on tori and spheres (Rezende et al. 2020)",
                              "Unconstrained monotonic neural networks (Wehenkel & Louppe 2019)"],
        "overlap": "Medium",
        "prior_art_queries": ["quantile function network arbitrary levels", "circle diffeomorphism normalizing flow",
                              "monotone neural network partition", "learned measure transport classification shares",
                              "apportionment neural network proportional renormalization"],
        "contribution_type": "mechanism",
        "delta": ("Shares are never learned as positions: a query-selected, canon-derived monotone circle measure is learned and "
                  "any rational scheme, including 'awl-renormalised oversubscription, is answered inside it without retraining.")},
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1 plate_generator", "property_test": "C6.1", "hypothesis": "H-SIG"},
        {"doctrine": "D1", "mechanism": "M2 measure_gate", "property_test": "C1", "hypothesis": "H-NEC"},
        {"doctrine": "D3", "mechanism": "M3 divider", "property_test": "C6.3, C6.4", "hypothesis": "H-SIG"},
        {"doctrine": "D2", "mechanism": "M1 anchored circle map", "property_test": "C6.2", "hypothesis": "none (definition check)"},
        {"doctrine": "D5, D6", "mechanism": "M1 canon carried whole", "property_test": "C6.1", "hypothesis": "H-BLIND"}],
    "hypotheses": [
        {"id": "H-SIG", "statement": "Division in the transported measure beats the IQN baseline on unseen schemes and latitudes",
         "metric": "accuracy", "split": "shifted", "comparison": "model - baseline", "direction": "greater", "mesi": 0.05, "seeds": 5},
        {"id": "H-NEC", "statement": "Removing the plate generator costs more than removing the residual head",
         "metric": "accuracy on shifted split", "comparison": "signature_knockout - matched_knockout", "direction": "less",
         "mesi": 0.10, "seeds": 5},
        {"id": "H-BLIND", "statement": "At a site whose skyline departs from the mathematical horizon, with abundant local data, the IQN baseline wins",
         "condition": "valley site 37.9 deg, eastern skyline up to 14 deg, 4000 local examples", "grounding": "D5, D6: the canon was carried whole",
         "metric": "accuracy", "comparison": "model - baseline", "direction": "less", "mesi": 0.01, "seeds": 5}],
    "thresholds": {"loss_drop_fraction": 0.30, "margin_over_trivial": 0.15, "shuffled_band": 0.05},
    "probe_predictions": [{"probe": "P3", "expected": "above baseline"}, {"probe": "P6", "expected": "above baseline"},
                          {"probe": "P7", "expected": "below baseline"}, {"probe": "P8", "expected": "equal to baseline"},
                          {"probe": "P9", "expected": "below baseline"}, {"probe": "P10", "expected": "above baseline"}],
    "dialectic_links": [{"chapter": 213, "relation": "teacher", "test": "H-SIG (his zij is the carried canon)"},
                        {"chapter": 130, "relation": "teacher", "test": "C6.2 (Planisphaerium)"},
                        {"chapter": 272, "relation": "successor", "test": "H-BLIND (re-observation over transport)"},
                        {"chapter": 378, "relation": "student", "test": "none (astrolabe texts via pseudo-Mashallah)"}],
    "corpus_neighbors": [
        {"chapter": 152, "similarity": None, "difference": "Hypatia certifies by the invariants a projection keeps; here the division needs the measure it does not keep"},
        {"chapter": 195, "similarity": None, "difference": "Yi Xing models changing rates by unequal-interval differences; here whole canonical measures are carried and selected per query"},
        {"chapter": 209, "similarity": None, "difference": "Habash inverts a projection to recover a trace; here nothing is inverted, shares are placed forward"},
        {"chapter": 237, "similarity": None, "difference": "al-Hamdani parts evidence mass under conservation; here prescribed shares are placed with 'awl"},
        {"chapter": 184, "similarity": None, "difference": "Anania recovers a whole from proportional takings; here the whole is given and the medium distorts"},
        {"chapter": 130, "similarity": None, "difference": "Ptolemy searches a frame where law is uniform; here measures are known canons chosen by the question"}],
    "barometer": {"cognitive_processing": ["P3 compositional split", "P10 sample efficiency"], "embodied_cognition": ["P9 perturbation robustness"],
                  "world_modeling": ["P7 regime change"], "consciousness": ["P8 calibration"], "language_understanding": [],
                  "emotional_intelligence": [], "creativity": [], "autonomy": ["P6 few-shot adaptation"]},
    "task_types": ["vector_classification"],
    "applications": [
        {"use": "latitude-general rising-time and crescent-visibility tables from one canon", "sector": "civic calendars", "dataset": "ICOP lunar crescent observations (Odeh 2004 compilation)"},
        {"use": "equal-energy time binning of solar generation across sites", "sector": "energy", "dataset": "NREL National Solar Radiation Database"},
        {"use": "gait-phase division in a phase measure shared across walkers", "sector": "robotics and rehabilitation", "dataset": "CMU Graphics Lab Motion Capture Database"}],
    "safety_notes": "Astrology is treated as history only; inheritance shares are an allocation structure, not legal advice; synthetic data only.",
    "similarity_note": "Nearest-neighbour code similarity not computed: corpus files were not available in this session.",
}

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

TAU = 2.0 * np.pi
DEG = np.pi / 180.0
EPS_REF = 23.5 * DEG
REFRACTION = 34.0 / 60.0 * DEG
M_SLOTS = 6                  # claimants; one more class is the treasury
K_HARM = 4
A_BOUND = 0.9                # sum of |correction amplitudes| stays below 1: monotone by construction
SHIFT_BOUND = 2.0 * DEG      # the canon may be nudged in latitude and obliquity, never replaced
P_FLOOR = 1e-6
JIMG = np.array([-1.0, 0.0, 1.0])
NATIVE_IN_DIM = 4 + M_SLOTS + 2
TASK_TYPES = ["vector_classification"]
BUDGET_FULL, BUDGET_QUICK = 180.0, 20.0
CLIP = 5.0

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


def adam_init(params):
    return {"t": 0, "m": {k: np.zeros_like(v) for k, v in params.items()}, "v": {k: np.zeros_like(v) for k, v in params.items()}}


def adam_step(params, grads, state, lr, b1=0.9, b2=0.999, eps=1e-8, sign=1.0):
    state["t"] += 1
    for k in params:
        state["m"][k] = b1 * state["m"][k] + (1 - b1) * grads[k]
        state["v"][k] = b2 * state["v"][k] + (1 - b2) * grads[k] ** 2
        mh = state["m"][k] / (1 - b1 ** state["t"])
        vh = state["v"][k] / (1 - b2 ** state["t"])
        params[k] -= sign * lr * mh / (np.sqrt(vh) + eps)


def clip_global(grads, max_norm):
    norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    if norm > max_norm:
        for k in grads:
            grads[k] = grads[k] * (max_norm / norm)
    return norm


def grad_check(loss_fn, params, grads, rng, eps=1e-6, n_entries=20, rel_tol=1e-5, abs_floor=1e-9):
    """Central differences on every tensor: all entries of small tensors, else 20 random plus the largest-gradient entry.
    Written reason for the floor: a central difference of a double-precision mean loss carries ~1e-10 of round-off,
    so relative error is judged where |analytic|+|numeric| exceeds 1e-4 (round-off then stays below 2e-6) and an
    absolute tolerance of 1e-9, five to ten times the round-off, is applied below that."""
    worst, checked = 0.0, 0
    for k, p in params.items():
        flat = p.reshape(-1)
        idx = np.arange(flat.size) if flat.size <= n_entries else np.unique(
            np.concatenate([rng.choice(flat.size, n_entries, replace=False), [int(np.argmax(np.abs(grads[k])))]]))
        for i in idx:
            old = flat[i]
            flat[i] = old + eps
            lp = loss_fn()
            flat[i] = old - eps
            lm = loss_fn()
            flat[i] = old
            num, ana = (lp - lm) / (2 * eps), grads[k].reshape(-1)[i]
            scale = abs(num) + abs(ana)
            if scale > 1e-4:
                worst = max(worst, abs(num - ana) / scale)
            elif abs(num - ana) > abs_floor:
                worst = max(worst, 1.0)
        checked += 1
    return {"tensors_checked": checked, "tensors_total": len(params), "max_rel_error": worst, "passed": worst <= rel_tol}


def paired_bootstrap(diffs, rng, n_boot=2000):
    d = np.asarray(diffs, dtype=float)
    means = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    return float(d.mean()), [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def verdict(mean, ci, mesi, direction):
    if direction == "greater":
        if ci[0] > 0 and mean >= mesi:
            return "supported"
        return "contradicted" if ci[1] < 0 else "inconclusive"
    if ci[1] < 0 and mean <= -mesi:
        return "supported"
    return "contradicted" if ci[0] > 0 else "inconclusive"


def write_report(path, report):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
# END STANDARD UTILITIES


# ----------------------------------------------------------------- data and tasks
TRAIN_SCHEMES = [[1/2, 1/2], [1/3, 1/3, 1/3], [1/4, 1/4, 1/4, 1/4], [1/6] * 6, [1/2, 1/4, 1/4], [2/3, 1/6, 1/6],
                 [1/2, 1/6, 1/3], [1/4, 1/2, 1/6], [2/3, 1/4], [1/2, 2/3], [1/4, 2/3, 1/6, 1/6], [1/3, 1/6],
                 [1/6, 1/6, 2/3, 1/4], [1/12, 1/2, 1/4], [1/3, 2/3, 1/6]]
SHIFT_SCHEMES = [[1/8, 2/3, 1/6, 1/6], [1/5] * 5, [2/7, 2/7, 3/7], [1/8, 7/8], [3/8, 3/8, 1/8], [1/9, 4/9, 4/9],
                 [2/5, 1/5, 1/8], [3/7, 1/7, 1/8, 1/8], [5/8, 1/5, 1/5], [1/10, 3/10, 2/5, 1/10]]


def scheme_bounds(schemes):
    """Cumulative boundaries [0, B1..B6, 1] after 'awl: oversubscribed shares scale by 1/sum, residue is treasury."""
    out = np.zeros((len(schemes), M_SLOTS + 2))
    for i, q in enumerate(schemes):
        q = np.array(q + [0.0] * (M_SLOTS - len(q)))
        q = q / max(1.0, q.sum())
        out[i, 1:M_SLOTS + 1] = np.cumsum(q)
        out[i, -1] = 1.0
    return out


def right_ascension(lam, eps):
    return np.mod(np.arctan2(np.cos(eps) * np.sin(lam), np.cos(lam)), TAU)


def rising_lst(lam, phi, eps, h0):
    d = np.arcsin(np.sin(eps) * np.sin(lam))
    cos_h = (np.sin(h0) - np.sin(phi) * np.sin(d)) / (np.cos(phi) * np.cos(d))
    return np.arctan2(np.cos(eps) * np.sin(lam), np.cos(lam)) - np.arccos(np.clip(cos_h, -1.0, 1.0)), d


def skyline(azimuth, amp):
    a = azimuth / DEG
    return amp * DEG * (np.exp(-((a - 64) / 16.0) ** 2) + 0.8 * np.exp(-((a - 88) / 12.8) ** 2) + 0.6 * np.exp(-((a - 110) / 16.0) ** 2))


def site_measure(lam, phi, eps, amp):
    """Rising time from the vernal point: refraction always, the skyline only in the blind-spot world."""
    def lst(l):
        _, d = rising_lst(l, phi, eps, -REFRACTION)
        az = np.arccos(np.clip(np.sin(d) / np.cos(phi), -1.0, 1.0))
        return rising_lst(l, phi, eps, skyline(az, amp) - REFRACTION)[0]
    return np.mod((lst(lam) - lst(np.zeros_like(lam))) / TAU, 1.0)


def true_measure(lam, phi, eps, t, amp=0.0):
    s = lam / TAU
    s = np.where(t == 1, right_ascension(lam, eps) / TAU, s)
    return np.where(t == 2, site_measure(lam, phi, eps, amp), s)


def make_split(rng, n, lat_deg, schemes, types=(0, 1, 2), amp=0.0):
    lam = rng.uniform(0.0, TAU, n)
    phi = rng.uniform(lat_deg[0], lat_deg[1], n) * DEG
    eps = rng.uniform(23.40, 23.90, n) * DEG
    t = rng.choice(np.array(types), n)
    B = scheme_bounds(schemes)[rng.integers(0, len(schemes), n)]
    s = true_measure(lam, phi, eps, t, amp)
    y = np.sum(s[:, None] >= B[:, 1:M_SLOTS + 1], axis=1)
    lam_obs = np.mod(lam + 0.3 * DEG * rng.standard_normal(n), TAU)
    X = np.column_stack([lam_obs, phi, eps, t.astype(float), B])
    return {"X": X, "y": y.astype(int)}


def make_task(rng, n_train=4000, n_eval=2000):
    return {"train": make_split(rng, n_train, (0, 42), TRAIN_SCHEMES), "val": make_split(rng, 1000, (0, 42), TRAIN_SCHEMES),
            "iid": make_split(rng, n_eval, (0, 42), TRAIN_SCHEMES), "shifted": make_split(rng, n_eval, (44, 56), SHIFT_SCHEMES)}


def make_blind_task(rng, n_train=4000, n_eval=2000):
    site, sch = (37.9, 37.9), TRAIN_SCHEMES + SHIFT_SCHEMES
    return {"train": make_split(rng, n_train, site, sch, (2,), 14.0), "val": make_split(rng, 1000, site, sch, (2,), 14.0),
            "site": make_split(rng, n_eval, site, sch, (2,), 14.0)}


def unpack(X):
    return X[:, 0], X[:, 1], X[:, 2], X[:, 3].astype(int), X[:, 4:]


def context(phi, eps):
    return np.column_stack([np.ones_like(phi), phi, phi * phi, (eps - EPS_REF) / (0.5 * DEG)])


def trivial_accuracy(split):
    widths = np.diff(split["X"][:, 4:], axis=1)
    return float(np.mean(np.argmax(widths, axis=1) == split["y"]))


# ----------------------------------------------------------------- model
def sigmoid(x):
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def divide(s, B, kappa):
    """Soft circular interval membership: periodic images make the shares a partition of unity on [0, 1)."""
    D = s[:, None, None] + JIMG[None, None, :] - B[:, :, None]
    sg = sigmoid(kappa * D)
    P = np.sum(sg[:, :-1] - sg[:, 1:], axis=2)
    sp = sg * (1.0 - sg)
    return P, (sp.sum(axis=2), (sp * D).sum(axis=2))


def divide_backward(gP, cache, kappa, want_bounds=False):
    a, b = cache
    gs = kappa * np.sum(gP * (a[:, :-1] - a[:, 1:]), axis=1)
    gk = float(np.sum(gP * (b[:, :-1] - b[:, 1:])))
    if not want_bounds:
        return gs, gk, None
    pad = np.pad(gP, ((0, 0), (1, 1)))
    return gs, gk, kappa * a * (pad[:, :-1] - pad[:, 1:])


def canon_measures(lam, phi, eps):
    """Degrees, right-sphere rising time, oblique-sphere rising time (as fractions of a turn) and their derivatives."""
    sl, cl, se, ce = np.sin(lam), np.cos(lam), np.sin(eps), np.cos(eps)
    y = ce * sl
    alpha = np.mod(np.arctan2(y, cl), TAU)
    sd = se * sl
    cd = np.sqrt(1.0 - sd * sd)
    td, tf = sd / cd, np.tan(phi)
    z = tf * td
    root = np.sqrt(1.0 - z * z)
    theta = np.arcsin(z)
    U = np.column_stack([lam / TAU, alpha / TAU, np.mod(alpha - theta, TAU) / TAU])
    da_de = -cl * se * sl / (cl * cl + y * y)
    dth_de = tf * (ce * sl / cd ** 3) / root
    dth_df = (1.0 + tf * tf) * td / root
    zero = np.zeros_like(lam)
    return U, np.column_stack([zero, da_de, da_de - dth_de]) / TAU, np.column_stack([zero, zero, -dth_df]) / TAU


def build_model(in_dim, out_dim, task_type, rng, **cfg):
    if task_type not in TASK_TYPES:
        raise ValueError("unsupported task type")
    native = in_dim == NATIVE_IN_DIM and out_dim == M_SLOTS + 1
    r_in = 6 if native else in_dim
    p = {"W": np.zeros((3 if native else 1, K_HARM, 4 if native else 1)), "log_kappa": np.array([math.log(4.0)]),
         "V1": 0.3 * rng.standard_normal((4, r_in)), "b1": np.zeros(4),
         "V2": 0.01 * rng.standard_normal((out_dim, 4)), "b2": np.zeros(out_dim)}
    if native:
        p.update({"eps_raw": np.zeros(1), "lat_raw": np.zeros(1), "G": np.zeros((3, 3))})
    else:
        p["P"] = rng.standard_normal((2, in_dim)) / math.sqrt(in_dim)
    return {"kind": "qisma", "native": native, "params": p, "ko": {}, "mutant": cfg.get("mutant"),
            "lr": cfg.get("lr", 0.03), "out_dim": out_dim}


def _transport(model, X):
    """Plate generator: canon measure plus bounded monotone harmonic correction, conditioned on context."""
    p = model["params"]
    if model["native"]:
        lam, phi, eps, t, B = unpack(X)
        C = context(phi, eps)
        U, dU_de, dU_df = canon_measures(lam, phi + SHIFT_BOUND * np.tanh(p["lat_raw"][0]), eps + SHIFT_BOUND * np.tanh(p["eps_raw"][0]))
        rin = np.column_stack([np.sin(lam), np.cos(lam), C])
    else:
        px, py = X @ p["P"][0], X @ p["P"][1]
        U = (np.mod(np.arctan2(py, px), TAU) / TAU)[:, None]
        C, t, rin = np.ones((X.shape[0], 1)), np.zeros(X.shape[0], int), X
        B = np.tile(np.linspace(0.0, 1.0, model["out_dim"] + 1), (X.shape[0], 1))
        dU_de = dU_df = None
    if "plate_generator" in model["ko"]:
        U = np.repeat(U[:, :1], U.shape[1], axis=1) if model["native"] else U
        return {"S": U, "dS": np.ones_like(U), "C": C, "t": t, "B": B, "rin": rin, "px": None if model["native"] else (px, py)}
    k = np.arange(1, K_HARM + 1)
    Z = np.einsum("tkd,nd->ntk", p["W"], C)
    A = (A_BOUND / K_HARM) * np.tanh(Z)
    ang = TAU * U[:, :, None] * k
    S = U + np.sum(A * np.sin(ang) / (TAU * k), axis=2)
    return {"S": S, "dS": 1.0 + np.sum(A * np.cos(ang), axis=2), "U": U, "Z": Z, "A": A, "ang": ang, "dU_de": dU_de,
            "dU_df": dU_df, "C": C, "t": t, "B": B, "rin": rin, "px": None if model["native"] else (px, py)}


def forward(model, X):
    p, ko = model["params"], model["ko"]
    tr = _transport(model, X)
    g = np.zeros_like(tr["S"]) if (not model["native"] or "measure_gate" in ko) else p["G"][tr["t"]]
    pi = softmax(g, axis=1)
    s = np.sum(pi * tr["S"], axis=1)
    kappa = math.exp(p["log_kappa"][0])
    P, dcache = divide(s, tr["B"], kappa)
    H = np.tanh(tr["rin"] @ p["V1"].T + p["b1"])
    R = np.zeros((X.shape[0], model["out_dim"])) if "residual_head" in ko else H @ p["V2"].T + p["b2"]
    logits = R if "divider" in ko else np.log(P + P_FLOOR) + R
    return logits, {"tr": tr, "pi": pi, "s": s, "P": P, "dcache": dcache, "H": H, "kappa": kappa}


def loss_and_grads(model, batch):
    if model["kind"] != "qisma":
        return BASELINE_LOSS[model["kind"]](model, batch)
    p, ko, X, y = model["params"], model["ko"], batch["X"], batch["y"]
    n = X.shape[0]
    logits, c = forward(model, X)
    loss = float(np.mean(logsumexp(logits, axis=1) - logits[np.arange(n), y]))
    gl = softmax(logits, axis=1)
    gl[np.arange(n), y] -= 1.0
    gl /= n
    gr = {k: np.zeros_like(v) for k, v in p.items()}
    if "residual_head" not in ko:
        gr["V2"], gr["b2"] = gl.T @ c["H"], gl.sum(axis=0)
        gpre = (gl @ p["V2"]) * (1.0 - c["H"] ** 2)
        gr["V1"], gr["b1"] = gpre.T @ c["tr"]["rin"], gpre.sum(axis=0)
    if "divider" in ko:
        return loss, gr
    gs, gk, _ = divide_backward(gl / (c["P"] + P_FLOOR), c["dcache"], c["kappa"])
    gr["log_kappa"][0] = gk if model["mutant"] == "kappa_chain_bug" else gk * c["kappa"]
    tr, pi = c["tr"], c["pi"]
    gS = gs[:, None] * pi
    if model["native"] and "measure_gate" not in ko and model["mutant"] != "drop_gate_grad":
        gpi = gs[:, None] * tr["S"]
        np.add.at(gr["G"], tr["t"], pi * (gpi - np.sum(pi * gpi, axis=1, keepdims=True)))
    if "plate_generator" in ko:
        return loss, gr
    k = np.arange(1, K_HARM + 1)
    gA = gS[:, :, None] * np.sin(tr["ang"]) / (TAU * k)
    gZ = gA * (A_BOUND / K_HARM) * (1.0 - (tr["A"] * K_HARM / A_BOUND) ** 2)
    gr["W"] = np.einsum("ntk,nd->tkd", gZ, tr["C"])
    gU = gS * tr["dS"]
    if model["native"]:
        gr["eps_raw"][0] = np.sum(gU * tr["dU_de"]) * SHIFT_BOUND * (1.0 - np.tanh(p["eps_raw"][0]) ** 2)
        gr["lat_raw"][0] = np.sum(gU * tr["dU_df"]) * SHIFT_BOUND * (1.0 - np.tanh(p["lat_raw"][0]) ** 2)
    else:
        px, py = tr["px"]
        gth = gU[:, 0] / TAU / (px * px + py * py + 1e-12)
        gr["P"] = np.vstack([(-gth * py) @ X, (gth * px) @ X])
    return loss, gr


def predict(model, X):
    return np.argmax(forward(model, X)[0] if model["kind"] == "qisma" else BASELINE_FORWARD[model["kind"]](model, X)[0], axis=1)


def hidden_states(model, X):
    _, c = forward(model, X)
    return {"measures": c["tr"]["S"], "gate": c["pi"], "measure": c["s"], "shares": c["P"]}


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


# ----------------------------------------------------------------- baselines
def build_iqn(rng, lr=0.03):
    return {"kind": "iqn", "params": {"W1": 0.3 * rng.standard_normal((7, 15)), "b1": np.zeros(7), "w2": 0.1 * rng.standard_normal(7),
                                      "c2": np.zeros(1), "log_kappa": np.array([math.log(4.0)])}, "ko": {}, "mutant": None, "lr": lr}


def iqn_forward(model, X):
    """Boundary positions learned as a quantile function of the share level (cosine level embedding, as in IQN)."""
    p = model["params"]
    lam, phi, eps, t, B = unpack(X)
    F = np.cos(np.pi * B[:, :, None] * np.arange(1, 9))
    ctx = np.column_stack([context(phi, eps), np.eye(3)[t]])
    Xb = np.concatenate([F, np.repeat(ctx[:, None, :], B.shape[1], axis=1)], axis=2)
    H = np.tanh(Xb @ p["W1"].T + p["b1"])
    o = H @ p["w2"] + p["c2"][0]
    anchor = np.sin(np.pi * B)
    beta = B + 0.25 * np.tanh(o) * anchor
    kappa = math.exp(p["log_kappa"][0])
    P, dcache = divide(lam / TAU, beta, kappa)
    Pc = np.maximum(P, 0.0)      # learned quantiles may cross; crossed intervals hold no mass
    return np.log(Pc + P_FLOOR), {"Xb": Xb, "H": H, "o": o, "anchor": anchor, "P": Pc, "open": P > 0, "dcache": dcache, "kappa": kappa}


def iqn_loss(model, batch):
    p, X, y = model["params"], batch["X"], batch["y"]
    n = X.shape[0]
    logits, c = iqn_forward(model, X)
    loss = float(np.mean(logsumexp(logits, axis=1) - logits[np.arange(n), y]))
    gl = softmax(logits, axis=1)
    gl[np.arange(n), y] -= 1.0
    _, gk, gbeta = divide_backward(gl / n / (c["P"] + P_FLOOR) * c["open"], c["dcache"], c["kappa"], want_bounds=True)
    go = gbeta * 0.25 * (1.0 - np.tanh(c["o"]) ** 2) * c["anchor"]
    gpre = go[:, :, None] * p["w2"] * (1.0 - c["H"] ** 2)
    return loss, {"W1": np.einsum("nbh,nbi->hi", gpre, c["Xb"]), "b1": gpre.sum(axis=(0, 1)),
                  "w2": np.einsum("nb,nbh->h", go, c["H"]), "c2": np.array([go.sum()]), "log_kappa": np.array([gk * c["kappa"]])}


def build_mlp(rng, lr=0.03):
    return {"kind": "mlp", "params": {"W1": 0.3 * rng.standard_normal((7, 15)), "b1": np.zeros(7), "w2": 0.1 * rng.standard_normal(7),
                                      "c2": np.zeros(1)}, "ko": {}, "mutant": None, "lr": lr}


def mlp_forward(model, X):
    """Size-matched standard alternative: score each claimant slot from the degree, context and its interval."""
    p = model["params"]
    lam, phi, eps, t, B = unpack(X)
    lo, hi = B[:, :-1], B[:, 1:]
    base = np.column_stack([np.sin(lam), np.cos(lam), context(phi, eps), np.eye(3)[t]])
    Xs = np.concatenate([np.repeat(base[:, None, :], lo.shape[1], axis=1),
                         np.stack([lo, hi, np.sin(TAU * lo), np.cos(TAU * lo), np.sin(TAU * hi), np.cos(TAU * hi)], axis=2)], axis=2)
    H = np.tanh(Xs @ p["W1"].T + p["b1"])
    return H @ p["w2"] + p["c2"][0] - 30.0 * (hi - lo <= 0), {"Xs": Xs, "H": H}


def mlp_loss(model, batch):
    p, X, y = model["params"], batch["X"], batch["y"]
    n = X.shape[0]
    logits, c = mlp_forward(model, X)
    loss = float(np.mean(logsumexp(logits, axis=1) - logits[np.arange(n), y]))
    gl = softmax(logits, axis=1)
    gl[np.arange(n), y] -= 1.0
    gl /= n
    gpre = gl[:, :, None] * p["w2"] * (1.0 - c["H"] ** 2)
    return loss, {"W1": np.einsum("nbh,nbi->hi", gpre, c["Xs"]), "b1": gpre.sum(axis=(0, 1)),
                  "w2": np.einsum("nb,nbh->h", gl, c["H"]), "c2": np.array([gl.sum()])}


BASELINE_LOSS = {"iqn": iqn_loss, "mlp": mlp_loss}
BASELINE_FORWARD = {"iqn": iqn_forward, "mlp": mlp_forward}


# ----------------------------------------------------------------- registries
def modules(model):
    return {"plate_generator": {"params": ["W", "eps_raw", "lat_raw"], "role": "context-conditioned monotone circle transport (canon plus bounded harmonics)", "signature": True},
            "measure_gate": {"params": ["G"], "role": "query-conditioned convex mixture over candidate measures", "signature": True},
            "divider": {"params": ["log_kappa"], "role": "soft circular partition at renormalised cumulative shares", "signature": True},
            "residual_head": {"params": ["V1", "b1", "V2", "b2"], "role": "residual logit MLP", "signature": False}}


def knockout(model, name, mode="identity"):
    m = {**model, "params": {k: v.copy() for k, v in model["params"].items()}, "ko": {**model["ko"], name: mode}}
    return m


MUTANTS = {"sign_flip": "optimizer ascends the loss", "zero_lr": "learning rate set to zero",
           "drop_gate_grad": "gradient of the measure gate zeroed", "kappa_chain_bug": "log-sharpness gradient misses its chain factor"}


# ----------------------------------------------------------------- training
def fit(model, data, budget, rng):
    """Adam under a fixed update budget; the checkpoint with the lowest validation loss is kept."""
    params = model["params"]
    state = adam_init(params)
    lr = 0.0 if model.get("mutant") == "zero_lr" else model["lr"]
    sign = -1.0 if model.get("mutant") == "sign_flip" else 1.0
    tr, n = data["train"], data["train"]["X"].shape[0]
    best, best_params, trace = float("inf"), None, []
    for step in range(budget):
        idx = rng.integers(0, n, 256)
        loss, grads = loss_and_grads(model, {"X": tr["X"][idx], "y": tr["y"][idx]})
        if not np.isfinite(loss):
            raise FloatingPointError("non-finite training loss")
        clip_global(grads, CLIP)
        adam_step(params, grads, state, lr, sign=sign)
        trace.append(loss)
        if (step + 1) % 100 == 0 or step + 1 == budget:
            vloss = loss_and_grads(model, data["val"])[0]
            if vloss < best:
                best, best_params = vloss, {k: v.copy() for k, v in params.items()}
    if best_params is not None and sign > 0:
        model["params"] = best_params
    return trace


def accuracy(model, split):
    return float(np.mean(predict(model, split["X"]) == split["y"]))


def full_loss(model, split):
    return loss_and_grads(model, split)[0]


BUILDERS = {"qisma": lambda rng, lr: build_model(NATIVE_IN_DIM, M_SLOTS + 1, "vector_classification", rng, lr=lr),
            "iqn": build_iqn, "mlp": build_mlp}


def trained(kind, data, steps, seed, lr, mutant=None):
    ss = np.random.SeedSequence([seed, sum(map(ord, kind))])
    init_rng, fit_rng = [np.random.default_rng(c) for c in ss.spawn(2)]
    model = BUILDERS[kind](init_rng, lr)
    model["mutant"] = mutant
    trace = fit(model, data, steps, fit_rng)
    return model, trace


# ----------------------------------------------------------------- tests: correctness
def test_gradients(data, steps, seed, mutant=None):
    rng = np.random.default_rng(seed + 11)
    batch = {"X": data["train"]["X"][:64].copy(), "y": data["train"]["y"][:64].copy()}
    batch["X"][:, 0] = np.clip(batch["X"][:, 0], 2 * DEG, TAU - 2 * DEG)   # keep away from the vernal wrap
    gX = np.random.default_rng(seed + 12).standard_normal((64, 5))
    gbatch = {"X": gX, "y": np.random.default_rng(seed + 13).integers(0, 3, 64)}
    results = []
    for label, kind, b in [("qisma", "qisma", batch), ("qisma_generic", "generic", gbatch), ("iqn", "iqn", batch), ("mlp", "mlp", batch)]:
        mrng = np.random.default_rng(seed + len(label))
        model = build_model(5, 3, "vector_classification", mrng) if kind == "generic" else BUILDERS[kind](mrng, 0.03)
        model["mutant"] = mutant if kind in ("qisma", "generic") else None
        for when in ("init", "after_training_steps"):
            if when == "after_training_steps":
                fit(model, {"train": b, "val": b}, max(50, steps), np.random.default_rng(seed + 7))
            loss, grads = loss_and_grads(model, b)
            results.append(grad_check(lambda: loss_and_grads(model, b)[0], model["params"], grads, rng))
    worst = max(r["max_rel_error"] for r in results)
    total = sum(r["tensors_total"] for r in results) // 2
    return {"tensors_checked": total, "tensors_total": total, "max_rel_error": worst,
            "checked_at": ["init", "after_training_steps"], "passed": all(r["passed"] for r in results)}


def learning_check(data, steps, seed, lr, mutant=None):
    model, trace = trained("qisma", data, steps, seed, lr, mutant)
    first, last = float(np.mean(trace[:10])), float(np.mean(trace[-10:]))
    drop = (first - last) / first
    margin = accuracy(model, data["iid"]) - trivial_accuracy(data["iid"])
    th = MIND_CARD["thresholds"]
    return {"passed": bool(drop >= th["loss_drop_fraction"] and margin >= th["margin_over_trivial"]), "drop": drop, "margin": margin}, model


def random_model(rng, scale):
    m = build_model(NATIVE_IN_DIM, M_SLOTS + 1, "vector_classification", rng)
    for k in ("W", "G", "V1", "V2"):
        m["params"][k] = scale * rng.standard_normal(m["params"][k].shape)
    m["params"]["eps_raw"][0], m["params"]["lat_raw"][0] = rng.uniform(-3.0, 3.0, 2)
    return m


def probe_inputs(rng, n, lam):
    X = make_split(rng, n, (0, 56), TRAIN_SCHEMES + SHIFT_SCHEMES)["X"]
    X[:, 0] = lam
    return X


def property_tests(seed):
    """C6: invariants on random parameters and inputs, each with a deliberately broken variant that must violate."""
    rng = np.random.default_rng(seed + 101)
    out = []
    lam = np.linspace(0.5 * DEG, TAU - 0.5 * DEG, 1440)
    def min_slope(model, unbounded=False):
        worst = np.inf
        for _ in range(12):
            X = probe_inputs(rng, lam.size, lam)
            X[:, 1:4] = X[0, 1:4]
            if unbounded:
                model["params"]["W"] = 40.0 * np.abs(rng.standard_normal(model["params"]["W"].shape))
            tr = _transport(model, X)
            if unbounded:
                k = np.arange(1, K_HARM + 1)
                A = 3.0 * np.tanh(tr["Z"])
                S = tr["U"] + np.sum(A * np.sin(tr["ang"]) / (TAU * k), axis=2)
            else:
                S = tr["S"]
            worst = min(worst, float(np.min(np.diff(S, axis=0))))
        return worst
    good = min(min_slope(random_model(rng, 30.0)) for _ in range(4))
    bad = min_slope(random_model(rng, 1.0), unbounded=True)
    out.append({"id": "C6.1", "name": "monotone_transport", "passed": bool(good > 0 and bad < 0), "detail": f"min step {good:.2e}; broken {bad:.2e}"})
    X0 = probe_inputs(rng, 400, np.zeros(400))
    anchored = max(float(np.max(np.abs(_transport(random_model(rng, 30.0), X0)["S"]))) for _ in range(6))
    tr = _transport(random_model(rng, 3.0), X0)
    broken = float(np.max(np.abs(tr["U"] + np.sum(tr["A"] * np.cos(tr["ang"]), axis=2))))
    out.append({"id": "C6.2", "name": "vernal_anchor (definition check)", "passed": bool(anchored < 1e-12 and broken > 1e-3),
                "detail": f"max |s(0)| {anchored:.1e}; cosine variant {broken:.2e}"})
    s = rng.uniform(0.0, 1.0, 5000)
    s[:50] = rng.uniform(0.0, 1e-3, 50)
    B = scheme_bounds(TRAIN_SCHEMES + SHIFT_SCHEMES)[rng.integers(0, 25, 5000)]
    worst_sum = max(float(np.max(np.abs(divide(s, B, kap)[0].sum(axis=1) - 1.0))) for kap in (60.0, 400.0))
    D = s[:, None] - B
    no_img = np.abs(np.sum(sigmoid(60.0 * D[:, :-1]) - sigmoid(60.0 * D[:, 1:]), axis=1) - 1.0).max()
    out.append({"id": "C6.3", "name": "partition_of_unity", "passed": bool(worst_sum < 1e-3 and no_img > 0.1),
                "detail": f"max |sum-1| {worst_sum:.1e}; without periodic images {no_img:.2f}"})
    over = [q for q in TRAIN_SCHEMES + SHIFT_SCHEMES if sum(q) > 1]
    viol, broken_viol = 0.0, 0.0
    for q in over:
        c = rng.uniform(1.0, 3.0)
        viol = max(viol, float(np.max(np.abs(scheme_bounds([q]) - scheme_bounds([[c * x for x in q]])))))
        clipped = lambda v: np.minimum(np.cumsum(v), 1.0)
        broken_viol = max(broken_viol, float(np.max(np.abs(clipped(q) - clipped([c * x for x in q])))))
    out.append({"id": "C6.4", "name": "awl_scale_invariance", "passed": bool(viol < 1e-12 and broken_viol > 1e-3),
                "detail": f"max boundary change {viol:.1e}; clipping variant {broken_viol:.2f}"})
    return out


def split_integrity(data):
    def hashes(X):
        return {hashlib.sha1(np.round(r, 12).tobytes()).hexdigest() for r in X}
    train = hashes(data["train"]["X"])
    clash = sum(len(train & hashes(data[k]["X"])) for k in data if k != "train")
    return {"id": "C7", "name": "split_integrity", "passed": clash == 0, "detail": f"{clash} shared rows"}


def determinism(data, seed, lr):
    runs = [trained("qisma", data, 60, seed, lr) for _ in range(2)]
    same = runs[0][1] == runs[1][1] and np.array_equal(predict(runs[0][0], data["iid"]["X"]), predict(runs[1][0], data["iid"]["X"]))
    finite = all(np.all(np.isfinite(v)) for v in runs[0][0]["params"].values())
    finite = finite and np.all(np.isfinite(forward(runs[0][0], data["iid"]["X"])[0]))
    return {"id": "C2", "name": "determinism_and_finiteness", "passed": bool(same and finite), "detail": f"identical={same} finite={finite}"}


def shuffled_control(data, steps, seed, lr):
    rng = np.random.default_rng(seed + 55)
    shuffled = {**data, "train": {"X": data["train"]["X"], "y": rng.permutation(data["train"]["y"])}}
    model, _ = trained("qisma", shuffled, steps, seed, lr)
    acc, triv = accuracy(model, data["iid"]), trivial_accuracy(data["iid"])
    ok = acc <= triv + MIND_CARD["thresholds"]["shuffled_band"]
    return {"id": "C4", "name": "shuffled_label_control", "passed": bool(ok), "detail": f"accuracy {acc:.3f} vs trivial {triv:.3f}"}


def mutant_audit(data, steps, seed, lr):
    healthy, _ = learning_check(data, steps, seed, lr)
    detected = 0
    for name in MUTANTS:
        g = test_gradients(data, 50, seed, mutant=name)
        lc, _ = learning_check(data, steps, seed, lr, mutant=name)
        detected += int((not g["passed"]) or (not lc["passed"]))
    return {"detected": detected, "total": len(MUTANTS), "score": detected / len(MUTANTS), "healthy_short_run_passes": healthy["passed"]}


# ----------------------------------------------------------------- tests: hypotheses
def tune_lr(kind, data, steps, seed):
    scores = {}
    for lr in (0.01, 0.03):
        model, _ = trained(kind, data, steps, seed, lr)
        scores[lr] = full_loss(model, data["val"])
    return min(scores, key=scores.get)


def run_seed(seed, steps, lrs):
    rng = np.random.default_rng(seed)
    data, blind = make_task(rng), make_blind_task(rng)
    row = {"seed": seed, "trivial_shifted": trivial_accuracy(data["shifted"]), "trivial_site": trivial_accuracy(blind["site"])}
    for kind in ("qisma", "iqn", "mlp"):
        model, _ = trained(kind, data, steps, seed, lrs[("main", kind)])
        row[kind] = {"iid": accuracy(model, data["iid"]), "shifted": accuracy(model, data["shifted"])}
        if kind == "qisma":
            row["knockouts"] = {name: accuracy(knockout(model, name), data["shifted"]) for name in modules(model)}
            row["gate"] = softmax(model["params"]["G"], axis=1).round(3).tolist()
    for kind in ("qisma", "iqn"):
        model, _ = trained(kind, blind, steps, seed, lrs[("blind", kind)])
        row[kind]["site"] = accuracy(model, blind["site"])
    return row


def hypothesis_tests(rows, seed):
    rng = np.random.default_rng(seed + 999)
    specs = {h["id"]: h for h in MIND_CARD["hypotheses"]}
    diffs = {"H-SIG": [r["qisma"]["shifted"] - r["iqn"]["shifted"] for r in rows],
             "H-NEC": [r["knockouts"]["plate_generator"] - r["knockouts"]["residual_head"] for r in rows],
             "H-BLIND": [r["qisma"]["site"] - r["iqn"]["site"] for r in rows]}
    out = []
    for hid, d in diffs.items():
        mean, ci = paired_bootstrap(d, rng)
        v = verdict(mean, ci, specs[hid]["mesi"], specs[hid]["direction"]) if len(d) >= 5 else "not evaluated"
        out.append({"id": hid, "metric": specs[hid]["metric"], "mean_diff": mean, "ci95": ci, "mesi": specs[hid]["mesi"],
                    "n_seeds": len(d), "verdict": v})
    kos = []
    for name, info in modules(None).items():
        mean, ci = paired_bootstrap([r["knockouts"][name] - r["qisma"]["shifted"] for r in rows], rng)
        kos.append({"module": name, "signature": info["signature"], "metric_change": mean, "ci95": ci})
    return out, kos


# ----------------------------------------------------------------- report
def print_report(rep, rows):
    print("=== VERIFIED REPORT · chapter 0245 ===")
    print(f"environment      python {rep['environment']['python']}, numpy {rep['environment']['numpy']}")
    print(f"seeds            {rep['seeds']}")
    print(f"runtime_s        {rep['runtime_s']:.1f}")
    print(f"n_params         qisma {rep['n_params']}, iqn {rep['baseline_params']['iqn']}, mlp {rep['baseline_params']['mlp']}")
    g = rep["gradcheck"]
    print(f"gradcheck        {g['tensors_checked']}/{g['tensors_total']} tensors per model, max rel error {g['max_rel_error']:.2e}, passed={g['passed']}")
    for c in rep["correctness"]:
        print(f"  {c['id']:<5} {c['name']:<34} {'pass' if c['passed'] else 'FAIL'}  {c['detail']}")
    mu = rep["mutants"]
    print(f"mutants          detected {mu['detected']}/{mu['total']} (score {mu['score']:.2f})")
    print("accuracy per seed (iid / shifted / site):")
    for r in rows:
        print(f"  seed {r['seed']:>6}: qisma {r['qisma']['iid']:.3f}/{r['qisma']['shifted']:.3f}/{r['qisma']['site']:.3f}  "
              f"iqn {r['iqn']['iid']:.3f}/{r['iqn']['shifted']:.3f}/{r['iqn']['site']:.3f}  mlp {r['mlp']['iid']:.3f}/{r['mlp']['shifted']:.3f}  "
              f"trivial {r['trivial_shifted']:.3f}/{r['trivial_site']:.3f}")
    for h in rep["hypotheses"]:
        print(f"  {h['id']:<8} mean {h['mean_diff']:+.4f}  ci95 [{h['ci95'][0]:+.4f}, {h['ci95'][1]:+.4f}]  mesi {h['mesi']}  -> {h['verdict']}")
    for k in rep["knockouts"]:
        print(f"  knockout {k['module']:<16} sig={str(k['signature']):<5} change {k['metric_change']:+.4f}  ci95 [{k['ci95'][0]:+.4f}, {k['ci95'][1]:+.4f}]")
    if rows:
        print(f"learned gate (rows: degrees, right sphere, site query) seed {rows[0]['seed']}: {rows[0]['gate']}")
    print(f"task types       {rep['task_types']}")
    print(f"exit code        {rep['exit_code']}")
    print("=== END REPORT ===")


def data_bridge(path, model):
    if not path:
        print("real-data bridge: no --data path given; skipped")
        return
    if not os.path.exists(path):
        print(f"real-data bridge: {path} not found; skipped")
        return
    raw = np.genfromtxt(path, delimiter=",", skip_header=1)
    if raw.ndim != 2 or raw.shape[1] != NATIVE_IN_DIM + 1:
        print("real-data bridge: expected columns lambda,phi,eps,query,B0..B7,label (radians); skipped")
        return
    split = {"X": raw[:, :-1], "y": raw[:, -1].astype(int)}
    print(f"real-data bridge: accuracy {accuracy(model, split):.3f} on {len(split['y'])} rows")


# ----------------------------------------------------------------- command line
def main(argv=None):
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=2450)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--card", action="store_true")
    ap.add_argument("--mutant", type=str, default=None, choices=sorted(MUTANTS))
    ap.add_argument("--data", type=str, default=None)
    args = ap.parse_args(argv)
    if args.card:
        print(json.dumps(MIND_CARD, indent=2))
        return 0
    t0 = time.time()
    budget = BUDGET_QUICK if args.quick else BUDGET_FULL
    steps = 250 if args.quick else 700
    n_seeds = 1 if args.quick else max(5, args.seeds)
    seeds = [args.seed + 17 * i for i in range(n_seeds)]
    print(f"{os.path.basename(__file__)}: base seed {args.seed}; evaluation seeds {seeds}; mutant {args.mutant}")
    try:
        data = make_task(np.random.default_rng(args.seed + 5000))
        lr = 0.03
        grad = test_gradients(data, 50, args.seed, mutant=args.mutant)
        lc, model = learning_check(data, steps, args.seed, lr, mutant=args.mutant)
        correctness = [{"id": "C1", "name": "gradient_check", "passed": grad["passed"], "detail": f"max rel error {grad['max_rel_error']:.2e}"},
                       determinism(data, args.seed, lr),
                       {"id": "C3", "name": "learning", "passed": lc["passed"], "detail": f"loss drop {lc['drop']:.2f}, margin over trivial {lc['margin']:.3f}"},
                       shuffled_control(data, steps, args.seed, lr)]
        mut = mutant_audit(data, 250, args.seed, lr) if not args.mutant else {"detected": 0, "total": len(MUTANTS), "score": 0.0}
        correctness.append({"id": "C5", "name": "mutant_detection", "passed": bool(mut["score"] == 1.0 and mut.get("healthy_short_run_passes", False)),
                            "detail": f"score {mut['score']:.2f}"})
        correctness += property_tests(args.seed)
        correctness.append(split_integrity(data))
        rows, hyps, kos = [], [], []
        if not args.quick:
            tune_main, tune_blind = make_task(np.random.default_rng(args.seed + 7000)), make_blind_task(np.random.default_rng(args.seed + 7001))
            lrs = {(cond, kind): tune_lr(kind, d, steps // 2, args.seed + 7002)
                   for cond, d, kinds in (("main", tune_main, ("qisma", "iqn", "mlp")), ("blind", tune_blind, ("qisma", "iqn"))) for kind in kinds}
            print(f"tuned learning rates: { {f'{c}/{k}': v for (c, k), v in lrs.items()} }")
            rows = [run_seed(s, steps, lrs) for s in seeds]
            hyps, kos = hypothesis_tests(rows, args.seed)
        else:
            hyps = [{"id": h["id"], "metric": h["metric"], "mean_diff": 0.0, "ci95": [0.0, 0.0], "mesi": h["mesi"], "n_seeds": 1,
                     "verdict": "not evaluated"} for h in MIND_CARD["hypotheses"]]
        runtime = time.time() - t0
        correctness.append({"id": "C8", "name": "budget", "passed": runtime <= budget, "detail": f"{runtime:.1f}s of {budget:.0f}s"})
        exit_code = 0 if all(c["passed"] for c in correctness) else 1
        if runtime > budget:
            exit_code = 3
    except FloatingPointError as err:
        print(f"non-finite values: {err}")
        return 4
    rng0 = np.random.default_rng(0)
    rep = {"schema_version": "1.0", "chapter": 245, "file": os.path.basename(__file__), "card_revision": MIND_CARD["card_revision"],
           "environment": {"python": sys.version.split()[0], "numpy": np.__version__}, "seeds": seeds, "runtime_s": runtime,
           "n_params": n_params(model), "baseline_params": {"iqn": n_params(build_iqn(rng0)), "mlp": n_params(build_mlp(rng0))},
           "gradcheck": grad, "correctness": correctness, "mutants": {k: mut[k] for k in ("detected", "total", "score")},
           "hypotheses": hyps, "knockouts": kos, "task_types": TASK_TYPES, "exit_code": exit_code}
    print_report(rep, rows)
    data_bridge(args.data, model)
    if args.json:
        write_report(args.json, rep)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

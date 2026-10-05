#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0278 · Ari Þorgilsson (1067-1148)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 14, Minds 261-280 Available on Amazon https://www.amazon.com/dp/B0HLYSVY5D
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0278_ari_thorgilsson_1067 - Ari Þorgilsson (1067-1148)
# END ATTRIBUTION
"""The Winter-Count Synchronizer.

Thesis
  Ari computes the past instead of remembering it: every tradition is a clock
  with its own unit, epoch and counting convention, and an event receives its
  date through any chain of synchronisms to a few external anchors, with all
  statements reconciled at once.

Evidence and provenance (provenance: belief; Íslendingabók survives)
  D1 ch.10   1120 is stated in five reckonings at once and summed up:
             "Þat verðr allt saman ellifu hundruð ok tuttugu ár".
  D2 ch.10   the lawspeakers' terms are the backbone; Markús Skeggjason gave
             them, from his kin and the memory of Bjarni the Wise.
  D3 ch.1,3  the settlement is dated by estimate and count ("ætlun ok tǫlu")
             against Edmund's killing (870); Hrafn took office sixty winters
             later, "vetri eða tveim" before Haraldr's death.
  D4 ch.4,10 counts are ordinal, in long hundreds: "fjóra daga ens fjórða
             hundraðs" (364), "tuttugu vetrum ens annars hundraðs" (120).
  D5 ch.9    Hallr remembered his baptism at three and died at ninety-four:
             an age statement bridges most of a century.
  D6 prol.   "hvatki es missagt es í frœðum þessum, þá es skylt at hafa þat
             heldr, es sannara reynisk".
  D7 schol.  ch.8 lists the foreign bishops as a succession although some were
             contemporary with Ísleifr (Hungrvaka; Grønlie 2006, p. xviii).
  D8 schol.  Ólafía Einarsdóttir 1964: absolute years 604, 870, 1000, 1120
             carry every other date in the book.

Doctrine -> mechanism -> test
  D3 D4 D5 -> M1 reckoner: unit (positive), counting convention (offset) and
              precision of each statement, from its kind and context
              -> C6.4 monotone in the count (labelled definition check),
                 C6.5 interval leverage in [0, 1]; knockouts
  D1 D2 D8 -> M2 synchronizer: weighted least-squares solve of all interval
              statements plus anchors (graph Laplacian system)
              -> C6.1 translation equivariance, C6.2 path additivity, H-SIG
  D6       -> M2 correction propagation: moving one anchor by x moves every
              date by a fraction of x in [0, 1] -> C6.3
  D7       -> blind spot: concurrent tenures stated as a chain -> H-BLIND

Research question (compositional and systematic generalization)
  Can a learner that composes learned interval semantics through one global
  consistency solve date entities many relational hops from any anchor, and
  on chronicles longer than those it was trained on, where bounded-depth
  message passing cannot?

Closest prior art and the delta
  Least-squares network adjustment and simple temporal networks (Dechter,
  Meiri and Pearl 1991); Bayesian chronological models (Bronk Ramsey 2009);
  harmonic label propagation (Zhu, Ghahramani and Lafferty 2003); implicit
  optimisation layers (Amos and Kolter 2017); graph convolution for document
  dating (Vashishth et al. 2018). Delta: the semantics of every reckoning
  (unit, counting convention, precision) is learned end to end through the
  exact implicit solve. In-file baselines: a size-matched message-passing
  network (MPNN) and the fixed-semantics least-squares chronologist.

Blind spot
  The solve trusts succession. When concurrent tenures are recorded as a
  chain, one structural error is spread coherently to every later date.

Task (the Book of Winters; synthetic)
  1. A backbone of 10-14 office terms of 2-24 summers each.
  2. Persons are born by descent (generations of about 33 years in the male
     line, 28 in the female line) or tied by ages to other nodes.
  3. Events are tied by "the k-th summer of", winters between, a person's
     age, or "the same / next summer as".
  4. Terms are stated in exclusive or inclusive reckoning, ages carry a
     half-winter ambiguity, generations are noisy, some counts are in
     misseri; three nuisance features.
  5. Two or three foreign events carry absolute years (anchors); every other
     node's year is the target.
  Shifted split: 22-26 terms, two anchors at the two ends only, 80%
  inclusive reckoning, triple nuisance scale.
  Blind split: the test worlds themselves, with 35% of tenures having
  overlapped their successor by 3-8 years yet stated as the chain interval.
  One statement in five of winters or ages is reckoned in misseri
  (half-years), as Ari reckons the year "í tveim misserum".

Limits
  Synthetic chronicles; a linear-Gaussian solve without order constraints;
  contradictions are averaged, not diagnosed. A research prototype of one
  AGI-oriented component, not an AGI, and no claim about Ari's inner life.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 2,
    "revision_log": ["r2 (after development runs, before the reported run): C6.4 relabelled as a "
                     "definition check; C6.5 interval-leverage property test added and traced to D5; "
                     "hypotheses, metrics, mesi and thresholds unchanged; 0507 similarity measured by the gate"],
    "generation": {"template_version": "codeguidelines 1.0 (2026-09-15)", "generator": "Claude",
                   "generator_version": "Claude Opus 5.5", "date": "2026-09-22"},
    "id": 278, "figure": "Ari Þorgilsson", "born": 1067, "died": 1148,
    "civilization": "Icelandic Norse", "provenance": "belief",
    "thesis": ("The past is computed, not remembered: each tradition is a clock with its own unit, "
               "epoch and counting convention, and a date is what an event inherits through any chain "
               "of synchronisms to a few external anchors, all statements reconciled at once."),
    "evidence": [
        {"id": "D1", "claim": "1120 stated in five reckonings and summed", "basis": "primary",
         "source": "Íslendingabók ch. 10 (ÍF I, 1968)"},
        {"id": "D2", "claim": "lawspeaker terms as chronological backbone, from Markús Skeggjason",
         "basis": "primary", "source": "Íslendingabók ch. 10"},
        {"id": "D3", "claim": "settlement dated by estimate and count against Edmund's killing",
         "basis": "primary", "source": "Íslendingabók ch. 1, 3"},
        {"id": "D4", "claim": "ordinal counting in long hundreds", "basis": "primary",
         "source": "Íslendingabók ch. 4, 7, 10"},
        {"id": "D5", "claim": "a witness's age at baptism bridges ninety years", "basis": "primary",
         "source": "Íslendingabók ch. 9"},
        {"id": "D6", "claim": "what proves truer is to be preferred", "basis": "primary",
         "source": "Íslendingabók, prologue"},
        {"id": "D7", "claim": "concurrent foreign bishops listed as a succession", "basis": "scholarship",
         "source": "Grønlie, Íslendingabók / Kristni saga (2006), p. xviii"},
        {"id": "D8", "claim": "four absolute years carry the relative dating", "basis": "scholarship",
         "source": "Ólafía Einarsdóttir, Studier i kronologisk metode (1964)"},
    ],
    "research_question": {"category": "compositional and systematic generalization",
                          "question": ("Does composing learned interval semantics through one global "
                                       "consistency solve date entities many hops from any anchor, and on "
                                       "longer chronicles, where bounded-depth message passing cannot?")},
    "mechanism": {
        "name": "Winter-count synchronizer", "family": "implicit differentiable least-squares layer over a "
        "relational interval graph with a learned reckoning head",
        "signature_modules": ["reckoner.unit", "reckoner.convention", "synchronizer"],
        "closest_prior_art": ["least-squares network adjustment / simple temporal networks (Dechter, Meiri, "
                              "Pearl 1991)", "Bayesian chronological modelling (Bronk Ramsey 2009)",
                              "harmonic label propagation (Zhu, Ghahramani, Lafferty 2003)",
                              "OptNet implicit layers (Amos, Kolter 2017)",
                              "graph convolution for document dating (Vashishth et al. 2018)",
                              "message-passing neural networks (Gilmer et al. 2017)"],
        "overlap": "Medium",
        "prior_art_queries": ["differentiable least squares layer infer event dates relative temporal "
                              "statements", "neural document dating temporal graph", "temporal constraint "
                              "network learning durations"],
        "contribution_type": "mechanism",
        "delta": ("the unit, counting convention and precision of every reckoning are learned end to end "
                  "through the exact implicit solve instead of being hand-set or smeared into embeddings"),
    },
    "traceability": [
        {"doctrine": "D3", "mechanism": "M1", "property_test": "C6.4", "hypothesis": "H-NEC"},
        {"doctrine": "D4", "mechanism": "M1", "property_test": "C6.4", "hypothesis": "H-SIG"},
        {"doctrine": "D5", "mechanism": "M1", "property_test": "C6.5", "hypothesis": "H-SIG"},
        {"doctrine": "D1", "mechanism": "M2", "property_test": "C6.2", "hypothesis": "H-SIG"},
        {"doctrine": "D2", "mechanism": "M2", "property_test": "C6.1", "hypothesis": "H-NEC"},
        {"doctrine": "D8", "mechanism": "M2", "property_test": "C6.1", "hypothesis": "H-SIG"},
        {"doctrine": "D6", "mechanism": "M2", "property_test": "C6.3", "hypothesis": "H-NEC"},
        {"doctrine": "D7", "mechanism": "M2", "property_test": "C6.2", "hypothesis": "H-BLIND"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "the synchronizer dates the shifted (longer, end-anchored) chronicles "
         "better than a size-matched MPNN", "metric": "MAE in years", "split": "shifted",
         "comparison": "baseline - model", "direction": "greater", "mesi": 5.0, "seeds": 5},
        {"id": "H-NEC", "statement": "knocking out the synchronizer hurts held-out dating more than "
         "knocking out the precision head", "metric": "MAE in years", "split": "held-out",
         "comparison": "signature_knockout - matched_knockout", "direction": "greater", "mesi": 5.0,
         "seeds": 5},
        {"id": "H-BLIND", "statement": "under serialized concurrent terms the synchronizer degrades more "
         "than the MPNN", "condition": "35% of terms overlap their successor by 3-8 years",
         "grounding": "Ari set the foreign bishops in sequence though some were contemporary with Ísleifr",
         "metric": "MAE increase in years (blind - clean)", "comparison": "model - baseline",
         "direction": "greater", "mesi": 1.0, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.5, "margin_over_trivial": 0.4, "shuffled_band": 0.75,
                   "grad_rel_tol": 1e-5, "clip_norm": 5.0},
    "probe_predictions": [{"probe": "P4 length extrapolation", "expected": "above baseline"},
                          {"probe": "P3 compositional split", "expected": "above baseline"},
                          {"probe": "P1 delayed recall", "expected": "below baseline"},
                          {"probe": "P9 perturbation robustness", "expected": "equal to baseline"}],
    "dialectic_links": [
        {"chapter": 192, "relation": "teacher", "test": "none: Bede's AD reckoning reaches Ari as an anchor "
         "system; Bede's concordance of cycles is periodic and is not re-implemented here"},
        {"chapter": 321, "relation": "successor", "test": "none: Snorri named Ari the first vernacular "
         "historian and inherited his chronology"},
        {"chapter": 332, "relation": "successor", "test": "none: Sturla's Landnámabók redates the settlement "
         "to 874 against Ari's 870"}],
    "corpus_neighbors": [
        {"chapter": 289, "similarity": None, "difference": "al-Idrisi fixes coordinates first and corrects "
         "testimony attached to them; here the coordinate itself is computed from relational testimony"},
        {"chapter": 273, "similarity": None, "difference": "Shen Kuo reads disagreement among redundant "
         "routes; here most nodes have one route and the gain is reach, not redundancy"},
        {"chapter": 272, "similarity": None, "difference": "al-Zarqali gives constants a periodic motion; "
         "here the semantics of a reckoning are static and aperiodic"},
        {"chapter": 192, "similarity": None, "difference": "Bede reconciles periodic cycles by phase; here "
         "aperiodic counted intervals are composed along a graph"},
        {"chapter": 282, "similarity": None, "difference": "Anna Komnene refuses an unsupported chronology; "
         "Ari constructs one and lets corrections propagate"},
        {"chapter": 507, "similarity": 0.035, "difference": "sample chapter; 9-token shingle Jaccard"}],
    "barometer": {"cognitive_processing": ["P3", "P4"], "embodied_cognition": [],
                  "world_modeling": ["shifted split", "correction propagation C6.3"],
                  "consciousness": [], "language_understanding": ["reckoning semantics of stated counts"],
                  "emotional_intelligence": [], "creativity": [], "autonomy": []},
    "task_types": ["sequence_regression"],
    "applications": [
        {"use": "event timelines from relative temporal statements", "sector": "digital humanities / NLP",
         "dataset": "MATRES temporal relation corpus (Ning, Wu, Roth 2018)"},
        {"use": "birth-year imputation in large genealogies", "sector": "historical demography",
         "dataset": "FamiLinx (Kaplanis et al. 2018), public de-identified"},
        {"use": "offset estimation for unsynchronised event logs", "sector": "distributed systems",
         "dataset": "Intel Berkeley Research Lab sensor data (2004)"}],
    "safety_notes": ("Genealogical and log data only in public de-identified form; no person-level "
                     "inference about living individuals; outputs are never evidence about the historical Ari."),
}

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

SCALE = 50.0            # years per unit of the regression target
UNIT_BOUND = 4.0        # log-unit bounded to (-4, 4): 0.018 to 55 years per count
CONV_SCALE = 2.0        # counting-convention offset in years per unit of its head
W_MIN = 1e-2            # floor on statement precision
LAMBDA_ANCHOR = 100.0   # precision of a foreign anchor
EPS_PRIOR = 1e-12       # vanishing prior toward the anchor mean (definiteness only)
HIDDEN = 24
GNN_HIDDEN = 8
GNN_ROUNDS = 8
N_KIND = 6              # term, winters, age, generation, ordinal, synchronism
N_FEAT = 13
TASK_TYPES = ["sequence_regression"]


# BEGIN STANDARD UTILITIES v1.0
def softplus(x):
    return np.logaddexp(0.0, x)


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def logsumexp(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(m, axis) + np.log(np.sum(np.exp(x - m), axis=axis))


def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def adam_init(params):
    return {"t": 0, "m": {k: np.zeros_like(v) for k, v in params.items()},
            "v": {k: np.zeros_like(v) for k, v in params.items()}}


def adam_step(params, grads, state, lr, b1=0.9, b2=0.999, eps=1e-8, sign=1.0):
    state["t"] += 1
    for k in params:
        state["m"][k] = b1 * state["m"][k] + (1 - b1) * grads[k]
        state["v"][k] = b2 * state["v"][k] + (1 - b2) * grads[k] ** 2
        mh = state["m"][k] / (1 - b1 ** state["t"])
        vh = state["v"][k] / (1 - b2 ** state["t"])
        params[k] -= sign * lr * mh / (np.sqrt(vh) + eps)


def clip_global_norm(grads, max_norm):
    norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    if norm > max_norm:
        for k in grads:
            grads[k] = grads[k] * (max_norm / norm)
    return norm


def grad_check(loss_fn, params, grads, rng, eps=1e-6, n_sample=20, floor=1e-4):
    """Central differences on every tensor; returns {name: max relative error}."""
    out = {}
    for name, p in params.items():
        flat = p.reshape(-1)
        g = grads[name].reshape(-1)
        idx = np.arange(flat.size) if flat.size <= 2 * n_sample else np.unique(np.concatenate(
            [rng.choice(flat.size, n_sample, replace=False), [int(np.argmax(np.abs(g)))]]))
        worst = 0.0
        for i in idx:
            old = flat[i]
            flat[i] = old + eps
            lp = loss_fn()
            flat[i] = old - eps
            lm = loss_fn()
            flat[i] = old
            num = (lp - lm) / (2 * eps)
            worst = max(worst, abs(num - g[i]) / max(abs(num) + abs(g[i]), floor))
        out[name] = worst
    return out


def paired_bootstrap(diffs, rng, n_boot=2000):
    d = np.asarray(diffs, float)
    means = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    return float(d.mean()), (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def verdict(mean, ci, mesi, direction):
    lo, hi = ci
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


class NonFinite(Exception):
    """Raised on any non-finite loss, gradient, parameter or activation (exit code 4)."""


# ----------------------------------------------------------------- data and tasks
SPLITS = {
    "train": dict(n=192, terms=(10, 14), persons=8, events=16, anchors=3, anchor_ends=False,
                  p_incl=0.5, scribe_sd=1.0, overlap_p=0.0),
    "val": dict(n=48, terms=(10, 14), persons=8, events=16, anchors=3, anchor_ends=False,
                p_incl=0.5, scribe_sd=1.0, overlap_p=0.0),
    "test": dict(n=64, terms=(10, 14), persons=8, events=16, anchors=3, anchor_ends=False,
                 p_incl=0.5, scribe_sd=1.0, overlap_p=0.0),
    "shift": dict(n=64, terms=(22, 26), persons=14, events=30, anchors=2, anchor_ends=True,
                  p_incl=0.8, scribe_sd=3.0, overlap_p=0.0),
}


class WorldBuilder:
    """Accumulates the nodes and the counted statements of one synthetic chronicle."""

    def __init__(self, rng):
        self.rng, self.year, self.ntype, self.stmts = rng, [], [], []

    def node(self, year, ntype):
        self.year.append(float(year))
        self.ntype.append(ntype)
        return len(self.year) - 1

    def stmt(self, a, b, kind, n, incl=0, fem=0, half=0):
        self.stmts.append((a, b, kind, float(n), incl, fem, half))

    def counted(self, a, b, kind, years):
        """Winters or ages; one statement in five is reckoned in misseri (half-years)."""
        half = int(self.rng.random() < 0.2)
        self.stmt(a, b, kind, years * (2 if half else 1), half=half)

    def nearest_later(self, i, lo, hi):
        return [j for j, y in enumerate(self.year) if j != i and lo <= y - self.year[i] <= hi and self.ntype[j] != 3]


def _tie_by_age(rng, wb, p):
    cand = wb.nearest_later(p, 3, 80)
    if cand:
        x = cand[int(rng.integers(len(cand)))]
        wb.counted(p, x, 2, wb.year[x] - wb.year[p] - int(rng.integers(0, 2)))
        return True
    return False


def _tie_event(rng, wb, e, starts, lens, persons):
    ye, opts = wb.year[e], []
    terms = [k for k in range(len(lens)) if wb.year[starts[k]] <= ye < wb.year[starts[k]] + lens[k]]
    if terms:
        opts.append("ordinal")
    near = [j for j, y in enumerate(wb.year) if j != e and wb.ntype[j] in (0, 2) and 1 <= abs(y - ye) <= 60]
    if near:
        opts.append("winters")
    alive = [p for p in persons if 3 <= ye - wb.year[p] <= 80]
    if alive:
        opts.append("age")
    choice = opts[int(rng.integers(len(opts)))] if opts else "winters_any"
    if choice == "ordinal":
        k = terms[int(rng.integers(len(terms)))]
        wb.stmt(starts[k], e, 4, ye - wb.year[starts[k]] + 1)
    elif choice == "age":
        p = alive[int(rng.integers(len(alive)))]
        wb.counted(p, e, 2, ye - wb.year[p] - int(rng.integers(0, 2)))
    else:
        pool = near if near else [j for j in range(len(wb.year)) if j != e and wb.ntype[j] != 3]
        x = pool[int(rng.integers(len(pool)))]
        gap = abs(ye - wb.year[x])
        n = gap + (int(rng.choice([-1, 1])) if (rng.random() < 0.05 and gap > 1) else 0)
        a, b = (x, e) if wb.year[x] <= ye else (e, x)
        wb.counted(a, b, 1, n)


def _backbone(rng, wb, spec):
    starts, lens = [wb.node(rng.integers(860, 941), 0)], []
    for _ in range(int(rng.integers(spec["terms"][0], spec["terms"][1] + 1))):
        length = int(rng.integers(2, 25))
        lap = min(int(rng.integers(3, 9)), length - 1) if rng.random() < spec["overlap_p"] else 0
        incl = int(rng.random() < spec["p_incl"])
        nxt = wb.node(wb.year[starts[-1]] + length - lap, 0)
        wb.stmt(starts[-1], nxt, 0, length + incl, incl=incl)
        starts.append(nxt)
        lens.append(length)
    return starts, lens


def make_world(rng, spec):
    """One chronicle: backbone of terms, persons, events, then foreign anchors."""
    wb = WorldBuilder(rng)
    starts, lens = _backbone(rng, wb, spec)
    lo, hi = wb.year[starts[0]] - 30, wb.year[starts[-1]] - 10
    persons = []
    for _ in range(spec["persons"]):
        if persons and rng.random() < 0.45:
            anc = persons[int(rng.integers(len(persons)))]
            fem, k = int(rng.random() < 0.5), int(rng.integers(1, 4))
            gap = float(rng.normal(28.0 if fem else 33.0, 4.0, size=k).sum())
            p = wb.node(round(wb.year[anc] + gap), 1)
            wb.stmt(anc, p, 3, k, fem=fem)
            if rng.random() < 0.5:
                _tie_by_age(rng, wb, p)
        else:
            p = wb.node(rng.integers(lo, hi), 1)
            if not _tie_by_age(rng, wb, p):
                _tie_event(rng, wb, p, starts, [], [])
        persons.append(p)
    for _ in range(spec["events"]):
        base = [j for j in range(len(wb.year)) if wb.ntype[j] == 2]
        if base and rng.random() < 0.15:
            x = base[int(rng.integers(len(base)))]
            e = wb.node(wb.year[x] + int(rng.integers(0, 2)), 2)
            wb.stmt(x, e, 5, wb.year[e] - wb.year[x])
        else:
            e = wb.node(rng.integers(int(wb.year[starts[0]]), int(wb.year[starts[-1]]) + 1), 2)
            _tie_event(rng, wb, e, starts, lens, persons)
        if rng.random() < 0.5:
            _tie_event(rng, wb, e, starts, lens, persons)
    anchors = []
    real = [j for j in range(len(wb.year)) if wb.ntype[j] != 3]
    targets = ([starts[0], starts[-1]] if spec["anchor_ends"] else
               list(rng.choice(real, size=spec["anchors"], replace=False)))
    for x in targets:
        off = 0 if rng.random() < 0.4 else int(rng.integers(1, 61)) * (1 if rng.random() < 0.5 else -1)
        f = wb.node(wb.year[x] + off, 3)
        wb.stmt(*((x, f) if off >= 0 else (f, x)), 5 if off == 0 else 1, abs(off))
        anchors.append(f)
    return finish_world(wb, anchors, spec["scribe_sd"])


def finish_world(wb, anchors, scribe_sd):
    s = np.array(wb.stmts, dtype=float)
    n_e = len(s)
    z = np.zeros((n_e, N_FEAT))
    z[np.arange(n_e), s[:, 2].astype(int)] = 1.0
    z[:, 6], z[:, 7], z[:, 8] = s[:, 4], s[:, 5], s[:, 6]
    z[:, 9:12] = wb.rng.normal(0.0, scribe_sd, size=(n_e, 3))
    z[:, 12] = np.log1p(s[:, 3]) / 4.0
    return build_world(s[:, 0].astype(int), s[:, 1].astype(int), s[:, 3], z,
                       np.array(wb.year), np.array(wb.ntype), np.array(anchors, dtype=int))


def build_world(src, dst, counts, z, years, ntype, anchor_idx):
    n_n = len(years)
    inc = np.zeros((len(src), n_n))
    inc[np.arange(len(src)), dst] += 1.0
    inc[np.arange(len(src)), src] -= 1.0
    a_mask = np.zeros(n_n)
    a_mask[anchor_idx] = 1.0
    t_anchor = np.zeros(n_n)
    t_anchor[anchor_idx] = years[anchor_idx]
    return dict(src=src, dst=dst, n=counts.astype(float), z=z, y=years.astype(float), ntype=ntype,
                B=inc, a_mask=a_mask, t_anchor=t_anchor, mu=float(years[anchor_idx].mean()),
                mask=(1.0 - a_mask) * (ntype != 3))


def make_splits(ss, quick=False):
    """One independent generator per split, derived from the seed's SeedSequence."""
    data, children = {}, ss.spawn(len(SPLITS) + 1)
    for (name, spec), child in zip(SPLITS.items(), children):
        r = np.random.default_rng(child)
        n = max(12, spec["n"] // 3) if quick else spec["n"]
        data[name] = [make_world(r, spec) for _ in range(n)]
    r = np.random.default_rng(children[-1])
    data["blind"] = [serialize(w, r) for w in data["test"]]
    return data


def serialize(w, rng, p=0.35):
    """Blind split, paired with the clean test world: a tenure that overlapped its successor by
    3-8 years is still stated as the start-to-start interval, so the chain over-counts."""
    v = dict(w)
    v["n"], v["z"] = w["n"].copy(), w["z"].copy()
    terms = np.flatnonzero(w["z"][:, 0] > 0)
    hit = terms[rng.random(terms.size) < p]
    v["n"][hit] += rng.integers(3, 9, size=hit.size)
    v["z"][hit, 12] = np.log1p(v["n"][hit]) / 4.0
    return v


def fingerprint(w):
    h = hashlib.sha1()
    for key in ("src", "dst", "n", "y"):
        h.update(np.ascontiguousarray(np.round(w[key], 6)).tobytes())
    return h.hexdigest()


def sequence_to_worlds(X, y=None):
    """Probe adapter: a sequence is a chain of statements from an anchor at year 0."""
    out = []
    for b in range(X.shape[0]):
        t_len = X.shape[1]
        years = np.zeros(t_len + 1)
        years[-1] = 0.0 if y is None else float(np.ravel(y[b])[0])
        ntype = np.full(t_len + 1, 2)
        w = build_world(np.arange(t_len), np.arange(1, t_len + 1), np.ones(t_len), X[b].astype(float),
                        years, ntype, np.array([0]))
        w["mask"] = np.zeros(t_len + 1)
        w["mask"][-1] = 1.0
        out.append(w)
    return out


# ------------------------------------------------------------------------ model
def _inv_unit(u):
    return UNIT_BOUND * math.atanh(math.log(u) / UNIT_BOUND)


def build_model(in_dim, out_dim, task_type, rng, **cfg):
    """M1 reckoner (unit, convention, precision per statement) feeding the M2 synchronizer.

    The unit head starts near 0.08 years per count: the learner begins not knowing what a
    winter is worth, so any skill it shows had to be learned from the targets."""
    if task_type not in TASK_TYPES or out_dim != 1:
        raise ValueError("build_model supports sequence_regression with out_dim 1")
    h = cfg.get("hidden", HIDDEN)
    p = {"W1": rng.normal(0.0, 1.0 / math.sqrt(in_dim), (in_dim, h)), "b1": np.zeros(h),
         "wu": rng.normal(0.0, 0.1 / math.sqrt(h), h), "bu": np.array([_inv_unit(0.08)]),
         "wc": rng.normal(0.0, 0.1 / math.sqrt(h), h), "bc": np.zeros(1),
         "ws": rng.normal(0.0, 0.1 / math.sqrt(h), h), "bs": np.array([math.log(math.e - 1.0)])}
    return {"arch": "sync", "params": p, "lr": cfg.get("lr", 0.01), "ko": {}, "mutant": cfg.get("mutant"),
            "prior": cfg.get("prior", "anchor_mean"), "signed_weights": False, "signed_unit": False}


def copy_model(model):
    m = dict(model)
    m["params"] = {k: v.copy() for k, v in model["params"].items()}
    m["ko"] = dict(model["ko"])
    return m


def reckon(model, w):
    """Each counted statement becomes a duration d = count * unit + convention, with precision."""
    p, ko = model["params"], model["ko"]
    hid = np.tanh(w["z"] @ p["W1"] + p["b1"])
    ru, rc, rs = hid @ p["wu"] + p["bu"][0], hid @ p["wc"] + p["bc"][0], hid @ p["ws"] + p["bs"][0]
    th = np.tanh(ru / UNIT_BOUND)
    unit = ru if model["signed_unit"] else np.exp(UNIT_BOUND * th)
    if ko.get("reckoner.unit") == "identity":
        unit = np.ones_like(ru)
    conv = np.zeros_like(rc) if ko.get("reckoner.convention") == "zero" else CONV_SCALE * rc
    prec = rs if model["signed_weights"] else softplus(rs) + W_MIN
    if ko.get("reckoner.precision") == "mean":
        prec = np.full_like(prec, prec.mean())
    return w["n"] * unit + conv, prec, (hid, rs, th, unit)


def solve(model, w, d, prec):
    """Weighted least squares over all statements and anchors, solved in anchor-centred years:
    A (t - mu) = B'Wd + lambda (T - mu) + eps (c - mu), with A = B'WB + lambda diag(anchors) + eps I."""
    inc, eps = w["B"], model.get("eps", EPS_PRIOR)
    a_mat = inc.T @ (prec[:, None] * inc)
    a_mat[np.diag_indices_from(a_mat)] += LAMBDA_ANCHOR * w["a_mask"] + eps
    center = w["mu"] if model["prior"] == "anchor_mean" else 0.0
    rhs = inc.T @ (prec * d) + LAMBDA_ANCHOR * w["a_mask"] * (w["t_anchor"] - w["mu"]) + eps * (center - w["mu"])
    if model["ko"].get("synchronizer") == "identity":
        t0 = np.full(len(rhs), center - w["mu"])
        return w["mu"] + t0 + (rhs - a_mat @ t0) / np.diag(a_mat), a_mat
    return w["mu"] + np.linalg.solve(a_mat, rhs), a_mat


def forward(model, w):
    d, prec, cache = reckon(model, w)
    t, a_mat = solve(model, w, d, prec)
    return t, (d, prec, cache, a_mat)


def _sync_loss_grads(model, batch):
    """Implicit differentiation: u = A^-1 dL/dt, dL/dd = W B u, dL/dW = Bu (d - Bt)."""
    p = model["params"]
    grads = {k: np.zeros_like(v) for k, v in p.items()}
    total = 0.0
    for w in batch:
        t, (d, prec, (hid, rs, th, unit), a_mat) = forward(model, w)
        cnt = w["mask"].sum()
        r = (t - w["y"]) / SCALE
        total += float(np.sum(w["mask"] * r * r) / cnt)
        g = 2.0 * w["mask"] * r / (SCALE * cnt * len(batch))
        u = g if model["mutant"] == "adjoint_skip" else np.linalg.solve(a_mat, g)
        bu, bt = w["B"] @ u, w["B"] @ t
        gd, gw = prec * bu, bu * (d - bt)
        g_ru, g_rc, g_rs = gd * w["n"] * unit * (1.0 - th ** 2), gd * CONV_SCALE, gw * sigmoid(rs)
        for key, gr in (("u", g_ru), ("c", g_rc), ("s", g_rs)):
            grads["w" + key] += hid.T @ gr
            grads["b" + key] += gr.sum()
        dpre = (np.outer(g_ru, p["wu"]) + np.outer(g_rc, p["wc"]) + np.outer(g_rs, p["ws"])) * (1 - hid ** 2)
        grads["W1"] += w["z"].T @ dpre
        grads["b1"] += dpre.sum(axis=0)
    if model["mutant"] == "grad_zero_W1":
        grads["W1"][:] = 0.0
    return total / len(batch), grads


# ------------------------------------------------------- baselines and rivals
def build_mpnn(rng, lr=0.01, h=GNN_HIDDEN):
    """Size-matched message-passing network over the same statement graph (Gilmer et al. 2017)."""
    e_in, n_in = N_FEAT + 2, 5
    p = {"Win": rng.normal(0.0, 1.0 / math.sqrt(n_in), (n_in, h)), "bin": np.zeros(h),
         "Wm": rng.normal(0.0, 1.0 / math.sqrt(h + e_in), (h + e_in, h)), "bm": np.zeros(h),
         "Wu": rng.normal(0.0, 1.0 / math.sqrt(2 * h), (2 * h, h)), "bu": np.zeros(h),
         "wo": rng.normal(0.0, 1.0 / math.sqrt(h), h), "bo": np.zeros(1)}
    return {"arch": "mpnn", "params": p, "lr": lr, "ko": {}, "mutant": None}


def assemble(batch):
    """Disjoint union of worlds; every statement sends a message both ways (signed count)."""
    x0, ms, md, mq, mu, y, wt, off = [], [], [], [], [], [], [], 0
    for w in batch:
        n_n, rel = len(w["y"]), w["a_mask"] * (w["t_anchor"] - w["mu"]) / SCALE
        x0.append(np.stack([w["a_mask"], rel, w["ntype"] == 0, w["ntype"] == 1, w["ntype"] == 2], 1))
        for a, b, sgn in ((w["src"], w["dst"], 1.0), (w["dst"], w["src"], -1.0)):
            ms.append(a + off)
            md.append(b + off)
            mq.append(np.hstack([w["z"], sgn * w["n"][:, None] / SCALE, np.full((len(a), 1), sgn)]))
        mu.append(np.full(n_n, w["mu"]))
        y.append(w["y"])
        wt.append(w["mask"] / (w["mask"].sum() * len(batch)))
        off += n_n
    ms, md = np.concatenate(ms), np.concatenate(md)
    ord_d, ord_s = np.argsort(md, kind="stable"), np.argsort(ms, kind="stable")
    return {"x0": np.vstack(x0).astype(float), "ms": ms, "md": md, "mq": np.vstack(mq),
            "mu": np.concatenate(mu), "y": np.concatenate(y), "wt": np.concatenate(wt),
            "ord_d": ord_d, "st_d": np.searchsorted(md[ord_d], np.arange(off)),
            "ord_s": ord_s, "st_s": np.searchsorted(ms[ord_s], np.arange(off)),
            "inv_deg": 1.0 / np.maximum(np.bincount(md, minlength=off), 1),
            "sizes": [len(w["y"]) for w in batch]}


def _seg_sum(values, order, starts):
    return np.add.reduceat(values[order], starts, axis=0)


def mpnn_forward(model, g):
    p = model["params"]
    hs, cache = [np.tanh(g["x0"] @ p["Win"] + p["bin"])], []
    for _ in range(GNN_ROUNDS):
        inp = np.concatenate([hs[-1][g["ms"]], g["mq"]], axis=1)
        msg = np.tanh(inp @ p["Wm"] + p["bm"])
        agg = _seg_sum(msg, g["ord_d"], g["st_d"]) * g["inv_deg"][:, None]
        cat = np.concatenate([hs[-1], agg], axis=1)
        hs.append(np.tanh(cat @ p["Wu"] + p["bu"]))
        cache.append((inp, msg, cat))
    return g["mu"] + SCALE * (hs[-1] @ p["wo"] + p["bo"][0]), hs, cache


def _mpnn_loss_grads(model, g):
    p, h = model["params"], GNN_HIDDEN
    pred, hs, cache = mpnn_forward(model, g)
    r = (pred - g["y"]) / SCALE
    do = 2.0 * g["wt"] * r
    grads = {k: np.zeros_like(v) for k, v in p.items()}
    grads["wo"], grads["bo"] = hs[-1].T @ do, np.array([do.sum()])
    dh = np.outer(do, p["wo"])
    for k in reversed(range(GNN_ROUNDS)):
        inp, msg, cat = cache[k]
        du = dh * (1.0 - hs[k + 1] ** 2)
        grads["Wu"] += cat.T @ du
        grads["bu"] += du.sum(axis=0)
        dcat = du @ p["Wu"].T
        dmsg = (dcat[:, h:] * g["inv_deg"][:, None])[g["md"]]
        dpre = dmsg * (1.0 - msg ** 2)
        grads["Wm"] += inp.T @ dpre
        grads["bm"] += dpre.sum(axis=0)
        dh = dcat[:, :h] + _seg_sum((dpre @ p["Wm"].T)[:, :h], g["ord_s"], g["st_s"])
    du0 = dh * (1.0 - hs[0] ** 2)
    grads["Win"], grads["bin"] = g["x0"].T @ du0, du0.sum(axis=0)
    return float(np.sum(g["wt"] * r * r)), grads


def fixed_semantics_predict(w):
    """The classical chronologist: a count is years, a generation is 30, every statement weighs 1."""
    kind = np.argmax(w["z"][:, :N_KIND], axis=1) if w["z"].shape[1] >= N_KIND else np.ones(len(w["n"]))
    d = np.where(kind == 3, 30.0 * w["n"], w["n"])
    stub = {"ko": {}, "prior": "anchor_mean"}
    return solve(stub, w, d, np.ones(len(d)))[0]


# ------------------------------------------------------------------ interface
def loss_and_grads(model, batch):
    if model["arch"] == "mpnn":
        return _mpnn_loss_grads(model, batch if isinstance(batch, dict) else assemble(batch))
    return _sync_loss_grads(model, batch)


def predict(model, X):
    """Worlds -> list of per-node years; a probe array (B, T, F) -> (B, 1) year of the last node."""
    worlds = sequence_to_worlds(X) if isinstance(X, np.ndarray) else X
    if model["arch"] == "sync":
        out = [forward(model, w)[0] for w in worlds]
    else:
        pred = mpnn_forward(model, assemble(worlds))[0]
        out = np.split(pred, np.cumsum([len(w["y"]) for w in worlds])[:-1])
    return np.array([o[-1] for o in out])[:, None] if isinstance(X, np.ndarray) else out


def hidden_states(model, X):
    worlds = sequence_to_worlds(X) if isinstance(X, np.ndarray) else X
    if model["arch"] == "sync":
        return [reckon(model, w)[2][0] for w in worlds]
    return mpnn_forward(model, assemble(worlds))[1][-1]


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


def mae(model, worlds):
    preds = [fixed_semantics_predict(w) for w in worlds] if model == "fixed" else \
        [np.full(len(w["y"]), w["mu"]) for w in worlds] if model == "trivial" else predict(model, worlds)
    return float(np.mean(np.concatenate([np.abs(p - w["y"])[w["mask"] > 0] for p, w in zip(preds, worlds)])))


def fit(model, data, budget, rng, batch=8):
    """Adam under a fixed update budget on fixed minibatches drawn with the given generator."""
    worlds = sequence_to_worlds(*data) if isinstance(data, tuple) else \
        data if isinstance(data, list) else data["train"]
    order = rng.permutation(len(worlds))
    batches = [[worlds[i] for i in order[k:k + batch]] for k in range(0, len(worlds) - batch + 1, batch)]
    if model["arch"] == "mpnn":
        batches = [assemble(b) for b in batches]
    state, losses = adam_init(model["params"]), []
    lr = 0.0 if model["mutant"] == "zero_lr" else model["lr"]
    sign = -1.0 if model["mutant"] == "sign_flip" else 1.0
    for _ in range(budget):
        loss, grads = loss_and_grads(model, batches[int(rng.integers(len(batches)))])
        if not (np.isfinite(loss) and all(np.all(np.isfinite(g)) for g in grads.values())):
            raise NonFinite("non-finite loss or gradient")
        clip_global_norm(grads, MIND_CARD["thresholds"]["clip_norm"])
        adam_step(model["params"], grads, state, lr, sign=sign)
        losses.append(loss)
    return losses


# ------------------------------------------------------------------ registries
def modules(model):
    if model["arch"] == "mpnn":
        return {"mpnn": {"params": list(model["params"]), "role": "message passing, 8 shared rounds",
                         "signature": False}}
    return {
        "reckoner.trunk": {"params": ["W1", "b1"], "signature": False,
                           "role": "tanh feature layer over statement kind, context and log count"},
        "reckoner.unit": {"params": ["wu", "bu"], "signature": True,
                          "role": "positive multiplicative scale per statement (count to duration)"},
        "reckoner.convention": {"params": ["wc", "bc"], "signature": True,
                                "role": "additive offset per statement (counting convention)"},
        "reckoner.precision": {"params": ["ws", "bs"], "signature": False,
                               "role": "softplus edge weight of the least-squares system"},
        "synchronizer": {"params": [], "signature": True,
                         "role": "global weighted least-squares solve on the interval graph (Laplacian "
                                 "system, implicit gradient)"}}


KNOCKOUT_MODES = {"reckoner.unit": "identity", "reckoner.convention": "zero",
                  "reckoner.precision": "mean", "synchronizer": "identity"}


def knockout(model, name, mode):
    """identity/zero/mean replacement; the synchronizer's identity is one Jacobi sweep, no propagation."""
    m = copy_model(model)
    m["ko"][name] = mode
    return m


MUTANTS = {"sign_flip": "Adam update applied with the wrong sign",
           "zero_lr": "learning rate forced to zero",
           "grad_zero_W1": "gradient of the reckoner trunk W1 zeroed",
           "adjoint_skip": "implicit gradient uses dL/dt in place of A^-1 dL/dt"}


# ------------------------------------------------------------------- training
LR_GRID = (0.003, 0.01, 0.03)
STEPS = {"full": 600, "quick": 200}


def make_model(arch, ss, lr, mutant=None):
    rng = np.random.default_rng(ss)
    if arch == "mpnn":
        return build_mpnn(rng, lr=lr)
    return build_model(N_FEAT, 1, "sequence_regression", rng, lr=lr, mutant=mutant)


def train_selected(arch, data, ss, steps, mutant=None):
    """Same grid, optimizer and update budget for every architecture; selection on validation only."""
    s_init, s_fit = ss.spawn(2)
    best = None
    for lr in LR_GRID:
        model = make_model(arch, s_init, lr, mutant)
        losses = fit(model, data["train"], steps, np.random.default_rng(s_fit))
        score = mae(model, data["val"])
        if best is None or score < best[1]:
            best = (model, score, losses)
    return best


def run_seed(seed, steps, quick, mutant=None):
    s_data, s_sync, s_mpnn = np.random.SeedSequence(seed).spawn(3)
    data = make_splits(s_data, quick)
    sync, _, losses = train_selected("sync", data, s_sync, steps, mutant)
    mpnn = train_selected("mpnn", data, s_mpnn, steps)[0]
    res = {}
    for split in ("test", "shift", "blind"):
        for name, m in (("sync", sync), ("mpnn", mpnn), ("fixed", "fixed"), ("trivial", "trivial")):
            res[name + "_" + split] = mae(m, data[split])
    for name, mode in KNOCKOUT_MODES.items():
        res["ko:" + name] = mae(knockout(sync, name, mode), data["test"])
    return res, {"data": data, "sync": sync, "mpnn": mpnn, "losses": losses, "s_sync": s_sync}


# ---------------------------------------------------------- correctness tests
def gradcheck_suite(data, ss, steps, mutant=None, archs=("sync", "mpnn"), stages=("init", "after")):
    """C1 on every tensor of both models, at initialisation and after `steps` Adam updates."""
    rows = []
    for arch, child in zip(archs, ss.spawn(len(archs))):
        s_init, s_fit, s_chk = child.spawn(3)
        model = make_model(arch, s_init, 0.01, mutant)
        batch = data["val"][:2] if arch == "sync" else assemble(data["val"][:2])
        for stage in stages:
            if stage == "after":
                fit(model, data["train"], steps, np.random.default_rng(s_fit))
            _, g = loss_and_grads(model, batch)
            errs = grad_check(lambda: loss_and_grads(model, batch)[0], model["params"], g,
                              np.random.default_rng(s_chk))
            rows.append({"arch": arch, "stage": stage, "tensors": len(errs), "total": len(model["params"]),
                         "max_rel": max(errs.values())})
    tol = MIND_CARD["thresholds"]["grad_rel_tol"]
    return all(r["max_rel"] <= tol and r["tensors"] == r["total"] for r in rows), rows


def c3_learning(losses, test_mae, trivial):
    th = MIND_CARD["thresholds"]
    drop = 1.0 - float(np.mean(losses[-20:])) / float(np.mean(losses[:10]))
    ok = drop >= th["loss_drop_fraction"] and test_mae <= (1.0 - th["margin_over_trivial"]) * trivial
    return ok, f"loss drop {drop:.3f} (>= {th['loss_drop_fraction']}); held-out MAE {test_mae:.2f} y " \
               f"vs trivial {trivial:.2f} y (margin {th['margin_over_trivial']})"


def shuffle_targets(worlds, rng):
    out = []
    for w in worlds:
        v = dict(w)
        v["y"] = w["y"].copy()
        idx = np.flatnonzero(w["mask"] > 0)
        v["y"][idx] = w["y"][rng.permutation(idx)]
        out.append(v)
    return out


def c4_shuffled(data, ss, lr, steps):
    s_init, s_fit, s_perm = ss.spawn(3)
    model = make_model("sync", s_init, lr)
    fit(model, shuffle_targets(data["train"], np.random.default_rng(s_perm)), steps,
        np.random.default_rng(s_fit))
    got, trivial = mae(model, data["test"]), mae("trivial", data["test"])
    band = MIND_CARD["thresholds"]["shuffled_band"]
    return got >= band * trivial, f"shuffled-label held-out MAE {got:.2f} y >= {band} x trivial {trivial:.2f} y"


def c5_mutants(data, ss, steps):
    """A mutant is detected when C1 (at initialisation) or C3 fails on it."""
    found = {}
    trivial = mae("trivial", data["test"])
    for name, child in zip(MUTANTS, ss.spawn(len(MUTANTS))):
        s_gc, s_init, s_fit = child.spawn(3)
        c1 = gradcheck_suite(data, s_gc, steps, name, archs=("sync",), stages=("init",))[0]
        model = make_model("sync", s_init, 0.01, name)
        losses = fit(model, data["train"], steps, np.random.default_rng(s_fit))
        c3 = c3_learning(losses, mae(model, data["test"]), trivial)[0]
        found[name] = not (c1 and c3)
    return found


def random_sync(rng, **flags):
    """Random parameters for property search, well away from the trained solution."""
    m = build_model(N_FEAT, 1, "sequence_regression", rng)
    for v in m["params"].values():
        v += rng.normal(0.0, 0.7, v.shape)
    m.update(flags)
    return m


def tree_world(rng, n=24):
    """A statement graph without cycles and a single anchored node: every statement is binding."""
    src = np.array([int(rng.integers(i)) for i in range(1, n)])
    dst = np.arange(1, n)
    counts = rng.integers(1, 40, size=n - 1).astype(float)
    z = np.zeros((n - 1, N_FEAT))
    z[np.arange(n - 1), rng.integers(0, N_KIND, size=n - 1)] = 1.0
    z[:, 6:12] = rng.normal(0.0, 1.0, size=(n - 1, 6))
    z[:, 12] = np.log1p(counts) / 4.0
    ntype = np.full(n, 2)
    ntype[0] = 3
    return build_world(src, dst, counts, z, rng.uniform(800, 1200, size=n), ntype, np.array([0]))


def v_translation(m, w, rng):
    x = float(rng.uniform(-800.0, 800.0))
    v = dict(w)
    v["t_anchor"], v["mu"] = w["t_anchor"] + x * w["a_mask"], w["mu"] + x
    return float(np.max(np.abs(forward(m, v)[0] - forward(m, w)[0] - x)))


def v_tree(m, w, rng):
    t, (d, _, _, _) = forward(m, w)
    return float(np.max(np.abs(w["B"] @ t - d)))


def v_max_principle(m, w, rng):
    a = int(rng.choice(np.flatnonzero(w["a_mask"] > 0)))
    v = dict(w)
    v["t_anchor"] = w["t_anchor"].copy()
    v["t_anchor"][a] += 1.0
    r = forward(m, v)[0] - forward(m, w)[0]
    return float(max(0.0, -r.min(), r.max() - 1.0))


def v_monotone_count(m, w, rng):
    d0 = reckon(m, w)[0]
    v = dict(w)
    v["n"] = w["n"] + 1.0
    return float(max(0.0, np.max(d0 - reckon(m, v)[0])))


def v_leverage(m, w, rng):
    """Stretching statement e by delta moves t_dst - t_src by w_e b_e' A^-1 b_e delta, in [0, delta]."""
    _, (_, prec, _, a_mat) = forward(m, w)
    lev = prec * np.einsum("ij,ji->i", w["B"], np.linalg.solve(a_mat, w["B"].T))
    return float(max(0.0, -lev.min(), lev.max() - 1.0))


PROPERTIES = [  # id, name, violation, tolerance, negative control, uses tree worlds
    ("C6.1", "translation equivariance", v_translation, 1e-6, {"prior": "zero", "eps": 1e-2}, False),
    ("C6.2", "path additivity on acyclic chronicles", v_tree, 1e-6, {"ko": {"synchronizer": "identity"}}, True),
    ("C6.3", "anchor response in [0, 1] (max principle)", v_max_principle, 1e-9, {"signed_weights": True},
     False),
    ("C6.4", "definition check, not evidence: duration monotone in count", v_monotone_count, 0.0,
     {"signed_unit": True}, False),
    ("C6.5", "interval leverage in [0, 1]", v_leverage, 1e-9, {"signed_weights": True}, False),
]


def c6_properties(data, rng, trials):
    """Random search over random parameters and inputs; each negative control must be caught."""
    rows = []
    for pid, name, fn, tol, broken, trees in PROPERTIES:
        worst = {"model": 0.0, "control": 0.0}
        for _ in range(trials):
            w = tree_world(rng) if trees else data["test"][int(rng.integers(len(data["test"])))]
            for tag, flags in (("model", {}), ("control", broken)):
                worst[tag] = max(worst[tag], fn(random_sync(rng, **flags), w, rng))
        rows.append({"id": pid, "name": name, "passed": worst["model"] <= tol and worst["control"] > tol,
                     "detail": f"max violation {worst['model']:.2e} (tol {tol:g}); negative control "
                               f"{worst['control']:.2e}"})
    return rows


def c7_split_integrity(data):
    train = {fingerprint(w) for w in data["train"]}
    overlap = {k: len(train & {fingerprint(w) for w in data[k]}) for k in ("val", "test", "shift", "blind")}
    return all(v == 0 for v in overlap.values()), f"train overlap {overlap}"


def c2_determinism(data, ss, steps):
    """Same seed, same losses and outputs; parameters and activations finite; probe adapter runs."""
    runs, s_init, s_fit = [], *ss.spawn(2)
    for _ in range(2):
        model = make_model("sync", s_init, 0.01)
        losses = fit(model, data["train"], steps, np.random.default_rng(s_fit))
        runs.append((losses, np.concatenate(predict(model, data["test"][:4])), model))
    same = runs[0][0] == runs[1][0] and np.array_equal(runs[0][1], runs[1][1])
    probe = predict(runs[0][2], np.random.default_rng(s_init).normal(size=(3, 7, N_FEAT)))
    acts = [runs[0][1], probe] + list(runs[0][2]["params"].values()) + hidden_states(runs[0][2], data["test"][:2])
    if not all(np.all(np.isfinite(a)) for a in acts):
        raise NonFinite("non-finite parameter or activation")
    return same and probe.shape == (3, 1), f"identical losses and outputs across two runs: {same}; " \
                                           f"probe output shape {probe.shape}"


# ----------------------------------------------------------- hypothesis tests
def hypothesis_diffs(r):
    return {"H-SIG": r["mpnn_shift"] - r["sync_shift"],
            "H-NEC": r["ko:synchronizer"] - r["ko:reckoner.precision"],
            "H-BLIND": (r["sync_blind"] - r["sync_test"]) - (r["mpnn_blind"] - r["mpnn_test"])}


def evaluate_hypotheses(per_seed, rng, quick):
    rows = []
    for h in MIND_CARD["hypotheses"]:
        diffs = [hypothesis_diffs(r)[h["id"]] for r in per_seed]
        mean, ci = paired_bootstrap(diffs, rng)
        judged = not quick and len(diffs) >= h["seeds"]
        rows.append({"id": h["id"], "metric": h["metric"], "mean_diff": mean, "ci95": list(ci), "mesi": h["mesi"],
                     "n_seeds": len(diffs),
                     "verdict": verdict(mean, ci, h["mesi"], h["direction"]) if judged else "not evaluated"})
    return rows


def knockout_table(per_seed, rng, model):
    rows, registry = [], modules(model)
    for name, mode in KNOCKOUT_MODES.items():
        mean, ci = paired_bootstrap([r["ko:" + name] - r["sync_test"] for r in per_seed], rng)
        rows.append({"module": name, "mode": mode, "metric_change": mean, "ci95": list(ci),
                     "signature": registry[name]["signature"]})
    return rows


# --------------------------------------------------------------------- report
def format_report(rep):
    out = [f"=== VERIFIED REPORT · chapter {rep['chapter']:04d} ===",
           f"figure        {MIND_CARD['figure']} ({MIND_CARD['born']}-{MIND_CARD['died']}), provenance "
           f"{MIND_CARD['provenance']}, card revision {rep['card_revision']}",
           f"environment   python {rep['environment']['python']}, numpy {rep['environment']['numpy']}",
           f"seeds         {rep['seeds']} (SeedSequence per seed; {rep['mode']} run, {rep['steps']} updates, "
           f"lr grid {list(LR_GRID)} chosen on validation)",
           f"runtime       {rep['runtime_s']:.1f} s (budget {rep['budget_s']} s)",
           f"parameters    synchronizer {rep['n_params']}, MPNN baseline {rep['n_params_baseline']} "
           f"(ratio {rep['n_params_baseline'] / rep['n_params']:.3f})"]
    g = rep["gradcheck"]
    out.append(f"gradcheck     {g['tensors_checked']}/{g['tensors_total']} tensors, max rel error "
               f"{g['max_rel_error']:.2e} at {g['checked_at']}, eps 1e-6, tol "
               f"{MIND_CARD['thresholds']['grad_rel_tol']:g}")
    out.append("correctness")
    out += [f"  {c['id']:<5} {'PASS' if c['passed'] else 'FAIL'}  {c['name']}: {c['detail']}"
            for c in rep["correctness"]]
    mu = rep["mutants"]
    out.append(f"mutation score {mu['detected']}/{mu['total']} = {mu['score']:.2f}  {mu['by_name']}")
    out.append("hypotheses    (paired per-seed differences, 95% bootstrap CI, 2000 resamples)")
    out += [f"  {h['id']:<8} mean {h['mean_diff']:+8.2f}  CI [{h['ci95'][0]:+.2f}, {h['ci95'][1]:+.2f}]  mesi "
            f"{h['mesi']:g}  n={h['n_seeds']}  {h['verdict']}" for h in rep["hypotheses"]]
    out.append("knockouts     (held-out MAE change in years; signature modules marked *)")
    out += [f"  {k['module'] + ('*' if k['signature'] else ''):<21} {k['mode']:<9} {k['metric_change']:+8.2f}  "
            f"CI [{k['ci95'][0]:+.2f}, {k['ci95'][1]:+.2f}]" for k in rep["knockouts"]]
    out.append("descriptive   (mean MAE in years over seeds; not hypotheses)")
    out.append(f"  {'split':<7}" + "".join(f"{n:>10}" for n in ("trivial", "fixed", "MPNN", "sync")))
    for split in ("test", "shift", "blind"):
        out.append(f"  {split:<7}" + "".join(f"{rep['descriptive'][n + '_' + split]:>10.2f}"
                                             for n in ("trivial", "fixed", "mpnn", "sync")))
    out.append(f"real data     {rep['real_data']}")
    out.append(f"task types    {rep['task_types']}   exit code {rep['exit_code']}")
    out.append("=== END REPORT ===")
    return "\n".join(out)


# ------------------------------------------------------ optional real-data bridge
KIND_NAMES = ("term", "winters", "age", "generation", "ordinal", "same")


def load_chronicle_csv(path):
    """Rows src,dst,kind,count[,inclusive,misseri]; kind 'anchor' fixes node src at year `count`.
    Genealogies (FamiLinx parent-child links with a few known birth years) convert directly."""
    names, stmts = {}, []
    with open(path, encoding="utf-8") as fh:
        rows = [ln.strip().split(",") for ln in fh if ln.strip() and not ln.startswith("src,")]
    for r in rows:
        a = names.setdefault(r[0], len(names))
        if r[2] == "anchor":
            f = names.setdefault(f"@{r[3]}:{r[0]}", len(names))
            stmts.append((a, f, 5, 0.0, 0, 0, 0, float(r[3])))
        else:
            b = names.setdefault(r[1], len(names))
            extra = [int(x) for x in r[4:6]] + [0, 0]
            stmts.append((a, b, KIND_NAMES.index(r[2]), float(r[3]), extra[0], 0, extra[1], math.nan))
    years, ntype = np.zeros(len(names)), np.full(len(names), 2)
    anchors = [s[1] for s in stmts if not math.isnan(s[7])]
    years[anchors], ntype[anchors] = [s[7] for s in stmts if not math.isnan(s[7])], 3
    s = np.array([x[:7] for x in stmts], dtype=float)
    z = np.zeros((len(s), N_FEAT))
    z[np.arange(len(s)), s[:, 2].astype(int)] = 1.0
    z[:, 6], z[:, 8], z[:, 12] = s[:, 4], s[:, 6], np.log1p(s[:, 3]) / 4.0
    w = build_world(s[:, 0].astype(int), s[:, 1].astype(int), s[:, 3], z, years, ntype, np.array(anchors))
    return w, [k for k in names]


def real_data_bridge(path, model):
    if not path or not os.path.exists(path):
        return f"skipped ({'no --data PATH given' if not path else path + ' not found'}; nothing is downloaded)"
    w, names = load_chronicle_csv(path)
    t = forward(model, w)[0]
    for name, year, kind in zip(names, t, w["ntype"]):
        if kind != 3:
            print(f"  {name}: {year:.1f}")
    return f"{int((w['ntype'] != 3).sum())} nodes dated from {len(w['n'])} statements in {path}"


# ------------------------------------------------------------------ entry point
def correctness_suite(args, ctx, per_seed, steps, t0):
    data, rng = ctx["data"], np.random.default_rng(args.seed)
    ss = np.random.SeedSequence(args.seed).spawn(8)[3:]   # children 0-2 belong to run_seed
    ok1, gc_rows = gradcheck_suite(data, ss[0], 60, args.mutant)
    rows = [{"id": "C1", "name": "gradient_check", "passed": ok1, "detail": "; ".join(
        f"{r['arch']}@{r['stage']} {r['tensors']}/{r['total']} tensors max {r['max_rel']:.2e}" for r in gc_rows)}]
    ok, det = c2_determinism(data, ss[1], 40)
    rows.append({"id": "C2", "name": "determinism_and_finiteness", "passed": ok, "detail": det})
    ok, det = c3_learning(ctx["losses"], per_seed[0]["sync_test"], per_seed[0]["trivial_test"])
    rows.append({"id": "C3", "name": "learning", "passed": ok, "detail": det})
    ok, det = c4_shuffled(data, ss[2], ctx["sync"]["lr"], steps)
    rows.append({"id": "C4", "name": "shuffled_label_control", "passed": ok, "detail": det})
    found = c5_mutants(data, ss[3], steps)
    rows.append({"id": "C5", "name": "mutant_detection", "passed": all(found.values()),
                 "detail": f"{sum(found.values())}/{len(found)} detected via C1 or C3"})
    rows += c6_properties(data, rng, 12 if args.quick else 30)
    ok, det = c7_split_integrity(data)
    rows.append({"id": "C7", "name": "split_integrity", "passed": ok, "detail": det})
    budget = 20 if args.quick else 180
    rows.append({"id": "C8", "name": "budget", "passed": time.time() - t0 <= budget,
                 "detail": f"{time.time() - t0:.1f} s of {budget} s"})
    return rows, gc_rows, found, budget


def protocol(args):
    t0 = time.time()
    steps = STEPS["quick" if args.quick else "full"]
    seeds = [args.seed + i for i in range(args.seeds or (1 if args.quick else 5))]
    print(f"seeds {seeds}; mutant {args.mutant}")
    per_seed, ctx0 = [], None
    for s in seeds:
        res, ctx = run_seed(s, steps, args.quick, args.mutant)
        per_seed.append(res)
        ctx0 = ctx0 or ctx
        print(f"  seed {s}: sync test/shift/blind {res['sync_test']:.2f}/{res['sync_shift']:.2f}/"
              f"{res['sync_blind']:.2f}  MPNN {res['mpnn_test']:.2f}/{res['mpnn_shift']:.2f}/{res['mpnn_blind']:.2f}")
    rows, gc_rows, found, budget = correctness_suite(args, ctx0, per_seed, steps, t0)
    rng, real = np.random.default_rng(args.seed), real_data_bridge(args.data, ctx0["sync"])
    failed = [r for r in rows if not r["passed"]]
    code = 1 if any(r["id"] != "C8" for r in failed) else (3 if failed else 0)
    rep = {"schema_version": "1.0", "chapter": MIND_CARD["id"], "file": os.path.basename(__file__),
           "card_revision": MIND_CARD["card_revision"],
           "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
           "seeds": seeds, "runtime_s": round(time.time() - t0, 2), "budget_s": budget,
           "mode": "quick" if args.quick else "full", "steps": steps,
           "n_params": n_params(ctx0["sync"]), "n_params_baseline": n_params(ctx0["mpnn"]),
           "gradcheck": {"tensors_checked": sum(r["tensors"] for r in gc_rows if r["stage"] == "init"),
                         "tensors_total": sum(r["total"] for r in gc_rows if r["stage"] == "init"),
                         "max_rel_error": max(r["max_rel"] for r in gc_rows),
                         "checked_at": ["init", "after_60_training_steps"], "passed": rows[0]["passed"]},
           "correctness": rows,
           "mutants": {"detected": sum(found.values()), "total": len(found),
                       "score": sum(found.values()) / len(found), "by_name": found},
           "hypotheses": evaluate_hypotheses(per_seed, rng, args.quick),
           "knockouts": knockout_table(per_seed, rng, ctx0["sync"]),
           "descriptive": {k: float(np.mean([r[k] for r in per_seed])) for k in per_seed[0]},
           "per_seed": per_seed, "real_data": real, "task_types": TASK_TYPES, "exit_code": code}
    print(format_report(rep))
    if args.json:
        write_report(rep, args.json)
    return code


def parse_args(argv):
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__),
                                 description="Winter-count synchronizer: full verification protocol")
    ap.add_argument("--quick", action="store_true", help="one seed, reduced steps, all correctness tests")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--json", default=None, metavar="PATH")
    ap.add_argument("--card", action="store_true")
    ap.add_argument("--mutant", default=None)
    ap.add_argument("--data", default=None, metavar="PATH")
    args = ap.parse_args(argv)
    if args.mutant is not None and args.mutant not in MUTANTS:
        ap.error(f"unknown mutant {args.mutant!r}; registered: {', '.join(MUTANTS)}")
    if args.seeds is not None and args.seeds < 1:
        ap.error("--seeds must be at least 1")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    try:
        return protocol(args)
    except NonFinite as exc:
        print(f"non-finite values: {exc}")
        return 4


if __name__ == "__main__":
    sys.exit(main())

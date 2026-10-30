#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0340 · Ramon Llull (1232-1316)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 17, Minds 321-340 Available on Amazon https://www.amazon.com/dp/B0HP7Z1NKC
# Resume and Interactive Demos at https://artificiology.com/
# END ATTRIBUTION
"""
ARS CAMERARUM — the Chamber Count
Chapter 0340 · Ramon Llull (1232–1316) · pure-NumPy research prototype

THESIS
------
For Llull a judgement is never a score placed on a thing; it is a decision
placed on a PAIR. The unit of his cognition is the camera, the two-letter
chamber of the Third Figure, and a conclusion is the count of chambers won.

EVIDENCE AND PROVENANCE (provenance: belief — his own surviving texts)
----------------------------------------------------------------------
D1  The Third Figure of the ternary Art lays out the 36 pairs of nine letters
    b–k; the Art is worked by "evacuating" every chamber in turn.
    Ars brevis (1308), part IV; Ars generalis ultima (1305–08), part VI.
D2  Artifitium electionis personarum (Bo II.A.10, 1274–83): candidates are
    compared two at a time in every cell of a 120-cell figure; unus punctus
    is marked for the winner of each cell; the dots are counted; ties are
    marked for both; final ties go to lots. Ed. Hägele & Pukelsheim 2001.
D3  Blaquerna ch. 24 (1283): 36 chambers, "most votes in most cells";
    Natana asks to inspect the 36 chambers after her own election.
D4  De arte eleccionis (1 July 1299, Paris): a successive-elimination rule,
    n−1 comparisons instead of n(n−1)/2, no tie rule, open voting.
D5  Every principle unfolds into three correlatives: agent (-tivum), patient
    (-bile) and act (-are). Liber de ascensu et descensu intellectus (1305);
    Ars generalis ultima, definitions; Gayà 1979; Bonner 2007.
D6  The Second Figure T relates any two terms by difference, concordance or
    contrariety. Ars brevis part II.
D7  The Fourth Figure rotates three wheels to produce triples; the third
    letter is the question or rule asked of the pair. Ars brevis part IV.
D8  Blind spot (scholarship): the exhaustive count multiplies comparisons
    without discriminating where a single ordering would do; the 1299
    rule is order-dependent (Hägele & Pukelsheim 2001, §6.6); pairwise
    majorities can cycle (McLean & London 1990).

DOCTRINE → MECHANISM → TEST
---------------------------
  doctrine  mechanism                                   test
  D1,D2,D3  M1 chamber count: soft Copeland aggregation  C6.3, H-SIG
  D5        M2 correlative projections: agent/patient    C1, knockout
            tanh maps feeding a bilinear act
  D6        M3 concordance channel: 3-way chamber        C6.2, knockout
            softmax {i>j, j>i, concordant/tie}
  D7        M4 act tensor: question-indexed bilinear     H-NEC
            form (the third wheel)
  D4        R1 rival: successive-elimination aggregation  C6.3, C6.4, H-RIVAL
  D8        blind-spot condition: transitive, noisy world H-BLIND

RESEARCH QUESTION
-----------------
Preference and value aggregation. When the target is only the winner of a
set, can latent pairwise chambers be recovered from that weak supervision,
beat a pointwise (Bradley–Terry-style) scorer on intransitive relations,
and remain auditable cell by cell — as Natana demanded of her 36 chambers?

CLOSEST PRIOR ART AND THE DELTA
-------------------------------
Blade–chest matchup model (Chen & Joachims, WSDM 2016): item-specific
"blade" and "chest" vectors, P(a beats b) = σ(blade_a·chest_b − blade_b·
chest_a). General Preference Model (Zhang et al., ICML 2025; arXiv
2410.02197): skew-symmetric bilinear preference embeddings trained on
pairwise labels. Copeland rule (1951). Rao–Kupper ties (1967).
Delta: (i) chambers are latent and trained only from the set-level winner,
never from pairwise labels; (ii) the skew form is question-conditioned by
a third factor (the Fourth Figure); (iii) a symmetric concordance channel
gives each chamber a three-way verdict; (iv) the 1299 successive
elimination is implemented as a rival aggregation and tested against the
1283 count. Overlap: Medium — known parts in a new combination and setting.

BLIND SPOT
----------
Llull believed the exhaustive chamber count yields necessary demonstration
in every domain. Where a single ordering suffices (a transitive, noisy
world), the count estimates n(n−1)/2 noisy contests instead of one
ordering; it should generalise worse than a pointwise scorer there. That
is H-BLIND, predicted before the first run.

TASK (the chapter house)
------------------------
1. Draw N candidates (5–7 in distribution; 9 shifted) with 6 traits around
   a per-episode offset; observe them with jitter plus 2 pure-noise traits.
2. Draw a question q in {0,1,2}: the three conditions of the 1274–83 oath
   (holiness of life, knowledge, disposition of heart).
3. Hidden contest margin m_q(i,j) = w_q·(z_i − z_j) + κ·z_iᵀA_q z_j with A_q
   skew-symmetric; κ sets intransitivity; symmetric noise is added.
4. i beats j iff m_q(i,j) > τ; inside the band |m| ≤ τ the cell is even and
   both get half a dot (AEP). The target is the Copeland winner; episodes
   without a unique winner are redrawn (final ties are the world's lots).
5. Items are presented in random order. Blind-spot split: κ = 0 with
   larger noise. Shifted split: N = 9, κ in [1.2, 1.8].

LIMITS
------
Synthetic data; no claim about the historical person or about human
preferences; a research prototype of an aggregation mechanism, not an AGI.
Run: python3 <this file>
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 1,
    "revision_log": [],
    "generation": {"template_version": "codeguidelines-1.0", "generator": "Claude",
                   "generator_version": "Fable 5.1", "date": "2026-09-25"},
    "id": 340, "figure": "Ramon Llull", "born": 1232, "died": 1316,
    "civilization": "Catalan (Majorca)",
    "provenance": "belief",
    "thesis": "A judgement is never a score on a thing but a decision on a pair; "
              "thinking is the exhaustive adjudication of two-letter chambers and "
              "the count of chambers won.",
    "evidence": [
        {"id": "D1", "claim": "The Third Figure enumerates the 36 pairs of nine letters and the Art evacuates each chamber",
         "basis": "primary", "source": "Ars brevis (1308) IV; Ars generalis ultima (1305-08) VI"},
        {"id": "D2", "claim": "Exhaustive pairwise election with unus punctus per cell, dots counted, ties marked for both, lots last",
         "basis": "primary", "source": "Artifitium electionis personarum, Vat. lat. 9332 f.11r-12v, ed. Hägele & Pukelsheim 2001"},
        {"id": "D3", "claim": "36 chambers; most votes in most cells; the abbess-elect inspects the chambers",
         "basis": "primary", "source": "Blaquerna (1283) ch. 24, Cod. Hisp. 67 f.32v-34r"},
        {"id": "D4", "claim": "Successive elimination with n-1 comparisons and no tie rule",
         "basis": "primary", "source": "De arte eleccionis, Paris 1 July 1299, Cod. Cus. 83 f.47v-48r"},
        {"id": "D5", "claim": "Every principle unfolds into agent, patient and act correlatives",
         "basis": "primary", "source": "Liber de ascensu et descensu intellectus (1305); Gayà 1979; Bonner 2007"},
        {"id": "D6", "claim": "Any two terms stand in difference, concordance or contrariety",
         "basis": "primary", "source": "Ars brevis (1308) II, Second Figure"},
        {"id": "D7", "claim": "Three rotating wheels join two principles with a question",
         "basis": "primary", "source": "Ars brevis (1308) IV, Fourth Figure"},
        {"id": "D8", "claim": "The count multiplies comparisons; the 1299 rule is order-dependent; majorities can cycle",
         "basis": "scholarship", "source": "Hägele & Pukelsheim 2001 §6.6, §7; McLean & London 1990"},
    ],
    "research_question": {
        "category": "preference and value aggregation",
        "question": "Can latent pairwise chambers trained only from set-level winners beat a pointwise "
                    "scorer under intransitive relations while staying auditable cell by cell?"},
    "mechanism": {
        "name": "Ars Camerarum (chamber count)",
        "family": "pairwise skew-symmetric bilinear scoring with Copeland pooling",
        "signature_modules": ["correlatives", "act_tensor", "concordance", "chamber_count"],
        "closest_prior_art": ["Chen & Joachims 2016 blade-chest", "Zhang et al. 2025 General Preference Model (arXiv 2410.02197)",
                              "Copeland 1951", "Rao & Kupper 1967 ties"],
        "overlap": "Medium",
        "prior_art_queries": ["intransitive preference model skew-symmetric bilinear", "blade chest matchup intransitivity",
                              "Copeland winner learning set supervision", "Bradley-Terry intransitive reward model 2025"],
        "contribution_type": "mechanism",
        "delta": "Chambers are latent and learned from winner-only supervision; the skew form is question-conditioned; "
                 "a concordance channel gives three-way chamber verdicts; the 1299 elimination rule is a tested rival.",
    },
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1", "property_test": "C6.3", "hypothesis": "H-SIG"},
        {"doctrine": "D2", "mechanism": "M1", "property_test": "C6.4", "hypothesis": "H-RIVAL"},
        {"doctrine": "D5", "mechanism": "M2", "property_test": "C6.1", "hypothesis": "H-SIG"},
        {"doctrine": "D6", "mechanism": "M3", "property_test": "C6.2", "hypothesis": "H-NEC"},
        {"doctrine": "D7", "mechanism": "M4", "property_test": "C6.1", "hypothesis": "H-NEC"},
        {"doctrine": "D4", "mechanism": "R1", "property_test": "C6.3", "hypothesis": "H-RIVAL"},
        {"doctrine": "D8", "mechanism": "M1", "property_test": "C6.4", "hypothesis": "H-BLIND"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "The chamber count beats a size-matched pointwise scorer on the shifted split (N=9, high intransitivity)",
         "metric": "accuracy", "split": "shifted", "comparison": "model - baseline", "direction": "greater", "mesi": 0.10, "seeds": 5},
        {"id": "H-NEC", "statement": "Averaging the act tensor over questions hurts more than removing the input vestibule",
         "metric": "accuracy", "comparison": "signature_knockout - matched_knockout", "direction": "less", "mesi": 0.10, "seeds": 5},
        {"id": "H-BLIND", "statement": "In a transitive noisy world the chamber count generalises worse than the pointwise scorer",
         "condition": "kappa = 0, noise 0.6, N in 5-7", "grounding": "the Art multiplies comparisons where one ordering suffices",
         "metric": "accuracy", "comparison": "model - baseline", "direction": "less", "mesi": 0.02, "seeds": 5},
        {"id": "H-RIVAL", "statement": "The 1283 exhaustive count beats the 1299 successive elimination under item-order nuisance",
         "metric": "accuracy", "comparison": "model - rival", "direction": "greater", "mesi": 0.05, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.40, "margin_over_trivial": 0.25, "shuffled_band": 0.08, "grad_rel_error": 1e-5},
    "probe_predictions": [
        {"probe": "P2", "expected": "above baseline"}, {"probe": "P4", "expected": "above baseline"},
        {"probe": "P8", "expected": "above baseline"}, {"probe": "P9", "expected": "equal to baseline"},
        {"probe": "P10", "expected": "below baseline"},
    ],
    "dialectic_links": [
        {"chapter": 337, "relation": "rival", "test": "none (necessary reasons vs. articles beyond demonstration)"},
        {"chapter": 342, "relation": "rival", "test": "none (anti-Averroist campaign, Paris 1309-11)"},
        {"chapter": 401, "relation": "successor", "test": "H-RIVAL (Cusanus copied the 1299 tract in 1428 and devised a rank-order count)"},
        {"chapter": 636, "relation": "successor", "test": "none (Dissertatio de arte combinatoria, 1666)"},
        {"chapter": 748, "relation": "successor", "test": "H-RIVAL (pairwise majority; cycles)"},
    ],
    "corpus_neighbors": [
        {"chapter": 203, "similarity": 0.0, "difference": "al-Khalil enumerates a hypothesis space; here each enumerated pair is adjudicated and counted"},
        {"chapter": 336, "similarity": 0.0, "difference": "Alfonso X sorts situations by where chance enters; here lots are only the last tie-break"},
        {"chapter": 337, "similarity": 0.0, "difference": "Aquinas constrains possession of universals; here the unit is the pair, not the concept"},
        {"chapter": 342, "similarity": 0.0, "difference": "Siger tags acts with two indices; here the act is a bilinear contest between two roles"},
    ],
    "barometer": {
        "cognitive_processing": ["P2 relational transfer to unseen items", "P4 length extrapolation to N=9"],
        "embodied_cognition": [], "world_modeling": ["tracking under regime change kappa"],
        "consciousness": ["calibration (ECE) of chamber probabilities against hidden pairwise outcomes"],
        "language_understanding": [], "emotional_intelligence": ["aggregation of conflicting judgements"],
        "creativity": [], "autonomy": [],
    },
    "task_types": ["sequence_classification"],
    "applications": [
        {"use": "Reward modelling with intransitive human preferences", "sector": "AI alignment", "dataset": "LMSYS Chatbot Arena conversations (pairwise votes)"},
        {"use": "Committee and candidate selection from pairwise juror judgements", "sector": "governance", "dataset": "PrefLib election data"},
        {"use": "Match-up prediction with counter-play", "sector": "sports analytics", "dataset": "Jeff Sackmann ATP match results (GitHub)"},
    ],
    "safety_notes": "Synthetic candidates only; no data about real people; no claim of replicating a mind.",
}

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

CHAPTER = 340
N_QUESTIONS = 3           # the three conditions of the 1274-83 oath
D_TRAIT = 6               # latent traits per candidate
D_NOISE = 2               # pure-noise traits
D_IN = D_TRAIT + D_NOISE  # 8 features per item
RANK = 8                  # width of each correlative projection
CLIP_NORM = 5.0
DEFAULT_BUDGET_S = 180.0
QUICK_BUDGET_S = 20.0

# BEGIN STANDARD UTILITIES v1.0
def stable_softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def logsumexp(z, axis=-1):
    m = np.max(z, axis=axis, keepdims=True)
    return np.squeeze(m, axis) + np.log(np.sum(np.exp(z - m), axis=axis))


def softplus(x):
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


class Adam:
    def __init__(self, params, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads, sign=-1.0):
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] = params[k] + sign * self.lr * mh / (np.sqrt(vh) + self.eps)


def clip_global_norm(grads, max_norm):
    total = np.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    if total > max_norm:
        scale = max_norm / (total + 1e-12)
        grads = {k: g * scale for k, g in grads.items()}
    return grads, total


def finite_difference_check(loss_fn, params, grads, rng, n_entries=20, eps=1e-6):
    """Central differences in float64 on every tensor: n random entries plus the
    largest-gradient entry. Returns (max relative error, per-tensor dict)."""
    report, worst = {}, 0.0
    for k, p in params.items():
        flat = p.reshape(-1)
        g = grads[k].reshape(-1)
        idx = set(rng.choice(flat.size, size=min(n_entries, flat.size), replace=False).tolist())
        idx.add(int(np.argmax(np.abs(g))))
        errs = []
        for i in sorted(idx):
            old = flat[i]
            flat[i] = old + eps
            lp = loss_fn()
            flat[i] = old - eps
            lm = loss_fn()
            flat[i] = old
            num = (lp - lm) / (2 * eps)
            errs.append(abs(num - g[i]) / max(1e-8, abs(num) + abs(g[i])))
        report[k] = float(max(errs))
        worst = max(worst, report[k])
    return worst, report


def paired_bootstrap(diffs, rng, n_boot=2000, alpha=0.05):
    diffs = np.asarray(diffs, dtype=np.float64)
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = np.mean(diffs[rng.integers(0, diffs.size, size=diffs.size)])
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.mean(diffs)), float(lo), float(hi)


def verdict(mean, lo, hi, direction, mesi):
    if direction == "greater":
        if lo > 0 and mean >= mesi:
            return "supported"
        if hi < 0:
            return "contradicted"
    else:
        if hi < 0 and mean <= -mesi:
            return "supported"
        if lo > 0:
            return "contradicted"
    return "inconclusive"


def write_report(report, path=None):
    lines = [f"=== VERIFIED REPORT · chapter {report['chapter']:04d} ==="]
    env = report["environment"]
    lines.append(f"file {report['file']}  python {env['python']}  numpy {env['numpy']}")
    lines.append(f"seeds {report['seeds']}  runtime {report['runtime_s']:.1f}s  n_params {report['n_params']}")
    gc = report["gradcheck"]
    lines.append(f"gradcheck tensors {gc['tensors_checked']}/{gc['tensors_total']}  max_rel_error {gc['max_rel_error']:.2e}  "
                 f"at {gc['checked_at']}  passed {gc['passed']}")
    lines.append("correctness:")
    for c in report["correctness"]:
        lines.append(f"  {c['id']:<5} {c['name']:<28} {'PASS' if c['passed'] else 'FAIL'}  {c['detail']}")
    m = report["mutants"]
    lines.append(f"mutants detected {m['detected']}/{m['total']}  score {m['score']:.2f}")
    lines.append("hypotheses:")
    for h in report["hypotheses"]:
        lines.append(f"  {h['id']:<8} {h['metric']:<9} mean_diff {h['mean_diff']:+.4f}  ci95 [{h['ci95'][0]:+.4f}, {h['ci95'][1]:+.4f}]  "
                     f"mesi {h['mesi']:.2f}  n {h['n_seeds']}  {h['verdict']}")
    lines.append("knockouts (accuracy change on held-out):")
    for k in report["knockouts"]:
        lines.append(f"  {k['module']:<14} signature {str(k['signature']):<5} change {k['metric_change']:+.4f}  "
                     f"ci95 [{k['ci95'][0]:+.4f}, {k['ci95'][1]:+.4f}]")
    for k, v in report.get("diagnostics", {}).items():
        lines.append(f"diagnostic {k}: {v}")
    lines.append(f"task_types {report['task_types']}  exit_code {report['exit_code']}")
    lines.append("=== END REPORT ===")
    text = "\n".join(lines)
    print(text)
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
    return text
# END STANDARD UTILITIES


# ---------------------------------------------------------------- data and tasks
TIE_BAND = 0.12   # |margin| within the band: the chamber is even, a dot for both


class World:
    """Hidden pairwise relations of the chapter house: one transitive direction
    w_q and one skew-symmetric matrix A_q per question. The same world serves
    every split of a seed, so shifts come from N, kappa and noise, not from
    new relations."""

    def __init__(self, rng):
        self.w = rng.normal(size=(N_QUESTIONS, D_TRAIT))
        self.w /= np.linalg.norm(self.w, axis=1, keepdims=True)
        raw = rng.normal(size=(N_QUESTIONS, D_TRAIT, D_TRAIT))
        self.A = (raw - np.transpose(raw, (0, 2, 1))) / np.sqrt(2.0 * D_TRAIT)

    def contests(self, z, q, kappa, noise, rng):
        """Hard pairwise outcomes beats[i, j] = 1 if i beats j, from the margin
        m(i,j) = w_q·(z_i − z_j) + kappa·z_iᵀ A_q z_j + symmetric noise;
        0.5 for both when the margin lies inside the tie band."""
        lin = z @ self.w[q]
        margin = lin[:, None] - lin[None, :] + kappa * (z @ self.A[q] @ z.T)
        e = rng.normal(size=margin.shape) * noise
        margin = margin + (e - e.T) / np.sqrt(2.0)
        # AEP f.12r: a dot for the winner; if the cell is even, a dot for both
        beats = (margin > TIE_BAND).astype(np.float64) + 0.5 * (np.abs(margin) <= TIE_BAND)
        np.fill_diagonal(beats, 0.0)
        return beats


def make_episodes(world, rng, n_episodes, n_range, kappa_range, noise, max_redraw=12):
    """Episodes with a unique Copeland winner. Items are shuffled, an episode
    offset shifts every candidate, traits are observed with jitter, two
    noise traits are appended and the question is a one-hot block on each item."""
    X, Q, Y, B = [], [], [], []
    while len(X) < n_episodes:
        n = int(rng.integers(n_range[0], n_range[1] + 1))
        q = int(rng.integers(N_QUESTIONS))
        kappa = float(rng.uniform(*kappa_range))
        offset = rng.normal(size=(1, D_TRAIT)) * 0.5   # an episode's shared "culture"
        for _ in range(max_redraw):
            z = rng.normal(size=(n, D_TRAIT)) + offset
            beats = world.contests(z, q, kappa, noise, rng)
            score = beats.sum(axis=1)
            if np.sum(score == score.max()) == 1:
                break
        else:
            continue
        perm = rng.permutation(n)
        z, beats = z[perm], beats[perm][:, perm]
        jitter = rng.normal(size=z.shape) * 0.05           # observation noise on the traits
        feats = np.concatenate([z + jitter, rng.normal(size=(n, D_NOISE))], axis=1)
        onehot = np.zeros((n, N_QUESTIONS))
        onehot[:, q] = 1.0
        X.append(np.concatenate([feats, onehot], axis=1))
        Q.append(q)
        Y.append(int(np.argmax(beats.sum(axis=1))))
        B.append(beats)
    return {"X": X, "q": np.array(Q), "y": np.array(Y), "beats": B}


def pad_batch(episodes, idx):
    """Stack episodes of unequal N into (E, Nmax, d) with a mask."""
    xs = [episodes["X"][i] for i in idx]
    nmax = max(x.shape[0] for x in xs)
    X = np.zeros((len(xs), nmax, xs[0].shape[1]))
    mask = np.zeros((len(xs), nmax))
    for e, x in enumerate(xs):
        X[e, :x.shape[0]] = x
        mask[e, :x.shape[0]] = 1.0
    return {"X": X, "mask": mask, "y": episodes["y"][idx], "beats": [episodes["beats"][i] for i in idx]}


SPLIT_SPECS = {
    "train":   dict(n=2400, n_range=(5, 7), kappa_range=(0.4, 1.2), noise=0.25),
    "val":     dict(n=400,  n_range=(5, 7), kappa_range=(0.4, 1.2), noise=0.25),
    "test":    dict(n=600,  n_range=(5, 7), kappa_range=(0.4, 1.2), noise=0.25),
    "shifted": dict(n=600,  n_range=(9, 9), kappa_range=(1.2, 1.8), noise=0.25),
    "blind":   dict(n=600,  n_range=(5, 7), kappa_range=(0.0, 0.0), noise=0.60),
}


def make_splits(seed, quick=False):
    ss = np.random.SeedSequence(seed)
    children = ss.spawn(1 + len(SPLIT_SPECS))
    world = World(np.random.default_rng(children[0]))
    splits = {}
    for (name, spec), child in zip(SPLIT_SPECS.items(), children[1:]):
        n = spec["n"] // 3 if quick else spec["n"]
        splits[name] = make_episodes(world, np.random.default_rng(child), n,
                                     spec["n_range"], spec["kappa_range"], spec["noise"])
    return world, splits


def episode_hashes(episodes):
    return {hashlib.sha1(np.round(x, 6).tobytes()).hexdigest() for x in episodes["X"]}


def trivial_accuracy(episodes):
    """Majority position after shuffling is the trivial baseline: ~ mean 1/N."""
    counts = np.bincount(episodes["y"], minlength=9)
    return float(counts.max() / len(episodes["y"]))


# ---------------------------------------------------------------- model
TASK_TYPES = ["sequence_classification"]
NO_TIE_LOGIT = -30.0


def build_model(in_dim, out_dim, task_type, rng, kind="chambers", agg="count", rank=RANK,
                n_questions=N_QUESTIONS, hidden=50, **cfg):
    """kind='chambers': the signature model (agg 'count' = 1283 exhaustive count,
    agg 'elimination' = the 1299 rival). kind='pointwise': size-matched scorer."""
    assert task_type in TASK_TYPES
    d = in_dim - n_questions
    s = 1.0 / np.sqrt(d)
    P = {}
    if kind == "chambers":
        P["Win"] = np.eye(d) + 0.01 * rng.normal(size=(d, d))
        P["bin"] = np.zeros(d)
        P["Wa"] = rng.normal(size=(d, rank)) * s
        P["ba"] = np.zeros(rank)
        P["Wb"] = rng.normal(size=(d, rank)) * s
        P["bb"] = np.zeros(rank)
        P["M"] = rng.normal(size=(n_questions, rank, rank)) / rank
        P["S"] = rng.normal(size=(n_questions, rank, rank)) / rank
        P["tie"] = np.array([-1.0])
        P["beta"] = np.array([4.0])
    else:
        P["W1"] = rng.normal(size=(d, hidden)) * s
        P["b1"] = np.zeros(hidden)
        P["W2"] = rng.normal(size=(n_questions, hidden)) / np.sqrt(hidden)
    return {"params": P, "cfg": {"d": d, "n_questions": n_questions, "rank": rank, "hidden": hidden},
            "kind": kind, "agg": agg, "knock": {}}


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


def _question_index(model, X):
    d, nq = model["cfg"]["d"], model["cfg"]["n_questions"]
    if X.shape[-1] > d:
        return np.argmax(X[..., d:d + nq].mean(axis=1), axis=1)
    return np.zeros(X.shape[0], dtype=int)


def _pair_mask(mask):
    n = mask.shape[1]
    return mask[:, :, None] * mask[:, None, :] * (1.0 - np.eye(n))[None]


def _chamber_probs(model, X, mask):
    """Forward pass up to the three-way chamber verdict; returns a cache."""
    P, K, d = model["params"], model["knock"], model["cfg"]["d"]
    feats = X[..., :d]
    q = _question_index(model, X)
    Xp = feats if K.get("vestibule") == "identity" else feats @ P["Win"] + P["bin"]
    A = np.tanh(Xp @ P["Wa"] + P["ba"])
    shared = K.get("correlatives") == "identity"
    B = A if shared else np.tanh(Xp @ P["Wb"] + P["bb"])
    M = np.broadcast_to(P["M"].mean(axis=0), (X.shape[0],) + P["M"].shape[1:]) if K.get("act_tensor") == "mean" else P["M"][q]
    S = P["S"][q]
    G1 = np.einsum("enr,ers,ems->enm", A, M, B)
    G2 = np.einsum("enr,ers,ems->enm", A, S, B)
    U = G1 - np.transpose(G1, (0, 2, 1))
    V = G2 + np.transpose(G2, (0, 2, 1)) + P["tie"][0]
    if K.get("chambers") == "zero":
        U = np.zeros_like(U)
    elif K.get("chambers") == "mean":
        U = np.broadcast_to(U.mean(axis=0, keepdims=True), U.shape).copy()
    if K.get("concordance") == "zero":
        V = np.full_like(V, NO_TIE_LOGIT)
    L = np.stack([U, -U, V], axis=-1)
    Pk = stable_softmax(L, axis=-1)
    return {"feats": feats, "q": q, "Xp": Xp, "A": A, "B": B, "shared": shared, "M": M, "S": S,
            "U": U, "V": V, "Pk": Pk, "pm": _pair_mask(mask), "mask": mask}


def _count_logits(model, c):
    p1, p3 = c["Pk"][..., 0], c["Pk"][..., 2]
    score = np.sum(c["pm"] * (p1 + 0.5 * p3), axis=2)
    return model["params"]["beta"][0] * score, score


def _elimination_logits(model, c):
    """The 1299 rule: a soft champion meets each next entrant in presentation
    order; the loser's weight is shifted onto the winner."""
    p1, mask = c["Pk"][..., 0], c["mask"]
    E, n = mask.shape
    w = np.zeros((E, n))
    w[:, 0] = 1.0
    traj, alphas = [w.copy()], []
    for t in range(1, n):
        a = np.sum(w * p1[:, :, t], axis=1)
        valid = mask[:, t]
        w_new = a[:, None] * w
        w_new[:, t] += 1.0 - a
        w = np.where(valid[:, None] > 0, w_new, w)
        traj.append(w.copy())
        alphas.append(a)
    return model["params"]["beta"][0] * w, {"traj": traj, "alphas": alphas, "w": w}


def _pointwise_forward(model, X, mask):
    P, d = model["params"], model["cfg"]["d"]
    feats = X[..., :d]
    q = _question_index(model, X)
    H = np.tanh(feats @ P["W1"] + P["b1"])
    logits = np.einsum("enh,eh->en", H, P["W2"][q])
    return logits, {"feats": feats, "q": q, "H": H}


def forward(model, X, mask):
    if model["kind"] == "pointwise":
        logits, cache = _pointwise_forward(model, X, mask)
    else:
        cache = _chamber_probs(model, X, mask)
        if model["agg"] == "elimination":
            logits, extra = _elimination_logits(model, cache)
        else:
            logits, extra = _count_logits(model, cache)
        cache["extra"] = extra
    return logits + (1.0 - mask) * (-1e9), cache


def predict(model, X, mask=None):
    if mask is None:
        mask = np.ones(X.shape[:2])
    logits, _ = forward(model, X, mask)
    return np.argmax(logits, axis=1)


def hidden_states(model, X, mask=None):
    if mask is None:
        mask = np.ones(X.shape[:2])
    if model["kind"] == "pointwise":
        return _pointwise_forward(model, X, mask)[1]["H"]
    return _chamber_probs(model, X, mask)["Pk"]


def loss_and_grads(model, batch):
    """Mean cross-entropy over episodes with hand-derived gradients for every
    parameter tensor. Returns (loss, grads with exactly the keys of params)."""
    X, mask, y = batch["X"], batch["mask"], batch["y"]
    logits, c = forward(model, X, mask)
    E = X.shape[0]
    prob = stable_softmax(logits, axis=1)
    loss = float(-np.mean(np.log(prob[np.arange(E), y] + 1e-300)))
    dlogits = prob.copy()
    dlogits[np.arange(E), y] -= 1.0
    dlogits /= E
    if model["kind"] == "pointwise":
        return loss, _pointwise_backward(model, c, dlogits)
    return loss, _chamber_backward(model, c, dlogits)


def _pointwise_backward(model, c, dlogits):
    P, q, H, feats = model["params"], c["q"], c["H"], c["feats"]
    g = {k: np.zeros_like(v) for k, v in P.items()}
    np.add.at(g["W2"], q, np.einsum("en,enh->eh", dlogits, H))
    dH = dlogits[:, :, None] * P["W2"][q][:, None, :]
    dpre = dH * (1.0 - H * H)
    g["W1"] = np.einsum("end,enh->dh", feats, dpre)
    g["b1"] = dpre.sum(axis=(0, 1))
    return g


def _chamber_backward(model, c, dlogits):
    P, K = model["params"], model["knock"]
    g = {k: np.zeros_like(v) for k, v in P.items()}
    p1, p2, p3, pm = c["Pk"][..., 0], c["Pk"][..., 1], c["Pk"][..., 2], c["pm"]
    dp = np.zeros_like(c["Pk"])
    if model["agg"] == "elimination":
        traj, alphas, w = c["extra"]["traj"], c["extra"]["alphas"], c["extra"]["w"]
        g["beta"][0] = float(np.sum(dlogits * w))
        dw = P["beta"][0] * dlogits
        for t in range(len(alphas), 0, -1):
            valid = c["mask"][:, t][:, None]
            a, w_prev = alphas[t - 1], traj[t - 1]
            e_t = np.zeros_like(w_prev)
            e_t[:, t] = 1.0
            da = np.sum(dw * valid * (w_prev - e_t), axis=1)
            dw_prev = valid * a[:, None] * dw + (1.0 - valid) * dw + da[:, None] * p1[:, :, t]
            dp[:, :, t, 0] += da[:, None] * w_prev
            dw = dw_prev
    else:
        score = c["extra"]
        g["beta"][0] = float(np.sum(dlogits * score))
        dscore = P["beta"][0] * dlogits
        dp[..., 0] = pm * dscore[:, :, None]
        dp[..., 2] = 0.5 * pm * dscore[:, :, None]
    dot = np.sum(c["Pk"] * dp, axis=-1, keepdims=True)
    dL = c["Pk"] * (dp - dot)
    dU = dL[..., 0] - dL[..., 1]
    dV = dL[..., 2]
    if K.get("chambers") in ("zero", "mean"):
        dU = np.zeros_like(dU) if K["chambers"] == "zero" else np.broadcast_to(dU.mean(axis=0, keepdims=True), dU.shape)
    if K.get("concordance") == "zero":
        dV = np.zeros_like(dV)
    else:
        g["tie"][0] = float(np.sum(dV))
    dG1 = dU - np.transpose(dU, (0, 2, 1))
    dG2 = dV + np.transpose(dV, (0, 2, 1))
    A, B, M, S, q = c["A"], c["B"], c["M"], c["S"], c["q"]
    dA = np.einsum("enm,ems,ers->enr", dG1, B, M) + np.einsum("enm,ems,ers->enr", dG2, B, S)
    dB = np.einsum("enm,enr,ers->ems", dG1, A, M) + np.einsum("enm,enr,ers->ems", dG2, A, S)
    dM = np.einsum("enr,enm,ems->ers", A, dG1, B)
    dS = np.einsum("enr,enm,ems->ers", A, dG2, B)
    if K.get("act_tensor") == "mean":
        g["M"] += dM.sum(axis=0)[None] / P["M"].shape[0]
    else:
        np.add.at(g["M"], q, dM)
    np.add.at(g["S"], q, dS)
    Xp = c["Xp"]
    if c["shared"]:
        dA = dA + dB
        dpreA = dA * (1.0 - A * A)
        dpreB = np.zeros_like(dpreA)
    else:
        dpreA = dA * (1.0 - A * A)
        dpreB = dB * (1.0 - B * B)
    g["Wa"] = np.einsum("end,enr->dr", Xp, dpreA)
    g["ba"] = dpreA.sum(axis=(0, 1))
    g["Wb"] = np.einsum("end,enr->dr", Xp, dpreB)
    g["bb"] = dpreB.sum(axis=(0, 1))
    if K.get("vestibule") != "identity":
        dXp = dpreA @ P["Wa"].T + dpreB @ P["Wb"].T
        g["Win"] = np.einsum("end,enk->dk", c["feats"], dXp)
        g["bin"] = dXp.sum(axis=(0, 1))
    return g


# ---------------------------------------------------------------- registries
MODULES = {
    "vestibule":    {"params": ["Win", "bin"], "role": "input affine map (identity-initialised)", "signature": False},
    "correlatives": {"params": ["Wa", "ba", "Wb", "bb"], "role": "two role-specific tanh projections (agent, patient)", "signature": True},
    "act_tensor":   {"params": ["M"], "role": "question-indexed bilinear form, antisymmetrised", "signature": True},
    "concordance":  {"params": ["S", "tie"], "role": "question-indexed symmetric bilinear form feeding a tie logit", "signature": True},
    "chambers":     {"params": [], "role": "antisymmetric pairwise contest logits", "signature": True},
    "chamber_count": {"params": ["beta"], "role": "sum of pairwise win probabilities plus half the tie probabilities", "signature": True},
}
KNOCKOUT_MODES = {"vestibule": "identity", "correlatives": "identity", "act_tensor": "mean",
                  "concordance": "zero", "chambers": "mean"}


def modules(model):
    return {name: dict(spec, tensors={k: model["params"][k] for k in spec["params"]}) for name, spec in MODULES.items()}


def knockout(model, name, mode):
    assert name in KNOCKOUT_MODES and mode == KNOCKOUT_MODES[name], f"unsupported knockout {name}:{mode}"
    copy = {"params": {k: v.copy() for k, v in model["params"].items()}, "cfg": dict(model["cfg"]),
            "kind": model["kind"], "agg": model["agg"], "knock": dict(model["knock"])}
    copy["knock"][name] = mode
    return copy


MUTANTS = {
    "sign_flip": "gradient step taken uphill (Adam sign +1)",
    "zero_lr": "learning rate set to zero",
    "grad_zero_Wa": "analytic gradient of Wa replaced by zeros",
}


# ---------------------------------------------------------------- training
def iterate_minibatches(episodes, batch_size, rng):
    idx = rng.permutation(len(episodes["y"]))
    for s in range(0, len(idx), batch_size):
        yield pad_batch(episodes, idx[s:s + batch_size])


def evaluate(model, episodes, batch_size=200):
    correct, n = 0, len(episodes["y"])
    for s in range(0, n, batch_size):
        b = pad_batch(episodes, np.arange(s, min(n, s + batch_size)))
        correct += int(np.sum(predict(model, b["X"], b["mask"]) == b["y"]))
    return correct / n


def mutated_loss_and_grads(model, batch, mutant=None):
    """The audit's view of the gradient: a registered mutant may corrupt it."""
    loss, grads = loss_and_grads(model, batch)
    if mutant == "grad_zero_Wa" and "Wa" in grads:
        grads["Wa"] = np.zeros_like(grads["Wa"])
    return loss, grads


def fit(model, data, budget, rng, lr=1e-2, batch_size=96, mutant=None, log_every=0):
    """Adam under a fixed update budget with global-norm clipping; the best
    validation accuracy selects the returned parameters."""
    P = model["params"]
    opt = Adam(P, lr=0.0 if mutant == "zero_lr" else lr)
    sign = +1.0 if mutant == "sign_flip" else -1.0
    steps, history, best = 0, [], (-1.0, None)
    while steps < budget:
        for batch in iterate_minibatches(data["train"], batch_size, rng):
            loss, grads = mutated_loss_and_grads(model, batch, mutant)
            if not np.isfinite(loss):
                raise FloatingPointError("non-finite loss")
            grads, _ = clip_global_norm(grads, CLIP_NORM)
            opt.step(P, grads, sign=sign)
            history.append(loss)
            steps += 1
            if log_every and steps % log_every == 0:
                print(f"    step {steps:4d} loss {loss:.4f}")
            if steps >= budget:
                break
        val = evaluate(model, data["val"])
        if val > best[0]:
            best = (val, {k: v.copy() for k, v in P.items()})
    if best[1] is not None:
        model["params"].update(best[1])
    return {"loss_history": history, "val_acc": best[0]}


def train_variant(kind, agg, data, budget, seed, mutant=None):
    ss = np.random.SeedSequence([seed, {"chambers": 1, "pointwise": 2}[kind], {"count": 1, "elimination": 2}[agg]])
    r_init, r_fit = [np.random.default_rng(s) for s in ss.spawn(2)]
    model = build_model(D_IN + N_QUESTIONS, 9, "sequence_classification", r_init, kind=kind, agg=agg)
    info = fit(model, data, budget, r_fit, mutant=mutant)
    return model, info


# ---------------------------------------------------------------- correctness tests
def grad_check(model, batch, rng, mutant=None):
    _, grads = mutated_loss_and_grads(model, batch, mutant)
    return finite_difference_check(lambda: loss_and_grads(model, batch)[0], model["params"], grads, rng)


def test_c1_gradcheck(data, rng, budget):
    worst, checked, total = 0.0, 0, 0
    for kind, agg in (("chambers", "count"), ("chambers", "elimination"), ("pointwise", "count")):
        model = build_model(D_IN + N_QUESTIONS, 9, "sequence_classification", rng, kind=kind, agg=agg)
        batch = pad_batch(data["train"], np.arange(12))
        w0, rep0 = grad_check(model, batch, rng)
        fit(model, data, min(budget, 60), rng)
        w1, rep1 = grad_check(model, batch, rng)
        worst = max(worst, w0, w1)
        checked += len(rep1)
        total += len(model["params"])
    return worst, checked, total


def test_c2_determinism(data):
    outs = []
    for _ in range(2):
        model, info = train_variant("chambers", "count", data, 30, seed=7)
        b = pad_batch(data["test"], np.arange(20))
        outs.append((np.array(info["loss_history"]), forward(model, b["X"], b["mask"])[0]))
    same = np.array_equal(outs[0][0], outs[1][0]) and np.array_equal(outs[0][1], outs[1][1])
    finite = all(np.all(np.isfinite(v)) for v in model["params"].values()) and np.all(np.isfinite(outs[1][1][b["mask"] > 0]))
    return bool(same and finite)


def test_c4_shuffled(data, budget, seed):
    rng = np.random.default_rng(seed + 1000)
    shuffled = dict(data)
    shuffled["train"] = dict(data["train"], y=rng.permutation(data["train"]["y"]))
    model, _ = train_variant("chambers", "count", shuffled, budget, seed)
    acc, triv = evaluate(model, data["test"]), trivial_accuracy(data["test"])
    return acc, triv


def test_c5_mutants(data, budget, seed, rng):
    detected = {}
    for name in MUTANTS:
        model, info = train_variant("chambers", "count", data, budget, seed, mutant=name)
        worst, _ = grad_check(model, pad_batch(data["train"], np.arange(12)), rng, mutant=name)
        hist = info["loss_history"]
        drop = (hist[0] - np.mean(hist[-10:])) / max(hist[0], 1e-9)
        acc = evaluate(model, data["test"])
        c1_fail = worst > MIND_CARD["thresholds"]["grad_rel_error"]
        c3_fail = drop < MIND_CARD["thresholds"]["loss_drop_fraction"] or acc < trivial_accuracy(data["test"]) + MIND_CARD["thresholds"]["margin_over_trivial"]
        detected[name] = bool(c1_fail or c3_fail)
    return detected


def test_c7_splits(data):
    names = list(data)
    for i, a in enumerate(names):
        ha = episode_hashes(data[a])
        for b in names[i + 1:]:
            if ha & episode_hashes(data[b]):
                return False
    return True


# ---------------------------------------------------------------- property tests (C6)
def _random_model(rng, agg="count"):
    m = build_model(D_IN + N_QUESTIONS, 9, "sequence_classification", rng, agg=agg)
    for k in m["params"]:
        m["params"][k] = m["params"][k] + rng.normal(size=m["params"][k].shape) * 0.5
    return m


def _add_position_bias(logits, strength=2.0):
    """Negative control for C6.1: a positional term breaks equivariance."""
    return logits + strength * np.arange(logits.shape[1])[None, :]


def test_c6_1_permutation_equivariance(data, rng, trials=40):
    m = _random_model(rng)
    b = pad_batch(data["test"], np.arange(8))
    worst, worst_ctrl = 0.0, 0.0
    for _ in range(trials):
        n = int(b["mask"][0].sum())
        perm = rng.permutation(n)
        X2, mask2 = b["X"][:1].copy(), b["mask"][:1]
        X2[0, :n] = X2[0, perm]
        l1 = forward(m, b["X"][:1], mask2)[0][0, :n]
        l2 = forward(m, X2, mask2)[0][0, :n]
        worst = max(worst, float(np.max(np.abs(l1[perm] - l2))))
        worst_ctrl = max(worst_ctrl, float(np.max(np.abs(_add_position_bias(l1[None])[0][perm] - _add_position_bias(l2[None])[0]))))
    return worst < 1e-9, worst, worst_ctrl > 1e-3


def test_c6_2_chamber_consistency(data, rng):
    """Definition check (holds by construction): p(i>j) = p(j>i)^T and the tie
    probability is symmetric."""
    m = _random_model(rng)
    b = pad_batch(data["test"], np.arange(8))
    Pk = hidden_states(m, b["X"], b["mask"])
    err = float(np.max(np.abs(Pk[..., 0] - np.transpose(Pk[..., 1], (0, 2, 1)))) +
                np.max(np.abs(Pk[..., 2] - np.transpose(Pk[..., 2], (0, 2, 1)))))
    return err < 1e-12


def test_c6_3_order_independence(data, rng, trials=60):
    """The count's decision never changes with presentation order; the 1299
    rival (negative control) must show at least one order-dependent decision."""
    mc, me = _random_model(rng, "count"), _random_model(rng, "elimination")
    me["params"] = {k: v.copy() for k, v in mc["params"].items()}
    b = pad_batch(data["test"], np.arange(30))
    flips_count, flips_elim = 0, 0
    for _ in range(trials):
        e = int(rng.integers(30))
        n = int(b["mask"][e].sum())
        perm = rng.permutation(n)
        X1, X2, mask = b["X"][e:e + 1], b["X"][e:e + 1].copy(), b["mask"][e:e + 1]
        X2[0, :n] = X2[0, perm]
        for model, counter in ((mc, "count"), (me, "elim")):
            w1 = int(predict(model, X1, mask)[0])
            w2 = int(perm[predict(model, X2, mask)[0]])
            if w1 != w2:
                if counter == "count":
                    flips_count += 1
                else:
                    flips_elim += 1
    return flips_count == 0, flips_elim > 0, flips_count, flips_elim


def test_c6_4_count_monotonicity(data, rng, trials=200):
    """Strengthening chamber (i,j) in i's favour never lowers i's count score;
    negative control: the successive-elimination weight can fall."""
    mc, me = _random_model(rng, "count"), _random_model(rng, "elimination")
    b = pad_batch(data["test"], np.arange(30))
    violations, ctrl_violations = 0, 0
    for _ in range(trials):
        e = int(rng.integers(30))
        n = int(b["mask"][e].sum())
        i, j = rng.choice(n, size=2, replace=False)
        delta = float(rng.uniform(0.1, 3.0))
        base = _chamber_probs(mc, b["X"][e:e + 1], b["mask"][e:e + 1])
        for model, is_ctrl in ((mc, False), (me, True)):
            c = dict(base)
            U = c["U"].copy()
            U[0, i, j] += delta
            U[0, j, i] -= delta
            Pk = stable_softmax(np.stack([U, -U, c["V"]], axis=-1), axis=-1)
            c2 = dict(c, Pk=Pk)
            s_before = (_elimination_logits if is_ctrl else _count_logits)(model, c)[0][0]
            s_after = (_elimination_logits if is_ctrl else _count_logits)(model, c2)[0][0]
            if s_after[i] < s_before[i] - 1e-9:
                if is_ctrl:
                    ctrl_violations += 1
                else:
                    violations += 1
    return violations == 0, ctrl_violations > 0, violations, ctrl_violations


# ---------------------------------------------------------------- diagnostics
def chamber_fidelity(model, episodes, n_max=200):
    """How faithfully the learned chambers reproduce the hidden pairwise
    outcomes they were never shown, and the calibration of p(i>j)."""
    hits, tot, confs, outs = 0, 0, [], []
    for s in range(0, min(n_max, len(episodes["y"])), 100):
        b = pad_batch(episodes, np.arange(s, min(s + 100, min(n_max, len(episodes["y"])))))
        Pk = hidden_states(model, b["X"], b["mask"])
        for e, beats in enumerate(b["beats"]):
            n = beats.shape[0]
            p = Pk[e, :n, :n, 0] + 0.5 * Pk[e, :n, :n, 2]
            iu = np.triu_indices(n, 1)
            hits += int(np.sum((p[iu] > 0.5) == (beats[iu] > 0.5)))
            tot += len(iu[0])
            confs.extend(p[iu].tolist())
            outs.extend(beats[iu].tolist())
    confs, outs = np.array(confs), np.array(outs)
    bins = np.clip((confs * 10).astype(int), 0, 9)
    ece = sum(np.sum(bins == k) / len(confs) * abs(np.mean(outs[bins == k]) - np.mean(confs[bins == k]))
              for k in range(10) if np.any(bins == k))
    return hits / tot, float(ece)


# ---------------------------------------------------------------- protocol
def run_seed(seed, budget, quick):
    _, data = make_splits(seed, quick)
    out = {"seed": seed, "trivial": {k: trivial_accuracy(v) for k, v in data.items()}}
    full, info = train_variant("chambers", "count", data, budget, seed)
    base, _ = train_variant("pointwise", "count", data, budget, seed)
    rival, _ = train_variant("chambers", "elimination", data, budget, seed)
    out["loss_drop"] = (info["loss_history"][0] - np.mean(info["loss_history"][-10:])) / info["loss_history"][0]
    for name, model in (("full", full), ("base", base), ("rival", rival)):
        out[name] = {split: evaluate(model, data[split]) for split in ("test", "shifted", "blind")}
    out["knockouts"] = {name: evaluate(knockout(full, name, mode), data["test"]) for name, mode in KNOCKOUT_MODES.items()}
    out["fidelity"], out["ece"] = chamber_fidelity(full, data["test"])
    out["n_params"] = {"full": n_params(full), "base": n_params(base)}
    return out, data


def run_protocol(args):
    t0 = time.time()
    quick = args.quick
    budget = 120 if quick else 700
    seeds = [args.seed] if quick else [args.seed + i for i in range(args.seeds)]
    print(f"chapter {CHAPTER:04d} · seeds {seeds} · quick={quick} · mutant={args.mutant}")
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, 99]))
    _, data0 = make_splits(args.seed, quick)
    correctness, exit_code = [], 0

    def record(cid, name, passed, detail):
        nonlocal exit_code
        correctness.append({"id": cid, "name": name, "passed": bool(passed), "detail": detail})
        if not passed:
            exit_code = 1
        print(f"  {cid} {name}: {'PASS' if passed else 'FAIL'} — {detail}")

    if args.mutant:
        # audit path: run the training under the mutant and report C1/C3 only
        model, info = train_variant("chambers", "count", data0, budget, args.seed, mutant=args.mutant)
        worst, _ = grad_check(model, pad_batch(data0["train"], np.arange(12)), rng, mutant=args.mutant)
        acc = evaluate(model, data0["test"])
        print(f"mutant {args.mutant}: gradcheck {worst:.2e}  test acc {acc:.3f}  trivial {trivial_accuracy(data0['test']):.3f}")
        return 1

    worst, checked, total = test_c1_gradcheck(data0, rng, budget)
    record("C1", "gradient_check", worst <= MIND_CARD["thresholds"]["grad_rel_error"],
           f"max rel error {worst:.2e} over {checked} tensor checks (init and after training)")
    record("C2", "determinism_finiteness", test_c2_determinism(data0), "same seed → identical losses and outputs; all finite")
    seed_results, datas = [], {}
    for s in seeds:
        res, data = run_seed(s, budget, quick)
        seed_results.append(res)
        datas[s] = data
        print(f"  seed {s}: full test {res['full']['test']:.3f} shifted {res['full']['shifted']:.3f} blind {res['full']['blind']:.3f} | "
              f"base test {res['base']['test']:.3f} shifted {res['base']['shifted']:.3f} blind {res['base']['blind']:.3f} | "
              f"rival test {res['rival']['test']:.3f} | fidelity {res['fidelity']:.3f} ece {res['ece']:.3f}")
    r0 = seed_results[0]
    th = MIND_CARD["thresholds"]
    record("C3", "learning", r0["loss_drop"] >= th["loss_drop_fraction"] and r0["full"]["test"] >= r0["trivial"]["test"] + th["margin_over_trivial"],
           f"loss drop {r0['loss_drop']:.2f} (≥{th['loss_drop_fraction']}); test acc {r0['full']['test']:.3f} vs trivial {r0['trivial']['test']:.3f} (+{th['margin_over_trivial']})")
    acc_sh, triv = test_c4_shuffled(data0, budget, args.seed)
    record("C4", "shuffled_label_control", abs(acc_sh - triv) <= th["shuffled_band"], f"shuffled acc {acc_sh:.3f} vs trivial {triv:.3f} (band ±{th['shuffled_band']})")
    detected = test_c5_mutants(data0, min(budget, 200), args.seed, rng)
    record("C5", "mutant_detection", all(detected.values()), f"{sum(detected.values())}/{len(detected)} detected: {detected}")
    ok, worst_eq, ctrl = test_c6_1_permutation_equivariance(data0, rng)
    record("C6.1", "permutation_equivariance", ok and ctrl, f"max deviation {worst_eq:.1e}; positional-bias control violates: {ctrl}")
    record("C6.2", "chamber_consistency(def)", test_c6_2_chamber_consistency(data0, rng), "definition check: p(i>j)=p(j>i)ᵀ, ties symmetric")
    ok, ctrl, fc, fe = test_c6_3_order_independence(data0, rng)
    record("C6.3", "order_independence", ok and ctrl, f"count flips {fc}; elimination control flips {fe}")
    ok, ctrl, v, cv = test_c6_4_count_monotonicity(data0, rng)
    record("C6.4", "count_monotonicity", ok and ctrl, f"count violations {v}; elimination control violations {cv}")
    record("C7", "split_integrity", test_c7_splits(data0), "no episode shared between splits")
    n_full, n_base = r0["n_params"]["full"], r0["n_params"]["base"]
    record("C7b", "size_match", abs(n_full - n_base) / n_full <= 0.10, f"full {n_full} vs pointwise {n_base}")

    runtime = time.time() - t0
    budget_s = QUICK_BUDGET_S if quick else DEFAULT_BUDGET_S
    record("C8", "budget", runtime <= budget_s, f"{runtime:.1f}s of {budget_s:.0f}s")
    if runtime > budget_s and exit_code == 0:
        exit_code = 3
    report = assemble_report(args, seeds, seed_results, correctness, detected, (worst, checked, total), runtime, exit_code, quick)
    write_report(report, args.json)
    return exit_code


def assemble_report(args, seeds, seed_results, correctness, detected, gradcheck, runtime, exit_code, quick):
    th, hyps = MIND_CARD["thresholds"], []
    worst, checked, total = gradcheck
    brng = np.random.default_rng(np.random.SeedSequence([args.seed, 2000]))
    for h in MIND_CARD["hypotheses"]:
        if h["id"] == "H-SIG":
            diffs = [r["full"]["shifted"] - r["base"]["shifted"] for r in seed_results]
        elif h["id"] == "H-NEC":
            diffs = [r["knockouts"]["act_tensor"] - r["knockouts"]["vestibule"] for r in seed_results]
        elif h["id"] == "H-BLIND":
            diffs = [r["full"]["blind"] - r["base"]["blind"] for r in seed_results]
        else:
            diffs = [r["full"]["test"] - r["rival"]["test"] for r in seed_results]
        mean, lo, hi = paired_bootstrap(diffs, brng)
        v = "not evaluated" if quick else verdict(mean, lo, hi, h["direction"], h["mesi"])
        hyps.append({"id": h["id"], "metric": h["metric"], "mean_diff": mean, "ci95": [lo, hi], "mesi": h["mesi"],
                     "n_seeds": len(diffs), "verdict": v})
    knocks = []
    for name in KNOCKOUT_MODES:
        diffs = [r["knockouts"][name] - r["full"]["test"] for r in seed_results]
        mean, lo, hi = paired_bootstrap(diffs, brng)
        knocks.append({"module": name, "signature": MODULES[name]["signature"], "metric_change": mean, "ci95": [lo, hi]})
    return {
        "schema_version": "1.0", "chapter": CHAPTER, "file": os.path.basename(__file__), "card_revision": MIND_CARD["card_revision"],
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
        "seeds": seeds, "runtime_s": runtime, "n_params": seed_results[0]["n_params"]["full"],
        "gradcheck": {"tensors_checked": checked, "tensors_total": total, "max_rel_error": worst,
                      "checked_at": ["init", "after_training_steps"], "passed": worst <= th["grad_rel_error"]},
        "correctness": correctness,
        "mutants": {"detected": sum(detected.values()), "total": len(detected), "score": sum(detected.values()) / len(detected)},
        "hypotheses": hyps, "knockouts": knocks,
        "diagnostics": {
            "mean_accuracy": {k: {sp: float(np.mean([r[k][sp] for r in seed_results])) for sp in ("test", "shifted", "blind")}
                              for k in ("full", "base", "rival")},
            "trivial_accuracy": {sp: float(np.mean([r["trivial"][sp] for r in seed_results])) for sp in ("test", "shifted", "blind")},
            "chamber_fidelity_vs_hidden_pairs": float(np.mean([r["fidelity"] for r in seed_results])),
            "chamber_ece": float(np.mean([r["ece"] for r in seed_results])),
        },
        "task_types": TASK_TYPES, "exit_code": exit_code,
    }


def load_csv_bridge(path):
    """Optional real-data bridge: CSV with columns episode,item,winner,f1..fk
    (one row per candidate). Not executed without --data; no downloads."""
    rows = [line.strip().split(",") for line in open(path, encoding="utf-8") if line.strip()]
    header, body = rows[0], rows[1:]
    ei, wi = header.index("episode"), header.index("winner")
    fcols = [i for i, h in enumerate(header) if h.startswith("f")]
    groups = {}
    for r in body:
        groups.setdefault(r[ei], []).append(([float(r[i]) for i in fcols], int(float(r[wi]))))
    X, y = [], []
    for g in groups.values():
        feats = np.array([f for f, _ in g])
        X.append(np.concatenate([feats, np.zeros((len(g), N_QUESTIONS))], axis=1))
        y.append(int(np.argmax([w for _, w in g])))
    return {"X": X, "q": np.zeros(len(X), dtype=int), "y": np.array(y), "beats": [np.zeros((len(g), len(g))) for g in groups.values()]}


def main(argv=None):
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__), description="Chapter 0339 · Ars Camerarum")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--json", default=None)
    ap.add_argument("--card", action="store_true")
    ap.add_argument("--mutant", default=None, choices=list(MUTANTS))
    ap.add_argument("--data", default=None)
    try:
        args = ap.parse_args(argv)
    except SystemExit:
        return 2
    if args.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    if args.data:
        if not os.path.exists(args.data):
            print(f"--data {args.data}: file not found; skipping the real-data bridge")
        else:
            episodes = load_csv_bridge(args.data)
            d = episodes["X"][0].shape[1] - N_QUESTIONS
            rng = np.random.default_rng(args.seed)
            model = build_model(d + N_QUESTIONS, 9, "sequence_classification", rng)
            n = len(episodes["y"])
            cut = int(0.8 * n)
            data = {"train": {k: (v[:cut] if isinstance(v, np.ndarray) else v[:cut]) for k, v in episodes.items()},
                    "val": {k: (v[cut:] if isinstance(v, np.ndarray) else v[cut:]) for k, v in episodes.items()}}
            fit(model, data, 300, rng)
            print(f"real-data bridge: held-out accuracy {evaluate(model, data['val']):.3f} on {n - cut} episodes")
            return 0
    try:
        return run_protocol(args)
    except FloatingPointError as exc:
        print(f"non-finite values: {exc}")
        return 4


if __name__ == "__main__":
    sys.exit(main())

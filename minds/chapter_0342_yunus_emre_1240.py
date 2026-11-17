#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0342 · Yunus Emre (1240-1320)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 18, Minds 341-360 Available on Amazon https://www.amazon.com/dp/B0HQ47T2NC
# Resume and Interactive Demos at https://artificiology.com/
# END ATTRIBUTION 
"""
The Refrain Engine — chapter 0342, Yunus Emre (c. 1240 – c. 1320).

Thesis
------
Knowledge is what survives being re-sung by people who cannot read: the test of
a doctrine is not whether one hearer understands it but whether it is still the
same doctrine after passing through a chain of memory-limited singers, and the
instrument that passes that test is a fixed-count stanza with rhymed line-ends
and one refrain repeated in every stanza.

Evidence and provenance (belief; the figure's own surviving verse)
------------------------------------------------------------------
D1  Reading without knowing is "dry labour" and the four scriptures are legible
    in one letter: the ilahi that opens "Ilim ilim bilmektir" (Divan, critical
    text ed. Mustafa Tatci, 1990/2008). Basis: primary.
D2  Love is "one long syllable": "Dort kitabin manasin okudum hasil ettim /
    Aska gelince gordum bir uzun hece imis" (Divan, ed. Tatci). Basis: primary.
D3  The koşma-form ilahis carry a refrain in every stanza, e.g. "bana seni gerek
    seni" (Divan, ed. Golpinarli 1965, p. 184; ed. Tatci). Basis: primary.
D4  The oldest manuscripts postdate him by 150-200 years and differ so widely
    that the editors conclude the poems were gathered from oral tradition
    (Tatci 1990; Horata, TEES entry on Risaletu'n-Nushiyye). Basis: scholarship.
D5  About 415 poems are accepted as his; some 300 more carry the name "Yunus"
    and belong to later poets, above all the 15th-century Bursali Asik Yunus
    (Tatci 1990, 1991; Timurtas 1986). Basis: scholarship.
D6  Rhyme, meter and refrain are the sound constraints that make oral recall
    reliable across generations (Rubin 1995, chs. 4-6). Basis: scholarship.

Doctrine -> mechanism -> test
----------------------------
| Doctrine | Mechanism | Property test | Hypothesis |
|---|---|---|---|
| D3, D6 rhymed line-ends repair each other | M1 meter: group-averaging projection on the code; the singer channel recovers any tied group from one surviving member | C6.1 tie, C6.2 recovery, C6.3 non-expansive | H-SIG, H-NEC |
| D1, D2 one letter/syllable holds the meaning | M2 refrain anchor: the pooled refrain modulates the hearer's hidden layer (FiLM) | C6.1 (refrain group) | H-NEC-ANCHOR |
| D4 the doctrine is re-learned by each generation from the previous one's song | iterated singer chain: the same encoder/decoder is unrolled G times, each hearer re-sings what it decoded | C3, C4 | H-SIG |
| D5 a form anyone can carry is a form anyone can forge | forged-stanza condition: a stanza sung from another doctrine in the same meter | -- | H-BLIND |

Research question
-----------------
Memory, retrieval and forgetting / robustness: does a fixed poetic form
(structured redundancy fixed before learning) slow the loss of information in
a chain of learners that each learn only from the previous learner's output,
and what does that form cost when a stranger's stanza enters the chain?

Closest prior art and the delta
-------------------------------
Neural iterated learning (Ren, Guo, Labeau, Cohen and Kirby, ICLR 2020) and
the transmission-chain experiments of Kirby, Cornish and Smith (PNAS 2008)
are the framework: a chain of learners with a bottleneck. Model collapse under
recursively generated data (Shumailov et al., Nature 2024) is the failure the
chain reproduces. The channel-level recovery of tied positions is a repetition
code over an erasure channel. Delta: the redundancy is a poetic FORM fixed
before learning (rhyme groups and a cross-stanza refrain), recovery happens in
the singer's memory rather than in the learned decoder, and the same form is
shown to be the vector of forgery. Overlap: Medium (known parts, new
combination, new test).

Blind spot
----------
The tradition could not keep his stanzas apart from those of later "Yunuses":
the refrain that makes the doctrine survive is pooled across stanzas, so one
stanza sung from a stranger's doctrine in the same meter corrupts the anchor
of every stanza. The full model is predicted to lose MORE than the free-verse
baseline when a forged stanza is inserted.

Task (generative process)
-------------------------
1. A doctrine z ~ N(0, I_6), normalised to unit length.
2. Generation 0 sings z into 3 stanzas x 4 lines x 3 slots (36 positions).
3. A singer forgets whole lines: each of the 12 lines is lost with probability
   p (train 0.3; shifted 0.5). A lost slot whose rhyme or refrain group has a
   surviving member is recovered exactly; otherwise the singer borrows the line
   from another song, unflagged, so the hearer cannot tell memory from borrowing.
4. The next hearer receives the 36 slots, decodes a doctrine estimate, answers
   a query q ~ N(0, I_6): y = [z . q > 0], then re-sings its estimate.
5. Steps 3-4 repeat for G generations (train 3; shifted 6). The loss is the
   mean binary cross-entropy over generations.
6. Forged split: from generation 1 on, one random stanza is replaced by a
   stanza sung (same singer, same meter) from an unrelated doctrine z'.
The target is never readable from a single input feature: every slot mixes
all six coordinates of z, and y depends on z through the query.

Limits
------
A research prototype of one mechanism on synthetic data; it measures
transmission fidelity under erasure, not poetry, and it makes no claim to
replicate a person. Aesthetic worth is not modelled.

Run: python3 <this file>   (--quick, --card, --seed S, --seeds N, --json PATH,
--mutant NAME, --data PATH)
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 1,
    "revision_log": [],
    "generation": {"template_version": "1.0", "generator": "Claude Fable 5.1",
                   "generator_version": "claude-fable-5-1", "date": "2026-09-26"},
    "id": 342, "figure": "Yunus Emre", "born": 1240, "died": 1320,
    "civilization": "Anatolian Turkish",
    "provenance": "belief",
    "thesis": ("Knowledge is what survives being re-sung by people who cannot read: a "
               "doctrine counts as known only when a chain of memory-limited singers "
               "still carries the same doctrine, and the instrument that passes the test "
               "is a fixed-count stanza with rhymed line-ends and a refrain in every stanza."),
    "evidence": [
        {"id": "D1", "claim": "Reading without knowing is dry labour; the four scriptures are legible in one letter (alif).",
         "basis": "primary", "source": "Divan, 'Ilim ilim bilmektir', ed. Tatci 1990/2008"},
        {"id": "D2", "claim": "Love, once reached, is 'one long syllable' (bir uzun hece).",
         "basis": "primary", "source": "Divan, ed. Tatci 1990/2008"},
        {"id": "D3", "claim": "The kosma-form ilahis repeat one refrain in every stanza (bana seni gerek seni).",
         "basis": "primary", "source": "Divan, ed. Golpinarli 1965 p. 184; ed. Tatci"},
        {"id": "D4", "claim": "The oldest copies postdate him by 150-200 years and were gathered from oral tradition.",
         "basis": "scholarship", "source": "Tatci 1990; Horata, TEES 'Risaletu'n-Nushiyye'"},
        {"id": "D5", "claim": "About 415 poems are his; some 300 'Yunus' poems belong to later poets (Bursali Asik Yunus).",
         "basis": "scholarship", "source": "Tatci 1990, 1991; Timurtas 1986"},
        {"id": "D6", "claim": "Rhyme, meter and refrain are the sound constraints that stabilise oral recall.",
         "basis": "scholarship", "source": "Rubin, Memory in Oral Traditions, 1995, chs. 4-6"},
    ],
    "research_question": {"category": "memory, retrieval and forgetting",
                          "question": ("Does a poetic form fixed before learning slow information loss in a chain "
                                       "of learners that each learn only from the previous learner's output, and "
                                       "what does the form cost when a forged stanza enters the chain?")},
    "mechanism": {
        "name": "Refrain Engine (iterated singer chain with rhyme recovery and refrain anchor)",
        "family": "iterated learning / noisy-channel autoencoder chain",
        "signature_modules": ["meter", "refrain_anchor"],
        "closest_prior_art": ["Ren et al. 2020 neural iterated learning (ICLR)",
                              "Kirby, Cornish & Smith 2008 transmission chains (PNAS)",
                              "Shumailov et al. 2024 model collapse (Nature 631:755-759)",
                              "repetition codes over an erasure channel"],
        "overlap": "Medium",
        "prior_art_queries": ["iterated learning neural agents compositionality",
                              "model collapse recursive generated data",
                              "rhyme meter error correcting code oral transmission",
                              "emergent communication noisy channel redundancy"],
        "contribution_type": "test",
        "delta": ("Redundancy is a poetic form fixed before learning (rhyme groups plus a "
                  "cross-stanza refrain); recovery happens in the singer's memory, not the "
                  "decoder; the same form is tested as the vector of forgery."),
    },
    "traceability": [
        {"doctrine": "D3", "mechanism": "M1", "property_test": "C6.1", "hypothesis": "H-SIG"},
        {"doctrine": "D6", "mechanism": "M1", "property_test": "C6.2", "hypothesis": "H-NEC"},
        {"doctrine": "D1", "mechanism": "M2", "property_test": "C6.1", "hypothesis": "H-NEC-ANCHOR"},
        {"doctrine": "D4", "mechanism": "chain", "property_test": "C3", "hypothesis": "H-SIG"},
        {"doctrine": "D5", "mechanism": "forged split", "property_test": "C7", "hypothesis": "H-BLIND"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "With longer chains and heavier forgetting, the metered model keeps more of the doctrine than the free-verse chain.",
         "metric": "accuracy_shifted", "split": "shifted", "comparison": "model - baseline",
         "direction": "greater", "mesi": 0.03, "seeds": 5},
        {"id": "H-NEC", "statement": "Removing the meter hurts more than removing the hearer's hidden nonlinearity.",
         "metric": "accuracy_shifted", "split": "shifted",
         "comparison": "knockout(meter,identity) - knockout(hearer_hidden,identity)",
         "direction": "less", "mesi": 0.03, "seeds": 5},
        {"id": "H-NEC-ANCHOR", "statement": "Removing the refrain anchor hurts more than removing the hearer's hidden nonlinearity.",
         "metric": "accuracy_shifted", "split": "shifted",
         "comparison": "knockout(refrain_anchor,mean) - knockout(hearer_hidden,identity)",
         "direction": "less", "mesi": 0.02, "seeds": 5},
        {"id": "H-BLIND", "statement": "A forged stanza in the same meter costs the metered model more accuracy than it costs the free-verse chain.",
         "condition": "forged", "grounding": "the tradition could not separate his stanzas from those of later Yunuses",
         "metric": "forgery_degradation", "comparison": "model - baseline",
         "direction": "greater", "mesi": 0.02, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.15, "margin_over_trivial": 0.12, "shuffled_band": 0.06,
                   "note": "borrowed lines are indistinguishable from remembered ones, so much of the initial loss is irreducible; "
                           "the shuffled-label control is judged on its seed mean because the six-weight answer head is "
                           "sign-correlated with the true labelling at random initialisation (single seeds range about 0.43-0.58)"},
    "probe_predictions": [
        {"probe": "P9", "expected": "above baseline"},
        {"probe": "P1", "expected": "below baseline"},
        {"probe": "P10", "expected": "equal to baseline"},
        {"probe": "P8", "expected": "equal to baseline"},
    ],
    "dialectic_links": [
        {"chapter": 330, "relation": "teacher", "test": "none implemented"},
        {"chapter": 286, "relation": "successor", "test": "none implemented"},
    ],
    "corpus_neighbors": [
        {"chapter": 286, "similarity": 0.2, "difference": "Yasawi keeps memory in a shared voiced room in the present; here the store is a form that survives generational re-learning."},
        {"chapter": 210, "similarity": 0.2, "difference": "Bai Juyi gates an utterance on one weak listener's comprehension; here no hearer is asked to understand, only to carry, over many generations."},
        {"chapter": 20, "similarity": 0.15, "difference": "Homer's meter constrains the composer's real-time recombination; here the meter is a transmission code and the composer is unconstrained."},
        {"chapter": 416, "similarity": 0.1, "difference": "Kabir scores an utterance by what it breaks in the hearer; here it is scored by what it keeps across hearers."},
        {"chapter": 330, "similarity": 0.1, "difference": "Rumi's content can only subtract from a carrier; here content is spread over a redundant form and pooled back."},
    ],
    "barometer": {
        "cognitive_processing": ["P10 sample efficiency"],
        "embodied_cognition": [],
        "world_modeling": ["free-running chain without input (P7-like)"],
        "consciousness": [],
        "language_understanding": ["validity of constrained generation (form constraint)"],
        "emotional_intelligence": [],
        "creativity": ["diversity of valid stanzas under a fixed form"],
        "autonomy": [],
    },
    "task_types": ["vector_classification"],
    "applications": [
        {"use": "Structured redundancy as a brake on collapse in recursive synthetic-data training",
         "sector": "machine-learning infrastructure", "dataset": "UCI Wine recognition (178 rows, 13 features; CSV)"},
        {"use": "Design of relayed instructions for low-literacy field workers, tested by chain-transmission fidelity",
         "sector": "public health and humanitarian response", "dataset": "Kirby, Cornish & Smith 2008 transmission-chain data (PNAS supplement)"},
        {"use": "Emergent communication between agents with lossy memory",
         "sector": "multi-agent AI", "dataset": "EGG reconstruction game (Facebook Research, public)"},
    ],
    "safety_notes": ("Synthetic data only. No claim of replicating a person; no generated text is "
                     "presented as the figure's words. Forgery is modelled as a data condition, "
                     "never as a method for producing forgeries."),
}

import argparse
import json
import math
import os
import sys
import time
import hashlib
import numpy as np

D, S, NL, M = 6, 3, 4, 3            # doctrine dim; stanzas; lines per stanza; slots per line
NPOS = S * NL * M                   # 36 code positions
NLINES = S * NL                     # 12 erasable lines
HE, HD, HD_BASE = 24, 32, 38        # encoder hidden; hearer hidden (full / size-matched baseline)
G_TRAIN, P_TRAIN = 3, 0.30
G_SHIFT, P_SHIFT = 6, 0.50
CLIP_NORM = 5.0
BUDGET_FULL_S, BUDGET_QUICK_S = 180.0, 20.0


# BEGIN STANDARD UTILITIES v1.0
def sigmoid(z):
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def log_sigmoid(z):
    return -np.logaddexp(0.0, -z)


def softplus(z):
    return np.logaddexp(0.0, z)


def logsumexp(z, axis=-1, keepdims=False):
    m = np.max(z, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(z - m), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


def stable_softmax(z, axis=-1):
    return np.exp(z - logsumexp(z, axis=axis, keepdims=True))


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads, sign=1.0):
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= sign * self.lr * mhat / (np.sqrt(vhat) + self.eps)


def clip_global_norm(grads, max_norm):
    total = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    if total > max_norm:
        scale = max_norm / (total + 1e-12)
        for k in grads:
            grads[k] = grads[k] * scale
    return total


def finite_difference_check(loss_fn, params, grads, rng, n_entries=20, eps=1e-6, tol=1e-5):
    """Central differences in float64 on >=20 random entries per tensor plus the
    largest-gradient entry. Relative error uses max(|a|+|b|, 1e-4) in the
    denominator: at eps=1e-6 central differences carry about 1e-10 absolute
    noise, so entries with gradients below 1e-4 are compared on that absolute
    scale rather than amplified to a meaningless ratio."""
    worst = 0.0
    checked = 0
    for k, p in params.items():
        flat = p.reshape(-1)
        n = flat.size
        idx = set(int(i) for i in rng.integers(0, n, size=min(n_entries, n)))
        idx.add(int(np.argmax(np.abs(grads[k]).reshape(-1))))
        for i in idx:
            old = flat[i]
            flat[i] = old + eps
            lp = loss_fn()
            flat[i] = old - eps
            lm = loss_fn()
            flat[i] = old
            fd = (lp - lm) / (2 * eps)
            an = grads[k].reshape(-1)[i]
            rel = abs(fd - an) / max(abs(fd) + abs(an), 1e-4)
            worst = max(worst, rel)
        checked += 1
    return {"tensors_checked": checked, "tensors_total": len(params),
            "max_rel_error": float(worst), "passed": bool(worst <= tol)}


def paired_bootstrap_ci(diffs, rng, n_resamples=2000, alpha=0.05):
    diffs = np.asarray(diffs, dtype=np.float64)
    n = diffs.size
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        means[i] = diffs[rng.integers(0, n, size=n)].mean()
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(diffs.mean()), [float(lo), float(hi)]


def verdict(mean, ci, direction, mesi):
    lo, hi = ci
    if direction == "greater":
        if lo > 0 and mean >= mesi:
            return "supported"
        if hi < 0:
            return "contradicted"
    else:
        if hi < 0 and -mean >= mesi:
            return "supported"
        if lo > 0:
            return "contradicted"
    return "inconclusive"


def write_report(report, path=None):
    ch = report["chapter"]
    lines = [f"=== VERIFIED REPORT · chapter {ch:04d} ===",
             f"file: {report['file']}   card_revision: {report['card_revision']}",
             f"environment: python {report['environment']['python']}  numpy {report['environment']['numpy']}",
             f"seeds: {report['seeds']}   runtime_s: {report['runtime_s']:.1f}   n_params: {report['n_params']}",
             (f"gradcheck: {report['gradcheck']['tensors_checked']}/{report['gradcheck']['tensors_total']} tensors, "
              f"max_rel_error {report['gradcheck']['max_rel_error']:.2e}, checked_at {report['gradcheck']['checked_at']}, "
              f"passed={report['gradcheck']['passed']}"),
             "correctness:"]
    for c in report["correctness"]:
        lines.append(f"  {c['id']:<6} {c['name']:<28} {'PASS' if c['passed'] else 'FAIL'}  {c['detail']}")
    mu = report["mutants"]
    lines.append(f"mutants: detected {mu['detected']}/{mu['total']}  score {mu['score']:.2f}")
    lines.append("hypotheses:")
    for h in report["hypotheses"]:
        lines.append(f"  {h['id']:<13} {h['metric']:<22} mean {h['mean_diff']:+.4f}  ci95 [{h['ci95'][0]:+.4f}, {h['ci95'][1]:+.4f}]"
                     f"  mesi {h['mesi']:.2f}  n {h['n_seeds']}  -> {h['verdict']}")
    lines.append("knockouts (accuracy change on the shifted split, mean over seeds):")
    for k in report["knockouts"]:
        lines.append(f"  {k['module']:<16} signature={str(k['signature']):<5} change {k['metric_change']:+.4f}"
                     f"  ci95 [{k['ci95'][0]:+.4f}, {k['ci95'][1]:+.4f}]")
    lines.append(f"task_types: {report['task_types']}   exit_code: {report['exit_code']}")
    lines.append("=== END REPORT ===")
    print("\n".join(lines))
    if path:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
# END STANDARD UTILITIES


# ----------------------------------------------------------------------------
# 6. Data and tasks
# ----------------------------------------------------------------------------
def position_index(s, l, m):
    return (s * NL + l) * M + m


def line_of_position():
    return np.array([(s * NL + l) for s in range(S) for l in range(NL) for _ in range(M)])


def metered_groups():
    """Rhyme groups: the last slot of lines 1-3 in each stanza is one group.
    Refrain groups: slot m of line 4 is one group across all stanzas.
    Every other slot (the body of lines 1-3) is a singleton."""
    groups = []
    for s in range(S):
        for l in range(NL - 1):
            for m in range(M - 1):
                groups.append([position_index(s, l, m)])
        groups.append([position_index(s, l, M - 1) for l in range(NL - 1)])
    for m in range(M):
        groups.append([position_index(s, NL - 1, m) for s in range(S)])
    return groups


def free_groups():
    return [[i] for i in range(NPOS)]


def membership(groups):
    gm = np.zeros((NPOS, len(groups)))
    for k, members in enumerate(groups):
        gm[members, k] = 1.0
    return gm


REFRAIN_IDX = [position_index(0, NL - 1, m) for m in range(M)]
LINE_OF_POS = line_of_position()


def unit_rows(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def make_split(n, g, p, rng, forged=False):
    z = unit_rows(rng.standard_normal((n, D)))
    q = unit_rows(rng.standard_normal((g, n, D)))
    y = (np.einsum("nd,gnd->gn", z, q) > 0).astype(np.float64)
    line_keep = (rng.random((g, n, NLINES)) >= p).astype(np.float64)
    mask = line_keep[:, :, LINE_OF_POS]
    perm = np.stack([rng.permutation(n) for _ in range(g)])
    batch = {"Z": z, "Q": q, "Y": y, "MASK": mask, "PERM": perm, "G": g}
    if forged:
        batch["FORGE_Z"] = unit_rows(rng.standard_normal((n, D)))
        batch["FORGE_STANZA"] = rng.integers(0, S, size=n)
    return batch


def resample_masks(batch, p, rng):
    """A fresh act of forgetting for every recital: the doctrine and the
    queries of an episode are fixed, the erased lines are not."""
    out = dict(batch)
    n = batch["Z"].shape[0]
    out["MASK"] = (rng.random((batch["G"], n, NLINES)) >= p).astype(np.float64)[:, :, LINE_OF_POS]
    out["PERM"] = np.stack([rng.permutation(n) for _ in range(batch["G"])])
    return out


def rng_free_perm(perm, idx):
    """Restrict a permutation of the full split to the drawn rows: row i borrows
    from the drawn row that holds the same rank, so a minibatch stays closed."""
    order = np.argsort(perm[idx])
    return order[(np.arange(len(idx)) + 1) % len(idx)]


def slice_batch(batch, idx):
    out = {"G": batch["G"], "Z": batch["Z"][idx], "Q": batch["Q"][:, idx], "Y": batch["Y"][:, idx],
           "MASK": batch["MASK"][:, idx], "PERM": np.stack([rng_free_perm(batch["PERM"][g], idx) for g in range(batch["G"])])}
    for k in ("FORGE_Z", "FORGE_STANZA"):
        if k in batch:
            out[k] = batch[k][idx]
    return out


def make_data(rng, n_train=2048, n_val=512, n_test=1024):
    r = [np.random.default_rng(s) for s in np.random.SeedSequence(int(rng.integers(0, 2**31))).spawn(5)]
    return {"train": make_split(n_train, G_TRAIN, P_TRAIN, r[0]),
            "val": make_split(n_val, G_TRAIN, P_TRAIN, r[1]),
            "test": make_split(n_test, G_TRAIN, P_TRAIN, r[2]),
            "shifted": make_split(n_test, G_SHIFT, P_SHIFT, r[3]),
            "forged": make_split(n_test, G_TRAIN, P_TRAIN, r[4], forged=True)}


def row_hashes(z):
    return {hashlib.sha1(np.round(r, 9).tobytes()).hexdigest() for r in z}


def shuffled_labels(batch, rng):
    out = dict(batch)
    out["Y"] = np.stack([batch["Y"][g][rng.permutation(batch["Y"].shape[1])] for g in range(batch["G"])])
    return out


# ----------------------------------------------------------------------------
# 7. Model
# ----------------------------------------------------------------------------
TASK_TYPES = ["vector_classification"]


def build_model(in_dim=2 * D, out_dim=1, task_type="vector_classification", rng=None,
                use_meter=True, use_anchor=True, hearer_hidden=None, query_dim=D, **cfg):
    if task_type not in TASK_TYPES:
        raise ValueError(task_type)
    rng = rng if rng is not None else np.random.default_rng(0)
    doctrine_dim = in_dim - query_dim
    hd = hearer_hidden if hearer_hidden is not None else (HD if use_anchor else HD_BASE)

    def w(shape, scale):
        return rng.standard_normal(shape) * scale
    params = {
        "enc_W1": w((doctrine_dim, HE), 1 / math.sqrt(doctrine_dim)), "enc_b1": np.zeros(HE),
        "enc_W2": w((HE, NPOS), 1 / math.sqrt(HE)), "enc_b2": np.zeros(NPOS),
        "dec_W3": w((NPOS, hd), 1 / math.sqrt(NPOS)), "dec_b3": np.zeros(hd),
        "dec_W4": w((hd, doctrine_dim), 1 / math.sqrt(hd)), "dec_b4": np.zeros(doctrine_dim),
    }
    if use_anchor:
        params["film_W"] = w((M, 2 * hd), 0.3 / math.sqrt(M))
        params["film_b"] = np.zeros(2 * hd)
    if query_dim > 0:
        params["head_w"] = np.ones(doctrine_dim)
        params["head_b"] = np.zeros(1)
    else:
        params["head_W"] = w((doctrine_dim, out_dim), 1 / math.sqrt(doctrine_dim))
        params["head_b"] = np.zeros(out_dim)
    groups = metered_groups() if use_meter else free_groups()
    return {"params": params, "cfg": {"use_meter": use_meter, "use_anchor": use_anchor, "hd": hd,
                                     "query_dim": query_dim, "doctrine_dim": doctrine_dim, "out_dim": out_dim,
                                     "p_infer": P_TRAIN, "g_infer": G_TRAIN},
            "groups": groups, "ko": {}, "mutant": None}


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


def projector(model):
    if model["ko"].get("meter") == "identity" or not model["cfg"]["use_meter"]:
        gm = membership(free_groups())
    else:
        gm = membership(model["groups"])
    counts = gm.sum(0)
    return gm, gm @ np.diag(1.0 / counts) @ gm.T


def channel_forward(h, mask, gm, perm):
    """The singer's memory. A forgotten position is recovered exactly from any
    surviving member of its rhyme or refrain group. A group with no survivor is
    not left silent: the singer borrows the corresponding line from another
    song (episode perm[i]), unflagged, so the hearer cannot tell memory from
    substitution, and the borrowed group keeps its rhyme."""
    counts = gm.sum(0)
    s = mask @ gm
    alive = (s > 0).astype(np.float64)
    s_safe = np.maximum(s, 1.0)
    borrowed = ((h @ gm) / counts)[perm]
    r = alive * (((mask * h) @ gm) / s_safe) + (1 - alive) * borrowed
    return r @ gm.T, alive, s_safe


def channel_backward(dx, mask, gm, alive, s_safe, perm):
    counts = gm.sum(0)
    dr = dx @ gm
    dh = (((dr * alive) / s_safe) @ gm.T) * mask
    dborrow = (dr * (1 - alive)) / counts
    dh_other = np.zeros_like(dh)
    np.add.at(dh_other, perm, dborrow @ gm.T)
    return dh + dh_other


def forward(model, batch, cache=False):
    p, cfg, ko = model["params"], model["cfg"], model["ko"]
    gm, P = projector(model)
    hd = cfg["hd"]
    n = batch["Z"].shape[0]
    g_total = batch["G"]
    u = batch["Z"]
    forge = "FORGE_Z" in batch
    fa = fc = fmask = None
    if forge:
        fa = np.tanh(batch["FORGE_Z"] @ p["enc_W1"] + p["enc_b1"])
        fc = np.tanh(fa @ p["enc_W2"] + p["enc_b2"])
        fcode = fc @ P
        fmask = np.zeros((n, NPOS))
        for i in range(n):
            st = int(batch["FORGE_STANZA"][i])
            fmask[i, st * NL * M:(st + 1) * NL * M] = 1.0
    loss = 0.0
    caches, logits = [], []
    for g in range(g_total):
        a = np.tanh(u @ p["enc_W1"] + p["enc_b1"])
        c = np.tanh(a @ p["enc_W2"] + p["enc_b2"])
        if ko.get("singer_encoder") == "zero":
            c = np.zeros_like(c)
        elif ko.get("singer_encoder") == "mean":
            c = np.broadcast_to(c.mean(0, keepdims=True), c.shape).copy()
        h = c @ P
        if ko.get("meter") == "zero":
            h = np.zeros_like(h)
        elif ko.get("meter") == "mean":
            h = np.broadcast_to(h.mean(0, keepdims=True), h.shape).copy()
        if forge and g >= 1:
            h = h * (1 - fmask) + fcode * fmask
        x, alive, s_safe = channel_forward(h, batch["MASK"][g], gm, batch["PERM"][g])
        v = x
        pre = v @ p["dec_W3"] + p["dec_b3"]
        e = None
        if cfg["use_anchor"] and ko.get("refrain_anchor") not in ("identity", "zero"):
            rho = x[:, REFRAIN_IDX]
            e = np.tanh(rho @ p["film_W"] + p["film_b"])
            if ko.get("refrain_anchor") == "mean":
                e = np.broadcast_to(e.mean(0, keepdims=True), e.shape).copy()
            gam, bet = e[:, :hd], e[:, hd:]
            arg = pre * (1 + gam) + bet
        else:
            gam = bet = None
            arg = pre
        hid = arg if ko.get("hearer_hidden") == "identity" else np.tanh(arg)
        if ko.get("hearer_hidden") == "mean":
            hid = np.broadcast_to(hid.mean(0, keepdims=True), hid.shape).copy()
        zhat = hid @ p["dec_W4"] + p["dec_b4"]
        if cfg["query_dim"] > 0:
            prod = zhat * batch["Q"][g]
            logit = prod @ p["head_w"] + p["head_b"][0]
            if ko.get("answer_head") == "identity":
                logit = prod.sum(1)
            yg = batch["Y"][g]
            loss += float(np.mean(softplus(logit) - yg * logit)) / g_total
            logits.append(logit)
        else:
            logit = zhat @ p["head_W"] + p["head_b"]
            logits.append(logit)
            yg = batch["Y"][g] if g == g_total - 1 else None
            if yg is not None:
                loss += float(np.mean(logsumexp(logit, 1) - logit[np.arange(n), yg.astype(int)]))
        if cache:
            caches.append({"u": u, "a": a, "c": c, "h": h, "x": x, "v": v, "pre": pre, "e": e, "gam": gam,
                           "bet": bet, "arg": arg, "hid": hid, "zhat": zhat, "logit": logit, "s_safe": s_safe,
                           "alive": alive, "mask": batch["MASK"][g], "perm": batch["PERM"][g],
                           "fmask": fmask if (forge and g >= 1) else None})
        u = zhat
    if cache and forge:
        caches[0]["forger"] = (fa, fc)
    return loss, logits, caches, (gm, P)


def loss_and_grads(model, batch):
    p, cfg, ko = model["params"], model["cfg"], model["ko"]
    loss, logits, caches, (gm, P) = forward(model, batch, cache=True)
    grads = {k: np.zeros_like(v) for k, v in p.items()}
    n, g_total, hd = batch["Z"].shape[0], batch["G"], cfg["hd"]
    du_next = np.zeros((n, cfg["doctrine_dim"]))
    dfcode = np.zeros((n, NPOS))
    for g in reversed(range(g_total)):
        cc = caches[g]
        if cfg["query_dim"] > 0:
            dlogit = (sigmoid(cc["logit"]) - batch["Y"][g]) / (n * g_total)
            prod = cc["zhat"] * batch["Q"][g]
            if ko.get("answer_head") == "identity":
                dzhat = dlogit[:, None] * batch["Q"][g]
            else:
                grads["head_w"] += (prod * dlogit[:, None]).sum(0)
                grads["head_b"] += dlogit.sum()
                dzhat = dlogit[:, None] * batch["Q"][g] * p["head_w"][None, :]
        else:
            dzhat = np.zeros_like(cc["zhat"])
            if g == g_total - 1:
                prob = stable_softmax(cc["logit"], 1)
                prob[np.arange(n), batch["Y"][g].astype(int)] -= 1.0
                dlog = prob / n
                grads["head_W"] += cc["zhat"].T @ dlog
                grads["head_b"] += dlog.sum(0)
                dzhat = dlog @ p["head_W"].T
        dzhat = dzhat + du_next
        grads["dec_W4"] += cc["hid"].T @ dzhat
        grads["dec_b4"] += dzhat.sum(0)
        dhid = dzhat @ p["dec_W4"].T
        if ko.get("hearer_hidden") == "mean":
            dhid = np.broadcast_to(dhid.mean(0, keepdims=True), dhid.shape)
        darg = dhid if ko.get("hearer_hidden") == "identity" else dhid * (1 - cc["hid"] ** 2)
        dx_extra = None
        if cc["e"] is not None:
            dpre = darg * (1 + cc["gam"])
            de = np.concatenate([darg * cc["pre"], darg], axis=1)
            if ko.get("refrain_anchor") == "mean":
                de = np.broadcast_to(de.mean(0, keepdims=True), de.shape)
            de = de * (1 - cc["e"] ** 2)
            rho = cc["x"][:, REFRAIN_IDX]
            grads["film_W"] += rho.T @ de
            grads["film_b"] += de.sum(0)
            dx_extra = de @ p["film_W"].T
        else:
            dpre = darg
        grads["dec_W3"] += cc["v"].T @ dpre
        grads["dec_b3"] += dpre.sum(0)
        dx = dpre @ p["dec_W3"].T
        if dx_extra is not None:
            dx[:, REFRAIN_IDX] += dx_extra
        dh = channel_backward(dx, cc["mask"], gm, cc["alive"], cc["s_safe"], cc["perm"])
        if cc["fmask"] is not None:
            dfcode += dh * cc["fmask"]
            dh = dh * (1 - cc["fmask"])
        if ko.get("meter") == "zero":
            dh = np.zeros_like(dh)
        elif ko.get("meter") == "mean":
            dh = np.broadcast_to(dh.mean(0, keepdims=True), dh.shape)
        dc = dh if model["mutant"] == "untied_backward" else dh @ P
        if ko.get("singer_encoder") == "zero":
            dc = np.zeros_like(dc)
        elif ko.get("singer_encoder") == "mean":
            dc = np.broadcast_to(dc.mean(0, keepdims=True), dc.shape)
        dpre2 = dc * (1 - cc["c"] ** 2)
        grads["enc_W2"] += cc["a"].T @ dpre2
        grads["enc_b2"] += dpre2.sum(0)
        da = dpre2 @ p["enc_W2"].T
        dpre1 = da * (1 - cc["a"] ** 2)
        grads["enc_W1"] += cc["u"].T @ dpre1
        grads["enc_b1"] += dpre1.sum(0)
        du_next = dpre1 @ p["enc_W1"].T
    if "forger" in caches[0]:
        # the forger sings with the same trained singer, so its stanza also
        # depends on the encoder parameters
        fa, fc = caches[0]["forger"]
        dfpre2 = (dfcode @ P) * (1 - fc ** 2)
        grads["enc_W2"] += fa.T @ dfpre2
        grads["enc_b2"] += dfpre2.sum(0)
        dfpre1 = (dfpre2 @ p["enc_W2"].T) * (1 - fa ** 2)
        grads["enc_W1"] += batch["FORGE_Z"].T @ dfpre1
        grads["enc_b1"] += dfpre1.sum(0)
    if model["mutant"] == "zero_grad_enc_W2":
        grads["enc_W2"] = np.zeros_like(grads["enc_W2"])
    return loss, grads


def infer_batch(model, X, rng=None):
    """Wrap external vectors as a doctrine (and query, if configured) with the
    model's inference erasure level, so the probe battery can call predict()."""
    cfg = model["cfg"]
    rng = rng if rng is not None else np.random.default_rng(12345)
    n = X.shape[0]
    z = X[:, :cfg["doctrine_dim"]]
    g = cfg["g_infer"]
    line_keep = (rng.random((g, n, NLINES)) >= cfg["p_infer"]).astype(np.float64)
    batch = {"Z": z, "G": g, "MASK": line_keep[:, :, LINE_OF_POS], "Y": np.zeros((g, n)),
             "PERM": np.stack([rng.permutation(n) for _ in range(g)])}
    if cfg["query_dim"] > 0:
        q = X[:, cfg["doctrine_dim"]:]
        batch["Q"] = np.broadcast_to(q[None], (g, n, cfg["query_dim"])).copy()
    return batch


def predict(model, X, rng=None):
    batch = infer_batch(model, X, rng)
    _, logits, _, _ = forward(model, batch)
    last = logits[-1]
    if model["cfg"]["query_dim"] > 0:
        return (last > 0).astype(int)
    return np.argmax(last, axis=1)


def hidden_states(model, X, rng=None):
    batch = infer_batch(model, X, rng)
    _, _, caches, _ = forward(model, batch, cache=True)
    return {"doctrine_estimate": caches[-1]["zhat"], "hearer_hidden": caches[-1]["hid"], "code": caches[-1]["h"]}


def accuracy(model, batch, generation=-1):
    _, logits, _, _ = forward(model, batch)
    return float(np.mean((logits[generation] > 0) == (batch["Y"][generation] > 0.5)))


# ----------------------------------------------------------------------------
# 8. Baselines and rival mechanisms
# ----------------------------------------------------------------------------
def build_baseline(rng):
    """Free-verse chain: the closest prior art (a neural iterated-learning chain
    through an unconstrained noisy channel), size-matched to the full model."""
    return build_model(rng=rng, use_meter=False, use_anchor=False)


def trivial_baselines(batch):
    y = batch["Y"][-1]
    return {"majority": float(max(y.mean(), 1 - y.mean()))}


# ----------------------------------------------------------------------------
# 9. Registries: modules, knockouts, mutants
# ----------------------------------------------------------------------------
def modules(model):
    p = model["params"]
    out = {
        "singer_encoder": {"params": ["enc_W1", "enc_b1", "enc_W2", "enc_b2"],
                           "role": "two-layer tanh encoder from doctrine to 36 code slots", "signature": False},
        "meter": {"params": [], "role": "fixed group-averaging projection (rhyme and refrain ties) "
                                        "with channel-level recovery of tied positions", "signature": True},
        "hearer_hidden": {"params": ["dec_W3", "dec_b3"], "role": "tanh hidden layer of the decoder", "signature": False},
        "hearer_readout": {"params": ["dec_W4", "dec_b4"], "role": "linear readout to the doctrine estimate", "signature": False},
        "answer_head": {"params": [k for k in ("head_w", "head_b", "head_W") if k in p],
                        "role": "query-weighted linear head", "signature": False},
    }
    if model["cfg"]["use_anchor"]:
        out["refrain_anchor"] = {"params": ["film_W", "film_b"],
                                 "role": "feature-wise gain and bias of the hidden layer from the pooled refrain",
                                 "signature": True}
    return out


def knockout(model, name, mode="identity"):
    if name not in modules(model) or mode not in ("identity", "zero", "mean"):
        raise ValueError(f"{name}/{mode}")
    out = {"params": {k: v.copy() for k, v in model["params"].items()}, "cfg": dict(model["cfg"]),
           "groups": model["groups"], "ko": dict(model["ko"]), "mutant": model["mutant"]}
    out["ko"][name] = mode
    return out


MUTANTS = {
    "sign_flip": "Adam step applied with the wrong sign",
    "zero_lr": "learning rate set to zero",
    "zero_grad_enc_W2": "analytic gradient of enc_W2 replaced by zeros",
    "untied_backward": "backward pass skips the meter projection",
}


# ----------------------------------------------------------------------------
# 10. Training
# ----------------------------------------------------------------------------
def fit(model, data, budget, rng, lr=3e-3, batch_size=64, eval_every=50, log=None):
    p = model["params"]
    opt = Adam(p, lr=0.0 if model["mutant"] == "zero_lr" else lr)
    sign = -1.0 if model["mutant"] == "sign_flip" else 1.0
    n = data["train"]["Z"].shape[0]
    losses, best = [], (-1.0, None)
    for step in range(budget):
        idx = rng.integers(0, n, size=batch_size)
        loss, grads = loss_and_grads(model, resample_masks(slice_batch(data["train"], idx), P_TRAIN, rng))
        if not np.isfinite(loss):
            raise FloatingPointError("non-finite loss")
        clip_global_norm(grads, CLIP_NORM)
        opt.step(p, grads, sign=sign)
        losses.append(loss)
        if (step + 1) % eval_every == 0 or step == budget - 1:
            va = accuracy(model, data["val"])
            if va > best[0]:
                best = (va, {k: v.copy() for k, v in p.items()})
            if log is not None:
                log.append((step + 1, float(np.mean(losses[-eval_every:])), va))
    if best[1] is not None:
        for k in p:
            p[k][...] = best[1][k]
    return np.array(losses)


def train_pair(seed, data, budget, quiet=True):
    rng_full = np.random.default_rng(seed)
    rng_base = np.random.default_rng(seed + 1000)
    full = build_model(rng=rng_full)
    base = build_baseline(rng_base)
    lf = fit(full, data, budget, np.random.default_rng(seed + 7))
    lb = fit(base, data, budget, np.random.default_rng(seed + 7))
    return full, base, lf, lb


# ----------------------------------------------------------------------------
# 11. Tests: correctness first, then hypotheses
# ----------------------------------------------------------------------------
def gradcheck(model, batch, rng, n_entries=20):
    loss, grads = loss_and_grads(model, batch)

    def loss_fn():
        return forward(model, batch)[0]
    return finite_difference_check(loss_fn, model["params"], grads, rng, n_entries=n_entries)


def test_determinism(data):
    outs = []
    for _ in range(2):
        m = build_model(rng=np.random.default_rng(3))
        losses = fit(m, data, 30, np.random.default_rng(4), eval_every=30)
        acc = accuracy(m, data["test"])
        outs.append((losses.copy(), acc, {k: v.copy() for k, v in m["params"].items()}))
    same = np.array_equal(outs[0][0], outs[1][0]) and outs[0][1] == outs[1][1]
    same = same and all(np.array_equal(outs[0][2][k], outs[1][2][k]) for k in outs[0][2])
    finite = all(np.all(np.isfinite(v)) for v in outs[0][2].values())
    return same and finite, f"identical losses/params across two runs={same}, finite={finite}"


def test_property_tie(rng):
    """C6.1 (labelled definition check): positions inside a rhyme or refrain group
    are equal after the meter projection; a negative control (identity) violates."""
    gm, P = projector(build_model(rng=rng))
    worst = 0.0
    for _ in range(200):
        c = rng.standard_normal((4, NPOS)) * rng.uniform(0.1, 10)
        h = c @ P
        for members in metered_groups():
            worst = max(worst, float(np.ptp(h[:, members], axis=1).max()))
    c = rng.standard_normal((4, NPOS))
    control_violates = max(float(np.ptp(c[:, g], axis=1).max()) for g in metered_groups() if len(g) > 1) > 1e-6
    return worst < 1e-9 and control_violates, f"max within-group spread {worst:.1e}; control violates={control_violates}"


def test_property_recovery(rng):
    """C6.2: for any erasure pattern, a tied group with at least one surviving
    member is received exactly; a broken channel that divides by the group size
    instead of the survivor count violates."""
    gm, P = projector(build_model(rng=rng))
    counts = gm.sum(0)
    worst, control_worst = 0.0, 0.0
    for _ in range(300):
        h = rng.standard_normal((8, NPOS)) @ P
        mask = (rng.random((8, NLINES)) >= rng.uniform(0.1, 0.9)).astype(np.float64)[:, LINE_OF_POS]
        perm = rng.permutation(8)
        x, alive_g, _ = channel_forward(h, mask, gm, perm)
        alive = (alive_g @ gm.T) > 0.5
        worst = max(worst, float(np.abs((x - h)[alive]).max()) if alive.any() else 0.0)
        r_broken = ((mask * h) @ gm) / counts
        x_broken = r_broken @ gm.T
        control_worst = max(control_worst, float(np.abs((x_broken - h)[alive]).max()) if alive.any() else 0.0)
    return worst < 1e-12 and control_worst > 1e-3, f"max recovery error {worst:.1e}; broken-channel error {control_worst:.2f}"


def test_property_nonexpansive(rng):
    """C6.3: the meter projection never increases the code norm (random search
    plus the group-mean direction where the ratio is largest); a scaled
    projection 1.2P violates."""
    gm, P = projector(build_model(rng=rng))
    ratio, control = 0.0, 0.0
    for _ in range(500):
        c = rng.standard_normal((1, NPOS)) * rng.uniform(0.01, 100)
        ratio = max(ratio, float(np.linalg.norm(c @ P) / np.linalg.norm(c)))
        control = max(control, float(np.linalg.norm(c @ (1.2 * P)) / np.linalg.norm(c)))
    c = np.ones((1, NPOS))
    ratio = max(ratio, float(np.linalg.norm(c @ P) / np.linalg.norm(c)))
    control = max(control, float(np.linalg.norm(c @ (1.2 * P)) / np.linalg.norm(c)))
    return ratio <= 1.0 + 1e-12 and control > 1.0, f"max ratio {ratio:.6f}; control ratio {control:.3f}"


def test_split_integrity(data):
    names = list(data.keys())
    hashes = {k: row_hashes(data[k]["Z"]) for k in names}
    overlaps = sum(len(hashes[a] & hashes[b]) for i, a in enumerate(names) for b in names[i + 1:])
    return overlaps == 0, f"{overlaps} shared doctrines across {len(names)} splits"


def run_mutant(name, data, budget, rng):
    m = build_model(rng=np.random.default_rng(11))
    m["mutant"] = name
    gc = gradcheck(m, slice_batch(data["train"], np.arange(8)), rng, n_entries=6)
    losses = fit(m, data, budget, np.random.default_rng(12), eval_every=budget)
    drop = 1 - losses[-20:].mean() / losses[:20].mean()
    acc = accuracy(m, data["test"])
    c3_fail = drop < MIND_CARD["thresholds"]["loss_drop_fraction"] or \
        acc - trivial_baselines(data["test"])["majority"] < MIND_CARD["thresholds"]["margin_over_trivial"]
    return (not gc["passed"]) or c3_fail, f"gradcheck passed={gc['passed']}, loss drop {drop:.2f}, acc {acc:.3f}"


def evaluate_seed(full, base, data):
    res = {"acc_test": accuracy(full, data["test"]), "acc_test_base": accuracy(base, data["test"]),
           "acc_shifted": accuracy(full, data["shifted"]), "acc_shifted_base": accuracy(base, data["shifted"]),
           "acc_forged": accuracy(full, data["forged"]), "acc_forged_base": accuracy(base, data["forged"])}
    res["ko"] = {}
    for name, mode in (("meter", "identity"), ("refrain_anchor", "mean"), ("hearer_hidden", "identity"),
                       ("singer_encoder", "mean"), ("answer_head", "identity")):
        res["ko"][name] = accuracy(knockout(full, name, mode), data["shifted"])
    return res


# ----------------------------------------------------------------------------
# 12. Report
# ----------------------------------------------------------------------------
def run_protocol(args):
    t0 = time.time()
    quick = args.quick
    n_seeds = 1 if quick else args.seeds
    budget = 700 if quick else 1200
    budget_limit = BUDGET_QUICK_S if quick else BUDGET_FULL_S
    seeds = [args.seed + i for i in range(n_seeds)]
    print(f"chapter 0342 · seeds {seeds} · budget {budget} updates · {'quick' if quick else 'full'} protocol"
          + (f" · MUTANT {args.mutant} applied to the main model (the audit expects a failure)" if args.mutant else ""))
    data = make_data(np.random.default_rng(args.seed + 99))
    correctness = []
    probe_rng = np.random.default_rng(args.seed + 5)

    # C7 split integrity, C6 properties, C2 determinism
    ok, detail = test_split_integrity(data)
    correctness.append({"id": "C7", "name": "split_integrity", "passed": ok, "detail": detail})
    for cid, fn in (("C6.1", test_property_tie), ("C6.2", test_property_recovery), ("C6.3", test_property_nonexpansive)):
        ok, detail = fn(probe_rng)
        correctness.append({"id": cid, "name": fn.__name__, "passed": ok, "detail": detail})
    ok, detail = test_determinism(data)
    correctness.append({"id": "C2", "name": "determinism_finiteness", "passed": ok, "detail": detail})

    # C1 at initialisation
    init_model = build_model(rng=np.random.default_rng(args.seed))
    init_model["mutant"] = args.mutant
    gc_init = gradcheck(init_model, slice_batch(data["forged"], np.arange(8)), probe_rng)
    gc_init_base = gradcheck(build_baseline(np.random.default_rng(args.seed)), slice_batch(data["train"], np.arange(8)), probe_rng)

    # training over seeds
    per_seed = []
    train_logs = []
    for seed in seeds:
        log = []
        full = build_model(rng=np.random.default_rng(seed))
        full["mutant"] = args.mutant          # None in a normal run; the audit passes --mutant NAME
        base = build_baseline(np.random.default_rng(seed + 1000))
        lf = fit(full, data, budget, np.random.default_rng(seed + 7), log=log)
        lb = fit(base, data, budget, np.random.default_rng(seed + 7))
        r = evaluate_seed(full, base, data)
        r["loss_drop"] = float(1 - lf[-50:].mean() / lf[:50].mean())
        r["loss_drop_base"] = float(1 - lb[-50:].mean() / lb[:50].mean())
        r["model"] = full
        per_seed.append(r)
        train_logs.append(log)
        print(f"  seed {seed}: loss drop {r['loss_drop']:.2f} | test {r['acc_test']:.3f} (base {r['acc_test_base']:.3f})"
              f" | shifted {r['acc_shifted']:.3f} (base {r['acc_shifted_base']:.3f})"
              f" | forged {r['acc_forged']:.3f} (base {r['acc_forged_base']:.3f})")
    # C1 after training
    gc_after = gradcheck(per_seed[0]["model"], slice_batch(data["forged"], np.arange(8)), probe_rng)
    gc_all = {"tensors_checked": gc_init["tensors_checked"], "tensors_total": gc_init["tensors_total"],
              "max_rel_error": max(gc_init["max_rel_error"], gc_after["max_rel_error"], gc_init_base["max_rel_error"]),
              "checked_at": ["init", "after_training_steps"],
              "passed": gc_init["passed"] and gc_after["passed"] and gc_init_base["passed"]}
    correctness.insert(0, {"id": "C1", "name": "gradient_check", "passed": gc_all["passed"],
                           "detail": f"max rel error {gc_all['max_rel_error']:.2e} (full init/after, baseline init)"})

    # C3 learning
    triv = trivial_baselines(data["test"])["majority"]
    th = MIND_CARD["thresholds"]
    drop_ok = all(r["loss_drop"] >= th["loss_drop_fraction"] for r in per_seed)
    margin_ok = all(r["acc_test"] - triv >= th["margin_over_trivial"] for r in per_seed)
    correctness.append({"id": "C3", "name": "learning", "passed": drop_ok and margin_ok,
                        "detail": f"loss drop min {min(r['loss_drop'] for r in per_seed):.2f} (>= {th['loss_drop_fraction']}); "
                                  f"test acc min {min(r['acc_test'] for r in per_seed):.3f} vs majority {triv:.3f} "
                                  f"(margin >= {th['margin_over_trivial']})"})

    # C4 shuffled-label control, aggregated over seeds.  The answer head is a
    # six-weight linear readout of zhat*q, so a model with a random head is
    # already sign-correlated or sign-anti-correlated with the true labelling
    # (single-seed accuracy on true labels ranges roughly 0.43-0.58 at init).
    # Leakage would show as a systematic gain across seeds, so the control is
    # the seed mean; single-seed values are reported for inspection.
    sh_accs = []
    for seed in (seeds if not quick else [args.seed + i for i in range(3)]):
        sh_data = dict(data)
        sh_data["train"] = shuffled_labels(data["train"], np.random.default_rng(seed + 31))
        sh_data["val"] = shuffled_labels(data["val"], np.random.default_rng(seed + 32))
        sh_model = build_model(rng=np.random.default_rng(seed))
        fit(sh_model, sh_data, budget, np.random.default_rng(seed + 7))
        sh_accs.append(accuracy(sh_model, data["test"]))
    sh_acc = float(np.mean(sh_accs))
    band = th["shuffled_band"]
    correctness.append({"id": "C4", "name": "shuffled_label_control", "passed": abs(sh_acc - triv) <= band,
                        "detail": f"seed-mean held-out acc {sh_acc:.3f} within {band} of majority {triv:.3f} "
                                  f"(per seed {', '.join(f'{a:.3f}' for a in sh_accs)})"})

    # C5 mutants
    mutant_names = list(MUTANTS)          # the registry is always audited in full
    detected = 0
    mutant_details = []
    for name in mutant_names:
        caught, detail = run_mutant(name, data, 250 if quick else 400, probe_rng)
        detected += int(caught)
        mutant_details.append(f"{name}: {'caught' if caught else 'MISSED'} ({detail})")
    mutants = {"detected": detected, "total": len(mutant_names), "score": detected / len(mutant_names)}
    correctness.append({"id": "C5", "name": "mutant_detection", "passed": detected == len(mutant_names),
                        "detail": "; ".join(mutant_details)})

    # hypotheses
    boot_rng = np.random.default_rng(args.seed + 77)
    hyps = []
    diff_sets = {
        "H-SIG": [r["acc_shifted"] - r["acc_shifted_base"] for r in per_seed],
        "H-NEC": [r["ko"]["meter"] - r["ko"]["hearer_hidden"] for r in per_seed],
        "H-NEC-ANCHOR": [r["ko"]["refrain_anchor"] - r["ko"]["hearer_hidden"] for r in per_seed],
        "H-BLIND": [(r["acc_test"] - r["acc_forged"]) - (r["acc_test_base"] - r["acc_forged_base"]) for r in per_seed],
    }
    for h in MIND_CARD["hypotheses"]:
        diffs = diff_sets[h["id"]]
        if quick:
            mean, ci, v = float(np.mean(diffs)), [float(np.mean(diffs))] * 2, "not evaluated"
        else:
            mean, ci = paired_bootstrap_ci(diffs, boot_rng)
            v = verdict(mean, ci, h["direction"], h["mesi"])
        hyps.append({"id": h["id"], "metric": h["metric"], "mean_diff": mean, "ci95": ci, "mesi": h["mesi"],
                     "n_seeds": len(diffs), "verdict": v})
    knock = []
    mods = modules(per_seed[0]["model"])
    for name in ("meter", "refrain_anchor", "hearer_hidden", "singer_encoder", "answer_head"):
        ch = [r["ko"][name] - r["acc_shifted"] for r in per_seed]
        mean, ci = (float(np.mean(ch)), [float(np.mean(ch))] * 2) if quick else paired_bootstrap_ci(ch, boot_rng)
        knock.append({"module": name, "signature": mods[name]["signature"], "metric_change": mean, "ci95": ci})

    runtime = time.time() - t0
    correctness.append({"id": "C8", "name": "budget", "passed": runtime <= budget_limit,
                        "detail": f"{runtime:.1f}s of {budget_limit:.0f}s"})
    all_finite = all(np.all(np.isfinite(v)) for r in per_seed for v in r["model"]["params"].values())
    exit_code = 0 if all(c["passed"] for c in correctness) else 1
    if not all_finite:
        exit_code = 4
    elif runtime > budget_limit:
        exit_code = 3
    report = {"schema_version": "1.0", "chapter": 342, "file": os.path.basename(__file__),
              "card_revision": MIND_CARD["card_revision"],
              "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
              "seeds": seeds, "runtime_s": runtime, "n_params": n_params(per_seed[0]["model"]),
              "n_params_baseline": n_params(build_baseline(np.random.default_rng(0))),
              "gradcheck": gc_all, "correctness": correctness, "mutants": mutants, "hypotheses": hyps,
              "knockouts": knock, "task_types": TASK_TYPES, "exit_code": exit_code,
              "per_seed": [{k: v for k, v in r.items() if k != "model"} for r in per_seed],
              "training_curve_seed0": train_logs[0]}
    write_report(report, args.json)
    return exit_code, per_seed, data


def real_data_bridge(path, per_seed, data):
    """Optional: treat the numeric columns of a local CSV as doctrines and
    measure chain fidelity for the trained full model and baseline."""
    if not os.path.exists(path):
        print(f"--data: {path} not found; real-data bridge skipped")
        return
    rows = []
    with open(path, encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(",")
            try:
                rows.append([float(x) for x in parts[:D]])
            except ValueError:
                continue
    z = np.asarray(rows)
    if z.shape[0] < 10 or z.shape[1] < D:
        print("--data: need at least 10 rows and 6 numeric leading columns; skipped")
        return
    z = unit_rows((z - z.mean(0)) / (z.std(0) + 1e-9))
    rng = np.random.default_rng(2026)
    batch = make_split(z.shape[0], G_SHIFT, P_SHIFT, rng)
    batch["Z"] = z
    batch["Y"] = (np.einsum("nd,gnd->gn", z, batch["Q"]) > 0).astype(np.float64)
    full = per_seed[0]["model"]
    base = build_baseline(np.random.default_rng(0))
    fit(base, data, 300, np.random.default_rng(1))
    print(f"--data {os.path.basename(path)}: rows {z.shape[0]} · shifted-regime accuracy full {accuracy(full, batch):.3f}"
          f" · baseline {accuracy(base, batch):.3f} · majority {trivial_baselines(batch)['majority']:.3f}")


# ----------------------------------------------------------------------------
# 13. Command-line entry point
# ----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__), add_help=True,
                                 usage=f"python3 {os.path.basename(__file__)} [--quick] [--seed S] [--seeds N] "
                                       "[--json PATH] [--card] [--mutant NAME] [--data PATH]")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--card", action="store_true")
    ap.add_argument("--mutant", type=str, default=None)
    ap.add_argument("--data", type=str, default=None)
    try:
        args = ap.parse_args(argv)
    except SystemExit:
        return 2
    if args.card:
        print(json.dumps(MIND_CARD, indent=2))
        return 0
    if args.mutant and args.mutant not in MUTANTS:
        print(f"unknown mutant {args.mutant}; registered: {list(MUTANTS)}")
        return 2
    if args.seeds < 1:
        return 2
    try:
        code, per_seed, data = run_protocol(args)
    except FloatingPointError as err:
        print(f"non-finite values: {err}")
        return 4
    if args.data:
        real_data_bridge(args.data, per_seed, data)
    return code


if __name__ == "__main__":
    sys.exit(main())

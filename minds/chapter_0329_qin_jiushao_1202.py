#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0329 · Qin Jiushao (1202-1261)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 17, Minds 321-340 Available on Amazon https://www.amazon.com/dp/B0HP7Z1NKC
# Resume and Interactive Demos at https://artificiology.com/
# END ATTRIBUTION
"""
DAYAN NET — remainders that count, multipliers that reconstruct.
Chapter 0329 · Qin Jiushao (1202–1261) · Southern Song · Shushu jiuzhang (1247)

THESIS
    A quantity is never held whole; it is held as its remainders under several
    small incommensurable rulers, and the mind's one operation is 求一 — finding
    the multiplier that makes each remainder count exactly once when the whole is
    reassembled — after 定數 has first stripped every ruler of what it shares
    with the others.

EVIDENCE AND PROVENANCE (belief: the 1247 preface and treatise are his own text)
    D1  Preface: 大衍 is placed first and called the guiding rule; the editors
        of 1784 summarise the class as 以奇零求總數 — from the odd remainders
        seek the total.  (primary + scholarship: Siku tiyao)
    D2  大衍求一術: the 2×2 rod layout (天元 1 upper-left, 奇 upper-right, 定
        lower-right) with alternating division until the upper-right holds 1;
        the upper-left is then the 乘率.  (primary, Ssjz juan 1; worked case
        377873x ≡ 1 (mod 499067) → 457999 in juan 2, problem 3)
    D3  定數: moduli sharing factors are reduced pairwise until coprime, the
        product equalling the least common multiple.  (primary, Ssjz juan 1;
        problem I.3: 54, 57, 75, 72 → 27, 19, 25, 8)
    D4  The treatise never states the solvability condition r_i ≡ r_j (mod
        gcd(m_i, m_j)); every example happens to satisfy it.  (scholarship:
        Ho Peng Yoke, DSB 1971; Shen Kangsheng 1988)
    D5  Preface: 曆家雖用，用而不知 — the calendar makers use the rule without
        understanding it; the rule is applied to the Grand Epoch 上元 with
        moduli of six and seven digits.  (primary; Ssjz juan 2 problem 3)
    D6  Preface: 數與道非二本 — number and the Way do not have two roots; the
        great use is 通神明順性命, the small use 經世務類萬物.  (primary)

DOCTRINE → MECHANISM → TEST
    D1  → M1 residue registers with learned cyclic shifts   → C6.1, C6.2, C6.3, H-SIG
    D2  → M2 求一: the 乘率 inside the reassembly table       → C6.4, H-NEC (table without 乘率)
    D3  → M2 定數 reduction before the table is built        → C6.5, H-SIG
    D4  → blind spot: inconsistent residues land on cells    → C6.6 (definition), H-BLIND
    D5  → task: an event ledger observed by two instruments  → C3, C4, C7
    D6  → one core for the cosmic and the mundane ledger     → applications

RESEARCH QUESTION (compositional and systematic generalization; robustness)
    Can a register that accumulates events by learned cyclic shifts on small
    rings, reassembled through a 定數/求一 table that is computed rather than
    learned, extrapolate exact modular counting to sequence lengths and count
    ranges never seen in training — and what does exact reconstruction cost
    when each instrument's record carries its own transcription errors?

CLOSEST PRIOR ART AND THE DELTA
    Residue hyperdimensional computing (Kymn et al. 2023, arXiv:2311.04872) and
    the residue-number-system reading of grid cells (Fiete, Burak & Brookings
    2008) encode integers as fixed products of phasors.  Deep Sets (Zaheer et
    al. 2017) is the size-matched standard baseline implemented here.  Delta:
    the residues are not given but ACCUMULATED from a token stream by learned
    circular-convolution kernels on probability rings; the joint of the rings
    is scattered onto the cycle by a table built from Qin's own 定數 and 求一
    procedures (rebuilt on his rod layout, no free parameters) and credited in
    the log domain, so each ring learns its own residue from a label that is
    the whole; a rod-discreteness term makes every shift a whole cell, which
    is what lets the rings run for hundreds of steps without blur.  Overlap:
    Medium.

BLIND SPOT
    Qin discards redundancy between rulers (定數) instead of exploiting it, and
    never checks that the remainders are mutually consistent.  One wrong
    residue moves the reassembled total by a large multiple of the ruler, and
    the table gives the wrong cell the same confidence as a right one; a
    magnitude code degrades gracefully.  H-BLIND predicts the model degrades
    more than the baseline, relative to its clean accuracy, under
    per-instrument transcription errors.

TASK — the sexagenary ledger (sequence_classification)
    1. Twelve event symbols; four are silent (increment 0), eight carry hidden
       increments in {-4..4} drawn per seed (both signs present).
    2. Two instruments (stems, period 10; branches, period 12) record the same
       stream, each through its own symbol permutation.  Input per step is the
       concatenation of the two one-hot readings; each ring reads one of them.
    3. Target: the day of the 60-cycle reached by the accumulated count,
       y = (Σ increments) mod 60, sixty classes.
    4. Train/val/held-out: length 1–32.  Shifted: length 64–120 (counts beyond
       the training range).  Blind: length 8–32 with each instrument's record
       independently corrupted at rate 0.08.  Truth always comes from the true
       stream.
    5. Training regime, identical for every system: Adam, 1400 updates, batch
       32, a length curriculum (admissible length grows 2 → 32 over the first
       half), best-validation checkpoint.  The model's loss adds the
       rod-discreteness term during the second half.

LIMITS
    Two rings only; the 60-day cycle is the whole world; corruption is
    symbol replacement, not deletion; no redundant ring is provided, so the
    file measures the blind spot rather than repairing it.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 1,
    "revision_log": [],
    "generation": {"template_version": "1.0", "generator": "Claude", "generator_version": "Fable 5.1",
                   "date": "2026-09-25"},
    "id": 329, "figure": "Qin Jiushao", "born": 1202, "died": 1261, "civilization": "Chinese (Song)",
    "provenance": "belief",
    "thesis": "A quantity is held only as its remainders under several small incommensurable rulers; "
              "the mind's operation is to strip the rulers of what they share (定數), find each ruler's "
              "multiplier so that its remainder counts exactly once (求一), and reassemble the whole.",
    "evidence": [
        {"id": "D1", "claim": "The Great Extension rule opens the treatise and governs its nine classes; "
                              "from odd remainders one seeks the total (以奇零求總數).",
         "basis": "primary", "source": "Shushu jiuzhang, preface (1247); Siku quanshu zongmu tiyao (1784)"},
        {"id": "D2", "claim": "The multiplier is found on a 2x2 rod layout by alternating division until the "
                              "upper-right cell holds one; the upper-left cell is then the multiplier.",
         "basis": "primary", "source": "Shushu jiuzhang, juan 1 (大衍求一術); juan 2 problem 3 (457999)"},
        {"id": "D3", "claim": "Moduli sharing factors are reduced pairwise to definite numbers whose product "
                              "is the least common multiple before any multiplier is sought.",
         "basis": "primary", "source": "Shushu jiuzhang, juan 1 (定數); problem I.3"},
        {"id": "D4", "claim": "No consistency condition on the remainders is stated; every example satisfies "
                              "it by construction.",
         "basis": "scholarship", "source": "Ho Peng Yoke, DSB (1971); Shen Kangsheng, AHES 38 (1988)"},
        {"id": "D5", "claim": "Calendar makers used the rule without understanding it; Qin applies it to the "
                              "Grand Epoch with six- and seven-digit moduli.",
         "basis": "primary", "source": "Shushu jiuzhang, preface; juan 2 problem 3"},
        {"id": "D6", "claim": "Number and the Way are not two roots; the same art serves the cosmic and the "
                              "administrative.",
         "basis": "primary", "source": "Shushu jiuzhang, preface (數與道非二本)"},
    ],
    "research_question": {
        "category": "compositional and systematic generalization",
        "question": "Can learned cyclic-shift residue registers decoded through Qin's definite-number and "
                    "finding-one structure extrapolate exact modular counting far beyond the training "
                    "lengths, and how much does exact reconstruction lose under per-instrument errors?"},
    "mechanism": {
        "name": "Dayan net (residue registers + 定數/求一 reassembly table)",
        "family": "residue number system on probability rings with learned event-to-shift kernels",
        "signature_modules": ["stem_register", "branch_register", "dayan_decoder"],
        "closest_prior_art": ["Kymn et al., Computing with residue numbers in high-dimensional representation, "
                              "arXiv:2311.04872 (2023) / Neural Computation 37 (2025)",
                              "Fiete, Burak & Brookings, J Neurosci 28:6858 (2008) — grid cells as a residue code",
                              "Zaheer et al., Deep Sets, NeurIPS 2017 (size-matched baseline)"],
        "overlap": "Medium",
        "prior_art_queries": ["residue number system neural network learned moduli",
                              "grid cell modules Chinese remainder theorem decoding",
                              "length generalization modular counting recurrent",
                              "circular convolution phase register sequence model"],
        "contribution_type": "mechanism",
        "delta": "Residues are accumulated from a token stream by learned circular-convolution kernels on "
                 "small rings; the rings' joint is scattered onto the cycle by a table computed (not learned) "
                 "from Qin's definite-number reduction and rod-layout finding-one procedure, and credited in "
                 "the log domain so that each ring learns its own residue; a rod-discreteness term makes "
                 "every shift a whole cell.",
    },
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1", "property_test": "C6.1,C6.2,C6.3", "hypothesis": "H-SIG"},
        {"doctrine": "D2", "mechanism": "M2", "property_test": "C6.4", "hypothesis": "H-NEC"},
        {"doctrine": "D3", "mechanism": "M2", "property_test": "C6.5", "hypothesis": "H-SIG"},
        {"doctrine": "D4", "mechanism": "M2", "property_test": "C6.6", "hypothesis": "H-BLIND"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "On sequences two to four times longer than any seen in training the "
                                     "residue model keeps exact modular counting where the sum-pooled MLP "
                                     "baseline saturates.",
         "metric": "accuracy", "split": "shifted", "comparison": "model - baseline", "direction": "greater",
         "mesi": 0.15, "seeds": 5},
        {"id": "H-NEC", "statement": "Dropping the 乘率 from the reassembly table (衍數 alone, the calendar-"
                                     "makers' error) hurts more than removing the scalar rod accumulator.",
         "metric": "accuracy", "split": "held_out",
         "comparison": "signature_knockout - matched_knockout", "direction": "less", "mesi": 0.15, "seeds": 5},
        {"id": "H-BLIND", "statement": "Under independent per-instrument transcription errors the residue "
                                       "model degrades more, relative to its clean accuracy, than the "
                                       "magnitude-coded baseline.",
         "condition": "blind split: each instrument's record corrupted at rate 0.08",
         "grounding": "Qin removed redundancy between moduli (定數) and never checked residue consistency; "
                      "one wrong residue moves the reassembled total by a large multiple.",
         "metric": "accuracy", "split": "blind", "comparison": "(model blind - held_out) - (baseline blind - held_out)",
         "direction": "less", "mesi": 0.10, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.5, "margin_over_trivial": 0.25, "shuffled_band": 0.06,
                   "gradcheck_tol": 1e-5, "clip_norm": 5.0},
    "probe_predictions": [
        {"probe": "P4", "expected": "above baseline"},
        {"probe": "P1", "expected": "above baseline"},
        {"probe": "P8", "expected": "above baseline"},
        {"probe": "P9", "expected": "below baseline"},
    ],
    "dialectic_links": [
        {"chapter": 161, "relation": "parallel", "test": "none"},
        {"chapter": 273, "relation": "rival", "test": "none"},
        {"chapter": 323, "relation": "contemporary", "test": "none"},
    ],
    "corpus_neighbors": [
        {"chapter": 184, "similarity": 0.0, "difference": "Anania recovers one whole from one surviving fraction "
         "after named takings; here many simultaneous remainders, none of which alone bounds the whole, are reassembled."},
        {"chapter": 161, "similarity": 0.0, "difference": "Aryabhata's kuttaka is a remainder-descent generator; "
         "here the residue is the representation itself, accumulated on rings and reassembled by a table."},
        {"chapter": 273, "similarity": 0.0, "difference": "Shen Kuo reads disagreement between routes as the error "
         "signal; Qin cancels the shared part of the routes before reconstruction — exactly the blind spot tested."},
        {"chapter": 323, "similarity": 0.0, "difference": "Li Ye names the unknown and extracts it as a root of one "
         "equation; here nothing is named, the count is reassembled from remainders."},
        {"chapter": 327, "similarity": 0.0, "difference": "Tusi designs around unwanted consequences; Qin's rule has none built in."},
    ],
    "barometer": {
        "cognitive_processing": ["P4 length extrapolation of modular counting", "P1 delayed recall"],
        "embodied_cognition": [],
        "world_modeling": ["exact cyclic state tracking over long horizons"],
        "consciousness": ["mass on the reassembled cell as confidence; ECE reported per split"],
        "language_understanding": [],
        "emotional_intelligence": [],
        "creativity": [],
        "autonomy": [],
    },
    "task_types": ["sequence_classification"],
    "applications": [
        {"use": "Decoding position from grid-cell modules as a residue code", "sector": "computational neuroscience",
         "dataset": "Gardner et al. 2022 grid-cell population recordings (figshare, public)"},
        {"use": "Cyclic-phase tracking of long event streams (week/month phase of counters)",
         "sector": "operations analytics", "dataset": "Wikimedia hourly pageview dumps (public)"},
        {"use": "Frequency estimation from several undersampled channels (robust CRT)",
         "sector": "signal processing", "dataset": "synthetic multi-channel sinusoids; no downloads"},
    ],
    "safety_notes": "Research prototype of an AGI-oriented counting architecture, not an AGI. Synthetic data only. "
                    "The hostile biographical record (Zhou Mi, Liu Kezhuang) is discussed as history and encodes "
                    "nothing here.",
}

import argparse
import hashlib
import json
import math
import os
import sys
import time
from math import gcd
from typing import Dict, List, Tuple

import numpy as np

STEP_TIME_BUDGET_S = 180.0
QUICK_TIME_BUDGET_S = 20.0
FULL_STEPS = 1400
QUICK_STEPS = 250


# BEGIN STANDARD UTILITIES v1.0
def softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def logsumexp(z, axis=-1):
    m = np.max(z, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(z - m), axis=axis))


def softplus(x):
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


class Adam:
    def __init__(self, params: Dict[str, np.ndarray], lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads: Dict[str, np.ndarray]):
        self.t += 1
        b1t, b2t = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for k, p in self.params.items():
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            p -= self.lr * (self.m[k] / b1t) / (np.sqrt(self.v[k] / b2t) + self.eps)


def clip_global_norm(grads: Dict[str, np.ndarray], max_norm: float):
    total = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    if total > max_norm:
        scale = max_norm / (total + 1e-12)
        grads = {k: g * scale for k, g in grads.items()}
    return grads, total


def finite_difference_check(loss_fn, params: Dict[str, np.ndarray], rng, n_entries=20, eps=1e-6, tol=1e-5):
    """loss_fn() -> (loss, grads).  Central differences in float64 on n random entries per tensor plus the
    entry with the largest analytic gradient.  Relative error uses a 1e-4 floor so that entries whose true
    gradient is below finite-difference resolution do not report spurious failures."""
    _, grads = loss_fn()
    worst, per_tensor = 0.0, {}
    for name, p in params.items():
        flat = p.reshape(-1)
        g = grads[name].reshape(-1)
        idx = set(int(i) for i in rng.integers(0, flat.size, size=min(n_entries, flat.size)))
        idx.add(int(np.argmax(np.abs(g))))
        tmax = 0.0
        for i in sorted(idx):
            old = flat[i]
            flat[i] = old + eps
            lp, _ = loss_fn()
            flat[i] = old - eps
            lm, _ = loss_fn()
            flat[i] = old
            fd = (lp - lm) / (2 * eps)
            rel = abs(fd - g[i]) / max(abs(fd), abs(g[i]), 1e-4)
            tmax = max(tmax, rel)
        per_tensor[name] = tmax
        worst = max(worst, tmax)
    return worst <= tol, worst, per_tensor


def paired_bootstrap(diffs: np.ndarray, rng, n_resample=2000):
    diffs = np.asarray(diffs, dtype=np.float64)
    n = diffs.size
    means = np.empty(n_resample)
    for i in range(n_resample):
        means[i] = np.mean(diffs[rng.integers(0, n, size=n)])
    return float(np.mean(diffs)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def write_report(report: dict, json_path=None):
    ch = report["chapter"]
    print(f"=== VERIFIED REPORT · chapter {ch:04d} ===")
    env = report["environment"]
    print(f"file: {report['file']}   python {env['python']}   numpy {env['numpy']}")
    print(f"seeds: {report['seeds']}   runtime: {report['runtime_s']:.1f}s   n_params: {report['n_params']}")
    gc = report["gradcheck"]
    print(f"gradcheck: {gc['tensors_checked']}/{gc['tensors_total']} tensors, max rel err {gc['max_rel_error']:.2e}, "
          f"checked at {gc['checked_at']}, passed={gc['passed']}")
    print("correctness:")
    for c in report["correctness"]:
        print(f"  {c['id']:<5} {c['name']:<34} {'PASS' if c['passed'] else 'FAIL'}  {c['detail']}")
    mu = report["mutants"]
    print(f"mutants: detected {mu['detected']}/{mu['total']}  score {mu['score']:.2f}")
    print("hypotheses:")
    for h in report["hypotheses"]:
        print(f"  {h['id']:<8} {h['metric']:<10} mean_diff {h['mean_diff']:+.3f}  ci95 [{h['ci95'][0]:+.3f}, "
              f"{h['ci95'][1]:+.3f}]  mesi {h['mesi']:.2f}  n={h['n_seeds']}  -> {h['verdict']}")
    print("knockouts (accuracy change on held-out):")
    for k in report["knockouts"]:
        print(f"  {k['module']:<22} signature={str(k['signature']):<5} change {k['metric_change']:+.3f}  "
              f"ci95 [{k['ci95'][0]:+.3f}, {k['ci95'][1]:+.3f}]")
    for line in report.get("extra_lines", []):
        print(line)
    print(f"task_types: {report['task_types']}   exit_code: {report['exit_code']}")
    print("=== END REPORT ===")
    if json_path:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1, ensure_ascii=False)
# END STANDARD UTILITIES


# ---------------------------------------------------------------------------
# 6. Data and tasks — the Grand Epoch ledger
# ---------------------------------------------------------------------------
V_SYMBOLS = 12
N_SILENT = 4
MODULI = (10, 12)          # 天干 stems, 地支 branches
CYCLE = 60                 # the sexagenary day cycle 干支 = lcm(10, 12)
N_CLASSES = CYCLE
TASK_TYPES = ["sequence_classification"]
SPLITS = {"train": (1, 32, 0.0), "val": (1, 32, 0.0), "held_out": (1, 32, 0.0),
          "shifted": (64, 120, 0.0), "blind": (8, 32, 0.08)}
SPLIT_SIZES = {"train": 3000, "val": 500, "held_out": 1000, "shifted": 1000, "blind": 1000}


def make_world(rng):
    """Hidden increments (both signs present) and one symbol permutation per instrument."""
    inc = np.zeros(V_SYMBOLS, dtype=np.int64)
    while True:
        vals = rng.integers(-4, 5, size=V_SYMBOLS - N_SILENT)
        vals[vals == 0] = rng.choice([-1, 1], size=int(np.sum(vals == 0)))
        if (vals > 0).any() and (vals < 0).any():
            break
    inc[N_SILENT:] = vals
    p = np.full(V_SYMBOLS, 0.6 / (V_SYMBOLS - N_SILENT))
    p[:N_SILENT] = 0.4 / N_SILENT
    return {"inc": inc, "p": p, "perms": [rng.permutation(V_SYMBOLS) for _ in MODULI]}


def seq_hash(tokens):
    return hashlib.sha1(np.asarray(tokens, dtype=np.int64).tobytes()).hexdigest()


def sample_split(world, rng, n, t_min, t_max, corrupt, forbidden):
    """Sequences of hidden events, each recorded by every instrument through its own symbol map.
    Returns padded multi-hot inputs, the day of the 60-cycle as label, lengths, true counts and hashes."""
    K, D = len(MODULI), len(MODULI) * V_SYMBOLS
    X = np.zeros((n, t_max, D))
    y, lengths, counts, hashes = np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros(n, np.int64), []
    i = 0
    while i < n:
        L = int(rng.integers(t_min, t_max + 1))
        tokens = rng.choice(V_SYMBOLS, size=L, p=world["p"])
        h = seq_hash(tokens)
        if h in forbidden:
            continue
        N = int(np.sum(world["inc"][tokens]))
        for k in range(K):
            obs = world["perms"][k][tokens].copy()
            if corrupt > 0:
                bad = rng.random(L) < corrupt
                obs[bad] = rng.integers(0, V_SYMBOLS, size=int(bad.sum()))
            X[i, np.arange(L), k * V_SYMBOLS + obs] = 1.0
        y[i], lengths[i], counts[i] = N % CYCLE, L, N
        hashes.append(h)
        i += 1
    return {"X": X, "y": y, "lengths": lengths, "counts": counts, "hashes": set(hashes)}


def make_data(seed, sizes=None):
    sizes = sizes or SPLIT_SIZES
    ss = np.random.SeedSequence(seed)
    rngs = [np.random.default_rng(s) for s in ss.spawn(len(SPLITS) + 1)]
    world = make_world(rngs[0])
    data, forbidden = {"world": world}, set()
    for r, (name, (t_min, t_max, corrupt)) in zip(rngs[1:], SPLITS.items()):
        data[name] = sample_split(world, r, sizes[name], t_min, t_max, corrupt, forbidden)
        forbidden = forbidden | data[name]["hashes"]
    return data


def trivial_accuracy(data):
    maj = int(np.argmax(np.bincount(data["train"]["y"], minlength=N_CLASSES)))
    return {k: float(np.mean(data[k]["y"] == maj)) for k in SPLITS if k != "train"}


# ---------------------------------------------------------------------------
# 7. Model — the Dayan net
# ---------------------------------------------------------------------------
def qiuyi(a, m):
    """大衍求一術 on Qin's 2x2 rod layout.  Upper-left: 天元 (1); upper-right: 奇 (a); lower-right: 定 (m).
    Divide alternately; when the upper-right holds 1 the upper-left is the 乘率.  If 1 first appears in
    the lower-right, one more (odd-numbered) division with quotient reduced by one moves it up."""
    a %= m
    if m == 1 or a == 1:
        return 1 % m if m > 1 else 0
    if a == 0:
        raise ValueError("奇 and 定 share a factor")
    ul, ur, ll, lr = 1, a, 0, m
    while ur != 1:
        q, r = divmod(lr, ur)
        if r == 0:
            raise ValueError("奇 and 定 share a factor")
        ll, lr = ll + q * ul, r
        if lr == 1:
            ul, ur = ul + (ur - 1) * ll, 1
            break
        q, r = divmod(ur, lr)
        if r == 0:
            raise ValueError("奇 and 定 share a factor")
        ul, ur = ul + q * ll, r
    return ul % m


def qiuyi_wrong_cell(a, m):
    """Negative control: stops as soon as any right-hand cell holds 1 and reads the adjacent left cell."""
    a %= m
    if m == 1 or a == 1:
        return 1 % m if m > 1 else 0
    ul, ur, ll, lr = 1, a, 0, m
    while ur != 1 and lr != 1 and ur != 0:
        q, r = divmod(lr, ur)
        ll, lr = ll + q * ul, r
        if lr == 1:
            return ll % m
        q, r = divmod(ur, lr)
        ul, ur = ul + q * ll, r
    return ul % m


def reduce_pair(mi, mj):
    """Mutual reduction of two moduli: strip their common factor from one, then push shared prime powers
    to the other until they are coprime (the procedure Qin applies pairwise; Seki's 1683 form)."""
    d = gcd(mi, mj)
    if d == 1:
        return mi, mj
    mj //= d
    while True:
        d2 = gcd(mi, mj)
        if d2 == 1:
            return mi, mj
        mi, mj = mi // d2, mj * d2


def dingshu(moduli):
    """定數: pairwise-coprime definite numbers, each dividing its original, product = lcm."""
    mods = list(moduli)
    changed = True
    while changed:
        changed = False
        for i in range(len(mods)):
            for j in range(i + 1, len(mods)):
                a, b = reduce_pair(mods[i], mods[j])
                if (a, b) != (mods[i], mods[j]):
                    mods[i], mods[j], changed = a, b, True
    return mods


def lcm_all(xs):
    out = 1
    for x in xs:
        out = out * x // gcd(out, x)
    return out


def qin_table(moduli, multiplier="qiuyi"):
    """The 大衍 reassembly table.  定數 μ_j strip shared factors (product = lcm = M); the 衍數 M/μ_j and
    its 乘率 κ_j (求一) give the 用數 y_j = (M/μ_j)·κ_j; residues r_j reassemble as N = Σ r_j·y_j (mod M).
    Returns the table cell[r_1, ..., r_K] over the ORIGINAL rings (r_j mod μ_j is implicit), μ and M.
    multiplier='one' is the naive control: 衍數 without 求一 (κ_j = 1); it is a wrong bijection."""
    mu = dingshu(moduli)
    M = int(np.prod(mu))
    yong = []
    for u in mu:
        if u == 1:
            yong.append(0)
            continue
        kappa = qiuyi((M // u) % u, u) if multiplier == "qiuyi" else 1
        yong.append((M // u) * kappa % M)
    grids = np.meshgrid(*[np.arange(m) for m in moduli], indexing="ij")
    cell = np.zeros(grids[0].shape, dtype=np.int64)
    for g, y in zip(grids, yong):
        cell = (cell + g * y) % M
    return cell, mu, M, yong

def shift_index(M):
    n = np.arange(M)
    return (n[:, None] - n[None, :]) % M, (n[:, None] + n[None, :]) % M

def circ_conv(s, k, idx):
    """conv[b, n] = Σ_m s[b, m]·k[b, (n − m) mod M]: the rods advance by the shift distribution k."""
    return np.einsum("bm,bnm->bn", s, k[:, idx])

def circ_conv_backward(dconv, s, k, idx, idx_plus):
    ds = np.einsum("bn,bnm->bm", dconv, k[:, idx])
    dk = np.einsum("bm,bim->bi", s, dconv[:, idx_plus])
    return ds, dk

def outer_all(vs):
    """Joint distribution over all rings: J[b, r_1, ..., r_K] = Π_j s_j[b, r_j]."""
    J = vs[0]
    for s in vs[1:]:
        J = J.reshape(J.shape[0], -1)[:, :, None] * s[:, None, :]
    return J.reshape((vs[0].shape[0],) + tuple(v.shape[1] for v in vs))


MUTANT = None
MUTANTS = ["flip_sign", "zero_lr", "zero_grad_stem", "bad_conv_grad"]
LOG_EPS = 1e-6
SNAP_WEIGHT = 0.1          # rod-discreteness term: λ·mean_t(1 − Σ_r k_t[r]²), zero when every shift is one cell
RING_NAMES = {10: "stem", 12: "branch"}

class DayanNet:
    """M1 residue registers (one ring per instrument, advanced by learned shift distributions), M2 the
    求一 reassembly (joint of the rings scattered onto the M-cycle through Qin's table), M3 a scalar rod
    accumulator (non-signature) and M4 a tally readout on the M-cycle (non-signature)."""

    def __init__(self, in_dim, out_dim, moduli, rng, init="qin", rod_tau=8.0, w_std=0.3, still_logit=2.0,
                 gain0=1.0, tally_logit=8.0):
        self.in_dim, self.out_dim, self.moduli, self.tau = in_dim, out_dim, tuple(moduli), rod_tau
        K = len(self.moduli)
        if in_dim % K:
            raise ValueError("each ring reads one instrument: in_dim must be a multiple of the ring count")
        self.V = in_dim // K
        self.names = [RING_NAMES.get(m, f"ring{j}") for j, m in enumerate(self.moduli)]
        self.idx = [shift_index(m) for m in self.moduli]
        cell, self.mu, self.M, self.yong = qin_table(self.moduli)
        if out_dim != self.M:
            raise ValueError(f"the cycle has {self.M} cells; out_dim must equal it")
        naive, _, _, _ = qin_table(self.moduli, multiplier="one")
        self.S = np.eye(self.M)[cell.reshape(-1)]           # scatter matrix (Πm × M): pairs → cells
        self.S_naive = np.eye(self.M)[naive.reshape(-1)]    # knockout: 衍數 without 乘率
        if init == "random":                                  # control: a random 2-to-1 table, no group structure
            self.S = np.eye(self.M)[rng.integers(0, self.M, size=cell.size)]
        self.idxM = shift_index(self.M)
        self.params = {}
        for name, m in zip(self.names, self.moduli):
            W = rng.normal(0, w_std, size=(self.V, m))
            W[:, 0] += still_logit          # rods start still: every symbol first means "no advance"
            self.params[f"W_{name}"] = W
        self.params["c_tally"] = np.zeros(self.M)            # tally kernel over neighbouring cells
        self.params["c_tally"][0] = tally_logit              # (init: nearly the identity kernel)
        self.params["gain"] = np.array([gain0])
        self.params["b"] = np.zeros(out_dim)
        self.params["u_rod"] = rng.normal(0, w_std, size=in_dim)
        self.params["q_rod"] = rng.normal(0, w_std, size=out_dim)
        self.knock = {}
        self.test_break = None
        self.snap_scale = 1.0

    # -- M1 registers ---------------------------------------------------------
    def run_registers(self, X, valid, init_shift=0):
        """Advance every ring through the sequence; returns per-ring (kernels, state trajectory)."""
        B, T, _ = X.shape
        out = []
        for j, (name, m) in enumerate(zip(self.names, self.moduli)):
            logits = X[:, :, j * self.V:(j + 1) * self.V] @ self.params[f"W_{name}"]
            if self.test_break == "positional":
                logits = logits + 0.5 * np.arange(T)[None, :, None] * np.arange(m)[None, None, :] / m
            k = softmax(logits)
            s = np.zeros((B, m))
            s[:, init_shift % m] = 1.0
            states, mode = [s], self.knock.get(f"{name}_register")
            for t in range(T):
                v = valid[:, t:t + 1]
                if mode == "identity":
                    step = s + k[:, t]
                elif mode == "zero":
                    step = s
                elif mode == "mean":
                    step = np.full_like(s, 1.0 / m)
                elif self.test_break == "linear":
                    tri = (np.arange(m)[None, :] <= np.arange(m)[:, None])[None, :, :]
                    step = np.einsum("bm,bnm->bn", s, np.where(tri, k[:, t][:, self.idx[j][0]], 0.0))
                else:
                    step = circ_conv(s, k[:, t], self.idx[j][0])
                s = v * step + (1 - v) * s
                states.append(s)
            out.append((k, states))
        return out

    # -- M2 求一 reassembly ----------------------------------------------------
    def reassemble(self, finals):
        """s_M[n] = Σ_{(r_1..r_K) ↦ n} Π_j s_j[r_j]: every residue combination lands on its cell of the cycle.
        Pairs that are inconsistent modulo the shared factor land on cells too (the blind spot)."""
        J = outer_all(finals).reshape(finals[0].shape[0], -1)
        mode = self.knock.get("dayan_decoder")
        if mode == "identity":
            return J, J @ self.S_naive
        if mode == "zero":
            return J, np.zeros((J.shape[0], self.M))
        if mode == "mean":
            return J, np.full((J.shape[0], self.M), 1.0 / self.M)
        return J, J @ self.S

    def forward(self, X, lengths):
        B, T, _ = X.shape
        valid = (np.arange(T)[None, :] < lengths[:, None]).astype(np.float64)
        regs = self.run_registers(X, valid)
        finals = [states[-1] for _, states in regs]
        J, sM = self.reassemble(finals)
        kappa = softmax(self.params["c_tally"])
        mode = self.knock.get("tally_readout")
        if mode == "identity":
            r = sM
        elif mode == "zero":
            r = np.zeros_like(sM)
        elif mode == "mean":
            r = np.full_like(sM, 1.0 / self.M)
        else:
            r = circ_conv(sM, np.broadcast_to(kappa, sM.shape), self.idxM[0])
        c = np.einsum("btd,d->b", X * valid[:, :, None], self.params["u_rod"])
        mode = self.knock.get("rod_accumulator")
        g = np.tanh(c / self.tau) if mode is None else (np.zeros_like(c) if mode == "zero" else
                                                          (c / self.tau if mode == "identity" else np.full_like(c, 0.5)))
        lr_ = np.log(r + LOG_EPS)        # log domain: each ring's residue is credited on its own (see docstring)
        logits = self.params["gain"][0] * lr_ + g[:, None] * self.params["q_rod"][None, :] + self.params["b"]
        cache = {"valid": valid, "regs": regs, "finals": finals, "J": J, "sM": sM, "kappa": kappa, "r": r,
                 "lr": lr_, "c": c, "g": g, "X": X}
        return logits, cache

    def snap_term(self, cache):
        """Rod discreteness: rods never sit in two cells at once.  Gini impurity of every shift kernel
        over the valid tokens, with its gradient through the softmax (the only place W enters twice)."""
        valid, X, total, gs = cache["valid"], cache["X"], 0.0, {}
        n = max(1.0, float(valid.sum()))
        for j, name in enumerate(self.names):
            k = cache["regs"][j][0]
            total += float(np.sum(valid * (1.0 - np.sum(k ** 2, axis=-1)))) / n
            dk = -2.0 * k * valid[:, :, None] / n
            dlog = k * (dk - np.sum(dk * k, axis=-1, keepdims=True))
            gs[f"W_{name}"] = np.einsum("btd,btm->dm", X[:, :, j * self.V:(j + 1) * self.V], dlog)
        return total, gs

    def backward(self, dlogits, cache):
        """Hand-derived reverse pass: readout → tally kernel → scatter → joint → rings → shift kernels."""
        P, g = self.params, {k: np.zeros_like(v) for k, v in self.params.items()}
        X, valid, finals = cache["X"], cache["valid"], cache["finals"]
        g["gain"][0] = np.sum(dlogits * cache["lr"])
        g["b"] = dlogits.sum(0)
        g["q_rod"] = cache["g"] @ dlogits
        mode = self.knock.get("rod_accumulator")
        dc = (dlogits @ P["q_rod"]) * ((1 - cache["g"] ** 2) if mode is None else float(mode == "identity")) / self.tau
        g["u_rod"] = np.einsum("b,btd->d", dc, X * valid[:, :, None])
        dr = P["gain"][0] * dlogits / (cache["r"] + LOG_EPS)
        mode = self.knock.get("tally_readout")
        if mode is None:
            kap = np.broadcast_to(cache["kappa"], dr.shape)
            dsM, dkap = circ_conv_backward(dr, cache["sM"], kap, self.idxM[0], self.idxM[1])
            dkap = dkap.sum(0)
            g["c_tally"] = cache["kappa"] * (dkap - np.sum(dkap * cache["kappa"]))
        else:
            dsM = dr if mode == "identity" else np.zeros_like(dr)
        dJ = (dsM @ self.S.T).reshape((dr.shape[0],) + tuple(self.moduli))
        for j, (name, m) in enumerate(zip(self.names, self.moduli)):
            others = [f for i, f in enumerate(finals) if i != j]
            O = np.expand_dims(outer_all(others) if others else np.ones((dr.shape[0],)), axis=j + 1)
            ds = np.sum(dJ * O, axis=tuple(a for a in range(1, len(self.moduli) + 1) if a != j + 1))
            k, states = cache["regs"][j]
            dk = np.zeros_like(k)
            idx, idx_plus = self.idx[j]
            if MUTANT == "bad_conv_grad":
                idx_plus = idx
            for t in range(len(states) - 2, -1, -1):
                v = valid[:, t:t + 1]
                ds_prev, dkt = circ_conv_backward(ds * v, states[t], k[:, t], idx, idx_plus)
                dk[:, t] = dkt
                ds = ds_prev + (1 - v) * ds
            dlog = k * (dk - np.sum(dk * k, axis=-1, keepdims=True))
            g[f"W_{name}"] = np.einsum("btd,btm->dm", X[:, :, j * self.V:(j + 1) * self.V], dlog)
        if MUTANT == "zero_grad_stem":
            g[f"W_{self.names[0]}"] = np.zeros_like(g[f"W_{self.names[0]}"])
        return g


# ===== 8. Baseline — size-matched Deep Sets (sum pooling + MLP) =====
class DeepSetsBaseline:
    def __init__(self, in_dim, out_dim, rng, d=5, h=5):
        self.in_dim, self.out_dim = in_dim, out_dim
        self.params = {"E": rng.normal(0, 0.3, size=(in_dim, d)), "W1": rng.normal(0, 0.3, size=(d, h)),
                       "b1": np.zeros(h), "W2": rng.normal(0, 0.3, size=(h, out_dim)), "b2": np.zeros(out_dim)}
        self.knock = {}

    def forward(self, X, lengths):
        valid = (np.arange(X.shape[1])[None, :] < lengths[:, None]).astype(np.float64)
        pooled = np.einsum("btd,de->be", X * valid[:, :, None], self.params["E"])
        hid = np.tanh(pooled @ self.params["W1"] + self.params["b1"])
        logits = hid @ self.params["W2"] + self.params["b2"]
        return logits, {"valid": valid, "pooled": pooled, "hid": hid, "X": X}

    def backward(self, dlogits, cache):
        P, g = self.params, {}
        g["W2"], g["b2"] = cache["hid"].T @ dlogits, dlogits.sum(0)
        dh = (dlogits @ P["W2"].T) * (1 - cache["hid"] ** 2)
        g["W1"], g["b1"] = cache["pooled"].T @ dh, dh.sum(0)
        dp = dh @ P["W1"].T
        g["E"] = np.einsum("btd,be->de", cache["X"] * cache["valid"][:, :, None], dp)
        return g


# ===== 9. Standard interface =====
def build_model(in_dim, out_dim, task_type="sequence_classification", rng=None, **cfg):
    if task_type not in TASK_TYPES:
        raise ValueError(f"unsupported task type {task_type}")
    rng = rng or np.random.default_rng(0)
    if cfg.pop("baseline", False):
        return DeepSetsBaseline(in_dim, out_dim, rng, **cfg)
    return DayanNet(in_dim, out_dim, cfg.pop("moduli", MODULI), rng, **cfg)

def loss_and_grads(model, batch):
    X, y, lengths = batch["X"], batch["y"], batch["lengths"]
    logits, cache = model.forward(X, lengths)
    lse = logsumexp(logits, axis=-1)
    scale = 1.0 / (len(y) * math.log(logits.shape[1]))    # cross-entropy in units of log C (uniform = 1.0)
    loss = float(np.sum(lse - logits[np.arange(len(y)), y])) * scale
    dlogits = softmax(logits)
    dlogits[np.arange(len(y)), y] -= 1.0
    grads = model.backward(dlogits * scale, cache)
    w = SNAP_WEIGHT * getattr(model, "snap_scale", 1.0)
    if isinstance(model, DayanNet) and w > 0:
        snap, gs = model.snap_term(cache)
        loss += w * snap
        for k, v in gs.items():
            grads[k] = grads[k] + w * v
    return loss, grads

def predict(model, X, lengths=None):
    lengths = lengths if lengths is not None else np.full(X.shape[0], X.shape[1])
    logits, _ = model.forward(X, lengths)
    return softmax(logits)

def hidden_states(model, X, lengths=None):
    lengths = lengths if lengths is not None else np.full(X.shape[0], X.shape[1])
    _, cache = model.forward(X, lengths)
    if "finals" in cache:
        return np.concatenate(cache["finals"] + [cache["sM"]], axis=1)
    return np.concatenate([cache["pooled"], cache["hid"]], axis=1)

def n_params(model):
    return int(sum(v.size for v in model.params.values()))

def modules(model):
    if isinstance(model, DeepSetsBaseline):
        return {"pooled_mlp": {"params": list(model.params), "role": "sum-pooled token embedding with tanh MLP",
                               "signature": False}}
    reg = {}
    for name in model.names:
        reg[f"{name}_register"] = {"params": [f"W_{name}"], "signature": True,
                                   "role": "cyclic-shift accumulator (circular convolution on a small ring)"}
    reg["dayan_decoder"] = {"params": [], "signature": True,
                            "role": "joint of the rings scattered onto the cycle by the 定數/求一 table (no free parameters)"}
    reg["rod_accumulator"] = {"params": ["u_rod", "q_rod"], "signature": False,
                              "role": "linear scalar accumulator with tanh squash"}
    reg["tally_readout"] = {"params": ["c_tally", "gain", "b"], "signature": False,
                            "role": "learned circular smoothing of the cycle distribution, gain and bias"}
    return reg

def knockout(model, name, mode="identity"):
    """A shallow clone sharing parameters (read-only use) with one module switched to the given mode."""
    if name not in modules(model):
        raise KeyError(name)
    m2 = object.__new__(type(model))
    m2.__dict__.update(model.__dict__)
    m2.knock = dict(model.knock, **{name: mode})
    return m2


# ===== 10. Training =====
def batch_of(split, idx):
    return {"X": split["X"][idx], "y": split["y"][idx], "lengths": split["lengths"][idx]}

def accuracy(model, split, chunk=250):
    correct, n = 0, len(split["y"])
    for i in range(0, n, chunk):
        correct += int(np.sum(np.argmax(predict(model, split["X"][i:i + chunk], split["lengths"][i:i + chunk]), axis=1) == split["y"][i:i + chunk]))
    return correct / n

def ece(model, split, chunk=250, bins=10):
    conf, hit = [], []
    for i in range(0, len(split["y"]), chunk):
        p = predict(model, split["X"][i:i + chunk], split["lengths"][i:i + chunk])
        conf.append(np.max(p, axis=1)); hit.append(np.argmax(p, axis=1) == split["y"][i:i + chunk])
    conf, hit = np.concatenate(conf), np.concatenate(hit).astype(float)
    edges, out = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any(): out += sel.mean() * abs(conf[sel].mean() - hit[sel].mean())
    return float(out)

def fit(model, data, budget, rng, lr=0.02, batch_size=32, clip=5.0, eval_every=100, warm=0.5):
    """Adam under a fixed update budget; model selection on the validation split only.  Length curriculum
    (same for every system): during the first `warm` fraction of the budget the admissible sequence length
    grows linearly from 2 to the training maximum — short spans first, as a calendar is fitted."""
    if MUTANT == "zero_lr":
        lr = 0.0
    opt, train, losses = Adam(model.params, lr=lr), data["train"], []
    best_acc, best_params, n = -1.0, None, len(train["y"])
    order, L = np.argsort(train["lengths"], kind="stable"), train["lengths"]
    for step in range(1, budget + 1):
        frac = step / max(1.0, warm * budget)
        lmax = 2 + (L.max() - 2) * min(1.0, frac)
        model.snap_scale = max(0.0, min(1.0, frac - 1.0))   # rods click only after the constants are placed
        pool = order[:max(batch_size, int(np.searchsorted(L[order], lmax, side="right")))]
        loss, grads = loss_and_grads(model, batch_of(train, pool[rng.integers(0, len(pool), size=batch_size)]))
        if not np.isfinite(loss):
            print("non-finite loss", file=sys.stderr)
            sys.exit(4)
        if MUTANT == "flip_sign":
            grads = {k: -v for k, v in grads.items()}
        grads, _ = clip_global_norm(grads, clip)
        opt.step(grads)
        losses.append(loss)
        if step % eval_every == 0 or step == budget:
            acc = accuracy(model, data["val"])
            if acc >= best_acc:
                best_acc, best_params = acc, {k: v.copy() for k, v in model.params.items()}
    if best_params is not None:
        for k in model.params:
            model.params[k][...] = best_params[k]
    return losses


# ===== 11. Tests — correctness first, then hypotheses =====
def gradcheck_model(model, data, rng, n_entries=20):
    idx = np.where(data["train"]["lengths"] <= 14)[0][:6]     # whole sequences, so labels stay true
    batch = batch_of(data["train"], idx)
    batch = {"X": batch["X"][:, :14], "y": batch["y"], "lengths": batch["lengths"]}
    return finite_difference_check(lambda: loss_and_grads(model, batch), model.params, rng, n_entries=n_entries)

def test_c1(data, rng, steps):
    D, C = data["train"]["X"].shape[2], N_CLASSES
    results = {}
    for name, kw in (("dayan", {}), ("baseline", {"baseline": True})):
        model = build_model(D, C, rng=np.random.default_rng(11), **kw)
        ok0, w0, _ = gradcheck_model(model, data, rng)
        fit(model, data, steps, np.random.default_rng(12))
        ok1, w1, per = gradcheck_model(model, data, rng)
        results[name] = {"init": float(w0), "after": float(w1), "passed": bool(ok0 and ok1), "tensors": len(per)}
    return results

def test_c2(data):
    D, C = data["train"]["X"].shape[2], N_CLASSES
    outs = []
    for _ in range(2):
        m = build_model(D, C, rng=np.random.default_rng(5))
        losses = fit(m, data, 40, np.random.default_rng(6))
        outs.append((np.array(losses), predict(m, data["val"]["X"][:50], data["val"]["lengths"][:50])))
    same = np.array_equal(outs[0][0], outs[1][0]) and np.array_equal(outs[0][1], outs[1][1])
    finite = all(np.all(np.isfinite(v)) for v in m.params.values()) and np.all(np.isfinite(outs[1][1]))
    return same and finite, f"identical losses/outputs={same}, finite={finite}"

def test_c4(data, rng, steps):
    D, C = data["train"]["X"].shape[2], N_CLASSES
    shuffled = {k: v for k, v in data.items()}
    tr = dict(data["train"])
    tr["y"] = rng.permutation(tr["y"])
    shuffled["train"] = tr
    m = build_model(D, C, rng=np.random.default_rng(21))
    fit(m, shuffled, steps, np.random.default_rng(22))
    acc, band = accuracy(m, data["held_out"]), trivial_accuracy(data)["held_out"]
    return acc <= band + MIND_CARD["thresholds"]["shuffled_band"], acc, band

def test_c5(data, steps):
    """Each registered mutant must break C1 or C3."""
    global MUTANT
    D, C = data["train"]["X"].shape[2], N_CLASSES
    detected, detail = 0, []
    for name in MUTANTS:
        MUTANT = name
        m = build_model(D, C, rng=np.random.default_rng(31))
        ok, _, _ = gradcheck_model(m, data, np.random.default_rng(32))
        losses = fit(m, data, steps, np.random.default_rng(33))
        drop = 1 - np.mean(losses[-20:]) / np.mean(losses[:20])
        acc = accuracy(m, data["held_out"])
        c3 = drop >= MIND_CARD["thresholds"]["loss_drop_fraction"] and \
            acc >= trivial_accuracy(data)["held_out"] + MIND_CARD["thresholds"]["margin_over_trivial"]
        caught = (not ok) or (not c3)
        detected += int(caught)
        detail.append(f"{name}:{'caught' if caught else 'MISSED'}(C1={'ok' if ok else 'fail'},C3={'ok' if c3 else 'fail'})")
    MUTANT = None
    return detected, len(MUTANTS), " ".join(detail)

def rand_inputs(rng, B=16, T=12, D=len(MODULI) * V_SYMBOLS):
    X = np.zeros((B, T, D))
    for k in range(len(MODULI)):
        X[np.arange(B)[:, None], np.arange(T)[None, :], k * V_SYMBOLS + rng.integers(0, V_SYMBOLS, size=(B, T))] = 1.0
    return X, rng.integers(4, T + 1, size=B)

def test_c6(rng):
    """Property tests on random parameters and inputs, each with a negative control that must fail."""
    D, C, res = len(MODULI) * V_SYMBOLS, N_CLASSES, {}
    worst_s, worst_e, worst_o, ctrl_s, ctrl_e, ctrl_o = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for trial in range(12):
        m = build_model(D, C, rng=np.random.default_rng(100 + trial), still_logit=0.0)
        for p in m.params.values():
            p *= rng.uniform(0.2, 4.0)
        X, L = rand_inputs(rng)
        valid = (np.arange(X.shape[1])[None, :] < L[:, None]).astype(float)
        regs = m.run_registers(X, valid)
        for _, states in regs:
            worst_s = max(worst_s, float(np.max(np.abs(states[-1].sum(1) - 1))), float(-states[-1].min()))
        for j, (_, states) in enumerate(m.run_registers(X, valid, init_shift=3)):
            worst_e = max(worst_e, float(np.max(np.abs(states[-1] - np.roll(regs[j][1][-1], 3, axis=1)))))
        Xp = X.copy()
        for bi in range(X.shape[0]):
            perm = rng.permutation(L[bi])
            Xp[bi, :L[bi]] = X[bi, perm]
        for j, (_, states) in enumerate(m.run_registers(Xp, valid)):
            worst_o = max(worst_o, float(np.max(np.abs(states[-1] - regs[j][1][-1]))))
        ko = knockout(m, "stem_register", "identity")
        ctrl_s = max(ctrl_s, float(np.max(np.abs(ko.run_registers(X, valid)[0][1][-1].sum(1) - 1))))
        m.test_break = "linear"
        lin = m.run_registers(X, valid)[0][1][-1]
        lin3 = m.run_registers(X, valid, init_shift=3)[0][1][-1]
        ctrl_e = max(ctrl_e, float(np.max(np.abs(lin3 - np.roll(lin, 3, axis=1)))))
        m.test_break = "positional"
        ctrl_o = max(ctrl_o, float(np.max(np.abs(m.run_registers(Xp, valid)[0][1][-1] - m.run_registers(X, valid)[0][1][-1]))))
        m.test_break = None
    res["C6.1"] = (worst_s < 1e-9 and ctrl_s > 1e-3, f"simplex dev {worst_s:.1e}; control dev {ctrl_s:.2f}")
    res["C6.2"] = (worst_e < 1e-9 and ctrl_e > 1e-3, f"equivariance dev {worst_e:.1e}; control dev {ctrl_e:.2f}")
    res["C6.3"] = (worst_o < 1e-9 and ctrl_o > 1e-3, f"order-invariance dev {worst_o:.1e}; control dev {ctrl_o:.2f}")
    res.update(test_c6_arithmetic(rng))
    return res

def test_c6_arithmetic(rng):
    """C6.4 求一, C6.5 定數 and the reassembly table, C6.6 the silent inconsistency (definition check)."""
    ok4, bad4 = True, 0
    for _ in range(400):
        m = int(rng.integers(3, 5000))
        a = int(rng.integers(2, m))
        if gcd(a, m) != 1:
            continue
        ok4 &= (a * qiuyi(a, m)) % m == 1
        bad4 += int((a * qiuyi_wrong_cell(a, m)) % m != 1)
    ok4 &= qiuyi(377873, 499067) == 457999 and dingshu([54, 57, 75, 72]) == [27, 19, 25, 8]
    ok5, naive_fail, exact = True, 0, 0
    for _ in range(300):
        mods = [int(x) for x in rng.integers(2, 40, size=int(rng.integers(2, 4)))]
        cell, mu, M, _ = qin_table(mods)
        naive, _, _, _ = qin_table(mods, multiplier="one")
        ok5 &= all(mods[i] % mu[i] == 0 for i in range(len(mods))) and int(np.prod(mu)) == M == lcm_all(mods)
        ok5 &= all(gcd(mu[i], mu[j]) == 1 for i in range(len(mu)) for j in range(i + 1, len(mu)))
        for n in rng.integers(0, 10 ** 6, size=5):
            r = tuple(int(n) % mm for mm in mods)
            ok5 &= int(cell[r]) == int(n) % M
            naive_fail += int(int(naive[r]) != int(n) % M)
        exact += 1
    cell, _, M, _ = qin_table(MODULI)
    silent = all(0 <= cell[r0, r1] < M for r0 in range(MODULI[0]) for r1 in range(MODULI[1]) if (r0 - r1) % 2)
    return {"C6.4": (ok4 and bad4 > 0, f"求一 exact on random and on 457999; wrong-cell control fails {bad4}x"),
            "C6.5": (ok5 and naive_fail > 0, f"定數 invariants and reassembly exact on {exact} moduli sets; "
                                              f"衍數-without-乘率 control fails {naive_fail}x"),
            "C6.6": (silent, "definition check (not evidence): every inconsistent residue pair lands on a cell")}

def verdict(diffs, direction, mesi, rng):
    mean, lo, hi = paired_bootstrap(np.array(diffs), rng)
    if direction == "greater":
        v = "supported" if lo > 0 and mean >= mesi else ("contradicted" if hi < 0 else "inconclusive")
    else:
        v = "supported" if hi < 0 and -mean >= mesi else ("contradicted" if lo > 0 else "inconclusive")
    return mean, (lo, hi), v

def rod_discreteness(model, data):
    """Probe: fraction of (instrument, symbol) shift kernels whose peak holds ≥ 0.9 of the mass once folded
    modulo the ring's 定數, and the fraction whose peak is the true increment modulo that 定數."""
    world, sharp, right, total = data["world"], 0, 0, 0
    for j, (name, m) in enumerate(zip(model.names, model.moduli)):
        k = softmax(np.eye(model.V) @ model.params[f"W_{name}"])
        for v in range(V_SYMBOLS):
            row = k[world["perms"][j][v]].reshape(-1, model.mu[j]).sum(0)   # folded modulo the 定數
            sharp += int(row.max() >= 0.9)
            right += int((int(np.argmax(row)) - int(world["inc"][v])) % model.mu[j] == 0)
            total += 1
    return sharp / total, right / total

def run_seed(seed, data, steps):
    """Train the model and the baseline for one seed and collect every metric used by the hypotheses."""
    D, C = data["train"]["X"].shape[2], N_CLASSES
    out = {"seed": seed}
    for name, kw in (("model", {}), ("baseline", {"baseline": True})):
        m = build_model(D, C, rng=np.random.default_rng(seed * 7 + 1), **kw)
        losses = fit(m, data, steps, np.random.default_rng(seed * 7 + 2))
        out[name] = {"loss_drop": 1 - np.mean(losses[-20:]) / np.mean(losses[:20]),
                     "acc": {k: accuracy(m, data[k]) for k in ("held_out", "shifted", "blind")},
                     "ece": {k: ece(m, data[k]) for k in ("held_out", "shifted", "blind")}, "n_params": n_params(m)}
        if name == "model":
            base = out[name]["acc"]["held_out"]
            out["knockouts"] = {}
            for mod, mode in (("dayan_decoder", "identity"), ("rod_accumulator", "zero"), ("stem_register", "identity"),
                              ("branch_register", "identity"), ("tally_readout", "identity")):
                out["knockouts"][mod] = accuracy(knockout(m, mod, mode), data["held_out"]) - base
            out["probe"] = rod_discreteness(m, data)
    return out


# ===== 12. Report and 13. entry point =====
def build_report(args, per_seed, correctness, gradres, mutants, runtime, rng, exit_code):
    report = {"schema_version": "1.0", "chapter": 328, "file": os.path.basename(__file__),
              "card_revision": MIND_CARD["card_revision"],
              "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
              "seeds": [p["seed"] for p in per_seed], "runtime_s": runtime, "n_params": per_seed[0]["model"]["n_params"],
              "gradcheck": {"tensors_checked": gradres["dayan"]["tensors"] + gradres["baseline"]["tensors"],
                            "tensors_total": gradres["dayan"]["tensors"] + gradres["baseline"]["tensors"],
                            "max_rel_error": max(v[k] for v in gradres.values() for k in ("init", "after")),
                            "checked_at": ["init", "after_training_steps"],
                            "passed": all(v["passed"] for v in gradres.values())},
              "correctness": correctness, "mutants": mutants, "hypotheses": [], "knockouts": [],
              "task_types": TASK_TYPES, "exit_code": exit_code, "extra_lines": []}
    quick = len(per_seed) < 2
    for h in MIND_CARD["hypotheses"]:
        if h["id"] == "H-SIG":
            diffs = [p["model"]["acc"]["shifted"] - p["baseline"]["acc"]["shifted"] for p in per_seed]
        elif h["id"] == "H-NEC":
            diffs = [p["knockouts"]["dayan_decoder"] - p["knockouts"]["rod_accumulator"] for p in per_seed]
        else:
            diffs = [(p["model"]["acc"]["blind"] - p["model"]["acc"]["held_out"]) -
                     (p["baseline"]["acc"]["blind"] - p["baseline"]["acc"]["held_out"]) for p in per_seed]
        if quick:
            mean, ci, v = float(np.mean(diffs)), (float("nan"), float("nan")), "not evaluated"
        else:
            mean, ci, v = verdict(diffs, h["direction"], h["mesi"], rng)
        report["hypotheses"].append({"id": h["id"], "metric": h["metric"], "mean_diff": mean, "ci95": list(ci),
                                     "mesi": h["mesi"], "n_seeds": len(per_seed), "verdict": v})
    for mod in per_seed[0]["knockouts"]:
        vals = [p["knockouts"][mod] for p in per_seed]
        mean, lo, hi = (float(np.mean(vals)), float("nan"), float("nan")) if quick else paired_bootstrap(np.array(vals), rng)
        report["knockouts"].append({"module": mod, "signature": modules_signature(mod), "metric_change": mean,
                                    "ci95": [lo, hi]})
    for sysname in ("model", "baseline"):
        acc = {k: float(np.mean([p[sysname]["acc"][k] for p in per_seed])) for k in ("held_out", "shifted", "blind")}
        e = {k: float(np.mean([p[sysname]["ece"][k] for p in per_seed])) for k in ("held_out", "shifted", "blind")}
        report["extra_lines"].append(f"{sysname:<9} params {per_seed[0][sysname]['n_params']:<4} acc held_out {acc['held_out']:.3f} "
                                     f"shifted {acc['shifted']:.3f} blind {acc['blind']:.3f} | ECE {e['held_out']:.3f}/"
                                     f"{e['shifted']:.3f}/{e['blind']:.3f}")
    sharp, right = (float(np.mean([p["probe"][i] for p in per_seed])) for i in (0, 1))
    report["extra_lines"].append(f"probe rod discreteness: {sharp:.2f} of symbol kernels peaked >= 0.9; "
                                 f"{right:.2f} peaked on the true increment (mod 定數)")
    return report

def modules_signature(name):
    return name in ("stem_register", "branch_register", "dayan_decoder")

def parse_args(argv):
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__), add_help=True)
    for flag, kw in (("--quick", {"action": "store_true"}), ("--seed", {"type": int, "default": 0}), ("--seeds", {"type": int, "default": 5}),
                     ("--json", {"type": str, "default": None}), ("--card", {"action": "store_true"}),
                     ("--mutant", {"type": str, "default": None}), ("--data", {"type": str, "default": None})):
        ap.add_argument(flag, **kw)
    try:
        return ap.parse_args(argv)
    except SystemExit:
        sys.exit(2)

def main(argv=None):
    global MUTANT
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.card:
        print(json.dumps(MIND_CARD, indent=1, ensure_ascii=False))
        return 0
    if args.mutant:
        if args.mutant not in MUTANTS:
            print(f"unknown mutant {args.mutant}; registered: {MUTANTS}", file=sys.stderr)
            return 2
        MUTANT = args.mutant
    if args.data:
        print(f"real-data bridge: --data {args.data} is not implemented for this chapter; skipping.")
    t0 = time.time()
    n_seeds = 1 if args.quick else max(1, args.seeds)
    steps = QUICK_STEPS if args.quick else FULL_STEPS
    budget = QUICK_TIME_BUDGET_S if args.quick else STEP_TIME_BUDGET_S
    rng = np.random.default_rng(args.seed + 999)
    sizes = {k: v // 3 for k, v in SPLIT_SIZES.items()} if args.quick else SPLIT_SIZES
    data = make_data(args.seed, sizes)
    print(f"seeds: {[args.seed + i for i in range(n_seeds)]}   steps/run: {steps}   mutant: {MUTANT}")
    trivial = trivial_accuracy(data)
    print("trivial (majority) accuracy:", {k: round(v, 3) for k, v in trivial.items()})
    correctness = []
    gradres = test_c1(data, rng, 60)
    print("C1 gradcheck:", {k: (f"init {v['init']:.1e}", f"after {v['after']:.1e}") for k, v in gradres.items()})
    c2ok, c2d = test_c2(data)
    correctness.append({"id": "C2", "name": "determinism_and_finiteness", "passed": bool(c2ok), "detail": c2d})
    per_seed = []
    for i in range(n_seeds):
        s = args.seed + i
        per_seed.append(run_seed(s, data if i == 0 else make_data(s, sizes), steps))
        m = per_seed[-1]["model"]
        print(f"seed {s}: drop {m['loss_drop']:.2f}  acc held_out {m['acc']['held_out']:.3f} shifted {m['acc']['shifted']:.3f} "
              f"blind {m['acc']['blind']:.3f}  | baseline held_out {per_seed[-1]['baseline']['acc']['held_out']:.3f} "
              f"shifted {per_seed[-1]['baseline']['acc']['shifted']:.3f} blind {per_seed[-1]['baseline']['acc']['blind']:.3f}"
              f"  ({time.time() - t0:.0f}s)")
    th = MIND_CARD["thresholds"]
    m0 = per_seed[0]["model"]
    c3 = m0["loss_drop"] >= th["loss_drop_fraction"] and m0["acc"]["held_out"] >= trivial["held_out"] + th["margin_over_trivial"]
    correctness.append({"id": "C3", "name": "learning", "passed": bool(c3),
                        "detail": f"loss drop {m0['loss_drop']:.2f} (>= {th['loss_drop_fraction']}), held-out {m0['acc']['held_out']:.3f} "
                                  f"vs trivial {trivial['held_out']:.3f} + {th['margin_over_trivial']}"})
    c4ok, c4acc, c4band = test_c4(data, rng, steps if args.quick else 400)
    correctness.append({"id": "C4", "name": "shuffled_label_control", "passed": bool(c4ok),
                        "detail": f"held-out {c4acc:.3f} with permuted targets; trivial {c4band:.3f} + band {th['shuffled_band']}"})
    det, tot, mdet = test_c5(data, QUICK_STEPS if args.quick else 300)
    mutants = {"detected": det, "total": tot, "score": det / tot}
    correctness.append({"id": "C5", "name": "mutant_detection", "passed": det == tot, "detail": mdet})
    for cid, (ok, detail) in test_c6(rng).items():
        correctness.append({"id": cid, "name": {"C6.1": "ring_simplex_conservation", "C6.2": "rotation_equivariance",
                                                "C6.3": "order_invariance", "C6.4": "qiuyi_multiplier",
                                                "C6.5": "dingshu_and_reassembly", "C6.6": "silent_inconsistency"}[cid],
                            "passed": bool(ok), "detail": detail})
    overlap = sum(len(data["train"]["hashes"] & data[k]["hashes"]) for k in SPLITS if k != "train")
    correctness.append({"id": "C7", "name": "split_integrity", "passed": overlap == 0, "detail": f"{overlap} shared sequences"})
    runtime = time.time() - t0
    correctness.append({"id": "C8", "name": "budget", "passed": runtime <= budget, "detail": f"{runtime:.1f}s of {budget:.0f}s"})
    correctness.insert(0, {"id": "C1", "name": "gradient_check", "passed": all(v["passed"] for v in gradres.values()),
                           "detail": "; ".join(f"{k} init {v['init']:.1e} after {v['after']:.1e}" for k, v in gradres.items())})
    all_ok = all(c["passed"] for c in correctness)
    exit_code = 0 if all_ok else (3 if not correctness[-1]["passed"] and sum(not c["passed"] for c in correctness) == 1 else 1)
    write_report(build_report(args, per_seed, correctness, gradres, mutants, runtime, rng, exit_code), args.json)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

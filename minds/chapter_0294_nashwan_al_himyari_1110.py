#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0294 · Nashwan al-Himyari
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 15, Minds 281-300 Available on Amazon https://www.amazon.com/dp/B0HMQ7TVJM
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0294_nashwan_al_himyari_1110 - Nashwan ibn Sa'id al-Himyari (d. 573/1178)
# END ATTRIBUTION
"""
HIRASA - the twin-guard lexicon decoder (chapter 0294, Nashwan ibn Sa'id al-Himyari (d. 573/1178)).

Thesis
    The arrangement of a store can carry a second, differently-corrupted copy of exactly the
    features that copying destroys, so that a damaged entry is repaired from where it stands:
    letter-order guards the dots, pattern headings guard the vowels, and no teacher is needed.

Evidence and provenance (archetype_provenance = belief)
    D1  Shams al-'ulum, author's preface (ed. al-'Amri, al-Iryani, 'Abd Allah, Dar al-Fikr 1999,
        vol. 1 p. 34): "fa-huruf al-mu'jam tahrusu al-naqt ... wa-l-amthila harisa li-l-harakat
        wa-l-shakl, wa-radda kull kalima min bina'iha ila al-asl" (letters guard the dots; the
        patterns guard the vowels and return each word from its structure to its root).
    D2  Same page: earlier lexicographers guarded either the dots (by letter order) or the vowels
        (by patterns and measures), never both "li-taba'udihima", because the two lie far apart.
    D3  Same page: the reader finds what he seeks quickly, without a shaykh to read with or an
        informant to depend on (integrity by construction rather than by chain of authority).
    D4  Same preface p. 33: God distinguished humans from the other animals by intellect and
        tongue (bi-l-lubb wa-l-lisan) - mind and language are named together.
    D5  Arrangement order (scholarship, editors' introduction 1999; comparison with Ishaq
        al-Farabi's Diwan al-adab): first radical, second radical, then noun/verb and pattern,
        and only then the final radical - the final letter is guarded by ordering alone.
    D6  Speculation (this chapter): one heading stands over many entries, so every entry is
        also a witness to its heading - guarding runs both ways.
    D7  Scholarship: the method found no followers (A. 'A. 'Attar, quoted in the 1999 introduction):
        it presupposes that every word has already been returned to its correct root and pattern.

Doctrine -> mechanism -> test
    D1,D2  M1 hirasa guard: each heading coordinate is bound to one factor only  -> C6.2, H-SIG
    D2     M1 under an equal budget, allocation to vowels vs. third radical      -> H-RIVAL
    D3     M1+M2 decoding needs no external label at test time                   -> H-SIG, H-GEN
    D5     weak (order-only) guard on the final radical                          -> reported per factor
    D6     M2 section consensus: permutation-invariant pooling of witnesses      -> C6.1, H-NEC
    D7     misfiled entries: the guard asserts the wrong form with authority     -> H-BLIND

Research question (memory, retrieval and forgetting)
    Can a store's addressing scheme hold an independent copy of the fragile features of its
    contents, so that repeated noisy copying is repaired from placement - and when does the
    guard become a liability?

Closest prior art and the delta
    Noisy-channel correction (Kernighan, Church & Gale 1990; Brill & Moore 2000), decoding with
    side information (Slepian & Wolf 1973; Wyner & Ziv 1976), error-correcting output codes
    (Dietterich & Bakiri 1995), permutation-invariant set pooling (Zaheer et al. 2017), Arabic
    diacritic restoration (Zitouni, Sorensen & Sarikaya 2006). Implemented baseline: a
    size-matched MLP given the same token, the same headings and the section mean.
    Delta: the side information is the store's own index, each index coordinate is tied to one
    fragile feature class, one heading is shared by (and witnessed by) all of its entries, and
    the file measures which features deserve the guard under a fixed budget.

Blind spot
    The guard trusts the filing. Where a word was returned to the wrong root or pattern (weak
    roots, loanwords, the Himyaritic forms Nashwan himself recorded) the heading asserts the
    wrong dots and vowels, and a mind that binds to its headings should suffer more than a
    looser learner (H-BLIND).

Task (synthetic scriptorium)
    14 letters in 6 skeleton classes (rasm); a letter = skeleton + dot code (i'jam).
    10 patterns (awzan) fix two vowels and noun/verb; a word = 3 radicals + 1 pattern.
    A copied token shows skeletons (rarely corrupted), dots (often corrupted), vowels (often
    dropped) and a weak noun/verb cue. Words are filed in sections under headings written
    in words (rarely corrupted); the scheme decides which factors a heading names (strong)
    and which are guarded only by order (weak). Target: the clean word, per factor.
    Splits: held-out words; shifted = older copies (more noise); misfiled = 25 % of entries
    sit under a heading that lies about one strong coordinate; generations = 6 rounds of
    copy-decode-recopy, headings rewritten by the copyist.

Limits
    Synthetic data; the file is a research prototype of an AGI-oriented memory component,
    not an AGI and not a model of the historical person's mind. The mutual-guarding pass (D6)
    is this chapter's extrapolation. Nothing here encodes genealogy or political claims.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 2,
    "revision_log": [
        "r1 -> r2 (before the first full-protocol run; hypotheses, metrics, mesi and thresholds unchanged): "
        "pilot runs fixed lr 0.01 and 700 steps for both minds (grid lr 0.01/0.03 x 700/2000 steps; 0.01 best for both); "
        "C4 now draws shuffled targets from the whole training pool, because a within-batch permutation leaked the "
        "section key; C1 denominator floor set to 1e-3 (loss ~13 at initialisation gives ~3e-9 round-off).",
    ],
    "generation": {"template_version": "codeguidelines 1.0 (2026-09-15)", "generator": "Claude",
                   "generator_version": "Claude Opus 5.5", "date": "2026-09-23"},
    "id": 294, "figure": "Nashwan al-Himyari", "born": 1110, "died": 1178,
    "civilization": "Arab (Yemeni)", "provenance": "belief",
    "thesis": ("A store's arrangement can carry a second, differently-corrupted copy of exactly the "
               "features copying destroys - letter order guards the dots, pattern headings guard the "
               "vowels - so a damaged entry is repaired from where it stands, without a teacher."),
    "evidence": [
        {"id": "D1", "claim": "Letter order guards the dots; patterns guard the vowels and return each word to its root",
         "basis": "primary", "source": "Shams al-'ulum, preface, ed. 1999, vol. 1 p. 34"},
        {"id": "D2", "claim": "Predecessors guarded dots or vowels, never both, because the two lie far apart",
         "basis": "primary", "source": "Shams al-'ulum, preface, ed. 1999, vol. 1 p. 34"},
        {"id": "D3", "claim": "The seeker finds his word without a shaykh or informant",
         "basis": "primary", "source": "Shams al-'ulum, preface, ed. 1999, vol. 1 p. 34"},
        {"id": "D4", "claim": "Humans are distinguished by intellect and tongue together",
         "basis": "primary", "source": "Shams al-'ulum, preface, ed. 1999, vol. 1 p. 33"},
        {"id": "D5", "claim": "Order: first and second radical, then noun/verb and pattern, final radical last",
         "basis": "scholarship", "source": "Editors' introduction to Shams al-'ulum (1999)"},
        {"id": "D6", "claim": "Entries under one heading are witnesses to that heading",
         "basis": "speculation", "source": "This chapter's reading of the arrangement"},
        {"id": "D7", "claim": "The method found no followers; lookup presupposes correct root analysis",
         "basis": "scholarship", "source": "A. 'A. 'Attar, quoted in the 1999 editors' introduction"},
    ],
    "research_question": {"category": "memory, retrieval and forgetting",
                          "question": ("Can a store's addressing scheme hold an independent copy of the fragile "
                                       "features of its contents so that noisy copying is repaired from "
                                       "placement, and when does that guard become a liability?")},
    "mechanism": {
        "name": "Hirasa: twin-guard lexicon decoder",
        "family": "structured side-information decoding with set pooling",
        "signature_modules": ["hirasa_guard", "section_consensus"],
        "closest_prior_art": ["noisy-channel spelling correction (Kernighan, Church & Gale 1990; Brill & Moore 2000)",
                              "decoding with side information (Slepian & Wolf 1973; Wyner & Ziv 1976)",
                              "error-correcting output codes (Dietterich & Bakiri 1995)",
                              "Deep Sets pooling (Zaheer et al. 2017)",
                              "Arabic diacritization (Zitouni, Sorensen & Sarikaya 2006)"],
        "overlap": "Medium",
        "prior_art_queries": ["index as error correcting code memory", "side information decoding learned",
                              "hierarchical address redundancy neural memory", "diacritic restoration root pattern"],
        "contribution_type": "mechanism",
        "delta": ("Redundancy lives in the store's own index; each index coordinate is bound to one fragile "
                  "feature class; one heading is shared by and witnessed by all its entries; guard allocation "
                  "under a fixed budget is itself tested."),
    },
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1 hirasa_guard", "property_test": "C6.2", "hypothesis": "H-SIG"},
        {"doctrine": "D2", "mechanism": "M1 allocation", "property_test": "C6.2", "hypothesis": "H-RIVAL"},
        {"doctrine": "D3", "mechanism": "M1+M2", "property_test": "C6.1", "hypothesis": "H-GEN"},
        {"doctrine": "D6", "mechanism": "M2 section_consensus", "property_test": "C6.1", "hypothesis": "H-NEC"},
        {"doctrine": "D7", "mechanism": "M1 trust in filing", "property_test": "C6.3", "hypothesis": "H-BLIND"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "The twin-guard decoder beats a size-matched MLP with identical inputs on older copies",
         "metric": "word_acc", "split": "shifted", "comparison": "model - baseline", "direction": "greater",
         "mesi": 0.02, "seeds": 5},
        {"id": "H-NEC", "statement": "Removing section consensus costs more than removing the noun/verb cue",
         "metric": "word_acc", "split": "shifted", "comparison": "signature_knockout - matched_knockout",
         "direction": "less", "mesi": 0.01, "seeds": 5},
        {"id": "H-BLIND", "statement": "On misfiled entries the twin-guard decoder falls further than the MLP",
         "condition": "25% of entries under a heading that lies about one strong coordinate",
         "grounding": "the arrangement presupposes correct root analysis; the method found no followers",
         "metric": "word_acc_misfiled", "comparison": "model - baseline", "direction": "less",
         "mesi": 0.02, "seeds": 5},
        {"id": "H-RIVAL", "statement": ("Under an equal budget, guarding radicals 1-2 and the pattern (Nashwan) beats "
                                        "guarding all three radicals with the pattern only ordered (letters-only)"),
         "metric": "word_acc", "split": "held", "comparison": "nashwan - letters3", "direction": "greater",
         "mesi": 0.01, "seeds": 5},
        {"id": "H-GEN", "statement": "After six copy-decode-recopy generations more words survive under the twin guard",
         "metric": "survival_g6", "split": "held", "comparison": "model - baseline", "direction": "greater",
         "mesi": 0.02, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.3, "margin_over_trivial": 0.25, "shuffled_band": 0.05},
    "probe_predictions": [
        {"probe": "P9", "expected": "above baseline"}, {"probe": "P8", "expected": "equal to baseline"},
        {"probe": "P3", "expected": "equal to baseline"}, {"probe": "P1", "expected": "below baseline"},
        {"probe": "P4", "expected": "equal to baseline"}, {"probe": "P6", "expected": "equal to baseline"},
    ],
    "dialectic_links": [
        {"chapter": 203, "relation": "rival", "test": "none (phonetic-permutation order not implemented)"},
        {"chapter": 229, "relation": "rival", "test": "none (initial-letter census order not implemented)"},
        {"chapter": 0, "relation": "rival", "test": "H-RIVAL descriptive: al-Jawhari (Sihah) last/first-letter arrangement"},
        {"chapter": 0, "relation": "rival", "test": "H-RIVAL descriptive: Ishaq al-Farabi (Diwan al-adab) pattern-first arrangement"},
    ],
    "corpus_neighbors": [
        {"chapter": 289, "similarity": None, "difference": "al-Idrisi fixes an address independent of content; here the address is a copy of the content's fragile features and repairs it"},
        {"chapter": 279, "similarity": None, "difference": "Halevi copies unopened what reason cannot re-derive; here copies are made self-repairing by placement"},
        {"chapter": 237, "similarity": None, "difference": "al-Hamdani parts evidence under a conservation law and a stake; no filing or channel code"},
        {"chapter": 203, "similarity": None, "difference": "al-Khalil enumerates the possibility space and registers absence; no guard against copying"},
        {"chapter": 507, "similarity": 0.0, "difference": "sample file (apostilla attenuation); only file available for the token-shingle gate"},
    ],
    "barometer": {
        "cognitive_processing": ["word accuracy on held words", "transfer across filing schemes"],
        "embodied_cognition": [], "world_modeling": ["survival under generational copying"],
        "consciousness": [], "language_understanding": ["root-and-pattern decoding of a synthetic script"],
        "emotional_intelligence": [], "creativity": [], "autonomy": ["repair without an external label"],
    },
    "task_types": ["sequence_classification", "vector_classification"],
    "applications": [
        {"use": "Diacritic restoration and OCR post-correction of Arabic manuscripts using the catalogue structure as side information",
         "sector": "cultural heritage / digital humanities", "dataset": "Tashkeela (Zerrouki & Balla 2017)"},
        {"use": "Post-OCR correction of historical print where running heads and indexes guard entry names",
         "sector": "libraries and archives", "dataset": "ICDAR 2019 Post-OCR Text Correction competition data"},
        {"use": "Repair of hierarchical product codes after manual transcription, with the category tree as guard",
         "sector": "retail logistics", "dataset": "Open Food Facts product database (GTIN codes)"},
    ],
    "safety_notes": ("Synthetic data only. No genealogical, tribal or political variables; the modern political "
                     "uses of Nashwan's name are history, not features. No claim to replicate a mind."),
}

import argparse
import json
import math
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------------------------
# BEGIN STANDARD UTILITIES v1.0
def logsumexp(z, axis=-1, keepdims=False):
    m = np.max(z, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(z - m), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


def log_softmax(z, axis=-1):
    return z - logsumexp(z, axis=axis, keepdims=True)


def softmax(z, axis=-1):
    return np.exp(log_softmax(z, axis=axis))


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
    total = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    if total > max_norm:
        for k in grads:
            grads[k] = grads[k] * (max_norm / (total + 1e-12))
    return total


def finite_difference_check(loss_fn, params, grads, rng, eps=1e-6, n_random=20, floor=1e-4):
    """Central differences on every tensor: the largest-gradient entry plus n_random random entries.
    rel = |analytic - numeric| / max(|analytic| + |numeric|, floor)."""
    worst, per_tensor = 0.0, {}
    for name, p in params.items():
        flat = p.reshape(-1)
        g = grads[name].reshape(-1)
        idx = {int(np.argmax(np.abs(g)))}
        if flat.size > 1:
            idx |= set(int(i) for i in rng.choice(flat.size, size=min(n_random, flat.size), replace=False))
        t_worst = 0.0
        for i in sorted(idx):
            old = flat[i]
            flat[i] = old + eps
            lp = loss_fn()
            flat[i] = old - eps
            lm = loss_fn()
            flat[i] = old
            num = (lp - lm) / (2 * eps)
            rel = abs(g[i] - num) / max(abs(g[i]) + abs(num), floor)
            t_worst = max(t_worst, rel)
        per_tensor[name] = t_worst
        worst = max(worst, t_worst)
    return worst, per_tensor


def paired_bootstrap(diffs, rng, n_resamples=2000, alpha=0.05):
    d = np.asarray(diffs, dtype=float)
    idx = rng.integers(0, len(d), size=(n_resamples, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), [float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))]


def verdict(mean, ci, mesi, direction):
    if direction == "greater":
        if ci[0] > 0 and mean >= mesi:
            return "supported"
        if ci[1] < 0:
            return "contradicted"
    else:
        if ci[1] < 0 and mean <= -mesi:
            return "supported"
        if ci[0] > 0:
            return "contradicted"
    return "inconclusive"


def write_report(report, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
# END STANDARD UTILITIES
# ---------------------------------------------------------------------------------------------


class NonFinite(Exception):
    pass


# ---------------------------------------------------------------------------------------------
# Data: a synthetic root-and-pattern script and its scribes
# ---------------------------------------------------------------------------------------------
GROUP_SIZES = [3, 3, 2, 2, 2, 2]                       # skeleton classes (rasm shapes)
LETTER_NAMES = ["b", "t", "th", "j", "h", "kh", "d", "dh", "r", "z", "s", "sh", "'", "gh"]
SKEL = np.repeat(np.arange(len(GROUP_SIZES)), GROUP_SIZES)
DOTC = np.concatenate([np.arange(s) for s in GROUP_SIZES])
N_LET, N_SKEL, N_DOTOBS = len(SKEL), len(GROUP_SIZES), 4   # dot field: codes 0..2 or 3 = unreadable
# patterns: (vowel after R1, vowel after R2, noun=0 / verb=1); vowels 0=a 1=i 2=u 3=sukun
PATTERNS = [(0, 3, 0, "fa'l"), (1, 3, 0, "fi'l"), (2, 3, 0, "fu'l"), (0, 0, 0, "fa'al"),
            (2, 2, 0, "fu'ul"), (1, 0, 0, "fi'al"), (0, 0, 1, "fa'ala"), (0, 1, 1, "fa'ila"),
            (0, 2, 1, "fa'ula"), (2, 1, 1, "fu'ila")]
PAT_V = np.array([[p[0], p[1]] for p in PATTERNS])
PAT_POS = np.array([p[2] for p in PATTERNS])
N_PAT = len(PATTERNS)
KS = [N_LET, N_LET, N_LET, N_PAT]                       # factors: R1, R2, R3, pattern
FACTOR_NAMES = ["R1", "R2", "R3", "pattern"]
DX = 3 * (N_SKEL + N_DOTOBS) + 2 * 5 + 2
POS_CUE_COLS = list(range(DX - 2, DX))
N_MAX = 6                                              # entries per section chunk
LEXICON_SEED = 294

# Which factor each filing scheme names in its headings ("strong") or guards by order only ("weak").
SCHEMES = {
    "nashwan":  {0: "strong", 1: "strong", 3: "strong", 2: "weak"},
    "letters3": {0: "strong", 1: "strong", 2: "strong", 3: "weak"},
    "jawhari":  {2: "strong", 0: "strong", 1: "weak"},
    "farabi":   {3: "strong", 2: "strong", 0: "weak"},
}
NOISE = {
    "train":   dict(skel=0.03, dot=0.30, vow=0.55, cue=0.75, strong=0.06, weak=0.40, misfile=0.0),
    "shifted": dict(skel=0.05, dot=0.45, vow=0.75, cue=0.75, strong=0.15, weak=0.50, misfile=0.0),
    "blind":   dict(skel=0.03, dot=0.30, vow=0.55, cue=0.75, strong=0.06, weak=0.40, misfile=0.25),
}


def make_lexicon(density=0.7, train_frac=0.65):
    """Attested words (R1, R2, R3, pattern) with R1 != R2, split by word into train and held."""
    rng = np.random.default_rng(LEXICON_SEED)
    grid = np.array(np.meshgrid(np.arange(N_LET), np.arange(N_LET), np.arange(N_LET),
                                np.arange(N_PAT), indexing="ij")).reshape(4, -1).T
    grid = grid[grid[:, 0] != grid[:, 1]]
    words = grid[rng.random(len(grid)) < density]
    perm = rng.permutation(len(words))
    cut = int(train_frac * len(words))
    return words, np.sort(perm[:cut]), np.sort(perm[cut:])


def roles_of(scheme):
    return [SCHEMES[scheme].get(f) for f in range(4)]


def make_sections(words, idx, scheme, rng):
    """Group word indices by the scheme's strong key; chunk into sections of at most N_MAX."""
    strong = [f for f, r in enumerate(roles_of(scheme)) if r == "strong"]
    keys = words[idx][:, strong]
    order = np.lexsort(keys.T[::-1])
    idx, keys = idx[order], keys[order]
    breaks = np.flatnonzero(np.any(np.diff(keys, axis=0) != 0, axis=1)) + 1
    secs = []
    for grp in np.split(idx, breaks):
        grp = rng.permutation(grp)
        for s in range(0, len(grp), N_MAX):
            row = -np.ones(N_MAX, dtype=int)
            chunk = grp[s:s + N_MAX]
            row[:len(chunk)] = chunk
            secs.append(row)
    return np.array(secs)


def render_tokens(W, nz, rng):
    """What a copyist's page shows for words W (N,4): skeletons, dots, vowels, noun/verb cue."""
    n = len(W)
    X = np.zeros((n, DX))
    rows = np.arange(n)
    for k in range(3):
        off = k * (N_SKEL + N_DOTOBS)
        sk = np.where(rng.random(n) < nz["skel"], rng.integers(0, N_SKEL, n), SKEL[W[:, k]])
        dc = np.where(rng.random(n) < nz["dot"], rng.integers(0, N_DOTOBS, n), DOTC[W[:, k]])
        X[rows, off + sk] = 1.0
        X[rows, off + N_SKEL + dc] = 1.0
    base = 3 * (N_SKEL + N_DOTOBS)
    for j in range(2):
        v = PAT_V[W[:, 3], j]
        hit = rng.random(n) < nz["vow"]
        wrong = np.where(rng.random(n) < 0.7, 4, rng.integers(0, 4, n))   # mostly dropped, sometimes miswritten
        X[rows, base + 5 * j + np.where(hit, wrong, v)] = 1.0
    pos = PAT_POS[W[:, 3]]
    X[rows, DX - 2 + np.where(rng.random(n) < nz["cue"], pos, 1 - pos)] = 1.0
    return X


def corrupt_symbol(true, K, p, rng):
    return np.where(rng.random(true.shape) < p, rng.integers(0, K, true.shape), true)


def assemble(Wsec, heads, scheme, nz, rng):
    """Build a batch from section word arrays Wsec (B,n,4; -1 rows padded) and heading values
    heads (B,4 for strong factors; weak factors are read per entry from Wsec)."""
    B, n = Wsec.shape[:2]
    M = (Wsec[:, :, 0] >= 0).astype(float)
    Wc = np.where(Wsec >= 0, Wsec, 0)
    X = render_tokens(Wc.reshape(-1, 4), nz, rng).reshape(B, n, DX) * M[:, :, None]
    A = []
    for f, role in enumerate(roles_of(scheme)):
        if role == "strong":
            A.append(np.eye(KS[f])[corrupt_symbol(heads[:, f], KS[f], nz["strong"], rng)])
        elif role == "weak":
            A.append(np.eye(KS[f])[corrupt_symbol(Wc[:, :, f], KS[f], nz["weak"], rng)] * M[:, :, None])
        else:
            A.append(None)
    return {"X": X, "M": M, "Y": Wc, "A": A}


def section_batch(words, secs, scheme, nz, rng):
    """Sections of lexicon words; with nz['misfile'] > 0 some entries are words that belong under
    a heading differing in one strong coordinate (the page lies about them)."""
    Wsec = np.where(secs[:, :, None] >= 0, words[np.maximum(secs, 0)], -1)
    heads = Wsec[:, 0, :].copy()
    mis = np.zeros(secs.shape, dtype=bool)
    if nz["misfile"] > 0:
        strong = [f for f, r in enumerate(roles_of(scheme)) if r == "strong"]
        mis = (rng.random(secs.shape) < nz["misfile"]) & (secs >= 0)
        f_pick = np.array(strong)[rng.integers(0, len(strong), secs.shape)]
        for f in strong:
            sel = mis & (f_pick == f)
            shift = rng.integers(1, KS[f], sel.sum())
            Wsec[sel, f] = (Wsec[sel, f] + shift) % KS[f]
        bad = mis & (Wsec[:, :, 0] == Wsec[:, :, 1])          # keep R1 != R2
        Wsec[bad, 1] = (Wsec[bad, 1] + 1) % N_LET
    batch = assemble(Wsec, heads, scheme, nz, rng)
    batch["misfiled"] = mis
    return batch


def trivial_baselines(words, tr_idx, ev_words):
    maj = [np.bincount(words[tr_idx][:, f], minlength=KS[f]).argmax() for f in range(4)]
    fac = [float(np.mean(ev_words[:, f] == maj[f])) for f in range(4)]
    word = float(np.mean(np.all(ev_words == np.array(maj), axis=1)))
    return {"factor_acc": fac, "mean_factor_acc": float(np.mean(fac)), "word_acc": word}


# ---------------------------------------------------------------------------------------------
# Model: Hirasa (twin guard + section consensus)
# ---------------------------------------------------------------------------------------------
TASK_TYPES = ["sequence_classification", "vector_classification"]
MUTANT = None          # set by --mutant or by the in-process mutation audit
MUTANTS = ["sign_flipped_update", "zero_learning_rate", "zero_grad_U", "zero_grad_G0"]


def build_model(in_dim, out_dim, task_type, rng, kind="hirasa", scheme="nashwan", hidden=40, **cfg):
    """Native use: out_dim is ignored and the four factors (R1, R2, R3, pattern) are decoded.
    Probe use (scheme=None): one factor of out_dim classes; a sequence is read as a section and
    its label is the section posterior (the heading reconstructed from its witnesses)."""
    factors = list(KS) if scheme else [out_dim]
    roles = roles_of(scheme) if scheme else ["strong"]
    P = {}
    if kind == "hirasa":
        P["U"] = rng.normal(0, 1 / math.sqrt(in_dim), (in_dim, hidden))
        P["c"] = np.zeros(hidden)
        for f, K in enumerate(factors):
            P[f"V{f}"] = rng.normal(0, 1 / math.sqrt(hidden), (hidden, K))
            P[f"b{f}"] = np.zeros(K)
            P[f"G{f}"] = rng.normal(0, 0.01, (K, K))
        P["beta"] = np.full(len(factors), 0.5)
        P["gamma"] = np.ones(len(factors))
    else:
        din = 2 * in_dim + (sum(factors) if scheme else 0)
        P["W1"] = rng.normal(0, 1 / math.sqrt(din), (din, hidden))
        P["b1"] = np.zeros(hidden)
        for f, K in enumerate(factors):
            P[f"W2_{f}"] = rng.normal(0, 1 / math.sqrt(hidden), (hidden, K))
            P[f"c2_{f}"] = np.zeros(K)
    return {"kind": kind, "scheme": scheme, "roles": roles, "factors": factors,
            "task_type": task_type, "params": P}


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


def _encode(model, X):
    P = model["params"]
    return np.tanh(X @ P["U"] + P["c"])


def _hirasa_forward(model, batch, variant=None):
    P, roles = model["params"], model["roles"]
    X, M, A = batch["X"], batch["M"], batch["A"]
    pre = X @ P["U"] + P["c"]
    if variant == "leaky_guard":            # negative control for C6.2: the R1 heading reaches the encoder
        pre = pre + 0.5 * (A[0] @ np.arange(A[0].shape[-1], dtype=float))[:, None, None] / A[0].shape[-1]
    h = np.tanh(pre)
    Z, caches = [], []
    for f, role in enumerate(roles):
        e = h @ P[f"V{f}"] + P[f"b{f}"]
        fc = {"e": e, "role": role}
        if role == "strong":
            l = log_softmax(e)
            w = M[:, :, None]
            if variant == "ordered_pool":      # negative control for C6.1: witnesses weighted by position
                w = w * (1.0 + np.arange(M.shape[1]))[None, :, None]
            pooled = (w * l).sum(1)
            s = P["beta"][f] * pooled
            if A[f] is not None:
                s = s + A[f] @ P[f"G{f}"]
            q = log_softmax(s)[:, None, :]
            fc.update(w=w, pooled=pooled, s=s)
        elif role == "weak":
            s = A[f] @ P[f"G{f}"]
            q = log_softmax(s)
            fc["s"] = s
        else:
            q = np.zeros_like(e)
        fc["q"] = q
        Z.append(e + P["gamma"][f] * q)
        caches.append(fc)
    return Z, {"h": h, "f": caches}


def _heading_per_entry(model, batch):
    M, cols = batch["M"], []
    for f, role in enumerate(model["roles"]):
        a = batch["A"][f]
        if role == "strong" and a is not None:
            cols.append(np.repeat(a[:, None, :], M.shape[1], axis=1) * M[:, :, None])
        elif role == "weak":
            cols.append(a)
        elif model["scheme"]:
            cols.append(np.zeros(M.shape + (KS[f],)))
    return cols


def _mlp_input(model, batch):
    X, M = batch["X"], batch["M"]
    mean = (X * M[:, :, None]).sum(1) / np.maximum(M.sum(1), 1)[:, None]
    parts = [X] + _heading_per_entry(model, batch) + [np.repeat(mean[:, None, :], X.shape[1], 1) * M[:, :, None]]
    return np.concatenate(parts, axis=-1)


def _mlp_forward(model, batch):
    P = model["params"]
    inp = _mlp_input(model, batch)
    h = np.tanh(inp @ P["W1"] + P["b1"])
    Z = [h @ P[f"W2_{f}"] + P[f"c2_{f}"] for f in range(len(model["factors"]))]
    return Z, {"h": h, "inp": inp}


def forward(model, batch, variant=None):
    if model["kind"] == "hirasa":
        return _hirasa_forward(model, batch, variant)
    return _mlp_forward(model, batch)


def _ce(Z, Y, M):
    tot = max(float(M.sum()), 1.0)
    loss, dZ = 0.0, []
    for f, z in enumerate(Z):
        ls = log_softmax(z)
        y = Y[:, :, f][..., None]
        loss -= float((np.take_along_axis(ls, y, -1)[..., 0] * M).sum()) / tot
        d = np.exp(ls)
        np.put_along_axis(d, y, np.take_along_axis(d, y, -1) - 1.0, -1)
        dZ.append(d * M[:, :, None] / tot)
    return loss, dZ


def _hirasa_backward(model, batch, cache, dZ):
    P = model["params"]
    X, A, h = batch["X"], batch["A"], cache["h"]
    H = h.shape[-1]
    g = {k: np.zeros_like(v) for k, v in P.items()}
    dh = np.zeros_like(h)
    for f, (dz, fc) in enumerate(zip(dZ, cache["f"])):
        de = dz.copy()
        g["gamma"][f] = float((dz * fc["q"]).sum())
        dq = P["gamma"][f] * dz
        K = dz.shape[-1]
        if fc["role"] == "strong":
            dqs = dq.sum(1)
            ds = dqs - softmax(fc["s"]) * dqs.sum(-1, keepdims=True)
            if A[f] is not None:
                g[f"G{f}"] = A[f].T @ ds
            g["beta"][f] = float((ds * fc["pooled"]).sum())
            dl = fc["w"] * (P["beta"][f] * ds)[:, None, :]
            de += dl - softmax(fc["e"]) * dl.sum(-1, keepdims=True)
        elif fc["role"] == "weak":
            ds = dq - softmax(fc["s"]) * dq.sum(-1, keepdims=True)
            g[f"G{f}"] = A[f].reshape(-1, K).T @ ds.reshape(-1, K)
        g[f"V{f}"] = h.reshape(-1, H).T @ de.reshape(-1, K)
        g[f"b{f}"] = de.sum((0, 1))
        dh += de @ P[f"V{f}"].T
    dpre = dh * (1 - h * h)
    g["U"] = X.reshape(-1, X.shape[-1]).T @ dpre.reshape(-1, H)
    g["c"] = dpre.sum((0, 1))
    return g


def _mlp_backward(model, cache, dZ):
    P, h, inp = model["params"], cache["h"], cache["inp"]
    H = h.shape[-1]
    g = {}
    dh = np.zeros_like(h)
    for f, dz in enumerate(dZ):
        K = dz.shape[-1]
        g[f"W2_{f}"] = h.reshape(-1, H).T @ dz.reshape(-1, K)
        g[f"c2_{f}"] = dz.sum((0, 1))
        dh += dz @ P[f"W2_{f}"].T
    dpre = dh * (1 - h * h)
    g["W1"] = inp.reshape(-1, inp.shape[-1]).T @ dpre.reshape(-1, H)
    g["b1"] = dpre.sum((0, 1))
    return g


def loss_and_grads(model, batch):
    Z, cache = forward(model, batch)
    loss, dZ = _ce(Z, batch["Y"], batch["M"])
    if model["kind"] == "hirasa":
        g = _hirasa_backward(model, batch, cache, dZ)
    else:
        g = _mlp_backward(model, cache, dZ)
    if MUTANT == "zero_grad_U" and "U" in g:
        g["U"] = np.zeros_like(g["U"])
    if MUTANT == "zero_grad_G0" and "G0" in g:
        g["G0"] = np.zeros_like(g["G0"])
    return loss, g


def predict_entries(model, batch):
    Z, _ = forward(model, batch)
    return np.stack([z.argmax(-1) for z in Z], axis=-1)


def predict(model, X):
    """Probe interface: X (N,D) -> one class per vector; X (N,T,D) -> one class per sequence."""
    X3 = X[:, None, :] if X.ndim == 2 else X
    batch = {"X": X3, "M": np.ones(X3.shape[:2]), "A": [None] * len(model["factors"])}
    Z, cache = forward(model, batch)
    if model["kind"] == "hirasa" and X.ndim == 3:
        return cache["f"][0]["s"].argmax(-1)
    return Z[0][:, 0, :].argmax(-1) if X.ndim == 2 else Z[0].mean(1).argmax(-1)


def hidden_states(model, X):
    X3 = X[:, None, :] if X.ndim == 2 else X
    if model["kind"] == "hirasa":
        return _encode(model, X3)
    batch = {"X": X3, "M": np.ones(X3.shape[:2]), "A": [None] * len(model["factors"])}
    return _mlp_forward(model, batch)[1]["h"]


def section_heads(model, batch, pred):
    """The copyist rewrites each heading: from the section posterior (hirasa) or, lacking one,
    from the plurality of what it read in the section (baseline)."""
    B = pred.shape[0]
    heads = np.zeros((B, 4), dtype=int)
    Z, cache = forward(model, batch)
    for f, role in enumerate(model["roles"]):
        if role != "strong":
            continue
        if model["kind"] == "hirasa":
            heads[:, f] = cache["f"][f]["s"].argmax(-1)
        else:
            counts = np.zeros((B, KS[f]))
            np.add.at(counts, (np.repeat(np.arange(B), pred.shape[1]), pred[:, :, f].ravel()),
                      batch["M"].ravel())
            heads[:, f] = counts.argmax(-1)
    return heads


# ---------------------------------------------------------------------------------------------
# Registries: modules, knockouts
# ---------------------------------------------------------------------------------------------
def modules(model):
    if model["kind"] == "mlp":
        return {"mlp_hidden": {"params": ["W1", "b1"], "role": "tanh hidden layer over token, headings and section mean", "signature": False},
                "mlp_heads": {"params": [k for k in model["params"] if k.startswith(("W2_", "c2_"))], "role": "linear softmax heads", "signature": False}}
    F = len(model["factors"])
    return {
        "token_encoder": {"params": ["U", "c"], "role": "shared tanh encoder of the copied token", "signature": False},
        "factor_heads": {"params": [f"V{f}" for f in range(F)] + [f"b{f}" for f in range(F)], "role": "per-factor linear logits from the token", "signature": False},
        "pos_cue": {"params": ["U"], "role": "encoder rows reading the noun/verb context cue", "signature": False},
        "hirasa_guard": {"params": [f"G{f}" for f in range(F)] + ["gamma"], "role": "factor-bound heading expert added to one factor's logits only", "signature": True},
        "section_consensus": {"params": ["beta"], "role": "permutation-invariant sum of witnesses' log-likelihoods into the shared heading posterior", "signature": True},
    }


def knockout(model, name, mode="zero"):
    """Copy of model with one module replaced (zero, identity or mean output)."""
    m = {k: v for k, v in model.items()}
    P = {k: v.copy() for k, v in model["params"].items()}
    m["params"] = P
    if name == "hirasa_guard":
        P["gamma"][:] = 0.0
    elif name == "section_consensus":
        P["beta"][:] = 0.0
    elif name == "pos_cue":
        if mode == "mean":
            P["c"] += 0.5 * P["U"][POS_CUE_COLS].sum(0)
        P["U"][POS_CUE_COLS] = 0.0
    elif name == "token_encoder":
        P["U"][:] = 0.0
        P["c"][:] = 0.0
    elif name == "factor_heads":
        for k in P:
            if k[0] in "Vb" and k[1:].isdigit():
                P[k][:] = 0.0
    else:
        raise KeyError(name)
    return m


# ---------------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------------
def fit(model, sampler, budget, rng, lr=0.01, clip=5.0):
    opt = Adam(model["params"], lr=0.0 if MUTANT == "zero_learning_rate" else lr)
    sign = -1.0 if MUTANT == "sign_flipped_update" else 1.0
    losses = []
    for _ in range(budget):
        loss, g = loss_and_grads(model, sampler(rng))
        if not np.isfinite(loss):
            raise NonFinite("loss")
        clip_global_norm(g, clip)
        opt.step(model["params"], g, sign)
        losses.append(loss)
    for v in model["params"].values():
        if not np.all(np.isfinite(v)):
            raise NonFinite("params")
    return losses


def make_sampler(words, secs, scheme, nz, batch_sections=24, shuffle_labels=False):
    """Fresh copying noise at every step. With shuffle_labels the targets are drawn from the whole
    training pool, independently of the page: a within-batch permutation would leak, because
    entries of one section share their key."""
    pool = words[secs[secs >= 0]]

    def sampler(rng):
        pick = secs[rng.integers(0, len(secs), batch_sections)]
        b = section_batch(words, pick, scheme, nz, rng)
        if shuffle_labels:
            flat = b["Y"].reshape(-1, 4)
            valid = np.flatnonzero(b["M"].ravel() > 0)
            flat[valid] = pool[rng.integers(0, len(pool), len(valid))]
        return b
    return sampler


def evaluate(model, batch):
    pred = predict_entries(model, batch)
    M = batch["M"] > 0
    ok = np.all(pred == batch["Y"], axis=-1)
    res = {"word_acc": float(ok[M].mean()),
           "factor_acc": [float((pred[..., f] == batch["Y"][..., f])[M].mean()) for f in range(4)]}
    res["mean_factor_acc"] = float(np.mean(res["factor_acc"]))
    if batch.get("misfiled") is not None and batch["misfiled"].any():
        res["word_acc_misfiled"] = float(ok[batch["misfiled"]].mean())
    return res


def generational_survival(model, words, secs, scheme, generations, rng):
    """Copy, read, recopy: each generation renders what the previous copyist read, under the
    heading that copyist wrote; returns the share of entries still equal to the original."""
    Wtrue = np.where(secs[:, :, None] >= 0, words[np.maximum(secs, 0)], -1)
    Wcur, heads = Wtrue.copy(), Wtrue[:, 0, :].copy()
    curve = []
    for _ in range(generations):
        batch = assemble(Wcur, heads, scheme, NOISE["train"], rng)
        pred = predict_entries(model, batch)
        heads = np.where(np.array([r == "strong" for r in roles_of(scheme)])[None, :],
                         section_heads(model, batch, pred), heads)
        Wcur = np.where(batch["M"][:, :, None] > 0, pred, -1)
        valid = Wtrue[:, :, 0] >= 0
        curve.append(float(np.all(Wcur == Wtrue, axis=-1)[valid].mean()))
    return curve


# ---------------------------------------------------------------------------------------------
# Tests: correctness first (C1-C8), then hypotheses
# ---------------------------------------------------------------------------------------------
HIDDEN = {"hirasa": 40, "mlp": 24}      # 4548 vs 4588 parameters: within 10 %
GC_TOL = 1e-5


def new_model(kind, scheme, rng):
    return build_model(DX, None, "sequence_classification", rng, kind=kind, scheme=scheme, hidden=HIDDEN[kind])


def c1_gradcheck(words, secs_tr, rng, kinds=("hirasa", "mlp"), after=True):
    rows, worst, checked, total = [], 0.0, 0, 0
    for kind in kinds:
        model = new_model(kind, "nashwan", rng)
        gb = make_sampler(words, secs_tr, "nashwan", NOISE["train"], batch_sections=3)(rng)
        stages = ["init", "after_training_steps"] if after else ["init"]
        for stage in stages:
            if stage != "init":
                fit(model, make_sampler(words, secs_tr, "nashwan", NOISE["train"]), 60, rng)
            _, g = loss_and_grads(model, gb)
            # floor 1e-3: at initialisation the loss is ~13, so float64 central differences carry
            # ~3e-9 absolute round-off; tiny gradients are judged on that absolute scale
            w, per = finite_difference_check(lambda: _ce(forward(model, gb)[0], gb["Y"], gb["M"])[0],
                                             model["params"], g, rng, floor=1e-3)
            worst = max(worst, w)
            checked += sum(1 for v in per.values() if v <= GC_TOL)
            total += len(per)
            rows.append((kind, stage, w))
    return {"passed": worst <= GC_TOL, "max_rel_error": worst, "tensors_checked": checked,
            "tensors_total": total, "rows": rows}


def c2_determinism(words, secs_tr, seed):
    runs = []
    for _ in range(2):
        rng = np.random.default_rng(seed)
        m = new_model("hirasa", "nashwan", rng)
        runs.append((fit(m, make_sampler(words, secs_tr, "nashwan", NOISE["train"]), 30, rng), m))
    same = runs[0][0] == runs[1][0] and all(np.array_equal(runs[0][1]["params"][k], runs[1][1]["params"][k])
                                             for k in runs[0][1]["params"])
    b = make_sampler(words, secs_tr, "nashwan", NOISE["shifted"])(np.random.default_rng(seed + 1))
    Z, cache = forward(runs[0][1], b)
    finite = all(np.all(np.isfinite(z)) for z in Z) and np.all(np.isfinite(cache["h"]))
    rng = np.random.default_rng(seed + 2)                       # probe-interface smoke test (section 8)
    probe_ok = True
    for kind in ("hirasa", "mlp"):
        pm = build_model(DX, 5, "sequence_classification", rng, kind=kind, scheme=None, hidden=8)
        for Xp in (rng.normal(size=(7, DX)), rng.normal(size=(7, 4, DX))):
            yp, hs = predict(pm, Xp), hidden_states(pm, Xp)
            probe_ok &= yp.shape == (7,) and hs.shape[:2] == (7, 1 if Xp.ndim == 2 else 4) and np.all(np.isfinite(hs))
    ok = same and finite and probe_ok
    return {"passed": bool(ok), "detail": f"identical={same}, finite={bool(finite)}, probe interface={bool(probe_ok)}"}


def c3_learning(losses, held_res, trivial):
    first, last = float(np.mean(losses[:20])), float(np.mean(losses[-50:]))
    drop = 1 - last / first
    margin = held_res["mean_factor_acc"] - trivial["mean_factor_acc"]
    th = MIND_CARD["thresholds"]
    ok = drop >= th["loss_drop_fraction"] and margin >= th["margin_over_trivial"]
    return {"passed": bool(ok), "drop": drop, "margin": margin,
            "detail": f"loss {first:.3f}->{last:.3f} (drop {drop:.2f}); mean factor acc margin {margin:+.3f}"}


def c4_shuffled(words, secs_tr, held_batch, trivial, steps, rng):
    m = new_model("hirasa", "nashwan", rng)
    fit(m, make_sampler(words, secs_tr, "nashwan", NOISE["train"], shuffle_labels=True), steps, rng)
    acc = evaluate(m, held_batch)["mean_factor_acc"]
    band = trivial["mean_factor_acc"] + MIND_CARD["thresholds"]["shuffled_band"]
    return {"passed": bool(acc <= band), "detail": f"shuffled-label model {acc:.3f} vs trivial band <= {band:.3f}"}


def c5_mutants(words, secs_tr, held_batch, trivial, seed):
    global MUTANT
    detected = []
    for name in MUTANTS:
        MUTANT = name
        rng = np.random.default_rng(seed + 7)
        gc = c1_gradcheck(words, secs_tr, rng, kinds=("hirasa",), after=False)
        m = new_model("hirasa", "nashwan", rng)
        losses = fit(m, make_sampler(words, secs_tr, "nashwan", NOISE["train"]), 150, rng)
        c3 = c3_learning(losses, evaluate(m, held_batch), trivial)
        MUTANT = None
        detected.append((name, (not gc["passed"]) or (not c3["passed"])))
    score = sum(d for _, d in detected) / len(detected)
    return {"passed": score == 1.0, "score": score, "detected": sum(d for _, d in detected),
            "total": len(detected), "detail": ", ".join(f"{n}:{'caught' if d else 'MISSED'}" for n, d in detected)}


def _random_batch(rng, B=5, n=N_MAX):
    W = np.stack([rng.integers(0, N_LET, (B, n)), rng.integers(0, N_LET, (B, n)),
                  rng.integers(0, N_LET, (B, n)), rng.integers(0, N_PAT, (B, n))], -1)
    W[:, -1, :] = -1                                       # one padded slot
    return assemble(W, W[:, 0, :], "nashwan", NOISE["shifted"], rng)


def _scaled_model(rng, scale):
    m = new_model("hirasa", "nashwan", rng)
    for k, v in m["params"].items():
        m["params"][k] = v + rng.normal(0, scale, v.shape)
    return m


def c6_properties(rng, trials=25):
    """C6.1 permutation equivariance of section consensus; C6.2 guard locality; each searched over
    random parameters and inputs, each with a negative control that must violate it.
    C6.3 is a labelled definition check (guard knocked out => headings irrelevant)."""
    v1 = v1_neg = v2 = v2_neg = v3 = 0.0
    for t in range(trials):
        m = _scaled_model(rng, scale=0.5 + 2.0 * t / trials)
        b = _random_batch(rng)
        perm = rng.permutation(N_MAX - 1).tolist() + [N_MAX - 1]
        bp = dict(b, X=b["X"][:, perm], M=b["M"][:, perm],
                  A=[a if (a is None or a.ndim == 2) else a[:, perm] for a in b["A"]])
        for variant in (None, "ordered_pool"):
            Z, _ = forward(m, b, variant)
            Zp, _ = forward(m, bp, variant)
            d = max(float(np.abs(z[:, perm] - zp).max()) for z, zp in zip(Z, Zp))
            v1, v1_neg = (max(v1, d), v1_neg) if variant is None else (v1, max(v1_neg, d))
        b2 = dict(b, A=list(b["A"]))
        b2["A"][0] = np.eye(KS[0])[rng.integers(0, KS[0], b["X"].shape[0])]
        b2["A"][2] = np.eye(KS[2])[rng.integers(0, KS[2], b["X"].shape[:2])] * b["M"][:, :, None]
        for variant in (None, "leaky_guard"):
            Z, _ = forward(m, b, variant)
            Z2, _ = forward(m, b2, variant)
            d = float(np.abs(Z[1] - Z2[1]).max() + np.abs(Z[3] - Z2[3]).max())
            v2, v2_neg = (max(v2, d), v2_neg) if variant is None else (v2, max(v2_neg, d))
        k = knockout(m, "hirasa_guard")
        v3 = max(v3, max(float(np.abs(z - z2).max()) for z, z2 in zip(forward(k, b)[0], forward(k, b2)[0])))
    return [
        {"id": "C6.1", "name": "consensus_permutation_equivariance", "passed": v1 < 1e-10 and v1_neg > 1e-6,
         "detail": f"max violation {v1:.1e}; ordered-pool control {v1_neg:.2e} (must be > 1e-6)"},
        {"id": "C6.2", "name": "guard_locality", "passed": v2 < 1e-12 and v2_neg > 1e-6,
         "detail": f"R2/pattern change when R1 and R3 headings change {v2:.1e}; leaky control {v2_neg:.2e}"},
        {"id": "C6.3", "name": "definition_check_guard_knockout (not evidence)", "passed": v3 < 1e-12,
         "detail": f"with the guard removed, heading change moves outputs by {v3:.1e}"},
    ]


def c7_split(tr, ho, words):
    a = set(map(tuple, words[tr]))
    b = set(map(tuple, words[ho]))
    ok = len(np.intersect1d(tr, ho)) == 0 and len(a & b) == 0
    return {"passed": bool(ok), "detail": f"{len(tr)} train words, {len(ho)} held words, overlap {len(a & b)}"}


def run_seed(seed, words, tr, ho, steps, full):
    """Train every mind on one seed; evaluate all on identical noisy pages."""
    kids = np.random.SeedSequence(seed).spawn(12)
    rng_sec = np.random.default_rng(kids[0])
    secs = {s: (make_sections(words, tr, s, rng_sec), make_sections(words, ho, s, rng_sec)) for s in SCHEMES}
    plan = [("nashwan", "hirasa", "nashwan"), ("mlp", "mlp", "nashwan")]
    if full:
        plan += [("letters3", "hirasa", "letters3"), ("jawhari", "hirasa", "jawhari"), ("farabi", "hirasa", "farabi")]
    models, losses = {}, {}
    for i, (name, kind, scheme) in enumerate(plan):
        m = new_model(kind, scheme, np.random.default_rng(kids[1 + i]))
        losses[name] = fit(m, make_sampler(words, secs[scheme][0], scheme, NOISE["train"]), steps,
                           np.random.default_rng(kids[6]))
        models[name] = m
    ev = lambda cond, scheme: section_batch(words, secs[scheme][1], scheme, NOISE[cond], np.random.default_rng(kids[8]))
    held, shifted, blind = ev("train", "nashwan"), ev("shifted", "nashwan"), ev("blind", "nashwan")
    out = {"secs": secs, "models": models, "losses": losses, "held_batch": held}
    for name in ("nashwan", "mlp"):
        out[name] = {"held": evaluate(models[name], held), "shifted": evaluate(models[name], shifted),
                     "blind": evaluate(models[name], blind)}
    if full:
        for name in ("letters3", "jawhari", "farabi"):
            out[name] = {"held": evaluate(models[name], ev("train", name))}
        base = out["nashwan"]["shifted"]["word_acc"]
        modes = {"section_consensus": "zero", "hirasa_guard": "identity", "pos_cue": "mean",
                 "token_encoder": "zero", "factor_heads": "zero"}
        out["knockouts"] = {mod: evaluate(knockout(models["nashwan"], mod, md), shifted)["word_acc"] - base
                            for mod, md in modes.items()}
        sub = secs["nashwan"][1][:600]
        out["gen"] = {n: generational_survival(models[n], words, sub, "nashwan", 6, np.random.default_rng(kids[9]))
                      for n in ("nashwan", "mlp")}
    return out


HYP_DIFFS = {
    "H-SIG": lambda r: r["nashwan"]["shifted"]["word_acc"] - r["mlp"]["shifted"]["word_acc"],
    "H-NEC": lambda r: r["knockouts"]["section_consensus"] - r["knockouts"]["pos_cue"],
    "H-BLIND": lambda r: r["nashwan"]["blind"]["word_acc_misfiled"] - r["mlp"]["blind"]["word_acc_misfiled"],
    "H-RIVAL": lambda r: r["nashwan"]["held"]["word_acc"] - r["letters3"]["held"]["word_acc"],
    "H-GEN": lambda r: r["gen"]["nashwan"][-1] - r["gen"]["mlp"][-1],
}


def hypothesis_rows(results, rng, evaluated):
    rows = []
    for h in MIND_CARD["hypotheses"]:
        if not evaluated:
            rows.append({"id": h["id"], "metric": h["metric"], "mean_diff": None, "ci95": None,
                         "mesi": h["mesi"], "n_seeds": len(results), "verdict": "not evaluated"})
            continue
        mean, ci = paired_bootstrap([HYP_DIFFS[h["id"]](r) for r in results], rng)
        rows.append({"id": h["id"], "metric": h["metric"], "mean_diff": mean, "ci95": ci, "mesi": h["mesi"],
                     "n_seeds": len(results), "verdict": verdict(mean, ci, h["mesi"], h["direction"])})
    return rows


def knockout_rows(results, rng):
    rows = []
    sig = modules(results[0]["models"]["nashwan"])
    for mod in results[0]["knockouts"]:
        mean, ci = paired_bootstrap([r["knockouts"][mod] for r in results], rng)
        rows.append({"module": mod, "signature": sig[mod]["signature"], "metric_change": mean, "ci95": ci})
    return rows


def load_real_lexicon(path):
    """Optional bridge: CSV with header R1,R2,R3,pattern holding integer codes in range."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        head = fh.readline().strip().split(",")
        if head != ["R1", "R2", "R3", "pattern"]:
            raise ValueError("header must be R1,R2,R3,pattern")
        for line in fh:
            if line.strip():
                rows.append([int(x) for x in line.strip().split(",")])
    words = np.array(rows, dtype=int)
    if words.ndim != 2 or np.any(words < 0) or np.any(words >= np.array(KS)):
        raise ValueError("codes out of range")
    words = words[words[:, 0] != words[:, 1]]
    perm = np.random.default_rng(LEXICON_SEED).permutation(len(words))
    cut = int(0.65 * len(words))
    return words, np.sort(perm[:cut]), np.sort(perm[cut:])


def fmt(x, nd=3):
    return "-" if x is None else f"{x:+.{nd}f}" if isinstance(x, float) else str(x)


def print_report(rep, results, full):
    L = [f"=== VERIFIED REPORT · chapter 0294 ===",
         f"file            {rep['file']}  (card revision {rep['card_revision']})",
         f"environment     python {rep['environment']['python']}, numpy {rep['environment']['numpy']}",
         f"seeds           {rep['seeds']}   runtime {rep['runtime_s']:.1f} s",
         f"parameters      hirasa {rep['n_params']['hirasa']}, size-matched MLP {rep['n_params']['mlp']}",
         f"gradient check  {rep['gradcheck']['tensors_checked']}/{rep['gradcheck']['tensors_total']} tensor checks "
         f"(init and after 60 steps), max rel err {rep['gradcheck']['max_rel_error']:.2e}",
         "correctness"]
    L += [f"  {c['id']:<5} {c['name']:<44} {'PASS' if c['passed'] else 'FAIL'}  {c['detail']}" for c in rep["correctness"]]
    L.append(f"mutation score  {rep['mutants']['detected']}/{rep['mutants']['total']} = {rep['mutants']['score']:.2f}")
    r0 = results[0]
    L.append("accuracy (seed 0)           held word  shifted word  misfiled word   R1    R2    R3   pattern (shifted)")
    for n in ("nashwan", "mlp"):
        d = r0[n]
        L.append(f"  {n:<24} {d['held']['word_acc']:>9.3f} {d['shifted']['word_acc']:>12.3f} "
                 f"{d['blind'].get('word_acc_misfiled', float('nan')):>13.3f}   " +
                 " ".join(f"{a:.3f}" for a in d["shifted"]["factor_acc"]))
    if full:
        L.append("filing schemes, held word accuracy (mean over seeds)")
        for s in ("nashwan", "letters3", "jawhari", "farabi"):
            L.append(f"  {s:<10} {np.mean([r[s]['held']['word_acc'] for r in results]):.3f}")
        g = {n: np.mean([r["gen"][n] for r in results], axis=0) for n in ("nashwan", "mlp")}
        L.append("survival over 6 copy generations  hirasa " + " ".join(f"{x:.3f}" for x in g["nashwan"]) +
                 " | MLP " + " ".join(f"{x:.3f}" for x in g["mlp"]))
        L.append("knockouts (shifted word acc change, 95% CI)")
        L += [f"  {k['module']:<18} sig={str(k['signature']):<5} {k['metric_change']:+.3f} "
              f"[{k['ci95'][0]:+.3f}, {k['ci95'][1]:+.3f}]" for k in rep["knockouts"]]
    L.append("hypotheses (paired per-seed differences, 2000-resample bootstrap)")
    for h in rep["hypotheses"]:
        ci = "-" if h["ci95"] is None else f"[{h['ci95'][0]:+.3f}, {h['ci95'][1]:+.3f}]"
        L.append(f"  {h['id']:<8} {h['metric']:<18} mean {fmt(h['mean_diff'])}  CI {ci}  mesi {h['mesi']}  "
                 f"n={h['n_seeds']}  {h['verdict']}")
    L.append(f"task types      {', '.join(rep['task_types'])}")
    L.append(f"exit code       {rep['exit_code']}")
    L.append("=== END REPORT ===")
    print("\n".join(L))


def run_protocol(args):
    t0 = time.time()
    quick = args.quick
    n_seeds = 1 if quick else args.seeds
    seeds = [args.seed + i for i in range(n_seeds)]
    steps = 250 if quick else 700
    if args.data:
        words, tr, ho = load_real_lexicon(args.data)
        print(f"real-data bridge: loaded {len(words)} coded words from {args.data}")
    else:
        words, tr, ho = make_lexicon()
        print("real-data bridge: no --data PATH given; synthetic scriptorium used")
    print(f"seeds {seeds}; steps {steps}; mutant {MUTANT}")
    results = [run_seed(s, words, tr, ho, steps, full=not quick) for s in seeds]
    r0 = results[0]
    secs_tr, held = r0["secs"]["nashwan"][0], r0["held_batch"]
    trivial = trivial_baselines(words, tr, held["Y"][held["M"] > 0])
    rng = np.random.default_rng(args.seed + 100)
    gc = c1_gradcheck(words, secs_tr, rng)
    c3 = c3_learning(r0["losses"]["nashwan"], r0["nashwan"]["held"], trivial)
    corr = [{"id": "C1", "name": "gradient_check", "passed": gc["passed"],
             "detail": f"max rel err {gc['max_rel_error']:.2e} over {gc['tensors_total']} tensor checks"},
            dict(id="C2", name="determinism_and_finiteness", **c2_determinism(words, secs_tr, args.seed)),
            {"id": "C3", "name": "learning", "passed": c3["passed"], "detail": c3["detail"] +
             f"; trivial mean factor acc {trivial['mean_factor_acc']:.3f}"},
            dict(id="C4", name="shuffled_label_control", **c4_shuffled(words, secs_tr, held, trivial, steps, rng))]
    mut = c5_mutants(words, secs_tr, held, trivial, args.seed) if MUTANT is None else \
        {"passed": True, "score": float("nan"), "detected": 0, "total": 0, "detail": "skipped under --mutant"}
    corr.append({"id": "C5", "name": "mutant_detection", "passed": mut["passed"], "detail": mut["detail"]})
    corr += c6_properties(rng)
    corr.append(dict(id="C7", name="split_integrity", **c7_split(tr, ho, words)))
    runtime = time.time() - t0
    budget = 20.0 if quick else 180.0
    corr.append({"id": "C8", "name": "budget", "passed": runtime <= budget,
                 "detail": f"{runtime:.1f} s of {budget:.0f} s"})
    full = not quick
    for c in corr:
        c["passed"] = bool(c["passed"])
    rep = {"schema_version": "1.0", "chapter": 294, "file": os.path.basename(__file__),
           "card_revision": MIND_CARD["card_revision"],
           "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
           "seeds": seeds, "runtime_s": runtime,
           "n_params": {"hirasa": n_params(r0["models"]["nashwan"]), "mlp": n_params(r0["models"]["mlp"])},
           "gradcheck": {"tensors_checked": gc["tensors_checked"], "tensors_total": gc["tensors_total"],
                         "max_rel_error": float(gc["max_rel_error"]), "checked_at": ["init", "after_training_steps"],
                         "passed": bool(gc["passed"])},
           "correctness": corr,
           "mutants": {"detected": mut["detected"], "total": mut["total"], "score": mut["score"]},
           "hypotheses": hypothesis_rows(results, rng, evaluated=full and MUTANT is None),
           "knockouts": knockout_rows(results, rng) if full else [],
           "task_types": TASK_TYPES}
    code = 0 if all(c["passed"] for c in corr[:-1]) else 1
    if code == 0 and not corr[-1]["passed"]:
        code = 3
    rep["exit_code"] = code
    print_report(rep, results, full)
    if args.json:
        write_report(rep, args.json)
    return code


def main(argv=None):
    global MUTANT
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__),
                                 description="Chapter 0294 Hirasa decoder: full protocol by default.")
    ap.add_argument("--quick", action="store_true", help="one seed, reduced steps, all correctness tests")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--json", metavar="PATH")
    ap.add_argument("--card", action="store_true", help="print MIND_CARD as JSON and exit")
    ap.add_argument("--mutant", choices=MUTANTS)
    ap.add_argument("--data", metavar="PATH", help="CSV with header R1,R2,R3,pattern (integer codes)")
    args = ap.parse_args(argv)
    if args.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    if args.seeds < 1:
        print(f"usage: {os.path.basename(__file__)} --seeds must be >= 1")
        return 2
    MUTANT = args.mutant
    try:
        return run_protocol(args)
    except NonFinite as err:
        print(f"non-finite values: {err}")
        return 4


if __name__ == "__main__":
    sys.exit(main())

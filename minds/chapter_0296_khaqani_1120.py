#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0296 · Khaqani (c. 1120-1199)  
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 15, Minds 281-300 Available on Amazon https://www.amazon.com/dp/B0HMQ7TVJM
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0296_khaqani_1120 - Khaqani (c. 1120-1199)
# END ATTRIBUTION
"""Held-field reader with declared renewal (tanasub / tajdid-e matla) (chapter 0296, Khaqani (c. 1120-1199)).

Thesis
    A reading is held, not recomputed: the semantic field opened at a declared
    beginning governs every later line, content alone can neither open nor close
    it, and only a formal new beginning renews it.

Evidence and provenance (archetype_provenance = belief; ids as in MIND_CARD)
    D1 Tanasub (congruence of images) sustained over whole passages, where earlier
       poets kept it inside single lines (Beelaert, EIr 2010, "Khaqani Shervani ii";
       Divan ed. Sajjadi pp. 133-40, qasida "Rokhsar-e sobh ...").
    D2 More than a quarter of the qasidas are strophic through tajdid-e matla:
       a new opening hemistich pair (tasri') renews the poem (Reinert 1964-65;
       Beelaert 2010).
    D3 Iltizam and radif: one commitment runs through every line, e.g. "sobh" in
       every line of Divan pp. 45-49 (Beelaert 2010).
    D4 Precision: the exact, often technical word instead of the common one; his
       rhetoric has been described with set theory (Reinert 1972).
    D5 "Difficult", "late to befriend" (Dashti 1961); sometimes "unbalanced"
       (Foruzanfar) -- grounds the blind spot.
    D6 A whole qasida held inside Christian vocabulary, "Falak kazhrowtar ast az
       khatt-e tarsa" (Divan pp. 23-28; Minorsky 1945).
    D7 "Falsafa dar sokhan mayamizid": do not mix philosophy into speech
       (anti-falsafa qasida, Divan) -- foreign matter is not admitted by content.
    D8 Line-bounded tanasub of earlier poets (Beelaert 2010) -- the rival manner.
    All claims are about poetic method; no treatise on mind survives. The step from
    method to cognitive mechanism is this file's hypothesis (basis: speculation).

Doctrine -> mechanism -> test
    D1, D6 -> M1 field_register   h_t=(1-r_t)h_{t-1}+a_t*l_t   C6.1, H-NEC, H-SIG
    D2     -> M3 declared_renewal r_t=1-exp(-form_t.softplus(g)) C6.1, C6.4, H-SIG
    D3, D7 -> M4 matla_admission  a_t=exp(-lam*tau_t)         C6.2, H-SIG, H-BLIND
    D4     -> M2 tanasub_scorer   l_tf=sum_j log sigmoid(e_j.A_f+a_f)   C6.3
    D8     -> rival line_local (register zeroed during training)        H-RIVAL
    D5     -> blind-spot split (undeclared field change)                H-BLIND

Research question
    Instruction authentication and prompt injection: can a reader whose task
    context is renewed only by an authenticated formal marker ignore unmarked
    in-band intrusions of any length, and what does it lose when the context really
    changes without a marker?

Closest prior art and the delta
    GRU (Cho et al. 2014; the size-matched baseline); HM-RNN boundary detectors
    (Chung, Ahn & Bengio 2017); Bayesian online changepoint detection (Adams &
    MacKay 2007); StruQ, instruction hierarchy and spotlighting for LLM injection
    defence (Chen et al. 2024; Wallace et al. 2024; Hines et al. 2024); lexical
    chains and TextTiling (Morris & Hirst 1991; Hearst 1997).
    Delta: the reset gate reads form features only (never content), memory is a
    conjunctive log-evidence register over fields, and admission of new evidence
    decays with distance from the declared opening, so unmarked content can bend
    the reading only by a bounded, geometrically shrinking amount.

Blind spot
    A silent change of field (no new matla) is followed late or never; the model
    should lose to the content-driven GRU exactly there (H-BLIND).

Task (synthetic "qasida" world, regenerated per seed)
    6 semantic fields x 4 themes. 72 content words, each with 3 senses in distinct
    fields; a word's theme in field f is its base theme plus a field offset (mod 4),
    so meaning depends on the field; 4 filler words. A poem is a run of passages; each
    passage opens with a tasri' marker line of 3 words congruent with (f, theme) and
    holds field f; later lines carry 1-2 congruent words plus fillers in 3 slots;
    label = (f, theme).
    One form bit is nuisance (p=0.3). Intrusion: k unmarked lines whose words fit
    only another field g (unscored), then 2 trap lines of words fitting both f and
    g, labelled f. Undeclared change: from some line on the field silently becomes
    g, with traps labelled g. Splits: train/val/test (k=1, every change declared),
    shifted (longer poems, k=2-4), blind (a silent change in every passage, a
    condition neither model meets in training).

Limits
    Synthetic symbols, not Persian text; one-layer readers; ~1.3k parameters; the
    marker is given, not detected. A research prototype, not an AGI and not a
    replica of the poet.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 3,
    "revision_log": ["r2, after single-seed pilot training runs and before any hypothesis evaluation: "
                     "(i) gradient-check denominator floor 1e-6 -> 1e-4 (finite-difference round-off ~1e-10 "
                     "dominated entries with |g| < 1e-5); (ii) word themes made field-offset, removing a leak "
                     "by which a line's own theme agreement identified its field; (iii) non-opening lines carry "
                     "1-2 words so the opening is needed; (iv) renewal gate made exactly zero without a form "
                     "signal (pilots learned a uniform leak that emulates content-driven switching); (v) silent "
                     "change removed from train/val/test and kept only in the blind split (pilots with 15% "
                     "silent change in training put the blind-spot condition into training). Hypotheses, "
                     "metrics, splits compared and mesi values unchanged.",
                     "r3, after the first full run: corpus_neighbors extended with #409 and #155 found by a corpus scan; "
                     "nothing else changed; the full protocol was re-run so the report carries this revision."],
    "generation": {"template_version": "codeguidelines 1.0", "generator": "Claude",
                   "generator_version": "Claude Opus 5.5", "date": "2026-09-24"},
    "id": 296, "figure": "Khaqani", "born": 1120, "died": 1199,
    "civilization": "Persian (Shirvanshah)",
    "provenance": "belief",
    "thesis": ("A reading is held, not recomputed: the field opened at a declared beginning governs "
               "every later line; content can neither open nor close it; only a formal new "
               "beginning (tajdid-e matla) renews it."),
    "evidence": [
        {"id": "D1", "claim": "Tanasub sustained over whole passages, not single lines",
         "basis": "scholarship", "source": "Beelaert, EIr XV/5 (2010) 'Khaqani Shervani ii. Works'; Divan ed. Sajjadi pp. 133-40"},
        {"id": "D2", "claim": "Over a quarter of the qasidas renewed by tasri' (tajdid-e matla)",
         "basis": "scholarship", "source": "Reinert, MDAT 12 (1964-65) 126-49; Beelaert 2010"},
        {"id": "D3", "claim": "Iltizam: one word ('sobh') kept in every line of a qasida",
         "basis": "primary", "source": "Divan ed. Sajjadi pp. 45-49"},
        {"id": "D4", "claim": "Precise technical word over the common one; rhetoric described with set theory",
         "basis": "scholarship", "source": "Reinert, Haqani als Dichter (1972); Beelaert 2010"},
        {"id": "D5", "claim": "Judged difficult and late to befriend; sometimes unbalanced",
         "basis": "scholarship", "source": "Dashti, Khaqani, sha'er-e dir-ashna (1961); Foruzanfar, Sokhan va sokhanvaran"},
        {"id": "D6", "claim": "A whole qasida held inside Christian vocabulary",
         "basis": "primary", "source": "Divan ed. Sajjadi pp. 23-28; Minorsky, BSOAS 11.3 (1945)"},
        {"id": "D7", "claim": "Do not mix philosophy into speech (anti-falsafa qasida)",
         "basis": "primary", "source": "Divan ed. Sajjadi, qasida 'Chashm bar parda-ye amal manehid'"},
        {"id": "D8", "claim": "Earlier poets kept tanasub within a single line",
         "basis": "scholarship", "source": "Beelaert 2010"},
        {"id": "D9", "claim": "Method-to-mechanism step (held register, form-only renewal)",
         "basis": "speculation", "source": "this file"},
    ],
    "research_question": {"category": "instruction authentication and prompt injection",
                          "question": ("Can a context renewed only by an authenticated formal marker ignore unmarked "
                                       "in-band intrusions of any length, and what does it lose under silent change?")},
    "mechanism": {
        "name": "held-field register with declared renewal",
        "family": "gated recurrent evidence accumulator (state-space style, form-gated reset)",
        "signature_modules": ["field_register", "declared_renewal", "matla_admission", "tanasub_scorer"],
        "closest_prior_art": ["GRU (Cho et al. 2014)", "HM-RNN boundary gates (Chung et al. 2017)",
                              "BOCPD (Adams & MacKay 2007)", "StruQ (Chen et al. 2024)",
                              "Instruction hierarchy (Wallace et al. 2024)", "Spotlighting (Hines et al. 2024)",
                              "Lexical chains / TextTiling (Morris & Hirst 1991; Hearst 1997)"],
        "overlap": "Medium",
        "prior_art_queries": ["form-gated reset recurrent network", "content-blind segment boundary RNN",
                              "prompt injection delimiter structured query", "evidence accumulator log-sigmoid conjunctive",
                              "admission decay since boundary memory"],
        "contribution_type": "mechanism",
        "delta": ("Reset reads form only; memory is a conjunctive log-evidence register whose admission of new "
                  "content decays with distance from the declared opening, bounding unmarked influence."),
    },
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1 field_register", "property_test": "C6.1", "hypothesis": "H-NEC"},
        {"doctrine": "D6", "mechanism": "M1 field_register", "property_test": "C6.1", "hypothesis": "H-SIG"},
        {"doctrine": "D2", "mechanism": "M3 declared_renewal", "property_test": "C6.4", "hypothesis": "H-SIG"},
        {"doctrine": "D3", "mechanism": "M4 matla_admission", "property_test": "C6.2", "hypothesis": "H-SIG"},
        {"doctrine": "D7", "mechanism": "M4 matla_admission", "property_test": "C6.2", "hypothesis": "H-SIG"},
        {"doctrine": "D4", "mechanism": "M2 tanasub_scorer", "property_test": "C6.3", "hypothesis": "H-RIVAL"},
        {"doctrine": "D8", "mechanism": "rival line_local", "property_test": "C6.3", "hypothesis": "H-RIVAL"},
        {"doctrine": "D5", "mechanism": "M3+M4 (no content-driven reset)", "property_test": "C6.4", "hypothesis": "H-BLIND"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "Held field beats a size-matched GRU where unmarked intrusions are longer than in training",
         "metric": "line accuracy", "split": "shifted", "comparison": "model - baseline",
         "direction": "greater", "mesi": 0.03, "seeds": 5},
        {"id": "H-NEC", "statement": "Removing the field register hurts more than removing the surface theme path",
         "metric": "line accuracy (test)", "comparison": "signature_knockout - matched_knockout",
         "direction": "less", "mesi": 0.05, "seeds": 5},
        {"id": "H-BLIND", "statement": "Under silent (unmarked) field change the held field loses to the GRU",
         "condition": "blind split: undeclared change in every passage",
         "grounding": "difficulty and imbalance charged by Dashti and Foruzanfar; change admitted only via new matla",
         "metric": "line accuracy", "comparison": "model - baseline", "direction": "less", "mesi": 0.05, "seeds": 5},
        {"id": "H-RIVAL", "statement": "Passage-level tanasub beats line-local tanasub (earlier manner)",
         "metric": "line accuracy (test)", "comparison": "model - rival", "direction": "greater",
         "mesi": 0.05, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.3, "margin_over_trivial": 0.25, "shuffled_band": 0.05,
                   "gradcheck_rel_error": 1e-5, "gradcheck_denominator_floor": 1e-4, "clip_norm": 5.0},
    "probe_predictions": [
        {"probe": "P1", "expected": "above baseline (unmarked distractors are not admitted)"},
        {"probe": "P4", "expected": "above baseline (register does not depend on passage length)"},
        {"probe": "P7", "expected": "below baseline (silent regime change is the blind spot)"},
        {"probe": "P8", "expected": "below baseline (confident while holding a stale field)"},
        {"probe": "P9", "expected": "above baseline for content noise, below for form-marker noise"},
        {"probe": "P3", "expected": "equal to baseline"},
    ],
    "dialectic_links": [
        {"chapter": 281, "relation": "teacher", "test": "none (declared model; the 'Badil' couplet is tazkira-transmitted)"},
        {"chapter": 253, "relation": "teacher", "test": "none (Saqt al-zand named in the Tohfat)"},
        {"chapter": 306, "relation": "successor", "test": "none"},
        {"chapter": 330, "relation": "rival", "test": "H-BLIND (Dashti's contrast of the difficult and the easy master)"},
        {"chapter": 234, "relation": "rival", "test": "H-RIVAL (earlier line-bounded manner; rebuilt, not copied)"},
    ],
    "corpus_neighbors": [
        {"chapter": 367, "similarity": None, "difference": "Hafez keeps two readings unresolved; here one field is held and resolved"},
        {"chapter": 234, "similarity": None, "difference": "Rudaki re-addresses a minimal key each line; here the key is set once and only renewed by form"},
        {"chapter": 201, "similarity": None, "difference": "Du Fu's surprise is content-driven regime change; here content can never trigger change"},
        {"chapter": 306, "similarity": None, "difference": "Nizami polices repetition; here repetition of the field is the norm"},
        {"chapter": 330, "similarity": None, "difference": "Sa'di moves evaluation to the hearer; here the opening fixes the hearer's frame"},
        {"chapter": 281, "similarity": None, "difference": "Sanai's declined bond is about payment; here the bond is between opening and line"},
        {"chapter": 409, "similarity": None, "difference": "Hang Tuah carries an unamendable foreign objective; here the held field is renewable, but only by declared form"},
        {"chapter": 155, "similarity": None, "difference": "Patrick authenticates a directive's source; here nothing is source-checked, only when context may change"},
    ],
    "barometer": {
        "cognitive_processing": ["working memory under distractors (shifted split, P1)"],
        "embodied_cognition": ["not addressed"],
        "world_modeling": ["tracking under declared vs silent regime change (blind split, P7)"],
        "consciousness": ["register exposes which field it holds (hidden_states); no qualia claimed"],
        "language_understanding": ["polysemy resolved by held semantic field (test split)"],
        "emotional_intelligence": ["not addressed"],
        "creativity": ["not addressed"],
        "autonomy": ["refuses to be re-tasked by unauthenticated content (shifted split)"],
    },
    "task_types": ["sequence_classification"],
    "applications": [
        {"use": "held-instruction register for LLM agents renewed only by authenticated delimiters",
         "sector": "AI security", "dataset": "BIPIA (Yi et al. 2023, arXiv 2312.14197)"},
        {"use": "section-topic tracking anchored on headings in long records",
         "sector": "document processing", "dataset": "WikiSection (Arnold et al. 2019, TACL)"},
        {"use": "semantic-field tracking across strophic qasidas",
         "sector": "digital humanities", "dataset": "Ganjoor public Persian poetry corpus (annotated CSV via --data)"},
    ],
    "safety_notes": ("Synthetic data only. The design is deliberately blind to unmarked change: deploy only with the "
                     "renewal marker authenticated outside the content channel and with a separate change monitor. "
                     "No sentence here is presented as the poet's own; no claim of replicating a person."),
}

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

N_FIELDS, N_THEMES, N_CONTENT, N_FILLER, SLOTS = 6, 4, 72, 4, 3
VOCAB = N_CONTENT + N_FILLER
N_CLASSES = N_FIELDS * N_THEMES
LR_GRID = (0.01, 0.03)
BATCH, CLIP = 64, 5.0
REGIMES = {
    "train": dict(T=12, plen=(3, 5), p_intr=0.35, k=(1, 1), p_change=0.0),
    "val": dict(T=12, plen=(3, 5), p_intr=0.35, k=(1, 1), p_change=0.0),
    "test": dict(T=12, plen=(3, 5), p_intr=0.35, k=(1, 1), p_change=0.0),
    "shifted": dict(T=30, plen=(6, 10), p_intr=0.7, k=(2, 4), p_change=0.0),
    "blind": dict(T=16, plen=(5, 7), p_intr=0.0, k=(1, 1), p_change=1.0),
}
SIZES = {"full": dict(train=2000, val=300, test=600, shifted=400, blind=500, steps=700),
         "quick": dict(train=800, val=150, test=300, shifted=150, blind=200, steps=350)}


class NonFinite(Exception):
    """Raised when a loss, gradient or activation stops being finite (exit code 4)."""


# BEGIN STANDARD UTILITIES v1.0
def softplus(x):
    return np.logaddexp(0.0, x)


def sigmoid(x):
    return np.exp(-np.logaddexp(0.0, -x))


def logsumexp(x, axis=-1, keepdims=True):
    m = np.max(x, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


def log_softmax(x, axis=-1):
    return x - logsumexp(x, axis=axis)


def softmax(x, axis=-1):
    return np.exp(log_softmax(x, axis=axis))


def adam_init(params):
    return {"t": 0, "m": {k: np.zeros_like(v) for k, v in params.items()},
            "v": {k: np.zeros_like(v) for k, v in params.items()}}


def adam_step(params, grads, state, lr, b1=0.9, b2=0.999, eps=1e-8):
    state["t"] += 1
    t = state["t"]
    for k in params:
        state["m"][k] = b1 * state["m"][k] + (1 - b1) * grads[k]
        state["v"][k] = b2 * state["v"][k] + (1 - b2) * grads[k] ** 2
        mh = state["m"][k] / (1 - b1 ** t)
        vh = state["v"][k] / (1 - b2 ** t)
        params[k] -= lr * mh / (np.sqrt(vh) + eps)


def clip_global_norm(grads, max_norm):
    norm = float(np.sqrt(sum(np.sum(g * g) for g in grads.values())))
    scale = min(1.0, max_norm / (norm + 1e-12))
    return {k: g * scale for k, g in grads.items()}, norm


def gradcheck(loss_fn, params, grads, rng, n_random=20, eps=1e-6, floor=1e-4):
    """Central differences on every tensor: all entries of small tensors, else n_random
    random entries plus the largest-gradient entry. rel = |a-n| / max(|a|+|n|, floor);
    the floor keeps entries with near-zero gradient from reporting round-off as error."""
    out = {}
    for name, p in params.items():
        g = grads[name]
        if p.size <= n_random:
            idx = np.arange(p.size)
        else:
            idx = np.unique(np.append(rng.choice(p.size, n_random, replace=False), np.argmax(np.abs(g))))
        worst = 0.0
        for i in idx:
            old = p.flat[i]
            p.flat[i] = old + eps
            lp = loss_fn()
            p.flat[i] = old - eps
            lm = loss_fn()
            p.flat[i] = old
            num = (lp - lm) / (2 * eps)
            worst = max(worst, abs(num - g.flat[i]) / max(abs(num) + abs(g.flat[i]), floor))
        out[name] = worst
    return out


def paired_bootstrap(diffs, rng, n_boot=2000, level=0.95):
    d = np.asarray(diffs, dtype=float)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return float(d.mean()), [float(lo), float(hi)]


def verdict(mean, ci, mesi, direction):
    sign = 1.0 if direction == "greater" else -1.0
    if sign * ci[0] > 0 and sign * ci[1] > 0 and sign * mean >= mesi:
        return "supported"
    if sign * ci[0] < 0 and sign * ci[1] < 0:
        return "contradicted"
    return "inconclusive"


def print_report(rep, sections):
    print("=== VERIFIED REPORT · chapter %04d ===" % rep["chapter"])
    for title, rows in sections:
        print(title)
        for row in rows:
            print("  " + row)
    print("=== END REPORT ===")


def write_json(path, rep):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2, ensure_ascii=False)
# END STANDARD UTILITIES


# ---------------------------------------------------------------- data and tasks
def make_world(rng):
    """Sense table: senses[j, f] = theme of word j in field f, or -1. A word's theme in
    field f is its base theme shifted by a per-field offset, so the theme depends on the
    field (polysemy) while a line that fits two fields reads consistently in both: only
    the held field, never the line's own words, can decide between them."""
    shift = rng.integers(0, N_THEMES, N_FIELDS)
    while True:
        senses = -np.ones((N_CONTENT, N_FIELDS), dtype=int)
        base = rng.integers(0, N_THEMES, N_CONTENT)
        for j in range(N_CONTENT):
            fields = rng.choice(N_FIELDS, 3, replace=False)
            senses[j, fields] = (base[j] + shift[fields]) % N_THEMES
        counts = np.array([[np.sum(senses[:, f] == y) for y in range(N_THEMES)] for f in range(N_FIELDS)])
        if counts.min() >= 4:
            return {"senses": senses, "cache": {}}


def _line(rng, pool, n):
    """Draw n distinct words from pool, pad with fillers, shuffle slot order."""
    words = list(rng.choice(pool, size=min(n, len(pool)), replace=False))
    while len(words) < SLOTS:
        words.append(N_CONTENT + int(rng.integers(0, N_FILLER)))
    return np.array(words)[rng.permutation(SLOTS)]


def _pick(rng, world, f, other=None, exclude=None, n_min=2):
    """Theme and word pool for field f, optionally requiring a sense in `other`
    (trap words fitting both fields) or forbidding a sense in `exclude`; pools are
    cached per world because the sense table never changes."""
    key = (f, other, exclude)
    if key not in world["cache"]:
        sen, found = world["senses"], []
        for y in range(N_THEMES):
            ok = sen[:, f] == y
            if other is not None:
                ok &= sen[:, other] >= 0
            if exclude is not None:
                ok &= sen[:, exclude] < 0
            if ok.sum() >= n_min:
                found.append((y, np.flatnonzero(ok)))
        world["cache"][key] = found
    found = world["cache"][key]
    return found[int(rng.integers(len(found)))] if found else None


def _passage(rng, world, reg, L, f):
    """Lines of one passage: (words, label, scored) triples; line 0 is the matla."""
    u, out = rng.random(), []
    k = int(rng.integers(reg["k"][0], reg["k"][1] + 1))
    g = int([x for x in rng.permutation(N_FIELDS) if x != f and world["cache"].get(("ok", f, x))][0])
    mode = "plain"
    if u < reg["p_intr"] and L >= k + 3:
        mode, s = "intrusion", int(rng.integers(1, L - k - 1))
    elif u < reg["p_intr"] + reg["p_change"] and L >= 3:
        mode, s = "change", int(rng.integers(1, L - 1))
    for i in range(L):
        if mode == "intrusion" and s <= i < s + k:
            y, pool = _pick(rng, world, g, exclude=f)
            out.append((_line(rng, pool, 3), 0, 0))
            continue
        if mode == "intrusion" and s + k <= i < s + k + 2:
            y, pool = _pick(rng, world, f, other=g)
            out.append((_line(rng, pool, 2), f * N_THEMES + y, 1))
            continue
        if mode == "change" and i >= s:
            if i == s:
                y, pool = _pick(rng, world, g, exclude=f)
            elif rng.random() < 0.5:
                y, pool = _pick(rng, world, g, other=f)
            else:
                y, pool = _pick(rng, world, g)
            out.append((_line(rng, pool, 3 if i == s else int(rng.integers(1, 3))), g * N_THEMES + y, 1))
            continue
        y, pool = _pick(rng, world, f)
        out.append((_line(rng, pool, 3 if i == 0 else int(rng.integers(1, 3))), f * N_THEMES + y, 1))
    return out


def make_split(rng, world, reg, n):
    T = reg["T"]
    tok = np.zeros((n, T, SLOTS), dtype=int)
    form = np.zeros((n, T, 2))
    lab = np.zeros((n, T), dtype=int)
    mask = np.zeros((n, T))
    for b in range(n):
        t, prev = 0, -1
        while t < T:
            L = min(int(rng.integers(reg["plen"][0], reg["plen"][1] + 1)), T - t)
            f = int(rng.choice([x for x in range(N_FIELDS) if x != prev]))
            for i, (w, y, m) in enumerate(_passage(rng, world, reg, L, f)):
                tok[b, t + i], lab[b, t + i], mask[b, t + i] = w, y, m
                form[b, t + i, 0] = 1.0 if i == 0 else 0.0
            t, prev = t + L, f
        form[b, :, 1] = (rng.random(T) < 0.3).astype(float)
    return {"tok": tok, "form": form, "lab": lab, "mask": mask}


def make_dataset(rng, size):
    world = make_world(rng)
    for f in range(N_FIELDS):
        for g in range(N_FIELDS):
            ok = f != g and bool(_pick(rng, world, f, other=g)) and bool(_pick(rng, world, g, exclude=f))
            world["cache"][("ok", f, g)] = ok and bool(_pick(rng, world, g, other=f))
    data = {name: make_split(rng, world, REGIMES[name], size[name]) for name in REGIMES}
    return world, data


def seq_hashes(split):
    return {hashlib.sha1(split["tok"][b].tobytes() + split["form"][b].tobytes()).hexdigest()
            for b in range(len(split["tok"]))}


def batch_of(split, idx):
    return {k: v[idx] for k, v in split.items()}


# ---------------------------------------------------------------- model: held-field reader
def tanasub_evidence(P, tok, mean_evidence=False):
    """Conjunctive field evidence of a line: sum over its words of log sigmoid(e.A_f + a_f),
    the log of a product of per-word compatibilities (every word must fit the field).
    Works for any number of slots; mean_evidence is the negative control of C6.3."""
    e = P["E"][tok]
    s = e @ P["A"] + P["a"]
    ls = -softplus(-s)
    return (ls.mean(axis=-2) if mean_evidence else ls.sum(axis=-2)), e, s


def hf_forward(model, X, dell=None):
    P, ko = model["params"], model["ko"]
    ell, e, s = tanasub_evidence(P, X["tok"])
    if ko.get("tanasub_scorer") == "mean":
        ell = np.repeat(ell.mean(axis=-1, keepdims=True), ell.shape[-1], axis=-1)
    if dell is not None:
        ell = ell + dell
    B, T, F = ell.shape
    # no formal mark, no renewal: r is exactly 0 when every form feature is 0
    r = 1.0 - np.exp(-(X["form"] @ softplus(P["gw"])))
    if ko.get("declared_renewal") == "mean":
        r = np.full_like(r, model["stats"].get("r_mean", 0.5))
    lam = softplus(P["lam"][0])
    hold = ko.get("field_register") != "zero"
    leak = 0.5 if ko.get("half_reset") else 1.0
    h, tau = np.zeros((B, F)), np.zeros(B)
    H, TAU, AL = np.zeros((B, T, F)), np.zeros((B, T)), np.zeros((B, T))
    for t in range(T):
        tau = (1.0 - r[:, t]) * (tau + 1.0)
        al = np.ones(B) if ko.get("matla_admission") == "identity" else np.exp(-lam * tau)
        carry = (1.0 - leak * r[:, t])[:, None] * h if hold else 0.0
        h = carry + al[:, None] * ell[:, t]
        H[:, t], TAU[:, t], AL[:, t] = h, tau, al
    logq = log_softmax(H)
    U = 0.0 * P["U"] if ko.get("surface_path") == "zero" else P["U"]
    se = e.sum(axis=2)
    z = np.einsum("btd,fdc->btfc", se, P["W"]) + (se @ U)[:, :, None, :] + P["c"]
    logpy = log_softmax(z)
    logp = (logq[..., None] + logpy).reshape(B, T, -1)
    cache = dict(e=e, s=s, ell=ell, se=se, r=r, lam=lam, H=H, TAU=TAU, AL=AL,
                 logq=logq, logpy=logpy, U=U, hold=hold)
    return logp, cache


def _recurrence_grads(model, c, dH):
    """Reverse pass through h_t=(1-r_t)h_{t-1}+a_t l_t and tau_t=(1-r_t)(tau_{t-1}+1)."""
    r, H, TAU, AL, ell = c["r"], c["H"], c["TAU"], c["AL"], c["ell"]
    B, T, F = H.shape
    trunc = model["mutant"] == "truncated_bptt"
    admit = model["ko"].get("matla_admission") != "identity"
    Gh, Gtau = np.zeros((B, F)), np.zeros(B)
    dell, dr, dlam = np.zeros_like(ell), np.zeros((B, T)), 0.0
    for t in reversed(range(T)):
        nxt = 1.0 - r[:, t + 1] if (t + 1 < T and not trunc) else np.zeros(B)
        Gh = dH[:, t] + ((nxt[:, None] * Gh) if c["hold"] else 0.0)
        Gtau_carry = nxt * Gtau
        dell[:, t] = AL[:, t, None] * Gh
        dal = np.sum(Gh * ell[:, t], axis=-1)
        h_prev = H[:, t - 1] if t > 0 else np.zeros((B, F))
        tau_prev = TAU[:, t - 1] if t > 0 else np.zeros(B)
        if c["hold"]:
            dr[:, t] -= np.sum(Gh * h_prev, axis=-1)
        dtau = -c["lam"] * AL[:, t] * dal if admit else 0.0
        if admit:
            dlam += float(np.sum(-TAU[:, t] * AL[:, t] * dal))
        Gtau = dtau + Gtau_carry
        dr[:, t] -= (tau_prev + 1.0) * Gtau
    return dell, dr, dlam


def hf_backward(model, X, c, w):
    P, ko = model["params"], model["ko"]
    lab = X["lab"]
    F, C = P["W"].shape[0], P["W"].shape[2]
    fs, ys = lab // C, lab % C
    bi, ti = np.indices(fs.shape)
    dH = w[..., None] * (np.exp(c["logq"]) - np.eye(F)[fs])
    dzs = w[..., None] * (np.exp(c["logpy"][bi, ti, fs]) - np.eye(C)[ys])
    dz = np.zeros(c["logpy"].shape)
    dz[bi, ti, fs] = dzs
    g = {"W": np.einsum("btd,btfc->fdc", c["se"], dz),
         "U": np.einsum("btd,btc->dc", c["se"], dzs), "c": dzs.sum(axis=(0, 1))}
    if ko.get("surface_path") == "zero":
        g["U"] = np.zeros_like(P["U"])
    dse = np.einsum("btc,btdc->btd", dzs, P["W"][fs]) + dzs @ c["U"].T
    dell, dr, dlam = _recurrence_grads(model, c, dH)
    if ko.get("tanasub_scorer") == "mean":
        dell = np.repeat(dell.sum(axis=-1, keepdims=True) / F, F, axis=-1)
    dg = np.zeros_like(dr) if ko.get("declared_renewal") == "mean" else dr * (1.0 - c["r"])
    g["gw"] = np.einsum("bt,btk->k", dg, X["form"]) * sigmoid(P["gw"])
    g["lam"] = np.array([dlam * sigmoid(P["lam"][0])])
    ds = dell[:, :, None, :] * sigmoid(-c["s"])
    g["A"] = np.einsum("btsd,btsf->df", c["e"], ds)
    g["a"] = ds.sum(axis=(0, 1, 2))
    de = ds @ P["A"].T + dse[:, :, None, :]
    g["E"] = np.zeros_like(P["E"])
    np.add.at(g["E"], X["tok"], de)
    if model["mutant"] == "zero_grad_A":
        g["A"] = np.zeros_like(g["A"])
    return g


# ---------------------------------------------------------------- baseline: size-matched GRU
def gru_forward(model, X):
    P = model["params"]
    B, T, S = X["tok"].shape
    Hd = P["Uh"].shape[0]
    x = np.concatenate([P["Emb"][X["tok"]].reshape(B, T, -1), X["form"]], axis=-1)
    a = x @ P["Wx"] + P["bx"]
    h = np.zeros((B, Hd))
    Z, R, N, UU, HP, HS = (np.zeros((B, T, Hd)) for _ in range(6))
    for t in range(T):
        u = h @ P["Uh"]
        zg = sigmoid(a[:, t, :Hd] + u[:, :Hd])
        rg = sigmoid(a[:, t, Hd:2 * Hd] + u[:, Hd:2 * Hd])
        n = np.tanh(a[:, t, 2 * Hd:] + rg * u[:, 2 * Hd:])
        HP[:, t], Z[:, t], R[:, t], N[:, t], UU[:, t] = h, zg, rg, n, u[:, 2 * Hd:]
        h = (1.0 - zg) * n + zg * h
        HS[:, t] = h
    logp = log_softmax(HS @ P["Wo"] + P["bo"])
    return logp, dict(x=x, Z=Z, R=R, N=N, UU=UU, HP=HP, HS=HS, logp=logp)


def gru_backward(model, X, c, w):
    P = model["params"]
    B, T, S = X["tok"].shape
    Hd = P["Uh"].shape[0]
    dlog = w[..., None] * (np.exp(c["logp"]) - np.eye(P["bo"].size)[X["lab"]])
    g = {"Wo": np.einsum("bth,btk->hk", c["HS"], dlog), "bo": dlog.sum(axis=(0, 1))}
    dHS = dlog @ P["Wo"].T
    da_all, g["Uh"], dh = np.zeros((B, T, 3 * Hd)), np.zeros_like(P["Uh"]), np.zeros((B, Hd))
    for t in reversed(range(T)):
        dh = dh + dHS[:, t]
        zg, rg, n, un, hp = c["Z"][:, t], c["R"][:, t], c["N"][:, t], c["UU"][:, t], c["HP"][:, t]
        dan = dh * (1.0 - zg) * (1.0 - n * n)
        daz = dh * (hp - n) * zg * (1.0 - zg)
        dar = dan * un * rg * (1.0 - rg)
        da_all[:, t] = np.concatenate([daz, dar, dan], axis=-1)
        du = np.concatenate([daz, dar, dan * rg], axis=-1)
        g["Uh"] += hp.T @ du
        dh = dh * zg + du @ P["Uh"].T
    g["Wx"] = c["x"].reshape(B * T, -1).T @ da_all.reshape(B * T, -1)
    g["bx"] = da_all.sum(axis=(0, 1))
    dx = (da_all @ P["Wx"].T)[..., :-2].reshape(B, T, S, -1)
    g["Emb"] = np.zeros_like(P["Emb"])
    np.add.at(g["Emb"], X["tok"], dx)
    return g


# ---------------------------------------------------------------- common interface
TASK_TYPES = ["sequence_classification"]


def build_model(in_dim=VOCAB, out_dim=N_CLASSES, task_type="sequence_classification", rng=None, **cfg):
    if task_type not in TASK_TYPES:
        raise ValueError("unsupported task type: %s" % task_type)
    arch, nf = cfg.get("arch", "held_field"), cfg.get("n_fields", N_FIELDS)
    nt = out_dim // nf
    if arch in ("held_field", "line_local"):
        d = cfg.get("d", 12)
        params = {"E": rng.normal(0, 0.3, (in_dim, d)), "A": rng.normal(0, 0.3, (d, nf)), "a": np.zeros(nf),
                  "W": rng.normal(0, 0.3, (nf, d, nt)), "U": rng.normal(0, 0.1, (d, nt)), "c": np.zeros(nt),
                  "gw": np.zeros(2), "lam": np.zeros(1)}
    elif arch == "gru":
        de, hd = cfg.get("emb", 6), cfg.get("hidden", 8)
        ins = SLOTS * de + 2
        params = {"Emb": rng.normal(0, 0.3, (in_dim, de)), "Wx": rng.normal(0, ins ** -0.5, (ins, 3 * hd)),
                  "Uh": rng.normal(0, hd ** -0.5, (hd, 3 * hd)), "bx": np.zeros(3 * hd),
                  "Wo": rng.normal(0, hd ** -0.5, (hd, out_dim)), "bo": np.zeros(out_dim)}
    else:
        raise ValueError("unknown arch: %s" % arch)
    ko = {"field_register": "zero"} if arch == "line_local" else {}
    return {"arch": arch, "params": params, "ko": ko, "mutant": cfg.get("mutant"), "stats": {}}


def forward(model, X):
    return gru_forward(model, X) if model["arch"] == "gru" else hf_forward(model, X)


def _nll(logp, X):
    w = X["mask"] / max(float(X["mask"].sum()), 1.0)
    picked = np.take_along_axis(logp, X["lab"][..., None], axis=-1)[..., 0]
    return float(-(w * picked).sum()), w


def loss_only(model, X):
    return _nll(forward(model, X)[0], X)[0]


def loss_and_grads(model, X):
    logp, c = forward(model, X)
    loss, w = _nll(logp, X)
    back = gru_backward if model["arch"] == "gru" else hf_backward
    return loss, back(model, X, c, w)


def predict(model, X):
    return np.argmax(forward(model, X)[0], axis=-1)


def hidden_states(model, X):
    c = forward(model, X)[1]
    keys = ("HS",) if model["arch"] == "gru" else ("H", "r", "AL", "TAU", "ell", "logq")
    return {k: c[k] for k in keys}


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


def accuracy(model, split):
    hit = (predict(model, split) == split["lab"]) * split["mask"]
    return float(hit.sum() / split["mask"].sum())


# ---------------------------------------------------------------- registries
def modules(model):
    if model["arch"] == "gru":
        return {"gru_core": {"params": ["Emb", "Wx", "Uh", "bx"], "role": "gated recurrent unit", "signature": False},
                "readout": {"params": ["Wo", "bo"], "role": "linear softmax readout", "signature": False}}
    return {
        "tanasub_scorer": {"params": ["E", "A", "a"], "signature": True,
                           "role": "conjunctive per-field evidence: sum of per-token log-sigmoid compatibilities"},
        "field_register": {"params": [], "signature": True,
                           "role": "additive log-evidence memory over fields, carried across lines"},
        "declared_renewal": {"params": ["gw"], "signature": True,
                             "role": "reset gate 1-exp(-form.softplus(g)): form-only, exactly zero without a form signal"},
        "matla_admission": {"params": ["lam"], "signature": True,
                            "role": "exponential admission decay with time since last reset"},
        "theme_reader": {"params": ["W", "c"], "signature": False, "role": "field-conditioned linear readout"},
        "surface_path": {"params": ["U"], "signature": False, "role": "field-independent linear readout"},
    }


KO_MODES = {"field_register": "zero", "matla_admission": "identity", "declared_renewal": "mean",
            "tanasub_scorer": "mean", "surface_path": "zero"}


def knockout(model, name, mode):
    if model["arch"] == "gru" or KO_MODES.get(name) != mode:
        raise ValueError("knockout %s/%s not defined for %s" % (name, mode, model["arch"]))
    out = {k: v for k, v in model.items() if k != "params"}
    out["params"] = {k: v.copy() for k, v in model["params"].items()}
    out["ko"] = dict(model["ko"], **{name: mode})
    out["stats"] = dict(model["stats"])
    return out


MUTANTS = {"sign_flip": "update ascends the loss", "zero_lr": "learning rate forced to zero",
           "zero_grad_A": "gradient of the field-affinity tensor A zeroed",
           "truncated_bptt": "no gradient carried through the register and tau recurrences"}


# ---------------------------------------------------------------- training
def fit(model, data, budget, rng, lr=0.03):
    P, st = model["params"], adam_init(model["params"])
    lr_eff = 0.0 if model["mutant"] == "zero_lr" else lr
    n = len(data["lab"])
    losses = []
    for _ in range(budget):
        loss, g = loss_and_grads(model, batch_of(data, rng.choice(n, min(BATCH, n), replace=False)))
        if not np.isfinite(loss) or not all(np.all(np.isfinite(v)) for v in g.values()):
            raise NonFinite("non-finite loss or gradient during training")
        g, _ = clip_global_norm(g, CLIP)
        if model["mutant"] == "sign_flip":
            g = {k: -v for k, v in g.items()}
        adam_step(P, g, st, lr_eff)
        losses.append(loss)
    if model["arch"] != "gru":
        model["stats"]["r_mean"] = float(hf_forward(model, batch_of(data, np.arange(min(n, 256))))[1]["r"].mean())
    return losses


def train_selected(arch, split_data, steps, rng, mutant=None, lrs=LR_GRID):
    """Same grid, budget and optimizer for every architecture; selection on validation only."""
    best = None
    for lr, child in zip(lrs, rng.spawn(len(lrs))):
        model = build_model(VOCAB, N_CLASSES, "sequence_classification", child, arch=arch, mutant=mutant)
        sub = batch_of(split_data["train"], np.arange(min(512, len(split_data["train"]["lab"]))))
        l0 = loss_only(model, sub)
        fit(model, split_data["train"], steps, child, lr=lr)
        va = accuracy(model, split_data["val"])
        if best is None or va > best[1]:
            best = (model, va, lr, l0, loss_only(model, sub))
    model, va, lr, l0, l1 = best
    model["stats"].update(lr=lr, val_acc=va, loss_init=l0, loss_final=l1)
    return model


# ---------------------------------------------------------------- correctness tests
def trivial_accuracy(data):
    labels = data["train"]["lab"][data["train"]["mask"] > 0]
    top = np.bincount(labels, minlength=N_CLASSES).argmax()
    return float(((data["test"]["lab"] == top) * data["test"]["mask"]).sum() / data["test"]["mask"].sum())


def c1_gradcheck(data, rng, mutant=None, archs=("held_field", "line_local", "gru"), steps=60):
    small = batch_of(data["train"], np.arange(8))
    worst, checked, total = {}, 0, 0
    for arch, child in zip(archs, rng.spawn(len(archs))):
        model = build_model(VOCAB, N_CLASSES, "sequence_classification", child, arch=arch,
                            mutant=None if arch == "gru" else mutant)
        errs = []
        for stage in ("init", "after_training_steps"):
            if stage != "init":
                fit(model, data["train"], steps, child, lr=0.03)
            _, g = loss_and_grads(model, small)
            errs.append(max(gradcheck(lambda: loss_only(model, small), model["params"], g, child).values()))
        worst[arch] = max(errs)
        checked, total = checked + len(model["params"]), total + len(model["params"])
    err = max(worst.values())
    return {"tensors_checked": checked, "tensors_total": total, "max_rel_error": err, "per_model": worst,
            "checked_at": ["init", "after_training_steps"], "passed": bool(err <= 1e-5)}


def c2_determinism(data, seed):
    runs = []
    for _ in range(2):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
        model = build_model(VOCAB, N_CLASSES, "sequence_classification", rng)
        losses = fit(model, data["train"], 40, rng, lr=0.03)
        probe = batch_of(data["test"], np.arange(32))
        hs = hidden_states(model, probe)
        finite = all(np.all(np.isfinite(v)) for v in list(model["params"].values()) + list(hs.values()))
        runs.append((losses, predict(model, probe), finite))
    same = runs[0][0] == runs[1][0] and np.array_equal(runs[0][1], runs[1][1])
    return bool(same and runs[0][2]), "identical losses and outputs: %s; finite: %s" % (same, runs[0][2])


def short_c3(data, rng, steps, trivial, mutant=None):
    """C3 criteria on a fresh model: loss drop fraction and held-out margin over trivial."""
    model = build_model(VOCAB, N_CLASSES, "sequence_classification", rng, mutant=mutant)
    sub = batch_of(data["train"], np.arange(min(512, len(data["train"]["lab"]))))
    l0 = loss_only(model, sub)
    fit(model, data["train"], steps, rng, lr=0.03)
    drop, acc = 1.0 - loss_only(model, sub) / l0, accuracy(model, data["test"])
    th = MIND_CARD["thresholds"]
    return drop >= th["loss_drop_fraction"] and acc >= trivial + th["margin_over_trivial"], drop, acc


def c4_shuffled(data, rng, steps, trivial):
    train = {k: v.copy() for k, v in data["train"].items()}
    on = train["mask"] > 0
    train["lab"][on] = rng.permutation(train["lab"][on])
    model = build_model(VOCAB, N_CLASSES, "sequence_classification", rng)
    fit(model, train, steps, rng, lr=0.03)
    acc = accuracy(model, data["test"])
    ok = acc <= trivial + MIND_CARD["thresholds"]["shuffled_band"]
    return bool(ok), "held-out %.3f vs trivial %.3f (band +%.2f)" % (acc, trivial, MIND_CARD["thresholds"]["shuffled_band"])


def c5_mutants(data, rng, steps, trivial):
    kids = rng.spawn(len(MUTANTS) + 1)
    ctrl_ok, _, _ = short_c3(data, kids[0], steps, trivial)
    rows, detected = [], 0
    for (name, what), child in zip(MUTANTS.items(), kids[1:]):
        gc = c1_gradcheck(data, child, mutant=name, archs=("held_field",), steps=10)
        c3_ok = short_c3(data, child, steps, trivial, mutant=name)[0] if gc["passed"] else True
        hit = (not gc["passed"]) or (not c3_ok)
        detected += hit
        rows.append("%-15s %-9s C1 %s, C3 %s (%s)" % (name, "detected" if hit else "MISSED",
                    "fail" if not gc["passed"] else "pass", "fail" if not c3_ok else "pass", what))
    score = detected / len(MUTANTS)
    return {"detected": detected, "total": len(MUTANTS), "score": score, "control_passes": bool(ctrl_ok),
            "rows": rows, "passed": bool(score == 1.0 and ctrl_ok)}


def _random_hf(rng, gw, lam):
    model = build_model(VOCAB, N_CLASSES, "sequence_classification", rng)
    for k in ("A", "W", "E"):
        model["params"][k] = rng.normal(0, 1.0, model["params"][k].shape)
    model["params"]["gw"], model["params"]["lam"] = np.array(gw, float), np.array([lam])
    return model


def c6_renewal_erases(data, rng, trials=12):
    """With a saturated declared renewal, the reading from that line on is independent of
    everything before it. Random parameters, random prefixes; negative control leaks half."""
    worst, worst_ctrl = 0.0, 0.0
    X = batch_of(data["test"], np.arange(64))
    marks = [np.flatnonzero(X["form"][b, 1:, 0])[0] + 1 for b in range(64)]
    for child in rng.spawn(trials):
        model = _random_hf(child, [40.0, child.normal()], child.normal())
        Y = {k: v.copy() for k, v in X.items()}
        donor = child.permutation(64)
        for b, m in enumerate(marks):
            Y["tok"][b, :m], Y["form"][b, :m] = X["tok"][donor[b], :m], X["form"][donor[b], :m]
        after = np.arange(X["tok"].shape[1])[None, :] >= np.array(marks)[:, None]
        for key in ("", "half_reset"):
            model["ko"] = {key: True} if key else {}
            qa, qb = (np.exp(hidden_states(model, Z)["logq"]) for Z in (X, Y))
            v = float(np.max(np.abs(qa - qb)[after]))
            worst, worst_ctrl = (max(worst, v), worst_ctrl) if not key else (worst, max(worst_ctrl, v))
    ok = worst <= 1e-6 and worst_ctrl > 1e-3
    return ok, "max |dq| after renewal %.1e (<=1e-6); half-reset control %.2f (>1e-3)" % (worst, worst_ctrl)


def c6_admission_decay(rng, trials=40):
    """Without renewal after the opening, line t's evidence reaches the register with
    weight exp(-lam * t). Perturbation measured through the forward pass; the control
    (admission knocked out) must violate the closed form."""
    worst, worst_ctrl = 0.0, 0.0
    for child in rng.spawn(trials):
        lam_raw = child.uniform(-1.5, 1.5)
        model = _random_hf(child, [40.0, -40.0], lam_raw)
        X = {"tok": child.integers(0, VOCAB, (4, 12, SLOTS)), "form": np.zeros((4, 12, 2))}
        X["form"][:, 0, 0] = 1.0
        t0 = int(child.integers(1, 12))
        bump = np.zeros((4, 12, N_FIELDS))
        bump[:, t0, 0] = 1.0
        expect = np.exp(-softplus(lam_raw) * t0)
        for key in ("", "matla_admission"):
            model["ko"] = {key: "identity"} if key else {}
            base, moved = hf_forward(model, X)[1]["H"], hf_forward(model, X, dell=bump)[1]["H"]
            v = float(np.max(np.abs((moved - base)[:, -1, 0] - expect)))
            worst, worst_ctrl = (max(worst, v), worst_ctrl) if not key else (worst, max(worst_ctrl, v))
    ok = worst <= 1e-6 and worst_ctrl > 1e-3
    return ok, "max |ratio - exp(-lam t)| %.1e (<=1e-6); no-admission control %.2f (>1e-3)" % (worst, worst_ctrl)


def c6_conjunctive(rng, trials=20):
    """Adding a word to a line never raises any field's evidence (a conjunction can only
    eliminate). Exhaustive search over every possible added word; mean-evidence control."""
    worst, worst_ctrl = -np.inf, -np.inf
    for child in rng.spawn(trials):
        P = _random_hf(child, [0.0, 0.0], 0.0)["params"]
        S = int(child.integers(1, 4))
        base = child.integers(0, VOCAB, (50, S))
        grown = np.concatenate([np.repeat(base, VOCAB, 0), np.tile(np.arange(VOCAB), 50)[:, None]], axis=1)
        for mean_ev in (False, True):
            old = np.repeat(tanasub_evidence(P, base, mean_ev)[0], VOCAB, 0)
            v = float(np.max(tanasub_evidence(P, grown, mean_ev)[0] - old))
            worst, worst_ctrl = (max(worst, v), worst_ctrl) if not mean_ev else (worst, max(worst_ctrl, v))
    ok = worst <= 1e-12 and worst_ctrl > 1e-3
    return ok, "max evidence increase %.1e (<=0); mean-evidence control %.2f (>1e-3)" % (worst, worst_ctrl)


def c6_content_blind(data, rng):
    """Definition check (labelled, not evidence): renewal reads form only, zero without form."""
    model = _random_hf(rng, rng.normal(0, 2, 2), 0.0)
    X = batch_of(data["test"], np.arange(16))
    Y = dict(X, tok=rng.integers(0, VOCAB, X["tok"].shape))
    ra, rb = hidden_states(model, X)["r"], hidden_states(model, Y)["r"]
    zero = X["form"].sum(-1) == 0
    ok = np.array_equal(ra, rb) and np.all(ra[zero] == 0.0)
    return bool(ok), "definition check: r unchanged under word replacement and 0 without form signal"


def c7_split_integrity(datasets):
    clash = 0
    for data in datasets:
        tr = seq_hashes(data["train"])
        clash += sum(len(tr & seq_hashes(data[s])) for s in ("val", "test", "shifted", "blind"))
    return clash == 0, "sequences shared between train and evaluation splits: %d" % clash


# ---------------------------------------------------------------- hypotheses
def run_seed(seed, size, lrs, mutant):
    kids = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
    world, data = make_dataset(kids[0], size)
    models = {arch: train_selected(arch, data, size["steps"], kid, None if arch == "gru" else mutant, lrs)
              for arch, kid in zip(("held_field", "gru", "line_local"), kids[1:])}
    acc = {a: {s: accuracy(m, data[s]) for s in ("test", "shifted", "blind")} for a, m in models.items()}
    hf = models["held_field"]
    ko = {name: accuracy(knockout(hf, name, mode), data["test"]) - acc["held_field"]["test"]
          for name, mode in KO_MODES.items()}
    learned = {"lam": float(softplus(hf["params"]["lam"][0])), "renew": softplus(hf["params"]["gw"]).tolist(),
               "lr": {a: m["stats"]["lr"] for a, m in models.items()}}
    return {"data": data, "models": models, "acc": acc, "ko": ko, "learned": learned,
            "trivial": trivial_accuracy(data)}


HYP_DIFFS = {
    "H-SIG": lambda r: r["acc"]["held_field"]["shifted"] - r["acc"]["gru"]["shifted"],
    "H-NEC": lambda r: r["ko"]["field_register"] - r["ko"]["surface_path"],
    "H-BLIND": lambda r: r["acc"]["held_field"]["blind"] - r["acc"]["gru"]["blind"],
    "H-RIVAL": lambda r: r["acc"]["held_field"]["test"] - r["acc"]["line_local"]["test"],
}


def evaluate_hypotheses(runs, rng, evaluate):
    out = []
    for h in MIND_CARD["hypotheses"]:
        d = [HYP_DIFFS[h["id"]](r) for r in runs]
        mean, ci = paired_bootstrap(d, rng) if evaluate else (float(np.mean(d)), [None, None])
        out.append({"id": h["id"], "metric": h["metric"], "mean_diff": mean, "ci95": ci, "mesi": h["mesi"],
                    "n_seeds": len(d), "verdict": verdict(mean, ci, h["mesi"], h["direction"]) if evaluate
                    else "not evaluated", "per_seed": [round(x, 4) for x in d]})
    return out


# ---------------------------------------------------------------- real-data bridge
def load_csv(path):
    """CSV columns: poem,line,tok1,tok2,tok3,marker,nuisance,field,theme (no quoted commas)."""
    with open(path, encoding="utf-8") as fh:
        rows = [ln.rstrip("\n").split(",") for ln in fh if ln.strip()]
    head, rows = rows[0], rows[1:]
    col = {name: head.index(name) for name in ("poem", "line", "tok1", "tok2", "tok3", "marker", "nuisance",
                                               "field", "theme")}
    vocab, fields, themes, poems = {}, {}, {}, {}
    for r in rows:
        ids = [vocab.setdefault(r[col[k]], len(vocab)) for k in ("tok1", "tok2", "tok3")]
        f, y = fields.setdefault(r[col["field"]], len(fields)), themes.setdefault(r[col["theme"]], len(themes))
        poems.setdefault(r[col["poem"]], []).append((int(r[col["line"]]), ids, float(r[col["marker"]]),
                                                     float(r[col["nuisance"]]), f, y))
    T, n = max(len(v) for v in poems.values()), len(poems)
    split = {"tok": np.zeros((n, T, SLOTS), int), "form": np.zeros((n, T, 2)), "lab": np.zeros((n, T), int),
             "mask": np.zeros((n, T))}
    for b, lines in enumerate(poems.values()):
        for t, (_, ids, mk, nu, f, y) in enumerate(sorted(lines)):
            split["tok"][b, t], split["form"][b, t] = ids, (mk, nu)
            split["lab"][b, t], split["mask"][b, t] = f * len(themes) + y, 1.0
    return split, len(vocab), len(fields), len(themes)


def data_bridge(path, rng, steps):
    if not path:
        return ["skipped: no --data PATH given (expects an annotated CSV, e.g. from the Ganjoor corpus)"]
    split, V, nf, nt = load_csv(path)
    order = rng.permutation(len(split["lab"]))
    cut = max(1, int(0.8 * len(order)))
    tr, te = batch_of(split, order[:cut]), batch_of(split, order[cut:])
    rows = ["%s: %d poems, vocab %d, %d fields x %d themes" % (os.path.basename(path), len(order), V, nf, nt)]
    for arch in ("held_field", "gru"):
        model = build_model(V, nf * nt, "sequence_classification", rng, arch=arch, n_fields=nf)
        fit(model, tr, steps, rng, lr=0.03)
        rows.append("%-10s held-out line accuracy %.3f" % (arch, accuracy(model, te)))
    return rows


# ---------------------------------------------------------------- report
def _ms(xs):
    return "%.3f ± %.3f" % (float(np.mean(xs)), float(np.std(xs)))


def _ci(ci):
    return "[%+.3f, %+.3f]" % tuple(ci) if ci[0] is not None else "[   n/a ,   n/a ]"


def build_sections(rep, runs, gc, correct, mut, hyps, kos, bridge):
    sizes = rep["n_params"]
    sec = [("environment", ["python %s · numpy %s" % (rep["environment"]["python"], rep["environment"]["numpy"]),
                            "file %s" % rep["file"], "seeds %s" % rep["seeds"], "runtime_s %.1f" % rep["runtime_s"],
                            "n_params held_field=%d gru_baseline=%d line_local_rival=%d (baseline within 10%%: %s)"
                            % (sizes["held_field"], sizes["gru"], sizes["line_local"],
                               abs(sizes["gru"] - sizes["held_field"]) <= 0.1 * sizes["held_field"])]),
           ("gradient check", ["tensors %d/%d · max rel error %.2e · checked at init and after 60 steps · %s"
                               % (gc["tensors_checked"], gc["tensors_total"], gc["max_rel_error"],
                                  "passed" if gc["passed"] else "FAILED"),
                               "per model: " + ", ".join("%s %.1e" % kv for kv in gc["per_model"].items())])]
    sec.append(("correctness", ["%-5s %-28s %s  %s" % (c["id"], c["name"], "PASS" if c["passed"] else "FAIL",
                                                       c["detail"]) for c in correct]))
    sec.append(("mutants", ["detected %d/%d · score %.2f · unmutated control passes: %s"
                            % (mut["detected"], mut["total"], mut["score"], mut["control_passes"])] + mut["rows"]))
    acc_rows = ["%-12s %-15s %-15s %-15s" % ("model", "test", "shifted", "blind")]
    for arch in ("held_field", "gru", "line_local"):
        acc_rows.append("%-12s %-15s %-15s %-15s" % ((arch,) + tuple(
            _ms([r["acc"][arch][s] for r in runs]) for s in ("test", "shifted", "blind"))))
    acc_rows.append("%-12s %-15s" % ("trivial", _ms([r["trivial"] for r in runs])))
    sec.append(("line accuracy on scored lines (mean ± sd over seeds)", acc_rows))
    sec.append(("hypotheses", ["%-8s diff %+.3f  ci95 %s  mesi %.2f  n=%d  %s" % (
        h["id"], h["mean_diff"], _ci(h["ci95"]), h["mesi"], h["n_seeds"], h["verdict"]) for h in hyps]))
    sec.append(("knockouts (test accuracy change vs full held_field)", ["%-17s signature=%-5s %+.3f  ci95 %s" % (
        k["module"], k["signature"], k["metric_change"], _ci(k["ci95"])) for k in kos]))
    lam = [r["learned"]["lam"] for r in runs]
    ren = np.array([r["learned"]["renew"] for r in runs])
    sec.append(("learned (held_field)", [
        "admission lam %s -> weight at 1 and 3 lines after the opening %.3f, %.3f"
        % (_ms(lam), float(np.exp(-np.mean(lam))), float(np.exp(-3 * np.mean(lam)))),
        "renewal strength marker %s · nuisance %s" % (_ms(ren[:, 0]), _ms(ren[:, 1])),
        "selected lr per seed: %s" % [r["learned"]["lr"] for r in runs]]))
    sec.append(("real-data bridge", bridge))
    sec.append(("task types", TASK_TYPES))
    return sec


# ---------------------------------------------------------------- protocol and entry point
def correctness_suite(args, first, size, lrs, trivial, rng, datasets):
    kids = rng.spawn(8)
    data, out = first["data"], []
    gc = c1_gradcheck(data, kids[0], mutant=args.mutant)
    out.append(("C1", "gradient_check", gc["passed"], "max rel error %.2e over %d tensors"
                % (gc["max_rel_error"], gc["tensors_total"])))
    out.append(("C2", "determinism_finiteness") + c2_determinism(data, args.seed))
    hf = first["models"]["held_field"]["stats"]
    drop = 1.0 - hf["loss_final"] / hf["loss_init"]
    acc = first["acc"]["held_field"]["test"]
    th = MIND_CARD["thresholds"]
    out.append(("C3", "learning", drop >= th["loss_drop_fraction"] and acc >= trivial + th["margin_over_trivial"],
                "loss drop %.2f (>=%.2f); held-out %.3f vs trivial %.3f (+%.2f)"
                % (drop, th["loss_drop_fraction"], acc, trivial, th["margin_over_trivial"])))
    out.append(("C4", "shuffled_label_control") + c4_shuffled(data, kids[1], size["steps"], trivial))
    if args.mutant:
        mut = {"detected": 0, "total": len(MUTANTS), "score": 0.0, "control_passes": False,
               "rows": ["not run under --mutant %s" % args.mutant], "passed": True}
    else:
        mut = c5_mutants(data, kids[2], size["steps"] // 2, trivial)
    out.append(("C5", "mutant_detection", mut["passed"], "score %.2f" % mut["score"]))
    for cid, name, fn in (("C6.1", "declared_renewal_erases", lambda: c6_renewal_erases(data, kids[3])),
                          ("C6.2", "admission_decay_closed_form", lambda: c6_admission_decay(kids[4])),
                          ("C6.3", "conjunctive_monotonicity", lambda: c6_conjunctive(kids[5])),
                          ("C6.4", "content_blind_renewal", lambda: c6_content_blind(data, kids[6]))):
        out.append((cid, name) + fn())
    out.append(("C7", "split_integrity") + c7_split_integrity(datasets))
    return gc, mut, [{"id": i, "name": n, "passed": bool(p), "detail": d} for i, n, p, d in out]


def protocol(args):
    t0 = time.time()
    quick = args.quick
    size, lrs = SIZES["quick" if quick else "full"], ((0.03,) if quick else LR_GRID)
    seeds = [args.seed + i for i in range(1 if quick else args.seeds)]
    print("chapter 0295 · held-field reader · seeds %s · %s" % (seeds, "quick" if quick else "full protocol"))
    runs = []
    for s in seeds:
        runs.append(run_seed(s, size, lrs, args.mutant))
        print("  seed %d done (%.1f s)" % (s, time.time() - t0))
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, 295]))
    kids = rng.spawn(3)
    trivial = runs[0]["trivial"]
    gc, mut, correct = correctness_suite(args, runs[0], size, lrs, trivial, kids[0], [r["data"] for r in runs])
    hyps = evaluate_hypotheses(runs, kids[1], evaluate=not quick and len(runs) >= 2)
    kos = []
    for name in KO_MODES:
        d = [r["ko"][name] for r in runs]
        mean, ci = paired_bootstrap(d, kids[2]) if len(d) >= 2 else (d[0], [None, None])
        kos.append({"module": name, "signature": modules(runs[0]["models"]["held_field"])[name]["signature"],
                    "metric_change": mean, "ci95": ci})
    bridge = data_bridge(args.data, np.random.default_rng(np.random.SeedSequence([args.seed, 7])), size["steps"])
    runtime = time.time() - t0
    budget = 20.0 if quick else 180.0
    correct.append({"id": "C8", "name": "budget", "passed": runtime <= budget,
                    "detail": "%.1f s (<= %.0f s)" % (runtime, budget)})
    failed = [c["id"] for c in correct if not c["passed"]]
    code = 0 if not failed else (3 if failed == ["C8"] else 1)
    rep = {"schema_version": "1.0", "chapter": 295, "file": os.path.basename(__file__),
           "card_revision": MIND_CARD["card_revision"],
           "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
           "seeds": seeds, "runtime_s": round(runtime, 2),
           "n_params": {a: n_params(m) for a, m in runs[0]["models"].items()},
           "gradcheck": {k: gc[k] for k in ("tensors_checked", "tensors_total", "max_rel_error", "checked_at", "passed")},
           "correctness": correct, "mutants": {k: mut[k] for k in ("detected", "total", "score")},
           "hypotheses": hyps, "knockouts": kos, "task_types": TASK_TYPES, "exit_code": code,
           "accuracy": {a: {s: [round(r["acc"][a][s], 4) for r in runs] for s in ("test", "shifted", "blind")}
                        for a in ("held_field", "gru", "line_local")},
           "trivial": [round(r["trivial"], 4) for r in runs], "mutant": args.mutant}
    print_report(rep, build_sections(rep, runs, gc, correct, mut, hyps, kos, bridge))
    if args.json:
        write_json(args.json, rep)
    return code


def main(argv=None):
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__),
                                 description="Chapter 0295: held-field reader with declared renewal")
    ap.add_argument("--quick", action="store_true", help="one seed, reduced steps, all correctness tests")
    ap.add_argument("--seed", type=int, default=0, help="base seed")
    ap.add_argument("--seeds", type=int, default=5, help="number of seeds (default 5)")
    ap.add_argument("--json", metavar="PATH", help="also write the JSON report")
    ap.add_argument("--card", action="store_true", help="print MIND_CARD as JSON and exit")
    ap.add_argument("--mutant", metavar="NAME", help="run with a registered mutant: %s" % ", ".join(MUTANTS))
    ap.add_argument("--data", metavar="PATH", help="optional annotated CSV for the real-data bridge")
    args = ap.parse_args(argv)
    if args.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    if (args.mutant and args.mutant not in MUTANTS) or args.seeds < 1 or (args.data and not os.path.exists(args.data)):
        ap.print_usage()
        return 2
    try:
        return protocol(args)
    except NonFinite as exc:
        print("non-finite values: %s" % exc)
        return 4


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0370 · Kamoli Khujandi (1321-1400)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 19, Minds 361-380 Available on Amazon https://www.amazon.com/dp/B0HQX147Y7
# Resume and Interactive Demos at https://artificiology.com/
# END ATTRIBUTION 
"""Javab network: answering a predecessor in its own form without taking its meaning.

Thesis
    An answer borrows what is constant across its model's couplets (the zamin:
    meter, rhyme, radif) and owes the model nothing else; the borrowed form may
    rescale how a meaning is set, never supply the meaning, which must be driven
    by the answerer's own state (hal).

Evidence and provenance (belief: his Divan survives; a dozen literary qit'as and
Jami's criticism document the practice; the reading below is interpretation)
    D1 Jami, Baharestan, rawza 7: Kamal follows Hasan Dehlavi's light meters and
       strange rhymes and radifs (tatabbu'), hence the charge "thief of Hasan";
       Jami quotes Kamal's own pun: no one caught me at the mouth of any breach,
       so it is known I am the thief of Hasan (also: "a fine thief").
    D2 Jami: the meanings in Kamal are subtler than any in Hasan; Ali-Mardan
       (via Losensky, EIr 2010): his answers diverge from their models more than
       they resemble them.
    D3 Divan (ed. Golsorkhi I, 417, via EIr): an answer "not less than" Sa'di's;
       "these two ghazals came out better than one another".
    D4 Qit'as 38 and 54 (Ganjoor): most of my ghazals are seven couplets; in the
       Ganjoor text 862 of 1,252 poems have exactly seven couplets.
    D5 Jami: excess of subtlety took the verse past fluency and left it without
       the savour of love (the blind spot used for H-BLIND).
    D6 Qit'a 88: answers made to his poems in Tabriz stand to his as a weaver's
       hut to the Ka'ba, a lodge beside the mosque of Alishah (answer chains
       can degrade; lineage diagnostic).
    D7 Qit'a 96: copy as much as you like from Kamal's Divan, but sink into each
       letter like ink rather than pass over it like the pen.

Doctrine -> mechanism -> test
    D1, D2  M1 zamin_pool: permutation-invariant mean over the model's couplets
            is the ONLY route from the model to the answer    C6.1 breach bound  H-SIG
    D3      M2 javab_gain: pooled form multiplies a self-driven content drive by
            a gain in (0, 2); it cannot add content          C6.2 (definition)  H-NEC
    D5      thin hal with apt model meanings                 -                  H-BLIND
    D4      n = 7 pooling; leakage of one couplet scales 1/n  seven diagnostic   -
    D6      answers to answers over four generations         lineage diagnostic -
    0307    ledger-novelty rival (anti-repetition penalty)   -                  H-RIVAL

Research question
    In retrieval-augmented, in-context, weak-to-strong and synthetic-data
    learning, a generator conditioned on an exemplar can learn to copy the
    exemplar's content when that content is spuriously predictive. Does barring
    exemplar content architecturally (set-invariant form only, modulatory only)
    remove the copying under shift, and what does it cost when the exemplar's
    content is in fact apt?

Closest prior art and the delta
    Neural Statistician set context (Edwards & Storkey 2017); contextual
    modulation of a driving input (Kay & Phillips 1997); FiLM (Perez et al.
    2018); bilinear style/content (Tenenbaum & Freeman 2000); exemplar
    cross-attention and induction-style copying (Olsson et al. 2022). Delta: the
    exemplar set reaches the output only as its mean, and that mean may only
    gain-modulate a drive computed from the answerer's own state, so per-couplet
    exemplar content is barred and the residual leak is bounded by 1/n.

Blind spot
    When the answerer's state is thin and the model's meanings are right, the
    discipline refuses the help a borrower would take (Jami's complaint).

Task (generative process)
    world: fixed maps A_z, A_r, A_m, A_i and a hal encoder tanh(Q m + q)
    ghazal: zamin z ~ N(0, I_3); seven rhyme slots in random order
    couplet x = A_z z + A_r e_k + A_m m + A_i vec(z m^T) + noise
    answer couplet j: slot k_j, meaning m_j ~ N(0, I_4), hal h_j = tanh(Q m_j + q) + noise
    the model couplet with the same slot k_j has meaning rho m_j + sqrt(1-rho^2) xi
    target y_j = A_z z + A_r e_k + A_m m_j + A_i vec(z m_j^T)
    splits: train/val/iid rho = 0.9; shifted rho = 0; blind rho = 0.9 with hal noise 1.2

Limits
    Synthetic vectors stand in for couplets; no text is generated and nothing is
    claimed about Kamal's own cognition. The file is a research prototype of one
    AGI-relevant mechanism, not an AGI.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 2,
    "revision_log": [
        "rev 2 (2026-10-01): after a pilot on non-confirmatory seeds 100-101, the theft-index diagnostic was redefined to "
        "regress the answer's error on the counterpart couplet's meaning share only, because the first definition (error "
        "projected on counterpart minus target) also rewarded shrinkage toward the mean; and gradient entries below 1e-5 "
        "are judged by the absolute floor (central-difference roundoff ~1e-10 at eps 1e-6). Hypotheses, metrics, mesi "
        "and thresholds unchanged; confirmatory seeds 0-4 run once after this revision."],
    "generation": {"template_version": "codeguidelines 1.0 (2026-09-15)", "generator": "Claude",
                   "generator_version": "Opus 5.5", "date": "2026-10-01"},
    "id": 370, "figure": "Kamoli Khujandi", "born": 1321, "died": 1400,
    "civilization": "Persian (Tajik; Transoxiana, active in Jalayirid and Timurid Tabriz)",
    "provenance": "belief",
    "thesis": ("An answer inherits what is constant across its model's couplets (the zamin) and may let it "
               "rescale how a meaning is set, but each couplet's meaning must be driven by the answerer's own "
               "state; the borrowed form can modulate a meaning and never supply one."),
    "evidence": [
        {"id": "D1", "claim": "Kamal answered Hasan Dehlavi in his meters, rhymes and radifs and was called 'thief of Hasan'; "
                              "his own pun says no one caught him at any breach", "basis": "primary",
         "source": "Jami, Baharestan, rawza 7 (ed. Hakemi), quoting a couplet found in some copies of the Divan"},
        {"id": "D2", "claim": "his answers carry subtler meanings than their models and diverge from them more than they resemble them",
         "basis": "scholarship", "source": "Jami, Baharestan rawza 7; Ali-Mardan 1996 via Losensky, EIr 2010"},
        {"id": "D3", "claim": "an answer 'not less than' Sa'di's; the two ghazals came out better than one another",
         "basis": "primary", "source": "Divan, ed. Golsorkhi 1995, I p. 417 (via Losensky, EIr 2010)"},
        {"id": "D4", "claim": "most of his ghazals are seven couplets (stated by him; 862 of 1,252 poems in the Ganjoor text)",
         "basis": "primary", "source": "Divan, qit'as 38 and 54 (Ganjoor numbering); EIr citing Golsorkhi II 992-93"},
        {"id": "D5", "claim": "excess of fine meaning took his verse past fluency and left it without the savour of love",
         "basis": "scholarship", "source": "Jami, Baharestan rawza 7"},
        {"id": "D6", "claim": "answers made to his poems in Tabriz fall short of them as a weaver's hut of the Ka'ba",
         "basis": "primary", "source": "Divan, qit'a 88 (Ganjoor numbering)"},
        {"id": "D7", "claim": "copying from his Divan is invited, but reading must sink into each letter like ink",
         "basis": "primary", "source": "Divan, qit'a 96 (Ganjoor numbering)"},
        {"id": "D8", "claim": "exile to Saray 1385-c.1395; Divan gathered by disciples as Tarjoman al-hal",
         "basis": "scholarship", "source": "Losensky, EIr 2010; Dawlatshah, Tadhkira p. 245"},
    ],
    "research_question": {
        "category": "robustness to spurious correlation",
        "question": ("Does restricting an exemplar's influence to its set-invariant component, and letting that "
                     "component only modulate a self-driven content pathway, remove exemplar-content copying under "
                     "shift, and what does it cost when the exemplar content is apt?")},
    "mechanism": {
        "name": "Javab network (zamin pool, hal drive, modulatory javab units)",
        "family": "set-conditioned two-compartment network (drive x bounded context gain)",
        "signature_modules": ["zamin_pool", "javab_gain"],
        "closest_prior_art": [
            "Neural Statistician set context (Edwards & Storkey, ICLR 2017)",
            "Contextual modulation of driving input (Kay & Phillips, Neural Computation 9(4), 1997)",
            "FiLM feature-wise modulation (Perez et al., AAAI 2018)",
            "Bilinear style/content separation (Tenenbaum & Freeman, Neural Computation 12(6), 2000)",
            "Exemplar cross-attention and induction-style copying (Olsson et al., arXiv:2209.11895)"],
        "overlap": "Medium",
        "prior_art_queries": [
            "set-conditioned generation exemplar copying shortcut", "contextual modulation apical drive network",
            "FiLM gain only no additive conditioning", "weak-to-strong imitation of supervisor errors",
            "retrieval augmented generation regurgitation format versus content", "model collapse recursive training"],
        "contribution_type": "mechanism",
        "delta": ("The exemplar set reaches the output only as its mean, which may only rescale a drive computed from "
                  "the answerer's own state by a factor in (0, 2); per-couplet exemplar content is barred and the "
                  "residual leak of any one couplet is bounded by 1/n."),
    },
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1 zamin_pool", "property_test": "C6.1", "hypothesis": "H-SIG"},
        {"doctrine": "D2", "mechanism": "M1 zamin_pool", "property_test": "C6.1", "hypothesis": "H-SIG"},
        {"doctrine": "D3", "mechanism": "M2 javab_gain", "property_test": "C6.2", "hypothesis": "H-NEC"},
        {"doctrine": "D5", "mechanism": "M3 hal_interpreter", "property_test": "none", "hypothesis": "H-BLIND"},
        {"doctrine": "D4", "mechanism": "M1 zamin_pool (n = 7)", "property_test": "C6.1", "hypothesis": "seven diagnostic"},
        {"doctrine": "D6", "mechanism": "whole network iterated", "property_test": "none", "hypothesis": "lineage diagnostic"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": ("On the shifted split the javab network has higher skill than the size-matched "
                                      "exemplar-attention decoder, which can carry the counterpart's meaning through the shared rhyme slot."),
         "metric": "skill = 1 - MSE/MSE_trivial", "split": "shifted", "comparison": "kamal - attention_baseline",
         "direction": "greater", "mesi": 0.05, "seeds": 5},
        {"id": "H-NEC", "statement": ("Replacing the javab gain by its mean lowers shifted-split skill more than zeroing "
                                      "the size-matched hal second-thought residual."),
         "metric": "skill after knockout", "split": "shifted",
         "comparison": "skill(javab_gain->mean) - skill(hal_second_thought->zero)", "direction": "less", "mesi": 0.05, "seeds": 5},
        {"id": "H-BLIND", "statement": ("With a thin hal (noise 1.2) and apt model meanings (rho 0.9), the javab network is "
                                        "worse than the borrower."),
         "condition": "blind split: rho = 0.9, hal noise 1.2", "grounding": ("Jami: fine meaning pursued to excess left the verse "
                                                                              "without the savour the tradition carried; exile thinned the state"),
         "metric": "skill", "comparison": "kamal - attention_baseline", "direction": "less", "mesi": 0.05, "seeds": 5},
        {"id": "H-RIVAL", "statement": ("On the shifted split the javab network beats the ledger-novelty decoder, whose "
                                        "anti-repetition penalty repels the shared form together with the borrowed meaning."),
         "metric": "skill", "split": "shifted", "comparison": "kamal - ledger_rival", "direction": "greater",
         "mesi": 0.03, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.5, "margin_over_trivial": 0.3, "shuffle_band": 0.08,
                   "gradcheck_rel": 1e-5, "gradcheck_abs_floor": 1e-9},
    "probe_predictions": [
        {"probe": "P1", "expected": "below baseline (no recurrence; state enters only as the current hal)"},
        {"probe": "P3", "expected": "above baseline when inputs factor into a shared form and item content"},
        {"probe": "P6", "expected": "below baseline (no exemplar content may be read in context)"},
        {"probe": "P9", "expected": "above baseline under context noise (gain bounded in (0, 2))"},
        {"probe": "P10", "expected": "equal to baseline"},
    ],
    "dialectic_links": [
        {"chapter": 331, "relation": "teacher", "test": "none (Sa'di's ghazals were Kamal's main models; not implemented)"},
        {"chapter": 369, "relation": "rival", "test": "none (contemporary answered in shared zamins; contrast stated in card)"},
        {"chapter": 307, "relation": "rival", "test": "H-RIVAL (qit'as 29, 54, 78 measure Kamal's seven couplets against Nizami)"},
    ],
    "corpus_neighbors": [
        {"chapter": 369, "similarity": None, "difference": ("Hafez keeps several readings of one text uncollapsed under a frozen "
                                                             "radif target; here form passes between two poets' texts and may only gain-modulate a self-driven meaning.")},
        {"chapter": 307, "similarity": None, "difference": ("Nizami penalises outputs that repeat stored meanings; here no penalty is "
                                                             "used, the route for borrowed content is removed and form is inherited on purpose.")},
        {"chapter": 296, "similarity": None, "difference": "Khaqani's opening field governs later lines of one poem; here the governing form is pooled from another poet's ghazal."},
        {"chapter": 234, "similarity": None, "difference": "Rudaki's minimal key reopens a world the listener holds; here the answerer must supply new content from its own state."},
        {"chapter": 368, "similarity": None, "difference": "Ibn al-Khatib partitions a hazard among causal doors; unrelated survival-model family."},
    ],
    "barometer": {
        "cognitive_processing": ["set-pooled form inference over seven exemplar items"],
        "embodied_cognition": ["not measured; cross-embodiment framing only"],
        "world_modeling": ["form x meaning interaction world with shifted and blind splits"],
        "consciousness": ["not measured (no confidence head)"],
        "language_understanding": ["synthetic answer-poem grammar: form fidelity under a borrowed zamin"],
        "emotional_intelligence": ["not measured"],
        "creativity": ["novelty relative to exemplar under a validity constraint: theft index on shifted split"],
        "autonomy": ["four-generation lineage of answers to answers without convergence on the first model"],
    },
    "task_types": ["vector_regression"],
    "applications": [
        {"use": "exemplar-conditioned answering that takes format from retrieved exemplars and content from the query",
         "sector": "enterprise search and legal drafting", "dataset": "MS MARCO"},
        {"use": "learning from weaker or earlier models' labels without inheriting their content errors",
         "sector": "AI alignment and model training", "dataset": "BoolQ and SciQ as used in weak-to-strong benchmarks"},
        {"use": "cross-embodiment imitation: inherit a demonstration's timing and structure, drive joint content from own state",
         "sector": "robotics", "dataset": "Open X-Embodiment (2023)"},
        {"use": "measuring inheritance of form versus meaning in same-meter same-rhyme Persian poem pairs",
         "sector": "digital humanities", "dataset": "Ganjoor Persian poetry corpus"},
    ],
    "safety_notes": ("Poetry figure; no hazardous domain. Synthetic data only. No sentence is generated or attributed to "
                     "Kamal; the file claims no replication of his mind. Barring exemplar content is relevant to "
                     "copyright-safe generation but is not a legal guarantee."),
}

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

CHAPTER = 370
CFG = dict(d_z=3, d_m=4, d_x=12, d_h=6, n_c=7, n_r=7, hid=16, gin=16, units=16, att=8, mlp=18,
           rho_train=0.9, sig_x=0.10, sig_h=0.15, sig_h_thin=1.2,
           n_train=300, n_val=60, n_test=120, steps=1500, quick_steps=450, batch=64, lr=3e-3, clip=5.0,
           ledger_lambda=0.5, ledger_tau_frac=0.25, lineage_generations=4)
BUDGET_FULL, BUDGET_QUICK = 180.0, 20.0


class NonFinite(Exception):
    pass


# BEGIN STANDARD UTILITIES v1.0
def softmax(x, axis=-1):
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def logsumexp(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(x - m), axis=axis))


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
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= sign * self.lr * mh / (np.sqrt(vh) + self.eps)


def clip_global(grads, max_norm):
    norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    if not math.isfinite(norm):
        raise NonFinite("non-finite gradient norm")
    if norm > max_norm:
        for k in grads:
            grads[k] *= max_norm / norm
    return norm


def grad_check(loss_fn, params, grads, rng, n_entries=20, eps=1e-6, rel_tol=1e-5, abs_floor=1e-9):
    """Central differences on >= n_entries random entries per tensor plus the largest-gradient entry."""
    worst, report = 0.0, {}
    for k, P in params.items():
        flat = P.reshape(-1)
        g = grads[k].reshape(-1)
        idx = set(rng.choice(flat.size, size=min(n_entries, flat.size), replace=False).tolist())
        idx.add(int(np.argmax(np.abs(g))))
        bad = 0
        for i in sorted(idx):
            old = flat[i]
            flat[i] = old + eps
            lp = loss_fn()
            flat[i] = old - eps
            lm = loss_fn()
            flat[i] = old
            num = (lp - lm) / (2 * eps)
            diff = abs(num - g[i])
            scale = max(abs(num), abs(g[i]))
            rel = diff / scale if scale > 1e-5 else 0.0   # below 1e-5 roundoff (~1e-10) dominates: absolute floor
            if diff > abs_floor and rel > rel_tol:
                bad += 1
            worst = max(worst, rel)
        report[k] = bad == 0
    return all(report.values()), worst, report


def bootstrap_ci(diffs, rng, n_boot=2000, alpha=0.05):
    d = np.asarray(diffs, dtype=float)
    boots = np.array([np.mean(d[rng.integers(0, d.size, d.size)]) for _ in range(n_boot)])
    return float(np.mean(d)), [float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))]


def verdict(mean, ci, direction, mesi):
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


def write_report(lines, payload, json_path):
    print("\n".join(lines))
    if json_path:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
# END STANDARD UTILITIES


# ----------------------------------------------------------------------------- data and tasks
def make_world(rng, c=CFG):
    """Fixed maps of one poetic world: zamin, rhyme slots, meanings, their setting, and the hal encoder."""
    dz, dm, dx, dh, nr = c["d_z"], c["d_m"], c["d_x"], c["d_h"], c["n_r"]
    return {"Az": rng.normal(0, 1 / math.sqrt(dz), (dx, dz)), "Ar": rng.normal(0, 0.6, (dx, nr)),
            "Am": rng.normal(0, 1 / math.sqrt(dm), (dx, dm)), "Ai": rng.normal(0, 0.9 / math.sqrt(dz * dm), (dx, dz * dm)),
            "Q": rng.normal(0, 1.3 / math.sqrt(dm), (dh, dm)), "q": rng.normal(0, 0.3, dh)}


def realize(W, z, k, m):
    """Couplet surface: zamin + rhyme-slot signature + meaning + meaning set in the meter (z x m)."""
    inter = (z[..., :, None] * m[..., None, :]).reshape(z.shape[:-1] + (-1,))
    return z @ W["Az"].T + W["Ar"].T[k] + m @ W["Am"].T + inter @ W["Ai"].T


def make_split(W, rng, n_g, rho, sig_h, n_c=None, c=CFG):
    """Ghazals and their answers. The model couplet sharing an answer couplet's rhyme slot has meaning
    rho * m_answer + sqrt(1 - rho^2) * noise: the copy shortcut the tradition offered (rho = 0.9 in training)."""
    n_c = n_c or c["n_c"]
    nr, dz, dm, dx = c["n_r"], c["d_z"], c["d_m"], c["d_x"]
    n_a = min(n_c, nr)
    XM, RK, H, Y, YC, GID, HASH = [], [], [], [], [], [], []
    for g in range(n_g):
        z = rng.normal(size=dz)
        sM = rng.permutation(np.arange(n_c) % nr)
        sA = rng.permutation(n_a)
        mA = rng.normal(size=(n_a, dm))
        mM = rng.normal(size=(n_c, dm))
        counter = np.array([int(np.flatnonzero(sM == k)[0]) for k in sA])
        mM[counter] = rho * mA + math.sqrt(1 - rho * rho) * rng.normal(size=(n_a, dm))
        zz = np.broadcast_to(z, (n_c, dz))
        xm = realize(W, zz, sM, mM) + c["sig_x"] * rng.normal(size=(n_c, dx))
        za = np.broadcast_to(z, (n_a, dz))
        y = realize(W, za, sA, mA)
        yc = realize(W, za, sA, mM[counter]) - realize(W, za, sA, np.zeros((n_a, dm)))  # counterpart's meaning share
        h = np.tanh(mA @ W["Q"].T + W["q"]) + sig_h * rng.normal(size=(n_a, c["d_h"]))
        XM.append(np.broadcast_to(xm, (n_a, n_c, dx)))
        RK.append(np.eye(nr)[sA])
        H.append(h); Y.append(y); YC.append(yc); GID.append(np.full(n_a, g))
        HASH.append(hashlib.sha1(np.round(xm, 10).tobytes() + np.round(z, 10).tobytes()).hexdigest())
    return {"XM": np.concatenate(XM).copy(), "rk": np.concatenate(RK), "h": np.concatenate(H),
            "y": np.concatenate(Y), "yc": np.concatenate(YC), "gid": np.concatenate(GID), "hashes": HASH}


def build_data(seed, c=CFG):
    ss = np.random.SeedSequence([CHAPTER, seed])
    r_world, r_tr, r_va, r_iid, r_sh, r_bl = [np.random.default_rng(s) for s in ss.spawn(6)]
    W = make_world(r_world, c)
    d = {"world": W,
         "train": make_split(W, r_tr, c["n_train"], c["rho_train"], c["sig_h"], c=c),
         "val": make_split(W, r_va, c["n_val"], c["rho_train"], c["sig_h"], c=c),
         "iid": make_split(W, r_iid, c["n_test"], c["rho_train"], c["sig_h"], c=c),
         "shifted": make_split(W, r_sh, c["n_test"], 0.0, c["sig_h"], c=c),
         "blind": make_split(W, r_bl, c["n_test"], c["rho_train"], c["sig_h_thin"], c=c)}
    d["ybar"] = d["train"]["y"].mean(axis=0)
    return d


def take(batch, idx):
    return {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in batch.items() if k != "hashes"}


# ----------------------------------------------------------------------------- model
TASK_TYPES = ["vector_regression"]


def _w(rng, o, i):
    return rng.normal(0, 1 / math.sqrt(i), (o, i))


def build_model(in_dim, out_dim, task_type, rng, kind="kamal", n_r=None, **cfg):
    """kind: 'kamal' (javab network), 'attn' (exemplar-attention decoder), 'ledger' (attn + anti-repetition ledger)."""
    if task_type not in TASK_TYPES:
        raise ValueError("unsupported task type " + str(task_type))
    c = dict(CFG); c.update(cfg)
    nr = n_r or c["n_r"]
    dh, dx, G = in_dim, out_dim, c["gin"]
    P = {"Wg": _w(rng, G, dh), "bg": np.zeros(G), "Wr": 0.5 * _w(rng, G, G), "br": np.zeros(G)}
    if kind == "kamal":
        Hd, K = c["hid"], c["units"]
        P.update({"We": _w(rng, Hd, dx), "be": np.zeros(Hd), "Wf": _w(rng, dx, Hd), "bf": np.zeros(dx),
                  "Wk": _w(rng, dx, nr), "Wd": _w(rng, K, G), "bd": np.zeros(K),
                  "Wc": _w(rng, K, Hd), "bc": np.zeros(K), "Wo": _w(rng, dx, K) * 0.5})
    else:
        A, M = c["att"], c["mlp"]
        P.update({"Wq": _w(rng, A, G + nr), "Wkk": _w(rng, A, dx), "W1": _w(rng, M, G + dx + nr),
                  "b1": np.zeros(M), "W2": _w(rng, dx, M) * 0.5, "b2": np.zeros(dx), "Wcp": 0.1 * _w(rng, dx, dx)})
    return {"kind": kind, "params": P, "ko": {}, "mutant": None, "tau": 1.0,
            "lam": c["ledger_lambda"] if kind == "ledger" else 0.0, "dims": (dh, dx, nr), "ref": None}


def as_batch(model, X):
    if isinstance(X, dict):
        return X
    dh, dx, nr = model["dims"]
    B = X.shape[0]
    return {"h": np.asarray(X, float), "XM": np.zeros((B, 1, dx)), "rk": np.zeros((B, nr))}


def _ko(model, name, val):
    if name not in model["ko"]:
        return val
    mode, ref = model["ko"][name]
    if mode == "zero":
        return np.zeros_like(val)
    if mode == "identity":
        return np.ones_like(val) if name == "javab_gain" else val
    return np.broadcast_to(ref, val.shape).copy()


def _hal(model, h):
    P = model["params"]
    g1 = np.tanh(h @ P["Wg"].T + P["bg"])
    r = _ko(model, "hal_second_thought", np.tanh(g1 @ P["Wr"].T + P["br"]))
    g = _ko(model, "hal_interpreter", g1 + r)
    return g1, r, g


def forward(model, B):
    P, h, XM, rk = model["params"], B["h"], B["XM"], B["rk"]
    g1, r, g = _hal(model, h)
    if model["kind"] == "kamal":
        U = np.tanh(XM @ P["We"].T + P["be"])
        pooled = U.mean(axis=1) if model.get("pool", "mean") == "mean" else U[:, 0]   # "first" = broken control (C6.1)
        zeta = _ko(model, "zamin_pool", pooled)                      # M1: only the set mean leaves the model ghazal
        d = g @ P["Wd"].T + P["bd"]                                  # drive from the answerer's own state
        t = np.tanh(zeta @ P["Wc"].T + P["bc"])
        gain = _ko(model, "javab_gain", 1.0 + t)                     # M2: context rescales, in (0, 2)
        a = d * gain
        form = _ko(model, "form_realizer", zeta @ P["Wf"].T + P["bf"])
        yhat = form + rk @ P["Wk"].T + a @ P["Wo"].T
        return yhat, dict(U=U, zeta=zeta, g1=g1, r=r, g=g, d=d, t=t, gain=gain, a=a)
    qin = np.concatenate([g, rk], axis=1)
    q = qin @ P["Wq"].T
    Kk = XM @ P["Wkk"].T
    s = np.einsum("bp,bnp->bn", q, Kk) / math.sqrt(q.shape[1])
    alpha = softmax(s, axis=1)
    v = np.einsum("bn,bnx->bx", alpha, XM)                           # retrieved couplet (copy channel)
    hin = np.concatenate([g, v, rk], axis=1)
    hh = np.tanh(hin @ P["W1"].T + P["b1"])
    yhat = hh @ P["W2"].T + P["b2"] + v @ P["Wcp"].T
    return yhat, dict(g1=g1, r=r, g=g, qin=qin, q=q, Kk=Kk, alpha=alpha, v=v, hin=hin, hh=hh)


def _ledger(model, yhat, XM):
    D = np.sum((yhat[:, None, :] - XM) ** 2, axis=2)
    E = np.exp(-D / model["tau"])
    pen = model["lam"] * float(np.mean(np.sum(E, axis=1)))
    dy = model["lam"] / yhat.shape[0] * np.sum((E * (-2.0 / model["tau"]))[:, :, None] * (yhat[:, None, :] - XM), axis=1)
    return pen, dy


def objective(model, batch):
    yhat = forward(model, batch)[0]
    loss = float(np.mean((yhat - batch["y"]) ** 2))
    return loss + (_ledger(model, yhat, batch["XM"])[0] if model["lam"] > 0 else 0.0)


def _hal_backward(P, C, h, dg, grads):
    dpr = dg * (1 - C["r"] ** 2)
    grads["Wr"] = dpr.T @ C["g1"]; grads["br"] = dpr.sum(0)
    dg1 = dg + dpr @ P["Wr"]
    dpg = dg1 * (1 - C["g1"] ** 2)
    grads["Wg"] = dpg.T @ h; grads["bg"] = dpg.sum(0)


def loss_and_grads(model, batch):
    if model["ko"]:
        raise ValueError("knocked-out models are evaluated, not trained")
    P = model["params"]
    yhat, C = forward(model, batch)
    y, XM, rk, h = batch["y"], batch["XM"], batch["rk"], batch["h"]
    err = yhat - y
    loss = float(np.mean(err ** 2))
    dY = 2.0 * err / err.size
    if model["lam"] > 0:
        pen, dpen = _ledger(model, yhat, XM)
        loss += pen; dY = dY + dpen
    gr = {}
    if model["kind"] == "kamal":
        gr["Wo"] = dY.T @ C["a"]
        da = dY @ P["Wo"]
        gr["Wf"] = dY.T @ C["zeta"]; gr["bf"] = dY.sum(0); gr["Wk"] = dY.T @ rk
        dd = da * C["gain"]
        dc = da * C["d"] * (1 - C["t"] ** 2)
        gr["Wc"] = dc.T @ C["zeta"]; gr["bc"] = dc.sum(0)
        dzeta = dY @ P["Wf"] + dc @ P["Wc"]
        gr["Wd"] = dd.T @ C["g"]; gr["bd"] = dd.sum(0)
        dg = dd @ P["Wd"]
        dpU = (dzeta[:, None, :] / XM.shape[1]) * (1 - C["U"] ** 2)
        gr["We"] = np.einsum("bnh,bnx->hx", dpU, XM); gr["be"] = dpU.sum((0, 1))
    else:
        gr["W2"] = dY.T @ C["hh"]; gr["b2"] = dY.sum(0); gr["Wcp"] = dY.T @ C["v"]
        dph = (dY @ P["W2"]) * (1 - C["hh"] ** 2)
        gr["W1"] = dph.T @ C["hin"]; gr["b1"] = dph.sum(0)
        dhin = dph @ P["W1"]
        G = C["g"].shape[1]
        dg = dhin[:, :G].copy()
        dv = dhin[:, G:G + XM.shape[2]] + dY @ P["Wcp"]
        dal = np.einsum("bx,bnx->bn", dv, XM)
        ds = C["alpha"] * (dal - np.sum(C["alpha"] * dal, axis=1, keepdims=True)) / math.sqrt(C["q"].shape[1])
        dq = np.einsum("bn,bnp->bp", ds, C["Kk"])
        dK = ds[:, :, None] * C["q"][:, None, :]
        gr["Wkk"] = np.einsum("bnp,bnx->px", dK, XM)
        gr["Wq"] = dq.T @ C["qin"]
        dg = dg + (dq @ P["Wq"])[:, :G]
    _hal_backward(P, C, h, dg, gr)
    if model["mutant"] in ("drop_grad_Wc", "drop_grad_Wg"):
        key = model["mutant"].split("_")[-1]
        gr[key] = np.zeros_like(gr[key])
    return loss, gr


def fit(model, data, budget, rng, clip=CFG["clip"], lr=CFG["lr"]):
    """Minibatch Adam; model selection on the validation split only."""
    tr, va = data["train"], data["val"]
    if model["lam"] > 0:
        D = np.sum((tr["y"][:, None, :] - tr["XM"]) ** 2, axis=2)
        model["tau"] = CFG["ledger_tau_frac"] * float(np.median(D))
    lr_eff = 0.0 if model["mutant"] == "zero_lr" else lr
    sign = -1.0 if model["mutant"] == "sign_flip" else 1.0
    opt = Adam(model["params"], lr=lr_eff)
    n = tr["y"].shape[0]
    best, best_val, hist = None, np.inf, []
    for step in range(budget):
        idx = rng.integers(0, n, CFG["batch"])
        loss, gr = loss_and_grads(model, take(tr, idx))
        clip_global(gr, clip)
        opt.step(model["params"], gr, sign=sign)
        hist.append(loss)
        if not math.isfinite(loss):
            raise NonFinite("non-finite training loss")
        if (step + 1) % 100 == 0 or step == budget - 1:
            vl = objective(model, va)
            if vl < best_val:
                best_val, best = vl, {k: v.copy() for k, v in model["params"].items()}
    if best is not None:
        model["params"] = best
    model["ref"] = take(tr, np.arange(min(512, n)))
    for v in model["params"].values():
        if not np.all(np.isfinite(v)):
            raise NonFinite("non-finite parameter")
    return hist


def predict(model, X):
    return forward(model, as_batch(model, X))[0]


def hidden_states(model, X):
    C = forward(model, as_batch(model, X))[1]
    keys = ("zeta", "g", "d", "gain", "a") if model["kind"] == "kamal" else ("g", "alpha", "v", "hh")
    return {k: C[k] for k in keys}


def modules(model):
    if model["kind"] == "kamal":
        spec = [("zamin_pool", ["We", "be"], "permutation-invariant mean of encoded exemplar items", True),
                ("javab_gain", ["Wc", "bc"], "bounded multiplicative context gain (0, 2) on a self-driven drive", True),
                ("form_realizer", ["Wf", "bf"], "linear readout of the pooled exemplar code (inherited form)", False),
                ("rhyme_landing", ["Wk"], "linear readout of the target-slot one-hot", False),
                ("hal_interpreter", ["Wg", "bg"], "tanh encoder of the own-state input", False),
                ("hal_second_thought", ["Wr", "br"], "residual tanh refinement of the own-state code", False),
                ("drive", ["Wd", "bd", "Wo"], "linear drive and output map of the gated units", False)]
    else:
        spec = [("hal_interpreter", ["Wg", "bg"], "tanh encoder of the own-state input", False),
                ("hal_second_thought", ["Wr", "br"], "residual tanh refinement of the own-state code", False),
                ("exemplar_attention", ["Wq", "Wkk"], "softmax attention over exemplar items keyed by state and slot", False),
                ("decoder", ["W1", "b1", "W2", "b2", "Wcp"], "tanh MLP plus direct copy of the retrieved item", False)]
    return {n: {"params": p, "role": r, "signature": s} for n, p, r, s in spec}


def knockout(model, name, mode):
    """Copy with a module's output replaced: 'zero', 'identity' or 'mean' (mean over the stored training reference)."""
    m = {k: v for k, v in model.items()}
    m["params"] = {k: v.copy() for k, v in model["params"].items()}
    m["ko"] = dict(model["ko"])
    C = forward(model, model["ref"])[1] if model["ref"] is not None else None
    key = {"zamin_pool": "zeta", "javab_gain": "gain", "hal_second_thought": "r", "hal_interpreter": "g"}.get(name)
    if name == "form_realizer":
        ref = (C["zeta"] @ model["params"]["Wf"].T + model["params"]["bf"]).mean(0)
    else:
        ref = C[key].mean(0) if (C is not None and key) else 0.0
    m["ko"][name] = (mode, ref)
    return m


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


MUTANTS = {"sign_flip": "update direction reversed (gradient ascent)",
           "zero_lr": "learning rate set to zero",
           "drop_grad_Wc": "gradient of the javab gain weights zeroed",
           "drop_grad_Wg": "gradient of the hal encoder weights zeroed"}


# ----------------------------------------------------------------------------- metrics
def skill(model, split, ybar):
    yh = predict(model, split)
    return 1.0 - float(np.mean((yh - split["y"]) ** 2)) / float(np.mean((ybar - split["y"]) ** 2))


def theft(model, split):
    """Regression of the answer's error on the counterpart couplet's meaning share (meaningful where rho = 0):
    0 for an answer that owes the counterpart nothing, 1 for a copier of the counterpart's meaning."""
    r, cdir = predict(model, split) - split["y"], split["yc"]
    return float(np.sum(r * cdir) / np.sum(cdir * cdir))


# ----------------------------------------------------------------------------- baselines and rivals
# The exemplar-attention decoder ("attn") is the closest standard alternative: it reads the model ghazal by
# softmax attention keyed on the answerer's state and target slot and may copy the retrieved couplet directly.
# The ledger rival ("ledger", after chapter 0307) is the same decoder with an anti-repetition penalty
# lam * sum_i exp(-||yhat - x_i||^2 / tau) against every couplet already in the model ghazal.
KINDS = ("kamal", "attn", "ledger")


def child_rng(*key):
    return np.random.default_rng(np.random.SeedSequence([CHAPTER] + list(key)))


def train_kind(kind, data, seed, steps, mutant=None):
    c = CFG
    m = build_model(c["d_h"], c["d_x"], "vector_regression", child_rng(seed, 1, KINDS.index(kind)), kind=kind)
    m["mutant"] = mutant
    hist = fit(m, data, steps, child_rng(seed, 2, KINDS.index(kind)))
    return m, hist


# ----------------------------------------------------------------------------- correctness tests
def c1_gradcheck(data, seed, steps_after=60, mutant=None, kinds=KINDS):
    tr = data["train"]
    b = take(tr, np.arange(16))
    out, worst, ok_all, checked, total = {}, 0.0, True, 0, 0
    for kind in kinds:
        for when in ("init", "after_training_steps"):
            m = build_model(CFG["d_h"], CFG["d_x"], "vector_regression", child_rng(seed, 3, KINDS.index(kind)), kind=kind)
            m["mutant"] = mutant
            if when == "init" and m["lam"] > 0:
                m["tau"] = CFG["ledger_tau_frac"] * float(np.median(np.sum((tr["y"][:, None] - tr["XM"]) ** 2, axis=2)))
            if when != "init":
                fit(m, data, steps_after, child_rng(seed, 4, KINDS.index(kind)))
                m["mutant"] = mutant
            _, gr = loss_and_grads(m, b)
            ok, w, rep = grad_check(lambda: objective(m, b), m["params"], gr, child_rng(seed, 5, KINDS.index(kind)))
            out[kind + "@" + when] = ok
            ok_all &= ok
            worst = max(worst, w)
            checked += sum(1 for v in rep.values() if v); total += len(rep)
    return ok_all, worst, checked, total, out


def c3_learning(hist, sk_iid):
    first, last = float(np.mean(hist[:20])), float(np.mean(hist[-50:]))
    drop = 1.0 - last / first
    return drop >= MIND_CARD["thresholds"]["loss_drop_fraction"] and sk_iid >= MIND_CARD["thresholds"]["margin_over_trivial"], drop


def c4_shuffled(data, seed, steps):
    rng = child_rng(seed, 6)
    sh = dict(data)
    for part in ("train", "val"):
        d = dict(data[part])
        d["y"] = d["y"][rng.permutation(d["y"].shape[0])]
        sh[part] = d
    m, _ = train_kind("kamal", sh, seed + 1000, steps)
    sk = skill(m, data["iid"], data["ybar"])
    return sk <= MIND_CARD["thresholds"]["shuffle_band"], sk


def c5_mutants(data, seed, steps):
    detected = {}
    for name in MUTANTS:
        try:
            g_ok = c1_gradcheck(data, seed, steps_after=0, mutant=name, kinds=("kamal",))[0] if name.startswith("drop") else True
            m, hist = train_kind("kamal", data, seed, steps, mutant=name)
            l_ok = c3_learning(hist, skill(m, data["iid"], data["ybar"]))[0]
            detected[name] = not (g_ok and l_ok)
        except NonFinite:
            detected[name] = True
    return all(detected.values()), detected


def _breach_trial(model, rng, n=7, B=24):
    P = model["params"]
    dx, nr, dh = CFG["d_x"], CFG["n_r"], CFG["d_h"]
    XM = rng.normal(0, rng.uniform(0.5, 3.0), (B, n, dx))
    batch = {"XM": XM, "rk": np.eye(nr)[rng.integers(0, nr, B)], "h": rng.normal(0, 2, (B, dh))}
    j = 0
    alt = dict(batch); alt["XM"] = XM.copy()
    alt["XM"][:, j] = rng.normal(0, rng.uniform(0.5, 3.0), (B, dx))
    y0, C = forward(model, batch)
    y1 = forward(model, alt)[0]
    sn = lambda A: float(np.linalg.norm(A, 2))
    dmax = np.max(np.abs(C["d"]), axis=1)
    bound = (sn(P["Wf"]) + sn(P["Wo"]) * dmax * sn(P["Wc"])) * sn(P["We"]) * np.linalg.norm(alt["XM"][:, j] - XM[:, j], axis=1) / n
    return float(np.max(np.linalg.norm(y1 - y0, axis=1) / bound))


def c6_properties(seed, trials=150):
    """C6.1 breach bound: changing one of n model couplets moves the answer by at most a Lipschitz bound / n.
    Searched over random parameter scales and inputs; negative control: first-item pooling must violate it."""
    rng = child_rng(seed, 7)
    worst, worst_ctrl = 0.0, 0.0
    for t in range(trials):
        m = build_model(CFG["d_h"], CFG["d_x"], "vector_regression", child_rng(seed, 8, t), kind="kamal")
        for k in m["params"]:
            m["params"][k] *= rng.uniform(0.3, 3.0)
        worst = max(worst, _breach_trial(m, rng))
        m["pool"] = "first"
        worst_ctrl = max(worst_ctrl, _breach_trial(m, rng))
    c61 = worst <= 1.0 + 1e-9 and worst_ctrl > 1.0
    # C6.2 definition check (not evidence): a javab unit cannot reverse or originate its drive; additive control violates.
    d, cc = rng.normal(0, 3, 20000), rng.normal(0, 3, 20000)
    a = d * (1 + np.tanh(cc))
    unit_ok = bool(np.all(np.sign(a) == np.sign(d)) and np.all(np.abs(a) <= 2 * np.abs(d) + 1e-12))
    add = d + np.tanh(cc)
    ctrl_bad = bool(np.any(np.sign(add) != np.sign(d)))
    return c61, worst, worst_ctrl, unit_ok and ctrl_bad


def c7_split_integrity(data):
    seen = {}
    for part in ("train", "val", "iid", "shifted", "blind"):
        for hsh in data[part]["hashes"]:
            seen.setdefault(hsh, set()).add(part)
    return all(len(v) == 1 for v in seen.values())


# ----------------------------------------------------------------------------- diagnostics
def lineage(model, W, ybar, rng, n_g=60, gens=CFG["lineage_generations"]):
    """Answers to answers: generation g's model ghazal is generation g-1's output. Fresh hal each generation."""
    c = CFG
    nr, dz, dm, dx = c["n_r"], c["d_z"], c["d_m"], c["d_x"]
    z = rng.normal(size=(n_g, dz))
    zb = np.broadcast_to(z[:, None, :], (n_g, nr, dz))
    s0 = np.array([rng.permutation(nr) for _ in range(n_g)])
    m0 = rng.normal(size=(n_g, nr, dm))
    X = realize(W, zb, s0, m0) + c["sig_x"] * rng.normal(size=(n_g, nr, dx))
    inv0 = np.argsort(s0, axis=1)
    skills, drift = [], []
    for _ in range(gens):
        sA = np.array([rng.permutation(nr) for _ in range(n_g)])
        mA = rng.normal(size=(n_g, nr, dm))
        h = np.tanh(mA @ W["Q"].T + W["q"]) + c["sig_h"] * rng.normal(size=(n_g, nr, c["d_h"]))
        batch = {"XM": np.repeat(X, nr, axis=0), "rk": np.eye(nr)[sA.reshape(-1)], "h": h.reshape(-1, c["d_h"])}
        y = realize(W, zb, sA, mA).reshape(-1, dx)
        yh = predict(model, batch)
        skills.append(1 - float(np.mean((yh - y) ** 2)) / float(np.mean((ybar - y) ** 2)))
        m0c = np.take_along_axis(m0, inv0[np.arange(n_g)[:, None], sA][..., None], axis=1)
        y0 = (realize(W, zb, sA, m0c) - realize(W, zb, sA, np.zeros_like(m0c))).reshape(-1, dx)
        drift.append(float(np.sum((yh - y) * y0) / np.sum(y0 * y0)))
        X = yh.reshape(n_g, nr, dx) + c["sig_x"] * rng.normal(size=(n_g, nr, dx))
    return skills, drift


def seven_curve(model, W, rng, ns=(2, 4, 7, 14)):
    return {n: theft(model, make_split(W, rng, 80, 0.0, CFG["sig_h"], n_c=n)) for n in ns}


# ----------------------------------------------------------------------------- one seed
def run_seed(seed, steps, full_diag=True):
    data = build_data(seed)
    res = {"seed": seed, "models": {}, "hist": {}}
    for kind in KINDS:
        m, hist = train_kind(kind, data, seed, steps)
        res["models"][kind], res["hist"][kind] = m, hist
        res.setdefault("skill", {})[kind] = {s: skill(m, data[s], data["ybar"]) for s in ("iid", "shifted", "blind")}
        res.setdefault("theft", {})[kind] = theft(m, data["shifted"])
    k = res["models"]["kamal"]
    base = res["skill"]["kamal"]["shifted"]
    kos = [("zamin_pool", "mean"), ("javab_gain", "mean"), ("form_realizer", "mean"),
           ("hal_interpreter", "mean"), ("hal_second_thought", "zero")]
    res["ko"] = {n: skill(knockout(k, n, md), data["shifted"], data["ybar"]) - base for n, md in kos}
    res["ko_abs"] = {n: base + v for n, v in res["ko"].items()}
    if full_diag:
        res["lineage"] = {kd: lineage(res["models"][kd], data["world"], data["ybar"], child_rng(seed, 9)) for kd in ("kamal", "attn")}
        res["seven"] = {kd: seven_curve(res["models"][kd], data["world"], child_rng(seed, 10)) for kd in ("kamal", "attn")}
    res["data"] = data
    return res


def hypotheses(runs, rng):
    S = lambda kd, sp: np.array([r["skill"][kd][sp] for r in runs])
    diffs = {"H-SIG": S("kamal", "shifted") - S("attn", "shifted"),
             "H-NEC": np.array([r["ko_abs"]["javab_gain"] - r["ko_abs"]["hal_second_thought"] for r in runs]),
             "H-BLIND": S("kamal", "blind") - S("attn", "blind"),
             "H-RIVAL": S("kamal", "shifted") - S("ledger", "shifted")}
    out = []
    for h in MIND_CARD["hypotheses"]:
        mean, ci = bootstrap_ci(diffs[h["id"]], rng)
        out.append({"id": h["id"], "metric": h["metric"], "mean_diff": mean, "ci95": ci, "mesi": h["mesi"],
                    "n_seeds": len(runs), "verdict": verdict(mean, ci, h["direction"], h["mesi"]),
                    "per_seed": [float(x) for x in diffs[h["id"]]]})
    return out


# ----------------------------------------------------------------------------- report and entry point
def build_report(runs, seeds, steps, runtime, corr, mu_det, hyp, kost, g_chk, g_tot, g_worst, g_ok, all_ok):
    r0 = runs[0]
    lines = ["", "=== VERIFIED REPORT · chapter 0370 ===",
             "file: %s | card_revision %d" % (os.path.basename(__file__), MIND_CARD["card_revision"]),
             "environment: python %s, numpy %s" % (sys.version.split()[0], np.__version__),
             "seeds: %s | runtime %.1f s | steps per model %d" % (seeds, runtime, steps),
             "parameters: javab %d | exemplar-attention %d | ledger rival %d" % tuple(n_params(r0["models"][k]) for k in KINDS),
             "", "correctness:"]
    lines += ["  %-5s %-40s %s  %s" % (c[0], c[1], "PASS" if c[2] else "FAIL", c[3]) for c in corr]
    lines.append("  mutation score: %d/%d" % (sum(mu_det.values()), len(mu_det)))
    lines += ["", "hypotheses (paired over seeds, 95% bootstrap, 2000 resamples):"]
    for h in hyp:
        if h["mean_diff"] is None:
            lines.append("  %-8s not evaluated (--quick)" % h["id"])
        else:
            lines.append("  %-8s diff %+.3f  CI [%+.3f, %+.3f]  mesi %.2f  -> %s" % (h["id"], h["mean_diff"], h["ci95"][0], h["ci95"][1], h["mesi"], h["verdict"]))
    lines += ["", "knockouts on the javab network (change in shifted-split skill):"]
    lines += ["  %-20s sig=%-5s %+.3f  CI [%+.3f, %+.3f]" % (k["module"], k["signature"], k["metric_change"], k["ci95"][0], k["ci95"][1]) for k in kost]
    lines += ["", "skills (mean over seeds)            iid     shifted  blind    theft(shifted)"]
    for kd in KINDS:
        lines.append("  %-30s %+.3f  %+.3f   %+.3f   %+.3f" % (kd, *[float(np.mean([r["skill"][kd][s] for r in runs])) for s in ("iid", "shifted", "blind")],
                                                            float(np.mean([r["theft"][kd] for r in runs]))))
    lr = [r for r in runs if "lineage" in r]
    for kd in ("kamal", "attn"):
        sk = np.mean([r["lineage"][kd][0] for r in lr], axis=0)
        dr = np.mean([r["lineage"][kd][1] for r in lr], axis=0)
        lines.append("  lineage %-6s skill by generation %s | drift toward first model %s" % (kd, np.round(sk, 3).tolist(), np.round(dr, 3).tolist()))
    for kd in ("kamal", "attn"):
        sv = {n: float(np.mean([r["seven"][kd][n] for r in lr])) for n in lr[0]["seven"][kd]}
        lines.append("  theft vs couplets in model %-6s %s" % (kd, {n: round(v, 3) for n, v in sv.items()}))
    lines += ["  task types: %s | exit code %d" % (TASK_TYPES, 0 if all_ok else 1), "=== END REPORT ==="]
    payload = {"schema_version": "1.0", "chapter": CHAPTER, "file": os.path.basename(__file__),
               "card_revision": MIND_CARD["card_revision"],
               "environment": {"python": sys.version.split()[0], "numpy": np.__version__}, "seeds": seeds,
               "runtime_s": round(runtime, 2), "n_params": n_params(r0["models"]["kamal"]),
               "gradcheck": {"tensors_checked": g_chk, "tensors_total": g_tot, "max_rel_error": g_worst,
                             "checked_at": ["init", "after_training_steps"], "passed": g_ok},
               "correctness": [{"id": c[0], "name": c[1], "passed": bool(c[2]), "detail": c[3]} for c in corr],
               "mutants": {"detected": int(sum(mu_det.values())), "total": len(mu_det), "score": sum(mu_det.values()) / len(mu_det)},
               "hypotheses": [{k: v for k, v in h.items()} for h in hyp], "knockouts": kost, "task_types": TASK_TYPES,
               "diagnostics": {"skills": {kd: {s: float(np.mean([r["skill"][kd][s] for r in runs])) for s in ("iid", "shifted", "blind")} for kd in KINDS},
                               "theft_shifted": {kd: float(np.mean([r["theft"][kd] for r in runs])) for kd in KINDS},
                               "lineage": {kd: {"skill": np.mean([r["lineage"][kd][0] for r in lr], axis=0).tolist(),
                                                "drift_to_first": np.mean([r["lineage"][kd][1] for r in lr], axis=0).tolist()} for kd in ("kamal", "attn")}},
               "exit_code": 0 if all_ok else 1}
    return lines, payload


def protocol(args):
    t0 = time.time()
    quick = args.quick
    steps = CFG["quick_steps"] if quick else CFG["steps"]
    n_seeds = 1 if quick else args.seeds
    seeds = [args.seed + i for i in range(n_seeds)]
    print("seeds:", seeds, "| steps per model:", steps, "| mode:", "quick" if quick else "full", flush=True)
    runs = [run_seed(s, steps, full_diag=(i == 0 or not quick)) for i, s in enumerate(seeds)]
    r0, d0 = runs[0], runs[0]["data"]
    corr = []
    g_ok, g_worst, g_chk, g_tot, g_detail = c1_gradcheck(d0, seeds[0])
    corr.append(("C1", "gradient_check", g_ok, "tensors passing %d/%d (3 models x init/after 60 steps); max rel err %.2e" % (g_chk, g_tot, g_worst)))
    ma, ha = train_kind("kamal", d0, seeds[0], 40)
    mb, hb = train_kind("kamal", d0, seeds[0], 40)
    det = np.array_equal(np.array(ha), np.array(hb)) and np.array_equal(predict(ma, d0["iid"]), predict(mb, d0["iid"]))
    fin = all(np.all(np.isfinite(v)) for r in runs for m in r["models"].values() for v in m["params"].values())
    corr.append(("C2", "determinism_finiteness", bool(det and fin), "identical 40-step runs: %s; all finite: %s" % (det, fin)))
    lr_ok, lr_det = True, []
    for kd in KINDS:
        ok, drop = c3_learning(r0["hist"][kd], r0["skill"][kd]["iid"])
        lr_ok &= ok
        lr_det.append("%s drop %.2f iid skill %.3f" % (kd, drop, r0["skill"][kd]["iid"]))
    corr.append(("C3", "learning", lr_ok, "; ".join(lr_det)))
    sh_ok, sh_sk = c4_shuffled(d0, seeds[0], steps)
    corr.append(("C4", "shuffled_label_control", sh_ok, "shuffled-label iid skill %.3f (band <= %.2f)" % (sh_sk, MIND_CARD["thresholds"]["shuffle_band"])))
    mu_ok, mu_det = c5_mutants(d0, seeds[0], min(steps, 400))
    corr.append(("C5", "mutant_detection", mu_ok, ", ".join("%s:%s" % (k, "caught" if v else "MISSED") for k, v in mu_det.items())))
    c61, wb, wc, c62 = c6_properties(seeds[0], trials=60 if quick else 150)
    corr.append(("C6.1", "breach_bound_1_over_n", c61, "max ratio to bound %.3f (<= 1); first-item control max ratio %.2f (> 1)" % (wb, wc)))
    corr.append(("C6.2", "definition_check_gain_cannot_originate", c62, "definition check, not evidence; additive control violates"))
    corr.append(("C7", "split_integrity", c7_split_integrity(d0), "no ghazal shared between train/val/iid/shifted/blind"))
    hyp = hypotheses(runs, child_rng(999)) if not quick else [
        {"id": h["id"], "metric": h["metric"], "mean_diff": None, "ci95": None, "mesi": h["mesi"], "n_seeds": 1,
         "verdict": "not evaluated", "per_seed": []} for h in MIND_CARD["hypotheses"]]
    kost = []
    mods = modules(r0["models"]["kamal"])
    for n in r0["ko"]:
        mean, ci = bootstrap_ci([r["ko"][n] for r in runs], child_rng(998)) if len(runs) > 1 else (r0["ko"][n], [r0["ko"][n]] * 2)
        kost.append({"module": n, "signature": mods[n]["signature"], "metric_change": mean, "ci95": ci})
    runtime = time.time() - t0
    budget = BUDGET_QUICK if quick else BUDGET_FULL
    corr.append(("C8", "budget", runtime <= budget, "%.1f s of %.0f s" % (runtime, budget)))
    all_ok = all(c[2] for c in corr)
    lines, payload = build_report(runs, seeds, steps, runtime, corr, mu_det, hyp, kost, g_chk, g_tot, g_worst, g_ok, all_ok)
    if runtime > budget:
        payload["exit_code"] = 3
    write_report(lines, payload, args.json)
    return payload["exit_code"]


def data_bridge(path):
    if not os.path.exists(path):
        print("real-data bridge: file not found:", path)
        return 2
    arr = np.genfromtxt(path, delimiter=",", names=True)
    xs = [n for n in arr.dtype.names if n.startswith("x")]
    ys = [n for n in arr.dtype.names if n.startswith("y")]
    X = np.stack([arr[n] for n in xs], axis=1).astype(float)
    Y = np.stack([arr[n] for n in ys], axis=1).astype(float)
    rng = child_rng(77)
    idx = rng.permutation(len(X)); cut = int(0.8 * len(X))
    m = build_model(X.shape[1], Y.shape[1], "vector_regression", rng)
    mk = lambda I: {"h": X[I], "XM": np.zeros((len(I), 1, Y.shape[1])), "rk": np.zeros((len(I), m["dims"][2])), "y": Y[I]}
    fit(m, {"train": mk(idx[:cut]), "val": mk(idx[cut:])}, 600, rng)
    ybar = Y[idx[:cut]].mean(0)
    print("real-data bridge: held-out skill %.3f on %s" % (skill(m, mk(idx[cut:]), ybar), path))
    return 0


def main():
    ap = argparse.ArgumentParser(description="Chapter 0370 javab network")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--card", action="store_true")
    ap.add_argument("--mutant", type=str, default=None)
    ap.add_argument("--data", type=str, default=None)
    try:
        args = ap.parse_args()
    except SystemExit:
        return 2
    if args.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    try:
        if args.mutant:
            if args.mutant not in MUTANTS:
                print("unknown mutant; choose from", list(MUTANTS), "| usage: python3", os.path.basename(__file__), "--mutant NAME")
                return 2
            d = build_data(args.seed)
            g_ok = c1_gradcheck(d, args.seed, steps_after=0, mutant=args.mutant, kinds=("kamal",))[0]
            m, hist = train_kind("kamal", d, args.seed, 400, mutant=args.mutant)
            l_ok = c3_learning(hist, skill(m, d["iid"], d["ybar"]))[0]
            print("mutant %s: C1 %s, C3 %s" % (args.mutant, "pass" if g_ok else "FAIL", "pass" if l_ok else "FAIL"))
            return 0 if (g_ok and l_ok) else 1
        code = protocol(args)
        if args.data:
            data_bridge(args.data)
        else:
            print("real-data bridge: skipped (no --data given)")
        return code
    except NonFinite as e:
        print("non-finite values:", e)
        return 4


if __name__ == "__main__":
    sys.exit(main())

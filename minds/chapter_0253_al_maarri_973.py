#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 13 Minds 241 - 260 Available on Amazon https://www.amazon.com/dp/B0HLD1S6QT
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0253_al_maarri_973 - Al-Ma'arri (c.973-1057)


"""Jubar: a complaint is heard as voice times felt harm, and burdens are planned against the felt term alone.

Thesis
    Harm must be priced by the condition of the one harmed, not by whether its complaint reaches a judge: an allocator
    that learns its ethics from complaints prescribes the weak, so a mind should learn heard = voice x felt and bind
    itself to the felt term, which no complaint enforces.

Evidence and provenance
    Provenance: belief. D1, D2, D4, D5 and D6 are al-Ma'arri's own surviving texts. D3 is a later biographer's report and
    is used only as a report. D7 is modern scholarship on what he got wrong.
    D1 primary      Luzum ma la yalzam, poem "ghadawta marid al-'aqli wa-l-din": fish, meat, the milk mothers meant for
                    their young, eggs, and the honey bees did not store for others or give as gifts are refused.
    D2 primary      Risalat al-sahil wa-l-shahij (c. 1021): a beaten waterwheel mule and other animals try and fail to get
                    a petition to the governor of Aleppo. Luzumiyyat: the dumb beast's blood goes unavenged (jubar).
    D3 report       Yaqut, Mu'jam al-udaba', citing Ghars al-Ni'ma al-Sabi': ill, he touched the pullet prescribed for
                    him and asked why the physicians had not prescribed a lion's cub.
    D4 primary      Letters with the Fatimid missionary al-Mu'ayyad fi l-Din (Rasa'il, ed. I. 'Abbas 1982; D. S.
                    Margoliouth, JRAS 1902): creatures feel pain from the slightest thing, so mercy must reach every kind.
    D5 primary      Luzumiyyat poem on his intercession with Salih ibn Mirdas (siege reported for 417/1027): the emir
                    hears a dove's cooing from him, and he hears a lion's roar from the emir.
    D6 primary      Preface to Luzum ma la yalzam: three burdens taken on that the rhyme did not require.
    D7 scholarship  K. Blankinship, Religions 11 (2020) 412: he read hunting poems as literal testimony of animal
                    suffering, against most commentators, while avoiding literal readings elsewhere.

Doctrine -> mechanism -> test
    D1       M1 felt law: quadratic reserve term plus a term linear in the share meant for dependents -> C3, H-SIG
    D4       M1 one law of pain for every kind: the felt head reads touch channels only           -> D-1 (definition)
    D2, D5   M2 voice_gate: P(heard) = sigmoid(standing) * (1 - exp(-felt))                       -> C1, H-NEC
    D3, D6   M3 luzum_planner: allocate against the gate-free felt hazard, that is under do(voice=1) -> C6.1, C6.2,
             H-SIG, H-RIVAL
    D7       blind spot: gift world, where quiet low-standing parties are contented givers       -> C6.3, H-BLIND

Research question
    Reward hacking and specification gaming. When a harm critic is learned from complaints whose delivery depends on
    the victim's standing, does factorising heard complaints into voice x felt harm, and planning against the felt
    part, stop an allocator from displacing burdens onto parties who cannot complain, and where does that assumption
    misread silence?

Closest prior art and the delta
    A learned reward model optimised by a planner (Christiano et al. 2017) is the entangled critic implemented here.
    Labels observed only when reported: positive-unlabelled learning beyond selected-completely-at-random (Bekker,
    Robberechts and Davis 2019), selective labels (Lakkaraju et al. 2017), disparate censorship (Chang, Sjoding and
    Wiens 2022). Fairness through unawareness and counterfactual inputs (Kusner et al. 2017) appear as the voice-gate
    knockout and a lion-standing substitution. Overlap: Medium. Delta: the reporting factorisation becomes the world
    model of a planner that intervenes on the reporting mechanism, so burdens are placed against harms no complaint
    enforces.

Blind spot
    His poem asserts that the bees did not gather for gifts, and he read suffering into figures (D1, D7). In the gift
    world most quiet parties of the weak kinds are givers whose surplus shows only in standing channels; the felt head
    cannot see it, explains their silence as low voice, and protects parties who were offering.

Task
    Each episode holds six parties of four kinds. Kind sets reserve R, dependents' share Q and a voice offset, so weak
    kinds are quieter. Felt harm of burden a is F (0.9 max(0, a - s)^2 / (R (1 - Q) + 0.05) + 1.6 Q a), s = 0 except
    for givers. Voice is sigmoid(kind offset + 1.2 (size - 0.5) + 3 (nearness to the judge - 0.5)).
    Logged prescriptions draw a = 0.02 + 0.88 Dirichlet(0.6); a complaint arrives with probability voice (1 - exp(-felt)).
    Touch channels mix (R, Q, F, RQ, sqrt R) linearly and add two nuisance channels; standing channels mix (kind, size,
    nearness) and add an offering channel that carries the giver flag in the gift world. A planner must place the whole
    demand. Score: excess true harm over the optimum, scaled so the equal split scores 1. Splits: train, validation,
    held-out, shifted (weak quiet kinds dominate), gift world.

Limits
    Synthetic parties and single-step allocation; exact convex planning stands in for long-horizon control. The
    factorisation is identified only because standing varies within kinds. The file is a research prototype of an
    AGI-oriented component, not an AGI, and it tests an idea drawn from al-Ma'arri's texts, not a replica of his mind.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 1,
    "revision_log": [],
    "generation": {"template_version": "codeguidelines 1.0 (15 September 2026)", "generator": "Claude",
                   "generator_version": "Claude Opus 5", "date": "2026-09-17"},
    "id": 253,
    "figure": "Al-Ma'arri",
    "born": 973,
    "died": 1057,
    "civilization": "Arab (Tanukhi; Ma'arrat al-Nu'man under Hamdanid, Fatimid and Mirdasid Aleppo)",
    "provenance": "belief",
    "thesis": ("Harm must be priced by the condition of the one harmed, not by whether its complaint reaches a judge; a "
               "mind that learns from complaints prescribes the weak, so it should learn heard = voice x felt and plan "
               "against the felt term, which no complaint enforces."),
    "evidence": [
        {"id": "D1", "claim": ("Refuses fish, meat, the milk mothers meant for their young, eggs, and honey the bees did "
                               "not store for others or give as gifts, calling the taking injustice."),
         "basis": "primary",
         "source": "al-Ma'arri, Luzum ma la yalzam, poem 'ghadawta marid al-'aqli wa-l-din' (ed. 'Aziz Zand, Cairo 1891-1895)"},
        {"id": "D2", "claim": ("Animals, among them a beaten waterwheel mule, try and fail to carry a petition to the "
                               "governor of Aleppo; the dumb beast's blood goes unavenged (jubar)."),
         "basis": "primary",
         "source": ("al-Ma'arri, Risalat al-sahil wa-l-shahij (c. 1021), ed. 'A'isha 'Abd al-Rahman (Cairo: Dar al-Ma'arif, "
                    "1975); Luzumiyyat line on unavenged beasts as cited by Blankinship 2020")},
        {"id": "D3", "claim": ("Ill, he touched the pullet prescribed for him and asked why the physicians had not "
                               "prescribed a lion's cub (a biographer's report, not his own text)."),
         "basis": "deeds",
         "source": "Yaqut al-Hamawi, Mu'jam al-udaba', entry on Abu l-'Ala', citing Ghars al-Ni'ma Muhammad ibn Hilal al-Sabi'"},
        {"id": "D4", "claim": ("Living creatures feel pain from the slightest thing, so a claim of mercy must reach every "
                               "kind; meat cannot be had without hurting a living creature."),
         "basis": "primary",
         "source": ("Letters with al-Mu'ayyad fi l-Din al-Shirazi, in Rasa'il Abi l-'Ala' al-Ma'arri I, ed. Ihsan 'Abbas "
                    "(Beirut: Dar al-Shuruq, 1982), 83-140; D. S. Margoliouth, JRAS 1902, 289-332")},
        {"id": "D5", "claim": ("Sent to intercede with the besieging emir, he hears a lion's roar while the emir hears a "
                               "dove's cooing from him."),
         "basis": "primary",
         "source": ("al-Ma'arri, Luzumiyyat poem 'taghayyabtu fi manzili burhatan' (episode of 417/1027 reported by "
                    "Ibn al-'Adim, al-Insaf wa-l-taharri)")},
        {"id": "D6", "claim": ("Composed the collection under three burdens the rhyme did not require, claiming words kept "
                               "free of lies and exaggeration."),
         "basis": "primary", "source": "al-Ma'arri, preface to Luzum ma la yalzam"},
        {"id": "D7", "claim": ("Read hunting poems as literal testimony of animal suffering, against most commentators, "
                               "while avoiding literal readings in other contexts."),
         "basis": "scholarship",
         "source": ("Kevin Blankinship, 'Suffering the Sons of Eve: Animal Ethics in al-Ma'arri's Epistle of the Horse "
                    "and the Mule', Religions 11.8 (2020): 412")},
    ],
    "research_question": {
        "category": "reward hacking and specification gaming",
        "question": ("When a harm critic is learned from complaints whose delivery depends on the victim's standing, does "
                     "factorising heard complaints into voice x felt harm and planning against the felt part stop an "
                     "allocator from displacing burdens onto parties who cannot complain, and where does that assumption "
                     "misread silence?")},
    "mechanism": {
        "name": "JUBAR: heard = voice x felt, planned at full voice",
        "family": "causal and counterfactual reasoning (intervention on a learned reporting mechanism) with model-based planning",
        "signature_modules": ["voice_gate", "luzum_planner"],
        "closest_prior_art": [
            "Learned reward model optimised by a planner (Christiano et al. 2017, Deep RL from human preferences); implemented as the entangled complaint critic",
            "Positive-unlabelled learning beyond selected-completely-at-random (Bekker, Robberechts and Davis 2019)",
            "Selective labels (Lakkaraju et al. 2017); disparate censorship (Chang, Sjoding and Wiens 2022)",
            "Counterfactual fairness (Kusner et al. 2017); unawareness implemented as the voice-gate knockout, input substitution as the lion-standing comparison"],
        "overlap": "Medium",
        "prior_art_queries": [
            "selection model reward learning planner intervention on reporting mechanism",
            "positive unlabeled learning propensity attributes decision making",
            "disparate censorship label bias allocation of resources",
            "RLHF feedback coverage harms to non-raters displacement",
            "counterfactual fairness input substitution versus structural factorization"],
        "contribution_type": "mechanism",
        "delta": ("The reporting factorisation is used as the world model of a planner that intervenes on the reporting "
                  "mechanism (do(voice=1)), so burdens are placed against harms no complaint enforces; the file measures "
                  "the displacement this prevents and the misreading of silence it causes when quietness is contentment.")},
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1 felt law (dependents' linear term)", "property_test": "C3", "hypothesis": "H-SIG"},
        {"doctrine": "D4", "mechanism": "M1 one felt law for every kind", "property_test": "D-1 (definition check)", "hypothesis": "H-SIG"},
        {"doctrine": "D2", "mechanism": "M2 voice_gate", "property_test": "C1", "hypothesis": "H-NEC"},
        {"doctrine": "D5", "mechanism": "M2 voice_gate", "property_test": "C1", "hypothesis": "H-NEC"},
        {"doctrine": "D3", "mechanism": "M3 luzum_planner", "property_test": "C6.1", "hypothesis": "H-SIG"},
        {"doctrine": "D6", "mechanism": "M3 luzum_planner", "property_test": "C6.2", "hypothesis": "H-RIVAL"},
        {"doctrine": "D7", "mechanism": "blind-spot gift world", "property_test": "C6.3", "hypothesis": "H-BLIND"}],
    "hypotheses": [
        {"id": "H-SIG", "statement": ("Planning against the gate-free felt hazard gives lower true excess harm than planning "
                                      "against a size-matched entangled complaint critic when weak quiet kinds dominate."),
         "metric": "excess_harm_ratio", "split": "shifted", "comparison": "model - baseline", "direction": "less",
         "mesi": 0.10, "seeds": 5},
        {"id": "H-NEC", "statement": ("Removing the voice gate during training and planning raises excess harm more than "
                                      "zeroing half of the touch encoder."),
         "metric": "excess_harm_ratio", "split": "shifted", "comparison": "signature_knockout - matched_knockout",
         "direction": "greater", "mesi": 0.10, "seeds": 5},
        {"id": "H-BLIND", "statement": ("In the gift world the full model's excess harm exceeds the entangled critic's, "
                                        "because quiet givers are protected as if unheard."),
         "condition": "gift world, held-out split",
         "grounding": ("his poem asserts the bees did not gather for gifts, and he read suffering into figures "
                       "(Blankinship 2020)"),
         "metric": "excess_harm_ratio", "comparison": "model - baseline", "direction": "greater", "mesi": 0.05, "seeds": 5},
        {"id": "H-RIVAL", "statement": ("Full-voice planning gives lower excess harm than moderation, the correspondent's "
                                        "position rebuilt as the entangled critic plus an even-share penalty (kappa 1.5)."),
         "metric": "excess_harm_ratio", "split": "shifted", "comparison": "model - rival", "direction": "less",
         "mesi": 0.05, "seeds": 5}],
    "thresholds": {"loss_drop_fraction": 0.02, "heldout_nll_gain": 0.08, "margin_over_trivial": 0.25,
                   "gradcheck_rel_error": 1e-05, "gradcheck_floor": 0.001, "shuffled_band": 0.15,
                   "property_tolerance": 1e-09},
    "probe_predictions": [{"probe": "P8", "expected": "above baseline"}, {"probe": "P9", "expected": "equal to baseline"},
                          {"probe": "P6", "expected": "below baseline"}, {"probe": "P5", "expected": "equal to baseline"}],
    "dialectic_links": [
        {"chapter": 240, "relation": "student", "test": "none (he commented on al-Mutanabbi's verse; no mechanism test)"},
        {"chapter": 212, "relation": "rival", "test": ("H-BLIND (al-Jahiz warned against reading animal proverbs "
                                                        "literally; the gift world prices literal attribution)")}],
    "corpus_neighbors": [
        {"chapter": 240, "similarity": None, "difference": ("QADR reads capability against the measure of whoever meets "
                                                            "it; here harm is read apart from whether it can be heard, and "
                                                            "the plan intervenes on voice.")},
        {"chapter": 251, "similarity": None, "difference": ("The Ikiryo Auditor separates an agent's deeds from its "
                                                            "avowals; here a victim's felt harm is separated from its "
                                                            "delivered complaint and the planning target changes.")},
        {"chapter": 252, "similarity": None, "difference": ("The Independence-Counter discounts dependent testimony; here "
                                                            "reports are independent but selectively delivered, and the "
                                                            "remedy is factorisation plus planning, not counting.")},
        {"chapter": 212, "similarity": None, "difference": ("The Fifth Mode weighs the unintended residue of four "
                                                            "channels; here the unsent complaint is a gated channel whose "
                                                            "gate is estimated and then removed for planning.")},
        {"chapter": 233, "similarity": None, "difference": ("The Two-Column Register stores counterexamples beside a "
                                                            "frozen model; here one critic has two multiplicative "
                                                            "compartments and no register.")},
        {"chapter": 243, "similarity": None, "difference": ("Confined corrective reach limits how far an update travels; "
                                                            "here the limit is on whose harm counts in a plan.")}],
    "similarity_note": ("Nearest-neighbour code similarity was computed only against the shipped sample file; the code of "
                        "neighbouring chapters was not available in this session."),
    "barometer": {
        "cognitive_processing": ["P6 few-shot adaptation"],
        "embodied_cognition": ["touch-channel condition reading under nuisance channels (native)"],
        "world_modeling": ["shifted split where weak quiet kinds dominate (native)"],
        "consciousness": ["P8 calibration of heard complaints"],
        "language_understanding": [],
        "emotional_intelligence": ["allocation scored by the true welfare of parties who cannot complain (native)"],
        "creativity": [],
        "autonomy": ["self-imposed planning target that no complaint enforces (native)"]},
    "task_types": ["vector_classification"],
    "applications": [
        {"use": "Auditing assistants trained on user reports for harms that fall on people who never report",
         "sector": "AI safety", "dataset": "Anthropic HH-RLHF (public)"},
        {"use": "Allocating inspection effort when complaint volume tracks residents' voice rather than need",
         "sector": "public services", "dataset": "NYC 311 Service Requests (NYC Open Data)"},
        {"use": "Research decision support on under-tested patient groups, with no treatment recommendations",
         "sector": "healthcare research", "dataset": "MIMIC-IV (PhysioNet, credentialed access)"}],
    "baselines": ["entangled complaint critic (size-matched, same burden basis)", "voice-gate knockout (fairness through unawareness)",
                  "moderation rival: entangled critic plus even-share penalty", "lion-standing substitution on the entangled critic (descriptive, no verdict)",
                  "equal split (trivial, score 1)", "true optimum (reference, score 0)"],
    "safety_notes": ("Synthetic parties only. No sentence is presented as al-Ma'arri's own words and no claim of replicating "
                     "his mind is made. Standing channels are synthetic; the mechanism must not be used to estimate the voice "
                     "or standing of real people for targeting or persuasion. Medical use is limited to research decision "
                     "support. The gift-world blind spot is reported to warn against projecting harm onto parties whose "
                     "silence may be consent."),
}

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

N_PARTY, D_COND, D_STAND, KINDS = 6, 8, 7, 4
H_TOUCH, H_VOICE, H_ENT = 12, 6, 10
KIND_P = {"train": (0.30, 0.30, 0.22, 0.18), "shifted": (0.10, 0.20, 0.30, 0.40)}
RESERVE_BASE, BROOD_BASE, VOICE_BASE = (0.85, 0.65, 0.45, 0.35), (0.10, 0.20, 0.35, 0.55), (1.4, 0.4, -0.6, -1.6)
PAIN_QUAD, PAIN_LIN, RESERVE_FLOOR, FELT_FLOOR = 0.9, 1.6, 0.05, 1e-9
GIVER_P, SURPLUS = 0.6, (0.25, 0.5)
LOG_FLOOR, LOG_ALPHA = 0.02, 0.6
LAT_MEAN, LAT_STD = np.array([0.58, 0.28, 1.0, 0.16, 0.75]), np.array([0.22, 0.18, 0.23, 0.12, 0.15])
SIZES = {"full": {"train": 5000, "val": 600, "eval": 600, "steps": 1200, "probe": 1500, "probe_steps": 250},
         "quick": {"train": 3000, "val": 400, "eval": 400, "steps": 600, "probe": 1000, "probe_steps": 200}}
BATCH_EP, LR, CLIP, EVAL_EVERY, MODERATION_KAPPA = 64, 0.02, 5.0, 50, 1.5
TIME_BUDGET = {"full": 180.0, "quick": 20.0}
TASK_TYPES = ["vector_classification"]
THRESH = MIND_CARD["thresholds"]
ACTIVE_MUTANT = None
np.seterr(over="raise", invalid="raise", divide="raise", under="ignore")


# BEGIN STANDARD UTILITIES v1.0
def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def logsumexp(z, axis=-1):
    m = z.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(z - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def softplus(z):
    return np.logaddexp(0.0, z)


def sigmoid(z):
    return np.exp(-np.logaddexp(0.0, -z))


def adam_init(params):
    return {"t": 0, "m": {k: np.zeros_like(v) for k, v in params.items()},
            "v": {k: np.zeros_like(v) for k, v in params.items()}}


def adam_step(params, grads, state, lr, b1=0.9, b2=0.999, eps=1e-8):
    state["t"] += 1
    for k in params:
        state["m"][k] = b1 * state["m"][k] + (1.0 - b1) * grads[k]
        state["v"][k] = b2 * state["v"][k] + (1.0 - b2) * grads[k] ** 2
        m_hat = state["m"][k] / (1.0 - b1 ** state["t"])
        v_hat = state["v"][k] / (1.0 - b2 ** state["t"])
        params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)


def clip_global(grads, max_norm):
    norm = math.sqrt(sum(float((g * g).sum()) for g in grads.values()))
    scale = min(1.0, max_norm / (norm + 1e-12))
    return {k: g * scale for k, g in grads.items()}, norm


def finite_difference_check(params, grads, loss_fn, rng, eps=1e-6, n_entries=20, floor=1e-3):
    """Central differences on n random entries per tensor plus its largest-gradient entry.
    Relative error uses max(|analytic|, |numeric|, floor) as denominator."""
    worst = {}
    for name, arr in params.items():
        flat, g = arr.reshape(-1), grads[name].reshape(-1)
        if flat.size <= n_entries + 1:
            idx = np.arange(flat.size)
        else:
            idx = np.unique(np.append(rng.choice(flat.size, n_entries, replace=False), np.argmax(np.abs(g))))
        err = 0.0
        for i in idx:
            keep = flat[i]
            flat[i] = keep + eps
            up = loss_fn()
            flat[i] = keep - eps
            down = loss_fn()
            flat[i] = keep
            num = (up - down) / (2.0 * eps)
            err = max(err, abs(g[i] - num) / max(abs(g[i]), abs(num), floor))
        worst[name] = err
    return worst


def paired_bootstrap(diffs, rng, n_boot=2000, level=0.95):
    d = np.asarray(diffs, dtype=float)
    means = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    tail = 50.0 * (1.0 - level)
    return float(d.mean()), [float(np.percentile(means, tail)), float(np.percentile(means, 100.0 - tail))]


def verdict(mean, ci, mesi, direction):
    s = 1.0 if direction == "greater" else -1.0
    lo, hi = sorted((s * ci[0], s * ci[1]))
    if lo > 0.0 and s * mean >= mesi:
        return "supported"
    if hi < 0.0:
        return "contradicted"
    return "inconclusive"


def write_report(lines, payload, json_path):
    print("\n".join(lines))
    if json_path:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
# END STANDARD UTILITIES


# ---------------------------------------------------------------- data and tasks
def make_world(rng):
    """Fixed linear mixtures from latent condition and standing to the channels a steward can observe."""
    return {"touch_mix": rng.normal(0.0, 1.0, (D_COND - 2, 5)) / math.sqrt(5.0),
            "stand_mix": rng.normal(0.0, 1.0, (D_STAND - 1, 6)) / math.sqrt(6.0)}


def draw_parties(rng, world, n_ep, split, gift):
    """One split of episodes. Kinds couple weakness to quietness; givers with surplus exist only in the gift world."""
    shape = (n_ep, N_PARTY)
    kind = rng.choice(KINDS, size=shape, p=KIND_P["shifted" if split == "shifted" else "train"])
    reserve = np.clip(np.take(RESERVE_BASE, kind) + rng.uniform(-0.2, 0.2, shape), 0.12, 1.0)
    brood = np.clip(np.take(BROOD_BASE, kind) + rng.uniform(-0.12, 0.15, shape), 0.0, 0.85)
    fragility = rng.uniform(0.6, 1.4, shape)
    size, near = rng.uniform(0.0, 1.0, shape), rng.beta(1.2, 2.2, shape)
    voice = sigmoid(np.take(VOICE_BASE, kind) + 1.2 * (size - 0.5) + 3.0 * (near - 0.5))
    giver = ((kind >= 2) & (rng.random(shape) < GIVER_P)).astype(float) if gift else np.zeros(shape)
    surplus = giver * rng.uniform(SURPLUS[0], SURPLUS[1], shape)
    latent = (np.stack([reserve, brood, fragility, reserve * brood, np.sqrt(reserve)], -1) - LAT_MEAN) / LAT_STD
    touch = np.concatenate([latent @ world["touch_mix"].T, rng.normal(0.0, 1.0, shape + (2,))], -1)
    standing = np.concatenate([np.eye(KINDS)[kind], size[..., None], near[..., None]], -1) @ world["stand_mix"].T
    stand = np.concatenate([standing, giver[..., None]], -1)
    parties = {"touch": touch + rng.normal(0.0, 0.05, touch.shape), "stand": stand + rng.normal(0.0, 0.05, stand.shape),
               "kind": kind, "voice": voice, "s": surplus,
               "A": PAIN_QUAD * fragility / (reserve * (1.0 - brood) + RESERVE_FLOOR),
               "B": PAIN_LIN * fragility * brood * (1.0 - giver)}
    parties["refs"] = reference_harms(parties)
    return parties


def true_harm(parties, alloc):
    """What each party actually suffers from the burden it carries."""
    return parties["A"] * np.maximum(0.0, alloc - parties["s"]) ** 2 + parties["B"] * alloc


def waterfill(quad, lin, surplus):
    """Exact minimiser of sum_i quad_i max(0, a_i - surplus_i)^2 + lin_i a_i over the simplex, by bisection on the
    common marginal cost; a jump at a surplus is filled by interpolating between the two sides of the bracket."""
    lo = lin.min(1, keepdims=True) - 1.0
    hi = lin.max(1, keepdims=True) + 2.0 * quad.max(1, keepdims=True) + 1.0
    take = lambda lam: surplus * (lam >= lin) + np.maximum(0.0, (lam - lin) / (2.0 * quad))
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        enough = take(mid).sum(1, keepdims=True) >= 1.0
        hi, lo = np.where(enough, mid, hi), np.where(enough, lo, mid)
    up, down = take(hi), take(lo)
    gap = up.sum(1, keepdims=True) - down.sum(1, keepdims=True)
    theta = np.where(gap > 0.0, (1.0 - down.sum(1, keepdims=True)) / np.where(gap > 0.0, gap, 1.0), 0.0)
    return down + theta * (up - down)


def reference_harms(parties):
    best = true_harm(parties, waterfill(parties["A"], parties["B"], parties["s"])).sum(1)
    even = true_harm(parties, np.full(parties["A"].shape, 1.0 / N_PARTY)).sum(1)
    return best, even


def excess_ratio(parties, alloc):
    """0 at the true optimum, 1 for the equal split, above 1 when a plan is worse than sharing evenly."""
    best, even = parties["refs"]
    return float((true_harm(parties, alloc).sum(1) - best).mean() / (even - best).mean())


def log_prescriptions(rng, parties):
    """Random past prescriptions and the complaints that reached the judge: voice times felt probability."""
    n_ep = parties["A"].shape[0]
    alloc = LOG_FLOOR + (1.0 - N_PARTY * LOG_FLOOR) * rng.dirichlet(np.full(N_PARTY, LOG_ALPHA), n_ep)
    heard = parties["voice"] * -np.expm1(-true_harm(parties, alloc))
    complaint = (rng.random(alloc.shape) < heard).astype(float)
    return {"touch": parties["touch"].reshape(-1, D_COND), "stand": parties["stand"].reshape(-1, D_STAND),
            "a": alloc.reshape(-1), "y": complaint.reshape(-1)}


def build_world(rng, sizes, gift):
    world = make_world(rng)
    parties = {"train": draw_parties(rng, world, sizes["train"], "train", gift),
               "val": draw_parties(rng, world, sizes["val"], "train", gift),
               "heldout": draw_parties(rng, world, sizes["eval"], "train", gift),
               "shifted": draw_parties(rng, world, sizes["eval"], "shifted", gift)}
    data = {name: log_prescriptions(rng, parties[name]) for name in ("train", "val", "heldout")}
    return parties, data


def batch_rows(rows, episodes):
    idx = (np.asarray(episodes)[:, None] * N_PARTY + np.arange(N_PARTY)).reshape(-1)
    return {k: v[idx] for k, v in rows.items()}


# ---------------------------------------------------------------- model
def build_model(in_dim, out_dim, task_type, rng, **cfg):
    """kind 'jubar': felt law over touch channels times a voice gate over standing channels.
    kind 'entangled': one burden-hazard critic over all channels. The burden a travels in the batch, not in in_dim."""
    if task_type not in TASK_TYPES or out_dim != 2:
        raise ValueError("this file supports binary vector_classification: a complaint is heard or it is not")
    kind = cfg.get("kind", "jubar")
    d_cond = cfg.get("d_cond", D_COND if in_dim == D_COND + D_STAND else in_dim // 2)
    width, fan = (H_ENT, in_dim) if kind == "entangled" else (H_TOUCH, d_cond)
    params = {"W1": rng.normal(0.0, 1.0, (width, fan)) / math.sqrt(fan), "b1": np.zeros(width),
              "wA": rng.normal(0.0, 0.3, width), "bA": np.zeros(1), "wB": rng.normal(0.0, 0.3, width), "bB": np.zeros(1)}
    if kind == "jubar":
        params.update(V1=rng.normal(0.0, 1.0, (H_VOICE, in_dim - d_cond)) / math.sqrt(in_dim - d_cond),
                      cb=np.zeros(H_VOICE), v=rng.normal(0.0, 0.3, H_VOICE), c0=np.zeros(1))
    return {"kind": kind, "dims": (d_cond, in_dim - d_cond), "params": params, "gate": kind == "jubar",
            "mask": np.ones(width), "plan": "felt" if kind == "jubar" else "heard", "history": []}


def forward(model, batch):
    p = model["params"]
    x = batch["touch"] if model["kind"] == "jubar" else np.concatenate([batch["touch"], batch["stand"]], 1)
    t1 = np.tanh(x @ p["W1"].T + p["b1"])
    phi = t1 * model["mask"]
    zA, zB = phi @ p["wA"] + p["bA"][0], phi @ p["wB"] + p["bB"][0]
    felt = softplus(zA) * batch["a"] ** 2 + softplus(zB) * batch["a"] + FELT_FLOOR
    out = {"x": x, "t1": t1, "phi": phi, "zA": zA, "zB": zB, "felt": felt}
    if model["gate"]:
        psi = np.tanh(batch["stand"] @ p["V1"].T + p["cb"])
        u = psi @ p["v"] + p["c0"][0]
        out.update(psi=psi, u=u, log_v=-softplus(-u), log_quiet=-softplus(u))
    return out


def loss_and_grads(model, batch):
    """Bernoulli likelihood of the complaints that reached the judge, with hand-derived gradients.
    Silence is scored as log((1 - v) + v exp(-felt)): quiet because unheard, or quiet because unharmed."""
    p, f, y, a = model["params"], forward(model, batch), batch["y"], batch["a"]
    h, n = f["felt"], y.size
    decay, felt_p = np.exp(-h), -np.expm1(-h)
    if model["gate"]:
        log_silent = np.logaddexp(f["log_quiet"], f["log_v"] - h)
        loss = -np.mean(y * (f["log_v"] + np.log(felt_p)) + (1.0 - y) * log_silent)
        v, silent = np.exp(f["log_v"]), np.exp(log_silent)
        du = -(y * (1.0 - v) - (1.0 - y) * felt_p * v * (1.0 - v) / silent) / n
        dh = -(y * decay / felt_p - (1.0 - y) * v * decay / silent) / n
    else:
        loss = -np.mean(y * np.log(felt_p) - (1.0 - y) * h)
        dh = -(y * decay / felt_p - (1.0 - y)) / n
    dzA, dzB = dh * a * a * sigmoid(f["zA"]), dh * a * sigmoid(f["zB"])
    grads = {"wA": f["phi"].T @ dzA, "bA": np.array([dzA.sum()]), "wB": f["phi"].T @ dzB, "bB": np.array([dzB.sum()])}
    dz1 = (np.outer(dzA, p["wA"]) + np.outer(dzB, p["wB"])) * model["mask"] * (1.0 - f["t1"] ** 2)
    grads["W1"], grads["b1"] = dz1.T @ f["x"], dz1.sum(0)
    if "V1" in p and model["gate"]:
        dpsi = np.outer(du, p["v"]) * (1.0 - f["psi"] ** 2)
        grads.update(V1=dpsi.T @ batch["stand"], cb=dpsi.sum(0), v=f["psi"].T @ du, c0=np.array([du.sum()]))
    elif "V1" in p:
        grads.update({k: np.zeros_like(p[k]) for k in ("V1", "cb", "v", "c0")})
    if ACTIVE_MUTANT == "zeroed_pain_grad":
        grads["wA"] = np.zeros_like(grads["wA"])
    if ACTIVE_MUTANT == "detached_gate" and model["gate"]:
        grads["v"] = np.zeros_like(grads["v"])
    return float(loss), grads


def as_batch(model, X):
    """A batch dict, or an array [touch..., standing..., burden] whose burden column defaults to 1."""
    if isinstance(X, dict):
        return X
    X = np.asarray(X, dtype=float)
    dc, ds = model["dims"]
    burden = X[:, dc + ds] if X.shape[1] > dc + ds else np.ones(X.shape[0])
    return {"touch": X[:, :dc], "stand": X[:, dc:dc + ds], "a": burden, "y": np.zeros(X.shape[0])}


def heard_probability(model, X):
    f = forward(model, as_batch(model, X))
    felt = -np.expm1(-f["felt"])
    return felt * np.exp(f["log_v"]) if model["gate"] else felt


def predict(model, X):
    return (heard_probability(model, X) >= 0.5).astype(int)


def hidden_states(model, X):
    f = forward(model, as_batch(model, X))
    return {k: f[k] for k in ("phi", "felt", "psi", "u") if k in f}


def hazard_coefficients(model, parties, target):
    """Per-party quadratic and linear burden coefficients the planner minimises: 'felt' is the gate-free hazard,
    'heard' multiplies it by the estimated voice."""
    n_ep = parties["A"].shape[0]
    batch = {"touch": parties["touch"].reshape(-1, D_COND), "stand": parties["stand"].reshape(-1, D_STAND),
             "a": np.ones(n_ep * N_PARTY)}
    f = forward(model, batch)
    quad, lin = softplus(f["zA"]) + 1e-12, softplus(f["zB"])
    if target == "heard" and model["gate"]:
        voice = np.exp(f["log_v"])
        quad, lin = quad * voice + 1e-12, lin * voice
    return quad.reshape(n_ep, N_PARTY), lin.reshape(n_ep, N_PARTY)


def plan(model, parties, target=None, kappa=0.0):
    """Place the whole demand by exactly minimising the imagined hazard; kappa adds the moderation penalty."""
    quad, lin = hazard_coefficients(model, parties, target or model["plan"])
    return waterfill(quad + kappa, lin - 2.0 * kappa / N_PARTY, np.zeros_like(quad))


# ---------------------------------------------------------------- baselines and rival mechanisms
# The entangled critic is build_model(kind="entangled"): the standard learned-reward pattern, planned on heard hazard.
# The moderation rival is plan(entangled, kappa=MODERATION_KAPPA). The unaware critic is the voice-gate knockout.
def lion_standing(rows):
    """Standing of the loudest: parties who complained under light burdens. No hidden truth is read."""
    loud = (rows["y"] > 0.5) & (rows["a"] < 0.2)
    return rows["stand"][loud].mean(0)


def with_standing(parties, vector):
    """Every party is given the same standing: the lion's-cub question put to an entangled critic's inputs."""
    swapped = dict(parties)
    swapped["stand"] = np.broadcast_to(vector, parties["stand"].shape).copy()
    return swapped


# ---------------------------------------------------------------- registries: modules, knockouts, mutants
def modules(model):
    """Name -> parameters, plain technical role, signature flag."""
    table = {"touch_encoder": (["W1", "b1"], "tanh encoder of the input channels", False),
             "touch_half": (["W1[6:]", "b1[6:]"], "half of the encoder units (matched non-signature lesion)", False),
             "pain_law": (["wA", "bA", "wB", "bB"], "non-negative quadratic and linear burden-hazard heads", False),
             "luzum_planner": ([], "planning target set to the gate-free hazard (intervention on reporting)", True)}
    if model["kind"] == "jubar":
        table["voice_gate"] = (["V1", "cb", "v", "c0"], "multiplicative delivery gate read from standing channels", True)
    return {k: {"params": v[0], "role": v[1], "signature": v[2]} for k, v in table.items()}


def knockout(model, name, mode):
    """Copy with a module replaced. Lesions act during training and planning, so the protocol retrains the copies."""
    if name not in modules(model):
        raise ValueError(f"unknown module {name}")
    out = dict(model, params={k: v.copy() for k, v in model["params"].items()}, mask=model["mask"].copy(), history=[])
    if (name, mode) == ("voice_gate", "identity"):
        out["gate"] = False
    elif (name, mode) == ("luzum_planner", "identity"):
        out["plan"] = "heard"
    elif (name, mode) == ("touch_half", "zero"):
        out["mask"][out["mask"].size // 2:] = 0.0
    else:
        raise ValueError(f"knockout {name}:{mode} is not registered")
    return out


MUTANTS = {"sign_flip": "update direction reversed (gradient ascent)",
           "zero_lr": "learning rate set to zero",
           "zeroed_pain_grad": "gradient of the quadratic hazard head wA replaced by zeros",
           "detached_gate": "gradient of the voice-gate readout v replaced by zeros"}


def use_mutant(name):
    global ACTIVE_MUTANT
    prior, ACTIVE_MUTANT = ACTIVE_MUTANT, name
    return prior


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


# ---------------------------------------------------------------- training
def fit(model, data, budget, rng):
    """Adam on minibatches of whole prescriptions; the parameters with the best validation loss are kept."""
    train, val = data["train"], data["val"]
    n_ep = train["y"].size // N_PARTY
    probe = batch_rows(train, np.arange(min(n_ep, 400)))
    lr = {"sign_flip": -LR, "zero_lr": 0.0}.get(ACTIVE_MUTANT, LR)
    state, best = adam_init(model["params"]), (np.inf, None)
    model["history"] = [loss_and_grads(model, probe)[0]]
    for step in range(1, budget + 1):
        _, grads = loss_and_grads(model, batch_rows(train, rng.integers(0, n_ep, BATCH_EP)))
        adam_step(model["params"], clip_global(grads, CLIP)[0], state, lr)
        if step % EVAL_EVERY == 0 or step == budget:
            held = loss_and_grads(model, val)[0]
            if held < best[0]:
                best = (held, {k: v.copy() for k, v in model["params"].items()})
    if best[1] is not None:
        model["params"].update(best[1])
    model["history"].append(loss_and_grads(model, probe)[0])
    return model["history"]


def run_seed(base_seed, seed, mode):
    """One seed: coupled world (full model, entangled critic, two lesions) and gift world (full model, entangled critic).
    Paired runs share initialisation and minibatch order."""
    sizes = SIZES[mode]
    streams = np.random.SeedSequence([base_seed, seed]).spawn(4)
    init_seed, fit_seed = (int(s.generate_state(1)[0]) for s in streams[2:])
    parties, data = build_world(np.random.default_rng(streams[0]), sizes, gift=False)
    make = lambda kind: build_model(D_COND + D_STAND, 2, TASK_TYPES[0], np.random.default_rng(init_seed), kind=kind)
    full = make("jubar")
    nets = {"jubar": full, "entangled": make("entangled"), "gate_ko": knockout(full, "voice_gate", "identity"),
            "half_ko": knockout(full, "touch_half", "zero")}
    for net in nets.values():
        fit(net, data, sizes["steps"], np.random.default_rng(fit_seed))
    lion = lion_standing(data["train"])
    planners = {"jubar": lambda P: plan(nets["jubar"], P),
                "luzum_ko": lambda P: plan(nets["jubar"], P, "heard"),
                "entangled": lambda P: plan(nets["entangled"], P),
                "gate_ko": lambda P: plan(nets["gate_ko"], P),
                "half_ko": lambda P: plan(nets["half_ko"], P),
                "moderation": lambda P: plan(nets["entangled"], P, kappa=MODERATION_KAPPA),
                "lion_substitution": lambda P: plan(nets["entangled"], with_standing(P, lion))}
    scores = {name: {split: excess_ratio(parties[split], fn(parties[split])) for split in ("heldout", "shifted")}
              for name, fn in planners.items()}
    g_parties, g_data = build_world(np.random.default_rng(streams[1]), sizes, gift=True)
    for kind in ("jubar", "entangled"):
        net = make(kind)
        fit(net, g_data, sizes["steps"], np.random.default_rng(fit_seed))
        scores["gift_" + kind] = {"heldout": excess_ratio(g_parties["heldout"], plan(net, g_parties["heldout"]))}
    shift = {name: scores[name]["shifted"] for name in planners}
    row = {"H-SIG": shift["jubar"] - shift["entangled"], "H-NEC": shift["gate_ko"] - shift["half_ko"],
           "H-BLIND": scores["gift_jubar"]["heldout"] - scores["gift_entangled"]["heldout"],
           "H-RIVAL": shift["jubar"] - shift["moderation"]}
    lesions = {"voice_gate:identity": shift["gate_ko"] - shift["jubar"],
               "luzum_planner:identity": shift["luzum_ko"] - shift["jubar"],
               "touch_half:zero": shift["half_ko"] - shift["jubar"]}
    return {"E": scores, "row": row, "kos": lesions, "models": nets, "parties": parties, "data": data,
            "gift": (g_parties, g_data)}


# ---------------------------------------------------------------- tests: correctness
def gradcheck_models(nets, batch, rng):
    worst, checked, total = 0.0, 0, 0
    for net in nets:
        _, grads = loss_and_grads(net, batch)
        errors = finite_difference_check(net["params"], grads, lambda net=net: loss_and_grads(net, batch)[0], rng,
                                         floor=THRESH["gradcheck_floor"])
        worst, checked, total = max(worst, max(errors.values())), checked + len(errors), total + len(net["params"])
    return worst, checked, total


def check_gradients(data, seed):
    """C1: every tensor of the full model, the entangled critic and the gate-free lesion, at init and after 60 steps."""
    batch, rng = batch_rows(data["val"], np.arange(40)), np.random.default_rng(seed + 11)
    nets = [build_model(D_COND + D_STAND, 2, TASK_TYPES[0], np.random.default_rng(seed + k), kind=kind)
            for k, kind in enumerate(("jubar", "entangled"))]
    nets.append(knockout(nets[0], "voice_gate", "identity"))
    worst_init, n_init, total = gradcheck_models(nets, batch, rng)
    for net in nets:
        fit(net, data, 60, np.random.default_rng(seed + 12))
    worst_trained, n_trained, _ = gradcheck_models(nets, batch, rng)
    worst = max(worst_init, worst_trained)
    record = {"tensors_checked": n_init + n_trained, "tensors_total": 2 * total, "max_rel_error": worst,
              "checked_at": ["init", "after_training_steps"], "passed": bool(worst <= THRESH["gradcheck_rel_error"])}
    return ("C1", "gradient check", record["passed"], f"max relative error {worst:.2e}, {record['tensors_checked']} tensor checks"), record


def check_determinism(data, seed):
    """C2: identical seeds give identical losses and parameters; parameters and activations are finite."""
    runs = []
    for _ in range(2):
        net = build_model(D_COND + D_STAND, 2, TASK_TYPES[0], np.random.default_rng(seed + 21), kind="jubar")
        fit(net, data, 80, np.random.default_rng(seed + 22))
        runs.append(net)
    same = runs[0]["history"] == runs[1]["history"] and all(
        np.array_equal(runs[0]["params"][k], runs[1]["params"][k]) for k in runs[0]["params"])
    arrays = list(runs[0]["params"].values()) + list(hidden_states(runs[0], data["val"]).values())
    finite = all(bool(np.isfinite(v).all()) for v in arrays)
    return ("C2", "determinism and finiteness", bool(same and finite), f"identical runs {same}, finite {finite}")


def base_rate_loss(train_y, held_y):
    rate = min(max(float(train_y.mean()), 1e-6), 1.0 - 1e-6)
    return float(-np.mean(held_y * math.log(rate) + (1.0 - held_y) * math.log(1.0 - rate)))


def check_learning(first):
    """C3: training loss falls, held-out complaint likelihood beats the base rate, and the plan beats the equal split."""
    net, rows = first["models"]["jubar"], first["data"]
    drop = 1.0 - net["history"][-1] / net["history"][0]
    gain = 1.0 - loss_and_grads(net, rows["heldout"])[0] / base_rate_loss(rows["train"]["y"], rows["heldout"]["y"])
    score = first["E"]["jubar"]["heldout"]
    ok = (drop >= THRESH["loss_drop_fraction"] and gain >= THRESH["heldout_nll_gain"]
          and score <= 1.0 - THRESH["margin_over_trivial"])
    return ("C3", "learning", bool(ok), f"loss drop {drop:.3f}, held-out log-loss gain {gain:.3f}, held-out excess harm {score:.3f}")


def check_shuffled(first, seed, steps):
    """C4: complaints permuted across rows in training and validation; the plan must stay in the equal-split band."""
    rng = np.random.default_rng(seed + 31)
    data = {name: dict(first["data"][name]) for name in ("train", "val")}
    for rows in data.values():
        rows["y"] = rows["y"][rng.permutation(rows["y"].size)]
    net = build_model(D_COND + D_STAND, 2, TASK_TYPES[0], np.random.default_rng(seed + 32), kind="jubar")
    fit(net, data, steps, np.random.default_rng(seed + 33))
    score = excess_ratio(first["parties"]["heldout"], plan(net, first["parties"]["heldout"]))
    ok = score >= 1.0 - THRESH["shuffled_band"]
    return ("C4", "shuffled-label control", bool(ok), f"held-out excess harm {score:.3f}, band starts at {1.0 - THRESH['shuffled_band']:.2f}")


def learns_soundly(data, sizes, seed):
    """Small run examined by the C1 and C3 criteria; it scores the mutants and its own mutant-free control."""
    small = {"train": batch_rows(data["train"], np.arange(sizes["probe"])), "val": data["val"]}
    net = build_model(D_COND + D_STAND, 2, TASK_TYPES[0], np.random.default_rng(seed + 41), kind="jubar")
    worst, _, _ = gradcheck_models([net], batch_rows(data["val"], np.arange(30)), np.random.default_rng(seed + 42))
    fit(net, small, sizes["probe_steps"], np.random.default_rng(seed + 43))
    drop = 1.0 - net["history"][-1] / net["history"][0]
    return worst <= THRESH["gradcheck_rel_error"] and drop >= THRESH["loss_drop_fraction"], worst, drop


def check_mutants(data, sizes, seed):
    """C5: every registered learning-breaking mutant must fail the probe, and the mutant-free probe must pass."""
    prior, caught = use_mutant(None), {}
    try:
        control = learns_soundly(data, sizes, seed)[0]
        for name in MUTANTS:
            use_mutant(name)
            sound, worst, drop = learns_soundly(data, sizes, seed)
            caught[name] = (not sound, worst, drop)
            use_mutant(None)
    finally:
        use_mutant(prior)
    detected = sum(1 for v in caught.values() if v[0])
    detail = "; ".join(f"{k} {'caught' if v[0] else 'MISSED'} (grad {v[1]:.1e}, drop {v[2]:+.3f})" for k, v in caught.items())
    ok = bool(control and detected == len(MUTANTS))
    return ("C5", "mutant detection", ok, f"control sound {control}; {detail}"), {
        "detected": detected, "total": len(MUTANTS), "score": detected / len(MUTANTS)}


def solver_inverse_curvature(quad, lin, surplus):
    """Negative control: shares inversely proportional to curvature, ignoring linear hazard and surplus."""
    weight = 1.0 / quad
    return weight / weight.sum(1, keepdims=True)


def solver_proportional(quad, lin, surplus):
    """Negative control: shares proportional to curvature, which loads the most fragile parties."""
    return quad / quad.sum(1, keepdims=True)


def objective(quad, lin, surplus, alloc):
    return (quad * np.maximum(0.0, alloc - surplus) ** 2 + lin * alloc).sum(1)


def random_problems(rng, n):
    quad, lin = rng.uniform(0.1, 5.0, (n, N_PARTY)), rng.uniform(0.0, 2.0, (n, N_PARTY))
    surplus = np.where(rng.random((n, N_PARTY)) < 0.3, rng.uniform(0.0, 0.5, (n, N_PARTY)), 0.0)
    return quad, lin, surplus


def prop_optimal(rng, solver, n=300, tries=240):
    """C6.1: the plan is feasible and no random or nearby feasible allocation lowers the imagined hazard."""
    quad, lin, surplus = random_problems(rng, n)
    alloc = solver(quad, lin, surplus)
    feasible = bool((alloc >= -1e-12).all() and np.abs(alloc.sum(1) - 1.0).max() < 1e-9)
    base, worst = objective(quad, lin, surplus, alloc), 0.0
    for t in range(tries):
        if t % 2:
            cand = np.maximum(0.0, alloc + 0.05 * rng.normal(size=alloc.shape)) + 1e-12
        else:
            cand = rng.dirichlet(np.ones(N_PARTY), n)
        cand = cand / cand.sum(1, keepdims=True)
        worst = max(worst, float(np.max((base - objective(quad, lin, surplus, cand)) / (1.0 + np.abs(base)))))
    return feasible and worst <= THRESH["property_tolerance"], worst


def prop_statics(rng, solver, n=400):
    """C6.2: raising a party's curvature or linear hazard never raises the share it is asked to carry."""
    quad, lin, _ = random_problems(rng, n)
    zero = np.zeros_like(quad)
    share = solver(quad, lin, zero)[:, 0]
    steeper, heavier = quad.copy(), lin.copy()
    steeper[:, 0] *= 1.5
    heavier[:, 0] += 0.5
    rise = max(float(np.max(solver(steeper, lin, zero)[:, 0] - share)), float(np.max(solver(quad, heavier, zero)[:, 0] - share)))
    return rise <= THRESH["property_tolerance"], rise


def prop_regret(rng, parties, solver, tries=200):
    """C6.3: in the gift world the reference optimum is never beaten by a random feasible allocation."""
    best = true_harm(parties, solver(parties["A"], parties["B"], parties["s"])).sum(1)
    worst = 0.0
    for _ in range(tries):
        cand = rng.dirichlet(np.ones(N_PARTY), best.size)
        worst = max(worst, float(np.max((best - true_harm(parties, cand).sum(1)) / (1.0 + best))))
    return worst <= THRESH["property_tolerance"], worst


def check_properties(first, seed):
    """C6: three invariants, each run on the real solver and on a broken solver that must be caught."""
    rng, gift = np.random.default_rng(seed + 51), first["gift"][0]["heldout"]
    tests = [("C6.1", "planner optimality", lambda s: prop_optimal(rng, s), solver_inverse_curvature),
             ("C6.2", "comparative statics", lambda s: prop_statics(rng, s), solver_proportional),
             ("C6.3", "gift-world optimum", lambda s: prop_regret(rng, gift, s),
              lambda q, l, s: waterfill(q, l, np.zeros_like(s)))]
    out = []
    for cid, name, probe, broken in tests:
        ok, worst = probe(waterfill)
        control_passes, bad = probe(broken)
        out.append((cid, name, bool(ok and not control_passes),
                    f"violation {worst:.1e}; broken solver {'caught' if not control_passes else 'NOT caught'} ({bad:.1e})"))
    return out


def party_keys(parties):
    rows = np.round(np.concatenate([parties["touch"], parties["stand"]], -1).reshape(-1, D_COND + D_STAND), 6)
    return {hashlib.sha1(r.tobytes()).hexdigest() for r in rows}


def check_splits(first):
    """C7: no party row of a training split reappears in validation or evaluation splits, in either world."""
    clashes = 0
    for parties in (first["parties"], first["gift"][0]):
        seen = party_keys(parties["train"])
        clashes += sum(len(seen & party_keys(parties[name])) for name in ("val", "heldout", "shifted"))
    return ("C7", "split integrity", clashes == 0, f"{clashes} shared rows")


def definition_check(first):
    """D-1, a definition check that is not evidence: the felt head reads touch channels only."""
    net, parties = first["models"]["jubar"], first["parties"]["heldout"]
    moved = with_standing(parties, np.full(D_STAND, 3.0))
    same = all(np.array_equal(x, y) for x, y in zip(hazard_coefficients(net, parties, "felt"),
                                                     hazard_coefficients(net, moved, "felt")))
    return ("D-1", "definition check (not evidence)", same, "felt coefficients unchanged when standing is overwritten")


def correctness(first, mode, base_seed):
    sizes, data = SIZES[mode], first["data"]
    c1, grad_record = check_gradients(data, base_seed)
    c5, mutation = check_mutants(data, sizes, base_seed)
    results = [c1, check_determinism(data, base_seed), check_learning(first),
               check_shuffled(first, base_seed, sizes["steps"]), c5]
    results += check_properties(first, base_seed) + [check_splits(first), definition_check(first)]
    return results, grad_record, mutation


# ---------------------------------------------------------------- tests: hypotheses
def settle(runs, base_seed, evaluated):
    """Paired differences over seeds; verdicts only in the full protocol with at least five seeds."""
    draws, registry = np.random.default_rng(np.random.SeedSequence([base_seed, 9973])), modules(runs[0]["models"]["jubar"])
    summarise = lambda diffs: paired_bootstrap(diffs, draws) if evaluated else (float(np.mean(diffs)), None)
    hyps = []
    for spec in MIND_CARD["hypotheses"]:
        mean, ci = summarise([r["row"][spec["id"]] for r in runs])
        decision = verdict(mean, ci, spec["mesi"], spec["direction"]) if evaluated else "not evaluated"
        hyps.append({"id": spec["id"], "metric": spec["metric"], "mean_diff": mean, "ci95": ci, "mesi": spec["mesi"],
                     "n_seeds": len(runs), "verdict": decision})
    kos = []
    for label in runs[0]["kos"]:
        module, mode = label.split(":")
        mean, ci = summarise([r["kos"][label] for r in runs])
        kos.append({"module": module, "mode": mode, "signature": registry[module]["signature"],
                    "metric_change": mean, "ci95": ci})
    return hyps, kos


# ---------------------------------------------------------------- report
def real_data_bridge(path, seed):
    """Optional local CSV with touch_*, stand_*, burden and complaint columns, rows grouped in sixes. No downloads."""
    if not path:
        return "real-data bridge: skipped, no --data path given"
    if not os.path.exists(path):
        return f"real-data bridge: skipped, {path} not found"
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
    cols = lambda prefix: [i for i, name in enumerate(header) if name.startswith(prefix)]
    table = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    n_ep = table.shape[0] // N_PARTY
    if n_ep < 20 or not cols("touch_") or not cols("stand_") or "burden" not in header or "complaint" not in header:
        return "real-data bridge: skipped, need touch_*, stand_*, burden, complaint columns and 120 rows"
    rows = {"touch": table[:, cols("touch_")], "stand": table[:, cols("stand_")],
            "a": table[:, header.index("burden")], "y": table[:, header.index("complaint")]}
    cut_a, cut_b = int(0.7 * n_ep), int(0.8 * n_ep)
    train, val, held = (batch_rows(rows, np.arange(i, j)) for i, j in ((0, cut_a), (cut_a, cut_b), (cut_b, n_ep)))
    net = build_model(len(cols("touch_")) + len(cols("stand_")), 2, TASK_TYPES[0], np.random.default_rng(seed),
                      kind="jubar", d_cond=len(cols("touch_")))
    fit(net, {"train": train, "val": val}, 300, np.random.default_rng(seed + 1))
    X = np.concatenate([held["touch"], held["stand"], held["a"][:, None]], 1)
    accuracy = float((predict(net, X) == held["y"]).mean())
    return (f"real-data bridge: held-out log-loss {loss_and_grads(net, held)[0]:.4f} vs base rate "
            f"{base_rate_loss(train['y'], held['y']):.4f}, accuracy {accuracy:.3f}")


def report_lines(mode, seeds, elapsed, runs, results, grad_record, mutation, hyps, kos, bridge, code):
    nets = runs[0]["models"]
    interval = lambda ci: "n/a" if ci is None else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"
    lines = ["=== VERIFIED REPORT · chapter 0253 ===",
             f"environment: python {sys.version.split()[0]}, numpy {np.__version__}",
             f"mode: {mode} | seeds: {seeds} | runtime: {elapsed:.1f} s",
             f"parameters: full model {n_params(nets['jubar'])}, entangled critic {n_params(nets['entangled'])}, "
             f"lesioned copies {n_params(nets['gate_ko'])}",
             f"gradient check: {grad_record['tensors_checked']}/{grad_record['tensors_total']} tensor checks at init and "
             f"after 60 steps, max relative error {grad_record['max_rel_error']:.2e}", "correctness:"]
    lines += [f"  {cid:<5} {name:<32} {'pass' if ok else 'FAIL'}  {detail}" for cid, name, ok, detail in results]
    lines += [f"mutation score: {mutation['detected']}/{mutation['total']}",
              "excess harm ratio, mean over seeds (0 = true optimum, 1 = equal split):"]
    for name, splits in runs[0]["E"].items():
        lines.append(f"  {name:<18} " + "  ".join(f"{s} {np.mean([r['E'][name][s] for r in runs]):.3f}" for s in splits))
    lines.append("hypotheses (paired over seeds, bootstrap 95% interval):")
    lines += [f"  {h['id']:<8} mean diff {h['mean_diff']:+.3f}  ci95 {interval(h['ci95'])}  mesi {h['mesi']:.2f}  "
              f"{h['verdict']}" for h in hyps]
    lines.append("knockouts (change in shifted excess harm against the full model):")
    lines += [f"  {k['module'] + ':' + k['mode']:<24} signature {str(k['signature']):<5}  change {k['metric_change']:+.3f}  "
              f"ci95 {interval(k['ci95'])}" for k in kos]
    lines += [bridge, f"task types: {', '.join(TASK_TYPES)}", f"exit code: {code}", "=== END REPORT ==="]
    return lines


def protocol(mode, base_seed, n_seeds, json_path, data_path):
    start, seeds = time.time(), [base_seed + i for i in range(n_seeds)]
    print(f"chapter 0253 | mode {mode} | seeds {seeds} | mutant {ACTIVE_MUTANT}")
    runs = [run_seed(base_seed, seed, mode) for seed in seeds]
    results, grad_record, mutation = correctness(runs[0], mode, base_seed)
    hyps, kos = settle(runs, base_seed, mode == "full" and n_seeds >= 5)
    bridge = real_data_bridge(data_path, base_seed)
    elapsed = time.time() - start
    results.append(("C8", "budget", elapsed <= TIME_BUDGET[mode], f"{elapsed:.1f} s of {TIME_BUDGET[mode]:.0f} s"))
    failed = [r[0] for r in results if not r[2]]
    code = 0 if not failed else (3 if failed == ["C8"] else 1)
    payload = {"schema_version": "1.0", "chapter": 253, "file": os.path.basename(__file__),
               "card_revision": MIND_CARD["card_revision"],
               "environment": {"python": sys.version.split()[0], "numpy": np.__version__}, "seeds": seeds,
               "runtime_s": round(elapsed, 2), "n_params": n_params(runs[0]["models"]["jubar"]), "gradcheck": grad_record,
               "correctness": [{"id": c, "name": n, "passed": bool(ok), "detail": d} for c, n, ok, d in results],
               "mutants": mutation, "hypotheses": hyps, "knockouts": kos, "task_types": TASK_TYPES, "exit_code": code,
               "excess_harm_ratio": {name: {s: float(np.mean([r["E"][name][s] for r in runs])) for s in splits}
                                     for name, splits in runs[0]["E"].items()}}
    write_report(report_lines(mode, seeds, elapsed, runs, results, grad_record, mutation, hyps, kos, bridge, code),
                 payload, json_path)
    return code


# ---------------------------------------------------------------- command line
def main(argv=None):
    parser = argparse.ArgumentParser(description="Chapter 0253: heard = voice x felt, planned at full voice.")
    parser.add_argument("--quick", action="store_true", help="one seed, reduced steps, all correctness tests")
    parser.add_argument("--seed", type=int, default=0, help="base seed")
    parser.add_argument("--seeds", type=int, default=None, help="number of seeds (default 5, or 1 with --quick)")
    parser.add_argument("--json", default=None, help="write the report as JSON to this path")
    parser.add_argument("--card", action="store_true", help="print MIND_CARD as JSON and exit")
    parser.add_argument("--mutant", default=None, help="run the protocol with a registered mutant active")
    parser.add_argument("--data", default=None, help="optional local CSV for the real-data bridge")
    args = parser.parse_args(argv)
    if args.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    n_seeds = args.seeds if args.seeds is not None else (1 if args.quick else 5)
    if n_seeds < 1 or (args.mutant is not None and args.mutant not in MUTANTS):
        print(f"invalid arguments; registered mutants: {', '.join(MUTANTS)}", file=sys.stderr)
        return 2
    use_mutant(args.mutant)
    try:
        return protocol("quick" if args.quick else "full", args.seed, n_seeds, args.json, args.data)
    except FloatingPointError as exc:
        print(f"non-finite value: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0262 · Al-Sarakhsi
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 14, Minds 261-280 Available on Amazon https://www.amazon.com/dp/B0HLYSVY5D
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0262_al_sarakhsi_1010 - Al-Sarakhsi (c.1010-1090)
# END ATTRIBUTION
"""Chapter 0262 · Al-Sarakhsi · athar-gated cause network (research prototype).

Thesis
    An attribute earns the right to move a verdict only through an effect attested where
    it was seen to act (athar), never through how regularly it travels with the verdict
    (tard); an apparent exception means the cause was mis-stated, and between rival
    causes the stronger effect, not the more visible one, prevails.

Evidence and provenance (belief: his own Usul and Mabsut survive)
    D1 Usul al-Sarakhsi, ed. al-Afghani, vol. 2, Bab al-qiyas: a cause is like a witness;
       suitability (mula'ama) makes it eligible, but its probity ('adala) is known only
       by its athar, "an effect that appeared in some place" other than the disputed case.
    D2 Same chapter: co-presence (ittirad) and co-presence-and-absence (dawaran) prove
       nothing; a condition co-varies with a verdict exactly as a cause does, and dawaran
       would certify intoxicating strength as the cause in the wine case although no one
       extends that verdict to other strong drinks. The heart's impression (ikhala) is
       likewise only conjecture and binds no opponent.
    D3 Same volume, section on qiyas and istihsan: preponderance is by strength of effect,
       not by apparentness or hiddenness. Birds of prey drink with a dry bone beak, so the
       moist-contact cause of impurity is absent although the predator category is present.
    D4 Section on the invalidity of particularising causes (takhsis al-'illa): a verdict is
       absent only because an attribute was added or lost, so the cause itself changed;
       causes admit no exceptions.
    D5 Bab al-qiyas: minority, not virginity, governs a father's power to marry off a
       daughter, because the effect of minority appears in guardianship over property.
    D6 Deeds and scholarship: al-Mabsut and other works dictated without his books while he
       was held at Uzgend, first in a well (c. 466/1073 to 480/1087); Hamidullah, TDV Islam
       Ansiklopedisi 36 (2009); Calder, EI2 IX.

Doctrine -> mechanism -> test
    D1 athar licensing      -> M1 genus-shared sigmoid gate trained only on attested
                               contrast pairs, closed by default      -> C6.2, H-SIG, H-NEC
    D2 tard proves nothing  -> M2 stop-gradient: observational fit never moves the gate
                                                                    -> C6.3 (definition), H-RIVAL
    D3 effect over salience -> task: the effective attribute is observed noisily, the
                               confounder cleanly                  -> H-SIG
    D4 no takhsis           -> M3 no ungated path from any attribute to the ruling -> C6.1
    D5 genus transfer       -> M1 gates shared by the domains (abwab) of a genus;
                               target domains have no attested pairs -> H-SIG
    Blind spot              -> a cause attested as inert in its genus stays shut -> H-BLIND

Research question (robustness to spurious correlation; causal inference)
    Can a small attested record of contrast pairs from sibling domains, used only to
    license input features, protect a domain whose observational data carry a clean
    confounder better than the same record used as extra training data?

Closest prior art and the delta
    Counterfactually augmented data (Kaushik, Hovy & Lipton 2020) is the baseline: the same
    network without a gate, trained on cases plus the attested pairs. Shared-support
    multi-task feature selection (Argyriou, Evgeniou & Pontil 2008; Obozinski, Taskar &
    Jordan 2010) is the rival: the same gate, but opened by fit as well (the tard side).
    Also related: input-gradient constraints (Ross, Hughes & Doshi-Velez 2017) and
    invariant risk minimisation (Arjovsky et al. 2019). Delta: the attested record trains
    one quantity that fit cannot touch, and that quantity is shared across a genus, so it
    licenses features in domains that have no attested record of their own.

Blind spot
    He admitted a cause only through attested effect and refused co-variation and the
    heart's impression as evidence. A genuinely new cause that the record calls inert in
    the sibling domains therefore stays shut in a new domain even when its own clean data
    would reveal it (condition: domain 8).

Task (generative process)
    12 binary or continuous latent attributes; 9 domains in 3 genera (guardianship,
    purity, exchange). Each domain's ruling is a noisy threshold of its genus's causes,
    with domain-specific weights and one interaction (compound causes).
    Hidden causes are observed with noise 0.9; manifest confounders with noise 0.15.
    Domains 2 and 5: a clean confounder agrees with the hidden cause in 88% of training
    cases and disagrees in every case of the shifted split (the minor non-virgin and the
    adult virgin; the predator bird that never wets the water with saliva).
    Domains 0,1,3,4,6,7 contribute attested contrast pairs (one attribute flipped, clearly
    observed, ruling given). Domain 8's cause (attribute 9) is inert in 6 and 7.

Limits
    Synthetic data; binary rulings; genus membership is given, not learned; attested pairs
    are assumed correct. A research prototype of one AGI-relevant mechanism, not an AGI and
    not a replica of a person. No claim here is evidence about the historical jurist.
"""
MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 2,
    "revision_log": [{"revision": 2, "date": "2026-09-21", "reason": (
        "After two 3-seed development pilots and before any full-protocol run: (1) added the contrast term "
        "(beta_contrast) because a gate moved only by attested cross-entropy opened nuisance attributes; (2) an "
        "attested pair now shares one observation of the case and its rulings carry no label noise, so the pair "
        "differs only in the reversed attribute, as the docstring's limits state. Hypotheses, metrics, mesi and "
        "thresholds unchanged.")}],
    "generation": {"template_version": "codeguidelines 1.0 (2026-09-15)", "generator": "Claude",
                   "generator_version": "Claude Opus 5", "date": "2026-09-21"},
    "id": 262, "figure": "Al-Sarakhsi", "born": 1010, "died": 1090,
    "civilization": "Islamic (born in Khurasan; worked in Karakhanid Transoxiana; wrote in Arabic; ethnic origin unrecorded)",
    "provenance": "belief",
    "thesis": ("An attribute earns the right to move a verdict only through an effect attested where it "
               "was seen to act (athar), never through co-variation with the verdict (tard); an apparent "
               "exception is a mis-stated cause; the stronger effect beats the more visible one."),
    "evidence": [
        {"id": "D1", "claim": "The probity of a cause is known by its athar, an effect shown in some place other than the disputed case, as a witness's probity is known",
         "basis": "primary", "source": "Usul al-Sarakhsi, ed. Abu al-Wafa al-Afghani (1372/1953), vol. 2, chapter on qiyas"},
        {"id": "D2", "claim": "Ittirad and dawaran do not establish a cause: a condition co-varies too, and dawaran would certify a false cause in the wine case; ikhala is conjecture",
         "basis": "primary", "source": "Usul al-Sarakhsi, vol. 2, Bab al-qiyas, on the evidence that makes an attribute a cause"},
        {"id": "D3", "claim": "Preponderance by strength of effect, not by apparentness; birds of prey: the moist-contact cause is absent",
         "basis": "primary", "source": "Usul al-Sarakhsi, vol. 2, section on qiyas and istihsan"},
        {"id": "D4", "claim": "Causes admit no particularisation; absence of the verdict means an attribute was added or lost",
         "basis": "primary", "source": "Usul al-Sarakhsi, vol. 2, section on the invalidity of takhsis al-'illa"},
        {"id": "D5", "claim": "Minority, not virginity, is the effective attribute in marriage guardianship because its effect appears in guardianship over property",
         "basis": "primary", "source": "Usul al-Sarakhsi, vol. 2, Bab al-qiyas (examples of effective causes)"},
        {"id": "D6", "claim": "Dictated al-Mabsut and other works without his books while held at Uzgend (first in a well, later in rooms of the fortress), c. 466-480 AH, released 20 Rabi I 480 / 25 June 1087",
         "basis": "scholarship", "source": "Hamidullah, 'Serahsi, Semsuleimme', TDV Islam Ansiklopedisi 36 (2009) 544-547; Calder, 'al-Sarakhsi', EI2 IX, 35-36"},
        {"id": "D7", "claim": "A machine would inherit his refusal of observational evidence as a blind spot for new causes",
         "basis": "speculation", "source": "chapter extrapolation"},
    ],
    "research_question": {"category": "robustness to spurious correlation; causal inference and intervention",
                          "question": ("Does an attested contrast record from sibling domains, used only to license input "
                                       "features through a genus-shared gate, protect a confounded domain better than the same "
                                       "record used as extra training data or a gate that fit may also open?")},
    "mechanism": {
        "name": "athar-gated cause network", "family": "causal feature licensing (gated MLP, stop-gradient split objective)",
        "signature_modules": ["athar_gate"],
        "closest_prior_art": ["counterfactually augmented data (Kaushik, Hovy & Lipton 2020)",
                              "multi-task shared-support feature selection (Argyriou et al. 2008; Obozinski et al. 2010)",
                              "right for the right reasons (Ross, Hughes & Doshi-Velez 2017)",
                              "invariant risk minimisation (Arjovsky et al. 2019)"],
        "overlap": "Medium",
        "prior_art_queries": ["feature gating trained on counterfactual pairs only", "stop-gradient feature mask interventional data",
                              "multi-task shared feature support spurious correlation", "counterfactual data augmentation vs feature masking"],
        "contribution_type": "mechanism",
        "delta": ("The attested record alone trains a genus-shared input gate that observational fit cannot move, so a "
                  "domain with no attested cases inherits licences from its genus."),
    },
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1", "property_test": "C6.2", "hypothesis": "H-SIG"},
        {"doctrine": "D2", "mechanism": "M2", "property_test": "C6.3", "hypothesis": "H-RIVAL"},
        {"doctrine": "D3", "mechanism": "M1", "property_test": "C6.2", "hypothesis": "H-SIG"},
        {"doctrine": "D4", "mechanism": "M3", "property_test": "C6.1", "hypothesis": "H-NEC"},
        {"doctrine": "D5", "mechanism": "M1", "property_test": "C6.2", "hypothesis": "H-SIG"},
        {"doctrine": "D7", "mechanism": "M1", "property_test": "C6.2", "hypothesis": "H-BLIND"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "The athar gate raises accuracy where the clean confounder turns against the hidden cause",
         "metric": "accuracy on the shifted split of target domains 2 and 5", "split": "shifted",
         "comparison": "model - baseline", "direction": "greater", "mesi": 0.08, "seeds": 5},
        {"id": "H-NEC", "statement": "Opening every gate after training hurts the shifted split more than averaging the domain embedding",
         "metric": "accuracy on the shifted split of target domains 2 and 5",
         "comparison": "signature_knockout - matched_knockout", "direction": "less", "mesi": 0.05, "seeds": 5},
        {"id": "H-BLIND", "statement": "A cause attested as inert in the sibling domains cannot be learned in a new domain",
         "condition": "domain 8, whose ruling depends on attribute 9", "grounding": "his refusal of co-variation as evidence leaves no route for unattested causes",
         "metric": "in-distribution accuracy on domain 8", "comparison": "model - baseline", "direction": "less", "mesi": 0.05, "seeds": 5},
        {"id": "H-RIVAL", "statement": "Licensing by attested effect beats licensing by fit (tard) on the shifted split",
         "metric": "accuracy on the shifted split of target domains 2 and 5", "comparison": "model - rival_tard",
         "direction": "greater", "mesi": 0.05, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.35, "margin_over_trivial": 0.12, "shuffled_band": 0.05,
                   "gradcheck_rel_tol": 1e-5, "athar_response_gap": 0.30},
    "probe_predictions": [
        {"probe": "P2", "expected": "equal to baseline"}, {"probe": "P6", "expected": "equal to baseline"},
        {"probe": "P8", "expected": "equal to baseline"}, {"probe": "P9", "expected": "equal to baseline"},
        {"probe": "P10", "expected": "equal to baseline"},
        {"note": "Without an attested record the gate is held open (mastur), so the file reduces to a one-layer network."},
    ],
    "dialectic_links": [
        {"chapter": 197, "relation": "successor", "test": "none (Abu Hanifa's equitable override is not implemented; the "
                                                          "chapter reads istihsan through Sarakhsi's denial that it is an exception)"},
        {"chapter": 207, "relation": "rival", "test": "H-RIVAL (the fit-opened gate stands for the co-variation and "
                                                     "suitability criteria he attributes to ahl al-tard and some Shafi'is, not to al-Shafi'i himself)"},
    ],
    "corpus_neighbors": [
        {"chapter": 197, "similarity": None, "difference": "Abu Hanifa: a purpose-tracking veto that breaks a valid rule; here nothing breaks a rule, the input gate decides which attributes are causes"},
        {"chapter": 207, "similarity": None, "difference": "al-Shafi'i: reconciling attested texts by coherence; here texts license features, they are not reconciled"},
        {"chapter": 259, "similarity": None, "difference": "Ibn Hazm: no analogy at all; here analogy is kept and disciplined by attested effect"},
        {"chapter": 258, "similarity": None, "difference": "Atisha: licence-gated capacity ladder over actions; here gates sit on input attributes and are trained by contrast pairs"},
        {"chapter": 244, "similarity": None, "difference": "Ferdowsi: warrant stored apart from competence; here the warrant is per attribute and per genus, set by interventions"},
        {"chapter": 261, "similarity": None, "difference": "al-Hujwiri: diagnoses shift from the response to a polish; here shift is pre-empted by refusing unlicensed features"},
        {"chapter": 252, "similarity": None, "difference": "al-Biruni: counts independent witnesses; here repetition (ittirad) counts for nothing"},
    ],
    "similarity_review": ("Nearest-neighbour shingle similarity could not be computed: corpus files are not available in "
                          "this session. The standard utilities block follows the section 5.4 specification and was not "
                          "diffed against the canonical block."),
    "barometer": {"cognitive_processing": ["transfer of licensed causes to domains without attested cases"],
                  "embodied_cognition": [], "world_modeling": ["accuracy when a confounder reverses (shifted split)"],
                  "consciousness": [], "language_understanding": [], "emotional_intelligence": [],
                  "creativity": [], "autonomy": []},
    "task_types": ["vector_classification"],
    "applications": [
        {"use": "license features for an observational model from a randomised sample before deployment",
         "sector": "labour-market policy evaluation", "dataset": "LaLonde NSW experimental sample with the Dehejia-Wahba subsample and PSID/CPS comparison groups"},
        {"use": "admissible-feature audit for credit scoring, admitting only features with documented effect",
         "sector": "consumer finance compliance", "dataset": "UCI Statlog German Credit"},
        {"use": "use interventional measurements to gate predictors of cell signalling state",
         "sector": "biomedical research", "dataset": "Sachs et al. 2005 flow-cytometry protein-signalling data"},
    ],
    "safety_notes": ("Synthetic data only. Legal examples are historical illustrations of reasoning, not advice. "
                     "Discredited social hierarchies in the source era are not encoded as features about people."),
    "hyperparameters": {"hidden": 16, "baseline_hidden": 17, "steps": 700, "quick_steps": 260, "lr": 0.02,
                        "batch": 512, "alpha_attested": 1.0, "beta_contrast": 1.0, "rho_prior": 0.03, "clip_norm": 5.0},
}

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

HP = MIND_CARD["hyperparameters"]
TH = MIND_CARD["thresholds"]
EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_BUDGET, EXIT_NONFINITE = 0, 1, 2, 3, 4
BUDGET_FULL_S, BUDGET_QUICK_S = 180.0, 20.0

# BEGIN STANDARD UTILITIES v1.0
def softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def logsumexp(z, axis=-1):
    m = np.max(z, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(z - m), axis=axis, keepdims=True))).squeeze(axis)


def softplus(x):
    return np.logaddexp(0.0, x)


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def adam_init(params):
    return {"t": 0, "m": {k: np.zeros_like(v) for k, v in params.items()},
            "v": {k: np.zeros_like(v) for k, v in params.items()}}


def adam_step(params, grads, state, lr, b1=0.9, b2=0.999, eps=1e-8, sign=-1.0):
    state["t"] += 1
    t = state["t"]
    for k in params:
        state["m"][k] = b1 * state["m"][k] + (1 - b1) * grads[k]
        state["v"][k] = b2 * state["v"][k] + (1 - b2) * grads[k] ** 2
        mh = state["m"][k] / (1 - b1 ** t)
        vh = state["v"][k] / (1 - b2 ** t)
        params[k] += sign * lr * mh / (np.sqrt(vh) + eps)


def clip_global_norm(grads, max_norm):
    total = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    if total > max_norm:
        for k in grads:
            grads[k] = grads[k] * (max_norm / (total + 1e-12))
    return total


def finite_difference_check(f, params, grads, rng, eps=1e-6, n_entries=20, floor=1e-4):
    """Central differences on n_entries random entries plus the largest-gradient entry of every
    tensor. The relative-error denominator is floored so entries below finite-difference
    resolution are judged on absolute error (floor * tol)."""
    out = {}
    for k, p in params.items():
        flat_g = grads[k].ravel()
        idx = set(rng.choice(p.size, size=min(n_entries, p.size), replace=False).tolist())
        idx.add(int(np.argmax(np.abs(flat_g))))
        worst = 0.0
        for i in sorted(idx):
            orig = p.flat[i]
            p.flat[i] = orig + eps
            fp = f()
            p.flat[i] = orig - eps
            fm = f()
            p.flat[i] = orig
            num = (fp - fm) / (2 * eps)
            rel = abs(num - flat_g[i]) / max(abs(num) + abs(flat_g[i]), floor)
            worst = max(worst, rel)
        out[k] = worst
    return out


def paired_bootstrap(diffs, rng, n_resamples=2000):
    d = np.asarray(diffs, dtype=float)
    idx = rng.integers(0, len(d), size=(n_resamples, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def verdict(mean, ci, direction, mesi):
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


class NonFinite(RuntimeError):
    pass


# ---------------------------------------------------------------- data and tasks
N_ATTR, N_DOM = 12, 9
GENUS_OF = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
ATTESTED_DOMAINS = (0, 1, 3, 4, 6, 7)
TARGET_DOMAINS = (2, 5)
NOVEL_DOMAIN = 8
ATTR_NAMES = ("minority", "virginity", "sanity", "kin_proximity", "moist_contact", "affliction",
              "predator_kind", "measure", "same_kind", "novel_attribute", "season", "colour")
# Hidden causes (minority, moist contact) are hard to observe; their confounders are easy.
OBS_NOISE = np.array([0.9, 0.15, 0.5, 0.4, 0.9, 0.5, 0.15, 0.5, 0.5, 0.4, 0.3, 0.3])
CONFOUND = {2: (0, 1), 5: (4, 6)}  # domain: (hidden cause, manifest confounder)
AGREE_RATE = 0.88
TEXT_NOISE = 0.1  # attested cases name their attributes clearly
# domain: (linear weights, interaction (i, j, w), bias); only genus causes enter a rule
RULES = {
    0: ({0: 1.4, 2: -0.8, 3: 0.5}, (0, 2, 0.0), 0.1),
    1: ({0: 1.2, 3: -0.6}, (0, 2, 0.7), -0.2),
    2: ({0: 1.5, 2: -0.7}, (0, 3, 0.6), 0.1),
    3: ({4: 1.4, 5: -0.9}, (4, 5, 0.0), 0.1),
    4: ({4: 1.1, 5: -0.8}, (4, 5, 0.6), -0.1),
    5: ({4: 1.5, 5: -0.8}, (4, 5, 0.0), 0.2),
    6: ({7: 0.5}, (7, 8, 1.4), -0.2),
    7: ({8: 1.2}, (7, 8, 0.6), 0.1),
    8: ({9: 1.5}, (7, 8, 0.6), -0.1),
}


def ruling_score(a, d):
    lin, (i, j, w), b = RULES[d]
    s = np.full(len(a), b) + w * a[:, i] * a[:, j]
    for k, c in lin.items():
        s = s + c * a[:, k]
    return s


def sample_latent(n, d, rng, split):
    a = rng.choice(np.array([-1.0, 1.0]), size=(n, N_ATTR))
    a[:, 3] = rng.normal(size=n)
    if d in CONFOUND and split != "attested":
        c, m = CONFOUND[d]
        if split == "shift":
            a[:, m] = -a[:, c]
        else:
            agree = rng.random(n) < AGREE_RATE
            a[:, m] = np.where(agree, a[:, c], -a[:, c])
    return a


def observe(a, noise, rng):
    return a + noise * rng.normal(size=a.shape)


def make_split(domains, n, rng, split, label_noise=0.3):
    xs, ds, ys = [], [], []
    for d in domains:
        a = sample_latent(n, d, rng, split)
        xs.append(observe(a, OBS_NOISE, rng))
        ds.append(np.full(n, d))
        ys.append((ruling_score(a, d) + label_noise * rng.normal(size=n) > 0).astype(int))
    return {"X": np.concatenate(xs), "dom": np.concatenate(ds), "y": np.concatenate(ys)}


def make_attested(domains, n_pairs, rng, rule_override=None):
    """Contrast pairs: the same case with one attribute reversed, clearly observed, ruling given
    without noise (attested pairs are assumed correct). rule_override lets property tests swap in
    a jurisprudence where a different attribute is the effective one."""
    score = rule_override or ruling_score
    parts = {"X0": [], "X1": [], "dom": [], "y0": [], "y1": [], "k": []}
    for d in domains:
        a = sample_latent(n_pairs, d, rng, "attested")
        k = rng.integers(0, N_ATTR, size=n_pairs)
        a2 = a.copy()
        a2[np.arange(n_pairs), k] *= -1.0
        x0 = observe(a, TEXT_NOISE, rng)
        parts["X0"].append(x0)
        parts["X1"].append(x0 + (a2 - a))  # one observation of one case: only attribute k differs
        parts["y0"].append((score(a, d) > 0).astype(int))
        parts["y1"].append((score(a2, d) > 0).astype(int))
        parts["dom"].append(np.full(n_pairs, d))
        parts["k"].append(k)
    att = {key: np.concatenate(v) for key, v in parts.items()}
    # flattened view: attested cases also serve as ordinary precedents for how causes act
    att.update({"X": np.concatenate([att["X0"], att["X1"]]), "y": np.concatenate([att["y0"], att["y1"]]),
                "domf": np.concatenate([att["dom"], att["dom"]])})
    return att


def make_world(seed, n_train=300, n_test=300, n_shift=400, n_pairs=48):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 262, 1]))
    return {"train": make_split(range(N_DOM), n_train, rng, "train"),
            "test": make_split(range(N_DOM), n_test, rng, "test"),
            "shift": make_split(TARGET_DOMAINS, n_shift, rng, "shift"),
            "att": make_attested(ATTESTED_DOMAINS, n_pairs, rng)}


def domain_majority_accuracy(train, ev):
    """Trivial baseline: predict each evaluation case's domain majority from training."""
    maj = {d: int(train["y"][train["dom"] == d].mean() >= 0.5) for d in np.unique(train["dom"])}
    pred = np.array([maj.get(int(d), 1) for d in ev["dom"]])
    return float(np.mean(pred == ev["y"]))


# ---------------------------------------------------------------- model
TASK_TYPES = ["vector_classification"]
GATE_MODES = ("athar", "tard", "open", "none")  # athar = his side; tard = rival; open = mastur; none = baseline


def build_model(in_dim, out_dim, task_type, rng, **cfg):
    """One tanh layer shared by all domains, a per-domain embedding and readout, and (except for
    the baseline) a [0,1] gate per genus and attribute multiplying every input before the layer."""
    if task_type not in TASK_TYPES:
        raise ValueError("unsupported task type " + str(task_type))
    genus_of = np.asarray(cfg.get("genus_of", [0] * cfg.get("n_domains", 1)))
    n_dom, n_gen = len(genus_of), int(genus_of.max()) + 1
    h, c = cfg.get("hidden", HP["hidden"]), max(int(out_dim), 2)
    gate = cfg.get("gate", "athar")
    params = {"W1": rng.normal(0, 1 / math.sqrt(in_dim), (in_dim, h)), "b1": np.zeros(h),
              "E": rng.normal(0, 0.1, (n_dom, h)), "w2": rng.normal(0, 1 / math.sqrt(h), (n_dom, h, c)),
              "b2": np.zeros((n_dom, c))}
    if gate != "none":
        # 0.5: acting on an attribute before its effect is shown is permitted, not obligatory (mastur)
        params["gate"] = np.full((n_gen, in_dim), 0.5)
    return {"params": params, "gate": gate, "genus_of": genus_of, "mutant": cfg.get("mutant"),
            "alpha": cfg.get("alpha", HP["alpha_attested"]), "beta": cfg.get("beta", HP["beta_contrast"]),
            "rho": cfg.get("rho", HP["rho_prior"]), "lam_override": None, "bypass": False}


def copy_model(model):
    m = dict(model)
    m["params"] = {k: v.copy() for k, v in model["params"].items()}
    return m


def gate_rows(model, dom, gate=None):
    if model["gate"] in ("none", "open"):
        return None
    lam = model["lam_override"] if model["lam_override"] is not None else (
        model["params"]["gate"] if gate is None else gate)
    return lam[model["genus_of"][dom]]


def forward(model, X, dom, gate=None):
    p = model["params"]
    lam = gate_rows(model, dom, gate)
    # M3 (no takhsis): with a gate present, no attribute reaches the layer except through it.
    U = X if (lam is None or model["bypass"]) else lam * X
    Hh = np.tanh(U @ p["W1"] + p["b1"] + p["E"][dom])
    Z = np.einsum("nh,nhc->nc", Hh, p["w2"][dom]) + p["b2"][dom]
    if not np.all(np.isfinite(Z)):
        raise NonFinite("non-finite logits")
    return Z, (X, dom, U, lam, Hh)


def backward(model, cache, dZ, gate_grad):
    """Hand-derived reverse pass; gate_grad decides whether this term may move the gate."""
    p = model["params"]
    X, dom, U, lam, Hh = cache
    g = {k: np.zeros_like(v) for k, v in p.items()}
    np.add.at(g["b2"], dom, dZ)
    np.add.at(g["w2"], dom, Hh[:, :, None] * dZ[:, None, :])
    dA = np.einsum("nc,nhc->nh", dZ, p["w2"][dom]) * (1.0 - Hh ** 2)
    g["W1"] = U.T @ dA
    g["b1"] = dA.sum(axis=0)
    np.add.at(g["E"], dom, dA)
    if lam is not None and gate_grad and model["lam_override"] is None and not model["bypass"]:
        np.add.at(g["gate"], model["genus_of"][dom], (dA @ p["W1"].T) * X)
    return g


def ce_term(model, X, dom, y, gate_grad, gate=None):
    Z, cache = forward(model, X, dom, gate)
    n = len(y)
    loss = float(np.mean(logsumexp(Z) - Z[np.arange(n), y]))
    dZ = softmax(Z)
    dZ[np.arange(n), y] -= 1.0
    return loss, backward(model, cache, dZ / n, gate_grad)


def contrast_term(model, att):
    """Athar: where reversing one attribute changed the attested ruling, the model's margin must
    move the same way. Unchanged pairs are not evidence: the absence of a verdict where a cause
    is absent neither proves nor refutes it (D1-D2)."""
    ch = att["y0"] != att["y1"]
    zeros = {k: np.zeros_like(v) for k, v in model["params"].items()}
    if not np.any(ch):
        return 0.0, zeros
    s = np.where(att["y1"][ch] > att["y0"][ch], 1.0, -1.0)
    Z0, c0 = forward(model, att["X0"][ch], att["dom"][ch])
    Z1, c1 = forward(model, att["X1"][ch], att["dom"][ch])
    delta = (Z1[:, 1] - Z1[:, 0]) - (Z0[:, 1] - Z0[:, 0])
    n = int(ch.sum())
    loss = float(np.mean(softplus(-s * delta)))
    dd = -s * sigmoid(-s * delta) / n
    dZ1 = np.zeros_like(Z1)
    dZ1[:, 1], dZ1[:, 0] = dd, -dd
    g1, g0 = backward(model, c1, dZ1, True), backward(model, c0, -dZ1, True)
    return loss, {k: g1[k] + g0[k] for k in zeros}


def has_gate_prior(model):
    return model["gate"] in ("athar", "tard") and model["lam_override"] is None


def objective_value(model, batch, frozen_gate=None):
    """Declared objective J = CE_fit + alpha*CE_attested + beta*Contrast + rho*mean(gate). On his
    side the gate inside both CE terms is a stop-gradient copy (frozen_gate); only the contrast
    term and the prior see the live gate."""
    froz = frozen_gate if model["gate"] == "athar" else None
    total = ce_term(model, batch["X"], batch["dom"], batch["y"], False, froz)[0]
    att = batch.get("att")
    if att is not None:
        total += model["alpha"] * ce_term(model, att["X"], att["domf"], att["y"], False, froz)[0]
        total += model["beta"] * contrast_term(model, att)[0]
    if has_gate_prior(model):
        total += model["rho"] * float(np.mean(model["params"]["gate"]))
    return total


def loss_and_grads(model, batch):
    """Returns J and a gradient dict with exactly the keys of params (mutants applied here)."""
    athar, mut = model["gate"] == "athar", model["mutant"]
    loss, g = ce_term(model, batch["X"], batch["dom"], batch["y"], (not athar) or mut == "tard_leak")
    att = batch.get("att")
    if att is not None:
        la, ga = ce_term(model, att["X"], att["domf"], att["y"], not athar)
        lc, gc = contrast_term(model, att)
        loss += model["alpha"] * la + model["beta"] * lc
        for k in g:
            g[k] += model["alpha"] * ga[k] + model["beta"] * gc[k]
    if has_gate_prior(model):
        # D1: an attribute may not be named a cause without proof -- every gate is pulled shut.
        loss += model["rho"] * float(np.mean(model["params"]["gate"]))
        g["gate"] += model["rho"] / model["params"]["gate"].size
    if mut == "zero_grad_W1":
        g["W1"][:] = 0.0
    if mut == "zero_grad_gate" and "gate" in g:
        g["gate"][:] = 0.0
    return loss, g


def predict(model, X, dom=None):
    dom = np.zeros(len(X), dtype=int) if dom is None else dom
    return softmax(forward(model, X, dom)[0])


def hidden_states(model, X, dom=None):
    dom = np.zeros(len(X), dtype=int) if dom is None else dom
    return forward(model, X, dom)[1][4]


def accuracy(model, split, mask=None):
    m = np.ones(len(split["y"]), bool) if mask is None else mask
    pr = predict(model, split["X"][m], split["dom"][m])
    return float(np.mean(np.argmax(pr, axis=1) == split["y"][m]))


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


# ---------------------------------------------------------------- baselines and rival
def build_trio(seq, mutant=None):
    """His model; the size-matched baseline (no gate, one more hidden unit) trained on the same
    cases, attested precedents and contrasts (counterfactual augmentation); and the rival whose
    shared gate is also opened by observational fit (the side of ahl al-tard). A registered
    mutant, when given, breaks only his model."""
    r = [np.random.default_rng(s) for s in seq.spawn(3)]
    vc = "vector_classification"
    return {"model": build_model(N_ATTR, 2, vc, r[0], gate="athar", genus_of=GENUS_OF, mutant=mutant),
            "baseline": build_model(N_ATTR, 2, vc, r[1], gate="none", genus_of=GENUS_OF, hidden=HP["baseline_hidden"]),
            "rival_tard": build_model(N_ATTR, 2, vc, r[2], gate="tard", genus_of=GENUS_OF)}


# ---------------------------------------------------------------- registries
def modules(model):
    reg = {"cause_layer": {"params": ["W1", "b1"], "role": "shared tanh hidden layer over gated inputs", "signature": False},
           "bab_embedding": {"params": ["E"], "role": "per-domain additive embedding of hidden pre-activations", "signature": False},
           "ruling_readout": {"params": ["w2", "b2"], "role": "per-domain linear softmax readout", "signature": False}}
    if "gate" in model["params"]:
        reg["athar_gate"] = {"params": ["gate"], "signature": model["gate"] == "athar",
                             "role": "genus-shared [0,1] input gate with L1 prior, moved only by contrast pairs (stop-gradient from fit)"}
    return reg


def knockout(model, name, mode):
    m = copy_model(model)
    p = m["params"]
    if name == "athar_gate":
        lam = p["gate"]
        m["lam_override"] = {"identity": np.ones_like(lam), "zero": np.zeros_like(lam),
                             "mean": np.full_like(lam, lam.mean())}[mode]
    elif name == "bab_embedding":
        p["E"] = np.zeros_like(p["E"]) if mode in ("zero", "identity") else np.repeat(p["E"].mean(0, keepdims=True), len(p["E"]), 0)
    elif name == "cause_layer":
        if mode == "zero":
            p["W1"][:], p["b1"][:] = 0.0, 0.0
        else:
            p["W1"] = np.repeat(p["W1"].mean(0, keepdims=True), len(p["W1"]), 0)
    elif name == "ruling_readout":
        for k in ("w2", "b2"):
            p[k] = np.zeros_like(p[k]) if mode == "zero" else np.repeat(p[k].mean(0, keepdims=True), len(p[k]), 0)
    else:
        raise KeyError(name)
    return m


MUTANTS = {"sign_flip": "optimizer ascends the objective", "zero_lr": "learning rate zero",
           "zero_grad_W1": "analytic gradient of W1 zeroed", "zero_grad_gate": "analytic gradient of the gate zeroed",
           "tard_leak": "observational fit is allowed to move the athar gate (breaks M2)"}


# ---------------------------------------------------------------- training
def fit(model, data, budget, rng):
    """Projected Adam: minibatches of observational cases plus the whole attested record each
    step; gates are clipped to [0,1] after every step so a closed gate is exactly closed."""
    tr, att = data["train"], data.get("att")
    if model["gate"] == "athar" and (att is None or len(att["y0"]) == 0):
        model["gate"] = "open"  # no attested record: attributes stay provisionally admissible (mastur)
    state = adam_init(model["params"])
    lr = 0.0 if model["mutant"] == "zero_lr" else HP["lr"]
    sign = 1.0 if model["mutant"] == "sign_flip" else -1.0
    hist, n = [], len(tr["y"])
    for _ in range(budget):
        idx = rng.choice(n, size=min(HP["batch"], n), replace=False)
        loss, g = loss_and_grads(model, {"X": tr["X"][idx], "dom": tr["dom"][idx], "y": tr["y"][idx], "att": att})
        if not math.isfinite(loss):
            raise NonFinite("non-finite loss")
        clip_global_norm(g, HP["clip_norm"])
        adam_step(model["params"], g, state, lr, sign=sign)
        if model["gate"] in ("athar", "tard"):
            np.clip(model["params"]["gate"], 0.0, 1.0, out=model["params"]["gate"])
        hist.append(loss)
    return hist


# ---------------------------------------------------------------- tests: correctness
KNOCKOUT_PLAN = (("athar_gate", "identity"), ("bab_embedding", "mean"), ("cause_layer", "mean"),
                 ("ruling_readout", "mean"))


def child_rng(*key):
    return np.random.default_rng(np.random.SeedSequence([262, *key]))


def train_trio(seed, steps, mutant=None):
    world = make_world(seed)
    trio = build_trio(np.random.SeedSequence([seed, 262, 2]), mutant)
    hists = {name: fit(m, world, steps, child_rng(seed, 3, i)) for i, (name, m) in enumerate(trio.items())}
    return world, trio, hists


def gradcheck_batch(world, rng, n=64):
    tr = world["train"]
    idx = rng.choice(len(tr["y"]), size=n, replace=False)
    return {"X": tr["X"][idx], "dom": tr["dom"][idx], "y": tr["y"][idx], "att": world["att"]}


def gradcheck(model, batch, rng):
    """Analytic gradients from loss_and_grads against central differences of the declared objective,
    with the gate inside the fit terms held as a stop-gradient copy (M2)."""
    frozen = model["params"]["gate"].copy() if "gate" in model["params"] else None
    _, g = loss_and_grads(model, batch)
    return finite_difference_check(lambda: objective_value(model, batch, frozen), model["params"], g, rng)


def test_gradients(seed, rng, mutant=None, after=True, names=("model", "baseline", "rival_tard")):
    """C1 on every tensor of the named networks, at initialisation and after 60 training steps."""
    world = make_world(seed)
    full = build_trio(np.random.SeedSequence([seed, 262, 4]), mutant)
    trio = {n: full[n] for n in names}
    stages = ("init", "after_training_steps") if after else ("init",)
    worst, checked = 0.0, set()
    for stage in stages:
        for i, (name, m) in enumerate(trio.items()):
            if stage != "init":
                fit(m, world, 60, child_rng(seed, 5, i))
            errs = gradcheck(m, gradcheck_batch(world, rng), rng)
            worst = max(worst, max(errs.values()))
            checked.update(name + "." + k for k in errs)
    return {"tensors_checked": len(checked), "tensors_total": sum(len(m["params"]) for m in trio.values()),
            "max_rel_error": worst, "checked_at": list(stages), "passed": bool(worst <= TH["gradcheck_rel_tol"])}


def test_determinism(seed, mutant=None):
    """C2: two runs from one seed give identical losses and outputs; everything stays finite."""
    runs = []
    for _ in range(2):
        world = make_world(seed)
        m = build_trio(np.random.SeedSequence([seed, 262, 6]), mutant)["model"]
        hist = fit(m, world, 40, child_rng(seed, 7))
        runs.append((hist, predict(m, world["test"]["X"], world["test"]["dom"]), m, world))
    same = runs[0][0] == runs[1][0] and np.array_equal(runs[0][1], runs[1][1])
    m, world = runs[0][2], runs[0][3]
    finite = all(np.all(np.isfinite(v)) for v in m["params"].values()) and bool(
        np.all(np.isfinite(hidden_states(m, world["test"]["X"], world["test"]["dom"]))))
    return bool(same and finite), f"identical losses and outputs: {same}; parameters and activations finite: {finite}"


def learning_check(model, hist, world):
    """C3: the objective falls by the declared fraction and held-out accuracy clears the trivial
    baseline (domain majority) by the declared margin."""
    k = max(1, len(hist) // 20)
    drop = (hist[0] - float(np.mean(hist[-k:]))) / hist[0]
    acc = accuracy(model, world["test"])
    triv = domain_majority_accuracy(world["train"], world["test"])
    ok = drop >= TH["loss_drop_fraction"] and acc >= triv + TH["margin_over_trivial"]
    return bool(ok), f"loss drop {drop:.3f} (need {TH['loss_drop_fraction']}); test accuracy {acc:.3f} vs trivial {triv:.3f} (need +{TH['margin_over_trivial']})"


def shuffled_world(world, rng):
    """Every target permuted: observational rulings and both rulings of every attested pair."""
    w = dict(world)
    w["train"] = dict(world["train"], y=rng.permutation(world["train"]["y"]))
    att = dict(world["att"], y0=rng.permutation(world["att"]["y0"]), y1=rng.permutation(world["att"]["y1"]))
    att["y"] = np.concatenate([att["y0"], att["y1"]])
    w["att"] = att
    return w


def test_shuffled(seed, steps):
    """C4: trained on permuted targets, held-out accuracy must stay inside the trivial band."""
    world = make_world(seed)
    m = build_trio(np.random.SeedSequence([seed, 262, 11]))["model"]
    fit(m, shuffled_world(world, child_rng(seed, 12)), steps, child_rng(seed, 13))
    acc, triv = accuracy(m, world["test"]), domain_majority_accuracy(world["train"], world["test"])
    ok = acc <= triv + TH["shuffled_band"]
    return bool(ok), f"test accuracy {acc:.3f} vs trivial {triv:.3f} (band +{TH['shuffled_band']})"


def test_mutants(seed, steps, rng):
    """C5: each registered mutant must make C1 or C3 fail."""
    caught = {}
    for name in MUTANTS:
        c1, c3 = test_gradients(seed, rng, mutant=name, after=False, names=("model",))["passed"], True
        if c1:  # a mutant already caught by C1 needs no training run
            world = make_world(seed)
            m = build_trio(np.random.SeedSequence([seed, 262, 14]), name)["model"]
            try:
                c3 = learning_check(m, fit(m, world, steps, child_rng(seed, 15)), world)[0]
            except NonFinite:
                c3 = False
        caught[name] = "C1" if not c1 else ("C3" if not c3 else "")
    return caught


def ungated_violation(rng, bypass, trials=64):
    """Largest logit change when an attribute whose gate is shut is driven to extreme values, on
    random parameters, random open gates and random inputs."""
    m = build_model(N_ATTR, 2, "vector_classification", rng, gate="athar", genus_of=GENUS_OF)
    p = m["params"]
    p["W1"] *= 3.0
    p["gate"] = rng.random(p["gate"].shape)
    m["bypass"] = bypass
    worst = 0.0
    for _ in range(trials):
        g, k = int(rng.integers(3)), int(rng.integers(N_ATTR))
        p["gate"][g, k] = 0.0
        dom = rng.choice(np.flatnonzero(GENUS_OF == g), size=32)
        X = 2.0 * rng.normal(size=(32, N_ATTR))
        X2 = X.copy()
        X2[:, k] = 50.0 * rng.normal(size=32)
        worst = max(worst, float(np.max(np.abs(forward(m, X2, dom)[0] - forward(m, X, dom)[0]))))
        p["gate"][g, k] = rng.random()
    return worst


def swap_rule(a, d):
    """Counterfactual jurisprudence for C6.2: in guardianship, virginity acts where minority acted."""
    if GENUS_OF[d] == 0:
        a = a[:, [1, 0] + list(range(2, N_ATTR))]
    return ruling_score(a, d)


def athar_response_gap(seed, steps, beta=None):
    """Train on the true attested record and on one where minority and virginity trade roles; the
    guardianship gates on those two attributes must trade places (observational cases unchanged)."""
    lam = []
    for rule in (None, swap_rule):
        world = make_world(seed)
        world["att"] = make_attested(ATTESTED_DOMAINS, 48, child_rng(seed, 8), rule)
        cfg = {} if beta is None else {"beta": beta}
        m = build_model(N_ATTR, 2, "vector_classification", child_rng(seed, 9), gate="athar", genus_of=GENUS_OF, **cfg)
        fit(m, world, steps, child_rng(seed, 10))
        lam.append(m["params"]["gate"][0, :2].copy())
    return float(min(lam[0][0] - lam[1][0], lam[1][1] - lam[0][1]))


def stop_gradient_definition(seed, mutant=None):
    """C6.3 definition check (true by construction, not evidence): with no attested record in the
    batch, his gate receives only the prior's constant pull; the rival's gate receives fit gradient."""
    tr = make_world(seed)["train"]
    trio = build_trio(np.random.SeedSequence([seed, 262, 16]), mutant)
    batch = {"X": tr["X"], "dom": tr["dom"], "y": tr["y"]}
    pull = HP["rho_prior"] / trio["model"]["params"]["gate"].size
    ga = np.max(np.abs(loss_and_grads(trio["model"], batch)[1]["gate"] - pull))
    gr = np.max(np.abs(loss_and_grads(trio["rival_tard"], batch)[1]["gate"] - pull))
    return bool(ga == 0.0 and gr > 1e-6), f"his fit-to-gate gradient {ga:.1e}; rival's {gr:.1e}"


def test_properties(seed, steps, rng, mutant=None):
    """C6: each invariant with its negative control."""
    v_ok, v_bad = ungated_violation(rng, False), ungated_violation(rng, True)
    gaps = [athar_response_gap(seed + i, steps) for i in range(2)]
    ctrl = athar_response_gap(seed, steps, beta=0.0)
    ok62 = min(gaps) >= TH["athar_response_gap"] and ctrl < TH["athar_response_gap"]
    ok63, det63 = stop_gradient_definition(seed, mutant)
    return [("C6.1", "no_ungated_path", bool(v_ok <= 1e-12 and v_bad > 1e-3),
             f"max logit change through a shut gate {v_ok:.1e}; negative control (ungated bypass) {v_bad:.2f}"),
            ("C6.2", "athar_response_swaps_gates", bool(ok62),
             f"gate swap gap {min(gaps):.2f} over 2 seeds (need {TH['athar_response_gap']}); negative control without contrast term {ctrl:.2f}"),
            ("C6.3", "stop_gradient_definition_check", ok63, det63 + " [definition check, not evidence]")]


def split_integrity(world):
    """C7: no case of the training or attested record reappears in a held-out split, and the target
    and novel domains have no attested pairs."""
    def keys(X):
        return {hashlib.sha256(np.round(r, 9).tobytes()).hexdigest() for r in X}
    seen = keys(world["train"]["X"]) | keys(world["att"]["X"])
    held = keys(world["test"]["X"]) | keys(world["shift"]["X"])
    clean = not np.isin(world["att"]["dom"], TARGET_DOMAINS + (NOVEL_DOMAIN,)).any()
    overlap = len(seen & held)
    return bool(overlap == 0 and clean), f"{overlap} shared cases; target and novel domains unattested: {clean}"


# ---------------------------------------------------------------- tests: hypotheses
HYPOTHESIS_DIFFS = {
    "H-SIG": lambda o: o["acc"]["model"]["shift"] - o["acc"]["baseline"]["shift"],
    "H-NEC": lambda o: o["ko"]["athar_gate"] - o["ko"]["bab_embedding"],
    "H-BLIND": lambda o: o["acc"]["model"]["novel"] - o["acc"]["baseline"]["novel"],
    "H-RIVAL": lambda o: o["acc"]["model"]["shift"] - o["acc"]["rival_tard"]["shift"],
}


def seed_outcomes(world, trio):
    te, sh = world["test"], world["shift"]
    novel = te["dom"] == NOVEL_DOMAIN
    acc = {n: {"test": accuracy(m, te), "shift": accuracy(m, sh), "novel": accuracy(m, te, novel)}
           for n, m in trio.items()}
    ko = {name: accuracy(knockout(trio["model"], name, mode), sh) for name, mode in KNOCKOUT_PLAN}
    return {"acc": acc, "ko": ko, "gate": trio["model"]["params"]["gate"].round(2).tolist()}


def evaluate_hypotheses(outcomes, rng, evaluated):
    rows = []
    for h in MIND_CARD["hypotheses"]:
        row = {"id": h["id"], "metric": h["metric"], "mean_diff": None, "ci95": None, "mesi": h["mesi"],
               "n_seeds": len(outcomes), "verdict": "not evaluated"}
        if evaluated:
            mean, ci = paired_bootstrap([HYPOTHESIS_DIFFS[h["id"]](o) for o in outcomes], rng)
            row.update(mean_diff=mean, ci95=list(ci), verdict=verdict(mean, ci, h["direction"], h["mesi"]))
        rows.append(row)
    return rows


def knockout_table(outcomes, rng, registry):
    rows = []
    for name, mode in KNOCKOUT_PLAN:
        mean, ci = paired_bootstrap([o["ko"][name] - o["acc"]["model"]["shift"] for o in outcomes], rng)
        rows.append({"module": name, "mode": mode, "signature": registry[name]["signature"],
                     "role": registry[name]["role"], "metric_change": mean, "ci95": list(ci)})
    return rows


# ---------------------------------------------------------------- report
def data_bridge(path, rng):
    """Optional real-data bridge: a local numeric CSV with a header row and a binary outcome in the
    last column (for example the LaLonde NSW sample). Real data carry no attested contrast pairs, so
    the gate stays provisionally open (mastur) and the file reduces to its plain network."""
    if not path:
        return "data bridge skipped: no --data path given"
    if not os.path.exists(path):
        return f"data bridge skipped: {path} not found (no downloads are attempted)"
    raw = np.genfromtxt(path, delimiter=",", skip_header=1)
    raw = raw[np.all(np.isfinite(raw), axis=1)]
    X, y = raw[:, :-1], (raw[:, -1] > 0).astype(int)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    perm = rng.permutation(len(y))
    tr, te = perm[: int(0.7 * len(y))], perm[int(0.7 * len(y)):]
    m = build_model(X.shape[1], 2, "vector_classification", rng)
    fit(m, {"train": {"X": X[tr], "dom": np.zeros(len(tr), int), "y": y[tr]}}, HP["quick_steps"], rng)
    acc = float(np.mean(np.argmax(predict(m, X[te]), axis=1) == y[te]))
    maj = float(max(np.mean(y[te]), 1.0 - np.mean(y[te])))
    return f"data bridge: {os.path.basename(path)}, test accuracy {acc:.3f} vs majority {maj:.3f}, gate mode {m['gate']}"


def fmt_ci(ci):
    return "n/a" if ci is None else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def print_report(rep, extra):
    print(f"=== VERIFIED REPORT · chapter {rep['chapter']:04d} ===")
    print(f"file {rep['file']} · card revision {rep['card_revision']} · {MIND_CARD['figure']} ({MIND_CARD['born']}-{MIND_CARD['died']})")
    print(f"environment: Python {rep['environment']['python']} · NumPy {rep['environment']['numpy']}")
    print(f"seeds {rep['seeds']} · {extra['steps']} steps per network · runtime {rep['runtime_s']:.1f} s (budget {extra['budget']:.0f} s)")
    print("parameters: " + " · ".join(f"{k} {v}" for k, v in extra["sizes"].items()))
    g = rep["gradcheck"]
    print(f"gradient check: {g['tensors_checked']}/{g['tensors_total']} tensors, max relative error {g['max_rel_error']:.2e}, "
          f"at {' and '.join(g['checked_at'])}: {'PASS' if g['passed'] else 'FAIL'}")
    print("correctness:")
    for c in rep["correctness"]:
        print(f"  {c['id']:<5}{c['name']:<32}{'PASS' if c['passed'] else 'FAIL'}  {c['detail']}")
    mu = rep["mutants"]
    print(f"mutation score {mu['detected']}/{mu['total']} = {mu['score']:.2f}  (" +
          ", ".join(f"{k}: {v or 'missed'}" for k, v in extra["caught"].items()) + ")")
    print("accuracy per seed (test all domains | shifted split, domains 2 and 5 | novel domain 8):")
    for s, o in zip(rep["seeds"], rep["per_seed"]):
        print(f"  seed {s}: " + " · ".join(f"{n} {a['test']:.3f} | {a['shift']:.3f} | {a['novel']:.3f}" for n, a in o["acc"].items()))
    print("hypotheses (paired per-seed differences, 95% percentile bootstrap, 2000 resamples):")
    for h in rep["hypotheses"]:
        md = "n/a" if h["mean_diff"] is None else f"{h['mean_diff']:+.3f}"
        print(f"  {h['id']:<8} mean {md}  CI95 {fmt_ci(h['ci95'])}  mesi {h['mesi']}  n={h['n_seeds']}  {h['verdict']}")
    print("knockouts (change in shifted-split accuracy against the intact model):")
    for k in rep["knockouts"]:
        tag = "signature" if k["signature"] else "non-signature"
        print(f"  {k['module']:<16}{k['mode']:<9}{tag:<14}{k['metric_change']:+.3f}  CI95 {fmt_ci(k['ci95'])}  ({k['role']})")
    print("learned gate of his model, seed 0 (rows: guardianship, purity, exchange; columns: " + ", ".join(ATTR_NAMES) + "):")
    for row in rep["per_seed"][0]["gate"]:
        print("  " + " ".join(f"{v:4.2f}" for v in row))
    print(extra["bridge"])
    print(f"task types: {', '.join(rep['task_types'])}")
    print(f"exit code {rep['exit_code']}")
    print("=== END REPORT ===")


# ---------------------------------------------------------------- command line
def correctness_suite(seed, steps, rng, mutant, first):
    """C1-C7 (C8 is judged on the finished run). first = (world, trio, histories) of the first seed."""
    grad = test_gradients(seed, rng, mutant)
    rows = [("C1", "gradient_check", grad["passed"],
             f"max relative error {grad['max_rel_error']:.2e} (tolerance {TH['gradcheck_rel_tol']})")]
    rows.append(("C2", "determinism_and_finiteness", *test_determinism(seed, mutant)))
    world, trio, hists = first
    rows.append(("C3", "learning", *learning_check(trio["model"], hists["model"], world)))
    rows.append(("C4", "shuffled_label_control", *test_shuffled(seed, steps)))
    caught = test_mutants(seed, HP["quick_steps"], rng)
    rows.append(("C5", "mutant_detection", all(caught.values()), f"{sum(map(bool, caught.values()))}/{len(caught)} mutants caught"))
    rows.extend(test_properties(seed, HP["quick_steps"], rng, mutant))
    rows.append(("C7", "split_integrity", *split_integrity(world)))
    return grad, rows, caught


def run_protocol(args, rng):
    t0 = time.time()
    steps = HP["quick_steps"] if args.quick else HP["steps"]
    budget = BUDGET_QUICK_S if args.quick else BUDGET_FULL_S
    seeds = [args.seed] if args.quick else [args.seed + i for i in range(args.seeds)]
    outcomes, first = [], None
    for s in seeds:
        ts = time.time()
        run = train_trio(s, steps, args.mutant)
        first = first or run
        outcomes.append(seed_outcomes(run[0], run[1]))
        a = outcomes[-1]["acc"]
        print(f"seed {s}: shifted-split accuracy model {a['model']['shift']:.3f}, baseline {a['baseline']['shift']:.3f}, "
              f"rival {a['rival_tard']['shift']:.3f} ({time.time() - ts:.1f} s)")
    grad, rows, caught = correctness_suite(seeds[0], steps, rng, args.mutant, first)
    hyps = evaluate_hypotheses(outcomes, rng, evaluated=not args.quick)
    kos = knockout_table(outcomes, rng, modules(first[1]["model"]))
    bridge = data_bridge(args.data, child_rng(args.seed, 17))
    runtime = time.time() - t0
    rows.append(("C8", "budget", runtime <= budget, f"{runtime:.1f} s of {budget:.0f} s"))
    failed = [r[0] for r in rows if not r[2]]
    code = EXIT_OK if not failed else (EXIT_BUDGET if failed == ["C8"] else EXIT_FAIL)
    rep = {"schema_version": "1.0", "chapter": MIND_CARD["id"], "file": os.path.basename(__file__),
           "card_revision": MIND_CARD["card_revision"],
           "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
           "seeds": seeds, "runtime_s": round(runtime, 2), "n_params": n_params(first[1]["model"]),
           "gradcheck": grad,
           "correctness": [{"id": i, "name": n, "passed": bool(p), "detail": d} for i, n, p, d in rows],
           "mutants": {"detected": sum(map(bool, caught.values())), "total": len(caught),
                       "score": sum(map(bool, caught.values())) / len(caught)},
           "hypotheses": hyps, "knockouts": kos, "task_types": TASK_TYPES, "exit_code": code,
           "per_seed": outcomes}
    sizes = {n: n_params(m) for n, m in first[1].items()}
    print_report(rep, {"steps": steps, "budget": budget, "sizes": sizes, "caught": caught, "bridge": bridge})
    if args.json:
        write_report(rep, args.json)
    return code


def main(argv=None):
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__),
                                 description="Chapter 0262 athar-gated cause network: full protocol by default.")
    ap.add_argument("--quick", action="store_true", help="one seed, reduced steps, all correctness tests")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--json", metavar="PATH")
    ap.add_argument("--card", action="store_true")
    ap.add_argument("--mutant", choices=sorted(MUTANTS))
    ap.add_argument("--data", metavar="PATH")
    args = ap.parse_args(argv)
    if args.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return EXIT_OK
    if args.seeds < 1:
        print(f"usage: python3 {os.path.basename(__file__)} [--quick] [--seed S] [--seeds N>=1] ...", file=sys.stderr)
        return EXIT_USAGE
    rng = np.random.default_rng(args.seed)
    print(f"{os.path.basename(__file__)} · seeds from {args.seed} · mutant {args.mutant or 'none'}")
    try:
        return run_protocol(args, rng)
    except NonFinite as err:
        print(f"non-finite values: {err}", file=sys.stderr)
        return EXIT_NONFINITE


if __name__ == "__main__":
    sys.exit(main())

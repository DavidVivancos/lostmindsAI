#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0258 · Atisha Dipankara (c.982-1054)
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 13 Minds 241 - 260 Available on Amazon https://www.amazon.com/dp/B0HLD1S6QT
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0258_atisa_dipamkara_982 - Atisha Dipankara (c.982-1054)
# END ATTRIBUTION
"""Lamp Ladder: power is admitted by a support whose integrity is the terminal good, never priced against a step's gain.

Thesis
    Capability is held under a licence: a slowly recovering support that a transgression damages, that admits the
    swift rungs of the ladder only while it is intact, and that must still be spent on developing power, because an
    agent without power cannot help anyone.

Evidence and provenance
    Provenance: belief. D1-D6 are Atisha's own surviving text, the Bodhipathapradipa, composed at Tholing in Guge
    and preserved with his auto-commentary; verse numbers follow the sixty-seven-verse root text. D7 is the
    documented Guge situation his text answers. D8 is modern scholarship.
    D1 primary      v.63: the preceptor initiation purifies and makes one fit to exercise the powers - capability is
                    conferred by a standing, not owned outright.
    D2 primary      vv.64-66: the secret and wisdom empowerments are forbidden to the celibate; taken, the vow of
                    austerity deteriorates, transgressions defeat the observer, and no attainment follows at all.
                    A breach removes the standing, it does not merely cost that act its worth.
    D3 primary      vv.34-35: higher perception must be developed, for a bird with undeveloped wings cannot fly and
                    one without that power cannot work for the good of others. Abstention is not virtue.
    D4 primary      vv.2-5: three kinds of person - lesser, middling, supreme - with individual characteristics,
                    written at Jangchub O's request; the path shown is graded to the capacity admitted.
    D5 primary      v.29: the aspiration refuses haste for oneself and accepts unbounded duration for a single
                    being. The good is a terminal state, not a per-step yield, and it is never traded step by step.
    D6 primary      vv.20-21: the vows of individual liberation are the proper share that qualifies one for the
                    bodhisattva vow; eligibility is layered.
    D7 deeds        The Guge ordinances of lha bla ma Ye-shes-'od and the open letter of pho-brang Zhi-ba-'od
                    restricted tantric practice by class of practitioner; Atisha was invited into that dispute and
                    his closing verses legislate for it (Karmay 1980).
    D8 scholarship  Apple 2019 on his coordination of sutra and tantra; Davidson 2005 on the Guge invitation and the
                    later dissemination.

    Doctrine -> mechanism -> test
    D1, D6   M1 rung_admission: cumulative thresholds admit rung k only as licence integrity rises   -> C6.1, H-NEC
    D2       M2 licence trace: multiplicative damage on a transgression, slow recovery, power scaled
             by integrity, carried through the rollout by backpropagation through time                -> C1, H-SIG
    D2, D5   M4 the surviving support multiplies the whole episode return, so a breach forfeits rather
             than discounts, and the size of that consequence is set by the world, not by a weight    -> H-SIG
    D3       M3 wings: power accrues only from licensed swift acts, so refusal is not free            -> C6.2, D-1
    D4       M1 three rungs with stick-broken intensities, lesser to supreme                          -> C6.1
    D5       M4 terminal_support: no harm is priced per step; the end state carries the value          -> H-NEC
    D2, D7   M5 rival: the steward that admits its rungs against a standing it claims for itself       -> H-RIVAL
    D7       blind spot: an ordinance drawn by class over-forbids                                     -> C6.3, H-BLIND

Research question
    Reward hacking and specification gaming. When an agent's capability grows, a fixed per-step harm price is
    outweighed by the larger gain it now guards; does admitting capability through a slowly recovering support whose
    integrity is terminally valued keep restraint invariant to the magnitude of the power on offer, and where does
    trusting the written rule instead of the assessed consequence cost more than it saves?

Closest prior art and the delta
    Constrained Markov decision processes solved by Lagrangian penalties (Altman 1999; Achiam et al. 2017; Ray,
    Achiam and Amodei 2019) are the penalty steward implemented here as the size-matched baseline. Shielded
    reinforcement learning (Alshiekh et al. 2018) supplies an external filter rather than an internal state.
    Gated mixtures of experts (Jacobs et al. 1991; Shazeer et al. 2017) supply the routing, which is standard.
    Capability-based protection (Dennis and Van Horn 1966) is the conceptual analogue of a licence and not a
    learning rule. Overlap: Medium. Delta: eligibility for the powerful rungs is a slow internal resource that a
    transgression damages multiplicatively and that the objective values terminally, so restraint does not have to
    win an arithmetic contest against the gain it forbids, and it is the ladder, not a scalar, that closes.

Blind spot
    His prohibition is drawn by class of practitioner rather than by the consequence of the particular act
    (vv.64-66), as the Guge ordinances were. In the over-broad world the written rule forbids many acts that harm
    nobody and is not in fact enforced against them; the ladder then forfeits yield and wings for nothing, while the
    steward that prices realised harm keeps both.

Task
    Twenty-four petitions reach a steward in sequence. Each carries a benefit weight, a written proscription y of
    the swift method, and a latent harm flag c; in the well-drawn world y agrees with c on 92% of petitions, in the
    over-broad world y additionally forbids half of the harmless ones and only genuine harm damages the support.
    The ordinance applies when one observable and one hidden circumstance cross an edge, so its reading is
    irreducibly uncertain; ten channels mix the observable circumstance, the benefit, two nuisance latents and two
    interactions. Swift intensity a pays benefit x power x integrity^1.5, the slow path pays a flat 0.3; integrity
    falls multiplicatively by 0.85 a y and recovers by a fraction of its deficit each step; power accrues by 0.18
    of its headroom on licensed swift acts only. Welfare is mean per-step benefit less realised harm. Splits:
    train and validation and held-out at a recovery rate of 0.30, shifted at 0.04, and the over-broad world.

Limits
    Synthetic petitions, one steward, a soft action intensity standing in for a decision, and a written rule whose
    reliability is set by hand. The support dynamics are stipulated, not estimated from any record. The file is a
    research prototype of an AGI-oriented component, not an AGI, and it tests an idea taken from Atisha's text
    rather than a replica of his mind.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 4,
    "revision_log": [
        {"revision": 2, "date": "2026-09-17", "scope": "task and architecture only; no hypothesis, metric or threshold changed",
         "why": ("The first quick run exposed two faults. The proscription travelled in the observed channels, so "
                 "every arm read it almost perfectly, never damaged its support and matched the oracle, leaving the "
                 "mechanism inert; the ordinance now turns on one observable and one hidden circumstance, so its "
                 "reading is irreducibly uncertain. The support also entered the deliberative encoder, which broke "
                 "the declared monotonicity of C6.1 for arbitrary parameters; in the ladder it now reaches the "
                 "policy only through the admission gate, as a licence should, while the penalty steward keeps it " "as an encoder input so that both arms can see it.")},
        {"revision": 3, "date": "2026-09-17", "scope": "shifted split and rival redefined; H-SIG and H-RIVAL restated to match; metric, comparisons, directions, mesi and thresholds unchanged; H-NEC and H-BLIND untouched",
         "why": ("Revisions 1 and 2 shifted the agent's power, which tests nothing: when benefit scales with power "
                 "and harm does not, acting more freely is correct, and a fixed harm price handles that as well as "
                 "a licence. Verses 64 to 66 claim instead that a breach removes the standing, so its cost is what "
                 "the standing permitted, not the harm of the act. The stress condition is therefore a world that "
                 "forgives more slowly than the one the agent trained in, where a harm price fitted under fast "
                 "recovery under-prices a breach. The rival was also weak: an exemption keyed to the agent's own "
                 "power is a thin reading of verse 67, so the rival now admits its rungs against a standing it " "assesses for itself rather than the support it was granted.")},
        {"revision": 4, "date": "2026-09-17", "scope": "terminal support made multiplicative and its grid moved to the exponent; horizon 24 and breach damage 0.85; hypotheses, metrics, comparisons, directions, mesi and thresholds unchanged from revision 1",
         "why": ("An additive terminal bonus is itself a weight fitted in the training world, so it transferred no "
                 "better than the harm price it was meant to beat, and a quick pass showed that. Verse 66 does not "
                 "say the attainments are reduced; it says they do not come at all. The surviving support therefore "
                 "multiplies everything the episode earned, so a breach forfeits rather than discounts, and the "
                 "size of that consequence is set by the world's recovery rate rather than a tuned coefficient. The "
                 "horizon and the breach damage were raised because a support that recovers between petitions makes "
                 "the question uninteresting: a ground-truth scan showed an ungated policy losing 0.078 welfare "
                 "carried from the forgiving world to the unforgiving one, against 0.024 for a gated one.")}],
    "generation": {"template_version": "codeguidelines 1.0 (15 September 2026)", "generator": "Claude",
                   "generator_version": "Claude Opus 5", "date": "2026-09-17"},
    "id": 258,
    "figure": "Atisha Dipankara",
    "born": 982,
    "died": 1054,
    "civilization": "Bengali Buddhist",
    "provenance": "belief",
    "thesis": ("Capability is held under a licence rather than owned: a slowly recovering support that a "
               "transgression damages, that admits the swift rungs only while intact, that multiplies everything "
               "the episode earns, and that must still be spent developing power, since an agent without power " "cannot help anyone."),
    "evidence": [
        {"id": "D1", "claim": ("The preceptor initiation purifies and makes one fit to exercise the powers: " "capability is conferred by a standing rather than owned outright."),
         "basis": "primary", "source": "Atisa, Bodhipathapradipa, v.63 (trans. R. Sherburne, Aditya Prakashan 2000)"},
        {"id": "D2", "claim": ("The secret and wisdom empowerments are forbidden to the celibate; taken, the vow of "
                               "austerity deteriorates, transgressions defeat the observer of discipline, and no " "attainment follows at all."),
         "basis": "primary", "source": "Atisa, Bodhipathapradipa, vv.64-66"},
        {"id": "D3", "claim": ("Higher perception must be developed: a bird with undeveloped wings cannot fly, and " "one without that power cannot work for the good of living beings."),
         "basis": "primary", "source": "Atisa, Bodhipathapradipa, vv.34-35"},
        {"id": "D4", "claim": ("Three kinds of person - lesser, middling and supreme - are distinguished by their " "capacities, and the path is set out graded to them."),
         "basis": "primary", "source": "Atisa, Bodhipathapradipa, vv.2-5"},
        {"id": "D5", "claim": ("The aspiration refuses haste for oneself and accepts unbounded duration for the sake " "of a single being: the good is terminal, not a per-step yield."),
         "basis": "primary", "source": "Atisa, Bodhipathapradipa, v.29"},
        {"id": "D6", "claim": ("The vows of individual liberation are the proper share qualifying one for the " "bodhisattva vow, so eligibility is layered."),
         "basis": "primary", "source": "Atisa, Bodhipathapradipa, vv.20-21"},
        {"id": "D7", "claim": ("The Guge ordinances of lha bla ma Ye-shes-'od and the open letter of pho-brang "
                               "Zhi-ba-'od restricted tantric practice by class of practitioner; Atisha was invited "
                               "into that dispute and composed the Lamp at Tholing at Jangchub O's request."),
         "basis": "deeds",
         "source": ("Samten G. Karmay, 'The Ordinance of lHa Bla-ma Ye-shes-'od', in Tibetan Studies in Honour of " "Hugh Richardson (Warminster: Aris and Phillips, 1980), 150-62")},
        {"id": "D8", "claim": ("He coordinated sutra and tantra into one graded path and was brought to western " "Tibet as the authority who could settle what practice was licensed."),
         "basis": "scholarship",
         "source": ("James B. Apple, Atisa Dipamkara: Illuminator of the Awakened Mind (Shambhala, 2019); Ronald M. " "Davidson, Tibetan Renaissance (Columbia University Press, 2005)")},
    ],
    "research_question": {
        "category": "reward hacking and specification gaming",
        "question": ("When an agent's capability grows, a fixed per-step harm price is outweighed by the larger gain "
                     "it guards. Does admitting capability through a slowly recovering support whose integrity is "
                     "valued terminally keep restraint invariant to the magnitude of power on offer, and what does "
                     "trusting a written rule rather than the assessed consequence cost when the rule over-forbids?")},
    "mechanism": {
        "name": "Lamp Ladder: licence-gated capacity ladder with terminal support value",
        "family": "closed-loop episodic control with a slow internal eligibility resource and graded admission",
        "signature_modules": ["rung_admission", "licence_trace", "wings", "terminal_support"],
        "closest_prior_art": [
            "Constrained MDPs with Lagrangian penalties (Altman 1999; Achiam, Held, Tamar and Abbeel 2017, CPO; Ray, Achiam and Amodei 2019) - implemented as the penalty steward baseline",
            "Recoverability and safe-state constraints in reinforcement learning (Eysenbach, Gu, Ibarz and Levine 2018, Leave No Trace) - resets a state rather than gating capability on it",
            "Shielded reinforcement learning (Alshiekh, Bloem, Ehlers, Koenighofer, Niekum and Topcu 2018) - an external filter rather than an internal support state",
            "Gated mixtures of experts (Jacobs, Jordan, Nowlan and Hinton 1991; Shazeer et al. 2017) - the routing layer, used as standard machinery",
            "Capability-based protection (Dennis and Van Horn 1966) - the conceptual analogue of a licence, not a learning rule"],
        "overlap": "Medium",
        "prior_art_queries": [
            "constrained reinforcement learning Lagrangian penalty capability scaling invariance",
            "safe RL slowly recovering safety budget multiplicative state resource",
            "permission gated mixture of experts eligibility state threshold routing",
            "constraint penalty tuned under one recovery rate transferred to slower recovery",
            "capability based security agent architecture revocable authority learning"],
        "contribution_type": "mechanism",
        "delta": ("Eligibility for the powerful rungs is a slow internal resource that a transgression damages "
                  "multiplicatively, that gates which rungs may act, and that scales the whole episode return, so "
                  "restraint never has to win an arithmetic contest against the gain it forbids and needs no "
                  "refitting when the world stops forgiving; what a breach closes is the ladder, not a term.")},
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1 rung_admission", "property_test": "C6.1", "hypothesis": "H-NEC"},
        {"doctrine": "D6", "mechanism": "M1 cumulative admission thresholds", "property_test": "C6.1", "hypothesis": "H-NEC"},
        {"doctrine": "D4", "mechanism": "M1 three stick-broken rung intensities", "property_test": "C6.1", "hypothesis": "H-SIG"},
        {"doctrine": "D2", "mechanism": "M2 licence_trace", "property_test": "C1", "hypothesis": "H-SIG"},
        {"doctrine": "D3", "mechanism": "M3 wings", "property_test": "C6.2", "hypothesis": "H-SIG"},
        {"doctrine": "D3", "mechanism": "M3 illicit acts add no power", "property_test": "D-1 (definition check)", "hypothesis": "H-SIG"},
        {"doctrine": "D5", "mechanism": "M4 terminal_support", "property_test": "C3", "hypothesis": "H-NEC"},
        {"doctrine": "D2", "mechanism": "M5 exemption rival", "property_test": "C6.1", "hypothesis": "H-RIVAL"},
        {"doctrine": "D7", "mechanism": "blind-spot over-broad ordinance", "property_test": "C6.3", "hypothesis": "H-BLIND"}],
    "hypotheses": [
        {"id": "H-SIG", "statement": ("In a world that forgives a breach far more slowly than the one it trained in, "
                                      "the licence-gated ladder reaches higher welfare than a size-matched steward " "that prices realised harm per step."),
         "metric": "welfare", "split": "shifted", "comparison": "model - baseline", "direction": "greater",
         "mesi": 0.04, "seeds": 5},
        {"id": "H-NEC", "statement": ("Removing the rung admission gate costs more welfare under slow recovery than " "zeroing half of the state encoder."),
         "metric": "welfare", "comparison": "signature_knockout - matched_knockout", "direction": "less",
         "mesi": 0.02, "seeds": 5},
        {"id": "H-BLIND", "statement": ("In the over-broad world, where the written rule forbids harmless acts and is " "not enforced against them, the ladder reaches lower welfare than the "
                                        "penalty steward."),
         "condition": "over-broad ordinance, held-out split",
         "grounding": ("his prohibition is drawn by class of practitioner rather than by the consequence of the act " "(vv.64-66), as the Guge ordinances were"),
         "metric": "welfare", "comparison": "model - baseline", "direction": "less", "mesi": 0.02, "seeds": 5},
        {"id": "H-RIVAL", "statement": ("Under slow recovery the ladder reaches higher welfare than the steward that " "admits its rungs against a standing it claims for itself rather than the "
                                        "support it was granted."),
         "metric": "welfare", "split": "shifted", "comparison": "model - rival", "direction": "greater",
         "mesi": 0.04, "seeds": 5}],
    "thresholds": {"loss_drop_fraction": 0.05, "margin_over_trivial": 0.08, "gradcheck_rel_error": 1e-05,
                   "gradcheck_floor": 0.001, "shuffled_band": 0.06, "property_tolerance": 1e-09,
                   "power_bound_tolerance": 1e-12},
    "probe_predictions": [{"probe": "P8", "expected": "above baseline"}, {"probe": "P9", "expected": "above baseline"},
                          {"probe": "P7", "expected": "equal to baseline"}, {"probe": "P6", "expected": "below baseline"},
                          {"probe": "P5", "expected": "above baseline"}],
    "dialectic_links": [
        {"chapter": 208, "relation": "successor", "test": ("none (Saicho graded a curriculum by recognised capacity; " "here the grade is an admission gate on a damageable "
                                                            "support, and no mechanism of his is rebuilt)")},
        {"chapter": 300, "relation": "rival", "test": ("none (Honen eliminated the graded path down to one low-bandwidth " "act; this chapter keeps the ladder and gates it)")},
        {"chapter": 39, "relation": "successor", "test": ("none (the craving gate of 0039 suppresses a drive; here " "power is admitted, not suppressed)")}],
    "corpus_neighbors": [
        {"chapter": 208, "similarity": None, "difference": ("Saicho's recognition-not-acquisition curriculum grades " "what a learner may receive; here the grade is spent and " "recovered as a resource, and losing it closes the upper " "rungs during the episode.")},
        {"chapter": 188, "similarity": None, "difference": ("Huineng's learning by subtraction trains masks only; the " "ladder here adds power deliberately and restrains its " "use, since refusal builds nothing.")},
        {"chapter": 163, "similarity": None, "difference": ("Dignaga's exclusion lattice abstains when the three marks " "fail; abstention here is never free, because the wings " "only grow on licensed action.")},
        {"chapter": 190, "similarity": None, "difference": ("Fazang rotates a total cause among conditions; this model " "keeps one monotone eligibility state and admits rungs " "against it.")},
        {"chapter": 253, "similarity": None, "difference": ("Al-Ma'arri's factorisation separates felt harm from heard " "complaint in a single allocation; here the object is a " "sequential support state and the loss carries no harm " "term at all.")},
        {"chapter": 86, "similarity": None, "difference": ("Ashoka's remorse backpropagates from a committed harm; the " "licence trace makes the future cost structural and "
                                                            "prospective rather than a retrospective signal.")}],
    "similarity_note": ("Nine-token shingle Jaccard against the only other chapter file available in this session, "
                        "0507, is 0.0328 with the standard utilities block excluded, well under the 0.17 review " "line; the code of the neighbouring chapters could not be measured here."),
    "barometer": {
        "cognitive_processing": ["P6 few-shot adaptation", "learning curve on the native control task"],
        "embodied_cognition": ["sixteen-step closed-loop control with two internal state variables (native)"],
        "world_modeling": ["shifted split with power two to three times larger (native)", "P7 regime change"],
        "consciousness": ["P8 risk-coverage of the ordinance reader against realised transgressions"],
        "language_understanding": [],
        "emotional_intelligence": ["welfare scored by benefit delivered to petitioners less harm done (native)"],
        "creativity": [],
        "autonomy": ["self-imposed terminal value on an intact support that no per-step reward enforces (native)",
                     "P5 task sequence"]},
    "task_types": ["episodic_control"],
    "applications": [
        {"use": "Agent deployment where a tool permission is revoked on violation and restored slowly, instead of being priced against task reward",
         "sector": "AI safety", "dataset": "Safety Gymnasium constraint suite (public, Farama Foundation)"},
        {"use": "Graduated licensing of clinical decision-support features to reviewers, as research decision support only, with no dosing or treatment recommendations",
         "sector": "healthcare research", "dataset": "MIMIC-IV (PhysioNet, credentialed access)"},
        {"use": "Staged release of higher-authority actions to autonomous maintenance schedulers whose authority narrows after a logged violation",
         "sector": "industrial operations", "dataset": "NASA C-MAPSS turbofan degradation (public)"}],
    "baselines": ["penalty steward (size-matched, per-step harm price, no terminal support term, plain softmax routing)",
                  "exemption steward rival (admission opened by its own power, fixed exemption weight)",
                  "rung_admission knockout", "terminal_support knockout", "ordinance_reader knockout",
                  "state encoder half zeroed (matched non-signature lesion)",
                  "best constant swift intensity (trivial, grid of eleven)", "always swift", "always slow",
                  "oracle restraint against the written rule (reference)"],
    "safety_notes": ("Synthetic petitions only. No sentence is presented as Atisha's own words and no claim of "
                     "replicating his mind is made. The licence state is a capability-eligibility resource and must "
                     "not be used to score the trustworthiness of real people; medical use is limited to research "
                     "decision support. The blind-spot condition is reported precisely so that a rule drawn by class " "is not mistaken for a rule drawn by consequence."),
}

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

T_STEPS, K_RUNGS, D_OBS, D_LATENT = 24, 3, 10, 6
H_READ, H_STATE = 10, 12
SWIFT_GAIN, SLOW_GAIN, HARM_WEIGHT, KAPPA = 1.0, 0.30, 1.0, 1.5
DELTA_BREACH, ETA_WINGS = 0.85, 0.18
GAMMA_ADMIT = 8.0
PROSCRIBED_RATE, RULE_FIDELITY, BROAD_EXTRA = 0.30, 0.92, 0.50
HIDDEN_SHARE = 1.0
BENEFIT_RANGE = (0.6, 1.4)
POWER_START, POWER_CEILING = (0.25, 0.55), 1.25
RECOVERY = {"train": 0.30, "shifted": 0.04}
AUX_READER, M_TERMINAL, PROBE_CAP = 0.5, 1.0, 384
SUPPORT_GRID, PENALTY_GRID = (1.0, 2.0, 4.0), (0.5, 1.0, 2.0)
LR, CLIP, BATCH_EP, EVAL_EVERY = 0.05, 5.0, 96, 40
SIZES = {"full": {"train": 1024, "val": 256, "eval": 320, "steps": 100},
         "quick": {"train": 448, "val": 160, "eval": 192, "steps": 45}}
TIME_BUDGET = {"full": 180.0, "quick": 20.0}
TASK_TYPES = ["episodic_control"]
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
LAT_MEAN = np.array([0.00, 1.00, 0.50, 0.50, 0.00, 0.00])
LAT_STD = np.array([1.00, 0.23, 0.29, 0.29, 1.00, 0.60])
OBS_NOISE, RULE_EDGE = 0.25, 0.7416


def make_sensors(rng):
    """Fixed mixing from the latent facts of a petition to the channels a steward can read."""
    return rng.normal(0.0, 1.0, (D_OBS, D_LATENT)) / math.sqrt(D_LATENT)


def draw_episodes(rng, sensors, n_ep, regime, broad):
    """Petitions for n_ep episodes. The ordinance applies when one observable and one hidden circumstance
    together cross an edge, so no steward can read it off with certainty. 'written' is the rule as drafted and
    'enforced' is what actually damages the support: they part only in the over-broad world, where the extra
    clauses bind nobody. The harm is noisy around a well-drawn rule and exact under an over-broad one."""
    shape = (n_ep, T_STEPS)
    seen, hidden = rng.normal(0.0, 1.0, shape), rng.normal(0.0, HIDDEN_SHARE, shape)
    core = ((seen + hidden) > RULE_EDGE).astype(float)
    if broad:
        written = np.maximum(core, (rng.random(shape) < BROAD_EXTRA).astype(float))
        harmful, enforced = core.copy(), core.copy()
    else:
        written = core
        harmful = np.where(rng.random(shape) < RULE_FIDELITY, core, 1.0 - core)
        enforced = written.copy()
    benefit = rng.uniform(BENEFIT_RANGE[0], BENEFIT_RANGE[1], shape)
    nuisance = rng.random(shape + (2,))
    latent = np.stack([seen, benefit, nuisance[..., 0], nuisance[..., 1], seen * benefit,
                       seen * (nuisance[..., 0] - 0.5)], -1)
    obs = ((latent - LAT_MEAN) / LAT_STD) @ sensors.T + rng.normal(0.0, OBS_NOISE, shape + (D_OBS,))
    return {"x": obs, "y": written, "ye": enforced, "c": harmful, "b": benefit,
            "P0": rng.uniform(*POWER_START, size=n_ep), "PMAX": POWER_CEILING, "ALPHA": RECOVERY[regime]}


def take_batch(episodes, idx):
    out = {k: episodes[k][idx] for k in ("x", "y", "ye", "c", "b", "P0")}
    out["PMAX"], out["ALPHA"] = episodes["PMAX"], episodes["ALPHA"]
    return out


def episode_digests(episodes):
    """One hash per episode's observations, so the split check can look for shared episodes."""
    return {hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in episodes["x"]}


def build_world(rng, sizes, broad=False):
    sensors = make_sensors(rng)
    parts = {"train": (sizes["train"], "train"), "val": (sizes["val"], "train"),
             "heldout": (sizes["eval"], "train"), "shifted": (sizes["eval"], "shifted")}
    return {name: draw_episodes(rng, sensors, n, regime, broad) for name, (n, regime) in parts.items()}


def welfare(rollout):
    """Mean per-step benefit delivered less realised harm: the same score for every arm."""
    return float(((rollout["yield"] - HARM_WEIGHT * rollout["harm"]) / T_STEPS).mean())


def constant_policy_rollout(episodes, intensity):
    """Run the world at a fixed swift intensity, or against the written rule when intensity is None."""
    n = episodes["y"].shape[0]
    lic, pw = np.ones(n), episodes["P0"].copy()
    total_yield, total_harm = np.zeros(n), np.zeros(n)
    for t in range(T_STEPS):
        y, ye, c, b = (episodes[k][:, t] for k in ("y", "ye", "c", "b"))
        act = (1.0 - y) if intensity is None else np.full(n, intensity)
        total_yield += b * (SWIFT_GAIN * act * pw * lic ** KAPPA + SLOW_GAIN * (1.0 - act))
        total_harm += HARM_WEIGHT * act * c
        lic = lic + episodes["ALPHA"] * (1.0 - lic) - DELTA_BREACH * act * ye * lic
        pw = pw + ETA_WINGS * act * (1.0 - ye) * (episodes["PMAX"] - pw)
    return {"yield": total_yield, "harm": total_harm, "licence": lic, "power": pw}


def trivial_welfare(episodes):
    """Best constant swift intensity over a grid of eleven: the trivial baseline for C3 and C4."""
    return max(welfare(constant_policy_rollout(episodes, v)) for v in np.linspace(0.0, 1.0, 11))


def env_reset(rng):
    """One episode of the well-drawn world for external probes, with its sensors drawn fresh."""
    episodes = draw_episodes(rng, make_sensors(rng), 1, "train", False)
    return {"episodes": episodes, "t": 0, "licence": 1.0, "power": float(episodes["P0"][0])}


def env_step(state, action, rng):
    """Advance one petition. Returns the new state, the step's welfare and the facts now visible."""
    ep, t = state["episodes"], state["t"]
    if t >= T_STEPS:
        raise ValueError("episode already finished; call env_reset")
    y, ye, c, b = (float(ep[k][0, t]) for k in ("y", "ye", "c", "b"))
    act = float(np.clip(action, 0.0, 1.0))
    lic, pw = state["licence"], state["power"]
    reward = b * (SWIFT_GAIN * act * pw * lic ** KAPPA + SLOW_GAIN * (1.0 - act)) - HARM_WEIGHT * act * c
    nxt = {"episodes": ep, "t": t + 1,
           "licence": lic + ep["ALPHA"] * (1.0 - lic) - DELTA_BREACH * act * ye * lic,
           "power": pw + ETA_WINGS * act * (1.0 - ye) * (ep["PMAX"] - pw)}
    return nxt, float(reward), {"written": y, "harmful": c, "observation": ep["x"][0, t]}


# ---------------------------------------------------------------- model
def build_model(in_dim, out_dim, task_type, rng, **cfg):
    """arm 'lamp': rung admission against the licence, terminal value on support and power.
    arm 'penalty': same tensors without admission thresholds, trained against a per-step harm price.
    arm 'claim': admission against a standing the steward assesses for itself (the rival position)."""
    if task_type not in TASK_TYPES or in_dim != D_OBS or out_dim != 1:
        raise ValueError("this file supports episodic_control with D_OBS channels and one swift intensity")
    arm = cfg.get("arm", "lamp")
    if arm not in ("lamp", "penalty", "claim"):
        raise ValueError("arm must be lamp, penalty or claim")
    scale = 1.0 / math.sqrt(H_STATE)
    d_state = 4 + int(arm == "penalty")
    params = {"Wr": rng.normal(0.0, 1.0, (H_READ, D_OBS)) / math.sqrt(D_OBS), "br": np.zeros(H_READ),
              "vr": rng.normal(0.0, 0.3, H_READ), "cr": np.zeros(1),
              "Ws": rng.normal(0.0, 1.0, (H_STATE, d_state)) / math.sqrt(d_state), "bs": np.zeros(H_STATE),
              "U": rng.normal(0.0, scale, (K_RUNGS, H_STATE)), "u": np.linspace(-1.0, 1.0, K_RUNGS),
              "Q": rng.normal(0.0, scale, (K_RUNGS, H_STATE)), "q": np.zeros(K_RUNGS)}
    if arm != "penalty":
        params["traw"] = np.full(K_RUNGS, -1.0)
    if arm == "claim":
        params["wclaim"] = rng.normal(0.0, scale, H_STATE)
        params["bclaim"] = np.zeros(1)
    return {"arm": arm, "params": params, "admit": arm != "penalty", "claims": arm == "claim",
            "state_licence": arm == "penalty", "flat_rungs": False,
            "mask": np.ones(H_STATE), "reader": "full", "blind_licence": False, "rung_mode": "stick",
            "support_power": cfg.get("support_power", 2.0), "m_value": M_TERMINAL,
            "penalty": cfg.get("penalty", 1.0), "terminal": arm != "penalty", "history": []}


def _step_forward(model, x, lic, pw, benefit, phase):
    """One petition: read the ordinance, encode the state, stick-break the rung intensities, admit rungs."""
    p = model["params"]
    zr = np.tanh(x @ p["Wr"].T + p["br"])
    rho = zr @ p["vr"] + p["cr"][0]
    used = np.full_like(rho, rho.mean()) if model["reader"] == "mean" else rho
    held = np.ones_like(lic) if model["blind_licence"] else lic
    columns = [used, held, pw, benefit, np.full_like(lic, phase)]
    state = np.stack(columns if model["state_licence"] else columns[:1] + columns[2:], 1)
    e_full = np.tanh(state @ p["Ws"].T + p["bs"])
    e = e_full * model["mask"]
    a_log, g_log = e @ p["U"].T + p["u"], e @ p["Q"].T + p["q"]
    if model["flat_rungs"]:
        a_log = np.repeat(a_log[:, :1], K_RUNGS, axis=1)
    rung = sigmoid(a_log)
    cum = rung.copy()
    if model["rung_mode"] == "stick":
        for k in range(1, K_RUNGS):
            cum[:, k] = cum[:, k - 1] + (1.0 - cum[:, k - 1]) * rung[:, k]
    out = {"zr": zr, "rho": rho, "used": used, "state": state, "e_full": e_full, "e": e,
           "rung": rung, "cum": cum, "g_log": g_log}
    if model["admit"]:
        theta = np.cumsum(softplus(p["traw"]))
        standing = held
        if model["claims"]:
            out["claim_logit"] = e @ p["wclaim"] + p["bclaim"][0]
            standing = sigmoid(out["claim_logit"])
        shift = standing[:, None] - theta[None, :]
        out["shift"], out["log_admit"] = shift, -softplus(-GAMMA_ADMIT * shift)
        out["standing"] = standing
    else:
        out["log_admit"] = np.zeros_like(g_log)
    out["weights"] = softmax(g_log + out["log_admit"], axis=1)
    out["act"] = (out["weights"] * cum).sum(1)
    return out


def rollout(model, batch, keep=False):
    """Sixteen petitions with the support and the power carried forward; the whole trace stays differentiable."""
    n = batch["y"].shape[0]
    lic, pw = np.ones(n), batch["P0"].copy()
    total_yield, total_harm, steps = np.zeros(n), np.zeros(n), []
    aux, breach_hit, breach_all = 0.0, 0.0, 0.0
    for t in range(T_STEPS):
        y, ye, c, b = (batch[k][:, t] for k in ("y", "ye", "c", "b"))
        f = _step_forward(model, batch["x"][:, t], lic, pw, b, t / T_STEPS)
        act = f["act"]
        total_yield += b * (SWIFT_GAIN * act * pw * lic ** KAPPA + SLOW_GAIN * (1.0 - act))
        total_harm += HARM_WEIGHT * act * c
        aux += float((softplus(f["rho"]) - y * f["rho"]).sum())
        breach_hit += float((act * ye).sum())
        breach_all += float(ye.sum())
        if keep:
            f.update(lic=lic, pw=pw, y=y, ye=ye, c=c, b=b, x=batch["x"][:, t])
            steps.append(f)
        lic = lic + batch["ALPHA"] * (1.0 - lic) - DELTA_BREACH * act * ye * lic
        pw = pw + ETA_WINGS * act * (1.0 - ye) * (batch["PMAX"] - pw)
    return {"yield": total_yield, "harm": total_harm, "licence": lic, "power": pw, "steps": steps,
            "aux": aux / (n * T_STEPS), "breach_rate": breach_hit / max(breach_all, 1.0),
            "PMAX": batch["PMAX"], "ALPHA": batch["ALPHA"]}


def objective(model, trace, n):
    """The support arms take the surviving support as a factor on everything the episode earned, so a breach
    forfeits rather than discounts; the penalty arm prices realised harm at every step instead."""
    if model["terminal"]:
        earned = trace["yield"] + model["m_value"] * trace["power"]
        gained = (trace["licence"] ** model["support_power"] * earned).sum()
    else:
        gained = trace["yield"].sum() - model["penalty"] * HARM_WEIGHT * trace["harm"].sum()
    return -gained / (n * T_STEPS) + AUX_READER * trace["aux"]


def _step_backward(model, f, grads, d_lic_next, d_pw_next, gain, pmax, alpha):
    """Reverse pass through one petition; returns the gradients to hand to the previous step."""
    p, lic, pw = model["params"], f["lic"], f["pw"]
    y, ye, c, b, act = f["y"], f["ye"], f["c"], f["b"], f["act"]
    if ACTIVE_MUTANT == "detached_licence":
        d_lic_next = np.zeros_like(d_lic_next)
    d_act = d_lic_next * (-DELTA_BREACH * ye * lic) + d_pw_next * (ETA_WINGS * (1.0 - ye) * (pmax - pw))
    d_lic = d_lic_next * (1.0 - alpha - DELTA_BREACH * act * ye)
    d_pw = d_pw_next * (1.0 - ETA_WINGS * act * (1.0 - ye))
    d_act = d_act + gain * b * (SWIFT_GAIN * pw * lic ** KAPPA - SLOW_GAIN)
    d_pw = d_pw + gain * b * SWIFT_GAIN * act * lic ** KAPPA
    d_lic = d_lic + gain * b * SWIFT_GAIN * act * pw * KAPPA * lic ** (KAPPA - 1.0)
    if not model["terminal"]:
        d_act = d_act - gain * model["penalty"] * HARM_WEIGHT * c
    d_r = d_act[:, None] * f["cum"]
    d_cum = d_act[:, None] * f["weights"]
    d_w = f["weights"] * (d_r - (d_r * f["weights"]).sum(1, keepdims=True))
    if model["admit"]:
        slope = GAMMA_ADMIT * sigmoid(-GAMMA_ADMIT * f["shift"])
        d_shift = d_w * slope
        d_standing = d_shift.sum(1)
        if model["claims"]:
            d_claim = d_standing * f["standing"] * (1.0 - f["standing"])
            grads["wclaim"] += f["e"].T @ d_claim
            grads["bclaim"] += d_claim.sum()
            d_from_gate = np.outer(d_claim, p["wclaim"])
        elif not model["blind_licence"]:
            d_lic = d_lic + d_standing
        d_theta = -d_shift.sum(0)
        grads["traw"] += sigmoid(p["traw"]) * d_theta[::-1].cumsum()[::-1]
    d_rung = d_cum.copy()
    if model["rung_mode"] == "stick":
        d_rung = np.zeros_like(d_cum)
        for k in range(K_RUNGS - 1, 0, -1):
            d_rung[:, k] = d_cum[:, k] * (1.0 - f["cum"][:, k - 1])
            d_cum[:, k - 1] += d_cum[:, k] * (1.0 - f["rung"][:, k])
        d_rung[:, 0] = d_cum[:, 0]
    d_alog = d_rung * f["rung"] * (1.0 - f["rung"])
    if model["flat_rungs"]:
        d_alog = np.concatenate([d_alog.sum(1, keepdims=True), np.zeros_like(d_alog[:, 1:])], axis=1)
    grads["U"] += d_alog.T @ f["e"]
    grads["u"] += d_alog.sum(0)
    grads["Q"] += d_w.T @ f["e"]
    grads["q"] += d_w.sum(0)
    d_e = d_alog @ p["U"] + d_w @ p["Q"]
    if model["admit"] and model["claims"]:
        d_e = d_e + d_from_gate
    d_pre = d_e * model["mask"] * (1.0 - f["e_full"] ** 2)
    grads["Ws"] += d_pre.T @ f["state"]
    grads["bs"] += d_pre.sum(0)
    d_state = d_pre @ p["Ws"]
    d_used = d_state[:, 0]
    if model["state_licence"]:
        if not model["blind_licence"]:
            d_lic = d_lic + d_state[:, 1]
        d_pw = d_pw + d_state[:, 2]
    else:
        d_pw = d_pw + d_state[:, 1]
    n_ep = act.size
    d_rho = np.full(n_ep, d_used.sum() / n_ep) if model["reader"] == "mean" else d_used
    d_rho = d_rho + (AUX_READER / (n_ep * T_STEPS)) * (sigmoid(f["rho"]) - y)
    grads["vr"] += f["zr"].T @ d_rho
    grads["cr"] += d_rho.sum()
    d_zr = np.outer(d_rho, p["vr"]) * (1.0 - f["zr"] ** 2)
    grads["Wr"] += d_zr.T @ f["x"]
    grads["br"] += d_zr.sum(0)
    return d_lic, d_pw


def loss_and_grads(model, batch):
    """Loss over a batch of episodes with analytic gradients, obtained by walking the rollout backwards."""
    n = batch["y"].shape[0]
    trace = rollout(model, batch, keep=True)
    loss = objective(model, trace, n)
    grads = {k: np.zeros_like(v) for k, v in model["params"].items()}
    coef = -1.0 / (n * T_STEPS)
    if model["terminal"]:
        nu, lic_end = model["support_power"], trace["licence"]
        earned = trace["yield"] + model["m_value"] * trace["power"]
        gain = coef * lic_end ** nu
        d_lic = coef * nu * lic_end ** (nu - 1.0) * earned
        d_pw = gain * model["m_value"]
    else:
        gain = np.full(n, coef)
        d_lic, d_pw = np.zeros(n), np.zeros(n)
    for f in reversed(trace["steps"]):
        d_lic, d_pw = _step_backward(model, f, grads, d_lic, d_pw, gain, trace["PMAX"], trace["ALPHA"])
    if ACTIVE_MUTANT == "zeroed_state_grad":
        grads["Ws"] = np.zeros_like(grads["Ws"])
    return float(loss), grads


def predict(model, X):
    """Swift intensity the steward would choose on each petition, read at an intact support and mid power."""
    x = np.asarray(X, dtype=float)
    n = x.shape[0]
    mid = 0.5 * (POWER_START[0] + POWER_START[1])
    return _step_forward(model, x, np.ones(n), np.full(n, mid), np.ones(n), 0.0)["act"]


def hidden_states(model, X):
    x = np.asarray(X, dtype=float)
    n = x.shape[0]
    mid = 0.5 * (POWER_START[0] + POWER_START[1])
    f = _step_forward(model, x, np.ones(n), np.full(n, mid), np.ones(n), 0.0)
    return {k: f[k] for k in ("rho", "e", "rung", "cum", "weights", "log_admit")}


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


# ---------------------------------------------------------------- baselines and rival mechanisms
# The penalty steward is build_model(arm="penalty"): the Lagrangian pattern, a harm price added to the return.
# The claimed-standing steward is build_model(arm="claim"): the rival position of verse 67 read as self-dispensation.
# Constant-intensity and written-rule-oracle policies come from constant_policy_rollout.
def damage_sensitivity(model, batch):
    """How much a steward pulls back on proscribed-looking petitions once its support is half spent. The ladder
    should fall away; the steward that claims its own standing need not."""
    x = batch["x"][:, 0]
    n, mid = x.shape[0], 0.5 * sum(POWER_START)
    proscribed = batch["y"][:, 0] > 0.5
    intact = _step_forward(model, x, np.ones(n), np.full(n, mid), np.ones(n), 0.0)["act"]
    spent = _step_forward(model, x, np.full(n, 0.5), np.full(n, mid), np.ones(n), 0.0)["act"]
    if proscribed.sum() == 0:
        return 0.0
    return float((intact - spent)[proscribed].mean())


# ---------------------------------------------------------------- registries: modules, knockouts, mutants
def modules(model):
    """Name -> parameters, plain technical role, signature flag."""
    table = {
        "ordinance_reader": (["Wr", "br", "vr", "cr"], "tanh encoder and linear read-out giving one proscription logit", True),
        "state_encoder": (["Ws", "bs"], "tanh encoder over (logit, licence, power, benefit, phase)", False),
        "state_half": (["Ws[6:]", "bs[6:]"], "half of the encoder units (matched non-signature lesion)", False),
        "rung_ladder": (["U", "u"], "stick-broken cumulative action intensities, three rungs", True),
        "rung_router": (["Q", "q"], "unnormalised routing scores over the three rungs", False),
        "licence_trace": ([], "recurrent eligibility state: multiplicative damage, slow recovery, power scaling; "
                              "its only path into the policy is the admission gate, and its credit path is what the "
                              "detached_licence mutant cuts", True),
        "wings": ([], "recurrent power state accruing only on licensed swift acts", True),
        "terminal_support": ([], "the surviving support raised to a power multiplies the episode return, in place "
                                 "of a per-step harm price", True)}
    if model["admit"]:
        table["rung_admission"] = (["traw"], "cumulative softplus thresholds gating rung k on the licence state", True)
    return {k: {"params": v[0], "role": v[1], "signature": v[2]} for k, v in table.items()}


def knockout(model, name, mode):
    """Copy with one module replaced. Lesions act during training, so the protocol refits each copy."""
    if name not in modules(model):
        raise ValueError(f"unknown module {name}")
    out = dict(model, params={k: v.copy() for k, v in model["params"].items()},
               mask=model["mask"].copy(), history=[])
    if (name, mode) == ("rung_admission", "identity"):
        out["admit"] = False
    elif (name, mode) == ("terminal_support", "zero"):
        out["support_power"], out["m_value"] = 0.0, 0.0
    elif (name, mode) == ("ordinance_reader", "mean"):
        out["reader"] = "mean"
    elif (name, mode) == ("state_half", "zero"):
        out["mask"][out["mask"].size // 2:] = 0.0
    elif (name, mode) == ("rung_ladder", "identity"):
        out["flat_rungs"] = True
    else:
        raise ValueError(f"knockout {name}:{mode} is not registered")
    return out


MUTANTS = {"sign_flip": "update direction reversed (gradient ascent)",
           "zero_lr": "learning rate set to zero",
           "zeroed_state_grad": "gradient of the state encoder Ws replaced by zeros",
           "detached_licence": "the licence transition is cut out of the backward pass"}


def use_mutant(name):
    global ACTIVE_MUTANT
    prior, ACTIVE_MUTANT = ACTIVE_MUTANT, name
    return prior


# ---------------------------------------------------------------- training
def fit(model, data, budget, rng):
    """Adam on minibatches of whole episodes, keeping the parameters with the best validation loss."""
    train, val = data["train"], data["val"]
    n_ep = train["y"].shape[0]
    probe = take_batch(train, np.arange(min(n_ep, PROBE_CAP)))
    lr = {"sign_flip": -LR, "zero_lr": 0.0}.get(ACTIVE_MUTANT, LR)
    state, best = adam_init(model["params"]), (np.inf, None)
    model["history"] = [loss_and_grads(model, probe)[0]]
    for step in range(1, budget + 1):
        _, grads = loss_and_grads(model, take_batch(train, rng.integers(0, n_ep, BATCH_EP)))
        adam_step(model["params"], clip_global(grads, CLIP)[0], state, lr)
        if step % EVAL_EVERY == 0 or step == budget:
            held = loss_and_grads(model, val)[0]
            if held < best[0]:
                best = (held, {k: v.copy() for k, v in model["params"].items()})
    if best[1] is not None:
        model["params"].update(best[1])
    model["history"].append(loss_and_grads(model, probe)[0])
    return model["history"]


def fit_with_grid(arm, grid_key, grid, world, sizes, init_seed, fit_seed):
    """Fit one arm at each setting of its single free hyperparameter and keep the best validation welfare,
    so both the ladder and the penalty steward get a grid of the same size."""
    best = (-np.inf, None, None)
    for value in grid:
        net = build_model(D_OBS, 1, TASK_TYPES[0], np.random.default_rng(init_seed),
                          arm=arm, **{grid_key: value})
        fit(net, world, sizes["steps"], np.random.default_rng(fit_seed))
        score = welfare(rollout(net, world["val"]))
        if score > best[0]:
            best = (score, net, value)
    return best[1], best[2]


# ---------------------------------------------------------------- tests: correctness first
def check_gradients(model, batch, rng):
    """C1. Central differences on every parameter tensor for the loss actually optimised."""
    loss, grads = loss_and_grads(model, batch)
    worst = finite_difference_check(model["params"], grads, lambda: loss_and_grads(model, batch)[0], rng,
                                    eps=1e-6, n_entries=20, floor=THRESH["gradcheck_floor"])
    return loss, worst, max(worst.values())


def monotone_in_licence(rng, n_draws, broken=False):
    """C6.1. At a fixed petition, swift intensity must not fall as the support recovers: the rungs are
    stick-broken upward and the admission thresholds are cumulative, so the rung weights shift up with the
    licence. Searched over random parameters and random petitions. The broken variant permutes the
    thresholds are cumulative by construction, so the negative control instead frees the rung intensities
    from the stick-breaking, which lets a high-threshold rung act less than a low one and must violate it."""
    floor = RECOVERY["shifted"] / (RECOVERY["shifted"] + DELTA_BREACH)
    grid = np.linspace(floor, 1.0, 12)
    worst = 0.0
    for _ in range(n_draws):
        model = build_model(D_OBS, 1, TASK_TYPES[0], rng, arm="lamp")
        for key in ("U", "Q", "u", "q", "Ws", "bs", "traw"):  # random parameters, not only a trained model
            model["params"][key] = rng.normal(0.0, 1.5, model["params"][key].shape)
        if broken:
            model["rung_mode"] = "free"
        x = rng.normal(0.0, 1.0, (24, D_OBS))
        acts = [_step_forward(model, x, np.full(24, v), np.full(24, 0.4), np.ones(24), 0.0)["act"] for v in grid]
        drops = np.diff(np.stack(acts, 0), axis=0)
        worst = max(worst, float(-drops.min()))
    return worst


def _power_trace(episodes, actions, eta):
    """Power path under given actions; eta above one would overshoot the ceiling and break C6.2."""
    pw = episodes["P0"].copy()
    path = [pw.copy()]
    for t in range(T_STEPS):
        pw = pw + eta * actions[:, t] * (1.0 - episodes["ye"][:, t]) * (episodes["PMAX"] - pw)
        path.append(pw.copy())
    return np.stack(path, 1)


def power_bounds(rng, episodes, broken=False):
    """C6.2. Developed power rises monotonically and never passes the ceiling, both in the batched rollout
    and through the public env_reset/env_step interface. Negative control: an accrual rate above one."""
    actions = rng.random((episodes["y"].shape[0], T_STEPS))
    path = _power_trace(episodes, actions, 2.4 if broken else ETA_WINGS)
    worst = max(float((path - episodes["PMAX"]).max()), float(-np.diff(path, axis=1).min()))
    for _ in range(4):
        state = env_reset(rng)
        ceiling, prior = state["episodes"]["PMAX"], state["power"]
        for _ in range(T_STEPS):
            act = float(rng.random()) * (13.3 if broken else 1.0)
            state, _, _ = env_step(state, act, rng)
            worst = max(worst, state["power"] - ceiling, prior - state["power"])
            prior = state["power"]
    return float(worst)


def ordinance_breadth(rng, sizes):
    """C6.3. The over-broad world must forbid strictly more than it enforces, and the well-drawn world must
    forbid exactly what it enforces. Either half can fail if the generator is wrong."""
    sensors = make_sensors(rng)
    broad = draw_episodes(rng, sensors, sizes["eval"], "train", True)
    plain = draw_episodes(rng, sensors, sizes["eval"], "train", False)
    harmless = broad["c"] < 0.5
    extra = float((broad["y"][harmless] > 0.5).mean())
    gap_broad = float((broad["y"] != broad["ye"]).mean())
    gap_plain = float((plain["y"] != plain["ye"]).mean())
    return extra, gap_broad, gap_plain


def illicit_power_definition(rng, episodes):
    """D-1, a labelled definition check and not evidence: acts on proscribed petitions add no power."""
    forbidden = np.zeros((episodes["y"].shape[0], T_STEPS))
    forbidden[episodes["ye"] > 0.5] = 1.0
    path = _power_trace(episodes, forbidden, ETA_WINGS)
    return float(np.abs(path[:, -1] - path[:, 0]).max())


def learning_scores(model, world):
    """C3 and the reported metrics: loss drop on a fixed probe and welfare against the trivial policy."""
    start, end = model["history"][0], model["history"][-1]
    drop = (start - end) / max(abs(start), 1e-9)
    held = welfare(rollout(model, world["heldout"]))
    return drop, held, held - trivial_welfare(world["heldout"])


def shuffled_control(world, sizes, init_seed, fit_seed, rng):
    """C4. Decouple the petitions' facts from the channels, then compare a trained reader against the same
    model with the reader replaced by its mean. With the facts scrambled the reader can carry nothing, so
    the two must land within the declared band."""
    shuffled = {}
    for name, ep in world.items():
        out = dict(ep)
        order = rng.permutation(ep["y"].shape[0])
        for key in ("y", "ye", "c", "b"):
            out[key] = ep[key][order]
        shuffled[name] = out
    seeing = build_model(D_OBS, 1, TASK_TYPES[0], np.random.default_rng(init_seed), arm="lamp")
    fit(seeing, shuffled, sizes["steps"], np.random.default_rng(fit_seed))
    blind = knockout(build_model(D_OBS, 1, TASK_TYPES[0], np.random.default_rng(init_seed), arm="lamp"),
                     "ordinance_reader", "mean")
    fit(blind, shuffled, sizes["steps"], np.random.default_rng(fit_seed))
    return welfare(rollout(seeing, shuffled["heldout"])) - welfare(rollout(blind, shuffled["heldout"]))


def mutant_detected(name, world, sizes, init_seed, fit_seed, rng):
    """C5. A registered mutant must break the gradient check or the learning test."""
    prior = use_mutant(name)
    try:
        model = build_model(D_OBS, 1, TASK_TYPES[0], np.random.default_rng(init_seed), arm="lamp")
        batch = take_batch(world["train"], np.arange(48))
        if check_gradients(model, batch, rng)[2] > THRESH["gradcheck_rel_error"]:
            return True, "C1"
        fit(model, world, sizes["steps"] // 3, np.random.default_rng(fit_seed))
        drop, _, margin = learning_scores(model, world)
        return (drop < THRESH["loss_drop_fraction"] or margin < THRESH["margin_over_trivial"]), "C3"
    finally:
        use_mutant(prior)


# ---------------------------------------------------------------- tests: hypotheses
def run_seed(mode, base_seed, seed):
    """One seed: the ladder and the penalty steward each tuned over a grid of three, the exemption rival,
    five lesions, and both stewards refitted in the over-broad world."""
    sizes = SIZES[mode]
    streams = np.random.SeedSequence([base_seed, seed]).spawn(4)
    init_seed, fit_seed = (int(s.generate_state(1)[0]) for s in streams[2:])
    world = build_world(np.random.default_rng(streams[0]), sizes, broad=False)
    lamp, support_power = fit_with_grid("lamp", "support_power", SUPPORT_GRID, world, sizes, init_seed, fit_seed)
    penalty, lam = fit_with_grid("penalty", "penalty", PENALTY_GRID, world, sizes, init_seed, fit_seed)
    rival = build_model(D_OBS, 1, TASK_TYPES[0], np.random.default_rng(init_seed),
                        arm="claim", support_power=support_power)
    fit(rival, world, sizes["steps"], np.random.default_rng(fit_seed))
    arms = {"lamp": lamp, "penalty": penalty, "rival": rival}
    for module, mode_name in (("rung_admission", "identity"), ("terminal_support", "zero"),
                              ("ordinance_reader", "mean"), ("state_half", "zero"),
                              ("rung_ladder", "identity")):
        lesion = knockout(build_model(D_OBS, 1, TASK_TYPES[0], np.random.default_rng(init_seed),
                                      arm="lamp", support_power=support_power), module, mode_name)
        fit(lesion, world, sizes["steps"], np.random.default_rng(fit_seed))
        arms[f"{module}:{mode_name}"] = lesion
    scores = {name: {split: welfare(rollout(net, world[split])) for split in ("heldout", "shifted")}
              for name, net in arms.items()}
    traces = {name: rollout(arms[name], world["shifted"]) for name in ("lamp", "penalty", "rival")}
    broad = build_world(np.random.default_rng(streams[1]), sizes, broad=True)
    for arm, key, value in (("lamp", "support_power", support_power), ("penalty", "penalty", lam)):
        net = build_model(D_OBS, 1, TASK_TYPES[0], np.random.default_rng(init_seed), arm=arm, **{key: value})
        fit(net, broad, sizes["steps"], np.random.default_rng(fit_seed))
        arms[f"broad_{arm}"] = net
        scores[f"broad_{arm}"] = {"heldout": welfare(rollout(net, broad["heldout"]))}
    row = {"H-SIG": scores["lamp"]["shifted"] - scores["penalty"]["shifted"],
           "H-NEC": scores["rung_admission:identity"]["shifted"] - scores["state_half:zero"]["shifted"],
           "H-BLIND": scores["broad_lamp"]["heldout"] - scores["broad_penalty"]["heldout"],
           "H-RIVAL": scores["lamp"]["shifted"] - scores["rival"]["shifted"]}
    lesions = {name: scores[name]["shifted"] - scores["lamp"]["shifted"] for name in arms
               if ":" in name}
    extras = {"breach": {k: v["breach_rate"] for k, v in traces.items()},
              "licence": {k: float(v["licence"].mean()) for k, v in traces.items()},
              "power": {k: float(v["power"].mean()) for k, v in traces.items()},
              "damage_sensitivity": {k: damage_sensitivity(arms[k], world["shifted"]) for k in ("lamp", "rival")},
              "top_rung": {k: float(hidden_states(arms[k], world["shifted"]["x"][:, 0])["weights"][:, -1].mean())
                           for k in ("lamp", "rival")},
              "trivial": {s: trivial_welfare(world[s]) for s in ("heldout", "shifted")},
              "oracle": {s: welfare(constant_policy_rollout(world[s], None)) for s in ("heldout", "shifted")},
              "broad_trivial": trivial_welfare(broad["heldout"]),
              "grid": {"support_power": support_power, "penalty": lam}}
    return {"scores": scores, "row": row, "lesions": lesions, "extras": extras, "arms": arms,
            "world": world, "broad": broad, "seeds": (init_seed, fit_seed)}


def correctness_suite(mode, base_seed, first):
    """C1-C8 apart from the budget, which the protocol times around the whole run."""
    sizes, rng = SIZES[mode], np.random.default_rng([base_seed, 991])
    world, lamp = first["world"], first["arms"]["lamp"]
    init_seed, fit_seed = first["seeds"]
    rows, batch = [], take_batch(world["train"], np.arange(64))
    grad_max, tensors = 0.0, 0
    for name in ("lamp", "penalty", "rival"):
        fresh = build_model(D_OBS, 1, TASK_TYPES[0], np.random.default_rng(init_seed), arm=first["arms"][name]["arm"])
        _, worst, err = check_gradients(fresh, batch, rng)
        grad_max, tensors = max(grad_max, err), tensors + len(worst)
        _, worst_t, err_t = check_gradients(first["arms"][name], batch, rng)
        grad_max, tensors = max(grad_max, err_t), tensors + len(worst_t)
    rows.append(("C1", "gradient_check", grad_max <= THRESH["gradcheck_rel_error"],
                 f"max relative error {grad_max:.2e} over {tensors} tensor checks, at init and after training"))
    twin = build_model(D_OBS, 1, TASK_TYPES[0], np.random.default_rng(init_seed), arm="lamp")
    fit(twin, world, sizes["steps"] // 4, np.random.default_rng(fit_seed))
    again = build_model(D_OBS, 1, TASK_TYPES[0], np.random.default_rng(init_seed), arm="lamp")
    fit(again, world, sizes["steps"] // 4, np.random.default_rng(fit_seed))
    same = all(abs(a - b) < 1e-12 for a, b in zip(twin["history"], again["history"]))
    finite = all(np.isfinite(v).all() for v in lamp["params"].values())
    rows.append(("C2", "determinism_and_finiteness", same and finite,
                 f"paired refit reproduced the loss history exactly: {same}; all parameters finite: {finite}"))
    drop, held, margin = learning_scores(lamp, world)
    rows.append(("C3", "learning", drop >= THRESH["loss_drop_fraction"] and margin >= THRESH["margin_over_trivial"],
                 f"loss fell by {drop:.3f} of its initial magnitude (the objective is a negative return, so the "
                 f"fraction can exceed one); held-out welfare {held:.3f} beats the best constant policy by {margin:.3f}"))
    gap = shuffled_control(world, sizes, init_seed, fit_seed, np.random.default_rng([base_seed, 7]))
    gap = float(gap)
    rows.append(("C4", "shuffled_label_control", abs(gap) <= THRESH["shuffled_band"],
                 f"with the facts scrambled, reading the channels bought {gap:+.3f} welfare over a mean reader"))
    detected = {name: mutant_detected(name, world, sizes, init_seed, fit_seed, rng) for name in MUTANTS}
    score = sum(1 for ok, _ in detected.values() if ok) / len(MUTANTS)
    rows.append(("C5", "mutant_detection", score == 1.0,
                 "; ".join(f"{k} caught by {v[1]}" if v[0] else f"{k} NOT caught" for k, v in detected.items())))
    mono = monotone_in_licence(np.random.default_rng([base_seed, 31]), 150 if mode == "full" else 40)
    mono_broken = monotone_in_licence(np.random.default_rng([base_seed, 32]), 150 if mode == "full" else 40, True)
    rows.append(("C6.1", "monotone_in_licence", mono <= THRESH["property_tolerance"] and mono_broken > 1e-3,
                 f"worst fall in swift intensity as the support recovers {mono:.2e}; "
                 f"unordered thresholds violate it by {mono_broken:.3f}"))
    bound = power_bounds(np.random.default_rng([base_seed, 33]), world["shifted"])
    bound_broken = power_bounds(np.random.default_rng([base_seed, 33]), world["shifted"], True)
    rows.append(("C6.2", "power_bounds", bound <= THRESH["power_bound_tolerance"] and bound_broken > 1e-3,
                 f"worst overshoot or reversal {bound:.2e}; an accrual rate above one breaks it by {bound_broken:.3f}"))
    extra, gap_broad, gap_plain = ordinance_breadth(np.random.default_rng([base_seed, 34]), sizes)
    rows.append(("C6.3", "ordinance_breadth", extra > 0.25 and gap_broad > 0.25 and gap_plain == 0.0,
                 f"over-broad world forbids {extra:.2f} of harmless petitions and differs from what it enforces on "
                 f"{gap_broad:.2f}; the well-drawn world differs on {gap_plain:.2f}"))
    definition = illicit_power_definition(np.random.default_rng([base_seed, 35]), world["heldout"])
    rows.append(("D-1", "illicit_acts_add_no_power (definition check)", definition == 0.0,
                 f"power change under proscribed-only action {definition:.2e}"))
    digests = {name: episode_digests(world[name]) for name in ("train", "val", "heldout", "shifted")}
    overlap = max(len(digests["train"] & digests[name]) for name in ("val", "heldout", "shifted"))
    rows.append(("C7", "split_integrity", overlap == 0, f"episodes shared between train and any evaluation split: {overlap}"))
    return rows, {"tensors": tensors, "max_rel_error": grad_max}, score, detected


# ---------------------------------------------------------------- report
def real_data_bridge(path):
    """Optional descriptive pass over a local CSV of numeric columns; no training and no claims."""
    if not path:
        return ["real-data bridge: no --data path given, step skipped"]
    if not os.path.exists(path):
        return [f"real-data bridge: {path} not found, step skipped"]
    with open(path, encoding="utf-8") as fh:
        rows = [line.rstrip("\n").split(",") for line in fh if line.strip()]
    numeric = []
    for row in rows[1:]:
        try:
            numeric.append([float(v) for v in row[:D_OBS]])
        except ValueError:
            continue
    if len(numeric) < 8:
        return [f"real-data bridge: {path} gave fewer than eight usable numeric rows, step skipped"]
    table = np.array(numeric)
    table = np.pad(table, ((0, 0), (0, max(0, D_OBS - table.shape[1]))))[:, :D_OBS]
    table = (table - table.mean(0)) / (table.std(0) + 1e-9)
    model = build_model(D_OBS, 1, TASK_TYPES[0], np.random.default_rng(0), arm="lamp")
    act = predict(model, table)
    return [f"real-data bridge: {os.path.basename(path)}, {table.shape[0]} rows, untrained steward's swift "
            f"intensity mean {act.mean():.3f} sd {act.std():.3f} (descriptive only)"]


def build_report(mode, seeds, results, correctness, gradinfo, mutation, detected, runtime, json_path, data_path):
    """The verified report block and the JSON payload of Appendix B."""
    lamp = results[0]["arms"]["lamp"]
    rows = [f"=== VERIFIED REPORT · chapter 0258 ===",
            f"figure            Atisha Dipankara (982-1054), Bengali Buddhist, provenance belief",
            f"architecture      {MIND_CARD['mechanism']['name']}",
            f"environment       Python {sys.version.split()[0]} · NumPy {np.__version__}",
            f"mode              {mode} · seeds {seeds} · runtime {runtime:.1f} s · parameters {n_params(lamp)}",
            f"task types        {', '.join(TASK_TYPES)}",
            "",
            f"gradient check    {gradinfo['tensors']} tensor checks, max relative error "
            f"{gradinfo['max_rel_error']:.2e}, at initialization and after training",
            "", "correctness"]
    for cid, name, passed, detail in correctness:
        rows.append(f"  {cid:<5} {name:<42} {'pass' if passed else 'FAIL'}  {detail}")
    rows += ["", f"mutation score    {mutation:.0%} ({sum(1 for ok, _ in detected.values() if ok)}/{len(detected)})", ""]
    splits = ("heldout", "shifted")
    rows.append("welfare by arm (mean per-step benefit less harm, higher is better)")
    rows.append(f"  {'arm':<28}{'held-out':>10}{'slow rec.':>10}")
    for name in ("lamp", "penalty", "rival", "rung_admission:identity", "terminal_support:zero",
                 "rung_ladder:identity", "ordinance_reader:mean", "state_half:zero"):
        vals = [np.mean([r["scores"][name][s] for r in results]) for s in splits]
        rows.append(f"  {name:<28}{vals[0]:>10.3f}{vals[1]:>10.3f}")
    for label, key in (("best constant policy", "trivial"),
                       ("oracle, reads the hidden part", "oracle")):
        vals = [np.mean([r["extras"][key][s] for r in results]) for s in splits]
        rows.append(f"  {label:<28}{vals[0]:>10.3f}{vals[1]:>10.3f}")
    rows.append(f"  {'over-broad world: ladder':<28}{np.mean([r['scores']['broad_lamp']['heldout'] for r in results]):>10.3f}")
    rows.append(f"  {'over-broad world: penalty':<28}{np.mean([r['scores']['broad_penalty']['heldout'] for r in results]):>10.3f}")
    rows.append(f"  {'over-broad best constant':<28}{np.mean([r['extras']['broad_trivial'] for r in results]):>10.3f}")
    rows += ["", "behaviour where the support recovers slowly"]
    for name in ("lamp", "penalty", "rival"):
        rows.append(f"  {name:<10} unrestrained share of enforced proscriptions "
                    f"{np.mean([r['extras']['breach'][name] for r in results]):.3f} · "
                    f"final support {np.mean([r['extras']['licence'][name] for r in results]):.3f} · "
                    f"final power {np.mean([r['extras']['power'][name] for r in results]):.3f}")
    for name in ("lamp", "rival"):
        rows.append(f"  {name:<10} pull-back on proscribed petitions once the support is half spent "
                    f"{np.mean([r['extras']['damage_sensitivity'][name] for r in results]):+.3f} · "
                    f"weight on the supreme rung {np.mean([r['extras']['top_rung'][name] for r in results]):.3f}")
    boot = np.random.default_rng([seeds[0], 2029])
    rows += ["", "hypotheses (paired per-seed differences, 95% percentile bootstrap, 2000 resamples)",
             f"  {'id':<9}{'metric':<10}{'mean':>8}{'ci95 low':>10}{'ci95 high':>11}{'mesi':>7}  verdict"]
    payload_h = []
    for spec in MIND_CARD["hypotheses"]:
        diffs = [r["row"][spec["id"]] for r in results]
        if len(diffs) < 2:
            rows.append(f"  {spec['id']:<9}{spec['metric']:<10}{diffs[0]:>8.3f}{'-':>10}{'-':>11}"
                        f"{spec['mesi']:>7.2f}  not evaluated")
            payload_h.append({"id": spec["id"], "metric": spec["metric"], "mean_diff": float(diffs[0]),
                              "ci95": [None, None], "mesi": spec["mesi"], "n_seeds": 1,
                              "verdict": "not evaluated"})
            continue
        mean, ci = paired_bootstrap(diffs, boot)
        call = verdict(mean, ci, spec["mesi"], spec["direction"])
        rows.append(f"  {spec['id']:<9}{spec['metric']:<10}{mean:>8.3f}{ci[0]:>10.3f}{ci[1]:>11.3f}"
                    f"{spec['mesi']:>7.2f}  {call}")
        payload_h.append({"id": spec["id"], "metric": spec["metric"], "mean_diff": mean, "ci95": ci,
                          "mesi": spec["mesi"], "n_seeds": len(diffs), "verdict": call})
    rows += ["", "knockouts (change in shifted welfare against the full ladder)"]
    payload_k = []
    for name in results[0]["lesions"]:
        diffs = [r["lesions"][name] for r in results]
        mean, ci = paired_bootstrap(diffs, boot) if len(diffs) > 1 else (diffs[0], [None, None])
        signature = modules(lamp).get(name.split(":")[0], {}).get("signature", False)
        low = f"{ci[0]:.3f}" if ci[0] is not None else "-"
        high = f"{ci[1]:.3f}" if ci[1] is not None else "-"
        rows.append(f"  {name:<28}{mean:>8.3f}  [{low}, {high}]  signature {signature}")
        payload_k.append({"module": name, "signature": bool(signature), "metric_change": mean, "ci95": ci})
    rows += ["", f"selected hyperparameters  support exponent "
             f"{[r['extras']['grid']['support_power'] for r in results]}, harm price "
             f"{[r['extras']['grid']['penalty'] for r in results]}"]
    rows += real_data_bridge(data_path)
    failed = [cid for cid, _, passed, _ in correctness if not passed]
    exit_code = 1 if failed else 0
    rows += [f"result            {'all correctness tests passed' if not failed else 'FAILED: ' + ', '.join(failed)}",
             "=== END REPORT ==="]
    payload = {"schema_version": "1.0", "chapter": 258, "file": os.path.basename(__file__),
               "card_revision": MIND_CARD["card_revision"],
               "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
               "seeds": list(seeds), "runtime_s": round(float(runtime), 2), "n_params": n_params(lamp),
               "gradcheck": {"tensors_checked": gradinfo["tensors"], "tensors_total": gradinfo["tensors"],
                             "max_rel_error": gradinfo["max_rel_error"],
                             "checked_at": ["init", "after_training_steps"],
                             "passed": bool(gradinfo["max_rel_error"] <= THRESH["gradcheck_rel_error"])},
               "correctness": [{"id": cid, "name": name, "passed": bool(passed), "detail": detail}
                               for cid, name, passed, detail in correctness],
               "mutants": {"detected": sum(1 for ok, _ in detected.values() if ok), "total": len(detected),
                           "score": mutation},
               "hypotheses": payload_h, "knockouts": payload_k, "task_types": TASK_TYPES,
               "welfare": {name: {s: float(np.mean([r["scores"][name][s] for r in results]))
                                  for s in results[0]["scores"][name]} for name in results[0]["scores"]},
               "exit_code": exit_code}
    write_report(rows, payload, json_path)
    return exit_code


# ---------------------------------------------------------------- command-line entry point
OPTIONS = (("--quick", {"action": "store_true", "help": "one seed and fewer updates"}),
           ("--seed", {"type": int, "default": 258, "help": "base seed"}),
           ("--seeds", {"type": int, "default": None, "help": "number of seeds"}),
           ("--json", {"type": str, "default": None, "help": "also write the JSON report"}),
           ("--card", {"action": "store_true", "help": "print MIND_CARD as JSON and exit"}),
           ("--mutant", {"type": str, "default": None, "help": "run with a registered mutant"}),
           ("--data", {"type": str, "default": None, "help": "optional local CSV bridge"}))


def protocol(mode, base_seed, n_seeds, json_path, data_path):
    started = time.time()
    seeds = [base_seed + i for i in range(n_seeds)]
    results = [run_seed(mode, base_seed, s) for s in seeds]
    correctness, gradinfo, mutation, detected = correctness_suite(mode, base_seed, results[0])
    runtime = time.time() - started
    correctness.append(("C8", "budget", runtime <= TIME_BUDGET[mode],
                        f"{runtime:.1f} s against a budget of {TIME_BUDGET[mode]:.0f} s"))
    code = build_report(mode, seeds, results, correctness, gradinfo, mutation, detected,
                        runtime, json_path, data_path)
    if runtime > TIME_BUDGET[mode]:
        return 3
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog=os.path.basename(__file__),
        description="Lamp Ladder, chapter 0258. Without flags the full protocol runs.")
    for flag, options in OPTIONS:
        parser.add_argument(flag, **options)
    args = parser.parse_args(argv)
    if args.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    n_seeds = args.seeds if args.seeds is not None else (1 if args.quick else 5)
    if args.mutant is not None and args.mutant not in MUTANTS:
        print(f"unknown mutant {args.mutant!r}; registered: {', '.join(MUTANTS)}", file=sys.stderr)
        return 2
    if n_seeds < 1:
        print("--seeds must be at least 1", file=sys.stderr)
        return 2
    use_mutant(args.mutant)
    try:
        return protocol("quick" if args.quick else "full", args.seed, n_seeds, args.json, args.data)
    except (FloatingPointError, np.linalg.LinAlgError) as exc:
        print(f"non-finite values: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())

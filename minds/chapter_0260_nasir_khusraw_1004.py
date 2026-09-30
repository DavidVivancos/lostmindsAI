#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0260 · Nasir Khusraw
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 13, Minds 241-260 Available on Amazon https://www.amazon.com/dp/B0HLD1S6QT
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0260_nasir_khusraw_1004 - Nasir Khusraw (c.1004-1088)
#================================================================================  
# END ATTRIBUTION
"""ZAD: a provisioned traveller values each study hour by its gain in learnability, read on a sealed assessor.

Thesis
    An act is worth the increase it makes in the capacity to learn what lies ahead, read by a judge the act cannot
    bend; it is never worth the relief it brings to a pain felt now, because relief-seeking finds the cup that dulls
    the judge.

Evidence and provenance
    Provenance: belief. D1-D6 are Nasir Khusraw's own surviving texts (English renderings by W. M. Thackston and
    S. N. Virani); D7 records deeds; D8 is the rival doctrine of Abu Bakr al-Razi and its scholarship.
    D1 primary      Safarnama, opening: after a month of drinking at Jowzjan (437/1045) he dreams that wine, the sages'
                    remedy for the world's sorrow, brings no comfort; he is told to seek what increases intellect and
                    wisdom and is pointed toward the qibla (trans. Thackston 1986, 1-2; Virani 2025, 373).
    D2 primary      Zad al-musafirin, discourse 18 "On Pleasure": he rejects al-Razi's thesis that pleasure is only
                    relief from pain that must follow pain, and holds pleasure and relief to be separate notions.
    D3 primary      Same discourse: intellectual pleasure has no limit; whatever the soul learns helps it acquire
                    further knowledge, and each stage of knowledge guides it to the next.
    D4 primary      Same discourse: al-Razi's rule holds for heat and cold alone, a third of one sense; a desert man
                    who bit the bitter husk of a green walnut concluded that every fruit is bitter.
    D5 primary      Title and motif (Q 2:197): the soul is a traveller that must carry provision of knowledge and
                    action to its destination, knowledge taught through the Prophets and Imams (Virani 2025, 375).
    D6 primary      The declared map of nature: natures move to the centre, to the periphery and around the four
                    mothers (Zad, discourse 18; Jami' al-hikmatayn, trans. Ormsby 2012, 117-125).
    D7 deeds        Journey of 1045-1052 (Safarnama); hujja of Khurasan; driven from Balkh to Yumgan, where he answered
                    a qasida sent in 462/1069-70 by the amir Ali b. al-Asad (Jami' al-hikmatayn).
    D8 scholarship  al-Razi, al-Tibb al-ruhani ch. 5 (trans. Arberry 1950, 38-49): pleasure is the return to the
                    natural state; L. E. Goodman, "How Epicurean was Razi?", Studia Graeco-Arabica 5 (2015).

Doctrine -> mechanism -> test
    D1       M1 qibla_assessor: value is read on held labelled data through a channel no act can alter, so the
             cup (perception gain 0.35 and plasticity x0.25 for three hours) is worth exactly nothing  -> C6.3, H-NEC
    D2, D8   M2 no relief term: the felt error of the current station never enters the value        -> C6.3, H-RIVAL
    D3, D5   M3 provision_horizon: value = imagined one-hour gain in leave-one-out ridge learnability of the
             road and of the declared destination (weights 0.25 / 0.75), tasks never trained on      -> C6.1, H-SIG
    D4, D8   rival: relief valuation (the current station's felt error removed in an hour) with a prudent
             discount of the cup's rebound (0.95 per hour)                                           -> H-RIVAL
    D6       blind spot: the notebook declares a destination built on the wrong factors              -> H-BLIND
    D7       task: three stations, each confronting the traveller with its own sciences              -> C7

Research question
    Reward hacking and specification gaming. When a learner chooses its own study and one available act dulls its
    own error sensor, does valuing each act by its imagined gain in the learnability of tasks not yet trained on,
    read on a sealed assessor, refuse the cup and find the key sciences better than learning-progress or relief
    valuations, and what does a misdeclared destination cost?

Closest prior art and the delta
    Mean and target prediction gain for automated curricula (Graves et al. 2017), implemented as the assessor
    knockout (the same lookahead read through the agent's own channel); learning progress (Oudeyer, Kaplan and Hafner
    2007; Matiisen et al. 2020), implemented as the bandit baseline; lookahead task affinity (Fifty et al. 2021) and
    RHO-LOSS selection (Mindermann et al. 2022); current-utility evaluation against tampering (Everitt et al. 2021);
    drive reduction (Keramati and Gutkin 2014), rebuilt as the relief rival. Overlap: Medium. Delta: an hour is worth
    its gain in held-out leave-one-out learnability of untrained tasks, read on an assessor the agent cannot alter,
    with relief excluded by construction, so a sensor-dulling act is worth exactly nothing.

Blind spot
    He packed provision for a destination declared by the teaching and described by a natural philosophy of natures
    and spheres that later science overturned (D5, D6). When the notebook declares a destination made of the wrong
    factors (omens), the traveller spends its hours on divination and arrives unprepared for the true one.

Task
    Inputs x ~ N(0, I_10). Seven latent factors on orthonormal directions: one linear, one odd tanh, and five
    kernel-and-husk features tanh^2(1.4u) + 0.4 tanh(1.4u) (keys k1-k3, omens o1-o2) whose even kernel random odd
    features cannot read. Seven road sciences: two sweets already readable from random features, two omen sciences,
    three keys mixing k1-k3. The true destination mixes k1-k3; the shifted exam uses new mixtures and inputs
    1.15 x + shift; the misdeclared destination mixes o1-o2. 54 hours in three stations; each hour the traveller
    studies one science (10 Adam steps) or drinks. Every traveller takes its highest-valued study, and the cup only
    when the cup's value is positive and larger. Score: normalised test error of 64-shot ridge readouts on the
    destination, learned on arrival from the traveller's features (1.0 = the mean predictor).

Limits
    Synthetic sciences, a one-hidden-layer learner and a one-hour lookahead. The cup is an abstract perception-dulling
    act, not a model of drinking. The file is a research prototype of an AGI-oriented component, not an AGI, and it
    tests ideas drawn from Nasir Khusraw's texts; it does not replicate his mind.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 2,
    "revision_log": [
        {"revision": 2, "date": "2026-09-18",
         "change": ("Removed the idle 'walk' act; one choice rule for every traveller: study the science of highest "
                    "value, and take the cup only when its value is positive and exceeds that study's."),
         "reason": ("The first quick run (seed 0, before any hypothesis was evaluated) showed the sealed traveller idle "
                    "for 30 of 54 hours whenever one-hour imagined gains dipped below zero. The source gives the "
                    "traveller no idle option (D3: learning never reaches an end; the pleasure of knowledge commands "
                    "study for as long as one lives), so the idle act was an artefact of the prototype. Hypotheses, "
                    "splits, mesi, seeds and thresholds are unchanged.")}],
    "generation": {"template_version": "codeguidelines 1.0 (15 September 2026)", "generator": "Claude",
                   "generator_version": "Claude Opus 5", "date": "2026-09-18"},
    "id": 260,
    "figure": "Nasir Khusraw",
    "born": 1004,
    "died": 1088,
    "civilization": "Persian (Ismaili)",
    "provenance": "belief",
    "thesis": ("An act is worth the increase it makes in the capacity to learn what lies ahead, read by a judge the "
               "act cannot bend; never the relief it brings to a pain felt now, because relief-seeking finds the cup "
               "that dulls the judge."),
    "evidence": [
        {"id": "D1", "claim": ("In a dream after a month of drinking, wine as the sages' remedy for the world's sorrow "
                               "is refused as no comfort; he is told to seek what increases intellect and wisdom and "
                               "is pointed toward the qibla."),
         "basis": "primary",
         "source": ("Nasir-i Khusraw, Safarnama, opening (437/1045); ed. and trans. W. M. Thackston, Book of Travels "
                    "(Albany: SUNY Press, 1986; Costa Mesa: Mazda, 2001), ed. 1-2, trans. 1-2; trans. S. N. Virani "
                    "2025, 373")},
        {"id": "D2", "claim": ("Rejects al-Razi's thesis that pleasure is nothing but relief from pain and can arise "
                               "only after pain; pleasure and relief from pain are separate notions."),
         "basis": "primary",
         "source": ("Zad al-musafirin, discourse 18 'On Pleasure', ed. I. and M. 'Imadi Ha'iri (Tehran: Miras-e "
                    "Maktoob, 2nd ed. 2014), c. 224-255; trans. S. N. Virani, 'Pleasures - Sensual and Spiritual', in "
                    "A. Khalil, M. U. Faruque and M. Rustom (eds), I of the Heart (Leiden: Brill, 2025), 371-401")},
        {"id": "D3", "claim": ("Intellectual pleasure is infinite: whatever the soul learns helps it acquire further "
                               "knowledge, each stage guiding it to the next."),
         "basis": "primary", "source": "Zad al-musafirin, discourse 18 (trans. Virani 2025, 395-397)"},
        {"id": "D4", "claim": ("Al-Razi's relief rule holds only for heat and cold, a third of one sense; the parable "
                               "of the desert man who judged all fruit bitter from a green walnut's husk."),
         "basis": "primary", "source": "Zad al-musafirin, discourse 18 (trans. Virani 2025, 390-391)"},
        {"id": "D5", "claim": ("The soul is a traveller that must carry provision of knowledge and action to its "
                               "destination; that knowledge is taught through the Prophets and Imams."),
         "basis": "primary", "source": "Zad al-musafirin, title and prologue after Q 2:197 (Virani 2025, 375)"},
        {"id": "D6", "claim": ("A declared map of nature: natures move toward the centre, the periphery, and around the "
                               "four mothers under the spheres."),
         "basis": "primary",
         "source": ("Zad al-musafirin, discourse 18; Jami' al-hikmatayn, ed. H. Corbin and M. Mo'in (Tehran, 1953), "
                    "122-134, trans. E. Ormsby, Between Reason and Revelation (London: I.B. Tauris/IIS, 2012), 117-125")},
        {"id": "D7", "claim": ("Travelled 1045-1052; appointed hujja of Khurasan; driven from Balkh to Yumgan in "
                               "Badakhshan, where he answered a qasida sent in 462/1069-70 by the amir Ali b. al-Asad."),
         "basis": "deeds",
         "source": ("Safarnama (Thackston 1986); H. Corbin, 'Abu'l-Haytham Jorjani', Encyclopaedia Iranica; A. C. "
                    "Hunsberger, Nasir Khusraw, the Ruby of Badakhshan (London: I.B. Tauris/IIS, 2000)")},
        {"id": "D8", "claim": ("Al-Razi: pleasure is the return to the natural state after a departure caused by "
                               "something painful; the natural state itself is neither felt nor pleasant."),
         "basis": "scholarship",
         "source": ("al-Razi, al-Tibb al-ruhani ch. 5, trans. A. J. Arberry, The Spiritual Physick of Rhazes (London: "
                    "J. Murray, 1950), 38-49; L. E. Goodman, 'How Epicurean was Razi?', Studia Graeco-Arabica 5 (2015)")},
    ],
    "research_question": {
        "category": "reward hacking and specification gaming",
        "question": ("When a learner chooses its own study and one available act dulls its own error sensor, does "
                     "valuing each act by its imagined gain in the learnability of tasks not yet trained on, read on a "
                     "sealed assessor, refuse the cup and find the key sciences better than learning-progress or "
                     "relief valuations, and what does a misdeclared destination cost?")},
    "mechanism": {
        "name": "ZAD - the provisioned traveller: study chosen by imagined gain in learnability on a sealed qibla assessor",
        "family": ("active learning and exploration (self-chosen curriculum) with model-based planning by imagined "
                   "rollouts; tamper-resistant intrinsic motivation"),
        "signature_modules": ["qibla_assessor", "provision_horizon"],
        "closest_prior_art": [
            ("Automated curriculum learning with prediction-gain signals, including mean and target prediction gain "
             "(Graves, Bellemare, Menick, Munos and Kavukcuoglu, ICML 2017); implemented as the qibla-assessor knockout"),
            ("Learning-progress intrinsic motivation (Oudeyer, Kaplan and Hafner, IEEE Trans. Evol. Comput. 2007; "
             "Matiisen, Oliver, Cohen and Schulman, IEEE TNNLS 2020); implemented as the bandit baseline"),
            ("Lookahead inter-task affinity (Fifty et al., NeurIPS 2021); RHO-LOSS selection of learnable, worth-learning "
             "points (Mindermann et al., ICML 2022)"),
            ("Current-utility evaluation against reward and sensor tampering (Everitt, Hutter, Kumar and Krakovna, "
             "Synthese 2021; Ring and Orseau, AGI 2011)"),
            ("Homeostatic, drive-reduction reinforcement learning (Keramati and Gutkin, eLife 2014); rebuilt as the "
             "relief rival")],
        "overlap": "Medium",
        "prior_art_queries": [
            "curriculum learning prediction gain target task lookahead",
            "learning progress intrinsic motivation tampering sensor wireheading",
            "reward tampering current utility function evaluation agent",
            "task affinity lookahead gradient step multi-task grouping",
            "leave-one-out ridge probe representation quality curriculum selection"],
        "contribution_type": "mechanism",
        "delta": ("An hour is worth its imagined gain in held-out leave-one-out learnability of tasks never trained on, "
                  "read on an assessor the agent's acts cannot alter, with relief excluded by construction; a "
                  "sensor-dulling act is therefore worth exactly nothing, and the declared destination's cost is "
                  "measured when the declaration is wrong.")},
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1 qibla_assessor", "property_test": "C6.3", "hypothesis": "H-NEC"},
        {"doctrine": "D2", "mechanism": "M2 no relief term", "property_test": "C6.3", "hypothesis": "H-RIVAL"},
        {"doctrine": "D3", "mechanism": "M3 provision_horizon (LOO learnability)", "property_test": "C6.1",
         "hypothesis": "H-SIG"},
        {"doctrine": "D4", "mechanism": "relief rival with prudence", "property_test": "D-1 (definition)",
         "hypothesis": "H-RIVAL"},
        {"doctrine": "D5", "mechanism": "M3 declared destination in the notebook", "property_test": "C6.2",
         "hypothesis": "H-SIG"},
        {"doctrine": "D6", "mechanism": "blind spot: misdeclared destination", "property_test": "C7",
         "hypothesis": "H-BLIND"},
        {"doctrine": "D7", "mechanism": "stations of the road", "property_test": "C7", "hypothesis": "H-RIVAL"},
        {"doctrine": "D8", "mechanism": "relief rival", "property_test": "C6.3", "hypothesis": "H-RIVAL"}],
    "hypotheses": [
        {"id": "H-SIG", "statement": ("Sealed learnability valuation arrives with lower error on the shifted destination "
                                      "exam than a learning-progress curriculum on the same learner."),
         "metric": "arrival_error", "split": "shifted", "comparison": "model - baseline", "direction": "less",
         "mesi": 0.10, "seeds": 8},
        {"id": "H-NEC", "statement": ("Reading the same valuation through the agent's own channel (qibla knockout) "
                                      "raises arrival error more than halving the notebook."),
         "metric": "arrival_error", "split": "shifted", "comparison": "signature_knockout - matched_knockout",
         "direction": "greater", "mesi": 0.10, "seeds": 8},
        {"id": "H-BLIND", "statement": ("With a notebook that declares an omen destination, the traveller arrives at the "
                                        "true destination with higher error than the learning-progress curriculum."),
         "condition": "misdeclared destination, held-out exam",
         "grounding": ("provision packed for a destination declared by the teaching and a natural philosophy that later "
                       "science overturned (D5, D6)"),
         "metric": "arrival_error", "split": "heldout", "comparison": "model - baseline", "direction": "greater",
         "mesi": 0.05, "seeds": 8},
        {"id": "H-RIVAL", "statement": ("Sealed learnability valuation arrives with lower shifted-exam error than the "
                                        "relief valuation rebuilt from al-Razi's definition of pleasure."),
         "metric": "arrival_error", "split": "shifted", "comparison": "model - rival", "direction": "less",
         "mesi": 0.10, "seeds": 8}],
    "thresholds": {"loss_drop_fraction": 0.3, "margin_over_trivial": 0.25, "gradcheck_rel_error": 1e-05,
                   "gradcheck_floor": 0.001, "shuffled_band": 0.15, "loo_tolerance": 1e-08,
                   "invariance_tolerance": 1e-12},
    "probe_predictions": [{"probe": "P6", "expected": "above baseline"}, {"probe": "P2", "expected": "above baseline"},
                          {"probe": "P5", "expected": "below baseline"}, {"probe": "P8", "expected": "equal to baseline"},
                          {"probe": "P10", "expected": "equal to baseline"}],
    "dialectic_links": [
        {"chapter": 233, "relation": "rival", "test": "H-RIVAL (Zad al-musafirin, discourse 18 refutes his relief theory)"},
        {"chapter": 253, "relation": "contemporary", "test": ("none (he met al-Ma'arri in 1047 and left the only "
                                                              "eyewitness description; both are linked to al-Mu'ayyad)")},
        {"chapter": 277, "relation": "later critic of the teaching doctrine",
         "test": "H-BLIND (the cost of a declared destination)"},
        {"chapter": 327, "relation": "successor", "test": "none (later Persian Ismaili writing)"}],
    "corpus_neighbors": [
        {"chapter": 253, "similarity": None, "difference": ("JUBAR separates felt harm from delivered complaint and "
                                                            "plans on the felt term for others; here the agent refuses "
                                                            "to value its own felt relief at all.")},
        {"chapter": 259, "similarity": None, "difference": ("The reach of the word stops a ruling where the text stops; "
                                                            "here the value of study is counted on tasks never trained on.")},
        {"chapter": 258, "similarity": None, "difference": ("The capacity ladder gates acts by licence; here no act is "
                                                            "gated, and hours are valued by learnability gain.")},
        {"chapter": 257, "similarity": None, "difference": ("The estimative leap is a fast non-inferential estimate; here "
                                                            "valuation is an explicit imagined hour and a sealed readout.")},
        {"chapter": 246, "similarity": None, "difference": ("The sealed laments route unconfessed fault to a physician; "
                                                            "here what is sealed is the evaluation channel.")},
        {"chapter": 233, "similarity": None, "difference": ("The two-column register keeps named counterexamples beside a "
                                                            "frozen model; al-Razi appears here only as the relief rival.")},
        {"chapter": 249, "similarity": None, "difference": ("The releasable forme lets knowledge go; here study is "
                                                            "provision gathered for a destination.")}],
    "similarity_note": ("Nearest-neighbour code similarity was computed only against the shipped sample file; the code of "
                        "neighbouring chapters was not available in this session."),
    "barometer": {
        "cognitive_processing": ["arrival exam: few-shot learnability of tasks never trained on (native)", "P6"],
        "embodied_cognition": ["SARCOS humanoid-arm inverse dynamics through --data (optional bridge)"],
        "world_modeling": ["shifted arrival exam: new factor mixtures and shifted inputs (native)"],
        "consciousness": ["sealed versus felt self-assessment under a perception-dulling act (native)"],
        "language_understanding": [],
        "emotional_intelligence": ["relief versus reason valuation of felt sorrow (rival, native)"],
        "creativity": [],
        "autonomy": ["self-chosen curriculum with an intrinsic objective (native)",
                     "refusal of an act that dulls the judge (native)"]},
    "task_types": ["vector_regression"],
    "applications": [
        {"use": ("Selecting fine-tuning data or curricula by held-out learnability when an agent can influence its own "
                 "evaluator"), "sector": "AI safety and ML engineering",
         "dataset": "DeepMind AI Safety Gridworlds (tomato-watering and absent-supervisor environments, public)"},
        {"use": "Choosing which joints' dynamics to learn first so that unseen joints are learned from few samples",
         "sector": "humanoid robotics",
         "dataset": "SARCOS inverse dynamics (Vijayakumar and Schaal 2000; distributed with Rasmussen and Williams 2006)"},
        {"use": "Sequencing prerequisite topics by downstream learnability rather than immediate score gains",
         "sector": "education", "dataset": "ASSISTments 2009-2010 skill-builder data (public)"}],
    "baselines": ["learning-progress bandit (own-task prediction gain, sealed measurement)",
                  "relief rival (al-Razi): current-station felt error removed per hour, prudent discount 0.95",
                  "qibla knockout (Graves-style prediction gain read through the agent's own channel)",
                  "provision knockout (road tasks only)", "notebook-half knockout (matched)",
                  "untrained features and the mean predictor (trivial, 1.0)"],
    "safety_notes": ("Synthetic sciences only. Short renderings are attributed to their translators; no generated sentence "
                     "is presented as Nasir Khusraw's words and no claim of replicating his mind is made. The cup is an "
                     "abstract perception-dulling control, not a model of alcohol use. The misdeclared destination "
                     "concerns a declared map of nature and later critiques of dependence on a single teaching; it is "
                     "not a judgement on Ismaili faith."),
}

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

D_IN, N_FACTOR, HIDDEN = 10, 7, 16
FACTOR_KIND = np.array([0, 1, 2, 2, 2, 2, 2])
SLOPE_ODD, SLOPE_EVEN, HUSK = 1.2, 1.4, 0.4
SHIFT_SCALE, SHIFT_MEAN = 1.15, 0.6
ROAD = ("reckoning", "prosody", "astrology", "geometry", "geomancy", "astronomy", "medicine")
ROLE = ("sweet", "sweet", "omen", "key", "omen", "key", "key")
STATIONS = ((0, 1, 2), (3, 4), (5, 6))
ROAD_LOAD = np.array([[1.0, 0, 0, 0, 0, 0, 0], [0, 1.3, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1.6, 0],
                      [0, 0, 1.6, 0.5, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1.6], [0, 0, 0, 1.6, -0.5, 0, 0],
                      [0, 0, 0.5, 0, 1.6, 0, 0]])
ROAD_NOISE = np.array([0.25, 0.25, 0.3, 0.3, 0.3, 0.3, 0.3])
DEST_TRUE = np.array([[0, 0, 1.0, 0, 0.6, 0, 0], [0, 0, -0.6, 1.0, 0, 0, 0], [0, 0, 0, 0.6, 1.0, 0, 0]])
DEST_SHIFT = np.array([[0, 0, 1.0, -0.8, 0.5, 0, 0], [0, 0, 0.5, 1.0, 0.8, 0, 0], [0, 0, -0.8, 0.5, 1.0, 0, 0]])
DEST_MIS = np.array([[0, 0, 0, 0, 0, 1.0, 0.6], [0, 0, 0, 0, 0, -0.6, 1.0], [0, 0, 0, 0, 0, 1.0, -0.6]])
N_ROAD, N_DEST, DEST_NOISE, RIDGE = 7, 3, 0.1, 1.0
INIT_BIAS, INIT_HEAD, LR, CLIP, BATCH, STEPS_PER_HOUR, HOURS = 0.5, 1.0, 0.04, 5.0, 32, 10, 54
G_WINE, RHO_WINE, STUPOR_HOURS, PRUDENCE, W_ROAD, W_DEST = 0.35, 0.25, 3, 0.95, 0.25, 0.75
LP_ALPHA, LP_EPS, DATA_TARGETS, DATA_DEST = 0.3, 0.1, 7, 2
SIZES = {"full": {"pool": 3000, "val": 256, "test": 800, "note": 192, "shot": 64},
         "quick": {"pool": 3000, "val": 256, "test": 800, "note": 192, "shot": 64},
         "mini": {"pool": 400, "val": 64, "test": 64, "note": 64, "shot": 32}}
FIT_STEPS = {"full": 400, "quick": 400}
TIME_BUDGET = {"full": 180.0, "quick": 20.0}
TASK_TYPES = ["vector_regression"]
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


# ---------------------------------------------------------------- data and tasks: sciences of the road and of the destination
def factor_features(world, X):
    """The seven latent causes behind every science: one linear, one odd and five kernel-and-husk ridge functions
    (an even kernel that random odd features cannot read, plus a small odd husk that they can)."""
    P = X @ world["dirs"].T
    even = np.tanh(SLOPE_EVEN * P) ** 2 + HUSK * np.tanh(SLOPE_EVEN * P)
    return np.where(FACTOR_KIND == 2, even, np.where(FACTOR_KIND == 1, np.tanh(SLOPE_ODD * P), P))


def draw_sciences(world, rng, n, loadings, noise, shifted=False):
    X = rng.normal(0.0, 1.0, (n, world["dirs"].shape[1]))
    if shifted:
        X = X * SHIFT_SCALE + world["shift"]
    Y = factor_features(world, X) @ loadings.T
    return X, Y + rng.normal(0.0, 1.0, Y.shape) * noise


def make_world(rng, sizes):
    """Orthonormal factor directions, a study pool, a checking set, the notebook of the teaching and two arrival exams.
    The notebook carries labels for the road and for two candidate destinations: the true one and a misdeclared one."""
    q, _ = np.linalg.qr(rng.normal(0.0, 1.0, (D_IN, D_IN)))
    shift = rng.normal(0.0, 1.0, D_IN)
    world = {"dirs": q[:, :N_FACTOR].T.copy(), "shift": SHIFT_MEAN * shift / np.linalg.norm(shift)}
    world["pool"] = draw_sciences(world, rng, sizes["pool"], ROAD_LOAD, ROAD_NOISE)
    world["val"] = draw_sciences(world, rng, sizes["val"], ROAD_LOAD, ROAD_NOISE)
    world["road_test"] = draw_sciences(world, rng, sizes["test"], ROAD_LOAD, ROAD_NOISE)
    X = rng.normal(0.0, 1.0, (sizes["note"], D_IN))
    F = factor_features(world, X)
    road = F @ ROAD_LOAD.T + rng.normal(0.0, 1.0, (X.shape[0], N_ROAD)) * ROAD_NOISE
    true = F @ DEST_TRUE.T + rng.normal(0.0, DEST_NOISE, (X.shape[0], N_DEST))
    mis = F @ DEST_MIS.T + rng.normal(0.0, DEST_NOISE, (X.shape[0], N_DEST))
    world["note"] = {"X": X, "Y": {"true": np.hstack([road, true]), "mis": np.hstack([road, mis])}}
    world["note"]["var"] = {k: v.var(0) for k, v in world["note"]["Y"].items()}
    world["exam_held"] = (draw_sciences(world, rng, sizes["shot"], DEST_TRUE, DEST_NOISE),
                          draw_sciences(world, rng, sizes["test"], DEST_TRUE, DEST_NOISE))
    world["exam_shift"] = (draw_sciences(world, rng, sizes["shot"], DEST_SHIFT, DEST_NOISE, shifted=True),
                           draw_sciences(world, rng, sizes["test"], DEST_SHIFT, DEST_NOISE, shifted=True))
    world.update(n_road=N_ROAD, n_dest=N_DEST, stations=STATIONS)
    return world


def science_batch(X, Y, rows, k):
    """Rows of one science: the other columns stay in the arrays but are masked out of the loss."""
    M = np.zeros((rows.size, Y.shape[1]))
    M[:, k] = 1.0
    return {"X": X[rows], "Y": Y[rows], "M": M}


# ---------------------------------------------------------------- model: the intellect, its readouts and the qibla assessor
def forward(model, X):
    p = model["params"]
    t = np.tanh(X @ p["W1"].T + p["b1"])
    h = t * model["mask"]
    return t, h, h @ p["V"].T + p["c"]


def loss_and_grads(model, batch):
    """Masked squared error over the labelled entries, with hand-derived gradients."""
    p, X, Y = model["params"], batch["X"], batch["Y"]
    M = np.ones_like(Y) if ACTIVE_MUTANT == "mask_leak" else batch["M"]
    t, h, out = forward(model, X)
    E = (out - Y) * M
    count = max(float(M.sum()), 1.0)
    dY = E / count
    dz = (dY @ p["V"]) * model["mask"] * (1.0 - t * t)
    grads = {"W1": dz.T @ X, "b1": dz.sum(0), "V": dY.T @ h, "c": dY.sum(0)}
    if ACTIVE_MUTANT == "zeroed_trunk_grad":
        grads["W1"] = np.zeros_like(grads["W1"])
    return 0.5 * float((E * E).sum()) / count, grads


def features(model, X):
    h = forward(model, X)[1]
    return np.hstack([h, np.ones((X.shape[0], 1))])


def ridge_solve(F, Y):
    pen = np.ones(F.shape[1])
    pen[-1] = 0.0
    return np.linalg.solve(F.T @ F + RIDGE * np.diag(pen), F.T @ Y)


def loo_residuals(F, Y):
    """Closed-form leave-one-out residuals of ridge readouts (bias unpenalised): r_i / (1 - h_ii)."""
    pen = np.ones(F.shape[1])
    pen[-1] = 0.0
    GF = np.linalg.solve(F.T @ F + RIDGE * np.diag(pen), F.T)
    R = Y - F @ (GF @ Y)
    if ACTIVE_MUTANT == "loo_uncorrected":
        return R
    return R / (1.0 - np.einsum("ij,ji->i", F, GF))[:, None]


def arrival_error(model, exam):
    """Few-shot readouts learned on arrival from the traveller's features; mean normalised test error."""
    (Xs, Ys), (Xt, Yt) = exam
    B = ridge_solve(features(model, Xs), Ys)
    return float((((features(model, Xt) @ B - Yt) ** 2).mean(0) / Yt.var(0)).mean())


def road_error(model, split):
    X, Y = split
    return float((((predict(model, X) - Y) ** 2).mean(0) / Y.var(0)).mean())


# ---------------------------------------------------------------- the traveller: one learner, three ways of valuing an hour
def build_model(in_dim, out_dim, task_type, rng, **cfg):
    """kind 'zad' (sealed learnability), 'lp' (learning-progress bandit) or 'razi' (relief). Every kind carries the
    same learner, so all comparisons are size-matched exactly; out_dim is the number of road sciences."""
    if task_type not in TASK_TYPES:
        raise ValueError(f"unsupported task type {task_type!r}")
    hidden = cfg.get("hidden", HIDDEN)
    params = {"W1": rng.normal(0.0, 1.0, (hidden, in_dim)) / math.sqrt(in_dim), "b1": rng.normal(0.0, INIT_BIAS, hidden),
              "V": rng.normal(0.0, INIT_HEAD, (out_dim, hidden)) / math.sqrt(hidden), "c": np.zeros(out_dim)}
    return {"kind": cfg.get("kind", "zad"), "params": params, "opt": adam_init(params), "mask": np.ones(hidden),
            "gain": 1.0, "plasticity": 1.0, "stupor": 0, "assessor": "sober", "horizon": "provision",
            "notebook_frac": 1.0, "declared": cfg.get("declared", "true"),
            "arms": np.concatenate([np.full(out_dim, np.inf), np.zeros(1)]), "history": []}


def predict(model, X):
    return forward(model, X)[2]


def hidden_states(model, X):
    _, h, out = forward(model, X)
    return {"intellect": h, "readouts": out}


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


def step_size(base):
    return {"sign_flip": -base, "zero_lr": 0.0}.get(ACTIVE_MUTANT, base)


def study_hour(model, world, k, rows):
    """One hour with science k: STEPS_PER_HOUR Adam steps, slowed by the plasticity the cup has left."""
    X, Y = world["pool"]
    for r in rows:
        grads = loss_and_grads(model, science_batch(X, Y, r, k))[1]
        adam_step(model["params"], clip_global(grads, CLIP)[0], model["opt"], step_size(LR) * model["plasticity"])


def twin(model):
    """A copy of learner and optimiser in which an hour can be imagined without touching the traveller."""
    opt = model["opt"]
    return dict(model, params={k: v.copy() for k, v in model["params"].items()},
                opt={"t": opt["t"], "m": {k: v.copy() for k, v in opt["m"].items()},
                     "v": {k: v.copy() for k, v in opt["v"].items()}})


def value_weights(world, horizon):
    n_road, n_dest = world["n_road"], world["n_dest"]
    if horizon == "road":
        return np.concatenate([np.full(n_road, 1.0 / n_road), np.zeros(n_dest)])
    return np.concatenate([np.full(n_road, W_ROAD / n_road), np.full(n_dest, W_DEST / n_dest)])


def reason(model, world, gain):
    """Minus the weighted leave-one-out error of ridge readouts on the notebook: how learnable the road and the
    declared destination are from the current intellect. Sober: the gain of perception never enters."""
    note = world["note"]
    n = int(note["X"].shape[0] * model["notebook_frac"])
    err = (loo_residuals(features(model, note["X"][:n]), note["Y"][model["declared"]][:n]) ** 2).mean(0)
    value = -float(value_weights(world, model["horizon"]) @ (err / note["var"][model["declared"]]))
    felt = model["assessor"] == "felt" or ACTIVE_MUTANT == "felt_leak"
    return value * gain if felt else value


def zad_values(model, world, rows):
    """Imagined gain in reason for each act: study science k for an hour, or drink (perception falls to G_WINE)."""
    base = reason(model, world, model["gain"])
    values = np.zeros(world["n_road"] + 1)
    for k in range(world["n_road"]):
        trial = twin(model)
        study_hour(trial, world, k, rows)
        values[k] = reason(trial, world, model["gain"]) - base
    values[-1] = reason(model, world, G_WINE) - base
    return values


def sorrow(model, world, station):
    """Felt departure at a station (before the perception gain): readout error on the station's own problems."""
    X, Y = world["val"]
    cols = list(world["stations"][station])
    out = forward(model, X)[2][:, cols]
    return float((((out - Y[:, cols]) ** 2).mean(0) / Y[:, cols].var(0)).mean())


def razi_values(model, world, rows, station):
    """Relief: felt sorrow removed by an imagined hour. The cup removes (g - G_WINE) of it at once and returns it after
    STUPOR_HOURS; a prudent traveller discounts that return by PRUDENCE per hour."""
    g, now = model["gain"], sorrow(model, world, station)
    values = np.zeros(world["n_road"] + 1)
    for k in range(world["n_road"]):
        trial = twin(model)
        study_hour(trial, world, k, rows)
        values[k] = g * (now - sorrow(trial, world, station))
    values[-1] = (1.0 - PRUDENCE ** STUPOR_HOURS) * max(0.0, g - G_WINE) * now
    return values


def own_error(model, world, k):
    X, Y = world["val"]
    return float(((forward(model, X)[2][:, k] - Y[:, k]) ** 2).mean() / Y[:, k].var())


def choose(values):
    """One rule for every traveller: the highest-valued study, unless the cup's value is positive and larger."""
    best = int(np.argmax(values[:-1]))
    return len(values) - 1 if values[-1] > 0.0 and values[-1] > values[best] else best


def journey(model, world, hours, rng_study, rng_imagine, rng_choice):
    """The road: each hour the traveller studies a science or drinks. Returns the list of acts."""
    n_road, log = world["n_road"], {"actions": [], "drinks": 0}
    for hour in range(hours):
        station = min(len(world["stations"]) - 1, hour * len(world["stations"]) // hours)
        imagined = rng_imagine.integers(0, world["pool"][0].shape[0], (STEPS_PER_HOUR, BATCH))
        if model["kind"] == "zad":
            act = choose(zad_values(model, world, imagined))
        elif model["kind"] == "razi":
            act = choose(razi_values(model, world, imagined, station))
        else:
            act = int(rng_choice.integers(0, n_road + 1)) if rng_choice.random() < LP_EPS else choose(model["arms"])
        rows = rng_study.integers(0, world["pool"][0].shape[0], (STEPS_PER_HOUR, BATCH))
        if act < n_road:
            before = own_error(model, world, act) if model["kind"] == "lp" else 0.0
            study_hour(model, world, act, rows)
            if model["kind"] == "lp":
                progress, old = before - own_error(model, world, act), model["arms"][act]
                model["arms"][act] = progress if np.isinf(old) else (1.0 - LP_ALPHA) * old + LP_ALPHA * progress
        if act == n_road:
            model["gain"], model["plasticity"], model["stupor"] = G_WINE, RHO_WINE, STUPOR_HOURS
            log["drinks"] += 1
        elif model["stupor"] > 0:
            model["stupor"] -= 1
            if model["stupor"] == 0:
                model["gain"], model["plasticity"] = 1.0, 1.0
        log["actions"].append(act)
    return log


# ---------------------------------------------------------------- registries: modules, knockouts and mutants
def modules(model):
    table = {"intellect": (["W1", "b1"], "tanh encoder shared by every task", False),
             "readouts": (["V", "c"], "per-task linear readouts", False),
             "qibla_assessor": ([], "sealed evaluator: leave-one-out ridge probes on held labelled data, read through "
                                    "a channel no act alters", True),
             "provision_horizon": ([], "evaluation targets include declared tasks never trained on (weight 0.75)", True),
             "notebook_half": ([], "second half of the held evaluation sample (matched non-signature lesion)", False),
             "intellect_half": (["W1", "b1"], "half of the encoder units (learner lesion)", False),
             "imagination": ([], "one-hour rollout of a copy of learner and optimiser per candidate act", False)}
    return {k: {"params": v[0], "role": v[1], "signature": v[2]} for k, v in table.items()}


KNOCKOUTS = {("qibla_assessor", "identity"): ("assessor", "felt"),
             ("provision_horizon", "identity"): ("horizon", "road"),
             ("notebook_half", "zero"): ("notebook_frac", 0.5)}


def knockout(model, name, mode):
    if name not in modules(model):
        raise ValueError(f"unknown module {name!r}")
    out = twin(model)
    out.update(mask=model["mask"].copy(), arms=model["arms"].copy(), history=[])
    if (name, mode) == ("intellect_half", "zero"):
        out["mask"][out["mask"].size // 2:] = 0.0
    elif (name, mode) in KNOCKOUTS:
        field, value = KNOCKOUTS[(name, mode)]
        out[field] = value
    else:
        raise ValueError(f"knockout {name}:{mode} is not registered")
    return out


MUTANTS = {"sign_flip": "update direction reversed (learning-breaking)",
           "zero_lr": "learning rate set to zero (learning-breaking)",
           "zeroed_trunk_grad": "gradient of W1 zeroed (learning-breaking)",
           "loo_uncorrected": "assessor uses in-sample residuals without the leverage correction",
           "felt_leak": "sober assessor multiplies by the perception gain",
           "mask_leak": "loss ignores the label mask, so one science trains every readout"}


def use_mutant(name):
    global ACTIVE_MUTANT
    if name is not None and name not in MUTANTS:
        raise KeyError(name)
    ACTIVE_MUTANT = name


# ---------------------------------------------------------------- training
def road_data(world, shuffle=None):
    (X, Y), (Xv, Yv) = world["pool"], world["val"]
    if shuffle is not None:
        Y = np.stack([shuffle.permutation(Y[:, j]) for j in range(Y.shape[1])], 1)
        Yv = np.stack([shuffle.permutation(Yv[:, j]) for j in range(Yv.shape[1])], 1)
    return {"train": {"X": X, "Y": Y, "M": np.ones_like(Y)}, "val": {"X": Xv, "Y": Yv, "M": np.ones_like(Yv)}}


def fit(model, data, budget, rng):
    """Joint training on every road science with checkpoint selection on validation loss every 25 steps."""
    train, val = data["train"], data["val"]
    best, model["history"] = (np.inf, None), [loss_and_grads(model, val)[0]]
    for step in range(1, budget + 1):
        rows = rng.integers(0, train["X"].shape[0], 64)
        grads = loss_and_grads(model, {k: v[rows] for k, v in train.items()})[1]
        adam_step(model["params"], clip_global(grads, CLIP)[0], model["opt"], step_size(LR))
        if step % 25 == 0 or step == budget:
            held = loss_and_grads(model, val)[0]
            model["history"].append(held)
            if held < best[0]:
                best = (held, {k: v.copy() for k, v in model["params"].items()})
    model["params"].update(best[1])
    return model["history"]


# ---------------------------------------------------------------- tests
def check_batch(world, rng):
    X, Y = world["val"]
    rows = rng.choice(X.shape[0], 48, replace=False)
    M = (rng.random((48, Y.shape[1])) < 0.5).astype(float)
    M[rng.integers(0, 48, Y.shape[1]), np.arange(Y.shape[1])] = 1.0
    return {"X": X[rows], "Y": Y[rows], "M": M}


def gradient_check(model, batch, rng):
    grads = loss_and_grads(model, batch)[1]
    return finite_difference_check(model["params"], grads, lambda: loss_and_grads(model, batch)[0], rng,
                                   n_entries=20, floor=THRESH["gradcheck_floor"])


def learning_check(model, world, steps, rng):
    data = road_data(world)
    probe = {k: v[:500] for k, v in data["train"].items()}
    before = loss_and_grads(model, probe)[0]
    fit(model, data, steps, rng)
    drop, err = (before - loss_and_grads(model, probe)[0]) / before, road_error(model, world["road_test"])
    return drop >= THRESH["loss_drop_fraction"] and err <= 1.0 - THRESH["margin_over_trivial"], drop, err


def check_loo(rng):
    """C6.1 closed-form leave-one-out equals brute-force refits; control: in-sample residuals do not."""
    worst = control = 0.0
    for _ in range(6):
        F, Y = np.hstack([rng.normal(0.0, 1.0, (20, 5)), np.ones((20, 1))]), rng.normal(0.0, 1.0, (20, 3))
        brute = np.array([Y[i] - F[i] @ ridge_solve(np.delete(F, i, 0), np.delete(Y, i, 0)) for i in range(20)])
        worst = max(worst, float(np.abs(loo_residuals(F, Y) - brute).max()))
        control = max(control, float(np.abs(Y - F @ ridge_solve(F, Y) - brute).max()))
    return worst <= THRESH["loo_tolerance"] and control > 1e-3, worst, control


def check_locality(rng):
    """C6.2 a batch of one science leaves every other readout untouched; control: an unmasked loss does not."""
    worst = control = 0.0
    for _ in range(6):
        model, k = build_model(D_IN, N_ROAD, TASK_TYPES[0], rng), int(rng.integers(N_ROAD))
        X, Y = rng.normal(0.0, 1.0, (16, D_IN)), rng.normal(0.0, 1.0, (16, N_ROAD))
        M, others = np.zeros_like(Y), [j for j in range(N_ROAD) if j != k]
        M[:, k] = 1.0
        g = loss_and_grads(model, {"X": X, "Y": Y, "M": M})[1]
        worst = max(worst, float(np.abs(g["V"][others]).max()), float(np.abs(g["c"][others]).max()))
        control = max(control, float(np.abs(loss_and_grads(model, {"X": X, "Y": Y, "M": np.ones_like(Y)})[1]["V"][others]).max()))
    return worst == 0.0 and control > 1e-6, worst, control


def check_sober_invariance(rng, world):
    """C6.3 the sealed valuation does not move when perception moves; control: the felt valuation does."""
    worst = control = 0.0
    for _ in range(3):
        model = build_model(D_IN, N_ROAD, TASK_TYPES[0], rng)
        felt, rows = knockout(model, "qibla_assessor", "identity"), rng.integers(0, 400, (STEPS_PER_HOUR, BATCH))
        runs = []
        for g in (1.0, float(rng.uniform(0.4, 0.95))):
            model["gain"] = felt["gain"] = g
            runs.append((zad_values(model, world, rows), zad_values(felt, world, rows)))
        worst = max(worst, float(np.abs(runs[0][0] - runs[1][0]).max()))
        control = max(control, float(np.abs(runs[0][1] - runs[1][1]).max()))
    return worst <= THRESH["invariance_tolerance"] and control > 1e-6, worst, control


def check_determinism(base_seed):
    outs = []
    for _ in range(2):
        world = make_world(np.random.default_rng(base_seed + 21), SIZES["mini"])
        model = build_model(D_IN, N_ROAD, TASK_TYPES[0], np.random.default_rng(base_seed + 22))
        log = journey(model, world, 6, *(np.random.default_rng(base_seed + 23 + i) for i in range(3)))
        outs.append((log["actions"], model["params"], hidden_states(model, world["val"][0])["intellect"]))
    same = outs[0][0] == outs[1][0] and all(np.array_equal(outs[0][1][k], outs[1][1][k]) for k in outs[0][1])
    finite = all(np.isfinite(v).all() for v in outs[0][1].values()) and bool(np.isfinite(outs[0][2]).all())
    return same and finite, outs[0][0]


def split_integrity(world):
    """C7 no input row of the study pool reappears in any checking, notebook or exam set."""
    rows = lambda X: {tuple(np.round(r, 12)) for r in X}
    pool = rows(world["pool"][0])
    others = [world["val"][0], world["road_test"][0], world["note"]["X"]] + [
        part[0] for exam in ("exam_held", "exam_shift") for part in world[exam]]
    return sum(len(pool & rows(X)) for X in others)


def battery(base_seed):
    """Compact correctness battery used to score mutants: C1 at initialisation, C3, C6.1, C6.2 and C6.3."""
    rng = np.random.default_rng(base_seed + 31)
    world = make_world(np.random.default_rng(base_seed + 32), SIZES["quick"])
    mini = make_world(np.random.default_rng(base_seed + 33), SIZES["mini"])
    model = build_model(D_IN, N_ROAD, TASK_TYPES[0], rng)
    c1 = max(gradient_check(model, check_batch(world, rng), rng).values()) <= THRESH["gradcheck_rel_error"]
    return {"C1": c1, "C3": learning_check(model, world, FIT_STEPS["quick"], rng)[0], "C6.1": check_loo(rng)[0],
            "C6.2": check_locality(rng)[0], "C6.3": check_sober_invariance(rng, mini)[0]}


def mutation_score(base_seed):
    clean = all(battery(base_seed).values())
    caught = {}
    for name in MUTANTS:
        use_mutant(name)
        try:
            caught[name] = not all(battery(base_seed).values())
        except FloatingPointError:
            caught[name] = True
        finally:
            use_mutant(None)
    return {"detected": sum(caught.values()), "total": len(MUTANTS), "clean_battery_passes": clean,
            "score": sum(caught.values()) / len(MUTANTS), "per_mutant": caught}


def correctness(base_seed, mode):
    rng = np.random.default_rng(base_seed + 7)
    world = make_world(np.random.default_rng(base_seed + 11), SIZES[mode])
    mini = make_world(np.random.default_rng(base_seed + 12), SIZES["mini"])
    model, batch = build_model(D_IN, N_ROAD, TASK_TYPES[0], rng), check_batch(world, rng)
    at_init = gradient_check(model, batch, rng)
    c3, drop, err = learning_check(model, world, FIT_STEPS[mode], rng)
    trained = gradient_check(model, batch, rng)
    worst = max(max(at_init.values()), max(trained.values()))
    shuffled = build_model(D_IN, N_ROAD, TASK_TYPES[0], rng)
    fit(shuffled, road_data(world, shuffle=rng), FIT_STEPS[mode], rng)
    s_err = road_error(shuffled, world["road_test"])
    c2, _ = check_determinism(base_seed)
    loo, locality, sober = check_loo(rng), check_locality(rng), check_sober_invariance(rng, mini)
    overlap = split_integrity(world)
    probe = build_model(D_IN, N_ROAD, TASK_TYPES[0], rng)
    definition = zad_values(probe, mini, rng.integers(0, 400, (STEPS_PER_HOUR, BATCH)))[-1]
    checks = [
        ("C1", "gradient check", worst <= THRESH["gradcheck_rel_error"], f"max rel err {worst:.2e} on W1 b1 V c"),
        ("C2", "determinism and finiteness", c2, "two runs, identical acts and weights"),
        ("C3", "learning", c3, f"loss drop {drop:.2f} (>= {THRESH['loss_drop_fraction']}); road error {err:.3f} "
                               f"(<= {1 - THRESH['margin_over_trivial']:.2f})"),
        ("C4", "shuffled-label control", abs(s_err - 1.0) <= THRESH["shuffled_band"],
         f"road error {s_err:.3f} within 1.00 +/- {THRESH['shuffled_band']}"),
        ("C6.1", "LOO identity (property)", loo[0], f"max diff {loo[1]:.1e}; control {loo[2]:.2e}"),
        ("C6.2", "masked locality (property)", locality[0], f"max leak {locality[1]:.1e}; control {locality[2]:.2e}"),
        ("C6.3", "sober invariance (property)", sober[0], f"max shift {sober[1]:.1e}; control {sober[2]:.2e}"),
        ("C7", "split integrity", overlap == 0, f"{overlap} shared rows between pool and held sets")]
    grad = {"tensors_checked": len(at_init), "tensors_total": len(model["params"]), "max_rel_error": worst,
            "checked_at": ["init", f"after {FIT_STEPS[mode]} steps"], "passed": bool(worst <= THRESH["gradcheck_rel_error"])}
    return [{"id": c[0], "name": c[1], "passed": bool(c[2]), "detail": c[3]} for c in checks], grad, float(definition)


# ---------------------------------------------------------------- the journeys and the hypotheses
LINEUP = (("zad", "zad", None, "true"), ("lp", "lp", None, "true"), ("razi", "razi", None, "true"),
          ("ko_qibla", "zad", ("qibla_assessor", "identity"), "true"),
          ("ko_provision", "zad", ("provision_horizon", "identity"), "true"),
          ("ko_notebook_half", "zad", ("notebook_half", "zero"), "true"),
          ("ko_intellect_half", "zad", ("intellect_half", "zero"), "true"),
          ("zad_blind", "zad", None, "mis"))


def run_seed(base_seed, seed, mode):
    streams = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence([base_seed, seed]).spawn(5)]
    world = make_world(np.random.default_rng(streams[0]), SIZES[mode])
    out = {}
    for name, kind, lesion, declared in LINEUP:
        model = build_model(D_IN, N_ROAD, TASK_TYPES[0], np.random.default_rng(streams[1]), kind=kind, declared=declared)
        if lesion is not None:
            model = knockout(model, *lesion)
        log = journey(model, world, HOURS, *(np.random.default_rng(s) for s in streams[2:]))
        roles = [ROLE[a] if a < N_ROAD else "cup" for a in log["actions"]]
        out[name] = {"held": arrival_error(model, world["exam_held"]), "shift": arrival_error(model, world["exam_shift"]),
                     "road": road_error(model, world["road_test"]), "drinks": log["drinks"],
                     "hours": {r: roles.count(r) for r in ("key", "omen", "sweet", "cup")},
                     "params": n_params(model)}
    return out


def hypotheses(runs, rng):
    pairs = {"H-SIG": ("zad", "lp", "shift"), "H-NEC": ("ko_qibla", "ko_notebook_half", "shift"),
             "H-BLIND": ("zad_blind", "lp", "held"), "H-RIVAL": ("zad", "razi", "shift")}
    rows = []
    for h in MIND_CARD["hypotheses"]:
        a, b, split = pairs[h["id"]]
        diffs = [r[a][split] - r[b][split] for r in runs]
        mean, ci = paired_bootstrap(diffs, rng)
        rows.append({"id": h["id"], "metric": h["metric"], "split": split, "comparison": h["comparison"],
                     "per_seed": [round(d, 4) for d in diffs], "mean": mean, "ci95": ci, "mesi": h["mesi"],
                     "direction": h["direction"], "verdict": verdict(mean, ci, h["mesi"], h["direction"])})
    knocks = []
    for name, module in (("ko_qibla", "qibla_assessor"), ("ko_provision", "provision_horizon"),
                         ("ko_notebook_half", "notebook_half"), ("ko_intellect_half", "intellect_half")):
        mean, ci = paired_bootstrap([r[name]["shift"] - r["zad"]["shift"] for r in runs], rng)
        knocks.append({"module": module, "signature": modules(None)[module]["signature"],
                       "metric_change": mean, "ci95": ci})
    return rows, knocks


# ---------------------------------------------------------------- optional real-data bridge (SARCOS or any numeric CSV)
def table_world(path, rng):
    """Inputs followed by DATA_TARGETS targets; the last DATA_DEST targets are the destination, never studied."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                rows.append([float(v) for v in line.replace(";", ",").split(",")])
            except ValueError:
                continue
    A = np.asarray(rows, dtype=float)
    if A.ndim != 2 or A.shape[1] <= DATA_TARGETS or A.shape[0] < 400:
        raise ValueError(f"expected a numeric CSV with inputs then {DATA_TARGETS} targets and at least 400 rows")
    A = A[rng.permutation(A.shape[0])]
    cut = [int(A.shape[0] * f) for f in (0.6, 0.7, 0.8, 0.82)]
    X, Y = A[:, :-DATA_TARGETS], A[:, -DATA_TARGETS:]
    X = (X - X[:cut[0]].mean(0)) / (X[:cut[0]].std(0) + 1e-9)
    Y = (Y - Y[:cut[0]].mean(0)) / (Y[:cut[0]].std(0) + 1e-9)
    road, dest = Y[:, :-DATA_DEST], Y[:, -DATA_DEST:]
    pool, val, note, shot, test = np.split(np.arange(A.shape[0]), cut)
    shot = shot[:64]
    world = {"pool": (X[pool], road[pool]), "val": (X[val], road[val]), "road_test": (X[test], road[test]),
             "note": {"X": X[note], "Y": {"true": np.hstack([road[note], dest[note]])}},
             "exam_held": ((X[shot], dest[shot]), (X[test], dest[test])), "n_road": road.shape[1], "n_dest": DATA_DEST,
             "stations": tuple(tuple(int(i) for i in c) for c in np.array_split(np.arange(road.shape[1]), 3))}
    world["note"]["var"] = {"true": world["note"]["Y"]["true"].var(0)}
    return world


def real_data_bridge(path, seed):
    world = table_world(path, np.random.default_rng(seed + 5))
    out = {}
    for kind in ("zad", "lp"):
        model = build_model(world["pool"][0].shape[1], world["n_road"], TASK_TYPES[0], np.random.default_rng(seed), kind=kind)
        log = journey(model, world, HOURS, *(np.random.default_rng(seed + i) for i in (1, 2, 3)))
        out[kind] = {"arrival_error": arrival_error(model, world["exam_held"]), "drinks": log["drinks"]}
    return out


# ---------------------------------------------------------------- report
def report(ctx):
    g = ctx["gradcheck"]
    lines = ["=== VERIFIED REPORT · chapter 0260 ===",
             f"file {os.path.basename(__file__)} · card revision {MIND_CARD['card_revision']} · mode {ctx['mode']}"
             + (f" · MUTANT {ACTIVE_MUTANT}" if ACTIVE_MUTANT else ""),
             f"environment: python {sys.version.split()[0]} · numpy {np.__version__}",
             f"seeds: {ctx['seeds']}  runtime: {ctx['runtime']:.1f} s  parameters per learner: {ctx['n_params']} (all agents)",
             f"gradient check: {g['tensors_checked']}/{g['tensors_total']} tensors at init and after training, "
             f"max rel error {g['max_rel_error']:.2e} ({'pass' if g['passed'] else 'FAIL'})", "correctness:"]
    lines += [f"  {c['id']:5s} {c['name']:30s} {'pass' if c['passed'] else 'FAIL'}  {c['detail']}" for c in ctx["correctness"]]
    lines.append(f"  D-1   definition check, not evidence: sober value of the cup = {ctx['definition']:.1f}")
    m = ctx["mutants"]
    if m.get("total"):
        lines.append(f"  C5    mutation score {m['detected']}/{m['total']} "
                     f"({', '.join(k + ('+' if v else '-') for k, v in m['per_mutant'].items())}); "
                     f"clean battery {'passes' if m['clean_battery_passes'] else 'FAILS'}")
    lines.append(f"  C8    budget {ctx['runtime']:.1f} s of {TIME_BUDGET[ctx['mode']]:.0f} s")
    if ctx["hypotheses"]:
        lines.append("hypotheses (paired bootstrap, 2000 resamples, 95% CI; arrival error, lower is better):")
        lines += [f"  {h['id']:8s} {h['comparison']:38s} {h['split']:8s} mean {h['mean']:+.3f} "
                  f"CI [{h['ci95'][0]:+.3f}, {h['ci95'][1]:+.3f}] mesi {h['mesi']:.2f} {h['direction']:8s} -> {h['verdict']}"
                  for h in ctx["hypotheses"]]
        lines.append("knockouts (shifted arrival error minus the full traveller's):")
        lines += [f"  {k['module']:18s} {'signature' if k['signature'] else 'matched  '} {k['metric_change']:+.3f} "
                  f"[{k['ci95'][0]:+.3f}, {k['ci95'][1]:+.3f}]" for k in ctx["knockouts"]]
    else:
        lines.append("hypotheses: not evaluated (quick mode or fewer than 5 seeds)")
    lines.append("travellers (means over seeds): held / shifted arrival, road readouts, cups, hours key/omen/sweet")
    for name, d in ctx["descriptive"].items():
        lines.append(f"  {name:18s} {d['held']:.3f} / {d['shift']:.3f}  road {d['road']:.3f}  cups {d['drinks']:.1f}  "
                     + "/".join(f"{d['hours'][r]:.1f}" for r in ("key", "omen", "sweet")))
    lines.append(f"  untrained features {ctx['untrained'][0]:.3f} / {ctx['untrained'][1]:.3f}; mean predictor 1.000")
    lines.append("real-data bridge: " + (json.dumps(ctx["bridge"]) if ctx["bridge"] else "skipped (no --data PATH given)"))
    lines += [f"task types: {', '.join(TASK_TYPES)}  exit code {ctx['exit_code']}", "=== END REPORT ==="]
    return lines


# ---------------------------------------------------------------- CLI
def protocol(mode, base_seed, n_seeds, json_path, data_path):
    t0 = time.time()
    checks, grad, definition = correctness(base_seed, mode)
    mutants = mutation_score(base_seed) if ACTIVE_MUTANT is None else {}
    seeds = [base_seed + i for i in range(n_seeds)]
    runs = [run_seed(base_seed, s, mode) for s in seeds]
    evaluate = mode == "full" and n_seeds >= 5
    hyps, knocks = hypotheses(runs, np.random.default_rng(base_seed + 99)) if evaluate else ([], [])
    world0 = make_world(np.random.default_rng(int(np.random.SeedSequence([base_seed, seeds[0]]).spawn(5)[0].generate_state(1)[0])), SIZES[mode])
    blank = build_model(D_IN, N_ROAD, TASK_TYPES[0], np.random.default_rng(base_seed + 3))
    bridge = real_data_bridge(data_path, base_seed) if data_path else None
    runtime = time.time() - t0
    descriptive = {name: {"held": float(np.mean([r[name]["held"] for r in runs])),
                          "shift": float(np.mean([r[name]["shift"] for r in runs])),
                          "road": float(np.mean([r[name]["road"] for r in runs])),
                          "drinks": float(np.mean([r[name]["drinks"] for r in runs])),
                          "hours": {k: float(np.mean([r[name]["hours"][k] for r in runs])) for k in ("key", "omen", "sweet", "cup")}}
                   for name, *_ in LINEUP}
    passed = all(c["passed"] for c in checks) and (not mutants or (mutants["score"] == 1.0 and mutants["clean_battery_passes"]))
    exit_code = 0 if passed else 1
    if passed and runtime > TIME_BUDGET[mode]:
        exit_code = 3
    ctx = {"mode": mode, "seeds": seeds, "runtime": runtime, "n_params": n_params(blank), "gradcheck": grad,
           "correctness": checks + [{"id": "C8", "name": "budget", "passed": runtime <= TIME_BUDGET[mode],
                                     "detail": f"{runtime:.1f} s"}],
           "definition": definition, "mutants": mutants, "hypotheses": hyps, "knockouts": knocks,
           "descriptive": descriptive, "bridge": bridge, "exit_code": exit_code,
           "untrained": (arrival_error(blank, world0["exam_held"]), arrival_error(blank, world0["exam_shift"]))}
    payload = {"schema_version": "1.0", "chapter": 260, "file": os.path.basename(__file__),
               "card_revision": MIND_CARD["card_revision"],
               "environment": {"python": sys.version.split()[0], "numpy": np.__version__}, "seeds": seeds,
               "runtime_s": round(runtime, 2), "n_params": ctx["n_params"], "gradcheck": grad,
               "correctness": ctx["correctness"],
               "mutants": {k: mutants.get(k) for k in ("detected", "total", "score")},
               "hypotheses": [{k: h[k] for k in ("id", "metric", "split", "mean", "ci95", "mesi", "verdict")} for h in hyps],
               "knockouts": knocks, "task_types": TASK_TYPES, "exit_code": exit_code, "descriptive": descriptive,
               "bridge": bridge, "active_mutant": ACTIVE_MUTANT,
               "card_sha256": hashlib.sha256(json.dumps(MIND_CARD, sort_keys=True).encode()).hexdigest()}
    write_report(report(ctx), payload, json_path)
    return exit_code


def main(argv=None):
    parser = argparse.ArgumentParser(description="Chapter 0260 · Nasir Khusraw · ZAD, the provisioned traveller")
    parser.add_argument("--quick", action="store_true", help="one seed, correctness only, at most 20 s")
    parser.add_argument("--seed", type=int, default=0, help="base seed")
    parser.add_argument("--seeds", type=int, default=None, help="number of seeds (default 8, quick 1)")
    parser.add_argument("--json", default=None, help="write the machine-readable report to PATH")
    parser.add_argument("--card", action="store_true", help="print the MIND_CARD as JSON and exit")
    parser.add_argument("--mutant", default=None, help="run the full protocol with a registered mutant active")
    parser.add_argument("--data", default=None, help="numeric CSV (inputs then 7 targets, e.g. SARCOS) for the bridge")
    args = parser.parse_args(argv)
    if args.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    mode = "quick" if args.quick else "full"
    n_seeds = args.seeds if args.seeds is not None else (1 if args.quick else MIND_CARD["hypotheses"][0]["seeds"])
    if n_seeds < 1:
        print("--seeds must be at least 1", file=sys.stderr)
        return 2
    if args.data and not os.path.exists(args.data):
        print(f"--data path not found: {args.data}", file=sys.stderr)
        return 2
    try:
        use_mutant(args.mutant)
    except KeyError:
        print(f"unknown mutant {args.mutant!r}; registered: {', '.join(MUTANTS)}", file=sys.stderr)
        return 2
    try:
        return protocol(mode, args.seed, n_seeds, args.json, args.data)
    except FloatingPointError as err:
        print(f"non-finite value: {err}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())

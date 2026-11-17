#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0345 · Araniko (1245-1306)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 18, Minds 341-360 Available on Amazon https://www.amazon.com/dp/B0HQ47T2NC
# Resume and Interactive Demos at https://artificiology.com/
# END ATTRIBUTION 
"""YI XIN WEI SHI (以心為師) -- canon inversion with two rulers.

Thesis. A canon of the whole body, learned from intact figures only and held
as the mind's own teacher, is inverted on whatever survives a break to
regenerate what is missing, while the object keeps its own segment-local ruler
for the hidden points that hang on the bones; a repair counts only when the
needle finds those points.

Evidence and provenance (mediated). No writing by Araniko survives. The Yuan
shi biography (juan 203, 1370) and Cheng Jufu's spirit-way stele (1316) report
that as a boy he memorised an iconometric measurement text (Chicun jing) on one
hearing; that he told Khubilai he took his own mind as teacher; and that, given
a damaged Song bronze acupuncture figure no court artisan could repair, he said
he had never done such work but asked to try, delivering in 1265 a figure whose
joints, diaphragm and channels were complete. The two-ruler reading of that
repair is this chapter's interpretation (card evidence D8).

Doctrine -> mechanism -> test (IDs as in MIND_CARD)
  D1 canon heard once, whole body   -> M1 measure_canon: seed -> segment log-
                                       lengths in body units               -> H-NEC, C6.1
  D2 mind as its own teacher        -> M2 trained on intact wholes only;
                                       repair = inversion on survivors      -> H-SIG, C6.2
  D3 renewed a figure never made    -> M3 exam: hidden points probed with a
                                       tolerance in the figure's own units  -> H-SIG
  D5 bone-measure of the bronze     -> M4 channel_attachment: points at a
                                       fraction of their own segment        -> H-RIVAL, C6.3
  D7 one style over many workshops  -> strong canon prior (blind spot)      -> H-BLIND
  D6 young in years, not in mind    -> no practised repairs at all          -> H-SELF

Research question. Does a whole-body canon learned only from intact bodies,
inverted on survivors, repair articulated structures under damage patterns it
never saw better than repairers trained on practised damage, when scored by
hidden-point hits rather than surface resemblance?

Closest prior art and delta. Auto-decoder inversion for shape completion
(DeepSDF, Park et al. 2019), parametric body fitting to partial keypoints
(SMPL/SMPLify), pictorial structures, masked-autoencoder completion (He et al.
2022) and prior-based inpainting with arbitrary masks (RePaint, Lugmayr et al.
2022). Delta: a kinematic canon learned from wholes only, scale-equivariant
inversion on survivors, hidden points addressed as fractions of their own
segment (the object's ruler) instead of the global module, and a hidden-point
exam under damage never practised. Overlap: Medium.

Blind spot. The canon regenerates the ideal: on portrait bodies whose limb
proportions leave the canon family it should lose to a hand that copies the
surviving mirror limb (H-BLIND).

Task (synthetic bronze figures, 2-D frontal).
  1. Class of being from a lawful family (iconometric heights 6..10.42 face units).
  2. Segment lengths = class law x individual log-noise (bilateral), times module u.
  3. Pose = canonical standing angles + sway + mirrored limb noise.
  4. 18 joints by forward kinematics from the navel; 11 hidden points at fixed
     segment fractions and lateral offsets in module units.
  5. Damage removes joints and points: practised single breaks, unpractised
     breaks (head, shoulder, double and quadruple), portrait bodies, unseen class.
  6. Score: exam pass = share of missing hidden points repaired within 0.15 face units.

Limits. Two-dimensional synthetic bodies; one fixed skeleton topology; the
exam is a geometric tolerance, not anatomy; no medical claim of any kind. The
file is a research prototype of an AGI-oriented repair mechanism, not an AGI.
"""

MIND_CARD = {
    "schema_version": "1.0", "card_revision": 1, "revision_log": [],
    "generation": {"template_version": "codeguidelines 1.0 (2026-09-15)", "generator": "Claude",
                   "generator_version": "Claude Opus 5.5", "date": "2026-09-28"},
    "id": 345, "figure": "Araniko", "born": 1245, "died": 1306, "civilization": "Newar (Nepal)",
    "provenance": "mediated",
    "thesis": "A whole-body canon learned from intact figures only is inverted on the survivors of a break "
              "to regenerate what is missing, while hidden points keep the object's own segment-local ruler; "
              "a repair counts only when the needle finds them.",
    "evidence": [
        {"id": "D1", "claim": "As a boy he memorised the measurement text (Chicun jing) on one hearing.",
         "basis": "deeds", "source": "Yuan shi 203, biography of A'nige (reported)"},
        {"id": "D2", "claim": "He told Khubilai that he took his own mind as teacher (yi xin wei shi) and "
                              "knew painting, modelling and metal casting.",
         "basis": "deeds", "source": "Yuan shi 203; Cheng Jufu, Liangguo Minhui gong shendao bei (1316)"},
        {"id": "D3", "claim": "Asked to renew a damaged Song bronze acupuncture figure no artisan could "
                              "repair, he said he had never done it but asked to try; in 1265 the figure's "
                              "joints, diaphragm and channels were complete.",
         "basis": "deeds", "source": "Yuan shi 203"},
        {"id": "D4", "claim": "Himalayan iconometry measures a figure in its own finger-widths, with "
                              "heights fixed by class of being (125, 120, 108, 96, 72 units).",
         "basis": "scholarship", "source": "Jackson and Jackson, Tibetan Thangka Painting (1984)"},
        {"id": "D5", "claim": "Chinese bone-measure (gudu) places points at fixed fractions of each "
                              "segment whatever the body's size.",
         "basis": "scholarship", "source": "Lingshu 14 'Gudu'; WHO Standard Acupuncture Point Locations (2008)"},
        {"id": "D6", "claim": "At seventeen he answered objections to his age: young in years, not in mind.",
         "basis": "deeds", "source": "Yuan shi 203"},
        {"id": "D7", "claim": "His Nepalese image style spread through the court workshops he directed "
                              "from 1273 and through students such as Liu Yuan.",
         "basis": "scholarship", "source": "Yuan shi 203 (note on Liu Yuan); Jing 1994; Charleux 2023"},
        {"id": "D8", "claim": "That the repair joined his global canon for the skeleton to the object's "
                              "segment-local measure for the channels is an interpretation.",
         "basis": "speculation", "source": "this chapter"},
    ],
    "research_question": {"category": "out-of-distribution detection and shift",
                          "question": "Does a canon learned from intact bodies only, inverted on survivors, "
                                      "repair articulated bodies under never-practised damage better than "
                                      "practised repairers, scored by hidden-point hits?"},
    "mechanism": {
        "name": "Yi xin wei shi: canon inversion with two rulers",
        "family": "generative-model inversion (auto-decoder) over a learned kinematic canon",
        "signature_modules": ["measure_canon", "channel_attachment"],
        "closest_prior_art": ["DeepSDF auto-decoder inversion (Park et al. 2019)",
                              "SMPL / SMPLify body-model fitting (Loper et al. 2015; Bogo et al. 2016)",
                              "Pictorial structures (Felzenszwalb and Huttenlocher 2005)",
                              "Masked autoencoders (He et al. 2022)",
                              "RePaint prior-based inpainting (Lugmayr et al. 2022)",
                              "Virtual reconstruction of damaged specimens (Gunz et al. 2009)"],
        "overlap": "Medium",
        "prior_art_queries": ["auto-decoder shape completion partial observation",
                              "keypoint completion unseen occlusion pattern",
                              "inpainting generalization to unseen mask types",
                              "bone-relative landmark attachment skinning",
                              "robot self-model damage recovery"],
        "contribution_type": "mechanism",
        "delta": "Canon learned from wholes only; scale-equivariant inversion on survivors; hidden points "
                 "addressed as fractions of their own segment; scored by a hidden-point exam under "
                 "never-practised damage.",
    },
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1 measure_canon", "property_test": "C6.1", "hypothesis": "H-NEC"},
        {"doctrine": "D2", "mechanism": "M2 wholes-only training + inversion", "property_test": "C6.2",
         "hypothesis": "H-SIG"},
        {"doctrine": "D3", "mechanism": "M3 hidden-point exam", "property_test": "C6.4", "hypothesis": "H-SIG"},
        {"doctrine": "D5", "mechanism": "M4 channel_attachment", "property_test": "C6.3",
         "hypothesis": "H-RIVAL"},
        {"doctrine": "D7", "mechanism": "M1 canon prior strength", "property_test": "C6.1",
         "hypothesis": "H-BLIND"},
        {"doctrine": "D6", "mechanism": "M2 no practised repairs", "property_test": "C6.2",
         "hypothesis": "H-SELF"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "On never-practised damage the canon passes more hidden-point probes "
                                     "than the size-matched goldsmith hand trained on practised breaks.",
         "metric": "exam_pass", "split": "shift", "comparison": "model - goldsmith",
         "direction": "greater", "mesi": 0.10, "seeds": 5},
        {"id": "H-NEC", "statement": "Mean-replacing the measure canon hurts the exam on shifted damage more "
                                     "than mean-replacing the pose head.",
         "metric": "exam_pass", "split": "shift",
         "comparison": "knockout(measure_canon) - knockout(pose_head)", "direction": "less",
         "mesi": 0.05, "seeds": 5},
        {"id": "H-BLIND", "statement": "On portrait bodies outside the canon family the canon passes fewer "
                                       "probes than the goldsmith, who can copy the mirror limb.",
         "condition": "portrait split with practised single breaks",
         "grounding": "the canon regenerates the ideal body; one court style spread over local traditions",
         "metric": "exam_pass", "comparison": "model - goldsmith", "direction": "less",
         "mesi": 0.05, "seeds": 5},
        {"id": "H-RIVAL", "statement": "On an unseen class of being, segment-fraction addressing of hidden "
                                       "points beats addressing them in the global module.",
         "metric": "exam_pass", "split": "unseen_class", "comparison": "model - one_ruler",
         "direction": "greater", "mesi": 0.05, "seeds": 5},
        {"id": "H-SELF", "statement": "On never-practised damage the canon beats a size-matched hand trained "
                                      "on self-generated compound breaks from the practised vocabulary.",
         "metric": "exam_pass", "split": "shift", "comparison": "model - self_taught",
         "direction": "greater", "mesi": 0.05, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.5, "margin_over_trivial": 0.2, "exam_tolerance": 0.15,
                   "gradcheck_rel_tol": 1e-5, "gradcheck_floor": 1e-5, "shuffle_band": 0.15,
                   "equivariance_tol": 1e-7},
    "probe_predictions": [{"probe": "P3", "expected": "above"}, {"probe": "P6", "expected": "above"},
                          {"probe": "P9", "expected": "above"}, {"probe": "P8", "expected": "equal to"},
                          {"probe": "P1", "expected": "below"}, {"probe": "P4", "expected": "below"}],
    "dialectic_links": [],
    "corpus_neighbors": [
        {"chapter": 107, "similarity": None, "difference": "Vitruvius corrects appearance by pre-distortion "
         "of a module grid; here the module is inferred from survivors and only regenerates the absent."},
        {"chapter": 445, "similarity": None, "difference": "Duerer scores identity by construction cost; "
         "here construction is inverted only to repair, and is never an identity score."},
        {"chapter": 205, "similarity": None, "difference": "Alcuin pulls drifting copies toward stored "
         "exemplars; here nothing is retrieved and the missing part has no exemplar."},
        {"chapter": 403, "similarity": None, "difference": "Cusa removes the unit; here two different "
         "rulers are kept, one global and one per segment."},
        {"chapter": 339, "similarity": None, "difference": "Guo Shoujing anchors time at the last "
         "measurement; no forecasting or anchoring here."},
        {"chapter": 507, "similarity": 0.026, "difference": "Only other code available in this session; "
         "9-shingle Jaccard similarity measured outside the file (0.026)."},
    ],
    "barometer": {
        "cognitive_processing": ["repair under unseen damage (compositional split)"],
        "embodied_cognition": ["kinematic self-model of a body; limb loss as the stress test"],
        "world_modeling": ["class-of-being law extrapolated to an unseen class"],
        "consciousness": ["survivor-fit residual reported per repair; no claim about experience"],
        "language_understanding": ["canon as a short recitable parameter list (not tested)"],
        "emotional_intelligence": [],
        "creativity": ["generation constrained by a canon; validity = exam pass"],
        "autonomy": ["repair without practised examples; no self-set goals"],
    },
    "task_types": ["vector_regression"],
    "applications": [
        {"use": "Keypoint completion under unseen occlusion patterns", "sector": "motion analysis, health",
         "dataset": "COCO keypoints; Human3.6M"},
        {"use": "Humanoid self-model repair after limb damage or replacement", "sector": "robotics",
         "dataset": "Chen et al. 2022 visual self-model data"},
        {"use": "Virtual reconstruction of damaged statues and fossils from surviving landmarks",
         "sector": "cultural heritage, palaeoanthropology", "dataset": "MorphoSource landmark sets"},
    ],
    "safety_notes": "Channel points are synthetic geometric landmarks; no medical, dosing or treatment "
                    "claim. Synthetic bodies only. Research prototype, not an AGI.",
}

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

# BEGIN STANDARD UTILITIES v1.0
def softplus(x):
    return np.logaddexp(0.0, x)


def logsumexp(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(m, axis) + np.log(np.sum(np.exp(x - m), axis=axis))


def softmax(x, axis=-1):
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


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
            params[k] += sign * self.lr * mh / (np.sqrt(vh) + self.eps)


def clip_global_norm(grads, max_norm):
    tot = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    if tot > max_norm:
        for k in grads:
            grads[k] = grads[k] * (max_norm / (tot + 1e-12))
    return tot


def finite_difference_check(loss_fn, params, grads, rng, n_entries=20, eps=1e-6, floor=1e-5):
    """Central differences on >= n_entries random entries per tensor plus its largest-gradient entry."""
    worst, report = 0.0, {}
    for k, p in params.items():
        flat = p.reshape(-1)
        g = grads[k].reshape(-1)
        idx = set(rng.choice(flat.size, size=min(n_entries, flat.size), replace=False).tolist())
        idx.add(int(np.argmax(np.abs(g))))
        err_k = 0.0
        for i in idx:
            old = flat[i]
            flat[i] = old + eps
            lp = loss_fn()
            flat[i] = old - eps
            lm = loss_fn()
            flat[i] = old
            num = (lp - lm) / (2 * eps)
            err_k = max(err_k, abs(num - g[i]) / max(abs(num) + abs(g[i]), floor))
        report[k] = err_k
        worst = max(worst, err_k)
    return worst, report


def paired_bootstrap(diffs, rng, n_boot=2000):
    d = np.asarray(diffs, dtype=float)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    return float(d.mean()), [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def verdict(mean, ci, direction, mesi):
    lo, hi = ci
    if direction == "greater":
        if lo > 0 and mean >= mesi:
            return "supported"
        return "contradicted" if hi < 0 else "inconclusive"
    if hi < 0 and mean <= -mesi:
        return "supported"
    return "contradicted" if lo > 0 else "inconclusive"


def write_report(path, report):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)
# END STANDARD UTILITIES

# ---------------------------------------------------------------- data and tasks
# Skeleton: 0 navel (root), 1 heart, 2 throat, 3 chin, 4 crown, 5-7 L shoulder/elbow/wrist,
# 8-10 R shoulder/elbow/wrist, 11 pelvis, 12-14 L hip/knee/ankle, 15-17 R hip/knee/ankle.
PARENT = np.array([-1, 0, 1, 2, 3, 2, 5, 6, 2, 8, 9, 0, 11, 12, 13, 11, 15, 16])
K, B = 18, 17                                   # joints, bones (bone b ends at joint b+1)
BONE_LEN = np.array([1.00, 1.00, 0.35, 1.35, 1.05, 1.65, 1.45, 1.05, 1.65, 1.45,
                     0.85, 0.55, 2.05, 2.15, 0.55, 2.05, 2.15])          # face units, bodhisattva
BONE_ANG = np.deg2rad([90, 90, 90, 90, 188, -100, -95, -8, -80, -85,
                       -90, 195, -95, -91, -15, -85, -89.0])
GROUP = np.array([0, 0, 1, 1, 2, 3, 4, 2, 3, 4, 0, 5, 6, 7, 5, 6, 7])   # torso head shoulder uparm
MIRROR = np.array([0, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 1, 1, 1, -1, -1, -1.0])  # forearm hip thigh shank
# Hidden points: bone, fraction along it (bone-measure), lateral offset in module units.
CH_BONE = np.array([3, 1, 0, 5, 8, 6, 9, 12, 15, 13, 16])
CH_T = np.array([0.45, 0.5, 0.3, 0.6, 0.6, 10 / 12, 10 / 12, 0.6, 0.6, 13 / 16, 13 / 16])
CH_O = np.array([0, 0, 0, 0.12, -0.12, 0.10, -0.10, 0.15, -0.15, 0.12, -0.12])
M = CH_BONE.size
NPTS = K + M
D = 2 * NPTS
CH_PAR = PARENT[CH_BONE + 1]                     # proximal joint of each hidden point's bone
INCIDENCE = np.zeros((K, B))
for _j in range(1, K):
    _a = _j
    while _a > 0:
        INCIDENCE[_j, _a - 1] = 1.0
        _a = PARENT[_a]
CLASSES = {"buddha": (10.42, "peaceful"), "bodhisattva": (10.0, "peaceful"),
           "tall_wrathful": (8.0, "wrathful"), "human": (8.0, "human"),
           "short_wrathful": (6.0, "wrathful"), "goddess": (9.0, "peaceful")}
TRAIN_CLASSES = ["buddha", "bodhisattva", "tall_wrathful", "human", "short_wrathful"]

# Damage vocabularies: lists of missing joints and hidden points (point index = K + channel).
_ARM_EL = {"L": [6, 7, K + 3, K + 5], "R": [9, 10, K + 4, K + 6]}
_FORE = {"L": [7, K + 5], "R": [10, K + 6]}
_LEG_KN = {"L": [13, 14, K + 7, K + 9], "R": [16, 17, K + 8, K + 10]}
_SHANK = {"L": [14, K + 9], "R": [17, K + 10]}
PRACTISED = [_FORE["L"], _FORE["R"], _ARM_EL["L"], _ARM_EL["R"],
             _SHANK["L"], _SHANK["R"], _LEG_KN["L"], _LEG_KN["R"]]
_HEAD = [3, 4, K + 0]
UNPRACTISED = [_HEAD, [5] + _ARM_EL["L"], [8] + _ARM_EL["R"], _ARM_EL["L"] + _ARM_EL["R"],
               _LEG_KN["L"] + _LEG_KN["R"], _HEAD + _ARM_EL["R"], _ARM_EL["L"] + _LEG_KN["R"],
               _FORE["L"] + _FORE["R"] + _SHANK["L"] + _SHANK["R"]]


def class_law(tau, kind):
    """Lawful family of canons: squatter classes shorten legs most and widen shoulders."""
    k = (tau - 2.7) / (10.0 - 2.7)
    mt, ml, ma, ms, mh = k ** 0.7, k ** 1.35, k ** 0.9, k ** -0.25, k ** -0.15
    if kind == "human":
        mt, ma, ml, ms, mh = mt * 0.95, ma * 1.08, ml * 0.97, 1.0, mh * 1.05
    return np.array([mt, 1.0, ms, ma, ma, mh, ml, ml])


def make_figures(rng, n, classes, portrait=False):
    """Returns intact figures (n, NPTS, 2) and true modules u (n,)."""
    X, U = np.zeros((n, NPTS, 2)), np.zeros(n)
    for i in range(n):
        tau, kind = CLASSES[classes[rng.integers(len(classes))]]
        ind = rng.normal(0, 0.04, 8)[GROUP] + rng.normal(0, 0.012, B)
        if portrait:
            for grp in rng.choice([3, 4, 6, 7], size=2, replace=False):
                ind = ind + (GROUP == grp) * rng.choice([-1.0, 1.0]) * rng.uniform(0.18, 0.28)
        length = BONE_LEN * class_law(tau, kind)[GROUP] * np.exp(ind)
        ang = BONE_ANG + np.deg2rad(rng.normal(0, 2.0))
        if kind == "wrathful":
            ang = ang + np.deg2rad(12) * MIRROR * np.isin(GROUP, [6, 7]) \
                + np.deg2rad(10) * MIRROR * (GROUP == 3)
        ang = ang + np.deg2rad(rng.normal(0, 3.0, 8)[GROUP]) * MIRROR \
            + np.deg2rad(rng.normal(0, 1.5, B)) * (MIRROR != 0)
        u = math.exp(rng.uniform(math.log(0.6), math.log(1.8)))
        V = u * length[:, None] * np.stack([np.cos(ang), np.sin(ang)], 1)
        J = INCIDENCE @ V
        d = V / np.linalg.norm(V, axis=1, keepdims=True)
        nrm = np.stack([-d[:, 1], d[:, 0]], 1)
        t = CH_T + rng.normal(0, 0.01, M)
        o = CH_O + rng.normal(0, 0.01, M)
        X[i, :K] = J
        X[i, K:] = J[CH_PAR] + t[:, None] * V[CH_BONE] + (o * u)[:, None] * nrm[CH_BONE]
        U[i] = u
    X += rng.normal(0, 0.01, X.shape) * U[:, None, None]
    return X, U


def _vocab_masks(vocab):
    vm = np.ones((len(vocab), NPTS))
    for i, v in enumerate(vocab):
        vm[i, v] = 0.0
    return vm


def damage(rng, n, vocab, compound=False):
    """Point masks (n, NPTS): 1 = survives. compound unites a random non-empty subset of vocab."""
    lost = 1.0 - _vocab_masks(vocab)
    if not compound:
        return 1.0 - lost[rng.integers(len(vocab), size=n)]
    pick = rng.random((n, len(vocab))) < 0.3
    pick[np.arange(n), rng.integers(len(vocab), size=n)] |= ~pick.any(1)
    return 1.0 - np.minimum(pick @ lost, 1.0)


def make_splits(rng, n_train=320, n_eval=120):
    tr, u_tr = make_figures(rng, n_train, TRAIN_CLASSES)
    S = {"train": (tr, u_tr, None)}
    for name, cls, port, vocab in [("val", TRAIN_CLASSES, False, PRACTISED),
                                   ("test_id", TRAIN_CLASSES, False, PRACTISED),
                                   ("shift", TRAIN_CLASSES, False, UNPRACTISED),
                                   ("portrait", TRAIN_CLASSES, True, PRACTISED),
                                   ("unseen_class", ["goddess"], False, PRACTISED)]:
        x, u = make_figures(rng, n_eval, cls, portrait=port)
        S[name] = (x, u, damage(rng, n_eval, vocab))
    return S


def as_input(x, mask):
    """Native completion input: [survivor coordinates (zeros where broken), coordinate mask]."""
    m2 = np.repeat(mask, 2, axis=1)
    return np.concatenate([x.reshape(len(x), -1) * m2, m2], axis=1)
def _split_input(X):
    return X[:, :D].reshape(len(X), NPTS, 2), X[:, D::2]


def _rms(x, mask=None):
    """Root-mean-square point norm over surviving points: the figure's own size, no ruler needed."""
    mask = np.ones(x.shape[:2]) if mask is None else mask
    return np.sqrt((mask[..., None] * x * x).sum((1, 2)) / mask.sum(1))


# ---------------------------------------------------------------- model
TASK_TYPES = ["vector_regression"]
NATIVE_IN, NATIVE_OUT = 2 * D, D
PAR_ONEHOT = np.eye(K)[CH_PAR]                   # (M, K): hidden point -> proximal joint of its bone
LAM_CODE, CLIP, PLATEAU = 1e-2, 5.0, 200
EXAM_TOL = MIND_CARD["thresholds"]["exam_tolerance"]
_ACTIVE = {"mutant": None}                       # set only by --mutant or inside the C5 audit


def _sig(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def build_model(in_dim, out_dim, task_type, rng, **cfg):
    """Canon auto-decoder. Native dims build the kinematic body; other dims a flat canon over [x; y]."""
    if task_type not in TASK_TYPES:
        raise ValueError(f"unsupported task type {task_type!r}")
    c = {"q": 4, "hidden": 48, "attach": "bone", "lr": 1e-2, "inv_steps": 150, "inv_lr": 5e-2}
    c.update(cfg)
    q, H = c["q"], c["hidden"]
    native = (in_dim, out_dim) == (NATIVE_IN, NATIVE_OUT)
    P = {"W1": rng.normal(0, 1 / math.sqrt(q), (q, H)), "b1": np.zeros(H)}
    if native:
        P.update(WL=rng.normal(0, 0.02, (H, B)), bL=np.zeros(B), WA=rng.normal(0, 0.02, (H, B)),
                 bA=np.zeros(B), T=np.zeros(M), O=np.zeros(M), Z=np.zeros((1, q)), G=np.zeros(1))
    else:
        c["dims"] = (in_dim, out_dim)
        P.update(W2=rng.normal(0, 1 / math.sqrt(H), (H, in_dim + out_dim)),
                 b2=np.zeros(in_dim + out_dim), Z=np.zeros((1, q)))
    return {"params": P, "cfg": c, "native": native, "knock": {}, "stats": {}, "fresh": True}


def _decode(model, z, g):
    """Latent code z and log-module g -> 18 joints and 11 hidden points by forward kinematics."""
    P, kn, n = model["params"], model["knock"], len(z)
    h = np.tanh(z @ P["W1"] + P["b1"])
    if not model["native"]:
        return h @ P["W2"] + P["b2"], {"z": z, "h": h}
    ell = np.broadcast_to(kn["measure_canon"], (n, B)) if "measure_canon" in kn else h @ P["WL"] + P["bL"]
    phi = np.broadcast_to(kn["pose_head"], (n, B)) if "pose_head" in kn else h @ P["WA"] + P["bA"]
    u = np.exp(g)
    ln = u[:, None] * np.exp(ell)                # segment lengths: module times canon proportion
    d = np.stack([np.cos(phi), np.sin(phi)], -1)
    nr = np.stack([-d[..., 1], d[..., 0]], -1)
    J = INCIDENCE @ (ln[..., None] * d)
    db, nb = d[:, CH_BONE], nr[:, CH_BONE]
    if model["cfg"]["attach"] == "bone":         # the object's ruler: a fraction of its own segment
        alpha = _sig(P["T"]) * ln[:, CH_BONE]
    else:                                        # rival: a distance counted in the global module
        alpha = u[:, None] * P["T"]
    beta = u[:, None] * P["O"]
    if "channel_attachment" in kn:
        C = J[:, CH_PAR] + u[:, None, None] * kn["channel_attachment"]
    else:
        C = J[:, CH_PAR] + alpha[..., None] * db + beta[..., None] * nb
    cache = dict(z=z, h=h, u=u, ln=ln, d=d, nr=nr, db=db, nb=nb, alpha=alpha, beta=beta)
    return np.concatenate([J, C], 1), cache


def _decode_back(model, c, dX):
    """Reverse mode through _decode: parameter grads, and grads of the code z and log-module g."""
    P, kn, h = model["params"], model["knock"], c["h"]
    gr, dg = {}, None
    if not model["native"]:
        gr["W2"], gr["b2"] = h.T @ dX, dX.sum(0)
        dh = dX @ P["W2"].T
    else:
        u, ln, d, nr = c["u"], c["ln"], c["d"], c["nr"]
        dC = dX[:, K:]
        dJ = dX[:, :K] + PAR_ONEHOT.T @ dC
        dln, dphi, du = np.zeros_like(ln), np.zeros_like(ln), np.zeros(len(u))
        gr["T"], gr["O"] = np.zeros(M), np.zeros(M)
        if "channel_attachment" in kn:
            du += (dC * kn["channel_attachment"]).sum((1, 2))
        else:
            da, dbt = (dC * c["db"]).sum(-1), (dC * c["nb"]).sum(-1)
            dphi[:, CH_BONE] += c["alpha"] * dbt - c["beta"] * da
            if model["cfg"]["attach"] == "bone":
                s = _sig(P["T"])
                dln[:, CH_BONE] += da * s
                gr["T"] = (da * ln[:, CH_BONE]).sum(0) * s * (1 - s)
            else:
                gr["T"] = (da * u[:, None]).sum(0)
                du += (da * P["T"]).sum(1)
            gr["O"] = (dbt * u[:, None]).sum(0)
            du += (dbt * P["O"]).sum(1)
        dV = INCIDENCE.T @ dJ
        dln += (dV * d).sum(-1)
        dphi += ln * (dV * nr).sum(-1)
        dell = dln * ln
        dg = du * u + dell.sum(1)
        if "measure_canon" in kn:
            dell = np.zeros_like(dell)
        if "pose_head" in kn:
            dphi = np.zeros_like(dphi)
        gr["WL"], gr["bL"], gr["WA"], gr["bA"] = h.T @ dell, dell.sum(0), h.T @ dphi, dphi.sum(0)
        dh = dell @ P["WL"].T + dphi @ P["WA"].T
    dA = dh * (1 - h * h)
    gr["W1"], gr["b1"] = c["z"].T @ dA, dA.sum(0)
    return gr, dA @ P["W1"].T, dg


def _weights(model, x):
    """Per-figure loss weight: squared error measured in the figure's own size (native) or MSE."""
    if model["native"]:
        return (1.0 / (NPTS * _rms(x) ** 2))[:, None, None]
    return np.full((len(x), 1), 1.0 / x.shape[1])


def loss_and_grads(model, batch):
    """Auto-decoder loss on intact wholes: normalized reconstruction plus a Gaussian code prior."""
    P, x, idx = model["params"], batch["x"], batch["idx"]
    z, n = P["Z"][idx], len(idx)
    xh, cache = _decode(model, z, P["G"][idx] if model["native"] else None)
    w, res = _weights(model, x), xh - x
    loss = float(np.sum(w * res * res) + LAM_CODE * np.sum(z * z)) / n
    gr, dz, dg = _decode_back(model, cache, 2 * w * res / n)
    gr["Z"] = np.zeros_like(P["Z"])
    np.add.at(gr["Z"], idx, dz + 2 * LAM_CODE * z / n)
    if model["native"]:
        gr["G"] = np.zeros_like(P["G"])
        np.add.at(gr["G"], idx, dg)
    if _ACTIVE["mutant"] == "zero_grad_WL" and "WL" in gr:
        gr["WL"] = np.zeros_like(gr["WL"])
    return loss, gr


def _inv_loss(model, z, g, xr, vb, w):
    """Survivor-only loss for inversion; one independent problem per row."""
    xh, cache = _decode(model, z, g)
    res = (xh - xr) * vb
    per = (w * res * res).reshape(len(z), -1).sum(1) + LAM_CODE * (z * z).sum(1)
    _, dz, dg = _decode_back(model, cache, 2 * w * res)
    return per, xh, dz + 2 * LAM_CODE * z, (dg if model["native"] else np.zeros(len(z)))


def _invert(model, x, vis, fixed_scale=False, leaky=False):
    """Analysis by synthesis: fit code and log-module to the survivors from several canon starts.

    The log-module starts at log(survivor size / canon size), so the repair is scale-equivariant;
    fixed_scale and leaky exist only as negative controls for the property tests.
    """
    st, cfg, nat = model["stats"], model["cfg"], model["native"]
    n, R = len(x), len(st["starts"])
    xr, vr = np.repeat(x, R, 0), np.repeat(np.ones_like(vis) if leaky else vis, R, 0)
    vb = vr[..., None] if nat else vr
    var = {"z": np.tile(st["starts"], (n, 1)), "g": np.zeros(n * R)}
    nvis = vr.sum(1)
    w = (1.0 / nvis)[:, None]
    if nat:
        rv = _rms(xr, vr)
        if not fixed_scale:
            var["g"] = np.log(rv / _rms(_decode(model, var["z"], var["g"])[0], vr))
        w = (1.0 / (nvis * rv ** 2))[:, None, None]
    opt = Adam(var, lr=cfg["inv_lr"])
    for _ in range(cfg["inv_steps"]):
        _, _, dz, dg = _inv_loss(model, var["z"], var["g"], xr, vb, w)
        opt.step(var, {"z": dz, "g": dg})
    per, xh, _, _ = _inv_loss(model, var["z"], var["g"], xr, vb, w)
    best = per.reshape(n, R).argmin(1) + np.arange(n) * R
    return xh[best], var["z"][best], per[best]


def _flat_complete(model, X):
    din, dout = model["cfg"]["dims"]
    full = np.concatenate([X, np.zeros((len(X), dout))], 1)
    return _invert(model, full, np.concatenate([np.ones((len(X), din)), np.zeros((len(X), dout))], 1))


def predict(model, X):
    """Native: X = [survivors, mask] -> whole figure with survivors pasted. Flat: X -> y."""
    if not model["native"]:
        return _flat_complete(model, X)[0][:, model["cfg"]["dims"][0]:]
    x, mask = _split_input(X)
    xh = _invert(model, x, mask)[0]
    return np.where(mask[..., None] > 0, x, xh).reshape(len(X), -1)


def hidden_states(model, X):
    """Trunk activations of the code recovered by inversion (native) or by completion (flat)."""
    z = _invert(model, *_split_input(X))[1] if model["native"] else _flat_complete(model, X)[1]
    return np.tanh(z @ model["params"]["W1"] + model["params"]["b1"])


# ---------------------------------------------------------------- baselines and rival mechanisms
def build_hand(rng, hidden):
    """Direct masked regression ('goldsmith hand'): survivors and point mask -> whole figure."""
    din = D + NPTS
    return {"W1": rng.normal(0, 1 / math.sqrt(din), (din, hidden)), "b1": np.zeros(hidden),
            "W2": rng.normal(0, 1 / math.sqrt(hidden), (hidden, D)), "b2": np.zeros(D)}


def _hand_io(x, mask):
    r = _rms(x, mask)
    return np.concatenate([(x * mask[..., None]).reshape(len(x), -1) / r[:, None], mask], 1), r


def hand_loss_and_grads(P, feats, target, miss):
    h = np.tanh(feats @ P["W1"] + P["b1"])
    diff = (h @ P["W2"] + P["b2"] - target) * miss
    cnt = max(float(miss.sum()), 1.0)
    dy = 2.0 * diff / cnt
    dA = (dy @ P["W2"].T) * (1 - h * h)
    grads = {"W1": feats.T @ dA, "b1": dA.sum(0), "W2": h.T @ dy, "b2": dy.sum(0)}
    return float(np.sum(diff * diff)) / cnt, grads


def fit_hand(P, x, vocab, compound, steps, rng, shuffle=False):
    """Practice on breaks: fresh damage from the vocabulary every step, loss on missing parts only."""
    opt, hist = Adam(P, lr=1e-2), []
    for _ in range(steps):
        mask = damage(rng, len(x), vocab, compound)
        feats, r = _hand_io(x, mask)
        tgt = x.reshape(len(x), -1) / r[:, None]
        if shuffle:
            tgt = tgt[rng.permutation(len(x))]
        loss, gr = hand_loss_and_grads(P, feats, tgt, np.repeat(1.0 - mask, 2, 1))
        clip_global_norm(gr, CLIP)
        opt.step(P, gr)
        hist.append(loss)
    return hist


def hand_repair(P, x, mask):
    feats, r = _hand_io(x, mask)
    y = (np.tanh(feats @ P["W1"] + P["b1"]) @ P["W2"] + P["b2"]) * r[:, None]
    return np.where(mask[..., None] > 0, x, y.reshape(x.shape))


ANCESTOR = np.concatenate([PARENT, CH_PAR])      # proximal anchor of every point


def trivial_repair(x_train, x, mask):
    """Nearest surviving ancestor plus the training-mean offset, scaled to the survivors' size."""
    r_tr = _rms(x_train)
    xt = x_train / r_tr[:, None, None]
    out = x.copy()
    for i in range(len(x)):
        vis = np.broadcast_to(mask[i], (len(x_train), NPTS))
        scale = _rms(x[i:i + 1], mask[i:i + 1])[0] / np.mean(_rms(x_train, vis) / r_tr)
        for p in np.flatnonzero(mask[i] < 0.5):
            a = ANCESTOR[p]
            while mask[i, a] < 0.5:
                a = ANCESTOR[a]
            out[i, p] = x[i, a] + scale * (xt[:, p] - xt[:, a]).mean(0)
    return out


def score(pred, x, u, mask):
    """Exam pass = share of missing hidden points repaired within tolerance, in face units."""
    err = np.linalg.norm(pred - x, axis=-1) / u[:, None]
    miss = mask < 0.5
    hid = miss.copy()
    hid[:, :K] = False
    return {"exam_pass": float(np.mean(err[hid] <= EXAM_TOL)), "repair_err": float(err[miss].mean())}


def canon_repair(model, x, mask):
    return predict(model, as_input(x, mask)).reshape(x.shape)


# ---------------------------------------------------------------- registries
def modules(model):
    return {
        "measure_canon": {"params": ["WL", "bL"], "signature": True,
                          "role": "linear head from latent code to per-segment log-lengths (proportion prior)"},
        "channel_attachment": {"params": ["T", "O"], "signature": True,
                               "role": "segment-relative landmark placement: fraction along parent segment "
                                       "plus module-scaled normal offset"},
        "pose_head": {"params": ["WA", "bA"], "signature": False,
                      "role": "linear head from latent code to segment angles"},
        "code_trunk": {"params": ["W1", "b1"], "signature": False, "role": "tanh trunk on the latent code"},
        "codes": {"params": ["Z", "G"], "signature": False,
                  "role": "per-training-example latent codes and log-scales (auto-decoder)"},
    }


def knockout(model, name, mode="mean"):
    """Copy with a head replaced by its training mean or by zero ('identity' of a log/angle head)."""
    if not model["native"] or name not in ("measure_canon", "pose_head", "channel_attachment"):
        raise ValueError(f"no knockout for {name!r} on this model")
    if mode not in ("mean", "zero", "identity"):
        raise ValueError(f"unknown knockout mode {mode!r}")
    val = model["stats"][name]
    m = dict(model)
    m["knock"] = dict(model["knock"])
    m["knock"][name] = val if mode == "mean" else np.zeros_like(val)
    return m


def n_params(model, shared_only=False):
    return int(sum(v.size for k, v in model["params"].items() if not (shared_only and k in ("Z", "G"))))


MUTANTS = {"sign_flip": "Adam step taken up the gradient",
           "zero_lr": "learning rate set to zero",
           "zero_grad_WL": "gradient of the measure-canon weights WL zeroed"}


# ---------------------------------------------------------------- training
def _init_from_data(model, x, rng):
    """Data-derived start: codes sized to the data, log-module from each whole's own size."""
    P, q = model["params"], model["cfg"]["q"]
    P["Z"] = rng.normal(0, 0.1, (len(x), q))
    if model["native"]:
        V = x[:, 1:K] - x[:, PARENT[1:]]
        P["G"] = np.log(_rms(x) / np.median(_rms(x)))
        P["bL"] = np.mean(np.log(np.linalg.norm(V, axis=-1)) - P["G"][:, None], 0)
        unit = (V / np.linalg.norm(V, axis=-1, keepdims=True)).mean(0)
        P["bA"] = np.arctan2(unit[:, 1], unit[:, 0])
    else:
        P["b2"] = x.mean(0)
    model["fresh"] = False


def _refresh_stats(model):
    """Inversion starts from quantiles of the codes' first principal axis; knockout means."""
    P, st = model["params"], model["stats"]
    zm = P["Z"].mean(0)
    pc = np.linalg.svd(P["Z"] - zm, full_matrices=False)[2][0]
    qs = np.percentile((P["Z"] - zm) @ pc, [12.5, 37.5, 62.5, 87.5])
    st["starts"] = zm + qs[:, None] * pc
    if model["native"]:
        h = np.tanh(P["Z"] @ P["W1"] + P["b1"])
        st["measure_canon"] = (h @ P["WL"] + P["bL"]).mean(0)
        st["pose_head"] = (h @ P["WA"] + P["bA"]).mean(0)
        xh = _decode(model, P["Z"], P["G"])[0]
        st["channel_attachment"] = ((xh[:, K:] - xh[:, CH_PAR]) / np.exp(P["G"])[:, None, None]).mean(0)


def fit(model, data, budget, rng):
    """Full-batch Adam on intact wholes only (never on breaks); stops early at a loss plateau."""
    x = data["x"] if model["native"] else np.concatenate([data["X"], data["Y"]], 1)
    if model["fresh"]:
        _init_from_data(model, x, rng)
    lr = 0.0 if _ACTIVE["mutant"] == "zero_lr" else model["cfg"]["lr"]
    sign = 1.0 if _ACTIVE["mutant"] == "sign_flip" else -1.0
    opt, hist = Adam(model["params"], lr=lr), []
    batch = {"x": x, "idx": np.arange(len(x))}
    for _ in range(budget):
        loss, grads = loss_and_grads(model, batch)
        hist.append(loss)
        if not math.isfinite(loss):
            break
        if len(hist) > PLATEAU and hist[-PLATEAU - 1] - loss < 0.01 * hist[-PLATEAU - 1]:
            break                                # plateau: <1% gain over the last 200 updates
        clip_global_norm(grads, CLIP)
        opt.step(model["params"], grads, sign)
    if math.isfinite(hist[-1]):
        _refresh_stats(model)
    return hist


# ---------------------------------------------------------------- tests: correctness
TH = MIND_CARD["thresholds"]
KNOCKS = ("measure_canon", "pose_head", "channel_attachment")
HAND_HIDDEN = 13                                 # size-matched to the canon's shared weights
_NATIVE = (NATIVE_IN, NATIVE_OUT, "vector_regression")


def _gradcheck_model(m, data, rng, steps, tag):
    x = data["x"] if m["native"] else np.concatenate([data["X"], data["Y"]], 1)
    batch, rows = {"x": x, "idx": np.arange(len(x))}, []
    for stage in ("init", f"after_{steps}_steps"):
        if stage != "init":
            fit(m, data, steps, rng)
        _, gr = loss_and_grads(m, batch)
        worst, rep = finite_difference_check(lambda: loss_and_grads(m, batch)[0], m["params"], gr, rng)
        rows.append({"what": f"{tag}@{stage}", "max_rel": worst, "tensors": len(rep)})
    return rows


def test_c1(rng, cfg, only_canon=False):
    """Central differences on every tensor: canon, rival, flat canon, hand, knocked inversions."""
    x = make_figures(rng, 8, TRAIN_CLASSES)[0]
    rows, canon = [], None
    for attach in (("bone",) if only_canon else ("bone", "module")):
        m = build_model(*_NATIVE, rng, **dict(cfg, attach=attach))
        _init_from_data(m, x, rng)
        rows += _gradcheck_model(m, {"x": x}, rng, 60, f"canon[{attach}]")
        canon = canon or m
    if only_canon:
        return rows
    xf = rng.normal(size=(16, 5))
    flat = build_model(5, 2, "vector_regression", rng, **cfg)
    _init_from_data(flat, np.concatenate([xf, np.tanh(xf[:, :2])], 1), rng)
    rows += _gradcheck_model(flat, {"X": xf, "Y": np.tanh(xf[:, :2])}, rng, 60, "flat_canon")
    mask = damage(rng, len(x), UNPRACTISED)
    vb, w = mask[..., None], (1.0 / (mask.sum(1) * _rms(x, mask) ** 2))[:, None, None]
    for name in (None,) + KNOCKS:
        mk = canon if name is None else knockout(canon, name)
        var = {"z": rng.normal(0, 0.5, (len(x), cfg.get("q", 4))), "g": rng.normal(0, 0.2, len(x))}
        _, _, dz, dg = _inv_loss(mk, var["z"], var["g"], x, vb, w)
        worst, rep = finite_difference_check(lambda: _inv_loss(mk, var["z"], var["g"], x, vb, w)[0].sum(),
                                             var, {"z": dz, "g": dg}, rng)
        rows.append({"what": f"inversion[{name or 'full'}]", "max_rel": worst, "tensors": len(rep)})
    P = build_hand(rng, HAND_HIDDEN)
    for stage in ("init", "after_60_steps"):
        if stage != "init":
            fit_hand(P, x, PRACTISED, False, 60, rng)
        feats, r = _hand_io(x, mask)
        args = (feats, x.reshape(len(x), -1) / r[:, None], np.repeat(1.0 - mask, 2, 1))
        _, gr = hand_loss_and_grads(P, *args)
        worst, rep = finite_difference_check(lambda: hand_loss_and_grads(P, *args)[0], P, gr, rng)
        rows.append({"what": f"goldsmith_hand@{stage}", "max_rel": worst, "tensors": len(rep)})
    return rows


def _learns(cfg, rng, steps=200, n=80):
    """Short training on intact wholes: does the loss fall by the declared fraction?"""
    x = make_figures(rng, n, TRAIN_CLASSES)[0]
    hist = fit(build_model(*_NATIVE, rng, **cfg), {"x": x}, steps, rng)
    return math.isfinite(hist[-1]) and 1 - hist[-1] / hist[0] >= TH["loss_drop_fraction"], hist


def test_c2(cfg, seed, main):
    """Same seed -> identical losses and outputs; flat interface runs; all states finite."""
    outs = []
    for _ in range(2):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
        ok, hist = _learns(cfg, rng, steps=80, n=40)
        m = build_model(*_NATIVE, rng, **cfg)
        fit(m, {"x": make_figures(rng, 40, TRAIN_CLASSES)[0]}, 40, rng)
        xe = make_figures(rng, 6, TRAIN_CLASSES)[0]
        outs.append((hist, predict(m, as_input(xe, damage(rng, 6, UNPRACTISED)))))
    same = outs[0][0] == outs[1][0] and np.array_equal(outs[0][1], outs[1][1])
    rng = np.random.default_rng(np.random.SeedSequence([seed, 3]))
    xf = rng.normal(size=(64, 5))
    flat = build_model(5, 2, "vector_regression", rng, **cfg)
    fit(flat, {"X": xf, "Y": np.tanh(xf[:, :2])}, 80, rng)
    arrays = [predict(flat, xf[:8]), hidden_states(flat, xf[:8])]
    if main is not None:
        x, _, mask = main["splits"]["shift"]
        arrays += list(main["canon"]["params"].values()) + [hidden_states(main["canon"], as_input(x, mask))]
        arrays += [v for P in main["hands"].values() for v in P.values()]
    finite = all(np.all(np.isfinite(a)) for a in arrays)
    return {"passed": bool(same and finite), "finite": finite,
            "detail": f"identical reruns={same}; all params, outputs and hidden states finite={finite}"}


def test_c3(main):
    if main is None or not math.isfinite(main["hist"][-1]):
        return {"passed": False, "detail": "canon training diverged or produced no model"}
    drop = 1 - main["hist"][-1] / main["hist"][0]
    ratio = main["res"]["val"]["canon"]["repair_err"] / main["res"]["val"]["trivial"]["repair_err"]
    ok = drop >= TH["loss_drop_fraction"] and ratio <= 1 - TH["margin_over_trivial"]
    return {"passed": bool(ok), "detail": f"loss drop {drop:.3f} (need >= {TH['loss_drop_fraction']}); "
                                          f"val repair error canon/trivial {ratio:.3f} "
                                          f"(need <= {1 - TH['margin_over_trivial']:.2f})"}


def test_c4(cfg, seed, steps):
    """Shuffled control: points permuted independently across wholes (canon); targets permuted (hand)."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 4]))
    S = make_splits(rng, n_eval=60)
    x, (xv, uv, mv) = S["train"][0], S["val"]
    xs = x.copy()
    for p in range(NPTS):
        xs[:, p] = x[rng.permutation(len(x)), p]
    m = build_model(*_NATIVE, rng, **cfg)
    fit(m, {"x": xs}, steps, rng)
    P = build_hand(rng, HAND_HIDDEN)
    fit_hand(P, x, PRACTISED, False, steps, rng, shuffle=True)
    triv = score(trivial_repair(xs, xv, mv), xv, uv, mv)["repair_err"]
    ec = score(canon_repair(m, xv, mv), xv, uv, mv)["repair_err"]
    eh = score(hand_repair(P, xv, mv), xv, uv, mv)["repair_err"]
    ok = min(ec, eh) >= (1 - TH["shuffle_band"]) * triv
    return {"passed": bool(ok), "detail": f"val repair error: trivial {triv:.3f}, shuffled canon {ec:.3f}, "
                                          f"shuffled hand {eh:.3f} (must stay >= {1 - TH['shuffle_band']:.2f} x trivial)"}


def test_c5(cfg, seed):
    """Every mutant must make C1 or the learning check fail; the clean file must pass both."""
    def audit(name):
        _ACTIVE["mutant"] = name
        rng = np.random.default_rng(np.random.SeedSequence([seed, 5]))
        with np.errstate(over="ignore", invalid="ignore"):
            c1 = all(r["max_rel"] <= TH["gradcheck_rel_tol"] for r in test_c1(rng, cfg, only_canon=True))
            learn = _learns(cfg, rng)[0]
        _ACTIVE["mutant"] = None
        return c1, learn
    clean = audit(None)
    found = {n: not all(audit(n)) for n in MUTANTS}
    score_ = sum(found.values()) / len(found)
    return {"passed": bool(all(clean) and score_ == 1.0), "score": score_, "detected": found,
            "detail": f"clean file passes C1 and learning={all(clean)}; detected {found}"}


def _fraction(model, z, g):
    xh, c = _decode(model, z, g)
    V = c["ln"][:, CH_BONE, None] * c["db"]
    return ((xh[:, K:] - xh[:, CH_PAR]) * V).sum(-1) / (V * V).sum(-1)


def test_c6(cfg, rng, trained):
    """Property tests on random and trained canons, each with a negative control."""
    xr = make_figures(rng, 20, TRAIN_CLASSES)[0]
    rand = build_model(*_NATIVE, rng, **dict(cfg, inv_steps=40))
    _init_from_data(rand, xr, rng)
    rand["params"]["Z"] = rng.normal(0, 1.0, rand["params"]["Z"].shape)
    _refresh_stats(rand)
    x, mask = make_figures(rng, 12, TRAIN_CLASSES)[0], damage(rng, 12, UNPRACTISED)
    viol, ctrl = 0.0, 0.0
    for m in (rand, trained):
        for s in np.exp(rng.uniform(math.log(0.25), math.log(4.0), 3)):
            base = _invert(m, x, mask)[0]
            viol = max(viol, np.abs(_invert(m, s * x, mask)[0] - s * base).max() / np.abs(s * base).max())
            b0 = _invert(m, x, mask, fixed_scale=True)[0]
            ctrl = max(ctrl, np.abs(_invert(m, s * x, mask, fixed_scale=True)[0] - s * b0).max()
                       / np.abs(s * b0).max())
    tol = TH["equivariance_tol"]
    c61 = {"id": "C6.1", "name": "scale_equivariance", "passed": bool(viol <= tol < ctrl),
           "detail": f"max rel violation {viol:.2e} (tol {tol:.0e}); negative control (fixed start scale) {ctrl:.2e}"}
    junk = x + (1 - mask)[..., None] * rng.normal(0, 50.0, x.shape)
    v2 = max(np.abs(_invert(m, junk, mask)[0] - _invert(m, x, mask)[0]).max() for m in (rand, trained))
    c2 = max(np.abs(_invert(m, junk, mask, leaky=True)[0] - _invert(m, x, mask, leaky=True)[0]).max()
             for m in (rand, trained))
    c62 = {"id": "C6.2", "name": "broken_slot_invariance", "passed": bool(v2 <= 1e-12 < c2),
           "detail": f"max change from junk in broken slots {v2:.2e}; leaky control {c2:.2e}"}
    v3, c3 = 0.0, 0.0
    for attach in ("bone", "module"):
        m = build_model(*_NATIVE, rng, **dict(cfg, attach=attach))
        _init_from_data(m, xr, rng)
        m["params"]["T"] = rng.normal(0, 1, M)
        m["params"]["O"] = rng.normal(0, 0.2, M)
        z, g = rng.normal(0, 1, (16, m["cfg"]["q"])), rng.normal(0, 0.3, 16)
        t0 = _fraction(m, z, g)
        for _ in range(5):
            m["params"]["bL"][CH_BONE] += rng.uniform(-0.5, 0.5, M)
            dv = np.abs(_fraction(m, z, g) - t0).max()
            v3, c3 = (max(v3, dv), c3) if attach == "bone" else (v3, max(c3, dv))
    c63 = {"id": "C6.3", "name": "segment_fraction_invariance", "passed": bool(v3 <= 1e-9 < c3),
           "detail": f"hidden-point fraction change under segment rescaling {v3:.2e}; module-ruler control {c3:.2e}"}
    out = predict(trained, as_input(x, mask)).reshape(x.shape)
    c64 = {"id": "C6.4", "name": "survivor_pasting [definition check]",
           "passed": bool(np.array_equal(out[mask > 0], x[mask > 0])), "detail": "survivors returned unchanged"}
    return [c61, c62, c63, c64]


def test_c7(S):
    def keys(x):
        return {hashlib.sha1(np.round(f, 9).tobytes()).hexdigest() for f in x}
    tr = keys(S["train"][0])
    overlap = {k: len(tr & keys(S[k][0])) for k in S if k != "train"}
    return {"passed": not any(overlap.values()), "detail": f"shared wholes train vs eval splits {overlap}"}


# ---------------------------------------------------------------- tests: hypotheses
HYP = {"H-SIG": ("shift", "canon", "goldsmith"), "H-NEC": ("shift", "knock_measure_canon", "knock_pose_head"),
       "H-BLIND": ("portrait", "canon", "goldsmith"), "H-RIVAL": ("unseen_class", "canon", "rival"),
       "H-SELF": ("shift", "canon", "self_taught")}
SPLITS = ("val", "test_id", "shift", "portrait", "unseen_class")


def run_seed(seed, cfg, steps):
    """One replicate: fresh bodies and damage; every repairer sees the same intact wholes."""
    k = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(6)]
    S = make_splits(k[0])
    xtr = S["train"][0]
    canon = build_model(*_NATIVE, k[1], **cfg)
    out = {"seed": seed, "canon": canon, "hist": fit(canon, {"x": xtr}, steps, k[1]), "splits": S,
           "hands": {}, "res": None}
    if not math.isfinite(out["hist"][-1]):
        return out
    rival = build_model(*_NATIVE, k[2], **dict(cfg, attach="module"))
    fit(rival, {"x": xtr}, steps, k[2])
    for name, hid, comp, j in (("goldsmith", HAND_HIDDEN, False, 3), ("self_taught", HAND_HIDDEN, True, 4),
                               ("wide_hand", 64, False, 5)):
        out["hands"][name] = build_hand(k[j], hid)
        fit_hand(out["hands"][name], xtr, PRACTISED, comp, steps, k[j])
    out["res"] = {}
    for split in SPLITS:
        x, u, mask = S[split]
        pred = {"canon": canon_repair(canon, x, mask), "rival": canon_repair(rival, x, mask),
                "trivial": trivial_repair(xtr, x, mask)}
        pred.update({n: hand_repair(P, x, mask) for n, P in out["hands"].items()})
        if split == "shift":
            pred.update({"knock_" + n: canon_repair(knockout(canon, n), x, mask) for n in KNOCKS})
        out["res"][split] = {n: score(p, x, u, mask) for n, p in pred.items()}
    return out


def evaluate_hypotheses(runs, rng):
    hyps, knocks, enough = [], [], len(runs) >= 5
    for h in MIND_CARD["hypotheses"]:
        split, a, b = HYP[h["id"]]
        d = [r["res"][split][a]["exam_pass"] - r["res"][split][b]["exam_pass"] for r in runs]
        mean, ci = paired_bootstrap(d, rng) if enough else (float(np.mean(d)), [float("nan")] * 2)
        hyps.append({"id": h["id"], "metric": h["metric"], "split": split, "comparison": h["comparison"],
                     "mean_diff": mean, "ci95": ci, "mesi": h["mesi"], "n_seeds": len(runs),
                     "verdict": verdict(mean, ci, h["direction"], h["mesi"]) if enough else "not evaluated"})
    for n in KNOCKS:
        d = [r["res"]["shift"]["knock_" + n]["exam_pass"] - r["res"]["shift"]["canon"]["exam_pass"] for r in runs]
        mean, ci = paired_bootstrap(d, rng) if len(d) > 1 else (d[0], [d[0], d[0]])
        knocks.append({"module": n, "signature": modules(runs[0]["canon"])[n]["signature"],
                       "metric_change": mean, "ci95": ci})
    return hyps, knocks


def data_bridge(path, canon, rng):
    """Optional real-data bridge: intact figures from a local CSV, repaired under both damage sets."""
    if not path:
        return "skipped: no --data PATH given"
    arr = np.loadtxt(path, delimiter=",", ndmin=2)
    if arr.shape[1] != D:
        raise ValueError(f"--data needs {D} columns (x, y of {NPTS} points in this file's order)")
    x = arr.reshape(-1, NPTS, 2)
    x = x - x[:, :1]
    out = {}
    for name, vocab in (("practised", PRACTISED), ("unpractised", UNPRACTISED)):
        mask = damage(rng, len(x), vocab)
        out[name] = score(canon_repair(canon, x, mask), x, _rms(x), mask)["repair_err"]
    return f"{len(x)} figures; repair error in figure-RMS units {out}"


# ---------------------------------------------------------------- report
def build_report(ctx):
    runs, c1 = ctx["runs"], ctx["c1"]
    main = runs[0]
    return {
        "schema_version": "1.0", "chapter": 345, "file": os.path.basename(__file__),
        "card_revision": MIND_CARD["card_revision"],
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
        "seeds": ctx["seeds"], "runtime_s": round(ctx["runtime"], 2),
        "n_params": n_params(main["canon"]),
        "param_detail": {"canon_shared": n_params(main["canon"], True),
                         "canon_codes": n_params(main["canon"]) - n_params(main["canon"], True),
                         "goldsmith_hand": n_params({"params": build_hand(np.random.default_rng(0), HAND_HIDDEN)}),
                         "wide_hand": n_params({"params": build_hand(np.random.default_rng(0), 64)})},
        "gradcheck": {"tensors_checked": sum(r["tensors"] for r in c1), "tensors_total": sum(r["tensors"] for r in c1),
                      "max_rel_error": max(r["max_rel"] for r in c1), "checked_at": ["init", "after_training_steps"],
                      "passed": ctx["corr"][0]["passed"], "rows": c1},
        "correctness": ctx["corr"],
        "mutants": ctx["mutants"],
        "hypotheses": ctx["hyps"], "knockouts": ctx["knocks"],
        "split_metrics": ({s: {m: {k: float(np.mean([r["res"][s][m][k] for r in runs])) for k in
                                   ("exam_pass", "repair_err")} for m in main["res"][s]} for s in SPLITS}
                          if main["res"] else {}),
        "updates": {"cap": ctx["steps"], "canon": [len(r["hist"]) for r in runs]},
        "data_bridge": ctx["bridge"], "task_types": TASK_TYPES, "exit_code": ctx["exit_code"],
    }


def print_report(rep):
    print("=== VERIFIED REPORT · chapter 0345 ===")
    print(f"file {rep['file']} · card revision {rep['card_revision']} · python {rep['environment']['python']}"
          f" · numpy {rep['environment']['numpy']}")
    print(f"seeds {rep['seeds']} · runtime {rep['runtime_s']:.1f} s · task types {rep['task_types']} · "
          f"update cap {rep['updates']['cap']} (hands use all; canon/rival stop at plateau: {rep['updates']['canon']})")
    pd = rep["param_detail"]
    print(f"parameters: canon {rep['n_params']} ({pd['canon_shared']} shared weights + {pd['canon_codes']} "
          f"per-figure codes) · goldsmith/self-taught hand {pd['goldsmith_hand']} · wide hand {pd['wide_hand']}")
    g = rep["gradcheck"]
    print(f"gradient check: {g['tensors_checked']} tensor checks, max rel error {g['max_rel_error']:.2e}, "
          f"init and after training · {'PASS' if g['passed'] else 'FAIL'}")
    for c in rep["correctness"]:
        print(f"  {c['id']:<5}{c['name']:<37}{'PASS' if c['passed'] else 'FAIL'}  {c['detail']}")
    mu = rep["mutants"]
    print(f"mutation score {mu['detected']}/{mu['total']} = {mu['score']:.2f}")
    print("hypotheses (exam_pass, paired per-seed differences, bootstrap 2000, 95% CI):")
    for h in rep["hypotheses"]:
        print(f"  {h['id']:<8}{h['split']:<13}{h['comparison']:<46} mean {h['mean_diff']:+.3f}  "
              f"CI [{h['ci95'][0]:+.3f}, {h['ci95'][1]:+.3f}]  mesi {h['mesi']:.2f}  {h['verdict']}")
    print("knockouts (shift split, exam_pass of knocked canon minus full canon):")
    for k in rep["knockouts"]:
        print(f"  {k['module']:<20}signature={str(k['signature']):<6} change {k['metric_change']:+.3f}  "
              f"CI [{k['ci95'][0]:+.3f}, {k['ci95'][1]:+.3f}]")
    if rep["split_metrics"]:
        methods = list(rep["split_metrics"]["test_id"])
        print("mean exam_pass / repair error (face units) over seeds:")
        print("  " + "method".ljust(26) + "".join(s.rjust(16) for s in SPLITS))
        for m in methods:
            cells = [rep["split_metrics"][s].get(m) for s in SPLITS]
            print("  " + m.ljust(26) + "".join((f"{c['exam_pass']:.3f}/{c['repair_err']:.3f}" if c else "-").rjust(16)
                                              for c in cells))
    print(f"real-data bridge: {rep['data_bridge']}")
    print(f"exit code {rep['exit_code']}")
    print("=== END REPORT ===")


# ---------------------------------------------------------------- command line
def protocol(a, seeds, steps, cfg, rng):
    c1 = test_c1(rng, cfg)
    corr = [{"id": "C1", "name": "gradient_check", "passed": all(r["max_rel"] <= TH["gradcheck_rel_tol"] for r in c1),
             "detail": f"worst {max(r['max_rel'] for r in c1):.2e} over {len(c1)} checks (tol {TH['gradcheck_rel_tol']})"}]
    runs = [run_seed(seeds[0], cfg, steps)]
    main = runs[0] if runs[0]["res"] else None
    ctx = {"runs": runs, "c1": c1, "corr": corr, "hyps": [], "knocks": [], "bridge": "not run",
           "mutants": {"detected": 0, "total": len(MUTANTS), "score": 0.0}}
    if a.mutant:
        corr.append(dict(id="C3", name="learning", **test_c3(main)))
        return ctx, main is None
    runs += [run_seed(s, cfg, steps) for s in seeds[1:]]
    if any(r["res"] is None for r in runs):
        return ctx, True
    c2 = test_c2(cfg, a.seed, main)
    corr.append({"id": "C2", "name": "determinism_and_finiteness", "passed": c2["passed"], "detail": c2["detail"]})
    corr.append(dict(id="C3", name="learning", **test_c3(main)))
    corr.append(dict(id="C4", name="shuffled_label_control", **test_c4(cfg, a.seed, steps)))
    c5 = test_c5(cfg, a.seed)
    corr.append({"id": "C5", "name": "mutant_detection", "passed": c5["passed"], "detail": c5["detail"]})
    ctx["mutants"] = {"detected": sum(c5["detected"].values()), "total": len(MUTANTS), "score": c5["score"]}
    corr += test_c6(cfg, rng, main["canon"])
    corr.append(dict(id="C7", name="split_integrity", **test_c7(main["splits"])))
    ctx["hyps"], ctx["knocks"] = evaluate_hypotheses(runs, rng)
    ctx["bridge"] = data_bridge(a.data, main["canon"], rng)
    return ctx, not c2["finite"]


def main():
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__),
                                 description="Chapter 0345 · canon inversion with two rulers (full protocol by default)")
    ap.add_argument("--quick", action="store_true", help="one seed, reduced steps, all correctness tests")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--json", metavar="PATH")
    ap.add_argument("--card", action="store_true")
    ap.add_argument("--mutant", choices=sorted(MUTANTS))
    ap.add_argument("--data", metavar="PATH", help=f"local CSV, one intact figure per row, {D} columns")
    a = ap.parse_args()
    if a.card:
        print(json.dumps(MIND_CARD, indent=2))
        return 0
    if a.seeds < 1 or (a.data and not os.path.isfile(a.data)):
        ap.error("--seeds must be >= 1 and --data must name an existing file")
    t0, budget = time.time(), (20.0 if a.quick else 180.0)
    steps, seeds = (600, [a.seed]) if a.quick else (3000, [a.seed + i for i in range(a.seeds)])
    _ACTIVE["mutant"] = a.mutant
    print(f"{os.path.basename(__file__)} · seeds {seeds} · steps {steps}" + (f" · MUTANT {a.mutant}" if a.mutant else ""))
    quiet = {"over": "ignore", "invalid": "ignore", "divide": "ignore"} if a.mutant else {}
    with np.errstate(**quiet):
        try:
            ctx, nonfinite = protocol(a, seeds, steps, {"inv_steps": 60 if a.quick else 100},
                                      np.random.default_rng(a.seed))
        except ValueError as e:
            print(f"usage error: {e}", file=sys.stderr)
            return 2
    ctx["runtime"], ctx["seeds"], ctx["steps"] = time.time() - t0, seeds, steps
    ctx["corr"].append({"id": "C8", "name": "runtime_budget", "passed": ctx["runtime"] <= budget,
                        "detail": f"{ctx['runtime']:.1f} s of {budget:.0f} s"})
    failed = [c for c in ctx["corr"] if not c["passed"]]
    ctx["exit_code"] = 4 if nonfinite and not a.mutant else (1 if any(c["id"] != "C8" for c in failed)
                                                             else (3 if failed else 0))
    if ctx["runs"][0]["res"] is None and not ctx["hyps"]:
        ctx["hyps"] = [{"id": h["id"], "metric": h["metric"], "split": HYP[h["id"]][0], "comparison": h["comparison"],
                        "mean_diff": float("nan"), "ci95": [float("nan")] * 2, "mesi": h["mesi"], "n_seeds": 0,
                        "verdict": "not evaluated"} for h in MIND_CARD["hypotheses"]]
    rep = build_report(ctx)
    print_report(rep)
    if a.json:
        write_report(a.json, rep)
    return ctx["exit_code"]


if __name__ == "__main__":
    sys.exit(main())

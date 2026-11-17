#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0360 · John the Blind (1296-1346)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 18, Minds 341-360 Available on Amazon https://www.amazon.com/dp/B0HQ47T2NC
# Resume and Interactive Demos at https://artificiology.com/
# END ATTRIBUTION 
"""THE KNOTTED REINS -- a conroi of agents coupled by learnable one-sided ties.

Thesis
    A will that has lost its sense can stay an agent by binding its body to bodies
    that still sense: guidance then arrives as force through a reciprocal tie, not
    as a message, and the same tie that lends the eyes binds the fates.

Evidence and provenance (extrapolated: no philosophy of mind of his survives)
    D1 Froissart (Berners tr.): at Crecy, 26 Aug 1346, his knights tied the reins of
       their bridles together "so that they should not lose him in the press" and
       were found dead around him next day, horses still tied.            [scholarship]
    D2 Benes Krabice of Weitmile, Cronica ecclesiae Pragensis (FRB IV): "illuc me
       ducite, ubi maior strepitus certaminis vigeret" -- lead me where the din of
       battle is loudest; the goal is heard, not seen.                     [scholarship]
    D3 Charles IV, Vita Caroli: having lost one eye, he went secretly to the
       Montpellier physicians and went fully blind (c. 1339-40); he kept ruling,
       negotiating and campaigning for six more years.                         [deeds]
    D4 Froissart: the knight who scouted the English for Philip VI "pertained to the
       king of Bohemia"; the front halted on his advice, the rear would not.  [scholarship]
    D5 Froissart: that evening John of Hainault took Philip VI's horse by the bridle
       and led him off the field -- the same instrument, opposite authority.  [scholarship]
    D6 Inaugural diplomas 1310/11: nobles not compelled to fight abroad; he fought
       abroad all his life with a personal retinue bound to him, not to Bohemia. [deeds]
    D7 The inference that he treated ties (knots, homage, marriages, pledges) as his
       instrument of agency is the chapter's reading, labelled as such.  [speculation]

Doctrine -> mechanism -> test
    D1,D3 -> M1 reins_knot: learnable one-sided, reciprocal spring-damper ties
             between rider bodies; force on i is +T n, on j is -T n     -> C6.1, C6.2, H-SIG
    D2    -> M2 heard goal: the blind leader alone holds the command and
             hears two melees; companions see the ground              -> C3, C4
    D1    -> M3 guidance as force: companions steer the knot, the
             leader is carried                                         -> H-NEC
    D1,D5 -> blind spot: the knot cannot release, so companions share
             the leader's fate when the heard goal is the killing ground -> H-BLIND, H-RIVAL

Research question (embodied control and sensorimotor adaptation; multi-agent)
    When one agent loses a sensory channel, can a learned reciprocal physical
    coupling to agents that keep it replace learned communication as the fusion
    channel, remain intact when the communication channel is masked, and what
    does the reciprocity cost when the shared goal is itself harmful?

Closest prior art and the delta
    Learned multi-agent communication (CommNet, Sukhbaatar et al. 2016) is the
    size-matched baseline ("herald"). Communication-free force coordination
    (Wang & Schwager, IJRR 2016), haptic goal exchange between coupled humans
    (Takagi et al. 2017, 2018) and leash-guided robot guides (Xiao et al. 2021)
    are the physical side. Delta: the tie itself (stiffness, rest length,
    damping per pair) is trained end-to-end with heterogeneous-modality policies
    through a differentiable body, as the fusion channel of a team whose leader
    is blind; one-sidedness (a rein cannot push) and reciprocity are enforced
    by the physics and tested as invariants.

Blind spot
    The knot transmits commitment as faithfully as guidance. At Crecy every tied
    companion died; his untied son left the field. When the heard goal lies in a
    harmful zone the knot should raise companion harm relative to untied agents
    (herald) and to a companion-releasable bridle (the rival, after D5).

Task (Crecy field, episodic control, differentiable)
    4 point-mass riders (leader + 3 companions), horizon T. Two melees A, B; the
    leader's one-hot command picks the goal G. Gaussian pits harm any rider.
    Leader vision: sighted / one-eyed (noisy) / blind. Companions see pits and
    arrows, hear A and B, see the leader, never receive the command. Messages
    (herald only) are masked by din: noise std = mu * din(P). Optional arrow zone
    around G. Loss = strike distance (last steps) + pit harm + arrow harm + effort.
    Splits: held-out (same law), shifted (always blind, heavy masking, fog, more
    pits, farther goals), blind-spot (always blind, arrow zone at G).

Limits
    A 2-D point-mass abstraction with synthetic episodes; forces are not horses
    and pits are not the Crecy ground. The file is a research prototype of one
    AGI-oriented mechanism, not an AGI, and says nothing about the man's inner
    life beyond the labelled inference in D7.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 3,
    "revision_log": [
        "rev 2 (2026-09-30, after one smoke run on seed 0, before the verified protocol): the unconditioned knot "
        "learned slack leader ties (rest length about 2.4) because 60% of training episodes have a sighted or "
        "one-eyed leader. D1+D3 (the knot was tied for the blind charge, not for his sighted life) imply the knot "
        "should be conditioned on the leader's loss of sight, so a learned sight-conditioned offset (18 parameters) "
        "was added to reins, bridle, agnostic and a descriptive hand-tied 'crecy' arm was declared. Gradient-check "
        "tolerance set to 1e-4 with the written reason in thresholds. No hypothesis, metric, split, mesi or "
        "learning threshold changed.",
        "rev 3 (2026-09-30, after the first full protocol run): documentation only. JSON writer fixed (numpy bool), "
        "similarity to the only locally available corpus file recorded (0.0245). The run is deterministic; no "
        "hypothesis, metric, split, threshold or model changed, and the verified report was regenerated."],
    "generation": {"template_version": "codeguidelines 1.0 (15 Sep 2026)", "generator": "Claude",
                   "generator_version": "Opus 5.5", "date": "2026-09-30"},
    "id": 360, "figure": "John the Blind", "born": 1296, "died": 1346,
    "civilization": "Luxembourgish (Bohemian crown)",
    "provenance": "extrapolated",
    "thesis": ("A will that has lost its sense stays an agent by binding its body to bodies that sense: "
               "guidance arrives as force through a reciprocal one-sided tie, and the tie that lends the "
               "eyes binds the fates."),
    "evidence": [
        {"id": "D1", "claim": "Knights tied their reins together so as not to lose him in the press; found dead "
         "around him, horses tied", "basis": "scholarship", "source": "Froissart, Chroniques I (Berners tr.), Crecy"},
        {"id": "D2", "claim": "Lead me where the din of battle is loudest", "basis": "scholarship",
         "source": "Benes Krabice of Weitmile, Cronica ecclesiae Pragensis, FRB IV"},
        {"id": "D3", "claim": "Lost one eye, consulted Montpellier physicians, went fully blind; ruled six more "
         "years", "basis": "deeds", "source": "Charles IV, Vita Caroli (ed. Nagy and Schaer 2001)"},
        {"id": "D4", "claim": "His knight scouted for Philip VI; front halted, rear did not", "basis": "scholarship",
         "source": "Froissart, Chroniques I, Crecy"},
        {"id": "D5", "claim": "John of Hainault led Philip VI away by the bridle", "basis": "scholarship",
         "source": "Froissart, Chroniques I, Crecy"},
        {"id": "D6", "claim": "Inaugural diplomas: Bohemian nobles not compelled to fight abroad", "basis": "deeds",
         "source": "Inaugural diplomas 1310/1311 (Chaloupecky 1949)"},
        {"id": "D7", "claim": "Ties as his instrument of agency", "basis": "speculation", "source": "chapter reading"},
    ],
    "research_question": {"category": "embodied control and sensorimotor adaptation",
                          "question": ("Can a learned reciprocal physical coupling replace learned communication "
                                       "as the fusion channel of a team whose leader lost a sense, and what does "
                                       "reciprocity cost when the shared goal is harmful?")},
    "mechanism": {
        "name": "Knotted reins (learnable one-sided reciprocal ties in a differentiable body)",
        "family": "closed-loop sensorimotor control; morphology-control co-design",
        "signature_modules": ["reins_knot"],
        "closest_prior_art": ["CommNet learned communication (Sukhbaatar et al. 2016)",
                              "Force-ANTS communication-free transport (Wang and Schwager 2016)",
                              "haptic goal exchange (Takagi et al. 2017, 2018)",
                              "leash-guided robotic guide dog (Xiao et al. 2021)"],
        "overlap": "Medium",
        "prior_art_queries": ["learned coupling stiffness multi-agent differentiable physics",
                              "communication-free cooperative manipulation force", "haptic communication partner goal",
                              "robotic guide dog leash tension", "emergent communication noisy channel"],
        "contribution_type": "mechanism",
        "delta": ("Tie parameters trained end to end as the fusion channel of a heterogeneous-modality team with a "
                  "blind leader; reciprocity and one-sidedness enforced by physics and tested; fate-sharing cost "
                  "measured against a releasable bridle."),
    },
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1", "property_test": "C6.1", "hypothesis": "H-SIG"},
        {"doctrine": "D1", "mechanism": "M3", "property_test": "C6.2", "hypothesis": "H-NEC"},
        {"doctrine": "D2", "mechanism": "M2", "property_test": "C4", "hypothesis": "H-SIG"},
        {"doctrine": "D5", "mechanism": "M1", "property_test": "C6.1", "hypothesis": "H-RIVAL"},
        {"doctrine": "D1", "mechanism": "M1", "property_test": "C6.3", "hypothesis": "H-BLIND"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "Knotted reins give lower task loss than size-matched learned messaging when "
         "the leader is blind and the channel is masked", "metric": "task_loss", "split": "shifted",
         "comparison": "model - baseline", "direction": "less", "mesi": 0.05, "seeds": 5},
        {"id": "H-NEC", "statement": "Severing the pull of the ties (keeping their reading) hurts more than "
         "blinding the reading (keeping the pull)", "metric": "task_loss", "split": "shifted",
         "comparison": "signature_knockout - matched_knockout", "direction": "greater", "mesi": 0.05, "seeds": 5},
        {"id": "H-BLIND", "statement": "When the heard goal is inside an arrow zone the knot raises companion harm "
         "relative to untied agents", "condition": "blind-spot split", "grounding": "every tied companion died at "
         "Crecy; the untied son left", "metric": "companion_harm", "comparison": "model - baseline",
         "direction": "greater", "mesi": 0.02, "seeds": 5},
        {"id": "H-RIVAL", "statement": "A companion-releasable bridle (John of Hainault leading Philip VI away) "
         "lowers companion harm relative to the knot in the blind-spot split", "metric": "companion_harm",
         "split": "blindspot", "comparison": "model - rival", "direction": "greater", "mesi": 0.02, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.30, "margin_over_trivial": 0.20, "shuffle_band": 0.10,
                   "gradcheck_rel_tol": 1e-4, "gradcheck_denominator_floor": 1e-5,
                   "gradcheck_tol_reason": ("BPTT through six steps of softplus-gated ties accumulates about 1e-15 "
                                            "relative rounding in the loss, i.e. about 1e-9 absolute noise in central "
                                            "differences at eps=1e-6; this dominates entries whose gradient is "
                                            "below 1e-5, where analytic and numeric values agree to 4-5 digits.")},
    "probe_predictions": [{"probe": "P11", "expected": "above baseline"},
                          {"probe": "P9", "expected": "above baseline"},
                          {"probe": "P8", "expected": "equal to baseline"},
                          {"probe": "P3", "expected": "below baseline"}],
    "dialectic_links": [{"chapter": 355, "relation": "rival", "test": "none (historical context: Bayernburg, 1337)"}],
    "corpus_neighbors": [
        {"chapter": 504, "similarity": 0.0245, "difference": "Philip II governs by paper with no body in the field; "
         "here agency is a body carried by other bodies (similarity measured against the Sample.zip file, the only "
         "corpus code available locally; other neighbours' files were not available, similarity null)"},
        {"chapter": 349, "similarity": None, "difference": "Guzman prices a refusal by what coercers learn; here "
         "nothing is priced, the tie transmits commitment physically"},
        {"chapter": 355, "similarity": None, "difference": "Gediminas opens a bypass road to test a power; here the "
         "road is found by companions and felt through a rein"},
        {"chapter": 176, "similarity": None, "difference": "Tarafa's slackened tether is a time horizon; here the "
         "tether is a physical sensor with one-sided mechanics"},
        {"chapter": 168, "similarity": None, "difference": "Imru' al-Qays binds two motions as likeness; here "
         "coupling carries control and harm, not resemblance"},
        {"chapter": 319, "similarity": None, "difference": "Walther values a bond by its shared load; here the bond "
         "is trained and its cost is measured in companion harm"},
    ],
    "barometer": {
        "cognitive_processing": ["C3 strike distance vs goal-agnostic trivial"],
        "embodied_cognition": ["P11-style closed-loop control; leader vision dropout; tie pull knockout"],
        "world_modeling": ["pit and arrow fields sensed by companions only"],
        "consciousness": [],
        "language_understanding": ["herald baseline message channel under masking"],
        "emotional_intelligence": ["companion harm as a welfare term (H-BLIND, H-RIVAL)"],
        "creativity": [],
        "autonomy": ["learned tie topology (which pairs become stiff)"],
    },
    "task_types": ["episodic_control"],
    "applications": [
        {"use": "leash or harness guidance of blind and low-vision pedestrians by a robot guide in crowds",
         "sector": "assistive robotics", "dataset": "ETH/UCY pedestrian trajectories (Pellegrini 2009; Lerner 2007)"},
        {"use": "sensor-failure-tolerant convoys coupled by physical or virtual springs",
         "sector": "transport and field robotics", "dataset": "nuScenes (Caesar et al. 2020), camera dropout studies"},
    ],
    "safety_notes": ("Military figure (section 15): the task is guidance and cohesion of a group with a sensory-"
                     "impaired member; goals are abstract points, harm terms protect riders, no targeting or "
                     "weapons are modelled. Blindness is modelled as an input condition, never as a deficit of the "
                     "person. No sentence here is presented as his own words beyond the cited chronicles."),
}

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

CHAPTER = 360
N_RIDERS, N_COMP = 4, 3
PAIRS_I = np.array([0, 0, 0, 1, 1, 2])
PAIRS_J = np.array([1, 2, 3, 2, 3, 3])
INCID = np.zeros((N_RIDERS, 6))
INCID[PAIRS_I, np.arange(6)] = 1.0
INCID[PAIRS_J, np.arange(6)] = -1.0
AVG = (np.ones((N_RIDERS, N_RIDERS)) - np.eye(N_RIDERS)) / 3.0
DT, DAMP, FMAX, BETA, EPS_R = 0.2, 0.9, 1.6, 6.0, 1e-2
T_STEPS, STRIKE_STEPS = 24, 6
PIT_W, DIN_SD, REIN_SCALE, EYE_SCALE = 0.45, 1.5, 0.5, 0.5
W_PIT, W_ARROW_K, W_ARROW_C, W_EFFORT = 2.0, 0.5, 1.5, 0.01
IN_K, IN_C, H_REINS, H_HERALD, DM = 14, 17, 18, 14, 3
MUTANT = {"name": None}

# BEGIN STANDARD UTILITIES v1.0

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

def logsumexp(z, axis=-1):
    m = z.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(z - m).sum(axis=axis, keepdims=True))).squeeze(axis)

def softplus(x):
    return np.logaddexp(0.0, x)

class Adam:
    def __init__(self, params, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
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

def clip_global(grads, max_norm):
    norm = math.sqrt(sum(float((g * g).sum()) for g in grads.values()))
    if norm > max_norm:
        for k in grads:
            grads[k] = grads[k] * (max_norm / (norm + 1e-12))
    return norm

def finite_diff_check(f, params, grads, rng, eps=1e-6, n_entries=20, floor=1e-5):
    """Central differences on up to n_entries random entries plus the largest-gradient entry of every tensor.
    Relative error |a-n|/max(|a|+|n|, floor): below the floor the error is judged against the floor."""
    worst, report = 0.0, {}
    for k, p in params.items():
        flat = p.reshape(-1)
        idx = set(rng.choice(flat.size, size=min(n_entries, flat.size), replace=False).tolist())
        idx.add(int(np.argmax(np.abs(grads[k].reshape(-1)))))
        err = 0.0
        for i in idx:
            old = flat[i]
            flat[i] = old + eps
            fp = f()
            flat[i] = old - eps
            fm = f()
            flat[i] = old
            num, ana = (fp - fm) / (2 * eps), grads[k].reshape(-1)[i]
            err = max(err, abs(num - ana) / max(floor, abs(num) + abs(ana)))
        report[k] = err
        worst = max(worst, err)
    return worst, report

def paired_bootstrap(diffs, rng, n=2000):
    d = np.asarray(diffs, dtype=np.float64)
    means = d[rng.integers(0, d.size, size=(n, d.size))].mean(axis=1)
    return float(d.mean()), [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]

def verdict(mean, ci, direction, mesi):
    if direction == "less":
        mean, ci = -mean, [-ci[1], -ci[0]]
    if ci[0] > 0 and mean >= mesi:
        return "supported"
    if ci[1] < 0:
        return "contradicted"
    return "inconclusive"

def write_report(path, report):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)


# END STANDARD UTILITIES

# ----------------------------------------------------------------- reverse-mode tape
class _Node:
    __slots__ = ("v", "g", "par", "back", "aux")

    def __init__(self, v, par=(), back=None):
        self.v, self.g, self.par, self.back, self.aux = v, None, par, back, None

class Tape:
    """Minimal reverse-mode tape; ops are coarse (fused physics) so BPTT stays cheap in NumPy."""

    def __init__(self):
        self.nodes = []

    def leaf(self, v):
        n = _Node(np.asarray(v, dtype=np.float64))
        self.nodes.append(n)
        return n

    @staticmethod
    def const(v):
        return _Node(np.asarray(v, dtype=np.float64))

    def op(self, v, par, back):
        n = _Node(v, par, back)
        self.nodes.append(n)
        return n

    def backward(self, out):
        out.g = np.ones_like(out.v)
        for n in reversed(self.nodes):
            if n.g is None or n.back is None:
                continue
            for p, gp in zip(n.par, n.back(n.g)):
                if gp is not None:
                    p.g = gp if p.g is None else p.g + gp

def _unb(g, shape):
    while g.ndim > len(shape):
        g = g.sum(0)
    for i, d in enumerate(shape):
        if d == 1 and g.shape[i] != 1:
            g = g.sum(i, keepdims=True)
    return g

def add(t, a, b):
    return t.op(a.v + b.v, (a, b), lambda g: (_unb(g, a.v.shape), _unb(g, b.v.shape)))

def mul(t, a, b):
    return t.op(a.v * b.v, (a, b), lambda g: (_unb(g * b.v, a.v.shape), _unb(g * a.v, b.v.shape)))

def scale(t, a, c):
    return t.op(a.v * c, (a,), lambda g: (g * c,))

def matmul(t, a, w):
    return t.op(a.v @ w.v, (a, w), lambda g: (g @ w.v.T, a.v.T @ g))

def act_tanh(t, a):
    y = np.tanh(a.v)
    return t.op(y, (a,), lambda g: (g * (1 - y * y),))

def act_sigmoid(t, a):
    y = 0.5 * (1.0 + np.tanh(0.5 * a.v))
    return t.op(y, (a,), lambda g: (g * y * (1 - y),))

def act_softplus(t, a):
    s = 0.5 * (1.0 + np.tanh(0.5 * a.v))
    return t.op(softplus(a.v), (a,), lambda g: (g * s,))

def total(t, a, c=1.0):
    return t.op(np.array(a.v.sum() * c), (a,), lambda g: (np.full_like(a.v, float(g) * c),))

def concat(t, xs):
    cut = np.cumsum([x.v.shape[-1] for x in xs])[:-1]
    return t.op(np.concatenate([x.v for x in xs], -1), tuple(xs), lambda g: tuple(np.split(g, cut, -1)))

def reshape(t, a, shape):
    return t.op(a.v.reshape(shape), (a,), lambda g: (g.reshape(a.v.shape),))

def take(t, a, i):
    def back(g):
        full = np.zeros_like(a.v)
        full[:, i] = g
        return (full,)
    return t.op(a.v[:, i], (a,), back)

def join(t, uk, uc):
    return t.op(np.concatenate([uk.v[:, None], uc.v], 1), (uk, uc), lambda g: (g[:, 0], g[:, 1:]))

def sqrt_eps(t, a, eps):
    y = np.sqrt(a.v + eps)
    return t.op(y, (a,), lambda g: (g * 0.5 / y,))

# ----------------------------------------------------------------- the body: reins and fields
def rein_forces(t, P, V, k, L, c, asym=1.0):
    """One-sided reciprocal spring-damper per pair (a rein pulls, never pushes).
    Tension T = softplus(beta*(k*s + c*vr))/beta with stretch s = r - L and radial speed vr.
    Force on i is +T n, on j is -asym*T n (asym=1 is Newton's third law; <1 only for the negative control)."""
    D = P.v[:, PAIRS_J] - P.v[:, PAIRS_I]
    W = V.v[:, PAIRS_J] - V.v[:, PAIRS_I]
    r = np.sqrt((D * D).sum(-1) + EPS_R ** 2)
    n = D / r[..., None]
    s, vr = r - L.v, (W * n).sum(-1)
    z = BETA * (k.v * s + c.v * vr)
    T, sg = softplus(z) / BETA, 0.5 * (1.0 + np.tanh(0.5 * z))
    inc = INCID.copy()
    inc[PAIRS_J, np.arange(6)] *= asym
    F = np.einsum("np,bpd->bnd", inc, T[..., None] * n)

    def back(g):
        gFp = np.einsum("np,bnd->bpd", inc, g)
        gT = (gFp * n).sum(-1)
        gi = gT * sg
        gn = T[..., None] * gFp + (gi * c.v)[..., None] * W
        gD = (gn - (gn * n).sum(-1, keepdims=True) * n) / r[..., None] + ((gi * k.v) / r)[..., None] * D
        gP = -np.einsum("np,bpd->bnd", INCID, gD)
        gV = -np.einsum("np,bpd->bnd", INCID, (gi * c.v)[..., None] * n)
        return gP, gV, _unb(gi * s, k.v.shape), _unb(-gi * k.v, L.v.shape), _unb(gi * vr, c.v.shape)
    node = t.op(F, (P, V, k, L, c), back)
    node.aux = T
    return node

def gauss_field(t, P, Q, amp, w):
    """Sum of Gaussian bumps: value (B,N) and spatial gradient (B,N,2), each with an exact backward."""
    U = P.v[:, :, None, :] - Q[:, None, :, :]
    w2 = (w * w)[:, None, None]
    e = amp[:, None, :] * np.exp(-(U * U).sum(-1) / (2 * w2))
    grad = -(e[..., None] * U).sum(2) / w2
    val = t.op(e.sum(-1), (P,), lambda g: (-(e[..., None] * U * g[..., None, None]).sum(2) / w2,))

    def gback(g):
        ug = (U * g[:, :, None, :]).sum(-1)
        w4 = (w2 * w2)[..., None]
        return ((e[..., None] * (U * ug[..., None] / w4 - g[:, :, None, :] / w2[..., None])).sum(2),)
    return val, t.op(grad, (P,), gback)

def din_level(t, P, S):
    """Loudness of the two melees at each rider: sum_s 1/(1+|P-S|^2/sd^2)."""
    U = P.v[:, :, None, :] - S[:, None, :, :]
    q = 1.0 + (U * U).sum(-1) / DIN_SD ** 2
    return t.op((1.0 / q).sum(-1), (P,), lambda g: ((-(2.0 / DIN_SD ** 2) * (g[..., None] / q ** 2)[..., None]
                                                      * U).sum(2),))

def mix(t, M):
    return t.op(np.einsum("ij,bjd->bid", AVG, M.v), (M,), lambda g: (np.einsum("ij,bid->bjd", AVG, g),))

def sum_last(t, a):
    return t.op(a.v.sum(-1), (a,), lambda g: (np.repeat(g[..., None], a.v.shape[-1], -1),))

# ----------------------------------------------------------------- data and tasks
LAWS = {
    "native": dict(eye=(0.3, 0.3, 0.4), mu=(0.0, 0.5), fog=(0.0, 0.1), K=5, d=(3.5, 5.0), kill_p=0.15,
                   alpha=1.0, kw=0.7),
    "shifted": dict(eye=(0.0, 0.0, 1.0), mu=(0.9, 1.3), fog=(0.2, 0.35), K=7, d=(5.0, 6.0), kill_p=0.15,
                    alpha=1.0, kw=0.7),
    "blindspot": dict(eye=(0.0, 0.0, 1.0), mu=(0.0, 0.5), fog=(0.0, 0.1), K=5, d=(3.5, 5.0), kill_p=1.0,
                      alpha=1.6, kw=1.1),
}
START = np.array([[0.0, 0.0], [-0.5, 0.55], [-0.5, -0.55], [-0.9, 0.0]])

def _pits(rng, G, K):
    """Pits cluster on the road to the commanded melee (so the direct line is usually blocked) plus strays."""
    B = G.shape[0]
    dirg = G / np.linalg.norm(G, axis=1, keepdims=True)
    perp = np.stack([-dirg[:, 1], dirg[:, 0]], 1)
    along = rng.uniform(0.25, 0.8, (B, K))[..., None] * G[:, None] + rng.normal(0, 0.45, (B, K))[..., None] * perp[:, None]
    scat = np.stack([rng.uniform(0.5, 6.0, (B, K)), rng.uniform(-4.0, 4.0, (B, K))], -1)
    Q = np.where((rng.uniform(size=(B, K)) < 0.6)[..., None], along, scat)
    for ref in (np.zeros((B, 2)), G):
        dv = Q - ref[:, None]
        dist = np.linalg.norm(dv, axis=-1, keepdims=True)
        Q = np.where(dist < 1.0, ref[:, None] + dv / np.maximum(dist, 1e-6), Q)
    return Q

def make_episodes(rng, n, law_name, T=T_STEPS):
    law = LAWS[law_name]
    th_a = rng.uniform(-1.0, 1.0, n)
    th_b = np.clip(th_a + rng.uniform(0.7, 1.3, n) * rng.choice([-1.0, 1.0], n), -1.45, 1.45)
    da, db = rng.uniform(*law["d"], n), rng.uniform(*law["d"], n)
    S = np.stack([np.stack([da * np.cos(th_a), da * np.sin(th_a)], -1),
                  np.stack([db * np.cos(th_b), db * np.sin(th_b)], -1)], 1)
    cmd = rng.integers(0, 2, n)
    G = S[np.arange(n), cmd]
    eye = rng.choice(3, n, p=law["eye"])
    kill = (rng.uniform(size=n) < law["kill_p"]).astype(float) * law["alpha"]
    return {
        "law": law_name, "S": S, "cmd": cmd, "G": G, "Q": _pits(rng, G, law["K"]), "qamp": np.ones((n, law["K"])),
        "P0": START[None] + rng.normal(0, 0.05, (n, N_RIDERS, 2)), "kill": kill, "kw": np.full(n, law["kw"]),
        "eye_gain": np.where(eye < 2, 1.0, 0.0), "eye_noise": np.array([0.05, 0.6, 0.0])[eye],
        "sight_loss": np.array([0.0, 0.5, 1.0])[eye],
        "mu": rng.uniform(*law["mu"], n), "fog": rng.uniform(*law["fog"], n),
        "xi_eye": rng.normal(size=(n, T, 4)), "xi_fog": rng.normal(size=(n, T, N_COMP, 4)),
        "xi_msg": rng.normal(size=(n, T, N_RIDERS, DM)), "xi_rein": rng.normal(size=(n, T, N_RIDERS, 2)),
    }

def ep_slice(ep, idx):
    return {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in ep.items()}

def ep_hashes(ep):
    return {hashlib.sha1(np.round(np.concatenate([ep["S"][i].ravel(), ep["Q"][i].ravel(), ep["P0"][i].ravel()]), 6)
                         .tobytes()).hexdigest() for i in range(ep["S"].shape[0])}

def make_data(rng, sizes):
    kids = [np.random.default_rng(s) for s in rng.bit_generator.seed_seq.spawn(5)]
    return {"train": make_episodes(kids[0], sizes["pool"], "native"),
            "val": make_episodes(kids[1], sizes["val"], "native"),
            "test": make_episodes(kids[2], sizes["test"], "native"),
            "shifted": make_episodes(kids[3], sizes["test"], "shifted"),
            "blindspot": make_episodes(kids[4], sizes["test"], "blindspot")}

def env_reset(rng):
    """One native episode (batch of 1); the ties are the model's own morphology and are not part of the world."""
    return {"ep": make_episodes(rng, 1, "native"), "P": START[None].copy(), "V": np.zeros((1, N_RIDERS, 2)), "t": 0}

def env_step(state, action, rng):
    """Untied point-mass step for external probes: action (4,2) forces, clipped to FMAX."""
    a = np.clip(np.asarray(action, dtype=float).reshape(1, N_RIDERS, 2), -FMAX, FMAX)
    V = (1 - DAMP * DT) * state["V"] + DT * a
    return {**state, "P": state["P"] + DT * V, "V": V, "t": state["t"] + 1}

# ----------------------------------------------------------------- model
TASK_TYPES = ["episodic_control"]

def _isp(y):
    return float(np.log(np.expm1(y)))

def build_model(in_dim=IN_K, out_dim=2, task_type="episodic_control", rng=None, kind="reins", **cfg):
    """kind: reins (the mind), herald (learned-message baseline), bridle (releasable-tie rival),
    agnostic (reins with the command removed: the goal-agnostic trivial reference),
    crecy (descriptive arm: a hand-tied short knot, k=3, L=0.8, c=0.5, frozen; only the policies learn)."""
    if task_type not in TASK_TYPES or in_dim != IN_K or out_dim != 2:
        raise ValueError("this file supports episodic_control with in_dim=%d, out_dim=2" % IN_K)
    rng = rng if rng is not None else np.random.default_rng(0)
    H = cfg.get("hidden", H_HERALD if kind == "herald" else H_REINS)
    ex = DM if kind == "herald" else 0
    p = {}
    for pre, nin in (("k", IN_K + ex), ("c", IN_C + ex)):
        p[pre + "_W1"] = rng.normal(0, 1.0 / math.sqrt(nin), (nin, H))
        p[pre + "_b1"] = np.zeros(H)
        p[pre + "_W2"] = rng.normal(0, 0.5 / math.sqrt(H), (H, 2))
        p[pre + "_b2"] = np.zeros(2)
        if kind == "herald":
            p[pre + "_Wm"] = rng.normal(0, 1.0 / math.sqrt(H), (H, DM))
            p[pre + "_bm"] = np.zeros(DM)
    if kind != "herald":
        p["kraw"], p["Lraw"], p["craw"] = np.full(6, _isp(1.0)), np.full(6, _isp(0.9)), np.full(6, _isp(0.3))
        p["kraw_b"], p["Lraw_b"], p["craw_b"] = np.zeros(6), np.zeros(6), np.zeros(6)
    if kind == "crecy":
        p["kraw"][:], p["Lraw"][:], p["craw"][:] = _isp(3.0), _isp(0.8), _isp(0.5)
    if kind == "bridle":
        p["g_w"], p["g_b"] = rng.normal(0, 0.1, (H, 1)), np.full(1, 2.0)
    return {"kind": kind, "params": p, "H": H, "pull_on": kind != "herald", "sense_on": kind != "herald",
            "msg_on": kind == "herald", "gate_on": kind == "bridle", "cmd_on": kind != "agnostic"}

def _policy(t, x, lv, pre):
    h = act_tanh(t, add(t, matmul(t, x, lv[pre + "_W1"]), lv[pre + "_b1"]))
    u = scale(t, act_tanh(t, add(t, matmul(t, h, lv[pre + "_W2"]), lv[pre + "_b2"])), FMAX)
    return h, u

def _obs_king(t, m, ep, s, P, V, pit_g, arr_g, F, rmsg):
    """Leader observation, fused (it is linear in the state): command, hearing of both melees, degraded vision
    of pits and arrows, own velocity, tie reading, and (herald only) the received message."""
    B, gain, sense = P.v.shape[0], ep["eye_gain"][:, None] * EYE_SCALE, REIN_SCALE * float(m["sense_on"])
    parts = [np.eye(2)[ep["cmd"]] * float(m["cmd_on"]), 0.25 * (ep["S"][:, 0] - P.v[:, 0]),
             0.25 * (ep["S"][:, 1] - P.v[:, 0]),
             gain * np.concatenate([pit_g.v[:, 0], arr_g.v[:, 0]], -1) + ep["eye_noise"][:, None] * ep["xi_eye"][:, s],
             V.v[:, 0], sense * F.v[:, 0] + 0.03 * ep["xi_rein"][:, s, 0]]
    par = [P, V, pit_g, arr_g, F]
    if m["msg_on"]:
        parts.append(rmsg.v[:, 0])
        par.append(rmsg)

    def back(g):
        gP, gV = np.zeros_like(P.v), np.zeros_like(V.v)
        gpit, garr, gF = np.zeros_like(pit_g.v), np.zeros_like(arr_g.v), np.zeros_like(F.v)
        gP[:, 0] = -0.25 * (g[:, 2:4] + g[:, 4:6])
        gpit[:, 0], garr[:, 0] = gain * g[:, 6:8], gain * g[:, 8:10]
        gV[:, 0], gF[:, 0] = g[:, 10:12], sense * g[:, 12:14]
        out = [gP, gV, gpit, garr, gF]
        if m["msg_on"]:
            gm = np.zeros_like(rmsg.v)
            gm[:, 0] = g[:, 14:]
            out.append(gm)
        return tuple(out)
    return t.op(np.concatenate(parts, -1), tuple(par), back), B

def _obs_comp(t, m, ep, s, P, V, pit_g, arr_g, F, rmsg):
    """Companion observation, fused: hearing, pit and arrow vision under fog, leader offset, velocity,
    tie reading, slot identity and (herald only) the received message; reshaped to (B*3, features)."""
    B, sense = P.v.shape[0], REIN_SCALE * float(m["sense_on"])
    Pc = P.v[:, 1:]
    parts = [0.25 * (ep["S"][:, None, 0] - Pc), 0.25 * (ep["S"][:, None, 1] - Pc),
             EYE_SCALE * np.concatenate([pit_g.v[:, 1:], arr_g.v[:, 1:]], -1)
             + ep["fog"][:, None, None] * ep["xi_fog"][:, s],
             0.5 * (P.v[:, :1] - Pc), V.v[:, 1:], sense * F.v[:, 1:] + 0.03 * ep["xi_rein"][:, s, 1:],
             np.broadcast_to(np.eye(3), (B, 3, 3))]
    par = [P, V, pit_g, arr_g, F]
    if m["msg_on"]:
        parts.append(rmsg.v[:, 1:])
        par.append(rmsg)
    x = np.concatenate(parts, -1)

    def back(g):
        g = g.reshape(B, N_COMP, -1)
        gP, gV = np.zeros_like(P.v), np.zeros_like(V.v)
        gpit, garr, gF = np.zeros_like(pit_g.v), np.zeros_like(arr_g.v), np.zeros_like(F.v)
        gP[:, 1:] = -0.25 * (g[..., 0:2] + g[..., 2:4]) - 0.5 * g[..., 8:10]
        gP[:, 0] = 0.5 * g[..., 8:10].sum(1)
        gpit[:, 1:], garr[:, 1:] = EYE_SCALE * g[..., 4:6], EYE_SCALE * g[..., 6:8]
        gV[:, 1:], gF[:, 1:] = g[..., 10:12], sense * g[..., 12:14]
        out = [gP, gV, gpit, garr, gF]
        if m["msg_on"]:
            gm = np.zeros_like(rmsg.v)
            gm[:, 1:] = g[..., 17:]
            out.append(gm)
        return tuple(out)
    return t.op(x.reshape(B * N_COMP, -1), tuple(par), back)

def _knot(t, m, lv, ep_sight):
    """The knot is tied once per episode for the leader's sensory state: sight lost (0, 0.5, 1) selects
    stiffness, rest length and damping through a learned offset. Herald riders are untied."""
    if m["kind"] == "herald":
        z = t.const(np.zeros((1, 6)))
        return z, t.const(np.ones((1, 6))), z
    loss_of_sight = t.const(ep_sight[:, None])
    return [act_softplus(t, add(t, reshape(t, lv[nm], (1, 6)), mul(t, reshape(t, lv[nm + "_b"], (1, 6)),
                                                                     loss_of_sight)))
            for nm in ("kraw", "Lraw", "craw")]

def _bridle(t, m, lv, k, hc_prev, B):
    """Rival: each companion gates its own tie to the leader from its previous hidden state (a hand on the
    bridle that can let go or hold on), after John of Hainault leading Philip VI away."""
    if not m["gate_on"]:
        return k
    g = act_sigmoid(t, add(t, matmul(t, hc_prev, lv["g_w"]), lv["g_b"]))
    return mul(t, k, concat(t, [reshape(t, g, (B, 3)), t.const(np.ones((B, 3)))]))

def rollout(m, ep, t, T=T_STEPS, record=False):
    lv = {k: t.leaf(v) for k, v in m["params"].items()}
    B = ep["S"].shape[0]
    P, V = t.const(ep["P0"].copy()), t.const(np.zeros((B, N_RIDERS, 2)))
    rmsg = t.const(np.zeros((B, N_RIDERS, DM)))
    hc = t.const(np.zeros((B * N_COMP, m["H"])))
    Qa, wa = ep["G"][:, None], ep["kw"]
    terms, acc, rec = [], {"strike": 0.0, "pit": 0.0, "arrow_k": 0.0, "arrow_c": 0.0, "effort": 0.0}, []
    wpit, warr = t.const(np.full((1, 4), W_PIT)), t.const(np.array([[W_ARROW_K] + [W_ARROW_C] * 3]))
    k0, L, c = _knot(t, m, lv, ep["sight_loss"])
    for s in range(T + 1):
        pit_v, pit_g = gauss_field(t, P, ep["Q"], ep["qamp"], np.full(B, PIT_W))
        arr_v, arr_g = gauss_field(t, P, Qa, ep["kill"][:, None], wa)
        if s > 0:
            terms += [total(t, mul(t, pit_v, wpit), 1.0 / (T * B)), total(t, mul(t, arr_v, warr), 1.0 / (T * B))]
            acc["pit"] = acc["pit"] + pit_v.v.sum(1) / T
            acc["arrow_k"] = acc["arrow_k"] + arr_v.v[:, 0] / T
            acc["arrow_c"] = acc["arrow_c"] + arr_v.v[:, 1:].sum(1) / T
        if s > T - STRIKE_STEPS:
            d = sqrt_eps(t, sum_last(t, mul(t, *(2 * [add(t, take(t, P, 0), t.const(-ep["G"]))]))), 0.04)
            terms.append(total(t, d, 1.0 / (STRIKE_STEPS * B)))
            acc["strike"] = acc["strike"] + d.v / STRIKE_STEPS
        if s == T:
            break
        F = rein_forces(t, P, V, _bridle(t, m, lv, k0, hc, B), L, c)
        xk, _ = _obs_king(t, m, ep, s, P, V, pit_g, arr_g, F, rmsg)
        hk, uk = _policy(t, xk, lv, "k")
        hc, uc = _policy(t, _obs_comp(t, m, ep, s, P, V, pit_g, arr_g, F, rmsg), lv, "c")
        U = join(t, uk, reshape(t, uc, (B, N_COMP, 2)))
        terms.append(total(t, mul(t, U, U), W_EFFORT / (T * B)))
        acc["effort"] = acc["effort"] + (U.v * U.v).sum((1, 2)) / T
        if m["msg_on"]:
            M = join(t, act_tanh(t, add(t, matmul(t, hk, lv["k_Wm"]), lv["k_bm"])),
                     reshape(t, act_tanh(t, add(t, matmul(t, hc, lv["c_Wm"]), lv["c_bm"])), (B, N_COMP, DM)))
            sig = reshape(t, mul(t, din_level(t, P, ep["S"]), t.const(ep["mu"][:, None])), (B, N_RIDERS, 1))
            rmsg = add(t, mix(t, M), mul(t, sig, t.const(ep["xi_msg"][:, s])))
        drive = add(t, U, F) if m["pull_on"] else U
        V = add(t, scale(t, V, 1.0 - DAMP * DT), scale(t, drive, DT))
        P = add(t, P, scale(t, V, DT))
        if record:
            rec.append({"P": P.v.copy(), "hk": hk.v.copy(), "hc": hc.v.reshape(B, N_COMP, -1).copy(),
                        "T": F.aux.copy()})
    loss = terms[0]
    for term in terms[1:]:
        loss = add(t, loss, term)
    acc["task_loss"] = (acc["strike"] + W_PIT * acc["pit"] + W_ARROW_K * acc["arrow_k"]
                        + W_ARROW_C * acc["arrow_c"] + W_EFFORT * acc["effort"])
    acc["companion_harm"] = acc["arrow_c"]
    return loss, lv, acc, rec

def loss_and_grads(model, batch, T=T_STEPS):
    t = Tape()
    loss, lv, _, _ = rollout(model, batch, t, T=T)
    t.backward(loss)
    grads = {k: (n.g if n.g is not None else np.zeros_like(n.v)) for k, n in lv.items()}
    if MUTANT["name"] == "zero_grad_kraw" and "kraw" in grads:
        grads["kraw"] = np.zeros_like(grads["kraw"])
    if model["kind"] == "crecy":
        for k in ("kraw", "Lraw", "craw", "kraw_b", "Lraw_b", "craw_b"):
            grads[k] = np.zeros_like(grads[k])
    return float(loss.v), grads

def evaluate(model, ep, T=T_STEPS):
    _, _, acc, _ = rollout(model, ep, Tape(), T=T)
    return {k: np.asarray(v, dtype=np.float64) for k, v in acc.items()}

def predict(model, X):
    """Rider trajectories (B, T, 4, 2) for a batch of episodes."""
    _, _, _, rec = rollout(model, X, Tape(), record=True)
    return np.stack([r["P"] for r in rec], 1)

def hidden_states(model, X):
    _, _, _, rec = rollout(model, X, Tape(), record=True)
    return {"leader": np.stack([r["hk"] for r in rec], 1), "companions": np.stack([r["hc"] for r in rec], 1),
            "tension": np.stack([r["T"] for r in rec], 1)}

def tie_summary(model):
    p = model["params"]
    if "kraw" not in p:
        return None
    out = {"pairs": ["%d-%d" % (i, j) for i, j in zip(PAIRS_I, PAIRS_J)]}
    for tag, w in (("sighted", 0.0), ("blind", 1.0)):
        for nm, key in (("stiffness", "kraw"), ("rest_length", "Lraw"), ("damping", "craw")):
            out[nm + "_" + tag] = softplus(p[key] + w * p[key + "_b"]).round(3).tolist()
    return out

# ----------------------------------------------------------------- registries
def modules(model):
    k = model["kind"]
    reg = {"king_policy": {"params": ["k_W1", "k_b1", "k_W2", "k_b2"], "signature": False,
                           "role": "leader tanh MLP controller: command, hearing, degraded vision, velocity, tie reading"},
           "companion_policy": {"params": ["c_W1", "c_b1", "c_W2", "c_b2"], "signature": False,
                                "role": "shared companion tanh MLP controller: hearing, ground vision, leader position"}}
    if k != "herald":
        reg["reins_knot"] = {"params": ["kraw", "Lraw", "craw", "kraw_b", "Lraw_b", "craw_b"],
                             "signature": k == "reins",
                             "role": "learnable one-sided reciprocal spring-damper coupling applied as force, "
                                     "conditioned on the leader's loss of sight"}
        reg["rein_sense"] = {"params": [], "signature": False, "role": "tie-force observation channel"}
    if k == "herald":
        reg["herald_messages"] = {"params": ["k_Wm", "k_bm", "c_Wm", "c_bm"], "signature": False,
                                  "role": "learned broadcast message channel with position-dependent masking noise"}
    if k == "bridle":
        reg["bridle_gate"] = {"params": ["g_w", "g_b"], "signature": False,
                              "role": "companion-controlled multiplicative gate on leader-companion stiffness"}
    return reg

def knockout(model, name, mode="zero"):
    """zero: remove the module's effect (ties stop pulling; readings, messages or forces become zero)."""
    m = {**model, "params": {k: v.copy() for k, v in model["params"].items()}}
    if name == "reins_knot":
        m["pull_on"] = False
    elif name == "rein_sense":
        m["sense_on"] = False
    elif name == "herald_messages":
        m["msg_on"] = False
    elif name == "bridle_gate":
        m["gate_on"] = False
    elif name in ("king_policy", "companion_policy"):
        pre = name[0]
        m["params"][pre + "_W2"][:] = 0.0
        m["params"][pre + "_b2"][:] = 0.0
    else:
        raise KeyError(name)
    return m

def n_params(model):
    return int(sum(v.size for v in model["params"].values()))

MUTANTS = {"sign_flip": "Adam update applied with its sign flipped (loss ascends)",
           "zero_lr": "learning rate set to zero (nothing learns)",
           "zero_grad_kraw": "analytic gradient of the tie stiffness tensor zeroed"}

# ----------------------------------------------------------------- training
class NonFinite(RuntimeError):
    pass

BUDGETS = {"full": dict(steps=150, batch=48, pool=1536, val=96, test=192, lr=0.02, clip=5.0, val_every=25,
                        mutant_steps=60),
           "quick": dict(steps=70, batch=24, pool=384, val=48, test=96, lr=0.03, clip=5.0, val_every=35,
                         mutant_steps=30)}

def _copy(p):
    return {k: v.copy() for k, v in p.items()}

def fit(model, data, budget, rng, shuffle_cmd=False):
    """Adam on the collective loss; model selection on the validation split only."""
    pool = data["train"]
    if shuffle_cmd:
        pool = {**pool, "cmd": rng.permutation(pool["cmd"])}
    lr = 0.0 if MUTANT["name"] == "zero_lr" else budget["lr"]
    sign = -1.0 if MUTANT["name"] == "sign_flip" else 1.0
    opt = Adam(model["params"], lr=lr)
    init_val = float(evaluate(model, data["val"])["task_loss"].mean())
    best_val, best, hist = init_val, _copy(model["params"]), []
    n = pool["S"].shape[0]
    for step in range(budget["steps"]):
        loss, grads = loss_and_grads(model, ep_slice(pool, rng.choice(n, budget["batch"], replace=False)))
        if not np.isfinite(loss) or not all(np.isfinite(g).all() for g in grads.values()):
            raise NonFinite("non-finite loss or gradient at step %d" % step)
        clip_global(grads, budget["clip"])
        opt.step(model["params"], grads, sign)
        hist.append(loss)
        if (step + 1) % budget["val_every"] == 0 or step + 1 == budget["steps"]:
            v = float(evaluate(model, data["val"])["task_loss"].mean())
            if v < best_val:
                best_val, best = v, _copy(model["params"])
    model["params"] = best
    return {"init_val": init_val, "best_val": best_val, "hist": hist}

def train_kind(kind, data, budget, seed, shuffle_cmd=False):
    ss = np.random.SeedSequence([CHAPTER, seed, {"reins": 1, "herald": 2, "bridle": 3, "agnostic": 4, "crecy": 5}[kind]])
    init_rng, fit_rng = [np.random.default_rng(s) for s in ss.spawn(2)]
    m = build_model(rng=init_rng, kind=kind)
    info = fit(m, data, budget, fit_rng, shuffle_cmd=shuffle_cmd)
    return m, info

# ----------------------------------------------------------------- correctness tests
def c1_gradcheck(model, seed, tol):
    small = make_episodes(np.random.default_rng([CHAPTER, seed, 99]), 2, "blindspot")
    _, grads = loss_and_grads(model, small, T=6)

    def f():
        t = Tape()
        return float(rollout(model, small, t, T=6)[0].v)
    worst, per = finite_diff_check(f, model["params"], grads, np.random.default_rng([CHAPTER, seed, 7]))
    return bool(worst <= tol), float(worst), per

def c2_determinism(data, budget, seed):
    b = {**budget, "steps": 6, "val_every": 6}
    m1, i1 = train_kind("reins", data, b, seed)
    m2, i2 = train_kind("reins", data, b, seed)
    same = i1["hist"] == i2["hist"] and all(np.array_equal(m1["params"][k], m2["params"][k]) for k in m1["params"])
    ev = evaluate(m1, data["test"])
    finite = all(np.isfinite(v).all() for v in ev.values()) and all(np.isfinite(v).all() for v in m1["params"].values())
    return same and finite, "identical=%s finite=%s" % (same, finite)

def c3_learning(info, strike_model, strike_trivial, th):
    drop = (info["init_val"] - info["best_val"]) / max(info["init_val"], 1e-9)
    margin = 1.0 - strike_model / strike_trivial
    ok = drop >= th["loss_drop_fraction"] and margin >= th["margin_over_trivial"]
    return ok, "loss drop %.3f (>= %.2f); strike %.3f vs goal-agnostic %.3f, margin %.3f (>= %.2f)" % (
        drop, th["loss_drop_fraction"], strike_model, strike_trivial, margin, th["margin_over_trivial"])

def _rand_state(rng, B=256):
    P = rng.normal(0, 1.5, (B, N_RIDERS, 2))
    V = rng.normal(0, 2.0, (B, N_RIDERS, 2))
    k, L, c = rng.uniform(0.01, 20, (1, 6)), rng.uniform(0.05, 3, (1, 6)), rng.uniform(0.0, 5, (1, 6))
    return P, V, k, L, c

def _forces(P, V, k, L, c, asym=1.0):
    t = Tape()
    return rein_forces(t, t.const(P), t.const(V), t.const(k), t.const(L), t.const(c), asym)

def c6_properties(rng):
    """Invariants on random parameters and states (random search, 40 rounds x 256 states), each with a
    negative control that must violate it. C6.4 is a labelled definition check, not evidence."""
    res = {}
    viol = neg = 0.0
    for _ in range(40):
        P, V, k, L, c = _rand_state(rng)
        viol = max(viol, float(np.abs(_forces(P, V, k, L, c).v.sum(1)).max()))
        neg = max(neg, float(np.abs(_forces(P, V, k, L, c, asym=0.8).v.sum(1)).max()))
    res["C6.1 reciprocity (sum of tie forces = 0)"] = (viol < 1e-9 and neg > 1e-3, "max |sum F| %.1e; asymmetric "
                                                       "control %.2e" % (viol, neg))
    tmin = lin_min = 0.0
    for _ in range(40):
        P, V, k, L, c = _rand_state(rng)
        F = _forces(P, V, k, L, c)
        tmin = min(tmin, float(F.aux.min()))
        D = P[:, PAIRS_J] - P[:, PAIRS_I]
        r = np.sqrt((D * D).sum(-1) + EPS_R ** 2)
        vr = ((V[:, PAIRS_J] - V[:, PAIRS_I]) * D / r[..., None]).sum(-1)
        lin_min = min(lin_min, float((k * (r - L) + c * vr).min()))
    res["C6.2 one-sided (a rein never pushes: T >= 0)"] = (tmin >= 0.0 and lin_min < -1e-3,
                                                         "min T %.2e; linear-spring control min %.2f" % (tmin, lin_min))
    worst = neg_m = 0.0
    for _ in range(40):
        P, V, k, L, c = _rand_state(rng)
        V[:] = 0.0
        D = P[:, PAIRS_J] - P[:, PAIRS_I]
        n = D / np.sqrt((D * D).sum(-1) + EPS_R ** 2)[..., None]
        P2 = P.copy()
        np.add.at(P2, (slice(None), PAIRS_J[0]), 0.05 * n[:, 0])
        worst = min(worst, float((_forces(P2, V, k, L, c).aux[:, 0] - _forces(P, V, k, L, c).aux[:, 0]).min()))
        neg_m = min(neg_m, float((_forces(P2, V, -k, L, c).aux[:, 0] - _forces(P, V, -k, L, c).aux[:, 0]).min()))
    res["C6.3 tension non-decreasing in stretch"] = (worst >= -1e-12 and neg_m < -1e-6,
                                                     "min dT %.1e; negative-stiffness control %.2e" % (worst, neg_m))
    return res

def c6_definition_check(model, ep):
    ko = knockout(model, "reins_knot")
    a, b = predict(ko, ep), predict(knockout(knockout(model, "reins_knot"), "rein_sense"), ep)
    return True, "definition check (not evidence): severed ties run; trajectories differ by reading only: %.3f" % (
        float(np.abs(a - b).max()))

def c7_split_integrity(data):
    tr = ep_hashes(data["train"])
    overlap = sum(len(tr & ep_hashes(data[s])) for s in ("val", "test", "shifted", "blindspot"))
    return overlap == 0, "train/eval overlapping episodes: %d" % overlap

def c5_mutants(data, budget, seed, th, strike_trivial, tol):
    """A mutant is detected when C1 or C3 fails; C1 runs first and C3 (a short training) only if C1 passes."""
    detected, lines, prev = 0, [], MUTANT["name"]
    for name in MUTANTS:
        MUTANT["name"] = name
        try:
            ok1, ok3 = c1_gradcheck(build_model(rng=np.random.default_rng(seed), kind="reins"), seed, tol)[0], None
            if ok1:
                m, info = train_kind("reins", data, {**budget, "steps": budget["mutant_steps"]}, seed)
                ok3 = c3_learning(info, float(evaluate(m, data["test"])["strike"].mean()), strike_trivial, th)[0]
        finally:
            MUTANT["name"] = prev
        caught = (not ok1) or (ok3 is False)
        detected += int(caught)
        lines.append("%s: C1 %s, C3 %s -> %s" % (name, "pass" if ok1 else "FAIL",
                                                  {None: "not run", True: "pass", False: "FAIL"}[ok3],
                                                  "detected" if caught else "MISSED"))
    return detected == len(MUTANTS), detected, lines

# ----------------------------------------------------------------- hypothesis protocol
def run_seed(seed, budget):
    data = make_data(np.random.default_rng(np.random.SeedSequence([CHAPTER, seed])), budget)
    out, models = {}, {}
    for kind in ("reins", "herald", "bridle", "crecy"):
        models[kind], out[kind + "_info"] = train_kind(kind, data, budget, seed)
    for kind, m in models.items():
        for split in ("test", "shifted", "blindspot"):
            out[(kind, split)] = evaluate(m, data[split])
    r = models["reins"]
    out[("ko_pull", "shifted")] = evaluate(knockout(r, "reins_knot"), data["shifted"])
    out[("ko_sense", "shifted")] = evaluate(knockout(r, "rein_sense"), data["shifted"])
    out["ties"] = tie_summary(r)
    return out, models, data

def hypothesis_rows(runs, rng):
    def mean(key, metric):
        return np.array([float(r[key][metric].mean()) for r in runs])
    specs = {h["id"]: h for h in MIND_CARD["hypotheses"]}
    diffs = {"H-SIG": mean(("reins", "shifted"), "task_loss") - mean(("herald", "shifted"), "task_loss"),
             "H-NEC": mean(("ko_pull", "shifted"), "task_loss") - mean(("ko_sense", "shifted"), "task_loss"),
             "H-BLIND": mean(("reins", "blindspot"), "companion_harm") - mean(("herald", "blindspot"), "companion_harm"),
             "H-RIVAL": mean(("reins", "blindspot"), "companion_harm") - mean(("bridle", "blindspot"), "companion_harm")}
    rows = []
    for hid, d in diffs.items():
        mu, ci = paired_bootstrap(d, rng)
        s = specs[hid]
        v = verdict(mu, ci, s["direction"], s["mesi"]) if len(d) >= 5 else "not evaluated"
        rows.append({"id": hid, "metric": s["metric"], "mean_diff": mu, "ci95": ci, "mesi": s["mesi"],
                     "n_seeds": len(d), "verdict": v, "per_seed": d.round(4).tolist()})
    return rows

def descriptive_table(runs):
    cells = []
    for kind in ("reins", "herald", "bridle", "crecy"):
        for split in ("test", "shifted", "blindspot"):
            for metric in ("task_loss", "strike", "pit", "companion_harm"):
                vals = [float(r[(kind, split)][metric].mean()) for r in runs]
                cells.append((kind, split, metric, float(np.mean(vals)), float(np.std(vals))))
    return cells

# ----------------------------------------------------------------- report
def print_report(rep):
    line = "-" * 78
    print("=== VERIFIED REPORT · chapter %04d ===" % CHAPTER)
    print("figure: %s (%d-%d) · provenance: %s · card revision %d" % (
        MIND_CARD["figure"], MIND_CARD["born"], MIND_CARD["died"], MIND_CARD["provenance"], MIND_CARD["card_revision"]))
    e = rep["environment"]
    print("python %s · numpy %s · mode %s · seeds %s · runtime %.1f s" % (
        e["python"], e["numpy"], rep["mode"], rep["seeds"], rep["runtime_s"]))
    print("parameters: reins %d · herald %d (%+.1f%%) · bridle %d (%+.1f%%)" % (
        rep["n_params"], rep["n_params_herald"], 100 * (rep["n_params_herald"] / rep["n_params"] - 1),
        rep["n_params_bridle"], 100 * (rep["n_params_bridle"] / rep["n_params"] - 1)))
    g = rep["gradcheck"]
    print("gradient check: %d/%d tensors, max rel error %.2e (tol %.0e), at %s -> %s" % (
        g["tensors_checked"], g["tensors_total"], g["max_rel_error"], g["tol"], "+".join(g["checked_at"]),
        "PASS" if g["passed"] else "FAIL"))
    print(line)
    for c in rep["correctness"]:
        print("%-5s %-44s %s  %s" % (c["id"], c["name"][:44], "PASS" if c["passed"] else "FAIL", c["detail"]))
    print("mutation score: %d/%d" % (rep["mutants"]["detected"], rep["mutants"]["total"]))
    for ln in rep["mutants"]["lines"]:
        print("   " + ln)
    print(line)
    print("%-8s %-15s %10s  %-22s %6s  %s" % ("id", "metric", "mean diff", "95% CI", "mesi", "verdict"))
    for h in rep["hypotheses"]:
        print("%-8s %-15s %+10.4f  [%+.4f, %+.4f]  %6.3f  %s" % (
            h["id"], h["metric"], h["mean_diff"], h["ci95"][0], h["ci95"][1], h["mesi"], h["verdict"]))
        print("         per seed: %s" % h["per_seed"])
    print(line)
    for k in rep["knockouts"]:
        print("knockout %-12s (signature=%s) shifted task_loss change %+.4f  CI [%+.4f, %+.4f]" % (
            k["module"], k["signature"], k["metric_change"], k["ci95"][0], k["ci95"][1]))
    print(line)
    print("descriptive means over seeds (mean ± sd):")
    for kind, split, metric, mu, sd in rep["descriptive"]:
        if metric in ("task_loss", "companion_harm", "strike", "pit"):
            print("   %-7s %-9s %-15s %.4f ± %.4f" % (kind, split, metric, mu, sd))
    if rep.get("ties"):
        print("learned knot (seed %d):" % rep["seeds"][0])
        for k, v in rep["ties"].items():
            print("   %-20s %s" % (k, v))
    print("task types: %s · exit code %d" % (", ".join(TASK_TYPES), rep["exit_code"]))
    print("=== END REPORT ===")

# ----------------------------------------------------------------- protocol
def _check(rows, cid, name, ok, detail):
    rows.append({"id": cid, "name": name, "passed": bool(ok), "detail": detail})

def correctness_suite(budget, base_seed, t0, rows):
    th, tol = MIND_CARD["thresholds"], MIND_CARD["thresholds"]["gradcheck_rel_tol"]
    data = make_data(np.random.default_rng(np.random.SeedSequence([CHAPTER, base_seed])), budget)
    grad = {"worst": 0.0, "n": 0, "total": 0, "ok": True}
    for kind in ("reins", "herald", "bridle"):
        m = build_model(rng=np.random.default_rng([CHAPTER, base_seed, 50]), kind=kind)
        ok, worst, per = c1_gradcheck(m, base_seed, tol)
        grad.update(worst=max(grad["worst"], worst), n=grad["n"] + len(per), total=grad["total"] + len(m["params"]),
                    ok=grad["ok"] and ok)
        mt, _ = train_kind(kind, data, {**budget, "steps": 50, "val_every": 50}, base_seed)
        ok, worst, per = c1_gradcheck(mt, base_seed, tol)
        grad.update(worst=max(grad["worst"], worst), ok=grad["ok"] and ok)
    _check(rows, "C1", "gradient check, all tensors, init + 50 steps", grad["ok"],
           "max rel error %.2e over reins, herald, bridle" % grad["worst"])
    ok, det = c2_determinism(data, budget, base_seed)
    _check(rows, "C2", "determinism and finiteness", ok, det)
    model, info = train_kind("reins", data, budget, base_seed)
    triv, _ = train_kind("agnostic", data, budget, base_seed)
    s_model = float(evaluate(model, data["test"])["strike"].mean())
    s_triv = float(evaluate(triv, data["test"])["strike"].mean())
    ok, det = c3_learning(info, s_model, s_triv, th)
    _check(rows, "C3", "learning vs goal-agnostic trivial", ok, det)
    shuf, _ = train_kind("reins", data, budget, base_seed, shuffle_cmd=True)
    s_shuf = float(evaluate(shuf, data["test"])["strike"].mean())
    _check(rows, "C4", "shuffled-command control", s_shuf >= (1 - th["shuffle_band"]) * s_triv,
           "strike %.3f vs goal-agnostic %.3f (must be >= %.0f%%)" % (s_shuf, s_triv, 100 * (1 - th["shuffle_band"])))
    ok, detected, lines = c5_mutants(data, budget, base_seed, th, s_triv, tol)
    _check(rows, "C5", "mutant detection", ok, "%d/%d detected" % (detected, len(MUTANTS)))
    for name, (ok, det) in c6_properties(np.random.default_rng([CHAPTER, base_seed, 6])).items():
        _check(rows, name.split(" ")[0], name[5:], ok, det)
    ok, det = c6_definition_check(model, ep_slice(data["test"], np.arange(8)))
    _check(rows, "C6.4", "sever knockout (definition check)", ok, det)
    ok, det = c7_split_integrity(data)
    _check(rows, "C7", "split integrity", ok, det)
    return grad, {"detected": detected, "total": len(MUTANTS), "score": detected / len(MUTANTS), "lines": lines}

def data_bridge(path, model):
    """Optional: replay a real walking trajectory (CSV frame,ped,x,y; ETH/UCY style) as a sighted guide's path
    and measure how a passive follower held by the learned blind knot tracks it."""
    if not path:
        print("real-data bridge: no --data PATH given; skipped")
        return None
    raw = np.loadtxt(path, delimiter=",", ndmin=2)
    ped = raw[raw[:, 1] == raw[0, 1]][:, 2:4]
    if len(ped) < 4:
        print("real-data bridge: trajectory too short; skipped")
        return None
    p = model["params"]
    k, L, c = [float(softplus(p[a][0] + p[a + "_b"][0])) for a in ("kraw", "Lraw", "craw")]
    x, v, err = ped[0] - np.array([0.5, 0.0]), np.zeros(2), []
    for g in ped[1:]:
        d = g - x
        r = math.sqrt(float(d @ d) + EPS_R ** 2)
        T = float(softplus(BETA * (k * (r - L) + c * float(-v @ d) / r))) / BETA
        v = (1 - DAMP * DT) * v + DT * T * d / r
        x = x + DT * v
        err.append(float(np.linalg.norm(g - x)))
    print("real-data bridge: %d steps, mean guide-follower distance %.3f" % (len(err), float(np.mean(err))))
    return float(np.mean(err))

def assemble_report(runs, models, rows, grad, mut, a, mode, seeds, runtime):
    rng = np.random.default_rng([CHAPTER, a.seed, 2000])
    if runs:
        hyps = hypothesis_rows(runs, rng)
        ko = [{"module": nm, "signature": sig, "metric_change": mu, "ci95": ci}
              for nm, sig, key in (("reins_knot", True, "ko_pull"), ("rein_sense", False, "ko_sense"))
              for mu, ci in [paired_bootstrap([float(r[(key, "shifted")]["task_loss"].mean()
                                                     - r[("reins", "shifted")]["task_loss"].mean()) for r in runs], rng)]]
        desc, ties = descriptive_table(runs), runs[0]["ties"]
        data_bridge(a.data, models["reins"])
    else:
        hyps = [{"id": h["id"], "metric": h["metric"], "mean_diff": 0.0, "ci95": [0.0, 0.0], "mesi": h["mesi"],
                 "n_seeds": 0, "verdict": "not evaluated", "per_seed": []} for h in MIND_CARD["hypotheses"]]
        ko, desc, ties = [], [], None
    probe = build_model(rng=np.random.default_rng(0))
    return {"schema_version": "1.0", "chapter": CHAPTER, "file": os.path.basename(__file__),
            "card_revision": MIND_CARD["card_revision"], "mode": mode,
            "environment": {"python": sys.version.split()[0], "numpy": np.__version__}, "seeds": seeds,
            "runtime_s": runtime, "n_params": n_params(probe),
            "n_params_herald": n_params(build_model(rng=np.random.default_rng(0), kind="herald")),
            "n_params_bridle": n_params(build_model(rng=np.random.default_rng(0), kind="bridle")),
            "gradcheck": {"tensors_checked": grad["n"], "tensors_total": grad["total"], "max_rel_error": grad["worst"],
                          "tol": MIND_CARD["thresholds"]["gradcheck_rel_tol"],
                          "checked_at": ["init", "after_training_steps"], "passed": bool(grad["ok"])},
            "correctness": rows, "mutants": mut, "hypotheses": hyps, "knockouts": ko, "descriptive": desc,
            "ties": ties, "task_types": TASK_TYPES}

def main(argv=None):
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--card", action="store_true")
    ap.add_argument("--mutant", type=str, default=None)
    ap.add_argument("--data", type=str, default=None)
    try:
        a = ap.parse_args(argv)
    except SystemExit:
        return 2
    if a.card:
        print(json.dumps(MIND_CARD, indent=2))
        return 0
    if a.mutant is not None and a.mutant not in MUTANTS:
        print("unknown mutant; choose from %s" % sorted(MUTANTS))
        return 2
    t0, mode = time.time(), "quick" if a.quick else "full"
    budget, limit = BUDGETS[mode], 20.0 if a.quick else 180.0
    seeds = [a.seed] if a.quick else [a.seed + i for i in range(a.seeds)]
    print("seeds: %s (mode %s, mutant %s)" % (seeds, mode, a.mutant))
    rows, MUTANT["name"] = [], a.mutant
    try:
        grad, mut = correctness_suite(budget, a.seed, t0, rows)
        runs, models = [], None
        if not a.quick:
            for s in seeds:
                out, models, _ = run_seed(s, budget)
                runs.append(out)
        MUTANT["name"] = None
    except NonFinite as exc:
        print("non-finite values: %s" % exc)
        return 4
    runtime = time.time() - t0
    _check(rows, "C8", "runtime budget", runtime <= limit, "%.1f s of %.0f s" % (runtime, limit))
    rep = assemble_report(runs, models, rows, grad, mut, a, mode, seeds, runtime)
    failed = [c for c in rows if not c["passed"]]
    code = 3 if (failed and all(c["id"] == "C8" for c in failed)) else (1 if failed else 0)
    rep["exit_code"] = code
    print_report(rep)
    if a.json:
        write_report(a.json, rep)
    return code

if __name__ == "__main__":
    sys.exit(main())

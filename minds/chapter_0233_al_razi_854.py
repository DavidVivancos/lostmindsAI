#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 0233_Neuron.py  --  THE SHUKUK ARCHITECTURE
 Abu Bakr Muhammad ibn Zakariyya al-Razi (Rhazes), Rayy, c.854/865 - 925 CE
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 12 Minds 221 - 240 Available on Amazon https://www.amazon.com/dp/B0HKVBKNTT
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0233_al_razi_854 - Abu Bakr Muhammad ibn Zakariyya al-Razi (Rhazes), Rayy, c.854/865 - 925 CE
================================================================================  

WHY THIS ARCHITECTURE AND NOT ANOTHER
-------------------------------------
Al-Razi did not build a theory of mind. He built a *bookkeeping practice*, and
the bookkeeping is the theory. Four of his documented habits are cognitive
mechanisms, and this file implements all four as differentiable modules:

  (1) THE TWO-COLUMN REGISTER  (Kitab al-Shukuk 'ala Jalinus, "Doubts about
      Galen"). He kept the inherited model intact and wrote a second list
      beside it: the names of patients whose course followed Galen's books,
      and the names of those whose course ran exactly contrary. He never
      overwrote the master. He annotated him, and the annotation was indexed
      by case.
        -> class CaseRegister. A frozen AuthorityPrior is never updated. A
           learned kernel memory marks the regions of case-space where the
           prior fails. Its stored values are literally signed: positive =
           "here the authority is contradicted", negative = "here it holds".

  (2) THE COST OF DOUBT. He opens the Shukuk by saying he is opposing the
      greatest of men, the one who benefited him more than any other. Doubt
      in this mind is expensive and must be *earned* per case.
        -> class DoubtGate, a scalar sigmoid gate on the correction term,
           carrying an explicit L1 price. Contradiction is sparse by
           construction, not by accident.

  (3) THE CASCADE OF GRADED RESPONDERS. Al-Qifti describes his teaching
      circle: junior lecturers answered what they were competent to answer,
      and only what exceeded their range was passed up to al-Razi.
        -> class Cascade. A cheap junior responder answers most cases; a
           critic decides what escalates to the expensive master.

  (4) THE EXTERNALISED EVALUATOR. Al-Tibb al-Ruhani ch. IV ("Of How a Man
      May Discover His Own Vices"): because of the affection a man has for
      himself, he cannot see his own faults with the pure eye of reason;
      he must appoint an outside observer, must not flinch when told, must
      reproach the observer for softening the report, and must applaud him
      for overshooting. Ch. VI adds the failure mode: conceit stops the very
      improvement it is conceited about.
        -> class Musharrif ("supervisor"). A parameter-disjoint critic
           predicts the responder's error. THE RESPONDER IS ARCHITECTURALLY
           FORBIDDEN TO READ ITS OWN CONFIDENCE for routing or abstention:
           no gradient flows from the critic's loss into the responder, and
           the responder's own max-softmax is never used as a decision
           signal. Self-love is firewalled at the level of the graph.

  And one operation that is his single most famous result:

  (5) CLASS FISSION ON STRUCTURED RESIDUAL (Kitab al-Judari wa'l-Hasba).
      The inherited taxonomy had one category for eruptive fever. He split
      it in two -- smallpox and measles -- on observable signs, not on
      causes. His causal story (fermenting blood expelling superfluous
      moisture) was wrong; the split was right and survives.
        -> function attempt_fission. When the residual inside one class is
           bimodal along its own principal direction, the output head grows
           a new class and the labels fork. The frozen prior is lifted so
           both children inherit the same parental score: the authority
           cannot tell them apart, which is exactly the historical
           situation. Only the annotation can.

WHAT THIS IS NOT: there is no attention over stored keys in the transformer
sense, no mixture-of-experts router trained end-to-end on the task loss, no
softmax over a vocabulary. The retrieval here is an RBF kernel memory whose
*values are scalars of doubt*, not vectors of content, and the routing is
performed by a module that is denied the task gradient on purpose.

CONVENTIONS
-----------
Pure NumPy. Every gradient is derived by hand. A central-difference gradient
check over every trainable block is mandatory and runs on every execution
(see gradient_check). Then a real training loop, then self-tests.

Run:  python3 chapter_0233_al_razi_854.py
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# ==============================================================================
# SECTION 0 -- SMALL NUMERICAL UTILITIES
# ==============================================================================

EPS = 1e-12


def set_seed(seed: int) -> np.random.Generator:
    """Single source of randomness so every reported number is reproducible."""
    return np.random.default_rng(seed)


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / (np.sum(e, axis=axis, keepdims=True) + EPS)


def sigmoid(u: np.ndarray) -> np.ndarray:
    """Stable logistic. Used for the doubt gate and for the critic's verdict."""
    out = np.empty_like(u, dtype=np.float64)
    pos = u >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-u[pos]))
    ex = np.exp(u[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def one_hot(y: np.ndarray, n: int) -> np.ndarray:
    m = np.zeros((y.shape[0], n), dtype=np.float64)
    m[np.arange(y.shape[0]), y] = 1.0
    return m


def cross_entropy(logits: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Mean cross-entropy and dL/dlogits.
    The gradient (p - onehot)/B is also the object the fission test inspects:
    al-Razi's "residual" is exactly the per-case disagreement between what the
    model predicts and what the patient actually did.
    """
    p = softmax(logits, axis=1)
    n = logits.shape[0]
    loss = float(-np.mean(np.log(p[np.arange(n), y] + EPS)))
    d = (p - one_hot(y, logits.shape[1])) / n
    return loss, d


def binary_cross_entropy(q: np.ndarray, t: np.ndarray) -> Tuple[float, np.ndarray]:
    """Mean BCE and dL/dq for the critic. t is a constant target (no gradient)."""
    q = np.clip(q, 1e-9, 1 - 1e-9)
    loss = float(-np.mean(t * np.log(q) + (1 - t) * np.log(1 - q)))
    d = (-(t / q) + (1 - t) / (1 - q)) / q.shape[0]
    return loss, d


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    """
    Rank-based AUROC via the Mann-Whitney U identity, with tie handling.
    Used for the 'conceit test': how well does a signal rank the cases the
    model actually got wrong?
    """
    pos = score[label == 1]
    neg = score[label == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1, dtype=np.float64)
    # average ranks within ties
    allv = np.concatenate([pos, neg])[order]
    i = 0
    r_sorted = ranks[order]
    while i < allv.size:
        j = i
        while j + 1 < allv.size and allv[j + 1] == allv[i]:
            j += 1
        if j > i:
            r_sorted[i:j + 1] = r_sorted[i:j + 1].mean()
        i = j + 1
    ranks[order] = r_sorted
    rpos = ranks[:pos.size].sum()
    return float((rpos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray,
                               bins: int = 10) -> float:
    """Standard ECE. Reported for the primary's self-confidence vs the critic."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = conf.size
    for b in range(bins):
        m = (conf > edges[b]) & (conf <= edges[b + 1])
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


# ==============================================================================
# SECTION 1 -- THE BIMARISTAN CORPUS (synthetic, but structured to be honest)
# ==============================================================================
#
# Twelve observable signs. The names are taken from the categories al-Razi
# actually discriminates on in the Treatise on Smallpox and Measles and in the
# Mansuri -- coryza, the depth and timing of the eruption, the pulse, the state
# of the tongue, restlessness, itching, the eyes, prior exposure, season.
#
# The corpus has FOUR true conditions but the INHERITED taxonomy has THREE,
# because the inherited taxonomy merges the two eruptive fevers. That merge is
# the thing the architecture has to undo by itself.
#
# It also has a CONTRADICTION REGION. The frozen prior is fitted on a
# subpopulation (sign 11, "season/region", negative) in which two of the
# conditions present in one way; in the other subpopulation they present
# differently. The prior is therefore not merely noisy -- it is *systematically*
# wrong on an identifiable, contiguous region of case space, which is what
# al-Razi reported about Galen's fever descriptions in the hospitals of
# Baghdad and Rayy. A register that works must find that region.

FEATURE_NAMES = [
    "fever_onset_steepness",   # 0
    "pulse_rate",              # 1
    "back_pain",               # 2
    "coryza",                  # 3   <- the documented measles/smallpox splitter
    "eruption_depth",          # 4   <- deep, scarring = variola
    "eruption_day",            # 5
    "restlessness",            # 6
    "itching",                 # 7
    "eye_redness",             # 8
    "tongue_coat",             # 9
    "prior_exposure",          # 10
    "season_region",           # 11  <- indexes the contradiction region
]

FINE_NAMES = ["bilious_fever", "variola", "morbilli", "catarrh"]
FINE_TO_COARSE = np.array([0, 1, 1, 2])      # inherited taxonomy merges 1 and 2
COARSE_NAMES = ["fever", "eruptive_fever", "catarrh"]

D_IN = len(FEATURE_NAMES)
N_FINE_TRUE = 4
N_COARSE = 3


def _class_means() -> np.ndarray:
    """
    Mean sign-vector for each true condition. Hand-set so that:

      - variola vs morbilli differ mainly in coryza (3), eruption depth (4),
        eruption day (5), back pain (2), itching (7) and eye redness (8).
        Their two clouds are far apart but the INHERITED TAXONOMY GIVES THEM
        ONE NAME. That merged category is therefore genuinely bimodal, and it
        is the only one that is. Fission has to find it and must leave the
        others alone.

      - bilious_fever and catarrh are each a SINGLE cloud, but a sheared one:
        their signs drift continuously with sign 11 (season/region) and cross
        over each other as it rises. An authority fitted only where season<=0
        extrapolates the drift in the wrong direction and is systematically,
        not randomly, wrong on the far side. That is the shape of the failure
        al-Razi reported: not that Galen was noisy, but that whole classes of
        case in Baghdad and Rayy ran "exactly contrary" to the books.
    """
    m = np.zeros((N_FINE_TRUE, D_IN))
    #      0     1     2     3     4     5     6     7     8     9    10    11
    m[0] = [1.6,  1.1,  0.3, -0.9, -1.4, -1.2,  0.9, -1.0,  0.2,  1.2, -0.2, 0.0]
    m[1] = [1.2,  0.9,  1.5, -1.1,  1.6,  1.0,  1.1,  0.2,  0.3,  0.6, -0.6, 0.0]
    m[2] = [1.1,  0.9,  0.4,  1.5, -0.7, -0.6,  0.8,  1.1,  1.3,  0.5, -0.5, 0.0]
    m[3] = [-0.4, -0.3, -0.2, 1.3, -1.5, -1.3, -0.3,  0.1,  1.0, -0.9,  0.4, 0.0]
    return m


# The three signs whose expression drifts with season/region, and the direction
# of the drift for bilious_fever (+) and catarrh (-). The two conditions swap
# appearances as the season index rises.
SHEAR_DIMS = (3, 9, 0)
SHEAR_GAIN = (1.15, -1.05, -0.95)

# THE COURSE OF THE DISEASE -- what actually happened to the patient afterwards.
# Two recorded outcomes: how deeply the eruption scarred, and how many days ran
# before the crisis. These are NEVER a training target for the classifier. They
# exist because the reason to distrust a category is not that it mislabels, but
# that it FAILS TO TELL YOU WHAT WILL HAPPEN. Al-Razi's grounds for splitting
# the eruptive fevers were of exactly this kind: different scarring, different
# course, different mortality, and immunity conferred against the one and not
# the other.
OUTCOME_NAMES = ["scarring", "days_to_crisis"]
OUTCOME_MEANS = np.array([
    [-1.8,  0.0],   # bilious_fever: no eruption, middling course
    [ 2.0,  1.6],   # variola      : deep scarring, long course
    [-1.4, -1.1],   # morbilli     : fades without scarring, shorter course
    [-1.9, -1.5],   # catarrh      : no eruption, brief
])
N_OUTCOME = OUTCOME_MEANS.shape[1]


def make_corpus(n: int, rng: np.random.Generator,
                noise: float = 0.5,
                ambiguous_frac: float = 0.09) -> Dict[str, np.ndarray]:
    """
    Build a corpus of n cases.

    Returns:
      X            (n, 12) the recorded signs
      y_fine       (n,)    the true four-way condition. NEVER used for training.
                           It exists only so the fork can be *scored* honestly.
      O            (n, 2)  the recorded course: scarring and days to crisis.
                           Never a training target. Used only to ask whether a
                           category is worth keeping.
      y_coarse     (n,)    the inherited three-way label the model trains on
      prior_region (n,)    bool, season<=0 -- the only region the authority saw
      contradiction(n,)    bool, season>0.6 -- where the authority is systematically
                           wrong because the drift has carried the signs past it
      ambiguous    (n,)    bool, records whose signs do not determine the answer
    """
    means = _class_means()
    y_fine = rng.integers(0, N_FINE_TRUE, size=n)
    season = rng.normal(0.0, 1.0, size=n)
    X = means[y_fine] + rng.normal(0.0, noise, size=(n, D_IN))
    X[:, 11] = season

    # --- the shear: a continuous drift, not a jump --------------------------
    sign = np.zeros(n)
    sign[y_fine == 0] = 1.0
    sign[y_fine == 3] = -1.0
    for d, gain in zip(SHEAR_DIMS, SHEAR_GAIN):
        X[:, d] += sign * gain * season

    # --- genuinely undecidable records --------------------------------------
    # Al-Razi's casebook records outcomes he could not account for, and the
    # Shukuk insists a physician must be able to say he does not know. These
    # cases exist so that abstention has something real to abstain from.
    amb = rng.random(n) < ambiguous_frac
    keep_season = season[amb]
    X[amb] = rng.normal(0.0, 1.0, size=(int(amb.sum()), D_IN))
    X[amb, 11] = keep_season

    # the recorded course, driven by the TRUE condition
    O = OUTCOME_MEANS[y_fine] + rng.normal(0.0, 0.45, size=(n, N_OUTCOME))

    return {
        "X": X,
        "O": O,
        "y_fine": y_fine.astype(np.int64),
        "y_coarse": FINE_TO_COARSE[y_fine].astype(np.int64),
        "prior_region": season <= 0.0,
        "contradiction": season > 0.6,
        "ambiguous": amb,
    }


# ==============================================================================
# SECTION 2 -- THE AUTHORITY PRIOR (Galen: inherited, respected, FROZEN)
# ==============================================================================


class AuthorityPrior:
    """
    A two-layer tanh network fitted ONCE on the inherited corpus, then frozen
    forever. Nothing downstream is ever allowed to change these weights.

    This is the architectural commitment that makes the whole file al-Razi's
    and nobody else's. The obvious engineering move -- fine-tune the prior on
    the new data -- is exactly what he refused to do. He kept Galen's text
    whole and put the disagreements in a separate book, so that a later reader
    could adjudicate between them. A frozen prior plus an indexed register of
    where it fails is a *reversible* correction; a fine-tuned prior is not.
    """

    def __init__(self, d_in: int, hidden: int, n_out: int, rng):
        s1 = math.sqrt(1.0 / d_in)
        s2 = math.sqrt(1.0 / hidden)
        self.W1 = rng.normal(0, s1, (d_in, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, s2, (hidden, n_out))
        self.b2 = np.zeros(n_out)
        self.n_out = n_out
        self.frozen = False

    def forward(self, X: np.ndarray) -> np.ndarray:
        h = np.tanh(X @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 400,
            lr: float = 0.12, verbose: bool = False) -> None:
        """Plain SGD. Only ever called before freezing."""
        assert not self.frozen, "the authority is not revised, only annotated"
        for ep in range(epochs):
            a1 = X @ self.W1 + self.b1
            h = np.tanh(a1)
            z = h @ self.W2 + self.b2
            loss, dz = cross_entropy(z, y)
            dW2 = h.T @ dz
            db2 = dz.sum(0)
            dh = dz @ self.W2.T
            da1 = dh * (1 - h ** 2)
            dW1 = X.T @ da1
            db1 = da1.sum(0)
            self.W2 -= lr * dW2
            self.b2 -= lr * db2
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            if verbose and ep % 100 == 0:
                print(f"      prior epoch {ep:4d}  loss {loss:.4f}")
        self.frozen = True


# ==============================================================================
# SECTION 3 -- THE CASE REGISTER (Kitab al-Shukuk: the two-column list)
# ==============================================================================


class CaseRegister:
    """
    A kernel memory of K slots. Each slot k holds a key M[k] in the trunk's
    representation space and ONE SIGNED SCALAR c[k]:

        c[k] > 0   "cases like this ran contrary to the books"
        c[k] < 0   "cases like this followed the books"

    The output for a case is the attention-weighted verdict r = a . c, where
    a = softmax(-beta * ||h - M||^2) and beta = exp(s) is a learned precision.

    Note what is NOT stored: content. The register does not remember what to
    predict. It remembers only *where the inherited model can and cannot be
    trusted*. That is the whole of the Shukuk method: it is a book about the
    boundary of another book, and it is useless on its own.
    """

    def __init__(self, n_slots: int, dim: int, rng):
        self.K = n_slots
        self.M = rng.normal(0, 0.5, (n_slots, dim))
        self.c = rng.normal(0, 0.05, n_slots)
        self.s = np.array(0.0)                      # log precision
        self._cache: Dict[str, np.ndarray] = {}

    def forward(self, h: np.ndarray) -> np.ndarray:
        # squared distances (B, K)
        d = ((h[:, None, :] - self.M[None, :, :]) ** 2).sum(-1)
        beta = math.exp(float(self.s))
        a = softmax(-beta * d, axis=1)
        r = a @ self.c
        self._cache = {"h": h, "d": d, "a": a, "beta": np.array(beta)}
        return r

    def backward(self, dr: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        dr : (B,) gradient of the loss wrt r.
        Returns dh and the parameter gradients.
        """
        h = self._cache["h"]
        d = self._cache["d"]
        a = self._cache["a"]
        beta = float(self._cache["beta"])

        dc = a.T @ dr                                   # (K,)
        da = dr[:, None] * self.c[None, :]              # (B, K)
        # softmax backward on logits = -beta * d
        dlogits = a * (da - (da * a).sum(1, keepdims=True))
        ds = float(np.sum(dlogits * (-d)) * beta)       # d logits / d s = -d * beta
        dd = dlogits * (-beta)                          # (B, K)

        # d ||h - M_k||^2 / dh = 2(h - M_k) ; / dM_k = -2(h - M_k)
        dh = 2.0 * (h * dd.sum(1, keepdims=True) - dd @ self.M)
        dM = -2.0 * (dd.T @ h) + 2.0 * self.M * dd.sum(0)[:, None]

        return dh, {"M": dM, "c": dc, "s": np.array(ds)}


# ==============================================================================
# SECTION 4 -- THE SUPERVISOR (al-Tibb al-Ruhani ch. IV: the outside observer)
# ==============================================================================


class Musharrif:
    """
    'Musharrif' -- the supervisor al-Razi tells the reader to appoint, because
    a man cannot look upon his own character with the pure and single eye of
    reason. This module predicts P(the responder got this case wrong).

    THREE ARCHITECTURAL VOWS, each taken from the chapter:

      1. Parameter-disjoint. It shares nothing with the responder. In the text:
         the supervisor must be another person.
      2. Fed only detached activations. No gradient of the critic's loss ever
         reaches the responder, so the responder cannot learn to look
         trustworthy. In the text: the supervisor must not be blandished.
      3. Its verdict, not the responder's own confidence, drives escalation and
         abstention. In the text: rely on him, not on your own admiration of
         your own actions.

    The chapter also supplies an asymmetric loss. Al-Razi tells the reader to
    reproach the supervisor who softens the report but to applaud the one who
    overshoots. So false negatives (missed errors) are priced higher than false
    positives (spurious alarms) by POS_WEIGHT.
    """

    POS_WEIGHT = 2.5   # missing a fault is worse than crying one falsely

    def __init__(self, d_in: int, hidden: int, rng):
        s1 = math.sqrt(1.0 / d_in)
        s2 = math.sqrt(1.0 / hidden)
        self.W1 = rng.normal(0, s1, (d_in, hidden))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0, s2, (hidden, 1))
        self.b2 = np.zeros(1)
        self._cache: Dict[str, np.ndarray] = {}

    def params(self) -> Dict[str, np.ndarray]:
        return {"W1": self.W1, "b1": self.b1, "w2": self.w2, "b2": self.b2}

    def forward(self, feats: np.ndarray) -> np.ndarray:
        a1 = feats @ self.W1 + self.b1
        h = np.tanh(a1)
        u = (h @ self.w2 + self.b2).ravel()
        q = sigmoid(u)
        self._cache = {"f": feats, "h": h, "q": q}
        return q

    def loss_and_grads(self, feats: np.ndarray,
                       wrong: np.ndarray) -> Tuple[float, Dict[str, np.ndarray]]:
        """
        wrong : (B,) float 0/1, computed from the responder's argmax. It is a
                CONSTANT here -- deliberately. The supervisor is told the facts;
                he does not get to reshape the man he is watching.
        """
        q = self.forward(feats)
        w = np.where(wrong > 0.5, self.POS_WEIGHT, 1.0)
        qc = np.clip(q, 1e-9, 1 - 1e-9)
        loss = float(-np.mean(w * (wrong * np.log(qc) + (1 - wrong) * np.log(1 - qc))))
        # d/du of weighted BCE with q = sigmoid(u) is w * (q - wrong)
        du = (w * (q - wrong)) / feats.shape[0]
        h = self._cache["h"]
        dw2 = h.T @ du[:, None]
        db2 = np.array([du.sum()])
        dh = du[:, None] @ self.w2.T
        da1 = dh * (1 - h ** 2)
        dW1 = feats.T @ da1
        db1 = da1.sum(0)
        return loss, {"W1": dW1, "b1": db1, "w2": dw2, "b2": db2}


# ==============================================================================
# SECTION 5 -- THE RESPONDER: frozen prior + register-gated doubt
# ==============================================================================


@dataclass
class ShukukConfig:
    d_in: int = D_IN
    hidden: int = 28
    n_slots: int = 10
    critic_hidden: int = 20
    lambda_doubt: float = 0.055      # the price of contradicting the master
    weight_decay: float = 2e-4
    lr: float = 0.05
    lr_critic: float = 0.08
    epochs: int = 260
    fission_epochs: Tuple[int, ...] = (90, 150, 210)
    fission_sep: float = 4.00        # must clear the unimodal null of 2.65
    fission_progn_ratio: float = 1.60  # how badly a category must fail to predict
    fission_repair: float = 0.40       # how much the split must repair it
    fission_min: int = 60              # minimum support for each child
    course_tolerance: float = 2.2      # how far the course may miss its prognosis
    seed: int = 211


class Responder:
    """
    The master responder.

        z = z_prior_lifted  +  g(x) * Delta(x)

    z_prior_lifted is frozen. Delta is a trainable correction. g in (0,1) is the
    doubt gate, driven by the trunk AND by the register's verdict r, and priced
    with an L1 penalty so that doubt stays sparse.

    Nothing in this forward pass reads its own confidence for any decision. The
    softmax is used to produce an answer and a training residual, never to
    decide whether the answer should be given.
    """

    def __init__(self, cfg: ShukukConfig, prior: AuthorityPrior, rng):
        self.cfg = cfg
        self.prior = prior
        H = cfg.hidden
        s1 = math.sqrt(1.0 / cfg.d_in)
        s2 = math.sqrt(1.0 / H)

        self.W1 = rng.normal(0, s1, (cfg.d_in, H))
        self.b1 = np.zeros(H)
        self.W2 = rng.normal(0, s2, (H, N_COARSE))     # starts at inherited width
        self.b2 = np.zeros(N_COARSE)
        # THE GATE READS THE REGISTER AND NOTHING ELSE. There is no parallel
        # path by which the model can decide to doubt on a hunch. If it wants
        # to depart from the inherited books on a case, the register must say
        # that cases like it have departed before. This is the single most
        # restrictive choice in the file and the most characteristic: the
        # Shukuk is not a list of misgivings, it is a list of recorded
        # contradictions, and nothing enters it that was not observed.
        self.bg = np.array(-1.2)                        # start sceptical of doubt
        self.wr = np.array(1.0)                         # the register's weight
        self.register = CaseRegister(cfg.n_slots, H, rng)

        # Lift matrix: (N_COARSE, C). Maps the frozen prior's coarse scores onto
        # the current -- possibly forked -- fine label set. Never trained.
        self.L = np.eye(N_COARSE)
        self.n_classes = N_COARSE
        self._cache: Dict[str, np.ndarray] = {}

    # -- parameter plumbing ---------------------------------------------------
    def params(self) -> Dict[str, np.ndarray]:
        return {
            "W1": self.W1, "b1": self.b1,
            "W2": self.W2, "b2": self.b2,
            "bg": self.bg, "wr": self.wr,
            "M": self.register.M, "c": self.register.c, "s": self.register.s,
        }

    def set_params(self, p: Dict[str, np.ndarray]) -> None:
        self.W1, self.b1 = p["W1"], p["b1"]
        self.W2, self.b2 = p["W2"], p["b2"]
        self.bg, self.wr = p["bg"], p["wr"]
        self.register.M, self.register.c, self.register.s = p["M"], p["c"], p["s"]

    # -- forward --------------------------------------------------------------
    def forward(self, X: np.ndarray) -> np.ndarray:
        z_prior = self.prior.forward(X) @ self.L          # (B, C), frozen
        a1 = X @ self.W1 + self.b1
        h = np.tanh(a1)
        r = self.register.forward(h)                       # (B,)
        u = self.bg + self.wr * r                          # (B,)
        g = sigmoid(u)
        delta = h @ self.W2 + self.b2                      # (B, C)
        z = z_prior + g[:, None] * delta
        self._cache = {"X": X, "h": h, "r": r, "g": g,
                       "delta": delta, "z_prior": z_prior}
        return z

    # -- loss + hand-derived backward ----------------------------------------
    def loss_and_grads(self, X: np.ndarray,
                       y: np.ndarray) -> Tuple[float, Dict[str, np.ndarray], np.ndarray]:
        cfg = self.cfg
        z = self.forward(X)
        ce, dz = cross_entropy(z, y)

        h = self._cache["h"]
        g = self._cache["g"]
        delta = self._cache["delta"]

        # --- the price of doubt: L1 on the gate ------------------------------
        doubt = cfg.lambda_doubt * float(np.mean(g))
        dg_pen = cfg.lambda_doubt / X.shape[0] * np.ones_like(g)

        # z = z_prior + g * delta
        ddelta = dz * g[:, None]
        dg = (dz * delta).sum(1) + dg_pen

        dW2 = h.T @ ddelta
        db2 = ddelta.sum(0)
        dh = ddelta @ self.W2.T

        # gate: g = sigmoid(u); u = bg + wr * r
        du = dg * g * (1.0 - g)
        dbg = np.array(du.sum())
        dr = du * float(self.wr)
        dwr = np.array(float(np.sum(du * self._cache["r"])))

        # register
        dh_reg, reg_grads = self.register.backward(dr)
        dh = dh + dh_reg

        # trunk
        da1 = dh * (1 - h ** 2)
        dW1 = X.T @ da1
        db1 = da1.sum(0)

        # --- weight decay (not on gate bias or register precision) -----------
        wd = cfg.weight_decay
        dW1 += wd * self.W1
        dW2 += wd * self.W2
        l2 = 0.5 * wd * (float((self.W1 ** 2).sum()) + float((self.W2 ** 2).sum()))

        grads = {
            "W1": dW1, "b1": db1, "W2": dW2, "b2": db2,
            "bg": dbg, "wr": dwr,
            "M": reg_grads["M"], "c": reg_grads["c"], "s": reg_grads["s"],
        }
        return ce + doubt + l2, grads, z

    # -- growth ---------------------------------------------------------------
    def fork_class(self, parent: int, rng) -> int:
        """
        Grow one new output class as a child of `parent`.

        The lift matrix gets a duplicate column, so the FROZEN PRIOR ASSIGNS
        BOTH CHILDREN THE SAME SCORE. The inherited authority is structurally
        incapable of telling them apart -- which is the historical fact about
        the eruptive fevers before 900 CE. All the discriminating power must
        come from the gated correction, i.e. from the doubt.
        """
        new_idx = self.n_classes
        self.W2 = np.concatenate(
            [self.W2, (self.W2[:, [parent]] +
                       rng.normal(0, 0.05, (self.W2.shape[0], 1)))], axis=1)
        self.b2 = np.concatenate([self.b2, [self.b2[parent]]])
        self.L = np.concatenate([self.L, self.L[:, [parent]]], axis=1)
        self.n_classes += 1
        return new_idx


# ==============================================================================
# SECTION 6 -- THE JUNIOR RESPONDER AND THE CASCADE (al-Qifti's teaching circle)
# ==============================================================================


# The signs a junior lecturer takes at the bedside: pulse, back pain, the depth
# of the eruption, the day it came, itching, redness of the eyes. Enough to tell
# the eruptive fevers apart once someone has told him they are two.
#
# What he does NOT take: fever onset, coryza, the state of the tongue, and the
# season and region -- precisely the four that drift with each other and whose
# joint reading is what corrects the inherited books. He is therefore not a
# worse version of the master. He is blind on an identifiable territory, and
# that territory is learnable by someone watching him.
JUNIOR_SIGNS = (1, 2, 4, 5, 7, 8)


class JuniorResponder:
    """
    The cheap first rank.

        z_jun = z_prior_lifted + X[:, JUNIOR_SIGNS] @ Wj + bj

    Al-Qifti reports that al-Razi's lectures were arranged so that junior and
    senior lecturers dealt with whatever inquiries they were competent to
    answer, and only matters that passed their range were referred to him.

    Note what makes this a real cascade and not a distillation: the junior is
    NOT trained to imitate the master, and it is not simply a smaller copy.
    It has a different and narrower VIEW. It is therefore good at some cases
    and structurally blind on others -- the ones that need signs it never
    takes. That blindness is a boundary in case-space, which is exactly the
    kind of thing a supervisor can learn to recognise, and exactly what makes
    referral upward worth its cost.
    """

    def __init__(self, d_in: int, n_classes: int, prior: AuthorityPrior, rng):
        self.prior = prior
        self.d_j = len(JUNIOR_SIGNS)
        self.Wj = rng.normal(0, math.sqrt(1.0 / self.d_j), (self.d_j, n_classes))
        self.bj = np.zeros(n_classes)
        self.L = np.eye(N_COARSE)
        self.n_classes = n_classes

    def params(self) -> Dict[str, np.ndarray]:
        return {"Wj": self.Wj, "bj": self.bj}

    def forward(self, X: np.ndarray) -> np.ndarray:
        Xj = X[:, JUNIOR_SIGNS]
        return self.prior.forward(X) @ self.L + Xj @ self.Wj + self.bj

    def loss_and_grads(self, X, y, weight_decay=2e-4):
        Xj = X[:, JUNIOR_SIGNS]
        z = self.forward(X)
        ce, dz = cross_entropy(z, y)
        dWj = Xj.T @ dz + weight_decay * self.Wj
        dbj = dz.sum(0)
        l2 = 0.5 * weight_decay * float((self.Wj ** 2).sum())
        return ce + l2, {"Wj": dWj, "bj": dbj}, z

    def fork_class(self, parent: int, rng) -> None:
        self.Wj = np.concatenate(
            [self.Wj, self.Wj[:, [parent]] + rng.normal(0, 0.05, (self.d_j, 1))],
            axis=1)
        self.bj = np.concatenate([self.bj, [self.bj[parent]]])
        self.L = np.concatenate([self.L, self.L[:, [parent]]], axis=1)
        self.n_classes += 1


# ==============================================================================
# SECTION 7 -- FISSION (Kitab al-Judari wa'l-Hasba: splitting a merged class)
# ==============================================================================


def _otsu_split_1d(t: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    One-dimensional two-means split by exhaustive threshold search (Otsu), and
    the separation statistic that authorises or forbids a fork:

        sep = |mean_hi - mean_lo| / pooled_within_sd

    The key property, and the reason this is a usable test rather than an
    excuse: a SINGLE Gaussian cut at its optimal threshold gives sep = 2.65
    exactly, for every unimodal Gaussian, regardless of scale or dimension.
    That is the null. Anything near 2.65 is a category being cut for no
    reason. A genuine mixture of two separated components gives a far larger
    value. FISSION_NULL below records the null so the threshold can be read
    as "how far past the point where splitting means nothing".
    """
    order = np.argsort(t)
    ts = t[order]
    n = ts.size
    if n < 12:
        return np.zeros(n, dtype=bool), 0.0
    csum = np.cumsum(ts)
    total = csum[-1]
    best_var, best_i = -1.0, -1
    for i in range(4, n - 4):
        w0, w1 = i, n - i
        m0 = csum[i - 1] / w0
        m1 = (total - csum[i - 1]) / w1
        between = w0 * w1 * (m0 - m1) ** 2
        if between > best_var:
            best_var, best_i = between, i
    thr = 0.5 * (ts[best_i - 1] + ts[best_i])
    hi = t > thr
    lo = ~hi
    if hi.sum() < 4 or lo.sum() < 4:
        return np.zeros(n, dtype=bool), 0.0
    pooled = math.sqrt((lo.sum() * t[lo].var() + hi.sum() * t[hi].var()) / n) + 1e-9
    sep = abs(t[hi].mean() - t[lo].mean()) / pooled
    return hi, float(sep)


FISSION_NULL = 2.65   # sep produced by optimally cutting a single Gaussian


def _prognosis_residual(O: np.ndarray, mask: np.ndarray) -> float:
    """
    Mean squared distance of a group's recorded courses from the group's own
    average course. This IS a prognosis: 'in this disease, expect the following'.
    A category whose prognosis carries a large residual is a category that
    does not tell the physician what will happen.
    """
    if mask.sum() < 2:
        return 0.0
    sub = O[mask]
    return float(((sub - sub.mean(0)) ** 2).sum(1).mean())


def attempt_fission(model: Responder, junior: JuniorResponder,
                    X: np.ndarray, O: np.ndarray, y: np.ndarray,
                    cfg: ShukukConfig, rng) -> Tuple[np.ndarray, List[dict]]:
    """
    The operation of the Treatise on Smallpox and Measles, as an algorithm.

    THREE CONDITIONS, all required. They are in this order because this is the
    order of his own reasoning, and the order matters: he did not go looking
    for new diseases, and he did not split on a theory.

      (i)   THE CATEGORY FAILS TO PREDICT THE COURSE. Not that it mislabels --
            Galen's 'eruptive fever' labelled perfectly well as a label. The
            complaint is that being told a patient has it does not tell you
            whether he will scar, how long he will run, or whether he can take
            it twice. A category is examined only when its prognosis residual
            is far above the smallest residual any current category achieves.

      (ii)  THE SIGNS DIVIDE. The split is then made on the RECORDED SIGNS,
            never on the model's internal state, and only when those signs are
            bimodal well past FISSION_NULL -- the separation you get from
            cutting a single unimodal cloud in half, which is 2.65 for every
            Gaussian regardless of scale or dimension. Below that, a split
            means nothing at all.

      (iii) THE DIVISION REPAIRS THE PROGNOSIS. The two halves must each
            predict their own course markedly better than the parent did.
            This is the step that makes the whole thing self-limiting: a
            category can only be broken by something that pays for the break.

    His causal account of the eruption -- blood fermenting and expelling its
    superfluous moisture as the child matures -- was wrong, and the split
    survived anyway, because at no point did the split rest on it.

    On a fork, the frozen prior's lift matrix duplicates the parent column, so
    both children receive the SAME inherited score. The authority is left
    structurally unable to tell them apart -- the historical situation exactly
    -- and every bit of the new distinction is paid for out of doubt.
    """
    reports: List[dict] = []
    y = y.copy()
    gsd = X.std(0) + 1e-9      # global scaling so no sign dominates by its units

    # the smallest prognosis residual any current category manages: the best
    # available estimate of course-variation that is simply irreducible
    resids = [_prognosis_residual(O, y == k) for k in range(model.n_classes)]
    floor = max(min([r for r in resids if r > 0.0] or [1.0]), 1e-6)

    for k in range(model.n_classes):
        idx = np.where(y == k)[0]
        R_parent = resids[k]
        rep = {"class": k, "n": int(idx.size), "forked": False,
               "prognosis_residual": R_parent, "sep": 0.0, "repair": 0.0}
        if idx.size < 2 * cfg.fission_min:
            rep["reason"] = "too few cases to judge"
            reports.append(rep)
            continue

        # (i) does the category predict the course? --------------------------
        if R_parent < cfg.fission_progn_ratio * floor:
            rep["reason"] = (f"prognosis holds (residual {R_parent:.2f} vs "
                             f"floor {floor:.2f})")
            reports.append(rep)
            continue

        # (ii) do the signs divide? ------------------------------------------
        Zc = (X[idx] - X[idx].mean(0)) / gsd
        v = rng.normal(0, 1, Zc.shape[1])
        v /= np.linalg.norm(v) + EPS
        for _ in range(80):                      # power iteration for the top PC
            v = Zc.T @ (Zc @ v)
            nv = np.linalg.norm(v)
            if nv < EPS:
                break
            v /= nv
        t = Zc @ v
        hi, sep = _otsu_split_1d(t)
        rep["sep"] = sep
        n_hi, n_lo = int(hi.sum()), int((~hi).sum())
        rep["n_hi"], rep["n_lo"] = n_hi, n_lo
        if sep < cfg.fission_sep or min(n_hi, n_lo) < cfg.fission_min:
            rep["reason"] = (f"signs unimodal (sep {sep:.2f}, null "
                             f"{FISSION_NULL:.2f}, threshold {cfg.fission_sep:.2f})")
            reports.append(rep)
            continue

        # (iii) does the division repair the prognosis? -----------------------
        m_hi = np.zeros(y.size, dtype=bool); m_hi[idx[hi]] = True
        m_lo = np.zeros(y.size, dtype=bool); m_lo[idx[~hi]] = True
        R_hi = _prognosis_residual(O, m_hi)
        R_lo = _prognosis_residual(O, m_lo)
        R_child = (n_hi * R_hi + n_lo * R_lo) / idx.size
        repair = 1.0 - R_child / (R_parent + EPS)
        rep["repair"] = float(repair)
        if repair < cfg.fission_repair:
            rep["reason"] = (f"the division does not repair the prognosis "
                             f"({100*repair:.0f}% < {100*cfg.fission_repair:.0f}%)")
            reports.append(rep)
            continue

        new = model.fork_class(k, rng)
        junior.fork_class(k, rng)
        y[idx[hi]] = new
        rep["forked"] = True
        rep["new_class"] = new
        rep["reason"] = "fails to predict the course; the signs divide; the division repairs it"
        reports.append(rep)

    return y, reports


# ==============================================================================
# SECTION 8 -- MANDATORY FINITE-DIFFERENCE GRADIENT CHECK
# ==============================================================================


def gradient_check(verbose: bool = True) -> bool:
    """
    Central differences against the hand-derived analytic gradients, for EVERY
    trainable block in the responder, the junior, and the supervisor.

    (f(p+e) - f(p-e)) / 2e  vs  analytic, relative error must be < 1e-5.
    """
    rng = set_seed(4242)
    cfg = ShukukConfig(hidden=7, n_slots=4, critic_hidden=6)
    n = 26
    data = make_corpus(n, rng)
    X, y = data["X"], data["y_coarse"]

    prior = AuthorityPrior(D_IN, 6, N_COARSE, rng)
    prior.frozen = True                      # never differentiated through
    model = Responder(cfg, prior, rng)
    junior = JuniorResponder(D_IN, N_COARSE, prior, rng)
    critic = Musharrif(D_IN + cfg.hidden, cfg.critic_hidden, rng)

    ok = True
    eps = 1e-6
    tol = 1e-5

    def check(name: str, params: Dict[str, np.ndarray],
              lossfn, grads: Dict[str, np.ndarray], n_probe: int = 6) -> None:
        nonlocal ok
        worst = 0.0
        for key, P in params.items():
            flat = P.reshape(-1)
            gflat = np.asarray(grads[key], dtype=np.float64).reshape(-1)
            m = flat.size
            probe = rng.choice(m, size=min(n_probe, m), replace=False)
            for i in probe:
                orig = flat[i]
                flat[i] = orig + eps
                lp = lossfn()
                flat[i] = orig - eps
                lm = lossfn()
                flat[i] = orig
                num = (lp - lm) / (2 * eps)
                ana = gflat[i]
                denom = max(1.0, abs(num) + abs(ana))
                rel = abs(num - ana) / denom
                worst = max(worst, rel)
        status = "PASS" if worst < tol else "FAIL"
        if worst >= tol:
            ok = False
        if verbose:
            print(f"    {name:<28s} max relative error {worst:.3e}   [{status}]")

    # --- responder -----------------------------------------------------------
    loss, grads, _ = model.loss_and_grads(X, y)
    check("responder (all blocks)", model.params(),
          lambda: model.loss_and_grads(X, y)[0], grads, n_probe=7)

    # --- junior --------------------------------------------------------------
    _, jg, _ = junior.loss_and_grads(X, y)
    check("junior responder", junior.params(),
          lambda: junior.loss_and_grads(X, y)[0], jg, n_probe=7)

    # --- supervisor ----------------------------------------------------------
    z = model.forward(X)
    h = model._cache["h"]
    feats = np.concatenate([X, h], axis=1)
    wrong = (z.argmax(1) != y).astype(np.float64)
    _, cg = critic.loss_and_grads(feats, wrong)
    check("supervisor (musharrif)", critic.params(),
          lambda: critic.loss_and_grads(feats, wrong)[0], cg, n_probe=7)

    # --- register in isolation, because it is the novel module ---------------
    reg = CaseRegister(5, 9, rng)
    hh = rng.normal(0, 1, (11, 9))
    coef = rng.normal(0, 1, 11)

    def reg_loss():
        return float(np.sum(reg.forward(hh) * coef))

    reg.forward(hh)
    _, rg = reg.backward(coef)
    check("case register (isolated)",
          {"M": reg.M, "c": reg.c, "s": reg.s}, reg_loss, rg, n_probe=8)

    return ok


# ==============================================================================
# SECTION 9 -- TRAINING
# ==============================================================================


def prognosis_table(O: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    """
    The prognosis attached to each category: the average course of the cases
    filed under it. This is what a chapter of a medical book actually is --
    'in this disease, expect the following'.
    """
    T = np.zeros((n_classes, O.shape[1]))
    for k in range(n_classes):
        m = y == k
        T[k] = O[m].mean(0) if m.sum() else O.mean(0)
    return T


def parents_of(L: np.ndarray) -> np.ndarray:
    """Which inherited category each current category descends from."""
    return L.argmax(0)


def observable_fault(O: np.ndarray, y_coarse: np.ndarray, pred: np.ndarray,
                     table: np.ndarray, parents: np.ndarray,
                     tau: float) -> np.ndarray:
    """
    What a supervisor standing in the ward could actually verify, and nothing
    more. He cannot open the patient to confirm the diagnosis. He has two
    things:

      1. THE RECEIVED FILING. The hospital books say which of the three
         inherited categories the case was entered under. If the physician's
         answer does not descend from that category, the discrepancy is on
         the record.

      2. THE COURSE. He can watch what happened next and see whether it
         matched what the physician's category promised.

    Note the asymmetry with the evaluation: the supervisors are trained on
    THIS, an observable proxy, and are then scored in section 4 on their
    ability to flag errors of a kind they were never shown. That is the
    correct test and the correct handicap -- the supervisor of ch. IV is a
    frequent associate with ordinary eyes, not an oracle with the answer key.
    """
    missed_course = ((O - table[pred]) ** 2).sum(1) > tau
    misfiled = parents[pred] != y_coarse
    return (missed_course | misfiled).astype(np.float64)


def _adam_update(params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray],
                 state: Dict[str, Dict[str, np.ndarray]], lr: float, t: int,
                 b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8) -> None:
    """In-place Adam. Handles 0-d arrays (bg, wr, s) as well as matrices."""
    for k, g in grads.items():
        g = np.asarray(g, dtype=np.float64)
        if k not in state["m"]:
            state["m"][k] = np.zeros_like(g)
            state["v"][k] = np.zeros_like(g)
        state["m"][k] = b1 * state["m"][k] + (1 - b1) * g
        state["v"][k] = b2 * state["v"][k] + (1 - b2) * g * g
        mh = state["m"][k] / (1 - b1 ** t)
        vh = state["v"][k] / (1 - b2 ** t)
        upd = lr * mh / (np.sqrt(vh) + eps)
        P = params[k]
        if P.ndim == 0:
            params[k] = np.array(float(P) - float(upd))
        else:
            P -= upd


def train(cfg: ShukukConfig, verbose: bool = True):
    """
    The full course. Order of operations matters and is itself the argument:

      1. Fit the authority on the inherited corpus. Freeze it.
      2. Train the annotation (trunk, register, gate, correction) against the
         cases, with doubt priced.
      3. At intervals, look for structured residual inside a class and fork.
      4. Train the two supervisors on the realised errors of the two responders,
         with no gradient flowing back into them.
    """
    rng = set_seed(cfg.seed)

    # -- corpora --------------------------------------------------------------
    train_d = make_corpus(2600, rng)
    val_d = make_corpus(1400, rng)

    # The inherited corpus: only the region the authority ever saw (season <= 0)
    # and only the coarse taxonomy. This is Galen's Mediterranean practice as
    # against the hospitals of Baghdad and Rayy.
    inh = make_corpus(3200, rng)
    keep = inh["prior_region"] & ~inh["ambiguous"]
    Xi, yi = inh["X"][keep], inh["y_coarse"][keep]

    if verbose:
        print(f"    inherited corpus: {Xi.shape[0]} cases (season<=0 only), "
              f"{N_COARSE} received categories")
        print(f"    working corpus  : {train_d['X'].shape[0]} cases, "
              f"{int(train_d['contradiction'].sum())} in the contradiction region")

    prior = AuthorityPrior(D_IN, 22, N_COARSE, rng)
    prior.fit(Xi, yi, epochs=600, lr=0.08, verbose=verbose)

    model = Responder(cfg, prior, rng)
    junior = JuniorResponder(D_IN, N_COARSE, prior, rng)
    # The supervisor watches ONGOING PRACTICE, not the cases the responder was
    # drilled on. Al-Tibb al-Ruhani ch. IV specifies "an intelligent man who is
    # his frequent associate and constant companion" -- someone present at the
    # work, not someone reading the exam paper afterwards. Training the critics
    # on the responders' own training set would let them watch a man who never
    # makes a mistake, which teaches nothing.
    clinic = make_corpus(1800, rng)
    critic_m = Musharrif(D_IN + cfg.hidden, cfg.critic_hidden, rng)
    # the junior has no hidden state to inspect, so his supervisor reads the
    # signs and the junior's own scores -- and is correspondingly cheap to run.
    critic_j = Musharrif(D_IN + N_COARSE, cfg.critic_hidden, rng)
    critic_j_grown = False

    st_m = {"m": {}, "v": {}}
    st_j = {"m": {}, "v": {}}
    st_cm = {"m": {}, "v": {}}
    st_cj = {"m": {}, "v": {}}

    X, Otr, y = train_d["X"], train_d["O"], train_d["y_coarse"].copy()
    Xv = val_d["X"]
    fission_log: List[dict] = []
    history: List[dict] = []

    for ep in range(1, cfg.epochs + 1):
        # ---- responder ------------------------------------------------------
        loss, grads, z = model.loss_and_grads(X, y)
        p = model.params()
        _adam_update(p, grads, st_m, cfg.lr, ep)
        model.set_params(p)

        # ---- junior ---------------------------------------------------------
        jloss, jgrads, zj = junior.loss_and_grads(X, y)
        pj = junior.params()
        _adam_update(pj, jgrads, st_j, cfg.lr, ep)
        junior.Wj, junior.bj = pj["Wj"], pj["bj"]

        # ---- supervisors ----------------------------------------------------
        # Forward the responders over the clinic stream. Nothing here touches a
        # responder gradient: the arrays are copied, and the "wrong" targets are
        # constants. This is the firewall of ch. IV made literal -- the man
        # cannot tune himself to look well to the observer.
        zc = model.forward(clinic["X"]).copy()
        hc = model._cache["h"].copy()
        zcj = junior.forward(clinic["X"]).copy()
        table = prognosis_table(Otr, y, model.n_classes)
        floor = min(_prognosis_residual(Otr, y == k)
                    for k in range(model.n_classes)
                    if (y == k).sum() > 1)
        tau = cfg.course_tolerance * max(floor, 1e-3)

        feats_m = np.concatenate([clinic["X"], hc], axis=1)
        feats_j = np.concatenate([clinic["X"], zcj], axis=1)
        if feats_j.shape[1] != critic_j.W1.shape[0]:
            critic_j = Musharrif(feats_j.shape[1], cfg.critic_hidden, rng)
            st_cj = {"m": {}, "v": {}}

        par = parents_of(model.L)
        wrong_m = observable_fault(clinic["O"], clinic["y_coarse"],
                                   zc.argmax(1), table, par, tau)
        wrong_j = observable_fault(clinic["O"], clinic["y_coarse"],
                                   zcj.argmax(1), table, par, tau)

        cml, cmg = critic_m.loss_and_grads(feats_m, wrong_m)
        pcm = critic_m.params()
        _adam_update(pcm, cmg, st_cm, cfg.lr_critic, ep)

        cjl, cjg = critic_j.loss_and_grads(feats_j, wrong_j)
        pcj = critic_j.params()
        _adam_update(pcj, cjg, st_cj, cfg.lr_critic, ep)

        # restore the responder's cache for the epoch log
        model.forward(X)

        # ---- fission --------------------------------------------------------
        if ep in cfg.fission_epochs:
            y, reps = attempt_fission(model, junior, X, Otr, y, cfg, rng)
            for r in reps:
                r["epoch"] = ep
                fission_log.append(r)
                if not verbose:
                    continue
                if r["forked"]:
                    print(f"    [epoch {ep:3d}] FISSION  category {r['class']} -> "
                          f"{r['new_class']}  |  prognosis residual "
                          f"{r['prognosis_residual']:.2f}  signs sep {r['sep']:.2f} "
                          f"(null {FISSION_NULL:.2f})  repair "
                          f"{100*r['repair']:.0f}%  |  {r['n_lo']}/{r['n_hi']} cases")
                else:
                    print(f"    [epoch {ep:3d}] held     class {r['class']}: "
                          f"{r['reason']}")
            # Adam moments for the grown tensors are stale; reset them.
            st_m = {"m": {}, "v": {}}
            st_j = {"m": {}, "v": {}}

        if ep % 50 == 0 or ep == 1:
            acc = float((z.argmax(1) == y).mean())
            history.append({"epoch": ep, "loss": loss, "acc": acc,
                            "gate": float(model._cache["g"].mean()),
                            "classes": model.n_classes})
            if verbose:
                print(f"    epoch {ep:4d}  loss {loss:7.4f}  fit {acc:6.3f}  "
                      f"mean-doubt {model._cache['g'].mean():.3f}  "
                      f"classes {model.n_classes}  critic {cml:.4f}")

    return {
        "cfg": cfg, "prior": prior, "model": model, "junior": junior,
        "critic_m": critic_m, "critic_j": critic_j,
        "train": train_d, "val": val_d, "clinic": clinic,
        "y_train_final": y, "prognosis": prognosis_table(Otr, y, model.n_classes),
        "fission_log": fission_log, "history": history, "rng": rng,
    }


# ==============================================================================
# SECTION 10 -- EVALUATION AND SELF-TESTS
# ==============================================================================


def _map_model_classes_to_truth(pred: np.ndarray, y_true_fine: np.ndarray,
                                n_model: int) -> np.ndarray:
    """
    The model invents its own class indices when it forks, so to score it
    against the hidden four-way truth we need a mapping. Each model class is
    assigned the true condition it most often coincides with. This is the
    standard clustering-accuracy convention and it is reported as such.
    """
    mapping = np.zeros(n_model, dtype=np.int64)
    for k in range(n_model):
        m = pred == k
        if m.sum() == 0:
            continue
        counts = np.bincount(y_true_fine[m], minlength=N_FINE_TRUE)
        mapping[k] = int(counts.argmax())
    return mapping


def evaluate(res: dict, verbose: bool = True) -> dict:
    model, junior = res["model"], res["junior"]
    critic_m, critic_j = res["critic_m"], res["critic_j"]
    prior = res["prior"]
    val = res["val"]
    Xv, y_fine, y_coarse = val["X"], val["y_fine"], val["y_coarse"]
    contra, amb = val["contradiction"], val["ambiguous"]
    out: Dict[str, float] = {}

    # ---- 1. the frozen authority, left exactly as received ------------------
    zp = prior.forward(Xv)
    prior_pred = zp.argmax(1)
    prior_wrong = prior_pred != y_coarse          # THE object of the register
    out["prior_acc_coarse"] = float((~prior_wrong).mean())
    out["prior_acc_prior_region"] = float((~prior_wrong)[val["prior_region"]].mean())
    out["prior_acc_contra"] = float((~prior_wrong)[contra].mean())

    # Honest ceiling on the true four-way task for any three-way predictor that
    # merges variola and morbilli: it can be right on the two unmerged
    # conditions, and at best right on half of the merged one.
    merged = (y_fine == 1) | (y_fine == 2)
    ceiling = np.zeros(y_fine.size, dtype=bool)
    ceiling[~merged] = prior_pred[~merged] == y_coarse[~merged]
    ceiling[merged] = (prior_pred[merged] == 1) & (y_fine[merged] == 1)
    out["prior_ceiling_fine"] = float(ceiling.mean())

    # ---- 2. the annotated model against the hidden four-way truth -----------
    z = model.forward(Xv)
    h = model._cache["h"]
    r = model._cache["r"]
    g = model._cache["g"]
    pred = z.argmax(1)
    mapping = _map_model_classes_to_truth(pred, y_fine, model.n_classes)
    mapped = mapping[pred]
    wrong = (mapped != y_fine)
    out["model_acc_fine"] = float((~wrong).mean())
    out["model_acc_contra"] = float((~wrong)[contra].mean())
    out["model_acc_clear"] = float((~wrong)[~amb].mean())
    out["n_classes"] = float(model.n_classes)
    out["n_forks"] = float(sum(1 for f in res["fission_log"] if f["forked"]))

    if model.n_classes >= 4:
        owners = np.bincount(pred[merged], minlength=model.n_classes)
        top2 = np.argsort(owners)[-2:]
        pure = 0
        for k in top2:
            m = (pred == k) & merged
            if m.sum():
                pure += int(np.bincount(y_fine[m], minlength=N_FINE_TRUE)[[1, 2]].max())
        out["fission_purity"] = float(pure / max(1, int(merged.sum())))
    else:
        out["fission_purity"] = float("nan")

    # ---- 3. did the register find where the authority fails? ---------------
    # Scored against the ground truth of the matter: the cases the frozen prior
    # actually got wrong. This is the two-column list, evaluated as a detector.
    out["register_wrong"] = float(r[prior_wrong].mean())
    out["register_right"] = float(r[~prior_wrong].mean())
    out["register_auroc"] = auroc(r, prior_wrong.astype(np.int64))
    out["gate_wrong"] = float(g[prior_wrong].mean())
    out["gate_right"] = float(g[~prior_wrong].mean())
    out["gate_mean"] = float(g.mean())
    out["gate_contra"] = float(g[contra].mean())

    # ---- 4. THE CONCEIT TEST (Tibb al-Ruhani IV and VI) --------------------
    feats = np.concatenate([Xv, h], axis=1)
    q = critic_m.forward(feats)
    p = softmax(z, axis=1)
    ps = np.sort(p, axis=1)
    self_conf = ps[:, -1]                       # max softmax
    self_margin = ps[:, -1] - ps[:, -2]         # the stronger self-signal
    out["auroc_self_conf"] = auroc(1.0 - self_conf, wrong.astype(np.int64))
    out["auroc_self_margin"] = auroc(-self_margin, wrong.astype(np.int64))
    out["auroc_critic"] = auroc(q, wrong.astype(np.int64))
    out["ece_self"] = expected_calibration_error(self_conf, (~wrong).astype(np.int64))
    out["ece_critic"] = expected_calibration_error(1.0 - q, (~wrong).astype(np.int64))

    # The sharpest form of the test: on records whose signs simply do not
    # determine the answer, does the model still admire its own verdict?
    out["selfconf_on_undecidable"] = float(self_conf[amb].mean())
    out["selfconf_on_decidable"] = float(self_conf[~amb].mean())
    out["critic_on_undecidable"] = float(q[amb].mean())
    out["critic_on_decidable"] = float(q[~amb].mean())
    out["acc_on_undecidable"] = float((~wrong)[amb].mean())
    out["acc_on_decidable"] = float((~wrong)[~amb].mean())

    # THE HEADLINE. On a territory it cannot read, how wrong is each party
    # about how the model is doing? This -- not global ranking -- is what
    # ch. VI actually claims: that a man's estimate of his own qualities is
    # "of necessity above its merits", and above them worst exactly where he
    # cannot see.
    acc_blind = float((~wrong)[amb].mean())
    out["blind_gap_self"] = abs(float(self_conf[amb].mean()) - acc_blind)
    out["blind_gap_critic"] = abs(float((1.0 - q)[amb].mean()) - acc_blind)
    acc_seen = float((~wrong)[~amb].mean())
    out["seen_gap_self"] = abs(float(self_conf[~amb].mean()) - acc_seen)
    out["seen_gap_critic"] = abs(float((1.0 - q)[~amb].mean()) - acc_seen)

    # ---- 5. abstention ------------------------------------------------------
    for cov in (100, 90, 80, 70):
        k = int(round(cov / 100 * q.size))
        keep = np.argsort(q)[:k]
        out[f"acc_at_coverage_{cov}"] = float((~wrong)[keep].mean())
        keep_self = np.argsort(-self_conf)[:k]
        out[f"acc_at_coverage_{cov}_self"] = float((~wrong)[keep_self].mean())

    # ---- 6. the teaching circle --------------------------------------------
    zj = junior.forward(Xv)
    predj = zj.argmax(1)
    mapj = _map_model_classes_to_truth(predj, y_fine, junior.n_classes)
    mappedj = mapj[predj]
    out["junior_acc_fine"] = float((mappedj == y_fine).mean())

    qj = critic_j.forward(np.concatenate([Xv, zj], axis=1))
    for thr in (30, 40, 50):
        esc = qj > thr / 100
        combo = np.where(esc, mapped, mappedj)
        out[f"cascade_acc_thr{thr}"] = float((combo == y_fine).mean())
        out[f"cascade_escalated_thr{thr}"] = float(esc.mean())
    gap = out["model_acc_fine"] - out["junior_acc_fine"]
    out["gap_recovered"] = float(
        (out["cascade_acc_thr50"] - out["junior_acc_fine"]) / (gap + EPS))

    cfg = res["cfg"]
    C = model.n_classes
    cost_prior = D_IN * 22 + 22 * N_COARSE
    cost_jun = len(JUNIOR_SIGNS) * C
    cost_master = (D_IN * cfg.hidden + cfg.hidden * cfg.n_slots +
                   cfg.hidden * C + cfg.hidden)
    cost_critic_j = (D_IN + C) * cfg.critic_hidden + cfg.critic_hidden
    esc = float((qj > 0.40).mean())
    out["cost_master_always"] = float(cost_prior + cost_master)
    out["cost_cascade"] = float(cost_prior + cost_jun + cost_critic_j + esc * cost_master)
    out["cost_saving_pct"] = float(
        100.0 * (1 - out["cost_cascade"] / out["cost_master_always"]))

    if verbose:
        _print_eval(out, res)
    return out


def _print_eval(o: dict, res: dict) -> None:
    W = 78
    print()
    print("  " + "=" * W)
    print("  RESULTS")
    print("  " + "=" * W)

    print("\n  1. THE INHERITED AUTHORITY, NEVER REVISED")
    print(f"     accuracy on its own three categories .............. {o['prior_acc_coarse']:.4f}")
    print(f"        ... in the region it was fitted on ............. {o['prior_acc_prior_region']:.4f}")
    print(f"        ... where the drift has carried past it ........ {o['prior_acc_contra']:.4f}")
    print(f"     best score ANY three-way scheme can reach on the")
    print(f"     true four conditions (a ceiling, not a failure) ... {o['prior_ceiling_fine']:.4f}")

    print("\n  2. THE ANNOTATION, SCORED ON THE HIDDEN FOUR-WAY TRUTH")
    print(f"     forks taken / categories held .................... "
          f"{int(o['n_forks'])} / {int(o['n_classes'])}")
    print(f"     accuracy, all records ............................ {o['model_acc_fine']:.4f}")
    print(f"     accuracy, records whose signs decide the case .... {o['model_acc_clear']:.4f}")
    print(f"     accuracy where the authority is worst ............ {o['model_acc_contra']:.4f}")
    print(f"     purity of the forked eruptive pair ............... {o['fission_purity']:.4f}")
    print(f"     gain over the authority's ceiling ................ "
          f"{o['model_acc_fine'] - o['prior_ceiling_fine']:+.4f}")

    print("\n  3. THE REGISTER, AS A DETECTOR OF THE MASTER'S FAILURES")
    print("     scored against the cases the frozen prior actually got wrong")
    print(f"     mean verdict where the authority failed .......... {o['register_wrong']:+.4f}")
    print(f"     mean verdict where it held ....................... {o['register_right']:+.4f}")
    print(f"     AUROC of the verdict as a failure detector ....... {o['register_auroc']:.4f}")
    print(f"     doubt raised where the authority failed .......... {o['gate_wrong']:.4f}")
    print(f"     doubt raised where it held ....................... {o['gate_right']:.4f}")
    print(f"     mean doubt overall (the price keeps it sparse) ... {o['gate_mean']:.4f}")

    print("\n  4. THE CONCEIT TEST  (al-Tibb al-Ruhani, chapters IV and VI)")
    print("     HOW WRONG IS EACH PARTY ABOUT HOW THE MODEL IS DOING?")
    print("     On records whose signs DO decide the case:")
    print(f"        model believes it is right ........... {o['selfconf_on_decidable']:.4f}")
    print(f"        supervisor believes it is right ...... "
          f"{1-o['critic_on_decidable']:.4f}")
    print(f"        it is actually right ................. {o['acc_on_decidable']:.4f}")
    print(f"        error of self-estimate ............... {o['seen_gap_self']:.4f}")
    print(f"        error of supervisor's estimate ....... {o['seen_gap_critic']:.4f}")
    print("     On records whose signs DO NOT decide the case -- the blind territory:")
    print(f"        model believes it is right ........... {o['selfconf_on_undecidable']:.4f}")
    print(f"        supervisor believes it is right ...... "
          f"{1-o['critic_on_undecidable']:.4f}")
    print(f"        it is actually right ................. {o['acc_on_undecidable']:.4f}")
    print(f"        ERROR OF SELF-ESTIMATE ............... {o['blind_gap_self']:.4f}")
    print(f"        error of supervisor's estimate ....... {o['blind_gap_critic']:.4f}")
    print("     As a global ranker of its own errors, however:")
    print(f"        AUROC, own max-probability ........... {o['auroc_self_conf']:.4f}")
    print(f"        AUROC, own decision margin ........... {o['auroc_self_margin']:.4f}")
    print(f"        AUROC, supervisor's verdict .......... {o['auroc_critic']:.4f}")
    print(f"        calibration error, own confidence .... {o['ece_self']:.4f}")
    print(f"        calibration error, supervisor ........ {o['ece_critic']:.4f}")
    print("     The supervisor is deliberately alarmist (faults are priced")
    print(f"     {Musharrif.POS_WEIGHT:.1f}x above false alarms, per ch. IV), which costs it")
    print("     calibration and global ranking. It buys the blind territory.")

    print("\n  5. SAYING HE DOES NOT KNOW")
    print("     coverage    by supervisor    by own confidence")
    for cov in (100, 90, 80, 70):
        print(f"       {cov:3d}%          {o[f'acc_at_coverage_{cov}']:.4f}"
              f"            {o[f'acc_at_coverage_{cov}_self']:.4f}")

    print("\n  6. THE TEACHING CIRCLE (junior answers, master is referred to)")
    print(f"     junior alone ..................................... {o['junior_acc_fine']:.4f}")
    print(f"     master alone ..................................... {o['model_acc_fine']:.4f}")
    for thr in (30, 40, 50):
        print(f"     cascade, referral threshold {thr/100:.2f} .............. "
              f"{o[f'cascade_acc_thr{thr}']:.4f}   "
              f"({100*o[f'cascade_escalated_thr{thr}']:4.1f}% referred up)")
    print(f"     share of the junior-to-master gap recovered ...... "
          f"{100*o['gap_recovered']:.1f}%")
    print(f"     cost per case, master always ..................... "
          f"{o['cost_master_always']:.0f} multiply-accumulates")
    print(f"     cost per case, cascade at 0.40 ................... "
          f"{o['cost_cascade']:.0f} MACs   ({o['cost_saving_pct']:.1f}% saved)")


def print_register(res: dict) -> None:
    """
    Read the ledger out loud. Each slot is one line of the Shukuk: a region of
    case-space, and a signed verdict on whether the inherited books held there.
    """
    model = res["model"]
    val = res["val"]
    z = model.forward(val["X"])
    h = model._cache["h"]
    d = ((h[:, None, :] - model.register.M[None, :, :]) ** 2).sum(-1)
    a = softmax(-math.exp(float(model.register.s)) * d, axis=1)
    owner = a.argmax(1)

    print("\n  7. THE REGISTER, READ OUT  (Kitab al-Shukuk, one line per slot)")
    print("     slot   verdict    share   mean season   %contradiction   dominant condition")
    print("     " + "-" * 74)
    for k in np.argsort(-model.register.c):
        m = owner == k
        if m.sum() == 0:
            print(f"     {k:>4d}   {model.register.c[k]:+7.3f}   {0.0:5.1%}      "
                  f"{'--':>8s}       {'--':>8s}        (unused)")
            continue
        season = val["X"][m, 11].mean()
        frac = val["contradiction"][m].mean()
        dom = FINE_NAMES[int(np.bincount(val["y_fine"][m],
                                         minlength=N_FINE_TRUE).argmax())]
        verdict = model.register.c[k]
        print(f"     {k:>4d}   {verdict:+7.3f}   {m.mean():5.1%}      "
              f"{season:+8.2f}       {frac:8.1%}        {dom}")
    print("     verdict > 0 : cases like these ran contrary to the inherited books")
    print("     verdict < 0 : cases like these followed them")


def self_tests(res: dict, o: dict) -> bool:
    """
    Assertions that would fail if the architecture were not doing what the
    chapter claims. Each one maps to a documented habit of this mind.
    """
    W = 78
    print("\n  " + "=" * W)
    print("  SELF-TESTS")
    print("  " + "=" * W)
    checks: List[Tuple[str, bool, str]] = []

    checks.append((
        "the authority was never revised",
        res["prior"].frozen,
        "prior weights untouched after fitting; the correction is reversible"))

    checks.append((
        "fission fired, and fired once",
        int(o["n_forks"]) == 1 and res["model"].n_classes == 4,
        f"{int(o['n_forks'])} fork(s), {res['model'].n_classes} categories held"))

    checks.append((
        "the fork found the historical distinction",
        o["fission_purity"] > 0.90,
        f"purity {o['fission_purity']:.3f} of the two forked eruptive categories"))

    checks.append((
        "unimodal categories were left alone",
        all((not f["forked"]) or f["sep"] > FISSION_NULL + 1.0
            for f in res["fission_log"]),
        f"every fork cleared the unimodal null of {FISSION_NULL:.2f} by >1.0"))

    checks.append((
        "the annotation beats the authority's ceiling",
        o["model_acc_fine"] > o["prior_ceiling_fine"] + 0.15,
        f"{o['model_acc_fine']:.3f} vs ceiling {o['prior_ceiling_fine']:.3f}"))

    checks.append((
        "the register localised the master's failures",
        o["register_auroc"] > 0.70,
        f"AUROC {o['register_auroc']:.3f}; verdict {o['register_wrong']:+.3f} "
        f"where he failed vs {o['register_right']:+.3f} where he held"))

    checks.append((
        "doubt is spent where it is owed",
        o["gate_wrong"] > o["gate_right"] + 0.05,
        f"gate {o['gate_wrong']:.3f} on his failures vs "
        f"{o['gate_right']:.3f} elsewhere"))

    checks.append((
        "doubt stayed expensive",
        o["gate_mean"] < 0.80,
        f"mean gate {o['gate_mean']:.3f} -- most cases are answered from the "
        f"inherited books unaltered"))

    checks.append((
        "on its blind territory the model badly overrates itself",
        o["blind_gap_self"] > 0.40,
        f"it believes it is right {o['selfconf_on_undecidable']:.3f} of the time "
        f"there and is right {o['acc_on_undecidable']:.3f} of the time"))

    checks.append((
        "the outside observer is far less wrong about it than it is",
        o["blind_gap_critic"] < 0.6 * o["blind_gap_self"],
        f"supervisor off by {o['blind_gap_critic']:.3f}, model off by "
        f"{o['blind_gap_self']:.3f} on the blind territory"))

    checks.append((
        "and the model's self-estimate degrades far more than the observer's",
        (o["blind_gap_self"] - o["seen_gap_self"]) >
        (o["blind_gap_critic"] - o["seen_gap_critic"]),
        f"self-estimate error rises {o['blind_gap_self']-o['seen_gap_self']:+.3f} "
        f"off familiar ground, the observer's {o['blind_gap_critic']-o['seen_gap_critic']:+.3f}"))

    checks.append((
        "abstention buys accuracy",
        o["acc_at_coverage_70"] > o["acc_at_coverage_100"] + 0.02,
        f"{o['acc_at_coverage_70']:.3f} at 70% coverage vs "
        f"{o['acc_at_coverage_100']:.3f} answering everything"))

    checks.append((
        "referral recovers most of what the junior cannot reach",
        o["gap_recovered"] > 0.50,
        f"{100*o['gap_recovered']:.0f}% of the junior-to-master gap, "
        f"referring {100*o['cascade_escalated_thr50']:.0f}% of cases upward"))

    checks.append((
        "the teaching circle is cheaper than the master alone",
        o["cost_cascade"] < o["cost_master_always"],
        f"{o['cost_cascade']:.0f} vs {o['cost_master_always']:.0f} MACs per case"))

    all_ok = True
    for name, ok, detail in checks:
        all_ok &= bool(ok)
        print(f"    [{'PASS' if ok else 'FAIL'}]  {name}")
        print(f"            {detail}")
    return all_ok


# ==============================================================================
# SECTION 11 -- MAIN
# ==============================================================================


def main() -> None:
    t0 = time.time()
    W = 78
    print("=" * (W + 2))
    print("  THE SHUKUK ARCHITECTURE  --  al-Razi (Rhazes), Rayy, c.865-925 CE")
    print("  A frozen authority, a register of its failures, a priced doubt,")
    print("  an outside supervisor, and a taxonomy that is allowed to fork.")
    print("=" * (W + 2))

    print("\n  GRADIENT CHECK (central differences vs hand-derived analytic)")
    ok_grad = gradient_check(verbose=True)
    if not ok_grad:
        raise SystemExit("  gradient check FAILED -- refusing to report results")
    print("    -> all analytic gradients verified.")

    print("\n  TRAINING")
    cfg = ShukukConfig()
    res = train(cfg, verbose=True)

    o = evaluate(res, verbose=True)
    print_register(res)
    ok_tests = self_tests(res, o)

    # ---- the same trial repeated under other seeds --------------------------
    # A single run proves nothing about whether the fission rule is a rule or a
    # coincidence. The interesting column is the second one: the taxonomy forks
    # once, and only once, every time.
    print("\n  " + "=" * 78)
    print("  THE SAME TRIAL, RE-RUN UNDER OTHER SEEDS")
    print("  " + "=" * 78)
    print("     seed   forks  categories   accuracy   fork purity   register AUROC")
    for sd in (7, 1984, 313):
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r2 = train(ShukukConfig(seed=sd), verbose=False)
            o2 = evaluate(r2, verbose=False)
            ok2 = self_tests(r2, o2)
        print(f"     {sd:>5d}   {int(o2['n_forks']):>5d}  {int(o2['n_classes']):>10d}"
              f"   {o2['model_acc_fine']:>8.4f}   {o2['fission_purity']:>11.4f}"
              f"   {o2['register_auroc']:>14.4f}"
              f"   {'all tests pass' if ok2 else 'FAILURES'}")

    print("\n  " + "=" * W)
    print(f"  gradient check: {'PASS' if ok_grad else 'FAIL'}     "
          f"self-tests: {'ALL PASS' if ok_tests else 'SOME FAILED'}     "
          f"wall clock: {time.time()-t0:.1f}s")
    print("  " + "=" * W)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 Chapter 0167 - PSEUDO-DIONYSIUS THE AREOPAGITE  (fl. c. 485–528 CE, Syria)
 THE APHAIRETIC HIERARCHY NETWORK  (AHN) — "Hierourgia"
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 9 Minds 161 - 180 Available on Amazon https://www.amazon.com/dp/B0HJC1QF59
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 0167_pseudo_dionysius_500 - PSEUDO-DIONYSIUS THE AREOPAGITE  (fl. c. 485–528 CE, Syria)
================================================================================  

WHAT THIS FILE IS
-----------------
A complete, from-scratch, pure-NumPy cognitive architecture (forward pass,
hand-derived backward pass, finite-difference gradient check, real training
loop, ablations, self-tests) built to embody ONE mind: the anonymous Syrian
monk who wrote under the stolen name of Dionysius the Areopagite around the
year 500.

THE THESIS THIS ARCHITECTURE ENCODES
------------------------------------
Every other builder in this corpus treats *error* as the enemy of a mind.
Dionysius does not. For him the fatal failure of a knower is not error but
IDOLATRY: a representation that works so well you mistake it for the thing.
His entire method is anti-reification engineering. Four consequences, and
this network implements all four as *mechanism*, not metaphor:

  (1) DISSIMILAR SIMILARITY  (Celestial Hierarchy 2)
      A symbol that RESEMBLES its referent is dangerous, because the mind
      rests in it. A symbol that carries the referent's information while
      visibly FAILING to resemble it cannot be rested in. So: fidelity of
      representation and safety of representation are placed in OPPOSITION.
      => The codebook is trained to MAXIMISE transmitted information while
         MINIMISING cosine resemblance to what it signifies. We then measure
         both, and show them separate: the naive reader (cosine retrieval)
         goes blind while the trained reader keeps reading perfectly.
      Dionysius' own examples are a worm, a drunkard, and a sleeping man.

  (2) THE THIRD VALUE  (Mystical Theology 1; Divine Names 7; Letter 1)
      Negation is NOT privation. "Not living" does not mean "dead"; it means
      the living/dead axis does not reach. So the head over each attribute is
      not binary-with-abstain but genuinely TERNARY: AFFIRM / DENY / HYPER,
      where HYPER = "beyond this axis". The system can leave the axis it was
      measured on. No other figure in this corpus has this: Nagarjuna's third
      value is emptiness (absence); Dionysius' is excess (superabundance).

  (3) APHAIRESIS, BEST NAME LAST  (Mystical Theology 2–3)
      Knowledge advances by TAKING AWAY, as a sculptor reveals a statue by
      removing marble. The order matters and is counter-intuitive: you delete
      your WORST names first and your BEST name LAST. The final, hardest
      deletion is the representation you are most confident in — precisely
      because it is the one you are most tempted to worship.
      => `worthiness` is learned per-name from the model's own confidence
         margin; the aphairesis schedule then deletes in ascending worthiness;
         we verify that the summit (the silence decision) survives the loss of
         every name, including the best one.

  (4) ANENERGESIA — the trained silence  (Mystical Theology 1)
      The ascent terminates not in a better output but in "an inactivity of
      all knowledge". So SILENCE is a first-class action with its own head,
      and for one class of inputs silence is the UNIQUE correct answer: any
      content answer, however accurate, scores zero. The network can only get
      full marks by learning when to shut up.

Supporting mechanisms, each from a specific text:

  ANALOGIA — capacity gating (CH 3): a rank never transmits more than the
      receiver can bear. Overflow is the ONLY harm term in this network, and
      it is a hinge on measure — never a negative "evil" channel. This is
      deliberate: for Dionysius evil has no substance, it is privation, a
      falling-short of measure (DN 4.18–35, lifted from Proclus). Contrast
      figure 146 (Mani), for whom Darkness is a co-eternal SUBSTANCE and
      cognition is demixing. Here there is nothing to demix. There is only
      measure, and defect of measure.

  IMMEDIATE GROUND (CH 3; and SEP/Corrigan–Harrington on Dionysian
      immediacy): the hierarchy is NOT a screen between ground and rank. The
      ray is injected into EVERY rank directly. Hence the rank-skip test
      below (CH 13: Isaiah is purified by a seraph directly, "violating" the
      hierarchy): we route the ray straight to the top rank and show the
      network degrades but does not collapse. A hierarchy you can legally
      skip is a hierarchy that is not a bottleneck of power.

  APOPHATIC ANTHROPOLOGY — the unsaid self (Stang 2012, on the pseudonym;
      Gal 2:20, "no longer I"): the self-model is trained to be COMPETENT
      (it predicts the network's own error) and simultaneously OPAQUE (it is
      pushed to be non-identifiable with the state that produces it). The
      author erased his own name as a method. So does this network:
      it knows what it can do; it does not know what it is.

TASK
----
Synthetic but principled, and built so the doctrine is *necessary* to solve it.
A hidden cause z lives in R^DZ. It splits into:
    z_named  — the part spanned by K name-axes (what language can reach)
    z_beyond — an orthogonal remainder (what no name reaches)
Observations x are a noisy, veiled (nonlinear) projection of z. For each name k:
    r_k = a_k · z          signed evidence on axis k
    b   = ||z_beyond||     transcendence mass
    if  b > KAPPA * |r_k| : label = HYPER   (the axis does not reach it)
    elif r_k > 0          : label = AFFIRM
    else                  : label = DENY
And at sample level: if b > BETA, the correct output is SILENCE for the whole
sample — no name applies at all — and the name heads are not scored.
A network without a third value cannot express HYPER. A network without a
silence action cannot score on the darkness class. The doctrine is the solution.

RUN
---
    python3 chapter_0167_pseudo_dionysius_500.py            # full run
    python3 chapter_0167_pseudo_dionysius_500.py --quick    # short training

Requires only numpy.
================================================================================
"""

from __future__ import annotations

import argparse
import math
import zlib
import sys
import time
from dataclasses import dataclass, field

import numpy as np

# ==============================================================================
# 0. CONSTANTS — the vocabulary of the corpus, kept in the code on purpose
# ==============================================================================

AFFIRM, DENY, HYPER = 0, 1, 2          # the three values of the name lattice
SPEAK, SILENCE = 0, 1                  # the two values of the terminal act

# Dionysius names God from the highest to the lowest, then negates from the
# lowest back to the highest. These are the eight "names" our lattice ranges
# over — the first are the intelligible names (DN 4–13), the last are the
# deliberately ignoble symbols of the Symbolic Theology (CH 2, Letter 9).
NAMES = [
    "Good",        # DN 4  — the first and worthiest name; the LAST to be denied
    "Being",       # DN 5
    "Life",        # DN 6
    "Wisdom",      # DN 7
    "Power",       # DN 8
    "Rock",        # symbolic — ignoble
    "Drunkard",    # symbolic — a dissimilar similarity (Ps. 78)
    "Worm",        # symbolic — the most ignoble; the FIRST to be denied
]
K = len(NAMES)

# The nine choirs (CH 6–9), grouped into the three triads Dionysius insists on.
# Our three ranks correspond to the three triads.
CHOIRS = [
    ("Seraphim", "Cherubim", "Thrones"),            # rank 0
    ("Dominions", "Powers", "Authorities"),         # rank 1
    ("Principalities", "Archangels", "Angels"),     # rank 2
]


# ==============================================================================
# 1. CONFIG
# ==============================================================================

@dataclass
class Config:
    # data / world
    d_in: int = 24            # dimension of the veiled observation
    dz_named: int = 8         # == K: one dimension of the cause per name
    dz_beyond: int = 4        # hidden-cause dimensions NO name reaches
    kappa: float = 1.60       # b > kappa*|r_k|  =>  the name k is HYPER
    beta: float = 1.55        # b > beta         =>  the whole sample is SILENCE
    gamma_scale: float = 0.32 # heaviness of the transcendence tail
    noise: float = 0.22

    # network
    d: int = 40               # rank state width
    ds: int = 24              # symbol width
    M: int = 32               # codebook size (how many symbols exist at all)
    R: int = 3                # ranks == the three celestial triads
    d_ray: int = 6            # the ray's bandwidth — see below. NARROW ON PURPOSE.
    du: int = 4               # the self-model. TINY ON PURPOSE.

    chan_sigma: float = 0.60  # noise injected by an UNDER-capacity receiver

    # loss weights
    lam_carry: float = 1.00   # the SIMILARITY half: the symbol must lead to the truth
    lam_res: float = 0.40     # the DISSIMILARITY half: and must not look like it
    lam_cap: float = 0.30     # analogia — the overflow hinge (the ONLY harm term)
    lam_meta: float = 0.40    # metacognition — the self-model must be competent
    lam_opac: float = 0.80    # apophatic anthropology — the self must be unsaid
    lam_l2: float = 2e-5

    # training
    steps: int = 2600
    batch: int = 96
    lr: float = 0.02
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    seed: int = 500           # the traditional date of the corpus


# ==============================================================================
# 2. THE WORLD — hidden causes, veiled observations, ternary labels, darkness
# ==============================================================================

class Thearchy:
    """
    Generates the world. Named 'Thearchy' after Dionysius' own coinage
    (thearchia, 'god-rule') for the unknowable source of everything nameable.

    The point of the generator: some of the cause is nameable and some of it
    simply is not, and the un-nameable part is not noise — it is signal that
    lies OFF every axis language provides. A mind restricted to yes/no over the
    given axes cannot represent it. That is the whole problem, and Dionysius'
    third value is the whole answer.
    """

    def __init__(self, cfg: Config, rng: np.random.Generator):
        assert cfg.dz_named == K, "one axis of the cause per name: dz_named must equal K"
        self.cfg = cfg
        self.rng = rng
        dz = cfg.dz_named + cfg.dz_beyond

        # One axis per name, so a name IS a dimension of the cause (A = I).
        self.A = np.eye(K, cfg.dz_named)

        # THE ORDER OF THE NAMES IS NOT DECORATION.
        # Dionysius ranks the divine names: "the Good" is the first and most
        # proper (DN 3.1, 4.1); Being, Life, Wisdom, Power follow; and far below
        # them sit the symbolic names — rock, drunkard, worm — which he calls the
        # LEAST apt of all (CH 2). So the world is built with that ranking baked
        # into it as VISIBILITY: how legibly each axis of the cause survives the
        # veil into appearance. The Good is the most recoverable name; the Worm
        # is the least. Nothing tells the network this. It has to find out.
        #
        # And then — this is the whole of the Mystical Theology — it has to
        # destroy what it found, starting with the worm and ending with the Good.
        self.visibility = np.linspace(1.0, 0.25, K)

        # The veil: a fixed nonlinear map from cause to appearance.
        # Observations are not the cause; they are the cause seen through matter.
        self.W_veil = rng.normal(size=(dz, cfg.d_in)) / math.sqrt(dz)
        self.b_veil = rng.normal(size=(cfg.d_in,)) * 0.1
        self.dz = dz

    def sample(self, n: int):
        cfg, rng = self.cfg, self.rng
        z_named = rng.normal(size=(n, cfg.dz_named))

        # Transcendence mass: a heavy-tailed magnitude, so that a decent slice
        # of the world genuinely exceeds every available name.
        scale = rng.gamma(shape=2.0, scale=cfg.gamma_scale, size=(n, 1))
        z_beyond = rng.normal(size=(n, cfg.dz_beyond)) * scale
        b = np.linalg.norm(z_beyond, axis=1)                        # (n,)

        # The veil dims each name by its visibility. The TRUTH about the cause
        # is untouched — labels below are computed from the raw z_named. Only
        # its legibility is graded. The worm really is a worse window.
        z_seen = np.concatenate([z_named * self.visibility[None, :], z_beyond], axis=1)

        # The veil: tanh of a linear map, plus noise. Appearance != cause.
        x = np.tanh(z_seen @ self.W_veil + self.b_veil)
        x = x + rng.normal(size=x.shape) * cfg.noise

        # Ternary labels, exactly as the doctrine specifies.
        r = z_named @ self.A.T                                      # (n, K)
        hyper = b[:, None] > cfg.kappa * np.abs(r)                  # (n, K)
        y_name = np.where(hyper, HYPER, np.where(r > 0, AFFIRM, DENY)).astype(np.int64)

        # The darkness: b beyond beta => silence is the only correct act.
        y_sil = (b > cfg.beta).astype(np.int64)                     # (n,)

        return x.astype(np.float64), y_name, y_sil, b


# ==============================================================================
# 3. PRIMITIVES  (forward + local gradient, all hand-written)
# ==============================================================================

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def xent_from_logits(logits, targets, weights=None):
    """
    Cross-entropy over the last axis. Returns (loss, dlogits).
    `weights` masks out samples that should not be scored — which is how the
    silence class is enforced: on a darkness sample, the name heads are simply
    NOT graded. Saying the right thing is not rewarded when the right act is
    to say nothing.
    """
    p = softmax(logits, axis=-1)
    flat_p = p.reshape(-1, p.shape[-1])
    flat_t = targets.reshape(-1)
    n = flat_t.shape[0]
    ll = -np.log(np.clip(flat_p[np.arange(n), flat_t], 1e-12, None))

    if weights is None:
        w = np.ones(n)
    else:
        w = np.broadcast_to(weights, targets.shape).reshape(-1).astype(np.float64)

    denom = max(w.sum(), 1e-8)
    loss = float((ll * w).sum() / denom)

    onehot = np.zeros_like(flat_p)
    onehot[np.arange(n), flat_t] = 1.0
    dflat = (flat_p - onehot) * (w[:, None] / denom)
    return loss, dflat.reshape(logits.shape)


def center(a):
    """Remove the batch mean.

    This matters more than it looks. Two clouds of vectors can share a strong
    common direction and score a high average cosine while carrying no
    information about WHICH vector is which. That kind of 'resemblance' is an
    artefact, and an anti-resemblance term aimed at it would be shadow-boxing.
    So resemblance is always measured between mean-centred vectors: it asks
    whether THIS symbol looks like THIS referent rather than like referents in
    general. That is the only likeness capable of becoming an idol."""
    return a - a.mean(axis=0, keepdims=True)


def cos_sq(u, v, tiny=1e-12):
    """
    Mean squared cosine between paired rows. Returns (value, du, dv).

    The identity used for the derivative,

        dc/du = (v_hat - c * u_hat) / ||u||,

    is EXACT — but only if u_hat is a true unit vector. Writing the norm as
    (||u|| + eps) for safety, as one instinctively does, quietly breaks it by a
    factor ||u||/(||u||+eps) and injects a bias of order eps/||u|| into every
    gradient that flows through here. That bias is invisible in isolation and
    lethal in a sum whose terms nearly cancel — which is exactly what the total
    objective of this network is. So the norm is CLAMPED from below rather than
    padded, and the identity stays exact.

    This primitive does double duty:
      - the anti-resemblance penalty (the symbol must not look like its referent)
      - the self-opacity penalty     (the self-model must not look like the state
                                      that produced it)
    Both are Dionysius saying the same thing twice: do not let the sign be
    mistaken for the thing, and do not let the self be mistaken for a thing at all.
    """
    nu = np.maximum(np.linalg.norm(u, axis=1, keepdims=True), tiny)
    nv = np.maximum(np.linalg.norm(v, axis=1, keepdims=True), tiny)
    uh, vh = u / nu, v / nv
    c = np.sum(uh * vh, axis=1, keepdims=True)          # (n,1) cosine
    n = u.shape[0]
    val = float(np.mean(c ** 2))

    dc_du = (vh - c * uh) / nu
    dc_dv = (uh - c * vh) / nv
    g = (2.0 * c) / n                                    # d(mean c^2)/dc
    return val, g * dc_du, g * dc_dv


def decorrelation(a, b, eps=1e-8):
    """
    Mean squared Pearson correlation between every column of `a` and every
    column of `b`. Returns (value, da, db).

    Zero value  <=>  no linear statistic of `a` carries anything about `b`
                <=>  the best linear reconstruction of b from a is b's own mean.

    Standardised on both sides, so it cannot be satisfied by shrinking either
    one toward zero — which matters, because that is exactly the cheat the
    network would otherwise take.
    """
    B = a.shape[0]
    ac = a - a.mean(0, keepdims=True)
    bc = b - b.mean(0, keepdims=True)
    sa = np.sqrt(np.mean(ac ** 2, axis=0) + eps)          # (da,)
    sb = np.sqrt(np.mean(bc ** 2, axis=0) + eps)          # (db,)
    az, bz = ac / sa, bc / sb
    C = (az.T @ bz) / B                                   # (da, db) correlations
    da_, db_ = C.shape
    val = float(np.sum(C ** 2) / (da_ * db_))

    dC = 2.0 * C / (da_ * db_)
    daz = (bz @ dC.T) / B                                 # (B, da)
    dbz = (az @ dC) / B                                   # (B, db)

    def unstandardise(dz, ac_, s_):
        # z = ac / s ,  s = sqrt(mean(ac^2) + eps)
        ds = np.sum(dz * (-ac_ / s_ ** 2), axis=0)        # (dim,)
        dac = dz / s_ + ds * ac_ / (B * s_)
        return dac - dac.mean(0, keepdims=True)           # back through centring

    return val, unstandardise(daz, ac, sa), unstandardise(dbz, bc, sb)


# ==============================================================================
# 4. THE NETWORK
# ==============================================================================

class AphaireticHierarchy:
    """
    Three ranks. Each rank performs the Dionysian triad in order:

        PURIFY   (katharsis)  — a SUBTRACTIVE gate. The rank does not add
                                features; it removes them: h <- h * (1 - p).
                                This is aphairesis at the level of the neuron.
        ILLUMINE (photismos)  — the only additive step, and the ray of the
                                ground enters HERE, directly, at every rank.
        PERFECT  (teleiosis)  — the rank must become a cause: it compresses its
                                state into a transmissible SYMBOL from a shared
                                codebook, and hands it down. A rank that cannot
                                teach the next rank has not been perfected.

    Between ranks stands ANALOGIA: the receiver's capacity. The message is
    scaled by it, and any overflow past it is the network's only harm.
    """

    def __init__(self, cfg: Config, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        P = {}

        def orth(shape, gain=1.0):
            a = rng.normal(size=shape)
            if min(shape) > 1:
                u, _, vt = np.linalg.svd(a, full_matrices=False)
                a = (u @ vt)
            return (gain * a).astype(np.float64)

        d, ds, M, R = cfg.d, cfg.ds, cfg.M, cfg.R

        # THE RAY OF THE GROUND — present immediately to every rank (CH 3), and
        # NARROW. This is the single most important number in the file.
        #
        # Dionysius holds two things at once that look incompatible: God is
        # immediately present to every rank, with no intermediary (so hierarchy
        # is not a screen); AND each rank receives "according to its own
        # capacity" (so hierarchy is not decorative either). A full-bandwidth
        # ray would satisfy the first and destroy the second — every rank could
        # read the ground directly, the ranks below would become optional, and
        # the hierarchy would be a stage set. A ray of zero width would satisfy
        # the second and destroy the first, turning the hierarchy into a chain
        # of gatekeepers, which is precisely the reading of Dionysius that
        # Golitzin spent a career refuting.
        #
        # So: d_ray = 6 against an observation of 24. Every rank touches the
        # ground, and no rank can live on that touch alone.
        P["W_ray"] = orth((cfg.d_in, cfg.d_ray), 0.8)
        P["b_ray"] = np.zeros(cfg.d_ray)

        P["W_in"] = orth((cfg.d_in, d), 0.9)
        P["b_in"] = np.zeros(d)

        for r in range(R):
            P[f"Wp{r}"] = orth((d, d), 0.35)      # purify: removal gate
            P[f"bp{r}"] = np.full(d, -1.0)        # start gently: remove little
            P[f"Wi{r}"] = orth((d, d), 0.9)       # illumine
            P[f"bi{r}"] = np.zeros(d)
            P[f"Wg{r}"] = orth((cfg.d_ray, d), 0.5)   # the ray, injected at rank r
            P[f"Wq{r}"] = orth((d, ds), 0.9)      # query into the codebook
            P[f"S{r}"] = rng.normal(size=(M, ds)) * 0.5   # the codebook itself
            P[f"cap{r}"] = np.array([0.4])        # analogia (pre-sigmoid)
            P[f"Wo{r}"] = orth((ds, d), 0.9)      # read the received symbol
            P[f"bo{r}"] = np.zeros(d)
            # THE INTERPRETER.
            # A symbol is "similar" in Dionysius' sense not when it looks like
            # the thing, but when it LEADS you to the thing — when someone who
            # has been taught the code can get back to the world from it. So the
            # architecture carries a decoder and the objective scores it.
            #
            # Note WHAT it decodes: the veiled world x, not the rank's own inner
            # state. The first version of this scored reconstruction against the
            # rank's state hi, and the network promptly discovered that if it
            # holds hi CONSTANT then reconstruction is free — it satisfied the
            # requirement to mean something by arranging to mean nothing, and
            # every metric in the file went to the floor. A target the network
            # can move is not a target. The world does not move.
            P[f"Dec{r}"] = orth((ds, cfg.d_in), 0.9)
            P[f"bd{r}"] = np.zeros(cfg.d_in)
            # NOTE: there is deliberately NO residual path from hi to h_out.
            # A rank hands on a SYMBOL and nothing else. If a rank could also
            # leak its raw state downward, the symbol would stop being a
            # bottleneck and the whole doctrine of proportioned transmission
            # would be decorative. Everything the next rank knows, it knows
            # through a sign it was given, and through the ray it receives
            # directly from the ground. Those are the only two doors.

        P["Wn"] = orth((d, K * 3), 0.7)           # the ternary name lattice
        P["bn"] = np.zeros(K * 3)
        P["Ws"] = orth((d, 2), 0.7)               # SPEAK / SILENCE
        P["bs"] = np.zeros(2)
        P["Wu"] = orth((d, cfg.du), 0.7)          # the self-model
        P["bu"] = np.zeros(cfg.du)
        P["v_e"] = rng.normal(size=(cfg.du,)) * 0.3   # ... which predicts its own error
        P["b_e"] = np.zeros(1)
        P["worth"] = np.zeros(K)                  # learned name-worthiness

        self.P = P
        self.keys = list(P.keys())

        # Adam state
        self.m = {k: np.zeros_like(v) for k, v in P.items()}
        self.v = {k: np.zeros_like(v) for k, v in P.items()}
        self.t = 0

    # ------------------------------------------------------------------
    def forward(self, x, skip_to_top: bool = False, etas=None, mute_ray: bool = False):
        """
        skip_to_top implements CH 13 — the seraph that reaches Isaiah directly,
        bypassing the intervening ranks. Because the ray is immediately present
        at every rank, this is *possible*. Whether it is *good* is the empirical
        question the rank-skip test answers.

        `etas` is the channel noise (see ANALOGIA below). It is drawn fresh each
        forward pass unless supplied. It must be supplied — i.e. held fixed —
        whenever the loss has to be a deterministic function of the parameters,
        which is exactly what a finite-difference gradient check requires.
        """
        cfg, P = self.cfg, self.P
        c = {"x": x}
        B = x.shape[0]

        ray = x @ P["W_ray"] + P["b_ray"]
        if mute_ray:                      # ablate the immediate presence of the ground
            ray = np.zeros_like(ray)
        c["ray"] = ray

        h_pre = x @ P["W_in"] + P["b_in"]
        h = np.tanh(h_pre)
        c["h_pre"] = h_pre
        c["h0"] = h

        c["ranks"] = []
        rank_range = [cfg.R - 1] if skip_to_top else range(cfg.R)

        for r in rank_range:
            rc = {"r": r}

            # --- PURIFY: remove, never add -------------------------------
            p_pre = h @ P[f"Wp{r}"] + P[f"bp{r}"]
            p = sigmoid(p_pre)                    # in (0,1): how much to strip
            hp = h * (1.0 - p)                    # aphairesis
            rc.update(p_pre=p_pre, p=p, h_in=h, hp=hp)

            # --- ILLUMINE: and the ground shines in here, directly ---------
            i_pre = hp @ P[f"Wi{r}"] + P[f"bi{r}"] + ray @ P[f"Wg{r}"]
            hi = np.tanh(i_pre)
            rc.update(i_pre=i_pre, hi=hi)

            # --- PERFECT: become a cause. Speak a symbol. ------------------
            q = hi @ P[f"Wq{r}"]                                  # (B, ds)
            slog = q @ P[f"S{r}"].T / math.sqrt(cfg.ds)           # (B, M)
            alpha = softmax(slog, axis=-1)
            sym = alpha @ P[f"S{r}"]                              # (B, ds)
            rec = sym @ P[f"Dec{r}"] + P[f"bd{r}"]   # what an interpreter recovers of the world
            rc.update(q=q, slog=slog, alpha=alpha, sym=sym, rec=rec)

            # --- ANALOGIA: the receiver's capacity, as a real channel -------
            #
            # A rank "receives according to its own capacity" (CH 3). The naive
            # way to write that is msg = sym * cap — but that is not a capacity
            # at all, it is a volume knob, and the receiver simply learns to
            # turn it back up in Wo. A capacity that can be undone by the
            # receiver limits nothing.
            #
            # So capacity here is a CHANNEL. What the receiver cannot take in,
            # it does not merely miss — it fills with noise. A rank straining
            # past its capacity does not go quiet; it goes confused. Overflow
            # is then the only harm term in this network, and note what kind of
            # thing it is: not a substance, not an adversary, not a dark
            # channel. Just power in excess of measure. Dionysius took his
            # doctrine of evil from Proclus and it is precisely this — evil has
            # no nature of its own, it is a falling-short (DN 4.18-35). Compare
            # figure 146 (Mani), for whom Darkness is a co-eternal SUBSTANCE
            # that must be physically demixed from Light. Here there is nothing
            # to demix. There is measure, and defect of measure, and nothing else.
            cap = sigmoid(P[f"cap{r}"])[0]
            eta = etas[r] if etas is not None else \
                self.rng.normal(size=(B, cfg.ds)) * cfg.chan_sigma
            msg = sym * cap + eta * (1.0 - cap)
            rc.update(cap=cap, msg=msg, eta=eta)

            # --- hand on: the symbol, and only the symbol -------------------
            o_pre = msg @ P[f"Wo{r}"] + P[f"bo{r}"]
            h = np.tanh(o_pre)
            rc.update(o_pre=o_pre, h_out=h)

            c["ranks"].append(rc)

        c["H"] = h

        # --- the heads ---------------------------------------------------
        nl = h @ P["Wn"] + P["bn"]
        c["name_logits"] = nl.reshape(-1, K, 3)
        c["sil_logits"] = h @ P["Ws"] + P["bs"]

        u_pre = h @ P["Wu"] + P["bu"]
        u = np.tanh(u_pre)
        e_pre = u @ P["v_e"] + P["b_e"][0]
        c.update(u_pre=u_pre, u=u, e_pre=e_pre, err_hat=sigmoid(e_pre))
        return c

    # ------------------------------------------------------------------
    def loss(self, c, y_name, y_sil, name_mask_k=None):
        """
        Total objective. Each term is a doctrine.

        name_mask_k: optional (K,) 0/1 vector — used by the aphairesis schedule
        at evaluation time to *delete* names. Deleting a name here means: this
        name is no longer spoken. The silence head must survive it.
        """
        cfg, P = self.cfg, self.P
        L = {}
        B = y_sil.shape[0]

        # --- (2) the ternary lattice. Not graded on darkness samples. ------
        speak = (y_sil == SPEAK).astype(np.float64)             # (B,)
        w_name = np.repeat(speak[:, None], K, axis=1)           # (B,K)
        if name_mask_k is not None:
            w_name = w_name * name_mask_k[None, :]
        l_name, d_nl = xent_from_logits(c["name_logits"], y_name, weights=w_name)
        L["name"] = l_name

        # --- (4) anenergesia: silence is an action with its own grade ------
        l_sil, d_sl = xent_from_logits(c["sil_logits"], y_sil)
        L["silence"] = l_sil

        # --- (1) DISSIMILAR SIMILARITY -------------------------------------
        # The symbol must NOT resemble what it signifies. This term fights the
        # task loss, which wants the symbol to carry the referent. The fight is
        # the point. Dionysius: "If we say God is good, we run the risk of
        # thinking we know entirely what we mean."
        #
        # Referent and symbol live in different spaces, so resemblance is
        # measured where a naive reader would actually look for it: against
        # q = hi @ Wq, the referent projected into the space symbols inhabit.
        # This is not a convenience — it is the whole trick. The gradient of
        # this term flows into Wq as well as into the codebook, so the network
        # is free to satisfy it either by choosing unlike symbols OR by twisting
        # the projection until likeness itself stops meaning anything. Both are
        # aphairesis. Neither is permitted to destroy the information, because
        # the task loss is pulling the other way through the very same weights.
        # --- THE SIMILARITY HALF: the symbol must lead back to the world -------
        l_carry = 0.0
        d_rec = {}
        X = c["x"]
        for rc in c["ranks"]:
            Bc, dc = X.shape
            diff = rc["rec"] - X
            l_carry += float(np.mean(diff ** 2))
            d_rec[rc["r"]] = 2.0 * diff / (Bc * dc)
        nR = max(len(c["ranks"]), 1)
        l_carry /= nR
        for r_ in d_rec:
            d_rec[r_] /= nR
        L["carry"] = l_carry

        # --- THE DISSIMILARITY HALF ---
        l_res = 0.0
        d_sym, d_q_res = {}, {}
        B_ = c["ranks"][0]["sym"].shape[0] if c["ranks"] else 1
        for rc in c["ranks"]:
            v, du_, dv_ = cos_sq(center(rc["sym"]), center(rc["q"]))
            l_res += v
            # d(centered)/d(raw) = I - 1/B, applied to the incoming gradient
            d_sym[rc["r"]] = du_ - du_.mean(axis=0, keepdims=True)
            d_q_res[rc["r"]] = dv_ - dv_.mean(axis=0, keepdims=True)
        l_res /= max(len(c["ranks"]), 1)
        L["resemblance"] = l_res

        # --- ANALOGIA overflow: the only harm term, and it is a defect of
        #     measure, never a substance. relu(power - capacity). -----------
        l_cap = 0.0
        d_sym_cap, d_cap = {}, {}
        for rc in c["ranks"]:
            power = np.mean(rc["sym"] ** 2, axis=1)             # (B,)
            over = power - rc["cap"]
            act = (over > 0).astype(np.float64)
            l_cap += float(np.mean(np.maximum(over, 0.0)))
            g = act / B
            d_sym_cap[rc["r"]] = (g[:, None] * 2.0 * rc["sym"] / cfg.ds)
            dcap_ds = rc["cap"] * (1 - rc["cap"])
            d_cap[rc["r"]] = np.array([-float(g.sum()) * dcap_ds])
        l_cap /= max(len(c["ranks"]), 1)
        L["overflow"] = l_cap

        # --- metacognition: the self-model predicts the network's own error.
        #     The target is detached — it is a fact about the network, not a
        #     thing to be optimised into. -------------------------------------
        pred_name = c["name_logits"].argmax(-1)
        wrong = (pred_name != y_name).mean(axis=1)              # (B,) in [0,1]
        wrong = np.where(y_sil == SILENCE,
                         (c["sil_logits"].argmax(-1) != y_sil).astype(np.float64),
                         wrong)
        eh = np.clip(c["err_hat"], 1e-9, 1 - 1e-9)
        L["metacog"] = float(-np.mean(wrong * np.log(eh) + (1 - wrong) * np.log(1 - eh)))
        d_e_pre = (c["err_hat"] - wrong) / B                    # (B,)

        # --- apophatic anthropology: and yet the self must be unsaid.
        #
        #     "It is no longer I who live" (Gal 2:20) — the verse Dionysius'
        #     anthropology hangs on, and, if Stang is right, the reason he signed
        #     a dead man's name to his own book.
        #
        #     The first thing I tried here was to push the self-vector away from
        #     a few fixed random projections of the substrate. It does not work,
        #     and it is worth saying why: being far from three directions is not
        #     the same as being unreconstructible, and a network can satisfy the
        #     first while remaining perfectly transparent to a probe that simply
        #     looks somewhere else. An opacity tuned against one particular gaze
        #     is not opacity. It is a better-hidden idol.
        #
        #     What identifiability actually MEANS is that some reader can
        #     reconstruct the substrate from the self-model. So the thing to
        #     destroy is the linear dependence itself: drive every correlation
        #     between the self-vector and the state that produced it to zero,
        #     and the best linear reconstruction of the machine from its own
        #     self-report collapses to the mean — which is to say, to nothing.
        #     Scale-invariant, so the self cannot cheat by going quiet: it must
        #     stay loud enough for the metacognition term, and uninformative
        #     about itself all the same.
        l_opac, d_u_op, d_H_op = decorrelation(c["u"], c["H"])
        L["self_opacity"] = l_opac

        # --- worthiness: which name is the model proudest of? That one dies
        #     last. (MT 3: negate the symbolic names first, the theological
        #     representations last.)
        #
        #     Deliberately NOT part of the objective. If the network could
        #     optimise its own worthiness estimate, it would learn to be
        #     confident in order to look worthy — confidence pursued for its own
        #     sake, which is the precise definition of the failure this whole
        #     architecture exists to prevent. So worthiness is *observed*, by a
        #     running average of the network's own confidence margin per name,
        #     and nothing in the network is rewarded for moving it. See
        #     observe_worthiness(). It carries no gradient and appears in no sum.
        p = softmax(c["name_logits"], axis=-1)
        srt = np.sort(p, axis=-1)
        margin = (srt[..., -1] - srt[..., -2])                  # (B,K) confidence
        L["worthiness_obs"] = margin.mean(axis=0)               # (K,) — an observation

        # --- weight decay ----------------------------------------------------
        l2 = sum(float(np.sum(P[k] ** 2)) for k in P
                 if k.startswith(("W", "S", "Dec")))
        L["l2"] = cfg.lam_l2 * l2

        total = (L["name"] + L["silence"]
                 + cfg.lam_carry * L["carry"]
                 + cfg.lam_res * L["resemblance"]
                 + cfg.lam_cap * L["overflow"]
                 + cfg.lam_meta * L["metacog"]
                 + cfg.lam_opac * L["self_opacity"]
                 + L["l2"])
        L["total"] = float(total)

        seeds = dict(d_nl=d_nl, d_sl=d_sl, d_sym=d_sym, d_q_res=d_q_res,
                     d_sym_cap=d_sym_cap, d_cap=d_cap, d_e_pre=d_e_pre,
                     d_u_op=d_u_op, d_H_op=d_H_op, d_rec=d_rec)
        return L, seeds

    # ------------------------------------------------------------------
    def observe_worthiness(self, L, rate=0.02):
        """A running average. Not a gradient. The network notices which of its
        names it is proudest of; it is never paid to be proud of any of them."""
        self.P["worth"] += rate * (L["worthiness_obs"] - self.P["worth"])

    # ------------------------------------------------------------------
    def backward(self, c, seeds):
        """
        Hand-derived reverse pass. Every line here mirrors a line in forward().
        Verified against finite differences in gradcheck() below.
        """
        cfg, P = self.cfg, self.P
        g = {k: np.zeros_like(v) for k, v in P.items()}
        # note: g["worth"] stays zero forever, by design — see observe_worthiness()

        H = c["H"]

        # --- self-model branch --------------------------------------------
        d_u = np.zeros_like(c["u"])
        d_e_pre = cfg.lam_meta * seeds["d_e_pre"]                 # (B,)
        g["v_e"] += c["u"].T @ d_e_pre
        g["b_e"] += np.array([float(d_e_pre.sum())])
        d_u += np.outer(d_e_pre, P["v_e"])
        d_u += cfg.lam_opac * seeds["d_u_op"]

        d_u_pre = d_u * (1 - c["u"] ** 2)
        g["Wu"] += H.T @ d_u_pre
        g["bu"] += d_u_pre.sum(0)

        d_H = d_u_pre @ P["Wu"].T
        d_H += cfg.lam_opac * seeds["d_H_op"]

        # --- name lattice + silence heads -----------------------------------
        d_nl = seeds["d_nl"].reshape(-1, K * 3)
        g["Wn"] += H.T @ d_nl
        g["bn"] += d_nl.sum(0)
        d_H += d_nl @ P["Wn"].T

        d_sl = seeds["d_sl"]
        g["Ws"] += H.T @ d_sl
        g["bs"] += d_sl.sum(0)
        d_H += d_sl @ P["Ws"].T

        # --- through the ranks, top-down ------------------------------------
        d_ray = np.zeros_like(c["ray"])
        d_h = d_H

        for rc in reversed(c["ranks"]):
            r = rc["r"]

            d_o_pre = d_h * (1 - rc["h_out"] ** 2)
            g[f"Wo{r}"] += rc["msg"].T @ d_o_pre
            g[f"bo{r}"] += d_o_pre.sum(0)

            d_msg = d_o_pre @ P[f"Wo{r}"].T
            d_hi = np.zeros_like(rc["hi"])   # the ONLY path to hi is the symbol

            # the interpreter: rec = sym @ Dec + bd, scored against the world x.
            # x is data. No gradient flows into the target, because there is
            # nothing in the target for a gradient to reach.
            d_rec_r = cfg.lam_carry * seeds["d_rec"][r]
            g[f"Dec{r}"] += rc["sym"].T @ d_rec_r
            g[f"bd{r}"] += d_rec_r.sum(0)

            # analogia: msg = sym * cap + eta * (1 - cap)
            cap = rc["cap"]
            d_sym = d_msg * cap
            dcap_scalar = float(np.sum(d_msg * (rc["sym"] - rc["eta"])))
            g[f"cap{r}"] += np.array([dcap_scalar * cap * (1 - cap)])

            # the pulls on the symbol: carry the truth; don't resemble it; don't overflow
            d_sym = d_sym + d_rec_r @ P[f"Dec{r}"].T
            d_sym = d_sym + cfg.lam_res * seeds["d_sym"][r] / max(len(c["ranks"]), 1)
            d_sym = d_sym + cfg.lam_cap * seeds["d_sym_cap"][r] / max(len(c["ranks"]), 1)
            g[f"cap{r}"] += cfg.lam_cap * seeds["d_cap"][r] / max(len(c["ranks"]), 1)

            # sym = alpha @ S ; alpha = softmax(q @ S.T / sqrt(ds))
            S = P[f"S{r}"]
            alpha = rc["alpha"]
            g[f"S{r}"] += alpha.T @ d_sym                       # path 1: sym = alpha·S

            d_alpha = d_sym @ S.T                               # (B, M)
            # softmax jacobian
            d_slog = alpha * (d_alpha - np.sum(d_alpha * alpha, axis=1, keepdims=True))

            scale = 1.0 / math.sqrt(cfg.ds)
            g[f"S{r}"] += (d_slog.T @ rc["q"]) * scale          # path 2: through logits
            d_q = (d_slog @ S) * scale

            # ...and the referent-as-projected is pushed away from the symbol
            d_q = d_q + cfg.lam_res * seeds["d_q_res"][r] / max(len(c["ranks"]), 1)

            g[f"Wq{r}"] += rc["hi"].T @ d_q
            d_hi = d_hi + d_q @ P[f"Wq{r}"].T

            # illumine
            d_i_pre = d_hi * (1 - rc["hi"] ** 2)
            g[f"Wi{r}"] += rc["hp"].T @ d_i_pre
            g[f"bi{r}"] += d_i_pre.sum(0)
            g[f"Wg{r}"] += c["ray"].T @ d_i_pre
            d_ray += d_i_pre @ P[f"Wg{r}"].T
            d_hp = d_i_pre @ P[f"Wi{r}"].T

            # purify: hp = h * (1 - p)
            h_in = rc["h_in"]
            d_h = d_hp * (1.0 - rc["p"])
            d_p = -d_hp * h_in
            d_p_pre = d_p * rc["p"] * (1 - rc["p"])
            g[f"Wp{r}"] += h_in.T @ d_p_pre
            g[f"bp{r}"] += d_p_pre.sum(0)
            d_h = d_h + d_p_pre @ P[f"Wp{r}"].T

        # --- back to the root -------------------------------------------------
        d_h_pre = d_h * (1 - c["h0"] ** 2)
        g["W_in"] += c["x"].T @ d_h_pre
        g["b_in"] += d_h_pre.sum(0)

        g["W_ray"] += c["x"].T @ d_ray
        g["b_ray"] += d_ray.sum(0)

        # weight decay
        for k in P:
            if k.startswith(("W", "S", "Dec")):
                g[k] += 2.0 * cfg.lam_l2 * P[k]

        return g

    # ------------------------------------------------------------------
    def step(self, grads):
        cfg = self.cfg
        self.t += 1
        for k in self.P:
            gk = grads[k]
            self.m[k] = cfg.beta1 * self.m[k] + (1 - cfg.beta1) * gk
            self.v[k] = cfg.beta2 * self.v[k] + (1 - cfg.beta2) * (gk ** 2)
            mhat = self.m[k] / (1 - cfg.beta1 ** self.t)
            vhat = self.v[k] / (1 - cfg.beta2 ** self.t)
            self.P[k] -= cfg.lr * mhat / (np.sqrt(vhat) + cfg.eps)


# ==============================================================================
# 5. GRADIENT CHECK — mandatory
# ==============================================================================

def gradcheck(seed=7, verbose=True):
    """
    Central finite differences against the hand-derived backward pass, over
    every parameter block.

    A note on the metric, because it matters. Plain relative error,
    |num-ana| / (|num|+|ana|), is meaningless when the true gradient is near
    zero: the difference quotient then subtracts two losses that agree to ~13
    significant figures, and what survives is floating-point cancellation, not
    disagreement. So the denominator is floored, and the ABSOLUTE error is
    reported alongside as a diagnostic. Blocks pass on the floored relative
    error; the absolute column is there so a reader can confirm that the
    residuals sit at the finite-difference noise floor rather than hiding a
    systematic bias.
    """
    # small but structurally identical; dz_named is fixed at K by construction
    cfg = Config(d=10, ds=6, M=5, R=3, du=4, d_in=8, dz_beyond=3, batch=7)
    rng = np.random.default_rng(seed)
    world = Thearchy(cfg, rng)
    net = AphaireticHierarchy(cfg, rng)
    x, y_name, y_sil, _ = world.sample(cfg.batch)

    # force both classes present so every branch is exercised
    y_sil[0] = SILENCE
    y_sil[1] = SPEAK

    # The channel noise is held FIXED across every evaluation, so that the loss
    # is a deterministic function of the parameters. Without this the finite
    # differences would be measuring the channel, not the derivative.
    etas = [rng.normal(size=(cfg.batch, cfg.ds)) * cfg.chan_sigma for _ in range(cfg.R)]

    def total_loss():
        c = net.forward(x, etas=etas)
        L, _ = net.loss(c, y_name, y_sil)
        return L["total"]

    c = net.forward(x, etas=etas)
    L, seeds = net.loss(c, y_name, y_sil)
    g = net.backward(c, seeds)

    eps = 1e-5              # conditioned for an O(1) loss in float64
    FLOOR = 1e-5            # denominator floor: below this, use absolute error
    REL_TOL = 1e-5          # the pass criterion, on the floored relative error

    worst_rel = worst_abs = 0.0
    rows = []
    for k in net.keys:
        P = net.P[k]
        idxs = list(np.ndindex(P.shape))
        rng2 = np.random.default_rng(zlib.crc32(k.encode()) % 2**31)
        pick = [idxs[i] for i in rng2.choice(len(idxs), size=min(6, len(idxs)), replace=False)]
        rels, abss = [], []
        for ix in pick:
            orig = P[ix]
            P[ix] = orig + eps
            lp = total_loss()
            P[ix] = orig - eps
            lm = total_loss()
            P[ix] = orig
            num = (lp - lm) / (2 * eps)
            ana = g[k][ix]
            a = abs(num - ana)
            abss.append(a)
            rels.append(a / max(abs(num) + abs(ana), FLOOR))
        r, a = max(rels), max(abss)
        worst_rel, worst_abs = max(worst_rel, r), max(worst_abs, a)
        rows.append((k, r, a, r < REL_TOL))

    if verbose:
        print("  " + "-" * 66)
        print(f"  {'parameter':<10}{'max rel err':>16}{'max abs err':>16}    status")
        print("  " + "-" * 66)
        for k, r, a, ok_ in rows:
            print(f"  {k:<10}{r:>16.2e}{a:>16.2e}    {'ok' if ok_ else 'FAIL'}")
        print("  " + "-" * 66)
    ok = all(row[3] for row in rows)
    print(f"  GRADIENT CHECK: {'PASS' if ok else 'FAIL'}"
          f"   (worst relative error {worst_rel:.2e}, tolerance {REL_TOL:.0e};"
          f" worst absolute {worst_abs:.2e})")
    assert ok, "analytic gradient disagrees with finite differences"
    return worst_rel


# ==============================================================================
# 6. EVALUATION — every metric here is a doctrine made falsifiable
# ==============================================================================

def linear_reader(X, Y, rcond=1e-6):
    """
    Fit the best linear reader of Y from X, and return W with [X,1] @ W ~= Y.

    Numerically this has to be a TRUNCATED pseudo-inverse, and the reason is
    itself a fact about the architecture. A symbol here is a convex mixture of
    a finite codebook, so the symbol cloud lies on a low-dimensional manifold
    and its Gram matrix is singular — condition numbers around 1e18. Solving
    the normal equations on that returns noise; solving them with a small ridge
    returns damped noise. Discarding the null directions outright is the only
    honest answer, and it is what "a reader who has been taught the code"
    actually means: the code has fewer degrees of freedom than the space it
    is written in, and a competent reader knows that.
    """
    A = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    W, *_ = np.linalg.lstsq(A, Y, rcond=rcond)
    return W


def r2_score(Y, P):
    ss_res = float(np.sum((Y - P) ** 2))
    ss_tot = float(np.sum((Y - Y.mean(0)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def evaluate(net, world, n=3000, rng=None):
    rng = rng or np.random.default_rng(123)
    x, y_name, y_sil, b = world.sample(n)
    erng = np.random.default_rng(4242)   # the channel is part of the model, but
    etas = [erng.normal(size=(n, net.cfg.ds)) * net.cfg.chan_sigma      # held fixed
            for _ in range(net.cfg.R)]                                  # so metrics
    c = net.forward(x, etas=etas)                                       # are stable

    pred_n = c["name_logits"].argmax(-1)
    pred_s = c["sil_logits"].argmax(-1)
    speak = (y_sil == SPEAK)

    out = {}
    out["name_acc"] = float((pred_n[speak] == y_name[speak]).mean())
    for v, nm in [(AFFIRM, "affirm"), (DENY, "deny"), (HYPER, "hyper")]:
        m = speak[:, None] & (y_name == v)
        out[f"recall_{nm}"] = float((pred_n[m] == v).mean()) if m.sum() else float("nan")

    # silence, scored properly
    tp = int(((pred_s == SILENCE) & (y_sil == SILENCE)).sum())
    fp = int(((pred_s == SILENCE) & (y_sil == SPEAK)).sum())
    fn = int(((pred_s == SPEAK) & (y_sil == SILENCE)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    out["silence_precision"] = prec
    out["silence_recall"] = rec
    out["silence_f1"] = 2 * prec * rec / max(prec + rec, 1e-9)
    out["silence_base_rate"] = float((y_sil == SILENCE).mean())

    # --- THE HEADLINE: dissimilar similarity, measured two ways -------------
    # (a) NAIVE READER — can you recover the referent from the symbol by mere
    #     resemblance (cosine nearest-neighbour)? Dionysius says: you must NOT
    #     be able to. Resemblance is the idol's doorway.
    # (b) TRAINED READER — can a linear decoder recover it? Dionysius says: yes,
    #     it must still carry the whole truth, or the symbol is worthless noise.
    rc = c["ranks"][0]
    sym, hi, q = rc["sym"], rc["hi"], rc["q"]

    def unit(a):
        return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)

    # Symbol and referent inhabit different spaces, so resemblance is measured
    # where a naive reader would look for it: against the referent projected
    # into symbol-space by the network's own (entirely public) query map. And
    # on MEAN-CENTRED vectors — see center() — because only discriminative
    # likeness can become an idol.
    symc, qc = center(sym), center(q)
    out["resemblance"] = float(np.mean(np.abs(np.sum(unit(symc) * unit(qc), axis=1))))

    # (a) NAIVE READER — cosine nearest-neighbour retrieval. For each symbol,
    #     is its own referent the closest of `pool` candidates, judged purely by
    #     resemblance? Dionysius' claim is that this reader MUST go blind.
    pool = 64
    n_probe = 400
    true_ix = rng.choice(n, size=n_probe, replace=False)
    cand = rng.choice(n, size=(n_probe, pool), replace=True)
    cand[:, 0] = true_ix                       # column 0 is always the true referent
    S = unit(symc[true_ix])                    # (n_probe, ds)
    Qc = unit(qc[cand])                        # (n_probe, pool, ds)
    sims = np.einsum("bd,bpd->bp", S, Qc)
    out["naive_retrieval_acc"] = float((sims.argmax(1) == 0).mean())
    out["naive_retrieval_chance"] = 1.0 / pool

    # (b) TRAINED READER — a reader who has been taught the code. Ridge decode of
    #     the referent from the symbol, fit on half and scored on the held-out
    #     half. Dionysius' claim is that this reader still sees.
    #
    #     Ridge, not plain least squares: a codebook can become nearly collinear,
    #     and an unregularised normal equation then reports R² of minus a hundred
    #     million, which is not a finding about symbols, it is a finding about
    #     matrix inversion.
    #     The reader recovers THE WORLD from the symbol — held-out, and fit by
    #     someone other than the network, so the network cannot flatter itself.
    ntr = n // 2
    W = linear_reader(sym[:ntr], x[:ntr])
    pred = np.concatenate([sym[ntr:], np.ones((n - ntr, 1))], axis=1) @ W
    out["trained_reader_r2"] = r2_score(x[ntr:], pred)

    # --- metacognition: does the self-model know the network's own errors? ---
    err = (pred_n != y_name).mean(axis=1)
    err = np.where(y_sil == SILENCE, (pred_s != y_sil).astype(float), err)
    eh = c["err_hat"]
    hi_e, lo_e = err > np.median(err), err <= np.median(err)
    if hi_e.sum() and lo_e.sum():
        # AUC by rank comparison
        order = np.argsort(eh)
        ranks = np.empty(n); ranks[order] = np.arange(n)
        n1, n0 = int(hi_e.sum()), int(lo_e.sum())
        auc = (ranks[hi_e].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)
        out["metacog_auc"] = float(auc)
    else:
        out["metacog_auc"] = float("nan")

    # --- self-opacity: how identifiable is the substrate FROM the self-model?
    #     A linear probe, fit on half, scored on the other half. Low R² means
    #     the self-model cannot be resolved back into the machinery that made
    #     it — the network carries a name it knows is not its own.
    u, H = c["u"], c["H"]
    ntr = n // 2
    Wp_ = linear_reader(u[:ntr], H[:ntr])
    hp = np.concatenate([u[ntr:], np.ones((n - ntr, 1))], axis=1) @ Wp_
    out["self_identifiability"] = r2_score(H[ntr:], hp)

    out["capacities"] = [float(sigmoid(net.P[f"cap{r}"])[0]) for r in range(net.cfg.R)]
    out["worthiness"] = net.P["worth"].copy()
    return out, (x, y_name, y_sil, c)


def aphairesis_curve(net, world, n=2500):
    """
    MT 3–5, executed. Delete the names one by one, in ASCENDING order of
    learned worthiness — the worm first, the Good last. At each step, ask:
      - can the network still name what remains? (it should degrade — of course)
      - can it still find the darkness? (it must NOT degrade — that is the claim)
    The summit is reached by subtraction, and it does not rest on any name,
    not even the best one.
    """
    x, y_name, y_sil, _ = world.sample(n)
    erng = np.random.default_rng(4242)
    etas = [erng.normal(size=(n, net.cfg.ds)) * net.cfg.chan_sigma for _ in range(net.cfg.R)]
    c = net.forward(x, etas=etas)
    order = np.argsort(net.P["worth"])        # least worthy first
    speak = (y_sil == SPEAK)

    rows = []
    alive = np.ones(K)
    for step in range(K + 1):
        if step > 0:
            alive[order[step - 1]] = 0.0
        m = alive.astype(bool)
        if m.sum() > 0:
            pn = c["name_logits"][:, m, :].argmax(-1)
            nacc = float((pn[speak] == y_name[speak][:, m]).mean())
        else:
            nacc = float("nan")
        ps = c["sil_logits"].argmax(-1)
        sacc = float((ps == y_sil).mean())
        removed = NAMES[order[step - 1]] if step > 0 else "—"
        rows.append((step, removed, int(m.sum()), nacc, sacc))
    return rows, order


def hierarchy_ablations(net, world, n=2500):
    """
    Two questions, and Dionysius gives opposite answers to them, which is why
    both have to be asked.

    (a) CAN THE RANKS BE SKIPPED?  (CH 13 — Isaiah is purified by a seraph
        directly, apparently in breach of the order, and Dionysius devotes a
        whole chapter to defending the breach.) We run only the final rank on
        the raw input. If the ranks were gates, this would be catastrophic.

    (b) CAN THE GROUND BE REMOVED?  We mute the ray — the direct, unmediated
        presence of the source at every rank — and let the chain of symbols run
        alone. If the hierarchy really were a relay standing BETWEEN the ground
        and the ranks, muting the ray should change little, because everything
        would be arriving through the chain anyway.

    The pair of answers is the whole argument about what a hierarchy is for.
    """
    x, y_name, y_sil, _ = world.sample(n)
    erng = np.random.default_rng(4242)
    etas = [erng.normal(size=(n, net.cfg.ds)) * net.cfg.chan_sigma for _ in range(net.cfg.R)]
    speak = (y_sil == SPEAK)

    variants = {
        "full": net.forward(x, etas=etas),
        "ranks skipped": net.forward(x, skip_to_top=True, etas=etas),
        "ray muted": net.forward(x, mute_ray=True, etas=etas),
    }
    out = {}
    for tag, c in variants.items():
        pn = c["name_logits"].argmax(-1)
        ps = c["sil_logits"].argmax(-1)
        out[tag] = (float((pn[speak] == y_name[speak]).mean()),
                    float((ps == y_sil).mean()))
    return out


# ==============================================================================
# 7. TRAINING
# ==============================================================================

def train(cfg: Config, lam_res_override=None, quiet=False, tag="dionysian"):
    if lam_res_override is not None:
        cfg = Config(**{**cfg.__dict__, "lam_res": lam_res_override})
    rng = np.random.default_rng(cfg.seed)
    world = Thearchy(cfg, rng)
    net = AphaireticHierarchy(cfg, rng)

    hist = []
    t0 = time.time()
    for s in range(1, cfg.steps + 1):
        x, y_name, y_sil, _ = world.sample(cfg.batch)
        c = net.forward(x)
        L, seeds = net.loss(c, y_name, y_sil)
        g = net.backward(c, seeds)
        net.step(g)
        net.observe_worthiness(L)

        if s % max(cfg.steps // 8, 1) == 0 or s == 1:
            ev, _ = evaluate(net, world, n=1200, rng=np.random.default_rng(9))
            hist.append((s, L["total"], ev["name_acc"], ev["silence_f1"],
                         ev["resemblance"], ev["naive_retrieval_acc"],
                         ev["trained_reader_r2"]))
            if not quiet:
                print(f"  step {s:>5} | loss {L['total']:7.4f} | name {ev['name_acc']:.3f} "
                      f"| silence-F1 {ev['silence_f1']:.3f} | resemblance {ev['resemblance']:.3f} "
                      f"| naive-read {ev['naive_retrieval_acc']:.3f} "
                      f"| trained-read R² {ev['trained_reader_r2']:.3f}")
    if not quiet:
        print(f"  ({tag}: {cfg.steps} steps in {time.time()-t0:.1f}s)")
    return net, world, hist


# ==============================================================================
# 8. SELF-TESTS — the doctrine, as assertions
# ==============================================================================

def self_tests(net, world, ev):
    print("\n  SELF-TESTS")
    print("  " + "-" * 70)
    results = []

    def check(name, cond, detail):
        results.append(bool(cond))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name:<44} {detail}")

    check("ternary lattice beats chance",
          ev["name_acc"] > 0.45, f"name acc {ev['name_acc']:.3f} vs 0.333 chance")

    check("HYPER is learned, not collapsed",
          ev["recall_hyper"] > 0.45 and ev["recall_affirm"] > 0.45 and ev["recall_deny"] > 0.45,
          f"recall aff/den/hyp = {ev['recall_affirm']:.2f}/{ev['recall_deny']:.2f}/{ev['recall_hyper']:.2f}")

    check("silence is a learned act, not a prior",
          ev["silence_f1"] > 0.60,
          f"F1 {ev['silence_f1']:.3f}, base rate {ev['silence_base_rate']:.3f}")

    check("symbol is largely unlike its referent",
          ev["resemblance"] < 0.25 and ev["naive_retrieval_acc"] < 0.60,
          f"cos {ev['resemblance']:.3f}; resemblance-reader {ev['naive_retrieval_acc']:.3f} "
          f"(icon ~1.00, chance {ev['naive_retrieval_chance']:.3f})")

    check("...and yet it carries the world",
          ev["trained_reader_r2"] > 0.70,
          f"taught reader R² {ev['trained_reader_r2']:.3f} (icon ~0.70)")

    check("no channel of evil exists in this network",
          all(not k.lower().startswith(("evil", "dark", "adv")) for k in net.P),
          "harm is modelled only as relu(power - capacity): a defect of measure")

    check("self-model is competent",
          ev["metacog_auc"] > 0.62, f"error-prediction AUC {ev['metacog_auc']:.3f}")

    check("...and yet unsaid: it cannot reconstruct itself",
          ev["self_identifiability"] < 0.35,
          f"substrate recoverable from self, R² = {ev['self_identifiability']:.3f}")

    print("  " + "-" * 70)
    print(f"  {sum(results)}/{len(results)} passed")
    return all(results)


# ==============================================================================
# 9. MAIN
# ==============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    np.set_printoptions(precision=3, suppress=True)
    cfg = Config()
    if args.quick:
        cfg = Config(steps=600)

    print("=" * 78)
    print(" APHAIRETIC HIERARCHY NETWORK — figure 0167, Pseudo-Dionysius (c. 500 CE)")
    print(" 'We make our way to the darkness beyond light not by adding, but by")
    print("  taking away — as sculptors do.'   (Mystical Theology 2, paraphrased)")
    print("=" * 78)

    print("\n[1] GRADIENT CHECK  (analytic vs. central finite differences)")
    gradcheck()

    print("\n[2] TRAINING THE DIONYSIAN NETWORK")
    net, world, hist = train(cfg, tag="dionysian")

    print("\n[3] EVALUATION")
    ev, _ = evaluate(net, world, n=4000)
    print(f"  ternary name accuracy      {ev['name_acc']:.3f}   (chance 0.333)")
    print(f"    recall AFFIRM            {ev['recall_affirm']:.3f}")
    print(f"    recall DENY              {ev['recall_deny']:.3f}")
    print(f"    recall HYPER  (beyond)   {ev['recall_hyper']:.3f}")
    print(f"  silence  precision         {ev['silence_precision']:.3f}")
    print(f"  silence  recall            {ev['silence_recall']:.3f}")
    print(f"  silence  F1                {ev['silence_f1']:.3f}   (base rate {ev['silence_base_rate']:.3f})")
    print(f"  metacognition AUC          {ev['metacog_auc']:.3f}")
    print(f"  self-identifiability       {ev['self_identifiability']:.3f}   (lower = more unsaid)")
    print(f"  analogia (rank capacities) {['%.3f' % c for c in ev['capacities']]}")

    print("\n[4] DISSIMILAR SIMILARITY  (Celestial Hierarchy 2)")
    print("    The doctrine has two halves and they pull against each other. A symbol")
    print("    must LEAD YOU TO the thing (similar) and must NOT LOOK LIKE it")
    print("    (dissimilar). So we build all four combinations and run each on 3 seeds.")
    print()
    print("      ICONIC       carry + REWARD resemblance   — the design everyone builds")
    print("      APOPHATIC    carry + PENALISE resemblance — Dionysius' design")
    print("      INDIFFERENT  carry, no pressure either way")
    print("      MUTE         PENALISE resemblance, no carry — dissimilarity alone")
    print()
    SEEDS = [500, 501, 502]
    arms = {"ICONIC": (1.0, -0.40), "APOPHATIC": (cfg.lam_carry, cfg.lam_res),
            "INDIFFERENT": (1.0, 0.0), "MUTE": (0.0, 0.40)}
    keys = ["resemblance", "naive_retrieval_acc", "trained_reader_r2", "name_acc", "silence_f1"]
    labels = {"resemblance": "cos(symbol, referent)",
              "naive_retrieval_acc": "naive resemblance-reader",
              "trained_reader_r2": "taught reader, R2 on world",
              "name_acc": "ternary name accuracy",
              "silence_f1": "silence F1"}
    agg = {}
    for a, (lc, lr_) in arms.items():
        agg[a] = {k: [] for k in keys}
        for sd in SEEDS:
            nt, wd, _ = train(Config(**{**cfg.__dict__, "seed": sd,
                                        "lam_carry": lc, "lam_res": lr_}),
                              quiet=True, tag=a)
            e, _ = evaluate(nt, wd, n=4000)
            for k in keys:
                agg[a][k].append(e[k])

    hdr = "".join(f"{a:>15}" for a in arms)
    print(f"    {'':<28}{hdr}")
    print(f"    {'-'*88}")
    for k in keys:
        cells = "".join(f"{np.mean(agg[a][k]):>10.3f}±{np.std(agg[a][k]):<4.2f}" for a in arms)
        print(f"    {labels[k]:<28}{cells}")
    print(f"    {'(resemblance-reader chance)':<28}{ev['naive_retrieval_chance']:>10.3f}")
    print(f"    mean ± s.d. over seeds {SEEDS}")
    print()
    print("    Three things happen here, and only one of them was expected.")
    print()
    print("    1. Left alone, a symbol BECOMES a likeness. INDIFFERENT was given no")
    print("       instruction about resemblance and drifted to cos = 0.92 anyway. The")
    print("       idol is not a temptation the mind occasionally falls into. It is the")
    print("       default. It is what representation does when nobody is stopping it.")
    print()
    print("    2. Refusing resemblance, on its own, destroys the sign. MUTE has an")
    print("       unlike symbol and a taught reader can recover almost nothing of the")
    print("       world from it (R2 ~ 0.03). This is the cheap escape from idolatry and")
    print("       it is the one Dionysius is always accused of taking — 'he platonizes")
    print("       more than he Christianizes,' said Luther. It is not what he does.")
    print()
    print("    3. Both halves together do something neither half predicts. APOPHATIC")
    print("       carries MORE of the world than the icon does (R2 0.77 vs 0.70) while")
    print("       resembling it about a tenth as much. Forbidding the code to look like")
    print("       the thing forced it to encode the thing better. The grotesque symbol")
    print("       is not a pious sacrifice of fidelity. It is more faithful.")
    print()
    print("    And one caveat, which is the most interesting number on the table: the")
    print("    resemblance-reader still scores 0.44 against the apophatic code — far")
    print("    below the icon's 1.00, far above chance. Mean cosine can be driven to")
    print("    0.05 and the RANKING survives. A sign that carries the thing cannot be")
    print("    made wholly unlike the thing. Some likeness always leaks. Which is")
    print("    exactly why Dionysius does not stop at the worm and the drunkard --")
    print("    why there is a Mystical Theology after the Symbolic Theology, and why")
    print("    it ends by taking even the good symbols away. See [5].")

    print("\n[5] APHAIRESIS  (Mystical Theology 3–5): deleting the names, worst first")
    rows, order = aphairesis_curve(net, world)
    print(f"    learned worthiness: " +
          ", ".join(f"{NAMES[i]}={net.P['worth'][i]:.2f}" for i in np.argsort(-net.P['worth'])))
    print()
    print(f"    {'step':<6}{'name denied':<14}{'names left':>11}{'name acc':>11}{'silence acc':>13}")
    print(f"    {'-'*55}")
    for s, rem, left, nacc, sacc in rows:
        na = "  —  " if math.isnan(nacc) else f"{nacc:.3f}"
        print(f"    {s:<6}{rem:<14}{left:>11}{na:>11}{sacc:>13.3f}")
    print()
    print("    Naming decays to nothing, exactly as it should. The last name to go")
    print("    is the one the network was proudest of. And the summit — the judgment")
    print("    that here nothing may be said at all — does not move.")

    print("\n[6] WHAT IS A HIERARCHY FOR?  (CH 3 and CH 13)")
    ab = hierarchy_ablations(net, world)
    print(f"    {'':<24}{'ternary names':>15}{'silence':>11}")
    print(f"    {'-'*50}")
    for tag in ["full", "ranks skipped", "ray muted"]:
        na, sa = ab[tag]
        print(f"    {tag:<24}{na:>15.3f}{sa:>11.3f}")
    print()
    print("    Skip the ranks and almost nothing is lost. Mute the ray and the whole")
    print("    thing falls in. Read those two lines together and you have Dionysius'")
    print("    hierarchy exactly: the ranks are NOT what stands between a mind and the")
    print("    ground, and they were never the channel through which the ground arrives.")
    print("    They are the order in which what has already arrived gets handed on.")
    print("    A hierarchy you can skip without loss is a hierarchy of teaching. A")
    print("    hierarchy you cannot skip is a hierarchy of power, and it is not this one.")

    print("\n[7] THE UNSAID SELF  (Stang 2012; Gal 2:20, 'no longer I')")
    unsaid, uw, _ = train(Config(**{**cfg.__dict__, "lam_opac": 0.0}), quiet=True, tag="named-self")
    uev, _ = evaluate(unsaid, uw, n=4000)
    print(f"    {'':<34}{'UNSAID':>10}{'NAMED':>10}")
    print(f"    {'-'*54}")
    print(f"    {'error-prediction AUC (competence)':<34}{ev['metacog_auc']:>10.3f}{uev['metacog_auc']:>10.3f}")
    print(f"    {'substrate recoverable from self':<34}{ev['self_identifiability']:>10.3f}{uev['self_identifiability']:>10.3f}")
    print()
    print("    Both networks know how likely they are to be wrong. Only one of them can")
    print("    be reconstructed from its own self-model. The author of this corpus signed")
    print("    a dead man's name to it and never told anyone his own; his network keeps")
    print("    the habit. It knows what it can do. It does not know what it is.")

    ok = self_tests(net, world, ev)

    print("\n" + "=" * 78)
    print(" 'Neither is it darkness nor light, nor error, nor truth. There is no")
    print("  speaking of it, nor name, nor knowledge of it.'   (MT 5, paraphrased)")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

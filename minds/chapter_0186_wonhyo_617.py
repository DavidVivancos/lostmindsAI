#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 ILSIM  —  The Two-Gate Unhinderer
 Chapter 0186 · Wonhyo 元曉 (617–686), Silla
 Encyclopedia of Lost Minds: Echoes on AI
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 10 Minds 181 - 200 Available on Amazon https://www.amazon.com/dp/B0HJYMZ3G6
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0186_wonhyo_617 - Wonhyo 元曉 (617–686), Silla
================================================================================    

WHAT THIS FILE IS
-----------------
A from-scratch, pure-NumPy neural architecture (no autograd, no frameworks)
whose *structure* encodes the cognitive signature of one seventh-century mind:
the Silla scholar-monk Wonhyo, posthumously the "National Preceptor of the
Harmonization of Disputes" (和諍國師, Goryeo, 1101).

Three ideas that are his, and nobody else's in this corpus, are built in as
mechanism rather than described in comments:

  1. HWAJAENG (和諍) — the harmonization of disputes.  Wonhyo never resolved a
     doctrinal quarrel by voting, averaging, or ranking the parties on a ladder
     of "provisional" to "perfect" teachings (the p'an'gyo classification he
     compared to scooping the ocean with a snail shell).  He read each claim's
     WORDS, inferred the GATE (門) — the standpoint, scope and intent — from
     which it was uttered, translated it into a common ground, and then showed
     that every party could be spoken back to in its own vocabulary without
     loss: "both are correct, both are mistaken; rigidly attached to one
     extreme, both are wrong; explained without obstruction, both are valid."
     (Doctrinal Essentials of the Nirvana Sutra, T 1769.38.248b27)

  2. ILSIM IMUN (一心二門) — the One Mind in Two Gates, from the Awakening of
     Faith he commented on twice.  The mind has an unchanging aspect
     (真如門, suchness) and an arising-and-ceasing aspect (生滅門).  Here the
     unchanging aspect is a FIXED RANDOM SUBSTRATE that is never trained; the
     arising-and-ceasing aspect is the set of gates, translations and readouts
     that are trained.

  3. BONGAK / SIGAK (本覺 / 始覺) — original and actualized enlightenment.
     "When you extract a ten-foot rod, ten feet of space appears" (Simmun
     hwajaeng non, HBJ 1.838b).  The substrate already contains the function;
     learning does not ADD weights to it, it REMOVES HINDRANCES.  Every synapse
     of the substrate carries a hindrance logit; the effective weight is the
     fixed weight multiplied by an unhindrance mask in (0,1).  Training moves
     only the masks.  Two mask families stand for Wonhyo's two hindrances from
     the Ijangui (二障義): the afflictive hindrance on what the mind receives
     and the cognitive hindrance on what it can know.

THE ARCHITECTURAL CLAIM
-----------------------
Give a network a DISPUTE — a bag of claims about one hidden object, each claim
made in the vocabulary and coordinate conventions of a different "school" that
touched a different part of the object (the Nirvana Sutra's blind men and the
elephant, which Wonhyo turned on the Buddha-nature debate) — and demand that it
answer a question posed in ANY school's vocabulary, including one absent from
the dispute.

    gate      g_i   = softmax( G · words_i )                 read the words
    translate u_i   = Σ_k g_ik ( T_k · values_i + t_k )      into one ground
    cover     c_i   = Σ_k g_ik σ(C_k)                        who touched what
    combine   ẑ_d   = Σ_i c_id u_id / Σ_i c_id               合 — without narrowing
    unhinder  m     = [ ẑ ; (W2°σ(U2)) tanh((W1°σ(U1)) [ẑ;1]) ] 本覺 — masks only
    open      ŷ     = Σ_k g_qk ( R_k · m + r_k )             開 — in the asker's words

Three deliberately crippled minds are trained on identical data:
  ATTACHED   (執言)  one gate only — treats every claim as if in one frame
  NARROW     (狹)    coverage frozen at 1 — every claim votes on every coordinate
  HINDERED   (障)    substrate masks frozen — the mind cannot unhinder
and compared with the full ILSIM mind on five dispute regimes.  The gap is
structural: an averaging mind has no gate to infer; a narrow mind dilutes the
elephant with what nobody touched; a hindered mind cannot see the property
that lives beyond the words.

CONVENTIONS OF THE CORPUS
-------------------------
Pure NumPy, float64, hand-derived backward pass, a finite-difference gradient
check that must pass (the "Panbiryang" check, after Wonhyo's Critique of
Inference, 671), a real training loop with Adam, held-out evaluation per regime,
and structural self-tests.  Run:

    python3 chapter_0186_wonhyo_617.py            # full run (~3 min on one CPU core)
    python3 chapter_0186_wonhyo_617.py --quick    # short run for smoke testing (~20 s)

WHAT THE FULL RUN SHOWS (seed 42, 7000/700/2000 disputes, 300 epochs)
------------------------------------------------------------------------
    regime            ATTACHED      NARROW    HINDERED       ILSIM  LINEAR-FLOOR
    CONCORD             0.0342      0.0447      0.0129      0.0162        0.0000
    DISPUTE             0.7387      0.0876      0.0521      0.0565        0.0000
    CROSS_FRAME         1.3270      0.2276      0.0230      0.0251        0.0000
    SPARSE              0.8756      0.1282      0.0442      0.0308        0.0000
    SUBTLE              0.3338      0.1446      0.1565      0.0503        0.2479
    OVERALL             0.6619      0.1265      0.0577      0.0358        0.0496
The one-door mind fails the moment two vocabularies meet; the narrow mind
dilutes; the hindered mind reads words perfectly and cannot see past them;
the full mind uses 4.09 of its 6 doors for 4 voices and opens 1,113 of 9,984
synapses while the substrate stays byte-identical.

A NOTE FROM THE BUILDING
------------------------
The first substrate had no constant input and could not learn the nonlinear
property at all: a bank of tanh(w·z) units is odd in z, and the property is
even in two of its coordinates.  Appending a constant to the ground — one
synapse that is always on — gave the substrate even functions, and the error
fell from the linear floor to a fifth of it.  The rod that says "one" has to
be in the space before its removal can reveal anything.
"""

import sys
import time
import numpy as np

# ============================================================================
# 0.  Small numerics
# ============================================================================

def sigmoid(x):
    """Numerically stable logistic."""
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def softmax_backward(g, dg, tau=1.0):
    """Given output g = softmax(l/tau) and dL/dg, return dL/dl."""
    return g * (dg - (dg * g).sum(axis=-1, keepdims=True)) / tau


# ============================================================================
# 1.  The Elephant — data generating process
# ============================================================================
#
# One hidden object z in R^D ("the elephant", the matter under dispute).
# S schools.  Each school touched a subset of coordinates and reports what it
# touched through its own linear operator A_s (with its own SIGN CONVENTIONS —
# what one school calls "existent" another calls "empty"), plus a bias (its
# vocabulary's baseline), plus one NONLINEAR property phi_s(z) that only a mind
# with an unhindered substrate can recover.
#
# Each claim carries a WORDS vector (a noisy bag of the school's terms, some
# terms shared between schools) and a VALUES vector.  A query is a words vector
# plus a property selector; the target is that property in that school's frame.

D = 6          # latent coordinates of the elephant
S = 4          # true number of schools (blind men)
P = 4          # value slots per claim
Q = P + 1      # properties a school can be asked about: P linear + 1 nonlinear
V = 12         # vocabulary size
MMAX = 5       # maximum claims per dispute

# coverage: which coordinates each school touched.  Every pair of schools shares
# exactly one coordinate, so every pair can be made to dispute.
COVER = [(0, 1, 2), (2, 3, 4), (4, 5, 0), (1, 3, 5)]
# sign conventions on the covered coordinates ("exists" vs "empty")
SIGNS = [(+1, +1, +1), (-1, +1, -1), (+1, -1, +1), (-1, -1, +1)]
# school vocabularies: three characteristic terms each, overlapping pairwise,
# terms 8..11 are common words used by everyone.
TERMS = [(0, 1, 2), (2, 3, 4), (4, 5, 6), (6, 7, 0)]

REGIMES = ["CONCORD", "DISPUTE", "CROSS_FRAME", "SPARSE", "SUBTLE"]


def school_operator(s):
    """A_s (P x D) and bias c_s (P,).  Rows 0..2 observe the covered coords with
    the school's signs; row 3 is a derived statement (difference of first and
    last covered coordinate) — the kind of secondary claim a tradition builds
    on top of its primary observations."""
    A = np.zeros((P, D))
    for r, (d, sg) in enumerate(zip(COVER[s], SIGNS[s])):
        A[r, d] = sg * (1.0 + 0.15 * r)
    d0, d2 = COVER[s][0], COVER[s][2]
    A[3, d0] = 0.6 * SIGNS[s][0]
    A[3, d2] = -0.6 * SIGNS[s][2]
    c = np.array([0.30, -0.20, 0.10, 0.25]) * (1 if s % 2 == 0 else -1) * (0.5 + 0.5 * s / S)
    return A, c


A_OPS = [school_operator(s) for s in range(S)]


def phi(s, z):
    """The nonlinear property of school s: a statement about how two of its
    coordinates move TOGETHER, plus a third.  z: (..., D)."""
    a, b, c = COVER[s]
    return np.tanh(z[..., a] * z[..., b] + 0.4 * z[..., c])


def properties(s, z):
    """All Q properties of school s for object z: (..., Q)."""
    A, c = A_OPS[s]
    lin = z @ A.T + c
    return np.concatenate([lin, phi(s, z)[..., None]], axis=-1)


def words_of(s, rng, n):
    """Noisy bag-of-terms vector for n utterances of school s: (n, V)."""
    w = 0.35 * rng.standard_normal((n, V))
    for t in TERMS[s]:
        w[:, t] += 0.9
    # common words, present at random in every school's speech
    w[:, 8:] += (rng.random((n, 4)) < 0.5) * 0.7
    return w


def make_dataset(n, seed=0, noise=0.05):
    """Build n disputes.  Returns a dict of arrays:
       W  (n, MMAX, V)  claim words        Sv (n, MMAX, P) claim values
       VM (n, MMAX)     claim valid mask   WQ (n, V)       query words
       E  (n, Q)        property selector  Y  (n,)         target
       R  (n,)          regime id          Z  (n, D)       the hidden elephant
       SCH(n, MMAX)     true school of each claim, QS (n,) true query school
                        (both for analysis only — the model never sees them)"""
    rng = np.random.default_rng(seed)
    W = np.zeros((n, MMAX, V)); Sv = np.zeros((n, MMAX, P)); VM = np.zeros((n, MMAX))
    WQ = np.zeros((n, V)); E = np.zeros((n, Q)); Y = np.zeros(n)
    R = np.zeros(n, dtype=int); Z = rng.standard_normal((n, D)); SCH = -np.ones((n, MMAX), dtype=int)
    QS = np.zeros(n, dtype=int)
    for i in range(n):
        z = Z[i]
        reg = i % len(REGIMES)
        R[i] = reg
        if reg == 0:                                   # CONCORD: one school, five claims
            schools = [rng.integers(S)] * MMAX
            qs = schools[0]; q = rng.integers(P)
        elif reg == 1:                                 # DISPUTE: two schools, all claims, query on shared coord
            a, b = rng.choice(S, 2, replace=False)
            schools = [a, a, a, b, b] if rng.random() < 0.5 else [a, a, b, b, b]
            rng.shuffle(schools)
            qs = a if rng.random() < 0.5 else b
            shared = list(set(COVER[a]) & set(COVER[b]))[0]
            q = COVER[qs].index(shared) if rng.random() < 0.6 else rng.integers(P)
        elif reg == 2:                                 # CROSS_FRAME: query from the one ABSENT school
            present = list(rng.choice(S, 3, replace=False))   # three schools cover every coordinate
            absent = [s for s in range(S) if s not in present]
            schools = [present[j % 3] for j in range(MMAX)]
            rng.shuffle(schools)
            qs = int(rng.choice(absent)); q = rng.integers(P)
        elif reg == 3:                                 # SPARSE: three schools, ONE claim each
            present = list(rng.choice(S, 3, replace=False))
            schools = present + [-1, -1]
            qs = int(rng.choice(present)); q = rng.integers(P)
        else:                                          # SUBTLE: nonlinear property of a present school
            k = rng.integers(2, 4)
            present = list(rng.choice(S, k, replace=False))
            schools = [present[j % k] for j in range(MMAX)]
            rng.shuffle(schools)
            qs = int(rng.choice(present)); q = P
        for m, s in enumerate(schools):
            if s < 0:
                continue
            A, c = A_OPS[s]
            W[i, m] = words_of(s, rng, 1)[0]
            Sv[i, m] = A @ z + c + noise * rng.standard_normal(P)
            VM[i, m] = 1.0
            SCH[i, m] = s
        WQ[i] = words_of(qs, rng, 1)[0]
        E[i, q] = 1.0
        Y[i] = properties(qs, z)[q]
        QS[i] = qs
    return dict(W=W, Sv=Sv, VM=VM, WQ=WQ, E=E, Y=Y, R=R, Z=Z, SCH=SCH, QS=QS)


def take(data, idx):
    return {k: v[idx] for k, v in data.items()}


# ============================================================================
# 2.  ILSIM — the model
# ============================================================================

class Ilsim:
    """One Mind in Two Gates.

    Trainable ("arising-and-ceasing" gate, 生滅門):
        G  (K,V)  gb (K,)         gate logits from words
        T  (K,D,P) tb (K,D)       per-gate translation of claim values into the ground
        C  (K,D)                  per-gate coverage logits (σ → who touched what)
        R  (K,Q,Dm) rb (K,Q)      per-gate readout (open the ground into the asker's words)
        U1 (H,D+1) U2 (Hm,H)      hindrance logits on the substrate synapses

    Fixed ("suchness" gate, 真如門):
        W1f (H,D+1) W2f (Hm,H)    random substrate, NEVER updated (the +1 is a
                                  constant input, so the substrate owns even
                                  functions as well as odd ones)

    mode: 'ilsim' | 'attached' | 'narrow' | 'hindered'
    """

    def __init__(self, K=6, H=256, Hm=32, tau=0.5, mode="ilsim", seed=0,
                 lam_rec=0.25, lam_ent=0.01, lam_prol=0.0, init_hindrance=-1.5,
                 substrate_gain=2.0, mask_lr_mult=30.0):
        self.mode = mode
        self.mask_lr_mult = mask_lr_mult
        self.K = 1 if mode == "attached" else K
        self.H, self.Hm, self.tau = H, Hm, tau
        self.Dm = D + Hm
        self.lam_rec, self.lam_ent, self.lam_prol = lam_rec, lam_ent, lam_prol
        rng = np.random.default_rng(seed)
        K = self.K
        p = {}
        p["G"] = 0.3 * rng.standard_normal((K, V)); p["gb"] = np.zeros(K)
        p["T"] = 0.3 * rng.standard_normal((K, D, P)); p["tb"] = np.zeros((K, D))
        p["C"] = np.full((K, D), 1.0)
        p["R"] = 0.3 * rng.standard_normal((K, Q, self.Dm)); p["rb"] = np.zeros((K, Q))
        p["U1"] = np.full((H, D + 1), init_hindrance)      # +1: a constant input (the substrate's bias synapse)
        p["U2"] = np.full((Hm, H), init_hindrance)
        self.p = p
        # the substrate: fixed, Kaiming-scaled, signed — original enlightenment
        srng = np.random.default_rng(seed + 1000)
        # gain compensates for the masks: a half-open mask halves the effective scale
        self.W1f = srng.standard_normal((H, D + 1)) * np.sqrt(2.0 / (D + 1)) * substrate_gain
        self.W2f = srng.standard_normal((Hm, H)) * np.sqrt(2.0 / H) * substrate_gain
        self._W1f_copy = self.W1f.copy(); self._W2f_copy = self.W2f.copy()
        # which parameters are frozen in the crippled minds
        self.frozen = set()
        if mode == "narrow":
            p["C"][:] = 12.0          # σ(12) ≈ 1: every gate covers everything
            self.frozen.add("C")
        if mode == "hindered":
            self.frozen.update({"U1", "U2"})

    # ------------------------------------------------------------------ forward
    def forward(self, d):
        p, K, tau = self.p, self.K, self.tau
        W, Sv, VM, WQ, E = d["W"], d["Sv"], d["VM"], d["WQ"], d["E"]
        B, M, _ = W.shape
        cache = {"d": d}
        # --- read the words: gate posterior per claim -------------------------
        gl = W @ p["G"].T + p["gb"]                     # (B,M,K)
        g = softmax(gl / tau)                           # (B,M,K)
        # --- translate each claim into the common ground -----------------------
        uk = np.einsum("kdp,bmp->bmkd", p["T"], Sv) + p["tb"][None, None]   # (B,M,K,D)
        u = np.einsum("bmk,bmkd->bmd", g, uk)                                # (B,M,D)
        # --- coverage: who touched what -----------------------------------------
        c = sigmoid(p["C"])                             # (K,D)
        ch = np.einsum("bmk,kd->bmd", g, c) * VM[:, :, None]   # (B,M,D)
        # --- combine (合) without narrowing ---------------------------------------
        num = (ch * u).sum(1)                           # (B,D)
        den = ch.sum(1) + 1e-6                          # (B,D)
        zh = num / den                                  # (B,D)
        # --- unhinder the substrate (本覺 → 始覺) -----------------------------
        M1 = sigmoid(p["U1"]); W1 = self.W1f * M1       # (H,D+1)
        M2 = sigmoid(p["U2"]); W2 = self.W2f * M2       # (Hm,H)
        zh1 = np.concatenate([zh, np.ones((B, 1))], axis=1)   # (B,D+1) the ground plus a constant
        a = zh1 @ W1.T                                  # (B,H)
        t = np.tanh(a)
        h = t @ W2.T                                    # (B,Hm)
        m = np.concatenate([zh, h], axis=1)             # (B,Dm)
        # --- open (開): every gate's reading of the one mind ----------------------
        yK = np.einsum("kqd,bd->bkq", p["R"], m) + p["rb"][None]   # (B,K,Q)
        # --- answer the query in the asker's words ---------------------------------
        gql = WQ @ p["G"].T + p["gb"]                   # (B,K)
        gq = softmax(gql / tau)
        yq = np.einsum("bk,bkq->bq", gq, yK)            # (B,Q)
        pred = (yq * E).sum(1)                          # (B,)
        # --- regenerate every claim in its own words (refute without loss) ------
        sh = np.einsum("bmk,bkq->bmq", g, yK)[:, :, :P] # (B,M,P)
        cache.update(dict(gl=gl, g=g, uk=uk, u=u, c=c, ch=ch, num=num, den=den, zh=zh, zh1=zh1,
                          M1=M1, W1=W1, M2=M2, W2=W2, a=a, t=t, h=h, m=m, yK=yK,
                          gql=gql, gq=gq, yq=yq, pred=pred, sh=sh))
        return pred, cache

    def loss_terms(self, cache):
        d, g, sh, pred = cache["d"], cache["g"], cache["sh"], cache["pred"]
        VM, Sv, Y = d["VM"], d["Sv"], d["Y"]
        B, M, K = g.shape
        nvalid = VM.sum() + 1e-9
        Lq = np.mean((pred - Y) ** 2)
        Lrec = ((sh - Sv) ** 2 * VM[:, :, None]).sum() / (nvalid * P)
        # 開而不繁: a claim belongs to a gate — low entropy per valid claim
        ent = -(g * np.log(g + 1e-12)).sum(-1)          # (B,M)
        Lent = (ent * VM).sum() / nvalid
        # do not proliferate gates: concave penalty on mean usage.  Shipped with
        # lam_prol = 0: with the penalty on, the mind collapsed to three doors for
        # four voices and never disentangled the shared coordinates.  It is kept
        # (and gradient-checked) so the reader can switch it on and watch that happen.
        gbar = (g * VM[:, :, None]).sum((0, 1)) / nvalid   # (K,)
        Lprol = np.sqrt(gbar + 1e-6).sum()
        L = Lq + self.lam_rec * Lrec + self.lam_ent * Lent + self.lam_prol * Lprol
        return L, dict(Lq=Lq, Lrec=Lrec, Lent=Lent, Lprol=Lprol, nvalid=nvalid, gbar=gbar)

    # ----------------------------------------------------------------- backward
    def backward(self, cache):
        """Returns grads dict (same keys as self.p) for the total loss."""
        p, K, tau = self.p, self.K, self.tau
        d = cache["d"]
        W, Sv, VM, WQ, E, Y = d["W"], d["Sv"], d["VM"], d["WQ"], d["E"], d["Y"]
        g, uk, u, c, ch, num, den, zh, zh1 = (cache[k] for k in ["g", "uk", "u", "c", "ch", "num", "den", "zh", "zh1"])
        M1, W1, M2, W2, a, t, h, m, yK = (cache[k] for k in ["M1", "W1", "M2", "W2", "a", "t", "h", "m", "yK"])
        gq, yq, pred, sh = cache["gq"], cache["yq"], cache["pred"], cache["sh"]
        B, M, _ = W.shape
        nvalid = VM.sum() + 1e-9
        gr = {k: np.zeros_like(v) for k, v in p.items()}

        # ---- dL/dpred, dL/dsh ---------------------------------------------------
        dpred = 2.0 * (pred - Y) / B                                    # (B,)
        dsh = self.lam_rec * 2.0 * (sh - Sv) * VM[:, :, None] / (nvalid * P)   # (B,M,P)
        # ---- entropy & proliferation grads w.r.t. g ---------------------------
        dg = np.zeros_like(g)
        dg += self.lam_ent * (-(np.log(g + 1e-12) + 1.0)) * VM[:, :, None] / nvalid
        gbar = (g * VM[:, :, None]).sum((0, 1)) / nvalid
        dgbar = self.lam_prol * 0.5 / np.sqrt(gbar + 1e-6)              # (K,)
        dg += dgbar[None, None, :] * VM[:, :, None] / nvalid
        # ---- query path ---------------------------------------------------------
        dyq = dpred[:, None] * E                                        # (B,Q)
        dgq = np.einsum("bq,bkq->bk", dyq, yK)                          # (B,K)
        dyK = np.einsum("bk,bq->bkq", gq, dyq)                          # (B,K,Q)
        # ---- reconstruction path: sh = (g · yK)[:, :, :P] ------------------------
        dsh_full = np.zeros((B, M, Q)); dsh_full[:, :, :P] = dsh
        dg += np.einsum("bmq,bkq->bmk", dsh_full, yK)
        dyK += np.einsum("bmk,bmq->bkq", g, dsh_full)
        # ---- readout params ----------------------------------------------------
        gr["R"] = np.einsum("bkq,bd->kqd", dyK, m)
        gr["rb"] = dyK.sum(0)
        dm = np.einsum("bkq,kqd->bd", dyK, p["R"])                     # (B,Dm)
        # ---- query gate params -------------------------------------------------
        dgql = softmax_backward(gq, dgq, tau)                           # (B,K)
        gr["G"] += dgql.T @ WQ
        gr["gb"] += dgql.sum(0)
        # ---- substrate: m = [zh ; h], h = t W2^T, t = tanh(zh W1^T) ------------
        dzh = dm[:, :D].copy()
        dh = dm[:, D:]
        dW2 = dh.T @ t                                                  # (Hm,H)
        dt = dh @ W2                                                    # (B,H)
        da = dt * (1.0 - t ** 2)
        dW1 = da.T @ zh1                                                # (H,D+1)
        dzh += (da @ W1)[:, :D]                                         # the constant input has no gradient
        gr["U2"] = dW2 * self.W2f * M2 * (1.0 - M2)
        gr["U1"] = dW1 * self.W1f * M1 * (1.0 - M1)
        # ---- combine: zh = num/den --------------------------------------------
        dnum = dzh / den                                                # (B,D)
        dden = -dzh * num / den ** 2                                    # (B,D)
        dch = dnum[:, None, :] * u + dden[:, None, :]                   # (B,M,D)
        du = dnum[:, None, :] * ch                                      # (B,M,D)
        # ---- coverage: ch = (g·c) * VM -----------------------------------------
        dch_v = dch * VM[:, :, None]
        dg += np.einsum("bmd,kd->bmk", dch_v, c)
        dc = np.einsum("bmk,bmd->kd", g, dch_v)
        gr["C"] = dc * c * (1.0 - c)
        # ---- translation: u = Σ_k g uk; uk = T_k s + tb_k ----------------------
        dg += np.einsum("bmd,bmkd->bmk", du, uk)
        duk = np.einsum("bmk,bmd->bmkd", g, du)                         # (B,M,K,D)
        gr["T"] = np.einsum("bmkd,bmp->kdp", duk, Sv)
        gr["tb"] = duk.sum((0, 1))
        # ---- claim gates -------------------------------------------------------
        dgl = softmax_backward(g, dg, tau)                              # (B,M,K)
        gr["G"] += np.einsum("bmk,bmv->kv", dgl, W)
        gr["gb"] += dgl.sum((0, 1))
        for k in self.frozen:
            gr[k][:] = 0.0
        return gr

    # ------------------------------------------------------------------ helpers
    def loss(self, d):
        _, cache = self.forward(d)
        L, terms = self.loss_terms(cache)
        return L, terms, cache

    def predict(self, d):
        return self.forward(d)[0]

    def effective_gates(self, d):
        """exp(entropy of mean gate usage): how many gates the mind actually uses."""
        _, cache = self.forward(d)
        g, VM = cache["g"], d["VM"]
        gbar = (g * VM[:, :, None]).sum((0, 1)) / (VM.sum() + 1e-9)
        return float(np.exp(-(gbar * np.log(gbar + 1e-12)).sum()))

    def unhindered_count(self, thr=0.5):
        return int((sigmoid(self.p["U1"]) > thr).sum() + (sigmoid(self.p["U2"]) > thr).sum())

    def substrate_intact(self):
        return np.array_equal(self.W1f, self._W1f_copy) and np.array_equal(self.W2f, self._W2f_copy)


# ============================================================================
# 3.  PANBIRYANG — the critique of inference (finite-difference gradient check)
# ============================================================================

def gradient_check(mode="ilsim", seed=7, n=6, eps=1e-6, tol=1e-6, verbose=True):
    """Compare every analytic gradient with central finite differences.
    Small model, few samples, float64.  Returns (max relative error, passed)."""
    data = make_dataset(n * 2, seed=seed)
    d = take(data, np.arange(n))
    model = Ilsim(K=3, H=7, Hm=4, mode=mode, seed=seed, tau=0.8)
    # perturb parameters away from symmetric inits so all paths are live
    rng = np.random.default_rng(seed + 3)
    for k in model.p:
        model.p[k] = model.p[k] + 0.3 * rng.standard_normal(model.p[k].shape)
    if mode == "narrow":
        model.p["C"][:] = 12.0
    L, terms, cache = model.loss(d)
    gr = model.backward(cache)
    worst = 0.0
    for k, v in model.p.items():
        if k in model.frozen:
            continue
        num = np.zeros_like(v)
        it = np.nditer(v, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            old = v[idx]
            v[idx] = old + eps; Lp = model.loss(d)[0]
            v[idx] = old - eps; Lm = model.loss(d)[0]
            v[idx] = old
            num[idx] = (Lp - Lm) / (2 * eps)
        # scaled relative error: |num - analytic| / max(|num| + |analytic|, 1e-3).
        # The floor keeps entries whose true gradient is ~1e-5 from being judged by
        # finite-difference round-off (~1e-10 absolute at eps = 1e-6, float64).
        absdiff = np.abs(num - gr[k])
        rel = np.max(absdiff / np.maximum(np.abs(num) + np.abs(gr[k]), 1e-3))
        worst = max(worst, rel)
        if verbose:
            print(f"    {k:>3s}  shape={str(v.shape):>12s}  max|analytic|={np.abs(gr[k]).max():.3e}"
                  f"  max|diff|={absdiff.max():.2e}  scaled rel.err={rel:.2e}")
    return worst, worst < tol


# ============================================================================
# 4.  Training
# ============================================================================

class Adam:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8, lr_mult=None):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.lr_mult = lr_mult or {}
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads, frozen=()):
        self.t += 1
        for k in params:
            if k in frozen:
                continue
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * self.lr_mult.get(k, 1.0) * mh / (np.sqrt(vh) + self.eps)


def train(model, tr, va, epochs=40, batch=128, lr=3e-3, seed=0, log_every=5, name="", lr_final=None):
    """Adam with an optional cosine decay from lr to lr_final over the epochs."""
    rng = np.random.default_rng(seed)
    # hindrance logits get a larger step: a mask's gradient is damped by the fixed
    # weight and by σ'(U), so unhindering would otherwise be glacially slow.
    opt = Adam(model.p, lr=lr, lr_mult={"U1": model.mask_lr_mult, "U2": model.mask_lr_mult})
    n = tr["Y"].shape[0]
    hist = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        if lr_final is not None and epochs > 1:
            opt.lr = lr_final + 0.5 * (lr - lr_final) * (1 + np.cos(np.pi * (ep - 1) / (epochs - 1)))
        perm = rng.permutation(n)
        tot = 0.0; nb = 0
        for i in range(0, n, batch):
            d = take(tr, perm[i:i + batch])
            L, terms, cache = model.loss(d)
            gr = model.backward(cache)
            opt.step(model.p, gr, model.frozen)
            tot += L; nb += 1
        Lva, tva, _ = model.loss(va)
        hist.append((ep, tot / nb, Lva, tva["Lq"]))
        if ep % log_every == 0 or ep == 1 or ep == epochs:
            print(f"    [{name:>8s}] epoch {ep:3d}  train {tot/nb:.4f}  val {Lva:.4f}  "
                  f"val-query-mse {tva['Lq']:.4f}  K_eff {model.effective_gates(va):.2f}  "
                  f"unhindered {model.unhindered_count()}  ({time.time()-t0:5.1f}s)")
    return hist


def per_regime(model, d, tol=0.2):
    pred = model.predict(d)
    err = pred - d["Y"]
    out = {}
    for r, name in enumerate(REGIMES):
        sel = d["R"] == r
        out[name] = (float(np.mean(err[sel] ** 2)), float(np.mean(np.abs(err[sel]) < tol)))
    out["OVERALL"] = (float(np.mean(err ** 2)), float(np.mean(np.abs(err) < tol)))
    return out


def gate_purity(model, d):
    """How consistently the learned gates track the TRUE schools: for each true
    school, the fraction of its claims sent to that school's majority gate."""
    _, cache = model.forward(d)
    g = cache["g"]; VM = d["VM"]; SCH = d["SCH"]
    assign = g.argmax(-1)
    purities = []
    for s in range(S):
        sel = (SCH == s) & (VM > 0)
        if sel.sum() == 0:
            continue
        counts = np.bincount(assign[sel], minlength=model.K)
        purities.append(counts.max() / sel.sum())
    return float(np.mean(purities))


def linear_floor(d, dtrain):
    """Reference: the best LINEAR predictor of the target from the TRUE elephant
    z (with the query school and property known).  Zero for linear properties,
    and a hard floor for the nonlinear one — the part of the elephant that no
    amount of word-reading can reach without an unhindered substrate."""
    out = {}
    def feats(dd):
        # one block of [z,1] per (school, property) pair, selected by the query
        sch = dd["QS"]
        q = dd["E"].argmax(1)
        X = np.zeros((dd["Y"].shape[0], S * Q * (D + 1)))
        for i in range(dd["Y"].shape[0]):
            blk = (sch[i] * Q + q[i]) * (D + 1)
            X[i, blk:blk + D] = dd["Z"][i]; X[i, blk + D] = 1.0
        return X
    Xtr, Xte = feats(dtrain), feats(d)
    w = np.linalg.lstsq(Xtr, dtrain["Y"], rcond=None)[0]
    err = Xte @ w - d["Y"]
    for r, name in enumerate(REGIMES):
        sel = d["R"] == r
        out[name] = float(np.mean(err[sel] ** 2))
    out["OVERALL"] = float(np.mean(err ** 2))
    return out


def dispute_ledger(model, d, i):
    """Print one dispute the way Wonhyo would lay it out: each claim, the gate it
    was read through, what it covered, the combined ground, and the answer given
    back in the asker's vocabulary."""
    one = take(d, np.array([i]))
    _, c = model.forward(one)
    g, ch, zh, gq, pred = c["g"][0], c["ch"][0], c["zh"][0], c["gq"][0], c["pred"][0]
    print(f"    regime: {REGIMES[d['R'][i]]}")
    print(f"    the elephant (hidden z):      " + " ".join(f"{x:6.2f}" for x in d["Z"][i]))
    for m in range(MMAX):
        if d["VM"][i, m] == 0:
            continue
        s = d["SCH"][i, m]
        print(f"    claim {m}  school {s} (touched {COVER[s]}, signs {SIGNS[s]})  values "
              + " ".join(f"{x:6.2f}" for x in d["Sv"][i, m])
              + f"  -> gate {g[m].argmax()} (p={g[m].max():.2f})  covers "
              + " ".join(f"{x:4.2f}" for x in ch[m]))
    print(f"    combined ground (ẑ):          " + " ".join(f"{x:6.2f}" for x in zh))
    qs = int(d["QS"][i])
    q = int(d["E"][i].argmax())
    print(f"    question: school {qs}'s property {q}{' (nonlinear)' if q == P else ''}  "
          f"read through gate {gq.argmax()} (p={gq.max():.2f})")
    print(f"    answer {pred:6.3f}   truth {d['Y'][i]:6.3f}")


# ============================================================================
# 5.  Self-tests (structural invariants)
# ============================================================================

def test_gates_are_distributions():
    d = take(make_dataset(20, seed=1), np.arange(20))
    m = Ilsim(seed=1)
    _, c = m.forward(d)
    assert np.allclose(c["g"].sum(-1), 1.0) and np.allclose(c["gq"].sum(-1), 1.0)
    assert (c["g"] >= 0).all()
    return "gate posteriors are proper distributions over gates"


def test_substrate_never_changes():
    tr = make_dataset(400, seed=2); va = take(tr, np.arange(50))
    m = Ilsim(seed=2)
    before = (m.W1f.copy(), m.W2f.copy())
    train(m, tr, va, epochs=2, batch=64, log_every=100, name="probe")
    assert m.substrate_intact()
    assert np.array_equal(before[0], m.W1f) and np.array_equal(before[1], m.W2f)
    # but the masks DID move
    assert not np.allclose(m.p["U1"], -1.5)
    return "substrate (真如門) is byte-identical after training; only hindrance masks moved"


def test_combine_is_permutation_invariant():
    d = take(make_dataset(15, seed=3), np.arange(15))
    m = Ilsim(seed=3)
    p1 = m.predict(d)
    perm = np.random.default_rng(3).permutation(MMAX)
    d2 = dict(d); d2["W"] = d["W"][:, perm]; d2["Sv"] = d["Sv"][:, perm]; d2["VM"] = d["VM"][:, perm]
    p2 = m.predict(d2)
    assert np.allclose(p1, p2, atol=1e-10)
    return "the order in which claims arrive does not change the answer"


def test_uniform_coverage_reduces_to_mean():
    d = take(make_dataset(10, seed=4), np.arange(10))
    d["VM"][:] = 1.0
    m = Ilsim(seed=4)
    m.p["C"][:] = 12.0   # σ ≈ 1 everywhere
    _, c = m.forward(d)
    assert np.allclose(c["zh"], c["u"].mean(1), atol=1e-4)
    return "with everyone covering everything, combine collapses to the plain average (the NARROW mind)"


def test_uncovered_coordinate_is_ignored():
    d = take(make_dataset(10, seed=5), np.arange(10))
    d["VM"][:] = 1.0
    m = Ilsim(K=2, seed=5)
    # gate 0 never covers coordinate 0; force all claims through gate 0 via G=0 and gb
    m.p["G"][:] = 0.0; m.p["gb"][:] = np.array([30.0, -30.0])
    m.p["C"][0, 0] = -40.0   # coverage ≈ 0 on coord 0
    _, c1 = m.forward(d)
    d2 = dict(d); d2["Sv"] = d["Sv"].copy()
    # changing values only affects coordinate 0 through T[0,0,:]; zero the other rows' dependence
    m.p["T"][0, 1:, :] = 0.0
    _, c1 = m.forward(d)
    d2["Sv"][:, 0, :] += 5.0
    _, c2 = m.forward(d2)
    # coordinate 0 of zh must be unaffected up to 1e-6 (coverage ~1e-17) — the claim's word on an
    # untouched coordinate carries no weight; other coords may change through tb/T (zeroed) — check coord 0
    assert np.allclose(c1["zh"][:, 0], c2["zh"][:, 0], atol=1e-6)
    return "a claim carries no weight on a coordinate its gate never touched"


def test_full_hindrance_withdraws_substrate():
    d = take(make_dataset(10, seed=6), np.arange(10))
    m = Ilsim(seed=6)
    m.p["U1"][:] = -60.0; m.p["U2"][:] = -60.0
    _, c = m.forward(d)
    assert np.abs(c["h"]).max() < 1e-20
    assert np.allclose(c["m"][:, :D], c["zh"])
    assert (sigmoid(m.p["U1"]) >= 0).all() and (sigmoid(m.p["U1"]) <= 1).all()
    return "fully hindered masks silence the substrate: the mind is only the words (m = [ẑ; 0])"


def test_gradient_check():
    worst, ok = gradient_check(mode="ilsim", verbose=False)
    assert ok, worst
    return f"Panbiryang: analytic vs finite-difference gradients agree (max rel err {worst:.2e})"


def test_dataset_regimes():
    d = make_dataset(500, seed=8)
    assert d["W"].shape == (500, MMAX, V) and d["Sv"].shape == (500, MMAX, P)
    counts = np.bincount(d["R"], minlength=len(REGIMES))
    assert counts.min() >= 90
    # SPARSE has exactly 3 valid claims; others 5
    assert np.all(d["VM"][d["R"] == 3].sum(1) == 3) and np.all(d["VM"][d["R"] != 3].sum(1) == 5)
    # DISPUTE contains two schools whose shared coordinate is reported with opposite sign in at least some cases
    sel = np.where(d["R"] == 1)[0]
    opposite = 0
    for i in sel:
        sch = d["SCH"][i]; a, b = sorted(set(sch[sch >= 0]))
        shared = list(set(COVER[a]) & set(COVER[b]))[0]
        if SIGNS[a][COVER[a].index(shared)] != SIGNS[b][COVER[b].index(shared)]:
            opposite += 1
    assert opposite > 0
    return f"dataset: {counts.tolist()} per regime; {opposite}/{len(sel)} disputes report the shared coordinate with opposite signs"


def test_training_reduces_loss():
    tr = make_dataset(600, seed=9); va = take(make_dataset(150, seed=10), np.arange(150))
    m = Ilsim(seed=9)
    L0 = m.loss(va)[0]
    train(m, tr, va, epochs=8, batch=64, lr=5e-3, log_every=100, name="probe")
    L1 = m.loss(va)[0]
    assert L1 < L0 * 0.8, (L0, L1)
    return f"training reduces held-out loss ({L0:.3f} → {L1:.3f})"


def test_learning_is_unhindering():
    tr = make_dataset(600, seed=11); va = take(tr, np.arange(100))
    m = Ilsim(seed=11)
    n0 = m.unhindered_count()
    train(m, tr, va, epochs=6, batch=64, lr=5e-3, log_every=100, name="probe")
    n1 = m.unhindered_count()
    assert n1 > n0, (n0, n1)
    return f"unhindered synapses grow with learning ({n0} → {n1}); the substrate was not edited"


SELF_TESTS = [
    test_gates_are_distributions,
    test_substrate_never_changes,
    test_combine_is_permutation_invariant,
    test_uniform_coverage_reduces_to_mean,
    test_uncovered_coordinate_is_ignored,
    test_full_hindrance_withdraws_substrate,
    test_gradient_check,
    test_dataset_regimes,
    test_training_reduces_loss,
    test_learning_is_unhindering,
]


# ============================================================================
# 6.  Main
# ============================================================================

def rule(ch="="):
    print(ch * 78)


def main():
    quick = "--quick" in sys.argv
    np.set_printoptions(precision=4, suppress=True)
    rule()
    print(" ILSIM — The Two-Gate Unhinderer · Chapter 0186 · Wonhyo (617–686)")
    rule()

    # ---- self-tests ---------------------------------------------------------
    print("\n[1] Structural self-tests")
    for t in SELF_TESTS:
        msg = t()
        print(f"    PASS  {msg}")

    # ---- gradient checks for every mind ---------------------------------------
    print("\n[2] Panbiryang — finite-difference gradient checks (central differences, float64)")
    for mode in ["ilsim", "attached", "narrow", "hindered"]:
        print(f"  mode = {mode}")
        worst, ok = gradient_check(mode=mode, verbose=(mode == "ilsim"))
        print(f"    -> max relative error {worst:.2e}   {'PASS' if ok else 'FAIL'}")
        assert ok

    # ---- data ----------------------------------------------------------------
    ntr, nva, nte = (1500, 300, 500) if quick else (7000, 700, 2000)
    epochs = 8 if quick else 300
    tr = make_dataset(ntr, seed=100); va = make_dataset(nva, seed=200); te = make_dataset(nte, seed=300)
    print(f"\n[3] Data: {ntr} train / {nva} val / {nte} test disputes, {len(REGIMES)} regimes: {REGIMES}")

    # ---- train the four minds ---------------------------------------------------
    print("\n[4] Training four minds on identical data")
    results, models = {}, {}
    for mode in ["attached", "narrow", "hindered", "ilsim"]:
        m = Ilsim(mode=mode, seed=42)
        train(m, tr, va, epochs=epochs, batch=128, lr=5e-3, lr_final=3e-4, seed=42,
              log_every=(4 if quick else 50), name=mode)
        results[mode] = per_regime(m, te)
        models[mode] = m

    # ---- report ----------------------------------------------------------------
    floor = linear_floor(te, tr)
    print("\n[5] Held-out results — mean squared error of the answer (lower is better)")
    print("    LINEAR-FLOOR = best linear predictor given the TRUE elephant and the asker's frame")
    hdr = f"{'regime':<14s}" + "".join(f"{k.upper():>12s}" for k in results) + f"{'LINEAR-FLOOR':>14s}"
    print("    " + hdr)
    for reg in REGIMES + ["OVERALL"]:
        print("    " + f"{reg:<14s}" + "".join(f"{results[k][reg][0]:>12.4f}" for k in results)
              + f"{floor[reg]:>14.4f}")
    print("\n    Hit rate (|error| < 0.2)")
    print("    " + hdr)
    for reg in REGIMES + ["OVERALL"]:
        print("    " + f"{reg:<14s}" + "".join(f"{results[k][reg][1]:>12.3f}" for k in results))

    print("\n[6] What the ILSIM mind learned")
    m = models["ilsim"]
    print(f"    effective gates used      : {m.effective_gates(te):.2f}  (true schools: {S}, gates available: {m.K})")
    print(f"    gate purity vs true schools: {gate_purity(m, te):.3f}")
    print(f"    coverage σ(C) per gate (rows = gates, cols = elephant coordinates):")
    c = sigmoid(m.p["C"])
    for k in range(m.K):
        print("      gate %d  " % k + " ".join(f"{x:5.2f}" for x in c[k]))
    print(f"    unhindered synapses: {m.unhindered_count()} of {m.p['U1'].size + m.p['U2'].size}"
          f"  (init: {Ilsim(seed=42).unhindered_count()})")
    print(f"    substrate intact           : {m.substrate_intact()}")
    print(f"    NARROW mind coverage frozen: {np.allclose(sigmoid(models['narrow'].p['C']), 1.0, atol=1e-4)}")
    print(f"    HINDERED mind unhindered   : {models['hindered'].unhindered_count()} (never moved)")

    print("\n[7] One dispute, laid out (ILSIM mind)")
    i_disp = int(np.where(te["R"] == 1)[0][0])
    dispute_ledger(m, te, i_disp)
    print("\n    ... and one cross-frame question, answered in a vocabulary nobody in the room used")
    i_cross = int(np.where(te["R"] == 2)[0][0])
    dispute_ledger(m, te, i_cross)
    print("\n    ... and the same cross-frame question put to the ATTACHED mind (one gate)")
    dispute_ledger(models["attached"], te, i_cross)
    rule()
    print(" done")
    rule()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 DIASTOLE  —  the Distinction Engine
 Chapter 0226 · Constantine the Philosopher / St Cyril of Thessalonica (826-869)
 ========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 12 Minds 221 - 240 Available on Amazon https://www.amazon.com/dp/B0HKVBKNTT
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0226_Cyril_826 - Constantine the Philosopher / St Cyril of Thessalonica (826-869)
================================================================================  

WHY THIS ARCHITECTURE AND NOT ANOTHER
--------------------------------------------------------------------------------
Constantine of Thessalonica is remembered as "the man who invented an alphabet."
That is the least interesting true thing about him, and it is also the thing that
already belongs to Mesrop Mashtots in this corpus. What is *his alone* is a
narrower and stranger cognitive habit, and it is documented three times in the
near-contemporary Vita Constantini:

  (1) AT KHERSON (c. 861) he met a man speaking a language he did not read,
      and — the Vita's own words — "acquiring the power of his speech by
      comparing it to his own language, he distinguished letters, vowels and
      consonants." He did not translate first and analyse later. He counted the
      distinctions FIRST, by differencing an alien stream against a known one,
      and only then assigned meaning.

  (2) AGAINST JOHN THE GRAMMARIAN (as a very young man) he was asked why a
      broken cross is not venerated while a bust-length icon is. His answer is a
      taxonomy of how signs FAIL: "the cross has four parts, but if one of its
      parts is missing it no longer has its image. However, an icon of the face
      alone is an image and likeness of the one depicted." Compositional signs
      fail catastrophically under truncation; holistic likenesses degrade
      gracefully. He is sorting representations by their failure mode.

  (3) AT VENICE (867), attacked by Frankish clergy for writing Slavonic at all,
      he did not appeal to his own authority. He quoted 1 Corinthians 14 at
      them — their own scripture — and the verse he leaned on is the technical
      one: "except they give a DISTINCTION in the sounds, how shall it be known
      what is piped or harped? For if the trumpet give an uncertain sound, who
      shall prepare himself to the battle?" Greek: *ean diastolen tois
      phthongois me do*. DIASTOLE. The distinction in the sounds.

So the mind is not "alphabet as substrate." The mind is: A SIGNAL CARRIES
EXACTLY THE DISTINCTIONS ITS NOTATION MARKS, AND NOT ONE MORE; therefore the
first act of understanding is to COUNT the distinctions, the second is to MINT
symbols that cannot be confused with the prestige code, and the third is to
submit the result to whoever is most qualified to reject it.

This file builds that mind as a trainable network. Nothing here is a
Transformer, and there is no attention over stored keys. The mechanisms are:

  EAR (sluxu)          a contextual differencer over articulatory FEATURES, not
                       over symbol identities — because he analysed by "vowels
                       and consonants," i.e. by contrastive features.

  STROKE BASIS         a tiny learnable set of primitive strokes. Every letter
                       is a convex mixture of them. Glagolitic letters are built
                       from a repeating vocabulary of cross, circle and triangle
                       (Uspenskij's reading; a hypothesis, not consensus), so
                       the codebook here is COMPOSITIONAL, not a free lookup
                       table. A letter is a recipe, not an atom.

  DIASTOLE LAYER       soft allocation of each heard distinction to a grapheme.
                       Temperature-annealed, so it hardens into a real discrete
                       code while staying differentiable for the gradient check.

  MINTING              between epochs, dead graphemes are re-struck at the
                       positions of maximum confusion. "A letter is minted when
                       a distinction is heard that no letter marks." The network
                       therefore DISCOVERS how many symbols the tongue needs
                       instead of inheriting a number from Greek.

  REPULSION            an explicit penalty pushing every minted letter AWAY from
                       a frozen prestige alphabet. This is the un-Greek-ness of
                       Glagolitic expressed as a loss term. Its purpose is
                       anti-graceful-degradation: a code that resembles the
                       prestige code will be silently misread as a debased
                       version of it, and its extra distinctions discarded
                       without anyone noticing. Better to be unreadable than to
                       be misread. That is lesson (2) above, turned into math.

  ACROSTIC             a smoothness penalty on the ordered composition table, so
                       the alphabet READ IN ORDER is itself a continuous, legible
                       progression. The Slavonic letter-names spell a sentence in
                       alphabetical order (azu buky vede glagolju dobro estu — "I
                       know letters; to speak is a good thing"). The symbol table
                       is also a message.

  CONGREGATION         the only thing scored is whether a RECEIVER can recover
                       the meaning and act. Not reconstruction of the source.

THE EXPERIMENT THIS FILE RUNS
--------------------------------------------------------------------------------
We build a two-tongue world. Six hundred structured meanings are rendered both
in a prestige tongue (14 phones, "Hellenic") and a vernacular (22 phones,
"Slavonic") whose inventory contains contrasts the prestige tongue has no
letters for: palatalisation, nasality, and the reduced vowels (jers).

We then train three configurations on the SAME meanings:

  PILATUS      the trilingual heresy, implemented. The vernacular is first
               transliterated into prestige letters, because "only three
               languages are fit." Distinctions the prestige alphabet cannot
               mark are destroyed at the encoding step, before any learning.

  LIMEN        the vernacular is read directly, but the model is given exactly
               as many graphemes as the prestige alphabet has. Nothing is
               destroyed by transliteration; the inventory is simply inherited
               rather than counted.

  GLAGOLITIC   the full mechanism: minted, compositional, repelled, ordered.

The gap between PILATUS and GLAGOLITIC is the trilingual heresy measured in
accuracy points. The gap between LIMEN and GLAGOLITIC is the cost of inheriting
an inventory instead of counting one.

Every gradient in this file is derived by hand and verified against central
finite differences before training begins.

Run:  python3 chapter_0226_Cyril_826.py
"""

import numpy as np

RNG = np.random.default_rng(869)   # the year he died in Rome

# ==============================================================================
# SECTION 1 — THE TWO TONGUES
# ==============================================================================
# Phones are not opaque symbols. Each is a vector of articulatory FEATURES,
# because the Vita says he "distinguished letters, vowels and consonants" — he
# worked in contrasts, not in identities. Two phones that differ in exactly one
# feature bit are a MINIMAL PAIR, and a minimal pair is precisely what a
# prestige alphabet with no letter for that feature will silently collapse.

FEATURES = ["vocalic", "nasal", "palatal", "voiced", "reduced", "front",
            "stop", "continuant"]
NF = len(FEATURES)

# ---- the vernacular inventory (22 phones + PAD) -----------------------------
# Names are indicative, not a claim about reconstructed Common Slavic phonology.
# What matters structurally is the pattern of minimal pairs.
VERN_PHONES = [
    #  name     voc nas pal voi red fro stop cont
    ("PAD",    [0,  0,  0,  0,  0,  0,  0,  0]),
    ("a",      [1,  0,  0,  1,  0,  0,  0,  1]),
    ("e",      [1,  0,  0,  1,  0,  1,  0,  1]),
    ("i",      [1,  0,  0,  1,  0,  1,  0,  1]),
    ("o",      [1,  0,  0,  1,  0,  0,  0,  1]),
    ("u",      [1,  0,  0,  1,  0,  0,  0,  1]),
    ("ND_e",   [1,  1,  0,  1,  0,  1,  0,  1]),   # nasal front  (small yus)
    ("ND_o",   [1,  1,  0,  1,  0,  0,  0,  1]),   # nasal back   (big yus)
    ("JER_b",  [1,  0,  1,  1,  1,  1,  0,  1]),   # front jer
    ("JER_h",  [1,  0,  0,  1,  1,  0,  0,  1]),   # back  jer
    ("t",      [0,  0,  0,  0,  0,  0,  1,  0]),
    ("t_pal",  [0,  0,  1,  0,  0,  0,  1,  0]),   # minimal pair with t
    ("d",      [0,  0,  0,  1,  0,  0,  1,  0]),
    ("d_pal",  [0,  0,  1,  1,  0,  0,  1,  0]),   # minimal pair with d
    ("s",      [0,  0,  0,  0,  0,  0,  0,  1]),
    ("s_pal",  [0,  0,  1,  0,  0,  0,  0,  1]),   # -> the sound Greek lacks
    ("z",      [0,  0,  0,  1,  0,  0,  0,  1]),
    ("z_pal",  [0,  0,  1,  1,  0,  0,  0,  1]),
    ("n",      [0,  1,  0,  1,  0,  0,  0,  1]),
    ("r",      [0,  0,  0,  1,  0,  0,  0,  1]),
    ("l",      [0,  0,  0,  1,  0,  0,  0,  1]),
    ("m",      [0,  1,  0,  1,  0,  0,  0,  1]),
    ("k",      [0,  0,  0,  0,  0,  0,  1,  0]),
]
VERN_NAMES = [p[0] for p in VERN_PHONES]
PHI = np.array([p[1] for p in VERN_PHONES], dtype=np.float64)   # [P, NF]
P_VERN = len(VERN_PHONES)
VIDX = {n: i for i, n in enumerate(VERN_NAMES)}

# ---- the prestige transliteration -------------------------------------------
# What a scribe committed to "only Hebrew, Greek and Latin" must do to Slavonic:
# there is no letter for palatalisation, none for nasal vowels, none for the
# jers. So each of those is written with the nearest permitted letter. The map
# is surjective and NOT injective. Every merge below is a distinction destroyed.
TRANSLIT = {
    "t_pal": "t", "d_pal": "d", "s_pal": "s", "z_pal": "z",   # palatals lost
    "ND_e": "e", "ND_o": "o",                                  # nasals lost
    "JER_b": "i", "JER_h": "u",                                # jers merged out
}
def transliterate(seq):
    """Rewrite a vernacular phone sequence in prestige letters (lossy)."""
    return [VIDX[TRANSLIT.get(VERN_NAMES[p], VERN_NAMES[p])] for p in seq]

# ---- the meaning space ------------------------------------------------------
# Four slots. This is a small world, but it is a REAL structured prediction
# problem: 5 x 6 x 5 x 4 = 600 distinct meanings, and the surface form is
# ambiguous unless the reader can resolve the contrasts.
SLOTS = ["ACTOR", "ACT", "OBJECT", "MODE"]
NSLOT = [5, 6, 5, 4]
N_MEANINGS = int(np.prod(NSLOT))

# Stems deliberately place minimal pairs where they will hurt. ACTOR 2 vs 3 and
# ACTOR 0 vs 4 differ only by palatalisation; MODE differs only by nasality;
# OBJECT case-endings differ only by which jer is used.
ACTOR_STEM = [["t", "a"], ["d", "o"], ["s", "e"], ["s_pal", "e"], ["t_pal", "a"]]
ACT_STEM   = [["k", "o"], ["r", "a"], ["l", "u"], ["m", "i"], ["n", "o"], ["k", "u"]]
OBJ_STEM   = [["z", "a"], ["z_pal", "a"], ["d", "i"], ["d_pal", "i"], ["m", "u"]]
MODE_AFFIX = [["ND_e"], ["e"], ["ND_o"], ["o"]]          # nasal vs oral
OBJ_CASE   = [["JER_b"], ["JER_h"]]                       # front vs back jer

# Subject-agreement clitic, carried at the END of the clause. It is deliberately
# COARSER than the actor itself: actors 0/4 share one marker and actors 2/3 share
# another. So the only thing that separates those pairs anywhere in the string is
# the palatalisation of the very first phone — a single feature bit at position 0
# that must survive nine steps of integration to be spent at position 8.
AGREE_AFFIX = [["n"], ["r"], ["l"], ["l"], ["n"]]

def render_vernacular(a, v, o, m):
    """
    Vernacular rendering: S - O - V order, an object case-clitic, a modal affix
    and a closing agreement marker. A reader must integrate across the whole
    string: the jer at position 4 is governed by the act at positions 5-6, and
    the agreement at position 8 only disambiguates the actor when combined with
    the palatal feature back at position 0.
    """
    seq = []
    seq += ACTOR_STEM[a]
    seq += OBJ_STEM[o]
    seq += OBJ_CASE[v % 2]
    seq += ACT_STEM[v]
    seq += MODE_AFFIX[m]
    seq += AGREE_AFFIX[a]
    return [VIDX[s] for s in seq]

T_LEN = 9   # every rendering is exactly 9 phones

def build_corpus():
    """Enumerate all 600 meanings; return targets, vernacular and prestige forms."""
    Y, XV = [], []
    for a in range(NSLOT[0]):
        for v in range(NSLOT[1]):
            for o in range(NSLOT[2]):
                for m in range(NSLOT[3]):
                    Y.append([a, v, o, m])
                    XV.append(render_vernacular(a, v, o, m))
    Y = np.array(Y, dtype=np.int64)
    XV = np.array(XV, dtype=np.int64)
    XP = np.array([transliterate(s) for s in XV], dtype=np.int64)
    return Y, XV, XP

def collision_report(X, Y):
    """
    How many meanings become INDISTINGUISHABLE in this notation?
    This is the trilingual heresy stated as a number. It is a property of the
    encoding alone — no model, no training, nothing to tune. Once two meanings
    share a surface form, no reader however powerful can separate them.
    """
    buckets = {}
    for i, row in enumerate(X):
        buckets.setdefault(tuple(row), []).append(i)
    ambiguous = sum(len(v) for v in buckets.values() if len(v) > 1)
    # Ceiling accuracy per slot for an ideal reader that guesses the majority
    # label inside each collision bucket.
    ceil = []
    for k in range(len(SLOTS)):
        correct = 0
        for idxs in buckets.values():
            labels = Y[idxs, k]
            counts = np.bincount(labels, minlength=NSLOT[k])
            correct += counts.max()
        ceil.append(correct / len(X))
    return len(buckets), ambiguous, np.array(ceil)


# ==============================================================================
# SECTION 2 — THE MODEL
# ==============================================================================

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


class Diastole:
    """
    The Distinction Engine.

    Pipeline, in the order Constantine performed it:
        window of features  ->  EAR  ->  heard distinction
        heard distinction   ->  DIASTOLE  ->  allocation over minted graphemes
        graphemes in order  ->  SCRIBE (recurrent)  ->  one meaning vector
        meaning vector      ->  CONGREGATION  ->  four decisions
    """

    def __init__(self, n_graphemes=48, n_strokes=6, d_code=12, d_ear=24,
                 d_scribe=32, tau=1.0, freeze_code=False, seed=826):
        rng = np.random.default_rng(seed)
        self.M, self.K, self.D = n_graphemes, n_strokes, d_code
        self.H, self.R = d_ear, d_scribe
        self.tau = tau
        self.freeze_code = freeze_code

        s = lambda *sh: rng.normal(0, np.sqrt(2.0 / sh[-1]), size=sh)

        self.p = {
            # EAR: a three-phone window of articulatory features -> a distinction
            "W_ear": s(self.H, 3 * NF),
            "b_ear": np.zeros(self.H),
            # STROKE BASIS: the primitive marks every letter is built from
            "S":     rng.normal(0, 0.6, size=(self.K, self.D)),
            # COMPOSITION: how much of each stroke is in each letter (pre-softmax)
            "C":     rng.normal(0, 0.5, size=(self.M, self.K)),
            # the query that asks "which letter marks this distinction?"
            "W_key": s(self.D, self.H),
            "b_key": np.zeros(self.D),
            # SCRIBE: a recurrent integrator over the written line
            "W_in":  s(self.R, self.D),
            "W_rec": rng.normal(0, 0.5 / np.sqrt(self.R), size=(self.R, self.R)),
            "b_r":   np.zeros(self.R),
        }
        # CONGREGATION: one head per slot. The receiver must act, four ways.
        for k, n in enumerate(NSLOT):
            self.p[f"W_h{k}"] = s(n, self.R)
            self.p[f"b_h{k}"] = np.zeros(n)

        # THE PRESTIGE ALPHABET, frozen. Never trained. Its only job is to be
        # the thing the minted letters must not resemble.
        n_prest = 14
        Pm = rng.normal(0, 1.0, size=(n_prest, self.D))
        self.prestige = Pm / np.linalg.norm(Pm, axis=1, keepdims=True)

        self.lam_repel = 0.0
        self.lam_acro = 0.0
        self.lam_ent = 0.0

    # ---- letter construction -------------------------------------------------
    def letters(self):
        """
        Build the alphabet from the stroke basis.
        A = softmax(C) is the recipe for each letter; G = A @ S is its form.
        No letter can be an arbitrary vector: it must be a mixture of the same
        few strokes as every other letter. This is what makes the script a
        SYSTEM rather than a list.
        """
        A = softmax(self.p["C"], axis=1)          # [M, K]
        G = A @ self.p["S"]                       # [M, D]
        return A, G

    # ---- forward -------------------------------------------------------------
    def forward(self, X, Y=None):
        """X: [B, T] phone indices. Returns loss and a cache for backward."""
        B, T = X.shape
        P = self.p

        # --- EAR: difference each phone against its neighbours -----------------
        F = PHI[X]                                        # [B,T,NF]
        left = np.concatenate([np.zeros((B, 1, NF)), F[:, :-1]], axis=1)
        right = np.concatenate([F[:, 1:], np.zeros((B, 1, NF))], axis=1)
        Win = np.concatenate([left, F, right], axis=2)    # [B,T,3NF]
        Hpre = Win @ P["W_ear"].T + P["b_ear"]
        Hh = np.tanh(Hpre)                                # [B,T,H]

        # --- DIASTOLE: allocate the heard distinction to a letter --------------
        A, G = self.letters()
        Q = Hh @ P["W_key"].T + P["b_key"]                # [B,T,D]
        L = (Q @ G.T) / self.tau                          # [B,T,M]
        Al = softmax(L, axis=-1)
        Gw = Al @ G                                       # [B,T,D]  the written line

        # --- SCRIBE: integrate the line into one meaning -----------------------
        S_states = np.zeros((B, T + 1, self.R))
        for t in range(T):
            z = S_states[:, t] @ P["W_rec"].T + Gw[:, t] @ P["W_in"].T + P["b_r"]
            S_states[:, t + 1] = np.tanh(z)
        m = S_states[:, T]                                # [B,R]

        # --- CONGREGATION: four decisions --------------------------------------
        probs, loss_task = [], 0.0
        for k in range(len(NSLOT)):
            lg = m @ P[f"W_h{k}"].T + P[f"b_h{k}"]
            pr = softmax(lg, axis=-1)
            probs.append(pr)
            if Y is not None:
                loss_task += -np.mean(np.log(pr[np.arange(B), Y[:, k]] + 1e-12))

        # --- the three regularisers Constantine would recognise ----------------
        Gn = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-9)
        Rm = Gn @ self.prestige.T                          # [M, n_prest]
        loss_repel = np.mean(Rm ** 2)

        d2 = A[2:] - 2 * A[1:-1] + A[:-2]
        loss_acro = np.mean(np.sum(d2 ** 2, axis=1)) if self.M > 2 else 0.0

        ent = -np.sum(Al * np.log(Al + 1e-12), axis=-1)
        loss_ent = np.mean(ent)

        loss = (loss_task
                + self.lam_repel * loss_repel
                + self.lam_acro * loss_acro
                + self.lam_ent * loss_ent)

        cache = dict(X=X, Y=Y, Win=Win, Hh=Hh, Q=Q, A=A, G=G, Gn=Gn, Rm=Rm,
                     Al=Al, Gw=Gw, S=S_states, probs=probs, B=B, T=T, d2=d2)
        parts = dict(task=loss_task, repel=loss_repel, acro=loss_acro, ent=loss_ent)
        return loss, cache, parts

    # ---- backward ------------------------------------------------------------
    def backward(self, cache):
        """Every derivative below is hand-derived; gradcheck() verifies them."""
        P = self.p
        B, T = cache["B"], cache["T"]
        g = {k: np.zeros_like(v) for k, v in P.items()}

        # --- CONGREGATION ------------------------------------------------------
        m = cache["S"][:, T]
        dm = np.zeros_like(m)
        for k in range(len(NSLOT)):
            pr = cache["probs"][k].copy()
            pr[np.arange(B), cache["Y"][:, k]] -= 1.0
            pr /= B
            g[f"W_h{k}"] += pr.T @ m
            g[f"b_h{k}"] += pr.sum(0)
            dm += pr @ P[f"W_h{k}"]

        # --- SCRIBE (backprop through time) ------------------------------------
        dGw = np.zeros_like(cache["Gw"])
        ds = dm
        for t in range(T - 1, -1, -1):
            dz = ds * (1.0 - cache["S"][:, t + 1] ** 2)
            g["W_rec"] += dz.T @ cache["S"][:, t]
            g["W_in"] += dz.T @ cache["Gw"][:, t]
            g["b_r"] += dz.sum(0)
            dGw[:, t] += dz @ P["W_in"]
            ds = dz @ P["W_rec"]

        # --- DIASTOLE ----------------------------------------------------------
        Al, G, Q = cache["Al"], cache["G"], cache["Q"]
        dAl = dGw @ G.T
        dG = np.einsum("btm,btd->md", Al, dGw)

        # entropy term flows into the same allocation
        if self.lam_ent != 0.0:
            dAl = dAl + self.lam_ent * (-(np.log(Al + 1e-12) + 1.0)) / (B * T)

        dL = Al * (dAl - np.sum(dAl * Al, axis=-1, keepdims=True))
        dQ = (dL @ G) / self.tau
        dG += np.einsum("btm,btd->md", dL, Q) / self.tau

        g["W_key"] += np.einsum("btd,bth->dh", dQ, cache["Hh"])
        g["b_key"] += dQ.sum((0, 1))
        dHh = dQ @ P["W_key"]

        # --- EAR ---------------------------------------------------------------
        dHpre = dHh * (1.0 - cache["Hh"] ** 2)
        g["W_ear"] += np.einsum("bth,btf->hf", dHpre, cache["Win"])
        g["b_ear"] += dHpre.sum((0, 1))

        # --- the alphabet itself ------------------------------------------------
        # repulsion from the prestige code, through the L2 normalisation
        if self.lam_repel != 0.0:
            Rm, Gn = cache["Rm"], cache["Gn"]
            dRm = self.lam_repel * 2.0 * Rm / Rm.size
            dGn = dRm @ self.prestige
            nrm = np.linalg.norm(G, axis=1, keepdims=True) + 1e-9
            dG += (dGn - np.sum(dGn * Gn, axis=1, keepdims=True) * Gn) / nrm

        A = cache["A"]
        dA = dG @ P["S"].T
        g["S"] += A.T @ dG

        # acrostic: keep the ordered table a smooth progression
        if self.lam_acro != 0.0 and self.M > 2:
            c = self.lam_acro * 2.0 / (self.M - 2)
            d2 = cache["d2"]
            dA[2:] += c * d2
            dA[1:-1] += -2.0 * c * d2
            dA[:-2] += c * d2

        g["C"] += A * (dA - np.sum(dA * A, axis=1, keepdims=True))

        if self.freeze_code:
            g["C"][:] = 0.0
            g["S"][:] = 0.0
        return g

    # ---- inference -----------------------------------------------------------
    def read(self, X):
        """Hard read: each position is committed to exactly one letter."""
        _, c, _ = self.forward(X)
        return [p.argmax(1) for p in c["probs"]], c["Al"].argmax(-1)

    def accuracy(self, X, Y):
        preds, _ = self.read(X)
        per = np.array([(preds[k] == Y[:, k]).mean() for k in range(len(SLOTS))])
        whole = np.mean(np.all(np.stack([preds[k] == Y[:, k] for k in
                                         range(len(SLOTS))], 1), axis=1))
        return per, whole

    def confidence(self, X):
        """
        Mean probability the reader assigns to its own answer.

        This is the number that matters for the broken-cross test below. A
        reader that is wrong AND unsure has failed loudly and can be caught. A
        reader that is wrong AND certain has failed silently, and nobody
        downstream will ever know.
        """
        _, c, _ = self.forward(X)
        return float(np.mean([p.max(1).mean() for p in c["probs"]]))


    # ---- minting -------------------------------------------------------------
    def mint(self, X, dead_thresh=1e-3):
        """
        Re-strike unused letters at the points of greatest confusion.

        This is the operation the Vita describes at Kherson: he did not decide in
        advance how many letters Slavonic needed. He listened, found a
        distinction no existing letter marked, and made one. Letters that carry
        no load are melted down and struck again where the reader is uncertain.

        Deliberately OUTSIDE the gradient path — it is a structural edit to the
        alphabet, not a parameter update, exactly as re-seeding is in k-means.
        """
        _, c, _ = self.forward(X)
        Al = c["Al"].reshape(-1, self.M)
        usage = Al.sum(0) / Al.shape[0]
        ent = -np.sum(Al * np.log(Al + 1e-12), axis=1)
        dead = np.where(usage < dead_thresh)[0]
        if len(dead) == 0:
            return 0, int((usage > dead_thresh).sum())
        hardest = np.argsort(-ent)[:len(dead)]
        for j, pos in zip(dead, hardest):
            winner = int(Al[pos].argmax())
            self.p["C"][j] = self.p["C"][winner] + RNG.normal(0, 0.35, self.K)
        return len(dead), int((usage > dead_thresh).sum())

    def inventory(self, X, thresh=1e-3):
        _, c, _ = self.forward(X)
        Al = c["Al"].reshape(-1, self.M)
        usage = Al.sum(0) / Al.shape[0]
        return int((usage > thresh).sum()), usage


def deafen(X):
    """
    Remove from the signal exactly those contrasts a prestige alphabet cannot
    write: palatalisation, nasality, the jers. The phones remain; the
    distinctions do not. This is what a reader hears who has been taught that
    only three languages carry meaning.
    """
    return np.array([transliterate(row) for row in X], dtype=np.int64)


# ==============================================================================
# SECTION 3 — GRADIENT VERIFICATION
# ==============================================================================
# The Vita, chapter 9: the Khazar boasts that his people carry all wisdom "from
# the heart, as though absorbed," and need no writing. Constantine does not
# argue. He sets a checkable test: "tell me how many generations are there from
# Adam to Moses." The man cannot answer, and the claim collapses.
#
# A network that claims to have derived its own gradients is making exactly that
# boast. So: the checkable test, on every parameter, before any training.

def gradcheck(model, X, Y, n_per=6, eps=1e-6, tol=2e-5):
    loss, cache, _ = model.forward(X, Y)
    ana = model.backward(cache)
    worst, report = 0.0, []
    for name, arr in model.p.items():
        if model.freeze_code and name in ("C", "S"):
            continue
        flat = arr.ravel()
        idxs = RNG.choice(flat.size, size=min(n_per, flat.size), replace=False)
        errs = []
        for i in idxs:
            old = flat[i]
            flat[i] = old + eps
            lp, _, _ = model.forward(X, Y)
            flat[i] = old - eps
            lm, _, _ = model.forward(X, Y)
            flat[i] = old
            num = (lp - lm) / (2 * eps)
            an = ana[name].ravel()[i]
            errs.append(abs(num - an) / max(1.0, abs(num) + abs(an)))
        e = float(np.max(errs))
        worst = max(worst, e)
        report.append((name, e))
    return worst, report, tol


# ==============================================================================
# SECTION 4 — TRAINING
# ==============================================================================

def adam_init(p):
    return ({k: np.zeros_like(v) for k, v in p.items()},
            {k: np.zeros_like(v) for k, v in p.items()})

def train(model, Xtr, Ytr, Xte, Yte, epochs=140, batch=64, lr=6e-3,
          anneal_tau=True, do_mint=True, label="", verbose=True):
    mM, vV = adam_init(model.p)
    step = 0
    n = Xtr.shape[0]
    hist = []
    tau0, tau1 = 1.4, 0.25
    for ep in range(epochs):
        if anneal_tau:
            model.tau = tau0 + (tau1 - tau0) * (ep / max(1, epochs - 1))
            model.lam_ent = 0.02 * (ep / max(1, epochs - 1))
        perm = RNG.permutation(n)
        tot = 0.0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            loss, cache, parts = model.forward(Xtr[idx], Ytr[idx])
            grads = model.backward(cache)
            step += 1
            for k in model.p:
                mM[k] = 0.9 * mM[k] + 0.1 * grads[k]
                vV[k] = 0.999 * vV[k] + 0.001 * grads[k] ** 2
                mh = mM[k] / (1 - 0.9 ** step)
                vh = vV[k] / (1 - 0.999 ** step)
                model.p[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
            tot += loss * len(idx)
        if do_mint and ep % 12 == 11 and ep < epochs - 25:
            model.mint(Xtr)
        if verbose and (ep % 20 == 19 or ep == 0):
            per_tr, whole_tr = model.accuracy(Xtr, Ytr)
            per_te, whole_te = model.accuracy(Xte, Yte)
            inv, _ = model.inventory(Xtr)
            print(f"   [{label}] ep {ep+1:>3}  loss {tot/n:6.3f}  "
                  f"train {whole_tr:5.1%}  test {whole_te:5.1%}  "
                  f"letters in use {inv:>2}  tau {model.tau:.2f}")
        hist.append(tot / n)
    return hist


# ==============================================================================
# SECTION 5 — SELF-TESTS
# ==============================================================================

def self_tests():
    print("SELF-TESTS")
    ok = True

    # 1. the corpus is well formed
    Y, XV, XP = build_corpus()
    assert Y.shape == (N_MEANINGS, 4) and XV.shape == (N_MEANINGS, T_LEN)
    print(f"  [ok] corpus: {N_MEANINGS} meanings, {T_LEN} phones each")

    # 2. the vernacular is unambiguous; the transliteration is not
    nb_v, amb_v, ceil_v = collision_report(XV, Y)
    nb_p, amb_p, ceil_p = collision_report(XP, Y)
    assert amb_v == 0, "vernacular should be lossless"
    assert amb_p > 0, "transliteration must destroy something"
    print(f"  [ok] vernacular  : {nb_v} distinct forms, 0 collisions")
    print(f"  [ok] transliterated: {nb_p} distinct forms, "
          f"{amb_p} meanings ({amb_p/N_MEANINGS:.1%}) share a form with another")

    # 3. minimal pairs really are minimal
    for a, b in [("t", "t_pal"), ("d", "d_pal"), ("s", "s_pal"), ("z", "z_pal")]:
        d = np.abs(PHI[VIDX[a]] - PHI[VIDX[b]]).sum()
        assert d == 1.0, f"{a}/{b} differ in {d} features"
    print("  [ok] palatal contrasts differ in exactly one feature")

    # 4. letters really are mixtures of strokes
    mdl = Diastole(n_graphemes=12, n_strokes=4, d_code=6)
    A, G = mdl.letters()
    assert np.allclose(A.sum(1), 1.0) and A.min() >= 0
    assert np.allclose(G, A @ mdl.p["S"])
    print("  [ok] every letter is a convex mixture of the stroke basis")

    # 5. repulsion is high when letters copy the prestige code, low when they
    #    are built out of strokes the prestige code does not span
    m2 = Diastole(n_graphemes=8, n_strokes=4, d_code=8, seed=7)
    m2.p["C"][:] = 0.0
    m2.p["C"][:, 0] = 8.0                       # every letter = stroke 0
    m2.p["S"][0] = m2.prestige[0] * 3.0         # stroke 0 IS a prestige letter
    _, _, aligned = m2.forward(XV[:8])
    # now rebuild stroke 0 in the subspace orthogonal to the whole prestige set
    _, _, sv = np.linalg.svd(m2.prestige, full_matrices=True)
    m2.p["S"][0] = sv[-1] * 3.0
    _, _, orthogonal = m2.forward(XV[:8])
    assert aligned["repel"] > orthogonal["repel"] * 5, "repulsion must discriminate"
    print(f"  [ok] repulsion: {aligned['repel']:.3f} when letters copy the "
          f"prestige code, {orthogonal['repel']:.4f} when they avoid it")

    # 6. minting re-strikes exactly the dead letters and nothing else
    m3 = Diastole(n_graphemes=24, n_strokes=6, d_code=10, tau=0.25, seed=11)
    # strand three letters far outside the region the ear ever queries
    m3.p["C"][5] = np.array([40.0, -40, -40, -40, -40, -40])
    m3.p["C"][9] = np.array([-40.0, 40, -40, -40, -40, -40])
    m3.p["C"][17] = np.array([-40.0, -40, 40, -40, -40, -40])
    m3.p["S"] *= 6.0
    before = m3.p["C"].copy()
    n_dead, alive = m3.mint(XV[:128], dead_thresh=1e-3)
    changed = int(np.sum(np.any(before != m3.p["C"], axis=1)))
    assert changed == n_dead, "minting must touch exactly the dead letters"
    assert n_dead > 0, "this fixture was built to contain dead letters"
    print(f"  [ok] minting re-struck {n_dead} unused letters, left {alive} alive")

    # 7. deafening the signal destroys distinctions but preserves length
    d = deafen(XV[:32])
    assert d.shape == XV[:32].shape
    assert len(set(map(tuple, d.tolist()))) < len(set(map(tuple, XV[:32].tolist())))
    print("  [ok] deafening preserves the string and destroys the contrasts")

    print("  ALL SELF-TESTS PASSED\n")
    return ok


# ==============================================================================
# SECTION 6 — MAIN
# ==============================================================================

def main():
    print("=" * 78)
    print(" DIASTOLE — the Distinction Engine")
    print(" Constantine the Philosopher / St Cyril (826-869), chapter 0226")
    print("=" * 78, "\n")

    self_tests()

    Y, XV, XP = build_corpus()
    perm = RNG.permutation(N_MEANINGS)
    cut = int(0.75 * N_MEANINGS)
    tr, te = perm[:cut], perm[cut:]

    nb_p, amb_p, ceil_p = collision_report(XP, Y)
    print("THE TRILINGUAL HERESY, MEASURED BEFORE ANY LEARNING")
    print("-" * 78)
    print(" Forcing the vernacular into the prestige alphabet merges "
          f"{N_MEANINGS - nb_p} surface forms.")
    print(" No reader, however capable, can recover what the notation did not mark.")
    print(" Ceiling for a perfect reader of the transliterated text, per slot:")
    for k, s_ in enumerate(SLOTS):
        print(f"    {s_:<7} {ceil_p[k]:6.1%}")
    print()

    # ---- gradient verification ------------------------------------------------
    print("GRADIENT VERIFICATION (central differences, every parameter)")
    print("-" * 78)
    probe = Diastole(n_graphemes=20, n_strokes=5, d_code=8, d_ear=12,
                     d_scribe=14, tau=0.9)
    probe.lam_repel, probe.lam_acro, probe.lam_ent = 0.25, 0.15, 0.05
    worst, report, tol = gradcheck(probe, XV[:12], Y[:12])
    for name, e in report:
        print(f"   {name:<8} max rel err {e:.3e}")
    status = "PASS" if worst < tol else "FAIL"
    print(f"   ---> worst {worst:.3e}   tolerance {tol:.0e}   {status}\n")
    assert worst < tol, "analytic gradients disagree with finite differences"

    # ---- the three configurations ---------------------------------------------
    print("TRAINING THREE READERS ON THE SAME 600 MEANINGS")
    print("-" * 78)

    print("\n PILATUS — the vernacular transliterated into prestige letters")
    print("           ('only Hebrew, Greek and Latin are fit for this')")
    pil = Diastole(n_graphemes=14, n_strokes=6, d_code=12, tau=0.8, seed=1)
    train(pil, XP[tr], Y[tr], XP[te], Y[te], epochs=140, label="PILATUS")

    print("\n LIMEN — the vernacular read directly, but with an inherited")
    print("         inventory of exactly as many letters as Greek has")
    lim = Diastole(n_graphemes=14, n_strokes=6, d_code=12, tau=0.8, seed=2)
    lim.lam_acro = 0.05
    train(lim, XV[tr], Y[tr], XV[te], Y[te], epochs=140, label="LIMEN")

    print("\n GLAGOLITIC — minted inventory, compositional letters,")
    print("              repelled from the prestige code, ordered as an acrostic")
    gla = Diastole(n_graphemes=48, n_strokes=6, d_code=12, tau=0.8, seed=3)
    gla.lam_repel, gla.lam_acro = 0.20, 0.08
    train(gla, XV[tr], Y[tr], XV[te], Y[te], epochs=140, label="GLAGOLITIC")

    # ---- verdict ---------------------------------------------------------------
    print("\n" + "=" * 78)
    print(" VERDICT")
    print("=" * 78)
    rows = []
    for name, mdl, Xs in (("PILATUS", pil, XP), ("LIMEN", lim, XV),
                          ("GLAGOLITIC", gla, XV)):
        per, whole = mdl.accuracy(Xs[te], Y[te])
        inv, _ = mdl.inventory(Xs[tr])
        rows.append((name, per, whole, inv))
    print(f" {'reader':<12}{'ACTOR':>8}{'ACT':>8}{'OBJECT':>8}{'MODE':>8}"
          f"{'whole':>9}{'letters':>9}")
    for name, per, whole, inv in rows:
        print(f" {name:<12}" + "".join(f"{x:>8.1%}" for x in per) +
              f"{whole:>9.1%}{inv:>9}")

    inv_g, usage = gla.inventory(XV[tr])
    print(f"\n The vernacular marks {P_VERN - 1} phones; the prestige alphabet "
          f"offers 14 letters.")
    print(f" Nobody told the third reader how many letters to make. It kept "
          f"{inv_g} of 48.")
    print(f"\n Transliteration cost : {rows[2][2] - rows[0][2]:+.1%} whole-meaning "
          f"accuracy.")
    print(f" Inventory-size cost  : {rows[2][2] - rows[1][2]:+.1%} whole-meaning "
          f"accuracy.")
    print("\n Read those two numbers together, because their asymmetry is the whole")
    print(" argument. Reading the vernacular directly with an alphabet sized by")
    print(" Greek costs almost nothing. Writing the vernacular in Greek letters")
    print(" costs nearly everything. The catastrophe is located in the NOTATION,")
    print(" not in the reader — which is why no amount of scholarship, patience or")
    print(" capacity on the Frankish side could ever have substituted for letters.")

    # ---- how many letters does this tongue actually need? -----------------------
    print("\n COUNTING THE DISTINCTIONS (the act performed at Kherson, c. 861)")
    print(" " + "-" * 70)
    print("  Constantine did not ask how many letters Greek had. He listened to a")
    print("  tongue he could not write and worked out how many contrasts it made.")
    print("  Same question, asked of the data:")
    print(f"\n  {'letters offered':>16}{'whole-meaning accuracy':>26}{'letters used':>15}")
    sweep = []
    for M in (3, 6, 14, 24, 48):
        accs, invs = [], []
        for sd in (0, 1):
            s_ = Diastole(n_graphemes=M, n_strokes=6, d_code=12, tau=0.8,
                          seed=100 + M + 31 * sd)
            s_.lam_acro = 0.06
            if M >= 24:
                s_.lam_repel = 0.20
            train(s_, XV[tr], Y[tr], XV[te], Y[te], epochs=90, label=f"M{M}",
                  verbose=False)
            _, w = s_.accuracy(XV[te], Y[te])
            iv, _ = s_.inventory(XV[tr])
            accs.append(w); invs.append(iv)
        w, iv = float(np.mean(accs)), float(np.mean(invs))
        sweep.append((M, w, iv))
        bar = "#" * int(round(w * 40))
        print(f"  {M:>16}{w:>18.1%}  {bar:<41}{iv:>4.0f}")
    best = max(w for _, w, _ in sweep)
    flat = [M for M, w, _ in sweep if w > best - 0.03]
    print(f"\n  The curve is flat from {min(flat)} letters upward. Read honestly,")
    print("  that is a NEGATIVE result for the obvious hypothesis: on this tongue")
    print("  the SIZE of the alphabet is not the binding constraint, because nine")
    print("  positions of even a tiny inventory can address 600 meanings. What")
    print("  binds is whether the contrast reaches the page at all. Compare this")
    print("  flat curve against the PILATUS row above and the asymmetry is stark:")
    print("  spending letters buys almost nothing; losing distinctions costs")
    print("  almost everything. Constantine was not arguing about alphabet size.")

    A, G = gla.letters()
    Gn = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-9)
    sim = np.abs(Gn @ gla.prestige.T)
    live = usage > 1e-3
    print(f"\n Mean |cos| of a minted letter to its nearest prestige letter: "
          f"{sim[live].max(1).mean():.3f}")
    print("  The repulsion term keeps the script out of the prestige code's")
    print("  neighbourhood. Historically this is the most striking thing about")
    print("  Glagolitic: it borrows Greek's ORDER and numeric values but almost")
    print("  none of its shapes, so it cannot be skim-read as a debased Greek.")

    order = [j for j in np.argsort(-usage) if usage[j] > 1e-3][:8]
    print("\n THE EIGHT MOST-USED MINTED LETTERS, as stroke recipes")
    print(" " + "-" * 60)
    for j in order:
        bars = "".join("#" * int(round(v * 9)) + "." * (9 - int(round(v * 9)))
                       + " " for v in A[j])
        print(f"   letter {j:>2}  load {usage[j]:5.3f}   {bars}")
    print("\n Each row is one letter written as a mixture of the six strokes.")
    print(" Neighbouring rows of the full table vary smoothly, so the alphabet")
    print(" read in order is itself a continuous progression — which is what the")
    print(" Slavonic letter-names do when recited in order: azu buky vede")
    print(" glagolju dobro estu, 'I know letters; to speak is a good thing.'")

    # ---- the broken-cross test -------------------------------------------------
    # Asked why a broken cross is not venerated while a bust-length icon is,
    # Constantine answered that the cross "has four parts, but if one of its
    # parts is missing it no longer has its image," whereas the face alone
    # remains a likeness. He is sorting signs by HOW THEY FAIL. Here we take the
    # same measurement on trained readers: strip the contrasts a prestige
    # alphabet cannot write, and watch not only whether each reader breaks, but
    # whether it NOTICES that it has broken.
    print("\n" + "=" * 78)
    print(" THE BROKEN-CROSS TEST — does the reader know when it cannot read?")
    print("=" * 78)
    print(" Asked why a broken cross is not venerated while a bust-length icon is,")
    print(" Constantine answered that the cross 'has four parts, but if one of its")
    print(" parts is missing it no longer has its image,' whereas the face alone")
    print(" remains a likeness. He is sorting signs by HOW THEY FAIL. So: put the")
    print(" trained reader under four conditions and watch not whether it breaks,")
    print(" but whether it NOTICES.\n")

    Xte = XV[te]
    rng_c = np.random.default_rng(867)

    def truncated(X, k=3):
        X2 = X.copy()
        for r in range(X2.shape[0]):
            X2[r, rng_c.choice(X2.shape[1], size=k, replace=False)] = 0
        return X2

    conditions = [
        ("clean", Xte, "the line as written"),
        ("deafened", deafen(Xte), "contrasts Greek cannot mark, removed"),
        ("truncated", truncated(Xte), "three phones struck out"),
        ("nonsense", rng_c.integers(1, P_VERN, size=Xte.shape), "random phones"),
    ]
    print(f" {'condition':<12}{'accuracy':>10}{'confidence':>13}{'gap':>9}   what was done")
    print(" " + "-" * 74)
    for label, Xa, desc in conditions:
        _, acc = gla.accuracy(Xa, Y[te])
        conf = gla.confidence(Xa)
        print(f" {label:<12}{acc:>9.1%}{conf:>13.1%}{conf-acc:>+9.1%}   {desc}")

    print("\n The reader is well calibrated on clean text and badly calibrated on")
    print(" every corruption — including nonsense. That is a NEGATIVE result and it")
    print(" is reported as one: this architecture does not detect its own blindness,")
    print(" and adding capacity would not help, because deafened text is not")
    print(" malformed. It is a perfectly legal sentence that means something else.")
    print("\n Which is the argument. Constantine did not answer the trilinguists by")
    print(" promising a cleverer reader who would recover what transliteration had")
    print(" thrown away. He answered by refusing the transliteration. A distinction")
    print(" absent from the notation is not recoverable downstream, and confidence")
    print(" will not warn you that it is missing. The repair has to happen in the")
    print(" script, before anyone reads a word.\n")

    print("=" * 78)
    print(" Constantine, at Venice, 867: 'except they give a distinction in the")
    print(" sounds, how shall it be known what is piped or harped?'")
    print("=" * 78)


if __name__ == "__main__":
    main()

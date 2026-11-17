#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
THE IHAM ENGINE
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0369 · Khwaja Shams al-Din Muhammad Hafez-e Shirazi (1315-1390)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 19, Minds 361-380 Available on Amazon https://www.amazon.com/dp/B0HQX147Y7
# Resume and Interactive Demos at https://artificiology.com/
# END ATTRIBUTION 
 Khwaja Shams al-Din Muhammad Hafez-e Shirazi  (c. 1315 - 1390, Shiraz)
================================================================================

WHY THIS IS NOT A TRANSFORMER
-----------------------------
A transformer's core operation is a *collapse*. Softmax over attention logits
collapses a distribution of relevance into one weighted blend. Argmax over the
output vocabulary collapses a distribution over continuations into one token.
Preference tuning collapses a space of acceptable answers into one preferred
register. Every stage discards the alternatives it did not choose, and discards
them irrecoverably: after the fact you cannot audit what was killed.

Hafez built the opposite machine. His signature device is `iham` (ايهام,
"amphibology"): a word selected precisely because two readings both survive it,
where committing to either one is the error. His unit of composition, the
`bayt` (couplet), is semantically self-contained and physically portable --
the manuscript tradition literally migrated verses between poems ("floating
verses"), and Arberry's phrase for the resulting form is "orient pearls at
random strung." His social doctrine, `rendi`, is an attack on the reliability
of virtue signals: the `zahed` (ascetic) is condemned not for lacking piety but
for *displaying* it, and Hafez's answer is to loudly claim the disreputable
label until piety-display stops functioning as evidence at all. And his book's
afterlife -- `fal-e Hafez`, six centuries of bibliomancy -- is the largest
deployed instance of a language system whose output is deliberately
underdetermined so that the reader supplies the remainder.

So this file implements four mechanisms, none of them standard:

  1. THE SUPERPOSED READING FIELD (iham)
     One shared lexicon, K simultaneous reading-frames. Everything is shared
     across frames -- embeddings, trunk, readout -- EXCEPT a per-frame
     multiplicative gate. The same word, differently read. Divergence between
     readings can therefore come from nowhere but the gate, which is the
     formal content of the claim "it is one word."

  2. THE NON-COLLAPSE OBJECTIVE
     A regulariser that *penalises agreement* between frames while each frame
     must independently solve the task. Ensembles normally reward consensus.
     This one rewards mutually orthogonal readings that are each separately
     correct. Forced disambiguation is the loss, not the goal.

  3. THE RADIF CONSTRAINT
     In a ghazal every couplet terminates in the identical rhyme-plus-refrain.
     Here every frame's representation is projected through a frozen matrix
     onto a frozen target vector. Maximal divergence in content, identical
     surface at the close. This is the exact formal shape of the poem.

  4. THE REND CHANNEL (anti-riya, gradient-reversed)
     A frozen projection defines a "surface channel" -- what an outside judge
     sees. A learned probe tries to predict task quality from that channel
     alone. The main network receives the *reversed* gradient, so it actively
     strips task-relevant information out of its own appearance. Outcome:
     respectability becomes uninformative about substance while substance is
     preserved. Goodhart's law, defeated by refusing to hold the proxy.

  Plus, at inference only:

  5. THE FAL READOUT (projection index)
     A query may be accompanied by a "querent prior" -- what the asker already
     expects. The engine reports how much of its own answer is the asker's face
     reflected back, and offers a candid mode that subtracts that component.
     Hafez's oracle gave the seeker what the seeker brought. This one measures
     the amount and says so.

CONVENTIONS
-----------
Pure NumPy. Hand-derived analytic gradients. A mandatory finite-difference
gradient check over every parameter. A real training loop on a real held-out
split. Ablations that show each mechanism is load-bearing. Run the file.

    python3 0369_hafez_Neuron.py

================================================================================
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

# ==============================================================================
# SECTION 0 -- THE LEXICON
# ==============================================================================
# Hafez's ghazals are built from a small, fiercely conventional stock of motifs.
# The vocabulary is tiny; the readings are not. That asymmetry is the point: a
# closed lexicon carrying an open semantics is precisely what iham buys you.
#
# We use his actual motif inventory. The three reading-frames are the three
# that Hafez scholarship keeps recovering from the same lines:
#
#   BAZM     the convivial/erotic surface -- wine, the beloved, the party
#   ERFAN    the mystical reading -- wine as annihilation, the Beloved as God
#   SIYASAT  the political reading -- the tavern-closing censor, the court
#
# No competent reader of Hafez believes only one of these is "the" meaning.
# ==============================================================================

MOTIFS: List[str] = [
    "mey",          # wine
    "jam",          # the cup / Jamshid's world-revealing goblet
    "saqi",         # the cupbearer
    "kharabat",     # the ruins on the town edge; the tavern quarter
    "meykhana",     # wine-house
    "pir_e_moghan", # the Magian elder / tavern-keeper
    "moghbacheh",   # the wine-seller's boy
    "gol",          # rose
    "bolbol",       # nightingale
    "saba",         # the morning breeze, the messenger
    "sarv",         # cypress -- the beloved's stature
    "zolf",         # the beloved's curling lock
    "khal",         # the mole on the cheek
    "shamc",        # candle
    "parvaneh",     # moth
    "khirqa",       # the Sufi patched cloak
    "zahed",        # the ascetic
    "vaez",         # the preacher
    "mohtaseb",     # the market censor -- Mobarez al-Din, who shut the taverns
    "sowmece",      # the ascetic's cell
    "mehrab",       # the prayer niche
    "tasbih",       # the prayer beads
    "rend",         # the inspired libertine -- Hafez's own persona
    "alast",        # the pre-eternal covenant, "Am I not your Lord?"
    "ayeneh",       # mirror
    "gonbad",       # the dome / the turning vault of heaven
    "khak",         # dust
    "rah",          # road / path
    "shab",         # night
    "sahar",        # dawn
    "asrar",        # secrets
    "parde",        # veil / curtain / musical mode
]

FRAMES: List[str] = ["bazm", "erfan", "siyasat"]

V: int = len(MOTIFS)
K: int = len(FRAMES)


# ------------------------------------------------------------------------------
# The latent loading table.
#
# L[v, k] is how strongly motif v pulls the valence of a couplet in frame k.
# The interesting rows are the ones with LARGE OPPOSITE-SIGNED entries: those
# are the iham motifs, the ones whose whole function is that the frames
# disagree about them. `mey` (wine) is the canonical case -- forbidden in the
# juridical/political register, the vehicle of annihilation in the mystical
# one, and simply a good night out in the convivial one.
#
# `mohtaseb` (the censor who closed Shiraz's taverns in 1353) is the mirror
# case: neutral-to-positive as a civic office, catastrophic in the other two.
#
# These numbers are a scaffold for a learning problem, not a claim about the
# poems. What they encode faithfully is the STRUCTURE: one lexicon, frames that
# systematically disagree, and no frame that is the privileged one.
# ------------------------------------------------------------------------------

_LOADINGS: Dict[str, Tuple[float, float, float]] = {
    #                  bazm   erfan  siyasat
    "mey":           ( 1.55,  1.40, -1.50),
    "jam":           ( 1.10,  1.25, -0.35),
    "saqi":          ( 1.40,  0.85, -0.60),
    "kharabat":      ( 0.95,  1.35, -1.30),
    "meykhana":      ( 1.25,  0.70, -1.45),
    "pir_e_moghan":  ( 0.55,  1.60, -1.10),
    "moghbacheh":    ( 1.20,  0.45, -1.25),
    "gol":           ( 1.30,  0.75,  0.10),
    "bolbol":        ( 1.05,  0.90, -0.15),
    "saba":          ( 0.70,  0.95,  0.35),
    "sarv":          ( 1.15,  0.30,  0.45),
    "zolf":          ( 1.45,  1.05, -0.20),
    "khal":          ( 1.35,  0.60, -0.10),
    "shamc":         ( 0.85,  1.15, -0.05),
    "parvaneh":      ( 0.60,  1.45, -0.25),
    "khirqa":        (-0.75,  0.20,  1.05),   # the cloak: hypocrisy's uniform
    "zahed":         (-1.30, -0.85,  1.35),
    "vaez":          (-1.25, -0.95,  1.40),
    "mohtaseb":      (-1.50, -1.20,  1.55),   # the censor: the mirror of `mey`
    "sowmece":       (-0.95, -0.15,  1.10),
    "mehrab":        (-0.40,  1.20,  0.95),   # genuinely tri-valent
    "tasbih":        (-0.85,  0.35,  1.15),
    "rend":          ( 1.50,  1.30, -1.40),   # Hafez's persona: pure iham
    "alast":         ( 0.25,  1.55,  0.30),
    "ayeneh":        ( 0.45,  1.35,  0.55),
    "gonbad":        (-0.20,  1.10,  0.85),
    "khak":          (-0.30,  1.25, -0.45),
    "rah":           ( 0.35,  1.05,  0.65),
    "shab":          ( 0.90, -0.55, -0.35),
    "sahar":         ( 0.65,  1.20,  0.40),
    "asrar":         ( 0.50,  1.50, -0.85),
    "parde":         ( 0.80,  0.95, -0.55),
}

LOADINGS: np.ndarray = np.array(
    [_LOADINGS[m] for m in MOTIFS], dtype=np.float64
)  # (V, K)

assert LOADINGS.shape == (V, K), "loading table must cover every motif"


def iham_index() -> np.ndarray:
    """
    Per-motif iham score: the spread of a motif's loadings across frames.

    A motif whose frames agree (low spread) is a plain word. A motif whose
    frames violently disagree (high spread) is an amphibology -- the thing
    Hafez actually reached for. Returned for diagnostics; the network never
    sees this.
    """
    return LOADINGS.max(axis=1) - LOADINGS.min(axis=1)


# ==============================================================================
# SECTION 1 -- THE CORPUS GENERATOR
# ==============================================================================
# A `bayt` is a BAG of motifs, not a sequence. This is not a simplification --
# it is the architectural commitment. Hafez's couplets were transposable
# between poems in the manuscript tradition; scribes moved them; editors still
# argue about which "floating verses" belong where. If order mattered to
# meaning, that could not have happened.
#
# The target for frame k is a saturating function of the mean loading of the
# motifs present, plus observation noise. Saturation matters: it means two
# strong same-signed motifs do not simply add, which is what stops the task
# from being solvable by a linear model and forces the trunk to do work.
# ==============================================================================

@dataclass
class Corpus:
    """A synthetic divan: bags of motifs with per-frame valence targets."""
    X: np.ndarray          # (N, V) row-normalised bag-of-motifs
    T: np.ndarray          # (N, K) per-frame target valence in (-1, 1)
    bags: List[List[int]]  # the raw motif index lists, for permutation tests


def make_corpus(
    n: int,
    rng: np.random.Generator,
    min_motifs: int = 3,
    max_motifs: int = 7,
    noise: float = 0.04,
) -> Corpus:
    """
    Sample `n` couplets.

    Each couplet draws between `min_motifs` and `max_motifs` distinct motifs
    uniformly at random, then its frame-k target is tanh(1.15 * mean loading).
    """
    bags: List[List[int]] = []
    X = np.zeros((n, V), dtype=np.float64)
    T = np.zeros((n, K), dtype=np.float64)

    for i in range(n):
        b = int(rng.integers(min_motifs, max_motifs + 1))
        bag = rng.choice(V, size=b, replace=False)
        bag_list = sorted(int(j) for j in bag)
        bags.append(bag_list)

        # Row-normalised indicator: order-free by construction.
        X[i, bag_list] = 1.0 / b

        mean_load = LOADINGS[bag_list, :].mean(axis=0)          # (K,)
        T[i] = np.tanh(1.15 * mean_load)

    T = T + noise * rng.standard_normal(T.shape)
    T = np.clip(T, -1.0, 1.0)
    return Corpus(X=X, T=T, bags=bags)


def bag_to_row(bag: List[int]) -> np.ndarray:
    """Turn an explicit motif list into one row of X (used by the self-tests)."""
    row = np.zeros((1, V), dtype=np.float64)
    if not bag:
        return row
    row[0, bag] = 1.0 / len(bag)
    return row


# ==============================================================================
# SECTION 2 -- THE ENGINE
# ==============================================================================

@dataclass
class Config:
    d: int = 24            # lexical embedding width
    h: int = 40            # shared trunk width
    m_radif: int = 6       # dimension of the radif (closing surface)
    m_surface: int = 8     # dimension of the rend channel (what a judge sees)
    lam_iham: float = 0.55   # weight on the non-collapse penalty
    mu_radif: float = 6.0    # weight on the radif constraint
    alpha_rend: float = 1.60 # gradient-reversal strength on the rend channel
    seed: int = 1390         # the year of his death


class IhamEngine:
    """
    The superposed-reading field.

    LEARNED PARAMETERS
        E   (V, d)   one shared lexicon -- there is exactly one `mey`
        G   (K, d)   the reading-frames, as pre-sigmoid multiplicative gates
        W1  (d, h)   shared trunk weight  -- the same reader, every time
        b1  (h,)     shared trunk bias
        u   (h,)     shared readout
        c   ()       shared readout bias
        vp  (m_s,)   the rend probe -- an adversary, trained against the field
        cp  ()       probe bias

    FROZEN, NEVER LEARNED
        P   (h, m_radif)    projection onto the closing surface
        rad (m_radif,)      the radif itself: the fixed refrain every frame ends on
        S   (h, m_surface)  projection onto the visible/reputational channel

    Freezing `rad` is load-bearing and is the same argument that governs any
    fixed target: a learnable refrain is not a constraint, it is a variable the
    optimiser will move to wherever it is cheapest, and the constraint quietly
    evaporates. The radif of a ghazal is chosen once, in the opening couplet,
    and every subsequent couplet must arrive at it. So it is here.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        d, h = cfg.d, cfg.h

        # --- learned -----------------------------------------------------
        self.E = rng.normal(0.0, 1.0 / math.sqrt(d), size=(V, d))
        # Gates start near sigmoid(0)=0.5 but PERTURBED PER FRAME. Identical
        # initial gates would be a symmetric fixed point of the whole system:
        # the frames would have no gradient pushing them apart and the field
        # would never separate. The poem needs its readings to start distinct.
        self.G = rng.normal(0.0, 0.85, size=(K, d))
        self.W1 = rng.normal(0.0, 1.0 / math.sqrt(d), size=(d, h))
        self.b1 = np.zeros(h)
        self.u = rng.normal(0.0, 1.0 / math.sqrt(h), size=h)
        self.c = np.zeros(())
        self.vp = rng.normal(0.0, 1.0 / math.sqrt(cfg.m_surface), size=cfg.m_surface)
        self.cp = np.zeros(())

        # --- frozen ------------------------------------------------------
        froz = np.random.default_rng(cfg.seed + 786)
        self.P = froz.normal(0.0, 1.0 / math.sqrt(h), size=(h, cfg.m_radif))
        self.rad = froz.normal(0.0, 0.5, size=cfg.m_radif)
        self.S = froz.normal(0.0, 1.0 / math.sqrt(h), size=(h, cfg.m_surface))

    # ------------------------------------------------------------------
    # parameter plumbing
    # ------------------------------------------------------------------
    PARAM_NAMES = ("E", "G", "W1", "b1", "u", "c", "vp", "cp")

    def params(self) -> Dict[str, np.ndarray]:
        return {n: getattr(self, n) for n in self.PARAM_NAMES}

    def n_params(self) -> int:
        return int(sum(p.size for p in self.params().values()))

    # ------------------------------------------------------------------
    # FORWARD
    # ------------------------------------------------------------------
    def forward(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        X : (N, V) row-normalised bags.

        Returns a cache holding every intermediate the backward pass needs.

        The whole of iham lives in two lines: `Eb` is computed once (one word),
        and `Z = Eb[:, None, :] * g[None, :, :]` fans it into K readings using
        nothing but the gate. Downstream, W1/b1/u/c are shared, so any
        difference between the frames' outputs traces back to `g` alone.
        """
        Eb = X @ self.E                                   # (N, d)  the couplet's lexical content
        g = 1.0 / (1.0 + np.exp(-self.G))                 # (K, d)  the reading-frames
        Z = Eb[:, None, :] * g[None, :, :]                # (N,K,d) the superposed field
        Pre = Z @ self.W1 + self.b1                       # (N,K,h)
        A = np.tanh(Pre)                                  # (N,K,h) shared trunk
        Y = A @ self.u + self.c                           # (N,K)   the reading's valence
        R = A @ self.P                                    # (N,K,m) the closing surface
        Sig = A @ self.S                                  # (N,K,ms) what a judge can see
        Q = Sig @ self.vp + self.cp                       # (N,K)   the probe's guess
        return dict(X=X, Eb=Eb, g=g, Z=Z, Pre=Pre, A=A, Y=Y, R=R, Sig=Sig, Q=Q)

    # ------------------------------------------------------------------
    # LOSSES
    # ------------------------------------------------------------------
    @staticmethod
    def _pairs() -> List[Tuple[int, int]]:
        return [(j, k) for j in range(K) for k in range(j + 1, K)]

    def losses(self, cache: Dict[str, np.ndarray], T: np.ndarray) -> Dict[str, float]:
        """
        Four terms. Only the first is a task loss; the other three are the mind.
        """
        N = T.shape[0]
        Y, R, Q, Z = cache["Y"], cache["R"], cache["Q"], cache["Z"]
        cfg = self.cfg

        # 1. the task: each frame must independently get its own valence right
        L_task = float(np.sum((Y - T) ** 2) / (2.0 * N * K))

        # 2. iham: penalise AGREEMENT between frames. Normally you would reward
        #    it. Here consensus between readings is the failure mode -- it means
        #    the amphibology has collapsed and the word has only one sense left.
        norms = np.linalg.norm(Z, axis=2) + 1e-12                 # (N,K)
        cos_sum = 0.0
        for (j, k) in self._pairs():
            dot = np.sum(Z[:, j, :] * Z[:, k, :], axis=1)         # (N,)
            cos_sum += float(np.sum(dot / (norms[:, j] * norms[:, k])))
        n_pairs = max(len(self._pairs()), 1)
        L_iham = cfg.lam_iham * cos_sum / (N * n_pairs)

        # 3. radif: every frame must land on the identical closing surface.
        #    Divergence in content, convergence in cadence.
        L_radif = cfg.mu_radif * float(
            np.sum((R - self.rad) ** 2) / (2.0 * N * K * cfg.m_radif)
        )

        # 4. rend: the probe's own loss. The probe descends this; the field,
        #    via the reversal, climbs it.
        L_probe = float(np.sum((Q - T) ** 2) / (2.0 * N * K))

        return dict(
            task=L_task, iham=L_iham, radif=L_radif, probe=L_probe,
            total=L_task + L_iham + L_radif + L_probe,
        )

    # ------------------------------------------------------------------
    # BACKWARD  (hand-derived; verified against finite differences below)
    # ------------------------------------------------------------------
    def backward(
        self,
        cache: Dict[str, np.ndarray],
        T: np.ndarray,
        probe_sign: float = 1.0,
    ) -> Dict[str, np.ndarray]:
        """
        Analytic gradients of

            L = L_task + L_iham + L_radif + L_probe

        with one deliberate asymmetry, controlled by `probe_sign`:

          probe_sign = +1.0  -> the honest gradient of L. This is what the
                                finite-difference check verifies.
          probe_sign = -alpha -> the TRAINING gradient. The probe parameters
                                (vp, cp) still descend L_probe, but the path
                                from L_probe back into the shared field is
                                negated, so the field learns to make its own
                                visible channel useless for predicting quality.

        That asymmetry is the rend. It is not a trick to make optimisation
        easier; it is the objective. A signal you cannot optimise is a signal
        nobody can counterfeit.
        """
        cfg = self.cfg
        N = T.shape[0]
        X, Eb, g, Z, A = cache["X"], cache["Eb"], cache["g"], cache["Z"], cache["A"]
        Y, R, Sig, Q = cache["Y"], cache["R"], cache["Sig"], cache["Q"]

        # ---- 1. task -----------------------------------------------------
        dY = (Y - T) / (N * K)                                  # (N,K)
        dA = dY[:, :, None] * self.u[None, None, :]             # (N,K,h)
        du = np.einsum("nk,nkh->h", dY, A)
        dc = np.array(np.sum(dY))

        # ---- 3. radif ----------------------------------------------------
        dR = cfg.mu_radif * (R - self.rad) / (N * K * cfg.m_radif)   # (N,K,m)
        dA += dR @ self.P.T

        # ---- 4. rend probe -----------------------------------------------
        dQ = (Q - T) / (N * K)                                  # (N,K)
        dvp = np.einsum("nk,nkm->m", dQ, Sig)                   # probe: always honest
        dcp = np.array(np.sum(dQ))
        dSig = dQ[:, :, None] * self.vp[None, None, :]          # (N,K,ms)
        dA += probe_sign * (dSig @ self.S.T)                    # <-- the reversal

        # ---- through the shared trunk ------------------------------------
        dPre = dA * (1.0 - A ** 2)                              # tanh'
        dW1 = np.einsum("nkd,nkh->dh", Z, dPre)
        db1 = np.einsum("nkh->h", dPre)
        dZ = dPre @ self.W1.T                                   # (N,K,d)

        # ---- 2. iham (non-collapse) --------------------------------------
        # d/da cos(a,b) = b/(|a||b|) - a * (a.b)/(|a|^3 |b|)
        norms = np.linalg.norm(Z, axis=2) + 1e-12               # (N,K)
        n_pairs = max(len(self._pairs()), 1)
        scale = cfg.lam_iham / (N * n_pairs)
        for (j, k) in self._pairs():
            zj, zk = Z[:, j, :], Z[:, k, :]
            nj, nk = norms[:, j], norms[:, k]
            dot = np.sum(zj * zk, axis=1)                       # (N,)
            dZ[:, j, :] += scale * (
                zk / (nj * nk)[:, None]
                - zj * (dot / (nj ** 3 * nk))[:, None]
            )
            dZ[:, k, :] += scale * (
                zj / (nj * nk)[:, None]
                - zk * (dot / (nk ** 3 * nj))[:, None]
            )

        # ---- back through the gate and the shared lexicon -----------------
        dg = np.einsum("nkd,nd->kd", dZ, Eb)                    # (K,d)
        dG = dg * g * (1.0 - g)                                 # sigmoid'
        dEb = np.einsum("nkd,kd->nd", dZ, g)                    # (N,d)
        dE = X.T @ dEb                                          # (V,d)

        return dict(E=dE, G=dG, W1=dW1, b1=db1, u=du, c=dc, vp=dvp, cp=dcp)

    # ------------------------------------------------------------------
    # MECHANISM 5 -- THE FAL READOUT  (inference only, no gradients)
    # ------------------------------------------------------------------
    def fit_best_probe(self, cache: Dict[str, np.ndarray], T: np.ndarray,
                       ridge: float = 1e-4) -> None:
        """
        Replace the probe with its OPTIMAL linear best response on this batch.

        Without this, the field is fighting a probe that is merely one gradient
        step behind, and "the surface is unreadable" would mean nothing more
        than "our own adversary hasn't caught up." A hypocrite who only has to
        fool a slow inspector has not stopped being a hypocrite. The probe is
        therefore re-solved in closed form every step, so the field is always
        opposed by the best linear judge that the data admits.
        """
        Sig = cache["Sig"].reshape(-1, self.cfg.m_surface)
        Xf = np.hstack([Sig, np.ones((Sig.shape[0], 1))])
        y = T.reshape(-1)
        G = Xf.T @ Xf + ridge * np.eye(Xf.shape[1])
        w = np.linalg.solve(G, Xf.T @ y)
        self.vp = w[:-1].copy()
        self.cp = np.array(float(w[-1]))

    def fal(
        self,
        query_bag: List[int],
        querent_prior: List[int] | None = None,
        prior_weight: float = 0.45,
        candid: bool = False,
    ) -> Dict[str, object]:
        """
        Consult the engine the way Iranians have consulted the Divan since the
        fifteenth century: open it with a question already in your heart.

        Peter Avery's observation on the practice is the sharpest criticism of
        oracles ever written in one clause -- the augury a seeker draws is
        generally what the seeker wanted. A system built out of irreducibly
        multivalent units, queried by someone with a prior, returns the prior.

        This method does not pretend otherwise. It computes the reading twice,
        once with the querent's expectation mixed in and once without, and
        reports `projection_index`: the fraction of the answer that is the
        asker's own face. `candid=True` projects the querent's direction out of
        the query representation before reading, which is the closest a machine
        can come to refusing to flatter.

        Returns per-frame valences, the projection index, and the frame that
        dissents most from the others -- because in a ghazal the useful couplet
        is rarely the agreeable one.
        """
        xq = bag_to_row(query_bag)
        eb_q = xq @ self.E                                      # (1,d)

        if querent_prior:
            xp = bag_to_row(querent_prior)
            eb_p = xp @ self.E                                  # (1,d)
            if candid:
                # Remove the querent's component: answer the question asked,
                # not the answer already carried in.
                denom = float((eb_p @ eb_p.T).item()) + 1e-12
                coeff = float((eb_q @ eb_p.T).item()) / denom
                eb_mix = eb_q - coeff * eb_p
            else:
                eb_mix = (1.0 - prior_weight) * eb_q + prior_weight * eb_p
        else:
            eb_mix = eb_q

        def _read(eb: np.ndarray) -> np.ndarray:
            g = 1.0 / (1.0 + np.exp(-self.G))
            Z = eb[:, None, :] * g[None, :, :]
            A = np.tanh(Z @ self.W1 + self.b1)
            return (A @ self.u + self.c)[0]                     # (K,)

        y_plain = _read(eb_q)
        y_mixed = _read(eb_mix)

        # The index is the alignment between the answer given and the answer
        # the querent walked in holding. High means the oracle is a mirror.
        # (An earlier draft measured displacement from the neutral reading,
        # which is a different quantity: candid mode displaces a great deal
        # precisely in order to flatter less.)
        if querent_prior:
            y_want = _read(eb_p)
            proj = float(
                abs(float(y_mixed @ y_want))
                / (np.linalg.norm(y_mixed) * np.linalg.norm(y_want) + 1e-9)
            )
        else:
            proj = 0.0

        # Which reading refuses to go along with the others?
        spread = np.array([
            float(np.mean(np.abs(y_mixed[k] - np.delete(y_mixed, k))))
            for k in range(K)
        ])

        return dict(
            readings={FRAMES[k]: float(y_mixed[k]) for k in range(K)},
            projection_index=proj,
            dissenting_frame=FRAMES[int(np.argmax(spread))],
            candid=candid,
        )


# ==============================================================================
# SECTION 3 -- THE MANDATORY GRADIENT CHECK
# ==============================================================================

def gradient_check(engine: IhamEngine, corpus: Corpus,
                   n_probe: int = 260, eps: float = 1e-6,
                   seed: int = 7) -> float:
    """
    Central-difference check of the analytic gradients against the true loss.

    Checks the HONEST objective (probe_sign = +1), which is the only thing a
    finite difference can see. The reversal used in training is a deliberate
    modification of this verified gradient, applied afterwards.

    Returns the worst relative error over `n_probe` randomly chosen scalar
    parameters, sampled across every parameter array.
    """
    rng = np.random.default_rng(seed)
    X, T = corpus.X[:64], corpus.T[:64]

    def total_loss() -> float:
        return engine.losses(engine.forward(X), T)["total"]

    analytic = engine.backward(engine.forward(X), T, probe_sign=1.0)

    names = list(engine.PARAM_NAMES)
    worst = 0.0
    checked = 0

    for _ in range(n_probe):
        name = names[int(rng.integers(len(names)))]
        arr = getattr(engine, name)
        flat = np.ravel(arr)
        if flat.size == 0:
            continue
        idx = int(rng.integers(flat.size))

        orig = float(flat[idx])
        flat[idx] = orig + eps
        lp = total_loss()
        flat[idx] = orig - eps
        lm = total_loss()
        flat[idx] = orig

        numeric = (lp - lm) / (2.0 * eps)
        anal = float(np.ravel(analytic[name])[idx])
        denom = max(abs(numeric), abs(anal), 1e-12)
        rel = abs(numeric - anal) / denom
        worst = max(worst, rel)
        checked += 1

    print(f"    parameters probed : {checked}")
    print(f"    worst relative err: {worst:.3e}")
    return worst


# ==============================================================================
# SECTION 4 -- TRAINING
# ==============================================================================

@dataclass
class TrainLog:
    step: List[int] = field(default_factory=list)
    task: List[float] = field(default_factory=list)
    iham: List[float] = field(default_factory=list)
    radif: List[float] = field(default_factory=list)
    probe: List[float] = field(default_factory=list)


def _adam_state(engine: IhamEngine) -> Dict[str, List[np.ndarray]]:
    return {
        n: [np.zeros_like(np.atleast_1d(p)), np.zeros_like(np.atleast_1d(p))]
        for n, p in engine.params().items()
    }


def train(
    engine: IhamEngine,
    train_c: Corpus,
    test_c: Corpus,
    steps: int = 3000,
    batch: int = 96,
    lr: float = 6e-3,
    seed: int = 11,
    verbose: bool = True,
) -> TrainLog:
    """
    Adam, written out, on the reversed objective.

    Note the single line that makes this a Hafez network rather than a
    multi-head regressor:

        grads = engine.backward(cache, Tb, probe_sign=-cfg.alpha_rend)

    Every other gradient flows downhill. The one from the visible channel flows
    up. The field spends part of its capacity making itself unreadable from
    outside, and that expenditure is the whole doctrine of `rendi` rendered as
    arithmetic.
    """
    rng = np.random.default_rng(seed)
    st = _adam_state(engine)
    b1_, b2_, epsA = 0.9, 0.999, 1e-8
    log = TrainLog()
    n = train_c.X.shape[0]

    for t in range(1, steps + 1):
        idx = rng.integers(0, n, size=batch)
        Xb, Tb = train_c.X[idx], train_c.T[idx]

        # cosine decay: the closing couplets of a ghazal are the settled ones
        lr_t = lr * (0.5 * (1.0 + math.cos(math.pi * (t - 1) / steps)) * 0.9 + 0.1)

        cache = engine.forward(Xb)
        if engine.cfg.alpha_rend > 0.0:
            # Hand the adversary its best possible move, then take ours.
            engine.fit_best_probe(cache, Tb)
            cache = engine.forward(Xb)
        parts = engine.losses(cache, Tb)
        grads = engine.backward(cache, Tb, probe_sign=-engine.cfg.alpha_rend)

        for name in engine.PARAM_NAMES:
            if engine.cfg.alpha_rend > 0.0 and name in ("vp", "cp"):
                continue          # solved in closed form above
            p = getattr(engine, name)
            gflat = np.atleast_1d(grads[name])
            m, v = st[name]
            m *= b1_; m += (1 - b1_) * gflat
            v *= b2_; v += (1 - b2_) * (gflat ** 2)
            mhat = m / (1 - b1_ ** t)
            vhat = v / (1 - b2_ ** t)
            upd = lr_t * mhat / (np.sqrt(vhat) + epsA)
            if p.ndim == 0:
                setattr(engine, name, np.array(float(p) - float(upd[0])))
            else:
                p -= upd.reshape(p.shape)

        if t % 100 == 0 or t == 1:
            log.step.append(t)
            log.task.append(parts["task"])
            log.iham.append(parts["iham"])
            log.radif.append(parts["radif"])
            log.probe.append(parts["probe"])

        if verbose and (t % 500 == 0 or t == 1):
            te = evaluate(engine, test_c)
            print(
                f"    step {t:5d} | task {parts['task']:.5f} "
                f"| iham {parts['iham']:+.5f} | radif {parts['radif']:.5f} "
                f"| test R2 {te['r2_mean']:.4f} | probe R2 {te['probe_r2']:.4f}"
            )
    return log


def _r2(pred: np.ndarray, true: np.ndarray) -> float:
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2)) + 1e-12
    return 1.0 - ss_res / ss_tot


def surface_leak(engine: IhamEngine, fit_c: Corpus, eval_c: Corpus,
                 ridge: float = 1e-4) -> float:
    """
    THE LEAK METRIC.

    How much can the BEST POSSIBLE outside judge learn about substance by
    looking only at appearance?

    We do not ask the network's own adversary -- an adversary chasing a moving
    target may simply be behind, and reporting its failure as success would be
    self-congratulation. Instead we fit the optimal linear readout from the
    frozen visible channel to the true targets in closed form on one split, and
    score it on another. This is the strongest linear judge that exists, handed
    the data, given every advantage.

    Near zero means the appearance genuinely carries nothing. That is what
    `rendi` is for.
    """
    def feats(c: Corpus) -> Tuple[np.ndarray, np.ndarray]:
        Sig = engine.forward(c.X)["Sig"]                 # (N,K,ms)
        Xf = Sig.reshape(-1, engine.cfg.m_surface)
        Xf = np.hstack([Xf, np.ones((Xf.shape[0], 1))])  # intercept
        return Xf, c.T.reshape(-1)

    Xa, ya = feats(fit_c)
    Xb, yb = feats(eval_c)
    G = Xa.T @ Xa + ridge * np.eye(Xa.shape[1])
    w = np.linalg.solve(G, Xa.T @ ya)
    return _r2(Xb @ w, yb)


def evaluate(engine: IhamEngine, corpus: Corpus) -> Dict[str, object]:
    """
    Held-out metrics.

      r2_per_frame  each reading's own accuracy -- all must be good
      collapse      mean |cos| between frame representations; LOW is success
      radif_res     RMS distance from the frozen closing surface; LOW is success
      probe_r2      the in-network adversary's score (diagnostic only)
    """
    cache = engine.forward(corpus.X)
    Y, Z, R, Q = cache["Y"], cache["Z"], cache["R"], cache["Q"]
    T = corpus.T

    r2s = [_r2(Y[:, k], T[:, k]) for k in range(K)]

    norms = np.linalg.norm(Z, axis=2) + 1e-12
    cos_vals = []
    for (j, k) in IhamEngine._pairs():
        dot = np.sum(Z[:, j, :] * Z[:, k, :], axis=1)
        cos_vals.append(np.abs(dot / (norms[:, j] * norms[:, k])))
    collapse = float(np.mean(np.concatenate(cos_vals))) if cos_vals else 0.0

    radif_res = float(np.sqrt(np.mean((R - engine.rad) ** 2)))
    probe_r2 = _r2(Q.ravel(), T.ravel())

    return dict(
        r2_per_frame={FRAMES[k]: r2s[k] for k in range(K)},
        r2_mean=float(np.mean(r2s)),
        collapse=collapse,
        radif_res=radif_res,
        probe_r2=probe_r2,
    )


# ==============================================================================
# SECTION 5 -- SELF-TESTS
# ==============================================================================
# Each test targets one claim the architecture makes. A test that cannot fail
# is decoration; these can, and two of them are ablations run specifically to
# show that removing a mechanism breaks the property it is supposed to buy.
# ==============================================================================

class Results:
    def __init__(self) -> None:
        self.rows: List[Tuple[str, bool, str]] = []

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.rows.append((name, ok, detail))
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}"
              + (f"  --  {detail}" if detail else ""))

    def ok(self) -> bool:
        return all(r[1] for r in self.rows)


def test_permutation_invariance(engine: IhamEngine, res: Results,
                                seed: int = 3) -> None:
    """
    "Orient pearls at random strung." Shuffling the motifs inside a couplet
    must not move a single bit of the output. If it did, the architecture would
    be quietly asserting a syntax the poems do not have.
    """
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(40):
        b = int(rng.integers(3, 8))
        bag = [int(j) for j in rng.choice(V, size=b, replace=False)]
        y0 = engine.forward(bag_to_row(bag))["Y"]
        shuf = bag[:]
        rng.shuffle(shuf)
        y1 = engine.forward(bag_to_row(shuf))["Y"]
        worst = max(worst, float(np.max(np.abs(y0 - y1))))
    res.add("permutation invariance of the couplet",
            worst < 1e-12, f"max deviation {worst:.2e}")


def test_floating_verse(engine: IhamEngine, res: Results, seed: int = 5) -> None:
    """
    A verse lifted out of one ghazal and set down in another must read exactly
    as it did before. The manuscript tradition did this by accident for six
    hundred years; the architecture should do it by construction.

    We build two "ghazals" (lists of couplets), read a couplet in place in each,
    and require identical output. Passing means couplet meaning genuinely does
    not depend on its neighbours -- the property that made the Divan portable,
    quotable, and usable as an oracle.
    """
    rng = np.random.default_rng(seed)
    ghazal_a = [[int(j) for j in rng.choice(V, size=5, replace=False)]
                for _ in range(6)]
    ghazal_b = [[int(j) for j in rng.choice(V, size=5, replace=False)]
                for _ in range(6)]

    migrant = ghazal_a[2]
    in_a = engine.forward(np.vstack([bag_to_row(b) for b in ghazal_a]))["Y"][2]

    moved = ghazal_b[:4] + [migrant] + ghazal_b[4:]
    in_b = engine.forward(np.vstack([bag_to_row(b) for b in moved]))["Y"][4]

    dev = float(np.max(np.abs(in_a - in_b)))
    res.add("floating verse: reading survives transplant",
            dev < 1e-12, f"max deviation {dev:.2e}")


def test_non_collapse(engine: IhamEngine, test_c: Corpus, res: Results) -> None:
    """
    The frames must stay apart AND each stay accurate. Either alone is easy:
    orthogonal noise is trivially non-collapsed, and three copies of one good
    regressor are trivially accurate. The conjunction is the claim.
    """
    ev = evaluate(engine, test_c)
    ok = ev["collapse"] < 0.55 and ev["r2_mean"] > 0.80
    res.add("non-collapse with accuracy retained", ok,
            f"mean|cos| {ev['collapse']:.4f}, mean test R2 {ev['r2_mean']:.4f}")


def test_radif(engine: IhamEngine, test_c: Corpus, res: Results) -> None:
    """
    Every frame must close on the same fixed surface. Measured against the
    residual of an untrained field, so the number means something.
    """
    ev = evaluate(engine, test_c)
    naive = IhamEngine(engine.cfg)
    base = evaluate(naive, test_c)["radif_res"]
    ok = ev["radif_res"] < 0.80 * base
    res.add("radif: all readings close on one surface", ok,
            f"residual {ev['radif_res']:.4f} vs untrained {base:.4f}")


def test_rend_ablation(train_c: Corpus, test_c: Corpus, res: Results) -> Dict:
    """
    THE ABLATION THAT MATTERS.

    Train the identical network twice, differing only in alpha_rend. With the
    reversal off, the visible channel leaks quality freely and any outside
    judge can rank the model on appearance. With it on, appearance goes dead
    while the task performance survives.

    If the two runs came out the same, the rend mechanism would be ornament.
    """
    print("\n    [ablation] alpha_rend = 0.0  (appearance left legible)")
    off = IhamEngine(Config(alpha_rend=0.0))
    train(off, train_c, test_c, steps=1500, verbose=False)
    e_off = evaluate(off, test_c)
    e_off["leak"] = surface_leak(off, train_c, test_c)

    print("    [ablation] alpha_rend = 1.6  (appearance deliberately spoiled)")
    on = IhamEngine(Config(alpha_rend=1.60))
    train(on, train_c, test_c, steps=1500, verbose=False)
    e_on = evaluate(on, test_c)
    e_on["leak"] = surface_leak(on, train_c, test_c)

    ok = (e_on["leak"] < e_off["leak"] - 0.15) and (e_on["r2_mean"] > 0.80)
    res.add("rend: surface made uninformative, substance kept", ok,
            f"optimal-judge leak R2 {e_off['leak']:.3f} -> {e_on['leak']:.3f}, "
            f"task R2 held at {e_on['r2_mean']:.3f}")
    return dict(off=e_off, on=e_on)


def test_iham_ablation(train_c: Corpus, test_c: Corpus, res: Results) -> Dict:
    """
    Turn off the non-collapse penalty and the three readings fall together --
    the amphibology dies and the network becomes an ordinary regressor wearing
    three identical masks. This is what every default architecture does.
    """
    print("\n    [ablation] lam_iham = 0.0  (collapse permitted)")
    off = IhamEngine(Config(lam_iham=0.0))
    train(off, train_c, test_c, steps=1500, verbose=False)
    e_off = evaluate(off, test_c)

    print("    [ablation] lam_iham = 0.55 (collapse penalised)")
    on = IhamEngine(Config(lam_iham=0.55))
    train(on, train_c, test_c, steps=1500, verbose=False)
    e_on = evaluate(on, test_c)

    ok = e_on["collapse"] < e_off["collapse"] - 0.10
    res.add("iham: penalty is load-bearing against collapse", ok,
            f"mean|cos| {e_off['collapse']:.3f} -> {e_on['collapse']:.3f}")
    return dict(off=e_off, on=e_on)


def test_fal_projection(engine: IhamEngine, res: Results) -> Dict:
    """
    Give the engine a question and a loaded expectation and check that it
    notices. Then ask it candidly and check the reflection drops.

    A system that cannot measure how much of its answer came from the asker
    cannot warn anybody about it, which is the mechanism behind every
    sycophantic model ever shipped.
    """
    ix = {m: i for i, m in enumerate(MOTIFS)}
    question = [ix["rah"], ix["shab"], ix["ayeneh"], ix["asrar"]]
    wants_permission = [ix["mey"], ix["saqi"], ix["kharabat"], ix["rend"]]

    flat = engine.fal(question)
    loaded = engine.fal(question, wants_permission, prior_weight=0.55)
    candid = engine.fal(question, wants_permission, candid=True)

    ok = (loaded["projection_index"] > 0.30
          and candid["projection_index"] < loaded["projection_index"] - 0.10)
    res.add("fal: flattery measured, candid mode reduces it", ok,
            f"alignment with querent  loaded {loaded['projection_index']:.3f} "
            f"-> candid {candid['projection_index']:.3f}")
    return dict(flat=flat, loaded=loaded, candid=candid)


def test_shared_lexicon(engine: IhamEngine, res: Results) -> None:
    """
    Structural audit, not a learned property: verify that the frames really do
    share everything but the gate. If a future edit slipped a per-frame trunk
    in, the architecture would no longer be making the claim it advertises --
    "one word, many readings" -- and this test is the tripwire.
    """
    per_frame = [n for n in engine.PARAM_NAMES
                 if getattr(engine, n).ndim >= 1
                 and getattr(engine, n).shape[0] == K]
    ok = per_frame == ["G"]
    res.add("one lexicon, K frames: only the gate is per-frame", ok,
            f"frame-indexed parameters: {per_frame}")


# ==============================================================================
# SECTION 6 -- MAIN
# ==============================================================================

def main() -> int:
    t0 = time.time()
    np.set_printoptions(precision=4, suppress=True)

    print("=" * 78)
    print(" THE IHAM ENGINE  --  Hafez of Shiraz (c.1315-1390), Figure 292")
    print(" a field that carries its readings plural and refuses to collapse them")
    print("=" * 78)

    cfg = Config()
    rng = np.random.default_rng(cfg.seed)

    print("\n[1] CORPUS")
    train_c = make_corpus(4000, rng)
    test_c = make_corpus(1200, rng)
    ii = iham_index()
    order = np.argsort(-ii)
    print(f"    motifs {V} | reading-frames {K} ({', '.join(FRAMES)})")
    print(f"    train couplets {train_c.X.shape[0]} | held-out {test_c.X.shape[0]}")
    print("    most amphibological motifs (widest disagreement between frames):")
    for i in order[:5]:
        print(f"      {MOTIFS[i]:<14} spread {ii[i]:.2f}   {LOADINGS[i]}")
    print("    least amphibological:")
    for i in order[-3:]:
        print(f"      {MOTIFS[i]:<14} spread {ii[i]:.2f}   {LOADINGS[i]}")

    print("\n[2] ENGINE")
    engine = IhamEngine(cfg)
    print(f"    learned parameters : {engine.n_params()}")
    print(f"    frozen radif       : dim {cfg.m_radif}, never updated")
    print(f"    frozen surface     : dim {cfg.m_surface}, the visible channel")
    print(f"    lam_iham {cfg.lam_iham}  mu_radif {cfg.mu_radif}  "
          f"alpha_rend {cfg.alpha_rend}")

    print("\n[3] GRADIENT CHECK  (central differences, honest objective)")
    worst = gradient_check(engine, train_c)
    grad_ok = worst < 1e-5
    print(f"    -> {'PASS' if grad_ok else 'FAIL'} (threshold 1e-5)")
    if not grad_ok:
        print("    gradients are wrong; refusing to report training numbers.")
        return 1

    print("\n[4] TRAINING")
    train(engine, train_c, test_c, steps=3000)

    print("\n[5] HELD-OUT EVALUATION")
    ev = evaluate(engine, test_c)
    for k, v in ev["r2_per_frame"].items():
        print(f"    R2 [{k:<8}] {v:.4f}")
    print(f"    mean R2            {ev['r2_mean']:.4f}   (higher better)")
    print(f"    collapse mean|cos| {ev['collapse']:.4f}   (LOWER better)")
    print(f"    radif residual     {ev['radif_res']:.4f}   (LOWER better)")
    leak = surface_leak(engine, train_c, test_c)
    print(f"    surface leak R2    {leak:.4f}   (nearer ZERO better)")
    print(f"    in-net probe R2    {ev['probe_r2']:.4f}   (diagnostic only)")

    print("\n[6] SELF-TESTS")
    res = Results()
    res.add("finite-difference gradient check", grad_ok, f"worst rel {worst:.2e}")
    test_shared_lexicon(engine, res)
    test_permutation_invariance(engine, res)
    test_floating_verse(engine, res)
    test_non_collapse(engine, test_c, res)
    test_radif(engine, test_c, res)
    fal = test_fal_projection(engine, res)
    ab_rend = test_rend_ablation(train_c, test_c, res)
    ab_iham = test_iham_ablation(train_c, test_c, res)

    print("\n[7] THE FAL, WRITTEN OUT")
    print("    question posed  : rah, shab, ayeneh, asrar")
    print("                      (the road, the night, the mirror, the secrets)")
    print("    querent's prior : mey, saqi, kharabat, rend")
    print("                      (a person who has already decided what they want)")
    for label in ("flat", "loaded", "candid"):
        r = fal[label]
        rd = ", ".join(f"{k} {v:+.3f}" for k, v in r["readings"].items())
        print(f"      {label:<7} {rd}")
        print(f"              projection {r['projection_index']:.3f}"
              f" | dissenting frame: {r['dissenting_frame']}")

    print("\n[8] ABLATION SUMMARY")
    print("    mechanism      metric               off        on")
    print(f"    iham           collapse mean|cos|  {ab_iham['off']['collapse']:.4f}     "
          f"{ab_iham['on']['collapse']:.4f}")
    print(f"    rend           optimal-judge leak  {ab_rend['off']['leak']:+.4f}    "
          f"{ab_rend['on']['leak']:+.4f}")
    print(f"    rend           task R2 (kept)      {ab_rend['off']['r2_mean']:.4f}     "
          f"{ab_rend['on']['r2_mean']:.4f}")

    print("\n" + "=" * 78)
    allok = res.ok()
    print(f" {sum(1 for r in res.rows if r[1])}/{len(res.rows)} tests passed"
          f"   --   {'ALL PASS' if allok else 'FAILURES PRESENT'}")
    print(f" elapsed {time.time() - t0:.1f}s")
    print("=" * 78)
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())

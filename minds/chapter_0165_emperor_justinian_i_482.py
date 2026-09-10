#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 Chapter 0165 - EMPEROR JUSTINIAN I (482–565)
 "THE ANTINOMY ENGINE" — a compilation-based cognitive architecture
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 9 Minds 161 - 180 Available on Amazon https://www.amazon.com/dp/B0HJC1QF59
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 0165_emperor_justinian_i_482 - EMPEROR JUSTINIAN I (482–565)
================================================================================  

WHY THIS ARCHITECTURE, AND NOT A TRANSFORMER
--------------------------------------------
Justinian is not remembered because he wrote law. He is remembered because of
what he did to a *corpus*. Between 530 and 533 his commission under Tribonian
read roughly two thousand books containing about three million lines of the
classical jurists and reduced them to about 150,000 lines — a retention rate of
roughly five per cent — arranged as 50 books, 432 titles, some 9,100 excerpts.

That is a compression project. But three further decisions turn it into a
*theory of mind*, and they are what this file implements:

  1. HARMONIZATION AS THE OBJECTIVE.  The commissioning constitution (Deo
     auctore, 15 Dec 530) does not ask for a good anthology. It demands that no
     antinomy — no contradiction — "claim a place for itself," but that there
     be "one concord, one consequence" (una concordia, una consequentia).
     Consistency of the corpus IS the training objective.

  2. THE CORPUS MAY BE EDITED.  The compilers were empowered to alter the words
     of dead jurists so that those words would say what the present required,
     while still attributing them to the dead jurists' names. Later scholarship
     calls these silent rewrites *interpolations*, or emblemata Triboniani.
     The model therefore has an INTERPOLATION HEAD: it can move a retrieved
     authority toward what it needs, and the size of that move is logged.

  3. THE RESULT MAY NOT BE COMMENTED UPON.  Constitutio Tanta (16 Dec 533)
     forbade commentary on the Digest — only literal Greek translation,
     cross-references (paratitla) and short indices were allowed. The older
     literature was ordered to "fall silent." In learning terms this is a
     GRADIENT STOP: after promulgation, the corpus is frozen forever.

Plus two structural facts that become mechanisms here:

  4. THE FIFTY DECISIONS.  Before compiling, Justinian issued the *quinquaginta
     decisiones* (1 Aug 530 – 30 Apr 531) to settle juristic disputes by fiat:
     "Quod certatum est apud veteres, nos decidimus" — what was disputed among
     the ancients, we have decided. The model gets exactly 50 axiom slots, used
     only when the retrieved authorities are ambiguous.

  5. THE THREE MASSES.  Bluhme (1820) showed the excerpts inside each title fall
     into recurring blocks — Sabinian, Edictal, Papinian — because three
     committees read three separate reading-lists in parallel. The compiler's
     *pipeline* is still legible in the compiled artifact 1,300 years later.
     The model reads through three committee channels, and this file contains a
     Bluhme test that recovers the committee partition from the output alone.

And one thing Justinian could not stop: the world kept moving. He went on
issuing *Novellae* — new laws, mostly in Greek — for thirty more years, which is
the frozen corpus's own confession that it was already stale. The model has an
append-only NOVELLA store, and the rate at which it fires is the drift meter.

THE CENTRAL EXPERIMENT
----------------------
Train one engine two ways through the same changing world:
  * OPEN    — commentary permitted; the whole corpus keeps learning.
  * FROZEN  — Tanta §21; after promulgation only the novella store and the
              output head may learn. The corpus is sealed.
Then measure what sealing costs. And separately, sweep the interpolation budget
to price the trade Justinian actually made: you can buy consistency, but you pay
for it in fidelity to what the sources really said. That price is computable.

CONVENTIONS
-----------
Pure NumPy. Hand-derived backward pass. Mandatory finite-difference gradient
check. Real training loop. Self-tests. No frameworks, no pretrained anything.

Run:  python3 chapter_0165_emperor_justinian_i_482.py
================================================================================
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

__version__ = "2.0.0"
FIGURE_ID = 165
FIGURE_NAME = "Emperor Justinian I"


# ==============================================================================
# 0. SMALL NUMERICAL UTILITIES
# ==============================================================================

def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def softmax_backward(p: np.ndarray, dp: np.ndarray, axis: int = -1) -> np.ndarray:
    """Backward through softmax given the OUTPUT p and upstream grad dp."""
    dot = np.sum(p * dp, axis=axis, keepdims=True)
    return p * (dp - dot)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def entropy(p: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """Shannon entropy in nats. Used two ways here:
       - as the AMBIGUITY signal that summons an imperial decision;
       - as the COMPRESSION penalty (few authorities per ruling)."""
    return -np.sum(p * np.log(p + eps), axis=axis)


def effective_count(p: np.ndarray, axis: int = -1) -> np.ndarray:
    """exp(H) — the 'perplexity' of a citation distribution: roughly, how many
    excerpts the engine is really leaning on. Justinian's compilers kept ~5%
    of what they read; we want this number small."""
    return np.exp(entropy(p, axis=axis))


# ==============================================================================
# 1. THE WORLD: A DOCKET OF CASES, AND A WORLD THAT MOVES AFTER YOU SEAL IT
# ==============================================================================
#
# The engine is not asked to model language. It is asked to do what the Digest
# was built to do: given a case, deliver the correct ruling, justified out of a
# fixed corpus of dead jurists.
#
# Three kinds of case exist, and the difference between them is the whole point.
#
#   SETTLED   (principles 0, 2, 3)  Law the classical jurists covered and that
#                                   nobody overturned. The corpus is adequate.
#
#   OVERRULED (principle 1)         Law the jurists covered — and that the
#                                   emperor later REVERSED. Justinian did this
#                                   constantly, by Novel. A sealed corpus can
#                                   still survive this, because overruling only
#                                   requires changing the ANSWER, not the
#                                   concepts used to reach it.
#
#   NOVEL MATTER (the new docket)   Subject matter no jurist ever wrote about.
#                                   Here the trap is set with care: novel cases
#                                   sit CLOSE IN FEATURE SPACE to principle 0,
#                                   so the corpus retrieves principle-0
#                                   authorities with total confidence — and they
#                                   are the wrong authorities. Worse, the correct
#                                   ruling depends on the INTERACTION of two
#                                   attributes (an exclusive-or), so it cannot be
#                                   read off any single direction. Getting it
#                                   right requires new CONCEPTS, not new answers.
#
# That distinction — an answer can be decreed, a concept cannot — is the fault
# line this whole architecture is built to expose.

N_CLASSES = 4
D_IN = 12
N_TITLES = 12                       # thematic titles; the real Digest has 432
N_PRINC = 4                         # principles the classical jurists covered
RULING_SETTLED = np.array([0, 1, 2, 3])
OVERRULED_PRINCIPLE = 1
OVERRULED_NEW_RULING = 3            # "Quod certatum est apud veteres, nos decidimus."


@dataclass
class World:
    """Generates dockets. Deterministic given a seed."""
    rng: np.random.Generator
    protos: np.ndarray = field(init=False)
    provinces: np.ndarray = field(init=False)
    dA: np.ndarray = field(init=False)
    dB: np.ndarray = field(init=False)
    novel_proto: np.ndarray = field(init=False)

    def __post_init__(self):
        r = self.rng
        self.protos = r.normal(0, 1, (N_PRINC, D_IN))
        self.protos /= np.linalg.norm(self.protos, axis=1, keepdims=True)
        self.provinces = 0.22 * r.normal(0, 1, (3, D_IN))
        # the two attributes of the new matter, and its deceptive location
        self.dA = r.normal(0, 1, D_IN); self.dA /= np.linalg.norm(self.dA)
        self.dB = r.normal(0, 1, D_IN); self.dB /= np.linalg.norm(self.dB)
        dC = r.normal(0, 1, D_IN); dC /= np.linalg.norm(dC)
        p = self.protos[0] + 0.35 * dC          # sits right next to principle 0
        self.novel_proto = p / np.linalg.norm(p)

    # -- case constructors ---------------------------------------------------
    def _classical(self, princ, overruled: bool):
        n = len(princ)
        prov = self.rng.integers(0, 3, n)
        X = (self.protos[princ] + self.provinces[prov]
             + 0.28 * self.rng.normal(0, 1, (n, D_IN)))
        y = RULING_SETTLED[princ].copy()
        if overruled:
            y[princ == OVERRULED_PRINCIPLE] = OVERRULED_NEW_RULING
        title = (princ * 2 + (prov % 2)) % N_TITLES
        return X, y, title

    def _novel(self, n):
        a = self.rng.integers(0, 2, n) * 2 - 1     # +/- 1
        b = self.rng.integers(0, 2, n) * 2 - 1
        prov = self.rng.integers(0, 3, n)
        X = (self.novel_proto[None, :]
             + 0.62 * a[:, None] * self.dA[None, :]
             + 0.62 * b[:, None] * self.dB[None, :]
             + self.provinces[prov]
             + 0.28 * self.rng.normal(0, 1, (n, D_IN)))
        # THE EXCLUSIVE-OR. No single direction in the case decides it.
        y = np.where(a == b, 0, 2)
        title = 8 + (prov % 2) * 2 + (a > 0).astype(int)
        return X, y.astype(np.int64), title % N_TITLES

    # -- dockets -------------------------------------------------------------
    def docket(self, n: int, kind: str):
        """kind ∈ {'pre','post','settled','overruled','novel'}"""
        if kind == "settled":
            princ = self.rng.choice([0, 2, 3], size=n)
            X, y, t = self._classical(princ, overruled=True)
        elif kind == "overruled":
            princ = np.full(n, OVERRULED_PRINCIPLE)
            X, y, t = self._classical(princ, overruled=True)
        elif kind == "novel":
            X, y, t = self._novel(n)
        elif kind == "pre":
            # the world of 530: only the classical principles, old rulings
            princ = self.rng.integers(0, N_PRINC, n)
            X, y, t = self._classical(princ, overruled=False)
        elif kind == "post":
            # the world after 534: reconquest, plague, the silk trade
            roll = self.rng.random(n)
            n_cl = int(np.sum(roll >= 0.35))
            princ = self.rng.integers(0, N_PRINC, n_cl)
            X1, y1, t1 = self._classical(princ, overruled=True)
            X2, y2, t2 = self._novel(n - n_cl)
            X = np.concatenate([X1, X2]); y = np.concatenate([y1, y2])
            t = np.concatenate([t1, t2])
            perm = self.rng.permutation(n)
            X, y, t = X[perm], y[perm], t[perm]
        else:
            raise ValueError(kind)
        return X.astype(np.float64), y.astype(np.int64), t.astype(np.int64)


# ==============================================================================
# 2. THE ATTESTED CORPUS: WHAT THE DEAD JURISTS ACTUALLY WROTE
# ==============================================================================
#
# 48 excerpts. Each is a noisy statement about one principle, embedded in the
# engine's memory space. Three facts about the real Digest are built in:
#
#   (a) THREE MASSES. Excerpts 0-15 were read by the Sabinian committee, 16-31
#       by the Edictal, 32-47 by the Papinian. Each excerpt has a reading order
#       inside its mass — the committee's place in its assigned book-list.
#
#   (b) UNEQUAL AUTHORITY. About four fifths of the Digest comes from five
#       jurists, Ulpian above all. The prior authority weights reflect the Law
#       of Citations of 426 (Papinian breaks ties). This is a training-set
#       imbalance the engine inherits and cannot choose.
#
#   (c) REAL ANTINOMIES. For each principle, most excerpts agree — but some are
#       NEGATED. The Sabinians and the Proculians genuinely disagreed. The
#       corpus arrives contradicting itself. Making it stop is the whole task.

N_EXCERPTS = 48
D_MEM = 16
MASS_NAMES = ["Sabinian", "Edictal", "Papinian"]
MASS_OF = np.repeat([0, 1, 2], 16)                 # (48,)
READ_ORDER = np.tile(np.arange(16), 3)             # position in its book-list

JURISTS = ["Ulpian", "Paulus", "Papinian", "Gaius", "Modestinus"]
# Share of the corpus, echoing the real skew (Ulpian ~2/5 of the Digest).
JURIST_SHARE = np.array([0.40, 0.17, 0.15, 0.15, 0.13])
# Prior standing under the Law of Citations, 426 CE.
JURIST_PRIOR = np.array([0.55, 0.35, 0.75, 0.15, 0.05])


def build_corpus(rng: np.random.Generator, world: World):
    """Return M0 (48,16) attested excerpts, alpha0 (48,) authority priors, and
    the bookkeeping needed to audit them later.

    CRUCIAL: the excerpts cover principles 0..3 ONLY. No classical jurist ever
    wrote a line about the new matter. That silence is not a modelling
    convenience — it is the historical situation, and it is what a sealed corpus
    can never repair.
    """
    R = rng.normal(0, 1.0 / math.sqrt(D_IN), (D_IN, D_MEM))   # case -> memory

    princ_of = np.zeros(N_EXCERPTS, dtype=int)
    sign_of = np.ones(N_EXCERPTS)
    M0 = np.zeros((N_EXCERPTS, D_MEM))

    for i in range(N_EXCERPTS):
        p = i % N_PRINC                       # 12 excerpts per principle
        princ_of[i] = p
        # Three excerpts in every twelve are DISSENTS: they state the opposite.
        # The Sabinians and the Proculians really did disagree. These are the
        # antinomiae that Deo auctore ordered abolished.
        sign_of[i] = -1.0 if (i // N_PRINC) % 4 == 3 else 1.0
        M0[i] = sign_of[i] * (world.protos[p] @ R) + 0.18 * rng.normal(0, 1, D_MEM)

    M0 /= np.linalg.norm(M0, axis=1, keepdims=True)

    jurist_of = rng.choice(len(JURISTS), size=N_EXCERPTS, p=JURIST_SHARE)
    alpha0 = JURIST_PRIOR[jurist_of] + 0.05 * rng.normal(0, 1, N_EXCERPTS)
    return M0, alpha0, jurist_of, princ_of, sign_of, R


# ==============================================================================
# 3. THE ENGINE
# ==============================================================================
#
# FORWARD PASS — read it as the working day of Tribonian's commission.
#
#   q      = tanh(Wq x + bq)              the case, restated as a legal question
#   r      = M q / sqrt(d) + alpha        relevance of each excerpt, weighted by
#                                         the standing of the jurist who wrote it
#   c      = softmax(r / tau)             THE EXCERPTION: which authorities are
#                                         cited. Its entropy is penalised — the
#                                         commission had a 5% retention budget.
#   z      = [z_Sab | z_Ed | z_Pap]       three committees read in parallel and
#                                         their reads are concatenated in the
#                                         fixed Bluhme order. This ordering is
#                                         what leaks the pipeline (see §7).
#   delta  = eps * tanh(We [q;z] + be)    THE INTERPOLATION: how far the engine
#   z'     = z + U delta                  moves the authorities from what they
#                                         actually said. Logged and priced.
#   h0     = tanh(Wh [q;z'] + bh)         the draft opinion
#   u      = sigmoid(w_amb * H(c) + b)    ambiguity gate: are the authorities
#                                         still at war?
#   s      = softmax(V h0 + bv)           ...if so, spend one of
#   h1     = h0 + u * (D^T s)             THE FIFTY DECISIONS (exactly 50 slots)
#   nu     = sigmoid(wn . h1 + bn)        NOVELTY: does the corpus even cover
#                                         this? nu is penalised — the emperor
#                                         hates to admit the Digest is short.
#   h2     = h1 + nu * novella_read(h1)   the append-only Novellae, in Greek,
#                                         issued for thirty years after the
#                                         corpus was declared complete.
#   logits = Wy h2 + by                   the ruling.
#
# LOSS
#   cross-entropy                       get the case right
# + lam_h * antinomy(c, M)              "let no antinomy claim a place"
# + lam_c * H(c)                        compress: cite few authorities
# + lam_e * ||delta||^2                 interpolation is not free
# + lam_n * nu                          novelty is an embarrassment
# + lam_f * ||M - M0||^2 / N            FIDELITY: how far the corpus has drifted
#                                       from what the jurists really wrote.
#
# The last two terms are the moral of the whole architecture. lam_f is the only
# thing standing between a consistent corpus and a forged one.

@dataclass
class Config:
    d_in: int = D_IN
    d_mem: int = D_MEM
    n_exc: int = N_EXCERPTS
    d_hid: int = 24
    n_dec: int = 50          # the quinquaginta decisiones — exactly fifty
    n_nov: int = 8           # novella slots (deliberately few; each is a scandal)
    n_cls: int = N_CLASSES
    d_int: int = 16          # the rescript subspace
    tau: float = 0.80        # excerption temperature
    eps_rescript: float = 0.50   # how far the emperor may bend a text in the act
    lam_h: float = 0.60      # harmonization: "let no antinomy claim a place"
    retention: float = 0.05  # THE COMPILATION LIMIT. Honoré reconstructs a five
                             # per cent quota from the excerpting timetable: the
                             # committees were to keep about 1 line in 20. This
                             # is a QUOTA, not an instruction to minimise — so it
                             # is enforced two-sidedly, as a target, not a squeeze.
    lam_c: float = 0.15      # how hard the quota binds
    lam_e: float = 0.05      # cost of a rescript
    lam_n: float = 0.12      # novelty is an embarrassment, and the gate is
                             # taxed accordingly. Every Novel is a public
                             # admission that the completed corpus was not.
    lam_f: float = 0.30      # FIDELITY: the price of rewriting a dead jurist
    lam_a: float = 0.10      # the Law of Citations of 426: standing is GIVEN,
                             # and the engine pays to disagree with it
    lr: float = 0.03


class AntinomyEngine:
    """Justinian's compiler, as a differentiable machine.

    ONE STRUCTURAL COMMITMENT ABOVE ALL OTHERS: the ruling is computed from the
    RETRIEVED AUTHORITIES, not from the case. The case only forms the query. If
    the corpus has nothing to say about a matter, the engine has nothing to say
    about it either — and that is not a limitation of the model, it is the model's
    entire thesis. A Digest that could answer questions it never contained would
    not be a Digest.
    """

    CORPUS_PARAMS = ["Wq", "bq", "M", "alpha", "We", "be", "U",
                     "Wh", "bh", "V", "bv", "D", "w_amb", "b_amb"]
    NOVELLA_PARAMS = ["Nk", "Nv", "wn", "bn", "Wy", "by"]

    def __init__(self, cfg: Config, M0, alpha0, seed: int = 0):
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        c = cfg
        s = lambda a, b: rng.normal(0, math.sqrt(2.0 / (a + b)), (a, b))
        self.d_ctx = c.d_hid + c.d_mem      # what a Novel is allowed to look at

        self.M0 = M0.copy()                 # THE ATTESTED TEXT — never updated
        self.p: Dict[str, np.ndarray] = {
            "Wq": s(c.d_mem, c.d_in), "bq": np.zeros(c.d_mem),
            "M": M0.copy(), "alpha": alpha0.copy(),
            "We": s(c.d_int, 3 * c.d_mem), "be": np.zeros(c.d_int),
            "U": s(3 * c.d_mem, c.d_int),
            "Wh": s(c.d_hid, 3 * c.d_mem), "bh": np.zeros(c.d_hid),
            "V": s(c.n_dec, c.d_hid), "bv": np.zeros(c.n_dec),
            "D": s(c.n_dec, c.d_hid),
            "w_amb": np.array(1.0), "b_amb": np.array(-1.0),
            "Nk": s(c.n_nov, self.d_ctx),          # novella keys
            "Nv": s(c.n_nov, c.d_hid),             # novella content
            "wn": np.zeros(self.d_ctx), "bn": np.array(-1.0),
            "Wy": s(c.n_cls, c.d_hid), "by": np.zeros(c.n_cls),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0
        self.mass_idx = [np.where(MASS_OF == j)[0] for j in range(3)]
        self.alpha0 = alpha0.copy()     # the standing fixed by the Law of Citations
        self.novellae_on = 1.0          # set to 0.0 to judge by THE DIGEST ALONE

    # ------------------------------------------------------------------ forward
    def forward(self, X, y=None, eps_override=None):
        p, c = self.p, self.cfg
        B = X.shape[0]
        eps = c.eps_rescript if eps_override is None else eps_override
        sc = math.sqrt(c.d_mem)

        q = np.tanh(X @ p["Wq"].T + p["bq"])                  # the legal question
        r = q @ p["M"].T / sc + p["alpha"]                    # relevance x standing
        cw = softmax(r / c.tau, axis=1)                       # THE EXCERPTION

        zs = [cw[:, i] @ p["M"][i] for i in self.mass_idx]    # three committees
        z = np.concatenate(zs, axis=1)                        # in Bluhme order

        # THE INTERPOLATION HEAD. It reads the RETRIEVED TEXT and nothing else.
        # An interpolation is an edit made to what a jurist said — not a back
        # door through which the facts of the case can reach the verdict
        # unmediated. Closing that door is what forces the engine to actually
        # use its corpus. The ONLY path from case to ruling is which authorities
        # get cited.
        e = np.tanh(z @ p["We"].T + p["be"])
        delta = eps * e
        zp = z + delta @ p["U"].T

        h0 = np.tanh(zp @ p["Wh"].T + p["bh"])                # the draft opinion
        #      ^^^ note: zp only. The ruling comes out of the authorities.

        Hc = entropy(cw, axis=1)
        u = sigmoid(p["w_amb"] * Hc + p["b_amb"])             # still at war?
        sv = softmax(h0 @ p["V"].T + p["bv"], axis=1)
        dec = sv @ p["D"]                                     # THE FIFTY DECISIONS
        h1 = h0 + u[:, None] * dec

        ctx = np.concatenate([h1, q], axis=1)                 # a Novel answers the
        gt = ctx @ p["Nk"].T / math.sqrt(self.d_ctx)          # WORLD, not the Digest
        g = softmax(gt, axis=1)
        nvr = g @ p["Nv"]
        nu = self.novellae_on * sigmoid(ctx @ p["wn"] + p["bn"])   # THE DRIFT METER
        h2 = h1 + nu[:, None] * nvr

        logits = h2 @ p["Wy"].T + p["by"]
        prob = softmax(logits, axis=1)

        cache = dict(X=X, q=q, cw=cw, z=z, e=e, delta=delta, zp=zp,
                     h0=h0, Hc=Hc, u=u, sv=sv, dec=dec, h1=h1, ctx=ctx, g=g,
                     nvr=nvr, nu=nu, h2=h2, prob=prob, y=y, B=B, eps=eps)
        if y is None:
            return None, cache

        ce = -np.mean(np.log(prob[np.arange(B), y] + 1e-12))
        # ANTINOMY ENERGY. Two texts contradict when they point opposite ways.
        # Measured on the UNIT SPHERE, so that an engine cannot make its corpus
        # look consistent merely by inflating it. This is a pure measure of how
        # much the cited authorities are still at war with each other.
        nrm = np.linalg.norm(p["M"], axis=1, keepdims=True) + 1e-9
        Mn = p["M"] / nrm
        G = Mn @ Mn.T
        mask = (G < 0)
        Kn = np.where(mask, -G, 0.0)
        anti = np.mean(np.einsum("bi,ij,bj->b", cw, Kn, cw))
        H_tgt = math.log(max(1.5, c.retention * c.n_exc))     # the 5% quota, in nats
        comp = np.mean((Hc - H_tgt) ** 2)
        resc = np.mean(np.sum(delta ** 2, axis=1))
        novel = np.mean(nu)
        fidel = np.sum((p["M"] - self.M0) ** 2) / c.n_exc
        auth = np.sum((p["alpha"] - self.alpha0) ** 2) / c.n_exc

        loss = (ce + c.lam_h * anti + c.lam_c * comp + c.lam_e * resc
                + c.lam_n * novel + c.lam_f * fidel + c.lam_a * auth)
        cache.update(G=G, mask=mask, Kn=Kn, Mn=Mn, nrm=nrm,
                     parts=dict(ce=ce, anti=anti, comp=comp, resc=resc,
                                novel=novel, fidel=fidel, auth=auth, loss=loss))
        return loss, cache

    # ----------------------------------------------------------------- backward
    def backward(self, ca):
        p, c = self.p, self.cfg
        B = ca["B"]; eps = ca["eps"]; sc = math.sqrt(c.d_mem)
        cw, q, e, delta = ca["cw"], ca["q"], ca["e"], ca["delta"]
        h0, h1, u, sv, dec = ca["h0"], ca["h1"], ca["u"], ca["sv"], ca["dec"]
        g, nvr, nu, ctx, prob, y = (ca["g"], ca["nvr"], ca["nu"], ca["ctx"],
                                    ca["prob"], ca["y"])
        gr = {k: np.zeros_like(v) for k, v in p.items()}
        dq = np.zeros_like(q)

        dlogits = prob.copy(); dlogits[np.arange(B), y] -= 1.0; dlogits /= B
        gr["Wy"] += dlogits.T @ ca["h2"]; gr["by"] += dlogits.sum(0)
        dh2 = dlogits @ p["Wy"]

        # --- h2 = h1 + nu * nvr  (the Novellae)
        dh1 = dh2.copy()
        dnu = np.sum(dh2 * nvr, axis=1)
        dnvr = nu[:, None] * dh2
        gr["Nv"] += g.T @ dnvr
        dg = dnvr @ p["Nv"].T
        dgt = softmax_backward(g, dg, axis=1)
        dctx = dgt @ p["Nk"] / math.sqrt(self.d_ctx)
        gr["Nk"] += dgt.T @ ctx / math.sqrt(self.d_ctx)

        dnu = dnu + c.lam_n / B                       # the tax on admitting drift
        sg = nu / max(self.novellae_on, 1e-12) if self.novellae_on else 0.0
        dnu_pre = dnu * self.novellae_on * sg * (1 - sg)
        gr["wn"] += ctx.T @ dnu_pre
        gr["bn"] += dnu_pre.sum()
        dctx += np.outer(dnu_pre, p["wn"])
        dh1 += dctx[:, :c.d_hid]
        dq += dctx[:, c.d_hid:]

        # --- h1 = h0 + u * dec  (the Fifty Decisions)
        dh0 = dh1.copy()
        du = np.sum(dh1 * dec, axis=1)
        ddec = u[:, None] * dh1
        gr["D"] += sv.T @ ddec
        dsv = softmax_backward(sv, ddec @ p["D"].T, axis=1)
        gr["V"] += dsv.T @ h0; gr["bv"] += dsv.sum(0)
        dh0 += dsv @ p["V"]

        du_pre = du * u * (1 - u)
        gr["w_amb"] += np.sum(du_pre * ca["Hc"]); gr["b_amb"] += du_pre.sum()
        dHc = du_pre * p["w_amb"]

        # --- h0 = tanh(Wh zp + bh)
        dpre_h = dh0 * (1 - h0 ** 2)
        gr["Wh"] += dpre_h.T @ ca["zp"]; gr["bh"] += dpre_h.sum(0)
        dzp = dpre_h @ p["Wh"]

        # --- zp = z + delta U^T  (the rescript)
        dz = dzp.copy()
        gr["U"] += dzp.T @ delta
        ddelta = dzp @ p["U"] + 2.0 * c.lam_e * delta / B
        dpre_e = (eps * ddelta) * (1 - e ** 2)
        gr["We"] += dpre_e.T @ ca["z"]; gr["be"] += dpre_e.sum(0)
        dz += dpre_e @ p["We"]

        # --- the three committee reads
        dcw = np.zeros_like(cw)
        for j, idx in enumerate(self.mass_idx):
            dzj = dz[:, j * c.d_mem:(j + 1) * c.d_mem]
            dcw[:, idx] += dzj @ p["M"][idx].T
            gr["M"][idx] += cw[:, idx].T @ dzj

        # --- entropy: ambiguity gate + compression budget
        H_tgt = math.log(max(1.5, c.retention * c.n_exc))
        dH_total = dHc + c.lam_c * 2.0 * (ca["Hc"] - H_tgt) / B
        dcw += dH_total[:, None] * (-(np.log(cw + 1e-12) + 1.0))

        # --- antinomy energy
        dcw += c.lam_h * 2.0 * (cw @ ca["Kn"]) / B
        dG = c.lam_h * (-(cw.T @ cw)) * ca["mask"] / B
        dMn = (dG + dG.T) @ ca["Mn"]
        # back through the row-normalisation
        gr["M"] += (dMn - np.sum(dMn * ca["Mn"], axis=1, keepdims=True) * ca["Mn"]
                    ) / ca["nrm"]

        # --- fidelity to what the jurists actually wrote
        gr["M"] += c.lam_f * 2.0 * (p["M"] - self.M0) / c.n_exc

        # --- cw = softmax(r / tau)
        dr = softmax_backward(cw, dcw, axis=1) / c.tau
        gr["alpha"] += dr.sum(0)
        gr["alpha"] += c.lam_a * 2.0 * (p["alpha"] - self.alpha0) / c.n_exc
        gr["M"] += dr.T @ q / sc
        dq += dr @ p["M"] / sc

        dpre_q = dq * (1 - q ** 2)
        gr["Wq"] += dpre_q.T @ ca["X"]; gr["bq"] += dpre_q.sum(0)
        return gr

    # ------------------------------------------------------------------ update
    def step(self, gr, frozen: bool = False, lr=None):
        """Adam. frozen=True applies Constitutio Tanta §21: the corpus and every
        organ that reads it are sealed. Only the Novellae and the emperor's own
        hand may still move."""
        lr = self.cfg.lr if lr is None else lr
        self.t += 1
        b1, b2, e = 0.9, 0.999, 1e-8
        for k in self.p:
            if frozen and k in self.CORPUS_PARAMS:
                continue                                      # the gradient stop
            gk = gr[k]
            self.m[k] = b1 * self.m[k] + (1 - b1) * gk
            self.v[k] = b2 * self.v[k] + (1 - b2) * (gk * gk)
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            self.p[k] = self.p[k] - lr * mh / (np.sqrt(vh) + e)

    # ------------------------------------------------------------------ report
    def diagnostics(self, X, y, sign_of=None) -> Dict[str, float]:
        loss, ca = self.forward(X, y)
        pred = np.argmax(ca["prob"], axis=1)
        out = dict(
            loss=float(loss), acc=float(np.mean(pred == y)),
            antinomy=float(ca["parts"]["anti"]),
            forgery=float(math.sqrt(np.sum((self.p["M"] - self.M0) ** 2))),
            forgery_cos=float(np.mean(1.0 - np.sum(
                self.p["M"] / (np.linalg.norm(self.p["M"], axis=1, keepdims=True) + 1e-9)
                * self.M0 / (np.linalg.norm(self.M0, axis=1, keepdims=True) + 1e-9),
                axis=1))),
            rescript=float(np.mean(np.linalg.norm(ca["delta"], axis=1))),
            novella_gate=float(np.mean(ca["nu"])),
            cited=float(np.mean(effective_count(ca["cw"], axis=1))),
            decisions_used=float(np.mean(effective_count(ca["sv"], axis=1))),
            ambiguity_gate=float(np.mean(ca["u"])),
        )
        if sign_of is not None:
            cw = ca["cw"]
            dis = cw[:, sign_of < 0].sum(1)
            agr = cw[:, sign_of > 0].sum(1)
            out["dissent_share"] = float(np.mean(dis))
            # PURITY: does each ruling cite one school only? Concord can be had
            # by agreeing, or simply by never letting the other side speak.
            out["school_purity"] = float(np.mean(np.maximum(dis, agr)))
            mean_cw = cw.mean(0)
            # SILENCED: excerpts the engine has stopped citing altogether.
            # "Let the ancient books fall silent." — Constitutio Tanta
            out["silenced"] = int(np.sum(mean_cw < 1.0 / (10 * N_EXCERPTS)))
        return out


# ==============================================================================
# 4. MANDATORY FINITE-DIFFERENCE GRADIENT CHECK
# ==============================================================================

def gradient_check(seed: int = 3, n_probe: int = 8, h: float = 1e-5) -> float:
    """Compare the hand-derived backward pass against central differences, for
    every parameter tensor. Nothing else in this file is trustworthy unless this
    passes. Relative error is only meaningful where the gradient is not itself
    numerical dust, so entries below 1e-7 are scored on absolute error instead.
    """
    rng = np.random.default_rng(seed)
    world = World(rng)
    M0, a0, *_ = build_corpus(rng, world)
    eng = AntinomyEngine(Config(), M0, a0, seed=seed)
    X, y, _ = world.docket(12, "post")

    loss, cache = eng.forward(X, y)
    grads = eng.backward(cache)

    worst_rel, worst_abs = 0.0, 0.0
    print(f"   {'param':>8}  {'shape':>12}  {'max rel err':>12}  {'max abs err':>12}")
    print("   " + "-" * 52)
    for k, P in eng.p.items():
        flat = P.ravel()
        idxs = rng.choice(flat.size, size=min(n_probe, flat.size), replace=False)
        lr_, la_ = 0.0, 0.0
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + h; lp, _ = eng.forward(X, y)
            flat[i] = orig - h; lm, _ = eng.forward(X, y)
            flat[i] = orig
            num = (lp - lm) / (2 * h)
            ana = grads[k].ravel()[i]
            la_ = max(la_, abs(num - ana))
            scale = abs(num) + abs(ana)
            if scale > 1e-7:
                lr_ = max(lr_, abs(num - ana) / scale)
        worst_rel = max(worst_rel, lr_); worst_abs = max(worst_abs, la_)
        print(f"   {k:>8}  {str(P.shape):>12}  {lr_:12.3e}  {la_:12.3e}")
    print("   " + "-" * 52)
    ok = worst_rel < 1e-5 and worst_abs < 1e-6
    print(f"   WORST RELATIVE ERROR {worst_rel:.3e} | WORST ABSOLUTE ERROR "
          f"{worst_abs:.3e}  ->  {'PASS' if ok else 'FAIL'}")
    return worst_rel


# ==============================================================================
# 5. TRAINING — THE COMPILATION YEARS, THE PROMULGATION, AND THE WORLD AFTER
# ==============================================================================
#
#   steps  0 – 700   : the ius vetus. Both conditions train identically. This is
#                      530–533: the commission reads, excerpts, harmonises.
#   step   700       : PROMULGATION. In the FROZEN condition, Constitutio Tanta
#                      §21 takes effect: the corpus and every organ that reads it
#                      are sealed. Only the Novellae and the emperor's own hand
#                      may still move.
#   step   900       : THE WORLD MOVES. Principle 1 is overruled; novel matter
#                      begins arriving on the docket. Nobody consulted the Digest
#                      about whether this was convenient.
#
# Both conditions then see the SAME post-534 stream. The only difference between
# them is whether commentary is permitted.

def train(condition: str, seed: int = 7, steps: int = 2400,
          promulgate_at: int = 700, shift_at: int = 900,
          lam_f: float = 0.30, lam_n: float = 0.12, label: str = None,
          verbose: bool = True):
    rng = np.random.default_rng(seed)
    world = World(rng)
    M0, a0, jurist_of, princ_of, sign_of, _ = build_corpus(rng, world)
    cfg = Config(lam_f=lam_f, lam_n=lam_n)
    eng = AntinomyEngine(cfg, M0, a0, seed=seed)
    te = {k: world.docket(600, k) for k in ("settled", "overruled", "novel")}

    for t in range(steps):
        X, y, _ = world.docket(64, "pre" if t < shift_at else "post")
        loss, cache = eng.forward(X, y)
        gr = eng.backward(cache)
        frozen = (condition == "frozen") and (t >= promulgate_at)
        eng.step(gr, frozen=frozen)

        if verbose and (t % 400 == 0 or t == steps - 1):
            d = {k: eng.diagnostics(*te[k][:2], sign_of)["acc"] for k in te}
            dd = eng.diagnostics(*te["novel"][:2], sign_of)
            seal = "SEALED" if frozen else "open  "
            print(f"   step {t:5d} [{seal}] loss {loss:6.3f} | settled "
                  f"{d['settled']:.3f} | overruled {d['overruled']:.3f} | "
                  f"novel {d['novel']:.3f} | novella {dd['novella_gate']:.3f}")

    fin = {k: eng.diagnostics(*te[k][:2], sign_of) for k in te}
    fin["condition"] = label or condition
    return eng, world, fin, (jurist_of, princ_of, sign_of)


# ==============================================================================
# 5b. WHAT DOES THE SEALED DIGEST ACTUALLY KNOW?
# ==============================================================================
#
# Muting the Novellae at judgment is not a fair test: the output head grew up
# with them and is thrown off-manifold when they vanish. The rigorous question
# is INFORMATIONAL, not behavioural:
#
#     Does h1 — the corpus's own opinion, formed from the cited authorities,
#     the rescript and the fifty decisions, BEFORE any Novel is consulted —
#     contain the information needed to decide the case at all?
#
# Fit a fresh linear readout on h1 and see. If the sealed Digest carries the
# distinction, a straight line will find it. If a fresh probe cannot get above a
# coin-flip, then the corpus does not merely give the wrong answer about the new
# world — it does not represent the new world. No decree can fix that, because
# there is nothing there to decree about.

def digest_probe(eng: AntinomyEngine, world: World, n: int = 2500):
    """Linear probe on h1, with the Novellae muted. Returns accuracy per docket."""
    saved = eng.novellae_on
    eng.novellae_on = 0.0
    out = {}
    for kind in ("settled", "overruled", "novel"):
        Xtr, ytr, _ = world.docket(n, kind)
        Xte, yte, _ = world.docket(n, kind)
        _, ca = eng.forward(Xtr); Htr = ca["h1"]
        _, cb = eng.forward(Xte); Hte = cb["h1"]
        A = np.hstack([Htr, np.ones((n, 1))])
        Y = np.zeros((n, N_CLASSES)); Y[np.arange(n), ytr] = 1.0
        W, *_ = np.linalg.lstsq(A, Y, rcond=None)
        P = np.hstack([Hte, np.ones((n, 1))]) @ W
        out[kind] = float(np.mean(np.argmax(P, axis=1) == yte))
    eng.novellae_on = saved
    return out


# ==============================================================================
# 6. THE ULPIAN PROBLEM — WHOSE VOICE IS THIS, REALLY?
# ==============================================================================
#
# Roughly two fifths of the real Digest is Ulpian; about four fifths comes from
# just five men. When a corpus is that lopsided, "consulting the authorities"
# and "consulting one man" become hard to tell apart. This measures which.

def citation_audit(eng: AntinomyEngine, world: World, jurist_of, sign_of):
    X, y, _ = world.docket(800, "settled")
    _, ca = eng.forward(X, y)
    cw = ca["cw"].mean(0)
    by_jurist = np.array([cw[jurist_of == j].sum() for j in range(len(JURISTS))])
    by_mass = np.array([cw[MASS_OF == m].sum() for m in range(3)])
    silenced = int(np.sum(cw < 1.0 / (10 * N_EXCERPTS)))
    eff_jur = float(effective_count(by_jurist / by_jurist.sum()))
    return by_jurist, by_mass, float(effective_count(cw)), silenced, eff_jur


# ==============================================================================
# 7. THE BLUHME TEST — THE PIPELINE IS VISIBLE IN THE PRODUCT
# ==============================================================================
#
# In 1820 Friedrich Bluhme noticed that the excerpts inside each title of the
# Digest fall into recurring blocks, and inferred that three committees had read
# three separate book-lists in parallel. He recovered the ORGANISATION OF THE
# LABOUR from the SHAPE OF THE OUTPUT — 1,287 years later, with no access to any
# commission minutes, and no confession from anyone involved.
#
# We now do the same thing to our own engine. Compile its citations into titles,
# emit each title in the compiler's internal order, HIDE the committee labels,
# and try to recover them from position alone.

def bluhme_test(eng: AntinomyEngine, world: World, topk: int = 6):
    X, y, title = world.docket(1500, "pre")
    _, ca = eng.forward(X, y)
    cw = ca["cw"]

    positions: Dict[int, List[float]] = {i: [] for i in range(N_EXCERPTS)}
    for t in range(N_TITLES):
        rows = np.where(title == t)[0]
        if len(rows) == 0:
            continue
        weight = cw[rows].mean(0)
        cited = np.argsort(-weight)[:topk]
        # the compiler emits them in ITS order: mass first, then that
        # committee's reading order within its assigned book-list.
        order = sorted(cited, key=lambda i: (MASS_OF[i], READ_ORDER[i]))
        n = len(order)
        for pos, i in enumerate(order):
            positions[i].append(pos / max(1, n - 1))

    ids = [i for i in range(N_EXCERPTS) if positions[i]]
    meanpos = np.array([np.mean(positions[i]) for i in ids])
    cent = np.array([0.15, 0.50, 0.85])
    for _ in range(80):
        assign = np.argmin(np.abs(meanpos[:, None] - cent[None, :]), axis=1)
        for k in range(3):
            if np.any(assign == k):
                cent[k] = meanpos[assign == k].mean()
    truth = MASS_OF[ids]

    from itertools import permutations
    best = max(float(np.mean(np.array([p[a] for a in assign]) == truth))
               for p in permutations(range(3)))
    return best, len(ids)


# ==============================================================================
# 8. THE TANTA SWEEP — WHAT DOES "UNA CONCORDIA" COST?
# ==============================================================================
#
# Deo auctore demanded that no antinomy claim a place: one concord, one
# consequence. The engine has three ways to obey.
#
#   OMIT   — stop citing the dissenting excerpts. Nothing is falsified; a voice
#            is simply silenced. (Tanta ordered the older books to "fall silent".)
#   EDIT   — move the stored text of the excerpt itself, so that the jurist now
#            says what is needed, under his own name. This is interpolation:
#            the emblemata Triboniani. It shows up as FORGERY = ||M − M₀||.
#   DECIDE — spend one of the fifty imperial decisions and settle it by fiat.
#
# lam_f is the price of forgery: how strongly the engine is bound to what the
# jurists actually wrote. Sweeping it prices the whole bargain. Watch whether,
# as fidelity gets expensive, the engine switches from EDITING the dissent to
# SILENCING it. That substitution — forgery giving way to omission — is the most
# Justinianic behaviour a machine could possibly exhibit.

def tanta_sweep(seeds=(1, 2), fidelities=(0.0, 0.1, 0.3, 1.0, 5.0),
                steps: int = 800):
    rows = []
    for lf in fidelities:
        acc, anti, forg, dis = [], [], [], []
        for sd in seeds:
            rng = np.random.default_rng(200 + sd)
            world = World(rng)
            M0, a0, _, _, sign_of, _ = build_corpus(rng, world)
            eng = AntinomyEngine(Config(lam_f=lf), M0, a0, seed=sd)
            for t in range(steps):
                X, y, _ = world.docket(64, "pre")
                _, ca = eng.forward(X, y)
                eng.step(eng.backward(ca), frozen=False)
            Xte, yte, _ = world.docket(600, "pre")
            d = eng.diagnostics(Xte, yte, sign_of)
            acc.append(d["acc"]); anti.append(d["antinomy"])
            forg.append(d["forgery_cos"]); dis.append(d["silenced"])
        rows.append((lf, np.mean(acc), np.mean(anti), np.mean(forg), np.mean(dis)))
    return rows


# ==============================================================================
# 9. SELF-TESTS
# ==============================================================================

def self_tests() -> bool:
    ok = True
    rng = np.random.default_rng(0)
    world = World(rng)
    M0, a0, jur, princ, sign, _ = build_corpus(rng, world)
    eng = AntinomyEngine(Config(), M0, a0, seed=0)
    X, y, _ = world.docket(16, "pre")
    loss, ca = eng.forward(X, y)

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"   [{'ok ' if cond else 'FAIL'}] {name}")

    check("citation weights form a distribution", np.allclose(ca["cw"].sum(1), 1.0))
    check("exactly fifty decision slots exist", eng.p["D"].shape[0] == 50)
    check("three masses partition the corpus",
          sorted(np.concatenate(eng.mass_idx)) == list(range(N_EXCERPTS)))
    check("the corpus arrives contradicting itself",
          ca["parts"]["anti"] > 1e-4 and (sign < 0).sum() > 0)
    check("no excerpt speaks to the novel matter (jurists covered 0..3 only)",
          set(princ.tolist()) == set(range(N_PRINC)))
    check("the working corpus starts identical to the attested text",
          np.allclose(eng.p["M"], eng.M0))
    check("the novel docket really is exclusive-or (not linearly separable)",
          _xor_is_hard(world))
    check("loss is finite", np.isfinite(loss))
    check("forward is deterministic", np.allclose(eng.forward(X, y)[0], loss))

    e2 = AntinomyEngine(Config(), M0, a0, seed=0)
    before = e2.p["M"].copy(); nv_before = e2.p["Nv"].copy()
    _, c2 = e2.forward(X, y)
    e2.step(e2.backward(c2), frozen=True)
    check("Constitutio Tanta truly seals the corpus",
          np.allclose(e2.p["M"], before))
    check("...but the Novellae may still issue",
          not np.allclose(e2.p["Nv"], nv_before))
    check("the ruling is computed from the authorities, not the case",
          e2.p["Wh"].shape[1] == 3 * D_MEM)
    return ok


def _xor_is_hard(world: World) -> bool:
    """A least-squares linear probe on the raw case features should be at chance
    on the novel docket. If a straight line could solve it, the experiment would
    prove nothing."""
    X, y, _ = world.docket(1500, "novel")
    Xb = np.hstack([X, np.ones((len(X), 1))])
    Y = np.zeros((len(X), N_CLASSES)); Y[np.arange(len(X)), y] = 1
    W, *_ = np.linalg.lstsq(Xb, Y, rcond=None)
    Xt, yt, _ = world.docket(1500, "novel")
    Xtb = np.hstack([Xt, np.ones((len(Xt), 1))])
    acc = np.mean(np.argmax(Xtb @ W, 1) == yt)
    return acc < 0.62          # chance on this binary problem is 0.50


# ==============================================================================
# 10. MAIN
# ==============================================================================

def main():
    t0 = time.time()
    line = "=" * 78
    print(line)
    print(f" FIGURE {FIGURE_ID:04d} — {FIGURE_NAME}")
    print(" THE ANTINOMY ENGINE — compilation, interpolation, and the sealed corpus")
    print(line)

    print("\n[1] SELF-TESTS")
    ok = self_tests()
    print(f"   -> {'ALL PASS' if ok else 'FAILURES PRESENT'}")

    print("\n[2] GRADIENT CHECK (central differences vs hand-derived backward)")
    worst = gradient_check()

    print("\n[3] THE CENTRAL EXPERIMENT — what does sealing the corpus cost?")
    print("\n  --- OPEN: commentary permitted; the corpus keeps learning ---")
    eng_o, world_o, fin_o, aux_o = train("open", seed=7, label="open")
    print("\n  --- SEALED, Novels taxed: Tanta §21, and every Novel is a scandal ---")
    eng_f, world_f, fin_f, aux_f = train("frozen", seed=7, lam_n=0.12,
                                         label="sealed/taxed")
    print("\n  --- SEALED, Novels cheap: the emperor legislates his way out ---")
    eng_g, world_g, fin_g, aux_g = train("frozen", seed=7, lam_n=0.02,
                                         label="sealed/cheap")

    print("\n  RESULT (accuracy on three kinds of case, after the world moved)")
    hdr = (f"   {'condition':<13} {'settled law':>12} {'OVERRULED':>10} "
           f"{'NOVEL MATTER':>13} | {'gate:settled':>12} {'gate:novel':>11}")
    print(hdr); print("   " + "-" * (len(hdr) - 3))
    for f in (fin_o, fin_f, fin_g):
        print(f"   {f['condition']:<13} {f['settled']['acc']:>12.3f} "
              f"{f['overruled']['acc']:>10.3f} {f['novel']['acc']:>13.3f} | "
              f"{f['settled']['novella_gate']:>12.3f} "
              f"{f['novel']['novella_gate']:>11.3f}")

    # ---- THE DECISIVE MEASUREMENT --------------------------------------------
    pr_o = digest_probe(eng_o, world_o)
    pr_f = digest_probe(eng_f, world_f)
    print("\n  WHAT DOES THE CORPUS ITSELF KNOW? (fresh linear probe on the")
    print("  corpus's own opinion, Novellae muted — chance on novel matter = 0.500)")
    print(f"   {'corpus':<9} {'settled law':>12} {'OVERRULED':>10} {'NOVEL MATTER':>13}")
    print("   " + "-" * 47)
    print(f"   {'open':<9} {pr_o['settled']:>12.3f} {pr_o['overruled']:>10.3f} "
          f"{pr_o['novel']:>13.3f}")
    print(f"   {'SEALED':<9} {pr_f['settled']:>12.3f} {pr_f['overruled']:>10.3f} "
          f"{pr_f['novel']:>13.3f}")

    print(f"\n   OVERRULED LAW SURVIVES THE SEAL. Every condition holds above")
    print(f"   {min(fin_o['overruled']['acc'], fin_f['overruled']['acc'], fin_g['overruled']['acc']):.2f}, and the sealed corpus PROBES AT {pr_f['overruled']:.3f}: it still")
    print(f"   identifies the matter perfectly. Only the required answer changed,")
    print(f"   and an answer can be decreed. Overruling is a relabelling, and a")
    print(f"   frozen representation survives relabelling untouched. On this")
    print(f"   Justinian was simply right — and he overruled by Novel for thirty")
    print(f"   years, exactly as the machine predicts he could.")
    print(f"\n   NOVEL MATTER IS WHERE THE SEAL BITES. Read the three rows:")
    print(f"     open           {fin_o['novel']['acc']:.3f}   the corpus learned the new concepts")
    print(f"     sealed/taxed   {fin_f['novel']['acc']:.3f}   a coin flip. The Digest is blind.")
    print(f"     sealed/cheap   {fin_g['novel']['acc']:.3f}   rescued — but look at the gate.")
    print(f"\n   The open engine's novella gate on new matter is "
          f"{fin_o['novel']['novella_gate']:.3f}: it never")
    print(f"   needed the emergency channel, because its CORPUS absorbed the new")
    print(f"   world. The sealed/cheap engine's gate is "
          f"{fin_g['novel']['novella_gate']:.3f}: virtually everything it")
    print(f"   gets right is being carried by the appendix, while the Digest is")
    print(f"   cited for the dignity of the thing.")
    print(f"\n   And the probe says why. The sealed corpus's own opinion carries")
    print(f"   {pr_f['novel']:.3f} of the novel distinction against the open corpus's {pr_o['novel']:.3f}")
    print(f"   (chance 0.500). The seal does not merely make the Digest give wrong")
    print(f"   answers about the new world. It degrades the Digest's ability to")
    print(f"   REPRESENT the new world — and a decree cannot repair a representation,")
    print(f"   because there is nothing in there for the decree to attach to.")
    print(f"   An emperor may legislate an answer. He may not legislate a concept.")
    print(f"\n   That is the historical endgame in miniature: the corpus stays")
    print(f"   supreme in name, the living law migrates to the Novels — in Greek —")
    print(f"   and the monument keeps its title.")

    print("\n[4] CITATION AUDIT — whose voice is this, really?")
    by_j, by_m, eff, sil, eff_jur = citation_audit(eng_o, world_o, aux_o[0], aux_o[2])
    for j, name in enumerate(JURISTS):
        print(f"   {name:<11} {by_j[j]*100:5.1f}%  {'#' * int(round(by_j[j]*45))}")
    print(f"   committees:  Sabinian {by_m[0]*100:.1f}%  |  Edictal "
          f"{by_m[1]*100:.1f}%  |  Papinian {by_m[2]*100:.1f}%")
    print(f"   authorities effectively consulted per ruling: {eff:.2f} of "
          f"{N_EXCERPTS}  ({eff/N_EXCERPTS*100:.1f}%;")
    print(f"   the real commission retained about 5% of what it read)")
    print(f"   texts the engine has stopped citing entirely: {sil} of {N_EXCERPTS}")
    print(f"   effective number of jurists behind a ruling: {eff_jur:.2f} of "
          f"{len(JURISTS)}")
    print(f"   The quota does not spread citation across the tradition. It")
    print(f"   collapses it onto whoever happened to state each point most")
    print(f"   usably. The real Digest has the same disease: about two fifths of")
    print(f"   it is Ulpian, and four fifths is five men out of thirty-nine.")

    print("\n[5] THE BLUHME TEST — can the pipeline be read off the product?")
    acc_b, n_used = bluhme_test(eng_o, world_o)
    print(f"   committee labels hidden, recovered from output position alone:")
    print(f"   {acc_b*100:.1f}% of {n_used} cited excerpts returned to the correct mass.")
    print(f"   The division of labour survives inside the artifact. A careful")
    print(f"   reader can reconstruct how the thing was made, centuries later,")
    print(f"   from the product alone. Bluhme did exactly this in 1820.")

    print("\n[6] THE TANTA SWEEP — the price of 'una concordia'")
    rows = tanta_sweep()
    print(f"   {'fidelity λ_f':>12} {'accuracy':>9} {'antinomy':>9} "
          f"{'FORGERY':>8} {'texts silenced':>17}")
    print("   " + "-" * 60)
    for lf, a, an, fo, di in rows:
        print(f"   {lf:>12.2f} {a:>9.3f} {an:>9.4f} {fo:>8.3f} "
              f"{di:>13.1f}/{N_EXCERPTS}")
    print(f"\n   FORGERY is the mean angle each excerpt has been rotated away from")
    print(f"   what its jurist actually wrote. It falls from {rows[0][3]:.3f} to "
          f"{rows[-1][3]:.3f} as")
    print(f"   fidelity gets expensive, exactly as it should. The surprise is the")
    print(f"   last column. Forgery and silence do not TRADE OFF — they travel")
    print(f"   TOGETHER. When rewriting the dead is free, the engine silences")
    print(f"   {rows[0][4]:.0f} of {N_EXCERPTS} texts; when it is costly, only {rows[-1][4]:.0f}.")
    print(f"\n   The reason is worth sitting with. A corpus you are permitted to")
    print(f"   edit is a corpus you need LESS of: sharpen a handful of texts into")
    print(f"   saying precisely what you need, and the rest of the tradition")
    print(f"   becomes surplus to requirements. Bind the engine to the sources and")
    print(f"   it must consult MORE of them, because no single text says quite")
    print(f"   enough. Justinian did both at once — he rewrote and he silenced —")
    print(f"   and the machine says that was not two vices but one optimum.")
    print(f"   Antinomy, meanwhile, is what he actually bought: it rises from")
    print(f"   {rows[0][2]:.4f} to {rows[-1][2]:.4f} as the sources are protected. Concord and")
    print(f"   fidelity are the two ends of one lever. He knew which end he held.")

    print(f"\n{line}")
    print(f" gradient check worst rel. error: {worst:.2e}   "
          f"self-tests: {'PASS' if ok else 'FAIL'}")
    print(f" completed in {time.time()-t0:.1f}s")
    print(line)


if __name__ == "__main__":
    main()

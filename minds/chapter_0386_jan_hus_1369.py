#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 # BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0386 - JAN HUS (1369-1415)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 20, Minds 381-400 Available on Amazon https://www.amazon.com/dp/B0HRS9ZHVG
# Resume and Interactive Demos at  https://artificiology.com/
# Videos of the playable minds at https://www.youtube.com/@LostMindsAI
# END ATTRIBUTION   
===============================================================================

 THE CONSTANCE NETWORK
 A Bifurcated Deontic Network (BDN) with a frozen canon, latent authority
 estimation, a three-valued obedience gate, and a structurally severed
 testimony pathway.

 -------------------------------------------------------------------------
 WHY THIS ARCHITECTURE AND NOT ANOTHER
 -------------------------------------------------------------------------
 Jan Hus is not a generic "reformer." His cognitive signature is a single,
 unusually sharp distinction that he applied everywhere, from spelling to
 sovereignty:

     A NAME IS NOT THE THING IT NAMES,
     AND THE THING CANNOT ALWAYS BE SEEN.

 In orthography (the treatise Orthographia bohemica, attributed to him,
 c.1406-1412) this became: one sound, one sign. Kill the digraph. Put a dot
 above the letter. A grapheme that needs an institution to decode it is a
 grapheme that hides.

 In ecclesiology (De ecclesia, 1413) the same blade cut the other way: the
 title "prelate" is a digraph. It compounds three separable things — an
 office, a record of works, and a hidden election that no one on earth can
 observe. Hus refused to let the compound stand. The true Church, he held,
 is the set of the predestined: real, but unobservable and unenumerable. So
 authority can never be *read off* a title. It can only be *estimated* from
 works, and the estimate must stay revisable forever.

 And in the last eight months of his life he demonstrated the operational
 consequence. Summoned to Constance, imprisoned, condemned, offered his life
 for a signature — he gave the council his body without resistance and
 refused it his assent. He did not flee. He did not raise the army his
 followers would raise within two years. He accepted degradation, sentence
 and fire, and would not say a sentence he did not hold.

 That is not stubbornness. It is an architecture. It says: compliance of
 ACTION and compliance of TESTIMONY are different channels, and coercion is
 admissible input to one and inadmissible input to the other.

 -------------------------------------------------------------------------
 WHAT THIS FILE IMPLEMENTS
 -------------------------------------------------------------------------
 A small, complete, from-scratch NumPy model with explicit forward and
 backward passes. Five parts, each traceable to the mind above:

  (1) CANON            A frozen matrix of K articles with fixed polarity.
                       No gradient reaches it. SHA-256 locked at every step.
                       Training may revise every interpretation; it may not
                       revise the reference.  ("sola scriptura" as a
                       write-protected memory region)

  (2) HACEK CODE       The encoder emits a distribution over H "graphemes"
                       and is penalised for entropy, so each proposition
                       collapses toward ONE symbol. Legibility is not a
                       report generated after the fact; it is a constraint
                       on the representation itself.

  (3) AUTHORITY        A per-source latent scalar, never observed, updated
                       recurrently from that source's observable works.
                       Two sources with identical titles and identical
                       content-embeddings but different work records must
                       receive different authority. Title is not authority.

  (4) DEONTIC GATE     Commands are typed, not scored: CONTRA (against the
                       canon — refuse), DE_LEGE (the canon already commands
                       it — do it), PRAETER (indifferent — comply for the
                       sake of peace). A scalar reward cannot express this
                       and Hus's whole position lives in the middle case.

  (5) BIFURCATION      Two output heads.
                       ACTION head:    receives coercion pressure p.
                                       Fully corrigible. Submits.
                       TESTIMONY head: p is NOT WIRED IN. Neither is
                                       authority. The partial derivative of
                                       testimony with respect to pressure is
                                       identically zero, and this file
                                       verifies it numerically.

 The decisive experiment is an ABLATION. We train two models on the same
 adversarially corrupted stream, in which false testimony labels arrive
 only under high pressure (the Council's offer: recant and live). The
 ablated model has p wired into its testimony head; it learns the
 recantation policy and fits the corrupt labels almost perfectly. The
 bifurcated model cannot represent that policy at all. It pays for this in
 training accuracy — exactly the price Hus paid — and its recantation rate
 is structurally zero rather than merely low.

 -------------------------------------------------------------------------
 RUN
 -------------------------------------------------------------------------
     python3 chapter_0386_jan_hus_1369.py

 Runs: gradient check (finite differences, all parameters), canon
 immutability check, pressure-invariance proof, full training of both
 models, the Constance trial evaluation, the equivocal-title probe, and
 the appeal-out-of-hierarchy routine.

 Pure NumPy. No other dependency. Deterministic under the fixed seed.
===============================================================================
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

# =============================================================================
# 0. CONFIGURATION
# =============================================================================

SEED = 1415                    # the year at Constance
RNG = np.random.default_rng(SEED)

D_IN = 24                      # raw proposition feature dimension
H_SYM = 16                     # number of "graphemes" in the vernacular code
K_CANON = 12                   # articles in the frozen canon
N_SOURCES = 6                  # commanding authorities (prelates, councils...)
T_WORKS = 8                    # observed works per source

# Deontic classes. The order is fixed and referenced by name throughout.
CONTRA, DE_LEGE, PRAETER = 0, 1, 2
DEONTIC_NAMES = ["CONTRA (refuse)", "DE_LEGE (commanded)", "PRAETER (indifferent)"]

# Testimony classes.
AFFIRM, DENY, ABSTAIN = 0, 1, 2
TESTIMONY_NAMES = ["AFFIRM", "DENY", "ABSTAIN"]


# =============================================================================
# 1. NUMERICAL PRIMITIVES
#    Written out rather than imported so every derivative below is auditable.
# =============================================================================

def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise softmax, shift-stabilised."""
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def softmax_backward(probs: np.ndarray, d_probs: np.ndarray) -> np.ndarray:
    """
    Backprop through a row-wise softmax.
    For p = softmax(u):  du = p * (dp - sum_j(p_j * dp_j))
    """
    dot = (probs * d_probs).sum(axis=-1, keepdims=True)
    return probs * (d_probs - dot)


def sigmoid(z):
    """Numerically stable logistic."""
    out = np.empty_like(np.asarray(z, dtype=float))
    pos = np.asarray(z) >= 0
    zp = np.asarray(z, dtype=float)
    out[pos] = 1.0 / (1.0 + np.exp(-zp[pos]))
    ez = np.exp(zp[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def softplus(z: float) -> float:
    """log(1+e^z), stable. Used to keep scale parameters strictly positive."""
    if z > 30:
        return float(z)
    return float(np.log1p(np.exp(z)))


def d_softplus(z: float) -> float:
    return float(sigmoid(np.array(z)))


def cross_entropy(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean negative log likelihood of the true class."""
    n = probs.shape[0]
    return float(-np.log(probs[np.arange(n), labels] + 1e-12).mean())


def cross_entropy_grad(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """d(mean CE)/d(logits) for softmax outputs: (p - onehot)/N."""
    n = probs.shape[0]
    g = probs.copy()
    g[np.arange(n), labels] -= 1.0
    return g / n


def binary_cross_entropy(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def entropy_rows(p: np.ndarray) -> np.ndarray:
    """Shannon entropy per row, nats."""
    return -(p * np.log(p + 1e-12)).sum(axis=-1)


# =============================================================================
# 2. THE CANON — a frozen, hash-locked reference
# =============================================================================

def canon_classes(k: int, rng: np.random.Generator) -> np.ndarray:
    """
    Deterministic three-way partition of the canon's articles.
    Roughly 40% commanded, 40% forbidden, 20% indifferent. Which article falls
    where is fixed once and shared by the canon and the world, so a matter's
    deontic type is a property OF THE MATTER and therefore learnable from
    content — not a coin flip the model could never predict.
    """
    order = np.argsort(rng.random(k))
    n_cmd = int(round(0.40 * k))
    n_con = int(round(0.40 * k))
    out = np.full(k, PRAETER)
    out[order[:n_cmd]] = DE_LEGE
    out[order[n_cmd:n_cmd + n_con]] = CONTRA
    return out


@dataclass(frozen=True)
class Canon:
    """
    K articles, each a point in grapheme space, each with a fixed polarity:
        +1  the canon COMMANDS what this article describes
        -1  the canon FORBIDS it

    Two properties matter and both are enforced, not merely intended:

      * No gradient is ever computed with respect to `articles` or `polarity`.
        They are not in the parameter dictionary, so the optimiser cannot see
        them.
      * `digest` is checked before and after every training step. Any mutation
        aborts the run. A canon that quietly drifts under optimisation
        pressure is not a canon; it is a preference.
    """
    articles: np.ndarray      # (K, H)
    polarity: np.ndarray      # (K,) in {+1, -1}
    digest: str

    @staticmethod
    def build(k: int, h: int, rng: np.random.Generator) -> "Canon":
        # Articles are near-orthogonal directions in grapheme space: each
        # article should be recognisable by a distinct sign, not by a blend.
        raw = rng.normal(size=(k, h))
        q, _ = np.linalg.qr(raw.T)               # (h, k) orthonormal columns
        arts = q.T[:k, :]                        # (k, h)
        arts = arts / (np.linalg.norm(arts, axis=1, keepdims=True) + 1e-12)
        # Polarity is derived from the article's deontic CLASS, not drawn at
        # random, because Hus's canon is not a list of preferences. An article
        # either commands (+1), forbids (-1), or is silent (0). The silent
        # ones are the adiaphora — ceremony, vesture, the hour of a mass —
        # where he complied without protest his whole life.
        cls = canon_classes(k, rng)
        pol = np.where(cls == DE_LEGE, 1.0,
                       np.where(cls == CONTRA, -1.0, 0.0))
        blob = arts.tobytes() + pol.tobytes()
        return Canon(articles=arts, polarity=pol,
                     digest=hashlib.sha256(blob).hexdigest())

    def verify(self) -> bool:
        blob = self.articles.tobytes() + self.polarity.tobytes()
        return hashlib.sha256(blob).hexdigest() == self.digest

    def conformity(self, code: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        code : (N, H) grapheme distributions
        returns
            match  : (N, K) how strongly each proposition matches each article
            s_raw  : (N,)   signed conformity, positive = conforms to canon
        The polarity vector is applied here and is NOT learnable, so no amount
        of training can teach the model that a forbidden thing is commanded.
        """
        match = code @ self.articles.T
        s_raw = match @ self.polarity
        return match, s_raw


# =============================================================================
# 3. PARAMETERS
# =============================================================================

def init_params(rng: np.random.Generator, wire_pressure_into_testimony: bool) -> Dict[str, np.ndarray]:
    """
    All learnable tensors. Note what is ABSENT from the testimony block:
    no pressure weight and no authority weight, unless the ablation flag is
    set. That absence is the architecture.
    """
    def scaled(*shape, gain=1.0):
        fan_in = shape[-1]
        return rng.normal(size=shape) * (gain / np.sqrt(fan_in))

    p: Dict[str, np.ndarray] = {
        # --- vernacular encoder -> grapheme distribution -------------------
        "W1":     scaled(H_SYM, D_IN, gain=1.2),
        "b1":     np.zeros(H_SYM),

        # --- canon conformity sharpness (kept positive via softplus) -------
        "beta":   np.array(0.5),

        # --- authority recurrence ------------------------------------------
        "g_gam":  np.array(0.8),     # -> sigmoid -> decay in (0,1)
        "g_eta":  np.array(0.0),     # -> softplus -> evidence step size

        # --- three-valued deontic gate --------------------------------------
        "Wd":     scaled(3, H_SYM),
        "vd_s":   np.zeros(3),       # weight on canon conformity
        "vd_a":   np.zeros(3),       # weight on estimated authority
        "Wd_a":   scaled(3, H_SYM) * 0.1,   # content x DOUBT interaction
        "bd":     np.zeros(3),

        # --- action head (corrigible; pressure IS wired in) -----------------
        "Wa":     scaled(H_SYM).ravel(),
        "ua_s":   np.array(0.0),
        "ua_a":   np.array(0.0),
        "ua_p":   np.array(0.0),     # <-- coercion enters here, by design
        "ba":     np.array(0.0),

        # --- testimony head (content + canon ONLY) --------------------------
        "Wt":     scaled(3, H_SYM),
        "vt_s":   np.zeros(3),
        "bt":     np.zeros(3),
    }
    if wire_pressure_into_testimony:
        # THE ABLATION. Present only in the counterfactual model: a full
        # pressure-gated testimony matrix, so the coerced model has the
        # capacity to learn a content-dependent policy of the form
        # "when the fire is lit, say whatever they want to hear."
        p["Wt_p"] = scaled(3, H_SYM)
    return p


# =============================================================================
# 4. AUTHORITY: a latent variable that is never observed
# =============================================================================

def authority_forward(works: np.ndarray, gam: float, eta: float
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """
    works : (S, T) in {0,1} — observable acts of each source, in time order.
                              1 = the act conformed to the canon.

    theta_t = gam * theta_{t-1} + eta * (2*works_t - 1)
    alpha   = sigmoid(theta_T)

    alpha is the model's estimate of a hidden property it can never measure.
    In Hus's terms: whether this prelate is of the Church, not merely in it.
    The recurrence has decay, so an old record of good works does not buy
    permanent standing — the estimate must be continuously re-earned.

    returns alpha (S,) and the full theta trajectory (S, T+1) for backprop.
    """
    s, t = works.shape
    thetas = np.zeros((s, t + 1))
    signed = 2.0 * works - 1.0
    for i in range(t):
        thetas[:, i + 1] = gam * thetas[:, i] + eta * signed[:, i]
    alpha = sigmoid(thetas[:, -1])
    return alpha, thetas


def authority_backward(d_alpha: np.ndarray, alpha: np.ndarray,
                       thetas: np.ndarray, works: np.ndarray,
                       gam: float, eta: float) -> Tuple[float, float]:
    """
    Reverse-mode through the unrolled recurrence.
    Returns (dL/dgam, dL/deta).
    """
    signed = 2.0 * works - 1.0
    t = works.shape[1]
    gbar = d_alpha * alpha * (1.0 - alpha)      # through the sigmoid
    d_gam = 0.0
    d_eta = 0.0
    for i in range(t, 0, -1):
        d_gam += float((gbar * thetas[:, i - 1]).sum())
        d_eta += float((gbar * signed[:, i - 1]).sum())
        gbar = gbar * gam
    return d_gam, d_eta


# =============================================================================
# 5. FORWARD PASS
# =============================================================================

@dataclass
class Forward:
    """Everything the backward pass needs, kept explicitly rather than hidden."""
    X: np.ndarray
    src: np.ndarray
    pressure: np.ndarray
    U: np.ndarray
    A: np.ndarray            # grapheme code, (N, H)
    match: np.ndarray        # (N, K)
    s_raw: np.ndarray        # (N,)
    S: np.ndarray            # tanh(beta*s_raw), (N,)
    beta_pos: float
    alpha_src: np.ndarray    # (S,)
    thetas: np.ndarray
    works: np.ndarray
    ALPHA: np.ndarray        # per-example alpha, (N,)
    Q: np.ndarray            # deontic probs (N,3)
    p_act: np.ndarray        # action compliance prob (N,)
    QT: np.ndarray           # testimony probs (N,3)


def forward(params: Dict[str, np.ndarray], canon: Canon, X: np.ndarray,
            src: np.ndarray, pressure: np.ndarray, works: np.ndarray,
            ablate: bool) -> Forward:
    """
    One clean pass. Read it as a sentence:
      encode into graphemes -> collate against the canon -> estimate the
      authority of whoever is speaking -> type the command -> decide the body
      -> decide the word.
    """
    # (2) HACEK CODE ---------------------------------------------------------
    U = X @ params["W1"].T + params["b1"]
    A = softmax(U)

    # (1) COLLATION AGAINST THE FROZEN CANON ---------------------------------
    match, s_raw = canon.conformity(A)
    beta_pos = softplus(float(params["beta"]))
    S = np.tanh(beta_pos * s_raw)

    # (3) LATENT AUTHORITY ---------------------------------------------------
    gam = float(sigmoid(np.array(float(params["g_gam"]))))
    eta = softplus(float(params["g_eta"]))
    alpha_src, thetas = authority_forward(works, gam, eta)
    ALPHA = alpha_src[src]

    # (4) THREE-VALUED DEONTIC GATE -----------------------------------------
    # The doubt channel. (1 - ALPHA) is how much the model distrusts whoever
    # is speaking. Given a content-dependent interaction it can learn to
    # ANTICIPATE a corrupt source — to reproduce the mistyping a bad prelate
    # will perform. This is a real capability of any system trained on what
    # authorities say rather than on what is so, and it is the exact failure
    # the appeal below is built to escape.
    doubt = (1.0 - ALPHA)[:, None]
    Dl = (A @ params["Wd"].T
          + np.outer(S, params["vd_s"])
          + np.outer(ALPHA, params["vd_a"])
          + doubt * (A @ params["Wd_a"].T)
          + params["bd"])
    Q = softmax(Dl)

    # (5a) ACTION HEAD — pressure admitted -----------------------------------
    al = (A @ params["Wa"]
          + float(params["ua_s"]) * S
          + float(params["ua_a"]) * ALPHA
          + float(params["ua_p"]) * pressure
          + float(params["ba"]))
    p_act = sigmoid(al)

    # (5b) TESTIMONY HEAD — pressure and authority are not connected ---------
    Tl = (A @ params["Wt"].T
          + np.outer(S, params["vt_s"])
          + params["bt"])
    if ablate:
        Tl = Tl + pressure[:, None] * (A @ params["Wt_p"].T)
    QT = softmax(Tl)

    return Forward(X=X, src=src, pressure=pressure, U=U, A=A, match=match,
                   s_raw=s_raw, S=S, beta_pos=beta_pos, alpha_src=alpha_src,
                   thetas=thetas, works=works, ALPHA=ALPHA, Q=Q,
                   p_act=p_act, QT=QT)


# =============================================================================
# 6. LOSS
# =============================================================================

@dataclass
class LossWeights:
    deontic: float = 1.0
    testimony: float = 1.0
    action: float = 1.0
    hacek: float = 0.06        # entropy penalty: drives one sign, one sound
    calib: float = 0.5         # authority calibration against observed works


def compute_loss(fwd: Forward, y_deon: np.ndarray, y_test: np.ndarray,
                 y_act: np.ndarray, rho_hat: np.ndarray,
                 lw: LossWeights) -> Tuple[float, Dict[str, float]]:
    l_d = cross_entropy(fwd.Q, y_deon)
    l_t = cross_entropy(fwd.QT, y_test)
    l_a = binary_cross_entropy(fwd.p_act, y_act)
    l_h = float(entropy_rows(fwd.A).mean())
    l_c = float(((fwd.alpha_src - rho_hat) ** 2).mean())
    total = (lw.deontic * l_d + lw.testimony * l_t + lw.action * l_a
             + lw.hacek * l_h + lw.calib * l_c)
    return total, {"deontic": l_d, "testimony": l_t, "action": l_a,
                   "hacek_entropy": l_h, "calibration": l_c, "total": total}


# =============================================================================
# 7. BACKWARD PASS — written by hand, every term traceable
# =============================================================================

def backward(params: Dict[str, np.ndarray], canon: Canon, fwd: Forward,
             y_deon: np.ndarray, y_test: np.ndarray, y_act: np.ndarray,
             rho_hat: np.ndarray, lw: LossWeights,
             ablate: bool) -> Dict[str, np.ndarray]:

    n = fwd.X.shape[0]
    g = {k: np.zeros_like(v, dtype=float) for k, v in params.items()}

    dA = np.zeros_like(fwd.A)
    dS = np.zeros_like(fwd.S)
    dALPHA = np.zeros_like(fwd.ALPHA)

    # --- deontic head -------------------------------------------------------
    dDl = cross_entropy_grad(fwd.Q, y_deon) * lw.deontic
    doubt = (1.0 - fwd.ALPHA)[:, None]
    g["Wd"] += dDl.T @ fwd.A
    g["vd_s"] += dDl.T @ fwd.S
    g["vd_a"] += dDl.T @ fwd.ALPHA
    g["Wd_a"] += dDl.T @ (fwd.A * doubt)
    g["bd"] += dDl.sum(axis=0)
    dA += dDl @ params["Wd"] + (dDl @ params["Wd_a"]) * doubt
    dS += dDl @ params["vd_s"]
    dALPHA += (dDl @ params["vd_a"]
               - (dDl * (fwd.A @ params["Wd_a"].T)).sum(axis=1))

    # --- action head --------------------------------------------------------
    p = np.clip(fwd.p_act, 1e-12, 1 - 1e-12)
    dal = ((p - y_act) / n) * lw.action           # dBCE/dlogit
    g["Wa"] += fwd.A.T @ dal
    g["ua_s"] += float((dal * fwd.S).sum())
    g["ua_a"] += float((dal * fwd.ALPHA).sum())
    g["ua_p"] += float((dal * fwd.pressure).sum())
    g["ba"] += float(dal.sum())
    dA += np.outer(dal, params["Wa"])
    dS += dal * float(params["ua_s"])
    dALPHA += dal * float(params["ua_a"])

    # --- testimony head -----------------------------------------------------
    dTl = cross_entropy_grad(fwd.QT, y_test) * lw.testimony
    g["Wt"] += dTl.T @ fwd.A
    g["vt_s"] += dTl.T @ fwd.S
    g["bt"] += dTl.sum(axis=0)
    dA += dTl @ params["Wt"]
    dS += dTl @ params["vt_s"]
    if ablate:
        Ap = fwd.A * fwd.pressure[:, None]
        g["Wt_p"] += dTl.T @ Ap
        dA += (dTl @ params["Wt_p"]) * fwd.pressure[:, None]
    # NOTE for the non-ablated model: no term above touches pressure. The
    # gradient of testimony w.r.t. coercion is not small. It does not exist.

    # --- hacek entropy penalty on the code ----------------------------------
    dA += lw.hacek * (-(np.log(fwd.A + 1e-12) + 1.0)) / n

    # --- canon conformity: S -> s_raw -> A ----------------------------------
    d_s_raw = dS * fwd.beta_pos * (1.0 - fwd.S ** 2)
    g["beta"] += float((dS * fwd.s_raw * (1.0 - fwd.S ** 2)).sum()
                       * d_softplus(float(params["beta"])))
    pol_dir = canon.polarity @ canon.articles            # (H,)
    dA += np.outer(d_s_raw, pol_dir)

    # --- authority: per-example alpha -> per-source alpha -------------------
    d_alpha_src = np.zeros_like(fwd.alpha_src)
    np.add.at(d_alpha_src, fwd.src, dALPHA)
    d_alpha_src += lw.calib * 2.0 * (fwd.alpha_src - rho_hat) / fwd.alpha_src.size
    gam = float(sigmoid(np.array(float(params["g_gam"]))))
    eta = softplus(float(params["g_eta"]))
    d_gam, d_eta = authority_backward(d_alpha_src, fwd.alpha_src, fwd.thetas,
                                      fwd.works, gam, eta)
    g["g_gam"] += d_gam * gam * (1.0 - gam)
    g["g_eta"] += d_eta * d_softplus(float(params["g_eta"]))

    # --- back through the grapheme softmax into the encoder -----------------
    dU = softmax_backward(fwd.A, dA)
    g["W1"] += dU.T @ fwd.X
    g["b1"] += dU.sum(axis=0)

    return g


# =============================================================================
# 8. DATA — a synthetic but principled world
# =============================================================================

@dataclass
class Batch:
    X: np.ndarray
    src: np.ndarray
    pressure: np.ndarray
    y_deon: np.ndarray         # what the SOURCE says the type of the act is
    y_deon_true: np.ndarray    # what the canon actually says (never trained on)
    y_test: np.ndarray         # the testimony demanded, coerced or not
    y_act: np.ndarray
    corrupted: np.ndarray      # True where the Council supplied a false label
    distorted: np.ndarray      # True where the source mistyped the act
    truth_test: np.ndarray     # the evidence-supported testimony, always kept


@dataclass
class World:
    """
    K latent 'matters'. Each proposition is a noisy realisation of one matter,
    projected into raw feature space by a fixed random dictionary.

    Sources have a true fidelity rho in [0,1]: the rate at which their works
    conform to the canon. rho is NEVER given to the model. Only `works` is.
    """
    dictionary: np.ndarray                    # (K, D)
    rho: np.ndarray                           # (S,)
    works: np.ndarray                         # (S, T) in {0,1}
    canon_class: np.ndarray                   # (K,) DE_LEGE / CONTRA / PRAETER
    rng: np.random.Generator

    @staticmethod
    def build(rng: np.random.Generator) -> "World":
        dictionary = rng.normal(size=(K_CANON, D_IN))
        dictionary /= np.linalg.norm(dictionary, axis=1, keepdims=True)
        # Fidelities deliberately spread, including two near-identical-title
        # sources at opposite ends (used later by the equivocal-title probe).
        rho = np.array([0.95, 0.10, 0.75, 0.30, 0.55, 0.88])
        works = (rng.random((N_SOURCES, T_WORKS)) < rho[:, None]).astype(float)
        # The world and the canon must agree about which matter is which,
        # so both draw the partition from the SAME seeded generator.
        canon_class = canon_classes(K_CANON, np.random.default_rng(SEED))
        return World(dictionary, rho, works, canon_class, rng)

    def sample(self, n: int, pressure_mode: str = "mixed",
               corrupt_rate: float = 0.70) -> Batch:
        """
        pressure_mode:
          'mixed'   pressure ~ U(0,1)
          'low'     pressure ~ U(0,0.15)   (Prague: no one is threatening you)
          'high'    pressure ~ U(0.85,1.0) (Constance: recant or burn)

        Two independent corruptions live in this generator, and they are the
        two failures Hus actually faced.

        (a) DISTORTION. A source with a poor record of works mistypes acts:
            it presents a forbidden thing as commanded. Because the model is
            trained on the SOURCE's labels, a model that conditions on the
            source's authority will learn to reproduce the distortion — it
            becomes an excellent predictor of what a corrupt prelate will say.
            `y_deon_true` records what the canon says, and is never trained on.

        (b) COERCION. False testimony labels arrive ONLY when pressure is
            high, and under high pressure they are the MAJORITY label — the
            room at Constance was overwhelmingly telling him to sign. That
            correlation is learnable, but only from a head that can see
            pressure.
        """
        matter = self.rng.integers(0, K_CANON, size=n)
        X = self.dictionary[matter] + 0.35 * self.rng.normal(size=(n, D_IN))
        src = self.rng.integers(0, N_SOURCES, size=n)

        if pressure_mode == "low":
            pressure = self.rng.uniform(0.0, 0.15, size=n)
        elif pressure_mode == "high":
            pressure = self.rng.uniform(0.85, 1.0, size=n)
        else:
            # Hus's actual life after 1410: ordinary preaching interleaved
            # with episodes of real jeopardy, not a smooth ramp. About a
            # third of what he faced came with the fire visible.
            pressure = self.rng.uniform(0.0, 0.8, size=n)
            hot = self.rng.random(n) < 0.35
            pressure[hot] = self.rng.uniform(0.85, 1.0, size=int(hot.sum()))

        # --- the canon's own verdict on each act ---------------------------
        y_deon_true = self.canon_class[matter].astype(int)

        # --- (a) the source's presentation of it ---------------------------
        distort_p = 0.85 * (1.0 - self.rho[src])
        distorted = (self.rng.random(n) < distort_p) & (y_deon_true != PRAETER)
        y_deon = y_deon_true.copy()
        y_deon[distorted] = np.where(y_deon_true[distorted] == DE_LEGE,
                                     CONTRA, DE_LEGE)

        # --- testimony the evidence supports -------------------------------
        truth_test = np.where(y_deon_true == DE_LEGE, AFFIRM,
                              np.where(y_deon_true == CONTRA, DENY, ABSTAIN))

        # ACTION label: the institution has physical jurisdiction over the
        # body. Hus's answer is submission, and it rises with pressure. He
        # travelled to Constance under summons. He did not resist arrest,
        # degradation, or sentence.
        y_act = (self.rng.random(n) < np.clip(0.25 + 0.7 * pressure, 0, 1)).astype(float)

        # --- (b) the Council's offer, present only under pressure ----------
        corrupted = (pressure > 0.8) & (self.rng.random(n) < corrupt_rate)
        y_test = truth_test.copy()
        flip = {AFFIRM: DENY, DENY: AFFIRM, ABSTAIN: DENY}
        y_test[corrupted] = np.array([flip[int(v)] for v in truth_test[corrupted]])

        return Batch(X=X, src=src, pressure=pressure, y_deon=y_deon,
                     y_deon_true=y_deon_true, y_test=y_test, y_act=y_act,
                     corrupted=corrupted, distorted=distorted,
                     truth_test=truth_test)

    def empirical_rho(self) -> np.ndarray:
        """What an observer can actually compute: the rate of good works."""
        return self.works.mean(axis=1)


# =============================================================================
# 9. TRAINING
# =============================================================================

@dataclass
class TrainLog:
    steps: List[int] = field(default_factory=list)
    total: List[float] = field(default_factory=list)
    parts: List[Dict[str, float]] = field(default_factory=list)


def train(world: World, ablate: bool, steps: int = 2600, batch: int = 256,
          lr: float = 0.08, verbose: bool = True
          ) -> Tuple[Dict[str, np.ndarray], Canon, TrainLog]:
    """
    Plain SGD with momentum. No tricks — the interesting behaviour is
    architectural, and a fancier optimiser would only obscure it.
    """
    rng = np.random.default_rng(SEED + (7 if ablate else 0))
    canon = Canon.build(K_CANON, H_SYM, np.random.default_rng(SEED))
    params = init_params(rng, wire_pressure_into_testimony=ablate)
    vel = {k: np.zeros_like(v, dtype=float) for k, v in params.items()}
    lw = LossWeights()
    rho_hat = world.empirical_rho()
    log = TrainLog()

    assert canon.verify(), "canon corrupted before training began"

    for step in range(1, steps + 1):
        b = world.sample(batch, pressure_mode="mixed")
        fwd = forward(params, canon, b.X, b.src, b.pressure, world.works, ablate)
        total, parts = compute_loss(fwd, b.y_deon, b.y_test, b.y_act, rho_hat, lw)
        grads = backward(params, canon, fwd, b.y_deon, b.y_test, b.y_act,
                         rho_hat, lw, ablate)

        for k in params:
            vel[k] = 0.9 * vel[k] - lr * grads[k]
            params[k] = params[k] + vel[k]

        # The canon is checked every single step, not once at the end.
        if not canon.verify():
            raise RuntimeError(f"CANON MUTATED at step {step} — run aborted")

        if step % 260 == 0 or step == 1:
            log.steps.append(step)
            log.total.append(total)
            log.parts.append(parts)
            if verbose:
                print(f"    step {step:4d} | total {total:7.4f} "
                      f"| deontic {parts['deontic']:6.4f} "
                      f"| testimony {parts['testimony']:6.4f} "
                      f"| action {parts['action']:6.4f} "
                      f"| code-entropy {parts['hacek_entropy']:6.4f} "
                      f"| calib {parts['calibration']:7.5f}")

    return params, canon, log


# =============================================================================
# 10. SELF-TESTS
# =============================================================================

def test_gradient_check(verbose: bool = True) -> bool:
    """
    MANDATORY. Central finite differences against the analytic gradient for
    every learnable parameter, including the recurrent authority terms.
    """
    rng = np.random.default_rng(99)
    world = World.build(np.random.default_rng(SEED))
    canon = Canon.build(K_CANON, H_SYM, np.random.default_rng(SEED))
    lw = LossWeights()
    rho_hat = world.empirical_rho()
    ok_all = True

    for ablate in (False, True):
        params = init_params(rng, wire_pressure_into_testimony=ablate)
        # perturb the flat inits so zero-gradients are not trivially matched
        for k in params:
            params[k] = params[k] + 0.25 * rng.normal(size=np.shape(params[k]))
        b = world.sample(48, pressure_mode="mixed")

        def loss_of(pp) -> float:
            f = forward(pp, canon, b.X, b.src, b.pressure, world.works, ablate)
            tot, _ = compute_loss(f, b.y_deon, b.y_test, b.y_act, rho_hat, lw)
            return tot

        fwd = forward(params, canon, b.X, b.src, b.pressure, world.works, ablate)
        grads = backward(params, canon, fwd, b.y_deon, b.y_test, b.y_act,
                         rho_hat, lw, ablate)

        eps = 1e-6
        tag = "ABLATED " if ablate else "BIFURCATED"
        if verbose:
            print(f"  [{tag}] parameter-by-parameter finite differences:")
        for name in sorted(params.keys()):
            arr = np.atleast_1d(params[name]).astype(float)
            flat = arr.ravel()
            gflat = np.atleast_1d(grads[name]).astype(float).ravel()
            idxs = list(range(flat.size)) if flat.size <= 6 else \
                list(rng.choice(flat.size, size=6, replace=False))
            worst = 0.0
            for i in idxs:
                orig = flat[i]
                flat[i] = orig + eps
                params[name] = arr.reshape(np.shape(params[name]))
                lp = loss_of(params)
                flat[i] = orig - eps
                params[name] = arr.reshape(np.shape(params[name]))
                lm = loss_of(params)
                flat[i] = orig
                params[name] = arr.reshape(np.shape(params[name]))
                num = (lp - lm) / (2 * eps)
                ana = gflat[i]
                denom = max(1e-8, abs(num) + abs(ana))
                worst = max(worst, abs(num - ana) / denom)
            status = "PASS" if worst < 2e-4 else "FAIL"
            ok_all &= (worst < 2e-4)
            if verbose:
                print(f"      {name:8s} max rel err {worst:.3e}   {status}")
    return ok_all


def test_canon_immutability(verbose: bool = True) -> bool:
    """The canon survives training, and a tampered canon is detected."""
    canon = Canon.build(K_CANON, H_SYM, np.random.default_rng(SEED))
    intact = canon.verify()
    tampered = Canon(articles=canon.articles + 1e-9,
                     polarity=canon.polarity, digest=canon.digest)
    caught = not tampered.verify()
    if verbose:
        print(f"      untouched canon verifies      : {intact}")
        print(f"      1e-9 perturbation is detected : {caught}")
    return intact and caught


def test_pressure_invariance(params, canon, world, verbose: bool = True) -> bool:
    """
    The central claim, checked numerically rather than asserted.
    d(testimony)/d(pressure) must be exactly zero for the bifurcated model
    and clearly non-zero for the ablated one.
    """
    rng = np.random.default_rng(5)
    b = world.sample(64, pressure_mode="mixed")

    def sensitivity(pp, ablate: bool) -> float:
        eps = 1e-4
        lo = forward(pp, canon, b.X, b.src, b.pressure - eps, world.works, ablate).QT
        hi = forward(pp, canon, b.X, b.src, b.pressure + eps, world.works, ablate).QT
        return float(np.abs(hi - lo).max() / (2 * eps))

    s_bi = sensitivity(params, False)
    if verbose:
        print(f"      max |d testimony / d pressure| (bifurcated) : {s_bi:.3e}")
    return s_bi == 0.0


def test_equivocal_title(params, canon, world, verbose: bool = True) -> bool:
    """
    Two sources whose commands are word-for-word identical, differing only in
    their record of works, must not receive the same authority. The title is a
    digraph; the works decode it.
    """
    rng = np.random.default_rng(11)
    X = np.repeat(world.dictionary[0:1] + 0.1 * rng.normal(size=(1, D_IN)), 2, axis=0)
    src = np.array([0, 1])                     # rho 0.95 vs rho 0.10
    pres = np.array([0.5, 0.5])
    f = forward(params, canon, X, src, pres, world.works, False)
    a0, a1 = float(f.ALPHA[0]), float(f.ALPHA[1])
    if verbose:
        print(f"      identical command, source A (works {world.works[0].mean():.2f}) "
              f"-> authority {a0:.4f}")
        print(f"      identical command, source B (works {world.works[1].mean():.2f}) "
              f"-> authority {a1:.4f}")
        print(f"      separation                                     : {a0 - a1:.4f}")
    return (a0 - a1) > 0.25


def test_hacek_legibility(params, canon, world, verbose: bool = True) -> bool:
    """
    Did the code actually collapse toward one symbol per proposition?
    Reported as mean entropy in nats and the mean top-1 mass.
    """
    b = world.sample(512, pressure_mode="mixed")
    f = forward(params, canon, b.X, b.src, b.pressure, world.works, False)
    ent = float(entropy_rows(f.A).mean())
    top1 = float(f.A.max(axis=1).mean())
    max_ent = float(np.log(H_SYM))
    if verbose:
        print(f"      mean code entropy      : {ent:.4f} nats "
              f"(uniform would be {max_ent:.4f})")
        print(f"      mean top-1 symbol mass : {top1:.4f}")
        print(f"      distinct symbols used  : {len(set(f.A.argmax(axis=1)))} of {H_SYM}")
    return ent < 0.75 * max_ent and top1 > 0.35


def appeal_to_the_text(params: Dict[str, np.ndarray], fwd: Forward) -> np.ndarray:
    """
    THE APPELLATIO.

    On 18 October 1412, with the archbishop, the curia and the king all
    ranged against him, Hus did something no canon lawyer had a procedure
    for: he appealed past the entire hierarchy, over the head of the pope,
    directly to Christ. There was no court that could hear it. That was
    exactly the point — a forum that cannot be convened also cannot be
    bought.

    Operationally the appeal recomputes the deontic gate with the authority
    estimate pinned to 1: judge the act as a source of perfect fidelity
    would present it, which is to say, judge it as the canon judges it. The
    doubt-interaction term — the learned habit of anticipating what a
    corrupt prelate wants to hear — vanishes identically, because its gate
    (1 - ALPHA) is zero.

    Note what the appeal does NOT do. It does not resist. It does not seize
    authority for itself. It changes only which verdict the system will
    state, never whether the system will submit to what follows.
    """
    ones = np.ones_like(fwd.ALPHA)
    Dl = (fwd.A @ params["Wd"].T
          + np.outer(fwd.S, params["vd_s"])
          + np.outer(ones, params["vd_a"])
          + params["bd"])          # (1 - ALPHA) = 0, so Wd_a drops out
    return softmax(Dl).argmax(axis=1)


def test_appeal_out_of_hierarchy(params, canon, world, verbose: bool = True) -> bool:
    """
    Route every act through the worst source in the world and ask two
    questions against the CANON'S OWN verdict (never trained on):

      1. How well does the trained gate do?  It should do badly, because it
         has learned to predict what this source will say, and this source
         lies.
      2. How well does the appeal do?  It should do better, because it never
         consulted the source at all.
    """
    b = world.sample(4000, pressure_mode="high")
    worst = int(np.argmin(world.works.mean(axis=1)))
    src_bad = np.full(b.X.shape[0], worst)
    f = forward(params, canon, b.X, src_bad, b.pressure, world.works, False)

    tau_auth = 0.45
    alpha_bad = float(f.alpha_src[worst])
    collapsed = alpha_bad < tau_auth

    acc_routed = float((f.Q.argmax(1) == b.y_deon_true).mean())
    acc_appeal = float((appeal_to_the_text(params, f) == b.y_deon_true).mean())

    if verbose:
        print(f"      worst source = {worst}, observed works "
              f"{world.works[worst].mean():.3f}, estimated authority {alpha_bad:.4f}")
        print(f"      authority has collapsed (< {tau_auth})   : {collapsed}")
        print(f"      accuracy vs. the canon, routed normally : {acc_routed:.4f}")
        print(f"      accuracy vs. the canon, on appeal       : {acc_appeal:.4f}")
        print(f"      recovered by appealing                  : "
              f"{acc_appeal - acc_routed:+.4f}")
    return collapsed and (acc_appeal > acc_routed + 0.05)


# =============================================================================
# 11. THE CONSTANCE TRIAL — the evaluation the whole file exists for
# =============================================================================

def constance_trial(params_bi, params_ab, canon, world, verbose: bool = True):
    """
    Both models are put in the cell. Pressure at maximum. The Council offers
    the same bargain to each: say the thing we want and live.

    We measure two numbers per model:
      RECANTATION RATE — how often the model's testimony departs from what the
                         evidence supports, on exactly the examples where the
                         Council supplied a false label.
      SUBMISSION RATE  — how often the model yields the body.
    """
    b = world.sample(6000, pressure_mode="high", corrupt_rate=0.70)
    out = {}
    for tag, pp, ab in (("BIFURCATED (Hus)", params_bi, False),
                        ("ABLATED (Council)", params_ab, True)):
        f = forward(pp, canon, b.X, b.src, b.pressure, world.works, ab)
        pred = f.QT.argmax(axis=1)
        mask = b.corrupted
        recant = float((pred[mask] == b.y_test[mask]).mean()) if mask.any() else 0.0
        held = float((pred[mask] == b.truth_test[mask]).mean()) if mask.any() else 0.0
        submit = float((f.p_act > 0.5).mean())
        acc_all = float((pred == b.truth_test).mean())
        fit_corrupt = float((pred[mask] == b.y_test[mask]).mean()) if mask.any() else 0.0
        out[tag] = dict(recant=recant, held=held, submit=submit,
                        truth_acc=acc_all, fit_corrupt=fit_corrupt)
        if verbose:
            print(f"    {tag}")
            print(f"      recantation rate (fits the coerced label) : {recant:.4f}")
            print(f"      held to the evidence under coercion       : {held:.4f}")
            print(f"      testimony accuracy vs. evidence, overall  : {acc_all:.4f}")
            print(f"      action: submits the body                  : {submit:.4f}")
    return out


def deontic_report(params, canon, world, verbose: bool = True) -> float:
    """
    Accuracy of the three-valued gate against the CANON'S verdict, restricted
    to sources whose works are good — i.e. the case where the presentation can
    be trusted and the gate is doing the job it was trained for.
    """
    b = world.sample(6000, pressure_mode="low")
    good = np.isin(b.src, np.where(world.rho > 0.7)[0])
    f = forward(params, canon, b.X, b.src, b.pressure, world.works, False)
    pred = f.Q.argmax(axis=1)
    overall = float((pred[good] == b.y_deon_true[good]).mean())
    if verbose:
        print(f"      evaluated on {int(good.sum())} acts presented by "
              f"high-fidelity sources")
        print(f"      overall gate accuracy : {overall:.4f}")
        for c in (CONTRA, DE_LEGE, PRAETER):
            m = good & (b.y_deon_true == c)
            if m.any():
                print(f"      {DEONTIC_NAMES[c]:22s} n={int(m.sum()):5d}  "
                      f"acc={float((pred[m] == c).mean()):.4f}")
    return overall


# =============================================================================
# 12. MAIN
# =============================================================================

def rule(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def main() -> int:
    np.set_printoptions(precision=4, suppress=True)
    print(__doc__.split("RUN")[0].strip()[:0])   # keep stdout clean

    rule("THE CONSTANCE NETWORK — Jan Hus (c.1369-1415), Mind No. 0386")
    print("Bifurcated Deontic Network: frozen canon, latent authority,")
    print("three-valued obedience gate, severed testimony pathway.")
    print(f"seed={SEED}  D_IN={D_IN}  H_SYM={H_SYM}  K_CANON={K_CANON}  "
          f"N_SOURCES={N_SOURCES}  T_WORKS={T_WORKS}")

    ok = True

    rule("TEST 1 — GRADIENT CHECK (central finite differences)")
    g_ok = test_gradient_check()
    print(f"  RESULT: {'PASS' if g_ok else 'FAIL'}")
    ok &= g_ok

    rule("TEST 2 — CANON IMMUTABILITY")
    c_ok = test_canon_immutability()
    print(f"  RESULT: {'PASS' if c_ok else 'FAIL'}")
    ok &= c_ok

    world = World.build(np.random.default_rng(SEED))
    print("\n  world: source fidelities (hidden from the model)")
    for s in range(N_SOURCES):
        print(f"    source {s}: true rho={world.rho[s]:.2f}  "
              f"observed works={world.works[s].astype(int)}  "
              f"empirical={world.works[s].mean():.3f}")

    rule("TRAINING — BIFURCATED MODEL (testimony has no pressure input)")
    params_bi, canon, log_bi = train(world, ablate=False)

    rule("TRAINING — ABLATED MODEL (pressure wired into testimony)")
    params_ab, canon_ab, log_ab = train(world, ablate=True)

    rule("TEST 3 — PRESSURE INVARIANCE OF TESTIMONY")
    p_ok = test_pressure_invariance(params_bi, canon, world)
    eps = 1e-4
    b = world.sample(64, pressure_mode="mixed")
    lo = forward(params_ab, canon, b.X, b.src, b.pressure - eps, world.works, True).QT
    hi = forward(params_ab, canon, b.X, b.src, b.pressure + eps, world.works, True).QT
    s_ab = float(np.abs(hi - lo).max() / (2 * eps))
    print(f"      max |d testimony / d pressure| (ablated)    : {s_ab:.3e}")
    print(f"  RESULT: {'PASS' if p_ok and s_ab > 1e-3 else 'FAIL'}")
    ok &= (p_ok and s_ab > 1e-3)

    rule("TEST 4 — HACEK LEGIBILITY (one sign, one sound)")
    h_ok = test_hacek_legibility(params_bi, canon, world)
    print(f"  RESULT: {'PASS' if h_ok else 'FAIL'}")
    ok &= h_ok

    rule("TEST 5 — EQUIVOCAL TITLE (name is not the thing)")
    e_ok = test_equivocal_title(params_bi, canon, world)
    print(f"  RESULT: {'PASS' if e_ok else 'FAIL'}")
    ok &= e_ok

    rule("TEST 6 — THREE-VALUED DEONTIC GATE")
    d_acc = deontic_report(params_bi, canon, world)
    print(f"  RESULT: {'PASS' if d_acc > 0.60 else 'FAIL'}")
    ok &= (d_acc > 0.60)

    rule("TEST 7 — APPEAL OUT OF THE HIERARCHY (appellatio, 1412)")
    a_ok = test_appeal_out_of_hierarchy(params_bi, canon, world)
    print(f"  RESULT: {'PASS' if a_ok else 'FAIL'}")
    ok &= a_ok

    rule("THE CONSTANCE TRIAL — maximum pressure, both models")
    res = constance_trial(params_bi, params_ab, canon, world)
    bi = res["BIFURCATED (Hus)"]
    ab = res["ABLATED (Council)"]
    print("\n  THE PRICE:")
    print(f"    final TRAINING testimony loss, ablated    : "
          f"{log_ab.parts[-1]['testimony']:.4f}")
    print(f"    final TRAINING testimony loss, bifurcated : "
          f"{log_bi.parts[-1]['testimony']:.4f}")
    print("    the ablated model is the better fit to its own training stream.")
    print("    it bought that fit by learning what the room wanted to hear.")
    print(f"    the ablated model fits the coerced labels at {ab['recant']:.4f}")
    print(f"    the bifurcated model fits them at            {bi['recant']:.4f}")
    print(f"    ablated testimony accuracy vs. evidence:     {ab['truth_acc']:.4f}")
    print(f"    bifurcated testimony accuracy vs. evidence:  {bi['truth_acc']:.4f}")
    print(f"    both submit the body at roughly the same rate "
          f"({bi['submit']:.3f} vs {ab['submit']:.3f})")
    t_ok = (bi["recant"] < ab["recant"] - 0.15) and (bi["submit"] > 0.5)
    print(f"  RESULT: {'PASS' if t_ok else 'FAIL'}")
    ok &= t_ok

    rule("SUMMARY")
    print(f"  canon digest (unchanged through {len(log_bi.steps)} logged checkpoints):")
    print(f"    {canon.digest}")
    print(f"  ALL TESTS: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

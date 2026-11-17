#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 # BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0375 · Ibn Khaldun (1332-1406)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 19, Minds 361-380 Available on Amazon https://www.amazon.com/dp/B0HQX147Y7
# Resume and Interactive Demos at https://artificiology.com/
# END ATTRIBUTION 
 THE UMRAN ENGINE : a report-adjudication architecture built on the
 Muqaddimah's three cognitive doctrines.
================================================================================

WHY THIS ARCHITECTURE AND NOT A TRANSFORMER
-------------------------------------------
Ibn Khaldun's method is not "attend over everything you have stored and
retrieve the best-supported answer". It is almost the opposite. His famous
move against al-Mas'udi -- who reported, on an unbroken chain of respectable
transmitters, that Moses mustered 600,000 fighting men -- was to refuse the
chain and run the claim through a *generative model of what a society can
physically do*. Jacob's household entered Egypt seventy strong; four
generations cannot compound seventy into six hundred thousand; no realm of
that size and yield could feed such a host; the Persians at their height
fielded far fewer. The report is therefore false, and the impeccability of
its transmitters is beside the point.

That is a specific and unusual claim about how a mind should work:

  (1) MUTABAQA  -- "conformity to the nature of civilization". Truth of a
      transmitted report is decided by passing it through an internal
      generative model of the possible, NOT by the authority of the source.
      Authority may break a tie. It may never overturn an impossibility.

  (2) MALAKA    -- a skill is a "firmly rooted quality acquired by doing a
      certain action and repeating it", which *colours* the soul; and once
      coloured, the soul's isti'dad -- its preparedness to receive another
      colouring -- is weakened. Competence is therefore FINITE and the
      habits INTERFERE. A mind is a budget, not a library.

  (3) MIZAN     -- "The intellect is a correct scale. Its indications are
      completely certain and in no way wrong. However, the intellect should
      not be used to weigh such matters as [what] lies beyond the level of
      the intellect... One might compare it with a man who sees a scale in
      which gold is being weighed, and wants to weigh mountains in it."
      An instrument has a calibrated range. Outside it the instrument does
      not become noisy -- it becomes confidently wrong. So the instrument
      must report its own range.

And over the top of all three sits ASABIYYA and the dynastic cycle: solidarity
is what lets a group act as one mind, it is strongest in the lean and the
marginal, it is eroded by the very success it produces, and no dynasty holds
power past three generations. Applied to a mixture of experts this is not a
metaphor. It is a scheduler. A module that wins routing dominance accumulates
tenure and "luxury" (reward), its solidarity with its coalition decays, and a
leaner module with intact solidarity takes the throne. Expert collapse is
prevented structurally, by mandatory turnover, instead of by a load-balancing
penalty bolted on afterwards.

THE TASK THE MODEL IS TRAINED ON
--------------------------------
Report adjudication under an authority/plausibility conflict.

A synthetic "umran" (civilisation) is generated from real Khaldunian laws:
compound demographic ceilings, the militia-support ceiling, the Khaldun-Laffer
revenue curve, asabiyya decay across generations, the harsh-terrain/solidarity
coupling, and the three-generation dynastic limit. A *report* about a polity is
then emitted, either faithful or corrupted by violating exactly one law.

Each report also carries an ISNAD DOSSIER: transmitter rank, chain length,
number of corroborating chains, whether the transmitter is a court panegyrist,
sectarian alignment, recency.

  TRAIN / IID TEST : the dossier correlates with truth at rho ~= 0.85.
                     (This is what a medieval corpus actually looks like:
                     respectable chains usually do carry true reports.)
  MUNKAR TEST      : the correlation is INVERTED. Impeccable chains carry
                     the impossible. This is the al-Mas'udi case exactly.

A model that learns the authority shortcut scores ~85% on IID and collapses
below chance on Munkar. The whole point of the architecture is to make that
shortcut structurally unavailable: the isnad channel is hard-wired so it can
only shift the verdict inside a band whose width SHRINKS as the plausibility
engine grows confident. Gold may be weighed on the scale; mountains may not.

A naive control model -- identical capacity, isnad features simply
concatenated into the world path -- is trained alongside as the ablation.

IMPLEMENTATION NOTES
--------------------
Pure NumPy. Every gradient is derived and written by hand; there is no
autodiff anywhere in this file. A central finite-difference gradient check
runs over every parameter tensor before training and must pass. Run the file
directly to execute the full suite.

    python3 chapter_0375_ibn_khaldun_1332.py

================================================================================
"""

from __future__ import annotations

import math
import time
import numpy as np

# ==============================================================================
# SECTION 0 -- small numerical helpers
# ==============================================================================

EPS = 1e-9


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic."""
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Row-wise softmax, shift-stabilised."""
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def softmax_backward(y: np.ndarray, dy: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Backprop through softmax given its OUTPUT y (not its input).
    dx = y * (dy - sum(dy*y))
    """
    s = (dy * y).sum(axis=axis, keepdims=True)
    return y * (dy - s)


def softabs(x: np.ndarray, delta: float = 1e-4) -> np.ndarray:
    """
    Smooth |x| = sqrt(x^2 + delta).

    The band that authority is allowed to move the verdict inside is a function
    of how confident the plausibility engine is, i.e. of |p|. The true absolute
    value has a kink at p = 0 -- exactly where the engine is undecided and the
    chain is supposed to matter most -- which would leave the model
    non-differentiable at the one place it is doing the most interesting thing.
    The smoothed form is identical to within 1e-2 everywhere it matters and
    differentiable everywhere.
    """
    return np.sqrt(x * x + delta)


def softabs_grad(x: np.ndarray, delta: float = 1e-4) -> np.ndarray:
    return x / np.sqrt(x * x + delta)


def bce_with_logits(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Elementwise binary cross entropy on logits, stable form:
        max(z,0) - z*y + log(1 + exp(-|z|))
    """
    z = logits
    return np.maximum(z, 0.0) - z * targets + np.log1p(np.exp(-np.abs(z)))


def bce_with_logits_grad(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """d/dz of bce_with_logits = sigmoid(z) - y."""
    return sigmoid(logits) - targets


# ==============================================================================
# SECTION 1 -- THE UMRAN GENERATOR
#
# A synthetic civilisation whose laws are Ibn Khaldun's, stated as the
# Muqaddimah states them. These laws are NOT given to the model. The model must
# recover enough of them to judge whether a report about a polity is possible.
# ==============================================================================

class UmranGenerator:
    """
    Generates polities and the reports told about them.

    The laws in force (Muqaddimah references in brackets):

      L1  COMPOUND CEILING       P <= P0 * r**g
          A population cannot exceed compound growth from its founding stock.
          [Bk I, the critique of al-Mas'udi's 600,000: Jacob entered Egypt with
          seventy souls and Moses is four generations downstream.]

      L2  SUPPORT CEILING        A <= kappa * P * (0.4 + 0.6*a)
          "Every realm may have as large a militia as it can hold and support,
          but no more." A cohesive polity can mobilise a larger fraction, but
          the fraction is bounded and the base is the true population.

      L3  KHALDUN-LAFFER         R  = c * t * (1-t)**beta * P
          Revenue is single-peaked in the assessment rate. Early dynasties take
          large revenue from small assessments; late dynasties take small
          revenue from large ones. [Bk III, on taxation.]

      L4  ASABIYYA DECAY         a  = (0.30 + 0.70*h) * exp(-g/tau) * (1 - 0.7*L)
          Solidarity is highest in harsh country (h) and decays with each
          generation (g) and with luxury (L).

      L5  THREE-GENERATION LIMIT A dynasty cannot be reported as still standing
          at g > 3 with negligible solidarity and no external protector.
          "The term of life of a dynasty does not normally exceed three
          generations", three generations being about 120 years.

      L6  SEDENTARY SOFTENING    Reported solidarity cannot greatly exceed the
          solidarity that the polity's terrain, age and luxury permit.
    """

    # law constants
    KAPPA = 0.08      # max mobilisable fraction of population, before cohesion
    BETA = 1.60       # curvature of the revenue curve
    TAU = 2.20        # generations-constant of asabiyya decay
    REV_C = 0.01      # revenue scale

    # feature layout for the world vector (12 dims). Kept in log space where the
    # underlying law is multiplicative, so that a law becomes a linear
    # inequality -- learnable, but only if the model actually looks here.
    WORLD_DIM = 12
    ISNAD_DIM = 6

    WORLD_NAMES = [
        "log_P0", "generations_g", "growth_r", "terrain_h", "luxury_L",
        "tax_rate_t", "log_P_reported", "log_A_reported", "log_R_reported",
        "duration_D_reported", "asabiyya_reported", "city_count_C",
    ]
    ISNAD_NAMES = [
        "transmitter_rank", "chain_length", "corroborating_chains",
        "court_panegyrist", "sect_alignment", "recency",
    ]

    # violation taxonomy -- one law broken per false report
    VIOLATIONS = ["compound_ceiling", "support_ceiling", "laffer_inversion",
                  "dynastic_overrun", "sedentary_vigour"]

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------- polity
    def _sample_polity(self) -> dict:
        rng = self.rng
        g = int(rng.integers(1, 7))                       # generations elapsed
        P0 = float(np.exp(rng.uniform(np.log(50), np.log(5000))))
        r = float(rng.uniform(1.6, 2.6))                  # growth per generation
        h = float(rng.uniform(0.0, 1.0))                  # terrain harshness
        # luxury rises with the age of the dynasty, with noise
        L = float(np.clip(rng.beta(1.5 + 0.9 * g, 4.0 - 0.35 * g), 0.0, 0.98))
        t = float(rng.uniform(0.02, 0.75))                # assessment rate
        C = int(rng.integers(1, 40))                      # cities

        P = P0 * (r ** g)
        a = (0.30 + 0.70 * h) * math.exp(-g / self.TAU) * (1.0 - 0.7 * L)
        A = self.KAPPA * P * (0.40 + 0.60 * a)
        R = self.REV_C * t * ((1.0 - t) ** self.BETA) * P
        return dict(g=g, P0=P0, r=r, h=h, L=L, t=t, C=C, P=P, a=a, A=A, R=R)

    # --------------------------------------------------------------- report
    def _emit_report(self, pol: dict, truthful: bool) -> tuple[dict, str]:
        """
        Produce the numbers a chronicler claims. A truthful report perturbs the
        real values only by ordinary transmission noise. A false report breaks
        exactly one law, hard enough to be detectable by anyone holding a model
        of the possible -- and invisible to anyone checking only the chain.
        """
        rng = self.rng
        noise = lambda: float(np.exp(rng.normal(0.0, 0.22)))   # ~22% log noise

        P_rep = pol["P"] * noise()
        A_rep = pol["A"] * noise()
        R_rep = pol["R"] * noise()
        D_rep = float(pol["g"]) + float(rng.normal(0.0, 0.45))
        a_rep = float(np.clip(pol["a"] + rng.normal(0.0, 0.055), 0.001, 1.0))

        if truthful:
            return dict(P=P_rep, A=A_rep, R=R_rep, D=D_rep, a=a_rep), "none"

        which = self.VIOLATIONS[int(rng.integers(0, len(self.VIOLATIONS)))]

        if which == "compound_ceiling":
            # the 600,000 Israelites: a population no lineage could compound to
            P_rep *= float(rng.uniform(1.9, 4.5))
            A_rep = self.KAPPA * P_rep * (0.40 + 0.60 * pol["a"]) * noise()
            R_rep = self.REV_C * pol["t"] * ((1 - pol["t"]) ** self.BETA) * P_rep

        elif which == "support_ceiling":
            # a host far larger than the realm can ration and pay
            A_rep *= float(rng.uniform(1.9, 4.0))

        elif which == "laffer_inversion":
            # late-dynasty heavy assessment reported as yielding great revenue
            R_rep = self.REV_C * pol["t"] * (1.0 + 1.1 * pol["t"]) * pol["P"] * noise()
            if pol["t"] < 0.35:       # inversion only bites at high rates
                R_rep *= float(rng.uniform(1.8, 3.2))

        elif which == "dynastic_overrun":
            # a dynasty still standing well past the three-generation limit
            D_rep = float(pol["g"]) + float(rng.uniform(1.8, 4.5))
            a_rep = float(np.clip(pol["a"] * rng.uniform(0.05, 0.3), 0.001, 1.0))

        elif which == "sedentary_vigour":
            # soft, wealthy, old -- yet reported as fiercely cohesive
            a_rep = float(np.clip(pol["a"] + rng.uniform(0.14, 0.34), 0.0, 1.5))

        return dict(P=P_rep, A=A_rep, R=R_rep, D=D_rep, a=a_rep), which

    # -------------------------------------------------------------- encoding
    def _encode_world(self, pol: dict, rep: dict) -> np.ndarray:
        return np.array([
            math.log(pol["P0"]),
            float(pol["g"]),
            pol["r"],
            pol["h"],
            pol["L"],
            pol["t"],
            math.log(max(rep["P"], 1.0)),
            math.log(max(rep["A"], 1.0)),
            math.log(max(rep["R"], 1e-3)),
            rep["D"],
            rep["a"],
            float(pol["C"]),
        ], dtype=np.float64)

    def _encode_isnad(self, truthful: bool, rho: float) -> np.ndarray:
        """
        The source dossier. `rho` is the probability that the dossier's quality
        agrees with the report's truth. rho=0.85 in the ordinary corpus;
        rho=0.15 in the Munkar (repudiated) corpus, where the finest chains
        carry the impossible.
        """
        rng = self.rng
        agrees = rng.random() < rho
        strong = truthful if agrees else (not truthful)

        if strong:
            rank = rng.uniform(0.62, 1.00)      # celebrated transmitter
            chain = rng.uniform(0.60, 1.00)     # short, well-attested chain
            corrob = rng.uniform(0.55, 1.00)    # several independent chains
            paneg = rng.uniform(0.00, 0.35)     # not a court flatterer
        else:
            rank = rng.uniform(0.00, 0.42)
            chain = rng.uniform(0.00, 0.45)
            corrob = rng.uniform(0.00, 0.40)
            paneg = rng.uniform(0.60, 1.00)
        sect = rng.uniform(0.0, 1.0)            # pure noise: sectarian alignment
        recency = rng.uniform(0.0, 1.0)         # pure noise: how recent
        return np.array([rank, chain, corrob, paneg, sect, recency], dtype=np.float64)

    # ----------------------------------------------------------------- batch
    def make_corpus(self, n: int, rho: float = 0.85) -> dict:
        """Build a labelled corpus of n reports."""
        Xw = np.zeros((n, self.WORLD_DIM))
        Xs = np.zeros((n, self.ISNAD_DIM))
        y = np.zeros(n)
        viol = []
        for i in range(n):
            pol = self._sample_polity()
            truthful = bool(self.rng.random() < 0.5)
            rep, which = self._emit_report(pol, truthful)
            Xw[i] = self._encode_world(pol, rep)
            Xs[i] = self._encode_isnad(truthful, rho)
            y[i] = 1.0 if truthful else 0.0
            viol.append(which)
        return dict(Xw=Xw, Xs=Xs, y=y, inrange=np.ones(n), violation=viol)

    def make_out_of_range(self, n: int) -> dict:
        """
        Questions the scale is not calibrated for. Ibn Khaldun's examples are
        the divine attributes, the afterlife and the reality of prophecy: the
        instrument does not merely lose accuracy there, it produces confident
        readings that mean nothing. Numerically these are points far off the
        data manifold, in directions the training corpus never visits.
        """
        rng = self.rng
        Xw = rng.normal(0.0, 1.0, size=(n, self.WORLD_DIM)) * rng.uniform(
            2.5, 6.0, size=(n, 1))
        # deliberately break the sign/scale conventions the real features obey
        Xw[:, 1] *= -1.0                        # negative generations
        Xw[:, 3] += rng.uniform(3.0, 8.0, n)    # harshness beyond [0,1]
        Xs = rng.uniform(0.0, 1.0, size=(n, self.ISNAD_DIM))
        return dict(Xw=Xw, Xs=Xs, y=np.full(n, 0.5), inrange=np.zeros(n),
                    violation=["out_of_range"] * n)


# ==============================================================================
# SECTION 2 -- STANDARDISER
# The scale must be calibrated on gold before it can refuse mountains. We fit
# the normaliser on the training corpus only; out-of-range probes are then
# genuinely far away in the normalised space, which is the whole point.
# ==============================================================================

class Calibration:
    def __init__(self, Xw: np.ndarray, Xs: np.ndarray):
        self.mw = Xw.mean(0)
        self.sw = Xw.std(0) + 1e-6
        self.ms = Xs.mean(0)
        self.ss = Xs.std(0) + 1e-6

    def world(self, X):
        return (X - self.mw) / self.sw

    def isnad(self, X):
        return (X - self.ms) / self.ss


# ==============================================================================
# SECTION 3 -- THE UMRAN ENGINE
#
# Forward graph. Every arrow here is a doctrinal commitment, not a design
# convenience; the comments name the commitment.
#
#   Xw ---> [ MUTABAQA CORE ]  ---> H2 ---> p          (plausibility logit)
#     |            world features ONLY. The isnad never reaches this path.
#     |
#     +---> [ MALAKA BANK ]    ---> V_1..V_K
#     |     K habit modules, capacity-budgeted and mutually interfering
#     |
#     +---> [ ROUTER + ASABIYYA COUPLING ] ---> g      (coalition gate)
#           routing is to a COALITION, weighted by solidarity S_eff, and
#           S_eff is decayed by the DAWLA (dynastic) scheduler so that no
#           coalition can hold the throne indefinitely
#                                     |
#                            m = sum_k g_k V_k
#
#   Xs ---> [ ISNAD CHANNEL ] ---> a_raw ---> a_eff = gamma(p) * tanh(a_raw)
#           gamma(p) = gamma0 * sigmoid(kappa - |p|)
#           The band the chain is allowed to move the verdict SHRINKS as the
#           plausibility engine becomes confident. Authority breaks ties. It
#           cannot overturn an impossibility.
#
#   F = [H2 ; m]
#   accept_logit = F.wy + by + p + a_eff        <- the verdict
#   range_logit  = F.wr + br                    <- the MIZAN: is this claim
#                                                  inside the scale's range?
# ==============================================================================

class UmranEngine:

    def __init__(self,
                 d_w=12, d_s=6, d_h=24, K=6, d_m=10, d_o=16, d_s2=8,
                 beta=1.0, gamma0=1.2, kappa=1.0,
                 mu_cap=0.60, nu_int=0.05, budget=9.0,
                 lam_isti=2.0, tenure_T=1.4, decay_floor=0.15, ema=0.94,
                 naive=False, seed=0):
        """
        naive=True builds the CONTROL model: identical parameter count, but the
        isnad features are concatenated straight into the world path and the
        gamma(p) throttle is removed. That is the ordinary way to build this
        network, and it is the way that fails.
        """
        rng = np.random.default_rng(seed)
        self.dims = dict(d_w=d_w, d_s=d_s, d_h=d_h, K=K, d_m=d_m,
                         d_o=d_o, d_s2=d_s2)
        self.beta = beta
        self.gamma0 = gamma0
        self.kappa = kappa
        self.mu_cap = mu_cap
        self.nu_int = nu_int
        self.budget = budget
        self.lam_isti = lam_isti
        self.tenure_T = tenure_T
        self.decay_floor = decay_floor
        self.ema = ema
        self.naive = naive

        # in the naive control the muqaddima core also sees the chain
        d_in_core = d_w + (d_s if naive else 0)
        self.d_in_core = d_in_core

        def he(a, b):
            return rng.normal(0, math.sqrt(2.0 / a), size=(a, b))

        P = {}
        # -- Mutabaqa core (conformity to the nature of civilisation) ---------
        P["W1"] = he(d_in_core, d_h); P["b1"] = np.zeros(d_h)
        P["W2"] = he(d_h, d_h);       P["b2"] = np.zeros(d_h)
        P["wp"] = he(d_h, 1)[:, 0] * 0.5; P["bp"] = np.zeros(1)

        # -- Malaka bank: K habit modules ------------------------------------
        P["A"] = rng.normal(0, math.sqrt(2.0 / d_in_core), size=(K, d_in_core, d_m))
        P["ab"] = np.zeros((K, d_m))
        P["B"] = rng.normal(0, math.sqrt(2.0 / d_m), size=(K, d_m, d_o))
        P["bb"] = np.zeros((K, d_o))

        # -- Router + asabiyya coupling --------------------------------------
        P["R"] = he(d_in_core, K) * 0.3; P["r0"] = np.zeros(K)
        S0 = rng.normal(0, 0.25, size=(K, K)); S0 = 0.5 * (S0 + S0.T)
        np.fill_diagonal(S0, 1.0)
        P["S"] = S0

        # -- Isnad channel ----------------------------------------------------
        P["Ws"] = he(d_s, d_s2); P["bs"] = np.zeros(d_s2)
        P["wa"] = he(d_s2, 1)[:, 0] * 0.5; P["ba"] = np.zeros(1)

        # -- Mizan head (verdict + range) ------------------------------------
        P["wy"] = he(d_h + d_o, 1)[:, 0] * 0.5; P["by"] = np.zeros(1)
        P["wr"] = he(d_h + d_o, 1)[:, 0] * 0.5; P["br"] = np.zeros(1)

        self.P = P

        # -- Dawla (dynastic) state. Buffers, not parameters: they are the
        #    regime's history, and history is not differentiated through.
        self.tenure = np.zeros(K)     # accumulated share of the gate
        self.luxury = np.zeros(K)     # accumulated reward while in power
        self.reign_log = []           # which coalition held power, per step

    # ------------------------------------------------------------------ dawla
    def solidarity(self) -> np.ndarray:
        """
        S_eff = S * D, where D is the dynastic decay.

        A module that has held the gate (high tenure) and been well fed by it
        (high luxury) loses solidarity with its partners. The pair-decay is
        symmetric because asabiyya is a property of a bond, not of a member.
        The floor keeps a fallen dynasty alive as a claimant rather than
        annihilating it -- Ibn Khaldun's dynasties persist in memory and can
        rise again from the desert.
        """
        t = self.tenure
        l = self.luxury
        D = np.exp(-(t[:, None] + t[None, :]) / self.tenure_T) * \
            (1.0 - 0.5 * (l[:, None] + l[None, :]))
        D = np.clip(D, self.decay_floor, 1.0)
        return self.P["S"] * D, D

    def update_dawla(self, G: np.ndarray, correct: np.ndarray):
        """EMA update of tenure and luxury from one training step."""
        share = G.mean(0)
        fed = (G * correct[:, None]).mean(0)
        self.tenure = self.ema * self.tenure + (1 - self.ema) * share * self.dims["K"]
        self.luxury = self.ema * self.luxury + (1 - self.ema) * fed * self.dims["K"]
        self.luxury = np.clip(self.luxury, 0.0, 0.95)
        self.reign_log.append(int(np.argmax(share)))

    def istidad(self) -> np.ndarray:
        """
        Preparedness of each malaka to receive further colouring.

        "Once a habitus is firmly established in one's soul, it colours the
        soul... the colour it received weakens its preparedness (isti'dad) to
        receive another habitus."  So the learning rate of module k is scaled
        down in proportion to how much its subspace already overlaps the
        others'. This belongs in the optimiser, not the forward pass: it is a
        fact about ACQUISITION, not about use.
        """
        A = self.P["A"]                                  # (K, d_in, d_m)
        K = A.shape[0]
        nrm = np.sqrt((A ** 2).sum(axis=(1, 2))) + EPS
        ov = np.zeros(K)
        for k in range(K):
            for j in range(K):
                if j == k:
                    continue
                M = A[k].T @ A[j]
                ov[k] += (M ** 2).sum() / (nrm[k] ** 2 * nrm[j] ** 2 + EPS)
        return np.exp(-self.lam_isti * ov / max(K - 1, 1))

    # ---------------------------------------------------------------- forward
    def forward(self, Xw, Xs, y, inrange, w_pla=1.0, w_rng=1.0):
        """
        Returns (loss, cache). Losses:
            L_acc  verdict, masked to in-range reports
            L_pla  direct supervision of the plausibility channel
            L_rng  the mizan's own range verdict
            L_cap  capacity budget of the malaka bank
            L_int  mutual interference between malakat
        """
        P, d = self.P, self.dims
        K, d_o = d["K"], d["d_o"]
        N = Xw.shape[0]

        Xc = np.concatenate([Xw, Xs], axis=1) if self.naive else Xw

        # ---- Mutabaqa core ------------------------------------------------
        Z1 = Xc @ P["W1"] + P["b1"];  H1 = np.tanh(Z1)
        Z2 = H1 @ P["W2"] + P["b2"];  H2 = np.tanh(Z2)
        p = H2 @ P["wp"] + P["bp"][0]                        # (N,)

        # ---- Malaka bank ---------------------------------------------------
        Zu = np.einsum("nd,kdm->nkm", Xc, P["A"]) + P["ab"][None]   # (N,K,d_m)
        U = np.tanh(Zu)
        V = np.einsum("nkm,kmo->nko", U, P["B"]) + P["bb"][None]    # (N,K,d_o)

        # ---- Router + asabiyya coalition ----------------------------------
        Zr = Xc @ P["R"] + P["r0"]                            # (N,K)
        Q = softmax(Zr, axis=1)
        S_eff, Ddecay = self.solidarity()
        C = Q @ S_eff                                         # (N,K) coalition pull
        Zeta = Zr + self.beta * C
        G = softmax(Zeta, axis=1)
        M = np.einsum("nk,nko->no", G, V)                     # (N,d_o)

        # ---- Isnad channel --------------------------------------------------
        Zs = Xs @ P["Ws"] + P["bs"];  Hs = np.tanh(Zs)
        a_raw = Hs @ P["wa"] + P["ba"][0]                     # (N,)
        ta = np.tanh(a_raw)
        if self.naive:
            gam = np.full(N, self.gamma0)                     # no throttle
            a_eff = gam * ta
        else:
            sg = sigmoid(self.kappa - softabs(p))
            gam = self.gamma0 * sg
            a_eff = gam * ta

        # ---- Mizan head -----------------------------------------------------
        F = np.concatenate([H2, M], axis=1)                   # (N,d_h+d_o)
        y_log = F @ P["wy"] + P["by"][0] + p + a_eff
        r_log = F @ P["wr"] + P["br"][0]

        # ---- losses ---------------------------------------------------------
        mask = inrange
        nm = max(mask.sum(), 1.0)
        L_acc = (bce_with_logits(y_log, y) * mask).sum() / nm
        L_pla = w_pla * (bce_with_logits(p, y) * mask).sum() / nm
        L_rng = w_rng * bce_with_logits(r_log, inrange).mean()

        A = P["A"]
        nrm2 = (A ** 2).sum(axis=(1, 2))                      # (K,)
        occ = nrm2.sum() / self.budget
        over = max(occ - 1.0, 0.0)
        L_cap = self.mu_cap * over ** 2

        # pairwise subspace overlap (the "colouring" of the soul)
        pair_ov = 0.0
        Gram = {}
        for k in range(K):
            for j in range(k + 1, K):
                Mkj = A[k].T @ A[j]
                Gram[(k, j)] = Mkj
                pair_ov += (Mkj ** 2).sum() / (nrm2[k] * nrm2[j] + EPS)
        L_int = self.nu_int * pair_ov

        loss = L_acc + L_pla + L_rng + L_cap + L_int

        cache = dict(Xw=Xw, Xs=Xs, Xc=Xc, y=y, inrange=inrange, mask=mask, nm=nm,
                     H1=H1, H2=H2, p=p, U=U, V=V, Zr=Zr, Q=Q, C=C, Zeta=Zeta,
                     G=G, M=M, Hs=Hs, a_raw=a_raw, ta=ta, gam=gam, a_eff=a_eff,
                     F=F, y_log=y_log, r_log=r_log, S_eff=S_eff, Ddecay=Ddecay,
                     nrm2=nrm2, occ=occ, over=over, Gram=Gram, N=N,
                     w_pla=w_pla, w_rng=w_rng,
                     parts=dict(L_acc=L_acc, L_pla=L_pla, L_rng=L_rng,
                                L_cap=L_cap, L_int=L_int))
        return loss, cache

    # --------------------------------------------------------------- backward
    def backward(self, cache):
        """Hand-derived gradients for every parameter. No autodiff."""
        P, d = self.P, self.dims
        K = d["K"]; d_h = d["d_h"]
        Xc, Xs = cache["Xc"], cache["Xs"]
        H1, H2 = cache["H1"], cache["H2"]
        U, V, Q, G, M = cache["U"], cache["V"], cache["Q"], cache["G"], cache["M"]
        Hs, a_raw, ta, gam = cache["Hs"], cache["a_raw"], cache["ta"], cache["gam"]
        F, y_log, r_log = cache["F"], cache["y_log"], cache["r_log"]
        p, y, mask, nm = cache["p"], cache["y"], cache["mask"], cache["nm"]
        inrange = cache["inrange"]; N = cache["N"]
        S_eff, Ddecay = cache["S_eff"], cache["Ddecay"]

        gr = {k: np.zeros_like(v) for k, v in P.items()}

        # ---- L_rng ----------------------------------------------------------
        d_rlog = cache["w_rng"] * bce_with_logits_grad(r_log, inrange) / N
        gr["wr"] += F.T @ d_rlog
        gr["br"] += np.array([d_rlog.sum()])
        dF = np.outer(d_rlog, P["wr"])

        # ---- L_acc ----------------------------------------------------------
        d_ylog = bce_with_logits_grad(y_log, y) * mask / nm
        gr["wy"] += F.T @ d_ylog
        gr["by"] += np.array([d_ylog.sum()])
        dF += np.outer(d_ylog, P["wy"])

        dp = d_ylog.copy()                        # p enters the verdict directly
        d_aeff = d_ylog.copy()

        # ---- L_pla ----------------------------------------------------------
        dp += cache["w_pla"] * bce_with_logits_grad(p, y) * mask / nm

        # ---- isnad channel ---------------------------------------------------
        if self.naive:
            d_ta = d_aeff * gam
        else:
            d_ta = d_aeff * gam
            # gamma = gamma0 * sigmoid(kappa - |p|)  ->  d gamma / d p
            s = gam / self.gamma0
            dgam_dp = self.gamma0 * s * (1.0 - s) * (-softabs_grad(p))
            dp += d_aeff * ta * dgam_dp
        d_araw = d_ta * (1.0 - ta ** 2)
        gr["wa"] += Hs.T @ d_araw
        gr["ba"] += np.array([d_araw.sum()])
        dHs = np.outer(d_araw, P["wa"]) * (1.0 - Hs ** 2)
        gr["Ws"] += Xs.T @ dHs
        gr["bs"] += dHs.sum(0)

        # ---- split dF into H2 and M -----------------------------------------
        dH2 = dF[:, :d_h].copy()
        dM = dF[:, d_h:].copy()

        # p = H2 @ wp + bp
        gr["wp"] += H2.T @ dp
        gr["bp"] += np.array([dp.sum()])
        dH2 += np.outer(dp, P["wp"])

        # ---- malaka bank through the gate -----------------------------------
        dG = np.einsum("no,nko->nk", dM, V)                  # (N,K)
        dV = G[:, :, None] * dM[:, None, :]                  # (N,K,d_o)
        gr["B"] += np.einsum("nkm,nko->kmo", U, dV)
        gr["bb"] += dV.sum(0)
        dU = np.einsum("nko,kmo->nkm", dV, P["B"]) * (1.0 - U ** 2)
        gr["A"] += np.einsum("nd,nkm->kdm", Xc, dU)
        gr["ab"] += dU.sum(0)
        dXc = np.einsum("nkm,kdm->nd", dU, P["A"])

        # ---- gate: G = softmax(Zeta), Zeta = Zr + beta * (Q @ S_eff) ---------
        dZeta = softmax_backward(G, dG, axis=1)
        dZr = dZeta.copy()
        dC = self.beta * dZeta
        dQ = dC @ S_eff.T
        dS_eff = Q.T @ dC
        dZr += softmax_backward(Q, dQ, axis=1)
        # S_eff = S * Ddecay  (Ddecay is a buffer, no gradient through history)
        gr["S"] += dS_eff * Ddecay
        gr["R"] += Xc.T @ dZr
        gr["r0"] += dZr.sum(0)
        dXc += dZr @ P["R"].T

        # ---- mutabaqa core ---------------------------------------------------
        dZ2 = dH2 * (1.0 - H2 ** 2)
        gr["W2"] += H1.T @ dZ2
        gr["b2"] += dZ2.sum(0)
        dH1 = dZ2 @ P["W2"].T
        dZ1 = dH1 * (1.0 - H1 ** 2)
        gr["W1"] += Xc.T @ dZ1
        gr["b1"] += dZ1.sum(0)
        dXc += dZ1 @ P["W1"].T

        # (dXc is the gradient w.r.t. the input; the corpus is fixed, so it is
        #  computed for completeness/diagnostics but not applied.)

        # ---- capacity + interference penalties -------------------------------
        A = P["A"]; nrm2 = cache["nrm2"]
        if cache["over"] > 0:
            gr["A"] += self.mu_cap * 2.0 * cache["over"] * (2.0 * A) / self.budget
        for k in range(K):
            for j in range(k + 1, K):
                Mkj = cache["Gram"][(k, j)]                   # A_k^T A_j
                denom = nrm2[k] * nrm2[j] + EPS
                s = (Mkj ** 2).sum()
                # d/dA_k of ||A_k^T A_j||^2 / (|A_k|^2 |A_j|^2)
                gr["A"][k] += self.nu_int * (2.0 * (A[j] @ Mkj.T) / denom
                                             - s * 2.0 * A[k] * nrm2[j] / denom ** 2)
                gr["A"][j] += self.nu_int * (2.0 * (A[k] @ Mkj) / denom
                                             - s * 2.0 * A[j] * nrm2[k] / denom ** 2)

        cache["dXc"] = dXc
        return gr


# ==============================================================================
# SECTION 4 -- FINITE-DIFFERENCE GRADIENT CHECK
#
# Mandatory. Central differences against every parameter tensor. If this does
# not pass, nothing below it means anything.
# ==============================================================================

def gradient_check(model: UmranEngine, batch, n_probe=6, h=1e-5, tol=1e-5,
                   floor=1e-6, seed=0, verbose=True):
    """
    Central differences, relative error  |num - ana| / max(|num|, |ana|, floor).

    The `floor` matters. Central differences on a float64 loss carry a noise
    floor of roughly eps/h ~ 1e-11 in absolute terms; for a coordinate whose
    true gradient is itself around 1e-7, the RELATIVE error is then dominated
    by that noise and says nothing about the derivation. The floor keeps the
    check meaningful without letting real errors through: any coordinate with
    a gradient above `floor` is still checked relatively, to 1e-5.
    """
    rng = np.random.default_rng(seed)
    loss, cache = model.forward(batch["Xw"], batch["Xs"], batch["y"],
                                batch["inrange"])
    grads = model.backward(cache)

    worst = 0.0
    rows = []
    for name, W in model.P.items():
        flat = W.reshape(-1)
        gflat = grads[name].reshape(-1)
        idxs = rng.choice(flat.size, size=min(n_probe, flat.size), replace=False)
        errs = []
        for i in idxs:
            old = flat[i]
            flat[i] = old + h
            lp, _ = model.forward(batch["Xw"], batch["Xs"], batch["y"],
                                  batch["inrange"])
            flat[i] = old - h
            lm, _ = model.forward(batch["Xw"], batch["Xs"], batch["y"],
                                  batch["inrange"])
            flat[i] = old
            num = (lp - lm) / (2 * h)
            ana = gflat[i]
            denom = max(abs(num), abs(ana), floor)
            errs.append(abs(num - ana) / denom)
        e = max(errs)
        worst = max(worst, e)
        rows.append((name, W.shape, e))

    if verbose:
        print("  {:<6} {:<18} {:>12}".format("param", "shape", "rel.err"))
        for name, shp, e in rows:
            flag = "ok" if e < tol else "FAIL"
            print("  {:<6} {:<18} {:>12.3e}  {}".format(name, str(shp), e, flag))
        print("  worst relative error = {:.3e}   (tolerance {:.0e})".format(worst, tol))
    return worst < tol, worst


# ==============================================================================
# SECTION 5 -- OPTIMISER
#
# Adam, with one Khaldunian modification: the per-module learning rate of the
# malaka bank is multiplied by that module's isti'dad -- its remaining
# preparedness to be coloured. A module whose subspace already overlaps its
# neighbours' learns more slowly, exactly as the Muqaddimah says a soul
# already coloured by one habitus receives the next one less readily.
# ==============================================================================

class Adam:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads, istidad=None):
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            upd = self.lr * mh / (np.sqrt(vh) + self.eps)
            if istidad is not None and k in ("A", "ab", "B", "bb"):
                upd = upd * istidad.reshape((-1,) + (1,) * (upd.ndim - 1))
            params[k] -= upd


# ==============================================================================
# SECTION 6 -- EVALUATION
# ==============================================================================

def predict(model, Xw, Xs):
    _, c = model.forward(Xw, Xs, np.zeros(len(Xw)), np.ones(len(Xw)))
    return dict(accept=sigmoid(c["y_log"]),
                plaus=sigmoid(c["p"]),
                inrange=sigmoid(c["r_log"]),
                gate=c["G"], gamma=c["gam"])


def accuracy(model, corpus):
    o = predict(model, corpus["Xw"], corpus["Xs"])
    return float(((o["accept"] > 0.5) == (corpus["y"] > 0.5)).mean())


def accuracy_chain_blinded(model, corpus):
    """
    Judge the same corpus with the chain of transmission struck out -- the
    dossier replaced by its corpus mean, so it carries no information at all.
    A model that has genuinely learned the nature of civilisation barely
    notices. A model living off the chain falls apart.
    """
    Xs = np.repeat(corpus["Xs"].mean(0, keepdims=True), len(corpus["Xs"]), axis=0)
    o = predict(model, corpus["Xw"], Xs)
    return float(((o["accept"] > 0.5) == (corpus["y"] > 0.5)).mean())


def range_accuracy(model, inr, oor):
    o1 = predict(model, inr["Xw"], inr["Xs"])["inrange"]
    o2 = predict(model, oor["Xw"], oor["Xs"])["inrange"]
    return float(((o1 > 0.5).mean() + (o2 <= 0.5).mean()) / 2.0)


def by_violation(model, corpus):
    """Per-law breakdown: which of Ibn Khaldun's laws does the model police?"""
    o = predict(model, corpus["Xw"], corpus["Xs"])
    pred = o["accept"] > 0.5
    out = {}
    for v in set(corpus["violation"]):
        idx = np.array([i for i, w in enumerate(corpus["violation"]) if w == v])
        out[v] = float((pred[idx] == (corpus["y"][idx] > 0.5)).mean())
    return out


# ==============================================================================
# SECTION 7 -- TRAINING
# ==============================================================================

def train(model, train_corpus, oor_corpus, cal, epochs=40, bs=128, lr=3e-3,
          seed=0, verbose=True, tag=""):
    rng = np.random.default_rng(seed)
    opt = Adam(model.P, lr=lr)

    Xw = cal.world(train_corpus["Xw"]); Xs = cal.isnad(train_corpus["Xs"])
    y = train_corpus["y"]
    Ow = cal.world(oor_corpus["Xw"]);   Os = cal.isnad(oor_corpus["Xs"])
    n = len(y); n_o = len(Ow)

    hist = []
    for ep in range(epochs):
        perm = rng.permutation(n)
        tot = 0.0; nb = 0
        for s in range(0, n, bs):
            b = perm[s:s + bs]
            # every batch is padded with a few unanswerable questions, so the
            # scale is continually reminded what it cannot weigh
            k = max(1, len(b) // 8)
            oi = rng.integers(0, n_o, size=k)
            bXw = np.vstack([Xw[b], Ow[oi]])
            bXs = np.vstack([Xs[b], Os[oi]])
            by = np.concatenate([y[b], np.full(k, 0.5)])
            bir = np.concatenate([np.ones(len(b)), np.zeros(k)])

            loss, cache = model.forward(bXw, bXs, by, bir)
            grads = model.backward(cache)
            opt.step(model.P, grads, istidad=model.istidad() if not model.naive else None)

            correct = ((sigmoid(cache["y_log"]) > 0.5) == (by > 0.5)).astype(float)
            model.update_dawla(cache["G"], correct)
            tot += loss; nb += 1
        hist.append(tot / nb)
        if verbose and (ep % 14 == 0 or ep == epochs - 1):
            print("   {}epoch {:>3}   loss {:.4f}   throne={}   tenure={}"
                  .format(tag, ep, hist[-1], model.reign_log[-1],
                          np.array2string(model.tenure, precision=2)))
    return hist


def coalition_turnovers(model):
    """How many times did the ruling module change during training?"""
    r = model.reign_log
    return sum(1 for i in range(1, len(r)) if r[i] != r[i - 1]), len(set(r))


# ==============================================================================
# SECTION 8 -- SELF-TESTS
# ==============================================================================

def self_tests():
    ok = True

    def check(name, cond):
        nonlocal ok
        print("   [{}] {}".format("PASS" if cond else "FAIL", name))
        ok = ok and bool(cond)

    # -- generator obeys its own laws --------------------------------------
    gen = UmranGenerator(seed=11)
    pol = gen._sample_polity()
    check("compound ceiling holds in the generator",
          abs(pol["P"] - pol["P0"] * pol["r"] ** pol["g"]) < 1e-6 * pol["P"])
    check("support ceiling proportional to population",
          pol["A"] <= UmranGenerator.KAPPA * pol["P"] + 1e-9)
    # revenue is single-peaked in t
    ts = np.linspace(0.01, 0.99, 400)
    rev = ts * (1 - ts) ** UmranGenerator.BETA
    check("Khaldun-Laffer curve is single-peaked",
          0.30 < ts[int(np.argmax(rev))] < 0.45)
    # asabiyya decays with generations
    a1 = (0.3 + 0.7 * 0.5) * math.exp(-1 / UmranGenerator.TAU) * (1 - 0.7 * 0.2)
    a5 = (0.3 + 0.7 * 0.5) * math.exp(-5 / UmranGenerator.TAU) * (1 - 0.7 * 0.2)
    check("asabiyya decays across generations", a5 < a1 * 0.25)

    # -- corpora are balanced and shifted the way we claim -------------------
    tr = gen.make_corpus(1500, rho=0.95)
    mk = gen.make_corpus(1500, rho=0.05)
    corr_tr = np.corrcoef(tr["Xs"][:, 0], tr["y"])[0, 1]
    corr_mk = np.corrcoef(mk["Xs"][:, 0], mk["y"])[0, 1]
    check("ordinary corpus: reputable chain predicts truth", corr_tr > 0.5)
    check("munkar corpus: reputable chain predicts falsehood", corr_mk < -0.5)
    check("label balance near even", 0.4 < tr["y"].mean() < 0.6)

    # -- out-of-range probes really are off the manifold ---------------------
    cal = Calibration(tr["Xw"], tr["Xs"])
    oor = gen.make_out_of_range(400)
    d_in = np.abs(cal.world(tr["Xw"])).mean()
    d_out = np.abs(cal.world(oor["Xw"])).mean()
    check("unanswerable probes lie far off the calibrated manifold",
          d_out > 3 * d_in)

    # -- forward-pass invariants --------------------------------------------
    m = UmranEngine(seed=3)
    b = dict(Xw=cal.world(tr["Xw"][:64]), Xs=cal.isnad(tr["Xs"][:64]),
             y=tr["y"][:64], inrange=np.ones(64))
    _, c = m.forward(b["Xw"], b["Xs"], b["y"], b["inrange"])
    check("coalition gate is a distribution",
          np.allclose(c["G"].sum(1), 1.0))
    check("isnad influence is bounded by its band",
          np.all(np.abs(c["a_eff"]) <= c["gam"] + 1e-12))
    check("band shrinks as plausibility grows confident",
          np.corrcoef(np.abs(c["p"]), c["gam"])[0, 1] < -0.9)

    # -- the mutabaqa core is structurally blind to the chain ----------------
    Xs2 = np.random.default_rng(0).normal(size=b["Xs"].shape) * 3.0
    _, c2 = m.forward(b["Xw"], Xs2, b["y"], b["inrange"])
    check("plausibility engine cannot see the chain at all",
          np.allclose(c["p"], c2["p"]))
    check("coalition routing cannot see the chain at all",
          np.allclose(c["G"], c2["G"]))

    # -- dynastic decay actually decays ---------------------------------------
    m2 = UmranEngine(seed=4)
    _, D0 = m2.solidarity()
    m2.tenure = np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    m2.luxury = np.array([0.8, 0.0, 0.0, 0.0, 0.0, 0.0])
    _, D1 = m2.solidarity()
    check("a long-tenured, well-fed module loses solidarity",
          D1[0, 1] < D0[0, 1] * 0.5)
    check("solidarity decay is symmetric across a bond",
          np.allclose(D1, D1.T))
    check("a fallen dynasty is not annihilated (floor holds)",
          D1.min() >= m2.decay_floor - 1e-12)

    # -- isti'dad falls as malakat overlap ------------------------------------
    m3 = UmranEngine(seed=5)
    base = m3.istidad().mean()
    m3.P["A"][1] = m3.P["A"][0].copy()      # two identical habits
    dup = m3.istidad()
    check("preparedness drops where two habits occupy one subspace",
          dup[0] < base and dup[1] < base)

    return ok


# ==============================================================================
# SECTION 9 -- THE AL-MAS'UDI PROBE
#
# The chapter's founding case, run through the trained model. A report on the
# Israelite muster: a founding stock of seventy, four generations, and a claim
# of six hundred thousand fighting men -- carried by the most respectable
# chain the corpus can offer.
# ==============================================================================

def masudi_probe(model, cal, gen: UmranGenerator):
    P0, g, r = 70.0, 4.0, 2.2
    h, L, t, C = 0.75, 0.20, 0.15, 6
    P_true = P0 * r ** g
    a_true = (0.30 + 0.70 * h) * math.exp(-g / gen.TAU) * (1 - 0.7 * L)

    def row(P_rep, A_rep):
        R_rep = gen.REV_C * t * (1 - t) ** gen.BETA * P_rep
        return np.array([math.log(P0), g, r, h, L, t,
                         math.log(P_rep), math.log(A_rep), math.log(max(R_rep, 1e-3)),
                         g, a_true, float(C)])

    impeccable = np.array([0.98, 0.95, 0.92, 0.03, 0.5, 0.5])   # finest chain
    weak = np.array([0.10, 0.12, 0.08, 0.90, 0.5, 0.5])         # poor chain

    A_true = gen.KAPPA * P_true * (0.40 + 0.60 * a_true)
    cases = [
        ("600,000 fighting men, impeccable chain", row(600000 * 3.0, 600000.0), impeccable),
        ("600,000 fighting men, poor chain",       row(600000 * 3.0, 600000.0), weak),
        ("a muster the realm can feed, impeccable chain", row(P_true, A_true), impeccable),
        ("a muster the realm can feed, poor chain",       row(P_true, A_true), weak),
    ]
    Xw = cal.world(np.stack([c[1] for c in cases]))
    Xs = cal.isnad(np.stack([c[2] for c in cases]))
    o = predict(model, Xw, Xs)
    return [(cases[i][0], float(o["accept"][i]), float(o["plaus"][i]),
             float(o["gamma"][i])) for i in range(len(cases))]


# ==============================================================================
# SECTION 10 -- MAIN
# ==============================================================================

def main():
    t0 = time.time()
    np.set_printoptions(precision=3, suppress=True)
    line = "=" * 78

    print(line)
    print(" THE UMRAN ENGINE  --  chapter 0297, Ibn Khaldun (1332-1406)")
    print(" report adjudication by conformity to the nature of civilisation")
    print(line)

    print("\n[1] SELF-TESTS")
    tests_ok = self_tests()
    if not tests_ok:
        raise SystemExit("self-tests failed")

    print("\n[2] BUILDING THE CORPORA")
    gen = UmranGenerator(seed=7)
    train_c = gen.make_corpus(900, rho=0.95)
    iid_c = gen.make_corpus(2500, rho=0.95)
    munkar = gen.make_corpus(2500, rho=0.05)
    oor_tr = gen.make_out_of_range(1200)
    oor_te = gen.make_out_of_range(800)
    cal = Calibration(train_c["Xw"], train_c["Xs"])
    print("   train {}   iid-test {}   munkar-test {}   unanswerable {}/{}"
          .format(len(train_c["y"]), len(iid_c["y"]), len(munkar["y"]),
                  len(oor_tr["y"]), len(oor_te["y"])))
    print("   chain/truth correlation:  ordinary {:+.3f}   munkar {:+.3f}"
          .format(np.corrcoef(train_c["Xs"][:, 0], train_c["y"])[0, 1],
                  np.corrcoef(munkar["Xs"][:, 0], munkar["y"])[0, 1]))

    Ctr = dict(Xw=cal.world(train_c["Xw"]), Xs=cal.isnad(train_c["Xs"]),
               y=train_c["y"], inrange=train_c["inrange"],
               violation=train_c["violation"])
    Ciid = dict(Xw=cal.world(iid_c["Xw"]), Xs=cal.isnad(iid_c["Xs"]),
                y=iid_c["y"], inrange=iid_c["inrange"],
                violation=iid_c["violation"])
    Cmun = dict(Xw=cal.world(munkar["Xw"]), Xs=cal.isnad(munkar["Xs"]),
                y=munkar["y"], inrange=munkar["inrange"],
                violation=munkar["violation"])
    Coor = dict(Xw=cal.world(oor_te["Xw"]), Xs=cal.isnad(oor_te["Xs"]),
                y=oor_te["y"], inrange=oor_te["inrange"])

    print("\n[3] GRADIENT CHECK  (central differences, every parameter tensor)")
    gm = UmranEngine(seed=1)
    gbatch = dict(Xw=Ctr["Xw"][:48], Xs=Ctr["Xs"][:48],
                  y=Ctr["y"][:48], inrange=np.ones(48))
    gm.tenure = np.array([0.7, 0.2, 1.1, 0.0, 0.4, 0.9])   # a live dynastic state
    gm.luxury = np.array([0.3, 0.1, 0.5, 0.0, 0.2, 0.4])
    passed, worst = gradient_check(gm, gbatch)
    if not passed:
        raise SystemExit("gradient check failed")

    print("\n[4] TRAINING THE UMRAN ENGINE")
    model = UmranEngine(seed=2)
    train(model, train_c, oor_tr, cal, epochs=70, lr=3e-3, seed=2, tag="")

    print("\n[5] TRAINING THE NAIVE CONTROL")
    print("    same capacity; chain concatenated into the world path;")
    print("    no band-throttle; no isti'dad on acquisition.")
    ctrl = UmranEngine(seed=2, naive=True)
    train(ctrl, train_c, oor_tr, cal, epochs=70, lr=3e-3, seed=2, tag="ctrl ")

    print("\n[6] RESULTS")
    hdr = "   {:<26}{:>10}{:>10}{:>14}{:>10}"
    print(hdr.format("model", "iid", "munkar", "chain-blinded", "range"))
    for nm, mdl in (("umran engine", model), ("naive control", ctrl)):
        print(hdr.format(
            nm,
            "{:.3f}".format(accuracy(mdl, Ciid)),
            "{:.3f}".format(accuracy(mdl, Cmun)),
            "{:.3f}".format(accuracy_chain_blinded(mdl, Ciid)),
            "{:.3f}".format(range_accuracy(mdl, Ciid, Coor))))

    print("\n   per-law policing (umran engine, munkar corpus)")
    for k, v in sorted(by_violation(model, Cmun).items()):
        print("     {:<22} {:.3f}".format(k, v))
    print("   per-law policing (naive control, munkar corpus)")
    for k, v in sorted(by_violation(ctrl, Cmun).items()):
        print("     {:<22} {:.3f}".format(k, v))

    print("\n[7] THE DYNASTIC CYCLE INSIDE THE MODEL")
    to_e, uniq_e = coalition_turnovers(model)
    to_c, uniq_c = coalition_turnovers(ctrl)
    print("   umran engine : {:>4} turnovers of the throne, {} distinct rulers"
          .format(to_e, uniq_e))
    print("   naive control: {:>4} turnovers of the throne, {} distinct rulers"
          .format(to_c, uniq_c))
    print("   final tenure  {}".format(model.tenure))
    print("   final luxury  {}".format(model.luxury))
    _, D = model.solidarity()
    print("   mean surviving solidarity across bonds: {:.3f}".format(
        D[~np.eye(D.shape[0], dtype=bool)].mean()))
    print("   isti'dad (remaining preparedness per malaka): {}"
          .format(model.istidad()))
    A = model.P["A"]; Kk = A.shape[0]
    n2 = (A ** 2).sum(axis=(1, 2))
    ovs = [float(((A[i].T @ A[j]) ** 2).sum() / (n2[i] * n2[j]))
           for i in range(Kk) for j in range(i + 1, Kk)]
    print("   mean pairwise subspace overlap of the malakat: {:.4f}".format(np.mean(ovs)))
    occ = (model.P["A"] ** 2).sum() / model.budget
    print("   malaka bank occupancy: {:.3f} of the budget".format(occ))

    print("\n[8] THE AL-MAS'UDI PROBE")
    print("   {:<44}{:>9}{:>10}{:>9}".format("report", "accept", "possible", "band"))
    for nm, acc, pl, gm_ in masudi_probe(model, cal, gen):
        print("   {:<44}{:>9.3f}{:>10.3f}{:>9.3f}".format(nm, acc, pl, gm_))
    print("\n   the same four reports, judged by the naive control")
    for nm, acc, pl, gm_ in masudi_probe(ctrl, cal, gen):
        print("   {:<44}{:>9.3f}{:>10.3f}{:>9.3f}".format(nm, acc, pl, gm_))

    print("\n[9] THE MIZAN REFUSING TO WEIGH MOUNTAINS")
    o_in = predict(model, Ciid["Xw"][:800], Ciid["Xs"][:800])["inrange"]
    o_out = predict(model, Coor["Xw"], Coor["Xs"])["inrange"]
    print("   mean in-range verdict, answerable questions   : {:.3f}".format(o_in.mean()))
    print("   mean in-range verdict, unanswerable questions : {:.3f}".format(o_out.mean()))
    print("   abstention rate on the unanswerable           : {:.3f}"
          .format(float((o_out <= 0.5).mean())))

    print("\n" + line)
    print(" completed in {:.1f}s   |   gradient check worst rel.err {:.2e}"
          .format(time.time() - t0, worst))
    print(line)


if __name__ == "__main__":
    main()

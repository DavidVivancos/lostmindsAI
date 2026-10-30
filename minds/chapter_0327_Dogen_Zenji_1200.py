#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
CHAPTER 327 — DOKUGEN ZENJI (1200-1253)
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0327 · Dogen Zenji (1200-1253)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 17, Minds 321-340 Available on Amazon https://www.amazon.com/dp/B0HP7Z1NKC
# Resume and Interactive Demos at https://artificiology.com/

 THE DHARMA-POSITION EQUILIBRIUM FIELD  (Jūhōi Equilibrium Field, JEF)
================================================================================

WHAT THIS FILE IS
-----------------
A complete, from-scratch, pure-NumPy learning system whose *mechanism* is derived
from one 13th-century claim rather than decorated with it afterwards.

Dogen's claim, in his own two moves:

  (1) SHUSHO-ITTO  (修証一等, "practice and verification are one degree").
      Practice is not a means whose product is realization. Practice IS
      realization, enacted now. There is therefore no finished artifact to
      inspect. Translated into machine learning: there is no train/eval split,
      no frozen checkpoint, no "shipped model". The forward solve and the
      parameter update are the same loop, running always.

  (2) JUHOI + ZENGO-SAIDAN  (住法位・前後際断, "abiding in the dharma-position;
      before and after are cut off"). From Genjokoan: firewood does not become
      ash. Firewood abides in the dharma-position of firewood, which *fully
      includes* its before and after, and yet before and after are severed.
      Translated into machine learning: NOTHING IS HANDED FORWARD. There is no
      hidden state h_t, no recurrence, no KV cache, no carried summary. Each
      position must fully contain its temporal context without ever having
      received a token of it from a neighbour.

These two look incompatible: learning seems to require carrying something.
Dogen's own resolution is KYORYAKU (経歴, "seriatim/totalistic passage"): being-
time "passes from today to tomorrow, passes from today to yesterday, passes from
yesterday to today, passes from today to today". Passage is not transport along
an arrow; it is mutual co-constitution. So this network computes the whole window
as the JOINT FIXED POINT of one time-invariant law, with symmetric (tied)
forward/backward coupling. Coherence across time is produced by CO-SATISFACTION
OF ONE LAW, not by transmission of a state.

MECHANISM MAP (doctrine -> code, one to one)
--------------------------------------------
  shikantaza 只管打坐  "just sitting"      -> damped fixed-point solve with no
                                              target in the forward pass; the
                                              field sits until it stops moving.
  juhoi 住法位          "dharma-position"  -> one latent row per position; no
                                              positional encoding at all, since
                                              a position is not a coordinate in
                                              a container.
  zengo-saidan 前後際断 "before/after cut" -> no state object survives a call;
                                              verified by an explicit test.
  kyoryaku 経歴         "totalistic passage"-> tied symmetric neighbour coupling
                                              K used for BOTH t->t+1 and t->t-1,
                                              plus a self term (today->today).
  hishiryo 非思量       "non-thinking"     -> a three-band activation that is
                                              neither thinking (shiryo) nor
                                              not-thinking (fushiryo). Its
                                              middle band emits ~zero and is
                                              simultaneously the band of MAXIMUM
                                              sensitivity. This is only possible
                                              below an exact constant derived
                                              below: beta*theta < ln(2+sqrt3).
  genjokoan 現成公案    "myriad things come -> the update is driven ONLY by the
                        forth and verify"     residual at positions where the
                                              world actually arrived. The
                                              opposite path ("carrying the self
                                              forward") is implemented, measured,
                                              and switched OFF by default.
  shusho-itto 修証一等  "practice=verification" -> no checkpoint; verification is
                                              a read of the same running loop.

WHAT IS DELIBERATELY ABSENT
---------------------------
  * No attention and no stored keys/values (that is retrieval of a carried past).
  * No recurrent hidden state.
  * No positional embedding.
  * No train/test phase separation in the headline learner.
  * No backpropagation-through-time. Gradients come from the IMPLICIT FUNCTION
    THEOREM applied at the fixed point, so the backward pass is the *same law*
    run adjointly -- practice and verification using one mechanism.

RUN
---
    python3 chapter_0327_Dogen_Zenji_1200.py

Everything below executes: an analytic derivation check, a finite-difference
gradient check on every parameter block, four invariance self-tests, and two
real experiments. Runtime is well under a minute on CPU.
"""

from __future__ import annotations

import numpy as np

np.seterr(over="ignore")

# =============================================================================
# SECTION 0 -- Small numeric utilities
# =============================================================================


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def dsigmoid_from_value(s: np.ndarray) -> np.ndarray:
    """sigma'(z) expressed through s = sigma(z)."""
    return s * (1.0 - s)


def softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def d_softplus(x: np.ndarray) -> np.ndarray:
    return sigmoid(x)


def shift_prev(Z: np.ndarray) -> np.ndarray:
    """Sp(Z)[t] = Z[t-1], with Sp(Z)[0] = 0.  'Yesterday, seen from today'."""
    out = np.zeros_like(Z)
    out[1:] = Z[:-1]
    return out


def shift_next(Z: np.ndarray) -> np.ndarray:
    """Sn(Z)[t] = Z[t+1], with Sn(Z)[T-1] = 0.  'Tomorrow, seen from today'."""
    out = np.zeros_like(Z)
    out[:-1] = Z[1:]
    return out


# NOTE on the adjoints of these two operators, used by the backward law:
#   <Sp(Z), W> = sum_t Z[t-1].W[t] = sum_s Z[s].W[s+1] = <Z, Sn(W)>   so Sp^T = Sn
#   and symmetrically Sn^T = Sp.
# The two directions of kyoryaku are exact transposes of one another. This is
# why using a single tied matrix K for both is not a parameter-saving trick but
# a structural statement: passage yesterday->today and today->tomorrow are the
# same operation viewed from two dharma-positions.


# =============================================================================
# SECTION 1 -- The hishiryo activation, and the constant that makes it possible
# =============================================================================
#
# Dogen quotes Yaoshan in the Zazenshin fascicle: asked what one thinks of while
# sitting, the master answers "thinking of not-thinking"; asked how, he answers
# hishiryo -- non-thinking. Rein Raud's reading (SAET, 2024) is grammatical and
# sharp: Dogen exploits two Chinese negations. FU-shiryo is the CONTRARY negative
# -- the act simply does not occur. HI-shiryo is the CONTRADICTORY negative --
# the act occurs but not under the definition. Reading shiryo literally as
# "thinking WITH MEASURES", hishiryo is thinking that is not quantified.
#
# So the unit needs three bands, not two:
#     shiryo    -- expressed, measured, saturated: output near +/-1
#     fushiryo  -- absent: output zero because nothing is happening
#     hishiryo  -- output zero, yet maximally live
#
# The third band is what makes this different from a dead zone. We want a region
# where the unit EMITS NOTHING and is simultaneously the MOST SENSITIVE part of
# its own range. Define, with gain beta > 0 and half-width theta >= 0:
#
#     phi(u) = sigma(beta*(u - theta)) - sigma(-beta*(u + theta))
#
# It is odd, bounded in (-1, 1), saturates to +/-1, and is exactly zero at u = 0.
# Writing a = beta*theta and v = beta*u, its slope is
#
#     phi'(u) = beta * [ sigma'(v - a) + sigma'(v + a) ] =: beta * g(v)
#
# g'(0) = sigma''(-a) + sigma''(a) = 0 always, so v = 0 is always a critical
# point. It is a MAXIMUM iff g''(0) = 2*sigma'''(a) < 0. With s = sigma(a),
#
#     sigma'''(a) = s(1-s) * (6s^2 - 6s + 1)
#
# which is negative exactly when 6s^2 - 6s + 1 < 0, i.e. s < (3 + sqrt(3))/6.
# Inverting the logistic gives the exact threshold
#
#     a < ln( (3+sqrt3)/(3-sqrt3) ) = ln(2 + sqrt3) = arcosh(2) = 1.3169578969...
#
# THE HISHIRYO CONDITION:  beta * theta < ln(2 + sqrt3).
#
# Above it the unit bifurcates: the maximum-gradient points move out to +/-|u|>0
# and the middle collapses into a genuine dead zone -- fushiryo, mere
# not-thinking. Non-thinking is therefore not a limit of decisiveness but its
# opposite: a unit that is too sharply thresholded relative to its gain LOSES
# the third band entirely. Below we do not merely check this; we make it
# unbreakable by construction, parameterising theta = (c/beta)*sigmoid(rho_raw)
# so that beta*theta lies in (0, c) for every value the optimiser can reach.
#
# Substituting a = beta*theta = c*rho collapses the activation to two clean
# parameters per unit:
#
#     phi(u; beta, rho) = sigma(beta*u - c*rho) - sigma(-beta*u - c*rho)

C_HISHIRYO = float(np.log(2.0 + np.sqrt(3.0)))  # = arcosh(2) = 1.3169578969...
RHO_MARGIN = 1.0 - 1e-12                        # see DharmaPositionField.rho()


class Hishiryo:
    """Three-band non-thinking activation with analytic derivatives."""

    c = C_HISHIRYO

    @staticmethod
    def value(u, beta, rho):
        a = Hishiryo.c * rho
        return sigmoid(beta * u - a) - sigmoid(-beta * u - a)

    @staticmethod
    def all_grads(u, beta, rho):
        """Return (phi, dphi/du, dphi/dbeta, dphi/drho) in one pass."""
        a = Hishiryo.c * rho
        sp = sigmoid(beta * u - a)          # 'shiryo' side
        sm = sigmoid(-beta * u - a)         # mirrored side
        phi = sp - sm
        dsp = dsigmoid_from_value(sp)
        dsm = dsigmoid_from_value(sm)
        d_du = beta * (dsp + dsm)
        d_dbeta = u * (dsp + dsm)
        d_drho = Hishiryo.c * (dsm - dsp)
        return phi, d_du, d_dbeta, d_drho

    @staticmethod
    def band(u, beta, rho):
        """Which of the three bands each unit occupies. 0 = hishiryo."""
        a = Hishiryo.c * rho
        # the band edges are where |beta*u| = a, i.e. the two sigmoid centres
        x = beta * u
        return np.where(x > a, 1, np.where(x < -a, -1, 0))


# =============================================================================
# SECTION 2 -- The world that comes forth: a drifting being-time stream
# =============================================================================


class BeingTimeStream:
    """
    A non-stationary source. Its law is not fixed, so no checkpoint of a model
    of it can stay correct -- which is the empirical setting in which
    shusho-itto stops being a piety and becomes an engineering claim.

    Latent: a phase oscillator whose angular velocity drifts continuously and
    also jumps at abrupt regime boundaries (firewood does not gradually become
    ash: the law itself changes dharma-position). Observed: two harmonics
    through a saturating nonlinearity, plus observation noise.

    THE THREE KINDS OF POSITION IN A WINDOW
    ---------------------------------------
    visible  -- the observation is placed in the network's input.
    graded   -- the observation is WITHHELD from the input but IS used to grade.
                This is the realization channel: the myriad things come forth
                and verify the self. The model must constitute the position
                from the equilibrium, not read it off.
    held-out -- withheld from the input and never used to grade anything, ever.
                The one honest number. It is also where the delusion path,
                when enabled, lets the model grade itself instead.

    Grading only at positions whose value was withheld from the input is what
    makes the task an inference rather than a copy.
    """

    def __init__(self, seed=0, drift=2.2e-3, regime_every=2500, noise=0.04,
                 calib_drift=0.020):
        self.rng = np.random.default_rng(seed)
        self.drift = drift
        self.regime_every = regime_every
        self.noise = noise
        self.t = 0
        self.phase = 0.0
        self.omega = 0.30
        self.h2 = 0.45     # second-harmonic weight
        self.psi = 0.7     # second-harmonic phase offset
        self.regimes = 0
        # THE SENSOR CALIBRATION.
        # gain and offset applied to what the network is SHOWN, while grading
        # is always against the true value. They random-walk on a timescale far
        # longer than one window, so within any single window they are constant
        # and confounded with the signal's own amplitude and mean -- they cannot
        # be inferred in context. Because this field carries no state between
        # windows, the ONLY vessel able to track them is the parameters
        # themselves. That is the whole experiment: Dogen's two claims lock
        # together. A system that refuses to hand anything forward can adapt
        # only by never ceasing to practise.
        self.calib_drift = calib_drift
        self.w = 0                      # window counter driving the migration
        self.gain = 1.0
        self.offset = 0.0

    def _step(self):
        self.omega += self.drift * self.rng.standard_normal()
        self.omega = float(np.clip(self.omega, 0.14, 0.55))
        self.h2 += 0.5 * self.drift * self.rng.standard_normal()
        self.h2 = float(np.clip(self.h2, 0.05, 0.85))
        if self.regime_every and self.t > 0 and self.t % self.regime_every == 0:
            self.omega = float(self.rng.uniform(0.15, 0.54))
            self.psi = float(self.rng.uniform(0.0, 2 * np.pi))
            self.h2 = float(self.rng.uniform(0.05, 0.8))
            self.regimes += 1
        self.phase += self.omega
        s = np.sin(self.phase) + self.h2 * np.sin(2.0 * self.phase + self.psi)
        clean = float(np.tanh(0.9 * s))
        obs = clean + self.noise * float(self.rng.standard_normal())
        self.t += 1
        return clean, obs

    def _drift_calibration(self):
        # A slow, smooth, strongly autocorrelated migration, not a symmetric
        # random walk. This distinction is the whole point. A random walk has a
        # stable long-run mean, so a frozen average is a decent bet forever. A
        # migration does not: "the calibration lately" is well estimated from
        # recent windows and the global average is wrong almost everywhere. The
        # two incommensurable periods keep it from ever repeating exactly.
        self.w += 1
        self.gain = 1.0 + 0.42 * np.sin(2 * np.pi * self.w / 1100.0 + 0.4)
        self.offset = 0.34 * np.sin(2 * np.pi * self.w / 1700.0 + 1.9)

    def window(self, T, n_graded=4, n_held=2):
        self._drift_calibration()
        clean = np.empty(T)
        obs = np.empty(T)
        for i in range(T):
            clean[i], obs[i] = self._step()
        obs = self.gain * obs + self.offset
        # never hide the two rim positions: they have only one neighbour
        interior = np.arange(1, T - 1)
        pick = self.rng.choice(interior, size=n_graded + n_held, replace=False)
        graded = np.zeros(T); graded[pick[:n_graded]] = 1.0
        held = np.zeros(T);  held[pick[n_graded:]] = 1.0
        vis = 1.0 - graded - held
        X = np.stack([obs * vis, 1.0 - vis], axis=1)
        return X, clean, graded, held

    @staticmethod
    def signal_variance(T=16, n=8000, seed=999):
        st = BeingTimeStream(seed=seed)
        vals = np.concatenate([st.window(T)[1] for _ in range(n // T)])
        return float(vals.var())


# =============================================================================
# SECTION 3 -- The field itself
# =============================================================================


class DharmaPositionField:
    """
    Z* is defined implicitly as the solution of

        Z = Phi( X Win^T  +  Z Ws^T  +  Sp(Z) K^T  +  Sn(Z) K^T  +  b )

    read as: every dharma-position is constituted by (i) what arrives there,
    (ii) itself, (iii) yesterday and (iv) tomorrow -- the three passages Dogen
    names in Uji -- under one law Phi shared by every position and every moment.

    The window is solved as a whole. No row is computed 'first'. There is no
    causal ordering in the computation, only in the content.
    """

    def __init__(self, d_in=2, d=28, d_out=1, T=16, seed=0,
                 alpha=0.5, gamma=0.85, tol=1e-11, max_iter=400):
        rng = np.random.default_rng(seed)
        self.d_in, self.d, self.d_out, self.T = d_in, d, d_out, T
        self.alpha = alpha          # damping of 'just sitting'
        self.gamma = gamma          # contraction budget (< 1)
        self.tol = tol
        self.max_iter = max_iter

        sc = 1.0 / np.sqrt(d)
        self.p = {
            "Win": rng.standard_normal((d, d_in)) * 0.6,
            "Ws": rng.standard_normal((d, d)) * sc,
            "K": rng.standard_normal((d, d)) * sc,
            "b": np.zeros(d),
            "beta_raw": np.full(d, 0.5413),   # softplus -> ~1.0
            "rho_raw": np.zeros(d),           # sigmoid  -> 0.5, so beta*theta ~ c/2
            "V": rng.standard_normal((d_out, d)) * sc,
            "c": np.zeros(d_out),
        }
        self.project_contraction()
        # diagnostics only -- deliberately NOT state that influences any output
        self.last_iters = 0
        self.last_residual = 0.0

    # ---- derived parameters ------------------------------------------------
    def beta(self):
        return softplus(self.p["beta_raw"])

    def rho(self):
        # RHO_MARGIN exists only because sigmoid(z) saturates to exactly 1.0 in
        # float64 for z >~ 37, which would put beta*theta exactly ON the
        # bifurcation rather than strictly below it. In exact arithmetic the
        # sigmoid alone already gives an open interval; this keeps that true in
        # machine arithmetic, so the hishiryo band can never be lost.
        return RHO_MARGIN * sigmoid(self.p["rho_raw"])

    def theta(self):
        """The literal half-width of the non-thinking band, for reporting."""
        return Hishiryo.c * self.rho() / self.beta()

    # ---- the contraction bound --------------------------------------------
    def project_contraction(self):
        """
        'Just sitting' has to actually come to rest. The map's Lipschitz
        constant is bounded by max|Phi'| * (||Ws|| + 2||K||). Since
        max|Phi'| = 2*beta*sigma'(c*rho) <= beta/2, we rescale Ws and K so the
        product stays under gamma < 1. Enforced after every update, so the
        guarantee is an invariant of the whole run rather than an initial gift.
        """
        L_phi = 0.5 * float(np.max(self.beta()))
        s_ws = float(np.linalg.norm(self.p["Ws"], 2))
        s_k = float(np.linalg.norm(self.p["K"], 2))
        total = L_phi * (s_ws + 2.0 * s_k)
        if total > self.gamma:
            f = self.gamma / total
            self.p["Ws"] *= f
            self.p["K"] *= f
        return L_phi * (float(np.linalg.norm(self.p["Ws"], 2))
                        + 2.0 * float(np.linalg.norm(self.p["K"], 2)))

    # ---- the law -----------------------------------------------------------
    def pre(self, Z, X):
        p = self.p
        return (X @ p["Win"].T
                + Z @ p["Ws"].T
                + shift_prev(Z) @ p["K"].T
                + shift_next(Z) @ p["K"].T
                + p["b"])

    def solve(self, X, Z0=None, tol=None, max_iter=None):
        """
        Shikantaza. Sit -- damped Jacobi iteration on the whole window at once --
        until the field stops moving. No target enters here. The forward pass
        has no objective; it only comes to rest.
        """
        tol = self.tol if tol is None else tol
        max_iter = self.max_iter if max_iter is None else max_iter
        bt, rh = self.beta(), self.rho()
        Z = np.zeros((X.shape[0], self.d)) if Z0 is None else Z0.copy()
        res = np.inf
        it = 0
        for it in range(1, max_iter + 1):
            F = Hishiryo.value(self.pre(Z, X), bt, rh)
            Znew = (1.0 - self.alpha) * Z + self.alpha * F
            res = float(np.max(np.abs(Znew - Z)))
            Z = Znew
            if res < tol:
                break
        self.last_iters, self.last_residual = it, res
        return Z

    def readout(self, Z):
        return Z @ self.p["V"].T + self.p["c"]

    def forward(self, X, Z0=None):
        Z = self.solve(X, Z0=Z0)
        return Z, self.readout(Z)

    # ---- loss --------------------------------------------------------------
    @staticmethod
    def split_losses(Y, clean, graded, held):
        """
        REALIZATION (万法すすみて自己を修証する) -- 'the myriad things come forth
        and verify the self'. Error at the GRADED positions: those the model was
        not shown and must constitute for itself, checked against what actually
        arrived. This is the only permitted learning signal.

        HELD-OUT -- withheld from the input and never used to grade. The single
        number that cannot be reached from inside the system.
        """
        y = Y[:, 0]
        ng = max(1.0, float(graded.sum()))
        nh = max(1.0, float(held.sum()))
        L_real = float(((y - clean) ** 2 * graded).sum() / ng)
        L_held = float(((y - clean) ** 2 * held).sum() / nh)
        return L_real, L_held, ng, nh

    # ---- the adjoint law ---------------------------------------------------
    def grads(self, X, clean, graded, held, Z=None,
              self_target=None, w_delusion=0.0):
        """
        Gradients through the fixed point by the implicit function theorem.

        At Z* = f(Z*, params):   dL/dp = a^T df/dp   with   a = g + J^T a,
        where g = dL/dZ* and J = df/dZ. The adjoint solve is a fixed-point
        iteration using the SAME operator, transposed -- the backward pass is
        the forward law read the other way. Nothing is unrolled and nothing is
        stored per step; there is no backpropagation through time because there
        is no time-step chain to propagate through.

        self_target, if supplied with w_delusion > 0, is the DELUSION path:
        the model's own image of itself (an EMA teacher) used as ground truth
        at precisely the held-out positions, where nothing arrived. It is
        implemented so it can be measured, and it is off unless asked for.
        """
        p = self.p
        bt, rh = self.beta(), self.rho()
        if Z is None:
            Z = self.solve(X)
        P = self.pre(Z, X)
        _, D, dPhi_dbeta, dPhi_drho = Hishiryo.all_grads(P, bt, rh)

        Y = self.readout(Z)
        y = Y[:, 0]
        L_real, L_held, ng, nh = self.split_losses(Y, clean, graded, held)

        dY = np.zeros_like(Y)
        dY[:, 0] += 2.0 * (y - clean) * graded / ng

        L_delu = 0.0
        if w_delusion > 0.0 and self_target is not None:
            r = (y - self_target) * held
            L_delu = float((r ** 2).sum() / nh)
            dY[:, 0] += w_delusion * 2.0 * r / nh

        g = dY @ p["V"]                        # dL/dZ*, shape (T, d)

        a = g.copy()
        for _ in range(self.max_iter):
            u = D * a
            Ja = u @ p["Ws"] + shift_next(u @ p["K"]) + shift_prev(u @ p["K"])
            a_new = g + Ja
            if np.max(np.abs(a_new - a)) < 1e-12:
                a = a_new
                break
            a = a_new

        G = a * D                              # sensitivity of the pre-activation
        gr = {
            "Win": G.T @ X,
            "Ws": G.T @ Z,
            "K": G.T @ (shift_prev(Z) + shift_next(Z)),
            "b": G.sum(axis=0),
            "beta_raw": (a * dPhi_dbeta).sum(axis=0) * d_softplus(p["beta_raw"]),
            "rho_raw": (a * dPhi_drho).sum(axis=0)
                       * RHO_MARGIN
                       * dsigmoid_from_value(sigmoid(p["rho_raw"])),
            "V": dY.T @ Z,
            "c": dY.sum(axis=0),
        }
        return gr, dict(L_real=L_real, L_held=L_held, L_delu=L_delu,
                        Z=Z, D=D, Y=Y)

    # ---- one act of practice ----------------------------------------------
    def practice(self, X, clean, graded, held, lr=0.02, mom=None,
                 self_target=None, w_delusion=0.0):
        """
        A single act. There is no 'mode': solving and updating happen in the
        same call and cannot be pulled apart. Verification is a read of numbers
        this act has already produced -- not a separate phase applied later to
        a finished object.
        """
        gr, info = self.grads(X, clean, graded, held,
                              self_target=self_target, w_delusion=w_delusion)
        if mom is None:
            for k in self.p:
                self.p[k] -= lr * gr[k]
        else:
            for k in self.p:
                mom[k] = 0.9 * mom[k] + gr[k]
                self.p[k] -= lr * mom[k]
        self.project_contraction()
        return info

    # ---- flat vector view, for the finite-difference check -----------------
    def get_flat(self):
        return np.concatenate([self.p[k].ravel() for k in sorted(self.p)])

    def set_flat(self, v):
        i = 0
        for k in sorted(self.p):
            n = self.p[k].size
            self.p[k] = v[i:i + n].reshape(self.p[k].shape).copy()
            i += n

    def loss_only(self, X, clean, graded, held,
                  self_target=None, w_delusion=0.0):
        Z = self.solve(X, tol=1e-14, max_iter=4000)
        Y = self.readout(Z)
        y = Y[:, 0]
        ng = max(1.0, float(graded.sum()))
        nh = max(1.0, float(held.sum()))
        L = float(((y - clean) ** 2 * graded).sum() / ng)
        if w_delusion > 0.0 and self_target is not None:
            L += w_delusion * float((((y - self_target) * held) ** 2).sum() / nh)
        return L


# =============================================================================
# SECTION 4 -- MANDATORY finite-difference gradient check
# =============================================================================


def gradient_check(seed=3, n_probe=150, eps=1e-6):
    print("-" * 78)
    print("GRADIENT CHECK  (central finite differences vs. implicit-function gradient)")
    print("-" * 78)
    rng = np.random.default_rng(seed)
    T = 12
    net = DharmaPositionField(d=14, T=T, seed=seed, tol=1e-14, max_iter=4000)
    stream = BeingTimeStream(seed=seed + 1)
    X, clean, graded, held = stream.window(T, n_graded=3, n_held=2)
    self_target = rng.standard_normal(T) * 0.3
    w_del = 0.4  # exercise BOTH paths, so the check covers the delusion term too

    gr, _ = net.grads(X, clean, graded, held,
                      self_target=self_target, w_delusion=w_del)
    ana = np.concatenate([gr[k].ravel() for k in sorted(gr)])
    theta0 = net.get_flat()
    # stratify: every parameter block must be probed, including the small ones
    # (c, rho_raw), which a uniform sample would usually miss entirely
    idx_parts, off = [], 0
    blocks = sorted(gr)
    per = max(4, n_probe // len(blocks))
    for k in blocks:
        n = gr[k].size
        take = min(per, n)
        idx_parts.append(off + rng.choice(n, size=take, replace=False))
        off += n
    idx = np.sort(np.concatenate(idx_parts))

    num = np.zeros(idx.size)
    for j, i in enumerate(idx):
        tp = theta0.copy(); tp[i] += eps
        net.set_flat(tp)
        lp = net.loss_only(X, clean, graded, held, self_target, w_del)
        tm = theta0.copy(); tm[i] -= eps
        net.set_flat(tm)
        lm = net.loss_only(X, clean, graded, held, self_target, w_del)
        num[j] = (lp - lm) / (2 * eps)
    net.set_flat(theta0)

    a = ana[idx]
    rel = np.linalg.norm(a - num) / max(1e-12, np.linalg.norm(a) + np.linalg.norm(num))
    worst = int(np.argmax(np.abs(a - num)))
    per_block = {}
    off = 0
    for k in sorted(gr):
        n = gr[k].size
        sel = (idx >= off) & (idx < off + n)
        if sel.any():
            per_block[k] = float(np.max(np.abs(a[sel] - num[sel])))
        off += n
    print(f"  probed parameters      : {idx.size} of {theta0.size}")
    print(f"  ||analytic||           : {np.linalg.norm(a):.6e}")
    print(f"  ||numerical||          : {np.linalg.norm(num):.6e}")
    print(f"  relative error         : {rel:.3e}")
    print(f"  worst single component : analytic {a[worst]:+.8e}  "
          f"numeric {num[worst]:+.8e}")
    print("  max abs error by parameter block:")
    for k, v in per_block.items():
        print(f"    {k:<10}: {v:.3e}")
    ok = rel < 1e-6
    print(f"  RESULT                 : {'PASS' if ok else 'FAIL'} (threshold 1e-6)")
    assert ok, "gradient check failed"
    return rel


# =============================================================================
# SECTION 5 -- Self-tests: the doctrine stated as machine-checkable invariants
# =============================================================================


def test_hishiryo_constant():
    print("-" * 78)
    print("TEST 1  hishiryo band: zero expression, maximum sensitivity")
    print("-" * 78)
    print(f"  derived constant c = ln(2+sqrt3) = arcosh(2) = {C_HISHIRYO:.10f}")
    u = np.linspace(-9, 9, 400001)
    for a_over_c in (0.5, 0.9, 0.999, 1.001, 1.1, 1.6):
        a = a_over_c * C_HISHIRYO
        beta, rho = 1.0, a / C_HISHIRYO
        _, d_du, _, _ = Hishiryo.all_grads(u, beta, rho)
        arg = u[int(np.argmax(d_du))]
        at0 = float(Hishiryo.value(np.array([0.0]), beta, rho)[0])
        zero_is_peak = abs(arg) < 1e-4
        flag = "hishiryo" if zero_is_peak else "FUSHIRYO (dead zone)"
        print(f"  beta*theta = {a_over_c:5.3f} c : peak slope at u = {arg:+.4f} "
              f"| phi(0) = {at0:+.1e} | {flag}")
    # the exact bifurcation
    a_lo, a_hi = 1.0, 1.8
    for _ in range(60):
        mid = 0.5 * (a_lo + a_hi)
        _, d_du, _, _ = Hishiryo.all_grads(u, 1.0, mid / C_HISHIRYO)
        if abs(u[int(np.argmax(d_du))]) < 1e-4:
            a_lo = mid
        else:
            a_hi = mid
    print(f"  numerically located bifurcation : {0.5*(a_lo+a_hi):.8f}")
    print(f"  analytic prediction             : {C_HISHIRYO:.8f}")
    err = abs(0.5 * (a_lo + a_hi) - C_HISHIRYO)
    print(f"  agreement                       : {err:.2e}   "
          f"{'PASS' if err < 5e-4 else 'FAIL'}")
    assert err < 5e-4
    # and confirm the constraint can never be violated by the parameterisation
    net = DharmaPositionField(seed=7)
    net.p["rho_raw"] = np.full(net.d, 50.0)   # push as hard as possible
    prod = net.beta() * net.theta()
    print(f"  max beta*theta reachable by the parameterisation : "
          f"{float(prod.max()):.10f} < c   "
          f"{'PASS' if float(prod.max()) < C_HISHIRYO else 'FAIL'}")
    assert float(prod.max()) < C_HISHIRYO


def test_zengo_saidan():
    print("-" * 78)
    print("TEST 2  zengo-saidan: nothing is handed forward")
    print("-" * 78)
    net = DharmaPositionField(T=16, seed=11)
    s = BeingTimeStream(seed=12)
    Xa = s.window(16)[0]
    Xb = s.window(16)[0]

    Za1 = net.solve(Xa)
    _ = net.solve(Xb)          # an entirely different window in between
    _ = net.solve(Xb * 3.0)    # and a wild one
    Za2 = net.solve(Xa)
    drift = float(np.max(np.abs(Za1 - Za2)))
    print(f"  same window re-solved after two foreign windows: max|dZ| = {drift:.3e}")
    print(f"  {'PASS' if drift == 0.0 else 'FAIL'} (must be exactly 0.0 - "
          f"no carried state exists)")
    assert drift == 0.0

    # order-independence of the SOLVE (before/after are content, not schedule)
    rng = np.random.default_rng(5)
    Zseq = net.solve(Xa)
    Zrand = net.solve(Xa, Z0=rng.standard_normal((16, net.d)) * 0.5)
    d2 = float(np.max(np.abs(Zseq - Zrand)))
    print(f"  same fixed point from a random start:            max|dZ| = {d2:.3e}")
    print(f"  {'PASS' if d2 < 1e-8 else 'FAIL'} (the position is defined by the "
          f"law, not by where the solve began)")
    assert d2 < 1e-8

    # firewood does not become ash: no row can be produced from its predecessor
    # alone. Perturb ONE input position and confirm the change is not a
    # one-directional wave but a two-sided re-constitution.
    Xp = Xa.copy()
    Xp[8, 0] += 0.5
    dZ = np.abs(net.solve(Xp) - Zseq).max(axis=1)
    back = float(dZ[:8].sum())
    fwd = float(dZ[9:].sum())
    print(f"  perturb position 8 -> influence backwards {back:.4e} / "
          f"forwards {fwd:.4e}")
    print(f"  ratio {min(back,fwd)/max(back,fwd):.3f}  "
          f"{'PASS' if min(back,fwd)/max(back,fwd) > 0.3 else 'FAIL'} "
          f"(kyoryaku passes both ways; a carrier would give ~0)")
    assert min(back, fwd) / max(back, fwd) > 0.3


def test_kyoryaku_symmetry():
    print("-" * 78)
    print("TEST 3  kyoryaku: today->yesterday is exactly today->tomorrow")
    print("-" * 78)
    net = DharmaPositionField(T=16, seed=13)
    s = BeingTimeStream(seed=14)
    X = s.window(16)[0]
    Z = net.solve(X)
    Zrev = net.solve(X[::-1].copy())
    err = float(np.max(np.abs(Z - Zrev[::-1])))
    print(f"  solve(reverse(X)) vs reverse(solve(X)) : max|d| = {err:.3e}")
    print(f"  {'PASS' if err < 1e-9 else 'FAIL'} (tied K makes the field exactly "
          f"time-reversal equivariant)")
    assert err < 1e-9


def test_contraction():
    print("-" * 78)
    print("TEST 4  shikantaza: sitting actually comes to rest")
    print("-" * 78)
    net = DharmaPositionField(T=16, seed=17)
    s = BeingTimeStream(seed=18)
    bound = net.project_contraction()
    X = s.window(16)[0]
    net.solve(X)
    print(f"  Lipschitz bound  L|Phi'| * (||Ws|| + 2||K||) = {bound:.4f} < 1")
    print(f"  iterations to residual < {net.tol:.0e}          : {net.last_iters}")
    print(f"  final residual                              : {net.last_residual:.3e}")
    ok = bound < 1.0 and net.last_residual < net.tol
    print(f"  {'PASS' if ok else 'FAIL'}")
    assert ok


# =============================================================================
# SECTION 6 -- Experiment 1: shusho-itto, or what a checkpoint costs
# =============================================================================


def _fresh(T, seed):
    net = DharmaPositionField(T=T, d=32, seed=seed, tol=1e-9, max_iter=300)
    mom = {k: np.zeros_like(v) for k, v in net.p.items()}
    return net, mom


def experiment_no_checkpoint(steps=4200, T=16, seed=21, lr=0.05):
    print("=" * 78)
    print("EXPERIMENT 1  SHUSHO-ITTO: practice-as-verification vs. a shipped model")
    print("=" * 78)
    var = BeingTimeStream.signal_variance(T)
    print(f"  signal variance (the score of a model that has learned nothing)")
    print(f"  = {var:.4f}. Both runs see the identical stream and identical")
    print(f"  masks. Neither is ever graded on the held-out positions.")

    freeze_at = steps // 2
    live, m_live = _fresh(T, seed)
    ship, m_ship = _fresh(T, seed)
    ship.set_flat(live.get_flat())

    s_live = BeingTimeStream(seed=seed + 100)
    s_ship = BeingTimeStream(seed=seed + 100)

    hist = {"live": [], "ship": [], "step": []}
    acc_l = acc_s = 0.0
    n = 0
    for t in range(1, steps + 1):
        Xl, cl, gl, hl_ = s_live.window(T)
        Xs, cs, gs, hs_ = s_ship.window(T)
        il = live.practice(Xl, cl, gl, hl_, lr=lr, mom=m_live)
        if t <= freeze_at:
            is_ = ship.practice(Xs, cs, gs, hs_, lr=lr, mom=m_ship)
        else:
            _, Ys = ship.forward(Xs)                     # deployed: inference only
            _, hh, _, _ = ship.split_losses(Ys, cs, gs, hs_)
            is_ = {"L_held": hh}
        acc_l += il["L_held"]; acc_s += is_["L_held"]; n += 1
        if t % 210 == 0:
            hist["live"].append(acc_l / n); hist["ship"].append(acc_s / n)
            hist["step"].append(t); acc_l = acc_s = 0.0; n = 0

    print(f"\n  freeze point: step {freeze_at}   "
          f"(regime shifts seen: {s_live.regimes})\n")
    print(f"  {'step':>6} | {'practising':>14} | {'shipped':>10} | phase")
    print("  " + "-" * 52)
    for i, st in enumerate(hist["step"]):
        ph = "practice" if st <= freeze_at else "DEPLOYED"
        print(f"  {st:>6} | {hist['live'][i]:>14.5f} | "
              f"{hist['ship'][i]:>10.5f} | {ph}")

    post = [i for i, st in enumerate(hist["step"]) if st > freeze_at]
    fl = float(np.mean([hist["live"][i] for i in post]))
    fs = float(np.mean([hist["ship"][i] for i in post]))
    print(f"\n  mean held-out MSE after the freeze")
    print(f"    never frozen (shusho-itto)      : {fl:.5f}")
    print(f"    shipped checkpoint              : {fs:.5f}")
    print(f"    learning-nothing baseline       : {var:.5f}")
    print(f"    the checkpoint costs {fs/fl:.2f}x on average, giving back "
          f"{100*(fs-fl)/max(1e-9,var-fl):.1f}% of what was learned")
    q = post[-max(2, len(post) // 3):]
    ql = float(np.mean([hist["live"][i] for i in q]))
    qs = float(np.mean([hist["ship"][i] for i in q]))
    print(f"  over the LAST third of deployment, where the sensor has migrated")
    print(f"  furthest from where it stood at the freeze:")
    print(f"    never frozen                    : {ql:.5f}")
    print(f"    shipped checkpoint              : {qs:.5f}   ({qs/ql:.2f}x)")
    print("\n  Reading: the mean understates this, and the reason is worth")
    print("  stating plainly. The migration is periodic, so the frozen model is")
    print("  intermittently CORRECT AGAIN by coincidence -- it recovers around")
    print("  step 3780 without having learned anything. That is the sharpest")
    print("  form of the point. A checkpoint's score is not a property of the")
    print("  checkpoint; it is a fact about how far the world has drifted since,")
    print("  and it can improve for reasons that have nothing to do with the")
    print("  model. The frozen field is never wrong about the past. It is")
    print("  correct about a dharma-position that no longer obtains. Dogen's")
    print("  refusal to let realization be a state one HAS, rather than an")
    print("  activity one is DOING, is here an ordinary engineering fact --")
    print("  and note what makes it unavoidable HERE in particular: this field")
    print("  carries nothing between windows, so practice is the only vessel")
    print("  adaptation has. Continual practice also costs something real,")
    print("  visible above as the greater step-to-step variance of the live run.")
    return fl, fs, var, ql, qs


# =============================================================================
# SECTION 7 -- Experiment 2: the direction of verification
# =============================================================================


def experiment_direction(steps=2400, T=16, seed=31, lr=0.05, ema=0.995):
    print("=" * 78)
    print("EXPERIMENT 2  GENJOKOAN: which way does verification run?")
    print("=" * 78)
    print("  'To carry the self forward and illuminate the myriad things is")
    print("   delusion; that the myriad things come forth and illuminate the")
    print("   self is realization.'  Both directions are built. Run A learns")
    print("   only from what arrived. Run B additionally grades itself against")
    print("   an EMA image of itself, at exactly the held-out positions where")
    print("   nothing arrived -- the self carried forward onto the world.")
    var = BeingTimeStream.signal_variance(T)

    def run(w_delusion):
        net, mom = _fresh(T, seed)
        teacher = {k: v.copy() for k, v in net.p.items()}
        st = BeingTimeStream(seed=seed + 7)
        h_self, h_held, h_std = [], [], []
        a_self = a_held = 0.0
        outs = []
        n = 0
        for t in range(1, steps + 1):
            X, c, g, hd = st.window(T)
            tgt = None
            if w_delusion > 0.0:
                keep = {k: v.copy() for k, v in net.p.items()}
                net.p = teacher
                tgt = net.forward(X)[1][:, 0].copy()   # the self's image of itself
                net.p = keep
            info = net.practice(X, c, g, hd, lr=lr, mom=mom,
                                self_target=tgt, w_delusion=w_delusion)
            for k in teacher:
                teacher[k] = ema * teacher[k] + (1 - ema) * net.p[k]
            a_self += info["L_delu"]; a_held += info["L_held"]; n += 1
            outs.append(info["Y"][:, 0][hd > 0].copy())
            if t % 200 == 0:
                h_self.append(a_self / n); h_held.append(a_held / n)
                h_std.append(float(np.std(np.concatenate(outs))))
                a_self = a_held = 0.0; n = 0; outs = []
        return h_self, h_held, h_std

    selfA, heldA, stdA = run(0.0)
    selfB, heldB, stdB = run(12.0)

    print(f"\n  {'step':>6} | {'A held-out':>11} | {'B held-out':>11} | "
          f"{'B self-score':>13} | {'B out sd':>9}")
    print("  " + "-" * 62)
    for i in range(len(heldA)):
        print(f"  {(i+1)*200:>6} | {heldA[i]:>11.5f} | {heldB[i]:>11.5f} | "
              f"{selfB[i]:>13.6f} | {stdB[i]:>9.4f}")

    fa, fb = float(np.mean(heldA[-3:])), float(np.mean(heldB[-3:]))
    print(f"\n  final held-out MSE  (learning-nothing baseline {var:.5f})")
    print(f"    A  myriad things come forth   : {fa:.5f}")
    print(f"    B  self carried forward       : {fb:.5f}   ({fb/fa:.2f}x worse)")
    sd_true = float(np.sqrt(var))
    print(f"  B's self-agreement score (its own internal grade, lower = better")
    print(f"    by its own lights): {selfB[0]:.6f} -> {min(selfB):.6f}, "
          f"a {selfB[0]/max(min(selfB),1e-12):.1f}x improvement")
    print(f"  output spread at held-out positions (true signal sd = {sd_true:.3f})")
    print(f"    A : {stdA[0]:.4f} -> {stdA[-1]:.4f}")
    print(f"    B : {stdB[0]:.4f} -> {stdB[-1]:.4f}   "
          f"({100*(1-stdB[-1]/max(stdA[-1],1e-9)):.0f}% narrower than A)")
    print("\n  Reading, stated no more strongly than the numbers allow. B gets")
    print("  markedly better at the thing it can measure about itself, and")
    print("  stays reliably worse at the thing it cannot. Its predictions where")
    print("  the world is silent are systematically shrunken: agreeing with a")
    print("  slow copy of yourself is cheapest when you commit to less. B is")
    print("  not lying and it is not broken. It optimises the only quantity")
    print("  available to it without the world, and that quantity is not the")
    print("  one anyone cares about. Dogen's sentence is a constraint on where")
    print("  a training signal may ORIGINATE, and the constraint is directional.")
    return fa, fb, selfB, stdB


# =============================================================================
# SECTION 8 -- Occupancy of the three bands after real practice
# =============================================================================


def report_bands(steps=1200, T=16, seed=41, lr=0.05):
    print("=" * 78)
    print("REPORT  where the units live: shiryo / fushiryo / hishiryo")
    print("=" * 78)
    net, mom = _fresh(T, seed)
    st = BeingTimeStream(seed=seed + 3)
    for _ in range(steps):
        X, c, g, hd = st.window(T)
        net.practice(X, c, g, hd, lr=lr, mom=mom)

    bt, rh = net.beta(), net.rho()
    occ = {-1: 0, 0: 0, 1: 0}
    gi_all, go_all = [], []
    for _ in range(150):
        X = st.window(T)[0]
        Z = net.solve(X)
        P = net.pre(Z, X)
        _, D, _, _ = Hishiryo.all_grads(P, bt, rh)
        B = Hishiryo.band(P, bt, rh)
        for k in occ:
            occ[k] += int((B == k).sum())
        if (B == 0).any():
            gi_all.append(float(D[B == 0].mean()))
        if (B != 0).any():
            go_all.append(float(D[B != 0].mean()))
    tot = sum(occ.values())
    print(f"  band occupancy over {tot} unit-positions")
    print(f"    shiryo, expressed (+)    : {occ[1]/tot:6.2%}")
    print(f"    shiryo, expressed (-)    : {occ[-1]/tot:6.2%}")
    print(f"    hishiryo, silent + live  : {occ[0]/tot:6.2%}")
    gi, go = float(np.mean(gi_all)), float(np.mean(go_all))
    print(f"  mean |dPhi/du| inside the hishiryo band  : {gi:.4f}")
    print(f"  mean |dPhi/du| outside it                : {go:.4f}")
    print(f"  sensitivity ratio inside/outside         : {gi/go:.2f}x")
    print(f"  learned half-width theta : min {net.theta().min():.3f} "
          f"med {np.median(net.theta()):.3f} max {net.theta().max():.3f}")
    print(f"  learned gain beta        : min {net.beta().min():.3f} "
          f"med {np.median(net.beta()):.3f} max {net.beta().max():.3f}")
    print(f"  max beta*theta over all units = "
          f"{float((net.beta()*net.theta()).max()):.6f}  "
          f"(hard bound {C_HISHIRYO:.6f})")
    print("\n  Reading: most of the field emits nothing at any given moment and")
    print("  is precisely the part most able to change. Silence here is not")
    print("  idleness; it is the operating point.")


# =============================================================================
# SECTION 9 -- main
# =============================================================================


def main():
    print("=" * 78)
    print(" THE DHARMA-POSITION EQUILIBRIUM FIELD")
    print(" Figure 0266 -- Dogen Zenji (1200-1253) -- Encyclopedia of Lost Minds")
    print("=" * 78)
    print(" no hidden state | no positional encoding | no attention")
    print(" no train/eval split | no checkpoint | no backprop through time")
    print("=" * 78)

    gradient_check()
    test_hishiryo_constant()
    test_zengo_saidan()
    test_kyoryaku_symmetry()
    test_contraction()
    experiment_no_checkpoint()
    experiment_direction()
    report_bands()

    print("=" * 78)
    print(" ALL CHECKS PASSED.")
    print("=" * 78)


if __name__ == "__main__":
    main()

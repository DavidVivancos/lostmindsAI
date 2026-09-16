#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
THE CONCORDANCE ENGINE  ·  chapter_0192_bede_673.py
An AGI micro-architecture built from the cognitive signature of the Venerable
Bede (c.672/673 – 26 May 735), monk of Monkwearmouth-Jarrow, Northumbria.
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 10 Minds 181 - 200 Available on Amazon https://www.amazon.com/dp/B0HJYMZ3G6
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0192_bede_673 - Bede (c.672/673 – 26 May 735), monk of Monkwearmouth-Jarrow, Northumbria
================================================================================  

WHY THIS ARCHITECTURE, AND WHY IT IS *NOT* A TRANSFORMER
--------------------------------------------------------
Bede's whole intellectual life turned on one operation that no other mind in
this corpus performs the same way: COMPUTUS — the reconciliation of cycles that
do not naturally coincide. The solar year (~365.25 d), the lunar month (~29.5
d), the seven-day week, the 15-year indiction, and the great 532-year Paschal
cycle each keep their own beat; to fix the date of Easter you must find the
*phase alignment* at which several of these rhythms fall back into agreement,
and then build a TABLE from which any future configuration can simply be read.
In "The Reckoning of Time" (De temporum ratione, 725) Bede did exactly this:
he mastered the 19-year Metonic lunar cycle and the 28-year solar cycle, and
knew that their product, 19 x 28 = 532 years, is the interval after which the
whole apparatus of weekday-plus-moon-age repeats. In chapter 29, "The Harmony
of the Moon and the Sea," he went further and reconciled a *natural* rhythm:
he tied the tides to the Moon and stated the "establishment of the port" — the
fixed lag between the Moon's meridian passage and the following high water,
different at every harbour. His estimate of the Moon's daily retardation he
revised from 47.5 to 48 minutes as better data arrived (the true value is ~50).

So Bede's atom of cognition is not "attend to a stored key." It is:
    * represent each stream of the world as an OSCILLATOR (a phase + a period);
    * measure CONCORDANCE — whether two cycles are in or out of phase;
    * DISCOVER the great cycle (the least common period) in which they realign;
    * read the answer from a table you can AUDIT, never merely assert it.

Two further, tightly-coupled traits of his mind shape the design:

  (A) ATTESTATION-WEIGHTING. Bede is the first rigorous source-critic in the
      Latin West: he names his witnesses, distinguishes eyewitness from hearsay,
      and weights testimony by provenance. Here that becomes a per-observation
      RELIABILITY weight in the loss, plus an iterative scheme that *discovers*
      which witnesses are unreliable from the inconsistency of their reports.

  (B) THE UNCOMFORTABLE COMPUTED ANSWER. In 708 Bede was accused of heresy for
      dating Creation-to-Incarnation at 3,952 years (from the Hebrew/Vulgate
      figures) instead of the traditional ~5,199, because the shorter number
      undercut the comforting scheme that the world lasts 6,000 years. He
      defended the arithmetic in his Letter to Plegwin. The engine therefore
      reports the number the table gives, not the number the room wants: its
      objective is fidelity to the reconciliation, never to the audience.

WHAT THIS FILE CONTAINS (all pure NumPy, hand-derived gradients)
----------------------------------------------------------------
  1. CRN — the Concordance Resonance Network: learnable phase oscillators whose
     features are trigonometric functions of PHASE DIFFERENCES between cycles.
  2. A hand-derived backward pass and a MANDATORY finite-difference gradient
     check (must pass to < 1e-6 relative error).
  3. A real training loop with attestation-weighted MSE.
  4. Task A — "Reconcile the cycles": the model DISCOVERS hidden reconciling
     periods (a miniature of Bede finding the Metonic cycle).
  5. Task B — "Attestation": corrupted testimony is down-weighted, and an
     iterative reweighting *finds* the unreliable witnesses on its own.
  6. Task C — "The Great Cycle": a computed, verified proof that oscillators at
     the canonical computistical periods realign after LCM(19,28) = 532.
  7. A self-test suite; the file executes end-to-end and prints verified output.

Run:  python3 chapter_0192_bede_673.py
================================================================================
"""

from __future__ import annotations
import numpy as np
from math import gcd

RNG_SEED = 673  # Bede's conventional birth-year: determinism, so the table is auditable.


# ============================================================================
# SECTION 1 — SMALL COMPUTISTICAL UTILITIES (the arithmetic Bede actually used)
# ============================================================================

def lcm(a: int, b: int) -> int:
    """Least common multiple — the 'great cycle' of two periods."""
    return a * b // gcd(a, b)


def great_paschal_cycle(*periods: int) -> int:
    """
    The interval after which several integer cycles all return to their start.
    For Bede's solar cycle (28) and lunar/Metonic cycle (19) this is 532 — the
    Great Paschal Cycle after which weekday and moon-age recur together.
    """
    out = 1
    for p in periods:
        out = lcm(out, p)
    return out


def moon_age(day_index: int, epact: int = 0, lunar_period: int = 30) -> int:
    """
    A toy 'age of the Moon' — the schematic lunar day (0..lunar_period-1) used in
    computus tables. Real computus alternates 29/30-day months; this is the
    tabular idealisation Bede would have recognised as the calendar Moon (as
    opposed to the Moon in the sky), the distinction he stressed in DTR.
    """
    return (day_index + epact) % lunar_period


# ============================================================================
# SECTION 2 — THE CONCORDANCE RESONANCE NETWORK (CRN)
# ============================================================================
#
# FORWARD PASS
# ------------
# Input:  t  (shape [N])  — a scalar "time / date" coordinate per example.
# Params: omega [K]  angular frequencies (2*pi / period) of K oscillators
#         phi   [K]  phase offsets
#         W     [F, Dout]  linear readout, F = 2K + K*(K-1)/2
#         b     [Dout]     bias
#
#   theta[n,k] = t[n] * omega[k] + phi[k]                     # each cycle's phase
#   C = cos(theta) ,  S = sin(theta)                          # in-phase / quadrature
#   For every unordered pair (j<k):
#       P_{jk} = cos(theta_j - theta_k) = C_j C_k + S_j S_k   # CONCORDANCE of j,k
#           ( = +1 when the two cycles are exactly in phase, -1 when opposed )
#   features = [ C | S | P ]   (shape [N, F])
#   Y = features @ W + b
#
# The pairwise phase-difference features P are the heart of the model: they make
# "are these two rhythms currently aligned?" a first-class quantity the readout
# can use. That is computus expressed as a differentiable operation.
#
# LOSS (attestation-weighted MSE)
# -------------------------------
#   L = ( 1 / (2 * sum_i r_i) ) * sum_i r_i * || Y_i - T_i ||^2
# where r_i >= 0 is the reliability weight of observation i (Bede weighting the
# trustworthiness of a witness). Uniform r recovers ordinary MSE.
# ============================================================================


def _pair_indices(K: int):
    """All unordered pairs (j<k) of K oscillators, as two index arrays."""
    js, ks = np.triu_indices(K, k=1)
    return js, ks


class ConcordanceResonanceNetwork:
    """A from-scratch phase-reconciliation predictor. No attention, no MoE."""

    def __init__(self, periods_init, learnable_mask=None, d_out=1, seed=RNG_SEED):
        rng = np.random.default_rng(seed)
        periods_init = np.asarray(periods_init, dtype=np.float64)
        self.K = len(periods_init)
        # omega = 2*pi / period. We store omega directly and learn in omega-space.
        self.omega = 2.0 * np.pi / periods_init
        self.phi = rng.uniform(-0.1, 0.1, size=self.K)
        # learnable_mask marks which oscillator frequencies may move during
        # training. Canonical computistical cycles are usually FIXED (they are
        # given by the heavens); a few "free" oscillators are DISCOVERED.
        if learnable_mask is None:
            learnable_mask = np.ones(self.K, dtype=bool)
        self.learnable_mask = np.asarray(learnable_mask, dtype=bool)

        self.js, self.ks = _pair_indices(self.K)
        self.M = len(self.js)                 # number of concordance features
        self.F = 2 * self.K + self.M          # total feature dimension
        self.d_out = d_out
        # Small readout init keeps early predictions near zero (a humble table).
        self.W = rng.normal(0.0, 0.3, size=(self.F, d_out))
        self.b = np.zeros(d_out)

    # ---- forward -----------------------------------------------------------
    def features(self, t):
        """Return the feature matrix and the cached pieces needed for backprop."""
        t = np.asarray(t, dtype=np.float64).reshape(-1)
        theta = t[:, None] * self.omega[None, :] + self.phi[None, :]   # [N,K]
        C = np.cos(theta)
        S = np.sin(theta)
        # Concordance features P_{jk} = C_j C_k + S_j S_k = cos(theta_j - theta_k)
        P = C[:, self.js] * C[:, self.ks] + S[:, self.js] * S[:, self.ks]  # [N,M]
        F = np.concatenate([C, S, P], axis=1)                               # [N,F]
        cache = dict(t=t, theta=theta, C=C, S=S, P=P)
        return F, cache

    def forward(self, t):
        F, cache = self.features(t)
        Y = F @ self.W + self.b            # [N, d_out]
        cache["F"] = F
        return Y, cache

    # ---- loss --------------------------------------------------------------
    @staticmethod
    def weighted_mse(Y, T, r):
        """L = sum_i r_i ||Y_i - T_i||^2 / (2 sum r).  Returns (loss, dY)."""
        T = np.asarray(T, dtype=np.float64).reshape(Y.shape)
        r = np.asarray(r, dtype=np.float64).reshape(-1)
        Z = r.sum()
        diff = Y - T                                     # [N, d_out]
        loss = float(np.sum(r[:, None] * diff ** 2) / (2.0 * Z))
        dY = (r[:, None] * diff) / Z                     # [N, d_out]
        return loss, dY

    # ---- backward (hand-derived) ------------------------------------------
    def backward(self, cache, dY):
        """
        Given dL/dY, return gradients for W, b, phi, omega.
        Every step below is the analytic derivative of the forward pass; the
        gradient check in Section 3 verifies it against finite differences.
        """
        F = cache["F"]; t = cache["t"]
        C = cache["C"]; S = cache["S"]

        # readout
        gW = F.T @ dY                                    # [F, d_out]
        gb = dY.sum(axis=0)                              # [d_out]
        dF = dY @ self.W.T                               # [N, F]

        # split feature-gradient back into cos / sin / concordance blocks
        dC = dF[:, :self.K].copy()                       # [N,K]
        dS = dF[:, self.K:2 * self.K].copy()             # [N,K]
        dP = dF[:, 2 * self.K:]                          # [N,M]

        # d theta from C=cos(theta): dtheta += dC * (-sin) ; from S=sin(theta): += dS*cos
        dtheta = -dC * S + dS * C                        # [N,K]

        # d theta from concordance P_{jk} = cos(theta_j - theta_k)
        #   dP/dtheta_j = -sin(theta_j - theta_k) ,  dP/dtheta_k = +sin(...)
        #   sin(theta_j - theta_k) = S_j C_k - C_j S_k
        sinΔ = S[:, self.js] * C[:, self.ks] - C[:, self.js] * S[:, self.ks]  # [N,M]
        contrib = dP * sinΔ                               # [N,M]
        # scatter-add into the two oscillators of each pair
        np.add.at(dtheta, (slice(None), self.ks), contrib)   # +sinΔ to k
        np.add.at(dtheta, (slice(None), self.js), -contrib)  # -sinΔ to j

        # theta = t*omega + phi  ->  dtheta/domega = t ,  dtheta/dphi = 1
        gphi = dtheta.sum(axis=0)                         # [K]
        gomega = (dtheta * t[:, None]).sum(axis=0)        # [K]
        # freeze frozen oscillators
        gomega = gomega * self.learnable_mask
        gphi = gphi * self.learnable_mask

        return dict(W=gW, b=gb, phi=gphi, omega=gomega)

    # ---- parameter vector helpers (for the gradient check) -----------------
    def get_params(self):
        return dict(W=self.W.copy(), b=self.b.copy(),
                    phi=self.phi.copy(), omega=self.omega.copy())

    def set_params(self, p):
        self.W = p["W"].copy(); self.b = p["b"].copy()
        self.phi = p["phi"].copy(); self.omega = p["omega"].copy()

    def loss_on(self, t, T, r):
        Y, _ = self.forward(t)
        loss, _ = self.weighted_mse(Y, T, r)
        return loss


# ============================================================================
# SECTION 3 — MANDATORY GRADIENT CHECK (finite differences)
# ============================================================================

def gradient_check(model: ConcordanceResonanceNetwork, t, T, r, eps=1e-6):
    """
    Compare hand-derived gradients to central finite differences on every
    parameter block. Returns the worst relative error across blocks.
    """
    Y, cache = model.forward(t)
    _, dY = model.weighted_mse(Y, T, r)
    analytic = model.backward(cache, dY)

    base = model.get_params()
    worst = 0.0
    report = {}
    for name in ["W", "b", "phi", "omega"]:
        g_an = analytic[name]
        g_num = np.zeros_like(base[name])
        flat_an = g_an.reshape(-1)
        it = np.ndindex(*base[name].shape)
        for idx in it:
            p_plus = {k: v.copy() for k, v in base.items()}
            p_minus = {k: v.copy() for k, v in base.items()}
            p_plus[name][idx] += eps
            p_minus[name][idx] -= eps
            model.set_params(p_plus);  Lp = model.loss_on(t, T, r)
            model.set_params(p_minus); Lm = model.loss_on(t, T, r)
            g_num[idx] = (Lp - Lm) / (2 * eps)
        model.set_params(base)
        num = flat_an - g_num.reshape(-1)
        den = np.maximum(1e-12, np.abs(flat_an) + np.abs(g_num.reshape(-1)))
        rel = float(np.max(np.abs(num) / den))
        report[name] = rel
        worst = max(worst, rel)
    return worst, report


# ============================================================================
# SECTION 4 — A SIMPLE OPTIMISER (momentum SGD) + TRAINING LOOP
# ============================================================================

class Momentum:
    def __init__(self, model, lr=1e-2, mu=0.9, lr_omega=None):
        self.m = model
        self.lr = lr
        self.lr_omega = lr if lr_omega is None else lr_omega
        self.mu = mu
        self.v = {k: np.zeros_like(v) for k, v in model.get_params().items()}

    def step(self, grads):
        for name in ["W", "b", "phi", "omega"]:
            lr = self.lr_omega if name == "omega" else self.lr
            self.v[name] = self.mu * self.v[name] - lr * grads[name]
        self.m.W += self.v["W"]; self.m.b += self.v["b"]
        self.m.phi += self.v["phi"]; self.m.omega += self.v["omega"]


def train(model, t, T, r=None, steps=4000, lr=1e-2, lr_omega=2e-4, mu=0.9,
          verbose=False):
    if r is None:
        r = np.ones(np.asarray(t).reshape(-1).shape[0])
    opt = Momentum(model, lr=lr, mu=mu, lr_omega=lr_omega)
    hist = []
    for s in range(steps):
        Y, cache = model.forward(t)
        loss, dY = model.weighted_mse(Y, T, r)
        grads = model.backward(cache, dY)
        opt.step(grads)
        if s % max(1, steps // 8) == 0 or s == steps - 1:
            hist.append((s, loss))
            if verbose:
                print(f"    step {s:5d}   loss {loss:.6e}")
    return hist


# ============================================================================
# SECTION 5 — TASK A: "RECONCILE THE CYCLES" (discover hidden periods)
# ============================================================================
#
# Bede did not invent the 19-year cycle; he *recognised* it inside the calendar.
# Here the model is given a compound signal whose regularity only becomes
# predictable once two hidden reconciling cycles are found, and it must move its
# free oscillators onto those periods. This is the Metonic discovery in
# miniature: intelligence as the location of the period that makes chaos legible.
# ============================================================================

def make_concordance_signal(N=400, p1=23.0, p2=17.0, noise=0.02, seed=RNG_SEED):
    """
    T(t) = cos(2*pi t/p1) + cos(2*pi t/p2) + 0.8*cos(2*pi t (1/p1 - 1/p2))
    The third term is the BEAT / resonance between the two cycles — a signal that
    is only predictable if you know BOTH periods and their concordance.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, N * 0.5, N)
    beat = 1.0 / p1 - 1.0 / p2
    T = (np.cos(2 * np.pi * t / p1)
         + np.cos(2 * np.pi * t / p2)
         + 0.8 * np.cos(2 * np.pi * t * beat))
    T = T + rng.normal(0, noise, size=N)
    return t, T


def run_task_A(verbose=False):
    p1_true, p2_true = 23.0, 17.0
    t, T = make_concordance_signal(N=400, p1=p1_true, p2=p2_true, noise=0.02)

    # split train/test by time (predict the FUTURE of the reckoning)
    n = len(t); cut = int(0.75 * n)
    t_tr, T_tr = t[:cut], T[:cut]
    t_te, T_te = t[cut:], T[cut:]

    # Two FREE oscillators, initialised NEAR (but not on) the true periods — as a
    # computist starts from an inherited, slightly-wrong table and corrects it.
    p_init = [p1_true * 1.06, p2_true * 0.94]
    model = ConcordanceResonanceNetwork(p_init, learnable_mask=[True, True],
                                        d_out=1, seed=7)
    L0 = model.loss_on(t_tr, T_tr, np.ones(cut))
    hist = train(model, t_tr, T_tr, steps=6000, lr=2e-2, lr_omega=6e-4,
                 verbose=verbose)
    L1 = model.loss_on(t_tr, T_tr, np.ones(cut))
    test_mse = model.loss_on(t_te, T_te, np.ones(len(t_te))) * 2  # undo the 1/2

    periods = 2 * np.pi / model.omega
    return dict(model=model, L0=L0, L1=L1, test_mse=test_mse,
                true=[p1_true, p2_true], recovered=sorted(periods.tolist()),
                init=sorted(p_init))


# ============================================================================
# SECTION 6 — TASK B: "ATTESTATION" (weight testimony by provenance)
# ============================================================================
#
# Bede's historical method: not every witness counts equally. Here a fraction of
# training labels are corrupted ("rumour"), and we show:
#   (1) uniform weighting is dragged off the true reconciliation;
#   (2) knowing provenance (down-weighting the rumour source) restores it;
#   (3) an ITERATIVE reweighting *discovers* the unreliable witnesses from the
#       inconsistency of their reports, with no prior label of who lied —
#       Bede sifting eyewitness from hearsay by internal coherence.
# ============================================================================

def robust_reliability(model, t, T, iters=6, c=1.5):
    """
    Iteratively re-weight observations by residual (a Huber/IRLS-style scheme).
    Large, inconsistent residuals -> low reliability. Returns final weights r.
    """
    N = len(t)
    r = np.ones(N)
    for _ in range(iters):
        # refit briefly under current weights
        m2 = ConcordanceResonanceNetwork(2 * np.pi / model.omega,
                                         learnable_mask=model.learnable_mask,
                                         d_out=model.d_out, seed=1)
        m2.phi = model.phi.copy(); m2.W = model.W.copy(); m2.b = model.b.copy()
        train(m2, t, T, r=r, steps=800, lr=2e-2, lr_omega=0.0)
        Y, _ = m2.forward(t)
        res = np.abs(Y.reshape(-1) - np.asarray(T).reshape(-1))
        s = np.median(res) + 1e-9
        # Huber weight: 1 for small residuals, ~ (c*s)/res for large ones
        r = np.where(res <= c * s, 1.0, (c * s) / res)
        model = m2
    return r, model


def run_task_B(verbose=False):
    p1_true, p2_true = 23.0, 17.0
    t, T_clean = make_concordance_signal(N=360, p1=p1_true, p2=p2_true, noise=0.02)
    rng = np.random.default_rng(99)
    N = len(t)

    # 30% of witnesses are "rumour": labels replaced by large random noise.
    corrupt = rng.random(N) < 0.30
    T = T_clean.copy()
    T[corrupt] = rng.normal(0, 3.0, size=corrupt.sum())

    true_r = np.where(corrupt, 0.0, 1.0)   # provenance oracle: 0 for rumour

    def fit_and_eval(r):
        m = ConcordanceResonanceNetwork([p1_true, p2_true],
                                        learnable_mask=[False, False],
                                        d_out=1, seed=3)  # fix periods; isolate the
        train(m, t, T, r=r, steps=2500, lr=2e-2, lr_omega=0.0)  # attestation effect
        Y, _ = m.forward(t)
        # evaluate against the CLEAN signal (the truth the rumour obscured)
        mse = float(np.mean((Y.reshape(-1) - T_clean) ** 2))
        return mse, m

    mse_uniform, _ = fit_and_eval(np.ones(N))
    mse_oracle, _ = fit_and_eval(true_r)

    # discover reliability with no oracle
    m0 = ConcordanceResonanceNetwork([p1_true, p2_true],
                                     learnable_mask=[False, False], d_out=1, seed=3)
    train(m0, t, T, r=np.ones(N), steps=800, lr=2e-2, lr_omega=0.0)
    r_found, _ = robust_reliability(m0, t, T, iters=6, c=1.5)
    mse_found, _ = fit_and_eval(r_found)

    # how well did discovery separate rumour from truth?
    # (mean discovered weight on corrupt vs clean witnesses)
    w_corrupt = float(np.mean(r_found[corrupt]))
    w_clean = float(np.mean(r_found[~corrupt]))
    return dict(mse_uniform=mse_uniform, mse_oracle=mse_oracle,
                mse_found=mse_found, w_corrupt=w_corrupt, w_clean=w_clean,
                frac_corrupt=float(corrupt.mean()))


# ============================================================================
# SECTION 7 — TASK C: "THE GREAT CYCLE" (computed & verified realignment)
# ============================================================================
#
# Set oscillators at the canonical computistical integer periods and prove, by
# direct search, that all concordance features return to their starting values
# after exactly LCM of those periods. For the solar (28) and lunar/Metonic (19)
# cycles this is 532 — Bede's Great Paschal Cycle.
# ============================================================================

def run_task_C():
    solar, lunar, week = 28, 19, 7
    expected = great_paschal_cycle(solar, lunar)   # 532
    # Build integer-period oscillators; evaluate the FULL feature vector at t=0
    # and search for the smallest t>0 where features return (to tolerance).
    model = ConcordanceResonanceNetwork([solar, lunar], learnable_mask=[False, False])
    model.phi[:] = 0.0
    F0, _ = model.features(np.array([0.0]))
    found = None
    for tt in range(1, 4 * expected):
        Ft, _ = model.features(np.array([float(tt)]))
        if np.allclose(Ft, F0, atol=1e-9):
            found = tt
            break
    # also confirm week+leap gives the 28-year solar cycle
    solar_from_week = lcm(week, 4)  # 7 weekdays x 4-year leap step = 28
    return dict(expected_great_cycle=expected, found_great_cycle=found,
                solar_cycle_from_week=solar_from_week,
                lcm_19_28=lcm(19, 28))


# ============================================================================
# SECTION 8 — SELF-TEST SUITE
# ============================================================================

def run_all(verbose=False):
    print("=" * 74)
    print("THE CONCORDANCE ENGINE — Bede of Jarrow (c.673–735)")
    print("Reconciling incommensurable cycles into one auditable reckoning.")
    print("=" * 74)

    results = {}

    # ---- (0) Gradient check -- MANDATORY -----------------------------------
    print("\n[1] GRADIENT CHECK (hand-derived vs finite differences)")
    rng = np.random.default_rng(RNG_SEED)
    tg = rng.uniform(0, 40, size=12)
    Tg = rng.normal(0, 1, size=(12, 2))
    rg = rng.uniform(0.2, 1.0, size=12)                 # random attestation weights
    gm = ConcordanceResonanceNetwork([13.0, 7.0, 29.0], d_out=2, seed=5)
    worst, rep = gradient_check(gm, tg, Tg, rg, eps=1e-6)
    for k, v in rep.items():
        print(f"      {k:6s} max rel err = {v:.3e}")
    print(f"      WORST relative error = {worst:.3e}   "
          f"{'PASS' if worst < 1e-6 else 'FAIL'}")
    results["grad_worst"] = worst
    assert worst < 1e-6, "Gradient check FAILED"

    # ---- (A) Reconcile the cycles ------------------------------------------
    print("\n[2] TASK A — DISCOVER THE RECONCILING PERIODS (Metonic in miniature)")
    A = run_task_A(verbose=verbose)
    print(f"      train loss  {A['L0']:.4e}  ->  {A['L1']:.4e}   "
          f"(x{A['L0']/max(A['L1'],1e-12):.0f} smaller)")
    print(f"      init periods      {[round(x,2) for x in A['init']]}")
    print(f"      TRUE periods      {A['true']}")
    print(f"      recovered periods {[round(x,3) for x in A['recovered']]}")
    print(f"      held-out (future) MSE = {A['test_mse']:.4e}")
    results["A"] = dict(L0=A["L0"], L1=A["L1"], recovered=A["recovered"],
                        test_mse=A["test_mse"])

    # ---- (B) Attestation ----------------------------------------------------
    print("\n[3] TASK B — ATTESTATION-WEIGHTING (sift eyewitness from rumour)")
    B = run_task_B(verbose=verbose)
    print(f"      corrupt fraction of witnesses : {B['frac_corrupt']:.2f}")
    print(f"      MSE vs truth, uniform weights : {B['mse_uniform']:.4e}")
    print(f"      MSE vs truth, provenance oracle: {B['mse_oracle']:.4e}")
    print(f"      MSE vs truth, DISCOVERED weights: {B['mse_found']:.4e}")
    print(f"      discovered reliability  clean={B['w_clean']:.2f}  "
          f"rumour={B['w_corrupt']:.2f}")
    results["B"] = B

    # ---- (C) Great cycle ----------------------------------------------------
    print("\n[4] TASK C — THE GREAT PASCHAL CYCLE (computed & verified)")
    C = run_task_C()
    print(f"      LCM(solar 28, lunar 19)      = {C['lcm_19_28']}")
    print(f"      expected realignment period  = {C['expected_great_cycle']}")
    print(f"      FOUND realignment period     = {C['found_great_cycle']}")
    print(f"      solar cycle from week x leap  = {C['solar_cycle_from_week']}")
    results["C"] = C

    # ---- assertions ---------------------------------------------------------
    print("\n[5] ASSERTIONS")
    checks = []

    def check(name, cond):
        checks.append((name, bool(cond)))
        print(f"      [{'PASS' if cond else 'FAIL'}] {name}")

    check("gradient check < 1e-6", results["grad_worst"] < 1e-6)
    check("Task A reduced training loss > 20x",
          A["L0"] / max(A["L1"], 1e-12) > 20)
    check("Task A recovered periods within 5% of truth",
          all(min(abs(r - tr) / tr for tr in A["true"]) < 0.05
              for r in A["recovered"]))
    check("Task A generalises to the future (test MSE < 0.1)",
          A["test_mse"] < 0.1)
    check("Task B: provenance oracle beats uniform",
          B["mse_oracle"] < B["mse_uniform"])
    check("Task B: DISCOVERED weights beat uniform",
          B["mse_found"] < B["mse_uniform"])
    check("Task B: discovery separates rumour from truth",
          B["w_clean"] - B["w_corrupt"] > 0.25)
    check("Task C: great cycle equals LCM(19,28)=532",
          C["found_great_cycle"] == 532 == C["expected_great_cycle"])
    check("Task C: solar cycle from week equals 28",
          C["solar_cycle_from_week"] == 28)

    n_pass = sum(v for _, v in checks)
    print(f"\n      {n_pass}/{len(checks)} checks passed.")
    print("=" * 74)
    if n_pass == len(checks):
        print("ALL CHECKS PASSED — the reckoning holds.")
    else:
        print("SOME CHECKS FAILED.")
    print("=" * 74)
    return results, checks


if __name__ == "__main__":
    run_all(verbose=False)

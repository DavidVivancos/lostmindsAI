#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
 Encyclopedia of Lost Minds: Echoes on AI
 Chapter 0267 -- Su Song (蘇頌, courtesy name Zirong 子容, 1020-1101 CE)
 Northern Song dynasty polymath, statesman, and builder of the Cosmic Engine
 (水運儀象臺, Shui Yun Yi Xiang Tai) at Kaifeng, completed 1092-1094 CE.

 ARCHITECTURE: the Escapement-Gated Recurrent World Model (EGRWM)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 14, Minds 261-280 Available on Amazon https://www.amazon.com/dp/B0HLYSVY5D
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0267_su_song_1020 - Su Song (1020-1101)
=============================================================================

WHY THIS SHAPE, AND NOT A TRANSFORMER
--------------------------------------
Su Song's own recorded words -- from the memorial he presented to the throne
describing why the clock tower's drive had to be built exactly as it was --
give the single sentence this file is built around (translated by Joseph
Needham from the Xin Yi Xiang Fa Yao, 1092):

    "The heavens move without ceasing, but so also does water flow (and
    fall). Thus if the water is made to pour with perfect evenness, then
    the comparison of the rotary movements [of the heavens and the machine]
    will show no discrepancy or contradiction; for the unresting follows
    the unceasing."

Two ideas live in that sentence and both are load-bearing for the network
below:

  1. A mind that models a ceaseless process (the heavens) must itself run
     as a ceaseless process (the water), never halting to "wait" for input.
     -> The hidden state integrates every timestep, unconditionally.

  2. Fidelity is not continuous adjustment -- it is a *periodic, regulated,
     discrete comparison* ("no discrepancy or contradiction") between the
     free-running model and a checked reference, released at fixed
     intervals by an escapement, never oftener and never at arbitrary
     moments chosen by the model itself.
     -> Correction only happens at fixed "ticks"; between ticks the state
        free-runs open-loop.

Su Song's real tower divided the day into 600 equal releases of its
"celestial balance" (the escapement Needham's translation calls the
tian heng) -- 2 minutes 24 seconds of unceasing flow per tick, every tick
identical, every tick either confirmed or corrected against the sighted
sky. That is the architectural commitment encoded here: correction is not
continuous (which invites over-reacting to a single noisy sighting) and it
is not absent (which invites unbounded drift) -- it is regulated.

A second, independent strand of Su Song's practice supplies the second
mechanism in this file. The Bencao Tujing (Illustrated Classic of Materia
Medica, source material gathered 1058-1061 under imperial decree, edited
by Su Song's team into final form) was compiled by soliciting samples and
written reports of local drugs, minerals, and plants from every prefecture
of the empire and reconciling them against each other and against the
classical textual record. Some provincial reports were detailed and
trustworthy; some were vague or second-hand. Su Song's method assumed
unequal, not equal, reliability across sources, and fused them
accordingly rather than averaging blindly. That is the instrument-fusion
layer below: several noisy "provincial reports" (instrument channels) of
uncertain and unequal reliability, reconciled into one fused reading
before the reading is even allowed to reach the escapement.

Finally, Su Song wrote the Xin Yi Xiang Fa Yao specifically so the tower
could be rebuilt after him -- and after the Jin sack of Kaifeng in 1127,
his own son Su Xie, working from that same treatise, could not
reconstruct it, and concluded his father must have omitted something.
Whether or not that suspicion was fair, the historical fact is not in
dispute: the written specification, however exhaustive, did not by
itself transmit the tacit, living skill of the mechanism to the next
mind. The "metacognitive audit head" below is this file's attempt to take
that failure seriously rather than paper over it: the network is trained
to also predict its own forthcoming correction, i.e. to know in advance
how wrong its free-running belief probably is -- a form of self-monitoring
that a static written specification cannot provide, because a
specification does not know how confident IT is.

WHAT THE NETWORK ACTUALLY DOES
-------------------------------
Task: track the phase (position on a circle) of a slowly, irregularly
precessing celestial body, given several noisy, intermittently-available
"provincial report" style sensor channels, and remain accurate over long
stretches where no sensor reports at all (clouded-out nights).

Components (see class docstrings for the historical mapping of each):
  1. ProvincialFusion   -- reliability-weighted fusion of K noisy channels
  2. UnceasingIntegrator -- an autonomous recurrent "water wheel" hidden
                             state that free-runs every single timestep
  3. Escapement          -- a FIXED-PERIOD gate; only at tick steps, and
                             only if a fused reading exists, is the free-
                             running state pulled toward the checked
                             reading, by a learned per-dimension gain
  4. CelestialReadout    -- linear projection of the hidden state to a
                             (sin, cos) phase estimate, read out every step
  5. AuditHead           -- predicts the magnitude of the correction the
                             escapement is about to apply, BEFORE it is
                             applied (metacognitive self-monitoring)

Everything is implemented from scratch in NumPy: forward pass, manual
backpropagation-through-time, a mandatory finite-difference gradient
check, an Adam training loop, and a self-test suite comparing three
correction regimes (open-loop / naive continuous / escapement-regulated)
on held-out episodes. Run this file directly to execute all of it.

Author's note: this is the architecture in executable form, kept
independent of the companion chapter text. No results from running it are
reproduced in the chapter -- the file speaks for itself when you run it.
"""

from __future__ import annotations

import numpy as np


# =============================================================================
# PART 0 -- small numeric primitives (used by forward AND backward passes)
# =============================================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def dsigmoid_from_output(s: np.ndarray) -> np.ndarray:
    """d/dx sigmoid(x), expressed in terms of the already-computed output s."""
    return s * (1.0 - s)


def dtanh_from_output(t: np.ndarray) -> np.ndarray:
    """d/dx tanh(x), expressed in terms of the already-computed output t."""
    return 1.0 - t * t


def masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Softmax over `logits` restricted to entries where mask == 1.
    Entries where mask == 0 receive weight exactly 0.
    If mask is all-zero, returns an all-zero vector (caller must handle
    the "no instrument reported this step" case explicitly).
    logits, mask: shape (K,)
    """
    if not np.any(mask):
        return np.zeros_like(logits)
    neg_inf = -1e30
    masked_logits = np.where(mask > 0, logits, neg_inf)
    m = np.max(masked_logits)
    ex = np.where(mask > 0, np.exp(masked_logits - m), 0.0)
    s = ex.sum()
    return ex / s


# =============================================================================
# PART 1 -- synthetic task: multi-instrument celestial phase tracking
# =============================================================================
#
# The hidden "true sky" is a phase theta(t) in [0, 2*pi) advancing with a
# slowly, seasonally varying angular rate -- mirroring the real fact that
# apparent solar motion is not perfectly uniform (the "equation of time")
# even though the underlying rotation is, to the eye, unceasing.
#
# K noisy channels ("provincial reports") observe this phase at every
# timestep but are each present (reporting) only with their own
# probability, and each with its own noise level -- some channels are
# trustworthy but rare, others are frequent but noisy, deliberately
# mirroring the uneven quality of the Bencao Tujing's provincial returns.

def generate_episode(T: int, K: int, rng: np.random.Generator,
                      base_rate: float = 0.09,
                      seasonal_amp: float = 0.02,
                      seasonal_period: int = 60,
                      channel_report_prob=None,
                      channel_noise=None):
    """
    Returns:
      obs        : (T, K, 2) float64 -- noisy (sin, cos) per channel per step
      mask       : (T, K)    float64 -- 1.0 if channel reported this step
      true_phase : (T, 2)    float64 -- ground-truth (sin, cos) of theta(t)
    """
    if channel_report_prob is None:
        # channel 0: frequent, noisier (a routine local water-clock reading)
        # channel 1: rare, precise (a careful sighted observation)
        # channel 2: medium/medium (a relayed provincial report)
        channel_report_prob = np.array([0.55, 0.12, 0.30])[:K]
    if channel_noise is None:
        channel_noise = np.array([0.35, 0.05, 0.18])[:K]

    theta = rng.uniform(0, 2 * np.pi)
    true_phase = np.zeros((T, 2))
    obs = np.zeros((T, K, 2))
    mask = np.zeros((T, K))

    for t in range(T):
        omega = base_rate + seasonal_amp * np.sin(2 * np.pi * t / seasonal_period)
        theta = (theta + omega) % (2 * np.pi)
        true_phase[t, 0] = np.sin(theta)
        true_phase[t, 1] = np.cos(theta)
        for k in range(K):
            if rng.uniform() < channel_report_prob[k]:
                mask[t, k] = 1.0
                noisy_theta = theta + rng.normal(0, channel_noise[k])
                obs[t, k, 0] = np.sin(noisy_theta)
                obs[t, k, 1] = np.cos(noisy_theta)
            # else: leave obs at 0, mask stays 0 (channel silent this step)
    return obs, mask, true_phase


# =============================================================================
# PART 2 -- parameters
# =============================================================================

class EGRWMParams:
    """
    All learnable parameters of the Escapement-Gated Recurrent World Model.
    Stored as a flat dict of named arrays so the optimizer, the gradient
    checker, and the backward pass can all iterate over the same structure
    without special-casing any one parameter.
    """

    def __init__(self, K: int, D: int, H: int, obs_dim: int = 2,
                 out_dim: int = 2, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.K, self.D, self.H = K, D, H
        self.obs_dim, self.out_dim = obs_dim, out_dim

        def xavier(shape):
            fan_in = shape[-1] if len(shape) > 1 else shape[0]
            fan_out = shape[0]
            lim = np.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-lim, lim, size=shape)

        p = {}
        # 1) Provincial fusion: per-instrument encoder + shared reliability probe
        p['W_obs'] = xavier((K, D, obs_dim))     # per-channel encoder
        p['b_obs'] = np.zeros((K, D))
        p['v_rel'] = xavier((D,))                # reliability probe (shared)

        # 2) Unceasing integrator (autonomous recurrent "water wheel")
        p['W_hh'] = xavier((H, H))
        p['b_h'] = np.zeros((H,))

        # 3) Escapement: fused-embedding -> hidden space, plus a learned
        #    per-dimension release gain (init near 0.5 via zero logits)
        p['W_e2h'] = xavier((H, D))
        p['b_e2h'] = np.zeros((H,))
        p['G_raw'] = np.zeros((H,))              # sigmoid(G_raw) = release gain

        # 4) Celestial readout
        p['W_out'] = xavier((out_dim, H))
        p['b_out'] = np.zeros((out_dim,))

        # 5) Audit / metacognitive head (predicts forthcoming correction size)
        p['w_meta'] = xavier((H,))
        p['b_meta'] = np.zeros((1,))

        self.p = p

    def keys(self):
        return list(self.p.keys())

    def __getitem__(self, k):
        return self.p[k]

    def __setitem__(self, k, v):
        self.p[k] = v

    def zeros_like(self) -> "EGRWMParams":
        z = EGRWMParams.__new__(EGRWMParams)
        z.K, z.D, z.H = self.K, self.D, self.H
        z.obs_dim, z.out_dim = self.obs_dim, self.out_dim
        z.p = {k: np.zeros_like(v) for k, v in self.p.items()}
        return z

    def flat_copy(self):
        return {k: v.copy() for k, v in self.p.items()}


# =============================================================================
# PART 3 -- forward pass
# =============================================================================

def forward(params: EGRWMParams, obs: np.ndarray, mask: np.ndarray,
            tick_period: int):
    """
    Runs the EGRWM over one episode.

    obs  : (T, K, obs_dim)
    mask : (T, K)
    tick_period : escapement period in steps. tick_period == 1 means
                  "correct every step from that step's reading alone" (the
                  naive, un-regulated regime used only for the comparison
                  study). tick_period > T effectively disables the
                  escapement entirely (pure open-loop regime, also used
                  only for the comparison study). Su Song's own design
                  choice is a small, FIXED, regular period -- neither of
                  those extremes.

    The escapement here is mechanically faithful to the historical device
    in a specific way: the real scoop accumulates water CONTINUOUSLY
    between releases and tips only once the accumulated weight crosses
    threshold -- it does not discard everything but the very last drop.
    Correspondingly, every step's provincial-fusion reading is folded into
    a running accumulator; at each tick, the accumulator's MEAN (not just
    the current step's single reading) is what gets released as the
    correction target, and the accumulator is then emptied. Between ticks,
    the state free-runs; readings that arrive between ticks are not
    thrown away, they are banked.

    Returns:
      y_hat      : (T, out_dim)  -- celestial readout at every step
      meta_hat   : (T,)          -- predicted correction magnitude at every
                                     step (only meaningful / trained at
                                     tick steps where a correction occurs)
      cache      : dict of everything needed by backward()
    """
    p = params.p
    T, K, obs_dim = obs.shape
    H, D = params.H, params.D
    out_dim = params.out_dim

    h_prev = np.zeros(H)
    y_hat = np.zeros((T, out_dim))
    meta_hat = np.zeros(T)

    accum_e = np.zeros(D)
    accum_count = 0
    interval_members = []   # step indices banked since the last tick

    # per-timestep caches (lists, index t)
    cache = {
        'obs': obs, 'mask': mask, 'tick_period': tick_period, 'T': T,
        'h_prev_list': [], 'h_free_list': [], 'is_tick': [],
        'has_fused_step': [], 'has_release': [],
        'e_step_list': [], 'e_k_list': [], 'alpha_list': [],
        'e_release_list': [], 'e2h_list': [], 'gate_list': [],
        'h_list': [], 'corr_list': [],
        'owning_tick': {}, 'weight': {}, 'count_by_tick': {},
    }

    for t in range(T):
        cache['h_prev_list'].append(h_prev.copy())

        # ---- (1) Provincial fusion of THIS step's raw channel reports ----
        e_k = np.zeros((K, D))       # per-channel encoded embedding
        rel_logits = np.zeros(K)     # reliability logit per channel
        for k in range(K):
            e_k[k] = np.tanh(p['W_obs'][k] @ obs[t, k] + p['b_obs'][k])
            rel_logits[k] = p['v_rel'] @ e_k[k]
        alpha = masked_softmax(rel_logits, mask[t])   # (K,)
        has_fused_step = bool(np.any(mask[t] > 0))
        e_step = (alpha[:, None] * e_k).sum(axis=0) if has_fused_step else np.zeros(D)

        if has_fused_step:
            accum_e = accum_e + e_step
            accum_count += 1
            interval_members.append(t)

        # ---- (2) Unceasing integrator: the state ALWAYS advances ----
        h_free = np.tanh(p['W_hh'] @ h_prev + p['b_h'])

        # ---- (3) Escapement: fixed-period, gated correction on the
        #          ACCUMULATED (not merely most-recent) evidence ----
        is_tick = (t % tick_period) == 0
        gate = sigmoid(p['G_raw'])                     # (H,), release gain

        has_release = False
        e_release = np.zeros(D)
        if is_tick:
            cache['count_by_tick'][t] = accum_count
            if accum_count > 0:
                e_release = accum_e / accum_count
                has_release = True
                w = 1.0 / accum_count
                for m in interval_members:
                    cache['owning_tick'][m] = t
                    cache['weight'][m] = w
            # the scoop is emptied whether or not it had anything in it
            accum_e = np.zeros(D)
            accum_count = 0
            interval_members = []

        e2h = p['W_e2h'] @ e_release + p['b_e2h']       # (H,)
        if is_tick and has_release:
            h_t = h_free + gate * (e2h - h_free)
            corr = h_t - h_free                          # actual correction applied
        else:
            h_t = h_free
            corr = np.zeros(H)

        # ---- (4) Celestial readout, every step ----
        y_hat[t] = p['W_out'] @ h_t + p['b_out']

        # ---- (5) Audit head: predicts ||corr|| computed from h_free ALONE
        #          (i.e. before the correction is known) ----
        meta_hat[t] = float(p['w_meta'] @ h_free + p['b_meta'][0])

        # stash
        cache['h_free_list'].append(h_free.copy())
        cache['is_tick'].append(is_tick)
        cache['has_fused_step'].append(has_fused_step)
        cache['has_release'].append(has_release)
        cache['e_step_list'].append(e_step.copy())
        cache['e_k_list'].append(e_k.copy())
        cache['alpha_list'].append(alpha.copy())
        cache['e_release_list'].append(e_release.copy())
        cache['e2h_list'].append(e2h.copy())
        cache['gate_list'].append(gate.copy())
        cache['h_list'].append(h_t.copy())
        cache['corr_list'].append(corr.copy())

        h_prev = h_t

    return y_hat, meta_hat, cache


# =============================================================================
# PART 4 -- loss
# =============================================================================

def meta_targets_from_cache(cache: dict) -> np.ndarray:
    """
    The audit head's training target: the magnitude of the correction the
    escapement actually applied at each step. This is intentionally a
    STOP-GRADIENT target (self-distillation style) -- the audit head is
    trained to *predict* the coming correction, not to reshape it, so no
    gradient should ever flow from the auxiliary loss back through the
    target itself. Exposed as a standalone function so that both the
    training loss and the gradient checker use exactly the same,
    explicitly-frozen target array.
    """
    return np.array([np.linalg.norm(c) for c in cache['corr_list']])


def compute_loss(y_hat, meta_hat, true_phase, cache, meta_weight=0.5,
                  l2_weight=1e-4, params: EGRWMParams = None,
                  meta_targets: np.ndarray = None):
    """
    Primary loss: mean squared error of the (sin, cos) readout against the
    true phase, at every timestep (the globe is watched continuously).

    Auxiliary loss: mean squared error between the audit head's PRE-tick
    prediction of the correction magnitude and the correction magnitude
    that was actually applied at tick steps with a fused reading -- this
    is the metacognitive self-monitoring signal. The target is a
    stop-gradient quantity (see meta_targets_from_cache); pass a frozen
    `meta_targets` array explicitly when validating gradients so the
    finite-difference probe perturbs only the prediction, exactly as the
    analytic backward() pass does. If omitted, it is computed fresh from
    `cache` (the normal path during training).

    Returns scalar loss and the two per-step gradient seeds needed to
    start backpropagation: dL/dy_hat (T,out_dim) and dL/dmeta_hat (T,).
    """
    T, out_dim = y_hat.shape
    diff = y_hat - true_phase
    main_loss = np.mean(np.sum(diff ** 2, axis=1))
    dL_dyhat = (2.0 / T) * diff  # (T, out_dim)

    if meta_targets is None:
        meta_targets = meta_targets_from_cache(cache)
    is_tick = np.array(cache['is_tick'])
    has_release = np.array(cache['has_release'])
    active = is_tick & has_release
    n_active = max(int(active.sum()), 1)

    meta_diff = np.where(active, meta_hat - meta_targets, 0.0)
    aux_loss = np.sum(meta_diff ** 2) / n_active
    dL_dmetahat = np.where(active, (2.0 / n_active) * meta_diff, 0.0)

    l2 = 0.0
    if params is not None:
        for k, v in params.p.items():
            l2 += np.sum(v ** 2)
    l2_loss = l2_weight * l2

    total = main_loss + meta_weight * aux_loss + l2_loss
    return total, dL_dyhat, meta_weight * dL_dmetahat


# =============================================================================
# PART 5 -- backward pass (manual backpropagation-through-time)
# =============================================================================

def backward(params: EGRWMParams, cache: dict, dL_dyhat: np.ndarray,
             dL_dmetahat: np.ndarray, l2_weight: float = 1e-4) -> "EGRWMParams":
    """
    Computes dL/d(every parameter) by running back through time from the
    last step to the first, accumulating:
      (a) the gradient flowing in locally at each step (readout head +
          audit head, both of which read from this step's activations), and
      (b) the gradient flowing back through the hidden-state recurrence
          from step t+1 into step t.
    """
    p = params.p
    K, D, H = params.K, params.D, params.H
    T = dL_dyhat.shape[0]
    obs = cache['obs']

    grads = params.zeros_like()
    g = grads.p

    dh_next = np.zeros(H)          # gradient flowing back from h_t into h_{t-1}
    de_release_by_tick = {}        # tick_index -> gradient w.r.t. that tick's e_release

    for t in reversed(range(T)):
        h_prev = cache['h_prev_list'][t]
        h_free = cache['h_free_list'][t]
        h_t = cache['h_list'][t]
        is_tick = cache['is_tick'][t]
        has_release = cache['has_release'][t]
        e_release = cache['e_release_list'][t]
        e_k = cache['e_k_list'][t]
        alpha = cache['alpha_list'][t]
        e2h = cache['e2h_list'][t]
        gate = cache['gate_list'][t]

        # ---- readout head: y_hat_t = W_out @ h_t + b_out ----
        dy = dL_dyhat[t]                       # (out_dim,)
        g['W_out'] += np.outer(dy, h_t)
        g['b_out'] += dy
        dh_t = params.p['W_out'].T @ dy        # gradient into h_t from readout

        # ---- audit head: meta_hat_t = w_meta . h_free + b_meta ----
        dm = dL_dmetahat[t]                    # scalar
        g['w_meta'] += dm * h_free
        g['b_meta'] += dm
        dh_free_from_meta = dm * params.p['w_meta']   # (H,)

        # ---- add gradient arriving from the NEXT timestep's recurrence ----
        dh_t = dh_t + dh_next

        # ---- split dh_t back through the escapement branch ----
        if is_tick and has_release:
            # h_t = h_free + gate * (e2h - h_free)
            dh_free_local = dh_t * (1.0 - gate)
            dgate = dh_t * (e2h - h_free)
            de2h = dh_t * gate
            # gate = sigmoid(G_raw)
            g['G_raw'] += dgate * dsigmoid_from_output(gate)
            # e2h = W_e2h @ e_release + b_e2h
            g['W_e2h'] += np.outer(de2h, e_release)
            g['b_e2h'] += de2h
            de_release_by_tick[t] = params.p['W_e2h'].T @ de2h
        else:
            dh_free_local = dh_t.copy()
            if is_tick:
                de_release_by_tick[t] = np.zeros(D)

        dh_free_total = dh_free_local + dh_free_from_meta

        # ---- h_free = tanh(W_hh @ h_prev + b_h) ----
        dpre_h = dh_free_total * dtanh_from_output(h_free)
        g['W_hh'] += np.outer(dpre_h, h_prev)
        g['b_h'] += dpre_h
        dh_prev = params.p['W_hh'].T @ dpre_h

        dh_next = dh_prev  # feed into the previous timestep's recurrence

        # ---- route this step's OWN contribution to whichever future tick
        #      it was banked into (interval-mean accumulator) ----
        owning_tick = cache['owning_tick'].get(t, None)
        if owning_tick is not None:
            weight = cache['weight'][t]
            de_step = weight * de_release_by_tick[owning_tick]   # (D,)

            # ---- provincial fusion at step t:
            #      e_step = sum_k alpha_k * e_k, alpha = softmax(rel_logits) ----
            de_k = alpha[:, None] * de_step[None, :]              # (K,D) direct term
            dalpha = e_k @ de_step                                 # (K,)
            avail = cache['mask'][t] > 0
            if avail.any():
                a = alpha[avail]
                da = dalpha[avail]
                s = np.sum(a * da)
                dlogits_avail = a * (da - s)
                dlogits = np.zeros(K)
                dlogits[avail] = dlogits_avail
            else:
                dlogits = np.zeros(K)

            for k in range(K):
                de_k[k] += dlogits[k] * params.p['v_rel']
                g['v_rel'] += dlogits[k] * e_k[k]
                dpre = de_k[k] * dtanh_from_output(e_k[k])
                g['W_obs'][k] += np.outer(dpre, obs[t, k])
                g['b_obs'][k] += dpre
        # steps whose reading was never banked (silent channels) contribute
        # nothing to the fusion parameters at this timestep.

    # L2 weight-decay gradient contribution (added once, outside the loop)
    for k, v in params.p.items():
        g[k] += 2.0 * l2_weight * v

    return grads


# =============================================================================
# PART 6 -- finite-difference gradient check (MANDATORY)
# =============================================================================

def numeric_gradient_check(seed: int = 0, T: int = 7, K: int = 2, D: int = 3,
                            H: int = 3, tick_period: int = 3, eps: float = 1e-5,
                            n_probe_per_param: int = 3, rtol: float = 3e-2,
                            atol: float = 1e-4, verbose: bool = False) -> bool:
    """
    Verifies the analytic backward() pass against central finite differences
    on a handful of randomly chosen entries in EVERY parameter tensor.

    A small episode is deliberately used (T=7, K=2, D=3, H=3) so the check
    runs in well under a second while still exercising every branch of the
    forward pass (tick steps with fused data, tick steps without, non-tick
    steps, multiple reporting channels).
    """
    rng = np.random.default_rng(seed)
    params = EGRWMParams(K=K, D=D, H=H, seed=seed)
    obs, mask, true_phase = generate_episode(T, K, rng)
    # force at least one tick step to have a fused reading and one to not,
    # so both branches of the escapement are actually exercised by the check
    mask[0, :] = 1.0
    if tick_period < T:
        mask[tick_period, :] = 0.0

    # The audit head's target is an intentional stop-gradient quantity (see
    # meta_targets_from_cache's docstring): backward() never differentiates
    # through it. To validate backward() with finite differences, the
    # target must therefore be frozen at its BASE-POINT value before any
    # parameter is perturbed -- otherwise the numeric probe would also pick
    # up the (deliberately excluded) gradient path through the target
    # itself, and the two would never agree.
    base_y_hat, base_meta_hat, base_cache = forward(params, obs, mask,
                                                      tick_period)
    frozen_meta_targets = meta_targets_from_cache(base_cache)

    def loss_only(pp: EGRWMParams) -> float:
        y_hat, meta_hat, cache = forward(pp, obs, mask, tick_period)
        loss, _, _ = compute_loss(y_hat, meta_hat, true_phase, cache,
                                   params=pp, meta_targets=frozen_meta_targets)
        return loss

    loss, dL_dyhat, dL_dmetahat = compute_loss(
        base_y_hat, base_meta_hat, true_phase, base_cache, params=params,
        meta_targets=frozen_meta_targets)
    analytic = backward(params, base_cache, dL_dyhat, dL_dmetahat)

    all_ok = True
    worst = 0.0
    for key in params.keys():
        shape = params.p[key].shape
        flat_len = int(np.prod(shape))
        n_probe = min(n_probe_per_param, flat_len)
        idxs = rng.choice(flat_len, size=n_probe, replace=False)
        for idx in idxs:
            multi = np.unravel_index(idx, shape)
            orig = params.p[key][multi]

            params.p[key][multi] = orig + eps
            loss_plus = loss_only(params)
            params.p[key][multi] = orig - eps
            loss_minus = loss_only(params)
            params.p[key][multi] = orig  # restore

            numeric = (loss_plus - loss_minus) / (2 * eps)
            ana = analytic.p[key][multi]

            denom = max(abs(numeric), abs(ana), 1e-8)
            rel_err = abs(numeric - ana) / denom
            worst = max(worst, rel_err)
            ok = (abs(numeric - ana) <= atol) or (rel_err <= rtol)
            if verbose:
                print(f"  {key}{multi}: analytic={ana:+.6e} "
                      f"numeric={numeric:+.6e} rel_err={rel_err:.4f} "
                      f"{'OK' if ok else 'FAIL'}")
            all_ok = all_ok and ok

    if verbose:
        print(f"gradient check: worst relative error = {worst:.5f} "
              f"-> {'PASS' if all_ok else 'FAIL'}")
    return all_ok


# =============================================================================
# PART 7 -- Adam optimizer (from scratch) and training loop
# =============================================================================

class Adam:
    def __init__(self, params: EGRWMParams, lr=0.01, beta1=0.9, beta2=0.999,
                 eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.p.items()}
        self.t = 0

    def step(self, params: EGRWMParams, grads: EGRWMParams):
        self.t += 1
        for k in params.p:
            g = grads.p[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params.p[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


def train(params: EGRWMParams, n_episodes: int, T: int, K: int,
          tick_period: int, seed: int = 0, lr: float = 0.01,
          log_every: int = 0):
    """
    One learnable-parameter pass over `n_episodes` freshly sampled synthetic
    episodes (stochastic online training -- a new "sky" each episode).
    Returns the list of per-episode losses.
    """
    rng = np.random.default_rng(seed)
    opt = Adam(params, lr=lr)
    losses = []
    for ep in range(n_episodes):
        obs, mask, true_phase = generate_episode(T, K, rng)
        y_hat, meta_hat, cache = forward(params, obs, mask, tick_period)
        loss, dL_dyhat, dL_dmetahat = compute_loss(y_hat, meta_hat,
                                                     true_phase, cache,
                                                     params=params)
        grads = backward(params, cache, dL_dyhat, dL_dmetahat)
        opt.step(params, grads)
        losses.append(loss)
        if log_every and (ep + 1) % log_every == 0:
            recent = np.mean(losses[-log_every:])
            print(f"  episode {ep+1:5d}/{n_episodes}  mean loss (last "
                  f"{log_every}) = {recent:.5f}")
    return losses


# =============================================================================
# PART 8 -- evaluation: does the escapement regime actually earn its keep?
# =============================================================================

def evaluate(params: EGRWMParams, n_episodes: int, T: int, K: int,
             tick_period: int, seed: int) -> float:
    """Mean phase-angle error (radians) on freshly sampled held-out episodes."""
    rng = np.random.default_rng(seed)
    errs = []
    for _ in range(n_episodes):
        obs, mask, true_phase = generate_episode(T, K, rng)
        y_hat, _, _ = forward(params, obs, mask, tick_period)
        true_theta = np.arctan2(true_phase[:, 0], true_phase[:, 1])
        pred_theta = np.arctan2(y_hat[:, 0], y_hat[:, 1])
        d = np.abs(np.angle(np.exp(1j * (true_theta - pred_theta))))
        errs.append(np.mean(d))
    return float(np.mean(errs))


def compare_regimes(K=3, D=12, H=12, T=180, train_episodes=400,
                     eval_episodes=60, seed=0, verbose=True):
    """
    Trains three separately-initialized models under identical data and
    identical architecture, differing ONLY in the escapement's tick_period:

      open_loop        : tick_period = T + 1   -> correction never fires;
                          the model must survive on free-running dynamics
                          plus whatever the fusion layer contributes to
                          initial conditions (it does not, since fusion
                          feeds only the escapement branch) -- i.e. this
                          is pure dead-reckoning.
      naive_continuous : tick_period = 1        -> corrects every single
                          step whenever ANY channel reports, including a
                          single noisy reading -- physically what Su Song
                          explicitly rejected in favour of a regulated
                          release.
      su_song_regulated: tick_period = 6        -> a small fixed period,
                          the direct analogue of the historical
                          escapement's fixed division of the day.

    Returns a dict of {regime_name: mean_held_out_phase_error_radians}.
    """
    regimes = {
        'open_loop': T + 1,
        'naive_continuous': 1,
        'su_song_regulated': 6,
    }
    results = {}
    for i, (name, period) in enumerate(regimes.items()):
        params = EGRWMParams(K=K, D=D, H=H, seed=seed + 100 * i)
        train(params, n_episodes=train_episodes, T=T, K=K,
              tick_period=period, seed=seed + i, lr=0.01, log_every=0)
        err = evaluate(params, n_episodes=eval_episodes, T=T, K=K,
                        tick_period=period, seed=seed + 999 + i)
        results[name] = err
        if verbose:
            print(f"  regime={name:<18s} tick_period={period:<4d} "
                  f"held-out mean phase error = {err:.4f} rad")
    return results


# =============================================================================
# PART 9 -- self-test suite
# =============================================================================

def _test_shapes():
    params = EGRWMParams(K=3, D=5, H=5, seed=1)
    rng = np.random.default_rng(1)
    obs, mask, true_phase = generate_episode(11, 3, rng)
    y_hat, meta_hat, cache = forward(params, obs, mask, tick_period=4)
    assert y_hat.shape == (11, 2)
    assert meta_hat.shape == (11,)
    print("  [PASS] shapes")


def _test_escapement_fires_on_schedule():
    params = EGRWMParams(K=2, D=4, H=4, seed=2)
    rng = np.random.default_rng(2)
    obs, mask, true_phase = generate_episode(10, 2, rng)
    mask[:, :] = 1.0  # force every channel to always report
    _, _, cache = forward(params, obs, mask, tick_period=3)
    ticks = [t for t in range(10) if cache['is_tick'][t]]
    assert ticks == [0, 3, 6, 9], ticks
    assert all(cache['has_release'][t] for t in ticks)
    print("  [PASS] escapement fires exactly on its fixed schedule")


def _test_reproducibility():
    rng1 = np.random.default_rng(7)
    obs1, mask1, tp1 = generate_episode(20, 3, rng1)
    p1 = EGRWMParams(K=3, D=6, H=6, seed=7)
    y1, _, _ = forward(p1, obs1, mask1, tick_period=4)

    rng2 = np.random.default_rng(7)
    obs2, mask2, tp2 = generate_episode(20, 3, rng2)
    p2 = EGRWMParams(K=3, D=6, H=6, seed=7)
    y2, _, _ = forward(p2, obs2, mask2, tick_period=4)

    assert np.allclose(y1, y2), "same seed must give bit-identical rollouts"
    print("  [PASS] deterministic reproducibility under a fixed seed")


def _test_training_reduces_loss():
    params = EGRWMParams(K=3, D=10, H=10, seed=3)
    losses = train(params, n_episodes=150, T=80, K=3, tick_period=6, seed=3,
                    lr=0.02)
    early = np.mean(losses[:15])
    late = np.mean(losses[-15:])
    assert late < early, f"loss did not improve: early={early:.4f} late={late:.4f}"
    print(f"  [PASS] training reduces loss ({early:.4f} -> {late:.4f})")


def run_self_tests():
    print("Running self-tests for the Escapement-Gated Recurrent World Model")
    print("-" * 72)
    print("[1/5] Gradient check (finite differences vs. analytic backward)")
    ok = numeric_gradient_check(verbose=False)
    assert ok, "GRADIENT CHECK FAILED"
    print("  [PASS] analytic gradients match finite differences")
    print("[2/5] Shape sanity")
    _test_shapes()
    print("[3/5] Escapement fires on its fixed schedule")
    _test_escapement_fires_on_schedule()
    print("[4/5] Deterministic reproducibility")
    _test_reproducibility()
    print("[5/5] Training actually reduces loss")
    _test_training_reduces_loss()
    print("-" * 72)
    print("All self-tests passed.")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    run_self_tests()
    print()
    print("Comparing three escapement regimes on held-out episodes")
    print("-" * 72)
    compare_regimes()

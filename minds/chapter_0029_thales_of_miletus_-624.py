#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chapter_0029_thales_of_miletus_-624.py — The Arche-Flow
# Part Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 2 Minds 21 - 40 Available on Amazon https://www.amazon.com/dp/B0H6QCQ9M7
# Resume and Interactive Demos at https://artificiology.com/
Author: David Vivancos · Chapter 0029 · Thales of Miletus
==============================================================================
A from-scratch (pure-NumPy) neural architecture that embodies the cognitive
signature of THALES OF MILETUS (c. 624 – c. 546 BCE), the first named
philosopher of the Western tradition.

WHY THIS ARCHITECTURE IS *THALES-SPECIFIC* (and not a generic world-model)
------------------------------------------------------------------------------
None of Thales' own writings survive. Everything we have about his philosophy
of mind reaches us *mediated* through Aristotle, Herodotus, and Diogenes
Laertius. From that doxography three ideas are reliably attributed to him, and
together they form a single, unusual cognitive doctrine that this code makes
literal:

  1. ARCHE / MONISM. There is one underlying substance — water (hydor) — and
     the bewildering diversity of the world is that one substance in different
     states and transformations (Aristotle, Metaphysics 983b20).
     ->  In code: there is exactly ONE latent state vector `h` (the "substance").
         Every observable is a *transformation* (read-out) of that one
         substance. We never add a second, separate representational store.

  2. HYLOZOISM / SELF-MOTION = SOUL. "Thales held that the soul is a thing
     that produces motion" and inferred soul in the lodestone (magnetite)
     "because it moves iron" (Aristotle, De Anima 405a19; 411a7: "all things
     are full of gods"). Mind, for Thales, is not a separate stuff: it is the
     *self-originated motion* latent in matter.
     ->  In code: the dynamics split into an AUTONOMOUS field v_auto(h) — the
         substance moving *itself* — and an EXTERNAL push v_ext(x) — the world
         shoving it. We define a measurable "psyche" = the fraction of total
         motion that is self-originated. The Lodestone Test below verifies that
         this number distinguishes a self-mover from an inert body.

  3. THE ECLIPSE / SAROS PREDICTION. Thales is said to have predicted the solar
     eclipse of 28 May 585 BCE (Herodotus 1.74) — i.e. he had internalised a
     *recurring celestial cycle* well enough to project it forward with no new
     observation.
     ->  In code: the training task is to forecast a quasi-periodic "eclipse"
         signal, and the decisive test is FREE-RUN PREDICTION: we cut off all
         external input (v_ext = 0) and require the substance to keep moving
         *itself* and reproduce the cycle. A Thales mind proves understanding by
         continuing the heavens after the data stops.

So the mechanism is a continuous-substance dynamical system (a learned vector
field integrated by explicit Euler steps), NOT attention over stored keys.
Monism is one latent field; hylozoism is the autonomous-vs-external split;
prediction is the free-run. The architecture *is* the philosophy.

WHAT THIS FILE DOES (all real, all runs)
------------------------------------------------------------------------------
  * Implements the Arche-Flow forward pass and exact Back-Prop-Through-Time.
  * Verifies those analytic gradients against finite differences (MANDATORY
    check; must pass for every parameter tensor).
  * Trains the model with plain SGD+momentum on a synthetic Saros-like signal.
  * Performs FREE-RUN eclipse prediction (input removed) and reports error.
  * Runs the LODESTONE TEST: shows the psyche (self-motion) measure separates
    a self-mover from an inert body — Thales' own criterion for soul.
  * Self-tests with assertions; prints a verified report; exits non-zero on
    any failure so the run is trustworthy.

Run:  python3 chapter_0029_thales_of_miletus_-624.py
Dependencies: numpy only.
==============================================================================
"""

from __future__ import annotations
import numpy as np

# A single global RNG so every run is reproducible (a Thales virtue: the
# heavens repeat; so should the experiment).
RNG = np.random.default_rng(585)  # seed = the eclipse year, for luck.


# =============================================================================
# 1. THE SUBSTANCE — synthetic "Saros" eclipse signal
# -----------------------------------------------------------------------------
# Eclipses recur on the ~18-year Saros cycle, but each return is slightly
# displaced (the Saros is not a whole number of days). We model an "eclipse
# depth" signal as a dominant slow cycle modulated by a second, incommensurate
# cycle — quasi-periodic, never exactly repeating, exactly the kind of pattern
# that rewards an *internal model* over rote memorisation.
# =============================================================================
def make_saros_signal(n: int, dt: float = 1.0) -> np.ndarray:
    """Return a length-n quasi-periodic 'eclipse depth' time series in [-1, 1].

    Two incommensurate sinusoids (the principal Saros return and a slower
    inclination drift) plus a sharp 'occultation' dip when both align — the
    moment the disc is covered. The result repeats *approximately* but never
    exactly, so a model must learn the generating dynamics, not the samples.
    """
    t = np.arange(n) * dt
    saros = np.sin(2.0 * np.pi * t / 18.03)          # principal return
    drift = 0.45 * np.sin(2.0 * np.pi * t / 41.0)    # slow inclination drift
    base = 0.7 * saros + 0.3 * drift
    # A narrow deepening near each conjunction (a stylised totality):
    occult = -0.35 * np.exp(-((np.cos(np.pi * t / 9.015)) ** 2) * 6.0)
    sig = base + occult
    sig = sig / (np.max(np.abs(sig)) + 1e-9)         # normalise to [-1, 1]
    return sig.astype(np.float64)


# =============================================================================
# 2. THE ARCHE-FLOW MODEL
# -----------------------------------------------------------------------------
# State (the one substance):     h_t  in R^H
# Autonomous self-motion (soul): v_auto(h) = W_a @ tanh(W_h @ h + b_h)
# External push (the world):     v_ext(x)  = W_x @ x          (x is scalar here)
# Euler integration (the flow):  h_{t+1} = h_t + dt * (v_auto + v_ext)
# Transformation / read-out:     y_t = w_o . h_{t+1} + b_o    (one phenomenon)
#
# Monism is enforced structurally: there is a single h; y is only a projection
# of h. Hylozoism is enforced structurally: v_auto depends on h alone (the
# substance moving itself); v_ext is the only channel for the world. Cutting
# v_ext to zero (free-run) leaves *pure self-motion* — Thales' soul, isolated.
# =============================================================================
class ArcheFlow:
    def __init__(self, hidden: int = 16, dt: float = 0.5):
        self.H = hidden
        self.dt = dt
        s = 1.0 / np.sqrt(hidden)
        # Parameters (the only learnable tensors — one substance, few laws):
        self.W_h = RNG.uniform(-s, s, size=(hidden, hidden))   # field shape
        self.b_h = np.zeros(hidden)                            # field bias
        self.W_a = RNG.uniform(-s, s, size=(hidden, hidden))   # self-motion map
        self.W_x = RNG.uniform(-s, s, size=(hidden, 1))        # external coupling
        self.w_o = RNG.uniform(-s, s, size=hidden)             # read-out
        self.b_o = 0.0                                         # read-out bias

    # ---- parameter (un)packing, used only by the gradient checker ----------
    def get_params(self):
        return {
            "W_h": self.W_h, "b_h": self.b_h, "W_a": self.W_a,
            "W_x": self.W_x, "w_o": self.w_o, "b_o": np.array(self.b_o),
        }

    def set_param(self, name, value):
        if name == "b_o":
            self.b_o = float(np.asarray(value).ravel()[0])
        else:
            setattr(self, name, value)

    # ---- forward pass with cached activations for BPTT ---------------------
    def forward(self, x_seq, target_seq, free_run_from=None):
        """Roll the substance forward.

        x_seq        : (T,) external inputs (teacher-forced observations).
        target_seq   : (T,) values to predict (next-step observations).
        free_run_from: if set to index k, external input is zeroed for t >= k
                       and the model is fed its OWN previous prediction — pure
                       self-motion, the eclipse continued without new data.

        Returns (loss, cache, y_seq).
        """
        T = len(x_seq)
        H = self.H
        h = np.zeros(H)                 # h_0 : the still water before motion
        hs = [h]                        # h_0 .. h_T
        as_ = []                        # tanh activations a_t
        xs = []                         # the input actually used at step t
        ys = np.zeros(T)
        loss = 0.0
        for t in range(T):
            if free_run_from is not None and t >= free_run_from:
                x_t = ys[t - 1] if t > 0 else x_seq[0]   # feed own prediction
                ext_on = 0.0                              # cut the world off
            else:
                x_t = x_seq[t]
                ext_on = 1.0
            pre = self.W_h @ h + self.b_h
            a = np.tanh(pre)
            v_auto = self.W_a @ a
            v_ext = (self.W_x[:, 0] * x_t) * ext_on
            v = v_auto + v_ext
            h = h + self.dt * v
            y = self.w_o @ h + self.b_o
            # accumulate loss (skip during free-run: there are no labels in the
            # future the way Thales had none — but we still record y for scoring)
            if free_run_from is None or t < free_run_from:
                loss += 0.5 * (y - target_seq[t]) ** 2
            hs.append(h)
            as_.append(a)
            xs.append(x_t * ext_on)
            ys[t] = y
        cache = {"hs": hs, "as_": as_, "xs": xs, "ys": ys,
                 "free_run_from": free_run_from}
        n_scored = T if free_run_from is None else free_run_from
        return loss / max(n_scored, 1), cache, ys

    # ---- exact Back-Prop-Through-Time --------------------------------------
    def backward(self, cache, target_seq):
        """Analytic gradient of the mean loss w.r.t. every parameter.

        Derivation (residual/Euler RNN). With
            h_{t+1} = h_t + dt*(W_a a_t + W_x x_t),   a_t = tanh(W_h h_t + b_h),
            y_t     = w_o . h_{t+1} + b_o,
            L       = (1/N) sum_t 0.5 (y_t - g_t)^2,
        the read-out feeds dL/dh_{t+1}; the recurrence carries gradient back
        through (I + dt * W_h^T diag(1-a^2) W_a^T). This is standard BPTT.
        """
        hs, as_, xs = cache["hs"], cache["as_"], cache["xs"]
        free_run_from = cache["free_run_from"]
        ys = cache["ys"]
        T = len(as_)
        N = T if free_run_from is None else free_run_from
        H = self.H

        gW_h = np.zeros_like(self.W_h)
        gb_h = np.zeros_like(self.b_h)
        gW_a = np.zeros_like(self.W_a)
        gW_x = np.zeros_like(self.W_x)
        gw_o = np.zeros_like(self.w_o)
        gb_o = 0.0

        grad_h_next = np.zeros(H)       # gradient flowing into h_{t+1} from t+1
        for t in reversed(range(T)):
            scored = (free_run_from is None) or (t < free_run_from)
            dy = ((ys[t] - target_seq[t]) / N) if scored else 0.0
            # total gradient w.r.t. h_{t+1}:
            gh = grad_h_next.copy()
            gh += dy * self.w_o
            # read-out params (y_t = w_o . h_{t+1} + b_o):
            gw_o += dy * hs[t + 1]
            gb_o += dy
            # h_{t+1} = h_t + dt*(W_a a_t + W_x x_t)
            gv = gh * self.dt                       # grad w.r.t. velocity v_t
            gW_a += np.outer(gv, as_[t])
            gW_x[:, 0] += gv * xs[t]
            ga = self.W_a.T @ gv                    # grad w.r.t. a_t
            gpre = ga * (1.0 - as_[t] ** 2)         # through tanh
            gW_h += np.outer(gpre, hs[t])
            gb_h += gpre
            # grad to h_t: residual path (+gh) plus through the tanh field:
            grad_h_next = gh + self.W_h.T @ gpre
        return {"W_h": gW_h, "b_h": gb_h, "W_a": gW_a,
                "W_x": gW_x, "w_o": gw_o, "b_o": np.array(gb_o)}

    # ---- the psyche measure: how much does the substance move itself? ------
    def psyche(self, h, x_t):
        """Thales' soul-meter. Returns the fraction of this step's motion that
        is *self-originated* (autonomous) rather than externally pushed.

        psyche = ||v_auto|| / (||v_auto|| + ||v_ext|| + eps),  in [0, 1].
        A lodestone (large autonomous field, no push needed) scores high; an
        inert pebble (motion only when shoved) scores low. This is exactly the
        criterion Aristotle reports Thales using on the magnet.
        """
        a = np.tanh(self.W_h @ h + self.b_h)
        v_auto = self.W_a @ a
        v_ext = self.W_x[:, 0] * x_t
        na, ne = np.linalg.norm(v_auto), np.linalg.norm(v_ext)
        return na / (na + ne + 1e-12)


# =============================================================================
# 3. GRADIENT CHECK (mandatory) — analytic BPTT vs central finite differences
# =============================================================================
def gradient_check(verbose=True):
    """Compare analytic gradients to central finite differences on a small
    model + short sequence. Returns the worst relative error over all params.
    """
    model = ArcheFlow(hidden=6, dt=0.4)
    T = 12
    sig = make_saros_signal(T + 1, dt=1.0)
    x_seq = sig[:T]
    target_seq = sig[1:T + 1]

    loss, cache, _ = model.forward(x_seq, target_seq)
    grads = model.backward(cache, target_seq)

    eps = 1e-6
    worst = 0.0
    report = []
    for name, P in model.get_params().items():
        P = np.atleast_1d(P).astype(np.float64)
        flat = P.ravel().copy()
        gflat = np.atleast_1d(grads[name]).ravel()
        num = np.zeros_like(flat)
        for i in range(flat.size):
            orig = flat[i]
            flat[i] = orig + eps
            model.set_param(name, flat.reshape(P.shape) if P.shape else flat[0])
            lp, _, _ = model.forward(x_seq, target_seq)
            flat[i] = orig - eps
            model.set_param(name, flat.reshape(P.shape) if P.shape else flat[0])
            lm, _, _ = model.forward(x_seq, target_seq)
            num[i] = (lp - lm) / (2 * eps)
            flat[i] = orig
            model.set_param(name, flat.reshape(P.shape) if P.shape else flat[0])
        denom = np.maximum(1e-8, np.abs(gflat) + np.abs(num))
        rel = np.max(np.abs(gflat - num) / denom)
        worst = max(worst, rel)
        report.append((name, rel))
    if verbose:
        for name, rel in report:
            print(f"    grad-check {name:>4s}: max rel err = {rel:.2e}")
    return worst


# =============================================================================
# 4. TRAINING — learn the cycle by SGD+momentum
# =============================================================================
def train(model, x_seq, target_seq, epochs=4000, lr=0.05, mom=0.9, clip=5.0,
          verbose=True):
    vel = {k: np.zeros_like(np.atleast_1d(v).astype(np.float64))
           for k, v in model.get_params().items()}
    history = []
    for ep in range(epochs):
        loss, cache, _ = model.forward(x_seq, target_seq)
        grads = model.backward(cache, target_seq)
        # global-norm gradient clipping (keep the flow from exploding):
        gn = np.sqrt(sum(float(np.sum(np.atleast_1d(g) ** 2))
                         for g in grads.values()))
        scale = (clip / gn) if gn > clip else 1.0
        for k in vel:
            g = np.atleast_1d(grads[k]).astype(np.float64) * scale
            vel[k] = mom * vel[k] - lr * g
            cur = np.atleast_1d(model.get_params()[k]).astype(np.float64)
            new = cur + vel[k]
            model.set_param(k, new.reshape(getattr(model, k).shape)
                            if k != "b_o" else float(new.ravel()[0]))
        history.append(loss)
        if verbose and (ep % 500 == 0 or ep == epochs - 1):
            print(f"    epoch {ep:5d}   train MSE = {loss:.6f}")
    return history


# =============================================================================
# 5. MAIN — run everything and print a verified report
# =============================================================================
def main():
    print("=" * 74)
    print("  THE ARCHE-FLOW  —  Thales of Miletus (c. 624–546 BCE)")
    print("  one substance · self-motion is soul · predict the eclipse")
    print("=" * 74)

    # ---- 5.1 gradient check (must pass) ----
    print("\n[1] Gradient check (analytic BPTT vs finite differences)")
    worst = gradient_check(verbose=True)
    print(f"    -> worst relative error = {worst:.2e}")
    assert worst < 1e-4, f"GRADIENT CHECK FAILED: {worst:.2e} >= 1e-4"
    print("    -> PASS (analytic gradients match numerics)")

    # ---- 5.2 build data: train window + a held-out future to predict ----
    total = 220
    sig = make_saros_signal(total + 1, dt=1.0)
    train_len = 170
    x_all, t_all = sig[:total], sig[1:total + 1]
    x_train, t_train = x_all[:train_len], t_all[:train_len]

    # ---- 5.3 train ----
    print("\n[2] Training the substance to internalise the Saros cycle")
    model = ArcheFlow(hidden=16, dt=0.5)
    hist = train(model, x_train, t_train, epochs=4000, lr=0.05, verbose=True)
    final = hist[-1]
    assert final < 0.02, f"TRAINING DID NOT CONVERGE: final MSE {final:.4f}"
    assert final < hist[0] * 0.2, "TRAINING DID NOT IMPROVE ENOUGH"
    print(f"    -> converged: MSE {hist[0]:.4f} -> {final:.6f}")

    # ---- 5.4 teacher-forced one-step accuracy on the held-out tail ----
    print("\n[3] One-step forecast on held-out data (teacher forced)")
    _, _, y_full = model.forward(x_all, t_all)
    tail = slice(train_len, total)
    mse_tail = float(np.mean((y_full[tail] - t_all[tail]) ** 2))
    print(f"    held-out one-step MSE = {mse_tail:.6f}")
    assert mse_tail < 0.05, f"HELD-OUT FORECAST POOR: {mse_tail:.4f}"
    print("    -> PASS (the model forecasts unseen returns)")

    # ---- 5.5 FREE-RUN eclipse prediction: cut the world off, self-move ----
    print("\n[4] Free-run prediction (external input REMOVED at t = %d)"
          % train_len)
    # warm up on the training window, then continue with NO external input:
    _, _, y_free = model.forward(x_all, t_all, free_run_from=train_len)
    horizon = slice(train_len, total)
    truth = t_all[horizon]
    pred = y_free[horizon]
    free_mse = float(np.mean((pred - truth) ** 2))
    # correlation: did the *shape* of the heavens survive the loss of data?
    corr = float(np.corrcoef(pred, truth)[0, 1])
    print(f"    free-run MSE over {total - train_len} steps = {free_mse:.6f}")
    print(f"    free-run shape correlation with truth   = {corr:+.3f}")
    # A self-moving substance should track the cycle's *shape* even unforced:
    assert corr > 0.6, f"FREE-RUN LOST THE CYCLE: corr {corr:.2f}"
    print("    -> PASS (the substance continues the eclipse without data)")

    # ---- 5.6 LODESTONE TEST: does psyche separate mover from inert body? ----
    print("\n[5] Lodestone test (Thales' soul-criterion: self-motion = soul)")
    # Take a real trajectory state from the trained model = the 'self-mover'.
    _, cache_lr, _ = model.forward(x_train, t_train)
    h_mid = cache_lr["hs"][train_len // 2]
    x_mid = x_train[train_len // 2]
    psyche_mover = model.psyche(h_mid, x_mid)

    # Build an 'inert' twin: zero its autonomous field (no self-motion left),
    # so it only ever moves when the world pushes it — a mere pebble.
    inert = ArcheFlow(hidden=model.H, dt=model.dt)
    inert.W_h, inert.b_h = model.W_h.copy(), model.b_h.copy()
    inert.W_a = np.zeros_like(model.W_a)      # soul removed
    inert.W_x = model.W_x.copy()
    psyche_inert = inert.psyche(h_mid, x_mid)

    print(f"    psyche(self-mover / lodestone) = {psyche_mover:.3f}")
    print(f"    psyche(inert pebble)           = {psyche_inert:.3f}")
    assert psyche_mover > 0.5 > psyche_inert, "PSYCHE METER FAILED TO DISCRIMINATE"
    assert abs(psyche_inert) < 1e-6, "INERT BODY SHOULD HAVE ZERO SELF-MOTION"
    print("    -> PASS (the meter finds soul exactly where motion is self-made)")

    # ---- 5.7 verified summary ----
    print("\n" + "=" * 74)
    print("  VERIFIED REPORT")
    print("-" * 74)
    print(f"  gradient check worst rel err : {worst:.2e}   (< 1e-4)")
    print(f"  train MSE                     : {final:.6f}")
    print(f"  held-out one-step MSE         : {mse_tail:.6f}")
    print(f"  free-run MSE (no input)       : {free_mse:.6f}")
    print(f"  free-run shape correlation    : {corr:+.3f}")
    print(f"  psyche  self-mover vs inert   : {psyche_mover:.3f} vs {psyche_inert:.3f}")
    print("  ALL SELF-TESTS PASSED.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

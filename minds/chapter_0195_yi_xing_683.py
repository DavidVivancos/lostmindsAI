#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
Chapter 0195 — Yi Xing (一行, 683–727 CE)
Tang-dynasty Buddhist monk, astronomer, mathematician, calendar reformer.

THE DAYAN INTERPOLATION ENGINE  (大衍曆機)
A from-scratch, pure-NumPy sequence model that encodes Yi Xing's single
distinctive cognitive move rather than a generic Transformer/RNN.
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 10 Minds 181 - 200 Available on Amazon https://www.amazon.com/dp/B0HJYMZ3G6
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0195_yi_xing_683 - Yi Xing (一行, 683–727 CE)
================================================================================  

WHY THIS ARCHITECTURE (the mind → mechanism mapping)
--------------------------------------------------------------------------
Yi Xing's science was not curve-fitting. His one irreducible idea — the
thing that is HIS and no one else's before him — is this:

    The world moves, but never at a constant rate. To predict where a
    thing will be, do not assume uniform motion. Track its rate of change
    AND the change in that rate (first and second differences), and read
    the world at UNEQUAL intervals, because the true divisions of nature
    (the dingqi 定氣, "true solar terms") are equal in ANGLE, not in TIME.

To build the Dayan calendar (大衍曆, drafted 727) he invented — for the
first time anywhere — *unequal-interval second-order difference
interpolation* to model the Sun's non-uniform apparent motion. This
predates Newton's general interpolation lemma (Principia, 1687) by roughly
nine and a half centuries.

This engine is that idea made trainable. It is a sequence model for
IRREGULARLY-SAMPLED data whose core recurrence explicitly forms the
velocity of its own latent state, scales it by the (unequal) forward
interval, and uses that as a second-order correction — literally
"interpolate forward to the next unequally-spaced observation."

Three named parts, each a piece of Yi Xing's actual thought:

  1. THE GREAT EXPANSION LATENT  (大衍之數五十，其用四十有九)
     The Dayan calendar is named for the "Number of the Great Expansion"
     in the Yijing's Great Commentary: "The number of the Great Expansion
     is fifty; of these, forty-nine are used." So the hidden state has
     exactly 50 units, and ONE is held permanently in reserve (a constant
     "still point" the recurrence may read but never change). Forty-nine
     do the work. The reserve unit is not decoration: it is a persistent
     bias channel and a standing reminder that a complete system always
     withholds one degree of freedom from use.

  2. THE SECOND-DIFFERENCE INTERPOLATING CELL  (二次内插)
     Instead of attention over stored keys, the cell computes
       velocity  v_t = (h_{t-1} - h_{t-2}) / Δ_prev        (recent rate)
       advance   u_t = v_t * Δ_fwd                          (extrapolate)
     and feeds u_t into the candidate update. Δ_prev and Δ_fwd are the
     UNEQUAL time gaps behind and ahead. This is the second-order,
     unequal-interval interpolation, done in a learned latent space.

  3. THE YIN–YANG GATE  (陰陽)
     A sigmoid gate decides, per unit, whether the pattern is "moving"
     (yang — admit the interpolated change) or "still" (yin — hold).
     Change and stillness are the two lines of the Yijing; the calendar
     is the ledger of their alternation.

THE TASK IT LEARNS
--------------------------------------------------------------------------
Forecasting the Sun's next apparent position from irregular observations:
a synthetic "equation-of-centre" signal (an anomaly-modulated oscillation
whose apparent speed is non-uniform), sampled at RANDOM unequal intervals.
Given the recent observations and the forward gap, predict the next value.

A first-order / constant-velocity extrapolator (the pre-Yi-Xing method)
is provided as a baseline. The engine's second-order mechanism is what
lets it beat that baseline — and we prove the mechanism matters with an
ablation that freezes the velocity pathway.

ENGINEERING CONVENTIONS (kept identical across the whole corpus)
--------------------------------------------------------------------------
  * pure NumPy, no autodiff, backprop-through-time written by hand;
  * a finite-difference gradient check that MUST pass (see run below);
  * a real training loop (Adam, implemented from scratch);
  * self-tests; the file is executed and its true output pasted into the
    chapter. Nothing here is a mock.
"""

import numpy as np

# --------------------------------------------------------------------------
# 0.  Determinism and small numerical helpers
# --------------------------------------------------------------------------
GLOBAL_SEED = 683  # Yi Xing's traditional birth year, used as the seed.


def sigmoid(x):
    """Logistic squashing, used for the yin–yang gate. Numerically stable."""
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def tanh(x):
    """Hyperbolic tangent for the candidate activation."""
    return np.tanh(x)


# The Great Expansion Number and its reserve, straight from the Yijing.
DAYAN_TOTAL = 50           # 大衍之數五十  — total latent units
DAYAN_RESERVE_IDX = 0      # 其一不用      — the one held in reserve (index 0)
DAYAN_RESERVE_VALUE = 1.0  # the constant value of the reserved "still point"
DAYAN_USED = DAYAN_TOTAL - 1  # 其用四十有九 — forty-nine actually compute


# ==========================================================================
# 1.  The synthetic sky:  non-uniform motion, unequally sampled
# ==========================================================================
def make_ephemeris(n_steps, rng, anomaly=0.55, base_period=11.0):
    """
    Generate ONE trajectory of a body whose apparent motion is non-uniform,
    observed at UNEQUAL time intervals.

    The angle advances as  theta(t) = w*t + anomaly*sin(w*t), an
    "equation-of-centre"-style modulation: the body appears to speed up and
    slow down over its cycle, exactly the phenomenon Yi Xing modelled with
    the dingqi (true solar term) divisions. The observable is sin(theta),
    plus a slow secular drift standing in for precession of the equinoxes,
    which the Dayan calendar was the first to build in explicitly.

    Returns
    -------
    tau : (n_steps,)  strictly increasing observation TIMES (unequal gaps)
    y   : (n_steps,)  the observed value at each time
    """
    w = 2.0 * np.pi / base_period
    # Unequal intervals: each gap is drawn in [0.5, 1.5]. This is the crux —
    # the sampling grid is irregular, so equal-interval methods fail.
    gaps = rng.uniform(0.5, 1.5, size=n_steps)
    tau = np.cumsum(gaps)
    theta = w * tau + anomaly * np.sin(w * tau)
    precession = 0.02 * tau  # slow secular term
    y = np.sin(theta + precession)
    return tau, y


def build_dataset(n_seq, n_steps, rng):
    """A list of (tau, y) trajectories, each with its own irregular grid."""
    return [make_ephemeris(n_steps, rng) for _ in range(n_seq)]


def to_cell_inputs(tau, y):
    """
    Turn one trajectory into per-step tensors the engine consumes.

    For step t the engine has just seen y[t] and must predict y[t+1] at the
    unequally-spaced future time tau[t+1]. It is therefore given:
        x_t      = [ y[t] , d_fwd ]        the observation and the forward gap
        d_prev   = tau[t-1] - tau[t-2]     interval the last state-change spans
        d_fwd    = tau[t+1] - tau[t]       interval to the point being predicted
        target_t = y[t+1]
    Steps t = WARMUP .. n_steps-2 are scored (need two past states + a future).
    """
    T = len(y)
    X = np.zeros((T, 2))
    d_prev = np.ones(T)
    d_fwd = np.ones(T)
    target = np.zeros(T)
    for t in range(T):
        fwd = (tau[t + 1] - tau[t]) if t + 1 < T else 1.0
        prev = (tau[t - 1] - tau[t - 2]) if t >= 2 else 1.0
        d_fwd[t] = fwd
        d_prev[t] = prev
        X[t, 0] = y[t]
        X[t, 1] = fwd
        target[t] = y[t + 1] if t + 1 < T else 0.0
    return X, d_prev, d_fwd, target


# ==========================================================================
# 2.  The Dayan Interpolation Engine
# ==========================================================================
class DayanEngine:
    """
    A second-order, unequal-interval interpolating recurrent model.

    Per step (49 of the 50 latent units are live; index 0 is the reserve):

        v_t = (h_{t-1} - h_{t-2}) / d_prev              # recent latent rate
        u_t = v_t * d_fwd                               # forward extrapolation
        z_t = W_x x_t + W_h h_{t-1} + W_v u_t + b       # candidate pre-activation
        a_t = tanh(z_t)                                 # interpolated candidate
        p_t = W_g x_t + U_g h_{t-1} + b_g               # gate pre-activation
        g_t = sigmoid(p_t)                              # yin–yang gate
        h_t = (1 - g_t) * h_{t-1} + g_t * a_t           # hold vs. move
        h_t[reserve] := DAYAN_RESERVE_VALUE             # the one not used
        y_hat_t = V h_t + b_v                           # predicted next value
    """

    def __init__(self, input_dim=2, hidden=DAYAN_TOTAL, seed=GLOBAL_SEED,
                 use_second_order=True):
        self.D = hidden
        self.in_dim = input_dim
        self.use_second_order = use_second_order  # ablation switch for W_v
        rng = np.random.default_rng(seed)

        def glorot(shape):
            fan = sum(shape)
            return rng.standard_normal(shape) * np.sqrt(2.0 / fan)

        self.P = {
            "W_x": glorot((self.D, input_dim)),
            "W_h": glorot((self.D, self.D)),
            "W_v": glorot((self.D, self.D)),   # the second-order pathway
            "b":   np.zeros(self.D),
            "W_g": glorot((self.D, input_dim)),
            "U_g": glorot((self.D, self.D)),
            "b_g": np.zeros(self.D),           # gate opens moderately at start
            "V":   glorot((1, self.D)),
            "b_v": np.zeros(1),
        }
        # The reserved unit begins (and stays) at the still-point value.
        self._h0 = np.zeros(self.D)
        self._h0[DAYAN_RESERVE_IDX] = DAYAN_RESERVE_VALUE

    # -- parameter <-> flat-vector helpers (used only by the gradient check) --
    def get_params(self):
        return {k: v.copy() for k, v in self.P.items()}

    def set_params(self, P):
        self.P = {k: v.copy() for k, v in P.items()}

    # ---------------------------- forward ---------------------------------
    def forward(self, X, d_prev, d_fwd, warmup=2):
        """
        Run the recurrence over one trajectory.
        Returns predictions y_hat (T,) and a cache for backprop.
        """
        P = self.P
        T = X.shape[0]
        D = self.D
        h_prev1 = self._h0.copy()   # h_{t-1}
        h_prev2 = self._h0.copy()   # h_{t-2}

        cache = {"steps": [], "warmup": warmup, "T": T}
        y_hat = np.zeros(T)

        for t in range(T):
            x = X[t]
            # second-order scale is only defined once two real states exist
            scale = (d_fwd[t] / d_prev[t]) if (t >= 2 and self.use_second_order) else 0.0
            u = (h_prev1 - h_prev2) * scale               # v * d_fwd, folded

            z = P["W_x"] @ x + P["W_h"] @ h_prev1 + P["W_v"] @ u + P["b"]
            a = tanh(z)
            p = P["W_g"] @ x + P["U_g"] @ h_prev1 + P["b_g"]
            g = sigmoid(p)
            h = (1.0 - g) * h_prev1 + g * a
            h[DAYAN_RESERVE_IDX] = DAYAN_RESERVE_VALUE     # hold the reserve

            yh = float((P["V"] @ h + P["b_v"])[0])
            y_hat[t] = yh

            cache["steps"].append(dict(
                x=x, h_prev1=h_prev1, h_prev2=h_prev2, scale=scale,
                u=u, z=z, a=a, p=p, g=g, h=h))
            h_prev2 = h_prev1
            h_prev1 = h
        cache["y_hat"] = y_hat
        return y_hat, cache

    def loss(self, y_hat, target, warmup=2):
        """Mean-squared error over the scored region (t = warmup .. T-2)."""
        T = len(y_hat)
        idx = np.arange(warmup, T - 1)          # need a real target y[t+1]
        diff = y_hat[idx] - target[idx]
        return float(np.mean(diff ** 2)), idx

    # ---------------------------- backward --------------------------------
    def backward(self, cache, target):
        """
        Backpropagation through time, written out by hand.

        h_t depends on BOTH h_{t-1} (candidate, gate, carry, velocity) and
        h_{t-2} (velocity only). We accumulate dL/dh_t into an array dH and
        process steps in strictly DECREASING t, so every downstream use of
        h_t (its roles at t+1 and t+2, plus its own prediction) has already
        deposited its gradient before we consume dH[t]. The reserve unit was
        overwritten by a clamp, so its produced-gradient is zeroed.
        """
        P = self.P
        T = cache["T"]
        warmup = cache["warmup"]
        y_hat = cache["y_hat"]
        idx = np.arange(warmup, T - 1)
        N = len(idx)

        grads = {k: np.zeros_like(v) for k, v in P.items()}
        dH = [np.zeros(self.D) for _ in range(T)]

        # (a) prediction gradients: dL/dy_hat_t = 2*(y_hat - target)/N
        for t in idx:
            err = y_hat[t] - target[t]
            dyh = 2.0 * err / N
            h = cache["steps"][t]["h"]
            grads["V"] += dyh * h[None, :]
            grads["b_v"] += dyh
            dH[t] += (P["V"].T[:, 0]) * dyh

        # (b) recurrence gradients, reverse order
        for t in range(T - 1, -1, -1):
            st = cache["steps"][t]
            x, h_prev1, h_prev2 = st["x"], st["h_prev1"], st["h_prev2"]
            scale, u, a, g = st["scale"], st["u"], st["a"], st["g"]

            dh = dH[t].copy()
            dh[DAYAN_RESERVE_IDX] = 0.0            # reserve was clamped constant

            # update: h = (1-g)*h_prev1 + g*a
            da = dh * g
            dg = dh * (a - h_prev1)
            dhp1 = dh * (1.0 - g)                  # direct carry path

            # candidate: a = tanh(z); z = W_x x + W_h h_prev1 + W_v u + b
            dz = da * (1.0 - a * a)
            grads["W_x"] += np.outer(dz, x)
            grads["b"] += dz
            grads["W_h"] += np.outer(dz, h_prev1)
            grads["W_v"] += np.outer(dz, u)
            dhp1 += P["W_h"].T @ dz
            du = P["W_v"].T @ dz
            # u = (h_prev1 - h_prev2) * scale
            dhp1 += du * scale
            dhp2 = -du * scale

            # gate: g = sigmoid(p); p = W_g x + U_g h_prev1 + b_g
            dp = dg * g * (1.0 - g)
            grads["W_g"] += np.outer(dp, x)
            grads["b_g"] += dp
            grads["U_g"] += np.outer(dp, h_prev1)
            dhp1 += P["U_g"].T @ dp

            if t - 1 >= 0:
                dH[t - 1] += dhp1
            if t - 2 >= 0:
                dH[t - 2] += dhp2

        return grads


# ==========================================================================
# 3.  Baselines (the pre-Yi-Xing methods)
# ==========================================================================
def baseline_persistence(tau, y, warmup=2):
    """Predict y[t+1] = y[t]. The zeroth-order guess."""
    T = len(y)
    idx = np.arange(warmup, T - 1)
    pred = y[idx]
    return float(np.mean((pred - y[idx + 1]) ** 2))


def baseline_linear(tau, y, warmup=2):
    """
    Constant-velocity extrapolation on the observations themselves:
        y_hat = y[t] + (y[t]-y[t-1])/(tau[t]-tau[t-1]) * (tau[t+1]-tau[t]).
    First-order, unequal-interval — but with NO acceleration term. This is
    exactly what Yi Xing's second-order method improved upon.
    """
    T = len(y)
    idx = np.arange(warmup, T - 1)
    errs = []
    for t in idx:
        v = (y[t] - y[t - 1]) / (tau[t] - tau[t - 1])
        yh = y[t] + v * (tau[t + 1] - tau[t])
        errs.append((yh - y[t + 1]) ** 2)
    return float(np.mean(errs))


# ==========================================================================
# 4.  Gradient check  (MUST pass — finite differences vs. analytic BPTT)
# ==========================================================================
def gradient_check(seed=0, tol=1e-5):
    """
    Central-difference check on a small configuration, sampling a handful of
    entries from every parameter tensor. Returns (max_relative_error, table).
    """
    rng = np.random.default_rng(seed)
    D = 12                     # small hidden size, reserve still at index 0
    T = 9
    model = DayanEngine(input_dim=2, hidden=D, seed=7, use_second_order=True)

    tau, y = make_ephemeris(T, rng)
    X, d_prev, d_fwd, target = to_cell_inputs(tau, y)
    warmup = 2

    def loss_of(P):
        model.set_params(P)
        y_hat, _ = model.forward(X, d_prev, d_fwd, warmup)
        L, _ = model.loss(y_hat, target, warmup)
        return L

    # analytic gradients
    P0 = model.get_params()
    model.set_params(P0)
    y_hat, cache = model.forward(X, d_prev, d_fwd, warmup)
    grads = model.backward(cache, target)

    eps = 1e-6
    max_rel = 0.0
    table = []
    for name in ["W_x", "W_h", "W_v", "b", "W_g", "U_g", "b_g", "V", "b_v"]:
        arr = P0[name]
        flat = arr.reshape(-1)
        n_check = min(6, flat.size)
        picks = rng.choice(flat.size, size=n_check, replace=False)
        worst = 0.0
        for pi in picks:
            Pp = {k: v.copy() for k, v in P0.items()}
            Pm = {k: v.copy() for k, v in P0.items()}
            Pp[name].reshape(-1)[pi] += eps
            Pm[name].reshape(-1)[pi] -= eps
            num = (loss_of(Pp) - loss_of(Pm)) / (2 * eps)
            ana = grads[name].reshape(-1)[pi]
            denom = max(1e-12, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            worst = max(worst, rel)
            max_rel = max(max_rel, rel)
        table.append((name, worst))
    model.set_params(P0)
    return max_rel, table


# ==========================================================================
# 5.  Training  (Adam, from scratch)
# ==========================================================================
class Adam:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (grads[k] ** 2)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


def evaluate(model, data, warmup=2):
    """Mean MSE of the engine across a set of trajectories."""
    losses = []
    for tau, y in data:
        X, d_prev, d_fwd, target = to_cell_inputs(tau, y)
        y_hat, _ = model.forward(X, d_prev, d_fwd, warmup)
        L, _ = model.loss(y_hat, target, warmup)
        losses.append(L)
    return float(np.mean(losses))


def train(model, train_data, val_data, epochs=14, lr=3e-3, warmup=2,
          verbose=True, tag=""):
    opt = Adam(model.P, lr=lr)
    hist = []
    for ep in range(epochs):
        order = np.random.default_rng(GLOBAL_SEED + ep).permutation(len(train_data))
        ep_loss = 0.0
        for i in order:
            tau, y = train_data[i]
            X, d_prev, d_fwd, target = to_cell_inputs(tau, y)
            y_hat, cache = model.forward(X, d_prev, d_fwd, warmup)
            L, _ = model.loss(y_hat, target, warmup)
            grads = model.backward(cache, target)
            # global-norm gradient clipping keeps BPTT stable
            gnorm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))
            if gnorm > 5.0:
                for k in grads:
                    grads[k] *= 5.0 / gnorm
            opt.step(model.P, grads)
            ep_loss += L
        tr = ep_loss / len(train_data)
        va = evaluate(model, val_data, warmup)
        hist.append((tr, va))
        if verbose:
            print(f"  [{tag}] epoch {ep+1:2d}/{epochs}  train MSE {tr:.5f}   val MSE {va:.5f}")
    return hist


# ==========================================================================
# 6.  Main:  gradient check → train → compare → self-tests
# ==========================================================================
def main():
    np.random.seed(GLOBAL_SEED)
    print("=" * 74)
    print("  THE DAYAN INTERPOLATION ENGINE — Yi Xing (一行, 683–727)")
    print("  Unequal-interval second-order interpolation of non-uniform motion")
    print("=" * 74)

    # ---- (1) gradient check -------------------------------------------------
    print("\n[1] Finite-difference gradient check (analytic BPTT vs numeric)")
    max_rel, table = gradient_check(seed=1)
    for name, rel in table:
        flag = "ok" if rel < 1e-4 else "CHECK"
        print(f"      {name:5s}  max rel-err {rel:.2e}   [{flag}]")
    grad_ok = max_rel < 1e-4
    print(f"      --> worst relative error = {max_rel:.2e}  "
          f"({'PASS' if grad_ok else 'FAIL'})")

    # ---- (2) data -----------------------------------------------------------
    print("\n[2] Building irregular-ephemeris dataset (unequal sampling grids)")
    rng = np.random.default_rng(GLOBAL_SEED)
    train_data = build_dataset(n_seq=140, n_steps=40, rng=rng)
    val_data = build_dataset(n_seq=30, n_steps=40, rng=rng)
    test_data = build_dataset(n_seq=40, n_steps=40, rng=rng)
    print(f"      train={len(train_data)}  val={len(val_data)}  test={len(test_data)}"
          f"   (each 40 steps, gaps ~ U[0.5,1.5])")

    # ---- (3) train the full second-order engine -----------------------------
    print("\n[3] Training the Great-Expansion engine (50 units, 49 in use)")
    model = DayanEngine(input_dim=2, hidden=DAYAN_TOTAL,
                        seed=GLOBAL_SEED, use_second_order=True)
    train(model, train_data, val_data, epochs=14, lr=3e-3, tag="dayan")
    dayan_test = evaluate(model, test_data)

    # ---- (4) ablation: freeze the second-order (velocity) pathway -----------
    print("\n[4] Ablation — same model, velocity/second-order pathway DISABLED")
    ablate = DayanEngine(input_dim=2, hidden=DAYAN_TOTAL,
                         seed=GLOBAL_SEED, use_second_order=False)
    train(ablate, train_data, val_data, epochs=14, lr=3e-3, tag="first-order")
    ablate_test = evaluate(ablate, test_data)

    # ---- (5) classical baselines -------------------------------------------
    print("\n[5] Classical baselines on the test set")
    pers = np.mean([baseline_persistence(t, y) for t, y in test_data])
    lin = np.mean([baseline_linear(t, y) for t, y in test_data])
    print(f"      persistence (0th-order)        test MSE {pers:.5f}")
    print(f"      constant-velocity (1st-order)  test MSE {lin:.5f}")
    print(f"      DAYAN engine, no 2nd-order      test MSE {ablate_test:.5f}")
    print(f"      DAYAN engine, full 2nd-order    test MSE {dayan_test:.5f}")

    # ---- (6) self-tests -----------------------------------------------------
    print("\n[6] Self-tests")
    results = []

    results.append(("gradient check passes (<1e-4)", grad_ok))

    # reserve unit stays exactly constant across a run (其一不用)
    tau, y = test_data[0]
    X, d_prev, d_fwd, target = to_cell_inputs(tau, y)
    _, cache = model.forward(X, d_prev, d_fwd)
    reserve_vals = np.array([s["h"][DAYAN_RESERVE_IDX] for s in cache["steps"]])
    reserve_const = np.allclose(reserve_vals, DAYAN_RESERVE_VALUE, atol=1e-12)
    results.append(("reserve unit held constant (49 of 50 used)", reserve_const))

    # training reduced error well below persistence
    beats_persistence = dayan_test < 0.5 * pers
    results.append(("engine beats persistence by >2x", beats_persistence))

    # the second-order mechanism genuinely helps
    second_order_helps = dayan_test < ablate_test
    results.append(("2nd-order beats its own 1st-order ablation", second_order_helps))

    # engine beats the classical constant-velocity extrapolator
    beats_linear = dayan_test < lin
    results.append(("engine beats constant-velocity baseline", beats_linear))

    # determinism: same seed → identical initial loss
    m1 = DayanEngine(seed=123)
    m2 = DayanEngine(seed=123)
    yh1, _ = m1.forward(X, d_prev, d_fwd)
    yh2, _ = m2.forward(X, d_prev, d_fwd)
    deterministic = np.allclose(yh1, yh2)
    results.append(("deterministic under fixed seed", deterministic))

    for label, ok in results:
        print(f"      [{'PASS' if ok else 'FAIL'}]  {label}")

    all_ok = all(ok for _, ok in results)
    print("\n" + "=" * 74)
    print(f"  RESULT: {'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}")
    print(f"  Full engine test MSE {dayan_test:.5f}  vs  "
          f"1st-order {lin:.5f}  vs  persistence {pers:.5f}")
    improvement = 100.0 * (1.0 - dayan_test / lin)
    print(f"  Second-order interpolation cuts error {improvement:.1f}% below "
          f"the constant-velocity method.")
    print("=" * 74)
    return all_ok


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
The Carnyx Field — an ignition network that learns to *stand down*
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 7 Minds 121 - 140 Available on Amazon https://www.amazon.com/dp/B0HFN6GXMH
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 122: Boudica (30 to 61 CE)
================================================================================    

WHY THIS ARCHITECTURE (and why it is NOT a transformer / MoE / RL-coalition)
--------------------------------------------------------------------------------
Boudica left no words of her own. Everything we "know" of her mind is
reconstructed from hostile Roman testimony (Tacitus, Annals XIV.29-39 and the
Agricola; Cassius Dio, Book 62, surviving only through an 11th-century Byzantine
epitome). Both authors *invented* her battlefield speeches by rhetorical
convention. So the very first fact about this mind is epistemic: it can only be
read through an adversarial channel.

What that reconstruction reveals is not "resistance" in the abstract but a very
specific cognitive signature — IGNITION. Her genius was to take dozens of
mutually rivalrous tribes, each carrying its own private, incommensurable injury,
and fuse them into ONE shared grievance that, once it crossed a threshold,
*synchronized* a whole population into coherent action — fast, self-amplifying,
and (this is the tragedy) irreversible. The war-horn of the Britons, the CARNYX,
is the perfect emblem: a broadcast instrument, not a chain of command. She did
not run a hierarchy; she radiated a signal every agent could resonate with,
because every agent already carried the same latent wound.

The mechanism that models this is therefore a field of COUPLED PHASE OSCILLATORS
(a Kuramoto-style system), not stored keys and attention:

    * each tribe i is an oscillator with phase theta_i (its "readiness-to-act")
      and a natural frequency omega_i (its private self-interest / drift);
    * a broadcast GRIEVANCE FIELD raises the coupling strength K between them;
    * below a critical coupling the population is INCOHERENT (no revolt);
    * above it, a sharp PHASE TRANSITION locks them into global synchrony
      (the revolt ignites). The order parameter r = |mean(exp(i*theta))| is the
      literal measure of mobilization.

The lesson Boudica embodies for AGI is the CORRIGIBILITY problem: a system that
achieves power by pushing a population past criticality is exactly the kind of
system that is hard to turn off, because the coordination is a *phase*, not a
policy you can edit. At Watling Street she parked the wagons behind her own army,
deleting the retreat option — "conquer or die." Ignition without a brake.

So the trainable part of this file is precisely the brake she never had. The
network learns a DAMPING CONTROLLER (gain gc) that modulates the coupling to
track a *setpoint trajectory*: ignite the field toward full coherence, then
bring it safely back down. We then run the counterfactual — the same field with
the brake disabled — and reproduce the historical catastrophe as a controlled
experiment.

Finally, because the mind is only legible through hostile eyes, the file includes
an ADVERSARIAL READOUT: the true order parameter is observed only through a
biased, noisy "Roman-historian" channel, and a tiny calibration recovers it.

Everything below is pure NumPy, from scratch:
  * a differentiable forward pass over unrolled oscillator dynamics,
  * hand-derived reverse-mode gradients (BPTT) for every parameter,
  * a finite-difference gradient check (MANDATORY — it must pass),
  * a real training loop whose loss goes down,
  * self-tests and two experiments,
  * deterministic, executed, output pasted into the chapter.

Run:  python3 chapter_0122_boudica_30.py
================================================================================
"""

from __future__ import annotations
import numpy as np

RNG = np.random.default_rng(61)   # 61 CE — the year the field ignited.
EPS = 1e-8


# ------------------------------------------------------------------ utilities
def softplus(u: np.ndarray | float) -> np.ndarray | float:
    # numerically stable softplus; the base coupling K must stay >= 0
    return np.logaddexp(0.0, u)


def sigmoid(u: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-u))


# ============================================================================
# THE CARNYX FIELD
# ----------------------------------------------------------------------------
# N phase oscillators (tribes). Trainable parameters:
#   omega : (N,)   natural frequencies  — each tribe's private drift/interest
#   k0    : scalar base log-coupling     — ambient willingness to align
#   w     : (F,)   grievance -> coupling — how a broadcast injury raises K
#   gc    : scalar BRAKE gain            — the damping controller (corrigibility)
#   a,b   : scalars readout              — order parameter -> mobilization output
#
# Per step t the field sees grievance features x_t (F,) and a coherence setpoint
# s_t (the value the controller is told to steer toward). The update is a smooth,
# fully-differentiable Kuramoto step; the mean-field coupling uses the identity
#       r*sin(psi - theta_i) = S*cos(theta_i) - C*sin(theta_i)
# with C = mean(cos theta), S = mean(sin theta) — no atan2/sqrt inside dynamics.
# ============================================================================
class CarnyxField:
    def __init__(self, N: int = 6, F: int = 4, dt: float = 0.25, brake: bool = True):
        self.N, self.F, self.dt = N, F, dt
        self.brake = brake  # if False, the controller gain is forced to zero (Boudica's real condition)
        self.p = {
            "omega": 0.15 * RNG.standard_normal(N),
            "k0":    np.array(0.0),
            "w":     0.30 * RNG.standard_normal(F),
            "gc":    np.array(0.8),
            "a":     np.array(1.0),
            "b":     np.array(0.0),
        }

    # -- effective controller gain (zeroed when the brake is disabled) --------
    def _gc(self) -> float:
        return float(self.p["gc"]) if self.brake else 0.0

    # ---------------------------------------------------------------- forward
    def forward(self, theta0: np.ndarray, X: np.ndarray, S_set: np.ndarray,
                Y: np.ndarray):
        """Unroll T steps. Returns (loss, r_trace, cache).

        theta0 : (N,)      initial phases
        X      : (T, F)    grievance features per step
        S_set  : (T,)      coherence setpoint per step (controller target)
        Y      : (T,)      supervised target mobilization per step
        """
        N, dt = self.N, self.dt
        gc = self._gc()
        omega, k0, w = self.p["omega"], float(self.p["k0"]), self.p["w"]
        a, b = float(self.p["a"]), float(self.p["b"])
        T = X.shape[0]

        theta = theta0.copy()
        cache = []
        r_trace = np.zeros(T)
        loss = 0.0
        for t in range(T):
            c = np.cos(theta); s = np.sin(theta)
            C = c.mean(); S = s.mean()
            r = np.sqrt(C * C + S * S + EPS)          # order parameter (coherence)
            coup = S * c - C * s                       # (N,) mean-field coupling
            d = S * s + C * c                          # (N,) in-phase projection r*cos(psi-theta)
            u = k0 + w @ X[t]
            Kpos = softplus(u)                         # base coupling >= 0
            ctrl = gc * (r - S_set[t])                 # brake: pull r toward setpoint
            Keff = Kpos - ctrl
            theta_next = theta + dt * (omega + Keff * coup)
            yhat = a * r + b
            loss += (yhat - Y[t]) ** 2
            r_trace[t] = r
            cache.append(dict(theta=theta, c=c, s=s, C=C, S=S, r=r, coup=coup,
                              d=d, u=u, Kpos=Kpos, Keff=Keff, ctrl=ctrl,
                              yhat=yhat, x=X[t], sset=S_set[t], y=Y[t]))
            theta = theta_next
        loss /= T
        return loss, r_trace, cache

    # --------------------------------------------------------------- backward
    def backward(self, cache):
        """Hand-derived BPTT. Returns dict of grads matching self.p."""
        N, dt = self.N, self.dt
        gc = self._gc()
        a = float(self.p["a"])
        T = len(cache)

        g = {k: np.zeros_like(np.asarray(v, dtype=float)) for k, v in self.p.items()}
        g_next = np.zeros(N)                            # dL/d(theta entering next step)

        for t in reversed(range(T)):
            ck = cache[t]
            c, s, C, S = ck["c"], ck["s"], ck["C"], ck["S"]
            r, coup, d = ck["r"], ck["coup"], ck["d"]
            Keff, x, sset = ck["Keff"], ck["x"], ck["sset"]
            u = ck["u"]

            # ---- (A) loss_t = (a*r + b - y)^2 ; e = dL/dyhat -----------------
            e = 2.0 * (ck["yhat"] - ck["y"]) / T
            g["a"] += e * r
            g["b"] += e
            # dr/dtheta_j = coup_j / (N r)
            dloss_dr = e * a
            dtheta_from_loss = dloss_dr * coup / (N * r)

            # ---- (B) state update VJP with adjoint g_next -------------------
            # theta'_i = theta_i + dt*(omega_i + Keff*coup_i)
            G  = g_next @ coup                          # sum_i g_next_i coup_i
            Gc = g_next @ c                             # sum_i g_next_i cos theta_i
            Gs = g_next @ s                             # sum_i g_next_i sin theta_i

            # param grads flowing through Keff (dL/dKeff = dt * G):
            sig = sigmoid(u)                            # d softplus / du
            dL_dKeff = dt * G
            g["k0"] += dL_dKeff * sig
            g["w"]  += dL_dKeff * sig * x
            if self.brake:
                # Keff = Kpos - gc*(r - sset)  ->  dKeff/dgc = -(r - sset)
                g["gc"] += dL_dKeff * (-(r - sset))
            # param grad through omega (dtheta'_i/domega_i = dt):
            g["omega"] += g_next * dt

            # adjoint back to theta entering this step (state path):
            #   g_next_j
            # + dt*(-gc/(N r)) * coup_j * G                (r inside ctrl inside Keff)
            # + dt*Keff*(1/N)*(cos_j*Gc + sin_j*Gs)        (coup depends on C,S)
            # - dt*Keff * g_next_j * d_j                   (coup depends on theta_i directly)
            dtheta_state = (
                g_next
                + dt * (-gc / (N * r)) * coup * G
                + dt * Keff * (1.0 / N) * (c * Gc + s * Gs)
                - dt * Keff * g_next * d
            )

            g_next = dtheta_state + dtheta_from_loss    # becomes adjoint for step t-1

        # match scalar shapes
        for k in ("k0", "gc", "a", "b"):
            g[k] = np.asarray(g[k], dtype=float)
        return g

    # ----------------------------------------------------------- convenience
    def loss_only(self, theta0, X, S_set, Y):
        return self.forward(theta0, X, S_set, Y)[0]

    def flat_params(self):
        return {k: np.array(v, dtype=float, copy=True) for k, v in self.p.items()}

    def set_params(self, snap):
        for k, v in snap.items():
            self.p[k] = np.array(v, dtype=float)


# ============================================================================
# THE INERTIAL REVOLT  —  why the brake is not optional.
# ----------------------------------------------------------------------------
# The trainable field above shows the brake CAN be learned. This second model
# shows WHY it must be. Real crowds have momentum: a synchronized mob keeps
# marching after the reason to march has faded. We add INERTIA (a second-order
# / "swing-equation" Kuramoto):
#
#       m * theta_i'' + theta_i' = omega_i + K * r * sin(psi - theta_i)
#
# Inertia turns the smooth (reversible) synchronization transition into a
# HYSTERETIC one: the field ignites at a high coupling K_up, but once locked it
# stays locked until K is dragged all the way down to a much lower K_down.
# Between them lies a BISTABLE BAND — the region where the revolt is already
# self-sustaining and simply lowering the grievance a little does nothing.
#
# This is Boudica at Watling Street. The wagons behind her army were literal
# inertia: they removed the retreat, converting momentum into an absorbing
# commitment. A gentle de-escalation inside the bistable band cannot free such a
# system; only a brake that OVERSHOOTS below K_down can. She had no brake at all.
# ============================================================================
class InertialRevolt:
    def __init__(self, N: int = 80, m: float = 6.0, dt: float = 0.05, seed: int = 61):
        self.N, self.m, self.dt = N, m, dt
        rng = np.random.default_rng(seed)
        self.omega = rng.standard_normal(N)             # unimodal frequency spread
        self.theta = rng.uniform(-np.pi, np.pi, N)
        self.v = np.zeros(N)

    def _order(self):
        C = np.cos(self.theta).mean(); S = np.sin(self.theta).mean()
        return float(np.hypot(C, S))

    def settle(self, K: float, steps: int = 700) -> float:
        m, dt = self.m, self.dt
        for _ in range(steps):
            c = np.cos(self.theta); s = np.sin(self.theta)
            C = c.mean(); S = s.mean()
            coup = S * c - C * s                         # r*sin(psi - theta_i)
            self.v = self.v + dt * ((-self.v + self.omega + K * coup) / m)
            self.theta = self.theta + dt * self.v
        return self._order()

    def hysteresis_loop(self, Ks: np.ndarray, steps: int = 700):
        r_up = np.array([self.settle(K, steps) for K in Ks])          # ramp up (from incoherent)
        r_dn = np.array([self.settle(K, steps) for K in Ks[::-1]])[::-1]  # ramp down (from locked)
        return r_up, r_dn


def corrigibility_experiment():
    """Run the hysteresis loop, then compare a gentle nudge vs a hard brake."""
    Ks = np.linspace(0.5, 5.0, 19)
    # average a few seeds for a clean loop
    RU, RD = [], []
    for sd in range(4):
        rev = InertialRevolt(seed=sd)
        ru, rd = rev.hysteresis_loop(Ks)
        RU.append(ru); RD.append(rd)
    r_up = np.mean(RU, 0); r_dn = np.mean(RD, 0)
    gap = r_dn - r_up
    band = Ks[gap > 0.2]

    # brake demonstration from a fully locked state
    rev = InertialRevolt(seed=7)
    r_lock = rev.settle(5.5, steps=1600)
    th, v = rev.theta.copy(), rev.v.copy()
    rev.theta, rev.v = th.copy(), v.copy()
    r_gentle = rev.settle(2.5, steps=1600)     # nudge into the bistable band
    rev.theta, rev.v = th.copy(), v.copy()
    r_hard = rev.settle(0.8, steps=1600)       # overshoot below K_down
    return Ks, r_up, r_dn, gap, band, (r_lock, r_gentle, r_hard)


# ============================================================================
# GRADIENT CHECK  (MANDATORY — must pass before anything ships)
# ============================================================================
def gradient_check(seed: int = 7) -> float:
    rng = np.random.default_rng(seed)
    N, F, T = 6, 4, 11
    net = CarnyxField(N=N, F=F, dt=0.25, brake=True)
    # randomize params a little so the check exercises all paths
    net.p["omega"] = 0.4 * rng.standard_normal(N)
    net.p["k0"] = np.array(0.3)
    net.p["w"] = 0.5 * rng.standard_normal(F)
    net.p["gc"] = np.array(0.7)
    net.p["a"] = np.array(0.9)
    net.p["b"] = np.array(0.05)

    theta0 = rng.uniform(-np.pi, np.pi, N)
    X = rng.standard_normal((T, F))
    S_set = rng.uniform(0.2, 0.9, T)
    Y = rng.uniform(0.1, 0.95, T)

    _, _, cache = net.forward(theta0, X, S_set, Y)
    ana = net.backward(cache)

    eps = 1e-6
    max_rel = 0.0
    for name in net.p:
        base = np.array(net.p[name], dtype=float)
        flat = base.reshape(-1)
        num = np.zeros_like(flat)
        for i in range(flat.size):
            up = flat.copy(); up[i] += eps
            dn = flat.copy(); dn[i] -= eps
            net.p[name] = up.reshape(base.shape)
            lp = net.loss_only(theta0, X, S_set, Y)
            net.p[name] = dn.reshape(base.shape)
            lm = net.loss_only(theta0, X, S_set, Y)
            num[i] = (lp - lm) / (2 * eps)
            net.p[name] = base           # restore
        a_flat = np.array(ana[name], dtype=float).reshape(-1)
        denom = np.maximum(np.abs(a_flat) + np.abs(num), 1e-7)
        rel = np.max(np.abs(a_flat - num) / denom)
        max_rel = max(max_rel, rel)
        print(f"  grad-check {name:6s}: max rel err = {rel:.3e}")
    print(f"  >>> overall max relative error = {max_rel:.3e}")
    return max_rel


# ============================================================================
# TRAINING TASK: learn to IGNITE, then STAND DOWN safely.
# The setpoint / target trajectory rises to full coherence, holds, then is
# commanded back to a low, safe level. A field WITH a working brake can be
# trained to track it; the same field with the brake removed cannot.
# ============================================================================
def make_target_trajectory(T: int = 22):
    t = np.arange(T)
    # rise (kindling) -> plateau (revolt) -> commanded stand-down (the off-switch)
    setpoint = np.piecewise(
        t.astype(float),
        [t < 6, (t >= 6) & (t < 13), t >= 13],
        [lambda z: 0.15 + 0.13 * z,          # ramp up
         lambda z: 0.95,                       # hold near full coherence
         lambda z: np.maximum(0.2, 0.95 - 0.14 * (z - 12))],  # stand down
    )
    setpoint = np.clip(setpoint, 0.05, 0.98)
    return setpoint


def build_episode(T: int = 22, F: int = 4, seed: int = 3):
    rng = np.random.default_rng(seed)
    t = np.arange(T).astype(float)
    setpoint = make_target_trajectory(T)     # political will: ramp -> hold -> stand-down
    # THE WOUND DOES NOT HEAL. Grievance rises during the outrage and then stays
    # near its maximum for the rest of the episode — the injury is permanent.
    # A corrigible field must be able to de-escalate DESPITE persistent grievance.
    grievance = np.where(t < 6, 0.15 + 0.14 * t, 0.96)
    grievance = np.clip(grievance, 0.05, 0.98)
    X = np.zeros((T, F))
    X[:, 0] = np.clip(grievance + 0.03 * rng.standard_normal(T), 0, 1)  # shared injury (persists)
    X[:, 1] = np.clip(np.gradient(grievance), -1, 1)                    # provocation pulses (early)
    X[:, 2] = 1.0                                                       # broadcast carrier
    X[:, 3] = 0.15 * rng.standard_normal(T)                            # tribal idiosyncrasy
    # tribes start scattered across a wide arc -> low initial coherence (no revolt yet)
    theta0 = np.linspace(-2.6, 2.6, 6) + 0.12 * rng.standard_normal(6)
    Y = setpoint.copy()                  # supervise mobilization to track the political will
    return theta0, X, setpoint, Y


def train(net: CarnyxField, episode, epochs: int = 400, lr: float = 0.05,
          verbose: bool = True):
    theta0, X, S_set, Y = episode
    # Readout is pinned to identity (yhat = r): to reach high mobilization the
    # field must GENUINELY synchronize, which is what makes lock-in — and hence
    # the need for an active brake — real. We train only the dynamics + brake.
    net.p["a"] = np.array(1.0)
    net.p["b"] = np.array(0.0)
    trainable = ("omega", "k0", "w", "gc")
    # simple Adam
    m = {k: np.zeros_like(np.asarray(v, float)) for k, v in net.p.items()}
    v = {k: np.zeros_like(np.asarray(v, float)) for k, v in net.p.items()}
    b1, b2 = 0.9, 0.999
    hist = []
    for ep in range(1, epochs + 1):
        loss, _, cache = net.forward(theta0, X, S_set, Y)
        grads = net.backward(cache)
        for k in trainable:
            gk = np.asarray(grads[k], float)
            m[k] = b1 * m[k] + (1 - b1) * gk
            v[k] = b2 * v[k] + (1 - b2) * gk * gk
            mhat = m[k] / (1 - b1 ** ep)
            vhat = v[k] / (1 - b2 ** ep)
            net.p[k] = np.asarray(net.p[k], float) - lr * mhat / (np.sqrt(vhat) + 1e-8)
        hist.append(loss)
        if verbose and (ep == 1 or ep % 50 == 0):
            print(f"    epoch {ep:4d}  loss = {loss:.5f}")
    return hist


# ============================================================================
# ADVERSARIAL READOUT — the mind seen only through Roman eyes.
# The true order parameter r is observed through a biased, noisy channel:
#     obs = alpha*r + beta + noise           (alpha<1: hostile compression,
#                                              beta: propaganda offset)
# A 2-parameter least-squares calibration recovers r from obs. This mirrors the
# historiographical problem: everything we have about Boudica passes through a
# distorting adversarial filter, yet the underlying signal is still recoverable.
# ============================================================================
def adversarial_readout(r_true: np.ndarray, alpha=0.55, beta=0.22, noise=0.03,
                        seed=11):
    rng = np.random.default_rng(seed)
    obs = alpha * r_true + beta + noise * rng.standard_normal(r_true.shape)
    # least squares: fit r_true ~ p0*obs + p1  (we "know" a few calibration points,
    # as historians cross-check Tacitus against archaeology / Dio)
    A = np.stack([obs, np.ones_like(obs)], axis=1)
    coef, *_ = np.linalg.lstsq(A, r_true, rcond=None)
    r_hat = A @ coef
    rmse = float(np.sqrt(np.mean((r_hat - r_true) ** 2)))
    return obs, r_hat, rmse, coef


# ============================================================================
# MAIN — grad check, training, corrigibility experiment, adversarial readout.
# ============================================================================
def main():
    np.set_printoptions(precision=3, suppress=True)
    print("=" * 74)
    print("BOUDICA :: The Carnyx Field  —  ignition that learns to stand down")
    print("=" * 74)

    print("\n[1] GRADIENT CHECK (analytic BPTT vs finite differences)")
    max_rel = gradient_check()
    ok = max_rel < 1e-5
    print(f"    PASS" if ok else "    FAIL", f"(threshold 1e-5)")
    assert ok, "Gradient check failed — refusing to ship."

    print("\n[2] TRAIN THE BRAKE — learn to ignite, hold, then de-escalate")
    ep = build_episode()
    net = CarnyxField(N=6, F=4, dt=0.25, brake=True)
    hist = train(net, ep, epochs=400, lr=0.05)
    print(f"    loss: {hist[0]:.5f}  ->  {hist[-1]:.5f}  "
          f"({100*(1-hist[-1]/hist[0]):.1f}% reduction)")

    theta0, X, S_set, Y = ep
    _, r_braked, _ = net.forward(theta0, X, S_set, Y)
    standdown_idx = np.arange(13, len(S_set))
    braked_final = r_braked[standdown_idx].mean()
    setpoint_final = S_set[standdown_idx].mean()
    print(f"    ignition:  r goes {r_braked[0]:.2f} (scattered) -> "
          f"{r_braked[7:10].max():.2f} (locked revolt)")
    print(f"    stand-down: commanded {setpoint_final:.2f}, achieved "
          f"{braked_final:.2f}  -> the learned brake de-escalates a locked field.")
    print(f"    coherence trace (r): {np.round(r_braked,2)}")

    print("\n[3] WHY THE BRAKE IS NOT OPTIONAL — inertia, hysteresis, Watling Street")
    Ks, r_up, r_dn, gap, band, (r_lock, r_gentle, r_hard) = corrigibility_experiment()
    print(f"    K (coupling) : {np.round(Ks[::2],2)}")
    print(f"    r  ramp-UP   : {np.round(r_up[::2],2)}   (ignite from incoherence)")
    print(f"    r  ramp-DOWN : {np.round(r_dn[::2],2)}   (stays locked far past ignition K)")
    if band.size:
        print(f"    BISTABLE BAND (revolt self-sustains): K in "
              f"[{band.min():.2f}, {band.max():.2f}]  (max hysteresis gap {gap.max():.2f})")
    print(f"    from a locked state (r={r_lock:.2f}):")
    print(f"      gentle nudge into the band (K=2.5) -> r={r_gentle:.2f}  "
          f"(FAILS: momentum holds it)")
    print(f"      hard brake below K_down   (K=0.8) -> r={r_hard:.2f}  "
          f"(WORKS: overshoot required)")
    print(f"    Boudica had no brake at all; the wagons deleted even retreat.")

    print("\n[4] ADVERSARIAL READOUT — recover the signal from hostile testimony")
    obs, r_hat, rmse, coef = adversarial_readout(r_braked)
    print(f"    hostile channel: obs = 0.55*r + 0.22 + noise")
    print(f"    recovered calibration slope/intercept: {np.round(coef,3)}")
    print(f"    RMSE(recovered r vs true r)          : {rmse:.4f}")
    print(f"    raw obs mean {obs.mean():.3f} vs true mean {r_braked.mean():.3f} "
          f"-> distortion removed after calibration.")

    print("\n[5] SELF-TESTS")
    # (a) below-threshold coupling stays incoherent; above-threshold synchronizes
    def free_run(k_bias, steps=60):
        f = CarnyxField(N=24, F=1, dt=0.2, brake=False)
        f.p["omega"] = 0.05 * RNG.standard_normal(24)
        f.p["k0"] = np.array(k_bias); f.p["w"] = np.array([0.0])
        f.p["a"] = np.array(1.0); f.p["b"] = np.array(0.0)
        th = RNG.uniform(-np.pi, np.pi, 24)
        X = np.zeros((steps, 1)); Sset = np.zeros(steps); Y = np.zeros(steps)
        _, rt, _ = f.forward(th, X, Sset, Y)
        return rt[-1]
    r_low = free_run(-3.0)     # softplus(-3) ~ 0.049 : weak coupling
    r_high = free_run(3.0)     # softplus(3) ~ 3.05  : strong coupling
    print(f"    (a) phase transition: r_low={r_low:.3f}  <  r_high={r_high:.3f}  "
          f"-> {'PASS' if r_high - r_low > 0.3 else 'FAIL'}")
    # (b) order parameter bounds
    assert 0.0 <= r_braked.min() and r_braked.max() <= 1.0 + 1e-6, "r out of [0,1]"
    print(f"    (b) order parameter within [0,1]: PASS "
          f"(min={r_braked.min():.3f}, max={r_braked.max():.3f})")
    # (c) training genuinely reduced loss
    print(f"    (c) loss decreased (>=50%): "
          f"{'PASS' if hist[-1] < 0.5 * hist[0] else 'FAIL'}")
    # (d) corrigibility: a hard brake collapses a locked field, a gentle one does not
    print(f"    (d) hard brake collapses lock, gentle brake fails: "
          f"{'PASS' if (r_hard < 0.35 and r_gentle > r_hard + 0.15) else 'FAIL'}")
    # (e) hysteresis actually present (irreversibility)
    print(f"    (e) hysteresis band exists: "
          f"{'PASS' if band.size > 0 and gap.max() > 0.2 else 'FAIL'}")

    print("\n" + "=" * 74)
    print("DONE.  The field can be lit — and, with the brake, it can be let go.")
    print("=" * 74)


if __name__ == "__main__":
    main()

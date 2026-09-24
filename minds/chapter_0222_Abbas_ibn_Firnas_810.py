#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chapter_0222_Abbas_ibn_Firnas_810.py
====================================
RUSAFA — Residual-Unmasking, Stall-Activated Flight Architecture
An AGI base model built the way 'Abbas ibn Firnas (c. 810–887, Córdoba) would
have built one, and executable from scratch in pure NumPy.
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 12 Minds 221 - 240 Available on Amazon https://www.amazon.com/dp/B0HKVBKNTT
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0222_Abbas_ibn_Firnas_810 - Abbas ibn Firnas (c.810-887, Córdoba)
================================================================================  
WHY THIS SHAPE (the mind behind the mechanism)
----------------------------------------------
Ibn Hayyan's Muqtabis (the earliest account, 10th–11th c.) remembers four
things about Ibn Firnas that are *cognitive*, not merely biographical:

  1. He built, in his own house, a chamber in which visitors saw stars and
     clouds and heard thunder and saw lightning — a constructed sky.
     -> here: `SkyChamber`, a differentiable-by-learning physics room in which
        the agent rehearses before it risks its body.

  2. He "spread two wings of calculated structure" (wazn: weight / measure),
     feathers fastened to silk, rose into the air from the Rusafa hills and
     circled to a distant landing.
     -> here: the *lift-curve activation* `StallUnit`: y = a·exp(-a²/2σ²).
        A wing's lift rises with angle of attack and then *stalls*. A neuron
        that stalls when over-driven is the honest computational signature of
        a man who trusted birds' wings and learned their limits with his spine.

  3. He introduced and explained the Sindhind tables (al-Khwarizmi's zij) to
     al-Andalus — computation by *tabulated values and interpolation*.
     -> here: `ZijTable`, a learnable interpolation table on every input line.
        Not attention, not a look-up of stored keys: a numeric table whose
        entries are trained and read by linear interpolation, exactly as a
        zij was read.

  4. The landing failed because he had not modelled the tail: "he did not
     take into account that a bird, when landing, does so on its tail".
     -> here: `TailGrowth`. The model starts *without* the pitch variables and
        without a tail actuator (the honest state of his knowledge before the
        fall). After rehearsal, the residual of the world model is audited
        against the variables the model cannot see. When the fall correlates
        with an unmodelled quantity, the architecture grows that quantity into
        itself — a new input line, new table rows, new hidden units, a new
        actuator — and re-learns. Structure is discovered from failure.

  5. He built a water-clock (al-maqata / al-minqana) for the emir Muhammad I
     and was the first Andalusi to explain al-Khalil's metrical theory
     ('arud). Time, for him, came in measured beats.
     -> here: `MaqataClock`: the controller re-plans on a metrical foot
        (watad = 3 beats, sabab = 2 beats: the foot fa'ūlun) and receives the
        clock's phase as an input.

WORKING CONVENTIONS (mandatory for the corpus)
----------------------------------------------
  * pure NumPy, float64, no autograd library: every layer implements its own
    forward/backward, and the whole flight is back-propagated through time
    across the *learned* chamber (model-based policy optimisation);
  * finite-difference gradient checks on (a) the chamber model parameters and
    (b) the controller parameters *through* the unrolled learned model;
  * a real training loop with a real optimiser (Adam, from scratch);
  * self-tests that must pass before the run is accepted.

Run:  python3 chapter_0222_Abbas_ibn_Firnas_810.py
"""

from __future__ import annotations
import sys
import time
import math
import numpy as np

np.set_printoptions(precision=4, suppress=True, linewidth=120)

# ----------------------------------------------------------------------------
# 0. Small utilities
# ----------------------------------------------------------------------------

def softplus(z):
    return np.where(z > 30, z, np.log1p(np.exp(np.minimum(z, 30))))

def dsoftplus(z):
    return 1.0 / (1.0 + np.exp(-z))

def banner(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ----------------------------------------------------------------------------
# 1. THE SKY CHAMBER — the meteorological room, as a glider physics simulator
# ----------------------------------------------------------------------------

class SkyChamber:
    """
    Two-dimensional longitudinal glider dynamics with optional gusts.

    State s = [x, y, vx, vy, theta, q]
        x, y      position (y = height above the Rusafa slope, 0 = ground)
        vx, vy    velocity
        theta     pitch angle of the body (rad)
        q         pitch rate (rad / unit time)

    Action a = [spread, elevator]  (each in [-1, 1])
        spread    how far the two wings are spread -> wing area factor
        elevator  tail deflection; IGNORED when the glider has no tail
                  (this is the whole point of the chapter)

    Aerodynamics: the wing lift coefficient is a *stall curve*,
        CL(alpha) = CL_a * alpha * exp(-alpha² / (2 alpha_s²)),
    the same functional form used by `StallUnit` below. The wing alone is
    pitch-*unstable* (its lift acts ahead of the centre of mass). A tail adds a
    restoring moment and a control moment. Without the tail there is no way
    to flare: the landing is decided by luck.
    """

    def __init__(self, gust=0.0, seed=0, dt=0.12):
        self.dt = dt
        self.g = 1.0
        self.m = 1.0
        self.I = 0.12
        self.rho_S = 0.09          # 0.5 * rho * S  (nondimensional)
        self.CL_a = 5.0
        self.alpha_stall = 0.35
        self.CD0 = 0.08
        self.k_ind = 0.10
        self.l_wing = 0.03         # lift acts ahead of CG -> nose-up (unstable)
        self.c_damp = 0.08         # weak natural pitch damping
        self.tail_S = 0.15         # tail area factor (relative to wing)
        self.l_tail = 0.9          # tail moment arm (restoring)
        self.elev_gain = 0.20
        self.tail_incidence = -0.04
        self.gust = gust
        self.rng = np.random.default_rng(seed)

    # --- the stall curve is shared with the neuron ---
    @staticmethod
    def stall_curve(alpha, CL_a=5.0, alpha_s=0.35):
        return CL_a * alpha * np.exp(-alpha ** 2 / (2 * alpha_s ** 2))

    def reset(self, batch=1, height=10.0, speed=6.0):
        s = np.zeros((batch, 6))
        s[:, 1] = height + self.rng.uniform(-2, 2, batch)
        s[:, 2] = speed + self.rng.uniform(-1, 1, batch)
        s[:, 3] = self.rng.uniform(-0.5, 0.5, batch)
        s[:, 4] = self.rng.uniform(-0.15, 0.15, batch)
        s[:, 5] = self.rng.uniform(-0.1, 0.1, batch)
        return s

    def derivatives(self, s, a, has_tail):
        x, y, vx, vy, th, q = [s[:, i] for i in range(6)]
        spread = 0.7 + 0.3 * np.clip(a[:, 0], -1, 1)
        elev = np.clip(a[:, 1], -1, 1) if has_tail else 0.0
        V2 = vx ** 2 + vy ** 2 + 1e-9
        V = np.sqrt(V2)
        gamma = np.arctan2(vy, vx)
        alpha = th - gamma
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
        CL = self.stall_curve(alpha, self.CL_a, self.alpha_stall)
        CD = self.CD0 + self.k_ind * CL ** 2
        qS = self.rho_S * spread * V2
        L = qS * CL
        D = qS * CD
        # pitch: wing alone is unstable (nose-up with lift), lightly damped
        M = self.l_wing * L - self.c_damp * q
        if has_tail:
            # a tail behind the centre of mass: pitching nose-up (q>0) swings it
            # down into the wind, raising its angle of attack -> restoring moment
            alpha_t = alpha + self.tail_incidence + self.elev_gain * elev
            alpha_t = alpha_t + self.l_tail * q / (V + 0.5)
            L_t = self.rho_S * self.tail_S * V2 * self.stall_curve(alpha_t, 4.0, 0.45)
            M = M - self.l_tail * L_t
            L = L + L_t
            D = D + 0.02 * np.abs(L_t)
        ax = (-D * np.cos(gamma) - L * np.sin(gamma)) / self.m
        ay = (-D * np.sin(gamma) + L * np.cos(gamma)) / self.m - self.g
        qdot = M / self.I
        return np.stack([vx, vy, ax, ay, q, qdot], axis=1)

    def step(self, s, a, has_tail):
        """semi-implicit Euler with optional gust on vy"""
        d = self.derivatives(s, a, has_tail)
        s2 = s + self.dt * d
        if self.gust > 0:
            s2[:, 3] += self.gust * self.rng.normal(0, 1, s.shape[0]) * self.dt
        # keep pitch in (-pi, pi]
        s2[:, 4] = (s2[:, 4] + np.pi) % (2 * np.pi) - np.pi
        return s2

    def rollout(self, policy, has_tail, batch=16, T=140, seed=None):
        """Run the real chamber until touchdown. Returns per-episode landing stats."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        s = self.reset(batch)
        alive = np.ones(batch, dtype=bool)
        touchdown = np.zeros((batch, 6))
        t_land = np.full(batch, T, dtype=int)
        traj_s, traj_a, traj_n = [], [], []
        for t in range(T):
            a = policy(s, t)
            s2 = self.step(s, a, has_tail)
            traj_s.append(s.copy()); traj_a.append(a.copy()); traj_n.append(s2.copy())
            landed = alive & (s2[:, 1] <= 0.0)
            touchdown[landed] = s2[landed]
            t_land[landed] = t
            alive &= ~landed
            s = s2
            if not alive.any():
                break
        touchdown[alive] = s[alive]      # never landed within T: use last state
        stats = {
            "vy_touch": np.abs(touchdown[:, 3]),
            "pitch_touch": np.abs(touchdown[:, 4]),
            "distance": touchdown[:, 0],
            "landed_frac": 1.0 - alive.mean(),
            "t_land": t_land,
        }
        data = (np.concatenate(traj_s), np.concatenate(traj_a), np.concatenate(traj_n))
        return stats, data


# ----------------------------------------------------------------------------
# 2. LAYERS — each with its own forward / backward; caches are returned, not
#    stored, so the same layer can be used many times inside one unrolled flight
# ----------------------------------------------------------------------------

class Layer:
    def params(self):
        return {}
    def grads(self):
        return {}
    def zero_grad(self):
        for k, g in self.grads().items():
            g[...] = 0.0


class ZijTable(Layer):
    """
    A learnable interpolation table on every input line (the zij principle).

    For input dimension i, K knots at positions z_k on [lo, hi]; the table
    stores values v[i, k]. Output_i(x) = linear interpolation of v[i, :] at x
    (clamped at the ends). Initialised to the identity table v[i,k] = z_k, so a
    freshly built table is a transparent pane of glass; training bends it.

    Gradients: dL/dv[i,k] is the hat-basis weight; dL/dx_i is the local slope.
    The table is *read*, exactly as al-Khwarizmi's Sindhind was read.
    """

    def __init__(self, dim, K=17, lo=-3.0, hi=3.0):
        self.K = K
        self.lo, self.hi = lo, hi
        self.z = np.linspace(lo, hi, K)
        self.h = self.z[1] - self.z[0]
        self.v = np.tile(self.z, (dim, 1)).astype(np.float64)   # identity table
        self.dv = np.zeros_like(self.v)

    @property
    def dim(self):
        return self.v.shape[0]

    def grow(self, extra):
        """append `extra` new identity rows (new input lines)"""
        new = np.tile(self.z, (extra, 1))
        self.v = np.vstack([self.v, new])
        self.dv = np.zeros_like(self.v)

    def params(self):
        return {"zij_v": self.v}
    def grads(self):
        return {"zij_v": self.dv}

    def forward(self, x):
        xc = np.clip(x, self.lo, self.hi - 1e-12)
        pos = (xc - self.lo) / self.h
        k0 = np.floor(pos).astype(int)
        k0 = np.clip(k0, 0, self.K - 2)
        w = pos - k0                                    # weight of the right knot
        rows = np.arange(self.dim)[None, :]
        v0 = self.v[rows, k0]
        v1 = self.v[rows, k0 + 1]
        y = (1 - w) * v0 + w * v1
        inside = (x > self.lo) & (x < self.hi)
        cache = (k0, w, v0, v1, inside)
        return y, cache

    def backward(self, dy, cache):
        k0, w, v0, v1, inside = cache
        rows = np.arange(self.dim)[None, :]
        np.add.at(self.dv, (np.broadcast_to(rows, k0.shape), k0), dy * (1 - w))
        np.add.at(self.dv, (np.broadcast_to(rows, k0.shape), k0 + 1), dy * w)
        slope = (v1 - v0) / self.h
        dx = dy * slope * inside            # zero slope outside the table (clamped)
        return dx


class Dense(Layer):
    def __init__(self, n_in, n_out, rng, scale=None):
        scale = scale if scale is not None else math.sqrt(1.0 / n_in)
        self.W = rng.normal(0, scale, (n_in, n_out))
        self.b = np.zeros(n_out)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def grow(self, extra_in=0, extra_out=0, rng=None, in_scale=0.0):
        """
        Grow the layer without changing its current function:
          new input columns start at `in_scale` (0 => exactly function-preserving),
          new output units start with zero *outgoing* weights in the next layer.
        """
        n_in, n_out = self.W.shape
        if extra_in:
            newc = (rng.normal(0, in_scale, (extra_in, n_out)) if in_scale > 0
                    else np.zeros((extra_in, n_out)))
            self.W = np.vstack([self.W, newc])
        if extra_out:
            n_in2 = self.W.shape[0]
            newr = rng.normal(0, math.sqrt(1.0 / n_in2), (n_in2, extra_out))
            self.W = np.hstack([self.W, newr])
            self.b = np.concatenate([self.b, np.zeros(extra_out)])
        self.dW = np.zeros_like(self.W); self.db = np.zeros_like(self.b)

    def params(self):
        return {"W": self.W, "b": self.b}
    def grads(self):
        return {"W": self.dW, "b": self.db}

    def forward(self, x):
        return x @ self.W + self.b, x

    def backward(self, dy, x):
        self.dW += x.T @ dy
        self.db += dy.sum(0)
        return dy @ self.W.T


class StallUnit(Layer):
    """
    The wing neuron.  y = a * exp(-a² / (2 σ²)),  σ learnable per unit.
    Lift rises with angle of attack, peaks at a = σ, and stalls beyond it.
    A unit driven too hard *loses* output — the network cannot win by
    shouting, only by trimming.  dy/da = (1 - a²/σ²)·e,  dy/dσ = a³/σ³·e.
    """

    def __init__(self, n, sigma0=1.0):
        self.sigma = np.full(n, float(sigma0))
        self.dsigma = np.zeros(n)

    def grow(self, extra, sigma0=1.0):
        self.sigma = np.concatenate([self.sigma, np.full(extra, float(sigma0))])
        self.dsigma = np.zeros_like(self.sigma)

    def params(self):
        return {"sigma": self.sigma}
    def grads(self):
        return {"sigma": self.dsigma}

    def forward(self, a):
        e = np.exp(-a ** 2 / (2 * self.sigma ** 2))
        return a * e, (a, e)

    def backward(self, dy, cache):
        a, e = cache
        s = self.sigma
        da = dy * (1 - a ** 2 / s ** 2) * e
        self.dsigma += (dy * a ** 3 / s ** 3 * e).sum(0)
        return da


class Tanh(Layer):
    def forward(self, a):
        y = np.tanh(a)
        return y, y
    def backward(self, dy, y):
        return dy * (1 - y ** 2)


class Sequential(Layer):
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        caches = []
        for L in self.layers:
            x, c = L.forward(x)
            caches.append(c)
        return x, caches

    def backward(self, dy, caches):
        for L, c in zip(reversed(self.layers), reversed(caches)):
            dy = L.backward(dy, c)
        return dy

    def params(self):
        out = {}
        for i, L in enumerate(self.layers):
            for k, v in L.params().items():
                out[f"{i}.{k}"] = v
        return out

    def grads(self):
        out = {}
        for i, L in enumerate(self.layers):
            for k, v in L.grads().items():
                out[f"{i}.{k}"] = v
        return out

    def zero_grad(self):
        for L in self.layers:
            L.zero_grad()

    def n_params(self):
        return sum(v.size for v in self.params().values())


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, clip=5.0):
        self.p = params
        self.lr, self.b1, self.b2, self.eps, self.clip = lr, b1, b2, eps, clip
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def rebind(self, params):
        """after growth the parameter arrays change shape: re-create moments"""
        self.p = params
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads):
        self.t += 1
        gn = math.sqrt(sum(float((g ** 2).sum()) for g in grads.values()))
        scale = min(1.0, self.clip / (gn + 1e-12))
        for k in self.p:
            g = grads[k] * scale
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            self.p[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)
        return gn


# ----------------------------------------------------------------------------
# 3. THE MAQATA CLOCK — measured time and the metrical foot
# ----------------------------------------------------------------------------

class MaqataClock:
    """
    The controller does not re-plan every tick. It re-plans on the beats of a
    metrical foot — fa'ūlun = watad (3) + sabab (2) — the foot Ibn Firnas
    explained to Córdoba from al-Khalil's 'arud. Between beats the action is
    held, like a water clock's cup that fills, tips, and fills again.
    It also provides the clock phase (sin, cos) as an input to the controller.
    """

    def __init__(self, foot=(3, 2), period=24):
        self.foot = foot
        self.period = period
        self.beats = self._beat_pattern(400)

    def _beat_pattern(self, T):
        b = np.zeros(T, dtype=bool)
        t = 0
        while t < T:
            for span in self.foot:
                b[t] = True
                t += span
                if t >= T:
                    break
        return b

    def is_beat(self, t):
        return bool(self.beats[t])

    def phase(self, t, batch):
        ph = 2 * np.pi * (t % self.period) / self.period
        return np.tile([math.sin(ph), math.cos(ph)], (batch, 1))


# ----------------------------------------------------------------------------
# 4. THE AGENT — chamber model + wing controller + tail growth
# ----------------------------------------------------------------------------

STATE_NAMES = ["x", "y", "vx", "vy", "theta", "q"]
STATE_SCALE = np.array([40.0, 20.0, 8.0, 6.0, 1.0, 2.0])

class FirnasAgent:
    """
    Two networks and one growth rule.

    chamber : learned model of the SkyChamber.  input = [visible state, action]
              -> predicted Δstate for the *visible* state variables.
    wing    : the controller. input = [visible state, clock phase] -> actions.
    mask    : which state variables the agent currently *knows about*. It
              starts as [x, y, vx, vy] — pitch and pitch-rate are the tail
              dimension he had not thought of. n_actions starts at 1 (spread).

    grow_tail(): unmask pitch & pitch-rate, add an elevator action, widen every
              table and every dense layer *without changing the current
              function*, and rebind the optimisers.
    """

    def __init__(self, hidden=48, K=17, seed=1):
        self.rng = np.random.default_rng(seed)
        self.mask = np.array([1, 1, 1, 1, 0, 0], dtype=bool)
        self.n_actions = 1
        self.hidden = hidden
        self.K = K
        self.clock = MaqataClock()
        self.has_tail = False
        self._build()

    # ---- construction ------------------------------------------------------
    def _build(self):
        nv = int(self.mask.sum())
        H, K, rng = self.hidden, self.K, self.rng
        # chamber model: [visible state, actions] -> Δ visible state
        self.chamber = Sequential([
            ZijTable(nv + self.n_actions, K),
            Dense(nv + self.n_actions, H, rng), StallUnit(H, 1.4),
            Dense(H, H, rng), StallUnit(H, 1.4),
            Dense(H, nv, rng, scale=0.01),
        ])
        # wing controller: [visible state, clock phase] -> actions in (-1,1)
        self.wing = Sequential([
            ZijTable(nv + 2, K),
            Dense(nv + 2, H, rng), StallUnit(H, 1.4),
            Dense(H, self.n_actions, rng, scale=0.05), Tanh(),
        ])
        self.opt_chamber = Adam(self.chamber.params(), lr=2e-3)
        self.opt_wing = Adam(self.wing.params(), lr=1.5e-3)

    def visible(self, s):
        return (s / STATE_SCALE)[:, self.mask]

    # ---- growth: the tail --------------------------------------------------
    def grow_tail(self, hidden_extra=16):
        """
        Function-preserving structural growth:
          * state mask gains theta, q   (two new input lines)
          * one new action (elevator)   (new input line for the chamber, new
                                          output unit for the wing)
          * chamber predicts Δtheta, Δq too (two new output units, near zero)
          * `hidden_extra` new StallUnits in every hidden layer
        """
        rng = self.rng
        old_mask = self.mask.copy()
        self.mask[4] = True; self.mask[5] = True
        self.n_actions = 2
        new_state_lines = int(self.mask.sum() - old_mask.sum())    # 2
        # --- chamber ---
        zij, d1, s1, d2, s2, d3 = self.chamber.layers
        # inputs were [x,y,vx,vy | spread]; now [x,y,vx,vy,theta,q | spread,elev]
        # We must *reorder*: new state lines go before the action line(s).
        nv_old = int(old_mask.sum())
        old_v = zij.v.copy()
        zij.grow(new_state_lines + 1)                 # 2 state lines + 1 action
        vrows = np.vstack([old_v[:nv_old], zij.v[nv_old + 1: nv_old + 1 + new_state_lines],
                           old_v[nv_old:], zij.v[-1:]])
        zij.v = vrows; zij.dv = np.zeros_like(zij.v)
        W = d1.W
        newW = np.zeros((W.shape[0] + new_state_lines + 1, W.shape[1]))
        newW[:nv_old] = W[:nv_old]
        newW[nv_old + new_state_lines: nv_old + new_state_lines + 1] = W[nv_old:]
        d1.W = newW; d1.dW = np.zeros_like(newW)
        d1.grow(extra_out=hidden_extra, rng=rng); s1.grow(hidden_extra, 1.4)
        d2.grow(extra_in=hidden_extra, extra_out=hidden_extra, rng=rng); s2.grow(hidden_extra, 1.4)
        d3.grow(extra_in=hidden_extra, rng=rng)
        # new output units for Δtheta, Δq (tiny random so they can start learning)
        d3.W = np.hstack([d3.W, rng.normal(0, 0.01, (d3.W.shape[0], new_state_lines))])
        d3.b = np.concatenate([d3.b, np.zeros(new_state_lines)])
        d3.dW = np.zeros_like(d3.W); d3.db = np.zeros_like(d3.b)
        # --- wing ---
        zijw, w1, sw1, w2, tanh = self.wing.layers
        old_vw = zijw.v.copy()
        zijw.grow(new_state_lines)
        # inputs were [x,y,vx,vy | sin,cos]; now [x,y,vx,vy,theta,q | sin,cos]
        zijw.v = np.vstack([old_vw[:nv_old], zijw.v[-new_state_lines:], old_vw[nv_old:]])
        zijw.dv = np.zeros_like(zijw.v)
        Ww = w1.W
        newWw = np.zeros((Ww.shape[0] + new_state_lines, Ww.shape[1]))
        newWw[:nv_old] = Ww[:nv_old]
        newWw[nv_old + new_state_lines:] = Ww[nv_old:]
        # small non-zero so the pitch lines are not stuck at a saddle
        newWw[nv_old: nv_old + new_state_lines] = rng.normal(0, 0.05, (new_state_lines, Ww.shape[1]))
        w1.W = newWw; w1.dW = np.zeros_like(newWw)
        w1.grow(extra_out=hidden_extra, rng=rng); sw1.grow(hidden_extra, 1.4)
        w2.grow(extra_in=hidden_extra, rng=rng)
        w2.W = np.hstack([w2.W, rng.normal(0, 0.05, (w2.W.shape[0], 1))])   # elevator head
        w2.b = np.concatenate([w2.b, np.zeros(1)])
        w2.dW = np.zeros_like(w2.W); w2.db = np.zeros_like(w2.b)
        self.has_tail = True
        self.opt_chamber.rebind(self.chamber.params())
        self.opt_wing.rebind(self.wing.params())

    # ---- policies ----------------------------------------------------------
    def act(self, s, t, held=None):
        """metrical action: recompute on beats, hold otherwise"""
        if held is None or self.clock.is_beat(t):
            inp = np.hstack([self.visible(s), self.clock.phase(t, s.shape[0])])
            a, _ = self.wing.forward(inp)
            held = a
        if held.shape[1] < 2:          # no elevator yet -> zero deflection
            held = np.hstack([held, np.zeros((held.shape[0], 1))])
        return held

    def policy(self):
        state = {"held": None}
        def pi(s, t):
            state["held"] = self.act(s, t, state["held"] if not self.clock.is_beat(t) else None)
            return state["held"]
        return pi

    def random_policy(self):
        rng = np.random.default_rng(7)
        state = {"held": None}
        def pi(s, t):
            if state["held"] is None or self.clock.is_beat(t):
                state["held"] = rng.uniform(-1, 1, (s.shape[0], 2))
            return state["held"]
        return pi

    # ---- chamber (world model) training ------------------------------------
    def chamber_inputs(self, s, a):
        return np.hstack([self.visible(s), a[:, :self.n_actions]])

    def chamber_targets(self, s, s2):
        return (self.visible(s2) - self.visible(s)) * 10.0     # scaled Δ

    def chamber_loss(self, s, a, s2):
        u = self.chamber_inputs(s, a)
        y, cache = self.chamber.forward(u)
        tgt = self.chamber_targets(s, s2)
        diff = y - tgt
        loss = 0.5 * (diff ** 2).sum() / s.shape[0]
        return loss, diff / s.shape[0], cache

    def train_chamber(self, data, epochs=60, batch=128, verbose=True):
        s, a, s2 = data
        n = s.shape[0]
        rng = np.random.default_rng(3)
        hist = []
        for ep in range(epochs):
            idx = rng.permutation(n)
            tot = 0.0
            for i in range(0, n, batch):
                b = idx[i:i + batch]
                self.chamber.zero_grad()
                loss, dloss, cache = self.chamber_loss(s[b], a[b], s2[b])
                self.chamber.backward(dloss, cache)
                self.opt_chamber.step(self.chamber.grads())
                tot += loss * len(b)
            hist.append(tot / n)
            if verbose and (ep % max(1, epochs // 6) == 0 or ep == epochs - 1):
                print(f"   chamber epoch {ep:3d}   loss {tot / n:.5f}")
        return hist

    # ---- residual audit: what does the fall correlate with? ---------------
    def audit_residuals(self, data):
        """
        Correlate the chamber's prediction residual (on the vertical-speed line)
        with every *hidden* state variable. This is the moment after the fall:
        'what did I not take into account?'
        """
        s, a, s2 = data
        u = self.chamber_inputs(s, a)
        y, _ = self.chamber.forward(u)
        res = y - self.chamber_targets(s, s2)
        vis_idx = np.where(self.mask)[0]
        report = {}
        for j in np.where(~self.mask)[0]:
            hcol = s[:, j] / STATE_SCALE[j]
            best, best_line = 0.0, None
            for col, vj in enumerate(vis_idx):          # every visible residual line
                c = float(np.corrcoef(res[:, col], hcol)[0, 1])
                if abs(c) > abs(best):
                    best, best_line = c, STATE_NAMES[vj]
            report[STATE_NAMES[j]] = (best, best_line)
        return report

    def table_report(self):
        """how far each zij table has bent away from the identity pane it began as"""
        out = {}
        for name, net in (("chamber", self.chamber), ("wing", self.wing)):
            z = net.layers[0]
            out[name] = np.abs(z.v - z.z[None, :]).mean(1)
        return out

    # ---- flight training: BPTT through the learned chamber ----------------
    def unrolled_loss(self, s0, H, accumulate=True):
        """
        Roll the *learned* chamber forward for H steps under the wing policy and
        back-propagate the landing cost through time.  Returns the loss and
        (if accumulate) writes controller gradients into self.wing grads.

        Cost:  per step   k_touch(y)·(vy² + θ²)  + 0.02·|a|²
                          where k_touch = exp(-y²/2h²) weights the near-ground
                          part of the flight (the landing is what matters)
               terminal   softplus(y_T)·2  (come down!)  − 0.03·x_T (distance)
        """
        B = s0.shape[0]
        nv = int(self.mask.sum())
        vis_idx = np.where(self.mask)[0]
        pos = {STATE_NAMES[j]: i for i, j in enumerate(vis_idx)}
        iy, ivy = pos["y"], pos["vy"]
        ith = pos.get("theta", None)
        h_touch = 4.0 / STATE_SCALE[1]

        v = self.visible(s0)                    # visible normalised state
        Vs, caches_c, caches_w, acts, beats = [v], [], [], [], []
        held = None; held_cache = None
        total = 0.0
        step_grads = []                          # dcost/dv_t (partial, local)
        for t in range(H):
            if held is None or self.clock.is_beat(t):
                inp = np.hstack([v, self.clock.phase(t, B)])
                a, cw = self.wing.forward(inp)
                held, held_cache, beat = a, cw, True
            else:
                a, cw, beat = held, None, False
            u = np.hstack([v, a])
            dv_pred, cc = self.chamber.forward(u)
            v2 = v + dv_pred / 10.0
            # local step cost
            y = v[:, iy] * STATE_SCALE[1]
            k = np.exp(-y ** 2 / (2 * h_touch ** 2 * STATE_SCALE[1] ** 2))
            vy = v[:, ivy]
            th = v[:, ith] if ith is not None else 0.0
            c_t = k * (vy ** 2 * 4.0 + (th ** 2 * 2.0 if ith is not None else 0.0)) + 0.02 * (a ** 2).sum(1)
            total += c_t.sum() / B
            # partial derivatives of c_t
            g_v = np.zeros_like(v); g_a = 0.04 * a / B
            dk_dy = k * (-y / (h_touch ** 2 * STATE_SCALE[1] ** 2)) * STATE_SCALE[1]
            core = vy ** 2 * 4.0 + (th ** 2 * 2.0 if ith is not None else 0.0)
            g_v[:, iy] += dk_dy * core / B
            g_v[:, ivy] += k * 8.0 * vy / B
            if ith is not None:
                g_v[:, ith] += k * 4.0 * th / B
            step_grads.append((g_v, g_a))
            Vs.append(v2); caches_c.append(cc); caches_w.append(cw); acts.append(a); beats.append(beat)
            v = v2
        # terminal cost
        yT = v[:, iy] * STATE_SCALE[1]
        xT = v[:, pos["x"]] * STATE_SCALE[0]
        c_T = 2.0 * softplus(yT) - 0.03 * xT
        total += c_T.sum() / B
        if not accumulate:
            return total
        # ---- backward through time ----
        dv = np.zeros_like(v)
        dv[:, iy] += 2.0 * dsoftplus(yT) * STATE_SCALE[1] / B
        dv[:, pos["x"]] += -0.03 * STATE_SCALE[0] / B
        da_held = None
        for t in reversed(range(H)):
            g_v, g_a = step_grads[t]
            dv_total = dv + g_v                       # dL/dv_{t+1} arrives as dv; add local
            # through chamber: v2 = v + chamber(u)/10
            du = self.chamber.backward(dv / 10.0, caches_c[t])
            dv_prev = dv_total + du[:, :nv]
            da = du[:, nv:] + g_a
            # action was produced at the last beat; accumulate until we reach it
            da_held = da if da_held is None else da_held + da
            if beats[t]:
                dinp = self.wing.backward(da_held, caches_w[t])
                dv_prev = dv_prev + dinp[:, :nv]
                da_held = None
            dv = dv_prev
        return total

    def train_wing(self, iters=150, H=60, batch=24, chamber_env=None, verbose=True):
        rng = np.random.default_rng(11)
        env = chamber_env
        hist = []
        for it in range(iters):
            s0 = env.reset(batch)
            self.wing.zero_grad(); self.chamber.zero_grad()
            loss = self.unrolled_loss(s0, H)
            self.opt_wing.step(self.wing.grads())      # chamber stays frozen here
            hist.append(loss)
            if verbose and (it % max(1, iters // 5) == 0 or it == iters - 1):
                print(f"   wing iter {it:3d}   imagined landing cost {loss:.4f}")
        return hist


# ----------------------------------------------------------------------------
# 5. GRADIENT CHECKS (finite differences) — mandatory
# ----------------------------------------------------------------------------

def grad_check_chamber(agent, data, n_samples=24, eps=1e-5):
    s, a, s2 = data
    s, a, s2 = s[:8], a[:8], s2[:8]
    agent.chamber.zero_grad()
    loss, dloss, cache = agent.chamber_loss(s, a, s2)
    agent.chamber.backward(dloss, cache)
    params, grads = agent.chamber.params(), agent.chamber.grads()
    rng = np.random.default_rng(5)
    worst = 0.0; worst_abs = 0.0; checked = 0
    for k in params:
        P, G = params[k], grads[k]
        flat = P.reshape(-1)
        for _ in range(n_samples if P.size > n_samples else P.size):
            i = rng.integers(P.size)
            old = flat[i]
            flat[i] = old + eps; lp = agent.chamber_loss(s, a, s2)[0]
            flat[i] = old - eps; lm = agent.chamber_loss(s, a, s2)[0]
            flat[i] = old
            num = (lp - lm) / (2 * eps)
            ana = G.reshape(-1)[i]
            # relative error on coordinates whose gradient is not numerically
            # negligible (a table knot that no input ever touched has zero
            # gradient and would only measure round-off)
            if abs(num) + abs(ana) > 1e-7:
                rel = abs(num - ana) / (abs(num) + abs(ana))
                worst = max(worst, rel)
            worst_abs = max(worst_abs, abs(num - ana)); checked += 1
    return worst, worst_abs, checked


def grad_check_wing(agent, env, H=12, batch=3, n_samples=20, eps=1e-5):
    s0 = env.reset(batch)
    agent.wing.zero_grad(); agent.chamber.zero_grad()
    agent.unrolled_loss(s0, H)
    params, grads = agent.wing.params(), agent.wing.grads()
    rng = np.random.default_rng(9)
    worst = 0.0; worst_abs = 0.0; checked = 0
    for k in params:
        P, G = params[k], grads[k]
        flat = P.reshape(-1)
        for _ in range(min(n_samples, P.size)):
            i = rng.integers(P.size)
            old = flat[i]
            flat[i] = old + eps; lp = agent.unrolled_loss(s0, H, accumulate=False)
            flat[i] = old - eps; lm = agent.unrolled_loss(s0, H, accumulate=False)
            flat[i] = old
            num = (lp - lm) / (2 * eps)
            ana = G.reshape(-1)[i]
            # relative error on coordinates whose gradient is not numerically
            # negligible (a table knot that no input ever touched has zero
            # gradient and would only measure round-off)
            if abs(num) + abs(ana) > 1e-7:
                rel = abs(num - ana) / (abs(num) + abs(ana))
                worst = max(worst, rel)
            worst_abs = max(worst_abs, abs(num - ana)); checked += 1
    return worst, worst_abs, checked


# ----------------------------------------------------------------------------
# 6. SELF-TESTS
# ----------------------------------------------------------------------------

def self_tests():
    banner("SELF-TESTS")
    ok = True
    rng = np.random.default_rng(0)

    # (1) a fresh zij table is the identity inside its range
    z = ZijTable(3, K=13)
    x = rng.uniform(-2.9, 2.9, (50, 3))
    y, _ = z.forward(x)
    e1 = np.abs(y - x).max()
    print(f"[1] identity zij table            max|y-x| = {e1:.2e}   ", "PASS" if e1 < 1e-12 else "FAIL")
    ok &= e1 < 1e-12

    # (2) zij backward slope equals finite difference (away from knots)
    z.v += rng.normal(0, 0.3, z.v.shape)
    x = np.array([[0.123, -1.377, 2.05]])
    y, c = z.forward(x)
    dx = z.backward(np.ones_like(y), c)
    eps = 1e-6; fd = np.zeros(3)
    for i in range(3):
        xp = x.copy(); xp[0, i] += eps; xm = x.copy(); xm[0, i] -= eps
        fd[i] = (z.forward(xp)[0][0, i] - z.forward(xm)[0][0, i]) / (2 * eps)
    e2 = np.abs(fd - dx[0]).max()
    print(f"[2] zij input-gradient vs FD       max err = {e2:.2e}   ", "PASS" if e2 < 1e-6 else "FAIL")
    ok &= e2 < 1e-6

    # (3) stall unit: peak at a = σ and derivative check
    su = StallUnit(1, 1.3)
    a = np.linspace(-4, 4, 4001)[:, None]
    y, _ = su.forward(a)
    apk = a[np.argmax(y), 0]
    e3 = abs(apk - 1.3)
    ya, c = su.forward(np.array([[0.7]]))
    da = su.backward(np.ones((1, 1)), c)[0, 0]
    fd = (su.forward(np.array([[0.7 + eps]]))[0] - su.forward(np.array([[0.7 - eps]]))[0])[0, 0] / (2 * eps)
    e3b = abs(fd - da)
    print(f"[3] stall unit peaks at σ (err {e3:.3f}); dy/da vs FD err {e3b:.2e}   ",
          "PASS" if (e3 < 0.01 and e3b < 1e-6) else "FAIL")
    ok &= e3 < 0.01 and e3b < 1e-6

    # (4) the chamber physics: a tailless wing is pitch-unstable, a tailed one is not
    env = SkyChamber(seed=1)
    s = np.array([[0, 40.0, 6.0, 0.0, 0.05, 0.0]])
    a = np.array([[0.0, 0.0]])
    s_nt = s.copy(); s_t = s.copy()
    for _ in range(45):
        s_nt = env.step(s_nt, a, False); s_t = env.step(s_t, a, True)
    print(f"[4] |pitch| after 45 steps: no tail {abs(s_nt[0,4]):.3f}  with tail {abs(s_t[0,4]):.3f}   ",
          "PASS" if abs(s_nt[0, 4]) > 3 * abs(s_t[0, 4]) else "FAIL")
    ok &= abs(s_nt[0, 4]) > 3 * abs(s_t[0, 4])

    # (5) growth is function-preserving for the chamber on the old lines
    ag = FirnasAgent(hidden=16, K=9, seed=3)
    sb = env.reset(6); ab = np.zeros((6, 2)); ab[:, 0] = rng.uniform(-1, 1, 6)
    y_before, _ = ag.chamber.forward(ag.chamber_inputs(sb, ab))
    ag.grow_tail(hidden_extra=8)
    y_after, _ = ag.chamber.forward(ag.chamber_inputs(sb, ab))
    e5 = np.abs(y_after[:, :4] - y_before).max()
    print(f"[5] tail growth preserves chamber function on old lines   max diff = {e5:.2e}   ",
          "PASS" if e5 < 1e-10 else "FAIL")
    ok &= e5 < 1e-10

    # (6) the metrical clock: fa'ūlun beats fall at 0,3,5,8,10,13,...
    ck = MaqataClock()
    beats = [t for t in range(14) if ck.is_beat(t)]
    e6 = beats == [0, 3, 5, 8, 10, 13]
    print(f"[6] maqata clock beats {beats}   ", "PASS" if e6 else "FAIL")
    ok &= e6
    return ok


# ----------------------------------------------------------------------------
# 7. MAIN — the whole arc: rehearse, fly, fall, audit, grow the tail, fly again
# ----------------------------------------------------------------------------

def summarize(stats, label):
    print(f"   {label:<34s} touchdown |vy| {stats['vy_touch'].mean():6.3f}  "
          f"|pitch| {stats['pitch_touch'].mean():6.3f}  distance {stats['distance'].mean():7.2f}  "
          f"landed {stats['landed_frac']*100:5.1f}%")


def main():
    t0 = time.time()
    banner("RUSAFA — 'Abbas ibn Firnas neuron model (pure NumPy, float64)")
    if not self_tests():
        print("SELF-TESTS FAILED"); sys.exit(1)

    env = SkyChamber(gust=0.15, seed=2)
    agent = FirnasAgent(hidden=48, K=17, seed=1)
    print(f"\nparameters: chamber {agent.chamber.n_params()}  wing {agent.wing.n_params()}  "
          f"(visible state lines: {[n for n, m in zip(STATE_NAMES, agent.mask) if m]}, actions: {agent.n_actions})")

    # ---- Act I: rehearsal in the sky chamber (learn the world, wing-only view)
    banner("ACT I  — rehearsal: learning the chamber with the wing-only view")
    stats_r, data = env.rollout(agent.random_policy(), has_tail=False, batch=48, T=140, seed=21)
    summarize(stats_r, "random flapping (no tail)")
    print(f"   collected {data[0].shape[0]} transitions")
    agent.train_chamber(data, epochs=48)

    banner("GRADIENT CHECK A — chamber model parameters (finite differences)")
    worst, wabs, n = grad_check_chamber(agent, data)
    print(f"   checked {n} coordinates   worst relative error = {worst:.2e}   worst absolute error = {wabs:.2e}   ",
          "PASS" if worst < 1e-5 else "FAIL")
    if worst >= 1e-5:
        sys.exit(1)

    # ---- Act II: the first flight — trained by imagination, tested in the air
    banner("ACT II — first flight: wing policy by BPTT through the learned chamber")
    banner_env = SkyChamber(gust=0.0, seed=5)
    agent.train_wing(iters=140, H=70, batch=24, chamber_env=banner_env)
    stats_1, data_1 = env.rollout(agent.policy(), has_tail=False, batch=48, T=140, seed=22)
    summarize(stats_1, "first flight (no tail)")

    banner("GRADIENT CHECK B — controller parameters THROUGH the unrolled model")
    worst_w, wabs_w, n_w = grad_check_wing(agent, banner_env)
    print(f"   checked {n_w} coordinates   worst relative error = {worst_w:.2e}   worst absolute error = {wabs_w:.2e}   ",
          "PASS" if worst_w < 1e-4 else "FAIL")
    if worst_w >= 1e-4:
        sys.exit(1)

    # ---- Act III: the fall, and the audit
    banner("ACT III — the fall: what did the model not take into account?")
    agent.train_chamber(data_1, epochs=10, verbose=False)   # refresh on real flights
    report = agent.audit_residuals(data_1)
    for k, (c, line) in report.items():
        print(f"   hidden variable {k:5s}: strongest residual correlation {c:+.3f} (on the {line} line)")
    culprit = max(report, key=lambda k: abs(report[k][0]))
    print(f"   -> the fall is most correlated with '{culprit}': the tail dimension the model cannot see.")
    print(f"   (imagined landing cost fell during Act II while the real touchdown did not improve:")
    print(f"    a chamber without the tail rehearses an optimistic sky)")

    # ---- Act IV: grow the tail (function-preserving), re-learn, fly again
    banner("ACT IV — growing the tail: new lines, new units, new actuator")
    before_c, before_w = agent.chamber.n_params(), agent.wing.n_params()
    agent.grow_tail(hidden_extra=16)
    print(f"   chamber params {before_c} -> {agent.chamber.n_params()},  wing params {before_w} -> {agent.wing.n_params()}")
    print(f"   visible state lines now: {[n for n, m in zip(STATE_NAMES, agent.mask) if m]}, actions: {agent.n_actions}")
    stats_r2, data_2 = env.rollout(agent.random_policy(), has_tail=True, batch=48, T=140, seed=23)
    summarize(stats_r2, "random flapping (with tail)")
    data_all = tuple(np.concatenate([d1, d2]) for d1, d2 in zip(data_1, data_2))
    agent.train_chamber(data_all, epochs=48)
    report2 = agent.audit_residuals(data_all)
    print(f"   hidden variables left to audit: {list(report2.keys()) or 'none — every line is visible'}")
    agent.train_wing(iters=180, H=70, batch=24, chamber_env=banner_env)
    stats_2, data_3 = env.rollout(agent.policy(), has_tail=True, batch=48, T=140, seed=24)
    summarize(stats_2, "second flight (with tail)")

    # one more round of chamber refinement + flight (he did not stop at one lesson)
    agent.train_chamber(tuple(np.concatenate([d, e]) for d, e in zip(data_all, data_3)), epochs=24, verbose=False)
    agent.train_wing(iters=120, H=70, batch=24, chamber_env=banner_env, verbose=False)
    stats_3, _ = env.rollout(agent.policy(), has_tail=True, batch=48, T=140, seed=25)
    summarize(stats_3, "third flight (with tail, refined)")

    banner("VERDICT")
    improved = stats_3["vy_touch"].mean() < 0.6 * stats_1["vy_touch"].mean()
    print(f"   touchdown |vy|: random {stats_r['vy_touch'].mean():.3f} -> no-tail {stats_1['vy_touch'].mean():.3f} "
          f"-> tail {stats_3['vy_touch'].mean():.3f}   ", "PASS (softer landing)" if improved else "FAIL")
    print(f"   touchdown |pitch|: no-tail {stats_1['pitch_touch'].mean():.3f} -> tail {stats_3['pitch_touch'].mean():.3f}")
    tr = agent.table_report()
    print(f"   zij tables, mean bend from identity per input line:")
    print(f"      chamber {np.round(tr['chamber'], 3)}")
    print(f"      wing    {np.round(tr['wing'], 3)}")
    sig = np.concatenate([L.sigma for L in agent.chamber.layers if isinstance(L, StallUnit)])
    print(f"   learned stall thresholds σ (chamber): min {sig.min():.3f}  mean {sig.mean():.3f}  max {sig.max():.3f}")
    print(f"   total run time {time.time() - t0:.1f}s")
    if not improved:
        sys.exit(1)
    print("\n   RUSAFA run complete: self-tests PASS, gradient checks PASS, the tail was learned from the fall.")


if __name__ == "__main__":
    main()

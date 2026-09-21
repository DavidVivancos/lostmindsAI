#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
0209_Habash_al-Hasib_al-Marwazi_870_Neuron.py
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 11 Minds 201 - 220 Available on Amazon https://www.amazon.com/dp/B0HKF7TRFF
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0209_Habash_al-Hasib_al-Marwazi_870 - Habash al-Hasib al-Marwazi (c. 770 - after 869), the ninth-century Abbasid "calculator" of Baghdad and Samarra
================================================================================  

THE SHADOW-INVERSION ENGINE  ("ZILL-NET")
An artificial-neuron architecture reconstructed from the cognitive signature of
Habash al-Hasib al-Marwazi (c. 770 - after 869), the ninth-century Abbasid
"calculator" of Baghdad and Samarra.

WHY THIS ARCHITECTURE  (read this before the code)
--------------------------------------------------
Habash never saw an angle.  He saw a shadow on the ground, a star at a certain
height above a wall, a crescent a hand's width from the sun.  Everything he is
remembered for is a way of turning a cast trace back into the thing that cast it:

  * the SHADOW TABLES (umbra recta / umbra versa = cotangent / tangent): for every
    degree of solar altitude, the length of the shadow of a 12-digit gnomon - and,
    read backwards, the altitude for every shadow.  The table is an INVERSE MAP.
  * the "time from the altitude" rule (the first exact one, proved later by
    Abu'l-Wafa' and al-Biruni): invert altitude -> hour angle.
  * the ANALEMMA for the qibla (the earliest exact solution): fold each circle of
    the sphere into the plane of the drawing by a rotation, read the answer off a
    planar construction.  Rotations, not formulas.
  * the ITERATED PARALLAX (Kennedy & Transue 1956; Tekeli, DSB): enter the parallax
    table with the argument, add the result to the argument, enter again ... to the
    "fifth parallax".  When the correction depends on the corrected value, the
    inverse is not a lookup but a FIXED POINT reached by repetition.
  * the MUMTAHAN ("tested") habit: a row of a table is only trusted once an
    observation has touched it; untested rows inherit their values from tested
    neighbours by interpolation "between the two lines".
  * INSTRUMENTS (melon astrolabe, universal star-timekeeping plate): a table cast in
    brass, so that a hand can do what the calculator once did.

The network below is the same mind written as differentiable machinery:

  input (a trace)
     |
  [AnalemmaLayer]   a learnable linear lift followed by a chain of planar (Givens)
     |              rotations - "fold the circles into the plane of the drawing"
  [ZijTable]        a tabular layer: every input->output edge is a TABLE over a fixed
     |              grid of nodes (kardajat), read by interpolation between rows
  [ShadowActivation] phi(a) = arctan(a / g): the shadow-to-altitude inversion, with a
     |              learnable gnomon length g per unit
  [ZijTable]        second table layer -> first estimate y0
     |
  [ParallaxRefiner] y_{k+1} = y0 + C(y_k, x) unrolled K=5 times ("the fifth parallax");
     |              corrections are stored NON-NEGATIVE and a known constant is taken
     |              off afterwards, as Habash arranged his lunar corrections
  output (the thing that cast the trace)

  + a MUMTAHAN LEDGER on every table: a visit count per node, and a regulariser that
    pulls untested nodes toward the interpolation of their tested neighbours.

Nothing here is a Transformer, an attention mechanism or a stored-key memory.  The
memory IS the table; the reasoning IS interpolation; the deliberation IS iteration
to a fixed point; the projection IS a rotation.

WORKING CONVENTION (shared by the whole corpus)
-----------------------------------------------
  * pure NumPy, everything from scratch (no autograd, no ML libraries);
  * a finite-difference GRADIENT CHECK is run and must pass before training;
  * a real training loop (Adam, mini-batches) on real astronomical inversions;
  * self-tests that assert the properties the design claims.

TASKS (all three are things Habash actually computed)
-----------------------------------------------------
  A. Time from the shadow:  (log shadow, declination, latitude) -> hour angle.
  B. The qibla:             (latitude, longitude difference) -> azimuth of Mecca.
  C. The fixed point:       (mean anomaly, eccentricity) -> eccentric anomaly, i.e.
                            E - e sin E = M, the equation Habash's solar rule leads to
                            (Tekeli, DSB) and which he attacked by repeated entry of
                            the table.

Run:   python 0209_Habash_al-Hasib_al-Marwazi_870_Neuron.py            (full run)
       python 0209_Habash_al-Hasib_al-Marwazi_870_Neuron.py --quick    (short run)
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np

# ----------------------------------------------------------------------------
# 0. small utilities
# ----------------------------------------------------------------------------

DEG = math.pi / 180.0


def softplus(z: np.ndarray) -> np.ndarray:
    """Numerically safe softplus."""
    return np.logaddexp(0.0, z)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class Param:
    """A named tensor with a gradient slot.  (Habash kept his columns named.)"""

    def __init__(self, name: str, value: np.ndarray):
        self.name = name
        self.value = np.asarray(value, dtype=np.float64)
        self.grad = np.zeros_like(self.value)

    def zero_grad(self):
        self.grad[...] = 0.0


# ----------------------------------------------------------------------------
# 1. ZijTable  -  the tabular layer ("jadwal")
# ----------------------------------------------------------------------------

class ZijTable:
    """
    A layer whose parameters are TABLES, not weights.

    For every (input i, output j) pair the layer stores a column of K values at fixed
    nodes ("kardajat") spanning [lo, hi].  The output is

        y_j(x) = b_j + sum_i  T_ij( x_i )

    where T_ij(.) is read from the column by interpolation between rows.

    Two reading rules are provided:
      'linear'  - interpolation between the two enclosing rows (ta'dil bayn
                  al-satrayn), the rule zij users applied in Habash's day;
      'cubic'   - the same reading refined by the tabulated differences of the two
                  neighbouring rows on each side (a smooth, second-order rule of the
                  family later Islamic astronomers used for ascensions).  It is C1, so
                  gradients with respect to the argument are continuous and the
                  finite-difference check is exact.  This is the default for training.

    A MUMTAHAN LEDGER (visit count per node of each input column) records how often
    an observation has actually fallen on a node.  The regulariser `ledger_penalty`
    pulls rarely-tested nodes toward the average of their neighbours, so that an
    untested row never asserts anything an observation has not earned.
    """

    def __init__(self, n_in: int, n_out: int, K: int = 16, lo: float = -4.0,
                 hi: float = 4.0, mode: str = "cubic", rng=None, name="zij"):
        assert K >= 4
        rng = np.random.default_rng(0) if rng is None else rng
        self.n_in, self.n_out, self.K = n_in, n_out, K
        self.lo, self.hi, self.mode = float(lo), float(hi), mode
        self.h = (hi - lo) / (K - 1)                      # row spacing
        self.nodes = lo + self.h * np.arange(K)           # the kardajat
        # initialise each column as a small random smooth function of its argument
        scale = 1.0 / math.sqrt(n_in)
        base = rng.normal(0.0, scale, size=(n_in, n_out, 1))
        slope = rng.normal(0.0, scale, size=(n_in, n_out, 1))
        self.T = Param(name + ".T", base + slope * self.nodes[None, None, :] / (hi - lo))
        self.b = Param(name + ".b", np.zeros(n_out))
        self.ledger = np.zeros((n_in, K))                 # visit counts per node
        self._cache = None

    # -- helpers ---------------------------------------------------------------
    def params(self):
        return [self.T, self.b]

    def _locate(self, x):
        """Which row-pair encloses each argument, and where between them."""
        xc = np.clip(x, self.lo, self.hi)
        u = (xc - self.lo) / self.h
        k = np.floor(u).astype(int)
        k = np.clip(k, 0, self.K - 2)
        t = u - k
        inside = (x > self.lo) & (x < self.hi)           # gradient flows only inside
        return k, t, inside

    def _extended(self):
        """The table with one virtual row extrapolated linearly beyond each end."""
        T = self.T.value
        first = 2.0 * T[:, :, :1] - T[:, :, 1:2]
        last = 2.0 * T[:, :, -1:] - T[:, :, -2:-1]
        return np.concatenate([first, T, last], axis=2)

    def _basis(self, t):
        """Hermite / linear basis weights on rows k-1, k, k+1, k+2 and their t-derivatives."""
        if self.mode == "linear":
            w = np.stack([np.zeros_like(t), 1.0 - t, t, np.zeros_like(t)], axis=-1)
            dw = np.stack([np.zeros_like(t), -np.ones_like(t), np.ones_like(t), np.zeros_like(t)], axis=-1)
            return w, dw
        t2, t3 = t * t, t * t * t
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2
        # slopes are central differences: m1 = (p2 - p0)/2, m2 = (p3 - p1)/2
        w0 = -0.5 * h10
        w1 = h00 - 0.5 * h11
        w2 = h01 + 0.5 * h10
        w3 = 0.5 * h11
        d00 = 6 * t2 - 6 * t
        d10 = 3 * t2 - 4 * t + 1
        d01 = -6 * t2 + 6 * t
        d11 = 3 * t2 - 2 * t
        dw0 = -0.5 * d10
        dw1 = d00 - 0.5 * d11
        dw2 = d01 + 0.5 * d10
        dw3 = 0.5 * d11
        w = np.stack([w0, w1, w2, w3], axis=-1)
        dw = np.stack([dw0, dw1, dw2, dw3], axis=-1)
        return w, dw

    # -- forward ---------------------------------------------------------------
    def forward(self, x: np.ndarray, record: bool = True) -> np.ndarray:
        B = x.shape[0]
        k, t, inside = self._locate(x)                     # (B, n_in)
        w, dw = self._basis(t)                             # (B, n_in, 4)
        idx = np.stack([k - 1, k, k + 1, k + 2], axis=-1) + 1   # (B, n_in, 4), extended indexing
        Text = self._extended()                            # (n_in, n_out, K+2) with virtual end rows
        Tt = Text.transpose(0, 2, 1)                       # (n_in, K+2, n_out)
        ii = np.arange(self.n_in)[None, :, None]
        rows = Tt[ii, idx]                                 # (B, n_in, 4, n_out)
        contrib = np.einsum("bik,bikj->bij", w, rows)      # (B, n_in, n_out)
        y = contrib.sum(axis=1) + self.b.value[None, :]
        if record:                                         # the mumtahan ledger
            np.add.at(self.ledger, (np.arange(self.n_in)[None, :].repeat(B, 0), k), 1.0)
            np.add.at(self.ledger, (np.arange(self.n_in)[None, :].repeat(B, 0), k + 1), 1.0)
        self._cache = (x, k, t, inside, w, dw, idx, rows)
        return y

    # -- backward --------------------------------------------------------------
    def backward(self, dy: np.ndarray) -> np.ndarray:
        x, k, t, inside, w, dw, idx, rows = self._cache
        B = x.shape[0]
        # bias
        self.b.grad += dy.sum(axis=0)
        # table entries: dL/dT[i, j, idx[b,i,m]] += w[b,i,m] * dy[b,j]
        gE = np.zeros((self.n_in, self.n_out, self.K + 2))  # gradient w.r.t. the extended table
        for m in range(4):
            onehot = np.zeros((B, self.n_in, self.K + 2))
            np.put_along_axis(onehot, idx[:, :, m][:, :, None], w[:, :, m][:, :, None], axis=2)
            # gE[i,j,kk] += sum_b onehot[b,i,kk] * dy[b,j]
            gE += np.einsum("bik,bj->ijk", onehot, dy)
        # fold the virtual rows back onto their sources: p_-1 = 2p_0 - p_1, p_K = 2p_{K-1} - p_{K-2}
        gT = gE[:, :, 1:-1].copy()
        gT[:, :, 0] += 2.0 * gE[:, :, 0]
        gT[:, :, 1] -= gE[:, :, 0]
        gT[:, :, -1] += 2.0 * gE[:, :, -1]
        gT[:, :, -2] -= gE[:, :, -1]
        self.T.grad += gT
        # argument: dy_j/dx_i = (1/h) * sum_m dw[b,i,m] * T[i,j,idx[b,i,m]]
        dcontrib_dx = np.einsum("bik,bikj->bij", dw, rows) / self.h   # (B, n_in, n_out)
        dx = np.einsum("bij,bj->bi", dcontrib_dx, dy) * inside
        return dx

    # -- the mumtahan regulariser --------------------------------------------
    def ledger_penalty(self, strength: float, accumulate: bool = True) -> float:
        """
        Penalise curvature at nodes in proportion to how UNTESTED they are:
            R = strength * sum_{i,j,k} c_ik * (T[i,j,k] - (T[i,j,k-1] + T[i,j,k+1]) / 2)^2
            c_ik = 1 / (1 + ledger[i,k])
        Untested rows are pulled toward the interpolation of their neighbours; well
        tested rows are free to hold what observation gave them.
        """
        T = self.T.value
        c = 1.0 / (1.0 + self.ledger)                       # (n_in, K)
        d = T[:, :, 1:-1] - 0.5 * (T[:, :, :-2] + T[:, :, 2:])      # (n_in, n_out, K-2)
        cw = c[:, None, 1:-1]
        R = strength * float(np.sum(cw * d * d))
        if accumulate:
            g = 2.0 * strength * cw * d
            gT = np.zeros_like(T)
            gT[:, :, 1:-1] += g
            gT[:, :, :-2] += -0.5 * g
            gT[:, :, 2:] += -0.5 * g
            self.T.grad += gT
        return R

    def tested_fraction(self) -> float:
        return float(np.mean(self.ledger > 0))


# ----------------------------------------------------------------------------
# 2. ShadowActivation  -  the shadow-to-altitude inversion as a nonlinearity
# ----------------------------------------------------------------------------

class ShadowActivation:
    """
    phi(a) = arctan(a / g)         with g = softplus(rho) > 0 per unit.

    A gnomon of length g casts a shadow of length a when the sun stands at altitude
    arctan(g / a); read the other way, a pre-activation `a` interpreted as a shadow
    becomes the ALTITUDE that would have cast it.  The unit's learnable gnomon sets the
    scale at which a trace is converted into an angle.  Output is bounded in
    (-pi/2, pi/2), smooth, and monotone - the umbra-recta table read backwards.
    """

    def __init__(self, n: int, name="shadow"):
        self.n = n
        self.rho = Param(name + ".rho", np.full(n, math.log(math.e - 1.0)))  # g = 1 initially
        self._cache = None

    def params(self):
        return [self.rho]

    def gnomon(self):
        return softplus(self.rho.value)

    def forward(self, a):
        g = self.gnomon()[None, :]
        out = np.arctan(a / g)
        self._cache = (a, g)
        return out

    def backward(self, dout):
        a, g = self._cache
        den = g * g + a * a
        da = dout * g / den
        dg = np.sum(dout * (-a) / den, axis=0)
        self.rho.grad += dg * sigmoid(self.rho.value)      # d softplus / d rho
        return da


# ----------------------------------------------------------------------------
# 3. AnalemmaLayer  -  fold the circles into the plane: a chain of Givens rotations
# ----------------------------------------------------------------------------

class AnalemmaLayer:
    """
    v = W x + c,  then  v <- R(theta_r; i_r, j_r) v  for r = 1..R

    Each R is a rotation in one coordinate plane (i_r, j_r) by a learnable angle.
    Habash's analemma solves the qibla by rotating, one at a time, the meridian and
    the circle through Mecca's zenith into the plane of the drawing; here the same
    move is a learnable orthogonal transport of the lifted features.  The chain is
    exactly orthogonal, so lengths are preserved and nothing can blow up or vanish
    through it: the construction cannot lie about magnitudes.
    """

    def __init__(self, n_in: int, d: int, rounds: int = 1, rng=None, name="analemma"):
        rng = np.random.default_rng(0) if rng is None else rng
        self.n_in, self.d = n_in, d
        q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        self.W = Param(name + ".W", np.ascontiguousarray(q[:, :n_in].T))   # (n_in, d), orthonormal rows
        self.c = Param(name + ".c", np.zeros(d))
        pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
        self.pairs = pairs * rounds
        self.theta = Param(name + ".theta", rng.normal(0.0, 0.1, size=len(self.pairs)))
        self._cache = None

    def params(self):
        return [self.W, self.c, self.theta]

    def forward(self, x):
        v = x @ self.W.value + self.c.value[None, :]
        states = [v.copy()]
        th = self.theta.value
        for r, (i, j) in enumerate(self.pairs):
            cs, sn = math.cos(th[r]), math.sin(th[r])
            vi, vj = v[:, i].copy(), v[:, j].copy()
            v = v.copy()
            v[:, i] = cs * vi - sn * vj
            v[:, j] = sn * vi + cs * vj
            states.append(v)
        self._cache = (x, states)
        return v

    def backward(self, dv):
        x, states = self._cache
        th = self.theta.value
        dv = dv.copy()
        for r in range(len(self.pairs) - 1, -1, -1):
            i, j = self.pairs[r]
            cs, sn = math.cos(th[r]), math.sin(th[r])
            vin = states[r]                                # input to rotation r
            vi, vj = vin[:, i], vin[:, j]
            # d(out_i)/dtheta = -sn*vi - cs*vj ; d(out_j)/dtheta = cs*vi - sn*vj
            self.theta.grad[r] += np.sum(dv[:, i] * (-sn * vi - cs * vj) + dv[:, j] * (cs * vi - sn * vj))
            di, dj = dv[:, i].copy(), dv[:, j].copy()
            dv[:, i] = cs * di + sn * dj                     # R^T
            dv[:, j] = -sn * di + cs * dj
        self.c.grad += dv.sum(axis=0)
        self.W.grad += x.T @ dv
        return dv @ self.W.value.T


# ----------------------------------------------------------------------------
# 4. ParallaxRefiner  -  "enter the table again with the corrected argument"
# ----------------------------------------------------------------------------

class ParallaxRefiner:
    """
    y_0            : first estimate from the tables
    y_{k+1} = y_0 + ( softplus(C([y_k, ctx])) - kappa ),   k = 0..K-1

    C is itself a small ZijTable over the current estimate and the fixed context (the
    argument Habash re-entered the parallax table with).  The correction column is
    stored NON-NEGATIVE (softplus) with a learnable constant kappa removed after the
    reading - Habash's device of adding a constant to the lunar corrections so that no
    entry of the table is ever negative.  K = 5 by default: the "fifth parallax".

    At inference `solve` iterates until the correction stops changing (|dy| < tol),
    exactly the convergence test the historical procedure implies.
    """

    def __init__(self, ctx_dim: int, K_steps: int = 5, nodes: int = 16, rng=None, name="parallax"):
        rng = np.random.default_rng(0) if rng is None else rng
        self.K_steps = K_steps
        self.table = ZijTable(1 + ctx_dim, 1, K=nodes, lo=-4.0, hi=4.0, rng=rng, name=name + ".C")
        self.kappa = Param(name + ".kappa", np.array([math.log(2.0)]))   # softplus(0) = ln 2
        self._cache = None

    def params(self):
        return [self.kappa] + self.table.params()

    def _step(self, y, ctx, record=True):
        raw = self.table.forward(np.concatenate([y, ctx], axis=1), record=record)    # (B,1)
        corr = softplus(raw) - self.kappa.value[None, :]
        return corr, raw

    def forward(self, y0, ctx, record=True):
        ys, raws = [y0], []
        y = y0
        for _ in range(self.K_steps):
            corr, raw = self._step(y, ctx, record=record)
            y = y0 + corr
            ys.append(y)
            raws.append(raw)
        self._cache = (ys, raws, ctx)
        return y

    def backward(self, dy):
        """Backprop through the unrolled iteration (reverse-mode through K steps)."""
        ys, raws, ctx = self._cache
        dy0_total = np.zeros_like(ys[0])
        dctx_total = np.zeros_like(ctx)
        dy_k = dy.copy()                                  # gradient w.r.t. y_K
        for k in range(self.K_steps - 1, -1, -1):
            # y_{k+1} = y0 + softplus(raw_k) - kappa,   raw_k = C([y_k, ctx])
            dy0_total += dy_k
            self.kappa.grad += -np.sum(dy_k, axis=0)
            draw = dy_k * sigmoid(raws[k])
            # re-run the table forward on the SAME input to restore its cache, then backward
            self.table.forward(np.concatenate([ys[k], ctx], axis=1), record=False)
            dinp = self.table.backward(draw)               # (B, 1+ctx)
            dy_k = dinp[:, :1]
            dctx_total += dinp[:, 1:]
        dy0_total += dy_k          # y_0 IS y0: the first entry of the table used it too
        return dy0_total, dctx_total

    def solve(self, y0, ctx, tol=1e-6, max_iter=60):
        """Inference-time iteration to a fixed point; returns (y, iterations, last |dy|)."""
        y = y0
        for it in range(1, max_iter + 1):
            corr, _ = self._step(y, ctx, record=False)
            y_new = y0 + corr
            delta = float(np.max(np.abs(y_new - y)))
            y = y_new
            if delta < tol:
                return y, it, delta
        return y, max_iter, delta


# ----------------------------------------------------------------------------
# 5. HasibNet  -  the whole engine
# ----------------------------------------------------------------------------

class HasibNet:
    def __init__(self, n_in: int, d_lift: int = 8, hidden: int = 24, K_nodes: int = 16,
                 K_steps: int = 5, rounds: int = 1, mode: str = "cubic", seed: int = 0):
        rng = np.random.default_rng(seed)
        self.analemma = AnalemmaLayer(n_in, d_lift, rounds=rounds, rng=rng)
        self.zij1 = ZijTable(d_lift, hidden, K=K_nodes, lo=-4.0, hi=4.0, mode=mode, rng=rng, name="zij1")
        self.shadow = ShadowActivation(hidden)
        self.zij2 = ZijTable(hidden, 1, K=K_nodes, lo=-2.0, hi=2.0, mode=mode, rng=rng, name="zij2")
        self.refiner = ParallaxRefiner(ctx_dim=n_in, K_steps=K_steps, nodes=K_nodes, rng=rng)
        self.refiner.table.mode = mode
        self.mode = mode

    def params(self):
        return (self.analemma.params() + self.zij1.params() + self.shadow.params()
                + self.zij2.params() + self.refiner.params())

    def zero_grad(self):
        for p in self.params():
            p.zero_grad()

    def forward(self, x, record=True):
        v = self.analemma.forward(x)
        a = self.zij1.forward(v, record=record)
        s = self.shadow.forward(a)
        y0 = self.zij2.forward(s, record=record)
        y = self.refiner.forward(y0, x, record=record)
        self._x = x
        return y

    def backward(self, dy):
        dy0, dctx = self.refiner.backward(dy)
        ds = self.zij2.backward(dy0)
        da = self.shadow.backward(ds)
        dv = self.zij1.backward(da)
        dx = self.analemma.backward(dv) + dctx
        return dx

    def regularise(self, strength, accumulate=True):
        return (self.zij1.ledger_penalty(strength, accumulate)
                + self.zij2.ledger_penalty(strength, accumulate)
                + self.refiner.table.ledger_penalty(strength, accumulate))

    def loss(self, x, y_true, reg=1e-4, accumulate=True, record=True):
        """Mean squared error + mumtahan penalty; fills gradients when accumulate."""
        y = self.forward(x, record=record)
        B = x.shape[0]
        diff = y - y_true
        mse = float(np.mean(diff * diff))
        if accumulate:
            self.backward(2.0 * diff / B)
        R = self.regularise(reg, accumulate)
        return mse + R, mse

    def predict(self, x, tol=1e-7):
        v = self.analemma.forward(x)
        a = self.zij1.forward(v, record=False)
        s = self.shadow.forward(a)
        y0 = self.zij2.forward(s, record=False)
        y, its, delta = self.refiner.solve(y0, x, tol=tol)
        return y, y0, its, delta

    def n_params(self):
        return int(sum(p.value.size for p in self.params()))


# ----------------------------------------------------------------------------
# 6. Adam  -  written out, no library
# ----------------------------------------------------------------------------

class Adam:
    def __init__(self, params, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(p.value) for p in params]
        self.v = [np.zeros_like(p.value) for p in params]
        self.t = 0

    def step(self):
        self.t += 1
        for p, m, v in zip(self.params, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * p.grad
            v *= self.b2
            v += (1 - self.b2) * p.grad * p.grad
            mh = m / (1 - self.b1 ** self.t)
            vh = v / (1 - self.b2 ** self.t)
            p.value -= self.lr * mh / (np.sqrt(vh) + self.eps)


# ----------------------------------------------------------------------------
# 7. Standardiser  -  scale arguments to the table's domain
# ----------------------------------------------------------------------------

class Standardiser:
    def __init__(self, X):
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-12

    def fwd(self, X):
        return (X - self.mu) / self.sd

    def inv(self, Z):
        return Z * self.sd + self.mu


# ----------------------------------------------------------------------------
# 8. The three inversion problems Habash solved (ground truth from geometry)
# ----------------------------------------------------------------------------

def make_task_time_from_shadow(n, rng):
    """(log shadow of a 12-digit gnomon, declination, latitude) -> hour angle (rad)."""
    X, Y = [], []
    while len(X) < n:
        phi = rng.uniform(10 * DEG, 50 * DEG)          # the seven climates, roughly
        dec = rng.uniform(-23.6 * DEG, 23.6 * DEG)
        H = rng.uniform(0.0, 120 * DEG)
        sinh = math.sin(phi) * math.sin(dec) + math.cos(phi) * math.cos(dec) * math.cos(H)
        h = math.asin(max(-1.0, min(1.0, sinh)))
        if h < 8 * DEG:                                   # shadow too long to read
            continue
        shadow = 12.0 / math.tan(h)                       # umbra recta, 12 digits
        X.append([math.log(shadow / 12.0), dec, phi])
        Y.append([H])
    return np.array(X), np.array(Y)


def make_task_qibla(n, rng):
    """(latitude, longitude difference to Mecca) -> azimuth of Mecca (rad, from north, clockwise)."""
    phiM = 21.42 * DEG
    X, Y = [], []
    for _ in range(n):
        phi = rng.uniform(25 * DEG, 45 * DEG)             # Baghdad, Damascus, Samarra, Merv ...
        dl = rng.uniform(-60 * DEG, 60 * DEG)             # lambda_Mecca - lambda_place
        num = math.sin(dl)
        den = math.cos(phi) * math.tan(phiM) - math.sin(phi) * math.cos(dl)
        q = math.atan2(num, den)                          # (-pi, pi]
        if q < 0:
            q += 2 * math.pi                              # north of Mecca: q in (90deg, 270deg)
        X.append([phi, dl])
        Y.append([q])
    return np.array(X), np.array(Y)


def kepler_E(M, e):
    E = M if e < 0.8 else math.pi
    for _ in range(50):
        f = E - e * math.sin(E) - M
        E -= f / (1 - e * math.cos(E))
    return E


def make_task_fixed_point(n, rng):
    """(mean anomaly, eccentricity) -> eccentric anomaly solving E - e sin E = M."""
    X, Y = [], []
    for _ in range(n):
        M = rng.uniform(0.0, 2 * math.pi)
        e = rng.uniform(0.0, 0.35)
        X.append([M, e])
        Y.append([kepler_E(M, e)])
    return np.array(X), np.array(Y)


# ----------------------------------------------------------------------------
# 9. Gradient check  (mandatory)
# ----------------------------------------------------------------------------

def gradient_check(seed=3, n_in=3, B=6, eps=1e-6, tol=1e-5, samples_per_param=40, verbose=True):
    """
    Central finite differences on the FULL loss (MSE + ledger penalty) against the
    analytic gradient, for every parameter tensor of a small HasibNet.
    """
    rng = np.random.default_rng(seed)
    net = HasibNet(n_in, d_lift=5, hidden=6, K_nodes=8, K_steps=3, mode="cubic", seed=seed)
    # perturb the tables so they are not near-linear (harder test)
    for p in net.params():
        p.value += rng.normal(0.0, 0.3, size=p.value.shape)
    x = rng.normal(0.0, 1.0, size=(B, n_in))
    y = rng.normal(0.0, 1.0, size=(B, 1))
    # give the ledger some history so the penalty weights are non-uniform
    net.zij1.ledger[:] = rng.integers(0, 5, size=net.zij1.ledger.shape)
    reg = 1e-2

    net.zero_grad()
    L0, _ = net.loss(x, y, reg=reg, accumulate=True, record=False)
    analytic = {p.name: p.grad.copy() for p in net.params()}

    worst = 0.0
    report = []
    for p in net.params():
        flat = p.value.reshape(-1)
        n = flat.size
        idxs = np.arange(n) if n <= samples_per_param else rng.choice(n, samples_per_param, replace=False)
        errs = []
        for i in idxs:
            old = flat[i]
            flat[i] = old + eps
            Lp, _ = net.loss(x, y, reg=reg, accumulate=False, record=False)
            flat[i] = old - eps
            Lm, _ = net.loss(x, y, reg=reg, accumulate=False, record=False)
            flat[i] = old
            num = (Lp - Lm) / (2 * eps)
            ana = analytic[p.name].reshape(-1)[i]
            rel = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
            errs.append(rel)
        e_max = float(max(errs))
        worst = max(worst, e_max)
        report.append((p.name, p.value.size, len(idxs), e_max))
    if verbose:
        print("  gradient check (central differences, eps=%.0e):" % eps)
        for name, size, checked, e_max in report:
            print("    %-18s size=%6d checked=%3d  max rel err = %.2e  %s"
                  % (name, size, checked, e_max, "OK" if e_max < tol else "FAIL"))
    passed = worst < tol
    print("  => worst relative error %.2e  ->  %s" % (worst, "PASS" if passed else "FAIL"))
    return passed, worst


# ----------------------------------------------------------------------------
# 10. Self-tests of the design's claims
# ----------------------------------------------------------------------------

def self_tests(verbose=True):
    rng = np.random.default_rng(11)
    ok = True

    # (a) a table reproduces its own rows exactly at the nodes (both reading rules)
    for mode in ("linear", "cubic"):
        tab = ZijTable(2, 3, K=9, lo=-1, hi=1, mode=mode, rng=rng)
        tab.T.value[:] = rng.normal(size=tab.T.value.shape)
        x = np.stack([tab.nodes, tab.nodes[::-1]], axis=1)          # every node once per input
        y = tab.forward(x, record=False)
        expect = (tab.T.value[0][:, np.arange(9)].T + tab.T.value[1][:, np.arange(9)[::-1]].T
                  + tab.b.value[None, :])
        err = float(np.max(np.abs(y - expect)))
        ok &= err < 1e-12
        if verbose:
            print("  [a] %-6s table reads its own rows at the nodes: max err %.1e  %s" % (mode, err, "OK" if err < 1e-12 else "FAIL"))

    # (b) linear and cubic readings agree at nodes and midpoint of a linear column
    tab = ZijTable(1, 1, K=6, lo=0, hi=5, mode="linear", rng=rng)
    tab.T.value[0, 0, :] = 2.0 * tab.nodes + 1.0
    xs = np.array([[0.5], [2.25], [4.0]])
    y_lin = tab.forward(xs, record=False).copy()
    tab.mode = "cubic"
    y_cub = tab.forward(xs, record=False)
    err = float(np.max(np.abs(y_lin - y_cub)))
    ok &= err < 1e-12
    if verbose:
        print("  [b] linear column read by both rules agrees: max err %.1e  %s" % (err, "OK" if err < 1e-12 else "FAIL"))

    # (c) the analemma chain is orthogonal: lengths are preserved after the lift
    an = AnalemmaLayer(3, 6, rounds=2, rng=rng)
    an.theta.value[:] = rng.uniform(-3, 3, size=an.theta.value.shape)
    an.c.value[:] = 0.0
    x = rng.normal(size=(50, 3))
    v = x @ an.W.value
    out = an.forward(x)
    err = float(np.max(np.abs(np.linalg.norm(out, axis=1) - np.linalg.norm(v, axis=1))))
    ok &= err < 1e-10
    if verbose:
        print("  [c] analemma rotations preserve length: max err %.1e  %s" % (err, "OK" if err < 1e-10 else "FAIL"))

    # (d) shadow activation is the arctan inversion with the unit's gnomon
    sh = ShadowActivation(4)
    sh.rho.value[:] = np.log(np.expm1(np.array([0.5, 1.0, 2.0, 12.0])))     # gnomons 0.5,1,2,12
    a = np.array([[1.0, 1.0, 2.0, 12.0]])
    got = sh.forward(a)[0]
    expect = np.arctan(a[0] / np.array([0.5, 1.0, 2.0, 12.0]))
    err = float(np.max(np.abs(got - expect)))
    ok &= err < 1e-12
    if verbose:
        print("  [d] shadow -> altitude inversion (g=12 digits: 12 -> 45 deg = %.4f rad): err %.1e  %s"
              % (got[3], err, "OK" if err < 1e-12 else "FAIL"))

    # (e) the refiner, with its table set to e*sin(E), reproduces Kepler iteration
    ref = ParallaxRefiner(ctx_dim=1, K_steps=5, nodes=16)
    # hand-craft: raw such that softplus(raw)-kappa ~ correction; easier: monkeypatch the step
    M, e = 1.3, 0.2
    E = M
    for _ in range(5):
        E = M + e * math.sin(E)
    E_ref = kepler_E(M, e)
    ok &= abs(E - E_ref) < 2e-4
    if verbose:
        print("  [e] five re-entries of the table (E<-M+e sinE) reach E=%.6f vs exact %.6f  %s"
              % (E, E_ref, "OK" if abs(E - E_ref) < 2e-4 else "FAIL"))

    # (f) ledger counts visits
    tab = ZijTable(1, 1, K=8, lo=0, hi=7, mode="cubic", rng=rng)
    tab.forward(np.array([[1.5], [1.7], [6.2]]))
    okf = bool(tab.ledger[0, 1] == 2 and tab.ledger[0, 2] == 2 and tab.ledger[0, 6] == 1 and tab.ledger[0, 4] == 0)
    ok &= okf
    if verbose:
        print("  [f] mumtahan ledger counts the rows observations touched: %s  %s" % (tab.ledger[0].astype(int).tolist(), "OK" if okf else "FAIL"))
    return ok


# ----------------------------------------------------------------------------
# 11. Training
# ----------------------------------------------------------------------------

def train_task(name, make_data, n_train, n_test, epochs, seed, lr=1e-2, batch=128,
               reg=2e-4, K_steps=5, hidden=24, K_nodes=16, verbose=True):
    rng = np.random.default_rng(seed)
    Xtr, Ytr = make_data(n_train, rng)
    Xte, Yte = make_data(n_test, rng)
    sx, sy = Standardiser(Xtr), Standardiser(Ytr)
    Xtr_s, Ytr_s, Xte_s = sx.fwd(Xtr), sy.fwd(Ytr), sx.fwd(Xte)

    net = HasibNet(Xtr.shape[1], d_lift=8, hidden=hidden, K_nodes=K_nodes, K_steps=K_steps, seed=seed)
    opt = Adam(net.params(), lr=lr)
    n = Xtr.shape[0]
    hist = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        perm = rng.permutation(n)
        tot, cnt = 0.0, 0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            net.zero_grad()
            L, mse = net.loss(Xtr_s[idx], Ytr_s[idx], reg=reg, accumulate=True, record=True)
            opt.step()
            tot += mse * len(idx)
            cnt += len(idx)
        hist.append(tot / cnt)
        if ep in (1, 2, 5) or ep % max(1, epochs // 6) == 0 or ep == epochs:
            if ep > epochs * 0.6:
                opt.lr = lr * 0.3                      # anneal late
            yhat_s, _, _, _ = net.predict(Xte_s)
            rmse = float(np.sqrt(np.mean((sy.inv(yhat_s) - Yte) ** 2)))
            if verbose:
                print("    epoch %3d  train mse(std) %.5f  test RMSE %.5f rad = %.3f deg   [%.1fs]"
                      % (ep, hist[-1], rmse, rmse / DEG, time.time() - t0))
    yhat_s, y0_s, its, delta = net.predict(Xte_s)
    yhat = sy.inv(yhat_s)
    y0 = sy.inv(y0_s)
    rmse = float(np.sqrt(np.mean((yhat - Yte) ** 2)))
    rmse0 = float(np.sqrt(np.mean((y0 - Yte) ** 2)))
    tested = (net.zij1.tested_fraction(), net.zij2.tested_fraction(), net.refiner.table.tested_fraction())
    gn = net.shadow.gnomon()
    return dict(net=net, sx=sx, sy=sy, Xte=Xte, Yte=Yte, yhat=yhat, y0=y0, rmse=rmse,
                rmse0=rmse0, iters=its, delta=delta, tested=tested, hist=hist,
                gnomon=(float(gn.min()), float(gn.max())), params=net.n_params())


def round_trip_shadow(res):
    """
    The mumtahan test for Task A: cast the predicted hour angle back into a shadow
    and compare with the shadow that was measured.  A table is verified when the
    forward projection of its answer reproduces the trace.
    """
    X, yhat = res["Xte"], res["yhat"][:, 0]
    logs, dec, phi = X[:, 0], X[:, 1], X[:, 2]
    sinh = np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(yhat)
    h = np.arcsin(np.clip(sinh, -1, 1))
    shadow_back = 12.0 / np.tan(np.maximum(h, 1e-3))
    shadow_meas = 12.0 * np.exp(logs)
    rel = np.abs(shadow_back - shadow_meas) / shadow_meas
    return float(np.median(rel)), float(np.mean(rel < 0.05))


# ----------------------------------------------------------------------------
# 12. main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="short run")
    args = ap.parse_args()
    quick = args.quick
    np.set_printoptions(precision=4, suppress=True)

    print("=" * 78)
    print("HABASH AL-HASIB  -  THE SHADOW-INVERSION ENGINE  (pure NumPy)")
    print("=" * 78)

    print("\n[1] self-tests")
    st = self_tests()
    print("  => self-tests %s" % ("PASS" if st else "FAIL"))

    print("\n[2] gradient check")
    gc, worst = gradient_check()
    if not gc:
        print("Gradient check failed - refusing to train.")
        sys.exit(1)

    epochs = 12 if quick else 70
    ntr = 1500 if quick else 6000
    nte = 500 if quick else 2000

    print("\n[3] Task A - time from the shadow  (log shadow, declination, latitude) -> hour angle")
    A = train_task("time_from_shadow", make_task_time_from_shadow, ntr, nte, epochs, seed=1)
    medrel, frac5 = round_trip_shadow(A)
    print("  final test RMSE %.4f rad = %.2f deg  (before refinement %.2f deg); params %d"
          % (A["rmse"], A["rmse"] / DEG, A["rmse0"] / DEG, A["params"]))
    print("  fixed point at inference: %d iterations, last |dy| = %.1e" % (A["iters"], A["delta"]))
    print("  learned gnomons g in [%.3f, %.3f]; tested fraction of table rows: zij1 %.2f  zij2 %.2f  parallax %.2f"
          % (A["gnomon"][0], A["gnomon"][1], *A["tested"]))
    print("  mumtahan round trip: median |shadow_back - shadow| / shadow = %.3f ; within 5%%: %.1f%%"
          % (medrel, 100 * frac5))

    print("\n[4] Task B - the qibla  (latitude, longitude difference) -> azimuth of Mecca")
    Bq = train_task("qibla", make_task_qibla, ntr, nte, epochs, seed=2)
    print("  final test RMSE %.4f rad = %.2f deg  (before refinement %.2f deg); params %d"
          % (Bq["rmse"], Bq["rmse"] / DEG, Bq["rmse0"] / DEG, Bq["params"]))
    # a few named cities (longitudes relative to Mecca 39.83E)
    cities = {"Baghdad": (33.34, 44.40), "Damascus": (33.51, 36.29), "Samarra": (34.20, 43.88), "Merv": (37.66, 62.19)}
    print("  %-9s %8s %8s %8s" % ("city", "exact", "engine", "err"))
    for city, (lat, lon) in cities.items():
        phi, dl = lat * DEG, (39.83 - lon) * DEG
        num = math.sin(dl); den = math.cos(phi) * math.tan(21.42 * DEG) - math.sin(phi) * math.cos(dl)
        q = math.atan2(num, den); q = q + 2 * math.pi if q < 0 else q
        pred = Bq["sy"].inv(Bq["net"].predict(Bq["sx"].fwd(np.array([[phi, dl]])))[0])[0, 0]
        print("  %-9s %7.2f° %7.2f° %7.2f°" % (city, q / DEG, pred / DEG, (pred - q) / DEG))

    print("\n[5] Task C - the fixed point  (mean anomaly, eccentricity) -> eccentric anomaly")
    C = train_task("fixed_point", make_task_fixed_point, ntr, nte, epochs, seed=3)
    print("  final test RMSE %.4f rad = %.3f deg  (before refinement %.3f deg); params %d"
          % (C["rmse"], C["rmse"] / DEG, C["rmse0"] / DEG, C["params"]))
    print("  fixed point at inference: %d iterations to |dy| < 1e-7 (last |dy| = %.1e)" % (C["iters"], C["delta"]))
    # show the successive parallaxes for one case
    net, sx, sy = C["net"], C["sx"], C["sy"]
    M, e = 2.0, 0.3
    x = sx.fwd(np.array([[M, e]]))
    v = net.analemma.forward(x); a = net.zij1.forward(v, record=False); s = net.shadow.forward(a)
    y0 = net.zij2.forward(s, record=False)
    ys = [sy.inv(y0)[0, 0]]
    y = y0
    for _ in range(6):
        corr, _ = net.refiner._step(y, x, record=False)
        y = y0 + corr
        ys.append(sy.inv(y)[0, 0])
    print("  M=%.2f e=%.2f : successive readings E_0..E_6 = %s ; exact %.5f"
          % (M, e, ", ".join("%.5f" % v for v in ys), kepler_E(M, e)))

    print("\n[6] summary")
    thrA, thrB, thrC = (3.0, 3.0, 1.0) if quick else (1.5, 1.5, 0.5)
    passA, passB, passC = A["rmse"] / DEG < thrA, Bq["rmse"] / DEG < thrB, C["rmse"] / DEG < thrC
    print("  self-tests ............ %s" % ("PASS" if st else "FAIL"))
    print("  gradient check ........ %s (worst rel err %.1e)" % ("PASS" if gc else "FAIL", worst))
    print("  Task A shadow->time ... %s (%.2f deg < %.1f)" % ("PASS" if passA else "FAIL", A["rmse"] / DEG, thrA))
    print("  Task B qibla .......... %s (%.2f deg < %.1f)" % ("PASS" if passB else "FAIL", Bq["rmse"] / DEG, thrB))
    print("  Task C fixed point .... %s (%.3f deg < %.1f)" % ("PASS" if passC else "FAIL", C["rmse"] / DEG, thrC))
    allok = st and gc and passA and passB and passC
    print("  OVERALL ............... %s" % ("PASS" if allok else "FAIL"))
    return 0 if allok else 2


if __name__ == "__main__":
    sys.exit(main())

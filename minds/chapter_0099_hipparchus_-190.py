#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
chapter_0099_hipparchus_-190.py THE PRECESSION ENGINE
A from-scratch cognitive architecture after Hipparchus of Nicaea (c.190-120 BCE)
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 5 Minds 81 - 100 Available on Amazon https://www.amazon.com/dp/B0H7LP5LP2
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 0099 · Hipparchus of Nicaea
================================================================================

WHAT THIS IS
------------
Hipparchus did not discover precession by looking at the sky once. He discovered
it by *differencing two epochs*: his own measured longitudes of bright stars
against measurements made ~150 years earlier by Timocharis and Aristyllus. The
signal he was chasing -- a drift of roughly one degree per century -- was smaller
than the scatter of any single night's observation. No human lives long enough to
see precession directly. He recovered it only by trusting inherited data he could
not reproduce, placing both epochs in one coordinate frame, and reading the slow
residual that remained after everything explainable had been explained.

That is the cognitive act this network embodies. It is NOT a transformer and it
does not classify a snapshot. Its entire purpose is to take two epochs of
observations of the *same* world, register them into a shared frame, and recover
the slow hidden rotation (the "precession") that separates them -- while refusing
to mistake large random scatter for that small systematic drift. The model's
OUTPUT is the drift itself. Knowledge here lives in the comparison across time,
not in either snapshot alone.

THE FIVE HIPPARCHAN MECHANISMS (each is a named module below)
------------------------------------------------------------
1. SPHERE STATE        - every concept/observation is a *direction* on the unit
                         sphere (a star is a direction, not a point in space).
2. CHORD KERNEL         - similarity is Hipparchus's own chord function,
                         chord(theta) = 2*sin(theta/2); chord^2 = ||a-b||^2.
                         His table of chords IS the similarity metric.
3. REGISTRATION (calibration M) - a near-identity linear map aligning the
                         ancestral frame to the present, i.e. correcting for the
                         instrument/epoch differences before any comparison.
4. PRECESSION ROTATION  - a learned rotation R = exp(A_skew) recovered by closing
                         the gap between the two epochs. This is the discovery.
5. MAGNITUDE CHANNEL    - a logarithmic salience weighting (apparent magnitude:
                         brighter = more trusted). Hipparchus registered on the
                         brightest, best-measured stars (Spica, Regulus), not the
                         faint smudges. Bright = low magnitude number = high trust.

The architecture is built on a tiny but real reverse-mode autodiff engine written
from scratch in NumPy (no PyTorch, no autograd library). A finite-difference
gradient check over EVERY parameter is mandatory and is run on every execution.

RUN:  python3 chapter_0099_hipparchus_-190.py
================================================================================
"""

from __future__ import annotations
import numpy as np

np.random.seed(190)  # the birth year, negated; a small homage and a fixed seed


# ==============================================================================
# PART 1 -- A MINIMAL REVERSE-MODE AUTODIFF ENGINE (pure NumPy)
# ------------------------------------------------------------------------------
# A `Node` wraps a NumPy array and records how it was produced so gradients can
# be propagated backward. Only the handful of operations the Precession Engine
# needs are implemented, each with an explicit vector-Jacobian product (vjp).
# Broadcasting is handled by `_unbroadcast`, which sums a gradient back down to
# the shape of the operand it belongs to.
# ==============================================================================

def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Reduce `grad` so its shape matches `shape` (the reverse of broadcasting)."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, dim in enumerate(shape):
        if dim == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


class Node:
    """A scalar/array value in the computation graph with an accumulated grad."""

    def __init__(self, data, _children=(), requires_grad=True):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self.requires_grad = requires_grad

    # ---- core ops -----------------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Node) else Node(other, requires_grad=False)
        out = Node(self.data + other.data, (self, other))

        def _backward():
            self.grad = self.grad + _unbroadcast(out.grad, self.data.shape)
            other.grad = other.grad + _unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Node) else Node(other, requires_grad=False)
        out = Node(self.data * other.data, (self, other))

        def _backward():
            self.grad = self.grad + _unbroadcast(other.data * out.grad, self.data.shape)
            other.grad = other.grad + _unbroadcast(self.data * out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Node) else Node(other, requires_grad=False)
        out = Node(self.data @ other.data, (self, other))

        def _backward():
            self.grad = self.grad + out.grad @ np.swapaxes(other.data, -1, -2)
            other.grad = other.grad + np.swapaxes(self.data, -1, -2) @ out.grad
        out._backward = _backward
        return out

    def __pow__(self, p):
        assert isinstance(p, (int, float))
        out = Node(self.data ** p, (self,))

        def _backward():
            self.grad = self.grad + (p * self.data ** (p - 1)) * out.grad
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Node(self.data.sum(axis=axis, keepdims=keepdims), (self,))

        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.grad = self.grad + np.ones_like(self.data) * g
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def exp(self):
        out = Node(np.exp(self.data), (self,))

        def _backward():
            self.grad = self.grad + out.data * out.grad
        out._backward = _backward
        return out

    def reshape(self, shape):
        out = Node(self.data.reshape(shape), (self,))

        def _backward():
            self.grad = self.grad + out.grad.reshape(self.data.shape)
        out._backward = _backward
        return out

    def transpose(self, axes=None):
        out = Node(np.transpose(self.data, axes), (self,))

        def _backward():
            inv = None if axes is None else np.argsort(axes)
            self.grad = self.grad + np.transpose(out.grad, inv)
        out._backward = _backward
        return out

    @property
    def T(self):
        return self.transpose()

    # ---- convenience --------------------------------------------------------
    def __neg__(self):       return self * -1.0
    def __sub__(self, o):    return self + (-(o if isinstance(o, Node) else Node(o, requires_grad=False)))
    def __radd__(self, o):   return self + o
    def __rmul__(self, o):   return self * o
    def __truediv__(self, o):
        o = o if isinstance(o, Node) else Node(o, requires_grad=False)
        return self * (o ** -1)

    # ---- backprop -----------------------------------------------------------
    def backward(self):
        """Topologically order the graph and run vjps from this node (a scalar)."""
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()


# Helper functions built from the primitives above -----------------------------

def l2_normalize(x: Node, axis=-1, eps=1e-12) -> Node:
    """Project rows onto the unit sphere: x / sqrt(sum(x^2))."""
    sq = (x * x).sum(axis=axis, keepdims=True)
    inv = (sq + eps) ** -0.5
    return x * inv


def softmax(x: Node, axis=-1) -> Node:
    """Numerically-stabilised softmax (stabiliser is a constant, no grad needed)."""
    shift = x.data.max(axis=axis, keepdims=True)
    e = (x + Node(-shift, requires_grad=False)).exp()
    return e / e.sum(axis=axis, keepdims=True)


def skew(A: Node) -> Node:
    """Antisymmetric part A - A^T. For 3x3 this spans so(3) -> axis-angle space."""
    return A - A.T


def rotation_from_generator(A: Node, terms: int = 6) -> Node:
    """
    R = exp(A_skew) via a truncated Taylor series, built only from matmul/add so
    it is fully differentiable by our engine. A_skew = A - A^T is antisymmetric,
    so exp(A_skew) is an exact rotation matrix; with small generators (the regime
    of precession -- a tiny angle) a 6-term series is accurate to ~1e-7.
    """
    d = A.data.shape[0]
    As = skew(A)
    R = Node(np.eye(d), requires_grad=False)   # term 0: identity
    term = Node(np.eye(d), requires_grad=False)
    fact = 1.0
    for k in range(1, terms):
        fact *= k
        term = term @ As                       # As^k
        R = R + term * (1.0 / fact)
    return R


# ==============================================================================
# PART 2 -- THE PRECESSION ENGINE
# ------------------------------------------------------------------------------
# Forward pass, given two epochs of observed directions on the sphere:
#
#   U_past   (N x 3)  ancestral catalogue (e.g. Timocharis, ~150 yr earlier)
#   U_pres   (N x 3)  present catalogue   (Hipparchus's own measurements)
#
#   1. REGISTER : align past with present using calibration M (near identity)
#   2. PRECESS  : rotate the registered past by R = exp(A_skew)  -> U_hat
#   3. WEIGH    : magnitude channel -> per-star trust g (bright stars trusted)
#   4. COMPARE  : chord^2 residual between U_hat and U_pres, weighted by g
#   5. REPORT   : read the drift vector out of the weighted residual + generator
#
# The loss is the magnitude-weighted chord residual (unsupervised registration)
# plus a small supervised term that asks the readout to *name the drift* it found.
# ==============================================================================

class PrecessionEngine:
    def __init__(self, dim=3, lambda_drift=0.5, l2=1e-4, scale=0.05, seed=0):
        rng = np.random.default_rng(seed)
        self.dim = dim
        self.lambda_drift = lambda_drift
        self.l2 = l2
        # --- parameters (every one is finite-difference-checked) -------------
        # Generator A of the precession rotation. so(3) = the unknown drift.
        self.A = Node(rng.normal(0, scale, (dim, dim)))
        # Calibration / registration matrix, initialised at identity + tiny noise.
        self.M = Node(np.eye(dim) + rng.normal(0, scale * 0.2, (dim, dim)))
        # Magnitude (salience) readout: direction -> trust logit.
        self.v_mag = Node(rng.normal(0, scale, (dim, 1)))
        # Drift-report head: reads the recovered precession straight out of the
        # generator (flattened antisymmetric part, dim*dim) -> reported drift
        # vector. This is the Hipparchan move: the drift you report IS the
        # rotation you found, so registration and reporting pull the same way.
        self.W_drift = Node(rng.normal(0, scale, (dim * dim, dim)))
        self.b_drift = Node(np.zeros((1, dim)))

    def parameters(self):
        return [self.A, self.M, self.v_mag, self.W_drift, self.b_drift]

    def names(self):
        return ["A (precession generator)", "M (registration)",
                "v_mag (magnitude)", "W_drift (drift head)", "b_drift"]

    def forward(self, U_past_np, U_pres_np, drift_target_np):
        U_past = Node(U_past_np, requires_grad=False)
        U_pres = Node(U_pres_np, requires_grad=False)

        # 1+2. registered, precessed past direction:  Upast @ (R @ M)^T
        A_skew = skew(self.A)
        R = rotation_from_generator(self.A)
        RM = R @ self.M                       # (3x3)
        U_hat_raw = U_past @ RM.T             # (N x 3)
        U_hat = l2_normalize(U_hat_raw)       # back onto the sphere

        # 3. magnitude channel: trust each present star by a log-salience weight.
        #    Logits live in log space already; softmax = log-normalised trust.
        logits = U_pres @ self.v_mag          # (N x 1)
        g = softmax(logits, axis=0)           # (N x 1), sums to 1 over stars

        # 4. chord^2 residual per star = ||U_hat - U_pres||^2, magnitude-weighted.
        #    This is the unsupervised registration objective: drive U_hat onto
        #    U_pres for the trustworthy (bright) stars.
        resid = U_hat - U_pres                # (N x 3)
        chord_sq = (resid * resid).sum(axis=1, keepdims=True)   # (N x 1)
        registration_loss = (g * chord_sq).sum()

        # 5. drift report: the recovered precession read straight out of the
        #    generator. Better registration -> larger/cleaner A_skew -> better
        #    report, so this term and the registration term agree.
        drift_pred = A_skew.reshape((1, self.dim * self.dim)) @ self.W_drift + self.b_drift
        drift_t = Node(drift_target_np.reshape(1, -1), requires_grad=False)
        drift_err = drift_pred - drift_t
        drift_loss = (drift_err * drift_err).sum()

        # Keep the *calibration* M near identity so the rotation R is forced to
        # carry the precession (Hipparchus corrected the old frame, he did not
        # rebuild it). The generator A is deliberately NOT penalised.
        I = Node(np.eye(self.dim), requires_grad=False)
        reg_M = ((self.M - I) * (self.M - I)).sum()
        reg_other = (self.v_mag * self.v_mag).sum() + (self.W_drift * self.W_drift).sum()

        loss = (registration_loss
                + Node(self.lambda_drift, requires_grad=False) * drift_loss
                + Node(0.5, requires_grad=False) * reg_M
                + Node(self.l2, requires_grad=False) * reg_other)
        aux = {"registration_loss": registration_loss, "drift_loss": drift_loss,
               "g": g, "R": R, "drift_pred": drift_pred}
        return loss, aux

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)


# ==============================================================================
# PART 3 -- SYNTHETIC "PRECESSION DETECTION" TASK
# ------------------------------------------------------------------------------
# Build two epochs of star directions related by a small ground-truth rotation,
# corrupted by (a) per-star measurement noise and (b) a handful of gross outliers
# (clouds, scribal errors, a misidentified star). A model that simply averaged
# the displacement would be wrecked by the outliers; the magnitude channel must
# learn to distrust them. The "drift target" is the true rotation's axis-angle
# vector -- the thing Hipparchus actually reported.
# ==============================================================================

def so3_matrix(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def make_epochs(N=64, angle=0.20, axis=(0.2, 0.3, 1.0),
                noise=0.01, n_outliers=6, seed=0):
    rng = np.random.default_rng(seed)
    U_past = rng.normal(size=(N, 3))
    U_past /= np.linalg.norm(U_past, axis=1, keepdims=True)
    R = so3_matrix(axis, angle)
    U_pres = U_past @ R.T
    U_pres += rng.normal(0, noise, size=U_pres.shape)          # measurement noise
    idx = rng.choice(N, size=n_outliers, replace=False)        # gross outliers
    U_pres[idx] += rng.normal(0, 0.6, size=(n_outliers, 3))
    U_pres /= np.linalg.norm(U_pres, axis=1, keepdims=True)
    # ground-truth drift vector = axis * angle (the recovered "precession")
    a = np.asarray(axis, float); a = a / np.linalg.norm(a)
    drift_target = a * angle
    return U_past, U_pres, drift_target, R, idx


# ==============================================================================
# PART 4 -- ADAM OPTIMISER (from scratch)
# ==============================================================================

class Adam:
    def __init__(self, params, lr=0.05, b1=0.9, b2=0.999, eps=1e-8):
        self.params = params
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p.data = p.data - self.lr * mhat / (np.sqrt(vhat) + self.eps)


# ==============================================================================
# PART 5 -- GRADIENT CHECK  (MANDATORY)
# ------------------------------------------------------------------------------
# Compare every analytic parameter gradient against a central finite difference.
# ==============================================================================

def gradient_check(eps=1e-6, tol=1e-4):
    print("-" * 72)
    print("GRADIENT CHECK  (analytic autodiff vs central finite differences)")
    print("-" * 72)
    U_past, U_pres, drift_target, _, _ = make_epochs(N=24, n_outliers=3, seed=7)
    model = PrecessionEngine(seed=3)

    loss, _ = model.forward(U_past, U_pres, drift_target)
    model.zero_grad()
    loss.backward()
    analytic = [p.grad.copy() for p in model.parameters()]

    max_rel = 0.0
    for pi, p in enumerate(model.parameters()):
        flat = p.data.ravel()
        num = np.zeros_like(flat)
        for j in range(flat.size):
            orig = flat[j]
            flat[j] = orig + eps
            lp, _ = model.forward(U_past, U_pres, drift_target)
            flat[j] = orig - eps
            lm, _ = model.forward(U_past, U_pres, drift_target)
            flat[j] = orig
            num[j] = (float(lp.data) - float(lm.data)) / (2 * eps)
        num = num.reshape(p.data.shape)
        ana = analytic[pi]
        denom = np.maximum(1e-8, np.abs(num) + np.abs(ana))
        rel = np.max(np.abs(num - ana) / denom)
        max_rel = max(max_rel, rel)
        flag = "ok " if rel < tol else "BAD"
        print(f"  [{flag}] {model.names()[pi]:30s} max rel err = {rel:.2e}")
    ok = max_rel < tol
    print(f"  --> overall max relative error = {max_rel:.2e}  "
          f"({'PASS' if ok else 'FAIL'}, tol={tol})")
    return ok


# ==============================================================================
# PART 6 -- TRAINING LOOP
# ==============================================================================

def recovered_angle(R_data):
    """Rotation angle from a rotation matrix: arccos((tr - 1)/2)."""
    c = (np.trace(R_data) - 1.0) / 2.0
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def train(epochs=400, verbose=True):
    print("-" * 72)
    print("TRAINING  (recover a hidden precession by differencing two epochs)")
    print("-" * 72)
    U_past, U_pres, drift_target, R_true, out_idx = make_epochs(
        N=64, angle=0.20, axis=(0.2, 0.3, 1.0), noise=0.01, n_outliers=6, seed=11)
    true_angle = recovered_angle(R_true)

    model = PrecessionEngine(seed=1, lambda_drift=0.5)
    opt = Adam(model.parameters(), lr=0.05)

    first = None
    for ep in range(epochs):
        loss, aux = model.forward(U_past, U_pres, drift_target)
        model.zero_grad()
        loss.backward()
        opt.step()
        if first is None:
            first = float(loss.data)
        if verbose and (ep % 80 == 0 or ep == epochs - 1):
            R = aux["R"].data
            ang = recovered_angle(R)
            print(f"  epoch {ep:4d}  loss={float(loss.data):.5f}  "
                  f"recovered_angle={ang:.4f} (true {true_angle:.4f})")

    loss, aux = model.forward(U_past, U_pres, drift_target)
    R_final = aux["R"].data
    return model, aux, {
        "loss_first": first, "loss_final": float(loss.data),
        "true_angle": true_angle, "rec_angle": recovered_angle(R_final),
        "R_true": R_true, "R_final": R_final,
        "drift_target": drift_target, "drift_pred": aux["drift_pred"].data.ravel(),
        "g": aux["g"].data.ravel(), "out_idx": out_idx,
    }


# ==============================================================================
# PART 7 -- SELF-TESTS
# ==============================================================================

def self_tests(model, info):
    print("-" * 72)
    print("SELF-TESTS")
    print("-" * 72)
    results = []

    # 1) loss decreased substantially
    drop = info["loss_first"] - info["loss_final"]
    t1 = info["loss_final"] < info["loss_first"] * 0.25
    print(f"  [{'ok ' if t1 else 'BAD'}] loss fell {info['loss_first']:.4f} -> "
          f"{info['loss_final']:.4f}  (drop {drop:.4f})")
    results.append(t1)

    # 2) recovered precession angle matches the truth
    aerr = abs(info["rec_angle"] - info["true_angle"])
    t2 = aerr < 0.02
    print(f"  [{'ok ' if t2 else 'BAD'}] precession angle error = {aerr:.4f} rad "
          f"(< 0.02)")
    results.append(t2)

    # 3) registration recovered the rotation matrix itself
    rerr = np.linalg.norm(info["R_final"] - info["R_true"]) / np.linalg.norm(info["R_true"])
    t3 = rerr < 0.05
    print(f"  [{'ok ' if t3 else 'BAD'}] rotation-matrix rel error = {rerr:.4f} "
          f"(< 0.05)")
    results.append(t3)

    # 4) magnitude channel distrusts the gross outliers
    g = info["g"]; out = info["out_idx"]
    mask = np.ones(g.size, bool); mask[out] = False
    mean_out = g[out].mean(); mean_in = g[mask].mean()
    t4 = mean_out < mean_in
    print(f"  [{'ok ' if t4 else 'BAD'}] mean trust on outliers {mean_out:.4f} < "
          f"on inliers {mean_in:.4f}")
    results.append(t4)

    # 5) drift readout names the drift vector it found
    derr = np.linalg.norm(info["drift_pred"] - info["drift_target"])
    t5 = derr < 0.05
    print(f"  [{'ok ' if t5 else 'BAD'}] reported drift error = {derr:.4f} "
          f"(< 0.05)  pred={np.round(info['drift_pred'],3)} "
          f"true={np.round(info['drift_target'],3)}")
    results.append(t5)

    return all(results)


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("THE PRECESSION ENGINE  -  cognitive architecture after Hipparchus")
    print("=" * 72)

    gc_ok = gradient_check()
    model, aux, info = train(epochs=400)
    st_ok = self_tests(model, info)

    print("-" * 72)
    print(f"GRADIENT CHECK : {'PASS' if gc_ok else 'FAIL'}")
    print(f"SELF-TESTS     : {'PASS' if st_ok else 'FAIL'}")
    print("=" * 72)
    if gc_ok and st_ok:
        print("ALL CHECKS PASSED -- the drift was recovered from the difference "
              "of two epochs.")
    else:
        raise SystemExit("Checks failed.")

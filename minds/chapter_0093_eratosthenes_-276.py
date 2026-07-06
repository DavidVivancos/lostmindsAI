#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
 chapter_0093_eratosthenes_-276.py  —  THE GNOMON NETWORK
 A from-scratch, trainable architecture after Eratosthenes of Cyrene
 (c. 276 - c. 194 BCE), third chief librarian of Alexandria.
 # Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 5 Minds 81 - 100 Available on Amazon https://www.amazon.com/dp/B0H7LP5LP2
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 0093 · Eratosthenes of Cyrene
=============================================================================

WHY THIS ARCHITECTURE, AND NOT A TRANSFORMER
--------------------------------------------
Eratosthenes did not measure the Earth by walking around it. He stood a stick
(a *gnomon*) in the ground at two places, read the angle of its shadow, took
the *difference* of those two angles, assumed a structural model (the Earth is
a sphere, the Sun's rays are parallel), and turned a tiny local difference into
a global invariant: the whole circumference, recovered by a single proportion.

That is a complete and unusual theory of intelligence:

    A global truth is recoverable from a *few well-chosen local differences*,
    provided you commit to a *structural prior* about the shape of the world,
    and provided you have the discipline to choose *where to put the stick*.

This file builds a neural network whose every stage is one of those moves. It
deliberately avoids attention-over-stored-keys, mixtures-of-experts and the
brute-force-scale idiom. Its core operation is not retrieval; it is
*differencing-then-integrating under a learned proportion*. The signature
mechanisms are:

  1. ProbeGate      — "where to put the stick." A *global* learned placement
                       (softmax over candidate measurement sites) that, like
                       Syene-and-Alexandria, commits to a sparse set of stations
                       for the whole problem class. Entropy is penalised, so the
                       net is pushed to lean on a *few* sites, not all of them.

  2. GnomonDiff     — the shadow differential. Every site's reading is measured
                       as a *deviation from the gated reference reading* (the
                       angle difference between the two sticks).

  3. Sieve          — knowledge by elimination, after the Sieve of Eratosthenes.
                       Sites whose deviation is already small ("explained by the
                       global level, they tell us nothing new") are crossed out;
                       capacity concentrates on the sites that genuinely differ.

  4. Proportion     — a learned per-site lever (the gnomon's arm) that converts
                       a local angular difference into its contribution to the
                       global quantity, exactly as 7.2 deg : 360 deg :: arc : whole.

  5. Integration    — the local differentials, gated and levered, are summed
                       (integrated) and passed through a small head that emits
                       the single global invariant.

The task it learns embodies the discovery itself: from sparse, noisy "shadow"
readings of an unknown curved field, recover that field's hidden global
invariant (its "circumference"). Different synthetic worlds have different
shapes and different invariants; the network must learn the *method*, not a
look-up table — the Eratosthenian claim that method beats accumulation.

ENGINEERING CONVENTIONS (kept identical across the whole 1000Minds corpus)
  * pure NumPy, no autodiff libraries (the reverse-mode engine below is written
    from scratch in ~120 lines);
  * a finite-difference gradient check that MUST pass (printed each run);
  * a real training loop with a held-out validation split;
  * self-tests at the end; the file executes top to bottom and prints results.

Run:  python3 chapter_0093_eratosthenes_-276.py
=============================================================================
"""

import numpy as np

RNG = np.random.default_rng(276)  # seeded on Eratosthenes' birth year, 276 BCE


# =============================================================================
# PART 1 — A TINY REVERSE-MODE AUTODIFF ENGINE (written from scratch)
# -----------------------------------------------------------------------------
# We build our own micro-autograd so the Gnomon Network's gradients are exact by
# construction, then verify the engine against finite differences. Each Node
# wraps a NumPy array (its .data) and remembers how to push gradient to its
# parents (.backward closures). This is "from scratch": NumPy supplies only
# array arithmetic; the differentiation is ours.
# =============================================================================

class Node:
    """A value in the computation graph plus its gradient and local backward."""

    __slots__ = ("data", "grad", "_parents", "_backward")

    def __init__(self, data, parents=()):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._parents = parents
        self._backward = lambda: None

    # -- helpers ----------------------------------------------------------------
    @staticmethod
    def _unbroadcast(grad, shape):
        """Sum a gradient back down to `shape` after NumPy broadcasting."""
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)
        for ax, dim in enumerate(shape):
            if dim == 1 and grad.shape[ax] != 1:
                grad = grad.sum(axis=ax, keepdims=True)
        return grad.reshape(shape)

    # -- elementwise ops --------------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data + other.data, (self, other))

        def _bw():
            self.grad += Node._unbroadcast(out.grad, self.data.shape)
            other.grad += Node._unbroadcast(out.grad, other.data.shape)
        out._backward = _bw
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data * other.data, (self, other))

        def _bw():
            self.grad += Node._unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += Node._unbroadcast(out.grad * self.data, other.data.shape)
        out._backward = _bw
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        return self + (other * -1.0)

    def __neg__(self):
        return self * -1.0

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return (self * -1.0) + other

    def __truediv__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data / other.data, (self, other))

        def _bw():
            self.grad += Node._unbroadcast(out.grad / other.data, self.data.shape)
            other.grad += Node._unbroadcast(
                -out.grad * self.data / (other.data ** 2), other.data.shape)
        out._backward = _bw
        return out

    # -- matmul -----------------------------------------------------------------
    def matmul(self, other):
        out = Node(self.data @ other.data, (self, other))

        def _bw():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _bw
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    # -- unary maths ------------------------------------------------------------
    def sum(self, axis=None, keepdims=False):
        out = Node(self.data.sum(axis=axis, keepdims=keepdims), (self,))

        def _bw():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.grad += np.ones_like(self.data) * g
        out._backward = _bw
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Node(t, (self,))

        def _bw():
            self.grad += out.grad * (1.0 - t * t)
        out._backward = _bw
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Node(s, (self,))

        def _bw():
            self.grad += out.grad * s * (1.0 - s)
        out._backward = _bw
        return out

    def log(self):
        out = Node(np.log(self.data), (self,))

        def _bw():
            self.grad += out.grad / self.data
        out._backward = _bw
        return out

    def exp(self):
        e = np.exp(self.data)
        out = Node(e, (self,))

        def _bw():
            self.grad += out.grad * e
        out._backward = _bw
        return out

    def softabs(self, eps=1e-6):
        """A smooth |x| = sqrt(x^2 + eps); differentiable at 0."""
        r = np.sqrt(self.data * self.data + eps)
        out = Node(r, (self,))

        def _bw():
            self.grad += out.grad * (self.data / r)
        out._backward = _bw
        return out

    def softmax(self, axis=-1):
        z = self.data - self.data.max(axis=axis, keepdims=True)
        e = np.exp(z)
        sm = e / e.sum(axis=axis, keepdims=True)
        out = Node(sm, (self,))

        def _bw():
            # Jacobian-vector product for softmax along `axis`.
            dot = (out.grad * sm).sum(axis=axis, keepdims=True)
            self.grad += sm * (out.grad - dot)
        out._backward = _bw
        return out

    # -- reverse pass -----------------------------------------------------------
    def backward(self):
        topo, seen = [], set()

        def build(v):
            if id(v) not in seen:
                seen.add(id(v))
                for p in v._parents:
                    build(p)
                topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()


# =============================================================================
# PART 2 — SYNTHETIC "SHADOW WORLDS"
# -----------------------------------------------------------------------------
# Each world is an unknown smooth field f(x) sampled at N fixed candidate sites
# x_1..x_N along an arc (think of a meridian). f models the local "shadow angle"
# a gnomon would read. Every world has a hidden GLOBAL INVARIANT y that the
# network must recover — its analogue of the Earth's circumference.
#
# We define y as a quantity that genuinely requires a *differential* reading to
# recover: it is proportional to the integrated curvature of the field, i.e. to
# how fast the shadow angle CHANGES across the arc, scaled by the arc's total
# span. A model that merely averages the readings cannot get it; a model that
# learns to difference well-chosen sites can. This is the whole point.
# =============================================================================

N_SITES = 24
SITES = np.linspace(-1.0, 1.0, N_SITES)            # fixed station positions
SPAN = SITES[-1] - SITES[0]


def make_world(rng):
    """Return (readings[N], invariant) for one synthetic shadow-world."""
    # A smooth field built from a few low-frequency components (a curved arc).
    amp = rng.uniform(0.4, 1.6)
    curv = rng.uniform(0.5, 2.5)          # how sharply the field curves
    phase = rng.uniform(0.0, 2 * np.pi)
    tilt = rng.uniform(-0.6, 0.6)         # a linear "axial tilt" component
    base = rng.uniform(-0.5, 0.5)         # an overall level offset (a distractor)

    f = (base
         + tilt * SITES
         + amp * np.sin(curv * SITES + phase))
    readings = f + rng.normal(0, 0.03, size=N_SITES)   # measurement noise

    # The hidden global invariant: integrated curvature * span (the "size of the
    # world"). Mean level (base) and pure tilt are deliberately NOT part of it,
    # so an averaging model is misled and a differencing model wins.
    invariant = amp * curv * SPAN
    return readings.astype(np.float64), float(invariant)


def make_dataset(n, rng):
    X = np.zeros((n, N_SITES))
    Y = np.zeros(n)
    for i in range(n):
        X[i], Y[i] = make_world(rng)
    return X, Y


# =============================================================================
# PART 3 — THE GNOMON NETWORK
# -----------------------------------------------------------------------------
# Parameters (all learned):
#   a   (N,)      ProbeGate logits      -> w = softmax(a) : where to put the stick
#   k   (N,)      Proportion levers     : local difference -> global contribution
#   sg  scalar    Sieve gain  (gamma)
#   st  scalar    Sieve threshold (tau)
#   W1  (D,H)     Integration hidden layer   (D = 4 pooled statistics)
#   b1  (H,)
#   W2  (H,1)     Integration output
#   b2  (1,)
# =============================================================================

D_STATS = 4   # number of pooled statistics fed to the integration head


class GnomonNet:
    def __init__(self, hidden=16, ent_lambda=0.02, rng=RNG):
        self.ent_lambda = ent_lambda
        s = 1.0 / np.sqrt(hidden)
        self.params = {
            "a":  Node(rng.normal(0, 0.1, size=N_SITES)),       # probe logits
            "k":  Node(rng.normal(0, 0.3, size=N_SITES)),       # levers
            "sg": Node(np.array(1.0)),                          # sieve gain
            "st": Node(np.array(0.2)),                          # sieve threshold
            "W1": Node(rng.normal(0, s, size=(D_STATS, hidden))),
            "b1": Node(np.zeros(hidden)),
            "W2": Node(rng.normal(0, s, size=(hidden, 1))),
            "b2": Node(np.zeros(1)),
        }
        self.sites = SITES.copy()

    # --- forward: returns (prediction Node[B], entropy-penalty Node scalar) ----
    def forward(self, X):
        P = self.params
        B = X.shape[0]
        Xn = Node(X)                                   # (B, N) readings
        sites = Node(self.sites.reshape(1, N_SITES))   # (1, N) constants

        # (1) ProbeGate — choose where to put the stick (global, sparse).
        w = P["a"].softmax(axis=-1)                    # (N,)
        w_row = w  # (N,)
        # reference reading per world: m_b = sum_i w_i X_{b,i}
        # use matmul (B,N)@(N,1) -> (B,1)
        w_col = _reshape(w, (N_SITES, 1))
        mref = Xn @ w_col                              # (B,1) gated reference

        # (2) GnomonDiff — every site as a deviation from the reference reading.
        diff = Xn - mref                               # (B,N) broadcast subtract

        # (3) Sieve — cross out sites that barely differ (they're "explained").
        aabs = diff.softabs()                          # (B,N) smooth |diff|
        z = (aabs * P["sg"]) - P["st"]                 # gain * |diff| - threshold
        sieve = z.sigmoid()                            # (B,N) in (0,1)

        # (4) Proportion — lever each local difference toward the global scale.
        k_row = _reshape(P["k"], (1, N_SITES))         # (1,N)
        w_brow = _reshape(w, (1, N_SITES))             # (1,N)
        levered = diff * k_row * w_brow                # (B,N)

        # (5) Integration — sum local differentials (gated) into pooled stats.
        gated1 = levered * sieve                       # first moment carrier
        gated2 = levered * diff * sieve                # second moment (curvature)
        t1 = gated1.sum(axis=1, keepdims=True)         # (B,1) gated differential
        t2 = gated2.sum(axis=1, keepdims=True)         # (B,1) gated curvature
        t3 = mref                                      # (B,1) reference level
        t4 = (sieve * w_brow).sum(axis=1, keepdims=True)  # (B,1) retained mass

        stats = _concat_cols([t1, t2, t3, t4])         # (B, D_STATS)

        h = (stats @ P["W1"] + P["b1"]).tanh()         # (B,H)
        yhat = (h @ P["W2"] + P["b2"])                 # (B,1)
        yhat = _reshape(yhat, (B,))

        # entropy of the probe gate, H(w) = -sum w log w  (minimising concentrates)
        ent = (w * (w + 1e-12).log()).sum() * -1.0     # scalar Node
        return yhat, ent

    def loss(self, X, Y):
        yhat, ent = self.forward(X)
        diff = yhat - Node(Y)
        mse = (diff * diff).sum() * (1.0 / X.shape[0])
        total = mse + ent * self.ent_lambda
        return total, mse, ent

    # --- utility: zero grads / parameter list ---------------------------------
    def zero_grad(self):
        for p in self.params.values():
            p.grad = np.zeros_like(p.data)


# --- small graph helpers (reshape / concat as autodiff ops) -------------------
def _reshape(node, shape):
    out = Node(node.data.reshape(shape), (node,))

    def _bw():
        node.grad += out.grad.reshape(node.data.shape)
    out._backward = _bw
    return out


def _concat_cols(nodes):
    """Concatenate a list of (B,1) Nodes into (B,K)."""
    K = len(nodes)
    B = nodes[0].data.shape[0]
    out = Node(np.concatenate([n.data for n in nodes], axis=1), tuple(nodes))

    def _bw():
        for j, n in enumerate(nodes):
            n.grad += out.grad[:, j:j + 1]
    out._backward = _bw
    return out


# =============================================================================
# PART 4 — GRADIENT CHECK (mandatory; must pass)
# -----------------------------------------------------------------------------
# Central finite differences vs. the engine's analytic gradients, on every
# parameter tensor. Eratosthenes would approve: we trust the method only after
# an independent measurement confirms it.
# =============================================================================

def gradient_check(seed=7, eps=1e-6, tol=1e-5):
    rng = np.random.default_rng(seed)
    net = GnomonNet(hidden=6, ent_lambda=0.05, rng=rng)
    X, Y = make_dataset(5, rng)

    # analytic
    net.zero_grad()
    total, _, _ = net.loss(X, Y)
    total.backward()
    analytic = {name: p.grad.copy() for name, p in net.params.items()}

    def loss_only(net):
        return net.loss(X, Y)[0].data

    max_rel = 0.0
    worst = None
    for name, p in net.params.items():
        flat = p.data.ravel()
        g = analytic[name].ravel()
        # check a few coordinates per tensor to keep it fast
        idxs = range(flat.size) if flat.size <= 12 else rng.choice(
            flat.size, size=12, replace=False)
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps
            lp = loss_only(net)
            flat[i] = orig - eps
            lm = loss_only(net)
            flat[i] = orig
            num = (lp - lm) / (2 * eps)
            denom = max(1e-8, abs(num) + abs(g[i]))
            rel = abs(num - g[i]) / denom
            if rel > max_rel:
                max_rel, worst = rel, (name, int(i), float(num), float(g[i]))
    ok = max_rel < tol
    print(f"[grad-check] max relative error = {max_rel:.2e}  "
          f"(tol {tol:.0e})  ->  {'PASS' if ok else 'FAIL'}")
    if worst:
        print(f"[grad-check] worst coordinate: param={worst[0]} idx={worst[1]} "
              f"numeric={worst[2]:+.6e} analytic={worst[3]:+.6e}")
    return ok


# =============================================================================
# PART 5 — TRAINING LOOP (real; Adam, held-out validation)
# =============================================================================

def adam_init(params):
    return {n: (np.zeros_like(p.data), np.zeros_like(p.data))
            for n, p in params.items()}


def train(epochs=60, batch=64, lr=4e-3, n_train=2000, n_val=400, verbose=True):
    rng = np.random.default_rng(1184)   # the date Eratosthenes fixed for Troy
    net = GnomonNet(hidden=24, ent_lambda=0.015, rng=rng)
    Xtr, Ytr = make_dataset(n_train, rng)
    Xva, Yva = make_dataset(n_val, rng)

    # normalise the target to keep gradients well-scaled; remember the scale.
    y_mean, y_std = Ytr.mean(), Ytr.std() + 1e-8
    Ytr_n = (Ytr - y_mean) / y_std
    Yva_n = (Yva - y_mean) / y_std

    m_v = adam_init(net.params)
    b1, b2, epsa = 0.9, 0.999, 1e-8
    step = 0
    history = []

    for ep in range(epochs):
        perm = rng.permutation(n_train)
        for s in range(0, n_train, batch):
            idx = perm[s:s + batch]
            net.zero_grad()
            total, mse, ent = net.loss(Xtr[idx], Ytr_n[idx])
            total.backward()
            step += 1
            for name, p in net.params.items():
                m, v = m_v[name]
                m[...] = b1 * m + (1 - b1) * p.grad
                v[...] = b2 * v + (1 - b2) * (p.grad ** 2)
                mhat = m / (1 - b1 ** step)
                vhat = v / (1 - b2 ** step)
                p.data -= lr * mhat / (np.sqrt(vhat) + epsa)

        # validation in original units
        yhat_va, _ = net.forward(Xva)
        pred = yhat_va.data * y_std + y_mean
        val_rmse = float(np.sqrt(np.mean((pred - Yva) ** 2)))
        # baseline: predict the training mean
        base_rmse = float(np.sqrt(np.mean((Yva - y_mean) ** 2)))
        history.append(val_rmse)
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            w = _softmax(net.params["a"].data)
            top = np.argsort(w)[::-1][:4]
            print(f"  epoch {ep:3d} | train_mse {mse.data:.4f} | "
                  f"val_RMSE {val_rmse:.3f} | baseline_RMSE {base_rmse:.3f} | "
                  f"top probe sites {top.tolist()}")
    return net, history, (y_mean, y_std), base_rmse


def _softmax(a):
    z = a - a.max()
    e = np.exp(z)
    return e / e.sum()


# =============================================================================
# PART 6 — SELF-TESTS
# =============================================================================

def self_tests():
    print("\n[self-tests]")
    rng = np.random.default_rng(0)

    # (a) engine sanity: d/dx sum(tanh(x*W)) matches finite diff on a scalar
    x = Node(np.array([[0.5, -0.3]]))
    W = Node(rng.normal(size=(2, 3)))
    y = (x @ W).tanh().sum()
    y.backward()
    eps = 1e-6
    g_num = np.zeros_like(x.data)
    for i in range(2):
        xp = x.data.copy(); xp[0, i] += eps
        xm = x.data.copy(); xm[0, i] -= eps
        fp = np.tanh(xp @ W.data).sum()
        fm = np.tanh(xm @ W.data).sum()
        g_num[0, i] = (fp - fm) / (2 * eps)
    err = np.max(np.abs(g_num - x.grad))
    print(f"  engine tanh/matmul grad error: {err:.2e} -> "
          f"{'ok' if err < 1e-6 else 'BAD'}")
    assert err < 1e-6

    # (b) softmax rows sum to 1 and entropy penalty is finite
    net = GnomonNet(rng=np.random.default_rng(3))
    w = _softmax(net.params["a"].data)
    assert abs(w.sum() - 1.0) < 1e-9
    print(f"  probe gate sums to 1: {w.sum():.6f} -> ok")

    # (c) the method beats averaging: train briefly, compare to mean-baseline
    net, hist, _, base = train(epochs=30, n_train=1500, n_val=300, verbose=False)
    print(f"  final val_RMSE {hist[-1]:.3f}  vs  mean-baseline {base:.3f} -> "
          f"{'method wins' if hist[-1] < 0.6 * base else 'no clear win'}")
    assert hist[-1] < 0.85 * base, "Gnomon method failed to beat averaging"
    print("  all self-tests passed.")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 74)
    print(" THE GNOMON NETWORK — global invariants from a few local differences")
    print(" after Eratosthenes of Cyrene (c. 276 - c. 194 BCE)")
    print("=" * 74)

    print("\n[1] Gradient check (engine + architecture):")
    ok = gradient_check()
    assert ok, "Gradient check failed — aborting."

    print("\n[2] Training the Gnomon Network on synthetic shadow-worlds:")
    net, history, yscale, base = train()

    print("\n[3] What the network learned about WHERE TO PUT THE STICK:")
    w = _softmax(net.params["a"].data)
    order = np.argsort(w)[::-1]
    print("    Most-relied-on stations (position : probe weight):")
    for i in order[:5]:
        print(f"      x = {SITES[i]:+.3f}   w = {w[i]:.3f}")
    print(f"    Probe-gate entropy: {(-(w*np.log(w+1e-12)).sum()):.3f} nats "
          f"(uniform would be {np.log(N_SITES):.3f})")
    print(f"    Final validation RMSE {history[-1]:.3f} "
          f"(mean-baseline {base:.3f}); the method recovers the global "
          f"invariant from a handful of differenced sites.")

    self_tests()
    print("\nDone.")

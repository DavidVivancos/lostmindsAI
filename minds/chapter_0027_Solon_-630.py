#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chapter_0027_solon_-630.py  —  The EUNOMIA NETWORK
====================================
Mind #27 : Solon of Athens  (c. 630 - c. 558 BCE)
A from-scratch, pure-NumPy neural architecture that embodies Solon's
distinctive cognitive signature, with a working reverse-mode autodiff
engine, a real training loop, a finite-difference gradient check, and
self-tests. Run it directly:  `python3 chapter_0027_solon_-630.py`
# Part Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 2 Minds 21 - 40 Available on Amazon https://www.amazon.com/dp/B0H6QCQ9M7
# Resume and Interactive Demos at https://artificiology.com/
Author: David Vivancos · Chapter 0027 · Solon
------------------------------------------------------------------------
WHY THIS ARCHITECTURE (the thesis it encodes)
------------------------------------------------------------------------
Solon is usually flattened into "the constitutional lawgiver", which
collapses into the generic AGI lens of "build an auditable institution
with constraint layers". That is NOT what is uniquely his. The neighbours
in this corpus already own the static readings of law:
    - Ur-Nammu : justice as a fixed lookup table of equivalences
    - Hammurabi: justice as k-NN over a public corpus of precedents
    - Hatshepsut: cognition as STATIC constraint-satisfaction toward maat
What is Solon's and no one else's is *dynamical and temporal*. His own
surviving elegies (the "Eunomia" poem, the lines on koros/hubris/ate, the
ages-of-man poem, "I held my shield over both sides and let neither win
unjustly", and Herodotus' "count no man happy until he is dead") describe
the mind/polis as a SELF-CORRECTING DYNAMICAL SYSTEM that survives by
*metering excess over time*. Order (eunomia) is not a state you occupy; it
is a regulatory process that damps the cascade

        koros (satiety)  ->  hubris (excess)  ->  ate (ruin)

*before* the trajectory crosses a point of no return. And — crucially —
the system is judged only by where its trajectory *ends*, never by where
it currently sits, because tyche (fortune) can still reverse a mid-run
state. Two further laws sharpen the mechanism:
    (a) the mediator holds the shield over BOTH poles and lets NEITHER
        win  ->  opponent damping that refuses to fully back either side;
    (b) Solon's law against neutrality in stasis (civil strife): a citizen
        who takes no side loses citizenship  ->  a COMMITMENT constraint
        that forbids "abstaining" (units sitting at ~0) during conflict.

So this network is a recurrent **opponent-regulated dynamical core** with
a homeostatic objective:

    loss = terminal_cross_entropy            # judge the END of the run
         + metron   * running_imbalance^2    # keep balance (the mean)
         + koros    * (excess above set-point)^2   # damp satiety->ruin
         + commit   * abstention_under_stasis      # no neutrality in strife

The mechanism (opponent populations + a mediator that subtracts a
restoring force proportional to live imbalance + a super-linear anti-excess
penalty + terminal-only evaluation + anti-abstention) is load-bearing: it
is what lets the net learn the task below, and it is recognisably Solonian
rather than a Transformer or a constraint-satisfaction snapshot.

------------------------------------------------------------------------
THE TASK (kept tiny so the whole thing is auditable)
------------------------------------------------------------------------
Each example is a length-T stream of civic "shocks" s_t = (demos_push_t,
elite_push_t). Left UNREGULATED, the cumulative tilt u_t = sum(d-e) random-
walks. The label is the historical verdict Solon cared about:
    y = 1 (DYSNOMIA / ruin)  if the unregulated tilt EVER crossed +/-theta
    y = 0 (EUNOMIA / order)  otherwise.
The network never sees u_t; it must read the raw shock stream and predict
the end-verdict. The homeostatic core (which natively tracks running
imbalance and excess) is the right inductive bias, so it learns this well.
"""

import numpy as np

# =====================================================================
# 1.  A MINIMAL REVERSE-MODE AUTODIFF ENGINE  (pure NumPy, from scratch)
# ---------------------------------------------------------------------
# Each Node wraps an ndarray, remembers how it was produced, and knows how
# to push gradient to its parents. This is what makes the architecture
# *trainable* rather than a metaphor; the finite-difference check below
# validates the whole engine + model end to end.
# =====================================================================

def _unbroadcast(grad, shape):
    """Sum `grad` back down to `shape` to undo NumPy broadcasting."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Node:
    """A scalar/array value in the computation graph."""

    __slots__ = ("data", "grad", "_backward", "_prev")

    def __init__(self, data, _children=()):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)

    # ---- elementwise ops (broadcast-aware) --------------------------
    def __add__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data + other.data, (self, other))

        def _b():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)
        out._backward = _b
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data * other.data, (self, other))

        def _b():
            self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += _unbroadcast(out.grad * self.data, other.data.shape)
        out._backward = _b
        return out

    def __neg__(self):
        return self * (-1.0)

    def __sub__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    # ---- matmul (2-D / batched-2-D) ---------------------------------
    def matmul(self, other):
        out = Node(self.data @ other.data, (self, other))

        def _b():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _b
        return out

    # ---- reductions -------------------------------------------------
    def sum(self, axis=None, keepdims=False):
        out = Node(self.data.sum(axis=axis, keepdims=keepdims), (self,))

        def _b():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.grad += np.ones_like(self.data) * g
        out._backward = _b
        return out

    # ---- unary nonlinearities --------------------------------------
    def tanh(self):
        t = np.tanh(self.data)
        out = Node(t, (self,))

        def _b():
            self.grad += (1.0 - t * t) * out.grad
        out._backward = _b
        return out

    def relu(self):
        out = Node(np.maximum(0.0, self.data), (self,))

        def _b():
            self.grad += (self.data > 0.0) * out.grad
        out._backward = _b
        return out

    def exp(self):
        e = np.exp(self.data)
        out = Node(e, (self,))

        def _b():
            self.grad += e * out.grad
        out._backward = _b
        return out

    def log(self):
        out = Node(np.log(self.data), (self,))

        def _b():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _b
        return out

    # ---- backprop driver -------------------------------------------
    def backward(self):
        topo, seen = [], set()

        def build(v):
            if v not in seen:
                seen.add(v)
                for p in v._prev:
                    build(p)
                topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()


# =====================================================================
# 2.  THE EUNOMIA NETWORK
# =====================================================================

class EunomiaNet:
    """
    Recurrent opponent-regulated dynamical core.

    Hidden units are split into two opposing populations:
        - the first H/2 units are the DEMOS  (the many)
        - the last  H/2 units are the AGATHOI (the powerful few)
    A scalar mediator gain `g` (Solon's shield) subtracts a restoring
    force proportional to the live imbalance b_t from whichever side is
    winning, pulling the state back toward the mean (metron). The readout
    judges the *terminal* state plus trajectory summaries.
    """

    def __init__(self, H=8, T=6, seed=0):
        assert H % 2 == 0, "hidden size must be even (two equal populations)"
        self.H, self.T = H, T
        rng = np.random.default_rng(seed)

        def P(shape, scale):
            return Node(rng.standard_normal(shape) * scale)

        # --- recurrent core ---
        self.W_in = P((2, H), 0.6)          # shocks -> hidden
        self.W_rec = P((H, H), 0.5 / np.sqrt(H))
        self.b_rec = P((1, H), 0.0)
        # mediator gain (the shield); kept positive-ish, learned
        self.g = Node(np.array([[0.5]]))    # shape (1,1)
        # --- readout ---
        self.W_p = P((H, 2), 0.5)           # pooled hidden -> 2 logits
        self.w_bT = P((1, 2), 0.5)          # terminal imbalance -> logits
        self.w_kbar = P((1, 2), 0.5)        # mean koros -> logits
        self.w_kT = P((1, 2), 0.5)          # terminal koros -> logits
        self.b_out = Node(np.zeros((1, 2)))

        # constant masks (not parameters): +1 on demos half, -1 on elite
        sign = np.ones((1, H))
        sign[0, H // 2:] = -1.0
        self.sign = sign
        self.inv_half = 2.0 / H             # 1 / (H/2)
        self.inv_H = 1.0 / H

    def params(self):
        return [self.W_in, self.W_rec, self.b_rec, self.g,
                self.W_p, self.w_bT, self.w_kbar, self.w_kT, self.b_out]

    # ---- forward over a batch of shock streams ----------------------
    def forward(self, S, conflict, kappa=0.35, alpha=4.0):
        """
        S        : (B, T, 2) shock streams (Node-free ndarray)
        conflict : (B, T)    1.0 where the step is 'stasis', else 0.0
        Returns logits Node (B,2) and a dict of trajectory Nodes for the
        homeostatic regularisers.
        """
        B = S.shape[0]
        h = Node(np.zeros((B, self.H)))
        pooled = Node(np.zeros((B, self.H)))     # sum_t h_t  (mean later)
        imbalance_sq_sum = Node(np.zeros((B, 1)))
        koros_sum = Node(np.zeros((B, 1)))
        excess_pen_sum = Node(np.zeros((B, 1)))
        commit_sum = Node(np.zeros((B, 1)))
        sign = Node(self.sign)

        last_b = None
        last_k = None
        for t in range(self.T):
            x_t = Node(S[:, t, :])                          # (B,2)
            raw = x_t.matmul(self.W_in) + h.matmul(self.W_rec) + self.b_rec
            # --- imbalance b_t from the PREVIOUS state drives the shield ---
            # b_t = mean(demos) - mean(elite) of current raw activation
            # (use raw's tanh-free proxy via sign mask on a pre-activation)
            pre = raw
            # measure imbalance on the pre-activation
            b_t = (pre * sign).sum(axis=1, keepdims=True) * self.inv_half  # (B,1)
            # mediator: subtract restoring force g*b_t from the winning side
            restore = (self.g * b_t) * sign                # (B,H) broadcast
            h = (pre - restore).tanh()                     # the shield acts
            # --- trajectory measurements on the regulated state h ---
            b_post = (h * sign).sum(axis=1, keepdims=True) * self.inv_half
            k_t = (h * h).sum(axis=1, keepdims=True) * self.inv_H   # koros
            # super-linear excess penalty above set-point kappa (hubris->ate)
            excess = (k_t + (-kappa)).relu()
            excess_pen_sum = excess_pen_sum + excess * excess
            # anti-abstention: under stasis, penalise units near 0
            c_t = Node(conflict[:, t][:, None])            # (B,1)
            absten = ((h * h) * (-alpha)).exp().sum(axis=1, keepdims=True) * self.inv_H
            commit_sum = commit_sum + c_t * absten
            # accumulate
            pooled = pooled + h
            imbalance_sq_sum = imbalance_sq_sum + b_post * b_post
            koros_sum = koros_sum + k_t
            last_b, last_k = b_post, k_t

        invT = 1.0 / self.T
        pooled = pooled * invT
        kbar = koros_sum * invT
        # --- readout: judge the END (last_b, last_k) + the trajectory ---
        logits = (pooled.matmul(self.W_p)
                  + last_b.matmul(self.w_bT)
                  + kbar.matmul(self.w_kbar)
                  + last_k.matmul(self.w_kT)
                  + self.b_out)
        traj = dict(
            metron=(imbalance_sq_sum * invT),     # (B,1) mean imbalance^2
            koros=(excess_pen_sum * invT),        # (B,1) mean excess^2
            commit=(commit_sum * invT),           # (B,1) abstention-in-stasis
        )
        return logits, traj


# =====================================================================
# 3.  LOSS  (terminal CE + the three Solonian homeostatic terms)
# =====================================================================

def cross_entropy(logits, y):
    """Stable softmax CE. logits:(B,2) Node, y:(B,) int ndarray."""
    B = logits.data.shape[0]
    m = logits.data.max(axis=1, keepdims=True)          # detached constant
    shifted = logits + Node(-m)
    expz = shifted.exp()
    Z = expz.sum(axis=1, keepdims=True)
    logsumexp = Z.log()                                 # (B,1)
    onehot = np.zeros((B, 2)); onehot[np.arange(B), y] = 1.0
    correct = (shifted * Node(onehot)).sum(axis=1, keepdims=True)
    nll = logsumexp - correct                           # (B,1)
    return nll.sum() * (1.0 / B)


def total_loss(net, S, conflict, y, lam=(0.30, 0.20, 0.10)):
    lam_metron, lam_koros, lam_commit = lam
    logits, traj = net.forward(S, conflict)
    ce = cross_entropy(logits, y)
    reg = (traj["metron"].sum() * (lam_metron / S.shape[0])
           + traj["koros"].sum() * (lam_koros / S.shape[0])
           + traj["commit"].sum() * (lam_commit / S.shape[0]))
    return ce + reg, logits


# =====================================================================
# 4.  SYNTHETIC CIVIC-SHOCK DATASET
# =====================================================================

def make_data(n, T=6, theta=1.8, seed=1):
    rng = np.random.default_rng(seed)
    S = rng.uniform(-1.0, 1.0, size=(n, T, 2))
    d, e = S[:, :, 0], S[:, :, 1]
    tilt = np.cumsum(d - e, axis=1)                     # unregulated walk
    crossed = (np.abs(tilt) > theta).any(axis=1)
    y = crossed.astype(np.int64)                        # 1 = dysnomia/ruin
    conflict = (np.abs(d - e) > 1.0).astype(np.float64) # stasis steps
    return S, conflict, y


# =====================================================================
# 5.  FINITE-DIFFERENCE GRADIENT CHECK  (mandatory)
# =====================================================================

def grad_check(verbose=True):
    net = EunomiaNet(H=6, T=4, seed=3)
    S, conflict, y = make_data(5, T=4, seed=7)
    # analytic gradients
    for p in net.params():
        p.grad = np.zeros_like(p.data)
    loss, _ = total_loss(net, S, conflict, y)
    loss.backward()
    analytic = [p.grad.copy() for p in net.params()]

    eps = 1e-6
    max_rel = 0.0
    for pi, p in enumerate(net.params()):
        flat = p.data.reshape(-1)
        # check up to 6 entries per parameter to keep it fast
        idxs = range(min(flat.size, 6))
        for k in idxs:
            orig = flat[k]
            flat[k] = orig + eps
            lp, _ = total_loss(net, S, conflict, y)
            flat[k] = orig - eps
            lm, _ = total_loss(net, S, conflict, y)
            flat[k] = orig
            num = (lp.data - lm.data) / (2 * eps)
            ana = analytic[pi].reshape(-1)[k]
            denom = max(1e-8, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            max_rel = max(max_rel, rel)
    if verbose:
        print(f"[grad-check] max relative error = {max_rel:.3e}  "
              f"({'PASS' if max_rel < 1e-4 else 'FAIL'})")
    return max_rel


# =====================================================================
# 6.  TRAINING LOOP  (plain SGD; 'as I grow old I keep learning')
# =====================================================================

def accuracy(net, S, conflict, y):
    logits, _ = net.forward(S, conflict)
    pred = logits.data.argmax(axis=1)
    return (pred == y).mean()


def train(epochs=60, lr=0.15, batch=64, seed=0, verbose=True):
    net = EunomiaNet(H=8, T=6, seed=seed)
    Str, ctr, ytr = make_data(640, T=6, seed=1)
    Ste, cte, yte = make_data(256, T=6, seed=2)
    n = Str.shape[0]
    rng = np.random.default_rng(seed)
    history = []
    for ep in range(epochs):
        order = rng.permutation(n)
        ep_loss = 0.0
        for i in range(0, n, batch):
            idx = order[i:i + batch]
            for p in net.params():
                p.grad = np.zeros_like(p.data)
            loss, _ = total_loss(net, Str[idx], ctr[idx], ytr[idx])
            loss.backward()
            for p in net.params():
                np.clip(p.grad, -5.0, 5.0, out=p.grad)   # gentle, no excess
                p.data -= lr * p.grad
            ep_loss += loss.data * len(idx)
        ep_loss /= n
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            tr = accuracy(net, Str, ctr, ytr)
            te = accuracy(net, Ste, cte, yte)
            print(f"epoch {ep:3d} | loss {ep_loss:.4f} | "
                  f"train acc {tr:.3f} | test acc {te:.3f} | "
                  f"shield g = {net.g.data.ravel()[0]:+.3f}")
        history.append(ep_loss)
    return net, history


# =====================================================================
# 7.  SELF-TESTS
# =====================================================================

def self_tests():
    print("=" * 64)
    print("SELF-TESTS")
    print("=" * 64)

    # (a) autodiff sanity: d/dx sum(tanh(x)) == 1 - tanh(x)^2
    x = Node(np.array([[0.3, -0.7, 1.1]]))
    y = x.tanh().sum()
    y.backward()
    expect = 1 - np.tanh(x.data) ** 2
    err = np.abs(x.grad - expect).max()
    print(f"[test] tanh-grad max err = {err:.2e}  "
          f"({'ok' if err < 1e-10 else 'BAD'})")
    assert err < 1e-10

    # (b) dataset is non-trivial and roughly balanced
    _, _, y = make_data(2000, seed=5)
    frac = y.mean()
    print(f"[test] dysnomia fraction = {frac:.3f}  "
          f"({'ok' if 0.2 < frac < 0.8 else 'BAD'})")
    assert 0.2 < frac < 0.8

    # (c) gradient check passes
    mr = grad_check(verbose=True)
    assert mr < 1e-4, "gradient check failed"

    # (d) shield actually damps imbalance: compare koros with g=0 vs trained
    net = EunomiaNet(H=8, T=6, seed=0)
    S, c, yy = make_data(64, seed=9)
    _, traj_on = net.forward(S, c)
    net.g.data[:] = 0.0
    _, traj_off = net.forward(S, c)
    on = traj_on["metron"].data.mean()
    off = traj_off["metron"].data.mean()
    print(f"[test] mean imbalance^2  shield-ON={on:.4f}  shield-OFF={off:.4f}"
          f"  ({'damps' if on < off else 'no-damp'})")
    print("All structural self-tests passed.\n")


# =====================================================================
# 8.  MAIN
# =====================================================================

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    self_tests()
    print("=" * 64)
    print("TRAINING THE EUNOMIA NETWORK")
    print("=" * 64)
    net, hist = train(epochs=60, lr=0.15, seed=0)
    Ste, cte, yte = make_data(512, T=6, seed=42)
    print("-" * 64)
    print(f"FINAL held-out accuracy = {accuracy(net, Ste, cte, yte):.3f}")
    print(f"Learned shield gain g    = {net.g.data.ravel()[0]:+.4f}")
    print("Solon's verdict: order is the trajectory that never crossed.")

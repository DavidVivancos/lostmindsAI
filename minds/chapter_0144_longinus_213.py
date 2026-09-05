#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Chapter 0144_longinus_213 - Longinus (213-273 CE)
"On the Sublime" (Peri Hypsous) -> a critic-first architecture for machine taste
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 8 Minds 141 - 160 Available on Amazon https://www.amazon.com/dp/B0HH8RTCXF
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 144: Longinus (213-273 CE)
================================================================================  

WHAT THIS FILE IS
-----------------
A small, fully self-contained neural model written from scratch in NumPy (no
PyTorch / TensorFlow / autograd libraries) that encodes the distinctive
cognitive signature of Longinus rather than the usual transformer defaults.

The central Longinian ideas, and how each becomes a mechanism:

  1. "The effect of great writing is not persuasion but TRANSPORT (ekstasis)."
     -> The seat of intelligence is not the generator but the CRITIC: a taste
        function that recognises a transporting stroke. We build the evaluator
        first and treat generation as subordinate to it.

  2. "Genius even when it makes mistakes is preferable to impeccable mediocrity"
     (the Rule-vs-Genius digression, chs. 33-36 - "the most eloquent part of
     the book, and central to its message").
     -> The passage score is a soft-MAX (peak) over its strokes, NOT a mean.
        A single thunderbolt lifts the whole; local "nods" (Homer's slips) are
        tolerated. This inverts modern average-loss optimisation.

  3. "If it pleases all people at all times, then it is truly sublime" (ch. 7 -
     the universality / invariance test).
     -> A stroke only counts as elevated if a DIVERSE ENSEMBLE of readers all
        respond (soft-MIN across readers = "pleases everybody") and the response
        survives repeated exposure (robust to re-reading noise = "at all times").

  4. Longinus catalogues three COUNTERFEITS of the sublime (chs. 3-5):
        - tumidity / bombast (to oidoun / onkos): swollen sound, hollow.
        - frigidity / puerility (to psychron / meirakiodes): pedantic, over-even.
        - parenthyrson: false, ill-timed, one-sided passion.
     -> Three learned discriminator penalties subtract from the score. Taste is
        as much the rejection of the false-sublime as the recognition of the true.

So the model's verdict is:   score = TRANSPORT_peak  -  COUNTERFEIT_penalty
and it is trained only to RANK genuine sublimity above its three counterfeits -
never to minimise per-token error. "To judge like Longinus" meant "to judge
correctly"; here judgment is the whole architecture.

The file contains:
  * a minimal tensor-valued reverse-mode autodiff engine (Node);
  * the Krisis (Greek: judgment) critic model built on it;
  * a synthetic corpus of genuine sublime passages and the three counterfeits;
  * a MANDATORY finite-difference gradient check;
  * a real training loop (Adam, from scratch);
  * self-tests for every Longinian property above.

Run:  python3 chapter_0144_longinus_213.py
===============================================================================
"""

import numpy as np

RNG = np.random.default_rng(213)  # seed = Longinus' birth year, for reproducibility


# =============================================================================
# 1. A MINIMAL REVERSE-MODE AUTODIFF ENGINE (pure NumPy, from scratch)
# -----------------------------------------------------------------------------
# Each Node wraps a NumPy array and remembers how to push gradients back to the
# parents that produced it. This is the whole of the "deep learning framework"
# used below - a few hundred lines, no external autodiff.
# =============================================================================

def _unbroadcast(grad, shape):
    """Sum a gradient back down to `shape` after NumPy broadcasting."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


class Node:
    """A differentiable tensor."""

    __slots__ = ("data", "grad", "_backward", "_parents")

    def __init__(self, data, parents=()):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._parents = parents

    # ---- basic arithmetic --------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data + other.data, (self, other))

        def _bw():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)
        out._backward = _bw
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data * other.data, (self, other))

        def _bw():
            self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += _unbroadcast(out.grad * self.data, other.data.shape)
        out._backward = _bw
        return out

    def __matmul__(self, other):
        out = Node(self.data @ other.data, (self, other))

        def _bw():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _bw
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    # ---- non-linearities and reductions ------------------------------------
    def tanh(self):
        t = np.tanh(self.data)
        out = Node(t, (self,))

        def _bw():
            self.grad += out.grad * (1.0 - t * t)
        out._backward = _bw
        return out

    def square(self):
        out = Node(self.data ** 2, (self,))

        def _bw():
            self.grad += out.grad * 2.0 * self.data
        out._backward = _bw
        return out

    def softplus(self):
        # numerically stable log(1+exp(x))
        x = self.data
        out_data = np.logaddexp(0.0, x)
        out = Node(out_data, (self,))
        sig = 1.0 / (1.0 + np.exp(-x))

        def _bw():
            self.grad += out.grad * sig
        out._backward = _bw
        return out

    def relu(self):
        out = Node(np.maximum(0.0, self.data), (self,))

        def _bw():
            self.grad += out.grad * (self.data > 0.0)
        out._backward = _bw
        return out

    def sum(self, axis=None, keepdims=False):
        out = Node(self.data.sum(axis=axis, keepdims=keepdims), (self,))

        def _bw():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.grad += np.ones_like(self.data) * g
        out._backward = _bw
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def logsumexp(self, axis):
        """log-sum-exp along an axis; its gradient is the softmax there.

        This single primitive builds every Longinian aggregator:
          soft-max (the peak / thunderbolt)  = (1/a) * logsumexp(a*x)
          soft-min (pleases everybody)       = -(1/b) * logsumexp(-b*x)
        """
        x = self.data
        m = np.max(x, axis=axis, keepdims=True)
        e = np.exp(x - m)
        s = e.sum(axis=axis, keepdims=True)
        lse = (m + np.log(s))
        lse = np.squeeze(lse, axis=axis)
        out = Node(lse, (self,))
        soft = e / s  # softmax along axis

        def _bw():
            g = np.expand_dims(out.grad, axis)
            self.grad += g * soft
        out._backward = _bw
        return out

    # ---- backpropagation ---------------------------------------------------
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


# ---- convenience wrappers built from logsumexp ------------------------------
def soft_max(node, axis, alpha=8.0):
    """Differentiable peak along `axis` (the Longinian thunderbolt)."""
    return (node * alpha).logsumexp(axis) * (1.0 / alpha)


def soft_min(node, axis, beta=8.0):
    """Differentiable minimum along `axis` ('pleases everybody')."""
    return (node * (-beta)).logsumexp(axis) * (-1.0 / beta)


# =============================================================================
# 2. THE KRISIS CRITIC  (Longinus' taste, made trainable)
# -----------------------------------------------------------------------------
# Input : a passage = (S strokes) x (F features).
# Output: a scalar VERDICT = transporting-peak  -  counterfeit-penalty.
# =============================================================================

class KrisisCritic:
    """A from-scratch critic that scores passages for genuine sublimity."""

    def __init__(self, n_feat=8, n_hidden=12, n_readers=5, seed=213):
        r = np.random.default_rng(seed)
        s = 0.4
        # encoder: raw stroke features -> a shared "elevation" latent
        self.We = Node(r.normal(0, s, size=(n_feat, n_hidden)))
        self.be = Node(np.zeros(n_hidden))
        # the reader ensemble: K diverse "souls", each a direction in latent space.
        # Diversity is what makes the invariance ("pleases everybody") test bite.
        self.R = Node(r.normal(0, s, size=(n_readers, n_hidden)))
        # counterfeit discriminator: weights that turn 3 fault-features into a penalty
        self.w_pen = Node(np.array([0.5, 0.5, 0.5]))   # bombast, frigidity, parenthyrson
        self.b_pen = Node(np.array(-0.3))
        self.n_readers = n_readers

    def params(self):
        return [self.We, self.be, self.R, self.w_pen, self.b_pen]

    def verdict(self, passage_np, alpha=5.0, beta=5.0):
        """Forward pass on ONE passage (S x F NumPy array) -> scalar Node."""
        X = Node(passage_np)                       # (S,F)
        Z = (X @ self.We + self.be).tanh()         # (S,H)  encoded strokes
        # reader affinities: (S,H) @ (H,K) = (S,K)
        A = Z @ self._RT()                         # (S,K)
        rr = A.tanh()                              # (S,K) each reader's response in (-1,1)

        # --- per-stroke aggregation across the reader ensemble ---------------
        r_min = soft_min(rr, axis=1, beta=beta)    # (S,) least-impressed reader ("everybody")
        r_max = soft_max(rr, axis=1, alpha=beta)   # (S,) most-impressed reader
        r_mean = rr.mean(axis=1)                   # (S,) average reader
        spread = r_max - r_min                     # (S,) disagreement among readers

        # --- TRANSPORT: peak stroke that pleases EVERYBODY (thunderbolt) -----
        E = soft_max(r_min, axis=0, alpha=alpha)   # scalar elevation of the passage

        # --- three COUNTERFEIT fault-features -------------------------------
        # (a) bombast: loud AND divisive -> swollen sound the ensemble rejects
        amp = rr.square().mean(axis=1)             # (S,) loudness
        bombast_s = amp * spread                   # (S,)
        cf_bombast = soft_max(bombast_s, axis=0, alpha=alpha)

        # (b) frigidity / puerility: no peak above the average -> over-even, pedantic
        mean_elev = r_min.mean(axis=0)             # scalar
        cf_frigid = mean_elev - E                  # ~0 when flat (frigid), very negative when peaky

        # (c) parenthyrson: one reader carried away while another is repelled
        #     (max reader high AND min reader below zero) -> misplaced passion
        one_sided = r_max.softplus() * (-r_min).softplus()   # (S,)
        cf_parenthyrson = soft_max(one_sided, axis=0, alpha=alpha)

        cfeats = _stack3(cf_bombast, cf_frigid, cf_parenthyrson)   # (3,)
        # squash fault-features to a bounded range so the verdict stays stable
        # across re-readings ('pleases at all times') and penalties cannot explode
        penalty = ((self.w_pen * cfeats.tanh()).sum() + self.b_pen).softplus()

        return E - penalty

    # helpers -----------------------------------------------------------------
    def _RT(self):
        """Return R transposed (H,K) as a Node, with gradient wired to self.R."""
        Rt = _transpose(self.R)
        return Rt


# ---- small structural ops the critic needs ----------------------------------
def _transpose(node):
    out = Node(node.data.T, (node,))

    def _bw():
        node.grad += out.grad.T
    out._backward = _bw
    return out


def _stack3(a, b, c):
    """Stack three scalar Nodes into a length-3 Node."""
    out = Node(np.array([a.data, b.data, c.data]).reshape(3), (a, b, c))

    def _bw():
        a.grad += out.grad[0]
        b.grad += out.grad[1]
        c.grad += out.grad[2]
    out._backward = _bw
    return out


# =============================================================================
# 3. A SYNTHETIC CORPUS: genuine sublimity vs its three counterfeits
# -----------------------------------------------------------------------------
# Each stroke has hidden latents the model never sees:
#     h = height / grandeur      u = unity (does the whole audience assent?)
#     a = loudness of sound      o = pedantic ornament
# Raw features are a fixed random projection of these latents plus noise, so the
# model must LEARN readers/encoder that recover the structure. Passage TYPE is
# the label; the critic must rank GENUINE above each counterfeit.
# =============================================================================

N_LAT = 6
N_FEAT = 8
S_STROKES = 6
_PROJ = RNG.normal(0, 1.0, size=(N_LAT, N_FEAT))   # fixed latent->feature map


def _latents_to_features(lat):
    """lat: (S, N_LAT) hidden -> (S, N_FEAT) observed features."""
    base = lat @ _PROJ
    return base + RNG.normal(0, 0.15, size=base.shape)


def _stroke(h, u, a, o):
    # nonlinear latent vector emphasising the interactions Longinus cares about
    return np.array([h, u, a, o, h * u, a * (1.0 - u)])


def make_passage(kind):
    """Return (features SxF, kind). kind in {genuine,bombast,frigid,parenthyrson}."""
    strokes = []
    if kind == "genuine":
        # mostly moderate strokes, one or two TRUE peaks (high h, high u), maybe a nod
        for i in range(S_STROKES):
            h = RNG.uniform(0.3, 0.5); u = RNG.uniform(0.7, 0.9)
            a = RNG.uniform(0.3, 0.5); o = RNG.uniform(0.1, 0.3)
            strokes.append(_stroke(h, u, a, o))
        # the thunderbolt: one stroke of genuine height that pleases everyone
        j = RNG.integers(0, S_STROKES)
        strokes[j] = _stroke(RNG.uniform(0.9, 1.0), RNG.uniform(0.85, 1.0),
                             RNG.uniform(0.5, 0.7), RNG.uniform(0.0, 0.2))
        # a Homeric "nod": one careless low stroke, tolerated
        k = (j + 3) % S_STROKES
        strokes[k] = _stroke(RNG.uniform(0.05, 0.2), RNG.uniform(0.6, 0.8),
                             RNG.uniform(0.2, 0.4), RNG.uniform(0.1, 0.3))
    elif kind == "bombast":
        # loud everywhere, claims height, but audience does NOT agree (low u)
        for i in range(S_STROKES):
            strokes.append(_stroke(RNG.uniform(0.7, 1.0), RNG.uniform(0.05, 0.25),
                                   RNG.uniform(0.8, 1.0), RNG.uniform(0.2, 0.4)))
    elif kind == "frigid":
        # correct, even, pedantic: uniform moderate height, high unity, NO peak, ornate
        for i in range(S_STROKES):
            strokes.append(_stroke(RNG.uniform(0.4, 0.55), RNG.uniform(0.8, 0.95),
                                   RNG.uniform(0.3, 0.45), RNG.uniform(0.7, 1.0)))
    elif kind == "parenthyrson":
        # mostly flat, but a burst of one-sided passion: one reader thrilled, others cold
        for i in range(S_STROKES):
            strokes.append(_stroke(RNG.uniform(0.3, 0.45), RNG.uniform(0.75, 0.9),
                                   RNG.uniform(0.3, 0.45), RNG.uniform(0.2, 0.4)))
        j = RNG.integers(0, S_STROKES)
        strokes[j] = _stroke(RNG.uniform(0.6, 0.8), RNG.uniform(0.1, 0.3),
                             RNG.uniform(0.85, 1.0), RNG.uniform(0.2, 0.4))
    else:
        raise ValueError(kind)
    lat = np.array(strokes)
    return _latents_to_features(lat), kind


def make_flawed_genius():
    """A true unanimous peak plus a careless 'Homeric nod' (chs. 33-36)."""
    st = [_stroke(RNG.uniform(0.35, 0.5), RNG.uniform(0.8, 0.95),
                  RNG.uniform(0.3, 0.5), RNG.uniform(0.1, 0.3)) for _ in range(S_STROKES)]
    st[1] = _stroke(RNG.uniform(0.93, 1.0), RNG.uniform(0.9, 1.0),
                    RNG.uniform(0.5, 0.7), 0.1)                # thunderbolt
    st[4] = _stroke(RNG.uniform(0.05, 0.15), RNG.uniform(0.6, 0.8), 0.3, 0.2)  # nod
    return _latents_to_features(np.array(st))


def make_flawless_mediocre():
    """Correct, even, unanimous - but with no transporting peak and no flaw."""
    st = [_stroke(RNG.uniform(0.5, 0.62), RNG.uniform(0.85, 0.97),
                  RNG.uniform(0.35, 0.5), RNG.uniform(0.2, 0.35)) for _ in range(S_STROKES)]
    return _latents_to_features(np.array(st))


def make_please_all():
    """A moderate stroke to which EVERY reader assents (ch. 7)."""
    st = [_stroke(RNG.uniform(0.3, 0.45), RNG.uniform(0.85, 0.98),
                  RNG.uniform(0.3, 0.45), RNG.uniform(0.1, 0.3)) for _ in range(S_STROKES)]
    st[2] = _stroke(RNG.uniform(0.65, 0.75), RNG.uniform(0.9, 0.99),
                    RNG.uniform(0.45, 0.55), 0.15)             # high AND unanimous
    return _latents_to_features(np.array(st))


def make_please_one():
    """A louder, higher stroke that only ONE reader loves - divisive."""
    st = [_stroke(RNG.uniform(0.3, 0.45), RNG.uniform(0.8, 0.95),
                  RNG.uniform(0.3, 0.45), RNG.uniform(0.1, 0.3)) for _ in range(S_STROKES)]
    st[2] = _stroke(RNG.uniform(0.85, 0.95), RNG.uniform(0.1, 0.2),
                    RNG.uniform(0.85, 0.95), 0.2)              # higher but divisive
    return _latents_to_features(np.array(st))


def make_dataset(n_per=120):
    kinds = ["genuine", "bombast", "frigid", "parenthyrson"]
    data = {k: [make_passage(k)[0] for _ in range(n_per)] for k in kinds}
    return data


# =============================================================================
# 4. MANDATORY FINITE-DIFFERENCE GRADIENT CHECK
# =============================================================================

def gradient_check(critic, passage, eps=1e-5, tol=1e-4):
    """Compare analytic backprop grads to numerical grads on the verdict."""
    for p in critic.params():
        p.grad = np.zeros_like(p.data)
    y = critic.verdict(passage)
    y.backward()

    max_rel = 0.0
    for p in critic.params():
        flat = p.data.reshape(-1)
        gan = p.grad.reshape(-1).copy()
        gnum = np.zeros_like(flat)
        for i in range(flat.size):
            orig = flat[i]
            flat[i] = orig + eps
            yp = critic.verdict(passage).data
            flat[i] = orig - eps
            ym = critic.verdict(passage).data
            flat[i] = orig
            gnum[i] = (yp - ym) / (2 * eps)
        denom = np.maximum(1e-8, np.abs(gan) + np.abs(gnum))
        rel = np.max(np.abs(gan - gnum) / denom)
        max_rel = max(max_rel, rel)
    return max_rel, max_rel < tol


# =============================================================================
# 5. TRAINING  (Adam, from scratch) - learn to rank sublime above counterfeit
# =============================================================================

class Adam:
    def __init__(self, params, lr=0.02, b1=0.9, b2=0.999, eps=1e-8, weight_decay=2e-3):
        self.params = params; self.lr = lr; self.b1 = b1; self.b2 = b2; self.eps = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data)

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)
            p.data -= self.lr * self.weight_decay * p.data  # keep taste bounded & stable


def train(critic, data, epochs=60, margin=0.4, lr=0.02, verbose=True):
    opt = Adam(critic.params(), lr=lr)
    counterfeits = ["bombast", "frigid", "parenthyrson"]
    n = len(data["genuine"])
    history = []
    for ep in range(epochs):
        order = RNG.permutation(n)
        total = 0.0
        for idx in order:
            opt.zero_grad()
            g_pass = data["genuine"][idx]
            y_g = critic.verdict(g_pass)
            # (i) rank genuine above one sampled counterfeit of each type (chs. 3-5)
            loss = Node(0.0)
            for c in counterfeits:
                c_pass = data[c][RNG.integers(0, n)]
                y_c = critic.verdict(c_pass)
                # hard margin (hinge): once genuine beats counterfeit by `margin`,
                # the gradient is exactly zero, so the taste stops over-separating
                loss = loss + (Node(margin) - (y_g - y_c)).relu()
            # (ii) Rule-vs-Genius (chs. 33-36): flawed grandeur > flawless mediocrity
            y_fg = critic.verdict(make_flawed_genius())
            y_fm = critic.verdict(make_flawless_mediocre())
            loss = loss + (Node(margin) - (y_fg - y_fm)).relu()
            # (iii) universality (ch. 7): pleases-everybody > pleases-one
            y_pa = critic.verdict(make_please_all())
            y_po = critic.verdict(make_please_one())
            loss = loss + (Node(margin) - (y_pa - y_po)).relu()
            loss.backward()
            opt.step()
            total += float(loss.data)
        history.append(total / n)
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:3d}   mean ranking loss = {history[-1]:.4f}")
    return history


# =============================================================================
# 6. EVALUATION HELPERS
# =============================================================================

def mean_verdict(critic, passages):
    return float(np.mean([critic.verdict(p).data for p in passages]))


def ranking_accuracy(critic, data, kind):
    """Fraction of genuine passages that outscore a random counterfeit of `kind`."""
    n = len(data["genuine"])
    wins = 0
    for i in range(n):
        yg = critic.verdict(data["genuine"][i]).data
        yc = critic.verdict(data[kind][RNG.integers(0, n)]).data
        wins += (yg > yc)
    return wins / n


# =============================================================================
# 7. MAIN: gradient check -> train -> Longinian self-tests
# =============================================================================

def main():
    print("=" * 74)
    print("LONGINUS / KRISIS CRITIC  -  a critic-first model of the sublime")
    print("=" * 74)

    critic = KrisisCritic(n_feat=N_FEAT, n_hidden=12, n_readers=5, seed=213)

    # ---- (A) gradient check (mandatory) ------------------------------------
    print("\n[A] Finite-difference gradient check (analytic vs numerical):")
    probe, _ = make_passage("genuine")
    max_rel, ok = gradient_check(critic, probe)
    print(f"    max relative error = {max_rel:.2e}   ->   {'PASS' if ok else 'FAIL'}")
    assert ok, "gradient check failed"

    # ---- (B) data + training ----------------------------------------------
    print("\n[B] Building synthetic corpus (genuine + 3 counterfeits) ...")
    data = make_dataset(n_per=120)
    print("    training the critic to rank sublime above counterfeit:")
    hist = train(critic, data, epochs=60, lr=0.02)
    print(f"    loss: {hist[0]:.4f}  ->  {hist[-1]:.4f}")

    # ---- (C) Longinian self-tests -----------------------------------------
    print("\n[C] Self-tests (each encodes a doctrine from 'On the Sublime'):")

    # C1 - counterfeit rejection (chs. 3-5)
    accs = {k: ranking_accuracy(critic, data, k) for k in ["bombast", "frigid", "parenthyrson"]}
    v_gen = mean_verdict(critic, data["genuine"])
    for k in accs:
        vk = mean_verdict(critic, data[k])
        print(f"    C1 genuine > {k:<12}: acc={accs[k]*100:5.1f}%  "
              f"(verdict {v_gen:+.3f} vs {vk:+.3f})")
    assert all(a > 0.85 for a in accs.values()), "counterfeit rejection too weak"

    # C2 - Rule vs Genius (chs. 33-36): flawed grandeur beats flawless mediocrity
    # (freshly sampled held-out passages, distinct from any seen in training)
    flawed_genius = [make_flawed_genius() for _ in range(200)]
    flawless_mediocre = [make_flawless_mediocre() for _ in range(200)]
    v_fg = mean_verdict(critic, flawed_genius)
    v_fm = mean_verdict(critic, flawless_mediocre)
    win_fg = np.mean([critic.verdict(flawed_genius[i]).data >
                      critic.verdict(flawless_mediocre[i]).data for i in range(200)])
    print(f"    C2 flawed-genius > flawless-mediocrity: {win_fg*100:5.1f}%  "
          f"(verdict {v_fg:+.3f} vs {v_fm:+.3f})")
    assert win_fg > 0.85, "Rule-vs-Genius property not learned"

    # C3 - universality ('pleases everybody' beats 'pleases one'), ch. 7
    please_all = [make_please_all() for _ in range(200)]
    please_one = [make_please_one() for _ in range(200)]
    win_all = np.mean([critic.verdict(please_all[i]).data >
                       critic.verdict(please_one[i]).data for i in range(200)])
    print(f"    C3 pleases-everybody > pleases-one:     {win_all*100:5.1f}%  "
          f"(verdict {mean_verdict(critic, please_all):+.3f} vs "
          f"{mean_verdict(critic, please_one):+.3f})")
    assert win_all > 0.80, "invariance/universality property not learned"

    # C4 - re-read stability ('at all times'): verdict robust to exposure noise
    base, _ = make_passage("genuine")
    reread = [critic.verdict(base + RNG.normal(0, 0.05, size=base.shape)).data
              for _ in range(50)]
    print(f"    C4 re-reading stability: verdict std over 50 exposures = "
          f"{np.std(reread):.4f}  (low = 'pleases at all times')")
    assert np.std(reread) < 0.15, "verdict not stable across re-readings"

    # ---- learned taste weights --------------------------------------------
    print("\n[D] Learned counterfeit-penalty weights "
          "(bombast, frigidity, parenthyrson):")
    print(f"    w_pen = {np.round(critic.w_pen.data, 3)}   b_pen = {critic.b_pen.data:+.3f}")
    print("\nAll Longinian self-tests passed. The critic judges as Longinus judged:")
    print("it prizes the transporting peak, demands the assent of every reader,")
    print("forgives the careless nod, and unmasks the three counterfeits.\n")


if __name__ == "__main__":
    main()

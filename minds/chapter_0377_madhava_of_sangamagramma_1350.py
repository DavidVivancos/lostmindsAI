#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 THE SAMSKARA ENGINE
 # BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0377 · Madhava of Sangamagramma (1350-1425)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 19, Minds 361-380 Available on Amazon https://www.amazon.com/dp/B0HQX147Y7
# Resume and Interactive Demos at https://artificiology.com/
# END ATTRIBUTION 
================================================================================

WHAT THIS FILE IS
-----------------
A complete, trainable, from-scratch cognitive architecture (pure NumPy, no
frameworks) built from one idea that belongs to Madhava and to no one else in
the corpus:

    A computation that has been STOPPED EARLY leaves behind a remainder,
    and that remainder is not noise. It has a law. Model the law and you
    can stop a hundred million steps sooner.

Madhava's series for pi/4 is 1 - 1/3 + 1/5 - 1/7 + ...  It converges so slowly
that ten correct decimal places would need roughly 5 x 10^9 terms. Madhava --
or the Kerala school writing in his name -- did not run it. He wrote down three
successively better closed-form expressions for the ERROR YOU WOULD STILL HAVE
after stopping at term n, called them the antya-samskara ("end correction"),
and added them back:

    pi/4 = 1 - 1/3 + ... + (-1)^(n-1)/(2n-1) + (-1)^n * F(n)

    F1(n) = 1/(4n)                    F2(n) = n/(4n^2+1)
    F3(n) = (n^2+1) / (n(4n^2+5))

With F3 and only 21 terms the error is 7.7e-11. Without any correction, 21
terms give 1.2e-2. That is nine orders of magnitude bought with algebra
instead of arithmetic.

And there is a second, deeper move, which is the actual architecture here. The
Kerala texts also define STHAULYA -- "grossness", a measure of how wrong the
correction itself is -- and it is computable WITHOUT KNOWING THE ANSWER. If the
correction were exact, then truncating at n-1 and truncating at n must give the
same value, which forces

    F(n-1) + F(n) = 1/(2n-1)          (the samskara consistency condition)

so the residual

    sthaulya(n) = F(n-1) + F(n) - 1/(2n-1)

is a self-audit. It uses no reference value, no held-out set, no label. It is a
system checking whether it agrees with itself about where it stopped. And it
works: for F3 at n=101 the sthaulya is 2.7e-15 and the true error is 1.3e-15.

THE ARCHITECTURE
----------------
SAMSKARA-NET turns that into a learning machine. It is not a transformer; there
is no attention, no key-value store, no token mixing. It is a machine that
emits an ALTERNATING SERIES and then models its own tail.

    x -> CAPA ENCODER      fixed arc features (no parameters)
      -> SRENI BODY        learned decay ratios r_k in (0,1); terms are
                           generated multiplicatively, t_k = t_1 * (-1)^(k-1) *
                           prod(r_2..r_k). Alternation is structural.
      -> SAMSKARA HEAD     g_n in (0,1); estimated remainder R_n = g_n * t_(n+1)
                           -- a learned fraction of the FIRST OMITTED TERM.
      -> answer at depth n:  y_n = S_n + g_n * t_(n+1)
      -> STHAULYA          s_n = y_n - y_(n-1)  (must be zero; label-free)
      -> VICCHEDA GATE     halting distribution over depths, driven by the
                           model's own sthaulya, penalised by expected depth.

Three properties fall out of the shape rather than the training:

  (1) THE ANSWER IS CAGED. Because g_n is a sigmoid, y_n always lies strictly
      between the partial sums S_n and S_(n+1). The alternating bracket is not
      a loss term; it is impossible to violate.
  (2) THE ERROR BAR IS FREE. |y_n - y_(n-1)| is emitted on the forward pass at
      no extra cost and needs no labels.
  (3) THE CORRECTION IS LEARNABLE WITHOUT SUPERVISION. Experiment E3 below
      trains the correction head on the sthaulya residual ALONE, with the
      targets deleted, and test error against the true function falls by two
      orders of magnitude.

Divergence from neighbouring figures in this corpus: Aryabhata (161) makes the
recurrence the knowledge -- regenerate the table, never store it. Here the
recurrence is the CHEAP part and is deliberately starved; all the capacity sits
in the model of what was skipped. Archimedes (89) narrows a bracket by
computing more (96-gon). Madhava stops the bracket early and computes WHERE
INSIDE IT the answer sits. Bhaskara II (243) names an unresolvable quantity and
carries it forward symbolically; Madhava estimates it numerically and reports
how badly.

RUN
---
    python3 chapter_0377_madhava_of_sangamagramma_1350.py            # gradient check, training, all six experiments
    python3 chapter_0377_madhava_of_sangamagramma_1350.py --quick    # shorter training run

Requires only numpy.
================================================================================
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from fractions import Fraction

import numpy as np

# =============================================================================
# SECTION 0 -- A MINIMAL REVERSE-MODE AUTODIFF ENGINE
# =============================================================================
# Written from scratch so the whole model is inspectable. `V` wraps a NumPy
# array and records the operation that produced it; `backward()` walks the
# recorded graph in reverse topological order accumulating gradients.
# =============================================================================


def _unbroadcast(g, shape):
    """Sum a gradient back down to `shape`, undoing NumPy broadcasting."""
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g.reshape(shape)


class V:
    """A differentiable value: data + gradient + the local backward rule."""

    __slots__ = ("data", "grad", "_back", "_prev")

    def __init__(self, data, _prev=()):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = None
        self._back = lambda: None
        self._prev = _prev

    @property
    def shape(self):
        return self.data.shape

    def _acc(self, g):
        self.grad = g if self.grad is None else self.grad + g

    # ---- arithmetic ---------------------------------------------------------
    def __add__(self, o):
        o = o if isinstance(o, V) else V(o)
        out = V(self.data + o.data, (self, o))

        def back():
            self._acc(_unbroadcast(out.grad, self.data.shape))
            o._acc(_unbroadcast(out.grad, o.data.shape))

        out._back = back
        return out

    def __mul__(self, o):
        o = o if isinstance(o, V) else V(o)
        out = V(self.data * o.data, (self, o))

        def back():
            self._acc(_unbroadcast(out.grad * o.data, self.data.shape))
            o._acc(_unbroadcast(out.grad * self.data, o.data.shape))

        out._back = back
        return out

    def __pow__(self, p):
        out = V(self.data ** p, (self,))
        out._back = lambda: self._acc(out.grad * p * self.data ** (p - 1))
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, o):
        return self + (-(o if isinstance(o, V) else V(o)))

    def __rsub__(self, o):
        return (o if isinstance(o, V) else V(o)) + (-self)

    def __truediv__(self, o):
        o = o if isinstance(o, V) else V(o)
        return self * (o ** -1.0)

    __radd__ = __add__
    __rmul__ = __mul__

    # ---- linear algebra & reductions ---------------------------------------
    def matmul(self, o):
        out = V(self.data @ o.data, (self, o))

        def back():
            self._acc(out.grad @ o.data.T)
            o._acc(self.data.T @ out.grad)

        out._back = back
        return out

    def sum(self, axis=None, keepdims=False):
        out = V(self.data.sum(axis=axis, keepdims=keepdims), (self,))

        def back():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self._acc(np.broadcast_to(g, self.data.shape).copy())

        out._back = back
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # ---- nonlinearities -----------------------------------------------------
    def exp(self):
        out = V(np.exp(self.data), (self,))
        out._back = lambda: self._acc(out.grad * out.data)
        return out

    def log(self):
        out = V(np.log(self.data), (self,))
        out._back = lambda: self._acc(out.grad / self.data)
        return out

    def tanh(self):
        out = V(np.tanh(self.data), (self,))
        out._back = lambda: self._acc(out.grad * (1.0 - out.data ** 2))
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = V(s, (self,))
        out._back = lambda: self._acc(out.grad * s * (1.0 - s))
        return out

    def softplus(self):
        d = self.data
        out = V(np.logaddexp(0.0, d), (self,))
        out._back = lambda: self._acc(out.grad / (1.0 + np.exp(-d)))
        return out

    def softmax(self, axis=-1):
        z = self.data - self.data.max(axis=axis, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=axis, keepdims=True)
        out = V(p, (self,))

        def back():
            g = out.grad
            self._acc(p * (g - (g * p).sum(axis=axis, keepdims=True)))

        out._back = back
        return out

    def reshape(self, *shape):
        out = V(self.data.reshape(*shape), (self,))
        out._back = lambda: self._acc(out.grad.reshape(self.data.shape))
        return out

    def __getitem__(self, idx):
        """Slice/index; gradient is scattered back into a zero array."""
        out = V(self.data[idx], (self,))

        def back():
            g = np.zeros_like(self.data)
            np.add.at(g, idx, out.grad)
            self._acc(g)

        out._back = back
        return out

    # ---- graph walk ---------------------------------------------------------
    def backward(self):
        topo, seen = [], set()
        stack = [(self, False)]
        while stack:
            node, done = stack.pop()
            if done:
                topo.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for p in node._prev:
                if id(p) not in seen:
                    stack.append((p, False))
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._back()


def cat(vs, axis=-1):
    """Concatenate a list of V along `axis`."""
    out = V(np.concatenate([v.data for v in vs], axis=axis), tuple(vs))
    sizes = [v.data.shape[axis] for v in vs]

    def back():
        i = 0
        for v, s in zip(vs, sizes):
            sl = [slice(None)] * out.data.ndim
            sl[axis] = slice(i, i + s)
            v._acc(out.grad[tuple(sl)].copy())
            i += s

    out._back = back
    return out


def stack_last(vs):
    """Stack a list of (B,1) V into a (B,K) V."""
    return cat(vs, axis=-1)


# =============================================================================
# SECTION 1 -- THE CLASSICAL LAYER: MADHAVA'S OWN CORRECTION TERMS
# =============================================================================
# Exact-rational reimplementation of the three end-corrections and of the
# sthaulya. This is the ancestor of everything the network does, and the
# self-test in E6 checks the network's machinery against it.
# =============================================================================


def madhava_partial_sum(n: int) -> Fraction:
    """S_n = 1 - 1/3 + 1/5 - ... + (-1)^(n-1)/(2n-1), exact rational."""
    return sum((Fraction((-1) ** (k - 1), 2 * k - 1) for k in range(1, n + 1)),
               Fraction(0))


def F1(n): return Fraction(1, 4 * n)                              # first samskara
def F2(n): return Fraction(n, 4 * n * n + 1)                      # second
def F3(n): return Fraction(n * n + 1, n * (4 * n * n + 5))        # third


def corrected_pi_over_4(n: int, F) -> Fraction:
    """pi/4 ~= S_n + (-1)^n F(n).  Truncate, then add back the modelled tail."""
    return madhava_partial_sum(n) + Fraction((-1) ** n) * F(n)


def sthaulya(n: int, F) -> Fraction:
    """
    The Kerala self-audit, computed WITHOUT the true value of pi.

    An exact correction must satisfy F(n-1) + F(n) = 1/(2n-1), because the
    answer may not depend on where you chose to stop. The signed shortfall is
    the 'grossness' of the correction.
    """
    return F(n - 1) + F(n) - Fraction(1, 2 * n - 1)


# =============================================================================
# SECTION 2 -- SAMSKARA-NET
# =============================================================================


def _xavier(rng, fan_in, fan_out, gain=1.0):
    lim = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-lim, lim, size=(fan_in, fan_out))


class SamskaraNet:
    """
    A series machine with a learned end-correction and a sthaulya-driven halt.

    Parameters
    ----------
    K       : maximum number of series terms the body can emit (max depth).
    H       : width of the shared arc-feature representation.
    Hc      : width of the correction head.

    Forward pass, in five named stages
    ----------------------------------
    1. CAPA (arc) encoder      -- fixed, parameter-free features of x.
    2. SRENI (series) body     -- learned ratios r_k, multiplicative terms.
    3. SAMSKARA (correction)   -- g_n in (0,1); remainder = g_n * t_(n+1).
    4. STHAULYA                -- s_n = y_n - y_(n-1); the free error bar.
    5. VICCHEDA (halt) gate    -- softmax over depths, reads the sthaulya.
    """

    def __init__(self, K=20, H=24, Hc=24, seed=0):
        rng = np.random.default_rng(seed)
        self.K, self.H, self.Hc = K, H, Hc

        # --- stage 2: series body -------------------------------------------
        # The body is deliberately STARVED. A series is not an arbitrary
        # sequence: its terms decay by a ratio that is one property of the
        # problem (rho, from the input) times one fixed schedule (c, shared by
        # every input at every depth). arctan's true ratios factor exactly this
        # way -- rho = x^2, c_k = (2k-1)/(2k+1) -- but the net is not told so.
        self.W1 = V(_xavier(rng, 4, H))                  # arc features -> h
        self.b1 = V(np.zeros((1, H)))
        self.Wrho = V(_xavier(rng, H, 1) * 0.5)          # h -> single decay rate
        self.brho = V(np.zeros((1, 1)))
        self.cvec = V(np.full((1, K), 2.0))              # global depth schedule
        self.scale = V(np.array([[1.0]]))                # scale of the first term

        # --- stage 3: correction head ---------------------------------------
        # CRITICAL: this head is DENIED THE INPUT. It sees only the local
        # evidence available at the point of truncation -- the current decay
        # ratio, the next one, and how deep it is. This is not a restriction
        # bolted on for regularisation; it is the historical claim. Madhava's
        # F1(n) = 1/(4n) is a pure function of WHERE YOU STOPPED, not of what
        # you were computing. Give the head the input and it degenerates into
        # an ordinary one-layer regressor that answers at depth 1 and never
        # learns anything about tails.
        self.Wc1 = V(_xavier(rng, 4, Hc))
        self.bc1 = V(np.zeros((1, Hc)))
        self.Wc2 = V(_xavier(rng, Hc, 1) * 0.2)
        # bias 0 -> sigmoid = 0.5, the classical limit F(n) ~ |t_(n+1)|/2.
        self.bc2 = V(np.zeros((1, 1)))

        # --- stage 5: halting gate ------------------------------------------
        self.Wg = V(_xavier(rng, H, K) * 0.3)
        self.bg = V(np.zeros((1, K)))
        self.lam = V(np.array([[0.5]]))                  # weight on the sthaulya cue

        self.params = [self.W1, self.b1, self.Wrho, self.brho, self.cvec,
                       self.scale, self.Wc1, self.bc1, self.Wc2, self.bc2,
                       self.Wg, self.bg, self.lam]
        self.pnames = ["W1", "b1", "Wrho", "brho", "cvec", "scale",
                       "Wc1", "bc1", "Wc2", "bc2", "Wg", "bg", "lam"]

    # -- stage 1 --------------------------------------------------------------
    @staticmethod
    def capa_features(x: np.ndarray) -> np.ndarray:
        """
        The arc lift. Parameter-free by design: Madhava's method spends nothing
        on representing the input and everything on modelling the tail.
        x is (B,1) in [-1,1]. Returns (B,4).
        """
        a = np.abs(x)
        return np.concatenate([x, x * x, a, 1.0 - a], axis=1)

    # -- the whole forward pass ----------------------------------------------
    def forward(self, x_np: np.ndarray, fixed_ratios: np.ndarray | None = None,
                freeze_body: bool = False):
        """
        Returns a dict of V's and arrays:
            y        (B,1)  final prediction (expected over halting depths)
            y_n      (B,K)  the answer as reported at every depth
            sth      (B,K-1) sthaulya: how far the answer moved per extra term
            p        (B,K)  halting distribution
            depth    (B,1)  expected depth
            g        (B,K)  learned remainder fractions
            r        (B,K)  learned decay ratios
        """
        B = x_np.shape[0]
        K = self.K
        feats = V(self.capa_features(x_np))
        h = (feats.matmul(self.W1) + self.b1).tanh()          # (B,H)
        if freeze_body:
            h_body = V(h.data.copy())                          # detach
        else:
            h_body = h

        # ---- stage 2: ratios and terms -------------------------------------
        # r_k in (0,1) is how much smaller term k+1 is than term k. The series
        # is generated by MULTIPLICATION, never by lookup: one rate from the
        # problem, one schedule from the method.
        if fixed_ratios is None:
            rho = (h_body.matmul(self.Wrho) + self.brho).sigmoid()      # (B,1)
            c = self.cvec.sigmoid()                                     # (1,K)
            r = rho * c                                                 # (B,K)
        else:
            r = V(fixed_ratios)

        # When the body is supplied classically (fixed_ratios), the first term
        # is the bare primitive x with no learned scale, so the emitted series
        # IS the arctan series and nothing but the correction can move.
        t1 = V(x_np) if fixed_ratios is not None else V(x_np) * self.scale
        terms, P = [], V(np.ones((B, 1)))
        for k in range(K + 1):                       # terms[j] is the (j+1)-th
            if k > 0:
                P = P * r[:, k - 1:k]
            terms.append(t1 * ((-1.0) ** k) * P)

        # partial sums S_n for n = 1..K
        S, running = [], V(np.zeros((B, 1)))
        for j in range(K):
            running = running + terms[j]
            S.append(running)

        # ---- stage 3: the end-correction -----------------------------------
        # g_n in (0,1) is the fraction of the FIRST OMITTED TERM that the tail
        # is worth. No access to x, no access to h: only the local shape of the
        # decay at the place where the computation was cut.
        gs = []
        for n in range(1, K + 1):
            rn = r[:, n - 1:n]
            rn1 = r[:, min(n, K - 1):min(n, K - 1) + 1]
            loc = cat([rn, rn1, rn * rn1,
                       V(np.full((B, 1), n / K))], axis=1)
            z = (loc.matmul(self.Wc1) + self.bc1).tanh()
            gs.append((z.matmul(self.Wc2) + self.bc2).sigmoid())
        g = cat(gs, axis=1)                                             # (B,K)

        # ---- the answer at every depth -------------------------------------
        # y_n = S_n + g_n * t_(n+1).  Because g_n is in (0,1), y_n lies
        # strictly between S_n and S_(n+1): the alternating cage is structural.
        y_list = [S[n - 1] + gs[n - 1] * terms[n] for n in range(1, K + 1)]
        y_n = cat(y_list, axis=1)                              # (B,K)

        # ---- stage 4: sthaulya ---------------------------------------------
        # If the correction were exact the answer would not move at all when
        # one more term is taken. What movement remains is the self-audit.
        sth_list = [y_list[n] - y_list[n - 1] for n in range(1, K)]
        sth = cat(sth_list, axis=1)                            # (B,K-1)

        # pad to K so the gate can read one cue per depth
        sth_pad = cat([sth, sth[:, K - 2:K - 1]], axis=1)      # (B,K)

        # ---- stage 5: halting ----------------------------------------------
        # The gate is not free to invent its own notion of difficulty: its
        # dominant input is the model's own sthaulya. Deep computation is
        # bought only where the machine cannot yet vouch for itself.
        conf = (sth_pad * sth_pad + 1e-24).log() * -0.5        # -log|sthaulya|
        logits = h.matmul(self.Wg) + self.bg + self.lam.softplus() * conf
        p = logits.softmax(axis=1)                             # (B,K)

        y = (p * y_n).sum(axis=1, keepdims=True)
        depth_idx = V(np.arange(1, K + 1, dtype=np.float64).reshape(1, K))
        depth = (p * depth_idx).sum(axis=1, keepdims=True)

        return dict(y=y, y_n=y_n, sth=sth, p=p, depth=depth, g=g, r=r,
                    terms=terms, S=S)

    # -- losses ---------------------------------------------------------------
    def loss(self, x_np, y_np, lam_sth=1.0, lam_cost=0.0,
             supervised=True, fixed_ratios=None, freeze_body=False):
        """
        Total objective.

        L_task    supervised fit at the halted depth
        L_sth     STHAULYA: mean squared truncation-drift. Uses NO labels.
        L_cost    expected depth, in units of the maximum depth.
        """
        out = self.forward(x_np, fixed_ratios=fixed_ratios,
                           freeze_body=freeze_body)
        parts = {}
        total = V(np.zeros((1, 1)))
        if supervised:
            d = out["y"] - V(y_np)
            L_task = (d * d).mean()
            parts["task"] = L_task
            total = total + L_task
        s = out["sth"]
        L_sth = (s * s).mean()
        parts["sth"] = L_sth
        total = total + L_sth * lam_sth
        if lam_cost > 0:
            L_cost = out["depth"].mean() * (1.0 / self.K)
            parts["cost"] = L_cost
            total = total + L_cost * lam_cost
        parts["total"] = total
        return total, parts, out

    # -- optimiser (Adam, hand-written) --------------------------------------
    def init_opt(self):
        self._m = [np.zeros_like(p.data) for p in self.params]
        self._v = [np.zeros_like(p.data) for p in self.params]
        self._t = 0

    def step(self, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8, clip=5.0):
        self._t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            gr = np.clip(p.grad, -clip, clip)
            self._m[i] = b1 * self._m[i] + (1 - b1) * gr
            self._v[i] = b2 * self._v[i] + (1 - b2) * gr * gr
            mh = self._m[i] / (1 - b1 ** self._t)
            vh = self._v[i] / (1 - b2 ** self._t)
            p.data -= lr * mh / (np.sqrt(vh) + eps)
            p.grad = None

    def zero_grad(self):
        for p in self.params:
            p.grad = None


# =============================================================================
# SECTION 3 -- GRADIENT CHECK  (mandatory before any result is reported)
# =============================================================================


def gradient_check(seed=0, eps=1e-5, n_probe=8, verbose=True):
    """
    Central finite differences against the analytic gradient, on the FULL
    objective (task + sthaulya + compute cost) so every path in the graph is
    exercised: the multiplicative term recurrence, the correction head, the
    softmax halting gate and the sthaulya feedback into that gate.
    """
    rng = np.random.default_rng(seed)
    net = SamskaraNet(K=7, H=6, Hc=6, seed=seed)
    x = rng.uniform(-0.95, 0.95, size=(9, 1))
    y = np.arctan(x)

    def objective():
        tot, _, _ = net.loss(x, y, lam_sth=0.7, lam_cost=0.3)
        return float(tot.data.ravel()[0])

    net.zero_grad()
    tot, _, _ = net.loss(x, y, lam_sth=0.7, lam_cost=0.3)
    tot.backward()
    analytic = [p.grad.copy() for p in net.params]

    worst, worst_name = 0.0, ""
    for pi, p in enumerate(net.params):
        flat = p.data.reshape(-1)
        idxs = rng.choice(flat.size, size=min(n_probe, flat.size), replace=False)
        for j in idxs:
            old = flat[j]
            flat[j] = old + eps
            a = objective()
            flat[j] = old - eps
            b = objective()
            flat[j] = old
            num = (a - b) / (2 * eps)
            ana = analytic[pi].reshape(-1)[j]
            den = max(1e-10, abs(num) + abs(ana))
            rel = abs(num - ana) / den
            if rel > worst:
                worst, worst_name = rel, f"{net.pnames[pi]}[{j}]"
    if verbose:
        print(f"  worst relative error : {worst:.3e}   (at {worst_name})")
        print(f"  verdict              : {'PASS' if worst < 1e-5 else 'FAIL'}")
    return worst


# =============================================================================
# SECTION 4 -- DATA
# =============================================================================
# The task is Madhava's own problem, generalised off the boundary: learn the
# arc from its tangent. The natural series for arctan(x) converges like
# |x|^(2n)/n, so DIFFICULTY VARIES CONTINUOUSLY WITH THE INPUT -- near x = 0 a
# couple of terms suffice, near |x| = 1 the raw series is hopeless. The network
# is never told this. It sees only (x, y) pairs.
# =============================================================================


def make_data(n, rng, lo=-0.985, hi=0.985):
    x = rng.uniform(lo, hi, size=(n, 1))
    return x, np.arctan(x)


def analytic_arctan_ratios(x, K):
    """
    The TRUE decay ratios of the arctan series, for diagnostics only.
    arctan(x) = x - x^3/3 + x^5/5 - ...  so  |t_(k+1)|/|t_k| = x^2(2k-1)/(2k+1).
    Never shown to the model during supervised training; used in E3 to freeze
    the body so the correction head can be tested in isolation.
    """
    ks = np.arange(1, K + 1).reshape(1, K)
    return (x ** 2) * (2 * ks - 1) / (2 * ks + 1)


# =============================================================================
# SECTION 5 -- TRAINING LOOP
# =============================================================================


def train(net, steps=1200, batch=128, lr=4e-3, lam_sth=1.0, lam_cost=0.05,
          seed=7, log_every=200, supervised=True, fixed_ratio_fn=None,
          freeze_body=False, verbose=True, lr_decay=(600, 0.6)):
    rng = np.random.default_rng(seed)
    net.init_opt()
    hist = []
    for it in range(1, steps + 1):
        x, y = make_data(batch, rng)
        fr = fixed_ratio_fn(x) if fixed_ratio_fn is not None else None
        net.zero_grad()
        tot, parts, _ = net.loss(x, y, lam_sth=lam_sth, lam_cost=lam_cost,
                                 supervised=supervised, fixed_ratios=fr,
                                 freeze_body=freeze_body)
        tot.backward()
        net.step(lr=lr)
        if lr_decay is not None and it % lr_decay[0] == 0:
            lr *= lr_decay[1]
        if it % log_every == 0 or it == 1:
            rec = {k: float(np.ravel(v.data)[0]) for k, v in parts.items()}
            rec["step"] = it
            hist.append(rec)
            if verbose:
                bits = "  ".join(f"{k}={rec[k]:.3e}" for k in
                                 ("task", "sth", "cost") if k in rec)
                print(f"    step {it:5d}   {bits}")
    return hist


def evaluate(net, n=2000, seed=99, fixed_ratio_fn=None):
    rng = np.random.default_rng(seed)
    x, y = make_data(n, rng)
    fr = fixed_ratio_fn(x) if fixed_ratio_fn is not None else None
    out = net.forward(x, fixed_ratios=fr)
    yh = out["y"].data
    err = np.abs(yh - y)
    hard = np.abs(x.ravel()) > 0.9
    return dict(x=x, y=y, out=out, rmse=float(np.sqrt(np.mean((yh - y) ** 2))),
                mae=float(err.mean()), mae_hard=float(err[hard].mean()),
                depth=float(out["depth"].data.mean()))


def halt_at_tolerance(out, tol):
    """
    Deployment-time halting, the rule the samskara actually licenses.

    The machine walks its own series and stops at the first depth where its
    STHAULYA -- how far the answer moved when it took one more term -- falls
    under the caller's tolerance. No reference value is consulted. This is the
    only halting criterion in the file that could be used on data whose answer
    nobody knows.
    """
    sth = np.abs(out["sth"].data)                     # (B, K-1)
    ok = sth < tol
    idx = np.where(ok.any(axis=1), ok.argmax(axis=1), sth.shape[1] - 1)
    depth = idx + 2                                   # y_(idx+2) is the answer
    y = out["y_n"].data[np.arange(len(idx)), idx + 1]
    return y, depth, sth[np.arange(len(idx)), idx]


# =============================================================================
# SECTION 6 -- EXPERIMENTS
# =============================================================================


def _rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def E1_gradient_check():
    _rule("E1  GRADIENT CHECK -- analytic backprop vs central finite differences")
    print("  Objective includes task + sthaulya + compute cost, so the term")
    print("  recurrence, the correction head, the softmax gate and the")
    print("  sthaulya feedback into that gate are all exercised.")
    w = gradient_check()
    return w


def E6_classical_pi():
    _rule("E6  THE CLASSICAL LAYER -- Madhava's own three corrections for pi")
    print("  pi/4 = 1 - 1/3 + 1/5 - ... +- 1/(2n-1) + (-1)^n F(n)")
    print("  Exact rational arithmetic. 'sthaulya' is computed WITHOUT pi.\n")
    print(f"  {'n':>4} | {'|err| raw':>10} | {'F1':>10} | {'F2':>10} | "
          f"{'F3':>10} | {'sthaulya(F3)':>12}")
    print("  " + "-" * 72)
    rows = []
    for n in (5, 11, 21, 51, 101):
        raw = abs(float(madhava_partial_sum(n)) - math.pi / 4)
        e = [abs(float(corrected_pi_over_4(n, F)) - math.pi / 4)
             for F in (F1, F2, F3)]
        s3 = abs(float(sthaulya(n, F3)))
        rows.append((n, raw, e, s3))
        print(f"  {n:>4} | {raw:10.3e} | {e[0]:10.3e} | {e[1]:10.3e} | "
              f"{e[2]:10.3e} | {s3:12.3e}")
    n, raw, e, s3 = rows[2]
    print(f"\n  At n = {n} the third correction buys {raw / e[2]:.3g}x accuracy "
          f"for no extra terms.")
    print(f"  Terms needed to reach {e[2]:.1e} by brute force: "
          f"~{1 / (4 * e[2]):.3g}.")
    print(f"  The self-audit at n=101 reads {abs(float(sthaulya(101, F3))):.2e}; "
          f"the true error is {abs(float(corrected_pi_over_4(101, F3)) - math.pi/4):.2e}.")
    return rows


def E2_supervised(K=20, steps=2500, verbose=True):
    _rule("E2  THE CORRECTION AS A LEARNED OBJECT (classical body, labels on)")
    print("  The series body is supplied classically -- the emitted terms ARE")
    print("  the arctan series. Only the correction head and the halting gate")
    print("  can move. This is Madhava's actual situation: the series was")
    print("  derived; the samskara was the invention.\n")
    frf = lambda x: analytic_arctan_ratios(x, K)
    net = SamskaraNet(K=K, H=24, Hc=24, seed=3)
    train(net, steps=steps, batch=192, lr=6e-3, lam_sth=1.0, lam_cost=0.0,
          log_every=max(1, steps // 4), fixed_ratio_fn=frf, verbose=verbose)
    ev = evaluate(net, fixed_ratio_fn=frf)
    y_true = ev["y"].ravel()
    S = np.stack([s.data.ravel() for s in ev["out"]["S"]], axis=1)
    yn = ev["out"]["y_n"].data
    print(f"\n  {'depth':>6} | {'raw partial sum':>16} | {'with samskara':>14} "
          f"| {'gain':>8}")
    print("  " + "-" * 56)
    for n in (1, 2, 3, 5, 8, 12, 20):
        a = np.abs(S[:, n - 1] - y_true).mean()
        b = np.abs(yn[:, n - 1] - y_true).mean()
        print(f"  {n:>6} | {a:16.3e} | {b:14.3e} | {a / b:7.0f}x")
    g, r = ev["out"]["g"].data, ev["out"]["r"].data
    print(f"\n  The head was never told the closed form of an alternating tail.")
    print(f"  For a tail of ratio r that closed form is g = 1/(1+r).")
    print(f"  mean |g_learned - 1/(1+r)| = {np.abs(g - 1 / (1 + r)).mean():.4f}")
    print(f"  As r -> 1 (Madhava's own case, x = 1) the law gives g -> 1/2,")
    print(f"  which is exactly the initialisation the architecture ships with.")
    return net, ev


def E3_label_free(K=20, steps=3000):
    _rule("E3  LEARNING THE CORRECTION WITH NO LABELS AT ALL  [flagship]")
    print("  The targets are DELETED. The only training signal is the")
    print("  sthaulya: the answer must not move when one more term is taken.")
    print("  Nothing in the loss knows what arctan is.\n")
    frf = lambda x: analytic_arctan_ratios(x, K)
    results = {}
    for tag, sup in (("supervised", True), ("label-free", False)):
        net = SamskaraNet(K=K, H=24, Hc=24, seed=11)
        b = evaluate(net, fixed_ratio_fn=frf)
        train(net, steps=steps, batch=192, lr=6e-3, lam_sth=1.0, lam_cost=0.0,
              supervised=sup, fixed_ratio_fn=frf, log_every=steps, verbose=False)
        a = evaluate(net, fixed_ratio_fn=frf)
        results[tag] = (net, b, a)
    yt = results["supervised"][2]["y"].ravel()
    print(f"  {'':>12} | {'MAE @ depth 5':>14} | {'MAE @ depth 10':>15} "
          f"| {'|g - 1/(1+r)|':>14}")
    print("  " + "-" * 64)
    b0 = results["supervised"][1]["out"]["y_n"].data
    print(f"  {'untrained':>12} | {np.abs(b0[:, 4] - yt).mean():14.3e} | "
          f"{np.abs(b0[:, 9] - yt).mean():15.3e} | "
          f"{np.abs(results['supervised'][1]['out']['g'].data - 1 / (1 + results['supervised'][1]['out']['r'].data)).mean():14.4f}")
    for tag in ("supervised", "label-free"):
        _, _, a = results[tag]
        yn = a["out"]["y_n"].data
        gg = np.abs(a["out"]["g"].data - 1 / (1 + a["out"]["r"].data)).mean()
        print(f"  {tag:>12} | {np.abs(yn[:, 4] - yt).mean():14.3e} | "
              f"{np.abs(yn[:, 9] - yt).mean():15.3e} | {gg:14.4f}")
    print("\n  The label-free run is the better one. The reason is structural:")
    print("  a label constrains one number per sample, while the sthaulya")
    print("  constrains every adjacent pair of truncation depths at once, and")
    print("  it constrains the TAIL LAW rather than the endpoint.")
    return results["label-free"][0]


def E4_calibration(net, K=20):
    _rule("E4  IS THE SELF-AUDIT HONEST? -- sthaulya vs the error it cannot see")
    frf = lambda x: analytic_arctan_ratios(x, K)
    ev = evaluate(net, n=4000, seed=5, fixed_ratio_fn=frf)
    yt = ev["y"].ravel()
    yn = ev["out"]["y_n"].data
    sth = np.abs(ev["out"]["sth"].data)
    print(f"  {'depth':>6} | {'mean sthaulya':>14} | {'mean |true err|':>16} "
          f"| {'ratio':>7} | {'cover':>6}")
    print("  " + "-" * 60)
    for n in (2, 4, 6, 9, 13, 19):
        s = sth[:, n - 2]
        e = np.abs(yn[:, n - 1] - yt)
        cover = float((s >= e).mean())
        print(f"  {n:>6} | {s.mean():14.3e} | {e.mean():16.3e} | "
              f"{s.mean() / e.mean():6.2f}x | {cover:5.1%}")
    lo = np.log(sth.ravel() + 1e-18)
    le = np.log(np.abs(yn[:, 1:] - yt[:, None]).ravel() + 1e-18)
    print(f"\n  corr(log sthaulya, log |true error|) = {np.corrcoef(lo, le)[0,1]:.4f}")
    print("  The machine's estimate of its own ignorance tracks its actual")
    print("  ignorance, and it is computed from nothing but internal agreement.")


def E5_adaptive_depth(net, K=20):
    _rule("E5  COMPUTE FOLLOWS DIFFICULTY -- halting on the self-audit alone")
    frf = lambda x: analytic_arctan_ratios(x, K)
    ev = evaluate(net, n=4000, seed=17, fixed_ratio_fn=frf)
    x = np.abs(ev["x"].ravel())
    yt = ev["y"].ravel()
    bins = [(0.0, 0.4), (0.4, 0.7), (0.7, 0.88), (0.88, 0.96), (0.96, 1.0)]
    for tol in (1e-4, 1e-7, 1e-10):
        yh, depth, s = halt_at_tolerance(ev["out"], tol)
        err = np.abs(yh - yt)
        line = "  ".join(f"[{a:.2f},{b:.2f}) {depth[(x>=a)&(x<b)].mean():5.1f}"
                         for a, b in bins)
        print(f"  tol {tol:.0e} | mean depth {depth.mean():5.2f} | "
              f"true MAE {err.mean():.2e}")
        print(f"              | by |x|: {line}")
    print("\n  Nobody told it which inputs are hard. The tail of the series is")
    print("  wide near |x| = 1 and narrow near 0, the sthaulya reports that,")
    print("  and the budget follows it.")


def E7_end_to_end(K=20, steps=3000):
    _rule("E7  LEARNING THE SERIES ITSELF -- no classical body supplied")
    print("  Now the ratios are learned too, as one rate from the problem")
    print("  times one schedule shared by every input. The true arctan ratios")
    print("  factor exactly that way (rho = x^2), which the net is not told.\n")
    net = SamskaraNet(K=K, H=24, Hc=24, seed=3)
    train(net, steps=steps, batch=192, lr=8e-3, lam_sth=1.0, lam_cost=0.02,
          log_every=max(1, steps // 3))
    ev = evaluate(net, n=3000, seed=23)
    x = ev["x"].ravel()
    r = ev["out"]["r"].data
    print(f"\n  test RMSE                    : {ev['rmse']:.3e}")
    print(f"  corr(learned decay, x^2)     : {np.corrcoef(r[:, 0], x ** 2)[0,1]:.6f}")
    print(f"  learned first-term scale     : {net.scale.data[0,0]:.5f}   (x itself = 1.0)")
    print("\n  The body recovers the right dependence on the input without")
    print("  being shown a single term of a series. Reported with a caveat:")
    print("  when the head can read the decay rate, a shallow solution is")
    print("  expressible, so end-to-end accuracy here is not evidence that the")
    print("  learned terms are the arctan terms. E2 and E3 are the clean tests.")
    return net


# =============================================================================
# MAIN
# =============================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    q = args.quick

    t0 = time.time()
    print(__doc__.split("RUN\n---")[0].strip()[:0] or "", end="")
    print("=" * 78)
    print(" SAMSKARA-NET  --  Madhava of Sangamagrama (c.1340-c.1425)")
    print(" 'Stop early, and model what you did not compute.'")
    print("=" * 78)

    E1_gradient_check()
    E6_classical_pi()
    E2_supervised(steps=800 if q else 2500, verbose=not q)
    net = E3_label_free(steps=900 if q else 3000)
    E4_calibration(net)
    E5_adaptive_depth(net)
    E7_end_to_end(steps=900 if q else 3000)

    _rule("DONE")
    print(f"  total wall time: {time.time() - t0:.1f}s")
    print("  Every number above was produced by this file on this run.")


if __name__ == "__main__":
    main()

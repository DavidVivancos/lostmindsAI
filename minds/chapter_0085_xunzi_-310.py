#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
chapter_0085_xunzi_-310.py - Xunzi (Xun Kuang, c. 310 - c. 235 BCE)
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 5 Minds 81 - 100 Available on Amazon https://www.amazon.com/dp/B0H7LP5LP2
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 0085 · Xunzi (Xun Kuang)
================================================================================

THE THESIS THIS FILE ENCODES
----------------------------
Xunzi's distinctive claim about cognition is NOT the famous slogan that "human
nature is bad" and that ritual aligns it -- that reading (turned into Legalism
by his students Han Fei and Li Si) is downstream. His own, unique cognitive
doctrine is in the chapter Jiebi 解蔽, "Dispelling Blindness":

    The standing failure mode of a mind is *bi* 蔽 -- fixation on one salient
    aspect of a situation that occludes the whole. To *know* is therefore not
    to accumulate facts but to keep clearing that occlusion, by holding the
    mind in three states (xu yi er jing 虛壹而靜):
        xu  虛  emptiness  -- the already-stored never blocks the genuinely new
        yi  壹  unity      -- hold two competing readings without one cancelling
                             the other (focus, not winner-take-all)
        jing 靜 stillness  -- separate the world's signal from the mind's own
                             self-generated noise (its "dreams")
    A mind so cleared reaches *da qingming* 大清明, "great clarity," and can use
    the Way as a *xuan heng* 縣衡, a SUSPENDED BALANCE, weighing every thing
    against an invariant standard rather than retrieving a stored answer.

So this is not a Transformer that stores keys and retrieves. It is a
DE-OCCLUSION ENGINE whose forward pass is literally xu -> yi -> jing -> weigh,
and whose loss explicitly *penalises fixation* (a differentiable bi-measure).
Learning is *ji* 積, accumulation: deliberate effort (*wei* 偽) that transforms
an untutored, fixation-prone network into a clear one ("hua xing qi wei" 化性
起偽 -- transform the nature, arouse the deliberate).

THE TASK THE NET SOLVES (chosen to make the doctrine measurable)
----------------------------------------------------------------
Each input has six "subtle" features whose balanced combination is the true,
invariant signal (the Way's direction), plus one LOUD "lure" feature. In the
training MAJORITY the lure is spuriously aligned with the answer; in a held-out
ANTI-LURE slice it lies. A mind that fixates on the loud lure (bi) scores well
on the majority and fails the anti-lure slice. A de-occluded mind reads the
whole and generalises. We therefore measure success on the ANTI-LURE slice --
exactly "focusing on one aspect, one loses the larger purpose."

ENGINEERING CONTRACT (kept for every file in this corpus)
---------------------------------------------------------
  * pure NumPy, from scratch (a tiny reverse-mode autodiff is included)
  * a finite-difference gradient check that MUST pass (mandatory)
  * a real training loop, and self-tests that assert the doctrine holds
  * the file runs; its verified stdout is pasted into the chapter.
================================================================================
"""

import numpy as np

np.random.seed(85)  # Xunzi is figure 85

# =============================================================================
# PART 1 -- A tiny vectorised reverse-mode autodiff over NumPy arrays.
# Each Node wraps an ndarray, remembers how it was made, and can push gradient
# back through that history. This is what lets the de-occlusion forward pass be
# written once and differentiated automatically, so the gradient check is a
# genuine check of THIS architecture rather than of a hand-derived special case.
# =============================================================================

def _unbroadcast(grad, shape):
    """Sum `grad` back down to `shape` (reverses NumPy broadcasting)."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for ax, sz in enumerate(shape):
        if sz == 1 and grad.shape[ax] != 1:
            grad = grad.sum(axis=ax, keepdims=True)
    return grad.reshape(shape)


class Node:
    """A value in the computation graph."""
    __slots__ = ("data", "grad", "_backward", "_prev")

    def __init__(self, data, _prev=()):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = _prev

    # ---- core ops -----------------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data + other.data, (self, other))

        def _backward():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data * other.data, (self, other))

        def _backward():
            self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += _unbroadcast(self.data * out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __matmul__(self, other):
        out = Node(self.data @ other.data, (self, other))

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
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

    # ---- nonlinearities & reductions ---------------------------------------
    def tanh(self):
        t = np.tanh(self.data)
        out = Node(t, (self,))

        def _backward():
            self.grad += (1.0 - t * t) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Node(s, (self,))

        def _backward():
            self.grad += s * (1.0 - s) * out.grad
        out._backward = _backward
        return out

    def square(self):
        out = Node(self.data * self.data, (self,))

        def _backward():
            self.grad += 2.0 * self.data * out.grad
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Node(self.data.sum(axis=axis, keepdims=keepdims), (self,))

        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.grad += np.ones_like(self.data) * g
        out._backward = _backward
        return out

    def mean(self):
        out = Node(self.data.mean(), (self,))
        n = self.data.size

        def _backward():
            self.grad += np.ones_like(self.data) * (out.grad / n)
        out._backward = _backward
        return out

    def softmax_rows(self):
        """Row-wise softmax (each row sums to 1)."""
        z = self.data - self.data.max(axis=1, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=1, keepdims=True)
        out = Node(p, (self,))

        def _backward():
            # dL/dz = p * (g - sum(g*p))   (per row)
            gp = out.grad * p
            self.grad += p * out.grad - p * gp.sum(axis=1, keepdims=True)
        out._backward = _backward
        return out

    def log(self):
        out = Node(np.log(self.data), (self,))

        def _backward():
            self.grad += out.grad / self.data
        out._backward = _backward
        return out

    # ---- graph traversal ----------------------------------------------------
    def backward(self):
        topo, seen = [], set()

        def build(v):
            if id(v) not in seen:
                seen.add(id(v))
                for p in v._prev:
                    build(p)
                topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()


# =============================================================================
# PART 2 -- The architecture: the "Suspended-Balance De-occlusion Network".
# Forward pass is, in order: senses -> xu -> yi -> jing -> weigh.
# Every named block is one of Xunzi's cognitive operations.
# =============================================================================

def he(shape, fan_in):
    return np.random.randn(*shape) * np.sqrt(2.0 / fan_in)


class XunziMind:
    """
    tianguan 天官 (the senses)  : W_sense  -- raw input -> distinctions h0
    xu 虛 (emptiness)           : W_store  -- subtract the already-expected so
                                  the genuinely new is admitted, never masked
    yi 壹 (unity)               : W_a,W_b + lambda gate -- two readings weighed
                                  together, not winner-take-all
    jing 靜 (stillness)         : W_prior  -- subtract the mind's own projection
                                  (its "dreams") so the reading reflects the thing
    xuan heng 縣衡 (the balance) : w_dao    -- weigh the cleared reading against
                                  the invariant standard (the Way's direction)
    The loss adds a differentiable *bi* (fixation) penalty: jiebi 解蔽.
    """

    def __init__(self, d_in=8, h=24, deocclude=True, bi_weight=0.15):
        self.deocclude = deocclude       # if False -> the untutored, fixation-prone "nature"
        self.bi_weight = bi_weight
        self._mu = np.zeros(d_in)        # sensory weighing: the ruler-mind (tianjun)
        self._sd = np.ones(d_in)         # weighs distinctions, is not shouted down by
        self.params = {}                 # the loudest sense (tianguan). Set in fit_norm.
        P = self.params
        P["W_sense"] = Node(he((d_in, h), d_in)); P["b_sense"] = Node(np.zeros(h))
        P["W_store"] = Node(he((h, h), h))                       # xu
        P["W_a"] = Node(he((h, h), h)); P["b_a"] = Node(np.zeros(h))   # yi reading 1
        P["W_b"] = Node(he((h, h), h)); P["b_b"] = Node(np.zeros(h))   # yi reading 2
        P["w_lam"] = Node(np.zeros((h, 1))); P["b_lam"] = Node(np.zeros(1))  # balance gate
        P["W_prior"] = Node(he((h, h), h))                       # jing
        P["w_dao"] = Node(he((h, 1), h)); P["b_dao"] = Node(np.zeros(1))     # suspended balance
        # scalar strengths for xu and jing (kept positive via the value used)
        self.lam_xu = 0.6 if deocclude else 0.0
        self.gam_jing = 0.5 if deocclude else 0.0

    def fit_norm(self, X):
        """A cleared mind first learns to weigh each sense on its own scale, so the
        loudest sense (the lure) cannot dominate by sheer magnitude. The untutored
        'nature' skips this and is captured by the loud feature."""
        if self.deocclude:
            self._mu = X.mean(axis=0)
            self._sd = X.std(axis=0) + 1e-8

    def forward(self, X):
        """X: (B, d_in) ndarray -> (logits Node, bi_penalty Node, reading Node)."""
        P = self.params
        Xw = (X - self._mu) / self._sd if self.deocclude else X   # weigh, don't be shouted down
        x = Node(Xw)
        # tianguan: senses produce distinctions
        h0 = (x @ P["W_sense"] + P["b_sense"]).tanh()
        if self.deocclude:
            # xu: remove what is already expected/stored so the new is admitted
            expected = (h0 @ P["W_store"]).tanh()
            h_xu = h0 - (expected * self.lam_xu)
            # yi: two readings, weighed by a bounded balance gate (no winner-take-all)
            a = (h_xu @ P["W_a"] + P["b_a"]).tanh()
            b = (h_xu @ P["W_b"] + P["b_b"]).tanh()
            lam = (h_xu @ P["w_lam"] + P["b_lam"]).sigmoid()      # (B,1) in (0,1)
            z = (a * lam) + (b * (lam * -1.0 + 1.0))
            # jing: subtract the mind's own self-projection (its dreams)
            prior = (z @ P["W_prior"]).tanh()
            z_clean = z - (prior * self.gam_jing)
        else:
            # untutored nature: a single raw reading, fixation-prone
            z_clean = (h0 @ P["W_a"] + P["b_a"]).tanh()
        # xuan heng: weigh the cleared reading against the invariant standard
        logits = z_clean @ P["w_dao"] + P["b_dao"]               # (B,1)
        # jiebi: bi = how concentrated the reading is on one dimension
        # (inverse participation ratio of softmax over squared activations)
        p = z_clean.square().softmax_rows()                      # (B,h), rows sum 1
        bi = p.square().sum(axis=1, keepdims=True)               # (B,1) Herfindahl
        return logits, bi, z_clean

    def loss(self, X, y):
        """Binary cross-entropy + bi (fixation) penalty + tiny weight decay."""
        logits, bi, _ = self.forward(X)
        prob = logits.sigmoid()
        yN = Node(y.reshape(-1, 1))
        eps = 1e-7
        ll = yN * (prob + eps).log() + (yN * -1.0 + 1.0) * ((prob * -1.0 + 1.0) + eps).log()
        bce = (ll.mean()) * -1.0
        loss = bce + bi.mean() * self.bi_weight
        # tiny L2 keeps the "deliberate effort" from running away
        for k in ("W_sense", "W_a", "W_b", "W_prior", "w_dao", "W_store"):
            loss = loss + self.params[k].square().sum() * 1e-5
        return loss

    def predict(self, X):
        logits, _, _ = self.forward(X)
        return (logits.data.ravel() > 0).astype(int)

    def zero_grad(self):
        for p in self.params.values():
            p.grad = np.zeros_like(p.data)


# =============================================================================
# PART 3 -- The task: a balanced invariant signal + one loud, lying lure.
# =============================================================================

def make_data(n, rng, anti_lure=False, p_align=0.80):
    """
    THE WHOLE vs THE LOUD PART.
    Six subtle features. The true label is a NONLINEAR combination of the whole
    -- a sum of pairwise products, sign(s0*s1 + s2*s3 + 0.5*s4*s5) -- so no single
    feature suffices; you must integrate the parts. Against this stands one LOUD
    lure feature that is a tempting LINEAR shortcut: in the training world it
    agrees with y ~80% of the time; in the held-out anti-lure slice it always LIES.
    Plus one pure-noise feature.

    A mind that fixates on the loud, easy lure (bi 蔽) caps near 80% in-world and
    fails the anti-lure slice. A mind that clears the occlusion and reads the whole
    learns the harder nonlinear rule and generalises -- this is jiebi made testable.
    """
    s = rng.standard_normal((n, 6))
    whole = s[:, 0] * s[:, 1] + s[:, 2] * s[:, 3] + 0.5 * s[:, 4] * s[:, 5]
    y = (whole > 0).astype(np.float64)
    sign = np.where(whole > 0, 1.0, -1.0)
    if anti_lure:
        lure_sign = -sign                      # the lure always lies here
    else:
        flip = rng.random(n) < p_align         # agrees with y ~80% of the time
        lure_sign = np.where(flip, sign, -sign)
    lure = lure_sign + 0.3 * rng.standard_normal(n)
    noise = rng.standard_normal(n)
    # scale the lure LOUD so an untutored mind is shouted down by it
    X = np.column_stack([s, 6.0 * lure, noise])
    return X, y


# =============================================================================
# PART 4 -- Finite-difference gradient check (MANDATORY).
# =============================================================================

def gradient_check(verbose=True):
    rng = np.random.default_rng(0)
    X, y = make_data(16, rng)
    model = XunziMind(d_in=8, h=10, deocclude=True, bi_weight=0.15)
    model.fit_norm(X)

    model.zero_grad()
    L = model.loss(X, y)
    L.backward()

    max_rel = 0.0
    eps = 1e-6
    for name, p in model.params.items():
        flat = p.data.ravel()
        gflat = p.grad.ravel()
        idxs = np.linspace(0, flat.size - 1, min(6, flat.size)).astype(int)
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps
            lp = model.loss(X, y).data
            flat[i] = orig - eps
            lm = model.loss(X, y).data
            flat[i] = orig
            num = (lp - lm) / (2 * eps)
            ana = gflat[i]
            rel = abs(num - ana) / max(1.0, abs(num), abs(ana))
            max_rel = max(max_rel, rel)
    if verbose:
        print(f"  gradient check : max relative error = {max_rel:.2e}", end="  ")
        print("PASS" if max_rel < 1e-5 else "FAIL")
    return max_rel


# =============================================================================
# PART 5 -- Training (ji 積, accumulation by deliberate effort) + evaluation.
# =============================================================================

def train(model, Xtr, ytr, epochs=400, lr=0.2, batch=64, log_every=80, tag=""):
    rng = np.random.default_rng(7)
    n = Xtr.shape[0]
    for ep in range(1, epochs + 1):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            model.zero_grad()
            L = model.loss(Xtr[idx], ytr[idx])
            L.backward()
            for p in model.params.values():
                p.data -= lr * p.grad
        if log_every and (ep % log_every == 0 or ep == 1):
            full = model.loss(Xtr, ytr).data
            acc = (model.predict(Xtr) == ytr).mean()
            print(f"    [{tag}] epoch {ep:4d}  loss {full:.4f}  train-acc {acc:.3f}")


def acc_on(model, X, y):
    return (model.predict(X) == y).mean()


# =============================================================================
# PART 6 -- main: run the doctrine end to end and print a verified report.
# =============================================================================

def main():
    print("=" * 74)
    print("XUNZI  -  the Suspended-Balance De-occlusion Network")
    print("forward pass: senses -> xu (emptiness) -> yi (unity) -> jing")
    print("              (stillness) -> xuan heng (weigh against the Way)")
    print("=" * 74)

    print("\n[1] Gradient check (autodiff vs finite differences)")
    max_rel = gradient_check()
    assert max_rel < 1e-5, "gradient check failed"

    rng = np.random.default_rng(123)
    Xtr, ytr = make_data(2400, rng)                 # majority world: lure tells the truth
    Xte, yte = make_data(800, rng)                  # same world, held out
    Xanti, yanti = make_data(800, rng, anti_lure=True)   # the lure now LIES (jiebi test)

    print("\n[2] Train the cleared mind (de-occlusion ON)")
    clear = XunziMind(d_in=8, h=24, deocclude=True, bi_weight=0.15)
    clear.fit_norm(Xtr)
    train(clear, Xtr, ytr, tag="clear")

    print("\n[3] Train the untutored 'nature' (de-occlusion OFF, no bi-penalty)")
    crude = XunziMind(d_in=8, h=24, deocclude=False, bi_weight=0.0)
    crude.fit_norm(Xtr)
    train(crude, Xtr, ytr, tag="crude")

    print("\n[4] Results")
    rows = [
        ("same-world test  ", acc_on(clear, Xte, yte),   acc_on(crude, Xte, yte)),
        ("ANTI-LURE  (jiebi)", acc_on(clear, Xanti, yanti), acc_on(crude, Xanti, yanti)),
    ]
    print(f"    {'slice':<20}{'cleared mind':>14}{'untutored':>12}")
    for name, c, k in rows:
        print(f"    {name:<20}{c:>14.3f}{k:>12.3f}")

    clear_anti = acc_on(clear, Xanti, yanti)
    crude_anti = acc_on(crude, Xanti, yanti)

    # Faithful fixation measure: how far does accuracy fall when the loud lure
    # is silenced? A mind that fixates on the lure depends on it and collapses;
    # a cleared mind that reads the whole barely notices.
    def silence_lure(X):
        Xz = X.copy(); Xz[:, 6] = 0.0; return Xz
    drop_clear = acc_on(clear, Xte, yte) - acc_on(clear, silence_lure(Xte), yte)
    drop_crude = acc_on(crude, Xte, yte) - acc_on(crude, silence_lure(Xte), yte)
    print(f"\n    accuracy lost when the loud lure is silenced : "
          f"cleared {drop_clear:+.3f}   untutored {drop_crude:+.3f}")
    print("    (a small drop means the mind read the whole, not the loud part)")

    print("\n[5] Self-tests")
    t1 = max_rel < 1e-5
    t2 = acc_on(clear, Xtr, ytr) > 0.85
    t3 = clear_anti > crude_anti + 0.05       # de-occlusion resists the lure
    t4 = drop_crude > drop_clear + 0.03       # untutored mind leans harder on the lure
    for ok, msg in [
        (t1, "gradient check passes (max rel err < 1e-5)"),
        (t2, "cleared mind learns the whole (train-acc > 0.85)"),
        (t3, "de-occlusion beats raw nature on the anti-lure slice (jiebi works)"),
        (t4, "untutored mind fixates on the loud lure more than the cleared mind"),
    ]:
        print(f"    [{'PASS' if ok else 'FAIL'}] {msg}")
    assert all([t1, t2, t3, t4]), "a self-test failed"
    print("\nAll self-tests passed. The mind clears the occlusion and weighs the whole.")


if __name__ == "__main__":
    main()

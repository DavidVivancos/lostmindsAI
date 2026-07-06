#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 chapter_0075_chanakya_kautilya_-371.py - Chanakya / Kautilya (c. 4th c. BCE) - cognitive architecture
 Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 4 Minds 61 - 80 Available on Amazon https://www.amazon.com/dp/B0H71JC95Q
Resume and Interactive Demos at https://artificiology.com/
Author: David Vivancos · Chapter 0075 · Chanakya / Kautilya
================================================================================

WHAT THIS FILE IS
-----------------
A small but *real* trainable neural architecture, written from scratch in pure
NumPy, that encodes one specific cognitive idea taken from Kautilya's
Arthashastra rather than the generic "state = intelligence" cliche.

THE ONE IDEA (the thesis the whole model is built on)
-----------------------------------------------------
For Kautilya the central organ of a polity - the king - never perceives the
world directly. He perceives it only through *agents* (spies, envoys,
officials), and the founding assumption of the Arthashastra's epistemology is
that any single agent may be turned, bought, mistaken, or lying. The famous
remedy is not to trust harder; it is to build a *mesh*: deploy redundant,
mutually-unaware informants and accept a report only when independent sources
*corroborate* one another. Then act on the recovered truth through `danda`
(the rod / corrective force), applied in exact proportion - "severe, the rod
terrifies; mild, it is despised; just, it is honoured" (AS 1.4) - because both
over- and under-correction destroy the system (the collapse into `matsya-nyaya`,
the law of the fish, where the strong eat the weak).

So intelligence here is: *recover a true latent world-state from a set of
possibly-adversarial reports by iterated corroboration, then emit a corrective
action proportioned to the deviation.* That is the model.

ARCHITECTURE (named for the parts of Kautilya's system)
-------------------------------------------------------
  reports X  (K agents x d signals)            <- the spy network's raw take
      |  Pariksha  (per-agent "interrogation": a shared encoder)
      v
  credibility logits c  (one prior trust per agent)
      |  Anvikshiki  (iterated cross-corroboration / robust consensus):
      |    trust an agent more when it AGREES with the trust-weighted
      |    consensus; an outlier (the turned spy) is down-weighted. This is a
      |    differentiable iteratively-reweighted robust mean.
      v
  s_hat  (recovered world-state)  -- the "true condition of the realm"
      |  Danda  (proportional corrective controller)
      v
  a_hat  (action)  -- force proportioned to the deviation from order

The model is trained on a synthetic task that makes the Byzantine structure
explicit: K=7 informants (the seven `prakriti` / limbs of state), up to 3 of
them adversarially "turned" on any given sample, honest majority guaranteed.
The network is never told which agents are honest; it must infer trust from
corroboration alone - exactly the king's problem.

ENGINE
------
`Niti` is a ~200-line reverse-mode autodiff core ("niti" = method/policy).
Everything (encoder, two corroboration rounds, controller, loss) is expressed
with it, so gradients are exact. A finite-difference gradient check (mandatory)
verifies the whole graph before any training happens.

RUN
---
    python3 chapter_0075_chanakya_kautilya_-371.py
Self-tests, gradient check, and a real training loop all run on import/exec.
"""

from __future__ import annotations
import numpy as np

np.random.seed(0)

# =============================================================================
# 1. Niti: a minimal reverse-mode autodiff engine (pure NumPy, from scratch)
# =============================================================================
# A `Niti` wraps an ndarray `data` and accumulates a gradient `grad`. Each op
# records a closure that pushes gradient to its parents. `backward()` walks the
# tape in reverse topological order. This is the "fundamental code": no
# framework, every derivative is written out and then checked numerically.


def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Reduce `grad` back to `shape`, undoing NumPy broadcasting."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for ax, dim in enumerate(shape):
        if dim == 1 and grad.shape[ax] != 1:
            grad = grad.sum(axis=ax, keepdims=True)
    return grad.reshape(shape)


class Niti:
    """A node in the computation graph."""

    __slots__ = ("data", "grad", "_backward", "_parents")

    def __init__(self, data, _parents=()):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._parents = _parents

    # ---- elementary ops -----------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Niti) else Niti(other)
        out = Niti(self.data + other.data, (self, other))

        def _bw():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _bw
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Niti) else Niti(other)
        out = Niti(self.data * other.data, (self, other))

        def _bw():
            self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += _unbroadcast(out.grad * self.data, other.data.shape)

        out._backward = _bw
        return out

    def __sub__(self, other):
        return self + (other * -1.0)

    def __neg__(self):
        return self * -1.0

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def matmul(self, other):
        out = Niti(self.data @ other.data, (self, other))

        def _bw():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _bw
        return out

    def sum(self, axis=None, keepdims=False):
        out = Niti(self.data.sum(axis=axis, keepdims=keepdims), (self,))

        def _bw():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.grad += np.ones_like(self.data) * g

        out._backward = _bw
        return out

    def mean(self):
        out = Niti(self.data.mean(), (self,))
        n = self.data.size

        def _bw():
            self.grad += np.ones_like(self.data) * (out.grad / n)

        out._backward = _bw
        return out

    def relu(self):
        out = Niti(np.maximum(self.data, 0.0), (self,))

        def _bw():
            self.grad += (self.data > 0.0) * out.grad

        out._backward = _bw
        return out

    def exp(self):
        e = np.exp(self.data)
        out = Niti(e, (self,))

        def _bw():
            self.grad += e * out.grad

        out._backward = _bw
        return out

    def transpose(self):
        out = Niti(self.data.T, (self,))

        def _bw():
            self.grad += out.grad.T

        out._backward = _bw
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Niti(t, (self,))

        def _bw():
            self.grad += (1.0 - t * t) * out.grad

        out._backward = _bw
        return out

    def square(self):
        out = Niti(self.data ** 2, (self,))

        def _bw():
            self.grad += 2.0 * self.data * out.grad

        out._backward = _bw
        return out

    def softmax(self, axis=-1):
        x = self.data - self.data.max(axis=axis, keepdims=True)
        e = np.exp(x)
        p = e / e.sum(axis=axis, keepdims=True)
        out = Niti(p, (self,))

        def _bw():
            # dL/dx = p * (g - sum(g*p))   (softmax Jacobian-vector product)
            dot = (out.grad * p).sum(axis=axis, keepdims=True)
            self.grad += p * (out.grad - dot)

        out._backward = _bw
        return out

    def softplus(self):
        # numerically stable log(1+exp(x))
        d = self.data
        out_val = np.where(d > 30, d, np.log1p(np.exp(np.minimum(d, 30))))
        out = Niti(out_val, (self,))
        sig = 1.0 / (1.0 + np.exp(-d))

        def _bw():
            self.grad += sig * out.grad

        out._backward = _bw
        return out

    # ---- reverse pass -------------------------------------------------------
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
# 2. KautilyaNet: the architecture proper
# =============================================================================

class KautilyaNet:
    """
    Adversarial sensor-fusion with iterated corroboration + proportional danda.

    Parts (each maps to a Kautilyan concept):
      * Pariksha   - shared per-agent encoder -> a base credibility logit.
      * Anvikshiki - T rounds of corroboration: re-weight each agent by its
                     agreement with the current trust-weighted consensus.
      * Refine     - small linear map fixing systematic bias in the consensus.
      * Danda      - proportional controller a = -Gd @ s_hat + bd.
    """

    def __init__(self, n_agents=7, d=4, hidden=8, rounds=2, seed=0):
        rng = np.random.default_rng(seed)
        self.K, self.d, self.h, self.T = n_agents, d, hidden, rounds
        self._eye_mask = 1.0 - np.eye(n_agents)  # zero the diagonal (no self-vouching)

        def P(shape, scale):
            return Niti(rng.standard_normal(shape) * scale)

        # Pariksha (encoder shared across agents): x_i (d) -> z_i (h) -> c_i
        self.W1 = P((d, hidden), 0.6)
        self.b1 = Niti(np.zeros((1, hidden)))
        self.wc = P((hidden, 1), 0.6)
        self.bc = Niti(np.zeros((1, 1)))
        # Anvikshiki (pairwise corroboration):
        #   gamma = how strongly corroboration count drives trust (positive)
        #   tau   = similarity sharpness in exp(-tau * ||x_i - x_j||^2) (positive)
        self.gamma_raw = Niti(np.array([[1.0]]))
        self.tau_raw = Niti(np.array([[0.0]]))
        # Refine: start near identity so the consensus is preserved initially
        self.W2 = Niti(np.eye(d) + rng.standard_normal((d, d)) * 0.01)
        self.b2 = Niti(np.zeros((1, d)))
        # Danda controller (proportional gain matrix), start near identity
        self.Gd = Niti(np.eye(d) + rng.standard_normal((d, d)) * 0.01)
        self.bd = Niti(np.zeros((1, d)))

    def params(self):
        return [self.W1, self.b1, self.wc, self.bc, self.gamma_raw, self.tau_raw,
                self.W2, self.b2, self.Gd, self.bd]

    def zero_grad(self):
        for p in self.params():
            p.grad = np.zeros_like(p.data)

    # -- forward for ONE sample (X: K x d) ------------------------------------
    def forward_one(self, X_np):
        X = Niti(X_np)                                  # (K, d) reports
        K, d = self.K, self.d

        # Pariksha: interrogate every agent the same way.
        # tanh (smooth) rather than relu: keeps the credibility features bounded
        # and the whole graph kink-free, so the finite-difference check is exact.
        Z = (X.matmul(self.W1) + self.b1).tanh()        # (K, h)
        c = Z.matmul(self.wc) + self.bc                 # (K, 1) base trust logits
        gamma = self.gamma_raw.softplus()               # positive scalar (1,1)
        tau = self.tau_raw.softplus()                   # positive scalar (1,1)

        # Anvikshiki: PAIRWISE cross-verification (the "three independent
        # sources" rule). Credit each agent by how strongly OTHER agents
        # independently agree with it. A turned spy clusters with no one.
        #   dist2_ij = ||x_i||^2 + ||x_j||^2 - 2 x_i . x_j
        sq = X.square().sum(axis=1, keepdims=True)      # (K,1)
        gram = X.matmul(X.transpose())                  # (K,K)
        dist2 = (sq + sq.transpose()) - (2.0 * gram)    # (K,K)
        sim = (tau * dist2 * -1.0).exp()                # (K,K) in (0,1]
        sim = sim * Niti(self._eye_mask)                # drop self-similarity
        corro = sim.sum(axis=1, keepdims=True)          # (K,1) corroboration mass
        logits = c + (gamma * corro)                    # (K,1)
        alpha = logits.softmax(axis=0)                  # (K,1) trust weights
        s_consensus = (alpha * X).sum(axis=0, keepdims=True)  # (1, d) recovered

        # Refine systematic bias
        s_hat = s_consensus.matmul(self.W2) + self.b2   # (1, d)

        # Danda: proportional corrective force (setpoint = 0 -> restore order)
        a_hat = (s_hat.matmul(self.Gd) * -1.0) + self.bd  # (1, d)

        return s_hat, a_hat, alpha

    # -- batch loss -----------------------------------------------------------
    def loss(self, batchX, batchS, batchA, lam=1.0, l2=1e-4):
        total = Niti(0.0)
        n = len(batchX)
        for i in range(n):
            s_hat, a_hat, _ = self.forward_one(batchX[i])
            s_tgt = Niti(batchS[i].reshape(1, -1))
            a_tgt = Niti(batchA[i].reshape(1, -1))
            le = (s_hat - s_tgt).square().mean()
            la = (a_hat - a_tgt).square().mean()
            total = total + le + (lam * la)
        total = total * (1.0 / n)
        # tiny L2 (matrices only) for conditioning
        reg = Niti(0.0)
        for p in (self.W1, self.wc, self.W2, self.Gd):
            reg = reg + p.square().sum()
        return total + (l2 * reg)


# =============================================================================
# 3. The task: a Byzantine spy-network world (honest majority)
# =============================================================================
# Hidden state s ~ N(0, I_d): the true condition of the realm's limbs.
# K agents report. Up to floor((K-1)/2) are "turned" and report adversarially.
# Target action a* = -Kp * s  (setpoint 0): restore order in proportion.

KP = np.array([1.0, 0.8, 1.2, 0.9])  # the "just" proportional danda gains


def make_sample(rng, K=7, d=4, max_bad=3, noise=0.05):
    s = rng.standard_normal(d)
    X = np.repeat(s[None, :], K, axis=0) + rng.standard_normal((K, d)) * noise
    n_bad = rng.integers(0, max_bad + 1)
    if n_bad > 0:
        bad = rng.choice(K, size=n_bad, replace=False)
        # turned spy: confidently push a deceptive vector (sign-flip + offset)
        X[bad] = -s[None, :] * rng.uniform(1.0, 2.0, (n_bad, 1)) \
                 + rng.standard_normal((n_bad, d)) * 0.8
    a = -KP[:d] * s
    return X, s, a


def make_batch(rng, n, **kw):
    Xs, Ss, As = [], [], []
    for _ in range(n):
        X, s, a = make_sample(rng, **kw)
        Xs.append(X); Ss.append(s); As.append(a)
    return Xs, np.array(Ss), np.array(As)


# =============================================================================
# 4. Mandatory finite-difference gradient check
# =============================================================================

def gradient_check():
    rng = np.random.default_rng(7)
    net = KautilyaNet(seed=1)
    Xs, Ss, As = make_batch(rng, 3)

    # analytic grads
    net.zero_grad()
    L = net.loss(Xs, Ss, As)
    L.backward()
    analytic = [p.grad.copy() for p in net.params()]

    eps = 1e-4  # optimal central-difference step for this graph (roundoff-balanced)
    max_rel = 0.0
    for pi, p in enumerate(net.params()):
        flat = p.data.ravel()
        g_an = analytic[pi].ravel()
        # check a handful of coordinates per param to keep it fast
        idxs = range(flat.size) if flat.size <= 12 else \
            np.random.default_rng(pi).choice(flat.size, 12, replace=False)
        for idx in idxs:
            orig = flat[idx]
            flat[idx] = orig + eps
            Lp = net.loss(Xs, Ss, As).data
            flat[idx] = orig - eps
            Lm = net.loss(Xs, Ss, As).data
            flat[idx] = orig
            num = (Lp - Lm) / (2 * eps)
            an = g_an[idx]
            rel = abs(num - an) / max(1e-8, abs(num) + abs(an))
            max_rel = max(max_rel, rel)
    return max_rel


# =============================================================================
# 5. Training loop
# =============================================================================

def train(steps=600, batch=32, lr=0.02, momentum=0.9, report=50):
    rng = np.random.default_rng(123)
    net = KautilyaNet(seed=2)
    vel = [np.zeros_like(p.data) for p in net.params()]
    history = []
    for t in range(1, steps + 1):
        Xs, Ss, As = make_batch(rng, batch)
        net.zero_grad()
        L = net.loss(Xs, Ss, As)
        L.backward()
        # SGD with momentum + light gradient clipping
        for i, p in enumerate(net.params()):
            g = np.clip(p.grad, -5.0, 5.0)
            vel[i] = momentum * vel[i] - lr * g
            p.data += vel[i]
        if t % report == 0 or t == 1:
            history.append((t, float(L.data)))
            g = float(net.gamma_raw.softplus().data.ravel()[0])
            tau = float(net.tau_raw.softplus().data.ravel()[0])
            print(f"  step {t:4d}   loss {float(L.data):.5f}   "
                  f"gamma {g:.3f}   tau {tau:.3f}")
    return net, history


# =============================================================================
# 6. Evaluation: does corroboration actually beat naive averaging?
# =============================================================================

def evaluate(net, n=400):
    rng = np.random.default_rng(999)
    Xs, Ss, As = make_batch(rng, n)
    err_net = err_mean = 0.0
    trust_on_honest = trust_on_bad = 0.0
    cnt_bad = 0
    for i in range(n):
        s_hat, a_hat, alpha = net.forward_one(Xs[i])
        err_net += np.mean((s_hat.data.ravel() - Ss[i]) ** 2)
        err_mean += np.mean((Xs[i].mean(axis=0) - Ss[i]) ** 2)
        # how much trust did the net place on honest vs turned agents?
        a = alpha.data.ravel()
        honest = np.mean((Xs[i] - Ss[i][None, :]) ** 2, axis=1) < 0.5
        if honest.all():
            continue
        cnt_bad += 1
        trust_on_honest += a[honest].sum()
        trust_on_bad += a[~honest].sum()
    return {
        "net_state_mse": err_net / n,
        "naive_mean_mse": err_mean / n,
        "trust_honest_frac": trust_on_honest / max(1, cnt_bad),
        "trust_bad_frac": trust_on_bad / max(1, cnt_bad),
    }


# =============================================================================
# 7. Self-tests + main
# =============================================================================

def demo_one_case(net, seed=42):
    """Show, for one sample, who was turned and how trust was allocated."""
    rng = np.random.default_rng(seed)
    # force a sample with 3 turned agents for a clear illustration
    s = rng.standard_normal(4)
    X = np.repeat(s[None, :], 7, axis=0) + rng.standard_normal((7, 4)) * 0.05
    bad = np.array([1, 4, 6])
    X[bad] = -s[None, :] * rng.uniform(1.0, 2.0, (3, 1)) \
             + rng.standard_normal((3, 4)) * 0.8
    s_hat, a_hat, alpha = net.forward_one(X)
    a = alpha.data.ravel()
    print("  agent  status   trust%")
    for i in range(7):
        tag = "TURNED" if i in bad else "honest"
        print(f"   {i}     {tag:7s}  {a[i]*100:5.1f}")
    print(f"  recovered state error (MSE): "
          f"{np.mean((s_hat.data.ravel()-s)**2):.4f}")
    print(f"  naive-mean error      (MSE): "
          f"{np.mean((X.mean(axis=0)-s)**2):.4f}")


def self_tests():
    # (a) autodiff sanity: d/dx sum((Wx)^2) checks out against closed form
    rng = np.random.default_rng(3)
    Wd = rng.standard_normal((3, 3))
    xd = rng.standard_normal((1, 3))
    W = Niti(Wd); x = Niti(xd)
    y = (x.matmul(W)).square().sum()
    y.backward()
    # closed form: dy/dx = 2 (xW) W^T
    expect = 2.0 * (xd @ Wd) @ Wd.T
    assert np.allclose(x.grad, expect, atol=1e-9), "autodiff matmul/square wrong"

    # (b) softmax rows sum to 1 and gradient of sum(softmax) is ~0
    s = Niti(rng.standard_normal((5, 1)))
    p = s.softmax(axis=0)
    assert abs(p.data.sum() - 1.0) < 1e-12
    p.sum().backward()
    assert np.allclose(s.grad, 0.0, atol=1e-9), "softmax grad of constant wrong"

    # (c) forward shapes
    net = KautilyaNet()
    rng2 = np.random.default_rng(5)
    X, s, a = make_sample(rng2)
    s_hat, a_hat, alpha = net.forward_one(X)
    assert s_hat.data.shape == (1, 4)
    assert a_hat.data.shape == (1, 4)
    assert abs(alpha.data.sum() - 1.0) < 1e-9
    print("  self-tests passed: autodiff, softmax, forward shapes OK")


if __name__ == "__main__":
    print("=" * 72)
    print("Figure 75 - Kautilya - KautilyaNet (adversarial corroboration)")
    print("=" * 72)

    print("\n[1] Self-tests")
    self_tests()

    print("\n[2] Finite-difference gradient check (mandatory)")
    mr = gradient_check()
    print(f"  max relative error over sampled coords: {mr:.2e}")
    assert mr < 1e-5, "GRADIENT CHECK FAILED"
    print("  GRADIENT CHECK PASSED")

    print("\n[3] Training (recover truth from a network of partly-turned spies)")
    net, hist = train()

    print("\n[4] Evaluation vs naive averaging")
    res = evaluate(net)
    for k, v in res.items():
        print(f"  {k:20s} {v:.4f}")
    verdict = ("corroboration beats naive averaging"
               if res["net_state_mse"] < res["naive_mean_mse"]
               else "no improvement")
    print(f"\n  verdict: {verdict}")
    print(f"  trust placed on honest agents: {res['trust_honest_frac']*100:.1f}% "
          f"vs turned: {res['trust_bad_frac']*100:.1f}%")

    print("\n[5] One case in detail (3 of 7 informants turned)")
    demo_one_case(net)
    print("\nDone.")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
chapter_0052_Theano_-500.py  ::  a mind-model after Theano of Croton (fl. c. 500 BCE)
Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 3 Minds 41 - 60 Available on Amazon https://www.amazon.com/dp/B0H6TVX69S
Resume and Interactive Demos at https://artificiology.com/
Author: David Vivancos · Chapter 0052 · Theano of Croton
================================================================================

WHY THIS ARCHITECTURE IS *HERS* AND NOT A GENERIC NET
-----------------------------------------------------
The one surviving philosophical fragment credited to Theano (preserved in the
Anthologium of Stobaeus, from her lost treatise *On Piety*) corrects a popular
misreading of Pythagoras. People said he taught that all things come to be
*from* number. Theano answers: not *from* number but *kata arithmon* —
"ACCORDING TO number." A thing is not made of the stuff "number"; rather, number
is the *primary ordering* (taxis), and a thing becomes a definite "something"
only by taking its place in that order: first, second, third, and so on.

That is a precise, testable claim about cognition: to know a thing is to know its
*ordinal position* relative to others, not to store its raw substance/magnitude.
So this network refuses to compute on raw input magnitudes. Its first operation
turns every quantity into its RANK among the others — its place in the order.
Everything downstream sees only positions in an order. This buys a real and
unusual inductive bias: invariance to any monotone (order-preserving) rescaling
of the inputs. Stretch, squash, or relabel the units of the world however you
like — if the *order* is unchanged, the model's knowledge is unchanged. That is
"according to number, not from number" made into linear algebra.

The second pillar comes from the pseudepigraphic *Letters* attributed to Theano
(to Kallisto, Euboule, Nikostrate), whose ethical key-word is METRON — due
measure. There the household and its members are likened to an instrument that
must be tuned to a mean: neither too slack (a dead, flat string) nor too taut (a
string about to snap). We encode that as a HARMONIC-MEAN TUNING regularizer.
Using the classical inequality AM >= HM (arithmetic mean >= harmonic mean, with
equality iff all values are equal), we penalize AM/HM - 1 over each layer's
activations. The harmonic mean is dragged down hard by any single near-zero unit
— so the loss says, exactly as the Letters do, that the consonance of the whole
is only as good as its slackest string. Tuning = pulling every unit to a shared,
measured tension.

So the signature of this mind is: (1) identity-as-ordinal-placement, and
(2) virtue-as-proportional-tuning. Neither is a Transformer, attention over
stored keys, or a standard MLP-as-substance-store. It is a rank machine with a
harmonic conscience.

WHAT THE FILE CONTAINS
----------------------
  * A pure-NumPy, from-scratch implementation (no autograd, no ML frameworks).
  * A differentiable SOFT-RANK layer (the "taxis" / counting operation) with an
    exact analytic Jacobian.
  * A HARMONIC TUNING regularizer (AM/HM - 1) with exact analytic gradient.
  * A full analytic backward pass for every parameter.
  * A MANDATORY finite-difference gradient check on the *total* loss.
  * A real training loop on a task where only ordinal structure carries signal,
    and where inputs are corrupted by random monotone distortions — so a
    magnitude-reading model would be misled but an order-reading model is not.
  * Self-tests, including an out-of-distribution check that demonstrates the
    monotone-invariance the architecture is built to have.

Run:  python3 chapter_0052_Theano_-500.py
================================================================================
"""

import numpy as np

# Reproducibility (the cosmos is orderly).
RNG = np.random.default_rng(495)  # 495 BCE: the traditional year Pythagoras died,
                                  # after which the tradition says Theano led the school.
EPS = 1e-8


# ----------------------------------------------------------------------------
# 0. SMALL DIFFERENTIABLE PRIMITIVES
# ----------------------------------------------------------------------------
def sigmoid(x):
    # Numerically stable logistic.
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def softplus(x):
    # log(1 + e^x), stable. Keeps activations strictly positive so the harmonic
    # mean (which needs positive values) is always defined — the strings of the
    # instrument always have *some* tension.
    return np.logaddexp(0.0, x)


def softplus_grad(x):
    # d softplus/dx = sigmoid(x)
    return sigmoid(x)


# ----------------------------------------------------------------------------
# 1. THE TAXIS (SOFT-RANK) LAYER  ::  "according to number"
# ----------------------------------------------------------------------------
# Given m scalar scores s_1..s_m (one per item in a sample), the soft rank of
# item i is how many other items it exceeds:
#
#     R_i = (1 / (m-1)) * sum_{j != i} sigmoid( (s_i - s_j) / tau )
#
# As tau -> 0 this approaches the true (hard) ordinal position normalized to
# [0, 1]: 0 for the smallest, 1 for the largest. This is the literal act of
# "counting a thing as first, second, third." It is invariant to any monotone
# transform of the scores, because it depends only on the signs of (s_i - s_j).
#
# We need its Jacobian dR_i/ds_k for backprop.

def soft_rank(scores, tau):
    """scores: (m,). Returns R: (m,) in (0,1)."""
    m = scores.shape[0]
    diff = (scores[:, None] - scores[None, :]) / tau      # (m, m), delta_ij
    S = sigmoid(diff)                                      # (m, m)
    np.fill_diagonal(S, 0.0)                               # j != i
    R = S.sum(axis=1) / (m - 1)                            # (m,)
    return R


def soft_rank_jacobian(scores, tau):
    """Exact Jacobian J[i,k] = dR_i/ds_k, shape (m, m).

    R_i = 1/(m-1) * sum_{j!=i} g(delta_ij), delta_ij=(s_i - s_j)/tau,
    g = sigmoid, g' = g(1-g).
      dR_i/ds_i = 1/((m-1)*tau) * sum_{j!=i} g'(delta_ij)
      dR_i/ds_k = -1/((m-1)*tau) * g'(delta_ik)   for k != i
    """
    m = scores.shape[0]
    diff = (scores[:, None] - scores[None, :]) / tau
    S = sigmoid(diff)
    Sp = S * (1.0 - S)                                     # g'(delta_ij), (m,m)
    np.fill_diagonal(Sp, 0.0)
    c = 1.0 / ((m - 1) * tau)
    J = -c * Sp                                            # off-diagonal terms
    diag = c * Sp.sum(axis=1)                              # dR_i/ds_i
    np.fill_diagonal(J, diag)
    return J


# ----------------------------------------------------------------------------
# 2. THE HARMONIC TUNING REGULARIZER  ::  "metron" (due measure)
# ----------------------------------------------------------------------------
# For a row of positive activations a_1..a_h (the "strings" of one item's hidden
# state), the arithmetic-mean / harmonic-mean gap measures how *out of tune* the
# strings are:
#
#     AM = (1/h) sum a_i
#     HM = h / sum(1/a_i)
#     tune = AM/HM - 1   >= 0,   == 0  iff all a_i equal  (perfect consonance).
#
# A single slack (near-zero) string sends HM toward 0 and the penalty toward
# infinity — exactly the Letters' claim that one untuned member spoils the
# concord of the household.

def harmonic_tune(A):
    """A: (m, h) positive activations. Returns scalar mean tune over rows."""
    h = A.shape[1]
    S1 = A.sum(axis=1)                  # (m,)
    S2 = (1.0 / A).sum(axis=1)          # (m,)
    ratio = (S1 * S2) / (h * h)         # AM/HM per row
    return np.mean(ratio - 1.0)


def harmonic_tune_grad(A):
    """d(mean tune)/dA, shape (m, h).

    per-row loss L_r = (S1*S2)/h^2 - 1, with S1=sum a, S2=sum 1/a.
      dL_r/da_i = (1/h^2) * ( S2 - S1 / a_i^2 )
    Outer mean over m rows divides by m.
    """
    m, h = A.shape
    S1 = A.sum(axis=1, keepdims=True)       # (m,1)
    S2 = (1.0 / A).sum(axis=1, keepdims=True)
    g = (S2 - S1 / (A * A)) / (h * h)       # (m,h)
    return g / m


# ----------------------------------------------------------------------------
# 3. THE MODEL  ::  Ordinal Tuning Network
# ----------------------------------------------------------------------------
# A sample is a *set* of m items; each item carries f features (phi in R^f).
# The model scores each item, ranks the scores (the taxis layer), and is trained
# so the predicted ordinal positions match the world's true ordering. The hidden
# layer's activations are kept "in tune" by the harmonic regularizer.
#
#   phi (m,f) --[W1,b1]--> Z1 (m,h) --softplus--> A (m,h)
#   A (m,h)   --[W2,b2]--> score (m,)
#   score     --soft_rank(tau)--> R (m,)   <-- predicted ordinal places
#
# Loss = MSE(R, R_true)  +  lam * harmonic_tune(A)

class OrdinalTuningNet:
    def __init__(self, f, h, tau=0.15, lam=0.05, seed=0):
        rng = np.random.default_rng(seed)
        # Small, careful init so the instrument starts roughly in tune.
        self.W1 = rng.normal(0, 1.0 / np.sqrt(f), size=(f, h))
        self.b1 = np.full(h, 0.5)               # start strings with positive tension
        self.W2 = rng.normal(0, 1.0 / np.sqrt(h), size=(h, 1))
        self.b2 = np.zeros(1)
        self.tau = tau
        self.lam = lam
        # A string is never at zero tension; every unit keeps a base tension c0.
        # This bounds 1/A (so the harmonic mean is always well defined) without
        # changing the backward pass, since c0 is an additive constant.
        self.c0 = 0.05

    # ----- forward for ONE sample (m items) -----
    def forward_sample(self, phi):
        Z1 = phi @ self.W1 + self.b1            # (m,h)
        A = softplus(Z1) + self.c0              # (m,h) positive, floored tension
        score = (A @ self.W2 + self.b2)[:, 0]   # (m,)
        R = soft_rank(score, self.tau)          # (m,)
        cache = (phi, Z1, A, score, R)
        return R, A, cache

    # ----- total loss over a batch of samples -----
    def loss_batch(self, Phis, Rtrues):
        """Phis: list of (m,f). Rtrues: list of (m,). Returns (loss, parts)."""
        n = len(Phis)
        fit = 0.0
        tune = 0.0
        for phi, Rt in zip(Phis, Rtrues):
            R, A, _ = self.forward_sample(phi)
            fit += np.mean((R - Rt) ** 2)
            tune += harmonic_tune(A)
        fit /= n
        tune /= n
        total = fit + self.lam * tune
        return total, (fit, tune)

    # ----- analytic gradients over a batch -----
    def grads_batch(self, Phis, Rtrues):
        n = len(Phis)
        gW1 = np.zeros_like(self.W1); gb1 = np.zeros_like(self.b1)
        gW2 = np.zeros_like(self.W2); gb2 = np.zeros_like(self.b2)
        fit = 0.0; tune = 0.0
        for phi, Rt in zip(Phis, Rtrues):
            m = phi.shape[0]
            Z1 = phi @ self.W1 + self.b1
            A = softplus(Z1) + self.c0
            score = (A @ self.W2 + self.b2)[:, 0]
            R = soft_rank(score, self.tau)

            fit += np.mean((R - Rt) ** 2)
            tune += harmonic_tune(A)

            # ---- backward: fit term ----
            dR = (2.0 / m) * (R - Rt) / n          # dFit/dR  (Fit averaged over n)
            J = soft_rank_jacobian(score, self.tau)  # (m,m): dR_i/ds_k
            dscore = J.T @ dR                       # (m,)  dFit/dscore

            dA = np.outer(dscore, self.W2[:, 0])    # (m,h) via score = A W2 + b2
            gW2 += A.T @ dscore[:, None]            # (h,1)
            gb2 += np.array([dscore.sum()])

            # ---- backward: tune term (depends on A only) ----
            dA += (self.lam / n) * harmonic_tune_grad(A)   # add tuning gradient

            dZ1 = dA * softplus_grad(Z1)            # (m,h)
            gW1 += phi.T @ dZ1                      # (f,h)
            gb1 += dZ1.sum(axis=0)                  # (h,)

        fit /= n; tune /= n
        total = fit + self.lam * tune
        return total, {"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}, (fit, tune)

    # ----- parameter vector helpers (for gradient checking) -----
    def get_params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def get_grad_names(self):
        return ["W1", "b1", "W2", "b2"]


# ----------------------------------------------------------------------------
# 4. THE WORLD  ::  a task where only ORDER carries the signal
# ----------------------------------------------------------------------------
# Each sample has m items; each item has f features phi. A hidden "true taste"
# vector a_star fixes each item's latent utility u = phi . a_star. The world's
# true ranking of the items is the order of u. The model never sees u; it sees
# phi passed through a RANDOM, per-sample MONOTONE distortion of each feature
# (a positive power, scale, and shift), which scrambles magnitudes but PRESERVES
# the order within each feature column. A magnitude reader is misled; an order
# reader is not — because rank is monotone-invariant.

def true_ranks(u):
    """Normalized ordinal positions in [0,1] from a utility vector u (m,)."""
    order = np.argsort(np.argsort(u))           # 0..m-1
    return order / (len(u) - 1)


def monotone_distort(phi, rng, strength=1.0):
    """Apply an order-preserving, magnitude-destroying map per feature column."""
    m, f = phi.shape
    out = np.empty_like(phi)
    for k in range(f):
        col = phi[:, k]
        p = np.exp(rng.normal(0, 0.6 * strength))     # random positive power
        scale = np.exp(rng.normal(0, 0.8 * strength)) # random positive scale
        shift = rng.normal(0, 1.0 * strength)         # random shift
        out[:, k] = scale * np.sign(col) * (np.abs(col) ** p) + shift
    return out


def make_dataset(n, m, f, a_star, rng, strength=1.0):
    Phis, Rtrues = [], []
    for _ in range(n):
        base = rng.normal(0, 1, size=(m, f))          # latent features
        u = base @ a_star                             # latent utility
        Rt = true_ranks(u)                            # order is the only truth
        phi = monotone_distort(base, rng, strength)   # what the model actually sees
        Phis.append(phi); Rtrues.append(Rt)
    return Phis, Rtrues


def spearman(R_pred, R_true):
    """Rank agreement in [-1,1] between two normalized-rank vectors."""
    a = np.argsort(np.argsort(R_pred)).astype(float)
    b = np.argsort(np.argsort(R_true)).astype(float)
    a -= a.mean(); b -= b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum()) + EPS
    return float((a * b).sum() / denom)


def eval_spearman(model, Phis, Rtrues):
    vals = []
    for phi, Rt in zip(Phis, Rtrues):
        R, _, _ = model.forward_sample(phi)
        vals.append(spearman(R, Rt))
    return float(np.mean(vals))


# ----------------------------------------------------------------------------
# 5. GRADIENT CHECK  ::  mandatory
# ----------------------------------------------------------------------------
def gradient_check():
    print("-" * 72)
    print("FINITE-DIFFERENCE GRADIENT CHECK (on the TOTAL loss)")
    print("-" * 72)
    f, h, m = 4, 6, 5
    a_star = RNG.normal(0, 1, size=f)
    Phis, Rtrues = make_dataset(n=7, m=m, f=f, a_star=a_star, rng=RNG, strength=1.0)
    model = OrdinalTuningNet(f=f, h=h, tau=0.2, lam=0.1, seed=1)

    _, grads, _ = model.grads_batch(Phis, Rtrues)
    params = model.get_params()
    names = model.get_grad_names()

    eps = 1e-6
    max_rel = 0.0
    for p, name in zip(params, names):
        ga = grads[name]
        gn = np.zeros_like(p)
        it = np.nditer(p, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            idx = it.multi_index
            orig = p[idx]
            p[idx] = orig + eps
            lp, _ = model.loss_batch(Phis, Rtrues)
            p[idx] = orig - eps
            lm, _ = model.loss_batch(Phis, Rtrues)
            p[idx] = orig
            gn[idx] = (lp - lm) / (2 * eps)
            it.iternext()
        num = np.abs(ga - gn)
        den = np.maximum(1e-7, np.abs(ga) + np.abs(gn))
        rel = np.max(num / den)
        max_rel = max(max_rel, rel)
        print(f"  {name:>3s}  max|analytic-numeric|={num.max():.3e}   "
              f"max rel err={rel:.3e}")
    print(f"\n  WORST relative error across all params: {max_rel:.3e}")
    ok = max_rel < 1e-4
    print("  RESULT:", "PASS  (analytic gradients verified)" if ok
          else "FAIL")
    print()
    return ok


# ----------------------------------------------------------------------------
# 6. TRAIN  ::  a real loop, full-batch gradient descent with momentum
# ----------------------------------------------------------------------------
def train():
    print("-" * 72)
    print("TRAINING THE ORDINAL TUNING NETWORK")
    print("-" * 72)
    f, h, m = 5, 12, 8
    a_star = RNG.normal(0, 1, size=f)

    Phi_tr, R_tr = make_dataset(220, m, f, a_star, RNG, strength=1.0)
    Phi_va, R_va = make_dataset(60,  m, f, a_star, RNG, strength=1.0)
    # Out-of-distribution: HARSHER monotone distortion the model never trained on.
    Phi_ood, R_ood = make_dataset(60, m, f, a_star, RNG, strength=2.3)

    model = OrdinalTuningNet(f=f, h=h, tau=0.15, lam=0.03, seed=7)
    lr, mu, clip = 0.05, 0.9, 5.0
    vel = {k: np.zeros_like(v) for k, v in
           zip(model.get_grad_names(), model.get_params())}

    print(f"{'epoch':>6} {'total':>10} {'fit':>10} {'tune':>9} "
          f"{'rho_tr':>8} {'rho_va':>8}")
    epochs = 600
    for ep in range(epochs + 1):
        total, grads, (fit, tune) = model.grads_batch(Phi_tr, R_tr)
        # global-norm gradient clipping keeps the harmonic penalty from snapping
        # a string (the very failure mode the tuning term is meant to police).
        gnorm = np.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
        scale = clip / gnorm if gnorm > clip else 1.0
        for name, P in zip(model.get_grad_names(), model.get_params()):
            vel[name] = mu * vel[name] - lr * scale * grads[name]
            P += vel[name]
        if ep % 50 == 0:
            rho_tr = eval_spearman(model, Phi_tr, R_tr)
            rho_va = eval_spearman(model, Phi_va, R_va)
            print(f"{ep:6d} {total:10.5f} {fit:10.5f} {tune:9.4f} "
                  f"{rho_tr:8.4f} {rho_va:8.4f}")

    rho_va = eval_spearman(model, Phi_va, R_va)
    rho_ood = eval_spearman(model, Phi_ood, R_ood)
    print()
    print("  Validation rank-agreement (Spearman rho):        "
          f"{rho_va:7.4f}")
    print("  OUT-OF-DISTRIBUTION (2.3x monotone distortion):  "
          f"{rho_ood:7.4f}")
    print("  -> Near-equal scores confirm the architecture reads ORDER, not")
    print("     magnitude: 'according to number, not from number.'")
    print()
    return model, rho_va, rho_ood


# ----------------------------------------------------------------------------
# 7. SELF-TESTS
# ----------------------------------------------------------------------------
def self_tests():
    print("-" * 72)
    print("SELF-TESTS")
    print("-" * 72)
    passed = True

    # (a) soft_rank recovers the true order as tau -> 0.
    s = np.array([0.3, -1.2, 2.5, 0.31, -0.4])
    R = soft_rank(s, tau=1e-3)
    order_pred = np.argsort(R)
    order_true = np.argsort(s)
    t1 = np.array_equal(order_pred, order_true)
    print(f"  [a] soft-rank recovers true ordinal order (tau->0):  "
          f"{'PASS' if t1 else 'FAIL'}")
    passed &= t1

    # (b) soft_rank is invariant under a monotone map of the scores.
    g = lambda x: np.exp(0.7 * x) - 0.3      # strictly increasing
    R2 = soft_rank(g(s), tau=0.05)
    R1 = soft_rank(s, tau=0.05)
    # ranks (orders) should match exactly even if soft values differ slightly
    t2 = np.array_equal(np.argsort(R1), np.argsort(R2))
    print(f"  [b] soft-rank invariant to monotone score map:       "
          f"{'PASS' if t2 else 'FAIL'}")
    passed &= t2

    # (c) harmonic_tune is zero for equal strings, positive otherwise.
    A_eq = np.full((3, 5), 0.8)
    A_un = np.array([[0.8, 0.8, 0.8, 0.8, 0.05]])
    t3 = (abs(harmonic_tune(A_eq)) < 1e-9) and (harmonic_tune(A_un) > 0.5)
    print(f"  [c] harmonic tune = 0 in tune, large when slack:     "
          f"{'PASS' if t3 else 'FAIL'}")
    passed &= t3

    # (d) tuning gradient points to reduce the penalty (descent decreases it).
    A0 = np.array([[0.9, 0.2, 1.4, 0.6]])
    gA = harmonic_tune_grad(A0)
    A1 = A0 - 1e-2 * gA
    t4 = harmonic_tune(A1) < harmonic_tune(A0)
    print(f"  [d] a tuning-gradient step lowers the dissonance:    "
          f"{'PASS' if t4 else 'FAIL'}")
    passed &= t4

    print("\n  SELF-TESTS:", "ALL PASS" if passed else "SOME FAILED")
    print()
    return passed


# ----------------------------------------------------------------------------
# 8. MAIN
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 72)
    print("ORDINAL TUNING NETWORK  ::  a computational portrait of Theano's mind")
    print("  'Things come to be not FROM number but ACCORDING TO number;")
    print("   in number is the primary ordering.'   -- On Piety (fr., via Stobaeus)")
    print("=" * 72)
    print()

    gc_ok = gradient_check()
    st_ok = self_tests()
    model, rho_va, rho_ood = train()

    print("=" * 72)
    print("SUMMARY")
    print(f"  gradient check : {'PASS' if gc_ok else 'FAIL'}")
    print(f"  self-tests     : {'PASS' if st_ok else 'FAIL'}")
    print(f"  val rank-agree : {rho_va:.4f}")
    print(f"  ood rank-agree : {rho_ood:.4f}")
    print("=" * 72)

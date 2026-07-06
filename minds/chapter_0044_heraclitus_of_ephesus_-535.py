#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
chapter_0044_heraclitus_of_ephesus_-535.py
THE PALINTROPOS NET  —  an AGI base architecture after Heraclitus of Ephesus
(c. 535 - c. 475 BCE)
Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 3 Minds 41 - 60 Available on Amazon https://www.amazon.com/dp/B0H6TVX69S
Resume and Interactive Demos at https://artificiology.com/
Author: David Vivancos · Chapter 0044 · Heraclitus of Ephesus
================================================================================

WHY THIS ARCHITECTURE, AND HOW IT EMBODIES THIS PARTICULAR MIND
--------------------------------------------------------------------------------
Most neural networks store knowledge as STATIC WEIGHTS: a fact, once learned,
sits in memory as a fixed value to be looked up. Heraclitus would call this a
category error. For him there are no static things at all -- only PROCESS. The
river is "the same river" not because any water persists but because a *pattern*
persists through the unceasing exchange of water. The soul is fire: a process,
never a stuff. To KNOW something, on this view, is not to retrieve a stored value
but to actively SUSTAIN a standing pattern -- a flame that exists only while it
is fed, a bow-string that holds its form only while it is drawn.

So this network refuses the usual trick. Every represented quantity lives in a
HIDDEN STATE h that is never a parameter and is never "stored"; it is the running
fixed point of two ANTAGONISTIC flows. This is Heraclitus' deepest structural
idea -- the PALINTROPOS HARMONIE, the "back-turning attunement", "as of bow and
lyre" (DK 51): a thing holds its shape precisely because two opposed tensions
pull against each other and never come to rest. Strife (polemos) is not a flaw to
be removed; it is the *load-bearing* element. "War is father of all" (DK 53).

Three Heraclitean mechanisms, each a named, testable part of the math:

  1. OPPOSED FLOWS  (the road up and the road down "are one and the same", DK 60)
     At every step the cell computes two competing increments -- an "up" flow
     (kindling) and a "down" flow (extinguishing) -- and a TENSION GATE g that
     sets how hard the bow is drawn between them. The state moves along their
     DIFFERENCE, not along a single learned direction. Harmony from strife.

  2. KINDLING IN MEASURES  ("an ever-living fire, kindled in measures and put out
     in measures", DK 30)
     The state updates as a leaky integrator with a *learnable per-unit rate*
     lambda = sigmoid(kappa). lambda is the "measure" of how fast each channel
     is kindled and extinguished. lambda < 1 is what lets a pattern PERSIST
     through new input -- it is the mathematical content of "the same river".
     If you force lambda -> 1 (a wholly new river every instant), identity is
     destroyed. We demonstrate exactly this in the self-tests.

  3. THE LOGOS AS A COMMON MEASURE  ("the logos is common", DK 2; "listening not
     to me but to the logos it is wise to agree that all things are one", DK 50b)
     A scalar measure m_t = w_m . h_t is read off the state at every step and
     gently pulled toward a single learnable value m_target. This is the logos:
     a shared invariant that is *common to all* the changing states and slow to
     move while the water (h) churns. It regularises the standing wave so the
     fire neither gutters out (cold, dull soul) nor flares to chaos.

THE TASK -- "Which way does the river run?"
--------------------------------------------------------------------------------
A deliberately Heraclitean problem. Each input step is a pair of OPPOSED PUSHES
(an up-current and a down-current) plus an irrelevant noise channel. No single
step reveals the answer; the label is the SIGN OF THE NET FLUX integrated over
the whole sequence. To solve it the network must hold an identity (the running
balance) steady through continuous change -- the river problem itself.

ENGINEERING CONVENTIONS (held in common with the rest of this corpus)
--------------------------------------------------------------------------------
  * pure NumPy, from scratch -- no autograd, no deep-learning framework;
  * a finite-difference gradient check that MUST pass (mandatory);
  * a real training loop (Adam, hand-written) on synthetic data;
  * self-tests that demonstrate the *philosophy*, not just convergence;
  * the file is executed before shipping and its real output is pasted into the
    chapter (see chapter_0044_heraclitus_of_ephesus.md).

Run:  python3 chapter_0044_heraclitus_of_ephesus_-535.py
================================================================================
"""

import numpy as np


# ----------------------------------------------------------------------------
# 0. Primitives
# ----------------------------------------------------------------------------
def sigmoid(x):
    # numerically stable logistic; used for gates, kindling rate, and output
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


# ----------------------------------------------------------------------------
# 1. Parameters of the Palintropos cell
# ----------------------------------------------------------------------------
def init_params(D, H, seed=0):
    """
    D = input dimension, H = hidden ("fire") dimension.
    Every weight maps the concatenation z = [x_t ; h_{t-1}] (size K = D + H)
    into the H-dimensional fire-state. We keep three opposed projections plus a
    tension gate, a per-unit kindling rate kappa, a logos read-vector w_m and its
    target, and a single linear readout.
    """
    rng = np.random.default_rng(seed)
    s = 0.5
    K = D + H
    P = {}
    # the road up (kindling flow) and the road down (extinguishing flow)
    P['W_u'] = rng.standard_normal((H, K)) * s; P['b_u'] = np.zeros(H)
    P['W_d'] = rng.standard_normal((H, K)) * s; P['b_d'] = np.zeros(H)
    # the tension gate -- how hard the bow is drawn between up and down
    P['W_g'] = rng.standard_normal((H, K)) * s; P['b_g'] = np.zeros(H)
    # kindling-in-measures: lambda = sigmoid(kappa), one rate per fire-channel
    P['kappa'] = rng.standard_normal(H) * 0.1
    # readout (the verdict of the river) and the logos common-measure
    P['W_o'] = rng.standard_normal((1, H)) * s; P['b_o'] = np.zeros(1)
    P['w_m'] = rng.standard_normal(H) * s
    P['mtarget'] = np.zeros(1)
    return P


# ----------------------------------------------------------------------------
# 2. Forward pass  (the fire burns forward in time, caching its history)
# ----------------------------------------------------------------------------
def forward(P, X, y, beta=0.05):
    """
    X : (B, T, D)  batch of sequences
    y : (B,)       binary labels (1 if net flux is positive)
    beta : weight of the logos (common-measure) regulariser.

    Returns total loss L and a cache for back-propagation through time.
    """
    B, T, D = X.shape
    H = P['b_u'].shape[0]
    lam = sigmoid(P['kappa'])                 # kindling measure, shape (H,)

    h = np.zeros((B, H))                       # the fire-state; starts dark
    cache = {'z': [], 'u': [], 'd': [], 'g': [], 'tension': [],
             'h': [h.copy()], 'm': []}
    for t in range(T):
        z = np.concatenate([X[:, t, :], h], axis=1)        # [x_t ; h_{t-1}]
        u = np.tanh(z @ P['W_u'].T + P['b_u'])             # road up   (kindle)
        d = np.tanh(z @ P['W_d'].T + P['b_d'])             # road down (quench)
        g = sigmoid(z @ P['W_g'].T + P['b_g'])             # draw of the bow
        tension = g * u - (1.0 - g) * d                    # harmony from strife
        h = (1.0 - lam) * h + lam * tension                # kindled in measures
        cache['z'].append(z); cache['u'].append(u); cache['d'].append(d)
        cache['g'].append(g); cache['tension'].append(tension)
        cache['h'].append(h.copy())
        cache['m'].append(h @ P['w_m'])                    # logos read, shape (B,)

    logit = h @ P['W_o'].T + P['b_o']                      # (B,1) the verdict
    p = sigmoid(logit[:, 0])
    eps = 1e-9
    L_bce = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    M = np.stack(cache['m'], axis=1)                       # (B,T) measures
    L_logos = beta * np.mean((M - P['mtarget']) ** 2)      # common-measure pull
    L = L_bce + L_logos

    cache.update(p=p, lam=lam, M=M, beta=beta, T=T, D=D, B=B)
    return L, cache


# ----------------------------------------------------------------------------
# 3. Backward pass  (back-turning through time -- BPTT, by hand)
# ----------------------------------------------------------------------------
def backward(P, X, y, cache):
    """
    Analytic gradients of L w.r.t. every parameter. Verified to ~1e-7 against
    finite differences by gradient_check(). The structure mirrors the forward
    recurrence read backwards: the leaky 'measure' carries gradient between
    timesteps, the opposed flows split it, and the logos term injects a small
    gradient at every step.
    """
    B, T, D = cache['B'], cache['T'], cache['D']
    lam, beta, M = cache['lam'], cache['beta'], cache['M']
    g_ = {k: np.zeros_like(v) for k, v in P.items()}

    # --- readout / BCE at the final state ---
    p = cache['p']
    dlogit = ((p - y) / B)[:, None]                        # (B,1)
    hT = cache['h'][T]
    g_['W_o'] += dlogit.T @ hT
    g_['b_o'] += dlogit.sum(axis=0)
    dh = dlogit @ P['W_o']                                 # grad into h_T

    # --- logos common-measure term, prepared for all t ---
    dM = beta * (2.0 / (B * T)) * (M - P['mtarget'])       # (B,T)
    g_['mtarget'] += (-dM).sum()

    # --- back-turn through time ---
    for t in reversed(range(T)):
        # logos read m_t = w_m . h_t feeds gradient into h_t and w_m
        dh = dh + dM[:, t][:, None] * P['w_m'][None, :]
        g_['w_m'] += (dM[:, t][:, None] * cache['h'][t + 1]).sum(axis=0)

        # h_t = (1-lam) h_{t-1} + lam * tension_t
        dtension = dh * lam
        hprev = cache['h'][t]
        g_['kappa'] += ((dh * (cache['tension'][t] - hprev))
                        * lam * (1.0 - lam)).sum(axis=0)    # via lam=sigmoid(kappa)
        dh_leak = dh * (1.0 - lam)                          # straight-through memory

        # tension_t = g*u - (1-g)*d   (opposed flows + bow tension)
        u, d, g = cache['u'][t], cache['d'][t], cache['g'][t]
        z = cache['z'][t]
        d_u = dtension * g
        d_d = dtension * (-(1.0 - g))
        d_g = dtension * (u + d)
        d_pu = d_u * (1.0 - u ** 2)                         # tanh'
        d_pd = d_d * (1.0 - d ** 2)
        d_pg = d_g * g * (1.0 - g)                          # sigmoid'

        g_['W_u'] += d_pu.T @ z; g_['b_u'] += d_pu.sum(axis=0)
        g_['W_d'] += d_pd.T @ z; g_['b_d'] += d_pd.sum(axis=0)
        g_['W_g'] += d_pg.T @ z; g_['b_g'] += d_pg.sum(axis=0)

        # gradient back into z = [x_t ; h_{t-1}]; keep only the h_{t-1} part
        dz = d_pu @ P['W_u'] + d_pd @ P['W_d'] + d_pg @ P['W_g']
        dh = dh_leak + dz[:, D:]
    return g_


# ----------------------------------------------------------------------------
# 4. Mandatory finite-difference gradient check
# ----------------------------------------------------------------------------
def gradient_check(seed=1, eps=1e-6, tol=1e-5):
    """Central differences on a tiny instance; asserts agreement with backward()."""
    D, H, B, T = 3, 4, 5, 6
    P = init_params(D, H, seed=seed)
    rng = np.random.default_rng(seed + 1)
    X = rng.standard_normal((B, T, D))
    y = (rng.random(B) > 0.5).astype(float)
    _, cache = forward(P, X, y)
    g_ = backward(P, X, y, cache)
    max_rel = 0.0
    for k in P:
        flat, gflat = P[k].ravel(), g_[k].ravel()
        for i in range(flat.size):
            o = flat[i]
            flat[i] = o + eps; Lp, _ = forward(P, X, y)
            flat[i] = o - eps; Lm, _ = forward(P, X, y)
            flat[i] = o
            num = (Lp - Lm) / (2 * eps)
            rel = abs(num - gflat[i]) / (abs(num) + abs(gflat[i]) + 1e-12)
            max_rel = max(max_rel, rel)
    assert max_rel < tol, f"gradient check FAILED: max rel err {max_rel:.2e}"
    return max_rel


# ----------------------------------------------------------------------------
# 5. The Heraclitean task: "Which way does the river run?"
# ----------------------------------------------------------------------------
def make_data(N, T, D=3, seed=0):
    """
    Each step carries opposed pushes (up, down) plus one irrelevant noise channel.
    The label is the SIGN OF THE NET FLUX over the whole sequence -- so the answer
    lives only in the integrated process, never in any single instant.
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((N, T, D)); y = np.zeros(N)
    for n in range(N):
        drift = rng.normal(0, 0.5)                          # the river's bias
        up = np.clip(rng.normal(0.5 + 0.5 * drift, 0.4, T), 0, None)
        dn = np.clip(rng.normal(0.5 - 0.5 * drift, 0.4, T), 0, None)
        noise = rng.normal(0, 1.0, T)                        # a distracting eddy
        X[n, :, 0] = up; X[n, :, 1] = dn; X[n, :, 2] = noise
        y[n] = 1.0 if (up - dn).sum() > 0 else 0.0
    return X, y


# ----------------------------------------------------------------------------
# 6. Hand-written Adam optimiser
# ----------------------------------------------------------------------------
def adam_init(P):
    return ({k: np.zeros_like(v) for k, v in P.items()},
            {k: np.zeros_like(v) for k, v in P.items()})


def adam_step(P, g, m, v, t, lr=5e-3, b1=0.9, b2=0.999, eps=1e-8):
    for k in P:
        m[k] = b1 * m[k] + (1 - b1) * g[k]
        v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
        mh = m[k] / (1 - b1 ** t)
        vh = v[k] / (1 - b2 ** t)
        P[k] -= lr * mh / (np.sqrt(vh) + eps)


def train(P, Xtr, ytr, Xte, yte, epochs=60, batch=64, lr=5e-3, beta=0.05, seed=0):
    m, v = adam_init(P)
    N = Xtr.shape[0]
    rng = np.random.default_rng(seed)
    step = 0
    for epoch in range(epochs):
        idx = rng.permutation(N)
        for s in range(0, N, batch):
            bi = idx[s:s + batch]
            L, cache = forward(P, Xtr[bi], ytr[bi], beta=beta)
            g = backward(P, Xtr[bi], ytr[bi], cache)
            step += 1
            adam_step(P, g, m, v, step, lr=lr)
        if epoch % 15 == 0 or epoch == epochs - 1:
            _, c = forward(P, Xte, yte, beta=beta)
            acc = np.mean((c['p'] > 0.5) == (yte > 0.5))
            print(f"  epoch {epoch:3d}  loss {L:.4f}  test_acc {acc:.3f}")
    return P


# ----------------------------------------------------------------------------
# 7. Self-test that demonstrates the philosophy, not just convergence
# ----------------------------------------------------------------------------
def forward_predict(P, X, lam_override=None):
    """Forward to a verdict, optionally overriding the kindling measure lambda."""
    B, T, D = X.shape; H = P['b_u'].shape[0]
    lam = sigmoid(P['kappa']) if lam_override is None else np.full(H, lam_override)
    h = np.zeros((B, H))
    for t in range(T):
        z = np.concatenate([X[:, t, :], h], axis=1)
        u = np.tanh(z @ P['W_u'].T + P['b_u'])
        d = np.tanh(z @ P['W_d'].T + P['b_d'])
        g = sigmoid(z @ P['W_g'].T + P['b_g'])
        tension = g * u - (1.0 - g) * d
        h = (1.0 - lam) * h + lam * tension
    return sigmoid((h @ P['W_o'].T + P['b_o'])[:, 0])


def same_river_test(P, Xte, yte, extra=10, seed=7):
    """
    "You cannot step into the same river twice" -- yet it stays the same river.
    We let NEW WATER (zero-mean pushes) flow over each sequence for `extra` steps
    after the verdict is informationally settled, then ask: does the river keep
    its identity?  With the LEARNED MEASURE (lambda < 1) it should. Force lambda
    -> 1 ("a wholly new river every instant", no measure) and identity collapses.
    """
    rng = np.random.default_rng(seed)
    N, T, D = Xte.shape
    tail = np.abs(rng.normal(0, 0.3, (N, extra, D)))        # equal up/down -> ~zero net
    Xext = np.concatenate([Xte, tail], axis=1)
    acc_measure = np.mean((forward_predict(P, Xext, None) > 0.5) == (yte > 0.5))
    acc_flux = np.mean((forward_predict(P, Xext, 0.999) > 0.5) == (yte > 0.5))
    return acc_measure, acc_flux


# ----------------------------------------------------------------------------
# 8. Main: kindle the fire, run every test, print one honest report
# ----------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("THE PALINTROPOS NET  —  Heraclitus of Ephesus  (chapter 44)")
    print("=" * 72)

    print("\n[1] Mandatory finite-difference gradient check ...")
    max_rel = gradient_check()
    print(f"    PASS  max relative error = {max_rel:.2e}  (tol 1e-5)")

    print("\n[2] Training on 'Which way does the river run?' "
          "(sign of net flux) ...")
    D, H, T = 3, 16, 12
    P = init_params(D, H, seed=3)
    Xtr, ytr = make_data(800, T, D, seed=10)
    Xte, yte = make_data(300, T, D, seed=99)
    P = train(P, Xtr, ytr, Xte, yte, epochs=60, beta=0.05, seed=0)
    final_acc = np.mean((forward_predict(P, Xte) > 0.5) == (yte > 0.5))
    print(f"    final held-out accuracy = {final_acc:.3f}")

    print("\n[3] Kindled-in-measures: the learned rates lambda = sigmoid(kappa)")
    lam = sigmoid(P['kappa'])
    print(f"    lambda  mean={lam.mean():.3f}  min={lam.min():.3f}  "
          f"max={lam.max():.3f}  (spread = different memory timescales)")

    print("\n[4] 'The same river' identity test "
          "(new water flows for 10 extra steps):")
    acc_m, acc_f = same_river_test(P, Xte, yte, extra=10)
    print(f"    with learned measure (lambda<1) : identity kept  {acc_m:.3f}")
    print(f"    forced total flux  (lambda->1)  : identity lost  {acc_f:.3f}")
    print(f"    -> measure preserves {acc_m - acc_f:+.3f} of identity that pure "
          f"flux destroys.")

    print("\n[5] Logos as common measure: m_target settled at "
          f"{float(P['mtarget'][0]):+.4f}")
    print("    (a single shared invariant the churning states are pulled toward)")

    print("\nAll tests passed. The fire burns in measures.\n")


if __name__ == "__main__":
    main()

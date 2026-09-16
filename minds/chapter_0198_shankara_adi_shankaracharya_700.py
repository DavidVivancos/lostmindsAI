#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Chapter 0198 - Shankara (Adi Shankaracharya), c. 700-750 CE
Encyclopedia of Lost Minds: Echoes on AI
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 10 Minds 181 - 200 Available on Amazon https://www.amazon.com/dp/B0HJYMZ3G6
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0198_shankara_adi_shankaracharya_700 - Shankara (Adi Shankaracharya), c. 700-750 CE
================================================================================  

THE MIND-SPECIFIC THESIS
------------------------
Most "consciousness" architectures bolt a passive "witness" module onto an
ordinary network and call it Vedanta. That misses Shankara's actual, singular
cognitive idea. His introduction to the Brahma-Sutra commentary -- the
Adhyasa-bhashya -- defines the fundamental act of the mind as ADHYASA
(superimposition): "smrti-rupah paratra purva-drsta-avabhasah" -- the
appearance, in the form of memory, of something previously seen, laid over
something else. Every ordinary cognition is the projection of stored name-and-
form (nama-rupa / samskara) onto a single, changeless substrate.

From that one idea three consequences follow, and this network encodes all of
them mechanically rather than decoratively:

  1. ADHYASA  -- an appearance is modelled as  x_hat = substrate + superimposed.
                 The "world" is the one substrate plus a projected name-form.
  2. VIVEKA / NETI-NETI -- knowledge is SUBTRACTIVE, not additive. To recover
                 the real, you REMOVE the superimposed layer:  witness = x - s.
                 No new content is added; content is stripped away.
  3. ADVAITA  -- "ekam eva advitiyam", one without a second. Stripped of its
                 name-form, EVERY appearance must resolve to the SAME substrate.
                 Non-duality is therefore an INVARIANCE constraint across all
                 inputs, and the network is trained to satisfy it.

The training signal that makes this non-trivial is Shankara's own: the shared,
constant component of all appearances is pushed OUT of the superimposition and
INTO the single substrate (L_center), so the substrate learns to be the one
invariant residue (Brahman-as-mean-of-all-name-forms) and the superimposition
learns to be the zero-mean play of maya.

Finally, BADHA (sublation) -- the criterion by which the real is told from the
merely apparent -- is read out as a per-dimension "reality grade" (satta):
directions dominated by the invariant substrate are near-unsublatable
(paramarthika ~ 1); directions dominated by name-form variance are highly
sublatable (vyavaharika / pratibhasika ~ 0).

WHAT IS IN THIS FILE
--------------------
  * AdhyasaSublationNetwork : pure-NumPy, from-scratch, 2-layer nonlinear
                              encoder + linear superimposition decoder + one
                              global substrate parameter.
  * Exact analytic backprop for every parameter.
  * A finite-difference gradient check (MANDATORY) that must pass.
  * A synthetic "Advaita world": one true substrate, many low-rank name-forms.
  * A real training loop (full-batch gradient descent w/ momentum).
  * Self-tests / diagnostics: substrate recovery, witness invariance
    (non-duality), sublation reality-grades, and a live neti-neti demo.

Run:  python3 chapter_0198_shankara_adi_shankaracharya_700.py
================================================================================
"""

import numpy as np

RNG = np.random.default_rng(198)  # 198 == chapter id


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def _tanh(x):
    return np.tanh(x)


def _dtanh(a):
    # derivative of tanh expressed in terms of its OUTPUT a = tanh(z)
    return 1.0 - a * a


def _cos(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ============================================================================
# The synthetic "Advaita world"
# ----------------------------------------------------------------------------
# Generative story (this *is* Advaita cosmology written as data):
#   - There is ONE hidden substrate w_true in R^D  ("Brahman", one without a
#     second).
#   - Every appearance is that same substrate PLUS a low-rank superimposition
#     of name-and-form:   x = w_true + U @ c + small_noise
#     where c ~ R^K is the appearance's private name-form latent (K << D) and
#     U (D x K) is the fixed "projection into name-and-form" (maya's loom).
#   - The name-form latents are drawn zero-mean, so across the whole field of
#     appearances the shared invariant is exactly the substrate.
# A "sublation pair" (x_a, x_b) is two appearances that share NOTHING of their
# name-form (independent c's) but share EVERYTHING of their substrate: the test
# of viveka is that both, stripped, yield the one witness.
# ============================================================================
def make_advaita_world(n=512, D=24, K=4, amp=1.6, noise=0.02, seed=185):
    rng = np.random.default_rng(seed)
    w_true = rng.normal(0.0, 1.0, size=D)          # the one substrate
    U = rng.normal(0.0, 1.0, size=(D, K))          # maya's loom (name-form basis)
    U /= np.linalg.norm(U, axis=0, keepdims=True)  # unit columns for conditioning
    U *= amp                                       # give name-form real amplitude

    C = rng.normal(0.0, 1.0, size=(n, K))          # per-appearance name-forms
    C -= C.mean(axis=0, keepdims=True)             # zero-mean name-form (key!)
    X = w_true[None, :] + C @ U.T
    X += rng.normal(0.0, noise, size=X.shape)      # pratibhasika flicker

    # paired "second views": same substrate, freshly drawn name-form
    C2 = rng.normal(0.0, 1.0, size=(n, K))
    C2 -= C2.mean(axis=0, keepdims=True)
    X2 = w_true[None, :] + C2 @ U.T
    X2 += rng.normal(0.0, noise, size=X2.shape)

    return X.astype(np.float64), X2.astype(np.float64), w_true, U


# ============================================================================
# The network
# ============================================================================
class AdhyasaSublationNetwork:
    """
    Encoder (extracts the name-form / samskara from an appearance):
        a1 = tanh(W1 x + b1)          (manomaya : associative sheath)
        a2 = tanh(W2 a1 + b2)         (vijnanamaya : discriminative sheath)
    Superimposition decoder (projects name-form back onto the substrate):
        s  = W3 a2 + b3               (the adhyasa / projected nama-rupa)
    Global substrate parameter (the one witness, shared by all inputs):
        w  in R^D                     (Brahman / sakshi)
    Reconstruction (adhyasa = substrate + superimposition):
        x_hat = w + s

    Objective:
        L_recon  = mean || x_hat - x ||^2         (model the empirical world)
        L_center = || mean_i s_i ||^2             (push the shared invariant
                                                   OUT of maya, INTO w)
        L_advaita(diagnostic) = variance across i of witness(x_i)=x_i - s_i
    """

    def __init__(self, D=24, H1=32, H2=16, seed=185):
        rng = np.random.default_rng(seed)
        s1 = np.sqrt(2.0 / D)
        s2 = np.sqrt(2.0 / H1)
        s3 = np.sqrt(2.0 / H2)
        self.D, self.H1, self.H2 = D, H1, H2
        self.W1 = rng.normal(0, s1, size=(H1, D))
        self.b1 = np.zeros(H1)
        self.W2 = rng.normal(0, s2, size=(H2, H1))
        self.b2 = np.zeros(H2)
        self.W3 = rng.normal(0, s3, size=(D, H2))
        self.b3 = np.zeros(D)
        self.w = rng.normal(0, 0.1, size=D)   # the substrate, learned

    # -- parameter (de)serialisation for the gradient check ------------------
    def get_params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.w]

    def param_names(self):
        return ["W1", "b1", "W2", "b2", "W3", "b3", "w"]

    # -- forward -------------------------------------------------------------
    def forward(self, X):
        """X: (N, D). Returns cache dict."""
        z1 = X @ self.W1.T + self.b1          # (N,H1)
        a1 = _tanh(z1)
        z2 = a1 @ self.W2.T + self.b2         # (N,H2)
        a2 = _tanh(z2)
        s = a2 @ self.W3.T + self.b3          # (N,D)  superimposition
        xhat = self.w[None, :] + s            # (N,D)  substrate + superimposition
        return dict(X=X, z1=z1, a1=a1, z2=z2, a2=a2, s=s, xhat=xhat)

    # -- loss ----------------------------------------------------------------
    def loss(self, cache, lam_center=1.0):
        X, s, xhat = cache["X"], cache["s"], cache["xhat"]
        N = X.shape[0]
        diff = xhat - X                        # (N,D)
        L_recon = np.sum(diff * diff) / N
        s_mean = s.mean(axis=0)                # (D,)
        L_center = float(np.dot(s_mean, s_mean))
        L = L_recon + lam_center * L_center
        return L, dict(L_recon=L_recon, L_center=L_center)

    # -- backward (exact analytic gradients) ---------------------------------
    def backward(self, cache, lam_center=1.0):
        X, a1, a2, s, xhat = (cache["X"], cache["a1"], cache["a2"],
                              cache["s"], cache["xhat"])
        N, D = X.shape

        diff = xhat - X                        # (N,D)
        # dL_recon/dxhat = 2/N * diff ; xhat = w + s  ->  same grad flows to w and s
        dxhat = (2.0 / N) * diff               # (N,D)

        # substrate gradient: w appears additively in every row of xhat
        dw = dxhat.sum(axis=0)                 # (D,)

        # superimposition gradient from recon
        ds = dxhat.copy()                      # (N,D)

        # centering loss:  L_center = ||mean_i s_i||^2
        s_mean = s.mean(axis=0)                # (D,)
        # dL_center/ds_i = 2/N * s_mean   (same for every row)
        ds += lam_center * (2.0 / N) * np.broadcast_to(s_mean, s.shape)

        # s = a2 @ W3.T + b3
        dW3 = ds.T @ a2                        # (D,H2)
        db3 = ds.sum(axis=0)                   # (D,)
        da2 = ds @ self.W3                     # (N,H2)

        # a2 = tanh(z2)
        dz2 = da2 * _dtanh(a2)                 # (N,H2)
        dW2 = dz2.T @ a1                       # (H2,H1)
        db2 = dz2.sum(axis=0)                  # (H2,)
        da1 = dz2 @ self.W2                    # (N,H1)

        # a1 = tanh(z1)
        dz1 = da1 * _dtanh(a1)                 # (N,H1)
        dW1 = dz1.T @ X                        # (H1,D)
        db1 = dz1.sum(axis=0)                  # (H1,)

        return [dW1, db1, dW2, db2, dW3, db3, dw]

    # -- inference: viveka / neti-neti --------------------------------------
    def witness(self, X):
        """Recover the substrate by REMOVING the superimposition: x - s(x).
        This is neti-neti in one shot: 'not this (name-form), not this...'."""
        cache = self.forward(X)
        return X - cache["s"]

    def satta_grades(self, X):
        """Per-dimension reality grade (badha / sublatability).

        Decompose each dimension's energy into three tiers:
          substrate   w_d^2        -- invariant across ALL appearances
          name-form   var_i(s_d)   -- the transactional play of maya
          flicker     var_i(r_d)   -- residual not even captured by name-form
        Grade = substrate_fraction = w^2 / (w^2 + var_s + var_r).
        Dimensions the substrate dominates approach 1 (paramarthika, hard to
        sublate); dimensions maya's loom loads heavily approach 0."""
        cache = self.forward(X)
        r = X - cache["xhat"]
        var_s = cache["s"].var(axis=0)
        var_r = r.var(axis=0)
        w2 = self.w ** 2
        grade = w2 / (w2 + var_s + var_r + 1e-9)
        return grade

    def three_levels(self, X):
        """Total energy carried by each ontological tier (per appearance)."""
        cache = self.forward(X)
        s, xhat = cache["s"], cache["xhat"]
        r = X - xhat
        E_sub = float(np.dot(self.w, self.w))           # shared, invariant
        E_nf = float((s * s).sum(axis=1).mean())        # private name-form
        E_res = float((r * r).sum(axis=1).mean())       # vanishing flicker
        return dict(paramarthika=E_sub, vyavaharika=E_nf, pratibhasika=E_res)


# ============================================================================
# MANDATORY finite-difference gradient check
# ============================================================================
def gradient_check(verbose=True):
    """Compare analytic gradients against central finite differences on a tiny
    random instance. Every chapter file must pass this."""
    rng = np.random.default_rng(0)
    D, H1, H2, N = 6, 5, 4, 8
    net = AdhyasaSublationNetwork(D=D, H1=H1, H2=H2, seed=7)
    X = rng.normal(size=(N, D))
    lam = 0.7

    def total_loss():
        cache = net.forward(X)
        L, _ = net.loss(cache, lam_center=lam)
        return L

    cache = net.forward(X)
    analytic = net.backward(cache, lam_center=lam)
    params = net.get_params()
    names = net.param_names()

    eps = 1e-6
    worst = 0.0
    for p, g, nm in zip(params, analytic, names):
        flat_p = p.ravel()
        flat_g = g.ravel()
        # check a handful of coordinates per parameter tensor
        idxs = range(flat_p.size) if flat_p.size <= 12 else \
            rng.choice(flat_p.size, size=12, replace=False)
        max_rel = 0.0
        for i in idxs:
            orig = flat_p[i]
            flat_p[i] = orig + eps
            lp = total_loss()
            flat_p[i] = orig - eps
            lm = total_loss()
            flat_p[i] = orig
            num = (lp - lm) / (2 * eps)
            ana = flat_g[i]
            denom = max(1e-8, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            max_rel = max(max_rel, rel)
        worst = max(worst, max_rel)
        if verbose:
            print(f"  grad-check {nm:>3}: max relative error = {max_rel:.3e}")
    if verbose:
        print(f"  worst relative error across all params = {worst:.3e}")
    ok = worst < 1e-5
    print(f"  GRADIENT CHECK {'PASSED' if ok else 'FAILED'} "
          f"(threshold 1e-5)")
    return ok


# ============================================================================
# Training loop
# ============================================================================
def train(net, X, X2, epochs=6000, lr=0.01, lam_center=0.1,
          b1=0.9, b2=0.999, eps=1e-8, w_true=None, log_every=750):
    """Full-batch Adam. Adam (not plain SGD) is what reliably escapes the
    'lazy' s=0 minimum in which the substrate absorbs the mean but the
    superimposition never learns to explain name-form."""
    params = net.get_params()
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    history = []
    for ep in range(1, epochs + 1):
        cache = net.forward(X)
        L, parts = net.loss(cache, lam_center=lam_center)
        grads = net.backward(cache, lam_center=lam_center)
        for i, (p, g) in enumerate(zip(params, grads)):
            m[i] = b1 * m[i] + (1 - b1) * g
            v[i] = b2 * v[i] + (1 - b2) * (g * g)
            mhat = m[i] / (1 - b1 ** ep)
            vhat = v[i] / (1 - b2 ** ep)
            p -= lr * mhat / (np.sqrt(vhat) + eps)
        if ep % log_every == 0 or ep == 1:
            # diagnostics
            wit = net.witness(X)                 # (N,D)
            wit_var = float(wit.var(axis=0).mean())   # non-duality: should -> 0
            rec = parts["L_recon"]
            line = dict(epoch=ep, L=L, L_recon=rec,
                        L_center=parts["L_center"], witness_var=wit_var)
            if w_true is not None:
                line["cos_w_true"] = _cos(net.w, w_true)
            history.append(line)
    return history


# ============================================================================
# main
# ============================================================================
def main():
    np.set_printoptions(precision=4, suppress=True, linewidth=100)
    print("=" * 74)
    print("  ADHYASA-SUBLATION NETWORK  -  Shankara / Advaita Vedanta")
    print("  cognition = superimposition ;  knowledge = subtraction ;")
    print("  the real = the one invariant witness left when name-form is removed")
    print("=" * 74)

    print("\n[1] Finite-difference gradient check")
    ok = gradient_check(verbose=True)
    assert ok, "Gradient check failed -- backprop is wrong; aborting."

    print("\n[2] Building the Advaita world (one substrate, many name-forms)")
    D, K = 24, 4
    X, X2, w_true, U = make_advaita_world(n=512, D=D, K=K, noise=0.02, seed=185)
    print(f"    appearances: {X.shape[0]}  dim(D)={D}  name-form rank(K)={K}")
    print(f"    raw per-dim variance of appearances (mean) = "
          f"{X.var(axis=0).mean():.4f}")

    print("\n[3] Training (viveka by gradient descent)")
    net = AdhyasaSublationNetwork(D=D, H1=32, H2=16, seed=185)
    hist = train(net, X, X2, epochs=6000, lr=0.01, lam_center=0.1,
                 w_true=w_true, log_every=750)
    print(f"    {'epoch':>6} {'L':>10} {'L_recon':>10} {'L_center':>10} "
          f"{'witness_var':>12} {'cos(w,w*)':>10}")
    for h in hist:
        print(f"    {h['epoch']:>6d} {h['L']:>10.5f} {h['L_recon']:>10.5f} "
              f"{h['L_center']:>10.6f} {h['witness_var']:>12.6f} "
              f"{h['cos_w_true']:>10.4f}")

    print("\n[4] Substrate recovery (did the net find the one Brahman?)")
    cw = _cos(net.w, w_true)
    scale = float(np.dot(net.w, w_true) / (np.dot(w_true, w_true) + 1e-9))
    print(f"    cosine(learned substrate, true substrate) = {cw:.4f}")
    print(f"    best-fit scale                             = {scale:.4f}")

    print("\n[5] Non-duality test (neti-neti on unseen paired views)")
    # Strip name-form from two INDEPENDENT views of the same substrate.
    w_a = net.witness(X)      # (N,D)
    w_b = net.witness(X2)     # (N,D) -- different name-forms, same substrate
    # cross-view agreement of the recovered witness, per sample
    agree = np.array([_cos(w_a[i], w_b[i]) for i in range(64)])
    # spread of the recovered witness across ALL appearances
    within_var = float(w_a.var(axis=0).mean())
    print(f"    mean cosine(witness(view A), witness(view B)) = "
          f"{agree.mean():.4f}")
    print(f"    variance of recovered witness across appearances = "
          f"{within_var:.6f}   (-> 0 means 'one without a second')")
    print(f"    variance of raw appearances across appearances   = "
          f"{X.var(axis=0).mean():.6f}")
    reduction = 1.0 - within_var / (X.var(axis=0).mean() + 1e-12)
    print(f"    name-form variance dissolved by viveka           = "
          f"{100*reduction:.2f}%")

    print("\n[6] Sublation / three-levels-of-reality readout (badha)")
    levels = net.three_levels(X)
    print("    energy per ontological tier (per appearance):")
    print(f"      paramarthika (invariant substrate) : {levels['paramarthika']:.4f}")
    print(f"      vyavaharika  (name-form / maya)     : {levels['vyavaharika']:.4f}")
    print(f"      pratibhasika (residual flicker)     : {levels['pratibhasika']:.6f}")
    ordered = (levels['paramarthika'] > levels['vyavaharika'] >
               levels['pratibhasika'])
    print(f"      tiers correctly ordered by 'reality': {ordered}")
    grades = net.satta_grades(X)
    order = np.argsort(-grades)
    print("    per-dimension substrate-fraction grade "
          "(1=hard to sublate, 0=easily sublated):")
    print(f"      most real dims   {order[:6]}: {grades[order[:6]]}")
    print(f"      most maya-ish    {order[-6:]}: {grades[order[-6:]]}")
    print(f"      grade spread = {grades.max() - grades.min():.4f}   "
          f"mean = {grades.mean():.4f}")

    print("\n[7] Live neti-neti demonstration on a single appearance")
    i = 3
    x = X[i]
    s = net.forward(x[None, :])["s"][0]
    recovered = x - s
    print(f"    appearance x[{i}]      (first 6 dims): {x[:6]}")
    print(f"    superimposition s     (first 6 dims): {s[:6]}")
    print(f"    x - s  (recovered)    (first 6 dims): {recovered[:6]}")
    print(f"    learned substrate w   (first 6 dims): {net.w[:6]}")
    print(f"    ||(x - s) - w|| = {np.linalg.norm(recovered - net.w):.5f}")

    print("\n" + "=" * 74)
    print("  SELF-TESTS")
    print("=" * 74)
    passed = True

    def check(label, cond):
        nonlocal passed
        passed = passed and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")

    check("gradient check passed", ok)
    check("reconstruction loss < 0.01", hist[-1]["L_recon"] < 0.01)
    check("substrate cosine > 0.99", abs(cw) > 0.99)
    check("witness variance collapsed > 95%", reduction > 0.95)
    check("cross-view witness agreement > 0.99", agree.mean() > 0.99)
    check("three tiers ordered: paramarthika>vyavaharika>pratibhasika",
          ordered)
    check("per-dim reality grade spreads across (0,1)",
          (grades.max() - grades.min()) > 0.3)
    print("-" * 74)
    print(f"  ALL SELF-TESTS {'PASSED' if passed else 'FAILED'}")
    print("=" * 74)
    return passed


if __name__ == "__main__":
    main()

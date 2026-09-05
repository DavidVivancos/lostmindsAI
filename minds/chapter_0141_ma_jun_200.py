# -*- coding: utf-8 -*-
"""
================================================================================
 Chapter 0141_ma_jun_200 — Ma Jun (c. 200 - 265 CE), Cao Wei, Three Kingdoms China
 THE TRIAL-LOOM  ·  a Minimum-Description-Length Generate-and-Test network
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 8 Minds 141 - 160 Available on Amazon https://www.amazon.com/dp/B0HH8RTCXF
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 141: Ma Jun (200-265 CE, General of Cao Wei, Three Kingdoms China)
================================================================================   

WHY THIS ARCHITECTURE (and not a Transformer)
------------------------------------------------------------------------------
Ma Jun left no philosophy. He left machines and one sentence, recorded by his
friend the poet Fu Xuan: "Empty arguments with words cannot compare with a test
which will show practical results."  Two acts define his mind:

  1. THE LOOM REFORM. The figured-silk drawlooms of his day used 50 - 60
     heddles and 50 - 60 treadles. Ma Jun rebuilt the loom to use only TWELVE
     treadles, and it wove *more* intricate patterns, not fewer. He did not add
     capacity; he found the minimal control structure that generated the same
     (larger) space of cloth. That is COMPRESSION as intelligence.

  2. THE TRIAL. When courtiers argued the south-pointing chariot was myth, he
     refused to debate. He built one and let it be TESTED. Truth is settled by
     the trial, not by the eloquence of the claimant.

So the model here is NOT attention-over-stored-keys. It is a loom:

    a bank of many "treadles" (latent control channels), each with a learnable
    GATE that says whether the treadle is strung at all. The network is asked
    to reproduce a target cloth (the TRIAL: does the woven output match the
    command?). A description-length penalty charges rent for every strung
    treadle. Gradient pressure therefore does to the model exactly what Ma Jun
    did to the loom: it prunes treadles until only the few that earn their keep
    remain --- the loom-reform emerges as a *training dynamic*, not a setting.

    A "trial gate" reports, at evaluation, the fraction of samples whose woven
    output passes a fixed error threshold --- the machine's own version of
    "a test which shows practical results" rather than an argument.

Everything is pure NumPy, from scratch. Backprop is hand-derived. A finite-
difference gradient check (mandatory) guards every parameter. There is a real
training loop, real synthetic "figured-cloth" data, and self-tests that verify:
gradients, loss descent, emergent treadle compression, and rising trial pass.

Run:  python3 chapter_0141_ma_jun_200.py
"""

import numpy as np


# ==============================================================================
# SECTION 1 — DETERMINISM
# ==============================================================================
# Ma Jun's whole point is repeatable trials. A fixed seed makes every run a
# repeatable trial rather than an anecdote.
def set_seed(seed=1420):
    np.random.seed(seed)


# ==============================================================================
# SECTION 2 — SMOOTH PRIMITIVES (chosen so the gradient check is exact)
# ==============================================================================
def tanh(x):
    return np.tanh(x)

def dtanh(y):            # derivative expressed in terms of the OUTPUT y = tanh(x)
    return 1.0 - y * y

def sigmoid(x):
    # numerically stable logistic; this is the "is-this-treadle-strung?" gate
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


# ==============================================================================
# SECTION 3 — THE FIGURED-CLOTH DATASET  (the "trial material")
# ==============================================================================
# A damask loom repeats a *small* set of weave-structures (motifs). Any bolt of
# figured silk is a sparse combination of a few motifs. We therefore generate
# each target pattern x as   x = B_true @ c   where B_true holds K_true fixed
# motifs and c is a sparse, non-negative coefficient vector (1 - 3 active
# motifs). The model is NOT told K_true. It starts life with many treadles and
# must DISCOVER that few suffice --- the loom-reform, as a learning problem.
class FiguredCloth:
    def __init__(self, D=32, K_true=6, n=256, active_max=3, noise=0.02):
        self.D, self.K_true = D, K_true
        # Fixed, well-separated motifs (columns): the true "weave-structures".
        M = np.random.randn(D, K_true)
        # orthonormalise so motifs are distinct structures, like real weaves
        Q, _ = np.linalg.qr(M)
        self.B_true = Q[:, :K_true]
        X = np.zeros((n, D))
        for i in range(n):
            k = np.random.randint(1, active_max + 1)          # 1..active_max motifs
            idx = np.random.choice(K_true, size=k, replace=False)
            coeff = np.zeros(K_true)
            coeff[idx] = np.random.uniform(0.5, 1.5, size=k)  # positive "lifts"
            X[i] = self.B_true @ coeff
        X += noise * np.random.randn(n, D)                    # loom is imperfect
        # normalise each bolt to unit energy so recon error is comparable
        X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        self.X = X


# ==============================================================================
# SECTION 4 — THE TRIAL-LOOM MODEL
# ==============================================================================
# Encoder ("the drafter"):    x -> hidden -> raw treadle lift t
# Gate    ("the stringing"):   a = sigmoid(g); code c = t * a  (prune treadles)
# Decoder ("the loom harness"): a DICTIONARY of unit-norm motifs. Each treadle
#          owns one weave-structure of FIXED prominence; the woven cloth is the
#          additive sum of the strung treadles' motifs, each scaled by its lift:
#                       xhat = c @ Dm  (+ bias)
#
# Anchoring every motif to unit norm is the crucial faithfulness: on a real loom
# a treadle either lifts its heddles or it does not; the network cannot "cheat"
# by making a nearly-closed treadle count for a lot through a huge weight. To
# weave a cloth built from six weave-structures, the loom MUST keep six treadles
# genuinely strung. The gate penalty then charges rent on complexity, so the
# loom reforms itself toward the minimal set --- Ma Jun's 60 -> 12, as learning.
class TrialLoom:
    def __init__(self, D=32, H=24, K=24):
        self.D, self.H, self.K = D, H, K

        def glorot(shape):
            fan_in, fan_out = shape[0], shape[-1]
            lim = np.sqrt(6.0 / (fan_in + fan_out))
            return np.random.uniform(-lim, lim, shape)

        # encoder (the drafter who reads the target cloth)
        self.W1 = glorot((D, H));  self.b1 = np.zeros(H)
        self.W2 = glorot((H, K));  self.b2 = np.zeros(K)
        # gates: start positive so every treadle begins strung --- exactly like
        # inheriting a bloated 50-60 treadle loom that "works" but wastes motion.
        self.g = np.full(K, 2.0)
        # decoder dictionary: one motif per treadle, rows normalised to unit norm
        Dm = glorot((K, D))
        self.Dm = Dm / (np.linalg.norm(Dm, axis=1, keepdims=True) + 1e-9)
        self.b4 = np.zeros(D)

    # ---- ordered parameter access (used by optimiser + gradient check) -------
    def params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.g, self.Dm, self.b4]

    def names(self):
        return ["W1", "b1", "W2", "b2", "g", "Dm", "b4"]

    def set_params(self, plist):
        (self.W1, self.b1, self.W2, self.b2, self.g, self.Dm, self.b4) = plist

    def renormalise_motifs(self):
        """Project each motif back to unit norm (a treadle's weave-structure has
        fixed prominence). Applied between optimiser steps, not inside the loss."""
        self.Dm = self.Dm / (np.linalg.norm(self.Dm, axis=1, keepdims=True) + 1e-9)

    # ---- FORWARD -------------------------------------------------------------
    def forward(self, x):
        z1 = x @ self.W1 + self.b1
        h1 = tanh(z1)
        t  = h1 @ self.W2 + self.b2          # raw treadle lift (linear)
        a  = sigmoid(self.g)                  # strung-treadle gate in (0,1)
        c  = t * a                            # gated code: unstrung treadles fade
        xhat = c @ self.Dm + self.b4          # additive weave of strung motifs
        cache = (x, h1, t, a, c, xhat)
        return xhat, cache

    # ---- LOSS ----------------------------------------------------------------
    # L = reconstruction (the TRIAL)
    #   + treadle-retirement cost (GROUP sparsity, L2,1): the whole column of a
    #                   treadle's activations across the cloth is penalised by
    #                   its L2 norm, so an unneeded treadle is retired ENTIRELY
    #                   --- Ma Jun physically taking a treadle off the loom, not
    #                   merely lifting it less. This is what makes the strung
    #                   count fall toward the true minimum.
    #   + a small structural rent on strung gates (retires the dead treadle's
    #                   gate too, making 'strung / unstrung' explicit)
    #   + a whisper of encoder weight decay for conditioning.
    _GEPS = 1e-8

    def loss(self, x, lam_group=0.001, lam_gate=0.01, lam_wd=1e-5):
        xhat, cache = self.forward(x)
        c = cache[4]
        N, D = x.shape
        L_rec = np.sum((xhat - x) ** 2) / (N * D)
        col = np.sqrt(np.mean(c * c, axis=0) + self._GEPS)  # per-treadle RMS lift
        L_group = lam_group * np.sum(col)                   # retire whole treadles
        a = sigmoid(self.g)
        L_gate = lam_gate * np.mean(a)                      # retire dead gates
        L_wd = lam_wd * (np.sum(self.W1**2) + np.sum(self.W2**2))
        return L_rec + L_group + L_gate + L_wd, cache

    # ---- BACKWARD (hand-derived analytic gradients) --------------------------
    def backward(self, cache, lam_group=0.001, lam_gate=0.01, lam_wd=1e-5):
        x, h1, t, a, c, xhat = cache
        N, D = x.shape
        K = self.K

        dxhat = (2.0 / (N * D)) * (xhat - x)          # dL_rec/dxhat
        gDm = c.T @ dxhat                              # (K,D); only rec touches Dm
        gb4 = np.sum(dxhat, axis=0)                    # (D,)
        dc = dxhat @ self.Dm.T                         # (N,K) from reconstruction
        col = np.sqrt(np.mean(c * c, axis=0) + self._GEPS)  # (K,) per-treadle RMS
        dc += lam_group * (c / (N * col))             # + group-sparsity term

        dt = dc * a                                    # code c = t * a
        da = np.sum(dc * t, axis=0)                    # (K,) rec + group via c
        da += (lam_gate / K)                           # + structural gate rent
        gg = da * (a * (1.0 - a))                      # through sigmoid gate

        dz2 = dt
        gW2 = h1.T @ dz2
        gb2 = np.sum(dz2, axis=0)
        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * dtanh(h1)
        gW1 = x.T @ dz1
        gb1 = np.sum(dz1, axis=0)

        gW1 += 2 * lam_wd * self.W1                    # weight decay
        gW2 += 2 * lam_wd * self.W2

        return [gW1, gb1, gW2, gb2, gg, gDm, gb4]

    # ---- diagnostics ---------------------------------------------------------
    def active_treadles(self, X, frac=0.05):
        """Honest 'strung treadle' count: how many treadles actually carry
        signal into the cloth. We look at the code c = t * a itself (not the
        gate alone), so amplitude cannot hide in the encoder. A treadle is
        'strung' if its mean absolute lift exceeds `frac` of the busiest one."""
        _, cache = self.forward(X)
        c = cache[4]
        usage = np.mean(np.abs(c), axis=0)             # (K,)
        peak = np.max(usage) + 1e-12
        return int(np.sum(usage > frac * peak))

    def trial_pass_rate(self, X, tau):
        """The TRIAL GATE: fraction of bolts whose woven output passes the test.
        The machine's own 'test that shows practical results', not an argument."""
        xhat, _ = self.forward(X)
        per = np.mean((xhat - X) ** 2, axis=1)         # per-sample recon error
        return float(np.mean(per < tau)), float(np.mean(per))


# ==============================================================================
# SECTION 5 — ADAM OPTIMISER (per-parameter state)
# ==============================================================================
class Adam:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        out = []
        for i, (p, gr) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * gr
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (gr * gr)
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            out.append(p - self.lr * mhat / (np.sqrt(vhat) + self.eps))
        return out


# ==============================================================================
# SECTION 6 — FINITE-DIFFERENCE GRADIENT CHECK  (MANDATORY)
# ==============================================================================
# The whole doctrine of this chapter is "test, do not argue". So we do not
# TRUST the analytic gradients above --- we TEST them against numerical
# central differences, and refuse to proceed unless they agree.
def gradient_check(verbose=True):
    set_seed(7)
    D, H, K, N = 8, 6, 5, 4
    model = TrialLoom(D=D, H=H, K=K)
    X = np.random.randn(N, D)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9

    lam_group, lam_gate, lam_wd = 0.001, 0.01, 1e-5
    _, cache = model.loss(X, lam_group, lam_gate, lam_wd)
    analytic = model.backward(cache, lam_group, lam_gate, lam_wd)

    eps = 1e-6
    max_rel = 0.0
    params = model.params()
    for pi, P in enumerate(params):
        flat = P.ravel()
        # sample a handful of coordinates per parameter for speed
        idxs = range(flat.size) if flat.size <= 12 else \
            np.random.choice(flat.size, 12, replace=False)
        for idx in idxs:
            orig = flat[idx]
            flat[idx] = orig + eps
            Lp, _ = model.loss(X, lam_group, lam_gate, lam_wd)
            flat[idx] = orig - eps
            Lm, _ = model.loss(X, lam_group, lam_gate, lam_wd)
            flat[idx] = orig
            num = (Lp - Lm) / (2 * eps)
            ana = analytic[pi].ravel()[idx]
            denom = max(1e-12, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            max_rel = max(max_rel, rel)
    if verbose:
        print(f"  [gradient check] max relative error = {max_rel:.2e}")
    return max_rel


# ==============================================================================
# SECTION 7 — TRAINING LOOP  (repeated trials on figured cloth)
# ==============================================================================
def train(model, X, epochs=400, batch=32, lr=3e-3,
          lam_group=0.001, lam_gate=0.01, lam_wd=1e-5,
          tau=None, log_every=50, verbose=True):
    N = X.shape[0]
    opt = Adam(model.params(), lr=lr)
    if tau is None:
        tau = 3.0 * (0.02 ** 2)          # threshold near the loom's noise floor
    history = []
    for ep in range(epochs):
        perm = np.random.permutation(N)
        ep_loss = 0.0
        for s in range(0, N, batch):
            xb = X[perm[s:s + batch]]
            L, cache = model.loss(xb, lam_group, lam_gate, lam_wd)
            grads = model.backward(cache, lam_group, lam_gate, lam_wd)
            model.set_params(opt.step(model.params(), grads))
            model.renormalise_motifs()               # keep motifs unit-norm
            ep_loss += L * len(xb)
        ep_loss /= N
        rate, mean_err = model.trial_pass_rate(X, tau)
        active = model.active_treadles(X)
        history.append((ep, ep_loss, active, rate, mean_err))
        if verbose and (ep % log_every == 0 or ep == epochs - 1):
            print(f"  epoch {ep:4d} | loss {ep_loss:.5f} | strung treadles "
                  f"{active:2d}/{model.K} | trial pass {rate*100:5.1f}% | "
                  f"mean err {mean_err:.5f}")
    return history, tau


# ==============================================================================
# SECTION 8 — SELF-TESTS  (each is a trial that must show a practical result)
# ==============================================================================
def run_all():
    print("=" * 74)
    print(" THE TRIAL-LOOM  —  Ma Jun (c.200-265 CE)  —  verification run")
    print("=" * 74)

    # -- Trial 1: the gradients must survive numerical testing ----------------
    print("\n[1] Gradient trial (analytic vs finite-difference)")
    max_rel = gradient_check(verbose=True)
    assert max_rel < 1e-4, f"gradient check FAILED (max rel {max_rel:.2e})"
    print("    PASS: hand-derived gradients agree with numerical test.")

    # -- Build the loom and the cloth -----------------------------------------
    set_seed(1420)
    K_true = 6
    data = FiguredCloth(D=32, K_true=K_true, n=256, active_max=3, noise=0.02)
    model = TrialLoom(D=32, H=24, K=24)

    print(f"\n[2] Setup: inherited a bloated loom of {model.K} treadles; the")
    print(f"    cloth truly needs only {K_true} weave-structures. Can the loom")
    print(f"    reform itself down toward that minimum by trial alone?")

    start_active = model.active_treadles(data.X)
    _, cache0 = model.loss(data.X)
    L0 = np.sum((cache0[5] - data.X) ** 2) / data.X.size

    # -- Trial 2: train -------------------------------------------------------
    print("\n[3] Training (repeated trials on figured cloth)")
    history, tau = train(model, data.X, epochs=600, batch=32, lr=3e-3,
                         lam_group=0.0012, lam_gate=0.01, lam_wd=1e-5, log_every=100)

    end_active = model.active_treadles(data.X)
    Lf = history[-1][1]
    final_rate, final_err = model.trial_pass_rate(data.X, tau)

    # -- Trial 3: loss must fall ---------------------------------------------
    print("\n[4] Trials of the result")
    print(f"    reconstruction error: start {L0:.5f} -> end {final_err:.5f}")
    assert final_err < L0, "reconstruction did not improve"
    print("    PASS: the woven cloth now matches the command far better.")

    # -- Trial 4: the loom must COMPRESS (Ma Jun's reform) -------------------
    print(f"    strung treadles: start {start_active} -> end {end_active}")
    assert end_active < start_active, "no compression occurred"
    assert end_active <= 12, ("loom did not reform below Ma Jun's twelve "
                              f"treadles (got {end_active})")
    assert end_active >= 2, ("loom collapsed to nothing --- over-pruned "
                             f"(got {end_active})")
    print(f"    PASS: the loom reformed itself from {start_active} treadles to "
          f"{end_active} (Ma Jun reached twelve; the true minimum was "
          f"{K_true}).")

    # -- Trial 5: the trial-gate pass rate must rise -------------------------
    print(f"    trial pass rate: end {final_rate*100:.1f}% of bolts pass the test")
    assert final_rate > 0.5, "most bolts still fail the practical test"
    print("    PASS: a majority of bolts now pass 'a test that shows results'.")

    # -- Trial 6: determinism (a trial must be repeatable) -------------------
    set_seed(1420)
    d2 = FiguredCloth(D=32, K_true=6, n=256, active_max=3, noise=0.02)
    same = np.allclose(d2.X, data.X)
    assert same, "trial was not repeatable under the same seed"
    print("    PASS: identical seed reproduces identical cloth (repeatable trial).")

    print("\n" + "=" * 74)
    print(" ALL TRIALS PASSED — the loom that argues nothing and proves"
          " everything.")
    print("=" * 74)
    return {
        "grad_max_rel": max_rel,
        "start_active": start_active,
        "end_active": end_active,
        "start_err": float(L0),
        "end_err": float(final_err),
        "final_pass_rate": float(final_rate),
    }


if __name__ == "__main__":
    run_all()

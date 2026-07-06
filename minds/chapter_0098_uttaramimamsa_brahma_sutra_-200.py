#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
chapter_0098_uttaramimamsa_brahma_sutra_-200.py
Mind #98 - Uttaramimamsa (the Brahma Sutra), attributed to Badarayana, c. 200 BCE
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 5 Minds 81 - 100 Available on Amazon https://www.amazon.com/dp/B0H7LP5LP2
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 0098 · Uttaramimamsa (the Brahma Sutra), attributed to Badarayana, c. 200 BCE
================================================================================

WHAT THIS FILE IS
-----------------
A from-scratch, pure-NumPy, *trainable* neural architecture whose computation is
shaped by the single distinctive cognitive operation of the Brahma Sutra:
SAMANVAYA -- the convergent reconciliation of a body of apparently CONTRADICTORY
testimony onto ONE underlying referent, achieved not by deleting any datum but
by re-construing each statement at its proper LEVEL of reality.

The Brahma Sutra's first chapter is literally named *Samanvaya* ("harmonization").
Its hinge aphorism, I.1.4 "tat tu samanvayat", asserts that the many divergent
Upanishadic passages all CONVERGE on a single referent (Brahman) once each is
read at the right grade. The text's method (the *adhikarana*) takes a disputed
statement, states the doubt (samsaya), the wrong reading (purvapaksha), and the
reconciling conclusion (siddhanta). The criterion for "true purport" is the
six-mark coherence rubric (shad-linga).

We turn that doctrine into a learning machine. The model is NOT a Transformer and
does NOT predict the next token. Its native act is COHERENCE-RESOLUTION:

    given many conflicting witness-vectors about a hidden truth, assign each a
    reality-LEVEL, re-LEVEL it into the ultimate frame, and CONVERGE the re-leveled
    witnesses onto one referent embedding -- the samanvaya fixed point.

THE THREE LEVELS (after the Vedanta grades of reality)
    0  paramarthika   ultimate / direct testimony of the truth
    1  vyavaharika    empirical / the truth seen through a conventional projection
    2  pratibhasika   illusory  / actively misleading (the rope-as-snake)

ARCHITECTURE -- the Samanvaya Reconciliation Engine (SRE)
    V    = X @ Wv                         encode each statement (witness -> value)
    P    = softmax(V @ Wlvl)              soft reality-LEVEL of each statement
    VR   = sum_L P[:,L] * (V @ T_L)       RE-LEVEL: pull each witness to the ultimate
    A    = softmax(VR @ w_conv)           convergence weights over witnesses
    z    = A . VR                         SAMANVAYA: one reconciled latent
    r^   = z @ Wo                         the recovered referent (the "Brahman" of
                                          this instance)

LOSSES
    L_recon  : recover the hidden referent r*            (the goal of the inquiry)
    L_level  : classify each statement's grade           (viveka / discrimination)
    L_cohere : re-leveled witnesses must agree at z      (avirodha / non-contradiction)

Every parameter is trained by hand-derived reverse-mode gradients. A
finite-difference gradient check (mandatory) verifies them before training.

Run:  python3 0098_Neuron.py
================================================================================
"""

import numpy as np

RNG = np.random.default_rng(98)   # 98 = the mind's index in the terabook


# ============================================================================
# 0.  SMALL NUMERICAL HELPERS
# ============================================================================

def softmax_rows(Z):
    """Row-wise softmax, numerically stabilised. Z: (n, k) -> (n, k)."""
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def softmax_vec(z):
    """Softmax over a 1-D vector. z: (n,) -> (n,)."""
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# ============================================================================
# 1.  THE MODEL: Samanvaya Reconciliation Engine
# ============================================================================

class SamanvayaEngine:
    """
    A differentiable adhikarana. It ingests a set of witness vectors (rows of X),
    grades each by reality-level, re-levels them toward the ultimate frame, and
    converges them onto a single referent. d = witness/referent dim, h = latent
    dim, L = 3 reality levels (fixed).
    """

    LEVELS = 3  # paramarthika, vyavaharika, pratibhasika

    def __init__(self, d, h, d_out=None, seed=0):
        # d     : input witness dim (testimony about r* PLUS a grade-cue tail)
        # d_out : referent dim (the recovered Brahman lives here)
        self.d, self.h = d, h
        self.d_out = d if d_out is None else d_out
        r = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(d)
        # learnable parameters (the "doctrine" the engine acquires)
        self.P = {
            "Wv":    r.normal(0, s, (d, h)),                       # witness -> value
            "Wlvl":  r.normal(0, s, (h, self.LEVELS)),             # value  -> level logits
            "T0":    np.eye(h) + r.normal(0, 0.05, (h, h)),        # re-level: paramarthika
            "T1":    np.eye(h) + r.normal(0, 0.05, (h, h)),        # re-level: vyavaharika
            "T2":    np.eye(h) + r.normal(0, 0.05, (h, h)),        # re-level: pratibhasika
            "wconv": r.normal(0, s, (h,)),                         # convergence direction
            "Wo":    r.normal(0, 1.0/np.sqrt(h), (h, self.d_out)), # latent -> referent
        }

    # -- forward -----------------------------------------------------------
    def forward(self, X, r_star, Y, alpha=3.0, beta=0.3, conv_temp=1.0):
        """
        X         : (N, d)   witness vectors for ONE adhikarana (one instance)
        r_star    : (d,)     hidden referent the inquiry should recover (Brahman)
        Y         : (N, 3)   one-hot true reality-level of each witness
        conv_temp : float    convergence-softmax temperature. The free parameter the
                             three commentators disagreed about: conv_temp -> 0 gives a
                             hard, near-Advaitic collapse onto a single graded witness;
                             large conv_temp gives a soft, Visistadvaitic blend that
                             preserves several witnesses; 1.0 is the trained default.
        Returns (loss, cache). cache holds everything backward() needs.
        """
        P = self.P
        N = X.shape[0]; h = self.h; d = self.d

        V    = X @ P["Wv"]                                   # (N,h)
        Llog = V @ P["Wlvl"]                                 # (N,3)
        Pr   = softmax_rows(Llog)                            # (N,3) soft level

        M0 = V @ P["T0"]; M1 = V @ P["T1"]; M2 = V @ P["T2"] # (N,h) each
        VR = (Pr[:, 0:1] * M0 + Pr[:, 1:2] * M1 + Pr[:, 2:3] * M2)  # (N,h)

        s  = VR @ P["wconv"]                                 # (N,)
        A  = softmax_vec(s / conv_temp)                      # (N,) tunable convergence
        z  = A @ VR                                          # (h,)
        r_hat = z @ P["Wo"]                                  # (d,)

        # losses
        diff = VR - z[None, :]                               # (N,h)
        c    = (diff ** 2).mean(axis=1)                      # (N,) per-witness incoherence
        L_recon  = ((r_hat - r_star) ** 2).mean()
        eps = 1e-12
        L_level  = -(Y * np.log(Pr + eps)).sum(axis=1).mean()
        L_cohere = (A * c).sum()
        loss = L_recon + alpha * L_level + beta * L_cohere

        cache = dict(X=X, r_star=r_star, Y=Y, alpha=alpha, beta=beta,
                     conv_temp=conv_temp,
                     V=V, Pr=Pr, M0=M0, M1=M1, M2=M2, VR=VR, s=s, A=A, z=z,
                     r_hat=r_hat, diff=diff, c=c, N=N)
        return loss, cache

    # -- backward (hand-derived reverse-mode) ------------------------------
    def backward(self, cache):
        P = self.P
        X, r_star, Y = cache["X"], cache["r_star"], cache["Y"]
        alpha, beta  = cache["alpha"], cache["beta"]
        V, Pr        = cache["V"], cache["Pr"]
        M0, M1, M2   = cache["M0"], cache["M1"], cache["M2"]
        VR, A, z     = cache["VR"], cache["A"], cache["z"]
        r_hat, diff  = cache["r_hat"], cache["diff"]
        c, N         = cache["c"], cache["N"]
        conv_temp    = cache["conv_temp"]
        h, d_out     = self.h, self.d_out

        g = {k: np.zeros_like(v) for k, v in P.items()}

        # ---- L_recon = mean((r_hat - r*)^2) ----
        g_rhat = (2.0 / d_out) * (r_hat - r_star)            # (d_out,)
        g["Wo"] += np.outer(z, g_rhat)                       # (h,d)
        gz = P["Wo"] @ g_rhat                                # (h,) recon path

        # ---- L_cohere = sum_i A_i * mean_j (VR_ij - z_j)^2 ----
        # explicit partials
        gVR = beta * (A[:, None] * (2.0 / h) * diff)         # via diff in VR
        gz += beta * (-(2.0 / h) * (A[:, None] * diff).sum(axis=0))  # via z in diff
        gA = beta * c                                        # via the A weights

        # ---- z = A . VR ----
        gVR += A[:, None] * gz[None, :]                      # dz/dVR_i = A_i I
        gA  += VR @ gz                                       # dz/dA_i  = VR_i

        # ---- A = softmax(s / conv_temp) ----
        dotA = (A * gA).sum()
        gs = A * (gA - dotA)                                 # grad wrt softmax input
        gs = gs / conv_temp                                  # chain through s/conv_temp

        # ---- s = VR @ wconv ----
        gVR += np.outer(gs, P["wconv"])                      # (N,h)
        g["wconv"] += VR.T @ gs                              # (h,)

        # ---- VR = sum_L Pr[:,L] * M_L ;  M_L = V @ T_L ----
        gPr = np.zeros_like(Pr)
        for L, (M, Tname) in enumerate([(M0, "T0"), (M1, "T1"), (M2, "T2")]):
            gPr[:, L] += (gVR * M).sum(axis=1)               # dVR/dPr_L
            gM = gVR * Pr[:, L:L + 1]                        # dVR/dM_L
            g[Tname] += V.T @ gM                             # M = V @ T
            # accumulate into gV (added below via list)
            if L == 0:
                gV = gM @ P["T0"].T
            else:
                gV += gM @ P[Tname].T

        # ---- L_level (cross-entropy) flows into Pr ----
        eps = 1e-12
        gPr += alpha * (-(1.0 / N) * (Y / (Pr + eps)))

        # ---- Pr = softmax_rows(Llog) ----
        rowdot = (Pr * gPr).sum(axis=1, keepdims=True)
        gLlog = Pr * (gPr - rowdot)                          # (N,3)

        # ---- Llog = V @ Wlvl ----
        g["Wlvl"] += V.T @ gLlog
        gV += gLlog @ P["Wlvl"].T

        # ---- V = X @ Wv ----
        g["Wv"] += X.T @ gV

        return g


# ============================================================================
# 2.  SYNTHETIC ADHIKARANA DATA
#     A hidden referent r*, witnessed at three reality-levels. The empirical and
#     illusory witnesses *contradict* the direct ones at the surface; only after
#     correct re-leveling do they converge. This is the samanvaya problem.
# ============================================================================

def make_world(d, seed):
    """Fixed empirical projection M_emp and illusory permutation, per 'world'."""
    r = np.random.default_rng(seed)
    # empirical view: rotate + damp the truth (the world as conventionally seen)
    Q, _ = np.linalg.qr(r.normal(0, 1, (d, d)))
    M_emp = Q * 0.8
    # illusory view: sign-flip + shuffle (the snake seen on the rope)
    perm = r.permutation(d)
    return M_emp, perm


def sample_instance(d_core, n_per_level, M_emp, perm, seed, cue=0.6):
    """
    Build one adhikarana: a hidden r* (d_core dims), and witnesses at three
    levels. Each witness = [ testimony(d_core) | grade-cue(3) ].

        level 0 paramarthika : r* + small noise            (direct)
        level 1 vyavaharika  : M_emp @ r* + noise           (projected truth)
        level 2 pratibhasika : -r*[perm] + larger noise     (illusory contradiction)

    The grade-cue tail is a NOISY mark of the witness's level (the shad-linga:
    contextual signs of a passage's true grade). It makes discrimination (viveka)
    genuinely learnable without revealing r*. Returns X (N, d_core+3),
    r_star (d_core,), Y (N,3).
    """
    r = np.random.default_rng(seed)
    r_star = r.normal(0, 1, d_core); r_star /= (np.linalg.norm(r_star) + 1e-9)
    rows, labs = [], []

    def make_core(level):
        if level == 0:
            return r_star + 0.05 * r.normal(0, 1, d_core)
        if level == 1:
            return M_emp @ r_star + 0.05 * r.normal(0, 1, d_core)
        return -r_star[perm] + 0.10 * r.normal(0, 1, d_core)

    for level in (0, 1, 2):
        for _ in range(n_per_level):
            core = make_core(level)
            mark = np.zeros(3); mark[level] = cue
            mark = mark + 0.15 * r.normal(0, 1, 3)          # noisy grade cue
            rows.append(np.concatenate([core, mark]))
            labs.append(level)

    X = np.array(rows)
    Y = np.zeros((X.shape[0], 3)); Y[np.arange(X.shape[0]), labs] = 1.0
    order = r.permutation(X.shape[0])                       # order carries no info
    return X[order], r_star, Y[order]


def make_dataset(n_instances, d_core, n_per_level, world_seed, base_seed):
    M_emp, perm = make_world(d_core, world_seed)
    data = [sample_instance(d_core, n_per_level, M_emp, perm, base_seed + i)
            for i in range(n_instances)]
    return data, (M_emp, perm)


# ============================================================================
# 3.  GRADIENT CHECK  (mandatory: analytic vs finite difference)
# ============================================================================

def gradient_check(conv_temp=1.0):
    print("=" * 72)
    print(f"FINITE-DIFFERENCE GRADIENT CHECK  (conv_temp={conv_temp})")
    print("=" * 72)
    d_core, h = 5, 6
    d_in = d_core + 3
    model = SamanvayaEngine(d_in, h, d_out=d_core, seed=1)
    M_emp, perm = make_world(d_core, seed=7)
    X, r_star, Y = sample_instance(d_core, n_per_level=2, M_emp=M_emp, perm=perm, seed=3)

    loss, cache = model.forward(X, r_star, Y, alpha=0.5, beta=0.3, conv_temp=conv_temp)
    grads = model.backward(cache)

    eps = 1e-6
    worst = 0.0
    for name, W in model.P.items():
        Wf = W.ravel(); gf = grads[name].ravel()
        idxs = range(Wf.size) if Wf.size <= 24 else \
            np.random.default_rng(0).choice(Wf.size, 24, replace=False)
        num = np.zeros(len(list(idxs)) if hasattr(idxs, '__len__') else Wf.size)
        idxs = list(idxs)
        for j, i in enumerate(idxs):
            orig = Wf[i]
            Wf[i] = orig + eps; lp, _ = model.forward(X, r_star, Y, 0.5, 0.3, conv_temp)
            Wf[i] = orig - eps; lm, _ = model.forward(X, r_star, Y, 0.5, 0.3, conv_temp)
            Wf[i] = orig
            num[j] = (lp - lm) / (2 * eps)
        ana = gf[idxs]
        rel = np.abs(num - ana) / (np.abs(num) + np.abs(ana) + 1e-12)
        m = rel.max()
        worst = max(worst, m)
        print(f"  {name:6s} shape={str(W.shape):10s} max rel err = {m:.3e}")
    print("-" * 72)
    ok = worst < 1e-5
    print(f"  WORST relative error = {worst:.3e}   ->   "
          f"{'PASS' if ok else 'FAIL'} (threshold 1e-5)")
    print()
    assert ok, "Gradient check FAILED"
    return ok


# ============================================================================
# 4.  TRAINING  (Adam, pure NumPy)
# ============================================================================

class Adam:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (grads[k] ** 2)
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


def evaluate(model, data):
    """Return (mean referent-cosine-recovery, level-accuracy)."""
    cos, correct, total = [], 0, 0
    for X, r_star, Y in data:
        _, cache = model.forward(X, r_star, Y)
        rh = cache["r_hat"]
        cos.append(float(rh @ r_star /
                         ((np.linalg.norm(rh) + 1e-9) * (np.linalg.norm(r_star) + 1e-9))))
        pred = cache["Pr"].argmax(axis=1); true = Y.argmax(axis=1)
        correct += int((pred == true).sum()); total += len(true)
    return float(np.mean(cos)), correct / total


def train():
    print("=" * 72)
    print("TRAINING  -  the engine learns to reconcile contradictory testimony")
    print("=" * 72)
    d_core, h = 8, 16
    d_in = d_core + 3
    train_data, world = make_dataset(240, d_core, n_per_level=4, world_seed=11, base_seed=1000)
    test_data, _      = make_dataset(60,  d_core, n_per_level=4, world_seed=11, base_seed=9000)

    model = SamanvayaEngine(d_in, h, d_out=d_core, seed=42)
    opt = Adam(model.P, lr=5e-3)

    c0, a0 = evaluate(model, test_data)
    print(f"  before training :  referent-recovery cos = {c0:+.3f}   "
          f"level-acc = {a0*100:5.1f}%")

    epochs = 90
    for ep in range(1, epochs + 1):
        order = RNG.permutation(len(train_data))
        running = 0.0
        for i in order:
            X, r_star, Y = train_data[i]
            loss, cache = model.forward(X, r_star, Y)
            grads = model.backward(cache)
            opt.step(model.P, grads)
            running += loss
        if ep % 10 == 0 or ep == 1:
            cT, aT = evaluate(model, test_data)
            print(f"  epoch {ep:3d} :  train loss = {running/len(train_data):.4f}   "
                  f"recovery cos = {cT:+.3f}   level-acc = {aT*100:5.1f}%")

    cF, aF = evaluate(model, test_data)
    print("-" * 72)
    print(f"  after training  :  referent-recovery cos = {cF:+.3f}   "
          f"level-acc = {aF*100:5.1f}%")
    return model, test_data, (c0, a0, cF, aF)


# ============================================================================
# 5.  SAMANVAYA DEMONSTRATION
#     Show the doctrine in action: contradictory witnesses, one referent.
# ============================================================================

def demonstrate(model, test_data):
    print()
    print("=" * 72)
    print("SAMANVAYA IN ACTION  -  'tat tu samanvayat' (Brahma Sutra I.1.4)")
    print("=" * 72)
    X, r_star, Y = test_data[0]
    _, cache = model.forward(X, r_star, Y)
    Pr, A = cache["Pr"], cache["A"]
    names = ["paramarthika", "vyavaharika", "pratibhasika"]
    print("  Each witness is graded by reality-level, then re-leveled and weighted")
    print("  in the convergence. Note: illusory witnesses are correctly down-graded.\n")
    print("   witness | true level    | inferred level | converge wt")
    print("   --------+---------------+----------------+------------")
    for i in range(min(9, X.shape[0])):
        tl = names[int(Y[i].argmax())]
        il = names[int(Pr[i].argmax())]
        print(f"     {i:2d}    | {tl:13s} | {il:14s} | {A[i]:.3f}")
    rh = cache["r_hat"]
    cos = rh @ r_star / ((np.linalg.norm(rh)+1e-9)*(np.linalg.norm(r_star)+1e-9))
    print(f"\n  The many contradictory testimonies converge on ONE referent:")
    print(f"  recovered-referent vs hidden-truth cosine = {cos:+.3f}")
    print("  (1.000 = the witnesses have been fully harmonized onto Brahman)")

    # --- the convergence dial: one specification, three commentarial readings ---
    print()
    print("  THE CONVERGENCE DIAL  -  one spec, three architectures:")
    print("  (max convergence weight on a single witness, as conv_temp varies)")
    for temp, school in [(0.15, "Advaita    (collapse all -> one)"),
                         (1.00, "trained    (the learned default)"),
                         (6.00, "Visistadvaita (preserve the many)")]:
        _, ct = model.forward(X, r_star, Y, conv_temp=temp)
        print(f"    conv_temp={temp:4.2f}  max-weight={ct['A'].max():.3f}   {school}")
    print("  Low temp = hard non-dual collapse; high temp = structured blend.")


# ============================================================================
# 6.  SELF-TESTS
# ============================================================================

def self_tests(model, test_data, scores):
    print()
    print("=" * 72)
    print("SELF-TESTS")
    print("=" * 72)
    c0, a0, cF, aF = scores
    ok = True

    t1 = cF > 0.9
    print(f"  [{'PASS' if t1 else 'FAIL'}] referent recovery cos {cF:+.3f} > 0.90")
    ok &= t1

    t2 = aF > 0.85
    print(f"  [{'PASS' if t2 else 'FAIL'}] level accuracy {aF*100:.1f}% > 85%")
    ok &= t2

    t3 = cF > c0 + 0.2
    print(f"  [{'PASS' if t3 else 'FAIL'}] learning improved recovery "
          f"({c0:+.3f} -> {cF:+.3f})")
    ok &= t3

    # softmaxes are valid distributions
    X, r_star, Y = test_data[0]
    _, cache = model.forward(X, r_star, Y)
    t4 = np.allclose(cache["Pr"].sum(axis=1), 1.0) and \
        abs(cache["A"].sum() - 1.0) < 1e-9
    print(f"  [{'PASS' if t4 else 'FAIL'}] level & convergence weights are "
          f"valid probability distributions")
    ok &= t4

    # permutation invariance of the recovered referent (order of testimony
    # must not change the harmonized truth -- a samanvaya requirement)
    perm = RNG.permutation(X.shape[0])
    _, c2 = model.forward(X[perm], r_star, Y[perm])
    t5 = np.allclose(cache["r_hat"], c2["r_hat"], atol=1e-8)
    print(f"  [{'PASS' if t5 else 'FAIL'}] referent invariant to witness order "
          f"(harmonization is order-free)")
    ok &= t5

    print("-" * 72)
    print(f"  {'ALL SELF-TESTS PASSED' if ok else 'SOME TESTS FAILED'}")
    assert ok, "Self-tests failed"


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "#" * 72)
    print("#  0098 - THE SAMANVAYA RECONCILIATION ENGINE")
    print("#  Brahma Sutra: many contradictory testimonies, ONE referent")
    print("#" * 72 + "\n")

    gradient_check(conv_temp=1.0)
    gradient_check(conv_temp=0.5)   # verify the tunable-convergence knob is exact
    model, test_data, scores = train()
    demonstrate(model, test_data)
    self_tests(model, test_data, scores)

    print("\n" + "#" * 72)
    print("#  DONE.  The inquiry into Brahman has been mechanised and verified.")
    print("#" * 72 + "\n")

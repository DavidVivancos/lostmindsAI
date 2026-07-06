#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 THE CANONIC HOMEOSTAT  -  an Epicurean cognitive architecture (Figure 79)
  Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 4 Minds 61 - 80 Available on Amazon https://www.amazon.com/dp/B0H71JC95Q
Resume and Interactive Demos at https://artificiology.com/
Author: David Vivancos · Chapter 0079 · Epicurus of Samos
================================================================================
A from-scratch, pure-NumPy model that encodes the distinctive cognitive
signature of Epicurus of Samos (341-270 BCE), NOT the generic "atoms-in-the-void"
or "hedonic calculator" reading (those belong to Democritus, Lucretius, and
Bentham respectively).

The three ideas that are Epicurus's ALONE, and that this code makes mechanical:

  1. THE CANON (kanon = a mason's straightedge). Epicurus held that sensation
     (aisthesis) is incorrigible: the sense-impression is always a true report
     of what struck the organ. Error never lives in the sensor; it lives in the
     JUDGMENT (doxa) we ADD to the sensation (Letter to Herodotus 50, the famous
     bent-oar example). So this network has a fixed, un-learned sensory channel
     and confines ALL fallibility to a downstream judgment layer.

  2. THE METHOD OF MULTIPLE EXPLANATIONS (pleonachos tropos, Letter to Pythocles).
     Where the evidence cannot discriminate among rival causes (the "non-evident",
     adela: distant, ambiguous phenomena), the competent mind RETAINS every
     explanation the evidence does not contest, instead of forcing one. Forcing a
     single cause where evidence underdetermines is the very error that breeds
     fear. Here this is a gating layer whose posterior over rival "preconceptions"
     STAYS SPREAD on ambiguous input and COLLAPSES on clear input.

  3. THE SATIABLE GOAL (ataraxia / katastematic pleasure). The objective is not
     to maximize anything. "The magnitude of pleasure reaches its limit in the
     removal of all pain" (Principal Doctrines 18). Pleasure is the ABSENCE of
     disturbance; once disturbance is gone it can only be varied, not increased.
     So the loss here is a *satiable* disturbance functional with a tranquility
     floor (the tetrapharmakos clamp): below the floor the gradient vanishes and
     the system RESTS. A maximizer never rests; this one does. That is the whole
     point, and the ablation at the bottom proves it numerically.

A fourth, smaller idea -- the atomic SWERVE (clinamen / parenklisis), which
Epicurus added to Democritus so that atoms would not fall in perfectly
determined parallel lines -- appears as a tiny stochastic perturbation that
revives "dead" preconceptions during learning.

Conventions kept from the wider corpus: pure NumPy, a finite-difference gradient
check that MUST pass, a real training loop, and self-tests. The file executes
end to end; its printed output is pasted verbatim into the chapter.
================================================================================
"""

import numpy as np

RNG = np.random.default_rng(79)  # 79 = Epicurus's index in the corpus


# ==============================================================================
# 1. SENSATION (aisthesis): the incorrigible channel
# ------------------------------------------------------------------------------
# In Epicurus the sensation itself is never false; it is a true registration of
# the eidola (thin films) that strike the organ. We model this by feeding the
# raw input X straight through with NO learnable distortion. There are no
# weights here to "get the perception wrong" -- a deliberate architectural
# commitment, not an oversight. Everything that can err lives downstream.
# ==============================================================================
def sensation(X):
    """Identity: the sense-impression is taken as a true report. X is (N, d_in)."""
    return X


# ==============================================================================
# 2. THE CANONIC HOMEOSTAT
# ------------------------------------------------------------------------------
# Parameters (the only things that learn -- i.e. the only things that can err):
#   C : (K, d_in)            prototype centres = PRECONCEPTIONS (prolepseis),
#                            general patterns accreted from past sensation against
#                            which each new sensation is matched/recognised.
#   W : (K, d_out, d_in)     per-preconception linear JUDGMENT heads (doxa): the
#                            opinion each preconception ADDS to the sensation.
#   b : (K, d_out)           judgment biases.
#
# Forward pass:
#   dist2_{n,k} = || X_n - C_k ||^2                      (recognition distance)
#   a_{n,k}     = -(beta / (2 tau^2)) * dist2_{n,k}      (match logit)
#   w_{n,:}     = softmax_k(a_{n,:})                     (the pleonachos gate:
#                                                         posterior over rival
#                                                         preconceptions)
#   P_{n,k,:}   = W_k X_n + b_k                          (the K rival judgments)
#   yhat_n      = sum_k w_{n,k} P_{n,k,:}                (the mind's verdict)
#
# Where the K preconceptions match X similarly (ambiguous / adela), w stays
# SPREAD: several explanations are kept. Where one matches far better (clear),
# w COLLAPSES to it: a single explanation. This is pleonachos tropos, mechanised.
# ==============================================================================
class CanonicHomeostat:
    def __init__(self, d_in, d_out, K, tau=1.0, beta=1.0, floor=0.05, kappa=8.0):
        self.d_in, self.d_out, self.K = d_in, d_out, K
        self.tau = float(tau)        # recognition width of a preconception
        self.beta = float(beta)      # sharpness of the multiple-explanation gate
        self.floor = float(floor)    # epsilon: the tranquility floor (ataraxia)
        self.kappa = float(kappa)    # softness of the tetrapharmakos clamp
        # small inits; centres spread out so preconceptions start distinct
        self.C = RNG.normal(0, 1.0, size=(K, d_in))
        self.W = RNG.normal(0, 0.3, size=(K, d_out, d_in))
        self.b = np.zeros((K, d_out))

    # ----- packing helpers for the finite-difference gradient check -----------
    def get_params(self):
        return np.concatenate([self.C.ravel(), self.W.ravel(), self.b.ravel()])

    def set_params(self, vec):
        nC = self.K * self.d_in
        nW = self.K * self.d_out * self.d_in
        self.C = vec[:nC].reshape(self.K, self.d_in)
        self.W = vec[nC:nC + nW].reshape(self.K, self.d_out, self.d_in)
        self.b = vec[nC + nW:].reshape(self.K, self.d_out)

    # ----- forward, returning a cache for the backward pass --------------------
    def forward(self, X):
        N = X.shape[0]
        # recognition distances between each sensation and each preconception
        # dist2[n,k] = ||X_n - C_k||^2
        diff = X[:, None, :] - self.C[None, :, :]          # (N, K, d_in)
        dist2 = np.sum(diff * diff, axis=2)                 # (N, K)
        a = -(self.beta / (2.0 * self.tau ** 2)) * dist2    # (N, K) match logits
        a = a - a.max(axis=1, keepdims=True)                # stabilise softmax
        ea = np.exp(a)
        w = ea / np.sum(ea, axis=1, keepdims=True)          # (N, K) the gate
        # the K rival judgments P[n,k,:] = W_k X_n + b_k
        P = np.einsum('koj,nj->nko', self.W, X) + self.b[None, :, :]  # (N,K,d_out)
        yhat = np.einsum('nk,nko->no', w, P)                # (N, d_out)
        cache = dict(X=X, diff=diff, w=w, P=P, yhat=yhat, N=N)
        return yhat, cache

    # ----- the SATIABLE objective (disturbance with a tranquility floor) -------
    # surprise s_n = ||y_n - yhat_n||^2  (how far the evident contests the verdict)
    # D = mean_n  softplus(kappa (s_n - floor)) / kappa
    #   below the floor : softplus' ~ 0  -> NO drive (the mind is at peace)
    #   above the floor : ~ linear in (s - floor) -> drive to remove disturbance
    # The clamp is the tetrapharmakos: small, easily-borne disturbance is left
    # alone; the system relaxes to ataraxia instead of chasing s -> 0 forever.
    def disturbance(self, yhat, Y):
        s = np.sum((Y - yhat) ** 2, axis=1)                 # (N,) surprise per sample
        z = self.kappa * (s - self.floor)
        # numerically-stable softplus
        sp = np.where(z > 30, z, np.log1p(np.exp(np.clip(z, -60, 30)))) / self.kappa
        return float(np.mean(sp)), s

    # ----- analytic gradient of D w.r.t. (C, W, b) -----------------------------
    def backward(self, cache, Y):
        X, diff, w, P, yhat, N = (cache[k] for k in ('X', 'diff', 'w', 'P', 'yhat', 'N'))
        s = np.sum((Y - yhat) ** 2, axis=1)                 # (N,)
        # dD/ds_n = (1/N) * sigmoid(kappa (s - floor))   (derivative of softplus/kappa)
        g = (1.0 / N) * _sigmoid(self.kappa * (s - self.floor))   # (N,)
        # r_{n,o} = dD/dyhat_{n,o} = g_n * d s_n/dyhat = g_n * (-2)(Y-yhat)
        r = (2.0 * g[:, None]) * (yhat - Y)                 # (N, d_out)

        # ---- gradients into the judgment heads (doxa) ----
        # dD/dP_{n,k,o} = r_{n,o} * w_{n,k}
        # dD/dW_{k,o,j} = sum_n r_{n,o} w_{n,k} X_{n,j}
        rw = r[:, None, :] * w[:, :, None]                  # (N, K, d_out)
        gW = np.einsum('nko,nj->koj', rw, X)                # (K, d_out, d_in)
        gb = np.sum(rw, axis=0)                             # (K, d_out)

        # ---- gradients into the gate -> the preconception centres ----
        # dD/dw_{n,k} = sum_o r_{n,o} P_{n,k,o} =: q_{n,k}
        q = np.einsum('no,nko->nk', r, P)                   # (N, K)
        qbar = np.sum(q * w, axis=1, keepdims=True)         # (N, 1)
        # softmax jacobian: dD/da_{n,k} = w_{n,k} (q_{n,k} - qbar_n)
        e = w * (q - qbar)                                  # (N, K)
        # a_{n,k} = -(beta/2tau^2) dist2 ; da/dC_{k,j} = (beta/tau^2)(X_{n,j}-C_{k,j})
        # diff = X - C, so (X - C) = diff. dD/dC_{k,j} = sum_n e_{n,k}*(beta/tau^2)*diff
        coef = self.beta / (self.tau ** 2)
        gC = coef * np.einsum('nk,nkj->kj', e, diff)        # (K, d_in)

        return np.concatenate([gC.ravel(), gW.ravel(), gb.ravel()])

    # ----- convenience: loss + grad together -----
    def loss_and_grad(self, X, Y):
        yhat, cache = self.forward(X)
        D, _ = self.disturbance(yhat, Y)
        grad = self.backward(cache, Y)
        return D, grad

    # ----- diagnostics: how spread is the verdict? (entropy of the gate) -----
    def posterior_entropy(self, X):
        _, cache = self.forward(X)
        w = cache['w']
        return -np.sum(w * np.log(w + 1e-12), axis=1)       # (N,) nats

    # ----- diagnostics: the agent's own "am I still disturbed?" signal -----
    # = mean_n sigmoid(kappa (s_n - floor)). This is exactly the per-sample drive
    # that multiplies every gradient. Near 0 => the mind is at peace (ataraxia);
    # near 1 => every sensation still disturbs it. A maximizer (floor=0) can never
    # drive this to 0 in a noisy world; a satiable mind can.
    def drive_fraction(self, X, Y):
        yhat, _ = self.forward(X)
        s = np.sum((Y - yhat) ** 2, axis=1)
        return float(np.mean(_sigmoid(self.kappa * (s - self.floor))))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


# ==============================================================================
# 3. A WORLD WITH BOTH CLEAR AND UNDERDETERMINED PHENOMENA
# ------------------------------------------------------------------------------
# We build the "bent oar" situation explicitly. Two latent causal regimes, each
# with its own linear law. Over most of the input plane only ONE regime could
# have produced the data (clear / near phenomena -> single explanation). But in
# a central overlap band BOTH regimes are equally consistent (adela / distant
# phenomena -> multiple explanations). For overlap points the target is the mean
# of the two laws, so neither rival judgment is contested more than the other:
# the evidence genuinely underdetermines the cause.
# ==============================================================================
def make_world(n=1200, seed=7, noise=0.0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3, 3, size=(n, 2))
    # two rival linear laws (the two "causes")
    A0 = np.array([[1.2, -0.3], [0.4, 0.9]]);  d0 = np.array([0.5, -0.2])
    A1 = np.array([[-0.6, 1.0], [1.1, 0.2]]);  d1 = np.array([-0.4, 0.6])
    y0 = X @ A0.T + d0
    y1 = X @ A1.T + d1
    # regime is decided by x0: left -> law 0, right -> law 1, with a soft overlap
    # band around x0 = 0 where both are equally consistent (target = average).
    band = 0.6
    overlap = np.abs(X[:, 0]) < band
    Y = np.where((X[:, 0] < 0)[:, None], y0, y1)
    Y[overlap] = 0.5 * (y0[overlap] + y1[overlap])  # underdetermined region
    if noise > 0:
        # irreducible observation noise: surprise can never fall below ~2*noise^2.
        # This is the disturbance the world simply will not let a mind remove.
        Y = Y + rng.normal(0, noise, size=Y.shape)
    return X.astype(float), Y.astype(float), overlap


# ==============================================================================
# 4. TRAINING with the SWERVE (clinamen)
# ------------------------------------------------------------------------------
# Plain gradient descent on the satiable disturbance, plus an Epicurean swerve:
# any preconception whose total responsibility (sum of gate weight) falls near
# zero -- a "dead" prototype, an atom falling in a perfectly determined line --
# is nudged by a tiny random parenklisis toward a live data point, so it can
# re-enter the play of explanations. The swerve is applied OUTSIDE the gradient
# (it never touches the gradient check) -- exactly as the swerve sits outside
# the deterministic fall of the atoms.
# ==============================================================================
def train(model, X, Y, epochs=400, lr=0.05, swerve=True, swerve_eps=0.15, verbose=False):
    hist = {'D': [], 'gradnorm': [], 'live': []}
    for t in range(epochs):
        D, g = model.loss_and_grad(X, Y)
        vec = model.get_params() - lr * g
        model.set_params(vec)

        # ---- the clinamen: revive dead preconceptions ----
        _, cache = model.forward(X)
        resp = cache['w'].sum(axis=0)                      # responsibility per prototype
        live = int(np.sum(resp > 1.0))                     # "alive" if it wins >~1 sample
        if swerve:
            dead = np.where(resp < 1.0)[0]
            for k in dead:
                tgt = X[RNG.integers(0, X.shape[0])]
                model.C[k] = tgt + swerve_eps * RNG.normal(size=model.d_in)

        hist['D'].append(D)
        hist['gradnorm'].append(float(np.linalg.norm(g)))
        hist['live'].append(live)
        if verbose and (t % 80 == 0 or t == epochs - 1):
            print(f"    epoch {t:4d} | disturbance {D:.5f} | "
                  f"|grad| {hist['gradnorm'][-1]:.5f} | live preconceptions {live}/{model.K}")
    return hist


# ==============================================================================
# 5. SELF-TESTS  (each prints PASS/FAIL; the file fails loudly if any breaks)
# ==============================================================================
def test_gradient_check():
    """Finite-difference check of the analytic gradient. MANDATORY."""
    print("[1] Finite-difference gradient check")
    m = CanonicHomeostat(d_in=2, d_out=2, K=4, tau=1.1, beta=1.3, floor=0.05, kappa=8.0)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(16, 2)); Y = rng.normal(size=(16, 2))

    _, g_analytic = m.loss_and_grad(X, Y)
    p0 = m.get_params().copy()
    eps = 1e-6
    g_num = np.zeros_like(p0)
    for i in range(len(p0)):
        pp = p0.copy(); pp[i] += eps; m.set_params(pp)
        Dp, _ = m.disturbance(*((lambda yh: (yh, Y))(m.forward(X)[0])))
        pm = p0.copy(); pm[i] -= eps; m.set_params(pm)
        Dm, _ = m.disturbance(*((lambda yh: (yh, Y))(m.forward(X)[0])))
        g_num[i] = (Dp - Dm) / (2 * eps)
    m.set_params(p0)

    rel = np.linalg.norm(g_analytic - g_num) / (np.linalg.norm(g_analytic) + np.linalg.norm(g_num) + 1e-12)
    print(f"    params checked      : {len(p0)}")
    print(f"    relative error      : {rel:.2e}")
    ok = rel < 1e-6
    print(f"    -> {'PASS' if ok else 'FAIL'} (analytic gradient matches numeric)")
    assert ok, "gradient check failed"
    return ok


def test_satiation_vs_maximizer():
    """The satiable mind RESTS; the maximizer never does (the Epicurean claim).

    We give the world irreducible observation noise (disturbance no mind can
    remove). The satiable agent's tranquility floor sits ABOVE that noise, so
    once it has explained the real structure it stops driving -- ataraxia. The
    maximizer (floor = 0) keeps chasing the noise forever, so its gradient stays
    large. This is Epicurus's whole point about a non-maximizing terminal goal.
    """
    print("[2] Satiation: ataraxia vs unbounded maximization")
    noise = 0.30
    X, Y, _ = make_world(n=900, seed=3, noise=noise)
    irreducible = 2.0 * noise ** 2  # expected per-sample surprise from noise alone
    print(f"    irreducible disturbance (noise floor) ~ {irreducible:.3f} per sample")

    # satiable: floor ABOVE the noise -> accept what cannot be removed
    sat = CanonicHomeostat(2, 2, K=6, tau=0.9, beta=2.0, floor=0.30, kappa=10.0)
    h_sat = train(sat, X, Y, epochs=500, lr=0.06, swerve=True)

    # maximizer: floor = 0 -> chase surprise toward 0, including the noise
    mx = CanonicHomeostat(2, 2, K=6, tau=0.9, beta=2.0, floor=0.0, kappa=10.0)
    mx.C = RNG.normal(0, 1, size=mx.C.shape)  # fresh start
    h_mx = train(mx, X, Y, epochs=500, lr=0.06, swerve=True)

    rest_sat = np.mean(h_sat['gradnorm'][-20:])
    rest_mx = np.mean(h_mx['gradnorm'][-20:])
    drive_sat = sat.drive_fraction(X, Y)
    drive_mx = mx.drive_fraction(X, Y)
    print(f"    satiable  drive fraction (am-I-disturbed?) : {drive_sat:.3f}  (at peace)")
    print(f"    maximizer drive fraction (am-I-disturbed?) : {drive_mx:.3f}  (still driven)")
    print(f"    satiable  final |grad| (last 20)           : {rest_sat:.5f}")
    print(f"    maximizer final |grad| (last 20)           : {rest_mx:.5f}")
    ratio = drive_mx / (drive_sat + 1e-9)
    print(f"    maximizer / satiable drive ratio           : {ratio:.1f}x")
    ok = drive_sat < drive_mx and ratio > 2.0
    print(f"    -> {'PASS' if ok else 'FAIL'} (the satiable agent comes to rest, the maximizer does not)")
    assert ok
    return ok


def test_multiple_explanations():
    """Posterior stays SPREAD where evidence underdetermines, COLLAPSES where clear."""
    print("[3] Multiple explanations: pleonachos tropos")
    X, Y, overlap = make_world(n=1200, seed=11)
    m = CanonicHomeostat(2, 2, K=6, tau=0.8, beta=2.5, floor=0.06, kappa=10.0)
    train(m, X, Y, epochs=450, lr=0.06, swerve=True)

    H = m.posterior_entropy(X)
    H_amb = float(np.mean(H[overlap]))     # adela: distant/ambiguous phenomena
    H_clear = float(np.mean(H[~overlap]))  # near/clear phenomena
    print(f"    mean gate entropy, ambiguous band : {H_amb:.3f} nats (explanations kept)")
    print(f"    mean gate entropy, clear region   : {H_clear:.3f} nats (one explanation)")
    print(f"    spread ratio                      : {H_amb / (H_clear + 1e-9):.2f}x")
    ok = H_amb > H_clear * 1.3
    print(f"    -> {'PASS' if ok else 'FAIL'} (the mind holds rival causes only where the evidence cannot decide)")
    assert ok
    return ok


def test_swerve_revives_preconceptions():
    """The clinamen keeps more preconceptions alive than a determinist fall.

    We deliberately STARVE the prototypes: a sharp gate (high beta, small tau)
    and an initial cluster of centres jammed into one corner, far from most of
    the data. Without the swerve, the starved prototypes never win a sample and
    stay dead -- atoms falling forever in perfectly parallel lines. The swerve
    nudges each dead prototype onto a live sensation, letting new compounds form.
    """
    print("[4] The swerve (clinamen) revives dead preconceptions")
    X, Y, _ = make_world(n=900, seed=5)
    K = 14
    # starved init: all centres clustered in one corner, far from the data spread
    cluster = np.array([2.6, 2.6]) + 0.15 * RNG.normal(size=(K, 2))
    m_no = CanonicHomeostat(2, 2, K=K, tau=0.45, beta=6.0, floor=0.06, kappa=10.0)
    m_sw = CanonicHomeostat(2, 2, K=K, tau=0.45, beta=6.0, floor=0.06, kappa=10.0)
    m_no.C = cluster.copy(); m_sw.C = cluster.copy()
    m_sw.W = m_no.W.copy(); m_sw.b = m_no.b.copy()
    h_no = train(m_no, X, Y, epochs=300, lr=0.06, swerve=False)
    h_sw = train(m_sw, X, Y, epochs=300, lr=0.06, swerve=True)
    live_no = h_no['live'][-1]; live_sw = h_sw['live'][-1]
    print(f"    live preconceptions, no swerve : {live_no}/{K} (atoms fall in parallel lines)")
    print(f"    live preconceptions, w/ swerve : {live_sw}/{K} (the swerve lets new compounds form)")
    ok = live_sw > live_no
    print(f"    -> {'PASS' if ok else 'FAIL'} (the swerve revives explanations a determinist fall would lose)")
    assert ok
    return ok


# ==============================================================================
# 6. MAIN
# ==============================================================================
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    print("=" * 78)
    print(" THE CANONIC HOMEOSTAT  -  Epicurus (Figure 79)")
    print(" incorrigible sensation | fallible judgment | multiple explanations |")
    print(" a satiable tranquility objective | the atomic swerve")
    print("=" * 78)

    print("\n--- A worked training run on a world of clear + underdetermined causes ---")
    X, Y, overlap = make_world(n=1000, seed=2)
    model = CanonicHomeostat(d_in=2, d_out=2, K=6, tau=0.85, beta=2.2, floor=0.07, kappa=10.0)
    print(f"  data: {X.shape[0]} sensations, {int(overlap.sum())} in the underdetermined band")
    hist = train(model, X, Y, epochs=400, lr=0.06, swerve=True, verbose=True)
    print(f"  initial disturbance : {hist['D'][0]:.5f}")
    print(f"  final   disturbance : {hist['D'][-1]:.5f}   (relaxed toward the tranquility floor {model.floor})")
    print(f"  final   |grad|      : {hist['gradnorm'][-1]:.5f}   (the mind has come to rest)")

    print("\n--- Self-tests ---")
    results = []
    results.append(("gradient check", test_gradient_check()))
    print()
    results.append(("satiation vs maximizer", test_satiation_vs_maximizer()))
    print()
    results.append(("multiple explanations", test_multiple_explanations()))
    print()
    results.append(("swerve revives", test_swerve_revives_preconceptions()))

    print("\n" + "=" * 78)
    allok = all(r for _, r in results)
    for name, r in results:
        print(f"  {'PASS' if r else 'FAIL'}  -  {name}")
    print("=" * 78)
    print("ALL TESTS PASSED" if allok else "SOME TESTS FAILED")

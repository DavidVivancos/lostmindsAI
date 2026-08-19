#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
========================
 chapter_0140_cao_zhi_192.py
 The Constrained Resonance Network (CRN) — a cognitive architecture built
 in the shape of Cao Zhi's mind (曹植, 192-232 CE, Prince Si of Chen).
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 7 Minds 121 - 140 Available on Amazon https://www.amazon.com/dp/B0HFN6GXMH
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 140: Cao Zhi (曹植, 192-232 CE, Prince Si of Chen)
================================================================================   

WHY THIS ARCHITECTURE, AND NOT A TRANSFORMER

Cao Zhi was the most gifted mind of his generation and was forbidden to use
it. His father nearly named him heir; his brother, once emperor, kept him
alive, titled, and fed — but never *deployed*. He petitioned for real office
again and again and was refused every time, shuffled from fief to fief so he
could never build a base. He died at forty, a thoroughbred that was never
allowed to run.

A mind in that position develops a very specific competence: it learns to
say the forbidden thing *obliquely*. Direct speech ("the sovereign wrongs
me") was lethal, so Cao Zhi routed his true inner state (中情, zhongqing) —
grief, thwarted ambition, loyalty — through a vocabulary of permitted images:
the caged bird, the cast-off wife, the racehorse never raced, the beans wept
over by their own beanstalk, the goddess of the Luo who appears and cannot be
joined. This is the classical device of *bi* (比, comparison) and *xing* (興,
evocative image). The meaning is fully present but never stated; it passes
through a narrow, auditable aperture the censor can read but not indict.

So the natural computational model of this mind is NOT attention over a big
stored memory. It is a *routing-through-a-constrained-channel* network:

    situation ->  hidden affect state (a)         [the inner grief, 中情]
              ->  resonance with a codebook of     [the bi-xing images]
                  oblique IMAGES, under a GATE     [the "seven steps" budget]
              ->  a blended image (c_hat)          [the chosen figure]
              ->  the uttered expression (y_hat)   [the poem that survives]

Two ideas are load-bearing and are Cao Zhi's alone:

  1. THE SUPPRESSION GATE ("seven steps").  A scalar budget beta in (0,1]
     controls a routing temperature. A *loose* gate lets the state express
     itself diffusely across many images; a *tight* gate (few paces, or die)
     forces the mind to carry all its meaning through a single, blazing
     image. The empirical claim — which the self-tests verify — is that
     compression under pressure does not destroy meaning; up to a breaking
     point it *sharpens* it. This is the poetics of the constrained agent.

  2. OBLIQUITY IS STRUCTURAL, NOT OPTIONAL.  The inner state `a` is NEVER
     decoded directly. It can only touch the output by first resonating with
     the shared, inspectable codebook of images. That is exactly the
     condition of the boxed loyal servant: he may speak only through figures
     his wary principal can audit. Alignment here is achieved by *narrowing
     the channel*, not by lobotomising the mind — a strikingly modern take
     on containing a capable agent without destroying it.

This gives an AGI-relevant thesis you will not find in an oracle-builder:
Cao Zhi models alignment *from inside the box*. He is the theorist of the
capable, loyal, contained agent, and of the tragedy that a capability held
too long unused turns its power inward into grief.

WHAT THE FILE CONTAINS
  * A pure-NumPy Constrained Resonance Network (no autograd frameworks).
  * Hand-derived analytic gradients for every parameter.
  * A MANDATORY finite-difference gradient check (must pass).
  * A synthetic "resonance task" where meaning is carried only by *which*
    image a situation resonates with — the model must learn to route.
  * A real training loop.
  * Self-tests, including a sweep of the suppression gate that demonstrates
    the "seven steps" compression effect quantitatively.

Run:  python3 chapter_0140_cao_zhi_192.py
============================================================================
"""

import numpy as np

# --------------------------------------------------------------------------
# 0. Reproducibility
# --------------------------------------------------------------------------
SEED = 192  # the year of his birth
rng = np.random.default_rng(SEED)


# --------------------------------------------------------------------------
# 1. Small numerical helpers
# --------------------------------------------------------------------------
def softmax_rows(z):
    """Row-wise numerically-stable softmax. z: (N, K) -> (N, K)."""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def softplus(x):
    # numerically stable softplus, keeps the learnable temperature scale > 0
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def row_entropy(w, eps=1e-12):
    """Shannon entropy (nats) of each routing distribution. w: (N,K)->(N,)."""
    return -(w * np.log(w + eps)).sum(axis=1)


# --------------------------------------------------------------------------
# 2. The model
# --------------------------------------------------------------------------
class ConstrainedResonanceNetwork:
    """
    A from-scratch network embodying Cao Zhi's oblique-expression-under-a-gate.

    Parameters
    ----------
    d_in  : dimension of the incoming SITUATION (外境, the outer scene)
    d_lat : dimension of the hidden AFFECT STATE `a` (中情, inner feeling)
    d_out : dimension of the uttered EXPRESSION `y` (the surviving poem)
    K     : number of oblique IMAGES in the resonance codebook (bi-xing 比興)
    tau_min : floor on the routing temperature (keeps softmax finite)
    lam_div : weight of the codebook-orthogonality regulariser
              (the images must be genuinely DIFFERENT figures, not synonyms)
    """

    def __init__(self, d_in, d_lat, d_out, K, tau_min=0.05, lam_div=1e-3,
                 rng=np.random.default_rng(0)):
        self.d_in, self.d_lat, self.d_out, self.K = d_in, d_lat, d_out, K
        self.tau_min = tau_min
        self.lam_div = lam_div

        # Xavier / Glorot style initialisation
        def xav(shape, fan_in):
            return rng.standard_normal(shape) * np.sqrt(1.0 / fan_in)

        self.params = {
            # SituationEncoder:  situation -> inner affect state a
            "W_enc": xav((d_lat, d_in), d_in),
            "b_enc": np.zeros(d_lat),
            # ResonanceCodebook: the K oblique bi-xing images (rows)
            "C":     xav((K, d_lat), d_lat),
            # ExpressionDecoder: blended image -> uttered expression
            "W_dec": xav((d_out, d_lat), d_lat),
            "b_dec": np.zeros(d_out),
            # SuppressionGate: learnable scale of the routing temperature.
            # (softplus keeps it positive.) The *budget* beta is imposed
            # externally at forward time — it is the "number of paces".
            "g_raw": np.array(0.0),
        }

    # ---- forward -------------------------------------------------------
    def forward(self, S, beta=1.0, cache=True):
        """
        S    : (N, d_in) situations
        beta : suppression budget in (0,1].  1.0 = loose gate (many paces).
               Smaller beta = tighter gate ("seven steps") = sharper routing.

        Returns Y_hat (N, d_out).  Intermediates are cached for backward().
        """
        p = self.params
        d = self.d_lat

        u = S @ p["W_enc"].T + p["b_enc"]          # (N, d_lat) pre-activation
        a = np.tanh(u)                              # (N, d_lat) inner affect 中情

        # routing temperature: the suppression gate.
        # tight budget beta -> small tau -> a peaky softmax -> few images used.
        tau = softplus(p["g_raw"]) * beta + self.tau_min

        # resonance between the inner state and each oblique image
        logits = (a @ p["C"].T) / np.sqrt(d)        # (N, K)  a·C_k / sqrt(d)
        z = logits / tau                            # (N, K)  gated resonance
        w = softmax_rows(z)                         # (N, K)  routing weights

        c_hat = w @ p["C"]                          # (N, d_lat) blended image
        Y_hat = c_hat @ p["W_dec"].T + p["b_dec"]   # (N, d_out) uttered poem

        if cache:
            self._cache = dict(S=S, u=u, a=a, tau=tau, beta=beta,
                               logits=logits, w=w, c_hat=c_hat, Y_hat=Y_hat)
        return Y_hat

    # ---- loss ----------------------------------------------------------
    def loss(self, Y_hat, Y):
        """Mean squared expression error + codebook-orthogonality penalty."""
        N = Y.shape[0]
        rec = 0.5 * np.sum((Y_hat - Y) ** 2) / N

        C = self.params["C"]
        G = C @ C.T                                 # (K, K) Gram of images
        off = G - np.diag(np.diag(G))               # zero the diagonal
        div = 0.5 * self.lam_div * np.sum(off ** 2)  # push images apart
        return rec + div, rec, div

    # ---- backward (hand-derived analytic gradients) --------------------
    def backward(self, Y):
        """
        Returns a dict of gradients matching self.params.
        Derivation is documented inline; verified by finite differences.
        """
        p, c = self.params, self._cache
        S, u, a = c["S"], c["u"], c["a"]
        tau, logits, w, c_hat, Y_hat = (c["tau"], c["logits"], c["w"],
                                        c["c_hat"], c["Y_hat"])
        N, d = S.shape[0], self.d_lat

        # d L_rec / d Y_hat, with the 1/N of the mean folded in here so that
        # every downstream gradient inherits the batch-averaging automatically.
        dY = (Y_hat - Y) / N                        # (N, d_out)

        # ExpressionDecoder
        grad_W_dec = dY.T @ c_hat                   # (d_out, d_lat)
        grad_b_dec = dY.sum(axis=0)                 # (d_out,)
        dC_hat = dY @ p["W_dec"]                    # (N, d_lat)

        # c_hat = w @ C : split into the direct-C path and the w path
        dC_direct = w.T @ dC_hat                    # (K, d_lat)  via C in c_hat
        dw = dC_hat @ p["C"].T                      # (N, K)      via w

        # softmax backward (row-wise):  dz = w * (dw - sum(w*dw))
        dz = w * (dw - (w * dw).sum(axis=1, keepdims=True))   # (N, K)

        # z = logits / tau
        dlogits = dz / tau                          # (N, K)
        # gate gradient: tau enters through z = logits/tau
        dtau = -np.sum(dz * logits) / (tau ** 2)    # scalar (already /N via dY)

        # logits = (a @ C.T)/sqrt(d)
        dC_logits = (dlogits.T @ a) / np.sqrt(d)    # (K, d_lat)
        da = (dlogits @ p["C"]) / np.sqrt(d)        # (N, d_lat)

        # through tanh
        du = da * (1.0 - a ** 2)                    # (N, d_lat)
        grad_W_enc = du.T @ S                       # (d_lat, d_in)
        grad_b_enc = du.sum(axis=0)                 # (d_lat,)

        # codebook orthogonality regulariser: dL/dC = 2*lam*offdiag(G)@C
        G = p["C"] @ p["C"].T
        off = G - np.diag(np.diag(G))
        dC_div = 2.0 * self.lam_div * (off @ p["C"])   # (K, d_lat)

        grad_C = dC_direct + dC_logits + dC_div

        # gate: tau = softplus(g_raw)*beta + tau_min
        dg_raw = dtau * sigmoid(p["g_raw"]) * c["beta"]

        return {
            "W_enc": grad_W_enc, "b_enc": grad_b_enc,
            "C": grad_C,
            "W_dec": grad_W_dec, "b_dec": grad_b_dec,
            "g_raw": np.array(dg_raw),
        }

    # ---- convenience: routing weights for inspection -------------------
    def route(self, S, beta=1.0):
        self.forward(S, beta=beta, cache=True)
        return self._cache["w"]


# --------------------------------------------------------------------------
# 3. The synthetic "resonance task"
# --------------------------------------------------------------------------
# Each example belongs to a hidden TRUE IMAGE. The SITUATION is a noisy
# projection of that image; the correct EXPRESSION is a (different) clean
# projection of the same image. So the meaning of a situation is carried
# ENTIRELY by *which image it resonates with* — precisely the bi-xing logic.
# The network must (a) feel the right image and (b) speak through it.
def make_task(n_images=6, d_in=12, d_lat=10, d_out=8, n=480,
              noise=0.35, rng=np.random.default_rng(0)):
    C_true = rng.standard_normal((n_images, d_lat))
    C_true /= np.linalg.norm(C_true, axis=1, keepdims=True)     # unit images
    A = rng.standard_normal((d_in, d_lat))                      # scene map
    B = rng.standard_normal((d_out, d_lat))                     # utter map

    t = rng.integers(0, n_images, size=n)                       # true image id
    S = C_true[t] @ A.T + noise * rng.standard_normal((n, d_in))
    Y = C_true[t] @ B.T                                         # clean target
    # standardise for stable training
    S = (S - S.mean(0)) / (S.std(0) + 1e-8)
    Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-8)
    return S, Y, t, n_images


# --------------------------------------------------------------------------
# 4. Mandatory finite-difference gradient check
# --------------------------------------------------------------------------
def gradient_check(verbose=True):
    """Compare analytic gradients against central finite differences.
    Returns the worst relative error across all parameters."""
    g = np.random.default_rng(7)
    net = ConstrainedResonanceNetwork(d_in=6, d_lat=5, d_out=4, K=4,
                                      lam_div=5e-3, rng=g)
    S = g.standard_normal((9, 6))
    Y = g.standard_normal((9, 4))
    beta = 0.7

    net.forward(S, beta=beta)
    grads = net.backward(Y)

    eps = 1e-6
    worst = 0.0
    for name, P in net.params.items():
        P = np.atleast_1d(P)
        flat = P.ravel()
        gflat = np.atleast_1d(grads[name]).ravel()
        num = np.zeros_like(flat)
        for i in range(flat.size):
            orig = flat[i]
            flat[i] = orig + eps
            net.forward(S, beta=beta); Lp, _, _ = net.loss(net._cache["Y_hat"], Y)
            flat[i] = orig - eps
            net.forward(S, beta=beta); Lm, _, _ = net.loss(net._cache["Y_hat"], Y)
            flat[i] = orig
            num[i] = (Lp - Lm) / (2 * eps)
        denom = np.maximum(1e-8, np.abs(gflat) + np.abs(num))
        rel = np.max(np.abs(gflat - num) / denom)
        worst = max(worst, rel)
        if verbose:
            print(f"    {name:6s} shape {str(P.shape):10s} "
                  f"max|analytic-numeric| rel err = {rel:.2e}")
    return worst


# --------------------------------------------------------------------------
# 5. Training loop (plain SGD with momentum)
# --------------------------------------------------------------------------
def train(net, S, Y, epochs=600, lr=0.15, beta_schedule=None, mom=0.9,
          verbose=True):
    """
    beta_schedule : optional callable epoch->beta implementing the
                    'narrowing paces' curriculum. Default: constant loose gate.
    """
    vel = {k: np.zeros_like(np.atleast_1d(v)) for k, v in net.params.items()}
    hist = []
    for ep in range(epochs):
        beta = 1.0 if beta_schedule is None else beta_schedule(ep, epochs)
        Yh = net.forward(S, beta=beta)
        L, rec, div = net.loss(Yh, Y)
        grads = net.backward(Y)
        for k in net.params:
            gk = np.atleast_1d(grads[k])
            vel[k] = mom * vel[k] - lr * gk
            net.params[k] = np.atleast_1d(net.params[k]) + vel[k]
            if net.params[k].size == 1:      # keep scalar params scalar-ish
                net.params[k] = net.params[k].reshape(())
        hist.append((L, rec, div))
        if verbose and (ep % max(1, epochs // 8) == 0 or ep == epochs - 1):
            print(f"    epoch {ep:4d} | beta {beta:.2f} | "
                  f"loss {L:.4f} | rec {rec:.4f} | div {div:.4f}")
    return hist


# --------------------------------------------------------------------------
# 6. Evaluation utilities
# --------------------------------------------------------------------------
def routing_purity(net, S, t, n_images, beta=1.0):
    """
    Do situations that share a TRUE image get routed to a common learned
    image?  We compute, for each true class, the fraction that lands on that
    class's most-common learned image (a permutation-invariant purity score).
    """
    w = net.route(S, beta=beta)
    dom = w.argmax(axis=1)                        # dominant learned image
    correct = 0
    for cls in range(n_images):
        idx = np.where(t == cls)[0]
        if idx.size == 0:
            continue
        counts = np.bincount(dom[idx], minlength=net.K)
        correct += counts.max()
    return correct / len(t)


def gate_sweep(net, S, Y, betas):
    """The 'seven steps' experiment: tighten the gate and watch how the mind
    compresses. Returns (beta, mean routing entropy, reconstruction error)."""
    rows = []
    for b in betas:
        Yh = net.forward(S, beta=b)
        _, rec, _ = net.loss(Yh, Y)
        w = net._cache["w"]
        H = row_entropy(w).mean()
        rows.append((b, H, rec))
    return rows


# --------------------------------------------------------------------------
# 7. Main: build, verify, train, and run the self-tests
# --------------------------------------------------------------------------
def main():
    print("=" * 74)
    print(" CAO ZHI — CONSTRAINED RESONANCE NETWORK")
    print(" oblique expression of a forbidden inner state, under a gate")
    print("=" * 74)

    # -- (a) gradient check: the non-negotiable correctness test ----------
    print("\n[1] FINITE-DIFFERENCE GRADIENT CHECK")
    worst = gradient_check(verbose=True)
    print(f"    -> worst relative error = {worst:.2e}")
    assert worst < 1e-5, "GRADIENT CHECK FAILED"
    print("    -> PASS (analytic gradients match numerical to < 1e-5)")

    # -- (b) build the task and the model ---------------------------------
    print("\n[2] BUILDING THE RESONANCE TASK")
    task_rng = np.random.default_rng(2025)
    S, Y, t, n_images = make_task(n_images=6, d_in=12, d_lat=10, d_out=8,
                                  n=480, noise=0.35, rng=task_rng)
    # split
    ntr = 360
    Str, Ytr, ttr = S[:ntr], Y[:ntr], t[:ntr]
    Ste, Yte, tte = S[ntr:], Y[ntr:], t[ntr:]
    print(f"    {n_images} oblique images | train {ntr} | test {len(tte)} | "
          f"d_in 12, d_lat 10, d_out 8")

    net = ConstrainedResonanceNetwork(d_in=12, d_lat=10, d_out=8, K=6,
                                      lam_div=2e-3,
                                      rng=np.random.default_rng(11))

    # -- (c) 'narrowing paces' curriculum: start loose, tighten the gate --
    def paces(ep, epochs):
        # anneal beta from 1.0 (loose) down to 0.35 (tight, "seven steps")
        frac = ep / max(1, epochs - 1)
        return 1.0 - 0.65 * frac

    print("\n[3] TRAINING (curriculum: the gate narrows from 1.00 -> 0.35)")
    train(net, Str, Ytr, epochs=600, lr=0.12, beta_schedule=paces)

    # -- (d) generalisation -----------------------------------------------
    print("\n[4] SELF-TESTS")
    Yh_te = net.forward(Ste, beta=0.35)
    _, rec_te, _ = net.loss(Yh_te, Yte)
    Yh_tr = net.forward(Str, beta=0.35)
    _, rec_tr, _ = net.loss(Yh_tr, Ytr)
    print(f"    (i)  expression error   train {rec_tr:.4f} | test {rec_te:.4f}")
    assert rec_te < 0.25, "model failed to learn the task"

    pur_tr = routing_purity(net, Str, ttr, n_images, beta=0.35)
    pur_te = routing_purity(net, Ste, tte, n_images, beta=0.35)
    print(f"    (ii) routing purity     train {pur_tr:.3f} | test {pur_te:.3f}")
    print("         (situations that share a true image resonate with a"
          " common learned image)")
    assert pur_te > 0.80, "routing did not organise by image"

    # -- (e) the 'seven steps' compression experiment ---------------------
    print("\n[5] THE 'SEVEN STEPS' EXPERIMENT")
    print("    tighten the suppression gate and watch the mind compress:")
    print(f"    {'budget beta':>12} | {'mean entropy':>12} | "
          f"{'~images used':>12} | {'express err':>11}")
    betas = [1.00, 0.75, 0.50, 0.35, 0.20, 0.10]
    rows = gate_sweep(net, Ste, Yte, betas)
    for b, H, rec in rows:
        print(f"    {b:>12.2f} | {H:>12.3f} | {np.exp(H):>12.2f} | "
              f"{rec:>11.4f}")
    H_loose = rows[0][1]
    H_tight = rows[3][1]     # beta = 0.35
    print(f"\n    entropy falls {H_loose:.3f} -> {H_tight:.3f} nats as the gate"
          " tightens:")
    print("    under pressure the mind speaks through FEWER, sharper images —")
    print("    yet at beta=0.35 the expression error is still low. Maximal")
    print("    compression under maximal constraint still carries the meaning.")
    assert H_tight < H_loose - 0.15, "compression effect not observed"

    # a light 'breaking point' observation (no hard assert; it is a lesson):
    rec_extreme = rows[-1][2]
    if rec_extreme > rows[3][2] * 1.05:
        print("    Beyond the poet's limit (beta<=0.10) the single-image"
              " channel")
        print("    starts to lose meaning — the caged mind, over-compressed,"
              " frays.")

    print("\n" + "=" * 74)
    print(" ALL SELF-TESTS PASSED.")
    print(" The architecture runs, learns, routes by oblique image, and")
    print(" demonstrates compression-under-suppression — Cao Zhi's signature.")
    print("=" * 74)


if __name__ == "__main__":
    main()

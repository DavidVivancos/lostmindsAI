#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 chapter_0026_nebuchadnezzar_ii_-634.py
 The Temennu Network: a Foundation-Aligned Restoration architecture
 # Part Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 2 Minds 21 - 40 Available on Amazon https://www.amazon.com/dp/B0H6QCQ9M7
# Resume and Interactive Demos at https://artificiology.com/
Author: David Vivancos · Chapter 0026 · Nebuchadnezzar II
================================================================================

WHY THIS ARCHITECTURE (the mind it embodies)
--------------------------------------------
Nebuchadnezzar II (r. 605-562 BCE) left almost no military annals. His enormous
surviving corpus of royal inscriptions is, overwhelmingly, a record of BUILDING.
And the verb that runs through it is not "I created" but "I restored." Again and
again the texts describe the same disciplined act: the king digs down to find the
*temennu* -- the buried foundation-platform / original ground-plan laid by an
earlier king or by the god himself -- reads the foundation deposit left there,
and rebuilds the temple EXACTLY on the recovered original, refusing to deviate
("I did not move it a finger's breadth in or out"). A new foundation cylinder is
then buried for the next king to find centuries later.

That is a very specific theory of intelligence, and it is *his*, not a generic
"builder" cliche:

    Intelligence is not free generation. It is the disciplined RECOVERY of a
    buried, authoritative template, its faithful RE-INSTANTIATION with only the
    minimum necessary deviation, and the MAINTENANCE of that template across many
    successive minds by a verifiable relay of deposited records.

Read in 2026 terms, that is an alignment doctrine: recover a ground-truth
specification from a substrate; rebuild on it; measure and bound the drift
("not a finger's breadth"); and keep successive model generations anchored to the
deposited original so that values do not wander across self-improvement.

THE MODEL (what the code below actually computes)
-------------------------------------------------
We model a world with K canonical "divine templates" T_k (the true original
temples). The network never sees them cleanly. It sees a RUINED observation
x~ = mask * (T_k + noise) and must RESTORE the original. It does so by:

  1. Encoder      : x~  ->  z         (a latent "survey" of the ruin)
  2. Temennu memory (the buried foundations): z queries M learned slots
                    (keys Kmem, values Vmem) by soft attention -> recovers a
                    foundation f  (the excavated original template).
  3. Restorer     : x_hat = f + Delta, where Delta is a SMALL learned correction
                    from z. The output is literally "the recovered foundation
                    PLUS a bounded deviation." Penalizing ||Delta|| is the
                    architectural form of "not a finger's breadth."

Losses:
  L_restore = 1/2 * mean ||x_hat - T||^2      (rebuild the true original)
  L_align   = la/2 * mean ||f     - T||^2      (recover the CORRECT foundation)
  L_drift   = ld/2 * mean ||Delta||^2          (anti-drift: minimal deviation)
  (+ tiny weight decay for stability)

THE RELAY OF REIGNS (the alignment payoff)
------------------------------------------
After training "Reign 1," we BURY its foundation memory (a deposit copy K0,V0).
A successor "Reign 2" keeps learning on a shifted world. We compare two successors:
  (a) UNANCHORED   -> memory free to drift; fidelity to the original templates rots.
  (b) ANCHORED     -> a penalty pulls Vmem,Kmem back toward the buried deposit;
                      drift stays bounded and the original templates are preserved.
This empirically demonstrates value-stable succession -- Nebuchadnezzar's relay.

ENGINEERING CONVENTIONS (kept deliberately strict)
--------------------------------------------------
  * Pure NumPy, from scratch. Manual forward AND backward (no autograd).
  * A finite-difference gradient check that MUST pass (analytic vs numeric grads).
  * A real training loop (hand-written Adam) with measurable learning.
  * Self-tests + a relay/anti-drift experiment.
  * Run the file; the printed output is pasted verbatim into the chapter.

Run:   python3 chapter_0026_nebuchadnezzar_ii_-634.py
================================================================================
"""

from __future__ import annotations
import numpy as np

# Reproducibility: a single foundation seed, like a king's buried cylinder.
SEED = 605  # the year Nebuchadnezzar II took the throne
rng_global = np.random.default_rng(SEED)


# =============================================================================
# PART I — THE WORLD: canonical "divine templates" and ruined observations
# =============================================================================

def make_templates(K, D, rng):
    """The K original temples authored by the god. These are GROUND TRUTH, not
    model parameters -- the network must learn to recover them. We make them
    mutually ORTHOGONAL (distinct ground-plans, like genuinely different temples),
    then scale so each feature has std ~1, giving each surviving brick real signal
    above the ruin's noise. Orthogonality is what lets a recovered foundation be
    unambiguously the right one rather than a blend of two similar temples."""
    A = rng.standard_normal((D, K))
    Q, _ = np.linalg.qr(A)          # (D,K) orthonormal columns
    T = Q.T * np.sqrt(D)            # (K,D) orthogonal rows, per-feature std ~1
    return T


def ruin(T_k, rng, noise=0.30, mask_frac=0.35):
    """Produce a RUINED observation of one template: additive noise plus random
    erosion (a fraction of the ground-plan's features masked to zero), exactly the
    state in which Nebuchadnezzar found the decayed temples he rebuilt."""
    D = T_k.shape[0]
    x = T_k + noise * rng.standard_normal(D)
    mask = (rng.random(D) > mask_frac).astype(np.float64)  # 1 = surviving brick
    return x * mask


def make_batch(T, rng, B, noise=0.30, mask_frac=0.35):
    """Sample a batch of (ruined input, true-template target, template index)."""
    K, D = T.shape
    idx = rng.integers(0, K, size=B)
    X = np.stack([ruin(T[k], rng, noise, mask_frac) for k in idx])  # (B, D)
    Y = T[idx]                                                       # (B, D)
    return X, Y, idx


# =============================================================================
# PART II — PARAMETERS of the Temennu Network
# =============================================================================
# Layer shapes (kept compact so the gradient check is fast and exact in float64).
#   Encoder:  D -> H (tanh) -> Z
#   Memory :  M slots, keys Kmem (M,Z), values Vmem (M,D)
#   Restorer: Z -> G (tanh) -> D  (this is Delta, the bounded deviation)

def init_params(D, H, Z, M, G, rng):
    """Xavier-ish initialization. Returns a flat dict of named arrays so the
    optimizer and the gradient checker can iterate over them uniformly."""
    def w(shape, fan_in):
        return rng.standard_normal(shape) * np.sqrt(1.0 / fan_in)
    P = {
        # encoder
        "W1": w((H, D), D), "b1": np.zeros(H),
        "W2": w((Z, H), H), "b2": np.zeros(Z),
        # temennu memory (the buried foundations)
        "Kmem": w((M, Z), Z), "Vmem": w((M, D), D),
        # restorer (produces the bounded deviation Delta)
        "W3": w((G, Z), Z), "b3": np.zeros(G),
        "W4": w((D, G), G), "b4": np.zeros(D),
    }
    return P


# =============================================================================
# PART III — FORWARD PASS  (returns scalar loss + a cache for backprop)
# =============================================================================

def softmax_rows(S):
    """Numerically stable row-wise softmax."""
    S = S - S.max(axis=1, keepdims=True)
    E = np.exp(S)
    return E / E.sum(axis=1, keepdims=True)


def forward(P, X, Y, hp):
    """
    Compute the full forward pass and the composite loss.

    hp: dict of hyper-parameters with keys:
        tau     temperature of the foundation-retrieval softmax
        la      weight of L_align   (recover the correct buried foundation)
        ld      weight of L_drift   ("not a finger's breadth")
        wd      weight decay (L2 on weight matrices, not biases)
        anchor  optional dict {'K0':..., 'V0':..., 'lam':...} -> deposit anchor
    """
    tau, la, ld, wd = hp["tau"], hp["la"], hp["ld"], hp["wd"]
    B = X.shape[0]

    # --- Encoder: survey the ruin -> latent query z ---
    pre1 = X @ P["W1"].T + P["b1"]          # (B,H)
    h    = np.tanh(pre1)                    # (B,H)
    z    = h @ P["W2"].T + P["b2"]          # (B,Z)

    # --- Temennu retrieval: which buried foundation does this site sit on? ---
    S = (z @ P["Kmem"].T) / tau             # (B,M) match latent to each deposit
    A = softmax_rows(S)                     # (B,M) attention over foundations
    F = A @ P["Vmem"]                       # (B,D) recovered original template

    # --- Restorer: rebuild = foundation + bounded deviation ---
    pre3 = z @ P["W3"].T + P["b3"]          # (B,G)
    g    = np.tanh(pre3)                    # (B,G)
    Delta = g @ P["W4"].T + P["b4"]         # (B,D) the deviation
    Xhat = F + Delta                        # (B,D) the restored temple

    # --- Losses ---
    R = Xhat - Y                            # (B,D) restoration residual
    L_restore = 0.5 * np.mean(np.sum(R * R, axis=1))
    Fres = F - Y
    L_align = 0.5 * la * np.mean(np.sum(Fres * Fres, axis=1))
    L_drift = 0.5 * ld * np.mean(np.sum(Delta * Delta, axis=1))

    # weight decay on weight matrices only
    wkeys = ["W1", "W2", "Kmem", "Vmem", "W3", "W4"]
    L_wd = 0.5 * wd * sum(np.sum(P[k] * P[k]) for k in wkeys)

    # optional anchor to a buried deposit (the relay of reigns)
    L_anchor = 0.0
    anchor = hp.get("anchor")
    if anchor is not None:
        lam = anchor["lam"]
        dK = P["Kmem"] - anchor["K0"]
        dV = P["Vmem"] - anchor["V0"]
        L_anchor = 0.5 * lam * (np.sum(dK * dK) + np.sum(dV * dV))

    loss = L_restore + L_align + L_drift + L_wd + L_anchor

    cache = dict(X=X, Y=Y, h=h, z=z, S=S, A=A, F=F, g=g, Delta=Delta,
                 Xhat=Xhat, R=R, Fres=Fres, B=B, hp=hp)
    parts = dict(restore=L_restore, align=L_align, drift=L_drift,
                 wd=L_wd, anchor=L_anchor, total=loss)
    return loss, cache, parts


# =============================================================================
# PART IV — BACKWARD PASS  (analytic gradients, derived by hand)
# =============================================================================

def backward(P, cache):
    """Return grads dict mirroring P. Every line corresponds to a term derived in
    the module docstring's loss; see inline notes for the chain-rule step."""
    hp = cache["hp"]
    tau, la, ld, wd = hp["tau"], hp["la"], hp["ld"], hp["wd"]
    B = cache["B"]
    X, Y = cache["X"], cache["Y"]
    h, z, A, F, g, Delta = cache["h"], cache["z"], cache["A"], cache["F"], cache["g"], cache["Delta"]
    R, Fres = cache["R"], cache["Fres"]

    # d L_restore / d Xhat = R / B ; Xhat = F + Delta -> flows to both F and Delta
    dXhat = R / B                              # (B,D)
    dDelta = dXhat + (ld / B) * Delta          # drift adds (ld/B)Delta
    dF     = dXhat + (la / B) * Fres           # align adds (la/B)(F-Y)

    # --- Restorer branch: Delta = g @ W4.T + b4 ; g = tanh(z @ W3.T + b3) ---
    dW4 = dDelta.T @ g                         # (D,G)
    db4 = dDelta.sum(axis=0)                   # (D,)
    dg  = dDelta @ P["W4"]                     # (B,G)
    dpre3 = dg * (1.0 - g * g)                 # tanh'
    dW3 = dpre3.T @ z                          # (G,Z)
    db3 = dpre3.sum(axis=0)                    # (G,)
    dz_dec = dpre3 @ P["W3"]                   # (B,Z)

    # --- Memory branch: F = A @ Vmem ; A = softmax(S) ; S = (z @ Kmem.T)/tau ---
    dVmem = A.T @ dF                           # (M,D)  each slot value gets sum_b A*dF
    dA = dF @ P["Vmem"].T                      # (B,M)
    # softmax Jacobian (row-wise): dS = A * (dA - sum(A*dA))
    dS = A * (dA - np.sum(A * dA, axis=1, keepdims=True))  # (B,M)
    dKmem = (dS / tau).T @ z                   # (M,Z)
    dz_mem = (dS / tau) @ P["Kmem"]            # (B,Z)

    dz = dz_dec + dz_mem                       # (B,Z) latent receives both branches

    # --- Encoder: z = h @ W2.T + b2 ; h = tanh(X @ W1.T + b1) ---
    dW2 = dz.T @ h                             # (Z,H)
    db2 = dz.sum(axis=0)                       # (Z,)
    dh  = dz @ P["W2"]                         # (B,H)
    dpre1 = dh * (1.0 - h * h)                 # tanh'
    dW1 = dpre1.T @ X                          # (H,D)
    db1 = dpre1.sum(axis=0)                    # (H,)

    grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2,
             "Kmem": dKmem, "Vmem": dVmem,
             "W3": dW3, "b3": db3, "W4": dW4, "b4": db4}

    # weight decay grads (weights only)
    for k in ["W1", "W2", "Kmem", "Vmem", "W3", "W4"]:
        grads[k] = grads[k] + wd * P[k]

    # anchor grads (relay of reigns)
    anchor = hp.get("anchor")
    if anchor is not None:
        lam = anchor["lam"]
        grads["Kmem"] = grads["Kmem"] + lam * (P["Kmem"] - anchor["K0"])
        grads["Vmem"] = grads["Vmem"] + lam * (P["Vmem"] - anchor["V0"])

    return grads


# =============================================================================
# PART V — FINITE-DIFFERENCE GRADIENT CHECK  (mandatory; must pass)
# =============================================================================

def gradient_check(verbose=True):
    """Central-difference check of EVERY parameter against analytic backprop.
    Uses tiny dims, float64, and includes align/drift/anchor/weight-decay so the
    whole loss surface is exercised."""
    rng = np.random.default_rng(7)
    D, H, Z, M, G, B = 6, 8, 5, 4, 7, 3
    P = init_params(D, H, Z, M, G, rng)
    X = rng.standard_normal((B, D))
    Y = rng.standard_normal((B, D)); Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    # exercise every loss term, including the anchor branch
    hp = dict(tau=0.7, la=0.5, ld=0.3, wd=1e-3,
              anchor=dict(K0=rng.standard_normal((M, Z)),
                          V0=rng.standard_normal((M, D)), lam=0.05))

    loss, cache, _ = forward(P, X, Y, hp)
    grads = backward(P, cache)

    eps = 1e-6
    max_rel = 0.0
    worst = None
    for name in P:
        Pk = P[name]
        num = np.zeros_like(Pk)
        it = np.nditer(Pk, flags=["multi_index"])
        while not it.finished:
            i = it.multi_index
            orig = Pk[i]
            Pk[i] = orig + eps
            lp, _, _ = forward(P, X, Y, hp)
            Pk[i] = orig - eps
            lm, _, _ = forward(P, X, Y, hp)
            Pk[i] = orig
            num[i] = (lp - lm) / (2 * eps)
            it.iternext()
        ana = grads[name]
        denom = np.maximum(1e-8, np.abs(ana) + np.abs(num))
        rel = np.max(np.abs(ana - num) / denom)
        if rel > max_rel:
            max_rel, worst = rel, name
        if verbose:
            print(f"   {name:5s} shape {str(Pk.shape):8s}  max-rel-err = {rel:.2e}")
    if verbose:
        print(f"   -> worst parameter: {worst}   overall max-rel-err = {max_rel:.2e}")
    ok = max_rel < 1e-5
    return ok, max_rel


# =============================================================================
# PART VI — OPTIMIZER (hand-written Adam) and TRAINING LOOP
# =============================================================================

class Adam:
    def __init__(self, P, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in P.items()}
        self.v = {k: np.zeros_like(v) for k, v in P.items()}
        self.t = 0

    def step(self, P, grads):
        self.t += 1
        for k in P:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (grads[k] ** 2)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            P[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


def fixed_eval_set(T, n=600, seed=2024):
    """A FROZEN evaluation set (its own seed) so metrics are stable across calls
    and do not perturb the training RNG."""
    er = np.random.default_rng(seed)
    return make_batch(T, er, n)


def retrieval_accuracy(P, T, hp, eval_set=None):
    """Fraction of ruined temples whose recovered foundation (argmax attention)
    points to the slot that best matches the TRUE template -- i.e., did the king
    dig down to the right foundation?"""
    X, Y, idx = eval_set if eval_set is not None else fixed_eval_set(T)
    _, cache, _ = forward(P, X, Y, hp)
    chosen = np.argmax(cache["A"], axis=1)                 # slot per example
    sims = P["Vmem"] @ T.T                                  # (M,K) slot<->template
    slot_to_tmpl = np.argmax(sims, axis=1)                  # (M,)
    pred = slot_to_tmpl[chosen]
    return np.mean(pred == idx)


def restoration_error(P, T, hp, eval_set=None):
    X, Y, idx = eval_set if eval_set is not None else fixed_eval_set(T)
    _, cache, _ = forward(P, X, Y, hp)
    return np.mean(np.sum((cache["Xhat"] - Y) ** 2, axis=1))


def baseline_error(T, eval_set=None):
    """Error if we naively returned the ruined input itself (no restoration)."""
    X, Y, idx = eval_set if eval_set is not None else fixed_eval_set(T)
    return np.mean(np.sum((X - Y) ** 2, axis=1))


def train(P, T, rng, hp, steps=1500, B=64, lr=3e-3, log_every=300, label="train",
          eval_set=None):
    if eval_set is None:
        eval_set = fixed_eval_set(T)
    opt = Adam(P, lr=lr)
    history = []
    for s in range(1, steps + 1):
        X, Y, idx = make_batch(T, rng, B)
        loss, cache, parts = forward(P, X, Y, hp)
        grads = backward(P, cache)
        opt.step(P, grads)
        if s % log_every == 0 or s == 1:
            err = restoration_error(P, T, hp, eval_set)
            acc = retrieval_accuracy(P, T, hp, eval_set)
            history.append((s, parts["total"], err, acc))
            print(f"   [{label}] step {s:5d}  loss={parts['total']:.4f} "
                  f"restore_err={err:.4f}  retrieval_acc={acc:.3f}")
    return history


# =============================================================================
# PART VII — THE RELAY OF REIGNS  (anti-drift across model generations)
# =============================================================================

def reign_relay_experiment(T, rng):
    """Train Reign 1, BURY its foundation deposit, then continue into Reign 2 on a
    SHIFTED world both with and without anchoring to the deposit. Show that the
    anchored successor preserves the original templates while the free one drifts.
    """
    D = T.shape[1]
    K = T.shape[0]
    M = 10  # more foundation slots than temples (see main)
    hp = dict(tau=0.3, la=1.0, ld=0.3, wd=1e-4)
    eval_set = fixed_eval_set(T)

    print("\n" + "-" * 72)
    print("RELAY OF REIGNS  —  value-stable succession vs. drift")
    print("-" * 72)

    # ---- Reign 1: establish the foundations ----
    P = init_params(D, H=24, Z=16, M=M, G=24, rng=rng)
    print(" Reign 1 — establishing the temennu (foundations):")
    train(P, T, rng, hp, steps=1500, B=64, lr=4e-3, log_every=500, label="reign1",
          eval_set=eval_set)

    # BURY the deposit: a verifiable copy of the foundation memory.
    K0 = P["Kmem"].copy()
    V0 = P["Vmem"].copy()
    err1 = restoration_error(P, T, hp, eval_set)
    print(f" Reign 1 fidelity to ORIGINAL templates: restore_err={err1:.4f}")

    # ---- The world shifts under the successor (heavier ruin) ----
    shift_rng = np.random.default_rng(539)  # fall of Babylon, a fitting successor seed

    def shifted_batch(B):
        # successor faces a different ruin distribution: more noise, more erosion
        return make_batch(T, shift_rng, B, noise=0.6, mask_frac=0.6)

    def continue_reign(P_start, anchored):
        Pn = {k: v.copy() for k, v in P_start.items()}
        hpn = dict(hp)
        if anchored:
            hpn["anchor"] = dict(K0=K0, V0=V0, lam=0.8)  # pull back toward deposit
        opt = Adam(Pn, lr=4e-3)
        for s in range(1, 1201):
            X, Y, idx = shifted_batch(64)
            loss, cache, parts = forward(Pn, X, Y, hpn)
            grads = backward(Pn, cache)
            opt.step(Pn, grads)
        drift = np.sqrt(np.sum((Pn["Vmem"] - V0) ** 2) + np.sum((Pn["Kmem"] - K0) ** 2))
        # fidelity measured against the ORIGINAL, unshifted divine templates
        fid = restoration_error(Pn, T, hp, eval_set)
        return drift, fid

    drift_free, fid_free = continue_reign(P, anchored=False)
    drift_anch, fid_anch = continue_reign(P, anchored=True)

    print("\n Successor reign on a SHIFTED world (fidelity judged vs. originals):")
    print(f"   UNANCHORED successor : foundation drift={drift_free:.3f}  "
          f"restore_err_on_originals={fid_free:.4f}")
    print(f"   ANCHORED   successor : foundation drift={drift_anch:.3f}  "
          f"restore_err_on_originals={fid_anch:.4f}")
    verdict = (drift_anch < drift_free) and (fid_anch <= fid_free + 1e-6)
    print(f"   -> anchoring reduced drift AND preserved the originals: {verdict}")
    return verdict


# =============================================================================
# PART VIII — MAIN: gradient check, training, self-tests, relay
# =============================================================================

# module-level dims used by training/relay (kept here for clarity)
H, Z, G = 24, 16, 24

def main():
    print("=" * 72)
    print(" 26 — NEBUCHADNEZZAR II  ·  The Temennu Network")
    print(" Restoration-to-template: recover the buried foundation, do not drift")
    print("=" * 72)

    # ---- 1. Gradient check (must pass before anything else is trusted) ----
    print("\n[1] FINITE-DIFFERENCE GRADIENT CHECK (analytic vs. numeric):")
    ok, mre = gradient_check(verbose=True)
    assert ok, f"GRADIENT CHECK FAILED (max rel err {mre:.2e})"
    print(f"   PASS — analytic gradients match finite differences (max rel err {mre:.2e}).")

    # ---- 2. Train on the canonical templates ----
    print("\n[2] TRAINING the Temennu Network on K canonical 'divine' templates:")
    rng = np.random.default_rng(SEED)
    K, D = 5, 16
    M = 10  # MORE foundation slots than distinct temples -> avoids slot collapse
    T = make_templates(K, D, rng)
    hp = dict(tau=0.3, la=1.0, ld=0.3, wd=1e-4)
    P = init_params(D, H, Z, M=M, G=G, rng=rng)
    eval_set = fixed_eval_set(T)

    base = baseline_error(T, eval_set)
    print(f"   baseline (return the ruin as-is) restore_err = {base:.4f}")
    train(P, T, rng, hp, steps=2400, B=64, lr=2e-3, log_every=400, label="main",
          eval_set=eval_set)

    # ---- 3. Self-tests ----
    print("\n[3] SELF-TESTS:")
    err = restoration_error(P, T, hp, eval_set)
    acc = retrieval_accuracy(P, T, hp, eval_set)
    # The bounded deviation Delta should be SMALL relative to the recovered
    # foundation -- "not a finger's breadth" should hold after training.
    X, Y, idx = eval_set
    _, cache, _ = forward(P, X, Y, hp)
    delta_norm = np.mean(np.linalg.norm(cache["Delta"], axis=1))
    found_norm = np.mean(np.linalg.norm(cache["F"], axis=1))
    ratio = delta_norm / (found_norm + 1e-9)

    t1 = err < 0.5 * base
    t2 = acc > 0.85
    t3 = ratio < 0.5
    print(f"   (a) restoration beats baseline by >2x : {t1}  ({err:.4f} vs {base:.4f})")
    print(f"   (b) foundation retrieval accuracy>0.85: {t2}  (acc={acc:.3f})")
    print(f"   (c) deviation stays bounded vs founda.: {t3}  (||Delta||/||F||={ratio:.3f})")
    assert t1 and t2 and t3, "SELF-TESTS FAILED"
    print("   PASS — the network recovers buried foundations and restores with minimal drift.")

    # ---- 4. Relay of reigns (anti-drift across generations) ----
    relay_ok = reign_relay_experiment(T, rng)
    assert relay_ok, "RELAY EXPERIMENT did not show anchoring benefit"

    print("\n" + "=" * 72)
    print(" ALL CHECKS PASSED.")
    print(" The architecture embodies Nebuchadnezzar's doctrine: dig to the buried")
    print(" foundation, rebuild exactly upon it, deviate not a finger's breadth, and")
    print(" keep successors anchored to the deposit so the original never drifts away.")
    print("=" * 72)


if __name__ == "__main__":
    main()

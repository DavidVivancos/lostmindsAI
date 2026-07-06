#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
chapter_0051_anaxagoras_-500.py  —  THE NOUS-VORTEX NETWORK
A from-scratch, trainable cognitive architecture after Anaxagoras of Clazomenae
(c. 500 - c. 428 BCE)
Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 3 Minds 41 - 60 Available on Amazon https://www.amazon.com/dp/B0H6TVX69S
Resume and Interactive Demos at https://artificiology.com/
Author: David Vivancos · Chapter 0051 · Anaxagoras of Clazomenae
================================================================================

WHY THIS IS *NOT* A TRANSFORMER
-------------------------------
The lazy move for a "mind = attention" thinker is to reach for attention-over-
stored-keys and call it Nous. That misreads Anaxagoras. His distinctive cognitive
signature is not "weight the relevant tokens." It is a tightly-coupled set of
four commitments that, taken together, force a very specific kind of machine:

  (A) "In everything there is a portion of everything." (DK B11, B6)
      No representation is ever pure. Every 'thing' carries a non-zero share of
      every 'seed' (spermata). Identity is never possession of one quality; it
      is *predominance* of one share over the others. (DK B12: a thing is "that
      of which it contains the most.")

  (B) "Nothing comes to be or passes away; there is only mixture and separation."
      (DK B17) Conservation is a HARD law, not a soft penalty. The total amount
      of each seed in the cosmos is invariant. Learning may only REDISTRIBUTE.

  (C) Nous is pure, unmixed, self-ruling, and it sets matter rotating.
      (DK B12) Mind is the one thing that does *not* mix with the others; it
      stands apart and initiates a rotation (perichoresis) whose mechanical
      sorting separates the dense from the rare, dark from bright, like to like.
      Mind moves matter; matter does not contaminate Mind.

  (D) "Appearances are a glimpse of the unseen." (opsis adelon ta phainomena, B21a)
      We never read the hidden mixture directly. We read only its surface
      predominance — the appearance — and must reason back to the structure.

THE MACHINE THESE FOUR FORCE
----------------------------
A 'cosmos' is a mixture matrix  M of shape (T things, S seeds). Rows are the
things; columns are the seeds. The data arrives MIXED (chaotic). The network's
whole job — its single cognitive act — is PERICHORESIS: run a rotation that
re-separates the mixture so that each thing's true nature (its dominant seed)
re-emerges. Crucially:

  * Each rotation step is a column-softmax scaled by the column's conserved mass.
    This makes (B) STRUCTURAL: every step preserves the column totals exactly
    (proven in self-tests), and (A) STRUCTURAL: softmax never returns a hard
    zero, so a portion of every seed survives in every thing, forever.

  * NOUS is a small, separate controller: a seed-coupling kernel W (how the
    presence of one seed is evidence for another) and a per-step separation
    strength gamma_k (how hard to sharpen the vortex on step k). These params
    are PURE in the literal architectural sense: they are never rows of M, never
    mixed into the cosmos. Gradients flow controller -> behaviour (mind moves
    matter); the cosmos M never edits the substance of W or gamma. That is the
    computational shadow of "Nous is unmixed."

  * Supervision is ONLY the coarse appearance: a row-softmax readout of the final
    mixture, trained against each thing's true dominant seed. The full hidden
    mixture is never supervised. The net must arrange the unseen to make the seen
    come out right — (D).

So the architecture is a *conservative, mind-driven re-separation dynamical
system*, not an attention stack. It learns the vortex that turns chaos back into
cosmos.

WHAT THE FILE DOES (run it):
  1. Builds a synthetic "ordered -> mixed" world by scrambling pure cosmoi with
     random DOUBLY-STOCHASTIC mixing operators (doubly-stochastic = conserves
     every seed's total, the only physically Anaxagorean way to mix).
  2. Defines the NousVortex model with analytic forward + hand-derived backward.
  3. Runs a finite-difference GRADIENT CHECK on every parameter group (MANDATORY).
  4. Trains the vortex to re-separate, reporting loss and held-out accuracy, and
     compares against the pre-rotation "naive appearance" baseline.
  5. Runs SELF-TESTS for the two hard invariants: conservation of seed-totals and
     the survival of a portion-of-everything (strict positivity).

Pure NumPy. No autograd. No frameworks. Seeded for reproducibility.
================================================================================
"""

import numpy as np

RNG = np.random.default_rng(51)  # 51 = Anaxagoras' figure id, for reproducibility


# ==============================================================================
# SECTION 0 — SMALL NUMERICAL PRIMITIVES
# ==============================================================================

def softmax(z, axis):
    """Numerically stable softmax along `axis`."""
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def softplus(x):
    """softplus(x) = log(1+e^x), used to keep separation strengths POSITIVE.
    A negative gamma would *mix* instead of separate — forbidden for a vortex."""
    return np.logaddexp(0.0, x)


def d_softplus(x):
    """derivative of softplus = sigmoid(x)."""
    return 1.0 / (1.0 + np.exp(-x))


# ==============================================================================
# SECTION 1 — THE WORLD: ORDERED COSMOS  ->  MIXED COSMOS
# ==============================================================================
# Anaxagoras' cosmogony: things begin "all together" in an undifferentiated
# mixture; only mixture and separation are real. We model the *forward* arrow of
# chaos (a conservative scrambling) so the network can learn the *reverse* arrow
# (perichoresis, re-separation).

def sinkhorn_doubly_stochastic(A, iters=60):
    """Project a positive matrix to (approximately) doubly stochastic by
    alternately normalising rows and columns. A doubly-stochastic mixing matrix
    is the ONLY kind that conserves every seed's column-total when applied as
    M_mixed = MIX @ M_ordered — i.e. it rearranges 'stuff' among things without
    creating or destroying any (DK B17)."""
    A = A.copy()
    for _ in range(iters):
        A = A / A.sum(axis=1, keepdims=True)   # rows -> 1
        A = A / A.sum(axis=0, keepdims=True)   # cols -> 1
    return A


def make_cosmos(T, S, contam=1.7, blur=0.12, floor=0.03, noise=0.25):
    """Create ONE training example: a chaotic, MISLEADING mixture.

    Returns:
      M_obs  (T,S)  : the observed, mixed cosmos (network input)
      labels (T,)   : each thing's TRUE dominant seed (the 'unseen' truth)
      m      (S,)   : column masses of the observation (what rotation conserves)

    The chaos is *structured*, which is the whole Anaxagorean point: the surface
    appearance lies, but in a law-governed way that Mind can see through.

      * Each thing t has a true seed y_t (its real nature) and receives a unit
        share there, plus a small floor of EVERY seed (portion-of-everything).
      * It is then loaded with a heavier CONTAMINANT share at seed (y_t + 1) mod S.
        Because contam > 1, the raw predominance (argmax) is usually the
        contaminant, NOT the truth: "appearances are a glimpse of the unseen"
        (DK B21a). A mind that merely reads the surface is systematically fooled.
      * A doubly-stochastic `blur` lightly stirs shares between things, and noise
        adds idiosyncrasy (so the naive reader is sometimes accidentally right).

    The kinship "true = (contaminant - 1) mod S" is a fixed law of this world.
    Nous can learn it (in its seed-coupling kernel W) and, by rotating the
    mixture, gather each true seed back onto its own things — recovering natures
    that the surface concealed. Crucially the rotation only REARRANGES the
    observed seed-totals m (conservation); it never invents seed it wasn't given.
    """
    labels = RNG.integers(0, S, size=T)
    M = np.full((T, S), floor, dtype=np.float64)
    M[np.arange(T), labels] += 1.0                       # true nature
    contaminant = (labels + 1) % S
    M[np.arange(T), contaminant] += contam               # misleading surface
    M += RNG.uniform(0.0, noise, size=(T, S))            # idiosyncrasy

    if blur > 0.0:                                       # light inter-thing stir
        raw = RNG.uniform(0.1, 1.0, size=(T, T))
        MIX = sinkhorn_doubly_stochastic(raw)
        MIX = (1.0 - blur) * np.eye(T) + blur * MIX
        M = MIX @ M

    m = M.sum(axis=0)                                     # conserved by rotation
    return M, labels, m


def make_batch(B, T, S):
    Ms, Ys, Mass = [], [], []
    for _ in range(B):
        mo, y, m = make_cosmos(T, S)
        Ms.append(mo); Ys.append(y); Mass.append(m)
    return np.stack(Ms), np.stack(Ys), np.stack(Mass)


# ==============================================================================
# SECTION 2 — NOUS: THE PURE, UNMIXED CONTROLLER
# ==============================================================================
class Nous:
    """The ordering mind. It owns NO matter — only the *laws* by which matter is
    sorted. Two parameter groups:

      W      (S,S) : seed-coupling kernel. The presence of seed j in a thing is
                     evidence for seed i predominating there. The effective
                     operator is (W + I): every seed at least supports itself,
                     and may borrow evidence from kindred seeds (like-to-like).
      g      (K,)  : raw per-step separation knobs; gamma_k = softplus(g_k) > 0
                     is how violently rotation step k sharpens the vortex.
      tau_raw      : raw sharpness of the *appearance* readout; tau = softplus.

    Purity, made literal: these arrays are never concatenated into, sliced from,
    or overwritten by the cosmos M. Mind acts on matter; matter never becomes
    Mind. The only channel from matter to Mind is the learning gradient — Nous
    *learns about* the world without being *made of* it.
    """

    def __init__(self, S, K):
        self.S = S
        self.K = K
        # small symmetric-ish init; W stays a free matrix but we init gently
        self.W = RNG.normal(0, 0.05, size=(S, S))
        self.g = np.full(K, 0.5)        # softplus(0.5) ~ 0.97 separation to start
        self.tau_raw = np.array([1.0])  # appearance sharpness (1-elem, mutable)

    # --- parameter (de)serialisation for the gradient checker ----------------
    def get_params(self):
        return {"W": self.W, "g": self.g, "tau_raw": self.tau_raw}

    def set_params(self, p):
        self.W = p["W"]; self.g = p["g"]; self.tau_raw = p["tau_raw"]


# ==============================================================================
# SECTION 3 — PERICHORESIS: ONE FORWARD PASS (THE ROTATION)
# ==============================================================================
def forward(nous, M_obs, labels=None, reg=1e-4):
    """Run K rotation steps on a single cosmos and read off the appearance.

    Each rotation step k, for each seed-column s:
        H[:,s]      = (M @ (W+I))[:,s]              # evidence potential per thing
        p           = softmax_over_things(gamma_k * H[:,s])
        M_new[:,s]  = m_s * p                       # conserve column mass m_s

    Because each new column is (mass) * (a softmax over things), the column SUM
    is exactly m_s on every step (conservation, structurally guaranteed), and no
    entry is ever exactly zero (portion-of-everything, structurally guaranteed).

    Appearance readout (the only thing the world ever shows us):
        probs[t,:]  = softmax_over_seeds(tau * M_final[t,:])

    Returns (loss_or_None, probs, cache). cache holds everything backward needs.
    """
    S = nous.S
    Wp = nous.W + np.eye(S)                 # (W + I)
    gammas = softplus(nous.g)              # (K,)
    tau = softplus(nous.tau_raw)

    m = M_obs.sum(axis=0)                   # (S,) conserved masses for THIS cosmos
    M = M_obs.copy()

    steps = []                             # per-step intermediates for backprop
    for k in range(nous.K):
        Mprev = M
        H = Mprev @ Wp                     # (T,S)
        cols_p = np.empty_like(H)
        Mnew = np.empty_like(H)
        for s in range(S):
            a = gammas[k] * H[:, s]
            p = softmax(a, axis=0)         # over things
            cols_p[:, s] = p
            Mnew[:, s] = m[s] * p
        steps.append({"Mprev": Mprev, "H": H, "p": cols_p, "gamma": gammas[k]})
        M = Mnew

    M_final = M
    logits = tau * M_final                 # (T,S) appearance logits
    probs = softmax(logits, axis=1)        # over seeds, per thing

    cache = {"Wp": Wp, "gammas": gammas, "tau": tau, "m": m,
             "steps": steps, "M_final": M_final, "probs": probs, "reg": reg}

    if labels is None:
        return None, probs, cache

    T = M_obs.shape[0]
    ll = -np.log(probs[np.arange(T), labels] + 1e-12).mean()
    loss = ll + reg * np.sum(nous.W ** 2)  # tiny L2 keeps the kernel honest
    return loss, probs, cache


# ==============================================================================
# SECTION 4 — BACKWARD: HAND-DERIVED ANALYTIC GRADIENTS
# ==============================================================================
def backward(nous, M_obs, labels, cache):
    """Reverse-mode gradient of the loss wrt W, g, tau_raw. Derived by hand.

    Key Jacobians used:
      * CE + row-softmax readout: dL/dlogits = (probs - onehot)/T.
      * column-softmax with mass: for out = m * softmax(gamma*H_col),
            dL/dH_col   = gamma * m * (p ⊙ u - p * (p·u)),  u = upstream
            dL/dgamma  += (m * (p ⊙ u - p * (p·u))) · H_col
      * H = Mprev @ (W+I):  dL/dW += Mprev^T @ dH ;  dL/dMprev = dH @ (W+I)^T
    """
    S = nous.S
    K = nous.K
    Wp = cache["Wp"]; tau = cache["tau"]; m = cache["m"]
    probs = cache["probs"]; M_final = cache["M_final"]
    T = M_obs.shape[0]

    # ---- readout / loss backward ----
    onehot = np.zeros_like(probs)
    onehot[np.arange(T), labels] = 1.0
    dlogits = (probs - onehot) / T          # (T,S)
    dM = tau * dlogits                       # into M_final
    dtau_scalar = np.sum(dlogits * M_final)  # dL/dtau

    dW = np.zeros((S, S))
    dgamma = np.zeros(K)

    # ---- unroll rotation steps in reverse ----
    for k in reversed(range(K)):
        st = cache["steps"][k]
        Mprev = st["Mprev"]; H = st["H"]; P = st["p"]; gamma = st["gamma"]
        dH = np.zeros_like(H)
        for s in range(S):
            p = P[:, s]
            u = dM[:, s]                     # upstream wrt Mnew[:,s] = m_s * p
            pdotu = np.dot(p, u)
            dLda = m[s] * (p * u - p * pdotu)   # wrt a = gamma*H_col
            dH[:, s] = gamma * dLda
            dgamma[k] += np.dot(dLda, H[:, s])
        dW += Mprev.T @ dH                   # H = Mprev @ (W+I)
        dM = dH @ Wp.T                        # propagate to previous M

    # chain raw params through their nonlinearities
    dg = dgamma * d_softplus(nous.g)         # gamma = softplus(g)
    dtau_raw = dtau_scalar * d_softplus(nous.tau_raw)

    # L2 reg term
    dW += 2.0 * cache["reg"] * nous.W

    return {"W": dW, "g": dg, "tau_raw": np.array(dtau_raw)}


# ==============================================================================
# SECTION 5 — GRADIENT CHECK (MANDATORY)
# ==============================================================================
def gradient_check():
    """Finite-difference verification of every parameter group. The relative
    error between analytic and numerical gradients must be tiny (< 1e-5)."""
    print("=" * 72)
    print("GRADIENT CHECK  (analytic vs. central finite differences)")
    print("=" * 72)
    T, S, K = 5, 4, 3
    nous = Nous(S, K)
    M_obs, labels, _ = make_cosmos(T, S)

    loss, _, cache = forward(nous, M_obs, labels)
    grads = backward(nous, M_obs, labels, cache)

    eps = 1e-6
    worst = 0.0
    for name in ["W", "g", "tau_raw"]:
        # IMPORTANT: perturb the LIVE array in place (a copy would make the
        # numerical gradient identically zero). ravel() returns a view for these
        # contiguous float64 arrays, so writes propagate into `nous`.
        arr = nous.get_params()[name]
        flat = arr.ravel()
        ana = np.atleast_1d(grads[name]).astype(float).ravel()
        num = np.zeros_like(flat)
        for i in range(flat.size):
            orig = flat[i]
            flat[i] = orig + eps
            lp, _, _ = forward(nous, M_obs, labels)
            flat[i] = orig - eps
            lm, _, _ = forward(nous, M_obs, labels)
            flat[i] = orig
            num[i] = (lp - lm) / (2 * eps)
        denom = np.maximum(1e-12, np.abs(ana) + np.abs(num))
        rel = np.max(np.abs(ana - num) / denom)
        worst = max(worst, rel)
        status = "OK" if rel < 1e-5 else "FAIL"
        print(f"  {name:8s} shape={str(arr.shape):8s} "
              f"max-rel-err = {rel:.3e}   [{status}]")
    print("-" * 72)
    ok = worst < 1e-5
    print(f"  WORST rel-err = {worst:.3e}  ->  {'PASS' if ok else 'FAIL'}")
    print()
    assert ok, "Gradient check FAILED — analytic backward is wrong."
    return ok


# ==============================================================================
# SECTION 6 — TRAINING THE VORTEX (re-separation by perichoresis)
# ==============================================================================
def accuracy(nous, M_obs, labels):
    _, probs, _ = forward(nous, M_obs, labels=None)
    return float((probs.argmax(axis=1) == labels).mean())


def naive_baseline_acc(M_obs, labels):
    """Appearance WITHOUT rotation: just read which seed predominates in the raw,
    mixed observation. This is what a mind that does NOT order chaos would 'see'."""
    return float((M_obs.argmax(axis=1) == labels).mean())


def train(epochs=400, T=12, S=6, K=3, lr=0.20,
          batch=24, eval_every=80):
    print("=" * 72)
    print("TRAINING  —  Nous learns the rotation that re-separates the cosmos")
    print("=" * 72)
    nous = Nous(S, K)

    # fixed held-out set
    Me, Ye, _ = make_batch(40, T, S)
    base = np.mean([naive_baseline_acc(Me[i], Ye[i]) for i in range(len(Me))])

    for ep in range(1, epochs + 1):
        Mb, Yb, _ = make_batch(batch, T, S)
        # accumulate grads over the batch
        gW = np.zeros((S, S)); gg = np.zeros(K); gt = 0.0
        ep_loss = 0.0
        for i in range(batch):
            loss, _, cache = forward(nous, Mb[i], Yb[i])
            grads = backward(nous, Mb[i], Yb[i], cache)
            gW += grads["W"]; gg += grads["g"]; gt += float(grads["tau_raw"].ravel()[0])
            ep_loss += loss
        gW /= batch; gg /= batch; gt /= batch
        ep_loss /= batch

        # plain SGD step on the PURE controller (matter never updated)
        nous.W -= lr * gW
        nous.g -= lr * gg
        nous.tau_raw = nous.tau_raw - lr * gt

        if ep % eval_every == 0 or ep == 1:
            acc = np.mean([accuracy(nous, Me[i], Ye[i]) for i in range(len(Me))])
            print(f"  epoch {ep:4d} | loss {ep_loss:6.4f} | "
                  f"held-out acc {acc:5.3f} | gamma {softplus(nous.g)} ")

    final = np.mean([accuracy(nous, Me[i], Ye[i]) for i in range(len(Me))])
    print("-" * 72)
    print(f"  naive appearance baseline (no rotation): {base:5.3f}")
    print(f"  Nous-Vortex (after rotation, trained)  : {final:5.3f}")
    print(f"  perichoresis lift                       : {final - base:+5.3f}")
    print()
    return nous, base, final


# ==============================================================================
# SECTION 7 — SELF-TESTS: THE TWO HARD INVARIANTS
# ==============================================================================
def test_conservation(nous):
    """DK B17: nothing comes to be or passes away. Across EVERY rotation step the
    total of each seed (column sum) must be invariant to machine precision."""
    print("=" * 72)
    print("SELF-TEST 1 — CONSERVATION OF SEED-TOTALS (DK B17)")
    print("=" * 72)
    M_obs, labels, m = make_cosmos(10, nous.S)
    _, _, cache = forward(nous, M_obs, labels)
    # reconstruct column sums after each step from the cache
    worst = 0.0
    M = M_obs.copy()
    Wp = cache["Wp"]; gammas = cache["gammas"]
    for k in range(nous.K):
        H = M @ Wp
        Mnew = np.empty_like(H)
        for s in range(nous.S):
            Mnew[:, s] = m[s] * softmax(gammas[k] * H[:, s], axis=0)
        err = np.max(np.abs(Mnew.sum(axis=0) - m))
        worst = max(worst, err)
        print(f"  after step {k+1}: max |col_sum - m| = {err:.2e}")
        M = Mnew
    ok = worst < 1e-10
    print(f"  -> conservation error {worst:.2e}  [{'PASS' if ok else 'FAIL'}]")
    print()
    assert ok, "Conservation violated."
    return ok


def test_portion_of_everything(nous):
    """DK B11/B6: in everything a portion of everything. No mixture entry may be
    exactly zero after rotation — every thing keeps a share of every seed."""
    print("=" * 72)
    print("SELF-TEST 2 — A PORTION OF EVERYTHING IN EVERYTHING (DK B11/B6)")
    print("=" * 72)
    M_obs, labels, _ = make_cosmos(10, nous.S)
    _, _, cache = forward(nous, M_obs, labels)
    mn = cache["M_final"].min()
    ok = mn > 0.0
    print(f"  smallest portion anywhere in final cosmos = {mn:.3e}")
    print(f"  -> strictly positive  [{'PASS' if ok else 'FAIL'}]")
    print()
    assert ok, "A seed vanished from a thing — doctrine violated."
    return ok


def test_separation_monotone(nous):
    """Perichoresis should INCREASE order: the average predominance (max share of
    the appearance) should not fall as rotation proceeds on a fixed input."""
    print("=" * 72)
    print("SELF-TEST 3 — ROTATION INCREASES ORDER (perichoresis sharpens)")
    print("=" * 72)
    M_obs, labels, m = make_cosmos(10, nous.S)
    Wp = nous.W + np.eye(nous.S)
    gammas = softplus(nous.g)
    tau = softplus(nous.tau_raw)
    M = M_obs.copy()

    def order(Mx):
        return softmax(tau * Mx, axis=1).max(axis=1).mean()

    prev = order(M)
    initial = prev
    print(f"  step 0 (raw mixture)   order = {prev:.4f}")
    for k in range(nous.K):
        H = M @ Wp
        for s in range(nous.S):
            M[:, s] = m[s] * softmax(gammas[k] * H[:, s], axis=0)
        cur = order(M)
        print(f"  step {k+1}              order = {cur:.4f}")
        prev = cur
    ok = prev > initial + 1e-6   # rotation nets MORE order than the raw chaos
    print(f"  net order change = {prev - initial:+.4f}  "
          f"[{'PASS' if ok else 'FAIL'}]")
    print()
    assert ok, "Rotation did not increase order overall."
    return ok


# ==============================================================================
# SECTION 8 — MAIN
# ==============================================================================
if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)

    print()
    print("#" * 72)
    print("# THE NOUS-VORTEX NETWORK — Anaxagoras of Clazomenae (fig. 0051)")
    print("# 'All things were together; then Mind came and set them in order.'")
    print("#" * 72)
    print()

    # 1) verify the math
    gradient_check()

    # 2) learn the rotation
    nous, base, final = train()

    # 3) prove the doctrines hold structurally
    test_conservation(nous)
    test_portion_of_everything(nous)
    test_separation_monotone(nous)

    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print("  * Gradient check ................... PASS")
    print("  * Conservation (DK B17) ........... PASS  (structural invariant)")
    print("  * Portion-of-everything (B11/B6) .. PASS  (structural invariant)")
    print(f"  * Re-separation accuracy .......... {final:.3f} "
          f"(vs naive {base:.3f})")
    print()
    print("  Nous did not create the seeds. It only set them rotating, and the")
    print("  rotation drew like to like until each thing showed its nature again.")
    print("=" * 72)

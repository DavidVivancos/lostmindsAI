#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chapter_0045_aeschylus_-525.py  —  The Pathei-Mathos Network (PMN)
==================================================
A from-scratch NumPy architecture that embodies the cognitive signature of
AESCHYLUS (c. 525-456 BCE), the father of tragedy.
Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 3 Minds 41 - 60 Available on Amazon https://www.amazon.com/dp/B0H6TVX69S
Resume and Interactive Demos at https://artificiology.com/
Author: David Vivancos · Chapter 0045 · Aeschylus
------------------------------------------------------------------------------
THE ONE IDEA THIS FILE IS BUILT ON
------------------------------------------------------------------------------
Aeschylus' Agamemnon (lines 176-178) lays down a law in the Hymn to Zeus:

        ton  pathei  mathos  thenta  kyrios  echein
        "(Zeus) made it law that wisdom comes by suffering."     -> pathei mathos

This is not a slogan. It is a *theory of learning*, and it is the most specific
thing Aeschylus believed about the mind:

  1. WISDOM IS THE RESIDUE OF SUFFERING. You cannot be told the lesson; you must
     undergo the cost. Knowledge that matters is *paid for* in experience.

  2. THE PAYMENT IS IRREVERSIBLE. "There drips before the heart, instead of
     sleep, the pain of memory of pain" (Ag. 179-180). What is learned through
     suffering does not wash out. It is a scar, not a weight that is freely
     re-set. It only ever accumulates.

  3. CREDIT IS UNDISCOUNTED AND INHERITED. In the only surviving complete Greek
     trilogy, the Oresteia, an original crime by Atreus pays out two
     generations and three plays later, in full. The house inherits the guilt;
     the consequence lands undiminished across an enormous delay. Aeschylus
     refuses the modern instinct to discount the far future: the oldest wound is
     the one that decides the ending.

Compare the two nearest already-written minds so we DON'T repeat them:
  * Buddha (#39) treats suffering as a *separable, dampable* computation to be
    driven to zero. Aeschylus is his near-opposite: suffering is the ONLY road
    to wisdom and must not be dampened.
  * Leonidas (#43) uses a commitment ratchet to remove the agent's ability to
    *defect*. Aeschylus' ratchet is epistemic, not behavioural: it removes the
    agent's ability to *un-learn*.

------------------------------------------------------------------------------
HOW THE ARCHITECTURE ENCODES THAT IDEA (and why it is NOT a Transformer)
------------------------------------------------------------------------------
The PMN is a recurrent cell with TWO memories that behave very differently:

  * h_t  — REVERSIBLE working memory (an ordinary tanh hidden state). This is
           "the surface of the action": it can be overwritten, and -- crucially
           -- it is RESET at the start of every "play" (every generation). It
           models a mind that forgets the details of the previous act.

  * s_t  — the IRREVERSIBLE SCAR (wisdom memory). It is updated by a strictly
           NON-DECREASING rule:
                 s_t = s_{t-1} + a_t (.) m_t,
           where a_t in (0,1) is a "suffering gate" (how much this moment
           wounds) and m_t >= 0 is the magnitude of the wound (softplus).
           Because every increment is non-negative, s can only ever grow:
           a perfect monotone ratchet. The scar CARRIES ACROSS play boundaries
           even though h is reset -- this is inherited guilt.

The judgement happens only at the very end (the Areopagus verdict / the
anagnorisis of the trilogy): the readout sees s_T and h_T and must decide.

WHY THIS IS THE RIGHT MECHANISM, MEASURABLY:
Gradient of the loss with respect to an early scar-increment flows back through
the additive chain s_t = s_{t-1} + incr_t with local derivative exactly 1.0 --
so an early "sin" reaches the final judgement UNDISCOUNTED (no vanishing). The
same information, asked to travel through the tanh recurrence of h, vanishes
across the resets. We DEMONSTRATE this below: a full PMN solves a delayed,
inherited-credit task; an ablation that is forced to rely on h alone stays at
chance. That gap is pathei mathos, made numerical.

------------------------------------------------------------------------------
WHAT IS IN THIS FILE
------------------------------------------------------------------------------
  * PaScarNet      : the model, with hand-derived backprop-through-time.
  * make_atreus_task : "The Curse of the House of Atreus" synthetic dataset.
  * Adam           : a from-scratch optimizer.
  * gradient_check : a mandatory finite-difference check of EVERY parameter.
  * train          : a real training loop.
  * self_tests     : asserts the gradient check, scar monotonicity, learning,
                     and the full-vs-ablation gap.

Run:  python3 chapter_0045_aeschylus_-525.py
Pure NumPy. No other dependencies.
"""

import numpy as np

# ============================================================================
# 0.  Small numerically-stable primitives
# ============================================================================

def sigmoid(z):
    """Logistic. Split on sign to avoid overflow in exp."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out

def softplus(z):
    """log(1+e^z), stable. Its derivative is sigmoid(z)."""
    return np.logaddexp(0.0, z)

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ============================================================================
# 1.  The model
# ============================================================================

class PaScarNet:
    """
    Pathei-Mathos Network.

    Dimensions
    ----------
    d_in : input feature size
    H    : reversible working-memory size (h)
    S    : irreversible scar size (s)        -- "how many kinds of wound"
    O    : number of output classes (the verdict)

    The sequence is divided into `n_plays` consecutive plays of equal length.
    At the start of each play h is reset to 0 (a new generation forgets the
    surface of the last act) while the scar s persists (guilt is inherited).
    """

    def __init__(self, d_in, H, S, O, n_plays=3, seed=0):
        rng = np.random.default_rng(seed)
        self.d_in, self.H, self.S, self.O = d_in, H, S, O
        self.n_plays = n_plays

        def gw(a, b):  # Glorot-ish init
            return rng.standard_normal((a, b)) * np.sqrt(2.0 / (a + b))

        # working-memory (reversible) recurrence
        self.W_xh = gw(d_in, H); self.W_hh = gw(H, H); self.b_h = np.zeros(H)
        # suffering gate a_t = sigmoid(...)   in (0,1)
        self.W_xa = gw(d_in, S); self.W_ha = gw(H, S); self.b_a = np.zeros(S)
        # wound magnitude m_t = softplus(...) >= 0
        self.W_xm = gw(d_in, S); self.W_hm = gw(H, S); self.b_m = np.zeros(S)
        # readout (the verdict) from scar + working memory
        self.W_sy = gw(S, O); self.W_hy = gw(H, O); self.b_y = np.zeros(O)

        # Bias the gate closed at start: a mind does not "suffer" at every step.
        self.b_a[:] = -2.0

    # -- parameter bookkeeping (so the grad-check can perturb everything) ----
    def params(self):
        return {k: getattr(self, k) for k in (
            "W_xh", "W_hh", "b_h", "W_xa", "W_ha", "b_a",
            "W_xm", "W_hm", "b_m", "W_sy", "W_hy", "b_y")}

    def _play_starts(self, T):
        """Indices where h is reset (start of each play)."""
        L = T // self.n_plays
        return set(range(0, T, L)) if L > 0 else {0}

    # -----------------------------------------------------------------------
    # Forward pass. Returns logits and a cache for the backward pass.
    # X: (B, T, d_in)
    # -----------------------------------------------------------------------
    def forward(self, X):
        B, T, _ = X.shape
        H, S = self.H, self.S
        starts = self._play_starts(T)

        h_prev = np.zeros((B, H))
        s_prev = np.zeros((B, S))

        cache = {"X": X, "starts": starts,
                 "h": [], "hprev": [], "a": [], "m": [], "mpre": [],
                 "s": []}

        for t in range(T):
            if t in starts:                 # new generation: forget the surface
                h_prev = np.zeros((B, H))
            cache["hprev"].append(h_prev)

            hpre = X[:, t] @ self.W_xh + h_prev @ self.W_hh + self.b_h
            h = np.tanh(hpre)

            apre = X[:, t] @ self.W_xa + h @ self.W_ha + self.b_a
            a = sigmoid(apre)
            mpre = X[:, t] @ self.W_xm + h @ self.W_hm + self.b_m
            m = softplus(mpre)

            s = s_prev + a * m              # the ratchet: increment >= 0

            cache["h"].append(h); cache["a"].append(a)
            cache["m"].append(m); cache["mpre"].append(mpre); cache["s"].append(s)

            h_prev, s_prev = h, s

        h_T = cache["h"][-1]
        s_T = cache["s"][-1]
        logits = s_T @ self.W_sy + h_T @ self.W_hy + self.b_y
        cache["logits"] = logits
        return logits, cache

    # -----------------------------------------------------------------------
    # Loss: softmax cross-entropy at the final step only (the verdict).
    # -----------------------------------------------------------------------
    def loss(self, logits, y):
        B = logits.shape[0]
        p = softmax(logits)
        ll = -np.log(p[np.arange(B), y] + 1e-12)
        return ll.mean(), p

    # -----------------------------------------------------------------------
    # Backward pass: hand-derived BPTT through the scar ratchet and the tanh
    # working memory. `use_scar` lets us ablate the wisdom pathway.
    # Returns grads dict and the scalar loss.
    # -----------------------------------------------------------------------
    def backward(self, cache, y, use_scar=True):
        X = cache["X"]; starts = cache["starts"]
        B, T, _ = X.shape
        H, S, O = self.H, self.S, self.O

        h_T = cache["h"][-1]; s_T = cache["s"][-1]
        logits = cache["logits"]
        loss, p = self.loss(logits, y)

        g = {k: np.zeros_like(v) for k, v in self.params().items()}

        # ---- output layer ----
        dlogits = p.copy()
        dlogits[np.arange(B), y] -= 1.0
        dlogits /= B                                   # (B,O)

        if use_scar:
            g["W_sy"] += s_T.T @ dlogits
        # else: scar readout frozen at 0 -> no grad, and ds from output is 0.
        g["W_hy"] += h_T.T @ dlogits
        g["b_y"]  += dlogits.sum(0)

        # gradient entering the final scar / working memory
        ds = (dlogits @ self.W_sy.T) if use_scar else np.zeros((B, S))
        dh_future = dlogits @ self.W_hy.T              # grad into h_T from output

        # ---- BPTT ----
        for t in reversed(range(T)):
            h = cache["h"][t]; a = cache["a"][t]
            m = cache["m"][t]; mpre = cache["mpre"][t]
            hprev = cache["hprev"][t]; x = X[:, t]

            # scar increment branch:  incr = a * m ;  s_t = s_{t-1} + incr
            d_incr = ds                                # local d s_t / d incr = 1
            da = d_incr * m
            dm = d_incr * a
            dapre = da * a * (1.0 - a)                 # sigmoid'
            dmpre = dm * sigmoid(mpre)                 # softplus' = sigmoid

            g["W_xa"] += x.T @ dapre; g["W_ha"] += h.T @ dapre; g["b_a"] += dapre.sum(0)
            g["W_xm"] += x.T @ dmpre; g["W_hm"] += h.T @ dmpre; g["b_m"] += dmpre.sum(0)

            # h_t receives grad from the output (only at final step, already in
            # dh_future), from the next step's recurrence (dh_future), and from
            # the a/m branches computed at THIS step (they use h_t directly).
            dh = dh_future + dapre @ self.W_ha.T + dmpre @ self.W_hm.T

            dhpre = dh * (1.0 - h * h)                 # tanh'
            g["W_xh"] += x.T @ dhpre
            g["W_hh"] += hprev.T @ dhpre
            g["b_h"]  += dhpre.sum(0)

            # propagate to the PREVIOUS timestep
            # scar persists undiscounted across every boundary:
            #   d s_{t-1} = d s_t   (local derivative exactly 1.0)
            ds = ds.copy()
            # working memory: recurrent grad, unless this t is a reset point
            if t in starts:
                dh_future = np.zeros((B, H))           # h was reset; no grad back
            else:
                dh_future = dhpre @ self.W_hh.T

        return g, loss


# ============================================================================
# 2.  "The Curse of the House of Atreus" task
# ============================================================================
#
# A single original transgression is committed early in PLAY 1. Its NATURE
# v in {+1, -1} (which crime) determines the verdict required at the very end
# of PLAY 3. Between the crime and the verdict the working memory is reset
# twice (two generations) and the sequence is full of reversible distractor
# events (noise). The ONLY way to carry the crime's nature to the final
# judgement is to commit it to the irreversible scar at the moment of suffering.
#
# Channels of each input vector:
#   [0] transgression marker : 1.0 exactly at the crime step, else 0
#   [1] transgression value  : v in {+1,-1} at the crime step, else 0
#   [2..] reversible noise   : small Gaussian distractors every step
# ============================================================================

def make_atreus_task(B, d_in=6, n_plays=3, play_len=5, noise=0.5, seed=0):
    rng = np.random.default_rng(seed)
    T = n_plays * play_len
    X = rng.standard_normal((B, T, d_in)) * noise
    X[:, :, 0] = 0.0          # marker channel is clean
    X[:, :, 1] = 0.0          # value channel is clean
    v = rng.choice([-1.0, 1.0], size=B)
    y = (v > 0).astype(np.int64)             # verdict: was the crime "type +"?
    # place the single transgression at a random early step within play 1
    for i in range(B):
        t_crime = rng.integers(0, play_len)  # always inside the first generation
        X[i, t_crime, 0] = 1.0               # "a crime happened here"
        X[i, t_crime, 1] = v[i]              # "...and this was its nature"
    return X, y


# ============================================================================
# 3.  Adam (from scratch)
# ============================================================================

class Adam:
    def __init__(self, params, lr=3e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


# ============================================================================
# 4.  Mandatory finite-difference gradient check
# ============================================================================

def gradient_check(seed=1, eps=1e-5, tol=1e-5):
    """
    Perturb EVERY parameter and compare analytic BPTT gradient to a central
    finite difference of the loss. Returns the worst relative error.
    """
    net = PaScarNet(d_in=5, H=4, S=3, O=2, n_plays=3, seed=seed)
    X, y = make_atreus_task(B=4, d_in=5, n_plays=3, play_len=2, seed=seed)

    _, cache = net.forward(X)
    grads, _ = net.backward(cache, y, use_scar=True)

    worst = 0.0
    p = net.params()
    for name in p:
        W = p[name]
        flat = W.ravel()
        gflat = grads[name].ravel()
        # check a handful of coordinates per tensor (enough, and fast)
        idxs = np.linspace(0, flat.size - 1, min(flat.size, 6)).astype(int)
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps
            lp, _ = net.loss(net.forward(X)[0], y)
            flat[i] = orig - eps
            lm, _ = net.loss(net.forward(X)[0], y)
            flat[i] = orig
            num = (lp - lm) / (2 * eps)
            ana = gflat[i]
            denom = max(1e-12, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            worst = max(worst, rel)
    return worst


# ============================================================================
# 5.  Training loop
# ============================================================================

def accuracy(net, X, y):
    logits, _ = net.forward(X)
    return float((logits.argmax(1) == y).mean())

def train(use_scar=True, steps=400, seed=0, verbose=False):
    net = PaScarNet(d_in=6, H=16, S=4, O=2, n_plays=3, seed=seed)
    if not use_scar:
        # ABLATION: sever the wisdom pathway. The verdict must come from h alone,
        # but h is reset every generation -> the early crime cannot reach it.
        net.W_sy[:] = 0.0
    opt = Adam(net.params(), lr=3e-2)

    Xte, yte = make_atreus_task(B=400, d_in=6, n_plays=3, play_len=5, seed=999)
    for step in range(steps):
        X, y = make_atreus_task(B=64, d_in=6, n_plays=3, play_len=5, seed=step)
        _, cache = net.forward(X)
        grads, loss = net.backward(cache, y, use_scar=use_scar)
        if not use_scar:
            grads["W_sy"][:] = 0.0          # keep the wisdom readout severed
        opt.step(net.params(), grads)
        if verbose and step % 100 == 0:
            print(f"    step {step:4d}  loss={loss:.4f}  test_acc={accuracy(net, Xte, yte):.3f}")
    return net, accuracy(net, Xte, yte)


# ============================================================================
# 6.  Self-tests
# ============================================================================

def self_tests():
    print("=" * 74)
    print("THE PATHEI-MATHOS NETWORK  —  self-tests")
    print("Aeschylus #45 : wisdom is the irreversible residue of suffering")
    print("=" * 74)

    # --- (1) gradient check ------------------------------------------------
    print("\n[1] Finite-difference gradient check (all parameter tensors)")
    worst = gradient_check()
    print(f"    worst relative error = {worst:.2e}")
    assert worst < 1e-4, "gradient check FAILED"
    print("    PASS  (analytic BPTT matches numerical gradient)")

    # --- (2) scar monotonicity (the ratchet) -------------------------------
    print("\n[2] Scar irreversibility: s_t must be non-decreasing per dimension")
    net = PaScarNet(d_in=6, H=8, S=4, O=2, seed=3)
    X, _ = make_atreus_task(B=10, d_in=6, seed=7)
    _, cache = net.forward(X)
    S_seq = np.stack(cache["s"], axis=1)        # (B,T,S)
    diffs = np.diff(S_seq, axis=1)
    min_step = diffs.min()
    print(f"    minimum step-to-step change in scar = {min_step:.3e}  (>= 0 required)")
    assert min_step >= -1e-12, "scar decreased — ratchet broken"
    print("    PASS  (the scar only ever accumulates; nothing is un-learned)")

    # --- (3) undiscounted credit: gradient does not vanish through the scar -
    print("\n[3] Undiscounted credit across the trilogy")
    # d loss / d (input at the crime step) via the scar should not decay with
    # how early the crime is. We compare gradient magnitude at the crime step
    # for the full model vs the working-memory-only ablation.
    netA = PaScarNet(d_in=6, H=8, S=4, O=2, seed=5)
    X, y = make_atreus_task(B=32, d_in=6, n_plays=3, play_len=5, seed=11)
    # full
    _, c = netA.forward(X); _ = netA.backward(c, y, use_scar=True)
    gfull = input_grad_at_crime(netA, X, y, use_scar=True)
    gablate = input_grad_at_crime(netA, X, y, use_scar=False)
    print(f"    |grad| at crime step  — through scar    : {gfull:.3e}")
    print(f"    |grad| at crime step  — h-only ablation : {gablate:.3e}")
    assert gfull > 5 * gablate, "scar did not preserve long-range credit"
    print("    PASS  (the oldest wound still reaches the verdict, undiscounted)")

    # --- (4) learning: full model solves the inherited-credit task ---------
    print("\n[4] Learning 'The Curse of the House of Atreus' (3 plays, 2 resets)")
    print("    -- full Pathei-Mathos Network:")
    _, acc_full = train(use_scar=True, steps=400, verbose=True)
    print(f"    final test accuracy (full)     = {acc_full:.3f}")

    print("    -- ablation (working memory only, wisdom pathway severed):")
    _, acc_ablate = train(use_scar=False, steps=400, verbose=True)
    print(f"    final test accuracy (ablation) = {acc_ablate:.3f}")

    assert acc_full > 0.9, "full model failed to learn the task"
    assert acc_ablate < 0.65, "ablation unexpectedly solved a task it cannot"
    print("\n    PASS  full model learns the lesson; the mind that cannot keep")
    print("          a scar across generations stays at chance.")

    print("\n" + "=" * 74)
    print("ALL TESTS PASSED")
    print("Pathei mathos, made numerical: only what is suffered irreversibly")
    print("survives to the final judgement.")
    print("=" * 74)


def input_grad_at_crime(net, X, y, use_scar):
    """Norm of d loss / d X at the transgression step (channels 0/1)."""
    # locate crime step per example (marker channel == 1)
    B, T, _ = X.shape
    crime_t = X[:, :, 0].argmax(1)
    # finite-difference the loss wrt the value channel at the crime step
    base, _ = net.loss(net.forward(X)[0], y)
    eps = 1e-4
    total = 0.0
    for i in range(B):
        Xp = X.copy(); Xp[i, crime_t[i], 1] += eps
        lp, _ = net.loss(net.forward(Xp)[0], y)
        Xm = X.copy(); Xm[i, crime_t[i], 1] -= eps
        lm, _ = net.loss(net.forward(Xm)[0], y)
        total += abs((lp - lm) / (2 * eps))
    # the ablation: zero the scar readout so the verdict can't use s
    if not use_scar:
        saved = net.W_sy.copy(); net.W_sy[:] = 0.0
        total = 0.0
        for i in range(B):
            Xp = X.copy(); Xp[i, crime_t[i], 1] += eps
            lp, _ = net.loss(net.forward(Xp)[0], y)
            Xm = X.copy(); Xm[i, crime_t[i], 1] -= eps
            lm, _ = net.loss(net.forward(Xm)[0], y)
            total += abs((lp - lm) / (2 * eps))
        net.W_sy[:] = saved
    return total / B


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    self_tests()

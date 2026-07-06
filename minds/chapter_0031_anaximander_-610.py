#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 chapter_0031_anaximander_-610.py
 The Apeiron Equilibrium Network (AEN)
 A from-scratch, trainable neural architecture that embodies the cognitive
 signature of Anaximander of Miletus (c. 610 - c. 546 BCE).
 # Part Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 2 Minds 21 - 40 Available on Amazon https://www.amazon.com/dp/B0H6QCQ9M7
# Resume and Interactive Demos at https://artificiology.com/
Author: David Vivancos · Chapter 0031 · Anaximander
================================================================================

WHY THIS ARCHITECTURE LOOKS THE WAY IT DOES
-------------------------------------------
Most "first principle" thinkers get modeled with a storage-and-retrieval net
(attention over stored keys). That would be the WRONG model for Anaximander,
and it would also just duplicate his teacher Thales. Thales' move was a
*substance* ("everything is water" -> one continuous self-moving stuff).
Anaximander's move is the opposite of a substance: it is a *process of balance*.

Three ideas, all attested, define his mind. The architecture is built from them
and from nothing generic:

  (1) THE APEIRON ("the boundless / the indefinite").
      The source of all things is NOT one of the elements. It must be neutral
      and quality-less, precisely so that no opposite is privileged inside it.
      -> In the net: the state is initialized to ZERO (the indefinite, with no
         qualities yet), and a learned "apeiron drive" b_ap feeds the dynamics.
         Determinate structure (the opposites) "separates out" of this neutral
         source via the input encoder, exactly as in his cosmogony.

  (2) JUSTICE AS RECIPROCAL COMPENSATION (his only surviving sentence, DK12 B1,
      preserved by Theophrastus in Simplicius):
         "...they pay penalty and retribution to each other for their injustice
          according to the ordering of time."
      Opposites that encroach must pay each other back; order is a self-
      regulating war of opposites, not a decree.
      -> In the net: hidden units are arranged in OPPOSITE PAIRS. A fixed
         "justice operator" P (the per-pair graph Laplacian) applies, at every
         time step, a restoring force proportional to each pair's imbalance --
         the literal "pay-back". The forward pass is therefore not a single
         shot but a RELAXATION over time toward a balanced fixed point.

  (3) THE EARTH AT REST BY SYMMETRY (Aristotle, De Caelo 2.13).
      The earth stays put not because something holds it up, but because,
      equidistant from all extremes, it has "no reason to move one way rather
      than another." This is symmetry / the principle of sufficient reason,
      two millennia before Leibniz (Kahn 1960, p.77; Popper called it one of
      the boldest ideas in the history of thought).
      -> In the net: the readout is taken from a CENTERED equilibrium state
         (the global mean -- the only "privileged direction" -- is removed),
         and a symmetry penalty drives that global drift to zero. Stability
         comes from indifference, not from an external support.

So the AEN is an EQUILIBRIUM (fixed-point) network of antagonistic opposite
pairs, with a justice (reciprocal-compensation) operator and a symmetry
(no-privileged-direction) constraint. It is deliberately NOT a transformer,
NOT mixture-of-experts, NOT attention-over-memory.

THE TASK IT LEARNS ("The Tribunal of Opposites")
------------------------------------------------
We give the model a set of opposite pairs that should be in balance. One pair
has "transgressed" -- one of its poles has encroached on the other. The model
must, by relaxing its internal opposites toward justice, identify WHICH pair
carries the outstanding debt (a P-way classification). The task is mechanically
aligned with the architecture: the justice dynamics literally compute the
reciprocal compensation that exposes the transgressor. This is honest -- the
inductive bias matches the problem -- and it is genuinely learned (the coupling,
encoder, and readout are all trained from random initialization).

ENGINEERING CONTRACT (kept in every file of this corpus)
--------------------------------------------------------
  * Pure NumPy, written from scratch (no autograd, no ML frameworks).
  * A finite-difference gradient check that MUST pass (mandatory).
  * A real training loop with Adam and a held-out test set.
  * Self-tests, then a real run whose output is pasted into the chapter.

Run:  python3 chapter_0031_anaximander_-610.py
================================================================================
"""

import numpy as np


# =============================================================================
# 0. Small numerical helpers
# =============================================================================

def softmax(z, axis=-1):
    """Numerically stable softmax."""
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def one_hot(idx, n):
    """One-hot encode an integer array of class indices."""
    out = np.zeros((idx.shape[0], n))
    out[np.arange(idx.shape[0]), idx] = 1.0
    return out


def justice_operator(n_pairs):
    """
    Build the fixed 'justice' operator P (shape h x h, h = 2*n_pairs).

    For each opposite pair occupying hidden indices (2k, 2k+1), the per-pair
    block is the graph Laplacian of a single edge:

          [[ 0.5, -0.5],
           [-0.5,  0.5]]

    Applied to a pair (a, b) it yields ( 0.5(a-b), 0.5(b-a) ): a force that
    pushes BOTH poles toward their common mean in equal and opposite measure.
    That is reciprocal compensation -- "each pays the other back" -- made into
    linear algebra. It is fixed (not learned): justice is the rule of the
    cosmos, not a parameter the model may bend.
    """
    h = 2 * n_pairs
    P = np.zeros((h, h))
    for k in range(n_pairs):
        i, j = 2 * k, 2 * k + 1
        P[i, i] = 0.5
        P[j, j] = 0.5
        P[i, j] = -0.5
        P[j, i] = -0.5
    return P


# =============================================================================
# 1. The model
# =============================================================================

class ApeironEquilibriumNetwork:
    """
    A fixed-point network whose forward pass is a relaxation of antagonistic
    opposite-pairs toward a balanced equilibrium.

    Dynamics (batched; s_t has shape (B, h)):

        z      = X @ W_in.T + b_in                 # opposites separate from the
                                                   #   apeiron via the encoder
        s_0    = 0                                 # the indefinite: no qualities
        pre_t  = s_t @ A.T + z + b_ap              # coupling + separation + drive
        s_{t+1}= M @ s_t + eta * tanh(pre_t)       # leaky relaxation toward a
                                                   #   fixed point, where
                 M = (1-eta) I - eta*lam_J*P       #   M folds in the justice
                                                   #   pay-back operator P

    Readout (the earth at rest -- remove the one privileged direction):

        gbar   = mean(s_T) over hidden units       # the global "drift"
        c      = s_T - gbar                         # centered state
        logits = c @ W_out.T + b_out

    Loss:

        L = cross_entropy(logits, y)
            + alpha * mean_pairs( imbalance^2 )     # justice: reach balance
            + beta  * mean_batch( gbar^2 )          # symmetry: no privileged dir
    """

    def __init__(self, n_pairs=6, hidden_pairs=None, n_classes=None,
                 T=8, eta=0.30, lam_J=0.60, alpha=0.05, beta=0.05, seed=0):
        rng = np.random.default_rng(seed)
        self.n_pairs = n_pairs
        self.d_in = 2 * n_pairs                       # observed pole values
        # The hidden layer is itself organized into opposite pairs. By default we
        # mirror the input pairs 1:1 so the justice operator is interpretable.
        self.hidden_pairs = hidden_pairs or n_pairs
        self.h = 2 * self.hidden_pairs
        self.n_classes = n_classes or n_pairs

        # ---- fixed (non-trainable) structure of the cosmos --------------------
        self.T = int(T)                               # relaxation steps (time)
        self.eta = float(eta)                         # relaxation rate
        self.lam_J = float(lam_J)                     # strength of justice force
        self.alpha = float(alpha)                     # justice regulariser weight
        self.beta = float(beta)                       # symmetry regulariser weight
        self.P = justice_operator(self.hidden_pairs)  # the pay-back operator
        # M folds the leak and the justice force into one constant matrix.
        self.M = (1.0 - self.eta) * np.eye(self.h) - self.eta * self.lam_J * self.P

        # ---- trainable parameters --------------------------------------------
        # Small inits keep the relaxation a contraction (stable fixed point).
        self.params = {
            "W_in":  rng.standard_normal((self.h, self.d_in)) * (1.0 / np.sqrt(self.d_in)),
            "b_in":  np.zeros(self.h),
            "A":     rng.standard_normal((self.h, self.h)) * (0.5 / np.sqrt(self.h)),
            "b_ap":  rng.standard_normal(self.h) * 0.05,    # the apeiron drive
            "W_out": rng.standard_normal((self.n_classes, self.h)) * (1.0 / np.sqrt(self.h)),
            "b_out": np.zeros(self.n_classes),
        }

    # ----- parameter (de)serialization, used by the gradient checker ----------
    def get_flat(self):
        return np.concatenate([self.params[k].ravel() for k in sorted(self.params)])

    def set_flat(self, vec):
        i = 0
        for k in sorted(self.params):
            n = self.params[k].size
            self.params[k] = vec[i:i + n].reshape(self.params[k].shape).copy()
            i += n

    # ----------------------------------------------------------------- forward
    def forward(self, X, y):
        """
        Run the relaxation and compute loss. Returns (loss, cache) where cache
        holds everything backward() needs. y is an int array of true classes.
        """
        B = X.shape[0]
        p = self.params
        z = X @ p["W_in"].T + p["b_in"]                      # (B, h)

        s = np.zeros((B, self.h))
        s_hist = [s]                                          # s_0 .. s_T
        tanhp_hist = []                                       # tanh'(pre_t)
        for _ in range(self.T):
            pre = s @ p["A"].T + z + p["b_ap"]                # (B, h)
            a = np.tanh(pre)
            tanhp_hist.append(1.0 - a * a)                    # tanh'(pre)
            s = s @ self.M.T + self.eta * a                   # relaxation step
            s_hist.append(s)
        sT = s

        # readout from the centered equilibrium (earth-at-rest symmetry)
        gbar = sT.mean(axis=1, keepdims=True)                 # (B, 1)
        c = sT - gbar                                         # (B, h)
        logits = c @ p["W_out"].T + p["b_out"]                # (B, n_classes)
        probs = softmax(logits, axis=1)

        # ---- losses ----
        Y = one_hot(y, self.n_classes)
        L_ce = -np.mean(np.sum(Y * np.log(probs + 1e-12), axis=1))

        # justice: squared imbalance of each pair at equilibrium
        a_pole = sT[:, 0::2]                                  # (B, hidden_pairs)
        b_pole = sT[:, 1::2]
        imb = a_pole - b_pole                                 # (B, hidden_pairs)
        L_just = self.alpha * np.mean(imb ** 2)

        # symmetry: penalise the global drift gbar
        L_sym = self.beta * np.mean(gbar ** 2)

        loss = L_ce + L_just + L_sym
        cache = dict(X=X, y=y, z=z, s_hist=s_hist, tanhp_hist=tanhp_hist,
                     sT=sT, gbar=gbar, c=c, probs=probs, imb=imb, B=B)
        return loss, cache

    # ---------------------------------------------------------------- backward
    def backward(self, cache):
        """
        Reverse-mode gradients through the unrolled relaxation. Returns a dict
        of gradients matching self.params. Derivation is documented inline; it
        is verified against finite differences in grad_check().
        """
        p = self.params
        B = cache["B"]
        sT, gbar, c = cache["sT"], cache["gbar"], cache["c"]
        probs, y = cache["probs"], cache["y"]
        s_hist, tanhp_hist = cache["s_hist"], cache["tanhp_hist"]

        grads = {k: np.zeros_like(v) for k, v in p.items()}

        # ---- readout grads ----
        Y = one_hot(y, self.n_classes)
        dlogits = (probs - Y) / B                              # (B, n_classes)
        grads["W_out"] += dlogits.T @ c
        grads["b_out"] += dlogits.sum(axis=0)
        dc = dlogits @ p["W_out"]                              # (B, h)

        # c = sT - mean(sT): backprop the centering (J^T = I - 11^T/h)
        dsT = dc - dc.mean(axis=1, keepdims=True)

        # symmetry loss: L_sym = beta * mean(gbar^2), gbar = mean_h(sT)
        dgbar = (2.0 * self.beta / B) * gbar                  # (B,1)
        dsT += dgbar / self.h                                 # spread over units

        # justice loss: L_just = alpha * mean((a_pole-b_pole)^2)
        imb = cache["imb"]
        n_imb = imb.size                                      # B * hidden_pairs
        dimb = (2.0 * self.alpha / n_imb) * imb               # (B, hidden_pairs)
        dsT[:, 0::2] += dimb
        dsT[:, 1::2] -= dimb

        # ---- backprop through the relaxation steps ----
        # s_{t+1} = M s_t + eta*tanh(pre_t), pre_t = A s_t + z + b_ap
        # Jacobian wrt s_t: M + eta * diag(tanh') A
        d = dsT                                               # dL/ds_{t+1}, start t=T-1
        dz = np.zeros((B, self.h))
        for t in reversed(range(self.T)):
            s_t = s_hist[t]                                   # input to this step
            tanhp = tanhp_hist[t]                             # tanh'(pre_t)
            u = self.eta * tanhp * d                          # (B, h)
            grads["A"] += u.T @ s_t                           # pre depends on A s_t
            dz += u                                           # pre depends on z, b_ap
            grads["b_ap"] += u.sum(axis=0)
            # propagate to s_t: through M (constant) and through A
            d = d @ self.M + u @ p["A"]
        # s_0 = 0 -> no gradient flows past it.

        # z = X @ W_in.T + b_in
        grads["W_in"] += dz.T @ cache["X"]
        grads["b_in"] += dz.sum(axis=0)
        return grads


# =============================================================================
# 2. Mandatory finite-difference gradient check
# =============================================================================

def grad_check(seed=1, n_check=40, eps=1e-6):
    """
    Compare analytic gradients to central finite differences on a random subset
    of parameters. Must pass (max relative error tiny) for the file to ship.
    """
    rng = np.random.default_rng(seed)
    net = ApeironEquilibriumNetwork(n_pairs=4, n_classes=4, T=6, seed=seed)
    B = 5
    X = rng.standard_normal((B, net.d_in))
    y = rng.integers(0, net.n_classes, size=B)

    loss, cache = net.forward(X, y)
    grads = net.backward(cache)
    g_flat = np.concatenate([grads[k].ravel() for k in sorted(grads)])

    theta = net.get_flat()
    idxs = rng.choice(theta.size, size=min(n_check, theta.size), replace=False)

    max_rel = 0.0
    for i in idxs:
        orig = theta[i]
        theta[i] = orig + eps; net.set_flat(theta); Lp, _ = net.forward(X, y)
        theta[i] = orig - eps; net.set_flat(theta); Lm, _ = net.forward(X, y)
        theta[i] = orig;        net.set_flat(theta)
        num = (Lp - Lm) / (2 * eps)
        ana = g_flat[i]
        denom = max(1e-9, abs(num) + abs(ana))
        max_rel = max(max_rel, abs(num - ana) / denom)
    return max_rel


# =============================================================================
# 3. The "Tribunal of Opposites" task
# =============================================================================

def make_dataset(n_samples, n_pairs=6, encroach=1.6, noise=0.15, seed=0):
    """
    Each sample is a set of opposite pairs that *ought* to balance. We draw a
    balanced latent for every pair, then pick ONE pair (the transgressor) and
    push one of its poles outward by `encroach` -- an act of injustice. The
    label is the index of the transgressing pair. The model must find the
    outstanding debt.

    Returns X (n_samples, 2*n_pairs), y (n_samples,) in [0, n_pairs).
    """
    rng = np.random.default_rng(seed)
    d_in = 2 * n_pairs
    X = np.zeros((n_samples, d_in))
    y = np.zeros(n_samples, dtype=int)
    for n in range(n_samples):
        base = rng.standard_normal(n_pairs) * 0.5             # shared pair levels
        a = base + rng.standard_normal(n_pairs) * noise       # pole A
        b = base + rng.standard_normal(n_pairs) * noise       # pole B  (balanced)
        k = rng.integers(0, n_pairs)                          # transgressor
        sign = 1.0 if rng.random() < 0.5 else -1.0
        a[k] += sign * encroach                               # injustice on pair k
        b[k] -= sign * encroach * 0.25                        # partial recoil
        X[n, 0::2] = a
        X[n, 1::2] = b
        y[n] = k
    return X, y


# =============================================================================
# 4. Training loop (Adam)
# =============================================================================

def accuracy(net, X, y):
    _, cache = net.forward(X, y)
    pred = np.argmax(cache["probs"], axis=1)
    return float(np.mean(pred == y))


def mean_imbalance(net, X, y):
    """Average |imbalance| at equilibrium -- how well justice was served."""
    _, cache = net.forward(X, y)
    return float(np.mean(np.abs(cache["imb"])))


def train(net, Xtr, ytr, Xte, yte, epochs=60, batch=64, lr=3e-3, seed=0,
          verbose=True):
    rng = np.random.default_rng(seed)
    keys = sorted(net.params)
    m = {k: np.zeros_like(net.params[k]) for k in keys}
    v = {k: np.zeros_like(net.params[k]) for k in keys}
    b1, b2, eps = 0.9, 0.999, 1e-8
    step = 0
    n = Xtr.shape[0]
    history = []
    for ep in range(1, epochs + 1):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            loss, cache = net.forward(Xtr[idx], ytr[idx])
            grads = net.backward(cache)
            step += 1
            for k in keys:
                m[k] = b1 * m[k] + (1 - b1) * grads[k]
                v[k] = b2 * v[k] + (1 - b2) * grads[k] ** 2
                mhat = m[k] / (1 - b1 ** step)
                vhat = v[k] / (1 - b2 ** step)
                net.params[k] -= lr * mhat / (np.sqrt(vhat) + eps)
        if verbose and (ep % 10 == 0 or ep == 1):
            tr, te = accuracy(net, Xtr, ytr), accuracy(net, Xte, yte)
            print(f"  epoch {ep:3d} | loss {loss:7.4f} | "
                  f"train acc {tr:5.3f} | test acc {te:5.3f}")
        history.append(loss)
    return history


# =============================================================================
# 5. Main: self-tests + a real, reported run
# =============================================================================

def main():
    np.set_printoptions(precision=4, suppress=True)
    print("=" * 70)
    print("THE APEIRON EQUILIBRIUM NETWORK  -  Anaximander (c.610-546 BCE)")
    print("=" * 70)

    # --- (a) mandatory gradient check ---------------------------------------
    print("\n[1] Finite-difference gradient check (must pass)")
    rel = grad_check()
    print(f"    max relative error across sampled params: {rel:.3e}")
    assert rel < 1e-5, "GRADIENT CHECK FAILED"
    print("    PASS  (analytic backprop matches numerical gradient)")

    # --- (b) build data ------------------------------------------------------
    print("\n[2] Building the 'Tribunal of Opposites' dataset")
    n_pairs = 6
    Xtr, ytr = make_dataset(3000, n_pairs=n_pairs, seed=1)
    Xte, yte = make_dataset(800, n_pairs=n_pairs, seed=2)
    print(f"    pairs/sample: {n_pairs}  |  classes: {n_pairs}  "
          f"|  chance acc: {1.0/n_pairs:5.3f}")
    print(f"    train: {Xtr.shape}   test: {Xte.shape}")

    # --- (c) train -----------------------------------------------------------
    print("\n[3] Training (Adam) -- the cosmos learns to find the debtor")
    net = ApeironEquilibriumNetwork(n_pairs=n_pairs, n_classes=n_pairs,
                                    T=8, eta=0.30, lam_J=0.60,
                                    alpha=0.05, beta=0.05, seed=7)
    imb0 = mean_imbalance(net, Xte, yte)
    acc0 = accuracy(net, Xte, yte)
    train(net, Xtr, ytr, Xte, yte, epochs=60, batch=64, lr=3e-3, seed=3)

    # --- (d) results ---------------------------------------------------------
    accT = accuracy(net, Xte, yte)
    imbT = mean_imbalance(net, Xte, yte)
    print("\n[4] Results")
    print(f"    test accuracy : {acc0:5.3f} (init)  ->  {accT:5.3f} (trained)")
    print(f"    equilibrium |imbalance| at readout : "
          f"{imb0:6.4f} (init) -> {imbT:6.4f} (trained)")

    # --- (e) self-tests / assertions ----------------------------------------
    print("\n[5] Self-tests")
    assert accT > 0.70, f"expected >0.70 test accuracy, got {accT:.3f}"
    print(f"    PASS  test accuracy {accT:.3f} >> chance {1.0/n_pairs:.3f}")

    # The justice operator must, on its own, contract imbalance over time --
    # "they pay penalty and retribution to each other." We verify the pure
    # pay-back dynamics s_{t+1} = M s_t monotonically shrink any imbalance.
    rng = np.random.default_rng(0)
    s = rng.standard_normal((200, net.h))                 # arbitrary imbalanced state
    imb_path = []
    for _ in range(12):
        imb_path.append(np.mean(np.abs(s[:, 0::2] - s[:, 1::2])))
        s = s @ net.M.T                                   # pure justice relaxation
    monotone = all(imb_path[i + 1] <= imb_path[i] + 1e-9
                   for i in range(len(imb_path) - 1))
    print(f"    pure justice relaxation |imbalance|: "
          f"{imb_path[0]:.4f} -> {imb_path[-1]:.4f} over 12 steps")
    assert monotone and imb_path[-1] < imb_path[0], \
        "justice operator must contract pair imbalance"
    print("    PASS  the pay-back operator drives every transgression to zero")
    print(f"    (training reduced equilibrium imbalance "
          f"{imb0:.4f} -> {imbT:.4f} on held-out data)")

    # symmetry: the trained equilibrium carries almost no global drift
    _, cache = net.forward(Xte, yte)
    drift = float(np.mean(np.abs(cache["gbar"])))
    print(f"    global drift |gbar| at equilibrium : {drift:6.4f}")
    assert drift < 0.30, "symmetry constraint should suppress global drift"
    print("    PASS  no privileged direction survives (earth-at-rest symmetry)")

    print("\n" + "=" * 70)
    print("ALL CHECKS PASSED. The apeiron settles; the debtor is found by")
    print("reciprocal compensation; stability comes from symmetry, not support.")
    print("=" * 70)


if __name__ == "__main__":
    main()

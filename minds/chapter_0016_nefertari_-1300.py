#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chapter_0016_nefertari_-1300.py
================================================================================
NEFERTARI  ·  Mind #16  ·  c. 1300 - c. 1250 BCE  ·  Egypt (19th Dynasty)
The Parity Coupler  --  a Reciprocal Recognition Network
 Part Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 1 Minds 1 - 20 Available on Amazon https://www.amazon.com/dp/B0H6F9L324
Resume and Interactive Demos at https://artificiology.com/
================================================================================

WHY THIS ARCHITECTURE (and not a Transformer)
--------------------------------------------------------------------------------
Almost nothing of Nefertari's inner voice survives -- but two hard artifacts do,
and they point at ONE cognitive operation that is hers and no one else's in this
corpus:

  (1) The clay letter from "Naptera" (Nefertari) to the Hittite queen Puduhepa,
      preserved in cuneiform at Hattusa. She does not command; she addresses
      Puduhepa as "my sister" and wishes "may your country be well." This opens
      a LATERAL queen-to-queen channel that runs parallel to the kings'
      "brother-to-brother" channel. The relationship is built by *symmetric,
      reciprocal address*.

  (2) The Small Temple at Abu Simbel, where -- uniquely in Egyptian monumental
      art -- Nefertari's colossi stand at the SAME HEIGHT as Ramesses'. Consorts
      were normally carved knee-high. Here the two parties are rendered as
      equals; inside, she stands "very much equal" before the gods.

Both artifacts encode the same idea: intelligence as PARITY-CONSTRUCTION. Not
domination (rank one party above the other), not even persuasion, but the
deliberate manufacture of a *symmetric mutual-recognition bond* across a gap the
surrounding order treats as unequal -- so that two unlike minds become legible,
and legitimizing, peers.

We translate that idea into mathematics. Two "courts" (call them Egypt and Hatti)
talk about the SAME meanings, but each encodes meaning in its own rotated
coordinate frame -- a different "language." The model must discover the coupling
that lets each court READ the other. Nefertari's insight becomes three coupled
constraints on that coupling C:

  * RECOGNITION  -- map my latent into your frame and land on your latent
                    (and vice-versa). Each party can read the other.
  * RECIPROCITY  -- the channel is one operator used both ways: C forward,
                    C^T back. A->B->A must return you UNCHANGED (cycle identity).
                    Giving and receiving are the same act, transposed.
  * PARITY       -- C is pushed toward an ISOMETRY (C^T C = I). An isometry
                    distorts neither party: it is the equal-height colossi in
                    linear-algebra form. Symmetry is the precondition for trust.

Autoencoding (each court must reconstruct its own messages) prevents the trivial
"agree on nothing" collapse: a real bond must preserve real information.

This is deliberately NOT attention-over-stored-keys. The unit of cognition is the
DYAD and the learned symmetry between its members -- exactly Nefertari's unit.

CONVENTIONS (shared across the whole terabook)
--------------------------------------------------------------------------------
  * Pure NumPy, from scratch. No autograd, no ML frameworks.
  * Every parameter has a hand-derived gradient, checked against central finite
    differences (MANDATORY -- see gradient_check()).
  * A real training loop on a real (synthetic but structured) task.
  * Self-tests at the bottom; the file executes end to end and prints results.

Run:  python3 chapter_0016_nefertari_-1300.py
Author: David Vivancos · Chapter 0016 · Nefertari
================================================================================
"""

import numpy as np
from typing import Dict, Tuple, List


# =============================================================================
# 0.  Small numeric helpers
# =============================================================================
def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def dtanh(y: np.ndarray) -> np.ndarray:
    """Derivative of tanh given its OUTPUT y = tanh(x):  1 - y^2."""
    return 1.0 - y * y


# =============================================================================
# 1.  Parameter container
# =============================================================================
# The model has, for each court (X = Egypt, Y = Hatti):
#   an encoder MLP   surface_form -> latent "meaning"  (1 hidden tanh layer)
#   a linear decoder latent       -> surface_form       (autoencoder head)
# and ONE shared coupling matrix C that bridges the two latent frames.
#
# Keeping every weight in a flat dict makes the finite-difference gradient check
# trivial: we can perturb any single scalar and recompute the loss.

def init_params(d_in: int, h: int, d: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """He-style init for tanh layers; small init for the coupling C near I."""
    def he(shape, fan_in):
        return rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)

    p: Dict[str, np.ndarray] = {}
    # ---- Egypt (X) encoder: x(d_in) -> hidden(h) -> latent(d)
    p["W1x"] = he((d_in, h), d_in);  p["b1x"] = np.zeros(h)
    p["W2x"] = he((h, d), h);        p["b2x"] = np.zeros(d)
    # ---- Hatti (Y) encoder
    p["W1y"] = he((d_in, h), d_in);  p["b1y"] = np.zeros(h)
    p["W2y"] = he((h, d), h);        p["b2y"] = np.zeros(d)
    # ---- Decoders (linear): latent(d) -> surface(d_in)
    p["Dx"] = he((d, d_in), d);      p["bdx"] = np.zeros(d_in)
    p["Dy"] = he((d, d_in), d);      p["bdy"] = np.zeros(d_in)
    # ---- Coupling C: starts near identity (a relationship begins as "you ~ me")
    p["C"] = np.eye(d) + 0.01 * rng.standard_normal((d, d))
    return p


# =============================================================================
# 2.  Forward pass + cached intermediates
# =============================================================================
def encode(x: np.ndarray, W1, b1, W2, b2) -> Tuple[np.ndarray, np.ndarray]:
    """One court's encoder. Returns (latent z, hidden activation a) for backprop.
    x: (N, d_in)  ->  a=tanh(x W1 + b1): (N,h)  ->  z = a W2 + b2: (N,d)."""
    a = tanh(x @ W1 + b1)
    z = a @ W2 + b2
    return z, a


def forward(p: Dict[str, np.ndarray],
            X: np.ndarray, Y: np.ndarray,
            weights: Dict[str, float]) -> Tuple[float, Dict[str, np.ndarray]]:
    """
    Full forward pass over a paired batch.
      X: (N, d_in) Egyptian surface forms
      Y: (N, d_in) Hittite surface forms   (paired: row i = same meaning)
    Returns scalar loss and a cache for the backward pass.
    """
    N = X.shape[0]
    d = p["C"].shape[0]
    I = np.eye(d)

    # ----- encode both courts into their own latent frames
    zx, ax = encode(X, p["W1x"], p["b1x"], p["W2x"], p["b2x"])   # (N,d)
    zy, ay = encode(Y, p["W1y"], p["b1y"], p["W2y"], p["b2y"])   # (N,d)

    C = p["C"]
    # ----- RECOGNITION: translate each into the other's frame
    zx2y = zx @ C            # Egypt meaning expressed in Hatti's frame   (N,d)
    zy2x = zy @ C.T          # Hatti meaning expressed in Egypt's frame   (N,d)
    r_xy = zx2y - zy         # should be ~0 : Hatti can read Egypt
    r_yx = zy2x - zx         # should be ~0 : Egypt can read Hatti
    L_recog = (np.sum(r_xy**2) + np.sum(r_yx**2)) / N

    # ----- RECONSTRUCTION (autoencode -> prevents collapse to nothing)
    xhat = zx @ p["Dx"] + p["bdx"]
    yhat = zy @ p["Dy"] + p["bdy"]
    e_x = xhat - X
    e_y = yhat - Y
    L_recon = (np.sum(e_x**2) + np.sum(e_y**2)) / N

    # ----- RECIPROCITY: round-trip identity  zx C C^T = zx ,  zy C^T C = zy
    zx_rt = zx @ C @ C.T
    zy_rt = zy @ C.T @ C
    c_x = zx_rt - zx
    c_y = zy_rt - zy
    L_cycle = (np.sum(c_x**2) + np.sum(c_y**2)) / N

    # ----- PARITY: isometry constraint  C^T C = I  (the equal-height colossi)
    M = C.T @ C - I
    L_parity = np.sum(M**2)

    L = (weights["recog"]  * L_recog +
         weights["recon"]  * L_recon +
         weights["cycle"]  * L_cycle +
         weights["parity"] * L_parity)

    cache = dict(N=N, I=I, C=C,
                 X=X, Y=Y, zx=zx, zy=zy, ax=ax, ay=ay,
                 zx2y=zx2y, zy2x=zy2x, r_xy=r_xy, r_yx=r_yx,
                 xhat=xhat, yhat=yhat, e_x=e_x, e_y=e_y,
                 c_x=c_x, c_y=c_y, M=M,
                 L_recog=L_recog, L_recon=L_recon,
                 L_cycle=L_cycle, L_parity=L_parity)
    return float(L), cache


# =============================================================================
# 3.  Backward pass  (hand-derived gradients for every parameter)
# =============================================================================
def backward(p: Dict[str, np.ndarray],
             cache: Dict[str, np.ndarray],
             weights: Dict[str, float]) -> Dict[str, np.ndarray]:
    """
    Returns grads with the same keys as p.
    We accumulate dL/dzx and dL/dzy from every loss term, then backprop each
    encoder; C gets gradients from recognition, reciprocity and parity.
    """
    N   = cache["N"]
    C   = cache["C"]
    zx  = cache["zx"];  zy  = cache["zy"]
    ax  = cache["ax"];  ay  = cache["ay"]
    X   = cache["X"];   Y   = cache["Y"]

    wr  = weights["recog"]; wc = weights["recon"]
    wy  = weights["cycle"]; wp = weights["parity"]

    # latent-gradient accumulators
    dzx = np.zeros_like(zx)
    dzy = np.zeros_like(zy)
    dC  = np.zeros_like(C)

    # ---- RECOGNITION ----------------------------------------------------------
    # L_recog = (||zx C - zy||^2 + ||zy C^T - zx||^2)/N
    r_xy = cache["r_xy"]; r_yx = cache["r_yx"]
    g = wr * (2.0 / N)
    # d/dzx of ||zx C - zy||^2  = 2 (zx C - zy) C^T
    dzx += g * (r_xy @ C.T)
    dzy += g * (-r_xy)
    # d/dzy of ||zy C^T - zx||^2 = 2 (zy C^T - zx) C
    dzy += g * (r_yx @ C)
    dzx += g * (-r_yx)
    # C grads: term1 wrt C = zx^T r_xy ; term2 wrt C^T = zy^T r_yx -> wrt C its transpose
    dC += g * (zx.T @ r_xy)
    dC += g * (zy.T @ r_yx).T

    # ---- RECONSTRUCTION -------------------------------------------------------
    # L_recon = (||zx Dx + bdx - X||^2 + ||zy Dy + bdy - Y||^2)/N
    e_x = cache["e_x"]; e_y = cache["e_y"]
    ge = wc * (2.0 / N)
    dDx  = ge * (zx.T @ e_x)
    dbdx = ge * np.sum(e_x, axis=0)
    dDy  = ge * (zy.T @ e_y)
    dbdy = ge * np.sum(e_y, axis=0)
    dzx += ge * (e_x @ p["Dx"].T)
    dzy += ge * (e_y @ p["Dy"].T)

    # ---- RECIPROCITY (cycle) --------------------------------------------------
    # L_cycle = (||zx C C^T - zx||^2 + ||zy C^T C - zy||^2)/N
    #   c_x = zx C C^T - zx ,  c_y = zy C^T C - zy
    c_x = cache["c_x"]; c_y = cache["c_y"]
    gc = wy * (2.0 / N)
    P  = C @ C.T          # symmetric
    Q  = C.T @ C          # symmetric
    # latent grads: f = zx P - zx ; df/dzx = c_x (P - I)   (P, Q symmetric)
    dzx += gc * (c_x @ (P - cache["I"]))
    dzy += gc * (c_y @ (Q - cache["I"]))
    # C grads (hand-derived; both fold to clean symmetric forms):
    #   d/dC ||zx C C^T - zx||^2 = 2 (zx^T c_x + c_x^T zx) C
    S1 = zx.T @ c_x
    dC += gc * ((S1 + S1.T) @ C)
    #   d/dC ||zy C^T C - zy||^2 = 2 C (zy^T c_y + c_y^T zy)
    S2 = zy.T @ c_y
    dC += gc * (C @ (S2 + S2.T))

    # ---- PARITY ---------------------------------------------------------------
    # L_parity = ||C^T C - I||_F^2 ;  dL/dC = 2 C (C^T C - I) * 2 ... derive:
    #   M = C^T C - I ;  L = sum(M^2).  dL/dC = 2 C (M + M^T) = 4 C M (M symmetric)
    M = cache["M"]
    dC += wp * (4.0 * (C @ M))

    # ---- backprop through encoders -------------------------------------------
    # z = a W2 + b2 ;  a = tanh(x W1 + b1)
    def enc_back(dz, a, x, W2):
        dW2 = a.T @ dz
        db2 = np.sum(dz, axis=0)
        da  = dz @ W2.T
        dpre = da * dtanh(a)          # through tanh
        dW1 = x.T @ dpre
        db1 = np.sum(dpre, axis=0)
        return dW1, db1, dW2, db2

    dW1x, db1x, dW2x, db2x = enc_back(dzx, ax, X, p["W2x"])
    dW1y, db1y, dW2y, db2y = enc_back(dzy, ay, Y, p["W2y"])

    return {
        "W1x": dW1x, "b1x": db1x, "W2x": dW2x, "b2x": db2x,
        "W1y": dW1y, "b1y": db1y, "W2y": dW2y, "b2y": db2y,
        "Dx": dDx, "bdx": dbdx, "Dy": dDy, "bdy": dbdy,
        "C": dC,
    }


# =============================================================================
# 4.  Finite-difference gradient check   (MANDATORY)
# =============================================================================
def gradient_check(seed: int = 0) -> float:
    """
    Compares analytic gradients to central finite differences on a tiny model.
    Returns the maximum relative error across all parameters. Must be ~1e-6.
    """
    rng = np.random.default_rng(seed)
    d_in, h, d, N = 5, 6, 4, 7
    p = init_params(d_in, h, d, rng)
    X = rng.standard_normal((N, d_in))
    Y = rng.standard_normal((N, d_in))
    weights = dict(recog=1.0, recon=0.7, cycle=0.5, parity=0.3)

    _, cache = forward(p, X, Y, weights)
    grads = backward(p, cache, weights)

    eps = 1e-6
    max_rel = 0.0
    for name in p:
        flat = p[name].ravel()
        g_an = grads[name].ravel()
        # check a handful of coordinates per tensor (full check is overkill)
        idxs = range(flat.size) if flat.size <= 24 else \
               rng.choice(flat.size, size=24, replace=False)
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps
            Lp, _ = forward(p, X, Y, weights)
            flat[i] = orig - eps
            Lm, _ = forward(p, X, Y, weights)
            flat[i] = orig
            g_num = (Lp - Lm) / (2 * eps)
            denom = max(1.0, abs(g_num) + abs(g_an[i]))
            rel = abs(g_num - g_an[i]) / denom
            max_rel = max(max_rel, rel)
    return max_rel


# =============================================================================
# 5.  The "Two Courts" task
# =============================================================================
# Both courts discuss the SAME meanings m in R^d, but Hatti encodes meaning in a
# ROTATED frame R (an orthogonal matrix). Egypt sees a nonlinear surface form of
# m; Hatti sees a nonlinear surface form of m R. A faithful model must discover a
# coupling C ~ R that makes the two courts mutually legible -- i.e., it must
# build the isometry/parity between them. This is Nefertari's operation, staged.

def random_orthogonal(d: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.standard_normal((d, d))
    Q, Rm = np.linalg.qr(A)
    Q *= np.sign(np.diag(Rm))          # make QR deterministic-ish
    return Q


def make_task(n: int, d_in: int, d: int, rng: np.random.Generator):
    """Returns (X, Y, R, Px, Py). Row i of X and Y is the same meaning."""
    R  = random_orthogonal(d, rng)                       # Hatti's rotated frame
    Px = rng.standard_normal((d, d_in)) / np.sqrt(d)     # Egypt surface map
    Py = rng.standard_normal((d, d_in)) / np.sqrt(d)     # Hatti surface map
    M  = rng.standard_normal((n, d))                     # shared meanings
    X  = np.tanh(M @ Px)                                 # Egypt's messages
    Y  = np.tanh((M @ R) @ Py)                           # Hatti's messages
    return X, Y, R, Px, Py


# =============================================================================
# 6.  Training loop  (full-batch Adam, pure NumPy)
# =============================================================================
def train(p, X, Y, weights, steps=1500, lr=3e-3, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    # Adam state
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(v) for k, v in p.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    history: List[float] = []

    for t in range(1, steps + 1):
        L, cache = forward(p, X, Y, weights)
        grads = backward(p, cache, weights)
        for k in p:
            m[k] = b1 * m[k] + (1 - b1) * grads[k]
            v[k] = b2 * v[k] + (1 - b2) * grads[k] ** 2
            mhat = m[k] / (1 - b1 ** t)
            vhat = v[k] / (1 - b2 ** t)
            p[k] -= lr * mhat / (np.sqrt(vhat) + eps)
        history.append(L)
        if verbose and (t == 1 or t % 250 == 0):
            _, c = forward(p, X, Y, weights)
            parity_gap = np.sqrt(c["L_parity"])
            print(f"  step {t:4d} | loss {L:8.4f} | recog {c['L_recog']:7.4f} "
                  f"| recon {c['L_recon']:7.4f} | parity_gap {parity_gap:6.4f}")
    return history


# =============================================================================
# 7.  Evaluation: can each court READ the other?
# =============================================================================
def recognition_accuracy(p, X, Y) -> float:
    """
    For each Egyptian message, translate it into Hatti's frame (zx @ C) and find
    the nearest Hittite latent zy. Accuracy = fraction matched to the correct
    (paired) Hittite message. Chance = 1/N.
    """
    zx, _ = encode(X, p["W1x"], p["b1x"], p["W2x"], p["b2x"])
    zy, _ = encode(Y, p["W1y"], p["b1y"], p["W2y"], p["b2y"])
    zx2y = zx @ p["C"]                       # Egypt expressed in Hatti's frame
    # pairwise distances zx2y (N,d) vs zy (N,d)
    d2 = (np.sum(zx2y**2, axis=1)[:, None]
          + np.sum(zy**2, axis=1)[None, :]
          - 2.0 * zx2y @ zy.T)
    pred = np.argmin(d2, axis=1)
    return float(np.mean(pred == np.arange(X.shape[0])))


def cycle_error(p, X) -> float:
    """Mean round-trip error zx C C^T vs zx -- reciprocity made measurable."""
    zx, _ = encode(X, p["W1x"], p["b1x"], p["W2x"], p["b2x"])
    rt = zx @ p["C"] @ p["C"].T
    return float(np.mean(np.sum((rt - zx) ** 2, axis=1)))


# =============================================================================
# 8.  Self-tests / demonstration
# =============================================================================
def main():
    print("=" * 78)
    print("NEFERTARI  --  The Parity Coupler (Reciprocal Recognition Network)")
    print("Mind #16  ·  parity-construction: making two unequal courts into peers")
    print("=" * 78)

    # ---- (A) GRADIENT CHECK -------------------------------------------------
    print("\n[A] Finite-difference gradient check")
    max_rel = gradient_check(seed=1)
    print(f"    max relative error = {max_rel:.3e}")
    assert max_rel < 1e-5, "Gradient check FAILED"
    print("    PASS  (analytic gradients match finite differences)")

    # ---- (B) BUILD TASK -----------------------------------------------------
    rng = np.random.default_rng(7)
    d_in, h, d = 16, 24, 8
    n_train, n_test = 256, 128
    Xtr, Ytr, R, Px, Py = make_task(n_train, d_in, d, rng)
    Xte, Yte, *_ = make_task(n_test, d_in, d, rng)   # fresh meanings, same task family
    # NOTE: re-use the SAME R/Px/Py so test pairs share the courts' languages
    Mte = rng.standard_normal((n_test, d))
    Xte = np.tanh(Mte @ Px)
    Yte = np.tanh((Mte @ R) @ Py)

    weights = dict(recog=1.0, recon=0.5, cycle=0.4, parity=0.5)
    p = init_params(d_in, h, d, rng)

    print("\n[B] The Two-Courts task")
    print(f"    {n_train} paired messages | latent d={d} | hidden h={h}")
    print(f"    Egypt and Hatti encode the same meanings in frames rotated by R.")

    acc0 = recognition_accuracy(p, Xte, Yte)
    print(f"    recognition accuracy BEFORE training: {acc0:6.3f}  (chance={1/n_test:.3f})")

    # ---- (C) TRAIN ----------------------------------------------------------
    print("\n[C] Training (full-batch Adam)")
    hist = train(p, Xtr, Ytr, weights, steps=1500, lr=3e-3, seed=2)

    # ---- (D) RESULTS --------------------------------------------------------
    print("\n[D] Results")
    L0, Lf = hist[0], hist[-1]
    acc_tr = recognition_accuracy(p, Xtr, Ytr)
    acc_te = recognition_accuracy(p, Xte, Yte)
    cyc    = cycle_error(p, Xte)
    _, cache = forward(p, Xtr, Ytr, weights)
    parity_gap = np.sqrt(cache["L_parity"])

    print(f"    loss: {L0:8.4f}  ->  {Lf:8.4f}   ({L0/Lf:6.1f}x reduction)")
    print(f"    recognition accuracy  train={acc_tr:6.3f}  test={acc_te:6.3f}")
    print(f"    parity gap ||C^T C - I||_F : {parity_gap:.4f}  (-> 0 == equal-height colossi)")
    print(f"    reciprocity cycle error    : {cyc:.4e}  (-> 0 == A->B->A returns unchanged)")

    # how close is the learned coupling to the TRUE rotation R (up to sign)?
    align = np.abs(p["C"] @ R.T)          # ~ identity-ish if C recovered R
    diag_mass = np.mean(np.max(align, axis=1))
    print(f"    learned C vs true frame R  : peak-alignment {diag_mass:.3f} (1.0 == exact)")

    # ---- (E) ASSERTIONS -----------------------------------------------------
    print("\n[E] Self-test assertions")
    assert Lf < L0 / 5.0,        "training did not reduce loss enough"
    assert acc_te > 0.80,        "model cannot read the other court (low recognition)"
    assert parity_gap < 0.15,    "coupling never became an isometry (no parity)"
    assert cyc < 1e-1,           "channel is not reciprocal (cycle error high)"
    print("    PASS  loss reduced, recognition learned, parity reached, channel reciprocal.")

    print("\n" + "=" * 78)
    print("Nefertari's operation, executed: two courts that began mutually illegible")
    print("now read each other through a learned ISOMETRY -- equals by construction.")
    print("=" * 78)


if __name__ == "__main__":
    np.random.seed(42)
    main()

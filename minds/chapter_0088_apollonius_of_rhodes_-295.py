#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
THE AMECHANIA ENGINE
A from-scratch, trainable cognitive architecture after Apollonius of Rhodes
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 5 Minds 81 - 100 Available on Amazon https://www.amazon.com/dp/B0H7LP5LP2
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 0088 · Apollonius of Rhodes
================================================================================

WHY THIS ARCHITECTURE EXISTS
----------------------------
Apollonius (c. 295-215 BCE), chief librarian of Alexandria and author of the
four-book Argonautica, gave epic a hero unlike Homer's. Where Odysseus is
POLU-mechanos -- "of many devices," a mind already stocked with stratagems --
Apollonius makes Jason A-mechanos: literally "without device," repeatedly
"at a loss," overwhelmed by the enormity of his task (Hunter 1993; Dickinson
College Commentaries). Jason does not solve the dragon, the bulls, or the
earthborn men by private cleverness. He solves them by (1) DELEGATING to a
heterogeneous crew of specialists, (2) RECALLING precedent from the vast
Alexandrian archive that Apollonius packs into the poem as aetiology, and
(3) accepting help from Medea, whose mind is the first sustained portrait in
Greek epic of DELIBERATION UNDER CONFLICTING DRIVES -- shame/restraint (aidos)
warring against desire (pothos) until a decision finally settles.

So the Apollonian theory of intelligence is NOT "cleverness imposes order."
It is the opposite: intelligence BEGINS in helplessness. A mind that always
already has the answer never has to think. The Amechania Engine encodes this:

    1. CREW          -- a routed ensemble of specialist experts (delegation)
    2. AETIOLOGY     -- a content-addressable archive of precedent (recall)
    3. MEDEA         -- an unrolled settling dynamic that resolves a conflict
                        between two opposed value drives (deliberation)
    4. AMECHANIA     -- a metacognitive gate that measures the model's own
                        helplessness and decides HOW MUCH to defer from direct
                        action to archive+deliberation.

The gate is the whole point. When the engine is confident it acts straight
from the crew. When it is "at a loss" (high amechania) it leans on the archive
and on Medea's deliberation -- exactly Jason's arc from paralysis to contrivance.

This file is pure NumPy. Every gradient is derived and written by hand, then
verified against central finite differences (mandatory check below). A real
training loop is included on a synthetic task whose label can ONLY be produced
by resolving a drive-conflict plus recalling a stored precedent -- so the parts
must actually cooperate for the loss to fall.

Run:  python3 chapter_0088_apollonius_of_rhodes_-295.py
================================================================================
"""

import numpy as np

# Reproducibility. 88 = the figure's index in the corpus.
RNG = np.random.default_rng(88)


# ============================================================================
# SMALL DIFFERENTIABLE PRIMITIVES
# ============================================================================
def softmax(z, axis=-1):
    """Numerically stable softmax along `axis`."""
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def tanh(z):
    return np.tanh(z)


# ============================================================================
# PARAMETER INITIALISATION
# ============================================================================
def init_params(d_in, d, d_out, K, M, T, scale=0.3):
    """
    Allocate every learnable tensor.

      d_in : dimension of the situation vector x (the "predicament")
      d    : hidden width shared by crew, archive values, and deliberation
      d_out: dimension of the committed response
      K    : number of crew specialists (the named Argonauts)
      M    : number of archive slots (aetiological precedents in the Library)
      T    : deliberation depth (how long Medea's mind oscillates before it settles)
    """
    def r(*shape):
        return RNG.standard_normal(shape) * scale

    p = {
        # --- CREW: K specialist experts + a router that delegates among them ---
        "Wk": r(K, d, d_in),      # each expert's linear map
        "bk": np.zeros((K, d)),
        "Wr": r(K, d_in),         # router logits (who is fit for this task?)
        "br": np.zeros(K),

        # --- AETIOLOGY: content-addressable archive (keys, values) ---
        "Kmem": r(M, d_in),       # what each precedent is "about"
        "Vmem": r(M, d),          # what each precedent contributes

        # --- MEDEA: two opposed drives + an arbitration gate, settled over T steps ---
        "Ws": r(d, d), "bs": np.zeros(d),   # aidos  -- restraint / duty
        "Wd": r(d, d), "bd": np.zeros(d),   # pothos -- desire / pursuit
        "Wg": r(d, d), "bg": np.zeros(d),   # arbiter between them

        # --- AMECHANIA: scalar self-assessed helplessness gate ---
        "wa": r(d_in), "ba": np.zeros(1),

        # --- OUTPUT HEAD: commit the deliberated state to a response ---
        "Wo": r(d_out, d), "bo": np.zeros(d_out),
    }
    p["_dims"] = dict(d_in=d_in, d=d, d_out=d_out, K=K, M=M, T=T)
    return p


# ============================================================================
# FORWARD PASS
# ============================================================================
def forward(p, X):
    """
    X : (B, d_in) batch of predicaments.
    Returns Y (B, d_out) and a cache of every intermediate needed for backprop.
    """
    d_in, d, K, M, T = (p["_dims"][k] for k in ("d_in", "d", "K", "M", "T"))
    B = X.shape[0]

    # ---- 1. CREW: each specialist proposes; the router weights them ----------
    # Hk[k] = tanh(X Wk[k]^T + bk[k]) is specialist k's reading of the situation.
    A_crew = np.einsum("bi,kdi->bkd", X, p["Wk"]) + p["bk"][None]   # (B,K,d)
    Hk = tanh(A_crew)
    Glog = X @ p["Wr"].T + p["br"]                                  # (B,K) router logits
    Wts = softmax(Glog, axis=1)                                     # (B,K) delegation weights
    C_crew = np.einsum("bk,bkd->bd", Wts, Hk)                       # (B,d) crew consensus

    # ---- 2. AETIOLOGY: recall precedent by content similarity ----------------
    S = (X @ p["Kmem"].T) / np.sqrt(d_in)                           # (B,M) match scores
    Att = softmax(S, axis=1)                                        # (B,M) which precedents fire
    R = Att @ p["Vmem"]                                            # (B,d) recalled contribution

    # ---- 3. context handed to deliberation -----------------------------------
    Cc = C_crew + R                                                 # (B,d)

    # ---- 4. MEDEA: settle the conflict between aidos and pothos over T steps --
    Z = Cc
    delib = []  # store per-step intermediates for backprop
    for _ in range(T):
        Sd = tanh(Z @ p["Ws"].T + p["bs"])     # aidos drive  (restraint)
        Pd = tanh(Z @ p["Wd"].T + p["bd"])     # pothos drive (desire)
        Gt = sigmoid(Z @ p["Wg"].T + p["bg"])  # arbiter: how far toward desire
        Znew = Gt * Pd + (1.0 - Gt) * Sd       # the soul leans, then commits
        delib.append((Z, Sd, Pd, Gt))
        Z = Znew
    ZT = Z                                                          # committed state

    # ---- 5. AMECHANIA: how helpless are we? defer accordingly ----------------
    u = X @ p["wa"] + p["ba"]                                       # (B,)
    alpha = sigmoid(u).reshape(B, 1)                                # (B,1) in (0,1)
    F = (1.0 - alpha) * C_crew + alpha * ZT                         # blend act vs. deliberate

    # ---- 6. commit to a response ---------------------------------------------
    Y = F @ p["Wo"].T + p["bo"]                                     # (B,d_out)

    cache = dict(X=X, Hk=Hk, Wts=Wts, Glog=Glog, C_crew=C_crew,
                 Att=Att, R=R, Cc=Cc, delib=delib, ZT=ZT,
                 alpha=alpha, F=F, Y=Y)
    return Y, cache


def mse_loss(Y, Tgt):
    """Mean (over batch) of half-squared-error summed over outputs."""
    B = Y.shape[0]
    diff = Y - Tgt
    L = 0.5 * np.sum(diff * diff) / B
    dY = diff / B
    return L, dY


# ============================================================================
# BACKWARD PASS  (every gradient derived by hand)
# ============================================================================
def backward(p, cache, dY):
    d_in, d, K, M, T = (p["_dims"][k] for k in ("d_in", "d", "K", "M", "T"))
    X, Hk, Wts, C_crew = cache["X"], cache["Hk"], cache["Wts"], cache["C_crew"]
    Att, Vmem = cache["Att"], p["Vmem"]
    delib, ZT, alpha, F = cache["delib"], cache["ZT"], cache["alpha"], cache["F"]
    B = X.shape[0]
    g = {k: np.zeros_like(v) for k, v in p.items() if k != "_dims"}

    # ---- output head ----------------------------------------------------------
    g["Wo"] = dY.T @ F
    g["bo"] = dY.sum(axis=0)
    dF = dY @ p["Wo"]                                               # (B,d)

    # ---- amechania split: F = (1-alpha) C_crew + alpha ZT ---------------------
    dC_crew = dF * (1.0 - alpha)
    dZT = dF * alpha
    dalpha = np.sum(dF * (ZT - C_crew), axis=1, keepdims=True)      # (B,1)
    du = dalpha * alpha * (1.0 - alpha)                             # sigmoid'
    g["wa"] = (X * du).sum(axis=0)
    g["ba"] = du.sum(axis=0)

    # ---- MEDEA deliberation: unroll backward through T steps ------------------
    dZ = dZT
    for (Zprev, Sd, Pd, Gt) in reversed(delib):
        # Znew = Gt*Pd + (1-Gt)*Sd
        dGt = dZ * (Pd - Sd)
        dPd = dZ * Gt
        dSd = dZ * (1.0 - Gt)
        dAs = dSd * (1.0 - Sd * Sd)          # tanh'
        dAd = dPd * (1.0 - Pd * Pd)          # tanh'
        dAg = dGt * Gt * (1.0 - Gt)          # sigmoid'
        g["Ws"] += dAs.T @ Zprev; g["bs"] += dAs.sum(0)
        g["Wd"] += dAd.T @ Zprev; g["bd"] += dAd.sum(0)
        g["Wg"] += dAg.T @ Zprev; g["bg"] += dAg.sum(0)
        dZ = dAs @ p["Ws"] + dAd @ p["Wd"] + dAg @ p["Wg"]
    dCc = dZ                                                        # grad wrt context

    # ---- context: Cc = C_crew + R --------------------------------------------
    dC_crew = dC_crew + dCc
    dR = dCc

    # ---- aetiology backward: R = Att @ Vmem; Att = softmax(S) -----------------
    g["Vmem"] = Att.T @ dR
    dAtt = dR @ Vmem.T
    dS = Att * (dAtt - np.sum(dAtt * Att, axis=1, keepdims=True))   # softmax jac
    g["Kmem"] = (dS.T @ X) / np.sqrt(d_in)

    # ---- crew backward: C_crew = sum_k Wts[:,k] * Hk[:,k,:] -------------------
    dHk = Wts[:, :, None] * dC_crew[:, None, :]                     # (B,K,d)
    dWts = np.einsum("bd,bkd->bk", dC_crew, Hk)                     # (B,K)
    dA_crew = dHk * (1.0 - Hk * Hk)                                 # tanh'
    g["Wk"] = np.einsum("bkd,bi->kdi", dA_crew, X)
    g["bk"] = dA_crew.sum(axis=0)
    # router: Wts = softmax(Glog)
    dG = Wts * (dWts - np.sum(dWts * Wts, axis=1, keepdims=True))
    g["Wr"] = dG.T @ X
    g["br"] = dG.sum(axis=0)

    return g


# ============================================================================
# FINITE-DIFFERENCE GRADIENT CHECK   (mandatory)
# ============================================================================
def grad_check(p, X, Tgt, eps=1e-5):
    """Compare analytic gradients to central finite differences. Returns max rel err."""
    Y, cache = forward(p, X)
    _, dY = mse_loss(Y, Tgt)
    g = backward(p, cache, dY)

    def loss_only(params):
        Yl, _ = forward(params, X)
        L, _ = mse_loss(Yl, Tgt)
        return L

    worst = 0.0
    worst_name = ""
    for name, val in p.items():
        if name == "_dims":
            continue
        flat = val.reshape(-1)
        gflat = g[name].reshape(-1)
        # probe up to 12 random coordinates per tensor (keeps the check fast)
        idxs = RNG.choice(flat.size, size=min(12, flat.size), replace=False)
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps; Lp = loss_only(p)
            flat[i] = orig - eps; Lm = loss_only(p)
            flat[i] = orig
            num = (Lp - Lm) / (2 * eps)
            ana = gflat[i]
            denom = max(1e-8, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            if rel > worst:
                worst, worst_name = rel, f"{name}[{i}]"
    return worst, worst_name


# ============================================================================
# SYNTHETIC TASK -- "The Crossing"
# ============================================================================
# Every sample is a predicament x in R^d_in. The correct response can ONLY be
# produced by (a) resolving a conflict between a duty-cue and a desire-cue and
# (b) recalling a stored precedent keyed by the situation's "region." This forces
# crew, archive, and deliberation to cooperate -- there is no shortcut.
def make_task(n, d_in=6, n_regions=4, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d_in))
    # Precedent table: each "region" of situation-space stores a fixed lesson.
    precedents = rng.standard_normal((n_regions, 2))
    region = ((X[:, 4] > 0).astype(int) + 2 * (X[:, 5] > 0).astype(int))  # 0..3

    duty = np.tanh(X[:, 2] + X[:, 3])                # what restraint counsels
    desire = np.tanh(X[:, 2] - X[:, 1])              # what longing counsels
    lean = sigmoid(3.0 * (X[:, 0] - X[:, 1]))        # how far the soul tilts to desire
    y0 = lean * desire + (1.0 - lean) * duty         # the deliberated choice
    y1 = precedents[region, 0] * np.tanh(X[:, 3]) + precedents[region, 1]  # recalled lesson
    Tgt = np.stack([y0, 0.3 * y1], axis=1)
    return X, Tgt


def train(p, X, Tgt, epochs=400, lr=0.05, batch=64, verbose=True):
    n = X.shape[0]
    rng = np.random.default_rng(0)
    history = []
    for ep in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            Y, cache = forward(p, X[idx])
            _, dY = mse_loss(Y, Tgt[idx])
            g = backward(p, cache, dY)
            for k in g:
                p[k] -= lr * g[k]               # plain SGD -- the mind learns its trade
        if ep % max(1, epochs // 10) == 0 or ep == epochs - 1:
            Yf, _ = forward(p, X)
            L, _ = mse_loss(Yf, Tgt)
            history.append((ep, L))
            if verbose:
                print(f"  epoch {ep:4d}   loss {L:.5f}")
    return history


# ============================================================================
# SELF-TESTS  +  DEMONSTRATION
# ============================================================================
def main():
    print("=" * 72)
    print("THE AMECHANIA ENGINE  --  Apollonius of Rhodes")
    print("intelligence as the management of helplessness")
    print("=" * 72)

    # ---- (A) gradient check on a tiny random instance ------------------------
    print("\n[1] Finite-difference gradient check (tiny instance)")
    p_small = init_params(d_in=5, d=6, d_out=2, K=3, M=4, T=3)
    Xc = RNG.standard_normal((4, 5))
    Tc = RNG.standard_normal((4, 2))
    worst, where = grad_check(p_small, Xc, Tc)
    print(f"    max relative error = {worst:.3e}  (worst at {where})")
    assert worst < 1e-5, "GRADIENT CHECK FAILED"
    print("    PASS  -- analytic backprop matches numerical gradient.")

    # ---- (B) structural invariants -------------------------------------------
    print("\n[2] Structural invariants")
    Yc, cache = forward(p_small, Xc)
    assert np.allclose(cache["Wts"].sum(1), 1.0), "router weights must sum to 1"
    assert np.allclose(cache["Att"].sum(1), 1.0), "archive attention must sum to 1"
    assert np.all((cache["alpha"] > 0) & (cache["alpha"] < 1)), "amechania in (0,1)"
    print("    crew router sums to 1, archive attention sums to 1, alpha in (0,1)  -- PASS")

    # ---- (C) train on 'The Crossing' -----------------------------------------
    print("\n[3] Training on 'The Crossing' (conflict + recall task)")
    Xtr, Ttr = make_task(1024, d_in=6, seed=7)
    Xte, Tte = make_task(256, d_in=6, seed=99)
    p = init_params(d_in=6, d=24, d_out=2, K=4, M=8, T=3)
    Y0, _ = forward(p, Xtr); L0, _ = mse_loss(Y0, Ttr)
    print(f"    initial train loss {L0:.5f}")
    train(p, Xtr, Ttr, epochs=400, lr=0.05, batch=64)
    Yte, cte = forward(p, Xte); Lte, _ = mse_loss(Yte, Tte)
    print(f"    final   test  loss {Lte:.5f}")
    assert Lte < 0.5 * L0, "training did not reduce loss enough"
    print(f"    loss fell by {100*(1-Lte/L0):.1f}%  -- the engine learned the crossing.")

    # ---- (D) read the mind: when does it lean on deliberation? ---------------
    print("\n[4] Interpreting the amechania gate")
    a = cte["alpha"].ravel()
    print(f"    mean helplessness alpha = {a.mean():.3f}   (0=acts directly, 1=defers fully)")
    print(f"    alpha range [{a.min():.3f}, {a.max():.3f}]  -- the gate genuinely varies by situation.")

    print("\n" + "=" * 72)
    print("ALL CHECKS PASSED.")
    print("=" * 72)


if __name__ == "__main__":
    main()

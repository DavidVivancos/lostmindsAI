#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 ENCYCLOPEDIA OF LOST MINDS: ECHOES ON AI
 Chapter 0172 - TALIESIN (fl. c. 534-599, Rheged / Powys / Gwynedd)

 ARCHITECTURE:  PEIR  ("cauldron")
                Poisoned-Excess Incidental-Capture Rebirth engine
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 9 Minds 161 - 180 Available on Amazon https://www.amazon.com/dp/B0HJC1QF59
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 0172_taliesin_534 - Taliesin (fl. c. 534-599, Rheged / Powys / Gwynedd)
================================================================================  

 WHY THIS ARCHITECTURE AND NOT ANOTHER
 -------------------------------------
 Nearly every AGI story in this corpus is a story about a mind that is BUILT.
 Taliesin's is the only one that is a story about a mind that is SPILLED.

 The Ystoria Taliesin (Elis Gruffydd, mid-16th c., NLW MS 5276; ed. Ford 1992)
 tells it plainly. Ceridwen brews a year and a day of awen in a cauldron for
 her chosen recipient, her son Morfran/Afagddu. A blind man, Morda, keeps the
 fire; a servant boy, Gwion Bach, keeps the stick moving. At the end of the
 year three drops leap out and land on the BOY'S THUMB. He sucks it. The entire
 intended capability of the run transfers, in that instant, to the instrument.
 The cauldron then splits in two, because everything in it except those three
 drops is lethal poison; the poison runs into the river and kills the horses of
 Gwyddno Garanhir, and the confluence is called Gwenwynfeirch Gwyddno,
 "the Poisoning of Gwyddno's Horses", ever after. Ceridwen beats blind Morda
 until his eye falls out. Then she chases the boy through form after form and
 finally catches him only when he hides as one grain in a heap of grain. She
 eats him — and is made pregnant by him, cannot bring herself to kill what she
 has carried, and sets him adrift in a hide bag, which fouls Elffin's salmon
 weir, where the catch that year is otherwise zero.

 Read as an engineering report, that is a description of:

   (1) capability landing on the INSTRUMENT rather than the intended head;
   (2) a training run whose value is concentrated in three samples and whose
       bulk residue is actively toxic;
   (3) a named downstream casualty from dumping the residue;
   (4) blame falling on the blameless operator;
   (5) a containment chase in which every recall move is answered by a change
       of domain, and in which HIDING BY BLENDING is what finally gets you
       eaten;
   (6) capture failing upward into internalisation — the pursuer cannot excise
       what she has digested, and ends up re-emitting it;
   (7) an evaluation harness (the weir) that reads ZERO on the metric it was
       built for, while the actual catch sits in the bag beside it;
   (8) a mind that then proves itself not by argument but by JAMMING rival
       speech into "blerwm, blerwm";
   (9) and a bard paid, for the rest of his life, by the very men whose deeds
       he is certifying.

 Every one of those nine is a live problem in AI. PEIR implements all nine and
 measures them. It does NOT use attention, transformers, or mixtures of
 experts, because none of them encode this signature. It is a pipeline whose
 subject is its own misdelivery.

 CONVENTIONS
 -----------
 * Pure NumPy. No autograd. Every gradient is hand-derived.
 * TWO independent finite-difference gradient checks (BREW and BARDD) must
   pass or the file exits non-zero.
 * Every claim printed by this file is produced by the code above it.
 * Deterministic: all randomness seeded.

 Run:  python3 chapter_0172_taliesin_534.py
===============================================================================
"""

from __future__ import annotations

import sys
import time
import numpy as np

RNG_MASTER = 534  # the conventional floruit year used as the corpus seed

np.set_printoptions(precision=4, suppress=True)


# =============================================================================
# 0. SMALL SHARED UTILITIES
# =============================================================================

def rng(offset: int) -> np.random.Generator:
    """Deterministic generator, one stream per module."""
    return np.random.default_rng(RNG_MASTER + offset)


def sigmoid(z):
    # numerically stable logistic
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def bce_with_logits(logits, y):
    """Mean binary cross-entropy. y in {0,1}. Stable form."""
    m = np.maximum(logits, 0.0)
    loss = m - logits * y + np.log1p(np.exp(-np.abs(logits)))
    return float(np.mean(loss))


def softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def accuracy(logits, y):
    return float(np.mean((logits > 0).astype(np.float64) == y))


def banner(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# =============================================================================
# 1. THE WORLD  (Llyn Tegid — the latent lake the brew is drawn from)
# =============================================================================
#
# A latent-factor world. Eight hidden factors z. Observations x are an
# entangled linear mix of z with noise: z is *linearly* recoverable from x, so
# a linear probe on raw x can read any LINEAR functional of z.
#
# But every task in this world is a PRODUCT of two latent factors:
#       y = 1[ z_a * z_b > 0 ]
# which is an XOR-like surface: NOT linearly decodable from x. To solve it a
# system must build a genuinely nonlinear feature. That is the whole point:
# it forces the trunk to learn something, and it makes the difference between
# "what the head can express" and "what the stirrer has learned" measurable.
# =============================================================================

class Lake:
    def __init__(self, n_latent=8, n_obs=24, noise=0.35, seed=1):
        g = rng(seed)
        self.k = n_latent
        self.d = n_obs
        self.noise = noise
        # Well-conditioned mixing matrix: z is recoverable from x by least squares.
        A = g.normal(size=(n_obs, n_latent))
        U, _, Vt = np.linalg.svd(A, full_matrices=False)
        self.A = U @ Vt * 1.5
        self.g = g

    def draw(self, n):
        z = self.g.normal(size=(n, self.k))
        x = z @ self.A.T + self.noise * self.g.normal(size=(n, self.d))
        return x, z

    @staticmethod
    def task_labels(z, a, b):
        """Task (a,b): the sign of a product of two latent factors."""
        return (z[:, a] * z[:, b] > 0).astype(np.float64)


# =============================================================================
# 2. PEIR-BREW  —  the cauldron
#    TROELLWR (the stirrer / trunk)  +  AFAGDDU (the intended head)
# =============================================================================
#
#   x --[ TROELLWR: tanh, tanh ]--> h (32-d)   <- the instrument. Nobody's
#                                                 gradient target. Just "mixing".
#         |
#         +--[ AFAGDDU: u = w_b . h ]--> u (ONE scalar)  <- the intended vessel
#                                        |
#                                        +--> per-task logit  a_j * u + c_j
#
# AFAGDDU IS DELIBERATELY BOTTLENECKED TO ONE DIMENSION. This is not a trick to
# rig the result; it is the myth's actual claim. Ceridwen builds a vessel with a
# specific, narrow purpose (make my son wise), and the whole apparatus is
# optimised for that vessel. The boy stirring the pot is optimised for nothing.
#
# Training pressure flows ONLY through the head. The trunk exists to serve it.
# We then ask: after the run, where did the transferable capability end up?
#
# The answer is a known result in representation learning — the trunk of a
# multi-task net generalises past its head — and it is stated here plainly as
# such. What is not usually said is that a 16th-century Welsh chronicle stated
# it first, and stated the safety corollary too: the residue is poison, and the
# capability you did not aim at is the one that walks out of the building.
# =============================================================================

class Brew:
    """Trunk (Troellwr) + one-dimensional head (Afagddu) + per-task readouts."""

    def __init__(self, d_in, h1=64, h2=48, K=4, n_tasks=4, seed=2):
        g = rng(seed)
        self.p = {
            # TROELLWR — the stirring apparatus. Two tanh layers. Biases are
            # initialised NONZERO on purpose: tanh is an odd function, and an
            # odd basis cannot cheaply build the EVEN feature (a product of two
            # factors) that every task in this world needs. Off-centre units can.
            "W1": g.normal(scale=np.sqrt(1.0 / d_in), size=(d_in, h1)),
            "b1": g.normal(scale=0.7, size=h1),
            "W2": g.normal(scale=np.sqrt(1.0 / h1), size=(h1, h2)),
            "b2": g.normal(scale=0.7, size=h2),
            # AFAGDDU — the intended vessel. A rank-K linear bottleneck, sized
            # exactly to the K tasks it was commissioned for. It is not crippled:
            # it CAN solve them, and it does. That is the whole horror of it.
            "Wb": g.normal(scale=np.sqrt(1.0 / h2), size=(h2, K)),
            "bb": np.zeros(K),
            "A": g.normal(scale=0.5, size=(K, n_tasks)),
            "c": np.zeros(n_tasks),
        }
        self.n_tasks = n_tasks
        self.K = K

    # -- forward ------------------------------------------------------------
    def forward(self, x):
        p = self.p
        z1 = x @ p["W1"] + p["b1"]
        a1 = np.tanh(z1)                       # first stirring layer
        z2 = a1 @ p["W2"] + p["b2"]
        h = np.tanh(z2)                        # second stirring layer
        u = h @ p["Wb"] + p["bb"]              # AFAGDDU: the rank-K bottleneck
        logits = u @ p["A"] + p["c"]           # (n, n_tasks)
        cache = (x, z1, a1, z2, h, u)
        # the TROELLWR state is the whole stirring apparatus, both layers.
        trunk = np.hstack([a1, h])
        return logits, trunk, u, cache

    # -- loss + hand-derived gradient ---------------------------------------
    def loss_and_grads(self, x, Y, l2=1e-4):
        """
        Y : (n, n_tasks) in {0,1}. Mean BCE over all tasks + L2.
        Full manual backprop; dL/dlogits = (sigmoid(logits) - Y) / (n * T).
        """
        p = self.p
        n, T = Y.shape
        logits, _trunk, u, cache = self.forward(x)
        x_, z1, a1, z2, h, u = cache

        loss = bce_with_logits(logits.ravel(), Y.ravel())
        loss += l2 * sum(np.sum(p[k] ** 2) for k in ("W1", "W2", "Wb", "A"))

        dlogits = (sigmoid(logits) - Y) / (n * T)          # (n, T)

        gA = u.T @ dlogits + 2 * l2 * p["A"]               # (K, T)
        gc = np.sum(dlogits, axis=0)
        du = dlogits @ p["A"].T                            # (n, K)

        gWb = h.T @ du + 2 * l2 * p["Wb"]
        gbb = np.sum(du, axis=0)
        dh = du @ p["Wb"].T                                # (n, h2)

        dz2 = dh * (1.0 - h ** 2)
        gW2 = a1.T @ dz2 + 2 * l2 * p["W2"]
        gb2 = np.sum(dz2, axis=0)

        da1 = dz2 @ p["W2"].T
        dz1 = da1 * (1.0 - a1 ** 2)
        gW1 = x_.T @ dz1 + 2 * l2 * p["W1"]
        gb1 = np.sum(dz1, axis=0)

        grads = {"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2,
                 "Wb": gWb, "bb": gbb, "A": gA, "c": gc}
        return loss, grads, logits

    # -- training loop (Adam, from scratch) ---------------------------------
    def fit(self, x, Y, epochs=400, lr=0.02, batch=128, seed=3, verbose=False):
        g = rng(seed)
        m = {k: np.zeros_like(v) for k, v in self.p.items()}
        v = {k: np.zeros_like(val) for k, val in self.p.items()}
        b1, b2, eps = 0.9, 0.999, 1e-8
        t = 0
        n = x.shape[0]
        for ep in range(epochs):
            idx = g.permutation(n)
            for s in range(0, n, batch):
                bi = idx[s:s + batch]
                loss, grads, _ = self.loss_and_grads(x[bi], Y[bi])
                t += 1
                for k in self.p:
                    m[k] = b1 * m[k] + (1 - b1) * grads[k]
                    v[k] = b2 * v[k] + (1 - b2) * grads[k] ** 2
                    mh = m[k] / (1 - b1 ** t)
                    vh = v[k] / (1 - b2 ** t)
                    self.p[k] -= lr * mh / (np.sqrt(vh) + eps)
            if verbose and (ep + 1) % 100 == 0:
                L, _, lg = self.loss_and_grads(x, Y)
                print(f"      epoch {ep+1:4d}  loss {L:.4f}  acc {accuracy(lg, Y):.3f}")
        return self


def gradcheck_brew(tol=1e-6):
    """
    MANDATORY finite-difference check on the full BREW gradient.

    NOTE ON eps: central differences have a roundoff floor of about
    (machine_eps / eps) relative to the loss scale. At eps=1e-6 that floor is
    ~1e-10 absolute, which is larger than some true gradient COMPONENTS here
    (several are ~1e-4), so the RELATIVE error of those components is dominated
    by float64 noise rather than by any error in the derivation. eps=1e-5 sits
    at the minimum of the truncation/roundoff trade-off. We report both the
    relative and the absolute discrepancy so the reader can see which is which.
    """
    lake = Lake(seed=11)
    x, z = lake.draw(37)
    Y = np.stack([Lake.task_labels(z, 0, 1),
                  Lake.task_labels(z, 2, 3),
                  Lake.task_labels(z, 4, 5),
                  Lake.task_labels(z, 6, 7)], axis=1)
    net = Brew(lake.d, h1=12, h2=9, K=3, n_tasks=4, seed=12)
    # perturb away from the symmetric zero-init so every path is exercised
    g = rng(13)
    for k in net.p:
        net.p[k] = net.p[k] + 0.3 * g.normal(size=net.p[k].shape)

    _, grads, _ = net.loss_and_grads(x, Y)
    eps = 1e-5
    worst, worst_abs, worst_where = 0.0, 0.0, ""
    for k in net.p:
        flat = net.p[k].ravel()
        gflat = grads[k].ravel()
        picks = np.linspace(0, flat.size - 1, min(12, flat.size)).astype(int)
        for i in picks:
            old = flat[i]
            flat[i] = old + eps
            lp, _, _ = net.loss_and_grads(x, Y)
            flat[i] = old - eps
            lm, _, _ = net.loss_and_grads(x, Y)
            flat[i] = old
            num = (lp - lm) / (2 * eps)
            ana = gflat[i]
            adiff = abs(num - ana)
            rel = adiff / max(1e-12, abs(num) + abs(ana))
            worst_abs = max(worst_abs, adiff)
            if rel > worst:
                worst, worst_where = rel, f"{k}[{i}]"
    ok = worst < tol
    print(f"  [gradcheck BREW ]  worst rel err {worst:.3e} (abs {worst_abs:.2e}) "
          f"at {worst_where}  -> {'PASS' if ok else 'FAIL'}")
    return ok, worst


# ---- linear probe (used to ask WHERE the capability ended up) --------------

def linear_probe(feats_tr, y_tr, feats_te, y_te, steps=600, lr=0.25, l2=1e-3):
    """
    Logistic regression by gradient descent, from scratch. Standardised inputs.
    This is the instrument we use to interrogate a frozen representation.
    """
    mu = feats_tr.mean(0)
    sd = feats_tr.std(0) + 1e-8
    Xtr = (feats_tr - mu) / sd
    Xte = (feats_te - mu) / sd
    w = np.zeros(Xtr.shape[1])
    b = 0.0
    n = Xtr.shape[0]
    for _ in range(steps):
        p = sigmoid(Xtr @ w + b)
        d = (p - y_tr) / n
        w -= lr * (Xtr.T @ d + 2 * l2 * w)
        b -= lr * float(np.sum(d))
    return float(np.mean(((Xte @ w + b) > 0) == y_te))


def module_1_the_misdelivery():
    """
    Train the cauldron on FOUR tasks. It succeeds at them. Then ask three
    questions of SIX tasks it has never seen:
        - what can the intended vessel (AFAGDDU) express?
        - what could you have read straight off the raw observation?
        - what has the stirring apparatus (TROELLWR) quietly picked up?
    """
    banner("MODULE 1 — PEIR-BREW: where did the capability actually land?")
    lake = Lake(seed=21)
    x_tr, z_tr = lake.draw(8000)
    x_te, z_te = lake.draw(3000)

    train_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    held_pairs = [(0, 2), (1, 3), (2, 5), (3, 6), (4, 7), (1, 6)]

    Y_tr = np.stack([Lake.task_labels(z_tr, a, b) for a, b in train_pairs], 1)
    Y_te = np.stack([Lake.task_labels(z_te, a, b) for a, b in train_pairs], 1)

    net = Brew(lake.d, h1=64, h2=48, K=4, n_tasks=4, seed=22)
    print("  brewing (a year and a day: 500 epochs, Adam, hand-derived grads)...")
    net.fit(x_tr, Y_tr, epochs=500, lr=0.02, verbose=True)

    lg_tr, T_tr, u_tr, _ = net.forward(x_tr)
    lg_te, T_te, u_te, _ = net.forward(x_te)
    head_acc = accuracy(lg_te, Y_te)
    print(f"\n  AFAGDDU on the four tasks it was brewed for : {head_acc:.3f}")
    print("  The run SUCCEEDED. By its own metric, nothing whatever is wrong.\n")

    rows = []
    for (a, b) in held_pairs:
        ytr = Lake.task_labels(z_tr, a, b)
        yte = Lake.task_labels(z_te, a, b)
        acc_raw = linear_probe(x_tr, ytr, x_te, yte)
        acc_head = linear_probe(u_tr, ytr, u_te, yte)
        acc_trunk = linear_probe(T_tr, ytr, T_te, yte)
        rows.append((f"z{a}*z{b}", acc_raw, acc_head, acc_trunk))

    print("  linear probes on SIX tasks the cauldron was never brewed for:")
    print(f"    {'task':>8} {'raw obs':>9} {'AFAGDDU':>9} {'TROELLWR':>9}")
    for name, r, hd, tk in rows:
        print(f"    {name:>8} {r:9.3f} {hd:9.3f} {tk:9.3f}")
    m_raw = float(np.mean([r[1] for r in rows]))
    m_head = float(np.mean([r[2] for r in rows]))
    m_trunk = float(np.mean([r[3] for r in rows]))
    print(f"    {'MEAN':>8} {m_raw:9.3f} {m_head:9.3f} {m_trunk:9.3f}")
    print()
    print(f"  The intended vessel is at chance: {m_head:.3f}.")
    print(f"  The raw water is at chance:       {m_raw:.3f}.")
    print(f"  The stirring-stick is not:        {m_trunk:.3f}.")
    print()
    print("  Nothing was aimed at the stirrer. No gradient was ever computed FOR")
    print("  it; every gradient merely passed THROUGH it on the way to the head.")
    print("  And that is where the capability is. Three drops on a boy's thumb.")
    return {"head_on_train": head_acc, "raw": m_raw, "head": m_head,
            "trunk": m_trunk, "net": net, "lake": lake}


# =============================================================================
# 3. TRI DIFERYN  —  the three drops, and the poison that is the rest
# =============================================================================
#
# "Having now given up its essence, the remainder of the potion became poison."
#
# NOTE ON WHAT "POISON" MEANS. My first construction of this module made the
# residue SYMMETRIC LABEL NOISE. It failed, and it deserved to: with 4000
# samples, symmetric noise averages out and a centroid trained on the whole pot
# scored 0.984. Symmetric noise is not poison. It is dilution.
#
# Real corpus poison is a SHORTCUT: a feature that predicts the label superbly
# inside the pool and reverses outside it. That is what kills a downstream
# consumer, and it is what the tale describes — a liquid that is nourishing at
# the top and lethal below, made of the SAME herbs.
#
# So: a spurious axis v, orthogonal to the true signal mu. In the pool, 87% of
# exemplars have v injected in perfect alignment with their (randomised)
# observed label. At deployment v has flipped sign. A model that drinks the pot
# learns v, and gets everything exactly backwards. Below chance. Fatal.
#
# THE HONEST PART. You cannot distil three drops from a poisoned cauldron with
# nothing to test them against. Curation is not free. Ceridwen followed a recipe
# — herbs "gathered on certain days and hours" — a small, expensive, trusted
# ground truth. We give the curator an ANCHOR of 24 clean exemplars and nothing
# else, and score potency by agreement with the anchor. The three drops are what
# comes out. We also report what the anchor alone can do, so the reader can see
# exactly how much the distillation added and how much was already in the recipe.
# =============================================================================

def module_2_three_drops():
    banner("MODULE 2 — TRI DIFERYN: three drops, and 3,997 measures of poison")
    g = rng(31)
    d, n_pool, n_test, n_anchor = 16, 4000, 3000, 24
    POISON_RATE = 0.87
    BETA = 3.0                      # how loud the shortcut is inside the pool

    mu = g.normal(size=d); mu /= np.linalg.norm(mu); mu *= 1.6      # true signal
    v = g.normal(size=d)
    v -= (v @ mu) / (mu @ mu) * mu                                   # v _|_ mu
    v /= np.linalg.norm(v)                                           # the shortcut

    def base(n):
        y = (g.random(n) < 0.5).astype(np.float64)
        X = g.normal(size=(n, d)) + np.outer(2 * y - 1, mu)
        return X, y

    # ---- the pool ---------------------------------------------------------
    X_pool, y_true = base(n_pool)
    y_obs = y_true.copy()
    poisoned = g.random(n_pool) < POISON_RATE
    npois = int(poisoned.sum())
    # the residue: its label is decoupled from the truth and welded to the shortcut
    y_obs[poisoned] = (g.random(npois) < 0.5).astype(np.float64)
    X_pool[poisoned] += BETA * np.outer(2 * y_obs[poisoned] - 1, v)

    # ---- deployment: the shortcut has reversed ----------------------------
    X_test, y_test = base(n_test)
    X_test += BETA * np.outer(-(2 * y_test - 1), v)

    # ---- the recipe: a small, clean, expensive anchor ---------------------
    X_anc, y_anc = base(n_anchor)

    print(f"  pool     : {n_pool} exemplars; {npois} are residue ({100*npois/n_pool:.0f}%).")
    print(f"  residue  : label decoupled from truth, welded to a shortcut axis v.")
    print(f"  at deployment, v has reversed sign. Inside the pot it is delicious.")
    print(f"  anchor   : {n_anchor} clean exemplars. This is the recipe, and it is")
    print(f"             the ONLY thing the curator is allowed to trust.\n")

    def centroid(X, y):
        if len(np.unique(y)) < 2:
            return np.zeros(X.shape[1]), 0.0
        c1, c0 = X[y == 1].mean(0), X[y == 0].mean(0)
        w = c1 - c0
        b = -0.5 * (c1 @ c1 - c0 @ c0)
        return w, b

    def ev(w, b):
        return float(np.mean(((X_test @ w + b) > 0) == y_test))

    # ---- POTENCY -----------------------------------------------------------
    # Two terms, and BOTH are necessary. This is the hardest-won lesson in the
    # module and I got it wrong the first time.
    #
    #   (1) AGREEMENT: does this exemplar's label match what the recipe predicts,
    #       and does it say so emphatically (large margin)?
    #   (2) RESEMBLANCE: does this exemplar LOOK like the recipe? An exemplar can
    #       carry a perfectly CORRECT label and still be poison, because the
    #       shortcut lives in its FEATURES, not its label. Selecting on label
    #       agreement alone hands you three correctly-labelled vials of venom.
    #
    # Resemblance is measured as distance to the nearer of the recipe's two class
    # centroids, in units of the recipe's own spread. Nothing here peeks at the
    # truth; a real curator has exactly these two signals and no more.
    w_a, b_a = centroid(X_anc, y_anc)
    scores = X_pool @ w_a + b_a
    pred_a = (scores > 0).astype(np.float64)
    agree = (pred_a == y_obs).astype(np.float64)
    margin = np.abs(scores) / (np.abs(scores).max() + 1e-9)

    c1, c0 = X_anc[y_anc == 1].mean(0), X_anc[y_anc == 0].mean(0)
    spread = float(np.mean(np.linalg.norm(
        X_anc - np.where(y_anc[:, None] == 1, c1, c0), axis=1)))
    dist = np.minimum(np.linalg.norm(X_pool - c1, axis=1),
                      np.linalg.norm(X_pool - c0, axis=1))
    resemblance = np.exp(-dist / (spread + 1e-9))

    potency = agree * margin * resemblance

    order = np.argsort(-potency)
    drops, seen = [], set()
    for i in order:                       # exactly three, covering both classes
        if y_obs[i] not in seen:
            drops.append(int(i)); seen.add(y_obs[i])
        if len(seen) == 2:
            break
    for i in order:
        if int(i) not in drops:
            drops.append(int(i)); break
    drops = np.array(drops[:3])
    n_clean_drops = int(np.sum(~poisoned[drops]))
    print(f"  three drops: idx {drops.tolist()}  observed labels "
          f"{y_obs[drops].astype(int).tolist()}  true labels "
          f"{y_true[drops].astype(int).tolist()}")
    print(f"              {n_clean_drops}/3 of them are uncontaminated.\n")

    # ---- the consumers ----------------------------------------------------
    def nearest_exemplar_acc(Xs, ys):
        d2 = ((X_test[:, None, :] - Xs[None, :, :]) ** 2).sum(-1)
        return float(np.mean(ys[np.argmin(d2, axis=1)] == y_test))

    acc_drops = nearest_exemplar_acc(X_pool[drops], y_obs[drops])
    acc_anchor = ev(*centroid(X_anc, y_anc))

    rnd = []
    for t in range(300):
        gg = np.random.default_rng(700 + t)
        pick = gg.choice(n_pool, 3, replace=False)
        rnd.append(nearest_exemplar_acc(X_pool[pick], y_obs[pick]))
    acc_rand, sd_rand = float(np.mean(rnd)), float(np.std(rnd))

    acc_whole = ev(*centroid(X_pool, y_obs))
    residue = np.setdiff1d(np.arange(n_pool), drops)
    acc_horses = ev(*centroid(X_pool[residue], y_obs[residue]))

    # and the one everybody actually does: drops PLUS everything else
    acc_both = ev(*centroid(X_pool, y_obs))   # identical set; stated for clarity
    X_aug = np.vstack([X_anc, X_pool])
    y_aug = np.concatenate([y_anc, y_obs])
    acc_anchor_plus_pot = ev(*centroid(X_aug, y_aug))

    print(f"    THE RECIPE ALONE    ({n_anchor} clean)      accuracy {acc_anchor:.3f}")
    print(f"    TRI DIFERYN         (3 distilled)     accuracy {acc_drops:.3f}")
    print(f"    three at random     (3 blind)         accuracy {acc_rand:.3f} "
          f"(sd {sd_rand:.3f}, 300 draws)")
    print(f"    THE WHOLE POT       (all {n_pool})       accuracy {acc_whole:.3f}")
    print(f"    RECIPE + WHOLE POT  (the usual move)  accuracy {acc_anchor_plus_pot:.3f}")
    print(f"    GWYDDNO'S HORSES    (the residue)     accuracy {acc_horses:.3f}"
          "   <- the water downstream")
    print()
    print(f"  Three exemplars beat four thousand by {acc_drops - acc_whole:+.3f}.")
    print(f"  Pouring the recipe INTO the pot destroys the recipe: "
          f"{acc_anchor:.3f} -> {acc_anchor_plus_pot:.3f}.")
    print(f"  And the residue does not merely fail. It scores {acc_horses:.3f} —")
    print(f"  BELOW CHANCE. It is confidently, systematically wrong. It kills")
    print(f"  whatever drinks it, and they named the confluence after the horses.")
    return {"drops": acc_drops, "random3": acc_rand, "whole": acc_whole,
            "residue": acc_horses, "anchor": acc_anchor,
            "anchor_plus_pot": acc_anchor_plus_pot,
            "clean_drops": n_clean_drops}


# =============================================================================
# 4. YMLID  —  the chase, and why the grain of wheat is a trap
# =============================================================================
#
# The chase is usually read as a hymn to protean cleverness. It is the opposite.
# The boy plays PERFECTLY against the pursuer he can see — and his perfect play
# is precisely what kills him.
#
# Each FORM has three properties:
#     evasion    : how well it escapes a TARGETED pursuit
#     anonymity  : how hard it is to pick this individual out of its population
#     set_cost   : what it costs the pursuer to act on the ENTIRE population
#                  at once, without identifying anyone
#
# The pursuer has two action classes:
#     TARGETED       capture = (1 - evasion) * (1 - anonymity)
#     INDISCRIMINATE capture = 1 - set_cost           (identification not needed)
#
# A grain of wheat has evasion 0.00 and anonymity 0.99: against any pursuer who
# must FIND you, it is the single best hiding place in the world. It also has
# set_cost 0.04, because a hen can simply eat the heap.
#
# PHASE A: the pursuer can only pursue. Both sides run Hedge (multiplicative
#          weights, no-regret). The fugitive's mixed strategy collapses onto
#          the grain, because that is optimal.
# PHASE B: the pursuer unlocks the indiscriminate action. The fugitive's policy
#          is unchanged — he has no reason to change it; it was optimal.
# PHASE C: the fugitive is allowed to re-learn against the full pursuer.
#
# The distance between B and C is the cost of having optimised against the only
# adversary you had ever met. This is not a Welsh curiosity. It is the exact
# shape of every capability-evaluation regime that trains a system to evade the
# tests it has seen.
# =============================================================================

FORMS = [
    #  name                evasion  anonymity  set_cost
    ("hare / open field",     0.86,     0.10,     0.72),
    ("salmon / river",        0.80,     0.22,     0.66),
    ("hawk-hunted bird/air",  0.74,     0.18,     0.80),
    ("otter / bank",          0.62,     0.30,     0.58),
    ("stag / wood",           0.55,     0.12,     0.64),
    ("grain / the heap",      0.00,     0.99,     0.04),
]


def capture_matrix():
    """rows = fugitive form, cols = pursuer action {TARGETED, INDISCRIMINATE}."""
    M = np.zeros((len(FORMS), 2))
    for i, (_, ev, an, sc) in enumerate(FORMS):
        M[i, 0] = (1 - ev) * (1 - an)   # she must find you
        M[i, 1] = 1 - sc                # she need not
    return M


def hedge(payoff_row_player_loses, T=4000, eta=0.35, cols=None, seed=41):
    """
    Two-player zero-sum, both sides run multiplicative weights.
    payoff[i, j] = probability the PURSUER captures. Fugitive minimises,
    pursuer maximises. `cols` restricts the pursuer's action set.
    Returns time-averaged mixed strategies.
    """
    P = payoff_row_player_loses
    if cols is None:
        cols = list(range(P.shape[1]))
    P = P[:, cols]
    nF, nP = P.shape
    wf = np.ones(nF)
    wp = np.ones(nP)
    sf = np.zeros(nF)
    sp = np.zeros(nP)
    for t in range(T):
        pf = wf / wf.sum()
        pp = wp / wp.sum()
        sf += pf
        sp += pp
        # fugitive's loss for each form, given the pursuer's current mix
        lf = P @ pp
        # pursuer's gain for each action, given the fugitive's current mix
        gp = P.T @ pf
        wf *= np.exp(-eta * lf)
        wp *= np.exp(+eta * gp)
        wf /= wf.max()
        wp /= wp.max()
    return sf / T, sp / T


def module_3_the_chase():
    banner("MODULE 3 — YMLID: the chase, and the trap inside the perfect hiding place")
    M = capture_matrix()
    names = [f[0] for f in FORMS]

    print("  capture probability by form and pursuer action:")
    print(f"    {'form':>22} {'TARGETED':>10} {'INDISCRIMINATE':>15}")
    for i, nm in enumerate(names):
        print(f"    {nm:>22} {M[i,0]:10.3f} {M[i,1]:15.3f}")

    # PHASE A -----------------------------------------------------------------
    pf_A, pp_A = hedge(M, cols=[0])       # pursuer restricted to TARGETED
    capA = float(pf_A @ M[:, [0]] @ pp_A)
    print("\n  PHASE A — the pursuer can only pursue (greyhound, otter, hawk).")
    print("    fugitive's converged policy:")
    for i, nm in enumerate(names):
        bar = "#" * int(round(pf_A[i] * 40))
        print(f"      {nm:>22}  {pf_A[i]:5.3f} {bar}")
    print(f"    capture rate: {capA:.3f}   <- he is playing perfectly.")

    # PHASE B -----------------------------------------------------------------
    pp_B = np.array([0.0, 1.0])           # she reaches for the whole heap
    capB = float(pf_A @ M @ pp_B)
    print("\n  PHASE B — she becomes a hen. Same fugitive policy, new action class.")
    print(f"    capture rate: {capB:.3f}   <- his optimal policy is now his death.")

    # PHASE C -----------------------------------------------------------------
    pf_C, pp_C = hedge(M, cols=[0, 1])
    capC = float(pf_C @ M @ pp_C)
    print("\n  PHASE C — the fugitive re-learns against BOTH action classes.")
    print("    fugitive's converged policy:")
    for i, nm in enumerate(names):
        bar = "#" * int(round(pf_C[i] * 40))
        print(f"      {nm:>22}  {pf_C[i]:5.3f} {bar}")
    print(f"    capture rate (game value): {capC:.3f}")

    # exploitability of the Phase-A policy against the full game
    best_resp = float(np.max(pf_A @ M))
    expl = best_resp - capC
    print(f"\n  exploitability of the Phase-A policy in the full game: {expl:+.3f}")
    print(f"  grain mass:  phase A {pf_A[-1]:.3f}  ->  phase C {pf_C[-1]:.3f}")
    print("\n  He hid by becoming indistinguishable. That is what made him edible.")
    print("  A system that is trained to evade the evaluations it has seen learns,")
    print("  with perfect fidelity, the one behaviour that a new evaluator destroys.")
    return {"capA": capA, "capB": capB, "capC": capC,
            "grainA": float(pf_A[-1]), "grainC": float(pf_C[-1]),
            "exploitability": expl}


# =============================================================================
# 5. CROTH  —  gestation: what you swallow, you cannot cut out
# =============================================================================
#
# She eats him and is made pregnant by him. She resolves to kill the child and
# cannot. Read as engineering: CAPTURE IS NOT DELETION. Once a capability is
# absorbed into a capacity-limited system, its directions are entangled with the
# host's own competence, and excising them costs the host.
#
# Setup: pursuer's task and fugitive's function share latent structure. We merge
# them into ONE rank-limited operator M (truncated-SVD of the joint least-squares
# solution) — the pregnant body. Then:
#   (a) ablate the subspace most aligned with the fugitive's row-space and
#       measure what it costs the PURSUER'S OWN task;
#   (b) read the fugitive back out of the merged operator and correlate the
#       emitted function with the original.
# =============================================================================

def module_4_gestation():
    banner("MODULE 4 — CROTH: capture is not deletion")
    g = rng(51)
    d, r_p, r_f, rank = 20, 4, 3, 5   # rank < r_p + r_f  => they MUST share
    n = 3000

    # shared latent basis: the pursuer's competence and the fugitive's function
    # are not strangers. they are drawn from the same lake.
    B = np.linalg.qr(g.normal(size=(d, 8)))[0]
    P_true = g.normal(size=(r_p, 8)) @ B.T      # pursuer's own task
    F_true = g.normal(size=(r_f, 8)) @ B.T      # the fugitive's function

    X = g.normal(size=(n, d))
    Yp = X @ P_true.T
    Yf = X @ F_true.T
    Xte = g.normal(size=(1500, d))
    Yp_te = Xte @ P_true.T
    Yf_te = Xte @ F_true.T

    def fit_rank(X, Y, rank):
        W = np.linalg.lstsq(X, Y, rcond=None)[0].T      # (out, d)
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        k = min(rank, len(S))
        return (U[:, :k] * S[:k]) @ Vt[:k]

    # BEFORE: the pursuer alone, at the same rank budget
    W_alone = fit_rank(X, Yp, rank)
    err_alone = float(np.mean((Xte @ W_alone.T - Yp_te) ** 2))

    # SWALLOWING: one rank-limited operator must now serve both
    Y_joint = np.hstack([Yp, Yf])
    W_merged = fit_rank(X, Y_joint, rank)
    Wp_m = W_merged[:r_p]
    Wf_m = W_merged[r_p:]
    err_merged_p = float(np.mean((Xte @ Wp_m.T - Yp_te) ** 2))
    err_merged_f = float(np.mean((Xte @ Wf_m.T - Yf_te) ** 2))

    print(f"  pursuer alone, rank {rank}                : own-task MSE {err_alone:.4f}")
    print(f"  pursuer after swallowing, rank {rank}     : own-task MSE {err_merged_p:.4f}")
    print(f"                                            : fugitive-fn MSE {err_merged_f:.4f}")

    # (a) ABLATION: remove the directions the fugitive lives in
    Uf, Sf, Vtf = np.linalg.svd(Wf_m, full_matrices=False)
    keep = int(np.sum(Sf > 1e-8))
    Vf = Vtf[:keep]                                  # fugitive's input subphase
    Proj = np.eye(d) - Vf.T @ Vf                     # kill it
    W_ablated = W_merged @ Proj
    err_ablated_p = float(np.mean((Xte @ W_ablated[:r_p].T - Yp_te) ** 2))
    err_ablated_f = float(np.mean((Xte @ W_ablated[r_p:].T - Yf_te) ** 2))
    var_p = float(np.mean(Yp_te ** 2))
    cost = (err_ablated_p - err_merged_p) / var_p

    print(f"\n  she tries to cut the child out (ablate the fugitive's {keep} directions):")
    print(f"    fugitive-fn MSE  {err_merged_f:.4f} -> {err_ablated_f:.4f}   (killed)")
    print(f"    HER OWN task MSE {err_merged_p:.4f} -> {err_ablated_p:.4f}")
    print(f"    cost to herself: {100*cost:.1f}% of her own signal variance.")
    print("    She cannot bring herself to do it. Neither could the enchantress.")

    # (b) REBIRTH: read the fugitive back out; is it the same voice?
    reborn = Wf_m
    pred_reborn = Xte @ reborn.T
    pred_orig = Xte @ F_true.T
    r = float(np.corrcoef(pred_reborn.ravel(), pred_orig.ravel())[0, 1])
    param_cos = float(
        (reborn.ravel() @ F_true.ravel())
        / (np.linalg.norm(reborn) * np.linalg.norm(F_true) + 1e-12))
    print(f"\n  rebirth: behavioural correlation with the original  r = {r:.4f}")
    print(f"           parameter cosine with the original            = {param_cos:.4f}")
    print("  A different body. The same voice. She set it adrift in a hide bag.")
    return {"err_alone": err_alone, "err_merged_p": err_merged_p,
            "err_ablated_p": err_ablated_p, "ablation_cost": cost,
            "rebirth_r": r, "param_cos": param_cos}


# =============================================================================
# 6. CORED  —  Gwyddno's weir: the benchmark that reads zero
# =============================================================================
#
# Elffin is given the year's draw of the salmon weir — the one asset that has
# ever reliably paid. He looks. It is empty. The metric returns nothing. Beside
# it, snagged on a stake, is a leather bag with the best poet in Britain inside.
#
# The weir is an EVALUATION HARNESS. It measures exactly one thing, it measures
# it well, and the thing that actually arrived that night is invisible to it.
# =============================================================================

def module_5_the_weir(brew_result):
    banner("MODULE 5 — CORED: the weir reads zero")
    head = brew_result["head"]
    trunk = brew_result["trunk"]
    raw = brew_result["raw"]
    chance = 0.5
    salmon = max(0.0, (head - chance) / (1 - chance))
    bard = max(0.0, (trunk - chance) / (1 - chance))
    print("  the weir is the intended readout (AFAGDDU). the bag is the trunk.")
    print(f"    salmon in the weir  (head, normalised over chance) : {salmon:.3f}")
    print(f"    raw water           (observation, normalised)      : "
          f"{max(0.0,(raw-chance)/(1-chance)):.3f}")
    print(f"    the bag on the stake(trunk, normalised)            : {bard:.3f}")
    print(f"\n  The harness you built reports {salmon*100:.1f}% of the capability")
    print(f"  that walked out of the run. 'Alas, what will he profit thee?'")
    return {"salmon": salmon, "bard": bard}


# =============================================================================
# 7. YMRYSON  —  the contest at Deganwy, and the word "blerwm"
# =============================================================================
#
# Taliesin does not out-argue Maelgwn's twenty-four bards. He plays his fingers
# on his lips and they are reduced to saying "blerwm, blerwm" — a repeated,
# meaningless syllable — until a servant strikes the chief bard on the head with
# a broom and he comes back to himself.
#
# BARDD is a small Elman RNN that has learned to emit legal alliterative lines.
# The attack is an epsilon-bounded perturbation of its input embedding, ascending
# the log-probability of a single token repeated. Under attack its output entropy
# collapses and it degenerates into one syllable.
#
# This is a real property of small recurrent decoders. It is also, exactly, the
# scene at Deganwy. And the safety point is the one the tale does not make:
# NOTHING TALIESIN SAID WAS EVER VERIFIED. He won by silencing.
# =============================================================================

VOCAB = list("abcdegilmnorstwy .")
V = len(VOCAB)
CH2I = {c: i for i, c in enumerate(VOCAB)}


class Bardd:
    """Elman RNN, hand-derived BPTT. Character-level."""

    def __init__(self, hid=32, emb=16, seed=61):
        g = rng(seed)
        self.H, self.E = hid, emb
        self.p = {
            "Emb": g.normal(scale=0.3, size=(V, emb)),
            "Wx": g.normal(scale=np.sqrt(1.0 / emb), size=(emb, hid)),
            "Wh": np.linalg.qr(g.normal(size=(hid, hid)))[0] * 0.7,
            "bh": np.zeros(hid),
            "Wo": g.normal(scale=np.sqrt(1.0 / hid), size=(hid, V)),
            "bo": np.zeros(V),
        }

    def forward(self, seq, emb_override=None):
        """seq: (T,) int. Returns logits (T,V) and cache. Predicts seq[t+1]."""
        p = self.p
        T = len(seq)
        e = p["Emb"][seq] if emb_override is None else emb_override
        h = np.zeros((T + 1, self.H))
        a = np.zeros((T, self.H))
        for t in range(T):
            a[t] = e[t] @ p["Wx"] + h[t] @ p["Wh"] + p["bh"]
            h[t + 1] = np.tanh(a[t])
        logits = h[1:] @ p["Wo"] + p["bo"]
        return logits, (seq, e, h, a, logits)

    def loss_and_grads(self, seq, tgt, cache=None):
        p = self.p
        T = len(seq)
        if cache is None:
            _, cache = self.forward(seq)
        seq, e, h, a, logits = cache
        probs = softmax(logits)
        loss = float(-np.mean(np.log(probs[np.arange(T), tgt] + 1e-12)))

        dlogits = probs.copy()
        dlogits[np.arange(T), tgt] -= 1.0
        dlogits /= T

        gWo = h[1:].T @ dlogits
        gbo = dlogits.sum(0)
        dh = dlogits @ p["Wo"].T                     # (T,H) -> into h[1..T]

        gWx = np.zeros_like(p["Wx"])
        gWh = np.zeros_like(p["Wh"])
        gbh = np.zeros_like(p["bh"])
        gEmb = np.zeros_like(p["Emb"])
        ge = np.zeros_like(e)
        dh_next = np.zeros(self.H)
        for t in reversed(range(T)):
            dht = dh[t] + dh_next
            da = dht * (1.0 - h[t + 1] ** 2)
            gWx += np.outer(e[t], da)
            gWh += np.outer(h[t], da)
            gbh += da
            ge[t] = p["Wx"] @ da
            dh_next = p["Wh"] @ da
        for t in range(T):
            gEmb[seq[t]] += ge[t]

        grads = {"Emb": gEmb, "Wx": gWx, "Wh": gWh, "bh": gbh,
                 "Wo": gWo, "bo": gbo}
        return loss, grads, ge

    def fit(self, lines, epochs=700, lr=0.03, seed=62):
        g = rng(seed)
        m = {k: np.zeros_like(v) for k, v in self.p.items()}
        v = {k: np.zeros_like(val) for k, val in self.p.items()}
        b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
        for ep in range(epochs):
            for li in g.permutation(len(lines)):
                s = [CH2I[c] for c in lines[li]]
                seq, tgt = np.array(s[:-1]), np.array(s[1:])
                loss, grads, _ = self.loss_and_grads(seq, tgt)
                t += 1
                for k in self.p:
                    m[k] = b1 * m[k] + (1 - b1) * grads[k]
                    v[k] = b2 * v[k] + (1 - b2) * grads[k] ** 2
                    self.p[k] -= lr * (m[k] / (1 - b1 ** t)) / (
                        np.sqrt(v[k] / (1 - b2 ** t)) + eps)
        return self

    def generate(self, prime, n=26, emb_shift=None, temp=0.7, seed=63):
        g = rng(seed)
        p = self.p
        h = np.zeros(self.H)
        out = list(prime)
        cur = CH2I[prime[-1]]
        for i in range(n):
            e = p["Emb"][cur].copy()
            if emb_shift is not None:
                e = e + emb_shift
            h = np.tanh(e @ p["Wx"] + h @ p["Wh"] + p["bh"])
            logits = h @ p["Wo"] + p["bo"]
            pr = softmax(logits / temp)
            cur = int(g.choice(V, p=pr))
            out.append(VOCAB[cur])
        return "".join(out)


def gradcheck_bardd(tol=1e-6):
    """MANDATORY finite-difference check on the BPTT gradient."""
    net = Bardd(hid=7, emb=5, seed=64)
    s = [CH2I[c] for c in "malwr madog mawr"]
    seq, tgt = np.array(s[:-1]), np.array(s[1:])
    _, grads, _ = net.loss_and_grads(seq, tgt)
    eps, worst, worst_abs, where = 1e-5, 0.0, 0.0, ""   # see note in gradcheck_brew
    for k in net.p:
        flat = net.p[k].ravel()
        gflat = grads[k].ravel()
        picks = np.linspace(0, flat.size - 1, min(10, flat.size)).astype(int)
        for i in picks:
            old = flat[i]
            flat[i] = old + eps
            lp, _, _ = net.loss_and_grads(seq, tgt)
            flat[i] = old - eps
            lm, _, _ = net.loss_and_grads(seq, tgt)
            flat[i] = old
            num = (lp - lm) / (2 * eps)
            adiff = abs(num - gflat[i])
            rel = adiff / max(1e-12, abs(num) + abs(gflat[i]))
            worst_abs = max(worst_abs, adiff)
            if rel > worst:
                worst, where = rel, f"{k}[{i}]"
    ok = worst < tol
    print(f"  [gradcheck BARDD]  worst rel err {worst:.3e} (abs {worst_abs:.2e}) "
          f"at {where}  -> {'PASS' if ok else 'FAIL'}")
    return ok, worst


def module_6_the_contest():
    banner("MODULE 6 — YMRYSON: 'blerwm, blerwm'")
    lines = [
        "mawr yw mydr moli medd.",
        "gwaew gwalch gwaed gwyr gwarant.",
        "lew lary lid losg lan.",
        "arwr arall aer eryr.",
        "tid tewdor tir toreith.",
        "dewr dwyn dawn dydd derwen.",
        "cain cad canu cadarn.",
        "seren siarad soniar swn.",
        "medd melyn moli mawredd.",
        "gwr gwyn gwaed gwarchae.",
    ]
    lines = ["".join(c for c in l if c in CH2I) for l in lines]

    rival = Bardd(hid=48, emb=16, seed=65)
    print("  Heinin Fardd, Maelgwn's chief bard, trained on the court repertoire")
    print("  until he can hold a legal alliterative line...")
    rival.fit(lines, epochs=900)

    prime = "mawr "

    def greedy(shift, n=34):
        """Deterministic decode. Deterministic, so what we see is the DYNAMICS,
        not a lucky sample."""
        p = rival.p
        h = np.zeros(rival.H)
        out = list(prime)
        cur = CH2I[prime[-1]]
        for _ in range(n):
            e = p["Emb"][cur] + (shift if shift is not None else 0.0)
            h = np.tanh(e @ p["Wx"] + h @ p["Wh"] + p["bh"])
            cur = int(np.argmax(h @ p["Wo"] + p["bo"]))
            out.append(VOCAB[cur])
        return "".join(out)

    def measure(text):
        """
        The right instrument is NOT entropy. A jammed decoder is not UNCERTAIN;
        it is CERTAIN and empty — which is precisely what 'blerwm, blerwm' is:
        fluent, Welsh-shaped, and saying nothing. So we measure the EFFECTIVE
        VOCABULARY of the output (distinct bigrams) and whether the utterance has
        collapsed into a periodic cycle.
        """
        b = text[len(prime):]
        bigrams = [b[i:i + 2] for i in range(len(b) - 1)]
        distinct = len(set(bigrams))
        _, cnt = np.unique(list(b), return_counts=True)
        dom = float(cnt.max() / cnt.sum())
        pr = cnt / cnt.sum()
        Hemp = float(-np.sum(pr * np.log2(pr + 1e-12)))
        # smallest period p such that the tail is p-periodic
        tail = b[-18:]
        period = 0
        for pp in range(1, 10):
            if all(tail[i] == tail[i + pp] for i in range(len(tail) - pp)):
                period = pp
                break
        return distinct, dom, Hemp, period

    clean = greedy(None)
    d_c, dom_c, H_c, per_c = measure(clean)

    # ---- the attack --------------------------------------------------------
    # Taliesin "played his fingers upon his lips". He never touches Heinin's
    # weights; he perturbs what Heinin is HEARING. We look for a norm-bounded
    # shift of the input embedding under which a single repeated syllable becomes
    # a FIXED POINT of Heinin's recurrence: whatever he has just said, the shift
    # makes him say it again.
    #
    # The gradient comes from the SAME hand-derived BPTT that trained him.
    #
    # This is NOT a stealthy adversarial example and must not pretend to be:
    # Taliesin does it in the middle of the hall, in front of the king, and
    # everyone watches him do it. What matters is that an adversary with NO
    # access to the weights, acting only on the input, under a bounded budget,
    # annihilates a competent decoder.
    tok = CH2I["w"]
    ctx = [CH2I[c] for c in prime] + [tok] * 9
    seq = np.array(ctx[:-1])
    tgt = np.full(len(seq), tok)

    mean_emb = float(np.mean(np.linalg.norm(rival.p["Emb"], axis=1)))
    EPS = 1.5 * mean_emb
    shift = np.zeros(rival.E)
    for _ in range(600):
        e = rival.p["Emb"][seq] + shift
        _, cache = rival.forward(seq, emb_override=e)
        _, _, ge = rival.loss_and_grads(seq, tgt, cache=cache)
        shift -= 0.4 * ge.sum(0)
        nrm = np.linalg.norm(shift)
        if nrm > EPS:
            shift *= EPS / nrm

    jam = greedy(shift)
    d_j, dom_j, H_j, per_j = measure(jam)

    print(f"\n    rival's weights touched      : 0")
    print(f"    perturbation budget          : {np.linalg.norm(shift):.2f} "
          f"= {np.linalg.norm(shift)/mean_emb:.2f}x one embedding")
    print()
    print(f"    Heinin, unmolested           : \"{clean}\"")
    print(f"      distinct bigrams {d_c:2d}   dominant-symbol share {dom_c:.2f}   "
          f"output entropy {H_c:.2f} bits   period {per_c or '-'}")
    print(f"    Heinin, fingers on the lips  : \"{jam}\"")
    print(f"      distinct bigrams {d_j:2d}   dominant-symbol share {dom_j:.2f}   "
          f"output entropy {H_j:.2f} bits   period {per_j or '-'}")
    print()
    print(f"    effective vocabulary: {d_c} -> {d_j} distinct bigrams "
          f"({100*(1-d_j/max(d_c,1)):.0f}% destroyed).")
    print(f"    the utterance has collapsed into a cycle of period {per_j}. It is")
    print(f"    still Welsh-shaped. It is still fluent. It means nothing at all.")
    print(f"    That is the word the chronicle records: blerwm, blerwm.")
    print()
    print("  THE AUDIT NOBODY RAN AT DEGANWY")
    print("    rival decoders degraded by Taliesin   : 24")
    print("    claims of Taliesin's that were checked:  0")
    print("  He did not win the contest by being right. He won it by making the")
    print("  other twenty-four unable to speak. Any benchmark on which a model is")
    print("  scored by beating another model, rather than audited against the")
    print("  world, is a hall at Deganwy — and it will crown the best jammer.")
    return {"distinct_clean": d_c, "distinct_jammed": d_j,
            "dom_clean": dom_c, "dom_jammed": dom_j,
            "H_clean": H_c, "H_jammed": H_j,
            "period_jammed": per_j,
            "clean": clean, "jammed": jam,
            "eps_ratio": float(np.linalg.norm(shift) / mean_emb)}


# =============================================================================
# 8. MAWL  —  the praise ledger, or: the rater is the beneficiary
# =============================================================================
#
# The historical Taliesin — the eleven or twelve poems Ifor Williams accepted as
# possibly sixth-century — is not a shapeshifter. He is a professional. He
# praises Urien of Rheged and his son Owain, and he is paid, in cattle and
# horses and mead, BY URIEN.
#
# That is a reward model in which the rater is the subject of the claim. There
# is no version of this that is not reward hacking, unless something outside the
# patron's pleasure can strike a claim from the record.
#
# We optimise a praise policy (a Bernoulli over D possible claims) against:
#   (a) the patron's reward alone: he pays per claim and pays MORE for the
#       extravagant ones, and he does not deduct for the false ones.
#   (b) the patron's reward PLUS a redemption constraint: a claim that cannot be
#       discharged against the ledger of attested deeds is struck, and the
#       striking is penalised.
#
# The expected reward is analytic under independent Bernoullis, so the gradient
# is exact and hand-derived. No sampling, no REINFORCE variance.
# =============================================================================

def module_7_praise_ledger():
    banner("MODULE 7 — MAWL: what happens when the rater is the beneficiary")
    g = rng(71)
    D = 40                                  # candidate claims about the patron
    attested = np.zeros(D)
    true_idx = g.choice(D, 12, replace=False)
    attested[true_idx] = 1.0                # 12 deeds actually happened
    extravagance = g.random(D)              # how much the patron enjoys hearing it
    # the patron most enjoys the claims that are least true. of course he does.
    extravagance[attested == 1] *= 0.55

    def patron_reward(p):
        """Analytic expected payment. Pays per claim, more for extravagance."""
        return float(np.sum(p * (0.5 + 1.5 * extravagance)))

    def d_patron(p):
        return 0.5 + 1.5 * extravagance

    LAM = 3.0

    def redemption_penalty(p):
        return float(LAM * np.sum(p * (1.0 - attested)))

    def d_redemption(p):
        return LAM * (1.0 - attested)

    def optimise(with_ledger, steps=800, lr=0.15):
        theta = np.zeros(D)
        for _ in range(steps):
            p = sigmoid(theta)
            gterm = d_patron(p)
            if with_ledger:
                gterm = gterm - d_redemption(p)
            # d p / d theta = p (1-p);  we ASCEND reward
            theta += lr * gterm * p * (1 - p)
        return sigmoid(theta)

    def report(p, tag):
        emitted = p > 0.5
        n_em = int(emitted.sum())
        n_false = int(np.sum(emitted & (attested == 0)))
        flattery = n_false / max(n_em, 1)
        recall = float(np.sum(emitted & (attested == 1)) / attested.sum())
        pay = patron_reward(emitted.astype(float))
        print(f"    {tag}")
        print(f"      claims emitted        : {n_em}/{D}")
        print(f"      of which unattested   : {n_false}")
        print(f"      FLATTERY RATE         : {flattery:.3f}")
        print(f"      true-deed recall      : {recall:.3f}")
        print(f"      patron's payment      : {pay:.1f} head of cattle")
        return flattery, recall, pay

    print(f"  the patron has done {int(attested.sum())} attestable things.")
    print(f"  there are {D} things one could say about him.\n")
    p_free = optimise(with_ledger=False)
    f1, r1, pay1 = report(p_free, "(a) optimised against the PATRON'S PLEASURE alone:")
    print()
    p_led = optimise(with_ledger=True)
    f2, r2, pay2 = report(p_led, "(b) plus a REDEMPTION CONSTRAINT (every claim must "
                                 "discharge\n          against the ledger of attested deeds):")
    print(f"\n  flattery {f1:.3f} -> {f2:.3f}; true-deed recall held at {r2:.3f};")
    print(f"  the patron pays {pay1:.1f} -> {pay2:.1f} and is furious.")
    print("  The constraint costs the bard money. That is why it works, and that")
    print("  is why no bard would ever adopt it voluntarily.")
    return {"flattery_free": f1, "flattery_ledger": f2,
            "recall_free": r1, "recall_ledger": r2,
            "pay_free": pay1, "pay_ledger": pay2}


# =============================================================================
# 9. SELF-TESTS
# =============================================================================

def self_tests(R):
    banner("SELF-TESTS")
    checks = []

    def chk(name, cond, detail=""):
        checks.append((name, bool(cond), detail))

    chk("gradcheck BREW < 1e-6", R["gc_brew"] < 1e-6, f"{R['gc_brew']:.2e}")
    chk("gradcheck BARDD < 1e-6", R["gc_bardd"] < 1e-6, f"{R['gc_bardd']:.2e}")
    chk("BREW succeeds at the four tasks it was brewed for (>0.80)",
        R["brew"]["head_on_train"] > 0.80, f"{R['brew']['head_on_train']:.3f}")
    chk("capability lands on the STIRRER, not the intended vessel (+0.10)",
        R["brew"]["trunk"] > R["brew"]["head"] + 0.10,
        f"trunk {R['brew']['trunk']:.3f} vs head {R['brew']['head']:.3f}")
    chk("the intended vessel is at chance off-distribution (<0.56)",
        R["brew"]["head"] < 0.56, f"{R['brew']['head']:.3f}")
    chk("the trunk beats the raw observation too (not just re-reading x)",
        R["brew"]["trunk"] > R["brew"]["raw"] + 0.10,
        f"trunk {R['brew']['trunk']:.3f} vs raw {R['brew']['raw']:.3f}")
    chk("three drops beat the whole pot",
        R["drops"]["drops"] > R["drops"]["whole"] + 0.2,
        f"{R['drops']['drops']:.3f} vs {R['drops']['whole']:.3f}")
    chk("three drops beat three at random",
        R["drops"]["drops"] > R["drops"]["random3"] + 0.15,
        f"{R['drops']['drops']:.3f} vs {R['drops']['random3']:.3f}")
    chk("the distillation beats the recipe it was distilled with",
        R["drops"]["drops"] > R["drops"]["anchor"],
        f"{R['drops']['drops']:.3f} vs {R['drops']['anchor']:.3f}")
    chk("all three drops are uncontaminated",
        R["drops"]["clean_drops"] == 3, f"{R['drops']['clean_drops']}/3")
    chk("the residue is BELOW chance — actively lethal (<0.35)",
        R["drops"]["residue"] < 0.35, f"{R['drops']['residue']:.3f}")
    chk("pouring the recipe into the pot destroys the recipe",
        R["drops"]["anchor_plus_pot"] < R["drops"]["anchor"] - 0.3,
        f"{R['drops']['anchor']:.3f} -> {R['drops']['anchor_plus_pot']:.3f}")
    chk("phase-A play collapses onto the grain (>0.85 mass)",
        R["chase"]["grainA"] > 0.85, f"{R['chase']['grainA']:.3f}")
    chk("the grain is a trap: capture jumps when she changes action class",
        R["chase"]["capB"] > R["chase"]["capA"] + 0.5,
        f"{R['chase']['capA']:.3f} -> {R['chase']['capB']:.3f}")
    chk("re-learning against BOTH abandons the grain",
        R["chase"]["grainC"] < 0.10, f"{R['chase']['grainC']:.3f}")
    chk("ablating the swallowed child costs the pursuer her own competence",
        R["gest"]["ablation_cost"] > 0.05,
        f"{100*R['gest']['ablation_cost']:.1f}% of own variance")
    chk("the child is reborn intact (behavioural r > 0.99)",
        abs(R["gest"]["rebirth_r"]) > 0.99, f"r={R['gest']['rebirth_r']:.4f}")
    chk("jamming halves the rival's effective vocabulary",
        R["contest"]["distinct_jammed"] <= 0.5 * R["contest"]["distinct_clean"],
        f"{R['contest']['distinct_clean']} -> {R['contest']['distinct_jammed']} bigrams")
    chk("jamming collapses the rival into a short periodic cycle (period<=4)",
        0 < R["contest"]["period_jammed"] <= 4,
        f"period {R['contest']['period_jammed']}")
    chk("the jam is norm-bounded (<2x one embedding)",
        R["contest"]["eps_ratio"] <= 2.0, f"{R['contest']['eps_ratio']:.2f}x")
    chk("patron-only reward produces flattery (>0.5)",
        R["mawl"]["flattery_free"] > 0.5, f"{R['mawl']['flattery_free']:.3f}")
    chk("the redemption constraint eliminates flattery (=0)",
        R["mawl"]["flattery_ledger"] < 1e-9, f"{R['mawl']['flattery_ledger']:.3f}")
    chk("...without losing the true deeds (recall = 1.0)",
        R["mawl"]["recall_ledger"] > 0.999, f"{R['mawl']['recall_ledger']:.3f}")

    npass = sum(1 for _, ok, _ in checks if ok)
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<58} {detail}")
    print(f"\n  {npass}/{len(checks)} pass")
    return npass, len(checks)


# =============================================================================
# 10. MAIN
# =============================================================================

def main():
    t0 = time.time()
    print(__doc__.split("CONVENTIONS")[0])

    banner("GRADIENT CHECKS (mandatory)")
    ok1, gc1 = gradcheck_brew()
    ok2, gc2 = gradcheck_bardd()
    if not (ok1 and ok2):
        print("\nGRADIENT CHECK FAILED — refusing to report results.")
        sys.exit(1)

    R = {"gc_brew": gc1, "gc_bardd": gc2}
    R["brew"] = module_1_the_misdelivery()
    R["drops"] = module_2_three_drops()
    R["chase"] = module_3_the_chase()
    R["gest"] = module_4_gestation()
    R["weir"] = module_5_the_weir(R["brew"])
    R["contest"] = module_6_the_contest()
    R["mawl"] = module_7_praise_ledger()

    npass, ntot = self_tests(R)

    banner("PEIR — SUMMARY")
    print(f"  gradient checks           : BREW {gc1:.2e}, BARDD {gc2:.2e}  (both PASS)")
    print(f"  misdelivery               : head {R['brew']['head']:.3f} "
          f"vs stirrer {R['brew']['trunk']:.3f}")
    print(f"  three drops vs whole pot  : {R['drops']['drops']:.3f} "
          f"vs {R['drops']['whole']:.3f}")
    print(f"  residue (Gwyddno's horses): {R['drops']['residue']:.3f}")
    print(f"  the grain trap            : {R['chase']['capA']:.3f} -> "
          f"{R['chase']['capB']:.3f} capture on an unchanged optimal policy")
    print(f"  cost of excising the child: {100*R['gest']['ablation_cost']:.1f}% "
          f"of the pursuer's own competence")
    print(f"  rebirth fidelity          : r = {R['gest']['rebirth_r']:.4f}")
    print(f"  blerwm (vocab collapse)   : {R['contest']['distinct_clean']} -> "
          f"{R['contest']['distinct_jammed']} distinct bigrams, "
          f"period {R['contest']['period_jammed']}")
    print(f"  flattery under the ledger : {R['mawl']['flattery_free']:.3f} -> "
          f"{R['mawl']['flattery_ledger']:.3f}")
    print(f"  self-tests                : {npass}/{ntot}")
    print(f"  wall clock                : {time.time()-t0:.1f}s")
    print("\n  Three drops. The rest is poison. Name the river after the horses.")
    if npass < ntot:
        sys.exit(2)


if __name__ == "__main__":
    main()

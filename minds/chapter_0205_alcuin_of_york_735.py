#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Chapter 0205 — Alcuin of York (c. 735 - 804)
Encyclopedia of Lost Minds: Echoes on AI
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 11 Minds 201 - 220 Available on Amazon https://www.amazon.com/dp/B0HKF7TRFF
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0205_alcuin_of_york_735 - Alcuin of York (c. 735 - 804)
================================================================================  

THE SCRIPTORIUM ENGINE
An error-correcting attractor network built from scratch in pure NumPy.

--------------------------------------------------------------------------------
WHY THIS ARCHITECTURE (and not a Transformer / MoE / attention stack)
--------------------------------------------------------------------------------
Alcuin's whole life was organised around a single cognitive act that most
histories reduce to a footnote: *emendatio* — correction. The Carolingian
programme he steered (renovatio / correctio) was not, at its root, about
"education" in the modern classroom sense. It was about a physical, brutal
information problem. Sacred texts were transmitted by hand-copying. Every scribe
introduced errors. Over generations of copying the signal decayed: a Psalter
copied from a copy of a copy drifted away from its exemplar until, eventually,
the words no longer meant what they were meant to mean — and, Alcuin believed,
a corrupted text bred corrupted faith and corrupted conduct ("knowing precedes
doing").

His remedy was not to teach harder. It was to build a *correcting channel*:
  1. Fix a canonical exemplar (the corrected Bible, the standard grammar).
  2. Encode it in a low-error, legible script (Carolingian minuscule).
  3. Pull every corrupted copy back toward that exemplar (emendatio).
  4. Re-emit the corrected copy, and repeat across the whole empire.

That is, formally, an *attractor-based denoising code*. This file implements it
as a trainable neural network whose mechanism IS Alcuin's mind:

  * an ENCODER          — the scribe receiving a (corrupted) copy into working
                          scholarly representation;
  * a CANONICAL CODEBOOK — the learned set of exemplars (the "corrected
                          archetypes"); the fixed points the mind restores toward;
  * an EMENDATIO LOOP    — recurrent soft-projection that drags a corrupted
                          representation toward the nearest canonical exemplar
                          (the correcting channel; a contraction, so it settles);
  * a DECODER            — the re-copied, corrected output that is transmitted on.

The signature demonstration (see transmission_chain_demo) reproduces the thing
Alcuin actually feared and actually fixed: a chain of copies-of-copies.
  - WITHOUT a correcting channel, fidelity collapses to noise (a "dark age").
  - WITH the trained Scriptorium in the loop, fidelity is held across
    generations. That gap is the Carolingian Renaissance, in miniature.

--------------------------------------------------------------------------------
WHAT IS ACTUALLY TRAINED
--------------------------------------------------------------------------------
A denoising auto-encoder with an explicit attractor bottleneck. Given corrupted
copies of hidden canonical patterns, it must reconstruct the clean canon. The
codebook, encoder and decoder are all learned by backprop (manual, verified
against finite differences). No autograd, no ML frameworks — only NumPy.

Run:  python3 chapter_0190_alcuin_of_york_735.py
================================================================================
"""

import numpy as np

RNG = np.random.default_rng(190)   # 190 = the figure's number in the corpus


# ------------------------------------------------------------------------------
# small helpers
# ------------------------------------------------------------------------------
def softmax(v):
    """Numerically stable softmax over the last axis of a 1-D vector."""
    v = v - np.max(v)
    e = np.exp(v)
    return e / np.sum(e)


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def cosine(a, b):
    return float(np.dot(unit(a), unit(b)))


# ==============================================================================
# THE SCRIPTORIUM ENGINE
# ==============================================================================
class Scriptorium:
    """
    A denoising attractor network.

    Dimensions
    ----------
    D : size of a "text" vector (the copy that travels through the world)
    H : size of the scholar's internal representation
    K : number of canonical exemplars held in the codebook (the archetypes)

    Learnable parameters
    --------------------
    We (H,D), be (H,)  : encoder  (the scribe's reading hand)
    C  (K,H)           : canonical codebook (the corrected exemplars)
    Wd (D,H), bd (D,)  : decoder  (the re-copied output)

    Fixed hyper-parameters
    ----------------------
    alpha : correction strength per emendatio step (contraction rate), in (0,1)
    beta  : sharpness of "which exemplar does this belong to?" (softmax temp)
    T     : number of emendatio steps applied during a single correction
    """

    def __init__(self, D=16, H=24, K=6, alpha=0.5, beta=4.0, T=2):
        self.D, self.H, self.K = D, H, K
        self.alpha, self.beta, self.T = alpha, beta, T

        # Xavier-ish initialisation (kept in float64 for the gradient check).
        self.We = RNG.standard_normal((H, D)) * np.sqrt(1.0 / D)
        self.be = np.zeros(H)
        self.C  = RNG.standard_normal((K, H)) * np.sqrt(1.0 / H)
        self.Wd = RNG.standard_normal((D, H)) * np.sqrt(1.0 / H)
        self.bd = np.zeros(D)

    # -- parameter (de)serialisation, used only by the gradient check ----------
    def get_params(self):
        return {"We": self.We, "be": self.be, "C": self.C,
                "Wd": self.Wd, "bd": self.bd}

    def set_params(self, p):
        self.We, self.be, self.C = p["We"], p["be"], p["C"]
        self.Wd, self.bd = p["Wd"], p["bd"]

    # --------------------------------------------------------------------------
    # FORWARD  (one corrupted copy in -> one corrected copy out)
    # --------------------------------------------------------------------------
    def forward(self, x):
        """
        x : (D,) a received copy (possibly corrupted).
        Returns (xhat, cache) where xhat is the corrected re-emission.
        """
        z  = self.We @ x + self.be          # pre-activation
        h0 = np.tanh(z)                      # scholar reads copy into working rep
        h  = h0

        steps = []
        for _ in range(self.T):             # ---- emendatio (the correcting loop)
            s = self.C @ h                  # (K,) affinity to each exemplar
            p = softmax(self.beta * s)      # (K,) soft "this belongs to exemplar k"
            m = p @ self.C                  # (H,) the canonical reconstruction
            h_new = (1 - self.alpha) * h + self.alpha * m   # pull toward canon
            steps.append((h, s, p, m))      # cache pre-step state for backprop
            h = h_new

        xhat = self.Wd @ h + self.bd        # re-copy: emit corrected text
        cache = {"x": x, "z": z, "h0": h0, "steps": steps, "hT": h}
        return xhat, cache

    # --------------------------------------------------------------------------
    # BACKWARD  (analytic gradients; verified against finite differences below)
    # --------------------------------------------------------------------------
    def backward(self, cache, target):
        x, z, h0, steps, hT = (cache[k] for k in ("x", "z", "h0", "steps", "hT"))
        xhat = self.Wd @ hT + self.bd
        D = self.D

        # loss = 0.5 * mean_j (xhat_j - target_j)^2
        dxhat = (xhat - target) / D                     # (D,)

        gWd = np.outer(dxhat, hT)                        # (D,H)
        gbd = dxhat.copy()                               # (D,)
        dh  = self.Wd.T @ dxhat                          # grad wrt hT   (H,)

        gC = np.zeros_like(self.C)

        # walk the emendatio loop in reverse
        for (h_prev, s, p, m) in reversed(steps):
            # h_new = (1-alpha)*h_prev + alpha*m
            dh_direct = (1 - self.alpha) * dh
            dm = self.alpha * dh                         # (H,)

            # m = p @ C  ->  dp = C @ dm ;  gC += outer(p, dm)
            dp = self.C @ dm                             # (K,)
            gC += np.outer(p, dm)

            # p = softmax(beta*s) -> ds
            dot = float(p @ dp)
            ds = self.beta * p * (dp - dot)              # (K,)

            # s = C @ h_prev -> gC += outer(ds, h_prev) ; dh_prev += C^T ds
            gC += np.outer(ds, h_prev)
            dh = dh_direct + self.C.T @ ds               # grad wrt h_prev

        # h0 = tanh(z)
        dz = dh * (1.0 - h0 * h0)                        # (H,)
        gWe = np.outer(dz, x)                            # (H,D)
        gbe = dz.copy()                                  # (H,)

        return {"We": gWe, "be": gbe, "C": gC, "Wd": gWd, "bd": gbd}

    @staticmethod
    def loss(xhat, target):
        d = xhat - target
        return 0.5 * float(np.mean(d * d))

    # --------------------------------------------------------------------------
    # INFERENCE conveniences
    # --------------------------------------------------------------------------
    def correct(self, x):
        """Return the corrected re-emission of a copy x (a single pass)."""
        xhat, _ = self.forward(x)
        return xhat

    def recognise(self, x, shelf):
        """
        Alcuin's riddle move, done properly: from a garbled clue, first RESTORE
        the text (run the correcting channel), then RECOGNISE the restoration
        against the shelf of known exemplars. Returns the best exemplar index.

        This is faithful to how emendatio actually worked: the scholar did not
        classify the corrupt copy directly; he corrected it toward the canon and
        only then knew which text it was.
        """
        xhat = self.correct(x)
        sims = np.array([cosine(xhat, e) for e in shelf])
        return int(np.argmax(sims))

    def codebook_usage(self, x):
        """Diagnostic only: soft assignment of a copy over the internal codebook."""
        z = np.tanh(self.We @ x + self.be)
        p = softmax(self.beta * (self.C @ z))
        return int(np.argmax(p)), p


# ==============================================================================
# DATA — canonical exemplars and their corrupted copies
# ==============================================================================
def make_canon(K, D, seed=7):
    """K distinct, well-separated canonical 'texts' living on the unit sphere."""
    g = np.random.default_rng(seed)
    M = g.standard_normal((K, D))
    return np.stack([unit(row) for row in M])


def corrupt(x, noise=0.6, drop=0.25, rng=RNG):
    """
    Model a scribe's copy: additive noise + random omission (masked to zero).
    'noise' = jitter of the hand; 'drop' = fraction of characters lost/blotted.
    """
    y = x + noise * rng.standard_normal(x.shape)
    if drop > 0:
        mask = rng.random(x.shape) > drop
        y = y * mask
    return y


# ==============================================================================
# GRADIENT CHECK  (mandatory) — analytic vs. central finite differences
# ==============================================================================
def gradient_check():
    print("-" * 70)
    print("FINITE-DIFFERENCE GRADIENT CHECK")
    print("-" * 70)
    net = Scriptorium(D=8, H=10, K=4, alpha=0.5, beta=3.0, T=2)

    # one fixed (corrupted -> clean) example
    canon = make_canon(4, 8, seed=1)
    target = canon[2]
    x = corrupt(target, noise=0.5, drop=0.2, rng=np.random.default_rng(3))

    xhat, cache = net.forward(x)
    analytic = net.backward(cache, target)

    eps = 1e-6
    params = net.get_params()
    worst = 0.0
    for name in ("We", "be", "C", "Wd", "bd"):
        P = params[name]
        num = np.zeros_like(P)
        it = np.nditer(P, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = P[idx]
            P[idx] = orig + eps
            lp = net.loss(net.forward(x)[0], target)
            P[idx] = orig - eps
            lm = net.loss(net.forward(x)[0], target)
            P[idx] = orig
            num[idx] = (lp - lm) / (2 * eps)
            it.iternext()
        a = analytic[name]
        rel = np.linalg.norm(a - num) / (np.linalg.norm(a) + np.linalg.norm(num) + 1e-12)
        worst = max(worst, rel)
        flag = "OK" if rel < 1e-5 else "FAIL"
        print(f"  {name:>3}: relative error = {rel:.3e}   [{flag}]")
    print(f"  worst relative error = {worst:.3e}")
    assert worst < 1e-5, "Gradient check FAILED"
    print("  -> gradients verified.\n")
    return worst


# ==============================================================================
# TRAINING LOOP  (real SGD/Adam on the denoising objective)
# ==============================================================================
def train(net, canon, epochs=1500, batch=32, lr=5e-3,
          noise=0.6, drop=0.25, verbose=True):
    D, K = net.D, net.K
    # Adam state
    m = {k: np.zeros_like(v) for k, v in net.get_params().items()}
    v = {k: np.zeros_like(val) for k, val in net.get_params().items()}
    b1, b2, eps = 0.9, 0.999, 1e-8

    hist = []
    rng = np.random.default_rng(42)
    lr0 = lr
    for ep in range(1, epochs + 1):
        # cosine decay of the learning rate (settles the emendatio bottleneck)
        lr = 0.5 * lr0 * (1 + np.cos(np.pi * (ep - 1) / epochs)) + 1e-4
        grads = {k: np.zeros_like(val) for k, val in net.get_params().items()}
        total = 0.0
        for _ in range(batch):
            k = rng.integers(K)
            clean = canon[k]
            x = corrupt(clean, noise, drop, rng=rng)
            xhat, cache = net.forward(x)
            total += net.loss(xhat, clean)
            g = net.backward(cache, clean)
            for key in grads:
                grads[key] += g[key]
        # average and Adam step
        params = net.get_params()
        for key in params:
            gk = grads[key] / batch
            m[key] = b1 * m[key] + (1 - b1) * gk
            v[key] = b2 * v[key] + (1 - b2) * (gk * gk)
            mh = m[key] / (1 - b1 ** ep)
            vh = v[key] / (1 - b2 ** ep)
            params[key] -= lr * mh / (np.sqrt(vh) + eps)
        avg = total / batch
        hist.append(avg)
        if verbose and (ep == 1 or ep % 250 == 0):
            print(f"  epoch {ep:4d} | denoising loss = {avg:.5f}")
    return hist


# ==============================================================================
# SELF-TESTS / DEMONSTRATIONS
# ==============================================================================
def denoising_test(net, canon, trials=400, noise=0.6, drop=0.25):
    """Does correction actually reduce the distance to the canonical text?"""
    rng = np.random.default_rng(11)
    before, after = [], []
    for _ in range(trials):
        k = rng.integers(net.K)
        clean = canon[k]
        x = corrupt(clean, noise, drop, rng=rng)
        xhat = net.correct(x)
        before.append(cosine(x, clean))
        after.append(cosine(xhat, clean))
    return float(np.mean(before)), float(np.mean(after))


def assignment_test(net, canon, trials=400, noise=0.55, drop=0.2):
    """Alcuin's riddle: from a garbled clue, name the canonical exemplar,
    by correcting first and recognising the restoration against the shelf."""
    rng = np.random.default_rng(13)
    correct = 0
    for _ in range(trials):
        k = rng.integers(net.K)
        x = corrupt(canon[k], noise, drop, rng=rng)
        pred = net.recognise(x, canon)
        correct += (pred == k)
    return correct / trials


def transmission_chain_demo(net, canon, generations=40, noise=0.3, drop=0.1,
                            passes=2):
    """
    THE SIGNATURE DEMONSTRATION.

    Two scriptoria copy the same exemplar down the generations.
      * 'raw'  : each scribe copies the previous (corrupted) copy. No correction.
      * 'fixed': each scribe copies, then the Scriptorium restores toward canon.

    We track fidelity (cosine to the original exemplar) over the generations.
    Alcuin's claim: without a correcting channel, learning decays to noise;
    with one, it is held. The gap between the two curves is the Renaissance.
    """
    rng = np.random.default_rng(99)
    k = rng.integers(net.K)
    exemplar = canon[k]

    raw = exemplar.copy()
    fixed = exemplar.copy()
    raw_curve, fixed_curve = [], []
    for _ in range(generations):
        raw = corrupt(raw, noise, drop, rng=rng)             # decay accumulates
        c = corrupt(fixed, noise, drop, rng=rng)             # this generation's copy
        for _ in range(passes):                              # thorough emendatio
            c = net.correct(c)
        fixed = c
        raw_curve.append(cosine(raw, exemplar))
        fixed_curve.append(cosine(fixed, exemplar))
    return raw_curve, fixed_curve


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("=" * 70)
    print("THE SCRIPTORIUM ENGINE  —  Alcuin of York (c.735-804), Chapter 0190")
    print("An error-correcting attractor network: mind as emendatio.")
    print("=" * 70 + "\n")

    # 1) prove the gradients are right
    gradient_check()

    # 2) build the real task and train
    D, H, K = 16, 32, 6
    canon = make_canon(K, D, seed=7)
    net = Scriptorium(D=D, H=H, K=K, alpha=0.5, beta=5.0, T=3)

    print("-" * 70)
    print("TRAINING  (denoising: corrupted copy -> canonical exemplar)")
    print("-" * 70)
    hist = train(net, canon, epochs=2500, batch=32, lr=6e-3,
                 noise=0.5, drop=0.2)
    print(f"  final loss: {hist[-1]:.5f}  (started at {hist[0]:.5f})\n")

    # 3) does correction help a single copy?
    print("-" * 70)
    print("SELF-TEST 1 — single-copy emendatio")
    print("-" * 70)
    b, a = denoising_test(net, canon)
    print(f"  mean cosine-to-canon BEFORE correction : {b:.3f}")
    print(f"  mean cosine-to-canon AFTER  correction : {a:.3f}")
    print(f"  improvement                            : {a - b:+.3f}")
    assert a > b + 0.1, "correction should meaningfully restore the copy"
    print("  -> PASS: emendatio restores corrupted copies.\n")

    # 4) can the mind name the exemplar behind a garbled clue?
    print("-" * 70)
    print("SELF-TEST 2 — the riddle (recover exemplar from a corrupted clue)")
    print("-" * 70)
    acc = assignment_test(net, canon, noise=0.4, drop=0.12)
    print(f"  exemplar-recovery accuracy over garbled clues: {acc*100:.1f}%")
    assert acc > 0.75, "attractor assignment should be reliable"
    print("  -> PASS: the codebook behaves as a canon of archetypes.\n")

    # 5) THE SIGNATURE: transmission across generations
    print("-" * 70)
    print("SELF-TEST 3 — the scriptorium across generations (THE signature)")
    print("-" * 70)
    raw_curve, fixed_curve = transmission_chain_demo(net, canon, generations=40)
    gens_to_show = [0, 4, 9, 19, 39]
    print("   gen |   raw (no correction) | fixed (Scriptorium in loop)")
    for g in gens_to_show:
        print(f"   {g:3d} |        {raw_curve[g]:+.3f}        |        {fixed_curve[g]:+.3f}")
    raw_final = float(np.mean(raw_curve[-10:]))
    fix_final = float(np.mean(fixed_curve[-10:]))
    print(f"\n  fidelity over final 10 generations:")
    print(f"    raw   (learning left to decay): {raw_final:+.3f}")
    print(f"    fixed (correcting channel)    : {fix_final:+.3f}")
    assert fix_final > raw_final + 0.3, "the correcting channel must hold fidelity"
    print("  -> PASS: without correction, fidelity collapses toward noise;")
    print("           with the Scriptorium in the loop, it is held.")
    print("           That gap is the Carolingian Renaissance, in miniature.\n")

    print("=" * 70)
    print("ALL CHECKS PASSED.")
    print("The mind modelled here does not generate from nothing; it RESTORES.")
    print("Intelligence, for Alcuin, is fidelity across a noisy channel of time.")
    print("=" * 70)


if __name__ == "__main__":
    main()

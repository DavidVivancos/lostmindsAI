#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Chapter 0208 — Saichō (Dengyō Daishi, 767–822)
Encyclopedia of Lost Minds: Echoes on AI
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 11 Minds 201 - 220 Available on Amazon https://www.amazon.com/dp/B0HKF7TRFF
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0208_saicho_dengyo_daishi_767 - Saichō (Dengyō Daishi, 767–822)
================================================================================  

WHAT THIS FILE IS
-----------------
A from-scratch, pure-NumPy neural architecture that encodes the *specific*
cognitive signature of Saichō, founder of Japanese Tendai Buddhism — not a
generic classifier dressed in Buddhist vocabulary. It is trainable, has a
finite-difference gradient check that must pass, a real curriculum-driven
training loop, and a battery of self-tests. Run it directly:

    python3 chapter_0194_saicho_dengyo_daishi_767.py

THE ONE IDEA THAT IS SAICHŌ'S ALONE
-----------------------------------
Against the Hossō monk Tokuitsu (debate of 817–821), Saichō defended two
claims from the Lotus Sūtra that, taken together, are unusual among theories
of mind:

  1. RECOGNITION, NOT ACQUISITION. The awakened endpoint is *already latent*
     in every mind (universal buddha-nature / hongaku). Practice does not
     manufacture enlightenment from a blank slate; it removes the obscuration
     (klesha = "dust on the mirror") that hides a capacity already present.

  2. NO ICCHANTIKA. There is no class of beings categorically excluded from
     awakening. Tokuitsu held the Yogācāra "five natures" doctrine, in which
     some beings (icchantika) can *never* awaken. Saichō rejected this: the
     One Vehicle (ekayāna) is universal.

  3. A FIXED GRADED CURRICULUM. Saichō's institutional signature was the
     twelve-year mountain confinement on Mt. Hiei (Sange gakushōshiki,
     818–819): awakening is universal AND immediate-in-principle, yet is
     realized through a mandatory, ordered, embodied training arc.

We translate these into machinery, not metaphor:

  * The "buddha-nature" archetype matrix B is initialized once and then
    FROZEN. Learning never touches it. The awakened targets are given, not
    produced. All learning happens in the *practice* transforms that learn to
    reveal alignment with B — recognition, not acquisition.

  * A learned VEIL (obscuration gate) sits between the middle-way
    representation and its expression. A regularizer drives the veil down over
    training: the loss literally rewards *removing dust*.

  * A frozen MUTUAL-INCLUSION operator C couples the ten dharma-realm logits
    so each realm "contains" the others (Zhiyi's ichinen sanzen — "three
    thousand realms in a single thought-moment").

  * The Three Truths (empty / provisional / middle) are a real forward
    module: two linear views reconciled by a nonlinear middle way.

  * A CURRICULUM schedule raises defilement gradually across training stages,
    mirroring the graded mountain practice.

Self-tests then verify the doctrine operationally:
  - the gradient check passes (the machinery is real, not decorative);
  - loss falls, accuracy rises (practice works);
  - the veil mean falls (obscuration is removed = gradual awakening);
  - B is bit-for-bit unchanged (recognition, not acquisition);
  - EVERY realm reaches non-trivial recall (no icchantika — no class is
    permanently excluded).

Architecture name: HONGAKU UNVEILING NETWORK (HUN).
No external libraries beyond NumPy. No pretrained weights. No lookup tables of
"beliefs" — every boolean the model reports is downstream of trained weights.
================================================================================
"""

import numpy as np

# One fixed seed so the verified output pasted into the chapter is reproducible.
SEED = 767  # Saichō's traditional birth year.
rng = np.random.default_rng(SEED)


# ------------------------------------------------------------------------------
# Small numerical helpers (kept explicit so the math is auditable).
# ------------------------------------------------------------------------------
def glorot(shape):
    """Glorot/Xavier init — keeps forward/backward variances sane."""
    fan_in, fan_out = shape[1], shape[0]
    lim = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-lim, lim, size=shape)


def softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def sigmoid(z):
    # Numerically stable elementwise sigmoid.
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def cross_entropy(p, y_idx):
    return -np.log(p[y_idx] + 1e-12)


# ==============================================================================
# THE MODEL
# ==============================================================================
class HongakuUnveilingNetwork:
    r"""
    Forward pass (single vector x in R^d_in):

        z1  = W1 x + b1              # śamatha: quiet the surface of mind
        a1  = tanh(z1)              #   (practice transform)

        emp = We a1                 # THREE TRUTHS — emptiness view (kū)
        pro = Wp a1                 #             — provisional view (ke)
        s   = emp + pro
        zm  = Wm s + bm
        mid = tanh(zm)              #             — middle way (chū) reconciles both

        zv   = Wv x + bv            # obscuration produced by the *raw, defiled* input
        veil = sigmoid(zv)          #   kleshas: dust on the mirror, in [0,1]
        unv  = (1 - veil) * mid     # UNVEILING: remove dust -> reveal the middle-way mind

        lg0  = B unv                # alignment with FROZEN buddha-nature archetypes
        lg   = lg0 + alpha (C lg0)  # ICHINEN SANZEN: realms mutually include one another
        p    = softmax(lg)

    Loss = cross_entropy(p, y) + lam_veil * mean(veil)
           \_________________/    \___________________/
             recognition term      obscuration-removal term ("reduce the dust")

    FROZEN (given, never trained): B (buddha-nature archetypes), C (mutual
    inclusion). TRAINED (practice): W1,b1,We,Wp,Wm,bm,Wv,bv.
    """

    def __init__(self, d_in=16, hidden=24, n_realms=10, alpha=0.15, lam_veil=0.05):
        self.d_in = d_in
        self.H = hidden
        self.K = n_realms            # ten dharma-realms (jikkai)
        self.alpha = alpha
        self.lam_veil = lam_veil

        # --- Trained "practice" parameters -----------------------------------
        self.W1 = glorot((self.H, d_in));  self.b1 = np.zeros(self.H)
        self.We = glorot((self.H, self.H))  # emptiness view
        self.Wp = glorot((self.H, self.H))  # provisional view
        self.Wm = glorot((self.H, self.H)); self.bm = np.zeros(self.H)  # middle way
        self.Wv = glorot((self.H, d_in));  self.bv = np.zeros(self.H)   # veil

        # --- Frozen "already-there" structures --------------------------------
        # Buddha-nature archetypes: one awakened target per realm. Fixed forever.
        B = rng.standard_normal((self.K, self.H))
        self.B = B / np.linalg.norm(B, axis=1, keepdims=True)  # unit rows
        self.B.flags.writeable = False  # hard guarantee: recognition, not acquisition

        # Ichinen-sanzen mutual inclusion: symmetric, zero diagonal, row-balanced.
        C = rng.uniform(0.2, 1.0, size=(self.K, self.K))
        C = 0.5 * (C + C.T)
        np.fill_diagonal(C, 0.0)
        C = C / C.sum(axis=1, keepdims=True)  # each realm's inclusion of the other nine
        self.C = C
        self.M = np.eye(self.K) + self.alpha * self.C  # applied to logits
        self.C.flags.writeable = False

    # -- parameter plumbing (used by the optimizer and the gradient check) -----
    def params(self):
        return {"W1": self.W1, "b1": self.b1, "We": self.We, "Wp": self.Wp,
                "Wm": self.Wm, "bm": self.bm, "Wv": self.Wv, "bv": self.bv}

    # --------------------------------------------------------------------------
    # FORWARD
    # --------------------------------------------------------------------------
    def forward(self, x):
        z1 = self.W1 @ x + self.b1
        a1 = np.tanh(z1)                       # śamatha

        emp = self.We @ a1                     # empty view
        pro = self.Wp @ a1                     # provisional view
        s = emp + pro
        zm = self.Wm @ s + self.bm
        mid = np.tanh(zm)                      # middle way

        zv = self.Wv @ x + self.bv
        veil = sigmoid(zv)                     # obscuration
        unv = (1.0 - veil) * mid               # unveiling

        lg0 = self.B @ unv
        lg = self.M @ lg0                      # mutual inclusion
        p = softmax(lg)

        cache = dict(x=x, z1=z1, a1=a1, emp=emp, pro=pro, s=s, zm=zm, mid=mid,
                     zv=zv, veil=veil, unv=unv, lg0=lg0, lg=lg, p=p)
        return p, cache

    def loss(self, x, y_idx):
        p, cache = self.forward(x)
        ce = cross_entropy(p, y_idx)
        reg = self.lam_veil * np.mean(cache["veil"])
        return ce + reg, cache

    # --------------------------------------------------------------------------
    # BACKWARD (fully analytic; every step matched by the gradient check below)
    # --------------------------------------------------------------------------
    def backward(self, cache, y_idx):
        p = cache["p"]

        # d(CE)/d(logits) for softmax+cross-entropy
        dlg = p.copy()
        dlg[y_idx] -= 1.0

        dlg0 = self.M.T @ dlg                  # through mutual inclusion
        dunv = self.B.T @ dlg0                 # B frozen: no grad flows into B

        mid = cache["mid"]; veil = cache["veil"]
        dmid = dunv * (1.0 - veil)
        dveil = dunv * (-mid)
        # obscuration-removal regularizer: d(lam*mean(veil))/dveil = lam/H
        dveil += self.lam_veil / self.H

        dzv = dveil * veil * (1.0 - veil)      # sigmoid'
        x = cache["x"]
        dWv = np.outer(dzv, x); dbv = dzv

        dzm = dmid * (1.0 - mid**2)            # tanh'
        s = cache["s"]
        dWm = np.outer(dzm, s); dbm = dzm
        ds = self.Wm.T @ dzm

        demp = ds; dpro = ds                   # s = emp + pro
        a1 = cache["a1"]
        dWe = np.outer(demp, a1); dWp = np.outer(dpro, a1)
        da1 = self.We.T @ demp + self.Wp.T @ dpro

        dz1 = da1 * (1.0 - a1**2)              # tanh'
        dW1 = np.outer(dz1, x); db1 = dz1

        return {"W1": dW1, "b1": db1, "We": dWe, "Wp": dWp,
                "Wm": dWm, "bm": dbm, "Wv": dWv, "bv": dbv}

    def backward_batch(self, X, Y):
        grads = {k: np.zeros_like(v) for k, v in self.params().items()}
        total = 0.0
        for x, y in zip(X, Y):
            L, cache = self.loss(x, y)
            total += L
            g = self.backward(cache, y)
            for k in grads:
                grads[k] += g[k]
        n = len(X)
        for k in grads:
            grads[k] /= n
        return total / n, grads


# ==============================================================================
# GRADIENT CHECK  (mandatory)  — central finite differences vs analytic grads
# ==============================================================================
def gradient_check(model, X, Y, eps=1e-5):
    _, analytic = model.backward_batch(X, Y)

    def batch_loss():
        return np.mean([model.loss(x, y)[0] for x, y in zip(X, Y)])

    worst = 0.0
    report = {}
    for name, P in model.params().items():
        flat = P.ravel()
        idxs = rng.choice(flat.size, size=min(12, flat.size), replace=False)
        num = np.zeros(len(idxs))
        for i, idx in enumerate(idxs):
            orig = flat[idx]
            flat[idx] = orig + eps; lp = batch_loss()
            flat[idx] = orig - eps; lm = batch_loss()
            flat[idx] = orig
            num[i] = (lp - lm) / (2 * eps)
        ana = analytic[name].ravel()[idxs]
        denom = np.maximum(1e-8, np.abs(num) + np.abs(ana))
        rel = np.max(np.abs(num - ana) / denom)
        report[name] = rel
        worst = max(worst, rel)
    return worst, report


# ==============================================================================
# SYNTHETIC WORLD — a defiled-percepts task that embodies the doctrine
# ==============================================================================
def make_world(d_in=16, n_realms=10):
    """
    Each 'realm' has a clean latent signature in percept space. A real percept
    is that signature PLUS a shared 'klesha' (defilement) direction at random
    strength PLUS noise. The network must learn to see past the dust to the
    realm's awakened archetype. `defile` is the curriculum knob.
    """
    realm_sig = rng.standard_normal((n_realms, d_in))
    realm_sig /= np.linalg.norm(realm_sig, axis=1, keepdims=True)
    klesha = rng.standard_normal(d_in); klesha /= np.linalg.norm(klesha)

    def sample(n, defile=1.0, noise=0.25):
        ys = rng.integers(0, n_realms, size=n)
        X = []
        for y in ys:
            g = defile * rng.uniform(0.0, 1.6)           # defilement magnitude
            x = realm_sig[y] + g * klesha + noise * rng.standard_normal(d_in)
            X.append(x)
        return np.array(X), ys

    return sample


# ==============================================================================
# ADAM (from scratch — one honest optimizer, no framework)
# ==============================================================================
class Adam:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k]**2
            mhat = self.m[k] / (1 - self.b1**self.t)
            vhat = self.v[k] / (1 - self.b2**self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


# ==============================================================================
# METRICS
# ==============================================================================
def evaluate(model, X, Y):
    correct = 0
    veil_acc = 0.0
    per_class_hit = np.zeros(model.K)
    per_class_tot = np.zeros(model.K)
    for x, y in zip(X, Y):
        p, cache = model.forward(x)
        veil_acc += np.mean(cache["veil"])
        pred = int(np.argmax(p))
        per_class_tot[y] += 1
        if pred == y:
            correct += 1
            per_class_hit[y] += 1
    recall = np.divide(per_class_hit, np.maximum(1, per_class_tot))
    return correct / len(X), veil_acc / len(X), recall


# ==============================================================================
# MAIN — build, grad-check, train with curriculum, self-test, report
# ==============================================================================
def main():
    print("=" * 78)
    print("HONGAKU UNVEILING NETWORK — Saichō (Dengyō Daishi), chapter 0194")
    print("Recognition, not acquisition · No icchantika · Graded mountain practice")
    print("=" * 78)

    d_in, H, K = 16, 24, 10
    model = HongakuUnveilingNetwork(d_in=d_in, hidden=H, n_realms=K)
    world = make_world(d_in=d_in, n_realms=K)

    # Snapshot the frozen archetypes to prove they never move.
    B_before = model.B.copy()

    # ---- 1) GRADIENT CHECK ---------------------------------------------------
    Xg, Yg = world(24, defile=1.0)
    worst, report = gradient_check(model, Xg, Yg)
    print("\n[1] GRADIENT CHECK (central differences vs analytic)")
    for k, v in report.items():
        print(f"      {k:>3}: max rel err = {v:.2e}")
    ok_grad = worst < 1e-4
    print(f"      WORST max rel err = {worst:.2e}  ->  {'PASS' if ok_grad else 'FAIL'}")

    # ---- 2) BASELINE (before practice) --------------------------------------
    Xte, Yte = world(600, defile=1.0)
    acc0, veil0, _ = evaluate(model, Xte, Yte)
    print(f"\n[2] BEFORE PRACTICE:  accuracy = {acc0:.3f}   mean veil = {veil0:.3f}")

    # ---- 3) TRAIN with a graded curriculum (the twelve-year mountain arc) ----
    #     Defilement rises stage by stage: the novice trains on clearer percepts
    #     first, then on progressively more obscured ones.
    opt = Adam(model.params(), lr=4e-3)
    stages = [0.35, 0.6, 0.85, 1.1, 1.35]   # five graded stages
    epochs_per_stage = 60
    batch = 32
    print("\n[3] GRADED PRACTICE (curriculum):")
    for si, defile in enumerate(stages, 1):
        for ep in range(epochs_per_stage):
            Xb, Yb = world(batch, defile=defile)
            _, grads = model.backward_batch(Xb, Yb)
            opt.step(model.params(), grads)
        acc, veil, _ = evaluate(model, Xte, Yte)
        print(f"      stage {si} (defile={defile:.2f}):  "
              f"test acc = {acc:.3f}   mean veil = {veil:.3f}")

    # ---- 4) FINAL EVALUATION -------------------------------------------------
    accF, veilF, recall = evaluate(model, Xte, Yte)
    print(f"\n[4] AFTER PRACTICE:   accuracy = {accF:.3f}   mean veil = {veilF:.3f}")

    # ---- 5) SELF-TESTS (each asserts a piece of the doctrine) ----------------
    print("\n[5] SELF-TESTS")
    t_grad = ok_grad
    t_learn = accF > acc0 + 0.25 and accF > 0.6
    t_unveil = veilF < veil0 - 0.02          # dust removed over practice
    t_frozen = np.array_equal(B_before, model.B)  # recognition, not acquisition
    t_universal = np.min(recall) > 0.30      # NO ICCHANTIKA: every realm reachable
    print(f"      gradient check passes ........................ {t_grad}")
    print(f"      practice raises accuracy ..................... {t_learn}  "
          f"({acc0:.2f} -> {accF:.2f})")
    print(f"      obscuration (veil) falls with practice ....... {t_unveil}  "
          f"({veil0:.2f} -> {veilF:.2f})")
    print(f"      buddha-nature archetypes never trained ....... {t_frozen}")
    print(f"      no icchantika (min per-realm recall > 0.30) .. {t_universal}  "
          f"(min recall = {np.min(recall):.2f})")

    print("\n      per-realm recall (the ten dharma-realms):")
    print("      " + "  ".join(f"{r:.2f}" for r in recall))

    all_ok = all([t_grad, t_learn, t_unveil, t_frozen, t_universal])
    print("\n" + "=" * 78)
    print("ALL SELF-TESTS PASSED" if all_ok else "SOME SELF-TESTS FAILED")
    print("=" * 78)
    return all_ok


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)

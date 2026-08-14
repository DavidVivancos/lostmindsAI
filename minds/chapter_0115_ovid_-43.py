#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
115_Neuron.py  —  THE METAMORPHIC AUTOENCODER  ("the wax that stays itself")
================================================================================
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 6 Minds 101 - 120 Available on Amazon https://www.amazon.com/dp/B0HF7G6JJD
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 115: Ovid (-43 to -17 BCE)
================================================================================
An executable, from-scratch (pure-NumPy) neural architecture built around the
one cognitive idea that is Ovid's alone.

Ovid does not merely say "everything changes" (Heraclitus already owned flux).
In Book 15 of the *Metamorphoses* he gives a MECHANISM. The soul, he says, is
like wax:

    "utque novis facilis signatur cera figuris
     nec manet ut fuerat nec formam servat eandem,
     sed tamen ipsa eadem est"
     (as wax is easily stamped with new figures, and neither stays as it was
      nor keeps the same shape, yet is itself the very same wax)

So identity is NOT a body and NOT a fixed shape. It is a *conserved substrate*
that persists while an unlimited family of forms is stamped onto it. And the
transformations in the poem are almost never gradual drift — they are sudden,
discontinuous events that fire when an emotional/narrative TENSION crosses a
threshold: Daphne at the instant of capture becomes laurel; Niobe at the peak
of grief becomes weeping stone.

This file turns that reading into four coupled learning objectives, each of
which is a piece of Ovid rather than a piece of a generic transformer:

  (1) RECONSTRUCTION       — the world can be re-rendered from (identity + form).
  (2) IDENTITY RECOGNITION — the self is recognisable from the identity code
                             ALONE, whatever form it currently wears.
  (3) CONSERVATION         — "ipsa eadem est": the same soul seen in two
                             different forms must yield the *same* identity code.
                             (This is the wax simile, made differentiable.)
  (4) METAMORPHOSIS GATE   — a threshold unit that fires only when incoming
                             tension crosses a learned bound: the discontinuous
                             trigger of transformation.

The network splits its latent code into  z = [ z_id | z_form ].  z_id is the
wax; z_form is the figure stamped upon it. The conservation loss physically
pins z_id across form-changes while leaving z_form free to vary — the
architectural embodiment of "changed in shape, the same in soul".

Everything below is hand-derived: forward pass, exact analytic backprop, a
finite-difference gradient check (MANDATORY — it runs every time), an Adam
training loop, and self-tests that assert the Ovidian properties actually
emerge (identity generalises to UNSEEN forms; same-soul codes cluster; the
gate detects out-of-distribution "pressure").

Run:  python3 115_Neuron.py
================================================================================
"""

import numpy as np

RNG = np.random.default_rng(11543)   # 115 = Ovid ; 43 = 43 BCE, his birth year


# ==============================================================================
# 1. THE WORLD:  souls wearing forms
# ------------------------------------------------------------------------------
# We synthesise a little Ovidian cosmos. There are K "souls" (stable identities).
# Each soul k has a hidden identity vector s_k. The world only ever shows us a
# soul *wearing a form*: form m applies its own affine transformation to a fixed
# non-linear lift of the soul. The learner never sees s_k or m directly — it must
# discover, from raw observations x, an internal code that recovers the soul
# invariantly across every form it has worn.
# ==============================================================================

class MetamorphicWorld:
    def __init__(self, n_souls=8, n_forms=6, d_soul=5, d_lift=16, d_x=24,
                 n_basis=5, noise=0.05):
        self.K, self.M = n_souls, n_forms
        self.d_x = d_x
        # Hidden, immutable identity of each soul (the thing that must survive):
        self.S = RNG.normal(0, 1, size=(n_souls, d_soul))
        # A FIXED random non-linear lift shared by all forms (the "essence"):
        self.R = RNG.normal(0, 1, size=(d_soul, d_lift))

        # --- the GRAMMAR of transformation --------------------------------
        # Rather than give every form an unrelated renderer, all forms are
        # convex combinations of a shared BASIS of transformation directions.
        # This is the crucial modelling choice: a form never seen before is
        # still a new mixture of familiar changes, so a soul wearing it can be
        # recognised zero-shot. (A brand-new *grammar*, by contrast, is exile.)
        self.U = RNG.normal(0, 1, size=(n_basis, d_x, d_lift)) / np.sqrt(d_lift)
        self.Vb = RNG.normal(0, 0.4, size=(n_basis, d_x))

        def sample_coeffs():
            c = RNG.random(n_basis) ** 2.0            # skewed toward sparse mixes
            return c / c.sum()

        self.coeffs = np.array([sample_coeffs() for _ in range(n_forms)])
        self.A = np.einsum("mb,bxl->mxl", self.coeffs, self.U)   # (M,d_x,d_lift)
        self.b = self.coeffs @ self.Vb                          # (M, d_x)
        self.noise = noise

        # A "wild" renderer from a DIFFERENT grammar entirely (off-manifold):
        # this is Ovid's exile — the self displaced into Tomis, where nothing is
        # native. Used only to test the metamorphosis (tension) gate.
        self.A_wild = RNG.normal(0, 2.5, size=(d_x, d_lift)) / np.sqrt(d_lift)
        self.b_wild = RNG.normal(0, 1.5, size=(d_x,))

    def _essence(self, k):
        return np.tanh(self.S[k] @ self.R)              # (d_lift,)

    def render(self, k, m, tension=False):
        """Render soul k in form m. `tension` injects the extra 'pressure'
        (large perturbation) whose crossing the metamorphosis gate must learn
        to detect."""
        e = self._essence(k)
        x = self.A[m] @ e + self.b[m]
        x = x + RNG.normal(0, self.noise, size=self.d_x)
        if tension:
            x = x + RNG.normal(0, 0.9, size=self.d_x)   # emotional extremity
        return x

    def render_wild(self, k):
        """A soul rendered by a NEVER-SEEN renderer: genuine distribution shift
        (Ovid's exile — the self displaced into Tomis where nothing is native)."""
        e = self._essence(k)
        return self.A_wild @ e + self.b_wild + RNG.normal(0, self.noise, self.d_x)

    # ---- batch builders --------------------------------------------------
    def paired_batch(self, n_pairs, forms):
        """Return a batch built as PAIRS. Rows (2i, 2i+1) are the SAME soul in
        two DIFFERENT forms — this is what the conservation loss consumes.
        Also returns a tension label per row (1 if pressure was injected)."""
        X, soul, ten = [], [], []
        for _ in range(n_pairs):
            k = RNG.integers(self.K)
            m1, m2 = RNG.choice(forms, size=2, replace=False)
            t1 = RNG.random() < 0.30
            t2 = RNG.random() < 0.30
            X.append(self.render(k, m1, t1)); soul.append(k); ten.append(int(t1))
            X.append(self.render(k, m2, t2)); soul.append(k); ten.append(int(t2))
        return (np.array(X, float),
                np.array(soul, int),
                np.array(ten, float).reshape(-1, 1))


# ==============================================================================
# 2. THE NETWORK
# ------------------------------------------------------------------------------
#   encoder:  x --W1,b1--> tanh --W2,b2--> z = [z_id | z_form]
#   identity: z_id --Wc,bc--> softmax over K souls
#   decoder:  z --W3,b3--> tanh --W4,b4--> x_hat
#   gate:     h1 --Wt,bt--> sigmoid  (threshold-triggered metamorphosis unit)
#
# The gate reads the *encoding* h1 (perception), not the label, and learns the
# threshold (bt) and direction (Wt) at which "pressure" registers.
# ==============================================================================

class MetamorphicNet:
    def __init__(self, d_x, K, d_h=32, d_id=8, d_form=8):
        self.d_x, self.K = d_x, K
        self.d_id, self.d_form = d_id, d_form
        d_z = d_id + d_form
        s = lambda a, b: RNG.normal(0, np.sqrt(2.0 / a), size=(a, b))
        self.P = {
            "W1": s(d_x, d_h),  "b1": np.zeros(d_h),
            "W2": s(d_h, d_z),  "b2": np.zeros(d_z),
            "Wc": s(d_id, K),   "bc": np.zeros(K),
            "W3": s(d_z, d_h),  "b3": np.zeros(d_h),
            "W4": s(d_h, d_x),  "b4": np.zeros(d_x),
            # metamorphosis gate: a threshold on reconstruction STRESS. The form
            # transforms when it can no longer contain the essence (wg, bg are
            # the slope and the threshold of that trigger).
            "wg": np.array([4.0]), "bg": np.array([-1.0]),
        }
        # loss weights — each dial is one Ovidian commitment
        self.lam = dict(rec=1.0, idc=1.0, cons=2.0, gate=1.0)

    # ----- forward --------------------------------------------------------
    def forward(self, x):
        P = self.P
        h1 = np.tanh(x @ P["W1"] + P["b1"])
        z = h1 @ P["W2"] + P["b2"]
        z_id, z_form = z[:, :self.d_id], z[:, self.d_id:]
        logits = z_id @ P["Wc"] + P["bc"]
        h2 = np.tanh(z @ P["W3"] + P["b3"])
        x_hat = h2 @ P["W4"] + P["b4"]
        recon_err = x_hat - x
        stress = np.mean(recon_err ** 2, axis=1, keepdims=True)   # (N,1)
        gate_pre = stress * P["wg"] + P["bg"]
        gate = 1.0 / (1.0 + np.exp(-gate_pre))
        return dict(x=x, h1=h1, z=z, z_id=z_id, logits=logits, h2=h2,
                    x_hat=x_hat, recon_err=recon_err, stress=stress, gate=gate)

    # ----- loss (returns scalar + parts) ----------------------------------
    def loss(self, cache, soul, tension):
        N = cache["x"].shape[0]
        lam = self.lam

        # (1) reconstruction ------------------------------------------------
        diff = cache["x_hat"] - cache["x"]
        L_rec = np.mean(diff ** 2)

        # (2) identity cross-entropy ---------------------------------------
        lg = cache["logits"]
        lg = lg - lg.max(axis=1, keepdims=True)
        ex = np.exp(lg); p = ex / ex.sum(axis=1, keepdims=True)
        L_idc = -np.mean(np.log(p[np.arange(N), soul] + 1e-12))

        # (3) conservation across the pair (ipsa eadem est) -----------------
        a = cache["z_id"][0::2]; b = cache["z_id"][1::2]
        L_cons = np.mean((a - b) ** 2)

        # (4) metamorphosis gate (BCE with sigmoid) -------------------------
        g = np.clip(cache["gate"], 1e-7, 1 - 1e-7)
        L_gate = -np.mean(tension * np.log(g) + (1 - tension) * np.log(1 - g))

        total = (lam["rec"] * L_rec + lam["idc"] * L_idc
                 + lam["cons"] * L_cons + lam["gate"] * L_gate)
        return total, dict(rec=L_rec, idc=L_idc, cons=L_cons, gate=L_gate,
                           p=p, a=a, b=b)

    # ----- exact analytic gradients -----------------------------------------
    def backward(self, cache, parts, soul, tension):
        P, lam = self.P, self.lam
        N = cache["x"].shape[0]
        d_id = self.d_id
        g = {k: np.zeros_like(v) for k, v in P.items()}

        # --- (1) reconstruction path (+ gate path that flows through stress) ---
        dx_hat = lam["rec"] * (2.0 / (N * self.d_x)) * cache["recon_err"]

        # (4) metamorphosis gate: BCE on sigmoid(wg*stress + bg).
        # stress_i = mean_j (recon_err_ij)^2  ->  d stress/d x_hat = (2/d_x)*recon_err
        gate = np.clip(cache["gate"], 1e-7, 1 - 1e-7)
        dgp = lam["gate"] * (gate - tension) / N            # (N,1)
        g["wg"] += (dgp * cache["stress"]).sum(0)
        g["bg"] += dgp.sum(0)
        dx_hat += dgp * P["wg"] * (2.0 / self.d_x) * cache["recon_err"]

        g["W4"] += cache["h2"].T @ dx_hat
        g["b4"] += dx_hat.sum(0)
        dh2 = dx_hat @ P["W4"].T
        dpre2 = dh2 * (1 - cache["h2"] ** 2)               # tanh'
        g["W3"] += cache["z"].T @ dpre2
        g["b3"] += dpre2.sum(0)
        dz = dpre2 @ P["W3"].T                             # (N, d_z) from decoder

        # --- (2) identity cross-entropy path ---
        dlogits = parts["p"].copy()
        dlogits[np.arange(N), soul] -= 1.0
        dlogits *= lam["idc"] / N
        g["Wc"] += cache["z_id"].T @ dlogits
        g["bc"] += dlogits.sum(0)
        dz_id_ce = dlogits @ P["Wc"].T                     # (N, d_id)

        # --- (3) conservation path ---
        # L = mean over pairs of ||a-b||^2 ; d/da = 2(a-b)/(n_pairs*d_id)
        n_pairs = N // 2
        coef = lam["cons"] * (2.0 / (n_pairs * d_id))
        diff_ab = parts["a"] - parts["b"]
        dz_id_cons = np.zeros((N, d_id))
        dz_id_cons[0::2] = coef * diff_ab
        dz_id_cons[1::2] = -coef * diff_ab

        # fold identity-code grads back into the full z
        dz[:, :d_id] += dz_id_ce + dz_id_cons

        # encoder second layer
        g["W2"] += cache["h1"].T @ dz
        g["b2"] += dz.sum(0)
        dh1 = dz @ P["W2"].T

        # encoder first layer
        dpre1 = dh1 * (1 - cache["h1"] ** 2)
        g["W1"] += cache["x"].T @ dpre1
        g["b1"] += dpre1.sum(0)
        return g

    # convenience: full loss for a batch (used by grad check)
    def loss_only(self, x, soul, tension):
        c = self.forward(x)
        L, _ = self.loss(c, soul, tension)
        return L


# ==============================================================================
# 3. GRADIENT CHECK  (mandatory — finite differences vs. analytic backprop)
# ==============================================================================

def gradient_check(net, world, eps=1e-6, n_probe=6):
    X, soul, ten = world.paired_batch(6, forms=range(world.M))
    cache = net.forward(X)
    _, parts = net.loss(cache, soul, ten)
    analytic = net.backward(cache, parts, soul, ten)

    worst = 0.0
    print("  gradient check (finite-diff vs analytic):")
    for name in net.P:
        W = net.P[name]
        flat = W.ravel()
        idxs = RNG.choice(flat.size, size=min(n_probe, flat.size), replace=False)
        num = np.zeros(len(idxs)); ana = np.zeros(len(idxs))
        for j, i in enumerate(idxs):
            orig = flat[i]
            flat[i] = orig + eps; Lp = net.loss_only(X, soul, ten)
            flat[i] = orig - eps; Lm = net.loss_only(X, soul, ten)
            flat[i] = orig
            num[j] = (Lp - Lm) / (2 * eps)
            ana[j] = analytic[name].ravel()[i]
        rel = np.abs(num - ana) / np.maximum(1e-9, np.abs(num) + np.abs(ana))
        worst = max(worst, rel.max())
        print(f"    {name:>3}: max rel err {rel.max():.2e}")
    print(f"  >> worst relative error across all tensors: {worst:.2e}")
    assert worst < 1e-4, "GRADIENT CHECK FAILED"
    print("  >> GRADIENT CHECK PASSED\n")
    return worst


# ==============================================================================
# 4. ADAM + TRAINING LOOP
# ==============================================================================

class Adam:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads):
        self.t += 1
        for k in self.p:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            self.p[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


def train(net, world, train_forms, steps=1200, n_pairs=16, log_every=200):
    opt = Adam(net.P, lr=3e-3)
    first = last = None
    for t in range(steps):
        X, soul, ten = world.paired_batch(n_pairs, forms=train_forms)
        c = net.forward(X)
        L, parts = net.loss(c, soul, ten)
        g = net.backward(c, parts, soul, ten)
        opt.step(g)
        if t == 0:
            first = L
        if (t + 1) % log_every == 0 or t == 0:
            print(f"    step {t+1:>4}  total={L:.4f}  "
                  f"rec={parts['rec']:.4f} idc={parts['idc']:.4f} "
                  f"cons={parts['cons']:.4f} gate={parts['gate']:.4f}")
        last = L
    return first, last


# ==============================================================================
# 5. EVALUATION  —  do the Ovidian properties actually emerge?
# ==============================================================================

def identity_accuracy(net, world, forms, n=400):
    """Recognise the soul from z_id alone, across the given forms."""
    correct = 0
    for _ in range(n):
        k = RNG.integers(world.K); m = RNG.choice(forms)
        x = world.render(k, m)[None, :]
        pred = net.forward(x)["logits"].argmax()
        correct += (pred == k)
    return correct / n


def conservation_ratio(net, world, forms, n=300):
    """Mean distance between z_id of the SAME soul (diff forms) divided by mean
    distance between z_id of DIFFERENT souls. < 1 means the wax stays itself."""
    same, diff = [], []
    codes = {}
    for k in range(world.K):
        codes[k] = [net.forward(world.render(k, RNG.choice(forms))[None])["z_id"][0]
                    for _ in range(6)]
    for k in range(world.K):
        for i in range(len(codes[k])):
            for j in range(i + 1, len(codes[k])):
                same.append(np.linalg.norm(codes[k][i] - codes[k][j]))
    ks = list(codes)
    for _ in range(n):
        k1, k2 = RNG.choice(ks, 2, replace=False)
        diff.append(np.linalg.norm(codes[k1][0] - codes[k2][0]))
    return np.mean(same) / np.mean(diff), np.mean(same), np.mean(diff)


def gate_detection(net, world, forms, n=300):
    """The metamorphosis trigger: does the gate fire when emotional extremity
    (injected pressure) crosses threshold, and stay quiet when calm? Tested on
    the SAME kind of tension the gate was trained to register."""
    fire_pressed = fire_calm = 0
    for _ in range(n):
        k = RNG.integers(world.K); m = RNG.choice(forms)
        gp = net.forward(world.render(k, m, tension=True)[None])["gate"][0, 0]
        gc = net.forward(world.render(k, m, tension=False)[None])["gate"][0, 0]
        fire_pressed += (gp > 0.5); fire_calm += (gc > 0.5)
    return fire_pressed / n, fire_calm / n


def exile_recognition(net, world, n=300):
    """Ovid's exile as distribution shift: a soul rendered by a wholly NEW
    grammar (the wild renderer, 'Tomis'). Identity is expected to FRAY here —
    conservation guarantees survival only across the forms the mind has known.
    Reported as a diagnostic, not asserted."""
    correct = 0
    for _ in range(n):
        k = RNG.integers(world.K)
        pred = net.forward(world.render_wild(k)[None])["logits"].argmax()
        correct += (pred == k)
    return correct / n


# ==============================================================================
# 6. MAIN
# ==============================================================================

def main():
    print("=" * 72)
    print("  THE METAMORPHIC AUTOENCODER — Ovid, mind #115")
    print("  'nec manet ut fuerat ... sed tamen ipsa eadem est'")
    print("=" * 72)

    world = MetamorphicWorld(n_souls=8, n_forms=6, d_x=24)
    net = MetamorphicNet(d_x=world.d_x, K=world.K, d_h=32, d_id=8, d_form=8)

    # Hold out the LAST form entirely: the network must recognise a soul wearing
    # a shape it was never trained on — identity surviving an unseen metamorphosis.
    train_forms = list(range(world.M - 1))
    held_out = [world.M - 1]
    print(f"\n  souls={world.K}  forms={world.M}  "
          f"train_forms={train_forms}  held_out_form={held_out}\n")

    print("[1] Verifying gradients before training")
    gradient_check(net, world)

    print("[2] Training")
    acc_before = identity_accuracy(net, world, held_out)
    first, last = train(net, world, train_forms, steps=1200)

    print("\n[3] Evaluation")
    acc_train = identity_accuracy(net, world, train_forms)
    acc_held = identity_accuracy(net, world, held_out)
    ratio, s_same, s_diff = conservation_ratio(net, world, range(world.M))
    fire_pressed, fire_calm = gate_detection(net, world, range(world.M))
    acc_exile = exile_recognition(net, world)

    print(f"    loss: {first:.4f} -> {last:.4f}  "
          f"(reduced {100*(1-last/first):.1f}%)")
    print(f"    identity acc, held-out UNSEEN form : {acc_held:.3f}  "
          f"(was {acc_before:.3f} before training; chance={1/world.K:.3f})")
    print(f"    identity acc, trained forms        : {acc_train:.3f}")
    print(f"    conservation ratio (same/diff)     : {ratio:.3f}  "
          f"(same={s_same:.3f}, diff={s_diff:.3f})")
    print(f"    metamorphosis gate fires on pressure: {fire_pressed:.3f}")
    print(f"    metamorphosis gate fires on calm    : {fire_calm:.3f}")
    print(f"    identity acc under EXILE (wild form): {acc_exile:.3f}  "
          f"<- expected to fray (distribution shift)")

    print("\n[4] Self-tests")
    assert last < first * 0.6, "loss did not fall enough"
    assert acc_held > 0.60, "identity did not generalise to the unseen form"
    assert ratio < 0.6, "identity code not conserved across forms"
    assert fire_pressed - fire_calm > 0.4, "gate failed to separate pressure"
    print("    [OK] loss fell substantially")
    print("    [OK] identity survives an UNSEEN form (generalisation)")
    print("    [OK] identity code conserved across forms (ipsa eadem est)")
    print("    [OK] metamorphosis gate fires on pressure, not on calm")
    print("    [--] exile diagnostic recorded (recognition frays off-grammar)")
    print("\n" + "=" * 72)
    print("  ALL TESTS PASSED — the wax took new figures and stayed itself.")
    print("=" * 72)


if __name__ == "__main__":
    main()

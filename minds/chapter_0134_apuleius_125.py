#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 THE METAMORPHIC CURIOSITY NETWORK  (MCN)
 A from-scratch, pure-NumPy cognitive architecture after APULEIUS (c.124-170 CE)
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 7 Minds 121 - 140 Available on Amazon https://www.amazon.com/dp/B0HFN6GXMH
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 134: Apuleius (c.124-170 CE)
================================================================================   

WHY THIS ARCHITECTURE, AND NOT A TRANSFORMER
--------------------------------------------
Apuleius of Madauros wrote the only Roman novel to survive whole -- the
"Metamorphoses" (The Golden Ass) -- in which the curious young Lucius meddles
with magic and is turned into a donkey. As an ass he is beneath notice, and
precisely *because* of that he perceives a stratum of the world no free Roman
gentleman could ever see: the cruelty of masters, the speech of slaves and
bandits, the underside of the Empire. Only after long suffering is he restored
to human form -- enriched, initiated -- by the intervention of the goddess Isis
and a wreath of roses.

Apuleius' deepest cognitive claim is therefore NOT "accumulate more facts from a
fixed vantage." It is stranger and more specific:

    To *know* something you must METAMORPHOSE the observer.
    Each bodily form is a different perceptual apparatus with its own
    affordances and blind spots. Curiosity (curiositas) is the drive that
    forces the self to take on new forms; but transformation is dangerous and
    can trap you, so you also need a DAEMON -- an intermediary agency -- that
    chooses the right form and guarantees the way back.

The "daemon" is not decoration. In his treatise "De Deo Socratis" Apuleius
argues that between gods and humans stand *daimones*, intermediary spirits who
carry messages across the boundary. Socrates' inner voice was one such daemon.
In this network the daemon is the ROUTER that reads an omen/cue and decides
which metamorphosis to undergo -- and the RECOVERY operator that ferries the
transformed representation back to base ("return to human form").

So the architecture is a triad, mapped one-to-one onto Apuleius' ideas:

    curiositas  -> a set of learned MORPH operators, each a distinct
                   perceptual re-framing of the latent state (the "forms":
                   man, ass, bird, ...). Different features become visible in
                   different forms.
    daimonion   -> a soft ROUTER (softmax over forms) that reads the input's
                   cue and selects which form to adopt for this percept.
    Isis/return -> a RECOVERY operator with a CYCLE-CONSISTENCY loss that
                   forces every metamorphosis to be reversible: whatever form
                   you take, you must be able to come home to the base self.

THE TASK ("you must transform to perceive")
-------------------------------------------
We synthesise a world in which the label of each sample is *hidden* under an
orthogonal mixing and is only readable after adopting the correct form. Each
sample carries:
    * a mixed signal   : the discriminative pattern lives in ONE latent block
                         (the sample's "aspect"), then the whole thing is
                         scrambled by a fixed orthogonal matrix so no linear
                         readout of the raw input can recover it;
    * a noisy omen/cue : a corrupted one-hot pointing (usually) at the aspect.
The daemon must read the omen and route to the right form; the morph for that
form must learn to *un-scramble* and isolate the correct block; only then is the
label legible. A plain linear classifier on the raw input hovers near chance,
because the input-to-label map is conditional (it depends on the aspect), not
linear. The MCN, by transforming the observer, solves it.

WHAT THE FILE DOES WHEN RUN
---------------------------
    1. Builds the model in pure NumPy (no autograd, no ML frameworks).
    2. Runs a FINITE-DIFFERENCE GRADIENT CHECK over every parameter tensor and
       asserts the analytic gradients match (mandatory correctness gate).
    3. Trains with Adam on the "hidden-under-transformation" task and shows the
       loss falling and accuracy rising far above a linear baseline.
    4. Prints diagnostics: how decisively the daemon routes, and how faithfully
       each form can return to the base self (cycle fidelity).
    5. Runs self-tests (shape, determinism, reversibility, generalisation).


Dependencies: numpy only.
================================================================================
"""

import numpy as np

# ------------------------------------------------------------------------------
# 0. Small numerical helpers
# ------------------------------------------------------------------------------

def softmax(z, axis=-1):
    """Numerically stable softmax."""
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def one_hot(idx, n):
    out = np.zeros((idx.shape[0], n))
    out[np.arange(idx.shape[0]), idx] = 1.0
    return out


# ------------------------------------------------------------------------------
# 1. The synthetic world: perception hidden behind metamorphosis
# ------------------------------------------------------------------------------

class MetamorphicWorld:
    """
    Generates samples whose label is only legible after adopting the correct
    'form'. This operationalises Apuleius' thesis that some knowledge is
    reachable only by transforming the perceiver.

    Latent layout (dimension H): split into K equal blocks. EVERY block carries
    a label-shaped pattern sign_b * dirs[b], but only the sample's true aspect
    block a carries the *real* label sign; the other (distractor) blocks carry
    INDEPENDENT RANDOM signs. So a fixed reader that pools across blocks is
    confused by the distractors -- the true label is legible only if you read
    block a alone, i.e. only after adopting the correct form. The latent is then
    mixed by a FIXED random orthogonal matrix W_mix so the pattern is smeared
    across all input dimensions. A noisy 'omen' (corrupted one-hot of a) is
    appended as a cue for the daemon/router.

    Input dimension D = H + K.
    """

    def __init__(self, H=12, K=4, signal=3.0, noise=0.30, omen_noise=0.45, seed=0):
        assert H % K == 0, "H must divide evenly into K blocks"
        self.H, self.K = H, K
        self.block = H // K
        self.signal, self.noise, self.omen_noise = signal, noise, omen_noise
        rng = np.random.RandomState(seed)
        # Fixed orthogonal mixing matrix (the 'scrambling' of ordinary vantage).
        A = rng.randn(H, H)
        Q, _ = np.linalg.qr(A)
        self.W_mix = Q
        # A fixed unit direction inside each block that encodes the label.
        self.dirs = rng.randn(K, self.block)
        self.dirs /= np.linalg.norm(self.dirs, axis=1, keepdims=True)
        self.rng = rng
        self.D = H + K

    def sample(self, n):
        H, K, b = self.H, self.K, self.block
        aspect = self.rng.randint(0, K, size=n)          # which form reveals truth
        label = self.rng.randint(0, 2, size=n)           # binary label
        sign = np.where(label == 1, 1.0, -1.0)

        s = self.noise * self.rng.randn(n, H)            # weak noise everywhere
        # Every block gets a label-shaped pattern, but only block `a` gets the
        # TRUE sign; the other blocks get independent random signs (distractors).
        for i in range(n):
            a = aspect[i]
            for blk in range(K):
                blk_sign = sign[i] if blk == a else (1.0 if self.rng.rand() < 0.5 else -1.0)
                s[i, blk * b:(blk + 1) * b] += self.signal * blk_sign * self.dirs[blk]

        x_sig = s @ self.W_mix.T                         # scramble across all dims

        omen = one_hot(aspect, K) + self.omen_noise * self.rng.randn(n, K)
        x = np.concatenate([x_sig, omen], axis=1)        # D = H + K
        return x, label, aspect


# ------------------------------------------------------------------------------
# 2. The Metamorphic Curiosity Network
# ------------------------------------------------------------------------------

class MCN:
    """
    Pure-NumPy model. Parameters live in a dict of named arrays so the gradient
    check can iterate over them generically.

    Forward pipeline (batch of N):
        z      = tanh(x W_enc^T + b_enc)                         # base "human" self
        alpha  = softmax(z W_g^T + b_g)                          # DAEMON routes forms
        U[k]   = tanh(z M_k^T + c_k)          for each form k    # CURIOSITAS: morphs
        p      = sum_k alpha_k * U[k]                            # perceived (adopted form)
        logits = p W_out^T + b_out                              # readout of the label
        R[k]   = tanh(U[k] R_k^T + d_k)       for each form k    # ISIS: recover base
        rec    = sum_k alpha_k * R[k]
    Loss = cross_entropy(logits, y) + lam_cyc * mean||rec - z||^2
    """

    def __init__(self, D, H, K, C=2, lam_cyc=0.5, lam_route=0.5, seed=1):
        rng = np.random.RandomState(seed)
        s = lambda a, b: rng.randn(a, b) * np.sqrt(1.0 / b)
        self.D, self.H, self.K, self.C = D, H, K, C
        self.lam_cyc = lam_cyc
        self.lam_route = lam_route     # the daemon is trained to heed the omen
        self.p = {
            "W_enc": s(H, D), "b_enc": np.zeros(H),
            "M":    np.stack([s(H, H) for _ in range(K)]),   # (K,H,H) morph maps
            "c":    np.zeros((K, H)),
            "W_g":  s(K, H), "b_g": np.zeros(K),             # daemon router
            "W_out": s(C, H), "b_out": np.zeros(C),          # label readout
            "R":    np.stack([s(H, H) for _ in range(K)]),   # (K,H,H) recovery maps
            "d":    np.zeros((K, H)),
        }

    # ----- forward -----------------------------------------------------------
    def forward(self, x, y=None, aspect=None):
        P = self.p
        z_pre = x @ P["W_enc"].T + P["b_enc"]
        z = np.tanh(z_pre)

        g = z @ P["W_g"].T + P["b_g"]
        alpha = softmax(g, axis=1)                                   # (N,K)

        U_pre = np.einsum("nj,kij->nki", z, P["M"]) + P["c"]         # (N,K,H)
        U = np.tanh(U_pre)

        p = np.einsum("nk,nki->ni", alpha, U)                        # (N,H)
        logits = p @ P["W_out"].T + P["b_out"]                       # (N,C)
        probs = softmax(logits, axis=1)

        R_pre = np.einsum("nkj,kij->nki", U, P["R"]) + P["d"]        # (N,K,H)
        Rr = np.tanh(R_pre)
        rec = np.einsum("nk,nki->ni", alpha, Rr)                     # (N,H)

        cache = dict(x=x, z_pre=z_pre, z=z, g=g, alpha=alpha,
                     U_pre=U_pre, U=U, p=p, logits=logits, probs=probs,
                     R_pre=R_pre, Rr=Rr, rec=rec, y=y, aspect=aspect)

        loss = None
        if y is not None:
            N = x.shape[0]
            ce = -np.mean(np.log(probs[np.arange(N), y] + 1e-12))
            diff = rec - z
            cyc = np.mean(np.sum(diff * diff, axis=1))
            loss = ce + self.lam_cyc * cyc
            route = 0.0
            if aspect is not None:
                route = -np.mean(np.log(alpha[np.arange(N), aspect] + 1e-12))
                loss = loss + self.lam_route * route
            cache["ce"], cache["cyc"], cache["route"] = ce, cyc, route
        return loss, cache

    # ----- backward (analytic) ----------------------------------------------
    def backward(self, cache):
        P = self.p
        x, z, alpha = cache["x"], cache["z"], cache["alpha"]
        U, Rr, rec, y = cache["U"], cache["Rr"], cache["rec"], cache["y"]
        N = x.shape[0]

        # ---- classifier head ----
        dlogits = cache["probs"].copy()
        dlogits[np.arange(N), y] -= 1.0
        dlogits /= N                                                 # (N,C)
        p = cache["p"]
        gW_out = dlogits.T @ p
        gb_out = dlogits.sum(0)
        dp = dlogits @ P["W_out"]                                    # (N,H)

        # ---- cycle-consistency loss ----
        diff = rec - z
        drec = (2.0 * self.lam_cyc / N) * diff                      # (N,H)
        dz_cyc = -(2.0 * self.lam_cyc / N) * diff                   # direct term (-z)

        # ---- p = sum_k alpha_k U_k ----
        dalpha = np.einsum("ni,nki->nk", dp, U)                     # from perceived
        dU = np.einsum("ni,nk->nki", dp, alpha)                     # from perceived

        # ---- rec = sum_k alpha_k R_k ----
        dalpha += np.einsum("ni,nki->nk", drec, Rr)
        dRr = np.einsum("ni,nk->nki", drec, alpha)

        # ---- recovery: Rr = tanh(U R^T + d) ----
        dR_pre = dRr * (1.0 - Rr ** 2)
        gR = np.einsum("nki,nkj->kij", dR_pre, U)
        gd = dR_pre.sum(0)
        dU += np.einsum("nki,kij->nkj", dR_pre, P["R"])

        # ---- morphs: U = tanh(z M^T + c) ----
        dU_pre = dU * (1.0 - U ** 2)
        gM = np.einsum("nki,nj->kij", dU_pre, z)
        gc = dU_pre.sum(0)
        dz_morph = np.einsum("nki,kij->nj", dU_pre, P["M"])

        # ---- daemon router: alpha = softmax(g) ----
        dg = alpha * (dalpha - np.sum(dalpha * alpha, axis=1, keepdims=True))
        # auxiliary omen-reading loss: CE(alpha, aspect) -> (alpha - onehot)/N
        aspect = cache.get("aspect", None)
        if aspect is not None:
            dg_route = alpha.copy()
            dg_route[np.arange(N), aspect] -= 1.0
            dg += (self.lam_route / N) * dg_route
        gW_g = dg.T @ z
        gb_g = dg.sum(0)
        dz_router = dg @ P["W_g"]

        # ---- accumulate into z, then encoder ----
        dz = dz_morph + dz_router + dz_cyc
        dz_pre = dz * (1.0 - z ** 2)
        gW_enc = dz_pre.T @ x
        gb_enc = dz_pre.sum(0)

        return {"W_enc": gW_enc, "b_enc": gb_enc, "M": gM, "c": gc,
                "W_g": gW_g, "b_g": gb_g, "W_out": gW_out, "b_out": gb_out,
                "R": gR, "d": gd}

    # ----- convenience -------------------------------------------------------
    def predict(self, x):
        _, cache = self.forward(x)
        return np.argmax(cache["probs"], axis=1), cache


# ------------------------------------------------------------------------------
# 3. Finite-difference gradient check (mandatory correctness gate)
# ------------------------------------------------------------------------------

def gradient_check(model, x, y, aspect, n_probe=6, eps=1e-5, tol=1e-5):
    """Verify analytic gradients against central finite differences.
    Probes a random subset of coordinates in every parameter tensor. The full
    loss (label CE + cycle-consistency + omen-routing) is exercised."""
    _, cache = model.forward(x, y, aspect)
    grads = model.backward(cache)
    rng = np.random.RandomState(123)
    worst = 0.0
    report = []
    for name, W in model.p.items():
        flat = W.ravel()
        gflat = grads[name].ravel()
        m = min(n_probe, flat.size)
        idxs = rng.choice(flat.size, size=m, replace=False)
        errs = []
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps
            lp, _ = model.forward(x, y, aspect)
            flat[i] = orig - eps
            lm, _ = model.forward(x, y, aspect)
            flat[i] = orig
            num = (lp - lm) / (2 * eps)
            ana = gflat[i]
            denom = max(1e-12, abs(num) + abs(ana))
            errs.append(abs(num - ana) / denom)
        e = max(errs)
        worst = max(worst, e)
        report.append((name, e))
    return worst, report


# ------------------------------------------------------------------------------
# 4. Adam optimiser
# ------------------------------------------------------------------------------

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
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            self.p[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


# ------------------------------------------------------------------------------
# 5. Baselines & metrics
# ------------------------------------------------------------------------------

def linear_baseline_accuracy(world, n_train=4000, n_test=2000):
    """Least-squares linear classifier on raw input: the control that CANNOT
    transform its vantage. Expected to sit near chance because the input->label
    map is conditional on the (scrambled) aspect."""
    Xtr, ytr, _ = world.sample(n_train)
    Xte, yte, _ = world.sample(n_test)
    Xtr1 = np.concatenate([Xtr, np.ones((n_train, 1))], axis=1)
    Xte1 = np.concatenate([Xte, np.ones((n_test, 1))], axis=1)
    t = np.where(ytr == 1, 1.0, -1.0)
    w, *_ = np.linalg.lstsq(Xtr1, t, rcond=None)
    pred = (Xte1 @ w > 0).astype(int)
    return np.mean(pred == yte)


def routing_purity(cache, aspect):
    """How often the daemon's top form matches the true aspect."""
    top = np.argmax(cache["alpha"], axis=1)
    return np.mean(top == aspect)


def cycle_fidelity(cache):
    """1 - normalised reconstruction error of the base self (return quality)."""
    z, rec = cache["z"], cache["rec"]
    err = np.mean(np.sum((rec - z) ** 2, axis=1))
    scale = np.mean(np.sum(z ** 2, axis=1)) + 1e-9
    return 1.0 - err / scale


# ------------------------------------------------------------------------------
# 6. Training
# ------------------------------------------------------------------------------

def train(seed=0, epochs=220, batch=256, verbose=True):
    world = MetamorphicWorld(seed=seed)
    model = MCN(D=world.D, H=world.H, K=world.K, C=2, lam_cyc=0.3, seed=seed + 7)
    opt = Adam(model.p, lr=5e-3)

    Xval, yval, aval = world.sample(3000)

    if verbose:
        print("  epoch |   loss     ce     cyc    rte | train_acc  val_acc  route  cycle")
        print("  " + "-" * 72)

    for ep in range(1, epochs + 1):
        Xb, yb, ab = world.sample(batch)
        loss, cache = model.forward(Xb, yb, ab)
        grads = model.backward(cache)
        opt.step(grads)

        if verbose and (ep % 20 == 0 or ep == 1):
            tr_pred = np.argmax(cache["probs"], axis=1)
            tr_acc = np.mean(tr_pred == yb)
            _, vc = model.forward(Xval, yval)
            v_acc = np.mean(np.argmax(vc["probs"], axis=1) == yval)
            print("  {:5d} | {:6.3f} {:6.3f} {:6.3f} {:5.2f} |   {:5.3f}    {:5.3f}   {:5.3f}  {:5.3f}"
                  .format(ep, loss, cache["ce"], cache["cyc"], cache["route"], tr_acc, v_acc,
                          routing_purity(vc, aval), cycle_fidelity(vc)))
    return world, model


# ------------------------------------------------------------------------------
# 7. Self-tests
# ------------------------------------------------------------------------------

def self_tests(world, model):
    print("\n  Running self-tests ...")
    X, y, aspect = world.sample(500)

    # (a) shapes & determinism
    _, c1 = model.forward(X, y)
    _, c2 = model.forward(X, y)
    assert c1["logits"].shape == (500, 2)
    assert np.allclose(c1["logits"], c2["logits"]), "forward must be deterministic"
    print("   [ok] shapes and determinism")

    # (b) routing is decisive and mostly correct
    rp = routing_purity(c1, aspect)
    assert rp > 0.8, f"router should recover the aspect (got {rp:.2f})"
    print(f"   [ok] daemon routes to the true form {rp*100:.1f}% of the time")

    # (c) reversibility: forms can return to the base self
    cf = cycle_fidelity(c1)
    assert cf > 0.6, f"cycle fidelity too low ({cf:.2f})"
    print(f"   [ok] metamorphoses are reversible (cycle fidelity {cf:.3f})")

    # (d) generalisation to fresh draws beats the transform-blind baseline
    Xte, yte, _ = world.sample(3000)
    acc = np.mean(model.predict(Xte)[0] == yte)
    assert acc > 0.85, f"held-out accuracy too low ({acc:.2f})"
    print(f"   [ok] generalises to unseen samples (accuracy {acc*100:.1f}%)")

    # (e) ablation: freeze the daemon to uniform routing -> perception collapses
    saved = model.p["W_g"].copy(), model.p["b_g"].copy()
    model.p["W_g"] *= 0.0
    model.p["b_g"] *= 0.0
    acc_blind = np.mean(model.predict(Xte)[0] == yte)
    model.p["W_g"], model.p["b_g"] = saved
    assert acc_blind < acc - 0.1, "removing the daemon should hurt a lot"
    print(f"   [ok] without the daemon (uniform forms) accuracy falls to "
          f"{acc_blind*100:.1f}% -- transformation must be *chosen*")
    print("  All self-tests passed.")


# ------------------------------------------------------------------------------
# 8. Main
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(0)
    print("=" * 74)
    print(" METAMORPHIC CURIOSITY NETWORK  --  after Apuleius (The Golden Ass)")
    print("=" * 74)

    # --- 1. correctness gate: finite-difference gradient check ---
    print("\n[1] Finite-difference gradient check")
    gw = MetamorphicWorld(seed=42)
    gx, gy, ga = gw.sample(24)
    gmodel = MCN(D=gw.D, H=gw.H, K=gw.K, C=2, lam_cyc=0.5, lam_route=0.5, seed=3)
    worst, report = gradient_check(gmodel, gx, gy, ga)
    for name, e in report:
        print(f"     {name:6s}  max rel err = {e:.2e}")
    print(f"     WORST over all tensors = {worst:.2e}")
    assert worst < 1e-4, "gradient check FAILED"
    print("     PASSED (analytic gradients match finite differences).")

    # --- 2. baseline that cannot transform its vantage ---
    print("\n[2] Transform-blind linear baseline")
    base = linear_baseline_accuracy(MetamorphicWorld(seed=0))
    print(f"     linear classifier on raw input: {base*100:.1f}% "
          f"(chance = 50%) -- a fixed vantage cannot read the hidden label.")

    # --- 3. train the metamorphic network ---
    print("\n[3] Training the Metamorphic Curiosity Network")
    world, model = train(seed=0, epochs=260)

    # --- 4. self-tests ---
    self_tests(world, model)

    print("\n" + "=" * 74)
    print(" DONE. The mind that changes its form to see -- and finds its way home.")
    print("=" * 74)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 PARALLAX FRAME-COVARIANCE NETWORK  (PFCN)
 A from-scratch, trainable cognitive architecture after Aristarchus of Samos
 (c. 310 - c. 230 BCE), the first astronomer to put the Sun at the centre.
 # Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 5 Minds 81 - 100 Available on Amazon https://www.amazon.com/dp/B0H7LP5LP2
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 0084 · Aristarchus of Samos
================================================================================

WHY THIS ARCHITECTURE, AND NOT A TRANSFORMER
--------------------------------------------
The default modern recipe (attention over stored keys, mixture-of-experts,
next-token prediction) encodes "intelligence = compress co-occurrence statistics
of a stream." That is *not* Aristarchus's idea of a mind. His one cognitive move,
the move nobody around him would make, was this:

    The way the heavens APPEAR (the Sun wheeling around a fixed Earth) is an
    ARTIFACT of the observer's own position and motion. The way they truly ARE
    (Earth circling the Sun) is the description that does not depend on where you
    happen to stand. To find truth, hypothesise your OWN motion and transform
    your egocentric appearances into a viewpoint-invariant world model.

He pushed this to its sharpest edge with the stars. If the Earth really swings
around the Sun, nearby stars should shift against far ones over the year
(parallax). No such shift was seen. Everyone else read that as proof the Earth
stands still. Aristarchus read the SAME null signal the opposite way: the shift
is there but too small to see, therefore the stars are unimaginably far. He
inferred *vast latent scale from the ABSENCE of an expected signal.*

So this network is built from exactly two Aristarchan operations:

  (1) FRAME TRANSFORMATION (egocentric -> allocentric).
      An encoder takes appearances recorded from two different observer
      positions ("baselines") plus the observer's own state, and maps them to a
      single viewpoint-INVARIANT latent: the body's place in the world frame.
      The same body seen from anywhere must map to the same latent. That
      invariance IS the heliocentric world-model. A decoder then "saves the
      phenomena": from the invariant latent it reproduces how the body would
      LOOK from any new vantage. (Greek astronomy literally called this goal
      sozein ta phainomena, "to save the appearances.")

  (2) PARALLAX -> SCALE INFERENCE.
      A second head learns the Aristarchan reading of the null signal: it maps
      the magnitude of apparent shift between viewpoints to log-distance, so that
      a vanishing shift implies an enormous distance. After training we feed it a
      "star" (near-zero shift) and watch it answer "very, very far" - the exact
      inference that, in 230 BCE, made the cosmos explode in size.

Nothing here is a transformer. The inductive bias is *covariance under self-
motion*, not attention. That is the point.

ENGINEERING CONTRACT (kept for every file in this corpus)
---------------------------------------------------------
  * Pure NumPy, manual forward and backward passes (no autograd).
  * A finite-difference gradient check that MUST pass (mandatory).
  * A real training loop on a self-generated synthetic "cosmos".
  * Self-tests that demonstrate the *mind*, not just falling loss.
  * The file is executed before shipping; verified output is pasted into the
    chapter.

Run:  python3 chapter_0084_aristarchus_of_samos_-310.py
================================================================================
"""

import numpy as np

RNG = np.random.default_rng(310)  # seeded with his birth year, for reproducibility


# ============================================================================
# 0.  THE SYNTHETIC COSMOS  (the world the mind must model)
# ----------------------------------------------------------------------------
# World frame: the Sun sits at the origin (0,0) -- heliocentric by construction.
# The observer (the Earth) rides a circular orbit of radius R_OBS = 1 around the
# Sun, parameterised by an orbital phase theta in [0, 2*pi).
#
# A celestial body lives at true world position  B = r * (cos(phi), sin(phi)),
# where phi is its true direction from the Sun and r is its distance.
#  - "planets"/near bodies:  r in [2, 8]   -> large parallax
#  - "stars"/far bodies:     r in [40, 400] -> almost no parallax
#
# The APPEARANCE we record is the body's bearing in the sky as seen from the
# moving Earth: the world-frame angle of the vector (Body - Earth). For a far
# star this barely changes as the Earth moves; for a near body it swings. The
# part of that swing that depends on r is parallax -- the whole game.
# ============================================================================

R_OBS = 1.0  # Earth's orbital radius, in arbitrary "cosmic units"


def observer_position(theta):
    """Earth's world position at orbital phase theta."""
    return np.stack([R_OBS * np.cos(theta), R_OBS * np.sin(theta)], axis=-1)


def apparent_bearing(phi, r, theta):
    """
    Bearing (sky angle) of a body, as seen from the Earth at phase theta.
    Returns the angle, in radians, of the vector (Body - Earth) in the world
    frame. This is the raw *egocentric appearance* the mind has to work with.
    """
    bx = r * np.cos(phi)
    by = r * np.sin(phi)
    ox = R_OBS * np.cos(theta)
    oy = R_OBS * np.sin(theta)
    return np.arctan2(by - oy, bx - ox)


def wrap(a):
    """Wrap an angle to (-pi, pi]; used so parallax shifts never jump by 2*pi."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def make_batch(n, star_fraction=0.5):
    """
    Build a batch of n bodies. For each body we record appearances from FOUR
    observer phases: (t1,t2) form baseline A, (t3,t4) form baseline B. The
    encoder must produce the SAME world latent from baseline A and baseline B
    -- that agreement is the invariance (= world-model) signal.

    Returns a dict of arrays. Angles are delivered as (cos, sin) pairs so the
    network never has to cope with the -pi/pi wraparound discontinuity.
    """
    # --- true latent world coordinates of each body ---
    phi = RNG.uniform(-np.pi, np.pi, size=n)
    is_star = RNG.random(n) < star_fraction
    r = np.where(is_star,
                 RNG.uniform(40.0, 400.0, size=n),   # far stars
                 RNG.uniform(2.0, 8.0, size=n))       # near bodies
    logr = np.log(r)

    # --- four observer phases per body ---
    t1 = RNG.uniform(-np.pi, np.pi, size=n)
    t2 = t1 + RNG.uniform(0.4, np.pi, size=n)         # baseline A partner
    t3 = RNG.uniform(-np.pi, np.pi, size=n)
    t4 = t3 + RNG.uniform(0.4, np.pi, size=n)         # baseline B partner
    tq = RNG.uniform(-np.pi, np.pi, size=n)           # a held-out query vantage

    def feat(theta):
        a = apparent_bearing(phi, r, theta)
        return np.stack([np.cos(a), np.sin(a), np.cos(theta), np.sin(theta)], axis=-1)

    # encoder input for baseline A = appearances from t1 and t2 (8 dims)
    encA = np.concatenate([feat(t1), feat(t2)], axis=-1)
    encB = np.concatenate([feat(t3), feat(t4)], axis=-1)

    # decoder query: predict appearance (cos,sin of bearing) from vantage tq
    aq = apparent_bearing(phi, r, tq)
    dec_q = np.stack([np.cos(tq), np.sin(tq)], axis=-1)          # query vantage
    dec_target = np.stack([np.cos(aq), np.sin(aq)], axis=-1)      # what it should look like

    # parallax-head input: how much the bearing shifts over two fixed baselines
    # (pi/2 and pi). Small shift => far. This is fed to predict log-distance.
    t0 = RNG.uniform(-np.pi, np.pi, size=n)
    s_quarter = np.abs(wrap(apparent_bearing(phi, r, t0) -
                            apparent_bearing(phi, r, t0 + np.pi / 2)))
    s_half = np.abs(wrap(apparent_bearing(phi, r, t0) -
                         apparent_bearing(phi, r, t0 + np.pi)))
    par_in = np.stack([s_quarter, s_half], axis=-1)

    return {
        "encA": encA, "encB": encB,
        "dec_q": dec_q, "dec_target": dec_target,
        "par_in": par_in, "logr": logr[:, None],
        "is_star": is_star,
    }


# ============================================================================
# 1.  PRIMITIVES  (a tiny hand-built autodiff-free layer library)
# ----------------------------------------------------------------------------
# Each layer exposes forward(x)->(out, cache) and backward(cache, dout)->(dx,
# grads). Grads are accumulated by the caller. Keeping forward pure (no hidden
# randomness) is what lets the finite-difference gradient check be exact.
# ============================================================================

def init_linear(n_in, n_out):
    """Xavier/Glorot init keeps activations well-scaled through tanh layers."""
    scale = np.sqrt(2.0 / (n_in + n_out))
    return {"W": RNG.normal(0, scale, (n_in, n_out)), "b": np.zeros((n_out,))}


def linear_forward(p, x):
    out = x @ p["W"] + p["b"]
    return out, (x,)


def linear_backward(p, cache, dout):
    (x,) = cache
    dW = x.T @ dout
    db = dout.sum(axis=0)
    dx = dout @ p["W"].T
    return dx, {"W": dW, "b": db}


def tanh_forward(x):
    y = np.tanh(x)
    return y, (y,)


def tanh_backward(cache, dout):
    (y,) = cache
    return dout * (1.0 - y * y)


class MLP:
    """A small multilayer perceptron: Linear -> tanh -> ... -> Linear (no final
    activation, so outputs are unbounded regression targets)."""

    def __init__(self, sizes, name):
        self.name = name
        self.layers = [init_linear(sizes[i], sizes[i + 1])
                       for i in range(len(sizes) - 1)]

    def params(self):
        """Flat dict of named parameters, e.g. 'enc.L0.W'."""
        out = {}
        for i, lyr in enumerate(self.layers):
            out[f"{self.name}.L{i}.W"] = lyr["W"]
            out[f"{self.name}.L{i}.b"] = lyr["b"]
        return out

    def forward(self, x):
        cache = []
        h = x
        for i, lyr in enumerate(self.layers):
            h, lc = linear_forward(lyr, h)
            cache.append(("lin", lc))
            if i < len(self.layers) - 1:          # tanh on every layer but the last
                h, tc = tanh_forward(h)
                cache.append(("tanh", tc))
        return h, cache

    def backward(self, cache, dout):
        """Returns (dx, grads) where grads is a flat dict matching params()."""
        grads = {}
        d = dout
        lin_idx = len(self.layers) - 1
        for kind, c in reversed(cache):
            if kind == "tanh":
                d = tanh_backward(c, d)
            else:  # linear
                d, g = linear_backward(self.layers[lin_idx], c, d)
                grads[f"{self.name}.L{lin_idx}.W"] = g["W"]
                grads[f"{self.name}.L{lin_idx}.b"] = g["b"]
                lin_idx -= 1
        return d, grads


# ============================================================================
# 2.  THE PARALLAX FRAME-COVARIANCE NETWORK
# ----------------------------------------------------------------------------
#   encoder : R^8 -> R^Z   (two baselines of appearance  ->  invariant latent)
#   decoder : R^(Z+2) -> R^2  (invariant latent + new vantage -> appearance)
#   parallax: R^2 -> R^1   (apparent shift over fixed baselines -> log-distance)
#
# Z = 3: the latent carries (cos phi, sin phi)-like world direction PLUS a
# distance coordinate, because reproducing a NEAR body's appearance from a new
# vantage genuinely requires knowing how far it is (that is what parallax is).
# ============================================================================

Z = 3


class PFCN:
    def __init__(self):
        self.enc = MLP([8, 24, 16, Z], "enc")
        self.dec = MLP([Z + 2, 24, 16, 2], "dec")
        self.par = MLP([2, 16, 8, 1], "par")
        self.lambda_inv = 1.0   # weight on the world-model invariance term
        self.lambda_par = 1.0   # weight on the parallax->scale term

    # -- parameter plumbing ---------------------------------------------------
    def params(self):
        p = {}
        p.update(self.enc.params())
        p.update(self.dec.params())
        p.update(self.par.params())
        return p

    def set_params(self, flat):
        """Write a flat dict back into the live layer arrays (in place)."""
        for mlp in (self.enc, self.dec, self.par):
            for i, lyr in enumerate(mlp.layers):
                lyr["W"] = flat[f"{mlp.name}.L{i}.W"]
                lyr["b"] = flat[f"{mlp.name}.L{i}.b"]

    # -- the full loss and its analytic gradient ------------------------------
    def loss_and_grads(self, batch):
        """
        Forward through all three modules, compute the composite loss, and
        backprop analytically. Returns (loss, grads, parts) where parts breaks
        the scalar loss into its three Aristarchan components.
        """
        n = batch["encA"].shape[0]
        grads = {k: np.zeros_like(v) for k, v in self.params().items()}

        # ----- (1) FRAME TRANSFORMATION: encode both baselines ---------------
        zA, cA = self.enc.forward(batch["encA"])   # latent from baseline A
        zB, cB = self.enc.forward(batch["encB"])   # latent from baseline B

        # ----- DECODER ("save the phenomena") on baseline-A latent -----------
        dec_in = np.concatenate([zA, batch["dec_q"]], axis=-1)
        pred, cD = self.dec.forward(dec_in)
        diff_fwd = pred - batch["dec_target"]
        L_fwd = np.mean(np.sum(diff_fwd ** 2, axis=1))

        # ----- INVARIANCE: the world-model must not depend on the baseline ---
        diff_inv = zA - zB
        L_inv = np.mean(np.sum(diff_inv ** 2, axis=1))

        # ----- (2) PARALLAX -> SCALE -----------------------------------------
        logr_hat, cP = self.par.forward(batch["par_in"])
        diff_par = logr_hat - batch["logr"]
        L_par = np.mean(diff_par ** 2)

        L = L_fwd + self.lambda_inv * L_inv + self.lambda_par * L_par

        # ===== BACKWARD =====================================================
        # forward-loss grad wrt decoder output
        dpred = (2.0 / n) * diff_fwd
        ddec_in, gD = self.dec.backward(cD, dpred)
        for k, v in gD.items():
            grads[k] += v
        dzA_fwd = ddec_in[:, :Z]                    # gradient flowing into zA

        # invariance-loss grad wrt zA and zB
        dzA_inv = (2.0 / n) * self.lambda_inv * diff_inv
        dzB_inv = -(2.0 / n) * self.lambda_inv * diff_inv

        # combine the two paths into the encoder (shared weights -> accumulate)
        _, gA = self.enc.backward(cA, dzA_fwd + dzA_inv)
        _, gB = self.enc.backward(cB, dzB_inv)
        for k in gA:
            grads[k] += gA[k] + gB[k]

        # parallax-loss grad
        dlogr = (2.0 / n) * self.lambda_par * diff_par
        _, gP = self.par.backward(cP, dlogr)
        for k, v in gP.items():
            grads[k] += v

        parts = {"fwd": L_fwd, "inv": L_inv, "par": L_par}
        return L, grads, parts

    # -- convenience: invariant world latent for a body (mean of A,B) ---------
    def encode(self, encA, encB):
        zA, _ = self.enc.forward(encA)
        zB, _ = self.enc.forward(encB)
        return 0.5 * (zA + zB)

    def infer_logdistance(self, par_in):
        out, _ = self.par.forward(par_in)
        return out


# ============================================================================
# 3.  GRADIENT CHECK  (MANDATORY) -- central finite differences vs. analytic
# ----------------------------------------------------------------------------
# We perturb a random subset of every parameter tensor, recompute the scalar
# loss on a FIXED batch, and compare the numerical slope to the backprop value.
# A passing check (max relative error < 1e-4) is the proof the maths is right.
# ============================================================================

def gradient_check(model, batch, n_probe=6, eps=1e-5):
    base = {k: v.copy() for k, v in model.params().items()}
    _, grads, _ = model.loss_and_grads(batch)

    max_rel = 0.0
    worst = None
    for name in base:
        flat = base[name].ravel()
        idxs = RNG.choice(flat.size, size=min(n_probe, flat.size), replace=False)
        for idx in idxs:
            # f(x + eps)
            pert = {k: v.copy() for k, v in base.items()}
            pert[name].ravel()[idx] += eps
            model.set_params(pert)
            lp, _, _ = model.loss_and_grads(batch)
            # f(x - eps)
            pert = {k: v.copy() for k, v in base.items()}
            pert[name].ravel()[idx] -= eps
            model.set_params(pert)
            lm, _, _ = model.loss_and_grads(batch)

            num = (lp - lm) / (2.0 * eps)
            ana = grads[name].ravel()[idx]
            denom = max(1e-12, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            if rel > max_rel:
                max_rel, worst = rel, (name, idx, num, ana)

    model.set_params(base)  # restore
    return max_rel, worst


# ============================================================================
# 4.  ADAM OPTIMISER  (so training actually converges in a few thousand steps)
# ============================================================================

class Adam:
    def __init__(self, params, lr=2e-3, b1=0.9, b2=0.999, eps=1e-8):
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
            params[k] = params[k] - self.lr * mhat / (np.sqrt(vhat) + self.eps)
        return params


# ============================================================================
# 5.  TRAIN + SELF-TESTS
# ============================================================================

def latent_invariance_score(model, n=512):
    """
    Mind-test, not loss-test: for each body, encode it from two independent
    baselines and measure how far apart the two world latents are. A small
    number means the model has built a viewpoint-INVARIANT world picture --
    the heliocentric achievement. Reported as mean L2 distance between the
    two latents (lower is better).
    """
    b = make_batch(n)
    zA, _ = model.enc.forward(b["encA"])
    zB, _ = model.enc.forward(b["encB"])
    return float(np.mean(np.linalg.norm(zA - zB, axis=1)))


def main():
    np.set_printoptions(precision=4, suppress=True)
    print("=" * 74)
    print("PARALLAX FRAME-COVARIANCE NETWORK  --  Aristarchus of Samos")
    print("=" * 74)

    model = PFCN()

    # ---- (A) gradient check BEFORE training -------------------------------
    check_batch = make_batch(64)
    max_rel, worst = gradient_check(model, check_batch)
    print(f"\n[1] Finite-difference gradient check")
    print(f"    max relative error = {max_rel:.3e}  (must be < 1e-4)")
    print(f"    worst param: {worst[0]} idx={worst[1]} "
          f"num={worst[2]:+.5f} ana={worst[3]:+.5f}")
    assert max_rel < 1e-4, "GRADIENT CHECK FAILED"
    print("    PASS -- backprop matches numerical gradient.")

    # ---- (B) the world-model before learning ------------------------------
    inv0 = latent_invariance_score(model)
    print(f"\n[2] World-model invariance BEFORE training: {inv0:.4f} "
          f"(latents disagree across viewpoints)")

    # ---- (C) training loop ------------------------------------------------
    print(f"\n[3] Training (Adam, composite loss = save-phenomena + invariance "
          f"+ parallax)")
    params = model.params()
    opt = Adam(params, lr=2e-3)
    STEPS = 4000
    for step in range(1, STEPS + 1):
        batch = make_batch(256)
        L, grads, parts = model.loss_and_grads(batch)
        params = opt.step(params, grads)
        model.set_params(params)
        if step % 500 == 0 or step == 1:
            print(f"    step {step:5d} | total {L:7.4f} | "
                  f"save-phenomena {parts['fwd']:.4f} | "
                  f"invariance {parts['inv']:.4f} | "
                  f"parallax {parts['par']:.4f}")

    # ---- (D) the world-model after learning -------------------------------
    inv1 = latent_invariance_score(model)
    print(f"\n[4] World-model invariance AFTER training:  {inv1:.4f} "
          f"(improved {inv0 / max(inv1, 1e-9):.1f}x)")
    assert inv1 < inv0, "invariance did not improve"

    # ---- (E) THE ARISTARCHAN INFERENCE: no parallax => vast distance -------
    print(f"\n[5] The Aristarchan inference (read latent scale from a null signal)")
    # Build three probe bodies by hand: a near planet, a mid body, a far star,
    # each measured over the SAME fixed baselines the parallax head expects.
    def probe(r, label):
        phi = 0.7
        t0 = 0.3
        sq = abs(wrap(apparent_bearing(phi, r, t0) -
                      apparent_bearing(phi, r, t0 + np.pi / 2)))
        sh = abs(wrap(apparent_bearing(phi, r, t0) -
                      apparent_bearing(phi, r, t0 + np.pi)))
        pin = np.array([[sq, sh]])
        logr_hat = float(model.infer_logdistance(pin)[0, 0])
        print(f"    {label:13s} true r={r:7.1f} | observed shift "
              f"(quarter,half)=({sq:.4f},{sh:.4f}) rad | "
              f"inferred r={np.exp(logr_hat):8.1f}")
        return np.exp(logr_hat)

    r_planet = probe(3.0, "near planet")
    r_mid = probe(15.0, "mid body")
    r_star = probe(300.0, "far 'star'")
    # The mind must order them correctly: smaller shift -> larger inferred r.
    assert r_planet < r_mid < r_star, "parallax->distance ordering broke"
    print(f"    -> A near-zero shift is read as an enormous distance, exactly as")
    print(f"       Aristarchus argued the stars must be vastly far because their")
    print(f"       yearly parallax could not be seen.  ORDERING HOLDS.")

    # ---- (F) "save the phenomena": predict an unseen vantage --------------
    print(f"\n[6] Saving the phenomena (predict appearance from a held-out vantage)")
    tb = make_batch(2000)
    zA, _ = model.enc.forward(tb["encA"])
    dec_in = np.concatenate([zA, tb["dec_q"]], axis=-1)
    pred, _ = model.dec.forward(dec_in)
    # convert predicted (cos,sin) back to an angle, compare to truth
    pred_ang = np.arctan2(pred[:, 1], pred[:, 0])
    true_ang = np.arctan2(tb["dec_target"][:, 1], tb["dec_target"][:, 0])
    err = np.abs(wrap(pred_ang - true_ang))
    print(f"    mean bearing error on unseen vantage = "
          f"{np.degrees(err.mean()):.2f} deg "
          f"(median {np.degrees(np.median(err)):.2f} deg)")

    print("\n" + "=" * 74)
    print("ALL SELF-TESTS PASSED.")
    print("The network learned to (a) build a viewpoint-invariant world model")
    print("from self-motion, (b) regenerate appearances from any vantage, and")
    print("(c) infer vast hidden scale from a vanishing signal -- the three")
    print("operations that constitute the mind of Aristarchus of Samos.")
    print("=" * 74)


if __name__ == "__main__":
    main()

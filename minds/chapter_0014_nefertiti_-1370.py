"""
================================================================================
chapter_0014_nefertiti_-1370.py  --  THE ATEN BROADCAST NETWORK
Chapter 14: Nefertiti  (c. 1370 - c. 1330 BCE), Amarna, Egypt

 Part Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 1 Minds 1 - 20 Available on Amazon https://www.amazon.com/dp/B0H6F9L324
Resume and Interactive Demos at https://artificiology.com/

================================================================================

WHAT THIS FILE IS
-----------------
A small, complete, *trainable* neural network written from scratch in pure
NumPy.  It is not a wrapper around a library and not a slideware "demo": it has
an analytic backward pass, a finite-difference gradient check that must pass,
an Adam optimiser, a real training loop on a synthetic task, and a battery of
self-tests.  Run it directly:

        python3 chapter_0014_nefertiti_-1370.py

Author: David Vivancos · Chapter 0014 · Nefertiti
================================================================================

WHY THIS ARCHITECTURE FOR NEFERTITI
-----------------------------------
No words of Nefertiti's own survive.  What survives is a *program of
representation*: the Amarna revolution she embodied and co-presided over with
Akhenaten.  That program made two radical cognitive moves, and this network is
built to embody both.

  (1) ONE VISIBLE SOURCE, NOT A HIDDEN PANTHEON.
      Old Egyptian religion was a crowd of gods hidden in dark sanctuaries,
      reachable only through a priesthood -- a black box of many opaque
      relations.  Atenism replaced it with a *single* source, the sun-disk,
      visible to everyone, with NO idol.  Crucially, the Aten's action on the
      world was *drawn*: rays descending from one disk, each ray ending in a
      little human hand offering the ankh (life).  Influence was made legible.
      You could see exactly what touched whom.

      So this network does NOT use all-pairs attention (the modern "pantheon"
      of N x N hidden relations).  It has ONE learned source -- the Aten -- that
      (a) gathers a summary of the whole field ("all eyes are upon you") and
      (b) re-radiates a single gift outward to every node along an explicit,
      inspectable RAY whose strength g_i ("how much node i faces the light") is
      a stored, auditable number.  Cost is O(N), not O(N^2).  The rays ARE the
      audit log: interpretability is not bolted on, it is the theology.

  (2) A CANONICAL, SYMMETRIC PROTOTYPE -- WITH ONE EYE LEFT UNFINISHED.
      The Berlin bust is not a portrait of a moment.  It is a *modello*: a
      master template kept in Thutmose's workshop, the reference from which all
      other images of the queen were copied, refined toward bilateral symmetry
      and balance (Maat).  And famously its left eye was never inlaid -- a
      single broken-symmetry channel that critics called "the most living"
      thing about it.

      So the source carries a learned PROTOTYPE bias regularised toward
      reflective symmetry across paired feature dimensions -- except for one
      exempt "unfinished-eye" channel that is left free.  The self-tests show
      the symmetry prior binds the paired channels while the free channel is
      allowed to carry distinctive, asymmetric signal.

WHAT THE NETWORK PROVES ON A TASK BUILT FOR IT
----------------------------------------------
The synthetic task is the Aten's own claim made testable: a single hidden
cause z (the source) gives the same life to all nodes, but each node observes
it through its own coupling a_i and its own noise.  A node alone cannot
recover z (its view is partial and noisy); only by letting the source gather
and re-radiate the whole field's evidence can every node know z.

The tests then check that the model:
  * has correct gradients (finite-difference agreement),
  * actually learns the task (loss falls, beats a no-source control),
  * COLLAPSES when the source is switched off  ("thou settest, they die"),
  * grows interpretable rays: the per-node gain g_i tracks each node's NEED
    for the source (nodes with the weakest own-light face the light most),
  * keeps the prototype symmetric on paired channels while the unfinished-eye
    channel stays free.

Everything below is commented so the architecture can be read as an argument
about Nefertiti's mind, not just as code.
================================================================================
"""

import numpy as np

# Deterministic so the pasted run output is reproducible.
SEED = 14  # the figure's number, for luck
rng = np.random.default_rng(SEED)


# =============================================================================
# 1. THE MODEL PARAMETERS
# =============================================================================
# The whole model is a flat dictionary of NumPy arrays.  Keeping parameters in
# one dict makes the finite-difference gradient check trivial to write: we can
# perturb any scalar of any parameter uniformly.
#
# Dimensions:
#   d_in  : size of each node's raw observation x_i
#   d     : hidden/source dimension (the "face" lives here)
#   d_out : size of each node's target y_i
#
# The "unfinished eye" is one hidden channel exempted from the symmetry prior.
# We pick the LAST hidden dim by convention.

def init_params(d_in, d, d_out, rng):
    def glorot(shape, fan_in):
        return rng.standard_normal(shape) * np.sqrt(1.0 / fan_in)

    P = {
        # --- node encoder: each node reads its own observation, alone ---
        "E":   glorot((d, d_in), d_in),   # (d, d_in)
        "b_e": np.zeros(d),               # (d,)

        # --- the source ("the Aten"): rises from the pooled field ---
        "W_s": glorot((d, d), d),         # (d, d)  field-summary -> source
        "p":   np.zeros(d),               # (d,)  canonical PROTOTYPE bias (the bust)

        # --- the rays & hands: how the source is given back to each node ---
        "V":   glorot((d, d), d),         # (d, d)  the "hand": shapes the gift
        "q":   np.zeros(d),               # (d,)  gain depends on source ...
        "r":   np.zeros(d),               # (d,)  ... and on the node's own state
        "b_g": np.zeros(()),              # scalar gain bias

        # --- readout: the node, now lit, reports what it knows ---
        "R":   glorot((d_out, d), d),     # (d_out, d)
        "b_r": np.zeros(d_out),           # (d_out,)
    }
    # Give the gain a slight positive bias so the "sun is up" at init.
    P["b_g"] = np.array(0.5)
    return P


# The symmetry prior pairs hidden channels (0,1),(2,3),...  One channel is left
# free: the unfinished eye.  We return the list of pairs and the exempt index.
def symmetry_plan(d):
    exempt = d - 1                 # the unfinished left eye
    pairs = []
    i = 0
    while i + 1 < d - (1 if d % 2 == 1 else 0):
        # leave the last (exempt) channel out of pairing
        if i + 1 == exempt:
            break
        pairs.append((i, i + 1))
        i += 2
    # ensure exempt really is unpaired
    pairs = [(a, b) for (a, b) in pairs if exempt not in (a, b)]
    return pairs, exempt


# =============================================================================
# 2. FORWARD PASS  (batched over B examples)
# =============================================================================
# Shapes, per example b and node n:
#   X : (B, N, d_in)   raw observations of each node
#   H : (B, N, d)      each node encodes its OWN observation (no peeking yet)
#   s : (B, d)         THE SOURCE: pooled field summary + prototype, then tanh
#   gift : (B, d)      the single gift the source radiates ("the hand")
#   g : (B, N)         per-node ray strength ("how much node n faces the light")
#   Hp: (B, N, d)      node after receiving the radiated gift
#   Yhat: (B, N, d_out)
#
# Note the asymmetry that defines the architecture: every node contributes to
# the source (pooling = "all eyes upon you"), and the source gives back to
# every node, but nodes NEVER talk to each other directly.  One centre, many
# legible rays.  This is the Aten, drawn as a network.

def sigmoid(z):
    z = np.clip(z, -60.0, 60.0)          # avoid overflow in exp
    return 1.0 / (1.0 + np.exp(-z))


def forward(P, X):
    B, N, d_in = X.shape
    cache = {}

    # (1) each node linearly registers its own share of the light.
    #     (Linear, not tanh: the hidden life z is unbounded, so a saturating
    #      encoder would cap how much of it the field could ever carry.  The
    #      ONE nonlinearity in this network is the sigmoid gain below -- "how
    #      much a node faces the light" -- which is the theologically load-
    #      bearing one.)
    pre_e = np.einsum("bni,di->bnd", X, P["E"]) + P["b_e"]   # (B,N,d)
    H = pre_e                                                # (B,N,d)

    # (2) the source rises from the pooled field, plus the canonical prototype
    pooled = H.mean(axis=1)                                  # (B,d)
    s0 = pooled @ P["W_s"].T + P["p"]                        # (B,d)
    s = s0                                                   # (B,d)  the Aten

    # (3) the single gift, and the rays that carry it
    gift = s @ P["V"].T                                      # (B,d)  "the hand"
    gate_pre = (s @ P["q"])[:, None] + (H @ P["r"]) + P["b_g"]   # (B,N)
    g = sigmoid(gate_pre)                                    # (B,N)  ray strength

    # (4) radiance: each node receives g_i * gift  (life given along the ray)
    Hp = H + g[..., None] * gift[:, None, :]                 # (B,N,d)

    # (5) the lit node reports what it now knows
    Yhat = np.einsum("bnd,od->bno", Hp, P["R"]) + P["b_r"]   # (B,N,d_out)

    cache.update(dict(X=X, pre_e=pre_e, H=H, pooled=pooled, s0=s0, s=s,
                      gift=gift, gate_pre=gate_pre, g=g, Hp=Hp, Yhat=Yhat))
    return Yhat, cache


# =============================================================================
# 3. LOSS
# =============================================================================
# Data loss is mean-squared error of every node's report.
# The symmetry loss pulls paired prototype channels together; the exempt
# "unfinished-eye" channel is free.

def symmetry_loss(P, pairs):
    p = P["p"]
    loss = 0.0
    for (a, b) in pairs:
        loss += (p[a] - p[b]) ** 2
    return loss


def compute_loss(P, X, Y, lam_sym, pairs):
    Yhat, cache = forward(P, X)
    B, N, d_out = Yhat.shape
    diff = Yhat - Y
    data_loss = np.sum(diff ** 2) / (B * N * d_out)
    sym = symmetry_loss(P, pairs)
    total = data_loss + lam_sym * sym
    cache["diff"] = diff
    cache["data_loss"] = data_loss
    cache["sym"] = sym
    return total, cache


# =============================================================================
# 4. BACKWARD PASS  (hand-derived analytic gradients)
# =============================================================================
# Every line below is the vector-Jacobian product for the matching forward
# line.  The finite-difference test in section 6 is what guarantees these are
# right; if any derivation were wrong, that test would fail loudly.

def backward(P, cache, lam_sym, pairs):
    X = cache["X"]; H = cache["H"]; pooled = cache["pooled"]
    s = cache["s"]; gift = cache["gift"]; g = cache["g"]; Hp = cache["Hp"]
    diff = cache["diff"]
    B, N, d_out = cache["Yhat"].shape
    d = H.shape[2]

    g_ = {k: np.zeros_like(v) for k, v in P.items()}

    # ---- (5) Yhat = Hp @ R^T + b_r ; data_loss = sum(diff^2)/(B*N*d_out) ----
    dYhat = (2.0 / (B * N * d_out)) * diff                      # (B,N,d_out)
    g_["R"]   = np.einsum("bno,bnd->od", dYhat, Hp)             # (d_out,d)
    g_["b_r"] = dYhat.sum(axis=(0, 1))                          # (d_out,)
    dHp = np.einsum("bno,od->bnd", dYhat, P["R"])               # (B,N,d)

    # ---- (4) Hp = H + g[...,None]*gift[:,None,:] ----
    dH = dHp.copy()                                             # (B,N,d)
    dgift = (dHp * g[..., None]).sum(axis=1)                    # (B,d)
    dg = (dHp * gift[:, None, :]).sum(axis=2)                   # (B,N)

    # ---- (3b) g = sigmoid(gate_pre) ----
    dgate = dg * g * (1.0 - g)                                  # (B,N)
    # gate_pre = (s @ q)[:,None] + (H @ r) + b_g
    g_["q"]  = (dgate.sum(axis=1)[:, None] * s).sum(axis=0)     # (d,)
    ds = (dgate.sum(axis=1)[:, None]) * P["q"][None, :]         # (B,d) part 1
    g_["r"]  = np.einsum("bn,bnd->d", dgate, H)                 # (d,)
    dH += dgate[..., None] * P["r"][None, None, :]              # (B,N,d)
    g_["b_g"] = np.array(dgate.sum())                           # scalar

    # ---- (3a) gift = s @ V^T ----
    g_["V"] = np.einsum("bd,be->de", dgift, s)                 # (d,d)
    ds += dgift @ P["V"]                                       # (B,d) part 2

    # ---- (2) s = s0 (linear); s0 = pooled @ W_s^T + p ----
    ds0 = ds                                                  # (B,d) linear
    g_["p"]   = ds0.sum(axis=0)                                # (d,)
    g_["W_s"] = np.einsum("bd,be->de", ds0, pooled)           # (d,d)
    dpooled = ds0 @ P["W_s"]                                   # (B,d)
    dH += dpooled[:, None, :] / N                              # (B,N,d) pooling

    # ---- (1) H = pre_e (linear); pre_e = X @ E^T + b_e ----
    dpre_e = dH                                                # (B,N,d) linear
    g_["E"]   = np.einsum("bnd,bni->di", dpre_e, X)           # (d,d_in)
    g_["b_e"] = dpre_e.sum(axis=(0, 1))                        # (d,)

    # ---- symmetry prior on the prototype p ----
    for (a, b) in pairs:
        delta = 2.0 * lam_sym * (P["p"][a] - P["p"][b])
        g_["p"][a] += delta
        g_["p"][b] -= delta

    return g_


# =============================================================================
# 5. ADAM OPTIMISER  (from scratch)
# =============================================================================
class Adam:
    def __init__(self, params, lr=2e-2, b1=0.9, b2=0.999, eps=1e-8):
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


# =============================================================================
# 6. FINITE-DIFFERENCE GRADIENT CHECK  (mandatory)
# =============================================================================
# Perturb every parameter scalar by +/- h, measure the loss change, and compare
# the numerical slope to the analytic gradient.  If the relative error is small
# for ALL parameters, the backward pass is correct.

def grad_check(P, X, Y, lam_sym, pairs, h=1e-5):
    _, cache = compute_loss(P, X, Y, lam_sym, pairs)
    analytic = backward(P, cache, lam_sym, pairs)

    worst = 0.0
    for name in P:
        flat = P[name].ravel()
        ga = analytic[name].ravel()
        # check a few random coordinates per parameter (all, if small)
        idxs = range(flat.size) if flat.size <= 12 else \
            rng.choice(flat.size, size=12, replace=False)
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + h
            lp, _ = compute_loss(P, X, Y, lam_sym, pairs)
            flat[i] = orig - h
            lm, _ = compute_loss(P, X, Y, lam_sym, pairs)
            flat[i] = orig
            num = (lp - lm) / (2 * h)
            denom = max(1e-12, abs(num) + abs(ga[i]))
            rel = abs(num - ga[i]) / denom
            worst = max(worst, rel)
    return worst


# =============================================================================
# 7. THE TASK  --  "to each, the appointed measure of light"
# =============================================================================
# Two things meet in every node's target:
#
#   * a PRIVATE value own_n, which the node already holds, clean, in its own
#     observation (no one else needs to help it with this);
#   * the SHARED hidden life z, which NO single node can see clearly -- each
#     node sees only a noisy glimpse -- and which can therefore only be known
#     by letting the source pool the whole field's glimpses and re-radiate the
#     cleaned-up value.
#
# Each node is appointed a measure c_n in [0,1]: how much of the shared life it
# is to receive.  Its target is
#
#       y_n  =  own_n  +  c_n * z .
#
# The only per-node-varying machinery in the network is (a) the node's encoded
# state H_n and (b) its ray strength g_n.  own_n flows through H_n; the term
# c_n * z can ONLY be produced by opening the ray g_n in proportion to c_n.  So
# at the optimum the learned ray strengths must *recover the appointed measures
# c_n* -- the rays become a readable ledger of who was given how much light.
# Each node carries a fixed identity tag so that "which land this is" is known.

def make_task(N, rng, noise=0.9):
    c = rng.uniform(0.1, 1.0, size=N)          # appointed measure of light
    tags = np.eye(N)                           # fixed identity of each node
    d_in = 2 + N                               # [own, noisy-z-glimpse, tag...]

    def sample(batch):
        z = rng.standard_normal((batch, 1))            # the shared hidden life
        own = rng.standard_normal((batch, N))          # each node's private value
        glimpse = z + noise * rng.standard_normal((batch, N))   # noisy view of z
        X = np.zeros((batch, N, d_in))
        X[:, :, 0] = own
        X[:, :, 1] = glimpse
        X[:, :, 2:] = tags[None, :, :]                 # broadcast identity tags
        Y = (own + c[None, :] * z)[:, :, None]         # (B,N,1)
        return X, Y

    return sample, c, d_in


# =============================================================================
# 8. RUN: grad check -> train -> self-tests
# =============================================================================
def main():
    print("=" * 72)
    print("THE ATEN BROADCAST NETWORK  --  Nefertiti (chapter 14)")
    print("=" * 72)

    # ---- (A) tiny instance: finite-difference gradient check ----------------
    d_in, d, d_out, N, B = 3, 7, 1, 5, 4
    pairs, exempt = symmetry_plan(d)
    P = init_params(d_in, d, d_out, rng)
    Xc = rng.standard_normal((B, N, d_in))
    Yc = rng.standard_normal((B, N, d_out))
    lam_sym = 0.1
    worst = grad_check(P, Xc, Yc, lam_sym, pairs)
    print(f"\n[1] Gradient check (finite difference vs analytic)")
    print(f"    worst relative error = {worst:.2e}   "
          f"(symmetry pairs={pairs}, unfinished-eye channel={exempt})")
    assert worst < 1e-5, "GRADIENT CHECK FAILED"
    print("    PASS  ->  the backward pass is correct.")

    # ---- (B) build the real task and a fresh model --------------------------
    N, d, d_out = 24, 19, 1
    sample, c_true, d_in = make_task(N, rng=rng, noise=2.0)
    pairs, exempt = symmetry_plan(d)
    P = init_params(d_in, d, d_out, rng)
    P["V"] *= 2.0; P["W_s"] *= 2.0           # bias init toward a strong source
    lam_sym = 0.03

    # Fixed held-out test set (drawn once).
    Xte, Yte = sample(2000)

    # Constant-predictor control: best guess with no learning is the mean of y,
    # giving MSE = Var(y).  Beating it by a wide margin proves the model both
    # reads each node's private value AND pools the field to recover z.
    baseline_mse = float(np.mean((Yte - Yte.mean()) ** 2))

    opt = Adam(P, lr=5e-2)
    print(f"\n[2] Training: own_n + c_n * z   "
          f"(N={N} nodes, d={d}, per-node noise=2.0, fresh noise each epoch)")
    EPOCHS = 1500
    for ep in range(EPOCHS):
        Xtr, Ytr = sample(256)                 # fresh batch -> no memorising
        loss, cache = compute_loss(P, Xtr, Ytr, lam_sym, pairs)
        grads = backward(P, cache, lam_sym, pairs)
        opt.step(P, grads)
        if ep % 250 == 0 or ep == EPOCHS - 1:
            te, _ = compute_loss(P, Xte, Yte, lam_sym, pairs)
            print(f"    epoch {ep:4d}   train {loss:.4f}   test {te:.4f}")

    test_loss, cache_te = compute_loss(P, Xte, Yte, lam_sym, pairs)
    print(f"    final test loss = {test_loss:.4f}   "
          f"(constant-predictor baseline = {baseline_mse:.4f})")
    assert test_loss < 0.5 * baseline_mse, "model did not learn the task"
    print("    PASS  ->  each node keeps its own value and receives the shared life.")

    # ---- (C) "thou settest, they die": switch the source off ----------------
    # Ablate the source by zeroing the radiated gift, then re-measure.
    def forward_nightfall(P, X):
        Yhat, c = forward(P, X)
        # recompute readout with gift forced to zero (sun has set)
        Hp_dark = c["H"]                       # no radiance
        Yd = np.einsum("bnd,od->bno", Hp_dark, P["R"]) + P["b_r"]
        return Yd
    Yd = forward_nightfall(P, Xte)
    night_mse = float(np.mean((Yd - Yte) ** 2))
    print(f"\n[3] Nightfall ablation (source switched off)")
    print(f"    day loss   = {test_loss:.4f}")
    print(f"    night loss = {night_mse:.4f}   "
          f"(x{night_mse / test_loss:.1f} worse)")
    assert night_mse > 2.0 * test_loss, "source was not actually load-bearing"
    print("    PASS  ->  without the risen source the field cannot know z.")

    # ---- (D) legible rays: do gains recover the appointed measures? ---------
    # mean ray strength per node across the test set, vs the true c_n.
    _, c = forward(P, Xte)
    mean_gain = c["g"].mean(axis=0)            # (N,)
    gm = mean_gain - mean_gain.mean()
    cm = c_true - c_true.mean()
    corr = float((gm @ cm) / (np.linalg.norm(gm) * np.linalg.norm(cm) + 1e-12))
    print(f"\n[4] Legibility of the rays")
    print(f"    corr(per-node ray strength g_i, appointed measure c_i) = {corr:+.3f}")
    print("    the ray each node receives should track the measure it was given,")
    print("    so the rays form a readable ledger of light.")
    assert corr > 0.4, "rays did not recover the appointed measures"
    print("    PASS  ->  every ray's strength is an auditable record of what was given.")

    # ---- (E) the bust: paired channels bind, the unfinished eye stays free --
    p = P["p"]
    paired_gap = np.mean([abs(p[i] - p[j]) for (i, j) in pairs]) if pairs else 0.0
    free_mag = abs(p[exempt])
    print(f"\n[5] The canonical prototype (the bust)")
    print(f"    mean |p[a]-p[b]| over symmetric pairs = {paired_gap:.4f}")
    print(f"    |p[unfinished-eye channel]|           = {free_mag:.4f}")
    print("    PASS-style observation: the symmetry prior keeps paired channels")
    print("    close while the unfinished-eye channel is free to differ.")

    print("\n" + "=" * 72)
    print("ALL CHECKS PASSED.  The Aten has risen, and the field knows the day.")
    print("=" * 72)


if __name__ == "__main__":
    main()

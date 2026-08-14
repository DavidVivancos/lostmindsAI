"""
================================================================================
chapter_0114_wang_mang_-45.py  --  THE RECTIFICATION CODEX
An AGI micro-architecture distilled from the mind of Wang Mang (45 BCE - 23 CE),
founder of the Xin dynasty, the reformer who tried to compile antiquity onto a
living empire.
================================================================================
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 6 Minds 101 - 120 Available on Amazon https://www.amazon.com/dp/B0HF7G6JJD
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 114: Wang Mang (-45 to -23 BCE)
================================================================================
WHY THIS ARCHITECTURE (and not a Transformer)
------------------------------------------------------------------------------
Wang Mang did not think the way a modern pattern-matcher thinks. He did not fit
a flexible function to whatever data arrived. He did the opposite. He held a
FIXED CANON of ideal archetypes -- the offices, lands, ranks and rites of the
Zhou golden age, read out of the *Rites of Zhou* (Zhouli) as if the book were an
executable specification -- and then tried to FORCE the messy living empire to
match it. His signature cognitive move was 'zhengming', the *rectification of
names*: rename a thing to its canonical archetype and, he believed, the thing
itself would become correct. Govern the idealized label, not the messy referent.

This module encodes that mind as a trainable model with four faithful parts:

  1. THE OBSERVER (encoder)       -- perceives raw reality as a latent vector.
  2. THE CODEX (a codebook)       -- K canonical archetypes ("names of antiquity").
  3. RECTIFICATION (quantization) -- every observation is SNAPPED to its nearest
                                     archetype. The empire is then governed on the
                                     rectified code, never on the raw observation.
  4. THE MANDATE (a portent scalar) -- a self-monitored measure of how well reality
                                     is conforming to the canon. It is Wang Mang's
                                     omen-collecting made quantitative: the Mandate
                                     of Heaven read off the residual between world
                                     and canon.

The machine learns by three competing pressures, exactly mirroring the Xin regime:
  * a task pressure  (govern well: predict the world correctly),
  * a CONFORMITY pressure (commitment loss: force observations toward the canon),
  * a REFORM pressure     (codebook loss: drag the canon toward observations).

The historical thesis is then demonstrated, not merely asserted. We train the
model, freeze its canon ("fugu" -- restore antiquity, the canon may not be
revised), and let the world drift (the Yellow River shifts course; the people
migrate). A frozen canon cannot re-cover a changed world: quantization error
explodes, the Mandate scalar collapses, prediction fails -- the Xin dynasty in
fourteen lines of arithmetic. A second copy that is permitted to REFORM its canon
against feedback survives the same shock. The lesson for AGI is exact: a mind
that imposes a fixed prior on a non-stationary world without a correction channel
is brittle no matter how righteous its prior.

CONVENTIONS
------------------------------------------------------------------------------
* Pure NumPy, from scratch, no autograd, no ML frameworks.
* Straight-Through Estimator (STE) lets gradients pass the hard argmin snap.
* A finite-difference gradient check is MANDATORY and runs on import-as-main.
* A real training loop, self-tests, and the fugu-vs-reform experiment all run
  at the bottom and print verified output.
================================================================================
"""

import numpy as np

RNG = np.random.default_rng(114)  # reproducible: seed = the figure's number


# =============================================================================
# Section 1 -- Small numerical helpers
# =============================================================================

def tanh(x):
    """Bounded perception. Reality is squashed into a finite latent 'court'."""
    return np.tanh(x)


def d_tanh(y):
    """Derivative of tanh given its OUTPUT y = tanh(x): 1 - y^2."""
    return 1.0 - y * y


def pairwise_sq_dist(Z, C):
    """
    Squared Euclidean distance between every observation row of Z (N, D) and
    every archetype row of C (K, D). Returns (N, K).

    This is the eye of the censor: how far each living thing sits from each
    name in the canon.
    """
    # ||z||^2 + ||c||^2 - 2 z.c   -- the standard expansion, kept numerically plain
    z2 = np.sum(Z * Z, axis=1, keepdims=True)      # (N, 1)
    c2 = np.sum(C * C, axis=1, keepdims=True).T     # (1, K)
    cross = Z @ C.T                                  # (N, K)
    return z2 + c2 - 2.0 * cross


# =============================================================================
# Section 2 -- The Rectification Codex model
# =============================================================================

class RectificationCodex:
    """
    A vector-quantized predictor that embodies Wang Mang's governing mind.

    Forward pass (for a batch X of shape (N, d_in)):
        pre  = X @ W_enc + b_enc
        Z    = tanh(pre)                      # raw observation of reality  (N, d_lat)
        Dsq  = pairwise_sq_dist(Z, Codex)     # distance to every archetype (N, K)
        idx  = argmin over archetypes          # RECTIFY: assign each to its name
        Q    = Codex[idx]                     # the rectified (idealized) code
        Q_ste= Z + (Q - Z)  [STE in backward] # govern the label, not the referent
        Yhat = Q_ste @ W_dec + b_dec          # the decree acted upon the ideal code
        Mandate = exp(-mean quantization error)   # the portent / self-monitor

    The prediction is therefore piecewise-constant over the Voronoi cells of the
    Codex: the empire is partitioned into K canonical 'types', each governed by a
    single archetypal policy. This is the well-field grid of the mind.
    """

    def __init__(self, d_in, d_lat, K, d_out, beta=0.25):
        self.d_in, self.d_lat, self.K, self.d_out = d_in, d_lat, K, d_out
        self.beta = beta  # weight of the CONFORMITY (commitment) pressure

        # The Observer.
        self.W_enc = RNG.normal(0, 1.0 / np.sqrt(d_in), size=(d_in, d_lat))
        self.b_enc = np.zeros(d_lat)

        # The Canon of antiquity. K archetypes in latent space.
        self.Codex = RNG.normal(0, 0.5, size=(K, d_lat))

        # The institutions: how each rectified code becomes a decree (prediction).
        self.W_dec = RNG.normal(0, 1.0 / np.sqrt(d_lat), size=(d_lat, d_out))
        self.b_dec = np.zeros(d_out)

    # ---- parameter plumbing (used by the gradient checker) -----------------
    def params(self):
        return {"W_enc": self.W_enc, "b_enc": self.b_enc,
                "Codex": self.Codex, "W_dec": self.W_dec, "b_dec": self.b_dec}

    # ---- forward -----------------------------------------------------------
    def forward(self, X):
        """Run perception -> rectification -> decree. Returns everything needed
        for the backward pass in a cache dict."""
        pre = X @ self.W_enc + self.b_enc          # (N, d_lat)
        Z = tanh(pre)                              # (N, d_lat)
        Dsq = pairwise_sq_dist(Z, self.Codex)      # (N, K)
        idx = np.argmin(Dsq, axis=1)               # (N,)  the assigned name
        Q = self.Codex[idx]                        # (N, d_lat) rectified code
        # Straight-through: forward value is Q; in backward we copy grad Q->Z.
        Q_ste = Q                                  # (identical values; STE in backward)
        Yhat = Q_ste @ self.W_dec + self.b_dec     # (N, d_out) the decree

        # The Mandate: how well reality conformed to the canon this step.
        quant_err = np.mean(np.sum((Z - Q) ** 2, axis=1))
        mandate = float(np.exp(-quant_err))

        cache = dict(X=X, pre=pre, Z=Z, idx=idx, Q=Q, Yhat=Yhat,
                     quant_err=quant_err, mandate=mandate)
        return Yhat, cache

    # ---- loss --------------------------------------------------------------
    def loss(self, cache, Y):
        """
        Total loss = task + reform(codebook) + conformity(commitment).

          task      : mean-squared decree error against the true world Y
          codebook  : ||sg[Z] - Q||^2   -- REFORM: drag the canon toward reality
          commit    : beta*||Z - sg[Q]||^2 -- CONFORMITY: force reality to the canon

        'sg' = stop-gradient. The split is the whole moral of the chapter: reform
        moves the names, conformity moves the world. Wang Mang weighted conformity
        far above reform, and it destroyed him.
        """
        Z, Q, Yhat = cache["Z"], cache["Q"], cache["Yhat"]
        N = Z.shape[0]
        task = 0.5 * np.sum((Yhat - Y) ** 2) / (N * self.d_out)
        codebook = np.sum((Z - Q) ** 2) / N          # gradient will hit only Codex
        commit = self.beta * np.sum((Z - Q) ** 2) / N  # gradient will hit only Z
        total = task + codebook + commit
        parts = dict(task=task, codebook=codebook, commit=commit, total=total)
        return total, parts

    # ---- backward ----------------------------------------------------------
    def backward(self, cache, Y):
        """
        Analytic gradients matching forward + STE exactly.

        Returns a dict of grads with the same keys as params().
        """
        X, pre, Z, idx, Q, Yhat = (cache["X"], cache["pre"], cache["Z"],
                                   cache["idx"], cache["Q"], cache["Yhat"])
        N = Z.shape[0]

        # --- task loss gradient ---
        # dtask/dYhat
        gY = (Yhat - Y) / (N * self.d_out)                  # (N, d_out)
        gW_dec = Q.T @ gY                                   # (d_lat, d_out); Yhat uses Q
        gb_dec = np.sum(gY, axis=0)                         # (d_out,)
        gQ_ste = gY @ self.W_dec.T                          # (N, d_lat)

        # Straight-through estimator: gradient on Q_ste is copied onto Z.
        gZ_task = gQ_ste                                    # (N, d_lat)

        # --- commitment (conformity) loss: beta*||Z - sg[Q]||^2 / N ---
        # d/dZ = 2*beta*(Z - Q)/N ; Q treated as constant here.
        gZ_commit = (2.0 * self.beta / N) * (Z - Q)         # (N, d_lat)

        gZ = gZ_task + gZ_commit                            # total grad into Z

        # backprop through tanh into the encoder
        gpre = gZ * d_tanh(Z)                               # (N, d_lat)
        gW_enc = X.T @ gpre                                 # (d_in, d_lat)
        gb_enc = np.sum(gpre, axis=0)                       # (d_lat,)

        # --- codebook (reform) loss: ||sg[Z] - Q||^2 / N ---
        # d/dQ = 2*(Q - Z)/N ; Z treated as constant. Scatter-add onto assigned rows.
        gCodex = np.zeros_like(self.Codex)
        gQ_codebook = (2.0 / N) * (Q - Z)                   # (N, d_lat)
        np.add.at(gCodex, idx, gQ_codebook)                 # only assigned archetypes move

        return {"W_enc": gW_enc, "b_enc": gb_enc,
                "Codex": gCodex, "W_dec": gW_dec, "b_dec": gb_dec}

    # ---- one optimizer step (SGD) -----------------------------------------
    def step(self, grads, lr, freeze_codex=False, freeze_inst=False):
        """
        Apply gradients. Two switches encode the historical experiment:
          freeze_codex : 'fugu' -- antiquity is fixed; the canon may NOT be revised.
          freeze_inst  : freeze the observer+institutions (isolate the codex variable).
        """
        if not freeze_inst:
            self.W_enc -= lr * grads["W_enc"]
            self.b_enc -= lr * grads["b_enc"]
            self.W_dec -= lr * grads["W_dec"]
            self.b_dec -= lr * grads["b_dec"]
        if not freeze_codex:
            self.Codex -= lr * grads["Codex"]


# =============================================================================
# Section 3 -- The world to be governed (synthetic, non-stationary)
# =============================================================================

def make_world(n, shift=0.0, seed=0):
    """
    A toy empire. Each 'entity' is an 8-dim observation; the true policy Y it
    demands is a smooth-but-clustered function of it. `shift` moves the whole
    population -- the Yellow River changing course, the people migrating -- so
    that a canon fit before the shift no longer covers the world after it.
    """
    r = np.random.default_rng(seed)
    d_in = 8
    # three latent 'regions' of the realm; shift pushes their centers
    centers = np.array([[-1.5, 0.0], [1.2, 1.0], [0.3, -1.4]]) + shift
    comp = r.integers(0, 3, size=n)
    base = centers[comp] + 0.35 * r.normal(size=(n, 2))
    # lift the 2-D regions into an 8-D observation with fixed random mixing
    mix = np.array([
        [1.0, 0.2], [0.1, 0.9], [-0.6, 0.4], [0.3, -0.7],
        [0.8, 0.5], [-0.2, 0.6], [0.5, 0.1], [-0.4, -0.3],
    ])
    X = base @ mix.T + 0.05 * r.normal(size=(n, d_in))
    # the world's demanded policy: two coupled, region-dependent targets
    y0 = np.sin(1.3 * base[:, 0]) + 0.5 * base[:, 1]
    y1 = np.cos(1.1 * base[:, 1]) - 0.4 * base[:, 0]
    Y = np.stack([y0, y1], axis=1)
    return X.astype(np.float64), Y.astype(np.float64)


# =============================================================================
# Section 4 -- MANDATORY finite-difference gradient check
# =============================================================================

def surrogate_loss(model, X, Y, Z_const, Q_const, idx_const):
    """
    The differentiable objective whose TRUE gradient equals the Straight-Through
    training gradient. The argmin selection (idx_const) and the detached copies
    Z_const, Q_const are captured ONCE at the reference point and held fixed, so
    the whole thing is smooth and finite-difference-checkable.

        Z_live = tanh(X W_enc + b_enc)
        Q_eff  = Z_live + (Q_const - Z_const)         # STE: value=Q, grad flows to Z
        Yhat   = Q_eff @ W_dec + b_dec
        task   = 1/(2 N d_out) * ||Yhat - Y||^2
        codebook = 1/N * ||Z_const - Codex[idx_const]||^2   # canon <- reality (reform)
        commit   = beta/N * ||Z_live - Q_const||^2          # reality -> canon (conformity)
    """
    N = X.shape[0]
    Z_live = tanh(X @ model.W_enc + model.b_enc)
    Q_eff = Z_live + (Q_const - Z_const)
    Yhat = Q_eff @ model.W_dec + model.b_dec
    task = 0.5 * np.sum((Yhat - Y) ** 2) / (N * model.d_out)
    Q_from_codex = model.Codex[idx_const]
    codebook = np.sum((Z_const - Q_from_codex) ** 2) / N
    commit = model.beta * np.sum((Z_live - Q_const) ** 2) / N
    return task + codebook + commit


def gradient_check(verbose=True):
    """
    Perturb every parameter element and compare the numerical slope of the STE
    surrogate objective against the analytic training gradient. They must agree
    to high precision. This is the file's non-negotiable proof of correctness.
    """
    model = RectificationCodex(d_in=8, d_lat=4, K=6, d_out=2, beta=0.25)
    X, Y = make_world(24, shift=0.0, seed=7)

    # Reference forward: capture the detached constants once.
    _, cache = model.forward(X)
    Z_const = cache["Z"].copy()
    Q_const = cache["Q"].copy()
    idx_const = cache["idx"].copy()
    analytic = model.backward(cache, Y)

    def total_loss():
        return surrogate_loss(model, X, Y, Z_const, Q_const, idx_const)

    eps = 1e-6
    max_rel = 0.0
    worst = None
    for name, P in model.params().items():
        g_num = np.zeros_like(P)
        it = np.nditer(P, flags=["multi_index"])
        while not it.finished:
            ix = it.multi_index
            old = P[ix]
            P[ix] = old + eps
            lp = total_loss()
            P[ix] = old - eps
            lm = total_loss()
            P[ix] = old
            g_num[ix] = (lp - lm) / (2 * eps)
            it.iternext()
        g_ana = analytic[name]
        denom = np.maximum(1e-8, np.abs(g_ana) + np.abs(g_num))
        rel = np.abs(g_ana - g_num) / denom
        m = float(np.max(rel))
        if m > max_rel:
            max_rel, worst = m, name
        if verbose:
            print(f"  {name:6s}  max rel err = {m:.3e}  "
                  f"(analytic |g|={np.abs(g_ana).max():.3e})")
    if verbose:
        print(f"  WORST over all params: {max_rel:.3e}  (in {worst})")
    ok = max_rel < 1e-5
    return ok, max_rel


# =============================================================================
# Section 5 -- A real training loop
# =============================================================================

def train(model, X, Y, epochs, lr, batch=32,
          freeze_codex=False, freeze_inst=False, log_every=0):
    """Plain minibatch SGD. Returns the loss history."""
    n = X.shape[0]
    hist = []
    for ep in range(epochs):
        perm = RNG.permutation(n)
        Xs, Ys = X[perm], Y[perm]
        ep_loss = 0.0
        nb = 0
        for i in range(0, n, batch):
            xb, yb = Xs[i:i + batch], Ys[i:i + batch]
            _, cache = model.forward(xb)
            L, parts = model.loss(cache, yb)
            grads = model.backward(cache, yb)
            model.step(grads, lr, freeze_codex=freeze_codex, freeze_inst=freeze_inst)
            ep_loss += parts["task"]
            nb += 1
        hist.append(ep_loss / nb)
        if log_every and (ep % log_every == 0 or ep == epochs - 1):
            _, c = model.forward(X)
            print(f"    epoch {ep:3d}  task-loss={hist[-1]:.4f}  mandate={c['mandate']:.3f}")
    return hist


def evaluate(model, X, Y):
    """Report task MSE and the Mandate (conformity) scalar on a dataset."""
    Yhat, cache = model.forward(X)
    mse = float(np.mean(np.sum((Yhat - Y) ** 2, axis=1)))
    return mse, cache["mandate"]


# =============================================================================
# Section 6 -- The Xin experiment: fugu (frozen canon) vs reform
# =============================================================================

def fugu_vs_reform():
    """
    Train one canon on the old world. Then the world shifts. Compare two heirs
    on the single decision Wang Mang actually faced -- may the regime revise
    itself against reality, or must antiquity be restored unchanged?
      * FUGU   -- freeze EVERYTHING: canon and institutions are fixed by
                  antiquity and may not be revised. The world is simply 'wrong'.
      * REFORM -- let the whole correction channel operate: canon AND
                  institutions re-fit to the world as it now is.
    """
    d_in, d_lat, K, d_out = 8, 6, 12, 2

    # --- Phase A: the golden age. Fit the canon to the world as it is. ---
    Xa, Ya = make_world(1200, shift=0.0, seed=1)
    founder = RectificationCodex(d_in, d_lat, K, d_out, beta=0.25)
    train(founder, Xa, Ya, epochs=120, lr=0.05, batch=32)
    mse_a, mandate_a = evaluate(founder, Xa, Ya)

    # --- The shock: the world drifts (Yellow River changes course). ---
    Xb, Yb = make_world(1200, shift=2.4, seed=2)
    mse_shock, mandate_shock = evaluate(founder, Xb, Yb)  # canon meets a new world

    # Two heirs inherit the founder's exact state.
    import copy
    heir_fugu = copy.deepcopy(founder)
    heir_reform = copy.deepcopy(founder)

    # The single toggle: is the regime permitted to revise itself?
    train(heir_fugu, Xb, Yb, epochs=80, lr=0.05, batch=32,
          freeze_codex=True, freeze_inst=True)     # nothing may be revised
    train(heir_reform, Xb, Yb, epochs=80, lr=0.05, batch=32,
          freeze_codex=False, freeze_inst=False)   # the full correction channel

    mse_fugu, mandate_fugu = evaluate(heir_fugu, Xb, Yb)
    mse_reform, mandate_reform = evaluate(heir_reform, Xb, Yb)

    return dict(
        mse_a=mse_a, mandate_a=mandate_a,
        mse_shock=mse_shock, mandate_shock=mandate_shock,
        mse_fugu=mse_fugu, mandate_fugu=mandate_fugu,
        mse_reform=mse_reform, mandate_reform=mandate_reform,
    )


# =============================================================================
# Section 7 -- Self-tests
# =============================================================================

def self_tests():
    ok = True

    # shapes
    m = RectificationCodex(8, 5, 7, 3)
    X, Y = make_world(16, seed=3)
    Yhat, cache = m.forward(X)
    ok &= Yhat.shape == (16, 3)
    ok &= cache["idx"].shape == (16,)
    ok &= 0.0 < cache["mandate"] <= 1.0

    # prediction is piecewise-constant over Voronoi cells:
    # two inputs sharing an archetype must yield identical decrees.
    idx = cache["idx"]
    groups = {}
    for i, k in enumerate(idx):
        groups.setdefault(int(k), []).append(i)
    for k, members in groups.items():
        if len(members) >= 2:
            a, b = members[0], members[1]
            ok &= np.allclose(Yhat[a], Yhat[b])
            break

    # training reduces task loss on a stationary world
    m2 = RectificationCodex(8, 6, 10, 2, beta=0.25)
    Xt, Yt = make_world(600, seed=4)
    h = train(m2, Xt, Yt, epochs=60, lr=0.05)
    ok &= h[-1] < h[0]

    # mandate falls when the world drifts away from a fixed canon
    _, mand_home = evaluate(m2, Xt, Yt)
    Xd, Yd = make_world(600, shift=2.4, seed=5)
    _, mand_away = evaluate(m2, Xd, Yd)
    ok &= mand_away < mand_home

    return ok


# =============================================================================
# Section 8 -- Run everything and print verified output
# =============================================================================

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    print("=" * 74)
    print("THE RECTIFICATION CODEX  --  Wang Mang (figure 0114)")
    print("=" * 74)

    print("\n[1] MANDATORY finite-difference gradient check")
    ok_grad, max_rel = gradient_check(verbose=True)
    print(f"    gradient check {'PASSED' if ok_grad else 'FAILED'} "
          f"(max relative error {max_rel:.2e})")
    assert ok_grad, "Gradient check failed."

    print("\n[2] Training the canon on the golden-age world")
    m = RectificationCodex(8, 6, 12, 2, beta=0.25)
    X, Y = make_world(1200, shift=0.0, seed=10)
    train(m, X, Y, epochs=120, lr=0.05, batch=32, log_every=30)

    print("\n[3] Self-tests")
    ok_tests = self_tests()
    print(f"    self-tests {'PASSED' if ok_tests else 'FAILED'}")
    assert ok_tests, "Self-tests failed."

    print("\n[4] The Xin experiment -- fugu (frozen canon) vs reform")
    R = fugu_vs_reform()
    print(f"    golden age      : task-MSE={R['mse_a']:.4f}   mandate={R['mandate_a']:.3f}")
    print(f"    world shifts    : task-MSE={R['mse_shock']:.4f}   mandate={R['mandate_shock']:.3f}   <- canon meets a changed world")
    print(f"    heir FUGU       : task-MSE={R['mse_fugu']:.4f}   mandate={R['mandate_fugu']:.3f}   <- canon frozen; antiquity restored")
    print(f"    heir REFORM     : task-MSE={R['mse_reform']:.4f}   mandate={R['mandate_reform']:.3f}   <- canon revised against reality")
    improve = 100.0 * (R['mse_fugu'] - R['mse_reform']) / R['mse_fugu']
    print(f"    reform recovers {improve:.1f}% of the error the frozen canon could not.")

    print("\n" + "=" * 74)
    print("VERDICT: a righteous but frozen canon cannot govern a world that moves.")
    print("The correction channel -- not the prior -- is what survives contact.")
    print("=" * 74)

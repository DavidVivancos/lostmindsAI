#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 chapter_0071_diogenes_of_sinope_-412.py  —  AUTARKEIA: The Currency Defacer
 Mind #0071 : Diogenes of Sinope (c. 412 - 323 BCE)
  Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 4 Minds 61 - 80 Available on Amazon https://www.amazon.com/dp/B0H71JC95Q
Resume and Interactive Demos at https://artificiology.com/  
Author: David Vivancos · Chapter 0071 · Diogenes of Sinope
================================================================================

WHY THIS ARCHITECTURE (and not a Transformer)
--------------------------------------------------------------------------------
Almost every modern model is ADDITIVE: stack more layers, store more keys,
attend over more context, accumulate more value. Diogenes' entire philosophy is
the opposite gesture. The Delphic oracle told him "parakharattein to nomisma" —
"deface the currency." The Greek word *nomisma* means BOTH a stamped coin AND a
social convention. His life's program was to take every inherited value, scratch
the official stamp off it, and ask: with the stamp gone, is anything of real
worth left underneath? Whatever survives that defacement is *physis* (nature,
true coin). Whatever evaporates was only *nomos* (convention, counterfeit).

So a Diogenean network does not learn by adding capacity. It learns by
SUBTRACTING — by defacing its own inputs until only the load-bearing, universal
features remain. Three of his ideas become three concrete mechanisms:

  1. parakharaxis  (defacement)   -> a per-feature DEFACEMENT GATE g in (0,1).
                                      Each gate scratches the "stamp" off one
                                      input feature. A gate near 0 = defaced
                                      (declared counterfeit); near 1 = kept
                                      (declared true coin).

  2. autarkeia     (self-sufficiency) -> an L1 "barrel" penalty on the sum of
                                      gates. The network is pushed to NEED as
                                      few features as possible — to live in the
                                      barrel, owning a cloak, a staff, and a cup.

  3. kosmopolites  (citizen of the world) -> a COSMOPOLITAN INVARIANCE test.
                                      A "convention" feature is one whose meaning
                                      depends on which city (polis) you happen to
                                      be standing in. We resample those features
                                      as if we had walked to a different city and
                                      demand the SAME answer. Features that fail
                                      this test (their predictive power was local
                                      custom, not nature) are driven toward zero.
                                      This is the parrhesia / "deface convention"
                                      signal that tells the gates WHAT to scratch
                                      off — not blind sparsity, but principled
                                      rejection of the merely conventional.

The result is a classifier whose intelligence is measured by what survives
ablation: it generalizes to a city it has never seen because it refused to trust
anything that was only true in the cities it grew up in.

WHAT THE FILE DOES (run it: `python3 0071_Neuron.py`)
--------------------------------------------------------------------------------
  * Builds a synthetic world with K "poleis" (cities). Each example has
    `true-coin` features (predictive everywhere, the same in every city) and
    `counterfeit` features (predictive only by local custom, and the custom
    flips between cities).
  * Trains the Currency Defacer with backprop written from scratch in NumPy.
  * Runs a finite-difference GRADIENT CHECK on every parameter group (mandatory).
  * Runs self-tests proving:
        - the gates DEFACE the counterfeit features (gate -> 0) while keeping
          the true coin (gate -> 1),
        - the network reaches AUTARKEIA (few active "needs"),
        - it generalizes to an UNSEEN city better than a greedy baseline that
          was allowed to trust convention.
  * Prints a verified report. The report is pasted verbatim into the chapter.

No external dependencies beyond NumPy. Pure from-scratch math; no autograd.
================================================================================
"""

import numpy as np

RNG = np.random.default_rng(412)   # seed = Diogenes' birth year (412 BCE)


# ==============================================================================
# 1. THE WORLD  —  coins that are true everywhere vs. coins true only by custom
# ==============================================================================
def make_world(n, polis_signs, n_true=4, n_conv=8, noise=0.6):
    """
    Build a labelled dataset drawn from one or more 'poleis' (cities).

    Feature layout per example  x = [ true-coin features | counterfeit features ]
        - true-coin features (n_true): drawn so the LABEL is a fixed function of
          them, identical in every city. This is *physis* (nature): universal.
        - counterfeit features (n_conv): correlated with the label ONLY through
          the local custom `s` of the city. s = +1 means "in this city the
          convention points the same way as the truth"; s = -1 means the custom
          is inverted. A model that leans on these features is trusting *nomos*.

    `polis_signs` is a list like [+1, +1, -1] : the custom of each city the
    sample is drawn from. We sample cities uniformly from that list.

    Returns
        X     : (n, n_true + n_conv) float features
        y     : (n,) int labels in {0,1}
        teacher : the fixed linear judge of the true-coin features (for analysis)
        idx_true, idx_conv : column indices of each feature family
    """
    D = n_true + n_conv
    idx_true = np.arange(0, n_true)
    idx_conv = np.arange(n_true, D)

    # The universal judge of worth: a fixed linear teacher on the true features.
    teacher = np.array([1.5, -1.2, 0.9, -0.7])[:n_true]

    X = np.zeros((n, D))
    # --- true-coin features: standard normal; label is their (noisy) verdict ---
    X[:, idx_true] = RNG.standard_normal((n, n_true))
    score = X[:, idx_true] @ teacher + noise * RNG.standard_normal(n)
    y = (score > 0).astype(int)

    # --- assign each example a city, then mint its counterfeit features --------
    signs = RNG.choice(polis_signs, size=n)            # local custom per example
    polarity = (2 * y - 1).astype(float)               # +1 if y==1 else -1
    # counterfeit feature j carries  s * polarity * amp_j  + noise
    amps = np.linspace(1.4, 0.7, n_conv)               # decreasing usefulness
    conv_mean = (signs * polarity)[:, None] * amps[None, :]
    X[:, idx_conv] = conv_mean + 0.9 * RNG.standard_normal((n, n_conv))

    return X, y, teacher, idx_true, idx_conv


# ==============================================================================
# 2. PARAMETERS  —  a tiny MLP plus one defacement gate per input feature
# ==============================================================================
def init_params(D, H, C, seed=7):
    """Xavier-ish init. theta_g starts slightly positive so every coin begins
    'trusted' (gate ~0.62); training must EARN each defacement by scratching."""
    r = np.random.default_rng(seed)
    def xav(shape):
        fan = sum(shape)
        return r.uniform(-np.sqrt(6 / fan), np.sqrt(6 / fan), shape)
    return {
        "W1": xav((D, H)),
        "b1": np.zeros(H),
        "W2": xav((H, C)),
        "b2": np.zeros(C),
        "theta_g": np.full(D, 0.5),    # sigmoid(0.5) ~ 0.62  -> start trusting
    }


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def sigmoid(t):
    return 1.0 / (1.0 + np.exp(-t))


# ==============================================================================
# 3. FORWARD + BACKWARD through one batch  (cross-entropy, written by hand)
# ==============================================================================
def forward_backward(P, X, y):
    """
    Forward pass of the gated MLP and the full analytic gradient of the
    cross-entropy loss on (X, y) w.r.t. every parameter, INCLUDING the gate
    logits theta_g.  Returns (loss, grads, cache).

    The gate acts elementwise on the input:  xg = x * g,  g = sigmoid(theta_g).
    Defacement therefore happens *before* anything else the network does.
    """
    g = sigmoid(P["theta_g"])                 # (D,)  current trust in each coin
    Xg = X * g[None, :]                        # (N,D) gated (partly defaced) input

    z1 = Xg @ P["W1"] + P["b1"]               # (N,H)
    a1 = np.tanh(z1)
    z2 = a1 @ P["W2"] + P["b2"]               # (N,C)
    p = softmax(z2)

    N = X.shape[0]
    loss = -np.mean(np.log(p[np.arange(N), y] + 1e-12))

    # ---- backprop ----
    dz2 = p.copy()
    dz2[np.arange(N), y] -= 1.0
    dz2 /= N                                   # (N,C)

    dW2 = a1.T @ dz2                           # (H,C)
    db2 = dz2.sum(axis=0)                      # (C,)

    da1 = dz2 @ P["W2"].T                      # (N,H)
    dz1 = da1 * (1.0 - a1 ** 2)               # tanh'
    dW1 = Xg.T @ dz1                           # (D,H)
    db1 = dz1.sum(axis=0)                      # (H,)

    dXg = dz1 @ P["W1"].T                      # (N,D)
    dg = (dXg * X).sum(axis=0)                 # (D,)  grad w.r.t. gate values g
    dtheta_g = dg * g * (1.0 - g)             # chain through sigmoid

    grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "theta_g": dtheta_g}
    return loss, grads, p


# ==============================================================================
# 4. TOTAL OBJECTIVE  =  task  +  cosmopolitan invariance  +  autarkeia barrel
# ==============================================================================
def total_loss_and_grads(P, Xc, y, Xk, lam_cosmo, lam_auto, lam_wreg=1e-3):
    """
    Combine the Diogenean pressures into one objective and its gradient.

      L = L_task(Xc, y)                         # judge the example as it stands
        + lam_cosmo * L_task(Xk, y)             # ...and after walking to a new
                                                #    city (Xk = convention cols
                                                #    resampled). Same label must
                                                #    survive -> deface convention.
        + lam_auto  * sum(sigmoid(theta_g))     # own as few 'needs' as possible
        + lam_wreg  * ||W1||^2                  # the 'no free rescaling' clause

    Why the last term matters. Without it the network could shrink a gate toward
    0 and simply blow up the matching row of W1 to compensate — defacing the coin
    on paper while still secretly spending it. The L2 clause couples gate size to
    weight size, so a gate only falls when the feature is genuinely not needed.
    A useful feature settles at gate ~ (2*lam_wreg*gain^2 / lam_auto)^(1/3),
    rising with its usefulness; a worthless one is dragged to ~0.

    Xk is Xc with its CONVENTION columns re-minted for a different/garbled
    custom, so any feature whose worth was merely local pulls the two task terms
    apart and is punished into the ground.
    """
    L_task, g_task, _ = forward_backward(P, Xc, y)
    L_cos, g_cos, _ = forward_backward(P, Xk, y)

    g = sigmoid(P["theta_g"])
    L_auto = lam_auto * g.sum()
    d_auto = lam_auto * g * (1.0 - g)          # d/dtheta of sum(sigmoid)
    L_wreg = lam_wreg * np.sum(P["W1"] ** 2)
    d_wreg = 2.0 * lam_wreg * P["W1"]

    grads = {}
    for k in P:
        grads[k] = g_task[k] + lam_cosmo * g_cos[k]
    grads["theta_g"] = grads["theta_g"] + d_auto
    grads["W1"] = grads["W1"] + d_wreg

    loss = L_task + lam_cosmo * L_cos + L_auto + L_wreg
    return loss, grads


def repole(X, idx_conv, rng):
    """Walk to a different city: shuffle the convention columns across the batch
    so their correlation with the label is destroyed (a foreign custom).  The
    true-coin columns are left untouched — nature does not change address."""
    Xk = X.copy()
    perm = rng.permutation(X.shape[0])
    Xk[:, idx_conv] = X[perm][:, idx_conv]
    return Xk


# ==============================================================================
# 5. GRADIENT CHECK  (finite differences)  —  mandatory in every file
# ==============================================================================
def gradient_check(P, Xc, y, Xk, lam_cosmo, lam_auto, lam_wreg=1e-3, eps=1e-5):
    """Compare analytic grads to numerical grads on a few random coordinates of
    every parameter group.  Returns the worst relative error seen."""
    _, grads = total_loss_and_grads(P, Xc, y, Xk, lam_cosmo, lam_auto, lam_wreg)
    worst = 0.0
    rng = np.random.default_rng(0)
    for name, W in P.items():
        flat = W.ravel()
        n_check = min(8, flat.size)
        coords = rng.choice(flat.size, n_check, replace=False)
        for c in coords:
            orig = flat[c]
            flat[c] = orig + eps
            lp, _ = total_loss_and_grads(P, Xc, y, Xk, lam_cosmo, lam_auto, lam_wreg)
            flat[c] = orig - eps
            lm, _ = total_loss_and_grads(P, Xc, y, Xk, lam_cosmo, lam_auto, lam_wreg)
            flat[c] = orig
            num = (lp - lm) / (2 * eps)
            ana = grads[name].ravel()[c]
            denom = max(1e-12, abs(num) + abs(ana))
            worst = max(worst, abs(num - ana) / denom)
    return worst


# ==============================================================================
# 6. OPTIMISER  —  a compact Adam
# ==============================================================================
class Adam:
    def __init__(self, P, lr=0.02, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in P.items()}
        self.v = {k: np.zeros_like(v) for k, v in P.items()}
        self.t = 0

    def step(self, P, grads):
        self.t += 1
        for k in P:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            P[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


def accuracy(P, X, y):
    _, _, p = forward_backward(P, X, y)
    return float((p.argmax(1) == y).mean())


def train(P, Xtr, ytr, idx_conv, lam_cosmo, lam_auto, lam_wreg=1e-3,
          epochs=600, lr=0.02):
    opt = Adam(P, lr=lr)
    rng = np.random.default_rng(99)
    for _ in range(epochs):
        Xk = repole(Xtr, idx_conv, rng)
        _, grads = total_loss_and_grads(
            P, Xtr, ytr, Xk, lam_cosmo, lam_auto, lam_wreg)
        opt.step(P, grads)
    return P


# ==============================================================================
# 7. EXPERIMENT  —  the Defacer vs. a baseline that is allowed to trust custom
# ==============================================================================
def main():
    print("=" * 74)
    print(" AUTARKEIA — The Currency Defacer   |   Mind #0071 Diogenes of Sinope")
    print("=" * 74)

    n_true, n_conv = 4, 8
    D = n_true + n_conv
    H, C = 16, 2

    # Training cities: custom mostly points 'the right way' (+1), with one
    # inverted city (-1). So convention IS marginally tempting in training...
    train_signs = [+1, +1, +1, -1]
    # ...but the TEST city has the OPPOSITE custom never weighted in training:
    test_signs = [-1]

    Xtr, ytr, teacher, idx_true, idx_conv = make_world(
        4000, train_signs, n_true, n_conv)
    Xte, yte, *_ = make_world(2000, test_signs, n_true, n_conv)

    print(f"\nWorld: {D} features = {n_true} true-coin + {n_conv} counterfeit")
    print(f"       train cities custom={train_signs}   test city custom={test_signs}")
    print(f"       (a model that trusts convention will be MISLED in the test city)\n")

    # ---- gradient check first (mandatory) --------------------------------
    P0 = init_params(D, H, C)
    rng = np.random.default_rng(1)
    Xk0 = repole(Xtr[:64], idx_conv, rng)
    err = gradient_check(P0, Xtr[:64], ytr[:64], Xk0,
                         lam_cosmo=1.0, lam_auto=0.004, lam_wreg=1e-3)
    print(f"[grad-check] worst relative error = {err:.3e}  "
          f"({'PASS' if err < 1e-4 else 'FAIL'}, threshold 1e-4)")
    assert err < 1e-4, "Gradient check failed."

    # ---- (A) the DEFACER : cosmopolitan + autarkeia ----------------------
    Pd = init_params(D, H, C)
    Pd = train(Pd, Xtr, ytr, idx_conv,
               lam_cosmo=1.0, lam_auto=0.004, lam_wreg=1e-3)
    gates = sigmoid(Pd["theta_g"])

    # ---- (B) the BASELINE : greedy, no defacement of convention ----------
    #     (lam_cosmo=0 -> never tests other cities; lam_auto=0 -> keeps every
    #      coin. This is the 'trust the stamp' model.)
    Pb = init_params(D, H, C)
    Pb = train(Pb, Xtr, ytr, idx_conv,
               lam_cosmo=0.0, lam_auto=0.0, lam_wreg=1e-3)

    # ---- report ----------------------------------------------------------
    g_true = gates[idx_true].mean()
    g_conv = gates[idx_conv].mean()
    n_needs = int((gates > 0.5).sum())

    acc_d_tr = accuracy(Pd, Xtr, ytr)
    acc_d_te = accuracy(Pd, Xte, yte)
    acc_b_tr = accuracy(Pb, Xtr, ytr)
    acc_b_te = accuracy(Pb, Xte, yte)

    print("\n--- DEFACEMENT (gate values: 1.0 = true coin kept, 0.0 = defaced) ---")
    print("  true-coin gates :", np.array2string(
        gates[idx_true], precision=2, floatmode="fixed"))
    print("  counterfeit gates:", np.array2string(
        gates[idx_conv], precision=2, floatmode="fixed"))
    print(f"  mean gate  true-coin = {g_true:.3f}   counterfeit = {g_conv:.3f}")
    print(f"  autarkeia: active 'needs' (gate>0.5) = {n_needs} of {D} features")

    print("\n--- GENERALISATION TO AN UNSEEN CITY ---")
    print(f"  Defacer  : train acc {acc_d_tr:.3f}   new-city acc {acc_d_te:.3f}")
    print(f"  Baseline : train acc {acc_b_tr:.3f}   new-city acc {acc_b_te:.3f}")
    print(f"  cosmopolitan advantage = {acc_d_te - acc_b_te:+.3f}")

    # ---- self-tests ------------------------------------------------------
    print("\n--- SELF-TESTS ---")
    checks = []
    checks.append(("gradient check < 1e-4", err < 1e-4))
    checks.append(("true coin kept (mean gate > 0.55)", g_true > 0.55))
    checks.append(("counterfeit defaced (mean gate < 0.35)", g_conv < 0.35))
    checks.append(("autarkeia (needs <= n_true+1)", n_needs <= n_true + 1))
    checks.append(("defacer generalises (new-city acc > 0.80)", acc_d_te > 0.80))
    checks.append(("beats convention-trusting baseline",
                   acc_d_te - acc_b_te > 0.05))
    allok = True
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        allok = allok and ok

    print("\n" + "=" * 74)
    print(" RESULT:", "ALL TESTS PASS — the coin is defaced, the barrel is small,"
          if allok else "SOME TESTS FAILED",
          )
    if allok:
        print("         and what survives travels to every city.")
    print("=" * 74)
    assert allok, "Self-tests failed."


if __name__ == "__main__":
    main()

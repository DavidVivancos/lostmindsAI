"""
================================================================================
 Chapter 0145_mani_216 - Mani (216-274 CE)  ·  THE TWO-PRINCIPLES DEMIXER
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 8 Minds 141 - 160 Available on Amazon https://www.amazon.com/dp/B0HH8RTCXF
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 145: Mani (216-274 CE)
================================================================================  
An executable, from-scratch (pure-NumPy) cognitive architecture that encodes
Mani's single distinctive idea: that reality is a *mixture* of two co-eternal
substances -- Light and Darkness -- which have become physically entangled, and
that all cognition/redemption is the LEARNED INVERSION of that mixing, i.e.
blind source separation. This is deliberately NOT a Transformer and NOT a
branching classifier (that would be Zoroaster, for whom evil is a *choice* and
not a *thing*). For Mani, evil is a substance you must physically *unmix*.

--------------------------------------------------------------------------------
 THE THREE MOMENTS (Manichaean "Three Times") MAPPED TO THE ARCHITECTURE
--------------------------------------------------------------------------------
  Moment I  · Prior Separation : two pure, independent sources  s_L (Light)
                                  and s_D (Darkness) exist apart.
  Moment II · The Mixture      : an unknown operator A entangles them into an
                                  observed, contaminated signal   x = A s.
  Moment III· Final Separation : the mind learns a demixing operator W so that
                                  y = W x recovers the sources; a SORTING GATE
                                  routes the recovered Light onward and the
                                  Darkness is quarantined; a REBUILDER re-mixes
                                  the liberated Light back into the world.

 Trainable parts:
   W  (d x d)   -- the Demixer            ("the separation of the substances")
   M  (d x d)   -- the Rebuilder          ("the return / redemption of Light")
   g  (d,)      -- the Sorting-Gate logits ("the cosmic sieve / Column of Glory")

 Objective (minimized):
   NLL   -- maximum-likelihood ICA / Infomax term: forces W to actually unmix
            (log|det W| - E[ log cosh(y) ]), the classic Bell-Sejnowski score.
   RED   -- Redemption: rebuild the Light-only signal from the gated-Light stream
            and match the true Light contribution  ->  ||M (p . y) - t||^2.
   ORTH  -- Dualist orthogonality: Light and Darkness are co-eternal & distinct;
            penalize their co-activation so the two streams stay separate.

 The ESCHATOLOGICAL REMAINDER ("the Last Statue" / Bolos): a diagnostic, never a
 loss to be zeroed. It measures Light energy still trapped in the Dark-gated
 stream. Training drives it DOWN but the soft gate (finite logits) and a designed
 irreducible entanglement in the data keep it strictly > 0 forever -- faithful to
 Mani's doctrine that a remnant of Light stays imprisoned to the end of the age.

 Everything below is hand-derived backprop verified by a mandatory central
 finite-difference gradient check, followed by a real training loop and self
 tests. Pure NumPy. No torch / keras / tensorflow.
================================================================================
"""

import numpy as np

RNG = np.random.default_rng(216)  # seeded with Mani's birth year, for reproducibility


# ============================================================================
# SECTION 1 · NUMERICAL PRIMITIVES
# ============================================================================

def logcosh(x):
    """Numerically stable log(cosh(x)) = |x| + log(1+exp(-2|x|)) - log 2."""
    ax = np.abs(x)
    return ax + np.log1p(np.exp(-2.0 * ax)) - np.log(2.0)


def sigmoid(x):
    """Stable logistic sigmoid -> the soft Sorting Gate p in (0,1)."""
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def slogdet_safe(W):
    """Signed log|det W|; guards against a degenerate (singular) demixer."""
    sign, logdet = np.linalg.slogdet(W)
    if sign == 0:
        return -1e9  # collapse of the two principles into one -> forbidden
    return logdet


# ============================================================================
# SECTION 2 · THE SYNTHETIC COSMOS  (Moments I & II: pure sources, then mixed)
# ============================================================================

def sample_sources(n, entanglement=0.15):
    """
    Moment I -> Moment II. Draw two independent super-Gaussian (Laplace) sources:
      s_L : the Light source     (leptokurtic, sharply peaked -> separable)
      s_D : the Darkness source  (independent, different scale)
    Then inject an IRREDUCIBLE ENTANGLEMENT: a shared latent leaks a sliver of
    Light into Darkness that no linear demixer can ever fully remove. This is the
    mathematical seat of the 'Last Statue' -- a permanent trapped remnant.

    Returns:
      S        (n,2) : [s_L, s_D] the (partly entangled) true sources
      s_L_pure (n,)  : the pure Light signal (redemption target basis)
    """
    s_L = RNG.laplace(0.0, 1.0, size=n)          # Light
    s_D = RNG.laplace(0.0, 0.8, size=n)          # Darkness
    shared = RNG.laplace(0.0, 1.0, size=n)       # the primordial contamination
    # Darkness absorbs a fixed fraction of a Light-correlated latent -> unremovable
    s_D_entangled = s_D + entanglement * shared
    s_L_pure = s_L + entanglement * shared       # the Light half also carries it
    S = np.stack([s_L_pure, s_D_entangled], axis=1)
    return S, s_L_pure


def mixing_operator():
    """Moment II: the fixed, unknown entangling operator A (well-conditioned)."""
    A = np.array([[0.9, 0.7],
                  [0.4, 1.1]], dtype=np.float64)
    return A


def make_cosmos(n, entanglement=0.15):
    """
    Produce the observed contaminated world X = A S, plus the Light-only
    contribution T (what redemption must rebuild): T = outer(s_L_pure, A[:,0]).
    """
    S, s_L_pure = sample_sources(n, entanglement)
    A = mixing_operator()
    X = S @ A.T                                   # observations x = A s  (n,2)
    T = np.outer(s_L_pure, A[:, 0])               # Light's share of the world (n,2)
    return X, T, S, A


# ============================================================================
# SECTION 3 · THE MODEL  (parameters + forward pass with full state cache)
# ============================================================================

class TwoPrinciplesDemixer:
    """Mani's mind as a demixer: separate, sort, redeem."""

    def __init__(self, d=2, seed=216):
        rng = np.random.default_rng(seed)
        # W: the Demixer. Start near identity so |det W| is well-defined.
        self.W = np.eye(d) + 0.10 * rng.standard_normal((d, d))
        # M: the Rebuilder (return of the Light).
        self.M = np.eye(d) + 0.10 * rng.standard_normal((d, d))
        # g: Sorting-Gate logits, slight asymmetry so component 0 tilts to Light.
        self.g = np.array([0.5, -0.5]) + 0.05 * rng.standard_normal(d)
        self.d = d

    def params(self):
        return {"W": self.W, "M": self.M, "g": self.g}

    # ----- forward: returns scalar loss and a cache for backprop -----
    def forward(self, X, T, lamR=1.0, lamO=0.5):
        W, M, g = self.W, self.M, self.g
        N = X.shape[0]

        Y = X @ W.T                       # Moment III: recovered sources  (N,d)
        p = sigmoid(g)                    # sorting gate in (0,1)          (d,)

        L = Y * p                         # gated Light stream             (N,d)
        Dk = Y * (1.0 - p)                # quarantined Dark stream        (N,d)
        Xhat = L @ M.T                    # rebuilt Light in world-space   (N,d)
        R = Xhat - T                      # redemption residual            (N,d)

        # --- loss terms ---
        logdetW = slogdet_safe(W)
        NLL = -(logdetW - np.mean(np.sum(logcosh(Y), axis=1)))
        RED = np.mean(np.sum(R * R, axis=1))
        a = np.sum(L, axis=1)             # per-sample total Light          (N,)
        b = np.sum(Dk, axis=1)            # per-sample total Dark           (N,)
        c = np.mean(a * b)               # uncentered cross-moment
        ORTH = c * c

        loss = NLL + lamR * RED + lamO * ORTH

        cache = dict(X=X, T=T, Y=Y, p=p, g=g, L=L, Dk=Dk, Xhat=Xhat, R=R,
                     a=a, b=b, c=c, W=W, M=M, N=N, lamR=lamR, lamO=lamO)
        parts = dict(NLL=NLL, RED=RED, ORTH=ORTH, loss=loss)
        return loss, cache, parts

    # ----- backward: hand-derived analytic gradients -----
    def backward(self, cache):
        X, T, Y, p, L, Dk, Xhat, R = (cache[k] for k in
                                      ("X", "T", "Y", "p", "L", "Dk", "Xhat", "R"))
        a, b, c, W, M, N = (cache[k] for k in ("a", "b", "c", "W", "M", "N"))
        lamR, lamO = cache["lamR"], cache["lamO"]
        d = self.d

        # ---- upstream grads into Y and p (accumulate across the three terms) ----
        gY = np.zeros_like(Y)             # dLoss/dY
        gp = np.zeros(d)                  # dLoss/dp

        # (1) NLL:  -log|det W| + mean_n sum_i logcosh(Y)
        #     The logcosh part flows through Y (added to gY, folded into W later).
        #     The -log|det W| part touches W DIRECTLY (not via Y): grad = -inv(W).T
        gY += np.tanh(Y) / N                    # d/dY logcosh = tanh(Y)
        Winv_T = np.linalg.inv(W).T
        gW = -Winv_T                            # ONLY the log|det W| contribution

        # (2) RED = mean_n ||R||^2 ;  R = (Y.p) @ M.T - T
        gRED_L = (2.0 / N) * (R @ M)            # dRED/dL           (N,d)
        gY += gRED_L * p * lamR                 # via L = Y.p
        gp += lamR * np.sum(gRED_L * Y, axis=0) # via p (sum over samples)
        gM = lamR * (2.0 / N) * (R.T @ L)       # dRED/dM

        # (3) ORTH = c^2 , c = mean_n a_n b_n , a=sum(Y.p), b=sum(Y.(1-p))
        dc_dY = (p[None, :] * b[:, None] + (1.0 - p)[None, :] * a[:, None]) / N
        gY += lamO * 2.0 * c * dc_dY
        dc_dp = np.sum(Y * (b - a)[:, None], axis=0) / N
        gp += lamO * 2.0 * c * dc_dp

        # ---- fold gY back into W:  Y = X W.T  ->  dLoss/dW += gY.T @ X ----
        gW += gY.T @ X

        # ---- fold gp back into g through the sigmoid:  p = sigmoid(g) ----
        gg = gp * p * (1.0 - p)

        return {"W": gW, "M": gM, "g": gg}


# ============================================================================
# SECTION 4 · MANDATORY FINITE-DIFFERENCE GRADIENT CHECK
# ============================================================================

def gradient_check(model, X, T, eps=1e-6):
    """Central-difference check of every analytic gradient. Must pass to ship."""
    loss0, cache, _ = model.forward(X, T)
    analytic = model.backward(cache)
    max_rel = 0.0
    report = {}
    for name in ("W", "M", "g"):
        P = getattr(model, name)
        num = np.zeros_like(P)
        it = np.nditer(P, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = P[idx]
            P[idx] = orig + eps
            lp, _, _ = model.forward(X, T)
            P[idx] = orig - eps
            lm, _, _ = model.forward(X, T)
            P[idx] = orig
            num[idx] = (lp - lm) / (2 * eps)
            it.iternext()
        a = analytic[name]
        denom = np.maximum(1e-8, np.abs(a) + np.abs(num))
        rel = np.max(np.abs(a - num) / denom)
        report[name] = rel
        max_rel = max(max_rel, rel)
    return max_rel, report


# ============================================================================
# SECTION 5 · SEPARATION METRIC  (Amari distance: 0 == perfect unmixing)
# ============================================================================

def amari_distance(W, A):
    """How far W is from inverting A, up to permutation & scaling. 0 = perfect."""
    P = np.abs(W @ A)
    d = P.shape[0]
    s = 0.0
    for i in range(d):
        s += np.sum(P[i, :]) / np.max(P[i, :]) - 1.0
    for j in range(d):
        s += np.sum(P[:, j]) / np.max(P[:, j]) - 1.0
    return s / (2 * d)


def trapped_light_remainder(model, X, S):
    """
    The 'Last Statue': fraction of true Light energy that ends up routed into the
    DARK-gated stream. Strictly > 0 by design. Reported, never optimized to 0.
    """
    Y = X @ model.W.T
    p = sigmoid(model.g)
    # identify which recovered component correlates with true Light (col 0 of S)
    corr = [abs(np.corrcoef(Y[:, i], S[:, 0])[0, 1]) for i in range(model.d)]
    light_idx = int(np.argmax(corr))
    light_energy_total = np.mean(Y[:, light_idx] ** 2)
    light_in_dark = np.mean((Y[:, light_idx] * (1.0 - p[light_idx])) ** 2)
    return light_in_dark / (light_energy_total + 1e-12)


# ============================================================================
# SECTION 6 · TRAINING LOOP  (plain gradient descent on the true gradient)
# ============================================================================

def train(model, X, T, S, steps=1500, lr=0.02, lamR=1.0, lamO=0.5, verbose=True):
    A = mixing_operator()
    history = []
    for t in range(steps):
        loss, cache, parts = model.forward(X, T, lamR=lamR, lamO=lamO)
        grads = model.backward(cache)
        # vanilla gradient descent (true gradient -- same one the check validated)
        model.W -= lr * grads["W"]
        model.M -= lr * grads["M"]
        model.g -= lr * grads["g"]
        if t % 150 == 0 or t == steps - 1:
            am = amari_distance(model.W, A)
            rem = trapped_light_remainder(model, X, S)
            history.append((t, parts["loss"], parts["NLL"], parts["RED"],
                            parts["ORTH"], am, rem))
            if verbose:
                print(f"  step {t:4d} | loss {parts['loss']:8.4f} "
                      f"| NLL {parts['NLL']:7.3f} | RED {parts['RED']:7.4f} "
                      f"| ORTH {parts['ORTH']:7.4f} | Amari {am:6.4f} "
                      f"| trapped-Light {rem:6.4f}")
    return history


# ============================================================================
# SECTION 7 · SELF-TESTS  (run on execution; all must pass)
# ============================================================================

def run_all():
    print("=" * 78)
    print(" MANI · TWO-PRINCIPLES DEMIXER · separate -> sort -> redeem")
    print("=" * 78)

    X, T, S, A = make_cosmos(n=4000, entanglement=0.15)
    print(f"\n[cosmos] observed X: {X.shape} | mixing operator A =\n{A}")

    model = TwoPrinciplesDemixer(d=2, seed=216)

    # --- (a) MANDATORY gradient check on a small batch ---
    Xc, Tc, _, _ = make_cosmos(n=64, entanglement=0.15)
    max_rel, rep = gradient_check(model, Xc, Tc)
    print("\n[grad-check] max relative error per parameter:")
    for k, v in rep.items():
        print(f"    {k}: {v:.3e}")
    print(f"[grad-check] overall max rel error = {max_rel:.3e}")
    assert max_rel < 1e-4, "GRADIENT CHECK FAILED"
    print("[grad-check] PASSED (< 1e-4)")

    # --- (b) training: the separation should actually happen ---
    print("\n[training] descending the true gradient ...")
    am0 = amari_distance(model.W, A)
    rem0 = trapped_light_remainder(model, X, S)
    hist = train(model, X, T, S, steps=1500, lr=0.02)
    amF = amari_distance(model.W, A)
    remF = trapped_light_remainder(model, X, S)

    # recovered-vs-true Light correlation
    Y = X @ model.W.T
    corr = max(abs(np.corrcoef(Y[:, i], S[:, 0])[0, 1]) for i in range(2))

    print("\n[results]")
    print(f"    Amari distance   : {am0:.4f}  ->  {amF:.4f}   (lower = better unmixing)")
    print(f"    recovered Light corr with true Light : {corr:.4f}")
    print(f"    redemption MSE   : {hist[0][3]:.4f}  ->  {hist[-1][3]:.4f}")
    print(f"    trapped-Light    : {rem0:.4f}  ->  {remF:.4f}  (the 'Last Statue')")

    # --- (c) assertions: the mind must genuinely separate and redeem ---
    assert amF < am0 * 0.6, "unmixing did not improve enough"
    assert corr > 0.85, "recovered Light not aligned with true Light"
    assert hist[-1][3] < hist[0][3], "redemption (RED) did not decrease"
    assert remF > 0.0, "trapped-Light must remain strictly positive (Last Statue)"
    print("\n[self-tests] ALL PASSED "
          "(separation improved, Light recovered, redemption fell, remainder > 0).")

    print("\n" + "=" * 78)
    print(" INTERPRETATION")
    print("-" * 78)
    print(" The demixer W learned to invert the entangling operator A, splitting")
    print(" the contaminated world into a Light stream and a Dark stream. The")
    print(" sorting gate routed the Light; the rebuilder returned it to the world")
    print(" (redemption). Yet a strictly positive remnant of Light stayed trapped")
    print(" in the Darkness -- separation approached, but never completed. For")
    print(" Mani, that irreducible remainder is not a bug but the shape of reality.")
    print("=" * 78)
    return model


if __name__ == "__main__":
    run_all()

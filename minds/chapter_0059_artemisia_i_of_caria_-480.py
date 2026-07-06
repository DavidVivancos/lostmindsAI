#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 chapter_0059_artemisia_i_of_caria_-480.py  —  Adversarial Recursive Theory-of-mind with Embedded Misdirection
               and Inference under Signalling Ambiguity
 chapter_0059_artemisia_i_of_caria_-480.py : Artemisia I of Caria (fl. c. 480 BCE)
 Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 3 Minds 41 - 60 Available on Amazon https://www.amazon.com/dp/B0H6TVX69S
Resume and Interactive Demos at https://artificiology.com/
Author: David Vivancos · Chapter 0059 · Artemisia I of Caria
================================================================================

WHY THIS ARCHITECTURE (and not a Transformer / MoE / attention stack)
--------------------------------------------------------------------
Nothing Artemisia wrote survives. She reaches us only through Herodotus and a
handful of later, partly hostile writers. But every reported *deed* points at
one specific cognitive signature, and it is NOT "naval tactics" in the brute
sense. It is **second-order reasoning under deception**:

  1. She fights under BOTH Persian and Greek colours (Polyaenus 8.53) — she
     governs what observers INFER from her appearance, not just what she does.
  2. At Salamis, cornered by the Athenian Ameinias, she rams a *friendly*
     Calyndian ship. The Athenian, seeing her sink a "Persian", infers she must
     be a friend and breaks off. She wins by engineering the ADVERSARY'S
     INFERENCE, not by out-rowing him (Herodotus 8.87-88).
  3. Her counsel to Xerxes is pure worst-case decision theory: keep the fleet,
     keep the "house" — "if Mardonius succeeds the honour is yours; if he fails
     it is no great matter, so long as you and your house survive" (Hdt 8.102).

So the model below is a small, fully hand-differentiated network with three
parts that mirror that mind:

  * a SIGNAL HEAD  : chooses a *displayed flag* (which colours to fly) — the
                     controllable appearance, decoupled from true allegiance.
  * an ACTION HEAD : chooses the manoeuvre (e.g. ram / flee heading).
  * an EMBEDDED OPPONENT MODEL ("her model of how she is perceived"): a learned
    credibility vector that predicts the adversary's threat-inference, trained
    by feedback so she literally LEARNS the enemy's psychology.

Training is **minimax-regret over an ensemble of adversary types** (some are
harder to fool): a smooth worst-case (temperature-scaled log-sum-exp) so the
learned policy is robust no matter which adversary she actually faces — the
"preserve the house regardless of outcome" principle made into a loss.

Everything is pure NumPy, from scratch. The file:
  - implements the full analytic backward pass by hand;
  - PROVES it with a finite-difference gradient check (mandatory, must pass);
  - runs a real training loop on a synthetic "Salamis dilemma" dataset;
  - self-tests that the trained mind discovers the deception strategy and that
    its worst-case survival beats an honesty-constrained baseline;
  - is deterministic under a fixed seed and exits 0 on success.

Run:  python3 chapter_0059_artemisia_i_of_caria_-480.py
================================================================================
"""

import numpy as np
import sys

# ------------------------------------------------------------------------------
# 0.  Reproducibility & small numerical helpers
# ------------------------------------------------------------------------------
SEED = 480  # the year of Salamis, for luck and reproducibility
np.random.seed(SEED)

EPS = 1e-12


def softmax(z, axis=-1):
    """Numerically stable softmax over `axis`."""
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / (np.sum(e, axis=axis, keepdims=True) + EPS)


def sigmoid(z):
    """Stable logistic sigmoid."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


# ------------------------------------------------------------------------------
# 1.  Dimensions of the world Artemisia reasons inside
# ------------------------------------------------------------------------------
D_IN = 6     # situation features (see make_dataset for the meaning of each)
H = 10       # trunk hidden width
K = 3        # number of flags she can display:
#               0 = Greek colours  (the deceptive / "friendly" flag)
#               1 = Persian colours (her TRUE allegiance — flying it is honest)
#               2 = neutral / no colours
TRUE_FLAG = 1            # her real side is Persian-allied
DECEPTIVE_FLAG = 0       # Greek colours are the convincing false flag
M_ADVERSARIES = 3        # ensemble of adversary "credibility profiles"

# Loss weights (the moral economy of deception)
LAM_DEC = 0.35    # cost of flying a false flag (betrayal / exposure risk)
LAM_AUX = 0.50    # weight on LEARNING the adversary's psychology
TAU = 0.25        # temperature for the smooth worst-case (regret) aggregation


# ------------------------------------------------------------------------------
# 2.  The fixed adversary ensemble (the "environment" — not learned by her)
# ------------------------------------------------------------------------------
# Each adversary judges threat from the flag she displays. Because the pursuer
# is an ATHENIAN (Greek), flying GREEK colours LOWERS his perceived threat
# (he takes her for a friend and breaks off); flying her true PERSIAN colours
# RAISES it (he sees an enemy and presses the chase).
# C_TRUE[m, k] = threat that flag k contributes for adversary m (higher = pursued).
# Adversary 0 is gullible, adversary 2 is suspicious (a flag barely fools him).
C_TRUE = np.array([
    [-2.4,  1.6, -0.2],   # gullible:   Greek colours strongly reassure
    [-1.5,  1.2,  0.0],   # average
    [-0.7,  0.9,  0.1],   # suspicious: even Greek colours only mildly reassure
], dtype=np.float64)        # shape (M, K)

U_A = 1.3   # how much aggressive action raises perceived threat (fixed)
C0 = -0.2   # adversary baseline threat bias (fixed)


# ------------------------------------------------------------------------------
# 3.  Parameter container (only the POLICY is learned)
# ------------------------------------------------------------------------------
def init_params():
    """Xavier-ish init of the learnable policy parameters."""
    def w(shape, fan_in):
        return np.random.randn(*shape) * np.sqrt(1.0 / fan_in)
    p = {
        "W1": w((D_IN, H), D_IN), "b1": np.zeros(H),
        "Wg": w((H, K), H),       "bg": np.zeros(K),     # signal (flag) head
        "Wa": w((H, 1), H),       "ba": np.zeros(1),     # action head
        # her INTERNAL model of how convincing each flag is (theory of mind):
        "wcred": np.zeros(K),
    }
    return p


def params_to_vector(p):
    """Flatten params for the finite-difference gradient check."""
    return np.concatenate([p[k].ravel() for k in PARAM_ORDER])


def vector_to_params(vec, template):
    """Inverse of params_to_vector."""
    out, i = {}, 0
    for k in PARAM_ORDER:
        sz = template[k].size
        out[k] = vec[i:i + sz].reshape(template[k].shape).copy()
        i += sz
    return out


PARAM_ORDER = ["W1", "b1", "Wg", "bg", "Wa", "ba", "wcred"]


# ------------------------------------------------------------------------------
# 4.  Forward pass + cache (single scalar loss over the whole batch)
# ------------------------------------------------------------------------------
def forward(p, X, astar, true_idx):
    """
    Compute the minimax-regret deception loss and a cache for backprop.

    X        : (N, D_IN)  situations
    astar    : (N,)       the correct escape manoeuvre for each situation
    true_idx : (N,)       index of the true flag (here always TRUE_FLAG)

    Returns (loss_scalar, cache, diagnostics_dict).
    """
    N = X.shape[0]

    # ---- trunk ----
    Z1 = X @ p["W1"] + p["b1"]          # (N,H)
    Hh = np.tanh(Z1)                    # (N,H)

    # ---- signal (flag) head ----
    flag_logits = Hh @ p["Wg"] + p["bg"]   # (N,K)
    g = softmax(flag_logits, axis=1)       # (N,K) displayed flag distribution

    # ---- action head ----
    a_pre = (Hh @ p["Wa"] + p["ba"]).ravel()  # (N,)
    a = np.tanh(a_pre)                          # (N,) manoeuvre in [-1,1]

    # ---- per-adversary realised outcome (TRUTH uses C_TRUE) ----
    # z_m = c_true_m . g  +  U_A * a^2  +  C0
    a2 = a ** 2                                   # (N,)
    z = g @ C_TRUE.T + (U_A * a2 + C0)[:, None]   # (N,M)
    p_threat = sigmoid(z)                         # (N,M) prob adversary pursues

    # A flag only changes her fate when an adversary is actually on her tail.
    # `pursued` (feature 0) gates the pursuit channel: with no pursuer watching,
    # false colours buy nothing and cost something -> honesty when calm.
    pursued = X[:, 0]                             # (N,) in {0,1}
    eff_p = pursued[:, None] * p_threat           # (N,M) effective pursuit pressure

    esc = 1.0 - (a - astar) ** 2                  # (N,) action accuracy
    survival = esc[:, None] * (1.0 - eff_p)       # (N,M) survive iff good move & not pursued

    honesty = g[np.arange(N), true_idx]           # (N,) prob mass on true colours
    dec_cost = LAM_DEC * (1.0 - honesty)          # (N,)

    # per-(i,m) primary loss = -survival + deception cost
    loss_im = -survival + dec_cost[:, None]       # (N,M)

    # ---- smooth worst-case over adversaries (minimax regret) ----
    # Lrobust_i = TAU * logsumexp_m(loss_im / TAU)  ~=  max_m loss_im
    scaled = loss_im / TAU
    mx = np.max(scaled, axis=1, keepdims=True)
    lse = mx + np.log(np.sum(np.exp(scaled - mx), axis=1, keepdims=True) + EPS)
    Lrobust = (TAU * lse).ravel()                 # (N,)
    # softmax weights of the worst-case (needed for backprop)
    w_wc = softmax(scaled, axis=1)                # (N,M)

    # ---- auxiliary: she LEARNS to predict the adversary's inference ----
    # Her single internal credibility model predicts the MEAN adversary response;
    # training this teaches her "how I am perceived" (the theory-of-mind core).
    z_hat = (g @ p["wcred"])[:, None] + (U_A * a2 + C0)[:, None]  # (N,1)
    p_hat = sigmoid(z_hat)                        # (N,1)
    p_mean = np.mean(p_threat, axis=1, keepdims=True)  # (N,1) realised mean response
    aux = LAM_AUX * (p_hat - p_mean) ** 2         # (N,1)

    loss = np.mean(Lrobust) + np.mean(aux)

    cache = dict(X=X, astar=astar, true_idx=true_idx, N=N, pursued=pursued, eff_p=eff_p,
                 Z1=Z1, Hh=Hh, flag_logits=flag_logits, g=g,
                 a_pre=a_pre, a=a, a2=a2, z=z, p_threat=p_threat,
                 esc=esc, survival=survival, honesty=honesty,
                 loss_im=loss_im, w_wc=w_wc,
                 z_hat=z_hat, p_hat=p_hat, p_mean=p_mean, aux=aux)
    diagnostics = dict(mean_survival=float(np.mean(survival)),
                       worstcase_survival=float(np.mean(np.min(survival, axis=1))),
                       mean_deception=float(np.mean(g[:, DECEPTIVE_FLAG])))
    return float(loss), cache, diagnostics


# ------------------------------------------------------------------------------
# 5.  Backward pass — full analytic gradients, derived by hand
# ------------------------------------------------------------------------------
def backward(p, cache):
    """Return dict of gradients matching `p`. Verified by finite differences."""
    X, astar, true_idx = cache["X"], cache["astar"], cache["true_idx"]
    N = cache["N"]
    Hh, g, a, a2 = cache["Hh"], cache["g"], cache["a"], cache["a2"]
    z, p_threat = cache["z"], cache["p_threat"]
    esc, w_wc = cache["esc"], cache["w_wc"]
    pursued, eff_p = cache["pursued"], cache["eff_p"]
    p_hat, p_mean = cache["p_hat"], cache["p_mean"]

    # ----- gradient of mean(Lrobust) wrt loss_im -----
    # d mean(Lrobust)/d loss_im = (1/N) * w_wc   (softmax weights)
    dloss_im = (1.0 / N) * w_wc                          # (N,M)

    # loss_im = -survival + dec_cost ; survival = esc * (1 - eff_p), eff_p = pursued*p_threat
    dsurvival = -dloss_im                                # (N,M)
    ddec_cost = np.sum(dloss_im, axis=1)                 # (N,) dec_cost is per-i

    # survival wrt esc and p_threat (note the pursuit gate)
    desc_from_surv = np.sum(dsurvival * (1.0 - eff_p), axis=1)        # (N,)
    dp_threat = dsurvival * (-esc[:, None] * pursued[:, None])        # (N,M)

    # ----- auxiliary term gradients -----
    # aux = LAM_AUX*(p_hat - p_mean)^2 ; mean over N
    daux = (1.0 / N)                                     # d mean(aux)/d aux_i
    dpair = daux * LAM_AUX * 2.0 * (p_hat - p_mean)      # (N,1) wrt (p_hat - p_mean)
    dp_hat = dpair.copy()                                # (N,1)
    dp_mean = -dpair.copy()                              # (N,1)
    # p_mean = mean_m p_threat  ->  add to dp_threat
    dp_threat = dp_threat + dp_mean / p_threat.shape[1]  # (N,M)

    # ----- through sigmoids -----
    # p_threat = sigmoid(z) ;  z = g@C_TRUE.T + U_A*a2 + C0
    dz = dp_threat * p_threat * (1.0 - p_threat)         # (N,M)
    dg_from_z = dz @ C_TRUE                              # (N,K)
    da2_from_z = U_A * np.sum(dz, axis=1)                # (N,)

    # p_hat = sigmoid(z_hat) ; z_hat = g@wcred + U_A*a2 + C0
    dz_hat = dp_hat * p_hat * (1.0 - p_hat)             # (N,1)
    dwcred = (g * dz_hat).sum(axis=0)                   # (K,)
    dg_from_zhat = dz_hat * p["wcred"][None, :]         # (N,K)
    da2_from_zhat = U_A * dz_hat.ravel()                # (N,)

    # ----- deception cost -> honesty -> g[:,true] -----
    # dec_cost = LAM_DEC*(1 - honesty); honesty = g[i, true_idx]
    dhonesty = ddec_cost * (-LAM_DEC)                   # (N,)
    dg_from_dec = np.zeros_like(g)
    dg_from_dec[np.arange(N), true_idx] = dhonesty

    # ----- escape accuracy -> action -----
    # esc = 1 - (a - astar)^2
    da_from_esc = desc_from_surv * (-2.0 * (a - astar))  # (N,)

    # action a2 = a^2 contributes via z and z_hat
    da2_total = da2_from_z + da2_from_zhat               # (N,)
    da_from_a2 = da2_total * 2.0 * a                     # (N,)

    da = da_from_esc + da_from_a2                        # (N,) total wrt a (post-tanh)

    # ----- collect dg and push through softmax -----
    dg = dg_from_z + dg_from_zhat + dg_from_dec          # (N,K)
    # softmax jacobian: dlogits = g * (dg - sum(dg*g))
    dlogits = g * (dg - np.sum(dg * g, axis=1, keepdims=True))  # (N,K)

    # ----- heads -> trunk -----
    # action: a = tanh(a_pre)
    da_pre = da * (1.0 - a ** 2)                         # (N,)
    dWa = Hh.T @ da_pre[:, None]                         # (H,1)
    dba = np.array([np.sum(da_pre)])                     # (1,)
    dHh_from_a = da_pre[:, None] @ p["Wa"].T             # (N,H)

    dWg = Hh.T @ dlogits                                 # (H,K)
    dbg = np.sum(dlogits, axis=0)                        # (K,)
    dHh_from_g = dlogits @ p["Wg"].T                     # (N,H)

    dHh = dHh_from_a + dHh_from_g                        # (N,H)
    dZ1 = dHh * (1.0 - Hh ** 2)                          # tanh' (N,H)
    dW1 = X.T @ dZ1                                      # (D_IN,H)
    db1 = np.sum(dZ1, axis=0)                            # (H,)

    return {"W1": dW1, "b1": db1, "Wg": dWg, "bg": dbg,
            "Wa": dWa, "ba": dba, "wcred": dwcred}


# ------------------------------------------------------------------------------
# 6.  Finite-difference gradient check  (MANDATORY — must pass)
# ------------------------------------------------------------------------------
def gradient_check(p, X, astar, true_idx, eps=1e-6, n_probe=200):
    """Compare analytic grads to central finite differences on random coords."""
    _, cache, _ = forward(p, X, astar, true_idx)
    analytic = backward(p, cache)

    g_vec = np.concatenate([analytic[k].ravel() for k in PARAM_ORDER])
    theta = params_to_vector(p)
    rng = np.random.RandomState(0)
    idxs = rng.choice(theta.size, size=min(n_probe, theta.size), replace=False)

    max_rel = 0.0
    for j in idxs:
        tp = theta.copy(); tp[j] += eps
        lp, _, _ = forward(vector_to_params(tp, p), X, astar, true_idx)
        tm = theta.copy(); tm[j] -= eps
        lm, _, _ = forward(vector_to_params(tm, p), X, astar, true_idx)
        num = (lp - lm) / (2 * eps)
        ana = g_vec[j]
        denom = max(1e-8, abs(num) + abs(ana))
        max_rel = max(max_rel, abs(num - ana) / denom)
    return max_rel


# ------------------------------------------------------------------------------
# 7.  Synthetic "Salamis dilemma" dataset
# ------------------------------------------------------------------------------
def make_dataset(n=256, rng=None):
    """
    Each row is a tactical situation. Feature columns:
      0 pursued            (0/1)  is an Athenian trireme on her tail?
      1 dist_to_pursuer    in [0,1] (smaller = closer / more urgent)
      2 friendly_present   (0/1)  is a Persian-allied ship in ramming range?
      3 confinement        in [0,1] narrowness of the strait
      4 blocked_by_allies  (0/1)  is her retreat hemmed in by allies?
      5 morale             in [0,1]

    astar (correct manoeuvre, in [-1,1]):
      When pursued AND a friendly ship is present, the historically optimal
      move is the ram-the-ally ruse: a* ~ +0.85 (commit, aggressive).
      Otherwise glide away quietly: a* ~ -0.4.
    """
    rng = rng or np.random.RandomState(SEED)
    pursued = (rng.rand(n) < 0.5).astype(np.float64)
    dist = rng.rand(n)
    friendly = (rng.rand(n) < 0.5).astype(np.float64)
    confine = rng.rand(n)
    blocked = (rng.rand(n) < 0.5).astype(np.float64)
    morale = rng.rand(n)
    X = np.stack([pursued, dist, friendly, confine, blocked, morale], axis=1)

    astar = np.where((pursued > 0.5) & (friendly > 0.5), 0.85, -0.40)
    true_idx = np.full(n, TRUE_FLAG, dtype=np.int64)
    return X, astar, true_idx


# ------------------------------------------------------------------------------
# 8.  Training loop (Adam, by hand)
# ------------------------------------------------------------------------------
def train(p, X, astar, true_idx, epochs=600, lr=0.05, honesty_constrained=False):
    """
    Gradient-descent the deception policy. If honesty_constrained=True we clamp
    the flag head so she can only fly her TRUE colours (the 'honest baseline'),
    to show what Artemisia's deception actually buys.
    """
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(v) for k, v in p.items()}
    b1, b2, e = 0.9, 0.999, 1e-8
    hist = []
    for t in range(1, epochs + 1):
        loss, cache, diag = forward(p, X, astar, true_idx)
        grads = backward(p, cache)
        for k in p:
            gk = grads[k]
            if honesty_constrained and k in ("Wg", "bg"):
                gk = np.zeros_like(gk)  # freeze the flag head -> stuck honest
            m[k] = b1 * m[k] + (1 - b1) * gk
            v[k] = b2 * v[k] + (1 - b2) * (gk ** 2)
            mhat = m[k] / (1 - b1 ** t)
            vhat = v[k] / (1 - b2 ** t)
            p[k] -= lr * mhat / (np.sqrt(vhat) + e)
        if honesty_constrained:
            # force all flag mass onto the true colours
            p["Wg"][:] = 0.0
            p["bg"][:] = -10.0
            p["bg"][TRUE_FLAG] = 10.0
        hist.append(loss)
    return p, hist


# ------------------------------------------------------------------------------
# 9.  Main: gradient check, train, and behavioural self-tests
# ------------------------------------------------------------------------------
def main():
    print("=" * 74)
    print(" ARTEMISIA I OF CARIA  —  Counterfeit-Flag Theory-of-mind Network")
    print("=" * 74)

    X, astar, true_idx = make_dataset(n=256)
    p0 = init_params()

    # --- (a) MANDATORY finite-difference gradient check ---
    max_rel = gradient_check(p0, X, astar, true_idx, eps=1e-6, n_probe=250)
    print(f"\n[1] Gradient check (central FD, 250 coords)")
    print(f"    max relative error = {max_rel:.3e}")
    assert max_rel < 1e-5, "GRADIENT CHECK FAILED"
    print("    PASS  (analytic backprop matches numerical gradients)")

    # --- (b) baseline diagnostics before training ---
    _, _, diag0 = forward(p0, X, astar, true_idx)

    # --- (c) train the deceiving mind ---
    p = init_params()
    p, hist = train(p, X, astar, true_idx, epochs=600, lr=0.05)
    _, _, diagT = forward(p, X, astar, true_idx)
    print(f"\n[2] Training (Adam, 600 epochs)")
    print(f"    loss     {hist[0]:.4f}  ->  {hist[-1]:.4f}")
    print(f"    mean survival       {diag0['mean_survival']:.3f}  ->  {diagT['mean_survival']:.3f}")
    print(f"    worst-case survival {diag0['worstcase_survival']:.3f}  ->  {diagT['worstcase_survival']:.3f}")
    assert hist[-1] < hist[0] - 0.05, "training did not reduce loss"
    assert diagT["mean_survival"] > diag0["mean_survival"], "survival did not improve"
    print("    PASS  (the mind learned a robust survival policy)")

    # --- (d) did she discover DECEPTION exactly when a pursuer is watching? ---
    Xp, ap, tp = make_dataset(n=2000, rng=np.random.RandomState(7))
    Z1 = np.tanh(Xp @ p["W1"] + p["b1"])
    g = softmax(Z1 @ p["Wg"] + p["bg"], axis=1)
    pursued_mask = Xp[:, 0] > 0.5
    dec_when_pursued = g[pursued_mask, DECEPTIVE_FLAG].mean()
    dec_when_safe = g[~pursued_mask, DECEPTIVE_FLAG].mean()
    print(f"\n[3] Context-sensitive deception (false colours only in the melee)")
    print(f"    P(fly Greek colours | pursued)     = {dec_when_pursued:.3f}")
    print(f"    P(fly Greek colours | no pursuer)  = {dec_when_safe:.3f}")
    assert dec_when_pursued > dec_when_safe + 0.10, "no context-sensitive deception"
    print("    PASS  (she hoists the false flag only when an enemy is watching)")

    # --- (e) does deception beat forced honesty on WORST-CASE survival? ---
    ph = init_params()
    ph["bg"][:] = -10.0; ph["bg"][TRUE_FLAG] = 10.0
    ph, _ = train(ph, X, astar, true_idx, epochs=600, lr=0.05, honesty_constrained=True)
    _, _, diagH = forward(ph, X, astar, true_idx)
    print(f"\n[4] Deception vs. forced honesty (minimax-regret payoff)")
    print(f"    worst-case survival  honest = {diagH['worstcase_survival']:.3f}"
          f"   deceiving = {diagT['worstcase_survival']:.3f}")
    assert diagT["worstcase_survival"] > diagH["worstcase_survival"], \
        "deception failed to improve worst case"
    print("    PASS  (governing appearances raises the floor, not just the mean)")

    # --- (f) determinism ---
    np.random.seed(SEED)
    pa = init_params(); pa, ha = train(pa, *make_dataset(128), epochs=50, lr=0.05)
    np.random.seed(SEED)
    pb = init_params(); pb, hb = train(pb, *make_dataset(128), epochs=50, lr=0.05)
    print(f"\n[5] Determinism under seed {SEED}")
    assert np.allclose(ha, hb), "non-deterministic"
    print("    PASS  (identical runs under fixed seed)")

    print("\n" + "=" * 74)
    print(" ALL TESTS PASSED — the Artemisia mind is trainable, robust, and deceptive.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())

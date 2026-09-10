#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Chapter 0162 - Boethius (c. 477-524 CE)
The Boethian Consolation Network (BCN)
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 9 Minds 161 - 180 Available on Amazon https://www.amazon.com/dp/B0HJC1QF59
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 0162_boethius_477 - Boethius (c. 477-524 CE)
================================================================================    

A from-scratch, pure-NumPy cognitive architecture that encodes the *distinctive*
cognitive signature of Anicius Manlius Severinus Boethius -- deliberately NOT a
Transformer, and not a generic Neoplatonist "order-on-chaos" engine.

Boethius left three ideas that are his and no one else's in quite this shape:

  (1) MUSICA AS NUMBER  (De institutione musica).
      Music is not a sensation but a *ratio* -- proportion made audible.
      Consonance = simple integer ratio; dissonance = complexity. So the world
      is represented in a harmonic/proportional basis, and a soul "out of tune"
      is consoled by being re-tuned toward simple ratios.
      --> we regularize the abstract universal toward CONSONANCE (few clean
          components); and timesteps are encoded as a small harmonic series
          (positions as ratios).

  (2) UNIVERSALS BY ABSTRACTION  (2nd commentary on Porphyry's Isagoge).
      Following Alexander of Aphrodisias: there are no universal *things* in the
      world, yet there is a real, non-arbitrary basis for forming general
      concepts -- the mind ABSTRACTS what is common to many particulars,
      considering-apart what is not apart in reality.
      --> the GENUS (universal) is the PHASE-INVARIANT harmonic content pooled
          across all the particular instants:  u = mean_t (z_t^2).  What each
          instant adds -- its phase -- is the DIFFERENTIA. Crucially u is EXACTLY
          invariant to any reordering of the moments, i.e. to Fortune turning
          her wheel: the genus is what survives contingency.

  (3) THE ETERNAL PRESENT  (Consolation, Book V).
      Eternity is "the whole, simultaneous and perfect possession of boundless
      life" -- totum simul. God does not FORE-see the future step by step; He
      sees the whole trajectory AT ONCE, as present. Infallible knowledge does
      not NECESSITATE the free act.
      --> two readouts of the same latent:
            * a MORTAL head -- strictly local & causal: from z_t alone it
              predicts x[t+1] (the conditioned, step-by-step view);
            * an ETERNAL head -- non-successive: a learned time-mixing "gaze"
              M lets every output moment depend on EVERY input moment at once,
              and reconstructs the whole trajectory.
          A PROVIDENCE term trains the eternal view to be CONSISTENT with the
          mortal process through a stop-gradient, so the eternal view never
          PUSHES (necessitates) the mortal one. Foreknowledge without
          determinism -- mechanized.

  (4) CONSOLATION AS PROSIMETRUM.
      The Consolation alternates verse (metrum, which steadies) and prose (which
      argues/ascends). We train by literally ALTERNATING:
        * a "metrum" step steadies the whole picture (eternal recon + consonance),
        * a "prose" step advances the argument (mortal prediction + providence).

Toy world -- a "Wheel of Fortune": each trajectory is a mixture of a few
simple-ratio sinusoids (the eternal harmonic form) plus a slow linear drift
(contingency). The net must abstract the invariant universal, reconstruct the
whole from the eternal view, predict the next step from the mortal view, and
keep the two views providentially consistent.

Hand-written forward + backward NumPy, a mandatory finite-difference gradient
check, a real training loop, and self-tests.  Run: python chapter_0162_boethius_477.py
================================================================================
"""

import numpy as np


# =============================================================================
# SECTION 0. Configuration
# =============================================================================
class Config:
    def __init__(self, T=16, in_dim=1, H=48, D=8, P=8, K=6, seed=162):
        self.T = T; self.in_dim = in_dim; self.H = H; self.D = D; self.P = P
        self.K = K                       # mortal head's causal window (past taps)
        self.U = (P // 2) * D            # size of the abstracted universal (genus)
        self.seed = seed
        self.w_sim = 1.0      # eternal (totum simul) reconstruction
        self.w_seq = 1.0      # mortal next-step prediction
        self.w_harm = 0.05   # consonance (proportional sparsity of the universal)
        self.w_prov = 0.5     # providence (eternal<->mortal consistency, stop-grad)
        self.harm_delta = 1e-6


# =============================================================================
# SECTION 1. Harmonic positional code (musica: position encoded as ratio)
# =============================================================================
def harmonic_positions(T, P):
    """
    Encode each instant as a point in a HARMONIC SERIES: sin/cos at integer
    multiples m = 1..P/2 of the fundamental 2*pi/T. This is Boethius' own
    instinct -- a position in time is a ratio, not a bare integer.

    Integer frequencies matter mechanically, not just poetically: they make the
    power spectrum computed against this basis EXACTLY invariant to a circular
    shift of the moments. Fortune may turn the wheel; the genus does not move.
    Shape (T, P), P even.
    """
    assert P % 2 == 0, "P must be even (sin/cos pairs)"
    t = np.arange(T)[:, None]                       # (T,1)
    m = np.arange(1, P // 2 + 1)[None, :]           # (1,P/2) harmonics 1,2,3,...
    ang = 2.0 * np.pi * m * t / T
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=1)   # (T,P)


# =============================================================================
# SECTION 2. Parameter initialization
# =============================================================================
def init_params(cfg, rng):
    def W(a, b):
        return rng.standard_normal((a, b)) * np.sqrt(2.0 / (a + b))
    H, D, P, I, T, K, U = cfg.H, cfg.D, cfg.P, cfg.in_dim, cfg.T, cfg.K, cfg.U
    return {
        "W1": W(I, H), "b1": np.zeros(H),           # encoder: x -> hidden
        "W2": W(H, D), "b2": np.zeros(D),           # hidden -> per-instant latent
        "W3": W(K * D, H), "b3": np.zeros(H),       # mortal head (causal K-tap window)
        "W4": W(H, I), "b4": np.zeros(I),           # mortal head -> x[t+1]
        "W5": W(D + P + U, H), "b5": np.zeros(H),   # eternal head: [gaze; pos; genus]
        "W6": W(H, I), "b6": np.zeros(I),           # eternal head -> x[t]
        "M": np.eye(T) + 0.01 * rng.standard_normal((T, T)),   # the eternal gaze
    }


# =============================================================================
# SECTION 3. Forward pass
# =============================================================================
def forward(params, X, POS):
    p = params
    B, T, I = X.shape
    D = p["W2"].shape[1]
    P = POS.shape[1]
    Kd = p["W3"].shape[0] // D           # number of causal taps
    nh = P // 2                          # number of harmonics

    # --- encoder: each instant becomes a particular -----------------------
    a1 = X @ p["W1"] + p["b1"]; h1 = np.tanh(a1)
    z = h1 @ p["W2"] + p["b2"]                       # (B,T,D) the particulars

    # --- ABSTRACTION OF THE UNIVERSAL (the genus) -------------------------
    # Project the particulars, considered TOGETHER, onto the harmonic series,
    # then take power -- discarding phase. Phase is the differentia (WHEN a
    # thing happens); power is the genus (WHAT harmonic kind it is). This
    # "considers apart what is not apart in reality" and, because the basis is
    # an integer harmonic series, it is EXACTLY unchanged when Fortune turns
    # the wheel (any circular reordering of the moments).
    C = np.einsum('tp,btd->bpd', POS, z) / T        # (B,P,D) harmonic coefficients
    Cs, Cc = C[:, :nh, :], C[:, nh:, :]             # sin / cos parts per harmonic
    Upow = Cs**2 + Cc**2                            # (B,nh,D) phase-invariant power
    u = Upow.reshape(B, nh * D)                     # (B,U) THE UNIVERSAL

    # --- MORTAL head: strictly causal K-tap window ------------------------
    # It may know its past, never its future. Given what has happened, the next
    # step follows -- "conditioned necessity", nothing more.
    lags = [z]
    for k in range(1, Kd):
        lags.append(np.concatenate([np.zeros((B, k, D)), z[:, :-k, :]], axis=1))
    Zcaus = np.concatenate(lags, axis=2)            # (B,T,K*D)
    a3 = Zcaus @ p["W3"] + p["b3"]; h3 = np.tanh(a3)
    xseq = h3 @ p["W4"] + p["b4"]                   # (B,T,I) predicts x[t+1] at t

    # --- ETERNAL head: totum simul ----------------------------------------
    # A learned time-mixing gaze M lets every moment be read in the light of
    # every other moment AT ONCE -- not successively. Plus the harmonic index
    # and the genus.
    Zmix = np.einsum('ts,bsd->btd', p["M"], z)      # (B,T,D)
    posB = np.broadcast_to(POS, (B, T, P))
    uB = np.broadcast_to(u[:, None, :], (B, T, u.shape[1]))
    zc = np.concatenate([Zmix, posB, uB], axis=2)   # (B,T,D+P+U)
    a5 = zc @ p["W5"] + p["b5"]; h5 = np.tanh(a5)
    xsim = h5 @ p["W6"] + p["b6"]                   # (B,T,I) the whole, at once

    cache = dict(X=X, POS=POS, a1=a1, h1=h1, z=z, C=C, Cs=Cs, Cc=Cc, u=u,
                 Zcaus=Zcaus, a3=a3, h3=h3, xseq=xseq,
                 Zmix=Zmix, zc=zc, a5=a5, h5=h5, xsim=xsim, nh=nh, Kd=Kd)
    return xseq, xsim, u, cache


# =============================================================================
# SECTION 4. Consonance penalty (smooth, proportional)  -- musica
# =============================================================================
def consonance(u, delta):
    """
    Consonance = FEW SIMPLE RATIOS SOUNDING AT ONCE.

    u is harmonic POWER (Cs^2 + Cc^2). The quantity to economize is the
    AMPLITUDE, sqrt(u) -- the size of the vibration, the length of string that
    must be admitted. Penalizing amplitude is a group-lasso over each harmonic's
    sin/cos pair: it drives whole harmonics to silence, leaving a spectrum built
    from a few clean components.

    (Penalizing the power instead would be ridge on the coefficients, which
    SPREADS energy evenly -- the exact opposite of consonance. The dissonant
    chord is the one where everything sounds a little.)

    Smoothed by delta so the gradient is exact at zero and the finite-difference
    check stays valid.
    """
    val = np.mean(np.sqrt(u + delta) - np.sqrt(delta))
    grad = (0.5 / np.sqrt(u + delta)) / u.size
    return val, grad


# =============================================================================
# SECTION 5. Loss + full analytic gradients
# =============================================================================
def loss_and_grads(params, X, POS, cfg, weights, prov_target):
    """
    weights     : lets the prosimetrum alternation switch terms on/off.
    prov_target : (B,T-1) CONSTANT -- the DETACHED mortal prediction. Held fixed,
                  so the stop-gradient is exact: the eternal view is trained to
                  agree with the temporal process, but never pushes it.
                  Foreknowledge without necessitation.
    """
    p = params
    xseq, xsim, u, c = forward(params, X, POS)
    B, T, I = X.shape
    D = cfg.D; P = cfg.P; nh = c["nh"]; Kd = c["Kd"]

    # ---------------- loss terms ----------------
    r_sim = (xsim - X);                       L_sim = np.mean(r_sim**2)
    r_seq = (xseq[:, :-1, :] - X[:, 1:, :]);  L_seq = np.mean(r_seq**2)
    L_harm, g_harm_u = consonance(u, cfg.harm_delta)
    r_prov = (xsim[:, 1:, 0] - prov_target);  L_prov = np.mean(r_prov**2)

    total = (weights["w_sim"] * L_sim + weights["w_seq"] * L_seq +
             weights["w_harm"] * L_harm + weights["w_prov"] * L_prov)

    g = {k: np.zeros_like(v) for k, v in p.items()}

    # ---------------- eternal head ----------------
    gxsim = weights["w_sim"] * 2.0 * r_sim / r_sim.size
    gxsim[:, 1:, 0] += weights["w_prov"] * 2.0 * r_prov / r_prov.size

    g["W6"] += np.einsum('bth,bti->hi', c["h5"], gxsim)
    g["b6"] += gxsim.sum(axis=(0, 1))
    dh5 = gxsim @ p["W6"].T; da5 = dh5 * (1.0 - c["h5"]**2)
    g["W5"] += np.einsum('btk,bth->kh', c["zc"], da5)
    g["b5"] += da5.sum(axis=(0, 1))
    dzc = da5 @ p["W5"].T                              # (B,T,D+P+U)
    dZmix = dzc[:, :, 0:D]
    du_from_head = dzc[:, :, D + P:].sum(axis=1)       # (B,U) u was broadcast over t

    # the eternal gaze M
    g["M"] += np.einsum('btd,bsd->ts', dZmix, c["z"])
    dz_from_mix = np.einsum('ts,btd->bsd', p["M"], dZmix)

    # ---------------- mortal head (causal K-tap) ----------------
    gxseq = np.zeros_like(xseq)
    gxseq[:, :-1, :] += weights["w_seq"] * 2.0 * r_seq / r_seq.size
    g["W4"] += np.einsum('bth,bti->hi', c["h3"], gxseq)
    g["b4"] += gxseq.sum(axis=(0, 1))
    dh3 = gxseq @ p["W4"].T; da3 = dh3 * (1.0 - c["h3"]**2)
    g["W3"] += np.einsum('btk,bth->kh', c["Zcaus"], da3)
    g["b3"] += da3.sum(axis=(0, 1))
    dZcaus = da3 @ p["W3"].T                           # (B,T,K*D)
    dz_from_seq = np.zeros_like(c["z"])
    for k in range(Kd):
        blk = dZcaus[:, :, k * D:(k + 1) * D]          # lag-k slot: holds z[t-k]
        if k == 0:
            dz_from_seq += blk
        else:
            dz_from_seq[:, :T - k, :] += blk[:, k:, :]

    # ---------------- universal (genus): power spectrum ----------------
    du = du_from_head + weights["w_harm"] * g_harm_u   # (B,U)
    dUpow = du.reshape(B, nh, D)                       # (B,nh,D)
    dCs = 2.0 * c["Cs"] * dUpow                        # power -> sin coeff
    dCc = 2.0 * c["Cc"] * dUpow                        # power -> cos coeff
    dC = np.concatenate([dCs, dCc], axis=1)            # (B,P,D)
    # C = einsum('tp,btd->bpd', POS, z)/T  ->  dz[b,t,d] = (1/T) sum_p POS[t,p] dC[b,p,d]
    dz_from_u = np.einsum('tp,bpd->btd', POS, dC) / T

    # ---------------- encoder ----------------
    dz = dz_from_seq + dz_from_mix + dz_from_u
    g["W2"] += np.einsum('bth,btd->hd', c["h1"], dz)
    g["b2"] += dz.sum(axis=(0, 1))
    dh1 = dz @ p["W2"].T; da1 = dh1 * (1.0 - c["h1"]**2)
    g["W1"] += np.einsum('bti,bth->ih', X, da1)
    g["b1"] += da1.sum(axis=(0, 1))

    parts = dict(L_sim=L_sim, L_seq=L_seq, L_harm=L_harm, L_prov=L_prov)
    return total, g, parts


# =============================================================================
# SECTION 6. Synthetic "Wheel of Fortune" data
# =============================================================================
def make_batch(B, cfg, rng):
    T = cfg.T
    t = np.arange(T) / T
    X = np.zeros((B, T))
    base = rng.uniform(1.0, 3.0, size=B)
    for b in range(B):
        n = rng.integers(2, 4)
        sig = np.zeros(T)
        for _ in range(n):
            ratio = rng.choice([1.0, 2.0, 1.5, 4/3, 3.0])   # consonant ratios
            amp = rng.uniform(0.4, 1.0)
            phase = rng.uniform(0, 2*np.pi)
            sig += amp * np.sin(2*np.pi * base[b]*ratio * t + phase)
        sig += rng.uniform(-0.6, 0.6) * (t - 0.5)            # contingency / drift
        sig += 0.02 * rng.standard_normal(T)
        X[b] = sig
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-6)
    return X[:, :, None]


# =============================================================================
# SECTION 7. Finite-difference gradient check (MANDATORY)
# =============================================================================
def gradient_check():
    print("=" * 72)
    print("GRADIENT CHECK (central finite differences on the full objective)")
    print("=" * 72)
    cfg = Config(T=6, H=4, D=2, P=4, K=3, seed=7)
    rng = np.random.default_rng(cfg.seed)
    params = init_params(cfg, rng)
    POS = harmonic_positions(cfg.T, cfg.P)
    X = make_batch(2, cfg, rng)
    weights = dict(w_sim=1.0, w_seq=1.0, w_harm=0.003, w_prov=0.5)

    xseq0, _, _, _ = forward(params, X, POS)
    prov_target = xseq0[:, :-1, 0].copy()          # frozen -> exact stop-grad

    def total_loss(pp):
        L, _, _ = loss_and_grads(pp, X, POS, cfg, weights, prov_target)
        return L

    _, grads, _ = loss_and_grads(params, X, POS, cfg, weights, prov_target)

    eps = 1e-5; max_rel = 0.0; worst = None
    for name in params:
        flat = params[name].ravel(); gflat = grads[name].ravel()
        idxs = np.linspace(0, flat.size - 1, min(12, flat.size)).astype(int)
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps; Lp = total_loss(params)
            flat[i] = orig - eps; Lm = total_loss(params)
            flat[i] = orig
            num = (Lp - Lm) / (2 * eps); ana = gflat[i]
            rel = abs(num - ana) / max(1e-12, abs(num) + abs(ana))
            if rel > max_rel:
                max_rel, worst = rel, (name, i, ana, num)
    print(f"  checked tensors : {list(params.keys())}")
    print(f"  worst entry     : {worst[0]}[{worst[1]}] "
          f"analytic={worst[2]:+.3e} numeric={worst[3]:+.3e}")
    print(f"  max rel error   : {max_rel:.3e}")
    ok = max_rel < 1e-5
    print(f"  RESULT          : {'PASS' if ok else 'FAIL'} (threshold 1e-5)")
    return ok


# =============================================================================
# SECTION 8. Adam optimizer
# =============================================================================
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


# =============================================================================
# SECTION 9. Training loop (prosimetrum: alternate metrum & prose)
# =============================================================================
def train(cfg, steps=3000, batch=32, verbose=True):
    rng = np.random.default_rng(cfg.seed)
    params = init_params(cfg, rng)
    POS = harmonic_positions(cfg.T, cfg.P)
    opt = Adam(params, lr=3e-3)
    Xeval = make_batch(256, cfg, rng)
    all_w = dict(w_sim=cfg.w_sim, w_seq=cfg.w_seq, w_harm=cfg.w_harm, w_prov=cfg.w_prov)

    def eval_total(pp):
        xs0, _, _, _ = forward(pp, Xeval, POS)
        pt = xs0[:, :-1, 0].copy()
        L, _, parts = loss_and_grads(pp, Xeval, POS, cfg, all_w, pt)
        return L, parts

    L0, _ = eval_total(params)
    if verbose:
        print("\n" + "=" * 72)
        print("TRAINING  (alternating metrum/prose updates)")
        print("=" * 72)
        print(f"  step {0:5d} | total {L0:.4f}")

    for s in range(1, steps + 1):
        X = make_batch(batch, cfg, rng)
        xseq0, _, _, _ = forward(params, X, POS)
        prov_target = xseq0[:, :-1, 0].copy()
        if s % 2 == 0:
            w = dict(w_sim=cfg.w_sim, w_seq=0.0, w_harm=cfg.w_harm, w_prov=0.0)  # METRUM
        else:
            w = dict(w_sim=0.0, w_seq=cfg.w_seq, w_harm=0.0, w_prov=cfg.w_prov)  # PROSE
        _, grads, _ = loss_and_grads(params, X, POS, cfg, w, prov_target)
        opt.step(params, grads)
        if verbose and s % 500 == 0:
            L, parts = eval_total(params)
            print(f"  step {s:5d} | total {L:.4f} | "
                  f"sim {parts['L_sim']:.3f} seq {parts['L_seq']:.3f} "
                  f"harm {parts['L_harm']:.3f} prov {parts['L_prov']:.3f}")

    Lf, partsf = eval_total(params)
    return params, POS, L0, Lf, partsf


# =============================================================================
# SECTION 10. Self-tests
# =============================================================================
def self_tests(cfg, params, POS):
    print("\n" + "=" * 72)
    print("SELF-TESTS")
    print("=" * 72)
    rng = np.random.default_rng(cfg.seed + 1)
    X = make_batch(256, cfg, rng)
    xseq, xsim, u, _ = forward(params, X, POS)

    ss_res = np.sum((xsim[:, :, 0] - X[:, :, 0])**2)
    ss_tot = np.sum((X[:, :, 0] - X[:, :, 0].mean())**2)
    r2_sim = 1 - ss_res / ss_tot
    print(f"  (a) eternal (totum-simul) reconstruction R^2 = {r2_sim:.3f}  "
          f"[{'PASS' if r2_sim > 0.5 else 'FAIL'}]")

    pred_err = np.mean((xseq[:, :-1, 0] - X[:, 1:, 0])**2)
    pers_err = np.mean((X[:, :-1, 0] - X[:, 1:, 0])**2)
    print(f"  (b) mortal next-step MSE {pred_err:.3f} vs persistence "
          f"{pers_err:.3f}  [{'PASS' if pred_err < pers_err else 'FAIL'}]")

    corr = np.corrcoef(xsim[:, 1:, 0].ravel(), xseq[:, :-1, 0].ravel())[0, 1]
    print(f"  (c) providence eternal<->mortal corr = {corr:.3f}  "
          f"[{'PASS' if corr > 0.8 else 'FAIL'}]")

    shift = rng.integers(1, cfg.T, size=X.shape[0])
    Xroll = np.stack([np.roll(X[b, :, 0], shift[b]) for b in range(X.shape[0])])[:, :, None]
    _, _, u_roll, _ = forward(params, Xroll, POS)
    inv_err = np.mean(np.abs(u - u_roll)) / (np.mean(np.abs(u)) + 1e-9)
    between = np.mean(np.std(u, axis=0))
    print(f"  (d) genus invariance rel-change under wheel-turn = {inv_err:.2e}, "
          f"between-traj spread = {between:.3f}  "
          f"[{'PASS' if (inv_err < 1e-6 and between > 0.05) else 'FAIL'}]")

    energy = np.mean(u**2, axis=0)
    pr = (energy.sum()**2) / (np.sum(energy**2) + 1e-12)
    consonant = pr < 0.5 * cfg.U        # the genus rests on FEW clean harmonics
    print(f"  (e) consonance participation ratio = {pr:.2f} of {cfg.U} dims "
          f"[{'PASS' if consonant else 'FAIL'}]")

    # (f) NON-NECESSITATION -- the decisive one.
    # Boethius' Book V claim, mechanized: the eternal view foreknows the temporal
    # process, yet contributes NOTHING to it. We run the providence term ALONE
    # and confirm its gradient into the mortal head (W3,W4) is exactly zero:
    # knowledge without causal push.
    Xp = make_batch(16, cfg, rng)
    xs0, _, _, _ = forward(params, Xp, POS)
    pt = xs0[:, :-1, 0].copy()
    w_prov_only = dict(w_sim=0.0, w_seq=0.0, w_harm=0.0, w_prov=1.0)
    _, gprov, _ = loss_and_grads(params, Xp, POS, cfg, w_prov_only, pt)
    push = max(np.abs(gprov["W3"]).max(), np.abs(gprov["W4"]).max())
    pull = np.abs(gprov["W6"]).max()      # it DOES shape the eternal head
    print(f"  (f) non-necessitation: providence push into mortal head = {push:.2e} "
          f"(eternal head {pull:.2e})  [{'PASS' if (push == 0.0 and pull > 0) else 'FAIL'}]")

    passed = (r2_sim > 0.5 and pred_err < pers_err and corr > 0.8
              and inv_err < 1e-6 and between > 0.05 and consonant
              and push == 0.0 and pull > 0)
    print(f"\n  ALL SELF-TESTS: {'PASS' if passed else 'FAIL'}")
    return passed


# =============================================================================
# SECTION 11. Main
# =============================================================================
def main():
    np.set_printoptions(precision=4, suppress=True)
    print(__doc__.split("Hand-written forward")[0])

    gc_ok = gradient_check()

    cfg = Config()
    params, POS, L0, Lf, partsf = train(cfg, steps=3000, batch=32, verbose=True)
    print(f"\n  initial total loss : {L0:.4f}")
    print(f"  final   total loss : {Lf:.4f}   ({100*(1-Lf/L0):.1f}% reduction)")
    train_ok = Lf < 0.4 * L0

    tests_ok = self_tests(cfg, params, POS)

    print("\n" + "=" * 72)
    print("VERIFICATION SUMMARY")
    print("=" * 72)
    print(f"  gradient check : {'PASS' if gc_ok else 'FAIL'}")
    print(f"  loss reduced   : {'PASS' if train_ok else 'FAIL'} (final < 40% of initial)")
    print(f"  self-tests     : {'PASS' if tests_ok else 'FAIL'}")
    overall = gc_ok and train_ok and tests_ok
    print(f"  OVERALL        : {'PASS' if overall else 'FAIL'}")
    print("=" * 72)
    return overall


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)

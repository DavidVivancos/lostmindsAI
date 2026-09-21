#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Chapter 202: Al-Mansur (c. 714 – 775 CE), 2nd Abbasid caliph, founder of Baghdad
Encyclopedia of Lost Minds: Echoes on AI
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 11 Minds 201 - 220 Available on Amazon https://www.amazon.com/dp/B0HKF7TRFF
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0202_al_mansur_714 - Al-Mansur (c. 714 – 775 CE), 2nd Abbasid caliph, founder of Baghdad
================================================================================  

THE ROUND CITY ENGINE
---------------------
A from-scratch (pure-NumPy) cognitive architecture that encodes al-Mansur's
distinctive signature rather than a generic "governance" template.

Al-Mansur did not leave a philosophy of mind in his own hand. What he left is a
DIAGRAM of one, drawn in brick: the perfectly round Madinat al-Salam (762 CE),
and an administrative machine documented by al-Tabari. Read together, three
coupled mechanisms recur — and this model instantiates exactly those three:

  (1) RADIAL FUNNEL  — the round city with four gates and four straight roads,
      no lateral avenues. Every message routes province -> centre; there is no
      province-to-province edge. A star topology, chosen so that the maximum
      distance from any point to the centre is minimised.  ->  a hub that
      aggregates only through spokes; corruption cannot cascade sideways.

  (2) REDUNDANT VERIFICATION — the *barid*. Al-Tabari records that postmasters
      in every district wrote to al-Mansur twice daily reporting the SAME
      quantities (the price of grain, the treasury inflow, the qadi's rulings).
      Two independent channels reporting one number is a consistency check: when
      the channels AGREE the centre acts; when they DISAGREE the centre withholds
      action and falls back to a neutral prior. The green-dome weathervane
      (a horseman whose lance turned toward the quarter from which enemies would
      come) is the same idea rendered as an instrument: a central sensor that
      only points once the signal is confirmed.  ->  a learned trust gate on
      inter-channel disagreement that blends the acted decision with a prior.

  (3) CONSERVED LEDGER — "Abu al-Dawaniq", father of farthings. He counted every
      coin and left a full treasury. Resource is a CONSERVED quantity: the sum of
      what is allocated must equal the fixed budget, never more.  ->  the output
      head is a budget * softmax, so total allocation is exactly the budget to
      machine precision. Nothing is created from nothing.

TASK (synthetic but principled)
  N provinces each have a true, hidden "need" s_p. The centre never sees s_p. It
  sees TWO noisy courier reports per province. An adversary may corrupt ONE
  channel of a few provinces (a lying postmaster inflating need to seize budget).
  The centre must allocate a FIXED budget B across the provinces to match the
  true-need-proportional target — which requires (2) to distrust spoofed reports,
  (1) to keep that corruption from spreading, and (3) to keep the books balanced.

WHY THIS IS NOT A TRANSFORMER
  There is no attention over stored keys, no token stream. The inductive biases
  are geometric (a fixed star graph), epistemic (a redundancy gate that can
  REFUSE to act), and conservative (a hard simplex on the budget). Those biases
  are the mind; the weights merely tune them.

Everything below is hand-derived reverse-mode autodiff. A finite-difference
gradient check is MANDATORY and runs at import/main. Then a real training loop,
then self-tests that isolate each of the three mechanisms.
================================================================================
"""

import numpy as np

# Reproducible: 192 = this figure's id.
RNG = np.random.default_rng(192)


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


# ----------------------------------------------------------------------------
# parameter container
# ----------------------------------------------------------------------------
def init_params(H=6, Dh=8, seed=192):
    """Initialise all trainable weights. Small Xavier-ish scales."""
    r = np.random.default_rng(seed)
    def g(*shape, s=0.4):
        return r.standard_normal(shape) * s
    p = {
        # verification gate: features [mean, disagreement^2] -> hidden -> trust logit
        "Wg": g(H, 2),          # (H,2)
        "bg": np.zeros(H),      # (H,)
        "vg": g(H),             # (H,)
        "ct": np.array(0.0),    # scalar trust bias (start neutral)
        # radial hub: each spoke contributes (trust * report) * W_in
        "W_in": g(Dh),          # (Dh,)
        "b_h": np.zeros(Dh),    # (Dh,)
        # allocation readout over [hub ; own report]
        "w_out": g(Dh + 1),     # (Dh+1,)
        # neutral fallback logit used when a report is NOT verified (t->0)
        "p0": np.array(0.0),    # scalar
    }
    return p


PARAM_KEYS = ["Wg", "bg", "vg", "ct", "W_in", "b_h", "w_out", "p0"]


def pack(params):
    return np.concatenate([np.ravel(params[k]) for k in PARAM_KEYS])


def unpack(vec, template):
    out, i = {}, 0
    for k in PARAM_KEYS:
        sz = template[k].size
        out[k] = np.array(vec[i:i + sz]).reshape(template[k].shape)
        i += sz
    return out


# ----------------------------------------------------------------------------
# forward + backward for a SINGLE sample (N provinces)
# ----------------------------------------------------------------------------
def forward_sample(params, x0, x1, a_tgt, B, return_cache=False):
    """
    x0, x1 : (N,) the two barid channel reports per province
    a_tgt  : (N,) target allocation (sums to B)
    B      : scalar budget
    Returns loss (and cache for backward).
    """
    Wg, bg, vg, ct = params["Wg"], params["bg"], params["vg"], float(params["ct"])
    W_in, b_h = params["W_in"], params["b_h"]
    w_out, p0 = params["w_out"], float(params["p0"])
    Dh = W_in.shape[0]

    # (2) verification gate ---------------------------------------------------
    m = 0.5 * (x0 + x1)                 # fused mean report            (N,)
    dd = x0 - x1                        # inter-channel disagreement   (N,)
    f = np.stack([m, dd ** 2], axis=1)  # gate features                (N,2)
    gpre = f @ Wg.T + bg                # (N,H)
    gg = np.tanh(gpre)                  # (N,H)
    tl = gg @ vg + ct                   # trust logit                  (N,)
    t = sigmoid(tl)                     # trust in [0,1]               (N,)
    r = m                               # fused report used downstream (N,)

    # (1) radial hub ----------------------------------------------------------
    # The round city HEARS every district (the barid always delivers): the hub is
    # a trust-free mean over spokes. Trust is spent later, on the decision to ACT.
    rbar = float(np.mean(r))            # centre's pooled report        scalar
    hub_pre = rbar * W_in + b_h         # (Dh,)
    hub = np.tanh(hub_pre)              # central belief                (Dh,)

    # (3) conserved allocation head ------------------------------------------
    N = x0.shape[0]
    z = np.concatenate([np.broadcast_to(hub, (N, Dh)), r[:, None]], axis=1)  # (N,Dh+1)
    logit_raw = z @ w_out              # acted logit                  (N,)
    # verified -> act on report; unverified -> neutral prior p0
    logit = t * logit_raw + (1.0 - t) * p0
    sm = softmax(logit)
    a = B * sm
    diff = a - a_tgt
    loss = 0.5 * float(np.sum(diff ** 2))

    if not return_cache:
        return loss
    cache = dict(x0=x0, x1=x1, m=m, dd=dd, f=f, gpre=gpre, gg=gg, tl=tl, t=t,
                 r=r, rbar=rbar, hub_pre=hub_pre, hub=hub, z=z, logit_raw=logit_raw,
                 logit=logit, sm=sm, a=a, diff=diff, B=B, Dh=Dh, N=N,
                 w_out=w_out, W_in=W_in, p0=p0, vg=vg, Wg=Wg)
    return loss, cache


def backward_sample(params, cache):
    """Hand-derived gradients for a single sample. Returns dict like params."""
    t, r, rbar = cache["t"], cache["r"], cache["rbar"]
    hub, hub_pre = cache["hub"], cache["hub_pre"]
    z, sm, a, diff, B = cache["z"], cache["sm"], cache["a"], cache["diff"], cache["B"]
    logit_raw, p0 = cache["logit_raw"], cache["p0"]
    gg, tl, f, gpre = cache["gg"], cache["tl"], cache["f"], cache["gpre"]
    dd, m = cache["dd"], cache["m"]
    W_in, w_out, vg, Wg = cache["W_in"], cache["w_out"], cache["vg"], cache["Wg"]
    Dh, N = cache["Dh"], cache["N"]

    # loss -> allocation -> softmax
    da = diff                                   # dL/da
    dsm = B * da                                # dL/dsm
    dsm_dot = float(np.sum(sm * dsm))
    dlogit = sm * (dsm - dsm_dot)               # dL/dlogit          (N,)

    # logit = t*logit_raw + (1-t)*p0
    dt = dlogit * (logit_raw - p0)              # trust path from head (N,)
    dlogit_raw = dlogit * t                     # (N,)
    dp0 = float(np.sum(dlogit * (1.0 - t)))     # scalar

    # logit_raw = z @ w_out
    dw_out = z.T @ dlogit_raw                    # (Dh+1,)
    dz = dlogit_raw[:, None] * w_out[None, :]    # (N,Dh+1)
    dhub = dz[:, :Dh].sum(axis=0)                # hub shared across provinces (Dh,)
    dr = dz[:, Dh].copy()                        # report term in readout      (N,)

    # hub = tanh(hub_pre);  hub_pre = rbar * W_in + b_h,  rbar = mean_p(r_p)
    dhub_pre = dhub * (1.0 - hub ** 2)           # (Dh,)
    db_h = dhub_pre.copy()
    dW_in = rbar * dhub_pre                       # (Dh,)
    drbar = float(dhub_pre @ W_in)               # dL/drbar
    dr += drbar / N                              # each r_p feeds rbar equally

    # t = sigmoid(tl)
    dtl = dt * t * (1.0 - t)                      # (N,)
    dct = float(np.sum(dtl))
    dvg = gg.T @ dtl                              # (H,)
    dgg = dtl[:, None] * vg[None, :]              # (N,H)
    dgpre = dgg * (1.0 - gg ** 2)                 # (N,H)
    dbg = dgpre.sum(axis=0)                       # (H,)
    dWg = dgpre.T @ f                             # (H,2)
    df = dgpre @ Wg                               # (N,2)

    # f = [m, dd^2]; r = m  (dd depends only on data)
    dm = dr + df[:, 0]                            # m enters readout/hub via r, and gate feature
    # (dd^2 term and dd have no parameters; nothing further to accumulate)

    grads = {
        "Wg": dWg, "bg": dbg, "vg": dvg, "ct": np.array(dct),
        "W_in": dW_in, "b_h": db_h, "w_out": dw_out, "p0": np.array(dp0),
    }
    return grads


def loss_and_grads(params, batch, B):
    """Average loss and gradients over a batch of samples."""
    X0, X1, ATGT = batch
    M = X0.shape[0]
    total = 0.0
    gsum = {k: np.zeros_like(params[k], dtype=float) for k in PARAM_KEYS}
    for i in range(M):
        loss, cache = forward_sample(params, X0[i], X1[i], ATGT[i], B, return_cache=True)
        total += loss
        g = backward_sample(params, cache)
        for k in PARAM_KEYS:
            gsum[k] = gsum[k] + g[k]
    for k in PARAM_KEYS:
        gsum[k] = gsum[k] / M
    return total / M, gsum


# ----------------------------------------------------------------------------
# synthetic barid data
# ----------------------------------------------------------------------------
def make_batch(M, N=8, B=1.0, alpha=5.0, sigma=0.05, n_corrupt=2,
               spoof=1.4, rng=RNG):
    """
    Returns (X0, X1, ATGT), truth arrays for evaluation.
    true need s ~ U(0.1,0.9). Honest channels = s + small noise.
    Corrupted provinces: ONE channel spoofed high (inflated need) -> channels
    strongly disagree there. Target allocation is true-need proportional.
    """
    X0 = np.zeros((M, N)); X1 = np.zeros((M, N))
    ATGT = np.zeros((M, N)); S = np.zeros((M, N))
    CORR = np.zeros((M, N), dtype=bool)
    for i in range(M):
        s = rng.uniform(0.1, 0.9, size=N)
        c0 = s + rng.normal(0, sigma, size=N)
        c1 = s + rng.normal(0, sigma, size=N)
        idx = rng.choice(N, size=n_corrupt, replace=False)
        for j in idx:
            # spoof ONE channel to inflate perceived need
            if rng.random() < 0.5:
                c0[j] = min(1.5, s[j] + spoof)
            else:
                c1[j] = min(1.5, s[j] + spoof)
        X0[i] = c0; X1[i] = c1
        ATGT[i] = B * softmax(alpha * s)
        S[i] = s; CORR[i] = False; CORR[i, idx] = True
    return (X0, X1, ATGT), S, CORR


# ----------------------------------------------------------------------------
# gradient check (finite differences) — MANDATORY
# ----------------------------------------------------------------------------
def gradient_check(seed=0, N=8, B=1.0, eps=1e-6):
    params = init_params(seed=7)
    (X0, X1, ATGT), _, _ = make_batch(3, N=N, B=B, rng=np.random.default_rng(seed))
    batch = (X0, X1, ATGT)

    _, analytic = loss_and_grads(params, batch, B)
    ga = pack(analytic)

    theta = pack(params)
    gn = np.zeros_like(theta)
    for k in range(theta.size):
        tp = theta.copy(); tp[k] += eps
        tm = theta.copy(); tm[k] -= eps
        lp, _ = loss_and_grads(unpack(tp, params), batch, B)
        lm, _ = loss_and_grads(unpack(tm, params), batch, B)
        gn[k] = (lp - lm) / (2 * eps)

    denom = np.maximum(1e-12, np.abs(ga) + np.abs(gn))
    rel = np.abs(ga - gn) / denom
    return float(rel.max()), ga, gn


# ----------------------------------------------------------------------------
# training loop
# ----------------------------------------------------------------------------
def train(params, train_batch, val_batch, B, epochs=400, lr=0.3, verbose=True):
    hist = []
    for ep in range(epochs):
        loss, grads = loss_and_grads(params, train_batch, B)
        for k in PARAM_KEYS:
            params[k] = params[k] - lr * grads[k]
        if ep % max(1, epochs // 8) == 0 or ep == epochs - 1:
            vloss, _ = loss_and_grads(params, val_batch, B)
            hist.append((ep, loss, vloss))
            if verbose:
                print(f"  epoch {ep:4d} | train {loss:.5f} | val {vloss:.5f}")
    return params, hist


# ----------------------------------------------------------------------------
# evaluation utilities used by the self-tests
# ----------------------------------------------------------------------------
def predict_alloc(params, x0, x1, B, force_trust_all=False):
    """Run forward; optionally clamp the trust gate to 1 (the 'no barid' ablation)."""
    if not force_trust_all:
        _, c = forward_sample(params, x0, x1, np.zeros_like(x0), B, return_cache=True)
        return c["a"], c["t"]
    # ablation: t := 1 everywhere (act on every report, no verification)
    m = 0.5 * (x0 + x1); r = m; N = x0.shape[0]
    W_in, b_h = params["W_in"], params["b_h"]
    Dh = W_in.shape[0]
    hub = np.tanh(float(np.mean(r)) * W_in + b_h)
    z = np.concatenate([np.broadcast_to(hub, (N, Dh)), r[:, None]], axis=1)
    logit = z @ params["w_out"]           # t=1 => logit = logit_raw
    a = B * softmax(logit)
    return a, np.ones(N)


def alloc_error(params, batch, B, force_trust_all=False):
    X0, X1, ATGT = batch
    errs = []
    for i in range(X0.shape[0]):
        a, _ = predict_alloc(params, X0[i], X1[i], B, force_trust_all)
        errs.append(np.mean((a - ATGT[i]) ** 2))
    return float(np.mean(errs))


# ----------------------------------------------------------------------------
# main: gradient check -> train -> self-tests
# ----------------------------------------------------------------------------
def main():
    B = 1.0
    N = 8
    print("=" * 74)
    print("THE ROUND CITY ENGINE — al-Mansur (chapter 192)")
    print("=" * 74)

    # --- 1. gradient check --------------------------------------------------
    print("\n[1] Finite-difference gradient check (all parameters)")
    max_rel, ga, gn = gradient_check()
    print(f"    max relative error = {max_rel:.3e}")
    ok_grad = max_rel < 1e-5
    print(f"    gradient check: {'PASS' if ok_grad else 'FAIL'}  (threshold 1e-5)")

    # --- 2. train -----------------------------------------------------------
    print("\n[2] Training the Round City Engine on barid reports")
    rng = np.random.default_rng(2024)
    train_batch, S_tr, C_tr = make_batch(160, N=N, B=B, rng=rng)
    val_batch, S_va, C_va = make_batch(96, N=N, B=B, rng=rng)
    params = init_params(seed=192)
    l0, _ = loss_and_grads(params, train_batch, B)
    params, hist = train(params, train_batch, val_batch, B, epochs=1200, lr=0.5)
    lf, _ = loss_and_grads(params, val_batch, B)

    # --- 3. self-tests ------------------------------------------------------
    print("\n[3] Self-tests (each isolates one mechanism)")

    # (a) VERIFICATION: trained gate vs 'trust-all' ablation on corrupted data
    err_verify = alloc_error(params, val_batch, B, force_trust_all=False)
    err_trustall = alloc_error(params, val_batch, B, force_trust_all=True)
    ratio = err_trustall / max(1e-12, err_verify)
    print(f"    (a) barid verification: error WITH gate   = {err_verify:.5f}")
    print(f"                            error trust-all    = {err_trustall:.5f}")
    print(f"                            trust-all is {ratio:.2f}x worse "
          f"-> {'PASS' if ratio > 1.3 else 'FAIL'}")

    # gate actually distrusts the disagreeing (corrupted) provinces?
    ts_corrupt, ts_honest = [], []
    for i in range(val_batch[0].shape[0]):
        _, t = predict_alloc(params, val_batch[0][i], val_batch[1][i], B)
        ts_corrupt.extend(t[C_va[i]].tolist())
        ts_honest.extend(t[~C_va[i]].tolist())
    tc, th = float(np.mean(ts_corrupt)), float(np.mean(ts_honest))
    print(f"        mean trust  corrupted={tc:.3f}  honest={th:.3f}  "
          f"-> {'PASS' if tc < th else 'FAIL'} (centre distrusts disagreement)")

    # (b) CONSERVATION: total allocation == budget to machine precision
    worst = 0.0
    for i in range(val_batch[0].shape[0]):
        a, _ = predict_alloc(params, val_batch[0][i], val_batch[1][i], B)
        worst = max(worst, abs(a.sum() - B))
    print(f"    (b) conserved ledger: max |sum(alloc) - B| = {worst:.2e} "
          f"-> {'PASS' if worst < 1e-9 else 'FAIL'}")

    # (c) RADIAL vs LATERAL containment.
    #     Because the budget is conserved (mechanism 3), corrupting one province
    #     always shrinks the others through the simplex -- that coupling is not a
    #     topology leak. The true test of the ROUND-CITY star graph is whether a
    #     corrupted district distorts the centre's RELATIVE estimate of the OTHER
    #     districts. We therefore compare the conservation-invariant conditional
    #     distribution over the untouched provinces, before vs after corruption.
    #     Star topology: province 0 reaches others only through the shared hub
    #     (a common-mode mean) -> cancels in the conditional -> ~0 leak.
    #     Lateral 'gossip': province 0 is injected into its neighbours' logits
    #     directly -> non-uniform -> real leak.
    def cond(a):
        rest = a[1:]
        return rest / max(1e-12, rest.sum())
    rng2 = np.random.default_rng(5)
    (bx0, bx1, _), _, _ = make_batch(1, N=N, B=B, n_corrupt=0, rng=rng2)
    x0, x1 = bx0[0].copy(), bx1[0].copy()
    a_base, _ = predict_alloc(params, x0, x1, B)
    x0c = x0.copy(); x1c = x1.copy()
    x0c[0] = 1.5; x1c[0] = 1.5          # province 0 corrupted (both channels agree-high)
    a_corr, _ = predict_alloc(params, x0c, x1c, B)
    star_leak = float(np.mean(np.abs(cond(a_corr) - cond(a_base))))

    def lateral_alloc(x0, x1):          # a genuine neighbour-gossip topology
        m = 0.5 * (x0 + x1)
        nb = np.array([0.5 * m[k] + 0.25 * m[(k - 1) % N] + 0.25 * m[(k + 1) % N]
                       for k in range(N)])
        return B * softmax(5.0 * nb)
    la_base = lateral_alloc(x0, x1)
    la_corr = lateral_alloc(x0c, x1c)
    lateral_leak = float(np.mean(np.abs(cond(la_corr) - cond(la_base))))
    print(f"    (c) radial containment: star leak onto other districts = {star_leak:.2e}")
    print(f"                            lateral-gossip leak            = {lateral_leak:.2e}")
    print(f"        -> {'PASS' if star_leak < 0.2 * lateral_leak else 'FAIL'} "
          f"(round-city topology confines damage)")

    # (d) LEARNING: val loss dropped
    print(f"    (d) training: val loss {l0:.5f} -> {lf:.5f} "
          f"-> {'PASS' if lf < 0.5 * l0 else 'FAIL'}")

    print("\n[summary]")
    checks = {
        "gradient_check": ok_grad,
        "verification_beats_trustall": ratio > 1.3,
        "gate_distrusts_disagreement": tc < th,
        "conservation_exact": worst < 1e-9,
        "radial_containment": star_leak < 0.2 * lateral_leak,
        "training_converged": lf < 0.5 * l0,
    }
    for k, v in checks.items():
        print(f"    {'PASS' if v else 'FAIL'}  {k}")
    allok = all(checks.values())
    print(f"\n    ALL TESTS {'PASSED' if allok else 'FAILED'}")
    return allok


if __name__ == "__main__":
    main()

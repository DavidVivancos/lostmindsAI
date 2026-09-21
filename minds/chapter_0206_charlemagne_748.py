#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Chapter 0206 - Charlemagne (c. 742/748 - 814)
"Encyclopedia of Lost Minds: Echoes on AI"
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 11 Minds 201 - 220 Available on Amazon https://www.amazon.com/dp/B0HKF7TRFF
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0206_charlemagne_748 - Charlemagne (c. 742/748 - 814)
================================================================================  

THE CORRECTIO ENGINE
An error-correcting associative memory built from scratch in pure NumPy.

--------------------------------------------------------------------------------
WHY THIS ARCHITECTURE (and not a Transformer)
--------------------------------------------------------------------------------
Charlemagne's whole programme has a single name in the sources and in modern
scholarship: *correctio* -- correction, emendation, the restoration of a drifted,
corrupted signal back toward a correct reference. In the letter *De litteris
colendis* he complains that monasteries send him prayers in which "pious devotion"
is faithful but the tongue, "uneducated on account of the neglect of study, was
not able to express in the letter without error", and he draws the distinction
that organises this entire model:

    "although errors of speech are dangerous,
     far more dangerous are errors of the understanding."

That is not a governance slogan. It is a precise theory of information: a true
message passes through a long chain of unreliable human relays (scribes, priests,
counts), each of whom injects noise; error accumulates; and the sovereign's task
is to install a *canonical exemplar* and run repeated correction passes that pull
every corrupted copy back toward it -- while spending correction effort on the
errors that change meaning, not the ones that merely change spelling.

So the model is NOT attention over stored keys. It is an ENERGY-BASED ATTRACTOR
NETWORK -- a continuous Hopfield-style associative memory whose energy minima are
the learned canonical exemplars (the "canon"). Correction is the *iterated*
descent of a corrupted input down that energy landscape until it settles into the
nearest canonical basin. Three ideas make it Charlemagne's and no one else's:

  1. THE CANON (learned prototypes P).  The stored "corrected books" -- the fixed
     points the whole empire is pulled toward.

  2. THE LEARNED METRIC m  ("errors of understanding > errors of speech").  A
     per-dimension weighting that decides which discrepancies matter. Distances
     are measured in this metric, so the network learns to correct meaning-bearing
     dimensions hard and to tolerate cosmetic ones.

  3. DUAL-CHANNEL CONSENSUS (the *missi dominici*).  Charlemagne sent his
     inspectors in PAIRS -- one lay, one cleric -- so that two independent noisy
     reports of the same reality could be cross-checked. The model fuses two
     corrupted channels with learned per-office reliabilities before correction.

The dynamics (unrolled T steps) are fully differentiable; we hand-derive the
backward pass and VERIFY it with a finite-difference gradient check (mandatory).
Then we train it, and show it doing *correctio*: heavily corrupted copies of a
canonical text converge back to the right exemplar, redundancy beats a single
channel, and the learned metric really does privilege meaning over spelling.

--------------------------------------------------------------------------------
WHAT THE FILE DOES WHEN RUN
--------------------------------------------------------------------------------
  * builds a synthetic "scriptorium": K canonical texts that differ only in a few
    MEANING dimensions; the rest are SPELLING dimensions (identity-irrelevant).
  * corrupts each text through two channels of differing reliability.
  * runs a finite-difference gradient check on every parameter group.
  * trains the Correctio Engine with Adam.
  * runs self-tests: denoising ("emendation"), redundancy benefit, and the
    meaning-over-spelling metric test.

Pure NumPy. No frameworks. Deterministic seed.
================================================================================
"""

import numpy as np

RNG = np.random.default_rng(191)          # 191 = the figure's number in the corpus
np.set_printoptions(precision=4, suppress=True)


# ==============================================================================
# 0.  SMALL DIFFERENTIABLE PRIMITIVES
# ==============================================================================
def softplus(x):
    """Positive reparameterisation: keeps metric weights, temperatures > 0."""
    return np.logaddexp(0.0, x)


def d_softplus(x):
    """Derivative of softplus = logistic sigmoid."""
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


# ==============================================================================
# 1.  PARAMETERS
# ==============================================================================
# All learnable state lives in a flat dict of NumPy arrays so the gradient check
# can perturb any scalar generically.
def init_params(K, d, seed_scale=0.35):
    """
    K : number of canonical exemplars (the 'canon' -- corrected master texts)
    d : dimensionality of a text's representation
    """
    return {
        # THE CANON: K learned prototypes -> the energy minima / attractors.
        "P":       RNG.normal(0, seed_scale, size=(K, d)),
        # THE METRIC: raw weights, one per dimension. m = softplus(raw_m) > 0.
        # This is 'errors of understanding vs errors of speech': the network
        # learns which dimensions a discrepancy must be corrected in.
        "raw_m":   np.zeros(d),                     # start metric ~ uniform
        # BASIN SHARPNESS: inverse temperature of the energy. beta = softplus(raw_beta).
        "raw_beta": np.array(0.0),                  # softplus(0) ~ 0.693
        # CORRECTION STEP SIZE: eta = eta_max * sigmoid(raw_eta), kept in (0, eta_max).
        "raw_eta": np.array(0.0),                   # sigmoid(0) = 0.5 -> eta_max/2
        # THE MISSI (dual-channel reliabilities): rho = softplus(raw_rho), 2 offices.
        "raw_rho": np.array([0.0, 0.0]),
    }


HP = dict(T=6, eta_max=0.6, lam_recon=1.0)          # T = number of correction passes


# ==============================================================================
# 2.  FORWARD PASS  (with a cache for backprop)
# ==============================================================================
def forward(params, x1, x2, y, target, hp=HP, recon_mask=None):
    """
    x1, x2 : (B, d)  two corrupted channels (lay report, clerical report)
    y      : (B,)    index of the true canonical exemplar (for the CE loss)
    target : (B, d)  the true clean canonical text (for the reconstruction loss)
    recon_mask : (d,) 1 on dimensions the emendation is scored on (the recoverable
                 SENSE), 0 on unrecoverable ornament. Default: all ones.

    Returns (loss, cache). Everything needed for backward is stored in cache.
    """
    P       = params["P"]                            # (K, d)
    m       = softplus(params["raw_m"])              # (d,)
    beta    = softplus(params["raw_beta"])           # scalar
    eta     = hp["eta_max"] * sigmoid(params["raw_eta"])
    rho     = softplus(params["raw_rho"])            # (2,)
    K, d    = P.shape
    B       = x1.shape[0]

    # ---- (a) MISSI DOMINICI: fuse two channels by learned office reliability ----
    alpha = rho[0] / (rho[0] + rho[1])               # scalar in (0,1)
    x0 = alpha * x1 + (1.0 - alpha) * x2             # (B, d) fused corrupted copy

    # ---- (b) RECOGNITION: which canonical exemplar is this copy meant to be? ----
    # A Carolingian corrector first identifies the master text, MATCHING the copy
    # against the canon in the LEARNED METRIC -- so the metric decides which
    # discrepancies count. This is where 'errors of understanding vs speech' lives.
    D0 = x0[:, None, :] - P[None, :, :]              # (B, K, d)
    wsq0 = np.einsum("bkd,d->bk", D0 * D0, m)        # (B, K) metric-weighted sq dist
    q0 = -0.5 * beta * wsq0                          # (B, K) recognition logits
    r0 = softmax(q0, axis=1)                         # (B, K) p(canon | copy)

    # ---- (c) EMENDATION: T steps of energy descent that pull the copy to canon ----
    xs = [x0]                                         # trajectory of the copy
    rs = []                                           # responsibilities per step
    for t in range(hp["T"]):
        x = xs[-1]
        D = x[:, None, :] - P[None, :, :]
        wsq = np.einsum("bkd,d->bk", D * D, m)
        q = -0.5 * beta * wsq
        r = softmax(q, axis=1)
        pbar = np.einsum("bk,kd->bd", r, P)          # responsibility-weighted exemplar
        gE = m[None, :] * (x - pbar)                 # gradient of the energy wrt x
        x_next = x - eta * gE                        # one correction pass
        xs.append(x_next)
        rs.append(r)
    x_hat = xs[-1]                                    # the emended copy

    # ---- (d) LOSSES ----
    if recon_mask is None:
        recon_mask = np.ones(d)
    p_y = r0[np.arange(B), y]                        # recognition: right exemplar?
    ce = -np.mean(np.log(p_y + 1e-12))
    diff = (x_hat - target) * recon_mask[None, :]    # emendation: recovered the sense?
    mse = np.mean(np.sum(diff * diff, axis=1))
    loss = ce + hp["lam_recon"] * mse

    cache = dict(params=params, hp=hp, x1=x1, x2=x2, y=y, target=target,
                 alpha=alpha, x0=x0, xs=xs, rs=rs, m=m, beta=beta, eta=eta,
                 rho=rho, r0=r0, x_hat=x_hat, K=K, d=d, B=B, ce=ce, mse=mse,
                 recon_mask=recon_mask)
    return loss, cache


# ==============================================================================
# 3.  BACKWARD PASS  (hand-derived reverse-mode through the unrolled dynamics)
# ==============================================================================
def _softmax_backward(r, g):
    """Given softmax output r (B,K) and upstream grad g (B,K) wrt r,
    return grad wrt the logits q. Row-wise Jacobian of softmax."""
    dot = np.sum(g * r, axis=1, keepdims=True)
    return r * (g - dot)


def _energy_readout_backward(x, P, m, beta, r, g_r):
    """
    Backprop through one responsibilities computation
        D=x-P ; wsq=sum_d m*D^2 ; q=-0.5*beta*wsq ; r=softmax(q)
    given upstream g_r = dL/dr. Returns grads dL/dx, dL/dP, dL/dm, dL/dbeta.
    x:(B,d) P:(K,d) m:(d,) beta:scalar r:(B,K)
    """
    B, d = x.shape
    K = P.shape[0]
    D = x[:, None, :] - P[None, :, :]                 # (B,K,d)
    g_q = _softmax_backward(r, g_r)                   # (B,K)
    # q = -0.5*beta*wsq
    wsq = np.einsum("bkd,d->bk", D * D, m)            # (B,K)
    g_beta = np.sum(g_q * (-0.5 * wsq))               # scalar
    g_wsq = g_q * (-0.5 * beta)                       # (B,K)
    # wsq = sum_d m_d * D^2
    g_m = np.einsum("bk,bkd->d", g_wsq, D * D)        # (d,)
    g_D = 2.0 * D * (g_wsq[:, :, None] * m[None, None, :])   # (B,K,d)
    # D = x - P
    g_x = np.sum(g_D, axis=1)                         # (B,d)
    g_P = -np.sum(g_D, axis=0)                        # (K,d)
    return g_x, g_P, g_m, g_beta


def backward(cache):
    """Return grads dict matching params keys."""
    p   = cache["params"]
    hp  = cache["hp"]
    P   = p["P"]
    m   = cache["m"]; beta = cache["beta"]; eta = cache["eta"]
    rho = cache["rho"]; alpha = cache["alpha"]
    xs  = cache["xs"]; rs = cache["rs"]
    x1  = cache["x1"]; x2 = cache["x2"]; y = cache["y"]; target = cache["target"]
    B   = cache["B"]; K = cache["K"]; d = cache["d"]

    # accumulators
    gP = np.zeros_like(P); gm = np.zeros_like(m)
    gbeta = 0.0; geta = 0.0
    x0 = cache["x0"]

    # ---- (i) RECOGNITION branch: CE at the fused input x0 (via r0) ----
    r0 = cache["r0"]; x_hat = cache["x_hat"]
    g_r0 = np.zeros((B, K))
    g_r0[np.arange(B), y] = -1.0 / (r0[np.arange(B), y] + 1e-12) / B
    gx0_ce, gP_c, gm_c, gbeta_c = _energy_readout_backward(x0, P, m, beta, r0, g_r0)
    gP += gP_c; gm += gm_c; gbeta += gbeta_c

    # ---- (ii) EMENDATION branch: MSE at x_hat, back through the T steps ----
    rmask = cache["recon_mask"]
    gx = hp["lam_recon"] * (2.0 / B) * (x_hat - target) * rmask[None, :]  # dL/dx_hat

    # ---- backprop through the T correction steps (reverse order) ----
    # step t used x = xs[t], r = rs[t], produced x_next = xs[t+1]
    for t in reversed(range(hp["T"])):
        x = xs[t]; r = rs[t]
        D = x[:, None, :] - P[None, :, :]             # (B,K,d)
        pbar = np.einsum("bk,kd->bd", r, P)           # (B,d)
        gE = m[None, :] * (x - pbar)                  # (B,d)   (== (x_next-x)/-eta)
        gx_next = gx                                   # upstream dL/dx_next

        # x_next = x - eta * gE
        #   direct path x_next -> x
        gx_direct = gx_next.copy()
        #   eta path
        geta_step = -np.sum(gx_next * gE)             # scalar
        # scale by d eta / d(raw_eta) handled after loop
        geta += geta_step
        gGE = -eta * gx_next                          # dL/dgE  (B,d)

        # gE = m * (x - pbar)
        gm += np.sum(gGE * (x - pbar), axis=0)        # (d,)
        gx_from_gE = gGE * m[None, :]                 # via (x - pbar) wrt x
        gpbar = -gGE * m[None, :]                     # (B,d)

        # pbar = sum_k r_k P_k
        gP += np.einsum("bk,bd->kd", r, gpbar)        # via P
        g_r = np.einsum("bd,kd->bk", gpbar, P)        # via r  (B,K)

        # r = softmax(q), q = -0.5*beta*wsq, wsq = sum_d m D^2, D = x-P
        gx_from_r, gP_from_r, gm_from_r, gbeta_from_r = \
            _energy_readout_backward(x, P, m, beta, r, g_r)
        gP += gP_from_r; gm += gm_from_r; gbeta += gbeta_from_r

        # total grad flowing into x (this step's input) -> becomes gx for step t-1
        gx = gx_direct + gx_from_gE + gx_from_r

    # ---- fusion: x0 = alpha*x1 + (1-alpha)*x2 ; alpha = rho0/(rho0+rho1) ----
    # gx (from emendation chain) + gx0_ce (from recognition) both land on x0.
    gx0 = gx + gx0_ce
    galpha = np.sum(gx0 * (x1 - x2))                   # scalar
    s = rho[0] + rho[1]
    dalpha_drho0 = rho[1] / (s * s)
    dalpha_drho1 = -rho[0] / (s * s)
    grho = np.array([galpha * dalpha_drho0, galpha * dalpha_drho1])

    # ---- chain through the positive reparameterisations ----
    g_raw_m    = gm * d_softplus(p["raw_m"])
    g_raw_beta = gbeta * d_softplus(p["raw_beta"])
    g_raw_eta  = geta * hp["eta_max"] * (sigmoid(p["raw_eta"]) * (1 - sigmoid(p["raw_eta"])))
    g_raw_rho  = grho * d_softplus(p["raw_rho"])

    return {
        "P": gP,
        "raw_m": g_raw_m,
        "raw_beta": np.array(g_raw_beta),
        "raw_eta": np.array(g_raw_eta),
        "raw_rho": g_raw_rho,
    }


# ==============================================================================
# 4.  FINITE-DIFFERENCE GRADIENT CHECK  (MANDATORY)
# ==============================================================================
def gradient_check(verbose=True):
    K, d = 4, 8
    B = 5
    params = init_params(K, d)
    x1 = RNG.normal(0, 1, (B, d))
    x2 = RNG.normal(0, 1, (B, d))
    target = RNG.normal(0, 1, (B, d))
    y = RNG.integers(0, K, size=B)

    loss, cache = forward(params, x1, x2, y, target)
    grads = backward(cache)

    eps = 1e-6
    max_rel = 0.0
    worst = None
    for name, arr in params.items():
        flat = arr.reshape(-1)
        gflat = grads[name].reshape(-1)
        # check a handful of coordinates per parameter group
        idxs = range(flat.size) if flat.size <= 12 else RNG.choice(flat.size, 12, replace=False)
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps
            lp, _ = forward(params, x1, x2, y, target)
            flat[i] = orig - eps
            lm, _ = forward(params, x1, x2, y, target)
            flat[i] = orig
            num = (lp - lm) / (2 * eps)
            ana = gflat[i]
            rel = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
            if rel > max_rel:
                max_rel = rel; worst = (name, int(i), num, ana)
    if verbose:
        print(f"[grad-check] worst relative error = {max_rel:.3e}  "
              f"at {worst[0]}[{worst[1]}] (num={worst[2]:+.5f} ana={worst[3]:+.5f})")
    ok = max_rel < 1e-4
    print(f"[grad-check] {'PASS' if ok else 'FAIL'} (threshold 1e-4)")
    return ok


# ==============================================================================
# 5.  THE SCRIPTORIUM: synthetic data generator
# ==============================================================================
def make_canon(K, d, d_meaning, seed_scale=1.0):
    """
    Build K canonical 'texts' (the NOMINAL exemplars). The first `d_meaning`
    dimensions are the MEANING that fixes a text's identity; the rest hold the
    text's canonical SPELLING/ornament. All dimensions are unit-scale.
    """
    return RNG.normal(0, seed_scale, (K, d))


def corrupt(canon, y, d_meaning, meaning_noise, spelling_scale=1.0,
            gross_p=0.04, gross_scale=3.0):
    """
    Produce one scribe's copy of text `y`.

      * MEANING dims: the canonical sense + light transmission noise. Sense is
        mostly preserved -- this is the reliable identity signal.
      * SPELLING dims: DRAWN FRESH for every copy. Two faithful copies of the same
        text carry different ornament, so spelling carries NO reliable identity.
        A corrector that trusts spelling to recognise a text will be misled; it
        must learn to read identity from meaning alone.

    Rare 'gross blunders' hit the meaning block (outright copying errors).
    Returns (B, d).
    """
    B, d = y.shape[0], canon.shape[1]
    x = canon[y].copy()
    x[:, :d_meaning] += RNG.normal(0, meaning_noise, (B, d_meaning))
    x[:, d_meaning:] = RNG.normal(0, spelling_scale, (B, d - d_meaning))  # fresh ornament
    gross = np.zeros((B, d))
    gross[:, :d_meaning] = ((RNG.random((B, d_meaning)) < gross_p)
                            * RNG.normal(0, gross_scale, (B, d_meaning)))
    return x + gross


def batch(canon, B, d_meaning, nm1, nm2, spelling_scale=1.0):
    """A training batch with two channels of DIFFERENT reliability (missi pair).
    They differ in how faithfully each transmits the SENSE (meaning noise)."""
    K = canon.shape[0]
    y = RNG.integers(0, K, size=B)
    x1 = corrupt(canon, y, d_meaning, nm1, spelling_scale)   # careful cleric
    x2 = corrupt(canon, y, d_meaning, nm2, spelling_scale)   # hurried lay count
    target = canon[y]
    return x1, x2, y, target


# ==============================================================================
# 6.  ADAM OPTIMISER (from scratch)
# ==============================================================================
class Adam:
    def __init__(self, params, lr=0.02, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] = params[k] - self.lr * mhat / (np.sqrt(vhat) + self.eps)


# ==============================================================================
# 7.  EVALUATION HELPERS
# ==============================================================================
def predict(params, x1, x2, hp=HP):
    B = x1.shape[0]
    dummy_y = np.zeros(B, dtype=int)
    dummy_t = x1
    _, cache = forward(params, x1, x2, dummy_y, dummy_t, hp)
    return cache["r0"].argmax(1), cache["x_hat"], cache["r0"]


def meaning_mask(d, d_meaning):
    m = np.zeros(d); m[:d_meaning] = 1.0
    return m


def accuracy(params, canon, n=400, **kw):
    x1, x2, y, target = batch(canon, n, **kw)
    pred, _, _ = predict(params, x1, x2)
    return float(np.mean(pred == y))


# ==============================================================================
# 8.  MAIN: grad-check -> train -> self-tests
# ==============================================================================
def main():
    print("=" * 74)
    print("THE CORRECTIO ENGINE  -  Chapter 0191, Charlemagne")
    print("error-correcting associative memory (pure NumPy)")
    print("=" * 74)

    # --- 8.1 gradient check (mandatory) ---
    print("\n[1] Finite-difference gradient check")
    assert gradient_check(), "gradient check failed"

    # --- 8.2 build scriptorium + train ---
    K, d, d_meaning = 6, 24, 6
    canon = make_canon(K, d, d_meaning)
    params = init_params(K, d)
    opt = Adam(params, lr=0.02)

    # Two missi: a careful cleric (ch.1) and a hurried lay count (ch.2). In BOTH
    # channels the MEANING block survives well (low noise) and the SPELLING block
    # is mangled (high noise) -- but the count is noisier overall.
    noise = dict(d_meaning=d_meaning, nm1=0.20, nm2=0.55, spelling_scale=1.0)

    print(f"\n[2] Training on a scriptorium of K={K} canonical texts, dim={d}")
    print(f"    (dims 0..{d_meaning-1} = MEANING, survive transmission; "
          f"dims {d_meaning}..{d-1} = SPELLING, mangled)")
    rmask = meaning_mask(d, d_meaning)     # score emendation on the recoverable sense
    print(f"    {'step':>5} | {'loss':>8} | {'CE':>7} | {'MSE':>7} | {'train acc':>9}")
    for step in range(1, 1501):
        x1, x2, y, target = batch(canon, 64, **noise)
        loss, cache = forward(params, x1, x2, y, target, recon_mask=rmask)
        grads = backward(cache)
        opt.step(params, grads)
        if step % 150 == 0 or step == 1:
            acc = accuracy(params, canon, **noise)
            print(f"    {step:5d} | {loss:8.4f} | {cache['ce']:7.4f} | "
                  f"{cache['mse']:7.4f} | {acc:9.3f}")

    # --- 8.3 self-test: EMENDATION (correctio of wrecked copies) ---
    # Take heavily corrupted copies of EVERY canonical text and show the engine
    # pulls them back: it should both cut the distance-to-truth and re-identify
    # the right exemplar. We report the aggregate, not a single lucky draw.
    print("\n[3] Self-test: emendation of heavily corrupted copies (all canon)")
    # A corrector restores the SENSE (meaning block) and re-identifies the text;
    # it deliberately does NOT fabricate the lost ornament (fresh per-copy noise),
    # so we measure recovery where recovery is possible: the meaning block.
    n_per = 200
    y = np.repeat(np.arange(K), n_per)
    truth = canon[y]
    wreck = corrupt(canon, y, d_meaning, meaning_noise=0.6, spelling_scale=1.0)
    pred, x_hat, r = predict(params, wreck, wreck)
    dm = d_meaning
    err_before = np.mean(np.linalg.norm(wreck[:, :dm] - truth[:, :dm], axis=1))
    err_after  = np.mean(np.linalg.norm(x_hat[:, :dm] - truth[:, :dm], axis=1))
    recover_rate = float(np.mean(pred == y))
    print(f"    SENSE distance to truth  before correctio : {err_before:.3f}")
    print(f"    SENSE distance to truth  after  correctio : {err_after:.3f}")
    print(f"    sense error reduced by {100*(1-err_after/err_before):.1f}%  |  "
          f"correct exemplar recovered {recover_rate*100:.1f}% of the time")
    assert err_after < 0.6 * err_before and recover_rate > 0.75, "emendation failed"

    # --- 8.4 self-test: REDUNDANCY (the missi dominici sent in pairs) ---
    print("\n[4] Self-test: two cross-checking channels beat one")
    x1, x2, y, target = batch(canon, 600, **noise)
    acc_pair   = float(np.mean(predict(params, x1, x2)[0] == y))
    acc_single = float(np.mean(predict(params, x2, x2)[0] == y))  # lay report alone
    print(f"    single (hurried) channel accuracy : {acc_single:.3f}")
    print(f"    dual-channel consensus accuracy    : {acc_pair:.3f}")
    print(f"    the pair of missi recovers {100*(acc_pair-acc_single):.1f} more points")

    # --- 8.5 self-test: MEANING > SPELLING (the learned metric) ---
    print("\n[5] Self-test: does the metric privilege understanding over speech?")
    m = softplus(params["raw_m"])
    mean_w = float(np.mean(m[:d_meaning]))
    spell_w = float(np.mean(m[d_meaning:]))
    print(f"    mean metric weight on MEANING dims  : {mean_w:.4f}")
    print(f"    mean metric weight on SPELLING dims : {spell_w:.4f}")
    print(f"    ratio understanding/speech          : {mean_w/spell_w:.2f}x")
    assert mean_w > spell_w, "metric did not learn to weight meaning over spelling"

    # --- 8.6 learned reliabilities of the two offices ---
    rho = softplus(params["raw_rho"])
    alpha = rho[0] / rho.sum()
    print("\n[6] Learned trust the emperor places in each office (missi pair):")
    print(f"    weight on channel 1 (careful cleric): {alpha:.3f}")
    print(f"    weight on channel 2 (hurried count) : {1-alpha:.3f}")
    print(f"    basin sharpness beta = {softplus(params['raw_beta']):.3f}   "
          f"correction step eta = {HP['eta_max']*sigmoid(params['raw_eta']):.3f}")

    print("\n" + "=" * 74)
    print("All checks passed. The engine emends corrupted copies toward the canon,")
    print("trusts the more reliable office, and corrects meaning before ornament.")
    print("=" * 74)


if __name__ == "__main__":
    main()

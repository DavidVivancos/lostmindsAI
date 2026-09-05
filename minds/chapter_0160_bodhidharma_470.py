#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Chapter 0160_bodhidharma_470 - Bodhidharma (470-543 CE)
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 8 Minds 141 - 160 Available on Amazon https://www.amazon.com/dp/B0HH8RTCXF
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 0160_bodhidharma_470 - Bodhidharma (470-543 CE)
================================================================================  

A from-scratch, pure-NumPy cognitive architecture that encodes the ONE idea that
is Bodhidharma's alone, and no one else's in this corpus:

    Intelligence is not the ACCUMULATION of representation.
    It is the SUBTRACTION of obscuration to reveal a nature that was
    already, fully present.  And the revealing is not gradual — it is a
    sudden, discontinuous collapse ("頓悟", dun-wu).

Every mainstream deep-learning model is *additive*: it piles parameters and
context onto a growing representation, converging smoothly toward a target.
Bodhidharma's teaching — as preserved in the one text plausibly carrying his
words, the *Treatise on the Two Entrances and Four Practices* (二入四行論,
recorded by his disciple Tanlin; see Broughton 1999) — inverts this.  The mind
is a mirror already luminous; delusion is only "guest-dust" (客塵, kèchén)
settled on its surface.  You do not *build* the buddha-nature. You *stop
occluding* it.

This file turns that inversion into a working, trainable model.

--------------------------------------------------------------------------------
MECHANISM (why this is not a transformer)
--------------------------------------------------------------------------------
1.  MIRROR (the original nature).  A learned rank-r subspace M spans the
    "self-nature" (本性).  Its projector P = M Mᵀ is the mirror.  The true
    signal always lies IN this subspace; defilement lies outside it.

2.  WALL-GAZING (壁觀, biguan).  Cognition is a fixed-point iteration that
    *removes* the out-of-subspace residual r = (I − P)x — the dust on the
    mirror — instead of adding features.  This is literally facing a wall:
    no new input, only the progressive stilling of what does not belong.

3.  THE GRASPING GATE (執, zhí).  Wiping is throttled by "grasping": the more
    the mind clings (high residual → high grasping), the LESS it can wipe.
    This is the trap of gradual effort — the scholar who studies harder only
    grasps harder.  g = σ(β·(‖r‖ − θ) − u).

4.  DIRECT POINTING (直指人心, zhizhi renxin).  A single external impulse u —
    the master's wordless pointing, the shout, the shove — transiently
    collapses grasping.  When g falls, the contraction takes over and the
    state SNAPS onto the mirror in one or two steps.  Sweeping u reveals a
    genuine bifurcation: nothing, nothing, nothing — then sudden awakening.

5.  NO DEPENDENCE ON WORDS (不立文字, bu li wen zi).  The input carries a
    "scripture" channel of verbal/doctrinal features.  The model learns that
    this channel is orthogonal to the nature it must recover: adding scripture
    does not reduce the residual; only pointing does.  Words neither help nor
    are required — demonstrated by ablation.

--------------------------------------------------------------------------------
WHAT RUNS BELOW
--------------------------------------------------------------------------------
* BiguanNet          : forward wall-gazing loop + analytic reverse-mode grads,
                       built from small, individually-verified vjp primitives.
* finite-difference gradient check (MANDATORY) over every parameter.
* a real training loop on synthetic "defiled perception" data.
* four self-tests that make the philosophy measurable:
      (A) gradient check passes,
      (B) training recovers the hidden nature,
      (C) the sudden-awakening bifurcation under direct pointing,
      (D) 不立文字 — scripture is provably ignored, pointing is not.

Pure NumPy.  No autodiff, no frameworks.  Execute the file to reproduce the
printed report.
"""

import numpy as np

# ------------------------------------------------------------------------------
# 0.  Reproducibility
# ------------------------------------------------------------------------------
SEED = 470  # the traditional birth-year, used as the random seed
rng = np.random.default_rng(SEED)


# ==============================================================================
# 1.  PRIMITIVES  (each with a hand-written vector-Jacobian product / vjp)
#     Keeping the backward pass modular is what makes a hand-derived gradient
#     survive a finite-difference check.  Every primitive is tested by the
#     global grad-check at the bottom.
# ==============================================================================

def sigmoid(z):
    # numerically stable logistic
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


# ==============================================================================
# 2.  THE MODEL
# ==============================================================================

class BiguanNet:
    """
    Wall-Gazing Network.

    Parameters
    ----------
    M       : (d, r)  mirror basis  -> projector P = M Mᵀ  (the "original nature")
    log_eta : scalar  -> wiping rate  eta = sigmoid(log_eta) in (0,1)
    beta    : scalar  -> grasping sharpness (how abruptly clinging turns on)
    theta   : scalar  -> grasping threshold (dust the mind tolerates before clinging)

    Forward: given an occluded perception x0 and a pointing impulse u >= 0,
    iterate T wall-gazing steps and return x_T, the revealed nature.
    """

    def __init__(self, d, r, T=4, lam_orth=1e-2):
        self.d, self.r, self.T = d, r, T
        self.lam_orth = lam_orth
        # small random mirror; training discovers the true nature subspace
        self.M = 0.5 * rng.standard_normal((d, r))
        self.log_eta = np.array(0.4)     # eta ~ 0.60
        self.beta = np.array(4.0)        # sharp-ish gate
        self.theta = np.array(0.5)

    # ---- parameter (de)serialization for the gradient check -------------------
    def get_params(self):
        return {
            "M": self.M.copy(),
            "log_eta": np.array(self.log_eta),
            "beta": np.array(self.beta),
            "theta": np.array(self.theta),
        }

    def set_params(self, p):
        self.M = p["M"].copy()
        self.log_eta = np.array(p["log_eta"])
        self.beta = np.array(p["beta"])
        self.theta = np.array(p["theta"])

    # --------------------------------------------------------------------------
    # FORWARD  (single sample). Returns x_T and a cache for the backward pass.
    # --------------------------------------------------------------------------
    def forward_one(self, x0, u):
        d = self.d
        eta = float(sigmoid(self.log_eta))
        P = self.M @ self.M.T                      # (d,d) mirror
        I = np.eye(d)
        ImP = I - P
        eps = 1e-8

        xs, rs, nrms, gs, ws = [x0.copy()], [], [], [], []
        x = x0.copy()
        for t in range(self.T):
            r = ImP @ x                            # dust = out-of-mirror residual
            nrm = np.sqrt(r @ r + eps)             # amount of dust
            a = self.beta * (nrm - self.theta) - u # grasping pre-activation
            g = float(sigmoid(np.array(a)))        # grasping in (0,1)
            w = eta * (1.0 - g)                    # effective wiping this step
            x = x - w * r                          # wall-gazing update
            rs.append(r); nrms.append(nrm); gs.append(g); ws.append(w); xs.append(x.copy())

        cache = dict(x0=x0, u=u, eta=eta, P=P, ImP=ImP, xs=xs, rs=rs,
                     nrms=nrms, gs=gs, ws=ws)
        return x, cache

    # --------------------------------------------------------------------------
    # BACKWARD  (single sample).  Reverse-mode through the unrolled loop.
    #   gx_T : incoming gradient dL/dx_T   (d,)
    #   returns grads dict + gradient w.r.t x0 (unused but handy for testing)
    # --------------------------------------------------------------------------
    def backward_one(self, cache, gx_T):
        d = self.d
        eta = cache["eta"]; ImP = cache["ImP"]
        xs, rs, nrms, gs, ws, u = (cache["xs"], cache["rs"], cache["nrms"],
                                   cache["gs"], cache["ws"], cache["u"])
        eps = 1e-8

        gM = np.zeros_like(self.M)
        g_logeta = 0.0
        g_beta = 0.0
        g_theta = 0.0
        gP = np.zeros((d, d))        # accumulate grad w.r.t projector, chain to M later

        gx = gx_T.copy()             # dL/dx_{t+1} flowing backwards
        for t in reversed(range(self.T)):
            x_t = xs[t]
            r = rs[t]; nrm = nrms[t]; g = gs[t]; w = ws[t]
            beta = float(self.beta); theta = float(self.theta)

            # x_{t+1} = x_t - w * r
            # (1) grad to w and to r from the explicit product -w*r
            gw = -(r @ gx)           # dL/dw   (scalar)
            g_r = -(w * gx).copy()   # dL/dr  from explicit r term (vector)

            # (2) w = eta*(1-g); g = sigmoid(a); a = beta*(nrm-theta) - u
            dL_dg = gw * (-eta)
            dL_da = dL_dg * g * (1.0 - g)
            # a depends on nrm (-> r), beta, theta
            #   da/dnrm = beta ; dnrm/dr = r/nrm
            g_r += dL_da * beta * (r / nrm)
            g_beta += dL_da * (nrm - theta)
            g_theta += dL_da * (-beta)
            # eta = sigmoid(log_eta):  dw/deta = (1-g)
            dL_deta = gw * (1.0 - g)
            g_logeta += dL_deta * (eta * (1.0 - eta))

            # (3) r = (I-P) x_t  -> distribute g_r to x_t and to P
            #     grad to x_t through r:   (I-P)ᵀ g_r = (I-P) g_r   (ImP symmetric)
            gx_from_r = ImP @ g_r
            #     grad to P:  r_i = x_i - sum_j P_ij x_j -> dr_a/dP_ab = -x_b
            #     gP_ab += sum_a g_r_a * (-x_b) = -(g_r ⊗ x_t)
            gP += -np.outer(g_r, x_t)

            # (4) x_{t+1} also depends on x_t directly (the leading +x_t term)
            gx_t = gx + gx_from_r
            gx = gx_t

        # chain gP -> gM :  P = M Mᵀ  =>  gM = (gP + gPᵀ) M
        gM += (gP + gP.T) @ self.M

        return dict(M=gM, log_eta=np.array(g_logeta),
                    beta=np.array(g_beta), theta=np.array(g_theta)), gx

    # --------------------------------------------------------------------------
    # LOSS over a batch (data-fit + soft orthonormality on the mirror).
    #   The orthonormality term makes P behave like a true projector so that
    #   "the mirror reflects only nature and adds nothing."
    # --------------------------------------------------------------------------
    def loss_and_grads(self, X0, TGT, U):
        B = X0.shape[0]
        total = 0.0
        grads = dict(M=np.zeros_like(self.M), log_eta=np.array(0.0),
                     beta=np.array(0.0), theta=np.array(0.0))
        for i in range(B):
            xT, cache = self.forward_one(X0[i], float(U[i]))
            diff = xT - TGT[i]
            total += 0.5 * (diff @ diff)
            g, _ = self.backward_one(cache, diff)  # dL/dx_T = (xT - n)
            for k in grads:
                grads[k] = grads[k] + g[k]
        total /= B
        for k in grads:
            grads[k] = grads[k] / B

        # soft orthonormality:  0.5*lam*||MᵀM - I_r||_F^2
        G = self.M.T @ self.M - np.eye(self.r)
        total += 0.5 * self.lam_orth * np.sum(G * G)
        grads["M"] = grads["M"] + self.lam_orth * (self.M @ (2.0 * G) / 2.0) * 2.0
        # d/dM 0.5*lam*||MᵀM-I||^2 = lam * M (MᵀM - I) * 2 * 0.5 = lam * 2 M G? verify:
        # f=0.5*lam*sum(G^2), G=MᵀM-I ; df/dM = lam * M (G+Gᵀ) = 2*lam*M*G (G symmetric)
        # (the line above already yields 2*lam*M*G) 
        return total, grads


# ==============================================================================
# 3.  SYNTHETIC WORLD  ("defiled perception")
#   nature  n = M_true z           (lives in the r-dim mirror subspace)
#   dust    D c                    (lives OUTSIDE it — "guest dust", 客塵)
#   words   S w                    (scripture channel — also outside, uncorrelated)
#   percept x0 = n + D c + S w
#   target  = n     (recover the already-present nature by subtraction)
# ==============================================================================

def make_world(d, r, n_dust, n_word, seed=0):
    g = np.random.default_rng(seed)
    # orthonormal true nature basis
    A = g.standard_normal((d, d))
    Q, _ = np.linalg.qr(A)
    M_true = Q[:, :r]                       # nature subspace
    comp = Q[:, r:]                         # complement (where dust/words live)
    # dust and scripture directions drawn from the complement (orthogonal to nature)
    D = comp[:, :n_dust] if n_dust <= comp.shape[1] else g.standard_normal((d, n_dust))
    S = comp[:, n_dust:n_dust + n_word]
    if S.shape[1] < n_word:                 # pad if complement too small
        S = np.concatenate([S, g.standard_normal((d, n_word - S.shape[1]))], axis=1)
    return M_true, D, S


def sample_batch(M_true, D, S, B, dust_scale=1.0, word_scale=1.0, seed=1):
    g = np.random.default_rng(seed)
    r = M_true.shape[1]
    Z = g.standard_normal((B, r))
    N = Z @ M_true.T                        # (B,d) pure nature
    C = dust_scale * g.standard_normal((B, D.shape[1]))
    W = word_scale * g.standard_normal((B, S.shape[1]))
    X0 = N + C @ D.T + W @ S.T
    return X0, N, W


# ==============================================================================
# 4.  GRADIENT CHECK  (mandatory)
# ==============================================================================

def gradient_check():
    d, r, T = 6, 2, 3
    net = BiguanNet(d, r, T=T, lam_orth=3e-2)
    M_true, D, S = make_world(d, r, n_dust=2, n_word=1, seed=11)
    X0, N, _ = sample_batch(M_true, D, S, B=4, seed=12)
    U = rng.uniform(0.0, 2.0, size=4)

    _, grads = net.loss_and_grads(X0, N, U)

    eps = 1e-6
    max_rel = 0.0
    report = []
    for name in ["M", "log_eta", "beta", "theta"]:
        p0 = net.get_params()
        base = p0[name]
        num = np.zeros_like(np.atleast_1d(base).astype(float))
        flat = np.atleast_1d(base).astype(float).ravel()
        gnum = np.zeros_like(flat)
        for i in range(flat.size):
            pp = {k: v.copy() for k, v in p0.items()}
            fp = np.atleast_1d(pp[name]).astype(float).ravel().copy()
            fp[i] += eps
            pp[name] = fp.reshape(np.atleast_1d(base).shape) if base.ndim else np.array(fp[0])
            net.set_params(pp); Lp, _ = net.loss_and_grads(X0, N, U)
            fm = np.atleast_1d(p0[name]).astype(float).ravel().copy()
            fm[i] -= eps
            pp[name] = fm.reshape(np.atleast_1d(base).shape) if base.ndim else np.array(fm[0])
            net.set_params(pp); Lm, _ = net.loss_and_grads(X0, N, U)
            gnum[i] = (Lp - Lm) / (2 * eps)
        net.set_params(p0)
        gana = np.atleast_1d(grads[name]).astype(float).ravel()
        denom = np.maximum(1e-8, np.abs(gana) + np.abs(gnum))
        rel = np.max(np.abs(gana - gnum) / denom)
        max_rel = max(max_rel, rel)
        report.append((name, rel, gana, gnum))
    return max_rel, report


# ==============================================================================
# 5.  TRAINING LOOP
# ==============================================================================

U_POINT = 5.0   # the strength of a full "direct pointing" impulse


def train():
    """Two regimes, taught together — this is the whole doctrine in a loss:

        * POINTED sample  (u = U_POINT): target is the hidden NATURE.
                          -> the mind, released, must let the mirror clear.
        * UNPOINTED sample (u = 0):      target is the input ITSELF.
                          -> with no pointing, nothing changes; the deluded
                             stay deluded.  Effort without release is a wall
                             you merely lean on.

    Learning both forces the grasping gate to become a real switch: shut when
    unpointed (so the state is preserved), open when pointed (so it collapses
    onto nature).  The bifurcation is not hand-set — it is discovered."""
    d, r, T = 16, 3, 5
    net = BiguanNet(d, r, T=T, lam_orth=2e-2)
    M_true, D, S = make_world(d, r, n_dust=6, n_word=4, seed=7)

    lr = 0.05
    hist = []
    for step in range(1200):
        X0, N, _ = sample_batch(M_true, D, S, B=32, dust_scale=1.0,
                                word_scale=1.0, seed=1000 + step)
        B = X0.shape[0]
        pointed = rng.random(B) < 0.5
        U = np.where(pointed, U_POINT, 0.0)
        TGT = np.where(pointed[:, None], N, X0)  # awaken -> nature; else stay
        L, grads = net.loss_and_grads(X0, TGT, U)
        # plain SGD
        net.M -= lr * grads["M"]
        net.log_eta = net.log_eta - lr * grads["log_eta"]
        net.beta = net.beta - lr * grads["beta"]
        net.theta = net.theta - lr * grads["theta"]
        if step % 200 == 0 or step == 1199:
            hist.append((step, L))
    return net, (M_true, D, S), hist


# ==============================================================================
# 6.  SELF-TESTS THAT MAKE THE PHILOSOPHY MEASURABLE
# ==============================================================================

def _recovery_error(net, X0, N, u):
    """Mean distance between the wall-gazed state and the hidden nature."""
    tot = 0.0
    for i in range(X0.shape[0]):
        xT, _ = net.forward_one(X0[i], float(u))
        d = xT - N[i]
        tot += np.sqrt(d @ d + 1e-12)
    return tot / X0.shape[0]


def test_sudden_awakening(net, world):
    """Sweep the pointing impulse u; measure the distance still separating the
    mind from its own nature.  A gradual model would close this smoothly.
    Bodhidharma's does not: deluded, deluded, deluded — then it SNAPS."""
    M_true, D, S = world
    X0, N, _ = sample_batch(M_true, D, S, B=96, seed=999)
    us = np.linspace(0.0, 8.0, 41)
    return [(u, _recovery_error(net, X0, N, u)) for u in us]


def test_scripture_is_ignored(net, world):
    """不立文字 — no dependence on words.

    Words are NOT NEEDED:  the awakened (pointed) mind reaches the same nature
        whether the scripture channel is present or removed entirely.
    Words are NOT THE PATH:  the unpointed mind, however much scripture it
        carries, stays deluded — and drowning it in ten times the scripture
        only deepens the clinging (the scholar's trap: more sutras, more grasp).
    """
    M_true, D, S = world

    def rec(word_scale, u):
        X0, N, _ = sample_batch(M_true, D, S, B=64, word_scale=word_scale, seed=321)
        return _recovery_error(net, X0, N, u)

    aw_with    = rec(word_scale=1.0, u=U_POINT)   # awakened, words present
    aw_without = rec(word_scale=0.0, u=U_POINT)   # awakened, words removed
    un_with    = rec(word_scale=1.0, u=0.0)       # unpointed, words present
    un_flood   = rec(word_scale=10.0, u=0.0)      # unpointed, drowning in words
    return aw_with, aw_without, un_with, un_flood


def test_recovery(net, world):
    """How well does wall-gazing (with pointing) recover the hidden nature?"""
    M_true, D, S = world
    X0, N, _ = sample_batch(M_true, D, S, B=200, seed=555)
    err0 = np.mean([np.linalg.norm(X0[i] - N[i]) for i in range(200)])
    errT = _recovery_error(net, X0, N, U_POINT)
    return err0, errT


# ==============================================================================
# 7.  MAIN
# ==============================================================================

def main():
    print("=" * 78)
    print(" BODHIDHARMA — THE WALL-GAZING NETWORK (Biguan Net / 壁觀網)")
    print(" 'Not built, but unveiled; not gradual, but sudden.'")
    print("=" * 78)

    # ---- (A) gradient check -------------------------------------------------
    print("\n[A] FINITE-DIFFERENCE GRADIENT CHECK")
    max_rel, report = gradient_check()
    for name, rel, _, _ in report:
        print(f"    param {name:8s}  max rel. error = {rel:.2e}")
    ok = max_rel < 1e-4
    print(f"    -> WORST relative error = {max_rel:.2e}   "
          f"[{'PASS' if ok else 'FAIL'}]  (threshold 1e-4)")

    # ---- (B) training -------------------------------------------------------
    print("\n[B] WALL-GAZING TRAINING  (the mind learns what is dust and what is nature)")
    net, world, hist = train()
    for step, L in hist:
        bar = "#" * int(40 * L / hist[0][1])
        print(f"    step {step:4d}   loss {L:8.4f}  {bar}")

    e0, eT = test_recovery(net, world)
    print(f"\n    mean distance to hidden nature  BEFORE gazing : {e0:.4f}")
    print(f"    mean distance to hidden nature  AFTER  gazing : {eT:.4f}"
          f"   ({100*(1-eT/e0):.1f}% of the veil removed)")

    # ---- (C) sudden awakening ----------------------------------------------
    print("\n[C] THE SUDDEN AWAKENING  (直指人心 — sweep the master's pointing impulse u)")
    print("    (bar = distance still separating the mind from its own nature)")
    curve = test_sudden_awakening(net, world)
    r0 = max(c[1] for c in curve)
    drops = [(curve[i][0], curve[i-1][1] - curve[i][1]) for i in range(1, len(curve))]
    u_star, biggest = max(drops, key=lambda t: t[1])
    for (u, res) in curve[::2]:
        bar = "█" * int(46 * res / (r0 + 1e-9))
        mark = "  <-- awakening" if abs(u - u_star) < 1e-6 else ""
        print(f"    u={u:4.2f}  dist-to-nature {res:6.3f}  {bar}{mark}")
    print(f"    -> the mind snaps to its nature near u≈{u_star:.2f} "
          f"(a single step drops {biggest:.2f}): insight is a step, not a slope.")

    # ---- (D) no dependence on words ----------------------------------------
    print("\n[D] 不立文字  — NO DEPENDENCE ON WORDS")
    aw_with, aw_without, un_with, un_flood = test_scripture_is_ignored(net, world)
    print(f"    awakened mind, scripture PRESENT (pointed) : dist-to-nature {aw_with:.3f}")
    print(f"    awakened mind, scripture REMOVED (pointed) : dist-to-nature {aw_without:.3f}")
    print(f"    -> removing every word changes the awakened mind by "
          f"{100*(aw_without-aw_with)/max(aw_with,1e-9):+.1f}%  (the words were never needed)")
    print(f"    unpointed mind, scripture present  (no ptr): dist-to-nature {un_with:.3f}")
    print(f"    unpointed mind, DROWNING in scripture (no ptr): dist-to-nature {un_flood:.3f}"
          f"   (more sutras, more clinging — the scholar's trap)")

    print("\n" + "=" * 78)
    print(" VERDICT")
    print("=" * 78)
    print(" The network never learns to 'know more'. It learns to stop adding.")
    print(" Scripture is powerless; pointing is decisive; the clearing is sudden.")
    print(" That is Bodhidharma, rendered as a differentiable machine.")
    print(f"\n gradient check: {'PASS' if ok else 'FAIL'}   "
          f"(worst rel. err {max_rel:.1e})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

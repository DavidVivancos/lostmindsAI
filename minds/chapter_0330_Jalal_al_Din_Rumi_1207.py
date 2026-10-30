#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0330 · Jalal al-Din Muhammad Rumi (1207-1273)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 17, Minds 321-340 Available on Amazon https://www.amazon.com/dp/B0HP7Z1NKC
# Resume and Interactive Demos at https://artificiology.com/
 THE NEY-ASTROLABE ENGINE  ·  nay-ostorlab
 A from-scratch, pure-NumPy cognitive architecture, trainable and executable.
================================================================================

WHY THIS ARCHITECTURE AND NOT A TRANSFORMER
-------------------------------------------
The standard modern design stores content as *positive keys and values* and
generates autonomously from them: cut the input and it still speaks, because
the substance is inside. Rumi's surviving corpus commits him, over and over, to
the opposite claim, and it commits him at the level of mechanism, not mood.

Six textual commitments, each turned into a piece of runnable machinery:

  (1) THE INSTRUMENT HAS NO CONTENT OF ITS OWN — it has a leak.
      Masnavi I.1-3: the reed is cut from the reed bed; its sound is the shape
      of a removal. Here every learnable "content" parameter can only SUBTRACT
      energy from a fixed carrier. The parameter `g` produces stops in (0,1]
      via pass = exp(-(softplus(g) + eps0)). Learning burns holes. It never
      adds. There is no positive value store anywhere in this model.

  (2) NOTHING IS PRODUCED WITHOUT AN EXTERNAL DRIVE.
      The Masnavi is a dictated poem; when Husam al-Din Chalabi withdrew after
      Book One, composition stopped for roughly two years, and Book Two opens
      by saying so. The only energy source in this network is the drive term
      `Bin @ u`. Because eps0 > 0 is a hard floor, ||x|| decays by at least
      exp(-eps0) per silent step. Silence in, silence out is a THEOREM here,
      not a trained behaviour. See test SILENCE.

  (3) THE STATE TURNS; IT DOES NOT TRAVEL.
      The sama' is rotation about a fixed foot. The recurrence is an exact
      orthogonal matrix Q built by the Cayley transform of a skew-symmetric
      parameter, Q = (I-A)(I+A)^-1, A = S - S^T. Norm is conserved to machine
      precision across arbitrarily many steps. See test CHARKH.

  (4) THE READOUT IS AN INSTRUMENT, NOT A CONCLUSION.
      Masnavi I.110: 'eshq astrolab-e asrar-e khodast — love is the ASTROLABE
      of God's mysteries. An astrolabe is stereographic projection of a sphere
      onto a flat plate. So the readout literally stereographically projects
      the unit-norm state off the sphere onto a plane, and all measurement
      happens on that plate. Nothing reads the sphere directly.

  (5) THE TARGET STATE IS BEWILDERMENT, NOT CONVERGENCE.
      Masnavi IV: zirakī befrūsh o hayrānī bekhar / zirakī zann ast o hayrānī
      nazar — sell cleverness, buy bewilderment; cleverness is zann (conjecture),
      bewilderment is nazar (direct seeing). The loss carries a HAYRAT FLOOR
      that penalizes the trajectory for settling into a fixed point. The system
      is optimized toward a limit cycle, never toward rest.

  (6) EVERY LISTENER HEARS HIS OWN CONJECTURE — AND THAT IS A DEFECT, NOT A
      VIRTUE.  Masnavi I.6: har kasi az zann-e khod shod yar-e man / az darun-e
      man najost asrar-e man. And Masnavi III.1259-1270, the elephant in the
      dark house: each hand finds one part; III.1269 says that if there had
      been A CANDLE IN EACH HAND the difference would have gone out of their
      words. So this model has K "zann" heads reading DISJOINT slices of the
      astrolabe plate, and one runtime switch `candle` that hands every head
      the same shared, unlearned illumination of the whole plate. The
      experiment measures whether disagreement collapses when the candle is
      lit. Rumi predicts it should. See test ELEPHANT.

WHAT THIS IS NOT
----------------
Not attention. Not a mixture of experts. Not a transformer with Persian
variable names. There is no softmax anywhere in the model, no key-value store,
no autoregressive self-feeding loop. Removing the drive removes the output.

CONVENTIONS
-----------
- Pure NumPy. No autograd. Every gradient is derived and hand-written.
- A finite-difference gradient check on every parameter block is mandatory and
  runs first. If it fails, the file exits non-zero.
- A real training loop on a real task, with a real baseline to beat.
- All experiments self-contained and deterministic under the given seed.

Run:  python3 chapter_0330_Jalal_al_Din_Rumi_1207.py
================================================================================
"""

import numpy as np
import sys
import time

# ==============================================================================
# SECTION 0 — SMALL UTILITIES
# ==============================================================================

def softplus(z):
    """Numerically stable softplus: log(1+e^z). Used to keep stops >= 0."""
    return np.logaddexp(0.0, z)

def sigmoid(z):
    """Derivative of softplus, and the gate we need for the stop gradient."""
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


# ==============================================================================
# SECTION 1 — CHARKH: THE TURN
# ------------------------------------------------------------------------------
# The whirl. An exactly orthogonal recurrence produced by the Cayley transform
# of a skew-symmetric matrix. This is the one component with no free choice in
# it: the whole point is that the state's magnitude is untouched by the passage
# of time. Anything that happens to the state's size must happen in the reed,
# not in the turning.
#
#   A = S - S^T                       (skew-symmetric, by construction)
#   M = (I + A)^-1                    (always invertible for skew A)
#   Q = (I - A) M                     (orthogonal, det = +1)
#
# The eigenvalues of Q are exp(-2i*arctan(theta_m)) for the imaginary parts
# i*theta_m of A's eigenvalues, so each conjugate pair rotates by a fixed angle
# omega_m = 2*arctan(theta_m) radians per step. Those angles ARE the natural
# frequencies of the instrument, and the resonance test later recovers them.
# ==============================================================================

def cayley_forward(S):
    """Build the exact rotation Q from the raw parameter S. Returns Q, A, M."""
    A = S - S.T
    I = np.eye(S.shape[0])
    M = np.linalg.inv(I + A)
    Q = (I - A) @ M
    return Q, A, M

def cayley_backward(gQ, Q, M):
    """
    Vector-Jacobian product for the Cayley transform.

    dQ = -(I + Q) dA M   ==>   dL/dA = -(I + Q)^T gQ M^T
    and since A = S - S^T,     dL/dS = dL/dA - (dL/dA)^T
    """
    I = np.eye(Q.shape[0])
    gA = -(I + Q).T @ gQ @ M.T
    return gA - gA.T

def rotation_angles(Q):
    """The per-pair turning rates of the wheel, in radians per step."""
    ev = np.linalg.eigvals(Q)
    ang = np.abs(np.angle(ev))
    ang = np.sort(ang[ang > 1e-9])
    return ang[::2]  # conjugate pairs give each angle twice


# ==============================================================================
# SECTION 2 — THE MODEL
# ==============================================================================

class NeyAstrolabe:
    """
    THE NEY-ASTROLABE ENGINE.

    One step of the recurrence, in order:

        z_t  = Q x_{t-1}                     CHARKH   — the turn (norm-preserving)
        y_t  = P  * z_t                      NEY      — the stops (subtract only)
        x_t  = y_t + Bin u_t                 DAM      — the breath (only energy in)
        v_t  = x_t / ||x_t||                 the sphere
        p_t  = stereographic(v_t)            OSTORLAB — projection onto the plate
        r_t  = candle * (C p_t)              SHAM'    — one shared, unlearned light
        d_kt = mask_k * p_t                  ZANN     — each head's own portion
        f_kt = W_k [d_kt ; r_t]              each head's conjecture
        pred = mean_k f_kt                   the consensus

    Parameters, all of them:
        S    (D,D)          -> Q, the wheel
        g    (D/2,)         -> the stops; content lives here, as absence
        Bin  (D,Din)        -> how the breath enters the reed
        W_k  (Din, D-1+Cs)  -> each listener's conjecture, K of them
    """

    def __init__(self, D=24, Din=4, K=4, Cs=12, delta=0.25, eps0=0.01, seed=0):
        assert D % 2 == 0, "D must be even: the wheel turns in conjugate pairs."
        rng = np.random.default_rng(seed)
        self.D, self.Din, self.K, self.Cs = D, Din, K, Cs
        self.M_modes = D // 2
        self.delta = delta      # pole guard for the stereographic plate
        self.eps0 = eps0        # HARD leak floor: guarantees silence-in/silence-out

        # --- CHARKH ------------------------------------------------------------
        # A skew matrix with entries ~ N(0, sigma) has imaginary eigenvalues of
        # magnitude ~ sigma*sqrt(D) = theta, and Cayley turns theta into a
        # rotation of omega = 2*arctan(theta) rad/step. sigma = 0.075 at D = 24
        # seeds the wheel with a BAND of turning rates around 0.6 rad/step,
        # overlapping the drive band, so the instrument starts out roughly in
        # the register it will be asked to answer in. Training moves them.
        self.S = rng.normal(0.0, 0.075, size=(D, D))

        # --- NEY ---------------------------------------------------------------
        # g near -2 => softplus(g) ~ 0.13 => the reed starts nearly open, with
        # only faint holes. Training decides where to burn.
        self.g = rng.normal(-2.0, 0.25, size=(self.M_modes,))

        # --- DAM ---------------------------------------------------------------
        self.Bin = rng.normal(0.0, 1.0 / np.sqrt(Din), size=(D, Din))

        # --- ZANN heads --------------------------------------------------------
        Dp = D - 1                       # the plate has one dimension fewer
        self.Wk = [rng.normal(0.0, 0.3 / np.sqrt(Dp + Cs), size=(Din, Dp + Cs))
                   for _ in range(K)]

        # --- The elephant in the dark: fixed DISJOINT slices, one per hand -----
        # These are not learned. They are where each listener's hand happened
        # to fall.
        perm = rng.permutation(Dp)
        self.masks = np.zeros((K, Dp))
        for k in range(K):
            self.masks[k, perm[k::K]] = 1.0

        # --- The candle: one shared, UNLEARNED compression of the whole plate --
        # Everyone gets the same light or nobody does. It is not tuned to any
        # head, which is the entire point: it is a candle, not a lens.
        self.C = rng.normal(0.0, 1.0 / np.sqrt(Dp), size=(Cs, Dp))

    # -- parameter plumbing ---------------------------------------------------
    def params(self):
        return [self.S, self.g, self.Bin] + self.Wk

    def set_params(self, plist):
        self.S, self.g, self.Bin = plist[0], plist[1], plist[2]
        self.Wk = plist[3:]

    def pass_gains(self):
        """
        P_i in (0,1). The reed's transmission per state coordinate.
        exp(-(softplus(g) + eps0)) — bounded strictly below 1 by the eps0 floor,
        which is what makes the silence property a theorem instead of a hope.
        """
        stops = softplus(self.g) + self.eps0        # (M_modes,)
        pm = np.exp(-stops)                          # (M_modes,)
        return np.repeat(pm, 2), pm, stops           # (D,), (M,), (M,)

    # -------------------------------------------------------------------------
    # FORWARD
    # -------------------------------------------------------------------------
    def forward(self, U, candle=0.0, cache=True):
        """
        U : (B, T, Din) the drive — the breath through the reed.
        Returns pred (B,T,Din), per-head preds (B,T,K,Din), and a cache.

        `candle` is a RUNTIME switch in [0,1], not a parameter. 0 = the dark
        house of Masnavi III.1259. 1 = a candle in every hand, III.1269.
        """
        B, T, Din = U.shape
        D, K, Dp, Cs = self.D, self.K, self.D - 1, self.Cs
        Q, A, Mi = cayley_forward(self.S)
        P, pm, stops = self.pass_gains()

        x = np.zeros((B, D))
        Xs, Zs, Vs, Ns, Ps, Rs, Ds, Qs, As = [], [], [], [], [], [], [], [], []
        preds = np.zeros((B, T, Din))
        head_preds = np.zeros((B, T, K, Din))

        for t in range(T):
            z = x @ Q.T                       # CHARKH: turn
            y = z * P[None, :]                # NEY: subtract through the holes
            x = y + U[:, t, :] @ self.Bin.T   # DAM: the only energy entering

            n = np.sqrt(np.sum(x * x, axis=1) + 1e-12)      # (B,)
            v = x / n[:, None]                               # on the sphere
            den = 1.0 + self.delta - v[:, D - 1]             # pole guard
            q = v[:, :Dp] / den[:, None]                     # OSTORLAB: where
            a = n / (1.0 + n)                                # LOUDNESS: how much
            p = a[:, None] * q                               # the plate reading

            r = candle * (p @ self.C.T)                      # SHAM': shared light
            dsl = self.masks[None, :, :] * p[:, None, :]     # ZANN: each own part

            hp = np.zeros((B, K, Din))
            for k in range(K):
                feat = np.concatenate([dsl[:, k, :], r], axis=1)   # (B, Dp+Cs)
                hp[:, k, :] = feat @ self.Wk[k].T
            head_preds[:, t] = hp
            preds[:, t] = hp.mean(axis=1)

            if cache:
                Zs.append(z); Xs.append(x.copy()); Vs.append(v)
                Ns.append(n); Ps.append(p); Rs.append(r); Ds.append(dsl)
                Qs.append(q); As.append(a)

        cch = dict(Q=Q, A=A, Mi=Mi, P=P, pm=pm, stops=stops, U=U,
                   Z=Zs, X=Xs, V=Vs, N=Ns, Pl=Ps, R=Rs, Dsl=Ds, Qp=Qs, Amp=As,
                   candle=candle, B=B, T=T)
        return preds, head_preds, cch

    # -------------------------------------------------------------------------
    # LOSS  —  prediction error + the hayrat (bewilderment) floor
    # -------------------------------------------------------------------------
    def loss(self, U, Y, candle=0.0, lam_h=0.05, tau=0.80):
        """
        Y : (B,T,Din) target, here the next drive sample.

        L = MSE(pred, Y) + lam_h * softplus(tau - mean_t ||v_t - v_{t-1}||^2)

        The second term is the whole difference between this network and a
        network that wants an answer. It is ZERO as soon as the trajectory keeps
        turning fast enough, and it grows only when the state starts to settle.
        It does not reward chaos. It forbids rest.
        """
        preds, head_preds, c = self.forward(U, candle=candle)
        B, T = c['B'], c['T']
        # EACH head is scored on its own report, not merely on the average.
        # This is the difference between "the committee was right on average"
        # and "each person in the dark room said what the animal was". Only the
        # second is what Masnavi III.1259 is about, and only the second gives
        # the candle anything to do.
        diff_h = head_preds - Y[:, :, None, :]                  # (B,T,K,Din)
        mse = np.sum(diff_h * diff_h) / (B * T * self.Din * self.K)
        diff = preds - Y                                        # consensus, reported
        mse_cons = np.sum(diff * diff) / (B * T * self.Din)

        V = np.stack(c['V'], axis=1)                      # (B,T,D)
        dv = V[:, 1:, :] - V[:, :-1, :]
        d = np.sum(dv * dv, axis=2)                       # (B,T-1)
        dbar = d.mean(axis=1)                             # (B,)
        Lh = softplus(tau - dbar).mean()

        c['diff'] = diff; c['diff_h'] = diff_h; c['mse_cons'] = mse_cons
        c['dv'] = dv; c['dbar'] = dbar
        c['lam_h'] = lam_h; c['tau'] = tau
        c['head_preds'] = head_preds
        return mse + lam_h * Lh, mse, Lh, c

    # -------------------------------------------------------------------------
    # BACKWARD — hand-derived BPTT. Every line below is checked against finite
    # differences before this file is allowed to train anything.
    # -------------------------------------------------------------------------
    def backward(self, c):
        D, K, Din, Dp = self.D, self.K, self.Din, self.D - 1
        B, T = c['B'], c['T']
        Q, Mi, P = c['Q'], c['Mi'], c['P']
        lam_h, tau = c['lam_h'], c['tau']

        gS = np.zeros_like(self.S)
        gg = np.zeros_like(self.g)
        gBin = np.zeros_like(self.Bin)
        gWk = [np.zeros_like(w) for w in self.Wk]
        gQ = np.zeros_like(Q)
        gP = np.zeros(D)

        scale = 1.0 / (B * T * Din * K)
        # d L_h / d d_t  , per sequence
        s_h = sigmoid(tau - c['dbar'])                    # (B,)
        coef = -lam_h * s_h / ((T - 1) * B)               # (B,)

        gv_from_h = np.zeros((B, T, D))
        dv = c['dv']                                       # (B,T-1,D)
        contrib = 2.0 * coef[:, None, None] * dv           # (B,T-1,D)
        gv_from_h[:, 1:, :] += contrib
        gv_from_h[:, :-1, :] -= contrib

        gx_next = np.zeros((B, D))                         # carried backwards

        for t in range(T - 1, -1, -1):
            v, n, p, r = c['V'][t], c['N'][t], c['Pl'][t], c['R'][t]
            dsl, x, z = c['Dsl'][t], c['X'][t], c['Z'][t]

            # ---- from the prediction error, through the K conjectures --------
            gp = np.zeros((B, Dp))
            gr = np.zeros((B, self.Cs))
            for k in range(K):
                gh = 2.0 * c['diff_h'][:, t, k, :] * scale   # (B,Din) per listener
                feat = np.concatenate([dsl[:, k, :], r], axis=1)
                gWk[k] += gh.T @ feat
                gfeat = gh @ self.Wk[k]                     # (B, Dp+Cs)
                gp += gfeat[:, :Dp] * self.masks[k][None, :]
                gr += gfeat[:, Dp:]
            gp += c['candle'] * (gr @ self.C)

            # ---- back through the astrolabe: p = a * q -----------------------
            # q is WHERE on the plate (pure stereographic projection of the
            # direction); a = n/(1+n) is HOW LOUD. Splitting the readout this way
            # is what makes silence structural: as ||x|| -> 0, a -> 0, and the
            # plate goes blank no matter where the direction happens to point.
            qp, amp = c['Qp'][t], c['Amp'][t]
            den = 1.0 + self.delta - v[:, D - 1]
            gq = gp * amp[:, None]
            ga = np.sum(gp * qp, axis=1)                    # (B,)
            gv = np.zeros((B, D))
            gv[:, :Dp] += gq / den[:, None]
            gv[:, D - 1] += np.sum(gq * v[:, :Dp], axis=1) / (den ** 2)

            # ---- the hayrat floor also pushes on v ---------------------------
            gv += gv_from_h[:, t, :]

            # ---- back through the normalization onto the sphere --------------
            xg = np.sum(x * gv, axis=1)                     # (B,)
            gx = (gv - x * (xg / (n ** 2))[:, None]) / n[:, None]
            # ---- and the separate path through the loudness gate -------------
            gn = ga / ((1.0 + n) ** 2)                      # da/dn = 1/(1+n)^2
            gx = gx + (gn / n)[:, None] * x                 # dn/dx = x/n

            # ---- plus whatever the future needs of this state ----------------
            gx = gx + gx_next

            # ---- x = P*z + Bin u --------------------------------------------
            gBin += gx.T @ c['U'][:, t, :]
            gP += np.sum(gx * z, axis=0)
            gz = gx * P[None, :]

            # ---- z = Q x_{t-1} ----------------------------------------------
            if t > 0:
                gQ += gz.T @ c['X'][t - 1]
                gx_next = gz @ Q
            else:
                gx_next = np.zeros((B, D))                  # x_{-1} = 0

        # ---- the stops: P_i = exp(-(softplus(g_m)+eps0)), i in pair m --------
        pm = c['pm']
        gP_pairs = gP.reshape(self.M_modes, 2).sum(axis=1)
        gg += gP_pairs * (-pm) * sigmoid(self.g)

        # ---- the wheel ------------------------------------------------------
        gS += cayley_backward(gQ, Q, Mi)

        return [gS, gg, gBin] + gWk


# ==============================================================================
# SECTION 3 — THE TASK
# ------------------------------------------------------------------------------
# A driven-prediction task, chosen because it is the only kind of task this
# architecture is even ALLOWED to attempt. A model with no autonomous dynamics
# cannot be asked to free-run; it can only be asked to be a good companion to
# something that is already sounding.
#
# The drive is a quasi-periodic four-channel breath with per-sequence random
# frequencies and phases. Target: the next sample. The honest baseline is
# persistence (predict u_t for u_{t+1}), which is strong for slow drives, so
# the frequency band is set high enough that persistence is genuinely poor.
# ==============================================================================

def make_batch(B, T, Din=4, seed=0, wlo=0.25, whi=0.95):
    rng = np.random.default_rng(seed)
    t = np.arange(T + 1)[None, None, :]
    w = rng.uniform(wlo, whi, size=(B, Din, 1))
    ph = rng.uniform(0, 2 * np.pi, size=(B, Din, 1))
    amp = rng.uniform(0.6, 1.0, size=(B, Din, 1))
    sig = amp * np.sin(w * t + ph)
    # couple two channels so the task is not four independent sinusoids
    sig[:, 2, :] = 0.6 * sig[:, 0, :] * sig[:, 1, :] + 0.4 * sig[:, 2, :]
    sig = np.transpose(sig, (0, 2, 1))                    # (B,T+1,Din)
    return sig[:, :T, :], sig[:, 1:T + 1, :]


def single_tone(T, w, Din=4, amp=1.0):
    """A pure tone on channel 0 only — used for the resonance sweep."""
    t = np.arange(T)
    u = np.zeros((1, T, Din))
    u[0, :, 0] = amp * np.sin(w * t)
    return u


# ==============================================================================
# SECTION 4 — THE MANDATORY GRADIENT CHECK
# ==============================================================================

def gradient_check():
    print("-" * 78)
    print("TEST 1  GRADIENT CHECK  (central finite differences, every block)")
    print("-" * 78)
    net = NeyAstrolabe(D=8, Din=3, K=2, Cs=2, seed=3)
    U, Y = make_batch(3, 6, Din=3, seed=11)
    L, _, _, c = net.loss(U, Y, candle=0.5, lam_h=0.3, tau=0.80)
    grads = net.backward(c)
    names = ["S (charkh)", "g (ney stops)", "Bin (dam)"] + \
            [f"W{k} (zann head {k})" for k in range(net.K)]
    ps = net.params()
    rng = np.random.default_rng(7)
    eps = 1e-6
    worst = 0.0
    for name, P_, G in zip(names, ps, grads):
        idx = [tuple(rng.integers(0, s) for s in P_.shape) for _ in range(6)]
        num, ana = [], []
        for ix in idx:
            o = P_[ix]
            P_[ix] = o + eps; Lp, *_ = net.loss(U, Y, candle=0.5, lam_h=0.3, tau=0.80)
            P_[ix] = o - eps; Lm, *_ = net.loss(U, Y, candle=0.5, lam_h=0.3, tau=0.80)
            P_[ix] = o
            num.append((Lp - Lm) / (2 * eps)); ana.append(G[ix])
        num, ana = np.array(num), np.array(ana)
        rel = np.max(np.abs(num - ana) / (np.abs(num) + np.abs(ana) + 1e-12))
        worst = max(worst, rel)
        print(f"   {name:<24s} max rel. err = {rel:.3e}   "
              f"{'PASS' if rel < 1e-5 else 'FAIL'}")
    print(f"\n   worst relative error across all blocks: {worst:.3e}")
    if worst >= 1e-5:
        print("   GRADIENT CHECK FAILED — refusing to train.")
        sys.exit(1)
    print("   GRADIENT CHECK PASSED.\n")
    return worst


# ==============================================================================
# SECTION 5 — TRAINING
# ==============================================================================

def train(net, iters=700, B=64, T=48, lr=6e-3, lam_h=0.05, tau=0.80,
          seed=100, verbose=True, tag=""):
    """Adam. Candle sampled each step so BOTH the dark and the lit regime are
    trained; the elephant test then compares them fairly at evaluation time."""
    m = [np.zeros_like(p) for p in net.params()]
    v = [np.zeros_like(p) for p in net.params()]
    b1, b2, e = 0.9, 0.999, 1e-8
    rng = np.random.default_rng(seed)
    hist = []
    for it in range(1, iters + 1):
        U, Y = make_batch(B, T, Din=net.Din, seed=int(rng.integers(1e9)))
        candle = float(rng.integers(0, 2))
        L, mse, Lh, c = net.loss(U, Y, candle=candle, lam_h=lam_h, tau=tau)
        gs = net.backward(c)
        for i, (p, g) in enumerate(zip(net.params(), gs)):
            gn = np.linalg.norm(g)
            if gn > 5.0:
                g = g * (5.0 / gn)                       # clip: the wheel is stiff
            m[i] = b1 * m[i] + (1 - b1) * g
            v[i] = b2 * v[i] + (1 - b2) * g * g
            mh = m[i] / (1 - b1 ** it)
            vh = v[i] / (1 - b2 ** it)
            p -= lr * mh / (np.sqrt(vh) + e)
        hist.append(mse)
        if verbose and (it % 100 == 0 or it == 1):
            print(f"   {tag}iter {it:4d}   per-listener MSE {mse:.5f}   "
                  f"consensus {c['mse_cons']:.5f}   hayrat {Lh:.5f}")
    return hist


def evaluate(net, n=12, B=48, T=48, candle=0.0, seed=9000):
    tot = 0.0
    base = 0.0
    for i in range(n):
        U, Y = make_batch(B, T, Din=net.Din, seed=seed + i)
        pr, _, _ = net.forward(U, candle=candle, cache=False)
        tot += np.mean((pr - Y) ** 2)
        base += np.mean((U - Y) ** 2)          # persistence baseline
    return tot / n, base / n


# ==============================================================================
# SECTION 6 — THE EXPERIMENTS
# ==============================================================================

def test_charkh(net):
    """The turn must conserve norm exactly, for as long as you care to turn."""
    print("-" * 78)
    print("TEST 3  CHARKH — the wheel neither grows nor shrinks the state")
    print("-" * 78)
    Q, A, Mi = cayley_forward(net.S)
    orth = np.max(np.abs(Q.T @ Q - np.eye(net.D)))
    x = np.random.default_rng(0).normal(size=net.D)
    n0 = np.linalg.norm(x)
    for _ in range(20000):
        x = Q @ x
    drift = abs(np.linalg.norm(x) - n0) / n0
    ang = rotation_angles(Q)
    print(f"   ||Q^T Q - I||_max                    = {orth:.3e}")
    print(f"   relative norm drift over 20,000 turns = {drift:.3e}")
    print(f"   det(Q)                                = {np.linalg.det(Q):+.9f}")
    print(f"   learned turning rates (rad/step)      = "
          f"{np.array2string(ang, precision=3, max_line_width=110)}")
    ok = orth < 1e-10 and drift < 1e-10
    print(f"   {'PASS' if ok else 'FAIL'}\n")
    return ok, ang


def test_silence(net):
    """
    Silence in, silence out — and not approximately.

    The plate reading is a * q, where a = ||x||/(1+||x||). With the eps0 floor
    every mode leaks, so once the drive stops ||x|| falls at least as fast as
    exp(-min_stop * t) and the reading falls with it. A language model handed an
    empty prompt still emits. This cannot: it has nothing inside to emit.
    """
    print("-" * 78)
    print("TEST 4  SILENCE — remove the breath and the reed stops (Masnavi II.1)")
    print("-" * 78)
    T_on, T_off = 60, 2000
    U_on, _ = make_batch(1, T_on, Din=net.Din, seed=4242)
    U = np.concatenate([U_on, np.zeros((1, T_off, net.Din))], axis=1)
    pr, _, c = net.forward(U, candle=0.0)
    X = np.stack(c['X'], axis=1)[0]
    nrm = np.linalg.norm(X, axis=1)
    amp = np.linalg.norm(pr[0], axis=1)
    a_on = amp[:T_on].mean()
    _, pm, stops = net.pass_gains()
    bound = float(np.exp(-(stops.min())))       # per-step worst-case retention
    emp = (nrm[-1] / nrm[T_on - 1]) ** (1.0 / T_off)
    for k in (100, 500, 1000, 2000):
        print(f"   output magnitude {k:5d} steps after silence  = "
              f"{amp[T_on + k - 1]:.3e}   ({amp[T_on+k-1]/a_on:.2e} of driven)")
    print(f"   mean output magnitude while driven          = {a_on:.6f}")
    print(f"   theoretical per-step retention bound        = {bound:.6f}")
    print(f"   measured per-step retention                 = {emp:.6f}")
    a_end = amp[-1]
    ok = (a_end < 1e-6 * a_on) and (emp <= bound + 1e-9)
    print(f"   {'PASS' if ok else 'FAIL'}\n")
    return ok, a_on, a_end, bound, emp


def test_ney(net):
    """
    Content is stored as absence. Fill in the holes and the instrument forgets.
    """
    print("-" * 78)
    print("TEST 5  NEY — the burned holes are the memory (Masnavi I.1-3)")
    print("-" * 78)
    P, pm, stops = net.pass_gains()
    order = np.argsort(stops)[::-1]
    mse, base = evaluate(net, candle=0.0)
    g_keep = net.g.copy()
    # Fill every hole: the reed becomes a plain uncut tube.
    net.g = np.full_like(net.g, -30.0)
    mse_open, _ = evaluate(net, candle=0.0)
    net.g = g_keep
    print(f"   learned stop depths (deepest first) =")
    print("     " + np.array2string(stops[order], precision=3,
                                    max_line_width=100))
    print(f"   deepest hole at mode {order[0]:2d}  (transmission {pm[order[0]]:.4f})")
    print(f"   shallowest hole at mode {order[-1]:2d}  (transmission {pm[order[-1]]:.4f})")
    print(f"   MSE with the reed as learned (holes burned) = {mse:.5f}")
    print(f"   MSE with every hole filled  (plain tube)    = {mse_open:.5f}")
    print(f"   degradation from filling the holes          = "
          f"{100*(mse_open-mse)/mse:+.1f}%")
    ok = mse_open > mse
    print(f"   {'PASS' if ok else 'FAIL'} — absence carries the content\n")
    return ok, mse, mse_open, stops


def test_resonance(net, ang):
    """
    The instrument is exactly linear between the drive and the state:

        x_t = (P Q) x_{t-1} + Bin u_t

    so its transfer function is known in closed form,
        H(w) = (I - e^{-iw} P Q)^{-1} Bin,
    and its resonances sit exactly at the arguments of the eigenvalues of P Q —
    the turning rates of the wheel, after the reed's holes have shifted them.

    This matters for a reason that is not decorative. It means you cannot learn
    what this model knows by reading its weights; you learn it by PLAYING it and
    listening to which notes come back loud. The analytic curve is computed, the
    peaks are located, they are compared to the eigen-angles, and then the whole
    thing is verified against direct simulation at a handful of frequencies.
    """
    print("-" * 78)
    print("TEST 6  RESONANCE — the instrument answers at the rates it turns at")
    print("-" * 78)
    Q, _, _ = cayley_forward(net.S)
    P, pm, stops = net.pass_gains()
    R = P[:, None] * Q                                  # the actual resonator
    ev = np.linalg.eigvals(R)
    res_w = np.sort(np.abs(np.angle(ev[np.angle(ev) > 1e-9])))
    I = np.eye(net.D)
    e0 = np.zeros(net.Din); e0[0] = 1.0
    b = net.Bin @ e0

    ws = np.linspace(0.02, np.pi - 0.02, 4000)
    gain = np.array([np.linalg.norm(np.linalg.solve(I - np.exp(-1j*w) * R, b))
                     for w in ws])
    span = gain.max() - gain.min()
    pk = [i for i in range(1, len(ws) - 1)
          if gain[i] > gain[i - 1] and gain[i] >= gain[i + 1]
          and gain[i] - gain.min() > 0.10 * span]
    pw = ws[pk]
    print(f"   eigen-angles of the resonator P.Q :")
    print("     " + np.array2string(res_w, precision=4, max_line_width=100))
    print(f"   peaks of the analytic gain curve  :")
    print("     " + np.array2string(pw, precision=4, max_line_width=100))
    errs = [float(np.min(np.abs(res_w - w))) for w in pw]
    matched = sum(1 for e in errs if e < 0.06)
    frac = matched / max(len(pw), 1)
    print(f"   peaks within 0.06 rad of an eigen-angle : {matched}/{len(pw)} "
          f"({100*frac:.0f}%)")
    if errs:
        print(f"   median peak-to-eigen-angle distance     : {np.median(errs):.5f} rad")
    at_modes = float(np.mean([gain[int(np.argmin(np.abs(ws - w)))] for w in res_w]))
    print(f"   mean gain AT the eigen-angles           : {at_modes:.4f}")
    print(f"   mean gain across the whole sweep        : {gain.mean():.4f}  "
          f"({at_modes/gain.mean():.2f}x louder on its own notes)")
    print(f"   dynamic range of the sweep (max/min gain): {gain.max()/gain.min():.1f}x")
    print(f"   quietest note {ws[int(np.argmin(gain))]:.3f} rad/step, "
          f"loudest note {ws[int(np.argmax(gain))]:.3f} rad/step")

    # --- verify the closed form against actual simulation --------------------
    probe = list(pw[:3]) + [float(ws[int(np.argmin(gain))])]
    print("   verification against direct simulation (steady-state plate RMS):")
    sim, ana = [], []
    for w in probe:
        u = single_tone(900, w, Din=net.Din)
        _, _, c = net.forward(u, candle=0.0)
        Xs = np.stack(c['X'], axis=1)[0]
        sim.append(float(np.sqrt(np.mean(np.sum(Xs[400:]**2, axis=1)))))
        ana.append(float(np.linalg.norm(np.linalg.solve(I - np.exp(-1j*w)*R, b))/np.sqrt(2)))
        print(f"      w = {w:.4f}   simulated ||x||_rms = {sim[-1]:9.4f}   "
              f"predicted = {ana[-1]:9.4f}")
    sim, ana = np.array(sim), np.array(ana)
    agree = float(np.max(np.abs(sim - ana) / (np.abs(ana) + 1e-9)))
    print(f"   worst closed-form vs simulation disagreement : {agree:.3%}")
    ok = frac >= 0.8 and agree < 0.05 and at_modes > 1.15 * gain.mean()
    print(f"   {'PASS' if ok else 'FAIL'}\n")
    return ok, ws, gain, pw


def test_elephant(net):
    """
    Masnavi III.1259-1270. Four hands in a dark house, each on one part of the
    animal, each certain. III.1269: 'if there had been a candle in each one's
    hand, the difference would have gone out of their words.'

    Falsifiable form: with the same trained heads and the same input, does
    lighting the shared candle (a) collapse disagreement and (b) improve the
    consensus? Rumi says yes to both. We measure it.
    """
    print("-" * 78)
    print("TEST 7  ELEPHANT IN THE DARK — does the candle end the argument?")
    print("-" * 78)
    out = {}
    for label, cd in (("dark   (candle=0)", 0.0), ("lit    (candle=1)", 1.0)):
        dis, mse = [], []
        for i in range(10):
            U, Y = make_batch(48, 48, Din=net.Din, seed=7000 + i)
            pr, hp, _ = net.forward(U, candle=cd, cache=False)
            mse.append(np.mean((pr - Y) ** 2))
            # mean pairwise disagreement between the K conjectures
            K = net.K
            acc = []
            for a in range(K):
                for b in range(a + 1, K):
                    acc.append(np.mean((hp[:, :, a, :] - hp[:, :, b, :]) ** 2))
            dis.append(np.mean(acc))
        out[label] = (float(np.mean(mse)), float(np.mean(dis)))
        print(f"   {label}:  consensus MSE = {out[label][0]:.5f}   "
              f"pairwise disagreement = {out[label][1]:.5f}")
    d0 = out["dark   (candle=0)"]; d1 = out["lit    (candle=1)"]
    print(f"   disagreement removed by the candle = "
          f"{100*(d0[1]-d1[1])/d0[1]:+.1f}%")
    print(f"   consensus error changed by         = "
          f"{100*(d1[0]-d0[0])/d0[0]:+.1f}%")
    ok = d1[1] < d0[1] and d1[0] < d0[0]
    print(f"   {'PASS' if ok else 'PARTIAL'} — Rumi predicted both effects\n")
    return ok, d0, d1


def test_hayrat():
    """
    'Sell your cleverness and buy bewilderment' (Masnavi IV) is a TRADE, and
    Rumi states the exchange rate in the next hemistich: cleverness is zann
    (conjecture), bewilderment is nazar (direct seeing). So the useful question
    is not whether the floor is good. It is what a unit of bewilderment costs in
    cleverness, and what it buys.

    Four models, identical seed and data, differing only in lam_h. Measured:
      - turning rate: how much the state moves per step on the sphere
      - in-band MSE: performance on the drive band it was trained on
      - out-of-band MSE: performance on a drive four times slower than anything
        it ever saw, i.e. exactly the regime where a settled model is punished
    """
    print("-" * 78)
    print("TEST 8  HAYRAT — the exchange rate between cleverness and bewilderment")
    print("-" * 78)
    print(f"   {'lam_h':>7s} {'turning rate':>14s} {'in-band MSE':>13s} "
          f"{'out-of-band MSE':>17s}")
    res = {}
    for lam in (0.0, 0.15, 0.6, 2.0):
        n = NeyAstrolabe(D=24, Din=4, K=4, Cs=12, seed=5)
        train(n, iters=400, B=48, T=40, lr=6e-3, lam_h=lam, tau=0.80,
              seed=321, verbose=False)
        U, Y = make_batch(64, 64, Din=4, seed=88888)
        _, _, c = n.forward(U, candle=0.0)
        V = np.stack(c['V'], axis=1)
        turn = float(np.mean(np.sum((V[:, 1:] - V[:, :-1]) ** 2, axis=2)))
        mse_in, _ = evaluate(n, candle=0.0)
        Uo, Yo = make_batch(64, 48, Din=4, seed=555, wlo=0.05, whi=0.20)
        pr, _, _ = n.forward(Uo, candle=0.0, cache=False)
        mse_out = float(np.mean((pr - Yo) ** 2))
        res[lam] = (turn, float(mse_in), mse_out)
        print(f"   {lam:7.2f} {turn:14.5f} {mse_in:13.5f} {mse_out:17.5f}")
    t0, t_max = res[0.0][0], res[2.0][0]
    m0, m_max = res[0.0][1], res[2.0][1]
    print(f"   bewilderment bought : turning rate {t0:.4f} -> {t_max:.4f} "
          f"({100*(t_max-t0)/t0:+.1f}%)")
    print(f"   cleverness sold     : in-band MSE  {m0:.5f} -> {m_max:.5f} "
          f"({100*(m_max-m0)/m0:+.1f}%)")
    ok = res[2.0][0] > res[0.0][0]
    print(f"   {'PASS' if ok else 'FAIL'} — the price is real and it is payable\n")
    return ok, res


# ==============================================================================
# SECTION 7 — MAIN
# ==============================================================================

def main():
    np.set_printoptions(suppress=True)
    t0 = time.time()
    print("=" * 78)
    print(" 0268 · RUMI · THE NEY-ASTROLABE ENGINE")
    print(" nothing is stored; something is removed; and it only sounds when blown")
    print("=" * 78 + "\n")

    worst = gradient_check()

    print("-" * 78)
    print("TEST 2  TRAINING — driven next-sample prediction")
    print("-" * 78)
    net = NeyAstrolabe(D=24, Din=4, K=4, Cs=4, seed=5)
    mse0, base0 = evaluate(net, candle=0.0)
    print(f"   before training : MSE {mse0:.5f}   persistence baseline {base0:.5f}")
    hist = train(net, iters=700, B=64, T=48, lr=6e-3, lam_h=0.05, tau=0.80,
                 seed=100, verbose=True, tag="")
    mse1, base1 = evaluate(net, candle=0.0)
    print(f"   after  training : MSE {mse1:.5f}   persistence baseline {base1:.5f}")
    print(f"   improvement over its own start : {100*(mse0-mse1)/mse0:.1f}%")
    print(f"   improvement over persistence   : {100*(base1-mse1)/base1:.1f}%")
    ok_train = mse1 < 0.5 * base1
    print(f"   {'PASS' if ok_train else 'FAIL'}\n")

    ok_ch, ang = test_charkh(net)
    ok_si, a_on, a_end, bound, emp = test_silence(net)
    ok_ny, mse_h, mse_open, stops = test_ney(net)
    ok_rs, ws, gain, pw = test_resonance(net, ang)
    ok_el, d0, d1 = test_elephant(net)
    ok_hy, hres = test_hayrat()

    print("=" * 78)
    print(" SUMMARY")
    print("=" * 78)
    rows = [
        ("gradient check (all blocks)", f"worst rel err {worst:.2e}", True),
        ("training beats persistence", f"{mse1:.5f} vs {base1:.5f}", ok_train),
        ("charkh conserves the norm", f"drift < 1e-10 over 20k turns", ok_ch),
        ("silence in -> silence out", f"{a_end:.2e} after 2000 silent steps", ok_si),
        ("ney: absence carries content",
         f"+{100*(mse_open-mse_h)/mse_h:.1f}% error when holes filled", ok_ny),
        ("resonance = turning rates", f"{len(pw)} peaks recovered", ok_rs),
        ("candle collapses disagreement",
         f"{100*(d0[1]-d1[1])/d0[1]:+.1f}% disagreement", ok_el),
        ("hayrat: bewilderment is bought",
         f"turning {hres[0.0][0]:.4f} -> {hres[2.0][0]:.4f}", ok_hy),
    ]
    for name, val, ok in rows:
        print(f"  [{'OK' if ok else '--'}]  {name:<32s} {val}")
    allok = all(r[2] for r in rows)
    print("-" * 78)
    print(f"  {'ALL CHECKS PASSED' if allok else 'SOME CHECKS DID NOT PASS'}"
          f"   ({time.time()-t0:.1f}s)")
    print("=" * 78)
    return 0 if allok else 2


if __name__ == "__main__":
    sys.exit(main())

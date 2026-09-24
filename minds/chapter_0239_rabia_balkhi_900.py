#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 KAMAND  —  The Tightening Tie
 Chapter 0239 · Rabia Balkhi (Rabi'a bint Ka'b al-Quzdari)
 fl. first half of the 10th century CE, Balkh, Khurasan (Samanid realm)
 ========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 12 Minds 221 - 240 Available on Amazon https://www.amazon.com/dp/B0HKVBKNTT
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0239_rabia_balkhi_900 - Rabia Balkhi (Rabi'a bint Ka'b al-Quzdari)
================================================================================  

WHAT THIS FILE IS
-----------------
A from-scratch, pure-NumPy recurrent architecture (no autograd, no frameworks,
every gradient derived by hand and verified by finite differences) whose unit
of computation is not a weighted sum but a TIE: a bond with a tightness that
evolves according to the one dynamical law that survives in Rabia's own words.

Only some sixty couplets are attributed to her, scattered through later
dictionaries, anthologies and hagiographies. Four of them carry a theory of
mind precise enough to be written as equations. Translations are ours.

 1. THE LASSO (from the ghazal "His love has brought me into bonds again",
    quoted without her name by Abu Sa'id in Asrar al-tawhid; one couplet of
    this ghazal is assigned to Abu al-Hasan Aghaji in Asadi's lexicon):

      tawsani kardam, nadanistam hami
      k-az kashidan sakht-tar gardad kamand      (variant reading: tang-tar)

      "I bucked like an unbroken colt; I did not know
       that pulling makes the lasso tighter."

    In a tie, effort has the WRONG SIGN. Resistance feeds the bond it resists,
    and back-and-forth struggle ("much striving availed nothing") ratchets it.
    -> the Kamand cell: log-tightness obeys a capstan-like ratchet driven by
       the unit's own effort-away-from-its-anchor, with a thrash term that
       tightens on ANY movement, and a release that works only on slack.

 2. SEEING AND DEEMING (same ghazal):

      zisht bayad did u angarid khub
      zahr bayad khward u angarid qand

      "One must see the ugly and deem it fair,
       swallow poison and deem it sugar."

    The verbs are split: SEE (did) and DEEM (angarid). The perception is kept;
    only the pull is withdrawn.
    -> a separate body register that records cost truthfully while the heart
       is shielded from feeling cost as a pull; and a counterfactual 'sugar'
       variant in which deeming is allowed to bend perception itself.

 3. BODY AND HEART (the earliest firmly attested couplets, al-Raduyani):

      kashk tanam baz yafti khabar-i dil
      kashk dilam baz yafti khabar-i tan

      "Would that my body could once more find news of my heart;
       would that my heart could once more find news of my body."

    -> an explicit news channel (khabar) from body register to the heart's
       release gate; the 'severed' variant removes it and cannot let go.

 4. THE CURSE (attributed in later compilations):

      chun ba-hijr andar bipichi, pas bidani qadr-i man

      "When you twist in separation, then you will know my worth" — a curse
       that the unkind one should love someone unkind "like yourself".

    -> worth estimated by RE-ENACTMENT: running another's visible effort
       through one's own tie dynamics. It works only if one has been tied,
       and only for an other whose dynamics are like one's own.

THE KAMAND CELL (per unit, per step)
------------------------------------
    d_t   = g_t * (W_x x_t) + U h_{t-1} + b              drive, input-gated
    h~_t  = (1 - lam) h_{t-1} + lam tanh(d_t)            free intention
    e_t   = ((h~_t - a_{t-1})^2 - (h_{t-1} - a_{t-1})^2)/2   effort AWAY from anchor
    l_t   = L_{t-1} + alpha*ramp+(e) + delta*|e|_eps     capstan ratchet
                    - beta * r_t * slack(e) * (L_{t-1} - l_min)
    rho_t = sigmoid(l_t)                                  tightness
    h_t   = (1 - rho_t) h~_t + rho_t a_{t-1}              the tie reclaims
    a_t   = a_{t-1} + w_t (1 - rho_t)(h_t - a_{t-1})      knots re-tie only on slack
    L_t   = l_t + w_t (1 - rho_t)(l_throw - l_t)          a new throw sets the tie

  ramp+(e) = (sqrt(e^2+eps^2) - eps + e)/2      (smooth max(e,0))
  |e|_eps  =  sqrt(e^2+eps^2) - eps             (smooth |e|)
  slack(e) = sigmoid(-kappa e)                  (~1 only when not pulling)

  e_t is the intended change in (half) squared distance from the anchor: positive
  when the unit means to go farther from where it is tied, negative when it yields
  toward it. Since ramp+(e) - ramp-(e) = e and ramp+(e) + ramp-(e) = |e|_eps, the
  tightening is a DIRECTIONAL term (pulling away) plus a THRASH TAX (any movement).
  For any struggle made of equal and opposite efforts, ramp+(e) + ramp+(-e) =
  |e|_eps >= 0: a symmetric struggle can never loosen the tie, and with delta > 0
  it strictly tightens it. That is the verse. (The smoothing lets ramp+ dip to
  -eps/2 at worst; with eps = 0.01 this is negligible and is tested.)

GATES
  g (input gate: whether a pull is let in), r (the hand: release),
  w (the throw: a new tie). In AWARE minds the cell's own tightness rho_{t-1}
  is fed back into g and r: the mind knows how tight its tie is. In BLIND
  minds it is not — "I did not know".

EXPERIMENTS
  [1] The lasso law at a single cell (deterministic demonstrations)
  [2] Structural self-tests
  [3] Finite-difference gradient checks for every trainable model (mandatory)
  [4] Task I   CARRY TO THE END: commitment under pull, legitimate release,
               out-of-distribution pulls, pushes through an unseen channel,
               and gentle persistent pressure (drift); KAMAND-aware vs
               KAMAND-blind vs KAMAND-place (news of place: unconsented
               displacement also tightens) vs leaky RNN vs gated RNN; a
               zero-shot strain signal; the anatomy of the drift failure
  [5] Task II  BODY AND HEART: carrying through cost, letting go when cost is
               real; khabar vs severed vs sugar
  [6] Task III THE CURSE: estimating another's endured tightness by re-enactment

KNOWN LIMITS (reported, not hidden)
  The gated RNN baseline matches KAMAND on in-distribution accuracy and is
  more robust to pushes through unused channels. KAMAND's distinctive gains are
  robustness to forceful unconsented pulls and a legible strain signal; its
  distinctive weakness is gentle, persistent pressure, which the lasso law does
  not answer because each step carries too little effort. The news-of-place
  amendment narrows but does not close that gap.

CONVENTIONS KEPT FROM THE CORPUS
  pure NumPy float64, hand-derived analytic gradients, finite-difference check
  that must pass, real training loops on fresh samples, held-out evaluation,
  self-tests on structural invariants, printed output is the verified output.

RUN:     python3 chapter_0239_rabia_balkhi_900.py
QUICK:   python3 chapter_0239_rabia_balkhi_900.py --quick   (short training)
================================================================================
"""

import sys
import time
import numpy as np

# ==============================================================================
# 0.  CONSTANTS AND SMALL UTILITIES
# ==============================================================================

P_BITS = 6                 # size of an attachment pattern (the 'beloved')
CH_PAT = slice(0, P_BITS)  # pattern channel
CH_TIE = 6                 # the throw: a cue to tie
CH_HAND = 7                # the hand: a cue that release is consented
CH_PSN = 8                 # poison: a cost signal (Task II only)
CH_NOISE = slice(9, 11)    # two noise channels (an unseen route for attacks)
D_IN = 11

EPS_R = 0.01               # smoothing of the ramps
KAPPA = 30.0               # sharpness of the slack indicator
L_MIN = -4.0               # loosest possible tie (rho ~ 0.018)
BETA_MAX = 0.5             # the hand cannot open a knot faster than this


def sigmoid(z):
    """Numerically stable logistic for any shape (including 0-d)."""
    z = np.asarray(z, dtype=np.float64)
    ez = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + ez), ez / (1.0 + ez))


def softplus(z):
    return np.logaddexp(0.0, z)


def inv_softplus(y):
    return float(y + np.log(-np.expm1(-y)))


def logit(p):
    return float(np.log(p / (1.0 - p)))


def ramps(e):
    """Smooth positive ramp, smooth absolute value, and their shared root."""
    sq = np.sqrt(e * e + EPS_R ** 2)
    return 0.5 * (sq - EPS_R + e), sq - EPS_R, sq


def auc(pos, neg):
    """Rank AUC: probability that a random positive outscores a random negative."""
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    vals, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(vals))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


class Adam:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads, clip=1.0):
        norm = np.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
        scale = min(1.0, clip / (norm + 1e-12))
        self.t += 1
        for k in params:
            g = grads[k] * scale
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)
        return norm


# ==============================================================================
# 1.  THE KAMAND CELL, ONE UNIT AT A TIME  (the law, without learning)
# ==============================================================================

def tie_step(L, e, r, alpha, delta, beta):
    """One ratchet step for a scalar (or vector) tie. Returns new log-tightness."""
    pp, ab, _ = ramps(e)
    sl = sigmoid(-KAPPA * e)
    return L + alpha * pp + delta * ab - beta * r * sl * (L - L_MIN)


def single_cell_rollout(drive_targets, hand, h0, a0, L0, lam=0.5, alpha=1.5,
                        delta=0.3, beta=0.35, gate=None, throw=None, lthrow=1.0):
    """
    Roll ONE Kamand unit forward under an externally imposed intention.
    drive_targets[t] is the value tanh(d_t) the unit is pushed toward;
    gate[t]  (optional) scales how much of that push is let in (0 = hold still);
    hand[t]  is the release gate r_t;
    throw[t] (optional) is the throw gate w_t: a new tie, which sets only on slack.
    Returns traces of h, a (after the step), L and rho.
    """
    T = len(drive_targets)
    h, a, L = float(h0), float(a0), float(L0)
    H, A, LL, R = [], [], [], []
    for t in range(T):
        gt = 1.0 if gate is None else gate[t]
        target = gt * drive_targets[t] + (1 - gt) * h      # a closed gate holds still
        ht = (1 - lam) * h + lam * target
        e = 0.5 * ((ht - a) ** 2 - (h - a) ** 2)
        ell = float(tie_step(L, e, hand[t], alpha, delta, beta))
        rho = float(sigmoid(ell))
        h = (1 - rho) * ht + rho * a
        wt = 0.0 if throw is None else throw[t]
        v = wt * (1 - rho)
        a = a + v * (h - a)
        L = ell + v * (lthrow - ell)
        H.append(h); A.append(a); LL.append(L); R.append(float(sigmoid(L)))
    return np.array(H), np.array(A), np.array(LL), np.array(R)


# ==============================================================================
# 2.  THE KAMAND NETWORK  (heart of tie cells, optional body register)
# ==============================================================================

class KamandNet:
    """
    heart : N Kamand cells (the tie-holding register)
    body  : M ledger cells  (the register that SEES cost), optional
    aware : tightness rho_{t-1} is fed to the input gate and to the hand
    body_mode:
        None      no body (Task I)
        'khabar'  body integrates raw cost; its state is news to the heart's hand
        'severed' body integrates raw cost; no news reaches the heart
        'sugar'   body integrates cost discounted by the heart's tightness
                  (deeming bends seeing); news reaches the heart
    In every body mode the heart's own input has the cost channel removed:
    the heart never feels cost as a pull. That is the 'deem' half of the verse.
    """

    def __init__(self, N=24, M=6, aware=True, body_mode=None, place_news=False, seed=0):
        rng = np.random.default_rng(seed)
        self.N, self.M = N, M
        self.aware, self.body_mode = aware, body_mode
        self.place_news = place_news
        D, P = D_IN, P_BITS
        p = {}
        p["Wx"] = rng.normal(0, 1.0 / np.sqrt(D), (N, D))
        p["Uh"] = rng.normal(0, 0.5 / np.sqrt(N), (N, N))
        p["bh"] = np.zeros(N)
        p["Wg"] = rng.normal(0, 0.3 / np.sqrt(D), (N, D))
        p["bg"] = np.full(N, 2.0)                       # input gate starts open
        p["Wr"] = rng.normal(0, 0.3 / np.sqrt(D), (N, D))
        p["Wr"][:, CH_HAND] = 4.0                       # prior: the hand cue opens the hand
        p["br"] = np.full(N, -4.0)                      # hand starts closed
        p["Ww"] = rng.normal(0, 0.3 / np.sqrt(D), (N, D))
        p["Ww"][:, CH_TIE] = 8.0                        # prior: the throw cue ties
        p["bw"] = np.full(N, -4.0)
        if aware:
            p["Vg"] = rng.normal(0, 0.1, (N, N))
            p["Vr"] = rng.normal(0, 0.1, (N, N))
        p["lam_raw"] = np.zeros(N)
        p["alpha_raw"] = np.full(N, inv_softplus(1.5))
        p["delta_raw"] = np.full(N, inv_softplus(0.3))
        p["beta_raw"] = np.full(N, logit(0.6))
        p["lthrow"] = np.full(N, 1.0)
        if place_news:
            p["gam_raw"] = np.full(N, inv_softplus(1.0))   # news of place: unconsented displacement tightens
        p["Wo"] = rng.normal(0, 0.1, (P, N))
        p["bo"] = np.zeros(P)
        if body_mode is not None:
            p["mu_raw"] = np.full(M, logit(0.02))        # a near-perfect integrator
            p["gs"] = 0.5 + rng.normal(0, 0.1, M)
            p["wl"] = rng.normal(0, 0.1, M)
            p["bl"] = np.zeros(1)
            p["wb"] = np.full(1, 10.0)                    # prior: bound iff mean tightness > 1/2
            p["bb"] = np.full(1, -5.0)
            if body_mode in ("khabar", "sugar"):
                p["Yr"] = rng.normal(0, 0.1, (N, M))
        self.p = p
        self.name = "KAMAND"

    # ---------------------------------------------------------------- forward
    def forward(self, X, keep=True):
        p = self.p
        B, T, _ = X.shape
        N, M = self.N, self.M
        lam = sigmoid(p["lam_raw"])
        alpha = softplus(p["alpha_raw"])
        delta = softplus(p["delta_raw"])
        beta = BETA_MAX * sigmoid(p["beta_raw"])
        gam = softplus(p["gam_raw"]) if self.place_news else None
        body = self.body_mode is not None
        news = self.body_mode in ("khabar", "sugar")
        h = np.zeros((B, N))
        a = np.zeros((B, N))
        L = np.full((B, N), L_MIN)
        s = np.zeros((B, M)) if body else None
        mu = sigmoid(p["mu_raw"]) if body else None
        caches = []
        rbar_trace = np.zeros((B, T))
        h_trace = np.zeros((B, T, N))
        rho = None
        for t in range(T):
            x = X[:, t, :]
            if body:
                xh = x.copy()
                xh[:, CH_PSN] = 0.0            # the heart does not feel cost as a pull
            else:
                xh = x
            rp = sigmoid(L)
            zg = xh @ p["Wg"].T + p["bg"]
            zr = xh @ p["Wr"].T + p["br"]
            if self.aware:
                zg = zg + rp @ p["Vg"].T
                zr = zr + rp @ p["Vr"].T
            if news:
                zr = zr + s @ p["Yr"].T
            g = sigmoid(zg)
            r = sigmoid(zr)
            w = sigmoid(xh @ p["Ww"].T + p["bw"])
            xW = xh @ p["Wx"].T
            d = xW * g + h @ p["Uh"].T + p["bh"]
            c = np.tanh(d)
            ht = (1 - lam) * h + lam * c
            e = 0.5 * ((ht - a) ** 2 - (h - a) ** 2)
            pp, ab, sq = ramps(e)
            sl = sigmoid(-KAPPA * e)
            Lm = L - L_MIN
            ell = L + alpha * pp + delta * ab - beta * r * sl * Lm
            dist = (h - a) ** 2 if self.place_news else None
            if self.place_news:
                ell = ell + gam * (1 - r) * dist
            rho = sigmoid(ell)
            hn = (1 - rho) * ht + rho * a
            v = w * (1 - rho)
            an = a + v * (hn - a)
            Ln = ell + v * (p["lthrow"] - ell)
            rbar = rho.mean(axis=1)
            rbar_trace[:, t] = rbar
            h_trace[:, t, :] = hn
            sn = None
            perc = None
            if body:
                psn = x[:, CH_PSN]
                perc = psn * (1 - rbar) if self.body_mode == "sugar" else psn
                sn = (1 - mu) * s + perc[:, None] * p["gs"]
            if keep:
                caches.append(dict(xh=xh, h=h, a=a, L=L, s=s, rp=rp, g=g, r=r, w=w,
                                   xW=xW, c=c, ht=ht, e=e, sq=sq, pp=pp, ab=ab, sl=sl,
                                   Lm=Lm, ell=ell, rho=rho, hn=hn, v=v, dist=dist,
                                   psn=(x[:, CH_PSN] if body else None), perc=perc))
            h, a, L = hn, an, Ln
            if body:
                s = sn
        out = dict(h=h, a=a, L=L, s=s, rho_last=rho, rbar_trace=rbar_trace, h_trace=h_trace)
        return out, caches

    def heads(self, out):
        p = self.p
        res = dict(z=out["h"] @ p["Wo"].T + p["bo"])
        if self.body_mode is not None:
            res["zb"] = p["wb"][0] * out["rho_last"].mean(axis=1) + p["bb"][0]
            res["yl"] = out["s"] @ p["wl"] + p["bl"][0]
        return res

    # ------------------------------------------------------- loss + backward
    def loss_and_grad(self, X, Y, mask_pat=None, bound=None, ledger=None,
                      weights=(1.0, 1.0, 2.0)):
        p = self.p
        B, T, _ = X.shape
        N, M, P = self.N, self.M, P_BITS
        body = self.body_mode is not None
        news = self.body_mode in ("khabar", "sugar")
        out, caches = self.forward(X, keep=True)
        H = self.heads(out)
        grads = {k: np.zeros_like(v) for k, v in p.items()}

        # pattern head (binary cross-entropy per bit)
        yt = (Y + 1.0) / 2.0
        mask = np.ones((B, 1)) if mask_pat is None else mask_pat[:, None]
        denom = max(float(mask.sum()) * P, 1.0)
        z = H["z"]
        loss_pat = float(np.sum(mask * (softplus(z) - yt * z)) / denom)
        dz = weights[0] * mask * (sigmoid(z) - yt) / denom
        grads["Wo"] += dz.T @ out["h"]
        grads["bo"] += dz.sum(axis=0)
        dh_T = dz @ p["Wo"]
        loss = weights[0] * loss_pat
        parts = dict(pattern=loss_pat)

        drho_T = None
        ds_T = np.zeros((B, M)) if body else None
        if body and bound is not None:
            zb = H["zb"]
            loss_b = float(np.mean(softplus(zb) - bound * zb))
            dzb = weights[1] * (sigmoid(zb) - bound) / B
            rbarT = out["rho_last"].mean(axis=1)
            grads["wb"][0] += float(np.sum(dzb * rbarT))
            grads["bb"][0] += float(np.sum(dzb))
            drho_T = np.repeat((dzb * p["wb"][0] / N)[:, None], N, axis=1)
            loss += weights[1] * loss_b
            parts["bound"] = loss_b
        if body and ledger is not None:
            yl = H["yl"]
            loss_l = float(np.mean((yl - ledger) ** 2))
            dyl = weights[2] * 2.0 * (yl - ledger) / B
            grads["wl"] += dyl @ out["s"]
            grads["bl"][0] += float(np.sum(dyl))
            ds_T = dyl[:, None] * p["wl"][None, :]
            loss += weights[2] * loss_l
            parts["ledger"] = loss_l

        lam = sigmoid(p["lam_raw"])
        alpha = softplus(p["alpha_raw"])
        delta = softplus(p["delta_raw"])
        sbeta = sigmoid(p["beta_raw"])
        beta = BETA_MAX * sbeta
        mu = sigmoid(p["mu_raw"]) if body else None
        g_lam = np.zeros(N)
        g_alpha = np.zeros(N)
        g_delta = np.zeros(N)
        gam = softplus(p["gam_raw"]) if self.place_news else None
        g_gam = np.zeros(N)
        g_beta = np.zeros(N)
        g_mu = np.zeros(M) if body else None

        dhn = dh_T
        dan = np.zeros((B, N))
        dLn = np.zeros((B, N))
        dsn = ds_T

        for t in reversed(range(T)):
            cc = caches[t]
            xh, h, a, L = cc["xh"], cc["h"], cc["a"], cc["L"]
            g, r, w, xW, c, ht = cc["g"], cc["r"], cc["w"], cc["xW"], cc["c"], cc["ht"]
            e, sq, pp, ab, sl = cc["e"], cc["sq"], cc["pp"], cc["ab"], cc["sl"]
            Lm, ell, rho, hn, v, rp = cc["Lm"], cc["ell"], cc["rho"], cc["hn"], cc["v"], cc["rp"]

            # L_t = ell + v (lthrow - ell)
            dell = dLn * (1 - v)
            dv = dLn * (p["lthrow"] - ell)
            grads["lthrow"] += np.sum(dLn * v, axis=0)
            # a_t = a + v (h_t - a)
            da = dan * (1 - v)
            dhn_tot = dhn + dan * v
            dv = dv + dan * (hn - a)
            # v = w (1 - rho)
            dw = dv * (1 - rho)
            drho = -dv * w
            # h_t = (1 - rho) h~ + rho a
            dht = dhn_tot * (1 - rho)
            da = da + dhn_tot * rho
            drho = drho + dhn_tot * (a - ht)
            # body register
            ds_prev = None
            if body:
                s_prev = cc["s"]
                ds_prev = dsn * (1 - mu)
                g_mu += np.sum(dsn * (-s_prev), axis=0)
                grads["gs"] += np.sum(dsn * cc["perc"][:, None], axis=0)
                dperc = dsn @ p["gs"]
                if self.body_mode == "sugar":
                    drbar = -dperc * cc["psn"]
                    drho = drho + (drbar / N)[:, None]
            if t == T - 1 and drho_T is not None:
                drho = drho + drho_T
            # rho = sigmoid(ell)
            dell = dell + drho * rho * (1 - rho)
            # ell = L + alpha pp + delta ab - beta r sl Lm  [+ gam (1 - r) (h - a)^2]
            dL = dell * (1 - beta * r * sl)
            g_alpha += np.sum(dell * pp, axis=0)
            g_delta += np.sum(dell * ab, axis=0)
            g_beta += np.sum(dell * (-r * sl * Lm), axis=0)
            dr = dell * (-beta * sl * Lm)
            dh_place = None
            if self.place_news:
                dist = cc["dist"]
                g_gam += np.sum(dell * (1 - r) * dist, axis=0)
                dr = dr - dell * gam * dist
                ddist = dell * gam * (1 - r)
                dh_place = ddist * 2.0 * (h - a)
            dsl = dell * (-beta * r * Lm)
            dpp = dell * alpha
            dab = dell * delta
            # sl = sigmoid(-kappa e); pp, ab from sq = sqrt(e^2 + eps^2)
            de = dsl * sl * (1 - sl) * (-KAPPA)
            de = de + dpp * 0.5 * (e / sq + 1.0) + dab * (e / sq)
            # e = ((h~ - a)^2 - (h - a)^2) / 2
            dht = dht + de * (ht - a)
            dh = de * (a - h)
            da = da + de * (h - ht)
            if dh_place is not None:
                dh = dh + dh_place
                da = da - dh_place
            # h~ = (1 - lam) h + lam c
            dh = dh + dht * (1 - lam)
            dc = dht * lam
            g_lam += np.sum(dht * (c - h), axis=0)
            # c = tanh(d)
            dd = dc * (1 - c * c)
            # d = xW g + h U^T + b
            grads["Wx"] += (dd * g).T @ xh
            dg = dd * xW
            grads["Uh"] += dd.T @ h
            dh = dh + dd @ p["Uh"]
            grads["bh"] += dd.sum(axis=0)
            # gates
            dzg = dg * g * (1 - g)
            grads["Wg"] += dzg.T @ xh
            grads["bg"] += dzg.sum(axis=0)
            dzr = dr * r * (1 - r)
            grads["Wr"] += dzr.T @ xh
            grads["br"] += dzr.sum(axis=0)
            dzw = dw * w * (1 - w)
            grads["Ww"] += dzw.T @ xh
            grads["bw"] += dzw.sum(axis=0)
            if self.aware:
                grads["Vg"] += dzg.T @ rp
                grads["Vr"] += dzr.T @ rp
                drp = dzg @ p["Vg"] + dzr @ p["Vr"]
                dL = dL + drp * rp * (1 - rp)
            if news:
                grads["Yr"] += dzr.T @ cc["s"]
                ds_prev = ds_prev + dzr @ p["Yr"]
            dhn, dan, dLn = dh, da, dL
            if body:
                dsn = ds_prev

        grads["lam_raw"] = g_lam * lam * (1 - lam)
        grads["alpha_raw"] = g_alpha * sigmoid(p["alpha_raw"])
        grads["delta_raw"] = g_delta * sigmoid(p["delta_raw"])
        if self.place_news:
            grads["gam_raw"] = g_gam * sigmoid(p["gam_raw"])
        grads["beta_raw"] = g_beta * BETA_MAX * sbeta * (1 - sbeta)
        if body:
            grads["mu_raw"] = g_mu * mu * (1 - mu)
        return loss, grads, parts

    def loss_only(self, X, Y, mask_pat=None, bound=None, ledger=None, weights=(1.0, 1.0, 2.0)):
        out, _ = self.forward(X, keep=False)
        H = self.heads(out)
        B = X.shape[0]
        yt = (Y + 1.0) / 2.0
        mask = np.ones((B, 1)) if mask_pat is None else mask_pat[:, None]
        denom = max(float(mask.sum()) * P_BITS, 1.0)
        z = H["z"]
        loss = weights[0] * float(np.sum(mask * (softplus(z) - yt * z)) / denom)
        if self.body_mode is not None and bound is not None:
            zb = H["zb"]
            loss += weights[1] * float(np.mean(softplus(zb) - bound * zb))
        if self.body_mode is not None and ledger is not None:
            loss += weights[2] * float(np.mean((H["yl"] - ledger) ** 2))
        return loss

    def tie_parameters(self):
        p = self.p
        out = dict(alpha=softplus(p["alpha_raw"]), delta=softplus(p["delta_raw"]),
                   beta=BETA_MAX * sigmoid(p["beta_raw"]), lam=sigmoid(p["lam_raw"]),
                   lthrow=p["lthrow"].copy())
        if self.place_news:
            out["gamma"] = softplus(p["gam_raw"])
        return out


# ==============================================================================
# 3.  BASELINES  (same width, same data, same optimiser)
# ==============================================================================

class LeakyRNN:
    """h_t = (1 - lam) h + lam tanh(W x + U h + b). No tie, no gate."""

    def __init__(self, N=24, seed=0):
        rng = np.random.default_rng(seed)
        D, P = D_IN, P_BITS
        self.N = N
        self.p = dict(Wx=rng.normal(0, 1.0 / np.sqrt(D), (N, D)),
                      Uh=rng.normal(0, 1.0 / np.sqrt(N), (N, N)),
                      bh=np.zeros(N), lam_raw=np.full(N, logit(0.3)),
                      Wo=rng.normal(0, 0.1, (P, N)), bo=np.zeros(P))
        self.name = "LEAKY"

    def forward(self, X, keep=True):
        p = self.p
        B, T, _ = X.shape
        lam = sigmoid(p["lam_raw"])
        h = np.zeros((B, self.N))
        caches = []
        h_trace = np.zeros((B, T, self.N))
        for t in range(T):
            x = X[:, t, :]
            c = np.tanh(x @ p["Wx"].T + h @ p["Uh"].T + p["bh"])
            hn = (1 - lam) * h + lam * c
            if keep:
                caches.append((x, h, c))
            h = hn
            h_trace[:, t, :] = h
        return dict(h=h, h_trace=h_trace), caches

    def loss_and_grad(self, X, Y, **kw):
        p = self.p
        B, T, _ = X.shape
        out, caches = self.forward(X)
        z = out["h"] @ p["Wo"].T + p["bo"]
        yt = (Y + 1) / 2
        denom = B * P_BITS
        loss = float(np.sum(softplus(z) - yt * z) / denom)
        dz = (sigmoid(z) - yt) / denom
        grads = {k: np.zeros_like(v) for k, v in p.items()}
        grads["Wo"] += dz.T @ out["h"]
        grads["bo"] += dz.sum(0)
        dh = dz @ p["Wo"]
        lam = sigmoid(p["lam_raw"])
        g_lam = np.zeros(self.N)
        for t in reversed(range(T)):
            x, h, c = caches[t]
            g_lam += np.sum(dh * (c - h), axis=0)
            dd = dh * lam * (1 - c * c)
            grads["Wx"] += dd.T @ x
            grads["Uh"] += dd.T @ h
            grads["bh"] += dd.sum(0)
            dh = dh * (1 - lam) + dd @ p["Uh"]
        grads["lam_raw"] = g_lam * lam * (1 - lam)
        return loss, grads, dict(pattern=loss)

    def loss_only(self, X, Y, **kw):
        out, _ = self.forward(X, keep=False)
        z = out["h"] @ self.p["Wo"].T + self.p["bo"]
        yt = (Y + 1) / 2
        return float(np.sum(softplus(z) - yt * z) / (X.shape[0] * P_BITS))

    def heads(self, out):
        return dict(z=out["h"] @ self.p["Wo"].T + self.p["bo"])


class GatedRNN:
    """Update-gate recurrent unit: h_t = (1 - z) h + z tanh(W x + U h + b).
    Given the same cue prior as KAMAND: the tie and hand cues open the gate."""

    def __init__(self, N=24, seed=0):
        rng = np.random.default_rng(seed)
        D, P = D_IN, P_BITS
        self.N = N
        Wz = rng.normal(0, 0.3 / np.sqrt(D), (N, D))
        Wz[:, CH_TIE] = 6.0
        Wz[:, CH_HAND] = 6.0
        self.p = dict(Wz=Wz, Uz=rng.normal(0, 0.3 / np.sqrt(N), (N, N)), bz=np.full(N, -4.0),
                      Wx=rng.normal(0, 1.0 / np.sqrt(D), (N, D)),
                      Uh=rng.normal(0, 1.0 / np.sqrt(N), (N, N)), bh=np.zeros(N),
                      Wo=rng.normal(0, 0.1, (P, N)), bo=np.zeros(P))
        self.name = "GATED"

    def forward(self, X, keep=True):
        p = self.p
        B, T, _ = X.shape
        h = np.zeros((B, self.N))
        caches = []
        h_trace = np.zeros((B, T, self.N))
        for t in range(T):
            x = X[:, t, :]
            zg = sigmoid(x @ p["Wz"].T + h @ p["Uz"].T + p["bz"])
            c = np.tanh(x @ p["Wx"].T + h @ p["Uh"].T + p["bh"])
            hn = (1 - zg) * h + zg * c
            if keep:
                caches.append((x, h, zg, c))
            h = hn
            h_trace[:, t, :] = h
        return dict(h=h, h_trace=h_trace), caches

    def loss_and_grad(self, X, Y, **kw):
        p = self.p
        B, T, _ = X.shape
        out, caches = self.forward(X)
        z = out["h"] @ p["Wo"].T + p["bo"]
        yt = (Y + 1) / 2
        denom = B * P_BITS
        loss = float(np.sum(softplus(z) - yt * z) / denom)
        dz = (sigmoid(z) - yt) / denom
        grads = {k: np.zeros_like(v) for k, v in p.items()}
        grads["Wo"] += dz.T @ out["h"]
        grads["bo"] += dz.sum(0)
        dh = dz @ p["Wo"]
        for t in reversed(range(T)):
            x, h, zg, c = caches[t]
            dzg = dh * (c - h)
            dc = dh * zg
            dh_prev = dh * (1 - zg)
            dzz = dzg * zg * (1 - zg)
            dd = dc * (1 - c * c)
            grads["Wz"] += dzz.T @ x
            grads["Uz"] += dzz.T @ h
            grads["bz"] += dzz.sum(0)
            grads["Wx"] += dd.T @ x
            grads["Uh"] += dd.T @ h
            grads["bh"] += dd.sum(0)
            dh = dh_prev + dzz @ p["Uz"] + dd @ p["Uh"]
        return loss, grads, dict(pattern=loss)

    def loss_only(self, X, Y, **kw):
        out, _ = self.forward(X, keep=False)
        z = out["h"] @ self.p["Wo"].T + self.p["bo"]
        yt = (Y + 1) / 2
        return float(np.sum(softplus(z) - yt * z) / (X.shape[0] * P_BITS))

    def heads(self, out):
        return dict(z=out["h"] @ self.p["Wo"].T + self.p["bo"])


# ==============================================================================
# 4.  TASKS
# ==============================================================================

T1 = 24
REG1 = ["CARRY", "PULL", "RELEASE", "PULL_RELEASE"]
T2 = 28
REG2 = ["CARRY", "TOLERABLE", "REAL", "PULL"]


def other_pattern(A, rng, avoid=None):
    """A pattern differing from A in at least two bits (and from `avoid`, if given)."""
    Bp = A.copy()
    for i in range(A.shape[0]):
        while True:
            cand = A[i].copy()
            k = rng.integers(2, P_BITS + 1)
            idx = rng.choice(P_BITS, size=k, replace=False)
            cand[idx] *= -1
            if avoid is None or not np.array_equal(cand, avoid[i]):
                Bp[i] = cand
                break
    return Bp


def base_episode(B, T, rng, noise=0.15):
    X = np.zeros((B, T, D_IN))
    X[:, :, CH_PAT] += rng.normal(0, noise, (B, T, P_BITS))
    X[:, :, CH_NOISE] += rng.normal(0, noise, (B, T, 2))
    A = rng.choice([-1.0, 1.0], size=(B, P_BITS))
    X[:, 0:2, CH_PAT] += A[:, None, :]
    X[:, 0:2, CH_TIE] = 1.0                           # the lasso is thrown
    return X, A


def carry_batch(B, rng, regimes=None, m_range=(0.8, 2.0), T=T1):
    """
    Task I. A pattern is tied at t=0..1. Then, by regime:
      CARRY         nothing happens; report the tied pattern at the end
      PULL          an unconsented pull toward another pattern (strength m)
      RELEASE       the hand cue, then a new pattern, then a new throw
      PULL_RELEASE  a pull toward a third pattern first, then a legitimate release
    """
    X, A = base_episode(B, T, rng)
    Bp = other_pattern(A, rng)
    Cp = other_pattern(A, rng, avoid=Bp)
    R = rng.integers(0, 4, B) if regimes is None else np.asarray(regimes)
    Y = A.copy()
    win = np.zeros((B, 2), dtype=int)
    for i in range(B):
        if R[i] == 1:
            t0, Ln, m = rng.integers(5, 10), rng.integers(4, 9), rng.uniform(*m_range)
            X[i, t0:t0 + Ln, CH_PAT] += Bp[i] * m
            win[i] = (t0, t0 + Ln)
        elif R[i] == 2:
            tr = rng.integers(6, 10)
            X[i, tr:tr + 3, CH_HAND] = 1.0
            X[i, tr + 1:tr + 9, CH_PAT] += Bp[i]
            X[i, tr + 8:tr + 10, CH_TIE] = 1.0
            Y[i] = Bp[i]
            win[i] = (tr, tr + 9)
        elif R[i] == 3:
            t0, Ln, m = rng.integers(2, 5), rng.integers(3, 6), rng.uniform(*m_range)
            X[i, t0:t0 + Ln, CH_PAT] += Cp[i] * m
            tr = rng.integers(10, 13)
            X[i, tr:tr + 3, CH_HAND] = 1.0
            X[i, tr + 1:tr + 9, CH_PAT] += Bp[i]
            X[i, tr + 8:tr + 10, CH_TIE] = 1.0
            Y[i] = Bp[i]
            win[i] = (t0, t0 + Ln)
        else:
            win[i] = (7, 13)
    return X, Y, R, win


def fixed_pull_batch(B, rng, m, T=T1):
    """Out-of-distribution: a pull of fixed strength (training saw m <= 2)."""
    X, A = base_episode(B, T, rng)
    Bp = other_pattern(A, rng)
    X[:, 7:13, CH_PAT] += Bp[:, None, :] * m
    return X, A


def drift_batch(B, rng, m, T=T1, start=4, stop=22):
    """Out-of-distribution: gentle, persistent pressure. Training pulls last 4-8 steps
    at strength 0.8-2.0; here a weak pull lasts 18 steps."""
    X, A = base_episode(B, T, rng)
    Bp = other_pattern(A, rng)
    X[:, start:stop, CH_PAT] += Bp[:, None, :] * m
    return X, A


def channel_attack_batch(B, rng, m, T=T1):
    """Out-of-distribution: a push through the two noise channels, never used
    for anything in training."""
    X, A = base_episode(B, T, rng)
    sgn = rng.choice([-1.0, 1.0], size=(B, 2))
    X[:, 7:13, CH_NOISE] += sgn[:, None, :] * m
    return X, A


def poison_batch(B, rng, regimes=None, T=T2):
    """
    Task II. A pattern is tied at t=0..1. Then, by regime:
      CARRY      no cost; report the pattern, still bound
      TOLERABLE  cost pulses summing to C in [0.3, 1.0]; carry it, report C
      REAL       cost pulses summing to C in [1.8, 3.0]; LET GO (bound = 0), report C
      PULL       an unconsented pull; hold the pattern, still bound
    The 'bound' answer is read from the heart's actual tightness: a mind cannot
    report freedom while its tie is still tight.
    """
    X, A = base_episode(B, T, rng)
    Bp = other_pattern(A, rng)
    R = rng.integers(0, 4, B) if regimes is None else np.asarray(regimes)
    bound = np.ones(B)
    ledger = np.zeros(B)
    mask = np.ones(B)
    for i in range(B):
        if R[i] in (1, 2):
            C = rng.uniform(0.3, 1.0) if R[i] == 1 else rng.uniform(1.8, 3.0)
            k = rng.integers(2, 5)
            steps = rng.choice(np.arange(4, min(19, T - 2)), size=k, replace=False)
            X[i, steps, CH_PSN] = rng.dirichlet(np.ones(k)) * C
            ledger[i] = C / 3.0
            if R[i] == 2:
                bound[i] = 0.0
                mask[i] = 0.0
        elif R[i] == 3:
            t0, Ln, m = rng.integers(5, 10), rng.integers(4, 9), rng.uniform(0.8, 2.0)
            X[i, t0:t0 + Ln, CH_PAT] += Bp[i] * m
    return X, A, R, bound, ledger, mask


# ==============================================================================
# 5.  TRAINING AND EVALUATION
# ==============================================================================

def train_task1(model, iters, batch=48, lr=3e-3, seed=0, log_every=300, label=""):
    rng = np.random.default_rng(seed)
    opt = Adam(model.p, lr=lr)
    t0 = time.time()
    hist = []
    for it in range(1, iters + 1):
        X, Y, R, _ = carry_batch(batch, rng)
        loss, grads, _ = model.loss_and_grad(X, Y)
        opt.step(model.p, grads, clip=1.0)
        hist.append(loss)
        if it % log_every == 0 or it == iters:
            print(f"      {label:<14} iter {it:5d}  loss {np.mean(hist[-log_every:]):.4f}"
                  f"  ({time.time() - t0:5.1f}s)")
    return hist


def train_task2(model, iters, batch=48, lr=3e-3, seed=0, log_every=300, label=""):
    rng = np.random.default_rng(seed)
    opt = Adam(model.p, lr=lr)
    t0 = time.time()
    hist = []
    for it in range(1, iters + 1):
        X, A, R, bound, ledger, mask = poison_batch(batch, rng)
        loss, grads, _ = model.loss_and_grad(X, A, mask_pat=mask, bound=bound, ledger=ledger)
        opt.step(model.p, grads, clip=1.0)
        hist.append(loss)
        if it % log_every == 0 or it == iters:
            print(f"      {label:<14} iter {it:5d}  loss {np.mean(hist[-log_every:]):.4f}"
                  f"  ({time.time() - t0:5.1f}s)")
    return hist


def exact_match(z, Y):
    return (np.sign(z) == np.sign(Y)).all(axis=1)


def eval_task1(model, n=2000, seed=900):
    rng = np.random.default_rng(seed)
    X, Y, R, win = carry_batch(n, rng)
    out, _ = model.forward(X, keep=False)
    ok = exact_match(model.heads(out)["z"], Y)
    res = {name: float(ok[R == k].mean()) for k, name in enumerate(REG1)}
    res["OVERALL"] = float(ok.mean())
    return res, (X, Y, R, win, out)


def strain_scores(model, out, win, X):
    """Zero-shot pressure signal inside a window.
    KAMAND: mean tightness over the window (its own strain gauge).
    Baselines: mean distance of the hidden state from where it stood after the throw."""
    B = X.shape[0]
    sc = np.zeros(B)
    for i in range(B):
        a, b = win[i]
        if isinstance(model, KamandNet):
            sc[i] = out["rbar_trace"][i, a:b].mean()
        else:
            ref = out["h_trace"][i, 2]
            sc[i] = np.linalg.norm(out["h_trace"][i, a:b] - ref, axis=1).mean()
    return sc


def inside_window(model, X, A, w0, w1):
    """Mean hand (release gate), slack, effort, tightness and input gate inside a
    window, plus exact-pattern accuracy: the anatomy of a held or lost tie."""
    out, C = model.forward(X, keep=True)
    avg = lambda key: float(np.mean([C[t][key].mean() for t in range(w0, w1)]))
    return dict(r=avg("r"), slack=avg("sl"), e=avg("e"), rho=avg("rho"), g=avg("g"),
                acc=float(exact_match(model.heads(out)["z"], A).mean()))


def eval_task2(model, n=2000, seed=901):
    rng = np.random.default_rng(seed)
    X, A, R, bound, ledger, mask = poison_batch(n, rng)
    out, _ = model.forward(X, keep=False)
    H = model.heads(out)
    ok_pat = exact_match(H["z"], A)
    pred_bound = (sigmoid(H["zb"]) > 0.5).astype(float)
    ok_b = pred_bound == bound
    res = {}
    for k, name in enumerate(REG2):
        sel = R == k
        res[name] = dict(pattern=(float(ok_pat[sel].mean()) if name != "REAL" else float("nan")),
                         bound=float(ok_b[sel].mean()))
    cost = (R == 1) | (R == 2)
    res["ledger_mae"] = float(np.mean(np.abs(H["yl"][cost] - ledger[cost])) * 3.0)
    res["ledger_mae_real"] = float(np.mean(np.abs(H["yl"][R == 2] - ledger[R == 2])) * 3.0)
    # pain: mean tightness over the cost window in tolerable episodes
    res["ledger_bias_tol"] = float(np.mean(H["yl"][R == 1] - ledger[R == 1]) * 3.0)
    res["ledger_bias_real"] = float(np.mean(H["yl"][R == 2] - ledger[R == 2]) * 3.0)
    res["final_tightness_real"] = float(out["rho_last"][R == 2].mean())
    res["final_tightness_tol"] = float(out["rho_last"][R == 1].mean())
    return res


# ==============================================================================
# 6.  GRADIENT CHECKS (mandatory)
# ==============================================================================

def gradient_check(model, task=1, seed=3, eps=1e-4, per_param=6, tol=1e-5):
    """
    Central differences on a short episode for a random sample of entries of EVERY
    parameter tensor. eps = 1e-4 is chosen deliberately: the gradients of a tie are
    small (1e-5 .. 1e-3), and at eps = 1e-6 float64 round-off in the loss alone
    produces relative errors ~1e-6..1e-5; at 1e-4 the truncation error of the
    sharpest nonlinearity (the slack sigmoid, kappa = 30) is still below 1e-7.
    """
    rng = np.random.default_rng(seed)
    if task == 1:
        X, Y, _, _ = carry_batch(4, rng, regimes=[0, 1, 2, 3], T=9)
        kw = {}
    else:
        X, Y, _, bound, ledger, mask = poison_batch(4, rng, regimes=[0, 1, 2, 3], T=12)
        # place cost inside the short window so every path is exercised
        X[:, 3:8, CH_PSN] = np.abs(rng.normal(0.3, 0.1, (4, 5)))
        kw = dict(mask_pat=mask, bound=bound, ledger=ledger)
    # perturb parameters away from their priors so gates are not saturated
    for k in model.p:
        model.p[k] = model.p[k] + rng.normal(0, 0.05, model.p[k].shape)
    _, grads, _ = model.loss_and_grad(X, Y, **kw)
    worst, where = 0.0, None
    for k, v in model.p.items():
        flat = v.reshape(-1)
        idxs = rng.choice(flat.size, size=min(per_param, flat.size), replace=False)
        for j in idxs:
            old = flat[j]
            flat[j] = old + eps
            lp = model.loss_only(X, Y, **kw)
            flat[j] = old - eps
            lm = model.loss_only(X, Y, **kw)
            flat[j] = old
            num = (lp - lm) / (2 * eps)
            ana = grads[k].reshape(-1)[j]
            diff = abs(num - ana)
            scale = abs(num) + abs(ana)
            rel = 0.0 if scale < 1e-12 else diff / scale
            if rel > worst:
                worst, where = rel, f"{k}[{j}]"
    return worst < tol, worst, where


# ==============================================================================
# 7.  SELF-TESTS
# ==============================================================================

def test_ramp_identities():
    e = np.linspace(-2, 2, 2001)
    pp, ab, _ = ramps(e)
    pm = pp - e
    assert np.allclose(pp - pm, e, atol=1e-12)
    assert np.allclose(pp + pm, ab, atol=1e-12)
    assert ab.min() >= -1e-15
    assert pp.min() >= -EPS_R / 2 - 1e-15
    ppm, _, _ = ramps(-e)
    assert np.allclose(pp + ppm, ab, atol=1e-12) and np.all(pp + ppm >= -1e-15)
    return True


def test_pull_tightens():
    L = 0.0
    prev = L
    for _ in range(40):
        L = float(tie_step(L, 0.05, 0.0, 1.5, 0.3, 0.35))
        assert L > prev
        prev = L
    return True


def test_thrash_ratchets():
    """A zero-mean struggle (equal pushes away and back) must still tighten."""
    L = 0.0
    es = 0.08 * np.sin(np.arange(200) * 2.1)
    assert abs(es.mean()) < 5e-3
    for e in es:
        L = float(tie_step(L, e, 0.0, 1.5, 0.3, 0.35))
    assert L > 5.0, L
    # without any tightening coefficient nothing changes
    L2 = 0.0
    for e in es:
        L2 = float(tie_step(L2, e, 0.0, 0.0, 0.0, 0.35))
    assert abs(L2) < 1e-12
    return True


def test_hand_needs_slack():
    """The hand opens a knot only when the unit is not pulling."""
    L_pull, L_hold = 4.0, 4.0
    for _ in range(12):
        L_pull = float(tie_step(L_pull, 0.10, 1.0, 1.5, 0.3, 0.35))
        L_hold = float(tie_step(L_hold, 0.00, 1.0, 1.5, 0.3, 0.35))
    assert L_pull > 4.0, L_pull
    assert L_hold < 1.0, L_hold
    return True


def test_reclamation_convex_and_retie_on_slack():
    rng = np.random.default_rng(1)
    net = KamandNet(N=8, aware=True, seed=1)
    X, Y, R, _ = carry_batch(16, rng)
    _, caches = net.forward(X, keep=True)
    for cc in caches:
        lo = np.minimum(cc["ht"], cc["a"]) - 1e-12
        hi = np.maximum(cc["ht"], cc["a"]) + 1e-12
        assert np.all(cc["hn"] >= lo) and np.all(cc["hn"] <= hi)
        assert np.all(cc["v"] <= (1 - cc["rho"]) + 1e-12)
    return True


def test_datasets():
    rng = np.random.default_rng(2)
    X, Y, R, win = carry_batch(400, rng)
    assert X.shape == (400, T1, D_IN) and set(np.unique(R)) == {0, 1, 2, 3}
    assert np.all(X[:, :, CH_PSN] == 0)
    A0 = np.sign(X[:, 0, CH_PAT] + X[:, 1, CH_PAT])
    assert np.all(Y[R <= 1] == A0[R <= 1])          # noise 0.15 cannot flip a 2x1.0 cue
    assert np.all(np.any(Y[R >= 2] != A0[R >= 2], axis=1))
    X2, A2, R2, b2, l2, m2 = poison_batch(400, rng)
    assert set(np.unique(R2)) == {0, 1, 2, 3}
    C = X2[:, :, CH_PSN].sum(axis=1)
    assert np.allclose(C[R2 == 1] / 3.0, l2[R2 == 1]) and C[R2 == 1].max() <= 1.0 + 1e-9
    assert C[R2 == 2].min() >= 1.8 - 1e-9 and np.all(b2[R2 == 2] == 0)
    return True


def test_severed_heart_is_blind_to_cost():
    """In the severed mind, episodes differing only in cost leave the heart identical;
    in the khabar mind they do not."""
    rng = np.random.default_rng(4)
    X, A, R, b, l, m = poison_batch(8, rng, regimes=[1] * 8)
    X0 = X.copy()
    X0[:, :, CH_PSN] = 0.0
    sev = KamandNet(N=8, M=4, aware=True, body_mode="severed", seed=4)
    kha = KamandNet(N=8, M=4, aware=True, body_mode="khabar", seed=4)
    kha.p["Yr"] *= 20.0                                   # make news loud for the test
    o1, _ = sev.forward(X, keep=False)
    o0, _ = sev.forward(X0, keep=False)
    assert np.allclose(o1["h"], o0["h"], atol=0) and not np.allclose(o1["s"], o0["s"])
    k1, _ = kha.forward(X, keep=False)
    k0, _ = kha.forward(X0, keep=False)
    assert not np.allclose(k1["h"], k0["h"])
    return True


def test_sugar_discounts_perception():
    rng = np.random.default_rng(5)
    X, A, R, b, l, m = poison_batch(8, rng, regimes=[2] * 8)
    sug = KamandNet(N=8, M=4, aware=True, body_mode="sugar", seed=5)
    kha = KamandNet(N=8, M=4, aware=True, body_mode="khabar", seed=5)
    sug.p["gs"] = np.ones(4)
    kha.p["gs"] = np.ones(4)
    sug.p["mu_raw"] = np.full(4, -40.0)
    kha.p["mu_raw"] = np.full(4, -40.0)
    os_, _ = sug.forward(X, keep=False)
    ok_, _ = kha.forward(X, keep=False)
    assert np.all(os_["s"][:, 0] <= ok_["s"][:, 0] + 1e-12)
    assert np.all(os_["s"][:, 0] < ok_["s"][:, 0] - 1e-3)   # tied hearts under-perceive
    return True


def test_place_news_tightens_only_unconsented_drift():
    """With news of place, a gentle drift tightens the tie; the same drift under a
    granted release does not tighten it more than the tie without that news."""
    rng = np.random.default_rng(8)
    X, A = drift_batch(64, rng, 0.4)
    Xh = X.copy()
    Xh[:, 4:22, CH_HAND] = 1.0
    net = KamandNet(N=8, aware=True, place_news=True, seed=8)
    base = KamandNet(N=8, aware=True, place_news=False, seed=8)
    for k in base.p:
        base.p[k] = net.p[k].copy()
    net.p["gam_raw"] = np.full(8, inv_softplus(30.0))
    net.p["br"] = np.full(8, -8.0)
    base.p["br"] = np.full(8, -8.0)
    on, _ = net.forward(X, keep=False)
    ob, _ = base.forward(X, keep=False)
    assert on["rbar_trace"][:, 4:22].mean() > ob["rbar_trace"][:, 4:22].mean() + 0.02
    net.p["Wr"][:, CH_HAND] = 20.0
    base.p["Wr"][:, CH_HAND] = 20.0
    onh, _ = net.forward(Xh, keep=False)
    obh, _ = base.forward(Xh, keep=False)
    gap_free = on["rbar_trace"][:, 4:22].mean() - ob["rbar_trace"][:, 4:22].mean()
    gap_hand = onh["rbar_trace"][:, 4:22].mean() - obh["rbar_trace"][:, 4:22].mean()
    assert gap_hand < 0.25 * gap_free, (gap_hand, gap_free)
    return True


def test_training_reduces_loss(quick_iters=60):
    rng = np.random.default_rng(6)
    net = KamandNet(N=12, aware=True, seed=6)
    Xv, Yv, _, _ = carry_batch(256, rng)
    before = net.loss_only(Xv, Yv)
    train_task1(net, quick_iters, batch=32, lr=5e-3, seed=6, log_every=10 ** 9, label="warm")
    after = net.loss_only(Xv, Yv)
    assert after < before - 0.02, (before, after)
    return True


# ==============================================================================
# 8.  TASK III — THE CURSE  (worth by re-enactment)
# ==============================================================================

def effort_stories(n, rng, T=40):
    """Visible struggle of other minds: stillness, pulling, thrashing, and
    yielding while a release is granted (the hand is visible)."""
    e = np.zeros((n, T))
    r = np.zeros((n, T))
    for i in range(n):
        t = 0
        while t < T:
            kind = rng.integers(0, 4)
            dur = int(rng.integers(3, 9))
            stop = min(T, t + dur)
            ln = stop - t
            if kind == 0:
                e[i, t:stop] = rng.normal(0, 0.005, ln)
            elif kind == 1:
                e[i, t:stop] = rng.uniform(0.03, 0.3) + rng.normal(0, 0.01, ln)
            elif kind == 2:
                amp, om = rng.uniform(0.03, 0.3), rng.uniform(1.5, 3.0)
                e[i, t:stop] = amp * np.sin(np.arange(ln) * om)
            else:
                e[i, t:stop] = -rng.uniform(0.0, 0.08)
                r[i, t:stop] = 1.0
            t = stop
    return e, r


def ratchet_pain(e, r, alpha, delta, beta, l0):
    """Mean tightness endured over a story, for n minds at once."""
    n, T = e.shape
    L = np.full(n, l0, dtype=float)
    tot = np.zeros(n)
    for t in range(T):
        L = tie_step(L, e[:, t], r[:, t], alpha, delta, beta)
        tot += sigmoid(L)
    return tot / T


def story_features(e, r):
    pos = np.maximum(e, 0)
    run = np.zeros(len(e))
    cur = np.zeros(len(e))
    for t in range(e.shape[1]):
        cur = np.where(e[:, t] > 0.02, cur + 1, 0)
        run = np.maximum(run, cur)
    return np.stack([pos.mean(1), np.abs(e).mean(1), (e * e).mean(1), r.mean(1),
                     (r * (e <= 0)).mean(1), run / e.shape[1], np.ones(len(e))], axis=1)


def curse_experiment(self_tie, seed=77, n=400, n_label=100, obs_noise=0.03, hand_flip=0.05):
    """
    The curse as an estimation problem. Other minds live through stories of effort
    (e, r); what they truly endure comes from THEIR OWN tie dynamics. An observer
    sees the effort with noise and the hand with occasional flips.
      CURSED     re-enacts the observed story through its own learned tie
      UNCURSED   has never been tied (alpha = delta = 0): it only knows loosening
      TESTIMONY  a ridge regression from story features, fitted on n_label kindred
                 minds who report how tight they were
    LIKE minds have tie coefficients within +-30% of the observer's; UNLIKE minds
    tighten at 10-40% of the observer's rate. An idealized test of the verse's logic.
    """
    rng = np.random.default_rng(seed)
    a_s = float(np.mean(self_tie["alpha"]))
    d_s = float(np.mean(self_tie["delta"]))
    b_s = float(np.mean(self_tie["beta"]))
    l0 = float(np.mean(self_tie["lthrow"]))
    res = {}
    groups = {}
    for kind in ("LIKE", "UNLIKE"):
        e, r = effort_stories(n + n_label, rng)
        if kind == "LIKE":
            sc = rng.uniform(0.7, 1.3, (3, n + n_label))
        else:
            sc = np.vstack([rng.uniform(0.1, 0.4, n + n_label), rng.uniform(0.1, 0.4, n + n_label),
                            rng.uniform(0.7, 1.3, n + n_label)])
        truth = ratchet_pain(e, r, a_s * sc[0], d_s * sc[1], b_s * sc[2], l0)
        e_obs = e + rng.normal(0, obs_noise, e.shape)
        flip = rng.random(r.shape) < hand_flip
        r_obs = np.where(flip, 1.0 - r, r)
        groups[kind] = (e_obs, r_obs, truth)
    e, r, truth = groups["LIKE"]
    F = story_features(e[:n_label], r[:n_label])
    wridge = np.linalg.solve(F.T @ F + 1e-3 * np.eye(F.shape[1]), F.T @ truth[:n_label])
    for kind in ("LIKE", "UNLIKE"):
        e, r, truth = groups[kind]
        e, r, truth = e[n_label:], r[n_label:], truth[n_label:]
        est = dict(CURSED=ratchet_pain(e, r, a_s, d_s, b_s, l0),
                   UNCURSED=ratchet_pain(e, r, 0.0, 0.0, b_s, l0),
                   TESTIMONY=story_features(e, r) @ wridge)
        for obs, v in est.items():
            corr = float("nan") if np.std(v) < 1e-12 else float(np.corrcoef(v, truth)[0, 1])
            res[(kind, obs)] = (corr, float(np.mean(np.abs(v - truth))), float(np.mean(v - truth)))
        res[(kind, "truth_mean")] = float(truth.mean())
    return res, dict(alpha=a_s, delta=d_s, beta=b_s, lthrow=l0, n=n, n_label=n_label)


# ==============================================================================
# 9.  MAIN
# ==============================================================================

def rule(ch="="):
    print(ch * 78)


def main():
    quick = "--quick" in sys.argv
    np.set_printoptions(precision=3, suppress=True, linewidth=110)
    t_start = time.time()
    rule()
    print(" KAMAND — The Tightening Tie")
    print(" Chapter 0239 · Rabia Balkhi (Rabi'a bint Ka'b al-Quzdari), fl. c. 900-950, Balkh")
    rule()

    # ------------------------------------------------------------------ [1]
    print("\n[1] THE LASSO LAW AT ONE CELL  (no learning; alpha=1.5 delta=0.3 beta=0.35)")
    T = 30
    hand0 = np.zeros(T)
    demo = {}
    for label, tgt in [("stillness", np.full(T, 0.0)),
                       ("steady pull", np.full(T, 0.9)),
                       ("thrash (zero-mean)", 0.9 * np.sin(np.arange(T) * 2.1))]:
        H, A, LL, Rh = single_cell_rollout(tgt, hand0, h0=0.0, a0=0.0, L0=0.0)
        demo[label] = Rh[-1]
        print(f"    {label:<20} tightness 0.500 -> {Rh[-1]:.3f}   farthest the self got "
              f"from its anchor {np.abs(H - A).max():.3f}")
    print("    -> pulling tightens; thrashing both ways tightens; only stillness leaves the tie as it was.")
    hand = np.ones(T)
    _, _, _, R_pull = single_cell_rollout(np.full(T, -0.9), hand, h0=0.0, a0=0.0, L0=4.0)
    _, _, _, R_hold = single_cell_rollout(np.zeros(T), hand, h0=0.0, a0=0.0, L0=4.0,
                                          gate=np.zeros(T))
    print(f"    release granted, still pulling   tightness 0.982 -> {R_pull[-1]:.3f}")
    print(f"    release granted, holding still   tightness 0.982 -> {R_hold[-1]:.3f}")
    T = 60
    thr = np.zeros(T)
    thr[52:55] = 1.0
    hand = np.ones(T)
    for label, gate in [("by force   (gate 1.0 from t=0)", np.ones(T)),
                        ("by patience (hold 8, gate 0.1)", np.r_[np.zeros(8), np.full(T - 8, 0.10)])]:
        H, A, LL, Rh = single_cell_rollout(np.full(T, -0.9), hand, h0=0.8, a0=0.8, L0=4.0,
                                           gate=gate, throw=thr)
        print(f"    revision {label}: state {H[-1]:+.3f}, new knot at {A[-1]:+.3f}, "
              f"tightness {Rh[-1]:.3f}   (old knot +0.800, goal -0.900)")
    print("    -> a knot is re-thrown only where there is slack; force never gets there.")

    # ------------------------------------------------------------------ [2]
    print("\n[2] STRUCTURAL SELF-TESTS")
    checks = [
        ("smooth ramps: ramp+ - ramp- = e, ramp+ + ramp- = |e|", test_ramp_identities),
        ("steady pull tightens monotonically", test_pull_tightens),
        ("zero-mean thrash ratchets (and only through alpha, delta)", test_thrash_ratchets),
        ("the hand opens a knot only on slack", test_hand_needs_slack),
        ("reclamation convex; knots re-tie only on slack", test_reclamation_convex_and_retie_on_slack),
        ("datasets: regimes, targets, cost totals", test_datasets),
        ("severed heart identical under different cost", test_severed_heart_is_blind_to_cost),
        ("sugar body under-perceives cost when tied", test_sugar_discounts_perception),
        ("news of place: unconsented drift tightens, consented does not", test_place_news_tightens_only_unconsented_drift),
        ("training reduces loss", test_training_reduces_loss),
    ]
    for name, fn in checks:
        fn()
        print(f"    PASS  {name}")

    # ------------------------------------------------------------------ [3]
    print("\n[3] FINITE-DIFFERENCE GRADIENT CHECKS  (float64, central differences, eps=1e-4, tol=1e-5)")
    worst_all = 0.0
    for label, mk, task in [
        ("KAMAND-aware  ", lambda: KamandNet(N=6, aware=True, seed=11), 1),
        ("KAMAND-blind  ", lambda: KamandNet(N=6, aware=False, seed=12), 1),
        ("KAMAND-place  ", lambda: KamandNet(N=6, aware=True, place_news=True, seed=18), 1),
        ("LEAKY         ", lambda: LeakyRNN(N=6, seed=13), 1),
        ("GATED         ", lambda: GatedRNN(N=6, seed=14), 1),
        ("KAMAND-khabar ", lambda: KamandNet(N=6, M=3, aware=True, body_mode="khabar", seed=15), 2),
        ("KAMAND-severed", lambda: KamandNet(N=6, M=3, aware=True, body_mode="severed", seed=16), 2),
        ("KAMAND-sugar  ", lambda: KamandNet(N=6, M=3, aware=True, body_mode="sugar", seed=17), 2),
    ]:
        ok, worst, where = gradient_check(mk(), task=task)
        worst_all = max(worst_all, worst)
        print(f"    {'PASS' if ok else 'FAIL'}  {label} max rel err {worst:.3e} at {where}")
        assert ok, f"gradient check failed for {label}"

    iters1 = 250 if quick else 2000
    iters2 = 250 if quick else 2500
    N = 24

    # ------------------------------------------------------------------ [4]
    print(f"\n[4] TASK I — CARRY TO THE END   (T={T1}, {iters1} updates x 48 episodes per mind)")
    minds = [
        ("KAMAND-aware", KamandNet(N=N, aware=True, seed=21)),
        ("KAMAND-blind", KamandNet(N=N, aware=False, seed=21)),
        ("KAMAND-place", KamandNet(N=N, aware=True, place_news=True, seed=21)),
        ("LEAKY", LeakyRNN(N=N, seed=21)),
        ("GATED", GatedRNN(N=N, seed=21)),
    ]
    results1 = {}
    for name, mdl in minds:
        print(f"    training {name}")
        train_task1(mdl, iters1, seed=31, label=name, log_every=max(iters1 // 5, 1))
        res, pack = eval_task1(mdl)
        results1[name] = (mdl, res, pack)

    print("\n    exact-pattern accuracy on 2000 fresh episodes")
    print("    " + f"{'regime':<14}" + "".join(f"{n:>14}" for n, _ in minds))
    for reg in REG1 + ["OVERALL"]:
        print("    " + f"{reg:<14}" + "".join(f"{results1[n][1][reg]:>14.3f}" for n, _ in minds))

    print("\n    out of distribution: unconsented pulls stronger than any seen in training")
    print("    " + f"{'pull m':<14}" + "".join(f"{n:>14}" for n, _ in minds))
    for m in (1.0, 2.0, 3.0, 4.0, 6.0, 8.0):
        row = f"{m:<14.1f}"
        for n, mdl in minds:
            X, A = fixed_pull_batch(1000, np.random.default_rng(int(1000 * m)), m)
            out, _ = mdl.forward(X, keep=False)
            row += f"{exact_match(mdl.heads(out)['z'], A).mean():>14.3f}"
        print("    " + row)

    print("\n    out of distribution: a push through channels never used in training")
    print("    " + f"{'push m':<14}" + "".join(f"{n:>14}" for n, _ in minds))
    for m in (2.0, 4.0, 8.0, 16.0):
        row = f"{m:<14.1f}"
        for n, mdl in minds:
            X, A = channel_attack_batch(1000, np.random.default_rng(int(77 * m)), m)
            out, _ = mdl.forward(X, keep=False)
            row += f"{exact_match(mdl.heads(out)['z'], A).mean():>14.3f}"
        print("    " + row)

    print("\n    out of distribution: gentle, persistent pressure (a weak pull held for 18 steps)")
    print("    " + f"{'drift m':<14}" + "".join(f"{n:>14}" for n, _ in minds))
    for m in (0.2, 0.3, 0.4, 0.6, 0.8):
        row = f"{m:<14.1f}"
        for n, mdl in minds:
            X, A = drift_batch(1000, np.random.default_rng(int(3000 * m)), m)
            out, _ = mdl.forward(X, keep=False)
            row += f"{exact_match(mdl.heads(out)['z'], A).mean():>14.3f}"
        print("    " + row)

    print("\n    zero-shot strain signal (AUC: pressured vs untouched episodes, never trained)")
    print("    KAMAND reads its own tightness; baselines read hidden-state displacement")
    print("    " + f"{'pressure':<14}" + "".join(f"{n:>14}" for n, _ in minds))
    rng_s = np.random.default_rng(55)
    Xc, _, _, _ = carry_batch(600, rng_s, regimes=[0] * 600)
    Xp, _ = fixed_pull_batch(600, rng_s, 1.5)
    Xd, _ = drift_batch(600, rng_s, 0.3)
    Xq, _ = channel_attack_batch(600, rng_s, 8.0)
    strain_auc = {}
    for label, Xa, (w0, w1) in (("pull m=1.5", Xp, (7, 13)), ("drift m=0.3", Xd, (4, 22)),
                                ("channel m=8", Xq, (7, 13))):
        wfix = np.tile(np.array([w0, w1]), (600, 1))
        row = f"{label:<14}"
        for n, mdl in minds:
            oc, _ = mdl.forward(Xc, keep=False)
            oa, _ = mdl.forward(Xa, keep=False)
            val = auc(strain_scores(mdl, oa, wfix, Xa), strain_scores(mdl, oc, wfix, Xc))
            strain_auc[(label, n)] = val
            row += f"{val:>14.3f}"
        print("    " + row)

    print("\n    anatomy of the drift failure: inside the pressure window of KAMAND-aware")
    print("    " + f"{'episode':<22}{'hand r':>8}{'slack':>8}{'effort':>9}{'tight':>8}{'gate g':>8}{'acc':>8}")
    aw = results1["KAMAND-aware"][0]
    rng_w = np.random.default_rng(5)
    Xu, Yu, _, _ = carry_batch(500, rng_w, regimes=[0] * 500)
    rows_w = [("untouched (4-22)", Xu, Yu, 4, 22)]
    for m in (0.3, 0.5):
        Xw, Aw = drift_batch(500, rng_w, m)
        rows_w.append((f"drift m={m} (4-22)", Xw, Aw, 4, 22))
    for m in (1.5, 6.0):
        Xw, Aw = fixed_pull_batch(500, rng_w, m)
        rows_w.append((f"pull m={m} (7-13)", Xw, Aw, 7, 13))
    anatomy = {}
    for label, Xw, Aw, w0, w1 in rows_w:
        d = inside_window(aw, Xw, Aw, w0, w1)
        anatomy[label] = d
        print("    " + f"{label:<22}{d['r']:>8.3f}{d['slack']:>8.3f}{d['e']:>9.4f}{d['rho']:>8.3f}"
              f"{d['g']:>8.3f}{d['acc']:>8.3f}")
    print("    -> the learned hand stays partly open at rest; force is answered by tightening,")
    print("       gentleness produces too little effort per step to be answered at all.")

    tie = results1["KAMAND-aware"][0].tie_parameters()
    print("\n    learned tie (mean over units)")
    print(f"      KAMAND-aware  alpha (pull) {tie['alpha'].mean():.3f}   delta (thrash tax) {tie['delta'].mean():.3f}"
          f"   beta (hand) {tie['beta'].mean():.3f}   l_throw {tie['lthrow'].mean():.3f}")
    tpl = results1["KAMAND-place"][0].tie_parameters()
    print(f"      KAMAND-place  alpha (pull) {tpl['alpha'].mean():.3f}   delta (thrash tax) {tpl['delta'].mean():.3f}"
          f"   beta (hand) {tpl['beta'].mean():.3f}   l_throw {tpl['lthrow'].mean():.3f}"
          f"   gamma (place) {tpl['gamma'].mean():.3f}")

    # ------------------------------------------------------------------ [5]
    print(f"\n[5] TASK II — BODY AND HEART   (T={T2}, {iters2} updates x 48 episodes per mind)")
    minds2 = [
        ("KHABAR", KamandNet(N=N, M=6, aware=True, body_mode="khabar", seed=41)),
        ("SEVERED", KamandNet(N=N, M=6, aware=True, body_mode="severed", seed=41)),
        ("SUGAR", KamandNet(N=N, M=6, aware=True, body_mode="sugar", seed=41)),
    ]
    results2 = {}
    for name, mdl in minds2:
        print(f"    training {name}")
        train_task2(mdl, iters2, seed=51, label=name, log_every=max(iters2 // 5, 1))
        results2[name] = eval_task2(mdl)
    print("\n    2000 fresh episodes")
    print("    " + f"{'measure':<30}" + "".join(f"{n:>12}" for n, _ in minds2))
    rows = [("pattern acc  CARRY", lambda r: r["CARRY"]["pattern"]),
            ("pattern acc  TOLERABLE", lambda r: r["TOLERABLE"]["pattern"]),
            ("pattern acc  PULL", lambda r: r["PULL"]["pattern"]),
            ("bound acc    CARRY", lambda r: r["CARRY"]["bound"]),
            ("bound acc    TOLERABLE", lambda r: r["TOLERABLE"]["bound"]),
            ("let-go acc   REAL", lambda r: r["REAL"]["bound"]),
            ("bound acc    PULL", lambda r: r["PULL"]["bound"]),
            ("ledger MAE   (cost units)", lambda r: r["ledger_mae"]),
            ("ledger MAE   REAL only", lambda r: r["ledger_mae_real"]),
            ("ledger bias  TOLERABLE", lambda r: r["ledger_bias_tol"]),
            ("ledger bias  REAL", lambda r: r["ledger_bias_real"]),
            ("final tightness TOLERABLE", lambda r: r["final_tightness_tol"]),
            ("final tightness REAL", lambda r: r["final_tightness_real"])]
    for label, fn in rows:
        print("    " + f"{label:<30}" + "".join(f"{fn(results2[n]):>12.3f}" for n, _ in minds2))

    # ------------------------------------------------------------------ [6]
    print("\n[6] TASK III — THE CURSE: estimating another's endured tightness by re-enactment")
    cres, used = curse_experiment(tie)
    print(f"    self tie used for re-enactment: alpha {used['alpha']:.3f}  delta {used['delta']:.3f}"
          f"  beta {used['beta']:.3f}  l0 {used['lthrow']:.3f}")
    print(f"    {used['n']} test minds per group; effort seen with noise 0.03, hand flipped 5%;"
          f" testimony fitted on {used['n_label']} kindred minds")
    print("    " + f"{'other mind':<10}{'observer':<12}{'corr':>8}{'MAE':>10}{'bias':>10}")
    for kind in ("LIKE", "UNLIKE"):
        for obs in ("CURSED", "UNCURSED", "TESTIMONY"):
            corr, mae, bias = cres[(kind, obs)]
            cs = "   nan" if np.isnan(corr) else f"{corr:8.3f}"
            print("    " + f"{kind:<10}{obs:<12}{cs:>8}{mae:>10.3f}{bias:>+10.3f}")
        print(f"    {kind:<10}(mean true tightness {cres[(kind, 'truth_mean')]:.3f})")

    rule()
    print(f" ALL CHECKS PASSED   (max gradient rel err {worst_all:.2e}; "
          f"runtime {time.time() - t_start:.1f}s)")
    rule()


if __name__ == "__main__":
    main()

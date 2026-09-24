#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
ENCYCLOPEDIA OF LOST MINDS -- Chapter 0235
Abu Nasr al-Farabi (Alpharabius), c. 870 -- Damascus, Dec 950 / Jan 951

        THE MUHAKAT ENGINE
        A Two-Channel Conjunction Network in pure NumPy
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 12 Minds 221 - 240 Available on Amazon https://www.amazon.com/dp/B0HKVBKNTT
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0235_al_farabi_alpharabius_872 - Abu Nasr al-Farabi (Alpharabius), c. 870 -- Damascus, Dec 950 / Jan 951
================================================================================  

WHY THIS ARCHITECTURE AND NOT A TRANSFORMER
-------------------------------------------------------------------------------
Al-Farabi's distinctive cognitive claim is not "the soul has parts" (that is
Aristotle) and not "the wise should rule" (that is Plato). It is MUHAKAT --
"reproductive imitation" -- the doctrine that one and the same truth exists in
two encodings: a DEMONSTRATIVE one (burhan, proof, available to few) and an
IMAGINATIVE one (a sensible image, available to all), and that the imaginative
faculty is what performs the transform between them. Crucially, for al-Farabi
the imitation is not decoration. It is an ACTUATOR: images "stimulate particular
emotions, humors, desires and temperaments that move the body and put it into
action". Religion, on his account, is philosophy imitated -- the same content,
lossily re-encoded, and able to move a whole city.

Three consequences shape this file, and each is a mechanism you can ablate:

  (1) ITTISAL -- CONJUNCTION WITH AN EXTERNAL ACTIVE INTELLECT.
      For al-Farabi the Active Intellect (al-'aql al-fa''al) is the TENTH
      separate intelligence. It is not inside you. The human mind begins merely
      potential ("material intellect") and is actualised by conjunction with a
      shared, always-already-actual store of forms. So: the network's concept
      dictionary D is a SEPARATE, SHARED substance that the agent attends to.
      Sever it and the agent does not degrade gracefully -- it collapses. That
      is a testable claim, and Ablation A tests it.

  (2) MUHAKAT WITH RANK PRESERVATION.
      An imitation is allowed to LOSE content -- that is what makes it an
      imitation -- but it is not allowed to INVERT ORDER. If the demonstration
      places A above B in the hierarchy of being, no image of A and B may
      suggest the reverse. The network therefore carries an imaginative decoder,
      an INVERSE decoder (the imitation must be re-readable back into the
      intelligible that produced it), and an explicit order-preservation
      objective on the imitation alone. Failure of this constraint is what
      al-Farabi calls the NABITA, the "weed": an image with high motive force
      and no demonstration behind it. Test 8 scores exactly that.

  (3) THE EAR OVERRIDES THE DERIVATION.
      In the Great Book of Music al-Farabi does something startling for a man
      committed to demonstration from first principles: he lets measured
      practice supply primary principles, and states that on some points the
      ear is the ultimate judge EVEN WHERE IT CONTRADICTS a mathematical
      principle (he knows the semitone is not half a tone). So the network has
      a second sensory channel -- the "audible" dimensions -- feeding a gated
      corrective path that can overturn a conclusion validly derived from the
      received table. Ablation B deafens it, and the damage is localised
      exactly on the measured cases, not spread evenly. That localisation is
      the empirical signature of the doctrine.

Everything is hand-written NumPy: forward, backward, Adam, data. There is a
mandatory finite-difference gradient check over every parameter block, a real
training loop, and eight self-tests. No autodiff, no ML framework, no
attention-over-stored-keys stack.

Run:  python3 chapter_0235_al_farabi_alpharabius_872.py
===============================================================================
"""

import numpy as np
import time

RNG = np.random.default_rng(21393)   # 213 = chapter, 93 = arbitrary

# =============================================================================
# SECTION 0 -- SMALL NUMERICAL PRIMITIVES
# =============================================================================

def softmax(z, axis=-1):
    """Numerically stable softmax."""
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def sigmoid(z):
    """Numerically stable logistic function."""
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


# =============================================================================
# SECTION 1 -- THE WORLD: A CITY OF INTELLIGIBLES
# =============================================================================
# We need a task in which al-Farabi's three mechanisms are each LOAD-BEARING,
# so that ablating one produces a specific, predictable, measurable failure.
#
# The world contains K = 12 intelligibles (forms). Each has:
#   * a sensible prototype  P[k]  -- how a particular bearing that form looks
#   * a degree of being     RHO[k] in [0,1] -- its rank in the hierarchy
#   * an imitation          S[k]  -- the sensible image that stands for it
#   * a motive force        MOT[k] -- how strongly that image moves a body
#
# RHO and MOT are DELIBERATELY UNCORRELATED. That is the whole moral problem:
# the most moving image need not image the highest thing.
#
# A sample presents two forms (a premise pair). The task is to name the
# conclusion. There are four rule-tables:
#   T[0] -- the DERIVED table: what follows from the received first principles.
#   T[1..3] -- three MEASURED tables: what practice actually yields.
# Which table governs a given case is signalled ONLY through the four "audible"
# dimensions. The premise pair alone is therefore genuinely ambiguous, and any
# model without a working ear is capped near the frequency of derived cases.
# =============================================================================

K       = 10    # number of intelligibles
D_SENS  = 32    # sensible (visible) input dimensions
D_AUD   = 4     # audible input dimensions -- the ear's channel
D_IN    = D_SENS + D_AUD
D_Y     = 10    # dimension of an imitation (a symbol / image)

# ---- prototypes: near-orthogonal directions in sensible space ----------------
_Q, _ = np.linalg.qr(RNG.normal(size=(D_SENS, K)))
PROTO = _Q.T.copy()                                   # (K, D_SENS)

# ---- degree of being: a strict descending hierarchy, arbitrarily assigned ----
RHO = np.linspace(1.0, 0.06, K)
RHO = RHO[RNG.permutation(K)]                         # (K,)

# ---- imitations: the sensible image standing for each form ------------------
S_IMG = RNG.normal(size=(K, D_Y))
S_IMG /= np.linalg.norm(S_IMG, axis=1, keepdims=True)
S_IMG *= 0.75                                         # keep inside tanh range

# ---- motive force: how much the image moves a body. NOT tied to rank --------
MOT = RNG.uniform(0.1, 1.0, size=K)

# ---- the four rule tables ---------------------------------------------------
# A demonstration is a RULE, not a lookup. The rule here is the scholastic
# maxim al-Farabi would have met in the Organon -- CONCLUSIO SEQUITUR PARTEM
# DEBILIOREM, the conclusion follows the weaker premise: a syllogism can never
# conclude something higher in the order of being than its lowest premise.
#
#   lo, hi = the premises ordered by degree of being
#   if the premises are far apart in rank : conclude DESC_m[lo]  (descend from lo)
#   if they are close                     : conclude ALT_m[hi]   (the near case)
#
# DESC_m and ALT_m are permutations, one pair per governing table m. Learning
# this demands identifying BOTH premises, recovering their ranks, comparing
# them, branching, and applying the branch's map -- a genuine two-argument
# function -- while remaining a rule rather than an arbitrary table.

GAP_THRESHOLD = 0.34

def _perm(seed):
    return np.random.default_rng(seed).permutation(K)

DESC = np.stack([_perm(2000 + i) for i in range(4)])   # (4, K)
ALT  = np.stack([_perm(3000 + i) for i in range(4)])   # (4, K)

def conclude(mode, a, b):
    """Vectorised application of the rule."""
    ra, rb = RHO[a], RHO[b]
    lo = np.where(ra <= rb, a, b)
    hi = np.where(ra <= rb, b, a)
    gap = np.abs(ra - rb)
    far = gap > GAP_THRESHOLD
    return np.where(far, DESC[mode, lo], ALT[mode, hi])

N_MODES  = 4
MODE_P   = np.array([0.55, 0.15, 0.15, 0.15])   # how often each table governs

# ---- the audible signatures: what the ear actually hears ---------------------
AUD_SIG = np.zeros((N_MODES, D_AUD))
_qa, _ = np.linalg.qr(RNG.normal(size=(D_AUD, N_MODES - 1)))
AUD_SIG[1:] = 1.6 * _qa.T                     # mutually orthogonal signatures
# AUD_SIG[0] stays zero: "nothing heard to the contrary" -> the derivation stands.


def make_dataset(n, rng):
    """
    Generate n particulars.

    Returns a dict of arrays:
      X     (n, D_IN)  sensible dims then audible dims
      c     (n,)       index of the true conclusion
      mode  (n,)       which table governed (0 = derived, 1..3 = measured)
      a, b  (n,)       the premise pair (kept for diagnostics only)
    """
    a = rng.integers(0, K, size=n)
    b = (a + 1 + rng.integers(0, K - 1, size=n)) % K       # guarantee b != a
    mode = rng.choice(N_MODES, size=n, p=MODE_P)

    # Sensible channel: an asymmetric mixture so premise ORDER stays recoverable.
    Xs = 1.00 * PROTO[a] + 0.62 * PROTO[b]
    Xs += rng.normal(scale=0.12, size=Xs.shape)

    # Audible channel: the measured signature, heard through noise.
    Xa = AUD_SIG[mode] + rng.normal(scale=0.30, size=(n, D_AUD))

    c = conclude(mode, a, b)
    return dict(X=np.hstack([Xs, Xa]), c=c, mode=mode, a=a, b=b)


# =============================================================================
# SECTION 2 -- PARAMETERS
# =============================================================================
# Naming follows the doctrine so the code reads as the philosophy:
#   W_mat  material intellect (potential)
#   W_q, DICT   conjunction with the Active Intellect
#   W_g, W_c    the emanative cascade (each level lit by the level above)
#   W_ear, w_trust, W_ovr   the ear and its authority to overrule
#   W_dem  demonstrative head       W_imag inverse-imagination head
#   W_img  imaginative head         w_rsym rank read back off the imitation
# =============================================================================

D_H   = 128   # width of the intellect
D_E   = 16    # width of the ear's representation
K_D   = 24    # forms held by the Active Intellect (deliberately != K:
              #   the separate intelligence is not a lookup table of our labels)
L_LEV = 3     # levels of the emanative cascade
N_HEAD = 4    # rays of illumination received at once (multi-head conjunction)
D_HEAD = D_H // N_HEAD
LAMBDA = np.array([1.0 / (l + 2.0) for l in range(L_LEV)])   # light attenuates


def init_params(rng):
    """Xavier-ish initialisation, returned as a flat dict of named arrays."""
    def g(shape, fan):
        return rng.normal(scale=np.sqrt(1.0 / fan), size=shape)

    p = {}
    p["W_mat"] = g((D_SENS, D_H), D_SENS); p["b_mat"] = np.zeros(D_H)
    p["W_q"]   = g((D_H, D_H), D_H)
    p["DICT"]  = g((K_D, D_H), D_H)                    # the Active Intellect
    for l in range(L_LEV):
        p[f"W_g{l}"] = g((D_H, D_H), D_H); p[f"b_g{l}"] = np.zeros(D_H)
        p[f"W_c{l}"] = g((D_H, D_H), D_H); p[f"b_c{l}"] = np.zeros(D_H)
    p["W_ear"]   = g((D_AUD, D_E), D_AUD); p["b_ear"] = np.zeros(D_E)
    p["w_trust"] = g((D_E, 1), D_E);       p["b_trust"] = np.zeros(1)
    p["W_ovi"]   = g((D_E, D_H), D_E)      # ear -> primary principles (early)
    p["W_ovr"]   = g((D_E, D_H), D_E)      # ear -> corrective veto (late)
    p["W_dem"]  = g((D_H, K), D_H);   p["b_dem"]  = np.zeros(K)
    p["w_rank"] = g((D_H, 1), D_H);   p["b_rank"] = np.zeros(1)
    p["W_img"]  = g((D_H, D_Y), D_H); p["b_img"]  = np.zeros(D_Y)
    p["W_inv"]  = g((D_Y, D_H), D_Y); p["b_inv"]  = np.zeros(D_H)
    p["w_rsym"] = g((D_Y, 1), D_Y);   p["b_rsym"] = np.zeros(1)
    p["w_sup"]  = g((D_H, 1), D_H);   p["b_sup"]  = np.zeros(1)
    p["w_mot"]  = g((D_Y, 1), D_Y);   p["b_mot"]  = np.zeros(1)
    return p


# Loss weights. Ldem dominates; the muhakat constraints are real but secondary.
W_DEM, W_RANK, W_IMIT = 1.0, 0.5, 1.0
W_CYC, W_ORD, W_SUP, W_MOT = 0.30, 0.50, 0.30, 0.30
W_L2 = 1e-5
ORD_MARGIN = 0.05


# =============================================================================
# SECTION 3 -- FORWARD PASS
# =============================================================================

def forward(p, X, gamma=1.0, beta=1.0):
    """
    gamma : conjunction gain.  gamma=0 severs the Active Intellect (Ablation A)
    beta  : ear gain.          beta=0  deafens the ear         (Ablation B)

    Returns the head outputs plus a cache of everything backward() needs.
    """
    Xs, Xa = X[:, :D_SENS], X[:, D_SENS:]

    # --- 1. material intellect: the merely potential state ------------------
    z1 = Xs @ p["W_mat"] + p["b_mat"]
    h  = np.tanh(z1)

    # --- 1b. the ear, heard BEFORE the descent begins -----------------------
    # For al-Farabi the measured facts of practice are not an afterthought
    # bolted onto a finished proof: they are PRIMARY PRINCIPLES, and a
    # demonstration that starts from them descends differently from the start.
    # So the ear enters twice -- here, conditioning the whole descent, and
    # again at the end as an outright veto on the acquired conclusion.
    ze  = Xa @ p["W_ear"] + p["b_ear"]; e = np.tanh(ze)
    zt  = e @ p["w_trust"] + p["b_trust"]; tau = sigmoid(zt)
    ovi = e @ p["W_ovi"]                       # heard primary principles
    ovr = e @ p["W_ovr"]                       # the late veto

    # --- 2. conjunction (ittisal): actualisation by a SEPARATE substance ----
    # NOTE THE ABSENCE OF h FROM z0. This is the doctrine, not an oversight.
    # The material intellect is PURE POTENTIALITY: it contributes the
    # disposition to receive -- the query -- and no content whatsoever. Every
    # intelligible the agent ends up holding arrives from DICT, a substance
    # the agent does not own and cannot edit at inference. Wire h directly
    # into z0 and you have quietly given the agent an inner store of forms,
    # which is precisely the position al-Farabi spent his life denying.
    B_ = h.shape[0]
    q     = h @ p["W_q"]
    qh    = q.reshape(B_, N_HEAD, D_HEAD)
    Dh    = p["DICT"].reshape(K_D, N_HEAD, D_HEAD)
    score = np.einsum("bnd,knd->bnk", qh, Dh) / np.sqrt(D_HEAD)
    alpha = softmax(score, axis=-1)            # which forms illuminate us
    ill_h = np.einsum("bnk,knd->bnd", alpha, Dh)
    illum = ill_h.reshape(B_, D_H)             # the light received
    z0    = gamma * illum + beta * tau * ovi
    a_lev = [np.tanh(z0)]                      # a_lev[0] = first actualisation

    # --- 3. the emanative cascade: each level lit by the level above --------
    gates, us, zgs, zcs = [], [], [], []
    a = a_lev[0]
    for l in range(L_LEV):
        zg = a @ p[f"W_g{l}"] + p[f"b_g{l}"]; gt = sigmoid(zg)
        zc = a @ p[f"W_c{l}"] + p[f"b_c{l}"]; u  = np.tanh(zc)
        a  = a + LAMBDA[l] * gt * u            # attenuated overflow downward
        zgs.append(zg); zcs.append(zc); gates.append(gt); us.append(u)
        a_lev.append(a)
    A = a                                      # the acquired intellect

    # --- 4. the ear's veto on the acquired conclusion -----------------------
    Ap  = A + beta * tau * ovr                 # the corrected intellect

    # --- 5. the two channels and their auditors -----------------------------
    logits = Ap @ p["W_dem"] + p["b_dem"]      # demonstrative channel
    prob   = softmax(logits)
    rhat   = Ap @ p["w_rank"] + p["b_rank"]    # rank as demonstrated
    zimg   = Ap @ p["W_img"] + p["b_img"]
    shat   = np.tanh(zimg)                     # imaginative channel: the image
    zinv   = shat @ p["W_inv"] + p["b_inv"]
    Arec   = np.tanh(zinv)                     # read the image BACK to intellect
    rsym   = shat @ p["w_rsym"] + p["b_rsym"]  # rank as legible in the image
    zsup   = Ap @ p["w_sup"] + p["b_sup"]
    sup    = sigmoid(zsup)                     # does this rest on proof?
    mot    = shat @ p["w_mot"] + p["b_mot"]    # how hard does the image push?

    cache = dict(Xs=Xs, Xa=Xa, ovi=ovi, qh=qh, z1=z1, h=h, q=q, score=score, alpha=alpha,
                 illum=illum, a_lev=a_lev, gates=gates, us=us, zgs=zgs,
                 zcs=zcs, A=A, ze=ze, e=e, tau=tau, ovr=ovr, Ap=Ap,
                 logits=logits, prob=prob, rhat=rhat, shat=shat, Arec=Arec,
                 rsym=rsym, sup=sup, mot=mot, gamma=gamma, beta=beta)
    return cache


# =============================================================================
# SECTION 4 -- LOSS AND BACKWARD PASS (hand-derived)
# =============================================================================

def loss_and_grads(p, batch, gamma=1.0, beta=1.0, want_grads=True):
    """
    The composite Farabian objective.

      Ldem   the demonstration must reach the right conclusion
      Lrank  the demonstration must know the conclusion's degree of being
      Limit  the imitation must be the right image
      Lcyc   the imitation must be RE-READABLE back into the intellect that
             produced it -- an imitation you cannot invert is not an imitation,
             it is a decoration
      Lord   the imitation must PRESERVE ORDER even where it loses content --
             the one thing muhakat is never permitted to do is invert rank
      Lsup   the system must know whether its conclusion rests on proof or on
             measurement (this is what makes the ear auditable rather than
             merely powerful)
      Lmot   the system must know how hard its own image pushes
    """
    X, c = batch["X"], batch["c"]
    B = X.shape[0]
    ca = forward(p, X, gamma=gamma, beta=beta)

    rho_c = RHO[c].reshape(-1, 1)
    S_c   = S_IMG[c]
    mot_c = MOT[c].reshape(-1, 1)
    sup_t = (batch["mode"] == 0).astype(np.float64).reshape(-1, 1)

    # ---- Ldem : cross entropy -------------------------------------------
    logp = np.log(ca["prob"][np.arange(B), c] + 1e-12)
    Ldem = -np.mean(logp)

    # ---- Lrank, Limit, Lcyc, Lsup, Lmot : squared errors -----------------
    dr    = ca["rhat"] - rho_c;      Lrank = np.mean(dr ** 2)
    di    = ca["shat"] - S_c;        Limit = np.mean(di ** 2)
    dc    = ca["Arec"] - ca["Ap"];   Lcyc  = np.mean(dc ** 2)
    ds    = ca["sup"] - sup_t;       Lsup  = np.mean(ds ** 2)
    dm    = ca["mot"] - mot_c;       Lmot  = np.mean(dm ** 2)

    # ---- Lord : squared hinge on order legible in the imitation ----------
    # For every pair (i,j) whose true ranks differ, the rank READ OFF THE IMAGE
    # must respect the same order by at least ORD_MARGIN. Squared hinge keeps
    # the objective C^1, which the finite-difference check needs.
    d_    = ca["rsym"][:, 0]
    diff  = d_[:, None] - d_[None, :]
    M     = (rho_c[:, 0][:, None] > rho_c[:, 0][None, :] + 1e-9)
    nM    = max(int(M.sum()), 1)
    v     = np.maximum(0.0, ORD_MARGIN - diff)
    Lord  = np.sum(M * v ** 2) / nM

    # ---- L2 ---------------------------------------------------------------
    l2 = sum(np.sum(p[k] ** 2) for k in p if k.startswith(("W_", "w_", "DICT")))
    Lreg = W_L2 * l2

    total = (W_DEM * Ldem + W_RANK * Lrank + W_IMIT * Limit + W_CYC * Lcyc +
             W_ORD * Lord + W_SUP * Lsup + W_MOT * Lmot + Lreg)

    parts = dict(total=total, dem=Ldem, rank=Lrank, imit=Limit, cyc=Lcyc,
                 ord=Lord, sup=Lsup, mot=Lmot)
    if not want_grads:
        return total, parts, ca

    # =====================================================================
    # BACKWARD -- every derivative written out by hand
    # =====================================================================
    g = {k: np.zeros_like(v_) for k, v_ in p.items()}

    # ---- heads -> dAp and dshat ------------------------------------------
    dlogits = ca["prob"].copy()
    dlogits[np.arange(B), c] -= 1.0
    dlogits *= (W_DEM / B)
    dAp = dlogits @ p["W_dem"].T
    g["W_dem"] += ca["Ap"].T @ dlogits
    g["b_dem"] += dlogits.sum(0)

    drhat = W_RANK * 2.0 * dr / dr.size
    dAp += drhat @ p["w_rank"].T
    g["w_rank"] += ca["Ap"].T @ drhat
    g["b_rank"] += drhat.sum(0)

    dsup_ = W_SUP * 2.0 * ds / ds.size
    dzsup = dsup_ * ca["sup"] * (1.0 - ca["sup"])
    dAp += dzsup @ p["w_sup"].T
    g["w_sup"] += ca["Ap"].T @ dzsup
    g["b_sup"] += dzsup.sum(0)

    dshat = W_IMIT * 2.0 * di / di.size                      # from Limit

    dmot_ = W_MOT * 2.0 * dm / dm.size                       # from Lmot
    dshat += dmot_ @ p["w_mot"].T
    g["w_mot"] += ca["shat"].T @ dmot_
    g["b_mot"] += dmot_.sum(0)

    # Lord -> rsym -> shat
    G = M * 2.0 * v * (-1.0) / nM * W_ORD                    # dLord/ddiff
    dd = G.sum(axis=1) - G.sum(axis=0)                       # diff = d_i - d_j
    drsym = dd.reshape(-1, 1)
    dshat += drsym @ p["w_rsym"].T
    g["w_rsym"] += ca["shat"].T @ drsym
    g["b_rsym"] += drsym.sum(0)

    # Lcyc has TWO paths into the graph: through Arec, and directly onto Ap.
    dcyc  = W_CYC * 2.0 * dc / dc.size
    dArec = dcyc
    dAp  += -dcyc                                            # d/dAp of (Arec-Ap)
    dzinv = dArec * (1.0 - ca["Arec"] ** 2)
    g["W_inv"] += ca["shat"].T @ dzinv
    g["b_inv"] += dzinv.sum(0)
    dshat += dzinv @ p["W_inv"].T

    # shat -> zimg -> Ap
    dzimg = dshat * (1.0 - ca["shat"] ** 2)
    g["W_img"] += ca["Ap"].T @ dzimg
    g["b_img"] += dzimg.sum(0)
    dAp += dzimg @ p["W_img"].T

    # ---- the ear: Ap = A + beta * tau * ovr ------------------------------
    dA   = dAp.copy()
    dtau = np.sum(dAp * ca["ovr"], axis=1, keepdims=True) * beta
    dovr = dAp * ca["tau"] * beta
    g["W_ovr"] += ca["e"].T @ dovr
    de = dovr @ p["W_ovr"].T

    # ---- the cascade, unwound from the bottom level upward ---------------
    da = dA
    for l in reversed(range(L_LEV)):
        a_in = ca["a_lev"][l]
        gt, u = ca["gates"][l], ca["us"][l]
        dres = da                                   # a_out = a_in + lam*gt*u
        dgt  = dres * LAMBDA[l] * u
        du   = dres * LAMBDA[l] * gt
        dzg  = dgt * gt * (1.0 - gt)
        dzc  = du * (1.0 - u ** 2)
        g[f"W_g{l}"] += a_in.T @ dzg; g[f"b_g{l}"] += dzg.sum(0)
        g[f"W_c{l}"] += a_in.T @ dzc; g[f"b_c{l}"] += dzc.sum(0)
        da = dres + dzg @ p[f"W_g{l}"].T + dzc @ p[f"W_c{l}"].T

    # ---- conjunction: a0 = tanh(h + gamma*illum + beta*tau*ovi) ----------
    dz0    = da * (1.0 - ca["a_lev"][0] ** 2)
    dillum = dz0 * ca["gamma"]

    # the early ear path joins here
    dtau += np.sum(dz0 * ca["ovi"], axis=1, keepdims=True) * beta
    dovi = dz0 * ca["tau"] * beta
    g["W_ovi"] += ca["e"].T @ dovi
    de += dovi @ p["W_ovi"].T

    # now close the ear: trust gate, then the ear's own representation
    dzt = dtau * ca["tau"] * (1.0 - ca["tau"])
    g["w_trust"] += ca["e"].T @ dzt
    g["b_trust"] += dzt.sum(0)
    de += dzt @ p["w_trust"].T
    dze = de * (1.0 - ca["e"] ** 2)
    g["W_ear"] += ca["Xa"].T @ dze
    g["b_ear"] += dze.sum(0)

    # multi-head conjunction, unwound head by head
    Bn = dillum.shape[0]
    Dh = p["DICT"].reshape(K_D, N_HEAD, D_HEAD)
    dill_h = dillum.reshape(Bn, N_HEAD, D_HEAD)
    gD = np.einsum("bnk,bnd->knd", ca["alpha"], dill_h)          # via ill_h
    dalpha = np.einsum("bnd,knd->bnk", dill_h, Dh)
    dscore = ca["alpha"] * (dalpha - np.sum(dalpha * ca["alpha"], axis=-1,
                                            keepdims=True))
    inv = 1.0 / np.sqrt(D_HEAD)
    gD += np.einsum("bnk,bnd->knd", dscore, ca["qh"]) * inv      # via score
    g["DICT"] += gD.reshape(K_D, D_H)
    dqh = np.einsum("bnk,knd->bnd", dscore, Dh) * inv
    dq = dqh.reshape(Bn, D_H)
    g["W_q"] += ca["h"].T @ dq
    dh = dq @ p["W_q"].T          # the ONLY gradient path back into the senses

    # ---- material intellect ----------------------------------------------
    dz1 = dh * (1.0 - ca["h"] ** 2)
    g["W_mat"] += ca["Xs"].T @ dz1
    g["b_mat"] += dz1.sum(0)

    # ---- L2 ---------------------------------------------------------------
    for k in g:
        if k.startswith(("W_", "w_", "DICT")):
            g[k] += 2.0 * W_L2 * p[k]

    return total, parts, g


# =============================================================================
# SECTION 5 -- ADAM
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
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


# =============================================================================
# SECTION 6 -- EVALUATION
# =============================================================================

def evaluate(p, data, gamma=1.0, beta=1.0):
    """Accuracy overall, on derived cases, and on measured cases separately."""
    ca = forward(p, data["X"], gamma=gamma, beta=beta)
    pred = np.argmax(ca["logits"], axis=1)
    ok = (pred == data["c"])
    derived = data["mode"] == 0
    measured = ~derived
    return dict(
        acc=float(ok.mean()),
        acc_derived=float(ok[derived].mean()),
        acc_measured=float(ok[measured].mean()),
        cache=ca,
    )


def muhakat_fidelity(p, data):
    """
    Two numbers that matter to al-Farabi and to nobody else's architecture.

    cycle_cos : how well the imitation can be read back into the intellect that
                produced it. An imitation you cannot invert has come loose.
    inv_rate  : the fraction of ordered pairs whose rank the imitation INVERTS.
                Content may be lost; order may not.
    """
    ca = forward(p, data["X"])
    A, Ar = ca["Ap"], ca["Arec"]
    cc = np.sum(A * Ar, 1) / (np.linalg.norm(A, axis=1) *
                              np.linalg.norm(Ar, axis=1) + 1e-12)
    r_true = RHO[data["c"]]
    r_sym = ca["rsym"][:, 0]
    n = min(600, len(r_true))
    rt, rs = r_true[:n], r_sym[:n]
    M = rt[:, None] > rt[None, :] + 1e-9
    inv = (rs[:, None] <= rs[None, :]) & M
    return float(cc.mean()), float(inv.sum() / max(M.sum(), 1))


def nabita_score(p, data, rng):
    """
    The weed test.

    A NABITA -- al-Farabi's 'weed' -- is an inhabitant of the virtuous city who
    speaks its language convincingly while pursuing something else. Translated
    into this architecture: an image with high motive force that does NOT follow
    from the demonstration it is attached to.

    We build genuine pairs (demonstration of c, image of c) and counterfeit
    pairs (demonstration of c, image of some other form), then score

        weed = motive(image) - kappa * agreement(image, image the proof implies)

    and report the AUC of that score at separating counterfeit from genuine.
    Nothing here is trained on counterfeits: the detector is assembled out of
    parts the model learned for other reasons.
    """
    ca = forward(p, data["X"])
    shat = ca["shat"]
    n = shat.shape[0]

    def score(offered):
        agree = np.sum(offered * shat, 1) / (
            np.linalg.norm(offered, axis=1) * np.linalg.norm(shat, axis=1) + 1e-12)
        motive = (offered @ p["w_mot"][:, 0]) + p["b_mot"][0]
        return motive - 1.5 * agree

    genuine = score(S_IMG[data["c"]])
    wrong_c = (data["c"] + 1 + rng.integers(0, K - 1, size=n)) % K
    counterf = score(S_IMG[wrong_c])

    # AUC by rank statistic (Mann-Whitney U), counterfeit expected to score high
    allv = np.concatenate([counterf, genuine])
    order = np.argsort(allv)
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    n1 = len(counterf); n2 = len(genuine)
    auc = (ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2)
    return float(auc)


# =============================================================================
# SECTION 7 -- MANDATORY FINITE-DIFFERENCE GRADIENT CHECK
# =============================================================================

def gradient_check(p, batch, n_probe=4, eps=1e-5):
    """
    Central differences on a random handful of coordinates in EVERY parameter
    block. The whole objective is C^1 by construction (squared hinge, tanh,
    sigmoid, softmax), so double-precision central differences are meaningful.

    Two numbers are reported per block, because a single one is misleading:

      max |analytic - numeric|                        (absolute)
      max |analytic - numeric| / (|a|+|n|)  but ONLY over coordinates whose
                                            gradient exceeds 1e-4 (relative)

    The restriction matters and is not a way of hiding a failure. With
    eps = 1e-5 in double precision the difference quotient carries an absolute
    error of order 1e-10 to 1e-11 no matter how small the true derivative is.
    A coordinate whose true derivative is 1e-6 therefore has ~1e-5 of purely
    numerical relative noise, and grading it on relative error measures the
    arithmetic of the quotient rather than the correctness of the backward
    pass. Above 1e-4 the relative figure is meaningful, so that is where the
    relative test is applied; the ABSOLUTE test is applied to every probe
    without exception, and it is the stronger of the two.
    """
    _, _, an = loss_and_grads(p, batch)
    rng = np.random.default_rng(7)
    worst_rel, worst_key, worst_abs = 0.0, None, 0.0
    rows = []
    for key in sorted(p.keys()):
        arr = p[key]
        idxs = [tuple(rng.integers(0, s) for s in arr.shape)
                for _ in range(min(n_probe, arr.size))]
        rels, abss = [0.0], [0.0]
        for idx in idxs:
            orig = arr[idx]
            arr[idx] = orig + eps
            lp, _, _ = loss_and_grads(p, batch, want_grads=False)
            arr[idx] = orig - eps
            lm, _, _ = loss_and_grads(p, batch, want_grads=False)
            arr[idx] = orig
            num = (lp - lm) / (2 * eps)
            ana = an[key][idx]
            abss.append(abs(num - ana))
            if max(abs(num), abs(ana)) > 1e-4:
                rels.append(abs(num - ana) / (abs(num) + abs(ana)))
        r, a = max(rels), max(abss)
        rows.append((key, r, a))
        if r > worst_rel:
            worst_rel, worst_key = r, key
        worst_abs = max(worst_abs, a)
    return worst_rel, worst_key, worst_abs, rows


# =============================================================================
# SECTION 8 -- TRAINING
# =============================================================================

def train(p, tr, steps=7000, bs=128, lr=4e-3, seed=0, log_every=1000,
          verbose=True):
    """
    Plain Adam with linear warm-up and cosine decay. Nothing exotic: the point
    of this file is the architecture, so the optimiser is kept boring and
    identical between the main engine and the control engine of Test 7.
    """
    opt = Adam(p, lr=lr)
    rng = np.random.default_rng(seed)
    n = len(tr["c"])
    warm = 300
    hist = []
    for s in range(1, steps + 1):
        if s <= warm:
            opt.lr = lr * s / warm
        else:
            t = (s - warm) / max(1, steps - warm)
            opt.lr = 0.05 * lr + 0.95 * lr * 0.5 * (1 + np.cos(np.pi * t))
        i = rng.integers(0, n, size=bs)
        batch = {k: tr[k][i] for k in ("X", "c", "mode")}
        tot, parts, gr = loss_and_grads(p, batch)
        opt.step(p, gr)
        hist.append(tot)
        if verbose and (s % log_every == 0 or s == 1):
            print(f"   step {s:5d} | total {tot:7.4f} | dem {parts['dem']:6.4f}"
                  f" | imit {parts['imit']:6.4f} | cyc {parts['cyc']:6.4f}"
                  f" | ord {parts['ord']:7.5f}")
    return hist


# =============================================================================
# SECTION 9 -- MAIN: EIGHT SELF-TESTS
# =============================================================================

def main():
    t0 = time.time()
    print("=" * 79)
    print("  THE MUHAKAT ENGINE -- al-Farabi (c.870-950/1), Chapter 0213")
    print("  Two-channel conjunction network, pure NumPy")
    print("=" * 79)

    rng = np.random.default_rng(4242)
    tr = make_dataset(24000, rng)
    te = make_dataset(3000, rng)
    print(f"\n[world] {K} intelligibles | {D_SENS} sensible + {D_AUD} audible dims"
          f" | {len(tr['c'])} train / {len(te['c'])} test")
    print(f"[world] derived cases {float((tr['mode']==0).mean()):.3f} | "
          f"measured cases {float((tr['mode']!=0).mean()):.3f}")

    p = init_params(np.random.default_rng(11))
    npar = sum(v.size for v in p.values())
    print(f"[model] D_H={D_H}  cascade levels={L_LEV}  Active-Intellect forms={K_D}"
          f"  parameters={npar}")

    # ---- TEST 1: shapes ---------------------------------------------------
    print("\n--- TEST 1: forward shapes ------------------------------------")
    ca = forward(p, te["X"][:8])
    assert ca["logits"].shape == (8, K)
    assert ca["shat"].shape == (8, D_Y)
    assert ca["Arec"].shape == (8, D_H)
    assert ca["alpha"].shape == (8, N_HEAD, K_D)
    print(f"    logits {ca['logits'].shape}  image {ca['shat'].shape}  "
          f"re-read {ca['Arec'].shape}  conjunction {ca['alpha'].shape}   PASS")

    # ---- TEST 2: conjunction is a proper distribution ---------------------
    print("\n--- TEST 2: conjunction weights form a distribution -----------")
    rs = ca["alpha"].sum(-1)
    assert np.allclose(rs, 1.0) and (ca["alpha"] >= 0).all()
    print(f"    row sums in [{rs.min():.12f}, {rs.max():.12f}]   PASS")

    # ---- TEST 3: gradient check (MANDATORY) -------------------------------
    print("\n--- TEST 3: finite-difference gradient check ------------------")
    gb = {k: tr[k][:24] for k in ("X", "c", "mode")}
    wrel, wkey, wabs, rows = gradient_check(p, gb)
    for k, r, a in rows:
        print(f"    {k:9s} rel {r:.3e}   abs {a:.3e}")
    print(f"    WORST relative: {wkey} -> {wrel:.3e}   |   worst absolute: {wabs:.3e}")
    assert wrel < 1e-6, f"gradient check FAILED at {wkey}: rel={wrel}"
    assert wabs < 1e-9, f"gradient check FAILED: abs={wabs}"
    print("    every analytic gradient agrees with central differences   PASS")

    # ---- training ---------------------------------------------------------
    print("\n--- training the full engine ----------------------------------")
    hist = train(p, tr, steps=7000, seed=5)

    # ---- TEST 4: the loss actually came down, and it generalises ----------
    print("\n--- TEST 4: optimisation and generalisation -------------------")
    first, last = float(np.mean(hist[:50])), float(np.mean(hist[-50:]))
    ev = evaluate(p, te)
    print(f"    loss {first:.4f} -> {last:.4f}   test accuracy {ev['acc']:.4f}")
    assert last < first * 0.5
    assert ev["acc"] > 0.90
    print("    PASS")

    # ---- TEST 5: ABLATION A -- sever the conjunction ----------------------
    print("\n--- TEST 5: ABLATION A -- sever conjunction (gamma=0) ---------")
    ev_a = evaluate(p, te, gamma=0.0)
    drop = ev["acc"] - ev_a["acc"]
    print(f"    with the Active Intellect : {ev['acc']:.4f}")
    print(f"    severed from it           : {ev_a['acc']:.4f}   (drop {drop:.4f})")
    assert drop > 0.20, "conjunction was not load-bearing"
    print("    the potential intellect does not degrade gracefully -- it")
    print("    collapses. Actualisation was never stored in the agent.   PASS")

    # ---- TEST 6: ABLATION B -- deafen the ear ----------------------------
    print("\n--- TEST 6: ABLATION B -- deafen the ear (beta=0) -------------")
    ev_b = evaluate(p, te, beta=0.0)
    print(f"    hearing  : overall {ev['acc']:.4f} | derived "
          f"{ev['acc_derived']:.4f} | measured {ev['acc_measured']:.4f}")
    print(f"    deafened : overall {ev_b['acc']:.4f} | derived "
          f"{ev_b['acc_derived']:.4f} | measured {ev_b['acc_measured']:.4f}")
    d_der = ev["acc_derived"] - ev_b["acc_derived"]
    d_mea = ev["acc_measured"] - ev_b["acc_measured"]
    print(f"    damage on derived cases  : {d_der:+.4f}")
    print(f"    damage on measured cases : {d_mea:+.4f}")
    assert d_mea > 0.30 and d_mea > 3 * max(d_der, 0.01)
    print("    the damage is LOCALISED on exactly the cases where practice")
    print("    contradicts the received table. The ear is not decoration.  PASS")

    # ---- TEST 7: muhakat fidelity ----------------------------------------
    print("\n--- TEST 7: muhakat fidelity (invertibility and order) --------")
    cc, inv = muhakat_fidelity(p, te)
    print(f"    imitation re-read into intellect, cosine : {cc:.4f}")
    print(f"    rank inversions committed by the image   : {inv:.4f}")

    print("    [control] retraining an identical engine with the order")
    print("    constraint switched off, to show the constraint is doing work...")
    global W_ORD
    saved = W_ORD
    W_ORD = 0.0
    p_ctl = init_params(np.random.default_rng(11))
    train(p_ctl, tr, steps=7000, seed=5, verbose=False)
    cc_c, inv_c = muhakat_fidelity(p_ctl, te)
    W_ORD = saved
    ev_c = evaluate(p_ctl, te)
    cc_r, inv_r = muhakat_fidelity(init_params(np.random.default_rng(77)), te)
    print(f"    control  (no order constraint) : accuracy {ev_c['acc']:.4f} | "
          f"re-read cosine {cc_c:.4f} | inversions {inv_c:.4f}")
    print(f"    baseline (untrained weights)   : "
          f"re-read cosine {cc_r:.4f} | inversions {inv_r:.4f}")
    assert cc > 0.70 and cc > 3.0 * abs(cc_r)
    assert inv < 0.10 and inv < 0.25 * inv_c
    print(f"    invertibility {cc:.4f} against an untrained {cc_r:.4f};")
    print(f"    inversions {inv:.4f} against an unconstrained {inv_c:.4f}, at")
    print(f"    a cost in accuracy of {ev_c['acc'] - ev['acc']:+.4f} -- keeping")
    print("    the order of being legible in the image is nearly free.   PASS")

    # ---- TEST 8: the weed detector ---------------------------------------
    print("\n--- TEST 8: nabita ('weed') detection -------------------------")
    auc = nabita_score(p, te, np.random.default_rng(99))
    print(f"    AUC separating counterfeit images from genuine ones: {auc:.4f}")
    assert auc > 0.90
    print("    a moving image with no demonstration behind it is detectable")
    print("    from parts the model learned for other purposes.   PASS")

    # ---- also report the proof-vs-measurement self-report -----------------
    ca_f = forward(p, te["X"])
    sup_pred = (ca_f["sup"][:, 0] > 0.5)
    sup_true = (te["mode"] == 0)
    print(f"\n[audit] the engine correctly reports whether a given conclusion")
    print(f"        rests on demonstration or on measurement: "
          f"{float((sup_pred == sup_true).mean()):.4f} accuracy")

    print("\n" + "=" * 79)
    print(f"  ALL 8 TESTS PASSED  ({time.time() - t0:.1f}s)")
    print("=" * 79)


if __name__ == "__main__":
    main()

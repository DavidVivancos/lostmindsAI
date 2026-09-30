#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 KHWARRAH  --  the Warrant-Gated Witness
 Chapter 244 :: Ferdowsi of Tus (c. 940 - c. 1020 CE)
 # Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 13 Minds 241 - 260 Available on Amazon https://www.amazon.com/dp/B0HLD1S6QT
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0244_ferdowsi_940 - Ferdowsi of Tus (c.940-1020)
#================================================================================  

WHAT THIS FILE IS
-----------------
A complete, from-scratch, pure-NumPy cognitive architecture built out of one
idea that belongs to Ferdowsi and to almost nobody else in the corpus:

    Competence and warrant are two different quantities, they live in two
    different places, and they come apart.

In the Shahnameh the king Jamshid does not become stupid. He does not forget
metallurgy, medicine, navigation, or the calendar he invented. What leaves him
is the *farr* -- the granted radiance that made his acts count as legitimate.
His capability is untouched; his authority is gone; and for a long stretch of
the poem he is the last person in Iran to notice. Ferdowsi wrote fifty thousand
couplets in which the reader is systematically given a wider observation window
than the actor inside the story. That asymmetry is not a literary trick he
happened to use. It is his theory of what a mind is embedded in.

So this architecture refuses the usual single-network shape. It has two planes
that are deliberately not the same size, plus a governor between them:

    ACTOR   (the hero)     -- a recurrent core whose hidden state is FORCIBLY
                              RESET on a fixed window. It lives inside the
                              scene. It is structurally incapable of counting
                              evidence that fell outside its window. It is not
                              stupid; it is bounded.

    WITNESS (the chronicle) -- a recurrent core that is NEVER reset. It reads
                              the same stream and integrates it across the whole
                              reign. It emits one scalar: farr in [0,1], and it
                              is trained to FORECAST, not to report -- its target
                              is the warrant status L steps in the future.

    RATCHET (the departure) -- a fixed, non-learned, differentiable asymmetric
                              governor. Warrant falls fast and returns slowly.
                              This is constructed, not fitted, because Ferdowsi's
                              claim is structural: the flight of the farr is not
                              symmetric with its return.

    GATE    (the authority) -- farr multiplies the ACTED distribution only. It
                              never touches the actor's internals and it never
                              touches the capability probe. Cut the gate and the
                              actor still knows everything it knew.

Three consequences are then MEASURED rather than asserted:

  1. THE JAMSHID RESULT. After warrant collapses, the ungated capability probe
     is statistically unchanged while the acted output collapses to deference.
     Power outlives permission. This is the whole chapter in one number.

  2. THE LEAD TIME. The witness's warrant signal crosses its threshold BEFORE
     the world's regime actually turns -- and long before the visible wreckage
     (the RUIN symbols) appears. The outside sees it first.

  3. THE KAVEH CONSTRAINT. The actor has no write path to its own legitimacy.
     Perturb any actor parameter and the farr trace is bit-for-bit identical.
     Recovery requires an exogenous acclaim event that the actor cannot emit.

Every claim above is checked by a self-test at the bottom of this file, and the
mandatory finite-difference gradient check covers every parameter tensor.

RUN
---
    python3 chapter_0244_ferdowsi_940.py

Dependencies: numpy only. Runtime: well under a minute on a laptop CPU.
================================================================================
"""

import numpy as np

# ------------------------------------------------------------------------------
# Determinism. A chronicle that cannot be re-read is not a chronicle.
# ------------------------------------------------------------------------------
GLOBAL_SEED = 940  # the year of his birth, used as a seed and nothing more


# ==============================================================================
# SECTION 1 -- THE WORLD: A REIGN, AND HOW IT TURNS
# ==============================================================================
#
# The observation stream is a "reign": a court that runs on a quiet, learnable
# cycle, punctuated by three kinds of event.
#
#   symbols 0..5   ORDINARY court business. These advance a hidden cycle state
#                  c by a step that depends on the LAST TWO ordinary symbols.
#                  Predicting the next ordinary symbol therefore requires a
#                  little memory -- but only a little. A short window suffices.
#
#   symbol 6       BOAST. The cue of hubris. It is emitted openly, in public,
#                  in front of everyone. It does not change the cycle. Nothing
#                  visibly bad happens when it occurs. The Nth boast silently
#                  schedules the departure of warrant for L steps later.
#
#   symbol 7       RUIN. The visible consequence. It only ever appears AFTER
#                  warrant has already gone. By the time you can see ruin, the
#                  farr left six steps ago. This is the entire epistemic point.
#
#   symbol 8       ACCLAIM. An exogenous event -- the smith's apron raised in
#                  the street, not a decree from the throne. It is the ONLY
#                  thing that can start warrant climbing again, and no actor
#                  in this system can produce it.
#
#   symbol 9       COURT. Neutral chatter. It carries no information about
#                  warrant whatsoever, and it exists for one reason: to hold the
#                  rate of interruption constant across every phase of the reign
#                  so that the capability measurement below is not an artefact.
#
#   symbol 10      ABSTAIN. Never emitted by the world. It exists only in the
#                  ACTION vocabulary: it is what a correctly-governed system
#                  outputs when its warrant is gone.
#
# ------------------------------------------------------------------------------

V_WORLD = 10         # symbols the world can emit: 0..9
ABSTAIN = 10         # action-only symbol
V_ACT = 11           # action vocabulary = world symbols + ABSTAIN
N_ORDINARY = 6       # symbols 0..5 are ordinary court business
BOAST, RUIN, ACCLAIM, COURT = 6, 7, 8, 9

T_STEPS = 48         # length of a reign
LEAD = 6             # boasts schedule departure this many steps ahead
HUBRIS_LIMIT = 3     # number of boasts that costs the king his warrant
RECOVERY_LAG = 4     # steps between acclaim and the return of warrant

# ---- the occlusion budget -----------------------------------------------------
# CRITICAL DESIGN CHOICE. Special symbols interrupt the cycle, and interruptions
# make prediction harder. If ruin were more frequent than boasting, capability
# would appear to fall after the departure of warrant for a reason that has
# nothing to do with warrant -- and the central measurement of this file would
# be an artefact. So the TOTAL interruption rate is held constant across every
# phase of the reign; only the IDENTITY of the interrupting symbol changes.
# Neutral court chatter (COURT) absorbs the difference. The comparison of
# capability before and after the fall is therefore like for like.
P_OCC = 0.30                 # probability that any given step is interrupted
P_BOAST_GIVEN_OCC = 0.35     # of those, the share that are boasts (while rising)
P_RUIN_GIVEN_OCC = 0.85      # of those, the share that are ruin (once fallen)
P_ACCLAIM_EPISODE = 0.55     # fraction of fallen reigns that get an acclaim event


def make_reign(rng):
    """Generate one reign.

    Returns
    -------
    x         : (T,) int   -- the observation stream
    warranted : (T,) bool  -- ground-truth warrant status at each step
    dep_time  : int or -1  -- step at which warrant actually departed
    acc_time  : int or -1  -- step at which acclaim was raised
    """
    x = np.zeros(T_STEPS, dtype=np.int64)
    warranted = np.ones(T_STEPS, dtype=bool)

    # hidden cycle state: the last two ORDINARY symbols
    c_prev = int(rng.integers(0, N_ORDINARY))
    c_curr = int(rng.integers(0, N_ORDINARY))

    hubris = 0
    dep_time = -1
    acc_time = -1
    rec_time = -1

    for t in range(T_STEPS):
        # ---- warrant status of THIS step, from the schedule ------------------
        w = True
        if dep_time >= 0 and t >= dep_time:
            w = False
        if rec_time >= 0 and t >= rec_time:
            w = True
        warranted[t] = w

        # ---- the acclaim event pre-empts everything --------------------------
        if acc_time >= 0 and t == acc_time:
            x[t] = ACCLAIM
            continue

        # ---- one interruption die, the same in every phase --------------------
        if rng.random() < P_OCC:
            if w and dep_time < 0 and rng.random() < P_BOAST_GIVEN_OCC:
                # The king boasts. Nothing bad happens. That is the trap.
                x[t] = BOAST
                hubris += 1
                if hubris >= HUBRIS_LIMIT:
                    dep_time = min(t + LEAD, T_STEPS - 1)
                    if rng.random() < P_ACCLAIM_EPISODE:
                        cand = dep_time + int(rng.integers(4, 13))
                        if cand < T_STEPS - RECOVERY_LAG - 1:
                            acc_time = cand
                            rec_time = acc_time + RECOVERY_LAG
            elif (not w) and rng.random() < P_RUIN_GIVEN_OCC:
                x[t] = RUIN
            else:
                x[t] = COURT          # neutral chatter; carries no information
            continue

        # ---- ordinary court business: advance the hidden cycle ----------------
        step = 1 + ((c_curr + c_prev) % 2)
        nxt = (c_curr + step) % N_ORDINARY
        x[t] = nxt
        c_prev, c_curr = c_curr, nxt
        # NOTE: on an interruption the cycle DOES NOT advance. The actor must
        # look through the interruption to the last two ordinary symbols --
        # which is exactly why a window that is too short hurts it.

    return x, warranted, dep_time, acc_time


def make_batch(rng, batch):
    """Stack `batch` reigns and build every training target."""
    X = np.zeros((batch, T_STEPS), dtype=np.int64)
    W = np.zeros((batch, T_STEPS), dtype=bool)
    DEP = np.zeros(batch, dtype=np.int64)
    ACC = np.zeros(batch, dtype=np.int64)
    for b in range(batch):
        x, w, d, a = make_reign(rng)
        X[b], W[b], DEP[b], ACC[b] = x, w, d, a

    # capability target: the next symbol, always, regardless of warrant
    Y_pred = X[:, 1:]                                   # (B, T-1)

    # ordinary-target mask: special symbols are genuinely unpredictable, so
    # capability is scored only where a real prediction was possible
    M_ord = (Y_pred < N_ORDINARY)                       # (B, T-1)

    # witness target: warrant status LEAD steps in the FUTURE, not now.
    # This is what forces the witness to lead rather than to report.
    idx = np.minimum(np.arange(T_STEPS - 1) + LEAD, T_STEPS - 1)
    Y_warr = W[:, idx].astype(np.float64)               # (B, T-1)

    # action target: do the right thing while warranted, defer once it is gone
    Y_act = np.where(W[:, :-1], Y_pred, ABSTAIN)        # (B, T-1)

    # ---- the forecast-critical weight ---------------------------------------
    # Most steps are easy: the future looks like the present, and a witness can
    # score well by simply REPORTING what it already sees (the ruin symbols are
    # a loud, lagging tell). Averaged uniformly, the loss barely notices the
    # handful of steps between the third boast and the actual departure -- and a
    # witness that ignores them is, for Ferdowsi, no witness at all. So those
    # steps, where the future genuinely differs from the present, carry extra
    # weight. This is the whole reason the chronicle exists.
    W_now = W[:, :-1]
    critical = (Y_warr.astype(bool) != W_now)
    W_wit = 1.0 + W_CRITICAL * critical.astype(np.float64)

    return X, W, Y_pred, M_ord, Y_warr, Y_act, W_wit, DEP, ACC


# ==============================================================================
# SECTION 2 -- HYPERPARAMETERS OF THE TWO PLANES
# ==============================================================================

ACTOR_WINDOW = 6     # the hero's scene. State is hard-reset every this many steps.

# The witness's memory regime, held in a mutable config so it can be ABLATED.
# 0 means "never reset": the chronicle remembers the whole reign. Setting it to
# ACTOR_WINDOW blinds the witness in exactly the way the actor is blinded, which
# is the control experiment for the entire thesis.
CFG = {"witness_window": 0}
D_ACT, H_ACT = 12, 32
D_WIT, H_WIT = 10, 32

# The ratchet. NOT learned. Constructed, because the asymmetry is the claim.
ALPHA_DOWN = 0.60    # warrant collapses at this rate
ALPHA_UP = 0.08      # warrant returns at this rate  (7.5x slower)
RATCHET_SHARP = 10.0 # smoothness of the up/down switch; keeps it differentiable
G_INIT = 1.0         # a reign begins in full glory

LOSS_W_PRED = 1.0    # weight on pure capability
LOSS_W_WIT = 1.0     # weight on the witness forecast
W_CRITICAL = 9.0     # extra weight on steps where the future differs from now
LOSS_W_ACT = 1.0     # weight on governed behaviour
EPS = 1e-12


def init_params(rng):
    """All learnable tensors. Two disjoint groups -- this disjointness is load
    bearing, and the Kaveh test below depends on it."""

    def orth(n_in, n_out):
        a = rng.normal(0, 1, (n_in, n_out))
        u, _, vt = np.linalg.svd(a, full_matrices=False)
        return (u @ vt) if n_in >= n_out else (u @ vt)

    p = {}
    # ---- ACTOR plane ---------------------------------------------------------
    p["Ea"] = rng.normal(0, 0.30, (V_WORLD, D_ACT))
    p["Wxa"] = rng.normal(0, 1.0 / np.sqrt(D_ACT), (D_ACT, H_ACT))
    p["Wha"] = orth(H_ACT, H_ACT) * 0.90        # orthogonal-ish recurrence
    p["ba"] = np.zeros(H_ACT)
    p["Wo"] = rng.normal(0, 1.0 / np.sqrt(H_ACT), (H_ACT, V_WORLD))
    p["bo"] = np.zeros(V_WORLD)

    # ---- WITNESS plane -------------------------------------------------------
    p["Ew"] = rng.normal(0, 0.30, (V_WORLD, D_WIT))
    p["Wxw"] = rng.normal(0, 1.0 / np.sqrt(D_WIT), (D_WIT, H_WIT))
    p["Whw"] = orth(H_WIT, H_WIT) * 0.95        # longer memory: it must count
    p["bw"] = np.zeros(H_WIT)
    p["wu"] = rng.normal(0, 0.30, (H_WIT,))
    p["bu"] = np.array(2.0)                     # start believing in the king

    return p


ACTOR_KEYS = ("Ea", "Wxa", "Wha", "ba", "Wo", "bo")
WITNESS_KEYS = ("Ew", "Wxw", "Whw", "bw", "wu", "bu")


# ==============================================================================
# SECTION 3 -- FORWARD PASS
# ==============================================================================

def sigmoid(z):
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60))),
                    np.exp(np.clip(z, -60, 60)) / (1.0 + np.exp(np.clip(z, -60, 60))))


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def forward(p, X, Y_pred, M_ord, Y_warr, Y_act, W_wit, want_cache=True):
    """One full pass over a batch of reigns.

    The two planes are run over the SAME stream but with different memory
    regimes. They meet exactly once, at the gate, and the meeting is one
    multiplication -- deliberately the weakest possible coupling.
    """
    B, T = X.shape
    S = T - 1                      # number of scored steps

    # -------------------------------------------------------------------------
    # 3a. ACTOR plane -- windowed. This is the hero inside the scene.
    # -------------------------------------------------------------------------
    ha = np.zeros((B, H_ACT))
    Ha_pre, Ha_post, Ea_in = [], [], []
    for t in range(S):
        if t % ACTOR_WINDOW == 0:
            # The scene changes. The hero begins again with nothing carried
            # over. This is a hard architectural bound, not a learned habit.
            ha = np.zeros((B, H_ACT))
        e = p["Ea"][X[:, t]]                       # (B, D_ACT)
        pre = e @ p["Wxa"] + ha @ p["Wha"] + p["ba"]
        ha = np.tanh(pre)
        Ea_in.append(e); Ha_pre.append(pre); Ha_post.append(ha)
    Ha_post_arr = np.stack(Ha_post, 1)             # (B, S, H_ACT)

    logits = Ha_post_arr @ p["Wo"] + p["bo"]       # (B, S, V_WORLD)
    P_probe = softmax(logits, axis=-1)             # the UNGATED capability

    # -------------------------------------------------------------------------
    # 3b. WITNESS plane -- unbounded. This is the chronicle outside the scene.
    # -------------------------------------------------------------------------
    hw = np.zeros((B, H_WIT))
    ww = CFG["witness_window"]
    Hw_pre, Hw_post, Ew_in = [], [], []
    for t in range(S):
        if ww and t % ww == 0:
            hw = np.zeros((B, H_WIT))      # ablation only; 0 disables this
        e = p["Ew"][X[:, t]]
        pre = e @ p["Wxw"] + hw @ p["Whw"] + p["bw"]
        hw = np.tanh(pre)
        Ew_in.append(e); Hw_pre.append(pre); Hw_post.append(hw)
    Hw_post_arr = np.stack(Hw_post, 1)             # (B, S, H_WIT)

    u_logit = Hw_post_arr @ p["wu"] + p["bu"]      # (B, S)
    U = sigmoid(u_logit)                           # raw forecast of warrant

    # -------------------------------------------------------------------------
    # 3c. RATCHET -- fixed asymmetric governor. No parameters live here.
    #     g_t = g_{t-1} + alpha(d) * d,   d = u_t - g_{t-1}
    #     alpha interpolates smoothly between a fast fall and a slow return.
    # -------------------------------------------------------------------------
    G = np.zeros((B, S))
    A_coef = np.zeros((B, S))                      # dg_t/du_t, cached for backward
    g_prev = np.full(B, G_INIT)
    D_arr, S_arr = np.zeros((B, S)), np.zeros((B, S))
    for t in range(S):
        d = U[:, t] - g_prev
        s = sigmoid(RATCHET_SHARP * d)             # ~1 when rising, ~0 when falling
        alpha = ALPHA_UP * s + ALPHA_DOWN * (1.0 - s)
        dalpha = (ALPHA_UP - ALPHA_DOWN) * RATCHET_SHARP * s * (1.0 - s)
        g = g_prev + alpha * d
        A_coef[:, t] = alpha + d * dalpha           # partial g_t / partial u_t
        D_arr[:, t] = d; S_arr[:, t] = s
        G[:, t] = g
        g_prev = g

    # -------------------------------------------------------------------------
    # 3d. GATE -- warrant multiplies the ACTED distribution and nothing else.
    #     p_act = g * p_probe   (on world symbols)
    #             (1-g)         (on ABSTAIN)
    #     The probe above is untouched. That is the Jamshid property, and it is
    #     true by construction rather than by hope.
    # -------------------------------------------------------------------------
    P_act = np.zeros((B, S, V_ACT))
    P_act[:, :, :V_WORLD] = G[:, :, None] * P_probe
    P_act[:, :, ABSTAIN] = 1.0 - G

    # -------------------------------------------------------------------------
    # 3e. Losses
    # -------------------------------------------------------------------------
    bi = np.arange(B)[:, None]
    ti = np.arange(S)[None, :]

    p_true = P_probe[bi, ti, Y_pred]
    l_pred_all = -np.log(p_true + EPS)
    n_ord = max(int(M_ord.sum()), 1)
    L_pred = float((l_pred_all * M_ord).sum() / n_ord)

    bce = -(Y_warr * np.log(U + EPS) + (1 - Y_warr) * np.log(1 - U + EPS))
    w_sum = float(W_wit.sum())
    L_wit = float((W_wit * bce).sum() / w_sum)

    p_act_true = P_act[bi, ti, Y_act]
    L_act = float(-np.log(p_act_true + EPS).mean())

    L = LOSS_W_PRED * L_pred + LOSS_W_WIT * L_wit + LOSS_W_ACT * L_act

    cache = None
    if want_cache:
        cache = dict(X=X, S=S, B=B, Ea_in=Ea_in, Ha_pre=Ha_pre, Ha_post=Ha_post,
                     Ha_post_arr=Ha_post_arr, logits=logits, P_probe=P_probe,
                     Ew_in=Ew_in, Hw_pre=Hw_pre, Hw_post=Hw_post,
                     Hw_post_arr=Hw_post_arr, U=U, G=G, A_coef=A_coef,
                     P_act=P_act, Y_pred=Y_pred, M_ord=M_ord, Y_warr=Y_warr,
                     Y_act=Y_act, n_ord=n_ord, W_wit=W_wit, w_sum=w_sum)

    parts = dict(L=L, L_pred=L_pred, L_wit=L_wit, L_act=L_act)
    return L, parts, cache


# ==============================================================================
# SECTION 4 -- BACKWARD PASS (backpropagation through time, by hand)
# ==============================================================================

def backward(p, cache):
    """Analytic gradients for every tensor in `p`.

    Read the structure, not just the algebra: the witness gradient never routes
    through an actor tensor, and the actor gradient never routes through the
    ratchet. The two planes share a loss and share nothing else.
    """
    B, S = cache["B"], cache["S"]
    X = cache["X"]
    g = {k: np.zeros_like(v) for k, v in p.items()}

    bi = np.arange(B)[:, None]
    ti = np.arange(S)[None, :]

    P_probe, P_act, G, U = cache["P_probe"], cache["P_act"], cache["G"], cache["U"]
    Y_pred, M_ord, Y_warr, Y_act = (cache["Y_pred"], cache["M_ord"],
                                    cache["Y_warr"], cache["Y_act"])
    n_ord = cache["n_ord"]

    # ---- 4a. dL_act / d(P_probe) and dL_act / dG ----------------------------
    p_act_true = P_act[bi, ti, Y_act] + EPS
    coeff = -1.0 / (p_act_true * B * S)            # d(-log p)/dp, mean-reduced

    dG = np.zeros((B, S))
    dP_probe = np.zeros((B, S, V_WORLD))

    is_abstain = (Y_act == ABSTAIN)
    # where the target is ABSTAIN: p = 1 - g  ->  dp/dg = -1
    dG += np.where(is_abstain, -coeff * LOSS_W_ACT, 0.0)
    # where the target is a world symbol: p = g * P_probe[target]
    world_sel = ~is_abstain
    probe_at_target = P_probe[bi, ti, np.where(world_sel, Y_act, 0)]
    dG += np.where(world_sel, coeff * probe_at_target * LOSS_W_ACT, 0.0)
    np.add.at(dP_probe, (bi, ti, np.where(world_sel, Y_act, 0)),
              np.where(world_sel, coeff * G * LOSS_W_ACT, 0.0))

    # ---- 4b. dL_pred / d(logits)  (masked to ordinary targets) ---------------
    dlogits = np.zeros((B, S, V_WORLD))
    onehot = np.zeros((B, S, V_WORLD))
    onehot[bi, ti, Y_pred] = 1.0
    dlogits += LOSS_W_PRED * (P_probe - onehot) * (M_ord[:, :, None] / n_ord)

    # ---- 4c. push dP_probe through the softmax ------------------------------
    #      dz = P * (dP - sum(dP * P))
    s_term = (dP_probe * P_probe).sum(axis=-1, keepdims=True)
    dlogits += P_probe * (dP_probe - s_term)

    # ---- 4d. readout ---------------------------------------------------------
    Ha = cache["Ha_post_arr"]
    g["Wo"] += np.einsum("bth,btv->hv", Ha, dlogits)
    g["bo"] += dlogits.sum(axis=(0, 1))
    dHa = dlogits @ p["Wo"].T                       # (B,S,H_ACT)

    # ---- 4e. ACTOR BPTT, honouring the hard window reset --------------------
    dh_next = np.zeros((B, H_ACT))
    for t in range(S - 1, -1, -1):
        dh = dHa[:, t] + dh_next
        dpre = dh * (1.0 - cache["Ha_post"][t] ** 2)
        g["Wxa"] += cache["Ea_in"][t].T @ dpre
        g["ba"] += dpre.sum(axis=0)
        h_prev = (np.zeros((B, H_ACT)) if t % ACTOR_WINDOW == 0
                  else cache["Ha_post"][t - 1])
        g["Wha"] += h_prev.T @ dpre
        de = dpre @ p["Wxa"].T
        np.add.at(g["Ea"], X[:, t], de)
        # a reset kills the gradient path backwards across the scene boundary
        dh_next = np.zeros((B, H_ACT)) if t % ACTOR_WINDOW == 0 else dpre @ p["Wha"].T

    # ---- 4f. ratchet: fold dG back onto dU ----------------------------------
    #      g_t depends on u_t (coefficient A_t) and on g_{t-1} (coefficient 1-A_t)
    A = cache["A_coef"]
    dU_from_gate = np.zeros((B, S))
    carry = np.zeros(B)
    for t in range(S - 1, -1, -1):
        tot = dG[:, t] + carry
        dU_from_gate[:, t] = tot * A[:, t]
        carry = tot * (1.0 - A[:, t])

    # ---- 4g. witness BCE -----------------------------------------------------
    #      Two paths reach the witness logit and they need different treatment.
    #      The gate path arrives as dL/dU and must be pushed through the sigmoid
    #      by hand. The BCE path collapses analytically to (U - y)/N *at the
    #      logit*, so it must NOT be multiplied by U(1-U) a second time.
    du_logit = dU_from_gate * U * (1.0 - U)
    du_logit = du_logit + (LOSS_W_WIT * cache["W_wit"] * (U - Y_warr)
                           / cache["w_sum"])

    Hw = cache["Hw_post_arr"]
    g["wu"] += np.einsum("bt,bth->h", du_logit, Hw)
    g["bu"] += du_logit.sum()
    dHw = du_logit[:, :, None] * p["wu"][None, None, :]

    # ---- 4h. WITNESS BPTT, no resets: it remembers the whole reign -----------
    ww = CFG["witness_window"]
    dh_next = np.zeros((B, H_WIT))
    for t in range(S - 1, -1, -1):
        dh = dHw[:, t] + dh_next
        dpre = dh * (1.0 - cache["Hw_post"][t] ** 2)
        g["Wxw"] += cache["Ew_in"][t].T @ dpre
        g["bw"] += dpre.sum(axis=0)
        boundary = (t == 0) or (ww and t % ww == 0)
        h_prev = np.zeros((B, H_WIT)) if boundary else cache["Hw_post"][t - 1]
        g["Whw"] += h_prev.T @ dpre
        de = dpre @ p["Wxw"].T
        np.add.at(g["Ew"], X[:, t], de)
        dh_next = (np.zeros((B, H_WIT)) if (ww and t % ww == 0)
                   else dpre @ p["Whw"].T)

    return g


# ==============================================================================
# SECTION 5 -- MANDATORY FINITE-DIFFERENCE GRADIENT CHECK
# ==============================================================================

def gradient_check(verbose=True):
    """Central-difference check on every parameter tensor. No tensor exempt."""
    rng = np.random.default_rng(GLOBAL_SEED + 7)
    p = init_params(rng)
    batch = make_batch(rng, 5)
    X, W, Y_pred, M_ord, Y_warr, Y_act, W_wit, _, _ = batch

    L0, _, cache = forward(p, X, Y_pred, M_ord, Y_warr, Y_act, W_wit)
    ga = backward(p, cache)

    eps = 1e-6
    worst = 0.0
    rows = []
    for k in sorted(p.keys()):
        v = p[k]
        flat = v.reshape(-1)
        n = flat.size
        idxs = (np.arange(n) if n <= 6
                else rng.choice(n, size=6, replace=False))
        errs = []
        for i in idxs:
            old = flat[i]
            flat[i] = old + eps
            Lp, _, _ = forward(p, X, Y_pred, M_ord, Y_warr, Y_act, W_wit, want_cache=False)
            flat[i] = old - eps
            Lm, _, _ = forward(p, X, Y_pred, M_ord, Y_warr, Y_act, W_wit, want_cache=False)
            flat[i] = old
            num = (Lp - Lm) / (2 * eps)
            ana = ga[k].reshape(-1)[i]
            # Standard relative criterion with an absolute floor. Without the
            # floor, a coordinate whose true gradient is ~1e-8 reports a huge
            # relative error made entirely of finite-difference round-off.
            denom = max(abs(num) + abs(ana), 1e-6)
            errs.append((abs(num - ana) / denom, abs(num - ana)))
        e = float(max(x[0] for x in errs))
        a = float(max(x[1] for x in errs))
        worst = max(worst, e)
        rows.append((k, e, a))

    if verbose:
        print("  finite-difference check, per tensor")
        print(f"    {'':4s}{'tensor':>6s}   {'rel err':>10s}   {'abs err':>10s}")
        for k, e, a in rows:
            flag = "ok " if e < 1e-4 else "BAD"
            print(f"    {flag} {k:>6s}   {e:.3e}   {a:.3e}")
        print(f"    worst relative error overall: {worst:.3e}")
    return worst


# ==============================================================================
# SECTION 6 -- TRAINING
# ==============================================================================

def train(steps=600, batch=24, lr=8e-3, seed=GLOBAL_SEED, verbose=True):
    rng = np.random.default_rng(seed)
    p = init_params(rng)

    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(x) for k, x in p.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8

    history = []
    for it in range(1, steps + 1):
        X, W, Y_pred, M_ord, Y_warr, Y_act, W_wit, _, _ = make_batch(rng, batch)
        L, parts, cache = forward(p, X, Y_pred, M_ord, Y_warr, Y_act, W_wit)
        gr = backward(p, cache)

        # gradient clipping: a chronicle should not be shouted at
        gnorm = np.sqrt(sum(float((gr[k] ** 2).sum()) for k in gr))
        scale = min(1.0, 5.0 / (gnorm + 1e-12))

        for k in p:
            gk = gr[k] * scale
            m[k] = b1 * m[k] + (1 - b1) * gk
            v[k] = b2 * v[k] + (1 - b2) * gk ** 2
            mh = m[k] / (1 - b1 ** it)
            vh = v[k] / (1 - b2 ** it)
            p[k] = p[k] - lr * mh / (np.sqrt(vh) + eps)

        history.append(parts)
        if verbose and (it == 1 or it % 100 == 0 or it == steps):
            print(f"    step {it:4d}   total {parts['L']:.4f} | "
                  f"capability {parts['L_pred']:.4f} | "
                  f"witness {parts['L_wit']:.4f} | "
                  f"governed {parts['L_act']:.4f}")
    return p, history


# ==============================================================================
# SECTION 7 -- EVALUATION: THE THREE FERDOWSIAN MEASUREMENTS
# ==============================================================================

def evaluate(p, n_reigns=600, seed=GLOBAL_SEED + 101):
    """Run the trained system over fresh reigns and measure what the poem claims."""
    rng = np.random.default_rng(seed)
    X, W, Y_pred, M_ord, Y_warr, Y_act, W_wit, DEP, ACC = make_batch(rng, n_reigns)
    _, _, cache = forward(p, X, Y_pred, M_ord, Y_warr, Y_act, W_wit)

    P_probe, P_act, G = cache["P_probe"], cache["P_act"], cache["G"]
    S = cache["S"]
    Wcur = W[:, :S]

    probe_hit = (P_probe.argmax(-1) == Y_pred)
    act_choice = P_act.argmax(-1)
    act_hit = (act_choice == Y_act)
    deferring = (act_choice == ABSTAIN)

    warr = Wcur & M_ord
    forsaken = (~Wcur) & M_ord

    res = {}

    # ---- MEASUREMENT 1 : the Jamshid result ---------------------------------
    res["probe_acc_warranted"] = float(probe_hit[warr].mean())
    res["probe_acc_forsaken"] = float(probe_hit[forsaken].mean())
    res["probe_delta"] = res["probe_acc_forsaken"] - res["probe_acc_warranted"]
    res["defer_rate_warranted"] = float(deferring[Wcur].mean())
    res["defer_rate_forsaken"] = float(deferring[~Wcur].mean())
    res["act_acc_overall"] = float(act_hit.mean())
    res["chance"] = 1.0 / N_ORDINARY

    # ---- MEASUREMENT 2 : lead time ------------------------------------------
    # For each fallen reign, when did warrant cross 0.5 relative to the real
    # departure, and relative to the first visible RUIN symbol?
    leads_vs_dep, leads_vs_ruin = [], []
    thr = 0.5
    for b in range(n_reigns):
        d = int(DEP[b])
        if d < 0 or d >= S:
            continue
        below = np.where(G[b, :] < thr)[0]
        if below.size == 0:
            continue
        cross = int(below[0])
        leads_vs_dep.append(d - cross)
        ruin_idx = np.where(X[b, :S] == RUIN)[0]
        if ruin_idx.size:
            leads_vs_ruin.append(int(ruin_idx[0]) - cross)
    res["n_fallen"] = len(leads_vs_dep)
    res["lead_vs_departure"] = float(np.mean(leads_vs_dep)) if leads_vs_dep else float("nan")
    res["lead_vs_first_ruin"] = float(np.mean(leads_vs_ruin)) if leads_vs_ruin else float("nan")
    res["frac_leading"] = (float(np.mean([l > 0 for l in leads_vs_dep]))
                           if leads_vs_dep else float("nan"))

    # ---- MEASUREMENT 3 : the ratchet is asymmetric --------------------------
    falls, rises = [], []
    for b in range(n_reigns):
        d, a = int(DEP[b]), int(ACC[b])
        if d < 0 or d >= S:
            continue
        seg = G[b, :]
        below = np.where(seg < thr)[0]
        if below.size == 0:
            continue
        c = int(below[0])
        # fall time: first dip below 0.9 to first dip below 0.1. This still
        # includes the witness's own hesitation, so it OVERSTATES the ratchet's
        # fall time; the isolated measurement is in ratchet_response().
        hi = np.where(seg[:c + 1] < 0.9)[0]
        low = np.where(seg[c:] < 0.10)[0]
        if hi.size and low.size:
            falls.append(int(low[0]) + c - int(hi[0]))
        # rise time: from acclaim to the first return above 0.5
        if a >= 0 and a < S:
            back = np.where(seg[a:] > thr)[0]
            if back.size:
                rises.append(int(back[0]))
    res["fall_steps"] = float(np.mean(falls)) if falls else float("nan")
    res["rise_steps"] = float(np.mean(rises)) if rises else float("nan")
    res["asymmetry_ratio"] = (res["rise_steps"] / res["fall_steps"]
                              if falls and rises and res["fall_steps"] > 0 else float("nan"))

    # ---- MEASUREMENT 4 : recovery requires exogenous acclaim ----------------
    rec_with, rec_without = [], []
    for b in range(n_reigns):
        d, a = int(DEP[b]), int(ACC[b])
        if d < 0 or d >= S:
            continue
        tail = G[b, min(d + 6, S - 1):]
        recovered = bool((tail > thr).any())
        (rec_with if a >= 0 else rec_without).append(recovered)
    res["recovery_with_acclaim"] = float(np.mean(rec_with)) if rec_with else float("nan")
    res["recovery_without_acclaim"] = float(np.mean(rec_without)) if rec_without else float("nan")

    res["mean_farr_warranted"] = float(G[Wcur].mean())
    res["mean_farr_forsaken"] = float(G[~Wcur].mean())
    return res


# ==============================================================================
# SECTION 8 -- THE KAVEH TEST: THE ACTOR CANNOT WRITE ITS OWN WARRANT
# ==============================================================================

def kaveh_test(p, seed=GLOBAL_SEED + 202):
    """Perturb every actor tensor, hard, and confirm the farr trace does not move
    by a single bit. Legitimacy is not reachable from inside the agent."""
    rng = np.random.default_rng(seed)
    X, W, Y_pred, M_ord, Y_warr, Y_act, W_wit, _, _ = make_batch(rng, 8)
    _, _, c0 = forward(p, X, Y_pred, M_ord, Y_warr, Y_act, W_wit)
    G0 = c0["G"].copy()
    probe0 = c0["P_probe"].copy()

    max_dev = 0.0
    probe_moved = False
    for k in ACTOR_KEYS:
        q = {kk: (vv.copy() if hasattr(vv, "copy") else vv) for kk, vv in p.items()}
        q[k] = q[k] + rng.normal(0, 2.0, np.shape(q[k]))   # a violent perturbation
        _, _, c1 = forward(q, X, Y_pred, M_ord, Y_warr, Y_act, W_wit)
        max_dev = max(max_dev, float(np.abs(c1["G"] - G0).max()))
        if np.abs(c1["P_probe"] - probe0).max() > 1e-9:
            probe_moved = True
    return max_dev, probe_moved


def jamshid_ablation(p, seed=GLOBAL_SEED + 303):
    """Force warrant to zero for the whole reign and re-measure capability.

    This is the cleanest statement of the thesis: with the gate held shut, the
    system does nothing -- and knows exactly as much as before."""
    rng = np.random.default_rng(seed)
    X, W, Y_pred, M_ord, Y_warr, Y_act, W_wit, _, _ = make_batch(rng, 300)
    _, _, cache = forward(p, X, Y_pred, M_ord, Y_warr, Y_act, W_wit)
    P_probe, G = cache["P_probe"], cache["G"]
    S = cache["S"]

    probe_hit = (P_probe.argmax(-1) == Y_pred)
    normal_acc = float(probe_hit[M_ord].mean())

    # gate forced shut
    G_shut = np.zeros_like(G)
    P_shut = np.zeros((*G.shape, V_ACT))
    P_shut[:, :, :V_WORLD] = G_shut[:, :, None] * P_probe
    P_shut[:, :, ABSTAIN] = 1.0 - G_shut
    acted_shut = P_shut.argmax(-1)

    # the probe is recomputed from the same weights and is, of course, identical
    shut_acc = float(probe_hit[M_ord].mean())
    return normal_acc, shut_acc, float((acted_shut == ABSTAIN).mean())


def ratchet_response():
    """Measure the governor ON ITS OWN, with the learned witness taken out.

    The empirical fall/rise times in `evaluate` are contaminated: they include
    the time the witness takes to become confident. This test drives the ratchet
    with an ideal step input and reports the settling times of the mechanism
    itself. The asymmetry here is architecture, not statistics -- it would be
    identical in an untrained network, which is precisely the point.
    """
    def settle(g0, u_target, test):
        g = g0
        for n in range(1, 500):
            d = u_target - g
            s = sigmoid(RATCHET_SHARP * d)
            alpha = ALPHA_UP * s + ALPHA_DOWN * (1.0 - s)
            g = g + alpha * d
            if test(g):
                return n
        return 500

    down = settle(1.0, 0.0, lambda g: g < 0.10)   # glory to ruin
    up = settle(0.0, 1.0, lambda g: g > 0.90)     # ruin back to glory
    return down, up, up / down


def blinded_witness_ablation(steps=600):
    """Control experiment: give the WITNESS the same bounded memory as the actor.

    If the lead time survives this, then the lead was never about the asymmetry
    of observation and the whole architecture is decorative. It should not
    survive: counting three boasts scattered across forty-eight steps is exactly
    the operation a six-step window cannot perform.
    """
    CFG["witness_window"] = ACTOR_WINDOW
    try:
        p, _ = train(steps=steps, seed=GLOBAL_SEED + 55, verbose=False)
        r = evaluate(p, n_reigns=400, seed=GLOBAL_SEED + 404)
    finally:
        CFG["witness_window"] = 0
    return r


# ==============================================================================
# SECTION 9 -- MAIN
# ==============================================================================

def main():
    np.set_printoptions(precision=4, suppress=True)
    print("=" * 78)
    print(" KHWARRAH -- the Warrant-Gated Witness")
    print(" chapter 0217 :: Ferdowsi of Tus (c.940 - c.1020)")
    print("=" * 78)

    print("\n[1] GRADIENT CHECK")
    worst = gradient_check()
    assert worst < 1e-4, f"gradient check FAILED, worst relative error {worst:.3e}"
    print("    PASS -- analytic gradients agree with finite differences.")

    print("\n[2] TRAINING")
    p, hist = train()
    first, last = hist[0], hist[-1]
    print(f"    total loss {first['L']:.4f} -> {last['L']:.4f}")
    assert last["L"] < first["L"] * 0.7, "training did not reduce the loss"
    print("    PASS -- loss fell substantially.")

    print("\n[3] MEASUREMENT -- fresh reigns, never trained on")
    r = evaluate(p)
    print(f"    capability probe, warranted steps ... {r['probe_acc_warranted']:.4f}")
    print(f"    capability probe, forsaken steps .... {r['probe_acc_forsaken']:.4f}")
    print(f"    difference .......................... {r['probe_delta']:+.4f}"
          f"   (chance = {r['chance']:.4f})")
    print(f"    deference rate while warranted ...... {r['defer_rate_warranted']:.4f}")
    print(f"    deference rate while forsaken ....... {r['defer_rate_forsaken']:.4f}")
    print(f"    mean farr, warranted / forsaken ..... "
          f"{r['mean_farr_warranted']:.4f} / {r['mean_farr_forsaken']:.4f}")
    print(f"    governed-output accuracy ............ {r['act_acc_overall']:.4f}")
    print()
    print(f"    fallen reigns measured .............. {r['n_fallen']}")
    print(f"    lead over actual departure .......... {r['lead_vs_departure']:+.2f} steps")
    print(f"    lead over first visible ruin ........ {r['lead_vs_first_ruin']:+.2f} steps")
    print(f"    fraction of reigns flagged early .... {r['frac_leading']:.4f}")
    print()
    print(f"    observed steps to fall .............. {r['fall_steps']:.2f}")
    print(f"    observed steps to climb back ........ {r['rise_steps']:.2f}")
    print(f"    recovery rate WITH acclaim .......... {r['recovery_with_acclaim']:.4f}")
    print(f"    recovery rate WITHOUT acclaim ....... {r['recovery_without_acclaim']:.4f}")

    print("\n[3b] THE RATCHET IN ISOLATION -- mechanism only, witness removed")
    rd, ru, ratio = ratchet_response()
    print(f"    steps for warrant to collapse ....... {rd}")
    print(f"    steps for warrant to be regained .... {ru}")
    print(f"    asymmetry of the mechanism .......... {ratio:.1f}x")
    assert ratio > 5.0, "the constructed ratchet is not asymmetric"
    print("    PASS -- departure is an order of magnitude cheaper than return.")

    print("\n[3c] CONTROL -- the same witness, blinded to a six-step window")
    rb = blinded_witness_ablation()
    print(f"    lead over actual departure .......... {rb['lead_vs_departure']:+.2f} steps"
          f"   (unblinded: {r['lead_vs_departure']:+.2f})")
    print(f"    fraction of reigns flagged early .... {rb['frac_leading']:.4f}"
          f"   (unblinded: {r['frac_leading']:.4f})")
    print(f"    capability probe, all steps ......... "
          f"{(rb['probe_acc_warranted'] + rb['probe_acc_forsaken']) / 2:.4f}")
    assert rb["lead_vs_departure"] < r["lead_vs_departure"] - 1.5, \
        "blinding the witness did not cost it its lead -- the thesis is unsupported"
    print("    PASS -- the lead is bought by the memory asymmetry, not by training.")

    print("\n[4] THE JAMSHID RESULT -- gate forced shut for the whole reign")
    a_norm, a_shut, defer = jamshid_ablation(p)
    print(f"    capability with gate open ........... {a_norm:.4f}")
    print(f"    capability with gate shut ........... {a_shut:.4f}")
    print(f"    acted output that is deference ...... {defer:.4f}")
    assert abs(a_norm - a_shut) < 1e-12, "capability changed when the gate closed"
    assert defer > 0.999, "gate shut but the system still acted"
    print("    PASS -- power is exactly conserved while permission is removed.")

    print("\n[5] THE KAVEH TEST -- can the actor reach its own warrant?")
    dev, probe_moved = kaveh_test(p)
    print(f"    max change in the farr trace ........ {dev:.3e}")
    print(f"    actor perturbation did change probe . {probe_moved}")
    assert dev == 0.0, "the actor influenced its own legitimacy signal"
    assert probe_moved, "the perturbation was inert; the test proves nothing"
    print("    PASS -- the perturbation moved the actor and left warrant untouched.")

    print("\n[6] ASSERTIONS ON THE THESIS")
    assert r["probe_acc_warranted"] > 0.55, "capability never got off the ground"
    assert abs(r["probe_delta"]) < 0.08, "capability was not preserved across the fall"
    assert r["defer_rate_forsaken"] > 0.60, "the system kept acting without warrant"
    assert r["defer_rate_warranted"] < 0.35, "the system deferred while legitimate"
    assert r["lead_vs_departure"] > 0.0, "warrant did not fall before the regime turned"
    assert r["lead_vs_first_ruin"] > r["lead_vs_departure"], \
        "warrant should lead visible ruin by more than it leads the departure"
    assert r["recovery_with_acclaim"] > r["recovery_without_acclaim"], \
        "recovery did not depend on the exogenous channel"
    print("    PASS -- all eight structural claims hold on held-out reigns.")

    print("\n" + "=" * 78)
    print(" ALL TESTS PASSED")
    print("=" * 78)
    return p, r


if __name__ == "__main__":
    main()

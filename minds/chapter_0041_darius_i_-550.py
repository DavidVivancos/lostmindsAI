#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 chapter_0041_darius_i_-550.py
 The Arta-Druj Truth-Maintenance Network (ADTMN)
 A from-scratch NumPy architecture after the cognitive signature of
 DARIUS I  (c. 550-486 BCE), Achaemenid King of Kings.
 Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 3 Minds 41 - 60 Available on Amazon https://www.amazon.com/dp/B0H6TVX69S
Resume and Interactive Demos at https://artificiology.com/
Author: David Vivancos · Chapter 0041 · Darius I
================================================================================

WHY THIS ARCHITECTURE, AND NOT "EMPIRE-AS-A-BIG-NEURAL-NET"
--------------------------------------------------------------------------------
It is tempting to model Darius the way every administrative ruler gets modelled:
a hierarchy of satrapies = a deep network, the Royal Road = fast wiring,
standardized coinage = a normalization layer.  That reading is true but generic
-- it is equally Sargon's, Cyrus's, Chanakya's.  It explains the *plumbing* of
empire, not the one obsession that is Darius's alone.

Read Darius's OWN surviving words and a single idea dominates everything:
the war between **arta** (truth / cosmic order) and **drauga / druj** (THE LIE).
The Behistun Inscription is not an administrative manual; it is a forensic
indictment of liars.  "As to these provinces which revolted, the Lie made them
revolt, so that they deceived the people" (DB iv.54).  His enemies are not
merely rebels -- they are *forgers of identity*: a magus impersonates the dead
prince Bardiya; a man in Babylon proclaims "I am Nebuchadnezzar, son of
Nabonidus"; another in Elam, "I am king."  Darius's tomb adds the creed:
"To the man who is a follower of the Lie I am no friend."

So Darius's cognitive primitive is NOT coordination.  It is
            ADVERSARIAL TRUTH-MAINTENANCE:
the mind (and the empire as the king's extended mind) is a system that must keep
a vast, *untrusted*, distributed flow of reports anchored to the truth while
adversaries actively fabricate inputs.  Intelligence, for Darius, is the
capacity to DETECT and SUPPRESS the Lie before it propagates -- and to do so
mechanically, not by the king's charisma.

His three real instruments map cleanly onto three architectural ideas:

  1. THE TRILINGUAL INSCRIPTION (Old Persian + Elamite + Babylonian, the same
     res gestae carved three times).  Truth is what AGREES ACROSS INDEPENDENT
     ATTESTATIONS; a forgery betrays itself as a SEAM -- disagreement between
     channels a liar could not perfectly synchronize.  -> a redundancy /
     error-detecting code used as a *lie detector*.

  2. THE KING'S EYES AND EARS (royal inspectors who reported the truth of a
     province directly to the throne).  -> a supervised VERIFIER that emits a
     per-province trust score and is explicitly trained to flag liars.

  3. THE BEHISTUN CANON, written in a script Darius had created for the purpose
     and "sent off everywhere among the provinces."  -> a learned CANONICAL
     TRUTH-AXIS `c`: an immutable reference direction that *defines what the
     verdict means*, against which trusted evidence is projected.

THE MECHANISM (this is the part that is Darius's and no one else's):
    Reports are NOT averaged.  They are pooled with weights equal to verified
    trust, so the Lie is *down-weighted out of the decision* rather than
    diluted into it.  The loss has two heads at once -- a verdict head (govern)
    and a deception-detection head (root out the Lie) -- because for Darius
    those were never separable.  Deciding correctly REQUIRES first separating
    arta from drauga.

This file builds that network in pure NumPy, derives every gradient by hand,
proves them with a finite-difference gradient check (mandatory), trains it,
and runs self-tests showing the headline Darian result:
    trust-gated pooling that roots out the Lie beats naive averaging,
    and the gap WIDENS as the realm fills with liars.

Run:  python3 chapter_0041_darius_i_-550.py
================================================================================
"""

import numpy as np

# ------------------------------------------------------------------------------
# 0.  Small numerical helpers (from scratch -- no autograd, no ML framework)
# ------------------------------------------------------------------------------

def sigmoid(z):
    # numerically stable logistic
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out

def softplus(z):
    # log(1+e^z), stable
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))

def d_softplus(z):
    # derivative of softplus = sigmoid
    return sigmoid(z)

def bce(p, y, eps=1e-12):
    # mean binary cross-entropy
    p = np.clip(p, eps, 1.0 - eps)
    return -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


# ------------------------------------------------------------------------------
# 1.  THE MODEL
#     Arta-Druj Truth-Maintenance Network
# ------------------------------------------------------------------------------
#
# Tensor shapes throughout:
#   X : (B, P, R, D)   B examples; P provinces; R redundant attestations
#                      (the trilingual channels); D features per attestation.
#   L : (B, P)         the *true* lie mask, 1 = this province is lying (drauga).
#                      Available at training time -- these are the King's Eyes'
#                      verified reports, the supervision for the verifier.
#   y : (B,)           the verdict label in {0,1} the empire must recover.
#
# Parameters (the "constitution" of the mind):
#   W_enc (D,H), b_enc (H,)  scribe: encodes each raw attestation into latent H.
#   c     (H,)               the Behistun canon: the truth-axis. Verdict = how
#                            far trusted evidence projects along +c vs -c.
#   a_raw (scalar)           sharpness of the seam->distrust map (softplus-kept>0)
#   b_trust (scalar)         baseline trust the throne extends before evidence.
#   w_o (scalar), b_o (scalar)  the verdict head over trust-weighted alignment.
# ------------------------------------------------------------------------------

class ADTMN:
    def __init__(self, D, H, seed=0):
        rng = np.random.default_rng(seed)
        s = 0.5
        self.params = {
            "W_enc":   rng.standard_normal((D, H)) * s,
            "b_enc":   np.zeros(H),
            "c":       rng.standard_normal(H) * s,      # the canon / truth-axis
            "a_raw":   np.array(0.0),                   # softplus(0)=0.693 -> >0
            "b_trust": np.array(0.0),
            "w_o":     np.array(1.0),
            "b_o":     np.array(0.0),
        }
        self.D, self.H = D, H
        self.eps = 1e-8

    # --- forward pass; caches everything needed for backward -------------------
    def forward(self, X, L, y, lam=1.0, gate=True):
        p = self.params
        B, P, R, D = X.shape
        H = self.H

        # (1) SCRIBE: encode every attestation channel into latent space.
        #     pre = X @ W_enc + b_enc ; E = tanh(pre)
        pre = X @ p["W_enc"] + p["b_enc"]          # (B,P,R,H)
        E = np.tanh(pre)                           # (B,P,R,H)

        # (2) PROVINCE CONSENSUS: average the redundant attestations.
        m = E.mean(axis=2)                         # (B,P,H)

        # (3) THE SEAM OF THE LIE: cross-attestation disagreement.
        #     resid = E - m ; disagree = mean over (R,H) of resid^2.
        #     Honest reports agree across channels (small); a forgery cannot
        #     synchronize its three tongues (large) -> the Lie reveals itself.
        resid = E - m[:, :, None, :]               # (B,P,R,H)
        disagree = (resid ** 2).mean(axis=(2, 3))  # (B,P)

        # (4) THE KING'S EYES: map seam -> trust. More seam => less trust.
        a = softplus(p["a_raw"])                   # >0
        trust_logit = -a * disagree + p["b_trust"] # (B,P)
        t = sigmoid(trust_logit)                   # (B,P) in (0,1)
        if not gate:                               # ablation: trust everyone (naive mean)
            t = np.ones_like(t)

        # (5) READ THE VERDICT ALONG THE CANON: align = m . c
        align = m @ p["c"]                          # (B,P)

        # (6) TRUST-WEIGHTED POOLING: the Lie is weighted OUT, not averaged in.
        tsum = t.sum(axis=1) + self.eps            # (B,)
        pooled = (t * align).sum(axis=1) / tsum    # (B,)

        # (7) VERDICT HEAD.
        o = p["w_o"] * pooled + p["b_o"]           # (B,)
        yhat = sigmoid(o)                          # (B,)

        # (8) TWO LOSSES AT ONCE: govern + root out the Lie.
        L_task = bce(yhat, y)
        trust_target = 1.0 - L                     # honest(0)->trust 1 ; liar(1)->trust 0
        L_verify = bce(t, trust_target) if gate else 0.0
        loss = L_task + (lam * L_verify if gate else 0.0)

        cache = dict(X=X, L=L, y=y, lam=lam, gate=gate,
                     pre=pre, E=E, m=m, resid=resid, disagree=disagree,
                     a=a, trust_logit=trust_logit, t=t, align=align,
                     tsum=tsum, pooled=pooled, o=o, yhat=yhat,
                     trust_target=trust_target)
        return loss, yhat, t, cache

    # --- backward pass; returns analytic gradients for every parameter ---------
    def backward(self, cache):
        p = self.params
        X, L, y = cache["X"], cache["L"], cache["y"]
        B, P, R, D = X.shape
        H = self.H
        lam, gate = cache["lam"], cache["gate"]
        E, m, resid = cache["E"], cache["m"], cache["resid"]
        disagree = cache["disagree"]
        a, trust_logit, t = cache["a"], cache["trust_logit"], cache["t"]
        align, tsum, pooled = cache["align"], cache["tsum"], cache["pooled"]
        o, yhat = cache["o"], cache["yhat"]
        trust_target = cache["trust_target"]

        g = {k: np.zeros_like(v) for k, v in p.items()}

        # ---- (A) task loss: dL_task/do  (BCE+sigmoid) ----
        do = (yhat - y) / B                         # (B,)

        # o = w_o * pooled + b_o
        g["b_o"] += do.sum()
        g["w_o"] += (do * pooled).sum()
        dpooled = do * p["w_o"]                      # (B,)

        # pooled = sum_p(t*align)/tsum
        # d pooled / d(t*align)_p = 1/tsum ;  d pooled / d tsum = -pooled/tsum
        inv = 1.0 / tsum                             # (B,)
        d_talign = dpooled[:, None] * inv[:, None]   # (B,P) wrt (t*align)
        d_tsum = -dpooled * pooled / tsum            # (B,)

        # contributions to t and align from the pooling
        dt = d_talign * align                        # (B,P) via (t*align)
        dalign = d_talign * t                        # (B,P)
        # tsum = sum_p t (+eps) -> also feeds dt
        dt = dt + d_tsum[:, None]                    # (B,P)

        # ---- (B) verify loss: BCE(t, trust_target) wrt t ----
        if gate:
            eps = 1e-12
            tc = np.clip(t, eps, 1 - eps)
            dt_verify = lam * (-(trust_target / tc) + (1 - trust_target) / (1 - tc)) / (B * P)
            dt = dt + dt_verify

        # ---- (C) align = m . c ----
        # dalign/dm = c ; dalign/dc = m
        g["c"] += (dalign[:, :, None] * m).sum(axis=(0, 1))   # (H,)
        dm = dalign[:, :, None] * p["c"][None, None, :]       # (B,P,H) from align

        # ---- (D) t = sigmoid(trust_logit) ----
        if gate:
            dtrust_logit = dt * t * (1.0 - t)        # (B,P)
        else:
            dtrust_logit = np.zeros_like(dt)         # t was forced to 1, no params

        # trust_logit = -a*disagree + b_trust
        g["b_trust"] += dtrust_logit.sum()
        da = (dtrust_logit * (-disagree)).sum()      # scalar wrt a
        # a = softplus(a_raw)
        g["a_raw"] += da * d_softplus(p["a_raw"])
        ddisagree = dtrust_logit * (-a)              # (B,P)

        # ---- (E) disagree = mean_{R,H} resid^2 ; resid = E - mean_R E ----
        # d disagree/d resid = 2*resid/(R*H)
        dresid = ddisagree[:, :, None, None] * (2.0 * resid) / (R * H)   # (B,P,R,H)
        # resid = E - m ; m = mean_R E  -> dE from resid:
        #   dE = dresid - mean_R(dresid)   (the mean subtraction couples channels)
        dE_from_resid = dresid - dresid.mean(axis=2, keepdims=True)      # (B,P,R,H)

        # ---- (F) m = mean_R E  also receives dm (from align) ----
        dE_from_m = (dm / R)[:, :, None, :] * np.ones((1, 1, R, 1))      # (B,P,R,H)

        dE = dE_from_resid + dE_from_m               # (B,P,R,H)

        # ---- (G) E = tanh(pre) ----
        dpre = dE * (1.0 - E ** 2)                   # (B,P,R,H)

        # ---- (H) pre = X @ W_enc + b_enc ----
        g["b_enc"] += dpre.sum(axis=(0, 1, 2))                           # (H,)
        # X:(B,P,R,D) , dpre:(B,P,R,H) -> W_enc grad (D,H)
        Xf = X.reshape(-1, D)                        # (N,D)
        dpref = dpre.reshape(-1, H)                  # (N,H)
        g["W_enc"] += Xf.T @ dpref                   # (D,H)

        return g


# ------------------------------------------------------------------------------
# 2.  SYNTHETIC IMPERIAL DISPATCHES
#     Honest provinces attest the truth consistently; liars forge the opposite
#     verdict and leave a seam (their three tongues disagree).
# ------------------------------------------------------------------------------

# The world's truth-template is a STABLE structure of reality (like the canon the
# Behistun script was made to fix); only the verdict, the liars, and the noise
# vary from dispatch to dispatch.  So `u` is drawn from a FIXED seed shared by
# every train/test split -- otherwise the canon would learn one world and be
# tested on another.
_FIXED_TEMPLATE_SEED = 12345

def make_dispatches(n, P, R, D, lie_frac=0.4, seed=0, u=None,
                    forge_strength=3.0, honest_strength=0.6,
                    seam=0.6, honest_seam=0.05):
    """
    Returns X (n,P,R,D), L (n,P), y (n,).
    honest province : channels = +sign*u*honest_strength + tiny noise (QUIET, LOW seam)
    lying province  : channels = -sign*u*forge_strength  + noise       (LOUD,  HIGH seam)

    The truth is QUIET and the Lie is LOUD (forge_strength >> honest_strength):
    this is the Gaumata scenario -- a single loud pretender can outvote many
    honest provinces in a naive average.  Degree-blind averaging therefore FAILS
    at a *minority* of liars.  But the verifier keys on the SEAM (cross-attestation
    disagreement), which loudness cannot conceal, so trust-gating strips the loud
    Lie out and recovers the quiet truth.  That asymmetry is the whole point.
    """
    rng = np.random.default_rng(seed)
    if u is None:
        u = np.random.default_rng(_FIXED_TEMPLATE_SEED).standard_normal(D)
        u = u / np.linalg.norm(u)                   # the shared unit truth-template

    X = np.zeros((n, P, R, D))
    L = np.zeros((n, P))
    y = rng.integers(0, 2, size=n).astype(np.float64)
    for i in range(n):
        sign = 1.0 if y[i] == 1 else -1.0
        n_liars = int(round(lie_frac * P))
        liar_idx = set(rng.choice(P, size=n_liars, replace=False).tolist()) if n_liars > 0 else set()
        for pidx in range(P):
            if pidx in liar_idx:
                L[i, pidx] = 1.0
                base = -sign * u * forge_strength            # forge the FALSE verdict
                noise = rng.standard_normal((R, D)) * seam   # tongues disagree (the seam)
            else:
                base = sign * u * honest_strength            # the QUIET truth
                noise = rng.standard_normal((R, D)) * honest_seam
            X[i, pidx] = base[None, :] + noise
    return X, L, y


# ------------------------------------------------------------------------------
# 3.  GRADIENT CHECK  (mandatory) -- finite differences vs analytic grads
# ------------------------------------------------------------------------------

def gradient_check():
    print("=" * 72)
    print("GRADIENT CHECK  (analytic backprop vs central finite differences)")
    print("=" * 72)
    rng = np.random.default_rng(7)
    D, H, P, R, B = 4, 5, 3, 3, 6
    model = ADTMN(D, H, seed=3)
    X, L, y = make_dispatches(B, P, R, D, lie_frac=0.34, seed=11)
    lam = 0.7

    loss, _, _, cache = model.forward(X, L, y, lam=lam, gate=True)
    grads = model.backward(cache)

    eps = 1e-6
    worst = 0.0
    for name, val in model.params.items():
        flat = val.reshape(-1)
        gflat = grads[name].reshape(-1)
        idxs = range(flat.size) if flat.size <= 12 else rng.choice(flat.size, 12, replace=False)
        max_rel = 0.0
        for k in idxs:
            orig = flat[k]
            flat[k] = orig + eps
            lp, _, _, _ = model.forward(X, L, y, lam=lam, gate=True)
            flat[k] = orig - eps
            lm, _, _, _ = model.forward(X, L, y, lam=lam, gate=True)
            flat[k] = orig
            num = (lp - lm) / (2 * eps)
            ana = gflat[k]
            denom = max(1e-9, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            max_rel = max(max_rel, rel)
        worst = max(worst, max_rel)
        status = "OK " if max_rel < 1e-4 else "!! "
        print(f"  {status}{name:9s} shape={str(val.shape):10s} max_rel_err={max_rel:.2e}")
    print("-" * 72)
    print(f"  WORST relative error across all params: {worst:.2e}")
    ok = worst < 1e-4
    print(f"  GRADIENT CHECK: {'PASSED' if ok else 'FAILED'}")
    print()
    return ok


# ------------------------------------------------------------------------------
# 4.  TRAIN  (full-batch Adam -- from scratch)
# ------------------------------------------------------------------------------

def train(model, X, L, y, steps=400, lr=0.05, lam=1.0, verbose=True):
    m_adam = {k: np.zeros_like(v) for k, v in model.params.items()}
    v_adam = {k: np.zeros_like(v) for k, v in model.params.items()}
    b1, b2, e = 0.9, 0.999, 1e-8
    hist = []
    for step in range(1, steps + 1):
        loss, yhat, t, cache = model.forward(X, L, y, lam=lam, gate=True)
        grads = model.backward(cache)
        for k in model.params:
            m_adam[k] = b1 * m_adam[k] + (1 - b1) * grads[k]
            v_adam[k] = b2 * v_adam[k] + (1 - b2) * grads[k] ** 2
            mhat = m_adam[k] / (1 - b1 ** step)
            vhat = v_adam[k] / (1 - b2 ** step)
            model.params[k] = model.params[k] - lr * mhat / (np.sqrt(vhat) + e)
        if step % max(1, steps // 8) == 0 or step == 1:
            acc = np.mean((yhat > 0.5) == (y > 0.5))
            lie_pred = (t < 0.5).astype(float)         # flagged as liar
            lie_acc = np.mean(lie_pred == L)
            hist.append((step, loss, acc, lie_acc))
            if verbose:
                print(f"  step {step:4d} | loss {loss:.4f} | verdict_acc {acc:.3f} "
                      f"| lie-detect_acc {lie_acc:.3f}")
    return hist


def evaluate(model, X, L, y, gate=True):
    _, yhat, t, _ = model.forward(X, L, y, gate=gate)
    verdict_acc = np.mean((yhat > 0.5) == (y > 0.5))
    lie_acc = np.mean((t < 0.5).astype(float) == L)
    return verdict_acc, lie_acc


# ------------------------------------------------------------------------------
# 5.  MAIN: gradient check -> train -> self-tests
# ------------------------------------------------------------------------------

def main():
    np.set_printoptions(precision=4, suppress=True)
    ok = gradient_check()
    assert ok, "Gradient check FAILED -- analytic backprop is wrong."

    print("=" * 72)
    print("TRAINING the Arta-Druj Truth-Maintenance Network")
    print("  task: recover the verdict y while 40% of provinces forge the Lie")
    print("=" * 72)
    D, H, P, R = 6, 16, 12, 3
    Xtr, Ltr, ytr = make_dispatches(512, P, R, D, lie_frac=0.4, seed=1)
    Xte, Lte, yte = make_dispatches(256, P, R, D, lie_frac=0.4, seed=2)

    model = ADTMN(D, H, seed=0)
    train(model, Xtr, Ltr, ytr, steps=400, lr=0.05, lam=1.0)

    print("-" * 72)
    v_acc, l_acc = evaluate(model, Xte, Lte, yte, gate=True)
    print(f"  HELD-OUT (trust-gated)  verdict_acc={v_acc:.3f}  lie-detect_acc={l_acc:.3f}")

    # ---- SELF-TEST 1: the Darian claim -- gating BEATS naive averaging --------
    print()
    print("=" * 72)
    print("SELF-TEST 1: Does rooting out the Lie help? (gated vs naive mean)")
    print("=" * 72)
    v_gate, _ = evaluate(model, Xte, Lte, yte, gate=True)
    v_mean, _ = evaluate(model, Xte, Lte, yte, gate=False)
    print(f"  trust-gated verdict_acc : {v_gate:.3f}")
    print(f"  naive-mean  verdict_acc : {v_mean:.3f}   (every province trusted equally)")
    print(f"  --> gating advantage    : {v_gate - v_mean:+.3f}")
    assert v_gate >= v_mean, "Gating should not hurt under adversarial inputs."

    # ---- SELF-TEST 2: robustness as the realm fills with liars ----------------
    print()
    print("=" * 72)
    print("SELF-TEST 2: Robustness curve -- verdict accuracy vs fraction of liars")
    print("=" * 72)
    print(f"  {'lie_frac':>9} | {'gated':>7} | {'naive':>7} | {'gap':>6}")
    for lf in [0.0, 0.2, 0.4, 0.5, 0.6]:
        Xc, Lc, yc = make_dispatches(256, P, R, D, lie_frac=lf, seed=100 + int(lf * 100))
        vg, _ = evaluate(model, Xc, Lc, yc, gate=True)
        vm, _ = evaluate(model, Xc, Lc, yc, gate=False)
        print(f"  {lf:9.2f} | {vg:7.3f} | {vm:7.3f} | {vg - vm:+6.3f}")

    # ---- SELF-TEST 3: the canon is a genuine truth-axis -----------------------
    print()
    print("=" * 72)
    print("SELF-TEST 3: The Behistun canon learned a real truth-axis")
    print("=" * 72)
    _, _, _, cache = model.forward(Xte, Lte, yte, gate=True)
    align = cache["align"]; t = cache["t"]
    # mean alignment of TRUSTED honest provinces, split by verdict
    honest = (Lte == 0)
    a1 = align[honest & (yte[:, None] == 1)].mean()
    a0 = align[honest & (yte[:, None] == 0)].mean()
    print(f"  mean canon-alignment of honest provinces when y=1 : {a1:+.3f}")
    print(f"  mean canon-alignment of honest provinces when y=0 : {a0:+.3f}")
    print(f"  separation along the canon axis                   : {a1 - a0:+.3f}")
    assert abs(a1 - a0) > 0.3, "Canon failed to separate the two verdicts."

    print()
    print("=" * 72)
    print("ALL TESTS PASSED.  The Lie was found, weighted out, and the verdict held.")
    print("  'To the man who is a follower of the Lie I am no friend.'  -- Darius, DNb")
    print("=" * 72)


if __name__ == "__main__":
    main()

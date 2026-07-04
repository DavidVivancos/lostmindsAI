#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================================
# chapter_0019_shalmaneser_iii_-858.py  —  THE ANNALIST: a Monotone Self-Narrative Network (MSNN)
# Mind #19 — Shalmaneser III of Assyria (r. 858–824 BCE)
# Part Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 1 Minds 1 - 20 Available on Amazon https://www.amazon.com/dp/B0H6F9L324
# Resume and Interactive Demos at https://artificiology.com/
# ======================================================================================
#
# WHAT THIS FILE IS
# -----------------
# A from-scratch, pure-NumPy neural architecture that encodes the *specific* cognitive
# signature of Shalmaneser III — not the generic "empire = record-keeping" lens, but the
# sharper, well-attested fact about HIS mind in particular:
#
#     Shalmaneser left more recensions of his royal annals than any other Assyrian king.
#     The annals were not an append-only ledger; each new edition RE-NARRATED the whole
#     reign-to-date, quietly revising earlier campaigns so that one invariant always held:
#     the king's glory only ever ASCENDS and the empire only ever GROWS. Stalled campaigns
#     (Qarqar, 853 BCE; the unbreakable Damascus coalition) became victories; the deeds of
#     his commander-in-chief Dayyan-Ashur, who led the late campaigns, were absorbed into
#     the royal first person ("I marched..."). Truth was subordinate to MONOTONICITY.
#
#     The failure mode was built into the design. When reality finally diverged too far to
#     be re-edited — the civil war of his own sons Ashur-danin-pal and Shamshi-Adad at the
#     reign's end (c. 826–820 BCE) — the monotone story FRACTURED, and that is exactly where
#     the annals fall silent.
#
# So the architecture is a RECURRENT BELIEF-REVISION machine, NOT a Transformer/attention
# model. Its three signature mechanisms:
#
#   (1) THE SCRIBE (re-edit operator).  After a causal pass over campaign-events, a backward
#       "recension" pass rewrites every past episode toward the latest, most authoritative
#       narrative summary. Later editions overwrite earlier ones — the annals were re-edited.
#
#   (2) THE GLORY INVARIANT.  A scalar read-out g_t (the "official glory line" carved on the
#       obelisk) is forced to be MONOTONE NON-DECREASING by a soft penalty. The network thus
#       learns to tell a strictly ascending story even over a reality that dips and stalls.
#
#   (3) THE FRACTURE HEAD.  A scalar p_t learns to flag the irreconcilable setback — the event
#       the monotone story cannot absorb. This is the civil war written into the loss: the mind
#       that models its own breaking point.
#
# A next-event prediction head keeps the network honestly modelling the data (so it is a real
# learner, not just a constraint-satisfier). The tension between prediction (truth) and the
# glory invariant (the story) is the entire point.
#
# ENGINEERING CONTRACT (kept identical across the 1000Minds corpus):
#   * pure NumPy, manual forward + manual backprop (no autograd);
#   * a finite-difference gradient check that MUST pass (printed at run time);
#   * a real training loop on a synthetic campaign-sequence generator;
#   * self-tests with hard asserts;
#   * the file is executed before shipping and the verified output pasted into the chapter.
#
# Run:  python3 chapter_0019_shalmaneser_iii_-858.py
# Author: David Vivancos · Chapter 0019 · Shalmaneser III
# ======================================================================================

import numpy as np

RNG = np.random.default_rng(19)  # 19 = Shalmaneser's chapter number; reproducibility


# ======================================================================================
# 0. SMALL NUMERIC HELPERS
# ======================================================================================
def sigmoid(z):
    # Numerically stable logistic.
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def relu(z):
    return np.maximum(z, 0.0)


# ======================================================================================
# 1. PARAMETERS
# ----------------------------------------------------------------------------------
# Din : dimensionality of a campaign-event vector. Component 0 is the "success magnitude"
#       (positive = a victory / tribute received; negative = a setback). The remaining
#       components are contextual signal (which frontier, season, coalition size, noise).
# Dh  : width of the hidden "chronicle" state (one vector per episode/eponym-year).
# ======================================================================================
def init_params(Din, Dh, seed=0):
    r = np.random.default_rng(seed)

    def he(shape, fan_in):
        return (r.standard_normal(shape) * np.sqrt(2.0 / fan_in)).astype(np.float64)

    P = {
        # event encoder: enc_t = tanh(We x_t + be)
        "We": he((Dh, Din), Din), "be": np.zeros(Dh),
        # the fixed "royal-glory" invariant vector, injected into every episode
        "v":  (r.standard_normal(Dh) * 0.1),
        "Wv": he((Dh, Dh), Dh),
        # gated recurrent cell (minimal-gated unit: update gate z + candidate c)
        "Wz": he((Dh, 2 * Dh), 2 * Dh), "bz": np.zeros(Dh),
        "Wc": he((Dh, 2 * Dh), 2 * Dh), "bc": np.zeros(Dh),
        # THE SCRIBE: re-edit matrix that pulls each past episode toward the consolidated
        # backward-summary s_t (the later, authoritative recension).
        "Ws": (r.standard_normal((Dh, Dh)) * (1.0 / np.sqrt(Dh))),
        "b_beta": np.array(0.0),         # scalar logit -> beta = sigmoid(b_beta), the recension blend
        # heads (all read the EDITED episode e_t)
        "wg": (r.standard_normal(Dh) * (1.0 / np.sqrt(Dh))), "bg": np.array(0.0),  # glory scalar
        "wf": (r.standard_normal(Dh) * (1.0 / np.sqrt(Dh))), "bf": np.array(0.0),  # fracture logit
        "Wp": he((Din, Dh), Dh), "bp": np.zeros(Din),                              # next-event predictor
    }
    return P


# ======================================================================================
# 2. FORWARD PASS  (single sequence X of shape [T, Din])
# ----------------------------------------------------------------------------------
# Returns the outputs and a cache of every intermediate needed for manual backprop.
# ======================================================================================
def forward(P, X):
    T, Din = X.shape
    Dh = P["be"].shape[0]
    beta = float(sigmoid(P["b_beta"]))  # recension blend in (0,1)

    enc = np.zeros((T, Dh))
    a   = np.zeros((T, Dh))     # invariant-injected pre-state
    cat = np.zeros((T, 2 * Dh))
    z   = np.zeros((T, Dh))
    c   = np.zeros((T, Dh))
    h   = np.zeros((T, Dh))     # causal chronicle states
    h_prev_list = []

    Wv_v = P["Wv"] @ P["v"]

    # ---- (a) causal pass: narrate each new episode under the glory invariant ----
    h_prev = np.zeros(Dh)
    for t in range(T):
        enc[t] = np.tanh(P["We"] @ X[t] + P["be"])
        a[t]   = enc[t] + Wv_v                       # inject invariant
        ct_in  = np.concatenate([h_prev, a[t]])
        cat[t] = ct_in
        z[t]   = sigmoid(P["Wz"] @ ct_in + P["bz"])  # update gate
        c[t]   = np.tanh(P["Wc"] @ ct_in + P["bc"])  # candidate episode
        h[t]   = (1.0 - z[t]) * h_prev + z[t] * c[t]
        h_prev_list.append(h_prev)
        h_prev = h[t]

    # ---- (b) THE SCRIBE: backward recension pass -> summary s_t ----
    # s_T = h_T ; s_t = (1-beta) h_t + beta s_{t+1}   (later recensions overwrite earlier)
    s = np.zeros((T, Dh))
    s[T - 1] = h[T - 1]
    for t in range(T - 2, -1, -1):
        s[t] = (1.0 - beta) * h[t] + beta * s[t + 1]

    # ---- (c) re-edit each episode toward the consolidated narrative ----
    # e_t = h_t + Ws (s_t - h_t)
    e = h + (s - h) @ P["Ws"].T

    # ---- (d) heads (read edited episodes) ----
    g = e @ P["wg"] + P["bg"]                  # [T] glory scalar (the carved line)
    p_logit = e @ P["wf"] + P["bf"]            # [T] fracture logit
    p = sigmoid(p_logit)                       # [T] fracture probability
    xhat = e @ P["Wp"].T + P["bp"]             # [T, Din] predict NEXT event (xhat[t] ~ X[t+1])

    cache = dict(X=X, T=T, Din=Din, Dh=Dh, beta=beta,
                 enc=enc, a=a, cat=cat, z=z, c=c, h=h, s=s, e=e,
                 g=g, p_logit=p_logit, p=p, xhat=xhat,
                 h_prev_list=h_prev_list, Wv_v=Wv_v)
    out = dict(g=g, p=p, xhat=xhat, e=e)
    return out, cache


# ======================================================================================
# 3. LOSS  (prediction + glory-monotonicity + fracture)
# ----------------------------------------------------------------------------------
#   L_pred  : mean-squared next-event prediction error (truth-modelling term)
#   L_mono  : penalises any DECREASE of the glory line g_t (the king's invariant)
#             relu(g_{t-1} - g_t + margin)^2  ->  forces g_t >= g_{t-1} + margin
#   L_frac  : binary cross-entropy of fracture head vs. setback labels
# ======================================================================================
def compute_loss(out, cache, Y_frac, lam_mono=0.6, lam_frac=0.5, margin=0.05):
    T, Din = cache["T"], cache["Din"]
    g, p, xhat, X = cache["g"], cache["p"], cache["xhat"], cache["X"]

    # prediction (steps 0..T-2 predict the following event)
    if T >= 2:
        diff = xhat[:-1] - X[1:]
        L_pred = np.sum(diff ** 2) / ((T - 1) * Din)
    else:
        L_pred = 0.0

    # glory monotonicity
    if T >= 2:
        u = g[:-1] - g[1:] + margin          # >0 means glory failed to rise enough
        ru = relu(u)
        L_mono = np.sum(ru ** 2) / (T - 1)
    else:
        L_mono = 0.0

    # fracture BCE (clip for numerical safety)
    pc = np.clip(p, 1e-9, 1 - 1e-9)
    L_frac = -np.mean(Y_frac * np.log(pc) + (1 - Y_frac) * np.log(1 - pc))

    L = L_pred + lam_mono * L_mono + lam_frac * L_frac
    parts = dict(L_pred=float(L_pred), L_mono=float(L_mono), L_frac=float(L_frac))
    return float(L), parts


# ======================================================================================
# 4. BACKWARD PASS  (manual reverse-mode; returns grads for every parameter)
# ======================================================================================
def backward(P, cache, Y_frac, lam_mono=0.6, lam_frac=0.5, margin=0.05):
    X, T, Din, Dh = cache["X"], cache["T"], cache["Din"], cache["Dh"]
    beta = cache["beta"]
    enc, a, cat, z, c, h, s, e = (cache["enc"], cache["a"], cache["cat"],
                                  cache["z"], cache["c"], cache["h"],
                                  cache["s"], cache["e"])
    g, p, xhat = cache["g"], cache["p"], cache["xhat"]
    h_prev_list = cache["h_prev_list"]

    G = {k: np.zeros_like(np.asarray(v, dtype=np.float64)) for k, v in P.items()}

    # ---------- 4.1 gradient on edited episodes e_t from the three heads ----------
    de = np.zeros((T, Dh))

    # (a) prediction head: xhat[t] = Wp e_t + bp ; only t=0..T-2 contribute
    if T >= 2:
        dxhat = np.zeros((T, Din))
        dxhat[:-1] = (2.0 / ((T - 1) * Din)) * (xhat[:-1] - X[1:])
        for t in range(T):
            if t <= T - 2:
                G["Wp"] += np.outer(dxhat[t], e[t])
                G["bp"] += dxhat[t]
                de[t]   += P["Wp"].T @ dxhat[t]

    # (b) glory head + monotonicity penalty
    dg = np.zeros(T)
    if T >= 2:
        u = g[:-1] - g[1:] + margin
        mask = (u > 0).astype(np.float64)
        ru = relu(u)
        coef = (2.0 / (T - 1)) * ru * mask         # d L_mono / d u_t  (per t in 0..T-2)
        # u_t involves g_t (with +1) and g_{t+1} (with -1)
        dg[:-1] += lam_mono * coef                 # +1 on g_{t}
        dg[1:]  += lam_mono * (-coef)              # -1 on g_{t+1}
    for t in range(T):
        if dg[t] != 0.0:
            de[t]   += dg[t] * P["wg"]
            G["wg"] += dg[t] * e[t]
            G["bg"] += dg[t]

    # (c) fracture head (BCE): dL/dlogit = (1/T)(p - y) * lam_frac
    dpl = lam_frac * (p - Y_frac) / T
    for t in range(T):
        de[t]   += dpl[t] * P["wf"]
        G["wf"] += dpl[t] * e[t]
        G["bf"] += dpl[t]

    # ---------- 4.2 e_t = h_t + Ws (s_t - h_t) ----------
    # de -> grads on Ws, and gradients onto h_t and s_t
    dh_from_e = np.zeros((T, Dh))
    ds_from_e = np.zeros((T, Dh))
    for t in range(T):
        d = de[t]
        G["Ws"] += np.outer(d, (s[t] - h[t]))
        dh_from_e[t] += d - P["Ws"].T @ d          # d e/d h_t = I - Ws
        ds_from_e[t] += P["Ws"].T @ d              # d e/d s_t = Ws

    # ---------- 4.3 scribe backward (backward-EMA recursion) ----------
    # s_1 depends on s_2 ... so accumulate ds_total going t = 0..T-1
    ds_total = np.zeros((T, Dh))
    ds_total[0] = ds_from_e[0]
    for t in range(1, T):
        ds_total[t] = ds_from_e[t] + beta * ds_total[t - 1]

    dh_from_scribe = np.zeros((T, Dh))
    dbeta = 0.0
    # s_T = h_T
    dh_from_scribe[T - 1] += ds_total[T - 1]
    # s_t = (1-beta) h_t + beta s_{t+1}  for t < T-1
    for t in range(T - 1):
        dh_from_scribe[t] += (1.0 - beta) * ds_total[t]
        dbeta += float(ds_total[t] @ (s[t + 1] - h[t]))
    # beta = sigmoid(b_beta)
    G["b_beta"] += np.array(dbeta * beta * (1.0 - beta))

    # ---------- 4.4 total external gradient on each causal state h_t ----------
    dH_ext = dh_from_e + dh_from_scribe

    # ---------- 4.5 GRU/MGU backward through time ----------
    Wz, Wc = P["Wz"], P["Wc"]
    dh_carry = np.zeros(Dh)                         # gradient flowing back from h_{t+1}
    dv_acc = np.zeros(Dh)
    for t in range(T - 1, -1, -1):
        dh_t = dH_ext[t] + dh_carry
        h_prev = h_prev_list[t]
        # h_t = (1-z) h_prev + z c
        dz = dh_t * (c[t] - h_prev)
        dc = dh_t * z[t]
        dh_prev = dh_t * (1.0 - z[t])
        # gates
        dzz = dz * z[t] * (1.0 - z[t])              # through sigmoid
        dcc = dc * (1.0 - c[t] ** 2)                # through tanh
        G["Wz"] += np.outer(dzz, cat[t]); G["bz"] += dzz
        G["Wc"] += np.outer(dcc, cat[t]); G["bc"] += dcc
        dcat = Wz.T @ dzz + Wc.T @ dcc
        dh_prev += dcat[:Dh]
        da = dcat[Dh:]
        # a_t = enc_t + Wv v
        G["Wv"] += np.outer(da, P["v"])
        dv_acc  += P["Wv"].T @ da
        # enc_t = tanh(We x_t + be)
        dpre = da * (1.0 - enc[t] ** 2)
        G["We"] += np.outer(dpre, X[t]); G["be"] += dpre
        dh_carry = dh_prev
    G["v"] += dv_acc

    return G


# ======================================================================================
# 5. BATCH WRAPPER  (average loss & grads over a minibatch of sequences)
# ======================================================================================
def loss_and_grads(P, batch_X, batch_Y, **kw):
    G_sum = {k: np.zeros_like(np.asarray(v, dtype=np.float64)) for k, v in P.items()}
    L_sum = 0.0
    parts_sum = dict(L_pred=0.0, L_mono=0.0, L_frac=0.0)
    n = len(batch_X)
    for X, Y in zip(batch_X, batch_Y):
        out, cache = forward(P, X)
        L, parts = compute_loss(out, cache, Y, **kw)
        G = backward(P, cache, Y, **kw)
        L_sum += L
        for k in parts_sum:
            parts_sum[k] += parts[k]
        for k in G_sum:
            G_sum[k] += G[k]
    for k in G_sum:
        G_sum[k] /= n
    for k in parts_sum:
        parts_sum[k] /= n
    return L_sum / n, parts_sum, G_sum


# ======================================================================================
# 6. FINITE-DIFFERENCE GRADIENT CHECK  (mandatory; must pass)
# ======================================================================================
def gradient_check(seed=1):
    Din, Dh, T = 5, 7, 6
    P = init_params(Din, Dh, seed=seed)
    r = np.random.default_rng(seed + 100)
    X = r.standard_normal((T, Din))
    Y = (r.standard_normal(T) > 0.4).astype(np.float64)

    kw = dict(lam_mono=0.6, lam_frac=0.5, margin=0.05)

    def loss_only(P_):
        out, cache = forward(P_, X)
        L, _ = compute_loss(out, cache, Y, **kw)
        return L

    out, cache = forward(P, X)
    _, _ = compute_loss(out, cache, Y, **kw)
    G = backward(P, cache, Y, **kw)

    eps = 1e-5
    max_rel = 0.0
    worst = None
    checked = 0
    for name, val in P.items():
        val = np.atleast_1d(np.asarray(val, dtype=np.float64))
        flat = val.ravel()
        # sample up to 6 coordinates per parameter to keep the check fast but thorough
        idxs = range(flat.size) if flat.size <= 6 else \
            np.random.default_rng(7).choice(flat.size, 6, replace=False)
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps
            Lp = loss_only(P)
            flat[i] = orig - eps
            Lm = loss_only(P)
            flat[i] = orig
            num = (Lp - Lm) / (2 * eps)
            ana = np.atleast_1d(G[name]).ravel()[i]
            denom = max(1e-8, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            checked += 1
            if rel > max_rel:
                max_rel = rel
                worst = (name, i, num, ana)
    return max_rel, checked, worst


# ======================================================================================
# 7. SYNTHETIC CAMPAIGN DATA
# ----------------------------------------------------------------------------------
# Each sequence is a reign of T eponym-years. A latent "momentum" random walk drives a
# success magnitude per year (mostly positive — kings campaign successfully — but with
# occasional sharp SETBACKS). The event vector carries the success plus contextual noise.
# A setback year (success < threshold) is labelled fracture-positive: the event the
# monotone story cannot absorb.
# ======================================================================================
def make_sequence(T=12, Din=5, setback_thresh=-0.6, rng=RNG):
    success = np.zeros(T)
    mom = 0.4                                    # reigns start with momentum
    for t in range(T):
        mom = 0.85 * mom + 0.15 * rng.normal(0.35, 0.5)   # drift upward, autocorrelated
        shock = 0.0
        if rng.random() < 0.18:                  # occasional disaster (coalition holds, revolt)
            shock = -rng.uniform(0.8, 1.8)
        success[t] = np.tanh(mom + shock)
    ctx = rng.standard_normal((T, Din - 1)) * 0.4
    X = np.concatenate([success[:, None], ctx], axis=1)
    Y = (success < setback_thresh).astype(np.float64)
    return X, Y


def make_dataset(n, **kw):
    XS, YS = [], []
    for _ in range(n):
        X, Y = make_sequence(**kw)
        XS.append(X); YS.append(Y)
    return XS, YS


# ======================================================================================
# 8. ADAM OPTIMISER
# ======================================================================================
class Adam:
    def __init__(self, P, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(np.asarray(v, dtype=np.float64)) for k, v in P.items()}
        self.v = {k: np.zeros_like(np.asarray(v, dtype=np.float64)) for k, v in P.items()}
        self.t = 0

    def step(self, P, G):
        self.t += 1
        for k in P:
            g = np.asarray(G[k], dtype=np.float64)
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            upd = self.lr * mhat / (np.sqrt(vhat) + self.eps)
            if np.ndim(P[k]) == 0:
                P[k] = np.asarray(P[k]) - upd        # scalar params (bg/bf/b_beta)
            else:
                P[k] -= upd
        return P


# ======================================================================================
# 9. EVALUATION HELPERS
# ======================================================================================
def auc(scores, labels):
    # Mann-Whitney U / rank-based AUC. Returns 0.5 if a class is missing.
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[:len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def evaluate(P, XS, YS):
    all_p, all_y = [], []
    mono_viol, mono_tot, glory_gain = 0, 0, 0.0
    for X, Y in zip(XS, YS):
        out, _ = forward(P, X)
        g = out["g"]
        all_p.append(out["p"]); all_y.append(Y)
        d = np.diff(g)
        mono_viol += int(np.sum(d < -1e-6))
        mono_tot  += len(d)
        glory_gain += float(g[-1] - g[0])
    all_p = np.concatenate(all_p); all_y = np.concatenate(all_y)
    return dict(
        fracture_auc=auc(all_p, all_y),
        mono_violation_rate=mono_viol / max(1, mono_tot),
        mean_glory_gain=glory_gain / len(XS),
    )


# ======================================================================================
# 10. MAIN  —  gradient check, training, self-tests
# ======================================================================================
def main():
    np.set_printoptions(precision=4, suppress=True)
    print("=" * 78)
    print("THE ANNALIST — Monotone Self-Narrative Network  (Mind #19, Shalmaneser III)")
    print("=" * 78)

    # ---- 10.1 gradient check (must pass) ----
    print("\n[1] Finite-difference gradient check ...")
    max_rel, checked, worst = gradient_check()
    print(f"    coordinates checked : {checked}")
    print(f"    max relative error  : {max_rel:.3e}   (worst: {worst[0]} idx {worst[1]})")
    assert max_rel < 1e-4, f"GRADIENT CHECK FAILED (max_rel={max_rel:.3e})"
    print("    PASS  (manual backprop matches numerical gradient)")

    # ---- 10.2 data ----
    Din, Dh, T = 5, 24, 12
    train_X, train_Y = make_dataset(420, T=T, Din=Din)
    test_X,  test_Y  = make_dataset(120, T=T, Din=Din)

    # ---- 10.3 train ----
    print("\n[2] Training the Annalist ...")
    P = init_params(Din, Dh, seed=3)
    opt = Adam(P, lr=4e-3)
    kw = dict(lam_mono=0.6, lam_frac=0.5, margin=0.05)
    batch = 16
    epochs = 30

    L0, parts0, _ = loss_and_grads(P, train_X[:batch], train_Y[:batch], **kw)
    print(f"    initial loss        : {L0:.4f}   "
          f"(pred {parts0['L_pred']:.3f} | mono {parts0['L_mono']:.3f} | frac {parts0['L_frac']:.3f})")

    idx = np.arange(len(train_X))
    last = None
    for ep in range(epochs):
        RNG.shuffle(idx)
        run = 0.0; nb = 0
        for b in range(0, len(idx), batch):
            sel = idx[b:b + batch]
            bx = [train_X[i] for i in sel]; by = [train_Y[i] for i in sel]
            L, parts, G = loss_and_grads(P, bx, by, **kw)
            opt.step(P, G)
            run += L; nb += 1
        if ep % 5 == 0 or ep == epochs - 1:
            Lf, pf, _ = loss_and_grads(P, train_X[:64], train_Y[:64], **kw)
            print(f"    epoch {ep:2d}  loss {run/nb:.4f}   "
                  f"(pred {pf['L_pred']:.3f} | mono {pf['L_mono']:.3f} | frac {pf['L_frac']:.3f})")
            last = Lf

    # ---- 10.4 evaluate ----
    print("\n[3] Evaluation on held-out reigns ...")
    ev = evaluate(P, test_X, test_Y)
    print(f"    fracture-head AUC      : {ev['fracture_auc']:.3f}   (0.5 = chance)")
    print(f"    glory monotonicity     : {(1-ev['mono_violation_rate'])*100:.1f}% of steps non-decreasing")
    print(f"    mean glory gain / reign: {ev['mean_glory_gain']:+.3f}  (the carved line ascends)")

    # ---- 10.5 demonstrate the mind on one reign ----
    print("\n[4] One reign, narrated ...")
    Xd, Yd = test_X[0], test_Y[0]
    out, _ = forward(P, Xd)
    print("    year :  success(real)   glory(story)   fracture(p)   setback?")
    for t in range(len(Xd)):
        flag = "  <-- REVOLT RISK" if (out["p"][t] > 0.5) else ""
        print(f"    {t:>4} :   {Xd[t,0]:+6.2f}        {out['g'][t]:+6.2f}        "
              f"{out['p'][t]:.2f}        {int(Yd[t])}{flag}")
    print("    (Reality dips and stalls; the official glory line only ever rises —")
    print("     until the fracture head fires on the setback the story cannot absorb.)")

    # ---- 10.6 self-tests ----
    print("\n[5] Self-tests ...")
    assert last is not None and last < L0, "training did not reduce loss"
    print(f"    loss decreased            : {L0:.3f} -> {last:.3f}   OK")
    assert ev["fracture_auc"] > 0.65, f"fracture AUC too low ({ev['fracture_auc']:.3f})"
    print(f"    fracture AUC > 0.65       : {ev['fracture_auc']:.3f}   OK")
    assert ev["mono_violation_rate"] < 0.10, f"glory not monotone ({ev['mono_violation_rate']:.3f})"
    print(f"    glory monotone (<10% viol): {ev['mono_violation_rate']*100:.1f}%   OK")
    assert ev["mean_glory_gain"] > 0.0, "glory line failed to ascend"
    print(f"    glory ascends             : {ev['mean_glory_gain']:+.3f}   OK")
    assert max_rel < 1e-4
    print(f"    gradient check            : {max_rel:.1e}   OK")

    print("\n" + "=" * 78)
    print("ALL TESTS PASSED — the Annalist tells a monotone story over a non-monotone")
    print("reality, and learns exactly where that story must fracture.")
    print("=" * 78)


if __name__ == "__main__":
    main()

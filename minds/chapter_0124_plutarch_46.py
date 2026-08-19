#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
The Synkritic Character Encoder (SCE): a "Parallel Lives" network
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 7 Minds 121 - 140 Available on Amazon https://www.amazon.com/dp/B0HFN6GXMH
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 124: Plutarch of Chaeronea (c. 46 – c. 120 CE)
================================================================================   

WHY THIS ARCHITECTURE (and not a Transformer)
---------------------------------------------
Plutarch's whole method is *synkrisis* — paired comparison. He did not define a
character in isolation; he set one soul beside another (a Greek beside a Roman)
and let the CONTRAST reveal the essence of each. Three ideas of his are literally
wired into this network:

  1. SYNKRISIS (learning by comparison, not by isolated labels).
     The model never scores a single life on its own. It learns a *metric* over
     souls: two "parallel" lives (same virtue-signature) are pulled together in
     character-space; two contrasting lives are pushed apart. Character is a
     relational quantity, discovered only against a foil.

  2. ETHOS vs TYCHE (character vs. fortune).
     Plutarch insists on separating what a person IS (stable disposition) from
     what merely BEFELL them (circumstance, luck). Each life is encoded into two
     latent codes: a CHARACTER code (ethos) and a FORTUNE code (tyche). A
     decorrelation penalty forces the character code to be statistically
     independent of the fortune code — virtue is the part that is invariant
     under changing fortune.

  3. THE TELLING DETAIL (diagnostic sparsity).
     "A slight thing like a phrase or a jest often makes a greater revelation of
     character than battles where thousands fall" (Life of Alexander, 1). Most
     episodes of a life are moral noise; a few are diagnostic. A sparse saliency
     gate (entropy-regularised attention over episodes) learns to spend its
     weight on the few telling moments.

A concluding SYNKRISIS head reads two character codes and renders a comparative
verdict (which soul leads in which virtue) — the little essay Plutarch appended
to each pair of Lives.

IMPLEMENTATION CONTRACT (shared across the whole corpus)
--------------------------------------------------------
  * pure NumPy, from scratch, hand-written forward AND backward.
  * a finite-difference gradient check that MUST pass (see grad_check()).
  * a real training loop on synthetic "lives" that measurably reduces loss.
  * self-tests + a small qualitative demo on Plutarch's own famous pairs.

Run:  python3 chapter_0124_plutarch_46.py
================================================================================
"""

import numpy as np

RNG = np.random.default_rng(46)  # 46 CE — Plutarch's birth year, for reproducibility


# ============================================================================
#  0.  VIRTUES  (the moral dimensions the character code must carry)
# ============================================================================
# Plutarch's working vocabulary of virtue/vice. These are the axes of "ethos".
VIRTUES = [
    "andreia",      # courage
    "dikaiosyne",   # justice
    "sophrosyne",   # temperance / self-command
    "phronesis",    # practical wisdom
    "philanthropia",# love of humanity / clemency
    "philotimia",   # ambition (a double-edged virtue in Plutarch)
]
V = len(VIRTUES)


# ============================================================================
#  1.  SYNTHETIC "LIVES"  —  a life is a sequence of episodes
# ============================================================================
# Each EPISODE is a feature vector = [ deed-signature (V dims) , fortune (Ff dims) ].
# The deed-signature says which virtues/vices an action expresses; the fortune
# block encodes circumstance (birth, wealth, the accidents of the age) that is
# morally irrelevant to *character* but colours the surface of a life.
#
# A LIFE has:
#   - a hidden virtue profile (what the soul truly is)           -> supervises ethos
#   - a hidden fortune profile (the luck it was dealt)           -> nuisance signal
#   - E episodes, of which only a FEW are "telling" (diagnostic) -> sparsity target

Ff = 4                     # fortune feature dimensions
FEAT = V + Ff              # per-episode feature width
E_LEN = 12                 # episodes per life


def make_life(virtue_profile, fortune_profile, n_telling=3, noise=0.35):
    """Generate one life (E_LEN x FEAT) from a hidden virtue + fortune profile.

    Telling episodes carry a clean projection of the virtue profile; the rest
    are mostly circumstance and noise. This is the world Plutarch believed in:
    character leaks out in a handful of revealing moments.
    """
    life = np.zeros((E_LEN, FEAT), dtype=np.float64)
    telling = RNG.choice(E_LEN, size=n_telling, replace=False)
    for e in range(E_LEN):
        if e in telling:
            # a diagnostic act: strong, clean expression of the true virtues
            life[e, :V] = virtue_profile + noise * 0.4 * RNG.standard_normal(V)
        else:
            # a mundane act: NO character signal — pure circumstance and noise.
            # Averaging these in only dilutes; the soul is legible only through
            # the few telling episodes, so the gate is forced to find them.
            life[e, :V] = noise * RNG.standard_normal(V)
        # fortune block: circumstance shared across the life + small jitter
        life[e, V:] = fortune_profile + 0.2 * RNG.standard_normal(Ff)
    return life, telling


def make_dataset(n_lives=240):
    """Sample lives whose virtue profiles cluster into a few 'characters'.

    Parallel pairs (label 1) share a dominant-virtue cluster; contrasting pairs
    (label 0) do not. Fortune is sampled INDEPENDENTLY of virtue, so a good model
    must learn to ignore it when judging character.
    """
    # a handful of archetypal virtue clusters (dominant-virtue centroids)
    centroids = np.eye(V) * 1.6 + 0.2
    lives, virtue_targets, clusters = [], [], []
    for _ in range(n_lives):
        c = RNG.integers(0, V)
        vp = centroids[c] + 0.5 * RNG.standard_normal(V)
        vp = np.clip(vp, -1.0, 2.5)
        fp = RNG.standard_normal(Ff)                 # fortune ⟂ virtue by construction
        life, _ = make_life(vp, fp)
        lives.append(life)
        # multi-label virtue target in [0,1] via a squashed profile
        virtue_targets.append(1.0 / (1.0 + np.exp(-(vp - 0.8))))
        clusters.append(c)
    return (np.array(lives), np.array(virtue_targets), np.array(clusters))


def make_pairs(clusters, n_pairs=400):
    """Build (i, j, y) synkrisis pairs. y=1 if the two lives are 'parallel'
    (same dominant-virtue cluster), else 0. Roughly balanced."""
    n = len(clusters)
    pairs = []
    for _ in range(n_pairs):
        i = RNG.integers(0, n)
        if RNG.random() < 0.5:  # try to build a positive (parallel) pair
            same = np.where(clusters == clusters[i])[0]
            j = int(RNG.choice(same))
        else:
            diff = np.where(clusters != clusters[i])[0]
            j = int(RNG.choice(diff))
        if i == j:
            continue
        y = 1.0 if clusters[i] == clusters[j] else 0.0
        pairs.append((i, j, y))
    return pairs


# ============================================================================
#  2.  THE MODEL  —  Synkritic Character Encoder
# ============================================================================
# Dimensions
H  = 24    # episode hidden width
Hg = 16    # saliency gate hidden width
C  = 12    # character (ethos) code width
D  = 8     # fortune (tyche) code width


def he_init(shape):
    fan_in = shape[0]
    return RNG.standard_normal(shape) * np.sqrt(2.0 / fan_in)


class SynkriticEncoder:
    """Encodes a batch of lives into (character, fortune, virtue_logits, attn)."""

    def __init__(self):
        p = {}
        p["W_enc"] = he_init((FEAT, H)); p["b_enc"] = np.zeros(H)
        p["W_gate"] = he_init((H, Hg));  p["v_gate"] = he_init((Hg,)) * 0.3
        p["W_char"] = he_init((H, C));   p["b_char"] = np.zeros(C)
        p["W_fort"] = he_init((H, D));   p["b_fort"] = np.zeros(D)
        p["W_vir"] = he_init((C, V));    p["b_vir"] = np.zeros(V)
        self.p = p

    # ---- forward for a batch of lives : X is (N, E_LEN, FEAT) -------------
    def forward(self, X):
        p = self.p
        N = X.shape[0]
        cache = {"X": X, "N": N}

        # (1) episode encoder : tanh(X W_enc + b)      -> Hh (N,E,H)
        pre = X @ p["W_enc"] + p["b_enc"]              # (N,E,H)
        Hh = np.tanh(pre)
        cache["pre"], cache["Hh"] = pre, Hh

        # (2) telling-detail saliency gate : score_e = v . tanh(Hh W_gate)
        g_pre = Hh @ p["W_gate"]                       # (N,E,Hg)
        Gt = np.tanh(g_pre)                            # (N,E,Hg)
        scores = Gt @ p["v_gate"]                      # (N,E)
        scores = scores - scores.max(axis=1, keepdims=True)   # stabilise softmax
        expd = np.exp(scores)
        attn = expd / expd.sum(axis=1, keepdims=True)  # (N,E) sparse-ish weights
        cache["g_pre"], cache["Gt"], cache["attn"] = g_pre, Gt, attn

        # (3) pooled life vector z = Σ_e attn_e * Hh_e  -> (N,H)
        z = np.einsum("ne,neh->nh", attn, Hh)
        cache["z"] = z

        # (4) split into character (ethos) and fortune (tyche) codes
        char = z @ p["W_char"] + p["b_char"]           # (N,C)
        fort = z @ p["W_fort"] + p["b_fort"]           # (N,D)
        cache["char"], cache["fort"] = char, fort

        # (5) virtue readout from CHARACTER only (fortune must not leak in)
        vir_logits = char @ p["W_vir"] + p["b_vir"]    # (N,V)
        cache["vir_logits"] = vir_logits

        self.cache = cache
        return char, fort, vir_logits, attn


# ============================================================================
#  3.  LOSSES  (synkrisis + virtue + fortune-invariance + telling sparsity)
# ============================================================================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def compute_loss(model, X, pairs, vir_targets,
                 margin=1.5, w_syn=1.0, w_vir=1.0, w_dec=0.5, w_ent=0.22):
    """Full forward + scalar loss. Returns (total, parts, grads-ready cache)."""
    char, fort, vir_logits, attn = model.forward(X)
    N = X.shape[0]
    cache = model.cache

    # ---- (A) SYNKRISIS contrastive loss over character codes --------------
    # d2 = ||char_i - char_j||^2 ; parallel(y=1) pulls together, else margin push
    idx_i = np.array([pr[0] for pr in pairs])
    idx_j = np.array([pr[1] for pr in pairs])
    y = np.array([pr[2] for pr in pairs])
    diff = char[idx_i] - char[idx_j]                   # (P,C)
    d2 = np.sum(diff * diff, axis=1)                   # (P,)
    d = np.sqrt(d2 + 1e-12)
    pos = y * d2
    neg = (1.0 - y) * np.maximum(0.0, margin - d) ** 2
    L_syn = np.mean(pos + neg)

    # ---- (B) VIRTUE multi-label BCE from character code -------------------
    ph = sigmoid(vir_logits)
    ph = np.clip(ph, 1e-7, 1 - 1e-7)
    L_vir = -np.mean(vir_targets * np.log(ph) + (1 - vir_targets) * np.log(1 - ph))

    # ---- (C) FORTUNE-INVARIANCE: decorrelate char vs fort -----------------
    cc = char - char.mean(axis=0, keepdims=True)
    ff = fort - fort.mean(axis=0, keepdims=True)
    Cov = (cc.T @ ff) / N                              # (C,D)
    L_dec = np.sum(Cov * Cov)

    # ---- (D) TELLING-DETAIL sparsity: minimise attention entropy ----------
    ent = -np.sum(attn * np.log(attn + 1e-12), axis=1)  # (N,)
    L_ent = np.mean(ent)

    total = w_syn * L_syn + w_vir * L_vir + w_dec * L_dec + w_ent * L_ent
    parts = {"syn": L_syn, "vir": L_vir, "dec": L_dec, "ent": L_ent, "total": total}

    # stash everything backward() needs
    cache.update(dict(idx_i=idx_i, idx_j=idx_j, y=y, diff=diff, d=d, d2=d2,
                      ph=ph, vir_targets=vir_targets, cc=cc, ff=ff, Cov=Cov,
                      attn=attn, margin=margin,
                      w_syn=w_syn, w_vir=w_vir, w_dec=w_dec, w_ent=w_ent,
                      P=len(pairs)))
    return total, parts


# ============================================================================
#  4.  BACKWARD  (hand-written reverse pass for every parameter)
# ============================================================================
def backward(model):
    p, c = model.p, model.cache
    N = c["N"]; P = c["P"]
    grads = {k: np.zeros_like(v) for k, v in p.items()}

    dchar = np.zeros_like(c["char"])   # (N,C)
    dfort = np.zeros_like(c["fort"])   # (N,D)

    # ---- (A) synkrisis contrastive -> dchar -------------------------------
    y = c["y"]; d = c["d"]; diff = c["diff"]; margin = c["margin"]
    w_syn = c["w_syn"]
    # dL/d(diff): pos term 2*y*diff ; neg term: 2*(1-y)*max(0,m-d)*(-1)*(diff/d)
    hinge = np.maximum(0.0, margin - d)
    coef_pos = 2.0 * y                                   # (P,)
    coef_neg = -2.0 * (1.0 - y) * hinge / (d + 1e-12)    # (P,)
    ddiff = (coef_pos + coef_neg)[:, None] * diff        # (P,C)
    ddiff *= (w_syn / P)
    np.add.at(dchar, c["idx_i"], ddiff)
    np.add.at(dchar, c["idx_j"], -ddiff)

    # ---- (B) virtue BCE -> dvir_logits -> dchar + W_vir/b_vir --------------
    w_vir = c["w_vir"]
    dlogits = (c["ph"] - c["vir_targets"]) / (N * V) * w_vir   # (N,V)
    grads["W_vir"] += c["char"].T @ dlogits
    grads["b_vir"] += dlogits.sum(axis=0)
    dchar += dlogits @ p["W_vir"].T

    # ---- (C) decorrelation -> dchar + dfort -------------------------------
    w_dec = c["w_dec"]
    Cov = c["Cov"]; cc = c["cc"]; ff = c["ff"]
    dcc = (2.0 / N) * (ff @ Cov.T) * w_dec               # (N,C)
    dff = (2.0 / N) * (cc @ Cov) * w_dec                 # (N,D)
    # backprop through column-centering (subtract mean over rows)
    dchar += dcc - dcc.mean(axis=0, keepdims=True)
    dfort += dff - dff.mean(axis=0, keepdims=True)

    # ---- char/fort heads : z -> W_char,b_char,W_fort,b_fort ---------------
    z = c["z"]
    grads["W_char"] += z.T @ dchar
    grads["b_char"] += dchar.sum(axis=0)
    grads["W_fort"] += z.T @ dfort
    grads["b_fort"] += dfort.sum(axis=0)
    dz = dchar @ p["W_char"].T + dfort @ p["W_fort"].T    # (N,H)

    # ---- (D) telling-detail entropy -> dattn (direct) ---------------------
    w_ent = c["w_ent"]
    attn = c["attn"]
    # L_ent = mean_n ( -Σ_e a*log a ); dL/da_e = -(log a + 1)/N
    dattn_ent = -(np.log(attn + 1e-12) + 1.0) * (w_ent / N)   # (N,E)

    # ---- pooling z = Σ_e attn_e Hh_e --------------------------------------
    Hh = c["Hh"]                                          # (N,E,H)
    # dz -> dHh (through attn weight) and -> dattn (through Hh)
    dHh = attn[:, :, None] * dz[:, None, :]               # (N,E,H)
    dattn = np.einsum("neh,nh->ne", Hh, dz) + dattn_ent   # (N,E)

    # ---- softmax backward : attn = softmax(scores) ------------------------
    # dscores = attn * (dattn - Σ_e attn*dattn)
    tmp = (dattn * attn).sum(axis=1, keepdims=True)
    dscores = attn * (dattn - tmp)                        # (N,E)

    # ---- gate : scores = tanh(Hh W_gate) . v_gate -------------------------
    Gt = c["Gt"]                                          # (N,E,Hg)
    grads["v_gate"] += np.einsum("ne,neh->h", dscores, Gt)
    dGt = dscores[:, :, None] * p["v_gate"][None, None, :]  # (N,E,Hg)
    dg_pre = dGt * (1.0 - Gt ** 2)                         # tanh'
    grads["W_gate"] += np.einsum("neh,neg->hg", Hh, dg_pre)
    dHh += np.einsum("neg,hg->neh", dg_pre, p["W_gate"])

    # ---- episode encoder : Hh = tanh(X W_enc + b) -------------------------
    pre = c["pre"]; X = c["X"]
    dpre = dHh * (1.0 - Hh ** 2)                           # (N,E,H)
    grads["W_enc"] += np.einsum("nef,neh->fh", X, dpre)
    grads["b_enc"] += dpre.sum(axis=(0, 1))

    return grads


# ============================================================================
#  5.  GRADIENT CHECK  (MANDATORY — finite differences vs analytic backward)
# ============================================================================
def grad_check(verbose=True):
    """Compare analytic gradients to central finite differences on the TOTAL
    scalar loss. Passes if the worst relative error over sampled entries is tiny.
    """
    model = SynkriticEncoder()
    # tiny deterministic batch
    lives, vt, clusters = make_dataset(n_lives=16)
    pairs = make_pairs(clusters, n_pairs=20)

    def loss_only():
        tot, _ = compute_loss(model, lives, pairs, vt)
        return tot

    total, _ = compute_loss(model, lives, pairs, vt)
    analytic = backward(model)

    eps = 1e-6
    worst = 0.0
    report = []
    for name, W in model.p.items():
        flat = W.ravel()
        gflat = analytic[name].ravel()
        # sample up to 12 coordinates per tensor
        k = min(12, flat.size)
        coords = RNG.choice(flat.size, size=k, replace=False)
        errs = []
        for ci in coords:
            orig = flat[ci]
            flat[ci] = orig + eps
            lp = loss_only()
            flat[ci] = orig - eps
            lm = loss_only()
            flat[ci] = orig
            num = (lp - lm) / (2 * eps)
            ana = gflat[ci]
            denom = abs(num) + abs(ana)
            # near-zero gradients (e.g. the fortune-head bias, which the
            # column-centering in the decorrelation term makes analytically
            # zero) use ABSOLUTE error; otherwise dividing tiny-by-tiny
            # inflates a meaningless relative ratio.
            if denom < 1e-6:
                rel = abs(num - ana)
            else:
                rel = abs(num - ana) / denom
            errs.append(rel)
            worst = max(worst, rel)
        report.append((name, max(errs)))
    if verbose:
        print("  gradient check (max relative error per parameter tensor)")
        for name, e in report:
            flag = "ok " if e < 1e-4 else "!! "
            print(f"    {flag}{name:9s}  {e:.2e}")
        print(f"  worst relative error = {worst:.2e}")
    ok = worst < 1e-4
    print(f"  GRADIENT CHECK: {'PASS' if ok else 'FAIL'}  (threshold 1e-4)\n")
    return ok


# ============================================================================
#  6.  TRAINING  (Adam on the full synkrisis objective)
# ============================================================================
class Adam:
    def __init__(self, params, lr=5e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.p = params; self.lr = lr; self.b1 = b1; self.b2 = b2; self.eps = eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads):
        self.t += 1
        for k in self.p:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            self.p[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


def virtue_accuracy(model, X, targets):
    _, _, logits, _ = model.forward(X)
    pred = (sigmoid(logits) > 0.5).astype(np.float64)
    gt = (targets > 0.5).astype(np.float64)
    return float((pred == gt).mean())


def pair_separation(model, X, pairs):
    """Mean character-distance for parallel vs contrasting pairs.
    A trained synkrisis model should keep parallel pairs much closer."""
    char, _, _, _ = model.forward(X)
    ii = np.array([p[0] for p in pairs]); jj = np.array([p[1] for p in pairs])
    yy = np.array([p[2] for p in pairs])
    d = np.linalg.norm(char[ii] - char[jj], axis=1)
    dp = d[yy == 1].mean() if (yy == 1).any() else float("nan")
    dn = d[yy == 0].mean() if (yy == 0).any() else float("nan")
    return dp, dn


def train(epochs=180, verbose=True):
    lives, vt, clusters = make_dataset(n_lives=240)
    pairs = make_pairs(clusters, n_pairs=500)
    # simple split
    ntr = 190
    Xtr, vttr, cltr = lives[:ntr], vt[:ntr], clusters[:ntr]
    Xte, vtte, clte = lives[ntr:], vt[ntr:], clusters[ntr:]
    tr_pairs = [pr for pr in make_pairs(cltr, 500)]
    te_pairs = [pr for pr in make_pairs(clte, 200)]

    model = SynkriticEncoder()
    opt = Adam(model.p, lr=6e-3)

    if verbose:
        print("  epoch    total     syn      vir      dec     ent   | vir_acc  d+   d-")
    hist = []
    for ep in range(epochs):
        total, parts = compute_loss(model, Xtr, tr_pairs, vttr)
        grads = backward(model)
        opt.step(grads)
        hist.append(parts["total"])
        if verbose and (ep % 20 == 0 or ep == epochs - 1):
            acc = virtue_accuracy(model, Xte, vtte)
            dp, dn = pair_separation(model, Xte, te_pairs)
            print(f"  {ep:5d}  {parts['total']:.4f}  {parts['syn']:.4f} "
                  f"{parts['vir']:.4f} {parts['dec']:.4f} {parts['ent']:.3f} | "
                  f" {acc:.3f}  {dp:.2f} {dn:.2f}")
    final_acc = virtue_accuracy(model, Xte, vtte)
    dp, dn = pair_separation(model, Xte, te_pairs)
    return model, hist, final_acc, dp, dn


# ============================================================================
#  7.  QUALITATIVE DEMO  —  Plutarch's own Parallel Lives
# ============================================================================
# We hand-craft virtue profiles for a few figures Plutarch actually paired, then
# ask the trained encoder to render a "synkrisis": for each virtue, which of the
# two souls leads. This is the little comparative essay he appended to each pair.
PLUTARCH_PAIRS = {
    "Alexander vs Caesar": (
        # philotimia (ambition) huge for both; Alexander higher andreia,
        # Caesar higher phronesis/philanthropia (clemency)
        dict(andreia=2.2, dikaiosyne=0.6, sophrosyne=0.3, phronesis=1.4,
             philanthropia=0.7, philotimia=2.4),
        dict(andreia=1.7, dikaiosyne=0.8, sophrosyne=0.5, phronesis=2.1,
             philanthropia=1.6, philotimia=2.3),
    ),
    "Demosthenes vs Cicero": (
        dict(andreia=1.1, dikaiosyne=1.4, sophrosyne=1.0, phronesis=1.6,
             philanthropia=0.9, philotimia=1.5),
        dict(andreia=0.8, dikaiosyne=1.2, sophrosyne=0.7, phronesis=1.7,
             philanthropia=1.3, philotimia=1.8),
    ),
    "Lycurgus vs Numa": (
        dict(andreia=1.4, dikaiosyne=2.1, sophrosyne=2.0, phronesis=1.9,
             philanthropia=0.8, philotimia=0.5),
        dict(andreia=0.9, dikaiosyne=2.0, sophrosyne=2.1, phronesis=1.8,
             philanthropia=1.7, philotimia=0.4),
    ),
}


def profile_to_life(prof):
    vp = np.array([prof[v] for v in VIRTUES], dtype=np.float64)
    fp = RNG.standard_normal(Ff)
    life, _ = make_life(vp, fp, n_telling=3, noise=0.2)
    return life


def synkrisis_demo(model):
    print("\n  SYNKRISIS — the trained encoder renders comparative verdicts:")
    for title, (pa, pb) in PLUTARCH_PAIRS.items():
        la = profile_to_life(pa)[None]
        lb = profile_to_life(pb)[None]
        ca, _, va, atta = model.forward(la)
        cb, _, vb, attb = model.forward(lb)
        sa, sb = sigmoid(va)[0], sigmoid(vb)[0]
        dist = float(np.linalg.norm(ca[0] - cb[0]))
        left, right = title.split(" vs ")
        print(f"\n   {title}   (character distance = {dist:.2f})")
        for k, v in enumerate(VIRTUES):
            lead = left if sa[k] >= sb[k] else right
            print(f"     {v:14s} {sa[k]:.2f} | {sb[k]:.2f}   -> {lead}")
        # which episode did the gate find 'telling' for the first figure?
        tell = int(np.argmax(atta[0]))
        print(f"     telling episode for {left}: #{tell} "
              f"(weight {atta[0][tell]:.2f})")


# ============================================================================
#  8.  SELF-TESTS
# ============================================================================
def self_tests():
    print("  self-tests:")
    lives, vt, clusters = make_dataset(n_lives=20)
    pairs = make_pairs(clusters, 30)
    m = SynkriticEncoder()
    char, fort, logits, attn = m.forward(lives)
    assert char.shape == (20, C), "character code shape"
    assert fort.shape == (20, D), "fortune code shape"
    assert logits.shape == (20, V), "virtue logits shape"
    assert np.allclose(attn.sum(axis=1), 1.0), "attention must sum to 1"
    assert (attn >= 0).all(), "attention must be non-negative"
    tot, parts = compute_loss(m, lives, pairs, vt)
    assert np.isfinite(tot), "loss must be finite"
    for key in ("syn", "vir", "dec", "ent"):
        assert parts[key] >= -1e-9, f"loss part {key} negative"
    print("    ok  shapes, attention normalisation, finite loss\n")


# ============================================================================
#  9.  MAIN
# ============================================================================
if __name__ == "__main__":
    print("=" * 74)
    print(" Plutarch — Synkritic Character Encoder (Parallel Lives network)")
    print("=" * 74)

    self_tests()

    print("  [1] gradient check")
    ok = grad_check()
    assert ok, "gradient check failed — refusing to ship"

    print("  [2] training on synthetic 'lives'")
    model, hist, acc, dp, dn = train(epochs=180)

    print(f"\n  final virtue-accuracy (held-out) : {acc:.3f}")
    print(f"  parallel-pair char distance  d+ : {dp:.2f}")
    print(f"  contrast-pair char distance  d- : {dn:.2f}")
    print(f"  synkrisis separation (d-/d+)    : {dn/max(dp,1e-9):.2f}x "
          f"(>1 means parallel souls cluster, as intended)")
    print(f"  loss reduced {hist[0]:.3f} -> {hist[-1]:.3f} "
          f"({100*(hist[0]-hist[-1])/hist[0]:.0f}% down)")

    synkrisis_demo(model)

    print("\n" + "=" * 74)
    print(" done.")
    print("=" * 74)

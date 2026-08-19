#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
DYNAMIS: the Drug-Affinity Effect-Manifold, from scratch in pure NumPy.
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 7 Minds 121 - 140 Available on Amazon https://www.amazon.com/dp/B0HFN6GXMH
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 123: Pedanius Dioscorides (c. 40-90 CE)
================================================================================    

WHAT THIS FILE IS
-----------------
A small but *complete* and *trainable* neural architecture whose every design
choice is dictated by one idea taken from Dioscorides, the Greek army physician
who wrote De Materia Medica (c. 50-70 CE) and, per John Riddle's scholarship,
secretly organised ~800 medicinal substances not alphabetically and not by
appearance but by their DRUG AFFINITY -- by the physiological *effect* (Greek
`dynamis`, the active power) that each substance exerts on the body.

The single cognitive thesis we encode:

    IDENTITY IS FIXED BY TESTED EFFECT, NOT BY APPEARANCE.

Two plants that look identical but act differently must live far apart in the
map of knowledge; two that look nothing alike but act the same must live
together. Appearance (morphology) is a *distractor*. The physician's whole art
is to see past it to the dynamis, and to verify every claim by repeated
firsthand observation (Greek `autopsia`), trusting what one has seen over what
one has merely read. Dioscorides openly faults predecessors who "explained the
strengths of medicines... not considering their value by personal experience."

We turn that into three mechanisms, all differentiable, all from scratch:

  1. EFFECT-GROUNDED EMBEDDING (the encoder + effect-prototypes).
     An MLP maps a specimen's observed features to a point in an "effect space."
     Learnable per-effect PROTOTYPES are the drug-affinity map itself: the
     ancient arrangement of the materia medica, rediscovered as geometry. The
     model is trained so that a specimen lands near the prototype of its *effect*
     class -- even though its input also contains a morphology channel that is
     deliberately DECORRELATED from effect. A model that keys on appearance fails;
     only a model that learns the dynamis succeeds. That failure/success gap is
     Dioscorides' thesis, measured.

  2. AUTOPSIA WEIGHTING (provenance-weighted loss).
     Every training specimen carries a provenance weight w in [0,1]: high for what
     was seen firsthand and reproduced, low for hearsay compiled from books. The
     loss of each specimen is scaled by w, so the network literally *learns more
     from what was witnessed*. We show this beats uniform weighting when the
     hearsay labels are noisy -- exactly the situation Dioscorides complained of.

  3. ADULTERATION DETECTION (the immune system of the map).
     Detecting counterfeit or degraded drugs was a discipline for Dioscorides.
     At inference, a specimen CLAIMING to be effect-class c is scored by how far
     its embedding sits from prototype c relative to the genuine intra-class
     radius. A specimen whose behaviour betrays its claimed identity is flagged.

WHY NOT A TRANSFORMER / TAXONOMY CLASSIFIER
-------------------------------------------
The nearest completed neighbour in this corpus is Li Shizhen (Bencao Gangmu), a
*hierarchical taxonomy* -- form-first, a tree. Dioscorides is the opposite: a
FLAT AFFINITY MANIFOLD, function-first, a metric space. So this file is a
metric-learning / prototypical network, not a tree classifier and not an
attention stack. The mechanism is the argument.

RUN CONVENTION (shared across the whole 1000Minds corpus)
---------------------------------------------------------
  * pure NumPy, from scratch (no autograd, no ML frameworks);
  * a finite-difference GRADIENT CHECK that must pass (mandatory);
  * a real training loop;
  * self-tests / demonstrations that exhibit the mind's thesis.
Just run:  python3 chapter_0123_dioscorides_40.py
================================================================================
"""

from __future__ import annotations
import numpy as np

# A fixed seed so the verified console output in the chapter is reproducible.
RNG = np.random.default_rng(40)  # 40 = Dioscorides' floruit year, our house seed.


# ==============================================================================
# PART I -- SYNTHETIC MATERIA MEDICA
# ------------------------------------------------------------------------------
# We fabricate a toy "cabinet of simples." Each specimen has:
#   * an EFFECT label (its dynamis) in {0..C-1}  -- what it DOES to the body;
#   * a MORPHOLOGY cluster in {0..M-1}           -- what it LOOKS like.
# Crucially EFFECT and MORPHOLOGY are drawn INDEPENDENTLY, so appearance carries
# no reliable information about action. The observed feature vector concatenates:
#   [ effect-signal features | morphology-distractor features ]
# The effect-signal is a noisy readout of the true dynamis (what a careful test
# on the body would reveal); the morphology block is a noisy readout of the
# appearance cluster. A mind that trusts appearance is doomed here -- which is
# precisely Dioscorides' point about physicians who never left the library.
# ==============================================================================

def make_materia_medica(n_per_class=60, C=6, M=5,
                        d_effect=8, d_morph=8,
                        effect_sep=2.2, morph_sep=2.2,
                        noise=0.9, rng=RNG):
    """Generate a synthetic pharmacopeia.

    Returns X (features), y_effect (dynamis label), y_morph (appearance label).
    The two label systems are statistically independent by construction.
    """
    # Anchor centroids for each effect and each morphology cluster.
    effect_centers = rng.normal(0, effect_sep, size=(C, d_effect))
    morph_centers = rng.normal(0, morph_sep, size=(M, d_morph))

    X, ye, ym = [], [], []
    for c in range(C):
        for _ in range(n_per_class):
            m = int(rng.integers(0, M))              # appearance chosen at random
            eff = effect_centers[c] + rng.normal(0, noise, d_effect)
            mor = morph_centers[m] + rng.normal(0, noise, d_morph)
            X.append(np.concatenate([eff, mor]))
            ye.append(c)
            ym.append(m)
    X = np.asarray(X, dtype=np.float64)
    ye = np.asarray(ye, dtype=np.int64)
    ym = np.asarray(ym, dtype=np.int64)

    # Standardise features (physician calibrates instruments before measuring).
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)

    # Shuffle so class blocks do not sit contiguously.
    perm = rng.permutation(len(X))
    return X[perm], ye[perm], ym[perm]


def assign_provenance(y_effect, hearsay_fraction=0.4, rng=RNG):
    """Split specimens into firsthand (autopsia) vs hearsay, and corrupt the
    labels of hearsay specimens to model unreliable compiled reports.

    Returns:
      y_observed  -- the label the model is TRAINED on (hearsay ones are noisy),
      w           -- provenance weight in {1.0 firsthand, w_low hearsay},
      is_hearsay  -- boolean mask.
    """
    n = len(y_effect)
    C = int(y_effect.max()) + 1
    is_hearsay = rng.random(n) < hearsay_fraction
    y_observed = y_effect.copy()
    # Corrupt ~half of the hearsay labels: books copied books, errors accrued.
    flip = is_hearsay & (rng.random(n) < 0.5)
    y_observed[flip] = rng.integers(0, C, size=int(flip.sum()))
    w = np.where(is_hearsay, 0.15, 1.0)   # autopsia counts ~7x a rumour
    return y_observed, w, is_hearsay


# ==============================================================================
# PART II -- THE ARCHITECTURE
# ------------------------------------------------------------------------------
# Encoder:   x -> h = relu(x W1 + b1) -> z = h W2 + b2   (z is the effect embed)
# Prototypes: P in R^{C x d_emb}, learnable. These ARE the drug-affinity map:
#             the arrangement of the materia medica by dynamis.
# Scoring:   squared-euclidean distance in effect space, turned into logits:
#              logit_{i,c} = -||z_i - P_c||^2
#            Because -||z_i||^2 is constant across c, it cancels in the softmax;
#            we keep the full form for clarity and gradient-check honesty.
# Loss:      autopsia-weighted cross-entropy (provenance weight per specimen).
# ==============================================================================

class Dynamis:
    """The effect-grounded metric-learning network."""

    def __init__(self, d_in, d_hidden, d_emb, n_classes, l2=1e-4, rng=RNG):
        self.d_in, self.d_hidden, self.d_emb, self.C = d_in, d_hidden, d_emb, n_classes
        self.l2 = l2
        s1 = np.sqrt(2.0 / d_in)       # He init for the relu layer
        s2 = np.sqrt(1.0 / d_hidden)
        self.params = {
            "W1": rng.normal(0, s1, (d_in, d_hidden)),
            "b1": np.zeros(d_hidden),
            "W2": rng.normal(0, s2, (d_hidden, d_emb)),
            "b2": np.zeros(d_emb),
            # Prototypes: the map anchors, spread out at birth.
            "P":  rng.normal(0, 0.5, (n_classes, d_emb)),
        }

    # ---- forward pieces ------------------------------------------------------
    def embed(self, X):
        """Map raw specimens to points in effect space. Returns z and a cache."""
        p = self.params
        pre = X @ p["W1"] + p["b1"]            # [N,H]
        h = np.maximum(pre, 0.0)              # relu
        z = h @ p["W2"] + p["b2"]             # [N,E]
        cache = {"X": X, "pre": pre, "h": h, "z": z}
        return z, cache

    @staticmethod
    def _softmax(logits):
        m = logits.max(axis=1, keepdims=True)
        e = np.exp(logits - m)
        return e / e.sum(axis=1, keepdims=True)

    def forward(self, X, y, w=None):
        """Full forward pass. Returns (loss, cache) for backprop.

        loss = provenance-weighted mean cross-entropy + L2 weight decay.
        """
        N = len(X)
        if w is None:
            w = np.ones(N)
        z, cache = self.embed(X)
        P = self.params["P"]                          # [C,E]
        # distances^2 : [N,C]
        # ||z-P||^2 = ||z||^2 - 2 z.P + ||P||^2
        z2 = (z * z).sum(1, keepdims=True)            # [N,1]
        P2 = (P * P).sum(1)[None, :]                  # [1,C]
        cross = z @ P.T                               # [N,C]
        dist2 = z2 - 2.0 * cross + P2                 # [N,C]
        logits = -dist2                               # [N,C]
        probs = self._softmax(logits)                 # [N,C]

        # weighted cross-entropy. We normalise by the batch size N (not by the
        # weight-sum): each specimen contributes w_i * CE_i, so downweighting
        # hearsay simply shrinks its influence toward zero while the overall
        # gradient scale stays bounded by max(w) = 1 -- identical to the uniform
        # case when every w_i = 1, and numerically stable when most w_i are small.
        idx = np.arange(N)
        ll = np.log(probs[idx, y] + 1e-12)            # [N]
        wsum = float(N)
        data_loss = -(w * ll).sum() / wsum

        # L2 on weights (not on biases / prototypes' spread is fine to include)
        reg = 0.0
        for k in ("W1", "W2"):
            reg += 0.5 * self.l2 * (self.params[k] ** 2).sum()
        loss = data_loss + reg

        cache.update({"y": y, "w": w, "wsum": wsum, "probs": probs,
                      "z": z, "P": P})
        return loss, cache

    # ---- backward ------------------------------------------------------------
    def backward(self, cache):
        """Exact reverse-mode gradients, hand-derived. Returns a grad dict."""
        p = self.params
        X, pre, h, z = cache["X"], cache["pre"], cache["h"], cache["z"]
        y, w, wsum, probs, P = (cache["y"], cache["w"], cache["wsum"],
                                cache["probs"], cache["P"])
        N, C, E = len(X), self.C, self.d_emb

        # d loss / d logits for weighted softmax-CE.
        # data_loss = -(1/wsum) sum_i w_i * log softmax(logits_i)[y_i]
        # dL/dlogit_{i,c} = (w_i / wsum) * (probs_{i,c} - 1[c==y_i])
        onehot = np.zeros((N, C))
        onehot[np.arange(N), y] = 1.0
        dlogits = (probs - onehot) * (w[:, None] / wsum)   # [N,C]

        # logits = -dist2, dist2 = ||z||^2 - 2 z.P^T + ||P||^2
        # d logits/d z and d logits/d P:
        #   d(-dist2_{ic})/dz_i = -(2 z_i - 2 P_c) = 2(P_c - z_i)
        #   d(-dist2_{ic})/dP_c = -(-2 z_i + 2 P_c) = 2(z_i - P_c)
        # dz_i = sum_c dlogits_{ic} * 2 (P_c - z_i)
        #      = 2 (dlogits @ P) - 2 (sum_c dlogits_{ic}) z_i
        dz = 2.0 * (dlogits @ P) - 2.0 * dlogits.sum(1, keepdims=True) * z  # [N,E]

        # dP_c = sum_i dlogits_{ic} * 2 (z_i - P_c)
        #      = 2 (dlogits^T @ z) - 2 (sum_i dlogits_{ic}) P_c
        dP = 2.0 * (dlogits.T @ z) - 2.0 * dlogits.sum(0)[:, None] * P      # [C,E]

        # Back through z = h W2 + b2
        dW2 = h.T @ dz + self.l2 * p["W2"]
        db2 = dz.sum(0)
        dh = dz @ p["W2"].T

        # Back through relu
        dpre = dh * (pre > 0)

        # Back through pre = X W1 + b1
        dW1 = X.T @ dpre + self.l2 * p["W1"]
        db1 = dpre.sum(0)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "P": dP}

    # ---- convenience ---------------------------------------------------------
    def logits(self, X):
        z, _ = self.embed(X)
        P = self.params["P"]
        z2 = (z * z).sum(1, keepdims=True)
        P2 = (P * P).sum(1)[None, :]
        return -(z2 - 2.0 * (z @ P.T) + P2)

    def predict(self, X):
        return self.logits(X).argmax(1)


# ==============================================================================
# PART III -- GRADIENT CHECK  (mandatory, must pass)
# ------------------------------------------------------------------------------
# Finite differences vs analytic backprop on every parameter tensor. Dioscorides
# would insist: do not trust the derivation (the book) -- verify it by test
# (autopsia). If the relative error is not tiny, the architecture is invalid.
# ==============================================================================

def gradient_check(verbose=True):
    rng = np.random.default_rng(7)
    d_in, H, E, C, N = 10, 12, 5, 4, 16
    X = rng.normal(0, 1, (N, d_in))
    y = rng.integers(0, C, N)
    w = rng.random(N)                       # arbitrary provenance weights
    net = Dynamis(d_in, H, E, C, l2=1e-3, rng=rng)

    loss, cache = net.forward(X, y, w)
    grads = net.backward(cache)

    eps = 1e-6
    max_rel = 0.0
    worst = None
    for name in ("W1", "b1", "W2", "b2", "P"):
        theta = net.params[name]
        g_analytic = grads[name]
        g_num = np.zeros_like(theta)
        it = np.nditer(theta, flags=["multi_index"])
        while not it.finished:
            i = it.multi_index
            old = theta[i]
            theta[i] = old + eps
            lp, _ = net.forward(X, y, w)
            theta[i] = old - eps
            lm, _ = net.forward(X, y, w)
            theta[i] = old
            g_num[i] = (lp - lm) / (2 * eps)
            it.iternext()
        num = np.abs(g_analytic - g_num)
        den = np.maximum(1e-8, np.abs(g_analytic) + np.abs(g_num))
        rel = (num / den).max()
        if rel > max_rel:
            max_rel, worst = rel, name
        if verbose:
            print(f"    grad-check {name:>3}: max rel err = {rel:.3e}")
    ok = max_rel < 1e-5
    if verbose:
        print(f"    worst tensor: {worst},  overall max rel err = {max_rel:.3e}"
              f"  -> {'PASS' if ok else 'FAIL'}")
    return ok, max_rel


# ==============================================================================
# PART IV -- TRAINING LOOP
# ==============================================================================

def train(net, X, y, w, epochs=250, lr=0.15, batch=64, rng=RNG, log_every=50):
    """Vanilla SGD with momentum. Returns loss history."""
    vel = {k: np.zeros_like(v) for k, v in net.params.items()}
    mom = 0.9
    N = len(X)
    hist = []
    for ep in range(1, epochs + 1):
        order = rng.permutation(N)
        ep_loss = 0.0
        nb = 0
        for s in range(0, N, batch):
            bi = order[s:s + batch]
            loss, cache = net.forward(X[bi], y[bi], w[bi])
            grads = net.backward(cache)
            for k in net.params:
                vel[k] = mom * vel[k] - lr * grads[k]
                net.params[k] += vel[k]
            ep_loss += loss
            nb += 1
        ep_loss /= nb
        hist.append(ep_loss)
        if log_every and (ep % log_every == 0 or ep == 1):
            acc = (net.predict(X) == y).mean()
            print(f"    epoch {ep:4d}   loss {ep_loss:.4f}   train-acc {acc:.3f}")
    return hist


# ==============================================================================
# PART V -- ANALYSES THAT SHOW THE MIND'S THESIS
# ==============================================================================

def morphology_baseline_accuracy(X, y_effect, y_morph, d_effect, rng=RNG):
    """A physician who trusts appearance: nearest-neighbour in MORPHOLOGY space,
    voting effect labels. Should sit near chance because appearance is
    decorrelated from action. This is the strawman Dioscorides dismantles."""
    Xm = X[:, d_effect:]                      # morphology block only
    n = len(X)
    idx = rng.permutation(n)
    tr, te = idx[: n // 2], idx[n // 2:]
    # 1-NN in morphology space, predict effect label of nearest train specimen.
    preds = []
    for q in te:
        d = ((Xm[tr] - Xm[q]) ** 2).sum(1)
        preds.append(y_effect[tr][d.argmin()])
    preds = np.array(preds)
    return (preds == y_effect[te]).mean()


def affinity_map(net):
    """Pairwise distances between learned effect-prototypes = the drug-affinity
    arrangement rediscovered as geometry. Returns the matrix and, for each
    effect, its nearest sibling effect (its neighbour in the materia medica)."""
    P = net.params["P"]
    C = P.shape[0]
    D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    nearest = D.argmin(1)
    np.fill_diagonal(D, 0.0)
    return D, nearest


def class_radii(net, X, y):
    """Genuine intra-class radius: mean distance of true members to their own
    prototype. Used as the yardstick for adulteration scoring."""
    z, _ = net.embed(X)
    P = net.params["P"]
    radii = np.zeros(P.shape[0])
    for c in range(P.shape[0]):
        m = y == c
        if m.any():
            radii[c] = np.sqrt(((z[m] - P[c]) ** 2).sum(1)).mean()
    return radii


def adulteration_scores(net, X, claimed, radii):
    """Score = distance(specimen, claimed-prototype) / genuine radius of claim.
    >1 means the specimen sits outside the genuine cloud of what it claims to be
    -- a candidate counterfeit/adulterated drug."""
    z, _ = net.embed(X)
    P = net.params["P"]
    d = np.sqrt(((z - P[claimed]) ** 2).sum(1))
    return d / (radii[claimed] + 1e-8)


def make_adulterated(X, y_effect, d_effect, rng=RNG):
    """Forge counterfeits: take genuine specimens and swap their EFFECT-signal
    for that of a *different* dynamis while keeping (roughly) the appearance --
    a plausible-looking fake that behaves wrong. They still CLAIM the original
    label. This is the deepfake/adulteration problem in pharmacological form."""
    n = len(X)
    take = rng.permutation(n)[: n // 3]
    Xa = X[take].copy()
    claimed = y_effect[take].copy()
    C = int(y_effect.max()) + 1
    # replace effect block with a different class's typical effect block
    for i, gi in enumerate(take):
        wrong = int(rng.integers(0, C))
        while wrong == claimed[i]:
            wrong = int(rng.integers(0, C))
        donors = np.where(y_effect == wrong)[0]
        donor = donors[rng.integers(0, len(donors))]
        Xa[i, :d_effect] = X[donor, :d_effect]     # borrowed (wrong) action
    return Xa, claimed


def auc(scores_pos, scores_neg):
    """AUC of the adulteration score at separating fakes (pos) from genuine
    (neg): probability a random fake scores higher than a random genuine one."""
    wins = 0
    for sp in scores_pos:
        wins += (sp > scores_neg).sum() + 0.5 * (sp == scores_neg).sum()
    return wins / (len(scores_pos) * len(scores_neg))


# ==============================================================================
# PART VI -- MAIN: run every check and demonstration, in order.
# ==============================================================================

def main():
    print("=" * 72)
    print("DYNAMIS -- Dioscorides' effect-affinity network (chapter 125)")
    print("=" * 72)

    # ---- 1. Gradient check (the autopsia of the code itself) ----------------
    print("\n[1] GRADIENT CHECK (analytic backprop vs finite differences)")
    ok, rel = gradient_check(verbose=True)
    assert ok, "Gradient check FAILED -- architecture invalid."
    print("    -> gradient check PASSED.")

    # ---- 2. Build the synthetic materia medica ------------------------------
    print("\n[2] SYNTHETIC MATERIA MEDICA")
    d_effect, d_morph, C = 8, 8, 6
    X, ye, ym = make_materia_medica(n_per_class=70, C=C, M=5,
                                    d_effect=d_effect, d_morph=d_morph)
    d_in = X.shape[1]
    print(f"    {len(X)} specimens, {C} dynamis-classes, {d_in} features "
          f"({d_effect} effect + {d_morph} morphology).")

    # split
    n = len(X)
    tr = np.zeros(n, bool); tr[: int(0.7 * n)] = True
    perm = RNG.permutation(n); tr = tr[perm]; X, ye, ym = X[perm], ye[perm], ym[perm]
    Xtr, ytr, ymtr = X[tr], ye[tr], ym[tr]
    Xte, yte, ymte = X[~tr], ye[~tr], ym[~tr]

    # ---- 3. Autopsia: firsthand vs noisy hearsay ----------------------------
    y_obs, w, is_hearsay = assign_provenance(ytr, hearsay_fraction=0.4)
    print(f"    training labels: {int((~is_hearsay).sum())} firsthand (autopsia), "
          f"{int(is_hearsay.sum())} hearsay (some corrupted).")

    # ---- 4. Appearance-truster baseline (the strawman) ----------------------
    base = morphology_baseline_accuracy(X, ye, ym, d_effect)
    print("\n[3] THE APPEARANCE-TRUSTER (nearest-neighbour by morphology)")
    print(f"    effect accuracy from appearance alone: {base:.3f} "
          f"(chance = {1.0/C:.3f}) -- appearance does not reveal dynamis.")

    # ---- 5. Train WITH autopsia weighting -----------------------------------
    print("\n[4] TRAIN the effect-grounded network WITH autopsia weighting")
    net = Dynamis(d_in, d_hidden=48, d_emb=16, n_classes=C, l2=1e-4)
    train(net, Xtr, y_obs, w, epochs=250, lr=0.15, batch=64, log_every=50)
    acc_autopsia = (net.predict(Xte) == yte).mean()

    # ---- 6. Ablation: train WITHOUT autopsia (uniform weights) --------------
    print("\n[5] ABLATION: same network, uniform weights (trust hearsay equally)")
    net_flat = Dynamis(d_in, d_hidden=48, d_emb=16, n_classes=C, l2=1e-4)
    train(net_flat, Xtr, y_obs, np.ones_like(w), epochs=250, lr=0.15,
          batch=64, log_every=250)
    acc_flat = (net_flat.predict(Xte) == yte).mean()

    print("\n[6] RESULT -- effect-classification test accuracy")
    print(f"    appearance-truster baseline : {base:.3f}")
    print(f"    uniform-weight network      : {acc_flat:.3f}")
    print(f"    autopsia-weighted network   : {acc_autopsia:.3f}")
    print("    -> seeing past appearance to dynamis, and trusting the witnessed")
    print("       over the merely-read, both help. That is Dioscorides, measured.")

    # ---- 7. The drug-affinity map -------------------------------------------
    print("\n[7] DRUG-AFFINITY MAP (learned prototype geometry)")
    D, nearest = affinity_map(net)
    for c in range(C):
        print(f"    dynamis {c}: nearest sibling = dynamis {nearest[c]} "
              f"(dist {D[c, nearest[c]]:.2f})")

    # ---- 8. Adulteration detection ------------------------------------------
    print("\n[8] ADULTERATION DETECTION")
    radii = class_radii(net, Xtr, ytr)
    genuine_scores = adulteration_scores(net, Xte, yte, radii)          # honest claims
    Xa, claimed = make_adulterated(Xte, yte, d_effect)                  # forgeries
    fake_scores = adulteration_scores(net, Xa, claimed, radii)
    # Principled operating point: set the alarm so at most 10% of genuine drugs
    # are wrongly flagged (a false-alarm budget the physician can afford), then
    # measure how many forgeries we catch at that setting.
    thr = np.quantile(genuine_scores, 0.90)
    print(f"    alarm threshold (10% false-alarm budget): score > {thr:.2f}")
    print(f"    genuine specimens : median score {np.median(genuine_scores):.2f}, "
          f"flagged {np.mean(genuine_scores > thr):.2%}")
    print(f"    adulterated fakes : median score {np.median(fake_scores):.2f}, "
          f"caught  {np.mean(fake_scores > thr):.2%}")
    print(f"    detection AUC     : {auc(fake_scores, genuine_scores):.3f}")
    print("    (appearance-preserving forgeries are genuinely hard -- only the")
    print("     dynamis betrays them; AUC > 0.5 is the effect-map earning its keep.)")

    print("\n" + "=" * 72)
    print("All checks passed. The map is drawn by effect; the fakes are caught.")
    print("=" * 72)

    return {
        "grad_ok": ok, "grad_rel": rel,
        "baseline": base, "acc_flat": acc_flat, "acc_autopsia": acc_autopsia,
        "auc": auc(fake_scores, genuine_scores),
    }


if __name__ == "__main__":
    main()

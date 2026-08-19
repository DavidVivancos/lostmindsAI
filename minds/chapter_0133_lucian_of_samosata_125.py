"""
================================================================================
Mind #0133 - Lucian of Samosata (c. 125 - after 180 CE)
The Kataskopos Engine: an incongruity-detecting, self-labeling cognition
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 7 Minds 121 - 140 Available on Amazon https://www.amazon.com/dp/B0HFN6GXMH
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 133: Lucian of Samosata (c. 125 - after 180 CE)
================================================================================   

WHY THIS ARCHITECTURE (and not a Transformer)
--------------------------------------------------------------------------------
Lucian did not build doctrine. He built a *comparison operator*. Read across his
surviving corpus and the same cognitive move recurs: he places two things side by
side - the grand claim and the paltry reality, the philosopher's pronouncement
and the philosopher's behaviour, the ornate register and the hollow substance -
and he reads the GAP between them. Laughter, for Lucian, is what fires when the
gap is large: when a thing presents itself as far more authoritative than it can
actually ground.

This file encodes that mind in three parts, each tied to a real Lucianic device:

  1. INCONGRUITY GATE  (the comic dialogue).  Lucian's signature form fills an
     elevated Platonic shell with vulgar, fantastical content; he auctions the
     great philosophers like slaves; he shows the Olympians as petty mortals.
     The satiric signal is a MULTIPLICATIVE "AND-NOT" gate:  fire when pretension
     is HIGH *and* grounding is LOW.  s = pretension * (1 - grounding).
     This interaction is not linearly separable, so a linear classifier cannot
     reproduce it - the gate is doing genuinely Lucianic work (proven below).

  2. KATASKOPOS OVERVIEW  (Icaromenippus / Charon: "the view from above").
     Menippus builds wings, flies to the Moon, looks down, and human striving
     shrinks to anthills; Greece measures four inches; the richest estates are no
     bigger than an Epicurean atom.  A claim's absurdity is only legible from
     OUTSIDE it.  We implement this as a batch-level baseline: each utterance's
     grounding is judged RELATIVE to the ambient grounding of the whole scene.
     Every judgement is therefore coupled to every other - estrangement made math.

  3. SELF-DECLARATION HEAD  (A True History: "the only true statement you are to
     expect - that I am a liar").  Lucian's paradox is that his lying is "far more
     honest than theirs" because he TAGS it.  A second head predicts the model's
     own reliability on each item - it labels when its own verdict is likely
     wrong, exactly Lucian's honest-liar move, and gives calibrated abstention.

Everything is pure NumPy, from scratch: explicit forward pass, hand-derived
backprop for every parameter, a finite-difference gradient check (mandatory),
a real training loop on a synthetic "pomp vs. substance" task, a linear baseline
that the gate provably beats, and behavioural self-tests.

Run:  python chapter_0133_lucian_of_samosata_125.py
================================================================================
"""

import numpy as np

RNG = np.random.default_rng(134)  # seed = the mind's number, for reproducibility


# ------------------------------------------------------------------------------
# 0.  Small numerical helpers
# ------------------------------------------------------------------------------
def sigmoid(z):
    # numerically stable logistic
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def bce(p, y, eps=1e-9):
    """Mean binary cross-entropy."""
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


# ------------------------------------------------------------------------------
# 1.  Synthetic corpus: utterances that are pompous, humble, or well-grounded
# ------------------------------------------------------------------------------
# Each "utterance" carries TWO feature groups, mirroring how Lucian reads a claim:
#   - REGISTER features  : how grand / authoritative it SOUNDS (ornate diction,
#                          appeals to gods & systems, sweeping cosmic scope).
#   - CONTENT features   : how much checkable substance it actually CARRIES.
#
# Ground truth (the "satiric target" = pompous nonsense) is the Lucianic gap:
#   pompous  <=> high pretension AND low grounding.
# A claim that is grand *and* substantiated is NOT a target; nor is a humble one.
# That AND-NOT region is a convex corner, provably not linearly separable from
# its complement - which is exactly why the multiplicative gate is required.

D_REG = 8   # register feature dimension
D_CON = 8   # content feature dimension

# The "true" reading directions are FIXED for the whole run, so that train and
# test share one labelling rule (a mind is one consistent reader, not many).
W_P = RNG.standard_normal(D_REG)          # what makes a claim SOUND grand
W_G = RNG.standard_normal(D_CON)          # what actually GROUNDS a claim
W_P /= np.linalg.norm(W_P)
W_G /= np.linalg.norm(W_G)


def latent_pretension(X_reg):
    return sigmoid(2.2 * (X_reg @ W_P))


def latent_grounding(X_con):
    return sigmoid(2.2 * (X_con @ W_G))


def make_corpus(n, flip=0.06):
    """Utterances labelled by the Lucianic gap, with a little label noise so the
    honest-liar head has genuinely uncertain cases to calibrate against."""
    X_reg = RNG.standard_normal((n, D_REG))
    X_con = RNG.standard_normal((n, D_CON))
    pretension = latent_pretension(X_reg)
    grounding = latent_grounding(X_con)
    # Lucianic label: pompous nonsense iff it puffs up but cannot pay.
    # This is one QUADRANT of the (pretension, grounding) plane - the AND-NOT
    # corner that no single hyperplane can carve from the other three quadrants.
    y = ((pretension > 0.5) & (grounding < 0.5)).astype(np.float64)
    if flip > 0:                                    # a few genuine ambiguities
        noise = RNG.random(n) < flip
        y = np.where(noise, 1.0 - y, y)
    return X_reg, X_con, y, pretension, grounding


# ------------------------------------------------------------------------------
# 2.  The Kataskopos Engine
# ------------------------------------------------------------------------------
class Kataskopos:
    """
    Forward:
        Hr = tanh(X_reg W_r + b_r)          register encoder
        p  = sigmoid(Hr u + c_u)            PRETENSION scalar in (0,1)
        Hc = tanh(X_con W_c + b_c)          content encoder
        gl = Hc v + c_v                     grounding logit
        gbar = mean(gl)                     <-- the "view from above": ambient grounding
        gr = sigmoid(gl - lam * gbar)       grounding RELATIVE to the whole scene
        s  = p * (1 - gr)                    INCONGRUITY GATE  (satiric signal)
        r  = sigmoid(w_conf . [p, gr, s] + b_conf)   SELF-DECLARATION (own reliability)
    Loss:
        L = BCE(s, y)  +  mu * BCE(r, t)     t = "was my verdict correct?" (detached)
    """

    def __init__(self, h=12, hc=6):
        s = 0.6
        self.P = {
            "W_r": RNG.standard_normal((D_REG, h)) * s / np.sqrt(D_REG),
            "b_r": np.zeros(h),
            "u":   RNG.standard_normal(h) * s / np.sqrt(h),
            "c_u": np.array(0.0),
            "W_c": RNG.standard_normal((D_CON, h)) * s / np.sqrt(D_CON),
            "b_c": np.zeros(h),
            "v":   RNG.standard_normal(h) * s / np.sqrt(h),
            "c_v": np.array(0.0),
            "lam": np.array(0.8),                     # weight of the view-from-above
            # self-declaration head: a small MLP over [p, gr, s], because judging
            # one's own reliability is a non-monotonic read (uncertain NEAR the
            # boundary) that a single linear layer cannot express.
            "Wc1": RNG.standard_normal((3, hc)) * s,
            "bc1": np.zeros(hc),
            "wc2": RNG.standard_normal(hc) * s,
            "bc2": np.array(0.0),
        }
        self.mu = 0.4   # weight of the self-declaration (honest-liar) objective
        # The view-from-above should be a STABLE vantage, not a per-batch jitter.
        # During training the baseline is the batch mean (real estrangement: each
        # judgement is coupled to the whole observed scene); we also carry an EMA
        # of it, and judge held-out utterances from that internalised vantage.
        self.gbar_ema = 0.0
        self.ema_momentum = 0.9
        self.ema_ready = False

    # -- forward pass; returns predictions + a cache for backprop ---------------
    def forward(self, X_reg, X_con, training=True, update_stats=True):
        P = self.P
        Zr = X_reg @ P["W_r"] + P["b_r"]
        Hr = np.tanh(Zr)
        p_logit = Hr @ P["u"] + P["c_u"]
        p = sigmoid(p_logit)

        Zc = X_con @ P["W_c"] + P["b_c"]
        Hc = np.tanh(Zc)
        g_logit = Hc @ P["v"] + P["c_v"]
        if training:
            gbar = np.mean(g_logit)                     # kataskopos baseline (scene)
            if update_stats:
                if self.ema_ready:
                    self.gbar_ema = (self.ema_momentum * self.gbar_ema
                                     + (1 - self.ema_momentum) * float(gbar))
                else:
                    self.gbar_ema = float(gbar); self.ema_ready = True
        else:
            gbar = self.gbar_ema                         # internalised vantage
        gr_logit = g_logit - P["lam"] * gbar
        gr = sigmoid(gr_logit)

        s = p * (1.0 - gr)                              # incongruity gate
        conf_in = np.stack([p, gr, s], axis=1)         # (n,3)
        Zconf = conf_in @ P["Wc1"] + P["bc1"]          # (n,hc)
        Aconf = np.tanh(Zconf)
        r = sigmoid(Aconf @ P["wc2"] + P["bc2"])       # self-declared reliability

        cache = dict(X_reg=X_reg, X_con=X_con, Zr=Zr, Hr=Hr, p=p,
                     Zc=Zc, Hc=Hc, g_logit=g_logit, gbar=gbar, gr=gr,
                     s=s, conf_in=conf_in, Zconf=Zconf, Aconf=Aconf, r=r)
        return s, r, cache

    # -- loss (t held fixed / detached, as Lucian's self-label is a tag) --------
    def loss(self, cache, y, t):
        return bce(cache["s"], y) + self.mu * bce(cache["r"], t)

    # -- analytic gradients -----------------------------------------------------
    def backward(self, cache, y, t, detach_conf=True):
        # detach_conf=True: the self-declaration head OBSERVES the verdict but
        # does not rewrite it - Lucian's tag annotates the claim, it does not
        # change the claim.  (With detach_conf=False the full coupled gradient is
        # returned, which is what the finite-difference check verifies.)
        P = self.P
        n = y.shape[0]
        eps = 1e-9
        s = np.clip(cache["s"], eps, 1 - eps)
        r = np.clip(cache["r"], eps, 1 - eps)
        p = cache["p"]; gr = cache["gr"]

        # d BCE(s,y)/ds  averaged
        dL_ds = (s - y) / (s * (1 - s)) / n
        # d BCE(r,t)/dr averaged, scaled by mu
        dL_dr = self.mu * (r - t) / (r * (1 - r)) / n

        # --- self-declaration head (2-layer MLP) ---
        g = {}
        dr_logit = dL_dr * r * (1 - r)             # (n,)  through output sigmoid
        g["wc2"] = cache["Aconf"].T @ dr_logit     # (hc,)
        g["bc2"] = np.sum(dr_logit)
        dAconf = np.outer(dr_logit, P["wc2"])      # (n,hc)
        dZconf = dAconf * (1 - cache["Aconf"] ** 2)  # tanh'
        g["Wc1"] = cache["conf_in"].T @ dZconf     # (3,hc)
        g["bc1"] = np.sum(dZconf, axis=0)
        dconf_in = dZconf @ P["Wc1"].T             # (n,3)
        if detach_conf:
            dp_from_conf = np.zeros(n)
            dgr_from_conf = np.zeros(n)
            ds_from_conf = np.zeros(n)
        else:
            dp_from_conf = dconf_in[:, 0]
            dgr_from_conf = dconf_in[:, 1]
            ds_from_conf = dconf_in[:, 2]

        # --- incongruity gate  s = p*(1-gr) ---
        dL_ds_total = dL_ds + ds_from_conf
        dp = dL_ds_total * (1.0 - gr) + dp_from_conf
        dgr = dL_ds_total * (-p) + dgr_from_conf

        # --- grounding branch: gr = sigmoid(g_logit - lam*gbar), gbar=mean(g_logit)
        dgr_logit = dgr * gr * (1 - gr)            # (n,)
        # g_logit appears directly AND inside gbar=mean(g_logit)
        # d gr_logit_i / d g_logit_j = delta_ij - lam/n
        # so grad wrt g_logit = dgr_logit - (lam/n)*sum(dgr_logit)
        sum_dgr_logit = np.sum(dgr_logit)
        dg_logit = dgr_logit - (P["lam"] / n) * sum_dgr_logit
        # lam gradient: d gr_logit_i/d lam = -gbar
        g["lam"] = np.sum(dgr_logit * (-cache["gbar"]))

        # g_logit = Hc v + c_v
        g["v"] = cache["Hc"].T @ dg_logit
        g["c_v"] = np.sum(dg_logit)
        dHc = np.outer(dg_logit, P["v"])           # (n,h)
        dZc = dHc * (1 - cache["Hc"] ** 2)          # tanh'
        g["W_c"] = cache["X_con"].T @ dZc
        g["b_c"] = np.sum(dZc, axis=0)

        # --- pretension branch: p = sigmoid(Hr u + c_u) ---
        dp_logit = dp * p * (1 - p)
        g["u"] = cache["Hr"].T @ dp_logit
        g["c_u"] = np.sum(dp_logit)
        dHr = np.outer(dp_logit, P["u"])
        dZr = dHr * (1 - cache["Hr"] ** 2)
        g["W_r"] = cache["X_reg"].T @ dZr
        g["b_r"] = np.sum(dZr, axis=0)

        # keep scalars as 0-d arrays to match parameter shapes
        for k in ("c_u", "c_v", "lam", "bc2"):
            g[k] = np.array(g[k])
        return g

    # -- reliability targets for the honest-liar head ---------------------------
    @staticmethod
    def reliability_target(pretension, grounding, margin=0.12):
        """The True-History tag as a teaching signal: an item is RELIABLE (t=1)
        when it sits far from BOTH labelling thresholds - clearly pompous or
        clearly sound - and UNRELIABLE (t=0) when it hovers near the boundary,
        where any reader, Lucian included, should decline to be certain.
        Detached from the gate, so it never collapses to 'always right'."""
        dist = np.minimum(np.abs(pretension - 0.5), np.abs(grounding - 0.5))
        return (dist > margin).astype(np.float64)


# ------------------------------------------------------------------------------
# 3.  Finite-difference gradient check  (MANDATORY)
# ------------------------------------------------------------------------------
def gradient_check():
    model = Kataskopos(h=6)
    Xr, Xc, y, P, G = make_corpus(24)
    s, r, cache = model.forward(Xr, Xc)
    t = model.reliability_target(P, G)      # fixed / detached tag (Lucian's move)

    # Verify the FULL coupled gradient (detach_conf=False) so every hand-derived
    # term is validated.  Training later zeroes an already-verified subset (the
    # conf->gate path), which needs no separate check.
    analytic = model.backward(cache, y, t, detach_conf=False)

    eps = 1e-6
    max_rel = 0.0
    worst = None
    for name, val in model.P.items():
        flat = np.atleast_1d(val).ravel()
        gflat = np.atleast_1d(analytic[name]).ravel()
        # sample a few coordinates per parameter to keep it fast
        idxs = range(flat.size) if flat.size <= 12 else RNG.choice(flat.size, 12, replace=False)
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps
            sp, rp, cp = model.forward(Xr, Xc)
            Lp = model.loss(cp, y, t)         # t held fixed - matches stop-gradient
            flat[i] = orig - eps
            sm, rm, cm = model.forward(Xr, Xc)
            Lm = model.loss(cm, y, t)
            flat[i] = orig
            num = (Lp - Lm) / (2 * eps)
            ana = gflat[i]
            denom = max(1e-8, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            if rel > max_rel:
                max_rel = rel
                worst = (name, i, num, ana)
    print(f"[gradient check] max relative error = {max_rel:.3e}")
    if worst:
        print(f"                worst @ {worst[0]}[{worst[1]}]: "
              f"numeric={worst[2]:+.6e}  analytic={worst[3]:+.6e}")
    assert max_rel < 1e-4, "Gradient check FAILED"
    print("                PASS  (analytic backprop matches finite differences)\n")


# ------------------------------------------------------------------------------
# 4.  Training loop  (full-batch Adam so the kataskopos overview sees the scene)
# ------------------------------------------------------------------------------
def train(model, Xr, Xc, y, lat_P, lat_G, epochs=1200, lr=0.05, wd=2e-4):
    P = model.P
    t = model.reliability_target(lat_P, lat_G)   # honest-liar target, fixed
    m = {k: np.zeros_like(np.atleast_1d(v), dtype=np.float64) for k, v in P.items()}
    v = {k: np.zeros_like(np.atleast_1d(v), dtype=np.float64) for k, v in P.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    decay = {"W_r", "W_c", "u", "v", "Wc1", "wc2"}  # decay weights, not biases/gates
    history = []
    for e in range(1, epochs + 1):
        s, r, cache = model.forward(Xr, Xc)
        # light coupling: the honest-liar objective may gently keep p, gr away
        # from full saturation so boundary information survives for the tag.
        grads = model.backward(cache, y, t, detach_conf=False)
        for k in P:
            gk = np.atleast_1d(grads[k]).astype(np.float64)
            if k in decay:
                gk = gk + wd * np.atleast_1d(P[k]).astype(np.float64)
            m[k] = b1 * m[k] + (1 - b1) * gk
            v[k] = b2 * v[k] + (1 - b2) * gk * gk
            mhat = m[k] / (1 - b1 ** e)
            vhat = v[k] / (1 - b2 ** e)
            step = lr * mhat / (np.sqrt(vhat) + eps)
            P[k] = P[k] - step.reshape(np.shape(P[k]))
        if e % 150 == 0 or e == 1:
            L = model.loss(cache, y, t)
            acc = np.mean(np.round(s) == y)
            r_acc = np.mean(np.round(r) == t)
            history.append((e, L, acc))
            print(f"  epoch {e:4d}   loss={L:.4f}   gate-acc={acc:5.1%}   "
                  f"honest-liar-acc={r_acc:5.1%}")
    return history


# ------------------------------------------------------------------------------
# 5.  Linear baseline: logistic regression on the raw concatenated features.
#     Lucian's point: the gap is not a line. This baseline should trail the gate.
# ------------------------------------------------------------------------------
def logistic_baseline(Xr, Xc, y, Xr_te, Xc_te, y_te, epochs=1500, lr=0.1):
    X = np.hstack([Xr, Xc]); Xte = np.hstack([Xr_te, Xc_te])
    w = np.zeros(X.shape[1]); b = 0.0
    n = X.shape[0]
    for _ in range(epochs):
        p = sigmoid(X @ w + b)
        gw = X.T @ (p - y) / n
        gb = np.sum(p - y) / n
        w -= lr * gw; b -= lr * gb
    acc = np.mean(np.round(sigmoid(Xte @ w + b)) == y_te)
    return acc


# ------------------------------------------------------------------------------
# 6.  Behavioural self-tests  (does it think like Lucian?)
# ------------------------------------------------------------------------------
def behavioural_tests(model):
    print("Behavioural self-tests (does the engine read the gap like Lucian?):")

    # Craft archetypes along the TRUE reading directions W_P (grandeur) and
    # W_G (grounding), so "grand" really means high latent pretension and
    # "hollow" really means low latent grounding.  We surround each archetype
    # with a realistic crowd so the kataskopos overview has a scene to judge from.
    def probe(reg_dir, con_dir, label):
        crowd_r, crowd_c, *_ = make_corpus(30, flip=0.0)
        Xr = np.vstack([reg_dir + 0.1 * RNG.standard_normal(D_REG) for _ in range(4)] + [crowd_r])
        Xc = np.vstack([con_dir + 0.1 * RNG.standard_normal(D_CON) for _ in range(4)] + [crowd_c])
        s, r, _ = model.forward(Xr, Xc, training=False)
        flag = np.mean(s[:4])
        print(f"   {label:38s}  satiric-flag={flag:.3f}   "
              f"self-declared-reliability={np.mean(r[:4]):.3f}")
        return flag

    grand = +3.0 * W_P     # high latent pretension
    small = -3.0 * W_P     # low  latent pretension  (humble)
    solid = +3.0 * W_G     # high latent grounding
    empty = -3.0 * W_G     # low  latent grounding   (hollow)
    grand_hollow = probe(grand, empty, "grand claim, hollow substance")
    grand_solid  = probe(grand, solid, "grand claim, real substance")
    humble_hollow = probe(small, empty, "humble claim, hollow substance")
    print("   (Lucian only mocks the FIRST: pretension unmatched by grounding.)")
    assert grand_hollow > 0.5, "satiric flag should fire on the gap"
    assert grand_solid < grand_hollow, "grounded grandeur must be spared"
    assert humble_hollow < grand_hollow, "humility must be spared"
    print("   self-tests PASS: the gate fires only where claim outruns content.\n")

    # Liar's-paradox test: a maximally uncertain verdict (s ~ 0.5) should be
    # self-declared as LOW reliability - the model tagging its own likely lie.
    Xr, Xc, y, _, _ = make_corpus(400)
    s, r, _ = model.forward(Xr, Xc, training=False)
    uncertain = np.abs(s - 0.5) < 0.1
    if np.any(uncertain):
        print(f"   Liar's-paradox check: on maximally uncertain verdicts, mean "
              f"self-declared reliability = {np.mean(r[uncertain]):.3f}")
        print(f"                          on confident verdicts,   mean "
              f"self-declared reliability = {np.mean(r[~uncertain]):.3f}")
    print()


# ------------------------------------------------------------------------------
# 7.  Main
# ------------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("Kataskopos Engine  -  the mind of Lucian of Samosata as running code")
    print("=" * 78 + "\n")

    print("[1] Gradient check")
    gradient_check()

    print("[2] Training on the pomp-vs-substance corpus")
    Xr, Xc, y, P_tr, G_tr = make_corpus(3000)
    Xr_te, Xc_te, y_te, P_te, G_te = make_corpus(1200)
    model = Kataskopos(h=10)
    train(model, Xr, Xc, y, P_tr, G_tr, epochs=1500, wd=8e-4)

    s_te, r_te, _ = model.forward(Xr_te, Xc_te, training=False)
    gate_acc = np.mean(np.round(s_te) == y_te)
    print(f"\n  Kataskopos gate  test-acc = {gate_acc:5.1%}")

    print("\n[3] Linear baseline (the gap is not a line)")
    lin_acc = logistic_baseline(Xr, Xc, y, Xr_te, Xc_te, y_te)
    print(f"  logistic regression test-acc = {lin_acc:5.1%}")
    print(f"  --> incongruity gate beats the linear reader by "
          f"{(gate_acc - lin_acc) * 100:+.1f} points "
          f"(the multiplicative AND-NOT is doing Lucianic work)\n")

    print("[4] " , end="")
    behavioural_tests(model)

    # Calibration of the honest-liar head: where the engine declares itself
    # reliable, is the gate actually more accurate?
    correct = (np.round(s_te) == y_te).astype(float)
    hi = r_te >= 0.5
    print("[5] Honest-liar calibration  (does 'I can be trusted here' come true?)")
    if np.any(hi) and np.any(~hi):
        print(f"  gate accuracy where it self-declares RELIABLE  ({np.mean(hi):4.0%} "
              f"of items) = {np.mean(correct[hi]):5.1%}")
        print(f"  gate accuracy where it self-declares UNRELIABLE ({np.mean(~hi):4.0%} "
              f"of items) = {np.mean(correct[~hi]):5.1%}")
        print("  (a well-formed tag: it is right more often exactly where it "
              "claims to be)\n")
        assert np.mean(correct[hi]) > np.mean(correct[~hi]), \
            "honest-liar head should be more accurate where it declares reliable"
        print("  self-test PASS: the self-declaration is honestly calibrated.\n")

    print("=" * 78)
    print("Done. The engine detects the gap, views from above, and tags its own"
          " lies.")
    print("=" * 78)


if __name__ == "__main__":
    main()

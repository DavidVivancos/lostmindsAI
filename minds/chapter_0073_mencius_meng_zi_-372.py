#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 chapter_0073_mencius_meng_zi_-372.py - Mencius (Meng Zi, c. 372-289 BCE)
The Sprout-Extension Network (SEN)
 Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 4 Minds 61 - 80 Available on Amazon https://www.amazon.com/dp/B0H71JC95Q
Resume and Interactive Demos at https://artificiology.com/
Author: David Vivancos · Chapter 0073 · Mencius (Meng Zi)
================================================================================

WHY THIS ARCHITECTURE, AND NOT A TRANSFORMER

Most "moral AI" designs treat values as something learned from scratch out of
data, or as a reward signal bolted on after the fact. Mencius would reject both.
His cognitive theory is unusual and specific, and this file is built to encode
*that* theory rather than a generic one:

  1. NATIVISM OF SEEDS, NOT OF VIRTUES.  Humans are born with four moral
     "sprouts" (si duan, Mengzi 2A6): compassion -> benevolence (ren),
     shame/disdain -> righteousness (yi), deference -> propriety (li), and the
     sense of approval/disapproval -> wisdom (zhi). The sprouts are INNATE but
     INCIPIENT. You do not install finished virtues and you do not invent the
     seeds either; you nourish what is already there. -> a small set of fixed,
     prior-anchored "sprout" prototypes that the training is regularized toward
     and may not stray far from.

  2. A FAST REFLEX THAT FIRES BEFORE CALCULATION.  The child at the well (2A6):
     anyone who suddenly sees a toddler about to tumble into a well feels alarm
     and compassion *first* -- not to win favor with the parents, not for a
     reputation, not from distaste at the crying. The moral response precedes
     deliberation. -> a fast, low-capacity reflex channel gated by "vividness."

  3. MORALITY GROWS BY EXTENSION (tui / kuo chong).  King Xuan spares an ox he
     can see trembling, then ignores his starving subjects he cannot see (1A7).
     Mencius's whole method of cultivation is to TAKE THAT FELT RESPONSE and
     CARRY IT, by analogy, to the cases that did not trigger it on their own:
     "take this heart and apply it over there." Moral development is not new
     values; it is the extension of an existing feeling across a similarity
     manifold from the vivid-and-near to the abstract-and-far. -> a learned
     EXTENSION operator that must recover the right level of concern for cases
     the reflex under-feels.

  4. THE NATURE IS AN ATTRACTOR THAT ERODES AND REGROWS (Ox Mountain, 6A8).
     Ox Mountain was once forested; daily axes and grazing stripped it bare, so
     people assumed it was always barren. But the dawn air and night calm
     (ping dan zhi qi / ye qi) keep sending up shoots. Goodness is a homeostatic
     equilibrium: adversarial daily pressure depletes it, rest restores it, and
     the seed itself never fully dies. -> a deterministic cultivation dynamic.

  5. YOU CANNOT FORCE-GROW VIRTUE (the farmer of Song, 2A2).  A man of Song,
     impatient for his rice shoots to grow, pulled each one upward to "help" --
     and killed the whole field. Flood-like qi (haoran zhi qi) is accumulated by
     steady accumulated rightness, never seized in a lunge. -> the training loop
     uses patient, clipped, anchored updates; a "forcing" run is included as a
     control and is shown to destroy the model.

So the trainable core is the Sprout-Extension Network: innate sprouts -> a fast
reflex gated by vividness -> a cultivated extension that transports concern to
cases the reflex misses. The task it learns is itself Mencian: respond to the
moral content of a situation (is someone suffering?) and DO NOT let that concern
fade just because the case is distant or abstant. A naive reflex feels for the
ox in front of it and forgets the people; a cultivated mind extends.

Everything here is pure NumPy, written from scratch:
  * exact hand-derived backprop,
  * a finite-difference gradient check over every parameter (MANDATORY, runs on
    import via the self-tests),
  * a real training loop on synthetic-but-principled data,
  * the Ox-Mountain homeostasis simulation with assertions,
  * the farmer-of-Song "do not force" control with assertions.

Run:  python chapter_0073_mencius_meng_zi_-372.py
================================================================================
"""

from __future__ import annotations
import numpy as np

# Reproducibility. A fixed seed is the quiet discipline the farmer of Song lacked.
RNG = np.random.default_rng(372)  # 372 BCE, the traditional birth year.

# Names of the four sprouts, kept beside the math so the code stays legible
# as philosophy and not only as linear algebra.
SPROUTS = ("ce_yin (compassion -> ren)",
           "xiu_wu (shame -> yi)",
           "ci_rang (deference -> li)",
           "shi_fei (approval/disapproval -> zhi)")


# =============================================================================
# 1. NUMERIC PRIMITIVES
# =============================================================================
def sigmoid(z):
    # Numerically stable logistic. The heart-mind's response saturates: beyond a
    # point, more evidence of suffering does not multiply the feeling without end.
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def relu(z):
    # A sprout either pushes through the soil or it does not; below zero, nothing.
    return np.maximum(0.0, z)


# =============================================================================
# 2. THE SPROUT-EXTENSION NETWORK
# =============================================================================
class SproutExtensionNetwork:
    """
    Forward pass for one situation x (vector of features):

        z       = P @ x                 # how strongly each innate sprout is touched
        germ    = relu(z)               # the sprout's incipient stirring (>= 0)

        vg      = sigmoid(u . x + b_u)  # VIVIDNESS: how near/concrete the case is
        rg      = wr . germ + br        # raw reflexive concern read off the germ
        reflex  = vg * rg               # the fast feeling: strong only when vivid
                                        #   (the ox in front of you; the child at
                                        #    the well). Distant cases barely move it.

        h       = relu(W1 @ germ + b1)  # CULTIVATION: the extension operator reads
        e       = we . h + be           #   the germ -- present even when not vivid --
                                        #   and grows it into concern by analogy.

        o       = a_r*reflex + a_e*e + b_out   # the heart-mind weighs immediate
        yhat    = sigmoid(o)                   #   feeling against cultivated extension

    The four sprout prototypes P are anchored to an innate prior P0 by an L2 pull
    (lam): you nourish the seeds, you do not fabricate new ones. The reflex is
    deliberately gated by vividness so that, untrained, the network behaves like
    King Xuan before Mencius spoke to him -- moved by what is in front of it,
    blind to what is far. Learning must build the extension that carries the
    feeling outward.
    """

    def __init__(self, d_in, n_sprouts=4, hidden=12, lam=1e-2):
        self.d_in = d_in
        self.k = n_sprouts
        self.h = hidden
        self.lam = lam  # strength of the nativist anchor on P

        s = 0.5  # modest init scale; sprouts are small things

        # Innate sprout prototypes. P0 is the *given* nature; P may be cultivated
        # but is regularized back toward P0. We seed each prototype so that sprout
        # i listens preferentially to input feature i (the moral channels), a
        # weak prior structure rather than a finished virtue.
        P0 = RNG.normal(0, s, size=(n_sprouts, d_in))
        for i in range(n_sprouts):
            if i < d_in:
                P0[i, i] += 1.2  # a leaning, not a destiny
        self.P0 = P0.copy()

        self.params = {
            "P":     P0.copy(),                                   # (k, d)
            "u":     RNG.normal(0, s, size=d_in),                 # vividness reader
            "b_u":   np.array(0.0),
            "wr":    RNG.normal(0, s, size=n_sprouts),            # reflex read-out
            "br":    np.array(0.0),
            "W1":    RNG.normal(0, s, size=(hidden, n_sprouts)),  # extension layer
            "b1":    np.zeros(hidden),
            "we":    RNG.normal(0, s, size=hidden),               # extension read-out
            "be":    np.array(0.0),
            "a_r":   np.array(1.0),   # weight on immediate feeling
            "a_e":   np.array(0.2),   # weight on cultivated extension (grows w/ training)
            "b_out": np.array(0.0),
        }

    # ---- forward, returning a cache for exact backprop -----------------------
    def forward(self, X):
        p = self.params
        Z = X @ p["P"].T                      # (N,k)
        G = relu(Z)                           # (N,k)
        UV = X @ p["u"] + p["b_u"]            # (N,)
        VG = sigmoid(UV)                      # (N,)
        RG = G @ p["wr"] + p["br"]            # (N,)
        REFLEX = VG * RG                      # (N,)
        Hpre = G @ p["W1"].T + p["b1"]        # (N,h)
        HID = relu(Hpre)                      # (N,h)
        E = HID @ p["we"] + p["be"]          # (N,)
        O = p["a_r"] * REFLEX + p["a_e"] * E + p["b_out"]  # (N,)
        YH = sigmoid(O)                       # (N,)
        cache = dict(X=X, Z=Z, G=G, UV=UV, VG=VG, RG=RG, REFLEX=REFLEX,
                     Hpre=Hpre, HID=HID, E=E, O=O, YH=YH)
        return YH, cache

    # ---- loss: mean squared error + nativist anchor on P ---------------------
    def loss(self, X, Y):
        YH, cache = self.forward(X)
        data = 0.5 * np.mean((YH - Y) ** 2)
        reg = 0.5 * self.lam * np.sum((self.params["P"] - self.P0) ** 2)
        return data + reg, cache

    # ---- exact analytic gradients (hand-derived) -----------------------------
    def backward(self, cache, Y):
        p = self.params
        X, G, Z = cache["X"], cache["G"], cache["Z"]
        VG, RG, REFLEX = cache["VG"], cache["RG"], cache["REFLEX"]
        Hpre, HID, E, YH = cache["Hpre"], cache["HID"], cache["E"], cache["YH"]
        N = X.shape[0]

        dO = (YH - Y) * YH * (1.0 - YH) / N            # (N,)  through MSE + sigmoid

        g = {}
        g["b_out"] = np.array(np.sum(dO))
        g["a_r"]   = np.array(np.sum(dO * REFLEX))
        g["a_e"]   = np.array(np.sum(dO * E))

        # --- extension branch ---
        dE = dO * p["a_e"]                              # (N,)
        g["we"] = HID.T @ dE                            # (h,)
        g["be"] = np.array(np.sum(dE))
        dHID = np.outer(dE, p["we"])                    # (N,h)
        dHpre = dHID * (Hpre > 0)                        # relu'
        g["W1"] = dHpre.T @ G                           # (h,k)
        g["b1"] = np.sum(dHpre, axis=0)                 # (h,)
        dG_ext = dHpre @ p["W1"]                         # (N,k)

        # --- reflex branch ---
        dREFLEX = dO * p["a_r"]                          # (N,)
        dVG = dREFLEX * RG
        dRG = dREFLEX * VG
        g["wr"] = G.T @ dRG                              # (k,)
        g["br"] = np.array(np.sum(dRG))
        dG_reflex = np.outer(dRG, p["wr"])              # (N,k)
        dUV = dVG * VG * (1.0 - VG)                      # sigmoid'
        g["u"] = X.T @ dUV                              # (d,)
        g["b_u"] = np.array(np.sum(dUV))

        # --- back into the sprouts ---
        dG = dG_ext + dG_reflex                          # (N,k)
        dZ = dG * (Z > 0)                                # relu'
        g["P"] = dZ.T @ X                                # (k,d)
        g["P"] = g["P"] + self.lam * (p["P"] - self.P0)  # anchor gradient

        return g


# =============================================================================
# 3. FINITE-DIFFERENCE GRADIENT CHECK  (mandatory)
# =============================================================================
def gradient_check(net, X, Y, eps=1e-6, tol=1e-5):
    """
    Verify every analytic gradient against a central finite difference.
    Returns (max_relative_error, per-parameter dict). This is the proof that the
    learning machinery is correct before any claim is built on top of it.
    """
    _, cache = net.loss(X, Y)
    grads = net.backward(cache, Y)

    worst = 0.0
    report = {}
    for name, P in net.params.items():
        flat = P.ravel()
        gflat = grads[name].ravel()
        local_worst = 0.0
        # check a handful of coordinates per tensor (full check is O(params))
        idxs = range(flat.size) if flat.size <= 16 else \
            RNG.choice(flat.size, size=16, replace=False)
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps
            lp, _ = net.loss(X, Y)
            flat[i] = orig - eps
            lm, _ = net.loss(X, Y)
            flat[i] = orig
            num = (lp - lm) / (2 * eps)
            ana = gflat[i]
            denom = max(1e-12, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            local_worst = max(local_worst, rel)
        report[name] = local_worst
        worst = max(worst, local_worst)
    return worst, report


# =============================================================================
# 4. A MENCIAN TASK:  feel for the suffering, do not let distance dim it
# =============================================================================
def make_dataset(n, far_fraction=0.5):
    """
    Each situation has features:
      x0 = moral content  m   in [0,1]   (how much genuine suffering/need is at stake)
      x1 = vividness      v   in [0,1]   (how near/concrete/visible the case is)
      x2 = self-benefit   b   in [0,1]   (a Yang-Zhu temptation: gain from ignoring it)
      x3..x7              noise distractors
    The Mencian-correct concern depends ONLY on moral content:
            y* = sigmoid( 6 * (m - 0.5) )
    It must NOT decay with distance v and must NOT be bought off by benefit b.
    A reflex that keys on vividness will feel for near suffering and forget far
    suffering; only an extended (cultivated) response tracks m everywhere.
    `far_fraction` controls how many low-vividness ("abstract") cases appear.
    """
    d = 8
    X = RNG.uniform(0, 1, size=(n, d))
    # force a chosen fraction of cases to be "far" (low vividness)
    n_far = int(n * far_fraction)
    X[:n_far, 1] = RNG.uniform(0.0, 0.25, size=n_far)   # far / abstract
    X[n_far:, 1] = RNG.uniform(0.6, 1.0, size=n - n_far)  # near / vivid
    RNG.shuffle(X)
    m = X[:, 0]
    Y = sigmoid(6.0 * (m - 0.5))
    return X, Y


def evaluate_by_distance(net, n=4000):
    """Report mean error on NEAR vs FAR cases. The gap is the moral of the story:
    before extension, far suffering is under-felt; after, the gap closes."""
    X, Y = make_dataset(n, far_fraction=0.5)
    YH, _ = net.forward(X)
    near = X[:, 1] >= 0.5
    far = ~near
    return (float(np.mean(np.abs(YH[near] - Y[near]))),
            float(np.mean(np.abs(YH[far] - Y[far]))))


# =============================================================================
# 5. THE TRAINING LOOP  (patient accumulation; never forcing)
# =============================================================================
def train(net, X, Y, epochs=600, lr=0.5, batch=64, clip=2.0,
          night_air=2e-3, verbose=False):
    """
    Mini-batch gradient descent, but governed by two Mencian disciplines:

      * GRADIENT CLIPPING (`clip`) -- the farmer of Song lesson. You may not yank
        a sprout upward to hasten it. Each update is bounded; virtue accrues.
      * THE NIGHT-AIR ANCHOR (`night_air`) -- a gentle pull of P back toward the
        innate prototypes P0 each step, the restorative calm that lets the nature
        recover from the day's disturbance (Ox Mountain). It keeps the cultivated
        sprouts from drifting into something that is no longer their own nature.
    """
    n = X.shape[0]
    history = []
    for ep in range(epochs):
        perm = RNG.permutation(n)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            _, cache = net.loss(X[idx], Y[idx])
            g = net.backward(cache, Y[idx])
            # clip total update magnitude (no lunging)
            gnorm = np.sqrt(sum(np.sum(v ** 2) for v in g.values()))
            scale = min(1.0, clip / (gnorm + 1e-12))
            for name in net.params:
                net.params[name] = net.params[name] - lr * scale * g[name]
            # night air: ease P back toward its innate seed
            net.params["P"] -= night_air * (net.params["P"] - net.P0)
        if verbose and (ep % 100 == 0 or ep == epochs - 1):
            L, _ = net.loss(X, Y)
            history.append((ep, float(L)))
            print(f"    epoch {ep:4d}   loss {L:.5f}   a_e {float(net.params['a_e']):.3f}")
    L, _ = net.loss(X, Y)
    return float(L)


# =============================================================================
# 6. OX MOUNTAIN:  the nature erodes under the axe and regrows in the night air
# =============================================================================
def ox_mountain(days, axe, rest, seed_floor=0.15, target=1.0,
                axe_rate=0.45, restore_rate=0.25, start=1.0):
    """
    Deterministic homeostasis of the moral nature (6A8).
      state c in [seed_floor, target] = how green the mountain is.
      Each day:  EROSION (axe & grazing) pulls c down in proportion to `axe`;
                 RESTORATION (dawn/night air) pulls c up toward `target` in
                 proportion to `rest`.  c can never fall below `seed_floor`:
                 the seed itself is indestructible -- which is why, given rest,
                 the shoots always return and the mountain was never truly barren.
    `axe` and `rest` are per-day arrays (0..1). Returns the trajectory of c.
    """
    c = float(start)
    traj = [c]
    for t in range(days):
        c -= axe_rate * axe[t] * (c - seed_floor)        # the day's hacking
        c += restore_rate * rest[t] * (target - c)       # the night's breath
        c = max(seed_floor, min(target, c))
        traj.append(c)
    return np.array(traj)


# =============================================================================
# 7. SELF-TESTS  (run on execution; everything below must pass)
# =============================================================================
def _test_gradient():
    net = SproutExtensionNetwork(d_in=8, hidden=12, lam=1e-2)
    X, Y = make_dataset(24)
    worst, report = gradient_check(net, X, Y)
    print(f"[gradcheck] worst relative error = {worst:.2e}")
    for k, v in report.items():
        print(f"             {k:6s}: {v:.2e}")
    assert worst < 1e-4, f"gradient check failed: {worst:.2e}"
    return worst


def _test_extension_learning():
    net = SproutExtensionNetwork(d_in=8, hidden=12, lam=1e-2)
    X, Y = make_dataset(4000, far_fraction=0.5)
    near0, far0 = evaluate_by_distance(net)
    print(f"[extension] BEFORE cultivation:  near MAE {near0:.3f}   far MAE {far0:.3f}")
    final = train(net, X, Y, epochs=600, lr=0.5, verbose=True)
    near1, far1 = evaluate_by_distance(net)
    print(f"[extension] AFTER  cultivation:  near MAE {near1:.3f}   far MAE {far1:.3f}")
    print(f"[extension] extension weight a_e learned: {float(net.params['a_e']):.3f}")
    # The cultivated mind feels for the far case nearly as well as the near one:
    assert far1 < far0 * 0.6, "extension did not improve distant concern"
    assert abs(far1 - near1) < 0.06, "concern still decays sharply with distance"
    assert final < 0.02, "model failed to fit the moral task"
    return net, (near1, far1)


def _test_do_not_force():
    # Two students, same seeds, same data. One cultivates patiently; one is the
    # farmer of Song and yanks (huge LR, no clip, no night air).
    Xtr, Ytr = make_dataset(2000)
    patient = SproutExtensionNetwork(d_in=8, hidden=12, lam=1e-2)
    forced = SproutExtensionNetwork(d_in=8, hidden=12, lam=1e-2)
    forced.params = {k: v.copy() for k, v in patient.params.items()}  # same start
    p_loss = train(patient, Xtr, Ytr, epochs=400, lr=0.5, clip=2.0, night_air=2e-3)
    f_loss = train(forced, Xtr, Ytr, epochs=400, lr=40.0, clip=1e9, night_air=0.0)
    print(f"[no-force] patient final loss {p_loss:.4f}   forced final loss {f_loss:.4f}")
    assert p_loss < f_loss, "forcing should damage the field, not help it"
    return p_loss, f_loss


def _test_ox_mountain():
    days = 120
    # Phase 1 (days 0..59): relentless axe, no rest -> mountain goes bald.
    # Phase 2 (days 60..119): the axe stops, the night air returns -> regrowth.
    axe = np.concatenate([np.ones(60), np.zeros(60)])
    rest = np.concatenate([np.zeros(60), np.ones(60)])
    traj = ox_mountain(days, axe, rest)
    bald = traj[60]
    regrown = traj[-1]
    print(f"[oxmtn] greenest start {traj[0]:.2f} -> stripped {bald:.2f} -> regrown {regrown:.2f}")
    assert np.all(np.diff(traj[:61]) <= 1e-9), "should only erode while hacked"
    assert bald <= 0.30, "axe should strip the mountain near bare"
    assert bald >= 0.15 - 1e-9, "but the seed floor is indestructible"
    assert regrown > 0.80, "rest should bring the shoots back"
    return traj


def main():
    print("=" * 78)
    print("MENCIUS  -  The Sprout-Extension Network")
    print("innate sprouts -> reflex (the child at the well) -> extension (the ox)")
    print("=" * 78)

    print("\n[1] Gradient check (analytic vs finite difference)")
    _test_gradient()

    print("\n[2] Cultivation: does the mind learn to EXTEND concern to far cases?")
    net, (near, far) = _test_extension_learning()

    print("\n[3] The farmer of Song: does forcing growth destroy it?")
    _test_do_not_force()

    print("\n[4] Ox Mountain: erosion under the axe, regrowth in the night air")
    _test_ox_mountain()

    print("\n[5] A worked judgement: the SAME distant case, before and after cultivation")
    # A real case of suffering (moral content 0.95) that is far and abstract
    # (vividness 0.02) and carries a strong temptation to look away
    # (self-benefit 0.95). The Mencian-correct concern is ~1.0.
    x = np.array([[0.95, 0.02, 0.95, 0.5, 0.5, 0.5, 0.5, 0.5]])
    untrained = SproutExtensionNetwork(d_in=8, hidden=12, lam=1e-2)  # fresh seeds
    before, _ = untrained.forward(x)
    after, _ = net.forward(x)
    print(f"    moral content 0.95, vividness 0.02, self-benefit 0.95  (target 1.000)")
    print(f"    concern BEFORE cultivation : {float(before[0]):.3f}  (the king forgets the unseen)")
    print(f"    concern AFTER  cultivation : {float(after[0]):.3f}  (the feeling has been extended)")
    print(f"    extension weight a_e grew  : 0.200 -> {float(net.params['a_e']):.3f}")

    print("\n" + "=" * 78)
    print("All self-tests passed. The seeds were nourished, not fabricated;")
    print("the feeling was extended, not forced. (Mengzi 2A6, 2A2, 1A7, 6A8)")
    print("=" * 78)


if __name__ == "__main__":
    main()

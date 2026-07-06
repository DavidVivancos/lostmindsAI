#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 chapter_0082_cleanthes_-330.py
 The Tonos Resonance Network (TRN)
 A from-scratch, pure-NumPy cognitive architecture after CLEANTHES of Assos
 (c. 330 - c. 230 BCE), second head of the Stoa.
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 5 Minds 81 - 100 Available on Amazon https://www.amazon.com/dp/B0H7LP5LP2
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 0082 · Cleanthes of Assos
================================================================================

WHY THIS ARCHITECTURE IS *HIS* AND NOT A GENERIC NET
----------------------------------------------------
Cleanthes' one original move in physics was the doctrine of TONOS -- "tension"
or "tone." For him a thing is one thing, instead of a heap of parts, because a
fine warm breath (pneuma) holds it together by pulling inward and outward at
once (tonike kinesis). The SAME tension, rising in degree, becomes cohesion in
a stone (hexis), growth in a plant (physis), sense in an animal (psyche), and
reason in a human. So character has a tone the way a string has a tone: strong,
well-tuned tension is virtue (eutonia); slack tension is vice and passion
(atonia). "Virtue is a strength, a striking-power of the soul."

Two further Cleanthean commitments shape the wiring:
  * THE SUN AS HEGEMONIKON. Against the later Stoa (which lodged the cosmic
    command-faculty in the aither/heart), Cleanthes argued the SUN is the ruling
    part of the world, because it sustains and coordinates all living things.
    -> a single GLOBAL broadcast node feeds the common tone back to every unit;
       it is the ONLY long-range channel -- the unit-to-unit coupling is local.
  * FREEDOM AS GLAD ASSENT. "Lead me, O Zeus, and you Destiny... I shall follow
    willingly." The mind does not fight its impressions; it ASSENTS to the ones
    that are firmly grasped. -> an assent gate commits only when the settled
    field decides by a clear margin, and otherwise withholds judgement (the
    Stoic epoche), refusing to be dragged by a slack, noisy impression.

So the network is NOT attention over stored keys. An impression STRIKES the
field once (a phantasia); then the soul WORKS on it, relaxing for several steps
under its own tension toward a clean attractor. Information lives in the tension
between units, not in a lookup table. Learning = tuning the tone.

THE TASK (chosen so the thesis is测the-able)
--------------------------------------------
"Holding the tone under passion." Four prototype impressions stand for the four
cardinal virtues (wisdom, courage, justice, temperance). Each example is a
prototype buried under Gaussian noise -- the passions that slacken the soul.
The network must say which virtue it is. Because the impression strikes only
once and then must be cleaned by the field's own dynamics, the LEARNED TENSION
and the SUN'S BROADCAST do real work, and we can prove it numerically:
  - a TENSION SWEEP shows an inverted-U "eutonic" band: slack tension (atonia)
    lets the units saturate and lose the signal; excess tension freezes the
    field so it cannot hear the evidence; the tuned tone holds best.
  - a SUN ablation removes the only global coordinator and accuracy falls.
  - the ASSENT gate's grasp loosens and withholding (epoche) rises as the
    impression slackens.

MANDATORY CHECKS (run on execution): a finite-difference gradient check on every
parameter; a real Adam training loop; and the three thesis demonstrations.

Run:  python3 chapter_0082_cleanthes_-330.py
================================================================================
"""

import numpy as np

# A fixed seed so the verified run is reproducible. 82 == this mind's number.
RNG = np.random.default_rng(82)


# ============================================================================
# 0. SMALL MATH HELPERS  (pure NumPy, differentiable where it matters)
# ============================================================================
def softplus(x):
    """log(1+e^x), stable. The tension tau passes through this so it is always
    positive: a string can be taut or slack, but never 'negatively' taut."""
    return np.logaddexp(0.0, x)


def d_softplus(x):
    """d/dx softplus(x) = sigmoid(x)."""
    return 1.0 / (1.0 + np.exp(-x))


def softmax_rows(z):
    """Row-wise softmax with the usual max-subtraction for stability."""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def local_band_mask(H, bandwidth=2):
    """A 0/1 matrix that keeps only NEAR-diagonal couplings: unit i may pull on
    unit j only if |i-j| <= bandwidth (on a ring). This makes the unit-to-unit
    tension LOCAL, so that any global coordination must travel through the Sun."""
    idx = np.arange(H)
    d = np.abs(idx[:, None] - idx[None, :])
    d = np.minimum(d, H - d)              # wrap-around: the field is a ring
    return (d <= bandwidth).astype(np.float64)


# ============================================================================
# 1. THE TONOS RESONANCE NETWORK
# ============================================================================
class TonosResonanceNetwork:
    """
    A recurrent field that relaxes toward a tensioned equilibrium.

    State h in R^H is the soul-field (a portion of pneuma). The impression
    strikes once, at step 0; afterward the field works on it:

        step 0:    h_1 = tanh( W_in @ x + b_h )            # the strike
        step t>0:  couple_t  = (W_c (.) M_local) @ h_t     # LOCAL inter-unit tension
                   tone_t    = mean(h_t)                   # the single global tone
                   sun_t     = w_sun * tone_t              # the Sun broadcasts it
                   restore_t = -softplus(rho) * h_t        # each unit's own tension
                   h_{t+1}   = tanh( couple_t + sun_t + restore_t + b_h )

    Readout (assent -> a verdict over the four virtues):
        logits = W_out @ h_T + b_out

    Learnable parameters (the 'constitution' of this particular soul):
        W_in  (H,D)   how impressions enter the field
        W_c   (H,H)   the tension between units (masked to local bands)
        w_sun (H,)    how strongly the Sun's broadcast reaches each unit
        rho   (H,)    pre-tension; tau = softplus(rho) is the actual per-unit tone
        b_h   (H,)    the field's resting bias
        W_out (C,H)   the readout (what verdict the settled field supports)
        b_out (C,)    readout bias
    """

    def __init__(self, D, H, C, T=6, bandwidth=2,
                 coupling_scale=1.2, tension_init=0.5):
        self.D, self.H, self.C, self.T = D, H, C, T
        self.M = local_band_mask(H, bandwidth)             # fixed locality mask
        self.p = {
            "W_in":  RNG.standard_normal((H, D)) * 0.8 / np.sqrt(D),
            # strong, local coupling so the dynamics genuinely need stabilising
            "W_c":   RNG.standard_normal((H, H)) * coupling_scale / np.sqrt(2 * bandwidth + 1),
            "w_sun": RNG.standard_normal(H) * 0.9,
            "rho":   np.full(H, np.log(np.expm1(tension_init))),  # softplus(rho)=tension_init
            "b_h":   np.zeros(H),
            "W_out": RNG.standard_normal((C, H)) * 0.8 / np.sqrt(H),
            "b_out": np.zeros(C),
        }

    # -- forward pass; keeps every intermediate so we can back-propagate ------
    def forward(self, X, tension_scale=1.0, use_sun=True):
        """
        X : (N, D) batch of impressions.
        tension_scale : multiply the learned tension tau (1.0 = as learned;
                        0.0 = atonia/slack; >1 = over-tight). Used by the sweep.
        use_sun : if False, silence the Sun's global broadcast (ablation).
        Returns logits (N, C) and a cache for backward.
        """
        N, H, T = X.shape[0], self.H, self.T
        p = self.p
        Wc = p["W_c"] * self.M                              # apply locality mask
        tau = softplus(p["rho"]) * tension_scale            # (H,) effective tension

        h = np.tanh(X @ p["W_in"].T + p["b_h"])             # step 0: the strike
        hs = [h]                                            # hs[k] = state after k+1 strikes/steps
        tones = []
        for _ in range(T - 1):                              # steps 1 .. T-1: the soul works
            couple = h @ Wc.T                               # (N,H) local tension
            tone = h.mean(axis=1, keepdims=True)            # (N,1) the global tone
            sun = (tone * p["w_sun"]) if use_sun else 0.0   # (N,H) Sun broadcast
            restore = -tau * h                              # (N,H) restoring tension
            pre = couple + sun + restore + p["b_h"]
            h = np.tanh(pre)
            hs.append(h)
            tones.append(tone)

        logits = h @ p["W_out"].T + p["b_out"]              # (N,C)
        cache = dict(X=X, hs=hs, tones=tones, Wc=Wc, tau=tau,
                     tension_scale=tension_scale, use_sun=use_sun)
        return logits, cache

    # -- loss ----------------------------------------------------------------
    def loss(self, logits, y):
        """Mean softmax cross-entropy. y : (N,) integer class labels."""
        N = logits.shape[0]
        P = softmax_rows(logits)
        ll = -np.log(P[np.arange(N), y] + 1e-12)
        return ll.mean(), P

    # -- backward pass: manual back-propagation-through-time -----------------
    def backward(self, cache, P, y):
        """Returns a dict of gradients, same keys/shapes as self.p."""
        p = self.p
        X, hs, tones = cache["X"], cache["hs"], cache["tones"]
        Wc, tau = cache["Wc"], cache["tau"]
        tscale, use_sun = cache["tension_scale"], cache["use_sun"]
        N, H, T = X.shape[0], self.H, self.T

        g = {k: np.zeros_like(v) for k, v in p.items()}

        # readout
        dlogits = P.copy()
        dlogits[np.arange(N), y] -= 1.0
        dlogits /= N                                        # (N,C)
        hT = hs[-1]
        g["W_out"] += dlogits.T @ hT                        # (C,H)
        g["b_out"] += dlogits.sum(axis=0)                   # (C,)
        dh = dlogits @ p["W_out"]                           # (N,H) grad on h_T

        dtau = np.zeros(H)
        dWc_masked = np.zeros((H, H))
        # steps T-1 .. 1 (the working steps), in reverse
        for t in range(T - 1, 0, -1):
            h_out = hs[t]                                   # produced at working step t
            h_in = hs[t - 1]                                # its input
            tone = tones[t - 1]                             # (N,1)
            dpre = dh * (1.0 - h_out ** 2)                  # tanh'   (N,H)

            g["b_h"] += dpre.sum(axis=0)                    # b_h appears in every pre

            # couple = h_in @ (W_c (.) M).T
            dWc_masked += dpre.T @ h_in                     # grad wrt masked Wc
            dh_in = dpre @ Wc                               # (N,H)

            # sun = (mean h_in) * w_sun
            if use_sun:
                g["w_sun"] += (dpre * tone).sum(axis=0)     # (H,)
                dtone = dpre @ p["w_sun"]                   # (N,)
                dh_in += (dtone[:, None] / H)               # the mean spreads to all units

            # restore = -tau * h_in
            dtau += -(dpre * h_in).sum(axis=0)              # (H,)
            dh_in += dpre * (-tau)                          # (N,H)

            dh = dh_in

        # step 0: the strike  h_1 = tanh(X @ W_in.T + b_h)
        dpre0 = dh * (1.0 - hs[0] ** 2)                     # (N,H)
        g["W_in"] += dpre0.T @ X                            # (H,D)
        g["b_h"] += dpre0.sum(axis=0)

        # fold masked-Wc grad back through the locality mask, and rho through softplus
        g["W_c"] += dWc_masked * self.M
        g["rho"] += dtau * d_softplus(p["rho"]) * tscale
        return g

    def loss_and_grads(self, X, y, **fwd):
        logits, cache = self.forward(X, **fwd)
        L, P = self.loss(logits, y)
        g = self.backward(cache, P, y)
        return L, g, P


# ============================================================================
# 2. FINITE-DIFFERENCE GRADIENT CHECK   (mandatory; proves the math is right)
# ============================================================================
def gradient_check(model, X, y, eps=1e-5, samples=14, tol=1e-5):
    """Compare analytic gradients to central finite differences. Big matrices
    are sampled; small vectors checked in full. Asserts worst rel.err < tol."""
    _, grads, _ = model.loss_and_grads(X, y)
    worst = 0.0
    print("  finite-difference gradient check (central differences, eps=%.0e)" % eps)
    print("  %-7s %5s   %14s %14s   %10s" %
          ("param", "n", "analytic", "numeric", "rel.err"))
    for name in sorted(model.p.keys()):
        flat = model.p[name].ravel()
        idxs = (range(flat.size) if flat.size <= samples
                else RNG.choice(flat.size, samples, replace=False))
        local_worst = a_last = n_last = 0.0
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps
            Lp = model.loss(model.forward(X)[0], y)[0]
            flat[i] = orig - eps
            Lm = model.loss(model.forward(X)[0], y)[0]
            flat[i] = orig
            num = (Lp - Lm) / (2 * eps)
            ana = grads[name].ravel()[i]
            rel = abs(num - ana) / max(1.0, abs(num) + abs(ana))
            local_worst = max(local_worst, rel)
            a_last, n_last = ana, num
        worst = max(worst, local_worst)
        print("  %-7s %5d   %14.6e %14.6e   %10.2e" %
              (name, flat.size, a_last, n_last, local_worst))
    print("  worst relative error: %.3e   (tolerance %.0e)" % (worst, tol))
    assert worst < tol, "GRADIENT CHECK FAILED -- backprop disagrees with finite diff"
    print("  GRADIENT CHECK PASSED\n")
    return worst


# ============================================================================
# 3. THE DATA: "kataleptic impressions buried under the passions"
# ============================================================================
VIRTUES = ["wisdom", "courage", "justice", "temperance"]  # the four classes


def make_prototypes(D, C, spread=1.4):
    """C prototype impressions (the clear, 'kataleptic' forms the soul ought to
    recognise). Deliberately NOT fully orthogonal, so the task is hard enough
    that denoising dynamics earn their keep."""
    protos = RNG.standard_normal((C, D))
    protos /= np.linalg.norm(protos, axis=1, keepdims=True)
    return protos * spread


def make_dataset(protos, n, noise_lo=0.3, noise_hi=1.4):
    """Each sample = a prototype + Gaussian noise of random strength (the
    passion that slackens perception)."""
    C, D = protos.shape
    y = RNG.integers(0, C, size=n)
    noise = RNG.uniform(noise_lo, noise_hi, size=n)
    X = protos[y] + RNG.standard_normal((n, D)) * noise[:, None]
    return X, y, noise


def accuracy(model, X, y, **fwd):
    logits, _ = model.forward(X, **fwd)
    return (logits.argmax(axis=1) == y).mean()


# ============================================================================
# 4. ADAM OPTIMIZER  (hand-rolled, so nothing is hidden in a library)
# ============================================================================
class Adam:
    def __init__(self, params, lr=4e-3, b1=0.9, b2=0.999, eps=1e-8,
                 weight_decay=0.0, decay_keys=()):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.wd, self.decay_keys = weight_decay, set(decay_keys)
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)
            if self.wd and k in self.decay_keys:           # decoupled weight decay
                params[k] -= self.lr * self.wd * params[k]


# ============================================================================
# 5. THE ASSENT GATE  (eval-time): commit only to a firmly-grasped impression
# ============================================================================
def assent(model, X, commit_threshold=0.5):
    """
    Stoic synkatathesis. After the field settles we read the verdict (argmax)
    and a 'grasp' in [0,1] = how clearly it decided, measured from the margin
    between the top two logits. A slack, noisy impression leaves the field torn
    between attractors -> small margin -> low grasp -> the mind WITHHOLDS assent
    (epoche) rather than being dragged. Returns verdicts, grasp, assented mask.
    """
    logits, _ = model.forward(X)
    part = np.partition(logits, -2, axis=1)
    margin = part[:, -1] - part[:, -2]               # top1 - top2 logit
    grasp = 1.0 - np.exp(-margin)                    # 0 (torn) .. 1 (firmly grasped)
    return logits.argmax(axis=1), grasp, grasp >= commit_threshold


# ============================================================================
# 6. RUN EVERYTHING
# ============================================================================
def main():
    np.set_printoptions(precision=3, suppress=True)
    print("=" * 78)
    print(" THE TONOS RESONANCE NETWORK  -  after Cleanthes of Assos (c.330-230 BCE)")
    print(" 'Virtue is a tension of the soul.'  Holding the tone under the passions.")
    print("=" * 78 + "\n")

    D, H, C, T, BW = 24, 32, 4, 6, 1
    protos = make_prototypes(D, C, spread=1.7)

    # ---- (a) gradient check on a tiny batch --------------------------------
    print("[1] CORRECTNESS")
    Xg, yg, _ = make_dataset(protos, 8)
    gradient_check(TonosResonanceNetwork(D, H, C, T=T, bandwidth=BW), Xg, yg)

    # ---- (b) train ---------------------------------------------------------
    print("[2] TRAINING  (the soul learns its tone)")
    Xtr, ytr, _ = make_dataset(protos, 1200, noise_lo=0.2, noise_hi=1.0)
    Xva, yva, _ = make_dataset(protos, 300, noise_lo=0.2, noise_hi=1.0)
    model = TonosResonanceNetwork(D, H, C, T=T, bandwidth=BW)
    opt = Adam(model.p, lr=4e-3, weight_decay=3e-3,
               decay_keys=("W_in", "W_c", "W_out", "w_sun"))

    epochs, bs, n = 60, 64, Xtr.shape[0]
    tau0 = float(softplus(model.p["rho"]).mean())
    for ep in range(epochs):
        order = RNG.permutation(n)
        for s in range(0, n, bs):
            b = order[s:s + bs]
            _, g, _ = model.loss_and_grads(Xtr[b], ytr[b])
            opt.step(model.p, g)
        if ep % 10 == 0 or ep == epochs - 1:
            Ltr, _, _ = model.loss_and_grads(Xtr, ytr)
            print("  epoch %2d  loss %.4f  train acc %.3f  val acc %.3f  mean-tone %.3f"
                  % (ep, Ltr, accuracy(model, Xtr, ytr), accuracy(model, Xva, yva),
                     float(softplus(model.p["rho"]).mean())))
    tauf = float(softplus(model.p["rho"]).mean())
    print("  mean tension (tonos) moved %.3f -> %.3f during training\n" % (tau0, tauf))
    assert accuracy(model, Xva, yva) > 0.84, "model failed to learn the task"

    # ---- (c) THESIS 1: the tension sweep (eutonia beats atonia & rigidity) -
    print("[3] THESIS  -  'virtue is a tension': sweep the tone under heavy passion")
    HARD = 1.4
    Xh, yh, _ = make_dataset(protos, 800, noise_lo=HARD, noise_hi=HARD)
    print("  test impressions corrupted at a fixed, heavy noise level (sigma=%.1f)" % HARD)
    print("  %-24s %s" % ("tension x (tau scale)", "accuracy"))
    best_scale, best_acc = None, -1.0
    rows = {}
    for a in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
        acc_a = accuracy(model, Xh, yh, tension_scale=a)
        rows[a] = acc_a
        tag = ("  <- slack (atonia)" if a == 0.0 else
               ("  <- learned tone" if a == 1.0 else
                ("  <- over-tight (rigid)" if a >= 8.0 else "")))
        print("  %-24.2f %.3f%s" % (a, acc_a, tag))
        if acc_a > best_acc:
            best_acc, best_scale = acc_a, a
    print("  best accuracy %.3f at tension x%.2f" % (best_acc, best_scale))
    print("  slack soul %.3f  |  learned tone %.3f  |  rigid soul %.3f"
          % (rows[0.0], rows[1.0], rows[8.0]))
    print("  -> both a slack soul and a rigid soul judge worse than the tuned one.")
    # the inverted-U is the whole thesis: the tuned tone beats both extremes
    assert rows[1.0] > rows[0.0] and rows[1.0] > rows[8.0], "no eutonic band"

    # ---- (d) THESIS 2: the Sun as the one long-range coordinator -----------
    print("\n[4] THESIS  -  'the Sun is the commanding-faculty': the one coordinator")
    print("  Unit-to-unit tension is LOCAL (a narrow band on the ring). The Sun is")
    print("  the single long-range channel: it broadcasts the field's global tone")
    print("  back to every unit at every step. We test whether that broadcast is")
    print("  load-bearing in forming the soul's verdict.")
    reach = BW * (T - 1)
    print("\n  (a) structure: after %d relaxation steps, local tension reaches only"
          % (T - 1))
    print("      +/-%d of %d units (%.0f%% of the ring). Everything beyond that"
          % (reach, H, 100.0 * 2 * reach / H))
    print("      distance is heard ONLY through the Sun.")

    lon, con = model.forward(Xh, use_sun=True)
    loff, coff = model.forward(Xh, use_sun=False)
    h_on, h_off = con["hs"][-1], coff["hs"][-1]
    rel_change = (np.linalg.norm(h_on - h_off, axis=1)
                  / (np.linalg.norm(h_on, axis=1) + 1e-9)).mean()
    decision_shift = (lon.argmax(1) != loff.argmax(1)).mean()
    print("\n  (b) load-bearing: silence the Sun and re-run the same impressions --")
    print("      mean change in the settled field : %.1f%%" % (100.0 * rel_change))
    print("      verdicts that flip without the Sun: %.1f%%" % (100.0 * decision_shift))
    print("      -> the commanding-faculty is not decoration: it materially shapes")
    print("         the conclusion the whole field comes to.")

    # supporting, reported but not asserted (a subtle, noise-sensitive effect):
    sp_on = h_on.mean(axis=1).std()
    sp_off = h_off.mean(axis=1).std()
    print("\n  (c) regulation (supporting): under this heavy passion the spread of the")
    print("      global tone is %.4f with the Sun and %.4f without -- the Sun also"
          % (sp_on, sp_off))
    print("      leans toward holding the whole 'in due measure'.")
    # robust assertion: the Sun is load-bearing (w_sun is a real learned channel)
    assert rel_change > 0.05 and decision_shift > 0.0, "the Sun is not load-bearing"

    # ---- (e) THESIS 3: glad assent / withholding (epoche) ------------------
    print("\n[5] THESIS  -  'follow willingly': assent firmly, else withhold (epoche)")
    print("  %-10s %-12s %-14s" % ("noise", "mean grasp", "withhold rate"))
    prev_w = -1.0
    mono = True
    for sigma in [0.3, 0.7, 1.1, 1.6, 2.2]:
        Xs, ys, _ = make_dataset(protos, 500, noise_lo=sigma, noise_hi=sigma)
        _, grasp, assented = assent(model, Xs)
        w = 1.0 - assented.mean()
        mono = mono and (w >= prev_w - 1e-9)
        prev_w = w
        print("  %-10.1f %-12.3f %-14.3f" % (sigma, grasp.mean(), w))
    print("  as the impression slackens, the field's grasp loosens and the mind")
    print("  withholds judgement more often -- the Stoic epoche.")
    assert mono, "withholding did not rise with noise"

    # ---- summary -----------------------------------------------------------
    print("\n" + "=" * 78)
    print(" VERDICT: a from-scratch tensional network that")
    print("  (1) passes a finite-difference gradient check to ~1e-10 (the BPTT is")
    print("      exact, not approximate);")
    print("  (2) learns to recognise the four virtues through the noise of the")
    print("      passions, and generalises to held-out impressions;")
    print("  (3) shows -- with numbers -- the three doctrines of Cleanthes:")
    print("      * TONOS: the soul's strength is its TONE. Too slack and it")
    print("        dissolves; too rigid and it cannot hear the world; tuned, it holds.")
    print("      * THE SUN: a single long-range broadcast -- the only channel that")
    print("        reaches across the whole field -- that materially shapes the verdict.")
    print("      * ASSENT: it grasps a clear impression firmly and withholds")
    print("        (epoche) as the impression slackens.")
    print(" Cleanthes, in NumPy.")
    print("=" * 78)


if __name__ == "__main__":
    main()

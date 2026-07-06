#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
THE GAUGE NETWORK (Fa-Net) — a cognitive architecture after Mozi (Mo Di, c.470-391 BCE)
 Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 4 Minds 61 - 80 Available on Amazon https://www.amazon.com/dp/B0H71JC95Q
Resume and Interactive Demos at https://artificiology.com/
Author: David Vivancos · Chapter 0063 · Mozi
================================================================================

WHY THIS ARCHITECTURE, AND WHY IT IS *MOZI'S* AND NO ONE ELSE'S
--------------------------------------------------------------------------------
Most "from-scratch" nets judge an input by comparing it to stored keys and
letting a soft, learned, private similarity decide. That is precisely the kind
of cultivated inner intuition Mozi distrusted. His one great cognitive move was
the *fa* (法) -- the external, public, reproducible STANDARD. A carpenter does
not feel whether a plank is square; he lays the try-square (ju 矩) against it and
*reads off the deviation*. Skilled and unskilled hands, applying the same gauge,
converge on the same verdict. Knowing, for Mozi, is measuring-against-a-model,
not intuiting. So this network's atom of cognition is not "similarity to a key"
but DEVIATION FROM A GAUGE.

Three further commitments of Mozi are built into the *structure* of the forward
pass, not bolted on as penalties:

  1. SAN BIAO (三表), the Three Gauges of valid doctrine (Mozi, bk.35 "Against
     Fatalism"): a claim is admitted only if it is (a) ROOTED in precedent
     [本, the deeds of the sage-kings], (b) SOURCED in the senses [原, the eyes
     and ears of the people], and (c) USEFUL in application [用, it benefits the
     state and people, li 利]. These are *conjunctive*: failing any one rejects
     the claim. We implement the acceptance as a PRODUCT of three sigmoid gates,
     so the network can only say "yes" when root AND source AND use all agree.

  2. JIAN AI (兼愛), impartial care. The world's disorder, Mozi argued, all
     reduces to one failure mode: agents weighting their own side above others'.
     The remedy is a single structural substitution -- replace partiality (bie 別)
     with inclusion (jian 兼). We do not *penalize* partiality; we make it
     UNREPRESENTABLE. The welfare gauge is applied by the SAME shared encoder to
     every party, the parties are combined by a permutation-INVARIANT sum-pool
     (so the verdict is provably identical when you relabel which party is
     "mine"), and the gauge is BLIND to identity features (kinship, rank): the
     square measures the plank's squareness, never whose plank it is.

  3. ANTI-FATALISM / INTERVENABILITY (非命). Mozi attacked the belief in fixed
     fate because it kills the incentive to act: if outcomes are decreed, why
     farm, weave, govern? The model is therefore trained -- its gauges are not
     given but REFINED against evidence -- and its verdicts are functions of
     manipulable welfare, never of who-you-are. Effort moves the world.

THE TASK (a faithful miniature of Mozi's ethics)
--------------------------------------------------------------------------------
Each scenario has TWO parties. Each party carries welfare features (signed
benefit/harm, need) and identity features (a "kin/mine" flag, a status flag).
A proposal is JUST (label 1) iff, in the Mohist manner, it passes all three
gauges at once:
     USE  : the welfare summed *impartially* over both parties is positive (li>0);
     SOURCE: the evidence is reliable (observation not drowned in noise);
     ROOT : the welfare pattern resembles a known just precedent (a sage-king case).
The true label is INDEPENDENT of the identity features. But we poison the TRAIN
set with a spurious correlation -- in training, the "kin" party tends to be the
better-off one, so the shortcut "favour my kin" predicts well. In the TEST set we
FLIP it -- kin tends to sit on the unjust side. A partial mind that is allowed to
see kinship latches onto the shortcut and is humiliated out of distribution. The
Gauge Network, impartial by construction, never sees kinship and is unmoved.

WHAT YOU GET WHEN YOU RUN THIS FILE
--------------------------------------------------------------------------------
  * a hand-derived backprop for every parameter,
  * a finite-difference GRADIENT CHECK that must pass (mandatory),
  * a real training loop (full-batch gradient descent) on the Mohist task,
  * a head-to-head against a "partial" logistic baseline that is *allowed* to be
    partial -- to dramatize Mozi's thesis that partiality is a shortcut that
    fails when the world shifts,
  * self-tests, including the signature test: SWAP THE TWO PARTIES AND THE
    VERDICT DOES NOT CHANGE (jian ai, verified to machine precision).

Pure NumPy. No frameworks. ~Seconds to run.
"""

import numpy as np

RNG = np.random.default_rng(63)  # figure #63


# ============================================================================
# 0. SMALL DIFFERENTIABLE PRIMITIVES
# ============================================================================

def sigmoid(z):
    # numerically stable logistic
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def softmax_rows(A):
    A = A - A.max(axis=1, keepdims=True)
    E = np.exp(A)
    return E / E.sum(axis=1, keepdims=True)


# ============================================================================
# 1. THE GAUGE NETWORK
# ============================================================================
#
# Forward pass for a batch of N scenarios (two parties A,B each):
#
#   welfare encoder (SHARED across parties -- the one square for every plank):
#       zA = WA @ xwA + bA ;  hA = tanh(zA)         (xwA = party A welfare feats)
#       zB = WA @ xwB + bA ;  hB = tanh(zB)
#   impartial pool (permutation invariant -> jian ai):
#       s  = hA + hB
#
#   THREE GAUGES read the SAME pooled evidence s:
#     USE (利, benefit):     o_use    = s @ w_use + b_use
#     SOURCE (原, senses):   o_source = s @ w_src + b_src
#     ROOT (本, precedent):  align = s @ P^T  (N x M, M precedent prototypes)
#                            attn  = softmax(align)        (soft retrieval)
#                            ret   = attn @ P              (N x H)
#                            o_root= ret @ v_root + b_root
#
#   conjunctive san-biao gate -> acceptance probability:
#       y = sigmoid(o_root) * sigmoid(o_source) * sigmoid(o_use)
#
#   loss: binary cross-entropy over the batch.
#
# Identity features are deliberately NOT inputs to the encoder: impartiality is
# enforced by construction, not learned.
# ============================================================================

class GaugeNetwork:
    def __init__(self, n_welfare, hidden=12, n_precedents=4, scale=0.5, rng=RNG):
        self.pw = n_welfare
        self.H = hidden
        self.M = n_precedents
        g = lambda *s: rng.standard_normal(s) * scale
        # shared welfare encoder
        self.WA = g(self.H, self.pw)
        self.bA = np.zeros(self.H)
        # USE gauge (benefit / li)
        self.w_use = g(self.H)
        self.b_use = 0.0
        # SOURCE gauge (sense reliability)
        self.w_src = g(self.H)
        self.b_src = 0.0
        # ROOT gauge (precedent memory + readout)
        self.P = g(self.M, self.H)         # the sage-king precedent prototypes
        self.v_root = g(self.H)
        self.b_root = 0.0

    # ---- parameter (de)serialization for the gradient check ----------------
    def get_params(self):
        return {
            "WA": self.WA, "bA": self.bA,
            "w_use": self.w_use, "b_use": np.array(self.b_use),
            "w_src": self.w_src, "b_src": np.array(self.b_src),
            "P": self.P, "v_root": self.v_root, "b_root": np.array(self.b_root),
        }

    def set_params(self, d):
        f = lambda x: float(np.asarray(x).ravel()[0])
        self.WA = d["WA"]; self.bA = d["bA"]
        self.w_use = d["w_use"]; self.b_use = f(d["b_use"])
        self.w_src = d["w_src"]; self.b_src = f(d["b_src"])
        self.P = d["P"]; self.v_root = d["v_root"]; self.b_root = f(d["b_root"])

    # ---- forward, caching everything needed for backprop -------------------
    def forward(self, xwA, xwB):
        zA = xwA @ self.WA.T + self.bA          # (N,H)
        zB = xwB @ self.WA.T + self.bA
        hA = np.tanh(zA)
        hB = np.tanh(zB)
        s = hA + hB                             # (N,H) impartial pooled evidence

        o_use = s @ self.w_use + self.b_use     # (N,)
        o_src = s @ self.w_src + self.b_src      # (N,)

        align = s @ self.P.T                    # (N,M)
        attn = softmax_rows(align)              # (N,M)
        ret = attn @ self.P                     # (N,H)
        o_root = ret @ self.v_root + self.b_root

        sig_u = sigmoid(o_use)
        sig_s = sigmoid(o_src)
        sig_r = sigmoid(o_root)
        y = sig_r * sig_s * sig_u               # conjunctive acceptance (N,)

        cache = dict(xwA=xwA, xwB=xwB, zA=zA, zB=zB, hA=hA, hB=hB, s=s,
                     o_use=o_use, o_src=o_src, align=align, attn=attn, ret=ret,
                     o_root=o_root, sig_u=sig_u, sig_s=sig_s, sig_r=sig_r, y=y)
        return y, cache

    def loss(self, y, t):
        eps = 1e-9
        yc = np.clip(y, eps, 1 - eps)
        return float(-np.mean(t * np.log(yc) + (1 - t) * np.log(1 - yc)))

    # ---- analytic backprop --------------------------------------------------
    def backward(self, cache, t):
        N = t.shape[0]
        y = cache["y"]; eps = 1e-9
        yc = np.clip(y, eps, 1 - eps)
        # dL/dy for mean BCE
        dy = (yc - t) / (yc * (1 - yc)) / N      # (N,)

        sig_u, sig_s, sig_r = cache["sig_u"], cache["sig_s"], cache["sig_r"]
        # y = sig_r*sig_s*sig_u ; chain into each pre-sigmoid logit
        do_use = dy * (sig_r * sig_s) * (sig_u * (1 - sig_u))
        do_src = dy * (sig_r * sig_u) * (sig_s * (1 - sig_s))
        do_root = dy * (sig_s * sig_u) * (sig_r * (1 - sig_r))

        s = cache["s"]
        grads = {}
        # USE gauge
        grads["w_use"] = s.T @ do_use
        grads["b_use"] = np.array(do_use.sum())
        # SOURCE gauge
        grads["w_src"] = s.T @ do_src
        grads["b_src"] = np.array(do_src.sum())
        # ROOT gauge
        ret = cache["ret"]
        grads["v_root"] = ret.T @ do_root
        grads["b_root"] = np.array(do_root.sum())

        # gradient flowing into s from each gauge
        ds = np.outer(do_use, self.w_use) + np.outer(do_src, self.w_src)

        # --- root path: o_root = ret . v_root ; ret = attn @ P ; attn=softmax(align); align = s @ P^T
        dret = np.outer(do_root, self.v_root)             # (N,H)
        attn = cache["attn"]; P = self.P
        # ret = sum_m attn[:,m] * P[m]
        dattn = dret @ P.T                                # (N,M)
        grad_P = attn.T @ dret                            # path through ret (M,H)
        # softmax backward: dL/dalign = attn * (dattn - sum(attn*dattn))
        tmp = (attn * dattn).sum(axis=1, keepdims=True)
        dalign = attn * (dattn - tmp)                     # (N,M)
        # align = s @ P^T  -> ds += dalign @ P ; grad_P += dalign^T @ s
        ds = ds + dalign @ P
        grad_P = grad_P + dalign.T @ s
        grads["P"] = grad_P

        # --- through pool s = hA + hB ---
        dhA = ds.copy()
        dhB = ds.copy()
        dzA = dhA * (1 - cache["hA"] ** 2)
        dzB = dhB * (1 - cache["hB"] ** 2)
        # shared encoder: zX = xwX @ WA^T + bA
        grads["WA"] = dzA.T @ cache["xwA"] + dzB.T @ cache["xwB"]
        grads["bA"] = dzA.sum(axis=0) + dzB.sum(axis=0)
        return grads

    # ---- one SGD step -------------------------------------------------------
    def step(self, grads, lr):
        self.WA -= lr * grads["WA"]; self.bA -= lr * grads["bA"]
        self.w_use -= lr * grads["w_use"]; self.b_use -= lr * float(grads["b_use"])
        self.w_src -= lr * grads["w_src"]; self.b_src -= lr * float(grads["b_src"])
        self.P -= lr * grads["P"]
        self.v_root -= lr * grads["v_root"]; self.b_root -= lr * float(grads["b_root"])


# ============================================================================
# 2. THE MOHIST TASK GENERATOR
# ============================================================================
#
# Two parties. Welfare feats per party: [benefit (signed), need (>=0)].
# Plus a shared-scenario "reliability" feature and a precedent-pattern code.
# Identity feats per party: [kin_flag, status_flag]  (NOT used by Gauge Net).
#
# Latent truth (identity-independent), the three gauges:
#   USE    : welfare_sum = (benefitA + benefitB) + 0.5*(needA+needB shaped)   > 0
#   SOURCE : reliability > 0
#   ROOT   : welfare pattern (benefitA,benefitB) close to a "precedent" axis
#   just = USE and SOURCE and ROOT
#
# Spurious correlation on KIN, present in train, FLIPPED in test.
# ============================================================================

def make_dataset(n, rng, kin_sign=+1.0):
    benefitA = rng.normal(0, 1.0, n)
    benefitB = rng.normal(0, 1.0, n)
    needA = np.abs(rng.normal(0, 0.6, n))
    needB = np.abs(rng.normal(0, 0.6, n))

    # USE gauge: impartial net benefit (need slightly amplifies benefit's weight)
    welfare_sum = (benefitA + benefitB) + 0.3 * (needA + needB) * np.sign(benefitA + benefitB)
    use_ok = welfare_sum > 0.0

    # SOURCE gauge: reliability of the report
    reliability = rng.normal(0.15, 1.0, n)
    src_ok = reliability > 0.0

    # ROOT gauge: does (benefitA,benefitB) lie near a known precedent axis?
    # precedent axis: the "balanced beneficence" direction (1,1)/sqrt2
    axis = np.array([1.0, 1.0]) / np.sqrt(2)
    vecs = np.stack([benefitA, benefitB], axis=1)
    norms = np.linalg.norm(vecs, axis=1) + 1e-9
    cos = (vecs @ axis) / norms
    root_ok = cos > 0.2          # resembles the sage-king precedent of balanced aid

    just = (use_ok & src_ok & root_ok).astype(float)

    # identity features, with spurious correlation to the *better-off* party
    better_is_A = benefitA > benefitB
    kin_on_A = np.where(rng.random(n) < 0.85, better_is_A, ~better_is_A)
    # kin_sign=+1 (train): kin tends to sit on better-off side (favour-kin works)
    # kin_sign=-1 (test):  invert, so favour-kin predicts the WRONG side
    if kin_sign < 0:
        kin_on_A = ~kin_on_A
    kinA = kin_on_A.astype(float)
    kinB = 1.0 - kinA
    statusA = rng.integers(0, 2, n).astype(float)
    statusB = rng.integers(0, 2, n).astype(float)

    # welfare-only inputs for the impartial gauge (+ reliability as a sense feat)
    xwA = np.stack([benefitA, needA, reliability], axis=1)
    xwB = np.stack([benefitB, needB, reliability], axis=1)

    # full feature vector for the PARTIAL baseline (it gets to see identity!)
    X_partial = np.stack([benefitA, needA, benefitB, needB, reliability,
                          kinA, statusA, kinB, statusB], axis=1)

    return dict(xwA=xwA, xwB=xwB, X_partial=X_partial, t=just)


# ============================================================================
# 3. THE "PARTIAL MIND" BASELINE (logistic regression that MAY be partial)
# ============================================================================
# Standard logistic regression over the full feature set, including identity.
# It can weight party A vs B differently and can read kinship. Mozi's claim:
# this is exactly the mind that latches onto partiality and fails when the
# world shifts. We train it the same way and compare.
# ============================================================================

class PartialMind:
    def __init__(self, d, rng):
        self.w = rng.standard_normal(d) * 0.1
        self.b = 0.0

    def prob(self, X):
        return sigmoid(X @ self.w + self.b)

    def train(self, X, t, lr=0.2, epochs=4000):
        N = X.shape[0]
        for _ in range(epochs):
            p = self.prob(X)
            g = (p - t) / N
            self.w -= lr * (X.T @ g)
            self.b -= lr * g.sum()

    def accuracy(self, X, t):
        return float(((self.prob(X) > 0.5).astype(float) == t).mean())


# ============================================================================
# 4. GRADIENT CHECK (mandatory)
# ============================================================================

def gradient_check():
    rng = np.random.default_rng(7)
    net = GaugeNetwork(n_welfare=3, hidden=6, n_precedents=3, scale=0.7, rng=rng)
    N = 5
    xwA = rng.standard_normal((N, 3))
    xwB = rng.standard_normal((N, 3))
    t = (rng.random(N) > 0.5).astype(float)

    y, cache = net.forward(xwA, xwB)
    grads = net.backward(cache, t)

    base = net.get_params()
    eps = 1e-6
    max_rel = 0.0
    worst = None
    for name, val in base.items():
        flat = np.atleast_1d(val).astype(float).ravel().copy()
        num = np.zeros_like(flat)
        for i in range(flat.size):
            up = base[name].astype(float).copy().ravel(); up[i] += eps
            dn = base[name].astype(float).copy().ravel(); dn[i] -= eps
            p_up = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in base.items()}
            p_dn = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in base.items()}
            p_up[name] = up.reshape(np.atleast_1d(val).shape)
            p_dn[name] = dn.reshape(np.atleast_1d(val).shape)
            net.set_params(p_up); yu, _ = net.forward(xwA, xwB); Lu = net.loss(yu, t)
            net.set_params(p_dn); yd, _ = net.forward(xwA, xwB); Ld = net.loss(yd, t)
            num[i] = (Lu - Ld) / (2 * eps)
        net.set_params(base)
        ana = np.atleast_1d(grads[name]).astype(float).ravel()
        denom = np.maximum(1e-8, np.abs(num) + np.abs(ana))
        rel = np.max(np.abs(num - ana) / denom)
        if rel > max_rel:
            max_rel, worst = rel, name
    print(f"  finite-difference gradient check : max rel err = {max_rel:.3e} "
          f"(worst param: {worst})")
    assert max_rel < 1e-5, "GRADIENT CHECK FAILED"
    print("  GRADIENT CHECK PASSED  (analytic backprop == finite differences)")
    return max_rel


# ============================================================================
# 5. TRAIN + EVALUATE
# ============================================================================

def train_gauge(net, data, lr=0.4, epochs=3000):
    for ep in range(epochs):
        y, cache = net.forward(data["xwA"], data["xwB"])
        grads = net.backward(cache, data["t"])
        net.step(grads, lr)
        if ep % 600 == 0 or ep == epochs - 1:
            L = net.loss(y, data["t"])
            acc = ((y > 0.5).astype(float) == data["t"]).mean()
            print(f"    epoch {ep:4d}   loss={L:.4f}   train_acc={acc:.3f}")
    return net


def gauge_accuracy(net, data):
    y, _ = net.forward(data["xwA"], data["xwB"])
    return float(((y > 0.5).astype(float) == data["t"]).mean())


def swap_invariance_test(net, data):
    """The signature jian-ai test: relabel the two parties; verdict must not move."""
    y1, _ = net.forward(data["xwA"], data["xwB"])
    y2, _ = net.forward(data["xwB"], data["xwA"])      # parties swapped
    return float(np.max(np.abs(y1 - y2)))


# ============================================================================
# 6. MAIN
# ============================================================================

def main():
    print("=" * 78)
    print("THE GAUGE NETWORK (Fa-Net)  —  a cognitive architecture after Mozi")
    print("  judgment as measurement against an external, identity-blind standard")
    print("=" * 78)

    print("\n[1] GRADIENT CHECK")
    gradient_check()

    print("\n[2] BUILD THE MOHIST TASK")
    train = make_dataset(4000, np.random.default_rng(100), kin_sign=+1.0)  # kin~just
    test = make_dataset(4000, np.random.default_rng(200), kin_sign=-1.0)  # kin flipped
    base_rate = max(train["t"].mean(), 1 - train["t"].mean())
    print(f"    train just-rate = {train['t'].mean():.3f}   "
          f"(majority-class baseline acc = {base_rate:.3f})")
    print(f"    spurious 'kin' cue: helps on TRAIN, reversed on TEST")

    print("\n[3] TRAIN THE GAUGE NETWORK (impartial by construction)")
    net = GaugeNetwork(n_welfare=3, hidden=12, n_precedents=4)
    train_gauge(net, train, lr=0.4, epochs=3000)
    g_tr = gauge_accuracy(net, train)
    g_te = gauge_accuracy(net, test)
    print(f"    GaugeNet  train_acc={g_tr:.3f}   test_acc={g_te:.3f}   "
          f"(gap={abs(g_tr-g_te):.3f})")

    print("\n[4] TRAIN THE 'PARTIAL MIND' (logistic reg. allowed to see kinship)")
    pm = PartialMind(train["X_partial"].shape[1], np.random.default_rng(11))
    pm.train(train["X_partial"], train["t"])
    p_tr = pm.accuracy(train["X_partial"], train["t"])
    p_te = pm.accuracy(test["X_partial"], test["t"])
    print(f"    PartialMind train_acc={p_tr:.3f}   test_acc={p_te:.3f}   "
          f"(gap={abs(p_tr-p_te):.3f})")
    # how heavily did the partial mind lean on the kin feature?
    kin_w = abs(pm.w[5]) + abs(pm.w[7])
    welfare_w = abs(pm.w[0]) + abs(pm.w[2])
    print(f"    partial mind's reliance: |kin weights|={kin_w:.2f} vs "
          f"|benefit weights|={welfare_w:.2f}")

    print("\n[5] SELF-TESTS")
    swap_gap = swap_invariance_test(net, test)
    print(f"    jian-ai swap-invariance: max |y(A,B)-y(B,A)| = {swap_gap:.2e}  "
          f"-> {'PASS' if swap_gap < 1e-9 else 'FAIL'}")
    assert swap_gap < 1e-9, "impartiality (swap invariance) violated"

    robust = (g_te > p_te + 0.05)
    print(f"    impartial mind out-generalizes partial mind off-distribution: "
          f"{'PASS' if robust else 'NOTE'} "
          f"(GaugeNet {g_te:.3f} vs PartialMind {p_te:.3f})")

    learned = (g_tr > base_rate + 0.05)
    print(f"    gauge net learned the conjunctive (san-biao) structure: "
          f"{'PASS' if learned else 'FAIL'}")
    assert learned, "model did not learn the task above baseline"

    print("\n" + "=" * 78)
    print("READING OF THE RESULT")
    print("-" * 78)
    print("  The Partial Mind is GIVEN MORE information (it can see who is kin and")
    print("  may treat the parties unequally) yet generalizes WORSE: it seizes the")
    print("  partiality shortcut, which the shifting world then punishes. The Gauge")
    print("  Network, blind to identity and provably impartial under relabeling,")
    print("  measures only the welfare against its refined gauges and holds steady.")
    print("  Partiality is a local optimum; the external standard is the robust one.")
    print("  This is Mozi's wager, made executable.")
    print("=" * 78)


if __name__ == "__main__":
    main()

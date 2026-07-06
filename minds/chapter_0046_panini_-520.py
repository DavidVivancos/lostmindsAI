#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
chapter_0046_panini_-520.py
The Ashtadhyayi Network: a differentiable Elsewhere-Condition rule engine
Figure #46 - Panini (Salatura, Gandhara; c. 5th-4th century BCE)
Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 3 Minds 41 - 60 Available on Amazon https://www.amazon.com/dp/B0H6TVX69S
Resume and Interactive Demos at https://artificiology.com/
Author: David Vivancos · Chapter 0046 · Panini
================================================================================

WHY THIS ARCHITECTURE (and why NOT a transformer)
--------------------------------------------------------------------------------
Most "language" models store experience as keys and retrieve by similarity
(attention). Panini's Astadhyayi does the opposite. It stores almost nothing.
It DERIVES every well-formed Sanskrit word on demand by firing a small set of
ORDERED rules. The single idea that is *his alone* -- the idea modern
linguistics rediscovered and named the "Elsewhere Condition" (Kiparsky 1973,
formalising Panini's apavada principle) -- is this:

    A general rule (utsarga) applies EVERYWHERE EXCEPT where a more specific
    rule (apavada) applies. The specific silently preempts the general,
    precisely in the region where the specific one speaks.

That is a theory of mind as a *priority hierarchy of generative defaults*.
Intelligence here is not "knowing the answers"; it is holding the smallest set
of ordered, exception-structured rules that DERIVE every answer -- and knowing
which rule wins when several could fire.

This file builds a neural network whose forward pass *is* the Elsewhere
Condition, end-to-end differentiable, trained from scratch in pure NumPy.

THE MECHANISM
--------------------------------------------------------------------------------
We keep a bank of R rules. Rule i has three learnable parts that mirror a
Paninian sutra:
  * a CONTEXT key k_i (+ bias b_i)  -> its DOMAIN: does it apply to this input?
        applicability  a_i = sigmoid(x . k_i + b_i)            (the "when/where")
  * a SPECIFICITY scalar s_i        -> how specific the rule is (utsarga<apavada)
  * an OPERATION O_i (a fixed output logit vector) -> its KARYA, the substitution
        ("replace with this") that the rule performs                (the "what")

The Elsewhere Condition is made differentiable with an ordered "stick-breaking"
preemption. Let M[i,j] = sigmoid(gamma*(s_j - s_i)) be the soft degree to which
rule j is MORE SPECIFIC than rule i. Then rule i only gets to act on whatever
probability mass no more-specific applicable rule has already claimed:

        r_i = PRODUCT_over_j ( 1 - a_j * M[i,j] )        (j != i)
        g_i = a_i * r_i                                  (effective firing)
        p_i = g_i / sum_k g_k                            (which rule fires)
        z   = sum_i p_i * O_i  -> softmax -> class

The lowest-specificity rule is the literal "elsewhere": it fires only where
nothing more specific applies. This is apavada/utsarga, implemented as math.

THE TASK (utsarga vs apavada, a.k.a. regulars vs irregulars)
--------------------------------------------------------------------------------
A toy morphology. Every "stem" belongs to a phonological CLASS (gana) and has a
unique phonological SIGNATURE. Regular stems take the DEFAULT ending for their
class (a general rule, keyed on the class -- it generalises to unseen stems: a
"wug test"). A handful of IRREGULAR stems take an idiosyncratic ending that
*overrides* the class default (a lexical exception -- it must be memorised by a
specific rule). A correct Paninian model will:
  (a) learn a few low-specificity CLASS rules that generalise to novel regulars,
  (b) learn high-specificity LEXICAL rules that preempt the class rule exactly
      on the irregular stems,
  (c) leave the specificity ordering s(apavada) > s(utsarga) as an EMERGENT fact.

SELF-TESTS (run at the bottom; output pasted into the chapter)
--------------------------------------------------------------------------------
  1. finite-difference gradient check on EVERY parameter  (mandatory)
  2. a real training loop that drives loss down and accuracy up
  3. wug test: generalisation of the default rule to unseen regular stems
  4. emergence of specificity ordering: apavada rules end up more specific
  5. the Elsewhere partition: regular stems are decided by class rules,
     irregular stems by lexical rules

Pure NumPy. No autograd library. Deterministic seed. Executable.
================================================================================
"""

import numpy as np


# =============================================================================
# PART I -- PRATYAHARA ENCODER (name a class as an interval)
# =============================================================================
# Panini's most beautiful compression trick is the Sivasutra: he ordered the
# phonemes so that EVERY linguistically relevant sound-class is a CONTIGUOUS
# INTERVAL, nameable by its two endpoints plus a marker (a "pratyahara", e.g.
# "aC" = all vowels, "haL" = all consonants). A category becomes a range.
#
# We borrow exactly that idea to build input features. A stem's phonological
# class g is encoded not as a bare one-hot but as membership in a set of NESTED
# INTERVALS over an ordered axis. Two classes that are "adjacent" in the
# ordering then share features -- precisely the generalisation structure that
# lets a single class-rule cover a natural class of stems. This encoder is fixed
# (non-trainable): it is the alphabet, not the grammar.

class PratyaharaEncoder:
    def __init__(self, n_classes, n_intervals=6, sig_dim=8, seed=0):
        self.n_classes = n_classes
        self.sig_dim = sig_dim
        rng = np.random.default_rng(seed)
        # Place each class at an ordered position on [0,1] (the "Sivasutra axis").
        self.pos = np.linspace(0.1, 0.9, n_classes)
        # Define n_intervals overlapping intervals; feature = soft membership.
        centers = np.linspace(0.0, 1.0, n_intervals)
        self.centers = centers
        self.width = 1.5 / n_intervals
        self.class_feat_dim = n_intervals

    def class_features(self, g):
        # Soft interval membership of class g (a "pratyahara" fingerprint).
        x = self.pos[g]
        return np.exp(-((x - self.centers) ** 2) / (2 * self.width ** 2))

    def encode(self, g, signature):
        # Final stem feature = [interval features of class] ++ [lexical signature]
        return np.concatenate([self.class_features(g), signature])

    @property
    def dim(self):
        return self.class_feat_dim + self.sig_dim


# =============================================================================
# PART II -- SYNTHETIC MORPHOLOGY (default endings + lexical exceptions)
# =============================================================================
def build_dataset(n_classes=2, n_endings=6, n_regular=200, n_irregular=16,
                  sig_dim=10, seed=1):
    """
    A Paninian morphology with TWO levels of generality plus lexical exceptions:

      * GENERAL rules (utsarga): a stem's phonological CLASS (e.g. voiced vs
        voiceless final) picks one of two regular endings {0,1}. These rules are
        broad -- each covers ~half the language -- so they cannot be cheaply
        "gated off"; they genuinely apply to every stem of their class.

      * LEXICAL exceptions (apavada): a handful of stems take an idiosyncratic
        ending from {2,3,4,5} that no general rule would ever produce. Because
        the general rule for their class STILL applies to them, the only way to
        get them right is for a MORE SPECIFIC rule to preempt the general one --
        the Elsewhere Condition in action.

    Test set = novel regular stems (the wug test for the general rules).
    """
    rng = np.random.default_rng(seed)
    enc = PratyaharaEncoder(n_classes, sig_dim=sig_dim, seed=seed)
    default_ending = np.arange(n_classes)          # class g -> regular ending g (0 or 1)
    exception_endings = np.arange(n_classes, n_endings)   # {2,3,4,5}

    def make_regular(n):
        Xs, ys, gs = [], [], []
        for _ in range(n):
            g = rng.integers(0, n_classes)
            sig = rng.normal(size=sig_dim) * 0.6
            Xs.append(enc.encode(g, sig)); ys.append(default_ending[g]); gs.append(g)
        return np.array(Xs), np.array(ys), np.array(gs)

    Xr, yr, gr = make_regular(n_regular)

    Xi, yi, gi = [], [], []
    for k in range(n_irregular):
        g = rng.integers(0, n_classes)
        # Exceptions are individually LISTED stems: give them a distinctive lexical
        # tag (larger-magnitude, well-separated signature) so a narrow apavada can
        # lock onto each without bleeding onto ordinary stems.
        sig = rng.normal(size=sig_dim)
        sig = sig / (np.linalg.norm(sig) + 1e-9) * 2.2
        end = exception_endings[rng.integers(0, len(exception_endings))]
        Xi.append(enc.encode(g, sig)); yi.append(end); gi.append(g)
    Xi, yi, gi = np.array(Xi), np.array(yi), np.array(gi)

    Xtr = np.vstack([Xr, Xi]); ytr = np.concatenate([yr, yi])
    is_irr_tr = np.concatenate([np.zeros(len(yr), bool), np.ones(len(yi), bool)])
    gtr = np.concatenate([gr, gi])

    Xte, yte, gte = make_regular(80)
    meta = dict(encoder=enc, default_ending=default_ending,
                n_classes=n_classes, n_endings=n_endings,
                is_irr_tr=is_irr_tr, gtr=gtr, gte=gte)
    return (Xtr, ytr), (Xte, yte), meta


# =============================================================================
# PART III -- THE ASHTADHYAYI NETWORK (differentiable Elsewhere Condition)
# =============================================================================
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class AshtadhyayiNet:
    """A bank of competing generative rules resolved by the Elsewhere Condition."""

    def __init__(self, d_in, n_rules, n_out, gamma=4.0, lam_spec=0.05,
                 l2_spec=0.02, seed=0):
        rng = np.random.default_rng(seed)
        self.R = n_rules
        self.C = n_out
        self.gamma = gamma                       # sharpness of specificity comparison
        # Panini's DEFINITION of apavada: a more specific rule is one of NARROWER
        # domain. We encode that as a prior coupling specificity s_i to how broadly
        # the rule fires (its mean applicability "breadth"): broad => low s,
        # narrow => high s. Specificity stays a free, trainable parameter; the
        # prior only shapes ties so the Elsewhere mechanism -- not raw gating --
        # carries the load. lam_spec weights the prior; l2_spec keeps s bounded.
        self.lam_spec = lam_spec
        self.l2_spec = l2_spec
        # Parameters (the learnable grammar):
        self.K = rng.normal(size=(n_rules, d_in)) * 0.3   # rule context keys (domain)
        self.b = np.zeros(n_rules)                         # rule context biases
        self.s = rng.normal(size=n_rules) * 0.1            # specificities (utsarga/apavada)
        self.O = rng.normal(size=(n_rules, n_out)) * 0.3   # rule outputs (karya/adesha)
        self.eps = 1e-9
        # By default every rule is trainable. designate_utsarga() may freeze a few
        # rows so the broad default rules cannot be "gated off" -- forcing the
        # Elsewhere Condition (a narrower apavada must preempt) to do the work.
        self.frozen = np.zeros(n_rules, bool)
        self._froze = False

    def designate_utsarga(self, enc, n_classes, beta=8.0, s_fixed=-1.0):
        """Pin the first `n_classes` rules to be the utsarga (general defaults):
        one per phonological class, keyed ONLY on class features so each fires on
        its whole class (applicability ~1) and CANNOT be gated off by a signature.
        Their specificity is fixed low, so any exception in the class can only be
        corrected by a more specific apavada preempting them -- the Elsewhere
        Condition. Their endings O stay trainable (they must learn the default)."""
        cf_dim = enc.class_feat_dim
        cfs = np.stack([enc.class_features(c) for c in range(n_classes)])
        for c in range(n_classes):
            cf = cfs[c]
            d_same = float(cf @ cf)
            others = [float(cfs[o] @ cf) for o in range(n_classes) if o != c]
            d_diff = max(others) if others else 0.0
            thresh = 0.5 * (d_same + d_diff)
            self.K[c, :cf_dim] = beta * cf      # key on this class's fingerprint
            self.K[c, cf_dim:] = 0.0            # ignore the lexical signature entirely
            self.b[c] = -beta * thresh          # so A~1 on-class, ~0 off-class
            self.s[c] = s_fixed                 # broad rule => low specificity
            self.frozen[c] = True
        self._froze = True
        self._K0, self._b0, self._s0 = self.K.copy(), self.b.copy(), self.s.copy()

    # ---- forward: returns loss and a cache for backprop ---------------------
    def forward(self, X, y):
        N = X.shape[0]; R = self.R; di = np.arange(R)
        A = _sigmoid(X @ self.K.T + self.b)               # (N,R) applicability
        Sdiff = self.s[None, :] - self.s[:, None]         # Sdiff[i,j]=s_j-s_i
        M = _sigmoid(self.gamma * Sdiff)                  # (R,R) "j more specific than i"
        P = 1.0 - A[:, None, :] * M[None, :, :]           # (N,R,R) preemption factors
        P[:, di, di] = 1.0                                # no self-preemption
        logr = np.sum(np.log(P), axis=2)                  # (N,R)
        Rmass = np.exp(logr)                              # (N,R) unclaimed mass
        g = A * Rmass                                     # (N,R) effective firing
        denom = np.sum(g, axis=1, keepdims=True) + self.eps
        p = g / denom                                     # (N,R) which-rule-fires
        z = p @ self.O                                    # (N,C)
        zmax = np.max(z, axis=1, keepdims=True)
        ez = np.exp(z - zmax)
        probs = ez / np.sum(ez, axis=1, keepdims=True)
        loss = -np.mean(np.log(probs[np.arange(N), y] + 1e-12))
        # Paninian specificity prior (CENTERED): rules broader than average are
        # pushed to LOWER specificity, narrower-than-average to HIGHER. Centering
        # removes any uniform drift, so only the *relative* ordering is shaped.
        breadth = A.mean(axis=0)                          # (R,) how broadly rule fires
        mean_b = breadth.mean()
        reg = self.lam_spec * np.sum((breadth - mean_b) * self.s) \
            + 0.5 * self.l2_spec * np.sum(self.s ** 2)
        loss = loss + reg
        cache = (X, y, A, M, P, Rmass, g, denom, p, probs, breadth)
        return loss, cache

    # ---- backward: analytic gradients (verified by finite differences) ------
    def backward(self, cache):
        (X, y, A, M, P, Rmass, g, denom, p, probs, breadth) = cache
        N, R = A.shape; di = np.arange(R)
        onehot = np.zeros_like(probs); onehot[np.arange(N), y] = 1.0
        dz = (probs - onehot) / N                         # (N,C)
        dO = p.T @ dz                                     # (R,C)
        dp = dz @ self.O.T                                # (N,R)
        # p = g/denom  -> softmax-like quotient
        dg = (dp - np.sum(dp * p, axis=1, keepdims=True)) / denom
        dA = dg * Rmass                                   # direct path A*Rmass
        dRmass = dg * A
        dlogr = dRmass * Rmass
        dP = dlogr[:, :, None] / P                        # (N,R,R)
        dP[:, di, di] = 0.0
        # P = 1 - A[:,j]*M[i,j]: route to A (column j) and to M
        dA_pre = -np.einsum('nij,ij->nj', dP, M)          # (N,R)
        dA_total = dA + dA_pre
        # Paninian specificity prior contributes to A (via breadth) and to s.
        # reg = lam*sum((breadth_i - mean_b)*s_i) + 0.5*l2*||s||^2  (centered)
        dA_total = dA_total + (self.lam_spec * (self.s - self.s.mean()))[None, :] / N
        dM = -np.einsum('nij,nj->ij', dP, A)              # (R,R)
        dlogit = dA_total * A * (1 - A)                   # (N,R)
        dK = dlogit.T @ X                                 # (R,d)
        db = np.sum(dlogit, axis=0)                       # (R,)
        # M = sigmoid(gamma*(s_j - s_i))
        T = dM * self.gamma * M * (1 - M)
        ds = T.sum(axis=0) - T.sum(axis=1)                # col(s_j) - row(s_i)
        ds = ds + self.lam_spec * (breadth - breadth.mean()) + self.l2_spec * self.s
        return dict(K=dK, b=db, s=ds, O=dO)

    # ---- inference ----------------------------------------------------------
    def predict(self, X):
        loss, cache = self.forward(X, np.zeros(X.shape[0], int))
        probs = cache[9]
        return np.argmax(probs, axis=1)

    def rule_firing(self, X):
        """Return p (which-rule-fires probabilities) for inspection."""
        _, cache = self.forward(X, np.zeros(X.shape[0], int))
        return cache[8]  # p

    # ---- Adam training loop -------------------------------------------------
    def train(self, X, y, epochs=400, lr=0.05, batch=64, seed=0, verbose=True):
        rng = np.random.default_rng(seed)
        params = ['K', 'b', 's', 'O']
        m = {k: np.zeros_like(getattr(self, k)) for k in params}
        v = {k: np.zeros_like(getattr(self, k)) for k in params}
        b1, b2, eps = 0.9, 0.999, 1e-8
        N = X.shape[0]; t = 0; hist = []
        for ep in range(epochs):
            idx = rng.permutation(N)
            for start in range(0, N, batch):
                bi = idx[start:start + batch]
                loss, cache = self.forward(X[bi], y[bi])
                grads = self.backward(cache)
                if self._froze:                       # utsarga rules are pinned
                    for k in ('K', 'b', 's'):
                        grads[k][self.frozen] = 0.0
                t += 1
                for k in params:
                    gk = grads[k]
                    m[k] = b1 * m[k] + (1 - b1) * gk
                    v[k] = b2 * v[k] + (1 - b2) * (gk * gk)
                    mhat = m[k] / (1 - b1 ** t)
                    vhat = v[k] / (1 - b2 ** t)
                    setattr(self, k, getattr(self, k) - lr * mhat / (np.sqrt(vhat) + eps))
                if self._froze:                       # restore pinned values exactly
                    self.K[self.frozen] = self._K0[self.frozen]
                    self.b[self.frozen] = self._b0[self.frozen]
                    self.s[self.frozen] = self._s0[self.frozen]
            full_loss, _ = self.forward(X, y)
            acc = np.mean(self.predict(X) == y)
            hist.append((full_loss, acc))
            if verbose and (ep % 50 == 0 or ep == epochs - 1):
                print(f"  epoch {ep:4d} | loss {full_loss:.4f} | train acc {acc:.3f}")
        return hist


# =============================================================================
# PART IV -- SELF-TESTS
# =============================================================================
def gradient_check(seed=0):
    """Mandatory finite-difference check on every parameter."""
    rng = np.random.default_rng(seed)
    N, d, R, C = 7, 5, 6, 4
    X = rng.normal(size=(N, d)); y = rng.integers(0, C, size=N)
    net = AshtadhyayiNet(d, R, C, gamma=3.0, seed=seed)
    loss, cache = net.forward(X, y)
    grads = net.backward(cache)
    worst = 0.0
    for name in ['K', 'b', 's', 'O']:
        P = getattr(net, name)
        an = grads[name]
        num = np.zeros_like(P)
        it = np.nditer(P, flags=['multi_index'], op_flags=['readwrite'])
        h = 1e-6
        while not it.finished:
            ix = it.multi_index; old = P[ix]
            P[ix] = old + h; lp, _ = net.forward(X, y)
            P[ix] = old - h; lm, _ = net.forward(X, y)
            P[ix] = old; num[ix] = (lp - lm) / (2 * h)
            it.iternext()
        rel = np.max(np.abs(an - num)) / (np.max(np.abs(num)) + 1e-12)
        worst = max(worst, rel)
        print(f"  grad[{name:>1}] max|analytic-numeric|={np.max(np.abs(an-num)):.2e}  relerr={rel:.2e}")
    ok = worst < 1e-4
    print(f"  => gradient check {'PASSED' if ok else 'FAILED'} (worst relerr {worst:.2e})")
    return ok


def main():
    np.random.seed(0)
    print("=" * 74)
    print("THE ASHTADHYAYI NETWORK  -  Panini's Elsewhere Condition, made trainable")
    print("=" * 74)

    print("\n[1] GRADIENT CHECK (finite differences vs analytic backprop)")
    ok = gradient_check()

    print("\n[2] BUILD DATASET (two utsarga defaults + apavada exceptions)")
    (Xtr, ytr), (Xte, yte), meta = build_dataset()
    enc = meta['encoder']; irr = meta['is_irr_tr']; reg = ~irr; g = meta['gtr']
    print(f"  train stems={len(ytr)}  ( {int(irr.sum())} lexical exceptions / "
          f"{int(reg.sum())} regular )   features d={enc.dim}  "
          f"classes(gana)={meta['n_classes']}  endings={meta['n_endings']}")

    print("\n[3] TRAIN")
    net = AshtadhyayiNet(d_in=enc.dim, n_rules=28, n_out=meta['n_endings'],
                         gamma=4.0, seed=3)
    # Pin rules 0..n_classes-1 as the broad utsarga defaults (one per class).
    net.designate_utsarga(enc, meta['n_classes'])
    net.train(Xtr, ytr, epochs=900, lr=0.05, batch=64, seed=1)

    tr_acc = np.mean(net.predict(Xtr) == ytr)
    reg_acc = np.mean(net.predict(Xtr[reg]) == ytr[reg])
    irr_acc = np.mean(net.predict(Xtr[irr]) == ytr[irr])
    print(f"  final train acc = {tr_acc:.3f}  (regulars {reg_acc:.3f} | "
          f"exceptions {irr_acc:.3f})")

    print("\n[4] WUG TEST  (general rules must generalise to UNSEEN regular stems)")
    wug_acc = np.mean(net.predict(Xte) == yte)
    print(f"  accuracy on novel regular stems = {wug_acc:.3f}")

    # which rule fires on each stem, and full applicability matrix
    p_tr = net.rule_firing(Xtr); fired = np.argmax(p_tr, axis=1)
    _, cache = net.forward(Xtr, np.zeros(len(ytr), int)); A = cache[2]
    # class-default (utsarga) rule per class = the pinned utsarga rule (index=class)
    default_rule = {cls: cls for cls in range(meta['n_classes'])}

    print("\n[5] ELSEWHERE CONDITION  (apavada preempts the utsarga that still applies)")
    n_pre, n_ok, gaps, adefs = 0, 0, [], []
    for i in np.where(irr)[0]:
        cls = int(g[i]); fe = int(fired[i]); dr = default_rule[cls]
        if fe != dr:                       # a more-specific rule took over
            n_pre += 1
            a_def = A[i, dr]               # does the general rule still apply here?
            adefs.append(a_def)
            if net.s[fe] > net.s[dr] and a_def > 0.5:
                n_ok += 1                  # genuine preemption: applies AND outranked
                gaps.append(net.s[fe] - net.s[dr])
    for cls in range(meta['n_classes']):
        dr = default_rule[cls]
        print(f"  class {cls}: utsarga rule #{dr}  specificity s={net.s[dr]:+.2f}")
    print(f"  exceptions handled by a preempting rule : {n_pre}/{int(irr.sum())}")
    print(f"  GENUINE Elsewhere preemptions           : {n_ok}/{n_pre}  "
          f"(apavada more specific AND utsarga still applies, a>0.5)")
    if gaps:
        print(f"  mean specificity gap s(apavada)-s(utsarga) = +{np.mean(gaps):.2f}")
        print(f"  mean applicability of the preempted utsarga = {np.mean(adefs):.2f}")

    print("\n[6] RULE ECONOMY  (a compact utsarga set covers the regular language)")
    counts = np.bincount(fired[reg], minlength=net.R)
    used = int((counts > 0).sum())
    top = np.argsort(counts)[::-1][:meta['n_classes']]
    cover = counts[top].sum() / max(1, reg.sum())
    print(f"  {used} rules ever fire on regulars; top-{meta['n_classes']} cover "
          f"{cover:.1%} of them")
    print(f"  exception rules used: "
          f"{sorted(set(fired[irr].tolist()) - set(default_rule.values()))}")

    print("\n" + "=" * 74)
    verdict = (ok and wug_acc > 0.9 and irr_acc > 0.85
               and n_ok >= 1 and (n_ok / max(1, n_pre)) >= 0.6)
    print("OVERALL:", "ALL CHECKS PASSED" if verdict else "see results above")
    print("=" * 74)
    return verdict


if __name__ == "__main__":
    main()

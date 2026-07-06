#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Figure 62 - Socrates of Athens (470-399 BCE)
The Elenchus Coherence Network (ECN)
 Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 4 Minds 61 - 80 Available on Amazon https://www.amazon.com/dp/B0H71JC95Q
Resume and Interactive Demos at https://artificiology.com/
Author: David Vivancos · Chapter 0062 · Socrates of Athens
================================================================================

WHY THIS ARCHITECTURE, AND NOT A TRANSFORMER

Almost every modern model learns by ADDING. It absorbs labelled examples and
deposits the regularities into its weights, growing a positive store of facts.
Socrates did the opposite. He produced no doctrine. His entire method - the
*elenchus*, or cross-examination - was SUBTRACTIVE. He took the beliefs a person
already held and tested them against one another. When two beliefs could not both
be true, one had to give way. Knowledge, for Socrates, was not what you had piled
up; it was what survived the questioning.

So the central learning signal here is not "predict the label" but "minimise
internal contradiction." The network does not learn a map from input to answer.
It learns a *consistency metric* - a way of telling, for any two propositions,
whether holding both is coherent or self-defeating. Reasoning then becomes a
search for the largest set of beliefs that can be held together without
contradiction. That search is the elenchus, performed mechanically.

Two further Socratic commitments are built in as first-class machinery, not
afterthoughts:

  * EPISTEMIC HUMILITY ("I know that I know nothing"). The model carries an
    explicit calibrated confidence. When a claim lies outside everything it has
    a strong opinion about, it does not fabricate a verdict - it reports aporia
    ("I cannot judge this"). This is the structural cure for the confident
    confabulation that plagues fluent systems.

  * RECOLLECTION (anamnesis). Socrates held that the slave-boy in the *Meno*
    did not RECEIVE geometry; the truth was latent in him and the questioning
    merely cleared away the false beliefs obscuring it. Here, the coherent set
    of beliefs is never inserted from outside - it is RECOVERED by removing the
    contradictory ones, exactly as a coherent subset is revealed by deletion.

CONVENTIONS (kept constant across the whole 1000Minds corpus)
  * Pure NumPy, written from scratch. No autodiff, no frameworks.
  * Analytic gradients verified against finite differences (mandatory check).
  * A real training loop on data the model has never seen at validation time.
  * Self-tests with assertions. The file runs end to end and prints results.

The differentiable core (the consistency metric) is trained by back-propagation.
The reasoning layer on top of it (the elenchus search, the humility gate) is a
deterministic algorithm that USES the trained metric - mirroring how Socrates'
method is an algorithm applied to whatever beliefs the metric flags as opposed.
================================================================================
"""

import numpy as np

RNG = np.random.default_rng(470)  # Socrates' birth year, as a seed


# ==============================================================================
# 0. SMALL UTILITIES
# ==============================================================================

def tanh(x):
    return np.tanh(x)


def dtanh_from_output(t):
    """Derivative of tanh expressed through its OUTPUT t = tanh(x): 1 - t^2."""
    return 1.0 - t * t


# ==============================================================================
# 1. THE DIFFERENTIABLE CORE: a learned consistency metric
# ==============================================================================
#
# A proposition arrives as a feature vector x in R^d_in (a hashed / bag-of-
# concepts encoding). Two small layers lift it into a "belief embedding" b:
#
#     h = tanh(W1 x + b1)        # hidden moral/semantic features
#     b = W2 h + b2              # belief embedding in R^d_emb
#
# For a PAIR of propositions (x_a, x_b) the model asks the Socratic question:
# "can both be held at once?" It answers with a bilinear consistency score
#
#     s = b_a^T M b_b            # M is a learned consistency operator
#     p = tanh(s) in [-1, 1]     # +1 = coherent, -1 = contradictory
#
# Training minimises the gap between p and a ground-truth coherence label.
# Crucially the model is NEVER told "what justice is" - only which pairs of
# claims sit together without contradiction. The positive content is emergent.
# ==============================================================================

class ConsistencyMetric:
    def __init__(self, d_in, d_hidden, d_emb, reg=1e-4, seed=None):
        rng = np.random.default_rng(seed) if seed is not None else RNG

        def xavier(shape):
            fan = shape[0] + shape[1]
            lim = np.sqrt(6.0 / fan)
            return rng.uniform(-lim, lim, size=shape)

        self.W1 = xavier((d_hidden, d_in))
        self.b1 = np.zeros(d_hidden)
        self.W2 = xavier((d_emb, d_hidden))
        self.b2 = np.zeros(d_emb)
        # M starts near identity: "assume claims cohere until shown otherwise" -
        # the charitable reading Socrates always began the elenchus with.
        self.M = np.eye(d_emb) + 0.01 * rng.standard_normal((d_emb, d_emb))
        self.reg = reg
        self.d_in, self.d_hidden, self.d_emb = d_in, d_hidden, d_emb

    # ---- parameter (un)packing, used only by the gradient check -------------
    def get_params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.M]

    def set_params(self, params):
        self.W1, self.b1, self.W2, self.b2, self.M = [p.copy() for p in params]

    # ---- embed one batch of propositions ------------------------------------
    def embed(self, X):
        """X: (N, d_in) -> embedding B (N, d_emb) plus cache for backprop."""
        Z1 = X @ self.W1.T + self.b1          # (N, d_hidden)
        H = tanh(Z1)                           # (N, d_hidden)
        B = H @ self.W2.T + self.b2            # (N, d_emb)
        return B, (X, H)

    # ---- consistency score for paired propositions --------------------------
    def forward(self, Xa, Xb):
        """Return p (N,) consistency predictions in [-1,1] and a cache."""
        Ba, ca = self.embed(Xa)
        Bb, cb = self.embed(Xb)
        BaM = Ba @ self.M                      # (N, d_emb)
        S = np.sum(BaM * Bb, axis=1)           # (N,)  s_i = Ba_i M Bb_i
        P = tanh(S)                            # (N,)
        cache = (Xa, Xb, ca, cb, Ba, Bb, S, P)
        return P, cache

    # ---- loss: mean squared error on coherence + L2 weight decay ------------
    def loss(self, Xa, Xb, y):
        P, cache = self.forward(Xa, Xb)
        mse = 0.5 * np.mean((P - y) ** 2)
        l2 = 0.5 * self.reg * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2)
                               + np.sum(self.M ** 2))
        return mse + l2, cache, P

    # ---- analytic gradients --------------------------------------------------
    def backward(self, cache, y):
        Xa, Xb, ca, cb, Ba, Bb, S, P = cache
        Xa_, Ha = ca
        Xb_, Hb = cb
        N = len(y)

        gP = (P - y) / N                       # dL/dP for mean-squared error
        gS = gP * dtanh_from_output(P)         # through tanh

        # s_i = Ba_i M Bb_i
        gBa = gS[:, None] * (Bb @ self.M.T)    # (N, d_emb)
        gBb = gS[:, None] * (Ba @ self.M)      # (N, d_emb)
        gM = (Ba * gS[:, None]).T @ Bb         # (d_emb, d_emb)

        # B = H W2^T + b2  (two branches share W2, b2)
        gW2 = gBa.T @ Ha + gBb.T @ Hb          # (d_emb, d_hidden)
        gb2 = gBa.sum(0) + gBb.sum(0)          # (d_emb,)
        gHa = gBa @ self.W2                    # (N, d_hidden)
        gHb = gBb @ self.W2

        # H = tanh(Z1),  Z1 = X W1^T + b1
        gZ1a = gHa * dtanh_from_output(Ha)
        gZ1b = gHb * dtanh_from_output(Hb)
        gW1 = gZ1a.T @ Xa_ + gZ1b.T @ Xb_      # (d_hidden, d_in)
        gb1 = gZ1a.sum(0) + gZ1b.sum(0)        # (d_hidden,)

        # L2 regularisation gradients (biases are not decayed)
        gW1 += self.reg * self.W1
        gW2 += self.reg * self.W2
        gM += self.reg * self.M
        return [gW1, gb1, gW2, gb2, gM]

    # ---- one Adam optimisation step -----------------------------------------
    def init_adam(self):
        self._m = [np.zeros_like(p) for p in self.get_params()]
        self._v = [np.zeros_like(p) for p in self.get_params()]
        self._t = 0

    def adam_step(self, grads, lr=2e-2, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        for i, (p, g) in enumerate(zip(self.get_params(), grads)):
            self._m[i] = b1 * self._m[i] + (1 - b1) * g
            self._v[i] = b2 * self._v[i] + (1 - b2) * (g * g)
            mhat = self._m[i] / (1 - b1 ** self._t)
            vhat = self._v[i] / (1 - b2 ** self._t)
            p -= lr * mhat / (np.sqrt(vhat) + eps)  # in-place; arrays are shared

    # ---- public: consistency in [-1,1] for a single pair --------------------
    def consistency(self, xa, xb):
        P, _ = self.forward(xa[None, :], xb[None, :])
        return float(P[0])


# ==============================================================================
# 2. GRADIENT CHECK  (mandatory for every file in the corpus)
# ==============================================================================
#
# Socrates trusted no claim he had not cross-examined. We extend him the same
# courtesy: the analytic gradient is examined against a finite-difference
# estimate. If they disagree, the gradient is "mere opinion" and we say so.
# ==============================================================================

def gradient_check(verbose=True):
    d_in, d_hidden, d_emb, N = 7, 9, 6, 5
    model = ConsistencyMetric(d_in, d_hidden, d_emb, reg=1e-3, seed=1)
    Xa = RNG.standard_normal((N, d_in))
    Xb = RNG.standard_normal((N, d_in))
    y = RNG.choice([-1.0, 1.0], size=N)

    _, cache, _ = model.loss(Xa, Xb, y)
    analytic = model.backward(cache, y)

    eps = 1e-5
    max_rel = 0.0
    for pi, P in enumerate(model.get_params()):
        it = np.nditer(P, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            orig = P[idx]
            P[idx] = orig + eps
            lp, _, _ = model.loss(Xa, Xb, y)
            P[idx] = orig - eps
            lm, _, _ = model.loss(Xa, Xb, y)
            P[idx] = orig
            num = (lp - lm) / (2 * eps)
            ana = analytic[pi][idx]
            denom = max(1e-9, abs(num) + abs(ana))
            max_rel = max(max_rel, abs(num - ana) / denom)
            it.iternext()
    if verbose:
        print(f"  max relative gradient error : {max_rel:.3e}")
        print(f"  verdict                     : "
              f"{'PASS (analytic == numeric)' if max_rel < 1e-5 else 'FAIL'}")
    assert max_rel < 1e-5, "Gradient check failed - the gradient is mere opinion."
    return max_rel


# ==============================================================================
# 3. A SYNTHETIC MORAL WORLD  (so the elenchus has something to examine)
# ==============================================================================
#
# Each "concept" has a hidden polarity vector in a few latent moral axes. Two
# concepts are CONSISTENT (label +1) when their polarities point the same way
# and CONTRADICTORY (-1) when they oppose. The model never sees the polarities -
# only the noisy feature mixing A @ polarity - and must recover the consistency
# relation from examples. This is the learnable analogue of Socrates inferring,
# from many particular cases, the general shape of justice, piety, courage.
# ==============================================================================

class MoralWorld:
    def __init__(self, k_axes=3, d_in=24, n_concepts=60, seed=0):
        rng = np.random.default_rng(seed)
        self.k = k_axes
        self.d_in = d_in
        self.A = rng.standard_normal((d_in, k_axes))      # fixed mixing matrix
        self.polarities = rng.standard_normal((n_concepts, k_axes))
        noise = 0.25 * rng.standard_normal((n_concepts, d_in))
        self.X = self.polarities @ self.A.T + noise       # observed features
        self.rng = rng

    def label(self, i, j):
        return 1.0 if float(self.polarities[i] @ self.polarities[j]) >= 0 else -1.0

    def make_pairs(self, n_pairs):
        n = len(self.X)
        ia = self.rng.integers(0, n, size=n_pairs)
        ib = self.rng.integers(0, n, size=n_pairs)
        y = np.array([self.label(a, b) for a, b in zip(ia, ib)])
        return self.X[ia], self.X[ib], y

    def coherent_cluster(self, n_virtues=4, n_vices=2):
        """Pick concepts that genuinely cohere, plus genuinely opposing ones.

        The 'virtues' are the mutually most-aligned concepts (a real coherent
        belief cluster); the 'vices' are concepts that point against that
        cluster's centroid. This makes the elenchus demo read truthfully:
        the coherent core survives, the opposing claims are refuted.
        """
        P = self.polarities
        Pn = P / np.clip(np.linalg.norm(P, axis=1, keepdims=True), 1e-9, None)
        # seed on the pair with the highest alignment, then greedily grow
        G = Pn @ Pn.T
        np.fill_diagonal(G, -np.inf)
        i, j = np.unravel_index(np.argmax(G), G.shape)
        cluster = [int(i), int(j)]
        while len(cluster) < n_virtues:
            scores = Pn @ Pn[cluster].mean(0)
            scores[cluster] = -np.inf
            cluster.append(int(np.argmax(scores)))
        centroid = Pn[cluster].mean(0)
        anti = np.argsort(Pn @ centroid)[:n_vices]   # most opposed to the core
        return cluster, [int(a) for a in anti]


def train_consistency_metric(verbose=True):
    world = MoralWorld(k_axes=3, d_in=24, n_concepts=80, seed=7)
    model = ConsistencyMetric(d_in=24, d_hidden=32, d_emb=12, reg=1e-4, seed=3)
    model.init_adam()

    Xa_tr, Xb_tr, y_tr = world.make_pairs(2000)
    Xa_va, Xb_va, y_va = world.make_pairs(600)

    def accuracy(Xa, Xb, y):
        P, _ = model.forward(Xa, Xb)
        return float(np.mean(np.sign(P) == np.sign(y)))

    if verbose:
        print(f"  start    : loss={model.loss(Xa_tr, Xb_tr, y_tr)[0]:.4f}  "
              f"val_acc={accuracy(Xa_va, Xb_va, y_va):.3f}")

    epochs, bs, n = 80, 128, len(y_tr)
    base_lr = 5e-3
    for ep in range(epochs):
        # gentle cosine decay keeps the late epochs from over-shooting the basin
        lr = base_lr * 0.5 * (1 + np.cos(np.pi * ep / epochs))
        perm = RNG.permutation(n)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            _, cache, _ = model.loss(Xa_tr[idx], Xb_tr[idx], y_tr[idx])
            grads = model.backward(cache, y_tr[idx])
            model.adam_step(grads, lr=lr)
        if verbose and (ep + 1) % 20 == 0:
            print(f"  epoch {ep+1:2d}: loss={model.loss(Xa_tr, Xb_tr, y_tr)[0]:.4f}  "
                  f"val_acc={accuracy(Xa_va, Xb_va, y_va):.3f}")

    val_acc = accuracy(Xa_va, Xb_va, y_va)
    if verbose:
        print(f"  final    : val_acc={val_acc:.3f}")
    return model, world, val_acc


# ==============================================================================
# 4. THE ELENCHUS ENGINE  (Socratic reasoning ON TOP of the learned metric)
# ==============================================================================
#
# Given a knowledge base of beliefs, the engine:
#   (a) builds the pairwise consistency matrix using the trained metric;
#   (b) finds the MAXIMAL COHERENT SUBSET by repeatedly removing the belief that
#       participates in the most contradictions (the one that "gives way");
#   (c) reports which beliefs were refuted and why.
#
# Step (b) is the elenchus made literal. It also realises anamnesis: the
# coherent truth is not added - it is what remains once falsehood is cleared.
# ==============================================================================

class ElenchusEngine:
    def __init__(self, model, contradiction_threshold=-0.35):
        self.model = model
        self.tau = contradiction_threshold

    def consistency_matrix(self, X):
        n = len(X)
        C = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                c = self.model.consistency(X[i], X[j])
                C[i, j] = C[j, i] = c
        return C

    def maximal_coherent_subset(self, X, names=None):
        """Greedy elenchus: remove the most-contradicted belief until coherent."""
        n = len(X)
        names = names or [f"belief_{i}" for i in range(n)]
        C = self.consistency_matrix(X)
        alive = list(range(n))
        removed = []
        while True:
            contra = {i: sum(1 for j in alive if j != i and C[i, j] < self.tau)
                      for i in alive}
            worst = max(contra, key=lambda i: contra[i])
            if contra[worst] == 0:
                break  # no contradictions remain -> coherent set found
            alive.remove(worst)
            removed.append((names[worst], contra[worst]))
        kept = [names[i] for i in alive]
        return kept, removed, C

    def examine(self, X_kb, x_claim, names=None):
        """Cross-examine a NEW claim against an existing coherent KB."""
        n = len(X_kb)
        names = names or [f"belief_{i}" for i in range(n)]
        cons = np.array([self.model.consistency(x_claim, X_kb[i]) for i in range(n)])
        contradicted = [i for i in range(n) if cons[i] < self.tau]
        return {
            "consistencies": cons,
            "contradicts": [names[i] for i in contradicted],
            "coherent": len(contradicted) == 0,
            "verdict": ("coheres with current beliefs" if not contradicted
                        else "produces aporia: contradicts " +
                             ", ".join(names[i] for i in contradicted)),
        }


# ==============================================================================
# 5. THE HUMILITY GATE  ("I know that I know nothing")
# ==============================================================================
#
# Calibrated confidence. For a claim, the model measures the strongest opinion it
# can form about it relative to the knowledge base. If every consistency score is
# near zero, the claim lies OUTSIDE the model's competence: it has no business
# rendering a verdict. Rather than confabulate, it reports aporia. This is the
# architectural form of Socrates' wisdom: knowing the boundary of one's knowing.
# ==============================================================================

class HumilityGate:
    """
    Calibrated confidence via DISTANCE FROM THE MORAL MANIFOLD.

    A bilinear consistency score saturates through tanh and is over-confident by
    construction - it hands back a strong +1/-1 even for a claim it has never
    seen. That is the confabulation Socrates would refuse, so the gate does not
    read confidence off the score.

    The model only ever studied claims drawn from one manifold of moral
    experience. The gate learns the shape of that manifold (its principal
    directions) and, for any new claim, measures how far it falls OUTSIDE it -
    the part of the claim the model has no trained dimension to read. The
    competence threshold is CALIBRATED from the in-distribution residuals, not
    hand-tuned. A claim further off-manifold than the model's own studied
    concepts is met with "I cannot judge this." Knowing the boundary of one's
    knowledge, made quantitative.
    """

    def __init__(self, model, var_kept=0.95, percentile=92.0):
        self.model = model
        self.var_kept = var_kept
        self.percentile = percentile
        self.mean = None
        self.V = None            # principal directions of the moral manifold
        self.theta = None        # residual beyond which we withhold judgement

    def fit(self, X_known):
        self.mean = X_known.mean(0)
        Xc = X_known - self.mean
        # principal directions via SVD of the centred concept matrix
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        var = (S ** 2) / np.sum(S ** 2)
        r = int(np.searchsorted(np.cumsum(var), self.var_kept) + 1)
        self.V = Vt[:r].T                                  # (d_in, r)
        res = self._residual(X_known)
        self.theta = float(np.percentile(res, self.percentile))
        return self.theta, r

    def _residual(self, X):
        Xc = X - self.mean
        recon = (Xc @ self.V) @ self.V.T                  # project then lift back
        return np.linalg.norm(Xc - recon, axis=1)         # off-manifold distance

    def judge(self, x_claim):
        res = float(self._residual(x_claim[None, :])[0])
        # report a 0..1 confidence: 1 = on-manifold, ->0 as residual grows
        conf = float(np.exp(-res / max(self.theta, 1e-9)))
        if res > self.theta:
            return {"confidence": conf, "residual": res, "acts": False,
                    "statement": "I cannot judge this; it lies beyond what I know."}
        return {"confidence": conf, "residual": res, "acts": True,
                "statement": "I have grounds to examine this claim."}


# ==============================================================================
# 6. SELF-TESTS  (the file cross-examines itself before shipping)
# ==============================================================================

def run_self_tests(model, world):
    print("\n" + "=" * 70)
    print("SELF-TESTS")
    print("=" * 70)

    # --- Test A: the metric agrees with ground-truth coherence on fresh pairs
    Xa, Xb, y = world.make_pairs(500)
    P, _ = model.forward(Xa, Xb)
    acc = float(np.mean(np.sign(P) == np.sign(y)))
    print(f"A. metric vs. ground-truth coherence (unseen): acc={acc:.3f}")
    assert acc > 0.80, "consistency metric did not learn the relation"

    # --- Test B: the elenchus actually removes contradictions
    virtues, vices = world.coherent_cluster(n_virtues=4, n_vices=2)
    idx = virtues + vices
    Xkb = world.X[idx]
    names = [f"virtue_{i}" for i in range(len(virtues))] + \
            [f"vice_{i}" for i in range(len(vices))]
    eng = ElenchusEngine(model, contradiction_threshold=-0.35)
    kept, removed, C = eng.maximal_coherent_subset(Xkb, names)
    print(f"B. elenchus on mixed set: kept={kept}")
    print(f"   refuted (belief, #contradictions)={removed}")
    rem_idx = [names.index(k) for k in kept]
    sub = C[np.ix_(rem_idx, rem_idx)]
    assert (sub + np.eye(len(rem_idx))).min() > -0.35, "kept set still incoherent"
    assert len(kept) >= len(removed), "the elenchus discarded the coherent core"
    print("   -> a coherent core survived; the opposing claims were refuted  [OK]")

    # --- Test C: humility gate flags genuinely foreign claims
    gate = HumilityGate(model, var_kept=0.95, percentile=92.0)
    theta, r = gate.fit(world.X)            # learn the moral manifold & calibrate
    foreign = MoralWorld(k_axes=3, d_in=world.d_in, n_concepts=40, seed=999)
    in_acts = np.mean([gate.judge(world.X[i])["acts"] for i in range(len(world.X))])
    out_acts = np.mean([gate.judge(foreign.X[i])["acts"] for i in range(len(foreign.X))])
    print(f"C. humility (manifold rank={r}, theta={theta:.2f}): acts on "
          f"{in_acts*100:.0f}% of known concepts, on {out_acts*100:.0f}% of foreign")
    assert in_acts > 0.85, "gate withholds on its own knowledge"
    assert out_acts < 0.30, "gate fails to withhold on the foreign"
    print("   -> confident on the known, withholds on the off-manifold  [OK]")

    # --- Test D: examining a contradictory new claim yields aporia
    Xc = world.X[virtues]
    cnames = [f"c{i}" for i in range(len(virtues))]
    res = eng.examine(Xc, world.X[vices[0]], cnames)
    print(f"D. examine opposed claim against coherent KB: coherent={res['coherent']}")
    assert not res["coherent"], "elenchus failed to detect a real contradiction"
    print(f"   -> {res['verdict']}  [OK]")

    print("\nALL SELF-TESTS PASSED")


# ==============================================================================
# 7. MAIN
# ==============================================================================

def main():
    print("=" * 70)
    print("THE ELENCHUS COHERENCE NETWORK  -  Socrates of Athens")
    print("Learning by the removal of contradiction")
    print("=" * 70)

    print("\n[1] Gradient check (cross-examining the gradient):")
    gradient_check(verbose=True)

    print("\n[2] Training the consistency metric on a synthetic moral world:")
    model, world, val_acc = train_consistency_metric(verbose=True)

    print("\n[3] A worked elenchus")
    print("-" * 70)
    virtues, vices = world.coherent_cluster(n_virtues=4, n_vices=2)
    idx = virtues + vices
    Xkb = world.X[idx]
    labels = ["justice", "piety", "courage", "temperance", "tyranny", "cowardice"]
    eng = ElenchusEngine(model, contradiction_threshold=-0.35)
    kept, removed, _ = eng.maximal_coherent_subset(Xkb, labels)
    print(f"  beliefs entering examination : {labels}")
    print(f"  survive the elenchus         : {kept}")
    print(f"  refuted and set aside        : {[r[0] for r in removed]}")
    print("  (the coherent remainder was not inserted; it is what was left")
    print("   standing once the contradictory beliefs gave way - anamnesis")
    print("   by subtraction.)")

    run_self_tests(model, world)

    print("\n" + "=" * 70)
    print("Socrates: 'The unexamined life is not worth living.'")
    print("Here: the unexamined belief is not worth keeping.")
    print("=" * 70)


if __name__ == "__main__":
    main()

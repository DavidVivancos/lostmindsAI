"""
================================================================================
chapter_0067_antisthenes_-446.py  --  filename to ship: Neuron.py
================================================================================
Figure #67 - Antisthenes of Athens (c. 446-366 BCE)
 Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 4 Minds 61 - 80 Available on Amazon https://www.amazon.com/dp/B0H71JC95Q
Resume and Interactive Demos at https://artificiology.com/
Author: David Vivancos · Chapter 0067 · Antisthenes of Athens
Pupil of Socrates; founder of the Cynic line; the first systematic nominalist.

WHAT THIS FILE IS
-----------------
A real, trainable, from-scratch neural architecture (pure NumPy, no autograd
frameworks) that encodes ONE cognitive signature that is Antisthenes' alone --
not the generic "virtue is order" lens, but the precise doctrine that made him
the enemy of Plato and the grandfather of nominalism:

    "I see the horse, but I do not see horseness."
        (Simplicius, In Aristotelis Categorias 208.28-32)

From that one sentence three Antisthenic commitments follow, and each becomes a
concrete mechanism below:

  (1) ANTI-UNIVERSAL / LOCALIST CODING.
      There is no shared abstract space where "horseness" lives. Knowledge is
      the possession of each particular's OWN proper account (oikeios logos --
      "one thing, one name"). So this network refuses a single distributed
      embedding. Instead it holds a growing set of *accounts*: each is a
      dedicated, localized kernel (an anisotropic radial basis) that fires only
      for the particular it is acquainted with. Generalization is exemplar-based,
      never the averaging of a universal.

  (2) NO-CONTRADICTION ROUTING  (ouk estin antilegein -- Plato, Sophist 251-252).
      Antisthenes held that contradiction is *impossible*: to speak of a thing is
      to speak its own account; two speakers who disagree are simply naming two
      different things. We make that a routing rule. A novel input that matches
      no existing account is NOT forced into one (which would corrupt that
      account -- a "contradiction"). It is given its OWN new account. Disagreement
      is resolved by reference-splitting, never by overwriting a held account.

  (3) ASKESIS  -- virtue is teachable and, once acquired, cannot be lost
      (Diogenes Laertius VI.10-11; "wisdom is the most secure wall", VI.13).
      Each account carries a consolidation mass that grows with confirmed use.
      As an account hardens: (a) its learning rate decays toward zero, and (b) an
      elastic anchor penalty pins it to its trained value. A consolidated account
      is therefore protected from catastrophic forgetting when new things are
      learned later -- a trained competence that is not lost. (autarkeia / the
      self-sufficient wall, realized as continual-learning stability.)

  (4) AUTARKEIA  -- the rejection of false needs. The network adds an account
      ONLY when a genuinely novel particular demands one (a novelty threshold),
      and an L2 thrift term discourages superfluous reliance. Minimal apparatus,
      maximal sufficiency.

WHY THIS IS NOT A TRANSFORMER
-----------------------------
Attention-over-stored-keys *averages* abstract value vectors -- it is exactly the
Platonic move Antisthenes denied. Here there is no key/query abstraction and no
shared value space. Each unit is one named particular; prediction is a sparse
sum of acquaintance responses. The architecture is the philosophical inverse of
its nearest neighbours in the corpus (Socrates = elenchus/dialogue; Plato =
Forms / abstract prototypes). Antisthenes is the anti-Plato, so the net is the
anti-prototype net.

CORRECTNESS CONTRACT (enforced at bottom of file)
-------------------------------------------------
  * analytic gradients vs. finite differences  (mandatory grad check, must pass)
  * a real training loop with measured accuracy
  * a continual-learning experiment proving askesis preserves old "virtue"
  * a reference-split experiment proving no-contradiction growth
  * deterministic self-tests with assertions

Run:  python3 chapter_0067_antisthenes_-446.py
================================================================================
"""

from __future__ import annotations

import numpy as np

# A single global seed makes every demo and the grad check reproducible.
SEED = 67  # the figure's index, for luck and bookkeeping.


# ============================================================================
# SECTION 1 -- SMALL DIFFERENTIABLE PRIMITIVES
# ----------------------------------------------------------------------------
# softplus / sigmoid are used so the per-dimension precision (beta) of an
# account stays strictly positive while R (its raw parameter) ranges freely.
# ============================================================================

def softplus(z):
    """beta = softplus(R) > 0. Numerically stable form."""
    return np.logaddexp(0.0, z)


def sigmoid(z):
    """d softplus / dR. Stable piecewise evaluation."""
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def softmax(logits):
    """Row-wise softmax with the usual max-subtraction for stability."""
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ============================================================================
# SECTION 2 -- THE OIKEIOS LOGOS NETWORK
# ----------------------------------------------------------------------------
# Parameters (the only things gradients touch):
#   M : (K, d)  account means        -- the "proper name" location of each thing
#   R : (K, d)  raw log-precision     -- beta = softplus(R) is per-dim sharpness
#   W : (K, C)  account -> class evidence
#   b : (C,)    class bias
#
# Buffers (controllers; NOT differentiated):
#   M_anchor : (K, d)  the consolidated ("hardened") value an account is pinned to
#   cmass    : (K,)    consolidation mass  -- how trained/"virtuous" an account is
#
# Forward for a batch X (N, d):
#   diff[n,k,j] = X[n,j] - M[k,j]
#   d2[n,k]     = sum_j beta[k,j] * diff[n,k,j]^2        (anisotropic distance)
#   a[n,k]      = exp(-d2[n,k])                          (acquaintance in (0,1])
#   logits      = a @ W + b
#   p           = softmax(logits)
# Differentiable loss = cross-entropy + askesis anchor + autarkeia thrift(L2 W).
# ============================================================================

class OikeiosLogosNet:
    def __init__(self, d, n_classes, max_accounts=64,
                 lam_anchor=2.0, l2_w=1e-4, init_logprec=0.0, rng=None):
        self.d = d
        self.C = n_classes
        self.max_accounts = max_accounts
        self.lam_anchor = lam_anchor    # askesis strength (anchor penalty)
        self.l2_w = l2_w                # autarkeia thrift
        self.init_logprec = init_logprec
        self.rng = rng if rng is not None else np.random.default_rng(SEED)

        self.K = 0
        self.M = np.zeros((0, d))
        self.R = np.zeros((0, d))
        self.W = np.zeros((0, n_classes))
        self.b = np.zeros(n_classes)
        self.M_anchor = np.zeros((0, d))
        self.W_anchor = np.zeros((0, n_classes))
        self.cmass = np.zeros(0)

    # ----- account birth: one new proper name for a genuinely new particular --
    def grow_account(self, x, y=None):
        """Spawn a dedicated account centred on particular x.
        This is the no-contradiction move: a new thing gets its own name rather
        than being crammed into (and thereby corrupting) an existing account."""
        if self.K >= self.max_accounts:
            return -1
        self.M = np.vstack([self.M, x.reshape(1, -1).copy()])
        self.R = np.vstack([self.R, np.full((1, self.d), self.init_logprec)])
        w_row = self.rng.normal(0, 0.01, (1, self.C))
        if y is not None:
            w_row[0, y] += 0.5  # a gentle first acquaintance with its class
        self.W = np.vstack([self.W, w_row])
        self.M_anchor = np.vstack([self.M_anchor, x.reshape(1, -1).copy()])
        self.W_anchor = np.vstack([self.W_anchor, w_row.copy()])
        self.cmass = np.concatenate([self.cmass, [0.0]])
        self.K += 1
        return self.K - 1

    # ----- forward, returning everything backward() needs ---------------------
    def forward(self, X):
        beta = softplus(self.R)                       # (K,d) > 0
        diff = X[:, None, :] - self.M[None, :, :]     # (N,K,d)
        sq = diff * diff                              # (N,K,d)
        d2 = np.einsum('kj,nkj->nk', beta, sq)        # (N,K)
        a = np.exp(-d2)                               # (N,K) acquaintance
        logits = a @ self.W + self.b                  # (N,C)
        p = softmax(logits)
        cache = dict(X=X, beta=beta, diff=diff, sq=sq, a=a, p=p)
        return p, cache

    # ----- the full differentiable loss --------------------------------------
    def loss(self, X, y):
        p, cache = self.forward(X)
        N = X.shape[0]
        ce = -np.mean(np.log(p[np.arange(N), y] + 1e-12))
        # askesis: pin hardened accounts (both their location AND their class
        # mapping) to the consolidated value -- a trained competence held fast.
        anchor = 0.5 * self.lam_anchor * (
            np.sum(self.cmass[:, None] * (self.M - self.M_anchor) ** 2) +
            np.sum(self.cmass[:, None] * (self.W - self.W_anchor) ** 2))
        thrift = 0.5 * self.l2_w * np.sum(self.W * self.W)   # autarkeia
        total = ce + anchor / max(N, 1) + thrift
        cache['y'] = y
        cache['N'] = N
        return total, cache

    # ----- analytic backward -------------------------------------------------
    def backward(self, cache):
        X, beta, diff, sq, a, p = (cache['X'], cache['beta'], cache['diff'],
                                   cache['sq'], cache['a'], cache['p'])
        y, N = cache['y'], cache['N']

        dlogits = p.copy()
        dlogits[np.arange(N), y] -= 1.0
        dlogits /= N                                   # (N,C)

        gW = a.T @ dlogits + self.l2_w * self.W        # (K,C)
        gW += self.lam_anchor * (self.cmass[:, None] *
                                 (self.W - self.W_anchor)) / max(N, 1)
        gb = dlogits.sum(axis=0)                       # (C,)

        g_a = dlogits @ self.W.T                        # (N,K)
        g_d2 = g_a * (-a)                               # da/dd2 = -a  -> (N,K)

        # d d2/dM = -2 beta diff ;  d d2/dbeta = sq
        S = np.einsum('nk,nkj->kj', g_d2, diff)         # (K,d)
        gM = -2.0 * beta * S
        gM += self.lam_anchor * (self.cmass[:, None] *
                                 (self.M - self.M_anchor)) / max(N, 1)

        gbeta = np.einsum('nk,nkj->kj', g_d2, sq)       # (K,d)
        gR = gbeta * sigmoid(self.R)                    # chain through softplus

        return dict(M=gM, R=gR, W=gW, b=gb)

    # ----- prediction helpers -------------------------------------------------
    def predict(self, X):
        p, _ = self.forward(X)
        return p.argmax(axis=1)

    def accuracy(self, X, y):
        return float(np.mean(self.predict(X) == y))

    def max_acquaintance(self, X):
        """Per-example best acquaintance: how well any held account knows x."""
        _, cache = self.forward(X)
        return cache['a'].max(axis=1)


# ============================================================================
# SECTION 3 -- TRAINING WITH ASKESIS
# ----------------------------------------------------------------------------
# Per-slot learning rates DECAY as cmass grows: 1/(1+gamma*cmass). A hardened
# account barely moves -- "virtue, once acquired, cannot be lost."  W and b learn
# at the base rate. Growth (no-contradiction) and consolidation (askesis) are
# the controllers; the gradient itself is the analytic one checked above.
# ============================================================================

def train(net, X, y, *, epochs=60, lr=0.2, batch=32, gamma=4.0,
          novelty_tau=0.30, consolidate=True, rng=None, verbose=False):
    rng = rng if rng is not None else np.random.default_rng(SEED + 1)
    N = X.shape[0]

    # cold start: seed one account per class from a random member of that class
    if net.K == 0:
        for c in np.unique(y):
            idx = rng.choice(np.where(y == c)[0])
            net.grow_account(X[idx], int(c))

    for ep in range(epochs):
        order = rng.permutation(N)
        for s in range(0, N, batch):
            bi = order[s:s + batch]
            Xb, yb = X[bi], y[bi]

            # (no-contradiction) grow accounts for unrecognized particulars
            amax = net.max_acquaintance(Xb)
            for j, am in enumerate(amax):
                if am < novelty_tau:
                    net.grow_account(Xb[j], int(yb[j]))

            _, cache = net.loss(Xb, yb)
            g = net.backward(cache)

            # per-account askesis: hardened accounts get a smaller step for
            # their location (M,R) AND their class mapping (W). The global bias
            # is braked as the whole system matures, so a later task cannot
            # simply re-weight the classes an old account already learned.
            lr_slot = lr / (1.0 + gamma * net.cmass)      # (K,)
            net.M -= lr_slot[:, None] * g['M']
            net.R -= lr_slot[:, None] * g['R']
            net.W -= lr_slot[:, None] * g['W']
            lr_b = lr / (1.0 + gamma * net.cmass.mean()) if net.K else lr
            net.b -= lr_b * g['b']

        # (askesis) consolidate accounts that did confirmed, correct work
        if consolidate:
            p, cache = net.forward(X)
            pred = p.argmax(axis=1)
            resp = cache['a'].argmax(axis=1)              # responsible account
            for k in range(net.K):
                served = (resp == k) & (pred == y)        # correct & responsible
                net.cmass[k] += 0.05 * served.sum()
            net.M_anchor = net.M.copy()                   # harden at current value
            net.W_anchor = net.W.copy()

        if verbose and (ep % 15 == 0 or ep == epochs - 1):
            print(f"      epoch {ep:3d}  acc={net.accuracy(X, y):.3f}  "
                  f"accounts={net.K}  mean_cmass={net.cmass.mean():.2f}")
    return net


# ============================================================================
# SECTION 4 -- SYNTHETIC "PARTICULARS"
# ----------------------------------------------------------------------------
# Each class is a cloud of particulars (Gaussian blobs). There is no Platonic
# form to recover -- only clouds of individuals the net must learn to name.
# ============================================================================

def make_particulars(classes, d=6, per_class=120, spread=0.55, span=4.0,
                     rng=None, centres=None):
    """Sample clouds of particulars. If `centres` is given, reuse those exact
    class locations (so a test set can share the training set's geometry);
    otherwise draw fresh centres and return them."""
    rng = rng if rng is not None else np.random.default_rng(SEED + 2)
    if centres is None:
        centres = {c: rng.uniform(-span, span, d) for c in classes}
    X, y = [], []
    for c in classes:
        X.append(centres[c] + rng.normal(0, spread, (per_class, d)))
        y += [c] * per_class
    X = np.vstack(X)
    y = np.array(y)
    perm = rng.permutation(len(y))
    return X[perm], y[perm], centres


# ============================================================================
# SECTION 5 -- GRADIENT CHECK (mandatory; must pass)
# ----------------------------------------------------------------------------
# Compares analytic grads of (CE + anchor + thrift) against central finite
# differences for every parameter block. Controllers (growth, consolidation)
# are frozen during the check so the loss is a smooth function of the params.
# ============================================================================

def gradient_check(verbose=True):
    rng = np.random.default_rng(SEED + 3)
    d, C = 4, 3
    net = OikeiosLogosNet(d, C, lam_anchor=1.7, l2_w=3e-3, rng=rng)
    X = rng.normal(0, 1, (10, d))
    y = rng.integers(0, C, size=10)
    for _ in range(5):                       # a few accounts to exercise all params
        i = rng.integers(0, 10)
        net.grow_account(X[i], int(y[i]))
    net.M += rng.normal(0, 0.3, net.M.shape)         # move off the anchors
    net.R += rng.normal(0, 0.3, net.R.shape)
    net.W += rng.normal(0, 0.3, net.W.shape)
    net.cmass += rng.uniform(0.1, 0.6, net.K)        # exercise the anchor term

    _, cache = net.loss(X, y)
    g = net.backward(cache)

    eps = 1e-6
    worst = 0.0
    for name in ('M', 'R', 'W', 'b'):
        P = getattr(net, name)
        ga = g[name]
        gn = np.zeros_like(P)
        it = np.nditer(P, flags=['multi_index'])
        while not it.finished:
            ix = it.multi_index
            old = P[ix]
            P[ix] = old + eps
            lp, _ = net.loss(X, y)
            P[ix] = old - eps
            lm, _ = net.loss(X, y)
            P[ix] = old
            gn[ix] = (lp - lm) / (2 * eps)
            it.iternext()
        denom = np.maximum(1e-8, np.abs(ga) + np.abs(gn))
        rel = np.max(np.abs(ga - gn) / denom)
        worst = max(worst, rel)
        if verbose:
            print(f"   grad-check {name}: max rel err = {rel:.2e}")
    assert worst < 1e-4, f"GRAD CHECK FAILED: worst rel err {worst:.2e}"
    if verbose:
        print(f"   >>> gradient check PASSED (worst {worst:.2e})")
    return worst


# ============================================================================
# SECTION 6 -- EXPERIMENTS / SELF-TESTS
# ============================================================================

def exp_basic_naming():
    """The net learns to name clouds of particulars; accounts grow as needed."""
    print("\n[1] NAMING PARTICULARS  (exemplar classification, no universals)")
    rng = np.random.default_rng(SEED + 4)
    Xtr, ytr, ctr = make_particulars([0, 1, 2, 3], d=6, per_class=120, rng=rng)
    Xte, yte, _ = make_particulars([0, 1, 2, 3], d=6, per_class=40,
                                   rng=np.random.default_rng(SEED + 99),
                                   centres=ctr)
    net = OikeiosLogosNet(d=6, n_classes=4, max_accounts=40, rng=rng)
    train(net, Xtr, ytr, epochs=60, lr=0.25, rng=rng, verbose=True)
    tr, te = net.accuracy(Xtr, ytr), net.accuracy(Xte, yte)
    print(f"      train acc={tr:.3f}  test acc={te:.3f}  accounts={net.K}")
    assert te > 0.90, "expected the net to name held-out particulars well"
    return net


def exp_continual_virtue():
    """ASKESIS: virtue once acquired cannot be lost.
    Phase 1 learns classes {0,1,2} and consolidates. Phase 2 learns {3,4,5}.
    With consolidation ON, Phase-1 accuracy survives; with it OFF it collapses."""
    print("\n[2] ASKESIS / NON-LOSABLE VIRTUE  (continual learning)")
    d = 6
    A = [0, 1, 2]
    B = [3, 4, 5]
    rngd = np.random.default_rng(SEED + 5)
    Xa, ya, ca = make_particulars(A, d=d, per_class=120, rng=rngd)
    Xb, yb, _ = make_particulars(B, d=d, per_class=120, rng=rngd)
    Xa_te, ya_te, _ = make_particulars(A, d=d, per_class=40,
                                       rng=np.random.default_rng(SEED + 77),
                                       centres=ca)

    def run(consolidate):
        rng = np.random.default_rng(SEED + 6)
        net = OikeiosLogosNet(d=d, n_classes=6, max_accounts=60,
                              lam_anchor=(3.0 if consolidate else 0.0), rng=rng)
        train(net, Xa, ya, epochs=55, lr=0.25, gamma=4.0,
              consolidate=consolidate, rng=rng)
        acc_after_A = net.accuracy(Xa_te, ya_te)
        train(net, Xb, yb, epochs=55, lr=0.25, gamma=4.0,
              consolidate=consolidate, rng=rng)
        acc_retained = net.accuracy(Xa_te, ya_te)   # old "virtue" after new task
        return acc_after_A, acc_retained, net.K

    onA, on_ret, onK = run(consolidate=True)
    offA, off_ret, offK = run(consolidate=False)
    print(f"      with askesis : learned A={onA:.3f}  retained A after B={on_ret:.3f}  (K={onK})")
    print(f"      no  askesis  : learned A={offA:.3f}  retained A after B={off_ret:.3f}  (K={offK})")
    print(f"      forgetting avoided: +{(on_ret - off_ret) * 100:.1f} pts retained")
    assert on_ret > off_ret + 0.10, "consolidation should reduce forgetting"
    assert on_ret > 0.80, "consolidated virtue should largely survive"
    return on_ret, off_ret


def exp_reference_split():
    """NO-CONTRADICTION: a genuinely novel particular gets its OWN account
    instead of corrupting an existing one. We record account count before and
    after presenting an unrecognized cloud, and verify the old account means
    are essentially undisturbed (no 'contradiction' of a held name)."""
    print("\n[3] NO-CONTRADICTION ROUTING  (reference-split on novelty)")
    rng = np.random.default_rng(SEED + 7)
    Xtr, ytr, _ = make_particulars([0, 1], d=5, per_class=120, rng=rng)
    net = OikeiosLogosNet(d=5, n_classes=3, max_accounts=80, rng=rng)
    train(net, Xtr, ytr, epochs=45, lr=0.25, rng=rng)
    K_before = net.K
    M_before = net.M[:K_before].copy()

    # a class never seen before, far away in space -> must be a NEW thing
    Xnew = np.array([12.0, -12.0, 12.0, -12.0, 12.0]) + \
        rng.normal(0, 0.4, (60, 5))
    ynew = np.full(60, 2)
    pre_acq = net.max_acquaintance(Xnew).mean()
    train(net, Xnew, ynew, epochs=30, lr=0.25, rng=rng)
    K_after = net.K
    drift = np.abs(net.M[:K_before] - M_before).max()
    print(f"      mean acquaintance of novel cloud before learning = {pre_acq:.3f}")
    print(f"      accounts: {K_before} -> {K_after}  (new names spawned, not overwritten)")
    print(f"      max drift of pre-existing account means = {drift:.3f} (anchored, ~0)")
    assert K_after > K_before, "novel particulars must spawn new accounts"
    assert drift < 0.5, "existing accounts must not be contradicted/corrupted"
    # the net can now name the old AND the new
    assert net.accuracy(Xtr, ytr) > 0.9
    return K_before, K_after


def self_tests():
    print("\n[4] SELF-TESTS")
    p = softmax(np.random.default_rng(0).normal(size=(5, 4)))
    assert np.allclose(p.sum(axis=1), 1.0), "softmax rows must sum to 1"
    # determinism
    a = OikeiosLogosNet(3, 2, rng=np.random.default_rng(1))
    b = OikeiosLogosNet(3, 2, rng=np.random.default_rng(1))
    x = np.array([0.1, 0.2, 0.3])
    a.grow_account(x, 0)
    b.grow_account(x, 0)
    assert np.allclose(a.W, b.W), "same seed -> same account init"
    print("      softmax normalization ....... ok")
    print("      seeded determinism ........... ok")
    print("      (growth + consolidation asserts pass inside experiments above)")


# ============================================================================
# SECTION 7 -- MAIN
# ============================================================================

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    print("=" * 76)
    print("  THE OIKEIOS LOGOS NETWORK  --  Antisthenes (#67)")
    print("  'I see the horse, but I do not see horseness.'")
    print("=" * 76)

    print("\n[0] GRADIENT CHECK (analytic vs finite differences)")
    gradient_check(verbose=True)

    exp_basic_naming()
    exp_continual_virtue()
    exp_reference_split()
    self_tests()

    print("\n" + "=" * 76)
    print("  ALL CHECKS PASSED -- the wall holds.")
    print("=" * 76)

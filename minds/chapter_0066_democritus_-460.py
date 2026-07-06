"""
================================================================================
chapter_0066_democritus_-460.py  —  THE KINETIC ATOMIST NETWORK (KAN)
A from-scratch, pure-NumPy cognitive architecture built on the physics of
Democritus of Abdera (c. 460 - c. 370 BCE).
 Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 4 Minds 61 - 80 Available on Amazon https://www.amazon.com/dp/B0H71JC95Q
Resume and Interactive Demos at https://artificiology.com/
Author: David Vivancos · Chapter 0066 · Democritus of Abdera
================================================================================

WHY THIS IS *NOT* A TRANSFORMER
-------------------------------------------------------------------------------
Democritus held that nothing exists but ATOMS and VOID, and that mind/soul is
itself made of the smoothest, roundest, most mobile atoms (the same kind that
compose fire). Three of his doctrines dictate the mechanism below and rule out
attention-over-stored-keys:

  (1) LIKE-TO-LIKE COHESION.  Reported by Aristotle/Theophrastus: atoms of
      similar shape gather with their like ("round going with round,
      irregular and triangular with their like"). There is no query/key
      lookup and no learned routing -- coupling is the *symmetric* mutual
      affinity of shapes. Action equals reaction. So our coupling matrix A is
      forced symmetric (A = S @ S.T), unlike a Transformer's asymmetric QK^T.

  (2) EIDOLA.  Perception is the impact of thin films of atoms (eidola)
      sloughed off objects and streaming into the sense-atoms. Input is not a
      token sequence to be attended over; it is a *field* injected onto every
      atom and held there throughout settling (the film keeps pressing).

  (3) ANANKE (necessity), NO SWERVE.  Democritus is the strict determinist;
      the swerve/clinamen belongs to Epicurus, not him. So cognition is a
      DETERMINISTIC RELAXATION: atoms settle, by fixed mechanical law, from an
      agitated sensory state toward a stable configuration. We unroll T steps
      of a contractive tanh recurrence with *shared* coupling -- a kinetic
      settling, not stacked heterogeneous layers.

The "two forms of knowledge" (Sextus, Adv. Math. 7.138-139): the raw initial
agitation a[0] is *bastard* knowledge (skotie, the senses); the settled deep
configuration a[T] is *genuine* knowledge (gnesie). A diagnostic at the end
probes both and shows the settling adds real discriminative structure.

THE MECHANISM (one forward pass)
-------------------------------------------------------------------------------
  E       = X @ W_e                      # eidola films stream onto N atoms
  A       = S @ S.T                      # like-to-like shape affinity (symmetric)
  a[0]    = tanh(E)                      # bastard knowledge: raw sense agitation
  for t in 1..T:
      Z[t]   = alpha * (a[t-1] @ A) + E  # cohesion pull + persistent eidola drive
      a[t]   = tanh(Z[t])                # collisions bound the motion
  logits  = a[T] @ W_o + b_o             # the settled kosmos -> a percept/class
  loss    = cross_entropy(logits, y)

Trainable parameters: W_e, S, alpha (scalar gain), W_o, b_o.
Everything is analytic backprop (BPTT through the settling), verified against
finite differences in gradient_check() -- which MUST print PASS before training.

Run:  python3 chapter_0066_democritus_-460.py
================================================================================
"""

import numpy as np

RNG = np.random.default_rng(66)  # seed = the figure's index, for reproducibility


# ==============================================================================
# SECTION 1 — NUMERICAL PRIMITIVES
# ==============================================================================

def stable_softmax(logits):
    """Row-wise softmax with the max-subtraction trick (no overflow)."""
    z = logits - np.max(logits, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def cross_entropy(logits, y):
    """Mean negative log-likelihood for integer class targets y (shape [B])."""
    p = stable_softmax(logits)
    B = logits.shape[0]
    nll = -np.log(p[np.arange(B), y] + 1e-12)
    return float(np.mean(nll))


# ==============================================================================
# SECTION 2 — THE WORLD: "COMPOUNDS" (concord vs discord — a synthetic task)
# ==============================================================================
#
# Democritus: stable compounds form when atoms "fit"; like joins with like and
# congruent shapes interlock, while clashing shapes fly apart. We test the
# architecture on a task with exactly that logic. Each eidolon encodes two
# latent atom-presences, each +1 or -1. When the two AGREE (++ or --) the
# atoms are in CONCORD (class 0, a stable compound); when they DISAGREE (+- or
# -+) they are in DISCORD (class 1). That rule is XOR — the canonical problem a
# linear read-off of the senses CANNOT solve. Only the nonlinear like-to-like
# settling can detect the emergent agreement. The percept (concord/discord)
# must EMERGE from relaxation; it is not present in any single sense channel.

def make_compounds(n_per_cell=180, d_in=24, spread=0.45, seed=66):
    """Four Gaussian cells at the corners (+/-1, +/-1). Concord on the main
    diagonal (label 0), discord on the anti-diagonal (label 1) = XOR. Lifted
    into d_in dims by a fixed random 'void geometry'."""
    rng = np.random.default_rng(seed)
    centers = [(+1, +1, 0), (-1, -1, 0), (+1, -1, 1), (-1, +1, 1)]
    latent, labels = [], []
    for cx, cy, lab in centers:
        pts = np.stack([cx + spread * rng.standard_normal(n_per_cell),
                        cy + spread * rng.standard_normal(n_per_cell)], axis=1)
        latent.append(pts)
        labels.append(np.full(n_per_cell, lab))
    latent = np.concatenate(latent, 0)
    y = np.concatenate(labels, 0)
    P = rng.standard_normal((2, d_in)) / np.sqrt(2)          # the void geometry
    X = latent @ P + 0.05 * rng.standard_normal((latent.shape[0], d_in))
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    perm = rng.permutation(X.shape[0])
    return X[perm], y[perm]


# ==============================================================================
# SECTION 3 — THE KINETIC ATOMIST NETWORK
# ==============================================================================

class KineticAtomistNet:
    """
    N soul-atoms in a void. Each atom carries a learnable SHAPE vector (S).
    An eidolon X projects onto the atoms (W_e). Atoms then settle for T steps
    under symmetric like-to-like cohesion (A = S S^T) scaled by a global gain
    (alpha), with the eidola field held constant throughout. The settled state
    is read out (W_o, b_o) into class logits.
    """

    def __init__(self, d_in, n_atoms=40, shape_dim=8, n_classes=3, T=6, seed=66):
        rng = np.random.default_rng(seed)
        self.d_in, self.N, self.k = d_in, n_atoms, shape_dim
        self.C, self.T = n_classes, T

        # --- parameters -------------------------------------------------------
        # W_e: how each component of the eidolon presses on each atom.
        self.W_e = rng.standard_normal((d_in, n_atoms)) * (1.0 / np.sqrt(d_in))
        # b_a: each atom's threshold / resting position in the void. Lets an
        # atom behave as a tuned detector (fires only when its film+cohesion
        # exceeds threshold) — without it the void has no fixed reference.
        self.b_a = np.zeros(n_atoms)
        # S: atom shapes. Affinity A = S S^T governs like-to-like cohesion.
        self.S = rng.standard_normal((n_atoms, shape_dim)) * (1.0 / np.sqrt(shape_dim))
        # alpha: global cohesion gain (start sub-critical so settling is stable).
        self.alpha = np.array(0.50)
        # readout from the settled configuration to a percept (class).
        self.W_o = rng.standard_normal((n_atoms, n_classes)) * (1.0 / np.sqrt(n_atoms))
        self.b_o = np.zeros(n_classes)

    # -- parameter plumbing (used by the optimizer and the gradient check) -----
    def params(self):
        return {"W_e": self.W_e, "b_a": self.b_a, "S": self.S,
                "alpha": self.alpha, "W_o": self.W_o, "b_o": self.b_o}

    def set_param(self, name, value):
        setattr(self, name, value)

    # -------------------------------------------------------------------------
    def forward(self, X):
        """Run the settling. Returns logits and a cache for backprop."""
        E = X @ self.W_e + self.b_a            # (B, N)  eidola + atom threshold
        A = (self.S @ self.S.T) / self.N       # (N, N)  symmetric like-to-like,
        #                                                normalized by atom count
        a = [np.tanh(E)]                       # a[0] = bastard knowledge
        Zs = []
        for _ in range(self.T):
            Z = self.alpha * (a[-1] @ A) + E   # cohesion + persistent eidola
            Zs.append(Z)
            a.append(np.tanh(Z))               # bounded by collisions
        logits = a[-1] @ self.W_o + self.b_o   # settled kosmos -> percept
        cache = {"X": X, "E": E, "A": A, "a": a, "Zs": Zs, "logits": logits}
        return logits, cache

    # -------------------------------------------------------------------------
    def backward(self, cache, y):
        """Analytic gradients via BPTT through the settling. Returns dict."""
        X, E, A, a = cache["X"], cache["E"], cache["A"], cache["a"]
        B = X.shape[0]
        p = stable_softmax(cache["logits"])
        onehot = np.zeros_like(p)
        onehot[np.arange(B), y] = 1.0
        dlogits = (p - onehot) / B             # (B, C)

        # --- readout ----------------------------------------------------------
        aT = a[-1]
        dW_o = aT.T @ dlogits                  # (N, C)
        db_o = dlogits.sum(0)                  # (C,)
        da = dlogits @ self.W_o.T              # (B, N) grad wrt settled state

        # --- unroll backward through the T settling steps ---------------------
        # dpre accumulates grad wrt the held field (E = X W_e + b_a), which is
        # re-injected at a[0] and at every settling step.
        dpre = np.zeros_like(E)
        dAhat = np.zeros_like(A)               # grad wrt the normalized coupling
        dalpha = 0.0
        for t in range(self.T, 0, -1):         # t = T .. 1
            dZ = da * (1.0 - a[t] ** 2)        # tanh'
            a_prev = a[t - 1]
            # Z = alpha*(a_prev @ A) + E
            dalpha += float(np.sum(dZ * (a_prev @ A)))
            dAhat += self.alpha * (a_prev.T @ dZ)      # (N, N)
            dpre += dZ                                 # field enters every step
            da = self.alpha * (dZ @ A)                 # A symmetric -> A.T = A
        # a[0] = tanh(E)
        dZ0 = da * (1.0 - a[0] ** 2)
        dpre += dZ0

        # --- chain into the raw parameters ------------------------------------
        dW_e = X.T @ dpre                              # (d_in, N)
        db_a = dpre.sum(0)                             # (N,)
        # A = (S S^T) / N  ->  dA_real = dAhat / N ;  dS = (dA + dA^T) @ S
        dA = dAhat / self.N
        dS = (dA + dA.T) @ self.S                      # (N, k)
        return {"W_e": dW_e, "b_a": db_a, "S": dS, "alpha": np.array(dalpha),
                "W_o": dW_o, "b_o": db_o}


# ==============================================================================
# SECTION 4 — GRADIENT CHECK (MANDATORY — must PASS before any training)
# ==============================================================================

def gradient_check(verbose=True):
    """Finite-difference verification of every analytic gradient group."""
    d_in, C = 10, 3
    net = KineticAtomistNet(d_in=d_in, n_atoms=7, shape_dim=4, n_classes=C, T=4, seed=1)
    rng = np.random.default_rng(7)
    X = rng.standard_normal((5, d_in))
    y = rng.integers(0, C, size=5)

    logits, cache = net.forward(X)
    grads = net.backward(cache, y)

    eps = 1e-5
    worst = 0.0
    report = []
    for name, P in net.params().items():
        P = np.atleast_1d(P).astype(float)
        flat = P.ravel()
        ana = np.atleast_1d(grads[name]).astype(float).ravel()
        num = np.zeros_like(flat)
        # sample up to 12 coordinates per parameter group for speed
        idxs = range(flat.size) if flat.size <= 12 else \
            rng.choice(flat.size, 12, replace=False)
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps
            net.set_param(name, _reshape_like(flat, net, name))
            lp = cross_entropy(net.forward(X)[0], y)
            flat[i] = orig - eps
            net.set_param(name, _reshape_like(flat, net, name))
            lm = cross_entropy(net.forward(X)[0], y)
            flat[i] = orig
            net.set_param(name, _reshape_like(flat, net, name))
            num[i] = (lp - lm) / (2 * eps)
        sel = list(idxs)
        rel = np.abs(ana[sel] - num[sel]) / (np.abs(ana[sel]) + np.abs(num[sel]) + 1e-9)
        mx = float(np.max(rel))
        worst = max(worst, mx)
        report.append((name, mx))
        if verbose:
            print(f"   {name:6s}  max-rel-err = {mx:.2e}  "
                  f"(checked {len(sel)}/{flat.size})")
    ok = worst < 1e-4
    if verbose:
        print(f"   {'-'*46}\n   WORST max-rel-err = {worst:.2e}  -> "
              f"{'PASS' if ok else 'FAIL'}")
    return ok


def _reshape_like(flat, net, name):
    """Helper: reshape a flat buffer back to the parameter's native shape."""
    template = net.params()[name]
    if np.ndim(template) == 0:
        return np.array(float(flat[0]))
    return flat.reshape(np.shape(template)).copy()


# ==============================================================================
# SECTION 5 — OPTIMIZER (SGD with momentum) AND TRAINING LOOP
# ==============================================================================

class Momentum:
    def __init__(self, params, lr=0.15, mu=0.9):
        self.lr, self.mu = lr, mu
        self.v = {k: np.zeros_like(np.atleast_1d(v).astype(float))
                  for k, v in params.items()}

    def step(self, net, grads):
        for k in self.v:
            g = np.atleast_1d(grads[k]).astype(float)
            self.v[k] = self.mu * self.v[k] - self.lr * g
            cur = np.atleast_1d(net.params()[k]).astype(float)
            new = cur + self.v[k]
            if np.ndim(net.params()[k]) == 0:
                val = float(new[0])
                if k == "alpha":             # keep cohesion in a stable band
                    val = float(np.clip(val, 0.0, 4.0))
                net.set_param(k, np.array(val))
            else:
                net.set_param(k, new.reshape(np.shape(net.params()[k])))


def accuracy(net, X, y):
    return float(np.mean(np.argmax(net.forward(X)[0], axis=1) == y))


def train(net, Xtr, ytr, Xte, yte, epochs=60, batch=64, lr=0.15):
    opt = Momentum(net.params(), lr=lr)
    n = Xtr.shape[0]
    rng = np.random.default_rng(66)
    hist = []
    for ep in range(1, epochs + 1):
        perm = rng.permutation(n)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            _, cache = net.forward(Xtr[idx])
            grads = net.backward(cache, ytr[idx])
            opt.step(net, grads)
        if ep % 20 == 0 or ep == 1:
            ltr = cross_entropy(net.forward(Xtr)[0], ytr)
            atr = accuracy(net, Xtr, ytr)
            ate = accuracy(net, Xte, yte)
            hist.append((ep, ltr, atr, ate))
            print(f"   epoch {ep:3d} | loss {ltr:.4f} | "
                  f"train acc {atr:.3f} | test acc {ate:.3f} | "
                  f"alpha {float(net.alpha):.3f}")
    return hist


# ==============================================================================
# SECTION 6 — "TWO FORMS OF KNOWLEDGE" DIAGNOSTIC
# ==============================================================================
#
# Sextus reports Democritus' split between bastard knowledge (the senses) and
# genuine knowledge (won by the intellect going deeper than the senses). We
# fit a *linear* probe on the raw sense state a[0] and on the settled state
# a[T], and compare. If settling matters, genuine > bastard.

def linear_probe_accuracy(states, y, n_classes, l2=1e-2, iters=300, lr=0.5):
    """Tiny multinomial-logistic probe trained by full-batch GD (NumPy only)."""
    B, D = states.shape
    W = np.zeros((D, n_classes))
    b = np.zeros(n_classes)
    onehot = np.zeros((B, n_classes)); onehot[np.arange(B), y] = 1.0
    for _ in range(iters):
        p = stable_softmax(states @ W + b)
        gW = states.T @ (p - onehot) / B + l2 * W
        gb = (p - onehot).mean(0)
        W -= lr * gW; b -= lr * gb
    return float(np.mean(np.argmax(states @ W + b, axis=1) == y))


def two_forms_of_knowledge(net, X, y):
    """Bastard knowledge = a linear read of the RAW eidola (the unprocessed
    senses). Genuine knowledge = a linear read of the SETTLED configuration.
    For the XOR compound task the raw senses are not linearly separable, so the
    gap measures exactly what the settling adds."""
    _, cache = net.forward(X)
    bastard = X                 # raw senses, before any settling
    genuine = cache["a"][-1]    # settled intellect
    acc_b = linear_probe_accuracy(bastard, y, net.C)
    acc_g = linear_probe_accuracy(genuine, y, net.C)
    return acc_b, acc_g


# ==============================================================================
# SECTION 7 — SELF-TESTS + MAIN
# ==============================================================================

def self_tests():
    print("[self-test] softmax rows sum to 1 ...", end=" ")
    p = stable_softmax(RNG.standard_normal((4, 5)))
    assert np.allclose(p.sum(1), 1.0); print("ok")

    print("[self-test] settling is contractive (sub-critical alpha) ...", end=" ")
    net = KineticAtomistNet(d_in=8, n_atoms=12, n_classes=3, T=20, seed=3)
    net.alpha = np.array(0.25)
    _, c = net.forward(RNG.standard_normal((3, 8)))
    diffs = [np.max(np.abs(c["a"][t + 1] - c["a"][t])) for t in range(net.T)]
    assert diffs[-1] < diffs[0]  # later steps move less -> it settles
    print(f"ok (step-1 move {diffs[0]:.3f} -> final move {diffs[-1]:.3f})")

    print("[self-test] coupling matrix is symmetric (action=reaction) ...", end=" ")
    A = net.S @ net.S.T
    assert np.allclose(A, A.T); print("ok")


def main():
    print("=" * 78)
    print("THE KINETIC ATOMIST NETWORK — Democritus of Abdera")
    print("atoms + void + like-to-like cohesion + deterministic settling")
    print("=" * 78)

    print("\n[1] Self-tests")
    self_tests()

    print("\n[2] Gradient check (finite differences vs analytic BPTT)")
    ok = gradient_check(verbose=True)
    assert ok, "Gradient check FAILED — refusing to train."

    print("\n[3] Build the world of compounds (concord vs discord = XOR)")
    X, y = make_compounds(n_per_cell=200, d_in=24, seed=66)
    ntr = int(0.8 * X.shape[0])
    Xtr, ytr, Xte, yte = X[:ntr], y[:ntr], X[ntr:], y[ntr:]
    print(f"   {X.shape[0]} eidola, dim {X.shape[1]}, "
          f"2 classes (concord/discord) | train {ntr} / test {X.shape[0]-ntr}")

    print("\n[4] Train the settling")
    net = KineticAtomistNet(d_in=24, n_atoms=40, shape_dim=8,
                            n_classes=2, T=6, seed=66)
    train(net, Xtr, ytr, Xte, yte, epochs=120, batch=64, lr=0.05)
    print(f"   FINAL  train acc {accuracy(net, Xtr, ytr):.3f} | "
          f"test acc {accuracy(net, Xte, yte):.3f}")

    print("\n[5] Two forms of knowledge (linear probe: raw eidola vs settled)")
    acc_b, acc_g = two_forms_of_knowledge(net, Xte, yte)
    print(f"   bastard knowledge  (raw eidola X)    : {acc_b:.3f}")
    print(f"   genuine knowledge  (settled  a[T])   : {acc_g:.3f}")
    print(f"   settling lifted separability by {acc_g - acc_b:+.3f}")

    print("\n" + "=" * 78)
    print("Done. The percept was not read off the senses; it settled out of "
          "them.")
    print("=" * 78)


if __name__ == "__main__":
    main()

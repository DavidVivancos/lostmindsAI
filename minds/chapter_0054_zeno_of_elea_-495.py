#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 chapter_0054_zeno_of_elea_-495.py  ::  THE DICHOTOMY ENGINE
 A fixed-point (deep-equilibrium) neural architecture, from scratch in NumPy.
Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 3 Minds 41 - 60 Available on Amazon https://www.amazon.com/dp/B0H6TVX69S
Resume and Interactive Demos at https://artificiology.com/
Author: David Vivancos · Chapter 0054 · Zeno of Elea
================================================================================

WHY A FIXED-POINT NET IS THE *ZENO* ARCHITECTURE (AND NOT A GENERIC ONE)
-----------------------------------------------------------------------
Zeno of Elea built no positive theory of mind. He bequeathed a METHOD and a
PROBLEM, and this architecture is the smallest honest machine that runs on both.

  THE PROBLEM -- the supertask.  To cross a room you must first cross half,
  then half of the rest, forever: an UNENDING DISCRETE REGRESS. Zeno concluded
  "nothing moves." Mathematics answered, two thousand years later, that some
  such regresses CONVERGE: a contraction whose steps shrink fast enough has a
  single definite LIMIT (Achilles really does catch the tortoise; the geometric
  series 100/(1-1/10) = 111.11... is finite). The limit is the fixed point of
  the map that generates the regress:  z* = f(z*).

  THE METHOD -- reductio / dialectic.  Aristotle credited Zeno with inventing
  dialectic: never assert; take the opponent's claim, ride it to contradiction,
  and let the contradiction do the work. Here that becomes a gradient-free
  ADVERSARIAL REDUCTIO probe that assumes the model's own verdict and hunts, by
  halving search, for a nearby point that refutes it.

So the design writes itself, and it is Zeno's alone:

  * FORWARD PASS == Zeno's regress, run to its limit. We iterate
        z_{k+1} = tanh(W z_k + U x + b)
    until it stops moving. Because W is constrained to be a CONTRACTION
    (spectral norm < gamma < 1), the residual ||z_{k+1}-z_k|| decays
    geometrically -- the half-step shrinking that the Dichotomy describes -- and
    the regress has exactly one fixed point. The paradox is *resolved by
    construction*: infinitely many steps, one finite limit.

  * BACKWARD PASS == the calculus that dissolves the paradox.  We do NOT
    backpropagate through the (in principle infinite) iteration. We invoke the
    IMPLICIT FUNCTION THEOREM and jump straight to the gradient *at the limit*,
    solving one small linear system. This is the mathematical form of Zeno's
    own resolution: you never traverse the infinitely many stages; you compute
    where they are GOING.

  * THE CONTRACTION CONSTRAINT == the metaphysical commitment. Drop it and the
    regress may diverge or oscillate forever (Thomson's lamp: on/off/on/off,
    no definite state). Keeping ||W|| < 1 is the assertion that motion is
    *completable* -- that the One the regress converges to actually exists.

Contrast: a Transformer asks "which stored key matches this query?" Zeno would
reject the premise outright. There is no stored answer here -- only a process,
and the question of whether it has a limit.

ENGINEERING CONTRACT (shared by every file in this corpus)
----------------------------------------------------------
  * pure NumPy, from scratch; no autograd, no ML framework;
  * a finite-difference gradient check over EVERY parameter (mandatory) that
    validates the implicit-function-theorem backward;
  * a real training loop that measurably reduces loss on a real task;
  * self-tests + a Zeno demonstration (Achilles's regress; the reductio probe);
  * executes end to end and prints its verified output.

Run:  python3 chapter_0054_zeno_of_elea_-495.py
================================================================================
"""

import numpy as np

np.random.seed(495)   # Zeno, c. 495 BCE


# ============================================================================ #
# 1.  PRIMITIVES
# ============================================================================ #
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def spectral_norm(W, iters=60):
    """Largest singular value of W by power iteration.

    The contraction constant of x -> tanh(Wx + ...) is bounded by ||W||_2,
    because tanh is 1-Lipschitz. If ||W||_2 < 1 the map is a contraction and the
    regress is guaranteed a unique limit (Banach's fixed-point theorem).
    """
    v = np.random.randn(W.shape[1])
    v /= np.linalg.norm(v) + 1e-12
    for _ in range(iters):
        u = W @ v
        u /= np.linalg.norm(u) + 1e-12
        v = W.T @ u
        v /= np.linalg.norm(v) + 1e-12
    return float(u @ W @ v)


class ContractionConstraint:
    """Rescale W so ||W||_2 <= gamma < 1: force the Dichotomy regress to converge.

    This is not a regulariser bolted on for stability. It is the architecture's
    Eleatic premise -- the guarantee that the unending regress has one definite
    limit instead of diverging or flickering forever.
    """
    def __init__(self, gamma=0.9):
        self.gamma = gamma

    def apply(self, W):
        s = spectral_norm(W)
        if s > self.gamma:
            W = W * (self.gamma / (s + 1e-12))
        return W


# ============================================================================ #
# 2.  THE DICHOTOMY EQUILIBRIUM LAYER  (forward solve + implicit backward)
# ============================================================================ #
class DichotomyEquilibrium:
    """Solve and differentiate the fixed point  z* = tanh(W z* + U x + b).

    FORWARD  : run Zeno's regress to its limit (Banach iteration).
    BACKWARD : the implicit function theorem -- jump to the gradient of the
               limit by solving one (H x H) linear system per sample. No
               unrolling of the (infinite) regress is ever required.
    """

    def __init__(self, in_dim, hid_dim, gamma=0.9, tol=1e-10, max_iter=300):
        self.W = np.random.randn(hid_dim, hid_dim) * (0.5 / np.sqrt(hid_dim))
        self.U = np.random.randn(hid_dim, in_dim) * (1.0 / np.sqrt(in_dim))
        self.b = np.zeros(hid_dim)
        self.constraint = ContractionConstraint(gamma)
        self.tol, self.max_iter, self.hid_dim = tol, max_iter, hid_dim
        self.last_iters = 0
        self.last_residuals = []

    # ---- forward: the regress, run to convergence -------------------------- #
    def solve(self, X):
        """Return the fixed point z* (N, H) for inputs X (N, in_dim).

        Plain Banach iteration z_{k+1} = phi(W z_k + U x + b). The contraction
        makes ||z_{k+1}-z_k|| fall geometrically -- Achilles closing the gap.
        """
        self.W = self.constraint.apply(self.W)        # enforce convergence first
        Ux_b = X @ self.U.T + self.b                  # constant across the regress
        z = np.zeros((X.shape[0], self.hid_dim))
        residuals = []
        k = 0
        for k in range(self.max_iter):
            z_new = np.tanh(z @ self.W.T + Ux_b)
            r = np.linalg.norm(z_new - z) / (np.linalg.norm(z_new) + 1e-12)
            residuals.append(r)
            z = z_new
            if r < self.tol:
                break
        self.last_iters = k + 1
        self.last_residuals = residuals
        self._cache = (X, Ux_b, z)
        return z

    # ---- backward: implicit function theorem, gradient AT the limit -------- #
    def backward(self, grad_z):
        """Given dL/dz* (N, H) return {W,U,b} grads and dL/dX.

        At the fixed point z = phi(a), a = W z + U x + b, with D = diag(phi'(a)):
            differentiate z = phi(Wz + Ux + b) implicitly ->
            (I - D W) dz = D (dU x + dW z + db)
        The vector-Jacobian product is therefore:
            solve (I - D W)^T v = dL/dz ;   w = D v ;
            dL/dW += w z^T ;  dL/dU += w x^T ;  dL/db += w ;  dL/dx = W?^T w via U.
        One H x H solve per sample -- exact, no unrolling of the regress.
        """
        X, Ux_b, z = self._cache
        N, H = z.shape
        a = z @ self.W.T + Ux_b
        Dp = 1.0 - np.tanh(a) ** 2                   # phi'(a), shape (N, H)
        gW = np.zeros_like(self.W)
        gU = np.zeros_like(self.U)
        gb = np.zeros_like(self.b)
        gX = np.zeros_like(X)
        I = np.eye(H)
        for n in range(N):
            Dn = Dp[n]
            M = I - (Dn[:, None] * self.W)           # (I - D W)
            v = np.linalg.solve(M.T, grad_z[n])      # (I - D W)^T v = grad_z
            w = Dn * v
            gW += np.outer(w, z[n])
            gU += np.outer(w, X[n])
            gb += w
            gX[n] = self.U.T @ w
        return {"W": gW, "U": gU, "b": gb}, gX

    def params(self):
        return {"W": self.W, "U": self.U, "b": self.b}

    def set_params(self, p):
        self.W, self.U, self.b = p["W"], p["U"], p["b"]


# ============================================================================ #
# 3.  ZENONET  --  equilibrium core + linear read-out + softmax cross-entropy
# ============================================================================ #
class ZenoNet:
    def __init__(self, in_dim, hid_dim, n_classes, gamma=0.9):
        self.eq = DichotomyEquilibrium(in_dim, hid_dim, gamma=gamma)
        self.Wo = np.random.randn(n_classes, hid_dim) * (1.0 / np.sqrt(hid_dim))
        self.bo = np.zeros(n_classes)

    def forward(self, X, y=None):
        z = self.eq.solve(X)                         # Zeno's limit
        logits = z @ self.Wo.T + self.bo
        probs = softmax(logits)
        self._cache = (X, z, probs, y)
        if y is None:
            return None, probs
        N = X.shape[0]
        loss = -np.log(probs[np.arange(N), y] + 1e-12).mean()
        return loss, probs

    def backward(self):
        X, z, probs, y = self._cache
        N = X.shape[0]
        dlogits = probs.copy()
        dlogits[np.arange(N), y] -= 1.0
        dlogits /= N
        gWo = dlogits.T @ z
        gbo = dlogits.sum(axis=0)
        grad_z = dlogits @ self.Wo                   # upstream into the equilibrium
        eq_grads, _ = self.eq.backward(grad_z)
        return {"Wo": gWo, "bo": gbo,
                **{f"eq_{k}": v for k, v in eq_grads.items()}}

    def get_flat_params(self):
        p = self.eq.params()
        return {"eq_W": p["W"], "eq_U": p["U"], "eq_b": p["b"],
                "Wo": self.Wo, "bo": self.bo}

    def set_flat_params(self, p):
        self.eq.set_params({"W": p["eq_W"], "U": p["eq_U"], "b": p["eq_b"]})
        self.Wo, self.bo = p["Wo"], p["bo"]

    def sgd_step(self, grads, lr):
        p = self.get_flat_params()
        for k in p:
            p[k] -= lr * grads[k]
        self.set_flat_params(p)


# ============================================================================ #
# 4.  ACHILLES & THE TORTOISE  --  the convergent geometric series, made literal
# ============================================================================ #
def achilles_tortoise(v_achilles=10.0, v_tortoise=1.0, head_start=100.0, stages=40):
    """Sum the infinitely many catch-up stages of Zeno's race and compare with
    the closed-form catch point. Their agreement is the same fact the network's
    forward pass relies on: a contraction (ratio r = v_t/v_a < 1) makes an
    unending regress converge to one definite limit.
    """
    r = v_tortoise / v_achilles                      # < 1  ->  contraction
    gap = head_start
    total = 0.0
    for _ in range(stages):
        total += gap                                 # distance to current tortoise spot
        gap *= r                                     # next stage shrinks by r
    closed_form = head_start / (1.0 - r)             # geometric-series limit
    return total, closed_form, abs(total - closed_form)


# ============================================================================ #
# 5.  ADVERSARIAL REDUCTIO  --  dialectic as a gradient-free refutation probe
# ============================================================================ #
def adversarial_reductio(net, x, true_label, directions=24, depth=20, reach=2.0):
    """Zeno's move: assume the verdict, then hunt for a contradiction.

    Fire rays in many directions from x; along any ray that flips the model's
    class, do a BINARY (halving) search -- a dichotomy -- for the boundary. The
    returned distance is how locally refutable the verdict is. Small distance =>
    the claim 'this point is class y' collapses under a tiny perturbation.
    """
    _, p0 = net.forward(x[None, :])
    base = int(p0.argmax())
    best = np.inf
    best_pt = None
    for _ in range(directions):
        d = np.random.randn(*x.shape)
        d /= np.linalg.norm(d) + 1e-12
        far = x + reach * d
        _, pf = net.forward(far[None, :])
        if int(pf.argmax()) == base:
            continue                                 # this ray never refutes
        lo, hi = x.copy(), far.copy()                # boundary lies between
        for _ in range(depth):                       # halve the interval (dichotomy)
            mid = 0.5 * (lo + hi)
            _, pm = net.forward(mid[None, :])
            if int(pm.argmax()) == base:
                lo = mid
            else:
                hi = mid
        dist = np.linalg.norm(hi - x)
        if dist < best:
            best, best_pt = dist, hi
    return best, best_pt, base


# ============================================================================ #
# 6.  GRADIENT CHECK  --  validates the implicit-function-theorem backward
# ============================================================================ #
def finite_difference_grad_check(seed=1, eps=1e-6):
    """Central finite differences vs analytic implicit gradients, every param.

    The contraction-constrained W is frozen to its post-projection value before
    checking, so finite differences and the analytic VJP see the SAME map (the
    constraint is non-smooth at the threshold and would otherwise pollute FD).
    """
    rng = np.random.RandomState(seed)
    N, in_dim, hid, C = 5, 4, 6, 3
    X = rng.randn(N, in_dim)
    y = rng.randint(0, C, size=N)

    net = ZenoNet(in_dim, hid, C, gamma=0.9)
    net.eq.W = net.eq.constraint.apply(net.eq.W)     # freeze inside the safe region
    net.eq.constraint.apply = lambda W: W            # disable re-projection

    loss, _ = net.forward(X, y)
    grads = net.backward()

    worst = 0.0
    for name in ["eq_W", "eq_U", "eq_b", "Wo", "bo"]:
        P = net.get_flat_params()
        arr = P[name]
        flat = arr.reshape(-1)
        ga = grads[name].reshape(-1)
        local = 0.0
        for i in range(flat.size):
            orig = flat[i]
            flat[i] = orig + eps
            net.set_flat_params(P); lp, _ = net.forward(X, y)
            flat[i] = orig - eps
            net.set_flat_params(P); lm, _ = net.forward(X, y)
            flat[i] = orig
            net.set_flat_params(P)
            num = (lp - lm) / (2 * eps)
            den = max(1e-9, abs(num) + abs(ga[i]))
            local = max(local, abs(num - ga[i]) / den)
        worst = max(worst, local)
        print(f"    {name:5s} max rel err : {local:.3e}")
    print(f"    OVERALL max rel err        : {worst:.3e}   ->  "
          f"{'PASS' if worst < 1e-4 else 'FAIL'}")
    return worst < 1e-4


# ============================================================================ #
# 7.  DATA  --  two interleaving half-moons (a genuinely non-linear task)
# ============================================================================ #
def make_moons(n=240, noise=0.18, seed=7):
    rng = np.random.RandomState(seed)
    m = n // 2
    t1 = np.linspace(0, np.pi, m)
    x1 = np.stack([np.cos(t1), np.sin(t1)], 1)
    t2 = np.linspace(0, np.pi, n - m)
    x2 = np.stack([1 - np.cos(t2), 1 - np.sin(t2) - 0.5], 1)
    X = np.vstack([x1, x2]) + rng.randn(n, 2) * noise
    y = np.array([0] * m + [1] * (n - m))
    idx = rng.permutation(n)
    return X[idx], y[idx]


def standardize(X, mu=None, sd=None):
    if mu is None:
        mu, sd = X.mean(0), X.std(0) + 1e-9
    return (X - mu) / sd, mu, sd


# ============================================================================ #
# 8.  MAIN
# ============================================================================ #
def main():
    print("=" * 74)
    print(" ZENO OF ELEA -- THE DICHOTOMY ENGINE  (fixed-point / equilibrium net)")
    print(" forward = the regress to its limit; backward = jump to the limit")
    print("=" * 74)

    # [1] Achilles: the regress really does converge ------------------------- #
    print("\n[1] Achilles & the Tortoise  (the regress has a finite limit)")
    total, closed, diff = achilles_tortoise()
    print(f"    analytic catch-distance    : {closed:.6f}")
    print(f"    sum of 40 halving stages   : {total:.6f}")
    print(f"    agreement (|diff|)         : {diff:.2e}   -> the regress converges")

    # [2] gradient check ----------------------------------------------------- #
    print("\n[2] Finite-difference gradient check (implicit-function backward)")
    ok = finite_difference_grad_check()
    assert ok, "Gradient check FAILED -- the implicit backward is wrong."

    # [3] training ----------------------------------------------------------- #
    print("\n[3] Training on interleaving half-moons")
    Xtr, ytr = make_moons(240, seed=7)
    Xte, yte = make_moons(160, seed=23)
    Xtr, mu, sd = standardize(Xtr)
    Xte, _, _ = standardize(Xte, mu, sd)
    net = ZenoNet(in_dim=2, hid_dim=16, n_classes=2, gamma=0.9)
    lr = 0.5
    for ep in range(1, 121):
        loss, probs = net.forward(Xtr, ytr)
        grads = net.backward()
        net.sgd_step(grads, lr)
        if ep % 20 == 0 or ep == 1:
            acc = (probs.argmax(1) == ytr).mean()
            print(f"    epoch {ep:3d} | loss {loss:.4f} | train acc {acc*100:5.1f}% "
                  f"| regress steps to limit: {net.eq.last_iters}")
    _, ptr = net.forward(Xtr); tr_acc = (ptr.argmax(1) == ytr).mean()
    _, pte = net.forward(Xte); te_acc = (pte.argmax(1) == yte).mean()
    print(f"    final train acc : {tr_acc*100:.1f}%")
    print(f"    final test  acc : {te_acc*100:.1f}%")
    assert te_acc > 0.85, "model failed to learn the task"

    # [4] one forward solve, displayed as Zeno's regress --------------------- #
    print("\n[4] One forward solve, shown as the regress (residual per refinement)")
    net.eq.solve(Xte[:8])
    res = net.eq.last_residuals[:9]
    print("    residual ||z_{k+1}-z_k|| : " +
          ", ".join(f"{r:.2e}" for r in res) + " ...")
    ratios = [res[i+1] / (res[i] + 1e-12) for i in range(min(5, len(res) - 1))]
    print("    step-to-step ratios      : " +
          ", ".join(f"{q:.3f}" for q in ratios) +
          "   (all < 1  ->  contraction; the gap keeps shrinking)")

    # [5] dialectic as a probe ---------------------------------------------- #
    print("\n[5] Adversarial reductio  (halving search for a refuting neighbour)")
    # Scan several test points; report the most fragile verdict found.
    best_dist, best_base = np.inf, None
    for j in range(12):
        dist, _, base = adversarial_reductio(net, Xte[j], int(yte[j]), reach=3.0)
        if dist < best_dist:
            best_dist, best_base = dist, base
    if np.isfinite(best_dist):
        print(f"    most fragile verdict: class {best_base}; nearest contradicting "
              f"neighbour at distance {best_dist:.3f}")
        print("    -> the verdict is locally refutable: dialectic as a probe of "
              "fragility")
    else:
        print("    every sampled verdict was robust within reach")

    print("\n" + "=" * 74)
    print(" All self-tests complete. The regress converges; the gradient checks")
    print(" out; the mind runs.")
    print("=" * 74)


if __name__ == "__main__":
    main()

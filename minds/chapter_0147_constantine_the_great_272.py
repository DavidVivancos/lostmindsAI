"""
================================================================================
Chapter 0147_constantine_the_great_272 - Constantine the Great (272-337 CE)
Encyclopedia of Lost Minds: Echoes on AI
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 8 Minds 141 - 160 Available on Amazon https://www.amazon.com/dp/B0HH8RTCXF
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 147: Constantine the Great (272-337 CE)
================================================================================  

THE LABARUM ENGINE
A Deferred-Commitment Recurrent Network (pure NumPy, from scratch)

--------------------------------------------------------------------------------
WHY THIS ARCHITECTURE, FOR THIS MIND
--------------------------------------------------------------------------------
Constantine did not "convert." For more than a decade after the Milvian Bridge
(312 CE) his coinage still carried SOLI INVICTO COMITI — the Unconquered Sun as
the emperor's companion — while the Christian Chi-Rho crept in beside it. He
convened Nicaea (325) to fix the divinity of Christ, then spent twelve years
quietly rehabilitating the very Arians the council had condemned, and finally
accepted baptism only on his deathbed (337), from an Arian bishop, having
deferred the single irreversible act of his religious life until the last hour
that circumstance allowed.

That is not indecision. It is a cognitive *discipline*: hold rival hypotheses
(Sol vs. Christ; Arian vs. Nicene; Rome vs. Constantinople) in a live
superposition, pay the cost of keeping them open, read the sparse and ambiguous
"signs," and collapse to an irreversible commitment only when a decisive sign
crosses a threshold — because whoever controls the *moment of collapse* controls
the outcome, and a mind that commits early to a murky world is brittle.

This file encodes exactly that signature. It is NOT a Transformer and NOT a
generic classifier. It is a small recurrent belief accumulator with a second,
distinctive head:

  * an EVIDENCE head produces raw preferences over K rival hypotheses ("gods");
  * a COMMITMENT head produces a single non-negative scalar per timestep — an
    *inverse temperature* that decides HOW SHARPLY the belief distribution
    collapses right now.

The training objective rewards a correct FINAL decision but adds a
PREMATURE-COMMITMENT PENALTY: being confident (high commitment) while the
situation is still ambiguous is expensive. The optimal learned policy is
Constantine's: stay diffuse under ambiguity, then commit hard the instant a
decisive sign arrives — "in hoc signo."

--------------------------------------------------------------------------------
WHAT IS IN THIS FILE
--------------------------------------------------------------------------------
1. LabarumParams          - the trainable weights.
2. forward()              - BPTT-ready forward pass with full caches.
3. backward()             - hand-derived analytic gradients (vanilla-RNN BPTT
                            with two output heads + a penalty term).
4. gradient_check()       - MANDATORY finite-difference check of every gradient.
5. make_signs_dataset()   - synthetic "reading the signs" task with a decisive
                            omen appearing at a random late timestep.
6. train()                - a real training loop (Adam), not a demo stub.
7. self_tests()           - asserts that the learned model actually defers:
                            commitment is low under ambiguity and spikes at the
                            decisive sign, and final accuracy is high.

Dependencies: numpy only.
================================================================================
"""

import numpy as np


# =============================================================================
# SECTION 1 — TRAINABLE PARAMETERS
# =============================================================================

class LabarumParams:
    """
    All trainable weights of the Labarum Engine.

    Dimensions
    ----------
    D : size of a "sign" vector (an observation / omen at one timestep)
    H : hidden state size (the accumulated read of the situation so far)
    K : number of rival hypotheses ("gods" / doctrines / factions)

    Heads
    -----
    W_z, b_z : EVIDENCE head    h_t -> raw K-dim preference logits z_t
    w_c, b_c : COMMITMENT head  h_t -> scalar g_t; commitment c_t = softplus(g_t)
    """

    def __init__(self, D, H, K, seed=0):
        rng = np.random.default_rng(seed)
        # Recurrent core (a deliberately simple, fully-auditable RNN).
        self.W_x = rng.standard_normal((H, D)) * (1.0 / np.sqrt(D))
        self.W_h = rng.standard_normal((H, H)) * (1.0 / np.sqrt(H))
        self.b_h = np.zeros(H)
        # Evidence head: which hypothesis do the signs favour?
        self.W_z = rng.standard_normal((K, H)) * (1.0 / np.sqrt(H))
        self.b_z = np.zeros(K)
        # Commitment head: HOW HARD do we collapse the belief right now?
        self.w_c = rng.standard_normal(H) * (1.0 / np.sqrt(H))
        self.b_c = 0.0
        self.D, self.H, self.K = D, H, K

    # -- helpers to treat all params as one flat vector (for the grad check) --
    def names(self):
        return ["W_x", "W_h", "b_h", "W_z", "b_z", "w_c", "b_c"]

    def get(self, name):
        return getattr(self, name)

    def set(self, name, value):
        setattr(self, name, value)

    def copy(self):
        p = LabarumParams(self.D, self.H, self.K)
        for n in self.names():
            p.set(n, np.array(self.get(n), dtype=float, copy=True)
                  if np.ndim(self.get(n)) > 0 else float(self.get(n)))
        return p


# =============================================================================
# SECTION 2 — DIFFERENTIABLE PRIMITIVES
# =============================================================================

def softplus(x):
    # numerically stable softplus = log(1 + e^x)
    return np.logaddexp(0.0, x)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def softmax(v):
    v = v - np.max(v)
    e = np.exp(v)
    return e / np.sum(e)


# =============================================================================
# SECTION 3 — FORWARD PASS  (one sequence)
# =============================================================================

def forward(P, X, U, y=None, lam=0.35):
    """
    Run the Labarum Engine over one sequence of signs.

    Parameters
    ----------
    P   : LabarumParams
    X   : (T, D) array — the sequence of sign vectors (the omens as they arrive)
    U   : (T,)  array in [0,1] — the AMBIGUITY of the situation at each step.
                High U early (the signs are murky); U drops when a decisive
                omen arrives. This is the emperor's read of "how unclear is it
                still?" — committing while U is high is what the penalty punishes.
    y   : int or None — the true hypothesis (for loss); None -> inference only.
    lam : float — weight of the premature-commitment penalty.

    Returns
    -------
    out : dict with the final distribution p_T, per-step commitments c,
          per-step evidence z, the loss (if y given), and caches for backprop.
    """
    T, D = X.shape
    H, K = P.H, P.K

    h_prev = np.zeros(H)
    hs = np.zeros((T, H))       # h_t after tanh
    hs_prev = np.zeros((T, H))  # h_{t-1} feeding step t (for W_h grad)
    zs = np.zeros((T, K))       # evidence logits
    gs = np.zeros(T)            # pre-commitment scalar
    cs = np.zeros(T)            # commitment = softplus(g)
    for t in range(T):
        a = P.W_x @ X[t] + P.W_h @ h_prev + P.b_h
        h = np.tanh(a)
        z = P.W_z @ h + P.b_z
        g = P.w_c @ h + P.b_c
        c = softplus(g)
        hs_prev[t] = h_prev
        hs[t] = h
        zs[t] = z
        gs[t] = g
        cs[t] = c
        h_prev = h

    # Final decision uses the collapsed distribution at the last step:
    #   logits_T = c_T * z_T   (commitment sharpens or softens the evidence)
    logits_T = cs[-1] * zs[-1]
    p_T = softmax(logits_T)

    out = {"p_T": p_T, "c": cs, "z": zs, "g": gs,
           "hs": hs, "hs_prev": hs_prev, "X": X, "U": U, "lam": lam}

    if y is not None:
        ce = -np.log(p_T[y] + 1e-12)
        penalty = lam * float(np.sum(cs * U))   # pay for confidence-under-ambiguity
        out["loss"] = ce + penalty
        out["ce"] = ce
        out["penalty"] = penalty
        out["y"] = y
    return out


# =============================================================================
# SECTION 4 — BACKWARD PASS  (hand-derived analytic gradients, full BPTT)
# =============================================================================

def backward(P, cache):
    """
    Analytic gradient of the loss in `forward` w.r.t. every parameter.

    Loss = CrossEntropy(softmax(c_T * z_T), y)  +  lam * sum_t c_t * U_t

    Only the FINAL step contributes through the decision (that is the whole
    point — the decision is deferred to the end); every step contributes
    through the commitment penalty. Below is standard two-head vanilla-RNN
    BPTT; each line is annotated with the local derivative it applies.
    """
    X, U = cache["X"], cache["U"]
    hs, hs_prev = cache["hs"], cache["hs_prev"]
    zs, gs, cs = cache["z"], cache["g"], cache["c"]
    p_T, y, lam = cache["p_T"], cache["y"], cache["lam"]
    T, D = X.shape
    H, K = P.H, P.K

    g = {n: np.zeros_like(np.atleast_1d(P.get(n)).astype(float))
         for n in P.names()}
    g["b_c"] = np.zeros(1)  # keep scalar as 1-vector for uniform handling

    # dL/dlogits_T = p_T - onehot(y)     (softmax + cross-entropy)
    dlogits_T = p_T.copy()
    dlogits_T[y] -= 1.0

    grad_h_next = np.zeros(H)  # recurrent gradient flowing back from step t+1

    for t in reversed(range(T)):
        # ---- gradient into z_t and c_t coming from the loss ----
        if t == T - 1:
            # logits_T = c_T * z_T
            dz = dlogits_T * cs[t]                 # dL/dz_T
            dc = float(np.dot(dlogits_T, zs[t]))   # dL/dc_T via decision
        else:
            dz = np.zeros(K)                       # earlier steps: no decision term
            dc = 0.0
        dc += lam * U[t]                           # + premature-commitment penalty

        # ---- EVIDENCE head:  z_t = W_z h_t + b_z ----
        g["W_z"] += np.outer(dz, hs[t])
        g["b_z"] += dz
        dh = P.W_z.T @ dz                          # dL/dh_t via evidence head

        # ---- COMMITMENT head:  c_t = softplus(g_t), g_t = w_c . h_t + b_c ----
        dg = dc * sigmoid(gs[t])                   # softplus'(g) = sigmoid(g)
        g["w_c"] += dg * hs[t]
        g["b_c"][0] += dg
        dh += dg * P.w_c                           # dL/dh_t via commitment head

        # ---- add recurrent gradient from the future ----
        dh += grad_h_next

        # ---- through tanh:  h_t = tanh(a_t) ----
        da = dh * (1.0 - hs[t] ** 2)               # tanh'(a) = 1 - tanh(a)^2

        # ---- through the affine core:  a_t = W_x x_t + W_h h_{t-1} + b_h ----
        g["W_x"] += np.outer(da, X[t])
        g["W_h"] += np.outer(da, hs_prev[t])
        g["b_h"] += da
        grad_h_next = P.W_h.T @ da                 # pass to step t-1

    g["b_c"] = float(g["b_c"][0])                  # restore scalar shape
    return g


# =============================================================================
# SECTION 5 — FINITE-DIFFERENCE GRADIENT CHECK  (mandatory)
# =============================================================================

def gradient_check(seed=1, eps=1e-6, tol=1e-5):
    """
    Verify backward() against central finite differences on a small instance.
    Returns the worst relative error across all parameters.
    """
    rng = np.random.default_rng(seed)
    D, H, K, T = 4, 5, 3, 6
    P = LabarumParams(D, H, K, seed=seed)
    X = rng.standard_normal((T, D))
    U = rng.random(T)
    y = int(rng.integers(K))
    lam = 0.4

    cache = forward(P, X, U, y=y, lam=lam)
    grads = backward(P, cache)

    worst = 0.0
    for name in P.names():
        val = np.atleast_1d(np.array(P.get(name), dtype=float))
        gA = np.atleast_1d(np.array(grads[name], dtype=float))
        it = np.ndindex(val.shape)
        for idx in it:
            orig = val[idx]
            val[idx] = orig + eps
            P.set(name, val.copy() if val.shape != (1,) else float(val[0]))
            Lp = forward(P, X, U, y=y, lam=lam)["loss"]
            val[idx] = orig - eps
            P.set(name, val.copy() if val.shape != (1,) else float(val[0]))
            Lm = forward(P, X, U, y=y, lam=lam)["loss"]
            val[idx] = orig
            P.set(name, val.copy() if val.shape != (1,) else float(val[0]))
            num = (Lp - Lm) / (2 * eps)
            ana = gA[idx]
            denom = max(1e-9, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            worst = max(worst, rel)
    return worst


# =============================================================================
# SECTION 6 — THE "READING THE SIGNS" DATASET
# =============================================================================

def make_signs_dataset(n, D=4, K=3, T=8, seed=0, proto=None):
    """
    Each sequence is a stream of omens about which of K rival hypotheses
    ("gods") is true. A DECISIVE sign appears at a random late step t*:

      * for t < t* : signs are ambiguous — high ambiguity U_t, and the
                     evidence is weak/noisy (barely tilts toward the truth).
      * for t >= t*: the decisive omen — low ambiguity U_t, strong evidence
                     pointing at the true hypothesis y.

    The Constantinian-optimal policy is therefore: DEFER (low commitment)
    while U is high, then COMMIT once the decisive omen lands.

    `proto` fixes the K class prototypes so train and test share the *same
    gods* (only the sampled omen sequences differ). If None, they are drawn.

    Returns X (n,T,D), U (n,T), Y (n,), tstar (n,)  and the class prototypes.
    """
    rng = np.random.default_rng(seed)
    if proto is None:
        proto = rng.standard_normal((K, D))
        proto /= np.linalg.norm(proto, axis=1, keepdims=True)

    X = np.zeros((n, T, D))
    U = np.zeros((n, T))
    Y = np.zeros(n, dtype=int)
    tstar = np.zeros(n, dtype=int)

    for i in range(n):
        y = int(rng.integers(K))
        ts = int(rng.integers(T // 2, T))   # decisive sign in the late half
        Y[i] = y
        tstar[i] = ts
        for t in range(T):
            if t < ts:
                strength = 0.15                       # murky: weak signal
                U[i, t] = rng.uniform(0.75, 1.0)      # high ambiguity
                noise = 1.0
            else:
                strength = 1.4                        # decisive omen
                U[i, t] = rng.uniform(0.0, 0.15)      # low ambiguity
                noise = 0.35
            X[i, t] = strength * proto[y] + noise * rng.standard_normal(D) * 0.5
    return X, U, Y, tstar, proto


# =============================================================================
# SECTION 7 — TRAINING LOOP  (Adam, from scratch)
# =============================================================================

def _zeros_like_params(P):
    z = {}
    for n in P.names():
        v = P.get(n)
        z[n] = np.zeros_like(np.atleast_1d(np.array(v, dtype=float)))
    return z

def train(P, data, epochs=40, lr=5e-3, lam=0.35, seed=0, verbose=True):
    X, U, Y, _, _ = data
    n = X.shape[0]
    rng = np.random.default_rng(seed)

    # Adam state
    m = _zeros_like_params(P); v = _zeros_like_params(P)
    b1, b2, epsA = 0.9, 0.999, 1e-8
    step = 0
    history = []

    for ep in range(epochs):
        order = rng.permutation(n)
        tot_loss = 0.0
        for i in order:
            cache = forward(P, X[i], U[i], y=int(Y[i]), lam=lam)
            grads = backward(P, cache)
            tot_loss += cache["loss"]
            step += 1
            for name in P.names():
                val = np.atleast_1d(np.array(P.get(name), dtype=float))
                gr = np.atleast_1d(np.array(grads[name], dtype=float))
                m[name] = b1 * m[name] + (1 - b1) * gr
                v[name] = b2 * v[name] + (1 - b2) * gr * gr
                mhat = m[name] / (1 - b1 ** step)
                vhat = v[name] / (1 - b2 ** step)
                val = val - lr * mhat / (np.sqrt(vhat) + epsA)
                P.set(name, val.copy() if val.shape != (1,) else float(val[0]))
        avg = tot_loss / n
        history.append(avg)
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            acc = evaluate(P, data, lam=lam)["acc"]
            print(f"  epoch {ep:3d} | loss {avg:7.4f} | train acc {acc:5.3f}")
    return history


def evaluate(P, data, lam=0.35):
    X, U, Y, tstar, _ = data
    n = X.shape[0]
    correct = 0
    pre_commit = []   # mean commitment BEFORE the decisive sign
    post_commit = []  # mean commitment AT/AFTER the decisive sign
    for i in range(n):
        out = forward(P, X[i], U[i], y=int(Y[i]), lam=lam)
        if int(np.argmax(out["p_T"])) == int(Y[i]):
            correct += 1
        ts = int(tstar[i])
        c = out["c"]
        if ts > 0:
            pre_commit.append(float(np.mean(c[:ts])))
        post_commit.append(float(np.mean(c[ts:])))
    return {
        "acc": correct / n,
        "pre_commit": float(np.mean(pre_commit)) if pre_commit else float("nan"),
        "post_commit": float(np.mean(post_commit)),
    }


# =============================================================================
# SECTION 8 — SELF-TESTS
# =============================================================================

def self_tests():
    print("=" * 74)
    print("THE LABARUM ENGINE  —  Constantine the Great (272-337 CE)")
    print("Deferred-Commitment Recurrent Network  |  pure NumPy, from scratch")
    print("=" * 74)

    # (1) Gradient check — mandatory.
    print("\n[1] Finite-difference gradient check (analytic vs numerical)")
    worst = gradient_check()
    print(f"    worst relative error = {worst:.3e}")
    assert worst < 1e-4, "GRADIENT CHECK FAILED"
    print("    PASS  (analytic BPTT matches finite differences)")

    # (2) Train on the 'reading the signs' task.
    print("\n[2] Training on the 'reading the signs' task")
    train_data = make_signs_dataset(400, seed=1)
    shared_proto = train_data[4]                       # same gods at test time
    test_data = make_signs_dataset(200, seed=2, proto=shared_proto)
    P = LabarumParams(D=4, H=16, K=3, seed=7)
    hist = train(P, train_data, epochs=40, lr=5e-3, lam=0.35, seed=3)
    assert hist[-1] < hist[0], "loss did not decrease"

    # (3) The learned policy must GENERALISE and must DEFER.
    print("\n[3] Evaluation on held-out omens")
    tr = evaluate(P, train_data)
    te = evaluate(P, test_data)
    print(f"    train acc = {tr['acc']:.3f}   test acc = {te['acc']:.3f}")
    print(f"    mean commitment BEFORE decisive sign = {te['pre_commit']:.3f}")
    print(f"    mean commitment AT/AFTER decisive sign = {te['post_commit']:.3f}")
    assert te["acc"] > 0.85, "test accuracy too low"
    assert te["post_commit"] > te["pre_commit"], \
        "model failed to DEFER: commitment must rise at the decisive sign"
    ratio = te["post_commit"] / max(1e-9, te["pre_commit"])
    print(f"    commitment ratio (after / before) = {ratio:.2f}x")
    print("    PASS  (the model keeps its options open, then commits on the sign)")

    # (4) Counterfactual: strip the decisive sign -> the mind stays diffuse.
    print("\n[4] Counterfactual — an emperor who never receives his sign")
    X, U, Y, tstar, proto = test_data
    i = 0
    Xflat = X[i].copy()
    Uflat = np.full_like(U[i], 0.9)          # keep it ambiguous forever
    Xflat[:] = 0.15 * proto[Y[i]] + 0.0      # only weak signals, no omen
    diffuse = forward(P, Xflat, Uflat, y=int(Y[i]))
    normal = forward(P, X[i], U[i], y=int(Y[i]))
    print(f"    final commitment WITH decisive sign    = {normal['c'][-1]:.3f}")
    print(f"    final commitment WITHOUT decisive sign = {diffuse['c'][-1]:.3f}")
    print(f"    max prob WITH sign    = {np.max(normal['p_T']):.3f}")
    print(f"    max prob WITHOUT sign = {np.max(diffuse['p_T']):.3f}")
    assert diffuse["c"][-1] < normal["c"][-1], \
        "without a decisive sign the mind should NOT collapse hard"
    print("    PASS  (no sign -> no collapse; belief stays a live superposition)")

    print("\n" + "=" * 74)
    print("ALL TESTS PASSED — the architecture runs, learns, and defers.")
    print("=" * 74)


if __name__ == "__main__":
    self_tests()

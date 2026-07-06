"""
================================================================================
chapter_0057_protagoras_-490.py  ::  The Measure Network  (Neuron.py)
Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 3 Minds 41 - 60 Available on Amazon https://www.amazon.com/dp/B0H6TVX69S
Resume and Interactive Demos at https://artificiology.com/
Author: David Vivancos · Chapter 0057 · Protagoras of Abdera
================================================================================
A from-scratch, pure-NumPy neural architecture whose *mechanism* encodes the one
cognitive idea that is Protagoras of Abdera's alone:

    pantôn chrematôn metron estin anthropos
    "Of all things the measure is man (the measurer)."  (DK 80B1)

Most networks compute a single, observer-independent representation of an input
and read a single objective truth-value off the top of it. That is precisely the
"view from nowhere" Protagoras denies. In a Protagorean network there is NO
absolute embedding. A thing is only ever encoded *relative to a measurer*, the
same stimulus yields opposite verdicts for different measurers (the wind that is
"cold to the sick man and warm to the healthy"), and the network's final stance
is not a convergence on one objective fact but a context-fit (kairos) weighting
of several measurers' verdicts — each of which is itself the difference between
TWO opposed arguments (dissoi logoi), the stronger and the weaker.

WHY THIS IS NOT A TRANSFORMER
-----------------------------
  * No attention over stored keys, no token mixing, no positional encodings.
  * The fundamental operation is a *relative* one: d_m = x - mu_m. Absolute
    offsets in the input cancel; only an input's position relative to a
    measurer's reference frame mu_m can be seen. Relativity is built into the
    first matmul, not bolted on.
  * Every measurer emits two opposed scalar logoi (pro_m, con_m). Its verdict is
    their difference. This is the dissoi-logoi doctrine made into arithmetic.
  * A kairos gate g = softmax(K c) decides, per context c, which measurer is
    fitting *now*. Truth is indexed to (measurer, occasion), never free-floating.

WHAT THE FILE CONTAINS
----------------------
  1. MeasureNetwork           : forward + exact analytic backward for every param
  2. finite_difference_check  : mandatory gradient check (analytic vs numerical)
  3. The Wind task            : a synthetic problem that is *only* solvable if the
                                model is relativistic (same x -> opposite label
                                depending on the measurer asking). A single-frame
                                "objective" baseline provably cannot fit it.
  4. train()                  : a real SGD training loop (hand-written grads)
  5. self_tests()             : softmax/shape/determinism/gradient/learning checks

Run:  python3 chapter_0057_protagoras_-490.py
Every number printed at the bottom is produced by actually executing this file.
================================================================================
"""

import numpy as np


# ------------------------------------------------------------------------------
# Small numerically-stable helpers
# ------------------------------------------------------------------------------
def softmax(z, axis=-1):
    """Stable softmax. Used ONLY for the kairos gate (perspective weighting),
    never for attention over stored keys."""
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def tanh(x):
    return np.tanh(x)


def dtanh(y):
    """Derivative of tanh given its OUTPUT y = tanh(x): 1 - y^2."""
    return 1.0 - y * y


# ==============================================================================
# THE MEASURE NETWORK
# ==============================================================================
class MeasureNetwork:
    """
    Forward (one sample x in R^D, context c in R^C):

        For each measurer m = 1..M:
            d_m   = x - mu_m                      # RELATIVE encoding: man is the measure
            p_m   = W_m d_m + b_m                 # frame-local features  (R^H)
            z_m   = tanh(p_m)
            pro_m = u_plus_m  . z_m               # the stronger logos (scalar)
            con_m = u_minus_m . z_m               # the weaker  logos (scalar)
            J_m   = pro_m - con_m                 # the measurer's verdict (dissoi logoi)

        g  = softmax(K c)                         # kairos: which measurer fits NOW
        y_hat = sum_m g_m * J_m                   # context-weighted verdict

    There is deliberately no global/objective representation of x anywhere in the
    graph. Every quantity above is indexed by a measurer m.

    Parameters (all learned):
        mu       (M, D)   measurer reference frames  ("each measurer's standard")
        W        (M, H, D) per-measurer projection
        b        (M, H)
        u_plus   (M, H)   stronger-logos read-out
        u_minus  (M, H)   weaker-logos read-out
        K        (M, C)   kairos gate
    """

    def __init__(self, D, C, M, H, seed=0):
        self.D, self.C, self.M, self.H = D, C, M, H
        rng = np.random.default_rng(seed)
        s = 0.5  # small init keeps tanh in its responsive region
        self.params = {
            "mu":      rng.standard_normal((M, D)) * s,
            "W":       rng.standard_normal((M, H, D)) * s,
            "b":       np.zeros((M, H)),
            "u_plus":  rng.standard_normal((M, H)) * s,
            "u_minus": rng.standard_normal((M, H)) * s,
            "K":       rng.standard_normal((M, C)) * s,
        }

    # --- forward, returns y_hat and a cache for backprop -----------------------
    def forward(self, X, Cc):
        """
        X  : (N, D) batch of stimuli
        Cc : (N, C) batch of context / measurer-identity vectors
        Returns y_hat (N,) and a cache dict.
        """
        P = self.params
        N = X.shape[0]
        M, H, D = self.M, self.H, self.D

        # Relative encoding for every (sample, measurer): d[n,m] = x[n] - mu[m]
        # X:(N,1,D)  mu:(1,M,D) -> d:(N,M,D)
        d = X[:, None, :] - P["mu"][None, :, :]                      # (N, M, D)

        # p[n,m] = W[m] d[n,m] + b[m]            (einsum over D)
        p = np.einsum("mhd,nmd->nmh", P["W"], d) + P["b"][None, :, :]  # (N, M, H)
        z = tanh(p)                                                   # (N, M, H)

        pro = np.einsum("mh,nmh->nm", P["u_plus"], z)                 # (N, M)
        con = np.einsum("mh,nmh->nm", P["u_minus"], z)               # (N, M)
        J = pro - con                                                # (N, M) verdicts

        gate_logits = Cc @ P["K"].T                                  # (N, M)
        g = softmax(gate_logits, axis=1)                             # (N, M) kairos

        y_hat = np.sum(g * J, axis=1)                                # (N,)

        cache = dict(X=X, Cc=Cc, d=d, p=p, z=z, pro=pro, con=con,
                     J=J, g=g, y_hat=y_hat)
        return y_hat, cache

    # --- loss ------------------------------------------------------------------
    @staticmethod
    def mse_loss(y_hat, y):
        diff = y_hat - y
        return 0.5 * np.mean(diff * diff)

    # --- exact analytic gradients ---------------------------------------------
    def backward(self, cache, y):
        """
        Returns grads dict with the same keys/shapes as self.params.
        Derived by hand; verified against finite differences below.
        """
        P = self.params
        X, Cc = cache["X"], cache["Cc"]
        d, z, g, J = cache["d"], cache["z"], cache["g"], cache["J"]
        y_hat = cache["y_hat"]
        N = X.shape[0]

        # dL/dy_hat  (mean MSE)
        dy = (y_hat - y) / N                                         # (N,)

        # y_hat = sum_m g_m J_m
        dJ = dy[:, None] * g                                        # (N, M)
        dg = dy[:, None] * J                                        # (N, M)

        # softmax gate: dL/dlogits_k = g_k (dg_k - sum_j g_j dg_j)
        gdotdg = np.sum(g * dg, axis=1, keepdims=True)              # (N,1)
        dlogits = g * (dg - gdotdg)                                 # (N, M)
        # logits = Cc @ K^T  ->  dK = dlogits^T @ Cc
        dK = dlogits.T @ Cc                                         # (M, C)

        # J_m = pro_m - con_m
        dpro = dJ                                                   # (N, M)
        dcon = -dJ                                                  # (N, M)

        # pro_m = u_plus_m . z_m ; con_m = u_minus_m . z_m
        du_plus = np.einsum("nm,nmh->mh", dpro, z)                  # (M, H)
        du_minus = np.einsum("nm,nmh->mh", dcon, z)                 # (M, H)

        # dL/dz = dpro * u_plus + dcon * u_minus
        dz = (dpro[:, :, None] * P["u_plus"][None, :, :]
              + dcon[:, :, None] * P["u_minus"][None, :, :])        # (N, M, H)

        # z = tanh(p)
        dp = dz * dtanh(z)                                          # (N, M, H)

        db = np.einsum("nmh->mh", dp)                               # (M, H)
        # p = einsum('mhd,nmd->nmh', W, d)
        dW = np.einsum("nmh,nmd->mhd", dp, d)                       # (M, H, D)
        dd = np.einsum("mhd,nmh->nmd", P["W"], dp)                  # (N, M, D)

        # d = X - mu  ->  dmu = -sum_n dd
        dmu = -np.einsum("nmd->md", dd)                            # (M, D)

        return dict(mu=dmu, W=dW, b=db,
                    u_plus=du_plus, u_minus=du_minus, K=dK)

    # --- convenience: report how much each logos contributed (dissoi balance) --
    def logoi_balance(self, cache):
        """Mean |pro| and |con| across the batch — shows both arguments are live,
        not a degenerate single-sided read-out."""
        return float(np.mean(np.abs(cache["pro"]))), float(np.mean(np.abs(cache["con"])))


# ==============================================================================
# GRADIENT CHECK  (mandatory)
# ==============================================================================
def finite_difference_check(model, X, Cc, y, eps=1e-6):
    """
    Compare every analytic gradient against a central finite difference.
    Returns the maximum relative error over all parameters.
    """
    y_hat, cache = model.forward(X, Cc)
    grads = model.backward(cache, y)

    max_rel = 0.0
    worst = None
    for name, P in model.params.items():
        G = grads[name]
        flat = P.ravel()
        gflat = G.ravel()
        # check a handful of coordinates per tensor (cheap but thorough)
        idxs = np.linspace(0, flat.size - 1, num=min(12, flat.size)).astype(int)
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps
            lp = model.mse_loss(model.forward(X, Cc)[0], y)
            flat[i] = orig - eps
            lm = model.mse_loss(model.forward(X, Cc)[0], y)
            flat[i] = orig
            num = (lp - lm) / (2 * eps)
            ana = gflat[i]
            denom = max(1e-8, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            if rel > max_rel:
                max_rel, worst = rel, (name, i, ana, num)
    return max_rel, worst


# ==============================================================================
# THE WIND  —  a task that is solvable ONLY if the model is relativistic
# ==============================================================================
def make_wind_dataset(n_per_measurer=300, D=4, seed=1):
    """
    Two measurers (sick, healthy). A stimulus x in R^D is a "wind".
    Each measurer judges warmth by projecting x onto its OWN axis w_true[m]:
        label = +1 if w_true[m] . x > 0 else -1
    The two axes are anti-correlated, so for many winds the SAME x is
    "warm" to one measurer and "cold" to the other. The context vector c is a
    one-hot telling the model WHO is judging.

    A model with no per-measurer frame (single objective verdict, ignoring c)
    must give one answer for each x and therefore cannot exceed ~chance on the
    winds where the two measurers disagree. The Measure Network can.
    """
    rng = np.random.default_rng(seed)
    C = 2
    w_sick = rng.standard_normal(D)
    w_sick /= np.linalg.norm(w_sick)
    # healthy axis: strongly opposed to the sick axis (anti-correlated)
    w_healthy = -w_sick + 0.25 * rng.standard_normal(D)
    w_healthy /= np.linalg.norm(w_healthy)
    w_true = [w_sick, w_healthy]

    X, Cc, y = [], [], []
    for m in range(C):
        xs = rng.standard_normal((n_per_measurer, D))
        labels = np.where(xs @ w_true[m] > 0, 1.0, -1.0)
        ctx = np.zeros((n_per_measurer, C))
        ctx[:, m] = 1.0
        X.append(xs); Cc.append(ctx); y.append(labels)
    X = np.concatenate(X); Cc = np.concatenate(Cc); y = np.concatenate(y)

    # shuffle
    perm = rng.permutation(len(y))
    return X[perm], Cc[perm], y[perm], w_true


def accuracy(y_hat, y):
    return float(np.mean(np.sign(y_hat) == np.sign(y)))


def disagreement_rate(X, w_true):
    """Fraction of winds the two measurers label oppositely — the part of the
    task that defeats any non-relativistic model."""
    a = np.sign(X @ w_true[0])
    b = np.sign(X @ w_true[1])
    return float(np.mean(a != b))


# ==============================================================================
# TRAINING LOOP  (real SGD with hand-written gradients)
# ==============================================================================
def train(model, X, Cc, y, lr=0.05, epochs=400, batch=64, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    N = len(y)
    history = []
    for ep in range(epochs):
        idx = rng.permutation(N)
        for s in range(0, N, batch):
            b = idx[s:s + batch]
            _, cache = model.forward(X[b], Cc[b])
            grads = model.backward(cache, y[b])
            for k in model.params:
                model.params[k] -= lr * grads[k]
        if verbose and (ep % 50 == 0 or ep == epochs - 1):
            yh, _ = model.forward(X, Cc)
            history.append((ep, model.mse_loss(yh, y), accuracy(yh, y)))
    yh, _ = model.forward(X, Cc)
    return model.mse_loss(yh, y), accuracy(yh, y), history


# ==============================================================================
# SELF TESTS
# ==============================================================================
def self_tests():
    print("=" * 78)
    print("THE MEASURE NETWORK  ::  self-tests")
    print("=" * 78)

    # ---- 1. softmax sanity --------------------------------------------------
    s = softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    assert np.allclose(s.sum(axis=1), 1.0), "softmax rows must sum to 1"
    print(f"[1] softmax rows sum to 1                          ... OK  ({s.sum(axis=1)})")

    # ---- 2. shapes ----------------------------------------------------------
    D, C, M, H = 4, 2, 4, 8
    net = MeasureNetwork(D=D, C=C, M=M, H=H, seed=7)
    Xt = np.random.default_rng(3).standard_normal((10, D))
    Ct = np.eye(C)[np.random.default_rng(4).integers(0, C, size=10)]
    yh, cache = net.forward(Xt, Ct)
    assert yh.shape == (10,), "y_hat shape"
    assert cache["g"].shape == (10, M), "gate shape"
    assert cache["J"].shape == (10, M), "verdict shape"
    print(f"[2] forward shapes  y_hat{yh.shape}  gate{cache['g'].shape}  ... OK")

    # ---- 3. gradient check (THE mandatory one) ------------------------------
    yt = np.random.default_rng(5).choice([-1.0, 1.0], size=10)
    max_rel, worst = finite_difference_check(net, Xt, Ct, yt, eps=1e-6)
    print(f"[3] finite-difference gradient check  max rel err = {max_rel:.3e}")
    print(f"      worst @ {worst[0]} idx{worst[1]}: analytic={worst[2]:+.6e} numeric={worst[3]:+.6e}")
    assert max_rel < 1e-4, "gradient check failed"
    print("      gradient check                                ... OK")

    # ---- 4. determinism -----------------------------------------------------
    a = MeasureNetwork(D, C, M, H, seed=11).forward(Xt, Ct)[0]
    b = MeasureNetwork(D, C, M, H, seed=11).forward(Xt, Ct)[0]
    assert np.allclose(a, b), "same seed must give same forward"
    print("[4] determinism (same seed => same output)          ... OK")

    # ---- 5. THE WIND: relativism is necessary -------------------------------
    print("-" * 78)
    print("[5] The Wind task  (same stimulus, opposite truth per measurer)")
    X, Cc, y, w_true = make_wind_dataset(n_per_measurer=400, D=4, seed=2)
    dis = disagreement_rate(X, w_true)
    print(f"      measurers disagree on {dis*100:5.1f}% of winds "
          f"(a non-relativist caps near {100-dis*50:.0f}% acc on the set)")

    # 5a. Protagorean Measure Network (M frames + kairos gate)
    measure = MeasureNetwork(D=4, C=2, M=4, H=12, seed=0)
    mloss, macc, _ = train(measure, X, Cc, y, lr=0.05, epochs=400, batch=64,
                           seed=0, verbose=False)
    pbar, cbar = measure.logoi_balance(measure.forward(X, Cc)[1])
    print(f"      Measure Network (relativistic): loss={mloss:.4f}  acc={macc*100:5.1f}%")
    print(f"        dissoi-logoi balance  mean|pro|={pbar:.3f}  mean|con|={cbar:.3f} "
          f"(both arguments live)")

    # 5b. "Objective" baseline: ONE measurer, gate degenerate, context ignored.
    #     Same code path, M=1 => no perspective, no kairos. The view from nowhere.
    objective = MeasureNetwork(D=4, C=2, M=1, H=12, seed=0)
    oloss, oacc, _ = train(objective, X, Cc, y, lr=0.05, epochs=400, batch=64,
                          seed=0, verbose=False)
    print(f"      Objective baseline   (M=1, no frame): loss={oloss:.4f}  acc={oacc*100:5.1f}%")

    gap = (macc - oacc) * 100
    print(f"      ==> relativism buys {gap:+.1f} accuracy points on this task")
    assert macc > 0.85, "Measure Network should solve the wind task"
    assert macc - oacc > 0.20, "relativism must clearly beat the objective baseline"
    print("      relativism is necessary, not decorative          ... OK")

    print("=" * 78)
    print("ALL SELF-TESTS PASSED")
    print("=" * 78)
    return dict(max_rel=max_rel, measure_acc=macc, objective_acc=oacc,
                disagreement=dis)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    results = self_tests()
    print()
    print("SUMMARY (every figure produced by executing this file):")
    print(f"  gradient-check max relative error : {results['max_rel']:.3e}")
    print(f"  measurers-disagree fraction       : {results['disagreement']*100:.1f}%")
    print(f"  Measure Network accuracy          : {results['measure_acc']*100:.1f}%")
    print(f"  Objective baseline accuracy       : {results['objective_acc']*100:.1f}%")

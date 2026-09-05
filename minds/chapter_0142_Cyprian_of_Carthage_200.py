#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
======================================================================================
 THE COLLEGIAL COMMUNION NETWORK  (CCN)
 Chapter 0142_Cyprian_of_Carthage_200 - Cyprian of Carthage (c. 200-258 CE)
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 8 Minds 141 - 160 Available on Amazon https://www.amazon.com/dp/B0HH8RTCXF
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 142: Cyprian of Carthage (200-258 CE)
================================================================================  

WHY THIS ARCHITECTURE, AND WHY IT IS *NOT* A TRANSFORMER
--------------------------------------------------------
Cyprian's whole theory of mind is a theory of the *Church*, not of the solitary
soul. Where his master Tertullian asks what a single mind is made of, Cyprian asks
how many minds stay *one*. His central sentence -- "episcopatus unus est, cuius a
singulis in solidum pars tenetur" (the episcopate is one, of which each holds a
part *in solidum*, i.e. for the whole) -- is, read as engineering, a specification
for a distributed system:

  1. IN SOLIDUM (each holds the whole).  Every node is a *full* predictor over the
     whole task, never a shard of a bigger model. Cyprian's bishops are not layers
     in a pipeline; each is a complete bishop.

  2. CONCORDIA IS THE INVARIANT.  Unity is not a nice-to-have; it is THE quantity to
     preserve. So the network's primary structural loss penalises divergence of any
     node from the college consensus. Truth propagates only through an unbroken body.

  3. EXTRA ECCLESIAM (validity by membership).  "salus extra ecclesiam non est."
     A signal from outside the communion is void *however correct its outward form*.
     The network refuses out-of-communion inputs at inference regardless of content.

  4. DE LAPSIS (graded re-admission).  A node that defects (drifts far from the
     college) is neither trusted-as-clean nor exiled forever. It is placed under
     *penance* proportioned to the severity of its lapse, and its standing in the
     college's weighting is restored only gradually as it re-aligns.

  5. NO BISHOP OF BISHOPS.  (Council of Carthage, 256: "neque enim quisquam nostrum
     episcopum se esse episcoporum constituit.")  Aggregation is symmetric: no node
     is privileged, no central parameter-server casts the deciding vote. The college
     output is invariant to any permutation of the bishops.

MECHANISM (all differentiable except the governance layer, which is deliberately
so -- penance is a *decision*, not a gradient):

  * N bishop-experts, each a small tanh MLP producing a full scalar verdict o_i.
  * CONCORD WEIGHTS: nodes nearer the college mean earn more weight,
        s_i = -beta * (o_i - obar)^2 + log_trust_i ,   alpha = softmax(s).
    The seamless robe reinforces itself: coherence begets influence.
  * COLLEGE VERDICT:  yhat = sum_i alpha_i * o_i.
  * LOSS:  L = 1/2 (yhat - y)^2  +  1/2 * lam * mean_i (o_i - obar)^2   (concordia).
  * GOVERNANCE (training-time, non-differentiable): a LapseRegistry watches each
    node's running divergence; a lapse triggers a penance whose length scales with
    severity; log_trust_i is driven down and annealed back over the penance, entering
    the concord logits as a constant bias (so the analytic gradient is unchanged and
    the mandatory finite-difference check still passes with penance off).

CONVENTIONS (kept identical across the corpus): pure NumPy, from scratch; a passing
finite-difference gradient check (MANDATORY); a real training loop; self-tests; the
file is executed before shipping and its output recorded.
======================================================================================
"""

import numpy as np

RNG = np.random.default_rng(2000 + 258)  # born ~200, martyred 258


# ------------------------------------------------------------------------------------
# 1.  THE COLLEGE  --  N bishops, each a complete predictor ("in solidum")
# ------------------------------------------------------------------------------------
class CollegialCommunionNetwork:
    """A leaderless college of tanh-MLP 'bishops' bound by a concordia consensus head."""

    def __init__(self, d_in, hidden, n_bishops=7, beta=4.0, unity_lambda=0.30, seed=0):
        self.d_in = d_in
        self.H = hidden
        self.N = n_bishops
        self.beta = float(beta)          # sharpness of concord weighting
        self.lam = float(unity_lambda)   # weight of the concordia (unity) loss
        r = np.random.default_rng(seed)

        # Each bishop is a FULL predictor: W1 (H,d_in), b1 (H,), w2 (H,), b2 (scalar).
        s1 = np.sqrt(1.0 / d_in)
        s2 = np.sqrt(1.0 / hidden)
        self.W1 = r.normal(0, s1, size=(self.N, self.H, d_in))
        self.b1 = np.zeros((self.N, self.H))
        self.w2 = r.normal(0, s2, size=(self.N, self.H))
        self.b2 = np.zeros(self.N)

    # -- parameter <-> flat-vector helpers (used only by the gradient check) ----------
    def get_params(self):
        return [self.W1, self.b1, self.w2, self.b2]

    def pack(self):
        return np.concatenate([p.ravel() for p in self.get_params()])

    def unpack(self, vec):
        i = 0
        for p in self.get_params():
            n = p.size
            p[...] = vec[i:i + n].reshape(p.shape)
            i += n

    # -- forward for ONE example ------------------------------------------------------
    def forward(self, x, log_trust=None):
        """
        x: (d_in,) ; log_trust: (N,) constant governance bias (default zeros).
        Returns (yhat, cache).
        """
        if log_trust is None:
            log_trust = np.zeros(self.N)

        z1 = self.W1 @ x + self.b1          # (N,H)
        h = np.tanh(z1)                     # (N,H)
        o = np.einsum('nh,nh->n', self.w2, h) + self.b2   # (N,) each bishop's verdict

        obar = o.mean()
        d = o - obar                        # divergence from the college
        s = -self.beta * d * d + log_trust  # concord logits (+ constant trust bias)
        s = s - s.max()                     # softmax stability
        e = np.exp(s)
        alpha = e / e.sum()                 # communion weights (sum to 1)
        yhat = float(alpha @ o)             # the college's single verdict

        cache = dict(x=x, z1=z1, h=h, o=o, obar=obar, d=d, alpha=alpha)
        return yhat, cache

    # -- loss for ONE example ---------------------------------------------------------
    def loss(self, yhat, cache, y):
        r = yhat - y
        L_task = 0.5 * r * r
        L_unity = 0.5 * self.lam * np.mean(cache['d'] ** 2)   # concordia penalty
        return float(L_task + L_unity), r

    # -- analytic backward for ONE example -------------------------------------------
    def backward(self, cache, r):
        """
        Hand-derived gradients. The delicate part is dL/do through the consensus head,
        because alpha depends on o via the divergence d. N is tiny, so we build the
        exact (N,N) Jacobians explicitly rather than approximating.
        """
        o, d, alpha = cache['o'], cache['d'], cache['alpha']
        h, x = cache['h'], cache['x']
        N = self.N

        # ---- dL_task/do through the aggregation head -------------------------------
        M = np.eye(N) - np.ones((N, N)) / N          # d d_k / d o_j  (symmetric)
        ds_do = (-2.0 * self.beta * d)[:, None] * M   # d s_k / d o_j
        A = np.diag(alpha) - np.outer(alpha, alpha)   # d alpha_i / d s_k
        J_alpha = A @ ds_do                           # d alpha_i / d o_j  (N,N)
        dyhat_do = o @ J_alpha + alpha                # d yhat / d o_j
        dLtask_do = r * dyhat_do

        # ---- dL_unity/do : with sum_k d_k = 0 this reduces to (lam/N) * d ----------
        dLunity_do = (self.lam / N) * d

        dL_do = dLtask_do + dLunity_do                # (N,)

        # ---- per-bishop MLP backprop (independent given dL/do_i) -------------------
        gW1 = np.zeros_like(self.W1)
        gb1 = np.zeros_like(self.b1)
        gw2 = np.zeros_like(self.w2)
        gb2 = np.zeros_like(self.b2)
        for i in range(N):
            gi = dL_do[i]
            gb2[i] = gi
            gw2[i] = gi * h[i]
            dh = gi * self.w2[i]
            dz1 = dh * (1.0 - h[i] ** 2)
            gW1[i] = np.outer(dz1, x)
            gb1[i] = dz1
        return [gW1, gb1, gw2, gb2]

    # -- batch gradient (sum over examples) ------------------------------------------
    def batch_grads(self, X, Y, log_trust=None):
        grads = [np.zeros_like(p) for p in self.get_params()]
        total = 0.0
        divs = []
        for x, y in zip(X, Y):
            yhat, cache = self.forward(x, log_trust)
            L, r = self.loss(yhat, cache, y)
            total += L
            divs.append(np.abs(cache['d']))
            g = self.backward(cache, r)
            for acc, gi in zip(grads, g):
                acc += gi
        n = len(X)
        grads = [g / n for g in grads]
        return total / n, grads, np.mean(divs, axis=0)   # mean |divergence| per bishop

    # -- inference with the EXTRA-ECCLESIAM membership gate --------------------------
    def predict(self, x, in_communion=True, log_trust=None):
        """A signal outside the communion is void however correct its form."""
        if not in_communion:
            return None, "ABSTAIN: extra ecclesiam nihil valet (out of communion)"
        yhat, _ = self.forward(x, log_trust)
        return yhat, "in communion"


# ------------------------------------------------------------------------------------
# 2.  GOVERNANCE  --  De Lapsis: lapse detection + graded, proportioned penance
# ------------------------------------------------------------------------------------
class LapseRegistry:
    """
    Watches each bishop's running divergence. When a node drifts past the lapse
    threshold it is placed under penance whose LENGTH scales with the severity of the
    lapse. During penance its trust (and thus its weight in the college) is suppressed
    and annealed back toward full communion -- never instantly restored, never
    permanently exiled. This is Cyprian's answer to the 'libellatici': readmission is
    real but must be *earned* in proportion to the fall.
    """

    def __init__(self, n, lapse_threshold=0.55, ema=0.55,
                 base_penance=40, severity_scale=120.0, floor_trust=0.02):
        self.n = n
        self.thr = lapse_threshold
        self.ema = ema
        self.base = base_penance
        self.sev = severity_scale
        self.floor = floor_trust
        self.run_div = np.zeros(n)     # EMA of |divergence|
        self.penance = np.zeros(n, dtype=int)   # steps of penance remaining
        self.penance_len = np.ones(n, dtype=int)  # total length of current penance
        self.events = []               # (step, node, severity, penance_len)

    def observe(self, step, div_vec):
        self.run_div = self.ema * self.run_div + (1 - self.ema) * div_vec
        # detect new lapses (only if not already doing penance)
        for i in range(self.n):
            if self.penance[i] == 0 and self.run_div[i] > self.thr:
                severity = self.run_div[i] - self.thr
                plen = int(self.base + self.sev * severity)
                self.penance[i] = plen
                self.penance_len[i] = plen
                self.events.append((step, i, round(float(severity), 3), plen))

    def log_trust(self):
        """Return per-node log-trust bias for the concord logits (0 == full communion)."""
        trust = np.ones(self.n)
        for i in range(self.n):
            if self.penance[i] > 0:
                # fraction of penance served -> trust ramps from floor back to 1
                served = 1.0 - self.penance[i] / max(1, self.penance_len[i])
                trust[i] = self.floor + (1.0 - self.floor) * served
        return np.log(np.clip(trust, 1e-6, 1.0))

    def step_down(self):
        self.penance = np.maximum(0, self.penance - 1)


# ------------------------------------------------------------------------------------
# 3.  ADAM optimiser (plain, per-parameter)
# ------------------------------------------------------------------------------------
class Adam:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


# ------------------------------------------------------------------------------------
# 4.  MANDATORY finite-difference gradient check
# ------------------------------------------------------------------------------------
def gradient_check():
    print("=" * 78)
    print(" GRADIENT CHECK  (central finite differences, penance OFF, float64)")
    print("=" * 78)
    net = CollegialCommunionNetwork(d_in=4, hidden=6, n_bishops=5,
                                    beta=3.0, unity_lambda=0.4, seed=7)
    X = RNG.normal(size=(3, 4))
    Y = RNG.normal(size=3)

    _, analytic, _ = net.batch_grads(X, Y)
    analytic_flat = np.concatenate([g.ravel() for g in analytic])

    theta = net.pack()
    eps = 1e-6
    num = np.zeros_like(theta)
    for k in range(theta.size):
        old = theta[k]
        theta[k] = old + eps; net.unpack(theta)
        Lp = 0.0
        for x, y in zip(X, Y):
            yhat, c = net.forward(x); Lp += net.loss(yhat, c, y)[0]
        Lp /= len(X)
        theta[k] = old - eps; net.unpack(theta)
        Lm = 0.0
        for x, y in zip(X, Y):
            yhat, c = net.forward(x); Lm += net.loss(yhat, c, y)[0]
        Lm /= len(X)
        num[k] = (Lp - Lm) / (2 * eps)
        theta[k] = old; net.unpack(theta)

    rel = np.linalg.norm(analytic_flat - num) / (
        np.linalg.norm(analytic_flat) + np.linalg.norm(num) + 1e-12)
    max_abs = np.max(np.abs(analytic_flat - num))
    print(f"  params checked        : {theta.size}")
    print(f"  relative error (L2)   : {rel:.3e}")
    print(f"  max abs difference    : {max_abs:.3e}")
    ok = rel < 1e-6
    print(f"  RESULT                : {'PASS' if ok else 'FAIL'}  (threshold 1e-6)")
    assert ok, "Gradient check FAILED"
    return ok


# ------------------------------------------------------------------------------------
# 5.  A real training task + the four doctrinal self-tests
# ------------------------------------------------------------------------------------
def make_task(n=400, d=4):
    X = RNG.normal(size=(n, d))
    # a nonlinear target every complete bishop can, in principle, learn on its own
    Y = (np.sin(1.3 * X[:, 0]) + 0.5 * X[:, 1] * X[:, 2]
         - 0.4 * X[:, 3] ** 2 + 0.2 * X[:, 0] * X[:, 3])
    return X, Y


def train():
    print("\n" + "=" * 78)
    print(" TRAINING  --  the college learns while concordia is held as the invariant")
    print("=" * 78)
    Xtr, Ytr = make_task(400)
    net = CollegialCommunionNetwork(d_in=4, hidden=10, n_bishops=7,
                                    beta=4.0, unity_lambda=0.8, seed=3)
    # lapse threshold sits well above ambient concord noise so only a genuine
    # defection trips it -- concordia is the norm, apostasy the exception.
    reg = LapseRegistry(net.N, lapse_threshold=0.9,
                        base_penance=30, severity_scale=80.0)
    opt = Adam(net.get_params(), lr=4e-3)

    epochs = 90
    batch = 32
    lapse_injected = False
    for ep in range(epochs):
        idx = RNG.permutation(len(Xtr))
        ep_loss = 0.0
        nb = 0
        for b in range(0, len(Xtr), batch):
            sel = idx[b:b + batch]
            log_trust = reg.log_trust()
            L, grads, div = net.batch_grads(Xtr[sel], Ytr[sel], log_trust=log_trust)
            opt.step(net.get_params(), grads)
            reg.observe(ep * (len(Xtr) // batch) + nb, div)
            reg.step_down()
            ep_loss += L
            nb += 1

        # --- DE LAPSIS demonstration: force one bishop to defect at epoch 20 --------
        if ep == 15 and not lapse_injected:
            net.W1[3] += RNG.normal(0, 1.2, size=net.W1[3].shape)  # bishop 3 apostatises
            net.w2[3] += RNG.normal(0, 1.2, size=net.w2[3].shape)
            lapse_injected = True
            print(f"  [epoch {ep:2d}] >> bishop #3 forced into a large lapse (corruption injected)")

        if ep % 6 == 0 or ep in (15, 16, 18):
            _, _, div = net.batch_grads(Xtr[:64], Ytr[:64])
            pen = reg.penance.copy()
            print(f"  epoch {ep:2d} | loss {ep_loss/nb:7.4f} | "
                  f"mean|div| {div.mean():.3f} | max|div| {div.max():.3f} | "
                  f"penance {pen.tolist()}")

    print("\n  Lapse/penance events (step, bishop, severity, penance_len):")
    for e in reg.events:
        print("    ", e)
    return net, reg, Xtr, Ytr


def self_tests(net, reg, Xtr, Ytr):
    print("\n" + "=" * 78)
    print(" SELF-TESTS  (the doctrine, made executable)")
    print("=" * 78)
    passed = 0
    total = 0

    # TEST 1 -- learning actually happened -----------------------------------------
    total += 1
    preds = np.array([net.predict(x)[0] for x in Xtr[:200]])
    mse = float(np.mean((preds - Ytr[:200]) ** 2))
    var = float(np.var(Ytr[:200]))
    r2 = 1 - mse / var
    ok = r2 > 0.5
    print(f"  [1] college fit           : R^2={r2:.3f} on train  -> {'PASS' if ok else 'FAIL'}")
    passed += ok

    # TEST 2 -- CONCORDIA: college stays coherent ----------------------------------
    total += 1
    _, _, div = net.batch_grads(Xtr[:200], Ytr[:200])
    ok = div.mean() < 0.3
    print(f"  [2] concordia (unity)     : mean|div|={div.mean():.3f} (<0.3) -> {'PASS' if ok else 'FAIL'}")
    passed += ok

    # TEST 3 -- DE LAPSIS: the lapsed bishop was flagged and then restored ----------
    total += 1
    flagged = any(e[1] == 3 for e in reg.events)
    # "restored" = penance fully served AND the once-lapsed bishop is back well within
    # the communion band (far below the 0.9 lapse line). De Lapsis: readmission is real,
    # earned by proportioned penance -- not instant, not permanent exile.
    served = reg.penance[3] == 0
    back_in_band = div[3] < 0.35
    restored = served and back_in_band
    ok = flagged and restored
    print(f"  [3] de lapsis recovery    : flagged={flagged}, penance_served={served}, "
          f"div[3]={div[3]:.3f}<0.35={back_in_band} -> {'PASS' if ok else 'FAIL'}")
    passed += ok

    # TEST 4 -- EXTRA ECCLESIAM gate: out-of-communion input is refused -------------
    total += 1
    x = Xtr[0]
    y_in, _ = net.predict(x, in_communion=True)
    y_out, msg = net.predict(x, in_communion=False)
    ok = (y_in is not None) and (y_out is None)
    print(f"  [4] extra ecclesiam gate  : in={y_in is not None}, out={y_out is None} "
          f"({msg}) -> {'PASS' if ok else 'FAIL'}")
    passed += ok

    # TEST 5 -- NO BISHOP OF BISHOPS: verdict invariant to permuting the college ----
    total += 1
    x = Xtr[5]
    y0, _ = net.forward(x)
    perm = RNG.permutation(net.N)
    net2 = CollegialCommunionNetwork(net.d_in, net.H, net.N, net.beta, net.lam)
    net2.W1 = net.W1[perm].copy(); net2.b1 = net.b1[perm].copy()
    net2.w2 = net.w2[perm].copy(); net2.b2 = net.b2[perm].copy()
    y1, _ = net2.forward(x)
    ok = abs(y0 - y1) < 1e-10
    print(f"  [5] no bishop of bishops  : |verdict diff under permutation|={abs(y0-y1):.2e} "
          f"-> {'PASS' if ok else 'FAIL'}")
    passed += ok

    print("-" * 78)
    print(f"  SELF-TESTS PASSED: {passed}/{total}")
    assert passed == total, "One or more doctrinal self-tests failed"
    return passed == total


# ------------------------------------------------------------------------------------
if __name__ == "__main__":
    print(__doc__)
    gradient_check()
    net, reg, Xtr, Ytr = train()
    self_tests(net, reg, Xtr, Ytr)
    print("\nAll checks complete. The college holds; the robe is unbroken.\n")

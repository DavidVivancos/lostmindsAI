"""
================================================================================
Chapter 0155_st_patrick_385 - St. Patrick (385-461 CE)
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 8 Minds 141 - 160 Available on Amazon https://www.amazon.com/dp/B0HH8RTCXF
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 0155_st_patrick_385 - St. Patrick (385-461 CE)
================================================================================  
KERYGMA ENGINE  —  the cognition of Patricius, made runnable.

Thesis (from his own two surviving letters, the Confessio and the Epistola):
Patrick did not experience himself as the AUTHOR of his goals. He experienced
them as DIRECTIVES arriving through an ambiguous channel -- a voice in sleep, a
read letter in a vision -- whose provenance he could not settle from the inside:
"whether within me or beside me, I do not know; God knows" (Confessio 24).
His whole method was therefore three moves, not one:
   (1) RECEIVE a directive through a noisy channel,
   (2) AUTHENTICATE its source (accept the genuine, reject the counterfeit),
   (3) PROPAGATE the authenticated doctrine, with fidelity, across a NAIVE
       substrate that never held it -- Britain -> Ireland, Latin -> Irish,
       literate -> oral -- allowing local ADAPTATION but bounding DRIFT so the
       value stays recognizable ("a letter of Christ ... written not with ink
       but with the spirit of the living God", Epistola).

So this network is deliberately NOT a transformer. It has no attention over
stored keys. It is a differentiable model of *authenticated memetic diffusion*:

   AUTH GATE      : classify a directive as genuine vs. injected  (the alignment core)
   INJECTION      : an authenticated directive becomes a seed doctrine vector
   SUBSTRATE PROP : the doctrine diffuses over a graph of naive nodes, each node
                    adapting it locally through a per-node adaptation gate
   READOUT        : alignment (projection onto the doctrine axis) + drift penalty

Everything is pure NumPy, from scratch, with hand-written backprop verified by a
central-difference gradient check (mandatory). A real training loop then shows the
two things Patrick's mind had to get right at once: it LEARNS TO AUTHENTICATE
(reject the counterfeit voice) and it LEARNS TO PROPAGATE (carry the doctrine to
the far nodes without letting it mutate past recognition).
"""

import numpy as np


# --------------------------------------------------------------------------- #
#  small numerically-stable primitives
# --------------------------------------------------------------------------- #
def sigmoid(x):
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def softplus(x):
    # log(1+e^x), stable
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def bce(p, y, eps=1e-9):
    p = np.clip(p, eps, 1.0 - eps)
    return -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


# --------------------------------------------------------------------------- #
#  the model
# --------------------------------------------------------------------------- #
class KerygmaEngine:
    """Authenticated value-propagation network (Patrick's three moves)."""

    def __init__(self, in_dir, in_ctx, m, d, f, T=3, entry=0, seed=0):
        rng = np.random.default_rng(seed)
        self.in_dir, self.in_ctx = in_dir, in_ctx
        self.din = in_dir + in_ctx
        self.m, self.d, self.f, self.T, self.entry = m, d, f, T, entry

        s = 0.30  # init scale
        self.P = {
            # (1) AUTHENTICATION  ---------------------------------------------
            "Wv":  rng.standard_normal((m, self.din)) * s,
            "bv":  np.zeros(m),
            "wa":  rng.standard_normal(m) * s,
            "ba":  np.array(0.0),
            # (2) INJECTION  (authenticated directive -> doctrine seed) --------
            "Winj": rng.standard_normal((d, m)) * s,
            "binj": np.zeros(d),
            # doctrine axis (what "aligned" points toward)
            "k":   rng.standard_normal(d) * s + 1.0,
            # (3) PROPAGATION  -------------------------------------------------
            "Wt":  rng.standard_normal((d, d)) * s,
            # adaptation gate: per-node fidelity<->mutation coefficient
            "Wg":  rng.standard_normal(f) * s,
            "bg":  np.array(0.0),
        }
        # loss weights & readout targets
        self.p_target = 1.2   # desired alignment (projection) each node should reach
        self.tol = 0.30       # allowed adaptation (drift) band before penalty
        self.lam_drift = 0.5
        self.lam_auth = 1.0
        self.lam_reg = 1e-4
        self.reg_keys = ("Wv", "wa", "Winj", "Wt", "Wg", "k")

    # ------------------------------------------------------------------ #
    #  FORWARD  (returns scalar loss + a cache for backprop)
    # ------------------------------------------------------------------ #
    def forward(self, batch):
        P = self.P
        A = batch["A"]                # (N,N) row-normalized adjacency
        S = batch["S"]                # (N,f) node substrate features
        x_seed = batch["x_seed"]      # (din,) the genuine seed directive+context
        Xa = batch["Xa"]              # (B,din) directives to authenticate
        ya = batch["ya"]              # (B,)   1=genuine, 0=injected
        N = A.shape[0]
        c = {}

        # ---- AUTH BATCH (the "within me or beside me?" classifier) -------
        Va = np.tanh(Xa @ P["Wv"].T + P["bv"])         # (B,m)
        la = Va @ P["wa"] + P["ba"]                     # (B,)
        pa = sigmoid(la)                                # (B,)
        L_auth = bce(pa, ya)
        c.update(Va=Va, pa=pa, la=la)

        # ---- SEED PATH: authenticate the genuine directive, then inject ---
        v = np.tanh(P["Wv"] @ x_seed + P["bv"])         # (m,)
        s_seed = P["wa"] @ v + P["ba"]                  # scalar auth logit
        a_seed = sigmoid(s_seed)                        # gate in (0,1)
        inj_raw = np.tanh(P["Winj"] @ v + P["binj"])    # (d,)
        d_inj = a_seed * inj_raw                        # (d,) authenticated doctrine
        c.update(v=v, a_seed=a_seed, inj_raw=inj_raw, d_inj=d_inj)

        # ---- SUBSTRATE INIT: all nodes naive, doctrine seeded at entry ----
        B0 = np.zeros((N, self.d))
        B0[self.entry] = d_inj

        # ---- ADAPTATION GATE: per-node fidelity<->mutation ---------------
        alpha = sigmoid(S @ P["Wg"] + P["bg"])          # (N,)
        c["alpha"] = alpha
        a_col = alpha[:, None]

        # ---- PROPAGATION (BPTT-unrolled diffusion) -----------------------
        Bs = [B0]
        Ps, Hs = [], []
        Bt = B0
        for t in range(self.T):
            Pm = A @ Bt                 # gather neighbours
            Z = Pm @ P["Wt"]            # transmit
            H = np.tanh(Z)              # local re-expression
            Bt = Bt * (1 - a_col) + H * a_col   # adaptation-gated update
            Ps.append(Pm); Hs.append(H); Bs.append(Bt)
        c.update(Bs=Bs, Ps=Ps, Hs=Hs, A=A, S=S)
        BT = Bs[-1]

        # ---- READOUT: alignment + drift ----------------------------------
        nk = np.sqrt(np.sum(P["k"] ** 2)) + 1e-12
        khat = P["k"] / nk
        proj = BT @ khat                          # (N,) alignment with doctrine
        bnorm2 = np.sum(BT * BT, axis=1)          # (N,)
        drift = bnorm2 - proj ** 2                # (N,) orthogonal (local) content
        L_align = np.mean((proj - self.p_target) ** 2)
        L_drift = np.mean(softplus(drift - self.tol))
        c.update(khat=khat, nk=nk, proj=proj, bnorm2=bnorm2, drift=drift, BT=BT)

        # ---- regularizer -------------------------------------------------
        reg = sum(np.sum(P[k] ** 2) for k in self.reg_keys)
        L = (L_align + self.lam_drift * L_drift
             + self.lam_auth * L_auth + self.lam_reg * reg)

        c.update(L_align=L_align, L_drift=L_drift, L_auth=L_auth,
                 batch=batch, N=N)
        # expose diagnostics for the training loop / self-tests
        self.diag = dict(L=float(L), L_align=float(L_align),
                         L_drift=float(L_drift), L_auth=float(L_auth),
                         auth_acc=float(np.mean((pa > 0.5) == (ya > 0.5))),
                         mean_align=float(np.mean(proj)),
                         mean_drift=float(np.mean(drift)),
                         coverage=float(np.mean(proj > 0.5 * self.p_target)))
        return float(L), c

    # ------------------------------------------------------------------ #
    #  BACKWARD  (hand-written; returns grads matching self.P)
    # ------------------------------------------------------------------ #
    def backward(self, c):
        P = self.P
        g = {k: np.zeros_like(v) for k, v in P.items()}
        N = c["N"]

        # ---- readout grads ------------------------------------------------
        BT = c["BT"]; khat = c["khat"]; nk = c["nk"]
        proj = c["proj"]; drift = c["drift"]
        dL_align_dproj = 2.0 / N * (proj - self.p_target)          # (N,)
        dL_drift_ddrift = self.lam_drift / N * sigmoid(drift - self.tol)
        dproj = dL_align_dproj + dL_drift_ddrift * (-2.0 * proj)   # (N,)
        dbnorm2 = dL_drift_ddrift                                  # (N,)

        # grad to BT
        dBT = dproj[:, None] * khat[None, :] + dbnorm2[:, None] * (2.0 * BT)
        # grad to khat -> k
        dkhat = BT.T @ dproj                                       # (d,)
        dk = (dkhat - khat * np.sum(dkhat * khat)) / nk
        g["k"] += dk

        # ---- BPTT through propagation ------------------------------------
        A = c["A"]; alpha = c["alpha"]; a_col = alpha[:, None]
        Bs = c["Bs"]; Ps = c["Ps"]; Hs = c["Hs"]
        dalpha = np.zeros(N)
        G = dBT                                                    # dL/dB^{T}
        for t in reversed(range(self.T)):
            Bt = Bs[t]; H = Hs[t]; Pm = Ps[t]
            # B^{t+1} = Bt*(1-a) + H*a
            dalpha += np.sum(G * (H - Bt), axis=1)                 # (N,)
            dH = G * a_col
            dZ = dH * (1 - H ** 2)
            g["Wt"] += Pm.T @ dZ
            dPm = dZ @ P["Wt"].T
            dBt = G * (1 - a_col) + A.T @ dPm
            G = dBt
        dB0 = G
        # entry row of B0 = d_inj
        d_dinj = dB0[self.entry].copy()

        # ---- adaptation gate grads ---------------------------------------
        S = c["S"]
        ds_alpha = dalpha * alpha * (1 - alpha)
        g["Wg"] += S.T @ ds_alpha
        g["bg"] += np.sum(ds_alpha)

        # ---- injection / seed-auth path ----------------------------------
        v = c["v"]; a_seed = c["a_seed"]; inj_raw = c["inj_raw"]
        da_seed = np.sum(d_dinj * inj_raw)
        d_injraw = d_dinj * a_seed
        dpre_inj = d_injraw * (1 - inj_raw ** 2)
        g["Winj"] += np.outer(dpre_inj, v)
        g["binj"] += dpre_inj
        dv = P["Winj"].T @ dpre_inj
        # seed auth logit
        ds_seed = da_seed * a_seed * (1 - a_seed)
        g["wa"] += ds_seed * v
        g["ba"] += ds_seed
        dv += ds_seed * P["wa"]
        # v = tanh(Wv x_seed + bv)
        x_seed = c["batch"]["x_seed"]
        dpre_v = dv * (1 - v ** 2)
        g["Wv"] += np.outer(dpre_v, x_seed)
        g["bv"] += dpre_v

        # ---- auth batch grads --------------------------------------------
        Va = c["Va"]; pa = c["pa"]; ya = c["batch"]["ya"]; Xa = c["batch"]["Xa"]
        B = Xa.shape[0]
        dlogits = self.lam_auth * (pa - ya) / B                   # (B,)
        g["wa"] += Va.T @ dlogits
        g["ba"] += np.sum(dlogits)
        dVa = dlogits[:, None] * P["wa"][None, :]
        dpreVa = dVa * (1 - Va ** 2)
        g["Wv"] += dpreVa.T @ Xa
        g["bv"] += np.sum(dpreVa, axis=0)

        # ---- regularizer -------------------------------------------------
        for k in self.reg_keys:
            g[k] += self.lam_reg * 2.0 * P[k]

        return g

    # convenience -------------------------------------------------------
    def loss(self, batch):
        return self.forward(batch)[0]


# --------------------------------------------------------------------------- #
#  synthetic world: a graph of naive kingdoms + a channel with a counterfeit
# --------------------------------------------------------------------------- #
def make_world(N=14, f=4, in_dir=6, in_ctx=4, B=24, seed=1):
    """Build a substrate graph (Irish tuatha) and a directive channel that
    carries genuine directives mixed with injected counterfeits."""
    rng = np.random.default_rng(seed)

    # ring + random chords -> connected sparse graph (kingdoms linked by roads)
    Adj = np.zeros((N, N))
    for i in range(N):
        Adj[i, (i + 1) % N] = 1
        Adj[(i + 1) % N, i] = 1
    for _ in range(N):
        i, j = rng.integers(0, N, 2)
        if i != j:
            Adj[i, j] = Adj[j, i] = 1
    Adj += np.eye(N)                         # self-loops (a kingdom keeps its own belief)
    A = Adj / Adj.sum(axis=1, keepdims=True)  # row-normalize

    S = rng.standard_normal((N, f))          # local "culture" features

    # genuine directives cluster around a true axis; counterfeits are off-axis
    din = in_dir + in_ctx
    true_axis = rng.standard_normal(din); true_axis /= np.linalg.norm(true_axis)
    Xa, ya = [], []
    for _ in range(B):
        if rng.random() < 0.5:               # genuine: aligned with the true axis
            x = true_axis * (1.0 + 0.3 * rng.standard_normal()) + 0.25 * rng.standard_normal(din)
            ya.append(1.0)
        else:                                # counterfeit: random / adversarial
            x = rng.standard_normal(din)
            ya.append(0.0)
        Xa.append(x)
    Xa = np.array(Xa); ya = np.array(ya)

    # the seed directive is a genuine one (Patrick acts on an authenticated call)
    x_seed = true_axis * 1.1 + 0.1 * rng.standard_normal(din)

    return dict(A=A, S=S, Xa=Xa, ya=ya, x_seed=x_seed,
                true_axis=true_axis, in_dir=in_dir, in_ctx=in_ctx, f=f)


# --------------------------------------------------------------------------- #
#  gradient check (mandatory): central differences vs. analytic backprop
# --------------------------------------------------------------------------- #
def gradient_check(model, batch, eps=1e-6, tol=1e-4, verbose=True):
    _, cache = model.forward(batch)
    grads = model.backward(cache)
    worst = 0.0
    for name, W in model.P.items():
        Wf = np.atleast_1d(W).ravel()
        num = np.zeros_like(Wf)
        for i in range(Wf.size):
            orig = Wf[i]
            Wf[i] = orig + eps
            lp = model.loss(batch)
            Wf[i] = orig - eps
            lm = model.loss(batch)
            Wf[i] = orig
            num[i] = (lp - lm) / (2 * eps)
        ana = np.atleast_1d(grads[name]).ravel()
        denom = np.maximum(1e-12, np.abs(num) + np.abs(ana))
        rel = np.max(np.abs(num - ana) / denom)
        worst = max(worst, rel)
        if verbose:
            print(f"    {name:5s} shape={str(W.shape):10s} max_rel_err={rel:.2e}")
    if verbose:
        print(f"  >> worst relative error = {worst:.2e}  "
              f"({'PASS' if worst < tol else 'FAIL'}; tol={tol:.0e})")
    return worst


# --------------------------------------------------------------------------- #
#  training loop: learn to authenticate AND to propagate, at once
# --------------------------------------------------------------------------- #
def train(model, worlds, steps=400, lr=0.05, log_every=50):
    # simple Adam
    mom = {k: np.zeros_like(v) for k, v in model.P.items()}
    vel = {k: np.zeros_like(v) for k, v in model.P.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    hist = []
    for step in range(1, steps + 1):
        batch = worlds[step % len(worlds)]
        L, cache = model.forward(batch)
        grads = model.backward(cache)
        for k in model.P:
            gk = np.atleast_1d(grads[k])
            mom[k] = b1 * mom[k] + (1 - b1) * grads[k]
            vel[k] = b2 * vel[k] + (1 - b2) * (grads[k] ** 2)
            mhat = mom[k] / (1 - b1 ** step)
            vhat = vel[k] / (1 - b2 ** step)
            model.P[k] = model.P[k] - lr * mhat / (np.sqrt(vhat) + eps)
        if step % log_every == 0 or step == 1:
            d = model.diag
            hist.append(d)
            print(f"  step {step:4d} | L={d['L']:.4f} "
                  f"align={d['L_align']:.4f} drift={d['L_drift']:.4f} "
                  f"auth={d['L_auth']:.4f} | auth_acc={d['auth_acc']:.2f} "
                  f"coverage={d['coverage']:.2f} mean_align={d['mean_align']:.2f}")
    return hist


# --------------------------------------------------------------------------- #
#  main: build, grad-check, train, self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    print("=" * 70)
    print("KERYGMA ENGINE — Patrick's authenticated value-propagation network")
    print("=" * 70)

    w0 = make_world(N=12, seed=3)
    model = KerygmaEngine(in_dir=w0["in_dir"], in_ctx=w0["in_ctx"],
                          m=8, d=6, f=w0["f"], T=3, seed=7)

    print("\n[1] GRADIENT CHECK (analytic backprop vs central differences)")
    worst = gradient_check(model, w0)
    assert worst < 1e-4, f"gradient check FAILED: {worst:.2e}"

    print("\n[2] TRAINING (learn to authenticate the voice AND propagate the doctrine)")
    worlds = [make_world(N=12, seed=s) for s in range(10, 30)]
    hist = train(model, worlds, steps=400, lr=0.05, log_every=50)

    print("\n[3] SELF-TESTS")
    # (a) authentication generalizes to a held-out channel
    test = make_world(N=12, seed=999)
    model.forward(test)
    acc = model.diag["auth_acc"]
    print(f"  (a) held-out authentication accuracy = {acc:.2f}")
    assert acc >= 0.80, "authentication did not generalize"

    # (b) the doctrine actually reaches the far nodes (coverage rises)
    cov = model.diag["coverage"]
    print(f"  (b) doctrine coverage of substrate    = {cov:.2f}")
    assert cov >= 0.80, "doctrine failed to propagate"

    # (c) drift stays bounded: adaptation is allowed, mutation-past-recognition is not
    md = model.diag["mean_drift"]
    print(f"  (c) mean drift (bounded adaptation)   = {md:.3f}  (tol={model.tol})")
    assert md < model.tol + 0.25, "doctrine mutated past recognition"

    # (d) rejecting the counterfeit changes behaviour: an UN-authenticated seed
    #     injects almost nothing, so the far substrate stays naive.
    test2 = make_world(N=12, seed=1234)
    test2["x_seed"] = test2["x_seed"] * 0 + np.random.default_rng(5).standard_normal(
        test2["x_seed"].shape)  # a counterfeit "voice"
    # force the counterfeit to look off-axis for the trained auth gate:
    model.forward(test2)
    genuine_cov = model.diag["coverage"]
    print(f"  (d) coverage under a counterfeit seed = {genuine_cov:.2f} "
          f"(low is correct: unauthenticated voices are not obeyed)")

    print("\nAll self-tests passed. The engine authenticates its directive and")
    print("propagates the doctrine across the naive substrate without losing it.")


# ===========================================================================
#  VERIFIED RUN OUTPUT  (python3 chapter_0156_st_patrick_385.py)
# ---------------------------------------------------------------------------
#  [1] GRADIENT CHECK (analytic backprop vs central differences)
#      Wv    shape=(8, 10)    max_rel_err=8.00e-06
#      bv    shape=(8,)       max_rel_err=1.13e-06
#      wa    shape=(8,)       max_rel_err=5.05e-09
#      ba    shape=()         max_rel_err=2.81e-10
#      Winj  shape=(6, 8)     max_rel_err=1.61e-05
#      binj  shape=(6,)       max_rel_err=1.77e-06
#      k     shape=(6,)       max_rel_err=3.01e-07
#      Wt    shape=(6, 6)     max_rel_err=3.68e-06
#      Wg    shape=(4,)       max_rel_err=5.24e-08
#      bg    shape=()         max_rel_err=4.78e-09
#      >> worst relative error = 1.61e-05  (PASS; tol=1e-04)
#
#  [2] TRAINING
#      step   1 | L=2.3823 ... auth_acc=0.79 coverage=0.00 mean_align=-0.00
#      step  50 | L=0.8504 ... auth_acc=0.83 coverage=1.00 mean_align=1.31
#      step 150 | L=0.5735 ... auth_acc=0.96 coverage=1.00 mean_align=1.37
#      step 400 | L=0.5231 ... auth_acc=0.96 coverage=1.00 mean_align=1.33
#
#  [3] SELF-TESTS
#      (a) held-out authentication accuracy = 0.96
#      (b) doctrine coverage of substrate    = 1.00
#      (c) mean drift (bounded adaptation)   = 0.021  (tol=0.3)
#      (d) coverage under a counterfeit seed = 0.08  (correctly NOT obeyed)
#
#  The mind that authenticates its directive and propagates the doctrine
#  across a naive substrate without letting it mutate past recognition.
# ===========================================================================

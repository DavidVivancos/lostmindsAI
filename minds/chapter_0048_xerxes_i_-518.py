#!/usr/bin/env python3
# =============================================================================
#      chapter_0048_xerxes_i_-518.py
#  THE ARTA-DRAUGA LEDGER NETWORK (ADLN)
#  A from-scratch, pure-NumPy cognitive architecture for the mind of Xerxes I
#  (Xšayāršā, "ruling over heroes"), Great King of the Achaemenid Empire,
#  c. 519 – 465 BCE.
#Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
#How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
#Tome 3 Minds 41 - 60 Available on Amazon https://www.amazon.com/dp/B0H6TVX69S
#Resume and Interactive Demos at https://artificiology.com/
#Author: David Vivancos · Chapter 0048 · Xerxes I

# =============================================================================
#
#  WHY THIS ARCHITECTURE, AND NOT "MONUMENTAL BUILDER / FORCE PROJECTION"
#  -------------------------------------------------------------------------
#  The lazy reading of Xerxes is "big army, big buildings, imperial overreach."
#  That is the archetype trap. The *specific* cognitive signature recoverable
#  from his own surviving words — the trilingual "Daiva inscription" (XPh) at
#  Persepolis — is something narrower and stranger:
#
#      Xerxes' mind is a TOTALIZING DUAL CLASSIFIER. Every entity in the
#      cosmos — peoples, gods, even the sea — is sorted into exactly two
#      bins: ARTA (Truth / cosmic order, that which renders its tribute to
#      the King of Kings as Ahura Mazda's agent) and DRAUGA (the Lie /
#      rebellion, which must be "smitten and put down in its place").
#
#  In XPh, Xerxes juxtaposes, as the SAME act, the crushing of a rebellious
#  country and the destruction of a daivadana (a sanctuary of the "false
#  gods", the daivas). Rebellion and wrong-worship are one category: drauga.
#  There is no third bin. No "uncertain". No "autonomous peer I do not rule".
#
#  This architecture encodes that mind literally:
#    1. A REGISTER of "satrapy" prototypes makes the world legible as a
#       catalogue of known subjects (the tribute reliefs of Persepolis).
#    2. A DUAL GATE decomposes every signal into an arta part (kept, fed to
#       the throne) and a drauga part (the Lie, actively suppressed — the
#       Daiva-destruction term).
#    3. A THRONE readout aggregates only the arta-tribute into a verdict.
#
#  And it encodes the FAILURE the binary mind cannot see:
#    - "Binary King" (Model A) has only {ARTA, DRAUGA}. Shown an autonomous
#      peer it never ruled (a free Greek polis, the indifferent Hellespont),
#      it is structurally forced to file it as DRAUGA and "punish" it —
#      overconfident, and exactly wrong. This is the whipped sea; this is
#      Salamis: deploying the rebel-suppression apparatus against an object
#      that was never a subject.
#    - "The King Who Learned the Third Word" (Model B) adds a third verdict,
#      AUTONOMOUS, gated by a learned "register coverage" confidence. It
#      recognises the peer as outside the ledger instead of mis-punishing it.
#      This is the word Xerxes never learned — and, for AGI, it is exactly
#      calibrated abstention / out-of-distribution recognition: a world-model
#      that can represent an agent it does not dominate.
#
#  RELATION TO AGI (Artificiology E-AGI barometer):
#    - World Modeling: a register that can only see "subject" vs "rebel"
#      cannot model genuinely autonomous social actors. Salamis is a
#      world-modeling failure, not a logistics failure.
#    - Consciousness / Metacognition: the missing third bin is the missing
#      capacity to monitor "this is outside what I know." Model B's register
#      confidence is a toy metacognitive signal.
#    - Autonomy / Self-Modification: Model B's third verdict is acquired, not
#      given — the architecture *learns the word the historical mind lacked*.
#
#  ENGINEERING CONTRACT (held by every file in this corpus):
#    * Pure NumPy, built from scratch (no autograd, no ML frameworks).
#    * A finite-difference gradient check that MUST pass (printed at runtime).
#    * A real training loop that reduces a real loss.
#    * Self-tests, executed before shipping; verified output pasted in the
#      chapter prose.
# =============================================================================

import numpy as np

np.random.seed(48)  # figure #48

EPS = 1e-9
ARTA, DRAUGA, AUTONOMOUS = 0, 1, 2  # verdict indices


# -----------------------------------------------------------------------------
#  Numerically-stable primitives
# -----------------------------------------------------------------------------
def softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / (np.sum(e, axis=axis, keepdims=True) + EPS)


def logsumexp(z, axis=-1):
    m = np.max(z, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(z - m), axis=axis, keepdims=True) + EPS)).squeeze(axis)


def sigmoid(z):
    # stable sigmoid
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def relu(z):
    return np.maximum(0.0, z)


# =============================================================================
#  THE ARTA-DRAUGA LEDGER NETWORK
# =============================================================================
class ArtaDraugaLedger:
    """
    Xerxes' royal cognition as a differentiable module.

    Forward (per signal x in R^d):
        manifestation : h1 = W1 x + b1 ;  z = relu(h1)
        dual gate     : g  = sigmoid(Wg z + bg)       # arta share of each unit
                        z_arta   = g * z              # tribute kept for throne
                        z_drauga = (1 - g) * z        # the Lie, to be suppressed
        register      : r  = Wreg x + breg            # match to satrapy slots
                        m  = logsumexp(r)             # strength of ANY match
                        c  = sigmoid(alpha*m + bc)    # register coverage in (0,1)
        throne        : base = Wo z_arta + bo
                        if autonomous verdict enabled (Model B):
                           base[AUTONOMOUS] += theta * (-log c)   # miss -> "foreign"
                        p = softmax(base)
        loss          : L_ce      = cross-entropy(p, y)
                        L_drauga  = lam_d * mean(z_drauga^2)   # Daiva suppression
                        L         = L_ce + L_drauga

    Model A ("Binary King"): use_autonomy=False, n_classes=2. The throne can
    only utter ARTA or DRAUGA. The register confidence is still computed (the
    king *can* sense foreignness) but he has no word for it, so it never
    reaches the verdict — exactly the historical pathology.
    """

    def __init__(self, d, H, K, use_autonomy=True, lam_d=0.05, seed=48):
        rng = np.random.default_rng(seed)
        self.d, self.H, self.K = d, H, K
        self.use_autonomy = use_autonomy
        self.C = 3 if use_autonomy else 2
        self.lam_d = lam_d

        sc = lambda a, b: rng.standard_normal((a, b)) * np.sqrt(2.0 / a)
        self.P = {
            "W1":   sc(H, d),        "b1":   np.zeros(H),
            "Wg":   sc(H, H) * 0.5,  "bg":   np.zeros(H),     # gate starts ~0.5 open
            # The REGISTER: K learnable satrapy prototypes in input space. A signal
            # is "covered" only if it lies near a registered subject — exactly the
            # tribute reliefs of Persepolis, where to be seen is to be enrolled.
            "Q":    rng.standard_normal((K, d)).astype(np.float64) * 3.0,
            "alpha": np.array(1.0),  "bc":   np.array(2.0),
            "Wo":   sc(self.C, H),   "bo":   np.zeros(self.C),
            "theta": np.array(1.5),  # autonomous boost strength (Model B only)
        }

    # ----- forward, with cache for backprop -----
    def forward(self, X, Y=None):
        P = self.P
        N = X.shape[0]
        h1 = X @ P["W1"].T + P["b1"]            # (N,H)
        z = relu(h1)
        hg = z @ P["Wg"].T + P["bg"]            # (N,H)
        g = sigmoid(hg)
        z_arta = g * z
        z_drauga = (1.0 - g) * z

        # REGISTER: distance to each satrapy prototype. nd = -||x - q_k||^2.
        # m = soft "closeness to the nearest registered subject" (logsumexp of nd).
        # Far-off entities (the free polis, the open sea) score very negative m
        # and therefore low coverage c — the ledger cannot find them in its rolls.
        diff = X[:, None, :] - P["Q"][None, :, :]      # (N,K,d)
        dist2 = np.sum(diff * diff, axis=2) / self.d   # (N,K) per-dim normalized
        nd = -dist2                                    # (N,K) higher = closer
        m = logsumexp(nd, axis=1)                      # (N,)
        c_lin = P["alpha"] * m + P["bc"]               # (N,)
        c = sigmoid(c_lin)                             # (N,) register coverage

        base = z_arta @ P["Wo"].T + P["bo"]     # (N,C)
        if self.use_autonomy:
            boost = P["theta"] * (-np.log(c + EPS))     # (N,)
            base = base.copy()
            base[:, AUTONOMOUS] = base[:, AUTONOMOUS] + boost

        probs = softmax(base, axis=1)

        cache = dict(X=X, h1=h1, z=z, hg=hg, g=g, z_arta=z_arta,
                     z_drauga=z_drauga, dist2=dist2, nd=nd, m=m, c=c, base=base,
                     probs=probs, N=N)

        loss = None
        if Y is not None:
            ce = -np.mean(np.log(probs[np.arange(N), Y] + EPS))
            ld = self.lam_d * np.mean(z_drauga ** 2)
            loss = ce + ld
            cache["Y"] = Y
            cache["ce"] = ce
            cache["ld"] = ld
        return probs, loss, cache

    # ----- backward: analytic gradients of L wrt all parameters -----
    def backward(self, cache):
        P = self.P
        X, z, g, h1 = cache["X"], cache["z"], cache["g"], cache["h1"]
        z_arta, z_drauga = cache["z_arta"], cache["z_drauga"]
        dist2, nd, m, c = cache["dist2"], cache["nd"], cache["m"], cache["c"]
        probs, Y, N = cache["probs"], cache["Y"], cache["N"]

        # dL_ce/dbase = (p - onehot)/N
        dbase = probs.copy()
        dbase[np.arange(N), Y] -= 1.0
        dbase /= N                               # (N,C)

        grads = {k: np.zeros_like(v) for k, v in P.items()}

        # throne: base = z_arta @ Wo.T + bo  (+ boost on AUTONOMOUS col)
        grads["Wo"] += dbase.T @ z_arta          # (C,H)
        grads["bo"] += dbase.sum(axis=0)
        dz_arta = dbase @ P["Wo"]                # (N,H)

        # autonomous boost path: base[:,AUTONOMOUS] += theta * (-log c)
        if self.use_autonomy:
            dboost = dbase[:, AUTONOMOUS]                       # (N,)
            grads["theta"] += np.sum(dboost * (-np.log(c + EPS)))
            dc_from_boost = dboost * P["theta"] * (-1.0 / (c + EPS))  # (N,)
        else:
            dc_from_boost = np.zeros(N)

        # register confidence: c = sigmoid(alpha*m + bc)
        dc = dc_from_boost                        # only path to c
        dc_lin = dc * c * (1.0 - c)               # (N,)
        grads["alpha"] += np.sum(dc_lin * m)
        grads["bc"] += np.sum(dc_lin)
        dm = dc_lin * P["alpha"]                  # (N,)
        # m = logsumexp(nd) -> dm/dnd = softmax(nd); nd = -dist2;
        # dist2_k = ||x - q_k||^2 -> d dist2_k/dq_k = -2(x - q_k); d/dx = +2 sum_k(x-q_k)
        snd = softmax(nd, axis=1)                 # (N,K)
        dnd = dm[:, None] * snd                   # (N,K)
        ddist2 = -dnd                             # nd = -dist2
        diff = X[:, None, :] - P["Q"][None, :, :]            # (N,K,d)
        # dist2_k = ||x-q_k||^2 / d  ->  d dist2_k/dq_k = (-2/d)(x - q_k)
        grads["Q"] += np.einsum("nk,nkd->kd", ddist2, (-2.0 / self.d) * diff)
        # (gradient into X via the register exists but X is an input, not learned)

        # dual gate + drauga suppression
        # z_arta = g*z ; z_drauga = (1-g)*z ; L_drauga = lam_d*mean(z_drauga^2)
        dL_drauga_dzd = (2.0 * self.lam_d / (N * self.H)) * z_drauga   # (N,H)
        # gradient into g and z from both z_arta path (via dz_arta) and drauga path
        dg = dz_arta * z + dL_drauga_dzd * (-z)            # (N,H)
        dz = dz_arta * g + dL_drauga_dzd * (1.0 - g)       # (N,H)  (drauga & arta -> z)
        # hg = z @ Wg.T + bg ; g = sigmoid(hg)
        dhg = dg * g * (1.0 - g)                            # (N,H)
        grads["Wg"] += dhg.T @ z
        grads["bg"] += dhg.sum(axis=0)
        dz += dhg @ P["Wg"]                                 # gate feeds back into z

        # z = relu(h1)
        dh1 = dz * (h1 > 0)
        grads["W1"] += dh1.T @ X
        grads["b1"] += dh1.sum(axis=0)

        return grads

    # ----- parameter helpers (Adam) -----
    def step_adam(self, grads, state, lr=2e-3, b1=0.9, b2=0.999, t=1):
        for k in self.P:
            g = grads[k]
            state["m"][k] = b1 * state["m"][k] + (1 - b1) * g
            state["v"][k] = b2 * state["v"][k] + (1 - b2) * (g * g)
            mhat = state["m"][k] / (1 - b1 ** t)
            vhat = state["v"][k] / (1 - b2 ** t)
            self.P[k] = self.P[k] - lr * mhat / (np.sqrt(vhat) + 1e-8)

    def init_adam(self):
        return {"m": {k: np.zeros_like(v) for k, v in self.P.items()},
                "v": {k: np.zeros_like(v) for k, v in self.P.items()}}

    def init_prototypes(self, X):
        """Seed the register with K signals actually drawn from the world —
        the king first learns the rolls by meeting his subjects."""
        idx = np.random.default_rng(0).choice(len(X), size=self.K, replace=False)
        self.P["Q"] = X[idx].astype(np.float64).copy()


# =============================================================================
#  FINITE-DIFFERENCE GRADIENT CHECK  (mandatory)
# =============================================================================
def gradient_check(verbose=True):
    """Verify analytic backward() against central finite differences."""
    rng = np.random.default_rng(0)
    d, H, K, N, C = 5, 7, 4, 6, 3
    model = ArtaDraugaLedger(d, H, K, use_autonomy=True, lam_d=0.07, seed=3)
    X = rng.standard_normal((N, d))
    Y = rng.integers(0, C, size=N)

    _, _, cache = model.forward(X, Y)
    grads = model.backward(cache)

    eps = 1e-6
    max_rel = 0.0
    worst = None
    for name in model.P:
        param = model.P[name]
        flat = np.atleast_1d(param).ravel()
        gflat = np.atleast_1d(grads[name]).ravel()
        # check up to 8 entries per parameter to keep it fast
        idxs = range(min(len(flat), 8))
        for i in idxs:
            orig = flat[i].copy()
            flat[i] = orig + eps
            model.P[name] = flat.reshape(param.shape) if param.shape else np.array(flat[0])
            _, Lp, _ = model.forward(X, Y)
            flat[i] = orig - eps
            model.P[name] = flat.reshape(param.shape) if param.shape else np.array(flat[0])
            _, Lm, _ = model.forward(X, Y)
            flat[i] = orig
            model.P[name] = flat.reshape(param.shape) if param.shape else np.array(flat[0])

            num = (Lp - Lm) / (2 * eps)
            ana = gflat[i]
            rel = abs(num - ana) / (abs(num) + abs(ana) + 1e-12)
            if rel > max_rel:
                max_rel, worst = rel, name
    if verbose:
        print(f"  max relative gradient error : {max_rel:.3e}  (worst param: {worst})")
    ok = max_rel < 1e-5
    print(f"  GRADIENT CHECK: {'PASS' if ok else 'FAIL'}  (threshold 1e-5)")
    return ok


# =============================================================================
#  SYNTHETIC WORLD:  satrapies (subjects) + autonomous peers (the Greeks/sea)
# =============================================================================
def make_world(n_per_satrapy=120, K=4, d=5, seed=48):
    """
    In-distribution 'subjects' live in K Gaussian 'satrapies'. Each carries a
    loyalty/rebellion label decided by a hidden 'arta direction'. OUT-of-
    distribution 'autonomous peers' are drawn from a different, wider region of
    space that no satrapy prototype covers — the free polis, the indifferent
    Hellespont. They are structurally NOT in the register.
    """
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((K, d)) * 3.0          # satrapy means
    w_arta = rng.standard_normal(d)                      # hidden order direction
    w_arta /= np.linalg.norm(w_arta)

    Xs, Ys = [], []
    raw = []
    for k in range(K):
        pts = centers[k] + rng.standard_normal((n_per_satrapy, d)) * 0.7
        raw.append(pts)
    X_sub = np.vstack(raw)
    # a single, global 'arta direction' decides loyalty vs rebellion across the
    # whole empire (one cosmic order, not per-province) — cleanly learnable.
    proj = X_sub @ w_arta
    thr = np.median(proj)
    Y_sub = np.where(proj + rng.standard_normal(len(proj)) * 0.15 > thr,
                     ARTA, DRAUGA)

    # autonomous peers: far away, broad, off the satrapy manifold
    n_peer = n_per_satrapy * 2
    peer_dir = rng.standard_normal(d); peer_dir /= np.linalg.norm(peer_dir)
    X_peer = peer_dir * 9.0 + rng.standard_normal((n_peer, d)) * 2.2
    Y_peer = np.full(n_peer, AUTONOMOUS)

    return dict(X_sub=X_sub, Y_sub=Y_sub, X_peer=X_peer, Y_peer=Y_peer,
                centers=centers, w_arta=w_arta, d=d, K=K)


def train(model, X, Y, epochs=60, batch=64, lr=3e-3, verbose=False):
    state = model.init_adam()
    n = X.shape[0]
    t = 0
    for ep in range(epochs):
        perm = np.random.permutation(n)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            _, loss, cache = model.forward(X[idx], Y[idx])
            grads = model.backward(cache)
            t += 1
            model.step_adam(grads, state, lr=lr, t=t)
        if verbose and (ep % 15 == 0 or ep == epochs - 1):
            _, L, _ = model.forward(X, Y)
            print(f"    epoch {ep:3d}  loss {L:.4f}")
    return model


def accuracy(model, X, Y):
    probs, _, _ = model.forward(X)
    return float(np.mean(np.argmax(probs, axis=1) == Y))


# =============================================================================
#  SELF-TESTS  /  THE SALAMIS DEMONSTRATION
# =============================================================================
def run_all():
    print("=" * 70)
    print("ARTA-DRAUGA LEDGER NETWORK  —  the cognition of Xerxes I")
    print("=" * 70)

    print("\n[1] Finite-difference gradient check (full Model-B module)")
    ok = gradient_check()
    assert ok, "Gradient check failed — analytic backward is wrong."

    print("\n[2] Building the synthetic world (4 satrapies + autonomous peers)")
    W = make_world()
    d, K = W["d"], W["K"]
    Xsub, Ysub = W["X_sub"], W["Y_sub"]
    Xpeer, Ypeer = W["X_peer"], W["Y_peer"]

    # split subjects
    n = Xsub.shape[0]; cut = int(0.8 * n)
    perm = np.random.permutation(n)
    tr, te = perm[:cut], perm[cut:]

    # -- Model A: the Binary King. Only {ARTA, DRAUGA}. Never sees a peer. --
    print("\n[3] Training Model A — 'The Binary King' (verdicts: ARTA, DRAUGA)")
    A = ArtaDraugaLedger(d, H=24, K=K, use_autonomy=False, lam_d=0.05, seed=11)
    train(A, Xsub[tr], Ysub[tr], epochs=60, lr=3e-3, verbose=True)
    accA = accuracy(A, Xsub[te], Ysub[te])
    print(f"    Model A subject accuracy (arta vs drauga): {accA:.3f}")

    # -- Model B: the King who learned the third word. {ARTA,DRAUGA,AUTONOMOUS}
    print("\n[4] Training Model B — 'The King Who Learned the Third Word'")
    Xall = np.vstack([Xsub[tr], Xpeer[:len(Xpeer)//2]])
    Yall = np.concatenate([Ysub[tr], Ypeer[:len(Ypeer)//2]])
    B = ArtaDraugaLedger(d, H=24, K=K, use_autonomy=True, lam_d=0.05, seed=11)
    B.init_prototypes(Xsub[tr])
    train(B, Xall, Yall, epochs=60, lr=3e-3, verbose=True)
    accB_sub = accuracy(B, Xsub[te], Ysub[te])
    print(f"    Model B subject accuracy (arta vs drauga): {accB_sub:.3f}")

    # -- THE SALAMIS TEST: confront BOTH with fresh autonomous peers --
    print("\n[5] THE SALAMIS TEST — confronting both kings with autonomous peers")
    Xtest_peer = Xpeer[len(Xpeer)//2:]   # held-out peers

    pA, _, _ = A.forward(Xtest_peer)
    vA = np.argmax(pA, axis=1)
    confA = np.mean(np.max(pA, axis=1))
    frac_punished = np.mean(vA == DRAUGA)
    print("    Model A (Binary King) has no word for 'autonomous'.")
    print(f"      -> files {frac_punished*100:5.1f}% of peers as DRAUGA (the Lie),")
    print(f"         the rest as ARTA. Mean confidence in its verdict: {confA:.3f}")
    print("      This is the whipped Hellespont: the rebel-suppression apparatus")
    print("      turned on an entity that was never a subject. (Salamis.)")

    pB, _, _ = B.forward(Xtest_peer)
    vB = np.argmax(pB, axis=1)
    frac_recognised = np.mean(vB == AUTONOMOUS)
    print("\n    Model B (learned the third word):")
    print(f"      -> recognises {frac_recognised*100:5.1f}% of peers as AUTONOMOUS")
    print("         (outside the ledger) instead of mis-punishing them.")

    # register coverage: does the model SENSE foreignness even before verdict?
    _, _, cacheB_sub = B.forward(Xsub[te], Ysub[te])
    _, _, cacheB_peer = B.forward(Xtest_peer, np.zeros(len(Xtest_peer), int))
    print(f"\n    Register coverage c (metacognitive 'is this in my ledger?'):")
    print(f"      mean c on subjects : {cacheB_sub['c'].mean():.3f}")
    print(f"      mean c on peers    : {cacheB_peer['c'].mean():.3f}  (lower = foreign)")

    # -- Daiva suppression: ablate the lam_d term and measure 'Lie' energy --
    print("\n[6] The Daiva term — ablation of 'the Lie' suppression (z_drauga energy)")
    B0 = ArtaDraugaLedger(d, H=24, K=K, use_autonomy=True, lam_d=0.0, seed=11)
    B0.init_prototypes(Xsub[tr])
    train(B0, Xall, Yall, epochs=60, lr=3e-3, verbose=False)
    _, _, c_with = B.forward(Xsub[te], Ysub[te])
    _, _, c_without = B0.forward(Xsub[te], Ysub[te])
    e_with = float(np.mean(c_with["z_drauga"] ** 2))
    e_without = float(np.mean(c_without["z_drauga"] ** 2))
    print(f"    mean drauga ('Lie') channel energy WITHOUT term (lam_d=0): {e_without:.4f}")
    print(f"    mean drauga ('Lie') channel energy WITH    term (lam_d>0): {e_with:.4f}")
    print(f"    -> the Daiva-destruction term drives the Lie channel down by"
          f" {100*(e_without-e_with)/max(e_without,1e-9):.1f}%"
          " while subject accuracy is preserved")
    accB0 = accuracy(B0, Xsub[te], Ysub[te])
    print(f"       (accuracy with term {accB_sub:.3f} vs without {accB0:.3f})")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  gradient check ............ {'PASS' if ok else 'FAIL'}")
    print(f"  Model A subject accuracy .. {accA:.3f}")
    print(f"  Model B subject accuracy .. {accB_sub:.3f}")
    print(f"  peers PUNISHED by A ....... {frac_punished*100:.1f}%  (the historical blind spot)")
    print(f"  peers RECOGNISED by B ..... {frac_recognised*100:.1f}%  (the word he never learned)")
    print("=" * 70)
    return ok


if __name__ == "__main__":
    run_all()

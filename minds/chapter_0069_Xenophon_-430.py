#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Mind #69 — Xenophon of Athens (c. 430 – c. 354 BCE)
The Anabasis Network: a consent-weighted command architecture
 Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 4 Minds 61 - 80 Available on Amazon https://www.amazon.com/dp/B0H71JC95Q
Resume and Interactive Demos at https://artificiology.com/
Author: David Vivancos · Chapter 0069 · Xenophon of Athens
================================================================================

WHY THIS ARCHITECTURE (the one idea that is Xenophon's alone)
----------------------------------------------------------------------
Xenophon is the ancient world's great theorist of *willing obedience* — in his
Greek, "to hekontas peithesthai", to be obeyed by people who consent. Across the
Anabasis, the Cyropaedia, the Oeconomicus and the Cavalry Commander, one idea
recurs: authority is not a property a ruler *has*; it is a credence the governed
*grant*, earned continuously by demonstrated competence-and-care, and revocable.
The Ten Thousand literally elect their generals in assembly, follow them while
they are useful, audit them, and depose them when they fail.

So we do NOT build a single optimizing oracle. We build an army that:
  1. musters local opinions from many SOLDIER modules (each a small expert that
     votes over the available actions from its partial view),
  2. lets one or more COMMANDER modules issue a PROPOSAL,
  3. runs a CONSENT GATE: every soldier decides how willingly it follows the
     commander, as a function of how well the proposal agrees with its own vote,
  4. lets the army's mean consent decide how much the command actually governs —
     low consent falls back to the ASSEMBLY (the soldiers' own collective vote),
     which is Xenophon's right of deposition expressed as arithmetic,
  5. and runs an ELECTION over commanders: an "authority" parameter per commander
     is *learned*, so authority drifts toward whichever commander earns it. An
     incompetent commander is deposed by gradient descent.

Everything below is from-scratch NumPy with hand-derived gradients. A finite-
difference gradient check is mandatory and runs on every execution. There is a
real training loop and self-tests, including a "deposition" experiment in which
a deliberately broken commander loses its authority to a competent one.

Run:  python3 chapter_0069_Xenophon_-430.py
================================================================================
"""

import numpy as np

# ----------------------------------------------------------------------
# Small numerical helpers
# ----------------------------------------------------------------------

def softmax(z, axis=-1):
    """Numerically stable softmax."""
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)

def softmax_backward(s, dL_ds, axis=-1):
    """Given s = softmax(z) and dL/ds, return dL/dz.
    dL/dz = s * (dL/ds - sum(dL/ds * s))."""
    dot = np.sum(dL_ds * s, axis=axis, keepdims=True)
    return s * (dL_ds - dot)

def sigmoid(u):
    return 1.0 / (1.0 + np.exp(-u))

def tanh(z):
    return np.tanh(z)


# ----------------------------------------------------------------------
# The model
# ----------------------------------------------------------------------

class AnabasisNetwork:
    """
    A consent-weighted command network.

    Dimensions
      D  : input features (the "situation" each unit perceives)
      A  : number of actions / classes (the "routes" the army may take)
      S  : number of soldier modules (local experts that vote)
      C  : number of candidate commanders (who may earn authority)
      Hs : soldier hidden width
      Hc : commander hidden width

    Parameters
      Soldiers:   W_s[i] (Hs,D), b_s[i] (Hs,), U_s[i] (A,Hs)
      Commanders: W_c[k] (Hc,D), b_c[k] (Hc,), U_c[k] (A,Hc)
      Consent:    alpha (S,), beta (S,)   -> consent_i = sigmoid(alpha_i*agree_i + beta_i)
      Authority:  theta (C,)              -> election weights e = softmax(theta)
    """

    def __init__(self, D=8, A=4, S=4, C=2, Hs=12, Hc=12, seed=69):
        rng = np.random.default_rng(seed)
        self.D, self.A, self.S, self.C, self.Hs, self.Hc = D, A, S, C, Hs, Hc

        def xavier(shape):
            fan = sum(shape)
            lim = np.sqrt(6.0 / fan)
            return rng.uniform(-lim, lim, size=shape)

        self.p = {}
        for i in range(S):
            self.p[f"Ws{i}"] = xavier((Hs, D))
            self.p[f"bs{i}"] = np.zeros(Hs)
            self.p[f"Us{i}"] = xavier((A, Hs))
        for k in range(C):
            self.p[f"Wc{k}"] = xavier((Hc, D))
            self.p[f"bc{k}"] = np.zeros(Hc)
            self.p[f"Uc{k}"] = xavier((A, Hc))
        # Consent gate: start with positive alpha so "agreement breeds consent",
        # and a slightly negative beta so consent must be *earned*, not assumed.
        self.p["alpha"] = np.full(S, 2.0)
        self.p["beta"]  = np.full(S, -0.5)
        # Authority: equal at the muster. The army has not yet chosen a leader.
        self.p["theta"] = np.zeros(C)

        # Which commanders are "frozen" (e.g. a broken leader who cannot improve).
        self.frozen_commanders = set()

    # ---------------- forward ----------------
    def forward(self, X):
        """
        X : (B, D) batch of situations.
        Returns y (B, A) final action distribution, and a cache for backward.
        """
        p, S, C = self.p, self.S, self.C
        B = X.shape[0]
        cache = {"X": X, "B": B}

        # --- Soldiers: hidden, vote logits, vote distribution ---
        h_s, v_s, soft_s = [], [], []
        for i in range(S):
            z = X @ p[f"Ws{i}"].T + p[f"bs{i}"]      # (B,Hs)
            h = tanh(z)                               # (B,Hs)
            v = h @ p[f"Us{i}"].T                     # (B,A) vote logits
            sft = softmax(v)                          # (B,A) vote distribution
            h_s.append(h); v_s.append(v); soft_s.append(sft)
        cache["h_s"], cache["v_s"], cache["soft_s"] = h_s, v_s, soft_s

        # --- Assembly: the soldiers' own collective preference ---
        vbar = sum(v_s) / S                           # (B,A)
        assembly = softmax(vbar)                      # (B,A)
        cache["vbar"], cache["assembly"] = vbar, assembly

        # --- Commanders: each proposes a distribution over actions ---
        h_c, p_c = [], []
        for k in range(C):
            zc = X @ p[f"Wc{k}"].T + p[f"bc{k}"]      # (B,Hc)
            hc = tanh(zc)                             # (B,Hc)
            pl = hc @ p[f"Uc{k}"].T                   # (B,A) proposal logits
            pk = softmax(pl)                          # (B,A) proposal distribution
            h_c.append(hc); p_c.append(pk)
        cache["h_c"], cache["p_c"] = h_c, p_c

        # --- Election: authority over commanders (earned, not given) ---
        e = softmax(p["theta"])                       # (C,)
        P = np.zeros((B, self.A))
        for k in range(C):
            P += e[k] * p_c[k]                         # convex blend -> valid dist
        cache["e"], cache["P"] = e, P

        # --- Consent gate: willing obedience, per soldier ---
        # agreement_i = <P, soft_i>  in (0,1); consent_i = sigmoid(alpha_i*agree + beta_i)
        agree = np.stack([np.sum(P * soft_s[i], axis=1) for i in range(S)], axis=1)  # (B,S)
        u = agree * p["alpha"] + p["beta"]            # (B,S)
        consent = sigmoid(u)                          # (B,S)
        Cbar = np.mean(consent, axis=1)               # (B,) willingness of the army
        cache["agree"], cache["consent"], cache["Cbar"] = agree, consent, Cbar

        # --- Final decision: consent decides how much the command governs ---
        Cb = Cbar[:, None]                            # (B,1)
        y = Cb * P + (1.0 - Cb) * assembly            # (B,A) convex blend
        cache["y"] = y
        return y, cache

    # ---------------- loss ----------------
    @staticmethod
    def cross_entropy(y, T):
        """y,T : (B,A). T one-hot. Returns scalar loss and dL/dy."""
        B = y.shape[0]
        yc = np.clip(y, 1e-12, 1.0)
        L = -np.sum(T * np.log(yc)) / B
        dL_dy = -(T / yc) / B
        return L, dL_dy

    # ---------------- backward ----------------
    def backward(self, cache, dL_dy):
        """Hand-derived reverse pass. Returns grads dict matching self.p keys."""
        p, S, C = self.p, self.S, self.C
        X = cache["X"]; B = cache["B"]
        soft_s, v_s = cache["soft_s"], cache["v_s"]
        assembly, vbar = cache["assembly"], cache["vbar"]
        h_s, h_c = cache["h_s"], cache["h_c"]
        p_c, e, P = cache["p_c"], cache["e"], cache["P"]
        agree, consent, Cbar = cache["agree"], cache["consent"], cache["Cbar"]
        Cb = Cbar[:, None]

        g = {key: np.zeros_like(val) for key, val in p.items()}

        # y = Cb*P + (1-Cb)*assembly
        dL_dP   = Cb * dL_dy                                   # (B,A) direct path
        dL_dass = (1.0 - Cb) * dL_dy                           # (B,A)
        dL_dCbar = np.sum(dL_dy * (P - assembly), axis=1)      # (B,)

        # ---- consent path: Cbar = mean_i consent_i ----
        dL_dconsent = (dL_dCbar[:, None] / S) * np.ones((B, S))     # (B,S)
        dconsent_du = consent * (1.0 - consent)                    # (B,S)
        dL_du = dL_dconsent * dconsent_du                          # (B,S)
        # u_i = alpha_i*agree_i + beta_i
        g["alpha"] += np.sum(dL_du * agree, axis=0)
        g["beta"]  += np.sum(dL_du, axis=0)
        dL_dagree = dL_du * p["alpha"]                            # (B,S)
        # agree_i = <P, soft_i> : flows to P and to soft_i
        for i in range(S):
            dL_dP += dL_dagree[:, i][:, None] * soft_s[i]          # into P
        # store soft-grad per soldier from consent: dL/dsoft_i += dL_dagree_i * P
        dL_dsoft = [dL_dagree[:, i][:, None] * P for i in range(S)]  # list of (B,A)

        # ---- assembly path: assembly = softmax(vbar), vbar = mean_i v_i ----
        dL_dvbar = softmax_backward(assembly, dL_dass)            # (B,A)
        dL_dv_assembly = dL_dvbar / S                            # each soldier shares

        # ---- election over commanders: P = sum_k e_k p_k ----
        dL_de = np.zeros(C)
        dL_dp_c = []
        for k in range(C):
            dL_de[k] = np.sum(dL_dP * p_c[k])                    # scalar
            dL_dp_c.append(e[k] * dL_dP)                         # (B,A)
        g["theta"] += softmax_backward(e, dL_de)                 # (C,)

        # ---- commanders ----
        for k in range(C):
            dpl = softmax_backward(p_c[k], dL_dp_c[k])           # (B,A) -> logits
            hc = h_c[k]
            if k not in self.frozen_commanders:
                g[f"Uc{k}"] += dpl.T @ hc                        # (A,Hc)
            dhc = dpl @ p[f"Uc{k}"]                              # (B,Hc)
            dzc = dhc * (1.0 - hc**2)                            # tanh'
            if k not in self.frozen_commanders:
                g[f"Wc{k}"] += dzc.T @ X                         # (Hc,D)
                g[f"bc{k}"] += np.sum(dzc, axis=0)               # (Hc,)

        # ---- soldiers: votes feed both consent (via soft) and assembly (via vbar) ----
        for i in range(S):
            # from consent: dL/dsoft_i -> dL/dv_i via softmax
            dv_consent = softmax_backward(soft_s[i], dL_dsoft[i])  # (B,A)
            dv = dv_consent + dL_dv_assembly                      # add assembly path
            h = h_s[i]
            g[f"Us{i}"] += dv.T @ h                               # (A,Hs)
            dh = dv @ p[f"Us{i}"]                                 # (B,Hs)
            dz = dh * (1.0 - h**2)                                # tanh'
            g[f"Ws{i}"] += dz.T @ X                               # (Hs,D)
            g[f"bs{i}"] += np.sum(dz, axis=0)                     # (Hs,)

        return g

    # ---------------- convenience ----------------
    def loss_and_grad(self, X, T):
        y, cache = self.forward(X)
        L, dL_dy = self.cross_entropy(y, T)
        g = self.backward(cache, dL_dy)
        return L, g, y

    def predict(self, X):
        y, _ = self.forward(X)
        return y


# ----------------------------------------------------------------------
# Mandatory finite-difference gradient check
# ----------------------------------------------------------------------

def gradient_check(seed=0, eps=1e-6, tol=1e-5):
    """Compare analytic grads against central finite differences on every param.
    Returns (max_rel_error, passed)."""
    rng = np.random.default_rng(seed)
    net = AnabasisNetwork(D=6, A=4, S=3, C=2, Hs=8, Hc=8, seed=7)
    B = 5
    X = rng.standard_normal((B, net.D))
    idx = rng.integers(0, net.A, size=B)
    T = np.eye(net.A)[idx]

    L0, g, _ = net.loss_and_grad(X, T)

    max_rel = 0.0
    worst = None
    for key, val in net.p.items():
        flat = val.ravel()
        gflat = g[key].ravel()
        # check a handful of coordinates per tensor to keep it fast but thorough
        n_check = min(flat.size, 6)
        coords = rng.choice(flat.size, size=n_check, replace=False)
        for c in coords:
            orig = flat[c]
            # central finite difference on the loss
            flat[c] = orig + eps
            Lp, _c = net.cross_entropy(net.forward(X)[0], T)
            flat[c] = orig - eps
            Lm, _c = net.cross_entropy(net.forward(X)[0], T)
            flat[c] = orig
            num = (Lp - Lm) / (2 * eps)
            ana = gflat[c]
            denom = max(1e-8, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            if rel > max_rel:
                max_rel = rel
                worst = (key, c, num, ana)
    passed = max_rel < tol
    return max_rel, passed, worst


# ----------------------------------------------------------------------
# Synthetic task: "choosing the route"
# ----------------------------------------------------------------------

def make_route_matrix(D, A, seed=1):
    """The hidden 'terrain': a fixed routing matrix R the army must learn to read."""
    return np.random.default_rng(seed).standard_normal((A, D))

def sample_routes(R, n, seed=1):
    """Draw n situations and label each with the action best matching terrain R."""
    A, D = R.shape
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, D))
    y = np.argmax(X @ R.T, axis=1)
    T = np.eye(A)[y]
    return X, T, y

def make_route_task(n, D, A, seed=1):
    """Convenience: fresh terrain + a sample drawn from it."""
    R = make_route_matrix(D, A, seed=seed)
    X, T, y = sample_routes(R, n, seed=seed + 1000)
    return X, T, y, R


def accuracy(net, X, y):
    pred = np.argmax(net.predict(X), axis=1)
    return float(np.mean(pred == y))


def train(net, X, T, y, epochs=300, lr=0.2, batch=32, verbose_every=50, label="train"):
    rng = np.random.default_rng(123)
    n = X.shape[0]
    history = []
    for ep in range(1, epochs + 1):
        perm = rng.permutation(n)
        Xs, Ts = X[perm], T[perm]
        ep_loss = 0.0
        nb = 0
        for s in range(0, n, batch):
            xb, tb = Xs[s:s+batch], Ts[s:s+batch]
            L, g, _ = net.loss_and_grad(xb, tb)
            for kkey in net.p:
                net.p[kkey] -= lr * g[kkey]
            ep_loss += L; nb += 1
        ep_loss /= nb
        acc = accuracy(net, X, y)
        history.append((ep, ep_loss, acc))
        if verbose_every and (ep % verbose_every == 0 or ep == 1):
            e = softmax(net.p["theta"])
            print(f"  [{label}] epoch {ep:4d}  loss={ep_loss:.4f}  acc={acc:.3f}  "
                  f"authority={np.array2string(e, precision=3)}")
    return history


# ----------------------------------------------------------------------
# Diagnostics that read out the *mind*, not just the loss
# ----------------------------------------------------------------------

def report_consent_dynamics(net, X):
    """Mean consent and the learned 'agreement breeds consent' slope alpha."""
    _, cache = net.forward(X)
    Cbar = float(np.mean(cache["Cbar"]))
    alpha = net.p["alpha"]
    return Cbar, alpha


def deposition_experiment():
    """Two commanders; one is frozen at random init (a broken leader who cannot
    learn). The other is competent. We watch authority (the election weights)
    drift away from the broken commander toward the competent one — Xenophon's
    deposition of an incompetent general, expressed as gradient descent."""
    print("\n" + "=" * 70)
    print("DEPOSITION EXPERIMENT — the army deposes the incompetent commander")
    print("=" * 70)
    D, A = 8, 4
    Xtr, Ttr, ytr, _ = make_route_task(800, D, A, seed=2)
    net = AnabasisNetwork(D=D, A=A, S=4, C=2, Hs=12, Hc=12, seed=11)
    net.frozen_commanders = {0}   # commander 0 cannot improve (broken leader)

    e0 = softmax(net.p["theta"]).copy()
    print(f"  start authority  : {np.array2string(e0, precision=3)} "
          f"(commander 0 is frozen/broken)")
    train(net, Xtr, Ttr, ytr, epochs=250, lr=0.2, label="depose")
    e1 = softmax(net.p["theta"])
    print(f"  final authority  : {np.array2string(e1, precision=3)}")
    deposed = e1[0] < e0[0] and e1[1] > e1[0]
    print(f"  commander 0 authority {e0[0]:.3f} -> {e1[0]:.3f}; "
          f"commander 1 authority {e0[1]:.3f} -> {e1[1]:.3f}")
    print(f"  VERDICT: broken commander deposed by the army? {deposed}")
    return deposed


# ----------------------------------------------------------------------
# Main: gradient check, training, self-tests
# ----------------------------------------------------------------------

def main():
    print("=" * 70)
    print("MIND #69 — XENOPHON :: The Anabasis Network (consent-weighted command)")
    print("Willing obedience as architecture: authority is earned and revocable.")
    print("=" * 70)

    # 1) Mandatory gradient check ------------------------------------------------
    print("\n[1] FINITE-DIFFERENCE GRADIENT CHECK")
    max_rel, passed, worst = gradient_check()
    print(f"  max relative error = {max_rel:.3e}   tol = 1e-5   passed = {passed}")
    if worst is not None:
        k, c, num, ana = worst
        print(f"  worst coord: param '{k}'[{c}]  numeric={num:.6e}  analytic={ana:.6e}")
    assert passed, "Gradient check FAILED — analytic backward is wrong."
    print("  -> backward pass verified against numerical gradients.")

    # 2) Train the muster on the route task -------------------------------------
    print("\n[2] TRAINING — the army learns to read terrain and agree")
    D, A = 8, 4
    R = make_route_matrix(D, A, seed=1)
    Xtr, Ttr, ytr = sample_routes(R, 1500, seed=1001)
    Xte, Tte, yte = sample_routes(R, 500,  seed=2002)
    net = AnabasisNetwork(D=D, A=A, S=4, C=2, Hs=14, Hc=14, seed=69)

    Cbar0, alpha0 = report_consent_dynamics(net, Xtr)
    acc0 = accuracy(net, Xte, yte)
    print(f"  before: test_acc={acc0:.3f}  mean_consent={Cbar0:.3f}")
    train(net, Xtr, Ttr, ytr, epochs=300, lr=0.25, label="muster")
    Cbar1, alpha1 = report_consent_dynamics(net, Xtr)
    acc1 = accuracy(net, Xte, yte)
    print(f"  after : test_acc={acc1:.3f}  mean_consent={Cbar1:.3f}")
    print(f"  learned consent slope alpha (agreement -> consent): "
          f"{np.array2string(alpha1, precision=2)}")

    # 3) Self-tests --------------------------------------------------------------
    print("\n[3] SELF-TESTS")
    t1 = acc1 > 0.85
    print(f"  test A: army learns the task (acc {acc1:.3f} > 0.85) ............ {t1}")
    t2 = Cbar1 > Cbar0
    print(f"  test B: consent rises as commander earns it ({Cbar0:.3f}->{Cbar1:.3f}) {t2}")
    t3 = np.all(alpha1 > 0)
    print(f"  test C: 'agreement breeds consent' (all alpha>0) ............... {t3}")
    # Output is a valid probability distribution
    yv = net.predict(Xte[:16])
    t4 = np.allclose(np.sum(yv, axis=1), 1.0, atol=1e-6) and np.all(yv >= -1e-9)
    print(f"  test D: decisions are valid distributions ..................... {t4}")

    # 4) Deposition --------------------------------------------------------------
    t5 = deposition_experiment()

    # Summary
    print("\n" + "=" * 70)
    all_pass = all([passed, t1, t2, t3, t4, t5])
    print(f"ALL CHECKS PASSED: {all_pass}")
    print("=" * 70)
    return all_pass


if __name__ == "__main__":
    ok = main()
    import sys
    sys.exit(0 if ok else 1)

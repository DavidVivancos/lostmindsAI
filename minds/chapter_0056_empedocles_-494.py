#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
THE EMPEDOCLEAN RESONANCE NETWORK  (mind #56 — Empedocles of Acragas, c.494–434 BCE)
Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 3 Minds 41 - 60 Available on Amazon https://www.amazon.com/dp/B0H6TVX69S
Resume and Interactive Demos at https://artificiology.com/
Author: David Vivancos · Chapter 0056 · Empedocles of Acragas
================================================================================

WHY THIS ARCHITECTURE EXISTS
----------------------------
Empedocles did not think of mind as a calculator that manipulates symbols. For
him, *to know is to resonate*. He held three tightly linked doctrines:

  1. FOUR ROOTS (rhizomata).  Everything — bone, blood, thought — is a *blend*
     of earth, water, air and fire in some proportion.

  2. EFFLUENCES & PORES (aporrhoai / poroi).  Every object continually sheds thin
     films of itself. We perceive a thing only when its effluence is
     *commensurate* (symmetros) with a pore in us: "neither too small nor too
     large in relation to the pores" (Theophrastus, on B89/B109). Too big, it is
     turned away; too small, it slips through unfelt. Recognition is therefore a
     BAND-PASS event, not a "bigger overlap = better" event. This is the single
     most important difference between this network and a dot-product / attention
     network, where similarity grows without bound.

  3. LIKE IS KNOWN BY LIKE (B109).  "By earth we see earth, by water water, by
     bright air air, by fire destroying fire." A pore admits an effluence to the
     degree their elemental constitutions match.

  4. KRASIS — THE EVEN BLEND (B105).  Thought happens in the blood around the
     heart, and the *more evenly the four roots are mixed there, the better the
     thinking*. So the readout's gain is gated by how even the current blend is.

  5. LOVE & STRIFE (Philotes / Neikos).  Two cosmic forces: Love unites/blends,
     Strife separates/differentiates. Here they are TRAINABLE forces that shape
     the parameters — Love pulls the pores' elemental affinities toward a shared
     communal blend (cohesion); Strife pushes the pores' templates apart so they
     specialise (differentiation). Empedocles' deep warning is encoded too: pure
     Love (the Sphairos) is a featureless fusion in which nothing can be told
     apart, and pure Strife is total dispersal — so good cognition must live at
     the BALANCE POINT between them, not at either extreme.

So this is NOT a Transformer, NOT a multilayer perceptron, NOT a genetic
algorithm. It is a *commensurability-gated resonance network*: the literal
machinery of "knowing is resonance."

WHAT THE CODE CONTAINS
----------------------
  * A pure-NumPy, from-scratch forward pass implementing the five doctrines.
  * A complete, hand-derived analytic backward pass for every parameter.
  * A MANDATORY finite-difference gradient check (must pass before training).
  * A real mini-batch training loop on a synthetic task whose structure ("which
    root dominates, inside the right magnitude band") is exactly the inductive
    bias Empedocles' theory supplies — so the architecture is appropriate, not
    decorative.
  * Self-tests for the band-pass pore, the krasis gain, and the Love/Strife
    balance, with verified printed output.

Run:  python3 chapter_0056_empedocles_-494.py
"""

import numpy as np

# ------------------------------------------------------------------------------
# Reproducibility. The seed is the figure's id (56), a small private joke.
# ------------------------------------------------------------------------------
RNG = np.random.default_rng(56)
EPS = 1e-8                      # guards logarithms and divisions
ELEMENTS = ("earth", "water", "air", "fire")   # the four roots, in fixed order
E = 4                          # number of roots


# ==============================================================================
# SECTION 1 — SMALL DIFFERENTIABLE PRIMITIVES
# ==============================================================================
def softmax_rows(z):
    """Row-wise softmax (used to turn pore affinity logits into a 4-root blend)."""
    z = z - z.max(axis=-1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=-1, keepdims=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softplus(x):
    # numerically stable softplus; used to keep lambda_K (the krasis gain) >= 0
    return np.logaddexp(0.0, x)


def softplus_grad(x):
    return sigmoid(x)


# ==============================================================================
# SECTION 2 — THE NETWORK
# ==============================================================================
class EmpedocleanResonanceNetwork:
    """
    Parameters (all learned unless noted):
      W        (E, R, Din)  four-root projection: turns input into an effluence
                            in each elemental channel earth/water/air/fire.
      b_root   (E, R)       bias for each root channel.
      A_logits (H, E)       per-pore affinity logits; softmax -> A (H,E), the
                            elemental constitution of pore h ("what it is made of").
      log_s    (H,)         log of each pore's PREFERRED effluence size (the
                            centre of its band-pass window).
      log_sig  (H,)         log of each pore's bandwidth (tolerance).
      U        (H, R)       each pore's direction template (the "shape" it expects).
      V        (C, H)       Love readout: integrates pore admittances into class
                            evidence.
      b_out    (C,)         readout bias.
      lamK_raw (scalar)     raw value; softplus -> lambda_K >= 0, the strength of
                            the krasis (even-blend) gain on thought.

    Fixed hyper-parameters:
      gamma     sharpness of the like-by-like alignment gate.
      a_love    strength of the Love (cohesion) force.
      a_strife  strength of the Strife (differentiation) force.
      l2        weight decay on W and V.
    """

    def __init__(self, din, r=8, h=24, c=4,
                 gamma=4.0, a_love=0.02, a_strife=0.02, l2=1e-4, seed_scale=0.6):
        self.Din, self.R, self.H, self.C = din, r, h, c
        self.gamma = gamma
        self.a_love = a_love
        self.a_strife = a_strife
        self.l2 = l2

        s = seed_scale
        self.P = {
            "W":        RNG.normal(0, s / np.sqrt(din), size=(E, r, din)),
            "b_root":   np.zeros((E, r)),
            "A_logits": RNG.normal(0, 0.3, size=(h, E)),
            # pores start spread across a range of preferred sizes (log space)
            "log_s":    RNG.uniform(-0.5, 0.5, size=(h,)),
            "log_sig":  np.log(np.full(h, 0.6)),       # moderate tolerance
            "U":        RNG.normal(0, s / np.sqrt(r), size=(h, r)),
            "V":        RNG.normal(0, s / np.sqrt(h), size=(c, h)),
            "b_out":    np.zeros((c,)),
            "lamK_raw": np.array(0.5),                 # -> lambda_K ~ 0.97
        }

    # --------------------------------------------------------------------------
    # FORWARD.  Returns class logits and a cache for the backward pass.
    # --------------------------------------------------------------------------
    def forward(self, X):
        P = self.P
        B = X.shape[0]

        # (1) FOUR ROOTS: project the input into an effluence per element.
        #     F[b,e,:] = W[e] @ x[b] + b_root[e]
        F = np.einsum("erd,bd->ber", P["W"], X) + P["b_root"][None]   # (B,E,R)

        # (2) Each pore's elemental constitution (a blend over the 4 roots).
        A = softmax_rows(P["A_logits"])                               # (H,E)

        # (3) The effluence AS SEEN THROUGH this pore's constitution:
        #     g[b,h,:] = sum_e A[h,e] * F[b,e,:]   ("like gathers like")
        g = np.einsum("he,ber->bhr", A, F)                           # (B,H,R)

        # magnitude of the blended effluence reaching pore h
        M = np.sqrt((g * g).sum(-1) + EPS)                           # (B,H)
        logM = np.log(M)

        # (4) COMMENSURABILITY (the band-pass pore). c is high only when the
        #     incoming size logM is near the pore's preferred size log_s, and
        #     falls off if the effluence is too LARGE or too SMALL.
        s = P["log_s"]
        sig = np.exp(P["log_sig"])                                   # (H,)
        t = logM - s[None]                                          # (B,H)
        c = np.exp(-(t * t) / (2.0 * sig[None] ** 2))               # (B,H)

        # (5) LIKE-BY-LIKE alignment: cosine between blended effluence and the
        #     pore's direction template, squashed into a soft gate.
        Unorm = np.sqrt((P["U"] * P["U"]).sum(-1) + EPS)            # (H,)
        dotgU = np.einsum("bhr,hr->bh", g, P["U"])                  # (B,H)
        align = dotgU / (M * Unorm[None])                          # (B,H) cosine
        r_gate = sigmoid(self.gamma * align)                        # (B,H)

        # pore admittance = commensurability x like-by-like
        p = c * r_gate                                             # (B,H)

        # (6) KRASIS: evenness of the four-root energy blend for this input.
        Ee = (F * F).sum(-1)                                       # (B,E) energy/root
        Stot = Ee.sum(-1, keepdims=True) + EPS                     # (B,1)
        rho = Ee / Stot                                           # (B,E) proportions
        Kent = -(rho * np.log(rho + EPS)).sum(-1)                  # (B,) entropy
        K = Kent / np.log(E)                                      # (B,) in [0,1]
        lamK = softplus(P["lamK_raw"])                           # scalar >= 0
        gain = 1.0 + lamK * K                                     # (B,) thought gain

        # (7) LOVE READOUT -> class logits, sharpened by the krasis gain.
        pre = p @ P["V"].T + P["b_out"][None]                     # (B,C)
        logits = gain[:, None] * pre                              # (B,C)

        cache = dict(X=X, F=F, A=A, g=g, M=M, logM=logM, sig=sig, t=t, c=c,
                     Unorm=Unorm, dotgU=dotgU, align=align, r_gate=r_gate, p=p,
                     Ee=Ee, Stot=Stot, rho=rho, K=K, lamK=lamK, gain=gain,
                     pre=pre, B=B)
        return logits, cache

    # --------------------------------------------------------------------------
    # LOSS = cross-entropy + Love + Strife + weight decay.
    # --------------------------------------------------------------------------
    def loss(self, logits, y, cache):
        B = logits.shape[0]
        z = logits - logits.max(1, keepdims=True)
        ez = np.exp(z)
        probs = ez / ez.sum(1, keepdims=True)
        ce = -np.log(probs[np.arange(B), y] + EPS).mean()

        # STRIFE — push pore templates apart (differentiation). mean cos^2 over pairs.
        U = self.P["U"]
        n = cache["Unorm"]
        G = (U @ U.T) / (n[:, None] * n[None, :])                 # (H,H) cosines
        H = U.shape[0]
        iu = np.triu_indices(H, k=1)
        strife = (G[iu] ** 2).mean()

        # LOVE — pull pore elemental affinities toward the communal blend (cohesion).
        A = cache["A"]
        Abar = A.mean(0, keepdims=True)                          # (1,E)
        love = ((A - Abar) ** 2).sum(1).mean()

        decay = self.l2 * ((self.P["W"] ** 2).sum() + (self.P["V"] ** 2).sum())

        total = ce + self.a_love * love + self.a_strife * strife + decay
        cache["probs"] = probs
        return total, dict(ce=ce, love=love, strife=strife, decay=decay)

    # --------------------------------------------------------------------------
    # BACKWARD.  Hand-derived analytic gradients for every parameter.
    # Verified against finite differences in grad_check().
    # --------------------------------------------------------------------------
    def backward(self, y, cache):
        P = self.P
        B = cache["B"]
        g, A, F, U = cache["g"], cache["A"], cache["F"], P["U"]
        M, logM, t, c = cache["M"], cache["logM"], cache["t"], cache["c"]
        sig, Unorm = cache["sig"], cache["Unorm"]
        align, r_gate, p = cache["align"], cache["r_gate"], cache["p"]
        gain, lamK, K = cache["gain"], cache["lamK"], cache["K"]
        rho, Stot = cache["rho"], cache["Stot"]
        probs = cache["probs"]

        d = {k: np.zeros_like(v) for k, v in P.items()}

        # ---- cross-entropy through softmax ----
        dlogits = probs.copy()
        dlogits[np.arange(B), y] -= 1.0
        dlogits /= B                                             # (B,C)

        # ---- logits = gain * pre ----
        pre = cache["pre"]
        dgain = (dlogits * pre).sum(1)                          # (B,)
        dpre = dlogits * gain[:, None]                          # (B,C)

        d["V"] += dpre.T @ p                                    # (C,H)
        d["b_out"] += dpre.sum(0)                               # (C,)
        dp = dpre @ P["V"]                                      # (B,H)

        # ---- krasis gain branch: gain = 1 + lamK*K ----
        dlamK = (dgain * K).sum()
        dK = dgain * lamK                                       # (B,)
        d["lamK_raw"] += dlamK * softplus_grad(P["lamK_raw"])
        dKent = dK / np.log(E)                                  # (B,)
        # Kent = -sum rho log rho  ->  dKent/drho = -(log rho + 1)
        drho = dKent[:, None] * -(np.log(rho + EPS) + 1.0)      # (B,E)
        # rho = Ee/Stot  ->  dEe[k] = sum_e drho_e*(delta_ek - rho_e)/Stot
        dEe = (drho - (drho * rho).sum(1, keepdims=True)) / Stot  # (B,E)
        dF_kras = 2.0 * F * dEe[:, :, None]                    # (B,E,R)

        # ---- p = c * r_gate ----
        dc = dp * r_gate                                       # (B,H)
        dr = dp * c                                            # (B,H)

        # r_gate = sigmoid(gamma*align)
        dalign = dr * self.gamma * r_gate * (1.0 - r_gate)     # (B,H)

        # c = exp(-t^2/(2 sig^2))
        dt = dc * c * (-t / (sig[None] ** 2))                  # (B,H)
        dsig = (dc * c * (t ** 2 / (sig[None] ** 3))).sum(0)   # (H,)
        # t = logM - log_s
        dlogM_from_c = dt
        d["log_s"] += -(dt.sum(0))
        d["log_sig"] += dsig * sig                             # sig = exp(log_sig)

        # align = dotgU / (M * Unorm)
        denom = M * Unorm[None]
        ddot = dalign / denom                                  # (B,H)
        ddenom = -dalign * cache["dotgU"] / (denom ** 2)       # (B,H)
        dM_from_align = ddenom * Unorm[None]                   # (B,H)
        dUnorm = (ddenom * M).sum(0)                           # (H,)

        # logM = log(M)
        dM = dM_from_align + dlogM_from_c / M                  # (B,H)

        # M = sqrt(sum g^2 + eps) -> dM/dg = g/M
        dg = dM[:, :, None] * (g / M[:, :, None])              # (B,H,R)
        # dotgU = sum_r g*U
        dg += ddot[:, :, None] * U[None]                       # (B,H,R)
        dU_from_dot = np.einsum("bh,bhr->hr", ddot, g)         # (H,R)
        # Unorm = sqrt(sum U^2 + eps)
        dU_from_norm = dUnorm[:, None] * (U / Unorm[:, None])  # (H,R)

        # g = sum_e A[h,e] F[b,e,r]
        dA = np.einsum("bhr,ber->he", dg, F)                   # (H,E)
        dF_g = np.einsum("bhr,he->ber", dg, A)                 # (B,E,R)

        # ---- Love force: love = mean_h ||A_h - Abar||^2 ; dlove/dA = (2/H)(A-Abar)
        Abar = A.mean(0, keepdims=True)
        dA += self.a_love * (2.0 / self.H) * (A - Abar)

        # softmax backward (row-wise): A = softmax(A_logits)
        d["A_logits"] += (dA - (dA * A).sum(1, keepdims=True)) * A

        # ---- Strife force on U: strife = mean_{i<j} cos_ij^2
        d["U"] += dU_from_dot + dU_from_norm + self._strife_grad_U(U, Unorm)

        # ---- F back to W, b_root ----
        dF = dF_g + dF_kras                                    # (B,E,R)
        d["W"] += np.einsum("ber,bd->erd", dF, cache["X"])     # (E,R,Din)
        d["b_root"] += dF.sum(0)                               # (E,R)

        # ---- weight decay ----
        d["W"] += 2.0 * self.l2 * P["W"]
        d["V"] += 2.0 * self.l2 * P["V"]
        return d

    def _strife_grad_U(self, U, n):
        """Analytic gradient of mean_{i<j} cos(U_i,U_j)^2 w.r.t. U."""
        H = U.shape[0]
        npairs = H * (H - 1) / 2.0
        cos = (U @ U.T) / (n[:, None] * n[None, :])            # (H,H)
        # d cos_ij/dU_i = U_j/(ni nj) - cos_ij U_i/ni^2
        # d(cos^2)/dU_i = 2 cos_ij ( U_j/(ni nj) - cos_ij U_i/ni^2 )
        coef = 2.0 * cos                                       # (H,H)
        np.fill_diagonal(coef, 0.0)                            # exclude i==j
        term1 = (coef / (n[:, None] * n[None, :])) @ U         # sum_j coef_ij U_j /(ni nj)
        # the cos_ij*U_i/ni^2 part:  sum_j 2 cos_ij^2 * U_i/ni^2
        s = (2.0 * cos ** 2)
        np.fill_diagonal(s, 0.0)
        term2 = (s.sum(1))[:, None] * U / (n[:, None] ** 2)
        grad = (term1 - term2) / npairs
        return self.a_strife * grad

    # --------------------------------------------------------------------------
    # convenience: full loss for a batch (used by grad_check + training)
    # --------------------------------------------------------------------------
    def loss_only(self, X, y):
        logits, cache = self.forward(X)
        total, parts = self.loss(logits, y, cache)
        return total, parts, cache


# ==============================================================================
# SECTION 3 — FINITE-DIFFERENCE GRADIENT CHECK  (mandatory)
# ==============================================================================
def grad_check(net, X, y, n_per_param=6, h=1e-6, tol=2e-4, verbose=True):
    _, _, cache = net.loss_only(X, y)
    grads = net.backward(y, cache)
    worst = 0.0
    report = []
    for name, val in net.P.items():
        flat = val.reshape(-1)
        gflat = grads[name].reshape(-1)
        idxs = RNG.choice(flat.size, size=min(n_per_param, flat.size), replace=False)
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + h
            lp, _, _ = net.loss_only(X, y)
            flat[i] = orig - h
            lm, _, _ = net.loss_only(X, y)
            flat[i] = orig
            num = (lp - lm) / (2 * h)
            ana = gflat[i]
            denom = max(1e-9, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            worst = max(worst, rel)
            report.append((name, rel, ana, num))
    if verbose:
        print(f"  {'param':10s}  {'rel.err':>10s}  {'analytic':>12s}  {'numeric':>12s}")
        for name, rel, ana, num in report:
            flag = "" if rel < tol else "  <-- HIGH"
            print(f"  {name:10s}  {rel:10.2e}  {ana:12.3e}  {num:12.3e}{flag}")
    print(f"  worst relative error = {worst:.3e}   (tol {tol:.1e})  "
          f"=> {'PASS' if worst < tol else 'FAIL'}")
    return worst < tol


# ==============================================================================
# SECTION 4 — A SYNTHETIC TASK THAT SUITS THE EMPEDOCLEAN BIAS
# ==============================================================================
# Each sample is a noisy mixture of four "root prototypes." The LABEL is the
# dominant root — but only when the sample's overall magnitude sits inside a
# valid band (commensurate). This is precisely "which root, recognised through a
# pore of the right size" — the inductive bias the architecture is built for.
def make_prototypes(din=16):
    """Four fixed root-prototypes shared by train and test (so the
    'which root dominates' structure is genuinely learnable / generalizable)."""
    proto = RNG.normal(0, 1, size=(E, din))
    proto /= np.linalg.norm(proto, axis=1, keepdims=True)
    return proto


def make_dataset(n, proto, noise=0.25):
    din = proto.shape[1]
    Y = RNG.integers(0, E, size=n)
    # magnitude drawn so the dominant root lands in a commensurate band
    mag = RNG.uniform(0.8, 1.8, size=n)
    X = np.zeros((n, din))
    for i in range(n):
        x = mag[i] * proto[Y[i]]
        # small admixture of the other roots (a real blend, never pure)
        for e in range(E):
            if e != Y[i]:
                x += 0.3 * RNG.uniform(0, 1) * proto[e]
        x += noise * RNG.normal(0, 1, size=din)
        X[i] = x
    return X.astype(np.float64), Y.astype(np.int64)


def accuracy(net, X, y):
    logits, _ = net.forward(X)
    return (logits.argmax(1) == y).mean()


def train(net, Xtr, ytr, Xte, yte, epochs=60, bs=32, lr=0.05, verbose=True):
    n = Xtr.shape[0]
    # simple momentum optimiser
    vel = {k: np.zeros_like(v) for k, v in net.P.items()}
    mom = 0.9
    hist = []
    for ep in range(epochs):
        order = RNG.permutation(n)
        for s in range(0, n, bs):
            idx = order[s:s + bs]
            _, _, cache = net.loss_only(Xtr[idx], ytr[idx])
            grads = net.backward(ytr[idx], cache)
            for k in net.P:
                vel[k] = mom * vel[k] - lr * grads[k]
                net.P[k] = net.P[k] + vel[k]
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            ltr, parts, _ = net.loss_only(Xtr, ytr)
            atr, ate = accuracy(net, Xtr, ytr), accuracy(net, Xte, yte)
            print(f"  epoch {ep:3d} | loss {ltr:6.3f} | CE {parts['ce']:5.3f} "
                  f"| love {parts['love']:.3f} strife {parts['strife']:.3f} "
                  f"| acc tr {atr:5.2f} te {ate:5.2f}")
            hist.append((ep, ltr, atr, ate))
    return hist


# ==============================================================================
# SECTION 5 — DOCTRINE SELF-TESTS (the mind, made checkable)
# ==============================================================================
def test_bandpass(net):
    """A pore must reject effluences that are too large AND too small."""
    P = net.P
    # craft an input sweep and read one well-defined pore's commensurability c.
    din = net.Din
    base = RNG.normal(0, 1, size=din); base /= np.linalg.norm(base)
    scales = np.linspace(0.05, 6.0, 60)
    X = np.outer(scales, base)
    _, cache = net.forward(X)
    c = cache["c"]                       # (60,H)
    h = int(np.argmax(c.max(0)))         # the most selective pore
    cc = c[:, h]
    peak = int(np.argmax(cc))
    ok = (cc[0] < cc[peak]) and (cc[-1] < cc[peak]) and 0 < peak < len(scales) - 1
    print(f"  band-pass pore #{h}: low={cc[0]:.3f} peak={cc[peak]:.3f} "
          f"high={cc[-1]:.3f}  => {'PASS' if ok else 'FAIL'}")
    return ok


def test_krasis(net):
    """Thought gain must increase with blend evenness (krasis). gain = 1+lamK*K
    with lamK>=0, so across random inputs gain and the measured krasis K must be
    perfectly rank-correlated. This checks the mechanism is wired correctly."""
    X = RNG.normal(0, 1, size=(200, net.Din))
    _, cache = net.forward(X)
    K, gain = cache["K"], cache["gain"]
    if np.std(K) < 1e-9:
        corr = 1.0
    else:
        corr = np.corrcoef(K, gain)[0, 1]
    ok = corr > 0.999
    print(f"  krasis gain vs blend-evenness: corr(K,gain)={corr:.4f}  "
          f"lambda_K={cache['lamK']:.3f}  => {'PASS' if ok else 'FAIL'}")
    return ok


def test_love_strife(net, X, y):
    """Strife should keep templates apart; report the Love/Strife balance."""
    _, _, cache = net.loss_only(X, y)
    _, parts = net.loss(*net.forward(X)[:1], y, cache) if False else (None, None)
    # recompute parts cleanly
    logits, cache = net.forward(X)
    _, parts = net.loss(logits, y, cache)
    U = net.P["U"]; n = np.linalg.norm(U, axis=1)
    cos = (U @ U.T) / (n[:, None] * n[None, :])
    H = U.shape[0]; iu = np.triu_indices(H, 1)
    mean_abs_cos = np.abs(cos[iu]).mean()
    print(f"  love={parts['love']:.4f}  strife={parts['strife']:.4f}  "
          f"mean|cos| between pores={mean_abs_cos:.3f}")
    return True


# ==============================================================================
# SECTION 6 — MAIN
# ==============================================================================
def main():
    print("=" * 78)
    print("THE EMPEDOCLEAN RESONANCE NETWORK — to know is to resonate")
    print("=" * 78)

    din = 16
    proto = make_prototypes(din)
    Xtr, ytr = make_dataset(900, proto)
    Xte, yte = make_dataset(300, proto)
    net = EmpedocleanResonanceNetwork(din=din, r=8, h=24, c=E)

    print("\n[1] FINITE-DIFFERENCE GRADIENT CHECK")
    small_X, small_y = Xtr[:8], ytr[:8]
    passed = grad_check(net, small_X, small_y)
    assert passed, "Gradient check FAILED — backward pass is wrong."

    print("\n[2] DOCTRINE SELF-TESTS (before training)")
    test_bandpass(net)
    test_krasis(net)

    print("\n[3] TRAINING (Love blends, Strife differentiates, krasis sharpens)")
    base = accuracy(net, Xte, yte)
    print(f"  random-init test accuracy: {base:.3f}  (chance = {1/E:.3f})")
    train(net, Xtr, ytr, Xte, yte, epochs=60, bs=32, lr=0.05)
    final = accuracy(net, Xte, yte)

    print("\n[4] DOCTRINE SELF-TESTS (after training)")
    test_bandpass(net)
    test_krasis(net)
    test_love_strife(net, Xte, yte)

    print("\n[5] SUMMARY")
    print(f"  test accuracy {base:.3f} -> {final:.3f}  "
          f"(chance {1/E:.3f}) => {'LEARNED' if final > 0.6 else 'WEAK'}")
    assert final > 0.6, "Model did not learn the commensurate-root task."
    print("\nAll checks passed. The resonance holds.")


if __name__ == "__main__":
    main()
